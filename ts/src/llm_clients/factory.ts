/**
 * Build the chat model an `llm_config` selects: the route (where requests go and how they
 * authenticate) and the wire protocol, from the validated config and its pinned connection.
 *
 * Routes are data: one small builder per provider family returns a `RouteModel`; there is no
 * plugin loading and no SDK. The environment is read only through the injected `env` (default
 * `processEnv`): what `connection.ts` reads for the snapshot and credentials, and
 * `OPENAI_BASE_URL` for a host key on a pinned subscription connection.
 */

import { type Env, processEnv } from "../defaults.ts";
import {
  type Auth,
  BearerAuth,
  ExtraHeadersAuth,
  type Secret,
  SubscriptionAuth,
} from "../llm/auth.ts";
import { RouteModel } from "../llm/executor.ts";
import type { ReplayDegradedReason } from "../llm/messages.ts";
import { OpenAIResponses } from "../llm/protocols/openai_responses.ts";
import { Route } from "../llm/route.ts";
import type { Scheduler, Send } from "../platform.ts";
import type { JsonObject } from "../typesys/json.ts";
import { fromHost } from "../typesys/values.ts";
import {
  ApiKeyCredentials,
  AuthError,
  type Credentials,
  OAuthCredentials,
} from "./auth/common.ts";
import { LLMConfig, SUPPORTED_LLM_PROVIDERS } from "./config.ts";
import {
  checkModelConfig,
  type Connection,
  connectionCredentials,
  type ConnectionSnapshot,
  keyIdentity,
  OPENAI_API_ENDPOINT,
  resolveModelConfig,
} from "./connection.ts";
import { openaiIdHeaders } from "./connection_settings.ts";
import { validateModel } from "./validators.ts";

/**
 * Providers on an OpenAI-compatible API. Ollama is not one: it has its own native route, which
 * sends the `ollama` block to `/api/chat` as written.
 */
export const OPENAI_COMPATIBLE: readonly string[] = Object.freeze([
  "openai",
  "xai",
  "deepseek",
  "qwen",
  "qwen-cn",
  "glm",
  "glm-cn",
  "minimax",
  "minimax-cn",
  "openrouter",
  "vllm",
]);

/** Providers with a dedicated route. */
export const DEDICATED_PROVIDERS: readonly string[] = Object.freeze([
  "ollama",
  "anthropic",
  "google",
  "azure",
]);

/**
 * The default base URL of each OpenAI-compatible provider other than `openai`, whose endpoint the
 * connection records. Dual-region providers keep separate endpoints because international and
 * China accounts cannot share credentials.
 */
export const PROVIDER_BASE_URL: Readonly<Record<string, string>> = Object.freeze({
  xai: "https://api.x.ai/v1",
  deepseek: "https://api.deepseek.com",
  qwen: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
  "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  glm: "https://api.z.ai/api/paas/v4/",
  "glm-cn": "https://open.bigmodel.cn/api/paas/v4/",
  minimax: "https://api.minimax.io/v1",
  "minimax-cn": "https://api.minimaxi.com/v1",
  openrouter: "https://openrouter.ai/api/v1",
  vllm: "http://localhost:8000/v1",
});

const ROUTED_PROVIDERS = [...OPENAI_COMPATIBLE, ...DEDICATED_PROVIDERS];
{
  const routed = new Set(ROUTED_PROVIDERS);
  const supported = new Set<string>(SUPPORTED_LLM_PROVIDERS);
  if (
    routed.size !== ROUTED_PROVIDERS.length ||
    routed.size !== supported.size ||
    ![...supported].every((provider) => routed.has(provider))
  ) {
    throw new Error("LLM factory routes and LLMConfig provider vocabulary are out of sync");
  }
}

const RESPONSES_PREFIXES = ["gpt-5-pro", "gpt-5.2-pro", "gpt-5.4-pro", "gpt-5.5-pro", "gpt-5.6-sol"];

/**
 * Whether an OpenAI-compatible or Azure `model` is served through the Responses API rather than
 * Chat Completions: it starts with `gpt-5-pro`, `gpt-5.2-pro`, `gpt-5.4-pro`, `gpt-5.5-pro` or
 * `gpt-5.6-sol`, or contains `codex`.
 */
export function prefersResponses(model: string): boolean {
  return RESPONSES_PREFIXES.some((prefix) => model.startsWith(prefix)) || model.includes("codex");
}

/**
 * Return the non-secret account id replay records carry for `identity` (an API key, a ChatGPT
 * account id or an Azure resource endpoint): the first 16 hex digits of the SHA-256 of its UTF-8
 * bytes.
 */
export function accountHash(identity: string): string {
  return keyIdentity(identity).slice(0, 16);
}

/**
 * Emit `message` through `process.emitWarning` as a `RuntimeWarning` unless `seen` already holds
 * it, then add it to `seen`.
 */
export function warnOnce(seen: Set<string>, message: string): void {
  if (seen.has(message)) return;
  seen.add(message);
  process.emitWarning(message, "RuntimeWarning");
}

// The texts the default `onWarning` has emitted, so each shows once per process.
const WARNED = new Set<string>();

/**
 * Return a saved login's header values, checked against the names its connection recorded;
 * `{}` when `credentials` is `null` (not read because the connection records no headers).
 *
 * Throws `AuthError` naming `provider` if the saved names differ from the recorded ones; the
 * message never holds a value.
 */
function savedHeaders(
  credentials: ApiKeyCredentials | null,
  connection: Connection,
  provider: string,
): Record<string, string> {
  if (credentials === null) return {};
  const names = new Set(credentials.headers.map(([name]) => name));
  const recorded = new Set(connection.header_names ?? []);
  if (names.size !== recorded.size || ![...names].every((name) => recorded.has(name))) {
    throw new AuthError(
      `The saved ${provider} login's connection headers differ from the ones this run recorded; ` +
        `run 'yian auth ${provider}' and start a new run`,
    );
  }
  return Object.fromEntries(credentials.headers);
}

/** Return `credentials` as a `kind` login's; throw `AuthError` for another kind. */
function credentialsOf<T extends Credentials>(
  credentials: Credentials,
  kind: new (...args: never[]) => T,
  provider: string,
): T {
  // `connectionCredentials` already refuses a login whose method differs from the connection's;
  // this narrows the type.
  if (!(credentials instanceof kind)) {
    throw new AuthError(
      `The saved connection method changed; run 'yian auth ${provider}' and start a new run`,
    );
  }
  return credentials;
}

/** What a route builder gets: the resolved key, endpoint and headers of one config. */
interface Resolved {
  model: string;
  block: JsonObject;
  /** What the auth sends: a key, or a function reading it per request. */
  key: Secret | null;
  /** The key as read at build, for the replay account. */
  keyAtBuild: string | null;
  baseUrl: string | null;
  /** OpenAI-Organization and OpenAI-Project, sent as route headers. */
  idHeaders: Record<string, string>;
  /** Connection header values, sent on the auth so the scrub knows them. */
  secretHeaders: Record<string, string>;
}

/** What every route passes on: the transport, the clock, the env and the replay observer. */
interface Wiring {
  env: Env;
  send: Send | undefined;
  scheduler: Scheduler | undefined;
  onReplayDegraded: ((reason: ReplayDegradedReason) => void) | undefined;
}

/** Read the credentials `connection` pins for `provider`. */
type Read = (connection: Connection, provider: string) => Promise<Credentials>;

/** Wrap `auth` so it also sends the secret connection headers, when there are any. */
function withHeaders(auth: Auth, headers: Record<string, string>): Auth {
  return Object.keys(headers).length > 0 ? new ExtraHeadersAuth(auth, headers) : auth;
}

/** Build the OpenAI API-key route: Responses with a bearer key. */
function openaiModel(resolved: Resolved, wiring: Wiring): RouteModel {
  const route = new Route({
    provider: "openai",
    // Only a host key on a pinned subscription connection has no endpoint.
    baseUrl: resolved.baseUrl ?? (wiring.env("OPENAI_BASE_URL") || OPENAI_API_ENDPOINT),
    path: "responses",
    // `resolveModelConfig` pins every OpenAI config to a connection, so a key is always resolved.
    auth: withHeaders(new BearerAuth(resolved.key!), resolved.secretHeaders),
    headers: resolved.idHeaders,
  });
  const protocol = new OpenAIResponses({
    model: resolved.model,
    provider: "openai",
    account: resolved.keyAtBuild === null ? null : accountHash(resolved.keyAtBuild),
    block: resolved.block,
    onReplayDegraded: wiring.onReplayDegraded,
  });
  return new RouteModel({ route, protocol, send: wiring.send, scheduler: wiring.scheduler });
}

/**
 * Build the ChatGPT subscription route: Responses at the connection's endpoint, its credentials
 * read once here (for the replay account) and again on every request.
 */
async function subscriptionModel(
  connection: Connection,
  model: string,
  block: JsonObject,
  wiring: Wiring,
  read: Read,
): Promise<RouteModel> {
  const access = async () => credentialsOf(await read(connection, "openai"), OAuthCredentials, "openai");
  const first = await access();
  const route = new Route({
    provider: "openai",
    // `validateConnection` requires a subscription's endpoint.
    baseUrl: connection.endpoint!,
    path: "responses",
    auth: new SubscriptionAuth(access),
  });
  const protocol = new OpenAIResponses({
    model,
    provider: "openai",
    account: accountHash(first.accountId),
    block,
    subscription: true,
    onReplayDegraded: wiring.onReplayDegraded,
  });
  return new RouteModel({ route, protocol, send: wiring.send, scheduler: wiring.scheduler });
}

/**
 * Resolve an `llm_config` (a record of host values, or an `LLMConfig`) into a ready chat model.
 *
 * Unset `provider`/`model` come from one saved and environment selection snapshot (`snapshot`,
 * else one read from `env`). A recorded connection keeps its pinned method and identity, so a
 * later login or removal fails the build instead of changing accounts. A ChatGPT subscription
 * connection with no config key goes to the subscription endpoint; its credentials are read once
 * at build and again on every request.
 *
 * Otherwise a config `api_key` authenticates in place of any saved login or environment key. An
 * OpenAI connection's key is read once at build (for the replay account) and again on every
 * request, so a key changed after the build fails the first request; another provider's saved
 * key is read once at build. A config `base_url` replaces the endpoint. A config
 * `organization`, `project` and `headers` travel with the config key; a saved login's key brings
 * the settings saved with it instead (its header names must equal the recorded ones); an
 * environment key sends none. A model outside the provider's known list gets
 * `onWarning("Model '<model>' is not in the known model list for provider '<p>'. Continuing
 * anyway.")` (default: `process.emitWarning`, once per text per process); the subscription never
 * warns.
 *
 * `send` and `scheduler` go to the model's transport (`scheduler` also to the credential store),
 * `signal` to every credential read, and `onReplayDegraded` to the protocol.
 *
 * Throws `RangeError` if `checkModelConfig` rejects the config (e.g. a block the API it reaches
 * refuses) or the provider has no route yet; `AuthError` if a pinned login changed or was removed,
 * an OpenAI key is missing or changed, or saved header names differ from the recorded ones; a
 * `TypeCheckError` for a record that holds a non-yian value.
 */
export async function chatModelFromConfig(
  config: Readonly<Record<string, unknown>> | LLMConfig,
  options: {
    env?: Env;
    snapshot?: ConnectionSnapshot;
    send?: Send;
    scheduler?: Scheduler;
    signal?: AbortSignal;
    onReplayDegraded?: (reason: ReplayDegradedReason) => void;
    onWarning?: (message: string) => void;
  } = {},
): Promise<RouteModel> {
  const { env = processEnv, snapshot, send, scheduler, signal, onReplayDegraded } = options;
  const onWarning = options.onWarning ?? ((message: string) => warnOnce(WARNED, message));
  const values =
    config instanceof LLMConfig ? config : (fromHost(config, "llm_config") as Record<string, unknown>);
  const [validated, connection] = checkModelConfig(resolveModelConfig(values, { env, snapshot }));
  // `resolveModelConfig` sets both or throws.
  const provider = validated.provider!;
  const model = validated.model!;
  const wiring: Wiring = { env, send, scheduler, onReplayDegraded };
  const read: Read = async (pinned, name) =>
    connectionCredentials(pinned, name, { env, signal, scheduler });

  if (validated.apiKey === null && connection !== null && connection.method === "subscription") {
    return subscriptionModel(connection, model, validated.blockFor("openai"), wiring, read);
  }

  let key: Secret | null = null;
  let keyAtBuild: string | null = null;
  let baseUrl: string | null = null;
  // The saved login's credentials, when its header values are needed.
  let saved: ApiKeyCredentials | null = null;
  if (validated.apiKey !== null) {
    // A host-supplied key wins over the environment and any saved connection.
    key = keyAtBuild = validated.apiKey;
    if (provider === "openai" && connection !== null && connection.method === "api_key") {
      baseUrl = connection.endpoint;
    }
  } else if (provider === "openai" && connection !== null) {
    // Read per request, so a re-login cannot move a built model to another account; read once
    // here too, for the replay account, so a missing or changed key fails the build.
    const apiKey = async () =>
      credentialsOf(await read(connection, "openai"), ApiKeyCredentials, "openai");
    key = async () => (await apiKey()).apiKey;
    const first = await apiKey();
    keyAtBuild = first.apiKey;
    baseUrl = connection.endpoint;
    if (connection.header_names?.length) saved = first;
  } else if (connection !== null && connection.source === "store") {
    saved = credentialsOf(await read(connection, provider), ApiKeyCredentials, provider);
    key = keyAtBuild = saved.apiKey;
  }
  // Validation ties a base_url to the config's key, or makes it the Azure resource endpoint.
  if (validated.baseUrl !== null) baseUrl = validated.baseUrl;
  let idHeaders: Record<string, string>;
  let secretHeaders: Record<string, string>;
  if (validated.apiKey === null && connection !== null && connection.source === "store") {
    // A saved login's settings travel with its key, never with a host or environment key.
    idHeaders = openaiIdHeaders(connection.organization ?? null, connection.project ?? null);
    secretHeaders = savedHeaders(saved, connection, provider);
  } else {
    idHeaders = openaiIdHeaders(validated.organization, validated.project);
    secretHeaders = { ...validated.headers };
  }
  const resolved: Resolved = {
    model,
    block: validated.blockFor(provider),
    key,
    keyAtBuild,
    baseUrl,
    idHeaders,
    secretHeaders,
  };
  // After the credential and header reads, as Python warns in `get_llm()`: a removed login or
  // mismatched header names raise without a warning, while a missing key still warns first.
  if (!validateModel(provider, model)) {
    onWarning(
      `Model '${model}' is not in the known model list for provider '${provider}'. Continuing anyway.`,
    );
  }
  if (provider === "openai") return openaiModel(resolved, wiring);
  throw new RangeError(`chatModelFromConfig does not route provider ${provider} yet`);
}
