/**
 * Build the chat model an `llm_config` selects: the route (where requests go and how they
 * authenticate) and the wire protocol, from the validated config and its pinned connection.
 *
 * Routes are data: one small builder per provider family returns a `RouteModel`; there is no
 * plugin loading and no SDK. The environment is read only through the injected `env` (default
 * `processEnv`), at build: what `connection.ts` reads for the snapshot and credentials;
 * `OPENAI_BASE_URL` for a host key on a pinned subscription connection; each Chat Completions
 * vendor's key variable (`getApiKeyEnv`) and `VLLM_BASE_URL`; `AZURE_OPENAI_AD_TOKEN`,
 * `AZURE_OPENAI_API_KEY` and `OPENAI_API_VERSION` for Azure; `ANTHROPIC_API_KEY` and
 * `ANTHROPIC_BASE_URL`; Google's key variables (`apiKeyEnvNames`); and `OLLAMA_BASE_URL`. An empty
 * value counts as unset.
 */

import { type Env, processEnv } from "../defaults.ts";
import {
  type Auth,
  BasicAuth,
  BearerAuth,
  ExtraHeadersAuth,
  HeaderKeyAuth,
  NoAuth,
  type Secret,
  SubscriptionAuth,
} from "../llm/auth.ts";
import type { ChatModel } from "../llm/events.ts";
import { RouteModel } from "../llm/executor.ts";
import type { ReplayDegradedReason } from "../llm/messages.ts";
import { AnthropicMessages } from "../llm/protocols/anthropic_messages.ts";
import { Gemini } from "../llm/protocols/gemini.ts";
import { OllamaChat } from "../llm/protocols/ollama_chat.ts";
import { OpenAIChat } from "../llm/protocols/openai_chat.ts";
import { OpenAIResponses } from "../llm/protocols/openai_responses.ts";
import { Route } from "../llm/route.ts";
import { type Scheduler, type Send, warn } from "../platform.ts";
import type { JsonObject } from "../typesys/json.ts";
import { fromHost } from "../typesys/values.ts";
import { apiKeyEnvNames, getApiKeyEnv, OPTIONAL_API_KEY_PROVIDERS } from "./api_key_env.ts";
import {
  ApiKeyCredentials,
  AuthError,
  type Credentials,
  OAuthCredentials,
} from "./auth/common.ts";
import { azureResourceEndpoint, LLMConfig, splitOllamaBaseUrl } from "./config.ts";
import {
  checkModelConfig,
  type Connection,
  connectionCredentials,
  type ConnectionSnapshot,
  keyIdentity,
  resolveModelConfig,
} from "./connection.ts";
import { openaiIdHeaders } from "./connection_settings.ts";
import {
  ANTHROPIC_ENDPOINT,
  ANTHROPIC_VERSION,
  GOOGLE_ENDPOINT,
  OLLAMA_ENDPOINT,
  OPENAI_API_ENDPOINT,
  prefersResponses,
  PROVIDER_BASE_URL,
} from "./endpoints.ts";
import { isKnownModel } from "./model_catalog.ts";

/** Providers with a dedicated route. */
export const DEDICATED_PROVIDERS: readonly string[] = Object.freeze([
  "ollama",
  "anthropic",
  "google",
  "azure",
]);

/**
 * Return the non-secret account id replay records carry for `identity` (an API key, a ChatGPT
 * account id or an Azure resource endpoint): the first 16 hex digits of the SHA-256 of its UTF-8
 * bytes.
 */
export function accountHash(identity: string): string {
  return keyIdentity(identity).slice(0, 16);
}

/**
 * Emit `message` through `warn` as a `RuntimeWarning` unless `seen` already holds it, then add it
 * to `seen`.
 */
export function warnOnce(seen: Set<string>, message: string): void {
  if (seen.has(message)) return;
  seen.add(message);
  warn(message, "RuntimeWarning");
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
  /**
   * What the auth sends: a key, or a function reading it per request (OpenAI only); `null` leaves
   * the key to the provider's environment default.
   */
  key: Secret | null;
  /** The key as read at build, for the replay account; outside OpenAI it equals `key`. */
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

/** The error for `provider` with no key, naming every environment variable that could hold it. */
function missingKey(provider: string, variables: readonly string[]): RangeError {
  return new RangeError(
    `API key for provider '${provider}' is not set. Please set the ${variables.join(" or ")} environment ` +
      `variable (e.g. add ${variables[0]}=your_key to your .env file).`,
  );
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
 * Build a Chat Completions vendor's route (xAI, DeepSeek, Qwen, GLM, MiniMax, OpenRouter, vLLM):
 * `chat/completions`, or `responses` for a name-routed model, with a bearer key.
 *
 * The key is the resolved one, else the provider's non-empty key variable, else `"EMPTY"` for
 * vLLM; the base URL the resolved one, else (vLLM) a non-empty `VLLM_BASE_URL` as written, else
 * `PROVIDER_BASE_URL`; each variable is read only when no earlier source matched. Throws
 * `RangeError` naming the key variable if no key is found.
 */
function vendorModel(provider: string, resolved: Resolved, wiring: Wiring): RouteModel {
  const { model, block } = resolved;
  // Every Chat Completions vendor has a key variable.
  const variable = getApiKeyEnv(provider)!;
  const key = resolved.keyAtBuild ?? (wiring.env(variable) || null);
  if (key === null && !OPTIONAL_API_KEY_PROVIDERS.has(provider)) {
    throw missingKey(provider, [variable]);
  }
  // `checkModelConfig` refused a block on a name-routed model.
  const responses = prefersResponses(model);
  const envBaseUrl =
    resolved.baseUrl === null && provider === "vllm" ? wiring.env("VLLM_BASE_URL") : undefined;
  const route = new Route({
    provider,
    baseUrl: resolved.baseUrl ?? (envBaseUrl || PROVIDER_BASE_URL[provider]!),
    path: responses ? "responses" : "chat/completions",
    // A keyless vLLM server ignores the value; sending "EMPTY" keeps today's wire request.
    auth: withHeaders(new BearerAuth(key ?? "EMPTY"), resolved.secretHeaders),
  });
  const { onReplayDegraded } = wiring;
  const protocol = responses
    ? new OpenAIResponses({
        model,
        provider,
        // The keyless placeholder names no account.
        account: key === null ? null : accountHash(key),
        onReplayDegraded,
      })
    : new OpenAIChat({ model, provider, block, onReplayDegraded });
  return new RouteModel({ route, protocol, send: wiring.send, scheduler: wiring.scheduler });
}

/**
 * Return the Azure auth, first match wins, and its replay account: the resolved key as `api-key`;
 * a non-empty `AZURE_OPENAI_AD_TOKEN` as a bearer token; a non-empty `AZURE_OPENAI_API_KEY` as
 * `api-key`. Each variable is read only when no earlier source matched.
 *
 * Throws `RangeError` if none is found.
 */
function azureAuth(key: string | null, endpoint: string, env: Env): [Auth, string] {
  if (key !== null) return [new HeaderKeyAuth("api-key", key), accountHash(key)];
  const adToken = env("AZURE_OPENAI_AD_TOKEN");
  // An AD token rotates about hourly while the reasoning stays bound to the resource, so the
  // replay account is the resource's.
  if (adToken) return [new BearerAuth(adToken), accountHash(endpoint)];
  const envKey = env("AZURE_OPENAI_API_KEY");
  if (envKey) return [new HeaderKeyAuth("api-key", envKey), accountHash(envKey)];
  throw new RangeError(
    "provider azure has no credential; set llm_config.api_key with llm_config.base_url, run " +
      "'yian auth azure', or set AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN",
  );
}

/**
 * Build the Azure OpenAI route: the deployment's `chat/completions` at the resource endpoint, or
 * `openai/responses` for a name-routed deployment, with `api-version` from `OPENAI_API_VERSION`.
 *
 * Auth comes from `azureAuth`. Throws `RangeError` if it finds no credential, then if
 * `OPENAI_API_VERSION` is unset or empty.
 */
function azureModel(resolved: Resolved, wiring: Wiring): RouteModel {
  const { model, block } = resolved;
  const { env } = wiring;
  // `checkModelConfig` requires an Azure config's base_url.
  const endpoint = azureResourceEndpoint(resolved.baseUrl!);
  const [auth, account] = azureAuth(resolved.keyAtBuild, endpoint, env);
  const version = env("OPENAI_API_VERSION");
  if (!version) {
    throw new RangeError(
      "provider azure needs OPENAI_API_VERSION, the Azure OpenAI API version (for example " +
        "2025-03-01-preview)",
    );
  }
  // `checkModelConfig` refused a block on a name-routed deployment.
  const responses = prefersResponses(model);
  const route = new Route({
    provider: "azure",
    baseUrl: endpoint,
    path: responses
      ? "openai/responses"
      : `openai/deployments/${encodeURIComponent(model)}/chat/completions`,
    query: { "api-version": version },
    auth: withHeaders(auth, resolved.secretHeaders),
  });
  const { onReplayDegraded } = wiring;
  const protocol = responses
    ? new OpenAIResponses({ model, provider: "azure", account, onReplayDegraded })
    : new OpenAIChat({ model, provider: "azure", block, onReplayDegraded });
  return new RouteModel({ route, protocol, send: wiring.send, scheduler: wiring.scheduler });
}

/**
 * Build the Anthropic route: `v1/messages` with the key as `x-api-key` and the API version as a
 * route header.
 *
 * The key is the resolved one, else a non-empty `ANTHROPIC_API_KEY`; the base URL the resolved
 * one, else a non-empty `ANTHROPIC_BASE_URL` as written, else `https://api.anthropic.com`. The
 * replay account is the key's (thinking replays only to the same account). Throws
 * `RangeError` naming `ANTHROPIC_API_KEY` if no key is found.
 */
function anthropicModel(resolved: Resolved, wiring: Wiring): RouteModel {
  const { env } = wiring;
  const key = resolved.keyAtBuild ?? (env("ANTHROPIC_API_KEY") || null);
  if (key === null) throw missingKey("anthropic", ["ANTHROPIC_API_KEY"]);
  const route = new Route({
    provider: "anthropic",
    baseUrl: resolved.baseUrl ?? (env("ANTHROPIC_BASE_URL") || ANTHROPIC_ENDPOINT),
    path: "v1/messages",
    auth: withHeaders(new HeaderKeyAuth("x-api-key", key), resolved.secretHeaders),
    headers: { "anthropic-version": ANTHROPIC_VERSION },
  });
  const protocol = new AnthropicMessages({
    model: resolved.model,
    provider: "anthropic",
    account: accountHash(key),
    block: resolved.block,
    onReplayDegraded: wiring.onReplayDegraded,
  });
  return new RouteModel({ route, protocol, send: wiring.send, scheduler: wiring.scheduler });
}

/**
 * Return the Gemini API path that streams `model`, and the id the protocol and its records use.
 *
 * One leading `models/` is dropped, as google-genai does; an id in `tunedModels/` keeps that
 * collection. The rest is percent-encoded, so it stays one path segment.
 */
function geminiPath(model: string): [path: string, id: string] {
  const id = model.startsWith("models/") ? model.slice("models/".length) : model;
  const tuned = id.startsWith("tunedModels/");
  const collection = tuned ? "tunedModels" : "models";
  const name = tuned ? id.slice("tunedModels/".length) : id;
  return [`v1beta/${collection}/${encodeURIComponent(name)}:streamGenerateContent`, id];
}

/**
 * Build the Gemini Developer API route: the model's `streamGenerateContent` with `alt=sse` and
 * the key as `x-goog-api-key`, never in the query.
 *
 * The key is the resolved one, else the first non-empty variable of `apiKeyEnvNames("google")`
 * (`GOOGLE_API_KEY`, then `GEMINI_API_KEY`); the base URL (the API root) the resolved one, else
 * `https://generativelanguage.googleapis.com`. Throws `RangeError` naming both variables if no
 * key is found.
 */
function googleModel(resolved: Resolved, wiring: Wiring): RouteModel {
  const { env } = wiring;
  let key = resolved.keyAtBuild;
  for (const name of apiKeyEnvNames("google")) key ??= env(name) || null;
  if (key === null) throw missingKey("google", apiKeyEnvNames("google"));
  const [path, id] = geminiPath(resolved.model);
  const route = new Route({
    provider: "google",
    baseUrl: resolved.baseUrl ?? GOOGLE_ENDPOINT,
    path,
    query: { alt: "sse" },
    auth: withHeaders(new HeaderKeyAuth("x-goog-api-key", key), resolved.secretHeaders),
  });
  const protocol = new Gemini({
    model: id,
    provider: "google",
    block: resolved.block,
    onReplayDegraded: wiring.onReplayDegraded,
  });
  return new RouteModel({ route, protocol, send: wiring.send, scheduler: wiring.scheduler });
}

/**
 * Build the native Ollama route: `api/chat` at a non-empty `OLLAMA_BASE_URL`, split by
 * `splitOllamaBaseUrl`, else at `http://localhost:11434`.
 *
 * Only the user name and password written in `OLLAMA_BASE_URL` are sent, as Basic auth; a config
 * key is accepted and never sent. Throws `RangeError` if `OLLAMA_BASE_URL` is malformed.
 */
function ollamaModel(resolved: Resolved, wiring: Wiring): RouteModel {
  const value = wiring.env("OLLAMA_BASE_URL");
  const [baseUrl, credentials] = value
    ? splitOllamaBaseUrl(value)
    : [OLLAMA_ENDPOINT, null];
  const route = new Route({
    provider: "ollama",
    baseUrl,
    path: "api/chat",
    auth: credentials === null ? new NoAuth() : new BasicAuth(...credentials),
  });
  const protocol = new OllamaChat({
    model: resolved.model,
    provider: "ollama",
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
  const access = async () =>
    credentialsOf(await read(connection, "openai"), OAuthCredentials, "openai");
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
 * anyway.")` (default: `warn`, once per text per process); the subscription never
 * warns. A Chat Completions vendor's or Azure deployment's model that `prefersResponses` goes to
 * the Responses API. Ollama sends only the credentials written in `OLLAMA_BASE_URL`, never a key.
 *
 * `send` and `scheduler` go to the model's transport (`scheduler` also to the credential store),
 * `signal` to every credential read, and `onReplayDegraded` to the protocol.
 *
 * Throws `RangeError` if `checkModelConfig` rejects the config (e.g. a block the API it reaches
 * refuses, or a Chat Completions block on a model routed to the Responses API), a vendor,
 * Anthropic or Google key is missing, `OLLAMA_BASE_URL` is malformed, or Azure has no credential
 * or no `OPENAI_API_VERSION`; `AuthError` if a pinned login changed or was removed, an OpenAI key is
 * missing or changed, or saved header names differ from the recorded ones; a `TypeCheckError` for
 * a record that holds a non-yian value.
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
  const values = config instanceof LLMConfig
    ? config
    : (fromHost(config, "llm_config") as Record<string, unknown>);
  const [validated, connection] = checkModelConfig(resolveModelConfig(values, { env, snapshot }));
  // `resolveModelConfig` sets both or throws.
  const provider = validated.provider!;
  const model = validated.model!;
  const wiring: Wiring = { env, send, scheduler, onReplayDegraded };
  const read: Read = (pinned, name) => connectionCredentials(pinned, name, { env, signal, scheduler });

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
  // After the credential and header reads, so a removed login or
  // mismatched header names raise without a warning, while a missing environment key (outside
  // OpenAI) still warns first.
  if (!isKnownModel(provider, model)) {
    onWarning(
      `Model '${model}' is not in the known model list for provider '${provider}'. ` +
        "Continuing anyway.",
    );
  }
  if (provider === "openai") return openaiModel(resolved, wiring);
  if (provider === "azure") return azureModel(resolved, wiring);
  if (provider === "anthropic") return anthropicModel(resolved, wiring);
  if (provider === "google") return googleModel(resolved, wiring);
  if (provider === "ollama") return ollamaModel(resolved, wiring);
  return vendorModel(provider, resolved, wiring);
}

/**
 * The run's environment and model builder, supplied to nodes that call models. Tests and the
 * conformance runner pass a builder that returns a `ScriptedModel`.
 */
export interface ModelAccess {
  /** The environment model defaults and credentials are read from. */
  readonly env: Env;
  /**
   * Build the chat model `config` (a record of host values) selects for node `nodeId`; `signal`
   * aborts any credential read the build makes, and the model calls `onReplayDegraded` with the
   * reason each time it leaves out replay state it can't send back.
   *
   * `consumer` names a model built outside a forward node, so a host can route its calls:
   * `"backward"` for AGENT backward (with the AGENT's `nodeId`), or `"memory:<key>"` for the
   * learner of the reflective or profile memory under `<key>` (with `nodeId` `"__memory__"`).
   * It is absent for a node's own forward calls.
   */
  createModel(
    config: Readonly<Record<string, unknown>>,
    options: {
      nodeId: string;
      signal: AbortSignal;
      onReplayDegraded?: (reason: ReplayDegradedReason) => void;
      consumer?: string;
    },
  ): ChatModel | Promise<ChatModel>;
}

/**
 * Return the model access that builds through `chatModelFromConfig` with `env` (default
 * `processEnv`), passing on `signal` and `onReplayDegraded`; it ignores `consumer`.
 */
export function defaultModelAccess(env: Env = processEnv): ModelAccess {
  return {
    env,
    createModel: (config, { signal, onReplayDegraded }) => chatModelFromConfig(config, { env, signal, onReplayDegraded }),
  };
}
