/**
 * Build the chat model an `llm_config` selects: the route (where requests go and how they
 * authenticate) and the wire protocol, from the validated config and its pinned connection. An
 * embedding model's `/embeddings` client resolves its credentials the same way.
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
import { type Embed, type Embedder, type EmbeddingModel, openaiEmbeddings } from "../llm/embeddings.ts";
import type { ChatModel, StructuredOutput } from "../llm/events.ts";
import { RouteModel, type WireProtocol } from "../llm/executor.ts";
import type { ReplayDegradedReason } from "../llm/messages.ts";
import { AnthropicMessages, PROTOCOL as ANTHROPIC_MESSAGES } from "../llm/protocols/anthropic_messages.ts";
import { Gemini, PROTOCOL as GEMINI } from "../llm/protocols/gemini.ts";
import { OllamaChat, PROTOCOL as OLLAMA_CHAT } from "../llm/protocols/ollama_chat.ts";
import { OpenAIChat, PROTOCOL as OPENAI_CHAT } from "../llm/protocols/openai_chat.ts";
import { OpenAIResponses, PROTOCOL as OPENAI_RESPONSES } from "../llm/protocols/openai_responses.ts";
import { Route } from "../llm/route.ts";
import type { StructuredTarget } from "../llm/structured.ts";
import { type Scheduler, type Send, warn } from "../platform.ts";
import { YianRangeError } from "../typesys/errors.ts";
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
  checkEmbeddingProvider,
  checkModelConfig,
  type Connection,
  connectionCredentials,
  type ConnectionSnapshot,
  keyIdentity,
  openaiEndpoint,
  resolveModelConfig,
} from "./connection.ts";
import { openaiIdHeaders } from "./connection_settings.ts";
import {
  ANTHROPIC_ENDPOINT,
  ANTHROPIC_VERSION,
  GOOGLE_ENDPOINT,
  OLLAMA_ENDPOINT,
  type ChatApi,
  PROVIDER_BASE_URL,
  sendsToResponses,
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

/** What a route builder gets: the provider and model one config selects, and their key, endpoint and headers. */
export interface Resolved {
  provider: string;
  model: string;
  block: JsonObject;
  /** The API `llm_config.api` names for Azure or a Chat Completions vendor, or `null` when unset. */
  api: ChatApi | null;
  /**
   * The pinned ChatGPT subscription connection when calls go to the subscription endpoint, else
   * `null`. With a subscription nothing else is resolved: the key, key at build and base URL are
   * `null` and the headers empty.
   */
  subscription: Connection | null;
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
  /** How the model's structured calls ask for their schema (`structured_output`). */
  structuredOutput: StructuredOutput;
}

/** What every route passes on: the transport, the clock, the env, the replay observer and `structured_output`. */
interface Wiring {
  env: Env;
  send: Send | undefined;
  scheduler: Scheduler | undefined;
  onReplayDegraded: ((reason: ReplayDegradedReason) => void) | undefined;
  structuredOutput: StructuredOutput;
}

/**
 * Return the model sending `protocol`'s bodies over `route`, with `wiring`'s transport, clock and
 * `structured_output`.
 */
function routeModel(route: Route, protocol: WireProtocol, wiring: Wiring): RouteModel {
  const { send, scheduler, structuredOutput } = wiring;
  return new RouteModel({ route, protocol, send, scheduler, structuredOutput });
}

/** Read the credentials `connection` pins for `provider`. */
type Read = (connection: Connection, provider: string) => Promise<Credentials>;

/**
 * The saved API keys one run has read, by store path, provider and credential id, so the run reads
 * each once. Subscription tokens are never kept: they expire and refresh.
 */
export type SavedKeys = Map<string, ApiKeyCredentials>;

/**
 * Return the `Read` that resolves pinned credentials through `env`, `signal` and `scheduler`; with
 * `savedKeys`, a saved API key it read before is returned again without a read.
 */
function reader(
  env: Env,
  signal: AbortSignal | undefined,
  scheduler: Scheduler | undefined,
  savedKeys: SavedKeys | undefined,
): Read {
  return async (connection, provider) => {
    const id =
      savedKeys !== undefined && connection.source === "store" && connection.method === "api_key"
        ? [connection.store_path, provider, connection.credential_id].join("\0")
        : null;
    const kept = id === null ? undefined : savedKeys!.get(id);
    if (kept !== undefined) return kept;
    const credentials = await connectionCredentials(connection, provider, { env, signal, scheduler });
    if (id !== null && credentials instanceof ApiKeyCredentials) savedKeys!.set(id, credentials);
    return credentials;
  };
}

/** Wrap `auth` so it also sends the secret connection headers, when there are any. */
function withHeaders(auth: Auth, headers: Record<string, string>): Auth {
  return Object.keys(headers).length > 0 ? new ExtraHeadersAuth(auth, headers) : auth;
}

/** The error for `provider` with no key, naming every environment variable that could hold it. */
function missingKey(provider: string, variables: readonly string[]): RangeError {
  return new YianRangeError(
    `API key for provider '${provider}' is not set. Please set the ${variables.join(" or ")} environment ` +
      `variable (e.g. add ${variables[0]}=your_key to your .env file).`,
  );
}

/**
 * Return the endpoint an OpenAI API-key call goes to: the resolved one, else `openaiEndpoint`'s.
 *
 * @throws {RangeError} `OPENAI_BASE_URL` is read and isn't an allowed endpoint.
 */
function openaiBaseUrl(resolved: Resolved, env: Env): string {
  // Only a host key on a pinned subscription connection has no endpoint.
  return resolved.baseUrl ?? openaiEndpoint(env);
}

/**
 * Return the key and base URL a Chat Completions vendor's calls use (xAI, DeepSeek, Qwen, GLM,
 * MiniMax, OpenRouter, vLLM).
 *
 * The key is the resolved one, else the provider's non-empty key variable, else `null` for vLLM;
 * the base URL the resolved one, else (vLLM) a non-empty `VLLM_BASE_URL` as written, else
 * `PROVIDER_BASE_URL`; each variable is read only when no earlier source matched. Throws
 * `RangeError` naming the key variable if no key is found.
 */
function vendorAccess(provider: string, resolved: Resolved, env: Env): { key: string | null; baseUrl: string } {
  // Every Chat Completions vendor has a key variable.
  const variable = getApiKeyEnv(provider)!;
  const key = resolved.keyAtBuild ?? (env(variable) || null);
  if (key === null && !OPTIONAL_API_KEY_PROVIDERS.has(provider)) {
    throw missingKey(provider, [variable]);
  }
  const envBaseUrl = resolved.baseUrl === null && provider === "vllm" ? env("VLLM_BASE_URL") : undefined;
  return { key, baseUrl: resolved.baseUrl ?? (envBaseUrl || PROVIDER_BASE_URL[provider]!) };
}

/** Build the OpenAI API-key route: Responses with a bearer key. */
function openaiModel(resolved: Resolved, wiring: Wiring): RouteModel {
  const route = new Route({
    provider: "openai",
    baseUrl: openaiBaseUrl(resolved, wiring.env),
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
  return routeModel(route, protocol, wiring);
}

/**
 * Build a Chat Completions vendor's route (xAI, DeepSeek, Qwen, GLM, MiniMax, OpenRouter, vLLM):
 * `chat/completions`, or `responses` for a call `sendsToResponses`, with a bearer key.
 *
 * The key and base URL come from `vendorAccess`, which throws `RangeError` naming the key
 * variable if no key is found; a keyless vLLM sends `"EMPTY"`.
 */
function vendorModel(provider: string, resolved: Resolved, wiring: Wiring): RouteModel {
  const { model, block } = resolved;
  const { key, baseUrl } = vendorAccess(provider, resolved, wiring.env);
  // `checkModelConfig` refused a block on a call to the Responses API.
  const responses = sendsToResponses(resolved.api, model);
  const route = new Route({
    provider,
    baseUrl,
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
  return routeModel(route, protocol, wiring);
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
  throw new YianRangeError(
    "provider azure has no credential; set llm_config.api_key with llm_config.base_url, run " +
      "'yian auth azure', or set AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN",
  );
}

/**
 * Build the Azure OpenAI route: the deployment's `chat/completions` at the resource endpoint, or
 * `openai/responses` for a call `sendsToResponses`, with `api-version` from `OPENAI_API_VERSION`.
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
    throw new YianRangeError(
      "provider azure needs OPENAI_API_VERSION, the Azure OpenAI API version (for example " +
        "2025-03-01-preview)",
    );
  }
  // `checkModelConfig` refused a block on a call to the Responses API.
  const responses = sendsToResponses(resolved.api, model);
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
  return routeModel(route, protocol, wiring);
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
  return routeModel(route, protocol, wiring);
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
  return routeModel(route, protocol, wiring);
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
  return routeModel(route, protocol, wiring);
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
  return routeModel(route, protocol, wiring);
}

/**
 * Resolve an `llm_config` (a record of host values, or an `LLMConfig`) into its provider and
 * model and the key, endpoint and headers its calls use, as `chatModelFromConfig` does before it
 * picks a route.
 *
 * Unset `provider`/`model` come from one saved and environment selection snapshot (`snapshot`,
 * else one read from `env`). A recorded connection keeps its pinned method and identity, so a
 * later login or removal fails here instead of changing accounts. A ChatGPT subscription
 * connection with no config key comes back as `subscription`, with nothing else resolved.
 *
 * Otherwise a config `api_key` authenticates in place of any saved login or environment key. An
 * OpenAI connection's key is read once here (for the replay account, so a missing or changed key
 * fails now) and `key` reads it again on every request; another provider's saved key is read once
 * here, and its environment key is left to the route builder. A config `base_url` replaces the
 * endpoint. A config `organization`, `project` and `headers` travel with the config key; a saved
 * login's key brings the settings saved with it instead (its header names must equal the recorded
 * ones); an environment key sends none. `signal` and `scheduler` go to every credential read.
 * With `savedKeys`, a saved API key read before is used again instead of read: a run reads each
 * once.
 *
 * Throws `RangeError` if `checkModelConfig` rejects the config; `AuthError` if a pinned login
 * changed or was removed, an OpenAI key is missing or changed, or saved header names differ from
 * the recorded ones; a `TypeCheckError` for a record that holds a non-yian value.
 */
export async function resolveProviderAccess(
  config: Readonly<Record<string, unknown>> | LLMConfig,
  options: {
    env?: Env;
    snapshot?: ConnectionSnapshot;
    scheduler?: Scheduler;
    signal?: AbortSignal;
    savedKeys?: SavedKeys;
  } = {},
): Promise<Resolved> {
  const { env = processEnv, snapshot, scheduler, signal, savedKeys } = options;
  const values = config instanceof LLMConfig
    ? config
    : (fromHost(config, "llm_config") as Record<string, unknown>);
  const [validated, connection] = checkModelConfig(resolveModelConfig(values, { env, snapshot }));
  // `resolveModelConfig` sets both or throws.
  const provider = validated.provider!;
  const model = validated.model!;
  const block = validated.blockFor(provider);
  const structuredOutput = validated.structuredOutput ?? "native";
  const { api } = validated;
  if (validated.apiKey === null && connection !== null && connection.method === "subscription") {
    const none = { key: null, keyAtBuild: null, baseUrl: null, idHeaders: {}, secretHeaders: {} };
    return { provider, model, block, api, subscription: connection, ...none, structuredOutput };
  }
  const read = reader(env, signal, scheduler, savedKeys);

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
  return {
    provider,
    model,
    block,
    api,
    subscription: null,
    key,
    keyAtBuild,
    baseUrl,
    idHeaders,
    secretHeaders,
    structuredOutput,
  };
}

/**
 * Resolve an `llm_config` (a record of host values, or an `LLMConfig`) into a ready chat model.
 *
 * The provider, model, key, endpoint and headers come from `resolveProviderAccess`. A ChatGPT
 * subscription connection with no config key goes to the subscription endpoint; its credentials
 * are read once at build and again on every request. A model outside the provider's known list
 * gets `onWarning("Model '<model>' is not in the known model list for provider '<p>'. Continuing
 * anyway.")` (default: `warn`, once per text per process); the subscription never warns. A
 * Chat Completions vendor's or Azure deployment's call goes to the Responses API when its `api` is
 * `responses`, or, unset, when the model `prefersResponses`. Ollama sends only the credentials
 * written in `OLLAMA_BASE_URL`, never a key.
 *
 * `send` and `scheduler` go to the model's transport (`scheduler` also to the credential store),
 * `signal` to every credential read, `savedKeys` to `resolveProviderAccess`, and
 * `onReplayDegraded` to the protocol.
 *
 * Throws what `resolveProviderAccess` throws; `RangeError` if a vendor, Anthropic or Google key is
 * missing, `OLLAMA_BASE_URL` is malformed, or Azure has no credential or no `OPENAI_API_VERSION`;
 * `AuthError` if the subscription login changed or was removed.
 */
export async function chatModelFromConfig(
  config: Readonly<Record<string, unknown>> | LLMConfig,
  options: {
    env?: Env;
    snapshot?: ConnectionSnapshot;
    send?: Send;
    scheduler?: Scheduler;
    signal?: AbortSignal;
    savedKeys?: SavedKeys;
    onReplayDegraded?: (reason: ReplayDegradedReason) => void;
    onWarning?: (message: string) => void;
  } = {},
): Promise<RouteModel> {
  const { env = processEnv, snapshot, send, scheduler, signal, savedKeys, onReplayDegraded } = options;
  const onWarning = options.onWarning ?? ((message: string) => warnOnce(WARNED, message));
  const resolved = await resolveProviderAccess(config, { env, snapshot, scheduler, signal, savedKeys });
  const { provider, model, subscription } = resolved;
  const wiring: Wiring = { env, send, scheduler, onReplayDegraded, structuredOutput: resolved.structuredOutput };
  if (subscription !== null) {
    return subscriptionModel(subscription, model, resolved.block, wiring, reader(env, signal, scheduler, undefined));
  }
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
 * Return what `chatModelFromConfig(config)` would build, as structured output reads it
 * (`structuredTarget`), without reading any credential: the provider and model, the wire protocol
 * the route speaks, the Anthropic provider block, whether the route is the ChatGPT subscription,
 * and its `structured_output`.
 *
 * A config with its provider, model and connection (a recorded one) reads nothing; otherwise the
 * gaps come from `snapshot`, else one read from `env`, as `resolveProviderAccess` fills them.
 *
 * Throws `RangeError` if `checkModelConfig` rejects the config, or no layer sets the provider or
 * its model; a `TypeCheckError` for a record that holds a non-yian value.
 */
export function chatModelTarget(
  config: Readonly<Record<string, unknown>> | LLMConfig,
  options: { env?: Env; snapshot?: ConnectionSnapshot } = {},
): StructuredTarget {
  const { env = processEnv, snapshot } = options;
  const values = config instanceof LLMConfig ? config : (fromHost(config, "llm_config") as Record<string, unknown>);
  const [validated, connection] = checkModelConfig(resolveModelConfig(values, { env, snapshot }));
  const provider = validated.provider!;
  const subscription = validated.apiKey === null && connection !== null && connection.method === "subscription";
  const protocol = routeProtocol(provider, validated.model!, validated.api, subscription);
  // The Gemini route names the model by its id, without a leading `models/`.
  const model = protocol === GEMINI ? geminiPath(validated.model!)[1] : validated.model!;
  const block = protocol === ANTHROPIC_MESSAGES ? validated.blockFor(provider) : null;
  const structuredOutput = validated.structuredOutput ?? "native";
  return { provider, model, protocol, block, subscription, structuredOutput };
}

/**
 * Return the wire protocol `chatModelFromConfig` builds for `provider` and `model`: Responses for
 * the subscription and the OpenAI API key, each native protocol for its provider, and Chat
 * Completions for Azure and the vendors unless the call `sendsToResponses`.
 */
function routeProtocol(provider: string, model: string, api: ChatApi | null, subscription: boolean): string {
  if (subscription || provider === "openai") return OPENAI_RESPONSES;
  if (provider === "anthropic") return ANTHROPIC_MESSAGES;
  if (provider === "google") return GEMINI;
  if (provider === "ollama") return OLLAMA_CHAT;
  return sendsToResponses(api, model) ? OPENAI_RESPONSES : OPENAI_CHAT;
}

/**
 * Build the embeddings client for `config`, an embedding model on an OpenAI-compatible provider,
 * with the key, endpoint and headers that provider's chat calls would use.
 *
 * The provider and model are both required, never filled from the chat defaults. Credentials
 * resolve through `resolveProviderAccess`, saved logins included: OpenAI's key is read again on
 * every request; a vendor's key and base URL fall back as its chat route's do (`vendorAccess`),
 * and a keyless vLLM sends no auth header. `send` and `scheduler` go to the client, `signal` to
 * every credential read the build makes.
 *
 * Throws `RangeError` for a provider outside `OPENAI_COMPATIBLE` or an empty model, before
 * anything is read; `AuthError` for an OpenAI ChatGPT subscription login, which can't embed; and
 * what `resolveProviderAccess` and `vendorAccess` throw.
 */
export async function embedderFromConfig(
  config: EmbeddingModel,
  options: {
    env?: Env;
    snapshot?: ConnectionSnapshot;
    send?: Send;
    scheduler?: Scheduler;
    signal?: AbortSignal;
  } = {},
): Promise<Embedder> {
  const { provider, model } = config;
  checkEmbeddingProvider(provider);
  if (typeof model !== "string" || model === "") throw new YianRangeError("an embedding model needs a model name");
  const { env = processEnv, send, scheduler } = options;
  const resolved = await resolveProviderAccess({ provider, model }, options);
  if (resolved.subscription !== null) {
    throw new AuthError("a ChatGPT subscription can't embed; sign in with an API key");
  }
  let auth: Auth;
  let baseUrl: string;
  if (provider === "openai") {
    baseUrl = openaiBaseUrl(resolved, env);
    // `resolveModelConfig` pins every OpenAI config to a connection, so a key is always resolved.
    auth = new BearerAuth(resolved.key!);
  } else {
    const access = vendorAccess(provider, resolved, env);
    baseUrl = access.baseUrl;
    auth = access.key === null ? new NoAuth() : new BearerAuth(access.key);
  }
  return openaiEmbeddings({
    provider,
    baseUrl,
    auth: withHeaders(auth, resolved.secretHeaders),
    model,
    headers: resolved.idHeaders,
    send,
    scheduler,
  });
}

/**
 * Return the `Embed` that builds a client through `embedderFromConfig` with `env` (default
 * `processEnv`) on every call, from that call's provider and model, so each call reads the
 * current saved login; the call's `signal` goes to the build and the requests. `send` and
 * `scheduler` go to each client.
 */
export function defaultEmbed(env: Env = processEnv, options: { send?: Send; scheduler?: Scheduler } = {}): Embed {
  return async (texts, { provider, model, signal }) => {
    const embedder = await embedderFromConfig({ provider, model }, { ...options, env, signal });
    return embedder(texts, { signal });
  };
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
   */
  createModel(
    config: Readonly<Record<string, unknown>>,
    options: {
      nodeId: string;
      signal: AbortSignal;
      onReplayDegraded?: (reason: ReplayDegradedReason) => void;
    },
  ): ChatModel | Promise<ChatModel>;
  /**
   * Return what the model `createModel(config)` builds gives structured output to read
   * (`structuredTarget`), without building it or reading a credential; when absent, a model is
   * built to read it off. Throws as `createModel` does for a config it refuses.
   */
  structuredTarget?(config: Readonly<Record<string, unknown>>): StructuredTarget;
}

/**
 * Return the model access that builds through `chatModelFromConfig` with `env` (default
 * `processEnv`), passing on `signal` and `onReplayDegraded`, and reads structured targets through
 * `chatModelTarget`. Its models share one `SavedKeys`, so a run, which makes one model access,
 * reads each saved API key once: a login replaced or removed meanwhile is caught when the run
 * resumes, or by the next run.
 */
export function defaultModelAccess(env: Env = processEnv): ModelAccess {
  const savedKeys: SavedKeys = new Map();
  return {
    env,
    createModel: (config, { signal, onReplayDegraded }) =>
      chatModelFromConfig(config, { env, signal, savedKeys, onReplayDegraded }),
    structuredTarget: (config) => chatModelTarget(config, { env }),
  };
}
