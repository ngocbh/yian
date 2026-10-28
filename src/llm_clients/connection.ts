/**
 * Record nonsecret connection identities and the selection they come from.
 *
 * There is no built-in provider or model: defaults come only from the environment and the saved
 * logins, read once into a `ConnectionSnapshot`. Auth storage owns credentials and refresh. Binding
 * records on every keyed agent which connection it uses (the saved login or the provider's key
 * environment variable; OpenAI also a host key, subscription, or missing key), and
 * `recordSelection` records the selection an agent whose config resolves only when it runs binds
 * from. `connectionCredentials` resolves the credentials a model call sends, `callCredentials`
 * lists them, and `scrubCallCredentials` scrubs them from the call's errors. Recorded metadata is
 * internal, never a YAML option, and never contains a token, API key or header value; its keys
 * are snake_case because it goes into run logs, which spell keys that way.
 *
 * The `YIAN_*` limits, such as `YIAN_MAX_BASE_URL_CHARS`, come through `defaults.ts`'s `current()`;
 * everything else is read only through the `env` parameter (default `processEnv`), one variable
 * at a time: `YIAN_DEFAULT_PROVIDER`, `YIAN_DEFAULT_MODEL`, `OPENAI_BASE_URL` and `OPENAI_API_KEY`
 * for the snapshot and OpenAI's credentials; for `callCredentials`, each provider's key variables
 * (`apiKeyEnvNames`, so `GEMINI_API_KEY` after Google's), `AZURE_OPENAI_AD_TOKEN` for Azure and
 * `OLLAMA_BASE_URL` for Ollama; and, through `AuthStore`, `YIAN_AUTH_FILE`, `XDG_CONFIG_HOME` and
 * `HOME` for the store path. An empty value counts as unset.
 */

import { createHash } from "node:crypto";
import { isAbsolute } from "node:path";

import { type Env, processEnv } from "../defaults.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { BasicAuth } from "../llm/auth.ts";
import { scrubError } from "../llm/errors.ts";
import type { Scheduler } from "../platform.ts";
import { isList, isRecord, valueRepr } from "../typesys/values.ts";
import { apiKeyEnvNames } from "./api_key_env.ts";
import {
  ApiKeyCredentials,
  AuthError,
  type Credentials,
  isUsableLogin,
  type OAuthCredentials,
  type ProviderSelection,
} from "./auth/common.ts";
import { OpenAIAuth } from "./auth/openai.ts";
import { AuthStore } from "./auth/store.ts";
import {
  AZURE_NEEDS_BASE_URL,
  azureResourceEndpoint,
  type HostConnection,
  LLMConfig,
  RETIRED_LLM_CONFIG_KEYS,
  splitOllamaBaseUrl,
  SUPPORTED_LLM_PROVIDERS,
  validateBaseUrl,
} from "./config.ts";
import { validateHeaderName, validateOrganization } from "./connection_settings.ts";
import { OPENAI_API_ENDPOINT, SUBSCRIPTION_ENDPOINT } from "./endpoints.ts";
import { blockSchema, checkBlock, type Issue } from "./provider_blocks.ts";
import { redactKeys } from "./redact.ts";

const FIELDS: ReadonlySet<string> = new Set(["method", "source", "credential_id", "endpoint", "store_path"]);
// Optional fields of a saved API-key login's connection: the connection settings saved with it,
// header names only. Records made before them carry none.
const SETTINGS_FIELDS: ReadonlySet<string> = new Set(["organization", "project", "header_names"]);
// Providers whose agents authenticate with a key and so carry a connection.
const KEYED: ReadonlySet<string> = new Set(SUPPORTED_LLM_PROVIDERS.filter((name) => name !== "ollama"));
const ENVIRONMENT_CONNECTION: Connection = {
  method: "api_key",
  source: "environment",
  credential_id: null,
  endpoint: null,
  store_path: null,
};
/**
 * Recorded on each agent that authenticates with a key the host supplies outside the record, so a
 * later model call without that key fails instead of falling back to the environment.
 */
export const HOST_KEY_MARKER = "_host_key";
/**
 * Recorded beside `HOST_KEY_MARKER`: the sorted names of the headers the host sent with its key.
 * Their values, like the key, are never recorded.
 */
export const HEADER_NAMES_KEY = "_header_names";
// Token variables listed whenever set, though only a call given no key sends them.
const ALWAYS_SENT_ENV: ReadonlyMap<string, readonly string[]> = new Map([["azure", ["AZURE_OPENAI_AD_TOKEN"]]]);
const SELECTION_FIELDS: ReadonlySet<string> = new Set([
  "defaults",
  "provider_models",
  "connections",
  "api_endpoint",
  "stray_model",
  "host",
]);
const HOST_RECORD_FIELDS: ReadonlySet<string> = new Set([
  "provider",
  "base_url",
  "organization",
  "project",
  "header_names",
]);

/**
 * The nonsecret identity of the connection an agent uses, as recorded on its config under
 * `_connection`.
 *
 * `organization`, `project` (OpenAI only) and the sorted `header_names` are the settings saved
 * with an API-key login, never a header value; older records carry none.
 */
export interface Connection {
  method: "api_key" | "subscription";
  source: "store" | "environment" | "missing" | "legacy_environment" | "host";
  credential_id: string | null;
  endpoint: string | null;
  store_path: string | null;
  organization?: string;
  project?: string;
  header_names?: string[];
}

/** One agent's `llm_config` as a mutable record of yian values. */
export type AgentConfig = Record<string, unknown>;

/** A selection's default provider and its model, either of which may be absent. */
export type Defaults = { provider?: string; model?: string };

/**
 * Say which agent has no model and the three ways to set one.
 *
 * `path` is the agent's static path, or `null` outside a flow; `provider` the selected provider,
 * if one is set. `strayModel` says `YIAN_DEFAULT_MODEL` is set without a default provider, and
 * `label`, the authored site of a synthesized agent's `llm_config`, is named instead of the path.
 */
export function missingModelMessage(
  path: string | null,
  provider: unknown,
  options: { strayModel?: boolean; label?: string | null } = {},
): string {
  const { strayModel = false, label = null } = options;
  let agent: string | null;
  if (label !== null) {
    // Uppercase the first code point; `slice(0, 1)` would split a surrogate pair.
    const [first = "", ...rest] = label;
    agent = first.toUpperCase() + rest.join("");
  } else if (path) {
    agent = `Agent ${reprText(path)}`;
  } else {
    agent = null;
  }
  let subject: string;
  let login: string;
  if (provider === null || provider === undefined) {
    subject = agent ? `${agent} has no provider or model` : "No model is set";
    login = "<provider>";
  } else {
    const quoted = typeof provider === "string" ? reprText(provider) : valueRepr(provider);
    subject = agent ? `${agent} has no model for provider ${quoted}` : `No model is set for provider ${quoted}`;
    login = typeof provider === "string" ? provider : valueRepr(provider);
  }
  let message = `${subject}; set llm_config.provider/model, pass --provider/--model, or run 'yian auth ${login}'`;
  if (strayModel) {
    message +=
      "; YIAN_DEFAULT_MODEL is set but names no provider; set YIAN_DEFAULT_PROVIDER or run 'yian auth <provider>'";
  }
  return message;
}

/**
 * One detached host selection shared across every static agent in a run. Instances are frozen and
 * hold copies of the values they are given.
 *
 * `defaults` holds the selected default provider and its model, either of which may be absent.
 * `connections` holds one connection per keyed provider: the saved login, else the provider's key
 * environment variable (for OpenAI, the environment key or a missing one). A saved Azure login's
 * connection carries its resource endpoint. `strayModel` records a `YIAN_DEFAULT_MODEL` that names
 * no provider, so an agent that needs a default can say why it was ignored.
 */
export class ConnectionSnapshot {
  readonly defaults: Readonly<Defaults>;
  readonly providerModels: Readonly<Record<string, string>>;
  readonly connections: Readonly<Record<string, Readonly<Connection>>>;
  readonly apiEndpoint: string;
  readonly strayModel: boolean;

  constructor(fields: {
    defaults: Readonly<Defaults>;
    providerModels: Readonly<Record<string, string>>;
    connections: Readonly<Record<string, Connection>>;
    apiEndpoint?: string;
    strayModel?: boolean;
  }) {
    this.defaults = Object.freeze(structuredClone(fields.defaults));
    this.providerModels = Object.freeze(structuredClone(fields.providerModels));
    const connections = structuredClone(fields.connections) as Record<string, Connection>;
    for (const connection of Object.values(connections)) Object.freeze(connection);
    this.connections = Object.freeze(connections);
    this.apiEndpoint = fields.apiEndpoint ?? OPENAI_API_ENDPOINT;
    this.strayModel = fields.strayModel ?? false;
    Object.freeze(this);
  }

  /**
   * Return `effective` with each agent's selected connection identity attached; no input map
   * changes.
   *
   * `effective` maps agent paths to configs of yian values with their `provider` set (converting
   * host values is the host entry points' job). Every keyed agent without a host key records its
   * provider's connection from the snapshot; an `ollama` agent records none. An `azure` agent with
   * a saved login and no `base_url` gets the login's endpoint as its `base_url`; one whose
   * `base_url` names another resource records the environment connection instead, since the saved
   * key belongs to the saved resource.
   *
   * Agents of `hostKeyProvider` authenticate with a key the host supplies outside the map. Each
   * records the nonsecret `HOST_KEY_MARKER`, and an OpenAI agent records a `host` connection
   * instead of the saved or environment one; other providers record no connection. `hostBaseUrl`,
   * the endpoint the host gave with that key, is recorded on those agents only, and is the OpenAI
   * host connection's endpoint. `hostSettings`, the connection settings the host gave with that
   * key, is recorded on the same agents: `organization` and `project` as given, and the header
   * names (never their values) sorted under `HEADER_NAMES_KEY`. Its key is ignored.
   */
  bind(
    effective: Readonly<Record<string, Readonly<AgentConfig>>>,
    options: { hostKeyProvider?: string | null; hostBaseUrl?: string | null; hostSettings?: HostConnection | null } = {},
  ): Record<string, AgentConfig> {
    const recorded = hostSettingsRecord(options.hostSettings ?? null);
    const result = structuredClone(effective) as Record<string, AgentConfig>;
    for (const config of Object.values(result)) {
      bindOne(this, config, options.hostKeyProvider ?? null, options.hostBaseUrl ?? null, recorded);
    }
    return result;
  }

  /**
   * Return the nonsecret identity of an OpenAI key supplied by the host: validated connection
   * metadata with source `host`, no credential identity, and `endpoint` (the host's `base_url`
   * for that key; `null` uses the snapshot's API endpoint).
   */
  hostConnection(endpoint: string | null = null): Connection {
    return validateConnection({
      method: "api_key",
      source: "host",
      credential_id: null,
      endpoint: endpoint ?? this.apiEndpoint,
      store_path: null,
    });
  }
}

/**
 * Mark one config for the host key, or attach its snapshot connection, in place.
 *
 * `config` is mutable with its `provider` set; `recorded` holds the host settings recorded beside
 * the marker: `organization`, `project` and `HEADER_NAMES_KEY`, each when set.
 */
function bindOne(
  snapshot: ConnectionSnapshot,
  config: AgentConfig,
  hostKeyProvider: string | null,
  hostBaseUrl: string | null,
  recorded: Readonly<Record<string, unknown>>,
): void {
  if (config["provider"] === hostKeyProvider) {
    config[HOST_KEY_MARKER] = true;
    if (hostBaseUrl !== null) config["base_url"] = hostBaseUrl;
    Object.assign(config, structuredClone(recorded));
    if (config["provider"] === "openai") config["_connection"] = snapshot.hostConnection(hostBaseUrl);
    return;
  }
  attach(snapshot, config);
}

/**
 * Record the provider's snapshot connection on one mutable config without a host key. An `azure`
 * config may get its `base_url` from the saved login.
 */
function attach(snapshot: ConnectionSnapshot, config: AgentConfig): void {
  const provider = config["provider"];
  if (typeof provider !== "string" || !Object.hasOwn(snapshot.connections, provider)) return;
  let connection = validateConnection(snapshot.connections[provider], provider);
  if (provider === "azure" && connection.source === "store") {
    const baseUrl = config["base_url"];
    if (baseUrl === undefined || baseUrl === null) {
      config["base_url"] = connection.endpoint;
    } else if (azureResourceEndpoint(baseUrl as string) !== azureResourceEndpoint(connection.endpoint as string)) {
      // The saved key belongs to its own resource; another one reads AZURE_OPENAI_API_KEY as before.
      connection = structuredClone(ENVIRONMENT_CONNECTION);
    }
  }
  config["_connection"] = connection;
}

/**
 * Return the nonsecret host settings recorded beside `HOST_KEY_MARKER`: `organization` and
 * `project` when set, and the sorted header names under `HEADER_NAMES_KEY` when there are headers.
 * The host's key and header values are ignored.
 */
function hostSettingsRecord(hostSettings: HostConnection | null): Record<string, unknown> {
  const recorded: Record<string, unknown> = {};
  if (hostSettings === null) return recorded;
  if (hostSettings.organization !== null) recorded["organization"] = hostSettings.organization;
  if (hostSettings.project !== null) recorded["project"] = hostSettings.project;
  const names = Object.keys(hostSettings.headers ?? {});
  if (names.length > 0) recorded[HEADER_NAMES_KEY] = names.sort(compareCodeUnits);
  return recorded;
}

/**
 * A recorded selection that binds the connection of a config resolved later. Instances are frozen.
 *
 * `snapshot` holds only the recorded providers' saved models and connections; `hostKeyProvider`
 * is the provider whose agents use the host key (`null` when the selection names none);
 * `hostBaseUrl` is where the host key is sent, if the host gave it; `hostSettings` holds the
 * nonsecret settings sent with the host key: `organization`, `project` and `HEADER_NAMES_KEY`,
 * each when set.
 */
export class ConnectionSelection {
  readonly snapshot: ConnectionSnapshot;
  readonly hostKeyProvider: string | null;
  readonly hostBaseUrl: string | null;
  readonly hostSettings: Readonly<Record<string, unknown>>;

  constructor(fields: {
    snapshot: ConnectionSnapshot;
    hostKeyProvider: string | null;
    hostBaseUrl: string | null;
    hostSettings: Readonly<Record<string, unknown>>;
  }) {
    this.snapshot = fields.snapshot;
    this.hostKeyProvider = fields.hostKeyProvider;
    this.hostBaseUrl = fields.hostBaseUrl;
    this.hostSettings = Object.freeze(structuredClone(fields.hostSettings));
    Object.freeze(this);
  }

  /**
   * Return `config` with its connection bound as `ConnectionSnapshot.bind` binds it.
   *
   * `config` is one agent's resolved config of yian values with its `provider` set (converting
   * host values is the host entry points' job); it is not mutated. The new config is marked for
   * the host key with its base_url and settings when its provider is the host key's, else it has
   * the recorded connection of its provider.
   */
  bind(config: Readonly<AgentConfig>): AgentConfig {
    const result = structuredClone(config) as AgentConfig;
    bindOne(this.snapshot, result, this.hostKeyProvider, this.hostBaseUrl, this.hostSettings);
    return result;
  }
}

/** The JSON record of a connection selection, as `recordSelection` writes it into a run log. */
export interface SelectionRecord {
  defaults: Defaults;
  provider_models: Record<string, string>;
  connections: Record<string, Connection>;
  api_endpoint: string;
  stray_model: boolean;
  host: Record<string, unknown> | null;
}

/**
 * Return the JSON record of a selection, limited to the given providers.
 *
 * Only the given providers' saved models and connections are recorded (`providers: null` records
 * every one); the defaults only when their provider is one of them; the OpenAI endpoint only when
 * OpenAI is one of them (else the built-in one); and the host key's provider, endpoint and
 * settings only when it is one of them. Of `hostSettings`, only the organization, project and
 * header names are recorded: the record never holds a key or header value.
 */
export function recordSelection(
  snapshot: ConnectionSnapshot,
  options: {
    providers: Iterable<string> | null;
    hostKeyProvider?: string | null;
    hostBaseUrl?: string | null;
    hostSettings?: HostConnection | null;
  },
): SelectionRecord {
  const allowed = options.providers === null ? null : new Set(options.providers);
  const reachable = (provider: unknown): boolean =>
    typeof provider === "string" && (allowed === null || allowed.has(provider));
  const hostKeyProvider = options.hostKeyProvider ?? null;
  let host: Record<string, unknown> | null = null;
  if (reachable(hostKeyProvider)) {
    const settings = hostSettingsRecord(options.hostSettings ?? null);
    host = { provider: hostKeyProvider };
    if (options.hostBaseUrl !== undefined && options.hostBaseUrl !== null) host["base_url"] = options.hostBaseUrl;
    for (const name of ["organization", "project"]) {
      if (Object.hasOwn(settings, name)) host[name] = settings[name];
    }
    if (Object.hasOwn(settings, HEADER_NAMES_KEY)) host["header_names"] = settings[HEADER_NAMES_KEY];
  }
  return {
    defaults: reachable(snapshot.defaults.provider) ? structuredClone(snapshot.defaults) : {},
    provider_models: Object.fromEntries(Object.entries(snapshot.providerModels).filter(([name]) => reachable(name))),
    connections: Object.fromEntries(
      Object.entries(snapshot.connections)
        .filter(([name]) => reachable(name))
        .map(([name, connection]) => [name, structuredClone(connection)]),
    ),
    // The endpoint configures only OpenAI; another run's records keep the built-in one.
    api_endpoint: reachable("openai") ? snapshot.apiEndpoint : OPENAI_API_ENDPOINT,
    stray_model: snapshot.strayModel,
    host,
  };
}

/**
 * Validate a `recordSelection` record and return the selection it holds, sharing no value with
 * `record`.
 *
 * Throws `RangeError` if the record does not have the recorded shape, names an unsupported
 * provider, or holds an invalid connection, endpoint or host setting. Errors never echo values.
 */
export function restoreSelection(record: unknown): ConnectionSelection {
  try {
    return restore(record);
  } catch (error) {
    if (error instanceof RangeError) {
      throw new RangeError(`Invalid recorded connection selection: ${error.message}`);
    }
    throw error;
  }
}

/** Validate and restore a selection record; `restoreSelection` prefixes errors. */
function restore(record: unknown): ConnectionSelection {
  if (!isRecord(record) || !hasExactKeys(record, SELECTION_FIELDS)) throw new RangeError("fields");
  const defaults = record["defaults"];
  if (
    !isRecord(defaults) ||
    !Object.keys(defaults).every((key) => key === "provider" || key === "model") ||
    (Object.hasOwn(defaults, "model") && !Object.hasOwn(defaults, "provider"))
  ) {
    throw new RangeError("defaults");
  }
  for (const value of Object.values(defaults)) nonempty(value, "defaults");
  if (Object.hasOwn(defaults, "provider")) supported(defaults["provider"], "defaults");
  const models = record["provider_models"];
  if (!isRecord(models)) throw new RangeError("provider_models");
  for (const [name, model] of Object.entries(models)) {
    supported(name, "provider_models");
    nonempty(model, "provider_models");
  }
  const connections = record["connections"];
  if (!isRecord(connections)) throw new RangeError("connections");
  for (const [name, connection] of Object.entries(connections)) {
    supported(name, "connections");
    validateConnection(connection, name);
  }
  if (typeof record["stray_model"] !== "boolean") throw new RangeError("stray_model");
  const apiEndpoint = nonempty(record["api_endpoint"], "api_endpoint");
  const snapshot = new ConnectionSnapshot({
    defaults: defaults as Defaults,
    providerModels: models as Record<string, string>,
    connections: connections as unknown as Record<string, Connection>,
    apiEndpoint,
    strayModel: record["stray_model"],
  });
  // The OpenAI host connection is the one use of the endpoint; check it the same way.
  snapshot.hostConnection();
  const host = record["host"];
  if (host === null) {
    return new ConnectionSelection({ snapshot, hostKeyProvider: null, hostBaseUrl: null, hostSettings: {} });
  }
  if (
    !isRecord(host) ||
    !Object.hasOwn(host, "provider") ||
    !Object.keys(host).every((key) => HOST_RECORD_FIELDS.has(key))
  ) {
    throw new RangeError("host fields");
  }
  const provider = supported(host["provider"], "host provider");
  const baseUrl = Object.hasOwn(host, "base_url") ? host["base_url"] : null;
  if (baseUrl !== null) {
    validateBaseUrl(baseUrl);
  } else if (provider === "azure") {
    // An Azure key names no resource of its own; the host always gives its endpoint.
    throw new RangeError("an azure host key needs its base_url");
  }
  const settings: Record<string, unknown> = {};
  for (const name of ["organization", "project"]) {
    if (Object.hasOwn(host, name)) {
      if (provider !== "openai") throw new RangeError(`host ${name} applies only to provider openai`);
      settings[name] = validateOrganization(host[name], name);
    }
  }
  if (Object.hasOwn(host, "header_names")) {
    settings[HEADER_NAMES_KEY] = validateHeaderNames(host["header_names"], provider);
  }
  return new ConnectionSelection({
    snapshot,
    hostKeyProvider: provider,
    hostBaseUrl: baseUrl as string | null,
    hostSettings: settings,
  });
}

/** Whether `value`'s own keys are exactly `keys`. */
function hasExactKeys(value: object, keys: ReadonlySet<string>): boolean {
  const own = Object.keys(value);
  return own.length === keys.size && own.every((key) => keys.has(key));
}

/** Return `value` if it is a non-empty string; otherwise throw `RangeError` naming `where`. */
function nonempty(value: unknown, where: string): string {
  if (typeof value !== "string" || !value) throw new RangeError(where);
  return value;
}

/** Whether `value` names a supported provider. */
function isSupported(value: unknown): value is string {
  return typeof value === "string" && (SUPPORTED_LLM_PROVIDERS as readonly string[]).includes(value);
}

/** Return `value` if it names a supported provider; otherwise throw `RangeError` naming `where`. */
function supported(value: unknown, where: string): string {
  if (!isSupported(value)) {
    throw new RangeError(`${where} names an unsupported provider`);
  }
  return value;
}

/** Return the nonsecret SHA-256 identity recorded for an environment API key. */
export function keyIdentity(key: string): string {
  return createHash("sha256").update(key, "utf8").digest("hex");
}

/**
 * Read saved choices and the environment once, without resolving credentials, and return the
 * snapshot.
 *
 * The default provider is `YIAN_DEFAULT_PROVIDER`, else the saved default, else none. Each saved
 * selection supplies its provider's model; `YIAN_DEFAULT_MODEL` overrides the default provider's.
 * Each keyed provider gets a connection: its saved API-key login (or, for OpenAI, subscription),
 * else its key environment variable. A saved login for a provider this version does not support
 * is skipped unless it is the default. `store` defaults to the store `env` names.
 *
 * Throws `RangeError` if `YIAN_DEFAULT_PROVIDER` or the saved default names an unsupported
 * provider, or a supported provider's saved login is of a kind it cannot use (such as a
 * subscription outside OpenAI, or an Azure login without a valid endpoint). Throws `AuthError` if
 * the store cannot be read.
 */
export function connectionSnapshot(options: { env?: Env; store?: AuthStore } = {}): ConnectionSnapshot {
  const { env = processEnv } = options;
  const store = options.store ?? new AuthStore(null, { env });
  const state = store.read();
  const envProvider = env("YIAN_DEFAULT_PROVIDER") || null;
  const apiEndpoint = env("OPENAI_BASE_URL") || OPENAI_API_ENDPOINT;
  const provider = (envProvider || state.defaultProvider || "").toLowerCase() || null;
  if (envProvider !== null && !isSupported(provider)) {
    throw new RangeError(
      `YIAN_DEFAULT_PROVIDER=${reprText(envProvider)} is not a supported provider; ` +
        `set it to one of: ${SUPPORTED_LLM_PROVIDERS.join(", ")}`,
    );
  }
  const models: Record<string, string> = {};
  const connections: Record<string, Connection> = {};
  for (const selection of state.selections) {
    if (!isSupported(selection.provider)) {
      if (envProvider === null && selection.provider === state.defaultProvider) {
        throw new RangeError(
          `The saved default provider ${reprText(selection.provider)} is not supported; ` +
            `run 'yian auth ${selection.provider} --remove'`,
        );
      }
      // A login for a provider this version no longer runs selects nothing.
      continue;
    }
    if (!isUsableLogin(selection)) {
      const name = selection.provider;
      throw new RangeError(`The saved ${name} login is invalid; run 'yian auth ${name}' again`);
    }
    models[selection.provider] = selection.model;
    if (selection.provider === "openai") {
      connections[selection.provider] = {
        // `isUsableLogin` refuses an OpenAI login without a credential.
        method: selection.method as Connection["method"],
        source: "store",
        credential_id: selection.credentialId,
        endpoint: selection.method === "subscription" ? SUBSCRIPTION_ENDPOINT : apiEndpoint,
        store_path: store.path,
      };
    } else if (selection.method === "api_key") {
      connections[selection.provider] = {
        method: "api_key",
        source: "store",
        credential_id: selection.credentialId,
        endpoint: selection.endpoint,
        store_path: store.path,
      };
    }
    if (selection.method === "api_key") Object.assign(connections[selection.provider]!, selectionSettings(selection));
  }
  for (const keyed of KEYED) {
    if (keyed !== "openai" && !Object.hasOwn(connections, keyed)) {
      connections[keyed] = structuredClone(ENVIRONMENT_CONNECTION);
    }
  }
  let strayModel = false;
  const envModel = env("YIAN_DEFAULT_MODEL");
  if (envModel) {
    if (provider === null) strayModel = true;
    else models[provider] = envModel;
  }
  if (!Object.hasOwn(connections, "openai")) {
    const key = env("OPENAI_API_KEY");
    connections["openai"] = {
      method: "api_key",
      source: key ? "environment" : "missing",
      credential_id: key ? keyIdentity(key) : null,
      endpoint: apiEndpoint,
      store_path: null,
    };
  }
  const defaults: Defaults = {};
  if (provider !== null) {
    defaults.provider = provider;
    if (Object.hasOwn(models, provider)) defaults.model = models[provider]!;
  }
  const validated = Object.fromEntries(
    Object.entries(connections).map(([name, value]) => [name, validateConnection(value, name)]),
  );
  return new ConnectionSnapshot({ defaults, providerModels: models, connections: validated, apiEndpoint, strayModel });
}

/**
 * Return the connection fields that record a saved API-key login's settings: `organization` and
 * `project` when set, and the sorted `header_names` when the login has headers; never a header
 * value.
 */
function selectionSettings(selection: ProviderSelection): Partial<Connection> {
  const settings: Partial<Connection> = {};
  if (selection.organization !== null) settings.organization = selection.organization;
  if (selection.project !== null) settings.project = selection.project;
  if (selection.headerNames.length > 0) settings.header_names = [...selection.headerNames].sort(compareCodeUnits);
  return settings;
}

/**
 * Validate the exact nonsecret metadata schema before use or replay, and return a deep copy of
 * `value`.
 *
 * An OpenAI connection records its endpoint and may be a subscription, a host key, or a missing
 * or legacy environment key. Another keyed provider's connection is an API key, either a saved
 * login (`store`, pinned by credential id and store path, with the resource endpoint for `azure`)
 * or the provider's key environment variable (`environment`, unpinned). A saved API-key login may
 * also record the settings saved with it: `organization` and `project` (`openai` only) and the
 * sorted `header_names`. `provider`, the provider of the config that carries it, defaults to
 * `"openai"`.
 *
 * Throws `RangeError` if the metadata does not fit the schema for `provider`, or `provider` takes
 * no connection. Errors never echo values.
 */
export function validateConnection(value: unknown, provider: unknown = "openai"): Connection {
  if (typeof provider !== "string" || !KEYED.has(provider)) {
    const name = isSupported(provider) ? `provider ${provider}` : "an unsupported provider";
    throw new RangeError(`Internal connection metadata is not used for ${name}`);
  }
  if (
    !isRecord(value) ||
    ![...FIELDS].every((field) => Object.hasOwn(value, field)) ||
    !Object.keys(value).every((key) => FIELDS.has(key) || SETTINGS_FIELDS.has(key))
  ) {
    throw new RangeError("Invalid internal connection metadata fields");
  }
  checkConnectionSettings(value, provider);
  if (provider !== "openai") return validateKeyedConnection(value, provider);
  const { method, source } = value;
  if (
    !["api_key", "subscription"].includes(method as string) ||
    !["store", "environment", "missing", "legacy_environment", "host"].includes(source as string)
  ) {
    throw new RangeError("Invalid internal connection method or source");
  }
  if (typeof value["endpoint"] !== "string") throw new RangeError("Invalid internal connection endpoint");
  if (!isHttpEndpoint(value["endpoint"])) {
    throw new RangeError("Connection endpoint must be an HTTP URL without credentials, query, or fragment");
  }
  const identity = value["credential_id"];
  const path = value["store_path"];
  if (source === "store") {
    if (typeof identity !== "string" || !identity || typeof path !== "string" || !isAbsolute(path)) {
      throw new RangeError("Invalid stored connection identity");
    }
  } else if (
    path !== null ||
    (source === "environment" && (typeof identity !== "string" || [...identity].length !== 64)) ||
    (source !== "environment" && identity !== null)
  ) {
    throw new RangeError("Invalid environment connection identity");
  }
  if (method === "subscription" && (source !== "store" || value["endpoint"] !== SUBSCRIPTION_ENDPOINT)) {
    throw new RangeError("Invalid subscription connection routing");
  }
  return structuredClone(value) as unknown as Connection;
}

/**
 * Whether `text` is an `http`/`https` URL of a host, without credentials, a query or a fragment.
 *
 * The scheme, `//` and a non-empty network location are checked on the text first, as
 * `urlsplit` reads them: `new URL` would also accept `https:host` and `https:///x`, which name no
 * host there.
 */
function isHttpEndpoint(text: string): boolean {
  if (!/^https?:\/\/[^/?#]/i.test(text)) return false;
  let url: URL;
  try {
    url = new URL(text);
  } catch {
    return false;
  }
  return url.hostname !== "" && url.username === "" && url.password === "" && url.search === "" && url.hash === "";
}

/**
 * Check the optional settings fields of a connection with the connection fields, for a keyed
 * `provider`.
 *
 * Throws `RangeError` if a settings field is on a connection other than a saved API-key login,
 * `organization` or `project` is on a provider other than `openai`, or a value is malformed.
 * Errors name only the field.
 */
function checkConnectionSettings(value: Readonly<Record<string, unknown>>, provider: string): void {
  if (!Object.keys(value).some((key) => SETTINGS_FIELDS.has(key))) return;
  if (value["source"] !== "store" || value["method"] !== "api_key") {
    throw new RangeError("Internal connection settings are only for a saved API-key login");
  }
  for (const field of ["organization", "project"]) {
    if (!Object.hasOwn(value, field)) continue;
    if (provider !== "openai") throw new RangeError(`Internal connection ${field} is only for provider openai`);
    rethrowRange(() => validateOrganization(value[field], field), `Invalid internal connection ${field}`);
  }
  if (Object.hasOwn(value, "header_names")) {
    rethrowRange(() => validateHeaderNames(value["header_names"], provider), "Invalid internal connection header_names");
  }
}

/** Run `check`, replacing a `RangeError` it throws with one saying `message`. */
function rethrowRange(check: () => unknown, message: string): void {
  try {
    check();
  } catch (error) {
    if (error instanceof RangeError) throw new RangeError(message);
    throw error;
  }
}

/**
 * Validate a non-OpenAI keyed provider's connection fields and return a deep copy of `value`.
 *
 * Throws `RangeError` if the method, source, identity, or endpoint is invalid.
 */
function validateKeyedConnection(value: Readonly<Record<string, unknown>>, provider: string): Connection {
  if (value["method"] !== "api_key" || (value["source"] !== "store" && value["source"] !== "environment")) {
    throw new RangeError("Invalid internal connection method or source");
  }
  const identity = value["credential_id"];
  const path = value["store_path"];
  const endpoint = value["endpoint"];
  if (value["source"] === "environment") {
    if (identity !== null || path !== null || endpoint !== null) {
      throw new RangeError("Invalid environment connection identity");
    }
    return structuredClone(value) as unknown as Connection;
  }
  if (typeof identity !== "string" || !identity || typeof path !== "string" || !isAbsolute(path)) {
    throw new RangeError("Invalid stored connection identity");
  }
  if (provider === "azure") {
    rethrowRange(() => validateBaseUrl(endpoint), "Invalid stored Azure connection endpoint");
  } else if (endpoint !== null) {
    throw new RangeError("Invalid stored connection endpoint");
  }
  return structuredClone(value) as unknown as Connection;
}

/**
 * Validate a recorded `HEADER_NAMES_KEY` value, the sorted names of host headers, and return a
 * copy of it. `provider` is the provider of the config that carries it.
 *
 * Throws `RangeError` if `value` is not a non-empty sorted list of distinct (ignoring case) header
 * names that `provider` may be sent; errors name only the key.
 */
export function validateHeaderNames(value: unknown, provider: unknown): string[] {
  const invalid = new RangeError(`Invalid internal ${HEADER_NAMES_KEY}`);
  if (!isList(value) || value.length === 0) throw invalid;
  // `for...of` visits a hole as `undefined`, so a sparse list fails too.
  for (const name of value) if (typeof name !== "string") throw invalid;
  const names = value as string[];
  const sorted = [...names].sort(compareCodeUnits);
  if (sorted.some((name, index) => name !== names[index])) throw invalid;
  if (new Set(names.map((name) => name.toLowerCase())).size !== names.length) throw invalid;
  for (const name of names) {
    rethrowRange(
      () => validateHeaderName(name, typeof provider === "string" ? provider : null),
      invalid.message,
    );
  }
  return [...names];
}

/**
 * Require a marked config's headers to have exactly its recorded header names.
 *
 * `raw` is the config with `HOST_KEY_MARKER` already removed; `names` the recorded header names,
 * empty when none were recorded. Throws `AuthError` if the header names differ, ignoring case; the
 * message names the missing and unexpected header names, never a value.
 */
function checkHostHeaders(raw: Readonly<AgentConfig>, names: readonly string[]): void {
  const value = raw["headers"];
  // An absent or empty value means no headers.
  const headers = !value || (Array.isArray(value) && value.length === 0) ? {} : value;
  // `LLMConfig` names the malformed field.
  if (!isRecord(headers)) return;
  const diff = headerNameDiff(names, Object.keys(headers));
  if (diff === null) return;
  if (Object.keys(headers).length === 0) {
    const expected = byLowerCase(names);
    const listed = [...expected.keys()].sort(compareCodeUnits).map((name) => expected.get(name)!);
    throw new AuthError(
      `This run's ${messageText(raw["provider"])} connection headers ${listed.join(", ")} came from the host's ` +
        "llm_config.headers and are not recorded; pass headers to resume_flow or RunResult.backward",
    );
  }
  throw new AuthError(`llm_config.headers must have the header names this run recorded for the host key: ${diff}`);
}

/** Return `value` as a message shows it for a provider: `null` and `undefined` as `None`. */
function messageText(value: unknown): string {
  return value === null || value === undefined ? "None" : String(value);
}

/** Say that this run's host key for `provider` is not recorded, and how to pass it again. */
function hostKeyMissing(provider: unknown): string {
  return (
    `This run's ${messageText(provider)} API key came from the host's llm_config.api_key and is not ` +
    "recorded; pass api_key to resume_flow or RunResult.backward"
  );
}

/** Return `names` keyed by their lowercase form; a later name wins a clash. */
function byLowerCase(names: Iterable<string>): Map<string, string> {
  return new Map(Array.from(names, (name): [string, string] => [name.toLowerCase(), name]));
}

/**
 * Describe how the given header names differ from the expected ones, ignoring case.
 *
 * Returns `"missing A, B; not recorded C"` (either part only when the other is empty), names
 * sorted ignoring case; `null` when the names match.
 */
export function headerNameDiff(expected: Iterable<string>, given: Iterable<string>): string | null {
  const expectedNames = byLowerCase(expected);
  const givenNames = byLowerCase(given);
  const missing = [...expectedNames.keys()]
    .sort(compareCodeUnits)
    .filter((name) => !givenNames.has(name))
    .map((name) => expectedNames.get(name)!);
  const extra = [...givenNames.keys()]
    .sort(compareCodeUnits)
    .filter((name) => !expectedNames.has(name))
    .map((name) => givenNames.get(name)!);
  const details: string[] = [];
  if (missing.length > 0) details.push(`missing ${missing.join(", ")}`);
  if (extra.length > 0) details.push(`not recorded ${extra.join(", ")}`);
  return details.length > 0 ? details.join("; ") : null;
}

/** Return `config`'s set fields as a new record, its key and header values included. */
function setFields(config: LLMConfig): AgentConfig {
  const raw: AgentConfig = Object.fromEntries(Object.entries(config.dump()).filter(([, value]) => value !== null));
  // Dumps leave the key and header values out; carry them across explicitly.
  if (config.apiKey !== null) raw["api_key"] = config.apiKey;
  if (config.headers !== null) raw["headers"] = { ...config.headers };
  return raw;
}

/** Remove `key` from `raw` and return its value; `undefined` when absent. */
function pop(raw: AgentConfig, key: string): unknown {
  const value = raw[key];
  delete raw[key];
  return value;
}

/**
 * Validate public fields separately from host-owned connection metadata, and return the validated
 * fields with the pinned connection, if any.
 *
 * `config` holds yian values (ints as `bigint`, as YAML and run records give them); host configs go
 * through `fromHost` at the host entry points (`runFlow`, the factory, memory). A config marked
 * with `HOST_KEY_MARKER` needs its `api_key`, and `headers` with exactly the recorded
 * `HEADER_NAMES_KEY` names (none when absent), compared without case; both are popped from the
 * validated fields. `config` is not mutated.
 *
 * Throws `AuthError` if the config carries `HOST_KEY_MARKER` but no `api_key`, or its headers do
 * not have the recorded names. Throws `RangeError` if a public field or the metadata is invalid,
 * the header names are malformed or appear without the marker, an OpenAI `base_url` differs from
 * the endpoint of its pinned host connection, or an Azure `base_url` names another resource than
 * its saved login.
 */
export function splitModelConfig(config: Readonly<AgentConfig> | LLMConfig): [LLMConfig, Connection | null] {
  const raw = config instanceof LLMConfig ? setFields(config) : Object.fromEntries(Object.entries(config));
  const recordedNames = pop(raw, HEADER_NAMES_KEY) ?? null;
  const headerNames = recordedNames === null ? null : validateHeaderNames(recordedNames, raw["provider"]);
  const marker = pop(raw, HOST_KEY_MARKER);
  if (marker !== undefined) {
    if (marker !== true) throw new RangeError("Invalid internal host key marker");
    if (raw["api_key"] === undefined || raw["api_key"] === null) throw new AuthError(hostKeyMissing(raw["provider"]));
    checkHostHeaders(raw, headerNames ?? []);
  } else if (headerNames !== null) {
    throw new RangeError(`Internal ${HEADER_NAMES_KEY} needs the host key marker`);
  }
  const metadata = pop(raw, "_connection") ?? null;
  const validated = new LLMConfig(raw);
  if (metadata === null) return [validated, null];
  const connection = validateConnection(metadata, validated.provider);
  if (
    validated.provider === "azure" &&
    validated.baseUrl !== null &&
    connection.source === "store" &&
    // `validateConnection` requires a saved Azure login to record its endpoint as a string.
    azureResourceEndpoint(validated.baseUrl) !== azureResourceEndpoint(connection.endpoint as string)
  ) {
    // The saved key belongs to one resource; never send it to another.
    throw new RangeError("llm_config.base_url names another Azure resource than its saved login");
  }
  if (
    validated.provider === "openai" &&
    validated.baseUrl !== null &&
    (connection.source !== "host" || connection.endpoint !== validated.baseUrl)
  ) {
    // A base_url serves only the host key; it must match the pinned route.
    throw new RangeError("llm_config.base_url must match its host connection endpoint");
  }
  return [validated, connection];
}

/**
 * Validate a resolved config as the factory does, without building a model, and return the
 * validated public fields with the pinned connection, if any.
 *
 * `config` holds yian values (ints as `bigint`); host configs go through `fromHost` at the host
 * entry points (`runFlow`, the factory, memory).
 *
 * Throws `AuthError` if the config carries `HOST_KEY_MARKER` but no `api_key`. Throws `RangeError`
 * if a field is invalid, the connection rejects a set field, an `azure` config has no `base_url`,
 * or the provider's block does not fit the API the call actually reaches (e.g. an `openai` block
 * with `temperature` on a ChatGPT subscription connection).
 */
export function checkModelConfig(config: Readonly<AgentConfig> | LLMConfig): [LLMConfig, Connection | null] {
  const [validated, connection] = splitModelConfig(config);
  if (validated.provider === "azure" && validated.baseUrl === null) throw new RangeError(AZURE_NEEDS_BASE_URL);
  if (validated.provider !== null) checkProviderBlock(validated, validated.provider, connection);
  return [validated, connection];
}

/**
 * Return `.<provider>` when `checkModelConfig` refused only the provider's block, else `""`.
 *
 * Its fields and connection are checked first and an Azure config needs a `base_url`; a config that
 * passes those was refused for its block alone. Never throws for a refused config.
 */
export function refusedBlock(config: Readonly<AgentConfig>): string {
  let validated: LLMConfig;
  try {
    [validated] = splitModelConfig(config);
  } catch (error) {
    // What `splitModelConfig` throws for a refused config; anything else is a bug to surface.
    if (error instanceof RangeError || error instanceof AuthError) return "";
    throw error;
  }
  if (validated.provider === null || (validated.provider === "azure" && validated.baseUrl === null)) return "";
  return `.${validated.provider}`;
}

const LEADING_KEY_PATH = /^`[^`.]+\.([^`]+)`: ([\s\S]*)$/;

/**
 * Name a provider block's problem under `where`, giving its key path once.
 *
 * A single problem that opens with its backticked key path and a colon, such as
 * `` `openai.temperature`: ... ``, reads `<where>.openai.temperature<callsite>: ...`. Any other
 * reads `<where>.<block><callsite>: <message>`, or `<where><callsite>: <message>` when `block` is
 * empty. `where` names the layer, such as `node 'chat' llm_config`; `block` is the block refused,
 * such as `openai`; `callsite` follows the field, such as `` in 'sub'``.
 */
export function blockProblem(where: string, block: string, message: string, callsite = ""): string {
  const match = LEADING_KEY_PATH.exec(message);
  // Several problems are joined with "; "; moving the first one's path would misname the rest.
  if (block && match !== null && !match[2]!.includes("; `")) {
    return `${where}.${block}.${match[1]}${callsite}: ${match[2]}`;
  }
  return `${where}${block ? `.${block}` : ""}${callsite}: ${message}`;
}

/**
 * Check the provider's block against the schema of the API the call reaches.
 *
 * Load time checks a block against the provider's usual API; the connection can route the call to
 * a narrower one (the ChatGPT subscription endpoint), known only here. Throws `RangeError` if the
 * block has a key or value that API refuses; the message is the block schema's own and never
 * echoes values.
 */
function checkProviderBlock(validated: LLMConfig, provider: string, connection: Connection | null): void {
  const schema = blockSchema(provider, connection, validated.apiKey);
  const { issues } = checkBlock(schema, validated.blockFor(provider));
  if (issues.length > 0) throw new RangeError(issues.map((issue) => blockErrorText(provider, issue)).join("; "));
}

/** Return one block validation issue as text naming its key path, without the input. */
function blockErrorText(provider: string, issue: Issue): string {
  if (issue.type === "value_error") return issue.error!;
  return `\`${provider}.${issue.loc.join(".")}\`: ${issue.msg}`;
}

/**
 * Resolve a fresh config, preserving any existing pinned connection, and return it as a new record.
 *
 * `config` holds yian values (ints as `bigint`); host configs go through `fromHost` at the host
 * entry points. An unset provider or model comes from the snapshot's defaults and saved models. A
 * config with a model and a pinned connection, or a non-OpenAI config with a provider and model
 * that needs no snapshot connection (keyless, or with its own `api_key`), returns without reading
 * the snapshot. Otherwise an OpenAI config with an `api_key` gets a `host` connection, and any
 * other keyed config gets its provider's connection as `ConnectionSnapshot.bind` attaches it,
 * including an Azure login's endpoint as `base_url`. The snapshot is `snapshot`, else one read
 * from `env` then.
 *
 * Throws `RangeError` if no layer sets the provider or its model, or a saved login is invalid
 * (and as `splitModelConfig` does). Throws `AuthError` if the credential store cannot be read.
 */
export function resolveModelConfig(
  config: Readonly<AgentConfig> | LLMConfig,
  options: { snapshot?: ConnectionSnapshot; env?: Env } = {},
): AgentConfig {
  const [validated, metadata] = splitModelConfig(config);
  const raw = setFields(validated);
  if (metadata !== null && validated.model) {
    raw["_connection"] = metadata;
    return raw;
  }
  if (
    validated.provider !== "openai" &&
    validated.provider &&
    validated.model &&
    (!KEYED.has(validated.provider) || validated.apiKey !== null)
  ) {
    return raw;
  }
  const snapshot = options.snapshot ?? connectionSnapshot({ env: options.env });
  const provider = validated.provider || snapshot.defaults.provider || null;
  const saved = provider !== null && Object.hasOwn(snapshot.providerModels, provider) ? snapshot.providerModels[provider] : null;
  const model = validated.model || saved || null;
  if (provider === null || model === null) {
    throw new RangeError(missingModelMessage(null, provider, { strayModel: snapshot.strayModel }));
  }
  raw["provider"] = provider;
  raw["model"] = model;
  if (metadata !== null) {
    raw["_connection"] = metadata;
  } else if (provider === "openai" && validated.apiKey !== null) {
    raw["_connection"] = snapshot.hostConnection(validated.baseUrl);
  } else if (validated.apiKey === null) {
    attach(snapshot, raw);
  }
  return raw;
}

/**
 * Pin an old record to its original route without I/O, and return the pinned copy; `config` is
 * not mutated.
 *
 * An old OpenAI record gets its legacy environment API-key route. Another keyed provider's record,
 * unless its key came from the host, gets the environment connection: those records always read the
 * provider's key variable.
 */
export function legacyConnectionConfig(config: Readonly<AgentConfig>): AgentConfig {
  const result = structuredClone(config) as AgentConfig;
  // Old records serialized every optional public field, including these empty retired defaults
  // (sandbox/cwd, and the keys now written in provider blocks). As nulls they configured nothing;
  // a set one still fails validation.
  const nullable = [...RETIRED_LLM_CONFIG_KEYS];
  if (result["provider"] !== "codex") nullable.push("sandbox", "cwd");
  for (const field of nullable) {
    if (Object.hasOwn(result, field) && result[field] === null) delete result[field];
  }
  const provider = result["provider"];
  if (provider === "openai" && result["_connection"] === undefined) {
    result["_connection"] = {
      method: "api_key",
      source: "legacy_environment",
      credential_id: null,
      endpoint: OPENAI_API_ENDPOINT,
      store_path: null,
    };
  } else if (
    typeof provider === "string" &&
    KEYED.has(provider) &&
    result["_connection"] === undefined &&
    result[HOST_KEY_MARKER] === undefined
  ) {
    result["_connection"] = structuredClone(ENVIRONMENT_CONNECTION);
  }
  return result;
}

/** Adapt a recorded effective map at the host replay boundary; `effective` is not mutated. */
export function legacyConnectionMap(
  effective: Readonly<Record<string, Readonly<AgentConfig>>>,
): Record<string, AgentConfig> {
  return Object.fromEntries(Object.entries(effective).map(([path, config]) => [path, legacyConnectionConfig(config)]));
}

/**
 * Resolve fresh credentials while enforcing the recorded connection identity.
 *
 * A `store` connection resolves the saved login it was pinned to, so a later login or removal
 * fails instead of switching accounts; an API key comes with the header values saved beside it.
 * `refresh` renews an expiring OpenAI subscription and defaults to `OpenAIAuth`'s; other providers
 * get none. OpenAI also resolves its environment key, checking that it is the recorded one. Another
 * provider's `environment` connection resolves nothing here: its client reads the key variable
 * itself. `provider` is the provider of the config that carries `connection`; `signal` and
 * `scheduler` go to the store and the default refresh.
 *
 * Rejects with `AuthError` if the pinned login changed, was removed, or cannot be read, or an
 * OpenAI key is missing or changed; with `RangeError` if the metadata is invalid for `provider`,
 * or it is a non-OpenAI `environment` connection.
 */
export async function connectionCredentials(
  connection: unknown,
  provider = "openai",
  options: {
    env?: Env;
    refresh?: (current: OAuthCredentials) => OAuthCredentials | Promise<OAuthCredentials>;
    signal?: AbortSignal;
    scheduler?: Scheduler;
  } = {},
): Promise<Credentials> {
  const { env = processEnv, refresh, signal, scheduler } = options;
  const checked = validateConnection(connection, provider);
  if (checked.source === "store") {
    let credentials: Credentials;
    try {
      // `validateConnection` requires a saved login's credential id and absolute store path.
      const store = new AuthStore(checked.store_path, { env, signal, scheduler });
      credentials = await store.credentials(provider, checked.credential_id!, {
        refresh:
          provider === "openai"
            ? (refresh ?? ((current) => new OpenAIAuth({ signal, scheduler }).refresh(current)))
            : undefined,
      });
    } catch (error) {
      if (error instanceof AuthError) {
        throw new AuthError(`${error.message}. Run 'yian auth ${provider}' and start a new run`, {
          cause: error,
          status: error.status ?? undefined,
        });
      }
      throw error;
    }
    if (credentials.method !== checked.method) {
      throw new AuthError(`The saved connection method changed; run 'yian auth ${provider}' and start a new run`);
    }
    return credentials;
  }
  if (provider !== "openai") throw new RangeError(`An environment connection for ${provider} is read by its client`);
  if (checked.source === "host") throw new AuthError(hostKeyMissing("openai"));
  if (checked.source === "missing") {
    throw new AuthError(
      "No OpenAI connection was selected; run 'yian auth openai' or set OPENAI_API_KEY before starting a new run",
    );
  }
  const key = env("OPENAI_API_KEY");
  if (!key) {
    throw new AuthError("This run needs its OpenAI API key in OPENAI_API_KEY; use 'yian auth openai' for a new run");
  }
  if (checked.source === "environment" && keyIdentity(key) !== checked.credential_id) {
    throw new AuthError("The OpenAI environment API key changed since this run started; start a new run");
  }
  return new ApiKeyCredentials(key);
}

/**
 * Return every credential the model call built from `config` may send: the distinct plain
 * credentials in first-seen order, possibly none. Never throws.
 *
 * `config` is an `LLMConfig` or a record of yian values; `null` and `undefined` resolve like `{}`.
 * A config `api_key` is sent in place of any saved login or key variable, and the config's
 * `headers` values are listed whenever given. Otherwise a `store` connection sends its provider's
 * saved login, read without refresh: its key and saved header values, or its subscription tokens
 * and account id, expired ones too. Otherwise (an `environment` or `legacy_environment`
 * connection, or none) the client reads the first set variable of `apiKeyEnvNames` (for Google,
 * `GOOGLE_API_KEY` then `GEMINI_API_KEY`). `AZURE_OPENAI_AD_TOKEN` is listed for `azure`
 * whenever set. For `ollama`, a non-empty `OLLAMA_BASE_URL` whose userinfo has a non-empty user
 * lists the Basic auth password and token the route sends (`splitOllamaBaseUrl`); a malformed
 * value lists nothing. A config without connection metadata is resolved first, as the factory
 * does. Each step is guarded, so a failing one still leaves the others' credentials. Every header
 * value is listed however short, so a value such as `prod` is scrubbed wherever it appears.
 */
export function callCredentials(config: unknown, options: { env?: Env } = {}): readonly string[] {
  const { env = processEnv } = options;
  const found: string[] = [];
  const value = config ?? {};
  let provider: unknown = null;
  let hostKey: string | null = null;
  let connection: unknown = null;
  let headers: unknown = null;
  try {
    let key: unknown = null;
    if (value instanceof LLMConfig) {
      [provider, key, headers] = [value.provider, value.apiKey, value.headers];
    } else if (isRecord(value)) {
      [provider, key] = [value["provider"] ?? null, value["api_key"]];
      [connection, headers] = [value["_connection"] ?? null, value["headers"]];
    }
    if (typeof key === "string" && key) {
      found.push(key);
      hostKey = key;
    }
  } catch {}
  try {
    if (isRecord(headers)) {
      for (const header of Object.values(headers)) if (typeof header === "string" && header) found.push(header);
    }
  } catch {}
  if (connection === null && (hostKey === null || provider === null)) {
    try {
      if (value instanceof LLMConfig || isRecord(value)) {
        const resolved = resolveModelConfig(value, { env });
        [provider, connection] = [resolved["provider"] ?? null, resolved["_connection"] ?? null];
      }
    } catch {}
  }
  if (typeof provider !== "string") return [...new Set(found)];
  const name = provider.toLowerCase();
  if (hostKey === null) {
    try {
      const source = isRecord(connection) ? connection["source"] : connection === null ? null : undefined;
      if (source === "store" && isRecord(connection)) {
        // A store connection without `store_path` adds nothing, and `AuthStore` takes only a
        // string or null path.
        const path = Object.hasOwn(connection, "store_path") ? connection["store_path"] : undefined;
        if (typeof path === "string" || path === null) found.push(...new AuthStore(path, { env }).savedSecrets(name));
      } else if (source === null || source === "environment" || source === "legacy_environment") {
        for (const variable of apiKeyEnvNames(name)) {
          const key = env(variable);
          if (key) {
            found.push(key);
            break;
          }
        }
      }
    } catch {}
  }
  try {
    for (const variable of ALWAYS_SENT_ENV.get(name) ?? []) {
      const token = env(variable);
      if (token) found.push(token);
    }
  } catch {}
  if (name === "ollama") {
    try {
      const url = env("OLLAMA_BASE_URL");
      const credentials = url ? splitOllamaBaseUrl(url)[1] : null;
      if (credentials !== null) found.push(...new BasicAuth(...credentials).credentials());
    } catch {}
  }
  return [...new Set(found)];
}

/**
 * Redact every credential `callCredentials` lists for `config`, and each of `sent`, from
 * `error` as `scrubError` does, and return `error` (a string error comes back redacted). Never
 * throws.
 *
 * `sent` is what the route's auth reported sending; `env` is read only through `callCredentials`.
 */
export function scrubCallCredentials<T>(
  error: T,
  config: unknown,
  options: { env?: Env; sent?: Iterable<string> } = {},
): T {
  try {
    const keys = [...callCredentials(config, options)];
    // A throwing `sent` must not drop the config's credentials from the scrub.
    try {
      keys.push(...(options.sent ?? []));
    } catch {}
    return scrubError(error, (text) => redactKeys(text, keys));
  } catch {
    // Scrubbing is best effort; the provider error must survive it.
    return error;
  }
}
