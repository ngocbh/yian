/**
 * `LLMConfig`: model selection for an AGENT node.
 *
 * Lives with the clients that consume it: the AGENT node and the Compose loader's `llm_config:`
 * field carry an `LLMConfig` (or the plain record that normalizes to one), and the factory turns
 * API selections into ready chat models.
 */

import { isIPv6 } from "node:net";

import { current } from "../defaults.ts";
import { isRecord } from "../typesys/index.ts";
import type { JsonObject } from "../typesys/json.ts";
import { validateHeaderName, validateHeaders, validateOrganization } from "./connection_settings.ts";
import { blockSchema, checkBlock, type Issue, ValidationError } from "./provider_blocks.ts";

// One provider vocabulary shared by the validated config carrier and the factory that consumes it.
/** Every provider an `llm_config` may name, in Python yian's order. */
export const SUPPORTED_LLM_PROVIDERS = Object.freeze([
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
  "ollama",
  "anthropic",
  "google",
  "azure",
] as const);

/** A provider name `llm_config.provider` takes. */
export type LLMProvider = (typeof SUPPORTED_LLM_PROVIDERS)[number];

// The native Ollama client sends no key, so it cannot take a host base_url.
const OLLAMA_NO_BASE_URL =
  "provider ollama does not take llm_config.base_url; it sends no api_key, so set OLLAMA_BASE_URL instead";

/** The error for an Azure call without its resource endpoint. */
export const AZURE_NEEDS_BASE_URL =
  "provider azure needs llm_config.base_url, the Azure OpenAI resource endpoint " +
  "(https://<resource>.openai.azure.com/); set llm_config.base_url beside " +
  "provider: azure. AZURE_OPENAI_ENDPOINT is not read";

const CODEX_REMOVED =
  "Native Codex configuration has been removed. Run 'yian auth openai', use provider: openai, " +
  "and remove sandbox/cwd. Yian tools are granted explicitly and run on the host.";

const LOCAL_HOSTS: readonly string[] = ["localhost", "127.0.0.1", "::1"];
// RFC 3986 characters minus "?" and "#", since a query or fragment is refused anyway. Everything
// else (whitespace, controls, non-ASCII) is something a URL parser or HTTP client would strip,
// normalise, or encode, so the string sent would differ from the one checked here.
const URL_CHARACTERS = /^[A-Za-z0-9\-._~:/\[\]@!$&'()*+,;=%]*$/;
// A DNS name or bracketed IPv6 literal, an optional all-digit port, then a path. The host part has
// no "@", so userinfo never matches.
const URL_SHAPE =
  /^(?<scheme>https?):\/\/(?<host>[A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(?::(?<port>[0-9]{1,5}))?(?<path>\/.*)?$/;

const PROVIDER_CHOICES = (() => {
  const quoted = SUPPORTED_LLM_PROVIDERS.map((provider) => `'${provider}'`);
  return `Input should be ${quoted.slice(0, -1).join(", ")} or ${quoted.at(-1)!}`;
})();

/** Every top-level `llm_config` key, spelled as authored. */
export const LLM_CONFIG_KEYS: ReadonlySet<string> = new Set([
  "provider",
  "model",
  "api_key",
  "base_url",
  "organization",
  "project",
  "headers",
  ...SUPPORTED_LLM_PROVIDERS,
]);

// Keys removed from `llm_config` -> the message showing their provider-block form. Recorded runs
// may still carry them as null (see `RETIRED_LLM_CONFIG_KEYS`).
const RETIRED_KEY_MESSAGES: ReadonlyMap<string, string> = new Map([
  ["anthropic_effort", "`anthropic_effort` was removed; write `anthropic: {output_config: {effort: <level>}}`"],
  [
    "openai_reasoning_effort",
    "`openai_reasoning_effort` was removed; write `openai: {reasoning: {effort: <level>}}`",
  ],
  [
    "google_thinking_level",
    "`google_thinking_level` was removed; write " +
      "`google: {generationConfig: {thinkingConfig: {thinkingLevel: <LEVEL>}}}`",
  ],
]);
// Where `temperature` lives inside each block whose API does not take it top-level.
const TEMPERATURE_FORMS: ReadonlyMap<string, string> = new Map([
  ["google", "`google: {generationConfig: {temperature: <value>}}`"],
  ["ollama", "`ollama: {options: {temperature: <value>}}`"],
]);

/** The keys retired from `llm_config`, each now written inside a provider block. */
export const RETIRED_LLM_CONFIG_KEYS: ReadonlySet<string> = new Set([...RETIRED_KEY_MESSAGES.keys(), "temperature"]);

type Data = Readonly<Record<string, unknown>>;

/** Return `data`'s own value for `key`, so inherited names such as `constructor` are absent. */
function own(data: Data, key: string): unknown {
  return Object.hasOwn(data, key) ? data[key] : undefined;
}

/** Whether `data` holds `key` as written: an own property that isn't `undefined` (`null` counts). */
function has(data: Data, key: string): boolean {
  return own(data, key) !== undefined;
}

/** Whether `value` is one of `SUPPORTED_LLM_PROVIDERS`. */
function isProvider(value: unknown): value is LLMProvider {
  return typeof value === "string" && (SUPPORTED_LLM_PROVIDERS as readonly string[]).includes(value);
}

/** Return the `value_error` issue at `loc` for a validator's `message`. */
function valueError(loc: readonly (string | number)[], message: string): Issue {
  return { loc, type: "value_error", msg: `Value error, ${message}`, error: message };
}

/**
 * Return the migration message for the first retired key in an `llm_config` layer.
 *
 * The retired keys are `anthropic_effort`, `openai_reasoning_effort`, `google_thinking_level` and
 * top-level `temperature`; each now lives in a provider block. A key counts whatever its value
 * (`null` too), and the message never echoes it. A non-record has no retired key. The
 * `temperature` message names the layer's `provider` when it is a supported one; otherwise `null`
 * when there is no retired key.
 *
 * Example: `retiredKeyError({provider: "ollama", temperature: 0.2})` gives
 * "`temperature` moved into the provider block: write `ollama: {options: {temperature: <value>}}`".
 */
export function retiredKeyError(config: unknown): string | null {
  if (!isRecord(config)) return null;
  for (const [key, message] of RETIRED_KEY_MESSAGES) {
    if (has(config, key)) return message;
  }
  if (!has(config, "temperature")) return null;
  const provider = own(config, "provider");
  if (isProvider(provider)) {
    const form = TEMPERATURE_FORMS.get(provider) ?? `\`${provider}: {temperature: <value>}\``;
    return `\`temperature\` moved into the provider block: write ${form}`;
  }
  return (
    "`temperature` moved into the provider block: write `<provider>: {temperature: <value>}` " +
    "(google: `generationConfig.temperature`; ollama: `options.temperature`)"
  );
}

/**
 * Return `value` unchanged if it is an allowed `llm_config.base_url`.
 *
 * The check runs on the exact string the client will use: it must be a lowercase `https://` URL
 * (`http://` only for exactly `localhost`, `127.0.0.1`, or `[::1]`) of a host, an optional port
 * from 1 to 65535, and a path, at most `maxBaseUrlChars` characters long (default 2048).
 * Credentials, a query, a fragment, whitespace, control characters, and non-ASCII characters are
 * refused rather than normalised; write an internationalised host in its `xn--` form. Throws
 * `RangeError` if `value` is not an allowed URL; messages never echo the value.
 */
export function validateBaseUrl(value: unknown): string {
  if (typeof value !== "string") throw new RangeError("base_url must be a string");
  const limit = current().maxBaseUrlChars;
  if ([...value].length > limit) throw new RangeError(`base_url must be at most ${limit} characters`);
  if (!URL_CHARACTERS.test(value)) {
    throw new RangeError(
      "base_url may contain only ASCII URL characters, with no whitespace or control characters; " +
        "write an internationalised host in its xn-- form",
    );
  }
  const groups = URL_SHAPE.exec(value)?.groups;
  let host = groups?.["host"]?.toLowerCase();
  if (host?.startsWith("[")) {
    host = host.slice(1, -1);
    if (!isIPv6(host)) host = undefined;
  }
  if (groups === undefined || !host) {
    throw new RangeError(
      "base_url must be a lowercase https:// or http:// URL of a host, an optional port, and a path, " +
        "without credentials, query, or fragment",
    );
  }
  const port = groups["port"];
  if (port !== undefined && !(Number(port) >= 1 && Number(port) <= 65535)) {
    throw new RangeError("base_url port must be a number from 1 to 65535");
  }
  if (groups["scheme"] !== "https" && !LOCAL_HOSTS.includes(host)) {
    throw new RangeError("base_url must use https; http is allowed only for localhost, 127.0.0.1, and ::1");
  }
  return value;
}

/**
 * Return why `data` uses a removed `llm_config` form, or `null`.
 *
 * Checks, first match wins: a Python spelling of a dash block (`qwen_cn` for `qwen-cn`), a
 * retired key (`retiredKeyError`), then the native Codex options (`provider: codex`, `sandbox`,
 * `cwd`), which point at the unified OpenAI connection.
 */
function removedFormError(data: Data): string | null {
  for (const key of SUPPORTED_LLM_PROVIDERS) {
    const name = key.replaceAll("-", "_");
    if (name !== key && has(data, name)) return `\`${name}\` is not an llm_config key; write \`${key}\``;
  }
  const retired = retiredKeyError(data);
  if (retired !== null) return retired;
  const provider = own(data, "provider");
  const codex = typeof provider === "string" && provider.toLowerCase() === "codex";
  return codex || has(data, "sandbox") || has(data, "cwd") ? CODEX_REMOVED : null;
}

/**
 * Return `value` read as connection headers, appending its issues to `issues`.
 *
 * A non-record is `dict_type` and a non-string value `string_type`; an empty record is `null`;
 * then the names and values must pass `validateHeaders` for every provider.
 */
function readHeaders(value: unknown, issues: Issue[]): Record<string, string> | null {
  if (value === null || value === undefined) return null;
  if (!isRecord(value)) {
    issues.push({ loc: ["headers"], type: "dict_type", msg: "Input should be a valid dictionary" });
    return null;
  }
  const entries = Object.entries(value).filter(([, item]) => item !== undefined);
  const before = issues.length;
  for (const [name, item] of entries) {
    if (typeof item !== "string") {
      issues.push({ loc: ["headers", name], type: "string_type", msg: "Input should be a valid string" });
    }
  }
  if (issues.length > before || entries.length === 0) return null;
  try {
    return validateHeaders(value, null);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    issues.push(valueError(["headers"], error.message));
    return null;
  }
}

/** The validated fields the connection rule reads. */
interface Connection {
  readonly provider: LLMProvider | null;
  readonly apiKey: string | null;
  readonly baseUrl: string | null;
  readonly organization: string | null;
  readonly project: string | null;
  readonly headers: Readonly<Record<string, string>> | null;
}

/**
 * Require base_url and the connection settings to travel with the key they serve.
 *
 * An Azure base_url is the resource endpoint and may come without a key. When `provider` is set,
 * the settings must also apply to it: organization and project only to openai, headers to every
 * provider but ollama, and openai refuses the `OpenAI-Organization`/`OpenAI-Project` header names.
 * A provider that does not take a setting is reported before a missing key, since adding the key
 * would not fix it. Throws `RangeError`.
 */
function bindConnectionToKey(c: Connection): void {
  if (c.baseUrl !== null && c.provider !== "azure") {
    if (c.apiKey === null || c.provider === null) {
      throw new RangeError(
        "llm_config.base_url needs llm_config.api_key and llm_config.provider in the same config, " +
          "since it decides where that key is sent",
      );
    }
    if (c.provider === "ollama") throw new RangeError(OLLAMA_NO_BASE_URL);
  }
  if (c.provider !== null) {
    if ((c.organization !== null || c.project !== null) && c.provider !== "openai") {
      throw new RangeError("llm_config.organization and llm_config.project apply only to provider openai");
    }
    if (c.headers !== null && c.provider === "ollama") {
      throw new RangeError("provider ollama does not take llm_config.headers; it sends no api_key");
    }
    // The field check could not see the provider; openai refuses more names.
    for (const name of Object.keys(c.headers ?? {})) validateHeaderName(name, c.provider);
  }
  for (const field of ["organization", "project", "headers"] as const) {
    if (c[field] !== null && (c.apiKey === null || c.provider === null)) {
      throw new RangeError(
        `llm_config.${field} needs llm_config.api_key and llm_config.provider in the same config, ` +
          "since it goes wherever that key is sent",
      );
    }
  }
}

/**
 * LLM selection for an AGENT node.
 *
 * `new LLMConfig(data)` validates a plain record and throws `ValidationError` (title "LLMConfig")
 * on any problem. A field given `null` or `undefined` is unset. Every field is optional: an unset
 * `provider`/`model` comes from `YIAN_DEFAULT_PROVIDER`/`YIAN_DEFAULT_MODEL`, then the login saved
 * by `yian auth`; with neither, building the model fails, since no provider or model is built in.
 * Only the block named by the selected provider is sent. An unknown key is refused, so a mistyped
 * `llm_config:` key is loud at load time, and a retired key (`temperature`, `anthropic_effort`,
 * ...) fails with its new form (see `retiredKeyError`). Validation errors never echo the input
 * values, so a rejected config cannot leak `api_key`. Instances are frozen.
 *
 * Fields (keys of `data`):
 * - `provider`: one of `SUPPORTED_LLM_PROVIDERS`. Unset takes `YIAN_DEFAULT_PROVIDER`, else the
 *   saved default provider.
 * - `model`: provider-specific model id. Unset takes that provider's default:
 *   `YIAN_DEFAULT_MODEL` for the default provider, else its saved model.
 * - `api_key`: provider API key supplied by the host. It takes precedence over the provider's
 *   saved login and key environment variable. Read it with `apiKey`; it never shows in
 *   `JSON.stringify`, `inspect` or `dump()`. Unset keeps the saved-login or environment key. An
 *   empty key is refused, since some clients would treat it as unset and read the environment.
 * - `base_url`: API endpoint that replaces the provider's default, such as an Azure OpenAI
 *   resource's `https://<resource>.openai.azure.com/openai/v1/` with `provider: "openai"`. It
 *   decides where `api_key` is sent, so it needs `api_key` and `provider` in the same config. It
 *   must pass `validateBaseUrl`: a lowercase `https://` URL without credentials, query, or
 *   fragment (`http://` only for exactly `localhost`, `127.0.0.1` and `[::1]`), checked and stored
 *   as written, at most `YIAN_MAX_BASE_URL_CHARS` characters (default 2048). The `ollama`
 *   provider refuses it. For `provider: "azure"` it is the Azure OpenAI resource endpoint, such as
 *   `https://<resource>.openai.azure.com/`, required when the model is called; it needs no
 *   `api_key`, since the key may come from `AZURE_OPENAI_API_KEY`, and a flow file may set it
 *   beside `provider: azure`. Unlike the key, it appears in `dump()` when set.
 * - `organization`: OpenAI organization id, sent as the `OpenAI-Organization` header. Only
 *   `provider: "openai"` takes it, and it needs `api_key` and `provider` in the same config, since
 *   it goes wherever that key is sent. It must be non-empty visible ASCII of at most
 *   `YIAN_MAX_ORGANIZATION_CHARS` characters (default 256). It appears in `dump()` when set.
 * - `project`: OpenAI project id, sent as the `OpenAI-Project` header, with the same rules as
 *   `organization`.
 * - `headers`: extra request headers, for example for a gateway. They need `api_key` and
 *   `provider` in the same config, and `provider: "ollama"` refuses them. Names are HTTP tokens
 *   compared without case; credential and framing headers (`Authorization`, `x-api-key`, `Host`,
 *   `Content-Type`, ...) are refused for every provider, and `OpenAI-Organization`/`OpenAI-Project`
 *   for `provider: "openai"` (see `REFUSED_HEADER_NAMES`). Values are visible ASCII, with spaces or
 *   tabs only between characters. Names and values are at most `YIAN_MAX_HEADER_VALUE_CHARS`
 *   characters (default 8192), with at most `YIAN_MAX_CONNECTION_HEADERS` headers (default 32).
 *   Values are secrets: like the key, they never show in `dump()`. An empty record is no headers.
 * - `<provider>`: one block per provider in `SUPPORTED_LLM_PROVIDERS` (`openai`, `qwen-cn`,
 *   `anthropic`, ...), named after it: request-body fields for the API that provider's calls
 *   reach, checked by its block in `PROVIDER_BLOCKS`. Every block is checked, but only the one
 *   named by the resolved `provider` is used. The dash names are written with their dash
 *   (`qwen_cn` is refused). Read a block with `blockFor`; an unset block is left out of `dump()`,
 *   and a set one dumps without its unset fields.
 *
 * Example: `new LLMConfig({provider: "anthropic", model: "claude-sonnet-4-5", api_key: key})`
 * has `apiKey === key` and a `dump()` without `api_key`.
 */
export class LLMConfig {
  readonly provider: LLMProvider | null;
  readonly model: string | null;
  readonly baseUrl: string | null;
  readonly organization: string | null;
  readonly project: string | null;
  // Secrets and blocks are private, so JSON.stringify, inspect and spreads never show them.
  readonly #apiKey: string | null;
  readonly #headers: Readonly<Record<string, string>> | null;
  readonly #blocks: ReadonlyMap<string, JsonObject>;

  constructor(data: unknown) {
    if (!isRecord(data)) {
      const msg = "Input should be a valid dictionary";
      throw new ValidationError("LLMConfig", [{ loc: [], type: "model_type", msg }]);
    }
    const removed = removedFormError(data);
    if (removed !== null) throw new ValidationError("LLMConfig", [valueError([], removed)]);

    const issues: Issue[] = [];
    /** Read the string field `key`, then `check` it; a `RangeError` becomes a value error. */
    const text = (key: string, check?: (value: string) => string): string | null => {
      const value = own(data, key);
      if (value === null || value === undefined) return null;
      if (typeof value !== "string") {
        issues.push({ loc: [key], type: "string_type", msg: "Input should be a valid string" });
        return null;
      }
      try {
        return check === undefined ? value : check(value);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        issues.push(valueError([key], error.message));
        return null;
      }
    };
    const given = own(data, "provider");
    const provider = isProvider(given) ? given : null;
    if (given !== null && given !== undefined && provider === null) {
      issues.push({ loc: ["provider"], type: "literal_error", msg: PROVIDER_CHOICES });
    }
    const model = text("model");
    const apiKey = text("api_key", (value) => {
      // Some clients would replace an empty key with an environment variable.
      if (value === "") throw new RangeError("api_key must not be empty");
      return value;
    });
    const baseUrl = text("base_url", validateBaseUrl);
    const organization = text("organization", (value) => validateOrganization(value, "organization"));
    const project = text("project", (value) => validateOrganization(value, "project"));
    const headers = readHeaders(own(data, "headers"), issues);
    const blocks = new Map<string, JsonObject>();
    for (const key of SUPPORTED_LLM_PROVIDERS) {
      const value = own(data, key);
      if (value === null || value === undefined) continue;
      const checked = checkBlock(blockSchema(key), value);
      issues.push(...checked.issues.map((issue) => ({ ...issue, loc: [key, ...issue.loc] })));
      if (checked.value !== null) blocks.set(key, checked.value);
    }
    for (const [key, value] of Object.entries(data)) {
      if (value !== undefined && !LLM_CONFIG_KEYS.has(key)) {
        issues.push({ loc: [key], type: "extra_forbidden", msg: "Extra inputs are not permitted" });
      }
    }
    if (issues.length > 0) throw new ValidationError("LLMConfig", issues);

    try {
      bindConnectionToKey({ provider, apiKey, baseUrl, organization, project, headers });
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      throw new ValidationError("LLMConfig", [valueError([], error.message)]);
    }
    this.provider = provider;
    this.model = model;
    this.baseUrl = baseUrl;
    this.organization = organization;
    this.project = project;
    this.#apiKey = apiKey;
    this.#headers = headers;
    this.#blocks = blocks;
    Object.freeze(this);
  }

  /** The host-supplied provider key, or `null`. */
  get apiKey(): string | null {
    return this.#apiKey;
  }

  /** A frozen copy of the extra request headers, or `null`. */
  get headers(): Readonly<Record<string, string>> | null {
    return this.#headers === null ? null : Object.freeze({ ...this.#headers });
  }

  /**
   * Return the block named `provider` (as written in `llm_config`, e.g. `"qwen-cn"`) as a fresh
   * record without unset fields; `{}` when the block is unset. Throws `RangeError` if `provider`
   * has no block.
   *
   * Example: `new LLMConfig({"glm-cn": {do_sample: false}}).blockFor("glm-cn")` gives
   * `{do_sample: false}`.
   */
  blockFor(provider: string): JsonObject {
    blockSchema(provider);
    const block = this.#blocks.get(provider);
    return block === undefined ? {} : structuredClone(block);
  }

  /**
   * Return the config as a fresh record, Python's `model_dump()`.
   *
   * `provider` and `model` are always there (`null` when unset); `base_url`, `organization` and
   * `project` only when set; then each set block by its authored name, in
   * `SUPPORTED_LLM_PROVIDERS` order. `api_key` and `headers` are never there.
   */
  dump(): JsonObject {
    const dumped: JsonObject = { provider: this.provider, model: this.model };
    if (this.baseUrl !== null) dumped["base_url"] = this.baseUrl;
    if (this.organization !== null) dumped["organization"] = this.organization;
    if (this.project !== null) dumped["project"] = this.project;
    for (const [key, block] of this.#blocks) dumped[key] = structuredClone(block);
    return dumped;
  }
}
