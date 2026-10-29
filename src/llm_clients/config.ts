/**
 * `LLMConfig`: model selection for an AGENT node.
 *
 * Lives with the clients that consume it: the AGENT node and the Compose loader's `llm_config:`
 * field carry an `LLMConfig` (or the plain record that normalizes to one), and the factory turns
 * API selections into ready chat models. `validateBaseUrl` checks a configured endpoint, and
 * `splitOllamaBaseUrl` checks an `OLLAMA_BASE_URL` value and splits off its credentials.
 */

import { isIPv6 } from "node:net";

import { current } from "../defaults.ts";
import { reprText } from "../diagnostics.ts";
import { isRecord } from "../typesys/index.ts";
import type { JsonObject } from "../typesys/json.ts";
import { fromHost as hostValue } from "../typesys/values.ts";
import { validateHeaderName, validateHeaders, validateOrganization } from "./connection_settings.ts";
import { PROVIDER_BASE_URL, prefersResponses } from "./endpoints.ts";
import { blockSchema, checkBlock, choices, type Issue, PROVIDER_BLOCKS, ValidationError } from "./provider_blocks.ts";

/**
 * Every provider an `llm_config` may name; the validated config carrier and the factory that
 * consumes it share this one list.
 */
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

/** The error for an Azure call without its resource endpoint. */
export const AZURE_NEEDS_BASE_URL =
  "provider azure needs llm_config.base_url, the Azure OpenAI resource endpoint " +
  "(https://<resource>.openai.azure.com/); set llm_config.base_url beside " +
  "provider: azure. AZURE_OPENAI_ENDPOINT is not read";

/**
 * Return why `provider`'s `block` can't be sent with `model`, or `null`.
 *
 * A Chat Completions vendor (any OpenAI-compatible provider but `openai`) or Azure sends a model
 * that `prefersResponses` to the Responses API, while its block mirrors the Chat Completions
 * request, so such a model needs the block empty. A block that is not a record is left to the
 * block's own check.
 */
export function responsesBlockError(provider: string, model: string, block: unknown): string | null {
  const chat = provider === "azure" || Object.hasOwn(PROVIDER_BASE_URL, provider);
  if (!chat || !prefersResponses(model) || !isRecord(block) || Object.keys(block).length === 0) return null;
  return (
    `The ${reprText(provider)} block mirrors the Chat Completions request, but model ${reprText(model)} ` +
    "is routed to the Responses API; remove the block or use a model served through Chat Completions."
  );
}

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

const PROVIDER_CHOICES = `Input should be ${choices(SUPPORTED_LLM_PROVIDERS)}`;

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
 * `temperature` message names the layer's `provider` when it is a supported one. Returns `null`
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

// LangChain's test for a value that already names its scheme; anything else holding a ":" gets
// "http://" prefixed, as `parse_url_with_auth` does.
const HAS_SCHEME = /^[A-Za-z][A-Za-z0-9+.-]*:\/\//;
// The userinfo is greedy and the host holds no "@", so the userinfo runs to the last "@" of the
// authority, as `urlparse` splits it.
const OLLAMA_URL =
  /^(?<scheme>https?):\/\/(?:(?<userinfo>[^/?#\x00-\x20\x7f]*)@)?(?<host>[A-Za-z0-9._-]+|\[[0-9A-Fa-f:.]+\])(?::(?<port>[0-9]*))?(?<path>\/[^?#\x00-\x20\x7f]*)?$/i;
const OLLAMA_SHAPE =
  "OLLAMA_BASE_URL must be an http:// or https:// URL of a host, an optional port, and a path, " +
  "without a query or fragment";
// A decoded leading BOM stays in the text.
const UTF8 = new TextDecoder("utf-8", { ignoreBOM: true });

/** Decode each run of `%XX` escapes in `text` as UTF-8; invalid escapes stay as written. */
function unquote(text: string): string {
  return text.replace(/(?:%[0-9A-Fa-f]{2})+/g, (run) =>
    UTF8.decode(Uint8Array.from(run.slice(1).split("%"), (hex) => parseInt(hex, 16))),
  );
}

/**
 * Split an `OLLAMA_BASE_URL` value into the base URL Ollama is called at and its Basic auth
 * credentials, rebuilding the URL as LangChain and the ollama SDK do.
 *
 * A value with no scheme that contains `:` gets `http://` prefixed. The value must then be an
 * `http://` or `https://` URL (any case) of a host (letters, digits, `.`, `_`, `-`, or a bracketed
 * IPv6 address), an optional port from 1 to 65535, and a path, with no query, fragment,
 * whitespace or control characters. The userinfo, split at its first `:` and percent-decoded,
 * gives `[user, password]` when the user is non-empty, else `null`. The rebuilt URL drops the
 * userinfo; lowercases the scheme and host; keeps a non-empty port as an integer, or adds `:11434`
 * when the scheme is not all lower case and there are no credentials; and strips trailing `/`
 * from the path, then a final `/v1`, then `/` at both ends. Throws `RangeError` if `value` is not
 * such a URL; messages never echo the value, which may hold a password.
 */
export function splitOllamaBaseUrl(value: unknown): [baseUrl: string, credentials: [string, string] | null] {
  if (typeof value !== "string") throw new RangeError(OLLAMA_SHAPE);
  const text = !HAS_SCHEME.test(value) && value.includes(":") ? `http://${value}` : value;
  const groups = OLLAMA_URL.exec(text)?.groups;
  if (groups === undefined) throw new RangeError(OLLAMA_SHAPE);
  // The pattern always matches `scheme` and `host`; the defaults only satisfy the group types.
  const { scheme = "", userinfo, host = "", port, path = "" } = groups;
  if (port && !(Number(port) >= 1 && Number(port) <= 65535)) {
    throw new RangeError("OLLAMA_BASE_URL port must be a number from 1 to 65535");
  }
  let credentials: [string, string] | null = null;
  if (userinfo !== undefined) {
    const colon = userinfo.indexOf(":");
    const user = colon < 0 ? userinfo : userinfo.slice(0, colon);
    if (user) credentials = [unquote(user), unquote(colon < 0 ? "" : userinfo.slice(colon + 1))];
  }
  // The ollama SDK gives 80/443 only to an exact "http"/"https" and 11434 to anything else, and
  // LangChain lowercases the scheme only when it rebuilds the URL, which it does for credentials.
  let suffix = port ? `:${Number(port)}` : "";
  if (!port && scheme !== scheme.toLowerCase() && credentials === null) suffix = ":11434";
  let trimmed = path.replace(/\/+$/, "");
  if (trimmed.endsWith("/v1")) trimmed = trimmed.slice(0, -3);
  trimmed = trimmed.replace(/^\/+|\/+$/g, "");
  if (trimmed) suffix += `/${trimmed}`;
  return [`${scheme.toLowerCase()}://${host.toLowerCase()}${suffix}`, credentials];
}

/**
 * Return why `data` uses a removed `llm_config` form, or `null`.
 *
 * Checks, first match wins: an underscore spelling of a dash block (`qwen_cn` for `qwen-cn`), a
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
function bindConnectionToKey(connection: Connection): void {
  if (connection.baseUrl !== null && connection.provider !== "azure") {
    if (connection.apiKey === null || connection.provider === null) {
      throw new RangeError(
        "llm_config.base_url needs llm_config.api_key and llm_config.provider in the same config, " +
          "since it decides where that key is sent",
      );
    }
    // The native Ollama client sends no key, so it cannot take a host base_url.
    if (connection.provider === "ollama") {
      throw new RangeError(
        "provider ollama does not take llm_config.base_url; it sends no api_key, so set OLLAMA_BASE_URL instead",
      );
    }
  }
  if (connection.provider !== null) {
    if ((connection.organization !== null || connection.project !== null) && connection.provider !== "openai") {
      throw new RangeError("llm_config.organization and llm_config.project apply only to provider openai");
    }
    if (connection.headers !== null && connection.provider === "ollama") {
      throw new RangeError("provider ollama does not take llm_config.headers; it sends no api_key");
    }
    // The field check could not see the provider; openai refuses more names.
    for (const name of Object.keys(connection.headers ?? {})) validateHeaderName(name, connection.provider);
  }
  for (const field of ["organization", "project", "headers"] as const) {
    if (connection[field] !== null && (connection.apiKey === null || connection.provider === null)) {
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
 * ...) fails with its new form (see `retiredKeyError`). Validation errors never hold an input
 * value, so a rejected config cannot leak `api_key`; they name record keys as written, at any depth
 * (an unknown key or block field, a header or `metadata` name). Instances are frozen.
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

  /**
   * Validate a host `llm_config` (JSON-shaped: a whole number becomes an int), converting values
   * with the host intake `runFlow` uses. Throws `ValidationError`, or `TypeCheckError` (a
   * `RangeError`) for a value that isn't a yian value, such as NaN or text with a lone surrogate,
   * which `JSON.parse` can produce.
   */
  static fromHost(data: unknown): LLMConfig {
    return new LLMConfig(hostValue(data, "llm_config"));
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
    // Refuses a name with no block.
    blockSchema(provider);
    const block = this.#blocks.get(provider);
    return block === undefined ? {} : structuredClone(block);
  }

  /**
   * Return the config as a fresh record.
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

/** Return `url` without its trailing slashes. */
function stripTrailingSlashes(url: string): string {
  // A loop, not `/\/+$/`: that regex backtracks quadratically on a long run of inner slashes.
  let end = url.length;
  while (end > 0 && url[end - 1] === "/") end -= 1;
  return url.slice(0, end);
}

/**
 * Return the base Azure resource URL, without an OpenAI-v1 suffix or a trailing slash.
 *
 * Some Azure portals hand out the `.../openai/v1` (OpenAI-compatible) surface, but the Azure
 * client appends its own `/openai/deployments/...` path, so that suffix yields a 404 ("Resource
 * not found"). Strips one `/openai/v1` or `/openai` suffix, and the slashes around it.
 */
export function azureResourceEndpoint(url: string): string {
  let base = stripTrailingSlashes(url);
  for (const suffix of ["/openai/v1", "/openai"]) {
    if (base.endsWith(suffix)) {
      base = base.slice(0, -suffix.length);
      break;
    }
  }
  return stripTrailingSlashes(base);
}

// The connection settings `popHostCredentials` takes with the host key.
const CONNECTION_SETTINGS = ["organization", "project", "headers"] as const;

// Fields only the host's run layer may set. Authored layers reach the recorded effective map, a
// flow-level base_url could send the host key elsewhere, and connection settings go wherever that
// key goes. Each message starts with `llm_config.`: the AGENT load check swaps that for its own
// prefix.
const HOST_ONLY_FIELDS: ReadonlyMap<string, string> = new Map([
  ["api_key", "llm_config.api_key is a host credential; pass it in the run's llm_config, not in a flow file"],
  [
    "base_url",
    "llm_config.base_url decides where the host's api_key is sent; pass it with that key in the run's " +
      "llm_config, not in a flow file. A flow file sets base_url only beside provider: azure, as the Azure " +
      "resource endpoint",
  ],
  ...CONNECTION_SETTINGS.map(
    (field) =>
      [
        field,
        `llm_config.${field} goes wherever the host's api_key is sent; pass it with that key in the run's ` +
          "llm_config, or save it with the login by `yian auth`, not in a flow file",
      ] as const,
  ),
]);

/**
 * Return the first host-only field an authored `llm_config` layer sets and why it may not, or
 * `null`.
 *
 * The fields are `api_key`, `base_url`, `organization`, `project` and `headers`; a `null` value
 * counts. A `base_url` beside `provider: azure` is allowed: it is the Azure resource endpoint,
 * and the Azure key comes from the environment, which flow code can already read, while a host
 * `api_key` for Azure must bring its own `base_url` (see `popHostCredentials`). A non-record has no
 * host-only field.
 */
export function hostOnlyFieldError(config: unknown): readonly [field: string, message: string] | null {
  if (!isRecord(config)) return null;
  for (const [field, message] of HOST_ONLY_FIELDS) {
    if (field === "base_url" && own(config, "provider") === "azure") continue;
    if (has(config, field)) return [field, message];
  }
  return null;
}

/**
 * Whether `value` counts as empty: unset, `false`, `""`, zero, or an empty list or record
 * (a record whose values are all `undefined` counts as empty).
 */
function isFalsy(value: unknown): boolean {
  if (value === undefined || value === null || value === false || value === "" || value === 0n || value === 0) {
    return true;
  }
  if (Array.isArray(value)) return value.length === 0;
  return isRecord(value) && Object.values(value).every((item) => item === undefined);
}

/**
 * Remove a host `api_key` from one mutable config layer and return it validated, or `null` when the
 * layer has none.
 *
 * The layer must name its `provider`, so the key authenticates only that provider. Errors never
 * echo the key. Throws `RangeError` if `provider` is missing, or `ValidationError` if the key is
 * empty or not a string or `provider` is unsupported.
 */
export function popApiKey(config: Record<string, unknown>): string | null {
  const raw = own(config, "api_key");
  delete config["api_key"];
  if (raw === null || raw === undefined) return null;
  const provider = own(config, "provider");
  if (isFalsy(provider)) {
    throw new RangeError("llm_config.api_key needs llm_config.provider, so the key goes only to that provider");
  }
  return new LLMConfig({ provider, api_key: raw }).apiKey;
}

/**
 * The credential and connection settings taken from a host `llm_config` layer. Instances are
 * frozen; the key and header values never show in `JSON.stringify` or `inspect`.
 *
 * Fields: `apiKey` (the host key), `baseUrl` (where the key is sent), `organization` and
 * `project` (OpenAI ids sent with the key), and `headers` (extra request headers sent with the
 * key; values are secrets). Each is `null` when unset.
 */
export class HostConnection {
  readonly baseUrl: string | null;
  readonly organization: string | null;
  readonly project: string | null;
  // Private, so JSON.stringify, inspect and spreads never show the secrets.
  readonly #apiKey: string | null;
  readonly #headers: Readonly<Record<string, string>> | null;

  constructor(
    fields: {
      apiKey?: string | null;
      baseUrl?: string | null;
      organization?: string | null;
      project?: string | null;
      headers?: Readonly<Record<string, string>> | null;
    } = {},
  ) {
    this.baseUrl = fields.baseUrl ?? null;
    this.organization = fields.organization ?? null;
    this.project = fields.project ?? null;
    this.#apiKey = fields.apiKey ?? null;
    this.#headers = fields.headers ? Object.freeze({ ...fields.headers }) : null;
    Object.freeze(this);
  }

  /** The host key, or `null`. */
  get apiKey(): string | null {
    return this.#apiKey;
  }

  /** A frozen copy of the extra request headers, or `null`. */
  get headers(): Readonly<Record<string, string>> | null {
    return this.#headers === null ? null : Object.freeze({ ...this.#headers });
  }
}

/**
 * Remove a host `api_key` and the connection it goes with from one mutable config layer.
 *
 * Taking them from the same layer is what binds the connection to the key: a base_url decides
 * where the key is sent, and the organization, project and headers go wherever it is sent, so each
 * is accepted only beside that key and the layer's `provider`. For `provider: azure` without a
 * key, `base_url` is the resource endpoint the environment key goes to; it stays in the layer as an
 * ordinary setting. An Azure key needs a `base_url` beside it, so it is never sent to an endpoint
 * another layer names. `api_key`, `organization`, `project` and `headers` are removed even when
 * `null`, and `base_url` unless it is a keyless Azure endpoint. Errors never echo the URL, a
 * header value or the key.
 *
 * Returns the validated key and settings, each `null` when the layer has none (an empty `headers`
 * record too). Throws `RangeError` if the key lacks `provider` (see `popApiKey`), an Azure key has
 * no `base_url`, or a keyless Azure `base_url` fails `validateBaseUrl`; throws `ValidationError`
 * if the key is invalid, the layer has a non-Azure `base_url` or a connection setting but no
 * `api_key`, or a value is not allowed (see `LLMConfig`).
 */
export function popHostCredentials(config: Record<string, unknown>): HostConnection {
  const apiKey = popApiKey(config);
  const settings: Record<string, unknown> = {};
  for (const field of CONNECTION_SETTINGS) {
    const value = own(config, field);
    delete config[field];
    if (value !== null && value !== undefined) settings[field] = value;
  }
  const provider = own(config, "provider");
  if (apiKey === null && provider === "azure") {
    const endpoint = own(config, "base_url");
    if (endpoint !== null && endpoint !== undefined) validateBaseUrl(endpoint);
    // Validation names the first setting that needs a key.
    if (Object.keys(settings).length > 0) new LLMConfig({ provider, ...settings });
    return new HostConnection();
  }
  const baseUrl = own(config, "base_url");
  delete config["base_url"];
  const noBaseUrl = baseUrl === null || baseUrl === undefined;
  if (noBaseUrl && apiKey !== null && provider === "azure") {
    throw new RangeError(
      "llm_config.api_key for provider azure needs llm_config.base_url in the same config, " +
        "since it decides where that key is sent",
    );
  }
  if (apiKey === null && noBaseUrl && Object.keys(settings).length === 0) return new HostConnection();
  const validated = new LLMConfig({ provider, api_key: apiKey, base_url: baseUrl, ...settings });
  return new HostConnection({
    apiKey: validated.apiKey,
    baseUrl: validated.baseUrl,
    organization: validated.organization,
    project: validated.project,
    headers: validated.headers,
  });
}

/** Return a provider block record without its unset fields; any other value unchanged. */
function unsetDropped(key: string, value: unknown): unknown {
  if (Object.hasOwn(PROVIDER_BLOCKS, key) && isRecord(value)) {
    return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== null && item !== undefined));
  }
  return value;
}

/**
 * Merge two fill-the-gap layers into a new record, without mutating either input.
 *
 * A set value in `specific` (the higher-precedence layer, e.g. a node's own `llm_config`) wins;
 * `base` (e.g. the enclosing flow's) fills every remaining field. `null` and `undefined` mean
 * unset at every layer, matching the optional `LLMConfig` fields' fallback. A provider block (a key
 * of `PROVIDER_BLOCKS`, e.g. `openai`) merges per block field when both layers set it to a record:
 * `specific`'s set fields win, `base` fills the rest, and a value inside a field (e.g.
 * `thinking: {...}`) is replaced whole. A block that is not a record is kept over the other
 * layer's block, whichever layer sets it (the `specific` one when both do), so it reaches
 * validation instead of being hidden. Every other key, `_connection` included, is replaced whole.
 * Dropping unset values keeps the merge associative, so a cascade can be grouped without changing
 * its result. Block records in the result are new; other values are shared with the inputs.
 *
 * Example: `mergeLlmConfig({anthropic: {output_config: {effort: "high"}}}, {provider: "anthropic",
 * anthropic: {max_tokens: 8000n, output_config: {effort: "low"}}})` gives `{provider: "anthropic",
 * anthropic: {max_tokens: 8000n, output_config: {effort: "high"}}}`.
 */
export function mergeLlmConfig(
  specific: Readonly<Record<string, unknown>>,
  base: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  // A Map, then Object.fromEntries, so keys such as `__proto__` stay own data keys.
  const out = new Map<string, unknown>();
  for (const [key, value] of Object.entries(base)) {
    if (value !== null && value !== undefined) out.set(key, unsetDropped(key, value));
  }
  for (const [key, value] of Object.entries(specific)) {
    if (value === null || value === undefined) continue;
    const existing = out.get(key);
    if (Object.hasOwn(PROVIDER_BLOCKS, key) && existing !== undefined) {
      if (!isRecord(value)) out.set(key, value);
      else if (isRecord(existing)) out.set(key, { ...existing, ...(unsetDropped(key, value) as object) });
      // Otherwise the base block is not a record and stays for validation to refuse.
    } else {
      out.set(key, unsetDropped(key, value));
    }
  }
  return Object.fromEntries(out);
}
