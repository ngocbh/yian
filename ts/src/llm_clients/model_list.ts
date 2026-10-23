/**
 * List the model ids a provider's host offers, for model pickers and key checks.
 *
 * A listing uses only the key, base URL and connection settings passed in, never values from the
 * environment, so a host can try a key before it saves it. Providers echo rejected keys and
 * headers in error bodies, so every failure throws a `ModelListError` with a short fixed reason
 * and no `cause`.
 */

import { current } from "../defaults.ts";
import { compareCodeUnits } from "../diagnostics.ts";
import { ProviderError } from "../llm/errors.ts";
import { realScheduler, type Scheduler } from "../platform.ts";
import { isRecord } from "../typesys/index.ts";
import { HttpResponse, isVisibleText } from "./auth/common.ts";
import { request as httpRequest } from "./auth/http.ts";
import { validateBaseUrl } from "./config.ts";
import { OPENAI_API_ENDPOINT } from "./connection.ts";
import { openaiIdHeaders, validateHeaders } from "./connection_settings.ts";
import { OPENAI_COMPATIBLE, PROVIDER_BASE_URL } from "./factory.ts";

const ANTHROPIC_BASE_URL = "https://api.anthropic.com";
const ANTHROPIC_VERSION = "2023-06-01";
const GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com";
const OLLAMA_BASE_URL = "http://localhost:11434";
// The largest page Anthropic and Google serve, so most listings take one request.
const PAGE_SIZE = "1000";

/** Seconds a listing may take when the caller gives no `timeout`. */
export const MODEL_LIST_TIMEOUT_SECONDS = 8;

/**
 * A model listing failed.
 *
 * The message is one of `HTTP <status>`, `can't reach host`, `timed out`, `invalid request`,
 * `invalid response`, or `unsupported provider`; it never holds the key, a header value, the URL,
 * or the response body. `status` is the HTTP status of an `HTTP <status>` failure, else `null`;
 * hosts read it rather than the message. A status of 401 or 403 makes `isAuthFailure` count the
 * error.
 */
export class ModelListError extends Error {
  override name = "ModelListError";
  readonly status: number | null;

  constructor(message: string, options: { status?: number } = {}) {
    // No options reach `super`, so the error never has a `cause` key.
    super(message);
    this.status = options.status ?? null;
  }
}

/** Options for `listModels`. */
export interface ListModelsOptions {
  /** Key sent as the provider's auth header; absent sends none. Ollama never sends one. */
  apiKey?: string | null;
  /**
   * Host to list instead of the provider default. For Anthropic and Google it is the API root
   * without the version path. For OpenAI-compatible providers it is the URL `/models` is appended
   * to. A trailing `/v1` on an Ollama URL is dropped. When a key is sent it must pass
   * `validateBaseUrl`.
   */
  baseUrl?: string | null;
  /** Seconds for the whole listing, shared by every page; must be positive. Defaults to 8. */
  timeout?: number;
  /** Extra headers sent beside the key, checked by `validateHeaders`; sent only with a key. */
  headers?: Readonly<Record<string, string>> | null;
  /** OpenAI organization id, sent as `OpenAI-Organization`; only for `openai` with a key. */
  organization?: string | null;
  /** OpenAI project id, sent as `OpenAI-Project`; only for `openai` with a key. */
  project?: string | null;
  /** Stops the listing; a stop rejects with `signal.reason`. */
  signal?: AbortSignal;
  /** The clock the deadline runs on; defaults to `realScheduler`. */
  scheduler?: Scheduler;
  /** Sends one GET; defaults to `auth/http.ts`'s `request`, which refuses redirects. */
  request?: typeof httpRequest;
}

/** Get one page: GET `path` with `params` and return its JSON object. */
type GetPage = (path: string, params: Record<string, string>) => Promise<Record<string, unknown>>;

/** Return each page's model ids, unchecked, following every page. */
type Pages = (getPage: GetPage) => Promise<unknown[]>;

/**
 * Return the model ids the provider's host offers, sorted (code-unit order) and without
 * duplicates, or `null` for `azure`, which has no model listing.
 *
 * `provider` is case-insensitive: `anthropic`, `google`, `ollama`, `azure`, or any provider routed
 * through the OpenAI client. Anthropic lists `{base}/v1/models` and Google `{base}/v1beta/models`,
 * both following every page; Google keeps only models that support `generateContent`, without
 * the `models/` prefix. OpenAI-compatible providers list `{base}/models`, Ollama its local tags at
 * `{base}/api/tags`. Redirects are refused by the default `request`, so the key and connection
 * settings go only to the host asked.
 *
 * Throws `ModelListError` for an unsupported provider, an invalid key, URL, timeout or connection
 * setting (or one given without a key), a non-2xx status, an unreachable host, an exhausted
 * timeout, or a response that is not a model list. A caller abort rejects with `signal.reason`.
 */
export async function listModels(
  provider: unknown,
  options: ListModelsOptions = {},
): Promise<readonly string[] | null> {
  const { apiKey, baseUrl, timeout = MODEL_LIST_TIMEOUT_SECONDS, organization, project, signal, scheduler = realScheduler } = options;
  const name = typeof provider === "string" ? provider.toLowerCase() : "";
  if (name === "azure") return null;
  let base: unknown;
  let pages: Pages;
  const protocol: Record<string, string> = {};
  let auth: ((key: string) => Record<string, string>) | null;
  if (name === "anthropic") {
    base = baseUrl || ANTHROPIC_BASE_URL;
    pages = anthropicIds;
    protocol["anthropic-version"] = ANTHROPIC_VERSION;
    auth = (key) => ({ "x-api-key": key });
  } else if (name === "google") {
    base = baseUrl || GOOGLE_BASE_URL;
    pages = googleIds;
    auth = (key) => ({ "x-goog-api-key": key });
  } else if (name === "ollama") {
    base = baseUrl || OLLAMA_BASE_URL;
    pages = ollamaIds;
    auth = null;
  } else if (OPENAI_COMPATIBLE.includes(name)) {
    // Only openai itself defaults to OpenAI's host; a routed provider with no endpoint of its own
    // must not send its key there.
    const own = Object.hasOwn(PROVIDER_BASE_URL, name) ? PROVIDER_BASE_URL[name] : undefined;
    base = baseUrl || (name === "openai" ? OPENAI_API_ENDPOINT : own);
    if (!base) throw new ModelListError("unsupported provider");
    pages = openaiIds;
    auth = (key) => ({ Authorization: `Bearer ${key}` });
  } else {
    throw new ModelListError("unsupported provider");
  }

  const sendsKey = Boolean(apiKey) && auth !== null;
  if (!validRequest(base, apiKey, timeout, sendsKey && baseUrl !== null && baseUrl !== undefined)) {
    throw new ModelListError("invalid request");
  }
  const settings = connectionHeaders(name, sendsKey, options.headers, organization, project);
  if (settings === null) throw new ModelListError("invalid request");
  // The same test as `sendsKey`, written so TypeScript narrows `auth` and `apiKey`.
  const keyHeaders = auth && apiKey ? auth(apiKey) : {};
  const fixed = { ...protocol, ...keyHeaders };
  // The protocol and key headers win a name clash; dropping the other spelling keeps the client
  // from joining both values into one header.
  const taken = new Set(Object.keys(fixed).map((key) => key.toLowerCase()));
  const kept = Object.entries(settings).filter(([key]) => !taken.has(key.toLowerCase()));
  const headers = { ...Object.fromEntries(kept), ...fixed };
  let root = (base as string).replace(/\/+$/, "");
  if (name === "ollama" && root.endsWith("/v1")) root = root.slice(0, -"/v1".length);

  const send = options.request ?? httpRequest;
  const bodyLimit = current().maxModelListBytes;
  const deadline = scheduler.now() + timeout * 1000;
  const getPage: GetPage = async (path, params) => {
    const timeoutMs = deadline - scheduler.now();
    if (timeoutMs <= 0) throw new ModelListError("timed out");
    const query = new URLSearchParams(params).toString();
    const url = root + path + (query ? `?${query}` : "");
    let response: unknown;
    try {
      response = await send("GET", url, { headers, timeoutMs, bodyLimit, signal, scheduler });
    } catch (error) {
      if (signal?.aborted) throw signal.reason;
      if (error instanceof ProviderError && error.kind === "timeout") throw new ModelListError("timed out");
      if (error instanceof ProviderError && error.kind === "protocol") throw new ModelListError("invalid response");
      throw new ModelListError("can't reach host");
    }
    if (!(response instanceof HttpResponse) || !Number.isInteger(response.status)) {
      throw new ModelListError("invalid response");
    }
    if (!(response.status >= 200 && response.status < 300)) {
      throw new ModelListError(`HTTP ${response.status}`, { status: response.status });
    }
    const data = response.data;
    if (!isRecord(data)) throw new ModelListError("invalid response");
    return data;
  };
  const ids = await pages(getPage);
  if (!ids.every((id) => typeof id === "string")) throw new ModelListError("invalid response");
  return Object.freeze([...new Set(ids as string[])].sort(compareCodeUnits));
}

/** Whether the URL, key and timeout can make a request; `checkUrl` holds `base` to `validateBaseUrl`. */
function validRequest(base: unknown, apiKey: unknown, timeout: unknown, checkUrl: boolean): boolean {
  if (typeof timeout !== "number" || !Number.isFinite(timeout) || timeout <= 0) return false;
  if ((apiKey !== null && apiKey !== undefined && typeof apiKey !== "string") || typeof base !== "string") {
    return false;
  }
  // A header value the client would refuse, with the key in its message.
  if (apiKey && !isVisibleText(apiKey)) return false;
  if (checkUrl) {
    // The key goes to this host, so hold it to the rules for a config base_url.
    try {
      validateBaseUrl(base);
    } catch (error) {
      if (error instanceof RangeError) return false;
      throw error;
    }
  }
  // Stands in for Python's urlsplit check: the URL parses, its scheme is http(s), and it names a host.
  return URL.canParse(base) && /^https?:\/\/[^/?#]/i.test(base);
}

/**
 * Return the connection settings as request headers, or `null` if they are refused.
 *
 * `{}` when no setting is given; `null` when a setting is refused, is given without a key, or
 * gives ids to a provider other than `openai`. The validators' messages are dropped: the caller
 * throws a fixed reason, so no header value or id can reach the error.
 */
function connectionHeaders(
  provider: string,
  sendsKey: boolean,
  headers: unknown,
  organization: unknown,
  project: unknown,
): Record<string, string> | null {
  const noHeaders =
    headers === null || headers === undefined || (isRecord(headers) && Object.keys(headers).length === 0);
  const noIds = (organization === null || organization === undefined) && (project === null || project === undefined);
  if (noHeaders && noIds) return {};
  if (!sendsKey || (provider !== "openai" && !noIds)) return null;
  try {
    return { ...validateHeaders(headers ?? {}, provider), ...openaiIdHeaders(organization, project) };
  } catch (error) {
    if (error instanceof RangeError) return null;
    throw error;
  }
}

/** Return `value[key]`, throwing `invalid response` unless `value` is a record that has `key`. */
function field(value: unknown, key: string): unknown {
  if (!isRecord(value) || !Object.hasOwn(value, key)) throw new ModelListError("invalid response");
  return value[key];
}

/** Return `value` if it is an array, else throw `invalid response`. */
function list(value: unknown): unknown[] {
  if (!Array.isArray(value)) throw new ModelListError("invalid response");
  return value;
}

/** Return `value` if it is a string, else throw `invalid response`. */
function text(value: unknown): string {
  if (typeof value !== "string") throw new ModelListError("invalid response");
  return value;
}

/**
 * Return model ids from every page of Anthropic's model listing.
 *
 * `last_id` must be a string, stricter than Python, whose `urlencode` takes any value: the JSON
 * decoder here turns integers into bigints.
 */
async function anthropicIds(getPage: GetPage): Promise<unknown[]> {
  const ids: unknown[] = [];
  let params: Record<string, string> = { limit: PAGE_SIZE };
  for (;;) {
    const page = await getPage("/v1/models", params);
    ids.push(...list(field(page, "data")).map((model) => field(model, "id")));
    if (!page["has_more"]) return ids;
    params = { limit: PAGE_SIZE, after_id: text(field(page, "last_id")) };
  }
}

/**
 * Return the `generateContent` model ids from every page of Google's listing.
 *
 * `nextPageToken` must be a string and `supportedGenerationMethods` an array, stricter than
 * Python, which takes any token `urlencode` accepts and any container `in` works on: the JSON
 * decoder here turns integers into bigints.
 */
async function googleIds(getPage: GetPage): Promise<unknown[]> {
  const ids: unknown[] = [];
  let params: Record<string, string> = { pageSize: PAGE_SIZE };
  for (;;) {
    const page = await getPage("/v1beta/models", params);
    for (const model of Object.hasOwn(page, "models") ? list(page["models"]) : []) {
      if (!isRecord(model)) throw new ModelListError("invalid response");
      const methods = Object.hasOwn(model, "supportedGenerationMethods")
        ? list(model["supportedGenerationMethods"])
        : [];
      if (methods.includes("generateContent")) ids.push(text(field(model, "name")).replace(/^models\//, ""));
    }
    const token = page["nextPageToken"];
    if (!token) return ids;
    params = { pageSize: PAGE_SIZE, pageToken: text(token) };
  }
}

/** Return model ids from an OpenAI-compatible `/models` listing. */
async function openaiIds(getPage: GetPage): Promise<unknown[]> {
  return list(field(await getPage("/models", {}), "data")).map((model) => field(model, "id"));
}

/** Return the local model tags from Ollama's native API. */
async function ollamaIds(getPage: GetPage): Promise<unknown[]> {
  return list(field(await getPage("/api/tags", {}), "models")).map((model) => field(model, "name"));
}
