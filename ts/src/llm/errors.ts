/**
 * Provider errors: one error class, a kind per failure, and openai's retry policy.
 *
 * `errorFromResponse` turns a non-2xx HTTP reply into a `ProviderError` whose kind comes
 * from the status (and, for quota, content filters, overload and context overflow, the
 * body) and whose `retryable` flag follows the openai SDK's `_should_retry`.
 * `parseRetryAfter` reads the server's requested delay the way that SDK does.
 * `scrubError` rewrites an error, its causes and its group members with a text redactor,
 * so a credential a provider echoed never reaches a log or a trace.
 */

import { redirectDetail } from "../platform.ts";
import { parseJson } from "../typesys/json.ts";

/** What went wrong with a model call, independent of the provider. */
export const ERROR_KINDS = Object.freeze([
  "auth",
  "permission",
  "not_found",
  "rate_limit",
  "quota",
  "overloaded",
  "bad_request",
  "context_length",
  "refused",
  "server",
  "timeout",
  "connection",
  "redirect",
  "incomplete",
  "protocol",
] as const);

/** One of `ERROR_KINDS`. */
export type ErrorKind = (typeof ERROR_KINDS)[number];

/** The kinds a `ProviderError` is retryable for when the caller doesn't say. */
const RETRYABLE_KINDS: ReadonlySet<ErrorKind> = new Set(["rate_limit", "overloaded", "server", "timeout", "connection"]);

/** openai's MAX_RETRY_AFTER_DELAY: the longest retry-after, in seconds, that still allows a retry by default. */
export const MAX_RETRY_AFTER_SECONDS = 120;

// openai's MAX_RETRY_AFTER_DELAY + 1: an overflowing delay stays finite, so the default
// 120 s refusal still applies to it.
const OVERFLOW_RETRY_AFTER = 121;
const DECIMAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
const FLOAT_LITERAL = /^([+-]?)(inf|infinity|nan)$/i;
const CONTEXT_PHRASES = [
  "context_length_exceeded",
  "prompt is too long",
  "maximum context length",
  "exceeds the maximum number of tokens allowed",
  // Ollama's own prompt-too-long text, and llama-server's, which Ollama passes through.
  "longer than the context length",
  "exceeds the available context size",
  // Anthropic's "input length and `max_tokens` exceed context limit".
  "exceed context limit",
];
// OpenAI's insufficient_quota, and ChatGPT's usage-limit codes on its 429s.
const QUOTA_CODES = ["insufficient_quota", "usage_limit_reached", "usage_not_included"];
// OpenAI's and Azure's content-filter codes, and OpenRouter's moderation error type.
const CONTENT_FILTER_CODES = ["content_filter", "content_policy_violation", "ResponsibleAIPolicyViolation"];
const MAX_DETAIL_CHARS = 2000;

// RFC 5322 dates as HTTP sends them: `[Day, ]D[D] Mon YYYY HH:MM[:SS] zone`.
const HTTP_DATE =
  /^(?:[a-z]+,\s*|(?:mon|tue|wed|thu|fri|sat|sun)\s+)?(\d{1,2})\s+([a-z]{3})\s+(\d{4})\s+(\d{2}):(\d{2})(?::(\d{2}))?\s+(gmt|utc|ut|z|[+-]\d{4})$/i;
const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

/** Options for a `ProviderError`. */
export interface ProviderErrorOptions {
  /** The HTTP status, when the provider replied. */
  status?: number | null;
  /** Seconds the provider asked to wait, if any. */
  retryAfter?: number | null;
  /** Whether retrying the same request may succeed; defaults by kind. */
  retryable?: boolean;
  /** The provider name, for example `"openai"`. */
  provider?: string | null;
  /** The `name` of the transport error behind this one; never the error or a body. */
  causeName?: string | null;
}

/**
 * A model call failed. The message must already be scrubbed of credentials, since the
 * error's `stack` keeps the text it was built with; the error keeps no raw response body.
 * `retryable` defaults to true for `rate_limit`, `overloaded`, `server`, `timeout` and
 * `connection`, else false.
 */
export class ProviderError extends Error {
  override name = "ProviderError";
  /** What went wrong. */
  readonly kind: ErrorKind;
  /** The HTTP status, when the provider replied. */
  readonly status: number | null;
  /** Seconds the provider asked to wait, if any. */
  readonly retryAfter: number | null;
  /** Whether retrying the same request may succeed. */
  readonly retryable: boolean;
  /** The provider name; writable because protocol parsers don't know the route and the model sets it. */
  provider: string | null;
  /** The `name` of the transport error behind this one. */
  readonly causeName: string | null;

  constructor(kind: ErrorKind, message: string, options: ProviderErrorOptions = {}) {
    // Python's ErrorKind(kind) raises on an unknown kind too.
    if (!ERROR_KINDS.includes(kind)) throw new TypeError(`unknown provider error kind: ${String(kind)}`);
    super(message);
    this.kind = kind;
    this.status = options.status ?? null;
    this.retryAfter = options.retryAfter ?? null;
    this.retryable = options.retryable ?? RETRYABLE_KINDS.has(kind);
    this.provider = options.provider ?? null;
    this.causeName = options.causeName ?? null;
  }
}

/**
 * Return the delay in seconds a reply asks for before a retry, or `null` when the header
 * is absent or unparsable.
 *
 * `retry-after-ms` (milliseconds) is preferred, then `retry-after` in seconds; either may
 * be float text. A positive number too large for a float gives `121`, just over the
 * default 120 s limit, so a retry is refused by default. A literal `inf` or `nan` is
 * returned as is. Otherwise `retry-after` is read as an HTTP date, in seconds from `now`
 * (epoch seconds, default the current time), negative when the date has passed.
 */
export function parseRetryAfter(headers: Headers, now: number = Date.now() / 1000): number | null {
  for (const [name, divisor] of [
    ["retry-after-ms", 1000],
    ["retry-after", 1],
  ] as const) {
    const value = headers.get(name);
    if (value === null) continue;
    const text = value.trim();
    const literal = FLOAT_LITERAL.exec(text);
    if (literal) {
      const delay = literal[2]!.toLowerCase() === "nan" ? NaN : literal[1] === "-" ? -Infinity : Infinity;
      return delay / divisor;
    }
    if (!DECIMAL.test(text)) continue;
    const delay = Number(text);
    // Python checks only for +inf, so a negative overflow stays -Infinity.
    if (delay === Infinity) return OVERFLOW_RETRY_AFTER;
    return delay / divisor;
  }
  const value = headers.get("retry-after");
  if (value === null) return null;
  const when = parseHttpDate(value.trim());
  return when === null ? null : when - now;
}

/** Return the epoch seconds of an HTTP date, or `null` when it isn't one this parser reads. */
function parseHttpDate(text: string): number | null {
  const m = HTTP_DATE.exec(text);
  if (!m) return null;
  const month = MONTHS.indexOf(m[2]!.toLowerCase());
  if (month === -1) return null;
  const zone = m[7]!;
  let offset = 0;
  if (zone[0] === "+" || zone[0] === "-") {
    const sign = zone[0] === "-" ? -1 : 1;
    offset = sign * (Number(zone.slice(1, 3)) * 3600 + Number(zone.slice(3, 5)) * 60);
  }
  // Python's parsedate_tz rule, which Date.UTC would otherwise read as 19YY.
  let year = Number(m[3]);
  if (year < 100) year += year > 68 ? 1900 : 2000;
  // Out-of-range fields (day 32, hour 24, ...) roll over on purpose, as in Python's parsedate_tz/mktime_tz.
  const utc = Date.UTC(year, month, Number(m[1]), Number(m[4]), Number(m[5]), Number(m[6] ?? 0));
  return utc / 1000 - offset;
}

/**
 * Return the error kind for an HTTP `status` and, where it matters, its `body`:
 *
 * - A 4xx whose JSON `error.code`, `error.innererror.code` or `error.metadata.error_type`
 *   is "content_filter", "content_policy_violation" or "ResponsibleAIPolicyViolation":
 *   refused.
 * - 401 auth, 402 quota, 403 permission, 404 not found, 408 timeout.
 * - 429: quota when the body says "insufficient_quota", "usage_limit_reached" or
 *   "usage_not_included", else rate limit.
 * - 529, or a 503 whose body says "overloaded": overloaded.
 * - Any other 5xx: server.
 * - 3xx: redirect.
 * - A 400 naming `API_KEY_INVALID`: auth (Gemini's invalid key).
 * - A 400, 413 or 422 whose body names a context overflow ("context_length_exceeded",
 *   "prompt is too long", "maximum context length", Gemini's "exceeds the maximum
 *   number of tokens allowed", Ollama's "longer than the context length", or llama-server's
 *   "exceeds the available context size", or Anthropic's "exceed context limit"): context length.
 * - Any other 4xx: bad request.
 * - Anything else: protocol.
 */
export function kindForStatus(status: number, body: string): ErrorKind {
  if (status >= 400 && status < 500 && isContentFilter(body)) return "refused";
  const lowered = body.toLowerCase();
  switch (status) {
    case 401:
      return "auth";
    case 402:
      return "quota";
    case 403:
      return "permission";
    case 404:
      return "not_found";
    case 408:
      return "timeout";
    case 429:
      return QUOTA_CODES.some((code) => lowered.includes(code)) ? "quota" : "rate_limit";
    case 529:
      return "overloaded";
  }
  if (status === 503 && lowered.includes("overloaded")) return "overloaded";
  if (status >= 500) return "server";
  if (status >= 300 && status < 400) return "redirect";
  if (status === 400 && lowered.includes("api_key_invalid")) return "auth";
  if ([400, 413, 422].includes(status) && CONTEXT_PHRASES.some((phrase) => lowered.includes(phrase))) {
    return "context_length";
  }
  if (status >= 400 && status < 500) return "bad_request";
  // A 1xx or 2xx never reaches here from the executor; treat it as a broken reply.
  return "protocol";
}

/** Return whether a JSON error body names a content-filter code in one of the fields providers use. */
function isContentFilter(body: string): boolean {
  let data;
  try {
    data = parseJson(body, "plain");
  } catch {
    return false;
  }
  const error = field(data, "error");
  const codes = [
    field(error, "code"),
    field(field(error, "innererror"), "code"),
    field(field(error, "metadata"), "error_type"),
  ];
  // Exact strings only: OpenRouter's `error.code` is a number, and a code in message text doesn't count.
  return codes.some((code) => typeof code === "string" && CONTENT_FILTER_CODES.includes(code));
}

/** Return member `name` of `value` when `value` is a JSON object, else `undefined`. */
function field(value: unknown, name: string): unknown {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)[name]
    : undefined;
}

/** Options for `errorFromResponse`. */
export interface ErrorFromResponseOptions {
  /** The provider name, used in the message. */
  provider: string;
  /** The longest retry-after in seconds that still allows a retry; `null` allows any. Defaults to 120. */
  maxRetryAfter?: number | null;
  /** Current time in epoch seconds for an HTTP-date retry-after. */
  now?: number;
}

/**
 * Build the error for a non-2xx HTTP reply; `body` must already be stripped of
 * credentials. The kind comes from `kindForStatus`. `retryable` follows openai's
 * `_should_retry`: a finite retry-after over `maxRetryAfter` refuses, then
 * `x-should-retry` decides, then a quota error refuses, then 408, 409, 429 and 5xx retry.
 */
export function errorFromResponse(
  status: number,
  headers: Headers,
  body: string,
  { provider, maxRetryAfter = MAX_RETRY_AFTER_SECONDS, now }: ErrorFromResponseOptions,
): ProviderError {
  const kind = kindForStatus(status, body);
  const retryAfter = parseRetryAfter(headers, now);
  const detail = kind === "redirect" ? redirectDetail(headers.get("location")) : bodyDetail(body);
  return new ProviderError(kind, `${provider} returned HTTP ${status} (${kind}): ${detail}`, {
    status,
    retryAfter,
    retryable: isRetryable(status, kind, headers, retryAfter, maxRetryAfter),
    provider,
  });
}

/** Return whether a reply allows a retry, in the order openai checks. */
function isRetryable(
  status: number,
  kind: ErrorKind,
  headers: Headers,
  retryAfter: number | null,
  maxRetryAfter: number | null,
): boolean {
  if (maxRetryAfter !== null && retryAfter !== null && Number.isFinite(retryAfter) && retryAfter > maxRetryAfter) {
    return false;
  }
  const shouldRetry = headers.get("x-should-retry");
  if (shouldRetry === "true") return true;
  if (shouldRetry === "false") return false;
  if (kind === "quota") return false;
  return [408, 409, 429].includes(status) || status >= 500;
}

/** Return the most useful short text of an error body. */
function bodyDetail(body: string): string {
  return capDetail(jsonDetail(body) ?? (body.replace(/\s+/g, " ").trim() || "no body"));
}

/** Return the body's JSON `error.message`, `message` or `detail` string, or a string `error`, or `null`. */
function jsonDetail(body: string): string | null {
  let data;
  try {
    data = parseJson(body, "plain");
  } catch {
    return null;
  }
  const error = field(data, "error");
  for (const candidate of [field(error, "message"), field(data, "message"), field(data, "detail"), error]) {
    if (typeof candidate === "string" && candidate.trim()) return candidate;
  }
  return null;
}

/** Return `text`, or its first 2,000 code points and "…" when longer. */
export function capDetail(text: string): string {
  // A string of at most 2,000 code units has at most 2,000 code points.
  if (text.length <= MAX_DETAIL_CHARS) return text;
  const points = Array.from(text);
  return points.length > MAX_DETAIL_CHARS ? points.slice(0, MAX_DETAIL_CHARS).join("") + "…" : text;
}

/**
 * Redact an error graph with `redact` and return `error`.
 *
 * A string is returned redacted. Any other non-null object is rewritten in place: its
 * `message`, `stack` and `cause` when they are strings, and string members of an `errors`
 * array; then the same for each object reached through `cause` or an `errors` array
 * (`AggregateError`), each once. A property assignment can't set (a getter-only `message`)
 * becomes an own data property when the object is extensible, else it is skipped, as on a
 * frozen error. Anything else is returned unchanged. It never throws: a throwing getter,
 * proxy trap or `redact` only skips that property, and the error is never replaced.
 */
export function scrubError<T>(error: T, redact: (text: string) => string): T {
  if (typeof error === "string") {
    try {
      return redact(error) as T;
    } catch {
      return error;
    }
  }
  if (typeof error !== "object" || error === null) return error;
  const seen = new Set<object>();
  // A stack instead of recursion, so a deep cause chain can't overflow.
  const pending: object[] = [error];
  while (pending.length > 0) {
    const current = pending.pop()!;
    if (seen.has(current)) continue;
    seen.add(current);
    // The message first: the stack is read after it, in case reading one rebuilds the other.
    redactProperty(current, "message", redact);
    redactProperty(current, "stack", redact);
    const cause = redactProperty(current, "cause", redact);
    if (typeof cause === "object" && cause !== null) pending.push(cause);
    const errors = readProperty(current, "errors");
    const count = arrayLength(errors);
    for (let i = 0; i < count; i++) {
      const member = redactProperty(errors as unknown[], i, redact);
      if (typeof member === "object" && member !== null) pending.push(member);
    }
  }
  return error;
}

/** Return `target[name]`, or `undefined` when reading it throws. */
function readProperty(target: object, name: PropertyKey): unknown {
  try {
    return Reflect.get(target, name);
  } catch {
    return undefined;
  }
}

/** Return the length of `value` when it is an array, else 0; 0 when reading it throws. */
function arrayLength(value: unknown): number {
  try {
    return Array.isArray(value) ? value.length : 0;
  } catch {
    return 0;
  }
}

/**
 * Read `target[name]` once, replace it with its redaction when it is a string, and return the
 * value read (`undefined` when reading throws). A throw or an unsettable property skips the write.
 */
function redactProperty(target: object, name: PropertyKey, redact: (text: string) => string): unknown {
  let value: unknown;
  try {
    value = Reflect.get(target, name);
    if (typeof value !== "string") return value;
    const redacted = redact(value);
    if (redacted === value) return value;
    if (Reflect.set(target, name, redacted)) return value;
    if (!Object.isExtensible(target)) return value;
    Object.defineProperty(target, name, { value: redacted, writable: true, enumerable: false, configurable: true });
  } catch {
    // Scrubbing must neither stop the walk nor replace the error it was given.
  }
  return value;
}
