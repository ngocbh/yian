/**
 * Test helpers for model calls: cassettes and a scripted model.
 *
 * A cassette is a JSON file of HTTP exchanges. `CassetteReplay` replays one as a `Send` (pass
 * it to `RouteModel` or `safeFetch`) and checks each request against the recording, where
 * `CASSETTE_REDACTED` stands for "present, any value". `findSecrets` flags anything shaped like a
 * credential, so a test can refuse to commit a cassette that still holds one. `CassetteRecorder`
 * writes a cassette from real traffic with every secret replaced by `CASSETTE_REDACTED`, and
 * `recordMode` says whether a test should record. `ScriptedModel` is a `ChatModel` that answers
 * from a script, for tests above the HTTP layer.
 */

import { Buffer } from "node:buffer";
import { readFileSync } from "node:fs";

import type { Env } from "../defaults.ts";
import type { Send } from "../platform.ts";
import { JSONDecodeError, type JsonObject, type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import { scrub, type SecretForm, secretForms } from "./auth.ts";
import { encodeBody } from "./body.ts";
import {
  EventStream,
  Finish,
  PartDone,
  ReasoningDelta,
  TextDelta,
  ToolCallDelta,
  UsageReport,
  type ChatModel,
  type Event,
} from "./events.ts";
import { isPlainObject, put, ReasoningPart, Response, TextPart, ToolCallPart, type Request, typeName } from "./messages.ts";

/** The `format` every cassette holds. */
export const CASSETTE_FORMAT = "yian.cassette.v1";
/** The cassette marker for a credential: in a recording it matches any value that is present. */
export const CASSETTE_REDACTED = "<redacted>";
/** Headers that carry credentials. */
export const CREDENTIAL_HEADERS: ReadonlySet<string> = new Set([
  "authorization",
  "x-api-key",
  "api-key",
  "x-goog-api-key",
  "chatgpt-account-id",
  "cookie",
  "set-cookie",
  "proxy-authorization",
]);
/** Headers a recorder never saves and replay never compares: the runtime or the transport sets them. */
export const IGNORED_HEADERS: ReadonlySet<string> = new Set([
  "user-agent",
  "content-length",
  "host",
  "connection",
  "accept-encoding",
  "transfer-encoding",
  "content-encoding",
]);

const SOURCES = ["synthetic", "recorded"];
const CASSETTE_KEYS = ["format", "source", "description", "interactions"];
const INTERACTION_KEYS = ["request", "response"];
const REQUEST_KEYS = ["method", "url", "headers", "json"];
const RESPONSE_KEYS = ["status", "headers", "chunks", "json", "text"];
const BODY_KEYS = ["chunks", "json", "text"];
const MAX_DIFFS = 10;
// Response headers a recorder saves: the ones the transport and the retry rules read.
const SAVED_RESPONSE_HEADERS: ReadonlySet<string> = new Set(["content-type", "retry-after", "retry-after-ms", "x-should-retry"]);
// A chunk ends after a blank line, so a recorded SSE body replays one event per chunk.
const CHUNK = /.*?(?:\r\n\r\n|\n\n|\r\r)|.+/gs;
// Statuses whose response can't have a body: the Response constructor throws for one.
const NULL_BODY_STATUSES = new Set([101, 204, 205, 304]);

const UNICODE_ESCAPE = /\\u([0-9A-Fa-f]{4})/g;
const SECRET_PATTERNS: [string, RegExp][] = [
  ["openai_key", /(?<![A-Za-z0-9_\-])sk-[A-Za-z0-9_\-]{16,}/g],
  ["anthropic_key", /(?<![A-Za-z0-9_\-])sk-ant-[A-Za-z0-9_\-]{16,}/g],
  ["google_api_key", /(?<![A-Za-z0-9_\-])AIza[0-9A-Za-z_\-]{30,}/g],
  ["jwt", /(?<![A-Za-z0-9_\-])eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}/g],
  ["bearer_token", /(?<![A-Za-z0-9_\-])bearer\s+(?!<redacted>)[A-Za-z0-9._\-]{16,}/gi],
];

/** A recorded request. */
export interface CassetteRequest {
  method: string;
  url: string;
  headers?: Record<string, string>;
  /** The JSON body; a request with none must send no body. */
  json?: JsonValue;
}

/** A recorded response; it has exactly one of `chunks`, `json` and `text`. */
export interface CassetteResponse {
  /** A `bigint` when read from a file. */
  status: bigint | number;
  headers?: Record<string, string>;
  /** The body as it arrived, one read per chunk. */
  chunks?: string[];
  json?: JsonValue;
  text?: string;
}

/** One recorded exchange. */
export interface CassetteInteraction {
  request: CassetteRequest;
  response: CassetteResponse;
}

/** A `yian.cassette.v1` cassette. */
export interface Cassette {
  format: typeof CASSETTE_FORMAT;
  source: "synthetic" | "recorded";
  description: string;
  interactions: CassetteInteraction[];
}

/** A request `CassetteReplay` received. */
export interface ReceivedRequest {
  /** Upper-cased. */
  method: string;
  url: string;
  headers: Headers;
  /** The body as text, or `null` when there was none. */
  body: string | null;
}

/**
 * Find text shaped like an API key, a JWT or a bearer token.
 *
 * Each pattern must start at a word boundary, so text such as `task-0123...` is not read as
 * an `sk-` key. `Bearer <redacted>` is not flagged. `\\uXXXX` escapes are decoded first, so a
 * key a JSON encoder escaped (`\\u0073k-...`) is flagged.
 *
 * @param text The text to search, e.g. a cassette's JSON.
 * @returns One kind name per match (`openai_key`, `anthropic_key`, `google_api_key`, `jwt`,
 *   `bearer_token`), never the matched values; empty when nothing matches.
 */
export function findSecrets(text: string): string[] {
  const decoded = text.replace(UNICODE_ESCAPE, (_, hex: string) => String.fromCharCode(parseInt(hex, 16)));
  return SECRET_PATTERNS.flatMap(([kind, pattern]) => Array.from(decoded.matchAll(pattern), () => kind));
}

/** Order strings by UTF-16 code units. */
function byCodeUnits(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Throw unless `data` is a plain object with every required key and no others. */
function checkKeys(
  data: unknown,
  where: string,
  required: readonly string[],
  allowed: readonly string[],
): asserts data is Record<string, unknown> {
  if (!isPlainObject(data)) throw new TypeError(`${where}: expected an object, got ${typeName(data)}`);
  for (const key of required) {
    if (!Object.hasOwn(data, key)) throw new TypeError(`${where}: missing '${key}'`);
  }
  const unknown = Object.keys(data)
    .filter((key) => !allowed.includes(key))
    .sort(byCodeUnits);
  if (unknown.length > 0) throw new TypeError(`${where}: unknown keys ${JSON.stringify(unknown)}`);
}

/** Throw unless `headers` maps strings to strings. */
function checkHeaders(headers: unknown, where: string): void {
  if (!isPlainObject(headers) || !Object.values(headers).every((value) => typeof value === "string")) {
    throw new TypeError(`${where}.headers: expected an object of strings`);
  }
}

/** Return `data[key]` when `data` has its own `key`, else `fallback`. */
function get(data: Record<string, unknown>, key: string, fallback: unknown): unknown {
  return Object.hasOwn(data, key) ? data[key] : fallback;
}

/**
 * Check that `data` is a well-formed `yian.cassette.v1` cassette.
 *
 * @returns `data`.
 * @throws {TypeError} Naming the offending part, e.g. `interactions[0].response`. A status is
 *   an int when it is a `bigint` (an int read in plain mode) or an integer `number`.
 */
export function checkCassette(data: unknown): Cassette {
  // The format is checked first, so a file of another kind is named as such.
  if (!isPlainObject(data) || data["format"] !== CASSETTE_FORMAT) {
    throw new TypeError(`cassette: format must be '${CASSETTE_FORMAT}'`);
  }
  checkKeys(data, "cassette", CASSETTE_KEYS, CASSETTE_KEYS);
  if (!SOURCES.includes(data["source"] as string)) throw new TypeError("cassette: source must be 'synthetic' or 'recorded'");
  if (typeof data["description"] !== "string") throw new TypeError("cassette: description must be a string");
  const interactions = data["interactions"];
  if (!Array.isArray(interactions)) throw new TypeError("cassette: interactions must be a list");
  interactions.forEach((interaction: unknown, number) => {
    const where = `interactions[${number}]`;
    checkKeys(interaction, where, INTERACTION_KEYS, INTERACTION_KEYS);
    const { request, response } = interaction;
    checkKeys(request, `${where}.request`, ["method", "url"], REQUEST_KEYS);
    if (typeof request["method"] !== "string" || typeof request["url"] !== "string") {
      throw new TypeError(`${where}.request: method and url must be strings`);
    }
    checkHeaders(get(request, "headers", {}), `${where}.request`);
    checkKeys(response, `${where}.response`, ["status"], RESPONSE_KEYS);
    const status = response["status"];
    if (typeof status !== "bigint" && !Number.isInteger(status)) throw new TypeError(`${where}.response: status must be an int`);
    checkHeaders(get(response, "headers", {}), `${where}.response`);
    if (BODY_KEYS.filter((key) => Object.hasOwn(response, key)).length !== 1) {
      throw new TypeError(`${where}.response: needs exactly one of chunks, json, text`);
    }
    const chunks = get(response, "chunks", []);
    if (!Array.isArray(chunks) || !chunks.every((chunk) => typeof chunk === "string")) {
      throw new TypeError(`${where}.response: chunks must be a list of strings`);
    }
    if (typeof get(response, "text", "") !== "string") throw new TypeError(`${where}.response: text must be a string`);
  });
  return data as unknown as Cassette;
}

/**
 * Read a cassette file in plain JSON mode and check its shape.
 *
 * @param path The cassette's JSON file.
 * @returns The cassette.
 * @throws {TypeError} The file is not JSON (`cassette <path> is not JSON: ...`) or not a
 *   `yian.cassette.v1` cassette, naming the offending part.
 */
export function loadCassette(path: string): Cassette {
  let data: JsonValue;
  try {
    data = parseJson(readFileSync(path, "utf8"), "plain");
  } catch (error) {
    if (!(error instanceof JSONDecodeError)) throw error;
    throw new TypeError(`cassette ${path} is not JSON: ${error.message}`);
  }
  return checkCassette(data);
}

/**
 * Append the paths where `actual` differs from `recorded`, without values.
 *
 * `CASSETTE_REDACTED` in `recorded` matches any value at that path. A `bigint` and a `number`
 * of the same value match.
 */
function jsonDiffs(recorded: JsonValue, actual: JsonValue, path: string, out: string[]): void {
  if (recorded === CASSETTE_REDACTED) return;
  const where = path || "(root)";
  if (isPlainObject(recorded) && isPlainObject(actual)) {
    for (const key of Object.keys(recorded)) {
      const child = path ? `${path}.${key}` : key;
      if (!Object.hasOwn(actual, key)) out.push(`body ${child}: missing`);
      else jsonDiffs(recorded[key]!, actual[key]!, child, out);
    }
    for (const key of Object.keys(actual)) {
      if (!Object.hasOwn(recorded, key)) out.push(`body ${path ? `${path}.` : ""}${key}: unexpected`);
    }
  } else if (Array.isArray(recorded) && Array.isArray(actual)) {
    if (recorded.length !== actual.length) out.push(`body ${where}: length ${actual.length}, recorded ${recorded.length}`);
    for (let index = 0; index < Math.min(recorded.length, actual.length); index++) {
      jsonDiffs(recorded[index]!, actual[index]!, `${path}[${index}]`, out);
    }
  } else if (recorded !== actual && !sameInt(recorded, actual)) {
    out.push(`body ${where}: differs`);
  }
}

/**
 * Return whether `a` and `b` are a `bigint` and a `number` of the same value.
 *
 * Python tells an int from an equal float; TS can't, since `encodeBody` never writes `1.0`, so a
 * sent body never holds an integral float, and a recorded `1` must match the `1` it sends.
 */
function sameInt(a: JsonValue, b: JsonValue): boolean {
  const [int, other] = typeof a === "bigint" ? [a, b] : [b, a];
  return typeof int === "bigint" && typeof other === "number" && Number.isInteger(other) && BigInt(other) === int;
}

/** Return a query's values by key. */
function multimap(params: URLSearchParams): Map<string, string[]> {
  const out = new Map<string, string[]>();
  for (const [key, value] of params) {
    const values = out.get(key);
    if (values) values.push(value);
    else out.set(key, [value]);
  }
  return out;
}

/** Compare query strings as multimaps, naming keys but never values. */
function queryDiffs(recordedUrl: URL, actualUrl: URL): string[] {
  const recorded = multimap(recordedUrl.searchParams);
  const actual = multimap(actualUrl.searchParams);
  const keys = [...new Set([...recorded.keys(), ...actual.keys()])].sort(byCodeUnits);
  const diffs: string[] = [];
  for (const key of keys) {
    const expected = recorded.get(key) ?? [];
    const got = actual.get(key) ?? [];
    if (expected.length !== got.length) {
      diffs.push(`query parameter '${key}': ${got.length} value(s), recorded ${expected.length}`);
      continue;
    }
    const remaining = [...got];
    for (const value of expected) {
      if (value === CASSETTE_REDACTED) continue;
      const at = remaining.indexOf(value);
      if (at === -1) {
        diffs.push(`query parameter '${key}': value differs`);
        break;
      }
      remaining.splice(at, 1);
    }
  }
  return diffs;
}

const URL_PARTS = [
  ["scheme", "protocol"],
  ["host", "hostname"],
  ["port", "port"],
  ["path", "pathname"],
] as const;

/** List how `request` differs from a recorded one, never showing header or query values. */
function requestDiffs(recorded: CassetteRequest, request: ReceivedRequest): string[] {
  const diffs: string[] = [];
  if (request.method !== recorded.method.toUpperCase()) {
    diffs.push(`method ${request.method}, recorded ${recorded.method.toUpperCase()}`);
  }
  const expectedUrl = new URL(recorded.url);
  const actualUrl = new URL(request.url);
  for (const [part, key] of URL_PARTS) {
    if (actualUrl[key] !== expectedUrl[key]) diffs.push(`url ${part} differs`);
  }
  diffs.push(...queryDiffs(expectedUrl, actualUrl));
  for (const [name, value] of Object.entries(recorded.headers ?? {})) {
    const lower = name.toLowerCase();
    if (IGNORED_HEADERS.has(lower)) continue;
    if (!request.headers.has(lower)) diffs.push(`header '${lower}': missing`);
    else if (value !== CASSETTE_REDACTED && request.headers.get(lower) !== value) diffs.push(`header '${lower}': value differs`);
  }
  if (Object.hasOwn(recorded, "json")) {
    let body: JsonValue;
    try {
      body = parseJson(request.body ?? "", "plain");
    } catch (error) {
      if (!(error instanceof JSONDecodeError)) throw error;
      diffs.push("body: not JSON");
      return diffs;
    }
    jsonDiffs(recorded.json!, body, "", diffs);
  } else if (request.body) {
    diffs.push("body: present, recorded none");
  }
  return diffs;
}

/** Return the request `input` and `init` describe, its body read as text. */
async function receive(input: string | URL, init: RequestInit): Promise<ReceivedRequest> {
  const body = init.body == null ? null : await new globalThis.Response(init.body).text();
  return { method: (init.method ?? "GET").toUpperCase(), url: String(input), headers: new Headers(init.headers), body };
}

/**
 * Build the response a recorded interaction describes, its body read one chunk per pull and
 * errored with `signal`'s reason once it aborts, as fetch's is.
 */
function replayResponse(recorded: CassetteResponse, signal: AbortSignal | null | undefined): globalThis.Response {
  let texts: string[];
  if (Object.hasOwn(recorded, "chunks")) texts = recorded.chunks!;
  else if (Object.hasOwn(recorded, "json")) texts = [encodeBody(recorded.json, "cassette json")];
  else texts = [recorded.text!];
  const encoder = new TextEncoder();
  let next = 0;
  const body = new ReadableStream<Uint8Array>(
    {
      start(controller) {
        signal?.addEventListener("abort", () => controller.error(signal.reason));
      },
      pull(controller) {
        if (next < texts.length) controller.enqueue(encoder.encode(texts[next++]!));
        else controller.close();
      },
    },
    // No read-ahead, so each read gets exactly one recorded chunk.
    { highWaterMark: 0 },
  );
  const status = Number(recorded.status);
  return new globalThis.Response(NULL_BODY_STATUSES.has(status) ? null : body, { status, headers: recorded.headers });
}

/**
 * Replay a cassette's interactions in order as a `Send`, recording every request.
 *
 * Each request must match the next interaction: the same method, case-insensitively; the same
 * URL scheme, host, port and path (the path compared undecoded, as `URL` normalizes it, which
 * is stricter than httpx: `%41` and `A` differ); the same query parameters as a multimap; every recorded
 * header except `IGNORED_HEADERS`, by case-insensitive name; and an equal plain-JSON body when
 * the recording has `json`, or no body when it hasn't. `CASSETTE_REDACTED` in a recorded query
 * value, header value or JSON value matches any value, as long as one is present. A mismatch or
 * an extra request throws an `Error` naming the differing parts (at most 10), never header or
 * query values.
 */
export class CassetteReplay {
  /** Every request received, in order. */
  readonly requests: ReceivedRequest[] = [];
  readonly #interactions: CassetteInteraction[];

  /**
   * @param cassette The cassette, or the path of its JSON file.
   * @throws {TypeError} The cassette is malformed.
   */
  constructor(cassette: Cassette | string) {
    const data = typeof cassette === "string" ? loadCassette(cassette) : checkCassette(cassette);
    this.#interactions = data.interactions;
  }

  /** Check a request against the next interaction and return its response. */
  readonly send: Send = async (input, init) => {
    const request = await receive(input, init);
    this.requests.push(request);
    const number = this.requests.length;
    const total = this.#interactions.length;
    if (number > total) throw new Error(`unexpected request ${number}: the cassette has ${total} interaction(s)`);
    const interaction = this.#interactions[number - 1]!;
    const diffs = requestDiffs(interaction.request, request);
    if (diffs.length > 0) {
      const shown = diffs.slice(0, MAX_DIFFS);
      if (diffs.length > MAX_DIFFS) shown.push(`... and ${diffs.length - MAX_DIFFS} more`);
      throw new Error(`request ${number} does not match the cassette:\n  ${shown.join("\n  ")}`);
    }
    return replayResponse(interaction.response, init.signal);
  };

  /**
   * Check that every interaction was used.
   *
   * @throws {Error} Some interactions were never requested, naming how many.
   */
  assertDone(): void {
    const total = this.#interactions.length;
    const unused = total - this.requests.length;
    if (unused > 0) throw new Error(`${unused} of ${total} interaction(s) were not used`);
  }
}

/** One exchange a recorder saw: the request, the response's status and headers, and the body chunks read so far. */
interface Exchange {
  request: ReceivedRequest;
  status: number;
  headers: Headers;
  chunks: Uint8Array[];
}

/** The values a recorder redacts, or a function returning them, called when a cassette is built. */
export type RecorderSecrets = Iterable<string> | (() => Iterable<string>);

/**
 * Pass requests to `inner` and keep a redacted copy of each exchange.
 *
 * The response body is teed as the caller reads it, so a stream closed early still records what
 * was read; the runtime has already undone any content-encoding. Everything else happens in
 * `cassette`.
 *
 * Replay treats `CASSETTE_REDACTED` as "any value" only when it is a whole query value, header
 * value or JSON value. A secret inside a larger value, such as a URL path segment or
 * `pre-<redacted>`, or in a JSON key, is still redacted but must match literally, so that
 * cassette will not replay with a different key.
 */
export class CassetteRecorder {
  readonly #inner: Send;
  readonly #secrets: readonly string[] | (() => Iterable<string>);
  readonly #exchanges: Exchange[] = [];

  /**
   * @param inner The `Send` that really sends requests.
   * @param options `secrets` are the values to redact, e.g. the API key in use: an iterable,
   *   copied now, or a function, called each time `cassette` runs (so `auth.credentials()` can be
   *   passed lazily). Empty values are ignored.
   */
  constructor(inner: Send, options: { secrets: RecorderSecrets }) {
    this.#inner = inner;
    // Copied, so a generator still yields its secrets on a second `cassette()`.
    this.#secrets = typeof options.secrets === "function" ? options.secrets : [...options.secrets];
  }

  /** Send a request through `inner`, its body as the text read, and return the response with the body teed. */
  readonly send: Send = async (input, init) => {
    const request = await receive(input, init);
    // The body read for the record is what goes on, so a stream body isn't sent used up.
    const response = await this.#inner(input, { ...init, body: request.body ?? undefined });
    const exchange: Exchange = { request, status: response.status, headers: response.headers, chunks: [] };
    this.#exchanges.push(exchange);
    const reader = response.body?.getReader();
    const body =
      reader &&
      new ReadableStream<Uint8Array>(
        {
          async pull(controller) {
            const { done, value } = await reader.read();
            if (done) return controller.close();
            exchange.chunks.push(value);
            controller.enqueue(value);
          },
          cancel: (reason) => reader.cancel(reason),
        },
        // No read-ahead, so only what the caller reads is recorded.
        { highWaterMark: 0 },
      );
    const { status, statusText, headers } = response;
    return new globalThis.Response(body ?? null, { status, statusText, headers });
  };

  /**
   * Return the exchanges so far as a redacted `recorded` cassette.
   *
   * Each secret is replaced by `CASSETTE_REDACTED` in each form `secretForms` lists, in the URL,
   * header values, JSON keys and strings and the response text. Request headers are saved by
   * lowercase name without `IGNORED_HEADERS`, credential headers as `CASSETTE_REDACTED`; the
   * response keeps `content-type`, `retry-after`, `retry-after-ms` and `x-should-retry`. The
   * response text is decoded as UTF-8 with replacement and redacted whole, so a secret split
   * across reads is caught, then saved one chunk per SSE event.
   *
   * @param options `description` says what the cassette exercises.
   * @returns A `yian.cassette.v1` cassette with source `recorded`.
   * @throws {Error} A saved string, the description included, still holds a secret's form, its
   *   base64 forms included, or something `findSecrets` flags: `the cassette still holds a
   *   credential (<form>) in <where>`, naming the form or kind and the place, never the value.
   * @throws {JSONDecodeError} A request body is not JSON.
   */
  cassette(options: { description: string }): Cassette {
    const secrets = [...(typeof this.#secrets === "function" ? this.#secrets() : this.#secrets)].filter(Boolean);
    const interactions = this.#exchanges.map(({ request, status, headers, chunks }): CassetteInteraction => {
      const saved: CassetteRequest = {
        method: request.method,
        url: redact(request.url, secrets),
        headers: saveHeaders(request.headers, (name) => !IGNORED_HEADERS.has(name), secrets),
      };
      if (request.body) saved.json = redactJson(parseJson(request.body, "plain"), secrets);
      const text = redact(new TextDecoder().decode(Buffer.concat(chunks)), secrets);
      const response = {
        status,
        headers: saveHeaders(headers, (name) => SAVED_RESPONSE_HEADERS.has(name), secrets),
        chunks: text.match(CHUNK) ?? [],
      };
      return { request: saved, response };
    });
    const cassette: Cassette = { format: CASSETTE_FORMAT, source: "recorded", description: options.description, interactions };
    scanCassette(cassette, secrets);
    return cassette;
  }
}

/** Return `text` with each of `secrets`, in each of its forms, replaced by `CASSETTE_REDACTED`. */
function redact(text: string, secrets: readonly string[]): string {
  return scrub(text, secrets, CASSETTE_REDACTED);
}

/** Return `value` with `secrets` redacted in every string and key. */
function redactJson(value: JsonValue, secrets: readonly string[]): JsonValue {
  if (typeof value === "string") return redact(value, secrets);
  if (Array.isArray(value)) return value.map((item) => redactJson(item, secrets));
  if (!isPlainObject(value)) return value;
  const out: JsonObject = {};
  for (const [key, item] of Object.entries(value)) put(out, redact(key, secrets), redactJson(item, secrets));
  return out;
}

/** Save the headers `keep` accepts by lowercase name, credential headers as `CASSETTE_REDACTED`, others redacted. */
function saveHeaders(headers: Headers, keep: (name: string) => boolean, secrets: readonly string[]): Record<string, string> {
  const saved: Record<string, string> = {};
  for (const [name, raw] of headers) {
    if (!keep(name)) continue;
    const value = CREDENTIAL_HEADERS.has(name) ? CASSETTE_REDACTED : redact(raw, secrets);
    // A repeated header keeps every value, joined as HTTP allows.
    saved[name] = Object.hasOwn(saved, name) ? `${saved[name]}, ${value}` : value;
  }
  return saved;
}

// Shorter base64 fragments would match too much unrelated text.
const MIN_BASE64_FRAGMENT = 8;

/**
 * Return the base64 forms of `secret`: for each of the three byte alignments, the characters of
 * its standard and URL-safe encoding that only the secret's bits decide, when there are
 * `MIN_BASE64_FRAGMENT` or more. Base64 inside a larger blob can't be replaced in place, so
 * these are only scanned for.
 */
function base64Forms(secret: string): SecretForm[] {
  const bytes = Buffer.from(secret, "utf8");
  const forms: SecretForm[] = [];
  for (let pad = 0; pad < 3; pad++) {
    const encoded = Buffer.concat([Buffer.alloc(pad), bytes]).toString("base64");
    // Character i holds bits [6i, 6i + 6); keep those wholly inside the secret's bits.
    const start = Math.ceil((8 * pad) / 6);
    const end = Math.floor((8 * (pad + bytes.length)) / 6);
    const fragment = encoded.slice(start, end);
    if (fragment.length < MIN_BASE64_FRAGMENT) continue;
    forms.push({ form: "base64", text: fragment });
    forms.push({ form: "URL-safe base64", text: fragment.replaceAll("+", "-").replaceAll("/", "_") });
  }
  return forms;
}

/** Yield every string in `value` with its place; an object's keys come before its values. */
function* strings(value: unknown, where: string): Generator<[string, string]> {
  if (typeof value === "string") {
    yield [where, value];
  } else if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index++) yield* strings(value[index], `${where}[${index}]`);
  } else if (isPlainObject(value)) {
    // The key is never part of a place: it may be what is being reported.
    for (const key of Object.keys(value)) yield [`${where} (a key)`, key];
    for (const [key, item] of Object.entries(value)) yield* strings(item, `${where}.${key}`);
  }
}

/** Throw if a string `cassette` saves still holds a form of a secret or a `findSecrets` shape. */
function scanCassette(cassette: Cassette, secrets: readonly string[]): void {
  const forms = secrets.flatMap((secret) => [...secretForms(secret), ...base64Forms(secret)]);
  const saved: [string, string][] = [["description", cassette.description], ...strings(cassette.interactions, "interactions")];
  for (const [where, text] of saved) {
    const found = forms.find((form) =>
      form.ignoreCase ? text.toLowerCase().includes(form.text.toLowerCase()) : text.includes(form.text),
    );
    const kind = found?.form ?? findSecrets(text)[0];
    if (kind !== undefined) throw new Error(`the cassette still holds a credential (${kind}) in ${where}`);
  }
}

/**
 * Return whether a test should record its cassettes rather than replay them: when `RECORD` is
 * `1`.
 *
 * @param env The environment; test files pass `processEnv`.
 * @throws {Error} `RECORD=1` while `CI` is set (non-empty): CI must never make live calls.
 */
export function recordMode(env: Env): boolean {
  if (env("RECORD") !== "1") return false;
  if (env("CI")) throw new Error("RECORD=1 is refused when CI is set: CI only replays cassettes");
  return true;
}

/** One scripted reply: a `Response`, the events to stream with errors at their positions, or an error. */
export type ScriptedReply = Response | readonly (Event | Error)[] | Error;

/**
 * Yield the events a provider would stream for `response`.
 *
 * A tool call's delta text is `logJson` text (sorted keys; ints written as digits), unlike
 * Python's `json.dumps`; `collect` is unaffected because the `PartDone` wins.
 */
function* responseEvents(response: Response): Generator<Event> {
  const parts = response.message.parts;
  for (let index = 0; index < parts.length; index++) {
    const part = parts[index]!;
    if (part instanceof TextPart) {
      yield new TextDelta(index, part.text);
    } else if (part instanceof ReasoningPart) {
      yield new ReasoningDelta(index, part.text);
    } else if (part instanceof ToolCallPart) {
      const args = part.arguments === null ? part.rawArguments! : logJson(part.arguments);
      yield new ToolCallDelta({ index, id: part.id, name: part.name, arguments: args });
    }
    yield new PartDone(index, part);
  }
  if (response.usage !== null) yield new UsageReport(response.usage);
  yield new Finish({
    reason: response.finishReason,
    responseId: response.responseId,
    protocol: response.message.protocol,
    providerPayload: response.message.providerPayload,
  });
}

/** Yield the reply's events, throwing a scripted error at its position or `signal.reason` once aborted. */
async function* replyEvents(reply: ScriptedReply, signal: AbortSignal | undefined): AsyncGenerator<Event> {
  const items = reply instanceof Response ? responseEvents(reply) : reply instanceof Error ? [reply] : [...reply];
  for (const item of items) {
    signal?.throwIfAborted();
    if (item instanceof Error) throw item;
    yield item;
  }
}

/**
 * A `ChatModel` that answers each request with the next scripted reply.
 *
 * A `Response` reply streams as events: per part a delta (text, reasoning, or the tool call's
 * arguments as `logJson` text, its raw text when the arguments didn't parse) then its
 * `PartDone`, or only the `PartDone` for parts with no delta kind (media, refusal), then a
 * `UsageReport` when there is usage, then a `Finish` carrying the finish reason, response id
 * and the message's protocol and provider payload, so `collect` of the events equals the
 * `Response`. A list reply streams as given, throwing any error in it at that position. An
 * error reply is thrown by the stream's first `next()`.
 */
export class ScriptedModel implements ChatModel {
  /** The provider name. */
  readonly provider: string;
  /** The model name. */
  readonly model: string;
  /** Every request received, in order. */
  readonly requests: Request[] = [];
  /** How many streams returned by `stream` have been closed. */
  closed = 0;
  #replies: ScriptedReply[];

  /**
   * @param replies The replies, one per request.
   * @param options `provider` and `model` name the model; both default to `scripted`.
   */
  constructor(replies: Iterable<ScriptedReply>, options: { provider?: string; model?: string } = {}) {
    this.#replies = Array.from(replies);
    this.provider = options.provider ?? "scripted";
    this.model = options.model ?? "scripted";
  }

  /**
   * Record `request` and answer it with the next reply's events.
   *
   * @param request The call's input; recorded.
   * @param options `signal` aborts the stream: the next `next()` throws its reason.
   * @returns The reply's events.
   * @throws {Error} No reply is left; the test's own error, so it is thrown at once.
   */
  stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    this.requests.push(request);
    const reply = this.#replies.shift();
    if (reply === undefined) throw new Error(`ScriptedModel has no reply left for request ${this.requests.length}`);
    return new EventStream(replyEvents(reply, options.signal), () => {
      this.closed++;
    });
  }
}
