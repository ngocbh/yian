/**
 * Test helpers for model calls: cassettes and a scripted model.
 *
 * A cassette is a JSON file of HTTP exchanges. `CassetteReplay` replays one as a `Send` (pass
 * it to `RouteModel` or `safeFetch`) and checks each request against the recording, where
 * `CASSETTE_REDACTED` stands for "present, any value". `findSecrets` flags anything shaped like a
 * credential, so a test can refuse to commit a cassette that still holds one. `ScriptedModel` is
 * a `ChatModel` that answers from a script, for tests above the HTTP layer.
 */

import { readFileSync } from "node:fs";

import type { Send } from "../platform.ts";
import { JSONDecodeError, type JsonObject, type JsonValue, logJson, parseJson } from "../typesys/json.ts";
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
import { isPlainObject, ReasoningPart, Response, TextPart, ToolCallPart, type Request, typeName } from "./messages.ts";

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
/** Headers replay never compares: the runtime or the transport sets them. */
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

/** Return Python's repr of a list of strings, for messages. */
function quoted(names: readonly string[]): string {
  return `[${names.map((name) => `'${name}'`).join(", ")}]`;
}

/** Throw unless `data` is a plain object with every required key and no others. */
function checkKeys(data: unknown, where: string, required: readonly string[], allowed: readonly string[]): asserts data is Record<string, unknown> {
  if (!isPlainObject(data)) throw new TypeError(`${where}: expected an object, got ${typeName(data)}`);
  for (const key of required) {
    if (!Object.hasOwn(data, key)) throw new TypeError(`${where}: missing '${key}'`);
  }
  const unknown = Object.keys(data)
    .filter((key) => !allowed.includes(key))
    .sort(byCodeUnits);
  if (unknown.length > 0) throw new TypeError(`${where}: unknown keys ${quoted(unknown)}`);
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
    const expected = recorded as JsonObject;
    const got = actual as JsonObject;
    for (const key of Object.keys(expected)) {
      const child = path ? `${path}.${key}` : key;
      if (!Object.hasOwn(got, key)) out.push(`body ${child}: missing`);
      else jsonDiffs(expected[key]!, got[key]!, child, out);
    }
    for (const key of Object.keys(got)) {
      if (!Object.hasOwn(expected, key)) out.push(`body ${path ? `${path}.` : ""}${key}: unexpected`);
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

/** Build the response a recorded interaction describes, its body read one chunk per pull. */
function replayResponse(recorded: CassetteResponse): globalThis.Response {
  let texts: string[];
  if (Object.hasOwn(recorded, "chunks")) texts = recorded.chunks!;
  else if (Object.hasOwn(recorded, "json")) texts = [encodeBody(recorded.json, "cassette json")];
  else texts = [recorded.text!];
  const encoder = new TextEncoder();
  let next = 0;
  const body = new ReadableStream<Uint8Array>(
    {
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
    const body = init.body == null ? null : await new globalThis.Response(init.body).text();
    const request = { method: (init.method ?? "GET").toUpperCase(), url: String(input), headers: new Headers(init.headers), body };
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
    return replayResponse(interaction.response);
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
