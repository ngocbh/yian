/**
 * Split a streamed HTTP body into server-sent events or NDJSON values.
 *
 * Both readers take the body as byte chunks of any size and do work linear in its length:
 * a partial line is kept as a list of pieces and joined once, never rescanned. Both read
 * the chunks with `for await`, so a consumer that stops early closes the source.
 */

import { ProviderError } from "./errors.ts";
import { JSONDecodeError, parseJson, type JsonValue } from "../typesys/json.ts";

const LINE_END = /\r\n|\r|\n/g;
const LF = 0x0a;
// The bytes Python's `bytes.strip()` removes: space, tab, LF, CR, VT, FF.
const ASCII_SPACE = new Set([0x20, 0x09, 0x0a, 0x0d, 0x0b, 0x0c]);
const STRICT_UTF8 = new TextDecoder("utf-8", { fatal: true });

/** One dispatched server-sent event. */
export class SSEEvent {
  /** The event's `data` lines joined with `\n`. */
  readonly data: string;
  /** The `event` field, or `null` when absent or empty. */
  readonly event: string | null;
  /** The last event id seen in the stream, or `null` if none. */
  readonly id: string | null;

  constructor({ data, event = null, id = null }: { data: string; event?: string | null; id?: string | null }) {
    this.data = data;
    this.event = event;
    this.id = id;
  }
}

/**
 * Parse a server-sent event stream, following the WHATWG rules, yielding each event as
 * soon as its blank line arrives.
 *
 * The bytes are decoded as UTF-8 (invalid bytes become U+FFFD) with one leading BOM
 * removed. Lines end at CRLF, CR or LF, even when a CRLF is split across chunks. Comment
 * lines (starting with `:`) and `retry` or unknown fields are ignored. A blank line
 * dispatches the event if it had a `data` field. An event still open at the end of the
 * stream is dropped.
 */
export async function* iterSse(chunks: AsyncIterable<Uint8Array> | Iterable<Uint8Array>): AsyncGenerator<SSEEvent> {
  // The default decoder removes one leading BOM, even one split across chunks.
  const decoder = new TextDecoder();
  let data: string[] = [];
  let event: string | null = null;
  let lastId: string | null = null;
  const pending: string[] = [];
  // A CR ending one piece may be the first half of a CRLF; its LF is then skipped.
  let afterCr = false;
  for await (const chunk of chunks) {
    let text = decoder.decode(chunk, { stream: true });
    if (afterCr && text.startsWith("\n")) {
      text = text.slice(1);
      afterCr = false;
    }
    if (text) afterCr = text.endsWith("\r");
    let start = 0;
    for (const match of text.matchAll(LINE_END)) {
      pending.push(text.slice(start, match.index));
      const line = pending.join("");
      pending.length = 0;
      start = match.index + match[0].length;
      if (!line) {
        if (data.length > 0) yield new SSEEvent({ data: data.join("\n"), event, id: lastId });
        data = [];
        event = null;
        continue;
      }
      if (line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const name = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (name === "data") data.push(value);
      else if (name === "event") event = value || null;
      else if (name === "id" && !value.includes("\0")) lastId = value;
    }
    if (start < text.length) pending.push(text.slice(start));
  }
}

/**
 * Parse newline-delimited JSON, yielding each non-blank line's value; a final line without
 * a newline still counts. Ints read as `bigint`.
 *
 * Throws `ProviderError` (kind `protocol`) when a line is not valid UTF-8 JSON; the message
 * gives the line number, never the line.
 */
export async function* iterNdjson(chunks: AsyncIterable<Uint8Array> | Iterable<Uint8Array>): AsyncGenerator<JsonValue> {
  const pending: Uint8Array[] = [];
  let number = 0;
  for await (const chunk of chunks) {
    let start = 0;
    for (let end = chunk.indexOf(LF); end !== -1; end = chunk.indexOf(LF, start)) {
      pending.push(chunk.subarray(start, end));
      start = end + 1;
      number += 1;
      const line = join(pending);
      pending.length = 0;
      if (!isBlank(line)) yield jsonLine(line, number);
    }
    // Copied, since the source may reuse the chunk's buffer once it moves on.
    if (start < chunk.length) pending.push(chunk.slice(start));
  }
  const line = join(pending);
  if (!isBlank(line)) yield jsonLine(line, number + 1);
}

/** Return the concatenation of `pieces`. */
function join(pieces: Uint8Array[]): Uint8Array {
  if (pieces.length === 1) return pieces[0]!;
  let size = 0;
  for (const piece of pieces) size += piece.length;
  const out = new Uint8Array(size);
  let offset = 0;
  for (const piece of pieces) {
    out.set(piece, offset);
    offset += piece.length;
  }
  return out;
}

/** Return whether `line` holds only ASCII whitespace. */
function isBlank(line: Uint8Array): boolean {
  return line.every((byte) => ASCII_SPACE.has(byte));
}

/** Parse one NDJSON line, throwing a `ProviderError` that does not echo it. */
function jsonLine(line: Uint8Array, number: number): JsonValue {
  let text: string;
  try {
    text = STRICT_UTF8.decode(line);
  } catch (err) {
    if (!(err instanceof TypeError)) throw err;
    throw invalidLine(number);
  }
  try {
    return parseJson(text, "plain");
  } catch (err) {
    if (!(err instanceof JSONDecodeError)) throw err;
    throw invalidLine(number);
  }
}

/** Return the error for NDJSON line `number`, which is not valid UTF-8 JSON. */
function invalidLine(number: number): ProviderError {
  return new ProviderError("protocol", `invalid JSON line ${number} in NDJSON stream`);
}
