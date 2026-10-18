/**
 * yian's lossless JSON reader: an int token reads as a `bigint` of any size, any other
 * number as a `number`, and errors carry Python 3.12 `json.loads`'s reasons and positions.
 * It refuses what yian refuses on top of that: the NaN and Infinity tokens, a float token
 * that overflows, a duplicate object key and a lone surrogate. It reads with an explicit
 * stack, so any depth reads.
 *
 * It imports nothing from yian, so the `llm/` leaf can use it.
 */

/** A JSON value as yian reads it: an int token is a `bigint`, any other number a `number`. */
export type JsonValue = null | boolean | bigint | number | string | JsonValue[] | JsonObject;

/** A JSON object: a plain object whose keys are own data properties. */
export interface JsonObject {
  [key: string]: JsonValue;
}

/** `codec` decodes yian's infinity tag and `$record` escape (only for JSON yian wrote); `plain` decodes neither. */
export type JsonMode = "codec" | "plain";

/** A JSON text yian can't read; the message is `<reason>: line L column C (char P)`. */
export class JSONDecodeError extends Error {
  override name = "JSONDecodeError";
  /** What is wrong, e.g. "Expecting value". */
  readonly reason: string;
  /** The 0-based UTF-16 index into the text. */
  readonly position: number;
  /** The 1-based line; lines break at "\n". */
  readonly line: number;
  /** The 1-based UTF-16 column. */
  readonly column: number;

  constructor(reason: string, text: string, position: number) {
    let line = 1;
    for (let i = text.indexOf("\n"); i !== -1 && i < position; i = text.indexOf("\n", i + 1)) line++;
    // `lastIndexOf("\n", -1)` would look at index 0, so position 0 needs its own case.
    const column = position - (position === 0 ? -1 : text.lastIndexOf("\n", position - 1));
    super(`${reason}: line ${line} column ${column} (char ${position})`);
    this.reason = reason;
    this.position = position;
    this.line = line;
    this.column = column;
  }
}

/** An open array or object while reading. */
interface Frame {
  container: JsonValue[] | JsonObject;
  /** The index of the opening bracket. */
  start: number;
  /** In an object, the key whose value is being read. */
  key?: string;
  /** The index of `key`'s opening quote. */
  keyStart?: number;
  /** The first key seen twice and the index of its second quote, reported as the object closes. */
  duplicate?: { key: string; position: number };
}

// Python's `NUMBER_RE`; the fraction and exponent are optional, so `1.` matches `1`.
const NUMBER = /-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?/y;
const HEX4 = /^[0-9a-fA-F]{4}$/;
const ESCAPES = new Map([
  ['"', '"'],
  ["\\", "\\"],
  ["/", "/"],
  ["b", "\b"],
  ["f", "\f"],
  ["n", "\n"],
  ["r", "\r"],
  ["t", "\t"],
]);

/** Return the index of the first non-whitespace character at or after `pos`. */
function skipSpace(text: string, pos: number): number {
  for (;;) {
    const c = text.charCodeAt(pos);
    if (c !== 0x20 && c !== 0x09 && c !== 0x0a && c !== 0x0d) return pos;
    pos++;
  }
}

/** Return the string whose opening quote is at `quote`, and the index after its closing quote. */
function scanString(text: string, quote: number): [string, number] {
  let out = "";
  let pos = quote + 1;
  for (;;) {
    let end = pos;
    while (end < text.length) {
      const c = text.charCodeAt(end);
      if (c === 0x22 || c === 0x5c || c < 0x20) break;
      end++;
    }
    out += text.slice(pos, end);
    pos = end;
    if (pos >= text.length) throw new JSONDecodeError("Unterminated string starting at", text, quote);
    const c = text.charCodeAt(pos);
    if (c === 0x22) return [out, pos + 1];
    if (c < 0x20) throw new JSONDecodeError("Invalid control character at", text, pos);
    // A backslash.
    if (pos + 1 >= text.length) throw new JSONDecodeError("Unterminated string starting at", text, quote);
    const letter = text.charAt(pos + 1);
    if (letter === "u") {
      const hex = text.slice(pos + 2, pos + 6);
      // CPython refuses a `\u` escape that ends the text, before seeing the string is unterminated.
      if (pos + 6 >= text.length || !HEX4.test(hex)) throw new JSONDecodeError("Invalid \\uXXXX escape", text, pos + 1);
      // Each escape adds one code unit, so an escaped pair joins into one character.
      out += String.fromCharCode(Number.parseInt(hex, 16));
      pos += 6;
    } else {
      const escaped = ESCAPES.get(letter);
      if (escaped === undefined) throw new JSONDecodeError("Invalid \\escape", text, pos);
      out += escaped;
      pos += 2;
    }
  }
}

/** Return the number whose token starts at `pos`, and the index after it. */
function scanNumber(text: string, pos: number): [bigint | number, number] {
  NUMBER.lastIndex = pos;
  const match = NUMBER.exec(text);
  if (match === null) throw new JSONDecodeError("Expecting value", text, pos);
  const token = match[0];
  const end = pos + token.length;
  if (match[1] === undefined && match[2] === undefined) {
    try {
      return [BigInt(token), end];
    } catch (error) {
      if (error instanceof RangeError) throw new JSONDecodeError("integer is too large for this runtime", text, pos);
      throw error;
    }
  }
  const value = Number(token);
  if (!Number.isFinite(value)) throw new JSONDecodeError("number is too large for a float", text, pos);
  return [value === 0 ? 0 : value, end];
}

/** Return the scalar value starting at `pos`, and the index after it. */
function scanScalar(text: string, pos: number): [JsonValue, number] {
  const c = text.charAt(pos);
  if (c === '"') {
    const [value, end] = scanString(text, pos);
    if (!value.isWellFormed()) throw new JSONDecodeError("string has an unpaired surrogate", text, pos);
    return [value, end];
  }
  if (c === "n" && text.startsWith("null", pos)) return [null, pos + 4];
  if (c === "t" && text.startsWith("true", pos)) return [true, pos + 4];
  if (c === "f" && text.startsWith("false", pos)) return [false, pos + 5];
  for (const token of ["NaN", "Infinity", "-Infinity"]) {
    if (text.startsWith(token, pos)) throw new JSONDecodeError(`${token} is not JSON`, text, pos);
  }
  return scanNumber(text, pos);
}

/** Read the key and `:` at `pos` into `frame`; return the index after the `:` and its whitespace. */
function scanKey(text: string, pos: number, frame: Frame): number {
  if (text.charAt(pos) !== '"') {
    throw new JSONDecodeError("Expecting property name enclosed in double quotes", text, pos);
  }
  const [key, end] = scanString(text, pos);
  if (!key.isWellFormed()) throw new JSONDecodeError("object key has an unpaired surrogate", text, pos);
  const colon = skipSpace(text, end);
  if (text.charAt(colon) !== ":") throw new JSONDecodeError("Expecting ':' delimiter", text, colon);
  frame.key = key;
  frame.keyStart = pos;
  return skipSpace(text, colon + 1);
}

/** Return the value `text` holds; throw `JSONDecodeError` when it isn't JSON yian reads. */
export function parseJson(text: string, mode: JsonMode): JsonValue {
  if (text.charCodeAt(0) === 0xfeff) {
    throw new JSONDecodeError("Unexpected UTF-8 BOM (decode using utf-8-sig)", text, 0);
  }
  const stack: Frame[] = [];
  let pos = skipSpace(text, 0);
  for (;;) {
    // Read the value at `pos`, or open a container and go on to its first value.
    let value: JsonValue;
    const c = text.charAt(pos);
    if (c === "[" || c === "{") {
      const frame: Frame = { container: c === "[" ? [] : {}, start: pos };
      pos = skipSpace(text, pos + 1);
      if (text.charAt(pos) === (c === "[" ? "]" : "}")) {
        value = frame.container;
        pos++;
      } else {
        stack.push(frame);
        if (c === "{") pos = scanKey(text, pos, frame);
        continue;
      }
    } else {
      [value, pos] = scanScalar(text, pos);
    }
    // Add the value to its container, closing each container it completes.
    for (;;) {
      const frame = stack.at(-1);
      if (frame === undefined) {
        pos = skipSpace(text, pos);
        if (pos !== text.length) throw new JSONDecodeError("Extra data", text, pos);
        return value;
      }
      const { container } = frame;
      if (Array.isArray(container)) {
        container.push(value);
      } else {
        const key = frame.key!;
        if (Object.hasOwn(container, key)) {
          frame.duplicate ??= { key, position: frame.keyStart! };
        } else if (key === "__proto__") {
          // Assigning `__proto__` would set the prototype; define it so it stays data.
          Object.defineProperty(container, key, { value, writable: true, enumerable: true, configurable: true });
        } else {
          container[key] = value;
        }
      }
      pos = skipSpace(text, pos);
      const next = text.charAt(pos);
      if (next === ",") {
        pos = skipSpace(text, pos + 1);
        if (!Array.isArray(container)) pos = scanKey(text, pos, frame);
        break;
      }
      if (next !== (Array.isArray(container) ? "]" : "}")) {
        throw new JSONDecodeError("Expecting ',' delimiter", text, pos);
      }
      // Like yian's `object_pairs_hook`, the duplicate check runs as the object closes.
      if (frame.duplicate !== undefined) {
        const { key, position } = frame.duplicate;
        throw new JSONDecodeError(`duplicate JSON object key: ${JSON.stringify(key)}`, text, position);
      }
      stack.pop();
      value = container;
      pos++;
    }
  }
}
