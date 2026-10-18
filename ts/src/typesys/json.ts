/**
 * yian's lossless JSON reader: an int token reads as a `bigint` of any size, any other
 * number as a `number`, and errors carry Python 3.12 `json.loads`'s reasons and positions.
 * It refuses what yian refuses on top of that: the NaN and Infinity tokens, a float token
 * that overflows, a duplicate object key and a lone surrogate. A float token that is zero
 * reads as `0`, never `-0`. It reads with an explicit stack, so any depth reads.
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
    let lineStart = 0;
    for (let i = text.indexOf("\n"); i !== -1 && i < position; i = text.indexOf("\n", i + 1)) {
      line++;
      lineStart = i + 1;
    }
    const column = position - lineStart + 1;
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
const CONSTANTS = ["NaN", "Infinity", "-Infinity"];
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

/** Return the value of the four hex digits at `pos`, or -1 when they aren't four hex digits. */
function hex4(text: string, pos: number): number {
  let value = 0;
  for (let i = pos; i < pos + 4; i++) {
    // Past the end, `charCodeAt` is NaN, which matches no range.
    const c = text.charCodeAt(i);
    let digit: number;
    if (c >= 0x30 && c <= 0x39) digit = c - 0x30;
    else if (c >= 0x61 && c <= 0x66) digit = c - 0x61 + 10;
    else if (c >= 0x41 && c <= 0x46) digit = c - 0x41 + 10;
    else return -1;
    value = value * 16 + digit;
  }
  return value;
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
      // CPython refuses a `\u` escape that ends the text, before seeing the string is unterminated.
      const unit = pos + 6 >= text.length ? -1 : hex4(text, pos + 2);
      if (unit < 0) throw new JSONDecodeError("Invalid \\uXXXX escape", text, pos + 1);
      // Each escape adds one code unit and adjacent code units join, so an escaped or raw
      // surrogate next to its pair forms one character.
      out += String.fromCharCode(unit);
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
  // yian has no negative zero: `-0.0` reads as `0` (Python keeps `-0.0`).
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
  if (c === "N" || c === "I" || c === "-") {
    for (const token of CONSTANTS) {
      if (text.startsWith(token, pos)) throw new JSONDecodeError(`${token} is not JSON`, text, pos);
    }
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

/**
 * Return the value `text` holds; throw `JSONDecodeError` when it isn't JSON yian reads.
 * `codec` reads as `plain` until codec decoding is added.
 */
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
      const isArray = Array.isArray(container);
      if (isArray) {
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
        if (!isArray) pos = scanKey(text, pos, frame);
        break;
      }
      if (next !== (isArray ? "]" : "}")) {
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
