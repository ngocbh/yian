/** Tests for the lossless JSON reader in `src/typesys/json.ts`. */

import { describe, expect, test } from "bun:test";

import { JSONDecodeError, parseJson, type JsonValue } from "../../src/typesys/json.ts";

/** Return the `JSONDecodeError` that parsing `text` throws, or fail. */
function decodeError(text: string): JSONDecodeError {
  try {
    parseJson(text, "plain");
  } catch (error) {
    if (error instanceof JSONDecodeError) return error;
    throw error;
  }
  throw new Error(`parsed without an error: ${JSON.stringify(text)}`);
}

/** Expect `text` to fail with `reason` at `position`, and the message to carry `line` and `column`. */
function expectError(text: string, reason: string, position: number, line = 1, column = position + 1): void {
  const error = decodeError(text);
  expect(error.name).toBe("JSONDecodeError");
  expect([error.reason, error.position, error.line, error.column]).toEqual([reason, position, line, column]);
  expect(error.message).toBe(`${reason}: line ${line} column ${column} (char ${position})`);
}

/** Return the number of nested containers along the first-child path of `value`, walking iteratively. */
function depth(value: JsonValue): number {
  let n = 0;
  let node: JsonValue | undefined = value;
  while (typeof node === "object" && node !== null) {
    n++;
    node = Array.isArray(node) ? node[0] : Object.values(node)[0];
  }
  return n;
}

// Python 3.12's `json.loads` errors, as [text, reason, position]; each text is ASCII on one line,
// so line is 1 and column is position + 1.
const PYTHON_ERRORS: [string, string, number][] = [
  ["", "Expecting value", 0],
  ["   ", "Expecting value", 3],
  ["-", "Expecting value", 0],
  ["-a", "Expecting value", 0],
  ["nul", "Expecting value", 0],
  ["tru", "Expecting value", 0],
  ["Na", "Expecting value", 0],
  ["Inf", "Expecting value", 0],
  ["-Inf", "Expecting value", 0],
  ["[", "Expecting value", 1],
  ["[1,", "Expecting value", 3],
  ["[1,]", "Expecting value", 3],
  ["[1,2,]", "Expecting value", 5],
  ['{"a":', "Expecting value", 5],
  ['{"a":1,"b":}', "Expecting value", 11],
  ['{"a":tru}', "Expecting value", 5],
  ["{bad", "Expecting property name enclosed in double quotes", 1],
  ["{1:2}", "Expecting property name enclosed in double quotes", 1],
  ["{", "Expecting property name enclosed in double quotes", 1],
  ['{"a":1,}', "Expecting property name enclosed in double quotes", 7],
  ['{ "a" : [ 1 , { } ] , }', "Expecting property name enclosed in double quotes", 22],
  ['{"a" 1}', "Expecting ':' delimiter", 5],
  ['{"a"', "Expecting ':' delimiter", 4],
  ['{"a":1,"b"}', "Expecting ':' delimiter", 10],
  ['{"a"\u0000:1}', "Expecting ':' delimiter", 4],
  ['{"a":1 "b":2}', "Expecting ',' delimiter", 7],
  ["[1 2]", "Expecting ',' delimiter", 3],
  ["[1,2", "Expecting ',' delimiter", 4],
  ['{"a":1', "Expecting ',' delimiter", 6],
  ["[01]", "Expecting ',' delimiter", 2],
  ["[1.5e]", "Expecting ',' delimiter", 4],
  // The duplicate check runs as the object closes, like yian's `object_pairs_hook`.
  ['{"a":1,"a":2', "Expecting ',' delimiter", 12],
  ["1 2", "Extra data", 2],
  ["01", "Extra data", 1],
  ["1.", "Extra data", 1],
  ["1e", "Extra data", 1],
  ["-01", "Extra data", 2],
  ['{"a":1}x', "Extra data", 7],
  ["[1]]", "Extra data", 3],
  ["true false", "Extra data", 5],
  ['"abc', "Unterminated string starting at", 0],
  ['"\\', "Unterminated string starting at", 0],
  ['"a\u0001"', "Invalid control character at", 2],
  ['"\t"', "Invalid control character at", 1],
  ['"a\nb"', "Invalid control character at", 2],
  ['"\\x"', "Invalid \\escape", 1],
  ['"ab\\x"', "Invalid \\escape", 3],
  ['"\\u12"', "Invalid \\uXXXX escape", 2],
  ['"\\u12G4"', "Invalid \\uXXXX escape", 2],
  ['"ab\\uZZZZ"', "Invalid \\uXXXX escape", 4],
  ['"ab\\u00"', "Invalid \\uXXXX escape", 4],
  ['"\\ud800\\u12"', "Invalid \\uXXXX escape", 8],
  // A `\u` escape that ends the text is bad, not an unterminated string.
  ['"\\u0041', "Invalid \\uXXXX escape", 2],
  ["\ufeff1", "Unexpected UTF-8 BOM (decode using utf-8-sig)", 0],
];

describe("parseJson", () => {
  test("reads ints as bigint and other numbers as number", () => {
    expect(parseJson("[0, -0, 1, 1.0, 1e5, 1E5, -1.0E-2, 123456789012345678901234567890]", "plain")).toEqual([
      0n,
      0n,
      1n,
      1,
      100000,
      100000,
      -0.01,
      123456789012345678901234567890n,
    ]);
  });

  test("reads -0.0 as 0 and an underflow as 0", () => {
    expect(Object.is(parseJson("-0.0", "plain"), 0)).toBe(true);
    expect(Object.is(parseJson("-1e-400", "plain"), 0)).toBe(true);
    expect(parseJson("1e-400", "plain") === 0).toBe(true);
  });

  test("reads strings, escapes and literals", () => {
    expect(parseJson('"\\" \\\\ \\/ \\b \\f \\n \\r \\t"', "plain")).toBe('" \\ / \b \f \n \r \t');
    expect(parseJson('"\\u00e9"', "plain")).toBe("é");
    expect(parseJson('"caf\\u00E9!"', "plain")).toBe("café!");
    expect(parseJson('"\\ud83d\\ude00"', "plain")).toBe("\u{1F600}");
    expect(parseJson('"é 日本 \u{1F600}"', "plain")).toBe("é 日本 \u{1F600}");
    expect(parseJson('""', "plain")).toBe("");
    expect(parseJson("true", "plain")).toBe(true);
    expect(parseJson("false", "plain")).toBe(false);
    expect(parseJson("null", "plain")).toBe(null);
    expect(parseJson(" \t\r\n [ ] \n", "plain")).toEqual([]);
    expect(parseJson("{}", "plain")).toEqual({});
    expect(parseJson('{ "a" : [ 1 , { "b" : null } , [ ] , { } ] , "c" : "d" }', "plain")).toEqual({
      a: [1n, { b: null }, [], {}],
      c: "d",
    });
  });

  test("keeps __proto__ and toString as data", () => {
    const text = '{"__proto__": {"x": 1}, "toString": 2}';
    const v = parseJson(text, "plain") as unknown as { x?: unknown; toString: unknown };
    expect(Object.getPrototypeOf(v)).toBe(Object.prototype);
    expect(Object.hasOwn(v, "__proto__")).toBe(true);
    expect(Object.keys(v)).toEqual(["__proto__", "toString"]);
    expect(v.x).toBeUndefined();
    expect(v.toString).toBe(2n);
  });

  test.each(PYTHON_ERRORS)("reports Python's syntax errors: %j", (text, reason, position) => {
    expectError(text, reason, position);
  });

  test("counts lines at \\n and columns in UTF-16 code units", () => {
    expectError("\n\n  [1,\n   ]", "Expecting value", 11, 4, 4);
    expectError('["a",\n"b"\n,]', "Expecting value", 11, 3, 2);
    expectError("\r\n[\r\n,]", "Expecting value", 5, 3, 1);
    expectError('"\u{1F600}" x', "Extra data", 5, 1, 6);
  });

  test("refuses NaN and Infinity tokens, overflow, duplicates and lone surrogates", () => {
    expectError("NaN", "NaN is not JSON", 0);
    expectError("[Infinity]", "Infinity is not JSON", 1);
    expectError("-Infinity", "-Infinity is not JSON", 0);
    expectError("1e400", "number is too large for a float", 0);
    expectError("[-1e400]", "number is too large for a float", 1);
    expectError('{"a":1,"a":2}', 'duplicate JSON object key: "a"', 7);
    expectError('{"b":{"a\\n":1,"a\\n":2}}', 'duplicate JSON object key: "a\\n"', 14);
    expectError('"\\ud800"', "string has an unpaired surrogate", 0);
    expectError('"\\ud800A"', "string has an unpaired surrogate", 0);
    expectError('"x\\udc00"', "string has an unpaired surrogate", 0);
    expectError('"\ud800"', "string has an unpaired surrogate", 0);
    expectError('[1, "\\udc00\\ud800"]', "string has an unpaired surrogate", 4);
    expectError('{"\\ud800":1}', "object key has an unpaired surrogate", 1);
  });

  test("turns a RangeError from BigInt into a defined error", () => {
    const original = globalThis.BigInt;
    globalThis.BigInt = (() => {
      throw new RangeError("Maximum BigInt size exceeded");
    }) as unknown as BigIntConstructor;
    try {
      expectError("[1.5, 12]", "integer is too large for this runtime", 6);
    } finally {
      globalThis.BigInt = original;
    }
  });

  test("reads any depth", () => {
    const n = 100_000;
    expect(depth(parseJson("[".repeat(n) + "]".repeat(n), "plain"))).toBe(n);
    const objects = parseJson('{"a":'.repeat(n) + "1" + "}".repeat(n), "plain");
    expect(depth(objects)).toBe(n);
    let leaf: JsonValue = objects;
    for (let i = 0; i < n; i++) leaf = (leaf as Record<string, JsonValue>).a!;
    expect(leaf).toBe(1n);
  });
});
