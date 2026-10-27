/** Tests for the lossless JSON reader and the canonical JSON encoders in `src/typesys/json.ts`. */

import { describe, expect, test } from "bun:test";

import { canonicalJson, JSONDecodeError, logJson, parseJson, type JsonValue } from "../../src/typesys/json.ts";

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
    expectError('{"a":1,"a":2,"b":1,"b":2}', 'duplicate JSON object key: "a"', 7);
    expectError('{"__proto__":1,"__proto__":2}', 'duplicate JSON object key: "__proto__"', 15);
    expectError('{"a":1,"a":{"b":1,"b":2}}', 'duplicate JSON object key: "b"', 18);
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

  test("decodes the codec form in codec mode only", () => {
    const text = '[{"$float":"inf"},{"$float":"-inf"},{"$float":"nan"},{"$float":"inf","b":1}]';
    expect(parseJson(text, "codec")).toEqual([Infinity, -Infinity, { $float: "nan" }, { $float: "inf", b: 1n }]);
    expect(parseJson(text, "plain")).toEqual([
      { $float: "inf" },
      { $float: "-inf" },
      { $float: "nan" },
      { $float: "inf", b: 1n },
    ]);
  });

  test("unwraps $record escapes in codec mode only", () => {
    const cases: [string, JsonValue, JsonValue][] = [
      ['{"$record":{"$float":"inf"}}', { $float: "inf" }, { $record: { $float: "inf" } }],
      ['{"$record":{"$record":{"$float":"inf"}}}', { $record: Infinity }, { $record: { $record: { $float: "inf" } } }],
      ['{"$record":{"$record":5}}', { $record: 5n }, { $record: { $record: 5n } }],
      ['{"$record":5}', { $record: 5n }, { $record: 5n }],
      ['{"$record":null}', { $record: null }, { $record: null }],
      ['{"$record":[1]}', { $record: [1n] }, { $record: [1n] }],
      ['{"$record":{"$record":{"a":1}}}', { $record: { a: 1n } }, { $record: { $record: { a: 1n } } }],
      ['{"$record":{}}', {}, { $record: {} }],
      [
        '{"$record":{"a":1},"b":{"$float":"inf"}}',
        { $record: { a: 1n }, b: Infinity },
        { $record: { a: 1n }, b: { $float: "inf" } },
      ],
    ];
    for (const [text, codec, plain] of cases) {
      expect(parseJson(text, "codec")).toEqual(codec);
      expect(parseJson(text, "plain")).toEqual(plain);
    }
  });

  test("decodes at any depth", () => {
    const n = 100_000;
    let node = parseJson('{"$record":'.repeat(n) + '{"a":1}' + "}".repeat(n), "codec");
    // Each escape pair is one record level; walk them rather than compare a deep value.
    let levels = 0;
    while (typeof node === "object" && node !== null && Object.keys(node).join() === "$record") {
      levels++;
      node = (node as Record<string, JsonValue>).$record!;
    }
    expect(levels).toBe(n / 2);
    expect(node).toEqual({ a: 1n });
  });
});

const ENCODERS: [string, (value: unknown) => string][] = [
  ["canonicalJson", canonicalJson],
  ["logJson", logJson],
];

/** Expect `encode(value)` to throw a `TypeError` with `message`. */
function expectRefusal(encode: (value: unknown) => string, value: unknown, message: string): void {
  let caught: unknown;
  try {
    encode(value);
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(TypeError);
  expect((caught as TypeError).message).toBe(message);
}

/** Return an object holding `key` as own data, even when `key` is `__proto__`. */
function own(key: string, value: unknown): Record<string, unknown> {
  const object: Record<string, unknown> = {};
  Object.defineProperty(object, key, { value, writable: true, enumerable: true, configurable: true });
  return object;
}

describe("canonicalJson", () => {
  test("writes JCS", () => {
    // U+10000 is the code units D800 DC00, so it sorts before U+E000 though its code point is higher.
    const keys = { "\uE000": 1n, "\u{10000}": 2n, b: 3n, a: 4n, "": 5n };
    expect(canonicalJson(keys)).toBe('{"":5,"a":4,"b":3,"\u{10000}":2,"\uE000":1}');
    const text = '\u0000\u001f\b\f\n\r\t"\\/\u007f é\u{1F600}';
    expect(canonicalJson(text)).toBe('"\\u0000\\u001f\\b\\f\\n\\r\\t\\"\\\\/\u007f é\u{1F600}"');
    expect(canonicalJson(null)).toBe("null");
    expect(canonicalJson(true)).toBe("true");
    expect(canonicalJson(false)).toBe("false");
    expect(canonicalJson([])).toBe("[]");
    expect(canonicalJson({})).toBe("{}");
    expect(canonicalJson({ b: [1n, { d: [], c: {} }], a: "x" })).toBe('{"a":"x","b":[1,{"c":{},"d":[]}]}');
    expect(canonicalJson([0n, -5n, 9007199254740991n, -9007199254740991n, 56])).toBe(
      "[0,-5,9007199254740991,-9007199254740991,56]",
    );
    const bare = Object.create(null) as Record<string, unknown>;
    bare.k = "v";
    expect(canonicalJson(bare)).toBe('{"k":"v"}');
  });

  test("refuses non-int numbers in JCS", () => {
    for (const value of [1.5, 9007199254740992n, -9007199254740992n, 2 ** 53, 1e300]) {
      expectRefusal(canonicalJson, value, "JCS numbers are ints within ±(2^53-1)");
    }
  });
});

describe("logJson", () => {
  test("writes the log number form", () => {
    const value = [1n, 1, 100, 1e21, 0.5, 1e-7, 2 ** 53, 5e-324, -1.5, 123456789012345678901234567890n];
    expect(logJson(value)).toBe(
      "[1,1.0,100.0,1e+21,0.5,1e-7,9007199254740992.0,5e-324,-1.5,123456789012345678901234567890]",
    );
  });

  test("round-trips the log form in codec mode", () => {
    const value = {
      ints: [0n, -1n, 2n ** 64n, -(2n ** 70n)],
      floats: [1, -2, 0.5, 2 ** 53, 1e21, Infinity, -Infinity],
      lookAlikes: [
        { $float: "inf" },
        { $float: "-inf" },
        { $record: 1n },
        { $record: "x" },
        { $record: null },
        { $record: [1n] },
        { $record: { $record: {} } },
        { $record: { $record: Infinity } },
      ],
      proto: own("__proto__", { x: 1n }),
      text: "café \u{1F600} \u{10000}",
      nested: [[[{ a: [true, false, null] }]]],
    };
    const back = parseJson(logJson(value), "codec");
    // One walk checks every leaf's kind and value (`toBe` is `Object.is`, so 1 isn't 1n), each
    // container's kind and prototype, and the key order.
    const pairs: [unknown, unknown][] = [[value, back]];
    while (pairs.length > 0) {
      const [want, got] = pairs.pop()!;
      if (typeof want !== "object" || want === null) {
        expect(got).toBe(want);
        continue;
      }
      expect(typeof got === "object" && got !== null).toBe(true);
      expect(Array.isArray(got)).toBe(Array.isArray(want));
      expect(Object.getPrototypeOf(got)).toBe(Object.getPrototypeOf(want));
      // The reader keeps the written order, which is sorted by code units.
      expect(Object.keys(got as object)).toEqual(Object.keys(want).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)));
      for (const key of Object.keys(want)) {
        pairs.push([(want as Record<string, unknown>)[key], (got as Record<string, unknown>)[key]]);
      }
    }
  });

  test("turns a RangeError from a bigint's toString into a defined error", () => {
    const original = BigInt.prototype.toString;
    BigInt.prototype.toString = () => {
      throw new RangeError("Maximum BigInt size exceeded");
    };
    try {
      expectRefusal(logJson, [1n], "integer is too large for this runtime");
    } finally {
      BigInt.prototype.toString = original;
    }
  });

  test("logJson equals canonicalJson without floats and big ints", () => {
    const value = { z: [1n, -2n, "téxt\n", true, false, null], a: { "": [], b: {} } };
    expect(logJson(value)).toBe(canonicalJson(value));
  });
});

describe.each(ENCODERS)("%s", (_name, encode) => {
  test("writes infinity as the tag in both forms", () => {
    expect(encode(Infinity)).toBe('{"$float":"inf"}');
    expect(encode(-Infinity)).toBe('{"$float":"-inf"}');
    expect(encode({ a: [Infinity, -Infinity] })).toBe('{"a":[{"$float":"inf"},{"$float":"-inf"}]}');
  });

  test("escapes look-alike records in both forms", () => {
    expect(encode({ $float: "inf" })).toBe('{"$record":{"$float":"inf"}}');
    expect(encode({ $float: "-inf" })).toBe('{"$record":{"$float":"-inf"}}');
    expect(encode({ $record: 1n })).toBe('{"$record":{"$record":1}}');
    expect(encode({ $record: { $record: {} } })).toBe('{"$record":{"$record":{"$record":{"$record":{}}}}}');
    expect(encode({ $float: "nan" })).toBe('{"$float":"nan"}');
    expect(encode({ $float: "inf", b: 1n })).toBe('{"$float":"inf","b":1}');
  });

  test("refuses what isn't canonical JSON in both forms", () => {
    expectRefusal(encode, NaN, "NaN is not valid canonical JSON");
    expectRefusal(encode, [-0], "negative zero is not valid canonical JSON");
    expectRefusal(encode, "a\ud800", "Unicode surrogate is not valid canonical JSON");
    expectRefusal(encode, { "\udc00": 1n }, "Unicode surrogate is not valid canonical JSON");
    expectRefusal(encode, undefined, "undefined is not a canonical JSON value");
    expectRefusal(encode, [1n, , 2n], "undefined is not a canonical JSON value");
    expectRefusal(encode, { a: undefined }, "undefined is not a canonical JSON value");
    expectRefusal(encode, Symbol("s"), "symbol is not a canonical JSON value");
    expectRefusal(encode, [() => 1], "function is not a canonical JSON value");
    expectRefusal(encode, new Date(0), "Date is not a canonical JSON value");
    expectRefusal(encode, new Map(), "Map is not a canonical JSON value");
    expectRefusal(encode, new Uint8Array(1), "Uint8Array is not a canonical JSON value");
    expectRefusal(encode, new (class FileRef {})(), "FileRef is not a canonical JSON value");
    expectRefusal(encode, Object.create({}), "Object is not a canonical JSON value");
    expectRefusal(encode, Object.create(Object.create(null)), "object is not a canonical JSON value");
    expectRefusal(encode, { a: 1n, [Symbol("k")]: 2n }, "canonical JSON object keys must be strings");
  });

  test("reads each object value once", () => {
    let reads = 0;
    const tag = {
      get $float() {
        reads++;
        return reads === 1 ? "inf" : "nan";
      },
    };
    expect(encode(tag)).toBe('{"$record":{"$float":"inf"}}');
    expect(reads).toBe(1);
  });

  test("refuses cycles and writes shared references", () => {
    const a: unknown[] = [];
    a.push(a);
    expectRefusal(encode, a, "cyclic container cannot be canonicalized");
    const o: Record<string, unknown> = {};
    o.self = { back: o };
    expectRefusal(encode, o, "cyclic container cannot be canonicalized");
    const s = { x: 1n };
    expect(encode([s, s])).toBe('[{"x":1},{"x":1}]');
    expect(encode({ a: s, b: [s] })).toBe('{"a":{"x":1},"b":[{"x":1}]}');
  });

  test("writes any depth", () => {
    const n = 100_000;
    let value: unknown[] = [];
    for (let i = 1; i < n; i++) value = [value];
    expect(encode(value)).toBe("[".repeat(n) + "]".repeat(n));
  });
});
