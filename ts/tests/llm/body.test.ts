/** Tests for merging a provider block into a request body and encoding the body in `src/llm/body.ts`. */

import { describe, expect, test } from "bun:test";

import { encodeBody, mergeRequestBody } from "../../src/llm/body.ts";
import { type JsonObject, type JsonValue, parseJson } from "../../src/typesys/json.ts";

/** Return the plain JSON object `text` holds, so a `"__proto__"` key stays an ordinary key. */
function plain(text: string): JsonObject {
  return parseJson(text, "plain") as JsonObject;
}

/** Return the error `run` throws, failing when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Assert `encodeBody(value)` throws an unchained `TypeError` with `message`. */
function refuses(value: unknown, message: string): void {
  const error = thrown(() => encodeBody(value));
  expect(error).toBeInstanceOf(TypeError);
  expect(error.message).toBe(message);
  expect(error.cause).toBeUndefined();
}

describe("mergeRequestBody", () => {
  test("mappings merge key by key", () => {
    const body = plain('{"text": {"format": {"type": "text"}}}');
    mergeRequestBody(body, plain('{"text": {"verbosity": "low"}}'));
    expect(body).toEqual(plain('{"text": {"format": {"type": "text"}, "verbosity": "low"}}'));
  });

  test("lists and scalars replace whole, sibling keys stay, and it returns body", () => {
    const body = plain('{"model": "gpt-5", "text": {"format": {"type": "json_schema"}}, "stop": ["a", "b"], "t": 1.5}');
    const result = mergeRequestBody(body, plain('{"text": {"verbosity": "low"}, "stop": ["c"], "t": 0.2}'));
    expect(result).toBe(body);
    expect(body).toEqual(
      plain('{"model": "gpt-5", "text": {"format": {"type": "json_schema"}, "verbosity": "low"}, "stop": ["c"], "t": 0.2}'),
    );
  });

  test("null leaves the body alone, at the top level and nested", () => {
    const body = plain('{"temperature": 1.5, "text": {"verbosity": "high"}}');
    mergeRequestBody(body, plain('{"temperature": null, "text": {"verbosity": null}, "top_k": null}'));
    expect(body).toEqual(plain('{"temperature": 1.5, "text": {"verbosity": "high"}}'));
  });

  test("the body never shares a container with the block", () => {
    const block = plain('{"metadata": {"a": "b"}, "include": ["x", {"y": 1}]}');
    const body: JsonObject = {};
    mergeRequestBody(body, block);
    (block["metadata"] as JsonObject)["a"] = "changed";
    const include = block["include"] as JsonValue[];
    (include[1] as JsonObject)["y"] = 2n;
    include.push("z");
    expect(body).toEqual(plain('{"metadata": {"a": "b"}, "include": ["x", {"y": 1}]}'));
  });

  test("an object replaces a scalar, and a scalar replaces an object", () => {
    const body = plain('{"a": 1, "b": {"c": 2}}');
    mergeRequestBody(body, plain('{"a": {"x": true}, "b": "s"}'));
    expect(body).toEqual(plain('{"a": {"x": true}, "b": "s"}'));
  });

  test("an object of nulls replacing a scalar leaves an empty object", () => {
    const body = plain('{"a": 1}');
    mergeRequestBody(body, plain('{"a": {"b": null}}'));
    expect(body).toEqual(plain('{"a": {}}'));
  });

  test("a null inside a copied array, or an object within it, is kept", () => {
    const body: JsonObject = {};
    mergeRequestBody(body, plain('{"a": [1, null, {"b": null}]}'));
    expect(body).toEqual(plain('{"a": [1, null, {"b": null}]}'));
  });

  test("a block key __proto__ becomes an own key of the body, at the top level and nested", () => {
    const body = plain('{"n": {"k": 1}}');
    mergeRequestBody(body, plain('{"__proto__": {"x": 1}, "n": {"__proto__": [2]}}'));
    expect(Object.getPrototypeOf(body)).toBe(Object.prototype);
    expect(Object.hasOwn(body, "__proto__")).toBe(true);
    expect(Object.keys(body)).toEqual(["n", "__proto__"]);
    const nested = body["n"] as JsonObject;
    expect(Object.getPrototypeOf(nested)).toBe(Object.prototype);
    expect(Object.keys(nested)).toEqual(["k", "__proto__"]);
    expect(encodeBody(body)).toBe('{"n":{"k":1,"__proto__":[2]},"__proto__":{"x":1}}');
  });

  test("a body key __proto__ is merged into", () => {
    const body = plain('{"__proto__": {"a": 1}}');
    mergeRequestBody(body, plain('{"__proto__": {"b": 2}}'));
    expect(Object.getPrototypeOf(body)).toBe(Object.prototype);
    expect(encodeBody(body)).toBe('{"__proto__":{"a":1,"b":2}}');
  });
});

describe("encodeBody", () => {
  test("keeps insertion order", () => {
    expect(encodeBody({ z: 1, a: [true, false, null], m: { y: "", b: 2 } })).toBe(
      '{"z":1,"a":[true,false,null],"m":{"y":"","b":2}}',
    );
  });

  test("writes floats, bigints and non-ASCII text exactly", () => {
    expect(encodeBody({ n: 1.5, i: 12345678901234567890n, s: "héllo ✓" })).toBe(
      '{"n":1.5,"i":12345678901234567890,"s":"héllo ✓"}',
    );
  });

  test("writes -0 as 0 and leaves out an undefined property", () => {
    expect(encodeBody({ a: -0, b: undefined, c: [-0] })).toBe('{"a":0,"c":[0]}');
    expect(encodeBody({ b: undefined })).toBe("{}");
  });

  test("refuses NaN and infinities, nested too", () => {
    refuses({ x: NaN }, "request body is not JSON: NaN is not a JSON number");
    refuses({ x: Infinity }, "request body is not JSON: Infinity is not a JSON number");
    refuses({ x: [{ y: -Infinity }] }, "request body is not JSON: -Infinity is not a JSON number");
  });

  test("refuses an unpaired surrogate in a value or a key, with its position in the output", () => {
    refuses({ text: "a\ud800b" }, "request body has an unpaired surrogate at position 10");
    refuses({ text: "a\udc00b" }, "request body has an unpaired surrogate at position 10");
    // The escaped prefix counts: `\n` is two characters of output, and a paired surrogate two code units.
    refuses({ x: 1, "k\n😀\udc00": 2 }, "request body has an unpaired surrogate at position 13");
    refuses(["\ud800"], "request body has an unpaired surrogate at position 2");
  });

  test("refuses values that are not JSON", () => {
    class Point {}
    refuses({ x: new Map() }, "request body is not JSON: Map is not a JSON value");
    refuses({ x: new Date(0) }, "request body is not JSON: Date is not a JSON value");
    refuses({ x: new Point() }, "request body is not JSON: Point is not a JSON value");
    refuses({ x: () => 1 }, "request body is not JSON: function is not a JSON value");
    refuses({ x: Symbol("s") }, "request body is not JSON: symbol is not a JSON value");
    refuses({ x: [1, undefined] }, "request body is not JSON: undefined is not a JSON value");
    refuses({ [Symbol("k")]: 1 }, "request body is not JSON: an object has a symbol key");
  });

  test("refuses a cycle but writes a shared value twice", () => {
    const cyclic: Record<string, unknown> = { a: 1 };
    cyclic["self"] = { inner: cyclic };
    refuses(cyclic, "request body is not JSON: it contains a cycle");
    const shared = [1];
    expect(encodeBody({ a: shared, b: shared })).toBe('{"a":[1],"b":[1]}');
  });

  test("what changes the prefix", () => {
    expect(thrown(() => encodeBody({ x: NaN }, "tool arguments")).message).toBe(
      "tool arguments is not JSON: NaN is not a JSON number",
    );
    expect(thrown(() => encodeBody("\ud800", "tool arguments")).message).toBe(
      "tool arguments has an unpaired surrogate at position 1",
    );
  });

  test("encodes a 100,000-deep array", () => {
    let deep: unknown[] = [];
    for (let i = 0; i < 99_999; i++) deep = [deep];
    expect(encodeBody(deep)).toBe("[".repeat(100_000) + "]".repeat(100_000));
  });

  test("round-trips through parseJson", () => {
    const sample = plain('{"model": "m", "n": [1, 2.5, -3e-7, null, true], "o": {"s": "é\\n\\"", "e": {}, "a": []}}');
    expect(parseJson(encodeBody(sample), "plain")).toEqual(sample);
  });
});
