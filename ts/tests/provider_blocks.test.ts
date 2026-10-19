/**
 * Mirrors `tests/test_provider_blocks.py`. Not ported: the SDK-signature tests
 * (`test_responses_fields_and_reserved_keys_are_keywords_of_responses_create`,
 * `test_responses_nested_fields_are_keys_of_the_sdk_params`,
 * `test_ollama_block_fields_are_keywords_of_client_chat`; TS has no SDKs) and the
 * `merge_request_body` tests (`llm/body.ts`, T5-2, `tests/llm/body.test.ts`; TS has no
 * `top_level`/`extra_body`).
 */

import { describe, expect, test } from "bun:test";

import { FileRef } from "../src/typesys/index.ts";
import {
  checkObject,
  type Field,
  type Issue,
  type ObjectSchema,
  ValidationError,
} from "../src/llm_clients/provider_blocks.ts";

/** Return a frozen schema with `fields`, of which `required` are required. */
function schema(fields: Record<string, Field>, required: string[] = []): ObjectSchema {
  return Object.freeze({ fields: Object.freeze(fields), required: Object.freeze(required) });
}

/** Return the `{loc, type, msg}` of each issue `checkObject` finds in `data`. */
function issues(of: ObjectSchema, data: unknown): Pick<Issue, "loc" | "type" | "msg">[] {
  const result = checkObject(of, data);
  if (result.issues.length > 0) expect(result.value).toBeNull();
  return result.issues.map(({ loc, type, msg }) => ({ loc, type, msg }));
}

/** Return the value `checkObject` gives for `data`, asserting there are no issues. */
function value(of: ObjectSchema, data: unknown): unknown {
  const result = checkObject(of, data);
  expect(result.issues).toEqual([]);
  return result.value;
}

const STR: Field = { kind: "str" };
const INT: Field = { kind: "int" };

describe("value pass", () => {
  test("str takes a string and checks short, long, then pattern, in code points", () => {
    const s = schema({ v: { kind: "str", minLength: 6, maxLength: 512, pattern: "^[a-z]+$" } });
    expect(value(s, { v: "abcdefg" })).toEqual({ v: "abcdefg" });
    expect(issues(s, { v: 5n })).toEqual([{ loc: ["v"], type: "string_type", msg: "Input should be a valid string" }]);
    expect(issues(s, { v: "!".repeat(600) })).toEqual([
      { loc: ["v"], type: "string_too_long", msg: "String should have at most 512 characters" },
    ]);
    expect(issues(s, { v: "!" })).toEqual([
      { loc: ["v"], type: "string_too_short", msg: "String should have at least 6 characters" },
    ]);
    expect(issues(s, { v: "abcdef!" })).toEqual([
      { loc: ["v"], type: "string_pattern_mismatch", msg: "String should match pattern '^[a-z]+$'" },
    ]);
    const points = schema({ v: { kind: "str", minLength: 2, maxLength: 2 } });
    expect(value(points, { v: "😀😀" })).toEqual({ v: "😀😀" });
    expect(issues(points, { v: "😀" })).toEqual([
      { loc: ["v"], type: "string_too_short", msg: "String should have at least 2 characters" },
    ]);
  });

  test("int takes only a bigint", () => {
    const s = schema({ v: INT });
    expect(value(s, { v: 4096n })).toEqual({ v: 4096n });
    for (const bad of [4096, true, "5"]) {
      expect(issues(s, { v: bad })).toEqual([{ loc: ["v"], type: "int_type", msg: "Input should be a valid integer" }]);
    }
  });

  test("float takes a finite number and keeps a bigint as is", () => {
    const s = schema({ v: { kind: "float" } });
    const kept = value(s, { v: 1n }) as { v: unknown };
    expect(typeof kept.v).toBe("bigint");
    expect(kept.v).toBe(1n);
    expect(value(s, { v: 0.5 })).toEqual({ v: 0.5 });
    for (const bad of [true, "1", 10n ** 400n]) {
      expect(issues(s, { v: bad })).toEqual([
        { loc: ["v"], type: "float_type", msg: "Input should be a valid number" },
      ]);
    }
    for (const bad of [Infinity, -Infinity, NaN]) {
      expect(issues(s, { v: bad })).toEqual([
        { loc: ["v"], type: "finite_number", msg: "Input should be a finite number" },
      ]);
    }
  });

  test("int and float bounds print the bound without an n", () => {
    const s = schema({
      ge: { kind: "int", ge: 1024n },
      gt: { kind: "float", gt: 0 },
      le: { kind: "int", le: 20n },
      lt: { kind: "float", lt: 2 },
      frac: { kind: "float", ge: 0.01, le: 1 },
    });
    expect(issues(s, { ge: 5n, gt: 0, le: 21n, lt: 2n, frac: 0.001 })).toEqual([
      { loc: ["ge"], type: "greater_than_equal", msg: "Input should be greater than or equal to 1024" },
      { loc: ["gt"], type: "greater_than", msg: "Input should be greater than 0" },
      { loc: ["le"], type: "less_than_equal", msg: "Input should be less than or equal to 20" },
      { loc: ["lt"], type: "less_than", msg: "Input should be less than 2" },
      { loc: ["frac"], type: "greater_than_equal", msg: "Input should be greater than or equal to 0.01" },
    ]);
    expect(value(s, { ge: 1024n, gt: 1n, le: 20n, lt: 1.5, frac: 1n })).toEqual({
      ge: 1024n,
      gt: 1n,
      le: 20n,
      lt: 1.5,
      frac: 1n,
    });
  });

  test("bool takes only a boolean", () => {
    const s = schema({ v: { kind: "bool" } });
    expect(value(s, { v: false })).toEqual({ v: false });
    for (const bad of ["true", 1n]) {
      expect(issues(s, { v: bad })).toEqual([
        { loc: ["v"], type: "bool_type", msg: "Input should be a valid boolean" },
      ]);
    }
  });

  test("literal takes one of its strings and lists them in the message", () => {
    const s = schema({
      three: { kind: "literal", values: ["a", "b", "c"] },
      two: { kind: "literal", values: ["a", "b"] },
      one: { kind: "literal", values: ["a"] },
    });
    expect(value(s, { three: "c", two: "a", one: "a" })).toEqual({ three: "c", two: "a", one: "a" });
    expect(issues(s, { three: "d", two: 1n, one: "A" })).toEqual([
      { loc: ["three"], type: "literal_error", msg: "Input should be 'a', 'b' or 'c'" },
      { loc: ["two"], type: "literal_error", msg: "Input should be 'a' or 'b'" },
      { loc: ["one"], type: "literal_error", msg: "Input should be 'a'" },
    ]);
  });

  test("list checks each item, or only its length when too long", () => {
    const s = schema({ v: { kind: "list", item: STR, maxLength: 4 } });
    expect(value(s, { v: ["a", "b"] })).toEqual({ v: ["a", "b"] });
    expect(issues(s, { v: "a" })).toEqual([{ loc: ["v"], type: "list_type", msg: "Input should be a valid list" }]);
    expect(issues(s, { v: ["a", 5n, undefined] })).toEqual([
      { loc: ["v", 1], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["v", 2], type: "string_type", msg: "Input should be a valid string" },
    ]);
    expect(issues(s, { v: [1n, 2n, 3n, 4n, 5n] })).toEqual([
      { loc: ["v"], type: "too_long", msg: "List should have at most 4 items after validation, not 5" },
    ]);
  });

  test("map checks each key then value, and its length only when they pass", () => {
    const s = schema({
      v: { kind: "map", key: { kind: "str", maxLength: 2 }, value: { kind: "str", maxLength: 3 }, maxLength: 16 },
    });
    expect(issues(s, { v: 5n })).toEqual([
      { loc: ["v"], type: "dict_type", msg: "Input should be a valid dictionary" },
    ]);
    expect(issues(s, { v: { abc: "defg", b: 1n } })).toEqual([
      { loc: ["v", "abc", "[key]"], type: "string_too_long", msg: "String should have at most 2 characters" },
      { loc: ["v", "abc"], type: "string_too_long", msg: "String should have at most 3 characters" },
      { loc: ["v", "b"], type: "string_type", msg: "Input should be a valid string" },
    ]);
    const keys = Array.from({ length: 17 }, (_, i) => `k${i.toString(36)}`);
    const bad = issues(s, { v: Object.fromEntries(keys.map((k) => [k, 1n])) });
    expect(bad).toEqual(
      keys.map((k) => ({ loc: ["v", k], type: "string_type", msg: "Input should be a valid string" })),
    );
    expect(issues(s, { v: Object.fromEntries(keys.map((k) => [k, "x"])) })).toEqual([
      { loc: ["v"], type: "too_long", msg: "Dictionary should have at most 16 items after validation, not 17" },
    ]);
  });

  test("map entries holding undefined are skipped and not counted", () => {
    const s = schema({ v: { kind: "map", key: STR, value: STR, maxLength: 2 } });
    expect(value(s, { v: { a: "x", b: undefined, c: "y" } })).toEqual({ v: { a: "x", c: "y" } });
    const result = value(s, { v: { a: "x", b: undefined, c: "y" } }) as { v: object };
    expect(Object.keys(result.v)).toEqual(["a", "c"]);
  });

  test("a __proto__ key in a map is an own key of the result", () => {
    const s = schema({ v: { kind: "map", key: STR, value: STR } });
    const result = value(s, { v: Object.fromEntries([["__proto__", "x"]]) }) as { v: object };
    expect(Object.hasOwn(result.v, "__proto__")).toBe(true);
    expect(Object.getPrototypeOf(result.v)).toBe(Object.prototype);
    expect(Object.entries(result.v)).toEqual([["__proto__", "x"]]);
  });

  test("object checks fields in declaration order, then extras in written order", () => {
    const inner = schema({ a: STR, b: INT }, ["b"]);
    const s = schema({ first: STR, nested: { kind: "object", schema: inner }, need: INT }, ["need"]);
    expect(issues(s, 5n)).toEqual([{ loc: [], type: "model_type", msg: "Input should be a valid dictionary" }]);
    expect(issues(s, ["a"])).toEqual([{ loc: [], type: "model_type", msg: "Input should be a valid dictionary" }]);
    expect(issues(s, { zz: 1n, nested: { a: 1n, x: 1n }, toString: 1n, first: 1n })).toEqual([
      { loc: ["first"], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["nested", "a"], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["nested", "b"], type: "missing", msg: "Field required" },
      { loc: ["nested", "x"], type: "extra_forbidden", msg: "Extra inputs are not permitted" },
      { loc: ["need"], type: "missing", msg: "Field required" },
      { loc: ["zz"], type: "extra_forbidden", msg: "Extra inputs are not permitted" },
      { loc: ["toString"], type: "extra_forbidden", msg: "Extra inputs are not permitted" },
    ]);
    expect(issues(s, { nested: "x", need: 1n })).toEqual([
      { loc: ["nested"], type: "model_type", msg: "Input should be a valid dictionary" },
    ]);
    // A required field given null is checked against its type; an optional one is unset.
    expect(issues(s, { first: null, need: null })).toEqual([
      { loc: ["need"], type: "int_type", msg: "Input should be a valid integer" },
    ]);
    expect(value(s, { first: null, nested: { a: null, b: 1n }, need: 2n })).toEqual({ nested: { b: 1n }, need: 2n });
  });

  test("undefined reads as absent", () => {
    const s = schema({ a: STR, need: INT }, ["need"]);
    expect(issues(s, { a: undefined, need: undefined, extra: undefined })).toEqual([
      { loc: ["need"], type: "missing", msg: "Field required" },
    ]);
    const result = value(s, { a: undefined, need: 1n, extra: undefined }) as object;
    expect(Object.keys(result)).toEqual(["need"]);
  });

  test("union takes the first member that passes and labels each member's issues", () => {
    const s = schema({
      stop: {
        kind: "union",
        members: [STR, { kind: "list", item: STR, maxLength: 4 }, { kind: "list", item: INT }],
      },
      keep: { kind: "union", members: [{ kind: "float" }, STR] },
      think: { kind: "union", members: [{ kind: "bool" }, { kind: "literal", values: ["low", "high"] }] },
    });
    const good = { stop: [1n, 2n], keep: "5m", think: "low" };
    expect(value(s, good)).toEqual(good);
    expect(issues(s, { stop: ["a", 5n], keep: true, think: 5n })).toEqual([
      { loc: ["stop", "str"], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["stop", "list[str]", 1], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["stop", "list[int]", 0], type: "int_type", msg: "Input should be a valid integer" },
      { loc: ["keep", "float"], type: "float_type", msg: "Input should be a valid number" },
      { loc: ["keep", "str"], type: "string_type", msg: "Input should be a valid string" },
      { loc: ["think", "bool"], type: "bool_type", msg: "Input should be a valid boolean" },
      { loc: ["think", "literal['low','high']"], type: "literal_error", msg: "Input should be 'low' or 'high'" },
    ]);
  });

  test("tagged picks the variant its type names, never echoing an unknown tag", () => {
    const enabled = schema({ type: { kind: "literal", values: ["enabled"] }, budget: { kind: "int", ge: 1024n } }, [
      "type",
      "budget",
    ]);
    const disabled = schema({ type: { kind: "literal", values: ["disabled"] } }, ["type"]);
    const s = schema({ thinking: { kind: "tagged", variants: Object.freeze({ enabled, disabled }) } });
    const unknown = { loc: ["thinking"], type: "unknown_type", msg: "`type` must be one of: disabled, enabled" };
    const bads = [5n, ["enabled"], {}, { type: "other" }, { type: "constructor" }, { type: 1n }, { type: undefined }];
    for (const bad of bads) {
      expect(issues(s, { thinking: bad })).toEqual([unknown]);
    }
    expect(issues(s, { thinking: { type: "enabled" } })).toEqual([
      { loc: ["thinking", "enabled", "budget"], type: "missing", msg: "Field required" },
    ]);
    expect(issues(s, { thinking: { type: "enabled", budget: 5n } })).toEqual([
      {
        loc: ["thinking", "enabled", "budget"],
        type: "greater_than_equal",
        msg: "Input should be greater than or equal to 1024",
      },
    ]);
    expect(value(s, { thinking: { type: "disabled" } })).toEqual({ thinking: { type: "disabled" } });
  });

  test("json takes JSON values at any depth and refuses the rest at the inner location", () => {
    const s = schema({ v: { kind: "map", key: STR, value: { kind: "json" } } });
    const ok = { a: null, b: [true, "x", 1n, 0.5, { c: [] }], d: { e: null } };
    expect(value(s, { v: ok })).toEqual({ v: ok });
    expect(issues(s, { v: { a: [Infinity] } })).toEqual([
      { loc: ["v", "a", 0], type: "finite_number", msg: "Input should be a finite number" },
    ]);
    const file = new FileRef({ uri: "file:///x.txt" });
    expect(issues(s, { v: { a: { b: NaN }, c: [undefined], d: file, e: () => 1, f: new Map() } })).toEqual([
      { loc: ["v", "a", "b"], type: "finite_number", msg: "Input should be a finite number" },
      { loc: ["v", "c", 0], type: "json_type", msg: "Input should be a JSON value" },
      { loc: ["v", "d"], type: "json_type", msg: "Input should be a JSON value" },
      { loc: ["v", "e"], type: "json_type", msg: "Input should be a JSON value" },
      { loc: ["v", "f"], type: "json_type", msg: "Input should be a JSON value" },
    ]);
  });

  test("json record entries holding undefined are skipped and not counted", () => {
    const s = schema({ v: { kind: "map", key: STR, value: { kind: "json" }, maxLength: 1 } });
    const result = value(s, { v: { a: { b: undefined, c: 1n }, d: undefined } }) as { v: { a: object } };
    expect(result).toEqual({ v: { a: { c: 1n } } });
    expect(Object.keys(result.v.a)).toEqual(["c"]);
  });

  test("the result is a fresh copy and the input is left as it was", () => {
    const s = schema({
      nested: { kind: "object", schema: schema({ a: STR, b: STR }) },
      list: { kind: "list", item: STR },
      map: { kind: "map", key: STR, value: { kind: "json" } },
    });
    const input = { nested: { a: "x", b: null }, list: ["y"], map: { k: { deep: [1n] } } };
    const before = structuredClone(input);
    const result = value(s, input) as typeof input;
    expect(input).toEqual(before);
    expect(result).toEqual({ nested: { a: "x" }, list: ["y"], map: { k: { deep: [1n] } } } as unknown as typeof input);
    expect(result.nested).not.toBe(input.nested);
    expect(result.list).not.toBe(input.list);
    expect(result.map).not.toBe(input.map);
    expect(result.map.k).not.toBe(input.map.k);
    expect(result.map.k.deep).not.toBe(input.map.k.deep);
  });
});

describe("ValidationError", () => {
  test("the message has a count line, dotted locations and two-space messages", () => {
    const error = new ValidationError("SomeBlock", [
      { loc: ["m", "abc", "[key]"], type: "string_too_long", msg: "String should have at most 2 characters" },
      { loc: ["stop", "list[str]", 1], type: "string_type", msg: "Input should be a valid string" },
      { loc: [], type: "value_error", msg: "Value error, bad", error: "bad" },
    ]);
    expect(error.message).toBe(
      "3 validation errors for SomeBlock\n" +
        "m.abc.[key]\n  String should have at most 2 characters\n" +
        "stop.list[str].1\n  Input should be a valid string\n" +
        "  Value error, bad",
    );
    expect(error.name).toBe("ValidationError");
    expect(error).toBeInstanceOf(RangeError);
    expect(error.title).toBe("SomeBlock");
    expect(Object.isFrozen(error.issues)).toBe(true);
    expect(error.issues[2]!.error).toBe("bad");
    const one = new ValidationError("B", [{ loc: ["a"], type: "missing", msg: "Field required" }]);
    expect(one.message).toBe("1 validation error for B\na\n  Field required");
  });
});
