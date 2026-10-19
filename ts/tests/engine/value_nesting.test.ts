/**
 * How deep a value may nest and which entry of a value refers back to a container enclosing
 * it, a port of `tests/engine/test_value_nesting.py` (`rejectDeepNesting`,
 * `recursiveEntryPath`). The levels are arrays, records and `FileRef`s; a `TypedValue` is
 * unwrapped.
 *
 * New in TS: `rejectLoneSurrogates`, which refuses text or a record key holding a lone
 * surrogate on the same walk.
 *
 * Also here: `test_limit_overrides.py::test_value_nesting_follows_the_variable`.
 *
 * Not ported: `test_tuple_dataclass_and_model_each_count_as_a_level` and
 * `test_model_extras_are_direct_children` (no tuples, dataclasses or pydantic models in TS),
 * and the `reject_non_text_keys` tests (a record key is always a string, so it has no TS
 * counterpart).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install, reload } from "../../src/defaults.ts";
import {
  buildValue,
  FileRef,
  recursiveEntryPath,
  rejectDeepNesting,
  rejectLoneSurrogates,
  TypeCheckError,
  TypedValue,
  type Value,
  ValueKind,
} from "../../src/typesys/values.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

const asRecord = (value: Value): Value => ({ a: value });
const asList = (value: Value): Value => [value];

/** Return `leaf` wrapped in `n` levels of `wrap`. */
function nest(n: number, wrap = asRecord, leaf: Value = 1n): Value {
  let value = leaf;
  for (let i = 0; i < n; i++) value = wrap(value);
  return value;
}

/** The message for a value too deep at `path` under `limit`, at `output`. */
function message(path: string, limit = 100): string {
  return (
    `output at ${path}: value nests deeper than ${limit} levels, ` +
    "the limit for a yian value (each record and list is a level)"
  );
}

/** Return the `TypeCheckError` `call()` throws. */
function typeCheckError(call: () => unknown): TypeCheckError {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("expected a TypeCheckError");
}

/** Return the error `rejectDeepNesting(value, "output", {enclosingLevels})` throws. */
function deepError(value: unknown, enclosingLevels = 0): TypeCheckError {
  return typeCheckError(() => rejectDeepNesting(value, "output", { enclosingLevels }));
}

const repeat = <T>(segment: T, n: number): T[] => Array<T>(n).fill(segment);

describe("rejectDeepNesting", () => {
  test("the limit is 100", () => {
    expect(BUILTIN.maxValueNesting).toBe(100);
  });

  test.each<[string, Value]>([
    ["an int", 1n],
    ["text", "s"],
    ["null", null],
    ["an empty record", {}],
    ["an empty list", []],
    ["100 records", nest(100)],
    ["100 lists", nest(100, asList)],
  ])("values within the limit pass: %s", (_id, value) => {
    rejectDeepNesting(value, "output");
  });

  test("a record chain past the limit names the path", () => {
    const error = deepError(nest(101));
    expect(error.path).toEqual(repeat("a", 100));
    expect(error.message).toBe(message(repeat("a", 100).join(".")));
  });

  test("a list chain past the limit names the path", () => {
    const error = deepError(nest(101, asList));
    expect(error.path).toEqual(repeat(0, 100));
    expect(error.message).toBe(message("[0]".repeat(100)));
  });

  test("enclosing levels count toward the limit", () => {
    expect(deepError(nest(100), 1).path).toEqual(repeat("a", 99));
  });

  test("negative enclosing levels let the value go one over", () => {
    rejectDeepNesting(nest(101), "output", { enclosingLevels: -1 });
    expect(deepError(nest(102), -1).path).toEqual(repeat("a", 101));
  });

  test("mixed path segments", () => {
    const error = deepError({ xs: nest(100, asList) });
    expect(error.path).toEqual(["xs", ...repeat(0, 99)]);
    expect(error.message).toBe(message("xs" + "[0]".repeat(99)));
  });

  test("a key that isn't an identifier renders quoted", () => {
    const error = deepError({ "my key": nest(100) });
    expect(error.path).toEqual(["my key", ...repeat("a", 99)]);
    expect(error.message).toBe(message('["my key"].' + repeat("a", 99).join(".")));
  });

  test("a FileRef at the last level passes", () => {
    const file = new FileRef({ uri: "file:///x" });
    rejectDeepNesting(nest(99, asRecord, file), "output");
    deepError(nest(100, asRecord, file));
  });

  test("a TypedValue adds no level", () => {
    rejectDeepNesting(buildValue(nest(100)), "output");
    expect(deepError(buildValue(nest(101))).path).toEqual(repeat("a", 100));
  });

  test("the first offender, depth first and in order", () => {
    const error = deepError({ ok: 1n, deep: nest(101), deeper: nest(150) });
    expect(error.path).toEqual(["deep", ...repeat("a", 99)]);
  });

  test("a very deep value does not recurse", () => {
    const deep = nest(100_000, asList);
    deepError(deep);
    // The variable's range stops at 100, so the raised limit is installed unchecked.
    install({ ...BUILTIN, maxValueNesting: 200_000 });
    rejectDeepNesting(deep, "output");
  });

  test("cycles are not reported as too deep", () => {
    const record: { [key: string]: Value } = {};
    record["self"] = record;
    rejectDeepNesting(record, "output");
    const items: Value[] = [];
    items.push(items);
    rejectDeepNesting(items, "output");
  });

  test("the limit is read at call time", () => {
    reload((name) => (name === "YIAN_MAX_VALUE_NESTING" ? "3" : undefined));
    rejectDeepNesting(nest(3), "output");
    const error = deepError(nest(4));
    expect(error.path).toEqual(["a", "a", "a"]);
    expect(error.message).toBe(message("a.a.a", 3));
  });

  // From test_limit_overrides.py::test_value_nesting_follows_the_variable.
  test("value nesting follows the variable", () => {
    reload((name) => (name === "YIAN_MAX_VALUE_NESTING" ? "3" : undefined));
    rejectDeepNesting(nest(3), "output");
    const error = deepError(nest(4));
    expect(error.message.startsWith("output at a.a.a: ")).toBe(true);
    expect(error.message).toContain("deeper than 3 levels");
  });

  test("an empty path's message has no location", () => {
    const error = deepError({}, 100);
    expect(error.path).toEqual([]);
    expect(error.message).toBe(
      "output: value nests deeper than 100 levels, " +
        "the limit for a yian value (each record and list is a level)",
    );
  });
});

describe("recursiveEntryPath", () => {
  test("names the entry that refers back", () => {
    const items: Value[] = [1n];
    items.push(items);
    const inner: { [key: string]: Value } = {};
    const record = { a: inner };
    inner["b"] = record;
    // Python's pair holds a tuple; a list here (no tuples in TS).
    const pair: Value[] = [];
    pair.push(["x", pair]);

    expect(recursiveEntryPath(items)).toEqual([1]);
    expect(recursiveEntryPath(record)).toEqual(["a", "b"]);
    expect(recursiveEntryPath(pair)).toEqual([0, 1]);
  });

  test("is null without a cycle", () => {
    const shared = [1n, 2n];

    expect(recursiveEntryPath({ a: shared, b: [shared] })).toBeNull();
    expect(recursiveEntryPath(1n)).toBeNull();
  });
});

describe("rejectLoneSurrogates", () => {
  const reason = "text holding a lone surrogate is not a yian value";

  test("refuses root text", () => {
    const error = typeCheckError(() => rejectLoneSurrogates("\uD800", "out"));
    expect(error.message).toBe(`out: ${reason}`);
    expect(error.path).toEqual([]);
  });

  test("refuses nested text at its path", () => {
    const error = typeCheckError(() => rejectLoneSurrogates(["a", ["\uDC00"]], "out"));
    expect(error.message).toBe(`out at [1][0]: ${reason}`);
    expect(error.path).toEqual([1, 0]);
  });

  test("refuses a record key at its record's path", () => {
    const error = typeCheckError(() => rejectLoneSurrogates({ a: { "\uD800": 1n } }, "out"));
    expect(error.message).toBe(`out at a: ${reason}`);
    expect(error.path).toEqual(["a"]);
  });

  test("passes a surrogate pair, a recursive list and a TypedValue of good text", () => {
    const items: Value[] = ["ok"];
    items.push(items);

    rejectLoneSurrogates(["\uD83D\uDE00", { "\uD83D\uDE00": "x" }], "out");
    rejectLoneSurrogates(items, "out");
    rejectLoneSurrogates(new TypedValue(ValueKind.STRING, "ok"), "out");
  });

  test("refuses a TypedValue wrapping bad text", () => {
    const error = typeCheckError(() =>
      rejectLoneSurrogates([new TypedValue(ValueKind.STRING, "\uD800")], "out"),
    );
    expect(error.message).toBe(`out at [0]: ${reason}`);
  });
});
