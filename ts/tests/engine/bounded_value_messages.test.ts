/**
 * Messages that show a value or key stay within the shown-value cap (200), a port of the
 * `value_repr` and unknown-record-field tests of `tests/engine/test_bounded_value_messages.py`.
 *
 * Also here: `test_long_ints.py::test_value_repr_of_a_small_int`, and the path renderings of
 * `test_nonfinite_values.py::test_reject_nonfinite_names_where_and_path`. The spelling,
 * placeholder and bounded-rendering rows are TS-only: a message shows a value in its text form
 * (JSON style, JS number spelling), not Python's `repr`. Also TS-only: a test pinning
 * `valueRepr` to `writeAsText`, the text form builtins and templates write, for every value that
 * isn't text.
 *
 * The unknown-field rows check `buildValueAs`;
 * `test_record_sorts_mixed_unknown_keys_by_type_without_crashing` keeps its two long names (a
 * record key is always a string, so the int and broken-repr keys have no TS counterpart).
 *
 * Later: the unknown-input message row (with bindings) and the rows that run a flow (with the
 * runner).
 *
 * Not ported: the long-int placeholders (`<int with N digits>`; an int is a `bigint` with no
 * digit limit), the set rendering (no sets in TS) and `_sort_repr` (Python-only: it orders
 * names whose cut renderings are equal, which TS sorts uncut).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import process from "node:process";

import { BUILTIN, DefaultsError, forget, install, reload } from "../../src/defaults.ts";
import {
  buildValueAs,
  FileRef,
  renderValuePath,
  shownText,
  Type,
  TypeCheckError,
  type Value,
  ValueKind,
  valueRepr,
  writeAsText,
} from "../../src/typesys/values.ts";

const CAP = 200;
const MESSAGE_BOUND = 1200;

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

class Foo {}

describe("valueRepr", () => {
  // A message shows text JSON-quoted, so the cut rendering starts with `"`, not Python's `'`.
  test("cuts a long string to the cap", () => {
    expect(valueRepr("x".repeat(500))).toBe('"' + "x".repeat(CAP - 2) + "…");
  });

  test("keeps a rendering exactly at the cap", () => {
    const value = "x".repeat(CAP - 2);
    expect(valueRepr(value)).toBe(`"${value}"`);
    expect(valueRepr(value)).toHaveLength(CAP);
  });

  test("follows the configured cap", () => {
    reload((name) => (name === "YIAN_MAX_SHOWN_VALUE_CHARS" ? "10" : undefined));
    expect(valueRepr("abcdefghijklmnop")).toBe('"abcdefgh…');
  });

  test("throws a bad defaults variable rather than showing a placeholder", () => {
    const variable = "YIAN_MAX_SHOWN_VALUE_CHARS";
    const saved = process.env[variable];
    process.env[variable] = "1";
    forget();
    try {
      expect(() => valueRepr("abc")).toThrow(DefaultsError);
    } finally {
      if (saved === undefined) delete process.env[variable];
      else process.env[variable] = saved;
    }
  });

  test("shows a small int as its digits", () => {
    expect(valueRepr(5n)).toBe("5");
  });

  test.each<[string, unknown, string]>([
    ["true", true, "true"],
    ["null", null, "null"],
    ["a float", 2.5, "2.5"],
    ["a whole float", 2, "2"],
    ["infinity", Infinity, "Infinity"],
    ["a long int", 12345678901234567890123n, "12345678901234567890123"],
    ["a list", ["a"], '["a"]'],
    ["a record", { k: 1n }, '{"k":1}'],
    ["a file", new FileRef({ uri: "s3://b/k" }), '{"uri":"s3://b/k","mime":null,"name":null}'],
    ["JS key order", { b: 1n, 2: 2n }, '{"2":2,"b":1}'],
    ["a null-prototype record", Object.assign(Object.create(null), { k: "v" }), '{"k":"v"}'],
    ["a Date", new Date(0), "<Date object>"],
    ["a Map", new Map(), "<Map object>"],
    ["a class instance", new Foo(), "<Foo object>"],
    // The prototype's own `constructor` names it, not one the value fakes.
    ["a faked constructor", Object.assign(new Foo(), { constructor: Map }), "<Foo object>"],
    ["a symbol", Symbol("s"), "<symbol object>"],
    ["undefined inside a list", [1n, undefined], "[1,<undefined object>]"],
  ])("spells %s", (_id, value, expected) => {
    expect(valueRepr(value)).toBe(expected);
  });

  test("gives a placeholder when a getter throws", () => {
    const value = {
      get a(): never {
        throw new Error("boom");
      },
    };
    expect(valueRepr(value)).toBe("<Object object>");
  });

  test("renders a cyclic and a very deep value within the cap without recursion", () => {
    const cyclic: unknown[] = [];
    cyclic.push(cyclic);
    expect(valueRepr(cyclic)).toBe("[".repeat(CAP - 1) + "…");

    let deep: unknown = "leaf";
    for (let i = 0; i < 100_000; i++) deep = { a: deep };
    expect(valueRepr(deep)).toBe('{"a":'.repeat(39) + '{"a"…');
  });

  test("renders a huge list in time bounded by the cap", () => {
    const huge: unknown[] = [];
    huge.length = 2 ** 32 - 1;
    const shown = valueRepr(huge);
    expect(shown).toHaveLength(CAP);
    expect(shown.startsWith("[<undefined object>,<undefined object>,")).toBe(true);
  });

  test("a message shows a value that isn't text in its text form", () => {
    const values: Value[] = [
      true,
      12n,
      10n ** 25n,
      1.5,
      1,
      Infinity,
      -Infinity,
      null,
      [1n, "a", null],
      { b: 1n, "2": [true] },
      new FileRef({ uri: "s3://b/k", mime: "text/plain" }),
    ];
    for (const value of values) expect(valueRepr(value)).toBe(writeAsText(value));
  });
});

describe("shownText", () => {
  test("counts code points and never splits a surrogate pair", () => {
    expect(shownText("😀".repeat(CAP))).toBe("😀".repeat(CAP));
    expect(shownText("😀".repeat(CAP + 1))).toBe("😀".repeat(CAP - 1) + "…");
  });
});

describe("renderValuePath", () => {
  test.each<[string, (string | number)[], string]>([
    ["a list index", ["a", 1], "a[1]"],
    ["a field after an index", ["a", 1, "b"], "a[1].b"],
    ["a key that isn't an identifier", [0, "my key"], "[0]['my key']"],
  ])("renders %s", (_id, path, expected) => {
    expect(renderValuePath(path)).toBe(expected);
  });

  test("cuts each key to the cap", () => {
    expect(renderValuePath(["x".repeat(301)])).toBe("x".repeat(CAP - 1) + "…");
    expect(renderValuePath([" ".repeat(301)])).toBe("['" + " ".repeat(CAP - 2) + "…]");
  });

  test("keeps the start and end of a path longer than four times the cap", () => {
    const shown = renderValuePath(Array.from({ length: 1000 }, (_, i) => `k${i}`));
    expect([...shown]).toHaveLength(4 * CAP);
    expect(shown.startsWith("k0.k1.k2.")).toBe(true);
    expect(shown.endsWith(".k998.k999")).toBe(true);
    expect(shown.slice(2 * CAP, 2 * CAP + 1)).toBe("…");
  });
});

describe("unknown record fields", () => {
  const record = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["a", Type.scalar(ValueKind.INTEGER)]]),
    required: new Set(["a"]),
  });

  /** Return the `TypeCheckError` `buildValueAs(record, value)` throws. */
  function buildError(value: unknown): TypeCheckError {
    try {
      buildValueAs(record, value);
    } catch (error) {
      expect(error).toBeInstanceOf(TypeCheckError);
      return error as TypeCheckError;
    }
    throw new Error("buildValueAs did not throw");
  }

  test("a record lists many unknown fields within the cap", () => {
    const value: Record<string, bigint> = {};
    for (let i = 0; i < 200_000; i++) value[`key${i}`] = BigInt(i);
    value["a"] = 1n;
    const error = buildError(value);
    expect(error.message.startsWith("record has unknown fields: [")).toBe(true);
    expect(error.message.length).toBeLessThanOrEqual(MESSAGE_BOUND);
    expect(error.path).toEqual(["key0"]);
  });

  test("a record sorts long unknown names past the cap", () => {
    // Equal up to the cap, so a sort on the cut text could not order them.
    const later = "x".repeat(300) + "b";
    const earlier = "x".repeat(300) + "a";
    const error = buildError({ a: 1n, [later]: 1n, [earlier]: 2n });
    expect(error.message.startsWith("record has unknown fields: ['xxx")).toBe(true);
    expect(error.message.length).toBeLessThanOrEqual(MESSAGE_BOUND);
    expect(error.path).toEqual([earlier]);
  });
});
