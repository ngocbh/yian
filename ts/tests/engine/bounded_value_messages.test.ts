/**
 * Messages that show a value or key stay within the shown-value cap (200), a port of the
 * `value_repr` tests of `tests/engine/test_bounded_value_messages.py`.
 *
 * Also here: `test_long_ints.py::test_value_repr_of_a_small_int`, and the path renderings of
 * `test_nonfinite_values.py::test_reject_nonfinite_names_where_and_path`. The spelling,
 * placeholder and bounded-rendering rows are TS-only: a message shows a value in its text form
 * (JSON style, JS number spelling), not Python's `repr`.
 *
 * Later: the unknown-field and unknown-input message rows (with the write boundary and
 * bindings) and the rows that run a flow (with the runner); the configured-cap rows
 * (`test_value_repr_follows_the_configured_cap`, `test_value_repr_raises_a_bad_defaults_variable`)
 * once `defaults.ts` lands.
 *
 * Not ported: the long-int placeholders (`<int with N digits>`; an int is a `bigint` with no
 * digit limit), the set rendering (no sets in TS) and `_sort_repr` (Python-only: it orders
 * names whose cut renderings are equal, which TS sorts uncut).
 */

import { describe, expect, test } from "bun:test";

import { FileRef, renderValuePath, shownText, valueRepr } from "../../src/typesys/values.ts";

const CAP = 200;

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
