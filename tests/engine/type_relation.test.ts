/**
 * Classification of an inferred type against a declared type: a port of every test of
 * `tests/engine/test_type_relation.py`.
 *
 * Changed: Python's `10**400` end is `10n ** 400n`; record fields are a `Map` and `required` a
 * `Set`.
 */

import { expect, test } from "bun:test";

import { type TypeRelation, typeRelation } from "../../src/compile/validation.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

const REGISTRY = readTypedefs({
  A: { a: "int" },
  AB: { a: "int", b: "Optional(str)" },
  ABStr: { a: "int", b: "str" },
  AX: { a: "int", x: "int" },
  AStr: { a: "str" },
  AOpt: { a: "Optional(int)" },
  AOptStr: { a: "Optional(str)" },
  ALit: { a: "Literal(x, y)" },
  Nested: { inner: "AB" },
  NestedA: { inner: "A" },
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", qty: "int" },
  Hold: { action: "Literal(hold)" },
  Decision: "Union(Buy, Sell, Hold)",
  X: { x: "int" },
});

/** Resolve a type expression against the test typedefs. */
const t = (text: string): Type => typeFor(text, REGISTRY);

test.each([
  ["int", "int"],
  ["int", "float"],
  ["int", "Optional(int)"],
  ["Literal(a)", "str"],
  ["Literal(a)", "Literal(a, b)"],
  ["A", "object"],
  ["date", "str"],
  ["datetime", "str"],
  ["A", "AB"],
  ["None", "Optional(int)"],
  ["None", "None"],
  ["int(0..5)", "int(0..9)"],
  ["int(0..5)", "float"],
  ["list(int)", "list(float)"],
  ["list(int)", "list(Optional(int))"],
  ["NestedA", "Nested"],
  ["file", "file"],
  ["int(5..)", "int(0..)"],
  ["int(0..10)", "int(..10)"],
  ["int(0..)", "float(0.0..)"],
])("fits: %s into %s", (inferred, declared) => {
  expect(typeRelation(t(inferred), t(declared))).toBe("fits");
});

test.each([
  ["Optional(int)", "int"],
  ["int", "int(0..5)"],
  ["int(0..5)", "int(3..9)"],
  ["str", "Literal(a, b)"],
  ["Literal(a, b)", "Literal(b, c)"],
  ["object", "A"],
  ["str", "date"],
  ["str", "datetime"],
  ["AB", "A"],
  ["AOpt", "A"],
  ["list(Optional(int))", "list(int)"],
  // A tag-less source into a Literal stays narrower at any depth.
  ["list(str)", "list(Literal(x, y))"],
  ["AStr", "ALit"],
  ['Literal(a, "2026-01-01")', "date"],
  ['Literal("2026-01-01T09:30:00")', "datetime"],
  ["date", 'Literal(a, "2026-01-01")'],
  // Records are closed at every level.
  ["Nested", "NestedA"],
  // Both sides may omit the field, so the empty record is shared.
  ["AOptStr", "AOpt"],
  ["float", "float(0..1)"],
  // A float-typed value may hold an int, since a typed binding never converts.
  ["float", "int"],
  ["int(0..)", "int(0..100)"],
  ["int", "int(0..)"],
  ["int(..5)", "int(0..)"],
  ["int(..5)", "int(..3)"],
  // A whole number lies in the overlap [1, 1.5].
  ["float(..1.5)", "int(1..)"],
  ["float", "float(0.0..)"],
])("narrower: %s into %s", (inferred, declared) => {
  expect(typeRelation(t(inferred), t(declared))).toBe("narrower");
});

test("an open range with a huge int end is narrower", () => {
  const huge = 10n ** 400n;
  const atLeastHuge = new Type({ kind: ValueKind.INTEGER, bounds: [huge, null] });
  const atMostMinusHuge = new Type({ kind: ValueKind.INTEGER, bounds: [null, -huge] });
  expect(typeRelation(t("int(0..)"), atLeastHuge)).toBe("narrower");
  expect(typeRelation(t("int(..0)"), atMostMinusHuge)).toBe("narrower");
});

test("a refined list into an unrefined list fits", () => {
  expect(typeRelation(t("list(int)"), Type.scalar(ValueKind.LIST_ANY))).toBe("fits");
});

test("an unrefined list into a refined list is narrower", () => {
  expect(typeRelation(Type.scalar(ValueKind.LIST_ANY), t("list(int)"))).toBe("narrower");
});

test.each([
  ["str", "int"],
  ["bool", "int"],
  ["int", "list(int)"],
  ["list(str)", "list(int)"],
  ["list(int)", "list(object)"],
  ["object", "int"],
  ["int", "object"],
  ["file", "str"],
  ["date", "datetime"],
  ["datetime", "date"],
  ["Literal(a)", "Literal(b)"],
  ["int(0..5)", "int(10..20)"],
  ["None", "int"],
  ["ABStr", "A"],
  ["A", "AX"],
  ["AStr", "A"],
  ["Optional(str)", "Optional(int)"],
  ["A", "list(int)"],
  ["list(A)", "list(AStr)"],
  ["AOptStr", "A"],
  ["int", "float(0.5..0.7)"],
  ["Optional(int)", "None"],
  // The required field is a str; the declared field is an optional int.
  ["AStr", "AOpt"],
  ["Literal(a)", "date"],
  ['Literal("2026-01-01")', "datetime"],
  ["date", "Literal(a)"],
  ["datetime", 'Literal("2026-01-01")'],
  ["int(..-1)", "int(0..)"],
  ["int(..0)", "int(1..)"],
  // The overlap [1, 0.5] is empty, and no whole number lies in it.
  ["float(..0.5)", "int(1..)"],
  ["int(5..)", "int(..4)"],
])("disjoint: %s into %s", (inferred, declared) => {
  expect(typeRelation(t(inferred), t(declared))).toBe("disjoint");
});

test("a record field both sides may hold None is narrower", () => {
  // An inferred literal record can require a field it holds as `None`, so neither side may
  // omit `a`, yet both admit `{"a": None}`.
  const inferred = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["a", t("Optional(str)")]]),
    required: new Set(["a"]),
  });
  expect(typeRelation(inferred, t("AOpt"))).toBe("narrower");
});

test.each<[string, string, TypeRelation]>([
  // A union member over the union is explicit narrowing.
  ["Decision", "Buy", "narrower"],
  ["Buy", "Decision", "fits"],
  ["Decision", "Decision", "fits"],
  ["Union(Buy, Sell)", "Decision", "fits"],
  ["Decision", "Union(Buy, Sell)", "narrower"],
  ["Decision", "int", "disjoint"],
  ["Decision", "list(Buy)", "disjoint"],
  ["Decision", "object", "fits"],
  ["object", "Decision", "narrower"],
  ["Optional(Decision)", "Decision", "narrower"],
  ["Decision", "Optional(Decision)", "fits"],
  // Records are closed, so a record matching no member shares no value.
  ["X", "Decision", "disjoint"],
  ["Union(Buy, Sell)", "Hold", "disjoint"],
  ["Union(Buy, Sell)", "Union(Sell, Hold)", "narrower"],
  ["list(Buy)", "list(Decision)", "fits"],
  ["list(Decision)", "list(Buy)", "narrower"],
])(
  "union: %s into %s is %s",
  (inferred, declared, expected) => {
    expect(typeRelation(t(inferred), t(declared))).toBe(expected);
  },
);
