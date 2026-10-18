/**
 * The union refinement's shape, tag fields, order, labels and overlap, a port of those tests
 * of `tests/engine/test_union_values.py`; value checks and the member helpers come later.
 */

import { describe, expect, test } from "bun:test";

import { compareCodeUnits } from "../../src/diagnostics.ts";
import {
  canonicalTypeLabel,
  canonicalUnionOrder,
  Type,
  typesOverlap,
  unionTagFields,
  ValueKind,
} from "../../src/typesys/values.ts";

const STR = Type.scalar(ValueKind.STRING);
const INT = Type.scalar(ValueKind.INTEGER);
const FLOAT = Type.scalar(ValueKind.NUMBER);
const OBJECT = Type.scalar(ValueKind.OBJECT);
const BOOL = Type.scalar(ValueKind.BOOLEAN);
const DATE = Type.scalar(ValueKind.DATE);
const DATETIME = Type.scalar(ValueKind.DATETIME);
const NONE = Type.scalar(ValueKind.NONE);

function lit(...tags: string[]): Type {
  return new Type({ kind: ValueKind.STRING, tags: new Set(tags) });
}

function optional(typ: Type): Type {
  return typ.replace({ nullable: true });
}

function range(kind: ValueKind, bounds: readonly [bigint | number, bigint | number]): Type {
  return new Type({ kind, bounds });
}

/** A closed record whose non-nullable fields are required. */
function rec(fields: Record<string, Type>): Type {
  const entries = Object.entries(fields);
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(entries),
    required: new Set(entries.filter(([, typ]) => !typ.nullable).map(([name]) => name)),
  });
}

function union(members: Type[], nullable = false): Type {
  return new Type({ kind: ValueKind.OBJECT, variants: members, nullable });
}

const buy = rec({ action: lit("buy"), qty: INT });
const sell = rec({ action: lit("sell"), price: FLOAT });
const hold = rec({ action: lit("hold") });
const U = union([buy, sell]);

const text = rec({ text: STR });
const call = rec({ name: STR, input: OBJECT });
const UNTAGGED = union([call, text]);

describe("union shape", () => {
  test("is valid", () => {
    U.validateShape();
    UNTAGGED.validateShape();
    union([buy, sell], true).validateShape();
  });

  test.each([
    [
      "kind",
      new Type({ kind: ValueKind.STRING, variants: [buy, sell] }),
      RangeError,
      /only object types may declare union variants/,
    ],
    [
      "other-refinement",
      new Type({ kind: ValueKind.OBJECT, fields: new Map([["a", INT]]), variants: [buy, sell] }),
      RangeError,
      /a type may declare only one structural refinement/,
    ],
    ["one-member", union([buy]), RangeError, /a union needs at least two members/],
    ["scalar-member", union([buy, STR]), RangeError, /every union member must be a record type/],
    ["object-member", union([buy, OBJECT]), RangeError, /every union member must be a record type/],
    [
      "nullable-member",
      union([buy, sell.replace({ nullable: true })]),
      RangeError,
      /union members must not be nullable/,
    ],
    [
      "not-an-array",
      new Type({ kind: ValueKind.OBJECT, variants: new Set([buy, sell]) } as never),
      TypeError,
      /type variants must be an array/,
    ],
  ] as const)("rejects %s", (_id, typ, cls, message) => {
    expect(() => typ.validateShape()).toThrow(cls);
    expect(() => typ.validateShape()).toThrow(message);
  });

  test("a list of a union stores as list[object]", () => {
    new Type({ kind: ValueKind.LIST_OBJECT, element: U }).validateShape();
    expect(() => new Type({ kind: ValueKind.LIST_STRING, element: U }).validateShape()).toThrow(
      /expected 'list\[object\]'/,
    );
  });
});

describe("unionTagFields", () => {
  test("tagged", () => {
    expect(unionTagFields(U)).toEqual(["action"]);
    expect(unionTagFields(union([buy, sell, hold]))).toEqual(["action"]);
  });

  test("untagged disjoint", () => {
    expect(unionTagFields(UNTAGGED)).toEqual([]);
  });

  test("of a record is empty", () => {
    expect(unionTagFields(buy)).toEqual([]);
  });

  test("excludes an optional Literal", () => {
    const first = rec({ action: optional(lit("buy")), qty: INT });
    const second = rec({ action: lit("sell"), price: FLOAT });
    expect(unionTagFields(union([first, second]))).toEqual([]);
  });

  test("excludes a Literal that is not required", () => {
    const first = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["action", lit("buy")],
        ["qty", INT],
      ]),
      required: new Set(["qty"]),
    });
    expect(unionTagFields(union([first, sell]))).toEqual([]);
  });

  test("excludes a multi-value Literal", () => {
    const first = rec({ action: lit("buy", "add"), qty: INT });
    expect(unionTagFields(union([first, sell]))).toEqual([]);
  });

  test("returns every qualifying field, sorted", () => {
    const first = rec({ side: lit("b"), action: lit("buy"), qty: INT });
    const second = rec({ side: lit("s"), action: lit("sell"), price: FLOAT });
    expect(unionTagFields(union([first, second]))).toEqual(["action", "side"]);
  });

  test("excludes equal values", () => {
    const first = rec({ kind: lit("trade"), action: lit("buy"), qty: INT });
    const second = rec({ kind: lit("trade"), action: lit("sell"), price: FLOAT });
    expect(unionTagFields(union([first, second]))).toEqual(["action"]);
  });
});

describe("typesOverlap", () => {
  const I = ValueKind.INTEGER;
  test.each([
    // nullable on both sides, with NONE counting as nullable
    ["Optional(str), Optional(int)", optional(STR), optional(INT), true],
    ["None, Optional(int)", NONE, optional(INT), true],
    ["None, int", NONE, INT, false],
    // same scalar kind, or int/float
    ["int, int", INT, INT, true],
    ["bool, bool", BOOL, BOOL, true],
    ["int, float", INT, FLOAT, true],
    ["bool, int", BOOL, INT, false],
    // str with date/datetime/Literal
    ["str, date", STR, DATE, true],
    ["datetime, str", DATETIME, STR, true],
    ["str, Literal(buy)", STR, lit("buy"), true],
    // Literal with date/datetime
    ["Literal(2026-01-02), date", lit("2026-01-02"), DATE, true],
    ["date, Literal(buy)", DATE, lit("buy"), false],
    ["Literal(2026-01-02T10:00:00), datetime", lit("2026-01-02T10:00:00"), DATETIME, true],
    ["Literal(2026-01-02), datetime", lit("2026-01-02"), DATETIME, false],
    // two Literals
    ["Literal(a, b), Literal(b, c)", lit("a", "b"), lit("b", "c"), true],
    ["Literal(a), Literal(b)", lit("a"), lit("b"), false],
    // numeric bounds
    ["int(0..5), int(3..9)", range(I, [0n, 5n]), range(I, [3n, 9n]), true],
    ["int(0..2), int(3..9)", range(I, [0n, 2n]), range(I, [3n, 9n]), false],
    // bigint ends against number ends
    ["int(0..2), float(2.5..3.0)", range(I, [0n, 2n]), range(ValueKind.NUMBER, [2.5, 3.0]), false],
    ["int(0..2), float", range(I, [0n, 2n]), FLOAT, true],
    // two lists
    [
      "list(str), list(int)",
      new Type({ kind: ValueKind.LIST_STRING, element: STR }),
      new Type({ kind: ValueKind.LIST_INTEGER, element: INT }),
      true,
    ],
    [
      "list[Any], list[object]",
      Type.scalar(ValueKind.LIST_ANY),
      Type.scalar(ValueKind.LIST_OBJECT),
      true,
    ],
    // object with a record or union
    ["object, buy", OBJECT, buy, true],
    ["U, object", U, OBJECT, true],
    // two records
    ["buy, sell", buy, sell, false],
    [
      "{a: int}, {a: int, b: Optional(str)}",
      rec({ a: INT }),
      rec({ a: INT, b: optional(STR) }),
      true,
    ],
    ["{a: int}, {b: int}", rec({ a: INT }), rec({ b: INT }), false],
    ["{a: int}, {a: str}", rec({ a: INT }), rec({ a: STR }), false],
    ["{a: int}, {a: float}", rec({ a: INT }), rec({ a: FLOAT }), true],
    [
      "{a: int}, {a?: str}",
      rec({ a: INT }),
      new Type({ kind: ValueKind.OBJECT, fields: new Map([["a", STR]]), required: new Set() }),
      false,
    ],
    // a union against anything
    ["U, buy", U, buy, true],
    ["hold, U", hold, U, false],
    ["U, Union(hold, sell)", U, union([hold, sell]), true],
    // different kinds otherwise
    ["str, int", STR, INT, false],
    ["date, datetime", DATE, DATETIME, false],
    ["buy, str", buy, STR, false],
    ["list[Any], object", Type.scalar(ValueKind.LIST_ANY), OBJECT, false],
  ] as const)("%s", (_id, a, b, expected) => {
    expect(typesOverlap(a, b)).toBe(expected);
    expect(typesOverlap(b, a)).toBe(expected);
  });
});

function deepUntaggedUnion(depth: number): Type {
  let typ = union([rec({ a: STR }), rec({ b: INT })]);
  for (let level = 0; level < depth; level++) {
    typ = union([rec({ inner: typ }), rec({ [`leaf${level}`]: INT })]);
  }
  return typ;
}

describe("union labels and order", () => {
  test("a deep untagged union labels in linear work", () => {
    // Python counts the nodes labelled; an exponential walk of depth 100 would not finish.
    const depth = 100;
    const typ = deepUntaggedUnion(depth);
    const start = performance.now();
    const label = canonicalTypeLabel(typ);
    expect(performance.now() - start).toBeLessThan(1000);
    expect(label.split("Union(").length - 1).toBe(depth + 1);
  });

  test("a union label lists members in canonical order", () => {
    expect(canonicalTypeLabel(union([sell, buy]))).toBe(
      `Union(${canonicalTypeLabel(buy)},${canonicalTypeLabel(sell)})`,
    );
    expect(canonicalTypeLabel(union([buy, sell], true))).toStartWith("Optional(Union(");
  });

  test("canonical union order", () => {
    expect(canonicalUnionOrder([sell, hold, buy])).toEqual([buy, hold, sell]);
    const labels = [canonicalTypeLabel(text), canonicalTypeLabel(call)].sort(compareCodeUnits);
    const ordered = canonicalUnionOrder([text, call]);
    expect(ordered.map(canonicalTypeLabel)).toEqual(labels);
  });
});
