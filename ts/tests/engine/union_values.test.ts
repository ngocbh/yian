/**
 * The union refinement's shape, tag fields, order, labels, overlap and value checks
 * (`buildValueAs`), a port of those tests of `tests/engine/test_union_values.py`.
 *
 * TS-only: a no-member error still reads as class `TypeCheckError` (`error.name`).
 *
 * Later: the member helpers (`unionField`, `unionMemberForTag`, `restrictUnion`).
 */

import { describe, expect, test } from "bun:test";

import { compareCodeUnits } from "../../src/diagnostics.ts";
import {
  type Bounds,
  buildValueAs,
  canonicalTypeLabel,
  canonicalUnionOrder,
  Type,
  TypeCheckError,
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

function range(kind: ValueKind, bounds: Bounds): Type {
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
  const INT_KIND = ValueKind.INTEGER;
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
    ["int(0..5), int(3..9)", range(INT_KIND, [0n, 5n]), range(INT_KIND, [3n, 9n]), true],
    ["int(0..2), int(3..9)", range(INT_KIND, [0n, 2n]), range(INT_KIND, [3n, 9n]), false],
    // bigint ends against number ends
    [
      "int(0..2), float(2.5..3.0)",
      range(INT_KIND, [0n, 2n]),
      range(ValueKind.NUMBER, [2.5, 3.0]),
      false,
    ],
    ["int(0..2), float", range(INT_KIND, [0n, 2n]), FLOAT, true],
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

/** Return the `TypeCheckError` `run` throws. */
function typeCheckError(run: () => unknown): TypeCheckError {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("expected a TypeCheckError");
}

function deepUntaggedUnion(depth: number): Type {
  let typ = union([rec({ a: STR }), rec({ b: INT })]);
  for (let level = 0; level < depth; level++) {
    typ = union([rec({ inner: typ }), rec({ [`leaf${level}`]: INT })]);
  }
  return typ;
}

describe("buildValueAs", () => {
  test("a tagged union accepts each member", () => {
    expect(buildValueAs(U, { action: "buy", qty: 2n }).toObject()).toEqual({
      action: "buy",
      qty: 2n,
    });
    expect(buildValueAs(U, { action: "sell", price: 1.5 }).kind).toBe(ValueKind.OBJECT);
  });

  test("a tagged union missing its tag", () => {
    const error = typeCheckError(() => buildValueAs(U, { qty: 2n }));
    expect(error.message).toBe("missing tag field 'action' (expected one of ['buy', 'sell'])");
    expect(error.path).toEqual(["action"]);
  });

  test("a tagged union with an unknown tag", () => {
    const error = typeCheckError(() => buildValueAs(U, { action: "hold" }));
    expect(error.message).toBe(
      `"hold" is not a tag of the union (expected one of ['buy', 'sell'])`,
    );
    expect(error.path).toEqual(["action"]);
  });

  test("a non-string tag value is not a tag", () => {
    expect(typeCheckError(() => buildValueAs(U, { action: ["buy"] })).path).toEqual(["action"]);
  });

  test("a tagged union checks the member record", () => {
    let error = typeCheckError(() => buildValueAs(U, { action: "sell", price: 1.0, qty: 2n }));
    expect(error.message).toContain("unknown fields: ['qty']");
    expect(error.path).toEqual(["qty"]);

    error = typeCheckError(() => buildValueAs(U, { action: "sell", qty: 2n }));
    expect(error.path).toEqual(["price"]);
  });

  test("a union rejects a non-record at the root", () => {
    expect(typeCheckError(() => buildValueAs(U, "buy")).path).toEqual([]);
    expect(typeCheckError(() => buildValueAs(UNTAGGED, 3n)).path).toEqual([]);
  });

  test("an Optional union accepts null", () => {
    expect(buildValueAs(optional(U), null).toObject()).toBeNull();
    typeCheckError(() => buildValueAs(U, null));
  });

  test("an untagged union picks the matching member", () => {
    expect(buildValueAs(UNTAGGED, { text: "hi" }).toObject()).toEqual({ text: "hi" });
    expect(buildValueAs(UNTAGGED, { name: "f", input: {} }).toObject()).toEqual({
      name: "f",
      input: {},
    });
  });

  test("an untagged union matching no member", () => {
    const error = typeCheckError(() => buildValueAs(UNTAGGED, { text: 3n }));
    const callLabel = canonicalTypeLabel(call);
    const textLabel = canonicalTypeLabel(text);
    expect(error.message).toBe(
      `value matches no member of ${canonicalTypeLabel(UNTAGGED)}: ` +
        `${callLabel}: record missing required fields: ['input', 'name']; ` +
        `${textLabel}: 3 does not match declared type str`,
    );
    expect(error.path).toEqual([]);
    expect(error.name).toBe("TypeCheckError");
  });

  test("an untagged no-match message does not nest member messages", () => {
    const inner = union([rec({ a: STR }), rec({ b: INT })]);
    const outer = union([rec({ wrap: inner }), rec({ c: BOOL })]);
    const error = typeCheckError(() => buildValueAs(outer, { wrap: { a: 1n } }));
    const wrapLabel = canonicalTypeLabel(rec({ wrap: inner }));
    const cLabel = canonicalTypeLabel(rec({ c: BOOL }));
    expect(error.message).toBe(
      `value matches no member of ${canonicalTypeLabel(outer)}: ` +
        `${wrapLabel}: value matches no member of ${canonicalTypeLabel(inner)}; ` +
        `${cLabel}: record missing required fields: ['c']`,
    );
    expect(error.path).toEqual([]);
  });

  test("a deep untagged union labels and checks in linear work", () => {
    // Python counts calls to its private label function, which TS can't monkeypatch; a time
    // bound stands in. An exponential walk of depth 30 would take minutes, not seconds.
    const depth = 30;
    const typ = deepUntaggedUnion(depth);
    let start = performance.now();
    const label = canonicalTypeLabel(typ);
    expect(performance.now() - start).toBeLessThan(2000);
    expect(label.split("Union(").length - 1).toBe(depth + 1);
    start = performance.now();
    const error = typeCheckError(() => buildValueAs(typ, { inner: 3n }));
    expect(performance.now() - start).toBeLessThan(2000);
    expect(error.message).toStartWith("value matches no member of ");
  });

  test("a list of a union", () => {
    const typ = new Type({ kind: ValueKind.LIST_OBJECT, element: U });
    const stored = buildValueAs(typ, [
      { action: "buy", qty: 1n },
      { action: "sell", price: 2.0 },
    ]);
    expect(stored.kind).toBe(ValueKind.LIST_OBJECT);
    const error = typeCheckError(() =>
      buildValueAs(typ, [
        { action: "buy", qty: 1n },
        { action: "buy", qty: "x" },
      ]),
    );
    expect(error.path).toEqual([1, "qty"]);
  });
});

describe("union labels and order", () => {
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
