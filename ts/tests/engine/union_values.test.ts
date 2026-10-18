/**
 * The union refinement's shape, tag fields, order, labels, overlap, value checks
 * (`buildValueAs`) and member helpers (`unionMemberForTag`, `restrictUnion`, `unionField`), a
 * port of `tests/engine/test_union_values.py`. Python's `ValueError` is a `RangeError` here.
 *
 * TS-only: a no-member error still reads as class `TypeCheckError` (`error.name`).
 */

import { describe, expect, test } from "bun:test";

import { compareCodeUnits } from "../../src/diagnostics.ts";
import {
  type Bounds,
  buildValueAs,
  canonicalTypeLabel,
  canonicalUnionOrder,
  type FieldAbsent,
  type FieldPartial,
  type FieldRead,
  restrictUnion,
  Type,
  TypeCheckError,
  typeEquals,
  typesOverlap,
  unionField,
  unionMemberForTag,
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

describe("unionMemberForTag and restrictUnion", () => {
  test("unionMemberForTag", () => {
    expect(unionMemberForTag(U, "buy")).toBe(buy);
    expect(unionMemberForTag(U, "sell")).toBe(sell);
    expect(unionMemberForTag(U, "hold")).toBeNull();
  });

  test("unionMemberForTag on another tag field", () => {
    const first = rec({ side: lit("b"), action: lit("buy"), qty: INT });
    const second = rec({ side: lit("s"), action: lit("sell"), price: FLOAT });
    const both = union([first, second]);
    expect(unionMemberForTag(both, "s", "side")).toBe(second);
    expect(() => unionMemberForTag(both, "buy", "qty")).toThrow(
      new RangeError("'qty' is not a tag field of the union"),
    );
  });

  test("restrictUnion on another tag field", () => {
    const first = rec({ side: lit("b"), action: lit("buy"), qty: INT });
    const second = rec({ side: lit("s"), action: lit("sell"), price: FLOAT });
    const third = rec({ side: lit("h"), action: lit("hold") });
    const three = union([first, second, third]);
    expect(typeEquals(restrictUnion(three, ["s"], "side"), second)).toBe(true);
    expect(typeEquals(restrictUnion(three, ["s", "b"], "side"), union([first, second]))).toBe(
      true,
    );
    expect(() => restrictUnion(three, ["sell"], "side")).toThrow(
      new RangeError("'sell' is not a tag of the union"),
    );
  });

  test("restrictUnion returns members in canonical order", () => {
    const restricted = restrictUnion(union([sell, hold, buy]), new Set(["sell", "buy"]));
    expect(typeEquals(restricted, union([buy, sell]))).toBe(true);
  });

  test("restrictUnion rejects a bare string", () => {
    expect(() => restrictUnion(U, "buy")).toThrow(
      new TypeError("restrictUnion keeps a collection of tags, not a bare string"),
    );
  });

  test("unionMemberForTag needs a tagged union", () => {
    expect(() => unionMemberForTag(UNTAGGED, "text")).toThrow(
      new RangeError("the type is not a tagged union"),
    );
  });

  test("restrictUnion to one tag gives the member", () => {
    expect(typeEquals(restrictUnion(union([buy, sell, hold]), ["buy"]), buy)).toBe(true);
    expect(typeEquals(restrictUnion(union([buy, sell], true), ["buy"]), optional(buy))).toBe(
      true,
    );
  });

  test("restrictUnion to two tags keeps nullable", () => {
    const three = union([buy, sell, hold], true);
    expect(typeEquals(restrictUnion(three, ["sell", "buy"]), union([buy, sell], true))).toBe(
      true,
    );
    expect(typeEquals(restrictUnion(union([buy, sell, hold]), ["buy", "sell"]), U)).toBe(true);
  });

  test("restrictUnion by member", () => {
    expect(typeEquals(restrictUnion(UNTAGGED, [text]), text)).toBe(true);
  });

  test("restrictUnion rejects empty and unknown", () => {
    expect(() => restrictUnion(U, new Set())).toThrow(
      new RangeError("restrictUnion needs at least one member"),
    );
    expect(() => restrictUnion(U, ["hold"])).toThrow(
      new RangeError("'hold' is not a tag of the union"),
    );
    expect(() => restrictUnion(UNTAGGED, [hold])).toThrow(
      new RangeError("restrictUnion got a type that is not a union member"),
    );
  });
});

/** Require `result` to be a `FieldRead` whose type equals `expected` (`null`: untyped). */
function expectRead(
  result: FieldRead | FieldPartial | FieldAbsent,
  expected: Type | null,
): void {
  expect(result.kind).toBe("read");
  const read = (result as FieldRead).type;
  if (expected === null) expect(read).toBeNull();
  else expect(typeEquals(read!, expected)).toBe(true);
}

describe("unionField", () => {
  test("a shared tag reads the merged Literal", () => {
    expectRead(unionField(U, "action"), lit("buy", "sell"));
  });

  test("equal types read that type", () => {
    const first = rec({ action: lit("buy"), note: STR });
    const second = rec({ action: lit("sell"), note: STR });
    expectRead(unionField(union([first, second]), "note"), STR);
  });

  test("equality ignores nullable and nullability spreads", () => {
    const first = rec({ action: lit("buy"), note: STR });
    const second = rec({ action: lit("sell"), note: optional(STR) });
    expectRead(unionField(union([first, second]), "note"), optional(STR));
  });

  test("a field that is not required is nullable", () => {
    const first = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["action", lit("buy")],
        ["note", STR],
      ]),
      required: new Set(["action"]),
    });
    const second = rec({ action: lit("sell"), note: STR });
    expectRead(unionField(union([first, second]), "note"), optional(STR));
  });

  test("a merged Literal is nullable if any member's is", () => {
    const first = rec({ action: lit("buy"), mode: lit("a") });
    const second = rec({ action: lit("sell"), mode: optional(lit("b")) });
    expectRead(unionField(union([first, second]), "mode"), optional(lit("a", "b")));
  });

  test("differing types read untyped", () => {
    const first = rec({ action: lit("buy"), size: INT });
    const second = rec({ action: lit("sell"), size: STR });
    expectRead(unionField(union([first, second]), "size"), null);
  });

  test("partial", () => {
    const partial = unionField(union([buy, sell, hold]), "qty");
    expect(partial).toEqual({ kind: "partial", members: [buy] });
    expect(unionField(UNTAGGED, "text")).toEqual({ kind: "partial", members: [text] });
  });

  test("absent", () => {
    expect(unionField(U, "zzz")).toEqual({ kind: "absent" });
  });
});
