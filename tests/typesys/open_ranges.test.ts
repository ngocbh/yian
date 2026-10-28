/**
 * One-sided numeric ranges' shape, labels, overlap, parsing, resolution and value checks. NaN is
 * never a value, so it is refused before any range check.
 */

import { describe, expect, test } from "bun:test";

import {
  formatTypeExpr,
  parseType,
  readTypedefs,
  resolveType,
  typeFor,
} from "../../src/typesys/types.ts";
import {
  type Bounds,
  buildValueAs,
  canonicalTypeLabel,
  Type,
  TypeCheckError,
  typesOverlap,
  ValueKind,
} from "../../src/typesys/values.ts";

const INT = ValueKind.INTEGER;
const NUM = ValueKind.NUMBER;

/** Return the error `typ.validateShape()` throws, failing unless it is exactly `cls`. */
function shapeError(typ: Type, cls: typeof TypeError | typeof RangeError): Error {
  try {
    typ.validateShape();
  } catch (error) {
    expect((error as Error).constructor).toBe(cls);
    return error as Error;
  }
  throw new Error("validateShape() did not throw");
}

describe("open range shape", () => {
  test.each([
    ["int [0n, null]", INT, [0n, null]],
    ["int [null, 100n]", INT, [null, 100n]],
    ["float [0.0, null]", NUM, [0.0, null]],
    ["float [null, 1n]", NUM, [null, 1n]],
    ["float [-1.5, null]", NUM, [-1.5, null]],
  ] as const)("one open end is a valid shape: %s", (_label, kind, bounds) => {
    new Type({ kind, bounds }).validateShape();
  });

  test.each([
    ["int [null, null]", INT, [null, null], RangeError],
    ["float [null, null]", NUM, [null, null], RangeError],
    ["int [null, 1.0]", INT, [null, 1.0], TypeError],
    ["float [null, Infinity]", NUM, [null, Infinity], RangeError],
    ["float [10n ** 400n, null]", NUM, [10n ** 400n, null], RangeError],
  ] as const)("an open end still rejects bad metadata: %s", (_label, kind, bounds, cls) => {
    shapeError(new Type({ kind, bounds }), cls);
  });
});

describe("open range labels and overlap", () => {
  test.each([
    [INT, [0n, null], "int(0..)"],
    [INT, [null, 100n], "int(..100)"],
    [INT, [-5n, null], "int(-5..)"],
    [NUM, [0.0, null], "float(0.0..)"],
  ] as const)("label spells the open end empty: %s %p", (kind, bounds, label) => {
    expect(canonicalTypeLabel(new Type({ kind, bounds }))).toBe(label);
    expect(canonicalTypeLabel(new Type({ kind, bounds, nullable: true }))).toBe(
      `Optional(${label})`,
    );
  });

  test("overlap with open ends", () => {
    const range = (bounds: Bounds) => new Type({ kind: INT, bounds });
    expect(typesOverlap(range([0n, null]), range([null, 0n]))).toBe(true);
    expect(typesOverlap(range([1n, null]), range([null, 0n]))).toBe(false);
    expect(typesOverlap(range([5n, null]), Type.scalar(INT))).toBe(true);
  });
});

describe("open range parsing", () => {
  test.each([
    "int(0..)",
    "int(..100)",
    "float(0.0..)",
    "float(..-1.5)",
    "Optional(int(0..))",
    "list(float(..1.0))",
  ])("round-trips %p", (text) => {
    expect(formatTypeExpr(parseType(text))).toBe(text);
    expect(canonicalTypeLabel(typeFor(text, new Map()))).toBe(text);
  });

  test("neither end says write the plain type", () => {
    expect(() => parseType("int(..)")).toThrow(TypeCheckError);
    expect(() => parseType("int(..)")).toThrow(
      "malformed type expression 'int(..)': a range needs at least one end; write int for any int",
    );
  });

  test("open ranges resolve in records, aliases and lists", () => {
    const reg = readTypedefs({
      Count: "int(0..)",
      Order: { qty: "int(1..)", price: "float(..100.0)" },
    });
    expect(resolveType(parseType("Count"), reg).bounds).toEqual([0n, null]);
    const order = resolveType(parseType("Order"), reg);
    expect(order.fields!.get("qty")!.bounds).toEqual([1n, null]);
    expect(order.fields!.get("price")!.bounds).toEqual([null, 100]);
    expect(typeFor("list(Optional(int(..3)))", new Map()).element!.bounds).toEqual([null, 3n]);
  });
});

/** Return the message of the `TypeCheckError` `buildValueAs(typ, value)` throws. */
function buildError(typ: Type, value: unknown): string {
  try {
    buildValueAs(typ, value);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error("buildValueAs did not throw");
}

describe("open range values", () => {
  test("a value check names the end it fails", () => {
    expect(buildValueAs(new Type({ kind: INT, bounds: [0n, null] }), 10n ** 50n).toObject()).toBe(
      10n ** 50n,
    );
    expect(
      buildValueAs(new Type({ kind: INT, bounds: [null, 100n] }), -(10n ** 50n)).toObject(),
    ).toBe(-(10n ** 50n));
    expect(buildError(new Type({ kind: INT, bounds: [0n, null] }), -1n)).toBe(
      "-1 is below the inclusive int minimum 0",
    );
    expect(buildError(new Type({ kind: INT, bounds: [null, 100n] }), 101n)).toBe(
      "101 is above the inclusive int maximum 100",
    );
    // Values in messages use the text form: the float bound 0.0 shows as `0`.
    expect(buildError(new Type({ kind: NUM, bounds: [0.0, null] }), -0.5)).toBe(
      "-0.5 is below the inclusive float minimum 0",
    );
    // The two-sided message is unchanged.
    expect(buildError(new Type({ kind: INT, bounds: [1n, 12n] }), 13n)).toBe(
      "13 is outside inclusive int bounds (1, 12)",
    );
    // Pins the bounds spelling: the ends in parentheses, a float with a fraction digit.
    expect(buildError(new Type({ kind: NUM, bounds: [0.0, 1.0] }), 2n)).toBe(
      "2 is outside inclusive float bounds (0.0, 1.0)",
    );
  });

  test("NaN is refused before any range check", () => {
    // NaN is never a value, so the "outside the inclusive range" message can't occur.
    expect(buildError(new Type({ kind: NUM, bounds: [0.0, null] }), NaN)).toBe(
      "NaN is not a yian value",
    );
    expect(buildError(new Type({ kind: NUM, bounds: [null, 1n], nullable: true }), NaN)).toBe(
      "NaN is not a yian value",
    );
  });

  // A one-sided range admits the infinity on its open side; a two-sided range admits neither.
  test.each<[Bounds, number, boolean]>([
    [[0.0, null], Infinity, true],
    [[0.0, null], -Infinity, false],
    [[null, 1.0], -Infinity, true],
    [[null, 1.0], Infinity, false],
    [[0.0, 10.0], Infinity, false],
    [[0.0, 10.0], -Infinity, false],
  ])("float bounds %p admit %p: %p", (bounds, value, admitted) => {
    const typ = new Type({ kind: NUM, bounds });
    if (admitted) expect(buildValueAs(typ, value).toObject()).toBe(value);
    else buildError(typ, value);
  });
});
