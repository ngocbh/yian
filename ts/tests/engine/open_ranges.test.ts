/**
 * One-sided numeric ranges' shape, labels, overlap and parsing, a port of those tests of
 * `tests/engine/test_open_ranges.py`; value checks and resolution come later.
 * `test_numeric_interval_maps_open_ends_to_infinity` is not ported (design 3.2.12: open ends
 * stay `null`).
 */

import { describe, expect, test } from "bun:test";

import { formatTypeExpr, parseType } from "../../src/typesys/types.ts";
import {
  type Bounds,
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
  });

  test("neither end says write the plain type", () => {
    expect(() => parseType("int(..)")).toThrow(TypeCheckError);
    expect(() => parseType("int(..)")).toThrow(
      "malformed type expression 'int(..)': a range needs at least one end; write int for any int",
    );
  });
});
