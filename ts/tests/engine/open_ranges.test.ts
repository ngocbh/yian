/**
 * One-sided numeric ranges' shape, a port of the shape tests of
 * `tests/engine/test_open_ranges.py`; labels, value checks, overlap and parsing come later.
 */

import { describe, expect, test } from "bun:test";

import { Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";

const INT = ValueKind.INTEGER;
const NUM = ValueKind.NUMBER;

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
    const typ = new Type({ kind, bounds } as never);
    let caught: unknown;
    try {
      typ.validateShape();
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(cls);
    expect(caught).not.toBeInstanceOf(TypeCheckError);
    expect((caught as Error).constructor).toBe(cls);
  });
});
