/**
 * The `Any` type's shape, a port of the shape tests of `tests/engine/test_any_type.py`;
 * value, parsing, compatibility and overlap tests come later.
 */

import { describe, expect, test } from "bun:test";

import { Type, ValueKind } from "../../src/typesys/values.ts";

describe("Any type shape", () => {
  test("is valid", () => {
    new Type({ kind: ValueKind.ANY }).validateShape();
  });

  test.each([
    ["fields", { fields: new Map<string, Type>() }, /only object types may declare record fields/],
    ["tags", { tags: new Set(["a"]) }, /only string types may declare variant tags/],
    ["bounds", { bounds: [0n, 1n] }, /only int and float types may declare bounds/],
    [
      "element",
      { element: Type.scalar(ValueKind.STRING) },
      /only list types may declare an element type/,
    ],
  ] as const)("rejects %s", (_name, refinement, message) => {
    const typ = new Type({ kind: ValueKind.ANY, ...refinement });
    expect(() => typ.validateShape()).toThrow(RangeError);
    expect(() => typ.validateShape()).toThrow(message);
  });
});
