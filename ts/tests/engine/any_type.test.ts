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
    ["fields", { fields: new Map<string, Type>() }],
    ["tags", { tags: new Set(["a"]) }],
    ["bounds", { bounds: [0n, 1n] as const }],
    ["element", { element: Type.scalar(ValueKind.STRING) }],
  ] as const)("rejects %s", (_name, refinement) => {
    expect(() => new Type({ kind: ValueKind.ANY, ...refinement }).validateShape()).toThrow(
      RangeError,
    );
  });
});
