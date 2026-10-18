/**
 * Canonical type labels, a port of the label tests of
 * `tests/engine/test_inline_composite_types.py`. Later: inline input inference, with
 * `test_inline_type_copy_preserves_bounded_nested_validation` (it builds values on the
 * inferred copy).
 */

import { describe, expect, test } from "bun:test";

import {
  canonicalTypeLabel,
  isListKind,
  listElementKind,
  Type,
  ValueKind,
} from "../../src/typesys/values.ts";

const LIST_VALUE_KINDS = Object.values(ValueKind).filter(isListKind);

/** The label of a list kind with no element type, e.g. `list(str)`. */
function listLabel(kind: ValueKind): string {
  const element = listElementKind(kind);
  return `list(${element === null ? "Optional(Any)" : element})`;
}

describe("canonical type labels", () => {
  test.each([
    ValueKind.NONE,
    ValueKind.STRING,
    ValueKind.INTEGER,
    ValueKind.NUMBER,
    ValueKind.DATE,
    ValueKind.DATETIME,
    ValueKind.BOOLEAN,
    ValueKind.OBJECT,
    ValueKind.FILE,
  ])("scalar labels use author vocabulary: %s", (kind) => {
    expect(canonicalTypeLabel(Type.scalar(kind))).toBe(kind);
  });

  test.each(LIST_VALUE_KINDS)("every list kind label uses its structural element: %s", (kind) => {
    const typ = new Type({ kind, element: new Type({ kind: ValueKind.INTEGER, nullable: true }) });
    expect(canonicalTypeLabel(typ)).toBe("list(Optional(int))");
  });

  test.each(LIST_VALUE_KINDS)("flat list label uses its storage kind: %s", (kind) => {
    expect(canonicalTypeLabel(Type.scalar(kind))).toBe(listLabel(kind));
  });

  test("variant label sorts and JSON-quotes adversarial tags", () => {
    const typ = new Type({ kind: ValueKind.STRING, tags: new Set(["é", "z", 'a,"b']) });
    expect(canonicalTypeLabel(typ)).toBe('Literal("a,\\"b", z, é)');
  });

  test("record label sorts and JSON-quotes adversarial keys", () => {
    const fields = new Map([
      ["é", Type.scalar(ValueKind.BOOLEAN)],
      ['quote"key', Type.scalar(ValueKind.STRING)],
      ["a: int, b", Type.scalar(ValueKind.INTEGER)],
    ]);
    const typ = new Type({ kind: ValueKind.OBJECT, fields, required: new Set(fields.keys()) });
    expect(canonicalTypeLabel(typ)).toBe('{"a: int, b":int,"quote\\"key":str,"é":bool}');
  });

  test("the nullable wrapper is outermost for a structural record label", () => {
    const typ = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["value", Type.scalar(ValueKind.STRING)]]),
      required: new Set(["value"]),
      nullable: true,
    });
    expect(canonicalTypeLabel(typ)).toBe('Optional({"value":str})');
  });
});
