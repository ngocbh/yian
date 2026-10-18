/**
 * The `Any` type's shape, labels and overlap, a port of those tests of
 * `tests/engine/test_any_type.py`; value, parsing and compatibility tests come later, and the
 * records written as typedefs there are built by hand here.
 */

import { describe, expect, test } from "bun:test";

import { canonicalTypeLabel, Type, typesOverlap, ValueKind } from "../../src/typesys/values.ts";

const ANY = Type.scalar(ValueKind.ANY);
const OPTIONAL_ANY = new Type({ kind: ValueKind.ANY, nullable: true });
const LIST_ANY = new Type({ kind: ValueKind.LIST_ANY, element: ANY });
const LIST_OPTIONAL_ANY = new Type({ kind: ValueKind.LIST_ANY, element: OPTIONAL_ANY });
const UNPROVEN_LIST = Type.scalar(ValueKind.LIST_ANY);
const FILE = Type.scalar(ValueKind.FILE);

/** A closed record whose fields are all required. */
function rec(fields: Record<string, Type>): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
  });
}

const FILE_RECORD = rec({ f: FILE });

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

describe("Any labels", () => {
  test.each([
    ["Any", ANY],
    ["Optional(Any)", OPTIONAL_ANY],
    ["list(Any)", LIST_ANY],
    ["list(Optional(Any))", LIST_OPTIONAL_ANY],
  ] as const)("labels %s", (label, typ) => {
    expect(canonicalTypeLabel(typ)).toBe(label);
  });

  test("a list with unproven items is labelled list of Optional(Any)", () => {
    expect(canonicalTypeLabel(UNPROVEN_LIST)).toBe("list(Optional(Any))");
  });
});

describe("Any overlap", () => {
  test.each([
    ["str", Type.scalar(ValueKind.STRING), true],
    ["Rec", rec({ a: Type.scalar(ValueKind.INTEGER) }), true],
    [
      "list(int)",
      new Type({ kind: ValueKind.LIST_INTEGER, element: Type.scalar(ValueKind.INTEGER) }),
      true,
    ],
    ["Any", ANY, true],
    ["None", Type.scalar(ValueKind.NONE), false],
    // A file slot reads a file's JSON form, which is an `Any` value.
    ["file", FILE, true],
    ["{f: file}", FILE_RECORD, true],
  ] as const)("Any overlaps every type but None: %s", (_id, other, expected) => {
    expect(typesOverlap(ANY, other)).toBe(expected);
    expect(typesOverlap(other, ANY)).toBe(expected);
  });

  test("a record with an Any field overlaps one with an int field", () => {
    expect(typesOverlap(rec({ v: ANY }), rec({ v: Type.scalar(ValueKind.INTEGER) }))).toBe(true);
  });

  test("Any overlaps a list of files through the empty list", () => {
    const listOfFiles = new Type({ kind: ValueKind.LIST_ANY, element: FILE });
    expect(typesOverlap(ANY, listOfFiles)).toBe(true);
  });
});
