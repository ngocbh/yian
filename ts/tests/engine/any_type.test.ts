/**
 * The `Any` type's shape, labels, overlap, parsing and resolution, a port of those tests of
 * `tests/engine/test_any_type.py`; value, compatibility and `readTypedefs` tests come later,
 * as does the `readTypedefs` half of `test_any_overlaps_a_list_of_files_through_the_empty_list`.
 * The records and registries written as typedefs there are built by hand here.
 */

import { describe, expect, test } from "bun:test";

import {
  parseType,
  type TypeDef,
  type TypeRegistry,
  typeFor,
} from "../../src/typesys/types.ts";
import {
  canonicalTypeLabel,
  Type,
  TypeCheckError,
  typeEquals,
  typesOverlap,
  ValueKind,
} from "../../src/typesys/values.ts";

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
  ] as const)("authored %s resolves and labels back", (text, typ) => {
    expect(typeEquals(typeFor(text, new Map()), typ)).toBe(true);
    expect(canonicalTypeLabel(typ)).toBe(text);
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

describe("Any parsing", () => {
  test("parses as its own scalar", () => {
    expect(parseType("Any")).toEqual({ kind: "scalar", name: "Any" });
  });

  test("lowercase any is not a type", () => {
    expect(() => parseType("any")).toThrow(TypeCheckError);
    expect(() => parseType("any")).toThrow(/^unknown type 'any'; write Any$/);
  });
});

/**
 * Build a registry by hand, without checks: a mapping is a record, `Literal(...)` text a
 * variant and other text an alias.
 */
function registry(raw: Record<string, string | Record<string, string>>): TypeRegistry {
  return new Map(
    Object.entries(raw).map(([name, value]): [string, TypeDef] => {
      if (typeof value !== "string") {
        return [name, { kind: "record", fields: new Map(Object.entries(value)) }];
      }
      const target = parseType(value);
      if (target.kind === "enum") return [name, { kind: "variant", tags: target.tags }];
      return [name, { kind: "alias", target }];
    }),
  );
}

/** Python's overlap error for `label`, members `a` and `b`, with the `file` note if `files`. */
function overlap(label: string, a: string, b: string, files = false): string {
  const why = files
    ? " (a `file` slot reads `{uri, mime, name}`, so `file` overlaps `object`, `Any` and a " +
      "record of that shape)"
    : "";
  return (
    `${label}: members '${a}' and '${b}' can match the same value${why}; add a tag field ` +
    "every member declares as a single-value Literal, with a different value in each " +
    `member, e.g. kind: Literal(${a.toLowerCase()})`
  );
}

describe("Any in untagged unions", () => {
  test("a union of Any and int fields does not resolve", () => {
    const reg = registry({ A: { v: "Any" }, B: { v: "int" }, U: "Union(A, B)" });
    expect(() => typeFor("U", reg)).toThrow(TypeCheckError);
    expect(() => typeFor("U", reg)).toThrow(overlap("Union(A, B)", "A", "B"));
  });

  test("a union of Any and a union with a file member does not resolve", () => {
    const reg = registry({
      F: { f: "file" },
      N: { n: "int" },
      U: "Union(F, N)",
      A: { v: "Any" },
      B: { v: "U" },
      W: "Union(A, B)",
    });
    expect(() => typeFor("W", reg)).toThrow(TypeCheckError);
    expect(() => typeFor("W", reg)).toThrow(overlap("Union(A, B)", "A", "B", true));
  });
});
