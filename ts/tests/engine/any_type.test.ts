/**
 * The `Any` type's shape, labels, overlap, parsing, typedefs and resolution, a port of those
 * tests of `tests/engine/test_any_type.py`. Later: the value tests and value halves (with
 * `buildValueAs`), the compatibility, proof and field-read tests (with the compile checks and
 * node contracts), the `requires_file`/`has_file` tests, and the flow loading tests.
 */

import { describe, expect, test } from "bun:test";

import { parseType, readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  canonicalTypeLabel,
  canonicalUnionOrder,
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

const REGISTRY = readTypedefs({
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", qty: "int" },
  Rec: { a: "int" },
});

/** Resolve a type expression against the compatibility typedefs. */
const t = (text: string): Type => typeFor(text, REGISTRY);

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
    ["str", t("str"), true],
    ["Rec", t("Rec"), true],
    ["list(int)", t("list(int)"), true],
    ["Any", ANY, true],
    ["None", t("None"), false],
    // A file slot reads a file's JSON form, which is an `Any` value.
    ["file", FILE, true],
    ["{f: file}", FILE_RECORD, true],
  ] as const)("Any overlaps every type but None: %s", (_id, other, expected) => {
    expect(typesOverlap(ANY, other)).toBe(expected);
    expect(typesOverlap(other, ANY)).toBe(expected);
  });

  test("a record with an Any field overlaps one with an int field", () => {
    const reg = readTypedefs({ A: { v: "Any" }, B: { v: "int" } });
    expect(typesOverlap(typeFor("A", reg), typeFor("B", reg))).toBe(true);
  });

  test("Any overlaps a list of files through the empty list", () => {
    const listOfFiles = new Type({ kind: ValueKind.LIST_ANY, element: FILE });
    expect(typesOverlap(ANY, listOfFiles)).toBe(true);
    expect(
      typedefsError({ A: { v: "Any" }, B: { v: "list(file)" }, W: "Union(A, B)" }),
    ).toBe(`typedef 'W': ${overlap("Union(A, B)", "A", "B")}`);
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

describe("Any in typedefs", () => {
  test("records that differ only in list item nullability sort deterministically", () => {
    const reg = readTypedefs({ A: { xs: "list(Any)" }, B: { xs: "list(Optional(Any))" } });
    const a = typeFor("A", reg);
    const b = typeFor("B", reg);
    for (const order of [canonicalUnionOrder([a, b]), canonicalUnionOrder([b, a])]) {
      expect(order).toHaveLength(2);
      expect(order[0]).toBe(a);
      expect(order[1]).toBe(b);
    }
  });

  test("a typedef cannot be named Any", () => {
    expect(typedefsError({ Any: "str" })).toBe(
      "typedef name 'Any' shadows a builtin scalar or typing constructor",
    );
  });

  test("Any resolves through an alias and a record field", () => {
    const reg = readTypedefs({ Payload: "Any", Rec: { data: "Any" } });
    expect(typeEquals(typeFor("Payload", reg), ANY)).toBe(true);
    expect(typeEquals(typeFor("Rec", reg).fields!.get("data")!, ANY)).toBe(true);
  });
});

/** Return the message of the `TypeCheckError` `readTypedefs(raw)` throws. */
function typedefsError(raw: Readonly<Record<string, unknown>>): string {
  try {
    readTypedefs(raw);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return (error as TypeCheckError).message;
  }
  throw new Error("readTypedefs did not throw");
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
    expect(typedefsError({ A: { v: "Any" }, B: { v: "int" }, U: "Union(A, B)" })).toBe(
      `typedef 'U': ${overlap("Union(A, B)", "A", "B")}`,
    );
  });

  test("a union of Any and a union with a file member does not resolve", () => {
    const typedefs = {
      F: { f: "file" },
      N: { n: "int" },
      U: "Union(F, N)",
      OF: { f: "Optional(file)", n: "int" },
      A: { v: "Any" },
      B: { v: "U" },
      W: "Union(A, B)",
    };
    expect(typedefsError(typedefs)).toBe(
      `typedef 'W': ${overlap("Union(A, B)", "A", "B", true)}`,
    );
  });
});
