/**
 * Files inside typed containers, a port of the overlap tests and the `field_of` table of
 * `tests/engine/test_files_in_containers.py`, with `FileRef` constructor refusals
 * (`RangeError`, as pydantic's `ValidationError` is a `ValueError`).
 *
 * Later: value checks and file normalization (with the write boundary) and flow runs (with the
 * runner).
 *
 * Not ported: the `field_of` rows on dict subclasses, `defaultdict` and a `FileRef` subclass; a
 * record is a plain object and a dotted read gives its own keys.
 */

import { describe, expect, test } from "bun:test";

import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  FileRef,
  fieldOf,
  Type,
  typesOverlap,
  type Value,
  ValueKind,
} from "../../src/typesys/values.ts";

const REGISTRY = readTypedefs({
  Rec: { f: "file", n: "int" },
  Form: { uri: "str", mime: "Optional(str)", name: "Optional(str)" },
  IntUri: { uri: "int" },
  Other: { uri: "str", other: "str" },
  XFile: { x: "file" },
  XObject: { x: "object" },
  Box: { x: "object" },
  Num: { n: "float" },
  TFile: { kind: "Literal(file)", x: "file" },
  TObject: { kind: "Literal(object)", x: "object" },
});

/** Resolve a type expression against this module's typedefs. */
const t = (text: string): Type => typeFor(text, REGISTRY);

const FILE = Type.scalar(ValueKind.FILE);
const OBJECT = Type.scalar(ValueKind.OBJECT);
const ANY = Type.scalar(ValueKind.ANY);

describe("untagged unions whose members can share a file", () => {
  test.each([
    ["file, object", FILE, OBJECT, true],
    ["file, Any", FILE, ANY, true],
    ["file, Form", FILE, t("Form"), true],
    ["XFile, XObject", t("XFile"), t("XObject"), true],
    ["file, int", FILE, t("int"), false],
    ["file, IntUri", FILE, t("IntUri"), false],
    ["file, Other", FILE, t("Other"), false],
  ] as const)("a file overlaps the types that admit its JSON form: %s", (_id, a, b, expected) => {
    expect(typesOverlap(a, b)).toBe(expected);
    expect(typesOverlap(b, a)).toBe(expected);
  });
});

/** A record holding an own `__proto__` key, as yian builds one. */
function protoKeyRecord(): Value {
  const record = {};
  Object.defineProperty(record, "__proto__", {
    value: 1n,
    writable: true,
    enumerable: true,
    configurable: true,
  });
  return record as Value;
}

describe("fieldOf", () => {
  const file = new FileRef({ uri: "f", mime: "m", name: "n" });
  test.each<[string, Value, string, [boolean, Value]]>([
    ["a key", { a: 1n }, "a", [true, 1n]],
    ["a key holding null", { a: null }, "a", [true, null]],
    ["a missing key", { a: 1n }, "b", [false, null]],
    ["a file's uri", file, "uri", [true, "f"]],
    ["a file's mime", file, "mime", [true, "m"]],
    ["a file's name", file, "name", [true, "n"]],
    ["an unset file field", new FileRef({ uri: "f" }), "name", [true, null]],
    ["another name on a file", new FileRef({ uri: "f" }), "other", [false, null]],
    ["__class__ on a file", new FileRef({ uri: "f" }), "__class__", [false, null]],
    ["model_dump on a file", new FileRef({ uri: "f" }), "model_dump", [false, null]],
    ["a name on text", "text", "upper", [false, null]],
    ["an index on a list", [1n], "0", [false, null]],
    ["a name on null", null, "a", [false, null]],
    // TS-only: a dotted read gives own keys only, never an inherited JS property.
    ["constructor on a record", { a: 1n }, "constructor", [false, null]],
    ["toString on a record", { a: 1n }, "toString", [false, null]],
    ["__proto__ on a record", { a: 1n }, "__proto__", [false, null]],
    ["constructor on a file", new FileRef({ uri: "f" }), "constructor", [false, null]],
    ["length on text", "text", "length", [false, null]],
    ["an own __proto__ key", protoKeyRecord(), "__proto__", [true, 1n]],
    ["a null-prototype record's key", Object.assign(Object.create(null), { a: 1n }), "a", [true, 1n]],
  ])("reads a record key or a file field: %s", (_id, value, name, expected) => {
    expect(fieldOf(value, name)).toEqual(expected);
  });
});

describe("FileRef", () => {
  test("defaults mime and name to null and is frozen", () => {
    const file = new FileRef({ uri: "s3://b/k" });
    expect([file.uri, file.mime, file.name]).toEqual(["s3://b/k", null, null]);
    expect(Object.isFrozen(file)).toBe(true);
  });

  test.each<[string, unknown]>([
    ["a missing uri", {}],
    ["a non-text uri", { uri: 1n }],
    ["a null uri", { uri: null }],
    ["a non-text mime", { uri: "u", mime: 1n }],
    ["a non-text name", { uri: "u", name: false }],
    ["a uri with a lone surrogate", { uri: "\ud800" }],
    ["a name with a lone surrogate", { uri: "u", name: "a\udc00" }],
  ])("refuses %s", (_id, fields) => {
    expect(() => new FileRef(fields as never)).toThrow(RangeError);
  });

  test("names the field in its message", () => {
    expect(() => new FileRef({ uri: 1n } as never)).toThrow("file uri must be text, got 1");
    expect(() => new FileRef({ uri: "u", name: "a\udc00" })).toThrow(
      "file name holds a lone surrogate",
    );
  });
});
