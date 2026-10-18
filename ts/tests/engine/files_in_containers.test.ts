/**
 * Files inside typed containers, a port of the overlap tests of
 * `tests/engine/test_files_in_containers.py`; value checks, file normalization and flow runs
 * come later.
 */

import { describe, expect, test } from "bun:test";

import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, typesOverlap, ValueKind } from "../../src/typesys/values.ts";

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
