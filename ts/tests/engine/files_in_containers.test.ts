/**
 * Files inside typed containers, a port of the overlap tests of
 * `tests/engine/test_files_in_containers.py`, with its typedef records built by hand; value
 * checks, file normalization and flow runs come later.
 */

import { describe, expect, test } from "bun:test";

import { Type, typesOverlap, ValueKind } from "../../src/typesys/values.ts";

const FILE = Type.scalar(ValueKind.FILE);
const OBJECT = Type.scalar(ValueKind.OBJECT);
const ANY = Type.scalar(ValueKind.ANY);
const STR = Type.scalar(ValueKind.STRING);
const INT = Type.scalar(ValueKind.INTEGER);
const OPTIONAL_STR = new Type({ kind: ValueKind.STRING, nullable: true });

/** A closed record whose non-nullable fields are required, as the typedef resolver builds. */
function rec(fields: Record<string, Type>): Type {
  const entries = Object.entries(fields);
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(entries),
    required: new Set(entries.filter(([, typ]) => !typ.nullable).map(([name]) => name)),
  });
}

const Form = rec({ uri: STR, mime: OPTIONAL_STR, name: OPTIONAL_STR });
const XFile = rec({ x: FILE });
const XObject = rec({ x: OBJECT });
const IntUri = rec({ uri: INT });
const Other = rec({ uri: STR, other: STR });

describe("untagged unions whose members can share a file", () => {
  test.each([
    ["file, object", FILE, OBJECT, true],
    ["file, Any", FILE, ANY, true],
    ["file, Form", FILE, Form, true],
    ["XFile, XObject", XFile, XObject, true],
    ["file, int", FILE, INT, false],
    ["file, IntUri", FILE, IntUri, false],
    ["file, Other", FILE, Other, false],
  ] as const)("a file overlaps the types that admit its JSON form: %s", (_id, a, b, expected) => {
    expect(typesOverlap(a, b)).toBe(expected);
    expect(typesOverlap(b, a)).toBe(expected);
  });
});
