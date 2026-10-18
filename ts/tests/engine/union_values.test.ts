/**
 * The union refinement's shape, a port of the shape tests of
 * `tests/engine/test_union_values.py`; tag fields, overlap, value checks and helpers come later.
 */

import { describe, expect, test } from "bun:test";

import { Type, ValueKind } from "../../src/typesys/values.ts";

const STR = Type.scalar(ValueKind.STRING);
const INT = Type.scalar(ValueKind.INTEGER);
const FLOAT = Type.scalar(ValueKind.NUMBER);
const OBJECT = Type.scalar(ValueKind.OBJECT);

function lit(...tags: string[]): Type {
  return new Type({ kind: ValueKind.STRING, tags: new Set(tags) });
}

/** A closed record whose non-nullable fields are required. */
function rec(fields: Record<string, Type>): Type {
  const entries = Object.entries(fields);
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(entries),
    required: new Set(entries.filter(([, typ]) => !typ.nullable).map(([name]) => name)),
  });
}

function union(members: Type[], nullable = false): Type {
  return new Type({ kind: ValueKind.OBJECT, variants: members, nullable });
}

const buy = rec({ action: lit("buy"), qty: INT });
const sell = rec({ action: lit("sell"), price: FLOAT });
const U = union([buy, sell]);

const text = rec({ text: STR });
const call = rec({ name: STR, input: OBJECT });
const UNTAGGED = union([call, text]);

describe("union shape", () => {
  test("is valid", () => {
    U.validateShape();
    UNTAGGED.validateShape();
    union([buy, sell], true).validateShape();
  });

  test.each([
    ["kind", new Type({ kind: ValueKind.STRING, variants: [buy, sell] }), RangeError],
    [
      "other-refinement",
      new Type({ kind: ValueKind.OBJECT, fields: new Map([["a", INT]]), variants: [buy, sell] }),
      RangeError,
    ],
    ["one-member", union([buy]), RangeError],
    ["scalar-member", union([buy, STR]), RangeError],
    ["object-member", union([buy, OBJECT]), RangeError],
    ["nullable-member", union([buy, sell.replace({ nullable: true })]), RangeError],
    [
      "not-an-array",
      new Type({ kind: ValueKind.OBJECT, variants: new Set([buy, sell]) } as never),
      TypeError,
    ],
  ] as const)("rejects %s", (_id, typ, cls) => {
    expect(() => typ.validateShape()).toThrow(cls);
  });

  test("a list of a union stores as list[object]", () => {
    new Type({ kind: ValueKind.LIST_OBJECT, element: U }).validateShape();
    expect(() => new Type({ kind: ValueKind.LIST_STRING, element: U }).validateShape()).toThrow(
      /expected 'list\[object\]'/,
    );
  });
});
