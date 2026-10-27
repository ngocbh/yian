/** The Inbox(T) type: constructor, shape, label, overlap. */
import { describe, expect, test } from "bun:test";

import { formatTypeExpr, parseType, readTypedefs, typeFor } from "../../src/typesys/types.ts";
import {
  canonicalTypeLabel,
  inboxType,
  isInboxType,
  Type,
  typesOverlap,
  ValueKind,
} from "../../src/typesys/values.ts";

const STR = Type.scalar(ValueKind.STRING);

test("inboxType builds kind inbox with its element; isInboxType tells it apart", () => {
  const typ = inboxType(STR);
  expect([typ.kind, typ.element, typ.nullable]).toEqual([ValueKind.INBOX, STR, false]);
  expect([isInboxType(typ), isInboxType(STR), isInboxType(null)]).toEqual([true, false, false]);
});

test("label", () => {
  expect(canonicalTypeLabel(inboxType(STR))).toBe("Inbox(str)");
});

test("shape", () => {
  expect(() => inboxType(STR).replace({ nullable: true }).validateShape()).toThrow("an inbox is never optional");
  expect(() => new Type({ kind: ValueKind.INBOX }).validateShape()).toThrow(RangeError);
  expect(() => inboxType(inboxType(STR)).validateShape()).toThrow("not inboxes");
  expect(() => inboxType(Type.scalar(ValueKind.INTEGER)).validateShape()).not.toThrow();
});

test("an inbox overlaps only an inbox", () => {
  expect(typesOverlap(inboxType(STR), inboxType(STR))).toBe(true);
  expect(typesOverlap(inboxType(STR), Type.scalar(ValueKind.ANY))).toBe(false);
  expect(typesOverlap(inboxType(STR), Type.scalar(ValueKind.OBJECT))).toBe(false);
});

describe("the type language", () => {
  const NONE = readTypedefs({});

  test("parseType reads Inbox(X) and formatTypeExpr writes it back", () => {
    const expr = parseType("Inbox(str)");
    expect(expr).toEqual({ kind: "inbox", element: { kind: "scalar", name: "str" } });
    expect(formatTypeExpr(expr)).toBe("Inbox(str)");
    expect(formatTypeExpr(parseType("Inbox(list(int))"))).toBe("Inbox(list(int))");
  });

  test("Inbox(X) resolves to kind inbox with its element", () => {
    const typ = typeFor("Inbox(int)", NONE);
    expect([typ.kind, typ.element?.kind, typ.nullable]).toEqual([
      ValueKind.INBOX,
      ValueKind.INTEGER,
      false,
    ]);
  });

  test.each([
    ["Optional(Inbox(str))", "never optional"],
    ["list(Inbox(str))", "can't be a list item"],
    ["Inbox(Inbox(str))", "not inboxes"],
    ["Inbox(str, int)", "takes one type"],
  ])("%p is refused (%s)", (text, message) => {
    expect(() => typeFor(text, NONE)).toThrow(message);
  });

  test("a record typedef field can't be an inbox", () => {
    expect(() => readTypedefs({ R: { m: "Inbox(str)" } })).toThrow(
      "typedef 'R' field 'm': an inbox is a whole input; it can't be a record field",
    );
  });

  test("an inbox alias resolves to the inbox, and is never optional", () => {
    const registry = readTypedefs({ Chat: "Inbox(str)" });
    expect(typeFor("Chat", registry).kind).toBe(ValueKind.INBOX);
    expect(() => typeFor("Optional(Chat)", registry)).toThrow("never optional");
    expect(() => typeFor("list(Chat)", registry)).toThrow("can't be a list item");
    expect(() => typeFor("Inbox(Chat)", registry)).toThrow("not inboxes");
  });

  test("a typedef can't be named Inbox", () => {
    expect(() => readTypedefs({ Inbox: "str" })).toThrow(
      "typedef name 'Inbox' shadows a builtin scalar or typing constructor",
    );
  });

  test("the typedef walkers look inside an inbox", () => {
    expect(() => readTypedefs({ R: { m: "Inbox(Unknown)" } })).toThrow("unknown type 'Unknown'");
    expect(() => readTypedefs({ A: "Inbox(B)", B: "Inbox(A)" })).toThrow();
  });
});

test("a dict can't hold an inbox, directly or through an alias", () => {
  const NONE = readTypedefs({});
  const CHAT = readTypedefs({ Chat: "Inbox(str)" });
  for (const [text, registry] of [
    ["dict(str, Inbox(str))", NONE],
    ["dict(Inbox(str), int)", NONE],
    ["list(dict(str, Inbox(str)))", NONE],
    ["dict(str, Chat)", CHAT],
  ] as const) {
    expect(() => typeFor(text, registry)).toThrow("an inbox is a whole input; it can't be a dict key or value");
  }
  expect(() => typeFor("dict(str, Inbox(str))", NONE)).toThrow("in 'dict(str, Inbox(str))'");
  expect(() => readTypedefs({ R: { m: "dict(str, Inbox(str))" } })).toThrow("can't be a dict key or value");
  expect(typeFor("dict(str, int)", NONE).kind).toBe(ValueKind.OBJECT);
  expect(typeFor("dict(str, Unknown)", NONE).kind).toBe(ValueKind.OBJECT);
});
