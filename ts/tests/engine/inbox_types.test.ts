/** The Inbox(T) type: constructor, shape, label, overlap. */
import { expect, test } from "bun:test";

import { canonicalTypeLabel, inboxType, isInboxType, Type, typesOverlap, ValueKind } from "../../src/typesys/values.ts";

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
