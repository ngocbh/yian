/** The inbox handle, its type and its codec. */
import { describe, expect, test } from "bun:test";

import { canonicalSeed, canonicalTyped } from "../../src/runtime/log.ts";
import {
  buildValue,
  buildValueAs,
  fromHost,
  hasFile,
  InboxRef,
  inboxName,
  inboxType,
  normalizeFiles,
  rejectDeepNesting,
  requiresFile,
  toHost,
  Type,
  TypeCheckError,
  TypedValue,
  ValueKind,
  valueRepr,
  writeAsJson,
} from "../../src/typesys/values.ts";

describe("InboxRef", () => {
  test("holds a top-level inbox name and is frozen", () => {
    const ref = new InboxRef("owner_messages");
    expect(ref.inbox).toBe("owner_messages");
    expect(Object.isFrozen(ref)).toBe(true);
    expect(InboxRef.isInboxRef(ref)).toBe(true);
  });

  test("an object given its prototype is not one", () => {
    expect(InboxRef.isInboxRef(Object.create(InboxRef.prototype))).toBe(false);
    expect(InboxRef.isInboxRef({ inbox: "m" })).toBe(false);
  });

  test.each([[""], ["\uD800"], [5 as unknown as string]])("refuses the name %p", (name) => {
    expect(() => new InboxRef(name)).toThrow(RangeError);
  });

  test("inboxName reads a handle or its mapping, else null", () => {
    expect(inboxName(new InboxRef("m"))).toBe("m");
    expect(inboxName({ inbox: "m" })).toBe("m");
    expect(inboxName({ inbox: "m", x: 1n })).toBeNull();
    expect(inboxName({ inbox: "" })).toBeNull();
    expect(inboxName("m")).toBeNull();
  });
});

describe("inbox wire", () => {
  test("toWire and fromWire round-trip the handle", () => {
    expect(new TypedValue(ValueKind.INBOX, new InboxRef("m")).toWire()).toEqual({
      kind: "inbox",
      value: { inbox: "m" },
    });
    const back = TypedValue.fromWire({ kind: "inbox", value: { inbox: "m" } });
    expect(InboxRef.isInboxRef(back.value) && back.value.inbox).toBe("m");
  });

  test.each([
    [{ kind: "inbox", value: "m" }],
    [{ kind: "inbox", value: { inbox: "m", x: null } }],
    [{ kind: "inbox", value: { inbox: "" } }],
  ])("fromWire refuses %p", (wire) => {
    expect(() => TypedValue.fromWire(wire)).toThrow(RangeError);
  });

  test("a seed writes a nested handle as its mapping", () => {
    const [wire] = canonicalSeed({ owner_messages: new InboxRef("owner_messages"), n: 1n });
    expect(wire).toEqual({ n: 1n, owner_messages: { inbox: "owner_messages" } });
  });

  test("an untyped record holding a handle commits as its mapping", () => {
    const [wire] = canonicalTyped({ m: new InboxRef("m") }, null);
    expect(wire).toEqual({ kind: "object", value: { m: { inbox: "m" } } });
  });
});

describe("inbox at the host boundary", () => {
  const ref = new InboxRef("m");

  test("host intake and hand-over keep the handle", () => {
    expect((fromHost({ a: ref }, "x") as { a: unknown }).a).toBe(ref);
    expect((toHost({ a: ref }) as { a: unknown }).a).toBe(ref);
  });

  test("the nesting check walks into a handle as one level", () => {
    expect(() => rejectDeepNesting({ a: ref }, "x")).not.toThrow();
  });

  test("a handle renders as its mapping", () => {
    expect(valueRepr(ref)).toBe(valueRepr({ inbox: "m" }));
    expect(writeAsJson({ a: ref })).toBe(writeAsJson({ a: { inbox: "m" } }));
  });
});

describe("inbox at the write boundary", () => {
  const ref = new InboxRef("m");
  const STR = Type.scalar(ValueKind.STRING);

  test("a bare handle infers kind inbox", () => {
    expect(buildValue(ref).kind).toBe(ValueKind.INBOX);
  });

  test.each([[ref], [{ inbox: "m" }]])("an Inbox(T) slot stores a handle for %p", (value) => {
    const stored = buildValueAs(inboxType(STR), value);
    expect(stored.kind).toBe(ValueKind.INBOX);
    expect(InboxRef.isInboxRef(stored.value) && stored.value.inbox).toBe("m");
  });

  test.each([["m"], [{ inbox: "m", x: 1n }], [{ inbox: "" }], [null], [[ref]]])(
    "an Inbox(T) slot refuses %p",
    (value) => {
      expect(() => buildValueAs(inboxType(STR), value)).toThrow(TypeCheckError);
    },
  );

  test.each([[ValueKind.ANY], [ValueKind.OBJECT]])("%p refuses a handle", (kind) => {
    expect(() => buildValueAs(kind, ref)).toThrow(TypeCheckError);
  });
});

describe("inbox reading type", () => {
  const STR = Type.scalar(ValueKind.STRING);

  test("a record's inbox field reads its mapping back as a handle", () => {
    const record = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["m", inboxType(STR)]]),
      required: new Set(["m"]),
    });
    const read = normalizeFiles(record, { m: { inbox: "m" } }) as { m: unknown };
    expect(InboxRef.isInboxRef(read.m) && read.m.inbox).toBe("m");
  });

  test("an inbox type holds a value that isn't plain JSON", () => {
    expect(hasFile(inboxType(STR))).toBe(true);
    expect(requiresFile(inboxType(STR))).toBe(true);
  });
});
