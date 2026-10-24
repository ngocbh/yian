/** The inbox handle, its type and its codec. */
import { describe, expect, test } from "bun:test";

import { InboxRef, inboxName } from "../../src/typesys/values.ts";

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
