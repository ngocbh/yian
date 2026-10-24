import { describe, expect, test } from "bun:test";
import {
  ASK,
  BUILTIN_TOOL_IDS,
  isPending,
  pending,
  PENDING_BRAND,
  SLEEP,
  TAKE,
  toolCallId,
} from "../../src/tools/context.ts";

describe("built-in ids and the pending marker", () => {
  test("ids", () => {
    expect([TAKE, SLEEP, ASK]).toEqual(["take", "sleep", "ask"]);
    expect([...BUILTIN_TOOL_IDS]).toEqual(["take", "sleep", "ask"]);
  });
  test("a TOOL call id is the run id and the expanded node path", () => {
    expect(toolCallId("run-7f3a", "loop#3/place")).toBe("run-7f3a/loop#3/place");
  });
  test("pending() is a frozen branded marker; isPending reads the brand only", () => {
    const marker = pending();
    expect(isPending(marker)).toBe(true);
    expect(Object.isFrozen(marker)).toBe(true);
    // A marker from another copy of yian carries the same registry symbol.
    expect(isPending({ [Symbol.for("yian.pending")]: true })).toBe(true);
    for (const other of [null, undefined, "pending", {}, { pending: true }, () => {}]) {
      expect(isPending(other)).toBe(false);
    }
    const hostile = new Proxy({}, {
      get() {
        throw new Error("trap");
      },
    });
    expect(isPending(hostile)).toBe(false);
    expect(PENDING_BRAND as symbol).toBe(Symbol.for("yian.pending"));
  });
});
