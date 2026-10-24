/**
 * The TOOL built-ins at run time: `take` and `sleep` return the wait the engine decides, a host
 * tool or `ask` that returns `pending()` records its call and waits, a held call is reused, and
 * `waitResultStep` shapes a host's answer. TS-only: Python yian has no inboxes or pending calls.
 */

import { describe, expect, test } from "bun:test";

import { Pending } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { type CallView, NodeSteps } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { deadlineUtc, formatRecordTime } from "../../src/typesys/temporal.ts";
import { InboxRef, type ValueRecord } from "../../src/typesys/values.ts";

/** Build a node's caps: recorded steps, `calls`, `tools`, and a virtual clock at 2026-10-23T00:00Z. */
function capsWith(calls: CallView[] = [], tools = new ToolSet({}).forRun("r1"), maxEventBytes?: number) {
  const recorded: ValueRecord[] = [];
  const steps = new NodeSteps(1, [], (step) => (recorded.push(step), recorded.length), calls, maxEventBytes);
  return { recorded, caps: { steps, tools, scheduler: virtualScheduler(Date.UTC(2026, 9, 23)) } };
}

/** Return the error `promise` rejects with; fail if it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

/** Expect `promise` to reject with a `contract.input` `BindingError` whose message is `message`. */
async function expectContractInput(promise: Promise<unknown>, message: string): Promise<void> {
  const error = await rejection(promise);
  expect(error).toBeInstanceOf(BindingError);
  expect((error as BindingError).failureCode).toBe("contract.input");
  expect((error as BindingError).message).toBe(message);
}

describe("temporal", () => {
  test("deadlineUtc reads a date at midnight UTC and normalises an offset", () => {
    expect(deadlineUtc("2026-10-24")).toBe(Date.UTC(2026, 9, 24));
    expect(deadlineUtc("2026-10-24T09:30+07:00")).toBe(Date.UTC(2026, 9, 24, 2, 30));
  });

  test("deadlineUtc refuses an instant before year 1", () => {
    expect(() => deadlineUtc("0001-01-01T00:00+01:00")).toThrow(new RangeError("date value out of range"));
  });

  test("formatRecordTime writes a record time and refuses one out of range", () => {
    expect(formatRecordTime(Date.UTC(2026, 9, 24, 9))).toBe("2026-10-24T09:00:00.000Z");
    expect(() => formatRecordTime(Number.NaN)).toThrow(
      new RangeError("run-log clock time NaN is outside the years 0000 to 9999"),
    );
    expect(() => formatRecordTime(Number.POSITIVE_INFINITY)).toThrow(RangeError);
  });
});

describe("take and sleep", () => {
  const take = new ToolNode("heard", { toolId: "take" });
  const sleep = new ToolNode("nap", { toolId: "sleep" });

  test("take returns a take request; the engine decides", async () => {
    const { caps } = capsWith();
    expect(await take.forward({ from: new InboxRef("m"), take: "all" }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "all", timeoutSeconds: null } }),
    );
  });

  test("take reads from as a handle or its {inbox} form, mode one by default, timeout int or text", async () => {
    const { caps } = capsWith();
    expect(await take.forward({ from: { inbox: "m" } }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "one", timeoutSeconds: null } }),
    );
    expect(await take.forward({ from: new InboxRef("m"), take: "one", timeout: 1800n }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "one", timeoutSeconds: 1800 } }),
    );
    expect(await take.forward({ from: new InboxRef("m"), timeout: "30m" }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "one", timeoutSeconds: 1800 } }),
    );
  });

  test("take refuses a bad mode, from or timeout as contract.input", async () => {
    const { caps } = capsWith();
    await expectContractInput(
      take.forward({ from: new InboxRef("m"), take: "some" }, caps),
      'take `take` must be one or all, got "some"',
    );
    await expectContractInput(
      take.forward({ from: new InboxRef("m"), timeout: "0s" }, caps),
      'take `timeout` "0s": a duration is a positive integer and a unit (90s | 30m | 24h | 2d)',
    );
    await expectContractInput(
      take.forward({ from: new InboxRef("m"), timeout: 0n }, caps),
      "take `timeout` must be a duration, got 0",
    );
    await expectContractInput(take.forward({ from: "m" }, caps), 'take `from` must be an inbox, got "m"');
  });

  test("take and sleep write no step and call no tool", async () => {
    const { recorded, caps } = capsWith();
    await take.forward({ from: new InboxRef("m") }, caps);
    await sleep.forward({ for: "1s" }, caps);
    expect(recorded).toEqual([]);
  });

  test("sleep for and until", async () => {
    const { caps } = capsWith();
    const slept = (wait: { until: string | null; forSeconds: number | null }) =>
      new Pending([], { wait: { tool: "sleep", ...wait } });
    expect(await sleep.forward({ for: "90s" }, caps)).toEqual(slept({ until: null, forSeconds: 90 }));
    expect(await sleep.forward({ for: 90n, until: null }, caps)).toEqual(slept({ until: null, forSeconds: 90 }));
    expect(await sleep.forward({ until: "2026-10-24" }, caps)).toEqual(
      slept({ until: "2026-10-24T00:00:00.000Z", forSeconds: null }),
    );
    expect(await sleep.forward({ until: "2026-10-24T09:30+07:00" }, caps)).toEqual(
      slept({ until: "2026-10-24T02:30:00.000Z", forSeconds: null }),
    );
    // A deadline already past is still a request; the engine commits it at once.
    expect(await sleep.forward({ until: "2000-01-01" }, caps)).toEqual(
      slept({ until: "2000-01-01T00:00:00.000Z", forSeconds: null }),
    );
  });

  test("sleep needs exactly one of until and for; a bad until is contract.input", async () => {
    const { caps } = capsWith();
    const one = "sleep takes exactly one of `until` and `for`";
    await expectContractInput(sleep.forward({}, caps), one);
    await expectContractInput(sleep.forward({ until: "2026-10-24", for: "1s" }, caps), one);
    await expectContractInput(
      sleep.forward({ until: "2026-13-01" }, caps),
      'sleep `until` "2026-13-01": not a valid date: month must be in 1..12',
    );
    await expectContractInput(
      sleep.forward({ until: 5n }, caps),
      "sleep `until` did not resolve to a date/datetime: 5",
    );
  });

  test("a held take or sleep call still returns its request (the engine reuses the call)", async () => {
    const held: CallView = {
      callId: "r1/heard",
      tool: "take",
      state: "answered",
      closedReason: null,
      result: null,
      deadline: null,
    };
    const { caps } = capsWith([held]);
    expect(await take.forward({ from: new InboxRef("m") }, caps)).toEqual(
      new Pending([], { wait: { tool: "take", inbox: "m", mode: "one", timeoutSeconds: null } }),
    );
  });
});
