import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { CallClosed, NodeWaiting, type RunEvent, RunCancelled, RunWaiting } from "../../src/events.ts";
import { PoolCommitRecord } from "../../src/runtime/log.ts";
import { DeliveryRefused } from "../../src/runtime/state.ts";
import { FakeHost, FakeTake, GateNode, inboxEngine, inboxFlow, recordNames } from "./inbox_fixtures.ts";

/** A run whose host call `ask` waits, with inbox `m` declared, already resolved waiting. */
async function waitingRun() {
  const host = new FakeHost("ask");
  const run = inboxEngine(inboxFlow(["m"], [host], [[START_ID, "ask"], ["ask", END_ID]]));
  await run.run();
  return { run, host };
}

/** The `DeliveryRefused` `deliver` throws for `op`. */
function refusal(run: Awaited<ReturnType<typeof waitingRun>>["run"], op: Parameters<typeof run.deliver>[0]): DeliveryRefused {
  try {
    run.deliver(op);
  } catch (error) {
    if (error instanceof DeliveryRefused) return error;
    throw error;
  }
  throw new Error("expected a DeliveryRefused");
}

test("deliver before run() is an Error, and nothing is appended", () => {
  const run = inboxEngine(inboxFlow(["m"], [new FakeHost("ask")], [[START_ID, "ask"], ["ask", END_ID]]));
  const before = run.log.nextSequence;
  expect(() => run.deliver({ kind: "send", inbox: "m", event: "hi" })).toThrow("FlowEngine.deliver needs run() first");
  expect(run.log.nextSequence).toBe(before);
});

test("a refused delivery appends nothing", async () => {
  const { run } = await waitingRun();
  const before = run.log.nextSequence;
  expect(refusal(run, { kind: "send", inbox: "nope", event: "hi" }).reason).toBe("unknown_inbox");
  expect(run.log.nextSequence).toBe(before);
});

test("a send to a declared inbox after run() resolved waiting appends it", async () => {
  const { run } = await waitingRun();
  const out = run.deliver({ kind: "send", inbox: "m", event: "hi" });
  expect(out.seq).toBe(run.log.nextSequence - 1);
  const record = run.log.records()[out.seq!]!;
  expect(record.seq).toBe(out.seq!);
  expect(record.record).toBeInstanceOf(PoolCommitRecord);
});

test("complete answers once; an identical repeat is a no-op and a different one is call_settled", async () => {
  const { run, host } = await waitingRun();
  const first = run.deliver({ kind: "complete", callId: host.callId, result: { output: "yes" } });
  expect(first.seq).toBe(run.log.nextSequence - 1);
  expect(run.deliver({ kind: "complete", callId: host.callId, result: { output: "yes" } })).toEqual({ seq: null, events: [], record: null });
  const settled = refusal(run, { kind: "complete", callId: host.callId, result: { output: "no" } });
  expect([settled.reason, settled.callState]).toEqual(["call_settled", "answered"]);
});

test("closing a closed inbox is a no-op", async () => {
  const { run } = await waitingRun();
  expect(run.deliver({ kind: "close", inbox: "m" }).seq).toBe(run.log.nextSequence - 1);
  expect(run.deliver({ kind: "close", inbox: "m" })).toEqual({ seq: null, events: [], record: null });
});

test.each([0, 2])("a cancel during a live run ends it cancelled and aborts running nodes (%p workers)", async (numWorkers) => {
  const host = new FakeHost("ask");
  const gate = new GateNode("gate");
  const waited = Promise.withResolvers<void>();
  const run = inboxEngine(
    inboxFlow([], [host, gate], [[START_ID, "ask"], [START_ID, "gate"], ["ask", END_ID], ["gate", END_ID]]),
    {
      numWorkers,
      onEvent: (event: RunEvent) => {
        if (event instanceof NodeWaiting && event.nodeId === "ask") waited.resolve();
      },
    },
  );
  const running = run.run();
  await Promise.all([gate.started, waited.promise]);
  const out = run.deliver({ kind: "cancel" });
  const events = await running;
  expect(out.events).toEqual([new RunCancelled(), new CallClosed({ callId: host.callId, reason: "cancelled" })]);
  expect(gate.aborted).toBe(true);
  expect(run.state.result).toBeInstanceOf(RunCancelled);
  expect(events.at(-2)).toBeInstanceOf(RunCancelled);
  const names = recordNames(run.log);
  expect(names.at(-1)).toBe("RunCancelledRecord");
  expect(names).not.toContain("RunStoppedRecord");
  expect(refusal(run, { kind: "complete", callId: host.callId, result: { output: "late" } }).reason).toBe("ended");
});

test("after the loop has ended, a throwing onEvent rethrows, the record stays, and no run_stopped follows", async () => {
  const failure = new Error("observer failed");
  let armed = false;
  const run = inboxEngine(inboxFlow(["m"], [new FakeHost("ask")], [[START_ID, "ask"], ["ask", END_ID]]), {
    onEvent: () => {
      if (armed) throw failure;
    },
  });
  const events = await run.run();
  expect(events.at(-1)).toBeInstanceOf(RunWaiting);
  armed = true;
  const before = run.log.nextSequence;
  // A send has no observers; a cancel's are RunCancelled and its closed calls.
  expect(() => run.deliver({ kind: "cancel" })).toThrow(failure);
  expect(run.log.nextSequence).toBe(before + 1);
  expect(recordNames(run.log).at(-1)).toBe("RunCancelledRecord");
});

test("after the loop has ended, a rejection from onEvent's promise is ignored", async () => {
  let armed = false;
  const run = inboxEngine(inboxFlow(["m"], [new FakeHost("ask")], [[START_ID, "ask"], ["ask", END_ID]]), {
    onEvent: () => (armed ? Promise.reject(new Error("late observer failed")) : undefined),
  });
  await run.run();
  armed = true;
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown): void => void unhandled.push(reason);
  process.on("unhandledRejection", onUnhandled);
  try {
    expect(run.deliver({ kind: "cancel" }).events[0]).toBeInstanceOf(RunCancelled);
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }
  expect(unhandled).toEqual([]);
});

test("an observer's delivery keeps run()'s events in log order, ending with the terminal", async () => {
  const run: ReturnType<typeof inboxEngine> = inboxEngine(
    inboxFlow(["m"], [new FakeTake("heard", "m")], [[START_ID, "heard"], ["heard", END_ID]]),
    {
      onEvent: (event: RunEvent) => {
        if (event instanceof RunWaiting) run.deliver({ kind: "cancel" });
      },
    },
  );
  const events = await run.run();
  expect(events.slice(-3).map((event) => event.constructor.name)).toEqual(["NodeWaiting", "RunWaiting", "RunCancelled"]);
  expect(recordNames(run.log).slice(-3)).toEqual(["NodeWaitingRecord", "RunWaitingRecord", "RunCancelledRecord"]);
});
