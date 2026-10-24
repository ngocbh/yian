import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import type { Node } from "../../src/nodes/base.ts";
import { FakeHost, inboxEngine, inboxFlow, recordNames } from "./inbox_fixtures.ts";
import { engine, graph, RecordNode, rejected } from "./run_state_fixtures.ts";

test("a host Pending waits, and the quiescent run records run_waiting and resolves", async () => {
  const host = new FakeHost("ask");
  const run = inboxEngine(inboxFlow([], [host], [[START_ID, "ask"], ["ask", END_ID]]), { defaults: { MAX_EVENT_BYTES: 4096n } });
  const events = await run.run();
  expect(recordNames(run.log).slice(-3)).toEqual(["NodeStepRecord", "NodeWaitingRecord", "RunWaitingRecord"]);
  expect(run.state.result).toBeNull();
  expect([...run.state.waits.waiting.keys()]).toEqual(["ask"]);
  expect(events.slice(-2).map((event) => event.constructor.name)).toEqual(["NodeWaiting", "RunWaiting"]);
  // The run's recorded limit, not the process's.
  expect(host.limits).toEqual([4096]);
});

test.each([0, 2])("with nothing waiting, nonterminal quiescence still throws (%p workers)", async (numWorkers) => {
  const built = engine(graph([new RecordNode("a", [])], [[START_ID, "a"], ["a", END_ID]]), { numWorkers });
  built.state.ready.length = 0;
  expect(await rejected(built.run())).toEqual(new Error("FlowEngine reached nonterminal quiescence"));
});

import { NodeStarted, NodeSucceeded, type RunEvent } from "../../src/events.ts";
import { NodeStartedRecord, NodeSucceededRecord, NodeWaitingRecord, PoolCommitRecord } from "../../src/runtime/log.ts";
import { FakeTake } from "./inbox_fixtures.ts";

/** A take `heard` on inbox `m` between START and END. */
const takeFlow = (take = new FakeTake("heard", "m")) => inboxFlow(["m"], [take], [[START_ID, "heard"], ["heard", END_ID]]);

/** The take commit in `run`'s log. */
function takeCommit(run: ReturnType<typeof inboxEngine>): PoolCommitRecord | undefined {
  return run.log.records().map((entry) => entry.record).find((record) => record instanceof PoolCommitRecord && record.take !== null) as
    | PoolCommitRecord
    | undefined;
}

test("a take with an unread event commits inline", async () => {
  let sent: number | null = null;
  const run: ReturnType<typeof inboxEngine> = inboxEngine(takeFlow(), {
    onEvent: (event: RunEvent) => {
      if (event instanceof NodeSucceeded && event.nodeId === START_ID) sent = run.deliver({ kind: "send", inbox: "m", event: "hi" }).seq;
    },
  });
  await run.run();
  const records = run.log.records();
  const commitAt = records.findIndex((entry) => entry.record instanceof PoolCommitRecord && entry.record.take !== null);
  expect(records[commitAt]!.record).toMatchObject({ nodeId: "heard", take: { inbox: "m", events: [sent!], closed: false } });
  expect(records[commitAt + 1]!.record).toBeInstanceOf(NodeSucceededRecord);
  expect(recordNames(run.log)).not.toContain("NodeWaitingRecord");
  expect(run.state.pool.get("heard")).toEqual({ events: ["hi"], closed: false });
});

test("a take on an empty open inbox waits", async () => {
  const run = inboxEngine(takeFlow());
  await run.run();
  expect(recordNames(run.log).slice(-2)).toEqual(["NodeWaitingRecord", "RunWaitingRecord"]);
  const waiting = run.log.records().at(-2)!.record as NodeWaitingRecord;
  expect(waiting.waits).toEqual([{ callId: "run-123/heard", tool: "take", inbox: "m", mode: "one", deadline: null }]);
});

test("a send landing while the take evaluates is read by its decision", async () => {
  let sent: number | null = null;
  const run: ReturnType<typeof inboxEngine> = inboxEngine(takeFlow(), {
    onEvent: (event: RunEvent) => {
      // The take's evaluation awaits before its outcome, so the send lands before the decision.
      if (event instanceof NodeStarted && event.nodeId === "heard") {
        queueMicrotask(() => {
          sent = run.deliver({ kind: "send", inbox: "m", event: "hi" }).seq;
        });
      }
    },
  });
  await run.run();
  expect(sent).not.toBeNull();
  expect(recordNames(run.log)).not.toContain("NodeWaitingRecord");
  expect(takeCommit(run)!.take).toEqual({ inbox: "m", events: [sent!], closed: false });
});

import { RunFailed, RunSucceeded } from "../../src/events.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { formatRecordTime } from "../../src/runtime/log.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { FakeSleep, type HandClock, handClock } from "./inbox_fixtures.ts";

const T0 = Date.UTC(2026, 9, 23, 12, 0, 0);

/** A sleep `nap` between START and END. */
const sleepFlow = (nap: FakeSleep) => inboxFlow([], [nap], [[START_ID, "nap"], ["nap", END_ID]]);

test("a sleep that is already over commits null inline", async () => {
  const run = inboxEngine(sleepFlow(new FakeSleep("nap", 0)), { scheduler: handClock(T0) });
  const events = await run.run();
  const records = run.log.records();
  const commitAt = records.findIndex((entry) => entry.record instanceof PoolCommitRecord && entry.record.nodeId === "nap");
  expect(records[commitAt + 1]!.record).toMatchObject({ nodeId: "nap" });
  expect(records[commitAt + 1]!.record).toBeInstanceOf(NodeSucceededRecord);
  expect(run.state.pool.get("nap")).toBeNull();
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("a sleep waits until its deadline", async () => {
  const run = inboxEngine(sleepFlow(new FakeSleep("nap", 60)), { scheduler: handClock(T0) });
  await run.run();
  const waiting = run.log.records().at(-2)!.record as NodeWaitingRecord;
  expect(waiting.waits).toEqual([{ callId: "run-123/nap", tool: "sleep", inbox: null, mode: null, deadline: formatRecordTime(T0 + 60_000) }]);
  expect(recordNames(run.log).at(-1)).toBe("RunWaitingRecord");
});

test("a sleep whose output contract refuses null fails contract.output", async () => {
  const nap = new FakeSleep("nap", 0);
  nap.outputContract = new ValueContract(Type.scalar(ValueKind.STRING));
  const run = inboxEngine(sleepFlow(nap), { scheduler: handClock(T0) });
  const events = await run.run();
  expect(events.at(-1)).toBeInstanceOf(RunFailed);
  expect((events.at(-1) as RunFailed).failure.code).toBe("contract.output");
});

import { NodeWoken, RunWaiting } from "../../src/events.ts";
import { NodeWokenRecord, RunWaitingRecord } from "../../src/runtime/log.ts";
import { DeliveryRefused } from "../../src/runtime/state.ts";
import { FakeHost as Host, GateNode } from "./inbox_fixtures.ts";

/** The index of the first record in `run`'s log matching `match`. */
function indexOf(run: ReturnType<typeof inboxEngine>, match: (record: unknown) => boolean): number {
  const at = run.log.records().findIndex((entry) => match(entry.record));
  if (at < 0) throw new Error("no such record");
  return at;
}
const succeededOf = (nodeId: string) => (record: unknown) => record instanceof NodeSucceededRecord && record.nodeId === nodeId;
const waitingOf = (nodeId: string) => (record: unknown) => record instanceof NodeWaitingRecord && record.nodeId === nodeId;
const wokenOf = (nodeId: string) => (record: unknown) => record instanceof NodeWokenRecord && record.nodeId === nodeId;

/** START fans out to `first` then `second` (in that order), both feeding END. */
const pairFlow = (first: Node, second: Node, inboxes: readonly string[] = []) =>
  inboxFlow(inboxes, [first, second], [[START_ID, first.id], [START_ID, second.id], [first.id, END_ID], [second.id, END_ID]]);

test("a due sleep wakes only after the running node settles", async () => {
  const clock = handClock(T0);
  const slow = new GateNode("slow");
  const run = inboxEngine(pairFlow(new FakeSleep("nap", 1), slow), { scheduler: clock });
  const running = run.run();
  await slow.started;
  clock.set(T0 + 2000);
  slow.release();
  const events = await running;
  const waited = indexOf(run, waitingOf("nap"));
  const settled = indexOf(run, succeededOf("slow"));
  const woken = indexOf(run, wokenOf("nap"));
  expect(waited < settled && settled < woken).toBe(true);
  expect((run.log.records()[woken]!.record as NodeWokenRecord).timedOut).toEqual(["run-123/nap"]);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("serial live wake: a complete while a sibling runs wakes the host call after it", async () => {
  const ask = new Host("ask");
  const slow = new GateNode("slow");
  const run = inboxEngine(pairFlow(ask, slow), { scheduler: handClock(T0) });
  const running = run.run();
  await slow.started;
  run.deliver({ kind: "complete", callId: ask.callId, result: { output: "yes" } });
  slow.release();
  const events = await running;
  const woken = indexOf(run, wokenOf("ask"));
  expect(indexOf(run, succeededOf("slow")) < woken).toBe(true);
  expect((run.log.records()[woken]!.record as NodeWokenRecord).timedOut).toEqual([]);
  expect(run.state.pool.get("ask")).toBe("yes");
  expect(ask.handlerCalls).toBe(1);
  expect(recordNames(run.log)).not.toContain("RunResumedRecord");
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("a complete for an ask closed by its timeout is call_settled, closed", async () => {
  const clock = handClock(T0);
  const ask = new Host("ask", { tool: "ask", deadlineAt: T0 + 1000 });
  const gate = new GateNode("gate");
  let refused: unknown = null;
  const run: ReturnType<typeof inboxEngine> = inboxEngine(pairFlow(ask, gate), {
    scheduler: clock,
    onEvent: (event: RunEvent) => {
      if (!(event instanceof NodeWoken) || event.nodeId !== "ask") return;
      try {
        run.deliver({ kind: "complete", callId: ask.callId, result: { output: "late" } });
      } catch (error) {
        refused = error;
      }
    },
  });
  const running = run.run();
  await gate.started;
  clock.set(T0 + 2000);
  gate.release();
  const events = await running;
  expect(refused).toBeInstanceOf(DeliveryRefused);
  expect([(refused as DeliveryRefused).reason, (refused as DeliveryRefused).callState]).toEqual(["call_settled", "closed"]);
  expect(run.state.pool.get("ask")).toBeNull();
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("a send made from onEvent(RunWaiting) still wakes the take in the same run()", async () => {
  const run: ReturnType<typeof inboxEngine> = inboxEngine(takeFlow(), {
    onEvent: (event: RunEvent) => {
      if (event instanceof RunWaiting) run.deliver({ kind: "send", inbox: "m", event: "hi" });
    },
  });
  const events = await run.run();
  const waitingAt = indexOf(run, (record) => record instanceof RunWaitingRecord);
  expect(recordNames(run.log).slice(waitingAt, waitingAt + 5)).toEqual([
    "RunWaitingRecord",
    "PoolCommitRecord",
    "NodeWokenRecord",
    "PoolCommitRecord",
    "NodeSucceededRecord",
  ]);
  expect(run.state.pool.get("heard")).toEqual({ events: ["hi"], closed: false });
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

import { NodeWaiting } from "../../src/events.ts";
import { FuncNode } from "./run_state_fixtures.ts";

const startedOf = (nodeId: string) => (record: unknown) =>
  record instanceof NodeStartedRecord && record.nodeId === nodeId;

test("a timer wakes a sleep while another branch still runs", async () => {
  const clock: HandClock = handClock(T0, (deadline) => {
    // The engine's timer fires as soon as it is armed.
    if (deadline === T0 + 5000) clock.set(deadline);
  });
  const a = new GateNode("a");
  const b = new FakeSleep("b", 5);
  const c = new FuncNode("c", () => "after");
  const flow = inboxFlow([], [a, b, c], [[START_ID, "a"], [START_ID, "b"], ["b", "c"], ["a", END_ID], ["c", END_ID]]);
  const run = inboxEngine(flow, {
    numWorkers: 2,
    scheduler: clock,
    onEvent: (event: RunEvent) => {
      if (event instanceof NodeSucceeded && event.nodeId === "c") a.release();
    },
  });
  const events = await run.run();
  const aDone = indexOf(run, succeededOf("a"));
  expect(indexOf(run, wokenOf("b")) < aDone && indexOf(run, succeededOf("c")) < aDone).toBe(true);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("a send after node_waiting wakes the take while a sibling still runs", async () => {
  const gate = new GateNode("gate");
  let sent: number | null = null;
  const run: ReturnType<typeof inboxEngine> = inboxEngine(pairFlow(new FakeTake("heard", "m"), gate, ["m"]), {
    numWorkers: 2,
    onEvent: (event: RunEvent) => {
      if (event instanceof NodeWaiting && event.nodeId === "heard") sent = run.deliver({ kind: "send", inbox: "m", event: "hi" }).seq;
      if (event instanceof NodeSucceeded && event.nodeId === "heard") gate.release();
    },
  });
  const events = await run.run();
  expect(indexOf(run, succeededOf("heard")) < indexOf(run, succeededOf("gate"))).toBe(true);
  expect(takeCommit(run)!.take).toEqual({ inbox: "m", events: [sent!], closed: false });
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("a woken node takes no worker slot", async () => {
  const gate = new GateNode("gate");
  const run: ReturnType<typeof inboxEngine> = inboxEngine(pairFlow(new FakeTake("heard", "m"), gate, ["m"]), {
    numWorkers: 1,
    onEvent: (event: RunEvent) => {
      if (event instanceof NodeWaiting && event.nodeId === "heard") run.deliver({ kind: "send", inbox: "m", event: "hi" });
      if (event instanceof NodeSucceeded && event.nodeId === "heard") gate.release();
    },
  });
  const events = await run.run();
  // The gate holds the one worker while the woken take runs beside it.
  expect(indexOf(run, startedOf("gate")) < indexOf(run, succeededOf("heard"))).toBe(true);
  expect(events.at(-1)).toBeInstanceOf(RunSucceeded);
});

test("the engine's timer is armed while work runs, and gone at the end", async () => {
  const clock = handClock(T0);
  const gate = new GateNode("gate");
  const waited = Promise.withResolvers<void>();
  const run = inboxEngine(pairFlow(new FakeSleep("nap", 60), gate), {
    numWorkers: 2,
    scheduler: clock,
    onEvent: (event: RunEvent) => {
      if (event instanceof NodeWaiting && event.nodeId === "nap") waited.resolve();
    },
  });
  const running = run.run();
  await Promise.all([waited.promise, gate.started]);
  expect(clock.pending()).toBe(1);
  gate.release();
  const events = await running;
  expect(events.at(-1)).toBeInstanceOf(RunWaiting);
  expect(clock.pending()).toBe(0);
});
