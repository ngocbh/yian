import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
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
import { NodeSucceededRecord, NodeWaitingRecord, PoolCommitRecord } from "../../src/runtime/log.ts";
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
import { FakeSleep, handClock } from "./inbox_fixtures.ts";

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
