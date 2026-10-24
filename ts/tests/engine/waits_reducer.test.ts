/**
 * The run reducer's rules for inboxes, calls and waits: sends, `node_waiting`, `node_woken` and
 * `run_waiting`, `pending` and `tool_result` steps, `complete`, the take commit, orphan takes at a
 * resume, `inbox_closed`, `run_cancelled` and `run_failed`, and the refusals a host delivery
 * meets. TS-only: Python yian has no inboxes or waits.
 */
import { describe, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeWaiting, NodeWoken, RunSucceeded, RunWaiting } from "../../src/events.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import {
  MemoryRunLog,
  NodeWaitingRecord,
  PoolCommitRecord,
  type RunLog,
  RunWaitingRecord,
  type WaitEntry,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  DeliveryRefused,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareNodeWoken,
  preparePoolCommit,
  type PreparedTransition,
  prepareRecord,
  prepareRunResumed,
  prepareRunStarted,
  prepareRunWaiting,
  prepareSend,
  RunState,
} from "../../src/runtime/state.ts";
import { TraceDelta } from "../../src/runtime/trace.ts";
import { logJson } from "../../src/typesys/json.ts";
import { inboxType, Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import { expectUnchanged, FIXED_CLOCK, RUN_ID, snapshot, startData } from "./run_state_fixtures.ts";
import { call, MSGS, paramFlow, started, succeed, tool, waitFlow } from "./waits_fixtures.ts";

describe("DeliveryRefused", () => {
  test("is a RangeError naming why, with the call's state and cause when given", () => {
    const cause = new TypeCheckError("expected str");
    const plain = new DeliveryRefused("unknown_inbox", "no inbox \"other\"");
    expect(plain).toBeInstanceOf(RangeError);
    expect([plain.name, plain.reason, plain.callState, plain.cause, plain.message]).toEqual([
      "DeliveryRefused",
      "unknown_inbox",
      null,
      undefined,
      "no inbox \"other\"",
    ]);
    const settled = new DeliveryRefused("call_settled", "settled", { callState: "answered", cause });
    expect([settled.reason, settled.callState, settled.cause]).toEqual(["call_settled", "answered", cause]);
  });
});

/** Return the `DeliveryRefused` `run` throws, after checking it appended nothing to `state`. */
function refused(state: RunState, log: RunLog, run: () => unknown): DeliveryRefused {
  const before = snapshot(state);
  const next = log.nextSequence;
  let caught: unknown;
  try {
    run();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(DeliveryRefused);
  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(next);
  return caught as DeliveryRefused;
}

/** Prepare and commit `prepared`, returning its seq. */
function commit(log: RunLog, state: RunState, prepared: PreparedTransition): number {
  commitPrepared(log, state, prepared);
  return prepared.expectedSequence;
}

/** Commit a send of `event` to `inbox` (default `msgs`) and return its seq. */
function send(log: RunLog, state: RunState, event: unknown, inbox = "msgs"): number {
  return commit(log, state, prepareSend(state, inbox, event));
}

/** Run `nodeId`'s start, commit `output` and its success, then END's; the run succeeds. */
function finish(log: RunLog, state: RunState, nodeId: string, output: unknown): void {
  commitPrepared(log, state, prepareNodeStarted(state, nodeId));
  succeed(log, state, nodeId, output);
  commitPrepared(log, state, prepareNodeStarted(state, END_ID));
  succeed(log, state, END_ID, output);
}

const FLOATS = new Parameter("floats", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.NUMBER))) });
const NAME = new Parameter("name", { contract: new ValueContract(Type.scalar(ValueKind.STRING)) });

describe("sends", () => {
  test("a send lands in the inbox store and unread, never the pool or the producers", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")));
    const pool = state.pool.toWire();
    const producers = [...state.producers];
    const ready = [...state.ready];
    const seq = send(log, state, "hi");
    expect(logJson(state.inboxEvents.get(seq)!.toWire())).toBe(logJson({ kind: "str", value: "hi" }));
    expect(state.waits.inboxes.get("msgs")!.unread).toEqual([seq]);
    expect(state.pool.toWire()).toEqual(pool);
    expect([...state.producers]).toEqual(producers);
    expect(state.ready).toEqual(ready);
    expect(state.pendingCommits.size).toBe(0);
  });

  test("refusals come in order and append nothing", () => {
    const ended = started(waitFlow(tool("ask", "ask")));
    finish(ended.log, ended.state, "ask", "done");
    expect(ended.state.result).toBeInstanceOf(RunSucceeded);
    expect(refused(ended.state, ended.log, () => prepareSend(ended.state, "other", 3)).reason).toBe("ended");

    const fresh = new RunState(waitFlow(tool("heard", "take")));
    const log = new MemoryRunLog(RUN_ID, { clock: FIXED_CLOCK });
    commitPrepared(log, fresh, prepareRunStarted(fresh, startData()));
    expect(refused(fresh, log, () => prepareSend(fresh, "other", 3)).reason).toBe("not_started");

    const run = started(paramFlow([MSGS, NAME], tool("heard", "take")));
    expect(refused(run.state, run.log, () => prepareSend(run.state, "other", "hi")).reason).toBe("unknown_inbox");
    expect(refused(run.state, run.log, () => prepareSend(run.state, "name", "hi")).reason).toBe("unknown_inbox");
    const invalid = refused(run.state, run.log, () => prepareSend(run.state, "msgs", 3));
    expect(invalid.reason).toBe("invalid_value");
    expect(invalid.cause).toBeInstanceOf(TypeCheckError);
  });

  test("the size cap is the run's MAX_EVENT_BYTES over the value's run-log JSON", () => {
    const run = started(waitFlow(tool("heard", "take")), { defaults: { MAX_EVENT_BYTES: 7n } });
    send(run.log, run.state, "hello");
    expect(refused(run.state, run.log, () => prepareSend(run.state, "msgs", "hello!")).reason).toBe("too_large");
    const floats = started(paramFlow([MSGS, FLOATS], tool("heard", "take")));
    send(floats.log, floats.state, 1.5, "floats");
    const small = started(paramFlow([MSGS, FLOATS], tool("heard", "take")), { defaults: { MAX_EVENT_BYTES: 2n } });
    expect(refused(small.state, small.log, () => prepareSend(small.state, "floats", 1.5)).reason).toBe("too_large");
  });

  test("replay refuses a send before the run start's inputs and a value not canonical for the inbox", () => {
    const fresh = new RunState(waitFlow(tool("heard", "take")));
    const log = new MemoryRunLog(RUN_ID, { clock: FIXED_CLOCK });
    commitPrepared(log, fresh, prepareRunStarted(fresh, startData()));
    expect(() => prepareRecord(fresh, new PoolCommitRecord(START_ID, `${START_ID}.msgs`, { kind: "str", value: "hi" }))).toThrow(
      /system\.<name> or the START/,
    );
    const run = started(waitFlow(tool("heard", "take")));
    expect(() => prepareRecord(run.state, new PoolCommitRecord(START_ID, `${START_ID}.msgs`, { kind: "int", value: 1n }))).toThrow(
      RangeError,
    );
    expect(() => prepareRecord(run.state, new PoolCommitRecord("heard", `${START_ID}.msgs`, { kind: "str", value: "hi" }))).toThrow(
      RangeError,
    );
  });

  test("a learnable run's send changes no trace but its sequence", () => {
    const run = started(waitFlow(tool("heard", "take")), { learnable: true });
    const prepared = prepareSend(run.state, "msgs", "hi");
    const delta = prepared._plan.traceDelta!;
    expect(delta).toBeInstanceOf(TraceDelta);
    expect([delta.started, delta.occurrence, delta.writer, delta.systemValue, delta.resumed]).toEqual([null, null, null, null, false]);
    commitPrepared(run.log, run.state, prepared);
    expect(run.state.trace!.nextSequence).toBe(run.state.nextSequence);
  });
});

const DEADLINE = "2026-10-21T00:00:00.000Z";

/** A take wait entry for `node` on `inbox`. */
function takeEntry(node: string, options: { inbox?: string; deadline?: string | null; mode?: "one" | "all" } = {}): WaitEntry {
  return { callId: call(node), tool: "take", inbox: options.inbox ?? "msgs", mode: options.mode ?? "one", deadline: options.deadline ?? null };
}

/** A sleep wait entry for `node` until `deadline`. */
function sleepEntry(node: string, deadline = DEADLINE): WaitEntry {
  return { callId: call(node), tool: "sleep", inbox: null, mode: null, deadline };
}

/** Start `node` (the ready head) and return the start's seq. */
function start(log: RunLog, state: RunState, node: string): number {
  return commit(log, state, prepareNodeStarted(state, node));
}

/** Commit `node`'s wait on its own take or sleep `entry`, returning the seq. */
function wait(log: RunLog, state: RunState, node: string, entry: WaitEntry): number {
  return commit(log, state, prepareNodeWaiting(state, node, { callIds: [entry.callId], waits: [entry] }));
}

describe("node_waiting, node_woken and run_waiting", () => {
  test("a waiting node leaves flight but keeps its start count; run_waiting names the waiting nodes", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")));
    start(log, state, "heard");
    const prepared = prepareNodeWaiting(state, "heard", { callIds: [call("heard")], waits: [takeEntry("heard")] });
    expect(prepared.observers).toEqual([new NodeWaiting({ nodeId: "heard", callIds: [call("heard")] })]);
    commitPrepared(log, state, prepared);
    expect([...state.inFlight]).toEqual([]);
    expect(state.acceptedStarts.has("heard")).toBe(false);
    expect(state.nodeStarts.get("heard")).toBe(1);
    expect([...state.waits.waiting.keys()]).toEqual(["heard"]);
    expect(state.waits.calls.get(call("heard"))!.state).toBe("open");
    const runWaiting = prepareRunWaiting(state);
    expect(runWaiting.record).toEqual(new RunWaitingRecord(["heard"]));
    expect(runWaiting.observers).toEqual([new RunWaiting({ nodeIds: ["heard"] })]);
    commitPrepared(log, state, runWaiting);
    expect(state.waits.lastFrontier).toBe("run_waiting");
  });

  test("node_waiting refusals", () => {
    const run = started(waitFlow(tool("a", "take"), tool("b", "take"), tool("ask", "ask"), tool("nap", "sleep")));
    const { log, state } = run;
    const attempt = (node: string, callIds: readonly string[], waits: readonly WaitEntry[] = [], steerInbox: string | null = null) =>
      expect(() => prepareRecord(state, new NodeWaitingRecord(node, callIds, { waits, steerInbox }))).toThrow(RangeError);
    attempt("a", [call("a")], [takeEntry("a")]); // not in flight
    for (const node of ["a", "b", "ask", "nap"]) start(log, state, node);
    attempt("ask", [call("ask")], [takeEntry("ask")]); // not a take or sleep
    attempt("a", [call("b")], [{ ...takeEntry("a"), callId: call("b") }]); // another call id
    attempt("a", [call("a")], [sleepEntry("a")]); // another tool
    attempt("a", [call("a")], [takeEntry("a", { inbox: "other" })]); // undeclared inbox
    attempt("a", [call("a")], [takeEntry("a")], "msgs"); // a steer inbox
    attempt("b", ["elsewhere"]); // an unregistered call
    commitPrepared(log, state, preparePoolCommit(state, "nap", null));
    attempt("nap", [call("nap")], [sleepEntry("nap")]); // a pending commit
    wait(log, state, "a", takeEntry("a"));
    attempt("b", [call("a")]); // another node's call
    const s = send(log, state, "hi");
    expect(state.waits.calls.get(call("a"))!.assigned).toEqual([s]);
    commitPrepared(log, state, prepareNodeWoken(state, "a"));
    attempt("a", [call("a")], [takeEntry("a")]); // a call the node already holds
    attempt("a", [call("a")]); // registered before the node's last wake
  });

  test("a take can't wait while its inbox holds an event it may read", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")));
    send(log, state, "early");
    start(log, state, "heard");
    expect(() => prepareNodeWaiting(state, "heard", { callIds: [call("heard")], waits: [takeEntry("heard")] })).toThrow(
      /holds unread events/,
    );
  });

  test("node_woken moves a wakeable node back in flight with the wake as its accepted start", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")));
    expect(() => prepareNodeWoken(state, "heard")).toThrow(/not waiting/);
    start(log, state, "heard");
    wait(log, state, "heard", takeEntry("heard"));
    expect(() => prepareNodeWoken(state, "heard")).toThrow(/still waits on open call/);
    expect(() => prepareNodeWoken(state, "heard", [call("heard")])).toThrow(/with a deadline/);
    send(log, state, "hi");
    const prepared = prepareNodeWoken(state, "heard");
    expect(prepared.observers).toEqual([new NodeWoken({ nodeId: "heard", timedOut: [] })]);
    const seq = commit(log, state, prepared);
    expect([...state.inFlight]).toEqual(["heard"]);
    expect(state.acceptedStarts.get("heard")).toBe(seq);
    expect(state.nodeStarts.get("heard")).toBe(1);
    expect(state.waits.waiting.size).toBe(0);
  });

  test("node_woken times out open calls with a deadline, but not an answered take", () => {
    const { log, state } = started(waitFlow(tool("a", "take"), tool("b", "take")));
    start(log, state, "a");
    start(log, state, "b");
    wait(log, state, "a", takeEntry("a", { deadline: DEADLINE }));
    wait(log, state, "b", takeEntry("b", { deadline: DEADLINE }));
    send(log, state, "hi");
    expect(() => prepareNodeWoken(state, "a", [call("a")])).toThrow(/open call/);
    commitPrepared(log, state, prepareNodeWoken(state, "b", [call("b")]));
    expect(state.waits.calls.get(call("b"))!.state).toBe("timed_out");
  });

  test("run_waiting refusals", () => {
    const { log, state } = started(waitFlow(tool("a", "take"), tool("b", "take")));
    expect(() => prepareRunWaiting(state)).toThrow(/in flight or ready/);
    start(log, state, "a");
    start(log, state, "b");
    expect(() => prepareRunWaiting(state)).toThrow(/in flight or ready/);
    wait(log, state, "a", takeEntry("a"));
    expect(() => prepareRunWaiting(state)).toThrow(/in flight or ready/);
    wait(log, state, "b", takeEntry("b"));
    expect(() => prepareRecord(state, new RunWaitingRecord(["b", "a"]))).toThrow(/exactly equal the waiting nodes/);
    commitPrepared(log, state, prepareRunWaiting(state));
    expect(() => prepareRunWaiting(state)).toThrow(/follows a RunWaiting/);
    send(log, state, "hi");
    expect(() => prepareRunWaiting(state)).toThrow(/wakeable/);
    const none = started(waitFlow(tool("a", "take")));
    none.state.ready.length = 0;
    expect(() => prepareRunWaiting(none.state)).toThrow(/requires a waiting node/);
  });

  test("an initialized state keeps waiting nodes out of flight, with start counts", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")));
    start(log, state, "heard");
    const seq = wait(log, state, "heard", takeEntry("heard"));
    state.inFlight.add("heard");
    state.acceptedStarts.set("heard", seq);
    expect(() => prepareRunWaiting(state)).toThrow(/waiting node must have a start count and not be in flight/);
    state.inFlight.delete("heard");
    state.acceptedStarts.delete("heard");
    state.nodeStarts.delete("heard");
    expect(() => prepareRunWaiting(state)).toThrow(/waiting node must have a start count/);
  });

  test("a learnable run parks a waiting node's start through a resume and brings it back at the wake", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")), { learnable: true });
    const startSeq = start(log, state, "heard");
    wait(log, state, "heard", takeEntry("heard"));
    expect([...state.trace!.started]).toEqual([]);
    expect([...state.trace!._parked]).toEqual([["heard", startSeq]]);
    commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
    expect([...state.trace!._parked]).toEqual([["heard", startSeq]]);
    send(log, state, "hi");
    commitPrepared(log, state, prepareNodeWoken(state, "heard"));
    expect([...state.trace!.started]).toEqual([["heard", startSeq]]);
    expect(state.trace!._parked.size).toBe(0);
    state.trace!._parked.set("ghost", 1);
    expect(() => prepareNodeStep(state, "heard", { kind: "note" })).toThrow(/parks other starts/);
  });
});
