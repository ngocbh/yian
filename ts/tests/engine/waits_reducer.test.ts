/**
 * The run reducer's rules for inboxes, calls and waits: sends, `node_waiting`, `node_woken` and
 * `run_waiting`, `pending` and `tool_result` steps, `complete`, the take commit, orphan takes at a
 * resume, `inbox_closed`, `run_cancelled` and `run_failed`, and the refusals a host delivery
 * meets. TS-only: Python yian has no inboxes or waits.
 */
import { describe, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { CallClosed, NodeWaiting, NodeWoken, RunSucceeded, RunWaiting } from "../../src/events.ts";
import type { CompleteResult } from "../../src/nodes/base.ts";
import { type CallView, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import {
  canonicalTyped,
  MemoryRunLog,
  NodeStepRecord,
  NodeWaitingRecord,
  PoolCommitRecord,
  type RunLog,
  RunWaitingRecord,
  type TakeCommit,
  type WaitEntry,
} from "../../src/runtime/log.ts";
import {
  callResultStep,
  commitPrepared,
  DeliveryRefused,
  prepareComplete,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeSucceeded,
  prepareNodeWaiting,
  prepareNodeWoken,
  preparePoolCommit,
  type PreparedTransition,
  prepareRecord,
  prepareRunResumed,
  prepareRunStarted,
  prepareRunWaiting,
  prepareSend,
  prepareTakeCommit,
  RunState,
} from "../../src/runtime/state.ts";
import { TraceDelta } from "../../src/runtime/trace.ts";
import { wakeableNodes } from "../../src/runtime/waits.ts";
import { eventBytes, type JsonObject, logJson } from "../../src/typesys/json.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { inboxType, Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import { expectUnchanged, FINGERPRINT, FIXED_CLOCK, RUN_ID, snapshot, startData } from "./run_state_fixtures.ts";
import { call, MSGS, paramFlow, started, succeed, tool, waitFlow, work } from "./waits_fixtures.ts";

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

/** An `ask` node's pending step for its own call. */
function pendingStep(node: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return { kind: "pending", call_id: call(node), tool: "ask", arguments: { prompt: "ok?" }, deadline: DEADLINE, ...extra };
}

/** A tool_result step answering `node`'s call with `answer`. */
function resultStep(node: string, answer = "yes"): Record<string, unknown> {
  return { kind: "tool_result", call_id: call(node), output: { kind: "str", value: answer } };
}

/** Commit `step` on `node` and return its seq. */
function step(log: RunLog, state: RunState, node: string, body: Record<string, unknown>): number {
  return commit(log, state, prepareNodeStep(state, node, body));
}

describe("pending and tool_result steps", () => {
  test("a running host tool node's pending step registers an open call", () => {
    const { log, state } = started(waitFlow(tool("ask", "ask")));
    start(log, state, "ask");
    const seq = step(log, state, "ask", pendingStep("ask"));
    const registered = state.waits.calls.get(call("ask"))!;
    expect([registered.nodeId, registered.tool, registered.state, registered.pendingSeq, registered.deadline]).toEqual([
      "ask",
      "ask",
      "open",
      seq,
      DEADLINE,
    ]);
  });

  test("pending refusals", () => {
    const { log, state } = started(waitFlow(tool("ask", "ask"), tool("heard", "take"), tool("echo", "echo"), work()));
    for (const node of ["ask", "heard", "echo", "work"]) start(log, state, node);
    const refuse = (node: string, body: Record<string, unknown>) =>
      expect(() => prepareNodeStep(state, node, body)).toThrow(RangeError);
    refuse("work", pendingStep("work", { tool: "work" })); // no host tool call
    refuse("heard", pendingStep("heard", { tool: "take", deadline: undefined })); // the engine decides a take
    refuse("ask", pendingStep("ask", { tool: "take" }));
    refuse("ask", pendingStep("ask", { tool: "sleep" }));
    refuse("ask", pendingStep("ask", { tool: "echo" }));
    refuse("ask", pendingStep("ask", { call_id: call("echo") }));
    for (const key of ["tool_call_id", "reply", "index"]) refuse("ask", pendingStep("ask", { [key]: "x" }));
    refuse("echo", { kind: "pending", call_id: call("echo"), tool: "echo", arguments: {}, deadline: DEADLINE });
    refuse("ask", pendingStep("ask", { deadline: "tomorrow" }));
    refuse("ask", pendingStep("ask", { arguments: ["ok?"] }));
    step(log, state, "echo", { kind: "pending", call_id: call("echo"), tool: "echo", arguments: {} });
    step(log, state, "ask", pendingStep("ask"));
    refuse("ask", pendingStep("ask")); // a second registration of an open call
    step(log, state, "ask", resultStep("ask"));
    refuse("ask", pendingStep("ask")); // or of an answered one
  });

  test("a pending step fits the run's MAX_EVENT_BYTES, measured over the whole step", () => {
    const size = eventBytes(pendingStep("ask"));
    const at = started(waitFlow(tool("ask", "ask")), { defaults: { MAX_EVENT_BYTES: BigInt(size) } });
    start(at.log, at.state, "ask");
    step(at.log, at.state, "ask", pendingStep("ask"));
    const over = started(waitFlow(tool("ask", "ask")), { defaults: { MAX_EVENT_BYTES: BigInt(size - 1) } });
    start(over.log, over.state, "ask");
    expect(() => prepareNodeStep(over.state, "ask", pendingStep("ask"))).toThrow(/over the run's limit/);
  });

  test("tool_result rules hold for registered calls only", () => {
    const { log, state } = started(waitFlow(tool("a", "ask"), tool("b", "ask")));
    start(log, state, "a");
    start(log, state, "b");
    step(log, state, "a", pendingStep("a"));
    expect(() => prepareNodeStep(state, "b", resultStep("a"))).toThrow(/of node 'a', not 'b'/);
    // An agent's result, which no pending step registered, is the node's own detail.
    step(log, state, "b", { kind: "tool_result", call_id: "call-1", output: "skipped", is_error: true, closed: "skipped" });
    step(log, state, "a", resultStep("a"));
    expect(state.waits.calls.get(call("a"))!.state).toBe("answered");
    expect(() => prepareNodeStep(state, "a", resultStep("a"))).toThrow(/answered/);
  });

  test("a waiting node takes only a tool_result answering one of its open calls", () => {
    const { log, state } = started(waitFlow(tool("ask", "ask")));
    start(log, state, "ask");
    step(log, state, "ask", pendingStep("ask"));
    commitPrepared(log, state, prepareNodeWaiting(state, "ask", { callIds: [call("ask")] }));
    expect(() => prepareNodeStep(state, "ask", { kind: "note" })).toThrow(/waiting node/);
    expect(() => prepareNodeStep(state, "ask", pendingStep("ask"))).toThrow(/waiting node/);
    expect(() => prepareNodeStep(state, "ask", { kind: "tool_result", call_id: "call-1", output: "x" })).toThrow(/waiting node/);
    step(log, state, "ask", resultStep("ask"));
    expect(wakeableNodes(state.waits)).toEqual(["ask"]);
    expect(() => prepareNodeStep(state, "ask", resultStep("ask"))).toThrow(/answered/);
  });

  test("node_waiting naming a call answered while the node ran is wakeable at once", () => {
    const { log, state } = started(waitFlow(tool("ask", "ask")));
    start(log, state, "ask");
    step(log, state, "ask", pendingStep("ask"));
    step(log, state, "ask", resultStep("ask"));
    commitPrepared(log, state, prepareNodeWaiting(state, "ask", { callIds: [call("ask")] }));
    expect(wakeableNodes(state.waits)).toEqual(["ask"]);
  });

  test("a crashed run accepts an answer; the restarted node commits from it without waiting", () => {
    const { log, state } = started(waitFlow(tool("ask", "ask")));
    start(log, state, "ask");
    step(log, state, "ask", pendingStep("ask"));
    step(log, state, "ask", resultStep("ask"));
    commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
    start(log, state, "ask");
    succeed(log, state, "ask", "yes");
    expect(state.waits.calls.size).toBe(0);
  });

  test("a restart after a pending step waits on the same call, with no second pending step", () => {
    const { log, state } = started(waitFlow(tool("ask", "ask")));
    start(log, state, "ask");
    step(log, state, "ask", pendingStep("ask"));
    commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
    start(log, state, "ask");
    expect(() => prepareNodeStep(state, "ask", pendingStep("ask"))).toThrow(/already registered/);
    commitPrepared(log, state, prepareNodeWaiting(state, "ask", { callIds: [call("ask")] }));
    expect(state.waits.waiting.get("ask")!.callIds).toEqual([call("ask")]);
  });

  test("a timed-out ask closes, publishing CallClosed, and can't be waited on again", () => {
    const { log, state } = started(waitFlow(tool("ask", "ask")));
    start(log, state, "ask");
    step(log, state, "ask", pendingStep("ask"));
    commitPrepared(log, state, prepareNodeWaiting(state, "ask", { callIds: [call("ask")] }));
    const woken = prepareNodeWoken(state, "ask", [call("ask")]);
    expect(woken.observers).toEqual([
      new NodeWoken({ nodeId: "ask", timedOut: [call("ask")] }),
      new CallClosed({ callId: call("ask"), reason: "timed_out" }),
    ]);
    commitPrepared(log, state, woken);
    expect([state.waits.calls.get(call("ask"))!.state, state.waits.calls.get(call("ask"))!.closedReason]).toEqual([
      "closed",
      "timed_out",
    ]);
    expect(() => prepareNodeWaiting(state, "ask", { callIds: [call("ask")] })).toThrow(/before the node's last node_woken/);
    expect(() => prepareNodeStep(state, "ask", resultStep("ask"))).toThrow(/closed/);
  });
});

/** An `ask` TOOL node that shapes a host's answer as a `str` output, or its error. */
class AskNode extends ToolNode {
  constructor(id: string) {
    super(id, { toolId: "ask" });
  }

  override waitResultStep(view: CallView, result: CompleteResult): JsonObject {
    if ("error" in result) return { kind: "tool_result", call_id: view.callId, error: { type: "CompletedError", message: result.error } };
    if (typeof result.output !== "string") throw new TypeCheckError(`ask answers a str, got ${typeof result.output}`);
    return { kind: "tool_result", call_id: view.callId, output: canonicalTyped(result.output, null)[0] as unknown as JsonObject };
  }
}

/** A state whose `ask` (an `AskNode`) waits on its pending call and whose `heard` take waits too. */
function askWaiting(options: { defaults?: Record<string, bigint> } = {}) {
  const run = started(waitFlow(new AskNode("ask"), tool("heard", "take")), { fingerprint: FINGERPRINT, ...options });
  start(run.log, run.state, "ask");
  start(run.log, run.state, "heard");
  step(run.log, run.state, "ask", pendingStep("ask"));
  commitPrepared(run.log, run.state, prepareNodeWaiting(run.state, "ask", { callIds: [call("ask")] }));
  wait(run.log, run.state, "heard", takeEntry("heard"));
  return run;
}

describe("complete", () => {
  test("appends the host's answer as the call's tool_result; an identical repeat is a no-op", () => {
    const { log, state } = askWaiting();
    const prepared = prepareComplete(state, call("ask"), { output: "yes" })!;
    expect(prepared.record).toEqual(new NodeStepRecord("ask", 2, resultStep("ask") as JsonObject));
    commitPrepared(log, state, prepared);
    expect(wakeableNodes(state.waits)).toEqual(["ask"]);
    expect(callResultStep(state, call("ask"))).toEqual(resultStep("ask") as JsonObject);
    expect(prepareComplete(state, call("ask"), { output: "yes" })).toBeNull();
  });

  test("refusals come in order and append nothing", () => {
    const fresh = new RunState(waitFlow(new AskNode("ask")));
    const log = new MemoryRunLog(RUN_ID, { clock: FIXED_CLOCK });
    commitPrepared(log, fresh, prepareRunStarted(fresh, startData()));
    expect(refused(fresh, log, () => prepareComplete(fresh, call("ask"), { output: "yes" })).reason).toBe("not_started");

    const run = askWaiting();
    const reason = (callId: string, result: CompleteResult) => refused(run.state, run.log, () => prepareComplete(run.state, callId, result));
    expect(reason("nobody", { output: "yes" }).reason).toBe("unknown_call");
    expect(reason(call("heard"), { output: "yes" }).reason).toBe("not_host_call");
    const invalid = reason(call("ask"), { output: 3 });
    expect(invalid.reason).toBe("invalid_value");
    expect(invalid.cause).toBeInstanceOf(TypeCheckError);
    commitPrepared(run.log, run.state, prepareComplete(run.state, call("ask"), { output: "yes" })!);
    const different = reason(call("ask"), { output: "no" });
    expect([different.reason, different.callState]).toEqual(["call_settled", "answered"]);

    const small = askWaiting({ defaults: { MAX_EVENT_BYTES: BigInt(eventBytes(pendingStep("ask"))) } });
    const long = "y".repeat(200);
    expect(refused(small.state, small.log, () => prepareComplete(small.state, call("ask"), { output: long })).reason).toBe(
      "too_large",
    );
  });

  test("a call closed by its timeout refuses any answer", () => {
    const { log, state } = askWaiting();
    commitPrepared(log, state, prepareNodeWoken(state, "ask", [call("ask")]));
    const closed = refused(state, log, () => prepareComplete(state, call("ask"), { output: "yes" }));
    expect([closed.reason, closed.callState]).toEqual(["call_settled", "closed"]);
  });

  test("a run that ended refuses any answer first", () => {
    const run = started(waitFlow(new AskNode("ask")), { fingerprint: FINGERPRINT });
    finish(run.log, run.state, "ask", "done");
    expect(refused(run.state, run.log, () => prepareComplete(run.state, call("ask"), { output: "yes" })).reason).toBe("ended");
  });

  test("an answer to a running node's call is accepted", () => {
    const run = started(waitFlow(new AskNode("ask")), { fingerprint: FINGERPRINT });
    start(run.log, run.state, "ask");
    step(run.log, run.state, "ask", pendingStep("ask"));
    commitPrepared(run.log, run.state, prepareComplete(run.state, call("ask"), { error: "nobody knows" })!);
    expect(callResultStep(run.state, call("ask"))).toEqual({
      kind: "tool_result",
      call_id: call("ask"),
      error: { type: "CompletedError", message: "nobody knows" },
    });
  });
});

const OTHERS = new Parameter("others", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.STRING))) });

/** A take commit of `events` from `inbox` (default `msgs`). */
function takeOf(events: readonly number[], closed = false, inbox = "msgs"): TakeCommit {
  return { inbox, events, closed };
}

/** Commit `node`'s take commit and its success; return the commit's seq. */
function takeAndSucceed(log: RunLog, state: RunState, node: string, take: TakeCommit): number {
  const seq = commit(log, state, prepareTakeCommit(state, node, take));
  commitPrepared(log, state, prepareNodeSucceeded(state, node, { forwardCapture: null }));
  return seq;
}

describe("the take commit", () => {
  test("a woken take commits its assigned events; its success rebuilds the value from the inbox store", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")));
    start(log, state, "heard");
    wait(log, state, "heard", takeEntry("heard"));
    const s = send(log, state, "hi");
    commitPrepared(log, state, prepareNodeWoken(state, "heard"));
    const seq = takeAndSucceed(log, state, "heard", takeOf([s]));
    expect(state.pool.store.get("heard")!.toObject()).toEqual({ events: ["hi"], closed: false });
    expect(state.producers.get("heard")).toBe(seq);
    expect(state.inboxEvents.has(s)).toBe(false);
    expect(state.waits.calls.size).toBe(0);
    expect(state.waits.inboxes.get("msgs")!.unread).toEqual([]);
  });

  test("a woken take commits exactly what its call settled", () => {
    const { log, state } = started(paramFlow([MSGS, OTHERS], tool("heard", "take")));
    start(log, state, "heard");
    wait(log, state, "heard", takeEntry("heard"));
    const first = send(log, state, "hi");
    const later = send(log, state, "again");
    commitPrepared(log, state, prepareNodeWoken(state, "heard"));
    for (const take of [takeOf([]), takeOf([later]), takeOf([first, later]), takeOf([first], true), takeOf([first], false, "others")]) {
      expect(() => prepareTakeCommit(state, "heard", take)).toThrow(RangeError);
    }
    expect(() => prepareRecord(state, new PoolCommitRecord("heard", "elsewhere", null, { take: takeOf([first]) }))).toThrow(
      /not 'heard'/,
    );
    takeAndSucceed(log, state, "heard", takeOf([first]));
    expect(state.waits.inboxes.get("msgs")!.unread).toEqual([later]);
  });

  test("a timed-out take commits nothing, whatever arrived after", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")));
    start(log, state, "heard");
    wait(log, state, "heard", takeEntry("heard", { deadline: DEADLINE }));
    commitPrepared(log, state, prepareNodeWoken(state, "heard", [call("heard")]));
    const late = send(log, state, "late");
    expect(() => prepareTakeCommit(state, "heard", takeOf([late]))).toThrow(RangeError);
    takeAndSucceed(log, state, "heard", takeOf([]));
    expect(state.pool.store.get("heard")!.toObject()).toEqual({ events: [], closed: false });
    expect(state.waits.inboxes.get("msgs")!.unread).toEqual([late]);
  });

  test("an inline take commits the oldest event a take may read or all of them; its events leave unread at its success", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")));
    const seqs = [send(log, state, "a"), send(log, state, "b"), send(log, state, "c")];
    start(log, state, "heard");
    for (const take of [takeOf(seqs.slice(0, 2)), takeOf([seqs[1]!]), takeOf([]), takeOf([seqs[0]!], true)]) {
      expect(() => prepareTakeCommit(state, "heard", take)).toThrow(RangeError);
    }
    commitPrepared(log, state, prepareTakeCommit(state, "heard", takeOf([seqs[0]!])));
    expect(state.waits.inboxes.get("msgs")!.unread).toEqual(seqs);
    commitPrepared(log, state, prepareNodeSucceeded(state, "heard", { forwardCapture: null }));
    expect(state.waits.inboxes.get("msgs")!.unread).toEqual(seqs.slice(1));
    expect(state.pool.store.get("heard")!.toObject()).toEqual({ events: ["a"], closed: false });
  });

  test("an inline take of nothing is accepted only with nothing to read, closed as the inbox is", () => {
    const { log, state } = started(waitFlow(tool("heard", "take")));
    start(log, state, "heard");
    expect(() => prepareTakeCommit(state, "heard", takeOf([], true))).toThrow(/closed false/);
    takeAndSucceed(log, state, "heard", takeOf([]));
  });

  test("an inline take commit holds its events: a second one can't name them", () => {
    const { log, state } = started(waitFlow(tool("a", "take"), tool("b", "take")));
    start(log, state, "a");
    start(log, state, "b");
    const s = send(log, state, "hi");
    commitPrepared(log, state, prepareTakeCommit(state, "a", takeOf([s])));
    expect(() => prepareTakeCommit(state, "b", takeOf([s]))).toThrow(RangeError);
    commitPrepared(log, state, prepareTakeCommit(state, "b", takeOf([])));
  });

  test("a take commit by another node, or a value commit by a take, is refused", () => {
    const { log, state } = started(waitFlow(tool("heard", "take"), tool("ask", "ask")));
    start(log, state, "heard");
    start(log, state, "ask");
    expect(() => prepareTakeCommit(state, "ask", takeOf([]))).toThrow(/which is no take/);
    expect(() => preparePoolCommit(state, "heard", { events: [], closed: false })).toThrow(/commits the events it took/);
    expect(() =>
      prepareRecord(state, new PoolCommitRecord("heard", "heard", { kind: "object", value: { events: [], closed: false } })),
    ).toThrow(/commits the events it took/);
  });

  test("a rebuilt value its target's contract refuses is a RangeError at the success", () => {
    const strict = tool("heard", "take");
    strict.outputContract = new ValueContract(
      new Type({
        kind: ValueKind.OBJECT,
        fields: new Map([
          ["events", typeFor("list(int)", new Map())],
          ["closed", Type.scalar(ValueKind.BOOLEAN)],
        ]),
        required: new Set(["events", "closed"]),
      }),
    );
    const { log, state } = started(waitFlow(strict));
    const s = send(log, state, "hi");
    start(log, state, "heard");
    commitPrepared(log, state, prepareTakeCommit(state, "heard", takeOf([s])));
    expect(() => prepareNodeSucceeded(state, "heard", { forwardCapture: null })).toThrow(/rebuilds to a value its target refuses/);
  });
});
