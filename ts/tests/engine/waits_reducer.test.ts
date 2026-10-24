/**
 * The run reducer's rules for inboxes, calls and waits: sends, `node_waiting`, `node_woken` and
 * `run_waiting`, `pending` and `tool_result` steps, `complete`, the take commit, orphan takes at a
 * resume, `inbox_closed`, `run_cancelled` and `run_failed`, and the refusals a host delivery
 * meets. TS-only: Python yian has no inboxes or waits.
 */
import { describe, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { RunSucceeded } from "../../src/events.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { MemoryRunLog, PoolCommitRecord, type RunLog } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  DeliveryRefused,
  prepareNodeStarted,
  type PreparedTransition,
  prepareRecord,
  prepareRunStarted,
  prepareSend,
  RunState,
} from "../../src/runtime/state.ts";
import { TraceDelta } from "../../src/runtime/trace.ts";
import { logJson } from "../../src/typesys/json.ts";
import { inboxType, Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import { expectUnchanged, FIXED_CLOCK, RUN_ID, snapshot, startData } from "./run_state_fixtures.ts";
import { MSGS, paramFlow, started, succeed, tool, waitFlow } from "./waits_fixtures.ts";

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
