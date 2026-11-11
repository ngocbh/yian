/**
 * TS-only: `inbox_withdrawn` in the waits fold and the reducer, and `withdraw`'s unread set.
 */
import { expect, test } from "bun:test";
import type { WaitRequest } from "../../src/nodes/base.ts";
import {
  canonicalTyped,
  InboxClosedRecord,
  InboxWithdrawnRecord,
  MemoryRunLog,
  NodeStartedRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  type RunRecord,
  RunResumedRecord,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  DeliveryRefused,
  prepareInboxClosed,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareNodeWoken,
  type PreparedTransition,
  prepareRecord,
  prepareRunCancelled,
  prepareRunStarted,
  prepareRunWaiting,
  prepareSend,
  prepareWithdraw,
  RunState,
} from "../../src/runtime/state.ts";
import { applyWaitsRecord, availableUnread, decideWait, withdrawableEvents, WaitsState } from "../../src/runtime/waits.ts";
import { prepareDelivery } from "../../src/runtime/engine.ts";
import { FIXED_CLOCK, RUN_ID, startData } from "../helpers/run_state_fixtures.ts";
import { call as callOf, started, steerFlow, takeEntry, tool, waitFlow } from "../helpers/waits_fixtures.ts";

const commit = (node: string, key: string, value: unknown) => new PoolCommitRecord(node, key, canonicalTyped(value, null)[0]);
const send = (inbox: string, value: unknown) => commit("__start__", `__start__.${inbox}`, value);
const takeWait = (node: string, mode: "one" | "all", inbox = "msgs") =>
  new NodeWaitingRecord(node, [`r/${node}`], { waits: [{ callId: `r/${node}`, tool: "take", inbox, mode, deadline: null }] });
const takeCommit = (node: string, events: number[], closed = false) =>
  new PoolCommitRecord(node, node, null, { take: { inbox: "msgs", events, closed } });
const takeOne = (): WaitRequest => ({ tool: "take", inbox: "msgs", mode: "one", timeoutSeconds: null });
const call = (node: string) => `r/${node}`;

/** Fold `records` at seqs 0, 1, ... into a fresh state. */
function fold(...records: RunRecord[]): WaitsState {
  const w = new WaitsState();
  records.forEach((record, seq) => applyWaitsRecord(w, { seq, record }));
  return w;
}

/** The run start's commits: a system value, then the START inputs (seqs 0 and 1). */
const START = [commit("__start__", "system.run_id", "r"), commit("__start__", "__start__", {})];

test("U is the unread sends and those assigned to a take that hasn't woken, in seq order", () => {
  // seqs: 0,1 START; 2 take a (one); 3 send x -> a; 4 send y unread
  const w = fold(...START, takeWait("a", "one"), send("msgs", "x"), send("msgs", "y"));
  expect(withdrawableEvents(w, "msgs")).toEqual([3, 4]);
  expect(withdrawableEvents(w, "other")).toEqual([]);
});

test("after node_woken the assigned sends are the take's, not withdrawable", () => {
  const w = fold(...START, takeWait("a", "one"), send("msgs", "x"), new NodeWokenRecord("a"));
  expect(withdrawableEvents(w, "msgs")).toEqual([]);
});

test("withdraw un-assigns a shifted take:one and a take:all; absorbing order is restored by call seq; the next send goes to the oldest", () => {
  // 2 take a (one), 3 take b (all), 4 x -> a (a shifts out), 5 y -> b, 6 withdraw [4, 5]
  const w = fold(
    ...START,
    takeWait("a", "one"),
    takeWait("b", "all"),
    send("msgs", "x"),
    send("msgs", "y"),
    new InboxWithdrawnRecord("msgs", [4, 5]),
  );
  expect(w.calls.get(call("a"))).toMatchObject({ state: "open", assigned: [] });
  expect(w.calls.get(call("b"))).toMatchObject({ state: "open", assigned: [] });
  expect(w.inboxes.get("msgs")).toEqual({ unread: [], closed: false, absorbing: [call("a"), call("b")] });
  applyWaitsRecord(w, { seq: 7, record: send("msgs", "z") });
  expect(w.calls.get(call("a"))!.assigned).toEqual([7]);
});

test("withdraw on a closed inbox answers the take with the close; it takes {events: [], closed: true}", () => {
  const w = fold(...START, takeWait("a", "one"), send("msgs", "x"), new InboxClosedRecord("msgs"), new InboxWithdrawnRecord("msgs", [3]));
  expect(w.calls.get(call("a"))).toMatchObject({ state: "answered", assigned: [], closedInbox: true });
  expect(decideWait(w, call("a"), takeOne(), 0)).toEqual({ kind: "take", commit: { inbox: "msgs", events: [], closed: true } });
});

test("an orphan inline take commit's events are withdrawn; run_resumed then re-offers nothing", () => {
  // send x (2); inline take commit by `t` holding [2] (3, orphan); withdraw [2]; take a waits; resume
  const w = fold(
    ...START,
    send("msgs", "x"),
    new NodeStartedRecord("t"),
    takeCommit("t", [2]),
    new InboxWithdrawnRecord("msgs", [2]),
    takeWait("a", "one"),
    new RunResumedRecord(["t"], 2),
  );
  expect(w.inboxes.get("msgs")!.unread).toEqual([]);
  expect(w.calls.get(call("a"))!.state).toBe("open");
});

test("a withdrawn send leaves the steer queue", () => {
  const w = fold(...START, send("msgs", "x"), new InboxWithdrawnRecord("msgs", [2]));
  expect(availableUnread(w, "msgs")).toEqual([]);
});

// --- The reducer ------------------------------------------------------------------------------ //

type Run = ReturnType<typeof started>;

/** START, then `heard` (take: one) waits on msgs, then run_waiting. */
function takeRun(): Run {
  const run = started(waitFlow(tool("heard", "take")));
  const { log, state } = run;
  commitPrepared(log, state, prepareNodeStarted(state, "heard"));
  commitPrepared(log, state, prepareNodeWaiting(state, "heard", { callIds: [callOf("heard")], waits: [takeEntry("heard")] }));
  commitPrepared(log, state, prepareRunWaiting(state));
  return run;
}

/** Commit a send of `event` to msgs and return its seq. */
function commitSend(run: Run, event: unknown): number {
  const prepared = prepareSend(run.state, "msgs", event);
  commitPrepared(run.log, run.state, prepared);
  return prepared.expectedSequence;
}

/** Return the `DeliveryRefused` reason `run` throws. */
function refusal(run: () => unknown): string {
  try {
    run();
  } catch (error) {
    expect(error).toBeInstanceOf(DeliveryRefused);
    return (error as DeliveryRefused).reason;
  }
  throw new Error("expected a DeliveryRefused");
}

test("prepareWithdraw names the whole unread set; commit drops the values and reopens the take", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  const y = commitSend(run, "y");
  const prepared = prepareWithdraw(run.state, "msgs")!;
  expect([...(prepared.record as InboxWithdrawnRecord).events]).toEqual([x, y]);
  expect(prepared.observers).toEqual([]);
  expect(commitPrepared(run.log, run.state, prepared)).toEqual([]);
  expect(run.state.inboxEvents.has(x) || run.state.inboxEvents.has(y)).toBe(false);
  expect(run.state.waits.calls.get(callOf("heard"))!.state).toBe("open");
  // Not a frontier kind: the last frontier record stays.
  expect(run.state.waits.lastFrontier).toBe("run_waiting");
});

test("nothing unread: null, nothing appended", () => {
  const run = takeRun();
  expect(prepareWithdraw(run.state, "msgs")).toBeNull();
  const x = commitSend(run, "x");
  commitPrepared(run.log, run.state, prepareNodeWoken(run.state, "heard"));
  // x is the woken take's now.
  expect(run.state.inboxEvents.has(x)).toBe(true);
  expect(prepareWithdraw(run.state, "msgs")).toBeNull();
});

test("allowed on a closed inbox; the waiting take is answered by the close", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  commitPrepared(run.log, run.state, prepareInboxClosed(run.state, "msgs")!);
  const prepared = prepareWithdraw(run.state, "msgs")!;
  expect([...(prepared.record as InboxWithdrawnRecord).events]).toEqual([x]);
  commitPrepared(run.log, run.state, prepared);
  expect(run.state.waits.calls.get(callOf("heard"))).toMatchObject({ state: "answered", assigned: [], closedInbox: true });
});

test("prepareWithdraw refusals: ended, not_started, unknown_inbox; a non-string inbox is a TypeError", () => {
  const run = takeRun();
  expect(refusal(() => prepareWithdraw(run.state, "nope"))).toBe("unknown_inbox");
  expect(() => prepareWithdraw(run.state, 7 as never)).toThrow(TypeError);
  commitPrepared(run.log, run.state, prepareRunCancelled(run.state));
  expect(refusal(() => prepareWithdraw(run.state, "msgs"))).toBe("ended");
  const fresh = new RunState(waitFlow(tool("heard", "take")));
  commitPrepared(new MemoryRunLog(RUN_ID, { clock: FIXED_CLOCK }), fresh, prepareRunStarted(fresh, startData()));
  expect(refusal(() => prepareWithdraw(fresh, "msgs"))).toBe("not_started");
});

test.each<[string, (x: number, y: number) => number[]]>([
  ["a partial list", (x) => [x]],
  ["an extra seq", (x, y) => [x, y, y + 1]],
])("replay refuses an inbox_withdrawn naming %s", (_, events) => {
  const run = takeRun();
  const x = commitSend(run, "x");
  const y = commitSend(run, "y");
  expect(() => prepareRecord(run.state, new InboxWithdrawnRecord("msgs", events(x, y)))).toThrow(
    `InboxWithdrawn must name the whole unread set [${x}, ${y}] of inbox 'msgs'`,
  );
});

test("replay refuses an inbox_withdrawn naming a seq a woken take holds, or an undeclared inbox", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  commitPrepared(run.log, run.state, prepareNodeWoken(run.state, "heard"));
  expect(() => prepareRecord(run.state, new InboxWithdrawnRecord("msgs", [x]))).toThrow(/whole unread set \[\]/);
  expect(() => prepareRecord(run.state, new InboxWithdrawnRecord("nope", [x]))).toThrow(/no inbox the START declares/);
});

test("a steer step can't consume a withdrawn send", () => {
  const run = started(steerFlow());
  const x = commitSend(run, "x");
  commitPrepared(run.log, run.state, prepareWithdraw(run.state, "msgs")!);
  commitPrepared(run.log, run.state, prepareNodeStarted(run.state, "chat"));
  expect(() => prepareNodeStep(run.state, "chat", { kind: "steer", inbox: "msgs", from: [BigInt(x)] })).toThrow(/whole unread queue \[\]/);
});

test("a withdraw replays to the same waits", () => {
  const run = takeRun();
  commitSend(run, "x");
  commitPrepared(run.log, run.state, prepareWithdraw(run.state, "msgs")!);
  const back = WaitsState.fromJSON(JSON.parse(JSON.stringify(run.state.waits.toJSON())));
  expect(back.toJSON()).toEqual(run.state.waits.toJSON());
});

test("the withdraw delivery op prepares the withdraw", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  const prepared = prepareDelivery(run.state, { kind: "withdraw", inbox: "msgs" }) as PreparedTransition;
  expect(prepared.record).toEqual(new InboxWithdrawnRecord("msgs", [x]));
  expect(() => prepareDelivery(run.state, { kind: "nope" } as never)).toThrow(/send, withdraw, complete, close, interrupt or cancel/);
});
