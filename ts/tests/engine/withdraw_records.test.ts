/**
 * TS-only: `inbox_withdrawn` in the waits fold and the reducer, and `withdraw`'s unread set.
 */
import { expect, test } from "bun:test";
import type { WaitRequest } from "../../src/nodes/base.ts";
import {
  canonicalTyped,
  InboxClosedRecord,
  InboxWithdrawnRecord,
  NodeStartedRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  type RunRecord,
  RunResumedRecord,
} from "../../src/runtime/log.ts";
import { applyWaitsRecord, availableUnread, decideWait, withdrawableEvents, WaitsState } from "../../src/runtime/waits.ts";

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
