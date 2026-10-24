/**
 * The waits fold (`applyWaitsRecord`) over records alone, no reducer: sends, absorbing takes,
 * inline take commits, host calls, wakes, closes and settles; its queries; and `decideWait`.
 * TS-only: Python yian has no inboxes or waits.
 */
import { expect, test } from "bun:test";
import { Failure } from "../../src/diagnostics.ts";
import type { WaitRequest } from "../../src/nodes/base.ts";
import {
  canonicalTyped,
  InboxClosedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  RunCancelledRecord,
  RunFailedRecord,
  type RunRecord,
  RunResumedRecord,
  RunStoppedRecord,
  RunWaitingRecord,
} from "../../src/runtime/log.ts";
import {
  applyWaitsRecord,
  availableUnread,
  decideWait,
  dueTimeouts,
  isWakeable,
  nodeCalls,
  unreadEvents,
  wakeableNodes,
  wakeAt,
  waitInfos,
  WaitsState,
} from "../../src/runtime/waits.ts";

const commit = (node: string, key: string, value: unknown) => new PoolCommitRecord(node, key, canonicalTyped(value, null)[0]);
const send = (inbox: string, value: unknown) => commit("__start__", `__start__.${inbox}`, value);
const takeWait = (node: string, mode: "one" | "all", inbox = "msgs", deadline: string | null = null) =>
  new NodeWaitingRecord(node, [`r/${node}`], { waits: [{ callId: `r/${node}`, tool: "take", inbox, mode, deadline }] });
const sleepWait = (node: string, deadline: string) =>
  new NodeWaitingRecord(node, [`r/${node}`], { waits: [{ callId: `r/${node}`, tool: "sleep", inbox: null, mode: null, deadline }] });
const pending = (node: string, n: number, callId: string, extra: Record<string, unknown> = {}) =>
  new NodeStepRecord(node, n, { kind: "pending", call_id: callId, tool: "ask", arguments: { q: "ok?" }, ...extra });
const answer = (node: string, n: number, callId: string) =>
  new NodeStepRecord(node, n, { kind: "tool_result", call_id: callId, output: { kind: "str", value: "yes" } });
const takeCommit = (node: string, events: number[], closed = false) =>
  new PoolCommitRecord(node, node, null, { take: { inbox: "msgs", events, closed } });
const failure = (node: string) => new Failure({ code: "node.execute", message: "boom", nodeId: node });

/** Fold `records` at seqs 0, 1, ... into a fresh state. */
function fold(...records: RunRecord[]): WaitsState {
  const w = new WaitsState();
  records.forEach((record, seq) => applyWaitsRecord(w, { seq, record }));
  return w;
}

/** The run start's commits: a system value, then the START inputs (seqs 0 and 1). */
const START = [commit("__start__", "system.run_id", "r"), commit("__start__", "__start__", {})];

test("a send before the run start is ignored; after it, unread", () => {
  const w = fold(send("msgs", "early"), ...START, send("msgs", "hi"));
  expect(unreadEvents(w)).toEqual(new Map([["msgs", [3]]]));
});

test("a commit that isn't a send leaves the inboxes alone", () => {
  const w = fold(...START, commit("work", "work", 1), commit("each#0/__start__", "each#0/__start__", {}));
  expect(w.inboxes.size).toBe(0);
});

test("two takes on one inbox: first node_waiting first; one stops at one", () => {
  const w = fold(...START, takeWait("a", "one"), takeWait("b", "all"), send("msgs", "x"), send("msgs", "y"), send("msgs", "z"));
  expect(nodeCalls(w, "a")[0]).toMatchObject({ state: "answered", assigned: [4] });
  expect(nodeCalls(w, "b")[0]).toMatchObject({ state: "answered", assigned: [5, 6] });
  expect(w.inboxes.get("msgs")!.absorbing).toEqual(["r/b"]);
  expect(wakeableNodes(w)).toEqual(["a", "b"]);
});

test("a (all) then b (one): after node_woken a, the next send goes to b, not unread", () => {
  const w = fold(...START, takeWait("a", "all"), takeWait("b", "one"), send("msgs", "x"), new NodeWokenRecord("a"), send("msgs", "y"));
  expect(nodeCalls(w, "a")[0]!.assigned).toEqual([4]);
  expect(nodeCalls(w, "b")[0]!.assigned).toEqual([6]);
  expect(unreadEvents(w).get("msgs") ?? []).toEqual([]);
});

test("frontier records set lastFrontier; others don't", () => {
  const kinds: [RunRecord, string][] = [
    [new NodeStartedRecord("heard"), "node_started"],
    [takeWait("heard", "one"), "node_waiting"],
    [new RunWaitingRecord(["heard"]), "run_waiting"],
    [send("msgs", "x"), "run_waiting"],
    [new NodeWokenRecord("heard"), "node_woken"],
    [new RunStoppedRecord("callback", "", ["heard"]), "run_stopped"],
    [new RunResumedRecord(["heard"], 2), "run_resumed"],
    [new NodeSucceededRecord("heard", 3), "node_succeeded"],
  ];
  const w = fold(...START);
  expect(w.lastFrontier).toBeNull();
  kinds.forEach(([record, kind], index) => {
    applyWaitsRecord(w, { seq: index + 2, record });
    expect(w.lastFrontier).toBe(kind);
  });
});

test("inbox_closed answers an absorbing take with nothing assigned and keeps a take:all's events", () => {
  const w = fold(...START, takeWait("a", "all"), send("msgs", "x"), takeWait("b", "one", "other"), new InboxClosedRecord("msgs"));
  expect(nodeCalls(w, "a")[0]).toMatchObject({ state: "answered", assigned: [3], closedInbox: false });
  expect(w.inboxes.get("msgs")).toEqual({ unread: [], closed: true, absorbing: [] });
  const empty = fold(...START, takeWait("a", "one"), new InboxClosedRecord("msgs"));
  expect(nodeCalls(empty, "a")[0]).toMatchObject({ state: "answered", assigned: [], closedInbox: true });
  expect(isWakeable(empty, "a")).toBe(true);
  expect(fold(...START, new InboxClosedRecord("msgs")).inboxes.get("msgs")).toEqual({ unread: [], closed: true, absorbing: [] });
});

test("an inline take holds its events until its success removes them from unread", () => {
  const records = [...START, send("msgs", "x"), send("msgs", "y"), new NodeStartedRecord("heard"), takeCommit("heard", [2])];
  const w = fold(...records);
  expect(unreadEvents(w).get("msgs")).toEqual([2, 3]);
  expect(availableUnread(w, "msgs")).toEqual([3]);
  applyWaitsRecord(w, { seq: 6, record: new NodeSucceededRecord("heard", 5) });
  expect(unreadEvents(w).get("msgs")).toEqual([3]);
  expect(availableUnread(w, "msgs")).toEqual([3]);
  expect(w._pendingTakes.size).toBe(0);
});

test("an orphan take commit then run_resumed leaves its events unread and available", () => {
  const w = fold(...START, send("msgs", "x"), new NodeStartedRecord("heard"), takeCommit("heard", [2]), new RunResumedRecord(["heard"], 2));
  expect(unreadEvents(w).get("msgs")).toEqual([2]);
  expect(availableUnread(w, "msgs")).toEqual([2]);
});

test("a waiting take's success retires its call and touches no unread event", () => {
  const w = fold(
    ...START,
    new NodeStartedRecord("heard"),
    takeWait("heard", "one"),
    send("msgs", "x"),
    new NodeWokenRecord("heard"),
    send("msgs", "y"),
    takeCommit("heard", [4]),
    new NodeSucceededRecord("heard", 7),
  );
  expect(nodeCalls(w, "heard")).toEqual([]);
  expect(unreadEvents(w).get("msgs")).toEqual([6]);
});

test("a pending step registers a host call; its tool_result answers it", () => {
  const w = fold(...START, new NodeStartedRecord("ask"), pending("ask", 1, "r/ask/1", { deadline: "2026-10-24T09:00:00.000Z" }));
  expect(nodeCalls(w, "ask")).toEqual([
    {
      callId: "r/ask/1",
      nodeId: "ask",
      tool: "ask",
      resultKey: "r/ask/1",
      inbox: null,
      mode: null,
      deadline: "2026-10-24T09:00:00.000Z",
      seq: 3,
      pendingSeq: 3,
      state: "open",
      resultSeq: null,
      assigned: [],
      closedInbox: false,
      closedReason: null,
    },
  ]);
  applyWaitsRecord(w, { seq: 4, record: new NodeWaitingRecord("ask", ["r/ask/1"]) });
  expect(isWakeable(w, "ask")).toBe(false);
  applyWaitsRecord(w, { seq: 5, record: answer("ask", 2, "r/ask/1") });
  expect(nodeCalls(w, "ask")[0]).toMatchObject({ state: "answered", resultSeq: 5 });
  expect(isWakeable(w, "ask")).toBe(true);
});

test("an agent's pending call is answered by the tool_result naming its tool_call_id", () => {
  const w = fold(
    ...START,
    new NodeStartedRecord("agent"),
    pending("agent", 1, "r/agent/1/0", { tool_call_id: "call_abc" }),
    new NodeStepRecord("agent", 2, { kind: "tool_result", tool_call_id: "call_abc", output: "yes" }),
  );
  expect(nodeCalls(w, "agent")[0]).toMatchObject({ resultKey: "call_abc", state: "answered", resultSeq: 4 });
});

test("an unregistered tool_result changes nothing", () => {
  const before = fold(...START, new NodeStartedRecord("agent"), pending("agent", 1, "r/agent/1"));
  const after = fold(
    ...START,
    new NodeStartedRecord("agent"),
    pending("agent", 1, "r/agent/1"),
    new NodeStepRecord("agent", 2, { kind: "tool_result", tool_call_id: "call_provider", output: "x" }),
    new NodeStepRecord("agent", 3, { kind: "tool_result", call_id: "r/agent/9", closed: "skipped" }),
    new NodeStepRecord("agent", 4, { kind: "model_call", attempt: 1n }),
  );
  expect(nodeCalls(after, "agent")).toEqual(nodeCalls(before, "agent"));
});

test("a second tool_result for an answered call keeps the first", () => {
  const w = fold(...START, new NodeStartedRecord("ask"), pending("ask", 1, "r/ask/1"), answer("ask", 2, "r/ask/1"), answer("ask", 3, "r/ask/1"));
  expect(nodeCalls(w, "ask")[0]).toMatchObject({ state: "answered", resultSeq: 4 });
});

test("node_woken times out takes and sleeps, closes host calls timed_out, and ends absorbing", () => {
  const w = fold(
    ...START,
    takeWait("heard", "all", "msgs", "2026-10-24T09:00:00.000Z"),
    sleepWait("nap", "2026-10-24T09:00:00.000Z"),
    pending("ask", 1, "r/ask/1", { deadline: "2026-10-24T09:00:00.000Z" }),
    new NodeWaitingRecord("ask", ["r/ask/1"]),
    new NodeWokenRecord("heard", ["r/heard"]),
    new NodeWokenRecord("nap", ["r/nap"]),
    new NodeWokenRecord("ask", ["r/ask/1"]),
  );
  expect(nodeCalls(w, "heard")[0]).toMatchObject({ state: "timed_out", closedReason: null });
  expect(nodeCalls(w, "nap")[0]).toMatchObject({ state: "timed_out" });
  expect(nodeCalls(w, "ask")[0]).toMatchObject({ state: "closed", closedReason: "timed_out" });
  expect(w.inboxes.get("msgs")!.absorbing).toEqual([]);
  expect(w.waiting.size).toBe(0);
  expect(w._lastWoken).toEqual(new Map([["heard", 6], ["nap", 7], ["ask", 8]]));
});

test("an outcome settles the node's calls and bookkeeping", () => {
  for (const outcome of [
    new NodeSucceededRecord("ask", 9),
    new NodeRoutedRecord("ask", "yes"),
    new NodeFailedRecord(failure("ask")),
  ]) {
    const w = fold(...START, pending("ask", 1, "r/ask/1"), new NodeWaitingRecord("ask", ["r/ask/1"]), answer("ask", 2, "r/ask/1"), new NodeWokenRecord("ask"), outcome);
    expect(nodeCalls(w, "ask")).toEqual([]);
    expect(w._lastWoken.size).toBe(0);
  }
});

test("node_failed closes every other open call failed", () => {
  const w = fold(...START, pending("ask", 1, "r/ask/1"), new NodeWaitingRecord("ask", ["r/ask/1"]), takeWait("heard", "one"), new NodeFailedRecord(failure("work")));
  expect(nodeCalls(w, "ask")[0]).toMatchObject({ state: "closed", closedReason: "failed" });
  expect(nodeCalls(w, "heard")[0]).toMatchObject({ state: "closed", closedReason: "failed" });
});

test("run_cancelled closes open calls cancelled and run_failed closes them failed; answered calls stay", () => {
  const records = [...START, pending("ask", 1, "r/ask/1"), answer("ask", 2, "r/ask/1"), takeWait("heard", "one")];
  const cancelled = fold(...records, new RunCancelledRecord());
  expect(nodeCalls(cancelled, "ask")[0]!.state).toBe("answered");
  expect(nodeCalls(cancelled, "heard")[0]).toMatchObject({ state: "closed", closedReason: "cancelled" });
  const unreachable = new Failure({ code: "graph.terminal_unreachable", message: "END is unreachable", nodeId: "choice" });
  const failed = fold(...records, new RunFailedRecord(unreachable));
  expect(nodeCalls(failed, "heard")[0]).toMatchObject({ state: "closed", closedReason: "failed" });
});

test("run_resumed closes its closed_calls crashed and drops pending take commits", () => {
  const w = fold(
    ...START,
    new NodeStartedRecord("agent"),
    pending("agent", 1, "r/agent/1/0", { tool_call_id: "c1" }),
    new RunResumedRecord(["agent"], 2, ["r/agent/1/0"]),
  );
  expect(nodeCalls(w, "agent")[0]).toMatchObject({ state: "closed", closedReason: "crashed" });
});

test("a node waiting on two host calls is wakeable only when both settle", () => {
  const w = fold(
    ...START,
    pending("agent", 1, "r/a/0"),
    pending("agent", 2, "r/a/1"),
    new NodeWaitingRecord("agent", ["r/a/0", "r/a/1"]),
    answer("agent", 3, "r/a/0"),
  );
  expect(isWakeable(w, "agent")).toBe(false);
  expect(wakeableNodes(w)).toEqual([]);
  applyWaitsRecord(w, { seq: 6, record: answer("agent", 4, "r/a/1") });
  expect(isWakeable(w, "agent")).toBe(true);
  expect(isWakeable(w, "nobody")).toBe(false);
});

const T9 = "2026-10-24T09:00:00.000Z";
const T10 = "2026-10-24T10:00:00.000Z";
const NOW = Date.UTC(2026, 9, 24, 9, 30);

test("dueTimeouts names the waiting nodes a timeout would make wakeable", () => {
  const w = fold(
    ...START,
    sleepWait("nap", T9),
    sleepWait("later", T10),
    takeWait("heard", "one", "msgs", T9),
    send("msgs", "x"),
    pending("agent", 1, "r/a/0", { deadline: T9 }),
    pending("agent", 2, "r/a/1"),
    new NodeWaitingRecord("agent", ["r/a/0", "r/a/1"]),
    pending("ask", 1, "r/ask/1", { deadline: T9 }),
    new NodeWaitingRecord("ask", ["r/ask/1"]),
  );
  // heard has an assigned event, so it is answered, never due; agent has another open call that isn't due.
  expect(dueTimeouts(w, NOW)).toEqual(new Map([["nap", ["r/nap"]], ["ask", ["r/ask/1"]]]));
  expect(dueTimeouts(w, Date.UTC(2026, 9, 24, 8))).toEqual(new Map());
  expect(dueTimeouts(w, Date.parse(T9))).toEqual(new Map([["nap", ["r/nap"]], ["ask", ["r/ask/1"]]]));
});

test("wakeAt is the earliest open deadline of a waiting node", () => {
  expect(wakeAt(fold(...START))).toBeNull();
  expect(wakeAt(fold(...START, takeWait("heard", "one")))).toBeNull();
  expect(wakeAt(fold(...START, sleepWait("later", T10), sleepWait("nap", T9)))).toBe(T9);
  // An answered take's deadline no longer counts, nor a running node's call.
  expect(wakeAt(fold(...START, takeWait("heard", "one", "msgs", T9), send("msgs", "x"), sleepWait("later", T10)))).toBe(T10);
  expect(wakeAt(fold(...START, new NodeStartedRecord("ask"), pending("ask", 1, "r/ask/1", { deadline: T9 })))).toBeNull();
  expect(wakeAt(fold(...START, sleepWait("nap", T9), new NodeWokenRecord("nap", ["r/nap"])))).toBeNull();
});

test("waitInfos lists each waiting node's calls, settled or not", () => {
  const w = fold(...START, takeWait("heard", "one"), sleepWait("nap", T9), send("msgs", "x"), pending("ask", 1, "r/ask/1"), new NodeWaitingRecord("ask", ["r/ask/1"]));
  expect(waitInfos(w)).toEqual([
    { node: "heard", callId: "r/heard", tool: "take", inbox: "msgs", deadline: null, settled: true },
    { node: "nap", callId: "r/nap", tool: "sleep", inbox: null, deadline: T9, settled: false },
    { node: "ask", callId: "r/ask/1", tool: "ask", inbox: null, deadline: null, settled: false },
  ]);
  expect(waitInfos(fold(...START))).toEqual([]);
});

const AT9 = Date.UTC(2026, 9, 24, 9);
const takeOne = (timeoutSeconds: number | null = null): WaitRequest => ({ tool: "take", inbox: "msgs", mode: "one", timeoutSeconds });
const takeAll = (timeoutSeconds: number | null = null): WaitRequest => ({ tool: "take", inbox: "msgs", mode: "all", timeoutSeconds });
const sleepFor = (forSeconds: number): WaitRequest => ({ tool: "sleep", until: null, forSeconds });
const sleepUntil = (until: string): WaitRequest => ({ tool: "sleep", until, forSeconds: null });

test.each<[string, RunRecord[], WaitRequest, unknown]>([
  ["unread x, y; take one", [send("msgs", "x"), send("msgs", "y")], takeOne(), { kind: "take", commit: { inbox: "msgs", events: [2], closed: false } }],
  ["unread x, y; take all", [send("msgs", "x"), send("msgs", "y")], takeAll(), { kind: "take", commit: { inbox: "msgs", events: [2, 3], closed: false } }],
  ["unread x on a closed inbox; take all", [send("msgs", "x"), new InboxClosedRecord("msgs")], takeAll(), { kind: "take", commit: { inbox: "msgs", events: [2], closed: true } }],
  [
    "unread x, y on a closed inbox; take one",
    [send("msgs", "x"), send("msgs", "y"), new InboxClosedRecord("msgs")],
    takeOne(),
    { kind: "take", commit: { inbox: "msgs", events: [2], closed: false } },
  ],
  ["a closed empty inbox; take one with a timeout", [new InboxClosedRecord("msgs")], takeOne(60), { kind: "take", commit: { inbox: "msgs", events: [], closed: true } }],
  ["an empty inbox; timeout 0", [], takeOne(0), { kind: "take", commit: { inbox: "msgs", events: [], closed: false } }],
  [
    "an empty inbox; take all for 1800 s",
    [],
    takeAll(1800),
    { kind: "wait", entry: { callId: "r/heard", tool: "take", inbox: "msgs", mode: "all", deadline: "2026-10-24T09:30:00.000Z" } },
  ],
  [
    "an empty inbox; take one, no timeout",
    [],
    takeOne(),
    { kind: "wait", entry: { callId: "r/heard", tool: "take", inbox: "msgs", mode: "one", deadline: null } },
  ],
  [
    "unread x held by another pending take commit",
    [send("msgs", "x"), new NodeStartedRecord("other"), takeCommit("other", [2])],
    takeOne(),
    { kind: "wait", entry: { callId: "r/heard", tool: "take", inbox: "msgs", mode: "one", deadline: null } },
  ],
  [
    "sleep for 90 s",
    [],
    sleepFor(90),
    { kind: "wait", entry: { callId: "r/heard", tool: "sleep", inbox: null, mode: null, deadline: "2026-10-24T09:01:30.000Z" } },
  ],
  [
    "sleep until tomorrow",
    [],
    sleepUntil("2026-10-25T00:00:00.000Z"),
    { kind: "wait", entry: { callId: "r/heard", tool: "sleep", inbox: null, mode: null, deadline: "2026-10-25T00:00:00.000Z" } },
  ],
  ["sleep until a past time", [], sleepUntil("2026-10-24T08:00:00.000Z"), { kind: "slept" }],
  ["sleep until now", [], sleepUntil("2026-10-24T09:00:00.000Z"), { kind: "slept" }],
  ["sleep for 0 s", [], sleepFor(0), { kind: "slept" }],
])("decideWait: %s", (_name, records, request, decision) => {
  const w = fold(...START, ...records);
  expect(decideWait(w, "r/heard", request, AT9)).toEqual(decision as never);
});

test("decideWait refuses a sleep until a time that isn't a record time", () => {
  expect(() => decideWait(fold(...START), "r/nap", sleepUntil("tomorrow"), AT9)).toThrow(
    "sleep until must be a UTC time such as 2026-10-20T14:02:11.123Z, found 'tomorrow'",
  );
});

test("decideWait reuses a held take or sleep call", () => {
  const answered = fold(...START, takeWait("heard", "one"), send("msgs", "x"), new NodeWokenRecord("heard"));
  expect(decideWait(answered, "r/heard", takeOne(), AT9)).toEqual({ kind: "take", commit: { inbox: "msgs", events: [3], closed: false } });
  const all = fold(...START, takeWait("heard", "all"), send("msgs", "x"), send("msgs", "y"), new InboxClosedRecord("msgs"), new NodeWokenRecord("heard"));
  expect(decideWait(all, "r/heard", takeAll(), AT9)).toEqual({ kind: "take", commit: { inbox: "msgs", events: [3, 4], closed: true } });
  // take: one with a later unread event: the inbox is closed, but not drained.
  const later = fold(...START, takeWait("heard", "one"), send("msgs", "x"), send("msgs", "y"), new InboxClosedRecord("msgs"), new NodeWokenRecord("heard"));
  expect(decideWait(later, "r/heard", takeOne(), AT9)).toEqual({ kind: "take", commit: { inbox: "msgs", events: [3], closed: false } });
  const byClose = fold(...START, takeWait("heard", "one"), new InboxClosedRecord("msgs"), new NodeWokenRecord("heard"));
  expect(decideWait(byClose, "r/heard", takeOne(), AT9)).toEqual({ kind: "take", commit: { inbox: "msgs", events: [], closed: true } });
  const timedOut = fold(...START, takeWait("heard", "one", "msgs", T9), new NodeWokenRecord("heard", ["r/heard"]), send("msgs", "x"));
  expect(decideWait(timedOut, "r/heard", takeOne(60), AT9)).toEqual({ kind: "take", commit: { inbox: "msgs", events: [], closed: false } });
  const slept = fold(...START, sleepWait("heard", T9), new NodeWokenRecord("heard", ["r/heard"]));
  expect(decideWait(slept, "r/heard", sleepFor(60), AT9)).toEqual({ kind: "slept" });
});

test("decideWait refuses a call still open or one that isn't a take or sleep", () => {
  expect(() => decideWait(fold(...START, takeWait("heard", "one")), "r/heard", takeOne(), AT9)).toThrow(
    "call 'r/heard' is still open; its node can't decide its wait",
  );
  const ask = fold(...START, pending("heard", 1, "r/heard"));
  expect(() => decideWait(ask, "r/heard", takeOne(), AT9)).toThrow("call 'r/heard' isn't a take or sleep (tool 'ask')");
  const cancelled = fold(...START, takeWait("heard", "one"), new RunCancelledRecord());
  expect(() => decideWait(cancelled, "r/heard", takeOne(), AT9)).toThrow("call 'r/heard' was closed (cancelled)");
});
