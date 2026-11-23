/**
 * TS-only: `inbox_withdrawn` in the waits fold and the reducer, and `withdraw`'s unread set, whole,
 * one sender's or the chosen sends.
 */
import { expect, test } from "bun:test";
import { InboxSent } from "../../src/events.ts";
import type { Author } from "../../src/memory/base.ts";
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
  wasWithdrawn,
} from "../../src/runtime/state.ts";
import { applyWaitsRecord, availableUnread, decideWait, withdrawableEvents, WaitsState } from "../../src/runtime/waits.ts";
import { prepareDelivery } from "../../src/runtime/engine.ts";
import { FIXED_CLOCK, RUN_ID, startData } from "../helpers/run_state_fixtures.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { call as callOf, MSGS, paramFlow, replayed, started, steerFlow, takeEntry, tool, waitFlow } from "../helpers/waits_fixtures.ts";

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

test("a withdraw of some sends: the sends kept go to the takes that lost theirs, oldest first; a take that lost nothing keeps them", () => {
  // 2 take a (one), 3 x -> a (a shifts out), 4 y unread, 5 withdraw [3]: a takes y
  const w = fold(...START, takeWait("a", "one"), send("msgs", "x"), send("msgs", "y"), new InboxWithdrawnRecord("msgs", [3]));
  expect(w.calls.get(call("a"))).toMatchObject({ state: "answered", assigned: [4] });
  expect(w.inboxes.get("msgs")).toEqual({ unread: [], closed: false, absorbing: [] });
  // 2 take a (all), 3 take b (one), 4 x -> a, 5 y -> a, 6 withdraw [4]: a keeps y, b waits
  const all = fold(
    ...START,
    takeWait("a", "all"),
    takeWait("b", "one"),
    send("msgs", "x"),
    send("msgs", "y"),
    new InboxWithdrawnRecord("msgs", [4]),
  );
  expect(all.calls.get(call("a"))).toMatchObject({ state: "answered", assigned: [5] });
  expect(all.calls.get(call("b"))).toMatchObject({ state: "open", assigned: [] });
  // 2 take a (one), 3 take b (one), 4 x -> a, 5 y -> b, 6 withdraw [4]: b keeps y, a waits again
  const two = fold(
    ...START,
    takeWait("a", "one"),
    takeWait("b", "one"),
    send("msgs", "x"),
    send("msgs", "y"),
    new InboxWithdrawnRecord("msgs", [4]),
  );
  expect(two.calls.get(call("a"))).toMatchObject({ state: "open", assigned: [] });
  expect(two.calls.get(call("b"))).toMatchObject({ state: "answered", assigned: [5] });
  expect(two.inboxes.get("msgs")!.absorbing).toEqual([call("a")]);
});

test("on a closed inbox a take keeps the sends a withdraw doesn't name, and takes the close once none is left", () => {
  // 2 take a (all), 3 x -> a, 4 y -> a, 5 close, 6 withdraw [3], 7 withdraw [4]
  const w = fold(...START, takeWait("a", "all"), send("msgs", "x"), send("msgs", "y"), new InboxClosedRecord("msgs"));
  applyWaitsRecord(w, { seq: 6, record: new InboxWithdrawnRecord("msgs", [3]) });
  expect(w.calls.get(call("a"))).toMatchObject({ state: "answered", assigned: [4], closedInbox: false });
  applyWaitsRecord(w, { seq: 7, record: new InboxWithdrawnRecord("msgs", [4]) });
  expect(w.calls.get(call("a"))).toMatchObject({ state: "answered", assigned: [], closedInbox: true });
});

test("on a closed inbox a take that lost its send reads what is still unread before it takes the close", () => {
  // 2 take a (one), 3 x -> a, 4 y unread, 5 close, 6 withdraw [3]: a takes y, then the close
  const w = fold(...START, takeWait("a", "one"), send("msgs", "x"), send("msgs", "y"), new InboxClosedRecord("msgs"));
  applyWaitsRecord(w, { seq: 6, record: new InboxWithdrawnRecord("msgs", [3]) });
  expect(w.calls.get(call("a"))).toMatchObject({ state: "answered", assigned: [4], closedInbox: false });
  expect(w.inboxes.get("msgs")!.unread).toEqual([]);
  expect(decideWait(w, call("a"), takeOne(), 0)).toEqual({ kind: "take", commit: { inbox: "msgs", events: [4], closed: true } });
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

/** Commit a send of `event` to `inbox`, made by `by`, and return its seq. */
function commitSendTo(run: Run, inbox: string, event: unknown, by: Author | null = null): number {
  const prepared = prepareSend(run.state, inbox, event, null, by) as PreparedTransition;
  commitPrepared(run.log, run.state, prepared);
  return prepared.expectedSequence;
}

/** Commit a send of `event` to msgs, made by `by`, and return its seq. */
const commitSend = (run: Run, event: unknown, by: Author | null = null) => commitSendTo(run, "msgs", event, by);

/** A second START inbox param, `other: Inbox(str)`. */
const OTHER = new Parameter("other", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.STRING))) });

const ADA: Author = { kind: "user", name: "Ada" };
const OWNER: Author = { kind: "agent", name: "yian-chat" };

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

test("a withdraw by a sender names only its unread sends; the others stay and the take gets the oldest", () => {
  const run = takeRun();
  const x = commitSend(run, "x", ADA);
  const y = commitSend(run, "y", OWNER);
  const z = commitSend(run, "z");
  const w = commitSend(run, "w", { ...ADA });
  // The same author as another object is the same sender; another field makes it another.
  commitSend(run, "v", { ...ADA, run: "chat" });
  const prepared = prepareWithdraw(run.state, "msgs", null, ADA)!;
  expect(prepared.record).toEqual(new InboxWithdrawnRecord("msgs", [x, w], { by: ADA }));
  commitPrepared(run.log, run.state, prepared);
  expect([x, y, z, w].map((seq) => run.state.inboxEvents.has(seq))).toEqual([false, true, true, false]);
  expect([x, y, z, w].map((seq) => run.state.sendBy.get(seq) ?? null)).toEqual([null, OWNER, null, null]);
  expect(run.state.waits.calls.get(callOf("heard"))).toMatchObject({ state: "answered", assigned: [y] });
  expect(prepareWithdraw(run.state, "msgs", null, ADA)).toBeNull();
  expect([...(prepareWithdraw(run.state, "msgs")!.record as InboxWithdrawnRecord).events]).toEqual([y, z, w + 1]);
});

test("replay refuses a withdraw by a sender that names another's send or leaves one of its own; chosen sends may be some of its own", () => {
  const run = takeRun();
  const x = commitSend(run, "x", ADA);
  const y = commitSend(run, "y", OWNER);
  const w = commitSend(run, "w", ADA);
  for (const events of [[x, y, w], [x]]) {
    expect(() => prepareRecord(run.state, new InboxWithdrawnRecord("msgs", events, { by: ADA }))).toThrow(
      `InboxWithdrawn must name the whole unread set [${x}, ${w}] its by sent on inbox 'msgs', found [${events.join(", ")}]`,
    );
  }
  expect(() => prepareRecord(run.state, new InboxWithdrawnRecord("msgs", [x, y], { by: ADA, chosen: true }))).toThrow(
    `InboxWithdrawn may name only the unread sends [${x}, ${w}] its by sent on inbox 'msgs', found [${x}, ${y}]`,
  );
  commitPrepared(run.log, run.state, prepareRecord(run.state, new InboxWithdrawnRecord("msgs", [w], { by: ADA, chosen: true })));
  expect(withdrawableEvents(run.state.waits, "msgs")).toEqual([x, y]);
});

test("a send publishes InboxSent with its inbox, seq and by; a keyed repeat publishes nothing", () => {
  const run = takeRun();
  const prepared = prepareSend(run.state, "msgs", "x", "k", ADA) as PreparedTransition;
  const sent = new InboxSent({ inbox: "msgs", seq: prepared.expectedSequence, by: ADA });
  expect(prepared.observers).toEqual([sent]);
  expect(commitPrepared(run.log, run.state, prepared)).toEqual([sent]);
  expect(prepareSend(run.state, "msgs", "x", "k", ADA)).toEqual({ repeatOf: prepared.expectedSequence });
  const bare = prepareSend(run.state, "msgs", "y");
  expect(bare.observers).toEqual([new InboxSent({ inbox: "msgs", seq: bare.expectedSequence, by: null })]);
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

test("prepareWithdraw of chosen sends names those still unread, in seq order; the rest stay and the take gets the oldest", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  const y = commitSend(run, "y");
  const z = commitSend(run, "z");
  // x is the waiting take's; y and z are queued.
  const prepared = prepareWithdraw(run.state, "msgs", [z, x])!;
  expect(prepared.record).toEqual(new InboxWithdrawnRecord("msgs", [x, z], { chosen: true }));
  commitPrepared(run.log, run.state, prepared);
  expect([x, y, z].map((seq) => run.state.inboxEvents.has(seq))).toEqual([false, true, false]);
  expect(run.state.waits.calls.get(callOf("heard"))).toMatchObject({ state: "answered", assigned: [y] });
  // Replayed from the log, the record leaves the same waits.
  expect(replayed(run.state.flow, run.path).waits.toJSON()).toEqual(run.state.waits.toJSON());
});

test("prepareWithdraw of chosen sends takes only the unread ones, and is null when none is", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  commitPrepared(run.log, run.state, prepareNodeWoken(run.state, "heard"));
  // x is the woken take's now: read.
  const y = commitSend(run, "y");
  expect(prepareWithdraw(run.state, "msgs", [x])).toBeNull();
  expect(prepareWithdraw(run.state, "msgs", [x, y])!.record).toEqual(new InboxWithdrawnRecord("msgs", [y], { chosen: true }));
});

test("prepareWithdraw of chosen sends refuses unknown_send: no send, another inbox's, or with by another sender's", () => {
  const run = started(paramFlow([MSGS, OTHER], tool("heard", "take")));
  const x = commitSend(run, "x", ADA);
  const y = commitSend(run, "y", OWNER);
  const o = commitSendTo(run, "other", "o", ADA);
  const refused = (events: number[], by: Author | null = null) => {
    try {
      prepareWithdraw(run.state, "msgs", events, by);
    } catch (error) {
      expect(error).toBeInstanceOf(DeliveryRefused);
      return [(error as DeliveryRefused).reason, (error as DeliveryRefused).message];
    }
    throw new Error("expected a DeliveryRefused");
  };
  expect(refused([x, o + 1])).toEqual(["unknown_send", `the run has no send ${o + 1} to inbox 'msgs'`]);
  expect(refused([1])).toEqual(["unknown_send", "the run has no send 1 to inbox 'msgs'"]);
  expect(refused([o])).toEqual(["unknown_send", `the run has no send ${o} to inbox 'msgs'`]);
  expect(refused([x, y], ADA)).toEqual(["unknown_send", `send ${y} to inbox 'msgs' was made by another sender`]);
  // Unread or not: a read send of another sender is refused too.
  commitPrepared(run.log, run.state, prepareWithdraw(run.state, "msgs", [y])!);
  expect(refused([y], ADA)).toEqual(["unknown_send", `send ${y} to inbox 'msgs' was made by another sender`]);
  expect(prepareWithdraw(run.state, "msgs", [x], { ...ADA })!.record).toEqual(new InboxWithdrawnRecord("msgs", [x], { by: ADA, chosen: true }));
});

test("the run keeps which sends a withdraw took back, through a replay", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  const y = commitSend(run, "y");
  const z = commitSend(run, "z");
  commitPrepared(run.log, run.state, prepareWithdraw(run.state, "msgs", [z])!);
  commitPrepared(run.log, run.state, prepareWithdraw(run.state, "msgs", [x])!);
  expect([x, y, z].map((seq) => wasWithdrawn(run.state, seq))).toEqual([true, false, true]);
  const back = replayed(run.state.flow, run.path);
  expect([x, y, z].map((seq) => wasWithdrawn(back, seq))).toEqual([true, false, true]);
});

test("prepareWithdraw's chosen sends: a TypeError for a list of other than integers, a RangeError for none, a seq below 1 or one named twice", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  for (const events of ["x", [1.5], [String(x)], [BigInt(x)], [x, , x + 1], new Array(1)]) {
    expect(() => prepareWithdraw(run.state, "msgs", events as never)).toThrow("a withdraw's events must be a list of send seqs");
  }
  expect(() => prepareWithdraw(run.state, "msgs", [])).toThrow("a withdraw's events must name at least one send");
  expect(() => prepareWithdraw(run.state, "msgs", [0])).toThrow("a withdraw's events must be positive seqs, got 0");
  expect(() => prepareWithdraw(run.state, "msgs", [x, 3, x])).toThrow(`a withdraw's events name send ${x} twice`);
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
])("replay refuses an inbox_withdrawn of every unread send naming %s", (_, events) => {
  const run = takeRun();
  const x = commitSend(run, "x");
  const y = commitSend(run, "y");
  expect(() => prepareRecord(run.state, new InboxWithdrawnRecord("msgs", events(x, y)))).toThrow(
    `InboxWithdrawn must name the whole unread set [${x}, ${y}] of inbox 'msgs', found [${events(x, y).join(", ")}]`,
  );
});

test("replay takes an inbox_withdrawn of chosen sends naming some unread sends, and refuses one naming a seq that is no unread send", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  const y = commitSend(run, "y");
  const chosen = (...events: number[]) => new InboxWithdrawnRecord("msgs", events, { chosen: true });
  expect(() => prepareRecord(run.state, chosen(x, y, y + 1))).toThrow(
    `InboxWithdrawn may name only the unread sends [${x}, ${y}] of inbox 'msgs', found [${x}, ${y}, ${y + 1}]`,
  );
  // A record names each send once, and at least one.
  expect(() => chosen(x, x)).toThrow("events must be strictly increasing");
  expect(() => chosen()).toThrow("events must be non-empty");
  commitPrepared(run.log, run.state, prepareRecord(run.state, chosen(y)));
  expect(withdrawableEvents(run.state.waits, "msgs")).toEqual([x]);
  // y is gone: naming it again is refused.
  expect(() => prepareRecord(run.state, chosen(y))).toThrow(`InboxWithdrawn may name only the unread sends [${x}] of inbox 'msgs', found [${y}]`);
});

test("replay refuses an inbox_withdrawn naming a seq a woken take holds, another inbox's send, or an undeclared inbox", () => {
  const run = started(paramFlow([MSGS, OTHER], tool("heard", "take")));
  commitPrepared(run.log, run.state, prepareNodeStarted(run.state, "heard"));
  commitPrepared(run.log, run.state, prepareNodeWaiting(run.state, "heard", { callIds: [callOf("heard")], waits: [takeEntry("heard")] }));
  const x = commitSend(run, "x");
  const o = commitSendTo(run, "other", "o");
  commitPrepared(run.log, run.state, prepareNodeWoken(run.state, "heard"));
  for (const seq of [x, o]) {
    expect(() => prepareRecord(run.state, new InboxWithdrawnRecord("msgs", [seq], { chosen: true }))).toThrow(
      `InboxWithdrawn may name only the unread sends [] of inbox 'msgs', found [${seq}]`,
    );
    expect(() => prepareRecord(run.state, new InboxWithdrawnRecord("msgs", [seq]))).toThrow(
      `InboxWithdrawn must name the whole unread set [] of inbox 'msgs', found [${seq}]`,
    );
  }
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

test("the withdraw delivery op prepares the withdraw, of chosen sends with events", () => {
  const run = takeRun();
  const x = commitSend(run, "x");
  const y = commitSend(run, "y", ADA);
  const chosen = prepareDelivery(run.state, { kind: "withdraw", inbox: "msgs", events: [y], by: ADA }) as PreparedTransition;
  expect(chosen.record).toEqual(new InboxWithdrawnRecord("msgs", [y], { by: ADA, chosen: true }));
  const prepared = prepareDelivery(run.state, { kind: "withdraw", inbox: "msgs" }) as PreparedTransition;
  expect(prepared.record).toEqual(new InboxWithdrawnRecord("msgs", [x, y]));
  expect(() => prepareDelivery(run.state, { kind: "nope" } as never)).toThrow(/send, withdraw, complete, release, close, interrupt or cancel/);
});
