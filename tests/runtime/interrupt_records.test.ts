/** The `interrupted` record and step in the waits fold and the reducer. */
import { describe, expect, test } from "bun:test";

import { Failure } from "../../src/diagnostics.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import {
  canonicalTyped,
  InboxClosedRecord,
  InboxWithdrawnRecord,
  InterruptedRecord,
  NodeFailedRecord,
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
import { applyWaitsRecord, awaitsNextMessage, isWakeable, nodeCalls, wakeableNodes, WaitsState } from "../../src/runtime/waits.ts";
import type { CallClosed } from "../../src/events.ts";
import { prepareDelivery } from "../../src/runtime/engine.ts";
import { TOOL_INTERRUPTED_OUTCOME } from "../../src/nodes/agent/memory.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { Binding, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { MemoryRunLog, type RunLog } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  DeliveryRefused,
  prepareComplete,
  prepareInboxClosed,
  prepareAppliedStop,
  prepareInterrupt,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareNodeWoken,
  prepareRecord,
  prepareRunCancelled,
  prepareRunResumed,
  prepareRunStarted,
  prepareSend,
  RunState,
} from "../../src/runtime/state.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { FIXED_CLOCK, RUN_ID, startData } from "./run_state_fixtures.ts";
import { call, MSGS, paramFlow, started, takeEntry, tool } from "./waits_fixtures.ts";

const commit = (node: string, key: string, value: unknown) => new PoolCommitRecord(node, key, canonicalTyped(value, null)[0]);
/** The run start's commits: a system value, then the START inputs (seqs 0 and 1). */
const START = [commit("__start__", "system.run_id", "r"), commit("__start__", "__start__", {})];
const modelCall = (node: string, n: number) => new NodeStepRecord(node, n, { kind: "model_call", purpose: "chat" });
/** An agent's `pending` step for call `index` of reply `reply`, provider id `toolCallId`. */
const agentPending = (node: string, n: number, reply: number, index: number, toolCallId: string) =>
  new NodeStepRecord(node, n, {
    kind: "pending",
    call_id: `r/${node}/${reply}/${index}`,
    tool: "ask_user",
    tool_call_id: toolCallId,
    reply: BigInt(reply),
    index: BigInt(index),
  });
const closer = (node: string, n: number, toolCallId: string, closed: string) =>
  new NodeStepRecord(node, n, { kind: "tool_result", call_id: toolCallId, output: "x", is_error: true, closed });
const honor = (node: string, n: number, by: number) => new NodeStepRecord(node, n, { kind: "interrupted", by: BigInt(by) });
const send = (inbox: string, value: unknown) => commit("__start__", `__start__.${inbox}`, value);
const takeWait = (node: string, mode: "one" | "all", inbox = "msgs") =>
  new NodeWaitingRecord(node, [`r/${node}`], { waits: [{ callId: `r/${node}`, tool: "take", inbox, mode, deadline: null }] });
const takeCommit = (node: string, events: number[], closed = false) =>
  new PoolCommitRecord(node, node, null, { take: { inbox: "msgs", events, closed } });
const stop = (inbox: string, nodes: string[] = []) => new InterruptedRecord(inbox, nodes);

/** Fold `records` at seqs 0, 1, ... into a fresh state. */
function fold(...records: RunRecord[]): WaitsState {
  const w = new WaitsState();
  records.forEach((record, seq) => applyWaitsRecord(w, { seq, record }));
  return w;
}

test("the record sets the pending interrupt and closes the node's open calls interrupted", () => {
  const w = fold(
    ...START,
    modelCall("reply", 1),
    agentPending("reply", 2, 1, 0, "tc-0"),
    new NodeWaitingRecord("reply", ["r/reply/1/0"], { steerInbox: "owner" }),
    new InterruptedRecord("owner", ["reply"]),
  );
  expect(w.interrupts.get("reply")).toBe(5);
  expect(w.calls.get("r/reply/1/0")).toMatchObject({ state: "closed", closedReason: "interrupted" });
  expect(isWakeable(w, "reply")).toBe(true);
});

test("a pending step on a node with a pending interrupt registers its call closed interrupted", () => {
  const w = fold(...START, modelCall("reply", 1), new InterruptedRecord("owner", ["reply"]), agentPending("reply", 2, 1, 0, "tc-0"));
  expect(w.calls.get("r/reply/1/0")).toMatchObject({ state: "closed", closedReason: "interrupted" });
});

test("the honor step drops the pending interrupt and marks the node honored until it settles", () => {
  const w = fold(...START, modelCall("reply", 1), new InterruptedRecord("owner", ["reply"]), honor("reply", 2, 3));
  expect(w.interrupts.has("reply")).toBe(false);
  expect([...w._honored]).toEqual(["reply"]);
  const back = WaitsState.fromJSON(JSON.parse(JSON.stringify(w.toJSON())));
  expect([...back._honored]).toEqual(["reply"]);
  const settled = fold(...START, modelCall("reply", 1), new InterruptedRecord("owner", ["reply"]), honor("reply", 2, 3), new NodeSucceededRecord("reply", 1));
  expect(settled._honored.size).toBe(0);
});

test("the node's outcome drops a pending interrupt it never honored", () => {
  const w = fold(...START, modelCall("reply", 1), new InterruptedRecord("owner", ["reply"]), new NodeSucceededRecord("reply", 1));
  expect(w.interrupts.size).toBe(0);
  expect(w._replies.size).toBe(0);
});

test("an interrupted closer attaches to its call of the latest reply", () => {
  const w = fold(
    ...START,
    modelCall("reply", 1),
    agentPending("reply", 2, 1, 0, "tc-0"),
    new InterruptedRecord("owner", ["reply"]),
    closer("reply", 3, "tc-0", "interrupted"),
  );
  expect(w.calls.get("r/reply/1/0")!.resultSeq).toBe(5);
});

test("a closer never attaches to an earlier reply's call that shares its provider id", () => {
  // Reply 1's call closed with no closer (no valid agent leaves one so); reply 3 reuses its id
  // inline, and that call's closer must not answer reply 1's call.
  const w = fold(
    ...START,
    modelCall("reply", 1),
    agentPending("reply", 2, 1, 0, "tc-0"),
    new InterruptedRecord("owner", ["reply"]),
    modelCall("reply", 3),
    closer("reply", 4, "tc-0", "interrupted"),
  );
  expect(w.calls.get("r/reply/1/0")!.resultSeq).toBeNull();
});

test("toJSON/fromJSON keeps interrupts and the latest replies", () => {
  const w = fold(...START, modelCall("reply", 1), agentPending("reply", 2, 1, 0, "tc-0"), new InterruptedRecord("owner", ["reply"]));
  const back = WaitsState.fromJSON(JSON.parse(JSON.stringify(w.toJSON())));
  expect(back.toJSON()).toEqual(w.toJSON());
  expect(back.interrupts.get("reply")).toBe(4);
  expect(back._replies.get("reply")).toBe(2);
  expect(nodeCalls(back, "reply")[0]!.closedReason).toBe("interrupted");
});

// --- The reducer ------------------------------------------------------------------------------ //

const OTHER = new Parameter("other", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.STRING))) });

/** An AGENT `id` granted ask_user and lookup, reading steer inbox `inbox` unless it is `null`. */
function reader(id: string, inbox: string | null = "msgs"): { node: AgentNode; wiring: [string, Map<string, Binding>] | null } {
  const node = new AgentNode(id, { prompt: "go", tools: ["ask_user", "lookup"], steerInbox: inbox !== null });
  return { node, wiring: inbox === null ? null : [id, new Map([[STEER_INBOX_KEY, new Binding(`\${input.${inbox}}`)]])] };
}

/**
 * START(msgs, other) -> a (reads msgs), w (reads msgs), w2 (reads other), t (an ask TOOL), t2 (a
 * take TOOL) -> END.
 */
function readersFlow() {
  const nodes = [reader("a"), reader("w"), reader("w2", "other")];
  const wiring = new Map(nodes.flatMap((each) => (each.wiring === null ? [] : [each.wiring])));
  return started(paramFlow([MSGS, OTHER], { wiring }, ...nodes.map((each) => each.node), tool("t", "ask"), tool("t2", "take")));
}

type Run = ReturnType<typeof readersFlow>;

/** Start ready nodes, head first, until `nodeId` has started. */
function startUpTo(run: Run, nodeId: string): void {
  while (!run.state.inFlight.has(nodeId)) commitPrepared(run.log, run.state, prepareNodeStarted(run.state, run.state.ready[0]!));
}

/** Start nodes up to `t2`, whose take then waits on msgs. */
function waitTake(run: Run): void {
  startUpTo(run, "t2");
  commitPrepared(run.log, run.state, prepareNodeWaiting(run.state, "t2", { callIds: [call("t2")], waits: [takeEntry("t2")] }));
}

/** Commit an interrupt of msgs with `running`, returning its record's nodes. */
function interruptNow(run: Run, running: readonly string[] | null): string[] {
  const prepared = prepareInterrupt(run.state, "msgs", running)!;
  commitPrepared(run.log, run.state, prepared);
  return [...(prepared.record as InterruptedRecord).nodes];
}

/** A recorded chat reply calling `calls` (`[id, name]`). */
const replyStep = (...calls: [string, string][]) => ({
  kind: "model_call",
  purpose: "chat",
  model: "m",
  request_sha256: "0".repeat(64),
  started_at: "2026-10-20T14:02:11.123Z",
  first_token_at: null,
  output: {
    format: "yian.message.v1",
    role: "assistant",
    parts: calls.map(([id, name]) => ({ type: "tool_call", id, name, arguments: {} })),
  },
  response_id: null,
  finish: "tool_calls",
  usage: null,
});
const agentCall = (node: string, n: number, index: number) => `${RUN_ID}/${node}/${n}/${index}`;
const pendingStep = (node: string, n: number, index: number, toolCallId: string, extra: Record<string, unknown> = {}) => ({
  kind: "pending",
  call_id: agentCall(node, n, index),
  tool: "ask_user",
  tool_call_id: toolCallId,
  reply: BigInt(n),
  index: BigInt(index),
  ...extra,
});
const closerStep = (toolCallId: string, closed: string) => ({
  kind: "tool_result",
  call_id: toolCallId,
  started_at: null,
  output: TOOL_INTERRUPTED_OUTCOME,
  is_error: true,
  closed,
});

function commitStep(log: RunLog, state: RunState, node: string, body: Record<string, unknown>): number {
  const prepared = prepareNodeStep(state, node, body);
  commitPrepared(log, state, prepared);
  return prepared.expectedSequence;
}

/** `a` runs with an open call `c1`; `w` waits on its call `c1` with steer inbox msgs; `w2` waits on other. */
function interruptible() {
  const run = readersFlow();
  const { log, state } = run;
  for (const node of ["a", "w", "w2", "t"]) commitPrepared(log, state, prepareNodeStarted(state, node));
  for (const node of ["a", "w", "w2"]) {
    commitStep(log, state, node, replyStep(["c1", "ask_user"]));
    commitStep(log, state, node, pendingStep(node, 1, 0, "c1"));
  }
  commitPrepared(log, state, prepareNodeWaiting(state, "w", { callIds: [agentCall("w", 1, 0)], steerInbox: "msgs" }));
  commitPrepared(log, state, prepareNodeWaiting(state, "w2", { callIds: [agentCall("w2", 1, 0)], steerInbox: "other" }));
  return run;
}

test("prepareInterrupt names running readers first, then waiting readers, and closes their calls", () => {
  const { log, state } = interruptible();
  const prepared = prepareInterrupt(state, "msgs", ["a"])!;
  expect([...(prepared.record as InterruptedRecord).nodes]).toEqual(["a", "w"]);
  const events = commitPrepared(log, state, prepared);
  expect(events.map((event) => [event.constructor.name, (event as CallClosed).callId, (event as CallClosed).reason])).toEqual([
    ["CallClosed", agentCall("a", 1, 0), "interrupted"],
    ["CallClosed", agentCall("w", 1, 0), "interrupted"],
  ]);
  expect(state.waits.interrupts.get("w")).toBe(prepared.expectedSequence);
  expect(state.waits.calls.get(agentCall("w2", 1, 0))!.state).toBe("open");
  expect(wakeableNodes(state.waits)).toEqual(["w"]);
  // The record isn't a frontier record: the last one stays.
  expect(state.waits.lastFrontier).toBe("node_waiting");
});

test.each<[string, (run: Run) => void, readonly string[] | null, string[] | null]>([
  ["idle at the take: no record", waitTake, [], null],
  [
    "the take was answered, not woken: an empty record",
    (run) => {
      waitTake(run);
      commitPrepared(run.log, run.state, prepareSend(run.state, "msgs", "x"));
    },
    [],
    [],
  ],
  ["no take waits, nothing runs: an empty record", () => {}, [], []],
  ["a running reader: named", waitTake, ["a"], ["a"]],
  // Offline no engine says which in-flight node opened the inbox, so any counts as a reader.
  ["offline, crashed nodes in flight and the take waiting: an empty record", waitTake, null, []],
])("prepareInterrupt: %s", (_, setup, running, nodes) => {
  const run = readersFlow();
  setup(run);
  const prepared = prepareInterrupt(run.state, "msgs", running);
  if (nodes === null) expect(prepared).toBeNull();
  else expect([...(prepared!.record as InterruptedRecord).nodes]).toEqual(nodes);
});

test("already stopping: while a Stop stands, a repeated interrupt records nothing, even with a reader just opened", () => {
  const run = readersFlow();
  expect(interruptNow(run, [])).toEqual([]);
  expect(prepareInterrupt(run.state, "msgs", [])).toBeNull();
  startUpTo(run, "a");
  // That reader gets its interrupt from the applied Stop, not the host's.
  expect(prepareInterrupt(run.state, "msgs", ["a"])).toBeNull();
});

test("already stopping: a running reader that has honored an interrupt isn't named again", () => {
  const run = readersFlow();
  startUpTo(run, "a");
  expect(interruptNow(run, ["a"])).toEqual(["a"]);
  commitStep(run.log, run.state, "a", { kind: "interrupted", by: BigInt(run.state.waits.interrupts.get("a")!) });
  // t2's take waiting on msgs is a pickup: the Stop ends.
  waitTake(run);
  expect(run.state.waits.stops.has("msgs")).toBe(false);
  expect(prepareInterrupt(run.state, "msgs", ["a"])).toBeNull();
});

test("already stopping: a reader holding an interrupt is left out; the others are named", () => {
  const run = readersFlow();
  startUpTo(run, "a");
  interruptNow(run, ["a"]);
  waitTake(run);
  expect(interruptNow(run, ["a", "w"])).toEqual(["w"]);
});

test("prepareAppliedStop names the opener and each waiting reader without an interrupt, whatever stands", () => {
  const run = readersFlow();
  const { log, state } = run;
  interruptNow(run, []);
  startUpTo(run, "t");
  commitStep(log, state, "w", replyStep(["c1", "ask_user"]));
  commitStep(log, state, "w", pendingStep("w", 1, 0, "c1"));
  commitPrepared(log, state, prepareNodeWaiting(state, "w", { callIds: [agentCall("w", 1, 0)], steerInbox: "msgs" }));
  const prepared = prepareAppliedStop(state, "msgs", "a");
  expect([...(prepared.record as InterruptedRecord).nodes]).toEqual(["a", "w"]);
  expect(prepareRecord(state, prepared.record).record).toEqual(prepared.record);
  commitPrepared(log, state, prepared);
  expect(state.waits.stops.get("msgs")!.stopped).toBe(true);
  expect(state.waits.calls.get(agentCall("w", 1, 0))).toMatchObject({ state: "closed", closedReason: "interrupted" });
});

test("prepareAppliedStop refusals: unknown inbox, not started, ended", () => {
  const run = readersFlow();
  startUpTo(run, "a");
  expect(() => prepareAppliedStop(run.state, "nope", "a")).toThrow(/no inbox 'nope'/);
  const fresh = new RunState(run.state.flow);
  commitPrepared(new MemoryRunLog(RUN_ID, { clock: FIXED_CLOCK }), fresh, prepareRunStarted(fresh, startData()));
  expect(() => prepareAppliedStop(fresh, "msgs", "a")).toThrow(DeliveryRefused);
  commitPrepared(run.log, run.state, prepareRunCancelled(run.state));
  expect(() => prepareAppliedStop(run.state, "msgs", "a")).toThrow(DeliveryRefused);
});

test("replay accepts an interrupted record naming no node unless a waiting reader lacks one", () => {
  const { state } = readersFlow();
  expect([...(prepareRecord(state, new InterruptedRecord("msgs", [])).record as InterruptedRecord).nodes]).toEqual([]);
  const busy = interruptible();
  expect(() => prepareRecord(busy.state, new InterruptedRecord("msgs", []))).toThrow(/must name waiting node 'w'/);
});

test("a node with a pending interrupt is left out; all left out is null (already stopping)", () => {
  const { log, state } = interruptible();
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  expect(prepareInterrupt(state, "msgs", ["a"])).toBeNull();
});

test("an interrupt on a closed inbox is accepted", () => {
  const { log, state } = interruptible();
  commitPrepared(log, state, prepareInboxClosed(state, "msgs")!);
  expect([...(prepareInterrupt(state, "msgs", [])!.record as InterruptedRecord).nodes]).toEqual(["w"]);
});

test("prepareInterrupt refusals: unknown inbox, ended run", () => {
  const { log, state } = interruptible();
  expect(() => prepareInterrupt(state, "nope", [])).toThrow(DeliveryRefused);
  expect(() => prepareInterrupt(state, "nope", [])).toThrow(/no inbox 'nope'/);
  commitPrepared(log, state, prepareRunCancelled(state));
  let refused: unknown;
  try {
    prepareInterrupt(state, "msgs", []);
  } catch (error) {
    refused = error;
  }
  expect((refused as DeliveryRefused).reason).toBe("ended");
});

test.each([
  ["a node not in flight or waiting", ["ghost"], /does not exist/],
  ["a running node that reads no steer inbox", ["t", "w"], /reads no steer inbox/],
  ["a waiting node reading another inbox", ["w2", "w"], /waits without reading steer inbox/],
  ["a waiting reader left out", ["a"], /must name waiting node 'w'/],
])("replay refuses an interrupted record naming %s", (_, nodes, message) => {
  const { state } = interruptible();
  expect(() => prepareRecord(state, new InterruptedRecord("msgs", nodes))).toThrow(message);
});

test("replay refuses an interrupted record naming a node with a pending interrupt, or an undeclared inbox", () => {
  const { log, state } = interruptible();
  expect(() => prepareRecord(state, new InterruptedRecord("nope", ["a"]))).toThrow(/no inbox the START declares/);
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  expect(() => prepareRecord(state, new InterruptedRecord("msgs", ["a"]))).toThrow(/already holds a pending interrupt/);
});

test("complete after the record is call_settled (closed (interrupted))", () => {
  const { log, state } = interruptible();
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  let refused: unknown;
  try {
    prepareComplete(state, agentCall("w", 1, 0), { output: "late" });
  } catch (error) {
    refused = error;
  }
  expect(refused).toBeInstanceOf(DeliveryRefused);
  expect((refused as DeliveryRefused).reason).toBe("call_settled");
  expect((refused as Error).message).toContain("interrupted");
});

test("run_resumed.closed_calls leaves out calls closed interrupted", () => {
  const { log, state } = interruptible();
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  const resumed = prepareRunResumed(state, { numWorkers: 0 });
  expect((resumed.record as RunResumedRecord).closedCalls).toEqual([]);
});

test("the honor step names the node's pending interrupt, once, on a running node", () => {
  const { log, state } = interruptible();
  const seq = prepareInterrupt(state, "msgs", ["a"])!.expectedSequence;
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  const refuse = (node: string, body: Record<string, unknown>, message: RegExp) =>
    expect(() => prepareNodeStep(state, node, body)).toThrow(message);
  refuse("a", { kind: "interrupted", by: BigInt(seq + 1) }, /must name its pending interrupt/);
  refuse("a", { kind: "interrupted", by: seq }, /must name its pending interrupt/); // not an int
  refuse("a", { kind: "interrupted", by: BigInt(seq), extra: 1 }, /holds only kind and by/);
  refuse("w", { kind: "interrupted", by: BigInt(seq) }, /records only a tool_result/); // waiting
  refuse("t", { kind: "interrupted", by: BigInt(seq) }, /holds no pending interrupt/);
  commitStep(log, state, "a", { kind: "interrupted", by: BigInt(seq) });
  expect(state.waits.interrupts.has("a")).toBe(false);
  refuse("a", { kind: "interrupted", by: BigInt(seq) }, /holds no pending interrupt/);
});

test("an agent writes one interrupted closer for its call the record closed", () => {
  const { log, state } = interruptible();
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  expect(() => prepareNodeStep(state, "a", closerStep("c1", "crashed"))).toThrow(/closed \(interrupted\)/);
  const seq = commitStep(log, state, "a", closerStep("c1", "interrupted"));
  expect(state.waits.calls.get(agentCall("a", 1, 0))!.resultSeq).toBe(seq);
  expect(() => prepareNodeStep(state, "a", closerStep("c1", "interrupted"))).toThrow(/already has its closer/);
});

test("a pending step after the record registers its call closed and withdraws it from the host", () => {
  const { log, state } = interruptible();
  commitStep(log, state, "a", pendingStep("a", 1, 1, "c2"));
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  const prepared = prepareNodeStep(state, "a", pendingStep("a", 1, 2, "c3"));
  expect(prepared.observers.map((event) => [(event as CallClosed).callId, (event as CallClosed).reason])).toEqual([
    [agentCall("a", 1, 2), "interrupted"],
  ]);
  expect(commitPrepared(log, state, prepared)).toEqual(prepared.observers);
  expect(state.waits.calls.get(agentCall("a", 1, 2))).toMatchObject({ state: "closed", closedReason: "interrupted" });
});

test("a node_waiting may name a call an interrupt closed between the agent's Pending and it", () => {
  const { log, state } = interruptible();
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [agentCall("a", 1, 0)], steerInbox: "msgs" }));
  expect(wakeableNodes(state.waits)).toEqual(["w", "a"]);
});

test("without a pending interrupt a node_waiting naming a closed call is still refused", () => {
  const { log, state } = interruptible();
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  commitStep(log, state, "a", closerStep("c1", "interrupted"));
  commitStep(log, state, "a", { kind: "interrupted", by: BigInt(state.waits.interrupts.get("a")!) });
  expect(() => prepareNodeWaiting(state, "a", { callIds: [agentCall("a", 1, 0)] })).toThrow(/was closed \(interrupted\)/);
});

test("a timeout re-wait whose open call an interrupt closed in between is accepted and wakeable at once", () => {
  const run = readersFlow();
  const { log, state } = run;
  commitPrepared(log, state, prepareNodeStarted(state, "a"));
  commitStep(log, state, "a", replyStep(["ask", "ask_user"], ["hold", "ask_user"]));
  commitStep(log, state, "a", pendingStep("a", 1, 0, "ask", { deadline: "2026-10-21T00:00:00.000Z" }));
  commitStep(log, state, "a", pendingStep("a", 1, 1, "hold"));
  const [ask, hold] = [agentCall("a", 1, 0), agentCall("a", 1, 1)];
  commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [ask, hold], steerInbox: "msgs" }));
  commitPrepared(log, state, prepareNodeWoken(state, "a", [ask]));
  commitStep(log, state, "a", { ...closerStep("ask", "timed_out"), output: "timed out: no answer" });
  // The woken agent pends `hold` again; the interrupt lands before its node_waiting.
  commitPrepared(log, state, prepareInterrupt(state, "msgs", ["a"])!);
  commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [hold], steerInbox: "msgs" }));
  expect(wakeableNodes(state.waits)).toEqual(["a"]);
});

test("the interrupt delivery op prepares the interrupt; a missing running is none", () => {
  const { state } = interruptible();
  const viaOp = prepareDelivery(state, { kind: "interrupt", inbox: "msgs", running: ["a"] })!;
  expect([...(viaOp.record as InterruptedRecord).nodes]).toEqual(["a", "w"]);
  // Without `running` the delivery is offline: `a` in flight counts as an open reader for the idle
  // rule, and only the waiting reader is named.
  expect([...(prepareDelivery(state, { kind: "interrupt", inbox: "msgs" })!.record as InterruptedRecord).nodes]).toEqual(["w"]);
  expect(() => prepareDelivery(state, { kind: "nope" } as never)).toThrow(/send, withdraw, complete, close, interrupt or cancel/);
});

// --- The remembered Stop's window ------------------------------------------------------------- //

describe("stops", () => {
  test("an interrupted record remembers the Stop; stopped says whether it named an agent", () => {
    expect([...fold(...START, stop("msgs")).stops]).toEqual([["msgs", { seq: 2, stopped: false }]]);
    expect(fold(...START, new NodeStartedRecord("reply"), stop("msgs", ["reply"])).stops.get("msgs")).toEqual({ seq: 3, stopped: true });
    // A replayed empty record after a stopping one doesn't reset stopped (so it can't outlive the next wait).
    expect(fold(...START, new NodeStartedRecord("reply"), stop("msgs", ["reply"]), stop("msgs")).stops.get("msgs")).toEqual({
      seq: 4,
      stopped: true,
    });
  });

  test("a run_waiting keeps a Stop that stopped no one and clears one that did", () => {
    const w = fold(
      ...START,
      stop("msgs"),
      new RunWaitingRecord(["approve"]),
      new RunStoppedRecord("signal", "", ["approve"]),
      new RunResumedRecord([], 0),
      new RunWaitingRecord(["approve"]),
    );
    expect(w.stops.has("msgs")).toBe(true);
    applyWaitsRecord(w, { seq: 7, record: new NodeStartedRecord("reply") });
    applyWaitsRecord(w, { seq: 8, record: stop("msgs", ["reply"]) }); // the applied record
    applyWaitsRecord(w, { seq: 9, record: new RunWaitingRecord(["approve"]) });
    expect(w.stops.has("msgs")).toBe(false);
  });

  test("a Stop on one inbox isn't closed by a pickup from another", () => {
    const w = fold(...START, stop("msgs"), takeWait("heard", "one", "other"));
    expect(w.stops.has("msgs")).toBe(true);
  });

  test.each<[string, RunRecord[], RunRecord[]]>([
    ["a take of the inbox waits", [], [takeWait("heard", "one")]],
    ["a send answers a waiting take", [takeWait("heard", "one")], [send("msgs", "x")]],
    ["a timeout wake times the take out", [takeWait("heard", "one")], [new NodeWokenRecord("heard", ["r/heard"])]],
    ["the close answers a waiting take", [takeWait("heard", "one")], [new InboxClosedRecord("msgs")]],
    ["an inline take commits (taking nothing)", [new NodeStartedRecord("t")], [takeCommit("t", [], true)]],
    // 2 take, 3 x -> heard, 4 Stop, 5 withdraw
    ["a withdraw reopens a waited take", [takeWait("heard", "one"), send("msgs", "x")], [new InboxWithdrawnRecord("msgs", [3])]],
    [
      "a withdraw on a closed inbox answers it",
      [takeWait("heard", "one"), send("msgs", "x"), new InboxClosedRecord("msgs")],
      [new InboxWithdrawnRecord("msgs", [3])],
    ],
    // 2 x, 3 t starts, 4 t's orphan take of [2], 5 heard waits, 6 Stop, 7 resume re-offers x to heard
    [
      "run_resumed re-offers an orphan's event to a waiting take",
      [send("msgs", "x"), new NodeStartedRecord("t"), takeCommit("t", [2]), takeWait("heard", "one")],
      [new RunResumedRecord(["t"], 2)],
    ],
  ])("a pickup closes the window: %s", (_, before, after) => {
    const w = fold(...START, ...before, stop("msgs"), ...after);
    expect(w.stops.has("msgs")).toBe(false);
  });

  test.each<[string, RunRecord[], RunRecord[]]>([
    // 2 take, 3 x -> heard, 4 woken, 5 Stop, 6 heard's take commit
    ["a waited take's own commit", [takeWait("heard", "one"), send("msgs", "x"), new NodeWokenRecord("heard")], [takeCommit("heard", [3])]],
    ["a second send to an already answered take: all", [takeWait("heard", "all"), send("msgs", "x")], [send("msgs", "y")]],
    ["run_stopped and run_resumed", [], [new RunStoppedRecord("signal", "", []), new RunResumedRecord([], 0)]],
    ["a node_woken that times nothing out", [takeWait("heard", "one"), send("msgs", "x")], [new NodeWokenRecord("heard")]],
    ["a node_step", [new NodeStartedRecord("reply")], [modelCall("reply", 1)]],
    ["a run_waiting before it stopped anyone", [], [new RunWaitingRecord(["approve"])]],
  ])("not a closer: %s", (_, before, after) => {
    const w = fold(...START, ...before, stop("msgs"), ...after);
    expect(w.stops.has("msgs")).toBe(true);
  });

  const FAILURE = new Failure({ code: "node.execute", message: "boom", nodeId: "work" });
  test.each<[string, RunRecord]>([
    ["run_cancelled", new RunCancelledRecord()],
    ["run_failed", new RunFailedRecord(FAILURE)],
    ["node_failed", new NodeFailedRecord(FAILURE)],
    ["END's success", new NodeSucceededRecord(EndNode.ID, 1)],
  ])("%s clears every Stop", (_, terminal) => {
    const w = fold(...START, stop("msgs"), stop("other"), terminal);
    expect(w.stops.size).toBe(0);
  });

  test("another node's success keeps the Stop", () => {
    const w = fold(...START, new NodeStartedRecord("work"), stop("msgs"), new NodeSucceededRecord("work", 1));
    expect(w.stops.has("msgs")).toBe(true);
  });

  test("toJSON/fromJSON keeps stops; a bad stop entry is refused", () => {
    const w = fold(...START, stop("msgs"));
    expect([...WaitsState.fromJSON(JSON.parse(JSON.stringify(w.toJSON()))).stops]).toEqual([...w.stops]);
    expect(() => WaitsState.fromJSON({ ...w.toJSON(), stops: [["msgs", { seq: 1 }]] } as never)).toThrow(/stops 'msgs' must be a stop/);
    expect(() => WaitsState.fromJSON({ ...w.toJSON(), stops: [["msgs", { seq: "1", stopped: true }]] } as never)).toThrow(
      /stops 'msgs' must be a stop/,
    );
  });
});

test.each<[string, RunRecord[], boolean]>([
  ["a take waits with nothing assigned", [takeWait("heard", "one")], true],
  ["the take was answered and hasn't woken", [takeWait("heard", "one"), send("msgs", "x")], false],
  ["a send is unread and no take waits", [send("msgs", "x")], false],
  ["no take waits", [], false],
  [
    "a waiting reader reads the inbox",
    [takeWait("heard", "one"), new NodeWaitingRecord("reply", ["r/reply/1/0"], { steerInbox: "msgs" })],
    false,
  ],
  ["a take of another inbox waits", [takeWait("heard", "one", "other")], false],
  ["the take woke and hasn't committed", [takeWait("heard", "one"), new NodeWokenRecord("heard", ["r/heard"])], false],
])("awaitsNextMessage: %s", (_, records, idle) => expect(awaitsNextMessage(fold(...START, ...records), "msgs")).toBe(idle));

test("a Stop between a take's wake and its commit stands through the commit", () => {
  // 2 take waits, 3 x -> heard, 4 woken, 5 Stop, 6 the take's commit, 7 its success, 8 the reply starts
  const w = fold(
    ...START,
    takeWait("heard", "one"),
    send("msgs", "x"),
    new NodeWokenRecord("heard"),
    stop("msgs"),
    takeCommit("heard", [3]),
    new NodeSucceededRecord("heard", 6),
    new NodeStartedRecord("reply"),
  );
  expect(w.stops.get("msgs")).toEqual({ seq: 5, stopped: false });
});
