/**
 * TS-only: the `interrupted` record and step in the waits fold and the reducer.
 */
import { expect, test } from "bun:test";

import {
  canonicalTyped,
  InterruptedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  NodeWaitingRecord,
  PoolCommitRecord,
  type RunRecord,
} from "../../src/runtime/log.ts";
import { applyWaitsRecord, isWakeable, nodeCalls, wakeableNodes, WaitsState } from "../../src/runtime/waits.ts";
import type { CallClosed } from "../../src/events.ts";
import { TOOL_INTERRUPTED_OUTCOME } from "../../src/nodes/agent/memory.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { Binding, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { RunLog, RunResumedRecord } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  DeliveryRefused,
  prepareComplete,
  prepareInboxClosed,
  prepareInterrupt,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareRecord,
  prepareRunCancelled,
  prepareRunResumed,
  type RunState,
} from "../../src/runtime/state.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { RUN_ID } from "./run_state_fixtures.ts";
import { MSGS, paramFlow, started, tool } from "./waits_fixtures.ts";

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

test("the honor step drops the pending interrupt", () => {
  const w = fold(...START, modelCall("reply", 1), new InterruptedRecord("owner", ["reply"]), honor("reply", 2, 3));
  expect(w.interrupts.has("reply")).toBe(false);
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

/** START(msgs, other) -> a (reads msgs), w (reads msgs), w2 (reads other), t (an ask TOOL) -> END. */
function readersFlow() {
  const nodes = [reader("a"), reader("w"), reader("w2", "other")];
  const wiring = new Map(nodes.flatMap((each) => (each.wiring === null ? [] : [each.wiring])));
  return started(paramFlow([MSGS, OTHER], { wiring }, ...nodes.map((each) => each.node), tool("t", "ask")));
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

test("no reader: prepareInterrupt returns null", () => {
  const { state } = readersFlow();
  expect(prepareInterrupt(state, "msgs", [])).toBeNull();
});

test("a node with a pending interrupt is left out; all left out is null", () => {
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
