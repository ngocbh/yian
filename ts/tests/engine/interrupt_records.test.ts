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
import { applyWaitsRecord, isWakeable, nodeCalls, WaitsState } from "../../src/runtime/waits.ts";

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
