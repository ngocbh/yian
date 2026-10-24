/**
 * TS-only: the run reducer's rules for an AGENT's pending host calls: registration, results,
 * closers, complete and pendingCalls.
 */

import { describe, expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { pendingCalls } from "../../src/run.ts";
import { TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/memory.ts";
import { NodeStepRecord, RunResumedRecord, type RunLog } from "../../src/runtime/log.ts";
import {
  callResultStep,
  commitPrepared,
  DeliveryRefused,
  prepareComplete,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareRunResumed,
  type RunState,
} from "../../src/runtime/state.ts";
import { RUN_ID } from "./run_state_fixtures.ts";
import { call, paramFlow, started, tool } from "./waits_fixtures.ts";

const agent = (id: string) => new AgentNode(id, { prompt: "go", tools: ["ask_user", "lookup"] });

/** A recorded chat reply calling `calls` (`[id, name, args]`). */
const reply = (...calls: [string, string, Record<string, unknown>][]) => ({
  kind: "model_call",
  purpose: "chat",
  model: "m",
  request_sha256: "0".repeat(64),
  started_at: "2026-10-20T14:02:11.123Z",
  first_token_at: null,
  output: {
    format: "yian.message.v1",
    role: "assistant",
    parts: calls.map(([id, name, args]) => ({ type: "tool_call", id, name, arguments: args })),
  },
  response_id: null,
  finish: "tool_calls",
  usage: null,
});

/** The yian call id of call `index` of `node`'s reply `n`. */
const agentCall = (node: string, n: number, index: number) => `${RUN_ID}/${node}/${n}/${index}`;

/** The `pending` step of call `index` of `node`'s reply `n`, whose provider id is `toolCallId`. */
const pendingOf = (node: string, n: number, index: number, toolCallId: string, tool = "ask_user") => ({
  kind: "pending",
  call_id: agentCall(node, n, index),
  tool,
  tool_call_id: toolCallId,
  reply: BigInt(n),
  index: BigInt(index),
});

/** An agent's `tool_result` step for provider call `toolCallId`. */
const result = (toolCallId: string, output = "yes", extra: Record<string, unknown> = {}) => ({
  kind: "tool_result",
  call_id: toolCallId,
  started_at: null,
  output,
  is_error: false,
  ...extra,
});

function start(log: RunLog, state: RunState, node: string): void {
  commitPrepared(log, state, prepareNodeStarted(state, node));
}

/** Commit `body` as `node`'s next step; return its seq. */
function step(log: RunLog, state: RunState, node: string, body: Record<string, unknown>): number {
  const prepared = prepareNodeStep(state, node, body);
  commitPrepared(log, state, prepared);
  return prepared.expectedSequence;
}

describe("an agent's pending step", () => {
  test("registers an open call keyed by its provider id; several per reply", () => {
    const { log, state } = started(paramFlow([], agent("a")));
    start(log, state, "a");
    step(log, state, "a", reply(["c1", "ask_user", { question: "q" }], ["c2", "lookup", { x: 1 }]));
    const seq = step(log, state, "a", pendingOf("a", 1, 0, "c1"));
    step(log, state, "a", pendingOf("a", 1, 1, "c2", "lookup"));
    const registered = state.waits.calls.get(agentCall("a", 1, 0))!;
    expect([registered.nodeId, registered.tool, registered.resultKey, registered.state, registered.pendingSeq]).toEqual([
      "a",
      "ask_user",
      "c1",
      "open",
      seq,
    ]);
    commitPrepared(log, state, prepareNodeWaiting(state, "a", { callIds: [agentCall("a", 1, 0), agentCall("a", 1, 1)] }));
    expect(state.waits.waiting.get("a")!.callIds).toEqual([agentCall("a", 1, 0), agentCall("a", 1, 1)]);
  });

  test("an ask_user's pending step may hold its deadline", () => {
    const { log, state } = started(paramFlow([], agent("a")));
    start(log, state, "a");
    step(log, state, "a", reply(["c1", "ask_user", { question: "q", timeout: "1m" }]));
    step(log, state, "a", { ...pendingOf("a", 1, 0, "c1"), deadline: "2026-10-21T00:00:00.000Z" });
    expect(state.waits.calls.get(agentCall("a", 1, 0))!.deadline).toBe("2026-10-21T00:00:00.000Z");
  });

  test("agent pending refusals", () => {
    const { log, state } = started(paramFlow([], agent("a"), agent("b")));
    start(log, state, "a");
    start(log, state, "b");
    const refuse = (node: string, body: Record<string, unknown>) =>
      expect(() => prepareNodeStep(state, node, body)).toThrow(RangeError);
    refuse("a", pendingOf("a", 1, 0, "c1")); // no reply yet
    step(log, state, "a", reply(["c1", "ask_user", { question: "q" }]));
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), call_id: agentCall("a", 1, 1) }); // id doesn't match index
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), call_id: agentCall("b", 1, 0) });
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), tool_call_id: "" });
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), index: -1n });
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), reply: 2n });
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), arguments: {} });
    // Only ask_user times out, and its deadline is a record time.
    refuse("a", { ...pendingOf("a", 1, 0, "c1", "lookup"), deadline: "2026-10-21T00:00:00.000Z" });
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), deadline: "tomorrow" });
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), tool: "take" });
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), tool: "" });
    step(log, state, "a", result("c0", "x")); // a step after the reply: still the latest reply
    step(log, state, "a", pendingOf("a", 1, 0, "c1"));
    refuse("a", pendingOf("a", 1, 0, "c1")); // registered once
    step(log, state, "a", result("c1"));
    step(log, state, "a", reply(["c9", "lookup", {}]));
    refuse("a", pendingOf("a", 1, 0, "c1")); // an earlier reply
  });
});

describe("an agent's tool_result", () => {
  test("answers its node's call of the latest reply; repeated provider ids don't collide", () => {
    const { log, state } = started(paramFlow([], agent("a"), agent("b")));
    start(log, state, "a");
    start(log, state, "b");
    step(log, state, "a", reply(["c0", "ask_user", { question: "q" }]));
    step(log, state, "a", pendingOf("a", 1, 0, "c0"));
    // Another agent's provider numbered its call c0 too: its own result, no collision.
    step(log, state, "b", reply(["c0", "lookup", {}]));
    step(log, state, "b", result("c0", "b's"));
    expect(state.waits.calls.get(agentCall("a", 1, 0))!.state).toBe("open");
    step(log, state, "a", result("c0"));
    expect(state.waits.calls.get(agentCall("a", 1, 0))!.state).toBe("answered");
    // The next reply reuses c0 for an inline call: the node's own detail, not a second answer.
    step(log, state, "a", reply(["c0", "lookup", {}]));
    step(log, state, "a", result("c0", "inline"));
    expect(callResultStep(state, agentCall("a", 1, 0))!["output"]).toBe("yes");
  });

  test("after a resume closes an agent's call, the agent writes its one closer", () => {
    const { log, state } = started(paramFlow([], agent("a")));
    start(log, state, "a");
    step(log, state, "a", reply(["c1", "ask_user", { question: "q" }]));
    step(log, state, "a", pendingOf("a", 1, 0, "c1"));
    const resumed = prepareRunResumed(state, { numWorkers: 0 });
    expect(resumed.record).toEqual(new RunResumedRecord(["a"], 0, [agentCall("a", 1, 0)]));
    commitPrepared(log, state, resumed);
    start(log, state, "a");
    const closer = result("c1", TOOL_OUTCOME_UNKNOWN, { is_error: true, closed: "crashed" });
    expect(() => prepareNodeStep(state, "a", { ...closer, closed: "timed_out" })).toThrow(/closed \(crashed\)/);
    expect(() => prepareNodeStep(state, "a", result("c1"))).toThrow(/closed \(crashed\)/);
    const seq = step(log, state, "a", closer);
    expect(state.waits.calls.get(agentCall("a", 1, 0))!.resultSeq).toBe(seq);
    expect(state.waits.calls.get(agentCall("a", 1, 0))!.state).toBe("closed");
    expect(() => prepareNodeStep(state, "a", closer)).toThrow(/closer/);
  });

  test("a closed result for an open call is refused", () => {
    const { log, state } = started(paramFlow([], agent("a")));
    start(log, state, "a");
    step(log, state, "a", reply(["c1", "ask_user", { question: "q" }]));
    step(log, state, "a", pendingOf("a", 1, 0, "c1"));
    expect(() => prepareNodeStep(state, "a", result("c1", "x", { closed: "crashed" }))).toThrow(/open/);
  });
});

describe("complete on an agent's call", () => {
  const waitingAgent = () => {
    const { log, state, path } = started(paramFlow([], agent("a")));
    start(log, state, "a");
    step(log, state, "a", reply(["c1", "ask_user", { question: "q" }]));
    step(log, state, "a", pendingOf("a", 1, 0, "c1"));
    return { log, state, path, id: agentCall("a", 1, 0) };
  };

  test("a call's view names the key its answer is stored under", () => {
    const { log, state } = started(paramFlow([], agent("a"), tool("t", "ask")));
    start(log, state, "a");
    start(log, state, "t");
    step(log, state, "a", reply(["c1", "ask_user", { question: "q" }]));
    step(log, state, "a", pendingOf("a", 1, 0, "c1"));
    step(log, state, "t", { kind: "pending", call_id: call("t"), tool: "ask", arguments: {} });
    // complete builds the agent call's view; its key is the provider id.
    expect((prepareComplete(state, agentCall("a", 1, 0), { output: "yes" })!.record as NodeStepRecord).step["call_id"]).toBe("c1");
    expect(state.waits.calls.get(agentCall("a", 1, 0))!.resultKey).toBe("c1");
    expect(state.waits.calls.get(call("t"))!.resultKey).toBe(call("t"));
  });

  test("an output becomes the text the model reads", () => {
    const { state, id } = waitingAgent();
    expect(prepareComplete(state, id, { output: { n: 1 } })!.record).toEqual(
      new NodeStepRecord("a", 3, { kind: "tool_result", call_id: "c1", started_at: null, output: '{"n":1}', is_error: false }),
    );
  });

  test("an error is a reported error result", () => {
    const { state, id } = waitingAgent();
    expect((prepareComplete(state, id, { error: "no" })!.record as NodeStepRecord).step).toEqual({
      kind: "tool_result",
      call_id: "c1",
      started_at: null,
      output: "no",
      is_error: true,
    });
  });

  test("an output that isn't a yian value is refused; the call stays open", () => {
    const { state, id } = waitingAgent();
    expect(() => prepareComplete(state, id, { output: () => 1 })).toThrow(DeliveryRefused);
    expect(state.waits.calls.get(id)!.state).toBe("open");
  });

  test("a repeat of the same answer is a no-op; another answer is refused", () => {
    const { log, state, id } = waitingAgent();
    commitPrepared(log, state, prepareComplete(state, id, { output: "yes" })!);
    expect(prepareComplete(state, id, { output: "yes" })).toBeNull();
    expect(() => prepareComplete(state, id, { output: "no" })).toThrow(/already answered/);
  });
});

test("pendingCalls reads an agent call's arguments from its reply", () => {
  const { log, state, path } = started(paramFlow([], agent("a")));
  start(log, state, "a");
  step(log, state, "a", reply(["c1", "ask_user", { question: "Ship?" }], ["c2", "lookup", { x: 1n }]));
  const seq = step(log, state, "a", pendingOf("a", 1, 1, "c2", "lookup"));
  log.close();
  expect(pendingCalls(path)).toEqual([
    { callId: agentCall("a", 1, 1), node: "a", tool: "lookup", arguments: { x: 1n }, pendingSeq: seq, deadline: null },
  ]);
});
