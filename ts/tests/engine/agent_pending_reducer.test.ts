/**
 * TS-only: the run reducer's rules for an AGENT's pending host calls: registration, results,
 * closers, complete and pendingCalls.
 */

import { describe, expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/memory.ts";
import { RunResumedRecord, type RunLog } from "../../src/runtime/log.ts";
import {
  callResultStep,
  commitPrepared,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareRunResumed,
  type RunState,
} from "../../src/runtime/state.ts";
import { RUN_ID } from "./run_state_fixtures.ts";
import { paramFlow, started } from "./waits_fixtures.ts";

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
    refuse("a", { ...pendingOf("a", 1, 0, "c1"), deadline: "2026-10-21T00:00:00.000Z" });
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
