/**
 * AGENT `retries` and tool-loop defaults come from `defaults` when the node sets none: mirrors
 * `tests/engine/test_agent_defaults.py`.
 *
 * Ported: `test_the_manifest_holds_two_retries_by_default` and
 * `test_the_manifest_holds_the_agent_retries_default`.
 *
 * Ported at node level, through `evalNode` (`drive`) with a `ScriptedModel` from `models`, where
 * Python loads `_STRUCTURED` or `_LOOPING` and runs it (`runFlow`, T4-8):
 * `test_agent_retries_default_of_zero_fails_on_the_first_bad_reply`,
 * `test_authored_retries_win_over_the_default`,
 * `test_agent_max_tool_iterations_default_caps_the_loop` and
 * `test_authored_max_tool_iterations_wins_over_the_default` (the authored bound set as the node's
 * `env`, the `lookup` tool in the run's `ToolSet`).
 *
 * Changed: `YIAN_AGENT_RETRIES=0` and `YIAN_AGENT_MAX_TOOL_ITERATIONS=1` are set with `reload`,
 * which Python's `set_defaults` fixture does. The manifest halves read a hand-built one-agent flow
 * where Python loads `_STRUCTURED`. The exhausted bad reply fails with code
 * `agent.output_invalid` (Python: `node.execute`).
 */

import { afterEach, expect, test } from "bun:test";

import { flowManifest } from "../../src/compile/manifest.ts";
import { Flow } from "../../src/compile/model.ts";
import { forget, reload } from "../../src/defaults.ts";
import { Message, Response, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, serving, textReply } from "./run_state_fixtures.ts";

afterEach(() => {
  forget();
});

/** The manifest `retries` of a flow whose one node is a default AGENT `agent`. */
function manifestRetries(): unknown {
  const flow = new Flow(new Map([["agent", new AgentNode("agent", { prompt: "${question}" })]]), [], {
    wiring: new Map([["agent", new Map()]]),
    startId: "agent",
    endId: "agent",
  });
  return ((flowManifest(flow).nodes as JsonObject).agent as JsonObject).retries;
}

test("the manifest holds two retries by default", () => {
  expect(manifestRetries()).toBe(2);
  expect(new AgentNode("n").retries).toBe(2);
});

test("the manifest holds the agent retries default", () => {
  reload((name) => (name === "YIAN_AGENT_RETRIES" ? "0" : undefined));
  expect(manifestRetries()).toBe(0);
  expect(new AgentNode("n").retries).toBe(0);
  expect(new AgentNode("n", { retries: 3 }).retries).toBe(3);
});

// The sentinel model has no native structured output, so the reply is parsed from text and a bad
// reply costs one retry.
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };

/**
 * Evaluate an AGENT with output `{score: float}` and `retries` (the default when omitted) over a
 * model replying a bad score, then a good one; return the outcome and the model.
 */
async function badThenGood(retries?: number) {
  const replies = ['{"score": "bad"}', '{"score": 1.5}'].map(textReply);
  const model = new ScriptedModel(replies, PROMPTED);
  const node = new AgentNode("agent", { prompt: "q", llmConfig: PROMPTED, retries });
  node.outputContract = new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["score", Type.scalar(ValueKind.NUMBER)]]),
      required: new Set(["score"]),
    }),
  );
  const outcome = await drive(node, { models: serving(model) });
  return { outcome, model };
}

test("the agent retries default of zero fails on the first bad reply", async () => {
  reload((name) => (name === "YIAN_AGENT_RETRIES" ? "0" : undefined));
  const { outcome, model } = await badThenGood();
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  expect((outcome as EvaluationFailure).failure.code).toBe("agent.output_invalid");
  expect(model.requests).toHaveLength(1);
});

test("authored retries win over the default", async () => {
  reload((name) => (name === "YIAN_AGENT_RETRIES" ? "0" : undefined));
  const { outcome, model } = await badThenGood(1);
  expect(outcome).toEqual(new Output({ score: 1.5 }));
  expect(model.requests).toHaveLength(2);
});

/** A reply calling `lookup` with no arguments as call `id`. */
function lookupCall(id: string): Response {
  const call = new ToolCallPart({ id, name: "lookup", arguments: {} });
  return new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
}

const CONFIG = { provider: "openai", model: "test-model" };

/**
 * Evaluate an AGENT with `env` granted `lookup` (which answers `ok`) over `replies`; return the
 * outcome and the model.
 */
async function looping(replies: Response[], env: Record<string, unknown> = {}) {
  const model = new ScriptedModel(replies, CONFIG);
  const node = new AgentNode("agent", { prompt: "q", llmConfig: CONFIG, tools: ["lookup"] });
  node.env = env;
  const lookup = new FunctionTool({
    name: "lookup",
    description: "Look it up.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "ok",
  });
  const outcome = await drive(node, { models: serving(model), tools: new ToolSet({ lookup }) });
  return { outcome, model };
}

test("the agent max_tool_iterations default caps the loop", async () => {
  reload((name) => (name === "YIAN_AGENT_MAX_TOOL_ITERATIONS" ? "1" : undefined));
  const { outcome, model } = await looping([0, 1, 2, 3, 4].map((i) => lookupCall(String(i))));
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = outcome as EvaluationFailure;
  expect(failure.code).toBe("agent.max_tool_iterations");
  expect(failure.message).toContain("tool-iteration cap (1)");
  expect(model.requests).toHaveLength(1);
});

test("the authored max_tool_iterations wins over the default", async () => {
  reload((name) => (name === "YIAN_AGENT_MAX_TOOL_ITERATIONS" ? "1" : undefined));
  const replies = [lookupCall("0"), lookupCall("1"), textReply("done")];
  const { outcome, model } = await looping(replies, { max_tool_iterations: 3n });
  expect(outcome).toEqual(new Output("done"));
  expect(model.requests).toHaveLength(3);
});
