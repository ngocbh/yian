/**
 * AGENT `retries` and tool-loop defaults come from `defaults` when the node sets none, and an
 * authored value wins over the default.
 */

import { afterEach, expect, test } from "bun:test";

import { flowManifest } from "../../src/compile/manifest.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { type Env, forget, reload } from "../../src/defaults.ts";
import { Message, Response, ToolCallPart, type ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { RunFlowOptions, RunResult } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { NO_ENV, runWith, textReply } from "../helpers/run_state_fixtures.ts";

afterEach(() => {
  forget();
});

// The agent's reply is prompted for, so it is parsed from text and a bad reply costs one retry.
const STRUCTURED = `id: scored
name: scored
llm_config: {provider: vllm, model: m, structured_output: prompt}
input:
  question: !str
nodes:
  agent:
    kind: agent
    prompt: \${question}
    input:
      question: \${input.question}
    output: {score: float}
output: \${agent.output}
`;

/** The manifest `retries` of the loaded `STRUCTURED` flow's AGENT `agent`. */
async function manifestRetries(): Promise<unknown> {
  const flow = await loadFlow(STRUCTURED);
  return ((flowManifest(flow.compiled).nodes as JsonObject).agent as JsonObject).retries;
}

test("the manifest holds two retries by default", async () => {
  expect(await manifestRetries()).toBe(2);
  expect(new AgentNode("n").retries).toBe(2);
});

test("the manifest holds the agent retries default", async () => {
  reload((name) => (name === "YIAN_AGENT_RETRIES" ? "0" : undefined));
  expect(await manifestRetries()).toBe(0);
  expect(new AgentNode("n").retries).toBe(0);
  expect(new AgentNode("n", { retries: 3 }).retries).toBe(3);
});

/** An environment setting `name` to `value` over `NO_ENV`. */
function setting(name: string, value: string): Env {
  return (each) => (each === name ? value : NO_ENV(each));
}

/** Load `text` and run it on `{question: "q"}` with `model`, in `env`; the other options go to `runFlow`. */
async function run(text: string, model: ScriptedModel, env: Env, options: RunFlowOptions = {}): Promise<RunResult> {
  return runWith(await loadFlow(text, { env }), { question: "q" }, model, { env, ...options });
}

/** A model replying a bad score, then a good one. */
function badThenGood(): ScriptedModel {
  return new ScriptedModel(['{"score": "bad"}', '{"score": 1.5}'].map(textReply), {
    provider: "vllm",
    model: "m",
    structuredOutput: "prompt",
  });
}

test("the agent retries default of zero fails on the first bad reply", async () => {
  const model = badThenGood();
  const result = await run(STRUCTURED, model, setting("YIAN_AGENT_RETRIES", "0"));
  expect([result.status, result.failure?.code]).toEqual(["failed", "agent.output_invalid"]);
  expect(model.requests).toHaveLength(1);
});

test("authored retries win over the default", async () => {
  const model = badThenGood();
  const text = STRUCTURED.replace("    output:", "    retries: 1\n    output:");
  const result = await run(text, model, setting("YIAN_AGENT_RETRIES", "0"));
  expect(result.output).toEqual({ score: 1.5 });
  expect(model.requests).toHaveLength(2);
});

const LOOPING = `id: looping
name: looping
llm_config: {provider: openai, model: test-model}
input:
  question: !str
nodes:
  agent:
    kind: agent
    tools: [lookup]
    prompt: \${question}
    input:
      question: \${input.question}
    output: str
output: \${agent.output}
`;

/** A reply calling `lookup` with no arguments as call `id`. */
function lookupCall(id: string): Response {
  const call = new ToolCallPart({ id, name: "lookup", arguments: {} });
  return new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
}

/** Run `text` over `replies` with `lookup` (which answers `ok`) and `YIAN_AGENT_MAX_TOOL_ITERATIONS=1`. */
async function looping(text: string, replies: Response[]) {
  const model = new ScriptedModel(replies, { provider: "openai", model: "test-model" });
  const lookup = new FunctionTool({
    name: "lookup",
    description: "Look it up.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "ok",
  });
  const result = await run(text, model, setting("YIAN_AGENT_MAX_TOOL_ITERATIONS", "1"), { tools: { lookup } });
  return { result, model };
}

test("the agent max_tool_iterations default caps the loop", async () => {
  const { result, model } = await looping(LOOPING, [0, 1, 2, 3, 4].map((i) => lookupCall(String(i))));
  // The reply at the cap runs no call; the one after it is the last, its calls unrun too.
  expect([result.status, result.output]).toEqual(["succeeded", ""]);
  expect(model.requests).toHaveLength(2);
  const answer = model.requests[1]!.messages.at(-1)!.parts[0] as ToolResultPart;
  expect(answer.content).toContain("Tool-call limit reached (1 turn)");
});

test("the authored max_tool_iterations wins over the default", async () => {
  const text = LOOPING.replace("    tools:", "    env: {max_tool_iterations: 3}\n    tools:");
  const { result, model } = await looping(text, [lookupCall("0"), lookupCall("1"), textReply("done")]);
  expect(result.output).toBe("done");
  expect(model.requests).toHaveLength(3);
});
