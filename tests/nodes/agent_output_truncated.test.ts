/**
 * An AGENT whose reply the provider cut off at the token limit (`finishReason` `length`) with no
 * runnable call fails with `OutputTruncatedError`, naming the setting that raises the limit: a
 * text-only reply, or one whose only calls didn't parse. A cut reply that still has a parsed call
 * runs it, as any reply does. A flow sees the failure as `agent.output_truncated`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Message, type Part, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { OutputTruncatedError, outputLimitSetting, userMessage } from "../../src/nodes/agent/structured.ts";
import type { Output } from "../../src/nodes/base.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, type ValueRecord, ValueKind } from "../../src/typesys/values.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { serving, textReply } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-output-truncated-"));
beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const CONFIG = { provider: "openai", model: "gpt-5.5" };
const CUT_OFF =
  "the reply was cut off at the output token limit before it finished; raise the limit in the agent's " +
  "llm_config (openai.max_output_tokens)";

/** A reply holding `parts` that the provider stopped at the token limit. */
function cut(...parts: Part[]): Response {
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "length" });
}

/** A tool `a` that logs each call and returns `a done`. */
function logged(log: string[]): ToolSet {
  const execute = () => {
    log.push("a");
    return "a done";
  };
  const parametersJsonSchema = { type: "object", properties: { n: { type: "integer" } } };
  return new ToolSet({ a: new FunctionTool({ name: "a", description: "Tool a.", parametersJsonSchema, execute }) });
}

/** One `runAgent` over `model` granting `a`, with its steps; `outputType` makes it structured. */
function agent(model: ScriptedModel, log: string[], outputType: Type | null = null) {
  const steps: ValueRecord[] = [];
  const ctx: AgentRunContext = {
    nodeId: "n",
    messages: [userMessage("go")],
    tools: ["a"],
    llmConfig: CONFIG,
    outputType,
    retries: 0,
    maxToolIterations: -1,
    steps: new NodeSteps(1, [], (step) => steps.push(step)),
    toolset: logged(log),
    systemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: serving(model),
    sent: () => [],
  };
  return { running: runAgent(ctx), steps };
}

/** The error `promise` rejects with. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("the promise resolved");
}

test("a text reply cut off at the token limit fails, naming the setting to raise", async () => {
  const model = new ScriptedModel([cut(new TextPart("The three steps are: first, open the valve; second,"))], CONFIG);
  const error = await rejection(agent(model, []).running);
  expect(error).toBeInstanceOf(OutputTruncatedError);
  expect((error as Error).message).toBe(CUT_OFF);
  expect(model.requests).toHaveLength(1);
});

test("a cut reply whose only calls didn't parse fails, and no call runs", async () => {
  const log: string[] = [];
  const unparsed = new ToolCallPart({ id: "c0", name: "a", arguments: null, rawArguments: '{"n": 1' });
  const model = new ScriptedModel([cut(unparsed), textReply("done")], CONFIG);
  const run = agent(model, log);
  expect(await rejection(run.running)).toBeInstanceOf(OutputTruncatedError);
  expect(log).toEqual([]);
  expect(run.steps.filter((step) => step["kind"] === "tool_result")).toEqual([]);
  expect(model.requests).toHaveLength(1);
});

test("a cut reply with a parsed call runs it, and the loop goes on", async () => {
  const log: string[] = [];
  const parsed = new ToolCallPart({ id: "c0", name: "a", arguments: { n: 1n } });
  const unparsed = new ToolCallPart({ id: "c1", name: "a", arguments: null, rawArguments: '{"n": 2' });
  const model = new ScriptedModel([cut(parsed, unparsed), textReply("done")], CONFIG);
  expect(((await agent(model, log).running) as Output).value).toBe("done");
  expect(log).toEqual(["a"]);
});

test("a structured agent's cut chat reply fails before its structured call", async () => {
  const model = new ScriptedModel([cut(new TextPart("Thinking it over"))], CONFIG);
  const error = await rejection(agent(model, [], Type.scalar(ValueKind.INTEGER)).running);
  expect(error).toBeInstanceOf(OutputTruncatedError);
  expect(model.requests).toHaveLength(1);
});

test.each([
  [{ provider: "openai", model: "m" }, "openai.max_output_tokens"],
  [{ provider: "anthropic", model: "m" }, "anthropic.max_tokens"],
  [{ provider: "google", model: "m" }, "google.generationConfig.maxOutputTokens"],
  [{ provider: "ollama", model: "m" }, "ollama.options.num_predict"],
  [{ provider: "deepseek", model: "m" }, "deepseek.max_tokens"],
  [{ provider: "azure", model: "m" }, "azure.max_tokens"],
])("the setting for %p is %s", (options, setting) => {
  expect(outputLimitSetting(new ScriptedModel([], options))).toBe(setting);
});

test("a flow sees the cut reply as agent.output_truncated", async () => {
  const env = hostEnv(dir);
  const model = new ScriptedModel([cut(new TextPart("A li"))], CONFIG);
  const text =
    "id: f\nname: f\nllm_config: {provider: openai, model: gpt-5.5}\n" +
    "nodes:\n  w: {kind: agent, prompt: Write a line., output: !str}\noutput: ${w.output}\n";
  const result = await runFlow(await loadFlow(text, { env }), {}, { env, models: serving(model, env) });
  expect(result.status).toBe("failed");
  expect(result.failure?.code).toBe("agent.output_truncated");
  expect(result.failure?.message).toBe(CUT_OFF);
  expect(result.failure?.nodeId).toBe("w");
});
