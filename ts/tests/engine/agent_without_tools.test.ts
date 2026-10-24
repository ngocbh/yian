/**
 * An AGENT without tools or skills makes a single model call; one with tools runs the loop:
 * mirrors `tests/engine/test_agent_without_tools.py`.
 *
 * Ported: `test_skills_and_steerable_need_no_mode`.
 *
 * Changed: `test_text_agent_without_tools_makes_one_unbound_call`,
 * `test_structured_agent_without_tools_makes_one_structured_call`,
 * `test_steer_joins_the_single_request_after_the_prompt` and
 * `test_interrupt_while_streaming_keeps_the_partial_text` load the flow and run it through
 * `runWith` (`runFlow` with a `ScriptedModel` served by `models`, where Python patches
 * `model_from_config`); the request sending no tools stands for Python's unbound and structured
 * calls, and the reply streams as listed text deltas.
 *
 * Ported through `evalNode` (`drive`), the tool in the run's `ToolSet`:
 * `test_python_built_agent_with_tools_runs_the_tool_loop` (the bound tools read from the request).
 *
 * TS-only (pins): an `Optional(str)` agent without tools makes one unbound call with no output
 * format and returns the text; an empty reply is `""`, not null; an interrupt before the first
 * call gives `""`, not null. The interrupt pin changes to a null output once an interrupted
 * `Optional(str)` agent returns null.
 */

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import type { StreamChunk } from "../../src/events.ts";
import { Finish, TextDelta } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import type { RunFlowOptions } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { drive, runWith, serving } from "./run_state_fixtures.ts";

const CONFIG = { provider: "anthropic", model: "claude-test" };

function reply(text: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" });
}

/** One `chat` agent with no tools, plus the given body lines (Python's `_flow`). */
function chatFlow(...lines: string[]): string {
  const body = lines.map((line) => `    ${line}\n`).join("");
  return `id: single
name: single
llm_config: {provider: anthropic, model: claude-test}
nodes:
  chat:
    kind: agent
    prompt: Help the user.
${body}output: \${chat.output}
`;
}

/** Load `chatFlow(...lines)` and run it with `model`; return the run's output once it succeeds. */
async function run(lines: string[], model: ScriptedModel, options: RunFlowOptions = {}): Promise<unknown> {
  const result = await runWith(await loadFlow(chatFlow(...lines)), {}, null, model, options);
  expect(result.status).toBe("succeeded");
  return result.output;
}

test("text agent without tools makes one unbound call", async () => {
  const model = new ScriptedModel([reply("hello")], CONFIG);
  expect(await run(["output: str"], model)).toBe("hello");
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.tools).toEqual([]);
});

test("structured agent without tools makes one structured call", async () => {
  const model = new ScriptedModel([reply('{"name": "Ada", "score": 3}')], CONFIG);
  expect(await run(["output: {name: str, score: int}"], model)).toEqual({ name: "Ada", score: 3n });
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.tools).toEqual([]);
});

test("skills and steerable need no mode", () => {
  expect(new AgentNode("n", { prompt: "p", skills: ["review"] }).skills).toEqual(["review"]);
  expect(new AgentNode("n", { prompt: "p", steerable: true }).steerable).toBe(true);
});

test("a Python-built agent with tools runs the tool loop", async () => {
  const seen: unknown[] = [];
  const value = new FunctionTool({
    name: "value",
    description: "Price a topic.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } } },
    execute: (args) => {
      seen.push(args);
      return `px:${args.topic}`;
    },
  });
  const call = new ToolCallPart({ id: "1", name: "value", arguments: { topic: "ACME" } });
  const calling = new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
  const model = new ScriptedModel([calling, reply("done")], CONFIG);
  const node = new AgentNode("n", { prompt: "hi", tools: ["value"], llmConfig: CONFIG });

  const outcome = await drive(node, { models: serving(model), tools: new ToolSet({ value }) });

  expect(outcome).toEqual(new Output("done"));
  expect(seen).toEqual([{ topic: "ACME" }]);
  expect(model.requests).toHaveLength(2);
  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual(["value"]);
});

test("a steer joins the single request after the prompt", async () => {
  const model = new ScriptedModel([reply("ok")], CONFIG);
  const steering = { interruptSignal: () => new AbortController().signal, drain: () => ["be brief"] };
  expect(await run(["steerable: true", "output: str"], model, { steering })).toBe("ok");
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.messages.slice(-2).map((message) => [message.role, message.text()])).toEqual([
    ["user", "Help the user."],
    ["user", "be brief"],
  ]);
});

test("an interrupt while streaming keeps the partial text", async () => {
  const model = new ScriptedModel(
    [[new TextDelta(0, "hello"), new TextDelta(0, " big"), new TextDelta(0, " world"), new Finish({ reason: "stop" })]],
    CONFIG,
  );
  const interrupt = new AbortController();
  const steering = { interruptSignal: () => interrupt.signal, drain: () => [] };
  const chunks: StreamChunk[] = [];
  const onStream = (chunk: StreamChunk) => {
    chunks.push(chunk);
    if (chunk.chunk === " big") interrupt.abort(new Error("interrupted"));
  };
  // Serial delivery, so the interrupt is set before the next chunk.
  const options = { steering, onStream, numWorkers: 0 };
  expect(await run(["steerable: true", "output: str"], model, options)).toBe("hello big");
  expect(chunks.map((chunk) => [chunk.chunk, chunk.final])).toEqual([
    ["hello", false],
    [" big", false],
    ["", true],
  ]);
});

test("an Optional(str) agent without tools makes one unbound call and returns the text", async () => {
  const model = new ScriptedModel([reply("hello")], CONFIG);
  expect(await run(["output: Optional(str)"], model)).toBe("hello");
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.tools).toEqual([]);
  expect(model.requests[0]!.outputFormat).toBeNull();
});

test("an empty reply from an Optional(str) agent is empty text, not null", async () => {
  const model = new ScriptedModel([reply("")], CONFIG);
  expect(await run(["output: Optional(str)"], model)).toBe("");
});

test('an interrupt before the first call gives an Optional(str) agent "", not null', async () => {
  // Null is reserved for an interrupt under the inbox model; until then an interrupt keeps text.
  const interrupt = new AbortController();
  interrupt.abort(new Error("interrupted"));
  const model = new ScriptedModel([], CONFIG);
  const steering = { interruptSignal: () => interrupt.signal, drain: () => [] };
  expect(await run(["steerable: true", "output: Optional(str)"], model, { steering })).toBe("");
  expect(model.requests).toEqual([]);
});
