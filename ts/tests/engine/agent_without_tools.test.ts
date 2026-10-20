/**
 * An AGENT without tools or skills makes a single model call; one with tools runs the loop:
 * mirrors `tests/engine/test_agent_without_tools.py`.
 *
 * Ported: `test_skills_and_steerable_need_no_mode`.
 *
 * Changed: `test_text_agent_without_tools_makes_one_unbound_call` and
 * `test_structured_agent_without_tools_makes_one_structured_call` run a hand-built
 * `START -> chat -> END` flow through `FlowEngine` (Python: `load_flow` and `run_flow`, T3-22/T4-8),
 * with a `ScriptedModel` from `FlowEngineOptions.models`; the request sending no tools stands for
 * Python's unbound and structured calls.
 *
 * Ported through `evalNode` (`drive`), the tool in the run's `ToolSet`:
 * `test_python_built_agent_with_tools_runs_the_tool_loop` (the bound tools read from the request).
 *
 * Changed: `test_steer_joins_the_single_request_after_the_prompt` and
 * `test_interrupt_while_streaming_keeps_the_partial_text` run a hand-built steerable `chat` with a
 * test inbox through `FlowEngine`; the reply streams as listed text deltas.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { RunSucceeded, type StreamChunk } from "../../src/events.ts";
import { Finish, TextDelta } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { type Node, Output } from "../../src/nodes/base.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import type { FlowEngineOptions } from "../../src/runtime/engine.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, engine, serving } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };

function reply(text: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" });
}

/** Run `START -> chat -> END`, `chat` an AGENT with no tools and output `output`; return the run's output. */
async function run(
  output: Type,
  model: ScriptedModel,
  options: { steerable?: boolean } & Partial<FlowEngineOptions> = {},
): Promise<unknown> {
  const { steerable = false, ...rest } = options;
  const chat = new AgentNode("chat", { prompt: "Help the user.", llmConfig: CONFIG, steerable });
  chat.outputContract = new ValueContract(output);
  const flow = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["chat", chat],
      [END_ID, EndNode.value(END_ID, { outputName: "output" })],
    ]),
    [
      new Edge("start-chat", START_ID, "chat", { optional: true, ordering: true }),
      new Edge("chat-end", "chat", END_ID, { inputGroup: "output" }),
    ],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["chat", new Map()],
        [END_ID, new Map([["output", "${chat.output}"]])],
      ]),
    },
  );
  const events = await engine(flow, { models: { env: () => undefined, createModel: () => model }, ...rest }).run();
  const last = events.at(-1);
  expect(last).toBeInstanceOf(RunSucceeded);
  return (last as RunSucceeded).output;
}

test("text agent without tools makes one unbound call", async () => {
  const model = new ScriptedModel([reply("hello")], CONFIG);
  expect(await run(Type.scalar(ValueKind.STRING), model)).toBe("hello");
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.tools).toEqual([]);
});

test("structured agent without tools makes one structured call", async () => {
  const output = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["name", Type.scalar(ValueKind.STRING)],
      ["score", Type.scalar(ValueKind.INTEGER)],
    ]),
    required: new Set(["name", "score"]),
  });
  const model = new ScriptedModel([reply('{"name": "Ada", "score": 3}')], CONFIG);
  expect(await run(output, model)).toEqual({ name: "Ada", score: 3n });
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
  expect(await run(Type.scalar(ValueKind.STRING), model, { steerable: true, steering })).toBe("ok");
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.messages.slice(-2).map((message) => [message.role, message.text()])).toEqual([
    ["user", "Help the user."],
    ["user", "be brief"],
  ]);
});

test("an interrupt while streaming keeps the partial text", async () => {
  const model = new ScriptedModel([[new TextDelta(0, "hello"), new TextDelta(0, " big"), new TextDelta(0, " world"), new Finish({ reason: "stop" })]], CONFIG);
  const interrupt = new AbortController();
  const steering = { interruptSignal: () => interrupt.signal, drain: () => [] };
  const chunks: StreamChunk[] = [];
  const onStream = (chunk: StreamChunk) => {
    chunks.push(chunk);
    if (chunk.chunk === " big") interrupt.abort(new Error("interrupted"));
  };
  expect(await run(Type.scalar(ValueKind.STRING), model, { steerable: true, steering, onStream })).toBe("hello big");
  expect(chunks.map((chunk) => [chunk.chunk, chunk.final])).toEqual([
    ["hello", false],
    [" big", false],
    ["", true],
  ]);
});
