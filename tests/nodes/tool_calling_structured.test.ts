/**
 * An AGENT's declared `output:` type, driving `runAgent` directly with a hand-built
 * `AgentRunContext`: without tools the structured call is the only call. An `Optional(str)` output
 * is text with or without tools; `Optional(int)` and `Optional(Literal(...))` stay structured.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallStarted } from "../../src/events.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import type { Output } from "../../src/nodes/base.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const NATIVE = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "m", structuredOutput: "prompt" as const };
const INT = Type.scalar(ValueKind.INTEGER);
const SENTIMENT = new Type({ kind: ValueKind.STRING, tags: new Set(["positive", "neutral", "negative"]) });

function textReply(body: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(body)] }), finishReason: "stop" });
}

/** Run `runAgent` over `model` for `outputType`; return the value and the calls' purposes. */
async function run(
  model: ChatModel,
  outputType: Type,
  options: { prompt?: string; tools?: string[]; toolset?: ToolSet } = {},
): Promise<{ value: unknown; purposes: string[] }> {
  const events: unknown[] = [];
  const ctx: AgentRunContext = {
    nodeId: "n",
    messages: [userMessage(options.prompt ?? "produce a value")],
    tools: options.tools ?? [],
    llmConfig: null,
    outputType,
    retries: 2,
    maxToolIterations: -1,
    toolset: options.toolset ?? null,
    systemPrompt: DEFAULT_SYSTEM,
    skills: [],
    emitEvent: (event) => void events.push(event),
    models: { env: NO_ENV, createModel: () => model },
    sent: () => [],
  };
  const value = ((await runAgent(ctx)) as Output).value;
  const purposes = events.filter((event) => event instanceof ModelCallStarted).map((event) => event.purpose);
  return { value, purposes };
}

for (const [id, outputType, reply, expected] of [
  ["literal", SENTIMENT, '{"value": "neutral"}', "neutral"],
  ["scalar", INT, '{"value": 7}', 7n],
  ["list", new Type({ kind: ValueKind.LIST_STRING, element: Type.scalar(ValueKind.STRING) }), '{"items": ["alpha", "beta"]}', ["alpha", "beta"]],
  [
    "record",
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["name", Type.scalar(ValueKind.STRING)],
        ["score", INT],
      ]),
      required: new Set(["name", "score"]),
    }),
    '{"name": "ACME", "score": 9}',
    { name: "ACME", score: 9n },
  ],
] as const) {
  test(`test_no_tools_structured_output_skips_discovery_call[${id}]`, async () => {
    const model = new ScriptedModel([textReply(reply)], NATIVE);
    const result = await run(model, outputType);
    expect(result.value).toEqual(expected);
    expect(result.purposes).toEqual(["structured_native"]);
    expect(model.requests[0]!.tools).toEqual([]);
  });
}

test("no tools structured fallback uses one schema prompt", async () => {
  const model = new ScriptedModel([textReply('{"value": 5}')], PROMPTED);
  const result = await run(model, INT, { prompt: "produce a number" });
  expect(result.value).toBe(5n);
  expect(result.purposes).toEqual(["structured_fallback"]);
  expect(model.requests[0]!.messages.at(-1)!.text().toLowerCase()).toContain("json");
});

test("no tools structured retry never adds a discovery call", async () => {
  const model = new ScriptedModel([textReply('{"value": "not-an-int"}'), textReply('{"value": 11}')], NATIVE);
  const result = await run(model, INT, { prompt: "produce a number" });
  expect(result.value).toBe(11n);
  expect(result.purposes).toEqual(["structured_native", "structured_native"]);
});

test("tool calling literal final answer is structured", async () => {
  const model = new ScriptedModel([textReply('{"value": "neutral"}')], { provider: "ollama", model: "qwen2.5:72b" });
  const result = await run(model, SENTIMENT, { prompt: "classify" });
  expect(result.value).toBe("neutral");
  expect(result.purposes).toEqual(["structured_native"]);
});

test("tool calling declared bare str answer unchanged", async () => {
  const model = new ScriptedModel([textReply("plain answer")]);
  const result = await run(model, Type.scalar(ValueKind.STRING), { prompt: "hi" });
  expect(result.value).toBe("plain answer");
  expect(result.purposes).toEqual(["chat"]);
  expect(model.requests[0]!.outputFormat).toBeNull();
});

const RECORD = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["name", Type.scalar(ValueKind.STRING)],
    ["score", INT],
  ]),
  required: new Set(["name", "score"]),
});

test("tool calling final answer is structured", async () => {
  const seen: Record<string, unknown>[] = [];
  const value = new FunctionTool({
    name: "value",
    description: "Look a topic up.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => {
      seen.push(args);
      return `px:${args.topic}`;
    },
  });
  const call = new ToolCallPart({ id: "1", name: "value", arguments: { topic: "ACME" } });
  const model = new ScriptedModel(
    [
      new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" }),
      textReply("done"),
      textReply('{"name": "ACME", "score": 9}'),
    ],
    NATIVE,
  );
  const result = await run(model, RECORD, { prompt: "hi", tools: ["value"], toolset: new ToolSet({ value }) });
  // A plain record: the structured final answer.
  expect(result.value).toEqual({ name: "ACME", score: 9n });
  expect(result.purposes).toEqual(["chat", "chat", "structured_native"]);
  // The mid-loop tool still ran.
  expect(seen).toEqual([{ topic: "ACME" }]);
  // The structured call reads the conversation, the final reply included.
  expect(model.requests[2]!.messages.map((message) => [message.role, message.parts])).toEqual([
    ["system", [new TextPart(DEFAULT_SYSTEM)]],
    ["user", [new TextPart("hi")]],
    ["assistant", [call]],
    ["tool", [new ToolResultPart({ callId: "1", name: "value", content: "px:ACME" })]],
    ["assistant", [new TextPart("done")]],
  ]);
});

const OPTIONAL_STR = new Type({ kind: ValueKind.STRING, nullable: true });

test("an Optional(str) output without tools makes one chat call and returns the text", async () => {
  const model = new ScriptedModel([textReply("plain answer")], NATIVE);
  const result = await run(model, OPTIONAL_STR, { prompt: "hi" });
  expect(result.value).toBe("plain answer");
  expect(result.purposes).toEqual(["chat"]);
  expect(model.requests[0]!.outputFormat).toBeNull();
});

test("an empty Optional(str) reply is empty text, not null", async () => {
  const model = new ScriptedModel([textReply("")], NATIVE);
  expect((await run(model, OPTIONAL_STR)).value).toBe("");
});

test("an Optional(str) output after tools makes no trailing structured call", async () => {
  const value = new FunctionTool({
    name: "value",
    description: "Look a topic up.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => `px:${args.topic}`,
  });
  const call = new ToolCallPart({ id: "1", name: "value", arguments: { topic: "ACME" } });
  const model = new ScriptedModel(
    [new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" }), textReply("done")],
    NATIVE,
  );
  const result = await run(model, OPTIONAL_STR, { prompt: "hi", tools: ["value"], toolset: new ToolSet({ value }) });
  expect(result.value).toBe("done");
  // ScriptedModel throws on a third request, so this also proves no structured call follows.
  expect(result.purposes).toEqual(["chat", "chat"]);
  expect(model.requests.map((request) => request.outputFormat)).toEqual([null, null]);
});

for (const [id, outputType, reply, expected] of [
  ["Optional(int)", new Type({ kind: ValueKind.INTEGER, nullable: true }), '{"value": null}', null],
  ["Optional(Literal(a, b))", new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b"]), nullable: true }), '{"value": "b"}', "b"],
] as const) {
  test(`${id} output stays structured`, async () => {
    const model = new ScriptedModel([textReply(reply)], NATIVE);
    const result = await run(model, outputType);
    expect(result.value).toEqual(expected);
    expect(result.purposes).toEqual(["structured_native"]);
    expect(model.requests[0]!.outputFormat).not.toBeNull();
  });
}
