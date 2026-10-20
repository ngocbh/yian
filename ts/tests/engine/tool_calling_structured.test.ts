/**
 * An AGENT's declared `output:` type: without tools the structured call is the only call; mirrors
 * `tests/engine/test_tool_calling_structured.py`, driving `runAgent` directly with a hand-built
 * `AgentRunContext`.
 *
 * Ported: `test_tool_calling_final_answer_is_structured` (a tool turn, then the structured call over
 * the conversation, the final reply included),
 * `test_no_tools_structured_output_skips_discovery_call` (literal, scalar, list and record),
 * `test_no_tools_structured_fallback_uses_one_schema_prompt`,
 * `test_no_tools_structured_retry_never_adds_a_discovery_call`,
 * `test_tool_calling_literal_final_answer_is_structured` and
 * `test_tool_calling_declared_bare_str_answer_unchanged` (both grant no tools).
 *
 * Changed: the rows run `runAgent` rather than `eval_node`; the model is a `ScriptedModel` rather
 * than a LangChain chat model that fails on an unexpected `invoke` or `bind_tools`, so each row
 * asserts the purposes of the calls made and the requests' tools instead. The native rows use the
 * `openai`/`gpt-5.5` pair (Python: `anthropic`/`claude-sonnet-4-5`); a protocol-less model takes
 * the native `json_schema` plan whatever its provider, so the Ollama row asserts the structured
 * purpose. The tool is in a `ToolSet` given as `toolset`, where Python patches `TOOL_REGISTRY`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallStarted } from "../../src/events.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const NATIVE = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
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
    prompt: options.prompt ?? "produce a value",
    tools: options.tools ?? [],
    llmConfig: null,
    outputType,
    retries: 2,
    maxToolIterations: -1,
    memoryContext: new MemoryContext(),
    toolset: options.toolset ?? null,
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    emitEvent: (event) => void events.push(event),
    models: { env: () => undefined, createModel: () => model },
    sent: () => [],
  };
  const value = (await runAgent(ctx)).value;
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

test("test_no_tools_structured_fallback_uses_one_schema_prompt", async () => {
  const model = new ScriptedModel([textReply('{"value": 5}')], PROMPTED);
  const result = await run(model, INT, { prompt: "produce a number" });
  expect(result.value).toBe(5n);
  expect(result.purposes).toEqual(["structured_fallback"]);
  expect(model.requests[0]!.messages.at(-1)!.text().toLowerCase()).toContain("json");
});

test("test_no_tools_structured_retry_never_adds_a_discovery_call", async () => {
  const model = new ScriptedModel([textReply('{"value": "not-an-int"}'), textReply('{"value": 11}')], NATIVE);
  const result = await run(model, INT, { prompt: "produce a number" });
  expect(result.value).toBe(11n);
  expect(result.purposes).toEqual(["structured_native", "structured_native"]);
});

test("test_tool_calling_literal_final_answer_is_structured", async () => {
  const model = new ScriptedModel([textReply('{"value": "neutral"}')], { provider: "ollama", model: "qwen2.5:72b" });
  const result = await run(model, SENTIMENT, { prompt: "classify" });
  expect(result.value).toBe("neutral");
  expect(result.purposes).toEqual(["structured_native"]);
});

test("test_tool_calling_declared_bare_str_answer_unchanged", async () => {
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

test("test_tool_calling_final_answer_is_structured", async () => {
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
