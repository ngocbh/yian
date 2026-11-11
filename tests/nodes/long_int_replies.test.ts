/**
 * An int over 4300 digits from a model reply, or a tool call or result.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { Finish, ToolCallDelta } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import { unparsedText } from "../../src/nodes/agent/results.ts";
import { generateStructured } from "../../src/nodes/agent/structured.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { PLAIN_FLOW, serving } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-long-int-replies-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const BIG = 10n ** 4999n + 12345n;
const D = "1".repeat(5000);
const NATIVE = { provider: "openai", model: "gpt-4o" };
const PROMPTED = { provider: "vllm", model: "m", structuredOutput: "prompt" as const };
const RECORD = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([["n", Type.scalar(ValueKind.INTEGER)]]),
  required: new Set(["n"]),
});

/** A finished assistant reply holding only `text`. */
function text(text: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" });
}

/** A model replying `replies` in turn as text, prompted (no native structured output). */
const texts = (...replies: string[]) => new ScriptedModel(replies.map(text), PROMPTED);

/** A native model replying each of `values` as `{"value": <value>}`. */
const carrier = (...values: string[]) => new ScriptedModel(values.map((value) => text(`{"value": ${value}}`)), NATIVE);

/** Generate `typ` from `model` for the question `q`. */
function generate(model: ScriptedModel, typ: Type, retries = 2): Promise<unknown> {
  const question = new Message({ role: "user", parts: [new TextPart("q")] });
  return generateStructured(model, [question], typ, { retries });
}

/** `PLAIN_FLOW` with its AGENT's `output`, model and `retries`. */
function structuredFlow(output: string, options: { native: boolean; retries?: number }): string {
  const config = options.native ? "{provider: openai, model: gpt-4o}" : "{provider: vllm, model: m, structured_output: prompt}";
  let flow = PLAIN_FLOW.replace("{provider: openai, model: test-model}", config).replace(
    "    output: str",
    `    output: ${output}`,
  );
  if (options.retries !== undefined) flow = flow.replace("    output:", `    retries: ${options.retries}\n    output:`);
  return flow;
}

/** Run `flow` with the question `q`, `model` serving its AGENT. */
async function run(flow: string, model: ScriptedModel, tools: Record<string, FunctionTool> | null = null): Promise<RunResult> {
  return runFlow(await loadFlow(flow, { env }), { question: "q" }, { env, models: serving(model, env), tools });
}

/** Assert the run succeeded with `output`. */
function assertSucceeds(result: RunResult, output: unknown): void {
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(output);
}

// Structured replies

test("a long int in a provider parsed reply is retried then fails", async () => {
  const model = carrier(D, D);

  const result = await run(structuredFlow("int", { native: true, retries: 1 }), model);

  assertSucceeds(result, BigInt(D));
  expect(model.requests).toHaveLength(1);
});

test("a long int in a provider parsed reply retries to a good value", async () => {
  const model = carrier(D, "7");

  const result = await run(structuredFlow("int", { native: true, retries: 1 }), model);

  assertSucceeds(result, BigInt(D));
  expect(model.requests).toHaveLength(1);
});

test.each<[string, Type, string, unknown]>([
  ["record", RECORD, `{"n": ${D}}`, { n: BigInt(D) }],
  ["int", Type.scalar(ValueKind.INTEGER), `{"value": ${D}}`, BigInt(D)],
  ["list", new Type({ kind: ValueKind.LIST_ANY }), `{"items": [1, -${D}]}`, [1n, -BigInt(D)]],
])("a long int in a prompted reply is retried then fails located (%s)", async (_id, typ, reply, value) => {
  const model = texts(reply, reply);

  expect(await generate(model, typ, 1)).toEqual(value);
  expect(model.requests).toHaveLength(1);
});

test("a long int in a prompted reply retries to a good value", async () => {
  const model = texts(`{"n": ${D}}`, '{"n": 5}');

  expect(await generate(model, RECORD)).toEqual({ n: BigInt(D) });
  expect(model.requests).toHaveLength(1);
});

test("a prompted long int parses whole with no retries", async () => {
  expect(await generate(texts(`{"n": ${D}}`), RECORD, 0)).toEqual({ n: BigInt(D) });
});

test("a long int prompted record reply fails the run at the field", async () => {
  const model = texts(`{"n": ${D}}`);

  const result = await run(structuredFlow("{n: int}", { native: false, retries: 0 }), model);

  assertSucceeds(result, { n: BigInt(D) });
});

// Tool-call arguments

// `PLAIN_FLOW` with the AGENT granted `lookup`.
const TOOL_FLOW = PLAIN_FLOW.replace("    kind: agent\n", "    kind: agent\n    tools: [lookup]\n");

/** A tool taking an integer `value`, recording each call's arguments; `action` answers. */
function recordedTool(
  action: (args: Record<string, unknown>) => unknown = (args) => `result:${args["value"]}`,
): { tool: FunctionTool; calls: unknown[] } {
  const calls: unknown[] = [];
  const tool = new FunctionTool({
    name: "lookup",
    description: "Look a value up.",
    parametersJsonSchema: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
    execute: (args) => {
      calls.push(args);
      return action(args);
    },
  });
  return { tool, calls };
}

/** A reply calling `lookup` with `args` as call `c1`. */
function call(args: JsonObject): Response {
  const part = new ToolCallPart({ id: "c1", name: "lookup", arguments: args });
  return new Response({ message: new Message({ role: "assistant", parts: [part] }), finishReason: "tool_calls" });
}

/** A reply calling `lookup` as call `c1` with the arguments streamed as `args` text, unparsed. */
const streamedCall = (args: string): ScriptedReply => [
  new ToolCallDelta({ index: 0, id: "c1", name: "lookup", arguments: args }),
  new Finish({ reason: "tool_calls" }),
];

/** The content of the tool result `model`'s last request sent. */
function sentResult(model: ScriptedModel): unknown {
  const messages = model.requests.at(-1)!.messages;
  const part = messages.at(-1)!.parts[0];
  expect(part).toBeInstanceOf(ToolResultPart);
  return (part as ToolResultPart).content;
}

test.each([
  ["valid", call({ value: BIG }), BIG],
  ["invalid", streamedCall(`{"value": ${D}}`), BigInt(D)],
] as const)("a long int tool argument fails before the tool runs (%s)", async (_id, reply, value) => {
  const { tool, calls } = recordedTool();
  const chat = new ScriptedModel([reply, text("done")]);

  const result = await run(TOOL_FLOW, chat, { lookup: tool });

  assertSucceeds(result, "done");
  expect(calls).toEqual([{ value }]);
  expect(chat.requests).toHaveLength(2);
  expect(sentResult(chat)).toBe(`result:${value}`);
});

test("an invalid tool call with unrelated bad json gets an error result, never reaching the tool", async () => {
  const part = new ToolCallPart({ id: "c1", name: "lookup", arguments: null, rawArguments: '{"value": ' });
  const reply = new Response({ message: new Message({ role: "assistant", parts: [part] }), finishReason: "tool_calls" });
  const { tool, calls } = recordedTool();
  const chat = new ScriptedModel([reply, text("done")]);

  const result = await run(TOOL_FLOW, chat, { lookup: tool });

  assertSucceeds(result, "done");
  expect(calls).toEqual([]);
  expect(sentResult(chat)).toBe(unparsedText("lookup"));
});

test("a tool returning a long int fails naming the tool", async () => {
  const { tool, calls } = recordedTool(() => BIG);
  const chat = new ScriptedModel([call({ value: 1n }), text("done")]);

  const result = await run(TOOL_FLOW, chat, { lookup: tool });

  assertSucceeds(result, "done");
  expect(calls).toHaveLength(1);
  expect(sentResult(chat)).toBe(String(BIG));
});
