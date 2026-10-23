/**
 * A port of `tests/engine/test_long_int_replies.py`: an int over 4300 digits from a model reply,
 * a tool call or result, a backward reply, or backward feedback.
 *
 * Ported: every row but those not ported below. The models are `ScriptedModel`s served through
 * `models`, where Python patches `model_from_config`; a native model (`gpt-4o`) replies with JSON
 * text holding the value under `value`, where Python's `_Carrier` puts it in a raw tool call. The
 * tool is a `FunctionTool` taking an integer `value` (Python's `_Tool`), given through `tools`.
 *
 * Changed: ints are `bigint`s with no digit limit (decisions 4a, 23), so a long int from a reply,
 * tool call or tool result is accepted whole where Python retries and then fails with its
 * digit-limit message. A structured reply's whole int is the output at the first reply, with no
 * correction, so the `retries_to_a_good_value` rows output the long int too and the next reply
 * is never asked for. A tool call's long int argument runs the tool with the whole value, and a
 * tool's long int result is sent back whole; the model then answers `done`. The `invalid` row
 * streams the arguments as text (Python's parser refused that text; TS's assembler parses it
 * whole). A long int in a backward reply's `self_feedback`, or as backward feedback, is refused
 * for being an int, not for its size: `self_feedback messages must be strings`, and
 * `feedback["note"] must be a string, object, list, or None` (Python gives both messages for any
 * int). NaN feedback is refused by host intake first, with a `TypeCheckError` (a `RangeError`),
 * where Python's carrier check raises a `TypeError`.
 *
 * Not ported: the `python_limit=0` and `python_limit=6000` rows of
 * `test_a_prompted_long_int_fails_the_same_whatever_pythons_limit` and the `python_limit=0` rows
 * of `test_a_tool_returning_a_long_int_fails_naming_the_tool` (Python's
 * `sys.set_int_max_str_digits`),
 * `test_a_parser_digit_limit_error_reads_as_yians_in_correction_and_failure` and
 * `test_a_node_raising_pythons_digit_limit_error_fails_with_yians_message` (Python's
 * digit-limit error text, which nothing in TS raises), and the `session` rows (TS yian has no session
 * memory).
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { FeedbackValue } from "../../src/expr/lineage.ts";
import { Finish, ToolCallDelta } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { generateStructured } from "../../src/nodes/agent/structured.ts";
import { BackwardContext } from "../../src/nodes/base.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { PLAIN_FLOW, rejected, serving } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-long-int-replies-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const BIG = 10n ** 4999n + 12345n;
const D = "1".repeat(5000);
const NATIVE = { provider: "openai", model: "gpt-4o" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
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
  const config = options.native ? "{provider: openai, model: gpt-4o}" : "{provider: vllm, model: no-structured-sentinel}";
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

test("a prompted long int fails the same whatever pythons limit (None)", async () => {
  expect(await generate(texts(`{"n": ${D}}`), RECORD, 0)).toEqual({ n: BigInt(D) });
});

test("a long int prompted record reply fails the run at the field", async () => {
  const model = texts(`{"n": ${D}}`);

  const result = await run(structuredFlow("{n: int}", { native: false, retries: 0 }), model);

  assertSucceeds(result, { n: BigInt(D) });
});

// Tool-call arguments

// Python's `_TOOL_FLOW`: `PLAIN_FLOW` with the AGENT granted `lookup`.
const TOOL_FLOW = PLAIN_FLOW.replace("    kind: agent\n", "    kind: agent\n    tools: [lookup]\n");

/** Python's `_Tool`: a tool taking an integer `value`, recording each call's arguments; `action` answers. */
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

test("an invalid tool call with unrelated bad json keeps todays behaviour", async () => {
  const part = new ToolCallPart({ id: "c1", name: "lookup", arguments: null, rawArguments: '{"value": ' });
  const reply = new Response({ message: new Message({ role: "assistant", parts: [part] }), finishReason: "tool_calls" });
  const { tool, calls } = recordedTool();

  const result = await run(TOOL_FLOW, new ScriptedModel([reply]), { lookup: tool });

  assertSucceeds(result, "");
  expect(calls).toEqual([]);
});

test("a tool returning a long int fails naming the tool", async () => {
  const { tool, calls } = recordedTool(() => BIG);
  const chat = new ScriptedModel([call({ value: 1n }), text("done")]);

  const result = await run(TOOL_FLOW, chat, { lookup: tool });

  assertSucceeds(result, "done");
  expect(calls).toHaveLength(1);
  expect(sentResult(chat)).toBe(String(BIG));
});

// Backward replies and feedback

/** Python's `_context`: a recorded AGENT backward context. */
function context(): BackwardContext {
  return new BackwardContext(
    { topic: "reefs", tags: ["a", "b"] },
    {
      inputNames: ["topic", "tags"],
      variableNames: ["tone"],
      output: { answer: "recorded answer", tags: ["a", "b"] },
      runId: "recorded-run",
      nodeId: "recorded-node",
      nodeStartedSeq: 3,
      nodeData: {
        prompt_template: "${topic}: ${tone}",
        variables: { tone: "concise" },
        variable_domains: { tone: 'Literal("concise","detailed")' },
        rendered_prompt: "reefs: concise",
        effective_llm_config: { provider: "anthropic", model: "recorded", anthropic: { temperature: 0.4 } },
        _evaluation_evidence: null,
      },
      downstreamContext: [{ message_index: 0, contexts: [{ expression: "items[1]" }] }],
    },
  );
}

/** Run the backward of an AGENT `writer` whose model replies `reply`. */
function backward(reply: string) {
  const node = new AgentNode("writer", { prompt: "p", llmConfig: { provider: "test", model: "recorded" } });
  return node.backward(context(), ["Review."], { models: serving(new ScriptedModel([text(reply)]), env) });
}

test("a long int in a backward reply fails the backward located", async () => {
  const reply = `{"input_feedback": {}, "self_feedback": ["ok", ${D}], "variable_feedback": {}}`;

  const error = await rejected(backward(reply));

  expect(error).toBeInstanceOf(TypeError);
  expect((error as TypeError).message).toBe("self_feedback messages must be strings");
});

/** Return `"x"` wrapped in `levels` one-item lists. */
function nestedMessage(levels: number): FeedbackValue {
  let message: FeedbackValue = "x";
  for (let level = 0; level < levels; level++) message = [message];
  return message;
}

/** Run a backward whose reply gives `message` as the only feedback on `topic`. */
function backwardReply(message: FeedbackValue) {
  return backward(JSON.stringify({ input_feedback: { topic: [message] }, self_feedback: [], variable_feedback: {} }));
}

test.each([99, 100])("a backward reply measures nesting from its feedback messages (%p)", async (levels) => {
  const message = nestedMessage(levels);

  expect((await backwardReply(message)).inputFeedback).toEqual({ topic: [message] });
});

test("a backward feedback message past the limit still fails", async () => {
  const error = await rejected(backwardReply(nestedMessage(102)));

  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as TypeCheckError).message).toStartWith("backward response at input_feedback.topic[0]");
  expect((error as TypeCheckError).message).toContain("deeper than 100 levels");
});

// Python's `_code_flow('return {"note": "x"}')`.
const CODE_FLOW = `
id: nesting
name: nesting
nodes:
  deep:
    kind: code
    code: |
      return {"note": "x"}
output: \${deep.output}
`;

test.each([
  ["record", { note: BIG }, 'feedback["note"]'],
  ["list", [BIG], "feedback[0]"],
])("long int feedback is rejected located and the result stays usable (%s)", async (_id, feedback, where) => {
  const result = await runFlow(await loadFlow(CODE_FLOW, { env }), {}, { env, learnable: true });
  expect(result.status).toBe("succeeded");

  const error = await rejected(result.backward(feedback));

  expect(error).toBeInstanceOf(TypeError);
  expect((error as TypeError).message).toBe(`${where} must be a string, object, list, or None`);
  expect(result.backwardOutcome).toBeNull();
  expect((await result.backward("good")).status).toBe("succeeded");
});

test("nan feedback still raises a type error", async () => {
  const result = await runFlow(await loadFlow(CODE_FLOW, { env }), {}, { env, learnable: true });

  const error = await rejected(result.backward({ note: Number.NaN }));

  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as TypeCheckError).message).toBe("feedback at note: NaN is not a yian value");
});
