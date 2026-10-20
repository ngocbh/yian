/**
 * Value nesting at the binding boundary and in AGENT structured replies, a port of
 * `tests/engine/test_value_nesting_boundaries.py`.
 *
 * Ported: `test_apply_defaults_rejects_a_default_over_the_limit` and
 * `test_bind_params_rejects_a_filled_default_over_the_limit`.
 *
 * Ported at node level, through `evalNode` (`drive`) with a `ScriptedModel` from `models` where
 * Python runs a loaded flow (T3-22, T4-8): the non-session rows of
 * `test_a_structured_agent_over_the_limit_self_corrects_under_retries`,
 * `test_a_deep_native_reply_that_misses_the_schema_retries_on_the_schema_error`,
 * `test_an_exhausted_deep_native_schema_miss_is_located_at_the_field` and
 * `test_a_structured_agent_over_the_limit_without_retries_fails`.
 *
 * Changed: a structured reply still invalid once `retries:` are spent fails with code
 * `agent.output_invalid` (Python: `node.execute`). Both strategies check a reply's nesting before
 * its declared types, where Python's native strategy has the provider validate the schema first:
 * so `test_a_deep_native_reply_that_misses_the_schema_retries_on_the_schema_error` is corrected
 * naming the nesting under `t` (not `n`), and `test_an_exhausted_deep_native_schema_miss_is_located_at_the_field`
 * keeps the node origin (the nesting error has no path into the output). The agent self-correct rows run at levels
 * 101 and 1500 on both strategies, so they also stand for
 * `test_a_fallback_structured_value_over_the_limit_self_corrects` and
 * `test_a_native_structured_value_over_the_limit_self_corrects` (Python calls
 * `generate_structured` directly). `test_an_exhausted_fallback_structured_value_over_the_limit_is_located`:
 * TS's nesting error has no path into the output (see `structured_retry.test.ts`), so the exhausted
 * failure keeps the node origin, as the without-retries rows check.
 *
 * Ported the same way, the AGENT tool loop's argument checks: the `session=False` rows of
 * `test_tool_arguments_over_the_limit_fail_the_agent_before_any_call` and
 * `test_tool_arguments_over_the_limit_in_a_later_call_stop_every_call`, and the run half of
 * `test_tool_arguments_at_the_limit_run_and_are_learned_from`. A parsed `ToolCallPart` refuses
 * arguments past the limit, so an over-deep call is unparsed (`arguments: null` with its
 * `rawArguments`, or `ToolCallDelta`s the assembler leaves unparsed), as a provider's reply gives
 * it; the loop parses and checks it the same way. From `test_agent_nonfinite_replies.py`, their
 * depth analogs: `test_a_nonfinite_argument_in_a_later_call_stops_every_call_of_the_reply` (a
 * too-deep later call, streamed: no tool runs) and `test_live_events_show_the_model_call_but_no_tool_call`
 * (model brackets, no tool brackets). The non-finite rows of `test_agent_nonfinite_replies.py`
 * are Not ported: plain JSON replies and arguments can't hold `NaN` or `Infinity`.
 *
 * TS-only: an over-deep call on a final reply (text and no parsed call) fails it too.
 *
 * Later (T6-6), session memory: the session rows of the agent tests above,
 * `test_a_native_session_value_at_the_limit_is_stored` and
 * `test_a_native_session_value_over_the_limit_is_not_stored_and_self_corrects`; the learning half
 * of `test_tool_arguments_at_the_limit_run_and_are_learned_from`.
 *
 * Later: the other 27 tests run flows, backward or loads and land with the runner (T4-8, T3-22),
 * AGENT backward (T6-6) and the node kinds.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { ModelCallFinished, ToolCallStarted } from "../../src/events.ts";
import { Finish, ToolCallDelta } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { applyDefaults, BindingError, bindParams } from "../../src/nodes/binding.ts";
import { ContractOrigins, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, type Value, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { drive } from "./run_state_fixtures.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

/** Return `"leaf"` wrapped in `levels` records under `a`. */
function nestRecord(levels: number): Value {
  let value: Value = "leaf";
  for (let i = 0; i < levels; i++) value = { a: value };
  return value;
}

/** Return the `BindingError` `call` throws. */
function bindingError(call: () => unknown): BindingError {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(BindingError);
    return error as BindingError;
  }
  throw new Error("call did not throw");
}

test("applyDefaults rejects a default over the limit", () => {
  const error = bindingError(() =>
    applyDefaults([new Parameter("p", { hasDefault: true, default: nestRecord(101) })], {}),
  );
  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.message.startsWith("default for 'p' at a.a")).toBe(true);
  expect(error.message).toContain("deeper than 100 levels");
});

test("bindParams rejects a filled default over the limit", () => {
  const parameter = new Parameter("p", { hasDefault: true, default: nestRecord(101) });

  const error = bindingError(() => bindParams([parameter], new Map(), new VariablePool()));

  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.message.startsWith("default for 'p' at a.a")).toBe(true);
  expect(error.message).toContain("deeper than 100 levels");
});

// A structured AGENT reply over the limit self-corrects under `retries:`, and fails once they are
// spent.

const NATIVE = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
const FREEFORM = Type.scalar(ValueKind.OBJECT);
/** `{t: object, n: int}`. */
const T_AND_N = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["t", Type.scalar(ValueKind.OBJECT)],
    ["n", Type.scalar(ValueKind.INTEGER)],
  ]),
  required: new Set(["t", "n"]),
});

/** Return JSON text nesting `"leaf"` `levels` records deep under `a`, built without recursion. */
function deepText(levels: number): string {
  return '{"a": '.repeat(levels) + '"leaf"' + "}".repeat(levels);
}

/** A model over `config` replying each of `texts` in turn. */
function texts(config: Record<string, string>, ...replies: string[]): ScriptedModel {
  return new ScriptedModel(
    replies.map(
      (text) =>
        new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" }),
    ),
    config,
  );
}

/** Evaluate an AGENT `agent` with output `output` and `retries` over `model`; return its outcome. */
function runAgent(model: ScriptedModel, output: Type, retries: number, origins?: ContractOrigins) {
  const llmConfig = { provider: model.provider, model: model.model };
  const node = new AgentNode("agent", { prompt: "q", llmConfig, retries });
  node.origin = new OriginRef("node");
  node.outputContract = new ValueContract(output, origins);
  return drive(node, { models: { env: () => undefined, createModel: () => model } });
}

/** The correction the model's second request ends with. */
function correction(model: ScriptedModel): string {
  return model.requests[1]!.messages.at(-1)!.text();
}

/** Expect `message` to name `where`, a path into the value, and the limit. */
function expectReplyTooDeep(message: string, where: string): void {
  expect(message).toContain(`${where} at `);
  expect(message).toContain("deeper than 100 levels");
  expect(message).not.toContain("recursion");
}

const STRATEGIES = [
  ["native", NATIVE],
  ["fallback", PROMPTED],
] as const;
const ROWS = STRATEGIES.flatMap(([name, config]) => [101, 1500].map((levels) => [name, config, levels] as const));

test.each(ROWS)("a structured agent over the limit self-corrects under retries (%s, %d levels)", async (_, config, levels) => {
  const model = texts(config, `{"value": ${deepText(levels)}}`, '{"value": {"k": 1}}');

  expect(await runAgent(model, FREEFORM, 1)).toEqual(new Output({ k: 1n }));
  expect(model.requests).toHaveLength(2);
  expectReplyTooDeep(correction(model), "structured output");
  expect(correction(model)).not.toContain("provider error (");
});

test.each([125, 1500])("a deep native reply that misses the schema retries (%d levels)", async (levels) => {
  const model = texts(NATIVE, `{"t": ${deepText(levels)}, "n": "notint"}`, '{"t": {}, "n": 1}');

  expect(await runAgent(model, T_AND_N, 1)).toEqual(new Output({ t: {}, n: 1n }));
  // The nesting is checked before the declared types, so the correction names it, not `n`.
  expectReplyTooDeep(correction(model), "structured output");
  expect(correction(model)).toContain("structured output at t.a.a");
  expect(correction(model)).not.toContain("provider error (");
});

test.each([125, 1500])("an exhausted deep native schema miss keeps the node origin (%d levels)", async (levels) => {
  const origins = new ContractOrigins({ default: new OriginRef("output"), entries: [[["n"], new OriginRef("output/n")]] });
  const model = texts(NATIVE, `{"t": ${deepText(levels)}, "n": "notint"}`);

  const outcome = await runAgent(model, T_AND_N, 0, origins);

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = outcome as EvaluationFailure;
  expect(failure.code).toBe("agent.output_invalid");
  expect(failure.trace).toEqual([new OriginRef("node")]);
});

test.each(ROWS)("a structured agent over the limit without retries fails (%s, %d levels)", async (_, config, levels) => {
  const model = texts(config, `{"value": ${deepText(levels)}}`);

  const outcome = await runAgent(model, FREEFORM, 0, new ContractOrigins({ default: new OriginRef("output") }));

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = outcome as EvaluationFailure;
  expect([failure.code, failure.nodeId]).toEqual(["agent.output_invalid", "agent"]);
  expectReplyTooDeep(failure.message, "structured output");
  expect(failure.trace).toEqual([new OriginRef("node")]);
});

// ---------- the tool loop's argument checks ----------

/** A run with an AGENT `agent` granted a `lookup` tool that records its arguments. */
function toolRun(model: ScriptedModel, options: { emitEvent?: (event: unknown) => void } = {}) {
  const calls: unknown[] = [];
  const lookup = new FunctionTool({
    name: "lookup",
    description: "Look one thing up.",
    parametersJsonSchema: { type: "object", properties: { x: {}, q: {} } },
    execute: (args) => {
      calls.push(args);
      return "ok";
    },
  });
  const node = new AgentNode("agent", { prompt: "q", llmConfig: NATIVE, tools: ["lookup"] });
  const outcome = drive(node, {
    models: { env: () => undefined, createModel: () => model },
    tools: new ToolSet({ lookup }),
    emitEvent: options.emitEvent ?? null,
  });
  return { outcome, calls };
}

/** An unparsed `lookup` call whose arguments `{"x": ...}` nest `levels` levels. */
function deepCall(levels: number, id = "c2"): ToolCallPart {
  return new ToolCallPart({ id, name: "lookup", arguments: null, rawArguments: `{"x": ${deepText(levels - 1)}}` });
}

/** A reply holding `parts`. */
function partsReply(...parts: (TextPart | ToolCallPart)[]): Response {
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/** The failure `outcome` carries; fails the test when it isn't one. */
function failureOf(outcome: unknown) {
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  return (outcome as EvaluationFailure).failure;
}

test.each([101, 1500])("tool arguments over the limit fail the agent before any call (%d levels)", async (levels) => {
  const model = new ScriptedModel([partsReply(deepCall(levels))], NATIVE);
  const { outcome, calls } = toolRun(model);

  const failure = failureOf(await outcome);
  expect([failure.code, failure.nodeId]).toEqual(["node.execute", "agent"]);
  expectReplyTooDeep(failure.message, "tool call 'lookup' arguments");
  expect(calls).toEqual([]);
  expect(model.requests).toHaveLength(1);
});

test("tool arguments over the limit in a later call stop every call", async () => {
  const first = new ToolCallPart({ id: "c1", name: "lookup", arguments: { x: 1.5 } });
  const { outcome, calls } = toolRun(new ScriptedModel([partsReply(first, deepCall(101))], NATIVE));

  const failure = failureOf(await outcome);
  expect(failure.code).toBe("node.execute");
  expect(failure.message).toContain("deeper than 100 levels");
  expect(calls).toEqual([]);
});

test("tool arguments at the limit run", async () => {
  const atLimit = nestRecord(99);
  const call = new ToolCallPart({ id: "1", name: "lookup", arguments: { q: atLimit } as JsonObject });
  const { outcome, calls } = toolRun(new ScriptedModel([partsReply(call), partsReply(new TextPart("finished"))], NATIVE));

  expect(await outcome).toEqual(new Output("finished"));
  expect(calls).toEqual([{ q: atLimit }]);
});

test("a too-deep later call of a streamed reply stops every call of the reply", async () => {
  const deep = `{"x": ${deepText(100)}}`;
  const events = [
    new ToolCallDelta({ index: 0, id: "c1", name: "lookup", arguments: '{"x": 1.5}' }),
    new ToolCallDelta({ index: 1, id: "c2", name: "lookup", arguments: deep.slice(0, 50) }),
    new ToolCallDelta({ index: 1, id: null, name: null, arguments: deep.slice(50) }),
    new Finish({ reason: "tool_calls" }),
  ];
  const { outcome, calls } = toolRun(new ScriptedModel([events], NATIVE));

  const failure = failureOf(await outcome);
  expect(failure.message).toContain("tool call 'lookup' arguments at x.a");
  expect(failure.message).toContain("deeper than 100 levels");
  expect(calls).toEqual([]);
});

test("live events show the model call but no tool call", async () => {
  const observed: unknown[] = [];
  const { outcome } = toolRun(new ScriptedModel([partsReply(deepCall(101))], NATIVE), {
    emitEvent: (event) => void observed.push(event),
  });

  failureOf(await outcome);
  const finished = observed.filter((event) => event instanceof ModelCallFinished);
  expect(finished).toHaveLength(1);
  expect(finished.every((event) => event.error === null)).toBe(true);
  expect(observed.some((event) => event instanceof ToolCallStarted)).toBe(false);
});

test("an over-deep call on a final reply fails it too", async () => {
  const model = new ScriptedModel([partsReply(new TextPart("the answer"), deepCall(101))], NATIVE);
  const { outcome, calls } = toolRun(model);

  expectReplyTooDeep(failureOf(await outcome).message, "tool call 'lookup' arguments");
  expect(calls).toEqual([]);
});
