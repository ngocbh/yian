/**
 * The AGENT node and its helpers: mirrors `tests/engine/test_agent.py`.
 *
 * Ported: `test_direct_agent_rejects_invalid_variable_declarations` (9 of its 10 rows; the tuple
 * row is not ported, an array is the only list),
 * `test_direct_agent_rejects_duplicate_variables_and_preserves_valid_default_payloads`,
 * `test_direct_agent_rejects_forged_variable_default_presence`,
 * `test_direct_agent_requires_an_exact_variable_contract`,
 * `test_env_max_tool_iterations_minus_one_accepted`, `test_env_max_tool_iterations_rejects_bad_value`
 * (the `maxToolIterations()` half and the `evalNode` half), `test_no_tools_returns_plain_text`,
 * the structured-origin tests (`test_structured_failure_selects_exact_or_ancestor_output_origin`,
 * `test_structured_scalar_failure_selects_contract_root`,
 * `test_non_validation_structured_failure_keeps_node_origin`,
 * `test_source_free_structured_failure_keeps_empty_trace`), and the `provider.auth` tests
 * (`test_model_auth_failure_is_provider_auth`, `test_permission_denied_is_provider_auth`,
 * `test_credential_resolution_auth_error_is_provider_auth`,
 * `test_structured_call_auth_failure_is_provider_auth`,
 * `test_fallback_structured_auth_failure_is_not_retried`,
 * `test_a_stop_during_a_refused_structured_call_stops_the_node` and
 * `test_provider_auth_failure_message_is_scrubbed`).
 *
 * Changed: each row pins its error class; ints are `bigint`s; a forged `hasDefault`/`required` is
 * set with `Object.defineProperty` on a copy, since a `Parameter` is frozen. The node rows run
 * through `evalNode` (`drive`) with a `ScriptedModel` served by `caps.models` where Python patches
 * `model_from_config`; Python's SDK error stand-ins are `ProviderError`s, so their `causeType` reads
 * `ProviderError`. A structured reply that fails the value check once `retries:` are spent is an
 * `OutputInvalidError` coded `agent.output_invalid` (Python: a `ValidationError`,
 * `node.execute`). `test_rate_limit_stays_node_execute`: a marked rate limit is
 * `provider.rate_limit` (with `modelRetries: 0`). `test_provider_auth_failure_message_is_scrubbed`
 * runs the node with the key in its `llmConfig` rather than `run_flow` (T4-8/T3-22).
 *
 * TS-only: `textOf` of a string, of a message with text, reasoning, tool-call and refusal parts
 * (only the text, joined with `"\n"`), and of a message without text; `ReplyInterrupted`'s message,
 * `reply`, `usage` (default `null`) and that it is an `Error`; `recordModelText`, `recordModelValue`
 * and `recordToolResult` (an interaction recorded, a carrier construction error ignored, a `null`
 * trace does nothing); the `DEFAULT_SYSTEM` text and the loop errors' names; `isTextOutput`;
 * `isMaxToolIterations`; the default `maxToolIterations()`; `grantedTools()`; an `LLMConfig` given
 * as `llmConfig`; `backward` not ported yet; the load-time hooks
 * (`reservedWiringKeys`, `modelHiddenInputs`, `reservedKeyLabel`, `llmNullable`,
 * `recordedLlmConfig`, `boundLlmConfig`, `backwardInputNames`, `bindReserved`, `validatePrompts`,
 * `validateExecutionSetup`) and `needsSteps` following `callTime`, also on a shallow copy;
 * `needsModels`; each `provider.*` code through the node (a `ProviderError` of the kind
 * `kindForStatus` reads from a status and message, the message kept); a call-time agent records
 * a copy of its bound config and calls the model with the host key joined, and a bound config that
 * differs from the recorded one is a host fault before any model is built; a reply calling a tool
 * at a cap of one model turn fails with `agent.max_tool_iterations`; without `models` the node
 * builds through `defaultModelAccess` (an unknown provider's `ValidationError`, no network); a
 * failing credentials read doesn't replace the error being scrubbed.
 *
 * Later (T5-11b), the tool loop: `test_tool_loop_round_trip`,
 * `test_tool_loop_executes_model_calls_in_returned_order`,
 * `test_agent_rejects_globally_registered_but_ungranted_tool`,
 * `test_tool_loop_rejects_model_injected_arguments`,
 * `test_tool_loop_propagates_tool_exception_as_node_failure`,
 * `test_tool_validation_error_does_not_claim_scalar_output_origin`, the cap tests
 * (`test_tool_loop_cap_trips_when_bounded`, `test_default_no_cap_runs_unbounded`,
 * `test_env_max_tool_iterations_overrides_cap`, `test_tool_loop_cap_does_not_run_the_last_replys_tools`,
 * `test_ungranted_tool_stays_node_execute`, `test_tool_loop_cap_of_one_still_answers_without_tools`)
 * and `test_tool_auth_failure_stays_node_execute`.
 *
 * Not ported: `test_agent_run_has_no_scratch_kwarg` (signature introspection);
 * `test_agent_ctx_type.py` and `test_agent_runcontext.py` (they introspect a Python dataclass; the
 * TS `AgentRunContext` is an interface checked statically); and
 * `compose/test_agent_mode_removed.py`'s `test_python_agent_node_has_no_mode` (Python's `MODES`,
 * `register_mode` and `mode=` keyword have no TS counterpart).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";

import { copyGraph } from "../../src/compile/model.ts";
import { BUILTIN, current, type Env, forget, install } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { TokenUsage } from "../../src/events.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import { AuthError } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { HostConnection, LLMConfig } from "../../src/llm_clients/config.ts";
import { ConnectionSelection, connectionSnapshot, HOST_KEY_MARKER } from "../../src/llm_clients/connection.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { ValidationError } from "../../src/llm_clients/provider_blocks.ts";
import { kindForStatus, ProviderError } from "../../src/llm/errors.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { RouteModel } from "../../src/llm/executor.ts";
import { Message, ReasoningPart, RefusalPart, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { Route } from "../../src/llm/route.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  type EvaluationTrace,
  ModelTextInteraction,
  ModelValueInteraction,
  ToolResultInteraction,
  type VisibleInteraction,
} from "../../src/memory/contracts.ts";
import { AgentLoopError, DEFAULT_SYSTEM, ToolIterationLimitError } from "../../src/nodes/agent/context.ts";
import { ReplyInterrupted } from "../../src/nodes/agent/interrupts.ts";
import { CallTimeLLM, contractNullable, LLM_CONFIG_KEY, splitLayers } from "../../src/nodes/agent/llm_values.ts";
import { AgentNode, isMaxToolIterations, isTextOutput } from "../../src/nodes/agent/node.ts";
import { recordModelText, recordModelValue, recordToolResult, textOf } from "../../src/nodes/agent/text.ts";
import { CapabilityFault, Output } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { Binding, ContractOrigins, NodeSteps, Parameter, ValueContract, ValueTemplate } from "../../src/nodes/contracts.ts";
import { type EvalNodeOptions, EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import { drive, rejected } from "./run_state_fixtures.ts";

test("textOf returns a string as it is", () => {
  expect(textOf("hello\nworld")).toBe("hello\nworld");
  expect(textOf("")).toBe("");
});

test("textOf keeps only a message's text parts, joined with a newline", () => {
  const message = new Message({
    role: "assistant",
    parts: [
      new ReasoningPart("thinking"),
      new TextPart("first"),
      new ToolCallPart({ id: "c1", name: "search", arguments: {} }),
      new RefusalPart("no"),
      new TextPart("second"),
    ],
  });
  expect(textOf(message)).toBe("first\nsecond");
});

test("textOf of a message without text is empty", () => {
  const message = new Message({
    role: "assistant",
    parts: [new ReasoningPart("thinking"), new ToolCallPart({ id: "c1", name: "search", arguments: {} })],
  });
  expect(textOf(message)).toBe("");
});

test("ReplyInterrupted carries the partial reply and the usage so far", () => {
  const reply = new Message({ role: "assistant", parts: [new TextPart("partial")] });
  const usage = new TokenUsage({ inputTokens: 3, outputTokens: 2, totalTokens: 5 });
  const error = new ReplyInterrupted(reply, usage);
  expect(error).toBeInstanceOf(Error);
  expect(error.name).toBe("ReplyInterrupted");
  expect(error.message).toBe("model reply interrupted by the host");
  expect(error.reply).toBe(reply);
  expect(error.usage).toBe(usage);
});

test("ReplyInterrupted's usage defaults to null", () => {
  const reply = new Message({ role: "assistant", parts: [] });
  expect(new ReplyInterrupted(reply).usage).toBeNull();
});

/** A trace that keeps the interactions it is given. */
function trace() {
  const interactions: VisibleInteraction[] = [];
  const sink: EvaluationTrace = {
    recordPrompt: () => {},
    recordInteraction: (interaction) => void interactions.push(interaction),
  };
  return { sink, interactions };
}

test("recordModelText and recordToolResult record their interactions", () => {
  const { sink, interactions } = trace();
  recordModelText(sink, "answer");
  recordToolResult(sink, "lookup", "found");
  expect(interactions).toEqual([new ModelTextInteraction("answer"), new ToolResultInteraction("lookup", "found")]);
});

test("recordModelText and recordToolResult ignore a carrier construction error", () => {
  const { sink, interactions } = trace();
  recordModelText(sink, "\ud800");
  recordToolResult(sink, "", "found");
  recordToolResult(sink, "lookup", 3 as unknown as string);
  expect(interactions).toEqual([]);
});

test("recordModelText and recordToolResult do nothing without a trace", () => {
  expect(recordModelText(null, "answer")).toBeUndefined();
  expect(recordToolResult(undefined, "lookup", "found")).toBeUndefined();
});

test("recordModelValue records the value, ignores a carrier construction error, and does nothing without a trace", () => {
  const { sink, interactions } = trace();
  recordModelValue(sink, { a: 1n });
  recordModelValue(sink, Number.NaN);
  expect(interactions).toEqual([new ModelValueInteraction({ a: 1n })]);
  expect(recordModelValue(null, 1n)).toBeUndefined();
});

test("DEFAULT_SYSTEM keeps Python's text and the loop errors are named", () => {
  expect(DEFAULT_SYSTEM).toBe(
    "You are an analysis agent inside a larger workflow. Follow the instructions " +
      "precisely, use the available tools to gather any facts you need, and return " +
      "a clear, self-contained answer. Ask the user only through an explicitly granted tool.",
  );
  const limit = new ToolIterationLimitError("cap");
  expect(limit).toBeInstanceOf(AgentLoopError);
  expect(limit).toBeInstanceOf(Error);
  expect([limit.name, new AgentLoopError("x").name]).toEqual(["ToolIterationLimitError", "AgentLoopError"]);
});

/** Return what `fn` throws; fails the test when it returns. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

const INT = () => new ValueContract(Type.scalar(ValueKind.INTEGER));

test.each([
  ["a record", () => ({}) as unknown as Parameter[], TypeError],
  ["a plain object item", () => [{}] as unknown as Parameter[], TypeError],
  ["no contract", () => [new Parameter("choice")], TypeError],
  ["a default but no contract", () => [new Parameter("choice", { hasDefault: true, default: 1n })], TypeError],
  ["an empty name", () => [new Parameter("", { contract: INT(), hasDefault: true, default: 1n })], TypeError],
  [
    "a non-string name",
    () => [new Parameter(1 as unknown as string, { contract: INT(), hasDefault: true, default: 1n })],
    TypeError,
  ],
  ["a bad default", () => [new Parameter("choice", { contract: INT(), hasDefault: true, default: "1" })], TypeCheckError],
  [
    "an out-of-bounds default",
    () => [
      new Parameter("choice", {
        contract: new ValueContract(new Type({ kind: ValueKind.INTEGER, bounds: [1n, 2n] })),
        hasDefault: true,
        default: 3n,
      }),
    ],
    TypeCheckError,
  ],
  [
    "inverted bounds",
    () => [
      new Parameter("choice", {
        contract: new ValueContract(new Type({ kind: ValueKind.INTEGER, bounds: [2n, 1n] })),
        hasDefault: true,
        default: 1n,
      }),
    ],
    RangeError,
  ],
] as const)("direct agent rejects invalid variable declarations: %s", (_label, variables, errorClass) => {
  const error = thrown(() => new AgentNode("writer", { variables: variables() }));
  expect((error as Error).constructor).toBe(errorClass);
});

test("direct agent rejects duplicate variables and preserves valid default payloads", () => {
  const variable = new Parameter("count", {
    contract: new ValueContract(new Type({ kind: ValueKind.NUMBER, bounds: [0n, 4n] })),
    hasDefault: true,
    default: 4n,
  });
  expect(() => new AgentNode("writer", { variables: [variable, variable] })).toThrow(
    new RangeError("duplicate variable name 'count'"),
  );
  const node = new AgentNode("writer", { variables: [variable] });
  expect(node.variables).toEqual([variable]);
  expect(node.variables[0]!.default).toBe(variable.default);
  expect(typeof node.variables[0]!.default).toBe("bigint");
  expect(node.params).toBeNull();
});

test.each([
  ["hasDefault", 1],
  ["required", 0],
  ["required", true],
] as const)("direct agent rejects forged variable default presence: %s = %p", (field, value) => {
  const variable = new Parameter("count", { contract: INT(), hasDefault: true, default: 4n });
  const forged = Object.assign(Object.create(Parameter.prototype), variable) as Parameter;
  Object.defineProperty(forged, field, { value });
  expect(() => new AgentNode("writer", { variables: [forged] })).toThrow(
    new RangeError("variable 'count' requires a default and required=false"),
  );
});

test("direct agent requires an exact variable contract", () => {
  class CustomContract extends ValueContract {}
  const variable = new Parameter("count", {
    contract: new CustomContract(Type.scalar(ValueKind.INTEGER)),
    hasDefault: true,
    default: 4n,
  });
  expect(() => new AgentNode("writer", { variables: [variable] })).toThrow(
    new TypeError("variable 'count' requires an exact ValueContract"),
  );
});

test("env max_tool_iterations minus one accepted", () => {
  const node = new AgentNode("n", { prompt: "hi" });
  node.env = { max_tool_iterations: -1n };
  expect(node.maxToolIterations()).toBe(-1);
});

test("env max_tool_iterations rejects a bad value", () => {
  const node = new AgentNode("n", { prompt: "hi" });
  node.env = { max_tool_iterations: 0n };
  expect(() => node.maxToolIterations()).toThrow(
    new RangeError("AGENT node 'n': env max_tool_iterations must be a positive int or -1 (no cap), got 0"),
  );
});

test("maxToolIterations defaults to the agent max tool iterations default", () => {
  expect(new AgentNode("n").maxToolIterations()).toBe(current().agentMaxToolIterations);
  const node = new AgentNode("n");
  node.env = { max_tool_iterations: 300n };
  expect(node.maxToolIterations()).toBe(300);
});

test("isTextOutput accepts no contract and an untagged str", () => {
  expect(isTextOutput(null)).toBe(true);
  expect(isTextOutput(new ValueContract(Type.scalar(ValueKind.STRING)))).toBe(true);
  const tagged = new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b"]) });
  expect(isTextOutput(new ValueContract(tagged))).toBe(false);
  expect(isTextOutput(new ValueContract(Type.scalar(ValueKind.INTEGER)))).toBe(false);
  class CustomContract extends ValueContract {}
  expect(isTextOutput(new CustomContract(Type.scalar(ValueKind.STRING)))).toBe(false);
});

test("isMaxToolIterations takes a positive bigint or -1n only", () => {
  for (const value of [1n, -1n, 300n]) expect(isMaxToolIterations(value)).toBe(true);
  for (const value of [0n, -2n, 1, 1.0, 1.5, true, "1"]) expect(isMaxToolIterations(value)).toBe(false);
});

test("grantedTools returns a copy of the authored grants", () => {
  const tools = ["search", "fetch"];
  const node = new AgentNode("n", { tools });
  const granted = node.grantedTools();
  expect(granted).toEqual(["search", "fetch"]);
  expect(granted).not.toBe(node.tools);
  expect(new AgentNode("n").grantedTools()).toEqual([]);
});

test("an LLMConfig given as llmConfig is refused", () => {
  const config = new LLMConfig({}) as unknown as Record<string, unknown>;
  expect(() => new AgentNode("n", { llmConfig: config })).toThrow(
    new TypeError("AGENT node 'n': llm_config must be a record, a ValueTemplate, a Binding or null"),
  );
});

test("backward is not ported yet", () => {
  const node = new AgentNode("n");
  expect(() => node.backward()).toThrow(new Error("node 'n' (AgentNode) backward is not ported yet (T6-6)"));
});

// ---------- load-time hooks ----------

const HOME = mkdtempSync(join(tmpdir(), "yian-agent-"));
afterAll(() => rmSync(HOME, { recursive: true, force: true }));
const ENV_VARS = new Map([
  ["YIAN_DEFAULT_PROVIDER", "openai"],
  ["YIAN_DEFAULT_MODEL", "gpt-default"],
  ["OPENAI_API_KEY", "sk-env-secret"],
]);
const env: Env = (name) => ENV_VARS.get(name);
const STR = Type.scalar(ValueKind.STRING);

/** Return what run start fixes for the agent `chat`, with the host key for `hostKeyProvider`. */
function callTime(base: Record<string, unknown> = {}, hostKeyProvider: string | null = null): CallTimeLLM {
  const snapshot = connectionSnapshot({ env, store: new AuthStore(join(HOME, "auth.json"), { env }) });
  const selection = new ConnectionSelection({ snapshot, hostKeyProvider, hostBaseUrl: null, hostSettings: {} });
  return new CallTimeLLM({ path: "chat", base, selection });
}

/** Return a pool in which node `pick` produced `value`. */
function pickPool(value: unknown): VariablePool {
  const pool = new VariablePool();
  pool.set(pool.startId, {});
  pool.set("pick", value);
  return pool;
}

/** Return `chat`'s wiring, whose `__llm_config` carries `nodeLayer` as the node layer. */
function llmWiring(nodeLayer: unknown): ReadonlyMap<string, unknown> {
  return new Map([[LLM_CONFIG_KEY, splitLayers(nodeLayer, {}, true, contractNullable)!]]);
}

test("reservedWiringKeys and modelHiddenInputs hold __llm_config once it is wired", () => {
  const node = new AgentNode("chat");
  expect(node.reservedWiringKeys()).toEqual(new Set());
  expect(node.modelHiddenInputs()).toEqual(new Set());
  node.wiresLlmConfig = true;
  expect(node.reservedWiringKeys()).toEqual(new Set([LLM_CONFIG_KEY]));
  expect(node.modelHiddenInputs()).toEqual(new Set([LLM_CONFIG_KEY]));
});

test("reservedKeyLabel names __llm_config as the authored llm_config field", () => {
  const node = new AgentNode("chat");
  expect(node.reservedKeyLabel(LLM_CONFIG_KEY)).toBe("llm_config");
  expect(node.reservedKeyLabel("other")).toBe("agent other");
});

test("llmNullable answers from llmValueTypes", () => {
  const node = new AgentNode("chat", { llmConfig: new ValueTemplate({ model: "${pick.output}" }) });
  // A value with no proven type may be null.
  expect(node.llmNullable()("${pick.output}", [])).toBe(true);
  node.llmValueTypes = [[["model"], STR]];
  expect(node.llmNullable()("${pick.output}", [])).toBe(false);
  node.llmValueTypes = [[["model"], null]];
  expect(node.llmNullable()("${pick.output}", [])).toBe(true);
});

test("needsSteps follows callTime, also on a shallow copy", () => {
  const node = new AgentNode("chat");
  expect(node.needsSteps).toBe(false);
  const at = callTime();
  node.callTime = at;
  expect(node.callTime).toBe(at);
  expect(node.needsSteps).toBe(true);

  const copy = copyGraph(node);
  expect(copy.callTime).toBe(at);
  expect(copy.needsSteps).toBe(true);
  copy.callTime = null;
  expect(copy.needsSteps).toBe(false);
  expect(node.callTime).toBe(at);
  expect(node.needsSteps).toBe(true);
});

test("recordedLlmConfig returns a copy of the last recorded config", () => {
  const node = new AgentNode("chat");
  const last = { model: "m2", openai: { temperature: 0.5 } };
  const steps = [{ effective_llm_config: { model: "m1" } }, { effective_llm_config: last }, { other: 1n }, "note"];
  const recorded = node.recordedLlmConfig(steps);
  expect(recorded).toEqual(last);
  expect(recorded).not.toBe(last);
  expect(recorded!["openai"]).not.toBe(last.openai);
  // A step that isn't a record, or whose config isn't one, records none.
  expect(node.recordedLlmConfig(["note", { effective_llm_config: "m" }, [1n]])).toBeNull();
  expect(node.recordedLlmConfig([])).toBeNull();
});

test("boundLlmConfig returns the bound record config, else null", () => {
  const node = new AgentNode("chat");
  const config = { provider: "openai", model: "m" };
  expect(node.boundLlmConfig({ [LLM_CONFIG_KEY]: config, topic: "t" })).toBe(config);
  expect(node.boundLlmConfig({ topic: "t" })).toBeNull();
  expect(node.boundLlmConfig({ [LLM_CONFIG_KEY]: "m" })).toBeNull();
});

test("backwardInputNames leaves out __llm_config", () => {
  const node = new AgentNode("chat");
  node.params = [new Parameter("topic"), new Parameter(LLM_CONFIG_KEY), new Parameter("tone")];
  expect(node.backwardInputNames()).toEqual(["topic", "tone"]);
});

test("bindReserved binds nothing for an agent run start resolved", () => {
  expect(new AgentNode("chat").bindReserved(new Map(), pickPool("gpt-x"))).toEqual({});
});

test("bindReserved resolves a call-time agent's config", () => {
  const node = new AgentNode("chat");
  node.callTime = callTime({ provider: "openai" });
  const bound = node.bindReserved(llmWiring(new ValueTemplate({ model: "${pick.output}" })), pickPool("gpt-x"));
  expect(Object.keys(bound)).toEqual([LLM_CONFIG_KEY]);
  const config = bound[LLM_CONFIG_KEY] as Record<string, unknown>;
  expect(config["provider"]).toBe("openai");
  expect(config["model"]).toBe("gpt-x");
  expect(JSON.stringify(config)).not.toContain("sk-env-secret");
});

test("bindReserved names a field by llmConfigLabel and passes the host credentials", () => {
  const node = new AgentNode("chat");
  node.callTime = callTime();
  node.llmConfigLabel = "node 'ask' adaptive_questions.llm_config";
  const typed = new Binding("${pick.output}", { contract: new ValueContract(STR, new ContractOrigins()) });
  const error = thrown(() => node.bindReserved(llmWiring(new ValueTemplate({ model: typed })), pickPool(null)));
  expect(error).toBeInstanceOf(BindingError);
  expect((error as Error).message).toStartWith("node 'ask' adaptive_questions.llm_config.model: ");

  node.llmConfigLabel = null;
  node.callTime = callTime({}, "anthropic");
  node.hostCredentials = new HostConnection({ apiKey: "sk-host" });
  const wiring = llmWiring(new ValueTemplate({ provider: "${pick.output}", model: "claude-x" }));
  expect(node.bindReserved(wiring, pickPool("anthropic"))).toEqual({
    [LLM_CONFIG_KEY]: { provider: "anthropic", model: "claude-x", [HOST_KEY_MARKER]: true },
  });
});

test("bindReserved needs the __llm_config binding", () => {
  const node = new AgentNode("chat");
  node.callTime = callTime();
  const message = "AGENT node 'chat' has no `__llm_config` wiring";
  expect(() => node.bindReserved(new Map(), pickPool(null))).toThrow(new Error(message));
  // A value that isn't a binding is the same setup fault.
  expect(() => node.bindReserved(new Map([[LLM_CONFIG_KEY, { model: "m" }]]), pickPool(null))).toThrow(
    new Error(message),
  );
});

test("validatePrompts names the node and the field of an unparseable span", () => {
  const node = new AgentNode("x", { prompt: "Hi ${name}", systemPrompt: "Be ${" });
  const error = thrown(() => node.validatePrompts());
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toStartWith("node 'x' system_prompt: ");
  expect((thrown(() => node.validatePrompts({ path: "outer/x" })) as Error).message).toStartWith(
    "node 'outer/x' system_prompt: ",
  );
  expect((thrown(() => new AgentNode("x", { prompt: "${" }).validatePrompts()) as Error).message).toStartWith(
    "node 'x' prompt: ",
  );
  expect(new AgentNode("x", { prompt: "Hi ${name}", systemPrompt: "Be ${tone}" }).validatePrompts()).toBeUndefined();
});

test("validateExecutionSetup lets a steerable agent with plain text output pass", () => {
  const node = new AgentNode("chat", { steerable: true });
  expect(node.validateExecutionSetup()).toBeUndefined();
  node.outputContract = new ValueContract(STR);
  expect(node.validateExecutionSetup()).toBeUndefined();
});

// ---------- forward ----------

describe("forward", () => {
  beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
  afterEach(forget);

  const NATIVE = { provider: "openai", model: "gpt-5.5" };
  const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
  const INT = Type.scalar(ValueKind.INTEGER);

  function reply(text: string): Response {
    return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" });
  }

  /** A model access over an empty environment that serves `model` and keeps each config it is given. */
  function access(model: ChatModel | (() => ChatModel)): ModelAccess & { configs: Record<string, unknown>[] } {
    const configs: Record<string, unknown>[] = [];
    return {
      configs,
      env: () => undefined,
      createModel: (config) => {
        configs.push({ ...config });
        return typeof model === "function" ? model() : model;
      },
    };
  }

  /** Evaluate `node` with `model`; return its raw outcome. */
  function evaluate(node: AgentNode, model: ChatModel | (() => ChatModel), options: EvalNodeOptions = {}) {
    return drive(node, { models: access(model), ...options });
  }

  /** The failure `outcome` carries; fails the test when it isn't one. */
  function failureOf(outcome: unknown) {
    expect(outcome).toBeInstanceOf(EvaluationFailure);
    return (outcome as EvaluationFailure).failure;
  }

  /** An AGENT `n` with prompt `hi` over `config`. */
  function agent(config: Record<string, unknown> = NATIVE, options: { retries?: number } = {}): AgentNode {
    return new AgentNode("n", { prompt: "hi", llmConfig: config, ...options });
  }

  /** An AGENT with an integer `output:` and no tools, over `config` (Python's `_structured_node`). */
  function structuredAgent(config: Record<string, unknown>): AgentNode {
    const node = agent(config, { retries: 2 });
    node.outputContract = new ValueContract(INT);
    return node;
  }

  /** A model whose call aborts `stop`, then throws `error`, as a provider failing while a stop is pending. */
  function stopThenThrow(stop: AbortController, error: unknown, config: Record<string, unknown>): ChatModel {
    return {
      ...(config as { provider: string; model: string }),
      stream: () =>
        new EventStream(
          (async function* (): AsyncGenerator<Event> {
            stop.abort(new Error("stopped"));
            throw error;
          })(),
        ),
    };
  }

  test("an AGENT declares needsModels", () => {
    expect(new AgentNode("n").needsModels).toBe(true);
  });

  test("no tools returns plain text", async () => {
    const model = new ScriptedModel([reply("answer")], NATIVE);
    const outcome = await evaluate(agent(), model);
    expect((outcome as Output).value).toBe("answer");
    expect(model.requests).toHaveLength(1);
  });

  test("env max_tool_iterations rejects a bad value as a node failure", async () => {
    const node = agent();
    node.env = { max_tool_iterations: 0n };
    const failure = failureOf(await evaluate(node, new ScriptedModel([reply("unused")], NATIVE)));
    expect(failure.message).toContain("max_tool_iterations");
  });

  const PROFILE = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["profile", new Type({ kind: ValueKind.OBJECT, fields: new Map([["age", INT]]), required: new Set(["age"]) })],
    ]),
    required: new Set(["profile"]),
  });

  test.each([
    [["profile", "age"], "output/profile/age"],
    [["profile"], "output/profile"],
  ])("structured failure selects the exact or ancestor output origin (%p)", async (path, key) => {
    const node = agent(NATIVE, { retries: 0 });
    node.origin = new OriginRef("node");
    node.outputContract = new ValueContract(
      PROFILE,
      new ContractOrigins({ default: new OriginRef("output"), entries: [[path, new OriginRef(key)]] }),
    );
    const model = new ScriptedModel([reply('{"profile": {"age": "bad"}}')], NATIVE);
    const failure = failureOf(await evaluate(node, model));
    expect([failure.code, failure.causeType]).toEqual(["agent.output_invalid", "OutputInvalidError"]);
    expect(failure.trace).toEqual([new OriginRef(key)]);
  });

  test.each([
    ["an int", INT, '{"value": "bad"}'],
    ["a tagged str", new Type({ kind: ValueKind.STRING, tags: new Set(["positive", "negative"]) }), '{"value": "not-a-tag"}'],
  ])("structured scalar failure selects the contract root (%s)", async (_name, typ, payload) => {
    const node = agent(NATIVE, { retries: 0 });
    node.origin = new OriginRef("node");
    node.outputContract = new ValueContract(typ, new ContractOrigins({ default: new OriginRef("output") }));
    const failure = failureOf(await evaluate(node, new ScriptedModel([reply(payload)], NATIVE)));
    expect(failure.trace).toEqual([new OriginRef("output")]);
  });

  test("a non-validation structured failure keeps the node origin", async () => {
    const node = agent(NATIVE, { retries: 0 });
    node.origin = new OriginRef("node");
    node.outputContract = new ValueContract(INT, new ContractOrigins({ default: new OriginRef("output") }));
    const failure = failureOf(await evaluate(node, new ScriptedModel([new Error("provider failed")], NATIVE)));
    expect([failure.code, failure.causeType]).toEqual(["node.execute", "Error"]);
    expect(failure.trace).toEqual([new OriginRef("node")]);
  });

  test("a source-free structured failure keeps an empty trace", async () => {
    const node = agent(NATIVE, { retries: 0 });
    node.outputContract = new ValueContract(INT);
    const failure = failureOf(await evaluate(node, new ScriptedModel([reply('{"value": "bad"}')], NATIVE)));
    expect(failure.trace).toEqual([]);
  });

  test("a model auth failure is provider.auth", async () => {
    const error = new ProviderError("auth", "Incorrect API key provided", { status: 401 });
    const failure = failureOf(await evaluate(agent(), new ScriptedModel([error], NATIVE)));
    expect([failure.code, failure.causeType, failure.message]).toEqual([
      "provider.auth",
      "ProviderError",
      "Incorrect API key provided",
    ]);
  });

  test("a permission denial is provider.auth", async () => {
    const error = new ProviderError("permission", "forbidden", { status: 403 });
    expect(failureOf(await evaluate(agent(), new ScriptedModel([error], NATIVE))).code).toBe("provider.auth");
  });

  test("a rate limit is provider.rate_limit (Changed)", async () => {
    const error = new ProviderError("rate_limit", "slow down", { status: 429 });
    expect(failureOf(await evaluate(agent(), new ScriptedModel([error], NATIVE))).code).toBe("provider.rate_limit");
  });

  test.each([
    ["provider.rate_limit", 429, "Rate limit reached for gpt-5.5"],
    ["provider.quota", 429, "You exceeded your current quota (insufficient_quota)"],
    ["provider.unavailable", 503, "The server is temporarily unavailable"],
    ["provider.context_length", 400, "This model's maximum context length is 128000 tokens (context_length_exceeded)"],
    ["provider.refused", 400, '{"error": {"code": "content_filter", "message": "The response was filtered"}}'],
  ])("%s through the node", async (code, status, message) => {
    const error = new ProviderError(kindForStatus(status, message), message, { status });
    const failure = failureOf(await evaluate(agent(), new ScriptedModel([error], NATIVE)));
    expect([failure.code, failure.message]).toEqual([code, message]);
  });

  test("a credential resolution AuthError is provider.auth", async () => {
    const message = "Subscription credentials expired; sign in again. Run 'yian auth openai' and start a new run";
    const failure = failureOf(
      await evaluate(agent(), () => {
        throw new AuthError(message);
      }),
    );
    expect([failure.code, failure.message]).toEqual(["provider.auth", message]);
  });

  test("a structured call auth failure is provider.auth and not retried", async () => {
    const model = new ScriptedModel([new ProviderError("auth", "unauthorized", { status: 401 })], NATIVE);
    expect(failureOf(await evaluate(structuredAgent(NATIVE), model)).code).toBe("provider.auth");
    // Refused credentials are not corrected and retried.
    expect(model.requests).toHaveLength(1);
  });

  test("a fallback structured auth failure is not retried", async () => {
    const model = new ScriptedModel([new ProviderError("auth", "unauthorized", { status: 401 })], PROMPTED);
    expect(failureOf(await evaluate(structuredAgent(PROMPTED), model)).code).toBe("provider.auth");
    expect(model.requests).toHaveLength(1);
  });

  test.each([
    ["native", NATIVE],
    ["fallback", PROMPTED],
  ])("a stop during a refused structured call stops the node (%s)", async (_name, config) => {
    // A pending stop wins over refused credentials, as it does over any provider error.
    const stop = new AbortController();
    const model = stopThenThrow(stop, new ProviderError("auth", "unauthorized", { status: 401 }), config);
    const error = await rejected(evaluate(structuredAgent(config), model, { signal: stop.signal }));
    expect(error).toBe(stop.signal.reason);
  });

  test("a provider auth failure message is scrubbed", async () => {
    const key = "sk-sentinel-401-zqxw";
    const config = { provider: "vllm", model: "m", api_key: key };
    const model = new ScriptedModel([new ProviderError("auth", `Incorrect API key: ${key}`, { status: 401 })], config);
    const failure = failureOf(await evaluate(agent(config), model));
    expect(failure.code).toBe("provider.auth");
    expect(failure.message).toBe("Incorrect API key: [REDACTED]");
  });

  test("a reply calling a tool at a cap of one model turn is agent.max_tool_iterations", async () => {
    const node = agent();
    node.env = { max_tool_iterations: 1n };
    const call = new ToolCallPart({ id: "c1", name: "lookup", arguments: { q: "x" } });
    const message = new Message({ role: "assistant", parts: [call] });
    const model = new ScriptedModel([new Response({ message, finishReason: "tool_calls" })], NATIVE);
    const failure = failureOf(await evaluate(node, model));
    expect([failure.code, failure.causeType]).toEqual(["agent.max_tool_iterations", "ToolIterationLimitError"]);
  });

  test("without models the node builds through defaultModelAccess", async () => {
    // An unknown provider fails in the factory's config check, before any network.
    const node = new AgentNode("n", { prompt: "hi", llmConfig: { provider: "no-such-provider", model: "m" } });
    const error = await rejected(node.run({}, {}));
    expect(error).toBeInstanceOf(ValidationError);
    expect((error as Error).message).toStartWith("1 validation error for LLMConfig");
  });

  test("a failing credentials read doesn't replace the error being scrubbed", async () => {
    const auth = {
      headers: async () => ({}),
      credentials: (): string[] => {
        throw new Error("read");
      },
    };
    const route = new Route({ provider: "openai", baseUrl: "https://api.example", path: "x", auth });
    class RefusedModel extends RouteModel {
      override stream(): EventStream {
        return new EventStream(
          (async function* (): AsyncGenerator<Event> {
            throw new ProviderError("auth", "unauthorized", { status: 401 });
          })(),
        );
      }
    }
    const model = new RefusedModel({ route, protocol: new OpenAIChat({ model: "gpt-5.5", provider: "openai" }) });
    const failure = failureOf(await evaluate(agent(), model));
    expect([failure.code, failure.message]).toEqual(["provider.auth", "unauthorized"]);
  });

  /** A call-time agent `chat` whose bound config is marked for the host key `sk-host`. */
  function callTimeAgent(): AgentNode {
    const node = new AgentNode("chat", { prompt: "hi" });
    node.callTime = callTime({}, "openai");
    node.hostCredentials = new HostConnection({ apiKey: "sk-host" });
    return node;
  }

  const BOUND = { provider: "openai", model: "gpt-5.5", [HOST_KEY_MARKER]: true };

  test("a call-time agent records a copy of its bound config and calls with the host key", async () => {
    const recorded: unknown[] = [];
    const steps = new NodeSteps(1, [], (step) => void recorded.push(step));
    const models = access(new ScriptedModel([reply("ok")], NATIVE));
    const bound = structuredClone(BOUND);
    const result = await callTimeAgent().run({ [LLM_CONFIG_KEY]: bound }, { steps, models });
    expect((result as Output).value).toBe("ok");
    expect(recorded).toEqual([{ effective_llm_config: BOUND }]);
    expect((recorded[0] as Record<string, unknown>)["effective_llm_config"]).not.toBe(bound);
    expect(models.configs).toEqual([{ ...BOUND, api_key: "sk-host" }]);
  });

  test("a call-time agent's config differing from the recorded one is a host fault before any model is built", async () => {
    const recorded: unknown[] = [];
    const prior = [{ effective_llm_config: { ...BOUND, model: "gpt-older" } }];
    const steps = new NodeSteps(2, prior, (step) => void recorded.push(step));
    const models = access(new ScriptedModel([reply("ok")], NATIVE));
    const error = await rejected(callTimeAgent().run({ [LLM_CONFIG_KEY]: { ...BOUND } }, { steps, models }));
    expect(error).toBeInstanceOf(CapabilityFault);
    expect((error as CapabilityFault).original).toEqual(
      new RangeError(
        "node 'chat' llm_config: recorded config differs from the recomputed one; resume the run with the flow " +
          "and yian version that started it",
      ),
    );
    expect(recorded).toEqual([]);
    expect(models.configs).toEqual([]);
    // An equal config, even one with its keys in another order, records again.
    const same = { model: "gpt-older", [HOST_KEY_MARKER]: true, provider: "openai" };
    await callTimeAgent().run({ [LLM_CONFIG_KEY]: same }, { steps, models });
    expect(recorded).toEqual([{ effective_llm_config: same }]);
  });
});
