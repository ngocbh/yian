/**
 * The AGENT node's helpers: mirrors `tests/engine/test_agent.py`.
 *
 * Ported: `test_direct_agent_rejects_invalid_variable_declarations` (9 of its 10 rows; the tuple
 * row is not ported, an array is the only list),
 * `test_direct_agent_rejects_duplicate_variables_and_preserves_valid_default_payloads`,
 * `test_direct_agent_rejects_forged_variable_default_presence`,
 * `test_direct_agent_requires_an_exact_variable_contract`,
 * `test_env_max_tool_iterations_minus_one_accepted`, and the `maxToolIterations()` half of
 * `test_env_max_tool_iterations_rejects_bad_value` (its `eval_node` half comes with `forward`,
 * T4/T5-11a).
 *
 * Changed: each row pins its error class; ints are `bigint`s; a forged `hasDefault`/`required` is
 * set with `Object.defineProperty` on a copy, since a `Parameter` is frozen.
 *
 * TS-only: `textOf` of a string, of a message with text, reasoning, tool-call and refusal parts
 * (only the text, joined with `"\n"`), and of a message without text; `ReplyInterrupted`'s message,
 * `reply`, `usage` (default `null`) and that it is an `Error`; `isTextOutput`;
 * `isMaxToolIterations`; the default `maxToolIterations()`; `grantedTools()`; an `LLMConfig` given
 * as `llmConfig`; `forward` and `backward` not ported yet; the load-time hooks
 * (`reservedWiringKeys`, `modelHiddenInputs`, `reservedKeyLabel`, `llmNullable`,
 * `recordedLlmConfig`, `boundLlmConfig`, `backwardInputNames`, `bindReserved`, `validatePrompts`,
 * `validateExecutionSetup`) and `needsSteps` following `callTime`, also on a shallow copy.
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
 * Later (T5-11a): `test_no_tools_returns_plain_text`; the structured-origin tests
 * (`test_structured_failure_selects_exact_or_ancestor_output_origin`,
 * `test_structured_scalar_failure_selects_contract_root`,
 * `test_non_validation_structured_failure_keeps_node_origin`,
 * `test_source_free_structured_failure_keeps_empty_trace`); the `provider.auth` tests
 * (`test_model_auth_failure_is_provider_auth`, `test_permission_denied_is_provider_auth`,
 * `test_rate_limit_stays_node_execute`, `test_credential_resolution_auth_error_is_provider_auth`,
 * `test_structured_call_auth_failure_is_provider_auth`,
 * `test_fallback_structured_auth_failure_is_not_retried`,
 * `test_a_stop_during_a_refused_structured_call_stops_the_node`,
 * `test_provider_auth_failure_message_is_scrubbed`).
 *
 * Not ported: `test_agent_run_has_no_scratch_kwarg` (signature introspection).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { current, type Env } from "../../src/defaults.ts";
import { TokenUsage } from "../../src/events.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { HostConnection, LLMConfig } from "../../src/llm_clients/config.ts";
import { ConnectionSelection, connectionSnapshot, HOST_KEY_MARKER } from "../../src/llm_clients/connection.ts";
import { Message, ReasoningPart, RefusalPart, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ReplyInterrupted } from "../../src/nodes/agent/interrupts.ts";
import { CallTimeLLM, contractNullable, LLM_CONFIG_KEY, splitLayers } from "../../src/nodes/agent/llm_values.ts";
import { AgentNode, isMaxToolIterations, isTextOutput } from "../../src/nodes/agent/node.ts";
import { textOf } from "../../src/nodes/agent/text.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { Binding, ContractOrigins, Parameter, ValueContract, ValueTemplate } from "../../src/nodes/contracts.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";

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

test("forward and backward are not ported yet", () => {
  const node = new AgentNode("n");
  expect(() => node.forward()).toThrow(new Error("node 'n' (AgentNode) forward is not ported yet (T5-11a)"));
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

  const copy = Object.assign(Object.create(Object.getPrototypeOf(node)) as AgentNode, node);
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

test("bindReserved needs the __llm_config wiring", () => {
  const node = new AgentNode("chat");
  node.callTime = callTime();
  expect(() => node.bindReserved(new Map(), pickPool(null))).toThrow(
    new Error("AGENT node 'chat' has no `__llm_config` wiring"),
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
