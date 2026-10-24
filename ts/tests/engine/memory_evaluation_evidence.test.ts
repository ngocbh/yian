/**
 * What a learner sees of an AGENT invocation: a port of
 * `tests/engine/test_memory_evaluation_evidence.py`.
 *
 * TS records no evidence: learning reads the steps an AGENT recorded, rebuilds its prompt from the
 * recorded flow and recalls against the recorded snapshots (`learningMaterials`). Each ported test
 * runs a learnable, logged run, replays its log and reads the rebuilt material, or what AGENT
 * backward reads of it (`evaluationEvidence`). Interactions follow step order: a reply's text when
 * it calls no tool, else its calls, then their results in call order.
 *
 * Ported as material, through `runFlow` with a `ScriptedModel` from `models` (the scripted model's
 * provider and model pick the structured strategy):
 * `test_scope_free_model_evidence_is_accepted_with_forward_capture`,
 * `test_plain_agent_records_only_visible_prompt_and_text` (the private material is a reasoning
 * part and the reply's usage, checked absent from the evidence's JSON),
 * `test_fallback_structured_parses_only_visible_text_blocks` (the private material is a reasoning
 * part, the text part's provider payload and the message's),
 * `test_tool_loop_records_valid_ungranted_call_before_failure` (with no grant the call never
 * reaches a tool loop), `test_tool_loop_records_calls_results_and_final_text_in_visible_order`,
 * `test_tool_trace_uses_the_same_effective_arguments_as_invocation` (the falsy arguments are a
 * streamed call with empty arguments text, which the assembler reads as `{}`, where Python's are
 * `[]`) and `test_tool_structured_output_retains_discovery_and_generation_order` (2 rows; the
 * native row runs OpenAI's `json_schema` plan, where Python's runs Anthropic's).
 *
 * Ported as material, through a learnable `FlowEngine` run, which records the node's model map by
 * hand (Python's run start derives it):
 * `test_actual_render_capture_survives_later_failures_without_fabricating_evidence` (4 rows).
 * Learning recalls through the run's own opener; an invocation that never called its model shows
 * no prompt. A model failure is a scripted `Error`; "the model ran" reads as the model receiving a
 * request.
 *
 * Kept without their evidence half, through `evalNode`:
 * `test_invalid_unicode_recall_prompt_fails_identically_with_or_without_capture` (no model call),
 * `test_recursive_ungranted_tool_arguments_do_not_change_failure_with_capture` (the 2000-deep
 * arguments arrive unparsed, since a parsed `ToolCallPart` refuses them; the reply's argument
 * check fails the node on their depth before the grant check) and
 * `test_invalid_unicode_tool_result_is_ignored_by_observational_capture` (a lone surrogate is not a
 * yian value, design 3.2.1, so the tool result fails the node after the call ran, where Python's
 * loop goes on).
 *
 * Not ported, since TS records no evidence (learning reads the steps):
 * `test_state_validates_evidence_scope_and_trait_for_live_and_replay`,
 * `test_evaluation_trace_protocol_is_runtime_checkable_and_write_only`,
 * `test_capture_gate_requires_exact_flag_and_trait`,
 * `test_capture_snapshots_canonical_inputs_before_node_mutation`,
 * `test_input_snapshot_canonicalization_fault_propagates_before_node_run`,
 * `test_binding_failure_captures_scope_with_absent_inputs_and_no_prompt`,
 * `test_dispatcher_contract_failures_retain_the_same_evidence`,
 * `test_pooled_dispatch_discards_late_sibling_evidence_with_its_outcome` and the TS-only trace
 * tests (one prompt per trace, no write after the evaluation settled).
 * `test_native_structured_agent_records_only_parsed_value` and
 * `test_fallback_structured_records_every_raw_retry_without_parsed_duplicate` are covered by
 * `learning_from_steps.test.ts`. `test_visible_string_subclass_has_capture_off_on_behavioral_parity`
 * and `test_tool_name_subclass_has_capture_off_on_behavioral_parity` (TS has no string subclasses);
 * `test_invalid_unicode_text_is_ignored_by_observational_capture` and
 * `test_invalid_unicode_fallback_trace_does_not_consume_a_retry` (a reply's `TextPart` refuses an
 * unpaired surrogate, so no model reply carries one);
 * `test_oversized_recall_inputs_fail_identically_with_or_without_capture` (TS has no
 * `memory_json_nesting` bound: a `RecallContext` takes inputs of any depth, and the binder's value
 * nesting limit bounds them).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { sourceKey } from "../../src/expr/lineage.ts";
import { Finish, type Event, ToolCallDelta } from "../../src/llm/events.ts";
import { Message, ReasoningPart, Response, TextPart, ToolCallPart, Usage } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import {
  DeclSite,
  MemoryContext,
  MemoryRecallError,
  MemoryScope,
  MemorySection,
  MemorySnapshotEnvelope,
  MemorySpec,
  ModelTextInteraction,
  ModelValueInteraction,
  type RecallContext,
  ToolCallInteraction,
  ToolResultInteraction,
} from "../../src/memory/index.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { replayRunLog, runFlow } from "../../src/run.ts";
import { learningMaterials } from "../../src/runtime/backward.ts";
import { EvaluationFailure, evalNode, type MemoryContextOpener } from "../../src/runtime/eval_node.ts";
import { JsonlRunLog, NodeFailedRecord, NodeSucceededRecord } from "../../src/runtime/log.ts";
import type { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import type { RunState } from "../../src/runtime/state.ts";
import { evaluationEvidence } from "../../src/runtime/trace.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { VariablePool } from "../../src/typesys/index.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { parametersFromYaml } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { engine, graph, NO_ENV, serving, stacks, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-memory-evaluation-evidence-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const NATIVE = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };

/** A tool `name`, answering what `execute` returns, `properties` its schema's. */
function tool(name: string, execute: (args: Record<string, unknown>) => unknown, properties: JsonObject = {}) {
  const parametersJsonSchema = { type: "object", properties };
  return new FunctionTool({ name, description: `The ${name} tool.`, parametersJsonSchema, execute });
}

/** `lookup`, answering `found:<topic>`. */
const lookup = () => tool("lookup", (args) => `found:${String(args["topic"])}`, { topic: { type: "string" } });

/** A reply of `parts`, finished for tool calls. */
function partsReply(...parts: (TextPart | ToolCallPart)[]): Response {
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/** A call of `lookup` on `{topic: "ACME"}` as call `id`. */
const lookupCall = (id: string) => new ToolCallPart({ id, name: "lookup", arguments: { topic: "ACME" } });

/** The flow of one AGENT `agent` with `prompt`, its `output` type, granted `tools`, with `retries`. */
function agentFlow(prompt: string, options: { output?: string; tools?: string[]; retries?: number } = {}): string {
  const { output = "str", tools, retries } = options;
  return [
    "id: agent",
    "name: agent",
    "llm_config: {provider: openai, model: gpt-5.5}",
    "nodes:",
    "  agent:",
    "    kind: agent",
    `    prompt: ${JSON.stringify(prompt)}`,
    `    output: ${output}`,
    ...(tools === undefined ? [] : [`    tools: [${tools.join(", ")}]`]),
    ...(retries === undefined ? [] : [`    retries: ${retries}`]),
    "output: ${agent.output}",
    "",
  ].join("\n");
}

let count = 0;

/** Rebuild the material of `nodeId` of the replayed `state`, with what AGENT backward reads of it. */
async function materialOf(state: RunState, nodeId: string, memory: MemoryRuntime | null = null) {
  const materials = await learningMaterials(state, { memory, models: null, onEvent: null });
  const occurrence = state.trace!.occurrences.find((item) => item.nodeId === nodeId)!;
  const material = materials.get(sourceKey(occurrence.source)) ?? null;
  return { occurrence, material, evidence: evaluationEvidence(occurrence, material) };
}

/**
 * Run `text` learnably with a model serving `replies` as `config`, replay its log, and rebuild what
 * learning reads of its `agent`.
 */
async function learned(
  text: string,
  replies: (Response | Event[])[],
  config: { provider: string; model: string },
  tools: Record<string, FunctionTool> = {},
) {
  const path = join(dir, `run-${++count}.jsonl`);
  const flow = await loadFlow(text);
  const model = new ScriptedModel(replies, config);
  const models = serving(model);
  const result = await runFlow(flow, {}, { log: path, learnable: true, models, tools, env: hostEnv(dir) });
  const log = JsonlRunLog.resume(path);
  let state: RunState;
  try {
    state = replayRunLog(flow, log);
  } finally {
    log[Symbol.dispose]();
  }
  // A structured call's plan is read off the model the run built, never called again.
  const materials = await learningMaterials(state, { memory: null, models, onEvent: null });
  const occurrence = state.trace!.occurrences.find((item) => item.nodeId === "agent")!;
  const material = materials.get(sourceKey(occurrence.source)) ?? null;
  return { result, model, material, evidence: evaluationEvidence(occurrence, material) };
}

describe("learning reads the steps", () => {
  test("a scope-free agent's material is read beside its forward capture", async () => {
    const { result, material, evidence } = await learned(agentFlow("hi"), [textReply("done")], NATIVE);
    expect(result.status).toBe("succeeded");
    expect(material!.interactions).toEqual([new ModelTextInteraction("done")]);
    expect(evidence).toEqual({
      scope: null,
      bound_inputs: {},
      base_system_prompt: DEFAULT_SYSTEM,
      prompt: "hi",
      memory_sections: [],
      interactions: [{ kind: "model_text", text: "done" }],
    });
  });

  test("a plain agent shows only the visible prompt and text", async () => {
    const message = new Message({ role: "assistant", parts: [new ReasoningPart("private"), new TextPart("answer")] });
    const usage = new Usage({ inputTokens: 4321, outputTokens: 1, totalTokens: 4322 });
    const reply = new Response({ message, finishReason: "stop", usage });

    const { result, material, evidence } = await learned(agentFlow("Answer now"), [reply], NATIVE);

    expect(result.output).toBe("answer");
    expect([material!.baseSystemPrompt, material!.prompt]).toEqual([DEFAULT_SYSTEM, "Answer now"]);
    expect(material!.memorySections).toEqual([]);
    expect(material!.interactions).toEqual([new ModelTextInteraction("answer")]);
    const rendered = JSON.stringify(evidence);
    expect(rendered).not.toContain("private");
    expect(rendered).not.toContain("usage");
    expect(rendered).not.toContain("4321");
  });

  test("fallback structured shows only visible text blocks", async () => {
    const message = new Message({
      role: "assistant",
      parts: [
        new ReasoningPart("SECRET_REASONING"),
        new TextPart('{"value": 13}', { secret: "SECRET_BLOCK_METADATA" }),
      ],
      providerPayload: { private: "SECRET_PROVIDER_METADATA" },
    });
    const reply = new Response({ message, finishReason: "stop" });

    const { result, material, evidence } = await learned(
      agentFlow("number", { output: "int", retries: 0 }),
      [reply],
      PROMPTED,
    );

    expect(result.output).toBe(13n);
    expect(material!.interactions).toEqual([new ModelTextInteraction('{"value": 13}')]);
    const rendered = JSON.stringify(evidence);
    expect(rendered).not.toContain("SECRET_REASONING");
    expect(rendered).not.toContain("SECRET_BLOCK_METADATA");
    expect(rendered).not.toContain("SECRET_PROVIDER_METADATA");
  });

  test("an ungranted tool call shows before the failure", async () => {
    const call = new ToolCallPart({ id: "provider-call-id", name: "lookup", arguments: { topic: "ACME" } });
    const reply = new Response({
      message: new Message({ role: "assistant", parts: [new TextPart("private plan"), call] }),
      finishReason: "stop",
    });

    const { result, material } = await learned(agentFlow("lookup"), [reply], NATIVE);

    expect(result.status).toBe("failed");
    expect(result.failure!.message).toContain("not granted");
    expect(material!.interactions).toEqual([new ToolCallInteraction("lookup", { topic: "ACME" })]);
  });

  test("the tool loop shows calls, results and final text in visible order", async () => {
    const usage = new Usage({ inputTokens: 5, outputTokens: 1, totalTokens: 6 });
    const first = new Response({
      message: new Message({ role: "assistant", parts: [new TextPart("private plan"), lookupCall("provider-call-id")] }),
      finishReason: "tool_calls",
      usage,
    });
    const last = new Message({ role: "assistant", parts: [new TextPart("finished")], providerPayload: { model: "secret" } });
    const replies = [first, new Response({ message: last, finishReason: "stop" })];

    const { result, material, evidence } = await learned(agentFlow("lookup", { tools: ["lookup"] }), replies, NATIVE, {
      lookup: lookup(),
    });

    expect(result.output).toBe("finished");
    expect(material!.interactions).toEqual([
      new ToolCallInteraction("lookup", { topic: "ACME" }),
      new ToolResultInteraction("lookup", "found:ACME"),
      new ModelTextInteraction("finished"),
    ]);
    const rendered = JSON.stringify(evidence);
    expect(rendered).not.toContain("provider-call-id");
    expect(rendered).not.toContain("private plan");
  });

  test("a call shows the same effective arguments the tool ran with", async () => {
    const seen: unknown[] = [];
    const noop = tool("noop", (args) => {
      seen.push(args);
      return "ok";
    });
    // A provider streams a call without arguments as empty arguments text.
    const noArguments = [
      new ToolCallDelta({ index: 0, id: "1", name: "noop", arguments: "" }),
      new Finish({ reason: "tool_calls" }),
    ];

    const { result, material } = await learned(
      agentFlow("run", { tools: ["noop"] }),
      [noArguments, textReply("finished")],
      NATIVE,
      { noop },
    );

    expect(result.output).toBe("finished");
    expect(seen).toEqual([{}]);
    expect(material!.interactions).toEqual([
      new ToolCallInteraction("noop", {}),
      new ToolResultInteraction("noop", "ok"),
      new ModelTextInteraction("finished"),
    ]);
  });

  test.each([
    ["native", NATIVE],
    ["fallback", PROMPTED],
  ] as const)("tool structured output keeps discovery and generation order (%s)", async (plan, config) => {
    const replies = [
      partsReply(new TextPart("hidden tool-turn prose"), lookupCall("provider-id")),
      textReply("visible discovery reply"),
      textReply('{"value": 9}'),
    ];
    const text = agentFlow("lookup then score", { output: "int", retries: 0, tools: ["lookup"] });

    const { result, material } = await learned(text, replies, config, { lookup: lookup() });

    expect(result.output).toBe(9n);
    expect(material!.digestOk).toBe(true);
    expect(material!.interactions).toEqual([
      new ToolCallInteraction("lookup", { topic: "ACME" }),
      new ToolResultInteraction("lookup", "found:ACME"),
      new ModelTextInteraction("visible discovery reply"),
      plan === "native" ? new ModelValueInteraction(9n) : new ModelTextInteraction('{"value": 9}'),
    ]);
  });

  test.each(["recall", "factory", "model", "success"] as const)(
    "the actual render capture survives later failures, and learning shows no prompt that wasn't sent: %s",
    async (stage) => {
      const variables = parametersFromYaml("variables:\n  instruction: !str Summarize clearly\n", new Map(), {
        field: "variables",
        requireDefault: true,
      });
      const llmConfig = { provider: "test", model: "chosen" };
      const node = new AgentNode("work", { prompt: "${instruction}: ${topic}", variables, llmConfig });
      node.params = [new Parameter("topic")];
      node.memoryScope = new MemoryScope(["flow", node.id], { memories: ["local", "outer"] });
      const section = new MemorySection("local", "Relevant memory.");
      const recallCalls: RecallContext[] = [];
      const factoryCalls: Record<string, unknown>[] = [];
      const recall: MemoryContextOpener = async (_scope, context) => {
        recallCalls.push(context);
        if (stage === "recall") throw new MemoryRecallError("recall failed");
        return new MemoryContext({ sections: [section] });
      };
      const answer = stage === "model" ? new Error("model failed") : textReply("answer");
      const model = new ScriptedModel([answer], NATIVE);
      const models: ModelAccess = {
        env: NO_ENV,
        createModel: (config) => {
          factoryCalls.push({ ...config });
          if (stage === "factory") throw new Error("factory failed");
          return model;
        },
      };
      const built = engine(
        graph([node], [
          [START_ID, "work"],
          ["work", END_ID],
        ], { wiring: { work: { topic: "reefs" } } }),
        {
          learnable: true,
          memories: ["local", "outer"].map((key) => new MemorySpec(key, key, {}, [new DeclSite([], ["root"])])),
          memoryStacks: stacks(node.memoryScope),
          memorySnapshots: ["local", "outer"].map((key) => new MemorySnapshotEnvelope(key, {})),
          memoryContextOpener: recall,
          models,
          effectiveLlmByStaticPath: { work: llmConfig },
        },
      );

      await built.run();

      const occurrence = built.state.trace!.occurrences.find((item) => item.nodeId === "work")!;
      const record = occurrence.record;
      expect(record).toBeInstanceOf(stage === "success" ? NodeSucceededRecord : NodeFailedRecord);
      expect(occurrence.boundInputs).toEqual({ topic: "reefs" });
      expect(occurrence.nodeData).toEqual({
        prompt_template: "${instruction}: ${topic}",
        variables: { instruction: "Summarize clearly" },
        variable_domains: { instruction: "str" },
        rendered_prompt: "Summarize clearly: reefs",
        effective_llm_config: llmConfig,
      });
      const visiblePrompt = stage === "model" || stage === "success";
      expect(recallCalls).toHaveLength(1);
      expect(factoryCalls).toHaveLength(stage === "recall" ? 0 : 1);
      expect(model.requests).toHaveLength(visiblePrompt ? 1 : 0);

      // Learning recalls through the run's own opener, standing in for the recorded snapshots.
      const memory = { openContext: recall } as unknown as MemoryRuntime;
      const { material } = await materialOf(built.state, "work", memory);
      expect(material!.digestOk).toBe(true);
      expect(material!.prompt).toBe(visiblePrompt ? "Summarize clearly: reefs" : null);
      expect(material!.baseSystemPrompt).toBe(visiblePrompt ? DEFAULT_SYSTEM : null);
      expect(material!.memorySections).toEqual(visiblePrompt ? [section] : []);
      expect(material!.interactions).toEqual(stage === "success" ? [new ModelTextInteraction("answer")] : []);
    },
  );
});

// Through AGENT's evaluator: failures that never depended on what learning sees.

describe("through AGENT", () => {
  beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
  afterEach(forget);

  /** Enabled memory whose policies recalled no visible sections. */
  const emptyContext: MemoryContextOpener = async () => new MemoryContext();

  /** A scoped AGENT `agent` with prompt `prompt` over `config`, granted `tools`. */
  function scopedAgent(prompt: string, config: Record<string, unknown>, tools?: string[]): AgentNode {
    const node = new AgentNode("agent", { prompt, llmConfig: config, tools });
    node.memoryScope = new MemoryScope(["flow", node.id], { memories: ["outer", "local"] });
    return node;
  }

  /** Evaluate `node` with the empty memory context, `model` and `tools`. */
  function evaluate(node: AgentNode, model: ScriptedModel, tools: ToolSet | null = null) {
    return evalNode(node, null, new VariablePool(), { memoryContextOpener: emptyContext, models: serving(model), tools });
  }

  test("an invalid unicode recall prompt fails before any model call", async () => {
    const model = new ScriptedModel([], NATIVE);
    const evaluated = await evaluate(scopedAgent("\ud800", NATIVE), model);

    expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
    expect((evaluated.outcome as EvaluationFailure).failure.message).toBe("memory recall context is invalid");
    expect(model.requests).toHaveLength(0);
  });

  test("recursive ungranted tool arguments fail on their depth", async () => {
    const raw = '{"nested":'.repeat(2000) + "{}" + "}".repeat(2000);
    const deep = partsReply(new ToolCallPart({ id: "1", name: "lookup", arguments: null, rawArguments: raw }));

    const evaluated = await evaluate(scopedAgent("lookup", NATIVE), new ScriptedModel([deep], NATIVE));

    expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
    expect((evaluated.outcome as EvaluationFailure).failure.message).toContain("tool call 'lookup' arguments");
  });

  test("an invalid unicode tool result fails the node after the call ran", async () => {
    const call = partsReply(new ToolCallPart({ id: "1", name: "lookup", arguments: {} }));
    const ran: unknown[] = [];
    const tools = new ToolSet({
      lookup: tool("lookup", (args) => {
        ran.push(args);
        return "\ud800";
      }),
    });

    const evaluated = await evaluate(
      scopedAgent("lookup", NATIVE, ["lookup"]),
      new ScriptedModel([call, textReply("finished")], NATIVE),
      tools,
    );

    expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
    const { failure } = evaluated.outcome as EvaluationFailure;
    expect(failure.message).toContain("lone surrogate");
    expect(failure.code).toBe("node.execute");
    expect(ran).toEqual([{}]);
  });
});
