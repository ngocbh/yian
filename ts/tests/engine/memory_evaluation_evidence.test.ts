/**
 * Evaluation evidence: a port of `tests/engine/test_memory_evaluation_evidence.py`.
 *
 * Ported, after rewrite (the start is built with prepareRunStarted, not the engine):
 * `test_state_validates_evidence_scope_and_trait_for_live_and_replay`, whose `_inflight_state`
 * runs on the shared `learnableStarted`. Each rejected row names its `RangeError` and full message,
 * where Python matches a part of its `ValueError`.
 *
 * Ported (adapted): `test_capture_gate_requires_exact_flag_and_trait` uses a local
 * `needsEvaluationTrace` node that records one interaction, for Python's AGENT;
 * `test_capture_snapshots_canonical_inputs_before_node_mutation` checks that the snapshot is
 * frozen, for Python's mutated copy; `test_input_snapshot_canonicalization_fault_propagates_before_node_run`
 * binds a `Set` and expects the `TypeCheckError` (a `RangeError`) `canonicalSeed` throws, for
 * Python's `ValueError` naming `unordered`; `test_binding_failure_captures_scope_with_absent_inputs_and_no_prompt`
 * uses the local node with a required parameter, for Python's AGENT.
 *
 * Changed: the replayed record with the mismatched evidence is rebuilt with the
 * `NodeSucceededRecord` constructor, for Python's `dataclasses.replace`.
 * `test_evaluation_trace_protocol_is_runtime_checkable_and_write_only`: TS has no runtime-checkable
 * protocol; in its place the trace a node receives has `recordPrompt` and `recordInteraction`,
 * no `snapshot` or `evidence` read, and none of the evaluator's `freeze` or `setBoundInputs`.
 * `test_invalid_unicode_tool_result_is_ignored_by_observational_capture`, through AGENT's tool
 * loop: a lone surrogate is not a yian value (design 3.2.1), so the tool result fails the node at
 * commit after the call ran (Python's loop goes on); the failure is the same with capture off and
 * on, and the evidence holds only the call.
 *
 * TS-only: a second `recordPrompt` throws, also after a refused prompt (which is ignored), and a
 * write after the evaluation settled throws.
 *
 * Ported, through AGENT's tool-free path with a `ScriptedModel` from `models`:
 * `test_scope_free_model_evidence_is_accepted_with_forward_capture` and
 * `test_actual_render_capture_survives_later_failures_without_fabricating_evidence` (4 rows) through
 * a learnable `FlowEngine` run, which records the node's model map by hand (Python's run start
 * derives it); `test_plain_agent_records_only_visible_prompt_and_text` (the private material is a
 * reasoning part and the reply's usage, checked absent from the evidence's JSON),
 * `test_invalid_unicode_recall_prompt_fails_identically_with_or_without_capture` (the failures
 * compared but for their tracebacks), `test_native_structured_agent_records_only_parsed_value`,
 * `test_fallback_structured_records_every_raw_retry_without_parsed_duplicate`,
 * `test_fallback_structured_parses_only_visible_text_blocks` (the private material is a reasoning
 * part, the text part's provider payload and the message's) and
 * `test_tool_loop_records_valid_ungranted_call_before_failure` (with no grant the call never
 * reaches a tool loop) through `evalNode`. A model failure is a scripted
 * `Error`; "the model ran" reads as the model receiving a request.
 *
 * Ported, through AGENT's tool loop with the tools in the run's `ToolSet`:
 * `test_tool_loop_records_calls_results_and_final_text_in_visible_order`,
 * `test_tool_trace_uses_the_same_effective_arguments_as_invocation` (the falsy arguments are a
 * streamed call with empty arguments text, which the assembler reads as `{}`, where Python's
 * are `[]`) and `test_tool_structured_output_retains_discovery_and_generation_order` (2 rows; the
 * native row runs OpenAI's `json_schema` plan, where Python's runs Anthropic's).
 * `test_recursive_ungranted_tool_arguments_do_not_change_failure_with_capture`: the 2000-deep
 * arguments arrive unparsed (a parsed `ToolCallPart` refuses them); as in Python, the reply's
 * argument check fails the node on their depth before the grant check, the same with capture off
 * and on.
 *
 * Not ported: `test_visible_string_subclass_has_capture_off_on_behavioral_parity` and
 * `test_tool_name_subclass_has_capture_off_on_behavioral_parity` (TS has no string subclasses);
 * `test_invalid_unicode_text_is_ignored_by_observational_capture` and
 * `test_invalid_unicode_fallback_trace_does_not_consume_a_retry` (a reply's `TextPart` refuses an
 * unpaired surrogate, so no model reply carries one);
 * `test_oversized_recall_inputs_fail_identically_with_or_without_capture` (TS has no
 * `memory_json_nesting` bound: a `RecallContext` takes inputs of any depth, and the binder's value
 * nesting limit bounds them).
 *
 * Ported, through the engine: `test_dispatcher_contract_failures_retain_the_same_evidence` (3 rows;
 * the grow row's builder throws `GrowthRejected`).
 *
 * Ported (adapted): `test_pooled_dispatch_discards_late_sibling_evidence_with_its_outcome`: the
 * sibling returns its late value once the failure aborts its signal, for Python's sleep.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { END_ID, GrowthRejected, START_ID } from "../../src/compile/model.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Finish, ToolCallDelta } from "../../src/llm/events.ts";
import { Message, ReasoningPart, Response, TextPart, ToolCallPart, Usage } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import {
  DeclSite,
  EvaluationEvidence,
  type EvaluationTrace,
  MemoryContext,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryScope,
  MemorySection,
  MemorySnapshotEnvelope,
  MemorySpec,
  type MemoryStack,
  MemoryRecallError,
  ModelTextInteraction,
  ModelValueInteraction,
  type RecallContext,
  ToolCallInteraction,
  ToolResultInteraction,
} from "../../src/memory/index.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Grow, Node, type NodeCaps, NodeKind, type NodeResult, Output, Route } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { FlowEngine } from "../../src/runtime/engine.ts";
import { EvaluationFailure, evalNode, type MemoryContextOpener } from "../../src/runtime/eval_node.ts";
import { NodeFailedRecord, NodeSucceededRecord, type RunRecord } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  type GrowBuilder,
  prepareNodeStarted,
  prepareNodeSucceeded,
  preparePoolCommit,
  prepareRecord,
  type RunState,
} from "../../src/runtime/state.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, TypeCheckError, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { parametersFromYaml } from "../compose/declarations.ts";
import {
  aborted,
  commitStartSuccess,
  engine,
  forwardCapture,
  FuncNode,
  graph,
  learnableStarted,
  NO_ENV,
  nodeFlow,
  noGrowth,
  rejected,
  serving,
  stacks,
  textReply,
} from "./run_state_fixtures.ts";

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** One valid flow-plus-agent key scope for a test node. */
function scope(nodeId: string, occurrence: readonly MemoryOccurrenceSegment[] = []): MemoryScope {
  return new MemoryScope(["flow", nodeId], { memories: ["outer", "local"], occurrencePath: occurrence });
}

/** One dynamic MAP occurrence at `index`. */
function occurrence(index: number): MemoryOccurrenceSegment[] {
  return [new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["flow"], index)];
}

/** One canonical root-owned key table, stacks and snapshots for the nodes' scopes. */
function learningConfiguration(...nodes: Node[]): {
  memories: MemorySpec[];
  memoryStacks: MemoryStack[];
  memorySnapshots: MemorySnapshotEnvelope[];
} {
  const scopes = nodes.flatMap((node) => (node.memoryScope instanceof MemoryScope ? [node.memoryScope] : []));
  const keys = [...new Set(scopes.flatMap((item) => item.memories))].sort();
  return {
    memories: keys.map((key) => new MemorySpec(key, key, {}, [new DeclSite([], ["root"])])),
    memoryStacks: stacks(...scopes),
    memorySnapshots: keys.map((key) => new MemorySnapshotEnvelope(key, {})),
  };
}

/** A kind-blind trace-capable test node returning one configured result. */
class EvidenceOutputNode extends Node {
  readonly kind: NodeKind = NodeKind.CODE;
  override readonly needsEvaluationTrace = true;
  readonly result: NodeResult;

  constructor(nodeId: string, result: NodeResult) {
    super(nodeId);
    this.result = result;
  }

  override forward(_inputs: Record<string, unknown>, _caps: NodeCaps): NodeResult | Promise<NodeResult> {
    return this.result;
  }
}

/**
 * Prepare one static test node as the sole in-flight evaluation; with `output`, also commit it as
 * the node's pending output.
 */
function inflightState(node: Node, output?: unknown): RunState {
  const flow = graph([node], [
    [START_ID, node.id],
    [node.id, END_ID],
  ]);
  const { state, log } = learnableStarted(flow, learningConfiguration(node));
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitStartSuccess(state, log, {});
  commitPrepared(log, state, prepareNodeStarted(state, node.id));
  if (output !== undefined) commitPrepared(log, state, preparePoolCommit(state, node.id, output));
  return state;
}

test("the state validates evidence scope and trait for live and replay", () => {
  const node = new EvidenceOutputNode("work", new Output("done"));
  node.memoryScope = scope(node.id);
  const evidence = new EvaluationEvidence(node.memoryScope, {}, null, null);
  const state = inflightState(node, "done");

  const prepared = prepareNodeSucceeded(state, node.id, {
    evidence,
    forwardCapture: forwardCapture(node, {}),
  });
  expect(prepared.record.constructor).toBe(NodeSucceededRecord);
  const record = prepared.record as NodeSucceededRecord;
  expect(record.evidence).toEqual(evidence);

  const mismatched = new EvaluationEvidence(scope(node.id, occurrence(1)), {}, null, null);
  expectError(
    () => prepareNodeSucceeded(inflightState(node, "done"), node.id, { evidence: mismatched }),
    RangeError,
    "node 'work' evidence scope does not match its compiled scope",
  );
  const replayed = new NodeSucceededRecord(record.nodeId, record.commit, {
    evidence: mismatched,
    forwardCapture: record.forwardCapture,
  });
  expectError(
    () => prepareRecord(inflightState(node, "done"), replayed),
    RangeError,
    "node 'work' evidence scope does not match its compiled scope",
  );

  const ordinary = new FuncNode("ordinary", () => "done");
  ordinary.memoryScope = scope(ordinary.id);
  const ordinaryEvidence = new EvaluationEvidence(ordinary.memoryScope, {}, null, null);
  expectError(
    () => prepareNodeSucceeded(inflightState(ordinary, "done"), ordinary.id, { evidence: ordinaryEvidence }),
    RangeError,
    "node 'ordinary' is ineligible for evaluation evidence",
  );
});

/** A kind-blind trace-capable node: its body hands the trace it gets to `body`. */
class TraceNode extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsEvaluationTrace = true;
  readonly traces: (EvaluationTrace | undefined)[] = [];
  readonly body: (trace: EvaluationTrace | undefined, inputs: Record<string, unknown>) => NodeResult;
  readonly reserved: Record<string, unknown>;

  constructor(
    nodeId: string,
    body: (trace: EvaluationTrace | undefined, inputs: Record<string, unknown>) => NodeResult = () =>
      new Output("done"),
    reserved: Record<string, unknown> = {},
  ) {
    super(nodeId);
    this.body = body;
    this.reserved = reserved;
  }

  override bindReserved(): Record<string, unknown> {
    return this.reserved;
  }

  override forward(inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    this.traces.push(caps.evaluationTrace);
    return this.body(caps.evaluationTrace, inputs);
  }
}

test("the trace a node receives is write-only", async () => {
  const node = new TraceNode("work");
  await evalNode(node, null, new VariablePool(), { captureEvidence: true });
  const trace = node.traces[0]!;
  expect(typeof trace.recordPrompt).toBe("function");
  expect(typeof trace.recordInteraction).toBe("function");
  expect("snapshot" in trace).toBe(false);
  expect("evidence" in trace).toBe(false);
  expect("freeze" in trace).toBe(false);
  expect("setBoundInputs" in trace).toBe(false);
});

test("capture gate requires exact flag and trait", async () => {
  const record = (trace: EvaluationTrace | undefined): NodeResult => {
    trace?.recordInteraction(new ModelTextInteraction("done"));
    return new Output("done");
  };
  const scoped = new TraceNode("scoped", record);
  scoped.memoryScope = scope(scoped.id);
  const disabled = await evalNode(scoped, null, new VariablePool());
  expect(disabled.evidence).toBeNull();
  expect(scoped.traces).toEqual([undefined]);

  const unscoped = new TraceNode("unscoped", record);
  const evidence = (await evalNode(unscoped, null, new VariablePool(), { captureEvidence: true })).evidence!;
  expect(evidence.scope).toBeNull();
  expect(evidence.interactions).toEqual([new ModelTextInteraction("done")]);

  const ordinary = new FuncNode("ordinary", () => "done");
  ordinary.memoryScope = scope(ordinary.id);
  expect((await evalNode(ordinary, null, new VariablePool(), { captureEvidence: true })).evidence).toBeNull();

  const error = await rejected(evalNode(scoped, null, new VariablePool(), { captureEvidence: 1 as never }));
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toBe("capture_evidence must be a bool");
});

test("capture snapshots canonical inputs before node mutation", async () => {
  const node = new TraceNode(
    "mutate",
    (_trace, inputs) => {
      ((inputs["payload"] as Record<string, unknown[]>)["items"]!).push(2n);
      return new Output(inputs);
    },
    { payload: { items: [1n] } },
  );
  node.memoryScope = scope(node.id);
  const evaluated = await evalNode(node, null, new VariablePool(), { captureEvidence: true });
  expect(evaluated.outcome).toEqual(new Output({ payload: { items: [1n, 2n] } }));
  expect(evaluated.evidence!.boundInputs).toEqual({ payload: { items: [1n] } });
  const items = (evaluated.evidence!.boundInputs!["payload"] as Record<string, unknown[]>)["items"]!;
  expect(Object.isFrozen(items)).toBe(true);
});

test("input snapshot canonicalization fault propagates before node run", async () => {
  const node = new TraceNode("bad-input", undefined, { unordered: new Set([1n, 2n]) });
  node.memoryScope = scope(node.id);
  const error = await rejected(evalNode(node, null, new VariablePool(), { captureEvidence: true }));
  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toMatch(/not a JSON value/);
  expect(node.traces).toEqual([]);
});

test("binding failure captures scope with absent inputs and no prompt", async () => {
  const node = new TraceNode("agent");
  node.params = [new Parameter("required", { required: true })];
  node.memoryScope = scope(node.id);
  const evaluated = await evalNode(node, nodeFlow(node, {}), new VariablePool(), { captureEvidence: true });
  expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
  expect(evaluated.evidence).toEqual(new EvaluationEvidence(node.memoryScope, null, null, null));
  expect(node.traces).toEqual([]);
});

test("a trace records one prompt; a refused prompt is ignored but still counts", async () => {
  const outside = new MemorySection("elsewhere", "notes");
  const errors: unknown[] = [];
  const node = new TraceNode("work", (trace) => {
    trace!.recordPrompt("system", "refused", [outside]);
    try {
      trace!.recordPrompt("system", "second");
    } catch (error) {
      errors.push(error);
    }
    return new Output("done");
  });
  node.memoryScope = scope(node.id);
  const evidence = (await evalNode(node, null, new VariablePool(), { captureEvidence: true })).evidence!;
  expect(evidence.prompt).toBeNull();
  expect(evidence.baseSystemPrompt).toBeNull();
  expect(errors).toHaveLength(1);
  expect(errors[0]).toBeInstanceOf(Error);
  expect((errors[0] as Error).message).toBe("evaluation trace prompt was already recorded");
});

test("a trace refuses writes after the evaluation settled", async () => {
  const node = new TraceNode("work", (trace) => {
    trace!.recordPrompt("system", "prompt");
    return new Output("done");
  });
  const evidence = (await evalNode(node, null, new VariablePool(), { captureEvidence: true })).evidence!;
  expect(evidence.prompt).toBe("prompt");
  const trace = node.traces[0]!;
  expect(() => trace.recordInteraction(new ModelTextInteraction("late"))).toThrow(
    "evaluation trace is already frozen",
  );
  expect(() => trace.recordPrompt("system", "late")).toThrow("evaluation trace is already frozen");
  expect(evidence.interactions).toEqual([]);
});

// Through the engine: contract failures and discarded siblings keep or drop their evidence.

/** A trace-capable CASE node returning one configured result. */
class EvidenceRouteNode extends EvidenceOutputNode {
  override readonly kind = NodeKind.CASE;
}

/** A trace-capable spawner returning one configured result. */
class EvidenceGrowNode extends EvidenceOutputNode {
  override readonly kind = NodeKind.CALL;
  override readonly isSpawner = true;
  override readonly growDepthDelta = 1;
}

/** Return the records of `built`'s log, in order. */
function logRecords(built: FlowEngine): RunRecord[] {
  return built.log.records().map((envelope) => envelope.record);
}

test.each(["output", "route", "grow"] as const)("dispatcher contract failures retain the same evidence: %s", async (arm) => {
  let node: EvidenceOutputNode;
  let nodes: Node[];
  let edges: [string, string, string?][];
  let growBuilder: GrowBuilder = noGrowth;
  if (arm === "output") {
    node = new EvidenceOutputNode("work", new Output("not-an-int"));
    node.outputContract = new ValueContract(Type.scalar(ValueKind.INTEGER));
    nodes = [node];
    edges = [
      [START_ID, node.id],
      [node.id, END_ID],
    ];
  } else if (arm === "route") {
    node = new EvidenceRouteNode("work", new Route("absent"));
    nodes = [node, new FuncNode("target", () => "done")];
    edges = [
      [START_ID, node.id],
      [node.id, "target", "present"],
      ["target", END_ID],
    ];
  } else {
    node = new EvidenceGrowNode("work", new Grow({ bad: true }));
    nodes = [node];
    edges = [
      [START_ID, node.id],
      [node.id, END_ID],
    ];
    growBuilder = () => {
      throw new GrowthRejected(new TypeError("rejected growth"));
    };
  }
  node.memoryScope = scope(node.id);
  const built = engine(graph(nodes, edges), { growBuilder, learnable: true, ...learningConfiguration(node) });

  await built.run();

  const failures = logRecords(built).filter((record) => record instanceof NodeFailedRecord) as NodeFailedRecord[];
  expect(failures.length).toBe(1);
  expect(failures[0]!.evidence).not.toBeNull();
  expect(failures[0]!.evidence!.scope).toEqual(node.memoryScope);
  expect(failures[0]!.evidence!.boundInputs).toEqual({});
});

test("pooled dispatch discards late sibling evidence with its outcome", async () => {
  /** Blocks until its signal aborts, then returns a late value. */
  class LateSibling extends EvidenceOutputNode {
    readonly entered = Promise.withResolvers<void>();

    override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
      this.entered.resolve();
      await aborted(caps.signal!).catch(() => {});
      return new Output("late");
    }
  }
  /** Returns a value its contract refuses once the sibling has entered. */
  class FailAfterSiblingStarts extends EvidenceOutputNode {
    override async forward(): Promise<NodeResult> {
      await sibling.entered.promise;
      return new Output("not-an-int");
    }
  }
  const failing = new FailAfterSiblingStarts("failing", new Output(null));
  failing.outputContract = new ValueContract(Type.scalar(ValueKind.INTEGER));
  const sibling = new LateSibling("sibling", new Output(null));
  failing.memoryScope = scope(failing.id);
  sibling.memoryScope = scope(sibling.id);
  const flow = graph([failing, sibling], [
    [START_ID, failing.id],
    [START_ID, sibling.id],
    [failing.id, END_ID],
    [sibling.id, END_ID],
  ]);
  const built = engine(flow, { numWorkers: 2, learnable: true, ...learningConfiguration(failing, sibling) });

  await built.run();

  const records = logRecords(built);
  const failure = records.find((record) => record instanceof NodeFailedRecord) as NodeFailedRecord;
  expect(failure.failure.nodeId).toBe(failing.id);
  expect(failure.evidence).not.toBeNull();
  expect(records.some((record) => record instanceof NodeSucceededRecord && record.nodeId === sibling.id)).toBe(false);
});

// Through AGENT's tool-free path: a `ScriptedModel` from `models` stands in for the provider.

describe("through AGENT", () => {
  beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
  afterEach(forget);

  const NATIVE = { provider: "openai", model: "gpt-5.5" };
  const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };

  /** Enabled memory whose policies recalled no visible sections. */
  const emptyContext: MemoryContextOpener = async () => new MemoryContext();

  /** A scoped AGENT `agent` with prompt `prompt` over `config`, its output `output` when given. */
  function scopedAgent(
    prompt: string,
    config: Record<string, unknown>,
    options: { output?: Type; retries?: number; tools?: string[] } = {},
  ): AgentNode {
    const { retries, tools } = options;
    const node = new AgentNode("agent", { prompt, llmConfig: config, retries, tools });
    if (options.output !== undefined) node.outputContract = new ValueContract(options.output);
    node.memoryScope = scope(node.id);
    return node;
  }

  /** Evaluate `node` capturing evidence (or not), with the empty memory context, `model` and `tools`. */
  function evaluate(node: AgentNode, model: ScriptedModel, captureEvidence = true, tools: ToolSet | null = null) {
    return evalNode(node, null, new VariablePool(), {
      captureEvidence,
      memoryContextOpener: emptyContext,
      models: serving(model),
      tools,
    });
  }

  test("scope-free model evidence is accepted with the forward capture", async () => {
    const node = new AgentNode("work", { llmConfig: NATIVE });
    const built = engine(
      graph([node], [
        [START_ID, node.id],
        [node.id, END_ID],
      ]),
      {
        learnable: true,
        models: serving(new ScriptedModel([textReply("done")], NATIVE)),
        effectiveLlmByStaticPath: { work: NATIVE },
      },
    );
    await built.run();
    const record = built.state.trace!.occurrences.find((item) => item.nodeId === node.id)!.record;
    expect(record.evidence!.scope).toBeNull();
    expect(record.evidence!.interactions).toEqual([new ModelTextInteraction("done")]);
    expect(record.forwardCapture!.boundInputs).toEqual({});
    expect(record.forwardCapture!.scope!.staticPath).toEqual([node.id]);
  });

  test.each(["recall", "factory", "model", "success"] as const)(
    "the actual render capture survives later failures without fabricating evidence: %s",
    async (stage) => {
      const variables = parametersFromYaml("variables:\n  instruction: !str Summarize clearly\n", new Map(), {
        field: "variables",
        requireDefault: true,
      });
      const llmConfig = { provider: "test", model: "chosen" };
      const node = new AgentNode("work", { prompt: "${instruction}: ${topic}", variables, llmConfig });
      node.params = [new Parameter("topic")];
      node.memoryScope = scope(node.id);
      const section = new MemorySection(node.memoryScope.memories[0]!, "Relevant memory.");
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
          ...learningConfiguration(node),
          memoryContextOpener: recall,
          models,
          effectiveLlmByStaticPath: { work: llmConfig },
        },
      );

      await built.run();

      const record = built.state.trace!.occurrences.find((item) => item.nodeId === "work")!.record;
      expect(record).toBeInstanceOf(stage === "success" ? NodeSucceededRecord : NodeFailedRecord);
      expect(record.forwardCapture!.boundInputs).toEqual({ topic: "reefs" });
      expect(record.forwardCapture!.nodeData).toEqual({
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
      const evidence = record.evidence!;
      expect(evidence.prompt).toBe(visiblePrompt ? "Summarize clearly: reefs" : null);
      expect(evidence.baseSystemPrompt).toBe(visiblePrompt ? DEFAULT_SYSTEM : null);
      expect(evidence.memorySections).toEqual(visiblePrompt ? [section] : []);
      expect(evidence.interactions).toEqual(stage === "success" ? [new ModelTextInteraction("answer")] : []);
    },
  );

  test("a plain agent records only the visible prompt and text", async () => {
    const message = new Message({ role: "assistant", parts: [new ReasoningPart("private"), new TextPart("answer")] });
    const usage = new Usage({ inputTokens: 4321, outputTokens: 1, totalTokens: 4322 });
    const model = new ScriptedModel([new Response({ message, finishReason: "stop", usage })], NATIVE);

    const evaluated = await evaluate(scopedAgent("Answer now", NATIVE), model);

    expect(evaluated.outcome).toEqual(new Output("answer"));
    const evidence = evaluated.evidence!;
    expect(evidence.boundInputs).toEqual({});
    expect([evidence.baseSystemPrompt, evidence.prompt]).toEqual([DEFAULT_SYSTEM, "Answer now"]);
    expect(evidence.memorySections).toEqual([]);
    expect(evidence.interactions).toEqual([new ModelTextInteraction("answer")]);
    const rendered = JSON.stringify(evidence);
    expect(rendered).not.toContain("private");
    expect(rendered).not.toContain("usage");
    expect(rendered).not.toContain("4321");
  });

  test("an invalid unicode recall prompt fails identically with or without capture", async () => {
    const models = [new ScriptedModel([], NATIVE), new ScriptedModel([], NATIVE)];
    const off = await evaluate(scopedAgent("\ud800", NATIVE), models[0]!, false);
    const on = await evaluate(scopedAgent("\ud800", NATIVE), models[1]!);

    // Alike but for the traceback, which names each call site.
    const failure = ({ failure }: EvaluationFailure) => ({ ...failure, traceback: null });
    expect(off.outcome).toBeInstanceOf(EvaluationFailure);
    expect(failure(on.outcome as EvaluationFailure)).toEqual(failure(off.outcome as EvaluationFailure));
    expect((on.outcome as EvaluationFailure).failure.message).toBe("memory recall context is invalid");
    expect(on.evidence!.baseSystemPrompt).toBeNull();
    expect(on.evidence!.prompt).toBeNull();
    expect(on.evidence!.interactions).toEqual([]);
    expect(models.every((model) => model.requests.length === 0)).toBe(true);
  });

  test("a native structured agent records only the parsed value", async () => {
    const node = scopedAgent("number", NATIVE, { output: Type.scalar(ValueKind.INTEGER) });

    const evaluated = await evaluate(node, new ScriptedModel([textReply('{"value": 7}')], NATIVE));

    expect(evaluated.outcome).toEqual(new Output(7n));
    expect(evaluated.evidence!.interactions).toEqual([new ModelValueInteraction(7n)]);
  });

  test("fallback structured records every raw retry without a parsed duplicate", async () => {
    const node = scopedAgent("number", PROMPTED, { output: Type.scalar(ValueKind.INTEGER), retries: 1 });
    const model = new ScriptedModel([textReply("not json"), textReply('{"value": 7}')], PROMPTED);

    const evaluated = await evaluate(node, model);

    expect(evaluated.outcome).toEqual(new Output(7n));
    expect(evaluated.evidence!.interactions).toEqual([
      new ModelTextInteraction("not json"),
      new ModelTextInteraction('{"value": 7}'),
    ]);
  });

  test("fallback structured parses only visible text blocks", async () => {
    const node = scopedAgent("number", PROMPTED, { output: Type.scalar(ValueKind.INTEGER), retries: 0 });
    const message = new Message({
      role: "assistant",
      parts: [
        new ReasoningPart("SECRET_REASONING"),
        new TextPart('{"value": 13}', { secret: "SECRET_BLOCK_METADATA" }),
      ],
      providerPayload: { private: "SECRET_PROVIDER_METADATA" },
    });
    const model = new ScriptedModel([new Response({ message, finishReason: "stop" })], PROMPTED);

    const evaluated = await evaluate(node, model);

    expect(evaluated.outcome).toEqual(new Output(13n));
    expect(evaluated.evidence!.interactions).toEqual([new ModelTextInteraction('{"value": 13}')]);
    const rendered = JSON.stringify(evaluated.evidence);
    expect(rendered).not.toContain("SECRET_REASONING");
    expect(rendered).not.toContain("SECRET_BLOCK_METADATA");
    expect(rendered).not.toContain("SECRET_PROVIDER_METADATA");
  });

  test("an ungranted tool call is recorded before the failure", async () => {
    const call = new ToolCallPart({ id: "provider-call-id", name: "lookup", arguments: { topic: "ACME" } });
    const message = new Message({ role: "assistant", parts: [new TextPart("private plan"), call] });
    const model = new ScriptedModel([new Response({ message, finishReason: "stop" })], NATIVE);

    const evaluated = await evaluate(scopedAgent("lookup", NATIVE), model);

    expect(evaluated.outcome).toBeInstanceOf(EvaluationFailure);
    expect((evaluated.outcome as EvaluationFailure).failure.message).toContain("not granted");
    expect(evaluated.evidence!.interactions).toEqual([new ToolCallInteraction("lookup", { topic: "ACME" })]);
  });

  /** A run tool set holding `name`, which runs `execute` on its arguments, `properties` its schema's. */
  function toolSet(
    name: string,
    execute: (args: Record<string, unknown>) => unknown,
    properties: JsonObject = {},
  ): ToolSet {
    const parametersJsonSchema = { type: "object", properties };
    return new ToolSet({ [name]: new FunctionTool({ name, description: `The ${name} tool.`, parametersJsonSchema, execute }) });
  }

  /** `lookup`, answering `found:<topic>`. */
  const lookupTools = () => toolSet("lookup", (args) => `found:${args.topic}`, { topic: { type: "string" } });

  /** A reply of `parts`, finished for tool calls. */
  function partsReply(...parts: (TextPart | ToolCallPart)[]): Response {
    return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
  }

  /** A call of `lookup` on `{topic: "ACME"}` as call `id`. */
  const lookupCall = (id: string) => new ToolCallPart({ id, name: "lookup", arguments: { topic: "ACME" } });

  test("the tool loop records calls, results and final text in visible order", async () => {
    const usage = new Usage({ inputTokens: 5, outputTokens: 1, totalTokens: 6 });
    const first = new Response({
      message: new Message({ role: "assistant", parts: [new TextPart("private plan"), lookupCall("provider-call-id")] }),
      finishReason: "tool_calls",
      usage,
    });
    const last = new Message({ role: "assistant", parts: [new TextPart("finished")], providerPayload: { model: "secret" } });
    const model = new ScriptedModel([first, new Response({ message: last, finishReason: "stop" })], NATIVE);

    const evaluated = await evaluate(scopedAgent("lookup", NATIVE, { tools: ["lookup"] }), model, true, lookupTools());

    expect(evaluated.outcome).toEqual(new Output("finished"));
    expect(evaluated.evidence!.interactions).toEqual([
      new ToolCallInteraction("lookup", { topic: "ACME" }),
      new ToolResultInteraction("lookup", "found:ACME"),
      new ModelTextInteraction("finished"),
    ]);
    const rendered = JSON.stringify(evaluated.evidence);
    expect(rendered).not.toContain("provider-call-id");
    expect(rendered).not.toContain("private plan");
  });

  test("recursive ungranted tool arguments do not change the failure with capture", async () => {
    const raw = '{"nested":'.repeat(2000) + "{}" + "}".repeat(2000);
    const deep = () => partsReply(new ToolCallPart({ id: "1", name: "lookup", arguments: null, rawArguments: raw }));
    const node = scopedAgent("lookup", NATIVE);

    const off = await evaluate(node, new ScriptedModel([deep()], NATIVE), false);
    const on = await evaluate(node, new ScriptedModel([deep()], NATIVE));

    const failure = (outcome: unknown) => (outcome as EvaluationFailure).failure;
    expect(off.outcome).toBeInstanceOf(EvaluationFailure);
    expect(on.outcome).toBeInstanceOf(EvaluationFailure);
    expect([failure(on.outcome).code, failure(on.outcome).message]).toEqual([
      failure(off.outcome).code,
      failure(off.outcome).message,
    ]);
    expect(failure(on.outcome).message).toContain("tool call 'lookup' arguments");
    expect(on.evidence!.interactions).toEqual([]);
  });

  test("the tool trace uses the same effective arguments as the invocation", async () => {
    const seen: unknown[] = [];
    const tools = toolSet("noop", (args) => {
      seen.push(args);
      return "ok";
    });
    // A provider streams a call without arguments as empty arguments text.
    const noArguments = [
      new ToolCallDelta({ index: 0, id: "1", name: "noop", arguments: "" }),
      new Finish({ reason: "tool_calls" }),
    ];
    const model = new ScriptedModel([noArguments, textReply("finished")], NATIVE);

    const evaluated = await evaluate(scopedAgent("run", NATIVE, { tools: ["noop"] }), model, true, tools);

    expect(evaluated.outcome).toEqual(new Output("finished"));
    expect(seen).toEqual([{}]);
    expect(evaluated.evidence!.interactions).toEqual([
      new ToolCallInteraction("noop", {}),
      new ToolResultInteraction("noop", "ok"),
      new ModelTextInteraction("finished"),
    ]);
  });

  test("an invalid unicode tool result fails the same with or without capture", async () => {
    const call = () => partsReply(new ToolCallPart({ id: "1", name: "lookup", arguments: {} }));
    const run = (captureEvidence: boolean) =>
      evaluate(
        scopedAgent("lookup", NATIVE, { tools: ["lookup"] }),
        new ScriptedModel([call(), textReply("finished")], NATIVE),
        captureEvidence,
        toolSet("lookup", () => "\ud800"),
      );

    const off = await run(false);
    const on = await run(true);

    // Alike but for the traceback, which names each call site.
    const failure = (outcome: unknown) => ({ ...(outcome as EvaluationFailure).failure, traceback: null });
    expect(off.outcome).toBeInstanceOf(EvaluationFailure);
    expect(failure(on.outcome)).toEqual(failure(off.outcome));
    expect(failure(on.outcome).message).toContain("lone surrogate");
    expect(failure(on.outcome).code).toBe("node.execute");
    expect(on.evidence!.interactions).toEqual([new ToolCallInteraction("lookup", {})]);
  });

  test.each([
    ["native", NATIVE],
    ["fallback", PROMPTED],
  ] as const)("tool structured output retains discovery and generation order (%s)", async (plan, config) => {
    const replies = [
      partsReply(new TextPart("hidden tool-turn prose"), lookupCall("provider-id")),
      textReply("visible discovery reply"),
      textReply('{"value": 9}'),
    ];
    const output = Type.scalar(ValueKind.INTEGER);
    const node = scopedAgent("lookup then score", config, { output, retries: 0, tools: ["lookup"] });

    const evaluated = await evaluate(node, new ScriptedModel(replies, config), true, lookupTools());

    expect(evaluated.outcome).toEqual(new Output(9n));
    expect(evaluated.evidence!.interactions).toEqual([
      new ToolCallInteraction("lookup", { topic: "ACME" }),
      new ToolResultInteraction("lookup", "found:ACME"),
      new ModelTextInteraction("visible discovery reply"),
      plan === "native" ? new ModelValueInteraction(9n) : new ModelTextInteraction('{"value": 9}'),
    ]);
  });
});
