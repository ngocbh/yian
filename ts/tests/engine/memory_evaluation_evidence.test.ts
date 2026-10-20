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
 * protocol; in its place the trace a node receives has `recordPrompt` and `recordInteraction` and
 * no `snapshot` or `evidence` read.
 *
 * TS-only: a second `recordPrompt` throws, also after a refused prompt (which is ignored), and a
 * write after the evaluation settled throws.
 *
 * Later (T5-11a), through AGENT:
 * `test_scope_free_model_evidence_is_accepted_with_forward_capture`,
 * `test_actual_render_capture_survives_later_failures_without_fabricating_evidence`,
 * `test_oversized_recall_inputs_fail_identically_with_or_without_capture`,
 * `test_plain_agent_records_only_visible_prompt_and_text`,
 * `test_visible_string_subclass_has_capture_off_on_behavioral_parity`,
 * `test_invalid_unicode_text_is_ignored_by_observational_capture`,
 * `test_invalid_unicode_recall_prompt_fails_identically_with_or_without_capture`,
 * `test_native_structured_agent_records_only_parsed_value`,
 * `test_fallback_structured_records_every_raw_retry_without_parsed_duplicate`,
 * `test_fallback_structured_parses_only_visible_text_blocks` and
 * `test_invalid_unicode_fallback_trace_does_not_consume_a_retry`.
 *
 * Later (T5-11b), through AGENT's tool loop:
 * `test_tool_loop_records_calls_results_and_final_text_in_visible_order`,
 * `test_tool_loop_records_valid_ungranted_call_before_failure`,
 * `test_recursive_ungranted_tool_arguments_do_not_change_failure_with_capture`,
 * `test_tool_name_subclass_has_capture_off_on_behavioral_parity`,
 * `test_tool_trace_uses_the_same_effective_arguments_as_invocation`,
 * `test_invalid_unicode_tool_result_is_ignored_by_observational_capture` and
 * `test_tool_structured_output_retains_discovery_and_generation_order`.
 *
 * Later (cycle 6a), through the engine: `test_dispatcher_contract_failures_retain_the_same_evidence`
 * and `test_pooled_dispatch_discards_late_sibling_evidence_with_its_outcome`.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import {
  DeclSite,
  EvaluationEvidence,
  type EvaluationTrace,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryScope,
  MemorySection,
  MemorySnapshotEnvelope,
  MemorySpec,
  type MemoryStack,
  ModelTextInteraction,
} from "../../src/memory/index.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import { NodeSucceededRecord } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  prepareNodeStarted,
  prepareNodeSucceeded,
  prepareRecord,
  type RunState,
} from "../../src/runtime/state.ts";
import { TypeCheckError, VariablePool } from "../../src/typesys/index.ts";
import {
  commitStartSuccess,
  forwardCapture,
  FuncNode,
  graph,
  learnableStarted,
  nodeFlow,
  rejected,
  stacks,
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
    memories: keys.map((key) => new MemorySpec(key, key, {}, [new DeclSite([], ["root"])], false)),
    memoryStacks: stacks(...scopes),
    memorySnapshots: keys.map((key) => new MemorySnapshotEnvelope(key, {})),
  };
}

/** A kind-blind trace-capable test node returning one configured result. */
class EvidenceOutputNode extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsEvaluationTrace = true;
  readonly result: NodeResult;

  constructor(nodeId: string, result: NodeResult) {
    super(nodeId);
    this.result = result;
  }

  override forward(): NodeResult {
    return this.result;
  }
}

/** Prepare one static test node as the sole in-flight evaluation. */
function inflightState(node: Node): RunState {
  const flow = graph([node], [
    [START_ID, node.id],
    [node.id, END_ID],
  ]);
  const { state, log } = learnableStarted(flow, learningConfiguration(node));
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitStartSuccess(state, log, {});
  commitPrepared(log, state, prepareNodeStarted(state, node.id));
  return state;
}

test("the state validates evidence scope and trait for live and replay", () => {
  const node = new EvidenceOutputNode("work", new Output("done"));
  node.memoryScope = scope(node.id);
  const evidence = new EvaluationEvidence(node.memoryScope, {}, null, null);
  const state = inflightState(node);

  const prepared = prepareNodeSucceeded(state, node.id, "done", {
    evidence,
    forwardCapture: forwardCapture(node, {}),
  });
  expect(prepared.record.constructor).toBe(NodeSucceededRecord);
  const record = prepared.record as NodeSucceededRecord;
  expect(record.evidence).toEqual(evidence);

  const mismatched = new EvaluationEvidence(scope(node.id, occurrence(1)), {}, null, null);
  expectError(
    () => prepareNodeSucceeded(inflightState(node), node.id, "done", { evidence: mismatched }),
    RangeError,
    "node 'work' evidence scope does not match its compiled scope",
  );
  const replayed = new NodeSucceededRecord(record.nodeId, record.commitTarget, record.value, {
    evidence: mismatched,
    forwardCapture: record.forwardCapture,
  });
  expectError(
    () => prepareRecord(inflightState(node), replayed),
    RangeError,
    "node 'work' evidence scope does not match its compiled scope",
  );

  const ordinary = new FuncNode("ordinary", () => "done");
  ordinary.memoryScope = scope(ordinary.id);
  const ordinaryEvidence = new EvaluationEvidence(ordinary.memoryScope, {}, null, null);
  expectError(
    () => prepareNodeSucceeded(inflightState(ordinary), ordinary.id, "done", { evidence: ordinaryEvidence }),
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
