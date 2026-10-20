/**
 * Evaluation evidence: a port of `tests/engine/test_memory_evaluation_evidence.py`.
 *
 * Ported, after rewrite (the start is built with prepareRunStarted, not the engine):
 * `test_state_validates_evidence_scope_and_trait_for_live_and_replay`, whose `_inflight_state`
 * runs on the shared `learnableStarted`. Each rejected row names its `RangeError` and full message,
 * where Python matches a part of its `ValueError`.
 *
 * Changed: the replayed record with the mismatched evidence is rebuilt with the
 * `NodeSucceededRecord` constructor, for Python's `dataclasses.replace`.
 *
 * Later (T4-6, with the evaluator and the engine):
 * `test_evaluation_trace_protocol_is_runtime_checkable_and_write_only`,
 * `test_scope_free_model_evidence_is_accepted_with_forward_capture`,
 * `test_actual_render_capture_survives_later_failures_without_fabricating_evidence`,
 * `test_capture_gate_requires_exact_flag_and_trait`,
 * `test_capture_snapshots_canonical_inputs_before_node_mutation`,
 * `test_input_snapshot_canonicalization_fault_propagates_before_node_run`,
 * `test_binding_failure_captures_scope_with_absent_inputs_and_no_prompt`,
 * `test_oversized_recall_inputs_fail_identically_with_or_without_capture`,
 * `test_plain_agent_records_only_visible_prompt_and_text`,
 * `test_visible_string_subclass_has_capture_off_on_behavioral_parity`,
 * `test_invalid_unicode_text_is_ignored_by_observational_capture`,
 * `test_invalid_unicode_recall_prompt_fails_identically_with_or_without_capture`,
 * `test_native_structured_agent_records_only_parsed_value`,
 * `test_fallback_structured_records_every_raw_retry_without_parsed_duplicate`,
 * `test_fallback_structured_parses_only_visible_text_blocks`,
 * `test_invalid_unicode_fallback_trace_does_not_consume_a_retry`,
 * `test_tool_loop_records_calls_results_and_final_text_in_visible_order`,
 * `test_tool_loop_records_valid_ungranted_call_before_failure`,
 * `test_recursive_ungranted_tool_arguments_do_not_change_failure_with_capture`,
 * `test_tool_name_subclass_has_capture_off_on_behavioral_parity`,
 * `test_tool_trace_uses_the_same_effective_arguments_as_invocation`,
 * `test_invalid_unicode_tool_result_is_ignored_by_observational_capture`,
 * `test_tool_structured_output_retains_discovery_and_generation_order`,
 * `test_dispatcher_contract_failures_retain_the_same_evidence` and
 * `test_pooled_dispatch_discards_late_sibling_evidence_with_its_outcome`.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import {
  DeclSite,
  EvaluationEvidence,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryScope,
  MemorySnapshotEnvelope,
  MemorySpec,
  type MemoryStack,
} from "../../src/memory/index.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { NodeSucceededRecord } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  prepareNodeStarted,
  prepareNodeSucceeded,
  prepareRecord,
  type RunState,
} from "../../src/runtime/state.ts";
import {
  commitStartSuccess,
  forwardCapture,
  FuncNode,
  graph,
  learnableStarted,
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
