/**
 * Default and structural backward on nodes: a port of `tests/engine/test_backward_structural.py`.
 * Ported: `test_behavioral_backward_stops_with_empty_channels_and_static_capture` (Python's bare
 * `Node` is a minimal subclass).
 *
 * Changed: `test_structural_output_paths_must_exist_in_the_available_accepted_output` builds an
 * END in Python; here a fake whose `backward` calls `projectStructuralFeedback`, as END, START and
 * LOOP do.
 *
 * Later (T3-8): `test_case_backward_inventory_excludes_generated_control_inputs`,
 * `test_start_projects_named_delivered_inputs_including_defaults_but_not_absence`,
 * `test_end_projects_its_zero_one_and_many_output_layouts`,
 * `test_list_end_keeps_whole_list_text_unchanged_and_records_each_output_position`,
 * `test_end_backward_uses_captured_layout_after_live_node_fields_change`,
 * `test_stopping_loop_projects_all_carried_fields_without_reevaluating_predicate`,
 * `test_failed_boundaries_forward_text_only_when_inputs_and_layout_were_captured`,
 * `test_successful_null_output_is_a_bare_contributor_but_zero_output_has_none`,
 * `test_stopped_loop_failure_does_not_route_an_unaccepted_carried_output`.
 * Later (T3-9, T4-x): `test_agent_capture_keeps_variable_inventory_separate_from_delivered_inputs`.
 * Later (T4-x): `test_input_inventory_capture_preserves_canonical_parameter_setup_faults`,
 * `test_static_backward_capture_and_names_are_detached_for_the_actual_invocation`,
 * `test_early_binding_failure_keeps_inventories_without_usable_inputs_or_fabricated_layout`,
 * `test_static_capture_serialization_faults_are_host_faults_before_the_body`,
 * `test_static_capture_exception_escapes_unchanged_and_capture_off_calls_no_hook`,
 * `test_static_layout_is_captured_before_a_preassert_failure_after_defaults`,
 * `test_custom_run_uses_its_existing_pre_run_capture_boundary`.
 */

import { expect, test } from "bun:test";

import type { FeedbackValue } from "../../src/expr/index.ts";
import {
  BackwardContext,
  BackwardResult,
  Node,
  NodeKind,
  projectStructuralFeedback,
} from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";

/** A minimal concrete node with the base backward hooks. */
class BareNode extends Node {
  readonly kind = NodeKind.CODE;
}

/** A structural node: its backward projects feedback through the captured output paths. */
class StructuralNode extends Node {
  readonly kind = NodeKind.END;

  override backward(context: BackwardContext, feedback: readonly FeedbackValue[]): BackwardResult {
    return projectStructuralFeedback(context, feedback);
  }
}

test("behavioral backward stops with empty channels and static capture", () => {
  const node = new BareNode("ordinary");
  node.params = [new Parameter("topic")];
  const context = new BackwardContext({ topic: "reefs" }, { inputNames: ["topic"], output: "answer" });

  expect(node.backward(context, ["Be more precise."])).toEqual(new BackwardResult());
  expect(node.captureBackwardContext()).toEqual({});
  expect(node.backwardInputNames()).toEqual(["topic"]);
  expect(new BareNode("empty").backwardInputNames()).toEqual([]);
});

test.each([
  ["a missing field", { actual: 1n }, ["missing"]],
  ["an index into a record", { actual: 1n }, [0n]],
  ["an index past the end", [1n], [1n]],
])("structural output paths must exist in the available accepted output: %s", (_label, output, path) => {
  const node = new StructuralNode("end");
  const context = new BackwardContext(
    { result: 1n },
    { inputNames: ["result"], output, nodeData: { input_output_paths: { result: path } } },
  );

  expect(() => node.backward(context, ["Review."])).toThrow(RangeError);
  expect(() => node.backward(context, ["Review."])).toThrow(/output path/);
});
