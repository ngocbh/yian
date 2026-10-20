/**
 * The backward context a node's retained invocation material builds: a port of the node-neutral
 * rows of `tests/engine/test_agent_backward.py`.
 *
 * Ported: `test_backward_context_rejects_captured_reserved_evidence_collision` (the node's
 * `captureBackwardContext` replaced on the instance; the error is a `RangeError` with its full
 * message, where Python matches a part of a `ValueError`) and
 * `test_backward_context_allows_absent_node_data_without_changing_the_record`.
 *
 * Later (T5-11a), through AGENT's forward:
 * `test_backward_context_exposes_only_accepted_evidence_and_replays_without_agent_calls`.
 *
 * Later (T6-6), with AGENT's backward: `test_agent_backward_uses_recorded_model_and_no_tools`,
 * `test_agent_backward_keeps_visible_tool_evidence_inert_and_separate`,
 * `test_agent_backward_skips_empty_effective_feedback`,
 * `test_agent_backward_validates_all_feedback_before_skipping_or_calling`,
 * `test_agent_backward_empty_string_is_effective_and_nested_arrays_keep_their_shape`,
 * `test_agent_backward_rejects_missing_retained_material_without_fallback`,
 * `test_agent_backward_rejects_malformed_retained_capture`,
 * `test_agent_backward_uses_partial_failure_capture_without_inventing_locals`,
 * `test_agent_backward_failure_is_attributed_without_forward_changes`,
 * `test_agent_backward_reads_no_live_render_model_or_variable_state` and
 * `test_agent_backward_host_controls_preserve_original_identity` (the last three also run AGENT's
 * forward).
 */

import { expect, test } from "bun:test";

import { FlowOutput, START_ID } from "../../src/compile/model.ts";
import { commitPrepared, prepareNodeStarted, prepareNodeSucceeded } from "../../src/runtime/state.ts";
import { engine, forwardCapture, FuncNode, graph } from "./run_state_fixtures.ts";

test("the backward context rejects a captured reserved evidence collision", async () => {
  const node = new FuncNode("custom", () => "done");
  node.captureBackwardContext = () => ({ _evaluation_evidence: null });
  const built = engine(graph([node], [[START_ID, "custom"]], { outputs: [new FlowOutput("result", "${custom.output}")] }), {
    learnable: true,
  });

  await built.run();

  const trace = built.state.trace!;
  const occurrence = trace.occurrences.find((item) => item.nodeId === "custom")!;
  expect(() => trace.backwardContext(occurrence.source)).toThrow(
    new RangeError("captured node data uses reserved _evaluation_evidence"),
  );
});

test("the backward context allows absent node data without changing the record", () => {
  const built = engine(graph([], []), { learnable: true });
  const { state, log } = built;
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(
    log,
    state,
    prepareNodeSucceeded(state, START_ID, {}, { forwardCapture: forwardCapture(state.flow.nodes.get(START_ID)!, {}) }),
  );

  const trace = state.trace!;
  expect(trace.occurrences).toHaveLength(1);
  const occurrence = trace.occurrences[0]!;
  expect(occurrence.record.forwardCapture!.nodeData).toBeNull();
  expect(trace.backwardContext(occurrence.source).nodeData).toEqual({ _evaluation_evidence: null });
  expect(occurrence.record.forwardCapture!.nodeData).toBeNull();
});
