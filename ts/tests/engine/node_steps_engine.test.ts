/**
 * The durable steps a node records: a port of `tests/engine/test_node_steps_engine.py`.
 *
 * Ported: `test_node_steps_checks_its_fields` (`NodeSteps` checks `starts`, `prior` and `record`).
 *
 * Ported (adapted): `test_eval_node_passes_steps_only_to_opted_in_nodes_and_checks_the_type`: the
 * nodes are evaluated with no flow (empty wiring), and the check's message is pinned.
 *
 * Changed: `ValueError` is `RangeError`; `True` for `starts` is `true`.
 *
 * Later (FlowEngine), they run the engine: `test_steps_are_durable_before_record_returns`,
 * `test_a_node_without_the_trait_receives_no_steps`,
 * `test_resume_sees_the_second_start_and_the_prior_steps`, `test_resume_hands_detached_prior_steps`,
 * `test_pooled_workers_never_read_the_step_authority`,
 * `test_stop_releases_a_pooled_worker_waiting_for_its_step`,
 * `test_a_sibling_failure_releases_a_pooled_worker_waiting_for_its_step`,
 * `test_a_recorder_log_fault_escapes_as_a_host_error`, `test_an_invalid_step_is_a_host_error`,
 * `test_serial_skips_a_step_once_stop_is_set`, `test_a_kept_recorder_is_closed_after_its_evaluation`
 * and `test_pooled_nodes_interleave_their_own_steps`.
 *
 * Not ported: the `prior=[{}]` row, a list where Python wants a tuple: TS has only arrays.
 */

import { expect, test } from "bun:test";

import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { evalNode } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

/** A node that opts into steps and runs a scripted body over its capabilities. */
class StepNode extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsSteps: boolean = true;
  readonly body: (caps: NodeCaps) => NodeResult;

  constructor(nodeId: string, body: (caps: NodeCaps) => NodeResult) {
    super(nodeId);
    this.body = body;
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    return this.body(caps);
  }
}

/** The same scripted node without the steps trait. */
class PlainNode extends StepNode {
  override readonly needsSteps = false;
}

test("NodeSteps checks its fields", () => {
  const record = (): void => {};

  expect(() => new NodeSteps(true as never, [], record)).toThrow(TypeError);
  expect(() => new NodeSteps(0, [], record)).toThrow(RangeError);
  expect(() => new NodeSteps(1, ["x"] as never, record)).toThrow(TypeError);
  expect(() => new NodeSteps(1, ["x"] as never, record)).toThrow(/^prior must be an array of records$/);
  expect(() => new NodeSteps(1, [], null as never)).toThrow(TypeError);
});

test("eval node passes steps only to opted in nodes and checks the type", async () => {
  const steps = new NodeSteps(1, [], () => {});
  const seen: unknown[] = [];
  const body = (caps: NodeCaps): NodeResult => {
    seen.push(caps.steps);
    return new Output("done");
  };
  const pool = new VariablePool();

  await evalNode(new StepNode("a", body), null, pool, { steps });
  await evalNode(new PlainNode("b", body), null, pool, { steps });
  await evalNode(new StepNode("c", body), null, pool);

  expect(seen).toEqual([steps, undefined, undefined]);
  expect(seen[0]).toBe(steps);
  await expect(evalNode(new StepNode("d", body), null, pool, { steps: { starts: 1 } as never })).rejects.toThrow(
    new TypeError("steps must be a NodeSteps or null"),
  );
});
