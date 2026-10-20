/**
 * A cloned child END's lowered post-assert fires record-scoped through `evalNode`: the `evalNode`
 * rows of `tests/engine/test_clone_child.py`.
 *
 * A cross-node ref (`${<callsite>/X.output}`) is lowered to a synth param (`__assert_tail`) fed by
 * a `${.../X.output:-null}` wiring, so binding puts the value into the record and the assert reads
 * it node-locally; nothing reads the pool around the record.
 *
 * Ported: `test_child_end_satisfied_synth_param_post_assert_succeeds` and
 * `test_child_end_violated_synth_param_post_assert_fails`.
 *
 * Changed: the END's wiring is passed as a one-node flow, where Python stamps it on the node for
 * the stub flow to derive.
 *
 * Later (T3-22), they test `clone_child` of `compile/expand`: `test_clone_child_namespaces_ids`,
 * `test_clone_child_preserves_node_origin_while_namespacing_runtime_ids`,
 * `test_clone_child_prefixes_existing_generated_origin_context`,
 * `test_clone_child_rejects_a_dynamic_static_memory_template_atomically`,
 * `test_clone_child_revalidates_forged_occurrence_before_propagation`,
 * `test_clone_child_revalidates_forged_static_scope_without_mutating_source`,
 * `test_clone_child_rewrites_one_binding_layer_and_preserves_its_origin`,
 * `test_clone_child_does_not_rewrite_a_nested_binding_payload`,
 * `test_clone_child_seeds_start_with_call_args_as_constants`,
 * `test_clone_child_alias_target_is_child_end`, `test_clone_child_end_carries_producer_edges`,
 * `test_clone_child_lowered_post_asserts_survive_clone` and
 * `test_spawner_assert_append_preserves_child_and_spawner_origins`.
 */

import { expect, test } from "bun:test";

import { Output } from "../../src/nodes/base.ts";
import { Assertion, Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { EvaluationFailure, type WorkerOutcome } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { drive, nodeFlow } from "./run_state_fixtures.ts";

/** Evaluate an END whose post-assert reads the synth param bound from `${each/tail.output}`. */
function runTailAssertEnd(pool: VariablePool): Promise<WorkerOutcome> {
  const end = EndNode.value("each/__end__", { outputName: "out" });
  end.postAsserts = [new Assertion("__assert_tail != ''", "each/tail.output != ''")];
  end.params = [...end.params!, new Parameter("__assert_tail")];
  const flow = nodeFlow(end, { out: "${each/tail.output}", __assert_tail: "${each/tail.output:-null}" });
  return drive(end, { pool, flow });
}

test("child end satisfied synth param post assert succeeds", async () => {
  const pool = new VariablePool();
  pool.set("each/tail", "ok");

  expect(await runTailAssertEnd(pool)).toBeInstanceOf(Output);
});

test("child end violated synth param post assert fails", async () => {
  const pool = new VariablePool();
  pool.set("each/tail", "");

  const result = await runTailAssertEnd(pool);

  expect(result).toBeInstanceOf(EvaluationFailure);
  expect((result as EvaluationFailure).failure.message).toContain("post-assert");
});
