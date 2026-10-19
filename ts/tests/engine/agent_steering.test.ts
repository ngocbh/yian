/**
 * Host steering of a running AGENT: mirrors `tests/engine/test_agent_steering.py`.
 *
 * Ported (split): the `run({})` half of
 * `test_python_built_steerable_agent_with_structured_output_fails_before_the_model`; its
 * `eval_node` half needs the evaluator (T4).
 *
 * Changed: the structured output contract is built by hand where Python reads it from YAML through
 * `load_flow`; the node runs with no model, so there is no model to check for calls. Python's
 * `ValueError` is a `RangeError`.
 *
 * Later (T4): `test_run_boundary_rejects_an_object_without_the_inbox_methods` (the run boundary).
 *
 * Later (T5-11a, T5-11b), they run the node: `test_non_steerable_agent_never_calls_the_inbox`,
 * `test_run_context_receives_steering_only_when_steerable`, and the steer and interrupt tests
 * (`test_steers_are_sent_as_user_messages_at_the_top_of_each_iteration` through
 * `test_retry_after_a_mid_node_kill_sees_the_recorded_steer`; the resume and loop ones also need
 * T3-22 and T4).
 *
 * Not ported: `test_inbox_satisfies_the_public_protocol` (it checks the test's inbox against a
 * `typing.Protocol`; TS checks `Steering` statically).
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

test("python built steerable agent with structured output fails before the model (run half)", async () => {
  const contract = new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["answer", Type.scalar(ValueKind.INTEGER)]]),
      required: new Set(["answer"]),
    }),
  );
  const llmConfig = { provider: "anthropic", model: "claude-test" };
  const node = new AgentNode("chat", { prompt: "hi", llmConfig, steerable: true });
  node.outputContract = contract;
  const run = node.run({});
  await expect(run).rejects.toBeInstanceOf(RangeError);
  await expect(run).rejects.toThrow("AGENT node 'chat': steerable needs `str` output");
});
