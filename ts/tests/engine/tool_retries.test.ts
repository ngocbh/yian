/**
 * The TOOL node's shell: a port of `tests/engine/test_tool_retries.py`.
 *
 * Ported: `test_tool_node_rejects_a_bad_retries_count`.
 *
 * Changed: the `1.0` row is `1.5`, since `1.0` is the integer `1` in JS; a negative count
 * throws `RangeError` where Python raises `ValueError`; each row checks the whole message, not
 * just `'call'`.
 *
 * TS-only: `hostToolCall()` names the tool, the inputs, their proven types and the output type;
 * the traits read on an instance (`needsTools`, `needsSteps`); `run` rejects as not ported yet.
 *
 * Later (T4-11): `test_retries_zero_fails_as_before_and_records_no_step`,
 * `test_two_failures_then_success_records_each_attempt`,
 * `test_jitter_draws_from_half_to_the_full_delay`, `test_backoff_stops_at_the_cap`,
 * `test_retry_after_replaces_the_backoff_up_to_the_cap`,
 * `test_permanent_error_on_the_first_attempt_fails_unchanged`,
 * `test_permanent_error_on_the_second_attempt_stops_the_retries`,
 * `test_an_error_both_permanent_and_transient_is_not_retried`,
 * `test_an_invalid_argument_is_not_retried`, `test_an_unknown_argument_is_not_retried`,
 * `test_validate_checks_arguments_without_calling_the_tool` (it tests `ToolSet.validate`, ported
 * with the node's body), `test_an_output_contract_failure_is_not_retried`,
 * `test_an_awaitable_from_a_function_tool_is_not_retried`,
 * `test_exhausted_retries_list_every_attempt`, `test_a_stop_during_the_backoff_wait_stops_the_run`,
 * `test_odd_error_text_is_recorded_safely`, `test_forward_without_steps_retries_in_memory`,
 * `test_a_resumed_start_continues_the_numbering_and_rewaits`,
 * `test_a_host_stop_from_the_tool_is_neither_retried_nor_recorded`,
 * `test_the_real_sleep_raises_at_once_on_a_set_event`,
 * `test_the_real_sleep_waits_without_an_event`,
 * `test_a_wait_longer_than_timeout_max_waits_in_chunks`,
 * `test_a_stop_between_timeout_max_chunks_ends_the_wait`,
 * `test_a_failure_after_100_attempts_lists_every_attempt`, `test_ten_attempts_are_all_listed`,
 * `test_a_long_error_type_name_is_capped` and `test_a_huge_failure_count_gets_the_cap`.
 */

import { expect, test } from "bun:test";

import { NodeKind } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { typeFor } from "../../src/typesys/index.ts";

test("tool node rejects a bad retries count", () => {
  const rows: [unknown, typeof TypeError | typeof RangeError, string][] = [
    ["2", TypeError, '"2"'],
    [true, TypeError, "true"],
    [1.5, TypeError, "1.5"],
    [-1, RangeError, "-1"],
  ];
  for (const [retries, error, got] of rows) {
    const build = () => new ToolNode("call", { toolId: "flaky", retries: retries as number });
    expect(build).toThrow(error);
    expect(build).toThrow(`node 'call': retries must be a non-negative int, got ${got}`);
  }
});

test("hostToolCall names the tool, the inputs, their proven types and the output type", () => {
  const int = typeFor("int", new Map());
  const node = new ToolNode("call", { toolId: "flaky" });
  node.params = [new Parameter("symbol"), new Parameter("qty")];
  node.inputTypes = new Map([["qty", int]]);
  node.outputContract = new ValueContract(int);
  const call = node.hostToolCall();
  expect(call.toolId).toBe("flaky");
  expect(call.inputs).toEqual(["symbol", "qty"]);
  expect([...call.inputTypes]).toEqual([["qty", int]]);
  expect(call.outputType).toBe(int);

  const bare = new ToolNode("bare", { toolId: "x" }).hostToolCall();
  expect(bare.inputs).toEqual([]);
  expect(bare.inputTypes.size).toBe(0);
  expect(bare.outputType).toBeNull();
});

test("a TOOL node needs the run's tools and its durable steps", () => {
  const node = new ToolNode("call", { toolId: "flaky", title: "Call" });
  expect(node.kind).toBe(NodeKind.TOOL);
  expect(node.title).toBe("Call");
  expect(node.needsTools).toBe(true);
  expect(node.needsSteps).toBe(true);
});

test("a TOOL forward is not ported yet", async () => {
  await expect(new ToolNode("call", { toolId: "flaky" }).run({})).rejects.toThrow(
    "node 'call' (ToolNode) forward is not ported yet (T4-11)",
  );
});
