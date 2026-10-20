/**
 * Per-run tools through the evaluator: the `evalNode` rows of `tests/engine/test_run_tools.py`.
 *
 * Ported: `test_eval_node_rejects_a_raw_tools_mapping` and `test_flow_engine_rejects_a_raw_tools_mapping`.
 *
 * Ported (adapted): `test_only_nodes_that_declare_needs_tools_receive_the_toolset`: every node also
 * receives `signal`, and the opted-in node's `tools` is the run's tool set itself.
 *
 * TS-only: a node declaring `needsTools` with no run tool set receives a fresh `ToolSet`.
 *
 * Not ported: the registry rows of Python-only entry shapes (a callable or bound-method entry, and
 * signature checks of a handler); TS tool sets hold `FunctionTool`s:
 * `test_callable_entry_becomes_a_tool_named_by_its_key`, `test_bound_method_entry_becomes_a_tool`,
 * `test_the_same_bound_method_may_sit_in_two_groups`, `test_ask_user_entry_must_be_a_plain_callable`
 * and `test_ask_user_handler_must_accept_every_field`.
 *
 * Not ported here: the other `ToolSet` and `ask_user` rows test the registry, not the evaluator, and
 * `tool_registry.test.ts` covers them: `test_per_run_entry_replaces_global_tool_for_that_run_only`,
 * `test_a_mapping_hides_the_global_registry`, `test_registered_tools_is_a_copy_a_run_can_extend`,
 * `test_public_invocation_rejects_unknown_arguments`, `test_base_tool_entry_must_use_its_key_as_name`,
 * `test_invalid_tools_are_rejected`, `test_group_entry_holds_tools_named_by_function_or_tool_name`,
 * `test_expand_replaces_each_group_with_its_tools_in_order`, `test_a_group_is_not_itself_a_tool`,
 * `test_group_tool_names_must_be_unique_in_the_run`,
 * `test_the_same_tool_object_may_sit_in_overlapping_groups`,
 * `test_expand_keeps_the_first_of_each_repeated_name`, `test_tool_ids_lists_the_resolvable_tools`,
 * `test_ask_user_entry_is_wrapped_so_the_model_sees_question_and_options`,
 * `test_model_options_reach_the_handler_normalized`,
 * `test_model_cannot_supply_trusted_ask_user_fields`,
 * `test_trusted_ask_user_call_passes_all_fields_to_the_handler` and
 * `test_ask_user_options_are_normalized_leniently`.
 *
 * Later (T4-11), they run the TOOL or HUMAN_INPUT body:
 * `test_eval_node_passes_the_run_toolset_to_a_tool_node`,
 * `test_flow_engine_passes_its_toolset_to_every_tool_node`,
 * `test_human_input_asks_the_run_ask_user_handler` and
 * `test_eval_node_gives_human_input_the_run_toolset`.
 *
 * Later (T5-11b), they run the AGENT tool loop: `test_agent_binds_and_calls_granted_per_run_tools`,
 * `test_agent_cannot_call_a_per_run_tool_the_flow_did_not_grant`,
 * `test_agent_granted_three_groups_sees_the_union_of_their_tools`,
 * `test_agent_grant_check_sees_expanded_group_names`,
 * `test_agent_binds_a_tool_shared_by_overlapping_groups_once` and
 * `test_agent_with_groups_and_ask_user_asks_the_top_level_handler`.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { evalNode } from "../../src/runtime/eval_node.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { engine, graph, rejected } from "./run_state_fixtures.ts";

/** A tool that returns a fixed reply. */
const PING = new FunctionTool({
  name: "ping",
  description: "Return a fixed reply.",
  parametersJsonSchema: { type: "object", properties: {} },
  execute: () => "run-pong",
});

/** Record the capabilities each evaluation supplies. */
class CapsProbe extends Node {
  readonly kind = NodeKind.CODE;
  seen: NodeCaps = {};

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    this.seen = caps;
    return new Output(null);
  }
}

/** The same probe with the tools trait. */
class ToolsProbe extends CapsProbe {
  override readonly needsTools = true;
}

test("only nodes that declare needsTools receive the toolset", async () => {
  const tools = new ToolSet({ ping: PING });
  const plain = new CapsProbe("plain");
  const optedIn = new ToolsProbe("opted_in");

  await evalNode(plain, null, new VariablePool(), { tools });
  await evalNode(optedIn, null, new VariablePool(), { tools });

  expect([Object.keys(plain.seen).sort(), Object.keys(optedIn.seen).sort()]).toEqual([["signal"], ["signal", "tools"]]);
  expect(optedIn.seen.tools).toBe(tools);
});

test("a node declaring needsTools without a run tool set receives a fresh ToolSet", async () => {
  const node = new ToolsProbe("opted_in");

  await evalNode(node, null, new VariablePool());

  expect(node.seen.tools).toBeInstanceOf(ToolSet);
});

test("eval node rejects a raw tools mapping", async () => {
  const error = await rejected(
    evalNode(new ToolNode("ping", { toolId: "ping" }), null, new VariablePool(), { tools: { ping: PING } as never }),
  );

  expect(error).toEqual(new TypeError("tools must be a ToolSet or null"));
});

test("flow engine rejects a raw tools mapping", () => {
  const flow = graph([new ToolNode("ping", { toolId: "ping" })], [
    [START_ID, "ping"],
    ["ping", END_ID],
  ]);

  expect(() => engine(flow, { tools: { ping: PING } as never })).toThrow(new TypeError("tools must be a ToolSet or null"));
});
