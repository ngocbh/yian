/**
 * Per-kind descriptor builds: rows of `tests/engine/test_build_nodes.py`.
 *
 * Ported: the CALL and MAP halves of `test_call_loop_descriptors_need_a_resolver`.
 *
 * Changed: Python's `None` resolver fails as CPython's `'NoneType' object is not callable`; TS's
 * `resolveChild` refuses a `null` resolver with its own text, so the rows pin the message's prefix.
 * The classes come from `compose/nodes/*.ts`, where Python imports them from `compose.parse.file`.
 *
 * Later (T3-16, the loop kind): the LOOP half of `test_call_loop_descriptors_need_a_resolver`.
 *
 * Not ported here: the leaf-build rows (`test_agent_node_built_with_type_and_bindings`,
 * `test_agent_node_object_output`, `test_agent_invalid_llm_config_uses_the_config_field_range`,
 * `test_agent_non_mapping_llm_config_is_a_ranged_load_error`, `test_agent_node_knobs`,
 * `test_node_name_becomes_title`, `test_code_node_built`, `test_tool_node_built_with_args_bindings`
 * and `test_output_contract_resolves_registry_name`): Later (T3-15 leftover; see the T3-16
 * handoff), since T3-15 merged without porting them.
 */

import { expect, test } from "bun:test";

import { BuildContext } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { loadError } from "../compose/declarations.ts";

test.each([
  ["call", new CallDescriptor({ id: "c", call: "child" }), "call node 'c'"],
  ["map", new MapDescriptor({ id: "m", call: "child", over: "${input.xs}" }), "call node 'm'"],
])("a %s descriptor needs a resolver", (_kind, desc, subject) => {
  // A leaf-only context (no resolver) can't resolve the callable: a LoadError, not a TypeError.
  const error = loadError(() => desc.build(new BuildContext({ registry: new Map() })));

  expect(error.message).toStartWith(`${subject}: could not resolve callable 'child'`);
});
