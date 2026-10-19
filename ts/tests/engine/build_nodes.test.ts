/**
 * Per-kind descriptor builds: rows of `tests/engine/test_build_nodes.py`.
 *
 * Ported: `test_call_loop_descriptors_need_a_resolver` (CALL, MAP and LOOP; the LOOP count is
 * `3n`, as YAML reads it).
 *
 * Changed: Python's `None` resolver fails as CPython's `'NoneType' object is not callable`; TS's
 * `resolveChild` refuses a `null` resolver with its own text, so the rows pin the message's prefix.
 * The classes come from `compose/nodes/*.ts`, where Python imports them from `compose.parse.file`.
 *
 * Not ported here: the leaf-build rows (`test_agent_node_built_with_type_and_bindings`,
 * `test_agent_node_object_output`, `test_agent_invalid_llm_config_uses_the_config_field_range`,
 * `test_agent_non_mapping_llm_config_is_a_ranged_load_error`, `test_agent_node_knobs`,
 * `test_node_name_becomes_title`, `test_code_node_built`, `test_tool_node_built_with_args_bindings`
 * and `test_output_contract_resolves_registry_name`): Later (a T3-15 leftover, reported to the
 * coordinator), since T3-15 merged without porting them.
 */

import { expect, test } from "bun:test";

import { BuildContext } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { loadError } from "../compose/declarations.ts";

test.each([
  ["call", new CallDescriptor({ id: "c", call: "child" }), "call node 'c'", "child"],
  ["map", new MapDescriptor({ id: "m", call: "child", over: "${input.xs}" }), "call node 'm'", "child"],
  ["loop", new LoopDescriptor({ id: "p", call: "body", times: 3n }), "call node 'p'", "body"],
])("a %s descriptor needs a resolver", (_kind, desc, subject, callable) => {
  // A leaf-only context (no resolver) can't resolve the callable: a LoadError, not a TypeError.
  const error = loadError(() => desc.build(new BuildContext({ registry: new Map() })));

  expect(error.message).toStartWith(`${subject}: could not resolve callable '${callable}'`);
});
