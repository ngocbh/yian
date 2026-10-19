/**
 * TOOL `retries:` lands on the node: a port of `tests/compose/test_tool_retries_load.py`.
 *
 * Ported: `test_tool_node_retries_defaults_to_the_tool_retries_default`.
 *
 * Later (T3-15, T3-22): `test_retries_loads_onto_the_node`,
 * `test_omitted_retries_uses_the_default` and `test_bad_retries_is_a_load_error_at_its_line`
 * need the loader.
 *
 * Later (T3-21, T3-22): `test_zero_retries_keeps_the_flow_fingerprint` and
 * `test_nonzero_retries_changes_the_flow_fingerprint` need the loader and the flow fingerprint.
 */

import { expect, test } from "bun:test";

import { current } from "../../src/defaults.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";

test("tool node retries defaults to the tool retries default", () => {
  expect(new ToolNode("t", { toolId: "x" }).retries).toBe(current().toolRetries);
  expect(new ToolNode("t", { toolId: "x", retries: 4 }).retries).toBe(4);
});
