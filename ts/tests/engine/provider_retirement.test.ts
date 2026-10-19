/**
 * Retired native settings fail before a flow can perform external work: mirrors
 * `tests/engine/test_provider_retirement.py`.
 *
 * Ported: `test_retired_tool_marker_is_not_an_implicit_tool_grant`.
 *
 * Elsewhere: `test_retired_native_config_points_to_unified_auth` and
 * `test_codex_is_not_a_separate_provider_route` belong to the `llm_clients` ports.
 *
 * Later (T3-15): `test_old_native_yaml_fails_during_loading`.
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";

test("retired tool marker is not an implicit tool grant", () => {
  expect(() => new AgentNode("agent", { tools: ["builtin_tools"] })).toThrow(RangeError);
  expect(() => new AgentNode("agent", { tools: ["builtin_tools"] })).toThrow(
    "grant registered Yian tools explicitly",
  );
});
