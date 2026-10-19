/**
 * `steerable:` on AGENT: mirrors `tests/compose/test_agent_steerable_yaml.py`.
 *
 * Ported: `test_agent_node_rejects_a_non_bool_steerable` and
 * `test_needs_steering_follows_steerable`.
 *
 * Later (T3-15): `test_steerable_reaches_the_agent_node_and_defaults_off`,
 * `test_steerable_agent_without_declared_output_is_text` and
 * `test_bad_steerable_points_at_the_steerable_line`. Later (T3-15, T3-21):
 * `test_steerable_is_identity_but_false_is_not`.
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";

test("agent node rejects a non-bool steerable", () => {
  expect(() => new AgentNode("chat", { prompt: "p", steerable: 1 as unknown as boolean })).toThrow(
    new TypeError("AGENT node 'chat': steerable must be a bool"),
  );
  expect(() => new AgentNode("chat", { prompt: "p", steerable: 1 as unknown as boolean })).toThrow(TypeError);
});

test("needsSteering follows steerable", () => {
  expect(new AgentNode("chat", { prompt: "p", steerable: true }).needsSteering).toBe(true);
  expect(new AgentNode("chat", { prompt: "p" }).needsSteering).toBe(false);
});
