/**
 * `system_prompt:` on AGENT: mirrors `tests/compose/test_agent_system_prompt_yaml.py`.
 *
 * Ported: `test_agent_node_rejects_a_non_string_system_prompt`, and the constructor half of
 * `test_system_prompt_defaults_to_none`.
 *
 * Changed: the message says `or null` where Python's says `or None`.
 *
 * Later (T3-15): the `load_flow` half of `test_system_prompt_defaults_to_none`,
 * `test_system_prompt_reaches_the_agent_node`, `test_compact_flow_accepts_system_prompt`,
 * `test_system_prompt_sees_the_node_variables`, `test_unknown_ref_points_at_the_system_prompt_line`,
 * `test_unknown_ref_in_a_compact_flow_is_located` and
 * `test_non_string_system_prompt_points_at_its_line`. Later (T3-15, T3-21):
 * `test_flow_without_system_prompt_keeps_its_fingerprint` and `test_system_prompt_is_identity`.
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";

test("system prompt defaults to null", () => {
  expect(new AgentNode("reply").systemPrompt).toBeNull();
});

test("agent node rejects a non-string system prompt", () => {
  const message = "AGENT node 'reply': system_prompt must be a non-empty string or null";
  expect(() => new AgentNode("reply", { systemPrompt: 7 as unknown as string })).toThrow(new TypeError(message));
  expect(() => new AgentNode("reply", { systemPrompt: 7 as unknown as string })).toThrow(TypeError);
  expect(() => new AgentNode("reply", { systemPrompt: "" })).toThrow(new TypeError(message));
  expect(() => new AgentNode("reply", { systemPrompt: "" })).toThrow(TypeError);
});
