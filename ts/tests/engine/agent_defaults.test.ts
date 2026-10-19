/**
 * AGENT `retries` comes from `defaults` when the node sets none: mirrors
 * `tests/engine/test_agent_defaults.py`.
 *
 * Ported: the node halves of `test_the_manifest_holds_two_retries_by_default` and
 * `test_the_manifest_holds_the_agent_retries_default`.
 *
 * Changed: `YIAN_AGENT_RETRIES=0` is set with `reload`, which Python's `set_defaults` fixture
 * does.
 *
 * Later (T3-21, T3-22): the manifest halves of the two tests above.
 *
 * Later (T5-11a, T3-22): `test_agent_retries_default_of_zero_fails_on_the_first_bad_reply` and
 * `test_authored_retries_win_over_the_default`. Later (T5-11b, T3-22):
 * `test_agent_max_tool_iterations_default_caps_the_loop` and
 * `test_authored_max_tool_iterations_wins_over_the_default`.
 */

import { afterEach, expect, test } from "bun:test";

import { forget, reload } from "../../src/defaults.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";

afterEach(() => {
  forget();
});

test("an agent holds two retries by default", () => {
  expect(new AgentNode("n").retries).toBe(2);
});

test("an agent holds the agent retries default", () => {
  reload((name) => (name === "YIAN_AGENT_RETRIES" ? "0" : undefined));
  expect(new AgentNode("n").retries).toBe(0);
  expect(new AgentNode("n", { retries: 3 }).retries).toBe(3);
});
