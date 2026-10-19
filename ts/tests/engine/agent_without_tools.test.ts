/**
 * An AGENT without tools or skills makes a single model call; one with tools runs the loop:
 * mirrors `tests/engine/test_agent_without_tools.py`.
 *
 * Ported: `test_skills_and_steerable_need_no_mode`.
 *
 * Later (T5-11a): `test_text_agent_without_tools_makes_one_unbound_call` and
 * `test_structured_agent_without_tools_makes_one_structured_call`. Later (T5-11b):
 * `test_python_built_agent_with_tools_runs_the_tool_loop`,
 * `test_steer_joins_the_single_request_after_the_prompt` and
 * `test_interrupt_while_streaming_keeps_the_partial_text` (the last two with T3-22).
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";

test("skills and steerable need no mode", () => {
  expect(new AgentNode("n", { prompt: "p", skills: ["review"] }).skills).toEqual(["review"]);
  expect(new AgentNode("n", { prompt: "p", steerable: true }).steerable).toBe(true);
});
