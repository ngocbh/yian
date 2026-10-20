/**
 * The adaptive composer's LLM config label: a port of
 * `tests/engine/test_adaptive_composer_run_start.py`.
 *
 * Ported: the label half of `test_composer_label_is_render_only`.
 *
 * Changed: the flows assemble with `assembleText`, where Python loads them.
 *
 * Later (T3-22): the fingerprint half of `test_composer_label_is_render_only` (the manifest's
 * AGENT entry), `test_replayed_map_error_names_the_gate` and
 * `test_resume_recorded_field_error_names_the_gate` (`llm_cascade`). Later (T4):
 * `test_run_start_error_names_the_gate`,
 * `test_run_start_error_in_a_child_flow_names_the_gate_and_callsite` and
 * `test_plain_agent_run_start_error_is_unchanged` (runs).
 */

import { expect, test } from "bun:test";

import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { assembleText } from "../compose/declarations.ts";

const LLM_SITE = "node 'ask' adaptive_questions.llm_config";

const OPENAI = "{provider: openai, model: m}";

const GATE_FLOW = `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '\${input.proj}'}
    adaptive_questions:
      prompt: "Q \${proj}"
      llm_config: ${OPENAI}
output: \${ask.output}
`;

const AGENT_FLOW = `\
id: f
name: f
nodes:
  w: {kind: agent, prompt: hi, llm_config: ${OPENAI}}
output: \${w.output}
`;

test("composer label is render only", () => {
  const composer = assembleText(GATE_FLOW).flow.compiled.nodes.get("ask__compose") as AgentNode;
  expect(composer.llmConfigLabel).toBe(LLM_SITE);
  const plain = assembleText(AGENT_FLOW).flow.compiled.nodes.get("w") as AgentNode;
  expect(plain.llmConfigLabel).toBeNull();
});
