/**
 * The adaptive composer's LLM config label: a port of
 * `tests/engine/test_adaptive_composer_run_start.py`.
 *
 * Ported: `test_replayed_map_error_names_the_gate` and `test_composer_label_is_render_only`; of
 * `test_resume_recorded_field_error_names_the_gate`, the `static_agent_labels` half (the label
 * the check names). Changed: the replayed-map errors are `RangeError`s where Python raises
 * `ValueError`.
 *
 * Later (T4-8): the `_check_recorded_llm_fields` half of
 * `test_resume_recorded_field_error_names_the_gate` (the run module's resume check),
 * `test_run_start_error_names_the_gate`,
 * `test_run_start_error_in_a_child_flow_names_the_gate_and_callsite` and
 * `test_plain_agent_run_start_error_is_unchanged` (runs).
 */

import { expect, test } from "bun:test";

import { applyEffectiveLlmMap, resolveLlmCascade, staticAgentLabels } from "../../src/compile/llm_cascade.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { copyGraph } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";

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

/** Replay `text`'s resolved map with a base_url added at `path`; return the error message. */
async function replayWithBaseUrl(text: string, path: string): Promise<string> {
  const flow = (await loadFlow(text)).compiled;
  const effective = resolveLlmCascade(copyGraph(flow), {}, { defaults: { provider: "openai", model: "m" } });
  effective[path]!["base_url"] = "https://x.example.com/v1";
  try {
    applyEffectiveLlmMap(copyGraph(flow), effective);
  } catch (error) {
    expect(error).toBeInstanceOf(RangeError);
    return (error as RangeError).message;
  }
  throw new Error("expected a RangeError");
}

test("a replayed map error names the gate", async () => {
  const gate = await replayWithBaseUrl(GATE_FLOW, "ask__compose");
  expect(gate).toStartWith(`effective LLM config for ${LLM_SITE} has a base_url without a host key`);
  const agent = await replayWithBaseUrl(AGENT_FLOW, "w");
  expect(agent).toStartWith("effective LLM config for 'w' has a base_url");
});

test("a resumed record names the gate by its static label", async () => {
  const flow = (await loadFlow(GATE_FLOW)).compiled;
  expect(staticAgentLabels(flow)).toEqual(new Map([["ask__compose", LLM_SITE]]));
});

test("composer label is render only", async () => {
  const flow = (await loadFlow(GATE_FLOW)).compiled;
  const composer = flow.nodes.get("ask__compose") as AgentNode;
  expect(composer.llmConfigLabel).toBe(LLM_SITE);
  const unlabeled = copyGraph(flow);
  (unlabeled.nodes.get("ask__compose") as AgentNode).llmConfigLabel = null;
  expect(flowFingerprint(unlabeled)).toBe(flowFingerprint(flow));
  const plain = (await loadFlow(AGENT_FLOW)).compiled.nodes.get("w") as AgentNode;
  expect(plain.llmConfigLabel).toBeNull();
});
