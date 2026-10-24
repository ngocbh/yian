/**
 * The adaptive composer's LLM config label: a port of
 * `tests/engine/test_adaptive_composer_run_start.py`.
 *
 * Ported: `test_replayed_map_error_names_the_gate`, `test_resume_recorded_field_error_names_the_gate`
 * (the label `staticAgentLabels` gives, and the run module's resume check naming it) and
 * `test_composer_label_is_render_only`. Changed: the replayed-map and recorded-field errors are
 * `RangeError`s where Python raises `ValueError`.
 *
 * Ported through `runFlow`, with no `AZURE_OPENAI_ENDPOINT` in the run's `env`:
 * `test_run_start_error_names_the_gate` and `test_plain_agent_run_start_error_is_unchanged`; the
 * run-start errors are `RangeError`s too.
 *
 * Changed: the gate is a TOOL `ask` where Python's is a HUMAN_INPUT node (TS yian has no
 * HUMAN_INPUT kind), so a run passes an ask handler declaring the gate's `proj` input.
 *
 * Covered by `run-agent/engine_adaptive_composer_run_start__an_error_in_a_child_flow_names_the_callsite`,
 * the harvested `test_run_start_error_in_a_child_flow_names_the_gate_and_callsite` (the same
 * flow and input; it pins the whole message).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { applyEffectiveLlmMap, resolveLlmCascade, staticAgentLabels } from "../../src/compile/llm_cascade.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { copyGraph } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { checkRecordedLlmFields, runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { rejected } from "./run_state_fixtures.ts";

const LLM_SITE = "node 'ask' adaptive_questions.llm_config";

const OPENAI = "{provider: openai, model: m}";

const GATE_FLOW = `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      proj: \${input.proj}
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
  const recorded = { ask__compose: { provider: "openai", model: "m", temperature: 0.5 } };
  let error: unknown = null;
  try {
    checkRecordedLlmFields(recorded, staticAgentLabels(flow));
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(RangeError);
  expect((error as RangeError).message).toStartWith(`${LLM_SITE}: `);
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

// ---------- run start ----------

const AZURE = "{provider: azure, model: d}";

const dir = mkdtempSync(join(tmpdir(), "yian-adaptive-run-start-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** The run's ask handler, declaring the gate's context input `proj`; never reached. */
const ASK_TOOL = { handler: () => "unused", inputs: { proj: { type: "string" } } };

/** The message of the `RangeError` running `text` on `inputs` rejects with. */
async function runStartError(text: string, inputs: Record<string, unknown>): Promise<string> {
  const error = await rejected(runFlow(await loadFlow(text), inputs, { env: hostEnv(dir), tools: { ask: ASK_TOOL } }));
  expect(error).toBeInstanceOf(RangeError);
  return (error as RangeError).message;
}

test("a run start error names the gate", async () => {
  const message = await runStartError(GATE_FLOW.replace(OPENAI, AZURE), { proj: "x" });
  expect(message).toStartWith(`${LLM_SITE}: provider azure needs llm_config.base_url`);
  expect(message).not.toContain("ask__compose");
});

test("a plain agent run start error is unchanged", async () => {
  const message = await runStartError(AGENT_FLOW.replace(OPENAI, AZURE), {});
  expect(message).toStartWith("Agent 'w': provider azure");
});
