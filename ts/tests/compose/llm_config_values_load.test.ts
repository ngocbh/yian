/**
 * Authored `llm_config` layers: mirrors `tests/compose/test_llm_config_values_load.py`.
 *
 * Ported: `test_a_python_built_agent_keeps_its_literal_config`.
 *
 * TS-only: the `ownLlmConfig`/`llmConfig` split for an empty record, `null`, a `ValueTemplate`
 * and a `Binding`.
 *
 * Later (T3-15), they load a flow: every other test.
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Binding, ValueTemplate } from "../../src/nodes/contracts.ts";

test("a hand-built agent keeps its literal config", () => {
  const config = { model: "m" };
  const agent = new AgentNode("a", { prompt: "p", llmConfig: config });
  expect(agent.llmConfig).toEqual({ model: "m" });
  expect(agent.ownLlmConfig).toEqual({ model: "m" });
  expect(agent.llmConfig).toBe(config);
  expect(agent.ownLlmConfig).not.toBe(config);
});

test("the config split for an empty record, null, a template and a binding", () => {
  const empty = {};
  const fromEmpty = new AgentNode("a", { llmConfig: empty });
  expect(fromEmpty.ownLlmConfig).toEqual({});
  expect(fromEmpty.ownLlmConfig).not.toBe(empty);
  expect(fromEmpty.llmConfig).toBe(empty);

  const omitted = new AgentNode("a");
  expect(omitted.ownLlmConfig).toEqual({});
  expect(omitted.llmConfig).toBeNull();

  const template = new ValueTemplate({ model: "${input.m}" });
  const fromTemplate = new AgentNode("a", { llmConfig: template });
  expect(fromTemplate.ownLlmConfig).toBe(template);
  expect(fromTemplate.llmConfig).toBeNull();

  const binding = new Binding("${input.cfg}");
  const fromBinding = new AgentNode("a", { llmConfig: binding });
  expect(fromBinding.ownLlmConfig).toBe(binding);
  expect(fromBinding.llmConfig).toBeNull();
});
