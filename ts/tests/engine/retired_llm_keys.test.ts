/**
 * Mirrors `tests/engine/test_retired_llm_keys.py`: the first three tests, the node site of
 * `test_a_retired_key_in_yaml_is_a_load_error` (the root flow assembled as the loader assembles
 * it), and the harvested case `engine_retired_llm_keys__temperature_on_a_node` (likewise).
 * Later (T3-22): its flow and def sites. The cascade, run and resume tests go with their modules.
 */

import { expect, test } from "bun:test";

import { LLMConfig, retiredKeyError } from "../../src/llm_clients/config.ts";
import { ValidationError } from "../../src/llm_clients/provider_blocks.ts";
import { assembled, assembleText, loadError, runCase } from "../compose/declarations.ts";

const SENTINEL = "sentinel-3f9a1c";
const SENTINEL_NUMBER = 0.734519;

// Retired key -> [sentinel value, text the message must show].
const RETIRED: Record<string, [unknown, string]> = {
  anthropic_effort: [SENTINEL, "`anthropic: {output_config: {effort: <level>}}`"],
  openai_reasoning_effort: [SENTINEL, "`openai: {reasoning: {effort: <level>}}`"],
  google_thinking_level: [SENTINEL, "`google: {generationConfig: {thinkingConfig: {thinkingLevel: <LEVEL>}}}`"],
  temperature: [SENTINEL_NUMBER, "`<provider>: {temperature: <value>}`"],
};

/** Assert that `message` names `key`, shows its new form and never echoes its value. */
function expectMigration(message: string, key: string): void {
  const [value, form] = RETIRED[key]!;
  expect(message).toContain(`\`${key}\``);
  expect(message).toContain(form);
  expect(message).not.toContain(String(value));
}

test.each(Object.keys(RETIRED))("llm config rejects retired key %s with its new form", (key) => {
  let caught: unknown;
  try {
    new LLMConfig({ [key]: RETIRED[key]![0] });
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(ValidationError);
  expect(caught).toBeInstanceOf(RangeError);
  expectMigration((caught as ValidationError).message, key);
});

test.each([
  ["anthropic", "`anthropic: {temperature: <value>}`"],
  ["qwen-cn", "`qwen-cn: {temperature: <value>}`"],
  ["google", "`google: {generationConfig: {temperature: <value>}}`"],
  ["ollama", "`ollama: {options: {temperature: <value>}}`"],
])("temperature message names the layer provider %s", (provider, form) => {
  expect(retiredKeyError({ provider, temperature: SENTINEL_NUMBER })).toBe(
    `\`temperature\` moved into the provider block: write ${form}`,
  );
});

test("temperature message never echoes an unknown provider", () => {
  const message = retiredKeyError({ provider: SENTINEL, temperature: SENTINEL_NUMBER })!;
  expect(message).not.toContain(SENTINEL);
  expect(message).toContain("`<provider>: {temperature: <value>}`");
});

/** A flow whose agent node's `llm_config:` sets retired `key`. */
function nodeSource(key: string): string {
  return `id: retired
name: retired
nodes:
  agent:
    kind: agent
    prompt: hi
    llm_config: {${key}: ${JSON.stringify(RETIRED[key]![0])}}
output: \${agent.output}
`;
}

test.each(Object.keys(RETIRED))("a retired key in yaml is a load error: node %s", (key) => {
  const { message } = loadError(() => assembleText(nodeSource(key)));
  expect(message).toContain("node 'agent': llm_config:");
  expectMigration(message, key);
});

test.each(["load-agent/engine_retired_llm_keys__temperature_on_a_node"])("case %s", (id) =>
  runCase({ id, call: assembled }),
);
