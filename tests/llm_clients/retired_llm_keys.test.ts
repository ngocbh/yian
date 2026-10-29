/**
 * Retired LLM config keys: refused at load in a flow, a def or a node, in a host layer of the
 * cascade, and in the run layer before the log; a finished record holding one still replays.
 */

import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { resolveRunLlm } from "../../src/compile/llm_cascade.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { LLMConfig, retiredKeyError } from "../../src/llm_clients/config.ts";
import { ValidationError } from "../../src/llm_clients/provider_blocks.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { advance, runFlow } from "../../src/run.ts";
import { decodeRecord, encodeRecord } from "../../src/runtime/log.ts";
import { assembled, assembleText, loadError, loadErrorAsync, runCase } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-retired-llm-keys-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

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
function expectMigration(message: string, key: string, provider: string | null = null): void {
  const [value, retiredForm] = RETIRED[key]!;
  const form = key === "temperature" && provider !== null ? `\`${provider}: {temperature: <value>}\`` : retiredForm;
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

/** A flow whose own `llm_config:` sets retired `key`. */
function flowSource(key: string): string {
  return `id: retired
name: retired
llm_config: {${key}: ${JSON.stringify(RETIRED[key]![0])}}
nodes:
  agent: {kind: agent, prompt: hi}
output: \${agent.output}
`;
}

/** A flow whose `body` def's `llm_config:` sets retired `key`. */
function defSource(key: string): string {
  return `id: retired
name: retired
defs:
  body:
    llm_config: {${key}: ${JSON.stringify(RETIRED[key]![0])}}
    nodes:
      agent: {kind: agent, prompt: hi}
    output: \${agent.output}
nodes:
  k: {kind: call, call: body}
output: \${k.output}
`;
}

test.each(
  Object.keys(RETIRED).flatMap((key) => [
    [key, "flow", flowSource(key), "flow: llm_config:"],
    [key, "def", defSource(key), "defs entry 'body': llm_config:"],
  ]),
)("a retired key in yaml is a load error: %s at the %s", async (key, _site, source, where) => {
  const { message } = await loadErrorAsync(loadFlow(source));
  expect(message).toContain(where);
  expectMigration(message, key);
});

const FLOW = `id: plain
name: plain
nodes:
  agent: {kind: agent, prompt: hi}
output: \${agent.output}
`;

test.each(Object.keys(RETIRED))("a retired key in a host layer fails the cascade %s", async (key) => {
  const flow = (await loadFlow(FLOW)).compiled;
  let caught: unknown;
  try {
    resolveRunLlm(flow, { [key]: RETIRED[key]![0] }, { defaults: { provider: "vllm", model: "served-model" } });
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(RangeError);
  expectMigration((caught as RangeError).message, key);
});

const VLLM = { provider: "vllm", model: "served-model" };

test.each(Object.keys(RETIRED))("a retired key in the run layer fails before the log %s", async (key) => {
  const log = join(dir, `run-layer-${key}.jsonl`);
  const error = await rejected(
    runFlow(await loadFlow(FLOW), {}, { log, llmConfig: { ...VLLM, [key]: RETIRED[key]![0] }, env: hostEnv(dir) }),
  );
  expect(error).toBeInstanceOf(RangeError);
  expectMigration((error as RangeError).message, key, "vllm");
  expect(existsSync(log)).toBe(false);
});

/** Rewrite the `run_started` record of `log` so the agent's recorded config sets `temperature`. */
function recordTemperature(log: string): void {
  const lines = readFileSync(log, "utf8").split(/(?<=\n)/);
  const started = JSON.parse(lines[0]!);
  expect(started.kind).toBe("run_started");
  started.payload.effective_llm_by_static_path.agent.temperature = 0.25;
  lines[0] = encodeRecord(decodeRecord(JSON.stringify(started))) + "\n";
  writeFileSync(log, lines.join(""));
}

test("resuming a record with a set retired key fails before appending", async () => {
  const env = hostEnv(dir, { VLLM_API_KEY: "sk-environment-key" });
  const log = join(dir, "stopped.jsonl");
  const stop = new Error("host stopped before model");
  const stopBeforeModel = (event: RunEvent) => {
    if (event instanceof NodeStarted && event.nodeId === "agent") throw stop;
  };
  const caught = await rejected(
    runFlow(await loadFlow(FLOW), {}, {
      log,
      onEvent: stopBeforeModel,
      llmConfig: VLLM,
      models: serving(new ScriptedModel([textReply("ok")])),
      env,
    }),
  );
  expect(caught).toBe(stop);
  recordTemperature(log);
  const before = readFileSync(log);

  const error = await rejected(advance({ log, env }));

  expect(error).toBeInstanceOf(RangeError);
  const message = (error as RangeError).message;
  expect(message).toStartWith("Agent 'agent': ");
  expect(message).toContain("`temperature` moved into the provider block");
  expect(message).toContain("`vllm: {temperature: <value>}`");
  expect(message).not.toContain("0.25");
  expect(readFileSync(log).equals(before)).toBe(true);
});

test("a finished record with a set retired key still replays", async () => {
  const env = hostEnv(dir, { VLLM_API_KEY: "sk-environment-key" });
  const log = join(dir, "finished.jsonl");
  const result = await runFlow(await loadFlow(FLOW), {}, {
    log,
    llmConfig: VLLM,
    models: serving(new ScriptedModel([textReply("ok")])),
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  recordTemperature(log);
  const before = readFileSync(log);

  const replayed = await advance({ log, env });

  expect(replayed.status).toBe("succeeded");
  expect(readFileSync(log).equals(before)).toBe(true);
});

test.each(["load-agent/engine_retired_llm_keys__temperature_on_a_node"])("case %s", (id) =>
  runCase({ id, call: assembled }),
);
