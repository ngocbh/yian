/**
 * `runFlow` applies a host `llmConfig`: the config `createModel` gets equals `resolveModelConfig`
 * under the same environment.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { resolveModelConfig } from "../../src/llm_clients/connection.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { serving, textReply } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-llm-cascade-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

test("run flow applies cli llm config", async () => {
  const env = hostEnv(dir);
  const models = serving(new ScriptedModel([textReply("ok")]), env);
  const text = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi}\noutput: ${a.output}\n";
  const loaded = await loadFlow(text, { env });
  await runFlow(loaded, {}, { env, models, llmConfig: { provider: "openai", model: "gpt-5.5" } });
  expect(models.configs).toEqual([resolveModelConfig({ provider: "openai", model: "gpt-5.5" }, { env })]);
});

test("a flow's api reaches its agents and is kept in their effective config", async () => {
  const env = hostEnv(dir);
  const models = serving(new ScriptedModel([textReply("ok")]), env);
  const text =
    "id: f\nname: f\nllm_config: {provider: vllm, api: chat}\n" +
    "nodes:\n  a: {kind: agent, prompt: hi, llm_config: {model: gpt-5-codex}}\noutput: ${a.output}\n";
  await runFlow(await loadFlow(text, { env }), {}, { env, models });
  expect(models.configs.map((config) => [config["model"], config["api"]])).toEqual([["gpt-5-codex", "chat"]]);
});
