/**
 * A port of `tests/engine/test_llm_cascade_run.py`.
 *
 * Ported: `test_run_flow_applies_cli_llm_config`, with a scripted model from `serving` where
 * Python patches `model_from_config`; the config `createModel` gets is compared with
 * `resolveModelConfig` under the same environment.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { resolveModelConfig } from "../../src/llm_clients/connection.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { serving, textReply } from "./run_state_fixtures.ts";

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
