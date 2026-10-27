/**
 * A port of `tests/engine/test_llm_cascade_e2e.py`: a 2-level flow run through `runFlow` with a
 * host `llmConfig` layer. The config each agent builds its model from reflects the whole authored
 * ladder (node, flow, parent, host), and an `inherit: false` agent skips those authored outer
 * layers while keeping the process provider and model defaults.
 *
 * Ported: `test_cascade_end_to_end_with_cli_layer`, with a scripted model from `serving` where
 * Python patches `model_from_config`; the `inherit: false` config is compared with
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

const dir = mkdtempSync(join(tmpdir(), "yian-llm-cascade-e2e-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// With no saved anthropic login, the agent reads ANTHROPIC_API_KEY.
const ENVIRONMENT = {
  method: "api_key",
  source: "environment",
  credential_id: null,
  endpoint: null,
  store_path: null,
};

// The top flow sets the provider and an anthropic temperature, and calls `sub`. Inside `sub`,
// `inner` overrides only `model` (so it fills the provider and its anthropic block from the flow
// and the host fields, merged per block field), while `solo` skips the authored outer layers with
// `inherit: false` (its own record already supplies both process-defaulted selection fields).
const FLOW = `
id: top
name: top
llm_config: {provider: anthropic, anthropic: {temperature: 0.2}}
defs:
  sub:
    nodes:
      inner: {kind: agent, prompt: hi, llm_config: {model: claude-opus-4-8}}
      solo:
        kind: agent
        depends_on: [inner]
        prompt: hi
        llm_config: {provider: openai, model: gpt-5.5, inherit: false}
    output: \${solo.output}
nodes:
  c: {kind: call, call: sub}
output: \${c.output}
`;

test("cascade end to end with cli layer", async () => {
  const env = hostEnv(dir);
  const models = serving(new ScriptedModel([textReply("ok"), textReply("ok")]), env);
  const loaded = await loadFlow(FLOW, { env });
  const result = await runFlow(loaded, {}, {
    env,
    models,
    llmConfig: { anthropic: { output_config: { effort: "low" } } },
  });
  expect(result.status, result.failure?.message).toBe("succeeded");

  // Keyed by the model field (unique per agent here) so the checks don't depend on call order.
  const byModel = Object.fromEntries(models.configs.map((config) => [config["model"], config]));
  // inner: its own model, the flow's provider and temperature, the host's effort, one anthropic block.
  expect(byModel["claude-opus-4-8"]).toEqual({
    model: "claude-opus-4-8",
    provider: "anthropic",
    anthropic: { temperature: 0.2, output_config: { effort: "low" } },
    _connection: ENVIRONMENT,
  });
  // solo: inherit: false, its own complete selection, no authored flow or host layers.
  expect(byModel["gpt-5.5"]).toEqual(resolveModelConfig({ provider: "openai", model: "gpt-5.5" }, { env }));
});
