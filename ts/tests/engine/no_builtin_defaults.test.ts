/**
 * Mirrors `tests/engine/test_no_builtin_defaults.py`.
 *
 * Ported: `test_each_layer_wins_over_lower_layers` (over `resolveLlmCascade`) and
 * `test_python_api_resolve_model_config_without_model`.
 *
 * Later (T4-8), since they run the flow: `test_no_model_anywhere_fails_before_any_node`,
 * `test_provider_without_model_names_that_provider`,
 * `test_synthesized_agent_without_model_names_its_llm_config`,
 * `test_run_records_the_layer_that_wins` and
 * `test_default_model_without_provider_is_named_only_when_needed`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test } from "bun:test";

import { resolveLlmCascade } from "../../src/compile/llm_cascade.ts";
import { copyGraph } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { ApiKeyCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { connectionSnapshot, resolveModelConfig } from "../../src/llm_clients/connection.ts";

const FIXES = ["llm_config.provider/model", "--provider/--model", "yian auth <provider>"];

// Each test's environment: a private store in a fresh directory and no default provider or
// model, as Python's `no_environment_defaults` leaves it.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-no-defaults-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Load the one-agent flow, with the flow layer `llmConfig` when given. */
function flow(llmConfig: string | null = null): Promise<ComposedFlow> {
  const flowLayer = llmConfig === null ? "" : `llm_config: ${llmConfig}\n`;
  return loadFlow(`id: f\nname: f\n${flowLayer}nodes:\n  write: {kind: agent, prompt: hi}\noutput: \${write.output}\n`);
}

/** Resolve a copy of `loaded` under the host layer `layer` and this test's connection snapshot. */
function resolve(loaded: ComposedFlow, layer: Record<string, unknown> = {}): Record<string, Record<string, unknown>> {
  const snapshot = connectionSnapshot({ env });
  return resolveLlmCascade(copyGraph(loaded.compiled), { ...layer }, {
    defaults: snapshot.defaults,
    providerModels: snapshot.providerModels,
    strayModel: snapshot.strayModel,
  });
}

test.each<[Record<string, string>, Record<string, unknown>, string | null, [string, string]]>([
  [{}, {}, null, ["anthropic", "saved-anth"]],
  [{ YIAN_DEFAULT_PROVIDER: "openai" }, {}, null, ["openai", "saved-oai"]],
  [{ YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "env-m" }, {}, null, ["openai", "env-m"]],
  [
    { YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "env-m" },
    { provider: "anthropic" },
    null,
    ["anthropic", "saved-anth"],
  ],
  [{ YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "env-m" }, { model: "host-m" }, null, ["openai", "host-m"]],
  [
    { YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "env-m" },
    { model: "host-m" },
    "{provider: anthropic, model: flow-m}",
    ["anthropic", "flow-m"],
  ],
])("each layer wins over lower layers %#", async (envVars, layer, flowConfig, expected) => {
  const store = new AuthStore(null, { env });
  await store.save("openai", new ApiKeyCredentials("k1"), "saved-oai");
  await store.save("anthropic", new ApiKeyCredentials("k2"), "saved-anth");
  for (const [name, value] of Object.entries(envVars)) vars.set(name, value);

  const config = resolve(await flow(flowConfig), layer)["write"]!;

  expect([config["provider"], config["model"]]).toEqual(expected);
});

test("python api resolve model config without model", () => {
  let error: unknown;
  try {
    resolveModelConfig({}, { env });
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(RangeError);
  const message = (error as RangeError).message;
  expect(message).toStartWith("No model is set");
  for (const fix of FIXES) expect(message).toContain(fix);
});
