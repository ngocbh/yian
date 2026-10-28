/**
 * No built-in provider or model: each layer wins over lower ones, and a run without a model fails
 * naming what is missing.
 */

import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
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
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type RunFlowOptions, runFlow } from "../../src/run.ts";
import { NO_ENV, rejected, textReply } from "./run_state_fixtures.ts";

const FIXES = ["llm_config.provider/model", "--provider/--model", "yian auth <provider>"];

// Each test's environment: a private store in a fresh directory and no default provider or
// model.
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

test("resolveModelConfig without a model names the fixes", () => {
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

/** A model builder that fails the test: no model may be built when the run can't start. */
const forbidden = {
  env: NO_ENV,
  createModel: (): never => {
    throw new Error("no model may be built when the run cannot start");
  },
};

/** The `RangeError` a run of `loaded` on `inputs` rejects with, with no model built. */
async function startError(
  loaded: ComposedFlow,
  inputs: Record<string, unknown> = {},
  options: RunFlowOptions = {},
): Promise<RangeError> {
  const log = join(home, "r.jsonl");
  const error = await rejected(runFlow(loaded, inputs, { log, env, models: forbidden, ...options }));
  expect(error).toBeInstanceOf(RangeError);
  expect(existsSync(log)).toBe(false);
  return error as RangeError;
}

test("provider without model names that provider", async () => {
  const message = (await startError(await flow("{provider: anthropic}"))).message;
  expect(message).toContain("Agent 'write'");
  expect(message).toContain("provider 'anthropic'");
  expect(message).toContain("yian auth anthropic");

  const nested = await loadFlow(
    "id: f\nname: f\nllm_config: {provider: anthropic}\n" +
      "defs:\n  sub:\n    nodes:\n      draft: {kind: agent, prompt: hi}\n" +
      "    output: ${draft.output}\n" +
      "nodes:\n  review: {kind: call, call: sub}\noutput: ${review.output}\n",
  );
  expect((await startError(nested)).message).toContain("Agent 'review/draft' has no model for provider 'anthropic'");
});

/** A flow whose TOOL `ask` composes its question with `adaptive_questions`. */
function gateFlow(adaptiveQuestions: string): Promise<ComposedFlow> {
  return loadFlow(
    "id: f\nname: f\ninput: {proj: !str }\nnodes:\n  ask:\n    kind: tool\n    tool_id: ask\n" +
      "    input:\n      proj: ${input.proj}\n" +
      `    adaptive_questions: ${adaptiveQuestions}\n` +
      "output: ${ask.output}\n",
  );
}

test.each([
  ["{prompt: 'Q ${proj}', llm_config: {provider: anthropic}}", "has no model for provider 'anthropic'"],
  ["{prompt: 'Q ${proj}'}", "has no provider or model"],
])("synthesized agent without model names its llm_config: %s", async (adaptiveQuestions, subject) => {
  // The ask handler declares the gate's context input `proj`; it is never reached.
  const tools = { ask: { handler: () => "unused", inputs: { proj: { type: "string" } } } };
  const message = (await startError(await gateFlow(adaptiveQuestions), { proj: "x" }, { tools })).message;
  expect(message).toStartWith(`Node 'ask' adaptive_questions.llm_config ${subject}; `);
  expect(message).not.toContain("ask__compose");
});

test("run records the layer that wins", async () => {
  const store = new AuthStore(null, { env });
  await store.save("openai", new ApiKeyCredentials("k1"), "saved-oai");
  await store.save("anthropic", new ApiKeyCredentials("k2"), "saved-anth");
  vars.set("YIAN_DEFAULT_PROVIDER", "openai");
  vars.set("YIAN_DEFAULT_MODEL", "env-m");
  const model = new ScriptedModel([textReply("ok")]);
  const log = join(home, "r.jsonl");

  const result = await runFlow(await flow(), {}, {
    log,
    env,
    llmConfig: { provider: "anthropic" },
    models: { env, createModel: () => model },
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  const started = JSON.parse(readFileSync(log, "utf8").split("\n")[0]!);
  expect(started.kind).toBe("run_started");
  const recorded = started.payload.effective_llm_by_static_path.write;
  expect([recorded.provider, recorded.model]).toEqual(["anthropic", "saved-anth"]);
});

test("default model without provider is named only when needed", async () => {
  vars.set("YIAN_DEFAULT_MODEL", "m");

  const config = resolve(await flow("{provider: anthropic, model: x}"))["write"]!;
  expect([config["provider"], config["model"]]).toEqual(["anthropic", "x"]);

  const message = (await startError(await flow())).message;
  expect(message).toContain("Agent 'write'");
  expect(message).toContain("YIAN_DEFAULT_MODEL is set but names no provider");
});
