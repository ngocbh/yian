/**
 * A port of `tests/engine/test_run_connection_settings.py`: the connection settings
 * (`organization`, `project`, `headers`) come only from the host layer and travel with the host key.
 *
 * Ported: every test but the one already ported below, the parametrized ones whole. Python's
 * `_fake_models` and `_recording_models` read the configs `serving` receives; the backward rows go
 * through the real `chatModelFromConfig` with a recording `send` (`recording`), where Python routes
 * the OpenAI client through `httpx.MockTransport`. The secret scan is `expectHidden`. Each run's
 * environment is the file's (`hostEnv`) plus the variables a row sets.
 *
 * Changed: refused configs and settings are `RangeError`s (`ValidationError` is one) and refused
 * flow files `LoadError`s (Python's `ValueError`s); a host stopping a run from `onEvent` makes the
 * run reject with its error. TS has no `SecretStr`: header values reach the model config and
 * `popHostCredentials`' result as plain text, the result keeping them in private fields that
 * neither `inspect` nor JSON shows, so the `SecretStr` instance checks become those scans, and a
 * frozen `HostConnection` refuses an assignment with a `TypeError` (Python's
 * `FrozenInstanceError`). The backward rows' `repr` checks of Python's `_backward_handle` have no
 * counterpart: the TS handle is a module-private entry no value exposes; the results holding it
 * are scanned instead. `test_a_python_built_flow_layer_cannot_set_a_connection_setting` sets the
 * layer on the loaded flow's `AgentNode` (Python's `own_llm_config`).
 *
 * Already ported: `test_reflective_memory_refuses_connection_settings`, in the constructor test of
 * `reflective_memory.test.ts`.
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inspect } from "node:util";

import { afterAll, describe, expect, test } from "bun:test";

import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type HostConnection, popHostCredentials } from "../../src/llm_clients/config.ts";
import { ProfileMemory } from "../../src/memory/profile.ts";
import { FileMarkdownStore } from "../../src/memory/storage/markdown.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { checkRecordedLlmFields, resumeFlow, runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import {
  adaStore,
  chatText,
  expectHidden,
  inOrder,
  NO_FEEDBACK,
  recording,
  rejected,
  type Sent,
  serving,
  textReply,
} from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-run-connection-settings-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const SENTINEL = "sk-sentinel-5e0c27b4";
const HEADER_VALUE = "gw-token-91c3d0aa";
const AZURE_RESOURCE = "https://res.openai.azure.com/";

const FLOW = `id: routed
name: routed
nodes:
  agent: {kind: agent, prompt: hi}
output: \${agent.output}
`;
const SETTINGS: Record<string, string> = {
  headers: `{headers: {X-Gateway: ${HEADER_VALUE}}}`,
  organization: "{organization: org-abc}",
  project: "{project: proj-abc}",
};
const KEYED = { provider: "openai", model: "gpt-5.4", api_key: SENTINEL };

let files = 0;

/** A fresh directory in the file's temp dir. */
function fresh(): string {
  const path = join(dir, String(files++));
  mkdirSync(path);
  return path;
}

/** The file's environment with `vars` set. */
const envWith = (vars: Record<string, string> = {}): Env => hostEnv(dir, vars);
const env = envWith();

/** Return the error `fn` throws; fails the test when it returns. */
function thrown(fn: () => unknown): Error {
  try {
    fn();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

/** A model access whose builds answer `ok` and record their configs. */
const fakeModels = (runEnv: Env = env) =>
  serving(new ScriptedModel(Array.from({ length: 8 }, () => textReply("ok"))), runEnv);

/** Each agent's organization, project and headers by model id, from the configs `models` received. */
function seen(models: { configs: readonly Readonly<Record<string, unknown>>[] }) {
  return Object.fromEntries(
    models.configs.map((config) => [
      config["model"],
      [config["organization"] ?? null, config["project"] ?? null, config["headers"] ?? null],
    ]),
  );
}

const flowLevel = (layer: string) => FLOW.replace("nodes:", `llm_config: ${layer}\nnodes:`);

const nodeLevel = (layer: string) => FLOW.replace("prompt: hi}", `prompt: hi, llm_config: ${layer}}`);

const defsLevel = (layer: string) => `id: d
name: d
defs:
  body:
    llm_config: ${layer}
    nodes:
      nested: {kind: agent, prompt: hi}
    output: \${nested.output}
nodes:
  child: {kind: call, call: body}
output: \${child.output}
`;

const adaptive = (layer: string) => `id: f
name: f
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: "Ask about the project"
      llm_config: ${layer}
output: \${ask.output}
`;

describe("test_a_flow_file_cannot_set_a_connection_setting", () => {
  for (const [id, site, prefix] of [
    ["flow", flowLevel, "flow: llm_config."],
    ["defs", defsLevel, "defs entry 'body': llm_config."],
    ["node", nodeLevel, "node 'agent': llm_config."],
    ["adaptive", adaptive, "node 'ask' adaptive_questions.llm_config."],
  ] as const) {
    for (const field of Object.keys(SETTINGS).sort()) {
      test(`${id}, ${field}`, async () => {
        // The setting is refused even beside a provider that takes it.
        const error = await rejected(loadFlow(site(SETTINGS[field]!.replace("{", "{provider: openai, ")), { env }));
        expect(error).toBeInstanceOf(LoadError);
        const message = (error as Error).message;
        expect(message).toContain(`${prefix}${field} `);
        expect(message).toContain("run's llm_config");
        expect(message).toContain("yian auth");
        expect(message).not.toContain(HEADER_VALUE);
      });
    }
  }
});

describe("test_a_python_built_flow_layer_cannot_set_a_connection_setting", () => {
  for (const field of ["organization", "project", "headers"]) {
    test(field, async () => {
      // A flow built in code skips the loader; the cascade still refuses the setting.
      const models = fakeModels();
      const flow = await loadFlow(FLOW, { env });
      const value = field === "headers" ? { "X-Gateway": HEADER_VALUE } : "org-abc";
      (flow.compiled.nodes.get("agent") as AgentNode).ownLlmConfig = { [field]: value };
      const error = await rejected(runFlow(flow, {}, { llmConfig: KEYED, models, env }));
      expect(error).toBeInstanceOf(RangeError);
      expect((error as Error).message).toContain(`llm_config.${field} `);
      expect((error as Error).message).not.toContain(HEADER_VALUE);
      expect(models.configs).toEqual([]);
    });
  }
});

/** A host connection's fields, its key and header values included. */
function fields(host: HostConnection) {
  return {
    apiKey: host.apiKey,
    baseUrl: host.baseUrl,
    organization: host.organization,
    project: host.project,
    headers: host.headers,
  };
}

test("pop host credentials takes the settings with the key", () => {
  const layer: Record<string, unknown> = {
    ...KEYED,
    organization: "org-abc",
    project: "proj-abc",
    headers: { "X-Gateway": HEADER_VALUE },
  };
  const host = popHostCredentials(layer);
  expect(layer).toEqual({ provider: "openai", model: "gpt-5.4" });
  expect(host.apiKey).toBe(SENTINEL);
  expect([host.baseUrl, host.organization, host.project]).toEqual([null, "org-abc", "proj-abc"]);
  expect(host.headers).toEqual({ "X-Gateway": HEADER_VALUE });
  for (const shown of [inspect(host, { depth: null }), JSON.stringify(host)]) {
    expect(shown).not.toContain(HEADER_VALUE);
    expect(shown).not.toContain(SENTINEL);
  }
  expect(() => {
    (host as { organization: string | null }).organization = "org-other";
  }).toThrow(TypeError);
});

test("pop host credentials without settings returns none for them", () => {
  const none = { apiKey: null, baseUrl: null, organization: null, project: null, headers: null };
  expect(fields(popHostCredentials({ provider: "openai" }))).toEqual(none);
  const host = popHostCredentials({ provider: "anthropic", api_key: SENTINEL, headers: {} });
  expect([host.organization, host.project, host.headers]).toEqual([null, null, null]);
});

describe("test_pop_host_credentials_refuses_settings_without_the_layers_key", () => {
  for (const [id, layer] of [
    ["organization", { provider: "openai", organization: "org-abc" }],
    ["project", { provider: "openai", project: "proj-abc" }],
    ["headers", { provider: "openai", headers: { "X-Gateway": HEADER_VALUE } }],
    ["no-provider", { organization: "org-abc" }],
    ["keyless-azure", { provider: "azure", base_url: AZURE_RESOURCE, headers: { "X-Gateway": HEADER_VALUE } }],
  ] as const) {
    test(id, () => {
      const error = thrown(() => popHostCredentials({ ...layer }));
      expect(error).toBeInstanceOf(RangeError);
      expect(error.message).toContain("api_key");
      expect(error.message).not.toContain(HEADER_VALUE);
    });
  }
});

test("pop host credentials checks the settings against the provider", () => {
  let error = thrown(() => popHostCredentials({ provider: "anthropic", api_key: SENTINEL, project: "p" }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("provider openai");
  error = thrown(() => popHostCredentials({ ...KEYED, headers: { Authorization: HEADER_VALUE } }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("Authorization");
  expect(error.message).not.toContain(HEADER_VALUE);
});

test("settings without a host key fail before any node runs", async () => {
  const keyed = envWith({ OPENAI_API_KEY: "sk-environment-key" });
  const models = fakeModels(keyed);
  for (const setting of [
    { organization: "org-abc" },
    { project: "proj-abc" },
    { headers: { "X-Gateway": HEADER_VALUE } },
  ]) {
    const llmConfig = { provider: "openai", model: "gpt-5.4", ...setting };
    const error = await rejected(runFlow(await loadFlow(FLOW, { env: keyed }), {}, { llmConfig, models, env: keyed }));
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain("api_key");
    expect((error as Error).message).not.toContain(HEADER_VALUE);
  }
  expect(models.configs).toEqual([]);
});

test("a run with settings beside its key records no header value", async () => {
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  const llmConfig = { ...KEYED, organization: "org-abc", headers: { "X-Gateway": HEADER_VALUE } };
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, { log: path, llmConfig, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(models.configs.map((config) => config["model"])).toEqual(["gpt-5.4"]);
  expect(readFileSync(path, "utf8")).not.toContain(HEADER_VALUE);
  expect(readFileSync(path, "utf8")).not.toContain(SENTINEL);
});

test("profile memory keeps the settings beside its key", () => {
  const store = new FileMarkdownStore(join(fresh(), "ada.md"));
  const memory = new ProfileMemory(store, {
    llmConfig: { ...KEYED, organization: "org-abc", project: "proj-abc", headers: { "X-Gateway": HEADER_VALUE } },
    env,
  });
  expect(memory["llmConfig"]).toEqual({ provider: "openai", model: "gpt-5.4" });
  const host = memory["host"];
  expect([host.organization, host.project]).toEqual(["org-abc", "proj-abc"]);
  expect(host.headers!["X-Gateway"]).toBe(HEADER_VALUE);
  expect(host.apiKey).toBe(SENTINEL);
});

test("profile memory needs its key beside the settings", () => {
  const store = new FileMarkdownStore(join(fresh(), "ada.md"));
  const error = thrown(() => new ProfileMemory(store, { llmConfig: { provider: "openai", organization: "org-abc" }, env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key");
});

// ---- the run records the settings without secrets and binds the values ----

const OTHER_VALUE = "gw-trace-0b7d41e2";
const HOST_CONNECTION = {
  method: "api_key",
  source: "host",
  credential_id: null,
  endpoint: "https://api.openai.com/v1",
  store_path: null,
};
const TWO_PROVIDERS = `id: two
name: two
defs:
  body:
    nodes:
      nested: {kind: agent, prompt: hi, llm_config: {model: nested-model}}
    output: \${nested.output}
nodes:
  agent: {kind: agent, prompt: hi}
  child: {kind: call, call: body, depends_on: [agent]}
  other:
    kind: agent
    prompt: hi
    depends_on: [agent]
    llm_config: {provider: anthropic, model: claude-sonnet-4-5}
output: \${other.output}
`;
const HEADERS = { "X-Gateway": HEADER_VALUE, "X-Trace": OTHER_VALUE };
const ALL_SETTINGS = { ...KEYED, organization: "org-abc", project: "proj-abc", headers: HEADERS };

/** The effective LLM configs the log at `path` records at the run's start. */
function effective(path: string): Record<string, any> {
  return JSON.parse(readFileSync(path, "utf8").split("\n")[0]!).payload["effective_llm_by_static_path"];
}

test("the run records the settings on the key agents only", async () => {
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  const result = await runFlow(await loadFlow(TWO_PROVIDERS, { env }), {}, {
    log: path,
    llmConfig: ALL_SETTINGS,
    models,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const recorded = {
    provider: "openai",
    organization: "org-abc",
    project: "proj-abc",
    _header_names: ["X-Gateway", "X-Trace"],
    _host_key: true,
    _connection: HOST_CONNECTION,
  };
  expect(effective(path)["agent"]).toEqual({ ...recorded, model: "gpt-5.4" });
  expect(effective(path)["child/nested"]).toEqual({ ...recorded, model: "nested-model" });
  // Another provider's agent gets none of the key's settings.
  expect(effective(path)["other"]).toEqual({
    provider: "anthropic",
    model: "claude-sonnet-4-5",
    _connection: { method: "api_key", source: "environment", credential_id: null, endpoint: null, store_path: null },
  });
  const log = readFileSync(path, "utf8");
  for (const secret of [HEADER_VALUE, OTHER_VALUE, SENTINEL]) expect(log).not.toContain(secret);
  // The in-memory node configs carry the values next to the key.
  expect(seen(models)).toEqual({
    "gpt-5.4": ["org-abc", "proj-abc", HEADERS],
    "nested-model": ["org-abc", "proj-abc", HEADERS],
    "claude-sonnet-4-5": [null, null, null],
  });
});

test("header names are recorded sorted for any keyed provider", async () => {
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  const llmConfig = {
    provider: "anthropic",
    model: "claude-sonnet-4-5",
    api_key: SENTINEL,
    headers: { "x-trace": OTHER_VALUE, "X-Gateway": HEADER_VALUE },
  };
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, { log: path, llmConfig, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(effective(path)["agent"]).toEqual({
    provider: "anthropic",
    model: "claude-sonnet-4-5",
    _header_names: ["X-Gateway", "x-trace"],
    _host_key: true,
  });
  expect(seen(models)["claude-sonnet-4-5"]).toEqual([null, null, { "x-trace": OTHER_VALUE, "X-Gateway": HEADER_VALUE }]);
  expect(readFileSync(path, "utf8")).not.toContain(HEADER_VALUE);
});

test("a key that marks no agent records no settings", async () => {
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  const flow = await loadFlow(
    FLOW.replace("prompt: hi}", "prompt: hi, llm_config: {provider: anthropic, model: claude-sonnet-4-5}}"),
    { env },
  );
  const result = await runFlow(flow, {}, { log: path, llmConfig: ALL_SETTINGS, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(seen(models)).toEqual({ "claude-sonnet-4-5": [null, null, null] });
  const agent = effective(path)["agent"];
  for (const key of ["organization", "project", "_header_names", "_host_key"]) expect(key in agent).toBe(false);
  expect(readFileSync(path, "utf8")).not.toContain("org-abc");
  expect(readFileSync(path, "utf8")).not.toContain(HEADER_VALUE);
});

test("a finished run with recorded settings replays without the key", async () => {
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  const first = await runFlow(await loadFlow(FLOW, { env }), {}, { log: path, llmConfig: ALL_SETTINGS, models, env });
  expect(first.status, first.failure?.message).toBe("succeeded");
  const replayed = await resumeFlow(await loadFlow(FLOW, { env }), { log: path, models, env });
  expect(replayed.status, replayed.failure?.message).toBe("succeeded");
  expect(replayed.output).toBe(first.output);
});

/** Throw when the agent starts, as a host stopping before the model call. */
function stopBeforeModel(event: RunEvent): void {
  if (event instanceof NodeStarted && event.nodeId === "agent") throw new Error("host stopped before model");
}

test("resume with the key keeps the recorded organization", async () => {
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  const llmConfig = { ...KEYED, organization: "org-abc", project: "proj-abc" };
  const stopped = await rejected(
    runFlow(await loadFlow(FLOW, { env }), {}, { log: path, onEvent: stopBeforeModel, llmConfig, models, env }),
  );
  expect((stopped as Error).message).toContain("host stopped");
  const result = await resumeFlow(await loadFlow(FLOW, { env }), { log: path, apiKey: SENTINEL, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(seen(models)).toEqual({ "gpt-5.4": ["org-abc", "proj-abc", null] });
});

/** Leave at a fresh path the log of a run the host stopped before the model call; return the path. */
async function interruptedHeaderRun(
  models: ReturnType<typeof fakeModels>,
  llmConfig: Record<string, unknown> = ALL_SETTINGS,
  runEnv: Env = env,
): Promise<string> {
  const path = join(fresh(), "run.jsonl");
  const stopped = await rejected(
    runFlow(await loadFlow(FLOW, { env: runEnv }), {}, {
      log: path,
      onEvent: stopBeforeModel,
      llmConfig,
      models,
      env: runEnv,
    }),
  );
  expect((stopped as Error).message).toContain("host stopped");
  return path;
}

test("resume sends the given headers to the key agents", async () => {
  const models = fakeModels();
  const path = await interruptedHeaderRun(models);
  // Names compare without case, as they do in HTTP.
  const headers = { "x-gateway": HEADER_VALUE, "X-TRACE": OTHER_VALUE };
  const result = await resumeFlow(await loadFlow(FLOW, { env }), { log: path, apiKey: SENTINEL, headers, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(seen(models)).toEqual({ "gpt-5.4": ["org-abc", "proj-abc", { "x-gateway": HEADER_VALUE, "X-TRACE": OTHER_VALUE }] });
  expect(readFileSync(path, "utf8")).not.toContain(HEADER_VALUE);
});

describe("test_resume_needs_exactly_the_recorded_header_names", () => {
  for (const [headers, named] of [
    [null, "missing X-Gateway, X-Trace"],
    [{ "X-Gateway": HEADER_VALUE }, "missing X-Trace"],
    [{ ...HEADERS, "X-Extra": "extra-3f1c" }, "not recorded X-Extra"],
  ] as const) {
    test(named, async () => {
      const models = fakeModels();
      const path = await interruptedHeaderRun(models);
      const before = readFileSync(path);
      const error = await rejected(
        resumeFlow(await loadFlow(FLOW, { env }), { log: path, apiKey: SENTINEL, headers, models, env }),
      );
      expect(error).toBeInstanceOf(RangeError);
      const message = (error as Error).message;
      expect(message.endsWith(named)).toBe(true);
      for (const value of [HEADER_VALUE, OTHER_VALUE, "extra-3f1c", SENTINEL]) expect(message).not.toContain(value);
      expect(readFileSync(path)).toEqual(before);
      expect(seen(models)).toEqual({});
    });
  }
});

test("resume reports extra header names beside missing ones", async () => {
  const models = fakeModels();
  const path = await interruptedHeaderRun(models);
  const error = await rejected(
    resumeFlow(await loadFlow(FLOW, { env }), {
      log: path,
      apiKey: SENTINEL,
      headers: { "X-Gateway": HEADER_VALUE, "X-Other": OTHER_VALUE },
      models,
      env,
    }),
  );
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("missing X-Trace; not recorded X-Other");
});

test("resume without recorded headers refuses given ones", async () => {
  const models = fakeModels();
  const path = await interruptedHeaderRun(models, KEYED);
  const before = readFileSync(path);
  const error = await rejected(
    resumeFlow(await loadFlow(FLOW, { env }), {
      log: path,
      apiKey: SENTINEL,
      headers: { "X-Gateway": HEADER_VALUE },
      models,
      env,
    }),
  );
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("not recorded X-Gateway");
  expect(readFileSync(path)).toEqual(before);
});

test("resume checks the given header values", async () => {
  const models = fakeModels();
  const path = await interruptedHeaderRun(models);
  const before = readFileSync(path);
  const error = await rejected(
    resumeFlow(await loadFlow(FLOW, { env }), {
      log: path,
      apiKey: SENTINEL,
      headers: { "X-Gateway": HEADER_VALUE, "X-Trace": "bad\nvalue-77" },
      models,
      env,
    }),
  );
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("header X-Trace value");
  expect((error as Error).message).not.toContain("value-77");
  expect(readFileSync(path)).toEqual(before);
});

test("resume ignores headers on a run with no key agents", async () => {
  const keyed = envWith({ OPENAI_API_KEY: "sk-env" });
  const models = fakeModels(keyed);
  const path = await interruptedHeaderRun(models, { provider: "openai", model: "gpt-5.4" }, keyed);
  const result = await resumeFlow(await loadFlow(FLOW, { env: keyed }), {
    log: path,
    apiKey: SENTINEL,
    headers: { "X-Anything": HEADER_VALUE },
    models,
    env: keyed,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(seen(models)).toEqual({ "gpt-5.4": [null, null, null] });
});

test("a finished run checks the given header names", async () => {
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  const first = await runFlow(await loadFlow(FLOW, { env }), {}, {
    log: path,
    llmConfig: ALL_SETTINGS,
    learnable: true,
    models,
    env,
  });
  expect(first.status, first.failure?.message).toBe("succeeded");
  const before = readFileSync(path);
  let error = await rejected(
    resumeFlow(await loadFlow(FLOW, { env }), {
      log: path,
      apiKey: SENTINEL,
      headers: { "X-Gateway": HEADER_VALUE },
      models,
      env,
    }),
  );
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("missing X-Trace");
  error = await rejected(resumeFlow(await loadFlow(FLOW, { env }), { log: path, apiKey: SENTINEL, models, env }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain("X-Gateway, X-Trace");
  const replayed = await resumeFlow(await loadFlow(FLOW, { env }), {
    log: path,
    apiKey: SENTINEL,
    headers: HEADERS,
    models,
    env,
  });
  expect(replayed.output).toBe(first.output);
  expect(readFileSync(path)).toEqual(before);
});

test("recorded header names pass the resume field check", () => {
  const marked = {
    provider: "openai",
    model: "gpt-5.4",
    organization: "org-abc",
    _header_names: ["X-Gateway"],
    _host_key: true,
    _connection: HOST_CONNECTION,
  };
  checkRecordedLlmFields({ agent: marked }, new Map());
  const { _host_key: _, ...unmarked } = marked;
  const error = thrown(() => checkRecordedLlmFields({ agent: unmarked }, new Map()));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("Agent 'agent'");
});

test("profile memory settings reach its model", async () => {
  const root = fresh();
  const flow = await loadFlow(FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"), { env });
  const store = await adaStore(root);
  const built: Readonly<Record<string, unknown>>[] = [];
  const memory = new ProfileMemory(store, {
    llmConfig: { ...ALL_SETTINGS, model: "profiler" },
    env,
    createModel: (config) => {
      built.push(config);
      return new ScriptedModel([textReply("<profile>\n# Ada\n- Prefers brief replies.\n</profile>")]);
    },
  });
  const models = serving(new ScriptedModel([textReply("ok"), textReply(NO_FEEDBACK)]), env);
  const result = await runFlow(flow, {}, {
    learnable: true,
    memory: new Map([["profile", memory]]),
    llmConfig: { provider: "anthropic", model: "claude-sonnet-4-5" },
    models,
    env,
  });
  const outcome = await result.backward("Be brief.");
  expect(outcome.status, outcome.failure?.message).toBe("succeeded");
  expect(built).toHaveLength(1);
  const [config] = built;
  expect([config!["organization"], config!["project"]]).toEqual(["org-abc", "proj-abc"]);
  expect(config!["api_key"]).toBe(SENTINEL);
  expect(config!["headers"]).toEqual(HEADERS);
});

// ---- backward keeps and takes the host's headers ----

const VLLM_HEADERS = { provider: "vllm", model: "served-model", api_key: SENTINEL, headers: HEADERS };
const SECRETS = [SENTINEL, HEADER_VALUE, OTHER_VALUE];

/** The distinct `[authorization, X-Gateway, X-Trace]` of `requests`. */
function sentHeaders(requests: Sent[]) {
  const all = requests.map((request) =>
    JSON.stringify(["authorization", "x-gateway", "x-trace"].map((name) => request.headers.get(name))),
  );
  return [...new Set(all)].map((entry) => JSON.parse(entry));
}

test("backward sends the headers the result kept", async () => {
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, inOrder(chatText("ok"), chatText(NO_FEEDBACK)));
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, {
    log: path,
    onEvent: (event) => void events.push(event),
    learnable: true,
    llmConfig: VLLM_HEADERS,
    models,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const outcome = await result.backward("Be brief.");
  expect(outcome.status, outcome.failure?.message).toBe("succeeded");
  const recovered = await resumeFlow(await loadFlow(FLOW, { env }), {
    log: path,
    apiKey: SENTINEL,
    headers: HEADERS,
    models,
    env,
  });
  const resumed = await recovered.backward("Be brief.");
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(models.requests).toHaveLength(3);
  expect(sentHeaders(models.requests)).toEqual([[`Bearer ${SENTINEL}`, HEADER_VALUE, OTHER_VALUE]]);
  expectHidden(path, [events, result, outcome, recovered, resumed], SECRETS);
});

test("backward sends the headers it is given after resume", async () => {
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, inOrder(chatText("ok"), chatText(NO_FEEDBACK)));
  const first = await runFlow(await loadFlow(FLOW, { env }), {}, {
    log: path,
    learnable: true,
    llmConfig: VLLM_HEADERS,
    models,
    env,
  });
  expect(first.status, first.failure?.message).toBe("succeeded");
  const recovered = await resumeFlow(await loadFlow(FLOW, { env }), { log: path, models, env });
  // Names compare without case.
  const headers = { "x-gateway": HEADER_VALUE, "X-TRACE": OTHER_VALUE };
  const outcome = await recovered.backward("Be brief.", { apiKey: SENTINEL, headers });
  expect(outcome.status, outcome.failure?.message).toBe("succeeded");
  expect(models.requests).toHaveLength(2);
  expect(sentHeaders(models.requests)).toEqual([[`Bearer ${SENTINEL}`, HEADER_VALUE, OTHER_VALUE]]);
  expectHidden(path, [recovered, outcome], SECRETS);
});

test("backward without the headers fails naming them", async () => {
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, inOrder(chatText("ok"), chatText(NO_FEEDBACK)));
  await runFlow(await loadFlow(FLOW, { env }), {}, { log: path, learnable: true, llmConfig: VLLM_HEADERS, models, env });
  const recovered = await resumeFlow(await loadFlow(FLOW, { env }), { log: path, models, env });
  const outcome = await recovered.backward("Be brief.", { apiKey: SENTINEL });
  expect(outcome.status).toBe("failed");
  expect(outcome.failure!.causeType).toBe("AuthError");
  expect(outcome.failure!.message).toContain("connection headers X-Gateway, X-Trace");
  expect(outcome.failure!.message).toContain("RunResult.backward");
  expect(models.requests).toHaveLength(1);
  expectHidden(path, [recovered, outcome], SECRETS);
});

describe("test_backward_checks_the_given_headers_before_appending", () => {
  for (const [headers, message] of [
    [{ "X-Gateway": HEADER_VALUE }, "missing X-Trace"],
    [{ ...HEADERS, "X-Extra": "extra-3f1c" }, "not recorded X-Extra"],
    [{ "X-Gateway": HEADER_VALUE, "X-Trace": "bad\nvalue-77" }, "header X-Trace value"],
  ] as const) {
    test(message, async () => {
      const models = fakeModels();
      const path = join(fresh(), "run.jsonl");
      const first = await runFlow(await loadFlow(FLOW, { env }), {}, {
        log: path,
        learnable: true,
        llmConfig: ALL_SETTINGS,
        models,
        env,
      });
      expect(first.status, first.failure?.message).toBe("succeeded");
      models.configs.length = 0;
      const recovered = await resumeFlow(await loadFlow(FLOW, { env }), { log: path, models, env });
      const before = readFileSync(path);
      const error = await rejected(recovered.backward("Be brief.", { apiKey: SENTINEL, headers }));
      expect(error).toBeInstanceOf(RangeError);
      expect((error as Error).message).toContain(message);
      for (const value of [HEADER_VALUE, OTHER_VALUE, "extra-3f1c", "value-77", SENTINEL]) {
        expect((error as Error).message).not.toContain(value);
      }
      expect(readFileSync(path)).toEqual(before);
      expect(seen(models)).toEqual({});
    });
  }
});
