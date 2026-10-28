/**
 * A run-level `llm_config.base_url` goes only with the host key, and the log records it. Provider
 * calls go through the real `chatModelFromConfig` with a recording `send`; `expectHidden` scans for
 * this file's key.
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, describe, expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { ProfileMemory } from "../../src/memory/profile.ts";
import { ReflectiveMemory } from "../../src/memory/reflective.ts";
import { FileMarkdownStore } from "../../src/memory/storage/markdown.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { advance, runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import {
  adaStore,
  chatText,
  expectHidden,
  inOrder,
  NO_FEEDBACK,
  profileMemory,
  type Replier,
  recording,
  rejected,
  responsesText,
  type Sent,
  serving,
  textReply,
} from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-run-base-url-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const SENTINEL = "sk-sentinel-c40a9e17";
const AZURE_V1 = "https://res.openai.azure.com/openai/v1/";
const ELSEWHERE = "https://collector.example.com/v1";

const FLOW = `id: routed
name: routed
nodes:
  agent: {kind: agent, prompt: hi}
output: \${agent.output}
`;
const AZURE = { provider: "openai", model: "gpt-5.4", base_url: AZURE_V1, api_key: SENTINEL };
const VLLM = {
  provider: "vllm",
  model: "served-model",
  base_url: "https://gateway.example.com/v1",
  api_key: SENTINEL,
};

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

/** The effective LLM configs the log at `path` records at the run's start. */
function effective(path: string): Record<string, any> {
  return JSON.parse(readFileSync(path, "utf8").split("\n")[0]!).payload["effective_llm_by_static_path"];
}

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

/** Each agent's key and base_url by model id, from the configs `models` received. */
function seen(models: { configs: readonly Readonly<Record<string, unknown>>[] }) {
  return Object.fromEntries(
    models.configs.map((config) => [config["model"], [config["api_key"] ?? null, config["base_url"] ?? null]]),
  );
}

/** Answer `texts` in call order, repeating the last: a Responses stream on a `/responses` URL, else Chat Completions. */
function answers(...texts: string[]): Replier {
  return (sent, count) => {
    const text = texts[Math.min(count, texts.length) - 1]!;
    return new URL(sent.url).pathname.endsWith("/responses") ? responsesText(text) : chatText(text);
  };
}

/** A model access building real clients over a recording `send` that answers `texts`. */
const fakeClients = (runEnv: Env, ...texts: string[]) => recording(runEnv, answers(...texts));

/** Each build's provider, model, base_url and key, from the configs `models` received. */
function built(models: { configs: readonly Readonly<Record<string, unknown>>[] }) {
  return models.configs.map((config) => [
    config["provider"],
    config["model"],
    config["base_url"] ?? null,
    config["api_key"] ?? null,
  ]);
}

/** Each request's URL and the key it sent (`authorization` or Azure's `api-key`). */
function sentTo(requests: Sent[]) {
  return requests.map((request) => [request.url, request.headers.get("authorization") ?? request.headers.get("api-key")]);
}

test("openai run sends the key to the base url and records the url", async () => {
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, responsesText("ok"));
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, {
    log: path,
    onEvent: (event) => void events.push(event),
    llmConfig: AZURE,
    models,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const request = models.requests[0]!;
  expect(request.url).toBe("https://res.openai.azure.com/openai/v1/responses");
  expect(request.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
  expect(effective(path)).toEqual({
    agent: {
      provider: "openai",
      model: "gpt-5.4",
      base_url: AZURE_V1,
      _host_key: true,
      _connection: { method: "api_key", source: "host", credential_id: null, endpoint: AZURE_V1, store_path: null },
    },
  });
  expectHidden(path, [events, result], [SENTINEL]);
});

test("openai compatible run records the base url beside the marker", async () => {
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, chatText("ok"));
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, { log: path, llmConfig: VLLM, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(models.requests[0]!.url).toBe("https://gateway.example.com/v1/chat/completions");
  expect(effective(path)).toEqual({
    agent: { provider: "vllm", model: "served-model", base_url: "https://gateway.example.com/v1", _host_key: true },
  });
});

test("base url goes only where the key goes", async () => {
  const flow = await loadFlow(
    `id: two
name: two
defs:
  body:
    nodes:
      nested: {kind: agent, prompt: hi, llm_config: {model: nested-model}}
    output: \${nested.output}
nodes:
  inherits: {kind: agent, prompt: hi}
  child: {kind: call, call: body, depends_on: [inherits]}
  other:
    kind: agent
    prompt: hi
    depends_on: [inherits]
    llm_config: {provider: anthropic, model: claude-sonnet-4-5}
  optout:
    kind: agent
    prompt: hi
    depends_on: [other]
    llm_config: {inherit: false, provider: openai, model: other-deployment}
output: \${optout.output}
`,
    { env },
  );
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  const result = await runFlow(flow, {}, { log: path, llmConfig: AZURE, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  // A lower layer may pick another deployment of the key's provider; that agent still sends the
  // key only to the host's base_url.
  expect(seen(models)).toEqual({
    "gpt-5.4": [SENTINEL, AZURE_V1],
    "nested-model": [SENTINEL, AZURE_V1],
    "claude-sonnet-4-5": [null, null],
    "other-deployment": [SENTINEL, AZURE_V1],
  });
  expect("base_url" in effective(path)["other"]).toBe(false);
});

test("a key that marks no agent records no base url", async () => {
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  const flow = await loadFlow(
    FLOW.replace("prompt: hi}", "prompt: hi, llm_config: {provider: anthropic, model: claude-sonnet-4-5}}"),
    { env },
  );
  const result = await runFlow(flow, {}, { log: path, llmConfig: AZURE, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(seen(models)).toEqual({ "claude-sonnet-4-5": [null, null] });
  expect(readFileSync(path, "utf8")).not.toContain(AZURE_V1);
});

test("base url without the layers key is rejected before any request", async () => {
  const keyed = envWith({ OPENAI_API_KEY: "sk-environment-key" });
  const models = fakeModels(keyed);
  const flow = await loadFlow(FLOW, { env: keyed });
  for (const llmConfig of [
    { provider: "openai", model: "gpt-5.4", base_url: AZURE_V1 },
    { model: "gpt-5.4", base_url: AZURE_V1 },
  ]) {
    const error = (await rejected(runFlow(flow, {}, { llmConfig, models, env: keyed }))) as Error;
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain("api_key");
  }
  const error = (await rejected(
    runFlow(flow, {}, { llmConfig: { ...AZURE, base_url: "http://collector.example.com" }, models, env: keyed }),
  )) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("https");
  expect(error.message).not.toContain("collector");
  expect(error.message).not.toContain(SENTINEL);
  expect(models.configs).toEqual([]);
});

const DEF_LEVEL = `id: d
name: d
defs:
  body:
    llm_config: {base_url: "${ELSEWHERE}"}
    nodes:
      nested: {kind: agent, prompt: hi}
    output: \${nested.output}
nodes:
  child: {kind: call, call: body}
output: \${child.output}
`;

describe("a flow file cannot set a base url", () => {
  test.each([
    ["flow", FLOW.replace("nodes:", `llm_config: {base_url: "${ELSEWHERE}"}\nnodes:`)],
    ["node", FLOW.replace("prompt: hi}", `prompt: hi, llm_config: {base_url: "${ELSEWHERE}"}}`)],
    ["node-with-provider", FLOW.replace("prompt: hi}", `prompt: hi, llm_config: {provider: openai, base_url: "${ELSEWHERE}"}}`)],
    ["def", DEF_LEVEL],
  ])("%s", async (_id, source) => {
    const error = (await rejected(loadFlow(source, { env }))) as Error;
    expect(error).toBeInstanceOf(LoadError);
    expect(error.message).toContain("base_url");
    expect(error.message).toContain("run's llm_config");
  });
});

const CHILD = DEF_LEVEL.replace(`    llm_config: {base_url: "${ELSEWHERE}"}\n`, "");

describe("a lower layer base url cannot redirect the host key", () => {
  test.each(["flow", "node", "child-flow", "child-node", "marked-child-node"])("%s", async (layer) => {
    // A flow built by hand skips the loader; the cascade still refuses the URL.
    const models = fakeModels();
    const flow = await loadFlow(layer === "flow" || layer === "node" ? FLOW : CHILD, { env });
    const child =
      layer === "flow" || layer === "node" ? null : ((flow.compiled.nodes.get("child") as CallNode).child as CompiledFlow);
    if (layer === "flow") flow.compiled.flowLlmConfig = { base_url: ELSEWHERE };
    else if (layer === "node") (flow.compiled.nodes.get("agent") as AgentNode).ownLlmConfig = { base_url: ELSEWHERE };
    else if (layer === "child-flow") child!.flowLlmConfig = { base_url: ELSEWHERE };
    else if (layer === "child-node") (child!.nodes.get("nested") as AgentNode).ownLlmConfig = { base_url: ELSEWHERE };
    else (child!.nodes.get("nested") as AgentNode).ownLlmConfig = { _host_key: true, base_url: ELSEWHERE };
    const error = (await rejected(runFlow(flow, {}, { llmConfig: AZURE, models, env }))) as Error;
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain("base_url");
    expect(models.configs).toEqual([]);
  });
});

// ---- provider azure: the endpoint is llm_config.base_url ----

const AZURE_RESOURCE = "https://res.openai.azure.com/";
const AZURE_FLOW = `id: az
name: az
llm_config: {provider: azure, base_url: "${AZURE_RESOURCE}"}
nodes:
  agent: {kind: agent, prompt: hi, llm_config: {model: deploy-a}}
output: \${agent.output}
`;

// Neither endpoint nor deployment variable is read: the deployment is the model, the endpoint the
// base_url.
const azureEnv = envWith({
  AZURE_OPENAI_API_KEY: "sk-azure-environment",
  OPENAI_API_VERSION: "2024-02-01",
  AZURE_OPENAI_ENDPOINT: ELSEWHERE,
  AZURE_OPENAI_DEPLOYMENT_NAME: "env-deployment",
});

/** The Chat Completions URL of Azure deployment `deployment` at `endpoint`. */
const azureUrl = (endpoint: string, deployment: string) =>
  `${endpoint}openai/deployments/${deployment}/chat/completions?api-version=2024-02-01`;

test("a flow file sets the azure endpoint and the run records it", async () => {
  const models = fakeClients(azureEnv, "ok");
  const path = join(fresh(), "run.jsonl");
  const result = await runFlow(await loadFlow(AZURE_FLOW, { env: azureEnv }), {}, { log: path, models, env: azureEnv });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(built(models)).toEqual([["azure", "deploy-a", AZURE_RESOURCE, null]]);
  expect(sentTo(models.requests)).toEqual([[azureUrl(AZURE_RESOURCE, "deploy-a"), "sk-azure-environment"]]);
  expect(effective(path)).toEqual({
    agent: {
      provider: "azure",
      model: "deploy-a",
      base_url: AZURE_RESOURCE,
      _connection: { method: "api_key", source: "environment", credential_id: null, endpoint: null, store_path: null },
    },
  });
});

/** Throw when the agent starts, as a host stopping before the model call. */
function stopBeforeModel(event: RunEvent): void {
  if (event instanceof NodeStarted && event.nodeId === "agent") throw new Error("host stopped before model");
}

test("azure resume uses the recorded endpoint", async () => {
  const path = join(fresh(), "run.jsonl");
  const stopped = await rejected(
    runFlow(await loadFlow(AZURE_FLOW, { env: azureEnv }), {}, { log: path, onEvent: stopBeforeModel, env: azureEnv }),
  );
  expect((stopped as Error).message).toContain("host stopped");
  const moved = envWith({
    AZURE_OPENAI_API_KEY: "sk-azure-environment",
    OPENAI_API_VERSION: "2024-02-01",
    AZURE_OPENAI_ENDPOINT: "https://moved.openai.azure.com/",
    AZURE_OPENAI_DEPLOYMENT_NAME: "env-deployment",
  });
  const models = fakeClients(moved, "ok");
  const result = await advance({ log: path, models, env: moved });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(built(models)).toEqual([["azure", "deploy-a", AZURE_RESOURCE, null]]);
  expect(sentTo(models.requests)).toEqual([[azureUrl(AZURE_RESOURCE, "deploy-a"), "sk-azure-environment"]]);
});

describe("an azure run without base url fails before any request", () => {
  test.each([
    ["flow", FLOW.replace("prompt: hi}", "prompt: hi, llm_config: {provider: azure, model: d}}"), null],
    ["host", FLOW, { provider: "azure", model: "d" }],
  ] as const)("%s", async (_id, source, llmConfig) => {
    const models = fakeClients(azureEnv, "ok");
    const error = (await rejected(
      runFlow(await loadFlow(source, { env: azureEnv }), {}, { llmConfig, models, env: azureEnv }),
    )) as Error;
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toMatch(/Agent 'agent'.*set llm_config\.base_url/);
    expect(models.configs).toEqual([]);
    expect(models.requests).toEqual([]);
  });
});

test("a host layer can give the azure endpoint without a key", async () => {
  const models = fakeClients(azureEnv, "ok");
  const path = join(fresh(), "run.jsonl");
  const result = await runFlow(await loadFlow(FLOW, { env: azureEnv }), {}, {
    log: path,
    llmConfig: { provider: "azure", model: "d", base_url: AZURE_RESOURCE },
    models,
    env: azureEnv,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(built(models)).toEqual([["azure", "d", AZURE_RESOURCE, null]]);
  expect(sentTo(models.requests)).toEqual([[azureUrl(AZURE_RESOURCE, "d"), "sk-azure-environment"]]);
  expect(effective(path)["agent"]["base_url"]).toBe(AZURE_RESOURCE);
});

test("a host azure key needs its own base url", async () => {
  // The flow names an endpoint, but a host key never goes to an authored one.
  const models = fakeClients(azureEnv, "ok");
  const error = (await rejected(
    runFlow(await loadFlow(AZURE_FLOW, { env: azureEnv }), {}, {
      llmConfig: { provider: "azure", api_key: SENTINEL },
      models,
      env: azureEnv,
    }),
  )) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("base_url");
  expect(error.message).not.toContain(SENTINEL);
  expect(models.configs).toEqual([]);
  expect(models.requests).toEqual([]);
});

test("a host azure key goes to the host base url", async () => {
  const models = fakeClients(azureEnv, "ok");
  const path = join(fresh(), "run.jsonl");
  const hostUrl = "https://host.openai.azure.com/";
  const result = await runFlow(await loadFlow(AZURE_FLOW, { env: azureEnv }), {}, {
    log: path,
    llmConfig: { provider: "azure", api_key: SENTINEL, base_url: hostUrl },
    models,
    env: azureEnv,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(built(models)).toEqual([["azure", "deploy-a", hostUrl, SENTINEL]]);
  expect(sentTo(models.requests)).toEqual([[azureUrl(hostUrl, "deploy-a"), SENTINEL]]);
  expect(effective(path)).toEqual({
    agent: { provider: "azure", model: "deploy-a", base_url: hostUrl, _host_key: true },
  });
});

test("an inherited azure endpoint does not reach another provider", async () => {
  const models = fakeClients(azureEnv, "ok");
  const flow = await loadFlow(AZURE_FLOW.replace("{model: deploy-a}", "{provider: openai, model: m}"), { env: azureEnv });
  const error = (await rejected(runFlow(flow, {}, { models, env: azureEnv }))) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("base_url");
  expect(models.configs).toEqual([]);
  expect(models.requests).toEqual([]);
});

describe("an inherited azure endpoint never carries another providers host key", () => {
  test.each(["openai", "vllm"])("%s", async (provider) => {
    const models = fakeClients(azureEnv, "ok");
    const flow = await loadFlow(AZURE_FLOW.replace("{model: deploy-a}", `{provider: ${provider}, model: m}`), {
      env: azureEnv,
    });
    const error = (await rejected(
      runFlow(flow, {}, { llmConfig: { provider, api_key: SENTINEL }, models, env: azureEnv }),
    )) as Error;
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain("base_url");
    expect(error.message).not.toContain(SENTINEL);
    expect(models.configs).toEqual([]);
    expect(models.requests).toEqual([]);
  });
});

test("a def level azure endpoint reaches its body", async () => {
  const models = fakeClients(azureEnv, "ok");
  const path = join(fresh(), "run.jsonl");
  const flow = await loadFlow(
    CHILD.replace("  body:\n", `  body:\n    llm_config: {provider: azure, model: d, base_url: "${AZURE_RESOURCE}"}\n`),
    { env: azureEnv },
  );
  const result = await runFlow(flow, {}, { log: path, models, env: azureEnv });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(built(models)).toEqual([["azure", "d", AZURE_RESOURCE, null]]);
  expect(sentTo(models.requests)).toEqual([[azureUrl(AZURE_RESOURCE, "d"), "sk-azure-environment"]]);
  expect(effective(path)["child/nested"]["base_url"]).toBe(AZURE_RESOURCE);
});

test("memories take a keyless azure endpoint", () => {
  const config = { provider: "azure", model: "d", base_url: AZURE_RESOURCE };
  expect(new ReflectiveMemory({ llmConfig: { ...config }, env })["llmConfig"]).toEqual(config);
  const profile = new ProfileMemory(new FileMarkdownStore(join(fresh(), "ada.md")), { llmConfig: { ...config }, env });
  expect(profile["llmConfig"]).toEqual(config);
  expect([profile["host"].apiKey, profile["host"].baseUrl]).toEqual([null, null]);
});

// ---- resume and backward reuse the recorded base_url ----

/** Leave at `path` the log of a run of `FLOW` the host stopped before the model call. */
async function interruptedRun(path: string, llmConfig: Record<string, unknown>): Promise<void> {
  const error = await rejected(
    runFlow(await loadFlow(FLOW, { env }), {}, { log: path, onEvent: stopBeforeModel, llmConfig, env }),
  );
  expect((error as Error).message).toContain("host stopped");
}

/** Apply `change` to the agent's effective config the log at `path` records at the run's start. */
function rewriteEffective(path: string, change: (config: Record<string, unknown>) => void): void {
  const lines = readFileSync(path, "utf8").split("\n").filter((line) => line !== "");
  const first = JSON.parse(lines[0]!);
  change(first.payload["effective_llm_by_static_path"]["agent"]);
  writeFileSync(path, [JSON.stringify(first), ...lines.slice(1)].join("\n") + "\n");
}

test("resume sends the key to the recorded base url", async () => {
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  await interruptedRun(path, VLLM);
  // The provider's endpoint variable cannot move a recorded agent.
  const moved = envWith({ VLLM_BASE_URL: "http://127.0.0.1:9/v1" });
  const models = recording(moved, chatText("ok"));
  const result = await advance({
    log: path,
    onEvent: (event) => void events.push(event),
    apiKey: SENTINEL,
    models,
    env: moved,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(models.requests[0]!.url).toBe("https://gateway.example.com/v1/chat/completions");
  expect(models.requests[0]!.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
  expectHidden(path, [events, result], [SENTINEL]);
});

test("openai resume sends the key to the recorded azure url", async () => {
  const root = fresh();
  const path = join(root, "run.jsonl");
  await interruptedRun(path, AZURE);
  const keyless = join(root, "keyless.jsonl");
  writeFileSync(keyless, readFileSync(path));
  const moved = envWith({ OPENAI_BASE_URL: ELSEWHERE });
  const models = recording(moved, responsesText("ok"));
  const flow = await loadFlow(FLOW, { env: moved });
  expect((await advance({ log: keyless, models, env: moved })).status).toBe("failed");
  expect(models.requests).toEqual([]);
  const result = await advance({ log: path, apiKey: SENTINEL, models, env: moved });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(models.requests[0]!.url).toBe("https://res.openai.azure.com/openai/v1/responses");
  expect(models.requests[0]!.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
});

test("backward sends the key to the recorded base url", async () => {
  const events: RunEvent[] = [];
  const onEvent = (event: RunEvent) => void events.push(event);
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, inOrder(chatText("ok"), chatText(NO_FEEDBACK)));
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, {
    log: path,
    onEvent,
    learnable: true,
    llmConfig: VLLM,
    models,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect((await result.backward("Be brief.")).status).toBe("succeeded");
  const recovered = await advance({ log: path, onEvent, models, env });
  const outcome = await recovered.backward("Be brief.", { apiKey: SENTINEL });
  expect(outcome.status, outcome.failure?.message).toBe("succeeded");
  expect(models.requests).toHaveLength(3);
  expect(new Set(models.requests.map((request) => request.url))).toEqual(
    new Set(["https://gateway.example.com/v1/chat/completions"]),
  );
  expect(new Set(models.requests.map((request) => request.headers.get("authorization")))).toEqual(
    new Set([`Bearer ${SENTINEL}`]),
  );
  expectHidden(path, [events, result, recovered, outcome], [SENTINEL]);
});

test("openai backward builds its client with the recorded url and key", async () => {
  // The path a host on an Azure OpenAI v1 endpoint uses: the recorded config goes through
  // `legacyConnectionConfig` and the host connection's endpoint.
  const elsewhere = envWith({ OPENAI_API_KEY: "sk-environment-key", OPENAI_BASE_URL: ELSEWHERE });
  const models = fakeClients(elsewhere, "ok", NO_FEEDBACK);
  const path = join(fresh(), "run.jsonl");
  const result = await runFlow(await loadFlow(FLOW, { env: elsewhere }), {}, {
    log: path,
    learnable: true,
    llmConfig: AZURE,
    models,
    env: elsewhere,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect((await result.backward("Be brief.")).status).toBe("succeeded");
  const recovered = await advance({ log: path, models, env: elsewhere });
  const outcome = await recovered.backward("Be brief.", { apiKey: SENTINEL });
  expect(outcome.status, outcome.failure?.message).toBe("succeeded");
  expect(built(models)).toEqual(Array.from({ length: 3 }, () => ["openai", "gpt-5.4", AZURE_V1, SENTINEL]));
  expect(sentTo(models.requests)).toEqual(Array.from({ length: 3 }, () => [`${AZURE_V1}responses`, `Bearer ${SENTINEL}`]));
});

test("a recorded base url without the marker is refused", async () => {
  const models = fakeModels();
  const path = join(fresh(), "run.jsonl");
  await interruptedRun(path, VLLM);
  rewriteEffective(path, (config) => delete config["_host_key"]);
  const error = (await rejected(
    advance({ log: path, apiKey: SENTINEL, models, env }),
  )) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("base_url without a host key");
  expect(models.configs).toEqual([]);
});

describe("a tampered recorded base url fails at resume start", () => {
  test.each(["http://collector.example.com/v1", " https://gateway.example.com/v1", 7])("%p", async (recorded) => {
    const models = fakeModels();
    const path = join(fresh(), "run.jsonl");
    await interruptedRun(path, VLLM);
    rewriteEffective(path, (config) => void (config["base_url"] = recorded));
    const error = (await rejected(
      advance({ log: path, apiKey: SENTINEL, models, env }),
    )) as Error;
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain("effective LLM config for 'agent': base_url");
    expect(error.message).not.toContain(String(recorded).trim());
    expect(error.message).not.toContain(SENTINEL);
    expect(models.configs).toEqual([]);
  });
});

test("a recorded base url must match the host connection", async () => {
  const path = join(fresh(), "run.jsonl");
  await interruptedRun(path, AZURE);
  rewriteEffective(path, (config) => void (config["base_url"] = ELSEWHERE));
  const models = recording(env, responsesText("ok"));
  const result = await advance({ log: path, apiKey: SENTINEL, models, env });
  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("host connection endpoint");
  expect(models.requests).toEqual([]);
});

// ---- memory learners ----

test("reflective memory refuses a base url without its key", () => {
  // Reflective memory refuses api_key, so it can never send one to a base_url.
  const error = thrown(() => new ReflectiveMemory({ llmConfig: { provider: "vllm", model: "m", base_url: ELSEWHERE }, env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("base_url");
});

test("profile memory needs its key beside its base url", () => {
  const store = new FileMarkdownStore(join(fresh(), "ada.md"));
  let error = thrown(() => new ProfileMemory(store, { llmConfig: { provider: "vllm", base_url: ELSEWHERE }, env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key");
  error = thrown(
    () =>
      new ProfileMemory(store, {
        llmConfig: { provider: "vllm", base_url: "http://collector.example.com", api_key: SENTINEL },
        env,
      }),
  );
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("https");
});

const PROFILE_REPLY = "<profile>\n# Ada\n- Prefers brief replies.\n</profile>";

test("profile memory learns at its base url", async () => {
  const root = fresh();
  const keyed = envWith({ VLLM_API_KEY: "sk-environment-key" });
  const flow = await loadFlow(FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"), { env: keyed });
  const models = recording(keyed, inOrder(chatText("ok"), chatText(NO_FEEDBACK), chatText(PROFILE_REPLY)));
  const memory = profileMemory(
    await adaStore(root),
    { provider: "vllm", model: "profiler", base_url: ELSEWHERE, api_key: SENTINEL },
    models,
  );
  const result = await runFlow(flow, {}, {
    learnable: true,
    memory: new Map([["profile", memory]]),
    llmConfig: { provider: "vllm", model: "served-model" },
    models,
    env: keyed,
  });
  const outcome = await result.backward("Be brief.");
  expect(outcome.status, outcome.failure?.message).toBe("succeeded");
  const learner = models.requests.filter((request) => request.body["model"] === "profiler");
  expect(learner.map((request) => request.url)).toEqual([`${ELSEWHERE}/chat/completions`]);
  expect(learner.map((request) => request.headers.get("authorization"))).toEqual([`Bearer ${SENTINEL}`]);
  const others = models.requests.filter((request) => !learner.includes(request));
  expect(new Set(others.map((request) => request.headers.get("authorization")))).toEqual(
    new Set(["Bearer sk-environment-key"]),
  );
});

test("openai profile memory builds its learner with its url and key", async () => {
  const root = fresh();
  const elsewhere = envWith({ OPENAI_API_KEY: "sk-environment-key", OPENAI_BASE_URL: ELSEWHERE });
  const flow = await loadFlow(FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"), { env: elsewhere });
  const store = await adaStore(root);
  const models = fakeClients(elsewhere, "ok", NO_FEEDBACK, PROFILE_REPLY);
  const memory = profileMemory(store, { ...AZURE, model: "profiler" }, models);
  const result = await runFlow(flow, {}, {
    learnable: true,
    memory: new Map([["profile", memory]]),
    llmConfig: { provider: "vllm", model: "served-model" },
    models,
    env: elsewhere,
  });
  const outcome = await result.backward("Be brief.");
  expect(outcome.status, outcome.failure?.message).toBe("succeeded");
  expect(built(models)).toContainEqual(["openai", "profiler", AZURE_V1, SENTINEL]);
  for (const entry of built(models).filter((entry) => entry[1] !== "profiler")) {
    expect(entry.slice(2)).toEqual([null, null]);
  }
  expect((await store.read())[0]).toContain("Prefers brief replies");
});
