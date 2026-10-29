/**
 * A run-level `llm_config.api_key` authenticates agents and stays out of every record. Provider
 * calls go through the real `chatModelFromConfig` with a recording `send`, and `expectHidden` scans
 * each value as `inspect` shows it and as `plain`'s JSON.
 */

import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inspect } from "node:util";

import { afterAll, describe, expect, test } from "bun:test";

import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { ProfileMemory } from "../../src/memory/profile.ts";
import { ReflectiveMemory } from "../../src/memory/reflective.ts";
import { FileMarkdownStore } from "../../src/memory/storage/markdown.ts";
import { advance, runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import {
  adaStore,
  chatText,
  chatTool,
  expectHidden,
  inOrder,
  keyEcho401,
  NO_FEEDBACK,
  profileMemory,
  type Reply,
  type Replier,
  recording,
  rejected,
  responsesText,
  type Sent,
  serving,
  textReply,
} from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-run-api-key-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const SENTINEL = "sk-sentinel-5b2e8d41";

const FLOW = `id: keyed
name: keyed
nodes:
  agent: {kind: agent, prompt: hi}
output: \${agent.output}
`;

const VLLM = { provider: "vllm", model: "served-model", api_key: SENTINEL };

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

/** The records of the JSONL log at `path`. */
function records(path: string): { kind: string; payload: Record<string, any> }[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line));
}

/** The effective LLM configs the log at `path` records at the run's start. */
const effective = (path: string) => records(path)[0]!.payload["effective_llm_by_static_path"];

/** The `authorization` header of each request. */
const auth = (requests: Sent[]) => requests.map((request) => request.headers.get("authorization"));

/** Throw when the agent starts, as a host stopping before the model call. */
function stopBeforeModel(event: RunEvent): void {
  if (event instanceof NodeStarted && event.nodeId === "agent") throw new Error("host stopped before model");
}

/** Leave at `path` the log of a run of `FLOW` the host stopped before the model call. */
async function interruptedRun(path: string, llmConfig: Record<string, unknown>, runEnv: Env = env): Promise<void> {
  const flow = await loadFlow(FLOW, { env: runEnv });
  const error = await rejected(runFlow(flow, {}, { log: path, onEvent: stopBeforeModel, llmConfig, env: runEnv }));
  expect((error as Error).message).toContain("host stopped");
}

test("openai compatible run sends the key and records none of it", async () => {
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, chatText("ok"));
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, {
    log: path,
    onEvent: (event) => void events.push(event),
    llmConfig: { provider: "vllm", model: "served-model", api_key: SENTINEL },
    models,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(models.requests[0]!.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
  // A nonsecret marker records that the agent authenticates with a host key.
  expect(effective(path)).toEqual({ agent: { provider: "vllm", model: "served-model", _host_key: true } });
  expectHidden(path, [events, result], [SENTINEL]);
});

test("openai run records a host connection", async () => {
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, responsesText("ok"));
  const llmConfig = { provider: "openai", model: "gpt-5.5", api_key: SENTINEL };
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, {
    log: path,
    onEvent: (event) => void events.push(event),
    llmConfig,
    models,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(models.requests[0]!.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
  const agent = effective(path)["agent"];
  expect(agent["_connection"]["source"]).toBe("host");
  expect(agent["_host_key"]).toBe(true);
  expectHidden(path, [events, result], [SENTINEL]);
});

test("resume sends the key passed again and records none of it", async () => {
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  await interruptedRun(path, VLLM);
  const models = recording(env, chatText("ok"));
  const result = await advance({
    log: path,
    onEvent: (event) => void events.push(event),
    apiKey: SENTINEL,
    models,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(models.requests[0]!.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
  expectHidden(path, [events, result], [SENTINEL]);
});

describe("resume without the key fails instead of reading the environment", () => {
  test.each([
    ["VLLM_API_KEY", VLLM],
    ["ANTHROPIC_API_KEY", { provider: "anthropic", model: "claude-sonnet-4-5", api_key: SENTINEL }],
    ["OPENAI_API_KEY", { provider: "openai", model: "gpt-5.5", api_key: SENTINEL }],
  ])("%s", async (name, llmConfig) => {
    const path = join(fresh(), "run.jsonl");
    await interruptedRun(path, llmConfig);
    const keyed = envWith({ [name]: "sk-environment-key" });
    const models = recording(keyed, chatText("ok"));
    const result = await advance({ log: path, models, env: keyed });
    expect(result.status).toBe("failed");
    expect(result.failure!.message).toContain("api_key");
    expect(models.requests).toEqual([]);
    expect(readFileSync(path, "utf8")).not.toContain(SENTINEL);
  });
});

test("resume ignores a key for a run started without one", async () => {
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  const keyed = envWith({ VLLM_API_KEY: "sk-environment-key" });
  await interruptedRun(path, { provider: "vllm", model: "served-model" }, keyed);
  const models = recording(keyed, chatText("ok"));
  const result = await advance({
    log: path,
    onEvent: (event) => void events.push(event),
    apiKey: SENTINEL,
    models,
    env: keyed,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  // The run recorded no marked agent, so its agents keep their own credentials.
  expect(models.requests[0]!.headers.get("authorization")).toBe("Bearer sk-environment-key");
  expectHidden(path, [events, result], [SENTINEL]);
});

const UNMARKED_FLOWS: [string, string][] = [
  [
    "other-provider",
    `id: other
name: other
nodes:
  agent:
    kind: agent
    prompt: hi
    llm_config: {provider: anthropic, model: claude-sonnet-4-5}
output: \${agent.output}
`,
  ],
  [
    "no-agents",
    `id: plain
name: plain
nodes:
  step:
    kind: code
    output: str
    code: |
      return "ok"
output: \${step.output}
`,
  ],
];

describe("a key that marks no agent does not block replay", () => {
  test.each(UNMARKED_FLOWS)("%s", async (_id, source) => {
    const models = serving(new ScriptedModel([textReply("ok"), textReply(NO_FEEDBACK), textReply(NO_FEEDBACK)]), env);
    const events: RunEvent[] = [];
    const onEvent = (event: RunEvent) => void events.push(event);
    const path = join(fresh(), "run.jsonl");
    const result = await runFlow(await loadFlow(source, { env }), {}, {
      log: path,
      onEvent,
      learnable: true,
      llmConfig: VLLM,
      models,
      env,
    });
    expect(result.status, result.failure?.message).toBe("succeeded");
    const retained = await result.backward("Be brief.");
    expect(retained.status, retained.failure?.message).toBe("succeeded");
    const recovered = await advance({ log: path, onEvent, apiKey: SENTINEL, models, env });
    expect(recovered.status, recovered.failure?.message).toBe("succeeded");
    const outcome = await recovered.backward("Be brief.");
    expect(outcome.status, outcome.failure?.message).toBe("succeeded");
    expect(models.configs.map((config) => config["api_key"] ?? null)).toEqual(models.configs.map(() => null));
    expect(readFileSync(path, "utf8")).not.toContain("_host_key");
    expectHidden(path, [events, result, retained, recovered, outcome], [SENTINEL]);
  });
});

test("backward sends the key and records none of it", async () => {
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
  // The result keeps the run's key in memory for its own backward calls.
  const retained = await result.backward("Be brief.");
  expect(retained.status, retained.failure?.message).toBe("succeeded");
  // After a restart, the key comes back through advance or backward.
  for (const [apiKey, key] of [
    [SENTINEL, null],
    [null, SENTINEL],
  ]) {
    const recovered = await advance({ log: path, onEvent, apiKey, models, env });
    const outcome = await recovered.backward("Be brief.", { apiKey: key });
    expect(outcome.status, outcome.failure?.message).toBe("succeeded");
    expectHidden(path, [recovered, outcome], [SENTINEL]);
  }
  expect(models.requests).toHaveLength(4);
  expect(new Set(auth(models.requests))).toEqual(new Set([`Bearer ${SENTINEL}`]));
  expect(events.length).toBeGreaterThan(0);
  expectHidden(path, [events, result, retained], [SENTINEL]);
});

test("backward without the key fails clearly", async () => {
  const events: RunEvent[] = [];
  const onEvent = (event: RunEvent) => void events.push(event);
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, inOrder(chatText("ok"), chatText(NO_FEEDBACK)));
  await runFlow(await loadFlow(FLOW, { env }), {}, { log: path, onEvent, learnable: true, llmConfig: VLLM, models, env });
  const keyed = envWith({ VLLM_API_KEY: "sk-environment-key" });
  const keyedModels = recording(keyed, chatText(NO_FEEDBACK));
  const recovered = await advance({
    log: path,
    onEvent,
    models: keyedModels,
    env: keyed,
  });
  const outcome = await recovered.backward("Be brief.");
  expect(outcome.status).toBe("failed");
  expect(inspect(outcome.failure, { depth: null })).toContain("api_key");
  expect(models.requests).toHaveLength(1);
  expect(keyedModels.requests).toEqual([]);
  expectHidden(path, [events, recovered, outcome], [SENTINEL]);
});

test("invalid config with a key fails before the run without echoing it", async () => {
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  const error = (await rejected(
    runFlow(await loadFlow(FLOW, { env }), {}, {
      log: path,
      onEvent: (event) => void events.push(event),
      llmConfig: { provider: "vllm", model: "served-model", api_key: SENTINEL, vllm: { temperature: "hot" } },
      env,
    }),
  )) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("temperature");
  expect(events).toEqual([]);
  expect(existsSync(path)).toBe(false);
  expectHidden(null, [error.message, inspect(error.cause)], [SENTINEL]);
});

test("key goes only to agents of its provider", async () => {
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
    llm_config: {inherit: false, provider: vllm, model: other-model}
output: \${optout.output}
`,
    { env },
  );
  const models = serving(new ScriptedModel(Array.from({ length: 4 }, () => textReply("ok"))), env);
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  const result = await runFlow(flow, {}, {
    log: path,
    onEvent: (event) => void events.push(event),
    llmConfig: { provider: "vllm", model: "served-model", api_key: SENTINEL },
    models,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expectHidden(path, [events, result], [SENTINEL]);
  const keys = Object.fromEntries(models.configs.map((config) => [config["model"], config["api_key"] ?? null]));
  expect(keys).toEqual({
    "served-model": SENTINEL,
    "nested-model": SENTINEL,
    "claude-sonnet-4-5": null,
    "other-model": SENTINEL,
  });
});

test("an empty key is rejected", async () => {
  const keyed = envWith({ VLLM_API_KEY: "sk-environment-key" });
  let error = (await rejected(
    runFlow(await loadFlow(FLOW, { env: keyed }), {}, { llmConfig: { ...VLLM, api_key: "" }, env: keyed }),
  )) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key");
  const path = join(fresh(), "run.jsonl");
  await interruptedRun(path, VLLM, keyed);
  error = (await rejected(advance({ log: path, apiKey: "", env: keyed }))) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key");
});

test("a key with a newline is rejected without echoing it", async () => {
  const key = "sk-dummy-key\nzz";
  let error = (await rejected(
    runFlow(await loadFlow(FLOW, { env }), {}, { llmConfig: { ...VLLM, api_key: key }, env }),
  )) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key must not contain a control character");
  expect(error.message).not.toContain("sk-dummy-key");
  const path = join(fresh(), "run.jsonl");
  await interruptedRun(path, VLLM, env);
  error = (await rejected(advance({ log: path, apiKey: key, env }))) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key must not contain a control character");
  expect(error.message).not.toContain("sk-dummy-key");
});

test("authored key is rejected without echoing it", async () => {
  const nodeLevel = FLOW.replace("prompt: hi}", `prompt: hi, llm_config: {api_key: ${SENTINEL}}}`);
  const flowLevel = FLOW.replace("nodes:", `llm_config: {api_key: ${SENTINEL}}\nnodes:`);
  const defLevel = `id: d
name: d
defs:
  body:
    llm_config: {api_key: ${SENTINEL}}
    nodes:
      nested: {kind: agent, prompt: hi}
    output: \${nested.output}
nodes:
  child: {kind: call, call: body}
output: \${child.output}
`;
  for (const source of [nodeLevel, flowLevel, defLevel]) {
    const error = (await rejected(loadFlow(source, { env }))) as Error;
    expect(error).toBeInstanceOf(LoadError);
    expect(error.message).toContain("api_key");
    expect(error.message).not.toContain(SENTINEL);
  }
});

test("host key marker is not an authored option", async () => {
  let error = (await rejected(
    runFlow(await loadFlow(FLOW, { env }), {}, { llmConfig: { provider: "vllm", model: "m", _host_key: true }, env }),
  )) as Error;
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("_host_key");
  error = (await rejected(loadFlow(FLOW.replace("nodes:", "llm_config: {_host_key: true}\nnodes:"), { env }))) as Error;
  expect(error).toBeInstanceOf(LoadError);
  expect(error.message).toContain("_host_key");
});

test("reflective memory refuses a key it would record", () => {
  const error = thrown(() => new ReflectiveMemory({ llmConfig: { provider: "vllm", model: "m", api_key: SENTINEL }, env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key");
  expect(error.message).not.toContain(SENTINEL);
});

test("profile memory needs a provider with its key", () => {
  const store = new FileMarkdownStore(join(fresh(), "ada.md"));
  for (const llmConfig of [{ api_key: SENTINEL }, { model: "profiler", api_key: SENTINEL }] as Record<string, string>[]) {
    const error = thrown(() => new ProfileMemory(store, { llmConfig, env }));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain("provider");
    expect(error.message).not.toContain(SENTINEL);
  }
  const error = thrown(() => new ProfileMemory(store, { llmConfig: { provider: "vllm", api_key: "" }, env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key");
  const memory = new ProfileMemory(store, { llmConfig: { provider: "vllm", api_key: SENTINEL }, env });
  expect(inspect(memory, { depth: null })).not.toContain(SENTINEL);
});

/** Return the error `fn` throws; fails the test when it returns. */
function thrown(fn: () => unknown): Error {
  try {
    fn();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

/** The text of each file under `root`, recursively. */
function writtenTexts(root: string): string[] {
  return readdirSync(root, { recursive: true })
    .map((name) => join(root, String(name)))
    .filter((path) => statSync(path).isFile())
    .map((path) => readFileSync(path, "latin1"));
}

test("profile memory learns with its key and records none of it", async () => {
  const root = fresh();
  const keyed = envWith({ VLLM_API_KEY: "sk-environment-key" });
  const flow = await loadFlow(FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"), { env: keyed });
  const store = await adaStore(root);
  const models = recording(
    keyed,
    inOrder(chatText("ok"), chatText(NO_FEEDBACK), chatText("<profile>\n# Ada\n- Prefers brief replies.\n</profile>")),
  );
  const memory = profileMemory(store, { provider: "vllm", model: "profiler", api_key: SENTINEL }, models);
  const events: RunEvent[] = [];
  const path = join(root, "run.jsonl");
  const result = await runFlow(flow, {}, {
    log: path,
    onEvent: (event) => void events.push(event),
    learnable: true,
    memory: new Map([["profile", memory]]),
    llmConfig: { provider: "vllm", model: "served-model" },
    models,
    env: keyed,
  });
  const outcome = await result.backward("Be brief.");
  expect(outcome.status, outcome.failure?.message).toBe("succeeded");
  const learner = models.requests.filter((request) => request.body["model"] === "profiler");
  expect(auth(learner)).toEqual([`Bearer ${SENTINEL}`]);
  expect((await memory.read())[0]).toBe("# Ada\n- Prefers brief replies.\n");
  expectHidden(path, [events, result, outcome], [SENTINEL]);
  for (const written of writtenTexts(join(root, "profile"))) expect(written).not.toContain(SENTINEL);
});

const REJECTED_KEY = "sk-sentinel-401-zqxw";
const REJECTED_FRAGMENTS = [REJECTED_KEY, REJECTED_KEY.slice(0, 8), `*${REJECTED_KEY.slice(-4)}`];

/** Answer the first `accept` requests with `replies` in order, then echo the key masked, as OpenAI's 401 does. */
function rejecting(accept = 0, ...replies: Reply[]): Replier {
  const ok = inOrder(...(replies.length > 0 ? replies : [chatText("ok")]));
  return (sent, count) => (count <= accept ? ok(sent, count) : keyEcho401(REJECTED_KEY));
}

test("a provider 401 records no part of the key", async () => {
  const events: RunEvent[] = [];
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, rejecting());
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, {
    log: path,
    onEvent: (event) => void events.push(event),
    llmConfig: { ...VLLM, api_key: REJECTED_KEY },
    models,
    env,
  });
  expect(result.status).toBe("failed");
  expect(models.requests[0]!.headers.get("authorization")).toBe(`Bearer ${REJECTED_KEY}`);
  expect(readFileSync(path, "utf8")).toContain("Incorrect API key provided: [REDACTED]");
  expectHidden(path, [events, result], REJECTED_FRAGMENTS);
});

test("a provider 401 in backward records no part of the key", async () => {
  const path = join(fresh(), "run.jsonl");
  const models = recording(env, rejecting(1));
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, {
    log: path,
    learnable: true,
    llmConfig: { ...VLLM, api_key: REJECTED_KEY },
    models,
    env,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const outcome = await result.backward("Be brief.");
  expect(outcome.status).toBe("failed");
  expect(models.requests).toHaveLength(2);
  expect(readFileSync(path, "utf8")).toContain("[REDACTED]");
  expectHidden(path, [result, outcome], REJECTED_FRAGMENTS);
});

test("a profile learner 401 records no part of the key", async () => {
  const root = fresh();
  const keyed = envWith({ VLLM_API_KEY: "sk-environment-key" });
  const flow = await loadFlow(FLOW.replace("nodes:", "long_term_memory: profile\nnodes:"), { env: keyed });
  const store = await adaStore(root);
  const models = recording(keyed, rejecting(2, chatText("ok"), chatText(NO_FEEDBACK)));
  const memory = profileMemory(store, { provider: "vllm", model: "profiler", api_key: REJECTED_KEY }, models);
  const events: RunEvent[] = [];
  const path = join(root, "run.jsonl");
  const result = await runFlow(flow, {}, {
    log: path,
    onEvent: (event) => void events.push(event),
    learnable: true,
    memory: new Map([["profile", memory]]),
    llmConfig: { provider: "vllm", model: "served-model" },
    models,
    env: keyed,
  });
  const outcome = await result.backward("Be brief.");
  expect(outcome.status).toBe("failed");
  expect(models.requests.at(-1)!.body["model"]).toBe("profiler");
  expectHidden(path, [events, result, outcome], REJECTED_FRAGMENTS);
});

const MAP_FLOW = `id: fan
name: fan
input:
  values: !list(str)
defs:
  body:
    input:
      value: !str
    nodes:
      inner:
        kind: agent
        input:
          value: \${input.value}
        prompt: \${value}
    output: \${inner.output}
nodes:
  fan:
    kind: map
    call: body
    over: \${input.values}
    input:
      value: \${item}
output: \${fan.output}
`;

const LOOP_FLOW = `id: turns
name: turns
defs:
  body:
    input:
      text: !str
    nodes:
      inner:
        kind: agent
        input:
          text: \${input.text}
        output:
          text: str
        prompt: \${text}
    output: \${inner.output}
nodes:
  turns:
    kind: loop
    call: body
    input:
      text: start
    times: 2
output: \${turns.output.text}
`;

const GROWN: [string, string, Record<string, unknown>][] = [
  ["map", MAP_FLOW, { values: ["a", "b"] }],
  ["loop", LOOP_FLOW, {}],
];

/** Answer record-schema requests with a call to their first tool and others with `replies` in order. */
function recordCalls(...replies: Reply[]): Replier {
  const texts = inOrder(...(replies.length > 0 ? replies : [chatText("ok")]));
  let text = 0;
  return (sent) => {
    const tools = sent.body["tools"] as { function: { name: string } }[] | undefined;
    if (tools === undefined) return texts(sent, ++text);
    return chatTool(tools[0]!.function.name, JSON.stringify({ text: "ok" }));
  };
}

/** Throw when an AGENT inside a grown body starts. */
function stopInGrownBody(event: RunEvent): void {
  if (event instanceof NodeStarted && event.nodeId.includes("inner")) throw new Error("host stopped inside a grown body");
}

describe("grown agents resume with the key and record none of it", () => {
  test.each(GROWN)("%s", async (_kind, source, inputs) => {
    const events: RunEvent[] = [];
    const path = join(fresh(), "run.jsonl");
    const error = await rejected(
      runFlow(await loadFlow(source, { env }), inputs, { log: path, onEvent: stopInGrownBody, llmConfig: VLLM, env }),
    );
    expect((error as Error).message).toContain("grown body");
    const keyed = envWith({ VLLM_API_KEY: "sk-environment-key" });
    const models = recording(keyed, recordCalls());
    const result = await advance({
      log: path,
      onEvent: (event) => void events.push(event),
      apiKey: SENTINEL,
      models,
      env: keyed,
    });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(models.requests).toHaveLength(2);
    expect(new Set(auth(models.requests))).toEqual(new Set([`Bearer ${SENTINEL}`]));
    expectHidden(path, [events, result], [SENTINEL]);
  });
});

describe("grown agents backward with the key and record none of it", () => {
  test.each(GROWN)("%s", async (_kind, source, inputs) => {
    const events: RunEvent[] = [];
    const onEvent = (event: RunEvent) => void events.push(event);
    const path = join(fresh(), "run.jsonl");
    const models = recording(env, recordCalls(chatText(NO_FEEDBACK)));
    const result = await runFlow(await loadFlow(source, { env }), inputs, {
      log: path,
      onEvent,
      learnable: true,
      llmConfig: VLLM,
      models,
      env,
    });
    expect(result.status, result.failure?.message).toBe("succeeded");
    const recovered = await advance({ log: path, onEvent, apiKey: SENTINEL, models, env });
    const outcome = await recovered.backward("Review.");
    expect(outcome.status, outcome.failure?.message).toBe("succeeded");
    // Without the key, the grown agents fail instead of reading the environment.
    const keyed = envWith({ VLLM_API_KEY: "sk-environment-key" });
    const keyedModels = recording(keyed, recordCalls(chatText(NO_FEEDBACK)));
    const keyless = await (
      await advance({ log: path, models: keyedModels, env: keyed })
    ).backward("Review.");
    expect(keyless.status).toBe("failed");
    expect(keyedModels.requests).toEqual([]);
    expect(models.requests.length).toBeGreaterThan(2);
    expect(new Set(auth(models.requests))).toEqual(new Set([`Bearer ${SENTINEL}`]));
    expectHidden(path, [events, result, recovered, outcome, keyless], [SENTINEL]);
  });
});
