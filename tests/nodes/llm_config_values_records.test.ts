/** The config a call-time `llm_config` agent resolves, its node step, and the resume check. */

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { main } from "../../src/cli/index.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AuthError } from "../../src/llm_clients/auth/common.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import {
  checkRecordedLlmFields,
  type AdvanceOptions,
  advance,
  type RunFlowOptions,
  runFlow,
} from "../../src/run.ts";
import { FakeInput, FakeOutput } from "../cli/terminal_both_runtimes.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { crashAfter, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-llm-values-records-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// The default provider and model.
const CONFTEST = { YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "gpt-5.5" };
const env = hostEnv(dir, CONFTEST);

let files = 0;

/** A fresh path in the file's temp dir. */
const fresh = (name: string) => join(dir, `${files++}-${name}`);

const HOST = {
  provider: "openai",
  api_key: "sk-host-secret",
  headers: { "X-Tenant": "tenant-secret" },
};

const PICKED = `id: f
name: f
llm_config: {provider: openai}
nodes:
  pick: {kind: code, output: str, code: "return 'gpt-picked'"}
  chat:
    kind: agent
    prompt: Go.
    llm_config: {model: '\${pick.output}'}
output: \${chat.output}
`;

/** A model answering `texts` in order (one `ok` by default), served under `runEnv`. */
function fakeModel(texts: string[] = ["ok"], runEnv: Env = env) {
  const model = new ScriptedModel(texts.map(textReply));
  return { model, models: serving(model, runEnv) };
}

/** Run `text` with no inputs in the file's environment and a model answering `ok` unless `options` says otherwise. */
async function run(text: string, options: RunFlowOptions = {}) {
  const runEnv = options.env ?? env;
  return runFlow(await loadFlow(text, { env: runEnv }), {}, { models: fakeModel(["ok"], runEnv).models, ...options, env: runEnv });
}

/** Resume `log` of `text` in the file's environment. */
async function resume(text: string, log: string, options: Omit<AdvanceOptions, "log"> = {}) {
  const runEnv = options.env ?? env;
  return advance({ ...options, log, env: runEnv });
}

/** The records of the JSONL log at `path`. */
function records(path: string): { kind: string; payload: Record<string, any> }[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line));
}

/** Whether `record` is an `llm_config` step: the config a call-time agent recorded. */
const isConfigStep = (record: { kind: string; payload: Record<string, any> }) =>
  record.kind === "node_step" && record.payload["step"]["kind"] === "llm_config";

/** The `llm_config` steps `nodeId` recorded. */
function steps(path: string, nodeId = "chat"): Record<string, any>[] {
  return records(path)
    .filter((record) => isConfigStep(record) && record.payload["node_id"] === nodeId)
    .map((record) => record.payload["step"]);
}

test("each call time start logs its config without secrets", async () => {
  const log = fresh("run.jsonl");
  const { models } = fakeModel();
  expect((await run(PICKED, { log, llmConfig: HOST, models })).status).toBe("succeeded");
  const all = steps(log);
  expect(all).toHaveLength(1);
  const config = all[0]!["effective_llm_config"];
  expect([config["provider"], config["model"]]).toEqual(["openai", "gpt-picked"]);
  expect(config["_host_key"]).toBe(true);
  expect(config["_header_names"]).toEqual(["X-Tenant"]);
  expect("api_key" in config).toBe(false);
  expect("headers" in config).toBe(false);
  const data = readFileSync(log, "utf8");
  expect(data).not.toContain("sk-host-secret");
  expect(data).not.toContain("tenant-secret");
  expect(models.configs.map((built) => built["model"])).toEqual(["gpt-picked"]);
});

test("an agent resolved at run start logs no llm_config step", async () => {
  const text = `id: f
name: f
input: {m: !str }
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: {provider: openai, model: '\${input.m}'}
output: \${chat.output}
`;
  const log = fresh("run.jsonl");
  const flow = await loadFlow(text, { env });
  expect((await runFlow(flow, { m: "gpt-x" }, { log, env, models: fakeModel().models })).status).toBe("succeeded");
  expect(records(log).filter(isConfigStep)).toEqual([]);
});

/**
 * The record count through `nodeId`'s step in an uncut run of `text` (models answering `texts`,
 * under `runEnv`): a log cut there has run the agent's model call and recorded nothing after it.
 */
async function throughStep(text: string, nodeId: string, texts: string[], runEnv: Env = env): Promise<number> {
  const log = fresh("uncut.jsonl");
  expect((await run(text, { log, env: runEnv, models: fakeModel(texts, runEnv).models })).status).toBe("succeeded");
  const at = records(log).findIndex((record) => record.kind === "node_step" && record.payload["node_id"] === nodeId);
  expect(at).toBeGreaterThan(-1);
  return at + 1;
}

/** Run `PICKED` to a fresh log until the host dies in `chat`'s model call; return the log. */
async function crashedInChat(): Promise<string> {
  const log = fresh("run.jsonl");
  const cut = await throughStep(PICKED, "chat", ["ok"]);
  await crashAfter(cut, () => run(PICKED, { log }));
  expect(steps(log)).toHaveLength(1);
  return log;
}

test("resume recomputes the recorded config", async () => {
  const log = await crashedInChat();
  const { models } = fakeModel();
  const result = await resume(PICKED, log, { models });
  expect(result.status).toBe("succeeded");
  const [first, second] = steps(log);
  expect(first).toEqual(second!);
  expect(models.configs.map((config) => config["model"])).toEqual(["gpt-picked"]);
});

test("resume compares a recorded config holding a float", async () => {
  const text = PICKED.replace("llm_config: {model: '\${pick.output}'}", "llm_config: {model: '\${pick.output}', openai: {temperature: 0.5}}");
  expect(text).toContain("temperature: 0.5");
  const log = fresh("float.jsonl");
  const cut = await throughStep(text, "chat", ["ok"]);
  await crashAfter(cut, () => run(text, { log }));
  const result = await resume(text, log, { models: fakeModel().models });
  expect(result.status, result.failure?.message).toBe("succeeded");
});

test("resume refuses a recorded config that differs", async () => {
  const log = await crashedInChat();
  const lines = readFileSync(log, "utf8").split("\n").filter((line) => line !== "");
  for (const [index, line] of lines.entries()) {
    const record = JSON.parse(line);
    if (isConfigStep(record)) {
      record.payload.step.effective_llm_config.model = "gpt-tampered";
      lines[index] = JSON.stringify(record);
    }
  }
  writeFileSync(log, `${lines.join("\n")}\n`);
  const before = records(log).length;
  const { models } = fakeModel();
  const error = await rejected(resume(PICKED, log, { models }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toMatch(/^node 'chat' llm_config: recorded config differs from the recomputed one/);
  expect(records(log).slice(before).map((record) => record.kind)).not.toContain("node_failed");
  expect(models.configs).toEqual([]);
});

const TWO_CALL_TIME = `id: f
name: f
llm_config: {provider: openai}
nodes:
  pick: {kind: code, output: str, code: "return 'anthropic'"}
  chat:
    kind: agent
    prompt: Go.
    llm_config: {provider: '\${pick.output}', model: claude-picked}
  later:
    kind: agent
    prompt: Go.
    llm_config: {model: !str '\${chat.output}'}
output: \${later.output}
`;

/** Resume `log`; return the map and host its key check received, appending nothing. */
async function resumeChecks(log: string, flowText: string, options: Omit<AdvanceOptions, "log"> = {}) {
  const seen: [Record<string, any>, Record<string, unknown> | null][] = [];
  const stop = new Error("stop");
  const before = readFileSync(log, "utf8");
  const error = await rejected(
    resume(flowText, log, {
      ...options,
      checkLlmConfigs: (effective, { callTimeHost }) => {
        seen.push([effective, callTimeHost]);
        throw stop;
      },
    }),
  );
  expect(error).toBe(stop);
  expect(readFileSync(log, "utf8")).toBe(before);
  expect(seen).toHaveLength(1);
  return seen[0]!;
}

/**
 * Run `yian run <flow> --resume <log> --quiet` for `text` with no terminal under `runEnv`; return
 * its exit code and its output, whitespace collapsed.
 */
async function cliResume(text: string, log: string, runEnv: Env) {
  const flow = fresh("flow.yaml");
  writeFileSync(flow, text);
  const chunks: string[] = [];
  const io: CliIo = {
    stdout: { write: async (chunk) => void chunks.push(chunk) },
    stderr: { write: async (chunk) => void chunks.push(chunk) },
    stdin: new FakeInput({ tty: false }),
    stderrStream: new FakeOutput(),
    env: runEnv,
  };
  const code = await main(["run", "--resume", log, "--quiet"], io);
  return { code, output: chunks.join("").split(/\s+/).filter(Boolean).join(" ") };
}

test("resume checks the provider an in flight call time agent recorded", async () => {
  const runEnv = hostEnv(dir, { ...CONFTEST, ANTHROPIC_API_KEY: "sk-ant-run" });
  const log = fresh("run.jsonl");
  const cut = await throughStep(TWO_CALL_TIME, "chat", ["ok", "ok"], runEnv);
  await crashAfter(cut, () => run(TWO_CALL_TIME, { log, env: runEnv, models: fakeModel(["ok"], runEnv).models }));
  expect(steps(log)).toHaveLength(1);
  const [effective, callTimeHost] = await resumeChecks(log, TWO_CALL_TIME, { env: runEnv });
  expect(callTimeHost).toBeNull();
  expect(effective["chat"]["provider"]).toBe("anthropic");
  expect(effective["chat"]["model"]).toBe("claude-picked");
  // `later` has not started, so its provider is checked when it resolves.
  expect("later" in effective).toBe(false);
  const before = readFileSync(log, "utf8");
  const { code, output } = await cliResume(TWO_CALL_TIME, log, env);
  expect(code, output).toBe(1);
  expect(output).toContain("$ANTHROPIC_API_KEY is not set; set it and resume again");
  expect(readFileSync(log, "utf8")).toBe(before);
});

/** Run `text` to `log`, stopping the host as `chat` starts. */
async function stoppedBeforeChat(log: string, text = PICKED, options: RunFlowOptions = {}) {
  const stop = (event: RunEvent) => {
    if (event instanceof NodeStarted && event.nodeId === "chat") throw new Error("host stopped before the agent");
  };
  const error = await rejected(run(text, { ...options, log, onEvent: stop }));
  expect((error as Error).message).toContain("host stopped");
}

test("resume passes the host key a call time selection names beside the map", async () => {
  const log = fresh("run.jsonl");
  await stoppedBeforeChat(log, PICKED, { llmConfig: HOST });
  const [effective, callTimeHost] = await resumeChecks(log, PICKED, {
    apiKey: "sk-host-secret",
    headers: { "X-Tenant": "tenant-secret" },
  });
  // `chat` has not started, so the map holds no config; the host key comes separately.
  expect(effective).toEqual({});
  expect(callTimeHost).toEqual({ provider: "openai", header_names: ["X-Tenant"] });
  const before = readFileSync(log, "utf8");
  const { code, output } = await cliResume(PICKED, log, env);
  expect(code, output).toBe(1);
  expect(output).toMatch(
    /this run's openai API key came from a host's llm_config.api_key.*resume_flow\(api_key=\.\.\., headers=\.\.\.\)/,
  );
  expect(readFileSync(log, "utf8")).toBe(before);
});

/** Run `body` with `AgentNode.prototype.bindReserved` spied on; return each call's result and `body`'s. */
async function spyingBindReserved<T>(body: () => Promise<T>): Promise<[Record<string, unknown>[], T]> {
  const bound: Record<string, unknown>[] = [];
  const original = AgentNode.prototype.bindReserved;
  AgentNode.prototype.bindReserved = function (this: AgentNode, ...args: Parameters<AgentNode["bindReserved"]>) {
    const result = original.apply(this, args);
    bound.push(result);
    return result;
  };
  try {
    return [bound, await body()];
  } finally {
    AgentNode.prototype.bindReserved = original;
  }
}

test("a host key call time run records no secret", async () => {
  const log = fresh("run.jsonl");
  const [bound, result] = await spyingBindReserved(() => run(PICKED, { log, llmConfig: HOST }));
  expect(result.status).toBe("succeeded");
  const nonEmpty = bound.filter((entry) => Object.keys(entry).length > 0);
  expect(nonEmpty).toHaveLength(1);
  for (const text of [JSON.stringify(nonEmpty[0]), readFileSync(log, "utf8")]) {
    expect(text).not.toContain("sk-host-secret");
    expect(text).not.toContain("tenant-secret");
  }
});

test("only an agent with the key reserves it", async () => {
  const flow = await loadFlow(PICKED, { env });
  expect(flow.compiled.nodes.get("chat")!.reservedWiringKeys()).toEqual(new Set(["__llm_config"]));
  const fixed = await loadFlow(PICKED.replace("{model: '${pick.output}'}", "{model: m1}"), { env });
  expect(fixed.compiled.nodes.get("chat")!.reservedWiringKeys()).toEqual(new Set());
  // MAP sends no request to rebuild.
  const map = new MapNode("m", { flowId: "f" });
  expect(Object.getPrototypeOf(map).rebuildRequests).toBe(Node.prototype.rebuildRequests);
});

test("resume without the host key raises before a call time agent calls", async () => {
  const log = fresh("run.jsonl");
  await stoppedBeforeChat(log, PICKED, { llmConfig: { provider: "openai", api_key: "sk-host-secret" } });
  const before = records(log).length;
  const { models } = fakeModel();
  expect(await rejected(resume(PICKED, log, { models }))).toBeInstanceOf(AuthError);
  const failed = records(log)
    .slice(before)
    .filter((record) => record.kind === "node_failed" && record.payload["node_id"] === "chat");
  expect(failed).toEqual([]);
  expect(models.configs).toEqual([]);
});

test("resume refuses a recorded config that no longer resolves", async () => {
  const log = await crashedInChat();
  const original = AgentNode.prototype.bindReserved;
  AgentNode.prototype.bindReserved = () => {
    throw new BindingError("node 'chat' llm_config.model: drifted", { failureCode: "contract.input" });
  };
  const before = records(log).length;
  const { models } = fakeModel();
  let error: unknown;
  try {
    error = await rejected(resume(PICKED, log, { models }));
  } finally {
    AgentNode.prototype.bindReserved = original;
  }
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toMatch(/^node 'chat' llm_config.model: drifted; an earlier start of this node resolved it/);
  expect(records(log).slice(before).map((record) => record.kind)).not.toContain("node_failed");
  expect(models.configs).toEqual([]);
});

const DEFS_CALL_TIME = `id: f
name: f
llm_config: {provider: openai}
defs:
  ask:
    input: {m: !str }
    nodes:
      chat: {kind: agent, prompt: Go., llm_config: {model: '\${input.m}'}}
    output: \${chat.output}
nodes:
  a: {kind: call, call: ask, input: {m: model-a}}
  b: {kind: call, call: ask, input: {m: '\${a.output}'}}
output: \${b.output}
`;

/** Run `DEFS_CALL_TIME` until the host dies in the model call of `b`'s agent; return the log. */
async function crashedInSecondCall(): Promise<string> {
  const log = fresh("run.jsonl");
  const cut = await throughStep(DEFS_CALL_TIME, "b/chat", ["ok", "ok"]);
  await crashAfter(cut, () => run(DEFS_CALL_TIME, { log, models: fakeModel(["ok", "ok"]).models }));
  return log;
}

/** The payloads of the log's `llm_config` steps. */
const instanceSteps = (log: string) =>
  records(log)
    .filter(isConfigStep)
    .map((record) => record.payload);

test("resume recomputes the config of each defs call instance", async () => {
  const log = await crashedInSecondCall();
  const recorded = instanceSteps(log);
  expect(recorded).toHaveLength(2);
  const [first, second] = recorded as [Record<string, any>, Record<string, any>];
  expect(first["node_id"]).not.toBe(second["node_id"]);
  expect(first["step"]["effective_llm_config"]["model"]).toBe("model-a");
  expect(second["step"]["effective_llm_config"]["model"]).toBe("ok");
  const { models } = fakeModel();
  const result = await resume(DEFS_CALL_TIME, log, { models });
  expect(result.status).toBe("succeeded");
  const all = instanceSteps(log);
  expect(all).toHaveLength(3);
  // The resumed start records the same step again, as the node's next step.
  expect({ ...all[2], n: null }).toEqual({ ...second, n: null });
  expect([Number(second["n"]), Number(all[2]!["n"])]).toEqual([1, 2]);
  expect(models.configs.map((config) => config["model"])).toEqual(["ok"]);
});

test("resume refuses a defs call instance whose recorded config differs", async () => {
  const log = await crashedInSecondCall();
  const lines = readFileSync(log, "utf8").split("\n").filter((line) => line !== "");
  let edited = 0;
  for (const [index, line] of lines.entries()) {
    const record = JSON.parse(line);
    const step = isConfigStep(record) ? record.payload.step : null;
    if (step && step.effective_llm_config.model === "ok") {
      step.effective_llm_config.model = "gpt-tampered";
      lines[index] = JSON.stringify(record);
      edited += 1;
    }
  }
  expect(edited).toBe(1);
  writeFileSync(log, `${lines.join("\n")}\n`);
  const before = records(log).length;
  const { models } = fakeModel();
  const error = await rejected(resume(DEFS_CALL_TIME, log, { models }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toMatch(/^node 'b\/chat' llm_config: recorded config differs from the recomputed one/);
  expect(records(log).slice(before).map((record) => record.kind)).not.toContain("node_failed");
  expect(models.configs).toEqual([]);
});

test("a recorded runtime instance config is labelled by its static path", () => {
  const recorded = { "b/chat": { provider: "openai", model: "m", temperature: 0.5 } };
  const label = "node 'ask' adaptive_questions.llm_config";
  const staticPaths = new Map([["b/chat", "a/chat"]]);
  for (const [labels, pattern] of [
    [new Map([["a/chat", label]]), /^node 'ask' adaptive_questions\.llm_config: /],
    [new Map([["b/chat", label]]), /^Agent 'b\/chat': /],
  ] as const) {
    let error: unknown = null;
    try {
      checkRecordedLlmFields(recorded, labels, staticPaths);
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toMatch(pattern);
  }
});
