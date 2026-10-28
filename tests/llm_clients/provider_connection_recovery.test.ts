/** A resumed or delayed run keeps each provider's recorded connection. */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { ApiKeyCredentials, AuthStore } from "../../src/llm_clients/auth/index.ts";
import { advance, runFlow, type RunFlowOptions } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import {
  anthropicText,
  chatText,
  NO_FEEDBACK,
  ollamaText,
  type Replier,
  recording,
  rejected,
  responsesText,
  type Sent,
} from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-provider-connection-recovery-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let files = 0;

/** A fresh environment with its own credential store at `auth` and `vars` set, and a log path. */
function fresh(vars: Record<string, string> = {}): { env: Env; auth: string; path: string } {
  const root = join(dir, String(files++));
  // The credential store refuses a directory others can read.
  mkdirSync(root, { mode: 0o700 });
  return { env: hostEnv(root, vars), auth: join(root, "auth.json"), path: join(root, "run.jsonl") };
}

/** The flow whose one agent calls `provider`'s `model`. */
const flowOf = (env: Env, provider: string, model: string) =>
  loadFlow(
    "id: pinned\nname: pinned\nnodes:\n" +
      `  agent: {kind: agent, prompt: hi, llm_config: {provider: ${provider}, model: ${model}}}\n` +
      "output: ${agent.output}\n",
    { env },
  );

/** Answer `texts` in call order, repeating the last, each in the stream its URL's provider speaks. */
function answers(...texts: string[]): Replier {
  return (sent, count) => {
    const text = texts[Math.min(count, texts.length) - 1]!;
    const path = new URL(sent.url).pathname;
    if (path.endsWith("/messages")) return anthropicText(text);
    if (path.endsWith("/api/chat")) return ollamaText(text);
    return path.endsWith("/responses") ? responsesText(text) : chatText(text);
  };
}

/** The key a request sent: Anthropic's `x-api-key`, else the bearer token. */
const keyOf = (sent: Sent) => sent.headers.get("x-api-key") ?? sent.headers.get("authorization")?.replace(/^Bearer /, "");

/** Throw when the agent starts, as a host stopping before the model call. */
function stopBeforeModel(event: RunEvent): void {
  if (event instanceof NodeStarted && event.nodeId === "agent") throw new Error("host stopped before model");
}

/** Run `flow` to a log at `path` and expect the host to stop it before the model call. */
async function interrupt(flow: ComposedFlow, path: string, options: RunFlowOptions): Promise<void> {
  const error = await rejected(runFlow(flow, {}, { log: path, onEvent: stopBeforeModel, ...options }));
  expect((error as Error).message).toContain("host stopped");
}

/** Make the store at `auth` fail every read, and expect a fresh run of `flow` to read it and fail. */
async function breakStore(env: Env, auth: string, flow: ComposedFlow): Promise<void> {
  writeFileSync(auth, "not json", { mode: 0o600 });
  const models = recording(env, answers("ok"));
  const error = await rejected(runFlow(flow, {}, { models, env }));
  expect((error as Error).message).toContain("Credential storage is malformed");
}

/** The agent's effective config the log at `path` records at the run's start. */
function recorded(path: string): Record<string, any> {
  return JSON.parse(readFileSync(path, "utf8").split("\n")[0]!).payload["effective_llm_by_static_path"]["agent"];
}

test("store connection is pinned across resume", async () => {
  const { env, path } = fresh();
  const saved = await new AuthStore(null, { env }).save("anthropic", new ApiKeyCredentials("old-key"), "claude-x");
  const flow = await flowOf(env, "anthropic", "claude-x");
  await interrupt(flow, path, { env });
  const connection = recorded(path)["_connection"];
  expect(connection["source"]).toBe("store");
  expect(connection["credential_id"]).toBe(saved.credentialId);
  expect(readFileSync(path, "utf8")).not.toContain("old-key");

  await new AuthStore(null, { env }).save("anthropic", new ApiKeyCredentials("new-key"), "claude-x");
  const models = recording(env, answers("ok"));
  const result = await advance({ log: path, models, env });

  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("yian auth anthropic");
  expect(models.requests.every((sent) => keyOf(sent) !== "new-key")).toBe(true);
  expect(readFileSync(path, "utf8")).not.toContain("new-key");
});

test("environment connection never switches to a later login", async () => {
  const { env, path } = fresh({ ANTHROPIC_API_KEY: "env-key" });
  const flow = await flowOf(env, "anthropic", "claude-x");
  await interrupt(flow, path, { env });
  expect(recorded(path)["_connection"]["source"]).toBe("environment");

  await new AuthStore(null, { env }).save("anthropic", new ApiKeyCredentials("later-key"), "claude-x");
  const models = recording(env, answers("ok"));
  const result = await advance({ log: path, models, env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(models.requests.length).toBeGreaterThan(0);
  expect(models.requests.map(keyOf)).toEqual(models.requests.map(() => "env-key"));
});

test("removed login fails cleanly", async () => {
  const { env, path } = fresh();
  await new AuthStore(null, { env }).save("anthropic", new ApiKeyCredentials("old-key"), "claude-x");
  const flow = await flowOf(env, "anthropic", "claude-x");
  await interrupt(flow, path, { env });

  await new AuthStore(null, { env }).remove("anthropic");
  const models = recording(env, answers("ok"));
  const result = await advance({ log: path, models, env });

  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("yian auth anthropic");
  expect(models.requests).toEqual([]);
});

test("ollama resume never reads the snapshot", async () => {
  const { env, auth, path } = fresh();
  const flow = await flowOf(env, "ollama", "llama3");
  await interrupt(flow, path, { env });
  expect("_connection" in recorded(path)).toBe(false);

  await breakStore(env, auth, flow);
  const models = recording(env, answers("ok"));
  const result = await advance({ log: path, models, env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(new URL(models.requests[0]!.url).pathname).toBe("/api/chat");
});

test("host key resume and backward", async () => {
  const { env, auth, path } = fresh();
  await new AuthStore(null, { env }).save("anthropic", new ApiKeyCredentials("saved-key"), "claude-x");
  const flow = await flowOf(env, "anthropic", "claude-x");
  await interrupt(flow, path, { llmConfig: { provider: "anthropic", api_key: "host-key" }, learnable: true, env });
  expect("_connection" in recorded(path)).toBe(false);
  expect(readFileSync(path, "utf8")).not.toContain("host-key");

  await breakStore(env, auth, flow);
  const models = recording(env, answers("ok", NO_FEEDBACK));
  const result = await advance({ log: path, apiKey: "host-key", models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const outcome = await result.backward("Be brief.", { apiKey: "host-key" });
  expect(outcome.status, outcome.failure?.message).toBe("succeeded");

  expect(models.requests.map((sent) => [sent.url, keyOf(sent)])).toEqual([
    ["https://api.anthropic.com/v1/messages", "host-key"],
    ["https://api.anthropic.com/v1/messages", "host-key"],
  ]);
});

const SAVED_HEADER = "saved-gw-51ad0e";
const CONNECTION_FIELDS = ["credential_id", "endpoint", "method", "source", "store_path"];

test("saved settings are recorded by name and sent on resume", async () => {
  const { env, path } = fresh();
  await new AuthStore(null, { env }).save(
    "openai",
    new ApiKeyCredentials("sk-saved", { "X-Gateway": SAVED_HEADER }),
    "gpt-x",
    { organization: "org-saved", project: "proj-saved" },
  );
  const flow = await flowOf(env, "openai", "gpt-x");
  await interrupt(flow, path, { env });
  const connection = recorded(path)["_connection"];
  expect(connection["organization"]).toBe("org-saved");
  expect(connection["project"]).toBe("proj-saved");
  expect(connection["header_names"]).toEqual(["X-Gateway"]);
  expect(readFileSync(path, "utf8")).not.toContain(SAVED_HEADER);
  expect(readFileSync(path, "utf8")).not.toContain("sk-saved");

  const models = recording(env, answers("ok"));
  const result = await advance({ log: path, models, env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(models.requests).toHaveLength(1);
  const { url, headers } = models.requests[0]!;
  expect(new URL(url).host).toBe("api.openai.com");
  expect(keyOf(models.requests[0]!)).toBe("sk-saved");
  expect(["OpenAI-Organization", "OpenAI-Project", "X-Gateway"].map((name) => headers.get(name))).toEqual([
    "org-saved",
    "proj-saved",
    SAVED_HEADER,
  ]);
});

test("a login without settings records and replays the old fields", async () => {
  const { env, path } = fresh();
  await new AuthStore(null, { env }).save("anthropic", new ApiKeyCredentials("saved-key"), "claude-x");
  const flow = await flowOf(env, "anthropic", "claude-x");
  await interrupt(flow, path, { env });
  expect(Object.keys(recorded(path)["_connection"]).sort()).toEqual(CONNECTION_FIELDS);

  const models = recording(env, answers("ok"));
  const result = await advance({ log: path, models, env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(keyOf(models.requests[0]!)).toBe("saved-key");
  expect(models.requests[0]!.headers.get("X-Gateway")).toBeNull();
});

test("a host key run never sends the saved settings", async () => {
  const { env, path } = fresh();
  await new AuthStore(null, { env }).save(
    "anthropic",
    new ApiKeyCredentials("saved-key", { "X-Gateway": SAVED_HEADER }),
    "claude-x",
  );
  const flow = await flowOf(env, "anthropic", "claude-x");
  await interrupt(flow, path, { llmConfig: { provider: "anthropic", api_key: "host-key" }, env });
  const config = recorded(path);
  expect("_connection" in config || "_header_names" in config).toBe(false);

  const models = recording(env, answers("ok"));
  const result = await advance({ log: path, apiKey: "host-key", models, env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(keyOf(models.requests[0]!)).toBe("host-key");
  expect(models.requests[0]!.headers.get("X-Gateway")).toBeNull();
});
