/** The recorded OpenAI connection identity governs resume and delayed backward. */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { ApiKeyCredentials, AuthStore, OAuthCredentials } from "../../src/llm_clients/auth/index.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { advance, runFlow, type RunFlowOptions } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { recording, rejected, responsesText, serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-openai-connection-recovery-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const FLOW = `id: pinned-openai
name: pinned_openai
nodes:
  agent: {kind: agent, prompt: hi, llm_config: {provider: openai, model: gpt-5.5}}
output: \${agent.output}
`;

let files = 0;

/** A fresh environment with its own credential store at `auth` and `vars` set, and a log path. */
function fresh(vars: Record<string, string> = {}): { env: Env; auth: string; path: string } {
  const root = join(dir, String(files++));
  // The credential store refuses a directory others can read.
  mkdirSync(root, { mode: 0o700 });
  return { env: hostEnv(root, vars), auth: join(root, "auth.json"), path: join(root, "run.jsonl") };
}

/** Save a subscription login for `openai` and return its selection. */
const connect = (env: Env) =>
  new AuthStore(null, { env }).save(
    "openai",
    new OAuthCredentials({
      accessToken: "access-secret",
      refreshToken: "refresh-secret",
      expiresAt: 4102444800,
      accountId: "account",
    }),
    "gpt-5.5",
  );

/** Throw when the agent starts, as a host stopping before the model call. */
function stopBeforeModel(event: RunEvent): void {
  if (event instanceof NodeStarted && event.nodeId === "agent") throw new Error("host stopped before model");
}

/** Run `flow` to a log at `path` and expect the host to stop it before the model call. */
async function interrupt(flow: ComposedFlow, path: string, options: RunFlowOptions): Promise<void> {
  const error = await rejected(runFlow(flow, {}, { log: path, onEvent: stopBeforeModel, ...options }));
  expect((error as Error).message).toContain("host stopped");
}

/** Delete the agent's recorded `_connection` from the log at `path`, as a record written before connections were. */
function dropConnection(path: string): void {
  const lines = readFileSync(path, "utf8").split("\n").filter(Boolean);
  const first = JSON.parse(lines[0]!);
  expect(first.payload["effective_llm_by_static_path"]["agent"]).toHaveProperty("_connection");
  delete first.payload["effective_llm_by_static_path"]["agent"]["_connection"];
  lines[0] = JSON.stringify(first);
  writeFileSync(path, lines.join("\n") + "\n");
}

test("saved identity is recorded and resume refuses reconnect", async () => {
  const { env, path } = fresh();
  const selected = await connect(env);
  const flow = await loadFlow(FLOW, { env });
  const models = recording(env, responsesText("ok"));
  await interrupt(flow, path, { models, env });
  const payload = JSON.parse(readFileSync(path, "utf8").split("\n")[0]!).payload;
  const config = payload["effective_llm_by_static_path"]["agent"];
  expect(config["_connection"]["credential_id"]).toBe(selected.credentialId);
  expect(readFileSync(path, "utf8")).not.toContain("secret");
  await new AuthStore(null, { env }).save("openai", new ApiKeyCredentials("new-key"), "different-model");
  const result = await advance({ log: path, models, env });
  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("changed");
});

test("old api record resumes via api despite saved subscription", async () => {
  const { env, path } = fresh({ OPENAI_API_KEY: "old-api-key" });
  const flow = await loadFlow(FLOW, { env });
  await interrupt(flow, path, { env });
  dropConnection(path);
  await connect(env);
  const models = recording(env, responsesText("ok"));
  const result = await advance({ log: path, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(new URL(models.requests[0]!.url).host).toBe("api.openai.com");
  expect(models.requests[0]!.headers.get("Authorization")).toBe("Bearer old-api-key");
});

test("backward refuses changed selected connection", async () => {
  const { env, path } = fresh();
  await connect(env);
  const models = recording(env, responsesText("ok"));
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, { learnable: true, log: path, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  await connect(env);
  const outcome = await result.backward("Review");
  expect(outcome.status).toBe("failed");
  expect(models.requests).toHaveLength(1);
});

test("an old api run resumes with the legacy connection", async () => {
  const { env, auth, path } = fresh();
  const flow = await loadFlow(FLOW, { env });
  const model = new ScriptedModel([textReply("reply"), textReply("reply")]);
  const models = serving(model, env);
  expect((await runFlow(flow, {}, { models, env })).status).toBe("succeeded");
  await interrupt(flow, path, { models, env });
  dropConnection(path);
  // A store no read survives, which a fresh run reads: the resume fails if it reads a selection.
  writeFileSync(auth, "not json", { mode: 0o600 });
  const reading = await rejected(runFlow(flow, {}, { models, env }));
  expect((reading as Error).message).toContain("Credential storage is malformed");
  const resumed = await advance({ log: path, models, env });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(model.requests.at(-1)!.messages.map((message) => message.text()).slice(1)).toEqual(["hi"]);
  expect((models.configs.at(-1)!["_connection"] as Record<string, unknown>)["source"]).toBe("legacy_environment");
});
