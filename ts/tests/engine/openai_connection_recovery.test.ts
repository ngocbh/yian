/**
 * A port of `tests/engine/test_openai_connection_recovery.py`: the recorded OpenAI connection
 * identity governs resume and delayed backward.
 *
 * Ported: every test but the one already ported below. The model calls go through the real
 * `chatModelFromConfig` with a recording `send` that answers one Responses stream (`recording`),
 * where Python routes the OpenAI clients to `ResponsesServer`; the session rows' earlier runs
 * answer from a scripted model (`serving`), where Python patches `model_from_config`. Each row has
 * its own environment (`hostEnv`) and so its own credential store, which logins are saved to.
 *
 * Adapted: a record from before connections were recorded is a run log without `_connection`; the
 * rows build it by deleting `_connection` from the run's own log, where
 * `test_old_api_run_resumes_shared_message_history` patches Python's `connection_snapshot` to
 * record none. That row's patched snapshot, which fails the test when the resume reads it, is a
 * store file holding `not json`, which fails any read of it (a fresh run of the flow reads it and
 * fails). Session memory keeps no connection, so
 * `test_fresh_connected_run_reuses_pre_cutover_api_message_history`'s seed run needs no patched
 * snapshot.
 *
 * Already ported: `test_old_backward_config_gets_api_semantics_before_factory`, in
 * `agent_backward.test.ts` (TS pins an old record without I/O, so no saved login is involved).
 */

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
import { SessionMemory } from "../../src/memory/session.ts";
import { resumeFlow, runFlow, type RunFlowOptions } from "../../src/run.ts";
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

/** `FLOW` with the agent keeping session memory. */
const SESSION_FLOW = FLOW.replace("{kind: agent,", "{kind: agent, long_term_memory: session,");

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
  const result = await resumeFlow(flow, { log: path, models, env });
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
  const result = await resumeFlow(flow, { log: path, models, env });
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

test("old api run resumes shared message history", async () => {
  const { env, auth, path } = fresh();
  const flow = await loadFlow(SESSION_FLOW, { env });
  const memory = new Map([["session", new SessionMemory()]]);
  const model = new ScriptedModel([textReply("reply"), textReply("reply")]);
  const models = serving(model, env);
  expect((await runFlow(flow, {}, { memory, models, env })).status).toBe("succeeded");
  await interrupt(flow, path, { memory, models, env });
  dropConnection(path);
  // A store no read survives, which a fresh run reads: the resume fails if it reads a selection.
  writeFileSync(auth, "not json", { mode: 0o600 });
  const other = new Map([["session", new SessionMemory()]]);
  const reading = await rejected(runFlow(flow, {}, { memory: other, models, env }));
  expect((reading as Error).message).toContain("Credential storage is malformed");
  const resumed = await resumeFlow(flow, { log: path, memory, models, env });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(model.requests.at(-1)!.messages.map((message) => message.text()).slice(-3)).toEqual(["hi", "reply", "hi"]);
  expect((models.configs.at(-1)!["_connection"] as Record<string, unknown>)["source"]).toBe("legacy_environment");
});

test("fresh connected run reuses pre cutover api message history", async () => {
  const { env } = fresh();
  const flow = await loadFlow(SESSION_FLOW, { env });
  const memory = new Map([["session", new SessionMemory()]]);
  const seed = serving(new ScriptedModel([textReply("old API reply")]), env);
  expect((await runFlow(flow, {}, { memory, models: seed, env })).status).toBe("succeeded");
  const models = recording(env, responsesText("6"));
  // Each explicit reconnect changes the stored credential identity. Both new runs retain
  // ordinary compatible history without a memory migration.
  for (let round = 0; round < 2; round++) {
    await connect(env);
    const result = await runFlow(flow, {}, { memory, models, env });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe("6");
  }
  expect(models.requests.map((sent) => sent.url)).toEqual(
    Array(2).fill("https://chatgpt.com/backend-api/codex/responses"),
  );
  for (const { body } of models.requests) {
    const input = body["input"] as Record<string, any>[];
    expect(
      input.some(
        (item) =>
          item["role"] === "assistant" &&
          (item["content"] ?? []).some((block: Record<string, unknown>) => block["text"] === "old API reply"),
      ),
    ).toBe(true);
  }
});
