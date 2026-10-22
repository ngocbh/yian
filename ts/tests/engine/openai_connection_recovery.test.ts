/**
 * A port of `tests/engine/test_openai_connection_recovery.py`: the recorded OpenAI connection
 * identity governs resume and delayed backward.
 *
 * Ported: `test_saved_identity_is_recorded_and_resume_refuses_reconnect` and
 * `test_backward_refuses_changed_selected_connection`. The model calls go through the real
 * `chatModelFromConfig` with a recording `send` that answers one Responses stream (`recording`),
 * where Python routes the subscription client to `ResponsesServer`. Each row has its own
 * environment (`hostEnv`) and so its own credential store, which logins are saved to.
 *
 * Already ported: `test_old_backward_config_gets_api_semantics_before_factory`, in
 * `agent_backward.test.ts` (TS pins an old record without I/O, so no saved login is involved).
 *
 * Not ported: `test_old_api_record_resumes_via_api_despite_saved_subscription` (a record written
 * before connections were recorded, made by deleting `_connection` from a run log),
 * `test_old_api_run_resumes_shared_message_history` and
 * `test_fresh_connected_run_reuses_pre_cutover_api_message_history` (runs and session memory from
 * before the cutover, made by patching Python's `connection_snapshot`): Python-era records and
 * stores TS never reads (decisions 8, 22).
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { ApiKeyCredentials, AuthStore, OAuthCredentials } from "../../src/llm_clients/auth/index.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { recording, rejected, responsesText } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-openai-connection-recovery-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const FLOW = `id: pinned-openai
name: pinned_openai
nodes:
  agent: {kind: agent, prompt: hi, llm_config: {provider: openai, model: gpt-5.5}}
output: \${agent.output}
`;

let files = 0;

/** A fresh environment with its own credential store, and a log path beside it. */
function fresh(): { env: Env; path: string } {
  const root = join(dir, String(files++));
  // The credential store refuses a directory others can read.
  mkdirSync(root, { mode: 0o700 });
  return { env: hostEnv(root), path: join(root, "run.jsonl") };
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

test("saved identity is recorded and resume refuses reconnect", async () => {
  const { env, path } = fresh();
  const selected = await connect(env);
  const flow = await loadFlow(FLOW, { env });
  const models = recording(env, responsesText("ok"));
  const stopped = await rejected(runFlow(flow, {}, { log: path, onEvent: stopBeforeModel, models, env }));
  expect((stopped as Error).message).toContain("host stopped");
  const payload = JSON.parse(readFileSync(path, "utf8").split("\n")[0]!).payload;
  const config = payload["effective_llm_by_static_path"]["agent"];
  expect(config["_connection"]["credential_id"]).toBe(selected.credentialId);
  expect(readFileSync(path, "utf8")).not.toContain("secret");
  await new AuthStore(null, { env }).save("openai", new ApiKeyCredentials("new-key"), "different-model");
  const result = await resumeFlow(flow, { log: path, models, env });
  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("changed");
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
