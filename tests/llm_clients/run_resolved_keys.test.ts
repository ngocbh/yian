/**
 * A key from the environment or a saved login stays out of every record when a provider echoes
 * it. Provider calls go through the real `chatModelFromConfig` with a recording `send` that answers
 * OpenAI's masked-key 401 (`keyEcho401`); each row has its own environment and credential store.
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import type { RunEvent } from "../../src/events.ts";
import { ApiKeyCredentials, AuthStore, OAuthCredentials } from "../../src/llm_clients/auth/index.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { chatText, expectHidden, inOrder, keyEcho401, type Replier, recording } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-run-resolved-keys-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const KEY = "sk-resolved-401-qvjm";
const FRAGMENTS = [KEY, KEY.slice(0, 8), `*${KEY.slice(-4)}`];

const FLOW = `id: keyed
name: keyed
nodes:
  agent: {kind: agent, prompt: hi}
output: \${agent.output}
llm_config: {provider: PROVIDER, model: MODEL}
`;

let files = 0;

/** A fresh environment, with its own credential store and `vars` set, and a log path beside it. */
function fresh(vars: Record<string, string> = {}): { env: Env; path: string } {
  const root = join(dir, String(files++));
  // The credential store refuses a directory others can read.
  mkdirSync(root, { mode: 0o700 });
  return { env: hostEnv(root, vars), path: join(root, "run.jsonl") };
}

/** The flow, calling `provider`'s `model`. */
const flow = (env: Env, provider = "vllm", model = "served-model") =>
  loadFlow(FLOW.replace("PROVIDER", provider).replace("MODEL", model), { env });

/** Answer the first `accept` requests, then echo the key masked, as OpenAI's 401 does. */
function rejecting(accept = 0): Replier {
  const ok = inOrder(chatText("ok"));
  return (sent, count) => (count <= accept ? ok(sent, count) : keyEcho401(KEY));
}

/** Run the flow against a provider that rejects the key, and expect no part of it recorded. */
async function runRejected(env: Env, path: string): Promise<void> {
  const events: RunEvent[] = [];
  const models = recording(env, rejecting());
  const result = await runFlow(await flow(env), {}, { log: path, onEvent: (event) => void events.push(event), models, env });
  expect(result.status).toBe("failed");
  expect(models.requests[0]!.headers.get("authorization")).toBe(`Bearer ${KEY}`);
  expect(readFileSync(path, "utf8")).toContain("[REDACTED]");
  expectHidden(path, [events, result], FRAGMENTS);
}

test("a 401 with the key variable records no part of it", async () => {
  const { env, path } = fresh({ VLLM_API_KEY: KEY });
  await runRejected(env, path);
});

test("a 401 with a saved login records no part of it", async () => {
  const { env, path } = fresh();
  await new AuthStore(null, { env }).save("vllm", new ApiKeyCredentials(KEY), "served-model");
  await runRejected(env, path);
});

test("a backward 401 with a saved login records no part of it", async () => {
  const { env, path } = fresh();
  await new AuthStore(null, { env }).save("vllm", new ApiKeyCredentials(KEY), "served-model");
  const models = recording(env, rejecting(1));
  const result = await runFlow(await flow(env), {}, { log: path, learnable: true, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const outcome = await result.backward("Be brief.");
  expect(outcome.status).toBe("failed");
  expect(models.requests).toHaveLength(2);
  expect(models.requests[1]!.headers.get("authorization")).toBe(`Bearer ${KEY}`);
  expect(readFileSync(path, "utf8")).toContain("[REDACTED]");
  expectHidden(path, [result, outcome], FRAGMENTS);
});

test("a subscription token echoed by the provider is not recorded", async () => {
  const [access, refresh] = ["atok47-echo-7h3k", "rtok91-echo-p9d2"];
  const { env, path } = fresh();
  await new AuthStore(null, { env }).save(
    "openai",
    new OAuthCredentials({ accessToken: access, refreshToken: refresh, expiresAt: 4102444800, accountId: "account" }),
    "gpt-5.5",
  );
  const models: ModelAccess = {
    env,
    createModel: () => {
      throw new Error(`401 for Bearer ${access}; refresh ${refresh.slice(0, 6)}****${refresh.slice(-4)}`);
    },
  };
  const events: RunEvent[] = [];
  const result = await runFlow(await flow(env, "openai", "gpt-5.5"), {}, {
    log: path,
    onEvent: (event) => void events.push(event),
    models,
    env,
  });
  expect(result.status).toBe("failed");
  expect(readFileSync(path, "utf8")).toContain("[REDACTED]");
  expectHidden(path, [events, result], [access, access.slice(0, 8), refresh, refresh.slice(0, 6), `*${refresh.slice(-4)}`]);
});
