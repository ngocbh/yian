/**
 * `yian run --provider/--model` build the outermost cascade layer (`llmConfig`). What the run was
 * given is read from the run log's `RunStarted` record, and a fake `fetch` answers as xAI's chat
 * endpoint; a lone `--model` keeps the default provider.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";

const FLOW = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi}\noutput: ${a.output}\n";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-llm-flags-"));
  // Keeps default run logs and saved logins away from the developer's real home.
  mkdirSync(join(tmp, "home"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** A one-word ("ok") chat-completions stream. */
const OK = [
  { choices: [{ index: 0, delta: { role: "assistant", content: "ok" }, finish_reason: null }] },
  { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
]
  .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
  .join("") + "data: [DONE]\n\n";

/** Run `yian run f.yaml <args>` against a fake model; return its code and agent `a`'s config. */
async function runWith(args: readonly string[], vars: Record<string, string>) {
  const flow = join(tmp, "f.yaml");
  writeFileSync(flow, FLOW);
  const log = join(tmp, "run.jsonl");
  const env: Record<string, string> = { HOME: join(tmp, "home"), XAI_API_KEY: "test-xai-key", ...vars };
  const io: CliIo = {
    stdout: { write: async () => {} },
    stderr: { write: async () => {} },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (Object.hasOwn(env, name) ? env[name] : undefined),
  };
  const send = globalThis.fetch;
  const answer = async () => new Response(OK, { headers: { "content-type": "text/event-stream" } });
  globalThis.fetch = answer as unknown as typeof fetch;
  let code: number;
  try {
    code = await main(["run", flow, "-q", "--log", log, ...args], io);
  } finally {
    globalThis.fetch = send;
  }
  const record = readRunLog(log)[0]!.record;
  if (!(record instanceof RunStartedRecord)) throw new Error("the log doesn't start with RunStarted");
  const { provider, model } = record.effectiveLlmByStaticPath["a"]!;
  return { code, provider, model };
}

const CONFTEST = {
  YIAN_DEFAULT_PROVIDER: "openai",
  YIAN_DEFAULT_MODEL: "gpt-5.5",
  OPENAI_API_KEY: "sk-test",
};

test("the CLI flags build the LLM config", async () => {
  const { code, provider, model } = await runWith(["--provider", "xai", "--model", "grok-4.3"], CONFTEST);
  expect(code).toBe(0);
  expect({ provider, model }).toEqual({ provider: "xai", model: "grok-4.3" });
});

test("with no CLI flags the defaults decide", async () => {
  const defaults = { YIAN_DEFAULT_PROVIDER: "xai", YIAN_DEFAULT_MODEL: "grok-4.3" };
  const { code, provider, model } = await runWith([], defaults);
  expect(code).toBe(0);
  // No CLI layer: the defaults decide.
  expect({ provider, model }).toEqual({ provider: "xai", model: "grok-4.3" });
});

test("a lone --model keeps the default provider", async () => {
  const defaults = { YIAN_DEFAULT_PROVIDER: "xai", YIAN_DEFAULT_MODEL: "grok-old" };
  const { code, provider, model } = await runWith(["--model", "grok-4.3"], defaults);
  expect(code).toBe(0);
  expect({ provider, model }).toEqual({ provider: "xai", model: "grok-4.3" });
});
