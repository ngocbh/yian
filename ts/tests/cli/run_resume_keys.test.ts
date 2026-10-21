/**
 * `yian run --resume` checks the recorded provider keys before it writes: a port of
 * `tests/cli/test_run_resume_keys.py`, every test under its name.
 *
 * Changed:
 * - Python's patched `model_from_config` is a fake `fetch`: a model that answers "written offline"
 *   (a Responses stream for OpenAI, a chat-completions stream otherwise) or one that records the
 *   call and fails, the test then checking no call was made.
 * - Python's environment fixture is the command's env, holding only `HOME`, `YIAN_AUTH_FILE` and
 *   `XDG_CONFIG_HOME` under a temp directory (and the key a row sets); logins are saved through
 *   that env, and the unfinished run starts with the key in its own env.
 * - Python's patched `_prompt_request` is a fake TTY stdin with the keys typed ahead; the prompts
 *   are counted from its raw-mode switches. A typed key reaching the run is checked by the
 *   model call's `Authorization` header (Python reads `os.environ`).
 * - The saved-login row needing no environment key runs with a stdin that isn't a TTY, so a prompt
 *   can't be answered (Python makes the prompt fail).
 * - The host-key message says "a host's" for Python's "a Python host's".
 * - `test_resume_checks_saved_login_of_non_openai_provider` uses `xai`, whose chat-completions
 *   stream the fake speaks, for Python's `anthropic`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { main } from "../../src/cli/index.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { type RunEvent, NodeStarted } from "../../src/events.ts";
import { ApiKeyCredentials, AuthStore } from "../../src/llm_clients/auth/index.ts";
import { type RunFlowOptions, runFlow } from "../../src/run.ts";
import { FakeInput, FakeOutput } from "./terminal_both_runtimes.ts";

const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;

const FLOW = `id: resume_key
name: resume_key
input:
  topic: !str
nodes:
  prep:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: |
      return inputs["topic"]
  write:
    kind: agent
    input:
      text: \${prep.output}
    prompt: "Write about \${text}"
    output: str
output: \${write.output}
llm_config: {provider: PROVIDER, model: offline-model}
`;

let tmp: string;
let vars: Record<string, string>;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-resume-keys-"));
  // No provider key, default or saved login.
  vars = {
    HOME: join(tmp, "home"),
    YIAN_AUTH_FILE: join(tmp, "auth", "auth.json"),
    XDG_CONFIG_HOME: join(tmp, "xdg"),
  };
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** The command's environment, as a store reads it. */
const env = (name: string) => (Object.hasOwn(vars, name) ? vars[name] : undefined);

/** Server-Sent Events, one per `[event, data]`. */
function sse(events: [string, Record<string, unknown>][]): string {
  return events.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify({ type: event, ...data })}\n\n`).join("");
}

const MESSAGE = {
  id: "msg_1",
  type: "message",
  role: "assistant",
  content: [{ type: "output_text", text: "written offline" }],
};

/** A Responses stream answering "written offline". */
const RESPONSES_REPLY = sse([
  ["response.output_item.added", { output_index: 0, item: { ...MESSAGE, content: [] } }],
  ["response.output_text.delta", { output_index: 0, content_index: 0, item_id: "msg_1", delta: "written offline" }],
  ["response.completed", { response: { id: "resp_1", status: "completed", output: [MESSAGE] } }],
]);

/** A chat-completions stream answering "written offline". */
const CHAT_REPLY =
  [
    { choices: [{ index: 0, delta: { role: "assistant", content: "written offline" }, finish_reason: null }] },
    { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
  ]
    .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
    .join("") + "data: [DONE]\n\n";

/** The `Authorization` header of each model request. */
let authorizations: (string | null)[];

/** Answer the AGENT locally (Python's `_fake_model`). */
async function fakeModel(input: string | URL | Request, init?: RequestInit): Promise<Response> {
  const request = new Request(input as string, init);
  authorizations.push(request.headers.get("authorization"));
  const body = request.url.endsWith("/responses") ? RESPONSES_REPLY : CHAT_REPLY;
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

/** Each request the failing model got (Python's `_no_model`). */
let reached: string[];

/** Record that the run reached a model, and fail the call. */
async function noModel(input: string | URL | Request): Promise<Response> {
  reached.push(input instanceof Request ? input.url : String(input));
  throw new Error("the resumed run must not reach a model");
}

type FakeFetch = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

/** Run `body` with `fake` as `fetch`. */
async function withFetch<T>(fake: FakeFetch, body: () => Promise<T>): Promise<T> {
  const send = globalThis.fetch;
  authorizations = [];
  reached = [];
  globalThis.fetch = fake as unknown as typeof fetch;
  try {
    return await body();
  } finally {
    globalThis.fetch = send;
  }
}

/** Write the flow for `provider` under `dir`; return its path. */
function flowFile(dir = tmp, provider = "openai", text = FLOW): string {
  const path = join(dir, "flow.yaml");
  writeFileSync(path, text.replace("PROVIDER", provider));
  return path;
}

/**
 * Record a run interrupted at `prep`, with `envVar` set to `key` only in its own env (Python's
 * `_unfinished_log`).
 */
async function unfinishedLog(
  flow: string,
  { envVar = "OPENAI_API_KEY" as string | null, key = "sk-fake", ...options }: {
    envVar?: string | null;
    key?: string;
  } & RunFlowOptions = {},
): Promise<string> {
  const dir = join(flow, "..");
  const loaded = await loadFlow(readFileSync(flow, "utf8"), { searchPaths: [dir] });
  const log = join(dir, "unfinished.jsonl");
  const runVars: Record<string, string> = { ...vars, ...(envVar === null ? {} : { [envVar]: key }) };
  const interrupt = (event: RunEvent) => {
    if (event instanceof NodeStarted && event.nodeId === "prep") throw new Error("host interruption");
  };
  const run = runFlow(loaded, { topic: "x" }, {
    log,
    onEvent: interrupt,
    env: (name) => (Object.hasOwn(runVars, name) ? runVars[name] : undefined),
    ...options,
  });
  await expect(run).rejects.toThrow("host interruption");
  return log;
}

/** Run `yian run <flow> --resume <log> --quiet`, stdin a fake TTY given `keys` when set. */
async function resume(flow: string, log: string, keys?: (input: FakeInput) => void) {
  const chunks: string[] = [];
  const stdout: string[] = [];
  const output = new FakeOutput();
  const input = new FakeInput({ tty: keys !== undefined });
  // Typed ahead: the terminal reads them once it prompts.
  keys?.(input);
  const io: CliIo = {
    stdout: { write: async (text) => void (chunks.push(text), stdout.push(text)) },
    stderr: { write: async (text) => void chunks.push(text) },
    stdin: input,
    stderrStream: output,
    env,
  };
  const code = await main(["run", flow, "--resume", log, "--quiet"], io);
  // Each prompt takes the terminal once, setting raw mode on and then off.
  const prompts = input.rawCalls.filter((mode) => mode).length;
  return { code, output: plain(chunks.join("")), stdout: stdout.join(""), prompts };
}

/** Remove styling and collapse whitespace, so wrapping can't split a message. */
function plain(text: string): string {
  return text.replace(ANSI, "").split(/\s+/).filter(Boolean).join(" ");
}

/** Assert the log is still recoverable and unlocked (Python's `_resumes_with_the_key`). */
async function resumesWithTheKey(flow: string, log: string): Promise<void> {
  vars["OPENAI_API_KEY"] = "sk-fake";
  const result = await withFetch(fakeModel, () => resume(flow, log));
  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
}

test("test_resume_without_the_key_exits_before_writing", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(
    "provider 'openai' needs an API key but $OPENAI_API_KEY is not set; set it and resume again",
  );
  expect(readFileSync(log)).toEqual(before);
  expect(reached).toEqual([]);
  await resumesWithTheKey(flow, log);
});

test("test_resume_with_a_changed_key_exits_before_writing", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);
  vars["OPENAI_API_KEY"] = "sk-other";

  const result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(
    "$OPENAI_API_KEY is not the key this run started with; set it to that key and resume again",
  );
  expect(readFileSync(log)).toEqual(before);
  expect(reached).toEqual([]);
});

test("test_resume_prompts_for_a_missing_key_on_a_tty", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);

  const result = await withFetch(fakeModel, () => resume(flow, log, (keys) => keys.write("sk-fake\r")));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
  expect(result.prompts).toBe(1);
  // The typed key is the one the run's model call sends.
  expect(authorizations).toEqual(["Bearer sk-fake"]);
});

test("test_resume_rejects_a_typed_key_that_does_not_match", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log, (keys) => keys.write("sk-other\r")));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(
    "the key entered is not the one this run started with; set $OPENAI_API_KEY to that key and resume again",
  );
  expect(readFileSync(log)).toEqual(before);
  expect(reached).toEqual([]);
});

test("test_resume_reprompts_on_empty_answers_then_gives_up", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log, (keys) => keys.write("\r  \r\r")));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("no API key provided for 'openai' ($OPENAI_API_KEY); set it and resume again");
  expect(result.prompts).toBe(3);
  expect(readFileSync(log)).toEqual(before);
});

test("test_resume_checks_other_providers_by_env_var", async () => {
  const flow = flowFile(tmp, "anthropic");
  const log = await unfinishedLog(flow, { envVar: "ANTHROPIC_API_KEY", key: "sk-ant" });
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(
    "provider 'anthropic' needs an API key but $ANTHROPIC_API_KEY is not set; set it and resume again",
  );
  expect(readFileSync(log)).toEqual(before);
});

test("test_resume_on_a_saved_api_key_login_needs_no_environment_key", async () => {
  const flow = flowFile();
  await new AuthStore(null, { env }).save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
  const log = await unfinishedLog(flow, { envVar: null });

  // Not a TTY: a prompt can't be answered.
  const result = await withFetch(fakeModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
});

test("test_agents_sharing_one_connection_get_one_prompt", async () => {
  const text = FLOW.replace(
    "output: ${write.output}",
    "  again:\n    kind: agent\n    input:\n      text: ${write.output}\n" +
      '    prompt: "Rewrite ${text}"\n    output: str\noutput: ${again.output}',
  );
  const flow = flowFile(tmp, "openai", text);
  const log = await unfinishedLog(flow);

  const result = await withFetch(fakeModel, () => resume(flow, log, (keys) => keys.write("sk-fake\r")));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
  expect(result.prompts).toBe(1);
});

test("test_resume_relays_a_reconnected_store_credential", async () => {
  const flow = flowFile();
  const store = new AuthStore(null, { env });
  await store.save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
  const log = await unfinishedLog(flow, { envVar: null });
  // A reconnect mints a new credential id, so the pinned one is gone.
  await store.save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("yian auth openai");
  expect(result.output).toContain("start a new run");
  expect(readFileSync(log)).toEqual(before);
});

test("test_resume_key_prompt_eof_leaves_the_log_unchanged", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log, (keys) => keys.end()));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("run cancelled");
  expect(readFileSync(log)).toEqual(before);
  await resumesWithTheKey(flow, log);
});

test("test_resume_key_prompt_ctrl_c_leaves_the_log_unchanged", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log, (keys) => keys.write("\x03")));

  expect(result.code, result.output).toBe(130);
  expect(result.output).toContain("cancelled by user");
  expect(readFileSync(log)).toEqual(before);
  await resumesWithTheKey(flow, log);
});

test("test_resume_of_a_host_key_run_points_to_python", async () => {
  const flow = flowFile();
  const llmConfig = { provider: "openai", model: "m", api_key: "sk-h" };
  const log = await unfinishedLog(flow, { envVar: null, llmConfig });
  const before = readFileSync(log);
  vars["OPENAI_API_KEY"] = "sk-h";

  const result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("resume_flow(api_key=...)");
  expect(result.output).toContain("cannot supply");
  expect(readFileSync(log)).toEqual(before);
});

test("test_resume_of_a_host_key_run_with_headers_names_both", async () => {
  const flow = flowFile();
  const llmConfig = { provider: "openai", model: "m", api_key: "sk-h", headers: { "X-Gateway": "gw-6a1f" } };
  const log = await unfinishedLog(flow, { envVar: null, llmConfig });
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("resume_flow(api_key=..., headers=...)");
  expect(result.output).not.toContain("gw-6a1f");
  expect(readFileSync(log)).toEqual(before);
});

test("test_resume_of_a_login_saved_again_with_other_settings_fails", async () => {
  const flow = flowFile();
  const store = new AuthStore(null, { env });
  const first = new ApiKeyCredentials("saved-key", { "X-Gateway": "gw-1" });
  await store.save("openai", first, "gpt-5.5", { organization: "org-a" });
  const log = await unfinishedLog(flow, { envVar: null });
  const second = new ApiKeyCredentials("saved-key", { "X-Other": "gw-2" });
  await store.save("openai", second, "gpt-5.5", { organization: "org-b" });
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("yian auth openai");
  expect(result.output).toContain("start a new run");
  expect(readFileSync(log)).toEqual(before);
});

test("test_terminal_resume_of_an_agent_run_skips_the_key_check", async () => {
  const flow = flowFile();
  const loaded = await loadFlow(readFileSync(flow, "utf8"), { searchPaths: [tmp] });
  const log = join(tmp, "finished.jsonl");
  const runVars: Record<string, string> = { ...vars, OPENAI_API_KEY: "sk-fake" };
  await withFetch(fakeModel, () =>
    runFlow(loaded, { topic: "x" }, { log, env: (name) => (Object.hasOwn(runVars, name) ? runVars[name] : undefined) }),
  );
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
  expect(readFileSync(log)).toEqual(before);
});

test("test_resume_checks_saved_login_of_non_openai_provider", async () => {
  const store = new AuthStore(null, { env });
  await store.save("xai", new ApiKeyCredentials("saved-xai"), "m", { makeDefault: false });
  const firstDir = join(tmp, "first");
  mkdirSync(firstDir);
  let flow = flowFile(firstDir, "xai");
  let log = await unfinishedLog(flow, { envVar: null });

  let result = await withFetch(fakeModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");

  const secondDir = join(tmp, "second");
  mkdirSync(secondDir);
  flow = flowFile(secondDir, "xai");
  log = await unfinishedLog(flow, { envVar: null });
  // A reconnect mints a new credential id, so the pinned one is gone.
  await store.save("xai", new ApiKeyCredentials("saved-xai"), "m", { makeDefault: false });
  let before = readFileSync(log);

  result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("yian auth xai");
  expect(readFileSync(log)).toEqual(before);

  await store.remove("xai");
  const thirdDir = join(tmp, "third");
  mkdirSync(thirdDir);
  flow = flowFile(thirdDir, "xai");
  log = await unfinishedLog(flow, { envVar: "XAI_API_KEY", key: "sk-xai" });
  before = readFileSync(log);

  result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("$XAI_API_KEY is not set");
  expect(readFileSync(log)).toEqual(before);
});

test("test_resume_of_keyless_vllm_run_needs_no_key", async () => {
  const flow = flowFile(tmp, "vllm");
  const log = await unfinishedLog(flow, { envVar: null });

  const result = await withFetch(fakeModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
});

const CALL_TIME_FLOW = FLOW.replace(
  '    prompt: "Write about ${text}"\n',
  "    prompt: \"Write about ${text}\"\n    llm_config: {model: '${prep.output}'}\n",
).replace("llm_config: {provider: PROVIDER, model: offline-model}", "llm_config: {provider: openai}");

test("test_resume_of_a_call_time_host_key_selection_points_to_python", async () => {
  const flow = flowFile(tmp, "openai", CALL_TIME_FLOW);
  const llmConfig = { provider: "openai", api_key: "sk-h", headers: { "X-Gateway": "gw-6a1f" } };
  const log = await unfinishedLog(flow, { envVar: null, llmConfig });
  const before = readFileSync(log);
  expect(before.toString()).toContain('"call_time_llm"');
  vars["OPENAI_API_KEY"] = "sk-h";

  const result = await withFetch(noModel, () => resume(flow, log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("this run's openai API key came from a host's llm_config.api_key");
  expect(result.output).toContain("resume_flow(api_key=..., headers=...)");
  expect(result.output).not.toContain("gw-6a1f");
  expect(result.output).not.toContain("sk-h");
  expect(readFileSync(log)).toEqual(before);
});

test("test_resume_of_a_call_time_selection_without_a_host_key_runs", async () => {
  const flow = flowFile(tmp, "openai", CALL_TIME_FLOW);
  const log = await unfinishedLog(flow);
  expect(readFileSync(log, "utf8")).toContain('"call_time_llm"');
  await resumesWithTheKey(flow, log);
});
