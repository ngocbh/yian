/**
 * `yian run --resume` checks the recorded provider keys before it writes: environment keys, saved
 * logins, keys typed at a prompt, and keys only the host can supply. The model is a fake `fetch`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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

/** Answer the AGENT locally. */
async function fakeModel(input: string | URL | Request, init?: RequestInit): Promise<Response> {
  const request = new Request(input as string, init);
  authorizations.push(request.headers.get("authorization"));
  const body = request.url.endsWith("/responses") ? RESPONSES_REPLY : CHAT_REPLY;
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

/** Each request the failing model got. */
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
 * Record a run interrupted at `prep`, with `envVar` set to `key` only in its own env.
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

/** Run `yian run --resume <log> --quiet`, stdin a fake TTY given `keys` when set. */
async function resume(log: string, keys?: (input: FakeInput) => void) {
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
  const code = await main(["run", "--resume", log, "--quiet"], io);
  // Each prompt takes the terminal once, setting raw mode on and then off.
  const prompts = input.rawCalls.filter((mode) => mode).length;
  return { code, output: plain(chunks.join("")), stdout: stdout.join(""), prompts };
}

/** Remove styling and collapse whitespace, so wrapping can't split a message. */
function plain(text: string): string {
  return text.replace(ANSI, "").split(/\s+/).filter(Boolean).join(" ");
}

/** Assert the log is still recoverable and unlocked. */
async function resumesWithTheKey(flow: string, log: string): Promise<void> {
  vars["OPENAI_API_KEY"] = "sk-fake";
  const result = await withFetch(fakeModel, () => resume(log));
  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
}

test("resume without the key exits before writing", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(
    "provider 'openai' needs an API key but $OPENAI_API_KEY is not set; set it and resume again",
  );
  expect(readFileSync(log)).toEqual(before);
  expect(reached).toEqual([]);
  await resumesWithTheKey(flow, log);
});

test("resume with a changed key exits before writing", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);
  vars["OPENAI_API_KEY"] = "sk-other";

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(
    "$OPENAI_API_KEY is not the key this run started with; set it to that key and resume again",
  );
  expect(readFileSync(log)).toEqual(before);
  expect(reached).toEqual([]);
});

test("resume prompts for a missing key on a tty", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);

  const result = await withFetch(fakeModel, () => resume(log, (keys) => keys.write("sk-fake\r")));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
  expect(result.prompts).toBe(1);
  // The typed key is the one the run's model call sends.
  expect(authorizations).toEqual(["Bearer sk-fake"]);
});

test("resume rejects a typed key that does not match", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log, (keys) => keys.write("sk-other\r")));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(
    "the key entered is not the one this run started with; set $OPENAI_API_KEY to that key and resume again",
  );
  expect(readFileSync(log)).toEqual(before);
  expect(reached).toEqual([]);
});

test("resume reprompts on empty answers then gives up", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log, (keys) => keys.write("\r  \r\r")));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("no API key provided for 'openai' ($OPENAI_API_KEY); set it and resume again");
  expect(result.prompts).toBe(3);
  expect(readFileSync(log)).toEqual(before);
});

test("resume checks other providers by env var", async () => {
  const flow = flowFile(tmp, "anthropic");
  const log = await unfinishedLog(flow, { envVar: "ANTHROPIC_API_KEY", key: "sk-ant" });
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(
    "provider 'anthropic' needs an API key but $ANTHROPIC_API_KEY is not set; set it and resume again",
  );
  expect(readFileSync(log)).toEqual(before);
});

test("resume accepts Google's later key variable, as the Google client does", async () => {
  const flow = flowFile(tmp, "google");
  const log = await unfinishedLog(flow, { envVar: "GEMINI_API_KEY", key: "sk-gemini" });
  vars["GEMINI_API_KEY"] = "sk-gemini";

  const result = await withFetch(noModel, () => resume(log));

  // Past the key check, the run calls the model, which fails.
  expect(result.output).not.toContain("needs an API key");
  expect(reached.length).toBeGreaterThan(0);
});

test("resume on a saved api key login needs no environment key", async () => {
  const flow = flowFile();
  await new AuthStore(null, { env }).save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
  const log = await unfinishedLog(flow, { envVar: null });

  // Not a TTY: a prompt can't be answered.
  const result = await withFetch(fakeModel, () => resume(log));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
});

test("agents sharing one connection get one prompt", async () => {
  const text = FLOW.replace(
    "output: ${write.output}",
    "  again:\n    kind: agent\n    input:\n      text: ${write.output}\n" +
      '    prompt: "Rewrite ${text}"\n    output: str\noutput: ${again.output}',
  );
  const flow = flowFile(tmp, "openai", text);
  const log = await unfinishedLog(flow);

  const result = await withFetch(fakeModel, () => resume(log, (keys) => keys.write("sk-fake\r")));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
  expect(result.prompts).toBe(1);
});

test("resume relays a reconnected store credential", async () => {
  const flow = flowFile();
  const store = new AuthStore(null, { env });
  await store.save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
  const log = await unfinishedLog(flow, { envVar: null });
  // A reconnect mints a new credential id, so the pinned one is gone.
  await store.save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("yian auth openai");
  expect(result.output).toContain("start a new run");
  expect(readFileSync(log)).toEqual(before);
});

test("resume on a damaged store says only restoring the file continues the run", async () => {
  const flow = flowFile();
  await new AuthStore(null, { env }).save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
  const log = await unfinishedLog(flow, { envVar: null });
  const auth = vars["YIAN_AUTH_FILE"]!;
  const saved = readFileSync(auth);
  writeFileSync(auth, '{"version": 1, "provi');
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(`Credential storage at ${auth} is malformed`);
  // Deleting the logins and logging in again mints a new credential id, which the resume refuses.
  expect(result.output).toContain(
    "Only restoring the file lets this run resume; after deleting its logins, log in again and start a new run",
  );
  expect(result.output).not.toContain("Resume again");
  expect(readFileSync(log)).toEqual(before);

  writeFileSync(auth, saved);
  const restored = await withFetch(fakeModel, () => resume(log));
  expect(restored.code, restored.output).toBe(0);
  expect(restored.stdout).toContain("written offline");
});

test("resume on a store folder others can read says to resume once it is fixed", async () => {
  const flow = flowFile();
  await new AuthStore(null, { env }).save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
  const log = await unfinishedLog(flow, { envVar: null });
  const folder = join(tmp, "auth");
  chmodSync(folder, 0o755);

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain(`fix it with: chmod 700 ${folder}. Resume again once it is fixed`);
  expect(result.output).not.toContain("start a new run");

  chmodSync(folder, 0o700);
  const fixed = await withFetch(fakeModel, () => resume(log));
  expect(fixed.code, fixed.output).toBe(0);
});

test("resume key prompt eof leaves the log unchanged", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log, (keys) => keys.end()));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("run cancelled");
  expect(readFileSync(log)).toEqual(before);
  await resumesWithTheKey(flow, log);
});

test("resume key prompt ctrl c leaves the log unchanged", async () => {
  const flow = flowFile();
  const log = await unfinishedLog(flow);
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log, (keys) => keys.write("\x03")));

  expect(result.code, result.output).toBe(130);
  expect(result.output).toContain("cancelled by user");
  expect(readFileSync(log)).toEqual(before);
  await resumesWithTheKey(flow, log);
});

test("resume of a host key run says the host must resume it", async () => {
  const flow = flowFile();
  const llmConfig = { provider: "openai", model: "m", api_key: "sk-h" };
  const log = await unfinishedLog(flow, { envVar: null, llmConfig });
  const before = readFileSync(log);
  vars["OPENAI_API_KEY"] = "sk-h";

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("advance({ log, apiKey })");
  expect(result.output).toContain("cannot supply");
  expect(readFileSync(log)).toEqual(before);
});

test("resume of a host key run with headers names both", async () => {
  const flow = flowFile();
  const llmConfig = { provider: "openai", model: "m", api_key: "sk-h", headers: { "X-Gateway": "gw-6a1f" } };
  const log = await unfinishedLog(flow, { envVar: null, llmConfig });
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("advance({ log, apiKey, headers })");
  expect(result.output).not.toContain("gw-6a1f");
  expect(readFileSync(log)).toEqual(before);
});

test("resume of a login saved again with other settings fails", async () => {
  const flow = flowFile();
  const store = new AuthStore(null, { env });
  const first = new ApiKeyCredentials("saved-key", { "X-Gateway": "gw-1" });
  await store.save("openai", first, "gpt-5.5", { organization: "org-a" });
  const log = await unfinishedLog(flow, { envVar: null });
  const second = new ApiKeyCredentials("saved-key", { "X-Other": "gw-2" });
  await store.save("openai", second, "gpt-5.5", { organization: "org-b" });
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("yian auth openai");
  expect(result.output).toContain("start a new run");
  expect(readFileSync(log)).toEqual(before);
});

test("terminal resume of an agent run skips the key check", async () => {
  const flow = flowFile();
  const loaded = await loadFlow(readFileSync(flow, "utf8"), { searchPaths: [tmp] });
  const log = join(tmp, "finished.jsonl");
  const runVars: Record<string, string> = { ...vars, OPENAI_API_KEY: "sk-fake" };
  await withFetch(fakeModel, () =>
    runFlow(loaded, { topic: "x" }, { log, env: (name) => (Object.hasOwn(runVars, name) ? runVars[name] : undefined) }),
  );
  const before = readFileSync(log);

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
  expect(readFileSync(log)).toEqual(before);
});

test("resume checks saved login of non openai provider", async () => {
  const store = new AuthStore(null, { env });
  await store.save("xai", new ApiKeyCredentials("saved-xai"), "m", { makeDefault: false });
  const firstDir = join(tmp, "first");
  mkdirSync(firstDir);
  let flow = flowFile(firstDir, "xai");
  let log = await unfinishedLog(flow, { envVar: null });

  let result = await withFetch(fakeModel, () => resume(log));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");

  const secondDir = join(tmp, "second");
  mkdirSync(secondDir);
  flow = flowFile(secondDir, "xai");
  log = await unfinishedLog(flow, { envVar: null });
  // A reconnect mints a new credential id, so the pinned one is gone.
  await store.save("xai", new ApiKeyCredentials("saved-xai"), "m", { makeDefault: false });
  let before = readFileSync(log);

  result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("yian auth xai");
  expect(readFileSync(log)).toEqual(before);

  await store.remove("xai");
  const thirdDir = join(tmp, "third");
  mkdirSync(thirdDir);
  flow = flowFile(thirdDir, "xai");
  log = await unfinishedLog(flow, { envVar: "XAI_API_KEY", key: "sk-xai" });
  before = readFileSync(log);

  result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("$XAI_API_KEY is not set");
  expect(readFileSync(log)).toEqual(before);
});

test("resume of keyless vllm run needs no key", async () => {
  const flow = flowFile(tmp, "vllm");
  const log = await unfinishedLog(flow, { envVar: null });

  const result = await withFetch(fakeModel, () => resume(log));

  expect(result.code, result.output).toBe(0);
  expect(result.stdout).toContain("written offline");
});

const CALL_TIME_FLOW = FLOW.replace(
  '    prompt: "Write about ${text}"\n',
  "    prompt: \"Write about ${text}\"\n    llm_config: {model: '${prep.output}'}\n",
).replace("llm_config: {provider: PROVIDER, model: offline-model}", "llm_config: {provider: openai}");

test("resume of a call time host key selection says the host must resume it", async () => {
  const flow = flowFile(tmp, "openai", CALL_TIME_FLOW);
  const llmConfig = { provider: "openai", api_key: "sk-h", headers: { "X-Gateway": "gw-6a1f" } };
  const log = await unfinishedLog(flow, { envVar: null, llmConfig });
  const before = readFileSync(log);
  expect(before.toString()).toContain('"call_time_llm"');
  vars["OPENAI_API_KEY"] = "sk-h";

  const result = await withFetch(noModel, () => resume(log));

  expect(result.code, result.output).toBe(1);
  expect(result.output).toContain("this run's openai API key came from a host's llm_config.api_key");
  expect(result.output).toContain("advance({ log, apiKey, headers })");
  expect(result.output).not.toContain("gw-6a1f");
  expect(result.output).not.toContain("sk-h");
  expect(readFileSync(log)).toEqual(before);
});

test("resume of a call time selection without a host key runs", async () => {
  const flow = flowFile(tmp, "openai", CALL_TIME_FLOW);
  const log = await unfinishedLog(flow);
  expect(readFileSync(log, "utf8")).toContain('"call_time_llm"');
  await resumesWithTheKey(flow, log);
});
