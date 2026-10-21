/**
 * CLI-owned session storage survives the command without using the run log as memory: a port of
 * `tests/cli/test_memory_persistence.py`, every test under its name.
 *
 * Changed:
 * - Each `yian run` is a `main` call in this process, not a child process; each builds and closes
 *   its own memory registry, so only the storage file carries history from one to the next. Only
 *   the run that dies is a child process (`bun` on a script in the temp directory), which exits
 *   with 42 from its fake model call.
 * - Python's patched model is a fake `fetch` answering as an OpenAI-compatible chat endpoint, so
 *   the flow's `llm_config` names xAI's route (its key in the command's env) instead of OpenAI's;
 *   the user messages are counted from the request body. With xAI's key set, the key checks pass
 *   unpatched.
 * - Python's `with SessionMemory(storage=...)` holder is a `SessionMemory` whose `openRun()` has
 *   opened the file, closed after the command.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";

const FLOW = `id: cli-session-restart
name: cli_session_restart
llm_config: {provider: xai, model: grok-4.3}
input:
  question: !str
nodes:
  answer:
    kind: agent
    long_term_memory: session
    input: {question: "\${input.question}"}
    prompt: \${question}
    output: str
output: \${answer.output}
`;

const STATUS_PREFIX = "Session history status (context only): ";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-persistence-"));
  mkdirSync(join(tmp, "home"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** A chat-completions stream replying `text`. */
function reply(text: string): string {
  return (
    [
      { choices: [{ index: 0, delta: { role: "assistant", content: text }, finish_reason: null }] },
      { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
    ]
      .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
      .join("") + "data: [DONE]\n\n"
  );
}

/** The text of a chat message's `content`, a string or a list of text parts. */
function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  return (content as { text?: string }[]).map((part) => part.text ?? "").join("");
}

/** Answer with the number of user messages the model saw, history status messages aside. */
async function fakeFetch(input: string | URL | Request, init?: RequestInit): Promise<Response> {
  const request = new Request(input as string, init);
  const body = (await request.json()) as { messages: { role: string; content: unknown }[] };
  const users = body.messages.filter((message) => message.role === "user").map((m) => contentText(m.content));
  const count = users.filter((text) => !text.startsWith(STATUS_PREFIX)).length;
  const unknown = users.some((text) => text.includes("completion unknown"));
  return new Response(reply(`Seen user messages: ${count}; Unknown history: ${unknown}`), {
    headers: { "content-type": "text/event-stream" },
  });
}

/** Run `yian run argv` with the fake model; return its code and output. */
async function invoke(argv: readonly string[]) {
  const out: string[] = [];
  const err: string[] = [];
  const vars: Record<string, string> = { HOME: join(tmp, "home"), XAI_API_KEY: "test-xai-key" };
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (Object.hasOwn(vars, name) ? vars[name] : undefined),
  };
  const send = globalThis.fetch;
  globalThis.fetch = fakeFetch as typeof fetch;
  try {
    const code = await main(["run", ...argv], io);
    return { code, stdout: out.join(""), stderr: err.join("") };
  } finally {
    globalThis.fetch = send;
  }
}

const DURABLE_ROW = "test_session_history_across_cli_processes_is_explicitly_durable (durable %p)";

test.each([false, true])(DURABLE_ROW, async (durable) => {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, FLOW);
  const storage = join(tmp, "session.sqlite");
  const options = durable ? ["--memory-storage", `session=${storage}`] : [];
  for (const index of [0, 1]) {
    const log = join(tmp, `run-${index}.jsonl`);
    const argv = [flow, "--quiet", "--input", `question=question-${index}`, "--log", log, ...options];
    const result = await invoke(argv);
    expect(result.code, result.stdout + result.stderr).toBe(0);
    const expected = durable ? index + 1 : 1;
    expect(result.stdout).toContain(`Seen user messages: ${expected}`);
    const records = readRunLog(log).map((envelope) => envelope.record);
    const started = records[0] as RunStartedRecord;
    expect(started).toBeInstanceOf(RunStartedRecord);
    expect(started.memories).not.toBeNull();
    expect(started.learnable).toBe(false);
    expect(records.some((record) => record.constructor.name.startsWith("Backward"))).toBe(false);
    expect(readFileSync(log, "utf8")).not.toContain(storage);
  }
  expect(existsSync(storage)).toBe(durable);
});

test("test_memory_root_keeps_session_history_under_the_flow_id", async () => {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, FLOW);
  const root = join(tmp, "mem");
  for (const index of [0, 1]) {
    const log = join(tmp, `run-${index}.jsonl`);
    const question = `question=question-${index}`;
    const result = await invoke([flow, "--quiet", "--input", question, "--log", log, "--memory-root", root]);
    expect(result.code, result.stdout + result.stderr).toBe(0);
    expect(result.stdout).toContain(`Seen user messages: ${index + 1}`);
  }
  expect(statSync(join(root, "cli-session-restart", "session.sqlite")).isFile()).toBe(true);
});

test("test_terminal_cli_resume_does_not_acquire_the_memory_store", async () => {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, FLOW);
  const storage = join(tmp, "session.sqlite");
  const log = join(tmp, "run.jsonl");
  const options = ["--memory-storage", `session=${storage}`];
  const first = await invoke([flow, "--quiet", "--input", "question=first", "--log", log, ...options]);
  expect(first.code, first.stdout + first.stderr).toBe(0);
  const prefix = readFileSync(log);
  // Opening a run opens the file, which the holder then owns until it closes.
  const holder = new SessionMemory({ storage });
  let inspected;
  try {
    holder.openRun().close();
    inspected = await invoke([flow, "--quiet", "--resume", log, ...options]);
  } finally {
    holder.close();
  }
  expect(inspected.code, inspected.stdout + inspected.stderr).toBe(0);
  expect(inspected.stdout).toContain("Seen user messages: 1");
  expect(readFileSync(log)).toEqual(prefix);
});

/** A script running `yian run` whose model call ends the process with exit code 42. */
const DYING_CHILD = `
import { main } from ${JSON.stringify(join(import.meta.dir, "../../src/cli/index.ts"))};
import { processIo } from ${JSON.stringify(join(import.meta.dir, "../../src/cli/utils.ts"))};
globalThis.fetch = async () => process.exit(42);
process.exitCode = await main(process.argv.slice(2), processIo());
`;

test("test_cli_resume_after_process_death_reuses_saved_history_and_repeats_request", async () => {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, FLOW);
  const storage = join(tmp, "session.sqlite");
  const log = join(tmp, "run.jsonl");
  const options = ["--memory-storage", `session=${storage}`];
  const script = join(tmp, "dying.ts");
  writeFileSync(script, DYING_CHILD);
  const argv = ["run", flow, "--quiet", "--input", "question=retry me", "--log", log, ...options];
  const crashed = spawnSync(process.execPath, [script, ...argv], {
    cwd: tmp,
    encoding: "utf8",
    timeout: 30_000,
    env: { PATH: process.env["PATH"] ?? "", HOME: join(tmp, "home"), XAI_API_KEY: "test-xai-key" },
  });
  expect(crashed.status, crashed.stdout + crashed.stderr).toBe(42);
  const prefix = readFileSync(log);

  const resumed = await invoke([flow, "--quiet", "--resume", log, ...options]);

  expect(resumed.code, resumed.stdout + resumed.stderr).toBe(0);
  expect(resumed.stdout).toContain("Seen user messages: 2");
  expect(resumed.stdout).toContain("Unknown history: true");
  const after = readFileSync(log);
  expect(after.subarray(0, prefix.length)).toEqual(prefix);
  expect(after.toString()).not.toContain(storage);
  const memory = new SessionMemory({ storage });
  try {
    const opened = memory.openRun();
    try {
      type Effect = { status: string; invocation: unknown };
      const { effects } = opened.snapshot["conversation"] as { effects: Effect[] };
      expect(effects.map((effect) => effect.status)).toEqual(["pending", "completed"]);
      expect(effects[0]!.invocation).not.toEqual(effects[1]!.invocation);
    } finally {
      opened.close();
    }
  } finally {
    memory.close();
  }
});
