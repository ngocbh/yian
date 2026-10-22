/**
 * Named memories end to end, YAML to load to `buildMemories` to run to backward to rerun: mirrors
 * `tests/engine/test_named_memory_e2e.py`.
 *
 * Ported, through `runFlow` and `RunResult.backward`: `test_python_api_named_memories_persist_and_stack`.
 * Changed: the model is a `ChatModel` from `models` (Python patches `model_from_config`); the runs'
 * environment is empty, so no `HOME` is set aside.
 *
 * Ported through the CLI, each `yian run` a `main` call (`cli`), as
 * `tests/cli/memory_persistence.test.ts` runs it: `test_cli_memory_root_persists_session_across_processes`
 * and `test_cli_memory_storage_override_places_one_key`. Changed: each command builds and closes
 * its own memory registry in this process, so only the storage files carry the session from one
 * to the next (Python runs a child process per command); the model is a fake `fetch` answering as
 * xAI's chat endpoint (`cliFetch`), so the flows' `llm_config` names xAI, its key in the command's
 * env, where Python patches `model_from_config`; the printed output is its text form, compact JSON
 * (design 3.2.3), not Python's `repr`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import type { Message } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { buildMemories } from "../../src/memory_host.ts";
import { memorySections, reply, runWith } from "./session_fakes.ts";

const ROOT_FLOW = `id: e2e-memory
name: e2e_memory
long_term_memory:
  notes:
    type: episodic
    config: {recall_limit: 3}
uses:
  lib: notes-lib
input:
  question: !str
nodes:
  ask:
    kind: agent
    llm_config: {provider: openai, model: fake-model}
    long_term_memory: {chat: session}
    input: {question: "\${input.question}"}
    prompt: \${question}
    output: str
  lib_step:
    kind: call
    call: lib
    input: {topic: "\${input.question}"}
output:
  answer: \${ask.output}
  summary: \${lib_step.output}
`;

const LIB_FLOW = `id: notes-lib
name: notes_lib
long_term_memory: {notes: episodic}
input:
  topic: !str
nodes:
  summarize:
    kind: agent
    llm_config: {provider: openai, model: fake-model}
    input: {topic: "\${input.topic}"}
    prompt: Summarize \${topic}.
    output: str
output: \${summarize.output}
`;

const STATUS_PREFIX = "Session history status (context only): ";

/** Describe what the model saw: real user turns and recalled memory keys (Python's `_reply`). */
function describeSeen(messages: readonly Message[]) {
  const users = messages.filter((message) => message.role === "user" && !message.text().startsWith(STATUS_PREFIX));
  const keys: string[] = [];
  for (const section of memorySections(messages)) if (!keys.includes(section.memory)) keys.push(section.memory);
  return reply(`users=${users.length} memories=${keys.join(",")}`);
}

/** Python's `_Model`: answers each request with `describeSeen`. */
const MODEL: ChatModel = {
  provider: "openai",
  model: "fake-model",
  stream: (request) => new ScriptedModel([describeSeen(request.messages)]).stream(request),
};

/** The paths, relative to `dir` and sorted, of the `.sqlite` files under it. */
function sqliteFiles(dir: string): string[] {
  return (readdirSync(dir, { recursive: true }) as string[]).filter((path) => path.endsWith(".sqlite")).sort();
}

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-named-memory-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

test("the API's named memories persist and stack", async () => {
  writeFileSync(join(tmp, "notes-lib.yaml"), LIB_FLOW);
  const flow = await loadFlow(ROOT_FLOW, { searchPaths: [tmp], sourceLabel: "flow.yaml" });
  const root = join(tmp, "mem");

  {
    await using memory = await buildMemories(flow, { root });
    expect(Object.fromEntries(memory.types)).toEqual({ chat: "session", "lib.notes": "episodic", notes: "episodic" });
    expect(existsSync(root)).toBe(false); // construction is lazy
    const first = await runWith(flow, { question: "q1" }, memory, MODEL, { learnable: true });
    expect(first.status).toBe("succeeded");
    expect(first.output).toEqual({ answer: "users=1 memories=", summary: "users=1 memories=" });
    const outcome = await first.backward();
    expect(outcome.status).toBe("succeeded");
    expect(outcome.learning).not.toBeNull();
    expect(outcome.learning!.status).toBe("succeeded");
  }

  expect(sqliteFiles(join(root, "e2e-memory"))).toEqual(["chat.sqlite", join("lib", "notes.sqlite"), "notes.sqlite"]);

  // A fresh registry over the same root sees the learned episodes and the session.
  let second;
  {
    await using memory = await buildMemories(flow, { root });
    second = await runWith(flow, { question: "q2" }, memory, MODEL);
  }
  expect(second.status).toBe("succeeded");
  const output = second.output as { answer: string; summary: string };
  // ask: one root-flow memory, plus two session user turns.
  expect(output.answer).toBe("users=2 memories=notes");
  // summarize: outer root memory first, then its own file's scoped memory.
  expect(output.summary).toBe("users=1 memories=notes,lib.notes");
});

/** The flows the CLI runs: each AGENT on xAI's chat route, which `cliFetch` answers. */
function writeFlows(dir: string): string {
  const xai = (text: string) =>
    text.replaceAll("{provider: openai, model: fake-model}", "{provider: xai, model: grok-4.3}");
  writeFileSync(join(dir, "notes-lib.yaml"), xai(LIB_FLOW));
  const flow = join(dir, "flow.yaml");
  writeFileSync(flow, xai(ROOT_FLOW));
  return flow;
}

/** A chat-completions stream replying `text`. */
function chatStream(text: string): string {
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

/** Python's `_Model` behind the CLI: a fake `fetch` answering what the model saw, as `describeSeen`. */
async function cliFetch(input: string | URL | Request, init?: RequestInit): Promise<Response> {
  const body = (await new Request(input as string, init).json()) as { messages: { role: string; content: unknown }[] };
  const texts = (role: string) => body.messages.filter((m) => m.role === role).map((m) => contentText(m.content));
  const users = texts("user").filter((text) => !text.startsWith(STATUS_PREFIX));
  const system = texts("system")[0] ?? "";
  const keys: string[] = [];
  if (system.includes("Memory context (canonical JSON")) {
    const payload = JSON.parse(system.slice(system.lastIndexOf("\n") + 1)) as { sections: { memory: string }[] };
    for (const section of payload.sections) if (!keys.includes(section.memory)) keys.push(section.memory);
  }
  return new Response(chatStream(`users=${users.length} memories=${keys.join(",")}`), {
    headers: { "content-type": "text/event-stream" },
  });
}

/** Run `yian run argv` with `cliFetch` as the model; return its code and output. */
async function cli(argv: readonly string[]) {
  const out: string[] = [];
  const err: string[] = [];
  mkdirSync(join(tmp, "home"), { recursive: true });
  const vars: Record<string, string> = { HOME: join(tmp, "home"), XAI_API_KEY: "test-xai-key" };
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (Object.hasOwn(vars, name) ? vars[name] : undefined),
  };
  const send = globalThis.fetch;
  globalThis.fetch = cliFetch as typeof fetch;
  try {
    const code = await main(["run", ...argv], io);
    return { code, stdout: out.join(""), stderr: err.join("") };
  } finally {
    globalThis.fetch = send;
  }
}

/** The exact line `yian run` prints for this flow's output. */
function resultLine(answerUsers: number): string {
  // The CLI never runs backward, so no episodic memory has anything to recall.
  return `{"answer":"users=${answerUsers} memories=","summary":"users=1 memories="}`;
}

test("the CLI's memory root persists the session across processes", async () => {
  const flow = writeFlows(tmp);
  const root = join(tmp, "mem");
  const outputs: string[][] = [];
  for (const [index, question] of ["first", "second"].entries()) {
    const log = join(tmp, `run-${index}.jsonl`);
    const result = await cli([flow, "--quiet", "--input", `question=${question}`, "--memory-root", root, "--log", log]);
    expect(result.code, result.stdout + result.stderr).toBe(0);
    outputs.push(result.stdout.split("\n"));
  }
  expect(outputs[0]).toContain(resultLine(1));
  expect(outputs[1]).toContain(resultLine(2)); // session history came from chat.sqlite
  const base = join(root, "e2e-memory");
  expect(existsSync(join(base, "chat.sqlite"))).toBe(true);
  expect(existsSync(join(base, "notes.sqlite"))).toBe(true);
  expect(existsSync(join(base, "lib", "notes.sqlite"))).toBe(true);
});

test("the CLI's memory storage override places one key", async () => {
  const flow = writeFlows(tmp);
  const root = join(tmp, "mem");
  const chat = join(tmp, "chat.sqlite");
  let stdout = "";
  for (const index of [0, 1]) {
    const log = join(tmp, `run-${index}.jsonl`);
    const argv = [flow, "--quiet", "--input", "question=hi", "--memory-root", root, "--memory-storage", `chat=${chat}`];
    const result = await cli([...argv, "--log", log]);
    expect(result.code, result.stdout + result.stderr).toBe(0);
    stdout = result.stdout;
  }
  expect(stdout.split("\n")).toContain(resultLine(2));
  expect(existsSync(chat)).toBe(true);
  expect(existsSync(join(root, "e2e-memory", "chat.sqlite"))).toBe(false);
});
