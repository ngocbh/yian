/**
 * Named memories end to end, YAML to load to `buildMemories` to run to backward to rerun: mirrors
 * `tests/engine/test_named_memory_e2e.py`.
 *
 * Ported, through `runFlow` and `RunResult.backward`: `test_python_api_named_memories_persist_and_stack`.
 * Changed: the model is a `ChatModel` from `models` (Python patches `model_from_config`); the runs'
 * environment is empty, so no `HOME` is set aside.
 *
 * Later (T7-1 PR 2, the CLI): `test_cli_memory_root_persists_session_across_processes` and
 * `test_cli_memory_storage_override_places_one_key`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import type { Message } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { buildMemories } from "../../src/memory_host.ts";
import { reply, runWith } from "./session_fakes.ts";

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

/** Describe what the model saw: real user turns and recalled memory keys. */
function describe(messages: readonly Message[]) {
  const users = messages.filter((message) => message.role === "user" && !message.text().startsWith(STATUS_PREFIX));
  const keys: string[] = [];
  const system = messages.find((message) => message.role === "system")?.text() ?? "";
  if (system.includes("Memory context (canonical JSON")) {
    const payload = JSON.parse(system.slice(system.lastIndexOf("\n") + 1)) as { sections: { memory: string }[] };
    for (const section of payload.sections) if (!keys.includes(section.memory)) keys.push(section.memory);
  }
  return reply(`users=${users.length} memories=${keys.join(",")}`);
}

/** Python's `_Model`: answers each request with `describe`. */
const MODEL: ChatModel = {
  provider: "openai",
  model: "fake-model",
  stream: (request) => new ScriptedModel([describe(request.messages)]).stream(request),
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
