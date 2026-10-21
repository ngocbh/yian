/**
 * Durable backends retain memory through the public run and recovery boundary: mirrors
 * `tests/engine/test_memory_storage_integration.py`.
 *
 * Ported: `test_process_local_backends_never_open_sqlite` (3 rows),
 * `test_snapshot_identity_survives_a_real_process_restart` (3 rows) and
 * `test_idle_run_does_not_keep_storage_owned_after_backend_close` (3 rows); through `runFlow`:
 * `test_api_session_history_is_used_after_reopening_storage`,
 * `test_api_summary_cache_survives_reopen_under_current_budget` and
 * `test_more_recent_turns_on_reopen_restores_raw_context`; through `runFlow` and `resumeFlow`:
 * `test_resume_reopens_latest_saved_history_instead_of_the_old_run_snapshot`.
 *
 * Changed: Python's `with <backend>:` is `withBackend`, which closes the backend in `finally`.
 * `test_process_local_backends_never_open_sqlite`: spies on `SQLiteStorage.prototype.ensureOpen`,
 * `load` and `save` that throw stand in for Python forbidding `sqlite3.connect`.
 * `test_snapshot_identity_survives_a_real_process_restart`: the child is a `bun -e` script
 * importing the backend's module that prints the snapshot as `logJson`. The host stop of the
 * resume row aborts `runFlow`'s `signal` once the first reply's last event is read (Python sets
 * the node's stop event inside the model call).
 *
 * Later (T6-6, with RunResult.backward): `test_explicit_learning_remains_saved_after_reopen` (2 rows)
 * and `test_api_session_learned_notes_survive_storage_reopen`.
 */

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type JsonObject, MemoryOperationError } from "../../src/memory/contracts.ts";
import { EpisodicMemory } from "../../src/memory/episodic.ts";
import { ReflectiveMemory } from "../../src/memory/reflective.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { SQLiteStorage } from "../../src/memory/storage/sqlite.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { rejected, reply, resumeWith, runWith, sentTexts, snapshot, StopAfterFirstReply } from "./session_fakes.ts";

const MODEL_CONFIG = { provider: "openai", model: "storage-test-model" };
const AGENT_FLOW = `id: durable-session
name: durable_session
llm_config: {provider: openai, model: storage-test-model}
input:
  question: !str
nodes:
  agent:
    kind: agent
    long_term_memory: session
    prompt: \${question}
    input:
      question: \${input.question}
    output: str
output: \${agent.output}
`;

const POLICIES = ["episodic", "reflective", "session"] as const;
type Policy = (typeof POLICIES)[number];
type Backend = EpisodicMemory | ReflectiveMemory | SessionMemory;

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-memory-storage-integration-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

/** Python's `_backend`: the `policy` backend over `storage`. */
function backend(policy: Policy, storage: string | null): Backend {
  if (policy === "reflective") return new ReflectiveMemory({ llmConfig: MODEL_CONFIG, storage });
  return policy === "episodic" ? new EpisodicMemory({ storage }) : new SessionMemory({ storage });
}

/** Await `fn` with `memory`, then close it (Python's `with <backend>:`). */
async function withBackend<M extends Backend, T>(memory: M, fn: (memory: M) => T | Promise<T>): Promise<T> {
  try {
    return await fn(memory);
  } finally {
    memory.close();
  }
}

/** The texts of every message of `model`'s request `index` (the last by default), system prompt included. */
function context(model: ScriptedModel, index = -1): string[] {
  return model.requests.at(index)!.messages.map((message) => message.text());
}

for (const policy of POLICIES) {
  test(`process local backends never open sqlite: ${policy}`, async () => {
    const spies = (["ensureOpen", "load", "save"] as const).map((name) =>
      spyOn(SQLiteStorage.prototype, name).mockImplementation(() => {
        throw new Error("process-local memory opened SQLite");
      }),
    );
    try {
      await withBackend(backend(policy, null), snapshot);
      for (const spy of spies) expect(spy).not.toHaveBeenCalled();
    } finally {
      for (const spy of spies) spy.mockRestore();
    }
  });

  test(`snapshot identity survives a real process restart: ${policy}`, async () => {
    const file = path.join(dir, "private-memory.sqlite");
    const module = (name: string) => JSON.stringify(pathToFileURL(path.join(import.meta.dir, "../../src", name)).href);
    const script = `
const { EpisodicMemory } = await import(${module("memory/episodic.ts")});
const { ReflectiveMemory } = await import(${module("memory/reflective.ts")});
const { SessionMemory } = await import(${module("memory/session.ts")});
const { logJson } = await import(${module("typesys/json.ts")});
const [policy, storage] = process.argv.slice(-2);
const memory = policy === "reflective"
  ? new ReflectiveMemory({ storage, llmConfig: ${JSON.stringify(MODEL_CONFIG)} })
  : policy === "episodic" ? new EpisodicMemory({ storage }) : new SessionMemory({ storage });
const opened = memory.openRun();
console.log(logJson(opened.snapshot));
opened.close();
memory.close();
`;
    const result = spawnSync(process.execPath, ["-e", script, policy, file], { encoding: "utf8", timeout: 20_000 });
    expect(result.stderr).toBe("");
    expect(result.status).toBe(0);
    const recorded = parseJson(result.stdout, "plain");
    await withBackend(backend(policy, file), (reopened) => expect(snapshot(reopened)).toEqual(recorded as JsonObject));
    expect(logJson(recorded)).not.toContain(file);
  });

  test(`idle run does not keep storage owned after backend close: ${policy}`, async () => {
    const file = path.join(dir, "memory.sqlite");
    await withBackend(backend(policy, file), async (memory) => {
      const opened = memory.openRun();
      const recorded = opened.snapshot;
      memory.close();
      expect(() => opened.snapshot).toThrow(MemoryOperationError);
      await withBackend(backend(policy, file), (reopened) => expect(snapshot(reopened)).toEqual(recorded));
      opened.close();
    });
  });
}

test("api session history is used after reopening storage", async () => {
  const model = new ScriptedModel([reply("reply 1"), reply("reply 2")], MODEL_CONFIG);
  const file = path.join(dir, "private-session.sqlite");
  const flow = await loadFlow(AGENT_FLOW);
  const recorded = await withBackend(new SessionMemory({ storage: file }), async (memory) => {
    const first = await runWith(flow, { question: "first" }, memory, model);
    expect(first.output).toBe("reply 1");
    expect(first.learning).toBeNull();
    return snapshot(memory);
  });
  await withBackend(new SessionMemory({ storage: file }), async (reopened) => {
    expect(snapshot(reopened)).toEqual(recorded);
    const second = await runWith(flow, { question: "second" }, reopened, model);
    expect(second.output).toBe("reply 2");
    expect(second.learning).toBeNull();
    expect(sentTexts(model, 1)).toEqual(["first", "reply 1", "second"]);
    expect((snapshot(reopened)["conversation"] as JsonObject)["items"]).toHaveLength(6);
  });
});

test("resume reopens the latest saved history instead of the old run snapshot", async () => {
  const controller = new AbortController();
  const model = new StopAfterFirstReply(
    [reply("reply saved before cancellation"), reply("retry complete")],
    controller,
    MODEL_CONFIG,
  );
  const flow = await loadFlow(AGENT_FLOW);
  const file = path.join(dir, "memory.sqlite");
  const log = path.join(dir, "run.jsonl");
  const saved = await withBackend(new SessionMemory({ storage: file }), async (memory) => {
    const options = { log, signal: controller.signal };
    const stop = await rejected(runWith(flow, { question: "question" }, memory, model, options));
    expect(stop).toBe(controller.signal.reason);
    const recorded = snapshot(memory);
    expect(((recorded["conversation"] as JsonObject)["effects"] as JsonObject[])[0]!["status"]).toBe("completed");
    return recorded;
  });
  const prefix = fs.readFileSync(log);
  await withBackend(new SessionMemory({ storage: file }), async (reopened) => {
    expect(snapshot(reopened)).toEqual(saved);
    const resumed = await resumeWith(flow, log, reopened, model);
    expect(resumed.output).toBe("retry complete");
    const texts = context(model);
    expect(texts).toContain("reply saved before cancellation");
    expect(texts.filter((text) => text === "question")).toHaveLength(2);
    const effects = (snapshot(reopened)["conversation"] as JsonObject)["effects"] as JsonObject[];
    expect(effects).toHaveLength(2);
    expect(effects[0]!["invocation"]).not.toEqual(effects[1]!["invocation"]);
  });
  const written = fs.readFileSync(log);
  expect(written.subarray(0, prefix.length)).toEqual(prefix);
  expect(written.includes(file)).toBe(false);
});

test("api summary cache survives reopen under current budget", async () => {
  const summary = "Saved historical summary";
  const model = new ScriptedModel(
    [
      ...[0, 1, 2, 3, 4].map((index) => reply(`answer ${index}`)),
      reply(summary),
      reply("answer 5"),
      reply("answer 6"),
    ],
    MODEL_CONFIG,
  );
  const file = path.join(dir, "session.sqlite");
  const flow = await loadFlow(AGENT_FLOW);
  const raw = await withBackend(new SessionMemory({ storage: file, historyBudgetChars: 1 }), async (memory) => {
    for (let index = 0; index < 6; index++) {
      const result = await runWith(flow, { question: `question ${index}` }, memory, model);
      expect(result.output).toBe(`answer ${index}`);
    }
    expect(model.requests).toHaveLength(7);
    const items = (snapshot(memory)["conversation"] as JsonObject)["items"] as JsonObject[];
    expect(logJson(items)).not.toContain(summary);
    return items;
  });
  await withBackend(new SessionMemory({ storage: file, historyBudgetChars: 100_000 }), async (reopened) => {
    const result = await runWith(flow, { question: "question 6" }, reopened, model);
    expect(result.output).toBe("answer 6");
    expect(model.requests).toHaveLength(8);
    const texts = context(model);
    expect(texts.some((text) => text.includes(summary))).toBe(true);
    expect(texts).not.toContain("question 0");
    for (let index = 1; index <= 6; index++) expect(texts).toContain(`question ${index}`);
    const items = (snapshot(reopened)["conversation"] as JsonObject)["items"] as JsonObject[];
    expect(items.slice(0, raw.length)).toEqual(raw);
  });
});

test("more recent turns on reopen restores raw context", async () => {
  const summary = "Previously summarized first turn";
  const model = new ScriptedModel(
    [reply("answer 0"), reply(summary), reply("answer 1"), reply("answer 2")],
    MODEL_CONFIG,
  );
  const file = path.join(dir, "session.sqlite");
  const flow = await loadFlow(AGENT_FLOW);
  await withBackend(new SessionMemory({ storage: file, historyBudgetChars: 1, recentTurns: 0 }), async (memory) => {
    for (let index = 0; index < 2; index++) {
      const result = await runWith(flow, { question: `question ${index}` }, memory, model);
      expect(result.output).toBe(`answer ${index}`);
    }
    expect(model.requests).toHaveLength(3);
  });
  await withBackend(
    new SessionMemory({ storage: file, historyBudgetChars: 100_000, recentTurns: 4 }),
    async (reopened) => {
      const result = await runWith(flow, { question: "question 2" }, reopened, model);
      expect(result.output).toBe("answer 2");
      expect(model.requests).toHaveLength(4);
      expect(sentTexts(model)).toEqual(["question 0", "answer 0", "question 1", "answer 1", "question 2"]);
    },
  );
});
