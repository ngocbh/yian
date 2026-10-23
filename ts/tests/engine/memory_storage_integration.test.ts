/**
 * Durable backends retain memory through the public run and recovery boundary: mirrors
 * `tests/engine/test_memory_storage_integration.py`.
 *
 * Ported: the episodic and reflective rows of `test_process_local_backends_never_open_sqlite`,
 * `test_snapshot_identity_survives_a_real_process_restart` and
 * `test_idle_run_does_not_keep_storage_owned_after_backend_close`; through `runFlow`,
 * `RunResult.backward` and `resumeFlow`: `test_explicit_learning_remains_saved_after_reopen`
 * (2 rows).
 *
 * Not ported: the session rows, `test_api_session_history_is_used_after_reopening_storage`,
 * `test_api_summary_cache_survives_reopen_under_current_budget`,
 * `test_more_recent_turns_on_reopen_restores_raw_context`,
 * `test_resume_reopens_latest_saved_history_instead_of_the_old_run_snapshot` and
 * `test_api_session_learned_notes_survive_storage_reopen` (TS yian has no session memory).
 *
 * Changed: Python's `with <backend>:` is `withBackend`, which closes the backend in `finally`.
 * `test_process_local_backends_never_open_sqlite`: spies on `SQLiteStorage.prototype.ensureOpen`,
 * `load` and `save` that throw stand in for Python forbidding `sqlite3.connect`.
 * `test_snapshot_identity_survives_a_real_process_restart`: the child is a `bun -e` script
 * importing the backend's module that prints the snapshot as `logJson`.
 * `test_explicit_learning_remains_saved_after_reopen`:
 * the reflector is a `ScriptedModel` from the backend's `createModel` option, answering with the
 * actions' JSON (Python patches `reflective.model_from_config` with a structured-output fake).
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
import { SQLiteStorage } from "../../src/memory/storage/sqlite.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { memorySnapshot, NO_ENV, reply } from "./run_state_fixtures.ts";

const MODEL_CONFIG = { provider: "openai", model: "storage-test-model" };
const CODE_FLOW = `id: durable-learning
name: durable_learning
long_term_memory: POLICY
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: \${work.output}
`;

const POLICIES = ["episodic", "reflective"] as const;
type Policy = (typeof POLICIES)[number];
type Backend = EpisodicMemory | ReflectiveMemory;

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-memory-storage-integration-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

/** Python's `_backend`: the `policy` backend over `storage`; a reflective one learns with `createModel`. */
function backend(policy: Policy, storage: string | null, createModel: (() => ScriptedModel) | null = null): Backend {
  if (policy === "reflective") return new ReflectiveMemory({ llmConfig: MODEL_CONFIG, storage, createModel });
  return new EpisodicMemory({ storage });
}

/** Await `fn` with `memory`, then close it (Python's `with <backend>:`). */
async function withBackend<M extends Backend, T>(memory: M, fn: (memory: M) => T | Promise<T>): Promise<T> {
  try {
    return await fn(memory);
  } finally {
    memory.close();
  }
}

for (const policy of POLICIES) {
  test(`process local backends never open sqlite: ${policy}`, async () => {
    const spies = (["ensureOpen", "load", "save"] as const).map((name) =>
      spyOn(SQLiteStorage.prototype, name).mockImplementation(() => {
        throw new Error("process-local memory opened SQLite");
      }),
    );
    try {
      await withBackend(backend(policy, null), memorySnapshot);
      for (const spy of spies) expect(spy).not.toHaveBeenCalled();
    } finally {
      for (const spy of spies) spy.mockRestore();
    }
  });

  test(`snapshot identity survives a real process restart: ${policy}`, async () => {
    const file = path.join(dir, "private-memory.sqlite");
    const module = (name: string) =>
      JSON.stringify(pathToFileURL(path.join(import.meta.dirname, "../../src", name)).href);
    const script = `
const { EpisodicMemory } = await import(${module("memory/episodic.ts")});
const { ReflectiveMemory } = await import(${module("memory/reflective.ts")});
const { logJson } = await import(${module("typesys/json.ts")});
const [policy, storage] = process.argv.slice(-2);
const memory = policy === "reflective"
  ? new ReflectiveMemory({ storage, llmConfig: ${JSON.stringify(MODEL_CONFIG)} })
  : new EpisodicMemory({ storage });
const opened = memory.openRun();
console.log(logJson(opened.snapshot));
opened.close();
memory.close();
`;
    const result = spawnSync(process.execPath, ["-e", script, policy, file], { encoding: "utf8", timeout: 20_000 });
    expect(result.stderr).toBe("");
    expect(result.status).toBe(0);
    const recorded = parseJson(result.stdout, "plain");
    await withBackend(backend(policy, file), (reopened) => expect(memorySnapshot(reopened)).toEqual(recorded as JsonObject));
    expect(logJson(recorded)).not.toContain(file);
  });

  test(`idle run does not keep storage owned after backend close: ${policy}`, async () => {
    const file = path.join(dir, "memory.sqlite");
    await withBackend(backend(policy, file), async (memory) => {
      const opened = memory.openRun();
      const recorded = opened.snapshot;
      memory.close();
      expect(() => opened.snapshot).toThrow(MemoryOperationError);
      await withBackend(backend(policy, file), (reopened) => expect(memorySnapshot(reopened)).toEqual(recorded));
      opened.close();
    });
  });
}

for (const policy of POLICIES) {
  test(`explicit learning remains saved after reopen: ${policy}`, async () => {
    const actions = [{ action: "add", lesson_id: "checked-output", text: "Verify the returned output." }];
    const reflector = () => new ScriptedModel([reply(JSON.stringify({ actions }))], MODEL_CONFIG);
    const file = path.join(dir, "memory.sqlite");
    const log = path.join(dir, "run.jsonl");
    const flow = await loadFlow(CODE_FLOW.replace("POLICY", policy));
    const field = policy === "episodic" ? "episodes" : "lessons";
    const recorded = await withBackend(backend(policy, file, reflector), async (memory) => {
      const options = { memory: new Map([[policy, memory]]), learnable: true, log, env: NO_ENV };
      const result = await runFlow(flow, {}, options);
      expect(result.status).toBe("succeeded");
      expect(result.output).toBe(7n);
      expect(memorySnapshot(memory)[field]).toEqual([]);
      await result.backward();
      expect(result.learning!.status).toBe("succeeded");
      const learned = memorySnapshot(memory);
      expect(learned[field]).not.toHaveLength(0);
      return learned;
    });
    const prefix = fs.readFileSync(log);
    await withBackend(backend(policy, file, reflector), async (reopened) => {
      expect(memorySnapshot(reopened)).toEqual(recorded);
      const inspected = await resumeFlow(flow, { log, memory: new Map([[policy, reopened]]), env: NO_ENV });
      expect(inspected.output).toBe(7n);
      expect(memorySnapshot(reopened)).toEqual(recorded);
    });
    expect(fs.readFileSync(log)).toEqual(prefix);
    expect(prefix.includes(file)).toBe(false);
  });
}
