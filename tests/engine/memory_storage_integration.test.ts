/** Durable backends retain memory through the public run and recovery boundary. */

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
import { advance, runFlow } from "../../src/run.ts";
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

/** The `policy` backend over `storage`; a reflective one learns with `createModel`. */
function backend(policy: Policy, storage: string | null, createModel: (() => ScriptedModel) | null = null): Backend {
  if (policy === "reflective") return new ReflectiveMemory({ llmConfig: MODEL_CONFIG, storage, createModel });
  return new EpisodicMemory({ storage });
}

/** Await `fn` with `memory`, then close it. */
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
      const inspected = await advance({ log, memory: new Map([[policy, reopened]]), env: NO_ENV });
      expect(inspected.output).toBe(7n);
      expect(memorySnapshot(reopened)).toEqual(recorded);
    });
    expect(fs.readFileSync(log)).toEqual(prefix);
    expect(prefix.includes(file)).toBe(false);
  });
}
