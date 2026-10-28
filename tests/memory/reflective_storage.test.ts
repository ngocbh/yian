/**
 * Reflective memory's optional SQLite storage: lessons, limits and commit tombstones survive a
 * reopen, failed saves publish nothing, and an active callback blocks close. With `YIAN_TEST_NODE`
 * set, a store Bun wrote is reopened, learned into and committed under Node.
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { BUILTIN, type Env, install } from "../../src/defaults.ts";
import { EventStream } from "../../src/llm/events.ts";
import { Message, type Request, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  type JsonObject,
  LearnContext,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOperationError,
  RecallContext,
} from "../../src/memory/contracts.ts";
import { ReflectiveMemory, type ReflectiveSession } from "../../src/memory/reflective.ts";
import { SQLiteStorage } from "../../src/memory/storage/sqlite.ts";
import type { Sqlite } from "../../src/platform.ts";
import { logJson } from "../../src/typesys/json.ts";

// Cloned inventories and private fields are edited and read freely.
type Mutable = any;

const CONFIG = { provider: "ollama", model: "current" };

type Action = Record<string, unknown>;

const NOOP: Action[] = [{ action: "noop" }];

let dir: string;
/** The test's current learner; every backend's `createModel` delegates to it. */
let learner: ReturnType<typeof installModel>;

beforeEach(() => {
  install(BUILTIN);
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-reflective-storage-"));
  learner = installModel([]);
});

afterEach(() => {
  install(BUILTIN);
  fs.rmSync(dir, { recursive: true, force: true });
});

/** The test environment: an empty private store and an explicit model default. */
function baseEnv(overrides: Record<string, string> = {}): Env {
  const vars = new Map([
    ["YIAN_AUTH_FILE", path.join(dir, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
    ...Object.entries(overrides),
  ]);
  return (name) => vars.get(name);
}

/** A scripted reflector reply whose text is the JSON of `{actions}`. */
function reply(actions: readonly Action[]): Response {
  return new Response({
    message: new Message({ role: "assistant", parts: [new TextPart(JSON.stringify({ actions }))] }),
  });
}

/**
 * A `createModel` that records each config and returns a new
 * `ScriptedModel` (named by the config) answering with the next reply of the shared list.
 */
function installModel(replies: readonly Response[]) {
  const queue = [...replies];
  const configs: Record<string, unknown>[] = [];
  const createModel = (config: Readonly<Record<string, unknown>>): ScriptedModel => {
    configs.push(structuredClone(config) as Record<string, unknown>);
    const next = queue.shift();
    const model = new ScriptedModel(next === undefined ? [] : [next], {
      provider: String(config.provider),
      model: String(config.model),
    });
    return model;
  };
  return {
    createModel,
    configs,
  };
}

/** A backend over `CONFIG`, `baseEnv()` and the current learner. */
function newMemory(
  file: string | null = null,
  options: ConstructorParameters<typeof ReflectiveMemory>[0] = {},
): ReflectiveMemory {
  return new ReflectiveMemory({
    llmConfig: CONFIG,
    storage: file,
    env: baseEnv(),
    createModel: (config) => learner.createModel(config),
    ...options,
  });
}

/** Return the `MemoryOperationError` `fn` throws. */
function failure(fn: () => unknown): MemoryOperationError {
  try {
    fn();
  } catch (error) {
    expect(error).toBeInstanceOf(MemoryOperationError);
    return error as MemoryOperationError;
  }
  throw new Error("expected a MemoryOperationError");
}

/** Return the `MemoryOperationError` `fn` throws or rejects with. */
async function rejection(fn: () => unknown): Promise<MemoryOperationError> {
  try {
    await fn();
  } catch (error) {
    expect(error).toBeInstanceOf(MemoryOperationError);
    return error as MemoryOperationError;
  }
  throw new Error("expected a MemoryOperationError");
}

/** Return whatever `fn` throws. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

/** Open lazy storage without starting a run. */
function firstOpen(memory: ReflectiveMemory): ReflectiveMemory {
  (memory as Mutable).operation(() => {});
  return memory;
}

/** Open `memory`'s storage, await `fn` with it and close it. */
async function withMemory<T>(
  memory: ReflectiveMemory,
  fn: (memory: ReflectiveMemory) => T | Promise<T>,
): Promise<T> {
  firstOpen(memory);
  try {
    return await fn(memory);
  } finally {
    memory.close();
  }
}

/** Open a run, await `fn` with it, and close it. */
async function withRun<T>(
  memory: ReflectiveMemory,
  fn: (session: ReflectiveSession) => T | Promise<T>,
  snapshot?: JsonObject,
): Promise<T> {
  const session = memory.openRun({ snapshot });
  try {
    return await fn(session);
  } finally {
    session.close();
  }
}

/** The pinned snapshot of a fresh run of `memory`. */
function snapshotOf(memory: ReflectiveMemory): JsonObject {
  const session = memory.openRun();
  try {
    return session.snapshot;
  } finally {
    session.close();
  }
}

/** One learning context whose single succeeded episode echoes `hello`. */
function context(runId = "run-1"): LearnContext {
  const episode = new MemoryEpisode(runId, [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, { text: "hello" }, {
    output: { answer: "HELLO" },
  });
  return new LearnContext(runId, "reflective", [episode]);
}

/** A hand-built reflective mutation attributed to `run-1`. */
function mutation(mutationId: string, action: string, lessonId: string, text?: string): MemoryMutation {
  const payload: Record<string, unknown> = { action, lesson_id: lessonId, source: { run_id: "run-1" } };
  if (text !== undefined) payload.text = text;
  return new MemoryMutation(mutationId, payload);
}

/** Learn one `action` on `lesson` and commit it under `commitId` in its own run; return the batch. */
async function learn(
  memory: ReflectiveMemory,
  commitId = "learned",
  action = "add",
  text = "Use evidence",
): Promise<MemoryMutation[]> {
  const proposed: Action = { action, lesson_id: "lesson" };
  if (action !== "remove") proposed.text = text;
  learner = installModel([reply([proposed])]);
  return withRun(memory, async (opened) => {
    const batch = await opened.learn(context(commitId));
    opened.commit(commitId, batch);
    return batch;
  });
}

/** Open the store at `file` directly, load its inventory and close it. */
function loadInventory(file: string): Mutable {
  const storage = new SQLiteStorage(file, "reflective");
  try {
    return storage.load();
  } finally {
    storage.close();
  }
}

/** Save `inventory` to the store at `file` directly. */
function saveInventory(file: string, inventory: unknown): void {
  const storage = new SQLiteStorage(file, "reflective");
  try {
    storage.save(inventory);
  } finally {
    storage.close();
  }
}

/** The private storage of `memory`. */
function storageOf(memory: ReflectiveMemory): SQLiteStorage {
  return (memory as Mutable).storage;
}

/** A copy of the publication fields of `memory`, in a fixed order. */
function ramState(memory: ReflectiveMemory): unknown[] {
  const m = memory as Mutable;
  return structuredClone([
    m.namespace,
    m.epoch,
    m.revision,
    m.state,
    m.appliedCommits,
    m.everOpened,
    m.everReset,
    m.everCommitted,
    m.llmConfig,
  ]);
}

test("learned lessons and limits survive reopen", async () => {
  const file = path.join(dir, "memory.sqlite");
  let old!: JsonObject;
  let expected!: JsonObject;
  await withMemory(newMemory(file, { maxItems: 2, recallLimit: 1 }), async (memory) => {
    old = snapshotOf(memory);
    await learn(memory);
    expected = snapshotOf(memory);
  });
  await withMemory(newMemory(file), async (memory) => {
    expect(snapshotOf(memory)).toEqual(expected);
    await withRun(memory, async (opened) => {
      expect(await opened.recall()).toBe("[lesson] Use evidence");
      expect(opened.snapshot.config).toEqual({ max_items: 2n, recall_limit: 1n });
    });
    await withRun(
      memory,
      async (stale) => {
        expect(stale.snapshot).toEqual(old);
        expect(await stale.recall()).toBe("");
        expect((await rejection(() => stale.learn(context()))).code).toBe("reflective.stale_generation");
      },
      old,
    );
  });
});

test("saved snapshots pin model but new sessions use host config", async () => {
  const file = path.join(dir, "memory.sqlite");
  const saved = await withMemory(newMemory(file, { llmConfig: { provider: "ollama", model: "old" } }), snapshotOf);
  await withMemory(newMemory(file), async (memory) => {
    learner = installModel([reply(NOOP), reply(NOOP)]);
    await withRun(
      memory,
      async (resumed) => {
        expect(resumed.snapshot).toEqual(saved);
        await resumed.learn(context());
      },
      saved,
    );
    await withRun(memory, (fresh) => fresh.learn(context()));
    expect(learner.configs.map((config) => config.model)).toEqual(["old", "current"]);
  });
});

test("pristine snapshot adoption keeps constructor model", async () => {
  const old = newMemory(null, { llmConfig: { provider: "ollama", model: "old" } });
  const saved = await withMemory(old, async (source) => {
    await learn(source);
    return snapshotOf(source);
  });
  const file = path.join(dir, "memory.sqlite");
  await withMemory(newMemory(file), async (memory) => {
    await withRun(memory, (resumed) => expect(resumed.snapshot).toEqual(saved), saved);
    expect((snapshotOf(memory).llm_config as JsonObject).model).toBe("current");
  });
  await withMemory(newMemory(file), (memory) => {
    const current = snapshotOf(memory);
    expect(current.lessons).toEqual(saved.lessons);
    expect((current.llm_config as JsonObject).model).toBe("current");
  });
});

test("first open persists identity without resolving model", async () => {
  const forbidden: Env = () => {
    throw new Error("ambient model must not be resolved at construction");
  };
  const file = path.join(dir, "memory.sqlite");
  const memory = firstOpen(new ReflectiveMemory({ storage: file, env: forbidden }));
  const namespace = (memory as Mutable).namespace;
  expect(failure(() => firstOpen(new ReflectiveMemory({ storage: file, env: forbidden }))).code).toBe(
    "storage.busy",
  );
  memory.close();
  memory.close();
  await withMemory(new ReflectiveMemory({ storage: file, env: forbidden }), (reopened) => {
    expect((reopened as Mutable).namespace).toBe(namespace);
    expect((reopened as Mutable).everOpened).toBe(false);
  });
  const encoded = logJson(loadInventory(file));
  for (const key of ["llm_config", "provider", "model", "api_key", "temperature"]) {
    expect(encoded).not.toContain(key);
  }
});

test("ambient config is current after reopen", async () => {
  const file = path.join(dir, "memory.sqlite");
  const ambient = (model: string) =>
    new ReflectiveMemory({
      storage: file,
      env: baseEnv({ YIAN_DEFAULT_PROVIDER: "ollama", YIAN_DEFAULT_MODEL: model }),
    });
  const saved = await withMemory(ambient("old"), snapshotOf);
  await withMemory(ambient("new"), async (memory) => {
    await withRun(memory, (resumed) => expect(resumed.snapshot).toEqual(saved), saved);
    expect((snapshotOf(memory).llm_config as JsonObject).model).toBe("new");
  });
});

test("none storage never opens sqlite", async () => {
  const spies = (["load", "save"] as const).map((name) =>
    spyOn(SQLiteStorage.prototype, name).mockImplementation(() => {
      throw new Error("SQLite must remain dormant");
    }),
  );
  try {
    await withMemory(newMemory(), (memory) => {
      expect(snapshotOf(memory).lessons).toEqual([]);
    });
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  } finally {
    for (const spy of spies) spy.mockRestore();
  }
});

test("invalid defaults do not create storage", () => {
  const file = path.join(dir, "memory.sqlite");
  expect(() => newMemory(file, { maxItems: 0 })).toThrow(new RangeError("max_items must be positive"));
  expect(fs.existsSync(file)).toBe(false);
});

test("whole commit is validated before saving", async () => {
  const file = path.join(dir, "memory.sqlite");
  let before!: JsonObject;
  await withMemory(newMemory(file), async (memory) => {
    await learn(memory);
    before = snapshotOf(memory);
    await withRun(memory, (opened) => {
      const batch = [mutation("good", "revise", "lesson", "Changed"), mutation("bad", "remove", "missing")];
      const caught = failure(() => opened.commit("invalid", batch));
      expect(caught.code).toBe("reflective.invalid_mutation");
    });
    expect(snapshotOf(memory)).toEqual(before);
    expect((memory as Mutable).appliedCommits.has("invalid")).toBe(false);
  });
  await withMemory(newMemory(file), (memory) => {
    expect(snapshotOf(memory)).toEqual(before);
  });
});

test.each([false, true])("idle handles do not block close and are invalidated (durable %p)", async (durable) => {
  const memory = newMemory(durable ? path.join(dir, "memory.sqlite") : null);
  const opened = memory.openRun();
  memory.close();
  for (const operation of [
    () => memory.openRun(),
    () => memory.reset(),
    () => opened.snapshot,
    () => opened.commit("c", []),
    () => opened.recall(),
    () => opened.learn(context()),
    () => memory.rewriteQuery(new RecallContext("", "query", {}, ["agent"])),
  ]) {
    expect((await rejection(operation)).code).toBe("reflective.closed");
  }
  opened.close();
  opened.close();
});

test.each(["opened", "reset", "committed"])("nonpristine flags survive reopen (%s)", async (prior) => {
  const file = path.join(dir, "memory.sqlite");
  const foreign = await withMemory(newMemory(), snapshotOf);
  await withMemory(newMemory(file), async (memory) => {
    if (prior === "opened") snapshotOf(memory);
    else if (prior === "reset") memory.reset();
    else await withRun(memory, (opened) => opened.commit("empty", []));
  });
  await withMemory(newMemory(file), (memory) => {
    expect(failure(() => memory.openRun({ snapshot: foreign })).code).toBe("reflective.invalid_snapshot");
    const flag = { opened: "everOpened", reset: "everReset", committed: "everCommitted" }[prior]!;
    expect((memory as Mutable)[flag]).toBe(true);
  });
});

test("commit tombstones survive removal reset and stale replay", async () => {
  const file = path.join(dir, "memory.sqlite");
  let old!: JsonObject;
  let batch!: MemoryMutation[];
  let expected!: JsonObject;
  await withMemory(newMemory(file), async (memory) => {
    old = snapshotOf(memory);
    batch = await learn(memory);
    await learn(memory, "removed", "remove");
    expected = snapshotOf(memory);
    expect(expected.lessons).toEqual([]);
  });
  await withMemory(newMemory(file), async (memory) => {
    await withRun(
      memory,
      (stale) => {
        stale.commit("learned", batch);
        expect(failure(() => stale.commit("learned", [])).code).toBe("reflective.commit_collision");
      },
      old,
    );
    expect(snapshotOf(memory)).toEqual(expected);
    memory.reset();
  });
  await withMemory(newMemory(file), async (memory) => {
    await withRun(memory, (opened) => {
      opened.commit("learned", batch);
      opened.commit("learned", batch);
      expect(failure(() => opened.commit("learned", [])).code).toBe("reflective.commit_collision");
    });
    expect(snapshotOf(memory).lessons).toEqual([]);
    expect((memory as Mutable).epoch).toBe(1);
    expect((memory as Mutable).revision).toBe(0);
  });
});

test("recorded limits survive reopen", async () => {
  const file = path.join(dir, "memory.sqlite");
  const saved = await withMemory(newMemory(file, { maxItems: 4, recallLimit: 2 }), (memory) =>
    withRun(memory, (opened) => opened.snapshot),
  );
  await withMemory(newMemory(file), (memory) =>
    withRun(
      memory,
      (resumed) => {
        expect(resumed.snapshot).toEqual(saved);
        resumed.commit("declare", []);
      },
      saved,
    ),
  );
  await withMemory(newMemory(file), (memory) => {
    expect(snapshotOf(memory).config).toEqual({ max_items: 4n, recall_limit: 2n });
  });
});

test("explicit limits must match stored limits", async () => {
  const file = path.join(dir, "memory.sqlite");
  firstOpen(newMemory(file, { maxItems: 4, recallLimit: 2 })).close();
  await withMemory(newMemory(file), (memory) => {
    expect((memory as Mutable).state.limits).toEqual({ maxItems: 4, recallLimit: 2 });
  });
  await withMemory(newMemory(file, { recallLimit: 2 }), (memory) => {
    expect((memory as Mutable).state.limits.maxItems).toBe(4);
  });
  expect(failure(() => firstOpen(newMemory(file, { maxItems: 5 }))).code).toBe("reflective.config_conflict");
  firstOpen(newMemory(file)).close();
});

test("a fresh store pins its default limits", () => {
  const file = path.join(dir, "memory.sqlite");
  firstOpen(newMemory(file)).close();
  expect(failure(() => firstOpen(newMemory(file, { maxItems: 5 }))).code).toBe("reflective.config_conflict");
});

test.each(["open", "adopt", "reset", "commit"])("failed save never publishes candidate RAM (%s)", async (operation) => {
  const memory = firstOpen(newMemory(path.join(dir, "memory.sqlite")));
  const session = operation === "commit" ? memory.openRun() : null;
  const foreign = operation === "adopt" ? await withMemory(newMemory(), snapshotOf) : null;
  const before = ramState(memory);
  const primary = new Error("save failed");
  const save = spyOn(storageOf(memory), "save").mockImplementation(() => {
    throw primary;
  });
  try {
    const caught = thrown(() => {
      if (operation === "open") memory.openRun();
      else if (operation === "adopt") memory.openRun({ snapshot: foreign });
      else if (operation === "reset") memory.reset();
      else session!.commit("commit", [mutation("m", "add", "l", "text")]);
    });
    expect(caught).toBe(primary);
    expect(ramState(memory)).toEqual(before);
  } finally {
    save.mockRestore();
    memory.close();
  }
});

/** Return a SQLite driver-shaped error whose message is a private path. */
function sqliteError(): Error {
  return Object.assign(new Error("private path"), { code: "ERR_SQLITE_ERROR", errcode: 1 });
}

test.each([
  [false, false],
  [false, true],
  [true, false],
  [true, true],
])(
  "uncertain save poisoning blocks RAM reads and keeps primary (after commit %p, cancel %p)",
  async (afterCommit, cancel) => {
    const primary: unknown = cancel ? { cancel: "stopped" } : sqliteError();
    const file = path.join(dir, "memory.sqlite");
    const memory = newMemory(file);
    const opened = memory.openRun();
    const before = ramState(memory);
    const connection: Sqlite = (storageOf(memory) as Mutable).connection;
    const realExec = connection.exec;
    const exec = spyOn(connection, "exec").mockImplementation((sql) => {
      if (sql !== "COMMIT") return Reflect.apply(realExec, connection, [sql]);
      if (afterCommit) Reflect.apply(realExec, connection, [sql]);
      throw primary;
    });
    try {
      const caught = thrown(() => opened.commit("uncertain", [mutation("m", "add", "l", "text")]));
      if (cancel) {
        expect(caught).toBe(primary);
      } else {
        expect(caught).toBeInstanceOf(MemoryOperationError);
        expect((caught as MemoryOperationError).code).toBe("storage.io");
        expect(String(caught)).not.toContain("private path");
      }
      expect(ramState(memory)).toEqual(before);
      for (const operation of [
        () => memory.openRun(),
        () => memory.reset(),
        () => opened.snapshot,
        () => opened.recall(),
        () => opened.learn(context()),
        () => opened.commit("c", []),
      ]) {
        expect((await rejection(operation)).code).toBe("storage.poisoned");
      }
      expect(failure(() => firstOpen(newMemory(file))).code).toBe("storage.busy");
    } finally {
      exec.mockRestore();
      opened.close();
      memory.close();
    }
    await withMemory(newMemory(file), (reopened) => {
      expect(snapshotOf(reopened).revision).toBe(afterCommit ? 1n : 0n);
    });
  },
);

test.each([
  "extra",
  "version",
  "counter",
  "flag",
  "digest",
  "unopened",
  "uncommitted",
  "reset",
  "config",
  "stale_frames",
  "source",
  "duplicate_lesson",
  "overflow",
  "host_config",
])("invalid inventory releases ownership without rewriting (%s)", async (kind) => {
  const file = path.join(dir, "memory.sqlite");
  await withMemory(newMemory(file, { maxItems: 1 }), (memory) => learn(memory));
  const inventory = loadInventory(file);
  const state = inventory.state;
  if (kind === "extra") inventory.extra = null;
  else if (kind === "version") inventory.version = true;
  else if (kind === "counter") state.revision = -1n;
  else if (kind === "flag") inventory.ever_opened = 1n;
  else if (kind === "digest") inventory.applied_commits = { c: "not-a-digest" };
  else if (kind === "unopened") inventory.ever_opened = false;
  else if (kind === "uncommitted") inventory.ever_committed = false;
  else if (kind === "reset") inventory.ever_reset = true;
  else if (kind === "config") state.config.max_items = true;
  else if (kind === "stale_frames") state.frames = [];
  else if (kind === "source") state.lessons[0].source.frame_ref = {};
  else if (kind === "duplicate_lesson") {
    state.config.max_items = 2n;
    state.lessons.push(structuredClone(state.lessons[0]));
  } else if (kind === "overflow") state.lessons.push({ ...structuredClone(state.lessons[0]), id: "extra" });
  else state.llm_config = CONFIG;
  saveInventory(file, inventory);
  const caught = failure(() => firstOpen(newMemory(file)));
  expect([caught.stage, caught.code, caught.message]).toEqual([
    "storage",
    "reflective.invalid_storage",
    "stored reflective state is invalid",
  ]);
  expect(loadInventory(file)).toEqual(inventory);
});

/** The package root. */
const ROOT = path.resolve(import.meta.dirname, "../..");

/**
 * A child-process script that imports the backend through `load` (a module name such as
 * `memory/reflective` to a quoted specifier) and runs `body` with `assert`, `ReflectiveMemory`, an
 * `env` holding only a temp `YIAN_AUTH_FILE`, `learner(actions)` (a `createModel` whose model
 * answers `{actions}` once) and `context(runId)`.
 */
function childScript(load: (name: string) => string, body: string): string {
  return `
const assert = (await import("node:assert/strict")).default;
const { ReflectiveMemory } = await import(${load("memory/reflective")});
const c = await import(${load("memory/contracts")});
const { Message, Response, TextPart } = await import(${load("llm/messages")});
const { ScriptedModel } = await import(${load("llm/testing")});
const vars = new Map([["YIAN_AUTH_FILE", ${JSON.stringify(path.join(dir, "auth.json"))}]]);
const env = (name) => vars.get(name);
const learner = (actions) => (config) =>
  new ScriptedModel(
    [new Response({ message: new Message({ role: "assistant", parts: [new TextPart(JSON.stringify({ actions }))] }) })],
    { provider: String(config.provider), model: String(config.model) },
  );
const context = (runId) =>
  new c.LearnContext(runId, "reflective", [
    new c.MemoryEpisode(runId, [], ["agent"], c.MemoryEpisodeStatus.SUCCEEDED, { text: "hello" }, {
      output: { answer: "HELLO" },
    }),
  ]);
${body}
console.log("ok");
`;
}

/** This process's environment without `YIAN_*` keys, so a child reads no ambient yian settings. */
function childEnv(): NodeJS.ProcessEnv {
  return Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("YIAN_")));
}

test("actual process restart restores learned lessons", async () => {
  const file = path.join(dir, "memory.sqlite");
  const script = childScript(
    (name) => JSON.stringify(path.join(ROOT, "src", `${name}.ts`)),
    `
const memory = new ReflectiveMemory({
  llmConfig: ${JSON.stringify(CONFIG)},
  storage: ${JSON.stringify(file)},
  env,
  createModel: learner([{ action: "add", lesson_id: "lesson", text: "Use evidence" }]),
});
try {
  const session = memory.openRun();
  session.commit("learned", await session.learn(context("learned")));
  session.close();
} finally {
  memory.close();
}
`,
  );
  const result = spawnSync(process.execPath, ["-e", script], {
    encoding: "utf8",
    timeout: 20_000,
    env: childEnv(),
  });
  expect(result.stderr).toBe("");
  expect(result.stdout).toBe("ok\n");
  expect(result.status).toBe(0);
  await withMemory(newMemory(file), (memory) =>
    withRun(memory, async (opened) => {
      expect(await opened.recall()).toBe("[lesson] Use evidence");
      expect(opened.snapshot.revision).toBe(1n);
    }),
  );
}, 30_000);

test.each([
  ["construction", false],
  ["construction", true],
  ["invoke", false],
  ["invoke", true],
  ["query", false],
  ["query", true],
])(
  "active callback blocks close without holding backend lock (%s, reentrant %p)",
  async (phase, reentrant) => {
    const file = path.join(dir, "memory.sqlite");
    let entered!: () => void;
    const enteredPromise = new Promise<void>((resolve) => {
      entered = resolve;
    });
    let release!: () => void;
    const released = new Promise<void>((resolve) => {
      release = resolve;
    });
    let memory!: ReflectiveMemory;
    let opened!: ReflectiveSession;
    let closeError: unknown = null;
    const callback = async () => {
      if (reentrant) {
        opened.close();
        try {
          memory.close();
        } catch (error) {
          closeError = error;
        }
      } else {
        entered();
        await released;
      }
    };
    /** A scripted model whose stream awaits the callback in the `invoke` phase. */
    class Model extends ScriptedModel {
      override stream(request: Request, options?: { signal?: AbortSignal }): EventStream {
        const events = super.stream(request, options);
        if (phase !== "invoke") return events;
        return new EventStream(
          (async function* () {
            await callback();
            yield* events;
          })(),
        );
      }
    }
    memory = newMemory(file, {
      createModel: async () => {
        if (phase === "construction") await callback();
        return new Model([reply(NOOP)], CONFIG);
      },
    });
    memory.rewriteQuery = async () => {
      await callback();
      return "query";
    };
    opened = memory.openRun();
    const work = (
      phase === "query" ? opened.recall(new RecallContext("", "query", {}, ["agent"])) : opened.learn(context())
    ).then(
      () => null,
      (error: unknown) => error,
    );
    if (!reentrant) {
      // An operation that settles before its callback waits reports its error, not a timeout.
      const early = await Promise.race([enteredPromise.then(() => null), work.then((error) => ({ error }))]);
      if (early) throw new Error("the operation settled before its callback waited", { cause: early.error });
      try {
        // This must finish while the model callback is still waiting.
        opened.close();
        expect(failure(() => memory.close()).code).toBe("reflective.busy");
        expect(failure(() => firstOpen(newMemory(file))).code).toBe("storage.busy");
      } finally {
        release();
      }
    }
    const error = await work;
    if (reentrant) {
      expect(closeError).toBeInstanceOf(MemoryOperationError);
      expect((closeError as MemoryOperationError).code).toBe("reflective.busy");
    }
    if (phase === "query") {
      expect(error).toBeNull();
    } else {
      expect(error).toBeInstanceOf(MemoryOperationError);
      expect((error as MemoryOperationError).code).toBe("reflective.session_closed");
    }
    opened.close();
    memory.close();
    await withMemory(newMemory(file), () => {});
  },
  10_000,
);

test.each([false, true])("structured result inspection preserves error translation (raises %p)", async (raises) => {
  const primary = new Error("private inspection failure");
  /** A model whose `stream` getter throws, or gives no function. */
  class Model {
    readonly provider = CONFIG.provider;
    readonly model = CONFIG.model;
    get stream(): undefined {
      if (raises) throw primary;
      return undefined;
    }
  }
  const memory = newMemory(path.join(dir, "memory.sqlite"), { createModel: () => new Model() as never });
  const caught = await withMemory(memory, (memory) =>
    withRun(memory, (opened) => rejection(() => opened.learn(context()))),
  );
  expect(caught.code).toBe("reflective.structured_output");
  expect(caught.retryable).toBe(false);
  if (raises) {
    expect(caught.cause).toBe(primary);
    expect(String(caught)).not.toContain("private inspection failure");
  } else {
    expect((caught.cause as object).constructor).toBe(TypeError);
    expect((caught.cause as Error).message).toBe("the reflector model must provide stream()");
  }
});

test("first open cleanup preserves cancellation and releases owner", async () => {
  const file = path.join(dir, "memory.sqlite");
  firstOpen(newMemory(file)).close();
  const primary = { cancel: "initialization stopped" };
  const close = SQLiteStorage.prototype.close;
  const load = spyOn(SQLiteStorage.prototype, "load").mockImplementation(() => {
    throw primary;
  });
  const cleanup = spyOn(SQLiteStorage.prototype, "close").mockImplementation(function (this: SQLiteStorage) {
    Reflect.apply(close, this, []);
    throw new Error("cleanup problem");
  });
  try {
    expect(thrown(() => firstOpen(newMemory(file)))).toBe(primary);
  } finally {
    load.mockRestore();
    cleanup.mockRestore();
  }
  await withMemory(newMemory(file), () => {});
});

const BUILT = path.join(ROOT, "dist/src/memory/reflective.js");

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/memory/memory_storage.test.ts`'s: importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!fs.existsSync(BUILT)) throw new Error(`${BUILT} is missing; run bun run build first`);
  return node;
})();

describe.skipIf(!NODE)("the built backend on Node", () => {
  test(
    "Node reopens a store Bun wrote, learns, commits, and Bun reads it back",
    async () => {
      const file = path.join(dir, "memory.sqlite");
      const batch = await withMemory(newMemory(file, { maxItems: 2, recallLimit: 1 }), (memory) => learn(memory));
      const script = childScript(
        (name) => JSON.stringify(pathToFileURL(path.join(ROOT, `dist/src/${name}.js`)).href),
        `
const memory = new ReflectiveMemory({
  llmConfig: ${JSON.stringify(CONFIG)},
  storage: ${JSON.stringify(file)},
  env,
  createModel: learner([{ action: "add", lesson_id: "node", text: "Cite sources" }]),
});
try {
  const session = memory.openRun();
  assert.deepEqual(session.snapshot.config, { max_items: 2n, recall_limit: 1n });
  assert.equal(session.snapshot.revision, 1n);
  assert.equal(await session.recall(), "[lesson] Use evidence");
  session.commit("node", await session.learn(context("node")));
  session.close();
} finally {
  memory.close();
}
`,
      );
      const result = spawnSync(NODE!, ["--input-type=module", "-e", script], {
        encoding: "utf8",
        timeout: 20_000,
        env: childEnv(),
      });
      expect(result.stderr).toBe("");
      expect(result.stdout).toBe("ok\n");
      expect(result.status).toBe(0);
      await withMemory(newMemory(file), (memory) =>
        withRun(memory, async (opened) => {
          expect(opened.snapshot.revision).toBe(2n);
          expect(opened.snapshot.config).toEqual({ max_items: 2n, recall_limit: 1n });
          expect(await opened.recall()).toBe("[lesson] Use evidence\n[node] Cite sources");
          opened.commit("learned", batch);
          expect(failure(() => opened.commit("node", [])).code).toBe("reflective.commit_collision");
        }),
      );
    },
    30_000,
  );
});
