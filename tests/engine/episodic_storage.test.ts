/**
 * Optional episodic persistence preserves state and backend ownership: a port of
 * `tests/engine/test_episodic_storage.py`.
 *
 * Ported: `test_first_open_initializes_identity_before_any_run_and_releases_owner`,
 * `test_none_storage_remains_resource_free`, `test_idle_handles_do_not_block_close_and_are_invalidated`,
 * `test_invalid_inventory_is_rejected_without_rewriting_or_retaining_owner`,
 * `test_history_limits_and_pinned_snapshots_survive_restart`, `test_nonpristine_flags_survive_restart`,
 * `test_pristine_adoption_persists_across_restart`,
 * `test_commit_tombstones_survive_eviction_reset_restart_and_stale_replay`,
 * `test_failed_save_never_publishes_candidate_ram`,
 * `test_uncertain_save_poisoning_blocks_ram_reads_and_keeps_primary`,
 * `test_first_open_cleanup_preserves_cancellation_and_releases_ownership`,
 * `test_active_callback_makes_close_busy_without_releasing_file`, `test_invalid_limits_do_not_create_storage`,
 * `test_recorded_limits_survive_a_reopen`, `test_explicit_limits_must_match_stored_limits`,
 * `test_a_fresh_store_pins_its_default_limits`, `test_corrupt_private_state_never_partially_adopts_ram`
 * and `test_invalid_last_mutation_never_reaches_storage`.
 *
 * TS-only: an `epoch` row (`snapshot.epoch = 2n ** 60n`, beyond 2^53 − 1) of
 * `test_invalid_inventory_is_rejected_without_rewriting_or_retaining_owner` is
 * `episodic.invalid_storage`; a failed first open whose primary is an object (an `Error` or a plain object) gets the
 * note `EpisodicMemory cleanup failed: Cleanup` when closing the storage also fails. With
 * `YIAN_TEST_NODE` set to a Node binary, the built backend reopens a store Bun wrote (episodes,
 * limits, tombstones) under Node and commits, and Bun reads the result back.
 *
 * Changed:
 * - Python's `with EpisodicMemory(...)` is `withMemory`, which opens the storage first (Python's
 *   `__enter__` runs an operation) and closes the backend in `finally`; `_first_open` runs the
 *   private `operation` guard with an empty body.
 * - `test_none_storage_remains_resource_free`: spies on `SQLiteStorage.prototype.load` and `save`
 *   that throw stand in for Python forbidding `sqlite3.connect`.
 * - `test_uncertain_save_poisoning_blocks_ram_reads_and_keeps_primary`: the failure is injected by
 *   a spy on the storage's private connection's `exec` for `COMMIT`, where Python subclasses
 *   `sqlite3.Connection`; the cancel rows throw a non-`Error` value (Python's `KeyboardInterrupt`),
 *   and the other rows a SQLite driver-shaped error.
 * - `test_first_open_cleanup_preserves_cancellation_and_releases_ownership`: the primary is a thrown
 *   non-`Error` value from a spy on `SQLiteStorage.prototype.load` (the same failure path), where
 *   Python patches the module's `_parse_inventory`.
 * - `test_active_callback_makes_close_busy_without_releasing_file`: a plain async test whose
 *   instance `rewriteQuery` awaits a deferred promise, where Python runs a thread in a spawned
 *   process.
 * - JSON ints written into inventories are `bigint`s; mutable copies are `structuredClone`s;
 *   snapshots are frozen; `recall` is async.
 *
 * Not ported: `test_inherited_handle_rejects_before_an_inherited_backend_lock`,
 * `test_run_context_fork_cleanup_preserves_primary` and
 * `test_query_callback_fork_rejects_before_return_and_preserves_primary` (JS can't fork a live
 * process, so there is no fork guard).
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import {
  type JsonObject,
  LearnContext,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOperationError,
  RecallContext,
} from "../../src/memory/contracts.ts";
import { EpisodicMemory, type EpisodicSession } from "../../src/memory/episodic.ts";
import { SQLiteStorage } from "../../src/memory/storage/sqlite.ts";
import type { Sqlite } from "../../src/platform.ts";
import { logJson } from "../../src/typesys/json.ts";

// Cloned inventories are edited freely.
type Mutable = any;

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-episodic-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

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

/** Open lazy storage without starting a run (Python's `_first_open`). */
function firstOpen(memory: EpisodicMemory): EpisodicMemory {
  (memory as Mutable).operation(() => {});
  return memory;
}

/** Open `memory`'s storage, await `fn` with it and close it (Python's `with EpisodicMemory(...)`). */
async function withMemory<T>(memory: EpisodicMemory, fn: (memory: EpisodicMemory) => T | Promise<T>): Promise<T> {
  firstOpen(memory);
  try {
    return await fn(memory);
  } finally {
    memory.close();
  }
}

/** Open a run, await `fn` with it and close it (Python's `with memory.open_run() as session`). */
async function withRun<T>(
  memory: EpisodicMemory,
  fn: (session: EpisodicSession) => T | Promise<T>,
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
function snapshotOf(memory: EpisodicMemory): JsonObject {
  const session = memory.openRun();
  try {
    return session.snapshot;
  } finally {
    session.close();
  }
}

/** A one-episode learning context whose output is `runId`. */
function context(runId = "run", memory = "episodic"): LearnContext {
  const episode = new MemoryEpisode(runId, [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, {}, { output: runId });
  return new LearnContext(runId, memory, [episode]);
}

/** Learn and commit one episode under `commitId` in its own run; return the batch. */
function append(memory: EpisodicMemory, commitId: string): MemoryMutation[] {
  const session = memory.openRun();
  try {
    const batch = session.learn(context(commitId));
    session.commit(commitId, batch);
    return batch;
  } finally {
    session.close();
  }
}

/** Open the store at `file` directly, load its inventory and close it. */
function loadInventory(file: string): Mutable {
  const storage = new SQLiteStorage(file, "episodic");
  try {
    return storage.load();
  } finally {
    storage.close();
  }
}

/** Save `inventory` to the store at `file` directly. */
function saveInventory(file: string, inventory: unknown): void {
  const storage = new SQLiteStorage(file, "episodic");
  try {
    storage.save(inventory);
  } finally {
    storage.close();
  }
}

/** The private storage of `memory`. */
function storageOf(memory: EpisodicMemory): SQLiteStorage {
  return (memory as Mutable).storage;
}

/** The publication fields of `memory`, in Python's tuple order. */
function published(memory: EpisodicMemory): unknown[] {
  const m = memory as Mutable;
  return [m.namespace, m.epoch, m.revision, m.state, m.appliedCommits, m.everOpened, m.everReset, m.everCommitted];
}

test("first open initializes identity before any run and releases owner", async () => {
  const file = path.join(dir, "memory.sqlite");
  const memory = firstOpen(new EpisodicMemory({ maxEpisodes: 7, recallLimit: 3, storage: file }));
  const namespace = (memory as Mutable).namespace;
  expect(failure(() => firstOpen(new EpisodicMemory({ storage: file }))).code).toBe("storage.busy");
  memory.close();
  memory.close();
  await withMemory(new EpisodicMemory({ storage: file }), (reopened) => {
    expect((reopened as Mutable).namespace).toBe(namespace);
    expect((reopened as Mutable).everOpened).toBe(false);
    expect(snapshotOf(reopened).version).toBe(2n);
  });
});

test("none storage remains resource free", async () => {
  const spies = (["load", "save"] as const).map((name) =>
    spyOn(SQLiteStorage.prototype, name).mockImplementation(() => {
      throw new Error("SQLite must stay dormant");
    }),
  );
  try {
    await withMemory(new EpisodicMemory({ maxEpisodes: 4, recallLimit: 2, storage: null }), (memory) => {
      expect(snapshotOf(memory).episodes).toEqual([]);
    });
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  } finally {
    for (const spy of spies) spy.mockRestore();
  }
});

test.each([null, "file"])("idle handles do not block close and are invalidated (%p)", async (storage) => {
  const memory = new EpisodicMemory({ storage: storage && path.join(dir, "memory.sqlite") });
  const opened = memory.openRun();
  memory.close();
  for (const operation of [
    () => memory.openRun(),
    () => memory.reset(),
    () => opened.snapshot,
    () => opened.commit("c", []),
    () => opened.recall(),
    () => opened.learn(context()),
  ]) {
    expect((await rejection(operation)).code).toBe("episodic.closed");
  }
  opened.close();
  opened.close();
});

test.each(["extra", "version", "counter", "flag", "ledger", "digest", "unopened", "uncommitted", "reset", "epoch"])(
  "invalid inventory is rejected without rewriting or retaining owner (%s)",
  async (kind) => {
    const file = path.join(dir, "memory.sqlite");
    await withMemory(new EpisodicMemory({ storage: file }), (memory) => snapshotOf(memory));
    const inventory = loadInventory(file);
    if (kind === "extra") inventory.extra = null;
    else if (kind === "version") inventory.version = true;
    else if (kind === "counter") inventory.snapshot.revision = -1n;
    else if (kind === "flag") inventory.ever_opened = 1n;
    else if (kind === "ledger") inventory.applied_commits = [];
    else if (kind === "digest") inventory.applied_commits = { c: "not-a-digest" };
    else if (kind === "unopened") {
      inventory.ever_opened = false;
      inventory.ever_committed = true;
    } else if (kind === "uncommitted") inventory.snapshot.revision = 1n;
    else if (kind === "reset") inventory.ever_reset = true;
    else inventory.snapshot.epoch = 2n ** 60n;
    saveInventory(file, inventory);
    const caught = failure(() => firstOpen(new EpisodicMemory({ storage: file })));
    expect(caught.code).toBe("episodic.invalid_storage");
    expect(caught.stage).toBe("storage");
    expect(caught.message).toBe("stored episodic state is invalid");
    expect(loadInventory(file)).toEqual(inventory);
  },
);

test("history limits and pinned snapshots survive restart", async () => {
  const file = path.join(dir, "memory.sqlite");
  let old!: JsonObject;
  let expected!: JsonObject;
  await withMemory(new EpisodicMemory({ maxEpisodes: 2, recallLimit: 1, storage: file }), (memory) => {
    old = snapshotOf(memory);
    append(memory, "first");
    expected = snapshotOf(memory);
  });
  await withMemory(new EpisodicMemory({ storage: file }), async (memory) => {
    expect(snapshotOf(memory)).toEqual(expected);
    await withRun(memory, async (opened) => {
      expect(await opened.recall()).toContain("first");
      expect(opened.snapshot.config).toEqual({ max_episodes: 2n, recall_limit: 1n });
      expect(logJson(opened.snapshot)).not.toContain("storage");
      expect(logJson(opened.snapshot)).not.toContain(file);
    });
    await withRun(
      memory,
      async (opened) => {
        expect(opened.snapshot).toEqual(old);
        expect(await opened.recall()).toBe("");
      },
      old,
    );
    expect(snapshotOf(memory)).toEqual(expected);
  });
});

test.each(["opened", "reset", "committed"])("nonpristine flags survive restart (%s)", async (prior) => {
  const file = path.join(dir, "memory.sqlite");
  const foreign = snapshotOf(new EpisodicMemory());
  await withMemory(new EpisodicMemory({ storage: file }), async (memory) => {
    if (prior === "opened") snapshotOf(memory);
    else if (prior === "reset") memory.reset();
    else await withRun(memory, (opened) => opened.commit("empty", []));
  });
  await withMemory(new EpisodicMemory({ storage: file }), (memory) => {
    expect(failure(() => memory.openRun({ snapshot: foreign })).code).toBe("episodic.invalid_snapshot");
    const flag = { opened: "everOpened", reset: "everReset", committed: "everCommitted" }[prior]!;
    expect((memory as Mutable)[flag]).toBe(true);
  });
});

test("pristine adoption persists across restart", async () => {
  const file = path.join(dir, "memory.sqlite");
  const saved = await withMemory(new EpisodicMemory({ maxEpisodes: 5, recallLimit: 2 }), (source) => {
    append(source, "adopted");
    return snapshotOf(source);
  });
  new EpisodicMemory({ storage: file }).close();
  await withMemory(new EpisodicMemory({ storage: file }), (memory) =>
    withRun(memory, (opened) => expect(opened.snapshot).toEqual(saved), saved),
  );
  await withMemory(new EpisodicMemory({ storage: file }), (memory) => {
    expect(snapshotOf(memory)).toEqual(saved);
    expect((memory as Mutable).everOpened).toBe(true);
    expect((memory as Mutable).everCommitted).toBe(true);
  });
});

test("commit tombstones survive eviction reset restart and stale replay", async () => {
  const file = path.join(dir, "memory.sqlite");
  let old!: JsonObject;
  let batch!: MemoryMutation[];
  let expected!: JsonObject;
  await withMemory(new EpisodicMemory({ maxEpisodes: 1, recallLimit: 1, storage: file }), (memory) => {
    old = snapshotOf(memory);
    batch = append(memory, "first");
    append(memory, "second");
    expected = snapshotOf(memory);
    expect(logJson(expected)).not.toContain("first");
  });
  await withMemory(new EpisodicMemory({ storage: file }), async (memory) => {
    await withRun(
      memory,
      (stale) => {
        stale.commit("first", batch);
        expect(failure(() => stale.commit("first", [])).code).toBe("episodic.commit_collision");
      },
      old,
    );
    expect(snapshotOf(memory)).toEqual(expected);
    memory.reset();
  });
  await withMemory(new EpisodicMemory({ storage: file }), async (memory) => {
    await withRun(memory, (opened) => {
      opened.commit("first", batch);
      opened.commit("first", batch);
      expect(failure(() => opened.commit("first", [])).code).toBe("episodic.commit_collision");
    });
    const current = snapshotOf(memory);
    expect(current.episodes).toEqual([]);
    expect(current.revision).toBe(0n);
    expect(current.epoch).toBe((expected.epoch as bigint) + 1n);
  });
});

test.each(["open", "adopt", "reset", "commit"])("failed save never publishes candidate RAM (%s)", (operation) => {
  const memory = firstOpen(new EpisodicMemory({ storage: path.join(dir, "memory.sqlite") }));
  let session: EpisodicSession | null = null;
  if (operation === "commit") session = memory.openRun();
  const before = structuredClone(published(memory));
  const primary = new Error("save failed");
  const save = spyOn(storageOf(memory), "save").mockImplementation(() => {
    throw primary;
  });
  try {
    const caught = thrown(() => {
      if (operation === "open") memory.openRun();
      else if (operation === "adopt") memory.openRun({ snapshot: snapshotOf(new EpisodicMemory()) });
      else if (operation === "reset") memory.reset();
      else session!.commit("commit", session!.learn(context()));
    });
    expect(caught).toBe(primary);
    expect(published(memory)).toEqual(before);
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
    const memory = new EpisodicMemory({ storage: file });
    const opened = memory.openRun();
    const before = opened.snapshot;
    const connection: Sqlite = (storageOf(memory) as Mutable).connection;
    const realExec = connection.exec;
    const exec = spyOn(connection, "exec").mockImplementation((sql) => {
      if (sql !== "COMMIT") return Reflect.apply(realExec, connection, [sql]);
      if (afterCommit) Reflect.apply(realExec, connection, [sql]);
      throw primary;
    });
    try {
      const caught = thrown(() => opened.commit("uncertain", opened.learn(context())));
      if (cancel) {
        expect(caught).toBe(primary);
      } else {
        expect(caught).toBeInstanceOf(MemoryOperationError);
        expect((caught as MemoryOperationError).code).toBe("storage.io");
        expect(String(caught)).not.toContain("private path");
      }
      const m = memory as Mutable;
      expect(m.revision).toBe(0);
      expect(m.state.episodes).toEqual([]);
      expect(m.appliedCommits.size).toBe(0);
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
      expect(failure(() => firstOpen(new EpisodicMemory({ storage: file }))).code).toBe("storage.busy");
    } finally {
      exec.mockRestore();
      opened.close();
      memory.close();
    }
    await withMemory(new EpisodicMemory({ storage: file }), (reopened) => {
      expect(snapshotOf(reopened).revision).toBe(afterCommit ? 1n : 0n);
      expect((reopened as Mutable).namespace).toBe(before.namespace);
    });
  },
);

class Cleanup extends Error {}

test.each([
  ["a thrown non-Error value", { cancel: "initialization stopped" }],
  ["an Error", new Error("initialization stopped")],
])("first open cleanup preserves cancellation and releases ownership (%s)", async (_label, primary) => {
  const file = path.join(dir, "memory.sqlite");
  firstOpen(new EpisodicMemory({ storage: file })).close();
  const close = SQLiteStorage.prototype.close;
  const load = spyOn(SQLiteStorage.prototype, "load").mockImplementation(() => {
    throw primary;
  });
  const cleanup = spyOn(SQLiteStorage.prototype, "close").mockImplementation(function (this: SQLiteStorage) {
    Reflect.apply(close, this, []);
    throw new Cleanup("cleanup problem");
  });
  try {
    expect(thrown(() => firstOpen(new EpisodicMemory({ storage: file })))).toBe(primary);
  } finally {
    load.mockRestore();
    cleanup.mockRestore();
  }
  expect((primary as Mutable).notes).toEqual(["EpisodicMemory cleanup failed: Cleanup"]);
  await withMemory(new EpisodicMemory({ storage: file }), () => {});
});

test.each([false, true])(
  "active callback makes close busy without releasing file (reentrant %p)",
  async (reentrant) => {
    const file = path.join(dir, "memory.sqlite");
    const memory = new EpisodicMemory({ storage: file });
    const opened = memory.openRun();
    let entered!: () => void;
    const enteredPromise = new Promise<void>((resolve) => {
      entered = resolve;
    });
    let release!: () => void;
    const released = new Promise<void>((resolve) => {
      release = resolve;
    });
    memory.rewriteQuery = async () => {
      if (reentrant) {
        opened.close();
        expect(failure(() => memory.close()).code).toBe("episodic.busy");
      } else {
        entered();
        await released;
      }
      return "query";
    };
    const recall = opened.recall(new RecallContext("", "query", {}, ["agent"]));
    if (!reentrant) {
      await enteredPromise;
      try {
        expect(failure(() => memory.close()).code).toBe("episodic.busy");
        expect(failure(() => firstOpen(new EpisodicMemory({ storage: file }))).code).toBe("storage.busy");
      } finally {
        release();
      }
    }
    expect(await recall).toBe("");
    opened.close();
    memory.close();
    await withMemory(new EpisodicMemory({ storage: file }), () => {});
  },
  10_000,
);

test("invalid limits do not create storage", () => {
  const file = path.join(dir, "memory.sqlite");
  expect(() => new EpisodicMemory({ maxEpisodes: 0, storage: file })).toThrow(
    new RangeError("max_episodes must be positive"),
  );
  expect(fs.existsSync(file)).toBe(false);
});

test("recorded limits survive a reopen", async () => {
  const file = path.join(dir, "memory.sqlite");
  const recorded = await withMemory(new EpisodicMemory({ maxEpisodes: 4, recallLimit: 2, storage: file }), (memory) =>
    withRun(memory, (opened) => opened.snapshot),
  );
  await withMemory(new EpisodicMemory({ storage: file }), (memory) =>
    withRun(
      memory,
      (replay) => {
        expect(replay.snapshot).toEqual(recorded);
        replay.commit("declare", []);
      },
      recorded,
    ),
  );
  await withMemory(new EpisodicMemory({ storage: file }), (memory) => {
    expect(snapshotOf(memory).config).toEqual({ max_episodes: 4n, recall_limit: 2n });
  });
});

test("explicit limits must match stored limits", async () => {
  const file = path.join(dir, "memory.sqlite");
  firstOpen(new EpisodicMemory({ maxEpisodes: 4, recallLimit: 2, storage: file })).close();
  await withMemory(new EpisodicMemory({ storage: file }), (memory) => {
    expect(snapshotOf(memory).config).toEqual({ max_episodes: 4n, recall_limit: 2n });
  });
  await withMemory(new EpisodicMemory({ maxEpisodes: 4, storage: file }), (memory) => {
    expect((snapshotOf(memory).config as JsonObject).recall_limit).toBe(2n);
  });
  const caught = failure(() => firstOpen(new EpisodicMemory({ maxEpisodes: 5, storage: file })));
  expect(caught.code).toBe("episodic.config_conflict");
  expect(caught.stage).toBe("storage");
  firstOpen(new EpisodicMemory({ storage: file })).close();
});

test("a fresh store pins its default limits", () => {
  const file = path.join(dir, "memory.sqlite");
  firstOpen(new EpisodicMemory({ storage: file })).close();
  expect(failure(() => firstOpen(new EpisodicMemory({ maxEpisodes: 5, storage: file }))).code).toBe(
    "episodic.config_conflict",
  );
});

test("corrupt private state never partially adopts RAM", async () => {
  const file = path.join(dir, "memory.sqlite");
  await withMemory(new EpisodicMemory({ storage: file }), (memory) => append(memory, "saved"));
  const value = loadInventory(file);
  value.applied_commits.bad = "invalid";
  saveInventory(file, value);
  const partial = new EpisodicMemory({ storage: file });
  expect(failure(() => firstOpen(partial)).code).toBe("episodic.invalid_storage");
  const m = partial as Mutable;
  expect(m.namespace).not.toBe(value.snapshot.namespace);
  expect(m.state.episodes).toEqual([]);
  expect(m.appliedCommits.size).toBe(0);
  expect(m.epoch).toBe(0);
  expect(m.revision).toBe(0);
});

test("invalid last mutation never reaches storage", async () => {
  await withMemory(new EpisodicMemory({ storage: path.join(dir, "memory.sqlite") }), (memory) =>
    withRun(memory, (opened) => {
      const first = opened.learn(context())[0]!;
      const second = new MemoryMutation("wrong", first.payload);
      const calls: unknown[] = [];
      const save = spyOn(storageOf(memory), "save").mockImplementation((value) => {
        calls.push(value);
      });
      try {
        expect(failure(() => opened.commit("invalid", [first, second])).code).toBe("episodic.invalid_batch");
      } finally {
        save.mockRestore();
      }
      expect(calls).toEqual([]);
      expect((memory as Mutable).state.episodes).toEqual([]);
      expect((memory as Mutable).revision).toBe(0);
    }),
  );
});

/** The package root. */
const ROOT = path.resolve(import.meta.dirname, "../..");
const BUILT = path.join(ROOT, "dist/src/memory/episodic.js");

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/engine/memory_storage.test.ts`'s: importing a test file would run its tests.
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
    "Node reopens a store Bun wrote, commits, and Bun reads it back",
    async () => {
      const file = path.join(dir, "memory.sqlite");
      let batch!: MemoryMutation[];
      await withMemory(new EpisodicMemory({ maxEpisodes: 1, recallLimit: 1, storage: file }), (memory) => {
        batch = append(memory, "first");
        append(memory, "second");
      });
      const module = (name: string) => JSON.stringify(pathToFileURL(path.join(ROOT, `dist/src/memory/${name}.js`)).href);
      const script = `
const assert = (await import("node:assert/strict")).default;
const { EpisodicMemory } = await import(${module("episodic")});
const c = await import(${module("contracts")});
const context = (runId) =>
  new c.LearnContext(runId, "episodic", [
    new c.MemoryEpisode(runId, [], ["agent"], c.MemoryEpisodeStatus.SUCCEEDED, {}, { output: runId }),
  ]);
const memory = new EpisodicMemory({ storage: ${JSON.stringify(file)} });
try {
  const session = memory.openRun();
  assert.deepEqual(session.snapshot.config, { max_episodes: 1n, recall_limit: 1n });
  assert.equal(session.snapshot.revision, 2n);
  assert.match(await session.recall(), /second/);
  session.commit("first", session.learn(context("first")));
  const fresh = memory.openRun();
  assert.equal(fresh.snapshot.revision, 2n);
  fresh.close();
  session.commit("third", session.learn(context("third")));
  session.close();
} finally {
  memory.close();
}
console.log("ok");
`;
      const result = spawnSync(NODE!, ["--input-type=module", "-e", script], { encoding: "utf8", timeout: 20_000 });
      expect(result.stderr).toBe("");
      expect(result.stdout).toBe("ok\n");
      expect(result.status).toBe(0);
      await withMemory(new EpisodicMemory({ storage: file }), async (memory) => {
        const snapshot = snapshotOf(memory);
        expect(snapshot.revision).toBe(3n);
        expect(snapshot.config).toEqual({ max_episodes: 1n, recall_limit: 1n });
        await withRun(memory, async (opened) => {
          expect(await opened.recall()).toContain("third");
          opened.commit("first", batch);
          expect(failure(() => opened.commit("first", [])).code).toBe("episodic.commit_collision");
          expect(failure(() => opened.commit("second", [])).code).toBe("episodic.commit_collision");
        });
      });
    },
    30_000,
  );
});
