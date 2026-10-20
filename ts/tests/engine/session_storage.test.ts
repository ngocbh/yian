/**
 * Optional session persistence retains raw history and private derived state: a port of
 * `tests/engine/test_session_storage.py`.
 *
 * Ported: `test_completed_history_and_identity_survive_reopen`,
 * `test_partial_model_observation_and_unknown_request_survive_reopen`,
 * `test_notes_and_commit_tombstones_survive_restart_and_reset`,
 * `test_pristine_adoption_eligibility_survives_restart`,
 * `test_idle_handles_do_not_block_close_and_are_invalidated`,
 * `test_inventory_has_no_host_settings_or_live_handles`, `test_process_local_backend_never_opens_sqlite`,
 * `test_owner_kind_and_corrupt_store_are_rejected`,
 * `test_invalid_inventory_never_rewrites_store_and_releases_owner`,
 * `test_failed_save_does_not_publish_candidate_or_advance_ordinal`,
 * `test_uncertain_write_poisoning_keeps_ram_and_drains_lease`,
 * `test_summary_cache_persists_and_extends_without_losing_raw_prefix`,
 * `test_callback_drain_owns_backend_even_after_run_close` (its `project`, `summary` and `effect`
 * phases), `test_waiting_turn_keeps_fifo_and_cannot_be_discarded_by_close`,
 * `test_first_finish_failure_survives_secondary_release_failure`,
 * `test_process_crash_keeps_completed_evidence_and_unknown_effect`,
 * `test_surfaced_summary_is_saved_before_cooperative_stop`,
 * `test_poisoned_storage_still_drains_queued_and_active_owners`,
 * `test_frame_era_snapshot_is_rejected_without_mutation`,
 * `test_legacy_native_snapshot_is_rejected_without_mutation` and
 * `test_legacy_native_storage_is_rejected_without_rewriting`.
 *
 * TS-only: a failed first open whose storage close also fails notes `Conversation cleanup also
 * failed: Cleanup: cleanup problem` on the primary and can be retried; `±Infinity` in an item
 * payload survives publication, a snapshot restore and a storage reopen. With `YIAN_TEST_NODE` set to
 * a Node binary, the built backend reopens a store Bun wrote (an item payload holding `Infinity`, a
 * note, a cached summary) under Node, runs one turn, and Bun reads the result back.
 *
 * Changed:
 * - Python's `with SessionMemory(...)` is `withMemory`, which opens the storage first and closes the
 *   backend in `finally`; `_first_open` calls the private `guard()`.
 * - `test_process_local_backend_never_opens_sqlite`: spies on `SQLiteStorage.prototype.load` and
 *   `save` that throw stand in for Python forbidding `sqlite3.connect`.
 * - `test_owner_kind_and_corrupt_store_are_rejected`: the duplicate-key envelope is written through
 *   `openSqlite`, where Python uses `sqlite3`.
 * - `test_failed_save_does_not_publish_candidate_or_advance_ordinal`: RAM is compared by reference
 *   (every publication replaces frozen values), where Python compares deep copies; `admit` is
 *   `openTurn` rejecting and `finish` is `close()` throwing.
 * - `test_uncertain_write_poisoning_keeps_ram_and_drains_lease`: the failure is injected by a spy on
 *   the storage's private connection's `exec` for `COMMIT`, where Python subclasses
 *   `sqlite3.Connection`; the cancel rows throw a non-`Error` value (Python's `KeyboardInterrupt`);
 *   `close(primary)` returns, where Python's `__exit__` re-raises, and the poisoned close is noted on
 *   the primary.
 * - `test_callback_drain_owns_backend_even_after_run_close`: deferred promises, where Python blocks
 *   a thread; the `stop` rows are Not ported (below).
 * - `test_waiting_turn_keeps_fifo_and_cannot_be_discarded_by_close`: "is queued" is a macrotask later
 *   with the private `queue` length, where Python's checker sets an event.
 * - `test_first_finish_failure_survives_secondary_release_failure`: also checks the release failure's
 *   note on the primary.
 * - `test_process_crash_keeps_completed_evidence_and_unknown_effect`: the child is a `bun -e` script
 *   importing `src/memory/session.ts` that exits with 23 once the tool request is saved.
 * - `test_poisoned_storage_still_drains_queued_and_active_owners`: `SQLiteStorage` keeps its poison
 *   flag private, so a failed `COMMIT` poisons the store; a waiter learns of the poison at its next
 *   wake-up, so it rejects with `storage.poisoned` once the active turn's `close()` releases, where
 *   Python's waiter polls and fails before the owner closes.
 * - `test_surfaced_summary_is_saved_before_cooperative_stop`: the stop is an abort from inside
 *   `summarize`.
 * - JSON ints written into inventories are `bigint`s; mutable copies are `structuredClone`s;
 *   `with` blocks are `withMemory`/`withHarness`/`withTurn`; the frame-era fixture is an inlined
 *   copy of Python's `tests/fixtures/session_message_snapshot.json`, read with `parseJson`.
 *
 * Not ported: `test_pid_guard_precedes_condition_and_cleanup_locks` and
 * `test_callback_fork_does_not_deadlock_or_replace_primary` (JS can't fork a live process, so there
 * is no fork guard); the `stop` rows of `test_callback_drain_owns_backend_even_after_run_close` (a
 * stop check is a signal read and can't block while the backend is probed).
 */

import crypto from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";

import {
  type ConversationInvocation,
  ConversationItem,
  type ConversationView,
  type JsonObject,
  MemoryMutation,
  MemoryOperationError,
  RecallContext,
} from "../../src/memory/contracts.ts";
import { type Session, SessionMemory, type Turn } from "../../src/memory/session.ts";
import { SQLiteStorage } from "../../src/memory/storage/sqlite.ts";
import { openSqlite, type Sqlite } from "../../src/platform.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import {
  appendReply,
  caught,
  failure,
  item,
  macrotask,
  type Mutable,
  NEVER,
  rejected,
  request,
  settle,
  snapshot,
  withHarness,
  withTurn,
} from "./session_fakes.ts";

const CLOSED = "memory backend is closed";
const CLOSE_BUSY = "cannot close active or queued conversations";
const RESET_BUSY = "cannot reset active or queued conversations";
const STALE_SESSION = "memory session is closed or belongs to an old generation";
const INVALID_STORAGE = "stored session state is invalid";
const STORAGE_BUSY = "Memory storage is already owned or busy.";
const IO_FAILED = "Memory storage I/O failed.";
const POISONED = "Memory storage must be closed and reopened after an I/O failure.";
const UNSUPPORTED = "Native session history is unsupported; use fresh SessionMemory.";

let dir: string;
let file: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-session-"));
  file = path.join(dir, "memory.sqlite");
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

/** Open lazy storage without starting a run (Python's `_first_open`). */
function firstOpen(memory: SessionMemory): SessionMemory {
  memory["guard"]();
  return memory;
}

/** Open `memory`'s storage, await `fn` with it and close it (Python's `with SessionMemory(...)`). */
async function withMemory<T>(memory: SessionMemory, fn: (memory: SessionMemory) => T | Promise<T>): Promise<T> {
  firstOpen(memory);
  try {
    return await fn(memory);
  } finally {
    memory.close();
  }
}

/** Complete one turn named `name` with one reply. */
async function complete(memory: SessionMemory, name = "a"): Promise<void> {
  await withHarness((h) => withTurn(h.frame, request(name), (turn) => appendReply(turn, name, name)), memory);
}

/** Record `count` turns, each with one 20-character reply (session_summaries.test.ts's seed). */
async function seed(memory: SessionMemory, count: number): Promise<void> {
  await withHarness(async (opened) => {
    for (let index = 0; index < count; index++) {
      await withTurn(opened.frame, request(String(index)), (turn) =>
        appendReply(turn, `item-agent-${index}`, "x".repeat(20)),
      );
    }
  }, memory);
}

/** Project the selected invocations as the JSON array of their run ids (session_summaries.test.ts's). */
function project(_view: ConversationView, invocations: readonly ConversationInvocation[]): string {
  return JSON.stringify(invocations.map((invocation) => invocation.runId));
}

/**
 * Prepare history with a summary callback that records its calls and returns `result`
 * (session_summaries.test.ts's `prepare`, with fewer options).
 */
function prepare(turn: Turn, calls: [string, number][] = [], result = "summary") {
  return turn.prepareHistory({
    projectionVersion: "test-v1",
    project,
    summarize: (text, target) => {
      calls.push([text, target]);
      return result;
    },
  });
}

/** One closed note with its source identity, as session_notes.test.ts builds it. */
function note(runId = "a"): Mutable {
  const value = {
    consumer_static_path: ["root", "agent"],
    run_id: runId,
    execution_seq: 1n,
    message_index: 0n,
    text: "Check units.",
  };
  return { note_id: crypto.createHash("sha256").update(logJson(value)).digest("hex"), ...value };
}

/** The append-note mutation of the default note. */
function mutation(value: Mutable = note()): MemoryMutation {
  return new MemoryMutation(value.note_id, { operation: "append_note", note: value });
}

/** Open the store at `file` directly, load its inventory and close it. */
function loadInventory(): Mutable {
  const storage = new SQLiteStorage(file, "session");
  try {
    return storage.load();
  } finally {
    storage.close();
  }
}

/** Save `inventory` to the store at `file` directly. */
function saveInventory(inventory: unknown): void {
  const storage = new SQLiteStorage(file, "session");
  try {
    storage.save(inventory);
  } finally {
    storage.close();
  }
}

/** The private storage of `memory`. */
function storageOf(memory: SessionMemory): SQLiteStorage {
  return memory["storage"]!;
}

/** The publication fields of `memory`, in Python's `_ram` order. */
function ram(memory: SessionMemory): unknown[] {
  return [
    memory["namespace"],
    memory["generation"],
    memory["view"],
    memory["notes"],
    memory["used"],
    memory["appliedCommits"],
    memory["summaries"],
  ];
}

/** Check that `memory`'s publication fields are the very values in `before`. */
function expectSameRam(memory: SessionMemory, before: unknown[]): void {
  ram(memory).forEach((value, index) => expect(value).toBe(before[index]));
}

/** Return a SQLite driver-shaped error whose message is a private path. */
function sqliteError(): Error {
  return Object.assign(new Error("private-path"), { code: "ERR_SQLITE_ERROR", errcode: 1 });
}

/** Make the next `COMMIT` of `memory`'s storage throw `primary`, after committing when `afterCommit`. */
function failCommit(memory: SessionMemory, primary: unknown, afterCommit = false): { mockRestore(): void } {
  const connection: Sqlite = (storageOf(memory) as Mutable).connection;
  const realExec = connection.exec;
  return spyOn(connection, "exec").mockImplementation((sql) => {
    if (sql !== "COMMIT") return Reflect.apply(realExec, connection, [sql]);
    if (afterCommit) Reflect.apply(realExec, connection, [sql]);
    throw primary;
  });
}

test("completed history and identity survive reopen", async () => {
  let opening!: JsonObject;
  let saved!: JsonObject;
  await withMemory(new SessionMemory({ storage: file }), async (memory) => {
    opening = snapshot(memory);
    await complete(memory);
    saved = snapshot(memory);
  });
  await withMemory(new SessionMemory({ storage: file }), async (memory) => {
    expect(snapshot(memory)).toEqual(saved);
    const restored = memory.openRun({ snapshot: opening });
    try {
      expect(restored.snapshot).toEqual(opening);
      await withTurn(restored, request("b"), (turn) => {
        expect(turn.view.items).toEqual([item("a", "a")]);
        appendReply(turn, "b", "b");
      });
    } finally {
      restored.close();
    }
    expect((snapshot(memory) as Mutable).conversation.items.length).toBe(2);
  });
});

test("partial model observation and unknown request survive reopen", async () => {
  let saved!: JsonObject;
  await withMemory(new SessionMemory({ storage: file }), async (memory) => {
    await withHarness(async (h) => {
      const turn = await h.frame.openTurn(request(), { signal: NEVER });
      const ordinal = turn.request("model", { prompt: "question" });
      turn.observe(ordinal, "pending", { items: [item("partial", "working")] });
      h.session["release"](turn["lease"]!);
    }, memory);
    saved = snapshot(memory);
  });
  await withMemory(new SessionMemory({ storage: file }), async (memory) => {
    expect(snapshot(memory)).toEqual(saved);
    await withHarness(
      (h) =>
        withTurn(h.frame, request("retry"), (turn) => {
          expect(turn.view.items).toEqual([item("partial", "working")]);
          expect(turn.view.effects[0]!.status).toBe("pending");
          const repeated = turn.request("model", { prompt: "question" });
          turn.observe(repeated, "completed");
        }),
      memory,
    );
    const effects = (snapshot(memory) as Mutable).conversation.effects;
    expect(effects.map((effect: Mutable) => effect.status)).toEqual(["pending", "completed"]);
  });
});

test("notes and commit tombstones survive restart and reset", async () => {
  const learned = mutation();
  let saved!: JsonObject;
  await withMemory(new SessionMemory({ storage: file }), async (memory) => {
    await complete(memory);
    await withHarness((h) => h.session.commit("learn", [learned]), memory);
    saved = snapshot(memory);
  });
  await withMemory(new SessionMemory({ storage: file }), async (memory) => {
    expect(snapshot(memory)).toEqual(saved);
    await withHarness((h) => {
      h.session.commit("learn", [learned]);
      expect(h.frame.recall(new RecallContext("", "", {}, ["root", "agent"])).text).toBe("Check units.");
    }, memory);
    memory.reset();
  });
  await withMemory(new SessionMemory({ storage: file }), async (memory) => {
    await withHarness((h) => {
      h.session.commit("learn", [learned]);
      expect(h.session.snapshot.notes).toEqual([]);
      const collision = "commit id identifies a different batch";
      failure(() => h.session.commit("learn", []), "session.commit_collision", collision);
    }, memory);
  });
});

describe("pristine adoption eligibility survives restart", () => {
  for (const used of [false, true]) {
    test(`used ${used}`, async () => {
      const source = snapshot(new SessionMemory());
      let identity!: string;
      await withMemory(new SessionMemory({ storage: file }), (memory) => {
        identity = memory["namespace"];
        if (used) snapshot(memory);
      });
      await withMemory(new SessionMemory({ storage: file }), (memory) => {
        expect(memory["namespace"]).toBe(identity);
        if (used) {
          failure(
            () => memory.openRun({ snapshot: source }),
            "session.invalid_snapshot",
            "snapshot is not proven history of this backend",
          );
        } else {
          const adopted = memory.openRun({ snapshot: source });
          expect(adopted.snapshot).toEqual(source);
          adopted.close();
        }
      });
    });
  }
});

test("idle handles do not block close and are invalidated", async () => {
  const memory = new SessionMemory({ storage: file });
  const opened = memory.openRun();
  memory.close();
  memory.close();
  for (const operation of [
    () => opened.snapshot,
    () => opened.recall(),
    () => memory.openRun(),
    () => memory.reset(),
  ]) {
    failure(operation, "session.closed", CLOSED);
  }
  opened.close();
  await withMemory(new SessionMemory({ storage: file }), () => {});
});

test("inventory has no host settings or live handles", async () => {
  await withMemory(new SessionMemory({ storage: file, historyBudgetChars: 17, recentTurns: 0 }), (memory) =>
    complete(memory),
  );
  const encoded = logJson(loadInventory());
  for (const forbidden of ["history_budget_chars", "recent_turns", "_condition", "_queues", "ticket", "lease"]) {
    expect(encoded).not.toContain(forbidden);
  }
});

test("process local backend never opens sqlite", async () => {
  const spies = (["load", "save"] as const).map((name) =>
    spyOn(SQLiteStorage.prototype, name).mockImplementation(() => {
      throw new Error("process-local memory must not use SQLite");
    }),
  );
  try {
    await withMemory(new SessionMemory(), async (memory) => {
      await complete(memory);
      memory.reset();
    });
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  } finally {
    for (const spy of spies) spy.mockRestore();
  }
  expect(fs.readdirSync(dir)).toEqual([]);
});

test("owner kind and corrupt store are rejected", async () => {
  await withMemory(new SessionMemory({ storage: file }), () => {
    failure(() => firstOpen(new SessionMemory({ storage: file })), "storage.busy", STORAGE_BUSY);
  });
  const kind = "Memory storage belongs to a different backend kind.";
  failure(() => new SQLiteStorage(file, "episodic"), "storage.kind", kind);
  const connection = openSqlite(file);
  try {
    connection.prepare("UPDATE memory_state SET envelope = ?").run('{"version":1,"version":1}');
  } finally {
    connection.close();
  }
  failure(
    () => firstOpen(new SessionMemory({ storage: file })),
    "storage.corrupt",
    "Memory storage contains an invalid state envelope.",
  );
});

describe("invalid inventory never rewrites store and releases owner", () => {
  const rows: [string, (value: Mutable, summary: Mutable) => void][] = [
    ["extra", (value) => (value.extra = null)],
    ["version", (value) => (value.version = true)],
    ["used_type", (value) => (value.used = 1n)],
    ["unused", (value) => (value.used = false)],
    ["generation", (value) => (value.snapshot.generation = -1n)],
    ["commits_type", (value) => (value.applied_commits = [])],
    ["commit_id", (value) => (value.applied_commits = { "": "a".repeat(64) })],
    ["digest", (value) => (value.applied_commits = { c: "A".repeat(64) })],
    [
      "duplicate_view",
      (value) => (value.snapshot.conversation = [value.snapshot.conversation, value.snapshot.conversation]),
    ],
    ["foreign_view", (value) => (value.snapshot.conversation.namespace = "foreign")],
    ["notes", (value) => (value.snapshot.notes = [note("unadmitted")])],
    ["summaries_type", (value) => (value.summaries = {})],
    ["summary_extra", (_, summary) => (summary.extra = null)],
    ["summary_namespace", (_, summary) => (summary.namespace = "foreign")],
    ["summary_generation", (_, summary) => (summary.generation = true)],
    ["summary_version", (_, summary) => (summary.projection_version = "")],
    ["summary_covered", (_, summary) => (summary.covered[0].start_sequence = 9n)],
    ["summary_empty", (_, summary) => (summary.covered = [])],
    ["summary_current", (_, summary) => summary.covered.push({ run_id: "current", start_sequence: 1n })],
    ["summary_duplicate", (value) => (value.summaries = [value.summaries[0], value.summaries[0]])],
    ["summary_digest", (_, summary) => (summary.digest = "a".repeat(64))],
    ["summary_text", (_, summary) => (summary.text = "   ")],
  ];
  for (const [name, mutate] of rows) {
    test(name, async () => {
      await withMemory(new SessionMemory({ storage: file, historyBudgetChars: 1, recentTurns: 0 }), async (memory) => {
        await seed(memory, 1);
        await withHarness((h) => withTurn(h.frame, request("current"), (turn) => prepare(turn)), memory);
      });
      const value = loadInventory();
      mutate(value, value.summaries[0]);
      saveInventory(value);
      const error = failure(
        () => firstOpen(new SessionMemory({ storage: file })),
        "session.invalid_storage",
        INVALID_STORAGE,
      );
      expect(error.stage).toBe("storage");
      expect(loadInventory()).toEqual(value);
    });
  }
});

describe("failed save does not publish candidate or advance ordinal", () => {
  for (const operation of [
    "open",
    "adopt",
    "reset",
    "commit",
    "admit",
    "request",
    "observe",
    "summary",
    "finish",
  ] as const) {
    test(operation, async () => {
      const memory = new SessionMemory({ storage: file, historyBudgetChars: 1, recentTurns: 0 });
      firstOpen(memory);
      let opened: Session | null = null;
      let turn: Turn | null = null;
      let ordinal = 0;
      if (!["open", "adopt", "reset"].includes(operation)) {
        await complete(memory);
        opened = memory.openRun();
        if (operation !== "commit" && operation !== "admit") {
          turn = await opened.openTurn(request("next"), { signal: NEVER });
          if (operation === "observe") ordinal = turn.request("model", {});
        }
      }
      const foreign = operation === "adopt" ? snapshot(new SessionMemory()) : null;
      const before = ram(memory);
      const beforeOrdinal = turn?.["ordinal"];
      const error = new Error("save rejected");
      const save = spyOn(storageOf(memory), "save").mockImplementation(() => {
        throw error;
      });
      try {
        let thrown: unknown;
        if (operation === "open") thrown = caught(() => memory.openRun());
        else if (operation === "adopt") thrown = caught(() => memory.openRun({ snapshot: foreign! }));
        else if (operation === "reset") thrown = caught(() => memory.reset());
        else if (operation === "commit") thrown = caught(() => opened!.commit("learn", [mutation()]));
        else if (operation === "admit") thrown = await rejected(opened!.openTurn(request("next"), { signal: NEVER }));
        else if (operation === "request") thrown = caught(() => turn!.request("model", {}));
        else if (operation === "observe") {
          thrown = caught(() => turn!.observe(ordinal, "completed", { items: [item("new", "observed")] }));
        } else if (operation === "summary") thrown = await rejected(prepare(turn!));
        else thrown = caught(() => turn!.close());
        expect(thrown).toBe(error);
        expectSameRam(memory, before);
        if (turn !== null) expect(turn["ordinal"]).toBe(beforeOrdinal!);
      } finally {
        save.mockRestore();
        if (turn?.["lease"]) opened!["release"](turn["lease"]);
        opened?.close();
      }
      memory.close();
    });
  }
});

describe("uncertain write poisoning keeps ram and drains lease", () => {
  for (const afterCommit of [false, true]) {
    for (const cancel of [false, true]) {
      test(`after commit ${afterCommit} cancel ${cancel}`, async () => {
        const error: unknown = cancel ? { cancel: "stop" } : sqliteError();
        const memory = new SessionMemory({ storage: file });
        const opened = memory.openRun();
        const turn = await opened.openTurn(request(), { signal: NEVER });
        const before = ram(memory);
        const beforeOrdinal = turn["ordinal"];
        const exec = failCommit(memory, error, afterCommit);
        let primary: unknown;
        try {
          const write = () => turn.request("model", {});
          primary = cancel ? caught(write) : failure(write, "storage.io", IO_FAILED);
        } finally {
          exec.mockRestore();
        }
        if (cancel) expect(primary).toBe(error);
        else expect(String(primary)).not.toContain("private-path");
        expectSameRam(memory, before);
        expect(turn["ordinal"]).toBe(beforeOrdinal);
        for (const operation of [
          () => memory.openRun(),
          () => memory.reset(),
          () => opened.snapshot,
          () => opened.recall(),
          () => opened.commit("c", []),
          () => turn.view,
          () => turn.request("model", {}),
        ]) {
          failure(operation, "storage.poisoned", POISONED);
        }
        const summary = (await rejected(prepare(turn))) as MemoryOperationError;
        expect([summary.code, summary.message]).toEqual(["storage.poisoned", POISONED]);
        failure(() => memory.close(), "session.busy", CLOSE_BUSY);
        expect(turn.close(primary)).toBeUndefined();
        const note = `Conversation cleanup also failed: MemoryOperationError: ${POISONED}`;
        expect((primary as Mutable).notes).toEqual([note]);
        expect(memory["active"]).toBeNull();
        expect(memory["queue"].length).toBe(0);
        opened.close();
        memory.close();
        await withMemory(new SessionMemory({ storage: file }), (reopened) => {
          const view = (snapshot(reopened) as Mutable).conversation;
          expect(view.effects.length).toBe(afterCommit ? 1 : 0);
          if (afterCommit) expect(view.effects[0].status).toBe("pending");
        });
      });
    }
  }
});

test("summary cache persists and extends without losing raw prefix", async () => {
  let saved!: Mutable;
  let cache!: Mutable[];
  await withMemory(new SessionMemory({ storage: file, historyBudgetChars: 1, recentTurns: 1 }), async (memory) => {
    await seed(memory, 3);
    await withHarness(
      (h) =>
        withTurn(h.frame, request("current"), async (turn) => {
          const calls: [string, number][] = [];
          expect((await prepare(turn, calls, "oversized saved summary"))[0]).toBe("oversized saved summary");
          expect((await prepare(turn, calls))[0]).toBe("oversized saved summary");
          expect(calls.length).toBe(1);
        }),
      memory,
    );
    saved = snapshot(memory);
    cache = [...memory["summaries"]];
  });
  await withMemory(new SessionMemory({ storage: file, historyBudgetChars: 1, recentTurns: 1 }), async (memory) => {
    expect([...memory["summaries"]]).toEqual(cache);
    expect(snapshot(memory)).toEqual(saved);
    await withHarness(
      (h) =>
        withTurn(h.frame, request("next"), async (turn) => {
          const calls: [string, number][] = [];
          expect((await prepare(turn, calls, "extended"))[0]).toBe("extended");
          expect(calls.length).toBe(1);
          expect(calls[0]![0]).toContain("oversized saved summary");
          expect(calls[0]![0]).toContain('"2"');
          expect(calls[0]![0]).not.toContain('"0"');
        }),
      memory,
    );
    expect((snapshot(memory) as Mutable).conversation.items).toEqual(saved.conversation.items);
    memory.reset();
  });
  await withMemory(new SessionMemory({ storage: file }), (memory) => {
    expect(memory["summaries"].size).toBe(0);
  });
});

describe("callback drain owns backend even after run close", () => {
  for (const phase of ["project", "summary", "effect"] as const) {
    for (const closeRun of [false, true]) {
      test(`${phase} close run ${closeRun}`, async () => {
        const memory = new SessionMemory({ storage: file, historyBudgetChars: 1, recentTurns: 0 });
        await seed(memory, 1);
        const opened = memory.openRun();
        const entered = Promise.withResolvers<void>();
        const release = Promise.withResolvers<void>();
        let once = false;
        const callback = async (): Promise<void> => {
          if (once) return;
          once = true;
          entered.resolve();
          await release.promise;
        };
        const work = settle(
          withTurn(opened, request("next"), async (turn) => {
            if (phase === "effect") {
              const ordinal = turn.request("model", {});
              await callback();
              turn.observe(ordinal, "completed", { items: [item("actual", "result")] });
            } else {
              await turn.prepareHistory({
                projectionVersion: "v1",
                project: async (view, selected) => {
                  if (phase === "project") await callback();
                  return project(view, selected);
                },
                summarize: async () => {
                  if (phase === "summary") await callback();
                  return "saved summary";
                },
              });
            }
          }),
        );
        await entered.promise;
        try {
          failure(() => memory.close(), "session.busy", CLOSE_BUSY);
          failure(() => memory.reset(), "session.busy", RESET_BUSY);
          if (closeRun) {
            opened.close();
            failure(() => memory.close(), "session.busy", CLOSE_BUSY);
          }
          failure(() => firstOpen(new SessionMemory({ storage: file })), "storage.busy", STORAGE_BUSY);
        } finally {
          release.resolve();
          await work.promise.catch(() => {});
        }
        if (closeRun && phase !== "effect") {
          const error = work.error as MemoryOperationError;
          expect(error).toBeInstanceOf(MemoryOperationError);
          expect([error.code, error.message]).toEqual(["session.stale", STALE_SESSION]);
        } else {
          expect(work.error).toBeUndefined();
        }
        if (phase === "summary") expect([...memory["summaries"].values()][0]!.text).toBe("saved summary");
        if (phase === "effect") {
          expect((snapshot(memory) as Mutable).conversation.items.at(-1).item_id).toBe("actual");
        }
        opened.close();
        memory.close();
        await withMemory(new SessionMemory({ storage: file }), () => {});
      });
    }
  }
});

test("waiting turn keeps fifo and cannot be discarded by close", async () => {
  const memory = new SessionMemory({ storage: file });
  const first = memory.openRun();
  const later = memory.openRun();
  const turn = await first.openTurn(request(), { signal: NEVER });
  const finished: string[] = [];
  const worker = settle(withTurn(later, request("b"), () => void finished.push("b")));
  await macrotask();
  expect(memory["queue"].length).toBe(1);
  first.close();
  failure(() => memory.close(), "session.busy", CLOSE_BUSY);
  failure(() => memory.reset(), "session.busy", RESET_BUSY);
  expect(finished).toEqual([]);
  turn.close();
  await worker.promise;
  expect(worker.error).toBeUndefined();
  expect(finished).toEqual(["b"]);
  later.close();
  memory.close();
});

test("first finish failure survives secondary release failure", async () => {
  const memory = new SessionMemory({ storage: file });
  const opened = memory.openRun();
  const turn = await opened.openTurn(request(), { signal: NEVER });
  const primary: Mutable = new Error("close publication failed");
  const save = spyOn(storageOf(memory), "save").mockImplementation(() => {
    throw primary;
  });
  const release = spyOn(opened as Mutable, "release").mockImplementation(() => {
    throw new Error("lease cleanup also failed");
  });
  try {
    expect(caught(() => turn.close())).toBe(primary);
  } finally {
    save.mockRestore();
    release.mockRestore();
  }
  expect(primary.notes).toEqual(["Conversation cleanup also failed: Error: lease cleanup also failed"]);
  opened["release"](turn["lease"]!);
  opened.close();
  memory.close();
});

/** The `ts/` directory. */
const ROOT = path.resolve(import.meta.dir, "../..");

test("process crash keeps completed evidence and unknown effect", async () => {
  const module = (name: string) => JSON.stringify(pathToFileURL(path.join(ROOT, `src/memory/${name}.ts`)).href);
  const script = `
const { SessionMemory } = await import(${module("session")});
const c = await import(${module("contracts")});
const item = (id, text) => new c.ConversationItem(id, "fake-v1", { role: "assistant", content: text });
const memory = new SessionMemory({ storage: ${JSON.stringify(file)} });
const opened = memory.openRun();
const invocation = new c.ConversationInvocation("a", 1);
const request = new c.ConversationRequest(["root", "agent"], invocation, { format: "fake-v1" });
const turn = await opened.openTurn(request, { signal: new AbortController().signal });
const completed = turn.request("model", {}, { items: [item("u", "question")] });
turn.observe(completed, "completed", { items: [item("a", "answer")] });
turn.request("tool", { name: "repeatable" });
process.exit(23);
`;
  const result = spawnSync(process.execPath, ["-e", script], { encoding: "utf8", timeout: 20_000 });
  expect(result.stderr).toBe("");
  expect(result.status).toBe(23);
  await withMemory(new SessionMemory({ storage: file }), (memory) =>
    withHarness(
      (h) =>
        withTurn(h.frame, request("retry"), (turn) => {
          expect(turn.view.items.map((entry) => entry.payload.content)).toEqual(["question", "answer"]);
          expect(turn.view.effects.map((effect) => effect.status)).toEqual(["completed", "pending"]);
          const ordinal = turn.request("tool", { name: "repeatable" });
          turn.observe(ordinal, "completed");
        }),
      memory,
    ),
  );
}, 30_000);

test("surfaced summary is saved before cooperative stop", async () => {
  const stop = new AbortController();
  const primary = new Error("stop after summary");
  let saved!: JsonObject;
  await withMemory(new SessionMemory({ storage: file, historyBudgetChars: 1, recentTurns: 0 }), async (memory) => {
    await seed(memory, 1);
    await withHarness(async (h) => {
      const error = await rejected(
        withTurn(
          h.frame,
          request("current"),
          (turn) =>
            turn.prepareHistory({
              projectionVersion: "v1",
              project,
              summarize: () => {
                stop.abort(primary);
                return "summary returned before stop";
              },
            }),
          stop.signal,
        ),
      );
      expect(error).toBe(primary);
      saved = snapshot(memory);
    }, memory);
  });
  await withMemory(new SessionMemory({ storage: file }), (memory) => {
    expect(snapshot(memory)).toEqual(saved);
    expect([...memory["summaries"].values()][0]!.text).toBe("summary returned before stop");
  });
});

test("poisoned storage still drains queued and active owners", async () => {
  const memory = new SessionMemory({ storage: file });
  const opened = memory.openRun();
  const other = memory.openRun();
  const turn = await opened.openTurn(request(), { signal: NEVER });
  const waiting = settle(
    withTurn(other, request("next"), () => {
      throw new Error("poisoned turn was admitted");
    }),
  );
  await macrotask();
  expect(memory["queue"].length).toBe(1);
  const exec = failCommit(memory, sqliteError());
  try {
    failure(() => turn.request("model", {}), "storage.io", IO_FAILED);
  } finally {
    exec.mockRestore();
  }
  // No polling: the waiter learns of the poison when the active turn's release wakes it.
  await macrotask();
  expect(waiting.settled).toBe(false);
  failure(() => turn.close(), "storage.poisoned", POISONED);
  expect(memory["active"]).toBeNull();
  const error = (await rejected(waiting.promise)) as MemoryOperationError;
  expect([error.code, error.message]).toEqual(["storage.poisoned", POISONED]);
  expect(memory["queue"].length).toBe(0);
  opened.close();
  other.close();
  memory.close();
});

// An inlined copy of Python's `tests/fixtures/session_message_snapshot.json`, a version 2
// (frame-era) snapshot; `String.raw` keeps its escaped quotes.
const MESSAGE_SNAPSHOT = String.raw`{
  "conversations": [
    {
      "applied": [
        {
          "after_digest": "340c77d03a14ee4069f5769053b5cdc18c901aadf30b3ce31ab1729707a10292",
          "before_digest": "ae8ed827e4dd0be75190e7e2d999d4e38f140474b2e9602d0ba9868390b66deb",
          "event_digest": "bce09040d538b94b4c0ba534257632f25b54697c7b01a9bb6cad8500bf591939",
          "operation_id": "[\"baseline\",1,0]"
        },
        {
          "after_digest": "7d4b88e03d26cf3c9ddd75a1ada5a2a3143988754ba3b816aa774727f55676a8",
          "before_digest": "340c77d03a14ee4069f5769053b5cdc18c901aadf30b3ce31ab1729707a10292",
          "event_digest": "8e509b4b2a02955856af5c709041189ae5cda07cbfa1dd457258c95ff9475248",
          "operation_id": "[\"baseline\",1,1]"
        },
        {
          "after_digest": "82ee368dce56d902e99070b284d1f2a33d33b18847f909674ca60bf49de470de",
          "before_digest": "7d4b88e03d26cf3c9ddd75a1ada5a2a3143988754ba3b816aa774727f55676a8",
          "event_digest": "e18aca61c8efea91373415b6471f494c77e38b2812fcfa2e3f82b62178c00b36",
          "operation_id": "[\"baseline\",1,2]"
        },
        {
          "after_digest": "00bde4f30b344d1981bed0af147166481cc0a137b4908dda8c62237ff2a11b62",
          "before_digest": "82ee368dce56d902e99070b284d1f2a33d33b18847f909674ca60bf49de470de",
          "event_digest": "3b3c81d604372bf8afe8e9681fa10d980d3d4686ff51f665b2ede0846ab5a3ab",
          "operation_id": "[\"baseline\",1,3]"
        }
      ],
      "closed_invocations": [
        {
          "closed_ordinal": 3,
          "invocation": {
            "run_id": "baseline",
            "start_sequence": 1
          },
          "status": "completed"
        }
      ],
      "compatibility": {
        "format": "fake-v1"
      },
      "effects": [
        {
          "effect_ordinal": 1,
          "failure": null,
          "invocation": {
            "run_id": "baseline",
            "start_sequence": 1
          },
          "kind": "model",
          "observation_ordinals": [
            2
          ],
          "request": {
            "prompt": "Hello"
          },
          "status": "completed"
        }
      ],
      "generation": 0,
      "items": [
        {
          "format": "fake-v1",
          "item_id": "input",
          "payload": {
            "content": "Hello",
            "role": "user"
          }
        },
        {
          "format": "fake-v1",
          "item_id": "output",
          "payload": {
            "content": "Hello back",
            "role": "assistant"
          }
        }
      ],
      "key": {
        "consumer_static_path": [
          "root",
          "agent"
        ],
        "frame_ref": {
          "owner": "flow",
          "root_flow_id": "root",
          "static_path": [
            "root"
          ]
        }
      },
      "namespace": "baseline-message-snapshot",
      "reference": null,
      "revision": 4
    }
  ],
  "generation": 0,
  "namespace": "baseline-message-snapshot",
  "notes": [],
  "version": 2
}`;

/** Python's frame-era snapshot fixture, ints as `bigint`s. */
function messageSnapshotFixture(): Mutable {
  return parseJson(MESSAGE_SNAPSHOT, "plain");
}

test("frame era snapshot is rejected without mutation", async () => {
  const original = messageSnapshotFixture();
  const expected = structuredClone(original);
  await withMemory(new SessionMemory({ storage: file }), (memory) => {
    const before = snapshot(memory);
    failure(() => memory.openRun({ snapshot: original }), "session.invalid_snapshot", "invalid SessionMemory snapshot");
    expect(snapshot(memory)).toEqual(before);
  });
  // A pristine backend would adopt any valid snapshot, so this isolates the version check from the
  // used-backend lineage check above.
  failure(
    () => new SessionMemory().openRun({ snapshot: original }),
    "session.invalid_snapshot",
    "invalid SessionMemory snapshot",
  );
  expect(original).toEqual(expected);
});

describe("legacy native snapshot is rejected without mutation", () => {
  for (const legacy of ["reference", "native_session"] as const) {
    test(legacy, async () => {
      const source = new SessionMemory();
      await complete(source);
      const original: Mutable = structuredClone(snapshot(source));
      if (legacy === "reference") original.conversation.reference = { provider: "codex", session_id: "thread" };
      else original.conversation.effects[0].kind = "native_session";
      const expected = structuredClone(original);
      let before!: JsonObject;
      await withMemory(new SessionMemory({ storage: file }), (memory) => {
        before = snapshot(memory);
        const stored = storageOf(memory).load();
        failure(() => memory.openRun({ snapshot: original }), "session.unsupported_history", UNSUPPORTED);
        expect(snapshot(memory)).toEqual(before);
        expect(storageOf(memory).load()).toEqual(stored);
      });
      expect(original).toEqual(expected);
      await withMemory(new SessionMemory({ storage: file }), (memory) => {
        expect(snapshot(memory)).toEqual(before);
      });
    });
  }
});

describe("legacy native storage is rejected without rewriting", () => {
  for (const legacy of ["reference", "native_session"] as const) {
    test(legacy, async () => {
      await withMemory(new SessionMemory({ storage: file }), (memory) => complete(memory));
      const value = loadInventory();
      const view = value.snapshot.conversation;
      if (legacy === "reference") view.reference = { provider: "codex", session_id: "thread" };
      else view.effects[0].kind = "native_session";
      saveInventory(value);
      const originalBytes = fs.readFileSync(file);
      failure(() => firstOpen(new SessionMemory({ storage: file })), "session.unsupported_history", UNSUPPORTED);
      expect(fs.readFileSync(file)).toEqual(originalBytes);
      expect(loadInventory()).toEqual(value);
    });
  }
});

describe("TS-only", () => {
  test("infinite payload values survive publication, restore and reopen", async () => {
    let exported!: Mutable;
    await withMemory(new SessionMemory({ storage: file }), async (memory) => {
      await withHarness(
        (h) =>
          withTurn(h.frame, request(), (turn) => {
            const ordinal = turn.request("model", { limit: Infinity });
            const infinite = new ConversationItem("inf", "fake-v1", { up: Infinity, down: -Infinity });
            turn.observe(ordinal, "completed", { items: [infinite] });
          }),
        memory,
      );
      exported = snapshot(memory);
    });
    expect(exported.conversation.items[0].payload).toEqual({ down: -Infinity, up: Infinity });
    expect(exported.conversation.effects[0].request).toEqual({ limit: Infinity });
    const restored = new SessionMemory().openRun({ snapshot: exported });
    try {
      expect(restored.snapshot).toEqual(exported);
      await withTurn(restored, request("b"), (turn) => {
        expect(turn.view.items[0]!.payload).toEqual({ down: -Infinity, up: Infinity });
      });
    } finally {
      restored.close();
    }
    await withMemory(new SessionMemory({ storage: file }), (memory) => {
      expect(snapshot(memory)).toEqual(exported);
    });
  });

  test("a failed first open notes a failed storage close and can be retried", async () => {
    firstOpen(new SessionMemory({ storage: file })).close();
    class Cleanup extends Error {}
    const primary: Mutable = new Error("initialization stopped");
    const close = SQLiteStorage.prototype.close;
    const load = spyOn(SQLiteStorage.prototype, "load").mockImplementation(() => {
      throw primary;
    });
    const cleanup = spyOn(SQLiteStorage.prototype, "close").mockImplementation(function (this: SQLiteStorage) {
      Reflect.apply(close, this, []);
      throw new Cleanup("cleanup problem");
    });
    const memory = new SessionMemory({ storage: file });
    try {
      expect(caught(() => firstOpen(memory))).toBe(primary);
    } finally {
      load.mockRestore();
      cleanup.mockRestore();
    }
    expect(primary.notes).toEqual(["Conversation cleanup also failed: Cleanup: cleanup problem"]);
    expect(memory["storage"]).toBeNull();
    await withMemory(memory, () => {});
  });
});

const BUILT = path.join(ROOT, "dist/src/memory/session.js");

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
    "Node reopens a store Bun wrote, runs a turn, and Bun reads it back",
    async () => {
      const settings = { storage: file, historyBudgetChars: 1, recentTurns: 0 };
      await withMemory(new SessionMemory(settings), async (memory) => {
        await withHarness(
          (h) =>
            withTurn(h.frame, request("0"), (turn) => {
              const ordinal = turn.request("model", {});
              const infinite = new ConversationItem("inf", "fake-v1", { content: "x", score: Infinity });
              turn.observe(ordinal, "completed", { items: [infinite] });
            }),
          memory,
        );
        await withHarness((h) => h.session.commit("learn", [mutation(note("0"))]), memory);
        const current = request("current");
        await withHarness((h) => withTurn(h.frame, current, (turn) => prepare(turn, [], "bun summary")), memory);
      });
      const module = (name: string) =>
        JSON.stringify(pathToFileURL(path.join(ROOT, `dist/src/memory/${name}.js`)).href);
      const script = `
const assert = (await import("node:assert/strict")).default;
const { SessionMemory } = await import(${module("session")});
const c = await import(${module("contracts")});
const memory = new SessionMemory({ storage: ${JSON.stringify(file)}, historyBudgetChars: 1, recentTurns: 0 });
try {
  const session = memory.openRun();
  const snapshot = session.snapshot;
  assert.equal(snapshot.conversation.items[0].payload.score, Infinity);
  assert.equal(snapshot.notes.length, 1);
  assert.deepEqual([...memory.summaries.values()].map((summary) => summary.text), ["bun summary"]);
  const invocation = new c.ConversationInvocation("node", 1);
  const request = new c.ConversationRequest(["root", "agent"], invocation, { format: "fake-v1" });
  const turn = await session.openTurn(request, { signal: new AbortController().signal });
  const ordinal = turn.request("model", { prompt: "node" });
  turn.observe(ordinal, "completed", {
    items: [new c.ConversationItem("node", "fake-v1", { role: "assistant", content: "from node" })],
  });
  turn.close();
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
      await withMemory(new SessionMemory(settings), (memory) => {
        const view = (snapshot(memory) as Mutable).conversation;
        expect(view.items[0].payload.score).toBe(Infinity);
        expect(view.items.at(-1).item_id).toBe("node");
        expect(view.closed_invocations.at(-1).status).toBe("completed");
        expect((snapshot(memory) as Mutable).notes.length).toBe(1);
        expect([...memory["summaries"].values()].map((summary) => summary.text)).toEqual(["bun summary"]);
      });
    },
    30_000,
  );
});
