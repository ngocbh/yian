/**
 * Single-owner SQLite storage for memory-backend state: a port of
 * `tests/engine/test_memory_storage.py`.
 *
 * Ported: `test_roundtrip_is_detached_and_survives_reopen`,
 * `test_invalid_candidate_is_rejected_without_poisoning`, `test_cycles_are_rejected_before_a_transaction`,
 * `test_only_file_paths_are_accepted`, `test_backend_kind_requires_a_nonempty_string`,
 * `test_missing_parent_and_directory_are_rejected_without_path_disclosure`,
 * `test_new_file_permissions_and_connection_settings`, `test_new_file_permissions_do_not_depend_on_umask`,
 * `test_path_io_errors_are_sanitized`, `test_zero_byte_file_initializes_without_changing_permissions`,
 * `test_invalid_store_rejected_without_overwriting_or_leaking_owner`, `test_unrelated_files_are_preserved`,
 * `test_lifetime_owner_survives_commits_and_failed_competitors`,
 * `test_failed_save_poisoning_retains_ownership_and_primary`,
 * `test_close_is_idempotent_and_closed_methods_do_not_touch_os_or_sqlite`,
 * `test_failed_constructor_rolls_back_and_releases_connection`,
 * `test_rollback_failure_does_not_replace_save_failure_or_release_owner`,
 * `test_load_io_failure_poisoning_preserves_owner`, `test_process_crash_at_commit_is_atomic`,
 * `test_creation_crash_leaves_recoverable_empty_file`,
 * `test_process_death_releases_owner_and_preserves_commit`,
 * `test_check_storage_path_returns_an_absolute_path_without_io`,
 * `test_storage_path_follows_symlinks_before_parent_steps` and
 * `test_create_parent_makes_missing_directories`.
 *
 * TS-only: the value number rule in a stored state (`±Infinity` round-trips, `-0` reads `0`), an
 * int beyond 2^64, records shaped like the infinity tag or its escape; `checkStoragePath` keeping
 * `..` in a relative path and writing no `//` when the current directory is `/`; and
 * `realStoragePath`, Python's non-strict `os.path.realpath` ported line for line (a symlink
 * followed after a missing component and its `..`, a dangling link, a loop, a relative link target).
 * With `YIAN_TEST_NODE` set to a Node binary, the built store also runs under Node: a round trip
 * with `±Infinity` and a big int, a reopen and damaged stores, and an owner on each runtime
 * excluding a competitor on the other.
 *
 * Changed:
 * - The envelope is version 3 (Python's is 2): the `version` row writes 4, a `python_version` row
 *   writes Python's 2, and both fail `storage.version`, as the frame-era 1 does. Each damaged store
 *   is pinned with its code (Python pins only "not busy" and the version rows).
 * - `test_invalid_candidate_is_rejected_without_poisoning`: the infinity row round-trips (stored
 *   states follow the yian value number rule); the `object()`, int-key and tuple rows are a class
 *   instance, a `Map` and a symbol key (a JS key is always a string, and there is no tuple).
 * - `test_cycles_are_rejected_before_a_transaction`: a spy on the connection's `exec` sees no
 *   statement, where Python checks `in_transaction`.
 * - `test_only_file_paths_are_accepted`: bytes are a `Buffer`; a `URL`, a number and `FILE:` rows
 *   are added.
 * - `test_new_file_permissions_and_connection_settings`: the PRAGMAs are read through the store's
 *   private `connection`, as Python reads `store._connection`.
 * - `test_path_io_errors_are_sanitized`: the path's symlink can't be read (a spy on
 *   `fs.readlinkSync` throws a system error), where Python patches `os.path.realpath`.
 * - Child processes are `bun -e` scripts importing `src/memory/storage/sqlite.ts`, where Python
 *   spawns `multiprocessing` children; a crash is `process.exit(23)`, Python's `os._exit(23)`.
 * - Failures are injected by spies on the store's private connection (`exec` for `COMMIT` and
 *   `ROLLBACK`, `prepare` for `SELECT id`) or on `fs`, where Python subclasses
 *   `sqlite3.Connection`. `test_failed_save_poisoning_retains_ownership_and_primary`: the
 *   `sqlite3.OperationalError` and `OSError` rows are a SQLite driver-shaped error and a system
 *   error. `test_failed_constructor_rolls_back_and_releases_connection`: the failure is the size
 *   check's `fs.statSync` inside the opening transaction, where Python's is the commit; the
 *   rows are a plain `Error` and a system error, and the release shows as a reopen that isn't
 *   busy (Python executes on the closed connection).
 * - `test_close_is_idempotent_and_closed_methods_do_not_touch_os_or_sqlite`: spies on `fs` and on
 *   the old connection see no call, where Python forbids `os.getpid` and `sqlite3.connect`.
 * - `test_creation_crash_leaves_recoverable_empty_file`: the child exits from the constructor's
 *   `fs.statSync`, just before SQLite opens, where Python patches `sqlite3.connect`.
 *
 * Not ported: `test_envelope_depth_rejection_preserves_old_state_and_usable_handle` and
 * `test_maximum_valid_envelope_depth_roundtrips` (TS has no 128-level memory JSON limit; a
 * 100 000-level state round-trips instead);
 * `test_shared_connection_serializes_threads` (JS runs one thread, so there is no per-handle lock);
 * `test_postfork_guard_precedes_python_lock_and_sqlite` (JS can't fork a live process, so there is no
 * fork guard); Python's exception classes (the `KeyboardInterrupt` and `SystemExit` rows are a plain
 * `Error` and a thrown non-`Error` value, which propagate as the same value).
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { type ChildProcess, execFileSync, spawn, spawnSync } from "node:child_process";
import { once } from "node:events";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { MemoryOperationError } from "../../src/memory/contracts.ts";
import { checkStoragePath, realStoragePath, SQLiteStorage } from "../../src/memory/storage/sqlite.ts";
import { openSqlite, type Sqlite } from "../../src/platform.ts";
import { type JsonObject, logJson } from "../../src/typesys/json.ts";

/** Assert `fn` throws a storage `MemoryOperationError` with code `storage.<code>`, and return it. */
function expectStorageError(fn: () => unknown, code: string): MemoryOperationError {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(MemoryOperationError);
  const error = caught as MemoryOperationError;
  expect(error.stage).toBe("storage");
  expect(error.code).toBe(`storage.${code}`);
  expect(error.retryable).toBe(code === "busy");
  return error;
}

/** Return the store's private connection, as Python's tests read `store._connection`. */
function connectionOf(store: SQLiteStorage): Sqlite {
  return store["connection"]!;
}

/** Return the one value of `sql`'s single result row. */
function pragma(connection: Sqlite, sql: string): unknown {
  return Object.values(connection.prepare(sql).get()!)[0];
}

/** Return the permission bits of the file at `file`. */
function mode(file: string): number {
  return fs.statSync(file).mode & 0o777;
}

/** Open the store at `file`, load its state and close it. */
function reloaded(file: string): JsonObject | null {
  const store = new SQLiteStorage(file, "session");
  try {
    return store.load();
  } finally {
    store.close();
  }
}

/** Return a SQLite driver-shaped error whose message is a secret path. */
function sqliteError(): Error {
  return Object.assign(new Error("secret-host-path"), { code: "ERR_SQLITE_ERROR", errcode: 1 });
}

/** Return a system error from `syscall` whose message is a secret path. */
function systemError(syscall: string): Error {
  return Object.assign(new Error("secret-host-path"), { code: "EIO", errno: -5, syscall });
}

/** The package root. */
const ROOT = path.resolve(import.meta.dirname, "../..");
const BUILT = path.join(ROOT, "dist/src/memory/storage/sqlite.js");

/** A runtime that runs a module script: its command line before the script, and the storage module the script imports. */
interface ChildRuntime {
  argv: string[];
  module: string;
}

const BUN: ChildRuntime = {
  argv: [process.execPath, "-e"],
  module: pathToFileURL(path.join(ROOT, "src/memory/storage/sqlite.ts")).href,
};

/** Return `body` as a script for `runtime`, with `SQLiteStorage` imported and `file` bound. */
function childScript(runtime: ChildRuntime, file: string, body: string): string {
  return `import { SQLiteStorage } from ${JSON.stringify(runtime.module)};
const file = ${JSON.stringify(file)};
${body}`;
}

/** Run `body` in a child process of `runtime`; return its exit status and stdout, after checking it wrote no stderr. */
function runChild(file: string, body: string, runtime = BUN): { status: number | null; stdout: string } {
  const [command, ...args] = runtime.argv;
  const result = spawnSync(command!, [...args, childScript(runtime, file, body)], { encoding: "utf8", timeout: 10_000 });
  expect(result.stderr).toBe("");
  return { status: result.status, stdout: result.stdout };
}

/** Open `file` in a child process and return `"opened"` or the failure's stage, code and retryability. */
function attemptOpen(file: string, runtime = BUN): unknown {
  const { status, stdout } = runChild(
    file,
    `try {
  new SQLiteStorage(file, "session").close();
  console.log(JSON.stringify("opened"));
} catch (error) {
  if (!error?.stage) throw error;
  console.log(JSON.stringify([error.stage, error.code, error.retryable]));
}`,
    runtime,
  );
  expect(status).toBe(0);
  return JSON.parse(stdout);
}

/**
 * Start a child process of `runtime` that opens `file`, saves `{saved: true}`, prints "ready" and
 * then holds the store until killed; resolve once it is ready.
 */
async function startOwner(file: string, runtime = BUN): Promise<ChildProcess> {
  const [command, ...args] = runtime.argv;
  const body = `const store = new SQLiteStorage(file, "session");
store.save({ saved: true });
console.log("ready");
// A deadline, so an orphan can't hold the lock forever.
setTimeout(() => process.exit(1), 60_000);`;
  const child = spawn(command!, [...args, childScript(runtime, file, body)], { stdio: ["ignore", "pipe", "pipe"] });
  try {
    await new Promise<void>((resolve, reject) => {
      let stdout = "";
      let stderr = "";
      const timer = setTimeout(() => reject(new Error("storage child did not report within its deadline")), 10_000);
      child.stdout!.on("data", (chunk) => {
        stdout += chunk;
        if (stdout.includes("ready\n")) {
          clearTimeout(timer);
          resolve();
        }
      });
      child.stderr!.on("data", (chunk) => {
        stderr += chunk;
      });
      child.on("error", reject);
      child.on("exit", (code, signal) => {
        clearTimeout(timer);
        reject(new Error(`storage child exited with ${code ?? signal}: ${stderr}`));
      });
    });
  } catch (error) {
    await kill(child);
    throw error;
  }
  return child;
}

/** Kill `child` and wait for it to exit. */
async function kill(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const exited = once(child, "exit");
  child.kill("SIGKILL");
  await exited;
}

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-storage-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a round trip is detached and survives reopening", () => {
  const file = path.join(dir, "memory.sqlite");
  const store = new SQLiteStorage(file, "session");
  const value = { nested: [null, true, 1n, 1.5, { text: "日本語" }] as unknown[], empty: {} };
  const expected = { nested: [null, true, 1n, 1.5, { text: "日本語" }], empty: {} };
  try {
    expect(store.load()).toBeNull();
    store.save(value);
    value.nested.push("caller mutation");
    const loaded = store.load()!;
    expect(loaded).toEqual(expected);
    (loaded.nested as unknown[]).push("reader mutation");
    expect(store.load()).toEqual(expected);
    store.ensureOpen();
  } finally {
    store.close();
  }
  expect(reloaded(file)).toEqual(expected);
});

// Compared as log JSON text: Bun's `toEqual` overflows the stack at this depth, and `===` keeps a
// failure from printing the whole 600 KB text.
test("a 100 000-level state round-trips and survives reopening", () => {
  const file = path.join(dir, "memory.sqlite");
  let deep: unknown = 1n;
  for (let i = 0; i < 100_000; i++) deep = i % 2 === 0 ? [deep] : { a: deep };
  const state = { deep };
  const text = logJson(state);
  const store = new SQLiteStorage(file, "session");
  try {
    store.save(state);
    expect(logJson(store.load()) === text).toBe(true);
  } finally {
    store.close();
  }
  expect(logJson(reloaded(file)) === text).toBe(true);
});

class Opaque {}

test.each([
  ["null", null],
  ["an array", []],
  ["NaN", { x: Number.NaN }],
  ["a class instance", { x: new Opaque() }],
  ["a Map", { x: new Map() }],
  ["a symbol key", { [Symbol("key")]: "bad" }],
  ["a lone surrogate", { x: "\ud800" }],
])("an invalid candidate (%s) is rejected without poisoning", (_label, invalid) => {
  const store = new SQLiteStorage(path.join(dir, "memory.sqlite"), "session");
  try {
    store.save({ old: true });
    expectStorageError(() => store.save(invalid), "invalid_state");
    expect(store.load()).toEqual({ old: true });
    store.save({ valid: true });
  } finally {
    store.close();
  }
});

test.each([
  ["Infinity", { x: Infinity, y: -Infinity }, { x: Infinity, y: -Infinity }],
  ["negative zero", { x: -0 }, { x: 0 }],
  ["an int beyond 2^64", { x: 2n ** 70n }, { x: 2n ** 70n }],
  ["a record shaped like the infinity tag", { x: { $float: "inf" } }, { x: { $float: "inf" } }],
  ["a record shaped like the escape", { x: { $record: 1n } }, { x: { $record: 1n } }],
])("a state holding %s round-trips", (_label, state, expected) => {
  const file = path.join(dir, "memory.sqlite");
  const store = new SQLiteStorage(file, "session");
  try {
    store.save(state);
    expect(store.load()).toEqual(expected);
  } finally {
    store.close();
  }
  expect(reloaded(file)).toEqual(expected);
});

test("cycles are rejected before a transaction", () => {
  const store = new SQLiteStorage(path.join(dir, "memory.sqlite"), "session");
  const cycle: Record<string, unknown> = {};
  cycle.self = cycle;
  const exec = spyOn(connectionOf(store), "exec");
  try {
    expectStorageError(() => store.save(cycle), "invalid_state");
    expect(exec).not.toHaveBeenCalled();
    expect(store.load()).toBeNull();
  } finally {
    exec.mockRestore();
    store.close();
  }
});

test.each([
  ["empty", ""],
  [":memory:", ":memory:"],
  ["a file: URI", "file:memory.sqlite"],
  ["a FILE: URI", "FILE:memory.sqlite"],
  ["a NUL", "bad\0path"],
  ["a Buffer", Buffer.from("memory.sqlite")],
  ["a URL", new URL("file:///tmp/memory.sqlite")],
  ["null", null],
  ["a number", 3],
])("only file paths are accepted (%s)", (_label, candidate) => {
  // A regression that accepted a relative path would create it here, not under the package root.
  const previous = process.cwd();
  process.chdir(dir);
  try {
    expectStorageError(() => new SQLiteStorage(candidate, "session"), "path");
  } finally {
    process.chdir(previous);
  }
});

test.each([
  ["empty", ""],
  ["null", null],
  ["a number", 1],
  ["a lone surrogate", "\ud800"],
])("the backend kind must be a nonempty string (%s)", (_label, backend) => {
  const file = path.join(dir, "memory.sqlite");
  expectStorageError(() => new SQLiteStorage(file, backend), "backend");
  expect(fs.existsSync(file)).toBe(false);
});

test("a missing parent and a directory are rejected without disclosing the path", () => {
  const file = path.join(dir, "private-host-name", "memory.sqlite");
  let caught: unknown;
  try {
    new SQLiteStorage(file, "session");
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(MemoryOperationError);
  expect((caught as MemoryOperationError).stage).toBe("storage");
  expect(String(caught)).not.toContain("private-host-name");
  expect(fs.existsSync(path.dirname(file))).toBe(false);
  expect(() => new SQLiteStorage(dir, "session")).toThrow(MemoryOperationError);
});

test("a new file is private and the connection has the storage settings", () => {
  const file = path.join(dir, "memory.sqlite");
  const store = new SQLiteStorage(file, "session");
  try {
    expect(mode(file)).toBe(0o600);
    const connection = connectionOf(store);
    expect(pragma(connection, "PRAGMA synchronous")).toBe(3);
    expect(pragma(connection, "PRAGMA locking_mode")).toBe("exclusive");
    expect(pragma(connection, "PRAGMA journal_mode")).toBe("delete");
    expect(pragma(connection, "PRAGMA busy_timeout")).toBe(0);
  } finally {
    store.close();
  }
});

test("a new file's permissions don't depend on the umask", () => {
  const file = path.join(dir, "memory.sqlite");
  const previous = process.umask(0o777);
  let store: SQLiteStorage;
  try {
    store = new SQLiteStorage(file, "session");
  } finally {
    process.umask(previous);
  }
  try {
    expect(mode(file)).toBe(0o600);
  } finally {
    store.close();
  }
});

test("path I/O errors are sanitized", () => {
  fs.mkdirSync(path.join(dir, "target"));
  fs.symlinkSync(path.join(dir, "target"), path.join(dir, "link"));
  const readlink = spyOn(fs, "readlinkSync").mockImplementation(() => {
    throw systemError("readlink");
  });
  try {
    const error = expectStorageError(() => new SQLiteStorage(path.join(dir, "link", "memory.sqlite"), "session"), "io");
    expect(String(error)).not.toContain("secret-host-path");
  } finally {
    readlink.mockRestore();
  }
  expect(fs.existsSync(path.join(dir, "target", "memory.sqlite"))).toBe(false);
});

test("a zero-byte file initializes without changing its permissions", () => {
  const file = path.join(dir, "empty.sqlite");
  fs.writeFileSync(file, "");
  fs.chmodSync(file, 0o640);
  const store = new SQLiteStorage(file, "session");
  try {
    expect(store.load()).toBeNull();
    expect(mode(file)).toBe(0o640);
  } finally {
    store.close();
  }
});

test.each([
  ["kind", "kind"],
  ["version", "version"],
  ["python_version", "version"],
  ["frame_era_version", "version"],
  ["boolean_version", "version"],
  ["extra_field", "corrupt"],
  ["missing_field", "corrupt"],
  ["duplicate_key", "corrupt"],
  ["bad_json", "corrupt"],
  ["nonfinite", "corrupt"],
  ["not_object", "corrupt"],
  ["no_row", "corrupt"],
  ["extra_table", "schema"],
  ["altered_schema", "schema"],
  ["blob", "corrupt"],
])("an invalid store (%s) is rejected without overwriting it or leaking the owner", (damage, code) => {
  const file = path.join(dir, "memory.sqlite");
  const store = new SQLiteStorage(file, "session");
  try {
    store.save({ old: true });
  } finally {
    store.close();
  }
  const connection = openSqlite(file);
  try {
    let raw: string | Uint8Array = connection.prepare("SELECT envelope FROM memory_state").get()!.envelope as string;
    const envelope = JSON.parse(raw);
    if (damage === "kind") envelope.backend = "episodic";
    else if (damage === "version") envelope.version = 4;
    else if (damage === "python_version") envelope.version = 2;
    else if (damage === "frame_era_version") envelope.version = 1;
    else if (damage === "boolean_version") envelope.version = true;
    else if (damage === "extra_field") envelope.extra = true;
    else if (damage === "missing_field") delete envelope.state;
    else if (damage === "not_object") envelope.state = [];
    raw = JSON.stringify(envelope);
    if (damage === "duplicate_key") raw = `${raw.slice(0, -1)}, "state": {}}`;
    else if (damage === "bad_json") raw = "{";
    else if (damage === "nonfinite") raw = raw.replace('"old":true', '"old":1e999');
    else if (damage === "blob") raw = new TextEncoder().encode(raw);
    connection.prepare("UPDATE memory_state SET envelope = ?").run(raw);
    if (damage === "no_row") connection.exec("DELETE FROM memory_state");
    else if (damage === "extra_table") connection.exec("CREATE TABLE unrelated(value)");
    else if (damage === "altered_schema") connection.exec("ALTER TABLE memory_state ADD COLUMN extra TEXT");
  } finally {
    connection.close();
  }
  const before = fs.readFileSync(file);
  for (let i = 0; i < 2; i++) {
    expectStorageError(() => new SQLiteStorage(file, "session"), code);
    expect(fs.readFileSync(file)).toEqual(before);
  }
});

test.each(["table", "empty_database", "corrupt"])("an unrelated file (%s) is preserved", (unrelated) => {
  const file = path.join(dir, "memory.sqlite");
  if (unrelated === "corrupt") {
    fs.writeFileSync(file, "caller data, not sqlite");
  } else {
    const connection = openSqlite(file);
    connection.exec("CREATE TABLE unrelated(value)");
    if (unrelated === "empty_database") connection.exec("DROP TABLE unrelated");
    connection.close();
  }
  const before = fs.readFileSync(file);
  expect(() => new SQLiteStorage(file, "session")).toThrow(MemoryOperationError);
  expect(fs.readFileSync(file)).toEqual(before);
});

test.each([
  ["the same path", false],
  ["a symlink alias", true],
])(
  "the lifetime owner survives commits and failed competitors (%s)",
  (_label, alias) => {
    const file = path.join(dir, "memory.sqlite");
    const contender = alias ? path.join(dir, "alias.sqlite") : file;
    const store = new SQLiteStorage(file, "session");
    if (alias) fs.symlinkSync(file, contender);
    try {
      for (let value = 0; value < 3; value++) {
        store.save({ value });
        expectStorageError(() => new SQLiteStorage(contender, "session"), "busy");
        expect(attemptOpen(contender)).toEqual(["storage", "storage.busy", true]);
        expect(store.load()).toEqual({ value });
      }
    } finally {
      store.close();
    }
    expect(attemptOpen(contender)).toBe("opened");
  },
  30_000,
);

const SAVE_FAILURES: [string, unknown, boolean][] = [
  ["a SQLite error", sqliteError(), true],
  ["a system error", systemError("write"), true],
  ["a plain Error", new Error("stop"), false],
  ["a non-Error value", { thrown: 7 }, false],
];

test.each(
  SAVE_FAILURES.flatMap(([label, failure, sanitized]) =>
    (["before", "after"] as const).map((cut) => [label, cut, failure, sanitized] as const),
  ),
)("a failed save (%s, %s the commit) poisons the handle and keeps its owner and primary failure", (_label, cut, failure, sanitized) => {
  const file = path.join(dir, "memory.sqlite");
  const store = new SQLiteStorage(file, "session");
  store.save({ old: true });
  const connection = connectionOf(store);
  const realExec = connection.exec;
  const exec = spyOn(connection, "exec").mockImplementation((sql) => {
    if (sql !== "COMMIT") return Reflect.apply(realExec, connection, [sql]);
    if (cut === "after") Reflect.apply(realExec, connection, [sql]);
    throw failure;
  });
  try {
    let caught: unknown;
    try {
      store.save({ new: true });
    } catch (error) {
      caught = error;
    }
    if (sanitized) {
      expect(caught).toBeInstanceOf(MemoryOperationError);
      expect((caught as MemoryOperationError).code).toBe("storage.io");
      expect((caught as MemoryOperationError).retryable).toBe(false);
      expect(String(caught)).not.toContain("secret-host-path");
    } else {
      expect(caught).toBe(failure);
    }
    for (const operation of [() => store.ensureOpen(), () => store.load(), () => store.save({})]) {
      expectStorageError(operation, "poisoned");
    }
    expectStorageError(() => new SQLiteStorage(file, "session"), "busy");
  } finally {
    exec.mockRestore();
    store.close();
  }
  expect(reloaded(file)).toEqual(cut === "after" ? { new: true } : { old: true });
});

test("close is idempotent and a closed handle's methods touch neither the OS nor SQLite", () => {
  const store = new SQLiteStorage(path.join(dir, "memory.sqlite"), "session");
  const connection = connectionOf(store);
  store.close();
  const fsNames = ["openSync", "statSync", "lstatSync", "readlinkSync", "mkdirSync", "fchmodSync", "closeSync"] as const;
  const spies = [
    ...fsNames.map((name) => spyOn(fs, name)),
    ...(["exec", "prepare", "close"] as const).map((name) => spyOn(connection, name)),
  ];
  try {
    store.close();
    for (const operation of [() => store.ensureOpen(), () => store.load(), () => store.save({})]) {
      expectStorageError(operation, "closed");
    }
    for (const spy of spies) expect(spy).not.toHaveBeenCalled();
  } finally {
    for (const spy of spies) spy.mockRestore();
  }
});

test.each([
  ["a plain Error", new Error("stop"), false],
  ["a system error", systemError("stat"), true],
])("a failed constructor (%s) rolls back and releases its connection", (_label, failure, sanitized) => {
  const file = path.join(dir, "memory.sqlite");
  const realStat = fs.statSync;
  let calls = 0;
  // The second stat, the size check, runs inside the constructor's exclusive transaction.
  const stat = spyOn(fs, "statSync").mockImplementation(((...args: unknown[]) => {
    if (++calls === 2) throw failure;
    return Reflect.apply(realStat, fs, args);
  }) as typeof fs.statSync);
  let caught: unknown;
  try {
    new SQLiteStorage(file, "session");
  } catch (error) {
    caught = error;
  } finally {
    stat.mockRestore();
  }
  expect(calls).toBe(2);
  if (sanitized) {
    expect(caught).toBeInstanceOf(MemoryOperationError);
    expect((caught as MemoryOperationError).code).toBe("storage.io");
    expect((caught as MemoryOperationError).retryable).toBe(false);
    expect(String(caught)).not.toContain("secret-host-path");
  } else {
    expect(caught).toBe(failure);
  }
  expect(reloaded(file)).toBeNull();
});

test("a rollback failure replaces neither the save failure nor the owner", () => {
  const file = path.join(dir, "memory.sqlite");
  const store = new SQLiteStorage(file, "session");
  store.save({ old: true });
  const primary = new Error("original");
  const connection = connectionOf(store);
  const realExec = connection.exec;
  const exec = spyOn(connection, "exec").mockImplementation((sql) => {
    if (sql === "COMMIT") throw primary;
    if (sql === "ROLLBACK") throw new Error("cleanup");
    return Reflect.apply(realExec, connection, [sql]);
  });
  try {
    expect(() => store.save({ new: true })).toThrow(primary);
    expect(attemptOpen(file)).toEqual(["storage", "storage.busy", true]);
  } finally {
    exec.mockRestore();
    store.close();
  }
  expect(reloaded(file)).toEqual({ old: true });
});

test("a load I/O failure poisons the handle and keeps its owner", () => {
  const file = path.join(dir, "memory.sqlite");
  const store = new SQLiteStorage(file, "session");
  const connection = connectionOf(store);
  const realPrepare = connection.prepare;
  const prepare = spyOn(connection, "prepare").mockImplementation((sql) => {
    if (sql.startsWith("SELECT id")) {
      throw sqliteError();
    }
    return Reflect.apply(realPrepare, connection, [sql]);
  });
  try {
    const error = expectStorageError(() => store.load(), "io");
    expect(String(error)).not.toContain("secret-host-path");
    expectStorageError(() => store.ensureOpen(), "poisoned");
    expect(attemptOpen(file)).toEqual(["storage", "storage.busy", true]);
  } finally {
    prepare.mockRestore();
    store.close();
  }
});

test.each(["before", "after"])(
  "a process crash %s the commit is atomic",
  (cut) => {
    const file = path.join(dir, "memory.sqlite");
    const store = new SQLiteStorage(file, "session");
    try {
      store.save({ value: "old" });
    } finally {
      store.close();
    }
    const large = "x".repeat(100_000);
    const { status } = runChild(
      file,
      `const store = new SQLiteStorage(file, "session");
const cut = ${JSON.stringify(cut)};
const connection = store["connection"];
const exec = connection.exec;
connection.exec = (sql) => {
  if (sql === "COMMIT" && cut === "before") process.exit(23);
  exec.call(connection, sql);
  if (sql === "COMMIT" && cut === "after") process.exit(23);
};
store.save({ value: "new", large: "x".repeat(${large.length}) });`,
    );
    expect(status).toBe(23);
    expect(reloaded(file)).toEqual(cut === "before" ? { value: "old" } : { value: "new", large });
  },
  30_000,
);

test(
  "a creation crash leaves a recoverable empty file",
  () => {
    const file = path.join(dir, "memory.sqlite");
    // The constructor stats the new file just before it opens SQLite.
    const { status } = runChild(
      file,
      `const fs = (await import("node:fs")).default;
fs.statSync = () => process.exit(23);
new SQLiteStorage(file, "session");`,
    );
    expect(status).toBe(23);
    expect(fs.readFileSync(file)).toEqual(Buffer.alloc(0));
    const store = new SQLiteStorage(file, "session");
    try {
      expect(store.load()).toBeNull();
      store.save({ recovered: true });
    } finally {
      store.close();
    }
  },
  30_000,
);

test(
  "process death releases the owner and preserves its commit",
  async () => {
    const file = path.join(dir, "memory.sqlite");
    const child = await startOwner(file);
    try {
      expectStorageError(() => new SQLiteStorage(file, "session"), "busy");
    } finally {
      await kill(child);
    }
    expect(child.signalCode).toBe("SIGKILL");
    expect(reloaded(file)).toEqual({ saved: true });
  },
  30_000,
);

test("checkStoragePath returns an absolute path without I/O", () => {
  const previous = process.cwd();
  process.chdir(dir);
  try {
    expect(checkStoragePath("m.sqlite")).toBe(`${process.cwd()}/m.sqlite`);
    expect(checkStoragePath("a/../m.sqlite")).toBe(`${process.cwd()}/a/../m.sqlite`);
    expect(checkStoragePath(path.join(dir, "x", "m.sqlite"))).toBe(path.join(dir, "x", "m.sqlite"));
    expect(fs.readdirSync(dir)).toEqual([]);
    for (const bad of ["", ":memory:", "FILE:m.sqlite", "a\0b", Buffer.from("m.sqlite"), 3]) {
      expectStorageError(() => checkStoragePath(bad), "path");
    }
    process.chdir("/");
    expect(checkStoragePath("m.sqlite")).toBe("/m.sqlite");
  } finally {
    process.chdir(previous);
  }
});

test("the storage path follows symlinks before parent steps", () => {
  fs.mkdirSync(path.join(dir, "target", "sub"), { recursive: true });
  fs.symlinkSync(path.join(dir, "target", "sub"), path.join(dir, "link"));
  const raw = `${dir}/link/../m.sqlite`;
  expect(checkStoragePath(raw)).toBe(raw);
  const store = new SQLiteStorage(raw, "session");
  try {
    store.save({ ok: true });
  } finally {
    store.close();
  }
  expect(fs.statSync(path.join(dir, "target", "m.sqlite")).isFile()).toBe(true);
  expect(fs.existsSync(path.join(dir, "m.sqlite"))).toBe(false);
});

test("createParent makes missing directories", () => {
  const file = path.join(dir, "a", "b", "m.sqlite");
  expectStorageError(() => new SQLiteStorage(file, "episodic"), "io");
  expect(fs.existsSync(path.join(dir, "a"))).toBe(false);
  new SQLiteStorage(file, "episodic", { createParent: true }).close();
  expect(fs.statSync(file).isFile()).toBe(true);
});

test("realStoragePath resolves like Python's non-strict realpath", () => {
  const real = fs.realpathSync(dir);
  fs.mkdirSync(path.join(dir, "target"));
  fs.mkdirSync(path.join(dir, "sub"));
  fs.symlinkSync(path.join(dir, "target"), path.join(dir, "link"));
  fs.symlinkSync(path.join(dir, "nowhere", "file"), path.join(dir, "dangling"));
  fs.symlinkSync("loop2", path.join(dir, "loop1"));
  fs.symlinkSync("loop1", path.join(dir, "loop2"));
  fs.symlinkSync("../target", path.join(dir, "sub", "rel"));
  // A missing component and its `..` are applied lexically, then later links still resolve.
  expect(realStoragePath(`${dir}/missing/../link/x`)).toBe(`${real}/target/x`);
  expect(realStoragePath(`${dir}/missing/x/../../link/x`)).toBe(`${real}/target/x`);
  expect(realStoragePath(`${dir}/dangling`)).toBe(`${real}/nowhere/file`);
  // A loop leaves the looping link joined with the unresolved rest.
  expect(realStoragePath(`${dir}/loop1/x`)).toBe(`${real}/loop1/x`);
  expect(realStoragePath(`${dir}/sub/rel/m`)).toBe(`${real}/target/m`);
  expect(realStoragePath(`${dir}//./link/`)).toBe(`${real}/target`);
});

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/engine/platform.test.ts`'s: importing a test file would run its tests.
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

describe.skipIf(!NODE)("the built storage on Node", () => {
  const node: ChildRuntime = { argv: [NODE!, "--input-type=module", "-e"], module: pathToFileURL(BUILT).href };

  test(
    "a round trip, a reopen and damaged stores",
    () => {
      const platform = JSON.stringify(pathToFileURL(path.join(ROOT, "dist/src/platform.js")).href);
      const { status, stdout } = runChild(
        path.join(dir, "memory.sqlite"),
        String.raw`
const assert = (await import("node:assert/strict")).default;
const { writeFileSync } = await import("node:fs");
const { openSqlite } = await import(${platform});
const code = (fn) => {
  try {
    fn();
  } catch (error) {
    return error.code;
  }
  assert.fail("expected a throw");
};
const state = { inf: Infinity, ninf: -Infinity, big: 2n ** 70n, text: "日本語" };
const store = new SQLiteStorage(file, "session");
try {
  assert.equal(store.load(), null);
  store.save(state);
  assert.deepEqual(store.load(), state);
  assert.equal(code(() => new SQLiteStorage(file, "session")), "storage.busy");
} finally {
  store.close();
}
const reopened = new SQLiteStorage(file, "session");
try {
  assert.deepEqual(reopened.load(), state);
} finally {
  reopened.close();
}
assert.equal(code(() => new SQLiteStorage(file, "episodic")), "storage.kind");
const connection = openSqlite(file);
connection.prepare("UPDATE memory_state SET envelope = ?").run("{");
connection.close();
assert.equal(code(() => new SQLiteStorage(file, "session")), "storage.corrupt");
const unrelated = file + ".other";
writeFileSync(unrelated, "caller data, not sqlite");
assert.equal(code(() => new SQLiteStorage(unrelated, "session")), "storage.corrupt");
console.log("ok");
`,
        node,
      );
      expect(stdout).toBe("ok\n");
      expect(status).toBe(0);
    },
    30_000,
  );

  test(
    "a Bun owner excludes a Node competitor, and a Node owner a Bun one",
    async () => {
      const file = path.join(dir, "memory.sqlite");
      const store = new SQLiteStorage(file, "session");
      try {
        expect(attemptOpen(file, node)).toEqual(["storage", "storage.busy", true]);
      } finally {
        store.close();
      }
      const child = await startOwner(file, node);
      try {
        expectStorageError(() => new SQLiteStorage(file, "session"), "busy");
      } finally {
        await kill(child);
      }
      expect(attemptOpen(file, node)).toBe("opened");
    },
    30_000,
  );
});
