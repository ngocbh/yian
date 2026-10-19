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
 * `test_check_storage_path_returns_an_absolute_path_without_io`,
 * `test_storage_path_follows_symlinks_before_parent_steps` and
 * `test_create_parent_makes_missing_directories`.
 *
 * TS-only: the value number rule in a stored state (`±Infinity` round-trips, `-0` reads `0`), an
 * int beyond 2^64, records shaped like the infinity tag or its escape; `checkStoragePath` keeping
 * `..` in a relative path and writing no `//` when the current directory is `/`; and
 * `realStoragePath`, Python's non-strict `os.path.realpath` ported line for line (a symlink
 * followed after a missing component and its `..`, a dangling link, a loop, a relative link target).
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
 *
 * Not ported yet (they follow with the ownership tests):
 * `test_lifetime_owner_survives_commits_and_failed_competitors`,
 * `test_failed_save_poisoning_retains_ownership_and_primary`,
 * `test_close_is_idempotent_and_closed_methods_do_not_touch_os_or_sqlite`,
 * `test_failed_constructor_rolls_back_and_releases_connection`,
 * `test_rollback_failure_does_not_replace_save_failure_or_release_owner`,
 * `test_load_io_failure_poisoning_preserves_owner`, `test_process_crash_at_commit_is_atomic`,
 * `test_creation_crash_leaves_recoverable_empty_file` and
 * `test_process_death_releases_owner_and_preserves_commit`.
 *
 * Not ported: `test_envelope_depth_rejection_preserves_old_state_and_usable_handle` and
 * `test_maximum_valid_envelope_depth_roundtrips` (TS has no memory JSON nesting limit);
 * `test_shared_connection_serializes_threads` (JS runs one thread, so there is no per-handle lock);
 * `test_postfork_guard_precedes_python_lock_and_sqlite` (JS can't fork a live process, so there is no
 * fork guard).
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { MemoryOperationError } from "../../src/memory/contracts.ts";
import { checkStoragePath, realStoragePath, SQLiteStorage } from "../../src/memory/storage/sqlite.ts";
import { openSqlite, type Sqlite } from "../../src/platform.ts";

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
  return (store as unknown as { connection: Sqlite }).connection;
}

/** Return the one value of `sql`'s single result row. */
function pragma(connection: Sqlite, sql: string): unknown {
  return Object.values(connection.prepare(sql).get()!);
}

/** Return the permission bits of the file at `file`. */
function mode(file: string): number {
  return fs.statSync(file).mode & 0o777;
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
  const reopened = new SQLiteStorage(file, "session");
  try {
    expect(reopened.load()).toEqual(expected);
  } finally {
    reopened.close();
  }
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
    if ("x" in expected && expected.x === 0) expect(Object.is(store.load()!.x, 0)).toBe(true);
  } finally {
    store.close();
  }
  const reopened = new SQLiteStorage(file, "session");
  try {
    expect(reopened.load()).toEqual(expected);
  } finally {
    reopened.close();
  }
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
  expectStorageError(() => new SQLiteStorage(candidate, "session"), "path");
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
    expect(pragma(connection, "PRAGMA synchronous")).toEqual([3]);
    expect(pragma(connection, "PRAGMA locking_mode")).toEqual(["exclusive"]);
    expect(pragma(connection, "PRAGMA journal_mode")).toEqual(["delete"]);
    expect(pragma(connection, "PRAGMA busy_timeout")).toEqual([0]);
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
    throw Object.assign(new Error("secret-host-path"), { code: "EIO", errno: -5, syscall: "readlink" });
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
  store.save({ old: true });
  store.close();
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
