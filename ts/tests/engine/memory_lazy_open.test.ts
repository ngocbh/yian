/**
 * Built-in SQLite backends open storage on first use, never in the constructor: a port of
 * `tests/engine/test_memory_lazy_open.py`, over `EpisodicMemory`, `ReflectiveMemory` and
 * `SessionMemory`.
 *
 * Ported: `test_construction_touches_no_file`, `test_bad_path_shape_fails_at_construction_without_io`,
 * `test_reset_also_opens_storage`, `test_missing_parent_fails_at_first_open_without_create_parent`,
 * `test_create_parent_makes_the_directory_at_first_open`,
 * `test_close_before_open_does_no_io_and_blocks_later_use`, `test_failed_open_can_be_retried` and
 * `test_failed_load_can_be_retried`.
 *
 * TS-only: none.
 *
 * Changed: the `reflective` backend gets an environment map standing in for Python's
 * `conftest.py` variables, beside Python's `_REFLECTIVE_LLM`.
 *
 * Not ported: `test_pid_ownership_starts_at_construction` and `test_ram_backend_ignores_pid` (JS
 * can't fork a live process, so there is no fork guard); `test_concurrent_first_open_constructs_storage_once`
 * (JS runs one thread, and the module's `SQLiteStorage` binding can't be patched);
 * `test_entering_the_backend_opens_storage` (there is no `with` form).
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { BUILTIN, type Env, install } from "../../src/defaults.ts";
import { MemoryOperationError } from "../../src/memory/contracts.ts";
import { EpisodicMemory } from "../../src/memory/episodic.ts";
import { ReflectiveMemory } from "../../src/memory/reflective.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { SQLiteStorage } from "../../src/memory/storage/sqlite.ts";

const REFLECTIVE_LLM = { provider: "ollama", model: "current" };

/** The surface every backend shares. */
interface Backend {
  openRun(): { close(): void };
  reset(): void;
  close(): void;
}

/** A backend's name (its closed code's prefix) and its factory over a storage path. */
const BACKENDS: [string, (file: string, options?: { createParent?: boolean }) => Backend][] = [
  ["episodic", (file, options = {}) => new EpisodicMemory({ storage: file, ...options })],
  [
    "reflective",
    (file, options = {}) =>
      new ReflectiveMemory({ llmConfig: REFLECTIVE_LLM, storage: file, env: baseEnv(), ...options }),
  ],
  ["session", (file, options = {}) => new SessionMemory({ storage: file, ...options })],
];

let dir: string;
// Apart from `dir`, whose listing the tests check.
let home: string;

beforeEach(() => {
  install(BUILTIN);
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-lazy-"));
  home = fs.mkdtempSync(path.join(os.tmpdir(), "yian-lazy-home-"));
});

afterEach(() => {
  install(BUILTIN);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.rmSync(home, { recursive: true, force: true });
});

/** Python's `conftest.py` environment: an empty private store and an explicit model default. */
function baseEnv(): Env {
  const vars = new Map([
    ["YIAN_AUTH_FILE", path.join(home, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
  ]);
  return (name) => vars.get(name);
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

describe.each(BACKENDS)("%s", (name, make) => {
  test("construction touches no file", () => {
    const file = path.join(dir, "m.sqlite");
    const memory = make(file);
    expect(fs.readdirSync(dir)).toEqual([]);
    memory.openRun().close();
    expect(fs.existsSync(file)).toBe(true);
    memory.close();
  });

  test.each(["", ":memory:", "file:m.sqlite", "a\0b"])(
    "bad path shape fails at construction without IO (%p)",
    (bad) => {
      const previous = process.cwd();
      process.chdir(dir);
      try {
        expect(failure(() => make(bad)).code).toBe("storage.path");
        expect(fs.readdirSync(dir)).toEqual([]);
      } finally {
        process.chdir(previous);
      }
    },
  );

  test("reset also opens storage", () => {
    const file = path.join(dir, "m.sqlite");
    const memory = make(file);
    memory.reset();
    expect(fs.existsSync(file)).toBe(true);
    memory.close();
  });

  test("missing parent fails at first open without create parent", () => {
    const memory = make(path.join(dir, "missing", "m.sqlite"));
    expect(failure(() => memory.openRun()).code).toBe("storage.io");
    expect(fs.existsSync(path.join(dir, "missing"))).toBe(false);
    memory.close();
  });

  test("create parent makes the directory at first open", () => {
    const file = path.join(dir, "a", "b", "m.sqlite");
    const memory = make(file, { createParent: true });
    expect(fs.existsSync(path.join(dir, "a"))).toBe(false);
    memory.openRun().close();
    expect(fs.existsSync(file)).toBe(true);
    memory.close();
  });

  test("close before open does no IO and blocks later use", () => {
    const memory = make(path.join(dir, "m.sqlite"));
    memory.close();
    for (const operation of [() => memory.openRun(), () => memory.reset()]) {
      expect(failure(operation).code).toBe(`${name}.closed`);
    }
    expect(fs.readdirSync(dir)).toEqual([]);
  });

  test("failed open can be retried", () => {
    const holder = make(path.join(dir, "m.sqlite"));
    // Takes the exclusive file lock.
    holder.openRun().close();
    const waiting = make(path.join(dir, "m.sqlite"));
    expect(failure(() => waiting.openRun()).code).toBe("storage.busy");
    holder.close();
    waiting.openRun().close();
    waiting.close();
  });

  test("failed load can be retried", () => {
    const realLoad = SQLiteStorage.prototype.load;
    let calls = 0;
    const load = spyOn(SQLiteStorage.prototype, "load").mockImplementation(function (this: SQLiteStorage) {
      calls += 1;
      if (calls === 1) throw new MemoryOperationError("storage", "storage.io", "transient load failure", true);
      return Reflect.apply(realLoad, this, []);
    });
    try {
      const memory = make(path.join(dir, "m.sqlite"));
      expect(failure(() => memory.openRun()).code).toBe("storage.io");
      memory.openRun().close();
      expect(calls).toBe(2);
      memory.close();
    } finally {
      load.mockRestore();
    }
  });
});
