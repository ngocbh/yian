/**
 * Contracts for `src/platform.ts`: runtime detection; the SQLite wrapper that
 * `src/yian/memory/storage/sqlite.py`'s TypeScript port runs on, with the result
 * codes Python reads from `sqlite_errorcode`; and the exclusive file lock that
 * replaces the `flock` calls in `runtime/log.py` and `llm_clients/auth/store.py`.
 */

import { afterEach, beforeEach, describe, expect, mock, spyOn, test } from "bun:test";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  type FileLock,
  FileLockError,
  isLocked,
  lockFile,
  openSqlite,
  runtime,
  type Sqlite,
  sqliteErrorCode,
} from "../../src/platform.ts";

let dir: string;
let file: string;
let opened: Sqlite[];
let held: FileLock[];

/** Open `p` and close it after the test. */
function open(p: string, options?: { readonly?: boolean }): Sqlite {
  const db = openSqlite(p, options);
  opened.push(db);
  return db;
}

/** Lock `p` and release it after the test. */
function hold(p: string, options?: { private?: boolean }): FileLock {
  const lock = lockFile(p, options);
  held.push(lock);
  return lock;
}

/** Return what `f` throws; fail if it returns. */
function thrown(f: () => unknown): unknown {
  try {
    f();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-platform-"));
  file = path.join(dir, "db.sqlite");
  opened = [];
  held = [];
});

afterEach(() => {
  mock.restore();
  try {
    for (const lock of held) lock.release();
    for (const db of opened) db.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

describe("runtime", () => {
  test("is bun under bun test", () => {
    expect(runtime).toBe("bun");
  });
});

describe("openSqlite", () => {
  test("creates the file", () => {
    open(file);
    expect(fs.existsSync(file)).toBe(true);
  });

  test("exec runs several statements", () => {
    const db = open(file);
    db.exec("CREATE TABLE t (a INTEGER, b TEXT); CREATE TABLE u (c BLOB)");
    expect(db.prepare("SELECT name FROM sqlite_master ORDER BY name").all()).toEqual([{ name: "t" }, { name: "u" }]);
  });

  test("get returns a plain row, or undefined for no row", () => {
    const db = open(file);
    db.exec("CREATE TABLE t (a INTEGER, b TEXT)");
    db.prepare("INSERT INTO t VALUES (?, ?)").run(1, "x");
    const row = db.prepare("SELECT a, b FROM t").get();
    expect(row).toEqual({ a: 1, b: "x" });
    expect(Object.getPrototypeOf(row)).toBe(Object.prototype);
    expect(db.prepare("SELECT a FROM t WHERE a = ?").get(2)).toBeUndefined();
  });

  test("all returns rows in insertion order", () => {
    const db = open(file);
    db.exec("CREATE TABLE t (a INTEGER, b TEXT)");
    const insert = db.prepare("INSERT INTO t VALUES (?, ?)");
    insert.run(3, "c");
    insert.run(1, "a");
    insert.run(2, "b");
    const rows = db.prepare("SELECT a, b FROM t").all();
    expect(rows).toEqual([
      { a: 3, b: "c" },
      { a: 1, b: "a" },
      { a: 2, b: "b" },
    ]);
    for (const row of rows) expect(Object.getPrototypeOf(row)).toBe(Object.prototype);
  });

  test("null and a Uint8Array round-trip", () => {
    const db = open(file);
    db.exec("CREATE TABLE u (c BLOB, d TEXT)");
    db.prepare("INSERT INTO u VALUES (?, ?)").run(new Uint8Array([0, 1, 255]), null);
    const row = db.prepare("SELECT c, d FROM u").get()!;
    expect(row.c).toBeInstanceOf(Uint8Array);
    expect([...(row.c as Uint8Array)]).toEqual([0, 1, 255]);
    expect(row.d).toBeNull();
  });

  test("integers read back as safe numbers, else throw RangeError", () => {
    const db = open(file);
    db.exec("CREATE TABLE t (a INTEGER)");
    const insert = db.prepare("INSERT INTO t VALUES (?)");
    const read = db.prepare("SELECT a FROM t");
    const clear = db.prepare("DELETE FROM t");
    insert.run(5n);
    expect(read.get()).toEqual({ a: 5 });
    for (const big of [2n ** 60n, 2n ** 53n, -(2n ** 53n)]) {
      clear.run();
      insert.run(big);
      expect(() => read.get()).toThrow(RangeError);
      expect(() => read.all()).toThrow(RangeError);
    }
    clear.run();
    insert.run(2n ** 53n - 1n);
    expect(read.get()).toEqual({ a: Number.MAX_SAFE_INTEGER });
  });

  test("preparing the same SQL twice returns the cached statement", () => {
    const db = open(file);
    expect(db.prepare("SELECT 1 AS one")).toBe(db.prepare("SELECT 1 AS one"));
  });

  test("close keeps committed data, and a second close is a no-op", () => {
    const db = open(file);
    db.exec("CREATE TABLE t (a INTEGER)");
    db.prepare("INSERT INTO t VALUES (?)").run(7);
    db.close();
    db.close();
    expect(open(file).prepare("SELECT a FROM t").all()).toEqual([{ a: 7 }]);
  });

  test("readonly reads and refuses writes", () => {
    const db = open(file);
    db.exec("CREATE TABLE t (a INTEGER)");
    db.prepare("INSERT INTO t VALUES (?)").run(1);
    db.close();
    const ro = open(file, { readonly: true });
    expect(ro.prepare("SELECT a FROM t").all()).toEqual([{ a: 1 }]);
    expect(() => ro.exec("INSERT INTO t VALUES (2)")).toThrow();
    expect(() => ro.prepare("INSERT INTO t VALUES (?)").run(2)).toThrow();
  });

  test("readonly on a missing file throws CANTOPEN and creates nothing", () => {
    const error = thrown(() => open(file, { readonly: true }));
    expect(sqliteErrorCode(error)).toBe(14);
    expect(fs.existsSync(file)).toBe(false);
  });

  test("close releases a held exclusive state", () => {
    const a = open(file);
    a.exec("PRAGMA busy_timeout = 0");
    a.exec("PRAGMA locking_mode = EXCLUSIVE");
    a.exec("CREATE TABLE t (a INTEGER)");
    a.exec("BEGIN EXCLUSIVE");
    a.prepare("INSERT INTO t VALUES (?)").run(1);
    a.exec("COMMIT");
    const kept = a.prepare("SELECT a FROM t");
    expect(kept.all()).toEqual([{ a: 1 }]);
    a.close();

    const b = open(file);
    b.exec("PRAGMA busy_timeout = 0");
    b.exec("BEGIN EXCLUSIVE");
    b.exec("COMMIT");
    expect(() => kept.all()).toThrow();
  });
});

describe("sqliteErrorCode", () => {
  test("a contended BEGIN EXCLUSIVE gives BUSY", () => {
    const a = open(file);
    const b = open(file);
    a.exec("PRAGMA busy_timeout = 0");
    b.exec("PRAGMA busy_timeout = 0");
    a.exec("BEGIN EXCLUSIVE");
    const error = thrown(() => b.exec("BEGIN EXCLUSIVE"));
    expect(sqliteErrorCode(error)).toBe(5);
    a.exec("COMMIT");
  });

  test("non-SQLite bytes give NOTADB", () => {
    fs.writeFileSync(file, "caller data, not sqlite");
    const db = open(file);
    const error = thrown(() => db.prepare("SELECT * FROM sqlite_master").all());
    expect(sqliteErrorCode(error)).toBe(26);
  });

  test("anything else gives undefined", () => {
    const enoent = thrown(() => fs.statSync(path.join(dir, "missing")));
    expect((enoent as NodeJS.ErrnoException).code).toBe("ENOENT");
    expect(sqliteErrorCode(enoent)).toBeUndefined();
    expect(sqliteErrorCode(new Error("x"))).toBeUndefined();
    expect(sqliteErrorCode("x")).toBeUndefined();
    expect(sqliteErrorCode(null)).toBeUndefined();
  });
});

/** Return what `f` throws, checking it's a `FileLockError` for `p`. */
function lockError(f: () => unknown, p: string): FileLockError {
  const error = thrown(f);
  expect(error).toBeInstanceOf(FileLockError);
  expect((error as FileLockError).path).toBe(p);
  return error as FileLockError;
}

/** Make `fs.lstatSync` run `replace` for the `nth` (1-based) call on `p`, the real call otherwise. */
function spyLstat(p: string, nth: number, replace: (real: typeof fs.lstatSync, ...args: unknown[]) => unknown) {
  const real = fs.lstatSync;
  let calls = 0;
  return spyOn(fs, "lstatSync").mockImplementation(((target: fs.PathLike, ...rest: unknown[]) => {
    if (target === p && ++calls === nth) return replace(real, target, ...rest);
    return Reflect.apply(real, fs, [target, ...rest]);
  }) as typeof fs.lstatSync);
}

describe("lockFile", () => {
  test("creates an empty 0600 file and holds it until release", () => {
    const lock = hold(file);
    expect(lock.path).toBe(file);
    const info = fs.statSync(file);
    expect(info.mode & 0o777).toBe(0o600);
    expect(info.size).toBe(0);
    expect(isLocked(file)).toBe(true);
    expect(fs.readdirSync(dir)).toEqual(["db.sqlite"]);
    lock.release();
    expect(isLocked(file)).toBe(false);
    lock.release();
    expect(isLocked(file)).toBe(false);
    expect(fs.statSync(file).size).toBe(0);
    expect(fs.readdirSync(dir)).toEqual(["db.sqlite"]);
  });

  test("a using block releases at its end", () => {
    {
      using lock = lockFile(file);
      expect(lock.path).toBe(file);
      expect(isLocked(file)).toBe(true);
    }
    expect(isLocked(file)).toBe(false);
  });

  test("a held lock makes another attempt busy, until released", () => {
    const lock = hold(file);
    const error = lockError(() => lockFile(file), file);
    expect(error.reason).toBe("busy");
    expect(error.name).toBe("FileLockError");
    lock.release();
    hold(file);
  });

  test("busy attempts close their connections", () => {
    const free = fs.readdirSync("/dev/fd").length;
    const lock = hold(file);
    // SQLite parks the first closed descriptor until the holder releases, since closing
    // it would drop the process's POSIX lock, and reuses it for later attempts.
    expect(lockError(() => lockFile(file), file).reason).toBe("busy");
    const before = fs.readdirSync("/dev/fd").length;
    for (let i = 0; i < 50; i++) expect(lockError(() => lockFile(file), file).reason).toBe("busy");
    expect(fs.readdirSync("/dev/fd").length).toBe(before);
    expect(isLocked(file)).toBe(true);
    lock.release();
    expect(fs.readdirSync("/dev/fd").length).toBe(free);
  });

  test("neither a busy attempt nor the probe opens the file through fs", () => {
    hold(file);
    const openSync = spyOn(fs, "openSync");
    expect(lockError(() => lockFile(file), file).reason).toBe("busy");
    expect(isLocked(file)).toBe(true);
    expect(openSync.mock.calls.filter((call) => call[0] === file)).toEqual([]);
  });

  test("a symlink, a directory and a FIFO are unsafe", () => {
    const target = path.join(dir, "target");
    fs.writeFileSync(target, "", { mode: 0o600 });
    const link = path.join(dir, "link");
    fs.symlinkSync(target, link);
    const directory = path.join(dir, "directory");
    fs.mkdirSync(directory);
    const fifo = path.join(dir, "fifo");
    execFileSync("mkfifo", [fifo]);
    for (const p of [link, directory, fifo]) {
      expect(lockError(() => lockFile(p), p).reason).toBe("unsafe");
      expect(isLocked(p)).toBe(false);
    }
    expect(isLocked(target)).toBe(false);
    hold(target);
  });

  test("private refuses a 0644 file and a second hard link; without it both lock", () => {
    const shared = path.join(dir, "shared");
    fs.writeFileSync(shared, "", { mode: 0o644 });
    fs.chmodSync(shared, 0o644);
    expect(lockError(() => lockFile(shared, { private: true }), shared).reason).toBe("not-private");
    const linked = path.join(dir, "linked");
    fs.writeFileSync(linked, "", { mode: 0o600 });
    fs.linkSync(linked, path.join(dir, "alias"));
    expect(lockError(() => lockFile(linked, { private: true }), linked).reason).toBe("unsafe");
    hold(shared, { private: false });
    hold(linked);
    hold(file, { private: true });
  });

  test("a non-SQLite file throws the driver error and isn't locked", () => {
    fs.writeFileSync(file, "caller data, not sqlite", { mode: 0o600 });
    const error = thrown(() => lockFile(file));
    expect(error).not.toBeInstanceOf(FileLockError);
    expect(sqliteErrorCode(error)).toBe(26);
    expect(isLocked(file)).toBe(false);
    expect(fs.readFileSync(file, "utf8")).toBe("caller data, not sqlite");
  });

  test("a missing parent propagates ENOENT", () => {
    const missing = path.join(dir, "missing", "db.sqlite");
    expect((thrown(() => lockFile(missing)) as NodeJS.ErrnoException).code).toBe("ENOENT");
  });

  test("a file created by a racing creator still locks", () => {
    fs.writeFileSync(file, "", { mode: 0o600 });
    spyLstat(file, 1, () => {
      throw Object.assign(new Error(`ENOENT: ${file}`), { code: "ENOENT" });
    });
    hold(file);
    mock.restore();
    expect(isLocked(file)).toBe(true);
  });

  test("a path swapped while locking throws changed and closes the connection", () => {
    fs.writeFileSync(file, "", { mode: 0o600 });
    // The call after the lock is the third: one to see the path exists, one for the baseline.
    spyLstat(file, 3, (real, ...args) => {
      const info = Reflect.apply(real, fs, args) as fs.BigIntStats;
      return { dev: info.dev, ino: info.ino + 1n };
    });
    expect(lockError(() => lockFile(file), file).reason).toBe("changed");
    mock.restore();
    expect(isLocked(file)).toBe(false);
    hold(file);
  });
});

describe("isLocked", () => {
  test("a missing path is free and stays missing", () => {
    expect(isLocked(file)).toBe(false);
    expect(fs.existsSync(file)).toBe(false);
    expect(isLocked(path.join(dir, "missing", "db.sqlite"))).toBe(false);
  });

  test("two probes don't see each other as held", () => {
    fs.writeFileSync(file, "", { mode: 0o600 });
    const reader = open(file, { readonly: true });
    reader.exec("BEGIN");
    reader.prepare("SELECT count(*) AS n FROM sqlite_master").get();
    expect(isLocked(file)).toBe(false);
    reader.exec("COMMIT");
  });
});
