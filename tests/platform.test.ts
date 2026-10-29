/**
 * Contracts for `src/platform.ts`: runtime detection; the SQLite wrapper and its result codes; the
 * exclusive file lock; the scheduler and its virtual twin for tests; the waiting lock; the type
 * stripper for CODE bodies, whose errors keep authored line positions; `safeFetch`, against real
 * local servers; and `warn`/`withWarnings`. Child processes check the lock across processes and,
 * with `YIAN_TEST_NODE` set to a Node binary, the built output on Node.
 */

import { afterEach, beforeEach, describe, expect, mock, spyOn, test } from "bun:test";
import { type ChildProcess, execFileSync, spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import type { AddressInfo } from "node:net";
import os from "node:os";
import path from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  type FileLock,
  FileLockError,
  insecureTlsWarning,
  isLocked,
  lockFile,
  openSqlite,
  realScheduler,
  RedirectError,
  redirectDetail,
  runtime,
  safeFetch,
  type Scheduler,
  type Send,
  SLEEP_CHUNK_MS,
  type Sqlite,
  sqliteErrorCode,
  stripTypes,
  syncFolder,
  TypeStripError,
  USER_AGENT,
  VERSION,
  virtualScheduler,
  waitForLock,
  warn,
  withWarnings,
} from "../src/platform.ts";

const realSetTimeout = globalThis.setTimeout;

let dir: string;
let file: string;
let opened: Sqlite[];
let held: FileLock[];
let children: Holder[];

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
  children = [];
});

afterEach(async () => {
  mock.restore();
  try {
    for (const child of children) await kill(child);
    for (const lock of held) lock.release();
    for (const db of opened) db.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

describe("runtime", () => {
  test.if(process.versions.bun !== undefined)("is bun under bun test", () => {
    expect(runtime).toBe("bun");
  });

  test.if(process.versions.bun === undefined)("is node under the Node runner", () => {
    expect(runtime).toBe("node");
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

  test("a safe-integer number binds as INTEGER, any other number as REAL", () => {
    const db = open(file);
    const type = db.prepare("SELECT typeof(?) AS t");
    expect(type.get(1)).toEqual({ t: "integer" });
    expect(type.get(1.5)).toEqual({ t: "real" });
  });

  test("foreign keys are off and double-quoted literals are accepted", () => {
    const db = open(file);
    expect(db.prepare("PRAGMA foreign_keys").get()).toEqual({ foreign_keys: 0 });
    expect(db.prepare('SELECT "x" AS v').get()).toEqual({ v: "x" });
  });

  test("run returns the rows changed and the last inserted rowid as numbers", () => {
    const db = open(file);
    db.exec("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)");
    const insert = db.prepare("INSERT INTO t (a) VALUES (?)");
    expect(insert.run(1)).toEqual({ changes: 1, lastInsertRowid: 1 });
    expect(insert.run(1)).toEqual({ changes: 1, lastInsertRowid: 2 });
    expect(db.prepare("UPDATE t SET a = ? WHERE a = ?").run(2, 1)).toEqual({ changes: 2, lastInsertRowid: 2 });
    expect(db.prepare("DELETE FROM t WHERE a = ?").run(9)).toEqual({ changes: 0, lastInsertRowid: 2 });
  });

  test("run returns a last inserted rowid outside the safe integer range as the exact bigint", () => {
    const db = open(file);
    db.exec("CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)");
    expect(db.prepare("INSERT INTO t VALUES (?, ?)").run(2n ** 53n, 1)).toEqual({
      changes: 1,
      lastInsertRowid: 2n ** 53n,
    });
    expect(db.prepare("UPDATE t SET a = ?").run(2)).toEqual({ changes: 1, lastInsertRowid: 2n ** 53n });
  });

  test("inTransaction is true only while a transaction is open", () => {
    const db = open(file);
    db.exec("CREATE TABLE t (a INTEGER)");
    expect(db.inTransaction).toBe(false);
    db.exec("BEGIN");
    expect(db.inTransaction).toBe(true);
    db.exec("COMMIT");
    expect(db.inTransaction).toBe(false);
    db.exec("BEGIN IMMEDIATE");
    db.prepare("INSERT INTO t VALUES (?)").run(1);
    expect(db.inTransaction).toBe(true);
    db.exec("ROLLBACK");
    expect(db.inTransaction).toBe(false);
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

  test("disposing releases the lock", () => {
    const lock = lockFile(file);
    try {
      expect(lock.path).toBe(file);
      expect(isLocked(file)).toBe(true);
    } finally {
      lock[Symbol.dispose]();
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
    fs.writeFileSync(shared, "");
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

  test("a path removed while locking throws changed and closes the connection", () => {
    fs.writeFileSync(file, "", { mode: 0o600 });
    spyLstat(file, 3, () => {
      throw Object.assign(new Error(`ENOENT: ${file}`), { code: "ENOENT" });
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
    reader.prepare("SELECT count(*) FROM sqlite_master").get();
    expect(isLocked(file)).toBe(false);
    reader.exec("COMMIT");
  });

  test("a path that vanishes before the open is free and stays missing", () => {
    spyLstat(file, 1, () => ({ isFile: () => true }));
    expect(isLocked(file)).toBe(false);
    expect(fs.existsSync(file)).toBe(false);
  });

  // Root ignores the mode, so the open would succeed.
  test.skipIf(process.getuid?.() === 0)("a lock file it can't open throws, held or not", () => {
    const lock = hold(file);
    fs.chmodSync(file, 0o000);
    try {
      expect(sqliteErrorCode(thrown(() => isLocked(file)))).toBe(14);
      lock.release();
      expect(sqliteErrorCode(thrown(() => isLocked(file)))).toBe(14);
    } finally {
      fs.chmodSync(file, 0o600);
    }
  });
});

/** Make `setTimeout` fire on the next macrotask whatever its delay, running `after(n)` after the n-th (1-based) callback. */
function instantTimers(after: (n: number) => void = () => {}) {
  let fired = 0;
  return spyOn(globalThis, "setTimeout").mockImplementation(((callback: () => void) =>
    realSetTimeout(() => {
      callback();
      after(++fired);
    }, 0)) as unknown as typeof setTimeout);
}

/** Return the delays `setTimeout` was armed with. */
function delays(spy: ReturnType<typeof instantTimers>): unknown[] {
  return spy.mock.calls.map((call) => call[1]);
}

/** Return what `promise` rejects with; fail if it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

describe("realScheduler", () => {
  test("sleep(0) and sleep(-5) resolve", async () => {
    await realScheduler.sleep(0);
    await realScheduler.sleep(-5);
  });

  test("sleep(NaN) and sleep(Infinity) reject with RangeError", async () => {
    expect(await rejection(realScheduler.sleep(Number.NaN))).toBeInstanceOf(RangeError);
    expect(await rejection(realScheduler.sleep(Number.POSITIVE_INFINITY))).toBeInstanceOf(RangeError);
  });

  test("sleep(30) takes at least 30 ms", async () => {
    const started = performance.now();
    await realScheduler.sleep(30);
    expect(performance.now() - started).toBeGreaterThanOrEqual(30);
  });

  test("a sleep longer than a chunk arms timers chunk by chunk and never reads the clock", async () => {
    const timers = instantTimers();
    const now = spyOn(Date, "now");
    await realScheduler.sleep(2.5 * SLEEP_CHUNK_MS);
    expect(delays(timers)).toEqual([SLEEP_CHUNK_MS, SLEEP_CHUNK_MS, SLEEP_CHUNK_MS / 2]);
    expect(now).not.toHaveBeenCalled();
  });

  test("an abort between chunks rejects with its reason and arms no further timer", async () => {
    const controller = new AbortController();
    const reason = new Error("stop");
    const timers = instantTimers((n) => {
      if (n === 2) controller.abort(reason);
    });
    expect(await rejection(realScheduler.sleep(3 * SLEEP_CHUNK_MS, controller.signal))).toBe(reason);
    expect(delays(timers)).toEqual([SLEEP_CHUNK_MS, SLEEP_CHUNK_MS]);
  });

  test("a pre-aborted signal rejects with its reason and arms no timer", async () => {
    const reason = new Error("stopped");
    const signal = AbortSignal.abort(reason);
    const timers = instantTimers();
    for (const ms of [10, 0, -5]) expect(await rejection(realScheduler.sleep(ms, signal))).toBe(reason);
    expect(timers).not.toHaveBeenCalled();
  });

  test("an abort during a long sleep rejects at once", async () => {
    const controller = new AbortController();
    const started = performance.now();
    realSetTimeout(() => controller.abort(), 20);
    const error = await rejection(realScheduler.sleep(10_000, controller.signal));
    expect((error as DOMException).name).toBe("AbortError");
    expect(performance.now() - started).toBeLessThan(1000);
  });

  test("completed sleeps leave no abort listener on the signal", async () => {
    const { signal } = new AbortController();
    let listeners = 0;
    const add = signal.addEventListener.bind(signal);
    const remove = signal.removeEventListener.bind(signal);
    signal.addEventListener = ((type: string, ...rest: [EventListener, AddEventListenerOptions?]) => {
      if (type === "abort") listeners++;
      add(type, ...rest);
    }) as typeof signal.addEventListener;
    signal.removeEventListener = ((type: string, ...rest: [EventListener, EventListenerOptions?]) => {
      if (type === "abort") listeners--;
      remove(type, ...rest);
    }) as typeof signal.removeEventListener;
    for (let i = 0; i < 20; i++) await realScheduler.sleep(1, signal);
    expect(listeners).toBe(0);
  });

  test("now() is the wall clock", () => {
    expect(Math.abs(realScheduler.now() - Date.now())).toBeLessThan(1000);
  });
});

describe("virtualScheduler", () => {
  test("now() starts at start and a sleep advances it", async () => {
    const clock = virtualScheduler(1000);
    expect(clock.now()).toBe(1000);
    await clock.sleep(500);
    expect(clock.now()).toBe(1500);
  });

  test("concurrent sleeps resolve in deadline order and end at the latest deadline", async () => {
    const clock = virtualScheduler(1000);
    const ended: [number, number][] = [];
    await Promise.all(
      [1000, 500].map((ms) => clock.sleep(ms).then(() => ended.push([ms, clock.now()]))),
    );
    expect(ended).toEqual([
      [500, 1500],
      [1000, 2000],
    ]);
    expect(clock.now()).toBe(2000);
  });

  test("same-deadline sleepers resolve in registration order", async () => {
    const clock = virtualScheduler();
    const ended: string[] = [];
    await Promise.all(["a", "b", "c"].map((name) => clock.sleep(100).then(() => ended.push(name))));
    expect(ended).toEqual(["a", "b", "c"]);
    expect(clock.now()).toBe(100);
  });

  test("an aborted sleeper rejects with its reason; a longer sleep still ends at its deadline", async () => {
    const clock = virtualScheduler(1000);
    const controller = new AbortController();
    const reason = new Error("stop");
    const aborted = rejection(clock.sleep(500, controller.signal));
    const longer = clock.sleep(2000);
    controller.abort(reason);
    expect(await aborted).toBe(reason);
    await longer;
    expect(clock.now()).toBe(3000);
  });

  test("a woken sleeper's next sleep is queued before the clock moves on", async () => {
    const clock = virtualScheduler();
    const woke: string[] = [];
    const a = (async () => {
      await clock.sleep(100);
      woke.push(`A@${clock.now()}`);
      await clock.sleep(100);
      woke.push(`A@${clock.now()}`);
    })();
    const b = clock.sleep(150).then(() => woke.push(`B@${clock.now()}`));
    await Promise.all([a, b]);
    expect(woke).toEqual(["A@100", "B@150", "A@200"]);
  });

  test("sleep(0) doesn't advance the clock", async () => {
    const clock = virtualScheduler(1000);
    await clock.sleep(0);
    expect(clock.now()).toBe(1000);
  });

  test("sleep(NaN) rejects with RangeError, and a pre-aborted signal rejects for 0", async () => {
    const clock = virtualScheduler();
    expect(await rejection(clock.sleep(Number.NaN))).toBeInstanceOf(RangeError);
    const reason = new Error("stopped");
    expect(await rejection(clock.sleep(0, AbortSignal.abort(reason)))).toBe(reason);
  });
});

/** Wrap `base`, counting sleeps and running `during(n)` once the n-th (1-based) sleep has begun. */
function countingSleeps(base: Scheduler, during: (n: number) => void = () => {}) {
  const counted = {
    sleeps: 0,
    now: () => base.now(),
    sleep(ms: number, signal?: AbortSignal) {
      const sleeping = base.sleep(ms, signal);
      during(++counted.sleeps);
      return sleeping;
    },
  };
  return counted;
}

describe("waitForLock", () => {
  test("a held lock times out busy once the virtual deadline passes", async () => {
    hold(file);
    const scheduler = countingSleeps(virtualScheduler(1000));
    const now = spyOn(scheduler, "now");
    const started = performance.now();
    const error = await rejection(waitForLock(file, { timeoutMs: 500, scheduler }));
    // The budget counts the sleeps, never the clock.
    expect(now).not.toHaveBeenCalled();
    expect(error).toBeInstanceOf(FileLockError);
    expect((error as FileLockError).reason).toBe("busy");
    expect((error as FileLockError).message).toContain("timed out");
    expect(scheduler.now()).toBe(1500);
    expect(scheduler.sleeps).toBe(5);
    expect(performance.now() - started).toBeLessThan(1000);
  });

  test("a NaN or negative timeout rejects with RangeError", async () => {
    for (const timeoutMs of [Number.NaN, -1]) {
      expect(await rejection(waitForLock(file, { timeoutMs, scheduler: virtualScheduler() }))).toBeInstanceOf(
        RangeError,
      );
    }
  });

  test("an abort during the third poll's sleep rejects with its reason", async () => {
    hold(file);
    const controller = new AbortController();
    const reason = new Error("stop");
    const scheduler = countingSleeps(virtualScheduler(), (n) => {
      if (n === 3) controller.abort(reason);
    });
    expect(await rejection(waitForLock(file, { signal: controller.signal, scheduler }))).toBe(reason);
    expect(scheduler.sleeps).toBe(3);
  });

  test("a holder released during the second poll's sleep hands over the lock", async () => {
    const holder = hold(file);
    const scheduler = countingSleeps(virtualScheduler(), (n) => {
      if (n === 2) holder.release();
    });
    const lock = await waitForLock(file, { scheduler });
    held.push(lock);
    expect(lock.path).toBe(file);
    expect(isLocked(file)).toBe(true);
    expect(scheduler.sleeps).toBe(2);
  });

  test("a pre-aborted signal on a free lock rejects with its reason and leaves it free", async () => {
    const reason = new Error("stopped");
    const scheduler = countingSleeps(virtualScheduler());
    expect(await rejection(waitForLock(file, { signal: AbortSignal.abort(reason), scheduler }))).toBe(reason);
    expect(isLocked(file)).toBe(false);
    expect(scheduler.sleeps).toBe(0);
  });

  test("a free lock is taken at once", async () => {
    const scheduler = countingSleeps(virtualScheduler());
    const lock = await waitForLock(file, { scheduler });
    held.push(lock);
    expect(isLocked(file)).toBe(true);
    expect(scheduler.sleeps).toBe(0);
  });

  test("private on a 0644 file fails not-private without waiting", async () => {
    fs.writeFileSync(file, "");
    fs.chmodSync(file, 0o644);
    const scheduler = countingSleeps(virtualScheduler());
    const error = await rejection(waitForLock(file, { private: true, scheduler }));
    expect((error as FileLockError).reason).toBe("not-private");
    expect(scheduler.sleeps).toBe(0);
  });
});

describe("syncFolder", () => {
  /** A Node system error with `code`, as `fs` throws. */
  const systemError = (code: string) => Object.assign(new Error(code), { code, syscall: "fsync" });

  test("syncs the folder through a descriptor it closes", () => {
    const opened = new Map<number, string>();
    const synced: string[] = [];
    const closed: number[] = [];
    const open = fs.openSync;
    const fsync = fs.fsyncSync;
    const close = fs.closeSync;
    spyOn(fs, "openSync").mockImplementation(((target: fs.PathLike, ...rest: unknown[]) => {
      const fd = (open as (...args: unknown[]) => number)(target, ...rest);
      opened.set(fd, String(target));
      return fd;
    }) as typeof fs.openSync);
    spyOn(fs, "fsyncSync").mockImplementation((fd) => {
      synced.push(opened.get(fd)!);
      fsync(fd);
    });
    spyOn(fs, "closeSync").mockImplementation((fd) => {
      closed.push(fd);
      close(fd);
    });
    syncFolder(dir);
    expect(synced).toEqual([dir]);
    expect(closed).toEqual([...opened.keys()]);
  });

  test.each(["EINVAL", "ENOTSUP", "EOPNOTSUPP", "EISDIR", "EPERM"])(
    "does nothing where a folder sync fails %s, open or sync",
    (code) => {
      const open = spyOn(fs, "openSync").mockImplementation(() => {
        throw systemError(code);
      });
      expect(syncFolder(dir)).toBeUndefined();
      open.mockRestore();
      spyOn(fs, "fsyncSync").mockImplementation(() => {
        throw systemError(code);
      });
      expect(syncFolder(dir)).toBeUndefined();
    },
  );

  test("any other failure propagates, and the descriptor is still closed", () => {
    expect(thrown(() => syncFolder(path.join(dir, "missing")))).toMatchObject({ code: "ENOENT" });
    const failure = systemError("EIO");
    spyOn(fs, "fsyncSync").mockImplementation(() => {
      throw failure;
    });
    const close = spyOn(fs, "closeSync");
    expect(thrown(() => syncFolder(dir))).toBe(failure);
    expect(close).toHaveBeenCalledTimes(1);
  });
});

describe("stripTypes", () => {
  const AsyncFunction = (async () => {}).constructor as new (...args: string[]) => (inputs: unknown) => Promise<unknown>;

  /** Return the offsets of every `\n` in `text`. */
  function newlines(text: string): number[] {
    return [...text.matchAll(/\n/g)].map((match) => match.index);
  }

  /** Return the `TypeStripError` that `stripTypes(body)` rejects with. */
  async function stripError(body: string): Promise<TypeStripError> {
    const error = await rejection(stripTypes(body));
    expect(error).toBeInstanceOf(TypeStripError);
    return error as TypeStripError;
  }

  test("blanks the types, keeping the length and newline offsets", async () => {
    const body = "const n: number = inputs.n;\nreturn (n as number) * 2;";
    const js = await stripTypes(body);
    expect(js.length).toBe(body.length);
    expect(newlines(js)).toEqual(newlines(body));
    expect(js).not.toMatch(/number/);
    expect(await new AsyncFunction("inputs", js)({ n: 1 })).toBe(2);
  });

  test("returns plain JavaScript unchanged", async () => {
    const body = "const a = [inputs.x, 2];\nreturn a.map((v) => v * 2);";
    expect(await stripTypes(body)).toBe(body);
  });

  test("keeps the length of a body with non-ASCII text", async () => {
    const body = "const s: string = '😀é';\nreturn s;";
    const js = await stripTypes(body);
    expect(js.length).toBe(body.length);
    expect(await new AsyncFunction("inputs", js)({})).toBe("😀é");
  });

  test("accepts top-level await and return", async () => {
    const js = await stripTypes("const v: number = await Promise.resolve(inputs.v);\nreturn v;");
    expect(await new AsyncFunction("inputs", js)({ v: 7 })).toBe(7);
  });

  test.each([
    ["let a = 1;\n\nconst x: = 1;", 3, 10],
    ["const x: = 1;", 1, 10],
    ["const s = '😀é'; const x: = 1;", 1, 27],
    ["\tconst x: = 1;", 1, 11],
    ["const s = '中'; const x: = 1;", 1, 25],
    ["let a = 1;\rconst x: = 1;", 2, 10],
    ["let a = 1;\r\nconst x: = 1;", 2, 10],
    ["enum E { A }", 1, 1],
    ["return (\n", 1, 9],
    ["let a = 1;\nreturn (\n\n", 2, 9],
    ["let a = 1;\r\nreturn (\r\n", 2, 9],
    ["/* open", 1, 8],
  ])("%j fails at line %d, column %d", async (body, line, column) => {
    const error = await stripError(body);
    expect(error).toBeInstanceOf(SyntaxError);
    expect(error.name).toBe("TypeStripError");
    expect([error.line, error.column]).toEqual([line, column]);
  });

  test("a body closing the wrapper comes back unchanged and fails to construct", async () => {
    const body = "return 1; } globalThis.escaped = true; function g() {";
    const js = await stripTypes(body);
    expect(js).toBe(body);
    expect(() => new AsyncFunction("inputs", js)).toThrow(SyntaxError);
    expect((globalThis as { escaped?: unknown }).escaped).toBeUndefined();
  });

  test("keeps authored line numbers in the built function's stack", async () => {
    /** Return the line of the `yian-code:t` frame in what running `body` throws. */
    async function thrownLine(body: string): Promise<number> {
      const js = await stripTypes(body);
      const error = await rejection(new AsyncFunction("inputs", js + "\n//# sourceURL=yian-code:t")({}));
      const frame = /yian-code:t:(\d+):\d+/.exec((error as Error).stack ?? "");
      expect(frame).not.toBeNull();
      return Number(frame![1]);
    }
    const header = (await thrownLine('throw new Error("h");')) - 1;
    const body = '// a comment\n\nconst n: number = 1;\nthrow new Error("x" + n);';
    expect((await thrownLine(body)) - header).toBe(4);
  });
});

/** One request a test server received. */
interface Received {
  method: string;
  headers: http.IncomingHttpHeaders;
  body: string;
}

/** A local test server: its base URL, its port and every request it received. */
interface TestServer {
  url: string;
  port: number;
  received: Received[];
}

/** The credential headers a redirect must never carry to a second origin. */
const CREDENTIALS = {
  authorization: "Bearer sk-secret",
  "x-api-key": "key-secret",
  "api-key": "azure-secret",
  "x-goog-api-key": "goog-secret",
};

/** A fake `Send` that records each call and answers `reply`. */
function fakeSend(reply: () => Response = () => new Response("ok")) {
  const calls: { input: string | URL; init: RequestInit }[] = [];
  const send: Send = async (input, init) => {
    calls.push({ input, init });
    return reply();
  };
  return { send, calls };
}

describe("redirectDetail", () => {
  test("redirectDetail", () => {
    expect(redirectDetail(null)).toBe("redirect without a Location header");
    expect(redirectDetail("")).toBe("redirect without a Location header");
    expect(redirectDetail("http://[::1")).toBe("redirect to an unparsable Location");
    expect(redirectDetail("//[::1")).toBe("redirect to an unparsable Location");
    expect(redirectDetail("https://[::1]:8443/x?t=1")).toBe("redirect to https://[::1]:8443");
    expect(redirectDetail("https://h.example:443/x")).toBe("redirect to https://h.example");
    expect(redirectDetail(" HTTP://H.Example/x ")).toBe("redirect to http://h.example");
    expect(redirectDetail("mailto:x")).toBe("redirect to a relative Location");
    expect(redirectDetail("///x")).toBe("redirect to a relative Location");
    expect(redirectDetail("https:///x")).toBe("redirect to a relative Location");
    expect(redirectDetail("//h.example/x")).toBe("redirect to a Location without a scheme");
    expect(redirectDetail("x/y")).toBe("redirect to a relative Location");
    // urlsplit("http:host") has no netloc.
    expect(redirectDetail("http:host")).toBe("redirect to a relative Location");
  });
});

/** The package root: Bun children run from it, so Bun never looks for packages to install. */
const ROOT = path.resolve(import.meta.dirname, "..");
const SOURCE_URL = pathToFileURL(path.join(ROOT, "src/platform.ts")).href;
const BUILT = path.join(ROOT, "dist/src/platform.js");
const BUILT_URL = pathToFileURL(BUILT).href;
/**
 * The public entry. It resolves by self-reference from inside ROOT (this file, and children whose
 * cwd is ROOT). Typed string so tsc doesn't resolve it to dist/, missing before a build.
 */
const ENTRY: string = "@ngocbh/yian/platform";

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node cases skipped.
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

/** Spawned tests wait on real processes, well past the default timeout. */
const SPAWNED_MS = 20_000;

describe("safeFetch", () => {
  let servers: http.Server[] = [];

  afterEach(async () => {
    for (const server of servers) {
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    }
    servers = [];
  });

  /** Start a server on 127.0.0.1 that records each request, then lets `handle` answer it. */
  async function serve(handle: (res: http.ServerResponse) => void): Promise<TestServer> {
    const received: Received[] = [];
    const server = http.createServer((req, res) => {
      let body = "";
      req.setEncoding("utf8");
      req.on("data", (chunk: string) => (body += chunk));
      req.on("end", () => {
        received.push({ method: req.method!, headers: req.headers, body });
        handle(res);
      });
    });
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;
    return { url: `http://127.0.0.1:${port}/`, port, received };
  }

  /** Start server B answering 200 and server A redirecting with `status` to a path on B. */
  async function redirectPair(status: number): Promise<{ a: TestServer; b: TestServer }> {
    const b = await serve((res) => res.end("reached"));
    const a = await serve((res) => {
      res.writeHead(status, { location: `${b.url}steal/path?token=abc` });
      res.end("moved");
    });
    return { a, b };
  }

  for (const status of [301, 302, 303, 307, 308]) {
    test(`a ${status} throws RedirectError and the target gets nothing`, async () => {
      const { a, b } = await redirectPair(status);
      const error = await rejection(safeFetch(a.url, { method: "POST", body: "payload", headers: CREDENTIALS }));
      expect(error).toBeInstanceOf(RedirectError);
      const redirect = error as RedirectError;
      expect(redirect.name).toBe("RedirectError");
      expect(redirect.status).toBe(status);
      expect(redirect.detail).toBe(`redirect to http://127.0.0.1:${b.port}`);
      expect(redirect.message).toBe(`HTTP ${status} redirect to http://127.0.0.1:${b.port}`);
      for (const leak of ["steal", "token", "abc"]) expect(redirect.message).not.toContain(leak);
      expect(a.received).toHaveLength(1);
      for (const [name, value] of Object.entries(CREDENTIALS)) expect(a.received[0]!.headers[name]).toBe(value);
      expect(b.received).toEqual([]);
    });
  }

  test('a caller\'s redirect: "follow" is overridden', async () => {
    const { a, b } = await redirectPair(302);
    const error = await rejection(safeFetch(a.url, { redirect: "follow", headers: CREDENTIALS }));
    expect(error).toBeInstanceOf(RedirectError);
    expect((error as RedirectError).status).toBe(302);
    expect(b.received).toEqual([]);
  });

  test("a 2xx and a 4xx are returned as is", async () => {
    const ok = await serve((res) => res.end("hello"));
    const missing = await serve((res) => {
      res.writeHead(404, { "content-type": "text/plain" });
      res.end("missing");
    });
    const first = await safeFetch(ok.url);
    expect(first.status).toBe(200);
    expect(await first.text()).toBe("hello");
    const second = await safeFetch(missing.url);
    expect(second.status).toBe(404);
    expect(second.headers.get("content-type")).toBe("text/plain");
    expect(await second.text()).toBe("missing");
  });

  test("User-Agent defaults to USER_AGENT and a caller's wins", async () => {
    const server = await serve((res) => res.end());
    await (await safeFetch(server.url)).text();
    await (await safeFetch(server.url, { headers: { "User-Agent": "host/1.2" } })).text();
    expect(server.received.map((r) => r.headers["user-agent"])).toEqual([USER_AGENT, "host/1.2"]);
    expect(USER_AGENT).toBe(`yian/${VERSION}`);
  });

  test("a URL with user info is refused without echoing it, and nothing is sent", async () => {
    const { send, calls } = fakeSend();
    for (const url of ["http://u:p@127.0.0.1/", "http://u@127.0.0.1/", new URL("http://u:p@127.0.0.1/")]) {
      const error = await rejection(safeFetch(url, {}, send));
      expect(error).toBeInstanceOf(TypeError);
      const message = (error as TypeError).message;
      expect(message).toBe("request URL must not contain user info");
      expect(message).not.toContain("u:");
      expect(message).not.toContain("p@");
      expect((error as TypeError).cause).toBeUndefined();
    }
    expect(calls).toEqual([]);
  });

  test("a URL that doesn't parse or isn't http(s) is refused without echoing it, and nothing is sent", async () => {
    const { send, calls } = fakeSend();
    for (const url of ["nope", "http://exa mple/", "ftp://127.0.0.1/"]) {
      const error = await rejection(safeFetch(url, {}, send));
      expect(error).toBeInstanceOf(TypeError);
      const message = (error as TypeError).message;
      expect(message).toBe("request URL is not a valid http(s) URL");
      for (const leak of ["nope", "exa", "ftp", "127.0.0.1"]) expect(message).not.toContain(leak);
      expect((error as TypeError).cause).toBeUndefined();
    }
    expect(calls).toEqual([]);
  });

  test("an invalid header value is refused without echoing it, and nothing is sent", async () => {
    const { send, calls } = fakeSend();
    const error = await rejection(safeFetch("https://api.example/v1", { headers: { "x-k": "sec\nret" } }, send));
    expect(error).toBeInstanceOf(TypeError);
    const message = (error as TypeError).message;
    expect(message).toBe("request headers are not valid HTTP header text");
    for (const leak of ["sec", "ret", "x-k"]) expect(message).not.toContain(leak);
    expect((error as TypeError).cause).toBeUndefined();
    expect(calls).toEqual([]);
  });

  test("send gets manual redirects, no runtime timeout and the caller's signal, method, body and headers", async () => {
    const { send, calls } = fakeSend();
    const signal = new AbortController().signal;
    const response = await safeFetch(
      "https://api.example/v1",
      { method: "POST", body: "payload", headers: { "x-api-key": "k" }, redirect: "follow", signal },
      send,
    );
    expect(await response.text()).toBe("ok");
    expect(calls).toHaveLength(1);
    const { input, init } = calls[0]!;
    expect(input).toBe("https://api.example/v1");
    expect(init.redirect).toBe("manual");
    expect((init as { timeout?: unknown }).timeout).toBe(false);
    expect(init.signal).toBe(signal);
    expect(init.method).toBe("POST");
    expect(init.body).toBe("payload");
    const headers = new Headers(init.headers);
    expect(headers.get("x-api-key")).toBe("k");
    expect(headers.get("user-agent")).toBe(USER_AGENT);
  });

  test("insecureTlsWarning gives the text for 0 and null otherwise", () => {
    expect(insecureTlsWarning("0")).toBe(TLS_WARNING);
    for (const value of [undefined, "", "1", "false", " 0"]) expect(insecureTlsWarning(value)).toBeNull();
  });

  test(
    "warns once per process when NODE_TLS_REJECT_UNAUTHORIZED is 0",
    () => {
      const result = runTlsWarningChild("bun");
      expect(result.status, result.stderr).toBe(0);
      expect(result.stderr.split(TLS_WARNING)).toHaveLength(2);
    },
    SPAWNED_MS,
  );

  test.skipIf(!NODE)(
    "warns once per process on Node when NODE_TLS_REJECT_UNAUTHORIZED is 0",
    () => {
      const result = runTlsWarningChild("node");
      expect(result.status, result.stderr).toBe(0);
      expect(result.stderr.split(TLS_WARNING)).toHaveLength(2);
    },
    SPAWNED_MS,
  );

  test("a 3xx from send throws RedirectError and cancels the body", async () => {
    let cancelled = false;
    const body = new ReadableStream({
      cancel() {
        cancelled = true;
      },
    });
    const { send } = fakeSend(() => new Response(body, { status: 307, headers: { location: "/elsewhere" } }));
    const error = await rejection(safeFetch("http://127.0.0.1/", {}, send));
    expect(error).toBeInstanceOf(RedirectError);
    expect((error as RedirectError).message).toBe("HTTP 307 redirect to a relative Location");
    expect(cancelled).toBe(true);
  });

  test("VERSION equals package.json's version", () => {
    expect(VERSION).toBe(JSON.parse(fs.readFileSync(path.join(ROOT, "package.json"), "utf8")).version);
  });
});

const BWRAP =
  process.platform === "linux" && spawnSync("bwrap", ["--ro-bind", "/", "/", "true"]).status === 0;

/** A child process holding a lock, its stdout read line by line. */
interface Holder {
  child: ChildProcess;
  lines: AsyncIterator<string>;
  closed: Promise<unknown>;
  /** Rejects when the child can't be spawned or signalled. */
  failed: Promise<never>;
  stderr: string;
  /** The child's own second `lockFile` and `isLocked` results, with `selfCheck`. */
  selfCheck?: unknown;
}

/** Return the holder's next stdout line; reject, with its stderr, when it exits or after 10 s. */
async function nextLine(holder: Holder): Promise<string> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`the holder printed nothing for 10 s; stderr:\n${holder.stderr}`)), 10_000);
  });
  try {
    const next = await Promise.race([holder.lines.next(), timeout, holder.failed]);
    if (!next.done) return next.value;
    await holder.closed;
    throw new Error(`the holder exited (${holder.child.exitCode ?? holder.child.signalCode}); stderr:\n${holder.stderr}`);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Start a child that locks `p`, and resolve once it prints `locked`. `"bun"` runs the
 * source on this Bun, `"node"` the build on `NODE`, `"bwrap"` the Bun child in its own
 * pid namespace; `specifier`, when given, is the module the child imports instead. With
 * `selfCheck`, the child first retries `lockFile(p)` and probes `isLocked(p)` itself.
 * Stdin line `release` releases the lock (the child answers `released`); stdin EOF ends
 * the child, so a crashed test leaves no orphan.
 */
async function spawnHolder(
  kind: "bun" | "node" | "bwrap",
  p: string,
  {
    selfCheck = false,
    specifier = kind === "node" ? BUILT_URL : SOURCE_URL,
  }: { selfCheck?: boolean; specifier?: string } = {},
): Promise<Holder> {
  // Plain JavaScript without `using`, which Node 22 can't parse.
  const script = `
const { FileLockError, isLocked, lockFile } = await import(${JSON.stringify(specifier)});
const { createInterface } = await import("node:readline");
const p = ${JSON.stringify(p)};
const lock = lockFile(p);
${
  selfCheck
    ? `let second = "acquired";
try {
  lockFile(p).release();
} catch (error) {
  second = error instanceof FileLockError ? error.reason : String(error);
}
console.log(JSON.stringify({ second, isLocked: isLocked(p) }));`
    : ""
}
console.log("locked");
for await (const line of createInterface({ input: process.stdin })) {
  if (line === "release") {
    lock.release();
    console.log("released");
  }
}
`;
  const argv =
    kind === "node"
      ? [NODE!, "--input-type=module", "-e", script]
      : kind === "bun"
        ? [process.execPath, "-e", script]
        : [
            "bwrap",
            ...["--unshare-pid", "--die-with-parent", "--ro-bind", "/", "/", "--bind", dir, dir],
            ...["--dev", "/dev", "--proc", "/proc", process.execPath, "-e", script],
          ];
  const child = spawn(argv[0]!, argv.slice(1), { cwd: ROOT, stdio: "pipe" });
  const holder: Holder = {
    child,
    lines: createInterface({ input: child.stdout! })[Symbol.asyncIterator](),
    closed: new Promise((resolve) => child.on("close", resolve)),
    failed: new Promise((_, reject) => child.on("error", reject)),
    stderr: "",
  };
  // `nextLine` races `failed`; this keeps a failure after the last line from going unhandled.
  holder.failed.catch(() => {});
  children.push(holder);
  child.stderr!.setEncoding("utf8").on("data", (chunk: string) => {
    holder.stderr += chunk;
  });
  if (selfCheck) holder.selfCheck = JSON.parse(await nextLine(holder));
  const line = await nextLine(holder);
  if (line !== "locked") throw new Error(`the holder printed ${line}; stderr:\n${holder.stderr}`);
  return holder;
}

const TLS_WARNING =
  "NODE_TLS_REJECT_UNAUTHORIZED=0 turns off TLS certificate checks for every request; unset it and trust a private CA with NODE_EXTRA_CA_CERTS instead";

/**
 * Run a child with `NODE_TLS_REJECT_UNAUTHORIZED=0` that calls `safeFetch` twice against its own
 * plain-http server: `"bun"` runs the source on this Bun, `"node"` the build on `NODE`.
 */
function runTlsWarningChild(kind: "bun" | "node"): { status: number | null; stderr: string } {
  // The server lives in the child, since spawnSync blocks this process's event loop.
  const script = `
const { safeFetch } = await import(${JSON.stringify(kind === "node" ? BUILT_URL : SOURCE_URL)});
const http = await import("node:http");
const server = http.createServer((req, res) => res.end("ok"));
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const url = "http://127.0.0.1:" + server.address().port + "/";
for (let i = 0; i < 2; i++) await (await safeFetch(url)).text();
server.closeAllConnections();
server.close();
`;
  const argv = kind === "node" ? [NODE!, "--input-type=module", "-e", script] : [process.execPath, "-e", script];
  const result = spawnSync(argv[0]!, argv.slice(1), {
    cwd: ROOT,
    env: { ...process.env, NODE_TLS_REJECT_UNAUTHORIZED: "0" },
    encoding: "utf8",
  });
  return { status: result.status, stderr: result.stderr };
}

/** SIGKILL the holder and wait for it to exit and close its pipes. */
async function kill(holder: Holder): Promise<void> {
  holder.child.kill("SIGKILL");
  await holder.closed;
}

describe("warn", () => {
  test("emits a process warning outside withWarnings", () => {
    const emit = spyOn(process, "emitWarning").mockImplementation(() => {});
    try {
      warn("m");
      warn("m", "RuntimeWarning");
      expect(emit.mock.calls).toEqual([["m"], ["m", "RuntimeWarning"]] as never);
    } finally {
      emit.mockRestore();
    }
  });

  test("goes to the innermost withWarnings handler, across awaits, until fn returns", async () => {
    const emit = spyOn(process, "emitWarning").mockImplementation(() => {});
    try {
      const outer: [string, string | undefined][] = [];
      const inner: [string, string | undefined][] = [];
      await withWarnings(
        (message, type) => void outer.push([message, type]),
        async () => {
          warn("a");
          await Promise.resolve();
          warn("b", "RuntimeWarning");
          withWarnings(
            (message, type) => void inner.push([message, type]),
            () => warn("c"),
          );
          warn("d");
        },
      );
      expect(outer).toEqual([
        ["a", undefined],
        ["b", "RuntimeWarning"],
        ["d", undefined],
      ]);
      expect(inner).toEqual([["c", undefined]]);
      expect(emit).not.toHaveBeenCalled();
      warn("e");
      expect(emit.mock.calls).toEqual([["e"]] as never);
    } finally {
      emit.mockRestore();
    }
  });
});

describe("the lock across processes", () => {
  test(
    "a Bun child's lock is busy here until the child is killed, and leaves nothing behind",
    async () => {
      const holder = await spawnHolder("bun", file);
      expect(lockError(() => lockFile(file), file).reason).toBe("busy");
      expect(isLocked(file)).toBe(true);
      await kill(holder);
      hold(file);
      expect(fs.readdirSync(dir)).toEqual(["db.sqlite"]);
    },
    SPAWNED_MS,
  );

  test(
    "a child's own retry and probe see busy and keep its lock",
    async () => {
      const holder = await spawnHolder("bun", file, { selfCheck: true });
      expect(holder.selfCheck).toEqual({ second: "busy", isLocked: true });
      // Closing the extra connections in the holder must not drop its POSIX lock.
      expect(lockError(() => lockFile(file), file).reason).toBe("busy");
    },
    SPAWNED_MS,
  );

  test(
    "a child's release frees the lock without a kill",
    async () => {
      const holder = await spawnHolder("bun", file);
      holder.child.stdin!.write("release\n");
      expect(await nextLine(holder)).toBe("released");
      hold(file);
    },
    SPAWNED_MS,
  );

  test(
    "waitForLock takes the lock once the holding child is killed",
    async () => {
      const holder = await spawnHolder("bun", file);
      const started = performance.now();
      // The one deliberate delay: the wait must outlast a few busy polls first.
      setTimeout(() => holder.child.kill("SIGKILL"), 200);
      held.push(await waitForLock(file, { timeoutMs: 5000 }));
      expect(performance.now() - started).toBeGreaterThanOrEqual(200);
    },
    SPAWNED_MS,
  );

  test.skipIf(!NODE)(
    "a Node child running the build holds the lock against Bun, and keeps it through its own checks",
    async () => {
      const holder = await spawnHolder("node", file, { selfCheck: true });
      expect(holder.selfCheck).toEqual({ second: "busy", isLocked: true });
      expect(lockError(() => lockFile(file), file).reason).toBe("busy");
      await kill(holder);
      hold(file);
    },
    SPAWNED_MS,
  );

  test.skipIf(!NODE)(
    "a Node child's isLocked and lockFile see a lock Bun holds",
    () => {
      const script = `
const { FileLockError, isLocked, lockFile } = await import(${JSON.stringify(BUILT_URL)});
const p = ${JSON.stringify(file)};
const locked = isLocked(p);
let lock = "acquired";
try {
  lockFile(p).release();
} catch (error) {
  lock = error instanceof FileLockError ? error.reason : String(error);
}
console.log(JSON.stringify({ isLocked: locked, lock }));
`;
      const probe = () => JSON.parse(execFileSync(NODE!, ["--input-type=module", "-e", script], { encoding: "utf8" }));
      const lock = hold(file);
      expect(probe()).toEqual({ isLocked: true, lock: "busy" });
      lock.release();
      expect(probe()).toEqual({ isLocked: false, lock: "acquired" });
    },
    SPAWNED_MS,
  );

  test.skipIf(!BWRAP)(
    "a child in another pid namespace holds the lock until killed",
    async () => {
      const holder = await spawnHolder("bwrap", file);
      expect(lockError(() => lockFile(file), file).reason).toBe("busy");
      expect(isLocked(file)).toBe(true);
      await kill(holder);
      hold(file);
      expect(fs.readdirSync(dir)).toEqual(["db.sqlite"]);
    },
    SPAWNED_MS,
  );
});

describe("the @ngocbh/yian/platform entry", () => {
  test.if(process.versions.bun !== undefined)("on Bun it is the source module yian itself uses", async () => {
    expect(fs.realpathSync(fileURLToPath(import.meta.resolve(ENTRY)))).toBe(
      fs.realpathSync(path.join(ROOT, "src/platform.ts")),
    );
    const entry = (await import(ENTRY)) as typeof import("../src/platform.ts");
    expect(entry.lockFile).toBe(lockFile);
    expect(entry.waitForLock).toBe(waitForLock);
    expect(entry.isLocked).toBe(isLocked);
    expect(entry.openSqlite).toBe(openSqlite);
    expect(entry.FileLockError).toBe(FileLockError);
    expect(entry.sqliteErrorCode).toBe(sqliteErrorCode);
  });

  test.if(process.versions.bun !== undefined)(
    "a Bun child importing the entry holds the lock against this process until it releases",
    async () => {
      const holder = await spawnHolder("bun", file, { specifier: ENTRY });
      expect(lockError(() => lockFile(file), file).reason).toBe("busy");
      expect(isLocked(file)).toBe(true);
      holder.child.stdin!.write("release\n");
      expect(await nextLine(holder)).toBe("released");
      hold(file);
    },
    SPAWNED_MS,
  );

  test.skipIf(!NODE)(
    "on Node it resolves to the build and carries the lock and SQLite names",
    () => {
      const script = `
const entry = await import(${JSON.stringify(ENTRY)});
const names = ["lockFile", "waitForLock", "isLocked", "FileLockError", "openSqlite", "sqliteErrorCode"];
const types = Object.fromEntries(names.map((n) => [n, typeof entry[n]]));
console.log(JSON.stringify({ resolved: import.meta.resolve(${JSON.stringify(ENTRY)}), types }));
`;
      const result = spawnSync(NODE!, ["--input-type=module", "-e", script], { cwd: ROOT, encoding: "utf8" });
      expect(result.status, result.stderr).toBe(0);
      expect(result.stderr).toBe("");
      const { resolved, types } = JSON.parse(result.stdout);
      expect(fs.realpathSync(fileURLToPath(resolved))).toBe(fs.realpathSync(BUILT));
      expect(types).toEqual({
        lockFile: "function",
        waitForLock: "function",
        isLocked: "function",
        FileLockError: "function",
        openSqlite: "function",
        sqliteErrorCode: "function",
      });
    },
    SPAWNED_MS,
  );

  test.skipIf(!NODE)(
    "a Node child importing the entry holds the lock against Bun until killed",
    async () => {
      const holder = await spawnHolder("node", file, { specifier: ENTRY });
      expect(lockError(() => lockFile(file), file).reason).toBe("busy");
      await kill(holder);
      hold(file);
    },
    SPAWNED_MS,
  );
});

describe("the built output on Node", () => {
  test.skipIf(!NODE)(
    "runs the SQLite wrapper, the lock, the schedulers and stripTypes without an experimental warning",
    async () => {
      const sample = "const n: number = inputs.n;\nreturn (n as number) * 2;";
      // One process for every check, so Node starts once; it reports a JSON object.
      const script = `
import fs from "node:fs";
import path from "node:path";
const p = await import(${JSON.stringify(BUILT_URL)});
const dir = ${JSON.stringify(dir)};
const out = { runtime: p.runtime };
const caught = (f) => {
  try {
    f();
    return "no throw";
  } catch (error) {
    return error;
  }
};

const file = path.join(dir, "round-trip.sqlite");
const db = p.openSqlite(file);
db.exec("CREATE TABLE t (a INTEGER, b TEXT, c BLOB, d TEXT)");
db.prepare("INSERT INTO t VALUES (?, ?, ?, ?)").run(Number.MAX_SAFE_INTEGER, "x", new Uint8Array([0, 1, 255]), null);
const row = db.prepare("SELECT a, b, c, d FROM t").get();
out.row = { ...row, c: [...row.c] };
out.rowIsPlain = Object.getPrototypeOf(row) === Object.prototype;
out.blobIsBytes = row.c instanceof Uint8Array;
out.noRow = db.prepare("SELECT a FROM t WHERE a = ?").get(0) === undefined;
db.exec("CREATE TABLE n (a INTEGER)");
db.prepare("INSERT INTO n VALUES (?)").run(5n);
out.bigintParam = db.prepare("SELECT a FROM n").get().a;
out.boundTypes = [1, 1.5].map((value) => db.prepare("SELECT typeof(?) AS t").get(value).t);
out.foreignKeys = db.prepare("PRAGMA foreign_keys").get().foreign_keys;
try {
  out.doubleQuoted = db.prepare('SELECT "x" AS v').get().v;
} catch (error) {
  out.doubleQuoted = String(error);
}
out.unsafeIntegers = [2n ** 53n, 2n ** 60n].map((big) => {
  db.prepare("DELETE FROM n").run();
  db.prepare("INSERT INTO n VALUES (?)").run(big);
  return caught(() => db.prepare("SELECT a FROM n").all()) instanceof RangeError;
});
db.close();
db.close();
const ro = p.openSqlite(file, { readonly: true });
out.readonlyRows = ro.prepare("SELECT count(*) AS n FROM t").get().n;
out.readonlyWriteThrows = caught(() => ro.exec("DELETE FROM t")) instanceof Error;
ro.close();
const missing = path.join(dir, "missing.sqlite");
out.readonlyMissing = [p.sqliteErrorCode(caught(() => p.openSqlite(missing, { readonly: true }))), fs.existsSync(missing)];

const closeFile = path.join(dir, "close.sqlite");
const a = p.openSqlite(closeFile);
a.exec("PRAGMA busy_timeout = 0");
a.exec("PRAGMA locking_mode = EXCLUSIVE");
a.exec("CREATE TABLE t (a INTEGER)");
a.exec("BEGIN EXCLUSIVE");
a.prepare("INSERT INTO t VALUES (?)").run(1);
a.exec("COMMIT");
const kept = a.prepare("SELECT a FROM t");
out.keptRows = kept.all();
a.close();
const b = p.openSqlite(closeFile);
b.exec("PRAGMA busy_timeout = 0");
const begun = caught(() => {
  b.exec("BEGIN EXCLUSIVE");
  b.exec("COMMIT");
});
out.closeReleases = begun === "no throw" ? "began" : String(begun);
out.keptAfterCloseThrows = caught(() => kept.all()) instanceof Error;
b.close();

const holderDb = p.openSqlite(file);
const waiterDb = p.openSqlite(file);
holderDb.exec("PRAGMA busy_timeout = 0");
waiterDb.exec("PRAGMA busy_timeout = 0");
holderDb.exec("BEGIN EXCLUSIVE");
out.busyCode = p.sqliteErrorCode(caught(() => waiterDb.exec("BEGIN EXCLUSIVE")));
holderDb.exec("COMMIT");
holderDb.close();
waiterDb.close();
const enoent = caught(() => fs.statSync(path.join(dir, "missing")));
out.enoent = [enoent.code, p.sqliteErrorCode(enoent) === undefined];

const lockPath = path.join(dir, "lock");
const lock = p.lockFile(lockPath);
const busy = caught(() => p.lockFile(lockPath));
out.lockBusy = busy instanceof p.FileLockError ? busy.reason : String(busy);
out.lockedWhileHeld = p.isLocked(lockPath);
lock.release();
out.lockedAfterRelease = p.isLocked(lockPath);
p.lockFile(lockPath).release();

const clock = p.virtualScheduler(1000);
const ended = [];
await Promise.all([1000, 500].map((ms) => clock.sleep(ms).then(() => ended.push([ms, clock.now()]))));
out.virtualOrder = ended;

const started = performance.now();
await p.realScheduler.sleep(30);
out.sleptMs = performance.now() - started;
const reason = new Error("stopped");
try {
  await p.realScheduler.sleep(0, AbortSignal.abort(reason));
  out.preAborted = "resolved";
} catch (error) {
  out.preAborted = error === reason;
}

const holder = p.lockFile(lockPath);
try {
  const scheduler = p.virtualScheduler(1000);
  try {
    await p.waitForLock(lockPath, { timeoutMs: 500, scheduler });
    out.waitTimeout = "acquired";
  } catch (error) {
    out.waitTimeout = [error.reason, error.message.includes("timed out"), scheduler.now()];
  }
} finally {
  holder.release();
}

out.stripped = await p.stripTypes(${JSON.stringify(sample)});
const stripError = async (body) => {
  try {
    await p.stripTypes(body);
    return "no throw";
  } catch (error) {
    return [error instanceof p.TypeStripError, error.name, error.line, error.column];
  }
};
out.nonAsciiError = await stripError(${JSON.stringify("const s = '😀é'; const x: = 1;")});
out.openReturn = await stripError(${JSON.stringify("return (\n")});
const AsyncFunction = (async () => {}).constructor;
const thrownLine = async (body) => {
  const js = await p.stripTypes(body);
  try {
    await new AsyncFunction("inputs", js + "\\n//# sourceURL=yian-code:t")({});
    return "no throw";
  } catch (error) {
    const frame = /yian-code:t:(\\d+):\\d+/.exec(error.stack ?? "");
    return frame ? Number(frame[1]) : error.stack;
  }
};
out.stackHeader = (await thrownLine('throw new Error("h");')) - 1;
out.stackLine =
  (await thrownLine(${JSON.stringify('// a comment\n\nconst n: number = 1;\nthrow new Error("x" + n);')})) - out.stackHeader;
console.log(JSON.stringify(out));
`;
      const result = spawnSync(NODE!, ["--input-type=module", "-e", script], { encoding: "utf8" });
      expect(result.stderr).not.toContain("ExperimentalWarning");
      expect(result.status, result.stderr).toBe(0);
      const out = JSON.parse(result.stdout);
      expect(out.runtime).toBe("node");
      expect(out.row).toEqual({ a: Number.MAX_SAFE_INTEGER, b: "x", c: [0, 1, 255], d: null });
      expect(out.rowIsPlain).toBe(true);
      expect(out.blobIsBytes).toBe(true);
      expect(out.noRow).toBe(true);
      expect(out.bigintParam).toBe(5);
      expect(out.boundTypes).toEqual(["integer", "real"]);
      expect(out.foreignKeys).toBe(0);
      expect(out.doubleQuoted).toBe("x");
      expect(out.unsafeIntegers).toEqual([true, true]);
      expect(out.readonlyRows).toBe(1);
      expect(out.readonlyWriteThrows).toBe(true);
      expect(out.readonlyMissing).toEqual([14, false]);
      expect(out.keptRows).toEqual([{ a: 1 }]);
      expect(out.closeReleases).toBe("began");
      expect(out.keptAfterCloseThrows).toBe(true);
      expect(out.busyCode).toBe(5);
      expect(out.enoent).toEqual(["ENOENT", true]);
      expect(out.lockBusy).toBe("busy");
      expect(out.lockedWhileHeld).toBe(true);
      expect(out.lockedAfterRelease).toBe(false);
      expect(out.virtualOrder).toEqual([
        [500, 1500],
        [1000, 2000],
      ]);
      // Node's timers can fire a millisecond early against performance.now().
      expect(out.sleptMs).toBeGreaterThanOrEqual(29);
      expect(out.preAborted).toBe(true);
      expect(out.waitTimeout).toEqual(["busy", true, 1500]);
      expect(out.stripped).toBe(await stripTypes(sample));
      expect(out.nonAsciiError).toEqual([true, "TypeStripError", 1, 27]);
      expect(out.openReturn).toEqual([true, "TypeStripError", 1, 9]);
      expect(out.stackLine).toBe(4);
    },
    SPAWNED_MS,
  );

  test.skipIf(!NODE)("imports no bun: module statically", () => {
    // A quoted specifier, as in `import type`, would make Node consumers need Bun's
    // types; the module doc naming `bun:sqlite` in backticks is fine.
    expect(fs.readFileSync(path.join(ROOT, "dist/src/platform.d.ts"), "utf8")).not.toMatch(/["']bun:/);
    // The lazy require("bun:sqlite") is a call, not an import or re-export.
    expect(fs.readFileSync(BUILT, "utf8")).not.toMatch(/^\s*(?:import|export)\b[^;(]*["']bun:/m);
  });
});
