/**
 * Contracts for `src/platform.ts`: runtime detection; the SQLite wrapper that
 * `src/yian/memory/storage/sqlite.py`'s TypeScript port runs on, with the result
 * codes Python reads from `sqlite_errorcode`; and the exclusive file lock that
 * replaces the `flock` calls in `runtime/log.py` and `llm_clients/auth/store.py`;
 * the scheduler that ports TOOL's chunked `_sleep` in `nodes/tool/node.py`, with a
 * virtual one for tests; and the waiting lock that ports the auth store's `_locked`.
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
  realScheduler,
  runtime,
  type Scheduler,
  SLEEP_CHUNK_MS,
  type Sqlite,
  sqliteErrorCode,
  virtualScheduler,
  waitForLock,
} from "../../src/platform.ts";

const realSetTimeout = globalThis.setTimeout;

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
    const started = performance.now();
    const error = await rejection(waitForLock(file, { timeoutMs: 500, scheduler }));
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
