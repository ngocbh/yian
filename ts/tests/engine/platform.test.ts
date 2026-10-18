/**
 * Contracts for `src/platform.ts`: runtime detection and the SQLite wrapper that
 * `src/yian/memory/storage/sqlite.py`'s TypeScript port runs on, with the result
 * codes Python reads from `sqlite_errorcode`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { openSqlite, runtime, type Sqlite, sqliteErrorCode } from "../../src/platform.ts";

let dir: string;
let file: string;
let opened: Sqlite[];

/** Open `p` and close it after the test. */
function open(p: string, options?: { readonly?: boolean }): Sqlite {
  const db = openSqlite(p, options);
  opened.push(db);
  return db;
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
});

afterEach(() => {
  for (const db of opened) db.close();
  fs.rmSync(dir, { recursive: true, force: true });
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
    const db = openSqlite(file);
    db.exec("CREATE TABLE t (a INTEGER)");
    db.prepare("INSERT INTO t VALUES (?)").run(7);
    db.close();
    db.close();
    expect(open(file).prepare("SELECT a FROM t").all()).toEqual([{ a: 7 }]);
  });

  test("readonly reads and refuses writes", () => {
    const db = openSqlite(file);
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
    const a = openSqlite(file);
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
