/**
 * The one module holding runtime-specific APIs, so the rest of yian runs unchanged
 * on Node and Bun: runtime detection, a SQLite wrapper over `bun:sqlite` or
 * `node:sqlite`, loaded lazily for the runtime in use, and an exclusive file lock
 * built on it.
 */

import type { Database } from "bun:sqlite";
// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { createRequire } from "node:module";
import type { DatabaseSync } from "node:sqlite";

/** The JavaScript runtime yian is running on, detected once at load. */
export const runtime: "bun" | "node" = typeof Bun !== "undefined" ? "bun" : "node";

/** A value bound to a positional `?` parameter. */
export type SqliteParam = null | number | bigint | string | Uint8Array;

/** A value read from a column; integers are always safe `number`s. */
export type SqliteValue = null | number | string | Uint8Array;

/** One result row: a plain object from column name to value. */
export type SqliteRow = Record<string, SqliteValue>;

/** A prepared statement taking positional `?` parameters. */
export interface SqliteStatement {
  /** Return every result row, in order. */
  all(...params: SqliteParam[]): SqliteRow[];
  /** Return the first result row, or `undefined` when there's none. */
  get(...params: SqliteParam[]): SqliteRow | undefined;
  /** Run the statement, discarding any rows. */
  run(...params: SqliteParam[]): void;
}

/** An open SQLite connection. */
export interface Sqlite {
  /** Run one or more `;`-separated statements without parameters. */
  exec(sql: string): void;
  /** Return the statement for `sql`, prepared once per connection and cached by its text. */
  prepare(sql: string): SqliteStatement;
  /** Finalize the cached statements and close the connection; a second call does nothing. */
  close(): void;
}

/** The part of either driver's statement the wrapper uses; only Bun's has `finalize`. */
interface DriverStatement {
  all(...params: SqliteParam[]): unknown[];
  get(...params: SqliteParam[]): unknown;
  run(...params: SqliteParam[]): unknown;
  finalize?(): void;
}

/** The part of either driver's connection the wrapper uses. */
interface DriverDatabase {
  exec(sql: string): unknown;
  prepare(sql: string): DriverStatement;
  close(): void;
}

const require = createRequire(import.meta.url);

let openDriver: ((path: string, readonly: boolean) => DriverDatabase) | undefined;

/** Return the opener for this runtime's SQLite driver, loading the driver on first use. */
function driver(): (path: string, readonly: boolean) => DriverDatabase {
  if (openDriver) return openDriver;
  if (runtime === "bun") {
    const { Database: BunDatabase } = require("bun:sqlite") as { Database: typeof Database };
    openDriver = (path, readonly) =>
      new BunDatabase(path, { create: !readonly, readwrite: !readonly, readonly, strict: true, safeIntegers: true });
  } else {
    const { DatabaseSync: NodeDatabase } = requireNodeSqlite();
    openDriver = (path, readonly) => new NodeDatabase(path, { readOnly: readonly });
  }
  return openDriver;
}

/** Load `node:sqlite`, dropping only the experimental-feature warning Node 22 emits for it. */
function requireNodeSqlite(): { DatabaseSync: typeof DatabaseSync } {
  const emitWarning = process.emitWarning;
  process.emitWarning = ((warning: string | Error, ...rest: unknown[]) => {
    const option = rest[0];
    const type = typeof option === "string" ? option : (option as { type?: unknown } | undefined)?.type;
    const text = typeof warning === "string" ? warning : warning.message;
    if (type === "ExperimentalWarning" && text.startsWith("SQLite")) return;
    Reflect.apply(emitWarning, process, [warning, ...rest]);
  }) as typeof process.emitWarning;
  try {
    return require("node:sqlite") as { DatabaseSync: typeof DatabaseSync };
  } finally {
    process.emitWarning = emitWarning;
  }
}

/** Copy a driver row into a plain object, turning Bun's `bigint`s into safe numbers. */
function plainRow(row: object): SqliteRow {
  // Spread defines each column as an own property, even one named `__proto__`.
  const copy = { ...row } as Record<string, unknown>;
  for (const [key, value] of Object.entries(copy)) {
    if (typeof value !== "bigint") continue;
    if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new RangeError(`SQLite integer ${value} in column ${key} is outside the safe integer range`);
    }
    copy[key] = Number(value);
  }
  return copy as SqliteRow;
}

/**
 * Open the SQLite database file at `path`, creating it unless `readonly`.
 *
 * Integers read back as `number` on both runtimes, and one outside
 * `Number.isSafeInteger` throws `RangeError`. Driver errors pass through; read their
 * result code with `sqliteErrorCode`.
 */
export function openSqlite(path: string, { readonly = false }: { readonly?: boolean } = {}): Sqlite {
  const db = driver()(path, readonly);
  const statements = new Map<string, { statement: SqliteStatement; raw: DriverStatement }>();
  let closed = false;
  return {
    exec(sql) {
      db.exec(sql);
    },
    prepare(sql) {
      const cached = statements.get(sql);
      if (cached) return cached.statement;
      const raw = db.prepare(sql);
      const statement: SqliteStatement = {
        all: (...params) => raw.all(...params).map((row) => plainRow(row as object)),
        get: (...params) => {
          // Bun returns null for no row, Node undefined.
          const row = raw.get(...params);
          return row == null ? undefined : plainRow(row);
        },
        run: (...params) => {
          raw.run(...params);
        },
      };
      statements.set(sql, { statement, raw });
      return statement;
    },
    close() {
      if (closed) return;
      closed = true;
      // Bun keeps the connection and its locks open while any statement is unfinalized;
      // Node's close finalizes them itself.
      for (const { raw } of statements.values()) raw.finalize?.();
      db.close();
    },
  };
}

/**
 * Return the SQLite extended result code a driver error carries, or `undefined` for
 * any other value. Apply `& 0xff` for the primary code.
 */
export function sqliteErrorCode(error: unknown): number | undefined {
  if (typeof error !== "object" || error === null) return undefined;
  const e = error as Record<string, unknown>;
  const result = e.name === "SQLiteError" ? e.errno : e.code === "ERR_SQLITE_ERROR" ? e.errcode : undefined;
  return typeof result === "number" ? result : undefined;
}

/** Why `lockFile` refused a path. */
export type FileLockReason = "busy" | "unsafe" | "not-private" | "changed";

const lockMessages: Record<FileLockReason, string> = {
  busy: "is locked by another holder",
  unsafe: "is not a regular file with one link",
  "not-private": "is not private to its owner",
  changed: "was replaced while being locked",
};

/** `lockFile` refused `path` for `reason`. */
export class FileLockError extends Error {
  override name = "FileLockError";
  readonly path: string;
  readonly reason: FileLockReason;

  constructor(path: string, reason: FileLockReason, options?: ErrorOptions) {
    super(`lock file ${path} ${lockMessages[reason]}`, options);
    this.path = path;
    this.reason = reason;
  }
}

/** A held exclusive lock on `path`; `release` (or disposal) frees it, and a second call does nothing. */
export interface FileLock {
  readonly path: string;
  release(): void;
  [Symbol.dispose](): void;
}

/** Return `error`'s `code` when it's a Node system error. */
function errnoCode(error: unknown): unknown {
  return (error as NodeJS.ErrnoException | null)?.code;
}

/** Return whether `error` is SQLite's BUSY or LOCKED result. */
function isBusy(error: unknown): boolean {
  const code = sqliteErrorCode(error);
  return code !== undefined && ((code & 0xff) === 5 || (code & 0xff) === 6);
}

/**
 * Take the exclusive lock on the file at `path` without waiting, creating it empty
 * with mode 0600 when missing.
 *
 * The lock is a SQLite exclusive transaction, so the kernel frees it when the
 * process dies; the file stays 0 bytes, no sibling file appears, and the file is
 * never unlinked. Throws `FileLockError` with reason `"busy"` when another holder
 * has it, `"unsafe"` unless `path` is a regular file (and, with `private`, one with
 * exactly one link), `"not-private"` when `private` and the file isn't owned by this
 * user or grants group or other bits, and `"changed"` when `path` names another file
 * once locked. That last check reads the path, not SQLite's own descriptor. OS
 * errors (`ENOENT` for a missing directory) and SQLite errors (`SQLITE_NOTADB` for
 * a foreign file) propagate unchanged.
 */
export function lockFile(path: string, { private: isPrivate = false }: { private?: boolean } = {}): FileLock {
  // The file is never opened through fs once it exists: POSIX locks belong to the
  // process and inode, so closing any other descriptor on it drops SQLite's lock
  // (Node). Hence lstat for the checks, and an exclusive create that only a missing
  // file reaches.
  try {
    fs.lstatSync(path);
  } catch (error) {
    if (errnoCode(error) !== "ENOENT") throw error;
    const { O_CREAT, O_EXCL, O_NOFOLLOW, O_WRONLY } = fs.constants;
    try {
      fs.closeSync(fs.openSync(path, O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY, 0o600));
    } catch (createError) {
      // Another creator won the race; check its file like any existing one.
      if (errnoCode(createError) !== "EEXIST") throw createError;
    }
  }
  const before = fs.lstatSync(path, { bigint: true });
  if (!before.isFile() || (isPrivate && before.nlink !== 1n)) throw new FileLockError(path, "unsafe");
  if (isPrivate && (before.uid !== BigInt(process.getuid!()) || (before.mode & 0o077n) !== 0n)) {
    throw new FileLockError(path, "not-private");
  }
  const db = openSqlite(path);
  try {
    db.exec("PRAGMA busy_timeout = 0");
    // An in-memory rollback journal keeps the lock from ever creating a sibling file;
    // `OFF` is silently ignored by some builds. This step usually reports a holder.
    db.exec("PRAGMA journal_mode = MEMORY");
    db.exec("BEGIN EXCLUSIVE");
    const after = fs.lstatSync(path, { bigint: true });
    if (after.dev !== before.dev || after.ino !== before.ino) throw new FileLockError(path, "changed");
  } catch (error) {
    // Close on every failure, so a caller polling a busy lock leaks nothing.
    db.close();
    if (isBusy(error)) throw new FileLockError(path, "busy", { cause: error });
    throw error;
  }
  return { path, release: () => db.close(), [Symbol.dispose]: () => db.close() };
}

/**
 * Return whether another connection holds the lock on `path`.
 *
 * A missing path or one that isn't a regular file is `false`, and is never opened
 * or created. Otherwise this reads the schema under a shared lock, so two probes
 * never see each other as held; a `lockFile` racing that brief read can still fail
 * busy. Errors other than busy, a vanished path and a foreign file propagate.
 */
export function isLocked(path: string): boolean {
  try {
    if (!fs.lstatSync(path).isFile()) return false;
  } catch (error) {
    if (errnoCode(error) === "ENOENT") return false;
    throw error;
  }
  let db: Sqlite | undefined;
  try {
    db = openSqlite(path, { readonly: true });
    db.exec("PRAGMA busy_timeout = 0");
    db.prepare("SELECT count(*) AS n FROM sqlite_master").get();
    return false;
  } catch (error) {
    if (isBusy(error)) return true;
    const code = (sqliteErrorCode(error) ?? 0) & 0xff;
    if (code === 14 || code === 26) return false;
    throw error;
  } finally {
    db?.close();
  }
}
