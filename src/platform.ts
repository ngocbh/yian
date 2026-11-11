/**
 * The one module holding runtime-specific APIs, so the rest of yian runs unchanged
 * on Node and Bun: runtime detection, whether this is a compiled binary, a SQLite
 * wrapper over `bun:sqlite` or `node:sqlite`, loaded lazily for the runtime in use,
 * an exclusive file lock built on it, `syncFolder` for durable folder entries, the OS
 * error, missing-path, open-file path, path-confinement and home-directory helpers, the
 * scheduler that clocks and timed waits go through, the TypeScript stripper for CODE
 * bodies, loaded lazily too, `safeFetch`, the one way out to the network: no redirects,
 * yian's User-Agent, and a warning when TLS checks are off, and `warn`, the path every
 * yian warning takes.
 */

import type { Database } from "bun:sqlite";
import { AsyncLocalStorage } from "node:async_hooks";
// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { createRequire } from "node:module";
import net from "node:net";
// Default import, like `fs`, so tests can spy on `userInfo`.
import os from "node:os";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import type { DatabaseSync } from "node:sqlite";

import { type Env, processEnv } from "./defaults.ts";

/** The JavaScript runtime yian is running on, detected once at load. */
export const runtime: "bun" | "node" = typeof Bun !== "undefined" ? "bun" : "node";

/**
 * Whether this process is a standalone binary built by `bun build --compile`, whose
 * `process.argv[1]` names a file inside the binary rather than a script to re-run.
 */
export const compiled: boolean = runtime === "bun" && Bun.isStandaloneExecutable === true;

/** The runtime and its version (`Bun 1.4.2`), and the OS and CPU (`linux x64`), for a report. */
export function hostPlatform(): { runtime: string; platform: string } {
  const version = runtime === "bun" ? Bun.version : process.versions.node;
  return {
    runtime: `${runtime === "bun" ? "Bun" : "Node"} ${version}`,
    platform: `${process.platform} ${process.arch}`,
  };
}

/**
 * A value bound to a positional `?` parameter. A `number` passing
 * `Number.isSafeInteger` binds as INTEGER; any other
 * `number` binds as REAL.
 */
export type SqliteParam = null | number | bigint | string | Uint8Array;

/** A value read from a column; integers are always safe `number`s. */
export type SqliteValue = null | number | string | Uint8Array;

/** One result row: a plain object from column name to value. */
export type SqliteRow = Record<string, SqliteValue>;

/** What `SqliteStatement.run` reports, the same on both runtimes. */
export interface SqliteRunResult {
  /**
   * The rows an INSERT, UPDATE or DELETE changed. Meaningful only after those: for any other
   * statement Bun reports 0 and Node the previous one's count.
   */
  changes: number;
  /**
   * The connection's most recent successful INSERT rowid, 0 when there's been none: a `number`
   * when it passes `Number.isSafeInteger`, else the exact `bigint`.
   */
  lastInsertRowid: number | bigint;
}

/**
 * A prepared statement taking positional `?` parameters. Passing some but too few
 * throws on Bun but binds NULL on Node (passing none binds NULL on both); the wrapper
 * doesn't normalize that programming error.
 */
export interface SqliteStatement {
  /** Return every result row, in order. */
  all(...params: SqliteParam[]): SqliteRow[];
  /** Return the first result row, or `undefined` when there's none. */
  get(...params: SqliteParam[]): SqliteRow | undefined;
  /** Run the statement, discarding any rows, and return what it changed. */
  run(...params: SqliteParam[]): SqliteRunResult;
}

/** An open SQLite connection. */
export interface Sqlite {
  /** Run one or more `;`-separated statements without parameters. */
  exec(sql: string): void;
  /** Return the statement for `sql`, prepared once per connection and cached by its text. */
  prepare(sql: string): SqliteStatement;
  /** Whether a transaction is open on this connection. */
  readonly inTransaction: boolean;
  /** Finalize the cached statements and close the connection; a second call does nothing. */
  close(): void;
}

/** The part of either driver's statement the wrapper uses; only Bun's has `finalize`. */
interface DriverStatement {
  all(...params: SqliteParam[]): unknown[];
  get(...params: SqliteParam[]): unknown;
  run(...params: SqliteParam[]): { changes: number | bigint; lastInsertRowid: number | bigint };
  finalize?(): void;
  setReadBigInts?(enabled: boolean): void;
}

/** The part of either driver's connection the wrapper uses. */
interface DriverDatabase {
  exec(sql: string): unknown;
  prepare(sql: string): DriverStatement;
  close(): void;
}

/** A driver connection and the way to ask it whether a transaction is open, which the drivers name differently. */
interface DriverConnection {
  db: DriverDatabase;
  inTransaction(): boolean;
}

const require = createRequire(import.meta.url);

let openDriver: ((path: string, readonly: boolean) => DriverConnection) | undefined;

/** Return the opener for this runtime's SQLite driver, loading the driver on first use. */
function driver(): (path: string, readonly: boolean) => DriverConnection {
  if (openDriver) return openDriver;
  if (runtime === "bun") {
    const { Database: BunDatabase } = require("bun:sqlite") as { Database: typeof Database };
    openDriver = (path, readonly) => {
      const db = new BunDatabase(path, {
        create: !readonly,
        readwrite: !readonly,
        readonly,
        strict: true,
        safeIntegers: true,
      });
      return { db, inTransaction: () => db.inTransaction };
    };
  } else {
    const { DatabaseSync: NodeDatabase } = requireNodeSqlite();
    // Node's defaults differ from Bun's, which leaves foreign keys off and accepts
    // double-quoted string literals.
    openDriver = (path, readonly) => {
      const db = new NodeDatabase(path, {
        readOnly: readonly,
        enableForeignKeyConstraints: false,
        enableDoubleQuotedStringLiterals: true,
      });
      return { db, inTransaction: () => db.isTransaction };
    };
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

/** Map safe-integer `number`s to `bigint`, which both drivers bind as INTEGER; Node binds any `number` as REAL. */
function bindable(params: SqliteParam[]): SqliteParam[] {
  return params.map((value) => (typeof value === "number" && Number.isSafeInteger(value) ? BigInt(value) : value));
}

/** Return a driver integer as a `number` when it is a safe integer, else as the exact `bigint`. */
function numberIfSafe(value: number | bigint): number | bigint {
  if (typeof value === "number") return value;
  return value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER) ? value : Number(value);
}

/** Return a driver integer as a safe number, throwing `RangeError` naming `what` when it isn't one. */
function safeNumber(value: number | bigint, what: string): number {
  const result = numberIfSafe(value);
  if (typeof result === "bigint") {
    throw new RangeError(`SQLite integer ${result} in ${what} is outside the safe integer range`);
  }
  return result;
}

/** Copy a driver row into a plain object, turning Bun's `bigint`s into safe numbers. */
function plainRow(row: object): SqliteRow {
  // Spread defines each column as an own property, even one named `__proto__`.
  const copy = { ...row } as Record<string, unknown>;
  for (const [key, value] of Object.entries(copy)) {
    if (typeof value === "bigint") copy[key] = safeNumber(value, `column ${key}`);
  }
  return copy as SqliteRow;
}

/**
 * Open the SQLite database file at `path`, creating it unless `readonly`.
 *
 * Integers in rows read back as `number` on both runtimes, and one outside
 * `Number.isSafeInteger` throws `RangeError`. On both runtimes foreign keys
 * are off and double-quoted string literals are accepted. Driver errors pass
 * through; read their result code with `sqliteErrorCode`.
 */
export function openSqlite(path: string, { readonly = false }: { readonly?: boolean } = {}): Sqlite {
  const { db, inTransaction } = driver()(path, readonly);
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
      // Node otherwise throws on an unsafe integer, even a rowid `run` reports; only Bun's
      // statements lack the setter, and they already read bigints.
      raw.setReadBigInts?.(true);
      const statement: SqliteStatement = {
        all: (...params) => raw.all(...bindable(params)).map((row) => plainRow(row as object)),
        get: (...params) => {
          // Bun returns null for no row, Node undefined.
          const row = raw.get(...bindable(params));
          return row == null ? undefined : plainRow(row);
        },
        run: (...params) => {
          const { changes, lastInsertRowid } = raw.run(...bindable(params));
          return {
            changes: safeNumber(changes, "changes"),
            lastInsertRowid: numberIfSafe(lastInsertRowid),
          };
        },
      };
      statements.set(sql, { statement, raw });
      return statement;
    },
    get inTransaction() {
      return inTransaction();
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

/** `lockFile` refused `path` for `reason`; the message reads "lock file <path> <detail>". */
export class FileLockError extends Error {
  override name = "FileLockError";
  readonly path: string;
  readonly reason: FileLockReason;

  constructor(path: string, reason: FileLockReason, detail: string, options?: ErrorOptions) {
    super(`lock file ${path} ${detail}`, options);
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

const SQLITE_BUSY = 5;
const SQLITE_LOCKED = 6;
const SQLITE_CORRUPT = 11;
const SQLITE_CANTOPEN = 14;
const SQLITE_NOTADB = 26;

/**
 * How long `lockFile` lets SQLite retry before it reports busy. A probe's shared lock lasts
 * microseconds, so this rides it out; a real holder keeps the lock and is still reported busy.
 */
const LOCK_BUSY_WAIT_MS = 50;

/** Return the SQLite primary result code `error` carries, or 255 for any other value. */
function primaryCode(error: unknown): number {
  return (sqliteErrorCode(error) ?? -1) & 0xff;
}

/**
 * Take the exclusive lock on the file at `path`, creating it empty with mode 0600
 * when missing. It waits up to 50 ms for the lock, so a probe (`isLocked`) or another
 * contender mid-attempt never makes it fail; a holder that keeps the lock past that
 * is reported busy.
 *
 * The lock is a SQLite exclusive transaction, so the kernel frees it when the
 * process dies; the file stays 0 bytes, no sibling file appears, and the file is
 * never unlinked. Throws `FileLockError` with reason `"busy"` when another holder
 * has it, `"unsafe"` unless `path` is a regular file (and, with `private`, one with
 * exactly one link) or when it holds data SQLite can't read as a database (with the
 * SQLite error as `cause`), `"not-private"` when `private` and the file isn't owned
 * by this user or grants group or other bits, and `"changed"` when `path` names
 * another file (or none) once locked. OS errors (`ENOENT` for a missing directory)
 * and other SQLite errors propagate unchanged.
 *
 * The "changed" check reads the path, not SQLite's own descriptor. So a FIFO swapped
 * in between the checks and SQLite's open would block that open, and a path deleted
 * in that window is recreated by SQLite with its default mode (0644): this call
 * reports `"changed"`, but later `private` locks fail `"not-private"` until the file
 * is removed.
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
  if (!before.isFile()) throw new FileLockError(path, "unsafe", "is not a regular file");
  if (isPrivate && before.nlink !== 1n) throw new FileLockError(path, "unsafe", "has more than one link");
  // `getuid` exists only on POSIX, like the `flock` this lock replaces.
  if (isPrivate && (before.uid !== BigInt(process.getuid!()) || (before.mode & 0o077n) !== 0n)) {
    throw new FileLockError(path, "not-private", "is not private to its owner");
  }
  const db = openSqlite(path);
  try {
    db.exec(`PRAGMA busy_timeout = ${LOCK_BUSY_WAIT_MS}`);
    // An in-memory rollback journal keeps the lock from ever creating a sibling file;
    // `OFF` is silently ignored by some builds. With a holder present, this pragma
    // (not BEGIN) is usually what fails busy, hence the catch around every step.
    db.exec("PRAGMA journal_mode = MEMORY");
    db.exec("BEGIN EXCLUSIVE");
    let after: fs.BigIntStats;
    try {
      after = fs.lstatSync(path, { bigint: true });
    } catch (error) {
      if (errnoCode(error) !== "ENOENT") throw error;
      throw new FileLockError(path, "changed", "was removed while being locked", { cause: error });
    }
    if (after.dev !== before.dev || after.ino !== before.ino) {
      throw new FileLockError(path, "changed", "was replaced while being locked");
    }
  } catch (error) {
    // Close on every failure, so a caller polling a busy lock leaks nothing.
    db.close();
    const code = primaryCode(error);
    if (code === SQLITE_BUSY || code === SQLITE_LOCKED) {
      throw new FileLockError(path, "busy", "is locked by another holder", { cause: error });
    }
    if (code === SQLITE_NOTADB || code === SQLITE_CORRUPT) {
      throw new FileLockError(path, "unsafe", "holds data yian didn't write; remove it if nothing holds the lock", {
        cause: error,
      });
    }
    throw error;
  }
  return { path, release: () => db.close(), [Symbol.dispose]: () => db.close() };
}

/**
 * Return whether another connection holds the lock on `path`.
 *
 * A missing path or one that isn't a regular file is `false`, and is never opened
 * or created. Otherwise this reads the schema under a shared lock, without waiting,
 * so two probes never see each other as held; a `lockFile` racing that brief read
 * waits it out. A foreign file is `false`. When SQLite can't open the file, a path that has
 * since vanished is `false`; any other cause (such as no read permission) rethrows
 * the driver error, so a held lock is never reported free. Other errors propagate.
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
    db.prepare("SELECT count(*) FROM sqlite_master").get();
    return false;
  } catch (error) {
    const code = primaryCode(error);
    if (code === SQLITE_BUSY || code === SQLITE_LOCKED) return true;
    if (code === SQLITE_NOTADB) return false;
    if (code === SQLITE_CANTOPEN) {
      // CANTOPEN also covers EACCES and EMFILE; only a vanished path means free.
      try {
        fs.lstatSync(path);
      } catch (statError) {
        if (errnoCode(statError) === "ENOENT") return false;
      }
    }
    throw error;
  } finally {
    db?.close();
  }
}

/**
 * The errors that mean this file system or platform can't sync a folder: Linux gives `EINVAL` for
 * a file system without directory sync, and some file systems give `ENOTSUP` or `EOPNOTSUPP`.
 * Windows can't open a folder (`EISDIR`, `EPERM`); elsewhere those are real failures.
 */
const FOLDER_SYNC_UNSUPPORTED = new Set([
  "EINVAL",
  "ENOTSUP",
  "EOPNOTSUPP",
  ...(process.platform === "win32" ? ["EISDIR", "EPERM"] : []),
]);

/**
 * Flush the entries of folder `path` to disk, so a file created, renamed or removed in it
 * survives a power loss. Does nothing where the file system or platform can't sync a folder; any
 * other error (`ENOENT` for a missing folder) propagates.
 */
export function syncFolder(path: string): void {
  let fd: number;
  try {
    fd = fs.openSync(path, "r");
  } catch (error) {
    if (FOLDER_SYNC_UNSUPPORTED.has(errnoCode(error) as string)) return;
    throw error;
  }
  try {
    fs.fsyncSync(fd);
  } catch (error) {
    if (!FOLDER_SYNC_UNSUPPORTED.has(errnoCode(error) as string)) throw error;
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Call `onEnd` once the writer of the pipe or socket at descriptor `fd` closes its end, however
 * it ended (a SIGKILL or a crash included); what it writes is discarded. A descriptor that isn't
 * open, or isn't a pipe or a socket, is left alone. The watch may keep the process up, so a
 * process that uses it ends with `process.exit`.
 */
export function onPipeEnd(fd: number, onEnd: () => void): void {
  try {
    const stats = fs.fstatSync(fd);
    if (!stats.isFIFO() && !stats.isSocket()) return;
  } catch {
    return;
  }
  // Bun's `net.Socket` over a descriptor never sees the end; Node's file stream would hold a
  // thread of its pool in a read for as long as the watch lasts.
  const stream =
    runtime === "bun" ? fs.createReadStream("", { fd }) : new net.Socket({ fd, readable: true, writable: false });
  let ended = false;
  const end = () => {
    if (ended) return;
    ended = true;
    onEnd();
  };
  stream.on("data", () => {});
  stream.once("end", end);
  stream.once("close", end);
  stream.on("error", end);
}

/** @internal Whether `error` is a Node SystemError (an OS error). */
export function isSystemError(error: unknown): error is NodeJS.ErrnoException & { code: string } {
  if (!(error instanceof Error)) return false;
  const { code, syscall, errno } = error as NodeJS.ErrnoException;
  return typeof code === "string" && (typeof syscall === "string" || typeof errno === "number");
}

/** The errors that make a path read as missing, so neither a file nor a directory. */
const MISSING_PATH_CODES = new Set(["ENOENT", "ENOTDIR", "ELOOP", "EBADF"]);

/** @internal Whether `error` is one of the OS errors that make a path read as missing. */
export function isMissingPath(error: unknown): boolean {
  return isSystemError(error) && MISSING_PATH_CODES.has(error.code);
}

/** @internal `fs.statSync`, or `null` for the errors that make a path read as missing. */
export function statOrNull(path: string): fs.Stats | null {
  try {
    return fs.statSync(path);
  } catch (error) {
    if (isMissingPath(error)) return null;
    throw error;
  }
}

/** The first bytes of a regular file and the file's device and inode, as `readRegularFile` reads them. */
export interface RegularFileHead {
  readonly bytes: Uint8Array;
  readonly dev: bigint;
  readonly ino: bigint;
}

/**
 * @internal Read at most `limit + 1` bytes of the file at `path` through one descriptor, and
 * return them with the file's device and inode; `null` when the file opened isn't a regular file.
 * The open doesn't block, so a FIFO can't hang it, and the read stops at `limit + 1` bytes
 * however large the file or whatever size `stat` gives it (a `/proc` file's is 0).
 *
 * @throws a system error when the file can't be opened or read.
 */
export async function readRegularFile(path: string, limit: number): Promise<RegularFileHead | null> {
  // Windows has no O_NONBLOCK, and no FIFO a path opens.
  const handle = await fs.promises.open(path, fs.constants.O_RDONLY | (fs.constants.O_NONBLOCK ?? 0));
  try {
    const stat = await handle.stat({ bigint: true });
    if (!stat.isFile()) return null;
    const chunks: Uint8Array[] = [];
    let size = 0;
    while (size <= limit) {
      const chunk = new Uint8Array(Math.min(64 * 1024, limit + 1 - size));
      const { bytesRead } = await handle.read(chunk, 0, chunk.length, null);
      if (bytesRead === 0) break;
      chunks.push(chunk.subarray(0, bytesRead));
      size += bytesRead;
    }
    return { bytes: Buffer.concat(chunks), dev: stat.dev, ino: stat.ino };
  } finally {
    await handle.close();
  }
}

/**
 * @internal The path the OS holds for open file `fd`, read from the descriptor rather than
 * resolved again from a path, or `null` where the runtime can't tell (Node on macOS, Linux
 * without `/proc`, Windows). On Linux an unlinked file's path ends in ` (deleted)`.
 */
export function openFilePath(fd: number): string | null {
  try {
    // One readlink: resolving the target again would walk paths another process can change.
    if (process.platform === "linux") return fs.readlinkSync(`/proc/self/fd/${fd}`);
    if (process.platform === "darwin") {
      // Bun answers from the descriptor (`F_GETPATH`); Node walks `/dev/fd` and returns a `/dev` path.
      const path = fs.realpathSync(`/dev/fd/${fd}`);
      return path.startsWith("/dev/") ? null : path;
    }
  } catch (error) {
    if (!isSystemError(error)) throw error;
  }
  return null;
}

/**
 * @internal Whether a path relative to a root (`relative(root, file)`) leaves that root: `..`, a
 * `../` prefix, or an absolute path (another drive).
 */
export function leavesRoot(rel: string): boolean {
  return rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel);
}

/**
 * @internal The real path of the longest existing prefix of `path`, with the rest appended unchanged.
 */
export function resolvedPath(path: string): string {
  const abs = resolve(path);
  let candidate = abs;
  for (;;) {
    try {
      const real = fs.realpathSync(candidate);
      return join(real, relative(candidate, abs));
    } catch (error) {
      if (!isSystemError(error)) throw error;
      const parent = dirname(candidate);
      if (parent === candidate) return abs;
      candidate = parent;
    }
  }
}

/**
 * @internal Return the user's home directory: `HOME` from `env` when set and non-empty, else the
 * account's home directory (`os.userInfo().homedir`), which throws when the account has none.
 */
export function homeDir(env: Env): string {
  return env("HOME") || os.userInfo().homedir;
}

/** A clock and a cancellable relative sleep; tests swap in `virtualScheduler`. */
export interface Scheduler {
  /** Return the wall-clock time in epoch milliseconds. */
  now(): number;
  /**
   * Resolve after `ms` milliseconds, or reject with `signal.reason` once `signal`
   * aborts, including when it already has (for any `ms`). Otherwise `ms <= 0`
   * resolves without waiting, and a non-finite `ms` rejects with `RangeError`.
   */
  sleep(ms: number, signal?: AbortSignal): Promise<void>;
}

/** The longest single timer `realScheduler.sleep` arms (one hour); longer sleeps run in chunks. */
export const SLEEP_CHUNK_MS = 3_600_000;

/**
 * Throw `signal.reason` once aborted, or `RangeError` for a non-finite `ms`; return
 * whether a sleep of `ms` must wait.
 */
function mustWait(ms: number, signal: AbortSignal | undefined): boolean {
  signal?.throwIfAborted();
  if (ms <= 0) return false;
  if (!Number.isFinite(ms)) throw new RangeError(`sleep needs a finite number of milliseconds, got ${ms}`);
  return true;
}

/** Wait one timer of `ms`, settling early with `signal.reason` on abort. */
function timer(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    signal?.throwIfAborted();
    const onAbort = () => {
      // Clearing the timer keeps an aborted sleep from holding the process open.
      clearTimeout(id);
      signal?.removeEventListener("abort", onAbort);
      reject(signal?.reason);
    };
    const id = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort);
  });
}

/**
 * The scheduler on the system clock and `setTimeout`.
 *
 * `sleep` is relative and never reads `now()`: it arms timers of at most
 * `SLEEP_CHUNK_MS`, subtracting each from what's left, since runtimes clamp delays
 * above 2^31 - 1 ms. Fractional `ms` passes to `setTimeout` unrounded, so the
 * runtime's timer granularity (whole milliseconds) sets the exact delay. Callers
 * waiting for a wall-clock deadline re-read `now()` between chunks themselves.
 */
export const realScheduler: Scheduler = {
  now: () => Date.now(),
  async sleep(ms, signal) {
    if (!mustWait(ms, signal)) return;
    while (ms > SLEEP_CHUNK_MS) {
      await timer(SLEEP_CHUNK_MS, signal);
      ms -= SLEEP_CHUNK_MS;
    }
    await timer(ms, signal);
  },
};

/**
 * Return a scheduler whose clock starts at `start` and moves only to the deadline of
 * its earliest sleeper.
 *
 * Each advance waits one real macrotask first, so whatever a woken sleeper does next
 * without real I/O (including registering a new sleep) settles before the clock
 * moves again. Sleepers with the same deadline wake in registration order. The
 * limit: work that awaits real I/O or real timers between virtual sleeps may see the
 * clock advance first.
 */
export function virtualScheduler(start = 0): Scheduler {
  let now = start;
  // Kept ordered by deadline, ties in registration order, so the earliest is first.
  const sleepers: { deadline: number; wake: () => void }[] = [];
  let ticking = false;

  /** Schedule the next advance unless one is pending or nobody sleeps. */
  function tick(): void {
    if (ticking || sleepers.length === 0) return;
    ticking = true;
    setTimeout(() => {
      ticking = false;
      const earliest = sleepers.shift();
      if (earliest) {
        now = earliest.deadline;
        earliest.wake();
      }
      tick();
    }, 0);
  }

  return {
    now: () => now,
    async sleep(ms, signal) {
      if (!mustWait(ms, signal)) return;
      await new Promise<void>((resolve, reject) => {
        const onAbort = () => {
          const at = sleepers.indexOf(sleeper);
          if (at !== -1) sleepers.splice(at, 1);
          signal?.removeEventListener("abort", onAbort);
          reject(signal?.reason);
        };
        const sleeper = {
          deadline: now + ms,
          wake: () => {
            signal?.removeEventListener("abort", onAbort);
            resolve();
          },
        };
        const later = sleepers.findIndex((other) => other.deadline > sleeper.deadline);
        sleepers.splice(later === -1 ? sleepers.length : later, 0, sleeper);
        signal?.addEventListener("abort", onAbort);
        tick();
      });
    },
  };
}

/** How long `waitForLock` sleeps between tries on a busy lock. */
const LOCK_POLL_MS = 100;

/**
 * Take the exclusive lock on `path` as `lockFile` does, retrying every 100 ms while
 * another holder has it.
 *
 * Gives up with `FileLockError` (reason `"busy"`) once its sleeps and `lockFile`'s
 * 50 ms waits add up to `timeoutMs`, after at least one try; it never reads
 * `scheduler.now()`, so a clock step doesn't change the wait. The default `Infinity` waits for as long as it takes. Each try first checks
 * `signal`, and an abort rejects with its reason. A `NaN` or negative `timeoutMs`
 * rejects with `RangeError`. Every other `lockFile` failure propagates at once.
 */
export async function waitForLock(
  path: string,
  {
    timeoutMs = Number.POSITIVE_INFINITY,
    signal,
    scheduler = realScheduler,
    private: isPrivate = false,
  }: { timeoutMs?: number; signal?: AbortSignal; scheduler?: Scheduler; private?: boolean } = {},
): Promise<FileLock> {
  if (Number.isNaN(timeoutMs) || timeoutMs < 0) {
    throw new RangeError(`waitForLock needs a timeout of 0 or more milliseconds, got ${timeoutMs}`);
  }
  let remaining = timeoutMs;
  for (;;) {
    signal?.throwIfAborted();
    try {
      return lockFile(path, { private: isPrivate });
    } catch (error) {
      if (!(error instanceof FileLockError) || error.reason !== "busy") throw error;
      remaining -= LOCK_BUSY_WAIT_MS;
      if (remaining <= 0) {
        throw new FileLockError(path, "busy", `timed out after ${timeoutMs} ms waiting for another holder`, {
          cause: error,
        });
      }
      const step = Math.min(LOCK_POLL_MS, remaining);
      await scheduler.sleep(step, signal);
      remaining -= step;
    }
  }
}

/** Opens the function wrapping a CODE body, on the body's first line so line numbers hold. */
const STRIP_PREFIX = "async function __yian(inputs) {";
const STRIP_SUFFIX = "\n}";

type TransformSync = typeof import("amaro").transformSync;

/**
 * A CODE body amaro can't strip: a syntax error, or TypeScript that strip-only mode
 * doesn't support, such as `enum`. `message` is amaro's; `line` and `column` are
 * 1-based in the body, and `column` counts UTF-16 code units. A position past the
 * end of the body (an unclosed construct) is reported at the end of the last
 * non-blank line.
 */
export class TypeStripError extends SyntaxError {
  override name = "TypeStripError";
  readonly line: number;
  readonly column: number;

  constructor(message: string, line: number, column: number) {
    super(message);
    this.line = line;
    this.column = column;
  }
}

/** The error amaro throws for a body it can't parse or strip. */
interface AmaroError {
  code: "InvalidSyntax" | "UnsupportedSyntax";
  message: string;
  startLine: number;
  startColumn: number;
}

/** Return whether `error` has the shape of amaro's syntax error. */
function isAmaroError(error: unknown): error is AmaroError {
  const e = error as Partial<AmaroError> | null | undefined;
  return (
    (e?.code === "InvalidSyntax" || e?.code === "UnsupportedSyntax") &&
    typeof e.startLine === "number" &&
    typeof e.startColumn === "number"
  );
}

/** Return how many display columns amaro gives the code point `ch`. */
function displayWidth(ch: string, transformSync: TransformSync): number {
  if (ch >= " " && ch <= "~") return 1;
  // The `)` always fails, at the column the comment before it ends on; the 4 is the
  // columns of `/*` and `*/`.
  try {
    transformSync(`/*${ch}*/)`, { mode: "strip-only" });
  } catch (error) {
    if (!isAmaroError(error)) throw error;
    return error.startColumn - 4;
  }
  throw new Error("unreachable: ')' always fails");
}

/** Return the UTF-16 index in `line` of amaro's 0-based display column `column`. */
function utf16Index(line: string, column: number, transformSync: TransformSync): number {
  let width = 0;
  let index = 0;
  for (const ch of line) {
    const next = width + displayWidth(ch, transformSync);
    if (next > column) break;
    width = next;
    index += ch.length;
  }
  return index;
}

/** Return `error`, from stripping the wrapped `body`, as a `TypeStripError` in body coordinates. */
function toTypeStripError(error: AmaroError, body: string, transformSync: TransformSync): TypeStripError {
  const lines = body.split(/\r\n|\r|\n/);
  if (error.startLine > lines.length) {
    // An unclosed construct is reported at the wrapper's closing `}`; report it on the last
    // line with text instead, which such an error always has.
    let last = lines.length - 1;
    while (last > 0 && lines[last]!.trim() === "") last--;
    return new TypeStripError(error.message, last + 1, lines[last]!.length + 1);
  }
  const column = error.startLine === 1 ? Math.max(0, error.startColumn - STRIP_PREFIX.length) : error.startColumn;
  return new TypeStripError(
    error.message,
    error.startLine,
    utf16Index(lines[error.startLine - 1]!, column, transformSync) + 1,
  );
}

/**
 * Strip the TypeScript types from a CODE body (a function body reading `inputs`,
 * which may use top-level `return` and `await`) and return the JavaScript body.
 *
 * Types become whitespace, so the result has the body's UTF-16 length and newline
 * offsets, and every line keeps its number. A body amaro can't parse or strip throws
 * `TypeStripError`; anything else amaro throws propagates unchanged, such as the
 * WASM `RuntimeError` amaro raises for a syntax error past display column 65535 of
 * its line, so callers report a non-`TypeStripError` at the whole body.
 *
 * Only text comes back. The caller builds the function, such as
 * `new AsyncFunction("inputs", js + "\n//# sourceURL=yian-code:<id>")`, and that
 * constructor is what stops a body that closes the wrapper from escaping it. Some
 * errors appear only then, with no position: `let inputs = 1`, `delete inputs`,
 * `import.meta`, and `arguments = 1` (strict mode).
 *
 * Async because amaro is large and slow to import, so it loads on first use through
 * a literal `import()`, which `bun build --compile` still bundles.
 */
export async function stripTypes(body: string): Promise<string> {
  // The module cache makes every call after the first cheap.
  const { transformSync } = await import("amaro");
  // A bare body with a top-level `return` or `await` doesn't parse, hence the wrapper.
  const text = STRIP_PREFIX + body + STRIP_SUFFIX;
  try {
    const { code } = transformSync(text, { mode: "strip-only" });
    return code.slice(STRIP_PREFIX.length, code.length - STRIP_SUFFIX.length);
  } catch (error) {
    if (!isAmaroError(error)) throw error;
    throw toTypeStripError(error, body, transformSync);
  }
}

/** What a `withWarnings` handler is given: a warning's text and its type, if any. */
type WarningHandler = (message: string, type: string | undefined) => void;

const warningHandlers = new AsyncLocalStorage<WarningHandler>();

/**
 * Raise a yian warning: to the innermost `withWarnings`
 * handler around this call, else through `process.emitWarning` (with `type` when given).
 */
export function warn(message: string, type?: string): void {
  const handler = warningHandlers.getStore();
  if (handler !== undefined) handler(message, type);
  else if (type === undefined) process.emitWarning(message);
  else process.emitWarning(message, type);
}

/**
 * Run `fn`, sending every `warn` raised inside it, across its awaits too, to `handler` instead of
 * `process.emitWarning`; return what `fn` returns.
 */
export function withWarnings<T>(handler: WarningHandler, fn: () => T): T {
  return warningHandlers.run(handler, fn);
}

/** The raw sender `safeFetch` calls: the runtime's fetch, a cassette, or a test fake. */
export type Send = (input: string | URL, init: RequestInit) => Promise<Response>;

/** yian's version; a test keeps it equal to `package.json`'s. */
export const VERSION = "0.0.4";

/** The `User-Agent` every request carries unless its caller sets one. */
export const USER_AGENT = `yian/${VERSION}`;

/**
 * The hosts a credential may be sent to over plain `http`: to any other host it would cross the
 * network in cleartext. IPv6 is without brackets.
 */
export const LOOPBACK_HOSTS: readonly string[] = ["localhost", "127.0.0.1", "::1"];

/** A request got a 3xx; its message names the target only by scheme, host and port. */
export class RedirectError extends Error {
  override name = "RedirectError";
  /** The HTTP status. */
  readonly status: number;
  /** The `redirectDetail` of the reply's `Location`. */
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(`HTTP ${status} ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Name a redirect's target by scheme, host and port only; the path, query and user info
 * can carry tokens, so they never reach the message.
 */
export function redirectDetail(location: string | null): string {
  if (!location) return "redirect without a Location header";
  const text = location.trim();
  const scheme = /^[A-Za-z][A-Za-z0-9+.-]*:/.exec(text)?.[0] ?? "";
  const rest = text.slice(scheme.length);
  // Absolute only when `//` comes before a non-empty authority; the URL parser alone would read
  // `http:///x` or `http:x` as host `x`. `//user@/x` and `//:80/x` give "unparsable" here.
  if (!/^\/\/[^/?#]/.test(rest)) return "redirect to a relative Location";
  let url: URL;
  try {
    url = new URL((scheme || "http:") + rest);
  } catch {
    return "redirect to an unparsable Location";
  }
  if (!url.hostname) return "redirect to a relative Location";
  if (!scheme) return "redirect to a Location without a scheme";
  return `redirect to ${url.protocol}//${url.hostname}` + (url.port ? `:${url.port}` : "");
}

/**
 * Return the warning for a `NODE_TLS_REJECT_UNAUTHORIZED` value, or `null` unless it is `"0"`,
 * the value that turns certificate checks off in both runtimes.
 */
export function insecureTlsWarning(value: string | undefined): string | null {
  return value === "0"
    ? "NODE_TLS_REJECT_UNAUTHORIZED=0 turns off TLS certificate checks for every request; unset it and trust a private CA with NODE_EXTRA_CA_CERTS instead"
    : null;
}

let warnedInsecureTls = false;

/**
 * Send one HTTP request through `send` (default the runtime's fetch) without following
 * redirects: every outbound request goes through here.
 *
 * Refuses, before sending and without echoing the URL or headers, a URL that doesn't parse
 * or isn't `http:`/`https:`, one with user info, and headers that aren't valid HTTP header
 * text, each with a `TypeError`. Sends the caller's `init` with `redirect: "manual"` over
 * whatever it says, `timeout: false` (Bun's own request timeout off; Node ignores the key)
 * and `User-Agent: USER_AGENT` unless the caller set one. Returns any non-3xx response as
 * is; for a 3xx it cancels the body and throws `RedirectError`, so credential headers never
 * reach a second origin. The first request sent while the process environment has
 * `NODE_TLS_REJECT_UNAUTHORIZED=0` emits `insecureTlsWarning`'s text through `warn`, once per
 * process.
 */
export async function safeFetch(
  input: string | URL,
  init: RequestInit = {},
  send: Send = globalThis.fetch,
): Promise<Response> {
  // Both runtimes echo the URL in their own errors, so refuse here, unchained.
  let url: URL;
  try {
    url = new URL(input);
  } catch {
    throw new TypeError("request URL is not a valid http(s) URL");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new TypeError("request URL is not a valid http(s) URL");
  }
  if (url.username || url.password) throw new TypeError("request URL must not contain user info");
  // Both runtimes echo an invalid header value, which may be a credential.
  let headers: Headers;
  try {
    headers = new Headers(init.headers);
  } catch {
    throw new TypeError("request headers are not valid HTTP header text");
  }
  if (!headers.has("user-agent")) headers.set("user-agent", USER_AGENT);
  const sent: RequestInit & { timeout: false } = { ...init, headers, redirect: "manual", timeout: false };
  if (!warnedInsecureTls) {
    // The runtime reads the process environment, not a host's injected one, so this does too.
    const warning = insecureTlsWarning(processEnv("NODE_TLS_REJECT_UNAUTHORIZED"));
    if (warning) {
      warnedInsecureTls = true;
      warn(warning);
    }
  }
  const response = await send(input, sent);
  // Only the status: the response type for a manual redirect differs between runtimes.
  if (response.status >= 300 && response.status < 400) {
    // Never awaited: an awaited cancel can hang, and a rejected one would go unhandled on Node.
    void response.body?.cancel().catch(() => {});
    throw new RedirectError(response.status, redirectDetail(response.headers.get("location")));
  }
  return response;
}
