/**
 * The one module holding runtime-specific APIs, so the rest of yian runs unchanged
 * on Node and Bun: runtime detection, a SQLite wrapper over `bun:sqlite` or
 * `node:sqlite`, loaded lazily for the runtime in use, an exclusive file lock
 * built on it, the scheduler that clocks and timed waits go through, the
 * TypeScript stripper for CODE bodies, loaded lazily too, and `safeFetch`, the one way out
 * to the network: no redirects, yian's User-Agent, and a warning when TLS checks are off.
 */

import type { Database } from "bun:sqlite";
// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { createRequire } from "node:module";
import type { DatabaseSync } from "node:sqlite";

import { processEnv } from "./defaults.ts";

/** The JavaScript runtime yian is running on, detected once at load. */
export const runtime: "bun" | "node" = typeof Bun !== "undefined" ? "bun" : "node";

/**
 * A value bound to a positional `?` parameter. A `number` passing
 * `Number.isSafeInteger` binds as INTEGER, as Python's `int` does; any other
 * `number` binds as REAL.
 */
export type SqliteParam = null | number | bigint | string | Uint8Array;

/** A value read from a column; integers are always safe `number`s. */
export type SqliteValue = null | number | string | Uint8Array;

/** One result row: a plain object from column name to value. */
export type SqliteRow = Record<string, SqliteValue>;

/**
 * A prepared statement taking positional `?` parameters. A missing positional
 * parameter throws on Bun but binds NULL on Node; the wrapper doesn't normalize that
 * programming error.
 */
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
    // Node's defaults differ from Bun's and Python's sqlite3, which leave foreign keys
    // off and accept double-quoted string literals.
    openDriver = (path, readonly) =>
      new NodeDatabase(path, {
        readOnly: readonly,
        enableForeignKeyConstraints: false,
        enableDoubleQuotedStringLiterals: true,
      });
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
 * `Number.isSafeInteger` throws `RangeError`. As with Python's sqlite3, foreign keys
 * are off and double-quoted string literals are accepted. Driver errors pass
 * through; read their result code with `sqliteErrorCode`.
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
        all: (...params) => raw.all(...bindable(params)).map((row) => plainRow(row as object)),
        get: (...params) => {
          // Bun returns null for no row, Node undefined.
          const row = raw.get(...bindable(params));
          return row == null ? undefined : plainRow(row);
        },
        run: (...params) => {
          raw.run(...bindable(params));
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
const SQLITE_CANTOPEN = 14;
const SQLITE_NOTADB = 26;

/** Return the SQLite primary result code `error` carries, or 255 for any other value. */
function primaryCode(error: unknown): number {
  return (sqliteErrorCode(error) ?? -1) & 0xff;
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
 * (or none) once locked. Busy can also be reported transiently while another
 * contender or a probe is mid-attempt (busy_timeout is 0); `waitForLock` absorbs it.
 * OS errors (`ENOENT` for a missing directory) and SQLite errors (`SQLITE_NOTADB`
 * for a foreign file) propagate unchanged.
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
    db.exec("PRAGMA busy_timeout = 0");
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
 * busy. A foreign file is `false`. When SQLite can't open the file, a path that has
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
 * Gives up with `FileLockError` (reason `"busy"`) once its sleeps add up to
 * `timeoutMs`; it never reads `scheduler.now()`, so a clock step doesn't change the
 * wait. The default `Infinity` waits for as long as it takes. Each try first checks
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
    // An unclosed construct is reported at the wrapper's closing `}`; Python reports it
    // on the last line with text, which such an error always has.
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

/** The raw sender `safeFetch` calls: the runtime's fetch, a cassette, or a test fake. */
export type Send = (input: string | URL, init: RequestInit) => Promise<Response>;

/** yian's version; a test keeps it equal to `package.json`'s. */
export const VERSION = "0.0.0";

/** The `User-Agent` every request carries unless its caller sets one. */
export const USER_AGENT = `yian/${VERSION}`;

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
  // Follows only urlsplit's "is there an authority" rule (`//` then a non-empty authority);
  // the URL parser alone would read `http:///x` or `http:x` as host `x`. `//user@/x` and
  // `//:80/x` give "unparsable" here where Python says relative.
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
 * `NODE_TLS_REJECT_UNAUTHORIZED=0` emits `insecureTlsWarning`'s text through
 * `process.emitWarning`, once per process.
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
      process.emitWarning(warning);
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
