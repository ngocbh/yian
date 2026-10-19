/**
 * Private single-owner SQLite storage for detached memory-backend state.
 *
 * One local file holds one table with one row: a versioned JSON envelope naming the backend kind
 * and its state. The handle holds an exclusive SQLite lock on the file from construction until
 * `close()`, so a second owner, in this process or another, fails `storage.busy`.
 */

// The default import keeps calls patchable by test spies.
import fs from "node:fs";

import { openSqlite, type Sqlite, sqliteErrorCode } from "../../platform.ts";
import { JSONDecodeError, type JsonObject, logJson, parseJson } from "../../typesys/json.ts";
import { freezeJsonObject, MemoryOperationError } from "../contracts.ts";

// Python yian's envelope is version 2; TS never reads Python store formats.
const VERSION = 3n;
const SCHEMA = "CREATE TABLE memory_state (id INTEGER PRIMARY KEY CHECK (id = 1), envelope TEXT NOT NULL)";

/** Return one storage failure; only `busy` is retryable. */
function storageError(reason: string, message: string): MemoryOperationError {
  return new MemoryOperationError("storage", `storage.${reason}`, message, reason === "busy");
}

/**
 * Whether `error` is a Node SystemError (an OS error). A copy of `skills/catalog.ts`'s
 * predicate: memory doesn't import skills.
 */
function isSystemError(error: unknown): error is NodeJS.ErrnoException & { code: string } {
  if (!(error instanceof Error)) return false;
  const { code, syscall, errno } = error as NodeJS.ErrnoException;
  return typeof code === "string" && (typeof syscall === "string" || typeof errno === "number");
}

/** Whether `error` is an I/O failure (Python's `OSError` or `sqlite3.Error`). */
function isIoError(error: unknown): boolean {
  return sqliteErrorCode(error) !== undefined || isSystemError(error);
}

/** Map an I/O failure to a sanitized storage error, by its SQLite primary code. */
function ioError(error: unknown): MemoryOperationError {
  const code = (sqliteErrorCode(error) ?? 0) & 0xff;
  if (code === 5 || code === 6) return storageError("busy", "Memory storage is already owned or busy.");
  if (code === 11 || code === 26) return storageError("corrupt", "Memory storage is not a valid SQLite database.");
  return storageError("io", "Memory storage I/O failed.");
}

/** Return the envelope text for `state`, which must already be a checked copy. */
function encode(backend: string, state: object | null): string {
  return logJson({ version: VERSION, backend, state });
}

/** Return a fresh copy of the state in envelope text `raw`, checking its version and backend kind. */
function decode(raw: unknown, backend: string): JsonObject | null {
  try {
    if (typeof raw !== "string") throw new TypeError("envelope must be text");
    const envelope = parseJson(raw, "codec");
    if (typeof envelope !== "object" || envelope === null || Array.isArray(envelope)) {
      throw new TypeError("envelope must be an object");
    }
    const keys = Object.keys(envelope);
    if (keys.length !== 3 || !["version", "backend", "state"].every((key) => Object.hasOwn(envelope, key))) {
      throw new RangeError("invalid envelope fields");
    }
    if (envelope.version !== VERSION) throw storageError("version", "Memory storage version is unsupported.");
    if (typeof envelope.backend !== "string" || !envelope.backend) throw new TypeError("backend must be a string");
    if (envelope.backend !== backend) {
      throw storageError("kind", "Memory storage belongs to a different backend kind.");
    }
    const state = envelope.state;
    if (state === null) return null;
    if (typeof state !== "object" || Array.isArray(state)) throw new TypeError("state must be an object");
    return state;
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError || error instanceof JSONDecodeError) {
      throw storageError("corrupt", "Memory storage contains an invalid state envelope.");
    }
    throw error;
  }
}

/** `posixpath.join`: no normalization; an absolute part restarts the path. */
function join(a: string, ...parts: string[]): string {
  let path = a;
  for (const b of parts) {
    if (b.startsWith("/")) path = b;
    else if (!path || path.endsWith("/")) path += b;
    else path += `/${b}`;
  }
  return path;
}

/** `posixpath.split`: the head without trailing slashes (unless all slashes) and the tail. */
function split(p: string): [string, string] {
  const i = p.lastIndexOf("/") + 1;
  let head = p.slice(0, i);
  const tail = p.slice(i);
  if (head && head !== "/".repeat(head.length)) head = head.replace(/\/+$/, "");
  return [head, tail];
}

/** `posixpath.normpath`: collapse `.`, `..` and repeated slashes lexically. */
function normpath(path: string): string {
  if (!path) return ".";
  let initialSlashes = path.startsWith("/") ? 1 : 0;
  // POSIX allows a leading `//` to mean something else; three or more is one.
  if (initialSlashes && path.startsWith("//") && !path.startsWith("///")) initialSlashes = 2;
  const comps: string[] = [];
  for (const comp of path.split("/")) {
    if (comp === "" || comp === ".") continue;
    if (comp !== ".." || (!initialSlashes && comps.length === 0) || comps.at(-1) === "..") comps.push(comp);
    else if (comps.length > 0) comps.pop();
  }
  return "/".repeat(initialSlashes) + comps.join("/") || ".";
}

/**
 * Join `rest` to the resolved `path`, following symlinks: Python 3.12's `_joinrealpath` with
 * `strict=False`. Returns the path and whether it was fully resolved (`false` on a loop).
 */
function joinRealPath(path: string, rest: string, seen: Map<string, string | null>): [string, boolean] {
  if (rest.startsWith("/")) {
    rest = rest.slice(1);
    path = "/";
  }
  while (rest) {
    const cut = rest.indexOf("/");
    let name = cut === -1 ? rest : rest.slice(0, cut);
    rest = cut === -1 ? "" : rest.slice(cut + 1);
    if (!name || name === ".") continue;
    if (name === "..") {
      if (path) {
        [path, name] = split(path);
        if (name === "..") path = join(path, "..", "..");
      } else {
        path = "..";
      }
      continue;
    }
    const newpath = join(path, name);
    let isLink: boolean;
    try {
      isLink = fs.lstatSync(newpath).isSymbolicLink();
    } catch (error) {
      // Non-strict: any OS error means "not a link", so a missing component is kept.
      if (!isSystemError(error)) throw error;
      isLink = false;
    }
    if (!isLink) {
      path = newpath;
      continue;
    }
    if (seen.has(newpath)) {
      const known = seen.get(newpath)!;
      if (known !== null) {
        path = known;
        continue;
      }
      // A symlink loop: return the rest unresolved.
      return [join(newpath, rest), false];
    }
    seen.set(newpath, null);
    let ok: boolean;
    [path, ok] = joinRealPath(path, fs.readlinkSync(newpath), seen);
    if (!ok) return [join(path, rest), false];
    seen.set(newpath, path);
  }
  return [path, true];
}

/**
 * Return `path` with its symlinks resolved, as Python's non-strict `os.path.realpath` does: a
 * missing component is kept and later ones are still resolved, a symlink is followed before a
 * later `..` applies, a loop leaves the rest unresolved, and the result is absolute and normalized.
 *
 * @throws Error when reading a symlink fails (a Node system error).
 */
export function realStoragePath(path: string): string {
  const [resolved] = joinRealPath("", path, new Map());
  return normpath(resolved.startsWith("/") ? resolved : join(process.cwd(), resolved));
}

/**
 * Return an absolute local file path, rejecting non-file targets without I/O. A relative path is
 * joined to the current directory without normalizing `..`; symlinks are resolved later, when the
 * file is opened.
 *
 * @throws MemoryOperationError `storage.path` for anything but a string, an empty path, `:memory:`,
 *   a SQLite `file:` URI, a NUL character or a lone surrogate.
 */
export function checkStoragePath(path: unknown): string {
  if (
    typeof path !== "string" ||
    !path ||
    path === ":memory:" ||
    path.toLowerCase().startsWith("file:") ||
    path.includes("\0") ||
    !path.isWellFormed()
  ) {
    throw storageError("path", "Memory storage requires a local file path.");
  }
  if (path.startsWith("/")) return path;
  // Collapsing ".." here would skip a symlink the OS follows on open.
  return join(process.cwd(), path);
}

/**
 * Own one local SQLite file until explicitly closed.
 *
 * The file contains one backend kind and one versioned JSON state envelope. Transactions keep the
 * exclusive file lock across commits. A failed I/O operation poisons the handle until close. This
 * private adapter does not own backend sessions, queues, or callbacks.
 */
export class SQLiteStorage {
  readonly #backend: string;
  readonly #path: string;
  private connection: Sqlite | null = null;
  #closed = false;
  #poisoned = false;

  /**
   * Open or initialize the file at `path` for backend kind `backend` and take its lock without
   * waiting. The parent directory must exist unless `createParent` is set.
   *
   * @throws MemoryOperationError for an invalid path or backend kind, an occupied file
   *   (`storage.busy`), an incompatible or corrupt store, or an ordinary I/O failure.
   */
  constructor(path: unknown, backend: unknown, { createParent = false }: { createParent?: boolean } = {}) {
    if (typeof backend !== "string" || !backend || !backend.isWellFormed()) {
      throw storageError("backend", "Memory storage requires a nonempty backend kind.");
    }
    this.#backend = backend;
    const checked = checkStoragePath(path);
    try {
      this.#path = realStoragePath(checked);
    } catch (error) {
      if (isIoError(error)) throw ioError(error);
      throw error;
    }
    try {
      if (createParent) fs.mkdirSync(split(this.#path)[0], { recursive: true });
      let fd: number | undefined;
      try {
        const { O_CREAT, O_EXCL, O_WRONLY } = fs.constants;
        fd = fs.openSync(this.#path, O_CREAT | O_EXCL | O_WRONLY, 0o600);
      } catch (error) {
        if ((error as NodeJS.ErrnoException | null)?.code !== "EEXIST") throw error;
      }
      if (fd !== undefined) {
        try {
          // The open's mode is masked by the umask; fchmod isn't.
          fs.fchmodSync(fd, 0o600);
        } catch (error) {
          try {
            fs.closeSync(fd);
          } catch {}
          throw error;
        }
        fs.closeSync(fd);
      }
      if (!fs.statSync(this.#path).isFile()) throw storageError("path", "Memory storage requires a regular file.");
      this.connection = openSqlite(this.#path);
      const connection = this.connection;
      connection.exec("PRAGMA busy_timeout = 0");
      connection.exec("PRAGMA locking_mode = EXCLUSIVE");
      connection.exec("PRAGMA synchronous = EXTRA");
      if (connection.prepare("PRAGMA journal_mode").get()?.journal_mode !== "delete") {
        throw storageError("schema", "Memory storage requires rollback-journal format.");
      }
      connection.exec("BEGIN EXCLUSIVE");
      // A creator can die before SQLite initialization. Only genuinely empty files may
      // initialize; an unrelated empty SQLite database is not ours.
      if (fs.statSync(this.#path).size === 0) {
        connection.exec(SCHEMA);
        connection.prepare("INSERT INTO memory_state (id, envelope) VALUES (1, ?)").run(encode(this.#backend, null));
      }
      this.#validateSchema(connection);
      this.#read(connection);
      connection.exec("COMMIT");
    } catch (error) {
      if (this.connection !== null) {
        this.#rollback();
        try {
          this.connection.close();
        } catch {}
      }
      this.#closed = true;
      if (isIoError(error)) throw ioError(error);
      throw error;
    }
  }

  #validateSchema(connection: Sqlite): void {
    const rows = connection.prepare("SELECT type, name, tbl_name, sql FROM sqlite_master").all();
    const [row] = rows;
    if (
      rows.length !== 1 ||
      row!.type !== "table" ||
      row!.name !== "memory_state" ||
      row!.tbl_name !== "memory_state" ||
      row!.sql !== SCHEMA
    ) {
      throw storageError("schema", "Memory storage schema is unsupported.");
    }
    const check = connection.prepare("PRAGMA quick_check").all();
    if (check.length !== 1 || check[0]!.quick_check !== "ok") {
      throw storageError("corrupt", "Memory storage failed its integrity check.");
    }
  }

  #read(connection: Sqlite): JsonObject | null {
    const rows = connection.prepare("SELECT id, envelope FROM memory_state").all();
    if (rows.length !== 1 || rows[0]!.id !== 1) {
      throw storageError("corrupt", "Memory storage must contain exactly one state envelope.");
    }
    return decode(rows[0]!.envelope, this.#backend);
  }

  #openConnection(): Sqlite {
    if (this.#closed) throw storageError("closed", "Memory storage is closed.");
    if (this.#poisoned) {
      throw storageError("poisoned", "Memory storage must be closed and reopened after an I/O failure.");
    }
    return this.connection!;
  }

  /**
   * Reject a closed or poisoned handle before backend RAM reads.
   *
   * @throws MemoryOperationError `storage.closed` or `storage.poisoned`.
   */
  ensureOpen(): void {
    this.#openConnection();
  }

  #rollback(): void {
    try {
      this.connection?.exec("ROLLBACK");
    } catch {}
  }

  /**
   * Return a fresh mutable copy of the last committed state, or `null` before the first save.
   *
   * @throws MemoryOperationError when the handle, the stored data or the I/O is invalid; each
   *   failure poisons the handle.
   */
  load(): JsonObject | null {
    const connection = this.#openConnection();
    try {
      return this.#read(connection);
    } catch (error) {
      this.#poisoned = true;
      this.#rollback();
      if (isIoError(error)) throw ioError(error);
      throw error;
    }
  }

  /**
   * Check and atomically commit a complete backend state: a plain JSON object whose numbers may be
   * `±Infinity` but not NaN (`-0` is stored as `0`).
   *
   * @throws MemoryOperationError `storage.invalid_state` for an invalid candidate, which doesn't
   *   poison the handle; a closed or poisoned handle; an I/O failure, which poisons it.
   */
  save(state: unknown): void {
    const connection = this.#openConnection();
    let encoded: string;
    try {
      encoded = encode(this.#backend, freezeJsonObject(state, "state", { limitNesting: false, valueNumbers: true }));
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) {
        throw storageError("invalid_state", "Memory storage requires a finite JSON object.");
      }
      throw error;
    }
    try {
      connection.exec("BEGIN EXCLUSIVE");
      connection.prepare("UPDATE memory_state SET envelope = ? WHERE id = 1").run(encoded);
      connection.exec("COMMIT");
    } catch (error) {
      this.#poisoned = true;
      this.#rollback();
      if (isIoError(error)) throw ioError(error);
      throw error;
    }
  }

  /**
   * Release file ownership; repeated closes do nothing.
   *
   * @throws MemoryOperationError when closing the connection fails.
   */
  close(): void {
    if (this.#closed) return;
    try {
      this.connection!.close();
    } catch (error) {
      this.#poisoned = true;
      if (isIoError(error)) throw ioError(error);
      throw error;
    }
    this.#closed = true;
    this.connection = null;
  }
}
