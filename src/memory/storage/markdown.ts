/**
 * Versioned markdown document stores.
 *
 * A markdown store holds one current markdown document and can return any version it has handed
 * out. Writes are compare-and-set against a base version so that a concurrent human edit is never
 * silently overwritten.
 */

import { createHash, randomBytes } from "node:crypto";
// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { basename, dirname, join } from "node:path";

import { current as currentDefaults } from "../../defaults.ts";
import { compareCodePoints, reprText } from "../../diagnostics.ts";
import {
  type FileLock,
  FileLockError,
  isSystemError,
  sqliteErrorCode,
  statOrNull,
  syncFolder,
  waitForLock,
} from "../../platform.ts";
import type { JsonObject } from "../../typesys/json.ts";
import { floatText } from "../../typesys/values.ts";
import { freezeJsonObject } from "../contracts.ts";
import { checkStoragePath, realStoragePath } from "./sqlite.ts";

/** A compare-and-set write named a base version that is no longer current. */
export class MarkdownVersionMismatch extends Error {
  override name = "MarkdownVersionMismatch";
  /** The base version the writer supplied. */
  readonly expected: string;
  /** The version found in the store. */
  readonly current: string;

  constructor(expected: string, current: string) {
    super(`base version ${reprText(expected)} is not current (${reprText(current)})`);
    this.expected = expected;
    this.current = current;
  }
}

/**
 * Versioned storage for one markdown document. A host implementation may answer synchronously or
 * with a promise; yian always awaits.
 *
 * Versions are opaque strings the store chooses: `FileMarkdownStore` uses `markdownVersion`, and a
 * git-backed store may use blob ids. yian only compares them for equality and hands them back.
 * `read(v)` returns `v` itself as the version. A store must keep every version it has handed out,
 * including the version it reports before the first write: profile memory's run snapshot records
 * only the version and a text hash, so resume and later backward attempts read that version again.
 *
 * Implementations throw `RangeError` for an unknown version or stored text they can't decode,
 * `MarkdownVersionMismatch` for a stale base version, and a Node system error (an `Error` with a
 * string `code` and a string `syscall` or a numeric `errno`) when storage can't be reached. In a
 * run, profile memory turns these, and on restore a `TypeError`, into typed memory failures; any
 * other error passes through it, `runFlow`, `advance` and `backward` unchanged. `signal`, when
 * given, is the run's stop: a store that waits should give up with `signal.reason` once it aborts.
 */
export interface MarkdownStore {
  /**
   * Return `[text, version]` for the current document (`version` omitted or `null`) or for a
   * version this store returned earlier.
   *
   * @throws RangeError when the named version is unknown.
   */
  read(
    version?: string | null,
    options?: { signal?: AbortSignal },
  ): readonly [text: string, version: string] | Promise<readonly [text: string, version: string]>;

  /**
   * Replace the current document with `text` if `baseVersion` is still current, and return the
   * stored document's version. `provenance` is caller-defined JSON describing who wrote and why;
   * from profile memory it holds `run_id`, `commit_id`, `seen_version` and `feedback`.
   * `commit_id` is unique per backward attempt and memory, so a store that retries its own
   * transport may use it to avoid writing twice.
   *
   * @throws MarkdownVersionMismatch when `baseVersion` is not current; nothing was written.
   * @throws Error a Node system error when storage can't be reached; the write may or may not
   *   have landed.
   */
  write(
    text: string,
    options: { baseVersion: string; provenance?: JsonObject | null; signal?: AbortSignal },
  ): string | Promise<string>;
}

/**
 * Return the content-addressed version of one document: `"sha256:<hex>"` of its UTF-8 bytes.
 *
 * @throws RangeError when `text` holds a lone surrogate, which has no UTF-8 encoding.
 */
export function markdownVersion(text: string): string {
  if (!text.isWellFormed()) throw new RangeError("text holds a lone surrogate, which has no UTF-8 encoding");
  return `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}

/**
 * Return `value` as JSON with `", "` and `": "` separators, keys in code-point order, non-ASCII and
 * DEL as lowercase `\uXXXX`, a `number` as `floatText` spells it (keeping a fraction digit) and a
 * `bigint` as an integer. Users' saved files use this format, so it doesn't change. `value` is
 * already a checked JSON copy.
 */
function spacedJson(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "number") return floatText(value);
  if (typeof value === "string") {
    return JSON.stringify(value).replace(/[^\x20-\x7e]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
  }
  if (Array.isArray(value)) return `[${value.map(spacedJson).join(", ")}]`;
  const object = value as Record<string, unknown>;
  const keys = Object.keys(object).sort(compareCodePoints);
  return `{${keys.map((key) => `${spacedJson(key)}: ${spacedJson(object[key])}`).join(", ")}}`;
}

/** Return the UTF-8 text of `source` exactly: line endings and a BOM kept. */
function readExact(source: string): string {
  const bytes = fs.readFileSync(source);
  try {
    // Fatal, so invalid bytes fail instead of turning into U+FFFD and a version of other bytes.
    return new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch {
    throw new RangeError("document is not valid UTF-8");
  }
}

/**
 * Write `text` to `target` through a synced temporary file in its directory, then sync the
 * directory so the renamed entry survives a power loss.
 */
function replace(target: string, text: string): void {
  const temporary = join(dirname(target), `.${basename(target)}.${randomBytes(8).toString("hex")}`);
  const { O_CREAT, O_EXCL, O_WRONLY } = fs.constants;
  const fd = fs.openSync(temporary, O_CREAT | O_EXCL | O_WRONLY, 0o600);
  try {
    try {
      fs.writeFileSync(fd, text);
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
    fs.renameSync(temporary, target);
  } catch (error) {
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
    throw error;
  }
  syncFolder(dirname(target));
}

/** Make private directory `path` unless it exists, and sync its parent when it was made. */
function makeDirectory(path: string): void {
  try {
    fs.mkdirSync(path, { mode: 0o700 });
  } catch (error) {
    if (isSystemError(error) && error.code === "EEXIST") return;
    throw error;
  }
  syncFolder(dirname(path));
}

const VERSION = /^sha256:[0-9a-f]{64}$/;

/**
 * Markdown store backed by one editable file.
 *
 * The current document is `path`; a missing file reads as an empty document. Text is stored and
 * read back exactly, line endings and a BOM included. Every version this store returns or writes
 * is kept under `.<name>.versions/` next to it, so recorded runs can reopen it later. Each write
 * appends its versions and provenance to `.<name>.writes.jsonl` as one JSON line: keys sorted by
 * code point, separators `, ` and `: `, non-ASCII as `\uXXXX`, floats with a fraction digit.
 * The write log and kept copies are made private to their owner (mode 0600, the versions folder
 * 0700), since provenance can hold a learner's feedback. Writers on one host take turns through
 * `.<name>.lock`, each waiting at most `YIAN_PROFILE_LOCK_TIMEOUT_SECONDS` for the one holding it.
 * Files and the directory entries naming them are synced before a call returns, so a version it
 * handed out survives a power loss. Nothing on disk is touched until the first read or write. A
 * person may edit `path` directly; the next compare-and-set write then reports a mismatch. A
 * symlinked `path` is resolved when the store is built: writes update the link's target, and the
 * versions, write log and lock sit next to the target.
 */
export class FileMarkdownStore implements MarkdownStore {
  /** The document's absolute path, symlinks resolved. */
  readonly path: string;
  readonly #createParent: boolean;
  readonly #directory: string;
  readonly #versions: string;
  readonly #lockPath: string;
  readonly #writes: string;

  /**
   * Bind the store to the document at `path`, resolving symlinks and creating nothing. With
   * `createParent` the parent directory is created on first use; otherwise a missing parent fails
   * on first use with an `ENOENT` system error.
   *
   * @throws TypeError when `createParent` isn't a boolean.
   * @throws MemoryOperationError `storage.path` when `path` isn't a local file path: not a
   *   string, empty, `:memory:`, a `file:` URI, or text with a NUL or a lone surrogate. A
   *   directory is detected on first use.
   */
  constructor(path: unknown, { createParent = false }: { createParent?: boolean } = {}) {
    if (typeof createParent !== "boolean") throw new TypeError("create_parent must be a bool");
    // Resolve now so a later chdir doesn't redirect the store, and so the atomic replace updates a
    // symlink's target instead of the link itself.
    this.path = realStoragePath(checkStoragePath(path));
    this.#createParent = createParent;
    this.#directory = dirname(this.path);
    const name = basename(this.path);
    this.#versions = join(this.#directory, `.${name}.versions`);
    this.#lockPath = join(this.#directory, `.${name}.lock`);
    this.#writes = join(this.#directory, `.${name}.writes.jsonl`);
  }

  /**
   * Return `[text, version]` for the current document or a named version.
   *
   * @throws RangeError when the named version is unknown or the file isn't valid UTF-8.
   * @throws Error a Node system error when storage can't be reached, including `ENOENT` for a
   *   missing parent directory without `createParent` and `EIO` once the lock wait times out.
   * @throws `signal.reason` when `signal` aborts while waiting for the lock.
   */
  async read(
    version: string | null = null,
    { signal }: { signal?: AbortSignal } = {},
  ): Promise<readonly [string, string]> {
    return this.#locked(signal, () => {
      const text = this.#current();
      const current = markdownVersion(text);
      if (version === null || current === version) {
        this.#keep(current, text);
        return [text, current] as const;
      }
      const kept = this.#versionPath(version);
      if (kept === null || !statOrNull(kept)?.isFile()) {
        throw new RangeError(`unknown document version ${reprText(version)}`);
      }
      return [readExact(kept), version] as const;
    });
  }

  /**
   * Replace the current document if `baseVersion` is still current, append the versions and a
   * copy of `provenance` to the write log, and return the new version.
   *
   * @throws TypeError when `text` or `baseVersion` isn't a string, or `provenance` isn't `null` or
   *   an exact JSON object; nothing was written.
   * @throws RangeError when `text` holds a lone surrogate, the current file isn't valid UTF-8, or
   *   `provenance` holds a non-finite number, a lone surrogate or a cycle; nothing was written.
   * @throws MarkdownVersionMismatch when `baseVersion` is not current.
   * @throws Error a Node system error when storage can't be reached, `EIO` once the lock wait
   *   times out.
   * @throws `signal.reason` when `signal` aborts while waiting for the lock; nothing was written.
   */
  async write(
    text: string,
    {
      baseVersion,
      provenance = null,
      signal,
    }: { baseVersion: string; provenance?: JsonObject | null; signal?: AbortSignal },
  ): Promise<string> {
    if (typeof text !== "string" || typeof baseVersion !== "string") {
      throw new TypeError("text and base_version must be strings");
    }
    const copy = provenance === null ? null : freezeJsonObject(provenance, "provenance", { limitNesting: false });
    const newVersion = markdownVersion(text);
    // Encode the log line before touching the document so a bad value can't leave a replaced
    // document without its log entry.
    const line = `${spacedJson({ base_version: baseVersion, version: newVersion, provenance: copy })}\n`;
    return this.#locked(signal, () => {
      const current = markdownVersion(this.#current());
      if (current !== baseVersion) throw new MarkdownVersionMismatch(baseVersion, current);
      this.#keep(newVersion, text);
      replace(this.path, text);
      const { O_APPEND, O_CREAT, O_EXCL, O_WRONLY } = fs.constants;
      let created = true;
      let fd: number;
      try {
        // Private like the document: provenance can hold the learner's feedback.
        fd = fs.openSync(this.#writes, O_CREAT | O_EXCL | O_WRONLY | O_APPEND, 0o600);
      } catch (error) {
        if (!isSystemError(error) || error.code !== "EEXIST") throw error;
        created = false;
        fd = fs.openSync(this.#writes, O_WRONLY | O_APPEND);
      }
      try {
        // The open's mode is masked by the umask; fchmod isn't.
        if (created) fs.fchmodSync(fd, 0o600);
        fs.appendFileSync(fd, line);
        fs.fsyncSync(fd);
      } finally {
        fs.closeSync(fd);
      }
      if (created) syncFolder(this.#directory);
      return newVersion;
    });
  }

  /**
   * Run `body` holding the host-wide exclusive lock for this document, waiting for it at most
   * `YIAN_PROFILE_LOCK_TIMEOUT_SECONDS` and until `signal` aborts.
   */
  async #locked<T>(signal: AbortSignal | undefined, body: () => T): Promise<T> {
    if (this.#createParent) {
      const first = fs.mkdirSync(this.#directory, { recursive: true });
      // Each new directory's entry lives in its parent, from the one above the first made down.
      if (first !== undefined) {
        for (let made = this.#directory; made !== dirname(first); made = dirname(made)) syncFolder(dirname(made));
      }
    } else if (!statOrNull(this.#directory)?.isDirectory()) {
      throw Object.assign(new Error(`document directory does not exist: ${this.#directory}`), {
        code: "ENOENT",
        syscall: "stat",
      });
    }
    let lock: FileLock;
    try {
      const timeoutMs = currentDefaults().profileLockTimeoutSeconds * 1000;
      lock = await waitForLock(this.#lockPath, { timeoutMs, signal });
    } catch (error) {
      // Checked before passing errors through: Bun's SQLiteError has a string `code` and a numeric
      // `errno`, so it already looks like a system error, but callers must get a real one.
      if (error instanceof FileLockError || sqliteErrorCode(error) !== undefined) {
        throw Object.assign(new Error((error as Error).message, { cause: error }), { code: "EIO", syscall: "lock" });
      }
      throw error;
    }
    try {
      return body();
    } finally {
      lock.release();
    }
  }

  /** Return the current file text, or an empty document when it's missing. */
  #current(): string {
    try {
      return readExact(this.path);
    } catch (error) {
      if (isSystemError(error) && error.code === "ENOENT") return "";
      throw error;
    }
  }

  /** Return the kept-copy path for a well-formed version, else `null`. */
  #versionPath(version: string): string | null {
    return VERSION.test(version) ? join(this.#versions, `${version.slice("sha256:".length)}.md`) : null;
  }

  /** Store one content-addressed copy unless it already exists. */
  #keep(version: string, text: string): void {
    const kept = this.#versionPath(version);
    if (kept !== null && statOrNull(kept) === null) {
      makeDirectory(this.#versions);
      replace(kept, text);
    }
  }
}

/**
 * Process-local markdown store with the same compare-and-set contract, answering synchronously.
 *
 * It keeps every version it has returned or written, so recorded runs in the same process can
 * reopen them. Provenance is accepted and discarded.
 */
export class InMemoryMarkdownStore implements MarkdownStore {
  #current: string;
  readonly #versions: Map<string, string>;

  /**
   * Start with `text` as the current document.
   *
   * @throws TypeError when `text` isn't a string.
   * @throws RangeError when `text` holds a lone surrogate.
   */
  constructor(text: string = "") {
    if (typeof text !== "string") throw new TypeError("text must be a string");
    this.#current = text;
    this.#versions = new Map([[markdownVersion(text), text]]);
  }

  /**
   * Return `[text, version]` for the current document or a kept version.
   *
   * @throws RangeError when the named version is unknown.
   */
  read(version: string | null = null): readonly [string, string] {
    if (version === null) return [this.#current, markdownVersion(this.#current)];
    const text = this.#versions.get(version);
    if (text === undefined) throw new RangeError(`unknown document version ${reprText(version)}`);
    return [text, version];
  }

  /**
   * Replace the document if `baseVersion` is still current and return the new version;
   * `provenance` is accepted for the interface and discarded.
   *
   * @throws TypeError when `text` or `baseVersion` isn't a string.
   * @throws RangeError when `text` holds a lone surrogate.
   * @throws MarkdownVersionMismatch when the current version isn't `baseVersion`.
   */
  write(text: string, { baseVersion }: { baseVersion: string; provenance?: JsonObject | null }): string {
    if (typeof text !== "string" || typeof baseVersion !== "string") {
      throw new TypeError("text and base_version must be strings");
    }
    const current = markdownVersion(this.#current);
    if (baseVersion !== current) throw new MarkdownVersionMismatch(baseVersion, current);
    const version = markdownVersion(text);
    this.#current = text;
    this.#versions.set(version, text);
    return version;
  }
}
