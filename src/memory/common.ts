/**
 * What yian's own memory types share: this process's queue on a memory's lock, the checks of
 * their options, arguments and authors, and the text of their messages and refusals.
 *
 * It imports `platform`, `tools`, `typesys` and `memory/base.ts`.
 */

import { createHash } from "node:crypto";
// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { basename, dirname, join, resolve } from "node:path";

import {
  type FileLock,
  FileLockError,
  isSystemError,
  realScheduler,
  type Scheduler,
  waitForLock,
} from "../platform.ts";
import { ReportedToolError } from "../tools/errors.ts";
import { YianRangeError, YianTypeError } from "../typesys/errors.ts";
import type { JsonObject } from "../typesys/json.ts";
import { type Applied, type Author, MemoryBusyError } from "./base.ts";

/**
 * A body that runs holding a memory's lock. It must finish before it returns, so it can't be async:
 * the lock is released when it returns, and an async body would go on without it.
 */
export type Held<T> = (() => T) & (T extends PromiseLike<unknown> ? never : unknown);

/** A memory's lock, and how long a call waits for it. */
export interface MemoryLock {
  /** The memory's real location: this process's calls on it take turns, whichever instance makes them. */
  readonly key: string;
  /** The lock file. */
  readonly path: string;
  /** Longest wait in milliseconds, from when a call is made, its time in the queue included. */
  readonly timeoutMs: number;
  /** The sleep the wait for the lock file uses between tries. */
  readonly scheduler: Scheduler;
  /**
   * Check the memory's own files: before the lock file is opened, and again once it is held,
   * since they may change while a call waits.
   */
  readonly check: () => void;
}

/**
 * This process's calls that hold a memory's lock, keyed by the memory's real location and
 * chained, so its sessions and opens take turns rather than poll the lock against each other.
 */
const TURNS = new Map<string, Promise<void>>();

/** The error a call gets when its wait for the memory's lock runs past its timeout. */
function busy(): MemoryBusyError {
  // No cause: a lock's error names its path, and this one can reach a model.
  return new MemoryBusyError("this memory is busy: another writer holds its lock");
}

/**
 * Call `fn` once `ms` real milliseconds have passed, unless `signal` aborts first; never for an
 * infinite `ms`. Even 0 ms waits for a macrotask: the turns ahead of a queued call settle in
 * microtasks unless one of them waits, so only a real wait makes a queued call late.
 */
function afterMs(ms: number, signal: AbortSignal, fn: () => void): void {
  if (!Number.isFinite(ms)) return;
  const wait = ms > 0 ? realScheduler.sleep(ms, signal) : new Promise<void>((resolve) => setTimeout(resolve, 0));
  wait.then(
    () => {
      if (!signal.aborted) fn();
    },
    () => undefined,
  );
}

/**
 * Run `fn` once this process's earlier turns on the memory `key` have finished, and return its
 * result.
 *
 * A call leaves the queue before its turn when `signal` aborts, rejecting with its reason, or
 * when it has waited `timeoutMs`, rejecting with `MemoryBusyError`, even while the turn ahead
 * still runs. Its turn, when it comes, does nothing, so the calls after it keep their order: once
 * the call has settled, `fn` has finished or will never run. Once its turn has started, `fn`
 * answers the abort and the timeout itself, so a change it applies is never reported as aborted.
 */
function inTurn<T>(key: string, signal: AbortSignal | undefined, timeoutMs: number, fn: () => Promise<T>): Promise<T> {
  let started = false;
  /** Why the call left the queue before its turn, once it has. */
  let left: { reason: unknown } | null = null;
  // Aborted when the turn starts or the call settles, which stops the queue's timer.
  const waiting = new AbortController();
  const run = (TURNS.get(key) ?? Promise.resolve()).then(() => {
    started = true;
    waiting.abort();
    if (left !== null) throw left.reason;
    signal?.throwIfAborted();
    return fn();
  });
  const done = run.then(
    () => undefined,
    () => undefined,
  );
  TURNS.set(key, done);
  void done.then(() => {
    if (TURNS.get(key) === done) TURNS.delete(key);
  });
  return new Promise<T>((resolve, reject) => {
    const leave = (reason: unknown) => {
      if (started || left !== null) return;
      left = { reason };
      reject(reason);
    };
    const onAbort = () => leave(signal!.reason);
    signal?.addEventListener("abort", onAbort, { once: true });
    // A signal aborted already never fires its listeners.
    if (signal?.aborted) onAbort();
    afterMs(timeoutMs, waiting.signal, () => leave(busy()));
    void run.then(resolve, reject).finally(() => {
      signal?.removeEventListener("abort", onAbort);
      waiting.abort();
    });
  });
}

/**
 * Run `body` holding the memory's lock `lock`, after this process's earlier calls on the memory:
 * they take turns, so only another process's writer makes it wait for the lock file. It waits at
 * most the lock's timeout from when it was called, its time in the queue included, and leaves the
 * queue as `inTurn` says: an abort while queued rejects at once, and so does the deadline, with
 * `MemoryBusyError`, and the body of a call that left never runs.
 */
export function locked<T>(lock: MemoryLock, signal: AbortSignal | undefined, body: () => T | Promise<T>): Promise<T> {
  // A monotonic clock, so a wall-clock step while queued neither stretches nor cuts the wait.
  const deadline = performance.now() + lock.timeoutMs;
  return inTurn(lock.key, signal, lock.timeoutMs, async () => {
    // Checked before the lock file is opened, and on every call: the files may change.
    lock.check();
    // Only what is left of the timeout: a call waits it out once, queued or not.
    const left = Math.max(0, deadline - performance.now());
    let held: FileLock;
    try {
      held = await waitForLock(lock.path, { timeoutMs: left, signal, scheduler: lock.scheduler, private: true });
    } catch (error) {
      if (error instanceof FileLockError && error.reason === "busy") throw busy();
      throw error;
    }
    try {
      // Again holding the lock: the files may have changed while the call waited for it.
      lock.check();
      return await body();
    } finally {
      held.release();
    }
  });
}

/**
 * A session's calls that haven't settled, which its `close` waits for. A call that left its
 * memory's queue has settled, and its body never runs, so the turn it left isn't waited for: it
 * may be behind another instance's turn that holds the lock for long.
 */
export class PendingCalls {
  readonly #calls = new Set<Promise<void>>();

  /** Return `promise`, kept among the pending calls until it settles. */
  track<T>(promise: Promise<T>): Promise<T> {
    const settled = promise.then(
      () => undefined,
      () => undefined,
    );
    this.#calls.add(settled);
    void settled.then(() => this.#calls.delete(settled));
    return promise;
  }

  /** Resolve once no call is pending, the calls tracked meanwhile included. */
  async settled(): Promise<void> {
    while (this.#calls.size > 0) await Promise.all([...this.#calls]);
  }
}

/**
 * Return `value` when it is a safe integer of at least `minimum`, else its default.
 *
 * @param type The memory type, as the message names it.
 * @throws {RangeError} If `value` is given and isn't such an integer.
 */
export function capOption(
  type: string,
  name: string,
  value: number | undefined,
  fallback: number,
  minimum: number,
): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new YianRangeError(`${type} ${name} must be an integer of at least ${minimum}, not ${String(value)}`);
  }
  return value;
}

/** `fs.lstatSync`, or `null` when nothing is at `path`. */
export function lstatOrNull(path: string): fs.Stats | null {
  try {
    return fs.lstatSync(path);
  } catch (error) {
    if (isSystemError(error) && error.code === "ENOENT") return null;
    throw error;
  }
}

/** Return `path`'s real path, resolving the symlinks of the part of it that exists. */
export function realPath(path: string): string {
  const rest: string[] = [];
  let head = resolve(path);
  for (;;) {
    try {
      return join(fs.realpathSync(head), ...rest.reverse());
    } catch (error) {
      if (!isSystemError(error) || error.code !== "ENOENT" || dirname(head) === head) throw error;
      rest.push(basename(head));
      head = dirname(head);
    }
  }
}

/**
 * Return `error` with the memory's location taken out of its message: each `[text, name]` of
 * `replacements` in order, every occurrence of `text` replaced by `name`. An OS error names
 * absolute paths, and a session's errors can reach a model and a run log, which never hold a
 * location. A system error keeps its code.
 */
export function withoutLocation(error: unknown, replacements: readonly (readonly [string, string])[]): unknown {
  if (!(error instanceof Error) || !replacements.some(([text]) => error.message.includes(text))) return error;
  let message = error.message;
  for (const [text, name] of replacements) message = message.replaceAll(text, name);
  const copy: Error & { code?: unknown; syscall?: unknown; errno?: unknown } = new Error(message);
  copy.name = error.name;
  // Not `path`, `dest` or `cause`, which hold the location too.
  const { code, syscall, errno } = error as NodeJS.ErrnoException;
  if (code !== undefined) copy.code = code;
  if (syscall !== undefined) copy.syscall = syscall;
  if (errno !== undefined) copy.errno = errno;
  return copy;
}

/**
 * @internal Open the regular file at `path` with `flags`, never following a symlink at its last
 * component, and return its descriptor.
 *
 * @throws Error a Node system error when the open fails or `path` isn't a regular file.
 */
export function openRegular(path: string, flags: number, mode?: number): number {
  // Non-blocking, so a FIFO's open returns (or fails ENXIO for writing) instead of blocking the
  // thread until a peer opens it; regular files ignore the flag. The flags are missing on Windows.
  const { O_NOCTTY, O_NOFOLLOW, O_NONBLOCK } = fs.constants;
  const fd = fs.openSync(path, flags | (O_NONBLOCK ?? 0) | (O_NOCTTY ?? 0) | (O_NOFOLLOW ?? 0), mode);
  // Checked on the open descriptor, so a swap after a path check can't slip through.
  const stats = fs.fstatSync(fd);
  if (stats.isFile()) return fd;
  fs.closeSync(fd);
  throw Object.assign(new Error(`not a regular file: ${path}`), {
    code: stats.isDirectory() ? "EISDIR" : "EINVAL",
    syscall: "open",
  });
}

/** Return `path` in backticks for a message, a control character escaped. */
export function shown(path: string): string {
  return `\`${JSON.stringify(path).slice(1, -1)}\``;
}

/** Return the sha256 hex digest of `data`. */
export function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** Return `count` and `noun`, plural unless `count` is 1: `plural(2, "file")` is `"2 files"`. */
export function plural(count: number, noun: string, nouns = `${noun}s`): string {
  return `${count} ${count === 1 ? noun : nouns}`;
}

/** @internal Return the number of code points in `text`, which holds no lone surrogate. */
export function charCount(text: string): number {
  let chars = text.length;
  for (let i = 0; i < text.length; i++) {
    const unit = text.charCodeAt(i);
    if (unit >= 0xd800 && unit <= 0xdbff) chars--;
  }
  return chars;
}

/** Return the first `limit` code points of `text`. */
export function firstChars(text: string, limit: number): string {
  let index = 0;
  for (let n = 0; n < limit && index < text.length; n++) index += text.codePointAt(index)! > 0xffff ? 2 : 1;
  return text.slice(0, index);
}

/** Return `names` joined for a message: `a`, `a and b`, `a, b and c`. */
export function listed(names: readonly string[]): string {
  return names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
}

/** A refusal the calling model reads: a TOOL node doesn't retry it, since it can't change. */
export function refusal(message: string): ReportedToolError {
  return new ReportedToolError(message, { permanent: true });
}

/** Return a refusal that wrote nothing. */
export function refused(reason: string): Applied {
  return { ok: false, reason, current: null };
}

/** @internal Return whether `value` is a plain JSON object. */
export function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return `args[name]` when it is a string, or `undefined` when it is absent and not `required`.
 *
 * @param type The memory type, as the message names it.
 * @throws {TypeError} If it is given and isn't a string.
 */
export function stringArg(type: string, args: JsonObject, name: string, required: true): string;
export function stringArg(type: string, args: JsonObject, name: string, required: false): string | undefined;
export function stringArg(type: string, args: JsonObject, name: string, required: boolean): string | undefined {
  const value = Object.hasOwn(args, name) ? args[name] : undefined;
  if (value === undefined && !required) return undefined;
  if (typeof value !== "string") throw new YianTypeError(`${type} ${name} must be a string`);
  return value;
}

/** @internal Return `value` as an `Author`, with only its own fields, when it is one. */
export function asAuthor(value: unknown): Author | null {
  if (!isObject(value) || typeof value.name !== "string") return null;
  const { kind, name, run, node } = value;
  if (kind !== "agent" && kind !== "tool" && kind !== "user" && kind !== "host") return null;
  if ((run !== undefined && typeof run !== "string") || (node !== undefined && typeof node !== "string")) return null;
  return authorOf({ kind, name, run, node });
}

/** @internal Return `author` with only its own fields, in a fixed order. */
export function authorOf(author: Author): Author {
  const copy: Author = { kind: author.kind, name: author.name };
  if (author.run !== undefined) copy.run = author.run;
  if (author.node !== undefined) copy.node = author.node;
  return copy;
}

/**
 * Return `author` with only its own fields.
 *
 * @throws {TypeError} If it isn't `{kind, name, run?, node?}` with a known kind.
 */
export function checkAuthor(author: Author): Author {
  const checked = asAuthor(author);
  if (checked === null) {
    throw new YianTypeError("a change's author must be {kind, name, run?, node?}, its kind agent, tool, user or host");
  }
  return checked;
}
