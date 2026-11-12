/**
 * A FileMemory's journal, `.yian/journal.jsonl`: each change it applied with its outcome, and the
 * change-id table, written so that a crash at any point leaves a state the next writer resolves.
 *
 * Each line is one JSON object:
 * - the first line, `{generation}`, names this copy of the journal; rewriting it starts a new one;
 * - an entry, `{id, digest, op, path, author, at, before, after, inverse}`, is written and synced
 *   before its content is. `before` and `after` are the target's file hashes (`absent` for no
 *   file), and `inverse` is what undoing the change writes back;
 * - the entry's outcome follows it: a commit, `{id, ok: true, version, result}`, or, for a change
 *   that may have landed, `{id, ok: false, reason, current}`; or an abort, `{id, aborted: true}`,
 *   for one that didn't land, whose id is then free.
 *
 * Every writer reads what the others appended (catches up) before it writes, holding the
 * memory's lock, so only the journal's last entry can lack an outcome.
 *
 * It imports `platform`, `typesys`, `memory/base.ts` and `memory/common.ts`.
 */

import { randomBytes } from "node:crypto";
// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { dirname, join } from "node:path";

import { isSystemError, syncFolder } from "../platform.ts";
import { YianTypeError } from "../typesys/errors.ts";
import type { JsonObject } from "../typesys/json.ts";
import type { Applied, Author } from "./base.ts";
import { asAuthor, authorOf, charCount, isObject, openRegular } from "./common.ts";

/** The hash `before` or `after` holds when there is no file. */
export const ABSENT = "absent";

/**
 * What undoing a change writes back to its path: `text`, or the file cut back to its first
 * `truncate` bytes (an append's), or no file (`absent`). `null` when the change can't be undone.
 */
export type Inverse = { text: string } | { truncate: number } | { absent: true } | null;

/** One change as the journal records it, before its content is written. */
export interface JournalEntry {
  /** The change id. */
  readonly id: string;
  /** The sha256 of the change's canonical JSON, which a repeated id must match. */
  readonly digest: string;
  /** The change's `op`, such as `append`. */
  readonly op: string;
  /** The member path it changes. */
  readonly path: string;
  readonly author: Author;
  /** When it was applied, as an ISO 8601 time. */
  readonly at: string;
  /** The target's file hash before the change, or `absent`. */
  readonly before: string;
  /** The target's file hash after the change, or `absent`. */
  readonly after: string;
  readonly inverse: Inverse;
  /** For an undo, the id of the change it undid. */
  readonly undoes?: string;
  /** For an erase, the ids of the changes whose undo it disabled. */
  readonly disables?: readonly string[];
}

/** A change and its outcome, as the journal holds them: an aborted change has none. */
export interface JournalChange {
  readonly entry: JournalEntry;
  readonly applied: Applied;
}

/** The caps a journal keeps its change ids and inverses within. */
export interface JournalLimits {
  /** Change ids kept to answer a repeated change. */
  readonly maxChangeIds: number;
  /** Changes, newest first, whose inverses are kept. */
  readonly maxUndo: number;
  /** Characters of old text the kept inverses may hold. */
  readonly maxHistoryChars: number;
}

/** A change id's digest and its stored outcome. */
export interface KnownChange {
  readonly digest: string;
  readonly applied: Applied;
}

/** A change as a journal holds it, with the bytes its lines take, to tell when to compact. */
interface Held extends JournalChange {
  /** The entry line's bytes, its line break included. */
  readonly entryBytes: number;
  /** The entry line's bytes with no inverse. */
  readonly bareBytes: number;
  /** The outcome line's bytes. */
  readonly outcomeBytes: number;
}

/** One parsed journal line. */
type JournalLine =
  | { kind: "generation"; generation: string }
  | { kind: "entry"; entry: JournalEntry }
  | { kind: "commit"; id: string; applied: Applied }
  | { kind: "abort"; id: string };

/** The journal's name in `.yian/`, as messages show it. */
const SHOWN = "`.yian/journal.jsonl`";

/** @internal Write all of `bytes` to `fd` at its current position. */
export function writeAll(fd: number, bytes: Uint8Array): void {
  for (let done = 0; done < bytes.length; ) done += fs.writeSync(fd, bytes, done, bytes.length - done);
}

/**
 * @internal Write `data` to `target` atomically: a private temp file in `tmp`, synced, renamed
 * onto `target`, then both folders synced, so a crash leaves the old file or the new one.
 */
export function replaceFile(target: string, data: Uint8Array, tmp: string): void {
  const temporary = join(tmp, randomBytes(8).toString("hex"));
  const { O_CREAT, O_EXCL, O_WRONLY } = fs.constants;
  const fd = openRegular(temporary, O_CREAT | O_EXCL | O_WRONLY, 0o600);
  try {
    try {
      // The open's mode is masked by the umask; fchmod isn't.
      fs.fchmodSync(fd, 0o600);
      writeAll(fd, data);
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
    fs.renameSync(temporary, target);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
  syncFolder(dirname(target));
  syncFolder(tmp);
}

/**
 * @internal Make the private folder `path` unless it exists, and sync its parent when it was
 * made. `name` is how a message shows it.
 *
 * @throws Error when `path` exists and isn't a folder (a symlink to one included).
 */
export function makeDirectory(path: string, name: string): void {
  try {
    fs.mkdirSync(path, { mode: 0o700 });
  } catch (error) {
    if (!isSystemError(error) || error.code !== "EEXIST") throw error;
    if (!fs.lstatSync(path).isDirectory()) throw new Error(`\`${name}\` isn't a folder`);
    return;
  }
  fs.chmodSync(path, 0o700);
  syncFolder(dirname(path));
}

/** Read `bytes.length` bytes of `fd` from `position` into `bytes`. */
function readAll(fd: number, bytes: Uint8Array, position: number): void {
  for (let done = 0; done < bytes.length; ) {
    const read = fs.readSync(fd, bytes, done, bytes.length - done, position + done);
    if (read === 0) throw new Error(`${SHOWN} shrank while it was read`);
    done += read;
  }
}

/** Return the error for a damaged journal: what is wrong at line `line`, and the way out. */
function damaged(line: number, what: string): Error {
  return new Error(
    `${SHOWN} line ${line} ${what}; restore the journal or remove it (undo history and change ids are lost)`,
  );
}

/** Return `value` as an `Inverse` when it is one, or `undefined`. */
function asInverse(value: unknown): Inverse | undefined {
  if (value === null) return null;
  if (!isObject(value)) return undefined;
  const keys = Object.keys(value);
  if (keys.length !== 1) return undefined;
  if (typeof value.text === "string") return { text: value.text };
  if (typeof value.truncate === "number" && Number.isSafeInteger(value.truncate) && value.truncate >= 0) {
    return { truncate: value.truncate };
  }
  if (value.absent === true) return { absent: true };
  return undefined;
}

/** Return the journal line of `entry`, holding `inverse` as its inverse. */
function entryLine(entry: JournalEntry, inverse: Inverse): JsonObject {
  const { id, digest, op, path, author, at, before, after, undoes, disables } = entry;
  const line: JsonObject = { id, digest, op, path, author: authorOf(author), at, before, after, inverse };
  if (undoes !== undefined) line.undoes = undoes;
  if (disables !== undefined) line.disables = [...disables];
  return line;
}

/** Return the journal line `text` parsed, or `null` when it isn't one. */
function parseLine(text: string): JournalLine | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isObject(value)) return null;
  const keys = Object.keys(value);
  if (keys.length === 1 && typeof value.generation === "string") {
    return { kind: "generation", generation: value.generation };
  }
  const { id } = value;
  if (typeof id !== "string") return null;
  if (value.aborted === true && keys.length === 2) return { kind: "abort", id };
  if (value.ok === true && typeof value.version === "string" && isObject(value.result)) {
    return { kind: "commit", id, applied: { ok: true, version: value.version, result: value.result } };
  }
  if (value.ok === false && typeof value.reason === "string" && (value.current === null || isObject(value.current))) {
    return { kind: "commit", id, applied: { ok: false, reason: value.reason, current: value.current } };
  }
  const { digest, op, path, at, before, after, undoes, disables } = value;
  const author = asAuthor(value.author);
  const inverse = asInverse(value.inverse);
  if (
    (undoes !== undefined && typeof undoes !== "string") ||
    (disables !== undefined && !(Array.isArray(disables) && disables.every((id) => typeof id === "string"))) ||
    typeof digest !== "string" ||
    typeof op !== "string" ||
    typeof path !== "string" ||
    typeof at !== "string" ||
    typeof before !== "string" ||
    typeof after !== "string" ||
    author === null ||
    inverse === undefined
  ) {
    return null;
  }
  const entry: JournalEntry = { id, digest, op, path, author, at, before, after, inverse };
  if (undoes !== undefined) return { kind: "entry", entry: { ...entry, undoes } };
  if (disables !== undefined) return { kind: "entry", entry: { ...entry, disables: disables as string[] } };
  return { kind: "entry", entry };
}

/** A generation line as `#create` writes it, its hex digits as zeros. */
const GENERATION_LINE = '{"generation":"0000000000000000"}';

/**
 * Return whether the journal open at `fd`, of `size` bytes and with no line break, is the start of
 * a generation line, as a crash while writing one leaves it.
 */
function isCutGenerationLine(fd: number, size: number): boolean {
  if (size > GENERATION_LINE.length) return false;
  const bytes = Buffer.alloc(size);
  readAll(fd, bytes, 0);
  const text = bytes.toString("latin1");
  const digits = GENERATION_LINE.indexOf("0");
  const end = GENERATION_LINE.lastIndexOf("0");
  for (let i = 0; i < text.length; i++) {
    const hex = i >= digits && i <= end;
    if (hex ? !/[0-9a-f]/.test(text[i]!) : text[i] !== GENERATION_LINE[i]) return false;
  }
  return true;
}

/**
 * Return the first line of the journal open at `fd`, of `size` bytes, or `null` when a crash cut
 * it off (or the journal is empty).
 */
function readFirstLine(fd: number, size: number): string | null {
  const chunks: Buffer[] = [];
  for (let position = 0; position < size; ) {
    const chunk = Buffer.alloc(Math.min(256, size - position));
    readAll(fd, chunk, position);
    const end = chunk.indexOf(0x0a);
    if (end !== -1) {
      chunks.push(chunk.subarray(0, end));
      return Buffer.concat(chunks).toString("utf8");
    }
    chunks.push(chunk);
    position += chunk.length;
  }
  return null;
}

/**
 * One session's view of a memory's journal: how far it has read, the change-id table, and the
 * entry with no outcome yet, if any. Every method runs holding the memory's lock.
 */
export class Journal {
  readonly #meta: string;
  readonly #path: string;
  readonly #limits: JournalLimits;
  /** The generation read, or `null` before a read and while there is no journal. */
  #generation: string | null = null;
  /** The bytes read, ending at a line break. */
  #offset = 0;
  /** The lines read. */
  #lines = 0;
  /** The last `maxChangeIds` change ids with an outcome, oldest first. */
  readonly #ids = new Map<string, KnownChange>();
  /** The changes the journal holds with an outcome, in journal order. */
  #changes: Held[] = [];
  /** The last entry, while it has no outcome, its line and its bytes. */
  #pending: { entry: JournalEntry; line: number; bytes: number } | null = null;
  /** The bytes of the generation line. */
  #headerBytes = 0;

  /**
   * @param meta The memory's `.yian` folder.
   * @param limits The caps it keeps change ids and inverses within.
   */
  constructor(meta: string, limits: JournalLimits) {
    this.#meta = meta;
    this.#path = join(meta, "journal.jsonl");
    this.#limits = limits;
  }

  /**
   * Read what other writers appended since the last read: the whole journal when it was rewritten
   * (a new generation) or removed. A last line a crash cut off is dropped from the file.
   *
   * @returns The last entry when it has no outcome yet, else `null`.
   * @throws Error when a line isn't a journal line, or the lines are out of order.
   */
  catchUp(): JournalEntry | null {
    let fd: number;
    try {
      fd = openRegular(this.#path, fs.constants.O_RDWR);
    } catch (error) {
      if (!isSystemError(error) || error.code !== "ENOENT") throw error;
      this.#reset(null);
      return null;
    }
    try {
      this.#read(fd);
    } catch (error) {
      // A half-read journal must not answer for change ids; the next catch-up reads it whole.
      this.#reset(null);
      throw error;
    } finally {
      fs.closeSync(fd);
    }
    return this.#pending?.entry ?? null;
  }

  /** Return change `id`'s digest and stored outcome, or `undefined` when the table doesn't hold it. */
  known(id: string): KnownChange | undefined {
    return this.#ids.get(id);
  }

  /** Return the changes the journal holds with an outcome, oldest first. */
  changes(): readonly JournalChange[] {
    return this.#changes;
  }

  /** Return the latest change with id `id` and an outcome, or `undefined`. */
  find(id: string): JournalChange | undefined {
    return this.#changes.findLast((change) => change.entry.id === id);
  }

  /**
   * Return the changes whose inverses the caps keep: among the last `maxUndo` changes, newest
   * first, each committed one with an inverse whose old text still fits in `maxHistoryChars`.
   */
  undoable(): Set<JournalChange> {
    const kept = new Set<JournalChange>();
    const { maxUndo, maxHistoryChars } = this.#limits;
    let chars = 0;
    for (let i = this.#changes.length - 1; i >= Math.max(0, this.#changes.length - maxUndo); i--) {
      const change = this.#changes[i]!;
      const { inverse } = change.entry;
      if (inverse === null || !change.applied.ok) continue;
      const cost = "text" in inverse ? charCount(inverse.text) : 0;
      // An old text too big for what is left is dropped; older, smaller ones may still fit.
      if (chars + cost > maxHistoryChars) continue;
      chars += cost;
      kept.add(change);
    }
    return kept;
  }

  /** Append `entry` and sync it, making the journal first when there is none. */
  writeEntry(entry: JournalEntry): void {
    this.#append(entryLine(entry, entry.inverse));
  }

  /**
   * Return whether the journal has grown past twice what a compaction would keep: the changes
   * whose ids the table holds, and the inverses the caps keep.
   */
  shouldCompact(): boolean {
    let kept = this.#headerBytes;
    for (const { change, inverse } of this.#kept()) {
      kept += (inverse ? change.entryBytes : change.bareBytes) + change.outcomeBytes;
    }
    return this.#offset > 2 * kept;
  }

  /**
   * Rewrite the journal as a new generation holding only what the caps keep: each change whose id
   * the table holds, with its outcome, and the inverses `undoable` keeps except those `drop` picks;
   * other inverses become `null`, and aborted changes go. Hold the lock, with no entry pending.
   */
  compact(drop: (change: JournalChange) => boolean = () => false): void {
    this.#rewrite(this.#keptLines(drop));
  }

  /**
   * Rewrite the journal as `compact` does, with no inverse that `drop` picks, and end it with the
   * erase `entry`, whose outcome comes once its content is written. Hold the lock, with no entry
   * pending. The text the dropped inverses held is then gone from the journal.
   */
  erase(entry: JournalEntry, drop: (change: JournalChange) => boolean): void {
    this.#rewrite([...this.#keptLines(drop), JSON.stringify(entryLine(entry, null))]);
  }

  /** Start the journal again, as a new generation with no changes. Hold the lock. */
  restart(): void {
    this.#rewrite([]);
  }

  /** Return the lines a compaction keeps, with no inverse that `drop` picks. */
  #keptLines(drop: (change: JournalChange) => boolean): string[] {
    return this.#kept().flatMap(({ change, inverse }) => [
      JSON.stringify(entryLine(change.entry, inverse && !drop(change) ? change.entry.inverse : null)),
      JSON.stringify({ id: change.entry.id, ...change.applied }),
    ]);
  }

  /** Return the changes a compaction keeps, in journal order, and whether each keeps its inverse. */
  #kept(): { change: Held; inverse: boolean }[] {
    const inverses = this.undoable();
    const latest = new Map<string, Held>();
    for (const change of this.#changes) latest.set(change.entry.id, change);
    const tabled = new Set<Held>();
    for (const id of this.#ids.keys()) {
      const change = latest.get(id);
      if (change !== undefined) tabled.add(change);
    }
    return this.#changes
      .filter((change) => tabled.has(change) || inverses.has(change))
      .map((change) => ({ change, inverse: inverses.has(change) }));
  }

  /**
   * Replace the journal with a new generation holding `lines`, through a synced temp file in
   * `.yian/tmp/`, then read it whole. A session that read the old generation sees the new one on
   * its first line and reads it whole too.
   */
  #rewrite(lines: string[]): void {
    // A rewrite keeps only changes with an outcome, so a pending entry would be lost unresolved.
    if (this.#pending !== null) {
      throw new Error(`${SHOWN} can't be rewritten while change \`${this.#pending.entry.id}\` has no outcome`);
    }
    const generation = randomBytes(8).toString("hex");
    const text = `${[JSON.stringify({ generation }), ...lines].join("\n")}\n`;
    const tmp = join(this.#meta, "tmp");
    makeDirectory(tmp, ".yian/tmp");
    replaceFile(this.#path, Buffer.from(text, "utf8"), tmp);
    this.#reset(null);
    this.catchUp();
  }

  /** Append the outcome of the pending entry `id`, a commit or an abort, and sync it. */
  writeOutcome(id: string, outcome: Applied | "aborted"): void {
    this.#append(outcome === "aborted" ? { id, aborted: true } : { id, ...outcome });
  }

  /** Read the journal open at `fd` from where the last read stopped, or whole when it changed. */
  #read(fd: number): void {
    const size = fs.fstatSync(fd).size;
    const first = readFirstLine(fd, size);
    if (first === null) {
      // Only a generation line a crash cut off has no line break, and no entry follows it. Anything
      // else is damage, which must not be wiped: its change ids would apply again.
      if (!isCutGenerationLine(fd, size)) throw damaged(1, "isn't a journal line");
      fs.ftruncateSync(fd, 0);
      fs.fsyncSync(fd);
      this.#reset(null);
      return;
    }
    const header = parseLine(first);
    if (header?.kind !== "generation") throw damaged(1, "isn't a journal line");
    this.#headerBytes = Buffer.byteLength(first, "utf8") + 1;
    if (header.generation !== this.#generation || size < this.#offset) this.#reset(header.generation);
    if (size === this.#offset) return;
    const bytes = Buffer.alloc(size - this.#offset);
    readAll(fd, bytes, this.#offset);
    const end = bytes.lastIndexOf(0x0a) + 1;
    if (end < bytes.length) {
      // A crash cut off the last line while it was written: its change never started (an entry)
      // or its outcome is decided again (a commit or an abort). Later lines must not join it.
      fs.ftruncateSync(fd, this.#offset + end);
      fs.fsyncSync(fd);
    }
    const lines = bytes.subarray(0, end).toString("utf8").split("\n");
    lines.pop();
    for (const text of lines) {
      const line = this.#lines + 1;
      const parsed = parseLine(text);
      if (parsed === null) throw damaged(line, "isn't a journal line");
      this.#fold(parsed, line, Buffer.byteLength(text, "utf8") + 1);
      this.#lines = line;
    }
    this.#offset += end;
  }

  /**
   * Check that `parsed` may be line `line`: a generation line comes first and only first, an entry
   * follows an outcome, and an outcome follows its own entry.
   *
   * @throws Error naming the line, as for a damaged journal.
   */
  #checkOrder(parsed: JournalLine, line: number): void {
    if ((parsed.kind === "generation") !== (line === 1)) throw damaged(line, "isn't a journal line");
    if (parsed.kind === "entry" && this.#pending !== null) {
      throw damaged(this.#pending.line, "has an entry with no outcome before later entries");
    }
    if ((parsed.kind === "commit" || parsed.kind === "abort") && this.#pending?.entry.id !== parsed.id) {
      throw damaged(line, "has an outcome for no entry");
    }
  }

  /** Take one journal line, of `bytes` bytes, into the table, the changes and the pending entry. */
  #fold(parsed: JournalLine, line: number, bytes: number): void {
    this.#checkOrder(parsed, line);
    if (parsed.kind === "generation") return;
    if (parsed.kind === "entry") {
      this.#pending = { entry: parsed.entry, line, bytes };
      return;
    }
    // An outcome: `#checkOrder` made sure its entry is the pending one.
    const { entry, bytes: entryBytes } = this.#pending!;
    if (parsed.kind === "commit") {
      this.#remember(parsed.id, entry.digest, parsed.applied);
      // The same line with `null` for its inverse: what a compaction that drops it writes.
      const bareBytes =
        entry.inverse === null ? entryBytes : entryBytes - Buffer.byteLength(JSON.stringify(entry.inverse)) + 4;
      this.#changes.push({ entry, applied: parsed.applied, entryBytes, bareBytes, outcomeBytes: bytes });
    }
    this.#pending = null;
  }

  /** Record change `id`'s outcome, dropping the oldest ids past `maxChangeIds`. */
  #remember(id: string, digest: string, applied: Applied): void {
    // An id applied again after it was dropped moves to the newest end.
    this.#ids.delete(id);
    this.#ids.set(id, { digest, applied });
    for (const old of this.#ids.keys()) {
      if (this.#ids.size <= this.#limits.maxChangeIds) break;
      this.#ids.delete(old);
    }
  }

  /** Forget what was read: the next read starts at `generation`'s first line. */
  #reset(generation: string | null): void {
    this.#generation = generation;
    this.#offset = 0;
    this.#lines = 0;
    this.#ids.clear();
    this.#changes = [];
    this.#pending = null;
  }

  /** Append `record` as one line and sync it, then take it in as a read line. */
  #append(record: JsonObject): void {
    const text = JSON.stringify(record);
    // Never write a line the next read would refuse.
    const parsed = parseLine(text);
    if (parsed === null) throw new YianTypeError(`not a journal line: ${text}`);
    // Checked against an empty journal's first entry line when there is none yet.
    this.#checkOrder(parsed, this.#generation === null ? 2 : this.#lines + 1);
    if (this.#generation === null) this.#create();
    const bytes = Buffer.from(`${text}\n`, "utf8");
    const fd = openRegular(this.#path, fs.constants.O_WRONLY | fs.constants.O_APPEND);
    try {
      writeAll(fd, bytes);
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
    this.#lines++;
    this.#offset += bytes.length;
    this.#fold(parsed, this.#lines, bytes.length);
  }

  /** Start the journal: its generation line, synced with its entry in `.yian/`. */
  #create(): void {
    const generation = randomBytes(8).toString("hex");
    const bytes = Buffer.from(`${JSON.stringify({ generation })}\n`, "utf8");
    const { O_APPEND, O_CREAT, O_WRONLY } = fs.constants;
    // Never truncated: a catch-up drops a cut-off first line itself, so anything here is a journal
    // this session hasn't read, which must not be overwritten.
    const fd = openRegular(this.#path, O_CREAT | O_WRONLY | O_APPEND, 0o600);
    try {
      if (fs.fstatSync(fd).size !== 0) {
        throw new Error(`${SHOWN} isn't empty, but this session hasn't read it; try again`);
      }
      // The open's mode is masked by the umask; fchmod isn't.
      fs.fchmodSync(fd, 0o600);
      writeAll(fd, bytes);
      fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
    syncFolder(this.#meta);
    this.#reset(generation);
    this.#lines = 1;
    this.#offset = bytes.length;
    this.#headerBytes = bytes.length;
  }
}
