/**
 * FileMemory: a memory that is a folder of UTF-8 text files, with an optional index file.
 *
 * Its members are the regular files whose path components are plain names, within the depth cap,
 * holding UTF-8 text within the per-file cap. Everything else in the folder (hidden entries such
 * as `.git` and `.yian/`, other names, symlinks, files with another hard link, binary or oversized
 * files) isn't part of the memory: tools refuse it and the version leaves it out. The memory's own metadata lives in
 * `<root>/.yian/`: `identity`, the journal, the lock and a temp folder.
 *
 * It imports `defaults`, `diagnostics`, `platform`, `tools` and `typesys`, and `memory/base.ts` and
 * `memory/file_journal.ts`.
 */

import { createHash, randomBytes } from "node:crypto";
// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";

import { current } from "../defaults.ts";
import { compareCodeUnits } from "../diagnostics.ts";
import {
  type FileLock,
  FileLockError,
  isSystemError,
  realScheduler,
  type Scheduler,
  syncFolder,
  waitForLock,
} from "../platform.ts";
import { ReportedToolError } from "../tools/errors.ts";
import { canonicalJson, type JsonObject } from "../typesys/json.ts";
import {
  type Applied,
  type Author,
  type ChangeEntry,
  ChangeIdReusedError,
  type Erased,
  MemoryBase,
  MemoryBusyError,
  MemoryMissingError,
  type MemoryContract,
  type MemoryInfo,
  type MemorySession,
  type MemoryToolSpec,
  type Recall,
} from "./base.ts";
import {
  ABSENT,
  asAuthor,
  authorOf,
  charCount,
  type Inverse,
  Journal,
  type JournalChange,
  type JournalEntry,
  isObject,
  makeDirectory,
  openRegular,
  replaceFile,
} from "./file_journal.ts";

/** The arguments `path` takes in every tool. */
const PATH_PARAMETER: JsonObject = {
  type: "string",
  description:
    "A file's path in the memory, such as `notes/tools.md`: folders and a file name made of letters, digits, " +
    "`.`, `_` and `-`, each starting with a letter or digit.",
};

/** Return a tool spec whose arguments are `properties`, the ones in `required` required. */
function toolSpec(
  name: string,
  write: boolean,
  description: string,
  properties: JsonObject,
  required: string[],
): MemoryToolSpec {
  return Object.freeze({
    name,
    write,
    description,
    parameters: { type: "object", properties, required, additionalProperties: false },
  });
}

/** The FileMemory contract's tools, in the order a whole grant shows them. */
const TOOLS: readonly MemoryToolSpec[] = Object.freeze([
  toolSpec(
    "list",
    false,
    "List the memory's files, each with its hash and its size in characters. Pass dir to list one folder.",
    { dir: { type: "string", description: "A folder of the memory to list; the whole memory when omitted." } },
    [],
  ),
  toolSpec(
    "read",
    false,
    "Read a file of the memory: its text, from line offset for limit lines, and the file's hash, which " +
      "delete and rewrite take as if_hash.",
    {
      path: PATH_PARAMETER,
      offset: {
        type: "integer",
        minimum: 1,
        description: "The first line to read, counting from one; one when omitted.",
      },
      limit: { type: "integer", minimum: 1, description: "The most lines to read; the rest of the file when omitted." },
    },
    ["path"],
  ),
  toolSpec(
    "search",
    false,
    "Find the lines that hold query, ignoring case, in every file of the memory. Each match gives its path, " +
      "line number and text, and the file's hash.",
    { query: { type: "string", description: "The text to find." } },
    ["query"],
  ),
  toolSpec(
    "append",
    true,
    "Add text to the end of a file, on a new line. Creates the file and its folders when missing, and never " +
      "conflicts.",
    { path: PATH_PARAMETER, text: { type: "string", description: "The text to add." } },
    ["path", "text"],
  ),
  toolSpec(
    "create",
    true,
    "Create a file holding text, and its folders when missing. Conflicts when the file exists.",
    { path: PATH_PARAMETER, text: { type: "string", description: "The file's whole text." } },
    ["path", "text"],
  ),
  toolSpec(
    "edit",
    true,
    "Replace the text old with new in a file. Conflicts unless old appears in the file exactly once; read the " +
      "file first.",
    {
      path: PATH_PARAMETER,
      old: { type: "string", description: "The exact text to replace, as the file holds it." },
      new: { type: "string", description: "The text to put in its place." },
    },
    ["path", "old", "new"],
  ),
  toolSpec(
    "delete",
    true,
    "Delete a file. Conflicts unless the file's hash is if_hash, as read gives it.",
    { path: PATH_PARAMETER, if_hash: { type: "string", description: "The file's hash, as read gave it." } },
    ["path", "if_hash"],
  ),
  toolSpec(
    "rewrite",
    true,
    "Replace the whole text of a file. Conflicts unless the file's hash is if_hash, as read gives it.",
    {
      path: PATH_PARAMETER,
      text: { type: "string", description: "The file's whole new text." },
      if_hash: { type: "string", description: "The file's hash, as read gave it." },
    },
    ["path", "text", "if_hash"],
  ),
]);

/** A plain name: what every component of a member's path must be. */
const NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** Why `open({create: false})` and `reset` refuse a folder that isn't a memory yet. */
const NOT_A_MEMORY = "the memory isn't at its location: its folder isn't a memory yet";

/** The folder under the root that holds the memory's own metadata, never a member. */
const META = ".yian";

/** The settings and caps a `FileMemory` is built with; each omitted one comes from `src/defaults.ts`. */
export interface FileMemoryOptions {
  /** The memory's folder. A symlinked root is resolved once, here. */
  root: string;
  /** The identity a new memory takes; an existing memory's stored identity must equal it. */
  id?: string;
  /** The index file, whose text is the recall when it exists (default `MEMORY.md`). */
  index?: string;
  /** Characters one file may hold. */
  maxFileChars?: number;
  /** Characters all files may hold. */
  maxTotalChars?: number;
  /** Files the memory may hold. */
  maxFiles?: number;
  /** Folders a file may sit under; 0 keeps every file at the top. */
  maxDepth?: number;
  /** Characters of the recall. */
  maxRecallChars?: number;
  /**
   * Characters of text one read tool result shows: the paths `list` shows, the lines `read` shows,
   * and each `search` match's path and line. Each shows at least one entry, a line cut to fit.
   */
  maxReadChars?: number;
  /** Entries `list` shows and lines `search` shows. */
  maxResults?: number;
  /** Changes the memory can undo. */
  maxUndo?: number;
  /**
   * Change ids the memory keeps to answer a repeated change. Every writer on one root must use the
   * same value: one that keeps fewer ids may apply again a change another still answers.
   */
  maxChangeIds?: number;
  /** Characters of old text the memory keeps for undo (default `maxTotalChars`). */
  maxHistoryChars?: number;
  /** Longest wait in milliseconds for the memory's lock; 0 fails at once when it is held. */
  lockTimeoutMs?: number;
  /**
   * The sleep the lock wait uses between tries, and the clock that stamps changes. A wait's
   * deadline runs on a monotonic clock, not this one.
   */
  scheduler?: Scheduler;
}

/** The caps a FileMemory checks, as numbers. */
interface Limits {
  readonly maxFileChars: number;
  readonly maxTotalChars: number;
  readonly maxFiles: number;
  readonly maxDepth: number;
  readonly maxRecallChars: number;
  readonly maxReadChars: number;
  readonly maxResults: number;
  readonly maxUndo: number;
  readonly maxChangeIds: number;
  readonly maxHistoryChars: number;
}

/**
 * Return `value` when it is a safe integer of at least `minimum`, else its default.
 *
 * @throws {RangeError} If `value` is given and isn't such an integer.
 */
function capOption(name: string, value: number | undefined, fallback: number, minimum: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`FileMemory ${name} must be an integer of at least ${minimum}, not ${String(value)}`);
  }
  return value;
}

/** Return `path` in backticks for a message, a control character escaped. */
function shown(path: string): string {
  return `\`${JSON.stringify(path).slice(1, -1)}\``;
}

/**
 * Return why the relative path `rel` can't name a member, or with `folder` a folder of members, by
 * its text alone, or `null` when it can. Its components are plain names, and a file sits under at
 * most `maxDepth` folders.
 */
function pathProblem(rel: string, maxDepth: number, folder = false): string | null {
  if (rel === "") return "an empty path";
  if (isAbsolute(rel)) return "an absolute path";
  // Past the common file system limits the OS fails the write after its change is journaled, so
  // such a path is refused here, with the rest of the path's checks.
  if (Buffer.byteLength(rel, "utf8") > 1024) return "a path longer than 1024 bytes";
  const parts = rel.split("/");
  for (const part of parts) {
    if (part === "") return "an empty part";
    if (part === "..") return "a `..` part";
    if (part === ".") return "a `.` part";
    if (part.startsWith(".")) return "a hidden name";
    if (!NAME.test(part)) return "a name other than letters, digits, `.`, `_` and `-`, starting with a letter or digit";
    if (part.length > 255) return "a name longer than 255 bytes";
  }
  const folders = folder ? parts.length : parts.length - 1;
  if (folders > maxDepth) return `more than ${maxDepth} folder${maxDepth === 1 ? "" : "s"} deep`;
  return null;
}

/**
 * Return why the existing components of `rel` under `root` can't lead to a member, or with
 * `folder` to a folder of members, checked with `lstat`, or `null` when they can. A missing
 * component is fine: a write creates it.
 *
 * The check and the use are separate path walks: Node has no `openat`, so a local process that
 * can write in the memory's folder and swaps a checked folder for a symlink in between can
 * redirect a read or a write. Only the last component is opened without following a symlink.
 */
function diskProblem(root: string, rel: string, folder = false): string | null {
  const parts = rel.split("/");
  let path = root;
  for (let i = 0; i < parts.length; i++) {
    path = join(path, parts[i]!);
    const stats = lstatOrNull(path);
    if (stats === null) return null;
    const last = i === parts.length - 1;
    if (stats.isSymbolicLink()) return last ? "a symlink" : `${shown(parts.slice(0, i + 1).join("/"))} is a symlink`;
    if (!last || folder) {
      if (stats.isDirectory()) continue;
      if (!last) return `${shown(parts.slice(0, i + 1).join("/"))} isn't a folder`;
      return stats.isFile() ? "a file" : "not a folder";
    }
    if (!stats.isFile()) return stats.isDirectory() ? "a folder" : "not a regular file";
    // A file with another link may be outside the root: it is read through like a symlink, and
    // keeps text that an erase removed from the memory.
    if (stats.nlink > 1) return "a hard link";
  }
  return null;
}

/** Return the first `limit` code points of `text`. */
function firstChars(text: string, limit: number): string {
  let index = 0;
  for (let n = 0; n < limit && index < text.length; n++) index += text.codePointAt(index)! > 0xffff ? 2 : 1;
  return text.slice(0, index);
}

/** Return `count` and `noun`, plural unless `count` is 1: `plural(2, "file")` is `"2 files"`. */
function plural(count: number, noun: string, nouns = `${noun}s`): string {
  return `${count} ${count === 1 ? noun : nouns}`;
}

/** Return the lines of `text`: a final line break ends the last line rather than starting one. */
function linesOf(text: string): string[] {
  if (text === "") return [];
  const lines = text.split("\n");
  if (lines.at(-1) === "") lines.pop();
  return lines;
}

/** Return the sha256 hex digest of `data`. */
function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/**
 * @internal One member: its path under the root, its text, the sha256 of its bytes and its
 * characters.
 */
export interface MemberFile {
  readonly path: string;
  readonly text: string;
  readonly hash: string;
  readonly chars: number;
}

/**
 * Read the file at `path` as a member, or return why it isn't one (not a regular file, a hard
 * link, not UTF-8 text, or over `maxFileChars` characters), or `null` when nothing is there.
 */
function readMemberFile(path: string, rel: string, maxFileChars: number): MemberFile | string | null {
  let fd: number;
  try {
    fd = openRegular(path, fs.constants.O_RDONLY);
  } catch (error) {
    if (!isSystemError(error)) throw error;
    // A writer may delete the file or replace it while a read walks the folder without the lock.
    if (error.code === "ENOENT") return null;
    if (error.code === "ELOOP") return "a symlink";
    if (error.code === "EISDIR") return "a folder";
    if (error.code === "EINVAL" || error.code === "ENXIO") return "not a regular file";
    throw error;
  }
  let bytes: Buffer;
  try {
    const stats = fs.fstatSync(fd);
    if (stats.nlink > 1) return "a hard link";
    // A UTF-8 character takes at most four bytes, so a bigger file is over the cap unread.
    if (stats.size > 4 * maxFileChars) return `over ${maxFileChars} characters`;
    bytes = fs.readFileSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  let text: string;
  try {
    // Fatal, so invalid bytes fail instead of turning into U+FFFD; a BOM is kept as text.
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch {
    return "not UTF-8 text";
  }
  const chars = charCount(text);
  if (chars > maxFileChars) return `over ${maxFileChars} characters`;
  return { path: rel, text, hash: sha256(bytes), chars };
}

/**
 * @internal Read the members under `root`, sorted by path in code-unit order, and count the
 * entries that aren't members. `.yian/` is the memory's own and isn't counted. Walks with `lstat`
 * and follows no symlink; a skipped folder counts once.
 */
export function readMembers(
  root: string,
  limits: { readonly maxDepth: number; readonly maxFileChars: number },
): { files: MemberFile[]; skipped: number } {
  const files: MemberFile[] = [];
  let skipped = 0;
  const walk = (rel: string, folders: number): void => {
    let names: string[];
    try {
      names = fs.readdirSync(rel === "" ? root : join(root, rel));
    } catch (error) {
      // A folder removed by hand while the walk ran holds nothing.
      if (rel !== "" && isSystemError(error) && error.code === "ENOENT") return;
      throw error;
    }
    for (const name of names) {
      if (rel === "" && name === META) continue;
      const path = rel === "" ? name : `${rel}/${name}`;
      if (!NAME.test(name)) {
        skipped++;
        continue;
      }
      const stats = lstatOrNull(join(root, path));
      if (stats === null) continue;
      if (stats.isDirectory() && folders + 1 <= limits.maxDepth) {
        walk(path, folders + 1);
      } else if (stats.isFile()) {
        const file = readMemberFile(join(root, path), path, limits.maxFileChars);
        if (typeof file === "string") skipped++;
        else if (file !== null) files.push(file);
      } else {
        skipped++;
      }
    }
  };
  walk("", 0);
  files.sort((a, b) => compareCodeUnits(a.path, b.path));
  return { files, skipped };
}

/**
 * @internal Return the version of a memory whose members are `files`: the sha256 hex digest of
 * the canonical JSON list of `[path, file hash]` pairs, sorted by path in code-unit order.
 */
export function memoryVersion(files: readonly MemberFile[]): string {
  const pairs = files.map((file) => [file.path, file.hash]);
  pairs.sort((a, b) => compareCodeUnits(a[0]!, b[0]!));
  return sha256(canonicalJson(pairs));
}

/** Return `path`'s real path, resolving the symlinks of the part of it that exists. */
function realRoot(path: string): string {
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
 * Return the recall of a memory whose members are `members`: the index's text when it is a
 * member, else each member's path and first line, in path order, within `maxRecallChars`. A
 * member whose line doesn't fit shows its path alone, if that fits.
 */
function recallOf(
  members: { files: MemberFile[]; skipped: number },
  index: string,
  maxRecallChars: number,
): Recall {
  const indexFile = members.files.find((file) => file.path === index);
  if (indexFile !== undefined) {
    const over = indexFile.chars - maxRecallChars;
    return {
      text: over > 0 ? firstChars(indexFile.text, maxRecallChars) : indexFile.text,
      cut: over > 0 ? `${plural(over, "more character")} of ${index} not shown` : null,
      hint: over > 0 || members.files.length > 1 ? { tool: "read", text: "Read the rest with {tool}." } : null,
    };
  }
  const lines: string[] = [];
  let used = 0;
  for (const file of members.files) {
    const first = (linesOf(file.text)[0] ?? "").replace(/\r$/, "");
    // A whole first line, or else the path alone: a line cut short would misquote its file, and
    // one long line mustn't hide the files after it.
    for (const line of first === "" ? [file.path] : [`${file.path}: ${first}`, file.path]) {
      const cost = charCount(line) + (lines.length > 0 ? 1 : 0);
      if (used + cost > maxRecallChars) continue;
      lines.push(line);
      used += cost;
      break;
    }
  }
  const notes: string[] = [];
  const unshown = members.files.length - lines.length;
  if (unshown > 0) notes.push(`${plural(unshown, "more file")} not shown`);
  if (members.skipped > 0) notes.push(`${plural(members.skipped, "entry", "entries")} skipped: not part of the memory`);
  return {
    text: lines.join("\n"),
    cut: notes.length > 0 ? notes.join("; ") : null,
    hint: members.files.length > 0 ? { tool: "read", text: "Read them with {tool}." } : null,
  };
}

/** A refusal the calling model reads: a TOOL node doesn't retry it, since it can't change. */
function refusal(message: string): ReportedToolError {
  return new ReportedToolError(message, { permanent: true });
}

/** Return `args[name]` when it is a string, or `undefined` when it is absent and not `required`. */
function stringArg(args: JsonObject, name: string, required: true): string;
function stringArg(args: JsonObject, name: string, required: false): string | undefined;
function stringArg(args: JsonObject, name: string, required: boolean): string | undefined {
  const value = Object.hasOwn(args, name) ? args[name] : undefined;
  if (value === undefined && !required) return undefined;
  if (typeof value !== "string") throw new TypeError(`FileMemory ${name} must be a string`);
  return value;
}

/** Return `args[name]` as a number when it is an integer of at least 1, or `undefined` when absent. */
function countArg(args: JsonObject, name: string): number | undefined {
  const value = Object.hasOwn(args, name) ? args[name] : undefined;
  if (value === undefined) return undefined;
  const number = typeof value === "bigint" ? Number(value) : value;
  if (typeof number !== "number" || !Number.isSafeInteger(number) || number < 1) {
    throw new TypeError(`FileMemory ${name} must be an integer of at least 1`);
  }
  return number;
}

/** A FileMemory change, checked for shape: a write tool's arguments plus `op`, its name. */
type Change =
  | { op: "append" | "create"; path: string; text: string }
  | { op: "edit"; path: string; old: string; new: string }
  | { op: "delete"; path: string; if_hash: string }
  | { op: "rewrite"; path: string; text: string; if_hash: string };

/** The fields each change takes besides `op`, in order. */
const CHANGE_FIELDS: Readonly<Record<Change["op"], readonly string[]>> = Object.freeze({
  append: ["path", "text"],
  create: ["path", "text"],
  edit: ["path", "old", "new"],
  delete: ["path", "if_hash"],
  rewrite: ["path", "text", "if_hash"],
});

/** Return `names` joined for a message: `a`, `a and b`, `a, b and c`. */
function listed(names: readonly string[]): string {
  return names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
}

/**
 * Return `change` as a FileMemory change.
 *
 * @throws {TypeError} If its `op` isn't a FileMemory change, or its fields aren't that change's,
 *   each a string.
 */
function checkChange(change: JsonObject): Change {
  const op: unknown = isObject(change) ? change.op : undefined;
  if (typeof op !== "string" || !Object.hasOwn(CHANGE_FIELDS, op)) {
    const ops = Object.keys(CHANGE_FIELDS).join(", ");
    throw new TypeError(`FileMemory has no change ${String(op)}; its changes are ${ops}`);
  }
  const fields = CHANGE_FIELDS[op as Change["op"]];
  const extra = Object.keys(change).filter((key) => key !== "op" && !fields.includes(key));
  if (extra.length > 0) throw new TypeError(`a FileMemory ${op} takes ${listed(fields)}, not ${listed(extra)}`);
  for (const field of fields) {
    const value = change[field];
    if (typeof value !== "string") throw new TypeError(`a FileMemory ${op} needs ${field} as a string`);
    if (!value.isWellFormed()) throw new TypeError(`a FileMemory ${op}'s ${field} holds a lone surrogate`);
  }
  return change as unknown as Change;
}

/**
 * Return `author` with only its own fields.
 *
 * @throws {TypeError} If it isn't `{kind, name, run?, node?}` with a known kind.
 */
function checkAuthor(author: Author): Author {
  const checked = asAuthor(author);
  if (checked === null) {
    throw new TypeError("a change's author must be {kind, name, run?, node?}, its kind agent, tool, user or host");
  }
  return checked;
}

/** Return a refusal that wrote nothing. */
function refused(reason: string): Applied {
  return { ok: false, reason, current: null };
}

/**
 * Return `inverse` as a new change's journal entry keeps it: `null` when no undo could ever use it,
 * with no undos kept or an old text too big for the history on its own, so the old text never
 * reaches the journal.
 */
function storedInverse(inverse: Inverse, limits: Limits): Inverse {
  if (inverse === null || limits.maxUndo === 0) return null;
  if ("text" in inverse && charCount(inverse.text) > limits.maxHistoryChars) return null;
  return inverse;
}

/**
 * Return `content` with every occurrence of `text` removed, again until none is left: removing
 * one can join its neighbours into another (`secsecretret` without `secret` is `secret`).
 */
function withoutText(content: string, text: string): string {
  let rest = content;
  while (rest.includes(text)) rest = rest.replaceAll(text, "");
  return rest;
}

/** Return the sum of `files`' characters. */
function totalChars(files: readonly MemberFile[]): number {
  return files.reduce((sum, file) => sum + file.chars, 0);
}

/** What a session needs of its memory: the folder, its caps, its lock and its clock. */
interface Store {
  /** The memory's folder, symlinks resolved. */
  readonly root: string;
  readonly index: string;
  readonly limits: Limits;
  readonly scheduler: Scheduler;
  /**
   * Run `body` holding the memory's lock, after this process's earlier calls on the memory: they
   * take turns, so only another process's writer makes it wait for the lock. It waits at most the
   * lock timeout from when it was called, its time in the queue included, and an abort while it
   * is queued rejects it at once.
   *
   * Returns the call's `result`, which rejects with `MemoryBusyError` if the lock is still held by
   * another writer at the deadline, and `done`, which settles once its turn has finished, later
   * than `result` when an abort let the call go before its turn came.
   */
  locked<T>(signal: AbortSignal | undefined, body: Held<T>): Turn<T>;
}

/** A call's place in a memory's queue: its result, and when its turn has finished. */
interface Turn<T> {
  readonly result: Promise<T>;
  /** Settles, never rejecting, once the turn has run or been passed over. */
  readonly done: Promise<void>;
}

/**
 * A body that runs holding a memory's lock. It must finish before it returns, so it can't be async:
 * the lock is released when it returns, and an async body would go on without it.
 */
type Held<T> = (() => T) & (T extends PromiseLike<unknown> ? never : unknown);

/**
 * This process's calls that hold a memory's lock, keyed by the memory's real root and chained, so
 * its sessions and opens take turns rather than poll the lock against each other.
 */
const TURNS = new Map<string, Promise<void>>();

/**
 * Run `fn` once this process's earlier turns on the memory at `root` have finished.
 *
 * An abort of `signal` while the call is queued rejects it at once with the signal's reason; its
 * turn, when it comes, does nothing, so the calls after it keep their order. Once its turn has
 * started, `fn` answers the abort itself, so a change it applies is never reported as aborted.
 */
function inTurn<T>(root: string, signal: AbortSignal | undefined, fn: () => Promise<T>): Turn<T> {
  let started = false;
  const run = (TURNS.get(root) ?? Promise.resolve()).then(() => {
    started = true;
    signal?.throwIfAborted();
    return fn();
  });
  const done = run.then(
    () => undefined,
    () => undefined,
  );
  TURNS.set(root, done);
  void done.then(() => {
    if (TURNS.get(root) === done) TURNS.delete(root);
  });
  if (signal === undefined) return { result: run, done };
  const result = new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      if (!started) reject(signal.reason);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    // A signal aborted already never fires its listeners.
    if (signal.aborted) onAbort();
    void run.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
  });
  return { result, done };
}

/**
 * A memory kept as a folder of UTF-8 text files.
 *
 * @example
 * const lessons = new FileMemory({ root: "/srv/memory/lessons", id: "team/lessons" });
 */
export class FileMemory extends MemoryBase {
  static override readonly contract: MemoryContract = Object.freeze({ type: "FileMemory", tools: TOOLS });

  /** The memory's folder, symlinks resolved. */
  readonly root: string;
  readonly #id: string | undefined;
  readonly #store: Store;

  /**
   * Bind a FileMemory to the folder `root`, creating nothing.
   *
   * @throws {TypeError} If `root` or `id` isn't a non-empty string, or `index` can't name a member.
   * @throws {RangeError} If a cap isn't an integer in its range.
   */
  constructor(options: FileMemoryOptions) {
    super();
    const { root, id, index = "MEMORY.md" } = options;
    if (typeof root !== "string" || root === "" || root.includes("\0")) {
      throw new TypeError("FileMemory root must be a folder path");
    }
    if (id !== undefined && (typeof id !== "string" || id === "" || !id.isWellFormed() || /[\r\n\0]/.test(id))) {
      throw new TypeError("FileMemory id must be a non-empty string on one line");
    }
    const defaults = current();
    const maxTotalChars = capOption("maxTotalChars", options.maxTotalChars, defaults.fileMemoryMaxTotalChars, 1);
    const limits: Limits = Object.freeze({
      maxFileChars: capOption("maxFileChars", options.maxFileChars, defaults.fileMemoryMaxFileChars, 1),
      maxTotalChars,
      maxFiles: capOption("maxFiles", options.maxFiles, defaults.fileMemoryMaxFiles, 1),
      maxDepth: capOption("maxDepth", options.maxDepth, defaults.fileMemoryMaxDepth, 0),
      maxRecallChars: capOption("maxRecallChars", options.maxRecallChars, defaults.fileMemoryMaxRecallChars, 1),
      maxReadChars: capOption("maxReadChars", options.maxReadChars, defaults.fileMemoryMaxReadChars, 1),
      maxResults: capOption("maxResults", options.maxResults, defaults.fileMemoryMaxResults, 1),
      maxUndo: capOption("maxUndo", options.maxUndo, defaults.memoryMaxUndo, 0),
      maxChangeIds: capOption("maxChangeIds", options.maxChangeIds, defaults.memoryMaxChangeIds, 1),
      maxHistoryChars: capOption("maxHistoryChars", options.maxHistoryChars, maxTotalChars, 0),
    });
    const why = typeof index === "string" ? pathProblem(index, limits.maxDepth) : "not a string";
    if (why !== null) throw new TypeError(`FileMemory index ${shown(String(index))} isn't a member path: ${why}`);
    const defaultTimeoutMs = defaults.memoryLockTimeoutSeconds * 1000;
    const timeoutMs = options.lockTimeoutMs ?? defaultTimeoutMs;
    if (typeof timeoutMs !== "number" || Number.isNaN(timeoutMs) || timeoutMs < 0) {
      throw new RangeError(`FileMemory lockTimeoutMs must be 0 or more milliseconds, not ${String(timeoutMs)}`);
    }
    // Resolved now, so a later chdir or a swapped symlink doesn't move the memory.
    const real = realRoot(root);
    const scheduler = options.scheduler ?? realScheduler;
    this.root = real;
    this.#id = id;
    this.#store = Object.freeze({
      root: real,
      index,
      limits,
      scheduler,
      locked<T>(signal: AbortSignal | undefined, body: Held<T>): Turn<T> {
        // A monotonic clock, so a wall-clock step while queued neither stretches nor cuts the wait.
        const deadline = performance.now() + timeoutMs;
        return inTurn(real, signal, async () => {
          // Checked before the lock file is opened, and on every call: the folder may change.
          checkMeta(real);
          // Only what is left of the timeout: a call waits it out once, queued or not.
          const left = Math.max(0, deadline - performance.now());
          let lock: FileLock;
          try {
            lock = await waitForLock(join(real, META, "lock"), { timeoutMs: left, signal, scheduler, private: true });
          } catch (error) {
            if (error instanceof FileLockError && error.reason === "busy") {
              // No cause: the lock's error names its path, and this one can reach a model.
              throw new MemoryBusyError("this memory is busy: another writer holds its lock");
            }
            throw error;
          }
          try {
            // Again holding the lock: the folder may have changed while the call waited for it.
            checkMeta(real);
            return body();
          } finally {
            lock.release();
          }
        });
      },
    });
  }

  /**
   * Open a session on the memory. With `create`, a missing folder is made (0700), and a folder
   * that isn't a memory yet is adopted with its files; either stores the identity in
   * `.yian/identity` and reports `created`.
   *
   * Before it returns, it catches up on the journal and resolves a change a crash cut off.
   *
   * @throws {TypeError} If `itemType` isn't `null`.
   * @throws {MemoryMissingError} If the memory doesn't exist and `create` is false.
   * @throws Error when its stored identity isn't the constructor's `id`, the root or `.yian` isn't
   *   a folder, or the journal is damaged.
   * @throws {MemoryBusyError} If it waits for the lock longer than the timeout.
   */
  override async open(o: {
    create: boolean;
    itemType: JsonObject | null;
    signal: AbortSignal;
  }): Promise<MemorySession> {
    o.signal.throwIfAborted();
    if (o.itemType !== null) {
      throw new TypeError("a FileMemory holds files, not typed items; give its input no item type");
    }
    const { root } = this;
    const stats = lstatOrNull(root);
    if (stats === null) {
      if (!o.create) throw new MemoryMissingError("the memory isn't at its location: there is no folder");
      makeRoot(root);
    } else if (!stats.isDirectory()) {
      throw new Error(`${shown(root)} isn't a folder`);
    }
    checkMeta(root);
    let identity = storedIdentity(root);
    let created = false;
    if (identity === null) {
      if (!o.create) throw new MemoryMissingError(NOT_A_MEMORY);
      makeDirectory(join(root, META), META);
      makeDirectory(join(root, META, "tmp"), `${META}/tmp`);
      // Two opens may adopt one folder at once: the lock lets only one of them store an identity.
      identity = await this.#store.locked(o.signal, () => {
        const stored = storedIdentity(root);
        if (stored !== null) return stored;
        const made = this.#id ?? `at-${sha256(root).slice(0, 16)}`;
        replaceFile(join(root, META, "identity"), Buffer.from(made, "utf8"), join(root, META, "tmp"));
        created = true;
        return made;
      }).result;
    }
    if (this.#id !== undefined && identity !== this.#id) {
      throw new Error(`this memory is ${shown(identity)}, not ${shown(this.#id)}`);
    }
    const session = new FileMemorySession(this.#store, sessionInfo(identity, this.#store), created);
    // A change a crash cut off is resolved before the session reads or writes anything.
    await session.recover(o.signal);
    return session;
  }

  /**
   * Empty the memory, keeping its identity: start a new journal generation, so its history and
   * change ids are gone, then delete its members and the folders that held them once they are
   * empty. Files that aren't part of the memory stay. Only a host or the CLI calls it, outside
   * runs; a session open on the memory reads the new generation on its next write.
   *
   * @throws {MemoryMissingError} If the folder isn't a memory.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   */
  override async reset(o: { signal: AbortSignal }): Promise<void> {
    o.signal.throwIfAborted();
    const { root, limits } = this.#store;
    if (lstatOrNull(root)?.isDirectory() === true) checkMeta(root);
    if (lstatOrNull(root)?.isDirectory() !== true || storedIdentity(root) === null) {
      throw new MemoryMissingError(NOT_A_MEMORY);
    }
    await this.#store.locked(o.signal, (): void => {
      // The journal goes first: a crash part way leaves members to delete, never change ids that
      // would answer for changes to them.
      new Journal(join(root, META), limits).restart();
      const folders = new Set<string>();
      for (const file of readMembers(root, limits).files) {
        fs.unlinkSync(join(root, file.path));
        for (let folder = dirname(file.path); folder !== "."; folder = dirname(folder)) folders.add(folder);
      }
      // Deepest first, so a folder's emptied subfolders are gone before it is tried.
      const deepestFirst = [...folders].sort((a, b) => b.split("/").length - a.split("/").length);
      for (const folder of deepestFirst) {
        try {
          fs.rmdirSync(join(root, folder));
        } catch (error) {
          if (!isSystemError(error) || (error.code !== "ENOTEMPTY" && error.code !== "EEXIST")) throw error;
        }
      }
      for (const folder of [".", ...folders]) {
        if (lstatOrNull(join(root, folder)) !== null) syncFolder(join(root, folder));
      }
      clearTmp(root);
    }).result;
  }
}

/** `fs.lstatSync`, or `null` when nothing is at `path`. */
function lstatOrNull(path: string): fs.Stats | null {
  try {
    return fs.lstatSync(path);
  } catch (error) {
    if (isSystemError(error) && error.code === "ENOENT") return null;
    throw error;
  }
}

/** Make the missing folder `root`, private, syncing each new folder's parent. */
function makeRoot(root: string): void {
  const first = fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  fs.chmodSync(root, 0o700);
  if (first !== undefined) {
    for (let made = root; made !== dirname(first); made = dirname(made)) syncFolder(dirname(made));
  }
}

/**
 * Return the identity in `<root>/.yian/identity`, or `null` when there is none.
 *
 * @throws Error when `.yian` isn't a folder, or the identity isn't one line of UTF-8 text.
 */
function storedIdentity(root: string): string | null {
  const meta = lstatOrNull(join(root, META));
  if (meta === null) return null;
  if (!meta.isDirectory()) throw new Error(`${shown(META)} isn't a folder`);
  let fd: number;
  try {
    fd = openRegular(join(root, META, "identity"), fs.constants.O_RDONLY);
  } catch (error) {
    if (isSystemError(error) && error.code === "ENOENT") return null;
    throw error;
  }
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(fs.readFileSync(fd));
  } catch {
    throw new Error(`${shown(`${META}/identity`)} isn't UTF-8 text`);
  } finally {
    fs.closeSync(fd);
  }
  // A synced or hand-written identity may end in a line break.
  const identity = text.replace(/\r?\n$/, "");
  if (identity === "" || /[\r\n]/.test(identity)) throw new Error(`${shown(`${META}/identity`)} isn't one line`);
  return identity;
}

/** Return a session's info for `identity`: its settings hold no location. */
function sessionInfo(identity: string, store: Store): MemoryInfo {
  const { limits } = store;
  return Object.freeze({
    type: "FileMemory",
    identity,
    settings: Object.freeze({
      index: store.index,
      max_file_chars: BigInt(limits.maxFileChars),
      max_total_chars: BigInt(limits.maxTotalChars),
      max_files: BigInt(limits.maxFiles),
      max_depth: BigInt(limits.maxDepth),
      max_recall_chars: BigInt(limits.maxRecallChars),
      max_read_chars: BigInt(limits.maxReadChars),
      max_results: BigInt(limits.maxResults),
      max_undo: BigInt(limits.maxUndo),
      max_change_ids: BigInt(limits.maxChangeIds),
      max_history_chars: BigInt(limits.maxHistoryChars),
    }),
    tools: TOOLS,
  });
}

/** Write `bytes` to the member `path`, making its missing folders (0700) first. */
function writeMember(root: string, path: string, bytes: Uint8Array): void {
  const parts = path.split("/");
  for (let i = 1; i < parts.length; i++) {
    const folder = parts.slice(0, i).join("/");
    makeDirectory(join(root, folder), folder);
  }
  const tmp = join(root, META, "tmp");
  makeDirectory(tmp, `${META}/tmp`);
  replaceFile(join(root, path), bytes, tmp);
}

/** Delete the member `path` and sync its folder. */
function removeMember(root: string, path: string): void {
  fs.unlinkSync(join(root, path));
  syncFolder(dirname(join(root, path)));
}

/**
 * Return the sha256 of the bytes at `rel` under `root`, `absent` when no file is there, or `null`
 * when the path can't hold a member, so that no hash a change recorded matches it.
 */
function fileHash(root: string, rel: string, limits: Limits): string | null {
  if (pathProblem(rel, limits.maxDepth) !== null || diskProblem(root, rel) !== null) return null;
  let fd: number;
  try {
    fd = openRegular(join(root, rel), fs.constants.O_RDONLY);
  } catch (error) {
    if (isSystemError(error) && error.code === "ENOENT") return ABSENT;
    if (isSystemError(error) && ["ELOOP", "EISDIR", "EINVAL", "ENXIO"].includes(error.code)) return null;
    throw error;
  }
  try {
    // A file this big was never a member, so no recorded hash can be its.
    if (fs.fstatSync(fd).size > 4 * limits.maxFileChars) return null;
    return sha256(fs.readFileSync(fd));
  } finally {
    fs.closeSync(fd);
  }
}

/** Return the result a committed change `entry` reports when the memory's version is `version`. */
function resultOf(entry: JournalEntry, version: string): JsonObject {
  const { path, after } = entry;
  if (entry.op === "erase") return { path, version, disabled_undos: [...(entry.disables ?? [])] };
  return after === ABSENT ? { path, version } : { path, hash: after, version };
}

/**
 * Return `target` as an erase's target: a member path, and the text to remove from it.
 *
 * @throws {TypeError} If it isn't `{path, text?}`, with a string path and a non-empty text.
 */
function checkEraseTarget(target: JsonObject): { path: string; text?: string } {
  if (!isObject(target) || typeof target.path !== "string") throw new TypeError("an erase needs path as a string");
  const extra = Object.keys(target).filter((key) => key !== "path" && key !== "text");
  if (extra.length > 0) throw new TypeError(`an erase takes path and text, not ${listed(extra)}`);
  const { path, text } = target;
  if (text === undefined) return { path };
  if (typeof text !== "string" || text === "" || !text.isWellFormed()) {
    throw new TypeError("an erase's text must be a non-empty string");
  }
  return { path, text };
}

/**
 * Remove the temp files a crash left in `.yian/tmp/`: holding the lock, no write is using them.
 * Only regular files directly in it go, each unlinked; anything else is left alone and nothing is
 * followed, and `checkMeta` has made sure the folder itself is no symlink.
 */
function clearTmp(root: string): void {
  const tmp = join(root, META, "tmp");
  // Never read through a symlink, should one appear after the check.
  if (lstatOrNull(tmp)?.isDirectory() !== true) return;
  let names: string[];
  try {
    names = fs.readdirSync(tmp);
  } catch (error) {
    if (isSystemError(error) && error.code === "ENOENT") return;
    throw error;
  }
  for (const name of names) {
    if (lstatOrNull(join(tmp, name))?.isFile() === true) fs.unlinkSync(join(tmp, name));
  }
}

/**
 * The memory's own entries in `.yian/`, what each must be, and what to do when it isn't: a
 * symlinked folder would let temp cleanup or a write reach outside the memory, and a linked
 * journal would keep what an erase removed.
 */
const META_ENTRIES: readonly { name: string; folder: boolean; fix: string }[] = [
  { name: META, folder: true, fix: "remove it" },
  { name: `${META}/tmp`, folder: true, fix: "remove it" },
  { name: `${META}/identity`, folder: false, fix: "replace it with a copy" },
  { name: `${META}/journal.jsonl`, folder: false, fix: "replace it with a copy" },
  { name: `${META}/lock`, folder: false, fix: "remove it" },
];

/**
 * Check, with `lstat`, that each of the memory's own entries in `.yian/` that exists is a real
 * folder or a regular file with one link.
 *
 * @throws Error naming the entry, what it is instead, and what to do.
 */
function checkMeta(root: string): void {
  for (const { name, folder, fix } of META_ENTRIES) {
    const stats = lstatOrNull(join(root, name));
    if (stats === null) continue;
    let why: string | null = null;
    if (stats.isSymbolicLink()) why = "a symlink";
    else if (folder) why = stats.isDirectory() ? null : "not a folder";
    else if (!stats.isFile()) why = "not a regular file";
    else if (stats.nlink > 1) why = "a hard link";
    if (why !== null) {
      throw new Error(`\`${name}\` isn't a ${folder ? "folder" : "file"} of this memory (${why}); ${fix}`);
    }
  }
}

/**
 * Return `error` with the memory's location taken out of its message: an OS error names absolute
 * paths, and a session's errors can reach a model and a run log, which never hold a location.
 * Paths in the memory stay, relative to its root. A system error keeps its code.
 */
function withoutLocation(error: unknown, root: string): unknown {
  if (!(error instanceof Error) || !error.message.includes(root)) return error;
  const copy: Error & { code?: unknown; syscall?: unknown; errno?: unknown } = new Error(
    error.message.replaceAll(`${root}/`, "").replaceAll(root, "the memory"),
  );
  copy.name = error.name;
  // Not `path`, `dest` or `cause`, which hold the location too.
  const { code, syscall, errno } = error as NodeJS.ErrnoException;
  if (code !== undefined) copy.code = code;
  if (syscall !== undefined) copy.syscall = syscall;
  if (errno !== undefined) copy.errno = errno;
  return copy;
}

/** One open FileMemory. Read tools take no lock; writes and snapshots take the memory's lock. */
class FileMemorySession implements MemorySession {
  readonly info: MemoryInfo;
  readonly created: boolean;
  readonly #store: Store;
  readonly #journal: Journal;
  /**
   * When the session's last queued turn has finished. Turns run in the order they were queued, so
   * every earlier turn has finished by then too: `close` waits for it.
   */
  #last: Promise<void> = Promise.resolve();
  #closed = false;

  /**
   * @param store The memory it reads and writes.
   * @param info What the memory is, as its recorded value holds it.
   * @param created Whether the open that made this session created the memory.
   */
  constructor(store: Store, info: MemoryInfo, created: boolean) {
    this.#store = store;
    this.#journal = new Journal(join(store.root, META), store.limits);
    this.info = info;
    this.created = created;
  }

  /**
   * Read the version and the recall in one read under the lock.
   *
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   */
  snapshot(): Promise<{ version: string; recall: Recall }> {
    return this.#locked(undefined, () => {
      const { root, index, limits } = this.#store;
      const members = readMembers(root, limits);
      return { version: memoryVersion(members.files), recall: recallOf(members, index, limits.maxRecallChars) };
    });
  }

  /**
   * Call the read tool `tool`: `list`, `read` or `search`.
   *
   * @throws {ReportedToolError} (permanent) For a path that isn't part of the memory or doesn't
   *   exist, with the reason.
   * @throws {TypeError} For a tool that isn't a read tool, or an argument of the wrong type.
   */
  async call(tool: string, args: JsonObject, o: { signal: AbortSignal }): Promise<JsonObject> {
    this.#checkOpen();
    o.signal.throwIfAborted();
    const spec = TOOLS.find((each) => each.name === tool);
    if (spec === undefined) {
      throw new TypeError(`FileMemory has no tool ${tool}; it has ${TOOLS.map((each) => each.name).join(", ")}`);
    }
    if (spec.write) throw new TypeError(`${tool} is a write tool: apply it as a change`);
    try {
      if (tool === "list") return this.#list(stringArg(args, "dir", false));
      if (tool === "read") {
        return this.#read(stringArg(args, "path", true), countArg(args, "offset"), countArg(args, "limit"));
      }
      return this.#search(stringArg(args, "query", true));
    } catch (error) {
      throw withoutLocation(error, this.#store.root);
    }
  }

  /**
   * Apply one change under the lock, unless its id is known: then return the stored outcome.
   *
   * The change's path, precondition and caps are checked first, and a refusal (`ok: false`)
   * writes nothing. Otherwise the journal entry is written and synced, then the file through a
   * synced temp file, then the commit line.
   *
   * @throws {ChangeIdReusedError} If `id` was applied with a different change.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   * @throws {TypeError} If `id`, `change` or `author` has the wrong shape.
   */
  async apply(c: { id: string; change: JsonObject; author: Author; signal: AbortSignal }): Promise<Applied> {
    this.#checkOpen();
    c.signal.throwIfAborted();
    if (typeof c.id !== "string" || c.id === "") throw new TypeError("a change id must be a non-empty string");
    const author = checkAuthor(c.author);
    const change = checkChange(c.change);
    const digest = sha256(canonicalJson(c.change));
    return this.#locked(c.signal, () => {
      this.#recover();
      const known = this.#journal.known(c.id);
      if (known !== undefined) {
        if (known.digest !== digest) throw new ChangeIdReusedError(c.id);
        return known.applied;
      }
      if (this.#journal.shouldCompact()) this.#journal.compact();
      return this.#write(c.id, digest, change, author);
    });
  }

  /**
   * Undo change `changeId` as a new change `id`: write back what it replaced, if its file still
   * holds what it wrote and the memory still keeps its inverse. Otherwise it is refused or
   * conflicts, writing nothing. An undo is in history and can itself be undone.
   *
   * @throws {ChangeIdReusedError} If `id` was applied with a different change.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   * @throws {TypeError} If `id`, `changeId` or `author` has the wrong shape.
   */
  async undo(o: { id: string; changeId: string; author: Author; signal: AbortSignal }): Promise<Applied> {
    this.#checkOpen();
    o.signal.throwIfAborted();
    if (typeof o.id !== "string" || o.id === "") throw new TypeError("a change id must be a non-empty string");
    if (typeof o.changeId !== "string") throw new TypeError("undo needs the change id to undo as a string");
    const author = checkAuthor(o.author);
    const digest = sha256(canonicalJson({ op: "undo", change: o.changeId }));
    return this.#locked(o.signal, () => {
      this.#recover();
      const known = this.#journal.known(o.id);
      if (known !== undefined) {
        if (known.digest !== digest) throw new ChangeIdReusedError(o.id);
        return known.applied;
      }
      if (this.#journal.shouldCompact()) this.#journal.compact();
      return this.#undo(o.id, digest, o.changeId, author);
    });
  }

  /**
   * List the last `limit` changes the journal holds, newest first. Aborted changes are left out;
   * a change whose outcome is unknown is listed, since it may have landed.
   *
   * @throws {RangeError} If `limit` isn't an integer of at least 0.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   */
  async history(o: { limit: number }): Promise<ChangeEntry[]> {
    this.#checkOpen();
    if (!Number.isSafeInteger(o.limit) || o.limit < 0) {
      throw new RangeError(`history needs a limit of 0 or more changes, not ${String(o.limit)}`);
    }
    return this.#locked(undefined, () => {
      this.#recover();
      const changes = this.#journal.changes();
      return changes
        .slice(Math.max(0, changes.length - o.limit))
        .reverse()
        .map(({ entry }) => {
          const target: JsonObject = { path: entry.path };
          if (entry.undoes !== undefined) target.undoes = entry.undoes;
          return { id: entry.id, op: entry.op, target, author: authorOf(entry.author), at: entry.at };
        });
    });
  }

  /**
   * Erase as change `id`: remove every occurrence of `target.text` from the member `target.path`,
   * until none is left, or the whole file when there is no text, and every copy of it in the
   * journal (each inverse for that path, and each holding the text), after resolving a change a
   * crash cut off. It can't be undone, and it reports the change ids whose undo it disabled: those
   * whose file still held what they wrote.
   *
   * The journal is rewritten first, ending in the erase's entry, which holds no text; then the
   * file is written and the commit line appended. A crash after the rewrite and before the file is
   * written aborts the erase: the journal no longer holds the text, but the file still does until
   * the erase is sent again, and the resend reports only the undos it disables then.
   *
   * The journal keeps the hashes of files and the digests of changes. They hold no text, but a
   * short secret, such as a PIN, can be found from them by trying every value.
   *
   * @throws Error naming the reason when the path isn't part of the memory. A path refused by its
   *   text writes nothing; for a file a hand edit took out of the memory, the history is scrubbed
   *   and the file is left for the person to remove.
   * @throws Error with the stored reason when this id's earlier erase has an unknown outcome.
   * @throws {ChangeIdReusedError} If `id` was applied with a different change.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   * @throws {TypeError} If `id`, `target` or `author` has the wrong shape.
   */
  async erase(o: { id: string; target: JsonObject; author: Author; signal: AbortSignal }): Promise<Erased> {
    this.#checkOpen();
    o.signal.throwIfAborted();
    if (typeof o.id !== "string" || o.id === "") throw new TypeError("a change id must be a non-empty string");
    const author = checkAuthor(o.author);
    const target = checkEraseTarget(o.target);
    const digest = sha256(canonicalJson({ op: "erase", ...target }));
    const applied = await this.#locked(o.signal, () => {
      this.#recover();
      const known = this.#journal.known(o.id);
      if (known !== undefined) {
        if (known.digest !== digest) throw new ChangeIdReusedError(o.id);
        return known.applied;
      }
      return this.#erase(o.id, digest, target, author);
    });
    if (!applied.ok) throw new Error(applied.reason);
    return { version: applied.version, disabledUndos: applied.result.disabled_undos as string[] };
  }

  /** Close the session once every turn it queued has finished, aborted calls' included; later calls throw. */
  async close(): Promise<void> {
    this.#closed = true;
    await this.#last;
  }

  /**
   * List the members, or those under `dir`, within `maxResults` and `maxReadChars`: the paths
   * shown count, and the first is shown whatever its size.
   */
  #list(dir: string | undefined): JsonObject {
    const { root, limits } = this.#store;
    let prefix = "";
    if (dir !== undefined) {
      const why = pathProblem(dir, limits.maxDepth, true) ?? diskProblem(root, dir, true);
      if (why !== null) throw refusal(`${shown(dir)} isn't a folder of the memory: ${why}`);
      if (lstatOrNull(join(root, dir)) === null) throw refusal(`${shown(dir)} doesn't exist`);
      prefix = `${dir}/`;
    }
    const files = readMembers(root, limits).files.filter((file) => file.path.startsWith(prefix));
    const listed: JsonObject[] = [];
    let used = 0;
    for (const file of files) {
      // A path is never cut, so the first one shows whatever its size.
      const cost = charCount(file.path);
      if (listed.length === limits.maxResults) break;
      if (listed.length > 0 && used + cost > limits.maxReadChars) break;
      listed.push({ path: file.path, hash: file.hash, chars: BigInt(file.chars) });
      used += cost;
    }
    const result: JsonObject = { files: listed };
    if (listed.length < files.length) result.cut = `${files.length - listed.length} more; list a folder with dir`;
    return result;
  }

  /** Read lines of the member `path` from line `offset` (1-based), at most `limit`, within `maxReadChars`. */
  #read(path: string, offset = 1, limit?: number): JsonObject {
    const { limits } = this.#store;
    const file = this.#member(path);
    const lines = linesOf(file.text);
    const start = offset - 1;
    const result: JsonObject = { path, hash: file.hash, text: "" };
    if (start > 0 && start >= lines.length) {
      result.cut = `the file has ${plural(lines.length, "line")}`;
      return result;
    }
    const end = limit === undefined ? lines.length : Math.min(lines.length, start + limit);
    const shownLines: string[] = [];
    let used = 0;
    for (let i = start; i < end; i++) {
      const cost = charCount(lines[i]!) + (shownLines.length > 0 ? 1 : 0);
      if (used + cost > limits.maxReadChars) {
        if (shownLines.length === 0) {
          // A first line longer than the cap is shown cut, so a read is never empty.
          result.text = firstChars(lines[i]!, limits.maxReadChars);
          const more = i + 1 < lines.length ? `; read on with offset ${i + 2}` : "";
          result.cut = `line ${i + 1} of ${lines.length} is cut after ${limits.maxReadChars} characters${more}`;
          return result;
        }
        break;
      }
      shownLines.push(lines[i]!);
      used += cost;
    }
    result.text = shownLines.join("\n");
    const last = start + shownLines.length;
    if (last < lines.length) {
      result.cut = `showing lines ${offset}-${last} of ${lines.length}; read on with offset ${last + 1}`;
    }
    return result;
  }

  /**
   * Find the members' lines holding `query`, ignoring case, within `maxResults` and
   * `maxReadChars`: each match's path and line count, and a first match too long is shown cut.
   */
  #search(query: string): JsonObject {
    const { root, limits } = this.#store;
    if (query === "") throw refusal("query is empty");
    const needle = query.toLowerCase();
    const matches: JsonObject[] = [];
    const notes: string[] = [];
    let used = 0;
    let unshown = 0;
    let full = false;
    for (const file of readMembers(root, limits).files) {
      const lines = linesOf(file.text);
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i]!;
        if (!line.toLowerCase().includes(needle)) continue;
        const match = { path: file.path, hash: file.hash, line: BigInt(i + 1) };
        const cost = charCount(file.path) + charCount(line);
        if (full || matches.length === limits.maxResults || used + cost > limits.maxReadChars) {
          full = true;
          if (matches.length > 0) {
            unshown++;
            continue;
          }
          // A first match longer than the cap is shown with its line cut, so a search that matched
          // isn't empty.
          const room = Math.max(0, limits.maxReadChars - charCount(file.path));
          matches.push({ ...match, text: firstChars(line, room) });
          notes.push(`line ${i + 1} of ${file.path} is cut after ${plural(room, "character")}`);
          continue;
        }
        matches.push({ ...match, text: line });
        used += cost;
      }
    }
    if (unshown > 0) notes.push(`${plural(unshown, "more matching line")} not shown`);
    const result: JsonObject = { matches };
    if (notes.length > 0) result.cut = notes.join("; ");
    return result;
  }

  /**
   * Return the member at `path`.
   *
   * @throws {ReportedToolError} (permanent) If `path` isn't part of the memory or doesn't exist.
   */
  #member(path: string): MemberFile {
    const { root, limits } = this.#store;
    const why = pathProblem(path, limits.maxDepth) ?? diskProblem(root, path);
    const file = why ?? readMemberFile(join(root, path), path, limits.maxFileChars);
    if (typeof file === "string") throw refusal(`${shown(path)} isn't part of the memory: ${file}`);
    if (file === null) throw refusal(`${shown(path)} doesn't exist`);
    return file;
  }

  /** Rewrite the journal without `target`'s text, ending in the erase's entry; write the file; commit. */
  #erase(id: string, digest: string, target: { path: string; text?: string }, author: Author): Applied {
    const { root, limits, scheduler } = this.#store;
    const { path, text } = target;
    const textual = pathProblem(path, limits.maxDepth);
    if (textual !== null) throw new Error(`${shown(path)} isn't part of the memory: ${textual}`);
    // An inverse holding the text goes whole, so no part of one is left to join into the text again.
    const drop = ({ entry }: JournalChange): boolean => {
      const { inverse } = entry;
      if (entry.path === path) return true;
      return text !== undefined && inverse !== null && "text" in inverse && inverse.text.includes(text);
    };
    const why = diskProblem(root, path);
    const old = why ?? readMemberFile(join(root, path), path, limits.maxFileChars);
    if (typeof old === "string") {
      // A hand edit made the file no member: it is the person's to remove, but the memory's history
      // is the memory's, so that is scrubbed anyway.
      this.#journal.compact(drop);
      throw new Error(
        `${shown(path)} isn't part of the memory: ${old}, so erase left the file alone; the memory's history ` +
          `no longer holds ${text === undefined ? "its old text" : "the text"}, but the file may: remove it by hand`,
      );
    }
    const next = text === undefined || old === null ? null : withoutText(old.text, text);
    // The undos it disables: those it drops that would still work, their file holding what they wrote.
    const undoable = this.#journal.undoable();
    const hashes = new Map<string, string | null>([[path, old?.hash ?? ABSENT]]);
    const hashOf = (rel: string) => {
      if (!hashes.has(rel)) hashes.set(rel, fileHash(root, rel, limits));
      return hashes.get(rel);
    };
    const disables = this.#journal
      .changes()
      .filter((change) => undoable.has(change) && drop(change) && hashOf(change.entry.path) === change.entry.after)
      .map((change) => change.entry.id);
    const bytes = next === null ? null : Buffer.from(next, "utf8");
    const entry: JournalEntry = {
      id,
      digest,
      op: "erase",
      path,
      author,
      at: new Date(scheduler.now()).toISOString(),
      before: old?.hash ?? ABSENT,
      after: bytes === null ? ABSENT : sha256(bytes),
      inverse: null,
      disables,
    };
    this.#journal.erase(entry, drop);
    if (bytes !== null && next !== old?.text) writeMember(root, path, bytes);
    else if (bytes === null && old !== null) removeMember(root, path);
    const version = memoryVersion(readMembers(root, limits).files);
    const applied: Applied = { ok: true, version, result: resultOf(entry, version) };
    this.#journal.writeOutcome(id, applied);
    return applied;
  }

  /** Check that change `changeId` can be undone and its file holds what it wrote, then commit the undo. */
  #undo(id: string, digest: string, changeId: string, author: Author): Applied {
    const { root, limits } = this.#store;
    const shownId = shown(changeId);
    const target = this.#journal.find(changeId);
    if (target === undefined) return refused(`change ${shownId} isn't in the memory's history`);
    if (!target.applied.ok) {
      return refused(`change ${shownId} may not have landed, so it can't be undone; check the file`);
    }
    const { inverse, path, after } = target.entry;
    if (inverse === null || !this.#journal.undoable().has(target)) {
      return refused(`change ${shownId} can't be undone: the memory no longer keeps what it replaced`);
    }
    // The path comes from the journal, which a person may have edited: it is checked like any other.
    const why = pathProblem(path, limits.maxDepth) ?? diskProblem(root, path);
    const old = why ?? readMemberFile(join(root, path), path, limits.maxFileChars);
    if (typeof old === "string") return refused(`${shown(path)} isn't part of the memory: ${old}`);
    if ((old?.hash ?? ABSENT) !== after) {
      const reason = `${shown(path)} changed after change ${shownId}; undoing it would lose that`;
      return old === null ? refused(reason) : this.#conflict(reason, old);
    }
    let text: string | null;
    if ("text" in inverse) text = inverse.text;
    else if ("absent" in inverse) text = null;
    // An append always leaves a file: only a hand-edited entry has none to cut back.
    else if (old === null) return refused(`change ${shownId} can't be undone: its journal entry is damaged`);
    // The file is what the append wrote, so its first bytes are exactly the ones it had before.
    else text = Buffer.from(old.text, "utf8").subarray(0, inverse.truncate).toString("utf8");
    const back: Inverse = old === null ? { absent: true } : { text: old.text };
    return this.#commit({ id, digest, op: "undo", path, author, undoes: changeId }, old, { text, inverse: back });
  }

  /** Check `change` against the folder and its precondition, then commit it. */
  #write(id: string, digest: string, change: Change, author: Author): Applied {
    const { root, limits } = this.#store;
    const { path } = change;
    const why = pathProblem(path, limits.maxDepth) ?? diskProblem(root, path);
    const old = why ?? readMemberFile(join(root, path), path, limits.maxFileChars);
    if (typeof old === "string") return refused(`${shown(path)} isn't part of the memory: ${old}`);
    const next = this.#next(change, old);
    if ("ok" in next) return next;
    return this.#commit({ id, digest, op: change.op, path, author }, old, next);
  }

  /**
   * Check the caps for writing `next.text` (or deleting, for `null`) over the member `old`, then
   * journal the change, write its content and commit it.
   */
  #commit(
    change: { id: string; digest: string; op: string; path: string; author: Author; undoes?: string },
    old: MemberFile | null,
    next: { text: string | null; inverse: Inverse },
  ): Applied {
    const { root, limits, scheduler } = this.#store;
    const { id, path } = change;
    const others = readMembers(root, limits).files.filter((file) => file.path !== path);
    let files = others;
    let bytes: Buffer | null = null;
    if (next.text !== null) {
      const chars = charCount(next.text);
      if (chars > limits.maxFileChars) {
        return refused(`${shown(path)} would hold ${chars} characters; a file may hold ${limits.maxFileChars}`);
      }
      const total = totalChars(others) + chars;
      // A memory over its cap by hand may still shrink: only a change that grows it is refused.
      if (total > limits.maxTotalChars && total > totalChars(others) + (old?.chars ?? 0)) {
        return refused(`the memory would hold ${total} characters; it may hold ${limits.maxTotalChars}`);
      }
      if (old === null && others.length + 1 > limits.maxFiles) {
        return refused(`the memory would hold ${others.length + 1} files; it may hold ${limits.maxFiles}`);
      }
      bytes = Buffer.from(next.text, "utf8");
      files = [...others, { path, text: next.text, hash: sha256(bytes), chars }];
    }
    const entry: JournalEntry = {
      ...change,
      at: new Date(scheduler.now()).toISOString(),
      before: old?.hash ?? ABSENT,
      after: bytes === null ? ABSENT : sha256(bytes),
      inverse: storedInverse(next.inverse, limits),
    };
    this.#journal.writeEntry(entry);
    if (bytes !== null) writeMember(root, path, bytes);
    else if (old !== null) removeMember(root, path);
    const version = memoryVersion(files);
    const applied: Applied = { ok: true, version, result: resultOf(entry, version) };
    this.#journal.writeOutcome(id, applied);
    return applied;
  }

  /**
   * Return what `change` makes of the member `old` (`null` when there is no file): the new text,
   * or `null` to delete it, and the inverse that undoes it; or the conflict when its precondition
   * fails.
   */
  #next(change: Change, old: MemberFile | null): { text: string | null; inverse: Inverse } | Applied {
    const path = shown(change.path);
    switch (change.op) {
      case "append": {
        if (old === null) return { text: change.text, inverse: { absent: true } };
        const lineBreak = old.text === "" || old.text.endsWith("\n") ? "" : "\n";
        // The file's bytes are its old bytes and then the appended ones, so undo cuts it back.
        const inverse = { truncate: Buffer.byteLength(old.text, "utf8") };
        return { text: old.text + lineBreak + change.text, inverse };
      }
      case "create":
        if (old !== null) return this.#conflict(`${path} already exists`, old);
        return { text: change.text, inverse: { absent: true } };
      case "edit": {
        if (old === null) return refused(`${path} doesn't exist`);
        if (change.old === "") return this.#conflict("the old text is empty; give the exact text to replace", old);
        const at = old.text.indexOf(change.old);
        if (at === -1) return this.#conflict(`the old text isn't in ${path}; read the file and copy it exactly`, old);
        // Searched again from the next character, so an old text that overlaps itself counts twice.
        if (old.text.indexOf(change.old, at + 1) !== -1) {
          return this.#conflict(`the old text appears more than once in ${path}; give more of it`, old);
        }
        const text = old.text.slice(0, at) + change.new + old.text.slice(at + change.old.length);
        return { text, inverse: { text: old.text } };
      }
      case "delete":
      case "rewrite":
        if (old === null) return refused(`${path} doesn't exist`);
        if (old.hash !== change.if_hash) {
          return this.#conflict(`${path} changed since it was read: its hash isn't if_hash`, old);
        }
        return { text: change.op === "delete" ? null : change.text, inverse: { text: old.text } };
    }
  }

  /** Return a conflict over the member `file`, holding its hash and content within `maxReadChars`. */
  #conflict(reason: string, file: MemberFile): Applied {
    const { maxReadChars } = this.#store.limits;
    const current: JsonObject = { path: file.path, hash: file.hash, text: firstChars(file.text, maxReadChars) };
    if (file.chars > maxReadChars) {
      current.cut = `${plural(file.chars - maxReadChars, "more character")} not shown; read the file for the rest`;
    }
    return { ok: false, reason, current };
  }

  /**
   * @internal Catch up on the journal and resolve a change a crash cut off, holding the lock, as
   * `open` does before it returns the session.
   *
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   */
  recover(signal: AbortSignal): Promise<void> {
    return this.#locked(signal, () => this.#recover());
  }

  /**
   * Catch up on the journal and resolve its last entry when it has no outcome: commit it when its
   * file is at the entry's after hash (the change landed), abort it at its before hash (nothing
   * landed, so its id is free), and otherwise store an unknown outcome, which is never applied
   * again. Then remove the temp files a crash left.
   *
   * Each outcome line is synced before anything else is written, and a crash at any point here
   * leaves a journal the next recovery resolves the same way.
   */
  #recover(): void {
    const { root, limits } = this.#store;
    const pending = this.#journal.catchUp();
    if (pending !== null) {
      const hash = fileHash(root, pending.path, limits);
      if (hash === pending.after) {
        const version = memoryVersion(readMembers(root, limits).files);
        this.#journal.writeOutcome(pending.id, { ok: true, version, result: resultOf(pending, version) });
      } else if (hash === pending.before) {
        this.#journal.writeOutcome(pending.id, "aborted");
      } else {
        this.#journal.writeOutcome(pending.id, {
          ok: false,
          reason: "this change may have landed; check the file",
          current: hash === null ? null : { path: pending.path, hash },
        });
      }
    }
    clearTmp(root);
  }

  /** Throw when the session is closed. */
  #checkOpen(): void {
    if (this.#closed) throw new Error("this FileMemory session is closed");
  }

  /** Run `body` holding the memory's lock, in this process's turn on the memory. */
  #locked<T>(signal: AbortSignal | undefined, body: Held<T>): Promise<T> {
    if (this.#closed) return Promise.reject(new Error("this FileMemory session is closed"));
    const { result, done } = this.#store.locked(signal, body);
    // The turn itself, not the call: an abort while queued lets the call go before its turn runs.
    this.#last = done;
    return result.catch((error: unknown) => {
      throw withoutLocation(error, this.#store.root);
    });
  }
}
