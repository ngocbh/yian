/**
 * FileMemory: a memory that is a folder of UTF-8 text files, with an optional index file.
 *
 * Its members are the regular files whose path components are plain names, within the depth cap,
 * holding UTF-8 text within the per-file cap. Everything else in the folder (hidden entries such
 * as `.git` and `.yian/`, other names, symlinks, binary or oversized files) isn't part of the
 * memory: tools refuse it and the version leaves it out. The memory's own metadata lives in
 * `<root>/.yian/`: `identity`, the journal, the lock and a temp folder.
 *
 * It imports `defaults`, `diagnostics`, `platform` and `typesys`, and `memory/base.ts`.
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
import { canonicalJson, type JsonObject } from "../typesys/json.ts";
import {
  type Applied,
  type Author,
  type ChangeEntry,
  type Erased,
  MemoryBase,
  MemoryBusyError,
  type MemoryContract,
  type MemoryInfo,
  type MemorySession,
  type MemoryToolSpec,
  type Recall,
} from "./base.ts";

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
  /** Folders a file may sit under. */
  maxDepth?: number;
  /** Characters of the recall. */
  maxRecallChars?: number;
  /** Characters one read tool result shows. */
  maxReadChars?: number;
  /** Entries `list` shows and lines `search` shows. */
  maxResults?: number;
  /** Changes the memory can undo. */
  maxUndo?: number;
  /** Change ids the memory keeps to answer a repeated change. */
  maxChangeIds?: number;
  /** Characters of old text the memory keeps for undo (default `maxTotalChars`). */
  maxHistoryChars?: number;
  /** Longest wait in milliseconds for the memory's lock; 0 fails at once when it is held. */
  lockTimeoutMs?: number;
  /** The clock and sleep the lock wait uses, and the clock that stamps changes. */
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
function count(name: string, value: number | undefined, fallback: number, minimum: number): number {
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
 * Return why the relative path `rel` can't name a member by its text alone, or `null` when it can.
 * A member's path is relative, its components plain names, and its file sits under at most
 * `maxDepth` folders.
 */
function pathProblem(rel: string, maxDepth: number): string | null {
  if (rel === "") return "an empty path";
  if (isAbsolute(rel)) return "an absolute path";
  const parts = rel.split("/");
  for (const part of parts) {
    if (part === "") return "an empty part";
    if (part === "..") return "a `..` part";
    if (part === ".") return "a `.` part";
    if (part.startsWith(".")) return "a hidden name";
    if (!NAME.test(part)) return "a name other than letters, digits, `.`, `_` and `-`, starting with a letter or digit";
  }
  if (parts.length - 1 > maxDepth) return `more than ${maxDepth} folder${maxDepth === 1 ? "" : "s"} deep`;
  return null;
}

/**
 * Return why the existing components of `rel` under `root` can't lead to a member, checked with
 * `lstat`, or `null` when they can. A missing component is fine: a write creates it.
 */
function diskProblem(root: string, rel: string): string | null {
  const parts = rel.split("/");
  let path = root;
  for (let i = 0; i < parts.length; i++) {
    path = join(path, parts[i]!);
    let stats: fs.Stats;
    try {
      stats = fs.lstatSync(path);
    } catch (error) {
      if (isSystemError(error) && error.code === "ENOENT") return null;
      throw error;
    }
    const last = i === parts.length - 1;
    if (stats.isSymbolicLink()) return last ? "a symlink" : `${shown(parts.slice(0, i + 1).join("/"))} is a symlink`;
    if (!last && !stats.isDirectory()) return `${shown(parts.slice(0, i + 1).join("/"))} isn't a folder`;
    if (last && !stats.isFile()) return stats.isDirectory() ? "a folder" : "not a regular file";
  }
  return null;
}

/** Return the number of code points in `text`, which holds no lone surrogate. */
function charCount(text: string): number {
  let chars = text.length;
  for (let i = 0; i < text.length; i++) {
    const unit = text.charCodeAt(i);
    if (unit >= 0xd800 && unit <= 0xdbff) chars--;
  }
  return chars;
}

/** Return the sha256 hex digest of `data`. */
function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/**
 * Open the regular file at `path` with `flags`, never following a symlink at its last component,
 * and return its descriptor.
 *
 * @throws Error a Node system error when the open fails or `path` isn't a regular file.
 */
function openRegular(path: string, flags: number, mode?: number): number {
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

/** One member: its path under the root, its text, the sha256 of its bytes and its characters. */
export interface MemberFile {
  readonly path: string;
  readonly text: string;
  readonly hash: string;
  readonly chars: number;
}

/**
 * Read the file at `path` as a member, or return why it isn't one (not a regular file, not UTF-8
 * text, or over `maxFileChars` characters), or `null` when nothing is there.
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
    // A UTF-8 character takes at most four bytes, so a bigger file is over the cap unread.
    if (fs.fstatSync(fd).size > 4 * maxFileChars) return `over ${maxFileChars} characters`;
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

/**
 * Write `data` to `target` atomically: a private temp file in `tmp`, synced, renamed onto
 * `target`, then both folders synced, so a crash leaves the old file or the new one.
 */
function replaceFile(target: string, data: string | Uint8Array, tmp: string): void {
  const temporary = join(tmp, randomBytes(8).toString("hex"));
  const { O_CREAT, O_EXCL, O_WRONLY } = fs.constants;
  const fd = openRegular(temporary, O_CREAT | O_EXCL | O_WRONLY, 0o600);
  try {
    try {
      // The open's mode is masked by the umask; fchmod isn't.
      fs.fchmodSync(fd, 0o600);
      fs.writeFileSync(fd, data);
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
 * Make the private folder `path` unless it exists, and sync its parent when it was made.
 *
 * @throws Error when `path` exists and isn't a folder (a symlink to one included).
 */
function makeDirectory(path: string, name: string): void {
  try {
    fs.mkdirSync(path, { mode: 0o700 });
  } catch (error) {
    if (!isSystemError(error) || error.code !== "EEXIST") throw error;
    if (!fs.lstatSync(path).isDirectory()) throw new Error(`${shown(name)} isn't a folder`);
    return;
  }
  fs.chmodSync(path, 0o700);
  syncFolder(dirname(path));
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

/** What a session needs of its memory: the folder, its caps, its lock and its clock. */
interface Store {
  /** The memory's folder, symlinks resolved. */
  readonly root: string;
  readonly index: string;
  readonly limits: Limits;
  readonly scheduler: Scheduler;
  /**
   * Take the memory's lock, waiting at most the lock timeout.
   *
   * @throws {MemoryBusyError} If another writer holds it past the timeout.
   */
  lock(signal?: AbortSignal): Promise<FileLock>;
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
    const maxTotalChars = count("maxTotalChars", options.maxTotalChars, defaults.fileMemoryMaxTotalChars, 1);
    const limits: Limits = Object.freeze({
      maxFileChars: count("maxFileChars", options.maxFileChars, defaults.fileMemoryMaxFileChars, 1),
      maxTotalChars,
      maxFiles: count("maxFiles", options.maxFiles, defaults.fileMemoryMaxFiles, 1),
      maxDepth: count("maxDepth", options.maxDepth, defaults.fileMemoryMaxDepth, 1),
      maxRecallChars: count("maxRecallChars", options.maxRecallChars, defaults.fileMemoryMaxRecallChars, 1),
      maxReadChars: count("maxReadChars", options.maxReadChars, defaults.fileMemoryMaxReadChars, 1),
      maxResults: count("maxResults", options.maxResults, defaults.fileMemoryMaxResults, 1),
      maxUndo: count("maxUndo", options.maxUndo, defaults.memoryMaxUndo, 0),
      maxChangeIds: count("maxChangeIds", options.maxChangeIds, defaults.memoryMaxChangeIds, 1),
      maxHistoryChars: count("maxHistoryChars", options.maxHistoryChars, maxTotalChars, 0),
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
      async lock(signal?: AbortSignal): Promise<FileLock> {
        try {
          return await waitForLock(join(real, META, "lock"), { timeoutMs, signal, scheduler, private: true });
        } catch (error) {
          if (error instanceof FileLockError && error.reason === "busy") {
            throw new MemoryBusyError("this memory is busy: another writer holds its lock", { cause: error });
          }
          throw error;
        }
      },
    });
  }

  /**
   * Open a session on the memory. With `create`, a missing folder is made (0700), and a folder
   * that isn't a memory yet is adopted with its files; either stores the identity in
   * `.yian/identity` and reports `created`.
   *
   * @throws {TypeError} If `itemType` isn't `null`.
   * @throws Error when the memory doesn't exist and `create` is false, its stored identity isn't
   *   the constructor's `id`, or the root or `.yian` isn't a folder.
   * @throws {MemoryBusyError} If creating it waits for the lock longer than the timeout.
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
      if (!o.create) throw new Error(`${shown(root)} isn't a memory yet: there is no folder`);
      makeRoot(root);
    } else if (!stats.isDirectory()) {
      throw new Error(`${shown(root)} isn't a folder`);
    }
    let identity = storedIdentity(root);
    let created = false;
    if (identity === null) {
      if (!o.create) throw new Error(`${shown(root)} isn't a memory yet`);
      makeDirectory(join(root, META), META);
      makeDirectory(join(root, META, "tmp"), `${META}/tmp`);
      // Two opens may adopt one folder at once: the lock lets only one of them store an identity.
      const lock = await this.#store.lock(o.signal);
      try {
        identity = storedIdentity(root);
        if (identity === null) {
          identity = this.#id ?? `at-${sha256(root).slice(0, 16)}`;
          replaceFile(join(root, META, "identity"), identity, join(root, META, "tmp"));
          created = true;
        }
      } finally {
        lock.release();
      }
    }
    if (this.#id !== undefined && identity !== this.#id) {
      throw new Error(`this memory is ${shown(identity)}, not ${shown(this.#id)}`);
    }
    return new FileMemorySession(this.#store, sessionInfo(identity, this.#store), created);
  }

  /** Empty the memory, keeping its identity. */
  override reset(_o: { signal: AbortSignal }): Promise<void> {
    return Promise.reject(new Error("FileMemory reset isn't available yet"));
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

/** One open FileMemory. */
class FileMemorySession implements MemorySession {
  readonly info: MemoryInfo;
  readonly created: boolean;
  readonly #store: Store;

  constructor(store: Store, info: MemoryInfo, created: boolean) {
    this.#store = store;
    this.info = info;
    this.created = created;
  }

  snapshot(): Promise<{ version: string; recall: Recall }> {
    return Promise.reject(new Error("FileMemory snapshot isn't available yet"));
  }

  call(_tool: string, _args: JsonObject, _o: { signal: AbortSignal }): Promise<JsonObject> {
    return Promise.reject(new Error("FileMemory tools aren't available yet"));
  }

  apply(_c: { id: string; change: JsonObject; author: Author; signal: AbortSignal }): Promise<Applied> {
    return Promise.reject(new Error("FileMemory changes aren't available yet"));
  }

  undo(_o: { id: string; changeId: string; author: Author; signal: AbortSignal }): Promise<Applied> {
    return Promise.reject(new Error("FileMemory undo isn't available yet"));
  }

  erase(_o: { id: string; target: JsonObject; author: Author; signal: AbortSignal }): Promise<Erased> {
    return Promise.reject(new Error("FileMemory erase isn't available yet"));
  }

  history(_o: { limit: number }): Promise<ChangeEntry[]> {
    return Promise.reject(new Error("FileMemory history isn't available yet"));
  }

  async close(): Promise<void> {
    void this.#store;
  }
}
