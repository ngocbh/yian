/**
 * How the command line binds a flow's memory inputs to instances: `--memory NAME=PATH` binds one
 * to a folder or a `.db` store, and an unbound one gets its default identity,
 * `<project key>/<flow id>/<input>`, at its location under `~/.yian/memory`, confined there. A new
 * store's embedding model is `--embedding`'s, else the saved default's. The instances are
 * `CliFileMemory`s and `CliVectorDBMemory`s, which say when their `open` created the memory.
 *
 * Knows about: compose (the flow's inputs), memory (`FileMemory`, `VectorDBMemory`), the embeddings
 * client's model names, the run log (a resumed run's START inputs), the type language, the
 * platform's path helpers and the file system.
 */

import { createHash } from "node:crypto";
import { lstatSync, type Stats } from "node:fs";
import { userInfo } from "node:os";
import { basename, dirname, join, relative, resolve, sep } from "node:path";

import { START_ID } from "../compile/model.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { reprText } from "../diagnostics.ts";
import { memoryErrorKind, type MemorySession } from "../memory/base.ts";
import { FileMemory, type FileMemoryOptions } from "../memory/file.ts";
import { VectorDBMemory, type VectorDBMemoryOptions } from "../memory/vectordb.ts";
import type { Embed, EmbeddingModel } from "../llm/embeddings.ts";
import { embeddingModel } from "../llm_clients/connection.ts";
import type { Parameter } from "../nodes/contracts.ts";
import { isSystemError, leavesRoot } from "../platform.ts";
import type { MemoryDrift } from "../run.ts";
import { iterRunRecords, type PoolCommitRecord } from "../runtime/log.ts";
import type { JsonObject } from "../typesys/json.ts";
import { isMemoryType, isRecord, type MemoryRef, TypedValue } from "../typesys/values.ts";
import { resolveFlowPath } from "./confined_load.ts";
import { UsageError } from "./utils.ts";

/** Hex digits of the real-path hash in a project key. */
const PROJECT_KEY_HEX = 8;

/** The most characters of the folder-name part of a project key. */
const PROJECT_NAME_CHARS = 32;

/** What a flow's `id:` must be for its memories to get a default identity: a plain name. */
const PLAIN_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** The longest flow `id:` a default identity takes, so its folder's name stays within the OS's limit. */
const PLAIN_ID_CHARS = 100;

/** Return the folder the CLI's default memories live in, `~/.yian/memory`, under `home`. */
export function memoryRoot(home: string): string {
  return join(home, ".yian", "memory");
}

/**
 * Return the project key of the folder `dir`, a real path: its name with every character outside
 * `[A-Za-z0-9._-]` made `-`, leading `.` and `-` dropped, cut to 32 characters (`project` when
 * nothing is left), then `-` and the first 8 hex digits of the path's SHA-256.
 */
export function projectKey(dir: string): string {
  const name = basename(dir).replace(/[^A-Za-z0-9._-]/g, "-").replace(/^[.-]+/, "").slice(0, PROJECT_NAME_CHARS);
  const hash = createHash("sha256").update(dir, "utf8").digest("hex").slice(0, PROJECT_KEY_HEX);
  return `${name === "" ? "project" : name}-${hash}`;
}

/**
 * Return the project folder of the flow file at `path`: the real path of the folder the path names,
 * not the folder a symlinked file resolves to, so a project's own link to a shared flow keeps the
 * project's memories. `yian run` and `yian chat` both key a flow's defaults by it.
 */
export function projectFolder(path: string): string {
  return resolveFlowPath(dirname(resolve(path)));
}

/**
 * Return the default identity of memory input `input` of the flow `flowId` whose file is in the
 * project folder `dir` (`projectFolder`): `<project key>/<flow id>/<input>`.
 *
 * @param memoryFlag Whether the host takes `--memory`, which the refusal then suggests.
 * @throws {RangeError} When the flow has no `id:`, or one that isn't a plain name of at most 100
 *   characters.
 */
export function defaultIdentity(
  dir: string,
  flowId: string | null | undefined,
  input: string,
  memoryFlag = true,
): string {
  if (flowId == null || flowId.length > PLAIN_ID_CHARS || !PLAIN_ID.test(flowId)) {
    const instead = memoryFlag ? `or bind \`${input}\` with --memory` : `so \`${input}\` gets a default memory`;
    const plain = "a plain `id:` of at most 100 characters (letters, digits, `.`, `_`, `-`)";
    throw new RangeError(`give the flow ${plain} ${instead}`);
  }
  return `${projectKey(dir)}/${flowId}/${input}`;
}

/**
 * Return the default location of a memory of type `type` with identity `identity`, under `home`:
 * the folder `~/.yian/memory/<identity>` for a FileMemory, the store `<identity>.db` for a
 * VectorDBMemory. It is checked as `checkDefaultLocation` checks it.
 *
 * @throws As `checkDefaultLocation` throws.
 */
export function defaultLocation(home: string, identity: string, type: "FileMemory" | "VectorDBMemory"): string {
  return checkDefaultLocation(home, type === "VectorDBMemory" ? `${identity}.db` : identity);
}

/**
 * Return `~/.yian/memory/<path>` under `home`, once it is confined there: the path stays inside
 * `~/.yian/memory` as written, each of its components that exists is a real entry (a folder, but
 * for the last, which its memory type checks), never a symlink, and its real path stays under the
 * real `~/.yian/memory`. A recorded identity is checked this way too, since a log can be edited.
 *
 * @throws {RangeError} When the path leaves `~/.yian/memory` or names it, a component is a symlink,
 *   a component above the last isn't a folder, or the real path leaves the real `~/.yian/memory`;
 *   each message names the path from `~`.
 * @throws {TypeError} When the path holds a NUL character, which no path can.
 * @throws Error (a system error, such as `EACCES`, `ENAMETOOLONG` or `ELOOP`) when a component
 *   can't be read or a real path can't be resolved; its message names the absolute path.
 */
export function checkDefaultLocation(home: string, path: string): string {
  const root = memoryRoot(home);
  const location = join(root, path);
  const rel = relative(root, location);
  if (rel === "" || leavesRoot(rel)) throw new RangeError(`memory location \`${path}\` leaves ~/.yian/memory`);
  const parts = rel.split(sep);
  let at = root;
  for (const [i, part] of parts.entries()) {
    at = join(at, part);
    const stats = lstatOrNull(at);
    if (stats === null) break;
    // Written from `~`, as a host may show the message to a model.
    const shown = `~/.yian/memory/${parts.slice(0, i + 1).join("/")}`;
    if (stats.isSymbolicLink()) {
      throw new RangeError(`${shown} is a symlink; a default memory stays inside ~/.yian/memory`);
    }
    if (i < parts.length - 1 && !stats.isDirectory()) throw new RangeError(`${shown} isn't a folder`);
  }
  // On the real paths too, as `confined_load.ts` confines flow files.
  if (leavesRoot(relative(resolveFlowPath(root), resolveFlowPath(location)))) {
    throw new RangeError(`memory location \`${path}\` leaves ~/.yian/memory`);
  }
  return location;
}

/** `lstatSync`, or `null` when nothing is at `path`. */
function lstatOrNull(path: string): Stats | null {
  try {
    return lstatSync(path);
  } catch (error) {
    if (isSystemError(error) && error.code === "ENOENT") return null;
    throw error;
  }
}

/**
 * Return the account's home folder as the system records it, whatever `HOME` says, or `null` when
 * the system has no entry for the account.
 */
export function accountHome(): string | null {
  try {
    return userInfo().homedir || null;
  } catch {
    // A uid with no passwd entry, as in some containers: `HOME` alone is checked.
    return null;
  }
}

/**
 * Refuse a folder that can't be a memory, compared by real paths: `/`, a home folder, or a folder
 * that holds a home's `.yian` (or is it). The homes are `home`, the one `HOME` names, and
 * `account`, the account's own (`accountHome`, `null` when unknown), so a `HOME` pointing elsewhere
 * can't let the account's home through. Binding adopts a folder's files as the memory's, and a
 * reset deletes them, so one of these would take everything under it.
 *
 * @throws {RangeError} Naming the folder and why.
 */
export function checkMemoryFolder(path: string, home: string, account: string | null): void {
  const real = realOr(resolve(path));
  const holds = (dir: string) => !leavesRoot(relative(real, realOr(join(dir, ".yian"))));
  const refuse = (why: string) => new RangeError(`\`${path}\` can't be a memory: ${why}`);
  if (real === "/") throw new RangeError("`/` can't be a memory: it holds the whole file system");
  if (real === realOr(home)) throw new RangeError("`~` can't be a memory: it holds your home's files");
  if (account !== null && real === realOr(account)) throw refuse("it holds your home's files");
  if (holds(home)) throw refuse("it holds ~/.yian");
  if (account !== null && holds(account)) throw refuse(`it holds ${join(account, ".yian")}`);
}

/**
 * The note for a memory a crash resume found changed since its run started, the run's own writes
 * included: the run goes on with the recall it recorded.
 */
export function driftNote(drift: MemoryDrift): string {
  return `memory \`${drift.input}\` changed since this run started; the run keeps its recorded recall`;
}

/** A memory an instance's `open` created (or adopted): the inputs bound to it, its identity and folder. */
export interface CreatedMemory {
  readonly inputs: readonly string[];
  readonly identity: string;
  readonly location: string;
}

/**
 * The line that says `created` was created: `memory lessons: <identity> at <folder> (new)`, or,
 * without `location`, for a host whose lines reach a model or a run log, the identity alone.
 */
export function createdLine(created: CreatedMemory, location: boolean): string {
  const where = location ? ` at ${created.location}` : "";
  return `memory ${created.inputs.join(", ")}: ${created.identity}${where} (new)`;
}

/**
 * Return `error` with the memory folders and stores `roots` and the home folder `home` taken out of
 * its message, for a host whose messages reach a model or a run log, which never hold a location: a
 * root becomes `the memory's folder` (`the memory's store` for a `.db`), and `~/.yian/memory`,
 * `~/.yian` and the home are written from `~`, each by its real path too, since `~/.yian/memory`
 * may point out of the home. A home of `/` stays, or every path would. The class, name and a system
 * error's code are kept.
 */
export function withoutLocations(error: unknown, home: string, roots: Iterable<string>): unknown {
  if (!(error instanceof Error)) return error;
  // Each path's name; the first given wins, so a folder inside another keeps its own.
  const names = new Map<string, string>();
  const name = (path: string, as: string) => {
    if (path !== "/" && !names.has(path)) names.set(path, as);
  };
  for (const root of roots) name(root, root.endsWith(".db") ? "the memory's store" : "the memory's folder");
  for (const [path, as] of [
    [memoryRoot(home), "~/.yian/memory"],
    [join(home, ".yian"), "~/.yian"],
    [home, "~"],
  ] as const) {
    name(realOr(path), as);
    name(path, as);
  }
  const message = replacePaths(error.message, names);
  if (message === error.message) return error;
  const Class = error instanceof RangeError ? RangeError : error instanceof TypeError ? TypeError : Error;
  const copy: Error & { code?: unknown; syscall?: unknown; errno?: unknown } = new Class(message);
  copy.name = error.name;
  // Not `path`, `dest` or `cause`, which hold the location too.
  const { code, syscall, errno } = error as NodeJS.ErrnoException;
  if (code !== undefined) copy.code = code;
  if (syscall !== undefined) copy.syscall = syscall;
  if (errno !== undefined) copy.errno = errno;
  return copy;
}

/** `path`'s real path, or `path` itself when it can't be resolved. */
function realOr(path: string): string {
  try {
    return resolveFlowPath(path);
  } catch {
    // Called while reporting another error, which this one mustn't replace.
    return path;
  }
}

/**
 * Return `text` with each whole occurrence of a path of `names`, alone or as a folder of a longer
 * path, written as its name, in one pass, so a name is never matched again; the longest path is
 * tried first at each place. A path that only starts with the same characters (`/home/ada2`)
 * stays.
 */
function replacePaths(text: string, names: ReadonlyMap<string, string>): string {
  if (names.size === 0) return text;
  const paths = [...names.keys()].sort((a, b) => b.length - a.length);
  const escaped = paths.map((path) => path.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  // A path ends where a message's punctuation or a folder of it begins.
  const pattern = new RegExp(`(?:${escaped.join("|")})(?=$|[/'"\`\\s),:;.\\]])`, "g");
  return text.replace(pattern, (path) => names.get(path)!);
}

/**
 * Return the `--embedding provider/model` flag's model, or `null` when it isn't given. The model is
 * everything after the first `/`.
 *
 * @throws {UsageError} When it isn't `provider/model`, or the provider has no embeddings API yian
 *   supports.
 */
export function embeddingFlag(value: string | undefined): EmbeddingModel | null {
  if (value === undefined) return null;
  try {
    return embeddingModel(value);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw new UsageError(`Invalid value for --embedding: ${error.message}`);
  }
}

/** What `bindMemories` needs beside the flow and the flags. */
export interface BindOptions {
  /** The home folder whose `~/.yian/memory` holds the default memories. */
  home: string;
  /** Told of each memory an instance's `open` creates (or adopts). */
  note: (created: CreatedMemory) => void;
  /** Whether the host takes `--memory` and `--embedding`, which its refusals then suggest; true when absent. */
  memoryFlag?: boolean;
  /**
   * The embedding model a new VectorDB store is made with (`--embedding`), or `null` for none; an
   * existing store keeps its own.
   */
  embedding?: EmbeddingModel | null;
  /**
   * The model a new VectorDB store is made with when `embedding` names none (`defaultEmbedding`), or
   * `null`. Called only for a store that doesn't exist yet.
   */
  defaultEmbedding?: () => EmbeddingModel | null;
  /** The embeddings call VectorDB memories make. */
  embed?: Embed;
  /** The account's home folder, which a `--memory` folder can't be beside `home`; `accountHome` when absent. */
  accountHome?: () => string | null;
}

/** A memory the command line binds, of either built-in type. */
export type CliMemory = FileMemory | VectorDBMemory;

/** Return where `memory` is: a FileMemory's folder, a VectorDBMemory's store file. */
export function memoryLocation(memory: CliMemory): string {
  return memory instanceof FileMemory ? memory.root : memory.path;
}

/**
 * For a default memory: the home, its path under `~/.yian/memory` (its identity, with `.db` for a
 * store), its identity, and whether its host takes `--memory`.
 */
interface Confined {
  readonly home: string;
  readonly path: string;
  readonly identity: string;
  readonly memoryFlag: boolean;
}

/**
 * Open a command line's memory through `open`, at `location`, a `noun` (`folder` or `store`). A
 * default one is first checked again as `checkDefaultLocation` checks it, and must still be at the
 * real path the bind resolved: prompts may come between the bind and the open. A resume's memory
 * that isn't there is said in the host's words, and a memory the open creates (or adopts) is told
 * to `note`.
 *
 * @throws {RangeError} When a default location no longer passes the checks, or its real path
 *   changed after the bind. For a resume's memory that isn't there: a path given with `--memory`,
 *   or a default of a host without the flag, whose run can't go on.
 */
async function openChecked(
  memory: { readonly inputs: readonly string[]; readonly note: (created: CreatedMemory) => void },
  confined: Confined | null,
  where: { location: string; noun: "folder" | "store" },
  open: () => Promise<MemorySession>,
): Promise<MemorySession> {
  const { location, noun } = where;
  if (confined !== null && resolveFlowPath(checkDefaultLocation(confined.home, confined.path)) !== location) {
    throw new RangeError(`~/.yian/memory/${confined.path} changed after it was checked; run again`);
  }
  let session: MemorySession;
  try {
    session = await open();
  } catch (error) {
    // A resume's "bind it with --memory" would point back at the flag the user gave, or at one
    // the host (`yian chat`) doesn't take.
    if (memoryErrorKind(error) !== "missing" || confined?.memoryFlag === true) throw error;
    const gone = lstatOrNull(location) === null ? `its ${noun} is gone` : "it isn't a memory any more";
    const reason =
      confined === null ? `the ${noun} given with --memory isn't a memory` : `${gone}, so the run can't continue`;
    throw new RangeError(reason, { cause: error });
  }
  if (session.created) memory.note({ inputs: [...memory.inputs], identity: session.info.identity, location });
  return session;
}

/**
 * A FileMemory the command line binds: its `open` is `openChecked`'s, telling `note` of each
 * memory it creates (or adopts), and checking a default's location again.
 */
class CliFileMemory extends FileMemory {
  /** The inputs bound to this instance, in the order they were bound. */
  readonly inputs: string[] = [];
  readonly note: (created: CreatedMemory) => void;
  readonly #confined: Confined | null;

  /** Whether this is an input's default memory, rather than one a `--memory` flag names. */
  get isDefault(): boolean {
    return this.#confined !== null;
  }

  /** A default memory's identity; `null` for another. */
  get defaultIdentity(): string | null {
    return this.#confined?.identity ?? null;
  }

  constructor(options: FileMemoryOptions, note: (created: CreatedMemory) => void, confined: Confined | null = null) {
    super(options);
    this.note = note;
    this.#confined = confined;
  }

  override open(o: { create: boolean; itemType: JsonObject | null; signal: AbortSignal }): Promise<MemorySession> {
    return openChecked(this, this.#confined, { location: this.root, noun: "folder" }, () => super.open(o));
  }
}

/** A VectorDBMemory the command line binds; its `open` is `openChecked`'s, as `CliFileMemory`'s. */
class CliVectorDBMemory extends VectorDBMemory {
  /** The inputs bound to this instance, in the order they were bound. */
  readonly inputs: string[] = [];
  readonly note: (created: CreatedMemory) => void;
  readonly #confined: Confined | null;

  /** Whether this is an input's default memory, rather than one a `--memory` flag names. */
  get isDefault(): boolean {
    return this.#confined !== null;
  }

  /** A default memory's identity; `null` for another. */
  get defaultIdentity(): string | null {
    return this.#confined?.identity ?? null;
  }

  constructor(
    options: VectorDBMemoryOptions,
    note: (created: CreatedMemory) => void,
    confined: Confined | null = null,
  ) {
    super(options);
    this.note = note;
    this.#confined = confined;
  }

  override open(o: { create: boolean; itemType: JsonObject | null; signal: AbortSignal }): Promise<MemorySession> {
    return openChecked(this, this.#confined, { location: this.path, noun: "store" }, () => super.open(o));
  }
}

/** An instance the command line binds. */
type Bound = CliFileMemory | CliVectorDBMemory;

/**
 * Return the `--memory NAME=PATH` flags as each named memory input's path, checked against the
 * flow's inputs `params`. The name is what comes before the first `=`, and the path what comes
 * after it, each trimmed; the binding reads a leading `~/` as the home folder.
 *
 * @throws {UsageError} For a flag without `=` or a path, a name that isn't a memory input or
 *   comes twice, a `.db` path for a FileMemory input, or another for a VectorDBMemory input.
 */
export function memoryFlags(params: readonly Parameter[], flags: readonly string[]): Map<string, string> {
  const paths = new Map<string, string>();
  const refuse = (why: string) => new UsageError(`Invalid value for --memory: ${why}`);
  for (const flag of flags) {
    const at = flag.indexOf("=");
    if (at === -1) throw new UsageError(`Invalid value: --memory must be NAME=PATH (got ${reprText(flag)})`);
    const name = flag.slice(0, at).trim();
    const path = flag.slice(at + 1).trim();
    const param = params.find((each) => each.name === name);
    const typ = param?.contract?.type;
    if (param === undefined) throw refuse(`the flow has no memory input \`${name}\``);
    if (!isMemoryType(typ)) throw refuse(`input \`${name}\` isn't a memory`);
    if (paths.has(name)) throw refuse(`\`${name}\` is given twice`);
    if (path === "") throw refuse(`\`${name}\` needs a path`);
    if (typ.memory === "FileMemory" && path.endsWith(".db")) {
      throw refuse(`\`${name}\` is a FileMemory, a folder, not a .db file`);
    }
    if (typ.memory === "VectorDBMemory" && !path.endsWith(".db")) {
      throw refuse(`\`${name}\` is a VectorDBMemory, a .db file, not a folder`);
    }
    paths.set(name, path);
  }
  return paths;
}

/**
 * Return the instance each memory input of `flow` is bound to, by input name: the folder or `.db`
 * store its `--memory` flag names, else, for an input that isn't Optional, its default
 * (`defaultIdentity` of `options.project` at `defaultLocation`). An Optional input left unbound
 * gets none, so it stays null. Inputs given one path, however it is spelled, share one instance. A
 * VectorDB store that doesn't exist yet is made with `options.embedding`; an existing one keeps its
 * own model. Nothing is created here: an instance's `open` creates its memory, and notes it.
 *
 * @param options.project The flow file's project folder (`projectFolder`), or `null` for a flow
 *   with no file, whose memories have no default.
 * @throws {UsageError} As `memoryFlags` throws, and for a flag naming another input's default.
 * @throws {RangeError} For an input of a memory type the command line can't bind, a new VectorDB
 *   store with no embedding model; for a default, as `defaultIdentity` throws, and when there is no
 *   project folder.
 * @throws As `defaultLocation` throws, for a default.
 */
export function bindMemories(
  flow: ComposedFlow,
  flags: readonly string[],
  options: BindOptions & { project: string | null },
): Record<string, CliMemory> {
  return bindFresh(flow, flags, options);
}

/**
 * How a binding is used: a fresh run's, which may make a new store; a resume's, which opens only
 * what the run started with; or a preview's, which names the memories and opens none.
 */
type BindMode = "fresh" | "resume" | "preview";

/**
 * Bind `flow`'s memory inputs as `bindMemories` describes, to the CLI's instances; a `preview`
 * needs no embedding model, since it makes nothing.
 */
function bindFresh(
  flow: ComposedFlow,
  flags: readonly string[],
  options: BindOptions & { project: string | null },
  mode: "fresh" | "preview" = "fresh",
): Record<string, Bound> {
  const { memoryFlag = true } = options;
  return bindEach(flow, flags, options, mode, (input, optional, type) => {
    if (optional) return null;
    if (options.project === null) {
      const refusal = "a flow with no file has no default memories";
      const instead = memoryFlag ? `; bind \`${input}\`` : `, so \`${input}\` can't be bound`;
      throw new RangeError(`${refusal}${instead}`);
    }
    const identity = defaultIdentity(options.project, flow.compiled.flowId, input, memoryFlag);
    return defaultInstance(input, identity, type, options, mode);
  });
}

/** A default memory a run would bind: its input, its identity, and whether it exists yet. */
export interface DefaultMemory {
  readonly input: string;
  readonly identity: string;
  readonly exists: boolean;
}

/**
 * Return the default memory each memory input of `flow` gets when no flag binds it, in declaration
 * order, as `bindMemories` binds them: its identity, and whether a memory is at its location yet.
 * An Optional input gets none, so it isn't listed. Nothing is created.
 *
 * @throws As `bindMemories` throws for a default.
 */
export function defaultMemories(
  flow: ComposedFlow,
  options: Omit<BindOptions, "note"> & { project: string | null },
): DefaultMemory[] {
  // With no flags, each instance is a default.
  const bound = bindFresh(flow, [], { ...options, note: () => {} }, "preview");
  return Object.entries(bound).map(([input, instance]) => ({
    input,
    identity: instance.defaultIdentity!,
    exists:
      instance instanceof FileMemory
        ? lstatOrNull(join(instance.root, ".yian", "identity")) !== null
        : lstatOrNull(instance.path) !== null,
  }));
}

/**
 * Return the instance each memory input of `flow`, the flow its run in `log` recorded, is bound to
 * again for a resume, by input name: the path its `--memory` flag names, else the default
 * location of the identity it recorded at the run's start (`recordedMemories`), checked as a fresh
 * default is, since a log can be edited. An Optional input left out at the start gets none. A
 * VectorDB store is bound with no embedding model, so the one it recorded is used. Nothing is
 * created here, and the resume opens each memory without creating it.
 *
 * @throws {UsageError} As `memoryFlags` throws, and for a flag naming another input's default.
 * @throws {RangeError} For an input of a memory type the command line can't bind, or when the log's
 *   START inputs hold a malformed memory value.
 * @throws As `defaultLocation` throws, for a default.
 * @throws Error when the log can't be read.
 */
export function resumeMemories(
  flow: ComposedFlow,
  log: string,
  flags: readonly string[],
  options: BindOptions,
): Record<string, CliMemory> {
  const recorded = recordedMemories(flow, log);
  return bindEach(flow, flags, options, "resume", (input, _optional, type) => {
    const identity = recorded.get(input);
    // An Optional memory left out at the start takes none again.
    if (identity === undefined) return null;
    return defaultInstance(input, identity, type, options, "resume");
  });
}

/**
 * Return the identity each memory input of `flow` recorded in the START inputs of its run in
 * `log`, by input name; an Optional memory left out recorded none.
 *
 * @throws {RangeError} When a memory input's recorded value isn't a memory value.
 * @throws Error when the log can't be read.
 */
export function recordedMemories(flow: ComposedFlow, log: string): Map<string, string> {
  const names = flow.input.filter((param) => isMemoryType(param.contract?.type)).map((param) => param.name);
  const identities = new Map<string, string>();
  if (names.length === 0) return identities;
  for (const entry of iterRunRecords(log, { kinds: ["pool_commit"] })) {
    const record = entry.envelope.record as PoolCommitRecord;
    if (record.nodeId !== START_ID || record.key !== START_ID) continue;
    const inputs = record.value?.value;
    if (!isRecord(inputs)) break;
    for (const name of names) {
      if (!Object.hasOwn(inputs, name)) continue;
      const memory = TypedValue.fromWire({ kind: "memory", value: inputs[name] }).value as MemoryRef;
      identities.set(name, memory.identity);
    }
    break;
  }
  return identities;
}

/**
 * Bind each memory input of `flow`: to the path its `--memory` flag names, else to what `fallback`
 * gives the input, given whether it is Optional and its memory type (`null` for none). `mode`
 * says whether a new VectorDB store may be made.
 */
function bindEach(
  flow: ComposedFlow,
  flags: readonly string[],
  options: BindOptions,
  mode: BindMode,
  fallback: (input: string, optional: boolean, type: string) => Bound | null,
): Record<string, Bound> {
  const paths = memoryFlags(flow.input, flags);
  const bound: Record<string, Bound> = {};
  const byPath = new Map<string, Bound>();
  for (const param of flow.input) {
    const typ = param.contract?.type;
    if (!isMemoryType(typ)) continue;
    const given = paths.get(param.name);
    // As the auth store reads its path: `--memory=NAME=~/x` reaches us unexpanded.
    const path = given === "~" || given?.startsWith("~/") ? options.home + given.slice(1) : given;
    // A store is one file of its own, which a reset empties: only a folder adopts what is in it.
    if (path !== undefined && typ.memory === "FileMemory") {
      try {
        checkMemoryFolder(path, options.home, (options.accountHome ?? accountHome)());
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        throw new UsageError(`Invalid value for --memory: ${error.message}`);
      }
    }
    // Nothing is opened yet, so refusing here leaves no trace.
    if (typ.memory !== "FileMemory" && typ.memory !== "VectorDBMemory" && (path !== undefined || !typ.nullable)) {
      throw new RangeError(
        `\`${param.name}\` is a ${typ.memory}; the command line binds only FileMemory and VectorDBMemory inputs`,
      );
    }
    const made =
      path === undefined
        ? fallback(param.name, typ.nullable, typ.memory)
        : typ.memory === "FileMemory"
          ? new CliFileMemory({ root: path }, options.note)
          : new CliVectorDBMemory(storeOptions(param.name, path, options, mode), options.note);
    if (made === null) continue;
    // Keyed by the real path, so two spellings of one path share one memory.
    const location = memoryLocation(made);
    const known = byPath.get(location);
    // Only flagged inputs share: a flag naming another input's default is a slip, not a choice.
    if (known !== undefined && known.isDefault !== made.isDefault) {
      const [flagged, other] = made.isDefault ? [known.inputs[0]!, param.name] : [param.name, known.inputs[0]!];
      throw new UsageError(
        `Invalid value for --memory: \`${flagged}=${paths.get(flagged)}\` points at \`${other}\`'s default memory`,
      );
    }
    const instance = known ?? made;
    byPath.set(location, instance);
    instance.inputs.push(param.name);
    bound[param.name] = instance;
  }
  return bound;
}

/**
 * Return the options of input `input`'s VectorDB store at `path`: its embedding model and
 * `options.embed`. Only a fresh run makes a store, and only one that doesn't exist yet takes a
 * model: `options.embedding`, else `options.defaultEmbedding`'s. A store that exists keeps the
 * model it recorded, as a resume's and a preview's (which opens nothing) do.
 *
 * @throws {RangeError} When a fresh run's store doesn't exist yet and there is no embedding model;
 *   as `options.defaultEmbedding` throws.
 */
function storeOptions(input: string, path: string, options: BindOptions, mode: BindMode): VectorDBMemoryOptions {
  const store: VectorDBMemoryOptions = { path };
  if (options.embed !== undefined) store.embed = options.embed;
  if (mode !== "fresh" || lstatOrNull(path) !== null) return store;
  // The default is read only for a new store, so a broken one fails only a run that needs it.
  const embedding = options.embedding ?? options.defaultEmbedding?.() ?? null;
  if (embedding === null) {
    const set =
      options.memoryFlag === false
        ? "run `yian config embedding provider/model`"
        : "pass --embedding provider/model or run `yian config embedding provider/model`";
    throw new RangeError(`\`${input}\` needs an embedding model: ${set}`);
  }
  store.embedding = embedding;
  return store;
}

/**
 * Return the instance of input `input`'s memory of type `type` and identity `identity`, at its
 * default location under `options.home`: a folder, or for a VectorDBMemory `<identity>.db`.
 *
 * @throws As `defaultLocation` and `storeOptions` throw.
 */
function defaultInstance(input: string, identity: string, type: string, options: BindOptions, mode: BindMode): Bound {
  const memoryFlag = options.memoryFlag ?? true;
  if (type === "VectorDBMemory") {
    const path = defaultLocation(options.home, identity, "VectorDBMemory");
    const confined = { home: options.home, path: `${identity}.db`, identity, memoryFlag };
    const store = { ...storeOptions(input, path, options, mode), id: identity };
    return new CliVectorDBMemory(store, options.note, confined);
  }
  const root = defaultLocation(options.home, identity, "FileMemory");
  const confined = { home: options.home, path: identity, identity, memoryFlag };
  return new CliFileMemory({ root, id: identity }, options.note, confined);
}
