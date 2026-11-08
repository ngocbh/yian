/**
 * How the command line binds a flow's memory inputs to instances: `--memory NAME=PATH` binds one
 * to a folder, and an unbound one gets its default identity, `<project key>/<flow id>/<input>`,
 * at its location under `~/.yian/memory`, confined there. The instances are `CliFileMemory`s,
 * which say when their `open` created the memory.
 *
 * Knows about: compose (the flow's inputs), memory (`FileMemory`), the run log (a resumed run's
 * START inputs), the type language, the platform's path helpers and the file system.
 */

import { createHash } from "node:crypto";
import { lstatSync, type Stats } from "node:fs";
import { basename, dirname, join, relative, resolve, sep } from "node:path";

import { START_ID } from "../compile/model.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { reprText } from "../diagnostics.ts";
import { memoryErrorKind, type MemorySession } from "../memory/base.ts";
import { FileMemory, type FileMemoryOptions } from "../memory/file.ts";
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
    throw new RangeError(`give the flow a plain \`id:\` (letters, digits, \`.\`, \`_\`, \`-\`) ${instead}`);
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
 * Return `error` with the memory folders `roots` and the home folder `home` taken out of its
 * message, for a host whose messages reach a model or a run log, which never hold a location: a
 * root becomes `the memory's folder`, and a path under `home` is written from `~`. The class, name
 * and a system error's code are kept.
 */
export function withoutLocations(error: unknown, home: string, roots: Iterable<string>): unknown {
  if (!(error instanceof Error)) return error;
  let message = error.message;
  const homes = [...new Set([resolveFlowPath(home), home])];
  for (const root of new Set(roots)) message = message.replaceAll(root, "the memory's folder");
  for (const each of homes) message = message.replaceAll(each, "~");
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

/** What `bindMemories` needs beside the flow and the flags. */
export interface BindOptions {
  /** The home folder whose `~/.yian/memory` holds the default memories. */
  home: string;
  /** Told of each memory an instance's `open` creates (or adopts). */
  note: (created: CreatedMemory) => void;
  /** Whether the host takes `--memory`, which its refusals then suggest; true when absent. */
  memoryFlag?: boolean;
}

/**
 * A FileMemory the command line binds: its `open` tells `note` of each memory it creates (or
 * adopts), with the inputs bound to it, its identity and its folder. A default one checks its
 * location again as it opens.
 */
class CliFileMemory extends FileMemory {
  /** The inputs bound to this instance, in the order they were bound. */
  readonly inputs: string[] = [];
  readonly #note: (created: CreatedMemory) => void;
  /** For a default memory, the home and its path under `~/.yian/memory`; `null` for another. */
  readonly #confined: { readonly home: string; readonly path: string } | null;

  /** Whether this is an input's default memory, rather than one a `--memory` flag names. */
  get isDefault(): boolean {
    return this.#confined !== null;
  }

  /** A default memory's identity, its path under `~/.yian/memory`; `null` for another. */
  get defaultIdentity(): string | null {
    return this.#confined?.path ?? null;
  }

  constructor(
    options: FileMemoryOptions,
    note: (created: CreatedMemory) => void,
    confined: { home: string; path: string } | null = null,
  ) {
    super(options);
    this.#note = note;
    this.#confined = confined;
  }

  /**
   * Open the memory as `FileMemory` does. A default one is first checked again as
   * `checkDefaultLocation` checks it, and must still be at the real folder the bind resolved:
   * prompts may come between the bind and the open.
   *
   * @throws {RangeError} When a default location no longer passes the checks, or its real path
   *   changed after the bind; for a resume's folder given with `--memory`, when it isn't a memory.
   */
  override async open(o: {
    create: boolean;
    itemType: JsonObject | null;
    signal: AbortSignal;
  }): Promise<MemorySession> {
    if (this.#confined !== null) {
      const { home, path } = this.#confined;
      if (resolveFlowPath(checkDefaultLocation(home, path)) !== this.root) {
        throw new RangeError(`~/.yian/memory/${path} changed after it was checked; run again`);
      }
    }
    let session: MemorySession;
    try {
      session = await super.open(o);
    } catch (error) {
      // A resume's "bind it with --memory" would point back at the flag the user gave.
      if (this.#confined !== null || memoryErrorKind(error) !== "missing") throw error;
      throw new RangeError("the folder given with --memory isn't a memory", { cause: error });
    }
    if (session.created) {
      this.#note({ inputs: [...this.inputs], identity: session.info.identity, location: this.root });
    }
    return session;
  }
}

/**
 * Return the `--memory NAME=PATH` flags as each named memory input's path, checked against the
 * flow's inputs `params`. The name is what comes before the first `=`, and the path what comes
 * after it, each trimmed; the binding reads a leading `~/` as the home folder.
 *
 * @throws {UsageError} For a flag without `=` or a path, a name that isn't a memory input or
 *   comes twice, or a `.db` path for a FileMemory input.
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
    paths.set(name, path);
  }
  return paths;
}

/**
 * Return the instance each memory input of `flow` is bound to, by input name: the folder its
 * `--memory` flag names, else, for an input that isn't Optional, its default (`defaultIdentity` of
 * `options.project` at `defaultLocation`). An Optional input left unbound gets none, so it stays
 * null. Inputs given one folder, however its path is spelled, share one instance. Nothing is
 * created here: an instance's `open` creates its memory, and notes it.
 *
 * @param options.project The flow file's project folder (`projectFolder`), or `null` for a flow
 *   with no file, whose memories have no default.
 * @throws {UsageError} As `memoryFlags` throws, and for a flag naming another input's default.
 * @throws {RangeError} For an input of a memory type the command line can't bind; for a default,
 *   as `defaultIdentity` throws, and when there is no project folder.
 * @throws As `defaultLocation` throws, for a default.
 */
export function bindMemories(
  flow: ComposedFlow,
  flags: readonly string[],
  options: BindOptions & { project: string | null },
): Record<string, FileMemory> {
  return bindFresh(flow, flags, options);
}

/** Bind `flow`'s memory inputs as `bindMemories` describes, to the CLI's instances. */
function bindFresh(
  flow: ComposedFlow,
  flags: readonly string[],
  options: BindOptions & { project: string | null },
): Record<string, CliFileMemory> {
  const { memoryFlag = true } = options;
  return bindEach(flow, flags, options, (input, optional) => {
    if (optional) return null;
    if (options.project === null) {
      const refusal = "a flow with no file has no default memories";
      const instead = memoryFlag ? `; bind \`${input}\`` : `, so \`${input}\` can't be bound`;
      throw new RangeError(`${refusal}${instead}`);
    }
    return defaultInstance(defaultIdentity(options.project, flow.compiled.flowId, input, memoryFlag), options);
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
  const bound = bindFresh(flow, [], { ...options, note: () => {} });
  return Object.entries(bound).map(([input, instance]) => ({
    input,
    identity: instance.defaultIdentity!,
    exists: lstatOrNull(join(instance.root, ".yian", "identity")) !== null,
  }));
}

/**
 * Return the instance each memory input of `flow`, the flow its run in `log` recorded, is bound to
 * again for a resume, by input name: the folder its `--memory` flag names, else the default
 * location of the identity it recorded at the run's start (`recordedMemories`), checked as a fresh
 * default is, since a log can be edited. An Optional input left out at the start gets none.
 * Nothing is created here, and the resume opens each memory without creating it.
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
): Record<string, FileMemory> {
  const recorded = recordedMemories(flow, log);
  return bindEach(flow, flags, options, (input) => {
    const identity = recorded.get(input);
    // An Optional memory left out at the start takes none again.
    if (identity === undefined) return null;
    return defaultInstance(identity, options);
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
 * Bind each memory input of `flow`: to the folder its `--memory` flag names, else to what
 * `fallback` gives the input, given whether it is Optional (`null` for none).
 */
function bindEach(
  flow: ComposedFlow,
  flags: readonly string[],
  options: BindOptions,
  fallback: (input: string, optional: boolean) => CliFileMemory | null,
): Record<string, CliFileMemory> {
  const paths = memoryFlags(flow.input, flags);
  const bound: Record<string, CliFileMemory> = {};
  const byRoot = new Map<string, CliFileMemory>();
  for (const param of flow.input) {
    const typ = param.contract?.type;
    if (!isMemoryType(typ)) continue;
    const given = paths.get(param.name);
    // As the auth store reads its path: `--memory=NAME=~/x` reaches us unexpanded.
    const path = given === "~" || given?.startsWith("~/") ? options.home + given.slice(1) : given;
    const made =
      path === undefined ? fallback(param.name, typ.nullable) : new CliFileMemory({ root: path }, options.note);
    if (made === null) continue;
    // Nothing is opened yet, so refusing here leaves no trace.
    if (typ.memory !== "FileMemory") {
      throw new RangeError(`\`${param.name}\` is a ${typ.memory}; the command line binds only FileMemory inputs`);
    }
    // Keyed by the real folder, so two spellings of one path share one memory.
    const known = byRoot.get(made.root);
    // Only flagged inputs share: a flag naming another input's default is a slip, not a choice.
    if (known !== undefined && known.isDefault !== made.isDefault) {
      const [flagged, other] = made.isDefault ? [known.inputs[0]!, param.name] : [param.name, known.inputs[0]!];
      throw new UsageError(
        `Invalid value for --memory: \`${flagged}=${paths.get(flagged)}\` points at \`${other}\`'s default memory`,
      );
    }
    const instance = known ?? made;
    byRoot.set(instance.root, instance);
    instance.inputs.push(param.name);
    bound[param.name] = instance;
  }
  return bound;
}

/**
 * Return the instance of the memory `identity` at its default location under `options.home`.
 *
 * @throws As `defaultLocation` throws.
 */
function defaultInstance(identity: string, options: BindOptions): CliFileMemory {
  const root = defaultLocation(options.home, identity, "FileMemory");
  return new CliFileMemory({ root, id: identity }, options.note, { home: options.home, path: identity });
}
