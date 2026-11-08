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
import { basename, join, relative, sep } from "node:path";

import { START_ID } from "../compile/model.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { reprText } from "../diagnostics.ts";
import type { MemoryBase, MemorySession } from "../memory/base.ts";
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
 * Return the default identity of memory input `input` of the flow `flowId` whose file is in the
 * folder `dir` (a real path): `<project key>/<flow id>/<input>`.
 *
 * @throws {RangeError} When the flow has no `id:`, or one that isn't a plain name.
 */
export function defaultIdentity(dir: string, flowId: string | null | undefined, input: string): string {
  if (flowId == null || !PLAIN_ID.test(flowId)) {
    throw new RangeError(
      `give the flow a plain \`id:\` (letters, digits, \`.\`, \`_\`, \`-\`) or bind \`${input}\` with --memory`,
    );
  }
  return `${projectKey(dir)}/${flowId}/${input}`;
}

/**
 * Return the default location of a memory of type `type` with identity `identity`, under `home`:
 * the folder `~/.yian/memory/<identity>` for a FileMemory, the store `<identity>.db` for a
 * VectorDBMemory. It is checked as `checkDefaultLocation` checks it.
 *
 * @throws {RangeError} As `checkDefaultLocation` throws.
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
 *   or a component above the last isn't a folder.
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
    if (stats.isSymbolicLink()) {
      throw new RangeError(`${at} is a symlink; a default memory stays inside ~/.yian/memory`);
    }
    if (i < parts.length - 1 && !stats.isDirectory()) throw new RangeError(`${at} isn't a folder`);
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

/** What `bindMemories` needs beside the flow and the flags. */
export interface BindOptions {
  /** The home folder whose `~/.yian/memory` holds the default memories. */
  home: string;
  /** Shows the user a line, such as the one an instance's `open` gives when it creates a memory. */
  note: (line: string) => void;
}

/**
 * A FileMemory the command line binds: its `open` notes, through `note`, each memory it creates
 * (or adopts), with the inputs bound to it, its identity and its folder.
 */
class CliFileMemory extends FileMemory {
  /** The inputs bound to this instance, in the order they were bound. */
  readonly inputs: string[] = [];
  readonly #note: (line: string) => void;

  constructor(options: FileMemoryOptions, note: (line: string) => void) {
    super(options);
    this.#note = note;
  }

  override async open(o: {
    create: boolean;
    itemType: JsonObject | null;
    signal: AbortSignal;
  }): Promise<MemorySession> {
    const session = await super.open(o);
    if (session.created) {
      this.#note(`memory ${this.inputs.join(", ")}: ${session.info.identity} at ${this.root} (new)`);
    }
    return session;
  }
}

/**
 * Return the `--memory NAME=PATH` flags as each named memory input's path, checked against the
 * flow's inputs `params`. The name is trimmed; the path is everything after the first `=`.
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
    const path = flag.slice(at + 1);
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
 * `--memory` flag names, else, for an input that isn't Optional, its default (`defaultIdentity`
 * at `defaultLocation`). An Optional input left unbound gets none, so it stays null. Inputs given
 * one folder, however its path is spelled, share one instance. Nothing is created here: an
 * instance's `open` creates its memory, and notes it.
 *
 * @throws {UsageError} As `memoryFlags` throws.
 * @throws {RangeError} For an input of a memory type the command line can't bind; for a default,
 *   as `defaultIdentity` and `defaultLocation` throw, and for a flow built in code.
 */
export function bindMemories(
  flow: ComposedFlow,
  flags: readonly string[],
  options: BindOptions,
): Record<string, MemoryBase> {
  return bindEach(flow, flags, options, (input, optional) => (optional ? null : defaultMemory(flow, input, options)));
}

/**
 * Return the instance each memory input of `flow`, the flow its run in `log` recorded, is bound to
 * again for a resume, by input name: the folder its `--memory` flag names, else the default
 * location of the identity it recorded at the run's start (`recordedMemories`), checked as a fresh
 * default is, since a log can be edited. An Optional input left out at the start gets none.
 * Nothing is created here, and the resume opens each memory without creating it.
 *
 * @throws {UsageError} As `memoryFlags` throws.
 * @throws {RangeError} For an input of a memory type the command line can't bind; as
 *   `defaultLocation` throws; when the log's START inputs hold a malformed memory value.
 * @throws Error when the log can't be read.
 */
export function resumeMemories(
  flow: ComposedFlow,
  log: string,
  flags: readonly string[],
  options: BindOptions,
): Record<string, MemoryBase> {
  const recorded = recordedMemories(flow, log);
  return bindEach(flow, flags, options, (input) => {
    const identity = recorded.get(input);
    // An Optional memory left out at the start takes none again.
    if (identity === undefined) return null;
    const root = defaultLocation(options.home, identity, "FileMemory");
    return new CliFileMemory({ root, id: identity }, options.note);
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
): Record<string, MemoryBase> {
  const paths = memoryFlags(flow.input, flags);
  const bound: Record<string, MemoryBase> = {};
  const byRoot = new Map<string, CliFileMemory>();
  for (const param of flow.input) {
    const typ = param.contract?.type;
    if (!isMemoryType(typ)) continue;
    const path = paths.get(param.name);
    const made =
      path === undefined ? fallback(param.name, typ.nullable) : new CliFileMemory({ root: path }, options.note);
    if (made === null) continue;
    // Nothing is opened yet, so refusing here leaves no trace.
    if (typ.memory !== "FileMemory") {
      throw new RangeError(`\`${param.name}\` is a ${typ.memory}; the command line binds only FileMemory inputs`);
    }
    // Keyed by the real folder, so two spellings of one path share one memory.
    const instance = byRoot.get(made.root) ?? made;
    byRoot.set(instance.root, instance);
    instance.inputs.push(param.name);
    bound[param.name] = instance;
  }
  return bound;
}

/**
 * Return the default instance of `flow`'s memory input `input`: its default identity, at its
 * default location under `options.home`.
 *
 * @throws {RangeError} As `defaultIdentity` and `defaultLocation` throw, and for a flow built in code.
 */
function defaultMemory(flow: ComposedFlow, input: string, options: BindOptions): CliFileMemory {
  if (flow.source === null) throw new RangeError(`a flow built in code has no default memories; bind \`${input}\``);
  const identity = defaultIdentity(flow.source.dir, flow.compiled.flowId, input);
  const root = defaultLocation(options.home, identity, "FileMemory");
  return new CliFileMemory({ root, id: identity }, options.note);
}
