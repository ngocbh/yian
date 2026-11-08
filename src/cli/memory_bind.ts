/**
 * How the command line binds a flow's memory inputs to instances: `--memory NAME=PATH` binds one
 * to a folder, and an unbound one gets its default identity, `<project key>/<flow id>/<input>`,
 * at its location under `~/.yian/memory`, confined there. The instances are `CliFileMemory`s,
 * which say when their `open` created the memory.
 *
 * Knows about: compose (the flow's inputs), memory (`FileMemory`), the type language, the
 * platform's path helpers and the file system.
 */

import { createHash } from "node:crypto";
import { lstatSync, type Stats } from "node:fs";
import { basename, join, relative, sep } from "node:path";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { reprText } from "../diagnostics.ts";
import type { MemoryBase, MemorySession } from "../memory/base.ts";
import { FileMemory, type FileMemoryOptions } from "../memory/file.ts";
import type { Parameter } from "../nodes/contracts.ts";
import type { JsonObject } from "../typesys/json.ts";
import { isMemoryType } from "../typesys/values.ts";
import { isSystemError, leavesRoot } from "../platform.ts";
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
    if (stats.isSymbolicLink()) throw new RangeError(`${at} is a symlink; a default memory stays inside ~/.yian/memory`);
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

  override async open(o: { create: boolean; itemType: JsonObject | null; signal: AbortSignal }): Promise<MemorySession> {
    const session = await super.open(o);
    if (session.created) this.#note(`memory ${this.inputs.join(", ")}: ${session.info.identity} at ${this.root} (new)`);
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
export function bindMemories(flow: ComposedFlow, flags: readonly string[], options: BindOptions): Record<string, MemoryBase> {
  const paths = memoryFlags(flow.input, flags);
  const bound: Record<string, MemoryBase> = {};
  const byRoot = new Map<string, CliFileMemory>();
  for (const param of flow.input) {
    const typ = param.contract?.type;
    const path = paths.get(param.name);
    if (!isMemoryType(typ) || (path === undefined && typ.nullable)) continue;
    if (typ.memory !== "FileMemory") {
      throw new RangeError(`\`${param.name}\` is a ${typ.memory}; the command line binds only FileMemory inputs`);
    }
    const made =
      path === undefined
        ? defaultMemory(flow, param.name, options)
        : new CliFileMemory({ root: path }, options.note);
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
  return new CliFileMemory({ root: defaultLocation(options.home, identity, "FileMemory"), id: identity }, options.note);
}
