/**
 * `yian memory`: list the memories under `~/.yian/memory`, and read or change one, named as
 * `FLOW NAME` (that input's default memory, as `yian run` binds it) or `--at PATH` (a memory's
 * folder). It is a host the user runs, so its own output may name a memory's folder; a change it
 * makes is authored by the user.
 *
 * Knows about: compose (the flow's inputs), memory (`FileMemory`), the type language, the CLI's
 * memory binding (`memory_bind.ts`) and start-up, and the file system.
 */

import { lstatSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { compareCodeUnits, Diagnostic } from "../diagnostics.ts";
import { memoryErrorKind, type MemorySession } from "../memory/base.ts";
import { FileMemory } from "../memory/file.ts";
import { homeDir, isSystemError } from "../platform.ts";
import { isMemoryType } from "../typesys/values.ts";
import { defaultLocation, defaultMemories, memoryRoot, projectFolder } from "./memory_bind.ts";
import { loadFlowFile } from "./startup.ts";
import { type CliIo, type ParsedArgs, printJson, UsageError } from "./utils.ts";

/** A memory `yian memory` acts on: its instance and folder, and, for a default, its identity. */
export interface MemoryTarget {
  readonly memory: FileMemory;
  /** The memory's folder. */
  readonly location: string;
  /** For `FLOW NAME`, the input's default identity; `null` for `--at`. */
  readonly identity: string | null;
}

/**
 * Return the memory `o` names: `flow` and `name` give that input's default memory, at its default
 * location under the home in `io.env`; `at` gives the memory in a folder. Nothing is opened or
 * created. A flow that doesn't load prints its diagnostic and gives it back.
 *
 * @throws {UsageError} When neither or both forms are given, `name` is missing, the flow has no
 *   such memory input or it is Optional (it has no default), or `at` names a `.db` store.
 * @throws {RangeError} When the input's default can't be bound (`defaultMemories`).
 */
export async function resolveTarget(
  o: { flow?: string | undefined; name?: string | undefined; at?: string | undefined },
  io: CliIo,
): Promise<MemoryTarget | Diagnostic> {
  const { flow: flowPath, name, at } = o;
  if (flowPath === undefined && at === undefined) {
    throw new UsageError("Invalid value: give the memory as FLOW NAME or --at PATH");
  }
  if (flowPath !== undefined && at !== undefined) {
    throw new UsageError("Invalid value: give the memory as FLOW NAME or --at PATH, not both");
  }
  if (at !== undefined) {
    if (at.endsWith(".db")) {
      throw new UsageError(
        "Invalid value for --at: a .db store is a VectorDBMemory, which yian memory doesn't open yet",
      );
    }
    const memory = new FileMemory({ root: at });
    return { memory, location: memory.root, identity: null };
  }
  if (name === undefined) throw new UsageError("Missing argument 'NAME'.");
  const flow = await loadFlowFile(flowPath!, io, { engineTrace: false });
  if (flow instanceof Diagnostic) return flow;
  const typ = flow.input.find((param) => param.name === name)?.contract?.type;
  if (!isMemoryType(typ)) throw new UsageError(`Invalid value: the flow has no memory input \`${name}\``);
  const home = homeDir(io.env);
  const defaults = defaultMemories(flow, { home, project: projectFolder(flowPath!) });
  const found = defaults.find((each) => each.input === name);
  if (found === undefined) {
    throw new UsageError(
      `Invalid value: \`${name}\` is Optional, so it has no default memory; name its folder with --at`,
    );
  }
  const location = defaultLocation(home, found.identity, "FileMemory");
  return { memory: new FileMemory({ root: location, id: found.identity }), location, identity: found.identity };
}

/**
 * Open `target`'s memory, never creating it.
 *
 * @throws {UsageError} When no memory is at its folder.
 * @throws What `FileMemory.open` throws otherwise.
 */
export async function openTarget(target: MemoryTarget, signal: AbortSignal): Promise<MemorySession> {
  try {
    return await target.memory.open({ create: false, itemType: null, signal });
  } catch (error) {
    if (memoryErrorKind(error) !== "missing") throw error;
    throw new UsageError(`Invalid value: no memory at ${target.location}`);
  }
}

/** A memory `yian memory list` found: its identity, type and folder. */
interface Listed {
  readonly identity: string;
  readonly type: string;
  readonly location: string;
}

/**
 * Return the memories under the folder `root`, by identity: each folder holding a memory's
 * identity file (`.yian/identity`). The walk follows no symlink and doesn't look inside a memory.
 */
function findMemories(root: string): Listed[] {
  const found: Listed[] = [];
  const walk = (dir: string): void => {
    let names: string[];
    try {
      names = readdirSync(dir);
    } catch (error) {
      // A folder removed, or never made, holds no memory.
      if (isSystemError(error) && (error.code === "ENOENT" || error.code === "ENOTDIR")) return;
      throw error;
    }
    for (const name of names.sort(compareCodeUnits)) {
      const path = join(dir, name);
      if (lstatOrNull(path)?.isDirectory() !== true) continue;
      const identity = storedIdentity(path);
      if (identity === null) walk(path);
      else found.push({ identity, type: "FileMemory", location: path });
    }
  };
  walk(root);
  return found.sort((a, b) => compareCodeUnits(a.identity, b.identity));
}

/** The identity in `<dir>/.yian/identity`, one line, or `null` when there is none. */
function storedIdentity(dir: string): string | null {
  const path = join(dir, ".yian", "identity");
  if (lstatOrNull(path)?.isFile() !== true) return null;
  // A synced or hand-written identity may end in a line break.
  const text = readFileSync(path, "utf8").replace(/\r?\n$/, "");
  return text === "" || /[\r\n]/.test(text) ? null : text;
}

/** `lstatSync`, or `null` when nothing is at `path`. */
function lstatOrNull(path: string): ReturnType<typeof lstatSync> | null {
  return lstatSync(path, { throwIfNoEntry: false }) ?? null;
}

/**
 * `yian memory list`: every memory under `~/.yian/memory`, by identity, as aligned columns of its
 * identity, type and folder after a header (nothing when there is none), or with `--json` one
 * document whose `memories` holds `{identity, type, location}` each.
 */
export async function memoryList({ options }: ParsedArgs, io: CliIo): Promise<number> {
  const memories = findMemories(memoryRoot(homeDir(io.env)));
  if (options["json"] === true) {
    const listed = memories.map(({ identity, type, location }) => ({ identity, type, location }));
    await printJson({ memories: listed }, io.stdout);
    return 0;
  }
  if (memories.length === 0) return 0;
  const table = [["identity", "type", "folder"], ...memories.map((each) => [each.identity, each.type, each.location])];
  const widths = [0, 1].map((column) => Math.max(...table.map((cells) => cells[column]!.length)));
  const lines = table.map((cells) => cells.map((cell, i) => (i < 2 ? cell.padEnd(widths[i]!) : cell)).join("  "));
  await io.stdout.write(`${lines.join("\n")}\n`);
  return 0;
}
