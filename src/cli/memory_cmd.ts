/**
 * `yian memory`: list the memories under `~/.yian/memory`, and read or change one, named as
 * `FLOW NAME` (that input's default memory, as `yian run` binds it) or `--at PATH` (a memory's
 * folder). It is a host the user runs, so its own output may name a memory's folder; a change it
 * makes is authored by the user.
 *
 * Knows about: compose (the flow's inputs), memory (`FileMemory`), the type language, the CLI's
 * memory binding (`memory_bind.ts`) and start-up, and the file system.
 */

import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { lstatSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir, userInfo } from "node:os";
import { basename, join } from "node:path";

import type { Env } from "../defaults.ts";
import { compareCodeUnits, Diagnostic } from "../diagnostics.ts";
import { type Author, type ChangeEntry, memoryErrorKind, type MemorySession } from "../memory/base.ts";
import { FileMemory, type MemberFile, readMembers } from "../memory/file.ts";
import { homeDir, isSystemError } from "../platform.ts";
import type { JsonObject } from "../typesys/json.ts";
import { isMemoryType } from "../typesys/values.ts";
import { defaultLocation, defaultMemories, memoryRoot, projectFolder } from "./memory_bind.ts";
import { diagnosticJson, printStyled } from "./render.ts";
import { loadFlowFile } from "./startup.ts";
import { type CliIo, forTerminal, type ParsedArgs, printJson, UsageError } from "./utils.ts";

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

/** What a command does with the memory it opened; returns the exit code. */
type MemoryBody = (session: MemorySession, target: MemoryTarget) => Promise<number>;

/**
 * Open the memory `command` names (`FLOW NAME`, or `--at`), run `body` on it, and close it. A flow
 * that doesn't load exits 1 after its diagnostic (a document under `--json`); the memory failing
 * prints `memory failed: <message>` on stderr (a `memory.failed` document) and exits 1.
 *
 * @throws {UsageError} As `resolveTarget` and `openTarget` throw.
 */
async function onMemory(command: ParsedArgs, io: CliIo, body: MemoryBody): Promise<number> {
  const { args, options } = command;
  const json = options["json"] === true;
  let session: MemorySession | null = null;
  try {
    const named = { flow: args["flow"] as string | undefined, name: args["name"] as string | undefined };
    const target = await resolveTarget({ ...named, at: options["at"] as string | undefined }, io);
    if (target instanceof Diagnostic) {
      if (json) await printJson({ error: diagnosticJson(target) }, io.stdout);
      return 1;
    }
    session = await openTarget(target, stopOf(io));
    return await body(session, target);
  } catch (error) {
    if (!(error instanceof Error) || error instanceof UsageError) throw error;
    return failed(io, json, { label: "memory failed", code: "memory.failed", message: error.message });
  } finally {
    await session?.close();
  }
}

/** Print `<label>: <message>` on stderr (a `{code, message}` error document under `json`); return 1. */
async function failed(io: CliIo, json: boolean, o: { label: string; code: string; message: string }): Promise<number> {
  await printStyled(io, `${o.label}: ${o.message}`, "red");
  if (json) await printJson({ error: { code: o.code, message: o.message } }, io.stdout);
  return 1;
}

/** The command's stop: `io.stop`, or one that never aborts. */
function stopOf(io: CliIo): AbortSignal {
  return io.stop ?? new AbortController().signal;
}

/** The user making a change from the command line: `USER`, `LOGNAME`, or the account's name. */
function userName(env: Env): string {
  return env("USER") || env("LOGNAME") || userInfo().username;
}

/** The author of a change the command line makes. */
function userAuthor(io: CliIo): Author {
  return { kind: "user", name: userName(io.env) };
}

/** A fresh id for a change the command line makes. */
function changeId(): string {
  return `cli-${randomBytes(8).toString("hex")}`;
}

/** The memory's files at `location`, read with the caps its session records. */
function membersOf(session: MemorySession, location: string): MemberFile[] {
  const { max_depth: maxDepth, max_file_chars: maxFileChars } = session.info.settings;
  return readMembers(location, { maxDepth: Number(maxDepth), maxFileChars: Number(maxFileChars) }).files;
}

/** `count` with its noun, plural unless it is one. */
function counted(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/**
 * `yian memory show`: a line naming the memory (identity, type, folder) and counting its files
 * and characters, then its recall, the part a prompt shows each run, and a note when it was cut.
 * With `--all`, every file too, each after a `== <path> (<n> characters) ==` line. `--json` prints
 * one document with the identity, type, location, version, characters, recall and files (each
 * file's text with `--all`).
 */
export async function memoryShow(command: ParsedArgs, io: CliIo): Promise<number> {
  const all = command.options["all"] === true;
  return onMemory(command, io, async (session, { location }) => {
    const { version, recall } = await session.snapshot();
    const files = membersOf(session, location);
    const characters = files.reduce((sum, file) => sum + file.chars, 0);
    const { identity, type } = session.info;
    if (command.options["json"] === true) {
      const listed = files.map((file) => ({
        path: file.path,
        characters: BigInt(file.chars),
        hash: file.hash,
        ...(all ? { text: file.text } : {}),
      }));
      const document = { identity, type, location, version, characters: BigInt(characters) };
      await printJson({ ...document, recall: { text: recall.text, cut: recall.cut }, files: listed }, io.stdout);
      return 0;
    }
    const counts = `${counted(files.length, "file")}, ${counted(characters, "character")}`;
    const lines = [`${identity} (${type}) at ${location}: ${counts}`, recall.text];
    if (recall.cut !== null) lines.push(`(${recall.cut})`);
    if (all) {
      for (const file of files) lines.push(`== ${file.path} (${counted(file.chars, "character")}) ==`, file.text);
    }
    await io.stdout.write(`${forTerminal(lines.join("\n"), io.stdout)}\n`);
    return 0;
  });
}

/** `author` as history shows it: its kind and name, and its run and node when it has them. */
function authorText(author: Author): string {
  const shown: string[] = [];
  if (author.run !== undefined) shown.push(`run ${author.run}`);
  if (author.node !== undefined) shown.push(`node ${author.node}`);
  return `${author.kind} ${author.name}${shown.length === 0 ? "" : ` (${shown.join(", ")})`}`;
}

/** `change` as one line of history: its id, time, op and path, the change it undoes, and its author. */
function changeLine(change: ChangeEntry): string {
  const { path, undoes } = change.target;
  const undoing = typeof undoes === "string" ? ` (undoes ${undoes})` : "";
  return `${change.id}  ${change.at}  ${change.op} ${String(path)}${undoing} by ${authorText(change.author)}`;
}

/**
 * `yian memory history`: the memory's last changes (`--limit`, 20 by default), newest first, one
 * line each with its id, time, op, path and author (with the run and node that made it). `--json`
 * prints one document whose `changes` holds each change's `id`, `op`, `target`, `author` and `at`.
 */
export async function memoryHistory(command: ParsedArgs, io: CliIo): Promise<number> {
  const limit = (command.options["limit"] as number | undefined) ?? 20;
  return onMemory(command, io, async (session) => {
    const changes = await session.history({ limit });
    if (command.options["json"] === true) {
      const listed = changes.map(({ id, op, target, author, at }) => ({ id, op, target, author, at }));
      await printJson({ changes: listed }, io.stdout);
      return 0;
    }
    if (changes.length > 0) await io.stdout.write(`${forTerminal(changes.map(changeLine).join("\n"), io.stdout)}\n`);
    return 0;
  });
}

/**
 * `yian memory undo CHANGE`: undo the change `CHANGE` as the user, and print the memory's new
 * version (`--json`: `{change, version}`, `change` the undo's own id). A change that can't be
 * undone (its file changed since, or it is no longer kept) prints `undo refused: <reason>` (a
 * `memory.conflict` document) and exits 1.
 */
export async function memoryUndo(command: ParsedArgs, io: CliIo): Promise<number> {
  const undone = command.args["change"] as string;
  const json = command.options["json"] === true;
  return onMemory(command, io, async (session) => {
    const id = changeId();
    const applied = await session.undo({ id, changeId: undone, author: userAuthor(io), signal: stopOf(io) });
    if (!applied.ok) {
      return failed(io, json, { label: "undo refused", code: "memory.conflict", message: applied.reason });
    }
    if (json) await printJson({ change: id, version: applied.version }, io.stdout);
    else await io.stdout.write(`undid ${forTerminal(undone, io.stdout)}; the memory's version is ${applied.version}\n`);
    return 0;
  });
}

/**
 * Run `$EDITOR` from `env` on the file `path`, the terminal its own, and resolve once it exits. The
 * variable may hold arguments (`code --wait`), so the shell runs it, the path passed as an argument.
 *
 * @throws {RangeError} When `$EDITOR` isn't set, or the editor fails or exits with an error.
 */
function runEditor(env: Env, path: string): Promise<void> {
  const editor = env("EDITOR");
  if (!editor) throw new RangeError("set $EDITOR to the editor to use");
  return new Promise((resolve, reject) => {
    const child = spawn("/bin/sh", ["-c", `${editor} "$1"`, "editor", path], { stdio: "inherit" });
    child.on("error", (error) => reject(new RangeError(`the editor didn't start: ${error.message}`)));
    child.on("exit", (code, signal) => {
      if (code === 0) resolve();
      else reject(new RangeError(`the editor exited with ${signal ?? `code ${code}`}; nothing was changed`));
    });
  });
}

/**
 * `yian memory edit`: open a file of the memory (`--file`, the memory's index by default) in an
 * editor (`$EDITOR`), on a private copy, then apply what the person wrote as the user: a `rewrite`
 * of the file, checked against the hash it had when it was read, or a `create` for a new file. An
 * unchanged file changes nothing. A file changed meanwhile prints `edit refused: <reason>` (a
 * `memory.conflict` document) and the copy that keeps the edited text, and exits 1. `--json`
 * prints `{change, version}` for an edit, or `{change: null}` for none.
 */
export async function memoryEdit(command: ParsedArgs, io: CliIo): Promise<number> {
  const json = command.options["json"] === true;
  return onMemory(command, io, async (session, { location }) => {
    const path = (command.options["file"] as string | undefined) ?? String(session.info.settings["index"]);
    const file = membersOf(session, location).find((each) => each.path === path) ?? null;
    // Private: the copy holds the memory's text.
    const folder = mkdtempSync(join(tmpdir(), "yian-memory-edit-"));
    const copy = join(folder, basename(path));
    let keep = false;
    try {
      writeFileSync(copy, file?.text ?? "", { mode: 0o600 });
      await (io.editor ?? ((each: string) => runEditor(io.env, each)))(copy);
      const text = readFileSync(copy, "utf8");
      if (text === (file?.text ?? "")) {
        if (json) await printJson({ change: null }, io.stdout);
        else await io.stdout.write(`${forTerminal(path, io.stdout)} is unchanged\n`);
        return 0;
      }
      const change: JsonObject =
        file === null ? { op: "create", path, text } : { op: "rewrite", path, text, if_hash: file.hash };
      const id = changeId();
      const applied = await session.apply({ id, change, author: userAuthor(io), signal: stopOf(io) });
      if (!applied.ok) {
        keep = true;
        const message = `${applied.reason}; your text is in ${copy}`;
        return failed(io, json, { label: "edit refused", code: "memory.conflict", message });
      }
      if (json) await printJson({ change: id, version: applied.version }, io.stdout);
      else {
        await io.stdout.write(`edited ${forTerminal(path, io.stdout)}; the memory's version is ${applied.version}\n`);
      }
      return 0;
    } finally {
      if (!keep) rmSync(folder, { recursive: true, force: true });
    }
  });
}
