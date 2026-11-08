/**
 * `yian memory`: list the memories under `~/.yian/memory`, and read or change one, named as
 * `FLOW NAME` (that input's default memory, as `yian run` binds it) or `--at PATH` (a memory's
 * folder). It is a host the user runs, so its own output may name a memory's folder; a change it
 * makes is authored by the user.
 *
 * Knows about: compose (the flow's inputs), memory (`FileMemory`), the run index (the runs that used
 * a memory), the type language, the CLI's memory binding (`memory_bind.ts`), start-up and
 * terminal, and the file system.
 */

import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, lstatSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir, userInfo } from "node:os";
import { basename, join } from "node:path";

import { type Env, processEnvNames } from "../defaults.ts";
import { compareCodeUnits, Diagnostic } from "../diagnostics.ts";
import { type Applied, type Author, type ChangeEntry, memoryErrorKind, type MemorySession } from "../memory/base.ts";
import { FileMemory, type MemberFile, readMembers } from "../memory/file.ts";
import { homeDir, isSystemError, withWarnings } from "../platform.ts";
import { readRunIndex, type RunIndexRow, updateRunIndex } from "../run.ts";
import type { JsonObject } from "../typesys/json.ts";
import { isMemoryType } from "../typesys/values.ts";
import { defaultLocation, defaultMemories, memoryRoot, projectFolder } from "./memory_bind.ts";
import { diagnosticJson, printStyled } from "./render.ts";
import { defaultRunsDir, loadFlowFile } from "./startup.ts";
import { Terminal } from "./terminal.ts";
import { atTerminal, checkPath, type CliIo, forTerminal, type ParsedArgs, printJson, UsageError } from "./utils.ts";

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

/** A memory named as `FLOW NAME`; both absent with `--at`. */
interface Named {
  readonly flow?: string | undefined;
  readonly name?: string | undefined;
}

/** The memory `command` names through its `flow` and `name` positionals. */
function namedOf(command: ParsedArgs): Named {
  return { flow: command.args["flow"] as string | undefined, name: command.args["name"] as string | undefined };
}

/**
 * Return the memory a command whose own argument comes last names, and that argument: its
 * positionals are `FLOW NAME <argument>`, or with `--at` the argument alone. The parser fills the
 * positionals `flow`, `name` and `argument` in order, so they are read as one list.
 *
 * @param argument The argument's positional, such as `change`.
 * @throws {UsageError} For another count of positionals, showing both forms, or a FLOW that isn't
 *   a readable file.
 */
function namedWith(command: ParsedArgs, argument: string): { named: Named; value: string } {
  const given = ["flow", "name", argument]
    .map((key) => command.args[key])
    .filter((value): value is string => typeof value === "string");
  const at = command.options["at"] !== undefined;
  if (given.length !== (at ? 1 : 3)) {
    const shown = argument.toUpperCase();
    throw new UsageError(`Invalid value: give FLOW NAME ${shown}, or --at PATH ${shown}`);
  }
  if (at) return { named: {}, value: given[0]! };
  checkPath("'FLOW'", given[0]!, { exists: true, dir: false, readable: true });
  return { named: { flow: given[0], name: given[1] }, value: given[2]! };
}

/**
 * Open the memory `named` (or `command`'s `--at`) names, run `body` on it, and close it. A flow
 * that doesn't load exits 1 after its diagnostic (a document under `--json`); the memory failing
 * prints `memory failed: <message>` on stderr (a `memory.failed` document) and exits 1.
 *
 * @throws {UsageError} As `resolveTarget` and `openTarget` throw.
 */
async function onMemory(command: ParsedArgs, io: CliIo, named: Named, body: MemoryBody): Promise<number> {
  const { options } = command;
  const json = options["json"] === true;
  let session: MemorySession | null = null;
  try {
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
  return onMemory(command, io, namedOf(command), async (session, { location }) => {
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
  return onMemory(command, io, namedOf(command), async (session) => {
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
 * `yian memory undo FLOW NAME CHANGE` (or `--at PATH CHANGE`): undo the change `CHANGE` as the
 * user, and print the memory's new version (`--json`: `{change, version}`, `change` the undo's own
 * id). A change that can't be undone (its file changed since, or it is no longer kept) prints
 * `undo refused: <reason>` (a `memory.conflict` document) and exits 1.
 */
export async function memoryUndo(command: ParsedArgs, io: CliIo): Promise<number> {
  const { named, value: undone } = namedWith(command, "change");
  const json = command.options["json"] === true;
  return onMemory(command, io, named, async (session) => {
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
 * The environment a child of the command gets: each variable as `env` reads it, over the process's
 * names and those an editor and its shell look for, so a command's own environment reaches it.
 */
function childEnv(env: Env): Record<string, string> {
  const names = new Set([...processEnvNames(), "HOME", "PATH", "TMPDIR", "TERM", "EDITOR", "USER", "LANG"]);
  const values: Record<string, string> = {};
  for (const name of names) {
    const value = env(name);
    if (value !== undefined) values[name] = value;
  }
  return values;
}

/** The signals a terminal sends its foreground group that are the editor's while it runs. */
const EDITOR_SIGNALS = ["SIGINT", "SIGQUIT"] as const;

/**
 * Run `$EDITOR` from `env` on the file `path`, the terminal its own and `env` its environment, and
 * resolve once it exits. The variable may hold arguments (`code --wait`), so the shell runs it,
 * the path passed as an argument. Meanwhile yian ignores Ctrl-C and Ctrl-\, as git does: the
 * terminal sends them to the editor too, which handles them, and yian must outlive it to apply or
 * keep the edit. With `json`, the editor's stdout is yian's stderr, so stdout holds only the
 * command's document.
 *
 * @throws {RangeError} When `$EDITOR` isn't set, or the editor fails or exits with an error.
 */
async function runEditor(env: Env, path: string, json: boolean): Promise<void> {
  const editor = env("EDITOR");
  if (!editor) throw new RangeError("set $EDITOR to the editor to use");
  const ignore = (): void => {};
  for (const signal of EDITOR_SIGNALS) process.on(signal, ignore);
  try {
    await new Promise<void>((resolve, reject) => {
      const command = ["-c", `${editor} "$1"`, "editor", path];
      const stdio = ["inherit", json ? 2 : "inherit", "inherit"] as const;
      const child = spawn("/bin/sh", command, { stdio: [...stdio], env: childEnv(env) });
      child.on("error", (error) => reject(new RangeError(`the editor didn't start: ${error.message}`)));
      child.on("exit", (code, signal) => {
        if (code === 0) resolve();
        else reject(new RangeError(`the editor exited with ${signal ?? `code ${code}`}; nothing was changed`));
      });
    });
  } finally {
    for (const signal of EDITOR_SIGNALS) process.off(signal, ignore);
  }
}

/**
 * `yian memory edit`: open a file of the memory (`--file`, the memory's index by default) in an
 * editor (`$EDITOR`), on a private copy, then apply what the person wrote as the user: a `rewrite`
 * of the file, checked against the hash it had when it was read, or a `create` for a new file. An
 * unchanged file changes nothing. Edited text that isn't applied is never lost: the copy stays and
 * the message says where. A file changed meanwhile prints `edit refused: <reason>` (a
 * `memory.conflict` document); a failure of the editor or the change prints `edit failed:
 * <message>` (`memory.failed`); either exits 1. `--json` prints `{change, version}` for an edit,
 * or `{change: null}` for none.
 */
export async function memoryEdit(command: ParsedArgs, io: CliIo): Promise<number> {
  const json = command.options["json"] === true;
  return onMemory(command, io, namedOf(command), async (session, { location }) => {
    const path = (command.options["file"] as string | undefined) ?? String(session.info.settings["index"]);
    const file = membersOf(session, location).find((each) => each.path === path) ?? null;
    const before = file?.text ?? "";
    // Private: the copy holds the memory's text. In the command's TMPDIR, as the editor would look.
    const folder = mkdtempSync(join(io.env("TMPDIR") || tmpdir(), "yian-memory-edit-"));
    const copy = join(folder, basename(path));
    let keep = false;
    // Once the person has typed, the copy holds their text until the change applies.
    const keptFailure = (label: string, code: string, message: string) => {
      keep = true;
      return failed(io, json, { label, code, message: `${message}; your text is in ${copy}` });
    };
    try {
      writeFileSync(copy, before, { mode: 0o600 });
      try {
        await (io.editor ?? ((each: string) => runEditor(io.env, each, json)))(copy);
      } catch (error) {
        if (!(error instanceof Error) || readFileSync(copy, "utf8") === before) throw error;
        return keptFailure("edit failed", "memory.failed", error.message);
      }
      const text = readFileSync(copy, "utf8");
      if (text === before) {
        if (json) await printJson({ change: null }, io.stdout);
        else await io.stdout.write(`${forTerminal(path, io.stdout)} is unchanged\n`);
        return 0;
      }
      const change: JsonObject =
        file === null ? { op: "create", path, text } : { op: "rewrite", path, text, if_hash: file.hash };
      const id = changeId();
      let applied: Applied;
      try {
        applied = await session.apply({ id, change, author: userAuthor(io), signal: stopOf(io) });
      } catch (error) {
        if (!(error instanceof Error)) throw error;
        return keptFailure("edit failed", "memory.failed", error.message);
      }
      if (!applied.ok) return keptFailure("edit refused", "memory.conflict", applied.reason);
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

/**
 * Return the runs in `~/.yian/runs` whose index rows record the memory `identity`, by start time,
 * the index updated first (a log it can't read is skipped quietly; `yian status` reports it).
 *
 * @throws What `updateRunIndex` and `readRunIndex` throw.
 */
function runsUsing(io: CliIo, identity: string): RunIndexRow[] {
  const runsDir = defaultRunsDir(io.env);
  if (!existsSync(runsDir)) return [];
  withWarnings(() => {}, () => updateRunIndex(runsDir, { timeoutMs: 1000 }));
  return readRunIndex(runsDir).filter((row) => row.memories.some((memory) => memory.identity === identity));
}

/** `runs` as `--json` lists them: each run's id, status, start time and log. */
function runsJson(io: CliIo, runs: readonly RunIndexRow[]): Record<string, unknown>[] {
  const runsDir = defaultRunsDir(io.env);
  return runs.map((row) => ({
    run_id: row.runId,
    status: row.status,
    started_at: row.startedAt,
    log: join(runsDir, row.runId, "log.jsonl"),
  }));
}

/** Print `runs` under `heading`, one line each with its id, status, start time and log. */
async function printRuns(io: CliIo, heading: string, runs: readonly RunIndexRow[]): Promise<void> {
  const lines = [heading, ...runsJson(io, runs).map((run) => `  ${Object.values(run).map(String).join("  ")}`)];
  await io.stdout.write(`${forTerminal(lines.join("\n"), io.stdout)}\n`);
}

/**
 * Refuse, before anything is shown, a command that asks first when it runs off a terminal without
 * `--yes`.
 *
 * @throws {UsageError} Then.
 */
function checkAsks(command: ParsedArgs, io: CliIo, verb: string): void {
  if (command.options["yes"] !== true && !atTerminal(io)) {
    throw new UsageError(`Invalid value: ${verb} asks first; off a terminal, pass --yes`);
  }
}

/** Whether the person goes on: `--yes`, or `Yes` to `question` at the terminal (`No` first). */
async function goesOn(command: ParsedArgs, io: CliIo, question: string): Promise<boolean> {
  if (command.options["yes"] === true) return true;
  const terminal = new Terminal({ input: io.stdin, output: io.stderrStream });
  try {
    return (await terminal.askChoice(question, ["No", "Yes"], { signal: stopOf(io) })) === "Yes";
  } finally {
    terminal.close();
  }
}

/**
 * `yian memory erase FLOW NAME FILE` (or `--at PATH FILE`): list the runs on this machine whose
 * index records the memory (their logs may hold copies of what is erased, and deleting them is the
 * person's choice), say that runs elsewhere can't be seen, ask (`--yes` doesn't; off a terminal it
 * is required), then erase the file, or with `--text` every occurrence of that text in it, from
 * the memory and its history, as the user, and print the new version and the undos the erase
 * disabled. A no prints `nothing erased` and exits 1. `--json` prints `{runs, version,
 * disabled_undos}`.
 */
export async function memoryErase(command: ParsedArgs, io: CliIo): Promise<number> {
  const json = command.options["json"] === true;
  const { named, value: path } = namedWith(command, "file");
  const text = command.options["text"] as string | undefined;
  checkAsks(command, io, "erase");
  return onMemory(command, io, named, async (session) => {
    const { identity } = session.info;
    const runs = runsUsing(io, identity);
    const unseen = "yian can't see runs on other machines or in a host's own logs";
    if (!json) {
      if (runs.length === 0) await io.stdout.write(`no run on this machine used ${identity}; ${unseen}\n`);
      else {
        await printRuns(io, `the runs on this machine that used ${identity}; their logs may hold copies:`, runs);
        await io.stdout.write(`${unseen}\n`);
      }
    }
    const what = text === undefined ? path : `${JSON.stringify(text)} from ${path}`;
    if (!(await goesOn(command, io, `Erase ${what} from ${identity}? It can't be undone.`))) {
      return failed(io, json, { label: "erase cancelled", code: "memory.cancelled", message: "nothing erased" });
    }
    const target: JsonObject = text === undefined ? { path } : { path, text };
    const erased = await session.erase({ id: changeId(), target, author: userAuthor(io), signal: stopOf(io) });
    const disabled = erased.disabledUndos;
    if (json) {
      await printJson({ runs: runsJson(io, runs), version: erased.version, disabled_undos: disabled }, io.stdout);
      return 0;
    }
    const undos = disabled.length === 0 ? "it disabled no undo" : `undos it disabled: ${disabled.join(", ")}`;
    const lines = [`erased ${what}; the memory's version is ${erased.version}`, undos];
    await io.stdout.write(`${forTerminal(lines.join("\n"), io.stdout)}\n`);
    return 0;
  });
}

/** The index statuses of a run that hasn't ended, whose resume would go on with the memory. */
const UNFINISHED: ReadonlySet<string> = new Set(["running", "waiting", "stopped", "crashed"]);

/**
 * `yian memory reset`: list the unfinished runs on this machine whose index records the memory
 * (running, waiting, stopped or crashed: a resume goes on with the emptied memory), ask (`--yes`
 * doesn't, and off a terminal it is required), then empty it: its files and history go, its
 * identity stays. A no prints `nothing reset` and exits 1. `--json` prints `{identity, runs}`.
 */
export async function memoryReset(command: ParsedArgs, io: CliIo): Promise<number> {
  const json = command.options["json"] === true;
  checkAsks(command, io, "reset");
  return onMemory(command, io, namedOf(command), async (session, target) => {
    const { identity } = session.info;
    const runs = runsUsing(io, identity).filter((row) => UNFINISHED.has(row.status));
    if (!json) {
      if (runs.length === 0) await io.stdout.write(`no unfinished run on this machine uses ${identity}\n`);
      else {
        const heading = `the unfinished runs on this machine that use ${identity}`;
        await printRuns(io, `${heading}; a resume goes on with the emptied memory:`, runs);
      }
    }
    if (!(await goesOn(command, io, `Empty ${identity}? Its files and history go.`))) {
      return failed(io, json, { label: "reset cancelled", code: "memory.cancelled", message: "nothing reset" });
    }
    await target.memory.reset({ signal: stopOf(io) });
    if (json) await printJson({ identity, runs: runsJson(io, runs) }, io.stdout);
    else await io.stdout.write(`emptied ${forTerminal(identity, io.stdout)}\n`);
    return 0;
  });
}
