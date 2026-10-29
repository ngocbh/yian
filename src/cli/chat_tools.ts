/**
 * The host tools `yian chat`'s assistant works on flow files with (`chatFileTools`), each confined
 * to the folder the chat started in (the root): a path that leaves it, through `..`, an absolute
 * path or a symlink, is refused. The model always gets text back: a tool returns its result, or
 * throws a `ReportedToolError` (a refusal, a bad argument, a failed read) that the model reads as
 * the call's result, and `write_flow` returns every outcome once it has asked the user.
 *
 * The run tools (`chatRunTools`) start flows in the chat's folder as child runs, once the user
 * approves each on the terminal, and read, wait on, answer, message and stop them.
 *
 * Knows about: the file system, the host tool API and its errors, confined loading, the
 * terminal's approval prompt, the run boundary's flow readers and the chat's child runs.
 */

import {
  closeSync,
  constants,
  type Dirent,
  fstatSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  readSync,
  realpathSync,
  renameSync,
  type Stats,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { basename, dirname, extname, join, relative, resolve, sep } from "node:path";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import type { Env } from "../defaults.ts";
import { compareCodeUnits, type Diagnostic, reprText } from "../diagnostics.ts";
import { coerceInputs } from "../nodes/binding.ts";
import { splitReference } from "../nodes/code/worker.ts";
import { isSystemError, leavesRoot, realScheduler, type Scheduler } from "../platform.ts";
import { hostInputs, skillGrants, staticNodes } from "../run.ts";
import { resolvedPath } from "../skills/catalog.ts";
import { isReportedToolError, ReportedToolError } from "../tools/errors.ts";
import { type FunctionTool, tool } from "../tools/function_tools.ts";
import { buildValueAs, canonicalTypeLabel, isInboxType } from "../typesys/values.ts";
import type { ChildRuns, ChildStatus } from "./chat_runs.ts";
import { loadConfinedText } from "./confined_load.ts";
import { unifiedDiff } from "./diff.ts";
import { renderDiagnostic } from "./render.ts";
import { decodeFlowText, NotUtf8Error, readFlowText } from "./startup.ts";
import type { ApproveRequest } from "./terminal.ts";
import { osReason } from "./utils.ts";

/** The most flows `list_flows` lists. */
const LIST_FLOWS_LIMIT = 500;

/** The most files and folders `list_flows` looks at. */
const LIST_FLOWS_ENTRIES = 50_000;

/** The largest flow file `read_flow` reads, in bytes. */
const READ_FLOW_BYTES = 256 * 1024;

/** The most of a focus flow's text `checkFocus` gives, in bytes. */
const FOCUS_BYTES = 32 * 1024;

/** What the chat's file tools work with. */
export interface ChatToolHost {
  /** The folder the chat started in; every path is confined to it. */
  root: string;
  /** Ask the user to approve a write, as `Terminal.approve` does. */
  approve: (request: ApproveRequest) => Promise<string | null>;
  /** The environment a flow loads with. */
  env: Env;
  /** The docs pages `read_docs` serves, by name. */
  docs: ReadonlyMap<string, string>;
}

/** A C0 or C1 control character, or DEL. */
const CONTROL = /[\x00-\x1f\x7f-\x9f]/;

/** A tool refused a call, for a path outside the root say; the model reads the message. */
export class ToolRefused extends ReportedToolError {
  override name = "ToolRefused";
}

/**
 * Return the real path `path` (relative to `root`, or absolute) names, which must be inside
 * `root`. A path that doesn't exist yet resolves through its nearest existing folder.
 *
 * @throws {ToolRefused} When the path has a control character, leaves `root`, goes through a
 *   file, or is or goes through a symlink whose target doesn't exist.
 * @throws {ReportedToolError} When the path can't be resolved, through a symlink loop say.
 */
export function confine(root: string, path: string): string {
  // A NUL makes the file system calls throw, and any control character could forge what the
  // approval card shows.
  if (CONTROL.test(path)) throw new ToolRefused(`${reprText(path)} has a control character, which a path can't`);
  const realRoot = resolveRoot(root);
  const target = resolve(realRoot, path);
  let real: string;
  try {
    real = realpathSync(target);
  } catch (error) {
    if (!isSystemError(error)) throw error;
    if (error.code === "ENOTDIR") throw new ToolRefused(`${path} goes through a file, not a folder`);
    if (error.code !== "ENOENT") throw new ReportedToolError(`can't resolve ${path}: ${osReason(error)}`);
    refuseDanglingLink(path, target);
    real = resolvedPath(target);
  }
  if (leavesRoot(relative(realRoot, real))) {
    throw new ToolRefused(`${path} is outside the folder the chat started in`);
  }
  return real;
}

/**
 * The real path of the chat's folder `root`.
 *
 * @throws {ReportedToolError} When it can't be resolved, removed since the chat started say.
 */
function resolveRoot(root: string): string {
  try {
    return realpathSync(root);
  } catch (error) {
    if (!isSystemError(error)) throw error;
    throw new ReportedToolError(`the chat's folder ${root} can't be used: ${osReason(error)}`);
  }
}

/**
 * The string argument `name` of a tool call.
 *
 * @throws {ReportedToolError} When it isn't a string.
 */
function stringArg(args: Record<string, unknown>, name: string): string {
  const value = args[name];
  if (typeof value !== "string") throw new ReportedToolError(`${name} must be a string`);
  return value;
}

/**
 * The optional string argument `name` of a tool call, undefined when not given.
 *
 * @throws {ReportedToolError} When it is given but isn't a string.
 */
function optionalStringArg(args: Record<string, unknown>, name: string): string | undefined {
  return args[name] === undefined ? undefined : stringArg(args, name);
}

/**
 * Refuse `path` (at `target`, which doesn't fully exist) when the nearest part of it that exists is
 * a symlink to something missing: creating the path would create that, wherever it is.
 */
function refuseDanglingLink(path: string, target: string): void {
  for (let at = target; at !== dirname(at); at = dirname(at)) {
    const stats = lstatOrNull(at);
    if (stats === null) continue;
    if (!stats.isSymbolicLink()) return;
    try {
      realpathSync(at);
      return;
    } catch (error) {
      if (!isSystemError(error)) throw error;
    }
    if (at === target) throw new ToolRefused(`${path} is a symlink to a missing file`);
    throw new ToolRefused(`${path} goes through a symlink to a missing folder`);
  }
}

/** `path`'s own status, a symlink not followed, or null when it doesn't exist. */
function lstatOrNull(path: string): Stats | null {
  try {
    return lstatSync(path);
  } catch (error) {
    if (isSystemError(error)) return null;
    throw error;
  }
}

/** Return `path` written with `/` between its parts, as the model sees it on every platform. */
function slashed(path: string): string {
  return path.split(sep).join("/");
}

/**
 * The error the model reads for a file `path` that can't be read: why, never what it holds.
 *
 * @throws `error` itself when it is neither a system error nor a `NotUtf8Error`.
 */
function toReadFailure(path: string, error: unknown): ReportedToolError {
  if (error instanceof NotUtf8Error) return new ReportedToolError(`can't read ${path}: not UTF-8 text`);
  if (isSystemError(error)) return new ReportedToolError(`can't read ${path}: ${osReason(error)}`);
  throw error;
}

/** Whether `path` names a flow file by its extension. */
function isFlowFile(path: string): boolean {
  const extension = extname(path);
  return extension === ".yaml" || extension === ".yml";
}

/**
 * @internal The flow files under `dir`, relative to `root` and sorted, skipping `node_modules`
 * and dot-folders and not following symlinks: the first `limits.flows` of them in that order,
 * then a note when there were more. The walk stops, with a note, after `limits.entries` files and
 * folders, so a huge tree stays quick.
 */
export function listFlows(
  root: string,
  dir: string,
  limits: { flows: number; entries: number } = { flows: LIST_FLOWS_LIMIT, entries: LIST_FLOWS_ENTRIES },
): string {
  const found: string[] = [];
  let visited = 0;
  // Set only where the walk leaves an entry unvisited, so a tree of exactly the limit is whole.
  let truncated = false;
  const walk = (folder: string): void => {
    let entries: Dirent[];
    try {
      entries = readdirSync(folder, { withFileTypes: true });
    } catch (error) {
      // A folder inside that can't be read (no permission, say) leaves only its flows out.
      if (folder !== dir && isSystemError(error)) return;
      throw toReadFailure(slashed(relative(root, folder)) || ".", error);
    }
    entries.sort((a, b) => compareCodeUnits(a.name, b.name));
    for (const entry of entries) {
      if (visited === limits.entries) {
        truncated = true;
        return;
      }
      visited++;
      const path = join(folder, entry.name);
      if (entry.isDirectory()) {
        if (entry.name !== "node_modules" && !entry.name.startsWith(".")) walk(path);
      } else if (entry.isFile() && isFlowFile(entry.name)) {
        found.push(slashed(relative(root, path)));
      }
    }
  };
  walk(dir);
  if (found.length === 0 && !truncated) return "no flows";
  found.sort(compareCodeUnits);
  const lines = found.slice(0, limits.flows);
  if (found.length > limits.flows) {
    lines.push(`(${found.length - limits.flows} more not shown; pass dir to list one folder)`);
  }
  if (truncated) {
    lines.push(`(stopped after ${limits.entries} files and folders; pass dir to list one folder)`);
  }
  return lines.join("\n");
}

/**
 * The file at `real` (`path` as the model gave it): its status and its bytes, read through one
 * descriptor so both are of the same file, and never more than `READ_FLOW_BYTES` of them.
 *
 * @throws {ToolRefused} When it isn't a file or is larger than `READ_FLOW_BYTES`.
 * @throws {Error} A system error when it can't be opened or read.
 */
function readCapped(path: string, real: string): { bytes: Buffer; stats: Stats } {
  // Non-blocking, so a FIFO named like a flow can't hang the open.
  const fd = openSync(real, constants.O_RDONLY | (constants.O_NONBLOCK ?? 0));
  try {
    const stats = fstatSync(fd);
    if (!stats.isFile()) throw new ToolRefused(`${path} is not a file`);
    const buffer = Buffer.alloc(READ_FLOW_BYTES + 1);
    let length = 0;
    for (let read = -1; read !== 0 && length < buffer.length; length += read) {
      read = readSync(fd, buffer, length, buffer.length - length, null);
    }
    if (length > READ_FLOW_BYTES) {
      throw new ToolRefused(`${path} is over ${READ_FLOW_BYTES} bytes, the most the chat's file tools read`);
    }
    return { bytes: buffer.subarray(0, length), stats };
  } finally {
    closeSync(fd);
  }
}

/**
 * The text of the flow file `path` (as the model gave it) at `real`, refusing anything but a YAML
 * file of at most `READ_FLOW_BYTES`.
 */
function readFlow(path: string, real: string): string {
  if (!isFlowFile(real)) throw new ToolRefused(`${path} is not a .yaml or .yml file`);
  try {
    return decodeFlowText(readCapped(path, real).bytes, path);
  } catch (error) {
    throw error instanceof ToolRefused ? error : toReadFailure(path, error);
  }
}

/** `diag` as `yian check` writes it, without colour. */
async function renderedDiagnostic(diag: Diagnostic, env: Env): Promise<string> {
  let text = "";
  await renderDiagnostic(diag, { write: async (chunk) => void (text += chunk), isTTY: false }, env);
  return text.trimEnd();
}

/**
 * Check `text` as the flow at `path`, confined to `root`: `ok`, the flow's name and its inputs as
 * `name: type` lines, or `invalid:` and the diagnostic.
 */
async function checkText(root: string, path: string, text: string, env: Env): Promise<string> {
  let flow: ComposedFlow;
  try {
    flow = await loadConfinedText(root, path, text, env);
  } catch (error) {
    if (error instanceof LoadError) return `invalid:\n${await renderedDiagnostic(error.diagnostic, env)}`;
    throw error;
  }
  // A flow's `input:` entry always declares its type, so its contract is set.
  const inputs = flow.input.map((decl) => `  ${decl.name}: ${canonicalTypeLabel(decl.contract!.type)}`);
  const name = flow.name ?? flow.compiled.flowId;
  return ["ok", `name: ${name}`, ...(inputs.length > 0 ? ["inputs:", ...inputs] : ["inputs: none"])].join("\n");
}

/** `text` cut to at most `FOCUS_BYTES` of UTF-8, between characters, with a note when it was cut. */
function focusText(text: string): string {
  const bytes = Buffer.from(text, "utf8");
  if (bytes.length <= FOCUS_BYTES) return text;
  let end = FOCUS_BYTES;
  // Back off a UTF-8 continuation byte, so the cut never splits a character.
  while (end > 0 && (bytes[end]! & 0xc0) === 0x80) end--;
  const kept = bytes.subarray(0, end).toString("utf8");
  return `${kept}${kept.endsWith("\n") ? "" : "\n"}(cut at ${FOCUS_BYTES} bytes)`;
}

/**
 * Read and check the flow the chat is about (its focus) at `path`, confined to its own folder, so
 * a flow outside the chat's root works: its text, cut at `FOCUS_BYTES` with a note, and its check
 * as `check_flow` gives it, of the whole text. A file that can't be read gives no text and why.
 */
export async function checkFocus(path: string, env: Env): Promise<{ text: string; check: string }> {
  let text: string;
  try {
    text = readFlowText(path);
  } catch (error) {
    return { text: "", check: toReadFailure(path, error).message };
  }
  const real = realpathSync(path);
  return { text: focusText(text), check: await checkText(dirname(real), real, text, env) };
}

/** The approval answers `write_flow` offers: the first writes, the last (Ctrl-C's) doesn't. */
const WRITE_CHOICES = ["Approve", "Reject"] as const;

/** A file as it was when its approval was asked: its bytes and identity, or null when absent. */
type Snapshot = { bytes: Buffer; dev: number; ino: number } | null;

/**
 * The file at `real` now, as a `Snapshot`.
 *
 * @throws {ToolRefused} When it is a folder or another non-file, or larger than `READ_FLOW_BYTES`.
 * @throws {ReportedToolError} When it can't be read.
 */
function snapshot(path: string, real: string): Snapshot {
  try {
    const { bytes, stats } = readCapped(path, real);
    return { bytes, dev: stats.dev, ino: stats.ino };
  } catch (error) {
    if (isSystemError(error) && (error.code === "ENOENT" || error.code === "ENOTDIR")) return null;
    throw error instanceof ToolRefused ? error : toReadFailure(path, error);
  }
}

/** Whether two snapshots are the same file with the same bytes, or both absent. */
function sameSnapshot(a: Snapshot, b: Snapshot): boolean {
  if (a === null || b === null) return a === b;
  return a.dev === b.dev && a.ino === b.ino && a.bytes.equals(b.bytes);
}

/**
 * Write `content` to the flow file `path` inside `host.root` once it loads there and the user
 * approves it on the terminal, with a diff against the file it replaces. A file that changed
 * during the approval is left alone; a new file is created only if still absent, and an existing
 * one is replaced through a temporary file in its folder, flushed to disk before the rename.
 *
 * Returns what happened, for the model: `wrote <path>`, `invalid:` and the diagnostic, or `not
 * written: <why>`. Once the user is asked, every outcome is returned, a refusal or a failed write
 * included.
 *
 * @throws {ToolRefused} Before asking, for a path that is refused or content or a file too large.
 * @throws {ReportedToolError} Before asking, for a file that can't be read.
 */
async function writeFlow(host: ChatToolHost, path: string, content: string, signal: AbortSignal): Promise<string> {
  const root = resolveRoot(host.root);
  const real = confine(root, path);
  if (!isFlowFile(real)) throw new ToolRefused(`${path} is not a .yaml or .yml file`);
  if (Buffer.byteLength(content) > READ_FLOW_BYTES) {
    throw new ToolRefused(`the content is over ${READ_FLOW_BYTES} bytes, the most write_flow writes`);
  }
  const shown = slashed(relative(root, real));
  // Taken first, so a file write_flow can't replace is refused before the content is checked.
  const before = snapshot(path, real);
  const check = await checkText(root, real, content, host.env);
  if (check.startsWith("invalid:")) return check;
  const diff = unifiedDiff(before === null ? "" : before.bytes.toString("utf8"), content, {
    oldLabel: before === null ? "/dev/null" : shown,
    newLabel: shown,
  });
  const answer = await host.approve({
    title: `Write ${shown}`,
    body: before === null ? "A new flow file." : "Replaces the flow file.",
    diff,
    choices: WRITE_CHOICES,
    signal,
  });
  if (answer === null) return "not written: not approved (no terminal, or the prompt ended)";
  if (answer !== WRITE_CHOICES[0]) return "not written: rejected";
  try {
    return writeApproved(root, path, real, before, content);
  } catch (error) {
    // The model reads why the write failed (a read-only folder, a full disk, a path that now
    // leaves the root) as the outcome.
    if (isReportedToolError(error)) return `not written: ${(error as Error).message}`;
    if (isSystemError(error)) return `not written: ${osReason(error)}`;
    throw error;
  }
}

/**
 * Write `content` to `real` (`path` as the model gave it) once approved, as `writeFlow` describes,
 * unless the file is no longer as `before` saw it.
 *
 * @throws {ReportedToolError} When the path is now refused or the file can't be read.
 * @throws {Error} A system error when a write fails; the temporary file is gone either way.
 */
function writeApproved(root: string, path: string, real: string, before: Snapshot, content: string): string {
  const changed = "not written: changed during approval";
  const shown = slashed(relative(root, real));
  // The path is resolved again: a folder on it may have become a symlink meanwhile.
  if (confine(root, path) !== real || !sameSnapshot(before, snapshot(path, real))) return changed;
  const folder = dirname(real);
  if (before === null) {
    // A refusal below can leave the folders made here empty; they hold nothing outside the root.
    mkdirSync(folder, { recursive: true });
    // Again after the folders exist, so one swapped for a symlink meanwhile can't steer the write.
    if (confine(root, path) !== real || realpathSync(folder) !== folder) return changed;
    try {
      // `wx` fails on any file there, a symlink included, so nothing created meanwhile is replaced.
      writeDurably(real, content, null);
    } catch (error) {
      if (isSystemError(error) && error.code === "EEXIST") return changed;
      throw error;
    }
    return `wrote ${shown}`;
  }
  const temporary = join(folder, `.${basename(real)}.yian-tmp-${process.pid}`);
  // A leftover from an earlier write goes first; a folder there is left alone and fails the write.
  try {
    unlinkSync(temporary);
  } catch (error) {
    if (!isSystemError(error) || error.code !== "ENOENT") throw error;
  }
  let created = false;
  try {
    writeDurably(temporary, content, statSync(real).mode & 0o777, () => (created = true));
    if (realpathSync(folder) !== folder) return changed;
    renameSync(temporary, real);
    created = false;
    syncFolder(folder);
  } finally {
    if (created) quietly(() => unlinkSync(temporary));
  }
  return `wrote ${shown}`;
}

/**
 * Create `path` (`wx`: failing if anything is there, a symlink included) with `content` and flush
 * it to disk; `mode` sets its permissions, null the default. `onCreated` runs once the file exists.
 */
function writeDurably(path: string, content: string, mode: number | null, onCreated?: () => void): void {
  const fd = openSync(path, "wx", mode ?? undefined);
  onCreated?.();
  try {
    writeFileSync(fd, content);
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
}

/** Flush `folder`'s entries to disk, so a rename in it survives a crash; best effort. */
function syncFolder(folder: string): void {
  quietly(() => {
    const fd = openSync(folder, "r");
    try {
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
  });
}

/** Run `action`, dropping what it throws: a cleanup step never replaces an outcome. */
function quietly(action: () => unknown): void {
  try {
    action();
  } catch {}
}

/** A tool's JSON Schema for string arguments, each `[name, description, required]`. */
function stringArguments(...fields: [name: string, description: string, required: boolean][]) {
  return {
    type: "object",
    properties: Object.fromEntries(fields.map(([name, description]) => [name, { type: "string", description }])),
    required: fields.filter(([, , required]) => required).map(([name]) => name),
    additionalProperties: false,
  };
}

/**
 * The chat assistant's file tools over `host`: `list_flows`, `read_flow`, `check_flow` and
 * `write_flow`, which asks the user through `host.approve` before it writes; and `read_docs`, over
 * `host.docs`.
 */
export function chatFileTools(host: ChatToolHost): FunctionTool[] {
  const { root } = host;
  return [
    tool({
      name: "list_flows",
      description:
        "List the flow files (.yaml, .yml) under the chat's folder, or under one folder in it, as paths " +
        `relative to the chat's folder. Skips node_modules and dot-folders, and lists at most ${LIST_FLOWS_LIMIT}; ` +
        "when there are more, pass dir to list one folder.",
      parameters: stringArguments([
        "dir",
        "A folder to list, relative to the chat's folder; all of it when omitted.",
        false,
      ]),
      execute: (args) => {
        const dir = optionalStringArg(args, "dir") ?? ".";
        return listFlows(resolveRoot(root), confine(root, dir));
      },
    }),
    tool({
      name: "read_flow",
      description: `Read a flow file (.yaml, .yml) in the chat's folder, up to ${READ_FLOW_BYTES} bytes.`,
      parameters: stringArguments(["path", "The flow file, relative to the chat's folder.", true]),
      execute: (args) => {
        const path = stringArg(args, "path");
        return readFlow(path, confine(root, path));
      },
    }),
    tool({
      name: "check_flow",
      description:
        "Check that a flow file in the chat's folder loads, running nothing: ok with its name and inputs, " +
        "or invalid with the error. Flows it imports (uses:) and code files it references must also be " +
        "inside the chat's folder; system.paths isn't supported.",
      parameters: stringArguments(["path", "The flow file, relative to the chat's folder.", true]),
      execute: async (args) => {
        const path = stringArg(args, "path");
        const real = confine(root, path);
        return checkText(resolveRoot(root), real, readFlow(path, real), host.env);
      },
    }),
    tool({
      name: "write_flow",
      description:
        "Write a flow file (.yaml, .yml) in the chat's folder, creating it (and any missing folders) or " +
        "replacing it whole; read the file first when replacing it. The content must load (as check_flow " +
        "checks it), and the user approves the write on their terminal first. Returns `wrote <path>`, " +
        "`invalid:` and the error, or `not written: <why>`; don't retry a rejected write unchanged.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "The flow file, relative to the chat's folder." },
          content: { type: "string", description: "The whole new text of the file." },
        },
        required: ["path", "content"],
        additionalProperties: false,
      },
      execute: (args, ctx) => writeFlow(host, stringArg(args, "path"), stringArg(args, "content"), ctx.signal),
    }),
    tool({
      name: "read_docs",
      description: "Read a page of yian's docs; without a page, list the pages there are.",
      parameters: stringArguments(["page", "The page's name, as the list gives it.", false]),
      execute: (args) => {
        const names = Array.from(host.docs.keys()).sort(compareCodeUnits);
        const page = optionalStringArg(args, "page");
        if (page === undefined) return names.length === 0 ? "no docs pages" : names.join("\n");
        const text = host.docs.get(page);
        if (text === undefined) {
          throw new ReportedToolError(`no docs page ${reprText(page)}; the pages are: ${names.join(", ")}`);
        }
        return text;
      },
    }),
  ];
}

/** The longest `wait_flow` waits, in seconds. */
const WAIT_FLOW_MAX_S = 120;

/** The most characters of each text field a run tool's status shows the model. */
const STATUS_TEXT_CHARS = 4000;

/** The approval answers `start_flow` offers: the first two start, the last (Ctrl-C's) doesn't. */
const START_CHOICES = ["Approve", "Approve this flow for this chat", "Reject"] as const;

/** What the run tools need beside the file tools' host. */
export interface ChatRunToolsOptions {
  /**
   * Told each status a run tool hands the model, by run and seq, so the child events up to it
   * aren't relayed again; `null` for none.
   */
  seen?: { markSeen(run: string, seq: number): void } | null;
  /** The clock `wait_flow` times out on; the real one by default. */
  scheduler?: Scheduler;
}

/** `text` cut to `STATUS_TEXT_CHARS` characters, with a note when it was cut. */
function cutText(text: string): string {
  if (text.length <= STATUS_TEXT_CHARS) return text;
  return `${text.slice(0, STATUS_TEXT_CHARS)}... (cut at ${STATUS_TEXT_CHARS} characters)`;
}

/** `status` as the model reads it: its text fields cut, without the seq it doesn't need. */
function shownStatus(status: ChildStatus): Record<string, unknown> {
  const { seq: _, output, failure, asks, replies, note, ...rest } = status;
  const shown: Record<string, unknown> = { ...rest };
  if (output !== undefined) {
    // An output too long as JSON is shown as its cut JSON text.
    const json = JSON.stringify(output) ?? "null";
    if (typeof output === "string") shown["output"] = cutText(output);
    else shown["output"] = json.length > STATUS_TEXT_CHARS ? cutText(json) : output;
  }
  if (failure !== undefined) shown["failure"] = { ...failure, message: cutText(failure.message) };
  if (asks !== undefined) shown["asks"] = asks.map((ask) => ({ ...ask, question: cutText(ask.question) }));
  if (replies !== undefined) shown["replies"] = replies.map(cutText);
  if (note !== undefined) shown["note"] = cutText(note);
  return shown;
}

/**
 * The real paths of the files the CODE references of `flow` and the flows it uses import, sorted:
 * each resolved in its flow's folder, as the CODE worker resolves it.
 */
function codeFiles(flow: ComposedFlow): string[] {
  const files = new Set<string>();
  for (const [, node] of staticNodes(flow)) {
    // Read kind-blind, as the run boundary reads a grant: only a CODE node has these fields.
    const code = node as { mode?: unknown; code?: unknown; baseDir?: unknown };
    if (node.kind !== "code" || code.mode !== "reference" || typeof code.code !== "string") continue;
    const base = typeof code.baseDir === "string" ? code.baseDir : ".";
    files.add(resolve(base, splitReference(code.code).path));
  }
  return Array.from(files).sort(compareCodeUnits);
}

/**
 * The sha256 of everything that runs when `flow` starts: its flow files (`text`, the root's, and
 * each file it was loaded from), the bytes of each of its CODE reference `files` (or why one
 * can't be read), and `can`, what the approval card says it can do.
 */
function startDigest(flow: ComposedFlow, text: string, files: readonly string[], can: readonly string[]): string {
  const hash = createHash("sha256").update(text).update("\0");
  const sources = flow.source?.files ?? {};
  for (const key of Object.keys(sources).sort(compareCodeUnits)) {
    hash.update(key).update("\0").update(sources[key]!.text).update("\0");
  }
  for (const file of files) {
    hash.update(file).update("\0");
    try {
      hash.update(readFileSync(file));
    } catch (error) {
      if (!isSystemError(error)) throw error;
      hash.update(`unreadable: ${error.code}`);
    }
    hash.update("\0");
  }
  return hash.update(can.join("\n")).digest("hex");
}

/**
 * Check `inputs` against `flow`'s inputs before it starts, as its start would: no unknown names,
 * every required one given, each value of its declared type.
 *
 * @throws {ReportedToolError} `bad inputs: <why>`.
 */
function checkStartInputs(flow: ComposedFlow, inputs: unknown): void {
  try {
    const coerced = coerceInputs(flow.input, hostInputs(flow.input, inputs));
    for (const param of flow.input) {
      if (isInboxType(param.contract?.type ?? null)) continue;
      if (!Object.hasOwn(coerced, param.name)) {
        if (!param.required || param.hasDefault) continue;
        throw new RangeError(`required input ${reprText(param.name)} is missing`);
      }
      if (param.contract === null) continue;
      try {
        buildValueAs(param.contract.type, coerced[param.name]);
      } catch (error) {
        if (error instanceof Error) throw new TypeError(`input ${reprText(param.name)}: ${error.message}`);
        throw error;
      }
    }
  } catch (error) {
    if (error instanceof Error) throw new ReportedToolError(`bad inputs: ${error.message}`);
    throw error;
  }
}

/**
 * What the flow `flow` can do, over its own nodes and the flows it uses: the node kinds, the
 * tools its TOOL nodes call, the tools and skills its agents are granted, and whether it runs
 * code, with the files its CODE references import (`files`, as shown); one line each, for the
 * approval card.
 */
function capabilities(flow: ComposedFlow, files: readonly string[]): string[] {
  const kinds = new Set<string>();
  const called = new Set<string>();
  const granted = new Set<string>();
  for (const [, node] of staticNodes(flow)) {
    // Every flow has its start and end; they do nothing to approve.
    if (node.kind !== "start" && node.kind !== "end") kinds.add(node.kind);
    const call = node.hostToolCall();
    if (call !== null) called.add(call.toolId);
    for (const grant of node.grantedTools()) granted.add(grant);
  }
  const skills = new Set<string>();
  for (const [, grant] of skillGrants(flow.compiled)) {
    for (const name of grant === "all" ? ["all the skills the run has"] : grant) skills.add(name);
  }
  const listed = (names: ReadonlySet<string>) => Array.from(names).sort(compareCodeUnits).join(", ");
  const lines = [`node kinds: ${listed(kinds) || "none"}`];
  if (called.size > 0) lines.push(`tools it calls: ${listed(called)}`);
  if (granted.size > 0) lines.push(`tools its agents may call: ${listed(granted)}`);
  if (skills.size > 0) lines.push(`skills its agents get: ${listed(skills)}`);
  if (kinds.has("code")) {
    const named = files.length > 0 ? `; files: ${files.join(", ")}` : "";
    lines.push(`runs code on this machine (CODE nodes${named})`);
  }
  return lines;
}

/**
 * The child run `id` names, for a tool.
 *
 * @throws {ReportedToolError} When no child has the id, or it isn't a string.
 */
function knownRun(runs: ChildRuns, args: Record<string, unknown>): string {
  const id = stringArg(args, "run");
  try {
    runs.status(id);
  } catch (error) {
    if (error instanceof RangeError) throw new ReportedToolError(error.message);
    throw error;
  }
  return id;
}

/**
 * Load the flow file `path` in `host.root` for `start_flow`: its real path, text and flow.
 *
 * @throws {ToolRefused} For a path that is refused, or a file that isn't a flow file or is too large.
 * @throws {ReportedToolError} For a file that can't be read, or a flow that doesn't load.
 */
async function loadForStart(
  host: ChatToolHost,
  path: string,
): Promise<{ real: string; shown: string; text: string; flow: ComposedFlow }> {
  const root = resolveRoot(host.root);
  const real = confine(root, path);
  const text = readFlow(path, real);
  try {
    const flow = await loadConfinedText(root, real, text, host.env);
    return { real, shown: slashed(relative(root, real)), text, flow };
  } catch (error) {
    if (error instanceof LoadError) {
      throw new ReportedToolError(`invalid:\n${await renderedDiagnostic(error.diagnostic, host.env)}`);
    }
    throw error;
  }
}

/**
 * The chat assistant's run tools over `runs`: `start_flow`, which asks the user through
 * `host.approve` before it starts a flow in `host.root`; `flow_status` and `wait_flow`, which
 * read a child's status; and `answer_flow`, `message_flow`, `close_inbox` and `stop_flow`, which
 * relay to it. Each returns the child's status as JSON, its text fields cut at
 * `STATUS_TEXT_CHARS`, and marks it read (`runs.markRead`, `options.seen`). An unknown run, or an
 * argument of the wrong type, is a `ReportedToolError`.
 */
export function chatRunTools(host: ChatToolHost, runs: ChildRuns, options: ChatRunToolsOptions = {}): FunctionTool[] {
  const { seen = null, scheduler = realScheduler } = options;
  // The approvals given "for this chat", by real path: the digest of what was approved.
  const approved = new Map<string, string>();
  // By child, the last seq a tool showed the model, so `wait_flow` returns at once for a newer one.
  const shown = new Map<string, number>();
  const show = (status: ChildStatus): string => {
    runs.markRead(status.run, status.seq);
    seen?.markSeen(status.run, status.seq);
    const last = shown.get(status.run) ?? 0;
    shown.set(status.run, Math.max(status.seq, last));
    // The stored status keeps the replies it was published with; once shown, they are read.
    const { replies: _, ...unrepeated } = status;
    return JSON.stringify(shownStatus(status.seq <= last ? unrepeated : status));
  };
  const runArgument = (description: string, ...more: [name: string, description: string][]) =>
    stringArguments(
      ["run", description, true],
      ...more.map(([name, text]): [string, string, boolean] => [name, text, true]),
    );
  const RUN = "The run's id, as start_flow returned it (r1, r2, ...).";
  const STATUS =
    "Returns the run's status as JSON: status (running, waiting, succeeded, failed or cancelled), output, " +
    "failure, waits, asks (questions with their call ids), replies (what it said since you last looked), note " +
    "(something that couldn't be done) and log.";
  return [
    tool({
      name: "start_flow",
      description:
        "Start a flow file (.yaml, .yml) in the chat's folder as a background run of this chat, once the user " +
        "approves it on their terminal. Run check_flow first to learn its inputs, and ask the user for input " +
        "values you don't know rather than inventing them; an Inbox input is never given here (use " +
        `message_flow). ${STATUS} The status's run is the id the other run tools take. Returns \`not ` +
        "started: <why>` when the user rejects it or it can't start; don't retry a rejected start unchanged.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "The flow file, relative to the chat's folder." },
          inputs: { type: "object", description: "The flow's inputs by name; {} for none." },
        },
        required: ["path", "inputs"],
        additionalProperties: false,
      },
      execute: (args, ctx) => startFlow(stringArg(args, "path"), args["inputs"], ctx.signal),
    }),
    tool({
      name: "flow_status",
      description: `Read where a run this chat started stands now, without waiting. ${STATUS}`,
      parameters: runArgument(RUN),
      isConcurrencySafe: () => true,
      execute: (args) => show(runs.status(knownRun(runs, args))),
    }),
    tool({
      name: "wait_flow",
      description:
        "Wait for a run this chat started to change (a reply, a question, its end), up to `seconds` " +
        `(at most ${WAIT_FLOW_MAX_S}), and return its status then, or at the timeout. Returns at once when ` +
        `it changed since you last saw it. ${STATUS}`,
      parameters: {
        type: "object",
        properties: {
          run: { type: "string", description: RUN },
          seconds: {
            type: "number",
            description: `How long to wait at most, in seconds (at most ${WAIT_FLOW_MAX_S}).`,
          },
        },
        required: ["run", "seconds"],
        additionalProperties: false,
      },
      isConcurrencySafe: () => true,
      execute: async (args, ctx) => {
        const id = knownRun(runs, args);
        const seconds = args["seconds"];
        if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
          throw new ReportedToolError("seconds must be a number of seconds, 0 or more");
        }
        const done = new AbortController();
        const end = () => done.abort();
        ctx.signal.addEventListener("abort", end, { once: true });
        if (ctx.signal.aborted) end();
        scheduler.sleep(Math.min(seconds, WAIT_FLOW_MAX_S) * 1000, done.signal).then(end, () => {});
        try {
          return show(await runs.next(id, shown.get(id) ?? 0, { signal: done.signal }));
        } finally {
          ctx.signal.removeEventListener("abort", end);
          end();
        }
      },
    }),
    tool({
      name: "answer_flow",
      description:
        "Answer a question a run this chat started asks (its status's asks), by its call id, with the user's " +
        "answer: relay the question to the user and pass on what they say, and don't answer a question twice. " +
        `The answer is any JSON value but null; text for a plain question. ${STATUS}`,
      parameters: {
        type: "object",
        properties: {
          run: { type: "string", description: RUN },
          call: { type: "string", description: "The question's call id, from the status's asks." },
          answer: { description: "The answer." },
        },
        required: ["run", "call", "answer"],
        additionalProperties: false,
      },
      execute: async (args) => {
        const id = knownRun(runs, args);
        const call = stringArg(args, "call");
        const answer = args["answer"];
        if (answer === null || answer === undefined) throw new ReportedToolError("answer can't be null");
        return show(await runs.answer(id, call, answer));
      },
    }),
    tool({
      name: "message_flow",
      description:
        "Send text to an Inbox input of a run this chat started, as a chat message to a flow that takes them; " +
        `its status's waits name the inbox a run waits on. ${STATUS}`,
      parameters: runArgument(RUN, ["inbox", "The Inbox input's name."], ["text", "The message."]),
      execute: async (args) => {
        const id = knownRun(runs, args);
        return show(await runs.message(id, stringArg(args, "inbox"), stringArg(args, "text")));
      },
    }),
    tool({
      name: "close_inbox",
      description:
        "Close an Inbox input of a run this chat started, so the flow stops waiting for messages there and " +
        `can finish. ${STATUS}`,
      parameters: runArgument(RUN, ["inbox", "The Inbox input's name."]),
      execute: async (args) => {
        const id = knownRun(runs, args);
        return show(await runs.closeInbox(id, stringArg(args, "inbox")));
      },
    }),
    tool({
      name: "stop_flow",
      description: `Cancel a run this chat started; use it only when the user asks. ${STATUS}`,
      parameters: runArgument(RUN),
      execute: async (args) => show(await runs.stop(knownRun(runs, args))),
    }),
  ];

  /**
   * Start the flow at `path` with `inputs` once the user approves it, as `start_flow` describes.
   * Every outcome once the user is asked is returned.
   */
  async function startFlow(path: string, inputs: unknown, signal: AbortSignal): Promise<string> {
    const { real, shown: file, text, flow } = await loadForStart(host, path);
    checkStartInputs(flow, inputs);
    const files = codeFiles(flow);
    const root = resolveRoot(host.root);
    const can = capabilities(flow, files.map((each) => slashed(relative(root, each))));
    const digest = startDigest(flow, text, files, can);
    if (approved.get(real) !== digest) {
      const answer = await host.approve({
        title: `Run ${file}`,
        body: [`Starts ${file} in the background.`, `inputs: ${JSON.stringify(inputs)}`, ...can].join("\n"),
        diff: null,
        choices: START_CHOICES,
        signal,
      });
      if (answer === null) return "not started: not approved (no terminal, or the prompt ended)";
      if (answer === START_CHOICES[1]) approved.set(real, digest);
      else if (answer !== START_CHOICES[0]) return "not started: rejected";
    }
    let id: string;
    try {
      ({ id } = await runs.start(flow, inputs as Record<string, unknown>, { signal }));
    } catch (error) {
      if (error instanceof Error) return `not started: ${error.message}`;
      throw error;
    }
    return show(runs.status(id));
  }
}
