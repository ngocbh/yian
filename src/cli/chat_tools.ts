/**
 * The host tools `yian chat`'s assistant works on flow files with (`chatFileTools`), each confined
 * to the folder the chat started in (the root): a path that leaves it, through `..`, an absolute
 * path or a symlink, is refused. A refusal or a failure is a `ReportedToolError`, which the model
 * reads as the call's result.
 *
 * Knows about: the file system, the host tool API and its errors, confined loading and the
 * terminal's approval prompt.
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
  readSync,
  realpathSync,
  renameSync,
  type Stats,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, extname, join, relative, resolve, sep } from "node:path";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import type { Env } from "../defaults.ts";
import { compareCodeUnits, type Diagnostic, reprText } from "../diagnostics.ts";
import { isSystemError, leavesRoot } from "../platform.ts";
import { resolvedPath } from "../skills/catalog.ts";
import { isReportedToolError, ReportedToolError } from "../tools/errors.ts";
import { type FunctionTool, tool } from "../tools/function_tools.ts";
import { canonicalTypeLabel } from "../typesys/values.ts";
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
  const realRoot = realpathSync(root);
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
      if (visited === limits.entries) return;
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
  if (found.length === 0 && visited < limits.entries) return "no flows";
  found.sort(compareCodeUnits);
  const lines = found.slice(0, limits.flows);
  if (found.length > limits.flows) {
    lines.push(`(${found.length - limits.flows} more not shown; pass dir to list one folder)`);
  }
  if (visited === limits.entries) {
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
  const root = realpathSync(host.root);
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
        "relative to the chat's folder. Skips node_modules and dot-folders.",
      parameters: stringArguments([
        "dir",
        "A folder to list, relative to the chat's folder; all of it when omitted.",
        false,
      ]),
      execute: (args) => {
        const dir = typeof args["dir"] === "string" ? args["dir"] : ".";
        return listFlows(realpathSync(root), confine(root, dir));
      },
    }),
    tool({
      name: "read_flow",
      description: `Read a flow file (.yaml, .yml) in the chat's folder, up to ${READ_FLOW_BYTES} bytes.`,
      parameters: stringArguments(["path", "The flow file, relative to the chat's folder.", true]),
      execute: (args) => {
        const path = String(args["path"]);
        return readFlow(path, confine(root, path));
      },
    }),
    tool({
      name: "check_flow",
      description:
        "Check that a flow file in the chat's folder loads, running nothing: ok with its name and inputs, " +
        "or invalid with the error. Its uses: flows and code files must be in the chat's folder too.",
      parameters: stringArguments(["path", "The flow file, relative to the chat's folder.", true]),
      execute: async (args) => {
        const path = String(args["path"]);
        const real = confine(root, path);
        return checkText(realpathSync(root), real, readFlow(path, real), host.env);
      },
    }),
    tool({
      name: "write_flow",
      description:
        "Write a flow file (.yaml, .yml) in the chat's folder, creating it or replacing it whole. The content " +
        "must load (as check_flow checks it), and the user approves the write on their terminal first.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "The flow file, relative to the chat's folder." },
          content: { type: "string", description: "The whole new text of the file." },
        },
        required: ["path", "content"],
        additionalProperties: false,
      },
      execute: (args, ctx) => writeFlow(host, String(args["path"]), String(args["content"]), ctx.signal),
    }),
    tool({
      name: "read_docs",
      description: "Read a page of yian's docs; without a page, list the pages there are.",
      parameters: stringArguments(["page", "The page's name, as the list gives it.", false]),
      execute: (args) => {
        const names = Array.from(host.docs.keys()).sort(compareCodeUnits);
        const page = args["page"];
        if (typeof page !== "string") return names.join("\n");
        const text = host.docs.get(page);
        if (text === undefined) {
          throw new ReportedToolError(`no docs page ${reprText(page)}; the pages are: ${names.join(", ")}`);
        }
        return text;
      },
    }),
  ];
}
