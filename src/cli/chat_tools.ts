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
  type Dirent,
  lstatSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  type Stats,
  statSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, extname, join, relative, resolve, sep } from "node:path";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import type { Env } from "../defaults.ts";
import { compareCodeUnits, type Diagnostic, reprText } from "../diagnostics.ts";
import { isSystemError, leavesRoot } from "../platform.ts";
import { resolvedPath } from "../skills/catalog.ts";
import { ReportedToolError } from "../tools/errors.ts";
import { type FunctionTool, tool } from "../tools/function_tools.ts";
import { canonicalTypeLabel } from "../typesys/values.ts";
import { loadConfinedText } from "./confined_load.ts";
import { unifiedDiff } from "./diff.ts";
import { renderDiagnostic } from "./render.ts";
import { NotUtf8Error, readFlowText } from "./startup.ts";
import type { ApproveRequest } from "./terminal.ts";
import { osReason } from "./utils.ts";

/** The most flows `list_flows` lists. */
const LIST_FLOWS_LIMIT = 500;

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
 * @throws {ToolRefused} When the path has a control character, leaves `root`, or is or goes
 *   through a symlink whose target doesn't exist.
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
    if (error.code !== "ENOENT" && error.code !== "ENOTDIR") {
      throw new ReportedToolError(`can't resolve ${path}: ${osReason(error)}`);
    }
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

/** The error the model reads for a file `path` that can't be read: why, never what it holds. */
function readFailure(path: string, error: unknown): ReportedToolError {
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
 * The flow files under `dir`, relative to `root` and sorted, skipping `node_modules` and
 * dot-folders and not following symlinks; at most `LIST_FLOWS_LIMIT` of them, then a note.
 */
function listFlows(root: string, dir: string): string {
  const found: string[] = [];
  let cut = false;
  const walk = (folder: string): void => {
    let entries: Dirent[];
    try {
      entries = readdirSync(folder, { withFileTypes: true });
    } catch (error) {
      // A folder inside that can't be read (no permission, say) leaves only its flows out.
      if (folder !== dir && isSystemError(error)) return;
      throw readFailure(slashed(relative(root, folder)) || ".", error);
    }
    entries.sort((a, b) => compareCodeUnits(a.name, b.name));
    for (const entry of entries) {
      if (cut) return;
      const path = join(folder, entry.name);
      if (entry.isDirectory()) {
        if (entry.name !== "node_modules" && !entry.name.startsWith(".")) walk(path);
      } else if (entry.isFile() && isFlowFile(entry.name)) {
        if (found.length === LIST_FLOWS_LIMIT) cut = true;
        else found.push(slashed(relative(root, path)));
      }
    }
  };
  walk(dir);
  if (found.length === 0) return "no flows";
  found.sort(compareCodeUnits);
  if (cut) found.push(`(stopped at ${LIST_FLOWS_LIMIT} flows; pass dir to list one folder)`);
  return found.join("\n");
}

/**
 * The text of the flow file `path` (as the model gave it) at `real`, refusing anything but a YAML
 * file of at most `READ_FLOW_BYTES`.
 */
function readFlow(path: string, real: string): string {
  if (!isFlowFile(real)) throw new ToolRefused(`${path} is not a .yaml or .yml file`);
  let size: number;
  try {
    const stats = statSync(real);
    if (!stats.isFile()) throw new ToolRefused(`${path} is not a file`);
    size = stats.size;
  } catch (error) {
    throw error instanceof ToolRefused ? error : readFailure(path, error);
  }
  if (size > READ_FLOW_BYTES) {
    throw new ToolRefused(`${path} is ${size} bytes; read_flow reads at most ${READ_FLOW_BYTES}`);
  }
  try {
    return readFlowText(real);
  } catch (error) {
    throw readFailure(path, error);
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
    return { text: "", check: readFailure(path, error).message };
  }
  const real = realpathSync(path);
  return { text: focusText(text), check: await checkText(dirname(real), real, text, env) };
}

/** The approval answers `write_flow` offers: the first writes, the last (Ctrl-C's) doesn't. */
const WRITE_CHOICES = ["Approve", "Reject"] as const;

/** A file as it was when its approval was asked: its bytes and identity, or null when absent. */
type Snapshot = { bytes: Buffer; dev: number; ino: number } | null;

/** The file at `real` now, as a `Snapshot`; a folder or another non-file is refused. */
function snapshot(path: string, real: string): Snapshot {
  let stats: Stats;
  try {
    stats = statSync(real);
  } catch (error) {
    if (isSystemError(error) && (error.code === "ENOENT" || error.code === "ENOTDIR")) return null;
    throw readFailure(path, error);
  }
  if (!stats.isFile()) throw new ToolRefused(`${path} is not a file`);
  try {
    return { bytes: readFileSync(real), dev: stats.dev, ino: stats.ino };
  } catch (error) {
    throw readFailure(path, error);
  }
}

/** Whether two snapshots are the same file with the same bytes, or both absent. */
function sameSnapshot(a: Snapshot, b: Snapshot): boolean {
  if (a === null || b === null) return a === b;
  return a.dev === b.dev && a.ino === b.ino && a.bytes.equals(b.bytes);
}

/**
 * Write `content` to the flow file `path` inside `host.root` once it loads there and the user
 * approves it on the terminal, with a diff against the file it replaces. Returns what happened, for
 * the model: `wrote <path>`, `invalid:` and the diagnostic, or `not written: <why>`. A file that
 * changed during the approval is left alone; a new file is created only if still absent, and an
 * existing one is replaced through a temporary file in its folder.
 */
async function writeFlow(host: ChatToolHost, path: string, content: string, signal: AbortSignal): Promise<string> {
  const root = realpathSync(host.root);
  const real = confine(root, path);
  if (!isFlowFile(real)) throw new ToolRefused(`${path} is not a .yaml or .yml file`);
  if (Buffer.byteLength(content) > READ_FLOW_BYTES) {
    throw new ToolRefused(`the content is over ${READ_FLOW_BYTES} bytes, the most write_flow writes`);
  }
  const shown = slashed(relative(root, real));
  const check = await checkText(root, real, content, host.env);
  if (check.startsWith("invalid:")) return check;
  const before = snapshot(path, real);
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
  if (answer === null) return "not written: needs a terminal to approve";
  if (answer !== WRITE_CHOICES[0]) return "not written: rejected";
  const changed = "not written: changed during approval";
  // The path is resolved again: a folder on it may have become a symlink meanwhile.
  if (confine(root, path) !== real || !sameSnapshot(before, snapshot(path, real))) return changed;
  const folder = dirname(real);
  if (before === null) {
    mkdirSync(folder, { recursive: true });
    if (realpathSync(folder) !== folder) return changed;
    try {
      // `wx` fails on any file there, a symlink included, so nothing created meanwhile is replaced.
      writeFileSync(real, content, { flag: "wx" });
    } catch (error) {
      if (isSystemError(error) && error.code === "EEXIST") return changed;
      throw error;
    }
    return `wrote ${shown}`;
  }
  const temporary = join(folder, `.${basename(real)}.yian-tmp-${process.pid}`);
  // A leftover from an earlier write goes first; `wx` then refuses anything put there since.
  rmSync(temporary, { force: true });
  try {
    writeFileSync(temporary, content, { flag: "wx", mode: statSync(real).mode & 0o777 });
    if (realpathSync(folder) !== folder) return changed;
    renameSync(temporary, real);
  } finally {
    rmSync(temporary, { force: true });
  }
  return `wrote ${shown}`;
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
