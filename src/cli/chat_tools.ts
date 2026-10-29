/**
 * The host tools `yian chat`'s assistant works on flow files with (`chatFileTools`), each confined
 * to the folder the chat started in (the root): a path that leaves it, through `..`, an absolute
 * path or a symlink, is refused. A refusal or a failure is a `ReportedToolError`, which the model
 * reads as the call's result.
 *
 * Knows about: the file system, the host tool API and its errors, confined loading and the
 * terminal's approval prompt.
 */

import { type Dirent, lstatSync, readdirSync, realpathSync, statSync } from "node:fs";
import { dirname, extname, join, relative, resolve, sep } from "node:path";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import type { Env } from "../defaults.ts";
import { compareCodeUnits, type Diagnostic } from "../diagnostics.ts";
import { isSystemError, leavesRoot } from "../platform.ts";
import { resolvedPath } from "../skills/catalog.ts";
import { ReportedToolError } from "../tools/errors.ts";
import { type FunctionTool, tool } from "../tools/function_tools.ts";
import { canonicalTypeLabel } from "../typesys/values.ts";
import { loadConfinedText } from "./confined_load.ts";
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

/** A tool refused a call, for a path outside the root say; the model reads the message. */
export class ToolRefused extends ReportedToolError {
  override name = "ToolRefused";
}

/**
 * Return the real path `path` (relative to `root`, or absolute) names, which must be inside
 * `root`. A path that doesn't exist yet resolves through its nearest existing folder.
 *
 * @throws {ToolRefused} When the path leaves `root`, or is a symlink whose target doesn't exist.
 * @throws {Error} A system error other than a missing file, such as a symlink loop.
 */
export function confine(root: string, path: string): string {
  const realRoot = realpathSync(root);
  const target = resolve(realRoot, path);
  let real: string;
  try {
    real = realpathSync(target);
  } catch (error) {
    if (!isSystemError(error) || (error.code !== "ENOENT" && error.code !== "ENOTDIR")) throw error;
    // A dangling symlink would create its target, wherever that is.
    if (lstatOrNull(target)?.isSymbolicLink()) throw new ToolRefused(`${path} is a symlink to a missing file`);
    real = resolvedPath(target);
  }
  if (leavesRoot(relative(realRoot, real))) throw new ToolRefused(`${path} is outside the folder the chat started in`);
  return real;
}

/** `path`'s own status, a symlink not followed, or null when it doesn't exist. */
function lstatOrNull(path: string): ReturnType<typeof lstatSync> | null {
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
 * The chat assistant's file tools over `host`: `list_flows`, `read_flow` and `check_flow`.
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
  ];
}
