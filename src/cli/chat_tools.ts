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
import { extname, join, relative, resolve, sep } from "node:path";

import type { Env } from "../defaults.ts";
import { compareCodeUnits } from "../diagnostics.ts";
import { isSystemError, leavesRoot } from "../platform.ts";
import { resolvedPath } from "../skills/catalog.ts";
import { ReportedToolError } from "../tools/errors.ts";
import { type FunctionTool, tool } from "../tools/function_tools.ts";
import { NotUtf8Error, readFlowText } from "./startup.ts";
import type { ApproveRequest } from "./terminal.ts";
import { osReason } from "./utils.ts";

/** The most flows `list_flows` lists. */
const LIST_FLOWS_LIMIT = 500;

/** The largest flow file `read_flow` reads, in bytes. */
const READ_FLOW_BYTES = 256 * 1024;

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
 * The chat assistant's file tools over `host`: `list_flows` and `read_flow`.
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
  ];
}
