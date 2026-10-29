/**
 * The host tools `yian chat`'s assistant works on flow files with, each confined to the folder the
 * chat started in (the root): a path that leaves it, through `..`, an absolute path or a symlink,
 * is refused.
 *
 * Knows about: the file system, the host tool API and its errors.
 */

import { lstatSync, realpathSync } from "node:fs";
import { relative, resolve } from "node:path";

import { isSystemError, leavesRoot } from "../platform.ts";
import { resolvedPath } from "../skills/catalog.ts";
import { ReportedToolError } from "../tools/errors.ts";

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
