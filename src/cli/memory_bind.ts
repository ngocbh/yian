/**
 * How the command line binds a flow's memory inputs: the default identity of an unbound input,
 * `<project key>/<flow id>/<input>`, and its location under `~/.yian/memory`, confined there.
 *
 * Knows about: the platform's path helpers and the file system.
 */

import { createHash } from "node:crypto";
import { lstatSync, type Stats } from "node:fs";
import { basename, join, relative, sep } from "node:path";

import { isSystemError, leavesRoot } from "../platform.ts";
import { resolveFlowPath } from "./confined_load.ts";

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
