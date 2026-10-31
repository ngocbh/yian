/**
 * The `composing-agents` skill `yian chat` carries (`SKILL` in `bundled.ts`), on disk: a skill
 * reads its bundled files from a folder, so the chat writes the skill once to a folder under
 * `~/.yian/cache/skills/` named by a hash of its content, and loads it from there.
 *
 * Knows about: the file system, the skill loader and the bundle.
 */

import { createHash, randomBytes } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import { loadSkillDir } from "../compose/skills.ts";
import type { Env } from "../defaults.ts";
import { compareCodeUnits } from "../diagnostics.ts";
import { isSystemError } from "../platform.ts";
import { Skill, SkillError } from "../skills/catalog.ts";
import { SKILL } from "./bundled.ts";

/** The bundled skill's name, which its folder's name starts with. */
const NAME = "composing-agents";

/** How many hex digits of the content hash name the folder. */
const HASH_DIGITS = 16;

/** The sha256 of `files`, path and text each, in path order. */
function contentHash(files: Readonly<Record<string, string>>): string {
  const hash = createHash("sha256");
  for (const path of Object.keys(files).sort(compareCodeUnits)) {
    hash.update(path).update("\0").update(files[path]!).update("\0");
  }
  return hash.digest("hex");
}

/** Create `dir` and the folders above it, `~/.yian` included, owner-only where this creates them. */
function makeDirs(dir: string): void {
  mkdirSync(dir, { recursive: true, mode: 0o700 });
}

/** A name for a folder in `cache` this call alone uses, beside the skill's folder. */
function scratchName(cache: string, what: string): string {
  return join(cache, `.${NAME}.${what}-${process.pid}-${randomBytes(6).toString("hex")}`);
}

/** Move `folder` aside and remove it; one another chat moved first is left to it. */
function discard(cache: string, folder: string): void {
  const aside = scratchName(cache, "damaged");
  try {
    renameSync(folder, aside);
  } catch (error) {
    if (isSystemError(error) && error.code === "ENOENT") return;
    throw error;
  }
  rmSync(aside, { recursive: true, force: true });
}

/** Whether `file` is a regular file holding exactly `text`; false when it is missing. */
function holds(file: string, text: string): boolean {
  const expected = Buffer.from(text);
  try {
    // Only a regular file of the right size is read: reading a FIFO or a device could block or never end.
    const stat = statSync(file);
    if (!stat.isFile() || stat.size !== expected.length) return false;
    return readFileSync(file).equals(expected);
  } catch (error) {
    if (isSystemError(error) && (error.code === "ENOENT" || error.code === "ENOTDIR")) return false;
    throw error;
  }
}

/**
 * The skill in `folder` when it holds every file of the bundled one, each with the bundle's
 * content, and loads; `absent` when there is no folder (or it went away meanwhile), `damaged`
 * otherwise.
 */
function loadIntact(folder: string): Skill | "absent" | "damaged" {
  if (!statSync(folder, { throwIfNoEntry: false })?.isDirectory()) return "absent";
  const whole = Object.entries(SKILL).every(([path, text]) => holds(join(folder, ...path.split("/")), text));
  if (!whole) return "damaged";
  try {
    return loadSkillDir(folder);
  } catch (error) {
    if (error instanceof SkillError) return "damaged";
    if (isSystemError(error) && error.code === "ENOENT") return "absent";
    throw error;
  }
}

/**
 * Return the bundled `composing-agents` skill, loaded from
 * `~/.yian/cache/skills/composing-agents-<hash>/` (`HOME` read from `env`, the process user's home
 * when it is unset or empty), writing the folder first when it doesn't exist.
 *
 * The folder is written whole under a temporary name beside it and renamed into place, so a chat
 * never reads a half-written one. When another chat renamed its own into place first, that one is
 * used and this call's is removed. A folder that exists is never written again, unless it is
 * damaged (a file of the skill missing or different from the bundle's, or a `SKILL.md` that doesn't
 * load) on two looks in a row, the second for a chat that is replacing it: then it is moved aside,
 * removed and written again.
 * The folders are the owner's alone, as `~/.yian` is.
 *
 * @param options.beforeRename For tests: runs just before the rename.
 * @throws {Error} A system error when the folder can't be written or read.
 * @throws {SkillError} When the folder's `SKILL.md` doesn't load.
 */
export function bundledSkill(env: Env, options: { beforeRename?: () => void } = {}): Skill {
  const cache = join(env("HOME") || homedir(), ".yian", "cache", "skills");
  const folder = join(cache, `${NAME}-${contentHash(SKILL).slice(0, HASH_DIGITS)}`);
  let found = loadIntact(folder);
  if (found === "damaged") found = loadIntact(folder);
  if (found instanceof Skill) return found;
  if (found === "damaged") discard(cache, folder);
  makeDirs(cache);
  const temporary = scratchName(cache, "tmp");
  try {
    for (const [path, text] of Object.entries(SKILL)) {
      const file = join(temporary, ...path.split("/"));
      makeDirs(dirname(file));
      writeFileSync(file, text);
    }
    options.beforeRename?.();
    try {
      renameSync(temporary, folder);
    } catch (error) {
      // A folder that is there now is another chat's, renamed in first: it holds the same files.
      const taken = isSystemError(error) && (error.code === "EEXIST" || error.code === "ENOTEMPTY");
      if (!taken || !statSync(folder, { throwIfNoEntry: false })?.isDirectory()) throw error;
    }
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
  return loadSkillDir(folder);
}
