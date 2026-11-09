/**
 * Skill values and confined reads of the files a skill bundles.
 *
 * A skill is a directory holding `SKILL.md` (YAML frontmatter with `name` and `description`,
 * then Markdown instructions) plus optional bundled files. Reading skill directories from disk
 * lives in `compose/skills.ts`, because only Compose parses YAML.
 */

import { closeSync, constants, fstatSync, openSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";

import { current } from "../defaults.ts";
import { reprText } from "../diagnostics.ts";
import { isMissingPath, isSystemError, leavesRoot, openFilePath, resolvedPath, statOrNull } from "../platform.ts";
import { valueRepr } from "../typesys/values.ts";

export const SKILL_FILE = "SKILL.md";
const NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/** A skill package, grant, or resource request is invalid. */
export class SkillError extends RangeError {
  override name = "SkillError";
}

/**
 * One skill a host makes available to a run.
 *
 * @example
 * new Skill({ name: "release-notes", description: "Write release notes.", instructions: "Group changes by area." });
 */
export class Skill {
  /** Grant name: 1-64 lowercase letters, digits, and single hyphens. */
  readonly name: string;
  /** Non-empty text telling the model when to use the skill. */
  readonly description: string;
  /** The `SKILL.md` body after the frontmatter, read once when loaded. */
  readonly instructions: string;
  /** Resolved skill directory for bundled files; `null` means the skill has no readable files. */
  readonly directory: string | null;

  /**
   * @throws {SkillError} If a value is one the grant syntax and system message cannot carry.
   */
  constructor(options: { name: string; description: string; instructions: string; directory?: string | null }) {
    const { name, description, instructions, directory } = options;
    checkSkillName(name);
    if (typeof description !== "string" || description.trim() === "") {
      throw new SkillError(`skill ${reprText(name)}: description must be a non-empty string`);
    }
    if (typeof instructions !== "string") {
      throw new SkillError(`skill ${reprText(name)}: instructions must be a string`);
    }
    if (directory !== null && directory !== undefined && typeof directory !== "string") {
      throw new SkillError(`skill ${reprText(name)}: directory must be a string or null`);
    }
    this.name = name;
    this.description = description;
    this.instructions = instructions;
    this.directory = directory ?? null;
    Object.freeze(this);
  }
}

/**
 * Return `name` when it is a valid skill name.
 *
 * @param name Candidate name from frontmatter, a directory, or a YAML grant.
 * @returns The unchanged name.
 * @throws {SkillError} If `name` is not 1 to `current().maxSkillNameChars` lowercase letters,
 *   digits, and single hyphens, starting and ending with a letter or digit.
 */
export function checkSkillName(name: unknown): string {
  const limit = current().maxSkillNameChars;
  if (typeof name !== "string" || Array.from(name).length > limit || !NAME.test(name)) {
    const shown = typeof name === "string" ? reprText(name) : valueRepr(name);
    throw new SkillError(`invalid skill name ${shown}: use 1-${limit} lowercase letters, digits, and single hyphens`);
  }
  return name;
}

/**
 * @internal Test seams for `readSkillFile`: a call between the path checks and the open, and the
 * read of an open file's path from its descriptor.
 */
export interface SkillFileDeps {
  beforeOpen?: (path: string) => void;
  openedPath?: (fd: number) => string | null;
}

/**
 * Return one UTF-8 file bundled with `skill`, confined to its directory.
 *
 * @param skill The granted skill.
 * @param path POSIX path relative to the skill directory, such as `references/api.md`. Absolute
 *   paths, `..`, backslashes, hidden (dot) files and folders, and symlinks that leave the
 *   directory or resolve to a hidden file or folder inside it are rejected. The checks apply to
 *   the file actually opened, so a folder swapped for a symlink mid-read is refused too. Where
 *   the OS can't report an open file's path (Node on macOS, Linux without `/proc`, Windows), that
 *   recheck reads paths again and only narrows the race with a writer that keeps swapping. A hard
 *   link to a hidden file can't be told from a visible file and is read.
 * @returns The file text.
 * @throws {SkillError} If the path is unsafe, hidden, missing, not a regular file, larger than
 *   `current().maxSkillFileBytes`, not UTF-8 text, or changed while it was read.
 */
export function readSkillFile(skill: Skill, path: string, deps: SkillFileDeps = {}): string {
  const where = `skill ${reprText(skill.name)}: file ${reprText(path)}`;
  if (skill.directory === null) throw new SkillError(`skill ${reprText(skill.name)} has no bundled files`);
  // Empty and `.` segments drop out.
  const parts = path.split("/").filter((part) => part !== "" && part !== ".");
  if (path === "" || path.includes("\\") || path.includes("\0") || path.startsWith("/") || parts.includes("..")) {
    throw new SkillError(`${where} must be a relative path inside the skill directory`);
  }
  if (parts.some((part) => part.startsWith("."))) {
    throw new SkillError(`${where}: hidden files and folders are not readable`);
  }
  const root = resolvedPath(skill.directory);
  const target = resolvedPath(join(root, ...parts));
  checkInside(root, target, where);
  if (!statOrNull(target)?.isFile()) throw new SkillError(`${where} does not exist`);
  deps.beforeOpen?.(target);
  let fd: number;
  try {
    // Non-blocking, so a FIFO swapped in after the check can't hang the open.
    fd = openSync(target, constants.O_RDONLY | constants.O_NONBLOCK);
  } catch (error) {
    if (isMissingPath(error)) throw new SkillError(`${where} does not exist`);
    throw error;
  }
  try {
    // Another process can swap a checked folder for a symlink before the open, so check the file
    // the descriptor holds: by the path the OS reports for it where it can, else by resolving
    // the path again and matching the file's identity. That fallback reads paths too, so a writer
    // that keeps swapping can still win it, just far less often.
    const opened = fstatSync(fd, { bigint: true });
    if (!opened.isFile()) throw new SkillError(`${where} does not exist`);
    const real = (deps.openedPath ?? openFilePath)(fd) ?? resolvedPath(target);
    checkInside(root, real, where);
    let now;
    try {
      now = statSync(real, { bigint: true });
    } catch (error) {
      if (isMissingPath(error)) throw new SkillError(`${where} does not exist`);
      throw error;
    }
    if (now.dev !== opened.dev || now.ino !== opened.ino) throw new SkillError(`${where} changed while it was read`);
    return readOpenText(fd, where);
  } finally {
    closeSync(fd);
  }
}

/**
 * Refuse `real`, the real path of a file under skill directory `root`, when it leaves `root` or
 * passes through a hidden file or folder.
 */
function checkInside(root: string, real: string, where: string): void {
  const rel = relative(root, real);
  if (leavesRoot(rel)) throw new SkillError(`${where} resolves outside the skill directory`);
  // A symlink with a visible name can still point at a hidden file or folder.
  if (rel.split(sep).some((part) => part.startsWith("."))) {
    throw new SkillError(`${where}: hidden files and folders are not readable`);
  }
}

/**
 * Read one bounded UTF-8 skill file, reading CRLF and lone CR line endings as LF.
 *
 * @param path File to read.
 * @param where Label for error messages; an empty or missing label means the path.
 * @returns The file text.
 * @throws {SkillError} If the file is larger than `current().maxSkillFileBytes` or not UTF-8 text.
 */
export function readSkillText(path: string, where: string | null = null): string {
  const fd = openSync(path, "r");
  try {
    return readOpenText(fd, where || path);
  } finally {
    closeSync(fd);
  }
}

/** Read open skill file `fd` as `readSkillText` does, labelling errors with `label`. */
function readOpenText(fd: number, label: string): string {
  const limit = current().maxSkillFileBytes;
  if (fstatSync(fd).size > limit) throw new SkillError(`${label} is larger than ${limit} bytes`);
  const bytes = readFileSync(fd);
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes);
  } catch (error) {
    if (error instanceof TypeError) throw new SkillError(`${label} is not UTF-8 text`);
    throw error;
  }
  // `\r\n` and `\r` read as `\n`.
  return text.replace(/\r\n?/g, "\n");
}
