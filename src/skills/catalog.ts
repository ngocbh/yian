/**
 * Skill values and confined reads of the files a skill bundles.
 *
 * A skill is a directory holding `SKILL.md` (YAML frontmatter with `name` and `description`,
 * then Markdown instructions) plus optional bundled files. Reading skill directories from disk
 * lives in `compose/skills.ts`, because only Compose parses YAML.
 */

import { readFileSync, realpathSync, statSync, type Stats } from "node:fs";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";

import { current } from "../defaults.ts";
import { reprText } from "../diagnostics.ts";
import { valueRepr } from "../typesys/values.ts";

export const SKILL_FILE = "SKILL.md";
const NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
// These errors mean the path isn't a file, rather than a failure.
const IGNORED_CODES = new Set(["ENOENT", "ENOTDIR", "ELOOP", "EBADF"]);

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
 * Return one UTF-8 file bundled with `skill`, confined to its directory.
 *
 * @param skill The granted skill.
 * @param path POSIX path relative to the skill directory, such as `references/api.md`. Absolute
 *   paths, `..`, backslashes, hidden (dot) files and folders, and symlinks that leave the
 *   directory are rejected.
 * @returns The file text.
 * @throws {SkillError} If the path is unsafe, hidden, missing, not a regular file, larger than
 *   `current().maxSkillFileBytes`, or not UTF-8 text.
 */
export function readSkillFile(skill: Skill, path: string): string {
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
  const rel = relative(root, target);
  if (rel === ".." || rel.startsWith(".." + sep) || isAbsolute(rel)) {
    throw new SkillError(`${where} resolves outside the skill directory`);
  }
  if (!statOrNull(target)?.isFile()) throw new SkillError(`${where} does not exist`);
  return readSkillText(target, where);
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
  const label = where || path;
  const limit = current().maxSkillFileBytes;
  if (statSync(path).size > limit) throw new SkillError(`${label} is larger than ${limit} bytes`);
  const bytes = readFileSync(path);
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

/**
 * @internal The real path of the longest existing prefix of `path`, with the rest appended unchanged.
 */
export function resolvedPath(path: string): string {
  const abs = resolve(path);
  let candidate = abs;
  for (;;) {
    try {
      const real = realpathSync(candidate);
      return join(real, relative(candidate, abs));
    } catch (error) {
      if (!isSystemError(error)) throw error;
      const parent = dirname(candidate);
      if (parent === candidate) return abs;
      candidate = parent;
    }
  }
}

/** @internal `statSync`, or `null` for the errors that mean the path isn't a file. */
export function statOrNull(path: string): Stats | null {
  try {
    return statSync(path);
  } catch (error) {
    if (isSystemError(error) && IGNORED_CODES.has(error.code)) return null;
    throw error;
  }
}

/** @internal Whether `error` is a Node SystemError (an OS error). */
export function isSystemError(error: unknown): error is NodeJS.ErrnoException & { code: string } {
  if (!(error instanceof Error)) return false;
  const { code, syscall, errno } = error as NodeJS.ErrnoException;
  return typeof code === "string" && (typeof syscall === "string" || typeof errno === "number");
}
