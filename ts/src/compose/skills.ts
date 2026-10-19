/**
 * Read Codex/Claude-compatible `SKILL.md` directories into `Skill` values.
 *
 * Hosts call these helpers to build the `skills` mapping for a run. Frontmatter is parsed with
 * the Compose safe YAML literal parser; only `name` and `description` are read and every other
 * field is ignored.
 *
 * Knows about:   skills, compose/source, defaults, diagnostics, typesys.
 * Never imports: other Compose phases, compile, runtime, the run boundary, or CLI.
 */

import { readdirSync } from "node:fs";
import { userInfo } from "node:os";
import { dirname, join } from "node:path";

import { type Env, processEnv } from "../defaults.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import {
  checkSkillName,
  isSystemError,
  readSkillText,
  resolvedPath,
  SKILL_FILE,
  Skill,
  SkillError,
  statOrNull,
} from "../skills/catalog.ts";
import { isRecord, type ValueRecord } from "../typesys/index.ts";
import { parseYamlLiteral, SourceSyntaxError } from "./source/index.ts";

/**
 * Read one skill directory.
 *
 * The frontmatter `name` wins; without it the directory name is used, as Claude Code does.
 * Fields other than `name` and `description` are ignored.
 *
 * @param directory Directory containing `SKILL.md`. A symlinked directory is allowed; its
 *   resolved location bounds bundled-file reads.
 * @returns The parsed skill with its resolved directory.
 * @throws {SkillError} If `SKILL.md` is missing, too large, not UTF-8, or has missing or
 *   malformed frontmatter, name, or description.
 */
export function loadSkillDir(directory: string): Skill {
  const root = resolvedPath(directory);
  const path = join(root, SKILL_FILE);
  if (!statOrNull(path)?.isFile()) throw new SkillError(`${pathText(directory)}: missing ${SKILL_FILE}`);
  const [meta, body] = splitFrontmatter(readSkillText(path), path);
  let name: string;
  try {
    name = checkSkillName(Object.hasOwn(meta, "name") ? meta.name : pathName(directory));
  } catch (error) {
    if (error instanceof SkillError) throw new SkillError(`${path}: ${error.message}`);
    throw error;
  }
  const description = Object.hasOwn(meta, "description") ? meta.description : undefined;
  if (typeof description !== "string" || description.trim() === "") {
    throw new SkillError(`${path}: frontmatter needs a non-empty \`description\``);
  }
  return new Skill({ name, description: description.trim(), instructions: body, directory: root });
}

/**
 * Index every skill under ordered roots; an earlier root wins a name.
 *
 * Each root holds skill directories (`<root>/<name>/SKILL.md`). Missing roots, hidden entries,
 * and subdirectories without `SKILL.md` are ignored. A name found again in a later root is
 * shadowed. Two skills with one name in the same root are an error.
 *
 * @param roots Skill roots, highest precedence first.
 * @param options `skipInvalid` reports invalid or unreadable skills through `warn` (default
 *   `process.emitWarning`) and skips them instead of throwing.
 * @returns Skills keyed by name, sorted by name.
 * @throws {SkillError} If `skipInvalid` is false and any skill is invalid, unreadable, or
 *   duplicated within one root. The message lists every problem.
 *
 * @example
 * const skills = loadSkills(["team/skills", "shared/skills"]);
 */
export function loadSkills(
  roots: Iterable<string>,
  { skipInvalid = false, warn = (message: string) => process.emitWarning(message) }: {
    skipInvalid?: boolean;
    warn?: (message: string) => void;
  } = {},
): Map<string, Skill> {
  const found = new Map<string, Skill>();
  const problems: string[] = [];
  for (const root of roots) {
    let entries: string[];
    try {
      entries = statOrNull(root)?.isDirectory() ? readdirSync(root).sort(compareCodeUnits) : [];
    } catch (error) {
      if (!isSystemError(error)) throw error;
      problems.push(`${pathText(root)}: cannot read skill root: ${error.message}`);
      continue;
    }
    const inRoot = new Map<string, string>();
    for (const name of entries) {
      if (name.startsWith(".")) continue;
      const entry = join(root, name);
      let skill: Skill;
      try {
        if (!statOrNull(join(entry, SKILL_FILE))?.isFile()) continue;
        skill = loadSkillDir(entry);
      } catch (error) {
        if (error instanceof SkillError) problems.push(error.message);
        else if (isSystemError(error)) problems.push(`${entry}: cannot read skill: ${error.message}`);
        else throw error;
        continue;
      }
      const first = inRoot.get(skill.name);
      if (first !== undefined) {
        problems.push(`${pathText(root)}: skill name ${reprText(skill.name)} is used by both ${first}/ and ${name}/`);
        continue;
      }
      inRoot.set(skill.name, name);
      if (!found.has(skill.name)) found.set(skill.name, skill);
    }
  }
  if (problems.length > 0 && !skipInvalid) throw new SkillError("invalid skills:\n- " + problems.join("\n- "));
  for (const problem of problems) warn(`skipped skill: ${problem}`);
  return new Map([...found].sort(([a], [b]) => compareCodeUnits(a, b)));
}

/**
 * Return the Claude Code and Codex skill roots, highest precedence first.
 *
 * User roots come first (Claude Code ranks personal over project), then project roots from
 * `cwd` up to the enclosing git root, nearest first. At each level the Claude root precedes the
 * Codex root.
 *
 * @param options `cwd` is the project directory (default the process working directory);
 *   `home` the user home (default `HOME` from `env`, else the account's home directory).
 * @returns `~/.claude/skills`, `~/.agents/skills`, then `<dir>/.claude/skills` and
 *   `<dir>/.agents/skills` for each project level.
 */
export function defaultSkillRoots({
  cwd,
  home,
  env = processEnv,
}: { cwd?: string; home?: string; env?: Env } = {}): string[] {
  const homeDir = home ?? env("HOME") ?? userInfo().homedir;
  const start = resolvedPath(cwd ?? process.cwd());
  const chain = [start];
  for (let parent = dirname(start); parent !== chain.at(-1); parent = dirname(parent)) chain.push(parent);
  // `statOrNull`, like Python's `exists()`, throws on errors other than a missing path.
  const depth = chain.findIndex((level) => statOrNull(join(level, ".git")) !== null);
  const levels = depth === -1 ? [start] : chain.slice(0, depth + 1);
  const roots = [join(homeDir, ".claude", "skills"), join(homeDir, ".agents", "skills")];
  for (const level of levels) roots.push(join(level, ".claude", "skills"), join(level, ".agents", "skills"));
  return roots;
}

/** Split `---` YAML frontmatter from the Markdown body. */
function splitFrontmatter(text: string, path: string): [ValueRecord, string] {
  // Lines keep their endings; `readSkillText` already read CRLF and CR as LF.
  const lines = text.split(/(?<=\n)/);
  if (lines[0]!.trim() !== "---") throw new SkillError(`${path}: must start with \`---\` YAML frontmatter`);
  const index = lines.findIndex((line, at) => at > 0 && line.trim() === "---");
  if (index === -1) throw new SkillError(`${path}: frontmatter has no closing \`---\``);
  let meta: unknown;
  try {
    meta = parseYamlLiteral(lines.slice(1, index).join("")) ?? {};
  } catch (error) {
    if (error instanceof SourceSyntaxError) {
      throw new SkillError(`${path}: frontmatter is not valid YAML: ${error.message}`);
    }
    throw error;
  }
  if (!isRecord(meta)) throw new SkillError(`${path}: frontmatter must be a mapping`);
  return [meta, lines.slice(index + 1).join("").replace(/^\n+/, "")];
}

/** The parts of `directory` Python's `Path` keeps: empty and `.` segments drop out. */
function pathParts(directory: string): string[] {
  return directory.split("/").filter((part) => part !== "" && part !== ".");
}

/** `directory` as Python's `str(Path(directory))` writes it. */
function pathText(directory: string): string {
  return (directory.startsWith("/") ? "/" : "") + pathParts(directory).join("/") || ".";
}

/** `directory`'s last part, as Python's `Path(directory).name`. */
function pathName(directory: string): string {
  return pathParts(directory).at(-1) ?? "";
}
