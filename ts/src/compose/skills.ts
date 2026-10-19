/**
 * Read Codex/Claude-compatible `SKILL.md` directories into `Skill` values.
 *
 * Hosts call these helpers to build the `skills` mapping for a run. Frontmatter is parsed with
 * the Compose safe YAML literal parser; only `name` and `description` are read and every other
 * field is ignored.
 *
 * Knows about:   skills, compose/source.
 * Never imports: other Compose phases, compile, runtime, the run boundary, or CLI.
 */

import { join } from "node:path";

import {
  checkSkillName,
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

/** Split `---` YAML frontmatter from the Markdown body. */
function splitFrontmatter(text: string, path: string): [ValueRecord, string] {
  // Lines keep their endings, as Python's `splitlines(keepends=True)`; the text is LF-only.
  const lines = text.split(/(?<=\n)/);
  if (lines[0]!.trim() !== "---") throw new SkillError(`${path}: must start with \`---\` YAML frontmatter`);
  const index = lines.findIndex((line, at) => at > 0 && line.trim() === "---");
  if (index === -1) throw new SkillError(`${path}: frontmatter has no closing \`---\``);
  let meta;
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
