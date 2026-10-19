/** The built-in `load_skill` tool and the system-message skill listing. */

import { current } from "../defaults.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { ToolSpec } from "../llm/messages.ts";
import { valueRepr } from "../typesys/values.ts";
import { type Skill, SkillError, readSkillFile } from "./catalog.ts";

export const LOAD_SKILL = "load_skill";

const LISTING_HEADER =
  "Skills: each skill below is a package of instructions. When a task matches a " +
  `skill's description, call \`${LOAD_SKILL}\` with its name before acting and follow ` +
  "the instructions it returns. When those instructions mention a file, read it " +
  `with \`${LOAD_SKILL}\` using the same name and the file's relative path.`;
const TOOL_DESCRIPTION =
  "Load a granted skill. Without `path`, returns the skill's SKILL.md " +
  "instructions. With `path`, returns the UTF-8 text of a file inside the " +
  "skill's directory, given relative to it (for example `references/api.md`).";

/**
 * Append the granted skills' names and descriptions to a system prompt.
 *
 * @param base System prompt before skills.
 * @param skills Granted skills. An empty array returns `base` unchanged.
 * @returns `base`, a blank line, a short usage paragraph, and one `- name: description` line
 *   per skill sorted by name.
 */
export function systemPromptWithSkills(base: string, skills: readonly Skill[]): string {
  if (skills.length === 0) return base;
  const limit = current().maxSkillDescriptionChars;
  const lines = [base, "", LISTING_HEADER];
  for (const skill of [...skills].sort((a, b) => compareCodeUnits(a.name, b.name))) {
    let description = skill.description.split(/\s+/).filter(Boolean).join(" ");
    const points = Array.from(description);
    if (points.length > limit) description = points.slice(0, limit - 1).join("") + "…";
    lines.push(`- ${skill.name}: ${description}`);
  }
  return lines.join("\n");
}

/**
 * Return the `load_skill` tool definition bound to the model.
 *
 * @returns A tool named `load_skill` with arguments `name: string` and `path: string | null =
 *   null`. The agent loop serves it with `invokeLoadSkill`, never through the tool registry.
 */
export function loadSkillTool(): ToolSpec {
  return new ToolSpec({
    name: LOAD_SKILL,
    description: TOOL_DESCRIPTION,
    parameters: {
      type: "object",
      properties: {
        name: { type: "string" },
        path: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
      },
      required: ["name"],
    },
  });
}

/**
 * Run one model call of `load_skill` against the granted skills.
 *
 * @param skills Skills granted to this node, keyed by name.
 * @param args Model-supplied arguments: `name` and optional `path`.
 * @returns The `SKILL.md` instructions, or the requested bundled file text.
 * @throws {SkillError} For unexpected or mistyped arguments, a skill that is not granted, or an
 *   unsafe or unreadable path.
 */
export function invokeLoadSkill(skills: ReadonlyMap<string, Skill>, args: Readonly<Record<string, unknown>>): string {
  const extras = Object.keys(args)
    .filter((key) => key !== "name" && key !== "path")
    .sort(compareCodeUnits);
  if (extras.length > 0) throw new SkillError(`${LOAD_SKILL} got unexpected arguments: ${valueRepr(extras)}`);
  // Own keys only: an inherited property (such as one on the prototype) is not an argument.
  const name = Object.hasOwn(args, "name") ? args.name : undefined;
  const path = Object.hasOwn(args, "path") ? args.path : undefined;
  if (typeof name !== "string") throw new SkillError(`${LOAD_SKILL} needs a string \`name\``);
  if (path !== undefined && path !== null && typeof path !== "string") {
    throw new SkillError(`${LOAD_SKILL} \`path\` must be a string or null`);
  }
  const skill = skills.get(name);
  if (skill === undefined) throw new SkillError(`${LOAD_SKILL}: skill ${reprText(name)} is not granted to this node`);
  if (path === undefined || path === null) return skill.instructions;
  return readSkillFile(skill, path);
}
