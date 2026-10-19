/** The load_skill tool schema, dispatch, and system-message listing; a port of `tests/skills/test_tool.py`. */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { Skill, SkillError } from "../../src/skills/index.ts";
import { LOAD_SKILL, invokeLoadSkill, loadSkillTool, systemPromptWithSkills } from "../../src/skills/tool.ts";

let tmp: string;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-skills-")));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Assert that `run` throws a `SkillError` whose message contains `text`. */
function expectSkillError(run: () => unknown, text: string): void {
  let caught: unknown;
  try {
    run();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(SkillError);
  expect((caught as SkillError).message).toContain(text);
}

test("listing is sorted, one line per skill, and empty is identity", () => {
  const skills = [
    new Skill({ name: "zeta", description: "Last\n  one.", instructions: "" }),
    new Skill({ name: "alpha", description: "First.", instructions: "" }),
  ];
  const text = systemPromptWithSkills("BASE", skills);
  expect(text.startsWith("BASE\n\nSkills: ")).toBe(true);
  expect(text).toContain(`call \`${LOAD_SKILL}\``);
  expect(text.endsWith("\n- alpha: First.\n- zeta: Last one.")).toBe(true);
  expect(systemPromptWithSkills("BASE", [])).toBe("BASE");
  expect(skills.map((skill) => skill.name)).toEqual(["zeta", "alpha"]);
  expect(text.split("\n")[2]).toBe(
    "Skills: each skill below is a package of instructions. When a task matches a " +
      "skill's description, call `load_skill` with its name before acting and follow " +
      "the instructions it returns. When those instructions mention a file, read it " +
      "with `load_skill` using the same name and the file's relative path.",
  );
});

test("listing truncates long descriptions", () => {
  const long = new Skill({ name: "long", description: "x".repeat(2000), instructions: "" });
  expect(systemPromptWithSkills("B", [long]).split("\n").at(-1)).toBe("- long: " + "x".repeat(1023) + "…");
  // TS-only: the cap counts code points, so a surrogate pair is never split.
  const emoji = new Skill({ name: "long", description: "😀".repeat(2000), instructions: "" });
  expect(systemPromptWithSkills("B", [emoji]).split("\n").at(-1)).toBe("- long: " + "😀".repeat(1023) + "…");
});

test("tool schema is name and optional path", () => {
  const tool = loadSkillTool();
  expect(tool.name).toBe("load_skill");
  expect(tool.parameters).toEqual({
    type: "object",
    properties: {
      name: { type: "string" },
      path: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
    },
    required: ["name"],
  });
  expect(tool.strict).toBeNull();
  expect(tool.description).toBe(
    "Load a granted skill. Without `path`, returns the skill's SKILL.md " +
      "instructions. With `path`, returns the UTF-8 text of a file inside the " +
      "skill's directory, given relative to it (for example `references/api.md`).",
  );
});

test("invoke returns instructions or file", () => {
  writeFileSync(join(tmp, "notes.md"), "note\n");
  const granted = new Map([
    ["notes", new Skill({ name: "notes", description: "Take notes.", instructions: "Use notes.md.", directory: tmp })],
  ]);
  expect(invokeLoadSkill(granted, { name: "notes" })).toBe("Use notes.md.");
  expect(invokeLoadSkill(granted, { name: "notes", path: null })).toBe("Use notes.md.");
  expect(invokeLoadSkill(granted, { name: "notes", path: "notes.md" })).toBe("note\n");
});

test("invoke rejects bad arguments", () => {
  const granted = new Map([["notes", new Skill({ name: "notes", description: "d", instructions: "" })]]);
  const rows: [Record<string, unknown>, string][] = [
    [{ name: "other" }, "not granted"],
    [{}, "string `name`"],
    [{ name: 3 }, "string `name`"],
    [{ name: "notes", path: 3 }, "`path` must be"],
    [{ name: "notes", extra: 1 }, "unexpected arguments"],
  ];
  for (const [args, message] of rows) expectSkillError(() => invokeLoadSkill(granted, args), message);
  expectSkillError(
    () => invokeLoadSkill(granted, { name: "notes", extra: 1, a: 2 }),
    'load_skill got unexpected arguments: ["a","extra"]',
  );
  expectSkillError(
    () => invokeLoadSkill(granted, { name: "other" }),
    "load_skill: skill 'other' is not granted to this node",
  );
  // TS-only: only own keys are arguments; an inherited `name` is not one, an own `__proto__` is.
  expectSkillError(() => invokeLoadSkill(granted, Object.create({ name: "notes" })), "needs a string `name`");
  expectSkillError(
    () => invokeLoadSkill(granted, JSON.parse('{"name": "notes", "__proto__": 1}')),
    'unexpected arguments: ["__proto__"]',
  );
});
