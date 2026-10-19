/**
 * Input-size caps come from `defaults`; mirrors `tests/test_input_caps_defaults.py`.
 *
 * Holds the skill and question option rows; the other rows come with their modules.
 */

import { afterEach, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadSkillDir } from "../src/compose/skills.ts";
import { BUILTIN, install } from "../src/defaults.ts";
import { checkSkillName } from "../src/skills/catalog.ts";
import { Skill, SkillError } from "../src/skills/index.ts";
import { systemPromptWithSkills } from "../src/skills/tool.ts";
import { normalizeOptions } from "../src/tools/registry.ts";

afterEach(() => install(BUILTIN));

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

test("skill file cap", () => {
  const tmp = mkdtempSync(join(tmpdir(), "yian-skills-"));
  try {
    const directory = join(tmp, "s");
    mkdirSync(directory);
    writeFileSync(join(directory, "SKILL.md"), "x".repeat(11));
    install({ ...BUILTIN, maxSkillFileBytes: 10 });
    expectSkillError(() => loadSkillDir(directory), "larger than 10 bytes");
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

test("skill name cap", () => {
  install({ ...BUILTIN, maxSkillNameChars: 3 });
  expect(checkSkillName("abc")).toBe("abc");
  expectSkillError(() => checkSkillName("abcd"), "use 1-3 lowercase");
});

test("skill description cap", () => {
  install({ ...BUILTIN, maxSkillDescriptionChars: 5 });
  const prompt = systemPromptWithSkills("base", [new Skill({ name: "a", description: "abcdefgh", instructions: "" })]);
  expect(prompt).toEndWith("- a: abcd…");
});

test("question option caps", () => {
  install({ ...BUILTIN, maxQuestionOptions: 2, maxQuestionOptionChars: 5 });
  expect(normalizeOptions(["abcdefg", "b", "c"])).toEqual(["abcd…", "b"]);
});
