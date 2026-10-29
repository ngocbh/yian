/**
 * `bundledSkill`: the `composing-agents` skill `yian chat` carries is written once to a folder
 * under `~/.yian/cache/skills/` named by its content, and loaded from there.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import { SKILL } from "../../src/cli/bundled.ts";
import { bundledSkill } from "../../src/cli/bundled_skill.ts";
import { invokeLoadSkill } from "../../src/skills/tool.ts";

let home: string;

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-bundled-skill-"));
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** An environment whose `HOME` is the test's. */
const env = (name: string) => (name === "HOME" ? home : undefined);

test("the skill is written under ~/.yian/cache/skills, named by its content, and loads from there", () => {
  const skill = bundledSkill(env);
  expect(skill.name).toBe("composing-agents");
  const cache = join(home, ".yian", "cache", "skills");
  const folders = readdirSync(cache);
  expect(folders).toHaveLength(1);
  expect(folders[0]).toMatch(/^composing-agents-[0-9a-f]{16}$/);
  expect(skill.directory).toBe(join(cache, folders[0]!));
  for (const [path, text] of Object.entries(SKILL)) {
    expect([path, readFileSync(join(cache, folders[0]!, path), "utf8")]).toEqual([path, text]);
  }
});

test("a second call loads the folder the first wrote, without writing it again", () => {
  const first = bundledSkill(env);
  const marker = join(first.directory!, "templates", "minimal.yaml");
  writeFileSync(marker, "kept\n");
  const second = bundledSkill(env);
  expect(second.directory).toBe(first.directory);
  expect(readFileSync(marker, "utf8")).toBe("kept\n");
  // No temporary folder is left beside it.
  expect(readdirSync(join(home, ".yian", "cache", "skills"))).toHaveLength(1);
});

test("load_skill reads the skill's instructions and its reference.md", () => {
  const skill = bundledSkill(env);
  const skills = new Map([[skill.name, skill]]);
  expect(invokeLoadSkill(skills, { name: "composing-agents" })).toContain("# Writing yian flows");
  expect(invokeLoadSkill(skills, { name: "composing-agents", path: "reference.md" })).toBe(SKILL["reference.md"]!);
});

test("a folder another chat renames into place first is used, and the call's own is removed", () => {
  const first = bundledSkill(env);
  const folder = first.directory!;
  rmSync(folder, { recursive: true });
  const raced = bundledSkill(env, {
    // Another chat wins the race: its folder appears just before this call's rename.
    beforeRename: () => {
      bundledSkill(env);
      writeFileSync(join(folder, "theirs.txt"), "theirs\n");
    },
  });
  expect(raced.directory).toBe(folder);
  expect(readFileSync(join(folder, "theirs.txt"), "utf8")).toBe("theirs\n");
  expect(readdirSync(join(home, ".yian", "cache", "skills"))).toEqual([basename(folder)]);
});

test("a damaged folder whose SKILL.md doesn't load is written again", () => {
  const folder = bundledSkill(env).directory!;
  writeFileSync(join(folder, "SKILL.md"), "no frontmatter\n");
  const skill = bundledSkill(env);
  expect(skill.directory).toBe(folder);
  expect(readFileSync(join(folder, "SKILL.md"), "utf8")).toBe(SKILL["SKILL.md"]!);
  expect(readdirSync(join(home, ".yian", "cache", "skills"))).toEqual([basename(folder)]);
});
