/**
 * `bundledSkill`: the `composing-agents` skill `yian chat` carries is written once to a folder
 * under `~/.yian/cache/skills/` named by its content, and loaded from there.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import {
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import { SKILL } from "../../src/cli/bundled.ts";
import { bundledSkill } from "../../src/cli/bundled_skill.ts";
import { invokeLoadSkill } from "../../src/skills/tool.ts";

let home: string;

beforeEach(() => {
  // The real path: the skill's directory is resolved, and macOS's temp folder is behind a symlink.
  home = realpathSync(mkdtempSync(join(tmpdir(), "yian-bundled-skill-")));
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** An environment whose `HOME` is the test's. */
const env = (name: string) => (name === "HOME" ? home : undefined);

/** Expect every file of the bundled skill in `folder` to hold the bundle's text. */
function expectWhole(folder: string): void {
  for (const [path, text] of Object.entries(SKILL)) {
    expect([path, readFileSync(join(folder, path), "utf8")]).toEqual([path, text]);
  }
}

test("the skill is written under ~/.yian/cache/skills, named by its content, and loads", () => {
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
  // A file that isn't the skill's, which writing the folder again would remove.
  const marker = join(first.directory!, "marker.txt");
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
  expect(invokeLoadSkill(skills, { name: "composing-agents", path: "reference.md" })).toBe(
    SKILL["reference.md"]!,
  );
});

test("a folder another chat renames into place first is used, and this call's removed", () => {
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

test("a folder missing one of the skill's files is written again", () => {
  const folder = bundledSkill(env).directory!;
  rmSync(join(folder, "reference.md"));
  bundledSkill(env);
  expect(readFileSync(join(folder, "reference.md"), "utf8")).toBe(SKILL["reference.md"]!);
});

test("the folder and its files are the owner's alone", () => {
  const folder = bundledSkill(env).directory!;
  expect(statSync(folder).mode & 0o777).toBe(0o700);
  expect(statSync(join(folder, "templates")).mode & 0o777).toBe(0o700);
});

test("a folder whose files were emptied or cut short is written again", () => {
  const first = bundledSkill(env);
  const folder = first.directory!;
  writeFileSync(join(folder, "reference.md"), "");
  // SKILL.md keeps its frontmatter, so it still loads, but loses most of its body.
  const text = SKILL["SKILL.md"]!;
  writeFileSync(join(folder, "SKILL.md"), text.slice(0, Math.floor(text.length / 3)));
  const skill = bundledSkill(env);
  expect(skill.directory).toBe(folder);
  expect(skill.instructions).toBe(first.instructions);
  expectWhole(folder);
});

test("a folder with one of the skill's files changed by hand is written again", () => {
  const folder = bundledSkill(env).directory!;
  writeFileSync(join(folder, "templates", "minimal.yaml"), "changed\n");
  bundledSkill(env);
  expectWhole(folder);
});

test("a folder with a folder in place of one of the skill's files is written again", () => {
  const folder = bundledSkill(env).directory!;
  rmSync(join(folder, "templates", "minimal.yaml"));
  mkdirSync(join(folder, "templates", "minimal.yaml"));
  bundledSkill(env);
  expectWhole(folder);
});

test("a folder with a file in place of one of the skill's folders is written again", () => {
  const folder = bundledSkill(env).directory!;
  rmSync(join(folder, "templates"), { recursive: true });
  writeFileSync(join(folder, "templates"), "not a folder\n");
  bundledSkill(env);
  expectWhole(folder);
});
