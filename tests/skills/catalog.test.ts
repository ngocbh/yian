/** SKILL.md parsing, root precedence, and diagnostics. */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadSkillDir, loadSkills } from "../../src/compose/skills.ts";
import { current } from "../../src/defaults.ts";
import { Skill, SkillError } from "../../src/skills/index.ts";

let tmp: string;
/** Paths a test made unreadable, given their modes back before the temp dir is removed. */
let locked: string[];
const asRoot = process.getuid?.() === 0;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-skills-"));
  locked = [];
});

afterEach(() => {
  for (const path of locked) chmodSync(path, 0o755);
  rmSync(tmp, { recursive: true, force: true });
});

/** Create `${root}/${dirname}/SKILL.md` and return the skill directory. */
function writeSkill(root: string, dirname: string, frontmatter: string, body = "Do the thing.\n"): string {
  const directory = join(root, dirname);
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, "SKILL.md"), `---\n${frontmatter}---\n\n${body}`);
  return directory;
}

/** Write `content` as `${tmp}/${dirname}/SKILL.md` and return the skill directory. */
function writeRaw(dirname: string, content: string): string {
  const directory = join(tmp, dirname);
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, "SKILL.md"), content);
  return directory;
}

/** Make `path` unreadable until the test ends. */
function lock(path: string): void {
  chmodSync(path, 0);
  locked.push(path);
}

/** Return what `run` throws, asserting it is a `SkillError` whose message contains `text`. */
function expectSkillError(run: () => unknown, text: string): SkillError {
  let caught: unknown;
  try {
    run();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(SkillError);
  expect((caught as SkillError).message).toContain(text);
  return caught as SkillError;
}

test("reads frontmatter and body", () => {
  const dir = writeSkill(
    tmp,
    "pdf-tools",
    "name: pdf-tools\ndescription: Extract text from PDFs.\nlicense: MIT\nallowed-tools: Bash\n",
    "# PDF\n\nRun `scripts/extract.py`.\n",
  );
  expect(loadSkillDir(dir)).toEqual(
    new Skill({
      name: "pdf-tools",
      description: "Extract text from PDFs.",
      instructions: "# PDF\n\nRun `scripts/extract.py`.\n",
      directory: realpathSync(dir),
    }),
  );
});

test("name defaults to the directory and frontmatter name wins", () => {
  const review = writeSkill(tmp, "review", "description: Review code.\n");
  expect(loadSkillDir(review).name).toBe("review");
  expect(loadSkillDir(writeSkill(tmp, "x", "name: triage\ndescription: Triage.\n")).name).toBe("triage");
  // A trailing `/` or `/.` is dropped before taking the name.
  expect(loadSkillDir(review + "/.").name).toBe("review");
  expect(loadSkillDir(review + "/").name).toBe("review");
});

test("frontmatter reads yes and no as text", () => {
  // Frontmatter follows the flow's YAML 1.2 rules, as Claude Code and Codex read it.
  const skill = loadSkillDir(writeSkill(tmp, "no", "name: no\ndescription: yes\n"));
  expect([skill.name, skill.description]).toEqual(["no", "yes"]);
});

describe("bad skill files name the file", () => {
  test.each([
    ["no frontmatter\n", "must start with `---`"],
    ["---\nname: a\n", "no closing `---`"],
    ["---\n: [\n---\nbody\n", "not valid YAML"],
    ["---\n- a\n---\nbody\n", "must be a mapping"],
    ["---\nname: a\n---\nbody\n", "non-empty `description`"],
    ["---\nname: Bad_Name\ndescription: d\n---\n", "invalid skill name"],
    ["---\nname: a--b\ndescription: d\n---\n", "invalid skill name"],
    ["---\nname: a\ndescription: d\ndate: 2026-13-45\n---\nbody\n", "not a valid date"],
    // The literal reader's refusals, and frontmatter or a name YAML reads as a non-text value.
    ["---\n1: x\n---\nbody\n", "mapping keys must be text"],
    ["---\na: !!binary aGk=\n---\nbody\n", "could not determine a constructor"],
    ["---\nfalse\n---\nbody\n", "frontmatter must be a mapping"],
    ["---\nname: ~\ndescription: d\n---\nbody\n", "invalid skill name null"],
  ])("%p", (content, message) => {
    const dir = writeRaw("a", content);
    const error = expectSkillError(() => loadSkillDir(dir), message);
    expect(error.message).toContain("SKILL.md");
  });
});

test("empty frontmatter, CRLF and a BOM", () => {
  const empty = writeRaw("review", "---\n---\n");
  expectSkillError(() => loadSkillDir(empty), "needs a non-empty `description`");
  const crlf = writeRaw("c", "---\r\nname: c\r\ndescription: d\r\n---\r\n\r\nbody\r\n");
  expect(loadSkillDir(crlf).instructions).toBe("body\n");
  const bom = writeRaw("bom", "\uFEFF---\ndescription: d\n---\nbody\n");
  expect(loadSkillDir(bom).name).toBe("bom");
});

test("missing and oversized skill files", () => {
  const empty = join(tmp, "empty");
  mkdirSync(empty);
  for (const given of [empty, empty + "/"]) {
    expect(expectSkillError(() => loadSkillDir(given), "missing").message).toBe(`${empty}: missing SKILL.md`);
  }
  const big = writeSkill(tmp, "big", "description: d\n", "x".repeat(current().maxSkillFileBytes));
  expectSkillError(() => loadSkillDir(big), "larger than");
});

test("earlier root wins and skips non-skills", () => {
  const first = join(tmp, "first");
  const second = join(tmp, "second");
  writeSkill(first, "review", "description: First review.\n");
  writeSkill(second, "review", "description: Second review.\n");
  writeSkill(second, "deploy", "description: Deploy.\n");
  mkdirSync(join(second, "notes"));
  writeSkill(second, ".hidden", "description: Hidden.\n");
  const skills = loadSkills([first, second, join(tmp, "missing")]);
  expect([...skills.keys()]).toEqual(["deploy", "review"]);
  expect(skills.get("review")!.description).toBe("First review.");
});

test("follows symlinked skill directories", () => {
  const real = writeSkill(join(tmp, "store"), "shared", "description: Shared.\n");
  const root = join(tmp, "root");
  mkdirSync(root);
  symlinkSync(real, join(root, "shared"));
  expect(loadSkills([root]).get("shared")!.directory).toBe(realpathSync(real));
});

test("reports every problem at once", () => {
  writeSkill(tmp, "one", "name: same\ndescription: One.\n");
  writeSkill(tmp, "two", "name: same\ndescription: Two.\n");
  writeSkill(tmp, "bad", "name: bad\n");
  const { message } = expectSkillError(() => loadSkills([tmp]), "'same' is used by both one/ and two/");
  expect(message.startsWith("invalid skills:\n- ")).toBe(true);
  expect(message).toContain("bad");
  expect(message).toContain("description");
});

test("skipInvalid warns and keeps the rest", () => {
  writeSkill(tmp, "good", "description: Good.\n");
  writeSkill(tmp, "bad", "name: bad\n");
  const file = join(realpathSync(tmp), "bad", "SKILL.md");
  const expected = `skipped skill: ${file}: frontmatter needs a non-empty \`description\``;
  const warnings: string[] = [];
  const skills = loadSkills([tmp], { skipInvalid: true, warn: (message) => warnings.push(message) });
  expect(warnings).toEqual([expected]);
  expect([...skills.keys()]).toEqual(["good"]);
  const emit = spyOn(process, "emitWarning").mockImplementation(() => {});
  try {
    expect([...loadSkills([tmp], { skipInvalid: true }).keys()]).toEqual(["good"]);
    expect(emit).toHaveBeenCalledTimes(1);
    expect(emit).toHaveBeenCalledWith(expected);
  } finally {
    emit.mockRestore();
  }
});

test.skipIf(asRoot)("reports unreadable skill files", () => {
  writeSkill(tmp, "good", "description: Good.\n");
  lock(join(writeSkill(tmp, "locked", "description: Locked.\n"), "SKILL.md"));
  expectSkillError(() => loadSkills([tmp]), "locked: cannot read skill");
  const warnings: string[] = [];
  const skills = loadSkills([tmp], { skipInvalid: true, warn: (message) => warnings.push(message) });
  expect([...skills.keys()]).toEqual(["good"]);
  expect(warnings).toHaveLength(1);
  expect(warnings[0]).toContain("locked: cannot read skill");
});

// Unreadable roots and directories, the sort order, and a non-path root.

test.skipIf(asRoot)("reports an unreadable root", () => {
  const root = join(tmp, "root");
  writeSkill(root, "good", "description: Good.\n");
  lock(root);
  expectSkillError(() => loadSkills([root]), `${root}: cannot read skill root:`);
});

test.skipIf(asRoot)("reports an unreadable skill directory", () => {
  writeSkill(tmp, "good", "description: Good.\n");
  lock(writeSkill(tmp, "locked", "description: Locked.\n"));
  expectSkillError(() => loadSkills([tmp]), "locked: cannot read skill: EACCES");
  const skills = loadSkills([tmp], { skipInvalid: true, warn: () => {} });
  expect([...skills.keys()]).toEqual(["good"]);
});

test("skills are sorted by name", () => {
  for (const name of ["zed", "123", "abc"]) writeSkill(tmp, name, "description: d\n");
  expect([...loadSkills([tmp]).keys()]).toEqual(["123", "abc", "zed"]);
});

test("a root that is not a path is not collected", () => {
  expect(() => loadSkills([5 as unknown as string])).toThrow(TypeError);
});
