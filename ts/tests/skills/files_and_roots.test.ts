/**
 * Bundled-file confinement and default Claude/Codex skill roots, a port of
 * `tests/skills/test_files_and_roots.py`.
 *
 * Later: the compose facade test (with the compose loaders).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { defaultSkillRoots } from "../../src/compose/skills.ts";
import { current } from "../../src/defaults.ts";
import { readSkillFile, readSkillText, resolvedPath } from "../../src/skills/catalog.ts";
import { Skill, SkillError } from "../../src/skills/index.ts";

let tmp: string;
let root: string;
let skill: Skill;

const HIDDEN = [".env", ".git/config", "references/.secret"];

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-skills-")));
  root = join(tmp, "skills", "api");
  mkdirSync(join(root, "references"), { recursive: true });
  writeFileSync(join(root, "references", "api.md"), "GET /items\n");
  writeFileSync(join(tmp, "secret.txt"), "top secret\n");
  skill = new Skill({
    name: "api",
    description: "Call the API.",
    instructions: "Read references/api.md.",
    directory: root,
  });
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

test("skill value validates name and description", () => {
  expectSkillError(() => new Skill({ name: "Has Space", description: "d", instructions: "" }), "invalid skill name");
  expectSkillError(() => new Skill({ name: "ok", description: "  ", instructions: "" }), "description");
  // TS-only: values the type system can't keep out of a JS caller.
  const loose = (options: Record<string, unknown>) =>
    new Skill({ name: "ok", description: "d", instructions: "", ...options } as never);
  expectSkillError(() => loose({ name: 5 }), "invalid skill name 5: use 1-64 lowercase letters");
  expectSkillError(() => loose({ instructions: 3 }), "skill 'ok': instructions must be a string");
  expectSkillError(() => loose({ directory: 3 }), "skill 'ok': directory must be a string or null");
  const made = new Skill({ name: "ok", description: "d", instructions: "" });
  expect(made.directory).toBeNull();
  expect(Object.isFrozen(made)).toBe(true);
});

test("read skill file returns bundled text", () => {
  expect(readSkillFile(skill, "references/api.md")).toBe("GET /items\n");
  expect(readSkillFile(skill, "./references/api.md")).toBe("GET /items\n");
  expect(readSkillFile(skill, "references//api.md")).toBe("GET /items\n");
});

describe("read skill file rejects paths outside the skill", () => {
  test.each(["", "../../secret.txt", "references/../../../secret.txt", "/etc/passwd", "references\\api.md", "a\0b"])(
    "%p",
    (path) => {
      expectSkillError(() => readSkillFile(skill, path), "relative path inside the skill directory");
    },
  );
});

test("read skill file rejects symlink escape", () => {
  symlinkSync(join(tmp, "secret.txt"), join(root, "leak.txt"));
  expectSkillError(() => readSkillFile(skill, "leak.txt"), "outside the skill directory");
});

describe("read skill file rejects hidden files", () => {
  test.each(HIDDEN)("%p", (path) => {
    mkdirSync(join(root, ".git"));
    for (const hidden of HIDDEN) {
      writeFileSync(join(root, hidden), "TOKEN=1\n");
    }
    expectSkillError(() => readSkillFile(skill, path), "hidden files and folders are not readable");
  });
});

test("read skill file rejects missing, directories, binary and large", () => {
  writeFileSync(join(root, "blob.bin"), new Uint8Array([0xff, 0xfe, 0x00]));
  writeFileSync(join(root, "big.txt"), "x".repeat(current().maxSkillFileBytes + 1));
  expectSkillError(() => readSkillFile(skill, "references"), "does not exist");
  expectSkillError(() => readSkillFile(skill, "nope.md"), "does not exist");
  expectSkillError(() => readSkillFile(skill, "blob.bin"), "not UTF-8");
  expectSkillError(() => readSkillFile(skill, "big.txt"), "larger than");
  const inline = new Skill({ name: "inline", description: "d", instructions: "body" });
  expectSkillError(() => readSkillFile(inline, "x.md"), "no bundled files");
  expectSkillError(() => readSkillFile(skill, "."), "does not exist");
});

test("default roots put user before project and stop at git root", () => {
  const home = join(tmp, "home");
  const repo = join(tmp, "outer", "repo");
  const work = join(repo, "pkg", "sub");
  mkdirSync(work, { recursive: true });
  mkdirSync(join(repo, ".git"));
  const base = (dir: string) => [join(dir, ".claude", "skills"), join(dir, ".agents", "skills")];
  expect(defaultSkillRoots({ cwd: work, home })).toEqual([
    ...base(home),
    ...base(work),
    ...base(join(repo, "pkg")),
    ...base(repo),
  ]);
});

test("default roots outside git use only cwd", () => {
  const roots = defaultSkillRoots({ cwd: tmp, home: join(tmp, "home") });
  expect(roots.slice(2)).toEqual([join(tmp, ".claude", "skills"), join(tmp, ".agents", "skills")]);
});

// TS-only: the rows below pin behavior Python gets from pathlib.

test("default roots read the home from the injected env", () => {
  const roots = defaultSkillRoots({ cwd: tmp, env: (name) => (name === "HOME" ? "/h" : undefined) });
  expect(roots.slice(0, 2)).toEqual(["/h/.claude/skills", "/h/.agents/skills"]);
});

test("dangling links and link loops do not exist", () => {
  symlinkSync(join(tmp, "nowhere"), join(root, "gone.md"));
  symlinkSync("loop.md", join(root, "loop.md"));
  expectSkillError(() => readSkillFile(skill, "gone.md"), "does not exist");
  expectSkillError(() => readSkillFile(skill, "loop.md"), "does not exist");
});

test("line endings read as LF and a BOM is kept", () => {
  writeFileSync(join(root, "crlf.md"), "a\r\nb\rc");
  writeFileSync(join(root, "bom.md"), "\uFEFFhi");
  expect(readSkillFile(skill, "crlf.md")).toBe("a\nb\nc");
  expect(readSkillFile(skill, "bom.md")).toBe("\uFEFFhi");
});

test("read skill text labels errors with the path by default", () => {
  const path = join(root, "blob.bin");
  writeFileSync(path, new Uint8Array([0xff]));
  expectSkillError(() => readSkillText(path), `${path} is not UTF-8 text`);
  expectSkillError(() => readSkillText(path, ""), `${path} is not UTF-8 text`);
});

test("resolved path resolves a symlink above a missing path", () => {
  mkdirSync(join(tmp, "real"));
  symlinkSync(join(tmp, "real"), join(tmp, "link"));
  expect(resolvedPath(join(tmp, "link", "missing", "x"))).toBe(join(realpathSync(join(tmp, "real")), "missing", "x"));
});
