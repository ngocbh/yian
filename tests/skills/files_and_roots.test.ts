/**
 * Bundled-file confinement and default Claude/Codex skill roots, a port of
 * `tests/skills/test_files_and_roots.py`; with `YIAN_TEST_NODE` set to a Node binary, the built
 * skills code on Node.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir, userInfo } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import * as compose from "../../src/compose/index.ts";
import { defaultSkillRoots, loadSkills } from "../../src/compose/skills.ts";
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

test("compose facade exports the skill loaders", () => {
  expect(compose.defaultSkillRoots).toBe(defaultSkillRoots);
  expect(compose.loadSkills).toBe(loadSkills);
});

// TS-only: the rows below pin behavior Python gets from pathlib.

test("default roots read the home from the injected env", () => {
  const roots = defaultSkillRoots({ cwd: tmp, env: (name) => (name === "HOME" ? "/h" : undefined) });
  expect(roots.slice(0, 2)).toEqual(["/h/.claude/skills", "/h/.agents/skills"]);
});

test("default roots treat an empty HOME as unset", () => {
  const roots = defaultSkillRoots({ cwd: tmp, env: () => "" });
  const homeDir = userInfo().homedir;
  expect(roots.slice(0, 2)).toEqual([join(homeDir, ".claude", "skills"), join(homeDir, ".agents", "skills")]);
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

/** The package root. */
const ROOT = resolve(import.meta.dirname, "../..");
const BUILT = join(ROOT, "dist/src/compose/skills.js");

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/engine/platform.test.ts`'s: importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!existsSync(BUILT)) throw new Error(`${BUILT} is missing; run bun run build first`);
  return node;
})();

test.skipIf(!NODE)(
  "the built skills code runs on Node",
  () => {
    const url = (file: string) => JSON.stringify(pathToFileURL(join(ROOT, file)).href);
    // `String.raw` leaves the `\n` and `\r\n` escapes for Node to read; `skillMd` uses `+` since `${}` would fill in here.
    const script = String.raw`
      import assert from "node:assert/strict";
      import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
      import { tmpdir } from "node:os";
      import { dirname, join } from "node:path";

      const { defaultSkillRoots, loadSkillDir, loadSkills } = await import(${url("dist/src/compose/skills.js")});
      const { readSkillFile } = await import(${url("dist/src/skills/catalog.js")});
      const { invokeLoadSkill } = await import(${url("dist/src/skills/tool.js")});

      const tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-skills-")));
      const write = (path, data) => {
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, data);
      };
      const skillMd = (name, description) => "---\nname: " + name + "\ndescription: " + description + "\n---\nUse " + name + ".\n";
      try {
        const api = join(tmp, "a", "api");
        write(join(api, "SKILL.md"), "---\r\nname: api\r\ndescription: Call the API.\r\n---\r\nRead references/api.md.\r\n");
        const loaded = loadSkillDir(api);
        assert.equal(loaded.name, "api");
        assert.equal(loaded.description, "Call the API.");
        assert.equal(loaded.instructions, "Read references/api.md.\n");

        write(join(tmp, "elsewhere", "linked", "SKILL.md"), skillMd("linked", "Linked."));
        symlinkSync(join(tmp, "elsewhere", "linked"), join(tmp, "a", "linked"));
        write(join(tmp, "a", ".hidden", "SKILL.md"), skillMd("hidden", "Hidden."));
        write(join(tmp, "b", "api", "SKILL.md"), skillMd("api", "Shadowed."));
        write(join(tmp, "b", "beta", "SKILL.md"), skillMd("beta", "Beta."));
        const skills = loadSkills([join(tmp, "a"), join(tmp, "b")]);
        assert.ok(skills instanceof Map);
        assert.deepEqual([...skills.keys()], ["api", "beta", "linked"]);
        assert.equal(skills.get("api").description, "Call the API.");

        write(join(api, "references", "api.md"), "GET /items\n");
        write(join(tmp, "secret.txt"), "top secret\n");
        symlinkSync(join(tmp, "secret.txt"), join(api, "leak.txt"));
        write(join(api, "blob.bin"), new Uint8Array([0xff, 0xfe, 0x00]));
        assert.equal(readSkillFile(loaded, "references/api.md"), "GET /items\n");
        assert.throws(() => readSkillFile(loaded, "leak.txt"), /outside/);
        assert.throws(() => readSkillFile(loaded, "blob.bin"), /not UTF-8/);
        assert.equal(invokeLoadSkill(skills, { name: "api" }), "Read references/api.md.\n");

        const repo = join(tmp, "outer", "repo");
        mkdirSync(join(repo, ".git"), { recursive: true });
        mkdirSync(join(repo, "sub"));
        const base = (dir) => [join(dir, ".claude", "skills"), join(dir, ".agents", "skills")];
        assert.deepEqual(defaultSkillRoots({ cwd: join(repo, "sub"), home: join(tmp, "home") }), [
          ...base(join(tmp, "home")),
          ...base(join(repo, "sub")),
          ...base(repo),
        ]);
        console.log("ok");
      } finally {
        rmSync(tmp, { recursive: true, force: true });
      }
    `;
    expect(execFileSync(NODE!, ["--input-type=module", "-e", script], { encoding: "utf8" })).toBe("ok\n");
  },
  30_000,
);
