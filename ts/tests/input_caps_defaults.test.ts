/**
 * Input-size caps come from `defaults`; mirrors `tests/test_input_caps_defaults.py`.
 *
 * Holds the base URL, skill, question option and auth store rows; the other rows come with their
 * modules.
 */

import { afterEach, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadSkillDir } from "../src/compose/skills.ts";
import { BUILTIN, install } from "../src/defaults.ts";
import { AuthError } from "../src/llm_clients/auth/common.ts";
import { AuthStore } from "../src/llm_clients/auth/store.ts";
import { validateBaseUrl } from "../src/llm_clients/config.ts";
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

test("base url cap", () => {
  const url = "https://a.example/v1";
  expect(url.length).toBe(20);
  install({ ...BUILTIN, maxBaseUrlChars: 20 });
  expect(validateBaseUrl(url)).toBe(url);
  expect(() => validateBaseUrl(`${url}x`)).toThrow(RangeError);
  expect(() => validateBaseUrl(`${url}x`)).toThrow("at most 20 characters");
});

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

test("auth store cap", () => {
  const tmp = mkdtempSync(join(tmpdir(), "yian-auth-"));
  try {
    const directory = join(tmp, "auth");
    mkdirSync(directory, { mode: 0o700 });
    const file = join(directory, "auth.json");
    const entry = {
      method: "api_key",
      model: "claude-x",
      credential_id: "c".repeat(32),
      credentials: { api_key: "sk-saved" },
    };
    writeFileSync(file, JSON.stringify({ version: 1, default_provider: "anthropic", providers: { anthropic: entry } }));
    chmodSync(file, 0o600);
    expect(new AuthStore(file).read().selections).not.toEqual([]);
    install({ ...BUILTIN, maxAuthStoreBytes: 10 });
    expect(() => new AuthStore(file).read()).toThrow(AuthError);
    expect(() => new AuthStore(file).read()).toThrow("Credential storage exceeds its size limit");
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});
