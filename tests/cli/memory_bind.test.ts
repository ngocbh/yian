/**
 * How the command line binds memory inputs: the default identity of an unbound input
 * (`<project key>/<flow id>/<input>`) and its location under `~/.yian/memory`, confined there.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { checkDefaultLocation, defaultIdentity, defaultLocation, projectKey } from "../../src/cli/memory_bind.ts";

let tmp: string;
let home: string;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-memory-bind-")));
  home = join(tmp, "home");
  mkdirSync(join(home, ".yian", "memory"), { recursive: true });
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

describe("default identities", () => {
  test("a project key is the cleaned folder name and 8 hex digits of its real path", () => {
    expect(projectKey("/home/ada/My Project!")).toMatch(/^My-Project--[0-9a-f]{8}$/);
    expect(projectKey("/home/ada/...")).toMatch(/^project-[0-9a-f]{8}$/);
    expect(projectKey("/home/ada/.-x")).toMatch(/^x-[0-9a-f]{8}$/);
    expect(projectKey(`/x/${"a".repeat(40)}`)).toMatch(/^a{32}-[0-9a-f]{8}$/);
    // The hash tells two folders of one name apart.
    expect(projectKey("/a/proj")).not.toBe(projectKey("/b/proj"));
    expect(projectKey("/a/proj")).toBe(projectKey("/a/proj"));
  });

  test("a default identity needs a plain flow id", () => {
    expect(defaultIdentity("/p/proj", "answer-questions", "lessons")).toMatch(
      /^proj-[0-9a-f]{8}\/answer-questions\/lessons$/,
    );
    const refusal = "give the flow a plain `id:` (letters, digits, `.`, `_`, `-`) or bind `lessons` with --memory";
    expect(() => defaultIdentity("/p/proj", "Answer questions", "lessons")).toThrow(refusal);
    expect(() => defaultIdentity("/p/proj", ".hidden", "lessons")).toThrow(refusal);
    expect(() => defaultIdentity("/p/proj", null, "lessons")).toThrow(refusal);
  });
});

describe("default locations", () => {
  test("a FileMemory default is a folder; a VectorDBMemory default is <identity>.db", () => {
    const root = join(home, ".yian", "memory");
    expect(defaultLocation(home, "proj-1234abcd/f/lessons", "FileMemory")).toBe(join(root, "proj-1234abcd/f/lessons"));
    expect(defaultLocation(home, "proj-1234abcd/f/cases", "VectorDBMemory")).toBe(
      join(root, "proj-1234abcd/f/cases.db"),
    );
  });

  test("a symlinked component under ~/.yian/memory, or a path leaving it, is refused", () => {
    const outside = join(tmp, "outside");
    mkdirSync(outside);
    symlinkSync(outside, join(home, ".yian/memory/proj-1234abcd"));
    expect(() => checkDefaultLocation(home, "proj-1234abcd/f/lessons")).toThrow("is a symlink");
    expect(() => checkDefaultLocation(home, "../x/f/lessons")).toThrow("leaves ~/.yian/memory");
    expect(() => checkDefaultLocation(home, "a/../../x")).toThrow("leaves ~/.yian/memory");
    expect(() => checkDefaultLocation(home, ".")).toThrow("leaves ~/.yian/memory");
  });

  test("a file where a folder of the location should be is refused", () => {
    const root = join(home, ".yian", "memory");
    mkdirSync(join(root, "proj-1234abcd"));
    writeFileSync(join(root, "proj-1234abcd", "f"), "");
    expect(() => checkDefaultLocation(home, "proj-1234abcd/f/lessons")).toThrow("isn't a folder");
  });

  test("a location that doesn't exist yet, or under a ~/.yian/memory that doesn't, passes", () => {
    expect(checkDefaultLocation(home, "proj-1234abcd/f/lessons")).toBe(
      join(home, ".yian", "memory", "proj-1234abcd/f/lessons"),
    );
    const bare = join(tmp, "bare");
    expect(checkDefaultLocation(bare, "proj-1234abcd/f/lessons")).toBe(
      join(bare, ".yian", "memory", "proj-1234abcd/f/lessons"),
    );
  });

  test("~/.yian/memory itself may be a symlink: the location stays under where it points", () => {
    const elsewhere = join(tmp, "elsewhere");
    mkdirSync(elsewhere);
    const other = join(tmp, "other-home");
    mkdirSync(join(other, ".yian"), { recursive: true });
    symlinkSync(elsewhere, join(other, ".yian", "memory"));
    expect(checkDefaultLocation(other, "p/f/lessons")).toBe(join(other, ".yian", "memory", "p/f/lessons"));
  });
});
