/**
 * How the command line binds memory inputs: the default identity of an unbound input
 * (`<project key>/<flow id>/<input>`) and its location under `~/.yian/memory`, confined there;
 * `--memory NAME=PATH`; and memory inputs refused as values. Runs go through `main`, with `HOME`
 * a temporary folder.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { checkDefaultLocation, defaultIdentity, defaultLocation, projectKey } from "../../src/cli/memory_bind.ts";
import type { CliIo } from "../../src/cli/utils.ts";

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

/** Run `yian` with `args`, `HOME` the temporary home; return the exit code and both streams. */
async function yian(args: readonly string[], vars: Record<string, string> = {}) {
  const out: string[] = [];
  const err: string[] = [];
  const env: Record<string, string> = { HOME: home, ...vars };
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (Object.hasOwn(env, name) ? env[name] : undefined),
  };
  const code = await main([...args], io);
  return { code, stdout: out.join(""), stderr: err.join("") };
}

/** Write `text` as `<tmp>/proj/<name>` (the project folder made if missing); return its path. */
function flowFile(text: string, name = "flow.yaml"): string {
  const dir = join(tmp, "proj");
  mkdirSync(dir, { recursive: true });
  const path = join(dir, name);
  writeFileSync(path, text);
  return path;
}

/** A flow that appends a line to its memory's index. */
const APPEND = `id: answer
name: answer
input:
  lessons: !FileMemory
nodes:
  add:
    kind: tool
    tool_id: {lessons: append}
    input:
      path: MEMORY.md
      text: Check sums twice.
output: \${add.output.path}
`;

/** A flow whose two memory inputs each append a file. */
const TWO = `id: two
name: two
input:
  lessons: !FileMemory
  notes: !FileMemory
nodes:
  a:
    kind: tool
    tool_id: {lessons: append}
    input: {path: a.md, text: from lessons}
  b:
    kind: tool
    tool_id: {notes: append}
    depends_on: [a]
    input: {path: b.md, text: from notes}
output: \${b.output.path}
`;

/** The run folders under the temporary home. */
function runFolders(): string[] {
  const runs = join(home, ".yian", "runs");
  return existsSync(runs) ? readdirSync(runs).filter((name) => name !== "index.sqlite") : [];
}

describe("--memory", () => {
  test("binds a memory input to a folder, which the run creates", async () => {
    const notes = join(tmp, "notes");
    const result = await yian(["run", flowFile(APPEND), "--quiet", "--memory", `lessons=${notes}`]);
    expect(result.code, result.stderr).toBe(0);
    expect(result.stdout).toBe("MEMORY.md\n");
    expect(readFileSync(join(notes, "MEMORY.md"), "utf8")).toContain("Check sums twice.");
    // A memory given a path and no identity takes one from its real path, never the path itself.
    expect(readFileSync(join(notes, ".yian", "identity"), "utf8")).toMatch(/^at-[0-9a-f]{16}$/);
  });

  test("two inputs given one path, however spelled, share one memory", async () => {
    const shared = join(tmp, "shared");
    const args = ["--memory", `lessons=${shared}`, "--memory", `notes=${join(tmp, "proj", "..", "shared")}/`];
    const result = await yian(["run", flowFile(TWO), "--quiet", ...args]);
    expect(result.code, result.stderr).toBe(0);
    expect(readFileSync(join(shared, "a.md"), "utf8")).toContain("from lessons");
    expect(readFileSync(join(shared, "b.md"), "utf8")).toContain("from notes");
  });

  test("a flag naming no memory input, a .db for a FileMemory, a repeat or a bad form is a usage error", async () => {
    const flow = flowFile(TWO);
    const cases: [string[], string][] = [
      [["--memory", "nope=x"], "the flow has no memory input `nope`"],
      [["--memory", "lessons=x.db"], "`lessons` is a FileMemory, a folder, not a .db file"],
      [["--memory", "lessons=a", "--memory", "lessons=b"], "`lessons` is given twice"],
      [["--memory", "lessons"], "--memory must be NAME=PATH (got 'lessons')"],
      [["--memory", "lessons="], "`lessons` needs a path"],
    ];
    for (const [flags, message] of cases) {
      const result = await yian(["run", flow, "--quiet", ...flags]);
      expect(result.code, result.stderr).toBe(2);
      expect(result.stderr).toContain("--memory");
      expect(result.stderr).toContain(message);
    }
    expect(runFolders()).toEqual([]);
  });

  test("a flag naming an input that isn't a memory is a usage error", async () => {
    const flow = flowFile(APPEND.replace("  lessons: !FileMemory\n", "  lessons: !FileMemory\n  topic: !str\n"));
    const result = await yian(["run", flow, "--quiet", "--input", "topic=x", "--memory", "topic=x"]);
    expect(result.code).toBe(2);
    expect(result.stderr).toContain("Invalid value for --memory: input `topic` isn't a memory");
  });

  test("a memory input given as a value is refused: bind it with --memory", async () => {
    const flow = flowFile(APPEND);
    const given = await yian(["run", flow, "--quiet", "--input", "lessons=x"]);
    expect(given.code).toBe(2);
    expect(given.stderr).toContain("`lessons` is a memory; bind it with --memory");
    const inputs = join(tmp, "inputs.json");
    writeFileSync(inputs, '{"lessons": 1}');
    const fromFile = await yian(["run", flow, "--quiet", "--inputs", inputs]);
    expect(fromFile.code).toBe(2);
    expect(fromFile.stderr).toContain("`lessons` is a memory; bind it with --memory");
    expect(runFolders()).toEqual([]);
  });

  test("a VectorDBMemory input can't be bound from the command line", async () => {
    const flow = flowFile(`id: cases
name: cases
input:
  cases: !VectorDBMemory
nodes:
  found:
    kind: tool
    tool_id: {cases: list}
    input: {}
output: \${found.output}
`);
    const result = await yian(["run", flow, "--quiet", "--memory", "cases=./c.db"]);
    expect(result.code).toBe(1);
    expect(result.stderr).toContain("`cases` is a VectorDBMemory; the command line binds only FileMemory inputs");
    expect(runFolders()).toEqual([]);
  });
});
