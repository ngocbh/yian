/**
 * `yian memory`: the memories under `~/.yian/memory`, and one named as `FLOW NAME` (an input's
 * default memory) or `--at PATH`. Memories are made by `yian run` on a temporary home, prompts
 * answered on a fake terminal, and the editor is injected.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { defaultIdentity, defaultLocation } from "../../src/cli/memory_bind.ts";
import { resolveTarget } from "../../src/cli/memory_cmd.ts";
import { type CliIo, UsageError } from "../../src/cli/utils.ts";
import { FileMemory } from "../../src/memory/file.ts";

let tmp: string;
let home: string;
let project: string;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-mem-cmd-")));
  home = join(tmp, "home");
  project = join(tmp, "proj");
  mkdirSync(home);
  mkdirSync(project);
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** The command's environment: the temporary home, and `vars`. */
function envOf(vars: Record<string, string> = {}): (name: string) => string | undefined {
  const values: Record<string, string> = { HOME: home, USER: "ada", ...vars };
  return (name) => (Object.hasOwn(values, name) ? values[name] : undefined);
}

/** An io over the temporary home whose streams are collected; `extra` adds to it. */
function testIo(extra: Partial<CliIo> = {}) {
  const out: string[] = [];
  const err: string[] = [];
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: envOf(),
    ...extra,
  };
  return { io, stdout: () => out.join(""), stderr: () => err.join("") };
}

/** Run `yian` with `args`; return the exit code and both streams. */
async function yian(args: readonly string[], extra: Partial<CliIo> = {}) {
  const { io, stdout, stderr } = testIo(extra);
  const code = await main([...args], io);
  return { code, stdout: stdout(), stderr: stderr() };
}

/** A flow that appends `text` (an input) to its memory's index. */
const APPEND = `id: answer
name: answer
input:
  line: !str
  lessons: !FileMemory
  notes: !Optional(FileMemory)
nodes:
  add:
    kind: tool
    tool_id: {lessons: append}
    input:
      path: MEMORY.md
      text: \${input.line}
output: \${add.output.path}
`;

/** Write `APPEND` as `<tmp>/proj/flow.yaml`; return its path. */
function flowFile(): string {
  const path = join(project, "flow.yaml");
  writeFileSync(path, APPEND);
  return path;
}

/** Run `flow` once, appending `line`; the run must succeed. */
async function runOnce(flow: string, line: string, ...flags: string[]): Promise<void> {
  const result = await yian(["run", flow, "--quiet", "--input", `line=${line}`, ...flags]);
  expect(result.code, result.stderr).toBe(0);
}

/** The default identity and folder of `flow.yaml`'s `lessons`. */
function lessons(): { identity: string; location: string } {
  const identity = defaultIdentity(project, "answer", "lessons");
  return { identity, location: defaultLocation(home, identity, "FileMemory") };
}

describe("the target", () => {
  test("FLOW NAME is the input's default memory, and --at PATH the memory in a folder", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const { io } = testIo();
    const named = await resolveTarget({ flow, name: "lessons" }, io);
    expect(named).toMatchObject({ location: lessons().location, identity: lessons().identity });
    expect((named as { memory: unknown }).memory).toBeInstanceOf(FileMemory);
    const at = await resolveTarget({ at: lessons().location }, io);
    expect(at).toMatchObject({ location: lessons().location, identity: null });
  });

  test("neither, both, a name that isn't a default memory, or a .db store is a usage error", async () => {
    const flow = flowFile();
    const { io } = testIo();
    const refusals: [Parameters<typeof resolveTarget>[0], string][] = [
      [{}, "give the memory as FLOW NAME or --at PATH"],
      [{ flow, name: "lessons", at: tmp }, "give the memory as FLOW NAME or --at PATH, not both"],
      [{ flow }, "Missing argument 'NAME'."],
      [{ flow, name: "line" }, "the flow has no memory input `line`"],
      [{ flow, name: "notes" }, "`notes` is Optional, so it has no default memory; name its folder with --at"],
      [{ at: join(tmp, "cases.db") }, "a .db store is a VectorDBMemory, which yian memory doesn't open yet"],
    ];
    for (const [given, message] of refusals) {
      const error = await resolveTarget(given, io).then(
        () => null,
        (thrown: unknown) => thrown,
      );
      expect(error, message).toBeInstanceOf(UsageError);
      expect((error as Error).message).toContain(message);
    }
  });
});

describe("yian memory list", () => {
  test("lists every memory under ~/.yian/memory with its type, identity and folder", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    // A memory elsewhere isn't under ~/.yian/memory, and isn't listed.
    await runOnce(flow, "second", "--memory", `lessons=${join(tmp, "elsewhere")}`);
    // A folder that isn't a memory isn't either.
    mkdirSync(join(home, ".yian", "memory", "loose"), { recursive: true });
    const text = await yian(["memory", "list"]);
    expect(text.code, text.stderr).toBe(0);
    const { identity, location } = lessons();
    expect(text.stdout.split("\n").filter(Boolean)).toEqual([
      expect.stringMatching(/^identity\s+type\s+folder$/),
      expect.stringMatching(new RegExp(`^${identity}\\s+FileMemory\\s+${location}$`)),
    ]);
    const json = await yian(["memory", "list", "--json"]);
    expect(JSON.parse(json.stdout)).toEqual({
      format: "yian.cli.v1",
      memories: [{ identity, type: "FileMemory", location }],
    });
  });

  test("with no memories, prints nothing, or an empty list", async () => {
    expect(await yian(["memory", "list"])).toEqual({ code: 0, stdout: "", stderr: "" });
    expect(JSON.parse((await yian(["memory", "list", "--json"])).stdout)).toEqual({
      format: "yian.cli.v1",
      memories: [],
    });
  });

  test("the memory group's help lists its commands", async () => {
    const help = await yian(["memory", "--help"]);
    expect(help.code).toBe(0);
    expect(help.stdout).toContain("list");
  });
});
