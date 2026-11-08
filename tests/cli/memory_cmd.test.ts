/**
 * `yian memory`: the memories under `~/.yian/memory`, and one named as `FLOW NAME` (an input's
 * default memory) or `--at PATH`. Memories are made by `yian run` on a temporary home, prompts
 * answered on a fake terminal, and the editor is injected.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { defaultIdentity, defaultLocation } from "../../src/cli/memory_bind.ts";
import { resolveTarget } from "../../src/cli/memory_cmd.ts";
import { type CliIo, UsageError } from "../../src/cli/utils.ts";
import { FileMemory } from "../../src/memory/file.ts";
import { DOWN, ENTER, FakeInput, FakeOutput } from "./terminal_both_runtimes.ts";

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

/** `yian memory history`'s document for `flow.yaml`'s `lessons`. */
async function historyOf(flow: string): Promise<{ changes: Record<string, any>[] }> {
  const result = await yian(["memory", "history", flow, "lessons", "--json"]);
  expect(result.code, result.stderr).toBe(0);
  return JSON.parse(result.stdout);
}

describe("yian memory show", () => {
  test("prints the identity, folder, counts and recall; --all every file", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    await runOnce(flow, "second");
    const { identity, location } = lessons();
    const shown = await yian(["memory", "show", flow, "lessons"]);
    expect(shown.code, shown.stderr).toBe(0);
    expect(shown.stdout).toStartWith(`${identity} (FileMemory) at ${location}: 1 file, 12 characters\n`);
    expect(shown.stdout).toContain("first\nsecond\n");
    expect(shown.stdout).not.toContain("== MEMORY.md");
    const all = await yian(["memory", "show", "--at", location, "--all"]);
    expect(all.stdout).toContain("== MEMORY.md (12 characters) ==\nfirst\nsecond\n");
    const json = JSON.parse((await yian(["memory", "show", flow, "lessons", "--all", "--json"])).stdout);
    expect(json).toMatchObject({
      identity,
      type: "FileMemory",
      location,
      characters: 12,
      recall: { text: "first\nsecond", cut: null },
      files: [{ path: "MEMORY.md", characters: 12, text: "first\nsecond" }],
    });
    expect(typeof json.version).toBe("string");
  });

  test("a memory not made yet is a usage error, and nothing is created", async () => {
    const flow = flowFile();
    const result = await yian(["memory", "show", flow, "lessons"]);
    expect(result.code).toBe(2);
    expect(result.stderr).toContain(`no memory at ${lessons().location}`);
    expect(existsSync(lessons().location)).toBe(false);
  });
});

describe("yian memory history", () => {
  test("lists the changes newest first, each with its author, run and node", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    await runOnce(flow, "second");
    const { changes } = await historyOf(flow);
    expect(changes.map((change) => [change.op, change.target.path])).toEqual([
      ["append", "MEMORY.md"],
      ["append", "MEMORY.md"],
    ]);
    const runs = readdirSync(join(home, ".yian", "runs")).filter((name) => name !== "index.sqlite");
    for (const change of changes) {
      expect(change.author).toMatchObject({ kind: "tool", name: "add", node: "add" });
      expect(runs).toContain(change.author.run);
    }
    expect(changes[0]!.author.run).not.toBe(changes[1]!.author.run);
    const text = await yian(["memory", "history", flow, "lessons"]);
    const lines = text.stdout.split("\n").filter(Boolean);
    expect(lines).toHaveLength(2);
    expect(lines[0]).toContain(`append MEMORY.md by tool add (run ${changes[0]!.author.run}, node add)`);
    expect(lines[0]).toStartWith(changes[0]!.id);
    expect((await yian(["memory", "history", flow, "lessons", "--limit", "1"])).stdout.split("\n").filter(Boolean))
      .toHaveLength(1);
  });
});

describe("yian memory undo", () => {
  test("undoes one change as the user and prints the new version; a conflict exits 1 with the reason", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    await runOnce(flow, "second");
    const [newest, oldest] = (await historyOf(flow)).changes;
    const undone = await yian(["memory", "undo", newest!.id, flow, "lessons"]);
    expect(undone.code, undone.stderr).toBe(0);
    expect(undone.stdout).toMatch(new RegExp(`^undid ${newest!.id}; the memory's version is [0-9a-f]{64}\\n$`));
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first");
    const [undo] = (await historyOf(flow)).changes;
    expect(undo).toMatchObject({ op: "undo", author: { kind: "user", name: "ada" } });
    expect(undo!.target.undoes).toBe(newest!.id);
    // The first append's file no longer holds only what it wrote.
    await runOnce(flow, "third");
    const conflict = await yian(["memory", "undo", oldest!.id, flow, "lessons", "--json"]);
    expect(conflict.code).toBe(1);
    expect(JSON.parse(conflict.stdout).error.code).toBe("memory.conflict");
    expect(conflict.stderr).toContain("undo refused: ");
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first\nthird");
  });
});

describe("yian memory edit", () => {
  /** An editor that changes the file it opens with `change`, as a person would. */
  const editing = (change: (text: string) => string) => async (path: string) =>
    writeFileSync(path, change(readFileSync(path, "utf8")));

  test("opens the index in the editor and applies the edit as the user, with the file's hash", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const editor = editing((text) => `${text}\nadded by hand`);
    const result = await yian(["memory", "edit", flow, "lessons"], { editor });
    expect(result.code, result.stderr).toBe(0);
    expect(result.stdout).toMatch(/^edited MEMORY\.md; the memory's version is [0-9a-f]{64}\n$/);
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first\nadded by hand");
    const [edit] = (await historyOf(flow)).changes;
    expect(edit).toMatchObject({ op: "rewrite", target: { path: "MEMORY.md" }, author: { kind: "user", name: "ada" } });
  });

  test("a file changed while the editor was open is a conflict; the edited text is kept", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const file = join(lessons().location, "MEMORY.md");
    const editor = async (path: string) => {
      // A run writes the memory while the person edits it.
      writeFileSync(file, "first\nfrom a run");
      writeFileSync(path, "first\nfrom the person");
    };
    const result = await yian(["memory", "edit", flow, "lessons", "--json"], { editor });
    expect(result.code).toBe(1);
    expect(JSON.parse(result.stdout).error.code).toBe("memory.conflict");
    const kept = /your text is in (.+)$/m.exec(result.stderr)![1]!;
    expect(readFileSync(kept, "utf8")).toBe("first\nfrom the person");
    expect(readFileSync(file, "utf8")).toBe("first\nfrom a run");
    rmSync(join(kept, ".."), { recursive: true });
  });

  test("--file names another file, made when it is new; an unchanged file changes nothing", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const made = await yian(["memory", "edit", flow, "lessons", "--file", "notes/tips.md"], {
      editor: editing(() => "Carry the one."),
    });
    expect(made.code, made.stderr).toBe(0);
    expect(readFileSync(join(lessons().location, "notes", "tips.md"), "utf8")).toBe("Carry the one.");
    expect((await historyOf(flow)).changes[0]).toMatchObject({ op: "create", target: { path: "notes/tips.md" } });
    const before = (await historyOf(flow)).changes.length;
    const same = await yian(["memory", "edit", flow, "lessons"], { editor: async () => {} });
    expect(same).toMatchObject({ code: 0, stdout: "MEMORY.md is unchanged\n" });
    expect((await historyOf(flow)).changes).toHaveLength(before);
  });

  test("the editor is $EDITOR, run on the file; with none set, edit says so", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    // Not an editor: a command that appends a line to the file it is given.
    const appendLine = `/bin/sh -c 'printf "%s" "
by EDITOR" >> "$0"'`;
    const result = await yian(["memory", "edit", flow, "lessons"], { env: envOf({ EDITOR: appendLine }) });
    expect(result.code, result.stderr).toBe(0);
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first\nby EDITOR");
    const none = await yian(["memory", "edit", flow, "lessons"]);
    expect(none.code).toBe(1);
    expect(none.stderr).toContain("set $EDITOR to the editor to use");
  });
});

/** Every file under `dir`, `.yian/` included, joined: what a search of the folder would read. */
function everything(dir: string): string {
  return readdirSync(dir, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => readFileSync(join(entry.parentPath, entry.name), "utf8"))
    .join("\n");
}

/** The run ids under the temporary home's runs folder. */
function runIds(): string[] {
  return readdirSync(join(home, ".yian", "runs")).filter((name) => name !== "index.sqlite");
}

describe("yian memory erase", () => {
  test("lists the runs that used the memory, erases the text everywhere, and prints the undos disabled", async () => {
    const flow = flowFile();
    await runOnce(flow, "keep this");
    await runOnce(flow, "my PIN 4321");
    const result = await yian(["memory", "erase", "MEMORY.md", flow, "lessons", "--text", "4321", "--yes"]);
    expect(result.code, result.stderr).toBe(0);
    for (const run of runIds()) expect(result.stdout).toContain(run);
    expect(result.stdout).toContain("yian can't see runs on other machines or in a host's own logs");
    expect(result.stdout).toMatch(/erased "4321" from MEMORY\.md; the memory's version is [0-9a-f]{64}/);
    expect(result.stdout).toMatch(/undos it disabled: \S+/);
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("keep this\nmy PIN ");
    expect(everything(lessons().location)).not.toContain("4321");
  });

  test("--json prints the runs, the version and the disabled undos", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const json = await yian(["memory", "erase", "MEMORY.md", flow, "lessons", "--yes", "--json"]);
    expect(json.code, json.stderr).toBe(0);
    const document = JSON.parse(json.stdout);
    expect(document.runs).toEqual([
      expect.objectContaining({ run_id: runIds()[0], status: "succeeded", log: expect.stringMatching(/log\.jsonl$/) }),
    ]);
    expect(document.version).toMatch(/^[0-9a-f]{64}$/);
    expect(document.disabled_undos).toHaveLength(1);
    expect(existsSync(join(lessons().location, "MEMORY.md"))).toBe(false);
  });

  test("asks first on a terminal; no, or no terminal without --yes, erases nothing", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const piped = await yian(["memory", "erase", "MEMORY.md", flow, "lessons"]);
    expect(piped.code).toBe(2);
    expect(piped.stderr).toContain("pass --yes");
    const no = new FakeInput({ tty: true });
    no.write(ENTER);
    const declined = await yian(["memory", "erase", "MEMORY.md", flow, "lessons"], {
      stdin: no,
      stderrStream: new FakeOutput({ tty: true }),
    });
    expect(declined.code).toBe(1);
    expect(declined.stderr).toContain("nothing erased");
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first");
    const yes = new FakeInput({ tty: true });
    yes.write(`${DOWN}${ENTER}`);
    const output = new FakeOutput({ tty: true });
    const args = ["memory", "erase", "MEMORY.md", flow, "lessons"];
    const erased = await yian(args, { stdin: yes, stderrStream: output });
    expect(erased.code, erased.stderr).toBe(0);
    expect(output.text).toContain("Erase MEMORY.md from");
    expect(existsSync(join(lessons().location, "MEMORY.md"))).toBe(false);
  });
});

/** `APPEND`'s memory, appended to, then a wait for a line on its inbox. */
const WAITS = `id: answer
name: answer
input:
  line: !str
  lessons: !FileMemory
  messages: !Inbox(str)
nodes:
  add:
    kind: tool
    tool_id: {lessons: append}
    input:
      path: MEMORY.md
      text: \${input.line}
  heard:
    kind: tool
    tool_id: take
    depends_on: [add]
    input:
      from: \${input.messages}
output: \${heard.output.closed}
`;

describe("yian memory reset", () => {
  /** A waiting run and a finished one on `flow.yaml`'s memory; the waiting run's id. */
  async function twoRuns(): Promise<string> {
    const waits = join(project, "wait.yaml");
    writeFileSync(waits, WAITS);
    await runOnce(waits, "waiting");
    const [waiting] = runIds();
    await runOnce(flowFile(), "finished");
    return waiting!;
  }

  test("lists the unfinished runs that use the memory, then empties it, keeping its identity", async () => {
    const waiting = await twoRuns();
    const finished = runIds().find((run) => run !== waiting)!;
    const flow = join(project, "flow.yaml");
    const result = await yian(["memory", "reset", flow, "lessons", "--yes"]);
    expect(result.code, result.stderr).toBe(0);
    expect(result.stdout).toContain(waiting);
    expect(result.stdout).not.toContain(finished);
    expect(result.stdout).toContain("a resume goes on with the emptied memory");
    expect(result.stdout).toContain(`emptied ${lessons().identity}`);
    expect(existsSync(join(lessons().location, "MEMORY.md"))).toBe(false);
    expect((await historyOf(flow)).changes).toEqual([]);
    expect(readFileSync(join(lessons().location, ".yian", "identity"), "utf8")).toBe(lessons().identity);
  });

  test("--json prints the identity and the unfinished runs; a no on the terminal keeps the memory", async () => {
    const waiting = await twoRuns();
    const flow = join(project, "flow.yaml");
    const no = new FakeInput({ tty: true });
    no.write(ENTER);
    const declined = await yian(["memory", "reset", flow, "lessons"], {
      stdin: no,
      stderrStream: new FakeOutput({ tty: true }),
    });
    expect(declined.code).toBe(1);
    expect(declined.stderr).toContain("nothing reset");
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("waiting\nfinished");
    const json = await yian(["memory", "reset", flow, "lessons", "--yes", "--json"]);
    expect(json.code, json.stderr).toBe(0);
    expect(JSON.parse(json.stdout)).toEqual({
      format: "yian.cli.v1",
      identity: lessons().identity,
      runs: [expect.objectContaining({ run_id: waiting, status: "waiting" })],
    });
    const piped = await yian(["memory", "reset", flow, "lessons"]);
    expect(piped.code).toBe(2);
    expect(piped.stderr).toContain("pass --yes");
  });
});
