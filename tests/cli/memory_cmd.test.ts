/**
 * `yian memory`: the memories under `~/.yian/memory`, and one named as `FLOW NAME` (an input's
 * default memory) or `--at PATH`. Memories are made by `yian run` on a temporary home, prompts
 * answered on a fake terminal, and the editor is injected.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  chmodSync,
  existsSync,
  linkSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { type ChildProcess, spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { fileURLToPath } from "node:url";

import { main } from "../../src/cli/index.ts";
import { defaultIdentity, defaultLocation } from "../../src/cli/memory_bind.ts";
import { resolveTarget } from "../../src/cli/memory_cmd.ts";
import { type CliIo, UsageError } from "../../src/cli/utils.ts";
import type { Embed } from "../../src/llm/embeddings.ts";
import { FileMemory } from "../../src/memory/file.ts";
import { VectorDBMemory } from "../../src/memory/vectordb.ts";
import { lockFile } from "../../src/platform.ts";
import { shellWords } from "../docs/doc_blocks.ts";
import { foreignWal } from "../helpers/memory_fakes.ts";
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
  mkdirSync(join(tmp, "tmpdir"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** The command's environment: the temporary home, and `vars`. */
function envOf(vars: Record<string, string> = {}): (name: string) => string | undefined {
  // The editor's copy goes in the test's own temporary folder, never the real one.
  const values: Record<string, string> = { HOME: home, USER: "ada", TMPDIR: join(tmp, "tmpdir"), ...vars };
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
    // The system's own record of the home folder is never read in a test.
    accountHome: () => null,
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
  test("a flow whose id can't name a default doesn't point at --memory, which yian memory lacks", async () => {
    const path = join(project, "flow.yaml");
    writeFileSync(path, APPEND.replace("id: answer", 'id: "Answer questions"'));
    const result = await yian(["memory", "show", path, "lessons"]);
    expect(result.code).toBe(1);
    expect(result.stderr).toContain("so `lessons` gets a default memory");
    expect(result.stderr).not.toContain("--memory");
  });

  test("--at takes ~/ as the home folder; an empty path or a file is a usage error", async () => {
    const flow = flowFile();
    await runOnce(flow, "first", "--memory", `lessons=${join(home, "notes")}`);
    const shown = await yian(["memory", "show", "--at=~/notes"]);
    expect(shown.code, shown.stderr).toBe(0);
    expect(shown.stdout).toContain("first");
    writeFileSync(join(tmp, "a-file"), "");
    for (const [at, refusal] of [
      ["", "Invalid value for --at: give the memory's folder"],
      [join(tmp, "a-file"), `Invalid value for --at: ${join(tmp, "a-file")} is a file, not a memory's folder`],
    ] as const) {
      const result = await yian(["memory", "show", "--at", at]);
      expect(result.code, at).toBe(2);
      expect(result.stderr).toContain(refusal);
    }
  });

  test("--at refuses /, the home folder and a folder holding ~/.yian", async () => {
    for (const [path, refusal] of [
      ["/", "`/` can't be a memory"],
      [home, "`~` can't be a memory: it holds your home's files"],
      [tmp, "it holds ~/.yian"],
    ] as const) {
      const result = await yian(["memory", "reset", "--at", path, "--yes"]);
      expect(result.code, path).toBe(2);
      expect(result.stderr).toContain(`Invalid value for --at: `);
      expect(result.stderr).toContain(refusal);
    }
  });

  test("--at refuses the account's own home when HOME is elsewhere", async () => {
    const account = join(tmp, "account");
    mkdirSync(join(account, ".yian"), { recursive: true });
    const accountHome = () => account;
    const result = await yian(["memory", "reset", "--at", account, "--yes"], { accountHome });
    expect(result.code).toBe(2);
    const refusal = `\`${account}\` can't be a memory: it holds your home's files`;
    expect(result.stderr).toContain(`Invalid value for --at: ${refusal}`);
    expect(readdirSync(join(account, ".yian"))).toEqual([]);
  });

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

  test("neither, both, or a name that isn't a default memory is a usage error", async () => {
    const flow = flowFile();
    const { io } = testIo();
    const refusals: [Parameters<typeof resolveTarget>[0], string][] = [
      [{}, "give the memory as FLOW NAME or --at PATH"],
      [{ flow, name: "lessons", at: tmp }, "give the memory as FLOW NAME or --at PATH, not both"],
      [{ flow }, "Missing argument 'NAME'."],
      [{ flow, name: "line" }, "the flow has no memory input `line`"],
      [{ flow, name: "notes" }, "`notes` is Optional, so it has no default memory; name its folder or store with --at"],
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
      expect.stringMatching(/^identity\s+type\s+location$/),
      expect.stringMatching(new RegExp(`^${identity}\\s+FileMemory\\s+${location}$`)),
    ]);
    const json = await yian(["memory", "list", "--json"]);
    expect(JSON.parse(json.stdout)).toEqual({
      format: "yian.cli.v1",
      memories: [{ identity, type: "FileMemory", location }],
    });
  });

  test.skipIf(process.getuid?.() === 0)(
    "an unreadable folder is a warning, not a failure, and a symlinked .yian isn't a memory",
    async () => {
      const flow = flowFile();
      await runOnce(flow, "first");
      const root = join(home, ".yian", "memory");
      mkdirSync(join(root, "locked"), { mode: 0o000 });
      // A folder whose .yian points at a real memory's: listing it would show that one twice.
      mkdirSync(join(root, "fake"));
      symlinkSync(join(lessons().location, ".yian"), join(root, "fake", ".yian"));
      try {
        const result = await yian(["memory", "list", "--json"]);
        expect(result.code, result.stderr).toBe(0);
        expect(result.stderr).toContain(`warning: can't read ${join(root, "locked")}`);
        expect(result.stderr).not.toContain("    at ");
        expect(JSON.parse(result.stdout).memories.map((each: { location: string }) => each.location)).toEqual([
          lessons().location,
        ]);
      } finally {
        chmodSync(join(root, "locked"), 0o700);
      }
    },
  );

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

describe("memory text on the terminal", () => {
  /** An identity a hand edit could leave, holding a screen clear and a title change. */
  const EVIL = "evil\x1b[2J\x1b]0;owned\x07name";

  /** A memory whose identity is `EVIL`, at `<tmp>/evil` and under `~/.yian/memory`. */
  async function evilMemory(): Promise<string> {
    const at = join(tmp, "evil");
    const session = await new FileMemory({ root: at, id: EVIL }).open({
      create: true,
      itemType: null,
      signal: new AbortController().signal,
    });
    await session.close();
    mkdirSync(join(home, ".yian", "memory", "p", "f"), { recursive: true });
    symlinkSync(at, join(home, ".yian", "memory", "p", "f", "linked"));
    const under = join(home, ".yian", "memory", "p", "f", "x");
    mkdirSync(join(under, ".yian"), { recursive: true });
    writeFileSync(join(under, ".yian", "identity"), EVIL);
    return at;
  }

  test("list and erase's lines drop an identity's control sequences, and its question shows them", async () => {
    const at = await evilMemory();
    const tty = { write: async (text: string) => void shownOut.push(text), isTTY: true };
    const shownOut: string[] = [];
    const listed = await main(["memory", "list"], { ...testIo().io, stdout: tty });
    expect(listed).toBe(0);
    expect(shownOut.join("")).toContain("evilname");
    expect(shownOut.join("")).not.toContain("\x1b");
    shownOut.length = 0;
    const input = new FakeInput({ tty: true });
    input.write(ENTER);
    const output = new FakeOutput({ tty: true });
    const io = { ...testIo().io, stdout: tty, stdin: input, stderrStream: output };
    expect(await main(["memory", "erase", "--at", at, "MEMORY.md"], io)).toBe(1);
    expect(shownOut.join("")).toContain("no run on this machine used evilname");
    expect(shownOut.join("")).not.toContain("\x1b");
    expect(output.text).toContain("evil^[[2J^[]0;owned^Gname");
    expect(output.text).not.toContain("\x1b[2J");
    expect(output.text).not.toContain("\x07");
  });
});

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
    const undone = await yian(["memory", "undo", flow, "lessons", newest!.id]);
    expect(undone.code, undone.stderr).toBe(0);
    expect(undone.stdout).toMatch(new RegExp(`^undid ${newest!.id}; the memory's version is [0-9a-f]{64}\\n$`));
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first");
    const [undo] = (await historyOf(flow)).changes;
    expect(undo).toMatchObject({ op: "undo", author: { kind: "user", name: "ada" } });
    expect(undo!.target.undoes).toBe(newest!.id);
    // The first append's file no longer holds only what it wrote.
    await runOnce(flow, "third");
    const conflict = await yian(["memory", "undo", "--at", lessons().location, oldest!.id, "--json"]);
    expect(conflict.code).toBe(1);
    expect(JSON.parse(conflict.stdout).error.code).toBe("memory.conflict");
    expect(conflict.stderr).toContain("undo refused: ");
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first\nthird");
  });
});

describe("a refusal is told from a conflict", () => {
  test("undo of an unknown change, or an edit over the size cap, is memory.refused", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const unknown = await yian(["memory", "undo", flow, "lessons", "no-such-change", "--json"]);
    expect(unknown.code).toBe(1);
    expect(JSON.parse(unknown.stdout).error.code).toBe("memory.refused");
    expect(unknown.stderr).toContain("undo refused: ");
    const huge = await yian(["memory", "edit", flow, "lessons", "--json"], {
      editor: async (path) => writeFileSync(path, "x".repeat(100_001)),
    });
    expect(huge.code).toBe(1);
    expect(JSON.parse(huge.stdout).error.code).toBe("memory.refused");
    rmSync(join(/your text is in (.+)$/m.exec(huge.stderr)![1]!, ".."), { recursive: true });
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

  test("edited text that isn't applied for any reason is kept, and edit says where", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    /** The copy `stderr` names, read and removed. */
    const keptText = (stderr: string): string => {
      const kept = /your text is in (.+)$/m.exec(stderr)![1]!;
      const text = readFileSync(kept, "utf8");
      rmSync(join(kept, ".."), { recursive: true });
      return text;
    };
    // The command is stopped while the person edits: the change never applies.
    const stop = new AbortController();
    const stopped = await yian(["memory", "edit", flow, "lessons", "--json"], {
      stop: stop.signal,
      editor: async (path) => {
        writeFileSync(path, "first\nstopped");
        stop.abort(new Error("stopped by the test"));
      },
    });
    expect(stopped.code).toBe(1);
    expect(stopped.stderr).toContain("edit failed: stopped by the test; your text is in ");
    expect(JSON.parse(stopped.stdout).error.code).toBe("memory.failed");
    expect(keptText(stopped.stderr)).toBe("first\nstopped");
    // The editor fails after the person typed.
    const crashed = await yian(["memory", "edit", flow, "lessons"], {
      editor: async (path) => {
        writeFileSync(path, "first\ntyped");
        throw new RangeError("the editor exited with code 1; nothing was changed");
      },
    });
    expect(crashed.code).toBe(1);
    expect(keptText(crashed.stderr)).toBe("first\ntyped");
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first");
    // An editor that fails before anything was typed leaves nothing behind.
    const untouched = await yian(["memory", "edit", flow, "lessons"], {
      editor: async () => {
        throw new RangeError("the editor didn't start");
      },
    });
    expect(untouched.code).toBe(1);
    expect(untouched.stderr).not.toContain("your text is in");
    expect(readdirSync(join(tmp, "tmpdir"))).toEqual([]);
  });

  test("an edit that isn't UTF-8 text isn't applied, and its copy is kept", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const result = await yian(["memory", "edit", flow, "lessons"], {
      editor: async (path) => writeFileSync(path, new Uint8Array([0x66, 0x69, 0xff, 0xfe])),
    });
    expect(result.code).toBe(1);
    expect(result.stderr).toContain("edit failed: the edited text isn't UTF-8; your text is in ");
    const kept = /your text is in (.+)$/m.exec(result.stderr)![1]!;
    expect([...readFileSync(kept)]).toEqual([0x66, 0x69, 0xff, 0xfe]);
    rmSync(join(kept, ".."), { recursive: true });
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first");
  });

  test("a --file that can't be a file of the memory is refused before the editor opens", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const { location } = lessons();
    writeFileSync(join(tmp, "outside.md"), "outside");
    symlinkSync(join(tmp, "outside.md"), join(location, "linked.md"));
    linkSync(join(tmp, "outside.md"), join(location, "hard.md"));
    writeFileSync(join(location, "binary.md"), new Uint8Array([0xff, 0xfe, 0x00]));
    const opened: string[] = [];
    const editor = async (path: string) => void opened.push(path);
    for (const [file, why] of [
      ["../x.md", "a `..` part"],
      [".yian/identity", "a hidden name"],
      ["linked.md", "a symlink"],
      ["hard.md", "a hard link"],
      ["binary.md", "it isn't UTF-8 text, or is over the memory's size cap"],
    ]) {
      const result = await yian(["memory", "edit", flow, "lessons", "--file", file!], { editor });
      expect(result.code, file).toBe(2);
      const refusal = `\`${file}\` can't be edited as a file of the memory: ${why}`;
      expect(result.stderr).toContain(`Invalid value for --file: ${refusal}`);
    }
    expect(opened).toEqual([]);
    expect(readdirSync(join(tmp, "tmpdir"))).toEqual([]);
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

  test("the copy is made in the command's TMPDIR, and $EDITOR runs with the command's environment", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const opened: string[] = [];
    const editor = async (path: string) => {
      opened.push(path);
      writeFileSync(path, "first\nedited");
    };
    expect((await yian(["memory", "edit", flow, "lessons"], { editor })).code).toBe(0);
    expect(opened[0]).toStartWith(join(tmp, "tmpdir", "yian-memory-edit-"));
    // The copy's folder is gone once the edit applied.
    expect(readdirSync(join(tmp, "tmpdir"))).toEqual([]);
    // The command's HOME is the test's, not the test process's.
    const appendHome = `/bin/sh -c 'printf "\n%s" "$HOME" >> "$0"'`;
    const marked = await yian(["memory", "edit", flow, "lessons"], { env: envOf({ EDITOR: appendHome }) });
    expect(marked.code, marked.stderr).toBe(0);
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe(`first\nedited\n${home}`);
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

describe("undo and erase name the memory first", () => {
  test("FLOW NAME then the argument, or --at PATH then it; another count shows both forms", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const wrong: [string[], string][] = [
      [["undo", flow, "lessons"], "give FLOW NAME CHANGE, or --at PATH CHANGE"],
      [["undo", "--at", lessons().location, flow, "c1"], "give FLOW NAME CHANGE, or --at PATH CHANGE"],
      [["erase", flow, "MEMORY.md", "--yes"], "give FLOW NAME TARGET, or --at PATH TARGET"],
      [["erase", "--at", lessons().location, "--yes"], "give FLOW NAME TARGET, or --at PATH TARGET"],
    ];
    for (const [args, message] of wrong) {
      const result = await yian(["memory", ...args]);
      expect(result.code, args.join(" ")).toBe(2);
      expect(result.stderr).toContain(`Invalid value: ${message}`);
    }
    const missing = await yian(["memory", "undo", join(project, "nope.yaml"), "lessons", "c1"]);
    expect(missing.code).toBe(2);
    expect(missing.stderr).toContain("Invalid value for 'FLOW': File");
    const help = await yian(["memory", "undo", "--help"]);
    expect(help.stdout).toContain("Usage: yian memory undo [OPTIONS] [FLOW] [NAME] [CHANGE]");
  });
});

describe("yian memory erase", () => {
  test("lists the runs that used the memory, erases the text everywhere, and prints the undos disabled", async () => {
    const flow = flowFile();
    await runOnce(flow, "keep this");
    await runOnce(flow, "my PIN 4321");
    const result = await yian(["memory", "erase", flow, "lessons", "MEMORY.md", "--text", "4321", "--yes"]);
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
    const json = await yian(["memory", "erase", "--at", lessons().location, "MEMORY.md", "--yes", "--json"]);
    expect(json.code, json.stderr).toBe(0);
    const document = JSON.parse(json.stdout);
    expect(document.runs).toEqual([
      expect.objectContaining({ run_id: runIds()[0], status: "succeeded", log: expect.stringMatching(/log\.jsonl$/) }),
    ]);
    expect(document.version).toMatch(/^[0-9a-f]{64}$/);
    expect(document.disabled_undos).toHaveLength(1);
    expect(existsSync(join(lessons().location, "MEMORY.md"))).toBe(false);
  });

  test("--json that asks on a terminal shows the runs on stderr; stdout holds one document", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const yes = new FakeInput({ tty: true });
    yes.write(`${DOWN}${ENTER}`);
    const args = ["memory", "erase", flow, "lessons", "MEMORY.md", "--json"];
    const result = await yian(args, { stdin: yes, stderrStream: new FakeOutput({ tty: true }) });
    expect(result.code, result.stderr).toBe(0);
    expect(result.stderr).toContain(`the runs on this machine that used ${lessons().identity}`);
    expect(result.stderr).toContain(runIds()[0]!);
    expect(result.stdout.trimEnd().split("\n")).toHaveLength(1);
    expect(JSON.parse(result.stdout).runs).toHaveLength(1);
  });

  test("a target erase can't take is refused before it asks or lists anything", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    for (const [extra, refusal] of [
      [["MEMORY.md", "--text", ""], "Invalid value for --text: give the text to erase"],
      [["../x.md"], "Invalid value for 'TARGET': `../x.md` can't be a file of the memory: a `..` part"],
      [[".yian/identity"], "Invalid value for 'TARGET': `.yian/identity` can't be a file of the memory: a hidden name"],
    ] as const) {
      const answer = new FakeInput({ tty: true });
      answer.write(`${DOWN}${ENTER}`);
      const output = new FakeOutput({ tty: true });
      const args = ["memory", "erase", flow, "lessons", ...extra];
      const result = await yian(args, { stdin: answer, stderrStream: output });
      expect(result.code, extra.join(" ")).toBe(2);
      expect(result.stderr).toContain(refusal);
      expect(result.stdout).toBe("");
      expect(output.text).not.toContain("Erase ");
    }
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first");
  });

  test("asks first on a terminal; no, or no terminal without --yes, erases nothing", async () => {
    const flow = flowFile();
    await runOnce(flow, "first");
    const piped = await yian(["memory", "erase", flow, "lessons", "MEMORY.md"]);
    expect(piped.code).toBe(2);
    expect(piped.stderr).toContain("pass --yes");
    const no = new FakeInput({ tty: true });
    no.write(ENTER);
    const declined = await yian(["memory", "erase", flow, "lessons", "MEMORY.md"], {
      stdin: no,
      stderrStream: new FakeOutput({ tty: true }),
    });
    expect(declined.code).toBe(1);
    expect(declined.stderr).toContain("nothing erased");
    expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first");
    const yes = new FakeInput({ tty: true });
    yes.write(`${DOWN}${ENTER}`);
    const output = new FakeOutput({ tty: true });
    const args = ["memory", "erase", flow, "lessons", "MEMORY.md"];
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

  test("its question names the folder and the files it deletes", async () => {
    await twoRuns();
    const flow = join(project, "flow.yaml");
    for (const name of ["a.md", "b.md", "c.md"]) writeFileSync(join(lessons().location, name), "x");
    const no = new FakeInput({ tty: true });
    no.write(ENTER);
    const output = new FakeOutput({ tty: true, columns: 400 });
    const declined = await yian(["memory", "reset", flow, "lessons"], { stdin: no, stderrStream: output });
    expect(declined.code).toBe(1);
    const { identity, location } = lessons();
    expect(output.text).toContain(
      `Empty ${identity} at ${location}? This deletes 4 files (MEMORY.md, a.md, b.md, …) and its history.`,
    );
  });

  test("a run that is running on the memory refuses the reset until it stops", async () => {
    const waiting = await twoRuns();
    const flow = join(project, "flow.yaml");
    // A live engine holds its log's lock.
    const lock = lockFile(`${realpathSync(join(home, ".yian", "runs", waiting, "log.jsonl"))}.lock`);
    try {
      const refused = await yian(["memory", "reset", flow, "lessons", "--yes", "--json"]);
      expect(refused.code).toBe(1);
      expect(refused.stderr).toContain(`reset refused: run ${waiting} is running on this memory; stop it first`);
      expect(JSON.parse(refused.stdout).error.code).toBe("memory.refused");
      expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("waiting\nfinished");
    } finally {
      lock.release();
    }
    expect((await yian(["memory", "reset", flow, "lessons", "--yes"])).code).toBe(0);
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

/** The `yian` entry script. */
const BIN = fileURLToPath(new URL("../../src/cli/bin.ts", import.meta.url));

/**
 * Run `yian args` as a process of its own, in a new process group (a terminal's foreground group),
 * with only the test's environment and `vars`; give back its exit code (null when a signal ended
 * it) and both streams. A run that outlives 20 s is killed with its group.
 */
async function yianProcess(args: readonly string[], vars: Record<string, string>) {
  const env = { PATH: "/usr/bin:/bin", HOME: home, USER: "ada", TMPDIR: join(tmp, "tmpdir"), ...vars };
  const child: ChildProcess = spawn(process.execPath, [BIN, ...args], {
    detached: true,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout!.setEncoding("utf8").on("data", (chunk: string) => void (stdout += chunk));
  child.stderr!.setEncoding("utf8").on("data", (chunk: string) => void (stderr += chunk));
  const killer = setTimeout(() => process.kill(-child.pid!, "SIGKILL"), 20_000);
  const code = await new Promise<number | null>((resolve) => child.once("close", (exitCode) => resolve(exitCode)));
  clearTimeout(killer);
  return { code, stdout, stderr };
}

/** Write the executable shell script `body` as `<tmp>/<name>`; return its path. */
function script(name: string, body: string): string {
  const path = join(tmp, name);
  writeFileSync(path, `#!/bin/sh\n${body}\n`, { mode: 0o755 });
  return path;
}

describe("yian memory edit as a process", () => {
  test(
    "Ctrl-C while the editor is open is the editor's: yian waits for it, and the edit applies",
    async () => {
      const flow = flowFile();
      await runOnce(flow, "first");
      // Not an editor: ignores Ctrl-C as vim does, presses it for its whole group as a terminal
      // would, then writes and exits.
      const body = `trap '' INT QUIT\nkill -INT 0\nkill -QUIT 0\nsleep 1\nprintf '\\nkept' >> "$1"`;
      const editor = script("editor.sh", body);
      const result = await yianProcess(["memory", "edit", flow, "lessons"], { EDITOR: editor });
      expect(result.code, result.stderr).toBe(0);
      expect(readFileSync(join(lessons().location, "MEMORY.md"), "utf8")).toBe("first\nkept");
    },
    30_000,
  );

  test(
    "under --json, what the editor writes to stdout goes to stderr: stdout holds one document",
    async () => {
      const flow = flowFile();
      await runOnce(flow, "first");
      const editor = script("chatty.sh", `echo "the editor's own output"\nprintf '\\nedited' >> "$1"`);
      const result = await yianProcess(["memory", "edit", flow, "lessons", "--json"], { EDITOR: editor });
      expect(result.code, result.stderr).toBe(0);
      expect(result.stdout.trimEnd().split("\n")).toHaveLength(1);
      expect(JSON.parse(result.stdout)).toMatchObject({ format: "yian.cli.v1", version: expect.any(String) });
      expect(result.stderr).toContain("the editor's own output");
    },
    30_000,
  );
});

/** An embedder that never calls a provider: a 4-dimension vector from each text's length. */
const fakeEmbed: Embed = async (texts) => texts.map((text) => Float32Array.of((text.length % 7) + 1, 1, 0, 0));

/** A flow that adds `line` (an input) to its VectorDB memory. */
const CASES = `id: cases
name: cases
input:
  line: !str
  cases: !VectorDBMemory
nodes:
  add:
    kind: tool
    tool_id: {cases: add}
    input:
      text: \${input.line}
output: \${add.output.id}
`;

describe("VectorDB stores", () => {
  /** The io a store's commands get: embeddings that never call a provider. */
  const vectors = { embed: fakeEmbed };

  /** Write `CASES` as `<tmp>/proj/cases.yaml`; return its path. */
  function casesFile(): string {
    const path = join(project, "cases.yaml");
    writeFileSync(path, CASES);
    return path;
  }

  /** Add `line` to the store through a run, made with a test model. */
  async function addCase(flow: string, line: string): Promise<void> {
    const args = ["run", flow, "--quiet", "--input", `line=${line}`, "--embedding", "openai/test-embedding"];
    const result = await yian(args, { embed: fakeEmbed });
    expect(result.code, result.stderr).toBe(0);
  }

  /** The default identity and store of `cases.yaml`'s `cases`. */
  function cases(): { identity: string; location: string } {
    const identity = defaultIdentity(project, "cases", "cases");
    return { identity, location: defaultLocation(home, identity, "VectorDBMemory") };
  }

  /** Retire item `id` (at `version`) as a TOOL node would. */
  async function retire(id: bigint, version: bigint): Promise<void> {
    const store = new VectorDBMemory({ path: cases().location, embed: fakeEmbed });
    const session = await store.open({ create: false, itemType: null, signal: new AbortController().signal });
    const author = { kind: "tool" as const, name: "tidy" };
    const change = { op: "retire", id, if_version: version, why: "stale" };
    const applied = await session.apply({ id: "retire-1", change, author, signal: new AbortController().signal });
    expect(applied.ok).toBe(true);
    await session.close();
  }

  test("list includes each .db store under ~/.yian/memory, with the identity it records", async () => {
    await addCase(casesFile(), "Check sums twice.");
    const listed = JSON.parse((await yian(["memory", "list", "--json"])).stdout).memories;
    expect(listed).toEqual([{ identity: cases().identity, type: "VectorDBMemory", location: cases().location }]);
  });

  test("show counts items and shows the recall; --all shows every item; --at x.db names a store", async () => {
    const flow = casesFile();
    await addCase(flow, "Check sums twice.");
    await addCase(flow, "Carry the one.");
    await retire(2n, 2n);
    const { identity, location } = cases();
    const shown = await yian(["memory", "show", flow, "cases"], { embed: fakeEmbed });
    expect(shown.code, shown.stderr).toBe(0);
    expect(shown.stdout).toStartWith(`${identity} (VectorDBMemory) at ${location}: 2 items, 1 retired\n1 item`);
    const all = await yian(["memory", "show", "--at", location, "--all"], { embed: fakeEmbed });
    expect(all.code, all.stderr).toBe(0);
    expect(all.stdout).toContain("== item 1 (version 1) ==\nCheck sums twice.\n");
    expect(all.stdout).toContain("== item 2 (version 3), retired: stale ==\nCarry the one.\n");
    const json = JSON.parse((await yian(["memory", "show", flow, "cases", "--all", "--json"], vectors)).stdout);
    expect(json).toMatchObject({ identity, type: "VectorDBMemory", location, count: 2, retired: 1 });
    expect(json.items.map((item: { id: number; text: string }) => [item.id, item.text])).toEqual([
      [1, "Check sums twice."],
      [2, "Carry the one."],
    ]);
  });

  test("history names each change's item; erase takes an item; edit has --add, --id --text and --delete", async () => {
    const flow = casesFile();
    await addCase(flow, "Check sums twice.");
    await addCase(flow, "Carry the one.");
    const history = await yian(["memory", "history", flow, "cases"], { embed: fakeEmbed });
    expect(history.stdout.split("\n")[0]).toContain("add item 2 by tool add (run ");
    const added = await yian(["memory", "edit", flow, "cases", "--add", "Round at the end."], { embed: fakeEmbed });
    expect(added.code, added.stderr).toBe(0);
    expect(added.stdout).toMatch(/^added item 3; the memory's version is \d+\n$/);
    const edited = await yian(["memory", "edit", flow, "cases", "--id", "1", "--text", "Check sums thrice."], {
      embed: fakeEmbed,
    });
    expect(edited.code, edited.stderr).toBe(0);
    expect(edited.stdout).toStartWith("edited item 1;");
    const deleted = await yian(["memory", "edit", flow, "cases", "--delete", "2", "--yes"], { embed: fakeEmbed });
    expect(deleted.code, deleted.stderr).toBe(0);
    const [newest] = JSON.parse((await yian(["memory", "history", flow, "cases", "--json"], vectors)).stdout).changes;
    expect(newest).toMatchObject({ op: "delete", target: { item: 2 }, author: { kind: "user", name: "ada" } });
    const erased = await yian(["memory", "erase", flow, "cases", "3", "--yes"], { embed: fakeEmbed });
    expect(erased.code, erased.stderr).toBe(0);
    expect(erased.stdout).toContain("erased item 3;");
    const json = JSON.parse((await yian(["memory", "show", flow, "cases", "--all", "--json"], vectors)).stdout);
    expect(json.items.map((item: { id: number; text: string }) => [item.id, item.text])).toEqual([
      [1, "Check sums thrice."],
    ]);
  });

  test("edit and erase take a store's own arguments, and refuse a FileMemory's", async () => {
    const STORE_EDITS = "give --add TEXT, --id N with --text TEXT or --fields JSON, or --delete N";
    const flow = casesFile();
    await addCase(flow, "Check sums twice.");
    for (const [args, message] of [
      [["edit", flow, "cases"], STORE_EDITS],
      [["edit", flow, "cases", "--id", "1"], STORE_EDITS],
      [["edit", flow, "cases", "--delete", "1", "--yes", "--fields", "{}"], "--fields goes with --add or --id"],
      [["edit", flow, "cases", "--add", "x", "--fields", "[1]"], "Invalid value for --fields: give a JSON object"],
      [["edit", flow, "cases", "--add", "x", "--fields", "{"], "Invalid value for --fields: give a JSON object"],
      [["edit", flow, "cases", "--add", "x", "--fields", "{}"], "this memory's items are text only"],
      [["edit", flow, "cases", "--file", "MEMORY.md"], "--file is a FileMemory's"],
      [["erase", flow, "cases", "MEMORY.md", "--yes"], "a VectorDBMemory's TARGET is an item's id"],
      [["erase", flow, "cases", "1", "--text", "x", "--yes"], "--text is a FileMemory's"],
    ] as const) {
      const result = await yian(["memory", ...args], { embed: fakeEmbed });
      expect(result.code, args.join(" ")).toBe(2);
      expect(result.stderr).toContain(message);
    }
    const files = flowFile();
    await runOnce(files, "first");
    const wrong = await yian(["memory", "edit", files, "lessons", "--add", "x"]);
    expect(wrong.code).toBe(2);
    expect(wrong.stderr).toContain("--add, --id, --text, --fields, --delete and --if-version are a VectorDBMemory's");
  });

  test("--fields gives a typed store's item its fields, and the CLI page's example runs as written", async () => {
    const triage = `id: triage
name: triage
typedefs:
  Case:
    title: str
    tags: list(str)
input:
  ticket: !str
  cases: !VectorDBMemory(Case)
nodes:
  add:
    kind: tool
    tool_id: {cases: add}
    input:
      text: \${input.ticket}
      fields: {title: first, tags: [setup]}
output: \${add.output.id}
`;
    const flow = join(project, "triage.yaml");
    writeFileSync(flow, triage);
    const run = ["run", flow, "--quiet", "--input", "ticket=Check sums twice.", "--embedding", "openai/test-embedding"];
    expect((await yian(run, vectors)).code).toBe(0);
    const bare = await yian(["memory", "edit", flow, "cases", "--add", "No fields."], vectors);
    expect(bare.code).toBe(2);
    expect(bare.stderr).toContain("an add to this memory needs fields, since its items are typed");
    // The line the CLI page shows, run from the flow's folder.
    const page = readFileSync(fileURLToPath(new URL("../../docs/src/content/docs/cli.md", import.meta.url)), "utf8");
    const line = page.split("\n").find((each) => each.startsWith("$ yian memory edit triage.yaml cases --add"))!;
    const args = shellWords(line.slice(2)).slice(1).map((word) => (word === "triage.yaml" ? flow : word));
    const added = await yian(args, vectors);
    expect(added.code, added.stderr).toBe(0);
    expect(added.stdout).toStartWith("added item 2;");
    const fields = JSON.stringify({ title: "sums", tags: ["math", "checks"] });
    const changed = await yian(["memory", "edit", flow, "cases", "--id", "1", "--fields", fields], vectors);
    expect(changed.code, changed.stderr).toBe(0);
    const shown = JSON.parse((await yian(["memory", "show", flow, "cases", "--all", "--json"], vectors)).stdout);
    expect(shown.items.map((item: { id: number; text: string; fields: unknown }) => [item.id, item.fields])).toEqual([
      [1, { title: "sums", tags: ["math", "checks"] }],
      [2, JSON.parse(shellWords(line.slice(2)).at(-1)!)],
    ]);
  });

  test("reembed re-embeds a store with --embedding's model; compact rebuilds it; a FileMemory is refused", async () => {
    const flow = casesFile();
    await addCase(flow, "Check sums twice.");
    const missing = await yian(["memory", "reembed", flow, "cases"], vectors);
    expect(missing.code).toBe(2);
    expect(missing.stderr).toContain("Invalid value: reembed needs --embedding provider/model");
    const reembedded = await yian(["memory", "reembed", flow, "cases", "--embedding", "openai/other"], vectors);
    expect(reembedded.code, reembedded.stderr).toBe(0);
    expect(reembedded.stdout).toBe(`re-embedded ${cases().identity} with openai/other\n`);
    const store = new VectorDBMemory({ path: cases().location });
    const stored = await store.stored({ signal: new AbortController().signal });
    expect(stored?.embedding).toEqual({ provider: "openai", model: "other" });
    const compacted = await yian(["memory", "compact", "--at", cases().location, "--json"], vectors);
    expect(compacted.code, compacted.stderr).toBe(0);
    expect(JSON.parse(compacted.stdout)).toEqual({ format: "yian.cli.v1", identity: cases().identity });
    const files = flowFile();
    await runOnce(files, "first");
    for (const command of [["reembed", "--embedding", "openai/x"], ["compact"]]) {
      const refused = await yian(["memory", command[0]!, files, "lessons", ...command.slice(1)]);
      expect(refused.code, command[0]).toBe(2);
      const refusal = `Invalid value: ${command[0]} is a VectorDBMemory's; ${lessons().identity} is a FileMemory`;
      expect(refused.stderr).toContain(refusal);
    }
  });

  test("show --at and list leave another program's WAL database as it was, with no lock beside it", async () => {
    const folder = join(home, ".yian", "memory", "apps");
    mkdirSync(folder, { recursive: true });
    const foreign = foreignWal(folder, "app.db");
    const before = [readFileSync(foreign), readFileSync(`${foreign}-wal`)];
    const shown = await yian(["memory", "show", "--at", foreign], vectors);
    expect(shown.code).toBe(1);
    expect(shown.stderr).toContain("isn't a VectorDBMemory store: it is a database in WAL mode");
    const listed = await yian(["memory", "list"], vectors);
    expect(listed.code, listed.stderr).toBe(0);
    expect(listed.stderr).toContain("it is a database in WAL mode");
    expect([readFileSync(foreign), readFileSync(`${foreign}-wal`)]).toEqual(before);
    expect(readdirSync(folder).sort()).toEqual(["app.db", "app.db-wal"]);
  });

  test("edit --delete lists the runs that used the store and asks, as erase does; no item, no ask", async () => {
    const flow = casesFile();
    await addCase(flow, "Check sums twice.");
    const { identity } = cases();
    const piped = await yian(["memory", "edit", flow, "cases", "--delete", "1"], vectors);
    expect(piped.code).toBe(2);
    expect(piped.stderr).toContain("Invalid value: delete asks first; off a terminal, pass --yes");
    const no = new FakeInput({ tty: true });
    no.write(ENTER);
    const output = new FakeOutput({ tty: true, columns: 400 });
    const terminal = { ...vectors, stdin: no, stderrStream: output };
    const declined = await yian(["memory", "edit", flow, "cases", "--delete", "1"], terminal);
    expect(declined.code).toBe(1);
    expect(declined.stdout).toContain(`the runs on this machine that used ${identity}; their logs may hold copies:`);
    expect(output.text).toContain(`Delete item 1 from ${identity}? Its text and history go, and it can't be undone.`);
    expect(declined.stderr).toContain("delete cancelled: nothing deleted");
    const missing = await yian(["memory", "edit", flow, "cases", "--delete", "99", "--yes", "--json"], vectors);
    expect(missing.code).toBe(1);
    expect(missing.stderr).not.toContain("the runs on this machine");
    expect(JSON.parse(missing.stdout).error).toEqual({ code: "memory.refused", message: "item 99 doesn't exist" });
    const deleted = await yian(["memory", "edit", flow, "cases", "--delete", "1", "--yes"], vectors);
    expect(deleted.code, deleted.stderr).toBe(0);
    expect(deleted.stdout).toContain("deleted item 1;");
  });

  test("--if-version pins the version the person saw; a newer one is a conflict", async () => {
    const flow = casesFile();
    await addCase(flow, "Check sums twice.");
    const [item] = JSON.parse((await yian(["memory", "show", flow, "cases", "--all", "--json"], vectors)).stdout).items;
    const seen = String(item.version);
    const pin = ["--if-version", seen];
    const first = await yian(["memory", "edit", flow, "cases", "--id", "1", "--text", "Thrice.", ...pin], vectors);
    expect(first.code, first.stderr).toBe(0);
    const stale = await yian(["memory", "edit", flow, "cases", "--id", "1", "--text", "Twice.", ...pin], vectors);
    expect(stale.code).toBe(1);
    expect(stale.stderr).toContain("edit refused: ");
    const gone = ["memory", "edit", flow, "cases", "--delete", "1", "--yes", "--if-version", seen, "--json"];
    const conflict = await yian(gone, vectors);
    expect(conflict.code).toBe(1);
    expect(JSON.parse(conflict.stdout).error.code).toBe("memory.conflict");
    const added = await yian(["memory", "edit", flow, "cases", "--add", "x", "--if-version", "1"], vectors);
    expect(added.code).toBe(2);
    expect(added.stderr).toContain("Invalid value for --if-version: --if-version goes with --id or --delete");
    const shown = JSON.parse((await yian(["memory", "show", flow, "cases", "--all", "--json"], vectors)).stdout);
    expect(shown.items.map((each: { text: string }) => each.text)).toEqual(["Thrice."]);
  });

  test("reset's question counts the store's items", async () => {
    const flow = casesFile();
    await addCase(flow, "Check sums twice.");
    const no = new FakeInput({ tty: true });
    no.write(ENTER);
    const output = new FakeOutput({ tty: true, columns: 400 });
    const declined = await yian(["memory", "reset", flow, "cases"], { ...vectors, stdin: no, stderrStream: output });
    expect(declined.code).toBe(1);
    const { identity, location } = cases();
    expect(output.text).toContain(`Empty ${identity} at ${location}? This deletes 1 item and its history.`);
  });

  test("config embedding saves the model a new store is made with; YIAN_DEFAULT_EMBEDDING overrides it", async () => {
    const signal = new AbortController().signal;
    const saved = await yian(["config", "embedding", "openai/text-embedding-3-small"]);
    expect(saved.code, saved.stderr).toBe(0);
    expect(saved.stdout).toBe("saved the default embedding model openai/text-embedding-3-small\n");
    const flow = casesFile();
    const run = ["run", flow, "--quiet", "--input", "line=Check sums twice."];
    const ran = await yian(run, vectors);
    expect(ran.code, ran.stderr).toBe(0);
    const made = await new VectorDBMemory({ path: cases().location }).stored({ signal });
    expect(made?.embedding).toEqual({ provider: "openai", model: "text-embedding-3-small" });

    rmSync(join(home, ".yian", "memory"), { recursive: true });
    const env = envOf({ YIAN_DEFAULT_EMBEDDING: "vllm/bge-m3" });
    const overridden = await yian(run, { ...vectors, env });
    expect(overridden.code, overridden.stderr).toBe(0);
    const remade = await new VectorDBMemory({ path: cases().location }).stored({ signal });
    expect(remade?.embedding).toEqual({ provider: "vllm", model: "bge-m3" });
    const shadowed = await yian(["config", "embedding", "openai/text-embedding-3-large"], { env });
    expect(shadowed.code, shadowed.stderr).toBe(0);
    expect(shadowed.stderr).toBe("YIAN_DEFAULT_EMBEDDING=vllm/bge-m3 is set, and runs use it instead\n");
  });

  test("yian config embedding refuses a value that isn't provider/model on a provider with embeddings", async () => {
    const bare = await yian(["config", "embedding", "text-embedding-3-small"]);
    expect(bare.code).toBe(2);
    expect(bare.stderr).toContain("Invalid value for 'MODEL': give it as provider/model");
    const chat = await yian(["config", "embedding", "anthropic/claude"]);
    expect(chat.code).toBe(2);
    expect(chat.stderr).toContain("provider `anthropic` has no embeddings API");
    expect(existsSync(join(home, ".config"))).toBe(false);
  });

  test("yian config --help says the group also sets the default embedding model", async () => {
    const help = await yian(["config", "--help"]);
    expect(help.code).toBe(0);
    expect(help.stdout).toContain("Check an llm_config layer, or set the default embedding model.");
    expect(help.stdout).toContain("embedding");
  });
});
