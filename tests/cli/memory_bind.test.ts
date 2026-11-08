/**
 * How the command line binds memory inputs: the default identity of an unbound input
 * (`<project key>/<flow id>/<input>`) and its location under `~/.yian/memory`, confined there;
 * `--memory NAME=PATH`; and memory inputs refused as values. Runs go through `main`, with `HOME`
 * a temporary folder.
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
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import {
  bindMemories,
  checkDefaultLocation,
  defaultIdentity,
  defaultLocation,
  projectKey,
} from "../../src/cli/memory_bind.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { FileMemory } from "../../src/memory/file.ts";
import { parseJson } from "../../src/typesys/json.ts";
import { loadSchema, type Schema, validate } from "../helpers/schema.ts";

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
    // Written from `~`: the message names no location.
    expect(() => checkDefaultLocation(home, "proj-1234abcd/f/lessons")).toThrow(
      "~/.yian/memory/proj-1234abcd is a symlink",
    );
    expect(() => checkDefaultLocation(home, "../x/f/lessons")).toThrow("leaves ~/.yian/memory");
    expect(() => checkDefaultLocation(home, "a/../../x")).toThrow("leaves ~/.yian/memory");
    expect(() => checkDefaultLocation(home, ".")).toThrow("leaves ~/.yian/memory");
  });

  test("a file where a folder of the location should be is refused", () => {
    const root = join(home, ".yian", "memory");
    mkdirSync(join(root, "proj-1234abcd"));
    writeFileSync(join(root, "proj-1234abcd", "f"), "");
    expect(() => checkDefaultLocation(home, "proj-1234abcd/f/lessons")).toThrow(
      "~/.yian/memory/proj-1234abcd/f isn't a folder",
    );
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

  test("a path starting ~/ is under the home, as the shell would expand it", async () => {
    const result = await yian(["run", flowFile(APPEND), "--quiet", "--memory=lessons=~/notes"]);
    expect(result.code, result.stderr).toBe(0);
    expect(readFileSync(join(home, "notes", "MEMORY.md"), "utf8")).toContain("Check sums twice.");
  });

  test("spaces around the name and the path are dropped", async () => {
    const notes = join(tmp, "notes");
    const result = await yian(["run", flowFile(APPEND), "--quiet", "--memory", ` lessons = ${notes} `]);
    expect(result.code, result.stderr).toBe(0);
    expect(readFileSync(join(notes, "MEMORY.md"), "utf8")).toContain("Check sums twice.");
  });

  test("two inputs given one path, however spelled, share one memory", async () => {
    const shared = join(tmp, "shared");
    const args = ["--memory", `lessons=${shared}`, "--memory", `notes=${join(tmp, "proj", "..", "shared")}/`];
    const result = await yian(["run", flowFile(TWO), "--quiet", ...args]);
    expect(result.code, result.stderr).toBe(0);
    expect(readFileSync(join(shared, "a.md"), "utf8")).toContain("from lessons");
    expect(readFileSync(join(shared, "b.md"), "utf8")).toContain("from notes");
  });

  test("a flag naming another input's default memory is a usage error, whichever input comes first", async () => {
    const flow = flowFile(TWO);
    const project = realpathSync(join(tmp, "proj"));
    for (const [flagged, other] of [
      ["notes", "lessons"],
      ["lessons", "notes"],
    ] as const) {
      const location = defaultLocation(home, defaultIdentity(project, "two", other), "FileMemory");
      const result = await yian(["run", flow, "--quiet", "--memory", `${flagged}=${location}`]);
      expect(result.code, result.stderr).toBe(2);
      expect(result.stderr).toContain(
        `Invalid value for --memory: \`${flagged}=${location}\` points at \`${other}\`'s default memory`,
      );
    }
    expect(runFolders()).toEqual([]);
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

/** The one project folder under `~/.yian/memory`, and the project key it is named by. */
function projectFolder(): { key: string; path: string } {
  const root = join(home, ".yian", "memory");
  const keys = readdirSync(root);
  expect(keys).toHaveLength(1);
  expect(keys[0]).toMatch(/^proj-[0-9a-f]{8}$/);
  return { key: keys[0]!, path: join(root, keys[0]!) };
}

/** Server-Sent Events of an OpenAI Responses stream answering "ok". */
const RESPONSES_REPLY = (() => {
  const message = { id: "msg_1", type: "message", role: "assistant", content: [{ type: "output_text", text: "ok" }] };
  const events: [string, Record<string, unknown>][] = [
    ["response.output_item.added", { output_index: 0, item: { ...message, content: [] } }],
    ["response.output_text.delta", { output_index: 0, content_index: 0, item_id: "msg_1", delta: "ok" }],
    ["response.completed", { response: { id: "resp_1", status: "completed", output: [message] } }],
  ];
  return events.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify({ type: event, ...data })}\n\n`).join("");
})();

/** Run `body` with a `fetch` that answers every model request "ok", recording each body sent. */
async function withModel<T>(body: (sent: string[]) => Promise<T>): Promise<T> {
  const original = globalThis.fetch;
  const sent: string[] = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    sent.push(await new Request(input as string, init).text());
    return new Response(RESPONSES_REPLY, { headers: { "content-type": "text/event-stream" } });
  }) as typeof fetch;
  try {
    return await body(sent);
  } finally {
    globalThis.fetch = original;
  }
}

/** A flow whose agent reads the memory's recall, then appends the question to its index. */
const RECALL = `id: answer
name: answer
llm_config: {provider: openai, model: offline-model}
input:
  question: !str
  lessons: !FileMemory
nodes:
  reply:
    kind: agent
    input:
      lessons: \${input.lessons}
      question: \${input.question}
    prompt:
      - user: |-
          <memory name="lessons">
          \${lessons}
          </memory>
      - user: \${question}
    output: str
  keep:
    kind: tool
    tool_id: {lessons: append}
    depends_on: [reply]
    input:
      path: MEMORY.md
      text: \${input.question}
output: \${reply.output}
`;

describe("a default location changed between the bind and the open", () => {
  /** Bind `APPEND`'s `lessons` to its default under the temporary home, the project folder made. */
  async function bound() {
    const flow = await loadFlow(APPEND, { searchPaths: [join(tmp, "proj")], sourceLabel: "flow.yaml" });
    const created: unknown[] = [];
    const note = (each: unknown) => void created.push(each);
    const { lessons } = bindMemories(flow, [], { home, note, project: realpathSync(join(tmp, "proj")) });
    return { lessons: lessons!, created };
  }
  const open = (memory: { open: FileMemory["open"] }) =>
    memory.open({ create: true, itemType: null, signal: new AbortController().signal });

  test("a folder swapped for a symlink after the bind is refused at the open, and nothing is made", async () => {
    mkdirSync(join(tmp, "proj"));
    const key = projectKey(join(tmp, "proj"));
    mkdirSync(join(home, ".yian", "memory", key));
    const { lessons, created } = await bound();
    // yian run prompts for inputs and keys between the bind and the open.
    const outside = join(tmp, "outside");
    mkdirSync(outside);
    rmSync(join(home, ".yian", "memory", key), { recursive: true });
    symlinkSync(outside, join(home, ".yian", "memory", key));
    await expect(open(lessons)).rejects.toThrow(`~/.yian/memory/${key} is a symlink`);
    expect(readdirSync(outside)).toEqual([]);
    expect(created).toEqual([]);
  });

  test("~/.yian/memory moved elsewhere after the bind is refused at the open", async () => {
    mkdirSync(join(tmp, "proj"));
    const { lessons } = await bound();
    const elsewhere = join(tmp, "elsewhere");
    mkdirSync(elsewhere);
    rmSync(join(home, ".yian", "memory"), { recursive: true });
    symlinkSync(elsewhere, join(home, ".yian", "memory"));
    await expect(open(lessons)).rejects.toThrow("changed after it was checked");
    expect(readdirSync(elsewhere)).toEqual([]);
  });
});

describe("a memory that fails to start", () => {
  test("another memory at the default location refuses the run: a setup failure, as text and --json", async () => {
    const flow = flowFile(APPEND);
    const identity = defaultIdentity(realpathSync(join(tmp, "proj")), "answer", "lessons");
    const location = defaultLocation(home, identity, "FileMemory");
    mkdirSync(join(location, ".yian", "tmp"), { recursive: true });
    writeFileSync(join(location, ".yian", "identity"), "someone/else");
    const text = await yian(["run", flow, "--quiet"]);
    expect(text.code).toBe(1);
    expect(text.stderr).toContain(`run failed: memory \`lessons\`: this memory is`);
    // Reported, not thrown: no stack.
    expect(text.stderr).not.toContain("    at ");
    const json = await yian(["run", flow, "--quiet", "--json"]);
    expect(json.code).toBe(1);
    const { error } = JSON.parse(json.stdout);
    expect(error.code).toBe("run.setup");
    expect(error.message).toStartWith("memory `lessons`: this memory is");
  });

  test("a file where the memory's folder goes refuses the run: a setup failure", async () => {
    const flow = flowFile(APPEND);
    const identity = defaultIdentity(realpathSync(join(tmp, "proj")), "answer", "lessons");
    const location = defaultLocation(home, identity, "FileMemory");
    mkdirSync(dirname(location), { recursive: true });
    writeFileSync(location, "");
    const json = await yian(["run", flow, "--quiet", "--json"]);
    expect(json.code).toBe(1);
    expect(json.stderr).not.toContain("    at ");
    expect(JSON.parse(json.stdout).error).toMatchObject({
      code: "run.setup",
      message: expect.stringContaining("isn't a folder"),
    });
  });

  test("a flow id longer than 100 characters isn't plain: refused before anything is written", async () => {
    const plain = "give the flow a plain `id:` (letters, digits, `.`, `_`, `-`) or bind `lessons` with --memory";
    expect(() => defaultIdentity("/p/proj", "x".repeat(101), "lessons")).toThrow(plain);
    expect(defaultIdentity("/p/proj", "x".repeat(100), "lessons")).toEndWith(`/${"x".repeat(100)}/lessons`);
    const long = flowFile(APPEND.replace("id: answer", `id: ${"x".repeat(300)}`));
    const result = await yian(["run", long, "--quiet", "--json"]);
    expect(result.code).toBe(1);
    expect(JSON.parse(result.stdout).error.message).toBe(plain);
    expect(runFolders()).toEqual([]);
    expect(readdirSync(join(home, ".yian", "memory"))).toEqual([]);
  });
});

describe("default memories", () => {
  test("an unbound memory input gets its project's default, and the run says it created it", async () => {
    const result = await yian(["run", flowFile(APPEND), "--quiet"]);
    expect(result.code, result.stderr).toBe(0);
    const { key, path } = projectFolder();
    const location = join(path, "answer", "lessons");
    expect(readFileSync(join(location, "MEMORY.md"), "utf8")).toContain("Check sums twice.");
    expect(readFileSync(join(location, ".yian", "identity"), "utf8")).toBe(`${key}/answer/lessons`);
    expect(result.stderr).toContain(`memory lessons: ${key}/answer/lessons at ${location} (new)\n`);
    // Private, as the run folders are.
    for (const folder of [path, join(path, "answer"), location]) expect(statSync(folder).mode & 0o777).toBe(0o700);
  });

  test("yian status --json lists each run's memory identities", async () => {
    expect((await yian(["run", flowFile(APPEND), "--quiet"])).code).toBe(0);
    const { key } = projectFolder();
    const status = await yian(["status", "--json"]);
    expect(status.code, status.stderr).toBe(0);
    const schema = loadSchema("cli.schema.json");
    const runsDocument = (schema.$defs as Record<string, Schema>).runs_document!;
    expect(validate(schema, runsDocument, parseJson(status.stdout, "plain"))).toEqual([]);
    const [run] = JSON.parse(status.stdout).runs;
    expect(run.memories).toEqual([{ input: "lessons", identity: `${key}/answer/lessons`, type: "FileMemory" }]);
    // Identities only: the folder is nowhere in the listing.
    expect(status.stdout).not.toContain(join(home, ".yian", "memory"));
  });

  test("a second run says nothing about the memory, and its prompt holds what the first appended", async () => {
    const vars = { OPENAI_API_KEY: "sk-fake", YIAN_AUTH_FILE: join(tmp, "auth.json"), XDG_CONFIG_HOME: join(tmp, "xdg") };
    const flow = flowFile(RECALL);
    await withModel(async (sent) => {
      const first = await yian(["run", flow, "--quiet", "--input", "question=What is 2+2?"], vars);
      expect(first.code, first.stderr).toBe(0);
      expect(first.stderr).toContain("(new)");
      expect(sent).toHaveLength(1);
      expect(sent[0]).not.toContain("What is 2+2?\\n</memory>");
      const second = await yian(["run", flow, "--quiet", "--input", "question=And 3+3?"], vars);
      expect(second.code, second.stderr).toBe(0);
      expect(second.stderr).not.toMatch(/^memory /m);
      expect(sent).toHaveLength(2);
      expect(sent[1]).toContain("What is 2+2?\\n</memory>");
    });
  });

  test("a --memory folder that holds files is adopted: one (new) line, and none on the next run", async () => {
    const notes = join(tmp, "notes");
    mkdirSync(notes);
    writeFileSync(join(notes, "old.md"), "Kept.\n");
    const flow = flowFile(APPEND);
    const first = await yian(["run", flow, "--quiet", "--memory", `lessons=${notes}`]);
    expect(first.code, first.stderr).toBe(0);
    const identity = readFileSync(join(notes, ".yian", "identity"), "utf8");
    expect(first.stderr.match(/\(new\)/g)).toHaveLength(1);
    expect(first.stderr).toContain(`memory lessons: ${identity} at ${notes} (new)\n`);
    expect(readFileSync(join(notes, "old.md"), "utf8")).toBe("Kept.\n");
    const second = await yian(["run", flow, "--quiet", "--memory", `lessons=${notes}`]);
    expect(second.code, second.stderr).toBe(0);
    expect(second.stderr).not.toMatch(/^memory /m);
  });

  test("an Optional memory input left unbound stays null: no default is made for it", async () => {
    const flow = flowFile(APPEND.replace("  lessons: !FileMemory\n", "  lessons: !FileMemory\n  notes: !Optional(FileMemory)\n"));
    const result = await yian(["run", flow, "--quiet"]);
    expect(result.code, result.stderr).toBe(0);
    expect(readdirSync(join(projectFolder().path, "answer"))).toEqual(["lessons"]);
    expect(result.stderr).not.toContain("memory notes");
  });

  test("a flow whose id isn't plain is refused before any log or memory is written", async () => {
    const result = await yian(["run", flowFile(APPEND.replace("id: answer", 'id: "Answer questions"')), "--quiet"]);
    expect(result.code).toBe(1);
    expect(result.stderr).toContain(
      "give the flow a plain `id:` (letters, digits, `.`, `_`, `-`) or bind `lessons` with --memory",
    );
    expect(runFolders()).toEqual([]);
    expect(readdirSync(join(home, ".yian", "memory"))).toEqual([]);
    // --memory binds it instead.
    const bound = await yian(["run", flowFile(APPEND.replace("id: answer", 'id: "Answer questions"')), "--quiet", "--memory", `lessons=${join(tmp, "n")}`]);
    expect(bound.code, bound.stderr).toBe(0);
  });
});
