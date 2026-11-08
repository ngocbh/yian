/**
 * `yian run --resume` binds a run's memories again: a default one is rebuilt from the identity
 * its run recorded (and checked as a fresh default is), `--memory` binds one instead, a resume
 * never creates a memory, and a crash resume notes each memory that changed since the run started.
 * The interrupted runs are started in code, with the instance `yian run` would have bound.
 */
import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { defaultIdentity, defaultLocation } from "../../src/cli/memory_bind.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import type { MemoryBase } from "../../src/memory/base.ts";
import { FileMemory } from "../../src/memory/file.ts";
import { runFlow, send } from "../../src/run.ts";

let tmp: string;
let home: string;
let vars: Record<string, string>;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-resume-mem-")));
  home = join(tmp, "home");
  vars = { HOME: home };
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

const env = (name: string) => (Object.hasOwn(vars, name) ? vars[name] : undefined);

/** Run `yian` with `args`; return the exit code and both streams. */
async function yian(args: readonly string[]) {
  const out: string[] = [];
  const err: string[] = [];
  const io: CliIo = {
    stdout: { write: async (text) => void out.push(text) },
    stderr: { write: async (text) => void err.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env,
  };
  const code = await main([...args], io);
  return { code, stdout: out.join(""), stderr: err.join("") };
}

/** Two appends to the memory's index, one after the other. */
const TWO_STEPS = `id: answer
name: answer
input:
  lessons: !FileMemory
nodes:
  first:
    kind: tool
    tool_id: {lessons: append}
    input: {path: MEMORY.md, text: first}
  second:
    kind: tool
    tool_id: {lessons: append}
    depends_on: [first]
    input: {path: MEMORY.md, text: second}
output: \${second.output.path}
`;

/** Takes a line from its inbox, then appends to the memory. */
const TAKES = `id: answer
name: answer
input:
  lessons: !FileMemory
  messages: !Inbox(str)
nodes:
  heard:
    kind: tool
    tool_id: take
    input:
      from: \${input.messages}
  keep:
    kind: tool
    tool_id: {lessons: append}
    depends_on: [heard]
    input:
      path: MEMORY.md
      text: heard
output: \${keep.output.path}
`;

/** Write `text` as `<tmp>/proj/flow.yaml`; return its path. */
function flowFile(text: string): string {
  mkdirSync(join(tmp, "proj"), { recursive: true });
  const path = join(tmp, "proj", "flow.yaml");
  writeFileSync(path, text);
  return path;
}

/** The default identity and location of `answer`'s `lessons`, as `yian run` binds them. */
function defaults(): { identity: string; location: string } {
  const identity = defaultIdentity(join(tmp, "proj"), "answer", "lessons");
  return { identity, location: defaultLocation(home, identity, "FileMemory") };
}

/** The default instance `yian run` would bind to `lessons`. */
function defaultInstance(): FileMemory {
  const { identity, location } = defaults();
  return new FileMemory({ root: location, id: identity });
}

/** Run `flow` with `lessons` into `<tmp>/run.jsonl`, interrupted as node `at` starts; return the log. */
async function interrupted(flow: string, lessons: MemoryBase, at: string): Promise<string> {
  const loaded = await loadFlow(readFileSync(flow, "utf8"), { searchPaths: [dirname(flow)] });
  const log = join(tmp, "run.jsonl");
  const onEvent = (event: RunEvent) => {
    if (event instanceof NodeStarted && event.nodeId === at) throw new Error("host interruption");
  };
  await expect(runFlow(loaded, { lessons }, { log, onEvent, env })).rejects.toThrow("host interruption");
  return log;
}

const DRIFT = "memory `lessons` changed since this run started; the run keeps its recorded recall";

test("a crashed run resumes with its default memory rebuilt from the recorded identity", async () => {
  const log = await interrupted(flowFile(TWO_STEPS), defaultInstance(), "second");
  const result = await yian(["run", "--resume", log, "--quiet"]);
  expect(result.code, result.stderr).toBe(0);
  expect(readFileSync(join(defaults().location, "MEMORY.md"), "utf8")).toBe("first\nsecond");
  // The run's own first write moved the version since it started.
  expect(result.stderr.split(DRIFT)).toHaveLength(2);
});

test("a memory that isn't at its location refuses the resume, and nothing is created", async () => {
  const log = await interrupted(flowFile(TWO_STEPS), defaultInstance(), "second");
  const { identity, location } = defaults();
  rmSync(location, { recursive: true });
  const before = readFileSync(log);
  const result = await yian(["run", "--resume", log, "--quiet"]);
  expect(result.code).toBe(1);
  expect(result.stderr).toContain(`memory \`lessons\` (\`${identity}\`) isn't at its location; bind it with --memory`);
  expect(existsSync(location)).toBe(false);
  expect(readFileSync(log)).toEqual(before);
});

test("another memory at the default location refuses the resume: a setup failure, text and --json", async () => {
  const log = await interrupted(flowFile(TWO_STEPS), defaultInstance(), "second");
  const { location } = defaults();
  rmSync(location, { recursive: true });
  mkdirSync(join(location, ".yian", "tmp"), { recursive: true });
  writeFileSync(join(location, ".yian", "identity"), "someone/else");
  const text = await yian(["run", "--resume", log, "--quiet"]);
  expect(text.code).toBe(1);
  expect(text.stderr).toContain("resume failed: memory `lessons`: this memory is");
  expect(text.stderr).not.toContain("    at ");
  const json = await yian(["run", "--resume", log, "--quiet", "--json"]);
  expect(json.code).toBe(1);
  expect(JSON.parse(json.stdout).error).toMatchObject({ code: "run.setup" });
});

test("an identity edited in the log to leave ~/.yian/memory is refused", async () => {
  const log = await interrupted(flowFile(TWO_STEPS), defaultInstance(), "second");
  const { identity } = defaults();
  writeFileSync(log, readFileSync(log, "utf8").replaceAll(`"identity":"${identity}"`, '"identity":"../../x"'));
  const result = await yian(["run", "--resume", log, "--quiet"]);
  expect(result.code).toBe(1);
  expect(result.stderr).toContain("memory location `../../x` leaves ~/.yian/memory");
  expect(existsSync(join(home, "x"))).toBe(false);
});

test("--memory on a resume binds that input instead of its default", async () => {
  const notes = join(tmp, "notes");
  const log = await interrupted(flowFile(TWO_STEPS), new FileMemory({ root: notes }), "second");
  const unbound = await yian(["run", "--resume", log, "--quiet"]);
  expect(unbound.code).toBe(1);
  expect(unbound.stderr).toContain("isn't at its location; bind it with --memory");
  const bound = await yian(["run", "--resume", log, "--quiet", "--memory", `lessons=${notes}`]);
  expect(bound.code, bound.stderr).toBe(0);
  expect(readFileSync(join(notes, "MEMORY.md"), "utf8")).toBe("first\nsecond");
});

test("a crash resume notes, once, a memory changed by hand since the run started", async () => {
  const log = await interrupted(flowFile(TWO_STEPS), defaultInstance(), "first");
  writeFileSync(join(defaults().location, "hand.md"), "Written by hand.\n");
  const result = await yian(["run", "--resume", log, "--quiet"]);
  expect(result.code, result.stderr).toBe(0);
  expect(result.stderr.split(DRIFT)).toHaveLength(2);
});

test("resuming a waiting run, even after a hand edit, notes no drift", async () => {
  const flow = flowFile(TAKES);
  const loaded = await loadFlow(readFileSync(flow, "utf8"), { searchPaths: [dirname(flow)] });
  const log = join(tmp, "run.jsonl");
  expect((await runFlow(loaded, { lessons: defaultInstance() }, { log, env })).status).toBe("waiting");
  await send(log, "messages", "heard", { env });
  writeFileSync(join(defaults().location, "hand.md"), "Written by hand.\n");
  const result = await yian(["run", "--resume", log, "--quiet"]);
  expect(result.code, result.stderr).toBe(0);
  expect(result.stderr).not.toContain("changed since this run started");
  expect(readFileSync(join(defaults().location, "MEMORY.md"), "utf8")).toBe("heard");
});
