/**
 * A resume binds the memories its run started with again, as instances: it never creates one,
 * refuses one that differs from the recorded value, keeps the recorded recall, and reports in its
 * result a memory that changed while a crashed run was down.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../src/compose/loader.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { FileMemory } from "../src/memory/file.ts";
import { withWarnings } from "../src/platform.ts";
import { advance, agentCall, agentCalls, complete, foldRunLog, inspectRun, type RunFlowOptions, runFlow } from "../src/run.ts";
import { pending } from "../src/tools/context.ts";
import { FunctionTool } from "../src/tools/function_tools.ts";
import { FakeMemory, fileInfo, RecordingSession } from "./helpers/memory_fakes.ts";
import { callReply, crashAfter, NO_ENV, serving, textReply } from "./helpers/run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-run-memory-resume-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

type Reply = ReturnType<typeof textReply>;

const FLOW = `
id: f
name: f
llm_config: {provider: anthropic, model: claude-test}
input:
  lessons: !FileMemory
  notes: !Optional(FileMemory)
nodes:
  answer:
    kind: agent
    input:
      lessons: \${input.lessons}
    tools: [{lessons: [append]}]
    prompt: "<memory>\${lessons}</memory> go"
    output: str
output: \${answer.output}
`;

/** A fresh memory folder holding `files`. */
function folder(files: Record<string, string> = {}): string {
  const root = mkdtempSync(join(tmp, "m-"));
  for (const [path, text] of Object.entries(files)) writeFileSync(join(root, path), text);
  return root;
}

/** A fresh run folder's log path. */
function runLog(): string {
  return join(mkdtempSync(join(tmp, "run-")), "log.jsonl");
}

/** A scripted model over `replies`. */
function scripted(replies: Reply[]): ScriptedModel {
  return new ScriptedModel(replies, { provider: "anthropic", model: "claude-test" });
}

/** The replies of a run that appends `second` to `a.md`, then answers. */
const appendReplies = (): Reply[] => [callReply(["lessons_append", { path: "a.md", text: "second" }, "c1"]), textReply("done")];

/** Run `FLOW` over `log`, crashing once the reply's `model_call` step is written. */
async function crashedRun(log: string, inputs: Record<string, unknown>, options: RunFlowOptions = {}): Promise<void> {
  // The k-th record is the AGENT's first node step (its model_call); the append runs after it.
  const probe = runLog();
  await runFlow(await loadFlow(FLOW), { lessons: new FileMemory({ root: folder(), id: "t/probe" }) }, {
    models: serving(scripted(appendReplies())),
    env: NO_ENV,
    log: probe,
  });
  const k = readFileSync(probe, "utf8").split("\n").findIndex((line) => line.includes('"model_call"')) + 1;
  await crashAfter(k, async () =>
    runFlow(await loadFlow(FLOW), inputs, { models: serving(scripted(appendReplies())), env: NO_ENV, log, ...options }),
  );
}

test("after a crash once a lessons_append call started, a resume sends it again with the same id, and it lands once", async () => {
  const root = folder({ "a.md": "first" });
  const log = runLog();
  await crashedRun(log, { lessons: new FileMemory({ root, id: "t/lessons" }) });
  // The append ran before the crash cut its result.
  expect(readFileSync(join(root, "a.md"), "utf8")).toBe("first\nsecond");
  const second = scripted([textReply("done")]);
  const resumed = await advance({ log, models: serving(second), env: NO_ENV, memory: { lessons: new FileMemory({ root, id: "t/lessons" }) } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(readFileSync(join(root, "a.md"), "utf8")).toBe("first\nsecond");
  // The recall stays the recorded one, and the changed folder is reported as drift.
  expect(second.requests[0]!.messages.find((message) => message.role === "user")!.text()).toBe("<memory>a.md: first</memory> go");
  expect(resumed.memoryDrift).toHaveLength(1);
  expect(resumed.memoryDrift[0]).toMatchObject({ input: "lessons", identity: "t/lessons" });
  expect(resumed.memoryDrift[0]!.recorded).not.toBe(resumed.memoryDrift[0]!.current);
  expect(Object.isFrozen(resumed.memoryDrift)).toBe(true);
});

test("a resume without the memory, or with a missing folder, is refused, and nothing is created", async () => {
  const root = folder({ "a.md": "first" });
  const log = runLog();
  await crashedRun(log, { lessons: new FileMemory({ root, id: "t/lessons" }) });
  const before = readFileSync(log, "utf8");
  const missing = "memory `lessons` (`t/lessons`) isn't at its location; bind it with --memory";
  await expect(advance({ log, models: serving(scripted([])), env: NO_ENV })).rejects.toThrow(missing);
  const gone = join(tmp, "nowhere");
  await expect(
    advance({ log, models: serving(scripted([])), env: NO_ENV, memory: { lessons: new FileMemory({ root: gone, id: "t/lessons" }) } }),
  ).rejects.toThrow(missing);
  expect(() => readFileSync(join(gone, ".yian/identity"))).toThrow();
  expect(readFileSync(log, "utf8")).toBe(before);
});

test("a different identity, type, settings or tool spec is refused", async () => {
  const root = folder({ "a.md": "first" });
  const log = runLog();
  await crashedRun(log, { lessons: new FileMemory({ root, id: "t/lessons" }) });
  const resume = (memory: unknown) =>
    advance({ log, models: serving(scripted([textReply("done")])), env: NO_ENV, memory: { lessons: memory as FakeMemory } });
  const other = new FakeMemory(() => new RecordingSession(fileInfo("t/other")));
  await expect(resume(other)).rejects.toThrow("resume refused: memory lessons is `t/other`, not the recorded `t/lessons`");
  await expect(resume(new FileMemory({ root, id: "t/lessons", maxFiles: 3 }))).rejects.toThrow(
    "resume refused: memory lessons settings differ from the recorded ones",
  );
  const recorded = (await new FileMemory({ root, id: "t/lessons" }).open({ create: false, itemType: null, signal: new AbortController().signal })).info;
  const retooled = new FakeMemory(
    () =>
      new RecordingSession({
        ...recorded,
        tools: recorded.tools.map((tool) => (tool.name === "read" ? { ...tool, description: "Read it." } : tool)),
      }),
  );
  await expect(resume(retooled)).rejects.toThrow("resume refused: memory lessons tools differ from the recorded ones");
  await expect(resume({ open() {} })).rejects.toThrow("memory `lessons` takes a MemoryBase instance");
  await expect(
    advance({ log, models: serving(scripted([])), env: NO_ENV, memory: { lessons: new FileMemory({ root }), stray: new FakeMemory() } }),
  ).rejects.toThrow("memory `stray` isn't a memory input of this run's flow");
});

test("an unchanged folder gives no drift, and the recall stays the recorded one", async () => {
  const root = folder({ "MEMORY.md": "v1" });
  const log = runLog();
  const memory = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/lessons"));
    session.onSnapshot = () => ({ version: "v1", recall: { text: "v1", cut: null, hint: null } });
    return session;
  });
  await crashedRun(log, { lessons: memory });
  const second = scripted([textReply("done")]);
  const resumed = await advance({ log, models: serving(second), env: NO_ENV, memory: { lessons: memory } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.memoryDrift).toEqual([]);
  expect(memory.openedWith.at(-1)).toEqual({ create: false, itemType: null });
  expect(memory.sessions.at(-1)!.snapshots).toBe(1);
  expect(memory.sessions.at(-1)!.closed).toBe(1);
  // The resent append went through the resumed run's session, with the reply's call id.
  expect(memory.sessions.at(-1)!.applied.map((each) => each.id)).toEqual([memory.sessions[0]!.applied[0]!.id]);
  expect(root).toBeTruthy();
});

test("advancing a waiting run opens the memories without a snapshot and reports no drift", async () => {
  const text = `
id: w
name: w
input:
  lessons: !FileMemory
nodes:
  hold:
    kind: tool
    tool_id: hold
output: \${hold.output}
`;
  const hold = new FunctionTool({ name: "hold", description: "Hold.", parametersJsonSchema: { type: "object" }, execute: () => pending() });
  const memory = new FakeMemory(() => new RecordingSession(fileInfo("t/lessons")));
  const log = runLog();
  const waiting = await runFlow(await loadFlow(text), { lessons: memory }, { env: NO_ENV, log, tools: { hold } });
  expect(waiting.status).toBe("waiting");
  expect(waiting.memoryDrift).toEqual([]);
  await complete(log, waiting.waits[0]!.callId, { output: "x" });
  const done = await advance({ log, env: NO_ENV, tools: { hold }, memory: { lessons: memory } });
  expect(done.status).toBe("succeeded");
  expect(done.memoryDrift).toEqual([]);
  expect(memory.opens).toBe(2);
  expect(memory.sessions[1]!.snapshots).toBe(0);
});

test("a resumed segment's memory that fails to close is a warning; the result stands", async () => {
  const text = `
id: w
name: w
input:
  lessons: !FileMemory
nodes:
  hold:
    kind: tool
    tool_id: hold
output: \${hold.output}
`;
  const hold = new FunctionTool({ name: "hold", description: "Hold.", parametersJsonSchema: { type: "object" }, execute: () => pending() });
  const memory = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/lessons"));
    session.close = async () => {
      throw new Error("rpc close failed");
    };
    return session;
  });
  const log = runLog();
  const warnings: string[] = [];
  const waiting = await withWarnings(
    (message) => warnings.push(message),
    async () => runFlow(await loadFlow(text), { lessons: memory }, { env: NO_ENV, log, tools: { hold } }),
  );
  expect(waiting.status).toBe("waiting");
  await complete(log, waiting.waits[0]!.callId, { output: "x" });
  const done = await withWarnings(
    (message) => warnings.push(message),
    () => advance({ log, env: NO_ENV, tools: { hold }, memory: { lessons: memory } }),
  );
  expect(done.status).toBe("succeeded");
  expect(warnings).toEqual([
    "memory `t/lessons` didn't close: rpc close failed",
    "memory `t/lessons` didn't close: rpc close failed",
  ]);
});

test("a memory recorded null needs no instance on resume, and one passed for it is refused", async () => {
  const root = folder({ "a.md": "first" });
  const log = runLog();
  await crashedRun(log, { lessons: new FileMemory({ root, id: "t/lessons" }) });
  await expect(
    advance({
      log,
      models: serving(scripted([textReply("done")])),
      env: NO_ENV,
      memory: { lessons: new FileMemory({ root }), notes: new FakeMemory() },
    }),
  ).rejects.toThrow("memory `notes` was null when the run started; drop it");
  // null for it, as runFlow takes one, is the same as leaving it out; a required memory takes none.
  await expect(
    advance({ log, models: serving(scripted([])), env: NO_ENV, memory: { lessons: null, notes: null } }),
  ).rejects.toThrow("memory `lessons` takes a MemoryBase instance");
  const resumed = await advance({
    log,
    models: serving(scripted([textReply("done")])),
    env: NO_ENV,
    memory: { lessons: new FileMemory({ root }), notes: null },
  });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
});

test("inspectRun, foldRunLog and agentCall need no memory", async () => {
  const root = folder({ "a.md": "first" });
  const log = runLog();
  const result = await runFlow(await loadFlow(FLOW), { lessons: new FileMemory({ root, id: "t/lessons" }) }, {
    models: serving(scripted(appendReplies())),
    env: NO_ENV,
    log,
  });
  expect(result.status).toBe("succeeded");
  expect(inspectRun(log).status).toBe("succeeded");
  expect(foldRunLog(log)).toBeTruthy();
  const [first] = agentCalls(log);
  const view = await agentCall(log, first!, { env: NO_ENV });
  expect(view.toolResults).toHaveLength(1);
});

test("runFlow still refuses a memory option", async () => {
  await expect(
    runFlow(await loadFlow(FLOW), {}, { env: NO_ENV, memory: {} } as RunFlowOptions),
  ).rejects.toThrow("the memory option was removed: pass each memory as a flow input");
});
