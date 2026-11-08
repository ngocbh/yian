/**
 * A run binds each memory input to the instance its host passes, before `RunStarted`: the memory's
 * value is recorded in START's inputs, and its session closes when the run's segment ends.
 */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../src/compose/loader.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { FileMemory } from "../src/memory/file.ts";
import { withWarnings } from "../src/platform.ts";
import { hostInputs, inspectRun, type RunFlowOptions, runFlow } from "../src/run.ts";
import { FunctionTool } from "../src/tools/function_tools.ts";
import type { JsonObject } from "../src/typesys/json.ts";
import { FakeMemory, fileInfo, RecordingSession } from "./helpers/memory_fakes.ts";
import { NO_ENV, serving, textReply } from "./helpers/run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-run-memory-inputs-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const FLOW = `
id: answer
name: answer
llm_config: {provider: anthropic, model: claude-test}
input:
  question: !str
  lessons: !FileMemory
  notes: !Optional(FileMemory)
nodes:
  answer:
    kind: agent
    input:
      question: \${input.question}
    prompt: \${question}
    output: str
output: \${answer.output}
`;

/** A fresh run folder's log path. */
function runLog(): string {
  return join(mkdtempSync(join(tmp, "run-")), "log.jsonl");
}

/** A scripted model answering `done` once. */
function model(): ScriptedModel {
  return new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
}

/** Run `text` with `inputs` over a fresh log. */
async function run(text: string, inputs: Record<string, unknown>, options: RunFlowOptions = {}) {
  const log = runLog();
  const result = await runFlow(await loadFlow(text), inputs, { models: serving(model()), env: NO_ENV, log, ...options });
  return { log, result };
}

/** The START inputs `pool_commit`'s value in the log at `log`. */
function startInputs(log: string): Record<string, JsonObject> {
  for (const line of readFileSync(log, "utf8").trim().split("\n")) {
    const record = JSON.parse(line) as { kind: string; payload: { key?: string; value?: unknown } };
    if (record.kind === "pool_commit" && record.payload.key === "__start__") {
      return record.payload.value as Record<string, JsonObject>;
    }
  }
  throw new Error("no START inputs commit");
}

test("runFlow binds a FileMemory input, records its value in START's inputs, and closes it", async () => {
  const root = mkdtempSync(join(tmp, "m-"));
  writeFileSync(join(root, "MEMORY.md"), "Prefer short answers.");
  const { log, result } = await run(FLOW, { question: "q", lessons: new FileMemory({ root, id: "t/lessons" }) });
  expect(result.status).toBe("succeeded");
  const start = startInputs(log);
  expect(start["lessons"]!["memory"]).toMatchObject({
    identity: "t/lessons",
    type: "FileMemory",
    recall: { text: "Prefer short answers.", cut: null },
  });
  expect(JSON.stringify(start)).not.toContain(root); // no location in the log
  expect(readFileSync(log, "utf8")).not.toContain(root);
  expect(result.input["question"]).toBe("q");
});

test("each session closes when the run ends, and one instance bound twice opens once", async () => {
  const memory = new FakeMemory(() => new RecordingSession(fileInfo("t/x")));
  const { result, log } = await run(FLOW, { question: "q", lessons: memory, notes: memory });
  expect(result.status).toBe("succeeded");
  expect(memory.opens).toBe(1);
  expect(memory.sessions[0]!.closed).toBe(1);
  const start = startInputs(log);
  expect(start["lessons"]).toEqual(start["notes"]!);
});

test("two instances with one identity fail before RunStarted, with no log", async () => {
  const root = mkdtempSync(join(tmp, "m-"));
  const log = runLog();
  const first = new FileMemory({ root, id: "t/x" });
  const second = new FileMemory({ root, id: "t/x" });
  await expect(
    runFlow(await loadFlow(FLOW), { question: "q", lessons: first, notes: second }, {
      models: serving(model()),
      env: NO_ENV,
      log,
    }),
  ).rejects.toThrow("`notes` and `lessons` are two memories with identity `t/x`; bind one instance");
  expect(existsSync(log)).toBe(false);
});

test("an Optional memory left out, or given null, is left out of START's inputs", async () => {
  const lessons = new FakeMemory(() => new RecordingSession(fileInfo("t/lessons")));
  for (const inputs of [{ question: "q", lessons }, { question: "q", lessons, notes: null }]) {
    const { result, log } = await run(FLOW, inputs);
    expect(result.status).toBe("succeeded");
    expect(Object.keys(startInputs(log)).sort()).toEqual(["lessons", "question"]);
  }
});

test("hostInputs takes an instance or, for an Optional memory, null, and leaves both out of the values", async () => {
  const flow = await loadFlow(FLOW);
  const lessons = new FakeMemory();
  expect(hostInputs(flow.input, { question: "q", lessons, notes: null })).toEqual({ question: "q" });
  expect(() => hostInputs(flow.input, { lessons: null })).toThrow(
    "memory `lessons` is bound to a MemoryBase instance, not a value",
  );
  expect(() => hostInputs(flow.input, { notes: "notes.md" })).toThrow(
    "memory `notes` is bound to a MemoryBase instance, not a value",
  );
});

test("a memory input named like a host tool group fails at run start", async () => {
  const text = FLOW.replace("    prompt: ", "    tools: [lessons]\n    prompt: ");
  const memory = new FakeMemory();
  const tool = new FunctionTool({
    name: "read",
    description: "Read.",
    parametersJsonSchema: { type: "object" },
    execute: () => "x",
  });
  await expect(run(text, { question: "q", lessons: memory }, { tools: { lessons: [tool] } })).rejects.toThrow(
    "AGENT node 'answer': `lessons` is both a memory input and a tool group of this run; rename one",
  );
  const named = new FunctionTool({ ...tool, name: "lessons" });
  await expect(run(text, { question: "q", lessons: memory }, { tools: { lessons: named } })).rejects.toThrow(
    "AGENT node 'answer': `lessons` is both a memory input and a tool of this run; rename one",
  );
  expect(memory.opens).toBe(0);
});

test("a memory tool whose visible name is a host tool's fails at run start", async () => {
  const text = FLOW.replace("    prompt: ", "    tools: [{lessons: [read]}, lessons_read]\n    prompt: ");
  const tool = new FunctionTool({
    name: "lessons_read",
    description: "Read.",
    parametersJsonSchema: { type: "object" },
    execute: () => "x",
  });
  const memory = new FakeMemory();
  await expect(run(text, { question: "q", lessons: memory }, { tools: { lessons_read: tool } })).rejects.toThrow(
    "AGENT node 'answer': the memory tool `lessons_read` of `{lessons: [read]}` has the name of a host tool it grants; " +
      "rename one",
  );
  expect(memory.opens).toBe(0);
  // A whole grant shows every tool of the memory's type.
  const whole = FLOW.replace("    prompt: ", "    tools: [lessons, lessons_read]\n    prompt: ");
  await expect(run(whole, { question: "q", lessons: memory }, { tools: { lessons_read: tool } })).rejects.toThrow(
    "the memory tool `lessons_read` of `lessons` has the name of a host tool it grants",
  );
});

test("a memory that fails to close is a warning; the run's result stands", async () => {
  const memory = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/lessons"));
    session.close = async () => {
      throw new Error("rpc close failed");
    };
    return session;
  });
  const warnings: string[] = [];
  const { result, log } = await withWarnings(
    (message) => warnings.push(message),
    () => run(FLOW, { question: "q", lessons: memory }),
  );
  expect(result.status).toBe("succeeded");
  expect(inspectRun(log).status).toBe("succeeded");
  expect(warnings).toEqual(["memory `t/lessons` didn't close: rpc close failed"]);
});

test("a memory that fails to close after the call failed is noted on its error, not warned", async () => {
  const memory = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/lessons"));
    session.close = async () => {
      throw new Error("rpc close failed");
    };
    return session;
  });
  // An existing log path fails the call once the memories are open.
  const taken = runLog();
  writeFileSync(taken, "");
  const warnings: string[] = [];
  const error = await withWarnings(
    (message) => warnings.push(message),
    async () => runFlow(await loadFlow(FLOW), { question: "q", lessons: memory }, { env: NO_ENV, log: taken }),
  ).then(
    () => null,
    (thrown: unknown) => thrown as Error & { code?: string; notes?: string[] },
  );
  expect(error?.code).toBe("EEXIST");
  expect(memory.opens).toBe(1);
  expect(error?.notes).toEqual(["closing memory `t/lessons` also raised Error: rpc close failed"]);
  expect(warnings).toEqual([]);
});
