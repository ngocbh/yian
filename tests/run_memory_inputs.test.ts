/**
 * A run binds each memory input to the instance its host passes, before `RunStarted`: the memory's
 * value is recorded in START's inputs, and its session closes when the run's segment ends. A
 * VectorDBMemory input opens with its item type's JSON Schema.
 */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../src/compose/loader.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { FileMemory } from "../src/memory/file.ts";
import { VectorDBMemory } from "../src/memory/vectordb.ts";
import { openSqlite, withWarnings } from "../src/platform.ts";
import * as runModule from "../src/run.ts";
import { hostInputs, inspectRun, type RunFlowOptions, runFlow } from "../src/run.ts";
import { FunctionTool } from "../src/tools/function_tools.ts";
import type { JsonObject } from "../src/typesys/json.ts";
import { FakeMemory, fileInfo, RecordingSession, wordEmbed } from "./helpers/memory_fakes.ts";
import { callReply, NO_ENV, serving, textReply } from "./helpers/run_state_fixtures.ts";

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
    output: !str
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

test("the run entry point doesn't export how it reads memory bindings", () => {
  expect("memoryBindings" in runModule).toBe(false);
});

/** A flow whose TOOL node `found` searches `cases`, typed `Case`, for `q`. */
const SEARCH_FLOW = `
id: cases
name: cases
typedefs:
  Case:
    title: str
    tags: list(str)
input:
  q: !str
  cases: !VectorDBMemory(Case)
nodes:
  found:
    kind: tool
    tool_id: {cases: search}
    input:
      query: \${input.q}
      k: 3
output: \${found.output}
`;

/** The JSON Schema `typeToSchema` gives `Case`, which a `!VectorDBMemory(Case)` store keeps. */
const CASE_SCHEMA = {
  properties: {
    title: { title: "Title", type: "string" },
    tags: { items: { type: "string" }, title: "Tags", type: "array" },
  },
  required: ["title", "tags"],
  title: "Record",
  type: "object",
};

/** A fresh store path in a fresh folder. */
function storePath(): string {
  return join(mkdtempSync(join(tmp, "v-")), "cases.db");
}

/** A VectorDBMemory on `path`, embedded by `embed` as `openai/small`. */
function cases(path: string, embed = wordEmbed()): VectorDBMemory {
  return new VectorDBMemory({ path, id: "t/cases", embedding: { provider: "openai", model: "small" }, embed });
}

/** The store's settings as its `meta` table holds them. */
function storedSettings(path: string): JsonObject {
  const db = openSqlite(path, { readonly: true });
  try {
    return JSON.parse(String(db.prepare("SELECT value FROM meta WHERE key = 'settings'").get()!.value));
  } finally {
    db.close();
  }
}

/** Add `texts` to the store at `path` as a host, each with a `Case`'s fields. */
async function seed(path: string, texts: string[]): Promise<void> {
  const signal = new AbortController().signal;
  const session = await cases(path).open({ create: true, itemType: CASE_SCHEMA, signal });
  for (const [i, text] of texts.entries()) {
    const change = { op: "add", text, fields: { title: `case ${i + 1}`, tags: ["seed"] } };
    const applied = await session.apply({ id: `seed-${i}`, change, author: { kind: "host", name: "seed" }, signal });
    expect(applied.ok).toBe(true);
  }
  await session.close();
}

test("runFlow binds a VectorDBMemory(Case) input with Case's JSON Schema as its item type", async () => {
  const path = storePath();
  const { log, result } = await run(SEARCH_FLOW, { q: "alpha", cases: cases(path) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(storedSettings(path)).toEqual({ embedding: { model: "small", provider: "openai" }, item_schema: CASE_SCHEMA });
  const value = startInputs(log)["cases"]!["memory"] as JsonObject;
  expect(value).toMatchObject({ identity: "t/cases", type: "VectorDBMemory", settings: { item_schema: CASE_SCHEMA } });
  const add = (value["tools"] as JsonObject[]).find((tool) => tool["name"] === "add")!;
  expect((add["parameters"] as { properties: JsonObject }).properties["fields"]).toMatchObject({
    properties: CASE_SCHEMA.properties,
    required: CASE_SCHEMA.required,
  });
  expect(readFileSync(log, "utf8")).not.toContain(path);
});

test("a store whose items have another type fails before RunStarted, with no log", async () => {
  const path = storePath();
  const other = { type: "object", properties: { name: { type: "string" } }, required: ["name"] };
  const signal = new AbortController().signal;
  await (await cases(path).open({ create: true, itemType: other, signal })).close();
  const log = runLog();
  await expect(
    runFlow(await loadFlow(SEARCH_FLOW), { q: "alpha", cases: cases(path) }, { env: NO_ENV, log }),
  ).rejects.toThrow("memory `cases`: this memory's items have another type than its input's");
  expect(existsSync(log)).toBe(false);
  // A bare VectorDBMemory input gives no item type, which a typed store refuses too.
  const bare = SEARCH_FLOW.replace("!VectorDBMemory(Case)", "!VectorDBMemory");
  await expect(runFlow(await loadFlow(bare), { q: "alpha", cases: cases(path) }, { env: NO_ENV, log })).rejects.toThrow(
    "memory `cases`: this memory's items have typed fields; give its input their type",
  );
});

test("a TOOL node tool_id: {cases: search} returns the hits as ${found.output}", async () => {
  const path = storePath();
  await seed(path, ["alpha bravo", "charlie", "alpha alpha"]);
  const embed = wordEmbed();
  const { result } = await run(SEARCH_FLOW, { q: "alpha", cases: cases(path, embed) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const items = (result.output as { items: JsonObject[] }).items;
  expect(items.map((item) => [item["id"], item["text"]])).toEqual([
    [3n, "alpha alpha"],
    [1n, "alpha bravo"],
    [2n, "charlie"],
  ]);
  expect(items[0]!["fields"]).toEqual({ title: "case 3", tags: ["seed"] });
  expect(embed.calls).toEqual([{ texts: ["alpha"], model: "openai/small" }]);
});

test("an agent granted {cases: [search, add]} adds an item with its fields", async () => {
  const text = `
id: keep
name: keep
llm_config: {provider: anthropic, model: claude-test}
typedefs:
  Case:
    title: str
    tags: list(str)
input:
  cases: !VectorDBMemory(Case)
nodes:
  keep:
    kind: agent
    input:
      cases: \${input.cases}
    tools: [{cases: [search, add]}]
    prompt: "<memory>\${cases}</memory> keep this case"
    output: !str
output: \${keep.output}
`;
  const path = storePath();
  const change = { text: "bravo outage", fields: { title: "Outage", tags: ["ops"] } };
  const model = new ScriptedModel([callReply(["cases_add", change, "c1"]), textReply("kept")], {
    provider: "anthropic",
    model: "claude-test",
  });
  const log = runLog();
  const result = await runFlow(await loadFlow(text), { cases: cases(path) }, {
    models: serving(model),
    env: NO_ENV,
    log,
  });
  expect([result.status, result.output]).toEqual(["succeeded", "kept"]);
  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual(["cases_search", "cases_add"]);
  const signal = new AbortController().signal;
  const session = await cases(path).open({ create: false, itemType: CASE_SCHEMA, signal });
  expect(await session.call("read", { id: 1n }, { signal })).toMatchObject({
    id: 1n,
    text: "bravo outage",
    fields: { title: "Outage", tags: ["ops"] },
    author: { kind: "agent", name: "keep" },
  });
  await session.close();
});
