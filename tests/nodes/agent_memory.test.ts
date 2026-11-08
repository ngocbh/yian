/**
 * An AGENT and the memories its flow binds: a memory input renders its recorded recall in prompt
 * text, with the hint only when the agent holds the hint's tool on that memory, and a granted
 * memory's tools reach the model as `<input>_<tool>`, routed to the run's session.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Request } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { FileMemory } from "../../src/memory/file.ts";
import { type RunFlowOptions, runFlow } from "../../src/run.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { FakeMemory, fileInfo, RecordingSession } from "../helpers/memory_fakes.ts";
import { callReply, NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-agent-memory-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

type Reply = ReturnType<typeof textReply>;

/** A fresh memory folder holding `files`. */
function folder(files: Record<string, string>): string {
  const root = mkdtempSync(join(tmp, "m-"));
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(join(root, path, ".."), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}

/** A flow taking `inputs` (YAML lines), with the nodes `nodes` (YAML), whose output is `output`. */
function flow(inputs: string, nodes: string, output = "${answer.output}"): string {
  const indent = (text: string) => text.trim().split("\n").map((line) => `  ${line}`).join("\n");
  return `id: f
name: f
llm_config: {provider: anthropic, model: claude-test}
input:
${indent(inputs)}
nodes:
${indent(nodes)}
output: ${output}
`;
}

/** Run `text` with `inputs` over a scripted model answering `replies`; return the result, the model and the log. */
async function run(text: string, inputs: Record<string, unknown>, replies: Reply[], options: RunFlowOptions = {}) {
  const model = new ScriptedModel(replies, { provider: "anthropic", model: "claude-test" });
  const log = join(mkdtempSync(join(tmp, "run-")), "log.jsonl");
  const result = await runFlow(await loadFlow(text), inputs, { models: serving(model), env: NO_ENV, log, ...options });
  return { result, model, log };
}

/** The text of the first user message of `request`. */
function firstUser(request: Request): string {
  return request.messages.find((message) => message.role === "user")!.text();
}

/** The `tool_result` steps the log at `log` holds, in order. */
function toolResults(log: string): JsonObject[] {
  return readFileSync(log, "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line) as { kind: string; payload: { step?: JsonObject } })
    .filter((record) => record.kind === "node_step" && record.payload.step?.["kind"] === "tool_result")
    .map((record) => record.payload.step!);
}

const LESSONS = `
question: !str
lessons: !FileMemory
`;

/** An agent `answer` with `tools` whose prompt frames `${<name>}` in a memory tag. */
function answer(tools: string, input = "lessons: ${input.lessons}", name = "lessons"): string {
  return `
answer:
  kind: agent
  input:
    ${input}
    question: \${input.question}
  tools: ${tools}
  prompt: |-
    <memory name="lessons">
    \${${name}}
    </memory>
    \${question}
  output: str
`;
}

test("the prompt renders the recorded recall; the hint shows only when the agent holds lessons_read", async () => {
  const root = folder({ "MEMORY.md": "Prefer short answers.", "notes/a.md": "A." });
  const lessons = new FileMemory({ root, id: "t/lessons" });
  const held = await run(flow(LESSONS, answer("[{lessons: [read]}]")), { question: "q", lessons }, [textReply("ok")]);
  expect(held.result.status).toBe("succeeded");
  expect(firstUser(held.model.requests[0]!)).toBe(
    '<memory name="lessons">\nPrefer short answers.\nRead the rest with lessons_read.\n</memory>\nq',
  );
  const appendOnly = await run(flow(LESSONS, answer("[{lessons: [append]}]")), { question: "q", lessons }, [
    textReply("ok"),
  ]);
  expect(firstUser(appendOnly.model.requests[0]!)).toBe('<memory name="lessons">\nPrefer short answers.\n</memory>\nq');
});

test("a renamed input's hint names the grant's input", async () => {
  const root = folder({ "MEMORY.md": "Prefer short answers.", "b.md": "B." });
  const lessons = new FileMemory({ root, id: "t/lessons" });
  const text = flow(LESSONS, answer("[{lessons: [read]}]", "notes: ${input.lessons}", "notes"));
  const { model } = await run(text, { question: "q", lessons }, [textReply("ok")]);
  expect(firstUser(model.requests[0]!)).toContain("Read the rest with lessons_read.");
});

test("an agent that holds read only on another memory shows no hint", async () => {
  const a = new FileMemory({ root: folder({ "MEMORY.md": "A", "x.md": "x" }), id: "t/a" });
  const b = new FileMemory({ root: folder({ "MEMORY.md": "B" }), id: "t/b" });
  const text = flow("question: !str\na: !FileMemory\nb: !FileMemory", answer("[{b: [read]}]", "notes: ${input.a}", "notes"));
  const { model } = await run(text, { question: "q", a, b }, [textReply("ok")]);
  expect(firstUser(model.requests[0]!)).toBe('<memory name="lessons">\nA\n</memory>\nq');
});

test("a null Optional memory renders empty and gives the agent no tools from it", async () => {
  const text = flow(
    "question: !str\nlessons: !Optional(FileMemory)",
    answer("[lookup, {lessons: [read, append]}]"),
  );
  const lookup = new FunctionTool({
    name: "lookup",
    description: "Look up.",
    parametersJsonSchema: { type: "object" },
    execute: () => "x",
  });
  const { result, model } = await run(text, { question: "q" }, [textReply("ok")], { tools: { lookup } });
  expect(result.status).toBe("succeeded");
  expect(firstUser(model.requests[0]!)).toBe('<memory name="lessons">\n\n</memory>\nq');
  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual(["lookup"]);
});

test("the memory tools follow the grant order among the host tools, before load_skill", async () => {
  const lessons = new FileMemory({ root: folder({}), id: "t/lessons" });
  const lookup = new FunctionTool({
    name: "lookup",
    description: "Look up.",
    parametersJsonSchema: { type: "object" },
    execute: () => "x",
  });
  const skills = new Map([["style", new Skill({ name: "style", description: "Style.", instructions: "Be brief." })]]);
  const withSkills = (tools: string) => answer(tools).replace("  output: str", "  skills: [style]\n  output: str");
  for (const [tools, names] of [
    ["[lookup, {lessons: [read, append]}]", ["lookup", "lessons_read", "lessons_append", "load_skill"]],
    ["[{lessons: [read]}, lookup]", ["lessons_read", "lookup", "load_skill"]],
  ] as const) {
    const { model } = await run(flow(LESSONS, withSkills(tools)), { question: "q", lessons }, [textReply("ok")], {
      tools: { lookup },
      skills,
    });
    expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual([...names]);
  }
  const { model } = await run(flow(LESSONS, answer("[lessons]")), { question: "q", lessons }, [textReply("ok")]);
  const specs = model.requests[0]!.tools;
  expect(specs.map((tool) => tool.name)).toEqual(FileMemory.contract.tools.map((tool) => `lessons_${tool.name}`));
  expect(specs[1]!.description).toBe(FileMemory.contract.tools[1]!.description);
  expect(specs[1]!.parameters).toEqual(FileMemory.contract.tools[1]!.parameters);
});

test("the model's lessons_append lands in the folder and its tool_result holds path, hash and version", async () => {
  const root = folder({ "a.md": "first" });
  const lessons = new FileMemory({ root, id: "t/lessons" });
  const { result, model, log } = await run(flow(LESSONS, answer("[{lessons: [append, edit]}]")), { question: "q", lessons }, [
    callReply(["lessons_append", { path: "a.md", text: "second" }, "c1"]),
    callReply(["lessons_edit", { path: "a.md", old: "missing", new: "x" }, "c2"]),
    textReply("done"),
  ]);
  expect(result.status).toBe("succeeded");
  expect(readFileSync(join(root, "a.md"), "utf8")).toBe("first\nsecond");
  const [appended, conflict] = toolResults(log);
  expect(appended!["is_error"]).toBe(false);
  const output = JSON.parse(appended!["output"] as string) as JsonObject;
  expect(Object.keys(output).sort()).toEqual(["hash", "path", "version"]);
  expect(output["path"]).toBe("a.md");
  // A conflict reaches the model as an error result holding the current text.
  expect(conflict!["is_error"]).toBe(true);
  expect(conflict!["output"]).toContain("the old text isn't in `a.md`");
  expect(conflict!["output"]).toContain('"text":"first\\nsecond"');
  expect(model.requests).toHaveLength(3);
  expect(readFileSync(log, "utf8")).not.toContain(root);
});

test("a session fault fails the node with node.execute", async () => {
  const memory = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/lessons"));
    session.onApply = () => {
      throw new Error("disk full");
    };
    return session;
  });
  const { result } = await run(flow(LESSONS, answer("[{lessons: [append]}]")), { question: "q", lessons: memory }, [
    callReply(["lessons_append", { path: "a.md", text: "x" }, "c1"]),
    textReply("done"),
  ]);
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("node.execute");
  expect(result.failure!.message).toContain("disk full");
  expect(memory.sessions[0]!.applied[0]!.author).toMatchObject({ kind: "agent", name: "answer", node: "answer" });
});

test("two granted memory inputs bound to one identity fail the node at bind with contract.input", async () => {
  const text = flow(
    "question: !str\nlessons: !FileMemory\nnotes: !FileMemory",
    answer("[lessons, notes]"),
  );
  const memory = new FakeMemory(() => new RecordingSession(fileInfo("t/x")));
  const { result, model } = await run(text, { question: "q", lessons: memory, notes: memory }, [textReply("ok")]);
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.message).toContain("`notes` and `lessons` are the same memory; grant one");
  expect(model.requests).toHaveLength(0);
});

test("every agent sees the run-start recall, even after a write in the same run", async () => {
  const root = folder({ "MEMORY.md": "v1" });
  const lessons = new FileMemory({ root, id: "t/lessons" });
  const nodes = `
write:
  kind: agent
  input:
    lessons: \${input.lessons}
  tools: [{lessons: [append]}]
  prompt: "<memory>\${lessons}</memory> write"
  output: str
answer:
  kind: agent
  input:
    lessons: \${input.lessons}
    prior: \${write.output}
  prompt: "<memory>\${lessons}</memory> \${prior}"
  output: str
`;
  const { result, model } = await run(flow("lessons: !FileMemory", nodes), { lessons }, [
    callReply(["lessons_append", { path: "MEMORY.md", text: "v2" }, "c1"]),
    textReply("wrote"),
    textReply("ok"),
  ]);
  expect(result.status).toBe("succeeded");
  expect(readFileSync(join(root, "MEMORY.md"), "utf8")).toBe("v1\nv2");
  expect(firstUser(model.requests[2]!)).toBe("<memory>v1</memory> wrote");
});

test("two inputs bound to one instance share its session", async () => {
  const memory = new FakeMemory(() => new RecordingSession(fileInfo("t/x")));
  const text = flow(
    "lessons: !FileMemory\nnotes: !FileMemory",
    `
answer:
  kind: agent
  input:
    lessons: \${input.lessons}
  tools: [{notes: [append]}]
  prompt: "\${lessons}"
  output: str
`,
  );
  const { result } = await run(text, { lessons: memory, notes: memory }, [
    callReply(["notes_append", { path: "a.md", text: "x" }, "c1"]),
    textReply("ok"),
  ]);
  expect(result.status).toBe("succeeded");
  expect(memory.opens).toBe(1);
  expect(memory.sessions[0]!.applied.map((each) => each.change)).toEqual([{ op: "append", path: "a.md", text: "x" }]);
});
