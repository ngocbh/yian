/**
 * Tool modules at a segment's start: each segment (a run's start, each `advance`) starts one
 * thread per module its nodes reach, beside the MCP servers, and ends them when it ends. A module
 * that fails to start refuses the run before any record, naming its file (relative to the root
 * flow's folder) and export. Each node's scope holds its modules' tools for the run-start checks
 * and its model's request, and a call runs on the module's thread: its result and its errors act
 * as a host tool's do, for an AGENT's model and for TOOL retries.
 */

import { afterAll, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

import { loadFlow, withUnfinishedToolModules } from "../src/compose/loader.ts";
import { ToolResultPart } from "../src/llm/messages.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { advance, complete, runFlow, type RunFlowOptions } from "../src/run.ts";
import { pending } from "../src/tools/context.ts";
import { FunctionTool, tool } from "../src/tools/function_tools.ts";
import { mcpServer } from "../src/tools/mcp.ts";
import { FakeMemory } from "./helpers/memory_fakes.ts";
import { expectClosed, hostEnv, killMarkers, stdioOptions } from "./helpers/mcp_helpers.ts";
import { callReply, serving, textReply } from "./helpers/run_state_fixtures.ts";

const root = realpathSync(mkdtempSync(join(tmpdir(), "yian-run-tool-modules-")));
afterAll(() => rmSync(root, { recursive: true, force: true }));
let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(root, "case-"));
});

/** A module exporting a tool per name, each taking a string `id` and returning 1; `output` sets one's output schema. */
function tools(names: string[], output = ""): string {
  const parameters = '{ type: "object", properties: { id: { type: "string" } }, required: ["id"] }';
  return names
    .map((name) => `export const ${name} = { name: "${name}", description: "${name}.", parameters: ${parameters}, ${output}execute: () => 1 };\n`)
    .join("");
}

/** Source that appends `name` to `imports` when imported, and a dot to `<name>.beat` then and every 20 ms after. */
function logged(name: string): string {
  const beat = `() => appendFileSync(${JSON.stringify(join(dir, `${name}.beat`))}, ".")`;
  return `import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(join(dir, "imports"))}, "${name}\\n");
(${beat})();
setInterval(${beat}, 20);
`;
}

/** Whether module `name`'s thread ran and ended: its beat file stopped growing. */
async function ended(name: string): Promise<boolean> {
  const beat = join(dir, `${name}.beat`);
  await sleep(100);
  const before = statSync(beat).size;
  await sleep(200);
  return before > 0 && statSync(beat).size === before;
}

/** The modules imported so far, in name order. */
const imports = () => readFileSync(join(dir, "imports"), "utf8").trim().split("\n").sort();

/** A flow importing `uses` (alias to ref), with `nodes`, one per line, after `head`. */
function app(uses: Record<string, string>, nodes: string[], head = ""): string {
  const refs = Object.entries(uses).map(([alias, ref]) => `  ${alias}: "${ref}"\n`).join("");
  const body = nodes.map((node) => `  ${node}\n`).join("");
  return `id: app\nname: app\nllm_config: {provider: openai, model: test-model}\n${head}uses:\n${refs}nodes:\n${body}`;
}

/** Load `text` from `dir` and run it, tool modules allowed. */
function run(text: string, options: RunFlowOptions = {}, inputs: Record<string, unknown> = {}) {
  return withUnfinishedToolModules(async () =>
    runFlow(await loadFlow(text, { searchPaths: [dir] }), inputs, { env: hostEnv(dir), ...options }),
  );
}

const hold = tool({ name: "hold", description: "Hold.", parameters: { type: "object" }, execute: () => pending() });

test("each segment starts one thread per module its nodes reach and ends them as it ends", async () => {
  writeFileSync(join(dir, "crm.ts"), logged("crm") + tools(["create", "lookup"]));
  writeFileSync(join(dir, "quote.ts"), logged("quote") + tools(["quote"]));
  writeFileSync(join(dir, "idle.ts"), logged("idle") + tools(["idle"]));
  const uses = { crm: "./crm.ts", find: "./crm.ts:lookup", quote: "./quote.ts:quote", idle: "./idle.ts" };
  const text = app(uses, ["h: {kind: tool, tool_id: hold}", "a: {kind: agent, input: {go: '${h.output}'}, prompt: '${go}', tools: [find, crm, quote]}"]);
  const log = join(dir, "log.jsonl");
  const model = new ScriptedModel([textReply("done")]);

  const first = await run(text, { log, runId: "r1", tools: { hold } });
  expect(first.status).toBe("waiting");
  expect(imports()).toEqual(["crm", "quote"]);
  expect([await ended("crm"), await ended("quote")]).toEqual([true, true]);

  await complete(log, "r1/h", { output: "go" });
  const done = await withUnfinishedToolModules(() => advance({ log, models: serving(model), env: hostEnv(dir), tools: { hold } }));
  expect(done.status).toBe("succeeded");
  expect(imports()).toEqual(["crm", "crm", "quote", "quote"]);
  // `find` and `crm`'s `lookup` are one export, bound once, at its first grant.
  expect(model.requests[0]!.tools.map((spec) => [spec.name, spec.description])).toEqual([
    ["find", "lookup."],
    ["crm_create", "create."],
    ["quote", "quote."],
  ]);
  expect(await ended("crm")).toBe(true);
});

/** A module whose `lookup` finds a customer by id, failing for `none` (reported) and `boom`. */
const CRM = `const TOOL_ERROR = Symbol.for("yian.tool_error");
export const lookup = {
  name: "lookup",
  description: "lookup.",
  parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
  execute(args, ctx) {
    if (args.id === "none") throw Object.assign(new Error("no such customer"), { [TOOL_ERROR]: ["reported"] });
    if (args.id === "boom") throw new Error("boom");
    return { id: args.id, node: ctx.node, tool: ctx.toolName };
  },
};
`;

/** The tool results' contents in `model`'s last request. */
function resultContents(model: ScriptedModel): string[] {
  return model.requests.at(-1)!.messages.flatMap((message) =>
    message.parts.filter((part) => part instanceof ToolResultPart).map((part) => part.content),
  );
}

test("an agent's model reads a module tool's result and its reported error; any other error fails the node", async () => {
  writeFileSync(join(dir, "crm.ts"), CRM);
  writeFileSync(join(dir, "quote.ts"), tools(["quote"]));
  const text = app({ crm: "./crm.ts", price: "./quote.ts:quote" }, ["a: {kind: agent, prompt: go, tools: [crm, price]}"], "output: ${a.output}\n");
  const calls = callReply(["crm_lookup", { id: "1" }, "c1"], ["crm_lookup", { id: "none" }, "c2"], ["price", { id: "1" }, "c3"]);
  const model = new ScriptedModel([calls, textReply("done")]);

  const done = await run(text, { models: serving(model) });

  expect([done.status, done.output]).toEqual(["succeeded", "done"]);
  expect(resultContents(model)).toEqual(['{"id":"1","node":"a","tool":"lookup"}', "no such customer", "1"]);
  const failing = new ScriptedModel([callReply(["crm_lookup", { id: "boom" }, "c1"])]);
  const failed = await run(text, { models: serving(failing) });
  expect([failed.failure?.code, failed.failure?.message]).toEqual(["node.execute", expect.stringContaining("boom")]);
});

test.each([
  ["transient", null],
  ["permanent", "attempt 1 failed"],
  ["check", "attempt 1 failed"],
  ["date", "crm.ts export 'flaky' result: Date is not a yian value"],
  ["checked", "crm.ts export 'checked': a tool module's validators aren't supported yet"],
])("a tool node's retries: 2 retry a module tool's %s error only when another attempt may help", async (how, message) => {
  writeFileSync(
    join(dir, "crm.ts"),
    `const TOOL_ERROR = Symbol.for("yian.tool_error");
let calls = 0;
const parameters = { type: "object", properties: { how: { type: "string" } } };
export const flaky = {
  name: "flaky",
  description: "flaky.",
  parameters,
  execute({ how }, ctx) {
    calls += 1;
    if (how === "date") return new Date(0);
    if (ctx.attempt < 3) throw Object.assign(new Error(\`attempt \${ctx.attempt} failed\`), { [TOOL_ERROR]: [how], retryAfter: 0 });
    return { attempt: ctx.attempt, calls };
  },
};
const checker = { "~standard": { version: 1, vendor: "t", validate: (value) => ({ value }), jsonSchema: { input: () => parameters } } };
export const checked = { name: "checked", description: "checked.", parameters: checker, execute: () => 1 };
`,
  );
  const tool = how === "checked" ? "checked" : "flaky";
  const node = `t: {kind: tool, tool_id: {crm: ${tool}}, input: {how: ${how}}, retries: 2}`;

  const result = await run(app({ crm: "./crm.ts" }, [node], "output: ${t.output}\n"));

  if (message === null) expect([result.status, result.output]).toEqual(["succeeded", { attempt: 3n, calls: 3n }]);
  else expect([result.failure?.code, result.failure?.message]).toEqual(["node.execute", expect.stringContaining(message)]);
  if (message !== null) expect(result.failure?.message).not.toContain("attempt 2");
});

test.each([
  ["./tools/crm.ts", 'throw new Error("boom");\n', "tools/crm.ts failed to import: Error: boom"],
  ["./tools/crm.ts:lookp", tools(["lookup"]), "tools/crm.ts has no export 'lookp'; did you mean 'lookup'?"],
])("uses: %s refuses the run before any record: %#", async (ref, source, message) => {
  mkdirSync(join(dir, "tools"));
  writeFileSync(join(dir, "tools", "crm.ts"), source);
  const log = join(dir, "log.jsonl");
  const refused = await run(app({ crm: ref }, ["a: {kind: agent, prompt: go, tools: [crm]}"]), { log }).catch((error) => error);
  expect([refused, refused.message]).toEqual([expect.any(RangeError), message]);
  expect(existsSync(log)).toBe(false);
});

test("a stop while a module imports refuses the run and ends the threads that started", async () => {
  writeFileSync(join(dir, "crm.ts"), logged("crm") + tools(["lookup"]));
  writeFileSync(join(dir, "slow.ts"), `${logged("slow")}await new Promise(() => {});\n`);
  const text = app({ crm: "./crm.ts", slow: "./slow.ts" }, ["a: {kind: agent, prompt: go, tools: [crm, slow]}"]);
  const stop = new AbortController();
  const stopped = run(text, { signal: stop.signal });
  // Both threads must be importing before the stop; under load one may start well after the other.
  const importing = () => ["crm", "slow"].every((name) => existsSync(join(dir, `${name}.beat`)));
  for (let tries = 0; tries < 250 && !importing(); tries++) await sleep(20);
  stop.abort(new Error("stopped"));
  await expect(stopped).rejects.toThrow("stopped");
  expect([await ended("crm"), await ended("slow")]).toEqual([true, true]);
});

test("a module that fails to start closes the run's MCP servers, and a failed check its modules", async () => {
  writeFileSync(join(dir, "crm.ts"), logged("crm") + tools(["lookup"]));
  writeFileSync(join(dir, "bad.ts"), 'throw new Error("boom");\n');
  const marker = join(dir, "pid");
  try {
    const server = mcpServer(stdioOptions("--marker", marker));
    const text = app({ crm: "./crm.ts", bad: "./bad.ts" }, ["a: {kind: agent, prompt: go, tools: [crm, bad]}"]);
    await expect(run(text, { tools: { fake: server } })).rejects.toThrow("bad.ts failed to import: Error: boom");
    await expectClosed(server, marker);
  } finally {
    killMarkers([marker]);
  }
  const clash = new FunctionTool({ name: "crm_lookup", description: "d", parametersJsonSchema: { type: "object" }, execute: () => 1 });
  const text = app({ crm: "./crm.ts" }, ["a: {kind: agent, prompt: go, tools: [crm, crm_lookup]}"]);
  await expect(run(text, { tools: { crm_lookup: clash } })).rejects.toThrow(
    "AGENT node 'a' grants crm: tools 'crm_lookup' (tool crm_lookup) and 'crm_lookup' (crm's lookup) have one name; grant one",
  );
  expect(await ended("crm")).toBe(true);
}, 30_000);

test.each([
  ["t: {kind: tool, tool_id: {crm: lookup}, input: {nope: x}}", "TOOL node 't' input 'nope': tool 'crm_lookup' takes no such input; it takes [\"id\"]"],
  ["t: {kind: tool, tool_id: quote, input: {id: x}, output: !str}", "TOOL node 't' output: declared str, but tool 'quote' returns int"],
  ["t: {kind: tool, tool_id: {crm: find}, input: {id: x}}", "TOOL node 't' names {crm: find}: `crm` has no tool `find`; it has lookup"],
  ["a: {kind: agent, prompt: go, tools: [quote]}", "AGENT node 'a': `quote` is both a tool module it grants and a tool of this run; rename one"],
  ["t: {kind: tool, tool_id: quote, input: {id: x}}", "TOOL node 't': `quote` is both a tool module it calls and a tool of this run; rename one"],
  ["a: {kind: agent, prompt: go, tools: [notes, notes_read]}", "AGENT node 'a': the memory tool `notes_read` of `notes` has the name of another tool it grants; rename one"],
])("run start checks each node's module tools: %s", async (node, message) => {
  writeFileSync(join(dir, "crm.ts"), tools(["lookup"]));
  writeFileSync(join(dir, "quote.ts"), tools(["quote"], 'output: { type: "integer" }, '));
  const uses = { crm: "./crm.ts", quote: "./quote.ts:quote", notes_read: "./crm.ts:lookup" };
  const quote = new FunctionTool({ name: "quote", description: "d", parametersJsonSchema: { type: "object" }, execute: () => 1 });
  const host: Record<string, FunctionTool> = message.includes("a tool of this run") ? { quote } : {};
  const text = app(uses, [node], "input:\n  notes: !FileMemory\n");
  await expect(run(text, { tools: host }, { notes: new FakeMemory() })).rejects.toThrow(new RangeError(message));
});
