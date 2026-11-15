/**
 * Tool modules at a segment's start: each segment (a run's start, each `advance`) starts one
 * thread per module its nodes reach, beside the MCP servers, and ends them when it ends. A module
 * that fails to start refuses the run before any record, naming its file (relative to the root
 * flow's folder) and export. Each node's scope holds its modules' tools for the run-start checks
 * and its model's request; calling one is refused for now.
 */

import { afterAll, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

import { loadFlow, withUnfinishedToolModules } from "../src/compose/loader.ts";
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

test("an agent's call of a module tool, and a tool node's, are refused until calls are supported", async () => {
  writeFileSync(join(dir, "crm.ts"), tools(["lookup"]));
  const refused = "crm.ts export 'lookup': calls to tool modules aren't supported yet";
  const model = new ScriptedModel([callReply(["crm_lookup", { id: "1" }, "c1"])]);
  const agent = await run(app({ crm: "./crm.ts" }, ["a: {kind: agent, prompt: go, tools: [crm]}"]), { models: serving(model) });
  expect(agent.failure?.message).toContain(refused);
  const node = "t: {kind: tool, tool_id: {crm: lookup}, input: {id: x}, retries: 2}";
  const called = await run(app({ crm: "./crm.ts" }, [node]));
  expect([called.failure?.code, called.failure?.message]).toEqual(["node.execute", expect.stringContaining(refused)]);
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
  for (let tries = 0; tries < 250 && !existsSync(join(dir, "slow.beat")); tries++) await sleep(20);
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
