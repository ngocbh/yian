/**
 * Tool modules at a segment's start: each segment (a run's start, each `advance`) starts one
 * thread per module its nodes reach, beside the MCP servers, and ends them when it ends. A module
 * that fails to start refuses the run before any record, naming its file (relative to the root
 * flow's folder) and export. Each node's scope holds its modules' tools for the run-start checks
 * and its model's request, and a call runs on the module's thread: its checks, its result and its
 * errors act as a host tool's do, for an AGENT's model and for TOOL retries, and an AGENT runs the
 * calls its `isConcurrencySafe` says are safe together. A thread that exits fails its calls in
 * flight with `tool.module_exited`, which an AGENT's model reads and TOOL retries stop at, and a
 * fresh thread serves the next call, or fails it with `tool.module_restart`. `run_started` records
 * the tools agents' grants reach, which a resume checks again and the agent-call view rebuilds
 * requests with.
 */

import { afterAll, beforeEach, expect, spyOn, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

import { loadFlow, withUnfinishedToolModules } from "../src/compose/loader.ts";
import { ToolResultPart } from "../src/llm/messages.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { advance, agentCall, agentCalls, complete, pendingCalls, runFlow, type RunFlowOptions } from "../src/run.ts";
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
  ["checked", null],
  ["audited", "tool 'audited' result failed its output validator: total: too low"],
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
// Its validator's output is checked once and kept on the thread, so every attempt adds to its one list.
const checker = { "~standard": { version: 1, vendor: "t", validate: (value) => ({ value: { ...value, seen: [] } }), jsonSchema: { input: () => parameters } } };
export const checked = {
  name: "checked",
  description: "checked.",
  parameters: checker,
  execute({ seen }, ctx) {
    seen.push(ctx.attempt);
    if (ctx.attempt < 3) throw Object.assign(new Error("not yet"), { [TOOL_ERROR]: ["transient"], retryAfter: 0 });
    return { attempt: ctx.attempt, calls: seen.length };
  },
};
const audit = { "~standard": { version: 1, vendor: "t", validate: () => ({ issues: [{ message: "too low", path: ["total"] }] }), jsonSchema: { input: () => ({ type: "object" }) } } };
export const audited = { name: "audited", description: "audited.", parameters, output: audit, execute: () => ({ total: 1 }) };
`,
  );
  const tool = how === "checked" || how === "audited" ? how : "flaky";
  const node = `t: {kind: tool, tool_id: {crm: ${tool}}, input: {how: ${how}}, retries: 2}`;

  const result = await run(app({ crm: "./crm.ts" }, [node], "output: ${t.output}\n"));

  if (message === null) expect([result.status, result.output]).toEqual(["succeeded", { attempt: 3n, calls: 3n }]);
  else expect([result.failure?.code, result.failure?.message]).toEqual(["node.execute", expect.stringContaining(message)]);
  if (message !== null) expect(result.failure?.message).not.toContain("attempt 2");
});

test("an agent runs a reply's module tool calls together when isConcurrencySafe says they are safe, else one at a time", async () => {
  writeFileSync(
    join(dir, "crm.ts"),
    `let running = 0;
export const meet = {
  name: "meet",
  description: "meet.",
  parameters: { type: "object", properties: { safe: { type: "boolean" } }, required: ["safe"] },
  isConcurrencySafe: ({ safe }) => safe,
  async execute({ safe }) {
    running += 1;
    // Waits for a second call to start beside it: a safe one long enough for one to, under load.
    for (let waited = 0; running < 2 && waited < (safe ? 5000 : 100); waited += 5) await new Promise((resolve) => setTimeout(resolve, 5));
    const met = running;
    await new Promise((resolve) => setTimeout(resolve, 20));
    running -= 1;
    return met;
  },
};
`,
  );
  const text = app({ meet: "./crm.ts:meet" }, ["a: {kind: agent, prompt: go, tools: [meet]}"]);
  const met = async (safe: boolean) => {
    const model = new ScriptedModel([callReply(["meet", { safe }, "c1"], ["meet", { safe }, "c2"]), textReply("done")]);
    expect((await run(text, { models: serving(model) })).status).toBe("succeeded");
    return resultContents(model);
  };

  expect(await met(true)).toEqual(["2", "2"]);
  expect(await met(false)).toEqual(["1", "1"]);
});

/** The source of a tool `name` taking any arguments, whose `execute` is `body`, after `more` fields. */
const plain = (name: string, body: string, more = "") =>
  `export const ${name} = { name: "${name}", description: "${name}.", parameters: { type: "object" }, ${more}execute: ${body} };\n`;

test.each([
  ["a tool node", "t: {kind: tool, tool_id: {crm: hold}, input: {}, retries: 2}", "${t.output}"],
  ["an agent", "a: {kind: agent, prompt: go, tools: [crm]}", "${a.output}"],
])("a module tool's pending() makes %s wait for complete() when the host completes calls, else fails its call", async (_name, node, output) => {
  const calls = join(dir, "calls");
  const hold = `() => { appendFileSync(${JSON.stringify(calls)}, "."); return { [Symbol.for("yian.pending")]: true }; }`;
  writeFileSync(join(dir, "crm.ts"), `import { appendFileSync } from "node:fs";\n${plain("hold", hold)}`);
  const text = app({ crm: "./crm.ts" }, [node], `output: ${output}\n`);
  const model = () => new ScriptedModel([callReply(["crm_hold", {}, "c1"]), textReply("done")]);
  const log = join(dir, "log.jsonl");

  const waiting = await run(text, { log, models: serving(model()), completesCalls: true });
  expect(waiting.status).toBe("waiting");
  const [call] = pendingCalls(log);
  await complete(log, call!.callId, { output: "held" });
  const advanced = await withUnfinishedToolModules(() =>
    advance({ log, models: serving(new ScriptedModel([textReply("done")])), env: hostEnv(dir) }),
  );
  expect([advanced.status, advanced.output]).toEqual(["succeeded", node.startsWith("t:") ? "held" : "done"]);

  const failed = await run(text, { models: serving(model()) });
  const message = "tool 'crm_hold' returned pending(), but this run's host doesn't complete calls";
  expect([failed.failure?.code, failed.failure?.message]).toEqual(["node.execute", expect.stringContaining(message)]);
  // One call each run: the refused one isn't tried again.
  expect(readFileSync(calls, "utf8")).toBe("..");
  await expect(run(text, { completesCalls: "yes" as never })).rejects.toThrow(new TypeError("completesCalls must be a boolean"));
});

test("what a module tool prints goes to onConsole under the calling node, and what its import prints to this process's console", async () => {
  writeFileSync(join(dir, "crm.ts"), 'console.log("importing");\n' + plain("talk", '(_args, ctx) => { console.info("serving", ctx.node); return 1; }'));
  const text = app({ crm: "./crm.ts" }, ["t: {kind: tool, tool_id: {crm: talk}, input: {}}", "a: {kind: agent, input: {go: '${t.output}'}, prompt: '${go}', tools: [crm]}"]);
  const model = new ScriptedModel([callReply(["crm_talk", {}, "c1"]), textReply("done")]);
  const lines: unknown[] = [];
  const log = spyOn(console, "log").mockImplementation(() => {});
  try {
    const done = await run(text, { runId: "r1", models: serving(model), onConsole: (line) => void lines.push(line) });

    expect(done.status).toBe("succeeded");
    expect(lines).toEqual([
      { runId: "r1", node: "t", method: "info", text: "serving t" },
      { runId: "r1", node: "a", method: "info", text: "serving a" },
    ]);
    expect(log.mock.calls).toEqual([["importing"]]);
  } finally {
    log.mockRestore();
  }
});

test("a thread exit fails an agent's calls in flight as errors its model reads, and its next call runs on a fresh thread", async () => {
  const safe = "isConcurrencySafe: () => true, ";
  writeFileSync(
    join(dir, "crm.ts"),
    plain("wait", '(_args, ctx) => new Promise((resolve) => ctx.signal.addEventListener("abort", resolve))', safe) +
      plain("exit", "() => new Promise(() => setTimeout(() => process.exit(3), 50))", safe) +
      plain("ok", '() => "ok"'),
  );
  const text = app({ crm: "./crm.ts" }, ["a: {kind: agent, prompt: go, tools: [crm]}"], "output: ${a.output}\n");
  const model = new ScriptedModel([
    callReply(["crm_wait", {}, "c1"], ["crm_exit", {}, "c2"]),
    callReply(["crm_ok", {}, "c3"]),
    textReply("done"),
  ]);

  const done = await run(text, { models: serving(model) });

  expect([done.status, done.output]).toEqual(["succeeded", "done"]);
  const exited = "crm.ts's thread exited (code 3); the call's outcome is unknown";
  expect(resultContents(model)).toEqual([exited, exited, "ok"]);
});

test.each([
  ["its output validator", "audited", "tool 'audited' result: the output validator threw: crm.ts's thread exited (code 4)"],
  ["a retry after a transient error", "flaky", "crm.ts's thread exited (code 5); the call's outcome is unknown"],
])("a thread exit in %s fails a tool node with tool.module_exited, and no attempt follows", async (_name, tool, message) => {
  writeFileSync(
    join(dir, "crm.ts"),
    `const TOOL_ERROR = Symbol.for("yian.tool_error");
const exits = { version: 1, vendor: "t", validate: () => process.exit(4), jsonSchema: { input: () => ({ type: "object" }) } };
const once = () => Object.assign(new Error("not yet"), { [TOOL_ERROR]: ["transient"], retryAfter: 0 });
${plain("audited", "() => ({ total: 1 })", 'output: { "~standard": exits }, ')}${plain("flaky", "(_args, ctx) => { if (ctx.attempt === 1) throw once(); process.exit(5); }")}`,
  );
  const node = `t: {kind: tool, tool_id: {crm: ${tool}}, input: {}, retries: 2}`;

  const failed = await run(app({ crm: "./crm.ts" }, [node], "output: ${t.output}\n"));

  expect([failed.failure?.code, failed.failure?.message]).toEqual(["tool.module_exited", expect.stringContaining(message)]);
  expect(failed.failure?.message).not.toContain("attempt 2 ");
});

test("a fresh thread that lists its module otherwise fails the agent with tool.module_restart", async () => {
  const path = join(dir, "crm.ts");
  // Its `exit` writes the module again, `ok` described otherwise, before it exits.
  const exit = `() => { copyFileSync(${JSON.stringify(join(dir, "changed.txt"))}, ${JSON.stringify(path)}); process.exit(6); }`;
  const source = `import { copyFileSync } from "node:fs";\n${plain("exit", exit)}${plain("ok", '() => "ok"')}`;
  writeFileSync(join(dir, "changed.txt"), source.replace('"ok."', '"Changed."'));
  writeFileSync(path, source);
  const text = app({ crm: "./crm.ts" }, ["a: {kind: agent, prompt: go, tools: [crm]}"]);
  const model = new ScriptedModel([callReply(["crm_exit", {}, "c1"]), callReply(["crm_ok", {}, "c2"])]);

  const failed = await run(text, { models: serving(model) });

  const why = "a fresh thread can't take over: export 'ok' description changed since the module was first imported";
  expect([failed.failure?.code, failed.failure?.message]).toEqual([
    "tool.module_restart",
    expect.stringContaining(`crm.ts's thread exited (code 6); ${why}`),
  ]);
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

/** The payload of the `run_started` record of the log at `log`. */
function started(log: string): Record<string, unknown> {
  return JSON.parse(readFileSync(log, "utf8").split("\n")[0]!).payload;
}

/** A tool of `tools(...)` as `run_started` records it. */
const recorded = (name: string) => ({
  name,
  description: `${name}.`,
  parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] },
  resumable: false,
});

test("run_started records, by module key, the tools agent grants reach, and no tool node's module", async () => {
  mkdirSync(join(dir, "tools"));
  writeFileSync(join(dir, "tools", "crm.ts"), tools(["lookup", "create"]));
  writeFileSync(join(dir, "quote.ts"), tools(["quote", "rate"]));
  writeFileSync(join(dir, "solo.ts"), tools(["solo"]));
  const call = "t: {kind: tool, tool_id: {solo: solo}, input: {id: x}}";
  const uses = { crm: "./tools/crm.ts", price: "./quote.ts:quote", solo: "./solo.ts" };
  const text = app(uses, ["a: {kind: agent, prompt: go, tools: [{crm: [lookup]}, price]}", call]);
  const log = join(dir, "log.jsonl");

  const done = await run(text, { log, models: serving(new ScriptedModel([textReply("done")])) });

  expect(done.status).toBe("succeeded");
  // A group lists every tool, in export-name order, so a resume sees one gained or lost.
  expect(started(log)["tool_modules"]).toEqual({
    "tools/crm.ts": [recorded("create"), recorded("lookup")],
    "quote.ts": [recorded("quote")],
  });
  const only = join(dir, "only.jsonl");
  expect((await run(app({ solo: "./solo.ts" }, [call]), { log: only })).status).toBe("succeeded");
  expect(Object.hasOwn(started(only), "tool_modules")).toBe(false);
});

/** A module whose `lookup` takes any arguments, beside `others` as `tools` writes them. */
const looseLookup = (others: string[]) =>
  `${tools(others)}export const lookup = { name: "lookup", description: "lookup.", parameters: { type: "object" }, execute: () => 1 };\n`;

test.each([
  ["description", tools(["create", "lookup"]).replace('"lookup."', '"Find."'), "lookup description differs from the recorded one"],
  ["parameters", looseLookup(["create"]), "lookup parameters differs from the recorded one"],
  ["gained tool", tools(["create", "extra", "lookup"]), "extra isn't recorded; the run started without it"],
  ["lost tool", tools(["lookup"]), "create is recorded, but this run's tool_modules lack it"],
])("a resume imports each module again and refuses a group whose tools differ from the recorded ones: %s", async (_name, source, message) => {
  writeFileSync(join(dir, "crm.ts"), tools(["create", "lookup"]));
  const agent = "a: {kind: agent, input: {go: '${h.output}'}, prompt: '${go}', tools: [crm]}";
  const text = app({ crm: "./crm.ts" }, ["h: {kind: tool, tool_id: hold}", agent]);
  const log = join(dir, "log.jsonl");
  expect((await run(text, { log, runId: "r1", tools: { hold } })).status).toBe("waiting");
  await complete(log, "r1/h", { output: "go" });
  writeFileSync(join(dir, "crm.ts"), source);
  const before = readFileSync(log, "utf8");

  const refused = withUnfinishedToolModules(() => advance({ log, env: hostEnv(dir), tools: { hold } }));

  await expect(refused).rejects.toThrow(new RangeError(`resume refused: tool_modules.crm.ts.${message}`));
  expect(readFileSync(log, "utf8")).toBe(before);
});

test("the agent-call view rebuilds the request of an agent granting module tools from the recorded ones", async () => {
  writeFileSync(join(dir, "crm.ts"), tools(["create", "lookup"]));
  writeFileSync(join(dir, "quote.ts"), tools(["quote"]));
  const uses = { crm: "./crm.ts", find: "./crm.ts:lookup", price: "./quote.ts:quote" };
  const text = app(uses, ["a: {kind: agent, prompt: go, tools: [find, crm, price]}"]);
  const log = join(dir, "log.jsonl");
  const model = new ScriptedModel([callReply(["crm_create", { id: "1" }, "c1"]), textReply("done")]);
  expect((await run(text, { log, models: serving(model) })).status).toBe("succeeded");
  // The view reads the recorded specs, never the module.
  writeFileSync(join(dir, "crm.ts"), tools(["create", "lookup"]).replace('"lookup."', '"Find."'));

  const view = (call: { nodeId: string; n: number }) => agentCall(log, call, { env: hostEnv(dir) });
  const views = await withUnfinishedToolModules(() => Promise.all(agentCalls(log).map(view)));

  const specs = (tools: readonly { name: string; description: string }[] | undefined) =>
    tools?.map(({ name, description }) => [name, description]);
  expect(views.map((each) => [each.digestMismatch, specs(each.request?.tools)])).toEqual(
    model.requests.map((request) => [false, specs(request.tools)]),
  );
  expect(views[0]!.request!.tools.map(({ name }) => name)).toEqual(["find", "crm_create", "price"]);
});
