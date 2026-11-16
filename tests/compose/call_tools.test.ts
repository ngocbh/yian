/**
 * `call:` names a tool on a `call` node, in a compact flow, in an inline `call(...)` and in a case
 * branch's `call(...)`: a name the file defines means the file's own flow or module, any other name
 * is a tool the run supplies, and a call naming a tool loads as the TOOL node `kind: tool` builds,
 * under its field rules, in the author's words.
 */

import { afterAll, beforeAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Binding, type CallableHint, outputContractType } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { runFlow, runStartLabel, type RunStartTool, runStartTools, staticNodes } from "../../src/run.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";

import { loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

let tmp: string;

// Its top level throws, so a load that imported it would fail.
const MODULE = 'throw new Error("imported");\nexport const lookup = {};\nexport const quote = {};\n';

beforeAll(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-call-tools-")));
  for (const file of ["crm.ts", "tools.ts"]) writeFileSync(join(tmp, file), MODULE);
  const quote = '  quote: {kind: code, code: "return 1", output: !int}\n';
  writeFileSync(join(tmp, "billing.yaml"), `id: billing\nname: billing\ndefs:\n${quote}`);
  writeFileSync(join(tmp, "clean.yaml"), "id: clean\nname: clean\nnodes:\n  t: !str hi\noutput: ${t.output}\n");
});

afterAll(() => rmSync(tmp, { recursive: true, force: true }));

/**
 * A flow with the memory input `lessons` and `inputs` (lines under `input:`), the module aliases
 * `crm` (a group) and `one` (one export), the library `billing`, the flow alias `clean`, the def
 * `greet`, and `nodes` (lines under `nodes:`).
 */
function app(nodes: string, inputs = ""): string {
  return `id: app
name: app
input:
  lessons: !FileMemory
${inputs}uses:
  crm: ./crm.ts
  one: ./tools.ts:quote
  billing: ./billing
  clean: ./clean
defs:
  greet: {kind: code, code: "return 1", output: !int}
nodes:
${nodes}`;
}

/** Load `text` from `tmp`. */
const load = (text: string) => loadFlow(text, { searchPaths: [tmp] });

/**
 * What the node at path `nodePath` of the flow loaded from `text` runs: `tool <id>`, with
 * ` module <alias>` or ` memory` when it names one, or `flow <id>`, with `.<def>` for a library's def.
 */
async function target(text: string, nodePath = "n"): Promise<string> {
  const node = staticNodes(await load(text)).find(([path]) => path === nodePath)![1];
  if (node instanceof ToolNode) {
    const where = node.module !== null ? ` module ${node.module.alias}` : node.memoryTool ? " memory" : "";
    return `tool ${node.toolId}${where}`;
  }
  const call = node as CallNode;
  return `flow ${call.flowId}${call.def === null ? "" : `.${call.def}`}`;
}

/** The message loading `text` fails with. */
const refusal = async (text: string) => (await loadErrorAsync(load(text))).message;

// Each row: the `call:` value and its node's other fields, any extra input lines, and what the
// node runs (`tool ...`, `flow ...`) or the load error.
test.each<[string, string, string]>([
  // A bare name.
  ["sleep, input: {for: 1s}", "", "tool sleep"],
  ["ask_user, input: {prompt: hi}", "", "tool ask"],
  ["load_skill", "", "node 'n' (kind=call): tool id `load_skill` is reserved by yian"],
  ["greet", "", "flow greet"],
  ["clean", "", "flow clean"],
  ["billing", "", "call node 'n': callable 'billing' is a library; name a def: `call: {billing: quote}`"],
  ["one", "", "tool one module one"],
  ["crm", "", "node 'n': `crm` is a tool module; name one of its tools: `call: {crm: <tool>}`"],
  ["lessons", "", "node 'n': `lessons` is a memory input; name one of its tools: `call: {lessons: <tool>}`"],
  ["search", "", "tool search"],
  // A group form.
  [
    "{crm: take}",
    "",
    "node 'n' (kind=call): `call: {crm: take}`: take is a built-in, not a group's tool; write `call: take`",
  ],
  ["{crm: lookup}", "  crm: !FileMemory\n", "node 'n': `crm` is both a memory input and a tool module; rename one"],
  ["{greet: x}", "  greet: !FileMemory\n", "node 'n': `greet` is both a memory input and a flow; rename one"],
  [
    "{billing: quote}",
    "  billing: !FileMemory\n",
    "node 'n': `billing` is both a memory input and a library; rename one",
  ],
  ["{lessons: read}, input: {path: a.md}", "", "tool lessons_read memory"],
  ["{billing: quote}", "", "flow billing.quote"],
  ["{crm: lookup}", "", "tool crm_lookup module crm"],
  [
    "{greet: x}",
    "",
    "node 'n': `greet` is a flow, not a library; a tool group named `greet` is shadowed in this file, so rename " +
      "the def to reach it",
  ],
  [
    "{clean: x}",
    "",
    "node 'n': `clean` is a flow, not a library; a tool group named `clean` is shadowed in this file, so rename " +
      "the alias to reach it",
  ],
  ["{one: quote}", "", "node 'n': `one` is one tool, ./tools.ts:quote; write `call: one`"],
  ["{github: search}", "", "tool github_search"],
])("call: %s (inputs %j) is %s", async (call, inputs, expected) => {
  const text = app(`  n: {kind: call, call: ${call}}\n`, inputs);
  const got = expected.startsWith("tool ") || expected.startsWith("flow ") ? await target(text) : await refusal(text);
  expect(got).toBe(expected);
});

test("a call node's id may equal the tool it names", async () => {
  const node = (await load(app("  place: {kind: call, call: place}\n"))).compiled.nodes.get("place") as ToolNode;
  expect(node.toolId).toBe("place");
});

test.each([
  ["retries: 2", "`retries:` doesn't apply to a flow call: a flow retries inside its own nodes"],
  ["output: !int", "`output:` doesn't apply to a flow call: a flow's result type comes from the flow"],
  ["adaptive_questions: {prompt: hi}", "`adaptive_questions:` applies only to `call: ask`"],
])("a flow call refuses %s", async (field, problem) => {
  expect(await refusal(app(`  n: {kind: call, call: greet, ${field}}\n`))).toBe(`node 'n' (kind=call): ${problem}`);
});

test("a tool call takes retries:, output: and asserts:", async () => {
  const text = app("  n: {kind: call, call: search, retries: 2, output: !int, asserts: [output > 0]}\n");
  const node = (await load(text)).compiled.nodes.get("n") as ToolNode;

  expect(node.retries).toBe(2);
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe("int");
  expect(node.postAsserts.map((assertion) => assertion.display)).toEqual(["output > 0"]);
});

// A `call` node naming a tool reads `call:` and `kind=call` where `kind: tool` reads `tool_id:` and
// `kind=tool`.
test.each([
  [
    "sleep, retries: 1, input: {for: 1s}",
    "node 'n' (call: sleep): `retries:` doesn't apply to sleep: there is no tool call to retry",
  ],
  [
    "search, adaptive_questions: {prompt: hi}",
    "node 'n' (call: search): `adaptive_questions:` applies only to call: ask",
  ],
  [
    "take, output: !int",
    "node 'n' (call: take): take's output is inferred ({events: list(T), closed: bool}); remove `output:`",
  ],
  [
    "ask, retries: 1, input: {prompt: hi}",
    "node 'n' (call: ask): `retries:` doesn't apply to ask: the host owns retries of a call it holds",
  ],
  ['""', "node 'n' (kind=call): `call:` must be a tool name or `{group: tool}`, but it is empty; write a tool name"],
  [
    "{notes: read}",
    "node 'n': `call: {notes: read}` names `notes`, an Optional memory; a tool call needs one that is always bound",
  ],
  ["{crm: [search]}", "node 'n' (kind=call): `call: {crm: ...}` names one tool; write `{crm: search}`"],
  ["{github: [search]}", "node 'n' (kind=call): `call: {github: ...}` names one tool; write `{github: search}`"],
  [
    "search, over: x",
    "node 'n' (kind=call): field 'over' is not allowed (allowed: adaptive_questions, asserts, call, depends_on, env, " +
      "input, kind, node_name, output, retries, runs_after)",
  ],
  [
    "search, node_name: 5",
    "node 'n' (kind=call): `node_name:` must be text, but YAML read it as an integer; put it in quotes",
  ],
  [
    "search, input: x",
    'node \'n\' (kind=call): `input:` must be a mapping of parameter names to bindings (got "x"); bind each ' +
      "parameter, such as `<field>: ...`",
  ],
])("call: %s reads in the author's words", async (call, message) => {
  expect(await refusal(app(`  n: {kind: call, call: ${call}}\n`, "  notes: !Optional(FileMemory)\n"))).toBe(message);
});

test.each([
  ["call", "call"],
  ["tool", "tool_id"],
])("adaptive_questions: on a group's tool is refused under kind: %s", async (kind, field) => {
  const text = app(`  n: {kind: ${kind}, ${field}: {github: search}, adaptive_questions: {prompt: hi}}\n`);
  expect(await refusal(text)).toBe(
    `node 'n' (${field}: {github: search}): \`adaptive_questions:\` applies only to ${field}: ask`,
  );
});

test("an empty tool_id: is a load error", async () => {
  expect(await refusal(app('  n: {kind: tool, tool_id: ""}\n'))).toBe(
    "node 'n' (kind=tool): `tool_id:` must be a tool name or `{group: tool}`, but it is empty; write a tool name",
  );
});

test("a call node naming a tool also written with tool_id: is refused at tool_id:", async () => {
  expect(await refusal(app("  n: {kind: call, call: search, tool_id: search}\n"))).toContain(
    "node 'n' (kind=call): field 'tool_id' is not allowed",
  );
});

// Each row: the tool as written, the node's other lines, and any extra input lines.
test.each<[string, string, string, string]>([
  ["sleep", "sleep", "    input: {for: 1s}\n", ""],
  ["take", "take", '    input: {from: "${input.inbox}"}\n', "  inbox: !Inbox(str)\n"],
  ["ask with adaptive questions", "ask", "    adaptive_questions: {prompt: Ask about the trip}\n", ""],
  ["a module's one export", "one", "", ""],
  ["a module group's tool", "{crm: lookup}", "", ""],
  ["a memory's tool", "{lessons: read}", "    input: {path: a.md}\n", ""],
  ["a host tool", "search", "    retries: 2\n    output: !int\n    asserts: [output > 0]\n", ""],
  ["a host group's tool", "{github: search}", "", ""],
])("kind: call naming %s has kind: tool's fingerprint", async (_label, written, rest, inputs) => {
  const fingerprint = async (head: string) =>
    flowFingerprint((await load(app(`  n:\n${head}${rest}`, inputs))).compiled);

  expect(await fingerprint(`    kind: call\n    call: ${written}\n`)).toBe(
    await fingerprint(`    kind: tool\n    tool_id: ${written}\n`),
  );
});

/** A host tool named `name` that returns `host`. */
const hostTool = (name: string) =>
  tool({ name, description: "A host tool.", parameters: { type: "object", properties: {} }, execute: () => "host" });

test("a call node runs a host tool and a host group's tool", async () => {
  const echo = (name: string, said: string) =>
    tool({
      name,
      description: "Echo the query.",
      parameters: { type: "object", properties: { q: { type: "string" } }, required: ["q"] },
      execute: (args: Record<string, unknown>) => `${said} ${String(args["q"])}`,
    });
  const text =
    "id: r\nname: r\ninput:\n  q: !str\nnodes:\n" +
    '  bare: {kind: call, call: lookup, input: {q: "${input.q}"}, output: !str}\n' +
    '  grouped: {kind: call, call: {github: search}, input: {q: "${input.q}"}, output: !str}\n' +
    'output: {bare: "${bare.output}", grouped: "${grouped.output}"}\n';
  const tools = { lookup: echo("lookup", "found"), github: [echo("search", "searched")] };
  const result = await runFlow(await loadFlow(text), { q: "hi" }, { env: hostEnv(tmp), tools });

  expect(result.output).toEqual({ bare: "found hi", grouped: "searched hi" });
});

test("a def named like a host tool runs the def", async () => {
  const text =
    'id: r\nname: r\ndefs:\n  greet: {kind: code, code: "return 1", output: !int}\n' +
    "nodes:\n  n: {kind: call, call: greet}\noutput: ${n.output}\n";
  const result = await runFlow(await loadFlow(text), {}, { env: hostEnv(tmp), tools: { greet: hostTool("greet") } });

  expect(result.output).toBe(1n);
});

test("a mistyped call: name fails at run start, before any record", async () => {
  const log = join(tmp, "typo.jsonl");
  const flow = await loadFlow("id: r\nname: r\nnodes:\n  n: {kind: call, call: serch}\n");
  const error = (await runFlow(flow, {}, { env: hostEnv(tmp), log, tools: { search: hostTool("search") } }).catch(
    (caught: unknown) => caught,
  )) as Error;

  expect(error.message).toBe(
    "TOOL node 'n' names tool 'serch', which isn't in this run's tools; known tools: [\"search\"]; " +
      "did you mean 'search'?",
  );
  expect(existsSync(log)).toBe(false);
});

/** A library whose def `price` calls `quot`, a typo of its def `quote`. */
const TYPO_LIBRARY =
  'id: lib\nname: lib\ndefs:\n  quote: {kind: code, code: "return 1", output: !int}\n' +
  "  price:\n    nodes:\n      n: {kind: call, call: quot}\n    output: ${n.output}\n";

/** A CODE node whose input `x` is `call`, as one line. */
const inlineNode = (call: string) => `{kind: code, code: return 1, output: !int, input: {x: ${call}}}`;

/** A hint: the name the file defines that is closest to a tool, and what it is. */
const hint = (what: CallableHint["what"], name: string): RunStartTool["hint"] => ({ what, name });

// Each row: node `n` as written, and the hint run start and `yian check` give for its tool.
test.each<[string, RunStartTool["hint"]]>([
  ["{kind: call, call: gret}", hint("def", "greet")],
  ["{kind: call, call: clen}", hint("alias", "clean")],
  ["{kind: call, call: quot}", hint("library def", "{billing: quote}")],
  ["{kind: call, call: on}", hint("module tool", "one")],
  ["{kind: call, call: search}", null],
  // `tool_id:` says the name is a tool, so no def is offered.
  ["{kind: tool, tool_id: gret}", null],
  [inlineNode("call(gret)"), hint("def", "greet")],
  [inlineNode("call(quot)"), hint("library def", "billing.quote")],
  // A mistyped group is matched against the file's libraries, tool modules and memory inputs.
  ["{kind: call, call: {bilin: quote}}", hint("library def", "{billing: quote}")],
  ["{kind: call, call: {bilin: qoute}}", hint("library def", "{billing: quote}")],
  ["{kind: call, call: {bilin: refund}}", hint("alias", "billing")],
  ["{kind: call, call: {crn: lookup}}", hint("module tool", "{crm: lookup}")],
  ["{kind: call, call: {lesons: read}}", hint("memory tool", "{lessons: read}")],
  ["{kind: map, over: [1], call: {lesons: read}}", hint("memory tool", "{lessons: read}")],
  ["{kind: map, over: [1], call: gret}", hint("def", "greet")],
  [inlineNode("call(lesons.read)"), hint("memory tool", "lessons.read")],
  ["{kind: call, call: {github: search}}", null],
  [inlineNode("call(bilin.quote)"), hint("library def", "billing.quote")],
  [inlineNode("call(crn.lookup)"), hint("module tool", "crm.lookup")],
])("n: %s is checked at run start, with hint %p", async (node, expected) => {
  const [tool] = runStartTools(await load(app(`  n: ${node}\n`)));
  expect(tool!.hint).toEqual(expected);
});

test("run start checks only the run's tools, each node named by its path or its call's site", async () => {
  const text = app(
    "  a: {kind: call, call: sleep, input: {for: 1s}}\n" +
      "  b: {kind: call, call: {crm: lookup}}\n" +
      "  c: {kind: call, call: one}\n" +
      "  d: {kind: call, call: {lessons: read}, input: {path: a.md}}\n" +
      "  e: {kind: call, call: ask, input: {prompt: hi}}\n" +
      "  f: {kind: call, call: {github: search}}\n" +
      "  g: {kind: code, code: return 1, output: !int, input: {x: call(serch)}}\n",
  );

  expect(runStartTools(await load(text))).toEqual([
    { node: "f", site: null, name: "{github: search}", hint: null },
    { node: "__call_0", site: "node 'g' input 'x' call(serch)", name: "serch", hint: null },
  ]);
});

test("a case branch's mistyped call(...) is checked at run start by its site, with the hint", async () => {
  expect(runStartTools(await load(caseApp("call(gret)", "call(greet)")))).toEqual([
    {
      node: "__call_0",
      site: "node 'g' cases[0].then call(gret)",
      name: "gret",
      hint: hint("def", "greet"),
    },
  ]);
});

test("a call(...) in a child flow is listed by its path, and its label says which flow", async () => {
  const text =
    "id: r\nname: r\ndefs:\n  inner:\n    nodes:\n      m: " +
    inlineNode("call(serch)") +
    "\n    output: ${m.output}\nnodes:\n  c: {kind: call, call: inner}\n";
  const site = "node 'm' input 'x' call(serch)";
  const tools = runStartTools(await loadFlow(text));
  expect(tools).toEqual([{ node: "c/__call_0", site, name: "serch", hint: null }]);
  expect(runStartLabel(tools[0]!)).toBe(`${site} in 'c'`);
});

test("a library's defs are checked at run start under their names, with hints from the library", async () => {
  const flow = await loadFlow(TYPO_LIBRARY);
  expect(runStartTools(flow)).toEqual([
    { node: "price/n", site: null, name: "quot", hint: hint("def", "quote") },
  ]);
});

test("the hint is diagnostic-only: the fingerprint is the same with and without it", async () => {
  const flow = (def: string) =>
    `id: r\nname: r\ndefs:\n  ${def}: {kind: code, code: "return 1", output: !int}\n` +
    "nodes:\n  n: {kind: call, call: gret}\n  m: {kind: code, code: return 1, output: !int, input: {x: call(gret)}}\n";
  const [hinted, plain] = [await loadFlow(flow("greet")), await loadFlow(flow("zzzzz"))];

  expect(runStartTools(hinted).map((each) => each.hint)).toEqual([hint("def", "greet"), hint("def", "greet")]);
  expect(runStartTools(plain).map((each) => each.hint)).toEqual([null, null]);
  expect(flowFingerprint(hinted.compiled)).toBe(flowFingerprint(plain.compiled));
});

/**
 * The message the flow `text` fails to start with, given the host tool `search`, which needs `q`,
 * and a tool for each of `others`.
 */
async function startError(text: string, others: readonly string[] = []): Promise<string> {
  const search = tool({
    name: "search",
    description: "A host tool.",
    parameters: { type: "object", properties: { q: { type: "string" } }, required: ["q"], additionalProperties: false },
    execute: () => "found",
  });
  const tools = { search, ...Object.fromEntries(others.map((name) => [name, hostTool(name)])) };
  const run = runFlow(await load(text), {}, { env: hostEnv(tmp), tools });
  return ((await run.catch((caught: unknown) => caught)) as Error).message;
}

const UNKNOWN = "names tool 'serch', which isn't in this run's tools; known tools: [\"search\"]";

// Each row: the def, the `call:` that misspells it, the run's other tools, and what run start offers.
test.each<[string, string, string[], string]>([
  ["greet", "gret", [], "the def `greet`"],
  // As close as the run's `search`: the file's name comes first.
  ["serch2", "serch", [], "the def `serch2` or the tool `search`"],
  ["fetch_all", "fetch_ur", ["fetch_url"], "the tool `fetch_url` or the def `fetch_all`"],
])("run start offers the def %s for call: %s, and a run's tool as close or closer", async (def, call, others, offer) => {
  const text =
    `id: r\nname: r\ndefs:\n  ${def}: {kind: code, code: "return 1", output: !int}\n` +
    `nodes:\n  n: {kind: call, call: ${call}}\n`;
  expect(await startError(text, others)).toEndWith(`; did you mean ${offer}?`);
});

/**
 * A flow whose file defines the def `greet`, the flow alias `clean`, the library `billing` and the
 * tool modules `crm` (a group) and `one` (one export), and whose node `n` is `node`.
 */
const typoApp = (node: string) =>
  "id: r\nname: r\nuses:\n  crm: ./crm.ts\n  one: ./tools.ts:quote\n  billing: ./billing\n  clean: ./clean\n" +
  `defs:\n  greet: {kind: code, code: "return 1", output: !int}\nnodes:\n  n: ${node}\n`;

const NO_TOOL = (tool: string) => `tool '${tool}', which isn't in this run's tools; known tools: ["search"]`;
/** How run start says `{<written>}`'s group, such as `bilin: quote`, isn't one of the run's. */
const NO_GROUP = (written: string) =>
  `{${written}}: unknown tool group '${written.split(":")[0]}'; tool groups in this run: []`;

// Each row: the `call:` of node `n`, why run start refuses it, and the name it offers.
test.each([
  ["clen", NO_TOOL("clen"), "the alias `clean`"],
  ["quot", NO_TOOL("quot"), "the library def `{billing: quote}`"],
  ["on", NO_TOOL("on"), "the module tool `one`"],
  ["{bilin: quote}", NO_GROUP("bilin: quote"), "the library def `{billing: quote}`"],
  ["{crn: lookup}", NO_GROUP("crn: lookup"), "the module tool `{crm: lookup}`"],
])("run start offers the file's closest name for call: %s", async (call, reason, hint) => {
  const error = await startError(typoApp(`{kind: call, call: ${call}}`));
  expect(error).toBe(`TOOL node 'n' names ${reason}; did you mean ${hint}?`);
});

test("run start spells an inline call's tool and the names it offers as the call writes them", async () => {
  expect(await startError(typoApp(inlineNode("call(crn.lookup)")))).toBe(
    "node 'n' input 'x' call(crn.lookup) names crn.lookup: unknown tool group 'crn'; tool groups in this run: []; " +
      "did you mean the module tool `crm.lookup`?",
  );
  const text = `id: r\nname: r\nnodes:\n  n: ${inlineNode("call(github_serch)")}\n`;
  const run = runFlow(await loadFlow(text), {}, { env: hostEnv(tmp), tools: { github: [hostTool("search")] } });
  expect(((await run.catch((caught: unknown) => caught)) as Error).message).toEndWith(
    "; did you mean `github.search`?",
  );
});

test("the check list spells an inline call's tool as the call writes it", async () => {
  const [tool] = runStartTools(await load(app(`  n: ${inlineNode("call(github.search)")}\n`)));
  expect(tool!.name).toBe("github.search");
});

// Each row: what fails, the flow's lines and the run-start error, which names the inline call's
// site.
test.each([
  [
    "an unknown tool",
    "nodes:\n  n: {kind: code, code: return 1, output: !int, input: {x: call(serch)}}\n",
    `node 'n' input 'x' call(serch) ${UNKNOWN}; did you mean 'search'?`,
  ],
  [
    "an unknown tool in a child flow",
    "defs:\n  inner:\n    nodes:\n      m: {kind: code, code: return 1, output: !int, input: {x: call(serch)}}\n" +
      "    output: ${m.output}\nnodes:\n  c: {kind: call, call: inner}\n",
    `node 'm' input 'x' call(serch) in 'c' ${UNKNOWN}; did you mean 'search'?`,
  ],
  [
    "an argument the tool doesn't take",
    "nodes:\n  n:\n    kind: code\n    code: return 1\n    output: !int\n    input:\n      x: call(search, k=1)\n",
    "node 'n' input 'x' call(search) argument 'k': tool 'search' takes no such input; it takes [\"q\"]",
  ],
  [
    "an argument the tool can't take",
    "nodes:\n  n:\n    kind: code\n    code: return 1\n    output: !int\n    input:\n      x: call(search, q=1)\n",
    "node 'n' input 'x' call(search) argument 'q': tool 'search' takes str, but the binding is 1 (int)",
  ],
  [
    "an argument the tool needs",
    "nodes:\n  n: {kind: code, code: return 1, output: !int, input: {x: call(search)}}\n",
    "node 'n' input 'x' call(search) is missing argument 'q' that tool 'search' requires",
  ],
])("run start names an inline tool call by its site: %s", async (_what, lines, message) => {
  expect(await startError(`id: r\nname: r\n${lines}`)).toBe(message);
});

test("a compact flow's call: names a tool", async () => {
  const flow = await loadFlow("id: s\nname: s\ninput:\n  q: !str\noutput: !int\nkind: call\ncall: search\n");
  const node = flow.compiled.nodes.get("s") as ToolNode;

  expect(node.toolId).toBe("search");
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe("int");
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["q"]);
});

test.each([
  ["call", "call"],
  ["tool", "tool_id"],
])("a kind: %s node binding the memory it calls is refused, as any tool argument holding one", async (kind, field) => {
  const text = app(`  n: {kind: ${kind}, ${field}: {lessons: read}, input: {lessons: "\${input.lessons}", path: a.md}}\n`);
  expect(await refusal(text)).toBe(
    "flow has unresolved references:\n  node 'n' input 'lessons' from: ${input.lessons} is a memory; use it whole as an " +
      "agent input, in prompt text, as a child's memory input of the same type, or name it in `tools:`/`tool_id:` " +
      "(a tool node can't take a memory)",
  );
});

test("a single-node def calling its memory input's tool doesn't take the memory as an argument", async () => {
  const text =
    "id: r\nname: r\ninput:\n  lessons: !FileMemory\n" +
    "defs:\n  rd: {input: {lessons: !FileMemory, path: !str}, kind: call, call: {lessons: read}}\n" +
    'nodes:\n  n: {kind: call, call: rd, input: {lessons: "${input.lessons}", path: a.md}}\n';
  expect(await target(text, "n/rd")).toBe("tool lessons_read memory");
});

test("a compact flow's call: of a memory input names one of its tools", async () => {
  const text = "id: s\nname: s\ninput:\n  lessons: !FileMemory\nkind: call\ncall: lessons\n";
  expect((await loadErrorAsync(loadFlow(text))).message).toBe(
    "node 's': `lessons` is a memory input; name one of its tools: `call: {lessons: <tool>}`",
  );
});

/** `app` whose node `n` is a CODE node with input `x` written as `value`. */
const inlineApp = (value: string, inputs = "") =>
  app(`  n:\n    kind: code\n    code: "return 1"\n    input:\n      x: ${value}\n    output: !int\n`, inputs);

// Each row: the inline call at node `n`'s input `x`, any extra input lines, and what its synth node
// runs or the load error, which names the call's site.
test.each<[string, string, string]>([
  ["call(sleep, for=1s)", "", "tool sleep"],
  ["call(take, from=${input.inbox})", "  inbox: !Inbox(str)\n", "tool take"],
  ["call(ask_user, prompt=hi)", "", "tool ask"],
  ["call(load_skill)", "", "node 'n' input 'x' call(load_skill): tool id `load_skill` is reserved by yian"],
  ["call(greet)", "", "flow greet"],
  ["call(billing.quote)", "", "flow billing.quote"],
  ["call(one, qty=3)", "", "tool one module one"],
  [
    "call(crm)",
    "",
    "node 'n' input 'x' call(crm): `crm` is a tool module; name one of its tools: `call(crm.<tool>, ...)`",
  ],
  [
    "call(lessons)",
    "",
    "node 'n' input 'x' call(lessons): `lessons` is a memory input; name one of its tools: " +
      "`call(lessons.<tool>, ...)`",
  ],
  ["call(search, q=hi)", "", "tool search"],
  [
    "call(crm.take)",
    "",
    "node 'n' input 'x' call(crm.take): `call(crm.take, ...)`: take is a built-in, not a group's tool; " +
      "write `call(take, ...)`",
  ],
  [
    "call(crm.lookup)",
    "  crm: !FileMemory\n",
    "node 'n' input 'x' call(crm.lookup): `crm` is both a memory input and a tool module; rename one",
  ],
  ["call(lessons.read, path=a.md)", "", "tool lessons_read memory"],
  ["call(crm.lookup)", "", "tool crm_lookup module crm"],
  [
    "call(greet.x)",
    "",
    "node 'n' input 'x' call(greet.x): `greet` is a flow, not a library; a tool group named `greet` is " +
      "shadowed in this file, so rename the def to reach it",
  ],
  [
    "call(one.quote)",
    "",
    "node 'n' input 'x' call(one.quote): `one` is one tool, ./tools.ts:quote; write `call(one, ...)`",
  ],
  ["call(github.search)", "", "tool github_search"],
  [
    "call(notes.read)",
    "  notes: !Optional(FileMemory)\n",
    "node 'n' input 'x' call(notes.read): `call(notes.read, ...)` names `notes`, an Optional memory; a tool " +
      "call needs one that is always bound",
  ],
  [
    "call(sleep, for=5)",
    "",
    "node 'n' input 'x' call(sleep) argument 'for': a duration is text such as 24h, got 5; write 5s",
  ],
])("inline %s (inputs %j) is %s", async (call, inputs, expected) => {
  const text = inlineApp(call, inputs);
  const got =
    expected.startsWith("tool ") || expected.startsWith("flow ") ? await target(text, "__call_0") : await refusal(text);
  expect(got).toBe(expected);
});

/** `app` whose node `g` is a case taking `then` on `input.go`, else `otherwise`. */
const caseApp = (then: string, otherwise: string) =>
  app(
    `  g:\n    kind: case\n    cases:\n      - when: input.go\n        then: ${then}\n    else: ${otherwise}\n`,
    "  go: !bool\n",
  );

test("a case branch's call(...) of a tool is a TOOL node", async () => {
  const text = caseApp("call(sleep, for=1s)", "call(github.search, q=hi)");

  expect(await target(text, "__call_0")).toBe("tool sleep");
  expect(await target(text, "__call_1")).toBe("tool github_search");
});

test.each([
  [
    "call(sleep, for=5)",
    "call(greet)",
    "node 'g' cases[0].then call(sleep) argument 'for': a duration is text such as 24h, got 5; write 5s",
  ],
  [
    "call(greet)",
    "call(lessons)",
    "node 'g' else call(lessons): `lessons` is a memory input; name one of its tools: `call(lessons.<tool>, ...)`",
  ],
])("a case branch's tool call refuses then: %s else: %s at its site", async (then, otherwise, message) => {
  expect(await refusal(caseApp(then, otherwise))).toBe(message);
});

test("an inline tool call is located at the binding it was written in", async () => {
  const compiled = (await load(inlineApp("call(search, q=hi)"))).compiled;
  const binding = compiled.wiring.get("n")!.get("x") as Binding;

  expect(binding.origin).toBeInstanceOf(OriginRef);
  expect(compiled.nodes.get("__call_0")!.origin).toBe(binding.origin);
});


test.each([
  ["call(ask)", "node 'n' input 'x' call(ask): ask needs argument prompt or questions"],
  [
    "call(ask, prompt=hi, __questions=x)",
    "node 'n' input 'x' call(ask): argument '__questions' is reserved for adaptive_questions",
  ],
])("inline %s names its arguments", async (call, message) => {
  expect(await refusal(inlineApp(call))).toBe(message);
});

/**
 * A host tool named `name`, taking the arguments in `properties`, that records each call's arguments
 * in `calls` and returns `result`.
 */
function recording(name: string, properties: Record<string, { type: string }>, result: unknown, calls: unknown[]) {
  return tool({
    name,
    description: "A host tool.",
    parameters: { type: "object", properties },
    execute: (args: unknown) => {
      calls.push(args);
      return result;
    },
  });
}

test("an inline tool call runs in a map's over: and in a flow assert", async () => {
  const text =
    "id: r\nname: r\n" +
    'defs:\n  double: {input: {x: !int}, kind: code, code: "return inputs.x * 2", output: !int}\n' +
    'nodes:\n  each:\n    kind: map\n    call: double\n    over: call(numbers)\n    input: {x: "${item}"}\n' +
    "asserts:\n  - call(check, n=1)\n" +
    "output: ${each.output}\n";
  const checked: unknown[] = [];
  const tools = {
    numbers: recording("numbers", {}, [1, 2], []),
    check: recording("check", { n: { type: "integer" } }, true, checked),
  };
  const result = await runFlow(await loadFlow(text), {}, { env: hostEnv(tmp), tools });

  expect(result.output).toEqual([2n, 4n]);
  expect(checked).toEqual([{ n: 1 }]);
});

test("an inline tool call on a branch not taken never runs", async () => {
  const text =
    "id: r\nname: r\ninput:\n  go: !bool\nnodes:\n" +
    "  g:\n    kind: case\n    cases:\n      - when: input.go\n        then: n\n    else: m\n" +
    '  n:\n    kind: code\n    input:\n      x: call(lookup, q=hi)\n    code: "return 1"\n    output: !int\n' +
    "  m: !int 2\n" +
    "output: ${g.output}\n";
  const calls: unknown[] = [];
  const result = await runFlow(await loadFlow(text), { go: false }, {
    env: hostEnv(tmp),
    tools: { lookup: recording("lookup", { q: { type: "string" } }, "found", calls) },
  });

  expect(result.output).toBe(2n);
  expect(calls).toEqual([]);
});

// Each row: the `call:` value on a map, and what its generated body's node runs or the load error.
test.each([
  ["gret", "tool gret"],
  ["sleep, input: {until: 2020-01-01}", "tool sleep"],
  ["one", "tool one module one"],
  ["{github: search}", "tool github_search"],
  ["{lessons: read}", "tool lessons_read memory"],
  ["{crm: lookup}", "tool crm_lookup module crm"],
  ["greet", "flow greet"],
  [
    "{crm: take}",
    "map node 'n': `call: {crm: take}` reads an inbox, and a map can't give each item its own (per-instance " +
      "inboxes aren't supported yet); take before the map",
  ],
  [
    "{crm: sleep}",
    "node 'n' (kind=map): `call: {crm: sleep}`: sleep is a built-in, not a group's tool; write `call: sleep`",
  ],
  ["load_skill", "map node 'n': tool id `load_skill` is reserved by yian"],
  ["lessons", "map node 'n': `lessons` is a memory input; name one of its tools: `call: {lessons: <tool>}`"],
  [
    "{greet: x}",
    "map node 'n': `greet` is a flow, not a library; a tool group named `greet` is shadowed in this file, so " +
      "rename the def to reach it",
  ],
])("a map naming %s runs %s", async (call, expected) => {
  const text = app(`  n: {kind: map, over: [1], call: ${call}}\n`);
  if (expected.startsWith("flow ")) expect(await target(text)).toBe(expected);
  else if (expected.startsWith("tool ")) expect(await target(text, "n/n")).toBe(expected);
  else expect(await refusal(text)).toBe(expected);
});

const RUNS_ONLY_FLOWS = "and a `loop` runs only flows for now; call it from a `call` node, or wrap it in a def";

// Each row: the `call:` value on a loop and the refusal.
test.each([
  ["gret", `\`gret\` names no def or alias in this file, so it is a tool, ${RUNS_ONLY_FLOWS}; did you mean \`greet\`?`],
  ["search", `\`search\` names no def or alias in this file, so it is a tool, ${RUNS_ONLY_FLOWS}`],
  ["sleep", `\`sleep\` is a built-in tool, ${RUNS_ONLY_FLOWS}`],
  ["load_skill", "tool id `load_skill` is reserved by yian"],
  ["one", `\`one\` is a tool module, ${RUNS_ONLY_FLOWS}`],
  ["lessons", "`lessons` is a memory input, not a flow; call one of its tools from a `call` node"],
  ["{crm: take}", `\`take\` is a built-in tool, ${RUNS_ONLY_FLOWS}`],
  [
    "{github: search}",
    `\`github\` names no def or alias in this file, so \`{github: search}\` is a tool, ${RUNS_ONLY_FLOWS}`,
  ],
  ["{lessons: read}", `\`lessons\` is a memory input, so \`{lessons: read}\` is a tool, ${RUNS_ONLY_FLOWS}`],
  ["{crm: lookup}", `\`crm\` is a tool module, so \`{crm: lookup}\` is a tool, ${RUNS_ONLY_FLOWS}`],
  ["{greet: x}", "`greet` is a flow, not a library"],
])("a loop naming %s refuses it", async (call, message) => {
  expect(await refusal(app(`  n: {kind: loop, call: ${call}, times: 1}\n`))).toBe(`loop node 'n': ${message}`);
});
