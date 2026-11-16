/**
 * `call:` names a tool on a `call` node and in a compact flow: a name the file defines means the
 * file's own flow or module, any other name is a tool the run supplies, and a `call` node naming a
 * tool loads as the TOOL node `kind: tool` builds, under its field rules, in the author's words.
 */

import { afterAll, beforeAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { outputContractType } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { runFlow } from "../../src/run.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";

import { loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

let tmp: string;

// Its top level throws, so a load that imported it would fail.
const MODULE = 'throw new Error("imported");\nexport const lookup = {};\n';

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
 * What node `n` of the flow loaded from `text` runs: `tool <id>`, with ` module <alias>` or
 * ` memory` when it names one, or `flow <id>`, with `.<def>` for a library's def.
 */
async function target(text: string): Promise<string> {
  const node = (await load(text)).compiled.nodes.get("n");
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
  expect(await refusal(app(`  n: {kind: call, call: ${call}}\n`))).toBe(message);
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

test("a compact flow's call: names a tool", async () => {
  const flow = await loadFlow("id: s\nname: s\ninput:\n  q: !str\noutput: !int\nkind: call\ncall: search\n");
  const node = flow.compiled.nodes.get("s") as ToolNode;

  expect(node.toolId).toBe("search");
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe("int");
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["q"]);
});

test("a compact flow's call: of a memory input names one of its tools", async () => {
  const text = "id: s\nname: s\ninput:\n  lessons: !FileMemory\nkind: call\ncall: lessons\n";
  expect((await loadErrorAsync(loadFlow(text))).message).toBe(
    "node 's': `lessons` is a memory input; name one of its tools: `call: {lessons: <tool>}`",
  );
});
