/**
 * A `map` whose `call:` names a tool runs it once per item through a one-node body yian generates
 * at load: the body's node takes the map's id and the author's `call:`, `input:` keys, `retries:`,
 * `output:`, `adaptive_questions:` and `asserts:`, so its load errors point at the author's fields.
 */

import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { FileMemory } from "../../src/memory/file.ts";
import { outputContractType } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { runFlow, runStartTools, staticNodes } from "../../src/run.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { canonicalTypeLabel } from "../../src/typesys/values.ts";

import { loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

let tmp: string;

beforeAll(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-map-tools-")));
  mkdirSync(join(tmp, "lessons"));
  writeFileSync(join(tmp, "lessons", "a.md"), "one");
  writeFileSync(join(tmp, "lessons", "b.md"), "two");
});

afterAll(() => rmSync(tmp, { recursive: true, force: true }));

/** The memory `lessons`, whose files `a.md` and `b.md` hold `one` and `two`. */
const lessons = () => new FileMemory({ root: join(tmp, "lessons"), id: "t/lessons" });

/** A flow with the inputs `ids`, the memory `lessons` and `inputs` (lines under `input:`), and `nodes`. */
const app = (nodes: string, inputs = "") =>
  `id: app\nname: app\ninput:\n  ids: !list(str)\n  lessons: !FileMemory\n${inputs}nodes:\n${nodes}`;

/** `app` whose map `fetch` over `${input.ids}` calls `call` with the node's other `lines`. */
const mapApp = (call: string, lines = "", inputs = "") =>
  app(`  fetch:\n    kind: map\n    over: \${input.ids}\n    call: ${call}\n${lines}`, inputs);

/** The TOOL node a map's generated body runs, by the map's id. */
async function bodyNode(text: string, mapId = "fetch"): Promise<ToolNode> {
  const found = staticNodes(await loadFlow(text)).find(([path]) => path === `${mapId}/${mapId}`);
  return found![1] as ToolNode;
}

/** The message and the source text at the range loading `text` fails with. */
async function failure(text: string): Promise<[string, string]> {
  const error = await loadErrorAsync(loadFlow(text));
  const range = error.diagnostic.frames.at(-1)!.range!;
  const lines = text.split("\n");
  const at =
    range.startLine === range.endLine
      ? lines[range.startLine - 1]!.slice(range.startCol - 1, range.endCol - 1)
      : `${lines[range.startLine - 1]!.slice(range.startCol - 1)}...`;
  return [error.message, at];
}

/** A host tool taking string `q` that records each call's id and returns `found <q>`. */
function search(calls: string[]) {
  return tool({
    name: "search",
    description: "A host tool.",
    parameters: { type: "object", properties: { q: { type: "string" } }, required: ["q"] },
    execute: (args: Record<string, unknown>, ctx) => {
      calls.push(ctx.callId);
      return `found ${String(args["q"])}`;
    },
  });
}

test("a map runs a host tool once per item, each call under the map's own node id", async () => {
  const text = `${mapApp("search", "    input:\n      q: ${item}\n")}output: \${fetch.output}\n`;
  const calls: string[] = [];
  const result = await runFlow(await loadFlow(text), { ids: ["a", "b"], lessons: lessons() }, {
    env: hostEnv(tmp),
    runId: "r",
    tools: { search: search(calls) },
  });

  expect(result.output).toEqual(["found a", "found b"]);
  expect(calls.sort()).toEqual(["r/fetch#0/fetch", "r/fetch#1/fetch"]);
});

test("retries:, output: and asserts: move to the body's node; the map keeps its own fields", async () => {
  const lines =
    "    input:\n      q: ${item}\n    retries: 2\n    output: !str\n    asserts: [output != '']\n" +
    "    node_name: Fetch\n";
  const node = await bodyNode(mapApp("search", lines));

  expect(node.id).toBe("fetch");
  expect(node.retries).toBe(2);
  expect(canonicalTypeLabel(outputContractType(node.outputContract)!)).toBe("str");
  expect(node.postAsserts.map((assertion) => assertion.display)).toEqual(["output != ''"]);
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["q"]);
  const map = (await loadFlow(mapApp("search", lines))).compiled.nodes.get("fetch")!;
  expect(map.title).toBe("Fetch");
  expect(canonicalTypeLabel(outputContractType(map.outputContract)!)).toBe("list(str)");
});

test("a body input takes its binding's proven type, else Optional(Any)", async () => {
  const lines =
    "    input:\n      q: ${item}\n      n: !int ${other.output}\n      raw: ${other.output}\n" +
    "  other: {kind: call, call: lookup}\n";
  const flow = await loadFlow(mapApp("search", lines));
  const start = staticNodes(flow).find(([path]) => path === "fetch/__start__")![1];
  const types = Object.fromEntries(
    start.params!.map((parameter) => [parameter.name, canonicalTypeLabel(parameter.contract!.type)]),
  );

  expect(types).toEqual({ q: "str", n: "int", raw: "Optional(Any)" });
});

// Each row: the map's `call:` and other lines, the load error, and the text it points at.
test.each([
  ["search", "    retries: -1\n", "node 'fetch' (kind=call): `retries:` must be a non-negative integer, got -1", "-1"],
  ["search", "    output: !Nope\n", "unknown type 'Nope'", "!Nope"],
  [
    "search",
    "    adaptive_questions: {prompt: hi}\n",
    "node 'fetch' (call: search): `adaptive_questions:` applies only to call: ask",
    "adaptive_questions",
  ],
  [
    "{crm: [a]}",
    "",
    "node 'fetch' (kind=map): `call: {crm: ...}` names one def or tool; write `call: {crm: a}`",
    "{crm: [a]}",
  ],
  [
    "sleep",
    "    input:\n      for: ${other.output}\n  other: {kind: call, call: lookup}\n",
    "node 'fetch' (call: sleep) input 'for': a duration reference must be typed str (tag it: `for: !str ${...}`)",
    "${other.output}",
  ],
])("call: %s with %j fails at the author's field", async (call, lines, message, at) => {
  const [got, where] = await failure(mapApp(call, lines));
  expect(got).toContain(message);
  expect(where).toBe(at);
});

test("take is refused at call:, since a map can't give each item an inbox", async () => {
  const text = mapApp("take", "    input:\n      from: ${input.box}\n", "  box: !Inbox(str)\n");
  expect(await failure(text)).toEqual([
    "map node 'fetch': `call: take` reads an inbox, and a map can't give each item its own (per-instance " +
      "inboxes aren't supported yet); take before the map",
    "take",
  ]);
});

test("a tool given an inbox gets the map's inbox rule, naming the tool", async () => {
  const text = mapApp("search", "    input:\n      box: ${input.box}\n", "  box: !Inbox(str)\n");
  expect((await failure(text))[0]).toBe(
    "map node 'fetch': tool `search` declares inbox input 'box'; a map body can't read an inbox (per-instance " +
      "inboxes aren't supported yet)",
  );
});

test("a memory's tool reads the memory through the body, never as an argument", async () => {
  const text = `${mapApp("{lessons: read}", "    input:\n      path: ${item}\n")}output: \${fetch.output}\n`;
  const node = await bodyNode(text);
  expect(node.params!.map((parameter) => parameter.name)).toEqual(["path"]);
  expect(node.memoryTool).toBe(true);

  const result = await runFlow(await loadFlow(text), { ids: ["a.md", "b.md"], lessons: lessons() }, {
    env: hostEnv(tmp),
  });
  expect((result.output as { text: string }[]).map((each) => each.text)).toEqual(["one", "two"]);
});

test("an input: key named as the memory the map calls is refused at the key", async () => {
  const text = mapApp("{lessons: read}", "    input:\n      lessons: ${input.lessons}\n      path: a.md\n");
  expect(await failure(text)).toEqual([
    "map node 'fetch': `lessons` is the memory this node calls; yian passes it, so remove the entry",
    "lessons",
  ]);
});

test("run start names a map's tool by the map", async () => {
  const text =
    "id: r\nname: r\ndefs:\n  c:\n    nodes:\n" +
    '      fetch: {kind: map, over: [a], call: serch, input: {q: "${item}"}}\n' +
    "    output: ${fetch.output}\nnodes:\n  run: {kind: call, call: c}\n";
  const flow = await loadFlow(text);

  expect(runStartTools(flow)).toEqual([{ node: "run/fetch", site: "map node 'fetch'", name: "serch", hint: null }]);
  const run = runFlow(flow, {}, { env: hostEnv(tmp), tools: { search: search([]) } });
  expect(((await run.catch((caught: unknown) => caught)) as Error).message).toBe(
    "map node 'fetch' in 'run' names tool 'serch', which isn't in this run's tools; known tools: [\"search\"]; " +
      "did you mean 'search'?",
  );
});

// Each row: the field on a map naming the flow `greet`, and the refusal.
test.each([
  ["retries: 2", "`retries:` doesn't apply to a flow call: a flow retries inside its own nodes"],
  ["output: !int", "`output:` doesn't apply to a flow call: a flow's result type comes from the flow"],
  ["adaptive_questions: {prompt: hi}", "`adaptive_questions:` applies only to `call: ask`"],
])("a map naming a flow refuses %s", async (field, problem) => {
  const text =
    'id: r\nname: r\ndefs:\n  greet: {input: {x: !int}, kind: code, code: "return 1", output: !int}\n' +
    `nodes:\n  m: {kind: map, over: [1], call: greet, input: {x: "\${item}"}, ${field}}\n`;
  expect((await failure(text))[0]).toBe(`node 'm' (kind=map): ${problem}`);
});

test("a map over a flow keeps its fingerprint", async () => {
  const text =
    "id: r\nname: r\ninput:\n  ns: !list(int)\n" +
    'defs:\n  double: {input: {x: !int}, kind: code, code: "return inputs.x * 2", output: !int}\n' +
    "nodes:\n" +
    '  named: {kind: map, call: double, over: "${input.ns}", input: {x: "${item}"}}\n' +
    '  inline:\n    kind: map\n    over: "${input.ns}"\n    input: {x: "${item}"}\n' +
    '    nodes:\n      d: {kind: code, code: "return inputs.x + 1", input: {x: "${input.x}"}, output: !int}\n' +
    "    output: ${d.output}\n" +
    "output:\n  named: ${named.output}\n  inline: ${inline.output}\n";
  expect(flowFingerprint((await loadFlow(text)).compiled)).toBe(
    "2a00c8f82aaae892fab1e07490e03b76f2109a09efffa7a6efd5a7c8976cd91d",
  );
});
