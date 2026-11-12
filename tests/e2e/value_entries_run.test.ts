/**
 * Runs of flows with value entries in `nodes:` (`exit_result: !object {exited: true}`).
 *
 * A tagged entry lowers to a CODE node that returns its value, so a flow output, another node, or
 * a CASE route reads it as `${<id>.output}`. These runs cover literal and template entries (a span
 * in a nested list too), named and nested tags, keys a node body would treat specially, `$$`
 * escapes, top-level `call(...)` entries, value entries inside `defs:`, inline CALL, LOOP and MAP
 * bodies, and YAML anchors, aliases and merge keys on value entries.
 *
 * An entry is skipped like any node: when a CASE routes away from it, or when a whole-span read
 * names a skipped node, while an embedded span renders a skipped read as empty text and an
 * unrouted entry with no reads always runs. A routed `call(...)` entry skips its lifted call with
 * it, and a resumed log replays an entry's recorded output.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { NodeSkipped, NodeStarted } from "../../src/events.ts";
import { type RunResult, advance, runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { rejected } from "../helpers/run_state_fixtures.ts";
import { outputTag } from "../helpers/output_tags.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-value-entries-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/** Whether `nodeId` is a node of the flow itself: not a boundary, not inside a call. */
const flowNode = (nodeId: string) => nodeId !== "__start__" && nodeId !== "__end__" && !nodeId.includes("/");

/** Return the ids of the flow's nodes that started in a run. */
const started = (result: RunResult) =>
  new Set(
    result.events.flatMap((e) => (e instanceof NodeStarted && flowNode(e.nodeId) ? [e.nodeId] : [])),
  );

/** Return the ids of the flow's nodes skipped in a run. */
const skipped = (result: RunResult) =>
  new Set(
    result.events.flatMap((e) => (e instanceof NodeSkipped && flowNode(e.nodeId) ? [e.nodeId] : [])),
  );

/** Return each recorded node output's commit in a run log, `{kind, value}`, keyed by node id. */
function succeededValues(path: string): Record<string, { kind: unknown; value: unknown }> {
  const values: Record<string, { kind: unknown; value: unknown }> = {};
  for (const line of readFileSync(path, "utf8").split("\n")) {
    if (line === "") continue;
    const record = JSON.parse(line);
    if (record.kind === "pool_commit") {
      values[record.payload.node_id] = { kind: record.payload.kind, value: record.payload.value };
    }
  }
  return values;
}

/** Load and run `text` in the test's environment. */
async function runText(text: string, inputs: Record<string, unknown> = {}): Promise<RunResult> {
  return runFlow(await loadFlow(text, { env }), inputs, { env });
}

/** Load and run `text`, expecting success, and return its output. */
async function run(text: string, inputs: Record<string, unknown> = {}): Promise<unknown> {
  const result = await runText(text, inputs);
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result.output;
}

/** Return a flow with a `name: str` input, int producers `a` and `b`, and `entries`. */
const flow = (entries: string, { output = "${x.output}", head = "" } = {}) =>
  "id: f\n" +
  "name: f\n" +
  head +
  "input:\n" +
  "  name: !str\n" +
  "nodes:\n" +
  "  a:\n" +
  "    kind: code\n" +
  "    output: !int\n" +
  "    code: |\n" +
  "      return 2\n" +
  "  b:\n" +
  "    kind: code\n" +
  "    output: !int\n" +
  "    code: |\n" +
  "      return 3\n" +
  entries +
  `output: ${output}\n`;

const BUY = "typedefs:\n  Buy:\n    action: Literal(buy)\n    conviction: float\n";

const ENTRY_ROWS: [string, unknown][] = [
  ["!int 3", 3n],
  ["!float 1.5", 1.5],
  ["!bool true", true],
  ["!Optional(str) null", null],
  ["!list(str) [a, b]", ["a", "b"]],
  ["!int ${a.output + b.output}", 5n],
  ["!str Hello, ${input.name}", "Hello, Ada"],
  ["!object {exited: true, reply: ''}", { exited: true, reply: "" }],
  ["!str costs $$5", "costs $5"],
  ["!object {a: '$${x}'}", { a: "${x}" }],
  ["!str 'call(f)'", "call(f)"],
];

test.each(ENTRY_ROWS)("a value entry runs to its value (%s)", async (entry, expected) => {
  const output = await run(flow(`  x: ${entry}\n`), { name: "Ada" });

  expect(output).toEqual(expected);
  expect(typeof output).toBe(typeof expected);
});

test("a nested list with a span runs", async () => {
  const text = flow('  x: !object\n    items:\n      - "${input.name}"\n      - x\n');

  expect(await run(text, { name: "Ada" })).toEqual({ items: ["Ada", "x"] });
});

test("a nested tag runs", async () => {
  const text = flow("  x: !object\n    d: !Buy {action: buy, conviction: 1}\n    n: ${a.output}\n", { head: BUY });

  expect(await run(text, { name: "Ada" })).toEqual({ d: { action: "buy", conviction: 1n }, n: 2n });
});

test("node body keys in a value are kept", async () => {
  const value = { input: 1n, output: 2n, inputs: 3n, kind: "call", call: "f", nodes: {} };
  const text = flow("  x: !object {input: 1, output: 2, inputs: 3, kind: call, call: f, nodes: {}}\n");

  expect(await run(text, { name: "Ada" })).toEqual(value);
});

const F_DEF =
  "defs:\n" +
  "  f:\n" +
  "    input:\n" +
  "      x: !str\n" +
  "    nodes:\n" +
  "      up:\n" +
  "        kind: code\n" +
  "        input:\n" +
  "          x: ${input.x}\n" +
  "        output: !str\n" +
  "        code: |\n" +
  "          return inputs['x'].toUpperCase()\n" +
  "    output: ${up.output}\n";

test("a value entry in an inline call body", async () => {
  const text = flow(
    "  x:\n" +
      "    kind: call\n" +
      "    input:\n" +
      "      who: ${input.name}\n" +
      "    nodes:\n" +
      "      hi: !str hi ${input.who}\n" +
      "    output: ${hi.output}\n",
  );

  expect(await run(text, { name: "Ada" })).toBe("hi Ada");
});

test("a value entry in an inline loop body", async () => {
  const text = flow(
    "  x:\n" +
      "    kind: loop\n" +
      "    times: 3\n" +
      "    input:\n" +
      "      n: 0\n" +
      "    nodes:\n" +
      "      bump: !Count\n" +
      "        n: ${input.n + 1}\n" +
      "    output: ${bump.output}\n",
    { head: "typedefs:\n  Count:\n    n: int\n" },
  );

  expect(await run(text, { name: "Ada" })).toEqual({ n: 3n });
});

test("an anchored value entry and its aliases", async () => {
  const text = flow("  base: &b !object {a: 1}\n  copy: *b\n  x: !object\n    inner: *b\n", {
    output: "\n  base: ${base.output}\n  copy: ${copy.output}\n  x: ${x.output}",
  });

  expect(await run(text, { name: "Ada" })).toEqual({ base: { a: 1n }, copy: { a: 1n }, x: { inner: { a: 1n } } });
});

test("a merge key inside a value entry", async () => {
  const text = flow("  x: !object\n    <<: *shared\n    c: 3\n", { head: "x-shared: &shared {a: 1, b: 2}\n" });

  expect(await run(text, { name: "Ada" })).toEqual({ a: 1n, b: 2n, c: 3n });
});

/** Return a CODE node `p` returning text, declaring `output` unless it is empty. */
const producer = (output: string) => {
  const declared = output ? `    output: ${outputTag(output)}\n` : "";
  return `  p:\n    kind: code\n${declared}    code: |\n      return 'hi'\n`;
};

const RECORD_ROWS: [string, string, unknown][] = [
  ["  r: !object {exited: true, reply: ''}\n", "${r.output.reply}", ""],
  ["  r: !object {exited: true, reply: ''}\n", "${r.output}", { exited: true, reply: "" }],
  [producer("str") + "  r: !object\n    a: ${p.output}\n    b: 1\n", "${r.output}", { a: "hi", b: 1n }],
  [producer("") + "  r: !object\n    a: ${p.output}\n    b: 1\n", "${r.output.a}", "hi"],
  [producer("str") + "  r: !object\n    a: ${p.output}\n    p: {q: 1}\n", "${r.output.p.q}", 1n],
  [producer("") + "  r: !object\n    a: ${p.output}\n    p: !object {q: 1}\n", "${r.output.p}", { q: 1n }],
  ["  r: !object\n    p:\n      q: [1, 2]\n", "${r.output.p.q}", [1n, 2n]],
  ["  r: !object {}\n", "${r.output}", {}],
  ['  r: !object {a: "$${input.name}"}\n', "${r.output}", { a: "${input.name}" }],
  [
    "  x:\n    kind: code\n    output: {q: !int}\n    code: |\n      return {'q': 1}\n  r: !object ${x.output}\n",
    "${r.output}",
    { q: 1n },
  ],
];

test.each(RECORD_ROWS)("a written out object entry runs with its record (row %#)", async (entries, output, expected) => {
  expect(await run(flow(entries, { output }), { name: "Ada" })).toEqual(expected);
});

const ROUTE_FLOW = `
id: f
name: f
input:
  cmd: !str
nodes:
  route:
    kind: case
    on: \${input.cmd}
    cases:
      - when: /exit
        then: exit_result
    else: reply
  exit_result: !object {exited: true, reply: ""}
  reply:
    kind: code
    input:
      text: \${input.cmd}
    code: |
      return {'exited': false, 'reply': inputs['text']}
output: \${route.output}
`;

const ROUTE_ROWS: [string, string, string, unknown][] = [
  ["/exit", "exit_result", "reply", { exited: true, reply: "" }],
  ["hi", "reply", "exit_result", { exited: false, reply: "hi" }],
];

test.each(ROUTE_ROWS)("a case routes to a value entry or a node (%s)", async (cmd, taken, untaken, expected) => {
  const result = await runText(ROUTE_FLOW, { cmd });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(expected);
  expect(started(result)).toEqual(new Set(["route", taken]));
  expect(skipped(result)).toEqual(new Set([untaken]));
});

const SKIP_FLOW = `
id: f
name: f
input:
  cmd: !str
nodes:
  route:
    kind: case
    on: \${input.cmd}
    cases:
      - when: go
        then: maybe
  maybe:
    kind: code
    output: !str
    code: |
      return 'm'
  whole: !object
    a: \${maybe.output}
  embedded: !str "got \${maybe.output}"
  lone: !int 7
output:
  whole: \${whole.output}
  embedded: \${embedded.output}
  lone: \${lone.output}
`;

test("entries reading a routed node", async () => {
  const result = await runText(SKIP_FLOW, { cmd: "go" });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ whole: { a: "m" }, embedded: "got m", lone: 7n });
  expect(started(result)).toEqual(new Set(["route", "maybe", "whole", "embedded", "lone"]));
  expect(skipped(result)).toEqual(new Set());
});

/** Return a flow whose CASE on a `Union(Buy, Sell)` tag routes `buy` to `buy`. */
const unionFlow = ({ buy, other }: { buy: string; other: string }) =>
  "id: f\n" +
  "name: f\n" +
  "typedefs:\n" +
  "  Buy:\n" +
  "    action: Literal(buy)\n" +
  "    qty: int\n" +
  "  Sell:\n" +
  "    action: Literal(sell)\n" +
  "    price: float\n" +
  "input:\n" +
  "  kind: !str\n" +
  "nodes:\n" +
  "  d:\n" +
  "    kind: code\n" +
  "    input:\n" +
  "      kind: ${input.kind}\n" +
  "    code: |\n" +
  '      if (inputs["kind"] === "buy")\n' +
  '        return {"action": "buy", "qty": 3}\n' +
  '      return {"action": "sell", "price": 1.5}\n' +
  "    output: !Union(Buy;Sell)\n" +
  "  route:\n" +
  "    kind: case\n" +
  "    on: ${d.output.action}\n" +
  "    cases:\n" +
  "      - when: buy\n" +
  "        then: buy\n" +
  "    else: other\n" +
  `  buy: !object\n    qty: ${buy}\n` +
  `  other: !object\n    qty: ${other}\n` +
  "output: ${route.output}\n";

test("a routed entry reads a union member field its branch selects", async () => {
  const loaded = await loadFlow(unionFlow({ buy: '"${d.output.qty}"', other: "0" }), { env });

  expect((await runFlow(loaded, { kind: "buy" }, { env })).output).toEqual({ qty: 3n });
  expect((await runFlow(loaded, { kind: "sell" }, { env })).output).toEqual({ qty: 0n });
});

const CALL_FLOW =
  F_DEF +
  `
id: f
name: f
input:
  cmd: !str
  t: !str
nodes:
  route:
    kind: case
    on: \${input.cmd}
    cases:
      - when: go
        then: e
    else: other
  other:
    kind: code
    code: |
      return 'other'
  e: !str call(f, x=\${input.t})
output: \${route.output}
`;

test("a routed flow call entry runs its call only when taken", async () => {
  const loaded = await loadFlow(CALL_FLOW, { env });
  const callIds = [...loaded.compiled.nodes.keys()].filter((nodeId) => nodeId.startsWith("__call_"));
  expect(callIds).toHaveLength(1);
  const callId = callIds[0]!;

  const taken = await runFlow(loaded, { cmd: "go", t: "ada" }, { env });
  const untaken = await runFlow(loaded, { cmd: "no", t: "ada" }, { env });

  expect(taken.failure).toBeNull();
  expect(taken.status).toBe("succeeded");
  expect(taken.output).toBe("ADA");
  expect(started(taken)).toEqual(new Set(["route", "e", callId]));
  expect(skipped(taken)).toEqual(new Set(["other"]));
  expect(untaken.failure).toBeNull();
  expect(untaken.status).toBe("succeeded");
  expect(untaken.output).toBe("other");
  expect(started(untaken)).toEqual(new Set(["route", "other"]));
  expect(skipped(untaken)).toEqual(new Set(["e", callId]));
});

const GATE_FLOW = `
id: f
name: f
input:
  t: !str
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      prompt: "Command?"
  route:
    kind: case
    on: \${ask.output}
    cases:
      - when: exit
        then: exit_result
    else: other
  exit_result: !object
    reply: \${input.t}
    exited: true
  other:
    kind: code
    code: |
      return {'reply': 'still here', 'exited': false}
output: \${route.output}
`;

test("a resumed run routes to a value entry and replays it", async () => {
  const path = join(dir, "gate.jsonl");
  const interruption = new Error("interrupt before human answer");
  const host = new AbortController();
  const interrupt = () => {
    host.abort(interruption);
    throw interruption;
  };

  const caught = await rejected(
    runFlow(await loadFlow(GATE_FLOW, { env }), { t: "bye" }, { env, log: path, signal: host.signal, tools: { ask: interrupt } }),
  );
  expect(caught).toBe(interruption);
  expect("exit_result" in succeededValues(path)).toBe(false);

  const answered = await advance({ log: path, env, tools: { ask: () => "exit" } });

  expect(answered.failure).toBeNull();
  expect(answered.status).toBe("succeeded");
  expect(answered.output).toEqual({ reply: "bye", exited: true });
  expect(started(answered)).toEqual(new Set(["ask", "route", "exit_result"]));
  expect(skipped(answered)).toEqual(new Set(["other"]));
  expect(succeededValues(path)["exit_result"]!.value).toEqual(answered.output);
  const before = readFileSync(path);

  const replayed = await advance({ log: path, env });

  expect(replayed.failure).toBeNull();
  expect(replayed.status).toBe("succeeded");
  expect(replayed.output).toEqual(answered.output);
  expect(replayed.events.filter((e) => e instanceof NodeStarted)).toEqual([]);
  expect(readFileSync(path).equals(before)).toBe(true);
});
