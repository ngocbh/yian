/**
 * Load-time field-read checks in prompts, bindings, node and flow asserts and on builtin results.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { registerTemplateFn, TEMPLATE_FNS } from "../../src/expr/builtins.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../engine/mcp_helpers.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadConformanceFlow,
  loadError,
  runCase,
} from "./declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-field-read-sites-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const PREAMBLE =
  "id: p\nname: p\n" +
  "typedefs:\n" +
  "  Rec: {a: str}\n" +
  "  Rec3: {a: str, b: Optional(str)}\n" +
  "  Buy: {action: 'Literal(buy)', qty: int}\n" +
  "  Sell: {action: 'Literal(sell)'}\n" +
  "  Order: Union(Buy, Sell)\n";

const INPUTS = "input:\n  s: !str\n  r: !Rec\n";

const STEP =
  "  step:\n" +
  "    kind: code\n" +
  "    input:\n      x: ${input.s}\n" +
  "    output: Rec\n" +
  '    code: "return ({a: \\"x\\"})"\n';

const ASSERT_QTY_HINT =
  "reads field 'qty', which only union members with action buy have; " +
  "check the member first with `isinstance(v, <Type>) and ...`";

const QTY_HINT =
  "reads field 'qty', which only union members with action buy have; read it inside a case on ${d.output.action}";

const UNION_TYPES =
  "  Short: {action: 'Literal(short)', qty: int}\n" +
  "  Hold: {action: 'Literal(hold)'}\n" +
  "  Trade: Union(Buy, Short, Hold)\n" +
  "  Sized: Union(Buy, Short)\n" +
  "  A: {x: int}\n" +
  "  B: {y: str}\n" +
  "  AB: Union(A, B)\n";

/**
 * Return a flow built from the shared preamble and the given sections: the shared typedefs plus
 * `typedefs` lines, `defs`, the inputs (plus `inputs` lines), node `step`, then `nodes`, `output`
 * and `tail`.
 */
function flow(
  nodes: string,
  output: string,
  options: { typedefs?: string; defs?: string; inputs?: string; tail?: string } = {},
): string {
  const { typedefs = "", defs = "", inputs = "", tail = "" } = options;
  return PREAMBLE + typedefs + defs + INPUTS + inputs + "nodes:\n" + STEP + nodes + `output: ${output}\n` + tail;
}

/** An AGENT node with `inputs`, a quoted `prompt`, and extra YAML lines. */
function agent(
  inputs: Record<string, string>,
  prompt: string,
  options: { nodeId?: string; extra?: string } = {},
): string {
  const { nodeId = "ag", extra = "" } = options;
  let text = `  ${nodeId}:\n    kind: agent\n`;
  const entries = Object.entries(inputs);
  if (entries.length > 0) {
    text += "    input:\n";
    for (const [name, source] of entries) text += `      ${name}: ${source}\n`;
  }
  return text + `    prompt: "${prompt}"\n` + extra;
}

/** Assemble `text`. */
const load = (text: string) => assembleText(text);

/** Return the message of the `LoadError` assembling `text` throws. */
const loadMessage = (text: string) => loadError(() => load(text)).message;

/** The inputs every run passes: the flow's `s` and `r`. */
const INPUT_VALUES = { s: "x", r: { a: "x" } };

/** Load and run `text` on `INPUT_VALUES` and `inputs`. */
const run = async (text: string, inputs: Record<string, unknown> = {}) =>
  runFlow(await loadConformanceFlow(text), { ...INPUT_VALUES, ...inputs }, { env });

/** A CODE node typed `output` reading the flow input `s`. */
function producer(nodeId: string, output: string): string {
  return `  ${nodeId}:\n    kind: code\n    input:\n      s: \${input.s}\n    output: ${output}\n    code: "return null"\n`;
}

/** Producer `d` (Order), a case on its action routing buy to `ag`, sell to `other`. */
function routed(node: string): string {
  return (
    producer("d", "Order") +
    "  route:\n" +
    "    kind: case\n" +
    "    on: ${d.output.action}\n" +
    "    cases:\n" +
    "      - when: buy\n        then: ag\n" +
    "      - when: sell\n        then: other\n" +
    node +
    "  other:\n" +
    "    kind: code\n" +
    "    input:\n      s: ${input.s}\n" +
    "    output: str\n" +
    "    code: return 'x'\n"
  );
}

/** A CODE node returning `'x'` (typed `str`) with `inputs` and `asserts`. */
function code(inputs: Record<string, string>, asserts: readonly string[], nodeId = "c"): string {
  let text = `  ${nodeId}:\n    kind: code\n    input:\n`;
  for (const [name, source] of Object.entries(inputs)) text += `      ${name}: ${source}\n`;
  text += "    output: str\n    code: return 'x'\n    asserts:\n";
  return text + asserts.map((a) => `      - "${a}"\n`).join("");
}

/** The number of times `needle` occurs in `text`. */
const count = (text: string, needle: string) => text.split(needle).length - 1;

/** The 1-based line of `text` holding `needle`. */
const lineOf = (text: string, needle: string) => text.split("\n").findIndex((line) => line.includes(needle)) + 1;

/** The `defs:` text of a `child` taking `v: !<typ>`. */
function childText(typ: string): string {
  return (
    "defs:\n" +
    "  child:\n" +
    `    input:\n      v: !${typ}\n` +
    "    nodes:\n" +
    "      inner:\n" +
    "        kind: code\n" +
    "        input:\n          v: ${input.v}\n" +
    "        output: str\n" +
    "        code: return 'x'\n" +
    "    output: ${inner.output}\n"
  );
}

// ---------- prompts ----------

test("agent prompt reads unknown field", () => {
  expect(loadMessage(flow(agent({ v: "${step.output}" }, "hi ${v.b}"), "${ag.output}"))).toContain(
    "node 'ag' prompt: reference ${v.b} reads unknown field 'b' on record",
  );
});

test("agent system_prompt reads unknown field", () => {
  const nodes = agent({ v: "${step.output}" }, "hi", { extra: '    system_prompt: "sys ${v.b}"\n' });
  expect(loadMessage(flow(nodes, "${ag.output}"))).toContain(
    "node 'ag' system_prompt: reference ${v.b} reads unknown field 'b' on record",
  );
});

test("typed input field read errors only when every alternative rejects", () => {
  expect(loadMessage(flow(agent({ w: "!Rec ${step.output}" }, "${w.b}"), "${ag.output}"))).toContain(
    "node 'ag' prompt: reference ${w.b} reads unknown field 'b' on record",
  );
  load(flow(agent({ w: "!Rec3 ${step.output}" }, "${w.b}"), "${ag.output}"));
});

test("typed union input reads the narrowed member inside a branch", () => {
  load(flow(routed(agent({ v: "!Order ${d.output}" }, "${v.qty}")), "${ag.output | other.output}"));
  const outside = producer("d", "Order") + agent({ v: "!Order ${d.output}" }, "${v.qty}");
  expect(loadMessage(flow(outside, "${ag.output}"))).toContain(`node 'ag' prompt: reference \${v.qty} ${QTY_HINT}`);
});

test.each([
  ["template", "!Order {action: buy, qty: '${n.output}'}"],
  ["literal", "!Order {action: buy, qty: 1}"],
])("typed record binding reads its tagged member: %s", (_id, source) => {
  load(flow(producer("n", "int") + agent({ v: source }, "${v.qty}"), "${ag.output}"));
});

test("optional binding field reads", () => {
  expect(loadMessage(flow(agent({ v: "${step.output :- null}" }, "${v.b}"), "${ag.output}"))).toContain(
    "node 'ag' prompt: reference ${v.b} reads unknown field 'b' on record",
  );
  load(flow(agent({ v: "${step.output :- null}" }, "${v?.a}"), "${ag.output}"));
});

test("adaptive brief field read is reported on the gate", () => {
  const gate =
    "  h:\n" +
    "    kind: tool\n" +
    "    tool_id: ask\n" +
    "    input:\n      v: ${step.output}\n" +
    '      prompt: "Pick for ${step.output.a}"\n' +
    "    adaptive_questions:\n" +
    '      prompt: "${v.b}"\n';
  expect(loadMessage(flow(gate, "${h.output}"))).toContain(
    "node 'h' adaptive_questions.prompt: reference ${v.b} reads unknown field 'b' on record",
  );
});

test("prompt inside a def reads unknown field", () => {
  const defs =
    "defs:\n" +
    "  child:\n" +
    "    input:\n      x: !Rec\n" +
    "    nodes:\n" +
    "      inner:\n" +
    "        kind: agent\n" +
    "        input:\n          x: ${input.x}\n" +
    '        prompt: "${x.b}"\n' +
    "    output: ${inner.output}\n";
  const nodes = "  c:\n    kind: call\n    call: child\n    input:\n      x: ${step.output}\n";

  expect(loadMessage(flow(nodes, "${c.output}", { defs }))).toContain(
    "reference ${x.b} reads unknown field 'b' on record",
  );
});

test("literal input field read", () => {
  expect(loadMessage(flow(agent({ mode: "fast" }, "${mode.x}"), "${ag.output}"))).toContain(
    "node 'ag' prompt: reference ${mode.x} reads field 'x' on type str, which has no fields",
  );
});

test("null literal input field read", () => {
  expect(loadMessage(flow(agent({ v: "null" }, "${v.x}"), "${ag.output}"))).toContain(
    "node 'ag' prompt: reference ${v.x} reads field 'x' on type None, which has no fields",
  );
});

test("field read inside a builtin argument", () => {
  expect(loadMessage(flow(agent({ v: "${step.output}" }, "${render_as_json(v.b)}"), "${ag.output}"))).toContain(
    "node 'ag' prompt: reference ${v.b} reads unknown field 'b' on record",
  );
});

test("agent variable field read", () => {
  const nodes = agent({}, "${cfg.b}", { extra: "    variables:\n      cfg: !Rec {a: x}\n" });
  expect(loadMessage(flow(nodes, "${ag.output}"))).toContain(
    "node 'ag' prompt: reference ${cfg.b} reads unknown field 'b' on record",
  );
});

test("ask prompt field read", () => {
  const gate = '  h:\n    kind: tool\n    tool_id: ask\n    input:\n      prompt: "${step.output.b}"\n';
  expect(loadMessage(flow(gate, "${h.output}"))).toContain(
    "node 'h' input 'prompt' from: reference ${step.output.b} reads unknown field 'b' on record",
  );
});

test("ask question field read", () => {
  const gate =
    "  h:\n" +
    "    kind: tool\n" +
    "    tool_id: ask\n" +
    "    input:\n" +
    '      prompt: "Pick for ${step.output.a}"\n' +
    "      questions:\n" +
    '        - question: "${step.output.b}"\n' +
    "          header: Pick\n" +
    "          options:\n" +
    "            - label: one\n" +
    "              description: the first\n";
  expect(loadMessage(flow(gate, "${h.output}"))).toContain(
    "node 'h' input 'questions' from: reference ${step.output.b} reads unknown field 'b' on record",
  );
});

test("a prompt field error is reported once per ref", () => {
  const message = loadMessage(flow(agent({ v: "${step.output}" }, "${v.b} and ${v.b}"), "${ag.output}"));
  expect(count(message, "reference ${v.b} reads unknown field 'b'")).toBe(1);
});

test.each([
  ["field", agent({ v: "${step.output}" }, "${v.a}")],
  ["default", agent({ v: "${step.output}" }, "${v.a :- 'n'}")],
  ["object", producer("o", "object") + agent({ v: "${o.output}" }, "${v.anything}")],
  ["any", agent({ v: "!Any ${step.output}" }, "${v.anything}")],
  [
    "untyped",
    "  u:\n    kind: code\n    input:\n      s: ${input.s}\n    code: return 1\n" +
      agent({ v: "${u.output}" }, "${v.x}"),
  ],
  ["optional-record", producer("m", "Optional(Rec)") + agent({ v: "${m.output}" }, "${v?.a}")],
])("readable fields still load: %s", (_id, nodes) => {
  load(flow(nodes, "${ag.output}"));
});

test("narrowed input reads the branch member", () => {
  load(flow(routed(agent({ v: "${d.output}" }, "${v.qty}")), "${ag.output | other.output}"));
  const outside = producer("d", "Order") + agent({ v: "${d.output}" }, "${v.qty}");
  expect(loadMessage(flow(outside, "${ag.output}"))).toContain(`node 'ag' prompt: reference \${v.qty} ${QTY_HINT}`);
});

test("a defaulted union binding hint names the bound source", () => {
  const nodes = producer("d", "Order") + agent({ v: '"${d.output :- null}"' }, "${v.qty}");
  expect(loadMessage(flow(nodes, "${ag.output}"))).toContain(`node 'ag' prompt: reference \${v.qty} ${QTY_HINT}`);
});

test("a transformed union binding hint names no bound source", () => {
  // A constructor wraps the value, so the hint can't name a path below its source.
  const nodes = producer("d", "Order") + agent({ v: "${Wrap(o=d.output)}" }, "${v.o.qty}");
  const message = loadMessage(flow(nodes, "${ag.output}", { typedefs: "  Wrap: {o: Order}\n" }));
  expect(message).toContain("read it inside a case on ${v.o.action}");
  expect(message).not.toContain("d.output.o");
});

// ---------- node asserts ----------

test.each([
  ["pre", "v.b == 'x'"],
  ["post", "output == 'x' and v.b == 'x'"],
])("a node assert reads an unknown input field: %s", (_id, assertion) => {
  expect(loadMessage(flow(code({ v: "${step.output}" }, [assertion]), "${c.output}"))).toContain(
    "node 'c' assert: reference ${v.b} reads unknown field 'b' on record",
  );
});

test("a node assert field error is reported once per ref", () => {
  const message = loadMessage(flow(code({ v: "${step.output}" }, ["v.b == null or v.b == 'y'"]), "${c.output}"));
  expect(count(message, "reference ${v.b} reads unknown field 'b'")).toBe(1);
});

test.each(["Rec", "Rec3"])("a call node assert reads a field its binding lacks: %s", (childType) => {
  const nodes =
    "  c:\n    kind: call\n    call: child\n" +
    "    input:\n      v: ${step.output}\n" +
    "    asserts:\n      - \"v.b == 'x'\"\n";
  const text = flow(nodes, "${c.output}", { defs: childText(childType) });
  expect(loadMessage(text)).toContain(
    "node 'c' assert: reference ${v.b} reads unknown field 'b' on record",
  );
});

const LOOP_NAMED_BODY =
  "defs:\n" +
  "  body:\n" +
  "    input:\n      st: !Rec3\n" +
  "    nodes:\n" +
  "      bump:\n" +
  "        kind: code\n" +
  "        input:\n          st: ${input.st}\n" +
  "        output:\n          st: Rec3\n" +
  '        code: "return ({st: {a: \\"x\\", b: \\"y\\"}})"\n' +
  "    output: ${bump.output}\n";

/** A LOOP over the named `body` (carries `Rec3`), seeded with `step`'s `Rec`. */
function namedLoop(assertion: string): string {
  const nodes =
    "  drive:\n    kind: loop\n    call: body\n    times: 3\n" +
    "    input:\n      st: ${step.output}\n" +
    `    asserts:\n      - "${assertion}"\n`;
  return flow(nodes, "${drive.output}", { defs: LOOP_NAMED_BODY });
}

/** A one-turn inline LOOP carrying `Rec3`, seeded with `step`'s `Rec`. */
function inlineLoop(assertion: string): string {
  const nodes =
    "  drive:\n    kind: loop\n    times: 1\n" +
    "    input:\n      st: ${step.output}\n" +
    `    asserts:\n      - "${assertion}"\n` +
    "    nodes:\n" +
    "      bump:\n" +
    "        kind: code\n" +
    "        input:\n          st: ${input.st}\n" +
    "        output:\n          st: Rec3\n" +
    '        code: "return ({st: {a: \\"x\\", b: \\"y\\"}})"\n' +
    "    output: ${bump.output}\n";
  return flow(nodes, "${drive.output}");
}

const LOOPS: [string, (assertion: string) => string][] = [
  ["named", namedLoop],
  ["inline", inlineLoop],
];

test.each(LOOPS)("a loop assert reads the carried field (loads): %s", (_id, build) => {
  load(build("st.b == null or st.b == 'y'"));
});

test.each(LOOPS)("a loop assert reads the carried field (runs): %s", async (_id, build) => {
  const result = await run(build("st.b == null or st.b == 'y'"));

  expect(result.status, result.failure?.message).toBe("succeeded");
});

test.each(LOOPS)("a loop assert reads a field neither seed nor carried has: %s", (_id, build) => {
  expect(loadMessage(build("st.zz == 'x'"))).toContain(
    "node 'drive' assert: reference ${st.zz} reads unknown field 'zz'",
  );
});

test("a node assert literal input field read", () => {
  expect(loadMessage(flow(code({ n: "3" }, ["n.x == 1"]), "${c.output}"))).toContain(
    "node 'c' assert: reference ${n.x} reads field 'x' on type int, which has no fields",
  );
});

test.each([
  ["any", code({ v: "!Any ${step.output}" }, ["v.k == 1"])],
  ["object", producer("o", "object") + code({ v: "${o.output}" }, ["v.k == 1"])],
])("node assert open input field reads load: %s", (_id, nodes) => {
  load(flow(nodes, "${c.output}"));
});

test("a node assert known input field loads and runs (loads)", () => {
  load(flow(code({ v: "${step.output}" }, ["v.a == 'x'"]), "${c.output}"));
});

test("a node assert known input field loads and runs (runs)", async () => {
  const result = await run(flow(code({ v: "${step.output}" }, ["v.a == 'x'"]), "${c.output}"));

  expect(result.status, result.failure?.message).toBe("succeeded");
});

test("a node assert reads the narrowed member inside a branch", () => {
  load(flow(routed(code({ v: "${d.output}" }, ["v.qty > 0"], "ag")), "${ag.output | other.output}"));
  const outside = producer("d", "Order") + code({ v: "${d.output}" }, ["v.qty > 0"]);
  expect(loadMessage(flow(outside, "${c.output}"))).toContain(`node 'c' assert: reference \${v.qty} ${ASSERT_QTY_HINT}`);
});

/**
 * Producer `d` typed `produced`, a case routing `isinstance(d.output, member)` to `ag` and anything
 * else to `other`, then `node` (id `ag`) and `other`.
 */
function isinstanceRouted(produced: string, member: string, node: string): string {
  return (
    producer("d", produced) +
    "  route:\n" +
    "    kind: case\n" +
    "    cases:\n" +
    `      - when: isinstance(d.output, ${member})\n        then: ag\n` +
    "    else: other\n" +
    node +
    "  other:\n" +
    "    kind: code\n" +
    "    input:\n      s: ${input.s}\n" +
    "    output: str\n" +
    "    code: return 'x'\n"
  );
}

test.each([
  ["untagged-prompt", "AB", "A", agent({ v: "${d.output}" }, "${v.x}")],
  ["untagged-assert", "AB", "A", code({ v: "${d.output}" }, ["v.x > 0"], "ag")],
  ["tagged-prompt", "Order", "Buy", agent({ v: "${d.output}" }, "${v.qty}")],
  ["shared-prompt", "Trade", "Sized", agent({ v: "${d.output}" }, "${v.qty}")],
  ["shared-assert", "Trade", "Sized", code({ v: "${d.output}" }, ["v.qty > 0"], "ag")],
])("an isinstance branch reads the narrowed member: %s", (_id, produced, member, node) => {
  load(flow(isinstanceRouted(produced, member, node), "${ag.output | other.output}", { typedefs: UNION_TYPES }));
});

/**
 * A flow whose node `c` asserts `assertion` on its local `v`, bound from `${d.output}` with `d`
 * typed `Order`; `c` is an AGENT node when `kind` is `agent`.
 */
function localAssert(assertion: string, kind = "code"): string {
  const node =
    kind === "agent"
      ? agent({ v: "${d.output}" }, "hi", { nodeId: "c", extra: `    asserts:\n      - "${assertion}"\n` })
      : code({ v: "${d.output}" }, [assertion]);
  return flow(producer("d", "Order") + node, "${c.output}");
}

test.each([
  "isinstance(v, Buy) and v.qty > 0",
  "not isinstance(v, Buy) or v.qty > 0",
  "isinstance(v, Sell) or v.qty > 0",
  "v.action == 'buy' and v.qty > 0",
  "v.action != 'buy' or v.qty > 0",
  "'sell' == v.action or v.qty > 0",
  "not (isinstance(v, Buy) and v.qty < 0)",
  "(isinstance(v, Buy) and v.qty > 0) or v.action == 'sell'",
  "isinstance(v, Buy) and (v.qty > 0 or v.qty < -10)",
])("node assert reads are narrowed by the checks before them: %s", (assertion) => {
  load(localAssert(assertion));
});

test.each([
  ["check-after-read", "v.qty > 0 and isinstance(v, Buy)", `reference \${v.qty} ${ASSERT_QTY_HINT}`],
  ["unguarded", "v.qty > 0", `reference \${v.qty} ${ASSERT_QTY_HINT}`],
  [
    "or-sees-the-other-member",
    "isinstance(v, Buy) or v.qty > 0",
    "reference ${v.qty} reads field 'qty', which Sell doesn't have; v is Sell here, by the checks before this read",
  ],
])("node assert reads the checks do not cover are rejected: %s", (_id, assertion, expected) => {
  expect(loadMessage(localAssert(assertion))).toContain(`node 'c' assert: ${expected}`);
});

test("a node assert read under two guards is checked under each", () => {
  const message = loadMessage(localAssert("v.qty > 0 or (isinstance(v, Buy) and v.qty > 1)"));
  expect(count(message, `reference \${v.qty} ${ASSERT_QTY_HINT}`)).toBe(1);
});

test.each<[string, Record<string, unknown>, "succeeded" | "failed"]>([
  ["sell", { action: "sell" }, "succeeded"],
  ["buy-positive", { action: "buy", qty: 2n }, "succeeded"],
  ["buy-zero", { action: "buy", qty: 0n }, "failed"],
])("guarded node assert runs: %s", async (_id, order, status) => {
  const node = code({ v: "${input.o}" }, ["not isinstance(v, Buy) or v.qty > 0"]);

  const result = await run(flow(node, "${c.output}", { inputs: ORDER_INPUT }), { o: order });

  expect(result.status, result.failure?.message).toBe(status);
  if (status === "failed") expect(result.failure!.code).toBe("assertion.failed");
});

test("agent node assert reads are narrowed", () => {
  load(localAssert("isinstance(v, Sell) or v.qty > 0", "agent"));
});

/** A flow whose CODE node `c` returns an `Order` and asserts `assertion`. */
function outputAssert(assertion: string): string {
  const node =
    "  c:\n    kind: code\n    input:\n      s: ${input.s}\n" +
    '    output: Order\n    code: "return ({action: \\"sell\\"})"\n' +
    `    asserts:\n      - "${assertion}"\n`;
  return flow(node, "${c.output}");
}

test("node assert output reads are narrowed", () => {
  load(outputAssert("isinstance(output, Buy) and output.qty > 0"));
  expect(loadMessage(outputAssert("output.qty > 0"))).toContain(
    "node 'c' assert: reference ${output.qty} reads field 'qty', which only union members with action " +
      "buy have; check the member first with `isinstance(output, <Type>) and ...`",
  );
});

test("a partial-union node assert reads the narrowed member", () => {
  // On a cycle a member-only field is readable, below a narrowed member too; the cycle is what the
  // load reports.
  const typedefs =
    "  Limit: {type: 'Literal(limit)', price: int}\n" +
    "  Market: {type: 'Literal(market)'}\n" +
    "  LBuy: {action: 'Literal(buy)', leg: 'Union(Limit, Market)'}\n" +
    "  LOrder: Union(LBuy, Sell)\n";
  const nodes =
    producer("d", "LOrder") +
    "  b:\n    kind: code\n    input:\n      y: ${c.output}\n" +
    "    output: str\n    code: return 'x'\n" +
    code({ v: "${d.output}", y: "${b.output}" }, ["isinstance(v, LBuy) and v.leg.price > 0"]);
  const message = loadMessage(flow(nodes, "${c.output}", { typedefs }));
  expect(message).toContain('flow has a cycle involving ["b","c"]');
  expect(message).not.toContain("reads field");
});

test("an impossible node assert check is located at the assert", () => {
  // The read after the impossible check is reached by no value, so it is not checked.
  const text = localAssert("isinstance(v, Buy) and v.action == 'sell' and v.zz > 0");
  const error = loadError(() => load(text));
  expect(error.message).toContain(
    "node 'c' assert: `v.action == \"sell\"` can never be true here: v is Buy, by the checks before it",
  );
  expect(error.message).not.toContain("${v.zz}");
  expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(lineOf(text, "v.action == 'sell'"));
});

test("a call node assert's tag checks narrow its local", () => {
  const nodes =
    producer("d", "Order") +
    "  c:\n    kind: call\n    call: child\n" +
    "    input:\n      v: ${d.output}\n" +
    "    asserts:\n      - \"v.action == 'buy' and v.qty > 0\"\n";
  load(flow(nodes, "${c.output}", { defs: childText("Order") }));
});

test("an assert reading many typed inputs loads", () => {
  // Equal alternatives of a typed binding count once, so the checks stay linear.
  const names = Array.from({ length: 14 }, (_, i) => `a${i}`);
  const assertion = names.map((name) => `${name}.a == 'x'`).join(" and ");
  const node = code(Object.fromEntries(names.map((name) => [name, "!Rec ${step.output}"])), [assertion]);
  load(flow(node, "${c.output}"));
});

/**
 * A flow with CODE producers typed by `producers` and node `c` asserting `assertion` on its
 * `inputs`.
 */
function localsAssert(assertion: string, producers: Record<string, string>, inputs: Record<string, string>): string {
  const nodes = Object.entries(producers)
    .map(([nodeId, typ]) => producer(nodeId, typ))
    .join("");
  return flow(nodes + code(inputs, [assertion]), "${c.output}");
}

test.each([
  [
    "no-overlap",
    "isinstance(v, Rec) and v.zz > 0",
    "`isinstance(v, Rec)` can never be true: v is declared Order, and no Order value fits Rec",
  ],
  [
    "not-a-tag",
    "v.action == 'byu' and v.zz > 0",
    '`v.action == "byu"` compares v.action with "byu", which is not one of its tags (buy, sell); ' +
      'did you mean "buy"?',
  ],
])("a node assert read after a reported check is not checked: %s", (_id, assertion, expected) => {
  const message = loadMessage(localAssert(assertion));
  expect(message).toContain(`node 'c' assert: ${expected}`);
  expect(message).not.toContain("${v.zz}");
});

test("a node assert read after an always-true negation is not checked", () => {
  const message = loadMessage(localsAssert("not isinstance(v, Buy) and v.zz > 0", { d: "Buy" }, { v: "${d.output}" }));
  expect(message).toContain(
    "node 'c' assert: `not isinstance(v, Buy)` can never be true: v is declared Buy, so " +
      "`isinstance(v, Buy)` is always true",
  );
  expect(message).not.toContain("${v.zz}");
});

test("a node assert read of one local after a reported check on another is not checked", () => {
  const message = loadMessage(
    localsAssert(
      "isinstance(v, Buy) and v.action == 'sell' and w.zz > 0",
      { d: "Order" },
      { v: "${d.output}", w: "${step.output}" },
    ),
  );
  expect(message).toContain('`v.action == "sell"` can never be true here');
  expect(message).not.toContain("${w.zz}");
});

// `v` is declared Order but bound from a Buy, so it has two alternatives: Order and Buy.
const TWO_ALTERNATIVES = { v: "!Order ${d.output}", w: "!Order ${e.output}" };

test("a node assert check impossible under every alternative is reported once", () => {
  const message = loadMessage(localsAssert("isinstance(v, Rec) or v.zz > 0", { d: "Buy", e: "Order" }, TWO_ALTERNATIVES));
  expect(count(message, "`isinstance(v, Rec)` can never be true")).toBe(1);
  expect(count(message, "reference ${v.zz} reads unknown field 'zz'")).toBe(1);
});

test("a node assert read dead under only some alternatives is checked", () => {
  // Under Order the tag check rules `v` out, but under Buy it is no check at all, so neither the
  // check is reported nor the read after it skipped.
  const message = loadMessage(
    localsAssert(
      "isinstance(v, Buy) and v.action == 'sell' and w.zz > 0",
      { d: "Buy", e: "Order" },
      TWO_ALTERNATIVES,
    ),
  );
  expect(message).toContain("node 'c' assert: reference ${w.zz} reads unknown field 'zz' on record");
  expect(message).not.toContain("can never be true");
});

test.each([
  [
    "no-overlap",
    "isinstance(output, Rec)",
    "`isinstance(output, Rec)` can never be true: output is declared Order, and no Order value fits Rec",
  ],
  [
    "always",
    "not isinstance(output, Order)",
    "`not isinstance(output, Order)` can never be true: output is declared Order, so " +
      "`isinstance(output, Order)` is always true",
  ],
  [
    "context",
    "isinstance(output, Buy) and isinstance(output, Sell)",
    "`isinstance(output, Sell)` can never be true here: output is Buy, by the checks before it",
  ],
])("a node assert impossible output check is reported: %s", (_id, assertion, expected) => {
  expect(loadMessage(outputAssert(assertion))).toContain(`node 'c' assert: ${expected}`);
});

test("a node assert output read after a reported check is not checked", () => {
  const message = loadMessage(outputAssert("isinstance(output, Rec) and output.action == 'buy'"));
  expect(message).toContain("`isinstance(output, Rec)` can never be true");
  expect(message).not.toContain("${output.action}");
});

test("a node assert over many locals with two alternatives loads", () => {
  // Each local is checked against its own alternatives, not every combination.
  const names = Array.from({ length: 12 }, (_, i) => `v${i}`);
  const assertion = names.map((x) => `(isinstance(${x}, Buy) or ${x}.action == 'sell')`).join(" and ");
  load(localsAssert(assertion, { d: "Buy" }, Object.fromEntries(names.map((x) => [x, "!Order ${d.output}"]))));
});

test("a call node assert's member-only hint compares the tag", () => {
  // `isinstance` is not allowed in a CALL node's asserts, so the hint compares the tag.
  const nodes =
    producer("d", "Order") +
    "  c:\n    kind: call\n    call: child\n" +
    "    input:\n      v: ${d.output}\n" +
    '    asserts:\n      - "v.qty > 0"\n';
  const message = loadMessage(flow(nodes, "${c.output}", { defs: childText("Order") }));
  expect(message).toContain(
    "node 'c' assert: reference ${v.qty} reads field 'qty', which only union members with action buy " +
      'have; check the member first with `v.action == "buy" and ...`',
  );
  expect(message).not.toContain("isinstance");
});

test("a call node assert's member-only hint on an untagged union names a case", () => {
  const nodes =
    producer("d", "AB") +
    "  c:\n    kind: call\n    call: child\n" +
    "    input:\n      v: ${d.output}\n" +
    '    asserts:\n      - "v.x > 0"\n';
  const text = flow(nodes, "${c.output}", { typedefs: UNION_TYPES, defs: childText("AB") });
  expect(loadMessage(text)).toContain(
    "node 'c' assert: reference ${v.x} reads field 'x', which only 1 of 2 union members have; " +
      "check the member with isinstance(d.output, <Type>) in a case first",
  );
});

/** A one-turn inline LOOP carrying `o: Order`, seeded from the flow input `o`. */
function orderLoop(assertion: string): string {
  const nodes =
    "  drive:\n    kind: loop\n    times: 1\n" +
    "    input:\n      o: ${input.o}\n" +
    `    asserts:\n      - "${assertion}"\n` +
    "    nodes:\n" +
    "      bump:\n" +
    "        kind: code\n" +
    "        input:\n          o: ${input.o}\n" +
    "        output:\n          o: Order\n" +
    '        code: "return ({o: inputs.o})"\n' +
    "    output: ${bump.output}\n";
  return flow(nodes, "${drive.output}", { inputs: "  o: !Order\n" });
}

test("a loop node assert's tag checks narrow its local (loads)", () => {
  load(orderLoop("o.action != 'buy' or o.qty > 0"));
  expect(loadMessage(orderLoop("o.qty > 0"))).toContain(
    "node 'drive' assert: reference ${o.qty} reads field 'qty', which only union members with action buy " +
      'have; check the member first with `o.action == "buy" and ...`',
  );
  const message = loadMessage(orderLoop("o.action == 'sell' and o.action == 'buy' and o.zz > 0"));
  expect(message).toContain('`o.action == "buy"` can never be true here: o is Sell');
  expect(message).not.toContain("${o.zz}");
});

test("a loop node assert's tag checks narrow its local (runs)", async () => {
  const result = await run(orderLoop("o.action != 'buy' or o.qty > 0"), { o: { action: "buy", qty: 2n } });

  expect(result.status, result.failure?.message).toBe("succeeded");
});

const PARTIAL_CHILD_TEXT =
  "defs:\n" +
  "  child:\n" +
  "    input:\n      x: !str\n" +
  "    nodes:\n" +
  "      typed:\n" +
  "        kind: code\n" +
  "        input:\n          x: ${input.x}\n" +
  "        output: str\n" +
  "        code: return 'a'\n" +
  "      untyped:\n" +
  "        kind: code\n" +
  "        input:\n          x: ${input.x}\n" +
  '        code: "return ({deep: {k: 1}})"\n' +
  "    output:\n" +
  "      a: ${typed.output}\n" +
  "      b: ${untyped.output}\n";

/**
 * A CALL to `child` (a partial record: typed `a`, opaque `b`), and a CODE node asserting
 * `assertion` on it through the local name `v`.
 */
function partialRead(assertion: string): string {
  const nodes =
    "  call_child:\n    kind: call\n    call: child\n" +
    "    input:\n      x: ${input.s}\n" +
    code({ v: "${call_child.output}" }, [assertion]);
  return flow(nodes, "${c.output}", { defs: PARTIAL_CHILD_TEXT });
}

test("partial record local root reads (loads)", () => {
  expect(loadMessage(partialRead("v.zz == 1"))).toContain(
    "node 'c' assert: reference ${v.zz} reads unknown field 'zz' on record",
  );
  load(partialRead("v.a == 'a' and v.b.deep.k == 1"));
});

test("partial record local root reads (runs)", async () => {
  expect((await run(partialRead("v.a == 'a' and v.b.deep.k == 1"))).status).toBe("succeeded");
});

// ---------- flow asserts ----------

/** A flow-level `asserts:` section. */
const asserts = (...items: string[]) => "asserts:\n" + items.map((a) => `  - "${a}"\n`).join("");

const ORDER_INPUT = "  o: !Order\n";

/** A flow with inputs `o: !Order` and `ab: !AB` asserting `assertion`. */
function flowAssert(assertion: string, nodes = "", output = "${step.output}"): string {
  return flow(nodes, output, { typedefs: UNION_TYPES, inputs: ORDER_INPUT + "  ab: !AB\n", tail: asserts(assertion) });
}

test.each([
  ["and", "isinstance(input.o, Buy) and input.o.qty > 0"],
  ["not-or", "not isinstance(input.o, Buy) or input.o.qty > 0"],
  ["tag-or", "input.o.action == 'sell' or input.o.qty > 0"],
  ["untagged", "not isinstance(input.ab, A) or input.ab.x > 0"],
  ["shorthand-check", "isinstance(o, Buy) and input.o.qty > 0"],
])("flow assert reads are narrowed by the checks before them: %s", (_id, assertion) => {
  load(flowAssert(assertion));
});

test("flow post-assert producer reads are narrowed", () => {
  load(flowAssert("isinstance(d.output, Buy) and d.output.qty > 0", producer("d", "Order"), "${d.output}"));
});

test("a flow assert read before its check is rejected", () => {
  expect(loadMessage(flowAssert("input.o.qty > 0 and isinstance(input.o, Buy)"))).toContain(
    "reference ${input.o.qty} reads field 'qty', which only union members with action buy have; " +
      "check the member first with `isinstance(input.o, <Type>) and ...`",
  );
});

test.each<[string, Record<string, unknown>, "succeeded" | "failed"]>([
  ["sell", { action: "sell" }, "succeeded"],
  ["buy-zero", { action: "buy", qty: 0n }, "failed"],
])("guarded flow assert runs: %s", async (_id, order, status) => {
  const result = await run(flowAssert("not isinstance(input.o, Buy) or input.o.qty > 0"), { o: order, ab: { x: 1n } });

  expect(result.status, result.failure?.message).toBe(status);
});

test("an impossible flow assert check is located at the assert", () => {
  const text = flowAssert("input.o.action == 'hold'");
  const error = loadError(() => load(text));
  expect(error.message).toContain(
    "assert \"input.o.action == 'hold'\": `input.o.action == \"hold\"` compares input.o.action with " +
      '"hold", which is not one of its tags (buy, sell)',
  );
  expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(lineOf(text, "input.o.action == 'hold'"));
});

/** A flow whose `case` runs `d` (typed `Order`) only for `s: go`, asserting `assertion`. */
function skippableAssert(assertion: string): string {
  const nodes =
    "  route:\n    kind: case\n    cases:\n" +
    "      - when: \"input.s == 'go'\"\n        then: d\n" +
    "    else: e\n" +
    producer("d", "Order") +
    "  e:\n    kind: code\n    input:\n      s: ${input.s}\n    code: return 1\n";
  return flow(nodes, "${step.output}", { tail: asserts(assertion) });
}

const SKIPPABLE_READS: [string, string][] = [
  ["always-on-the-declared-type", "not isinstance(d.output, Order) or d.output.action == 'sell'"],
  ["every-member-ruled-out", "not isinstance(d.output, Buy) and not isinstance(d.output, Sell)"],
];

test.each(SKIPPABLE_READS)("a flow assert on a skippable producer reads null (loads): %s", (_id, assertion) => {
  // END binds a producer it reads with `:- null`, so a skipped one reads null.
  load(skippableAssert(assertion));
});

test.each(SKIPPABLE_READS)("a flow assert on a skippable producer reads null (runs): %s", async (_id, assertion) => {
  const result = await run(skippableAssert(assertion), { s: "stop" });

  expect(result.status, result.failure?.message).toBe("succeeded");
});

test.each([
  [
    "no-overlap",
    "isinstance(d.output, Rec)",
    "`isinstance(d.output, Rec)` can never be true: d.output is declared Order, and no Order value fits Rec",
  ],
  [
    "context",
    "isinstance(d.output, Buy) and isinstance(d.output, Sell)",
    "`isinstance(d.output, Sell)` can never be true here: d.output is Buy, by the checks before it",
  ],
])("a flow assert check impossible even on null is reported: %s", (_id, assertion, expected) => {
  expect(loadMessage(skippableAssert(assertion))).toContain(expected);
});

/** Each row: id, assertion, the expected error, and the always-run one when it differs. */
const DECLARED_TYPE_READS: [string, string, string, string?][] = [
  [
    "narrowed-typo",
    "isinstance(d.output, Sell) or d.output.qtty > 0",
    "reference ${d.output.qtty} reads field 'qtty', which Buy doesn't have; " +
      "d.output is Buy here, by the checks before this read; did you mean 'qty'?",
  ],
  [
    "null-ruled-out",
    "d.output != null and not isinstance(d.output, Buy) and d.output.qty > 0",
    "reference ${d.output.qty} reads field 'qty', which Sell doesn't have; " +
      "d.output is Sell here, by the checks before this read",
  ],
  [
    "only-null-left",
    "d.output.action == 'buy' or d.output.action == 'sell' or d.output.zz > 0",
    "reference ${d.output.zz} reads unknown field 'zz' on record",
    // A producer that always runs is never null, so no null is left to walk on the declared type.
    "reference ${d.output.zz} reads field 'zz', which Sell doesn't have; d.output is Sell here",
  ],
];

test.each(
  DECLARED_TYPE_READS.flatMap(([id, assertion, expected, alwaysRun]) => [
    [`${id}, skippable`, assertion, expected, true] as const,
    [`${id}, always-run`, assertion, alwaysRun ?? expected, false] as const,
  ]),
)("a flow assert read error names the declared type: %s", (_id, assertion, expected, skippable) => {
  // A producer that may be skipped reads null, but its read errors name the declared type.
  const text = skippable
    ? skippableAssert(assertion)
    : flow(producer("d", "Order"), "${step.output}", { tail: asserts(assertion) });
  const message = loadMessage(text);
  expect(message).toContain(expected);
  expect(message).not.toContain("Optional(");
  expect(message).not.toContain("type None");
});

test("a flow assert shorthand check is reported as written", () => {
  const message = loadMessage(flowAssert("isinstance(o, Rec)"));
  expect(message).toContain("`isinstance(o, Rec)` can never be true: o is declared Order, and no Order value fits Rec");
  expect(message).not.toContain("input.o");
});

test.each([
  [
    "record",
    "input.r.b == 'x'",
    "assert \"input.r.b == 'x'\": reference ${input.r.b} reads unknown field 'b' on record",
  ],
  ["scalar", "input.s.x == 'y'", "assert \"input.s.x == 'y'\": reference ${input.s.x} reads field 'x' on type str"],
])("a flow assert reads an unknown input field: %s", (_id, assertion, expected) => {
  expect(loadMessage(flow("", "${step.output}", { tail: asserts(assertion) }))).toContain(expected);
});

test("a flow assert known input field loads and runs (loads)", () => {
  load(flow("", "${step.output}", { tail: asserts("input.r.a == 'x'") }));
});

test("a flow assert known input field loads and runs (runs)", async () => {
  const text = flow("", "${step.output}", { tail: asserts("input.r.a == 'x'") });

  expect((await run(text)).status).toBe("succeeded");
  expect((await run(text, { r: { a: "y" } })).status).toBe("failed");
});

test.each([
  ["object", "  o: !object\n", "input.o.k == 1"],
  ["post", "", "step.output.a == 'x' and input.r.a == 'x'"],
])("flow assert readable input fields load: %s", (_id, inputs, assertion) => {
  load(flow("", "${step.output}", { inputs, tail: asserts(assertion) }));
});

// ---------- builtin results ----------

const TEXT_FIELD = (name: string) => `reference \${${name}(...).f} reads field 'f' on type str, which has no fields`;

/** CODE nodes `a` and `b`, the targets of the CASE rows. */
const CASE_TARGETS = `  a: {kind: code, code: "return 'a'", output: str}
  b: {kind: code, code: "return 'b'", output: str}
`;

test("a binding reads a field on a builtin result", () => {
  const nodes =
    "  c:\n    kind: code\n    input:\n      v: ${upper(input.s).f}\n    output: str\n    code: return 'x'\n";
  expect(loadMessage(flow(nodes, "${c.output}"))).toContain(`node 'c' input 'v' from: ${TEXT_FIELD("upper")}`);
});

test("a flow output reads a field on a builtin result", () => {
  expect(loadMessage(flow("", "\n  o: ${lower(input.s).f}"))).toContain(`flow output from: ${TEXT_FIELD("lower")}`);
});

test("a prompt reads a field on a builtin result", () => {
  expect(loadMessage(flow(agent({ v: "${step.output}" }, "${upper(v.a).x}"), "${ag.output}"))).toContain(
    "node 'ag' prompt: reference ${upper(...).x} reads field 'x' on type str, which has no fields",
  );
});

test("a case when reads a field on a builtin result", () => {
  const nodes =
    "  g:\n    kind: case\n    cases:\n" +
    "      - when: \"upper(input.s).f == 'x'\"\n        then: a\n" +
    "    else: b\n" +
    CASE_TARGETS;
  expect(loadMessage(flow(nodes, "${g.output}"))).toContain(`case node 'g' cases[0].when: ${TEXT_FIELD("upper")}`);
});

test("a node assert reads a field on a builtin result", () => {
  expect(loadMessage(flow(code({ v: "${step.output}" }, ["upper(v.a).f == 'x'"]), "${c.output}"))).toContain(
    `node 'c' assert "upper(v.a).f == 'x'": ${TEXT_FIELD("upper")}`,
  );
});

test("a flow assert reads a field on a builtin result", () => {
  expect(loadMessage(flow("", "${step.output}", { tail: asserts("upper(input.s).f == 'x'") }))).toContain(
    `flow assert "upper(input.s).f == 'x'": ${TEXT_FIELD("upper")}`,
  );
});

test("a loop while reads a field on a builtin result", () => {
  const nodes =
    "  drive:\n    kind: loop\n    call: body\n" +
    "    while: ${upper(st.a).f == 'x'}\n    max: 3\n" +
    "    input:\n      st: ${step.output}\n";
  expect(loadMessage(flow(nodes, "${drive.output}", { defs: LOOP_NAMED_BODY }))).toContain(
    `loop node 'drive' while: ${TEXT_FIELD("upper")}`,
  );
});

const MAP_CHILD_TEXT =
  "defs:\n" +
  "  child:\n" +
  "    input:\n      x: !str\n" +
  "    nodes:\n" +
  "      inner:\n" +
  "        kind: code\n" +
  "        input:\n          x: ${input.x}\n" +
  "        output: str\n" +
  "        code: return 'x'\n" +
  "    output: ${inner.output}\n";

test.each([
  [
    "map-over",
    "  m:\n    kind: map\n    call: child\n    over: ${upper(input.s).f}\n    input:\n      x: ${item}\n",
    "${m.output}",
    "node 'm' map over",
  ],
  [
    "case-on",
    "  g:\n    kind: case\n    on: ${upper(input.s).f}\n    cases:\n      - when: x\n        then: a\n    else: b\n" +
      CASE_TARGETS,
    "${g.output}",
    "node 'g' on",
  ],
  [
    "inline-call-arg",
    "  c:\n    kind: code\n    input:\n      v: call(child, x=${upper(input.s).f})\n" +
      "    output: str\n    code: return 'x'\n",
    "${c.output}",
    "node 'c' input 'v' call(child) argument 'x' from",
  ],
])("scanned sites read a field on a builtin result once: %s", (_id, nodes, output, site) => {
  const message = loadMessage(flow(nodes, output, { defs: MAP_CHILD_TEXT }));
  expect(message).toContain(site);
  expect(count(message, TEXT_FIELD("upper"))).toBe(1);
});

test.each([
  ["upper", agent({ v: "${upper(input.s)}" }, "${v}"), "${ag.output}"],
  ["join", agent({ v: "!list(str) [a, b]" }, "${join(v, ',')}"), "${ag.output}"],
  ["assert", code({ v: "${input.s}" }, ["upper(v) == 'X'"]), "${c.output}"],
])("builtin results without a field read load: %s", (_id, nodes, output) => {
  load(flow(nodes, output));
});

test("host builtin field reads load", () => {
  const saved = ["mkrec", "upper"].map((name) => [name, TEMPLATE_FNS.get(name)] as const);
  try {
    for (const [name] of saved) registerTemplateFn(name, { params: [{ name: "s" }], call: (s) => ({ f: s }) });
    for (const [name] of saved) {
      const nodes =
        "  c:\n    kind: code\n" +
        `    input:\n      v: \${${name}(input.s).f}\n` +
        "    output: str\n    code: return 'x'\n" +
        `    asserts:\n      - "${name}(v).f == v"\n`;
      load(flow(nodes, "${c.output}"));
    }
  } finally {
    for (const [name, fn] of saved) {
      if (fn === undefined) TEMPLATE_FNS.delete(name);
      else TEMPLATE_FNS.set(name, fn);
    }
  }
});

test("an unparsable condition keeps its located parse error", () => {
  expect(loadMessage(flow(code({ v: "${step.output}" }, ["upper(v.a).f =="]), "${c.output}"))).toContain(
    "node 'c' assert 'upper(v.a).f ==': not a valid boolean expression",
  );
});

// ---------- harvested conformance cases ----------

const CASES = "load-core/compose_field_read_sites__";

test.each<CaseRow>([
  ...[
    "adaptive_questions_prompt_reads_unknown_field",
    "agent_prompt_reads_unknown_field",
    "flow_output_reads_a_field_on_a_builtin_result",
    "human_input_question_reads_unknown_field",
    "transformed_union_binding_hint_names_no_bound_source",
    "union_member_field_read_outside_a_branch",
  ].map((name) => ({ id: `${CASES}${name}`, call: assembled })),
  { id: `${CASES}call_node_assert_reads_a_field_its_binding_lacks`, call: assembled },
  { id: `${CASES}flow_assert_compares_with_a_non_tag`, call: assembled },
  { id: `${CASES}flow_assert_read_before_its_check`, call: assembled },
  { id: `${CASES}flow_assert_read_on_the_other_member`, call: assembled },
  { id: `${CASES}flow_assert_read_typo_suggests_a_field`, call: assembled },
  { id: `${CASES}flow_assert_readable_input_fields_load`, call: assembled },
  { id: `${CASES}flow_assert_reads_unknown_input_field`, call: assembled },
  { id: `${CASES}flow_assert_shorthand_check_is_reported_as_written`, call: assembled },
  { id: `${CASES}impossible_flow_assert_check`, call: assembled },
  { id: `${CASES}impossible_node_assert_check`, call: assembled },
  { id: `${CASES}loop_assert_reads_an_unknown_carried_field`, call: assembled },
  { id: `${CASES}member_only_hint_on_an_untagged_union`, call: assembled },
  { id: `${CASES}node_assert_check_no_value_fits`, call: assembled },
  { id: `${CASES}node_assert_compares_with_a_non_tag`, call: assembled },
  { id: `${CASES}node_assert_negated_check_always_true`, call: assembled },
  { id: `${CASES}node_assert_open_input_field_reads_load`, call: assembled },
  { id: `${CASES}node_assert_read_before_its_check`, call: assembled },
  { id: `${CASES}node_assert_read_on_the_other_member`, call: assembled },
  { id: `${CASES}node_assert_reads_unknown_input_field`, call: assembled },
  { id: `${CASES}prompt_inside_a_def_reads_unknown_field`, call: assembled, full: true },
  { id: `${CASES}loop_while_reads_a_field_on_a_builtin_result`, call: assembled, full: true },
])("assemble: $id", runCase);

// --------------------------------------------------------------------------- //
// Nullable field reads (`expr.nullable_field_read`): a `.` step on a value that may be null is a
// load error at every site, and each narrowing form lifts it.
// --------------------------------------------------------------------------- //

const NULLABLE_PREAMBLE =
  "id: nf\nname: nf\n" +
  "typedefs:\n" +
  "  Rec: {a: str}\n" +
  "  RecK: {kind: 'Literal(rec)', a: str}\n" +
  "  OtherK: {kind: 'Literal(other)', a: str, b: str}\n" +
  "  Tagged: Union(RecK, OtherK)\n" +
  "  T: {f: Optional(str), g: str}\n" +
  "input:\n  maybe: !Optional(Rec)\n  tagged: !Optional(Tagged)\n  rec: !Rec\n";

/** The narrowing forms of design rule 3, each a condition over subject `S` that reads `S.a`. */
const FORMS: readonly [name: string, form: (s: string) => string][] = [
  ["!= null and", (s) => `${s} != null and ${s}.a == 'x'`],
  ["== null or", (s) => `${s} == null or ${s}.a == 'x'`],
  ["isinstance", (s) => `isinstance(${s}, Rec) and ${s}.a == 'x'`],
];
/**
 * The tag-test form, over a subject that is an `Optional` tagged union; both members have `a`, so
 * only the null matters.
 */
const TAG_FORM = (s: string) => `${s}?.kind == 'rec' and ${s}.a == 'x'`;
/** The read the forms guard, unguarded: a load error at every site. */
const UNGUARDED = (s: string) => `${s}.a == 'x'`;

/** A flow whose CODE node `use` binds `v: ${<text>}`. */
const nullableBinding = (text: string) =>
  NULLABLE_PREAMBLE +
  "nodes:\n" +
  `  use:\n    kind: code\n    input:\n      v: \${${text}}\n    code: return 1\n` +
  "output: ${use.output}\n";

/** A flow where a CASE routes `use` when `input.maybe` is set, and `use` binds `v: ${<text>}`. */
const routedBinding = (text: string) =>
  NULLABLE_PREAMBLE +
  "nodes:\n" +
  "  gate:\n    kind: case\n    cases:\n      - when: ${input.maybe != null}\n        then: use\n    else: skip\n" +
  `  use:\n    kind: code\n    output: str\n    input:\n      v: \${${text}}\n    code: return 'u'\n` +
  "  skip:\n    kind: code\n    output: str\n    code: return 'n'\n" +
  "output: ${use.output | skip.output}\n";

/**
 * A flow where an untyped `mk` (`output: Any`) is routed by `isinstance(mk.output, T)` to `go`, and
 * the `binding` sits on `reader`: `go`, or `other`, the `else:`.
 */
const typeBranch = (binding: string, reader: "go" | "other" = "go") => {
  const node = (id: string) =>
    `  ${id}:\n    kind: code\n    output: str\n` +
    (id === reader ? `    input:\n      v: ${binding}\n` : "") +
    `    code: return '${id}'\n`;
  return (
    NULLABLE_PREAMBLE +
    "nodes:\n" +
    "  mk:\n    kind: code\n    output: Any\n    code: \"return {'g': 'x'}\"\n" +
    "  route:\n    kind: case\n    cases:\n      - when: isinstance(mk.output, T)\n        then: go\n    else: other\n" +
    node("go") +
    node("other") +
    "output: ${go.output | other.output}\n"
  );
};

test.each(FORMS)("a binding narrowed by %s loads", (_name, form) => {
  load(nullableBinding(form("input.maybe")));
});

test("a binding narrowed by a tag test loads", () => {
  load(nullableBinding(TAG_FORM("input.tagged")));
});

test("an unguarded Optional read in a binding is a load error", () => {
  const error = loadError(() => load(nullableBinding(UNGUARDED("input.maybe"))));
  expect(error.diagnostic.code).toBe("expr.nullable_field_read");
  expect(error.message).toContain("reads field 'a' of input.maybe, which may be null (it is Optional)");
});

test("a CASE branch that proves the value lets a routed binding read it with .", () => {
  load(routedBinding("input.maybe.a"));
});

test("an untyped producer narrowed by a CASE type branch needs ?. for an optional field", () => {
  const message = loadError(() => load(typeBranch("${mk.output.f}"))).message;
  expect(message).toContain("which is untyped and only narrowed to T; write mk.output?.f");
  load(typeBranch("${mk.output?.f}"));
  load(typeBranch("${mk.output.g}"));
});

test("an untyped producer narrowed by isinstance in a binding needs ?. for an optional field", () => {
  expect(
    loadError(() => load(typeBranch("${isinstance(mk.output, T) and mk.output.f == 'a'}", "other"))).message,
  ).toContain("only narrowed to T; write mk.output?.f");
});

/** A flow whose output is `${<text>}`. */
const nullableOutput = (text: string) =>
  NULLABLE_PREAMBLE + "nodes:\n" + STEP.replace("${input.s}", "${input.rec}") + `output: \${${text}}\n`;

test.each(FORMS)("a flow output narrowed by %s loads", (_name, form) => {
  load(nullableOutput(form("input.maybe")));
});

test("an unguarded Optional read in a flow output is a load error", () => {
  const error = loadError(() => load(nullableOutput(UNGUARDED("input.maybe"))));
  expect(error.diagnostic.code).toBe("expr.nullable_field_read");
  expect(error.message).toContain("output: ${input.maybe.a} reads field 'a' of input.maybe, which may be null");
});

/**
 * A flow where `gate` may skip `ms` (`output: Rec`), and AGENT `ag` binds `r: <source>` and reads it
 * in `prompt` (or in `extra` lines); `routed` puts `ag` behind a CASE that proves `input.maybe`.
 */
function promptFlow(source: string, prompt: string, options: { extra?: string; routed?: boolean } = {}): string {
  const { extra = "", routed = false } = options;
  const ag =
    `  ag:\n    kind: agent\n    input:\n      r: ${source}\n` + (prompt !== "" ? `    prompt: "${prompt}"\n` : "") + extra;
  const route = routed
    ? "  route:\n    kind: case\n    cases:\n      - when: ${input.maybe != null}\n        then: ag\n    else: skip\n" +
      "  skip:\n    kind: code\n    output: str\n    code: return 'n'\n"
    : "";
  return (
    NULLABLE_PREAMBLE +
    "nodes:\n" +
    "  gate:\n    kind: case\n    cases:\n      - when: ${input.maybe != null}\n        then: ms\n" +
    "  ms:\n    kind: code\n    output: Rec\n    code: \"return {'a': 'x'}\"\n" +
    route +
    ag +
    `output: ${routed ? "${ag.output | skip.output}" : "${ag.output}"}\n`
  );
}

const SKIPPED = "${ms.output :- null}";

test.each([
  ["a skippable producer", SKIPPED, "(node 'ms' may be skipped while 'ag' runs)"],
  ["an Optional input", "${input.maybe}", "(it is Optional)"],
])("an unguarded prompt read of %s is a load error", (_name, source, why) => {
  const error = loadError(() => load(promptFlow(source, "Say ${r.a}")));
  expect(error.diagnostic.code).toBe("expr.nullable_field_read");
  expect(error.message).toBe(
    `node 'ag' prompt: \${r.a} reads field 'a' of r, which may be null ${why}; write r?.a, or check r != null first`,
  );
});

test("a prompt read through ?. with a default loads", () => {
  load(promptFlow(SKIPPED, "Say ${r?.a :- 'x'}"));
});

test("a system prompt and a prompt-list item check nullable reads", () => {
  expect(loadError(() => load(promptFlow(SKIPPED, "hi", { extra: '    system_prompt: "${r.a}"\n' }))).message).toContain(
    "node 'ag' system_prompt: ${r.a} reads field 'a' of r, which may be null",
  );
  expect(loadError(() => load(promptFlow(SKIPPED, "", { extra: '    prompt:\n      - user: "${r.a}"\n' }))).message).toContain(
    "node 'ag' prompt[0]: ${r.a} reads field 'a' of r, which may be null",
  );
});

test("a prompt in a node a CASE routes when the value is set reads it with .", () => {
  load(promptFlow("${input.maybe}", "Say ${r.a}", { routed: true }));
});

test("a prompt read of a local that is never null loads", () => {
  load(promptFlow("${input.rec}", "Say ${r.a}"));
});

// `isinstance` is not a prompt function.
test.each(FORMS.filter(([name]) => name !== "isinstance"))("a prompt narrowed by %s loads", (_name, form) => {
  load(promptFlow("${input.maybe}", `\${${form("r")}}`));
});

/** An AGENT whose steer inbox is `Inbox(Ev)`, with `Ev: {o: Optional(Inner)}`, rendering `render`. */
const steerRender = (render: string) =>
  "id: steer\nname: steer\n" +
  "typedefs:\n  Inner: {x: str}\n  Ev: {o: Optional(Inner)}\n" +
  "input:\n  notes: !Inbox(Ev)\n" +
  "nodes:\n  chat:\n    kind: agent\n    prompt: Help the user.\n" +
  `    steer_inbox:\n      from: \${input.notes}\n      render: |-\n        ${render}\n` +
  "    output: str\n" +
  "output: ${chat.output}\n";

test("a render reading a field through an Optional field of event is a load error", () => {
  const error = loadError(() => load(steerRender("${event.o.x}")));
  expect(error.diagnostic.code).toBe("expr.nullable_field_read");
  expect(error.message).toBe(
    "node 'chat' steer_inbox render: ${event.o.x} reads field 'x' of event.o, which may be null " +
      "(field 'o' is Optional); write event.o?.x, or check event.o != null first",
  );
});

test.each(["${event.o?.x :- ''}", "${event.o != null and event.o.x == 'a'}"])("a guarded render loads: %s", (render) => {
  load(steerRender(render));
});
