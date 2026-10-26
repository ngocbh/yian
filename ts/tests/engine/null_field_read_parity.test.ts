/**
 * Null field reads agree across the ways a node binds (#1149, "Traced = untraced"): untraced,
 * traced (learnable), as a derived node (a CALL child's END), and on a resume that replays the
 * log (which rebuilds derived values and, when learnable, rebinds every traced capture).
 *
 * TS-only: Python yian keeps the silent miss.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { advance, type RunResult, runFlow } from "../../src/run.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-null-parity-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path under the test's folder. */
const freshLog = () => join(dir, `log-${++logs}.jsonl`);

const TYPES = `typedefs:
  Rec: {k: str, opt: Optional(str)}
`;

/** CODE node `id` returning `value`, typed `output` when given. */
function code(id: string, value: string, output: string | null = null, input = ""): string {
  const typed = output === null ? "" : `    output: ${output}\n`;
  return `  ${id}:\n    kind: code\n${typed}${input}    code: |\n      return ${value}\n`;
}

/** A flow whose `use` node binds `v` from `source` over `nodes`, returning it. */
function live(nodes: string, source: string, inputs = ""): string {
  const use = code("use", 'inputs["v"]', null, `    input:\n      v: ${source}\n`);
  return `id: f\nname: f\n${TYPES}${inputs}nodes:\n${nodes}${use}output: \${use.output}\n`;
}

/** The same read as a derived node: the output of a CALL child holding `nodes`. */
function derived(nodes: string, source: string): string {
  const child = nodes.replace(/^ {2}/gm, "      ");
  return (
    `id: f\nname: f\n${TYPES}defs:\n  child:\n    nodes:\n${child}    output: ${source}\n` +
    "nodes:\n  c: {kind: call, call: child}\noutput: ${c.output}\n"
  );
}

const A_REC = code("a", '{k: "x"}', "Rec");
const A_UNTYPED = code("a", '{k: "x"}');
const A_LIST = code("a", '[{k: "x"}, {k: "y"}]', "list(Rec)");

/** Flows whose reads cover the step table, each with its expected output or failure code. */
const CASES: { name: string; flow: string; inputs?: Record<string, unknown>; output?: unknown; code?: string }[] = [
  { name: "a declared absent field reads null", flow: live(A_REC, '${a.output.opt :- "d"}'), output: "d" },
  { name: "a declared absent field, derived", flow: derived(A_REC, '${a.output.opt :- "d"}'), output: "d" },
  { name: "?. on an untyped record reads null", flow: live(A_UNTYPED, '${a.output?.opt :- "d"}'), output: "d" },
  { name: "?. on an untyped record, derived", flow: derived(A_UNTYPED, '${a.output?.opt :- "d"}'), output: "d" },
  { name: "an untyped absent key raises", flow: live(A_UNTYPED, "${a.output.opt}"), code: "binding.evaluate" },
  {
    name: "an untyped absent key raises, derived",
    flow: derived(A_UNTYPED, "${a.output.opt}"),
    code: "binding.evaluate",
  },
  {
    name: "a comprehension variable reads its element's declared field",
    flow: live(A_LIST, '${[m.opt :- "d" for m in a.output]}'),
    output: ["d", "d"],
  },
  {
    name: "a comprehension variable, derived",
    flow: derived(A_LIST, '${[m.opt :- "d" for m in a.output]}'),
    output: ["d", "d"],
  },
  {
    name: "a MAP item reads its element's declared field",
    flow:
      `id: f\nname: f\n${TYPES}nodes:\n${A_LIST}` +
      "  each:\n    kind: map\n    over: ${a.output}\n    input:\n      v: !str ${item.opt :- \"d\"}\n" +
      "    nodes:\n      one:\n        kind: code\n        input:\n          v: ${input.v}\n" +
      '        code: return inputs["v"]\n' +
      "    output: ${one.output}\noutput: ${each.output}\n",
    output: ["d", "d"],
  },
  {
    name: "a flow assert reads an omitted optional input as null",
    flow:
      "id: f\nname: f\ninput:\n  note: !Optional(str)\nasserts:\n  - input.note == null\nnodes: {}\n" +
      'output: ${input.note :- "none"}\n',
    output: "none",
  },
];

/** Expect `result` to match `c`'s output or failure code. */
function expectResult(result: RunResult, c: (typeof CASES)[number]): void {
  if (c.code !== undefined) {
    expect(result.status).toBe("failed");
    expect(result.failure?.code).toBe(c.code as never);
  } else {
    expect(result.failure?.message).toBeUndefined();
    expect(result.output).toEqual(c.output);
  }
}

for (const c of CASES) {
  test.each([
    ["untraced", false],
    ["traced", true],
  ] as const)(`${c.name}: %s, and its resume`, async (_mode, learnable) => {
    const log = freshLog();
    const result = await runFlow(await loadFlow(c.flow), c.inputs ?? {}, { learnable, log });
    expectResult(result, c);
    // Replaying the log rebuilds every derived value and, when learnable, rebinds each capture.
    const resumed = await advance({ log });
    expect(resumed.status).toBe(result.status);
    expect(resumed.output).toEqual(result.output);
    expect(resumed.failure?.code).toEqual(result.failure?.code);
  });
}

/** Flows whose failing read sits behind a rewritten root, with the message naming the authored one. */
const MESSAGES: { name: string; flow: string; inputs?: Record<string, unknown>; message: string }[] = [
  {
    name: "a flow assert names its producer, not END's local",
    flow: `id: f\nname: f\nnodes:\n${code("a", "null")}asserts:\n  - a.output.k == 1\noutput: \${a.output}\n`,
    message:
      "node '__end__' post-assert 'a.output.k == 1' (`a.output.k == 1`) could not be evaluated: cannot read " +
      "${a.output.k}: a.output is null; write a.output?.k to read null instead, or check a.output != null first",
  },
  {
    name: "a flow assert names a flow input, not END's local",
    flow:
      `id: f\nname: f\ninput:\n  r: !object\nnodes:\n${code("a", "1")}` +
      "asserts:\n  - a.output == 1 and input.r.k == 1\noutput: ${a.output}\n",
    inputs: { r: {} },
    message:
      "node '__end__' post-assert 'a.output == 1 and input.r.k == 1' (`a.output == 1 and input.r.k == 1`) " +
      "could not be evaluated: cannot read ${input.r.k}: input.r has no field 'k'",
  },
  {
    name: "a CALL child's node binding names the child's input",
    flow:
      "id: f\nname: f\ndefs:\n  child:\n    input:\n      r: !object\n    nodes:\n" +
      code("u", 'inputs["v"]', null, "    input:\n      v: ${input.r.nosuch}\n").replace(/^ {2}/gm, "      ") +
      "    output: ${u.output}\nnodes:\n  c: {kind: call, call: child, input: {r: {}}}\noutput: ${c.output}\n",
    message: "cannot read ${input.r.nosuch}: input.r has no field 'nosuch'",
  },
  {
    name: "a CALL child's output names the child's input",
    flow:
      "id: f\nname: f\ndefs:\n  child:\n    input:\n      r: !object\n    nodes: {}\n" +
      "    output: ${input.r.nosuch}\nnodes:\n  c: {kind: call, call: child, input: {r: {}}}\noutput: ${c.output}\n",
    message: "cannot read ${input.r.nosuch}: input.r has no field 'nosuch'",
  },
];

for (const c of MESSAGES) {
  test(`${c.name}`, async () => {
    const result = await runFlow(await loadFlow(c.flow), c.inputs ?? {}, { log: freshLog() });
    expect(result.status).toBe("failed");
    expect(result.failure?.message).toBe(c.message);
  });
}
