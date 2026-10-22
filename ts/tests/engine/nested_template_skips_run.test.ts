/**
 * Skips through strings nested in a value template, end to end: a port of
 * `tests/engine/test_nested_template_skips_run.py`.
 *
 * Each nested `${...}` string behaves as its own input: a whole-value read whose producers are all
 * skipped skips the reading node, an interpolation of a skipped node renders empty text and the
 * node runs, a flow output gets null at the nested position, and these groups survive runtime
 * growth.
 *
 * Ported: every row. The CODE bodies are Python's, which read the same in JS (decision 11), but
 * for the slow producer, which awaits a 50 ms timer where Python sleeps. Ints read as `bigint`s
 * (decision 4a). The `run-case/engine_nested_template_skips_run__*` cases pin one run of
 * `test_nested_whole_value_read_of_a_skipped_node_skips_the_reader`,
 * `test_nested_interpolation_of_a_skipped_node_renders_empty_text`,
 * `test_one_skipped_nested_string_skips_the_reader_even_beside_a_live_one`,
 * `test_flow_output_gets_null_at_a_skipped_nested_position` and
 * `test_nested_skips_hold_inside_a_called_child` each, where the rows make both.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { cloneChild } from "../../src/compile/expand.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { NodeSkipped } from "../../src/events.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-nested-template-skips-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const GATED_NODES = `
  gate:
    kind: case
    cases:
      - when: \${input.approve}
        then: order
    else: rejected
  order:
    kind: code
    output: str
    code: |
      return "filled"
  rejected:
    kind: code
    output: str
    code: |
      return "no"
`;

/** Return a flow gated on `input.approve` with `extraNodes` and `output`. */
const flow = (extraNodes: string, output: string) =>
  "id: nested_skips\nname: nested_skips\ninput:\n  approve: !bool\nnodes:" + GATED_NODES + extraNodes + `output:${output}\n`;

/** Return a CODE node that returns its input `x`, bound as `binding`. */
const code = (nodeId: string, binding: string) =>
  `  ${nodeId}:\n    kind: code\n    input:\n      x: ${binding}\n    code: |\n      return inputs["x"]\n`;

/** Run `text` with `approve`; the run must succeed. */
async function run(text: string, approve: boolean): Promise<RunResult> {
  const result = await runFlow(await loadFlow(text, { env }), { approve }, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result;
}

/** Return the ids of the nodes the run skipped. */
const skipped = (result: RunResult) =>
  new Set(result.events.filter((event) => event instanceof NodeSkipped).map((event) => (event as NodeSkipped).nodeId));

test("nested whole value read of a skipped node skips the reader", async () => {
  const text = flow(code("c", "{a: '${order.output}', b: 1}"), "\n  c: ${c.output}\n  r: ${rejected.output}");

  const skips = await run(text, false);
  expect(skipped(skips).has("c")).toBe(true);
  expect(skips.output).toEqual({ c: null, r: "no" });

  const taken = await run(text, true);
  expect(skipped(taken).has("c")).toBe(false);
  expect(taken.output).toEqual({ c: { a: "filled", b: 1n }, r: null });
});

test("nested interpolation of a skipped node renders empty text", async () => {
  const text = flow(code("c", "{a: 'got ${order.output}!'}"), " ${c.output}");

  const skips = await run(text, false);
  expect(skipped(skips).has("c")).toBe(false);
  expect(skips.output).toEqual({ a: "got !" });

  expect((await run(text, true)).output).toEqual({ a: "got filled!" });
});

test("one skipped nested string skips the reader even beside a live one", async () => {
  const text = flow(code("c", "{a: '${rejected.output}', b: ['${order.output}']}"), "\n  c: ${c.output}");

  expect(skipped(await run(text, false)).has("c")).toBe(true);
  expect(skipped(await run(text, true)).has("c")).toBe(true);
});

test("nested coalesce over both branches runs the reader", async () => {
  const text = flow(code("c", "{a: '${order.output | rejected.output}'}"), " ${c.output}");

  expect((await run(text, false)).output).toEqual({ a: "no" });
  expect((await run(text, true)).output).toEqual({ a: "filled" });
});

test("flow output gets null at a skipped nested position", async () => {
  const text = flow("", "\n  r: {a: {b: '${order.output}'}, n: 1}");

  expect((await run(text, false)).output).toEqual({ r: { a: { b: null }, n: 1n } });
  expect((await run(text, true)).output).toEqual({ r: { a: { b: "filled" }, n: 1n } });
});

test("value mode output gets null at a skipped nested position", async () => {
  const text = flow("", " {a: {b: '${order.output}'}}");

  expect((await run(text, false)).output).toEqual({ a: { b: null } });
});

test("consumer waits for a nested producer", async () => {
  const text =
    "id: wait\nname: wait\nnodes:\n" +
    "  slow:\n    kind: code\n    code: |\n" +
    "      await new Promise((resolve) => setTimeout(resolve, 50))\n      return 7\n" +
    code("c", "{deep: [{v: '${slow.output}'}]}") +
    "output: ${c.output}\n";
  const compiled = (await loadFlow(text, { env })).compiled;

  expect(compiled.edges.filter((edge) => edge.to === "c").map((edge) => [edge.from, edge.inputGroup])).toEqual([
    ["slow", "x/deep/0/v"],
  ]);
  const result = await runFlow(await loadFlow(text, { env }), {}, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ deep: [{ v: 7n }] });
});

const CHILD = `
id: child
name: child
input:
  approve: !bool
nodes:${GATED_NODES}  c:
    kind: code
    input:
      x: {a: '\${order.output}', t: 'got \${order.output}'}
    code: |
      return inputs["x"]
output:
  c: \${c.output}
  r: {b: '\${order.output}'}
`;

test("cloned child keeps the nested groups", async () => {
  const child = (await loadFlow(CHILD, { env })).compiled;

  const clone = cloneChild(child, "k", { approve: true });

  const byGroup = clone.edges
    .filter((edge) => edge.from === "k/order" && edge.inputGroup !== null)
    .map((edge) => [edge.to, edge.inputGroup, edge.optional])
    .sort((a, b) => (JSON.stringify(a) < JSON.stringify(b) ? -1 : 1));
  expect(byGroup).toEqual([
    ["k/__end__", "r/b", true],
    ["k/c", "x/a", false],
    ["k/c", "x/t", true],
  ]);
});

const PARENT = `
id: parent
name: parent
input:
  approve: !bool
defs:
  child:
    input:
      approve: !bool
    nodes:
      gate:
        kind: case
        cases:
          - when: \${input.approve}
            then: order
        else: rejected
      order:
        kind: code
        output: str
        code: |
          return "filled"
      rejected:
        kind: code
        output: str
        code: |
          return "no"
      c:
        kind: code
        input:
          x: {a: '\${order.output}', t: 'got \${order.output}'}
        code: |
          return inputs["x"]
    output:
      c: \${c.output}
      r: {b: '\${order.output}'}
nodes:
  k:
    kind: call
    call: child
    input:
      approve: \${input.approve}
output: \${k.output}
`;

const CALLED_CHILD_ROWS: [boolean, unknown, boolean][] = [
  [false, { c: null, r: { b: null } }, true],
  [true, { c: { a: "filled", t: "got filled" }, r: { b: "filled" } }, false],
];

test.each(CALLED_CHILD_ROWS)("nested skips hold inside a called child (approve %p)", async (approve, expected, cSkipped) => {
  const result = await run(PARENT, approve);

  expect(result.output).toEqual(expected);
  expect(skipped(result).has("k/c")).toBe(cSkipped);
});
