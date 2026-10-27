/**
 * Each string nested in a value template is its own input group.
 *
 * Every leaf string of a `ValueTemplate` binding behaves as if it were its own input: its data
 * edges carry their own input group (the parameter name followed by the string's JSON-pointer path)
 * and their own co-skip stance, a flow output gets null at a nested position whose producers may
 * all be skipped, and arithmetic on a nested value that may be skipped is rejected. A plain
 * top-level string keeps the parameter name as its group.
 */

import { expect, test } from "bun:test";

import { type CompiledFlow, END_ID } from "../../src/compile/model.ts";
import { ValueTemplate } from "../../src/nodes/contracts.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "./declarations.ts";

const HEADER = `
id: nested_skips
name: nested_skips
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
`;

/** Return a gated flow whose CODE node `c` binds `x` as `consumerInput`. */
function flow(consumerInput: string, output = "${c.output}"): string {
  return `${HEADER}  c:
    kind: code
    input:
      x: ${consumerInput}
    code: |
      return inputs["x"]
output: ${output}
`;
}

/** Return `[from, inputGroup, optional]` for each data edge into `nodeId`. */
function edgesInto(compiled: CompiledFlow, nodeId: string): [string, string | null, boolean][] {
  return compiled.edges
    .filter((edge) => edge.to === nodeId && edge.sourceHandle === null && !edge.ordering && edge.inputGroup !== null)
    .map((edge) => [edge.from, edge.inputGroup, edge.optional]);
}

const compiled = (text: string) => assembleText(text).flow.compiled;

test("each nested string has its own group and stance", () => {
  const flowCompiled = compiled(flow("{a: '${order.output}', b: ['note ${rejected.output}']}"));

  expect(edgesInto(flowCompiled, "c")).toEqual([
    ["order", "x/a", false],
    ["rejected", "x/b/0", true],
  ]);
});

test("nested group names escape pointer characters", () => {
  expect(edgesInto(compiled(flow("{'a/b~c': '${order.output}'}")), "c")).toEqual([["order", "x/a~1b~0c", false]]);
});

test("a plain string keeps the parameter as its group", () => {
  const flowCompiled = compiled(flow("${order.output}"));

  expect(edgesInto(flowCompiled, "c")).toEqual([["order", "x", false]]);
  expect(edgesInto(flowCompiled, END_ID)).toEqual([["c", "result", true]]);
});

test("a flow output has a group per nested string", () => {
  const flowCompiled = compiled(flow("${order.output}", "\n  r: {a: {b: '${order.output}'}, n: ['${rejected.output}']}"));

  expect(edgesInto(flowCompiled, END_ID)).toEqual([
    ["order", "r/a/b", true],
    ["rejected", "r/n/0", true],
  ]);
});

test("a skipped nested output string defaults to null", () => {
  const flowCompiled = compiled(
    flow(
      "${order.output}",
      "\n  r: {a: {b: '${order.output}'}, t: 'got ${order.output}', s: '${order.output | rejected.output}'}",
    ),
  );

  const source = flowCompiled.outputs.find((output) => output.name === "r")!.binding.source;
  expect(source).toBeInstanceOf(ValueTemplate);
  const template = source as ValueTemplate;
  // Only the co-skipping string whose producers may all skip gets the default.
  expect(template.value).toEqual({
    a: { b: "${order.output:-null}" },
    t: "got ${order.output}",
    s: "${order.output | rejected.output}",
  });
  expect(template.origins.map(([path]) => path)).toEqual([["a", "b"], ["t"], ["s"]]);
});

test("nested arithmetic on a skipped value is rejected at its line", () => {
  const text = flow("\n        a: ok\n        b: ${order.output + 1}");

  const error = loadError(() => compiled(text));

  expect(error.diagnostic.message).toBe(
    "node 'c' input 'x': ${order.output + 1} does arithmetic on order.output, which is null when node 'order' is " +
      "skipped; give it a default with :-, e.g. (order.output :- <default>)",
  );
  const line = text.split("\n").indexOf("        b: ${order.output + 1}") + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

test("nested output arithmetic on a skipped value is rejected", () => {
  const text = flow("${order.output}", "\n  r: {a: '${order.output + 1}'}");

  expect(loadError(() => compiled(text)).message).toBe(
    "output 'r': ${order.output + 1} does arithmetic on order.output, which is null when node 'order' is " +
      "skipped; give it a default with :-, e.g. (order.output :- <default>)",
  );
});

/** Return a flow whose CODE node `d` sums `a`, `b` and `c`, with `then`/`otherwise` gated. */
function chainFlow(then: string, otherwise: string): string {
  const producers = ["a", "b", "c", "e"]
    .map((node) => `  ${node}:\n    kind: code\n    output: int\n    code: |\n      return 1\n`)
    .join("");
  return `
id: chain_skips
name: chain_skips
input:
  approve: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: \${input.approve}
        then: ${then}
    else: ${otherwise}
${producers}  d:
    kind: code
    input:
      x: \${a.output + b.output + c.output}
    code: |
      return inputs["x"]
output: \${d.output}
`;
}

test("chain arithmetic names its possibly skipped operand", () => {
  expect(loadError(() => compiled(chainFlow("c", "e"))).diagnostic.message).toBe(
    "node 'd' input 'x': ${a.output + b.output + c.output} does arithmetic on c.output, which is null when node " +
      "'c' is skipped; give it a default with :-, e.g. (c.output :- <default>)",
  );
});

test("chain arithmetic names the first skipped operand in source order", () => {
  expect(loadError(() => compiled(chainFlow("a", "c"))).diagnostic.message).toBe(
    "node 'd' input 'x': ${a.output + b.output + c.output} does arithmetic on a.output, which is null when node " +
      "'a' is skipped; give it a default with :-, e.g. (a.output :- <default>)",
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-case/compose_nested_template_skips__arithmetic_in_a_nested_node_input", call: assembled },
  { id: "load-case/compose_nested_template_skips__arithmetic_in_a_nested_output", call: assembled },
  { id: "load-case/compose_nested_template_skips__chain_arithmetic_names_the_first_skipped", call: assembled },
])("assembled: $id", runCase);
