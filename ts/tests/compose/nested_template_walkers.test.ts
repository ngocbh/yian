/**
 * Load-time walkers see the strings inside a nested binding value: a port of
 * `tests/compose/test_nested_template_walkers.py`, through `assembleText` where Python loads the
 * flow; the MAP row parses the flow's nodes (`nodesOf`), since the MAP parse raises its error.
 *
 * Later (T4): `test_nested_case_output_expands_in_inputs_and_outputs` (a run).
 * Later (T3-22, T4): `test_map_over_literal_list_without_spans_still_loads` and
 * `test_nested_references_in_a_map_child_run_after_growth` (loads and runs).
 * Later (T3-16 for the loop kind, T3-22, T4):
 * `test_nested_references_in_a_loop_body_run_after_growth`.
 *
 * Changed: `test_retired_inline_flow_call_in_a_nested_output_fails` assembles the flow without its
 * `defs:` entry, since the inline-call desugar raises before any callee resolves.
 */

import { expect, test } from "bun:test";

import { desugarInlineCalls } from "../../src/compose/desugar/inline_calls.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { Binding, ValueTemplate } from "../../src/nodes/contracts.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import { assembled, assembleText, type CaseRow, loadError, nodesOf, parsedNodes, runCase } from "./declarations.ts";

/** Return a flow whose CODE node `c` binds `x` as `binding`. */
function codeFlow(binding: string): string {
  return `
id: nested
name: nested
nodes:
  p:
    kind: code
    code: |
      return "hi"
  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs["x"]
output: \${c.output}
`;
}

/** Return one node's wiring binding source by parameter name. */
function sourceOf(compiled: CompiledFlow, nodeId: string, name: string): unknown {
  const binding = compiled.wiring.get(nodeId)!.get(name);
  expect(binding).toBeInstanceOf(Binding);
  return (binding as Binding).source;
}

/** Return the 1-based line number of `line` in `text`. */
function lineOf(text: string, line: string): number {
  return text.split("\n").indexOf(line) + 1;
}

test("untagged nested binding reads as a template", () => {
  const { compiled } = assembleText(codeFlow("{a: '${p.output}', b: 1}")).flow;

  const source = sourceOf(compiled, "c", "x");
  expect(source).toBeInstanceOf(ValueTemplate);
  expect((source as ValueTemplate).value).toEqual({ a: "${p.output}", b: 1n });
  expect((compiled.wiring.get("c")!.get("x") as Binding).contract).toBeNull();
});

test("a flow output field reads as a template", () => {
  const text = codeFlow("${p.output}").replace("output: ${c.output}", "output:\n  r: {a: '${c.output}'}");
  const { compiled } = assembleText(text).flow;

  expect(compiled.outputs.map((output) => output.name)).toEqual(["r"]);
  expect(compiled.outputs[0]!.binding.source).toBeInstanceOf(ValueTemplate);
});

test("dangling nested reference fails at its own line", () => {
  const text = codeFlow("\n        a: ${p.output}\n        b: ${nope.output}");

  const error = loadError(() => assembleText(text));

  expect(error.diagnostic.message).toContain(
    "node 'c' input 'x' from: reference ${nope.output} uses unknown namespace 'nope'",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, "        b: ${nope.output}"));
});

test("dangling nested output reference fails at its own line", () => {
  const text = codeFlow("${p.output}").replace(
    "output: ${c.output}",
    "output:\n  r:\n    a: ${c.output}\n    b:\n      - ${nope.output}",
  );

  const error = loadError(() => assembleText(text));

  expect(error.diagnostic.message).toContain("flow output from: reference ${nope.output} uses unknown namespace 'nope'");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, "      - ${nope.output}"));
});

test("nested reference field is checked", () => {
  const text =
    "typedefs:\n  Rec: {a: str}\n" +
    codeFlow("{a: '${p.output.zz}'}").replace("  p:\n    kind: code\n", "  p:\n    kind: code\n    output: Rec\n");

  const error = loadError(() => assembleText(text));

  expect(error.diagnostic.message).toContain("reference ${p.output.zz} reads unknown field 'zz' on record");
});

test("a nested reference orders its consumer after the producer", () => {
  const { compiled } = assembleText(codeFlow("{a: ['${p.output}']}")).flow;

  expect(compiled.edges.map((edge) => [edge.from, edge.to])).toContainEqual(["p", "c"]);
});

test("nested isinstance name resolves", () => {
  const text = "typedefs:\n  Rec: {a: str}\n" + codeFlow("{ok: '${isinstance(p.output, Rec)}'}");

  const { compiled } = assembleText(text).flow;

  expect([...compiled.nodes.get("c")!.types.keys()]).toEqual(["Rec"]);
});

test("unknown nested isinstance name fails", () => {
  const error = loadError(() => assembleText(codeFlow("{ok: '${isinstance(p.output, Nope)}'}")));

  expect(error.diagnostic.message).toContain("unknown type 'Nope'");
});

test("wait until container with a span fails at the container", () => {
  const text = `
id: w
name: w
nodes:
  a:
    kind: code
    code: return '2030-01-01'
  w:
    kind: wait
    until:
      - \${a.output}
output: \${w.output}
`;
  const error = loadError(() => assembleText(text));

  expect(error.diagnostic.message).toContain("wait node 'w' until: a value holding ${...} can't be a list or mapping");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, "      - ${a.output}"));
});

test.each(["\n      - ${a.output}\n      - 2", "\n      x: ${a.output}"])(
  "map over container with a span fails at the container: %j",
  (value) => {
    const text = `
id: m
name: m
defs:
  echo:
    input:
      v: !int
    nodes: {}
    output: \${input.v}
nodes:
  a:
    kind: code
    code: return 1
  m:
    kind: map
    call: echo
    over: ${value}
    input:
      v: \${item}
output: \${m.output}
`;
    const error = loadError(() => nodesOf(text));

    expect(error.diagnostic.message).toBe(
      "map node 'm' over: a value holding ${...} can't be a list or mapping; write " +
        "the list as an expression, e.g. `${[a.output, b.output]}`",
    );
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, "    over: ") + 1);
  },
);

const CASE_FLOW = `
id: nested_case
name: nested_case
input:
  go: !bool
nodes:
  route:
    kind: case
    cases:
      - when: \${input.go}
        then: yes_branch
    else: no_branch
  yes_branch:
    kind: code
    code: return "yes"
  no_branch:
    kind: code
    code: return "no"
  c:
    kind: code
    input:
      x:
        picked: \${route.output}
        note: took \${route.output}
    code: return inputs["x"]
output:
  c: \${c.output}
  direct:
    - \${route.output}
`;

test("nested case output in a default fails", () => {
  const text = CASE_FLOW.replace("note: took ${route.output}", "note: ${yes_branch.output :- route.output}");

  const error = loadError(() => assembleText(text));

  expect(error.diagnostic.message).toBe(
    "node 'c': ${route.output} — a case value can't be read in a coalesce operand, a default, an " +
      "expression, a condition, or an assert; read it as a whole binding or inside text",
  );
});

test("retired inline flow call in a nested output fails", () => {
  const text = codeFlow("${p.output}").replace("output: ${c.output}", "output:\n  r:\n    a: ${ sub(x=c.output) }");

  const error = loadError(() => assembleText(text));

  expect(error.diagnostic.message).toContain("a flow call 'sub' inside ${...} is not supported");
});

test("nested call directive is not lifted", () => {
  const template = new ValueTemplate({ a: "call(sub, x=1)" });
  const descriptors = new Map([["c", new CodeDescriptor({ id: "c", code: "return 1", inputs: { x: template } })]]);

  expect(() => desugarInlineCalls(descriptors, null)).toThrow(
    "c: a `call(...)` inside a value isn't supported; bind it at the top level",
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-core/compose_nested_template_walkers__map_over_container_with_a_span", call: parsedNodes },
  { id: "load-core/compose_nested_template_walkers__wait_until_container_with_a_span", call: parsedNodes },
])("parseNodes: $id", runCase);

// The inline-call desugar raises the retired call before the `sub` callee resolves.
test.each<CaseRow>([
  { id: "load-core/compose_nested_template_walkers__nested_case_output_in_a_default_fails", call: assembled },
  { id: "load-core/compose_nested_template_walkers__retired_inline_flow_call_in_an_output", call: assembled },
])("assembled: $id", runCase);

// The reference check raises the dangling reference.
test("assembled: load-core/compose_nested_template_walkers__dangling_nested_reference_fails_at_its_own_line", () => {
  runCase({
    id: "load-core/compose_nested_template_walkers__dangling_nested_reference_fails_at_its_own_line",
    call: assembled,
  });
});

// The type-table pass raises the unknown name.
test("assembled: load-core/compose_nested_template_walkers__unknown_nested_isinstance_name_fails", () => {
  runCase({ id: "load-core/compose_nested_template_walkers__unknown_nested_isinstance_name_fails", call: assembled });
});
