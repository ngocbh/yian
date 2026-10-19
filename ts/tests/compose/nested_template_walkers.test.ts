/**
 * Load-time walkers see the strings inside a nested binding value: a port of
 * `tests/compose/test_nested_template_walkers.py`, through `assembleText` where Python loads the
 * flow; the MAP row parses the flow's nodes (`nodesOf`), since the MAP parse raises its error.
 *
 * Later (T3-17): `test_flow_output_field_reads_as_a_template` (flow outputs),
 * `test_nested_reference_orders_its_consumer_after_the_producer` (data edges) and
 * `test_nested_call_directive_is_not_lifted` (inline-call desugaring).
 * Later (T3-18): `test_dangling_nested_reference_fails_at_its_own_line`,
 * `test_dangling_nested_output_reference_fails_at_its_own_line`,
 * `test_nested_reference_field_is_checked` (reference checks), `test_nested_isinstance_name_resolves`
 * and `test_unknown_nested_isinstance_name_fails` (type tables; its case is pinned in
 * `isinstance_load.test.ts`).
 * Later (T3-17): `test_nested_case_output_in_a_default_fails` (CASE output expansion).
 * Later (T3-17, T4): `test_nested_case_output_expands_in_inputs_and_outputs` (a run).
 * Later (T3-22, T4): `test_map_over_literal_list_without_spans_still_loads` and
 * `test_nested_references_in_a_map_child_run_after_growth` (loads and runs).
 * Later (T3-16 for the loop kind, T3-22, T4):
 * `test_nested_references_in_a_loop_body_run_after_growth`.
 * Later (T3-17, T3-22): `test_retired_inline_flow_call_in_a_nested_output_fails` (defs, outputs).
 */

import { expect, test } from "bun:test";

import { Binding, ValueTemplate } from "../../src/nodes/contracts.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import { assembleText, type CaseRow, loadError, nodesOf, parsedNodes, runCase } from "./declarations.ts";

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

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-core/compose_nested_template_walkers__map_over_container_with_a_span", call: parsedNodes },
  { id: "load-core/compose_nested_template_walkers__wait_until_container_with_a_span", call: parsedNodes },
])("parseNodes: $id", runCase);
