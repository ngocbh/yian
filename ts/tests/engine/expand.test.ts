/**
 * Runtime graph expansion: a port of `tests/engine/test_expand.py`.
 *
 * Ported: `test_rens_internal_input_and_output_heads` and
 * `test_rens_internal_computed_whole_span_renamespaces_all_leaves`.
 *
 * Later (T4-10): `test_call_grow_clones_namespaced_and_substitutes_spawner`,
 * `test_call_inside_map_expands_and_runs_uniformly`,
 * `test_deep_static_call_map_chain_loads_and_succeeds`,
 * `test_deep_static_call_map_chain_resumes_after_its_last_expansion` and
 * `test_over_budget_expansion_trips_max_total_nodes`.
 */

import { expect, test } from "bun:test";

import { rensInternal } from "../../src/compile/expand.ts";
import { Binding } from "../../src/nodes/contracts.ts";

test("rensInternal re-namespaces input and output heads", () => {
  // `${input.k}` reads the namespaced child START's output object; `${x.output}` is node-first.
  expect(rensInternal(new Binding("${input.k}"), "cs")).toEqual(new Binding("${cs/__start__.output.k}"));
  expect(rensInternal(new Binding("${x.output.field}"), "cs")).toEqual(new Binding("${cs/x.output.field}"));
  // `system.X` and a non-text source pass through untouched.
  expect(rensInternal(new Binding("${system.now}"), "cs")).toEqual(new Binding("${system.now}"));
  expect(rensInternal(new Binding(5n), "cs")).toEqual(new Binding(5n));

  expect(() => rensInternal("${input.k}" as unknown as Binding, "cs")).toThrow(
    new TypeError("compiled flow wiring must contain Binding values, got string"),
  );
});

test("rensInternal re-namespaces every leaf of a computed whole span", () => {
  expect(rensInternal(new Binding("${x.output - y.output}"), "cs")).toEqual(
    new Binding("${cs/x.output - cs/y.output}"),
  );
  // Embedded text and two separate spans: the literal text is kept, both spans re-namespaced.
  expect(rensInternal(new Binding("pre ${x.output} mid ${input.k} post"), "cs")).toEqual(
    new Binding("pre ${cs/x.output} mid ${cs/__start__.output.k} post"),
  );
});
