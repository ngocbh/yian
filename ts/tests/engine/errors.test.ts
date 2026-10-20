/**
 * Negative fixtures and their located source diagnostics: a port of `tests/engine/test_errors.py`
 * (the rows the parse phase decides), through `parseFile`, `flowNodesCursor` and `parseNodes`
 * with the registered kinds.
 *
 * All five gallery rows read conformance cases holding Python's `tests/seeds/errors/` flows:
 * `load-core/seed_e09_unknown_top_level_key`, `seed_e10_unknown_node_kind` and
 * `seed_e42_removed_model_kind`, and `load-leaves/seed_e11_field_not_allowed` and
 * `seed_e12_missing_required_field`. The `env` collision row reads
 * `load-core/seed_e33_input_named_like_field`.
 *
 * `test_e02_cycle_located` assembles conformance case `load-core/seed_e02_cycle` with `assembled`
 * (`assemble` as the loader calls it), pinned with its notes and both frames. Changed: its message
 * writes the node list in the text form (`["a","b"]`, rule `messages`).
 *
 * Later (T3-22): `test_e01_dangling_ref_located`,
 * `test_e03_unknown_field_located`, `test_negative_fixtures_are_located`,
 * `test_e04_exhaustiveness_mechanism_loud`, `test_e04_else_satisfies_coverage`,
 * `test_e06_cross_flow_type_mismatch_is_loud`, `test_single_node_def_model_kind_is_removed`,
 * the e13, e14, e15, e16, e17, e30, e31 and e32 rows of `test_compile_error_gallery`,
 * `test_inline_code_construction_errors_carry_ranges`,
 * `test_invalid_authored_default_carries_input_range`,
 * `test_bad_input_ref_locates_the_offending_entry`,
 * `test_bad_map_per_element_input_ref_locates_the_offending_entry`,
 * `test_map_over_ref_locates_the_over_line`, the `load_flow` half of
 * `test_input_named_like_a_field_does_not_collide`,
 * `test_llm_config_error_carries_the_field_range`,
 * `test_agent_controls_field_is_retired_and_located`,
 * `test_agent_mode_field_is_removed_and_located`,
 * `test_synth_node_locates_at_its_call_directive_line`, `test_case_when_ref_locates_at_when_line`.
 * Later (T4-8): `test_runtime_error_gallery`, `test_e23_unknown_tool_fails_at_run_start`,
 * `test_flow_input_type_failure_origin_resolves_to_declaration`,
 * `test_false_boundary_assert_origin_resolves_to_assertion`,
 * `test_duplicate_asserts_caret_at_first_occurrence`, `test_structured_output_origin_is_exact`.
 */

import { afterEach, expect, test } from "bun:test";

import { siteKey } from "../../src/compose/nodes/base.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { assembled, assembleText, type CaseRow, caseText, loadError, nodesOf, runCase } from "../compose/declarations.ts";

afterEach(() => forget());

/** The flow text of conformance case `id`. */
function seed(id: string): () => string {
  return () => caseText(id).text;
}

test.each([
  ["e09", seed("load-core/seed_e09_unknown_top_level_key"), "unknown top-level key 'bogus'", 14],
  // Per-kind parse anchors these at the precise offending field (not the node body):
  ["e10", seed("load-core/seed_e10_unknown_node_kind"), "unknown kind 'llm'", 8], // the value
  ["e42", seed("load-core/seed_e42_removed_model_kind"), "kind 'model' was removed", 8], // the line
  ["e11", seed("load-leaves/seed_e11_field_not_allowed"), "field 'prompt' is not allowed", 13],
  ["e12", seed("load-leaves/seed_e12_missing_required_field"), "missing required field 'code'", 8],
] as const)("compile error gallery: %s", (_, flow, substr, line) => {
  const error = loadError(() => nodesOf(flow()));

  expect(error.message).toContain(substr);
  expect(error.diagnostic.frames[0]!.range.startLine).toBe(line);
});

test("input named like a field does not collide", () => {
  // e33 has an `env:` field on L32 and an input named `env` on L35. `siteKey` keys keep
  // separate source ranges for the two.
  const flow = caseText("load-core/seed_e33_input_named_like_field").text;
  const document = SourceDocument.parse(flow.replace("${nope.output}", '"lit"'));
  const fr = parseNodes(document.root.child("nodes")).get("calc")!.fieldRanges;

  expect(fr.get(siteKey("field", "env"))!.startLine).toBe(32); // the `env:` field's VALUE
  expect(fr.get(siteKey("input", "env"))!.startLine).toBe(35); // the input NAMED `env`
  expect(fr.get(siteKey("field", "env"))).not.toEqual(fr.get(siteKey("input", "env")));
});

test.each<CaseRow>([{ id: "load-core/seed_e02_cycle", call: assembled }])("assemble: $id", runCase);

test("e02's cycle is located at every member", () => {
  // Nodes `a` and `b` begin at lines 12 and 18, in authored source order.
  const error = loadError(() => assembleText(caseText("load-core/seed_e02_cycle").text));
  expect(error.message).toBe('flow has a cycle involving ["a","b"]; flows must be acyclic');
  expect(error.diagnostic.frames.map(({ range }) => [range.startLine, range.startCol, range.endLine, range.endCol])).toEqual([
    [12, 3, 12, 4],
    [18, 3, 18, 4],
  ]);
  expect(error.diagnostic.notes).toEqual(["a depends on b (a.input.x)", "b depends on a (b.input.x)"]);
});
