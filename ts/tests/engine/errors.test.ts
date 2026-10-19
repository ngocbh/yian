/**
 * Negative fixtures and their located source diagnostics: a port of `tests/engine/test_errors.py`
 * (the rows the parse phase decides), through `parseFile`, `flowNodesCursor` and `parseNodes`
 * with stand-in kinds.
 *
 * The e09, e10 and e42 gallery rows read the conformance cases
 * `load-core/seed_e09_unknown_top_level_key`, `seed_e10_unknown_node_kind` and
 * `seed_e42_removed_model_kind`, which hold Python's `tests/seeds/errors/` flows; the e11 and e12
 * flows are copied here verbatim. The `env` collision row reads
 * `load-core/seed_e33_input_named_like_field`.
 *
 * Later (T3-22): `test_e01_dangling_ref_located`, `test_e02_cycle_located`,
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
import { flowNodesCursor, parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { caseDocument, loadError } from "../compose/declarations.ts";
import { ALL_KIND_NAMES, StandInCode, standIn, withKinds } from "../compose/stand_ins.ts";

afterEach(() => forget());

const E11 = `# e11 — a flat field illegal for the kind. COMPILE: "field 'prompt' is not allowed" (code).
id: err-field-not-allowed
name: err_field_not_allowed
input:
  topic: !str
nodes:
  calc:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: tests.seeds.fns:echo_value
    prompt: "nope"     # ERROR: a CODE node has no prompt
output: \${calc.output}
`;

const E12 = `# e12 — a kind-required field omitted. COMPILE: "missing required field 'code'".
id: err-missing-required-field
name: err_missing_required_field
input:
  topic: !str
nodes:
  calc:
    kind: code         # ERROR: missing \`code:\`
    input:
      topic: \${input.topic}
    output: str
output: \${calc.output}
`;

/** The flow of seed case `name` in `load-core`. */
function seed(name: string): () => SourceDocument {
  return () => caseDocument(`load-core/${name}`).document;
}

test.each([
  ["e09", seed("seed_e09_unknown_top_level_key"), "unknown top-level key 'bogus'", 14],
  // Per-kind parse anchors these at the precise offending field (not the node body):
  ["e10", seed("seed_e10_unknown_node_kind"), "unknown kind 'llm'", 8], // the `kind:` value
  ["e42", seed("seed_e42_removed_model_kind"), "kind 'model' was removed", 8], // the `kind:` line
  ["e11", () => parseSourceDocument(E11, "<flow>"), "field 'prompt' is not allowed", 13],
  ["e12", () => parseSourceDocument(E12, "<flow>"), "missing required field 'code'", 8],
] as const)("compile error gallery: %s", (_, document, substr, line) => {
  const kinds = [StandInCode, ...ALL_KIND_NAMES.filter((k) => k !== "code").map((k) => standIn(k))];
  const error = loadError(() => {
    const flow = document();
    const cursor = flowNodesCursor(flow, parseFile(flow).nodes);
    withKinds(kinds, () => parseNodes(cursor));
  });

  expect(error.message).toContain(substr);
  expect(error.diagnostic.frames[0]!.range.startLine).toBe(line);
});

test("input named like a field does not collide", () => {
  // e33 has an `env:` field on L32 and an input named `env` on L35. Tuple-tagged keys retain
  // separate source ranges for those author constructs.
  const flow = caseDocument("load-core/seed_e33_input_named_like_field").case_["flow"] as string;
  const document = SourceDocument.parse(flow.replace("${nope.output}", '"lit"'));
  const fr = withKinds([StandInCode], () => parseNodes(document.root.child("nodes")))
    .get("calc")!.fieldRanges;

  expect(fr.get(siteKey("field", "env"))!.startLine).toBe(32); // the `env:` field's VALUE
  expect(fr.get(siteKey("input", "env"))!.startLine).toBe(35); // the input NAMED `env`
  expect(fr.get(siteKey("field", "env"))).not.toEqual(fr.get(siteKey("input", "env")));
});
