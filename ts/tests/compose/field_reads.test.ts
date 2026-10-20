/**
 * Load-time checks for reading a field a value's known type can't have: a port of the node-assert
 * rows of `tests/compose/test_field_reads.py`,
 * `::test_node_assert_on_a_partial_record_output_checks_typed_fields` and
 * `::test_node_assert_on_an_exact_record_output_checks_typed_fields`, each pinning the whole
 * message where Python matches a fragment. The flows assemble with `assembleText` where Python
 * calls `load_flow`; the partial `child` def is built by hand (`handChild`, a typed `a` and an
 * opaque `b`), since def bodies load in T3-22.
 *
 * Changed: the CODE bodies are TS's, as the harvested cases spell them.
 *
 * TS-only: the harvested `load-core/compose_field_reads__node_assert_reads_a_field_of_a_str_field`,
 * pinned at the end.
 *
 * Later (T3-20, `lowerFlowAsserts`): `test_field_on_str_in_a_flow_assert_is_a_load_error` and its
 * case.
 *
 * Later (T3-18, the reference validator): `test_field_on_a_fieldless_output_in_a_flow_output_is_a_load_error`,
 * `test_field_on_str_in_a_node_binding_is_a_load_error`, `test_field_on_str_in_a_case_when_is_a_load_error`,
 * `test_field_on_str_in_a_case_on_is_a_load_error`, `test_field_on_str_in_any_reference_form_is_a_load_error`,
 * `test_nested_field_on_a_record_field_of_type_str_is_a_load_error`,
 * `test_undeclared_record_field_keeps_its_message`, `test_field_on_str_in_map_over_is_a_load_error`,
 * `test_field_on_str_in_wait_until_is_a_load_error`, `test_field_on_a_bare_call_output_is_a_load_error`,
 * `test_field_on_a_map_output_is_a_load_error`, `test_partial_record_output_checks_names_and_typed_fields`,
 * `test_reads_the_type_allows_still_load`, `test_untyped_output_reads_still_load`,
 * `test_flow_input_field_reads_are_checked`, `test_valid_flow_input_reads_still_load`,
 * `test_inline_body_input_field_reads_are_checked`, `test_map_item_field_reads_are_checked` and
 * `test_map_item_reads_the_type_allows_still_load`; and their cases.
 *
 * Later (T3-22, the child's own assembly): `test_defs_body_input_field_reads_are_checked`.
 *
 * Later (T4): `test_optional_record_field_reads_null_when_the_whole_value_is_null` and
 * `test_case_when_on_a_missing_optional_record_input_is_falsy` (runs).
 */

import { expect, test } from "bun:test";

import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { assembled, assembleText, handChild, loadError, parametersFromYaml, runCase } from "./declarations.ts";

/**
 * A flow with one CODE node `step` typed `stepOutput`, plus `rest` (YAML tail); `preamble` holds
 * top-level blocks placed before `input:`.
 */
function flow(stepOutput: string, rest: string, preamble = ""): string {
  return (
    "id: p\nname: p\n" +
    preamble +
    "input:\n  s: !str\n" +
    "nodes:\n" +
    "  step:\n" +
    "    kind: code\n" +
    "    input:\n" +
    "      x: ${input.s}\n" +
    `    output: ${stepOutput}\n` +
    '    code: "return null"\n' +
    rest
  );
}

// A child whose output is a partial record: `a` is typed, `b` comes from an untyped CODE node.
const PARTIAL_CHILD_TEXT =
  "id: p\nname: p\n" +
  "input:\n  s: !str\n" +
  "defs:\n" +
  "  child:\n" +
  "    input:\n      s: !str\n" +
  "    nodes:\n" +
  "      typed:\n" +
  "        kind: code\n" +
  "        input:\n          v: ${input.s}\n" +
  "        output: str\n" +
  "        code: return 'x'\n" +
  "      untyped:\n" +
  "        kind: code\n" +
  "        input:\n          v: ${input.s}\n" +
  '        code: "return ({k: 1})"\n' +
  "    output:\n      a: ${typed.output}\n      b: ${untyped.output}\n" +
  "nodes:\n" +
  "  c:\n" +
  "    kind: call\n" +
  "    call: child\n" +
  "    input:\n      s: ${input.s}\n";

/** The hand-built partial `child`: `s: !str` in, a record of a typed `a` and an opaque `b` out. */
const PARTIAL_CHILD = handChild({
  inputs: parametersFromYaml("input:\n  s: !str\n", new Map()),
  producers: { typed: new ValueContract(Type.scalar(ValueKind.STRING)), untyped: null },
  output: { a: "${typed.output}", b: "${untyped.output}" },
});

test("a node assert on a partial record output checks typed fields", () => {
  const text =
    PARTIAL_CHILD_TEXT.replace(
      "    input:\n      s: ${input.s}\n",
      "    input:\n      s: ${input.s}\n    asserts:\n      - output.a.x == 1\n",
    ) + "output: ${c.output.a}\n";
  const error = loadError(() => assembleText(text, "flow.yaml", { children: { child: PARTIAL_CHILD } }));
  expect(error.message).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'c' assert: reference ${output.a.x} reads field 'x' on type str, which has no fields",
  );
});

test("a node assert on an exact record output checks typed fields", () => {
  const text = flow("{a: str}", "output: ${step.output.a}\n").replace(
    '    code: "return null"\n',
    '    code: "return null"\n    asserts:\n      - output.a.b == 1\n',
  );
  expect(loadError(() => assembleText(text)).message).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'step' assert: reference ${output.a.b} reads field 'b' on type str, which has no fields",
  );
});

// ---------- harvested conformance cases ----------

test("assembled: load-core/compose_field_reads__node_assert_reads_a_field_of_a_str_field", () => {
  runCase({ id: "load-core/compose_field_reads__node_assert_reads_a_field_of_a_str_field", call: assembled });
});
