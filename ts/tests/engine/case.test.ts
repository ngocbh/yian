/**
 * The `case` build to a `CaseNode`: the `on:`-form rows of `tests/engine/test_case.py`.
 *
 * Ported: `test_on_desugar_builds_case`, `test_on_control_edges`, `test_on_data_edge_reconciled`,
 * `test_desugared_on_routes_via_evaluate_when_record`, `test_handles_align_with_control_edges`,
 * `test_on_value_with_double_quote_uses_single_quote_and_routes`,
 * `test_on_value_with_both_quotes_is_loud_load_error`, and the build halves of
 * `test_exhaustiveness_missing_tag_no_else_raises` and
 * `test_on_non_enum_producer_skips_exhaustiveness`.
 *
 * Changed: the seed rows parse seed 06 through the conformance case `load-case/seed_06_case_on`
 * (`parseFile`, `flowNodesCursor`, `parseNodes`), where Python reads the seed file. The quote rows
 * build an `on:` case whose `when:` holds the quotes, where Python calls `_quote`, which stays
 * module-private. The exhaustiveness build halves build without the producers, which a build
 * doesn't read.
 *
 * Later (T3-16, the searched lowering): `test_searched_desugar_builds_case`,
 * `test_searched_control_edges`, `test_searched_data_edge_reconciled_to_input_group`,
 * `test_searched_multiref_allocates_r0_r1`,
 * `test_searched_when_all_spellings_desugar_equivalently`,
 * `test_searched_when_default_reads_through_an_optional_edge`,
 * `test_searched_when_coalesce_adds_a_required_join_input`,
 * `test_searched_when_join_reuses_a_plain_read`,
 * `test_searched_when_input_read_inside_an_escape_stays_plain`,
 * `test_searched_when_same_escape_in_two_cases_shares_locals`,
 * `test_searched_when_malformed_is_loud`, `test_desugared_searched_routes_via_evaluate_when_record`
 * and the build half of `test_build_emits_reconciled_edges_no_provisional_group`.
 * Later (T3-17): the `infer_data_edges` half of
 * `test_build_emits_reconciled_edges_no_provisional_group`.
 * Later (T3-20, `case_exhaustive`): the exhaustiveness halves of
 * `test_exhaustiveness_missing_tag_no_else_raises` and
 * `test_on_non_enum_producer_skips_exhaustiveness`, and
 * `test_exhaustiveness_else_satisfies_coverage`, `test_exhaustiveness_all_tags_covered_no_else_ok`,
 * `test_exhaustiveness_walks_dotted_field_into_record`,
 * `test_exhaustiveness_dotted_missing_tag_raises`,
 * `test_exhaustiveness_projects_known_field_from_partial_record`,
 * `test_exhaustiveness_rejects_noncanonical_producer_contract`,
 * `test_exhaustiveness_rejects_noncanonical_unused_producer_contract` and
 * `test_no_exhaustiveness_for_searched_form`.
 */

import { expect, test } from "bun:test";

import type { Edge } from "../../src/compile/model.ts";
import { BuildContext, type BuildResult } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { evaluateWhenRecord } from "../../src/expr/expressions.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import { CaseNode } from "../../src/nodes/case/node.ts";
import { caseText, loadError, nodesOf } from "../compose/declarations.ts";

/** A build context with no typedefs; a CASE build reads none. */
const CTX = new BuildContext({ registry: new Map() });

/** Return case node `nodeId` of conformance case `id`'s flow, parsed as the loader does. */
function caseDesc(id: string, nodeId: string): CaseDescriptor {
  const { text, label } = caseText(id);
  const desc = nodesOf(text, [], label).get(nodeId);
  expect(desc).toBeInstanceOf(CaseDescriptor);
  return desc as CaseDescriptor;
}

/** Seed 06's `route`: `on: ${classify.output}`, when pro, con and mixed. */
const seed06 = () => caseDesc("load-case/seed_06_case_on", "route");

/**
 * Split a CASE build's edges into its data edges (an `inputGroup`, no `sourceHandle`) and its
 * control edges (a `sourceHandle`, no `inputGroup`).
 */
function split(result: BuildResult): { data: Edge[]; control: Edge[] } {
  return {
    data: result.edges.filter((edge) => edge.sourceHandle === null),
    control: result.edges.filter((edge) => edge.sourceHandle !== null),
  };
}

/** Return the built `CaseNode` of a built CASE. */
const caseNode = (result: BuildResult) => result.node as CaseNode;

/** Return an `on:` case over `${cls.output}` routing each of `values` to `<value>_note`. */
function onCase(values: readonly string[], else_: string | null = null): CaseDescriptor {
  return new CaseDescriptor({
    id: "route",
    on: "${cls.output}",
    cases: values.map((value) => ({ when: value, then: `${value}_note` })),
    else_,
  });
}

test("on desugar builds case", () => {
  const result = seed06().build(CTX);

  expect(result.wiring).toEqual(new Map([["__on", "${classify.output}"]]));
  const node = caseNode(result);
  expect(node.cases.map((branch) => branch.when)).toEqual(['__on == "pro"', '__on == "con"', '__on == "mixed"']);
  expect(node.cases.map((branch) => branch.handle)).toEqual(["pro_note", "con_note", "choppy_note"]);
});

test("on control edges", () => {
  const { control } = split(seed06().build(CTX));

  const ctl = control.map((edge) => [edge.from, edge.to, edge.sourceHandle]);
  expect(ctl).toContainEqual(["route", "pro_note", "pro_note"]);
  expect(ctl).toContainEqual(["route", "con_note", "con_note"]);
  expect(ctl).toContainEqual(["route", "choppy_note", "choppy_note"]);
  // `else: choppy_note` takes the else handle: a distinct edge to the same target.
  expect(ctl).toContainEqual(["route", "choppy_note", ELSE_HANDLE]);
});

test("on data edge reconciled", () => {
  const { data } = split(seed06().build(CTX));

  expect(data.map((edge) => [edge.from, edge.to, edge.inputGroup])).toContainEqual(["classify", "route", "__on"]);
});

test("desugared on routes via evaluateWhenRecord", () => {
  const node = caseNode(seed06().build(CTX));

  expect(evaluateWhenRecord(node.cases[0]!.when!, { __on: "pro" })).toBe(true);
  expect(evaluateWhenRecord(node.cases[0]!.when!, { __on: "con" })).toBe(false);
  expect(evaluateWhenRecord(node.cases[1]!.when!, { __on: "con" })).toBe(true);
});

test("exhaustiveness missing tag with no else: the build lowers", () => {
  // The build lowers unconditionally; the exhaustiveness check is a later pass.
  expect(onCase(["pro", "con"]).build(CTX).node).toBeInstanceOf(CaseNode);
});

test("on over a non-enum producer: the build lowers", () => {
  expect(seed06().build(CTX).node).toBeInstanceOf(CaseNode);
});

test("handles align with control edges", () => {
  const result = seed06().build(CTX);
  const { control } = split(result);

  const caseHandles = new Set(caseNode(result).cases.map((branch) => branch.handle));
  const edgeHandles = new Set(control.map((edge) => edge.sourceHandle!));
  // Every case handle has an outgoing edge, and every edge handle is a case's or the else handle.
  expect([...caseHandles].every((handle) => edgeHandles.has(handle))).toBe(true);
  expect([...edgeHandles].every((handle) => caseHandles.has(handle) || handle === ELSE_HANDLE)).toBe(true);
});

test("on value with a double quote uses single quotes and routes", () => {
  // The `when:` grammar has no escapes, so a value holding `"` is single-quoted.
  const when = caseNode(onCase(['a"b']).build(CTX)).cases[0]!.when!;

  expect(when).toBe(`__on == 'a"b'`);
  expect(evaluateWhenRecord(when, { __on: 'a"b' })).toBe(true);
  expect(evaluateWhenRecord(when, { __on: "other" })).toBe(false);
});

test("on value with both quotes is a loud load error", () => {
  // A value holding both quote characters can't be a `when:` string literal.
  const error = loadError(() => onCase([`a'b"c`]).build(CTX));

  expect(error.message).toBe(
    `case \`on:\` match value 'a\\'b"c' contains both ' and " — unrepresentable in a \`when:\` string ` +
      "literal (no escape support); rename the matched value",
  );
});
