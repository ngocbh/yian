/**
 * The `case` build to a `CaseNode`: a port of `tests/engine/test_case.py`.
 *
 * Ported: `test_searched_desugar_builds_case`, `test_searched_control_edges`,
 * `test_searched_data_edge_reconciled_to_input_group`, `test_searched_multiref_allocates_r0_r1`,
 * `test_searched_when_all_spellings_desugar_equivalently`,
 * `test_searched_when_default_reads_through_an_optional_edge`,
 * `test_searched_when_coalesce_adds_a_required_join_input`,
 * `test_searched_when_join_reuses_a_plain_read`,
 * `test_searched_when_input_read_inside_an_escape_stays_plain`,
 * `test_searched_when_same_escape_in_two_cases_shares_locals`,
 * `test_searched_when_malformed_is_loud`, `test_on_desugar_builds_case`, `test_on_control_edges`,
 * `test_on_data_edge_reconciled`, `test_desugared_searched_routes_via_evaluate_when_record`,
 * `test_desugared_on_routes_via_evaluate_when_record`, `test_handles_align_with_control_edges`,
 * `test_on_value_with_double_quote_uses_single_quote_and_routes`,
 * `test_on_value_with_both_quotes_is_loud_load_error`, the build halves of
 * `test_build_emits_reconciled_edges_no_provisional_group`,
 * `test_exhaustiveness_missing_tag_no_else_raises` and
 * `test_on_non_enum_producer_skips_exhaustiveness`, the build half of
 * `tests/engine/test_validate_refs.py`'s `test_case_node_local_ref_does_not_error`, and the
 * allocation order of `tests/compose/test_null_narrowing_case.py`'s
 * `test_a_case_param_read_after_a_null_check_sees_the_value_as_not_null` (`__r0`, `__r1`, `__r2` in
 * read order).
 *
 * Changed: the seed rows parse seeds 02, 06 and 10 through the conformance cases
 * `load-case/seed_02_case`, `load-case/seed_06_case_on` and
 * `load-leaves/seed_10_asserts_arithmetic` (`parseFile`, `flowNodesCursor`, `parseNodes`), where
 * Python reads the seed files. The quote rows build an `on:` case whose `when:` holds the quotes,
 * where Python calls `_quote`, which stays module-private. The exhaustiveness build halves build
 * without the producers, which a build doesn't read. The malformed-`when:` row pins the whole
 * message, where Python matches a fragment. The multiref row pins the whole data-edge list, where
 * Python checks for the `score` edge and against a `weight` one.
 *
 * TS-only: the seed `e38` row (`load-case/seed_e38_case_when_ref_locates_node_header`): the build
 * locates each `__rN` at the whole value of its `cases[N].when` (`__r1` on line 36) and labels it
 * so; the conformance case's full error comes from the reference check (T3-18).
 *
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
 * `test_no_exhaustiveness_for_searched_form`. Later (T3-18): the `validate_references` half of
 * `test_case_node_local_ref_does_not_error`. Later (T3-20): the narrowing half of
 * `test_a_case_param_read_after_a_null_check_sees_the_value_as_not_null`.
 */

import { expect, test } from "bun:test";

import type { Edge } from "../../src/compile/model.ts";
import { BuildContext, type BuildResult, siteKey } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import type { SourceRange } from "../../src/diagnostics.ts";
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

/** Seed 02's `gate`: when `score.output >= 0.5` then positive, else cautious. */
const seed02 = () => caseDesc("load-case/seed_02_case", "gate");

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

/** Return a range as `[startLine, startCol, endLine, endCol]`, or `null`. */
const span = (range: SourceRange | undefined) =>
  range === undefined ? null : [range.startLine, range.startCol, range.endLine, range.endCol];

/** Return the built `CaseNode` of a built CASE. */
const caseNode = (result: BuildResult) => result.node as CaseNode;

/**
 * Build a searched case `gate` with one case per `when:` (case `i` routes to `t<i>`), else
 * `cautious`.
 */
function searched(...whens: string[]): BuildResult {
  return new CaseDescriptor({
    id: "gate",
    cases: whens.map((when, i) => ({ when, then: `t${i}` })),
    else_: "cautious",
  }).build(CTX);
}

/** Return each data edge of a CASE build as `[from, inputGroup, optional]`, in order. */
function edgeStances(result: BuildResult): [string, string | null, boolean][] {
  return split(result).data.map((edge) => [edge.from, edge.inputGroup, edge.optional]);
}

/** Return an `on:` case over `${cls.output}` routing each of `values` to `<value>_note`. */
function onCase(values: readonly string[], else_: string | null = null): CaseDescriptor {
  return new CaseDescriptor({
    id: "route",
    on: "${cls.output}",
    cases: values.map((value) => ({ when: value, then: `${value}_note` })),
    else_,
  });
}

// ---------- the searched form ----------

test("searched desugar builds case", () => {
  const result = seed02().build(CTX);
  const node = caseNode(result);

  expect(node).toBeInstanceOf(CaseNode);
  expect(node.id).toBe("gate");
  // One distinct reference, so one input `__r0`, wired to `${score.output}`.
  expect(node.params!.map((param) => param.name)).toEqual(["__r0"]);
  expect(result.wiring).toEqual(new Map([["__r0", "${score.output}"]]));
  // The `when:` reads the local `__r0`, in the seed's bare spelling.
  expect(node.cases).toHaveLength(1);
  expect(node.cases[0]!.when).toBe("__r0 >= 0.5");
  expect(node.cases[0]!.handle).toBe("positive");
});

test("searched control edges", () => {
  const { control } = split(seed02().build(CTX));

  const ctl = control.map((edge) => [edge.from, edge.to, edge.sourceHandle]);
  expect(ctl).toContainEqual(["gate", "positive", "positive"]);
  expect(ctl).toContainEqual(["gate", "cautious", ELSE_HANDLE]);
});

test("searched data edge reconciled to its input group", () => {
  const { data } = split(seed02().build(CTX));

  expect(data.map((edge) => [edge.from, edge.to, edge.inputGroup])).toContainEqual(["score", "gate", "__r0"]);
  // No provisional `<case>:<n>` group.
  expect(data.some((edge) => (edge.inputGroup ?? "").includes(":"))).toBe(false);
});

test("searched multiref allocates __r0 and __r1", () => {
  // Seed 10: `score.output > 0 and input.weight * 100 <= 5`.
  const result = caseDesc("load-leaves/seed_10_asserts_arithmetic", "size").build(CTX);
  const { data } = split(result);

  expect(result.wiring).toEqual(
    new Map([
      ["__r0", "${score.output}"],
      ["__r1", "${input.weight}"],
    ]),
  );
  expect(caseNode(result).cases[0]!.when).toBe("__r0 > 0 and __r1 * 100 <= 5");
  // Only the node output is a data edge; `input.weight` isn't.
  expect(data.map((edge) => [edge.id, edge.from, edge.to, edge.inputGroup, edge.optional])).toEqual([
    ["score->size#0", "score", "size", "__r0", false],
  ]);
});

// The reads and the rewrite come from the same parse, so both spellings allocate, wire and route
// alike.
test.each([
  ["score.output >= 0.5", "__r0 >= 0.5"],
  ["${score.output >= 0.5}", "${__r0 >= 0.5}"],
])("searched when: all spellings desugar equivalently: %s", (when, expectWhen) => {
  const result = new CaseDescriptor({ id: "gate", cases: [{ when, then: "positive" }], else_: "cautious" }).build(CTX);
  const node = caseNode(result);

  expect(node.params!.map((param) => param.name)).toEqual(["__r0"]);
  expect(result.wiring).toEqual(new Map([["__r0", "${score.output}"]]));
  expect(node.cases[0]!.when).toBe(expectWhen);
  expect(evaluateWhenRecord(node.cases[0]!.when!, { __r0: 0.7 })).toBe(true);
  expect(evaluateWhenRecord(node.cases[0]!.when!, { __r0: 0.3 })).toBe(false);
  expect(split(result).data.map((edge) => [edge.from, edge.to, edge.inputGroup])).toContainEqual([
    "score",
    "gate",
    "__r0",
  ]);
});

test.each([
  ["(score.output :- 0) >= 0.5", "(__r0 :- 0) >= 0.5"],
  ["${(score.output :- 0) >= 0.5}", "${(__r0 :- 0) >= 0.5}"],
])("searched when: a default reads through an optional edge: %s", (when, expectWhen) => {
  const result = searched(when);
  const lowered = caseNode(result).cases[0]!.when!;

  expect(result.wiring).toEqual(new Map([["__r0", "${score.output :- null}"]]));
  expect(lowered).toBe(expectWhen);
  expect(edgeStances(result)).toEqual([["score", "__r0", true]]);
  // The rewrite evaluates as the authored condition: a missing score takes the default.
  expect(evaluateWhenRecord(lowered, { __r0: null })).toBe(false);
  expect(evaluateWhenRecord(lowered, { __r0: 0.7 })).toBe(true);
});

test("searched when: a coalesce adds a required join input", () => {
  const result = searched("(score.output | other.output) >= 0.5");

  expect(result.wiring).toEqual(
    new Map([
      ["__r0", "${score.output :- null}"],
      ["__r1", "${other.output :- null}"],
      ["__r2", "${score.output | other.output}"],
    ]),
  );
  expect(caseNode(result).params!.map((param) => param.name)).toEqual(["__r0", "__r1", "__r2"]);
  expect(caseNode(result).cases[0]!.when).toBe("(__r0 | __r1) >= 0.5");
  expect(edgeStances(result)).toEqual([
    ["score", "__r0", true],
    ["other", "__r1", true],
    ["score", "__r2", false],
    ["other", "__r2", false],
  ]);
});

test("searched when: a join reuses a plain read", () => {
  const result = searched("score.output > 5 or (0 :- score.output) < 1");

  expect(result.wiring).toEqual(
    new Map([
      ["__r0", "${score.output}"],
      ["__r1", "${score.output :- null}"],
    ]),
  );
  expect(caseNode(result).cases[0]!.when).toBe("__r0 > 5 or (0 :- __r1) < 1");
  expect(edgeStances(result)).toEqual([
    ["score", "__r0", false],
    ["score", "__r1", true],
  ]);
});

test("searched when: an input read inside an escape stays plain", () => {
  // `:- null` only changes a node-output read's edge; an input read has none to change.
  const result = searched("(input.weight :- 0) > 1 and input.weight < 5");

  expect(result.wiring).toEqual(new Map([["__r0", "${input.weight}"]]));
  expect(caseNode(result).cases[0]!.when).toBe("(__r0 :- 0) > 1 and __r0 < 5");
});

test("searched when: the same escape in two cases shares locals", () => {
  const result = searched("(score.output | other.output) > 1", "(score.output | other.output) < 0");

  expect([...result.wiring.keys()]).toEqual(["__r0", "__r1", "__r2"]);
  expect(caseNode(result).cases.map((branch) => branch.when)).toEqual(["(__r0 | __r1) > 1", "(__r0 | __r1) < 0"]);
});

test("searched when: malformed is loud", () => {
  const cases = [{ when: "score >= >= 5", then: "positive" }];

  const error = loadError(() => new CaseDescriptor({ id: "gate", cases, else_: "cautious" }).build(CTX));

  expect(error.message).toBe(
    "case node 'gate': `when:` 'score >= >= 5' is not a valid boolean expression: " +
      "could not parse expression 'score >= >= 5': unexpected '>=' at line 1, column 10",
  );
});

test("searched reads are allocated in read order", () => {
  const result = searched("x.output.a == 'p' and x.output != null and x.output.b == 'q'");

  expect(result.wiring).toEqual(
    new Map([
      ["__r0", "${x.output.a}"],
      ["__r1", "${x.output}"],
      ["__r2", "${x.output.b}"],
    ]),
  );
});

test("a case node's local reads: the build half", () => {
  // The rewritten `when:` reads only the node-local `__r0`.
  const result = seed02().build(CTX);
  const node = caseNode(result);

  expect(node.params!.map((param) => param.name)).toEqual(["__r0"]);
  expect(result.wiring).toEqual(new Map([["__r0", "${score.output}"]]));
  expect(node.cases[0]!.when).toBe("__r0 >= 0.5");
});

test("seed e38: each generated input locates at its when and is labelled so", () => {
  const result = caseDesc("load-case/seed_e38_case_when_ref_locates_node_header", "gate").build(CTX);

  // Each at its `when:` value: `"score.output >= 0.5"` on line 34, `"nope.output >= 0.9"` on 36.
  expect(span(result.fieldRanges.get(siteKey("input", "__r0")))).toEqual([34, 15, 34, 36]);
  expect(span(result.fieldRanges.get(siteKey("input", "__r1")))).toEqual([36, 15, 36, 35]);
  expect(result.node.wiringLabels).toEqual(
    new Map([
      ["__r0", "cases[0].when"],
      ["__r1", "cases[1].when"],
    ]),
  );
});

test("desugared searched routes via evaluateWhenRecord", () => {
  const node = caseNode(seed02().build(CTX));

  // The bound input record uses the bare local names.
  expect(evaluateWhenRecord(node.cases[0]!.when!, { __r0: 0.7 })).toBe(true);
  expect(evaluateWhenRecord(node.cases[0]!.when!, { __r0: 0.3 })).toBe(false);
});

test("the build emits reconciled edges and no provisional group", () => {
  // The case's own build gives its `__r0` data edge and its control edges.
  const result = seed02().build(CTX);

  expect(result.edges.some((edge) => (edge.inputGroup ?? "").includes(":"))).toBe(false);
  expect(result.edges.map((edge) => [edge.from, edge.to, edge.inputGroup])).toContainEqual(["score", "gate", "__r0"]);
  const ctl = split(result).control.map((edge) => [edge.from, edge.to, edge.sourceHandle]);
  expect(ctl).toContainEqual(["gate", "positive", "positive"]);
  expect(ctl).toContainEqual(["gate", "cautious", ELSE_HANDLE]);
});

// ---------- the on: form ----------

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
