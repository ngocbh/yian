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
 * `test_on_value_with_both_quotes_is_loud_load_error`,
 * `test_build_emits_reconciled_edges_no_provisional_group`,
 * `test_exhaustiveness_missing_tag_no_else_raises`, `test_exhaustiveness_else_satisfies_coverage`,
 * `test_exhaustiveness_all_tags_covered_no_else_ok`,
 * `test_exhaustiveness_walks_dotted_field_into_record`,
 * `test_exhaustiveness_dotted_missing_tag_raises`,
 * `test_exhaustiveness_projects_known_field_from_partial_record`,
 * `test_exhaustiveness_rejects_noncanonical_producer_contract`,
 * `test_exhaustiveness_rejects_noncanonical_unused_producer_contract`,
 * `test_no_exhaustiveness_for_searched_form`, `test_on_non_enum_producer_skips_exhaustiveness`, the
 * build half of
 * `tests/engine/test_validate_refs.py`'s `test_case_node_local_ref_does_not_error`, and the
 * allocation order of `tests/compose/test_null_narrowing_case.py`'s
 * `test_a_case_param_read_after_a_null_check_sees_the_value_as_not_null` (`__r0`, `__r1`, `__r2` in
 * read order); the whole row, with its narrowing, is ported in `compose/null_narrowing_case.test.ts`.
 *
 * Changed: the seed rows parse seeds 02, 06 and 10 through the conformance cases
 * `load-case/seed_02_case`, `load-case/seed_06_case_on` and
 * `load-leaves/seed_10_asserts_arithmetic` (`parseFile`, `flowNodesCursor`, `parseNodes`), where
 * Python reads the seed files. The quote rows build an `on:` case whose `when:` holds the quotes,
 * where Python calls `_quote`, which stays module-private. The exhaustiveness rows build without
 * the producers, which a build doesn't read. The non-canonical producer row pins the whole message,
 * which names the class (`Type`). The malformed-`when:` row pins the whole
 * message, where Python matches a fragment. The multiref row pins the whole data-edge list, where
 * Python checks for the `score` edge and against a `weight` one.
 *
 * TS-only: the seed `e38` row (`load-case/seed_e38_case_when_ref_locates_node_header`): the build
 * locates each `__rN` at the whole value of its `cases[N].when` (`__r1` on line 36) and labels it
 * so; the conformance case's full error comes from the reference check (pinned in
 * `errors.test.ts`). The missing-`then:` rows: a searched entry without a `then:` ("each searched
 * case needs a `then:`") is located at its `then:` range when the descriptor has one (a parsed
 * entry's is null, so no frame), else at `cases:`, else at the node; an `on:` entry with a `when:`
 * and no `then:` ("each on:-form case needs a `when:` value and a `then:`") at `cases:`, else at
 * the node.
 *
 * Pinned in `validate_refs.test.ts`: the `validate_references` half of
 * `test_case_node_local_ref_does_not_error`.
 */

import { expect, test } from "bun:test";

import type { Edge } from "../../src/compile/model.ts";
import { inferDataEdges } from "../../src/compose/edges/data.ts";
import { BuildContext, type BuildResult, siteKey } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { LoadDiagnostics, type LoadError } from "../../src/compose/helpers/errors.ts";
import { checkCaseExhaustive } from "../../src/compose/validate/case_exhaustive.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { evaluateWhenRecord } from "../../src/expr/expressions.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import { CaseNode } from "../../src/nodes/case/node.ts";
import { type OutputContract, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
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

/** Return a build context whose load errors locate in `text`, labelled `flow.yaml`. */
const located = (text: string) =>
  new BuildContext({ registry: new Map(), diagnostics: new LoadDiagnostics("flow.yaml", text) });

/** Return the range a load error points at, or `null` when it has no frame. */
const errorRange = (error: LoadError) => error.diagnostic.frames.at(-1)?.range ?? null;

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

test("a searched case without then: is refused at its then:, else at cases:, else at the node", () => {
  const message = "case node 'gate': each searched case needs a `then:`";
  const text =
    "id: c\nname: c\nnodes:\n  gate:\n    kind: case\n    cases:\n      - when: x.output > 1\n        then: hot\n" +
    "      - when: x.output < 0\n    else: cold\n";
  const ctx = located(text);
  const cases = [{ when: "x.output > 1" }];
  const thenRange = new SourceRange(8, 15, 8, 18);
  const casesRange = new SourceRange(7, 7, 10, 5);
  const nodeRange = new SourceRange(5, 5, 11, 1);

  // Parsed, the second entry's `then:` range is null, so the error is at `cases:`.
  const parsed = nodesOf(text, [], "flow.yaml").get("gate") as CaseDescriptor;
  const atParsedCases = loadError(() => parsed.build(ctx));
  expect(atParsedCases.message).toBe(message);
  expect(errorRange(atParsedCases)).toEqual(parsed.fieldRanges.get(siteKey("field", "cases"))!);
  // A descriptor built in code with a `then:` range for the entry: at that range.
  const atThen = loadError(() => new CaseDescriptor({ id: "gate", cases, thenRanges: [thenRange] }).build(ctx));
  expect(atThen.message).toBe(message);
  expect(errorRange(atThen)).toEqual(thenRange);
  // With no `then:` ranges: at `cases:`, else at the node.
  const fieldRanges = new Map([[siteKey("field", "cases"), casesRange]]);
  const atCases = loadError(() => new CaseDescriptor({ id: "gate", cases, fieldRanges, range: nodeRange }).build(ctx));
  expect(errorRange(atCases)).toEqual(casesRange);
  const atNode = loadError(() => new CaseDescriptor({ id: "gate", cases, range: nodeRange }).build(ctx));
  expect(errorRange(atNode)).toEqual(nodeRange);
});

test("a parsed searched case without when: is refused at cases:", () => {
  const text =
    "id: c\nname: c\nnodes:\n  gate:\n    kind: case\n    cases:\n      - when: x.output > 1\n        then: hot\n" +
    "      - then: cold\n    else: cold\n";
  const parsed = nodesOf(text, [], "flow.yaml").get("gate") as CaseDescriptor;

  const error = loadError(() => parsed.build(located(text)));

  expect(error.message).toBe("case node 'gate': each searched case needs a string `when:` expression");
  expect(errorRange(error)).toEqual(parsed.fieldRanges.get(siteKey("field", "cases"))!);
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
  // `inferDataEdges` emits no edge for the case node (it skips a case); the case's own build gives
  // its `__r0` data edge and its control edges.
  const { text, label } = caseText("load-case/seed_02_case");
  const descriptors = nodesOf(text, [], label);
  const flowWiring = new Map(
    [...descriptors]
      .filter(([, desc]) => !(desc instanceof CaseDescriptor))
      .map(([id, desc]) => [id, desc.build(CTX).wiring]),
  );
  expect(inferDataEdges(descriptors, flowWiring).some((edge) => edge.to === "gate")).toBe(false);

  const result = descriptors.get("gate")!.build(CTX);

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

/** The producers of an `on: ${cls.output}` case: `cls` gives a `Literal` of `tags`. */
function enumProducers(tags: readonly string[]): Map<string, OutputContract> {
  return new Map([["cls", new ValueContract(new Type({ kind: ValueKind.STRING, tags: new Set(tags) }))]]);
}

/** The `Literal(positive, negative, neutral)` stance tag. */
const STANCE = new Type({ kind: ValueKind.STRING, tags: new Set(["positive", "negative", "neutral"]) });

/** A record type whose `stance` field is `Literal(positive, negative, neutral)`, with `fields`. */
function view(fields: Record<string, Type> = {}): Type {
  const all = new Map([["stance", STANCE], ...Object.entries(fields)]);
  return new Type({ kind: ValueKind.OBJECT, fields: all, required: new Set(all.keys()) });
}

/** An `on: ${synth.output.stance}` case routing positive and negative, with `options`. */
function stanceCase(options: { else_?: string | null; range?: SourceRange | null } = {}): CaseDescriptor {
  return new CaseDescriptor({
    id: "route",
    on: "${synth.output.stance}",
    cases: [
      { when: "positive", then: "pro_note" },
      { when: "negative", then: "con_note" },
    ],
    else_: options.else_ ?? null,
    range: options.range ?? null,
  });
}

/** Return the one-entry descriptor map of `desc`. */
const only = (desc: CaseDescriptor) => new Map([[desc.id, desc]]);

test("exhaustiveness missing tag with no else raises", () => {
  const desc = onCase(["pro", "con"]);

  // The build lowers unconditionally; the exhaustiveness check is a later pass.
  expect(desc.build(CTX).node).toBeInstanceOf(CaseNode);
  expect(loadError(() => checkCaseExhaustive(only(desc), enumProducers(["pro", "con", "mixed"]))).message).toContain(
    "mixed",
  );
});

test("exhaustiveness: an else satisfies coverage", () => {
  // Two of three tags and an `else:` is not flagged.
  const desc = onCase(["pro", "con"], "neutral_note");

  expect(desc.build(CTX).node).toBeInstanceOf(CaseNode);
  checkCaseExhaustive(only(desc), enumProducers(["pro", "con", "mixed"]));
});

test("exhaustiveness: every tag covered with no else is ok", () => {
  const desc = onCase(["pro", "con"]);

  expect(desc.build(CTX).node).toBeInstanceOf(CaseNode);
  checkCaseExhaustive(only(desc), enumProducers(["pro", "con"]));
});

test("exhaustiveness walks a dotted field into a record", () => {
  // The `else:` covers `neutral`.
  checkCaseExhaustive(
    only(stanceCase({ else_: "neutral_note" })),
    new Map([["synth", new ValueContract(view({ claim: Type.scalar(ValueKind.STRING) }))]]),
  );
});

test("exhaustiveness: a dotted missing tag raises at the case", () => {
  const range = new SourceRange(17, 3, 17, 8);

  const error = loadError(() =>
    checkCaseExhaustive(only(stanceCase({ range })), new Map([["synth", new ValueContract(view())]]), {
      diagnostics: new LoadDiagnostics("flow.yaml", `${"x".repeat(80)}\n`.repeat(30)),
    }),
  );

  expect(error.message).toContain("neutral");
  expect(error.diagnostic.frames[0]!.range).toEqual(range);
});

test("exhaustiveness projects a known field from a partial record", () => {
  // An opaque sibling does not hide a proven enum field from coverage checks.
  const partial = new PartialRecordContract([
    ["stance", new ValueContract(STANCE)],
    ["note", null],
  ]);

  expect(loadError(() => checkCaseExhaustive(only(stanceCase()), new Map([["synth", partial]]))).message).toContain(
    "neutral",
  );
});

test("exhaustiveness rejects a non-canonical producer contract", () => {
  const notAContract = new Type({ kind: ValueKind.STRING, tags: new Set(["pro", "con"]) });

  expect(() =>
    checkCaseExhaustive(only(onCase(["pro"])), new Map([["cls", notAContract as unknown as OutputContract]])),
  ).toThrow(
    new TypeError(
      "output producer contracts must be ValueContract or PartialRecordContract values, got Type for 'cls'",
    ),
  );
});

test("exhaustiveness rejects a non-canonical unused producer contract", () => {
  // The CASE validator checks its contract map before looking at any case's form.
  const notAContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;

  expect(() => checkCaseExhaustive(only(seed02()), new Map([["unused", notAContract]]))).toThrow(
    "ValueContract or PartialRecordContract",
  );
});

test("no exhaustiveness for the searched form", () => {
  // A searched case over an enum-ish producer is not checked for exhaustiveness.
  checkCaseExhaustive(
    only(seed02()),
    new Map([["score", new ValueContract(new Type({ kind: ValueKind.STRING, tags: new Set(["a"]) }))]]),
  );
});

test("on over a non-enum producer skips exhaustiveness", () => {
  // A plain scalar producer (no tags) is lenient.
  const producers = new Map([["classify", new ValueContract(Type.scalar(ValueKind.STRING))]]);

  expect(seed06().build(CTX).node).toBeInstanceOf(CaseNode);
  checkCaseExhaustive(only(seed06()), producers);
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

test("an on: case without then: is refused at cases:, else at the node", () => {
  const message = "case node 'route': each on:-form case needs a `when:` value and a `then:`";
  const text =
    "id: c\nname: c\nnodes:\n  route:\n    kind: case\n    on: ${x.output}\n" +
    "    cases:\n      - when: a\n        then: hot\n      - when: b\n";
  const ctx = located(text);

  // Parsed, the second entry has a `when:` and a null `then:` range: at `cases:`.
  const parsed = nodesOf(text, [], "flow.yaml").get("route") as CaseDescriptor;
  const atCases = loadError(() => parsed.build(ctx));
  expect(atCases.message).toBe(message);
  expect(errorRange(atCases)).toEqual(new SourceRange(8, 7, 11, 1));
  // A descriptor built in code with no ranges but its own: at the node.
  const nodeRange = new SourceRange(5, 5, 11, 1);
  const cases = [{ when: "a" }];
  const desc = new CaseDescriptor({ id: "route", on: "${x.output}", cases, range: nodeRange });
  const atNode = loadError(() => desc.build(ctx));
  expect(atNode.message).toBe(message);
  expect(errorRange(atNode)).toEqual(nodeRange);
});
