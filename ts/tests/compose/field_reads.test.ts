/**
 * Load-time checks for reading a field a value's known type can't have: a port of
 * `tests/compose/test_field_reads.py`, loading each flow with `assembleText` (`assemble` as the
 * loader calls it) where Python calls `load_flow`.
 *
 * Ported: `test_field_on_a_fieldless_output_in_a_flow_output_is_a_load_error`,
 * `test_field_on_str_in_a_node_binding_is_a_load_error`,
 * `test_field_on_str_in_a_case_when_is_a_load_error`,
 * `test_field_on_str_in_a_case_on_is_a_load_error`,
 * `test_field_on_str_in_any_reference_form_is_a_load_error`,
 * `test_nested_field_on_a_record_field_of_type_str_is_a_load_error`,
 * `test_undeclared_record_field_keeps_its_message`, `test_field_on_str_in_map_over_is_a_load_error`,
 * `test_field_on_str_in_wait_until_is_a_load_error` (on a sleep's `until` input),
 * `test_field_on_a_bare_call_output_is_a_load_error`,
 * `test_field_on_a_map_output_is_a_load_error`, `test_field_on_str_in_a_flow_assert_is_a_load_error`,
 * `test_partial_record_output_checks_names_and_typed_fields`,
 * `test_node_assert_on_a_partial_record_output_checks_typed_fields`,
 * `test_node_assert_on_an_exact_record_output_checks_typed_fields`, `test_reads_the_type_allows_still_load`,
 * `test_untyped_output_reads_still_load`, `test_flow_input_field_reads_are_checked`,
 * `test_valid_flow_input_reads_still_load`, `test_inline_body_input_field_reads_are_checked`,
 * `test_map_item_field_reads_are_checked`, `test_map_item_reads_the_type_allows_still_load` and
 * `test_defs_body_input_field_reads_are_checked`. The assert rows pin the whole message where
 * Python matches a fragment.
 *
 * Pinned at the end, the harvested `load-core/compose_field_reads__*` cases the reference validator
 * decides, and the assert checks' `load-core/compose_field_reads__field_on_str_in_a_flow_assert` and
 * `load-core/compose_field_reads__node_assert_reads_a_field_of_a_str_field`.
 *
 * Ported, as a flow run (first, as in Python): `test_case_when_on_a_missing_optional_record_input_is_falsy`,
 * both runs (its case `run-core/compose_field_reads__case_when_on_a_missing_optional_record_is_falsy`
 * holds only the first).
 *
 * Covered by `run-core/compose_field_reads__optional_record_field_reads_null`:
 * `test_optional_record_field_reads_null_when_the_whole_value_is_null`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { fieldReadErrors, NULLABLE_FIELD_READ, nullableReadErrors, type ReadRoots, untypedAt } from "../../src/compose/validate/field_reads.ts";
import { parseCondition } from "../../src/expr/expressions.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { hostEnv } from "../engine/mcp_helpers.ts";
import { assembled, assembleText, type CaseRow, loadConformanceFlow, loadError, runCase } from "./declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-field-reads-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/**
 * A flow with one CODE node `step` typed `stepOutput`, plus `rest` (YAML tail); `preamble` holds
 * top-level blocks placed before `input:`, such as `typedefs:` or `defs:`.
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
    "    code: return None\n" +
    rest
  );
}

/** Load `text`. */
const load = (text: string) => assembleText(text);

/** Return the message of the `LoadError` loading `text` raises. */
const failure = (text: string) => loadError(() => load(text)).message;

test("case when on a missing optional record input is falsy", async () => {
  // A `when:` reading a field of an omitted `Optional(Rec)` input through `?.` takes the else branch.
  const text =
    "id: p\nname: p\n" +
    "typedefs:\n  Rec: {a: str}\n" +
    "input:\n  s: !str\n  maybe: !Optional(Rec)\n" +
    "nodes:\n" +
    "  route:\n" +
    "    kind: case\n" +
    "    cases:\n" +
    "      - when: ${input.maybe?.a == 'x'}\n" +
    "        then: hit\n" +
    "    else: miss\n" +
    "  hit:\n" +
    "    kind: code\n" +
    "    input:\n      s: ${input.s}\n" +
    "    output: str\n" +
    "    code: return 'hit'\n" +
    "  miss:\n" +
    "    kind: code\n" +
    "    input:\n      s: ${input.s}\n" +
    "    output: str\n" +
    "    code: return 'miss'\n" +
    "output: ${hit.output | miss.output}\n";
  const loaded = await loadConformanceFlow(text);

  expect((await runFlow(loaded, { s: "x" }, { env })).output).toBe("miss");
  expect((await runFlow(loaded, { s: "x", maybe: { a: "x" } }, { env })).output).toBe("hit");
});

test.each([
  ["str", "str"],
  ["int", "int"],
  ["float", "float"],
  ["bool", "bool"],
  ["date", "date"],
  ["datetime", "datetime"],
  ["int(0..10)", "int(0..10)"],
  ["Literal(a, b)", "Literal(a, b)"],
  ["list(str)", "list(str)"],
  ["Optional(str)", "Optional(str)"],
])("a field on a fieldless output in a flow output is a load error: %s", (stepOutput, label) => {
  expect(failure(flow(stepOutput, "output:\n  o: ${step.output.order}\n"))).toContain(
    `flow output from: reference \${step.output.order} reads field 'order' on type ${label}, which has no fields`,
  );
});

test("a field on str in a node binding is a load error", () => {
  const rest =
    "  sink:\n" +
    "    kind: code\n" +
    "    input:\n" +
    "      v: ${step.output.order}\n" +
    "    output: str\n" +
    "    code: return 'x'\n" +
    "output: ${sink.output}\n";
  expect(failure(flow("str", rest))).toContain(
    "node 'sink' input 'v' from: reference ${step.output.order} reads field 'order'",
  );
});

const SINK = "  a:\n    kind: code\n    input:\n      v: ${input.s}\n    output: str\n    code: return 'x'\n";

test("a field on str in a case when is a load error", () => {
  const rest =
    SINK +
    "  route:\n" +
    "    kind: case\n" +
    "    cases:\n" +
    "      - when: ${step.output.order == 'x'}\n" +
    "        then: a\n" +
    "    else: a\n" +
    "output: ${a.output}\n";
  expect(failure(flow("str", rest))).toContain(
    "node 'route' cases[0].when from: reference ${step.output.order} reads field 'order' on type str",
  );
});

test("a field on str in a case on is a load error", () => {
  const rest =
    SINK +
    "  route:\n" +
    "    kind: case\n" +
    "    on: ${step.output.kind}\n" +
    "    cases:\n" +
    "      - when: x\n" +
    "        then: a\n" +
    "    else: a\n" +
    "output: ${a.output}\n";
  expect(failure(flow("str", rest))).toContain(
    "node 'route' on: from: reference ${step.output.kind} reads field 'kind' on type str",
  );
});

test.each([
  ["'note: ${step.output.order}'"], // embedded in a template
  ["${step.output.order | input.s}"], // coalesce
  ["${input.s :- step.output.order}"], // default RHS
])("a field on str in any reference form is a load error: %s", (binding) => {
  expect(failure(flow("str", `output:\n  o: ${binding}\n`))).toContain(
    "reference ${step.output.order} reads field 'order' on type str",
  );
});

test("a nested field on a record field of type str is a load error", () => {
  expect(failure(flow("{a: str}", "output:\n  o: ${step.output.a.b}\n"))).toContain(
    "reference ${step.output.a.b} reads field 'b' on type str, which has no fields",
  );
});

test("an undeclared record field keeps its message", () => {
  expect(failure(flow("{a: str}", "output:\n  o: ${step.output.b}\n"))).toContain(
    "reference ${step.output.b} reads unknown field 'b' on record",
  );
});

const EACH_DEFS =
  "defs:\n" +
  "  each:\n" +
  "    input:\n      v: !str\n" +
  "    nodes:\n" +
  "      inner:\n" +
  "        kind: code\n" +
  "        input:\n          v: ${input.v}\n" +
  "        output: str\n" +
  "        code: return 'x'\n" +
  "    output: ${inner.output}\n";

test("a field on str in map over is a load error", () => {
  const text = flow(
    "str",
    "  m:\n" +
      "    kind: map\n" +
      "    call: each\n" +
      "    over: ${step.output.items}\n" +
      "    input:\n      v: ${item}\n" +
      "output: ${m.output}\n",
    EACH_DEFS,
  );
  expect(failure(text)).toContain(
    "node 'm' map over: reference ${step.output.items} reads field 'items' on type str",
  );
});

test("a field on str in a sleep's until is a load error", () => {
  const text = flow(
    "str",
    "  settle:\n    kind: tool\n    tool_id: sleep\n    input:\n      until: ${step.output.at}\noutput: ${step.output}\n",
  );
  expect(failure(text)).toContain(
    "node 'settle' input 'until' from: reference ${step.output.at} reads field 'at' on type str",
  );
});

const BARE_CHILD =
  "id: p\nname: p\n" +
  "input:\n  s: !str\n  xs: !list(str)\n" +
  "defs:\n" +
  "  child:\n" +
  "    input:\n      s: !str\n" +
  "    nodes:\n" +
  "      step:\n" +
  "        kind: code\n" +
  "        input:\n          v: ${input.s}\n" +
  "        output: str\n" +
  "        code: return 'x'\n" +
  "    output: ${step.output}\n";

test("a field on a bare call output is a load error", () => {
  const text =
    BARE_CHILD +
    "nodes:\n  c:\n    kind: call\n    call: child\n    input:\n      s: ${input.s}\noutput: ${c.output.order}\n";
  expect(failure(text)).toContain(
    "flow output from: reference ${c.output.order} reads field 'order' on type str, which has no fields",
  );
});

test("a field on a map output is a load error", () => {
  const text =
    BARE_CHILD +
    "nodes:\n  m:\n    kind: map\n    call: child\n    over: ${input.xs}\n    input:\n      s: ${item}\n" +
    "output: ${m.output.first}\n";
  expect(failure(text)).toContain("reference ${m.output.first} reads field 'first' on type list(str)");
});

test("a field on str in a flow assert is a load error", () => {
  const text = flow("str", "asserts:\n  - ${step.output.order == 'x'}\noutput: ${step.output}\n");
  expect(failure(text)).toBe(
    "assert \"${step.output.order == 'x'}\": reference ${step.output.order} reads field 'order' on type str, " +
      "which has no fields",
  );
});

// A child whose output is a partial record: `a` is typed, `b` comes from an untyped CODE node.
const PARTIAL_CHILD =
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
  '        code: "return {k: 1}"\n' +
  "    output:\n      a: ${typed.output}\n      b: ${untyped.output}\n" +
  "nodes:\n" +
  "  c:\n" +
  "    kind: call\n" +
  "    call: child\n" +
  "    input:\n      s: ${input.s}\n";

test("a partial record output checks names and typed fields", () => {
  expect(failure(PARTIAL_CHILD + "output: ${c.output.zzz}\n")).toContain(
    "reads unknown field 'zzz' on record",
  );
  expect(failure(PARTIAL_CHILD + "output: ${c.output.a.x}\n")).toContain(
    "reference ${c.output.a.x} reads field 'x' on type str",
  );
  load(PARTIAL_CHILD + "output: ${c.output.b.anything}\n"); // opaque field
});

test("a node assert on a partial record output checks typed fields", () => {
  const text =
    PARTIAL_CHILD.replace(
      "    input:\n      s: ${input.s}\n",
      "    input:\n      s: ${input.s}\n    asserts:\n      - output.a.x == 1\n",
    ) + "output: ${c.output.a}\n";
  expect(failure(text)).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'c' assert: reference ${output.a.x} reads field 'x' on type str, which has no fields",
  );
});

test("a node assert on an exact record output checks typed fields", () => {
  const text = flow("{a: str}", "output: ${step.output.a}\n").replace(
    "    code: return None\n",
    "    code: return None\n    asserts:\n      - output.a.b == 1\n",
  );
  expect(failure(text)).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'step' assert: reference ${output.a.b} reads field 'b' on type str, which has no fields",
  );
});

test.each([
  ["object", "${step.output.anything.deeper}", ""], // opaque
  ["Optional(Rec)", "${step.output?.a}", "typedefs:\n  Rec: {a: str}\n"],
  ["{a: str, b: object}", "${step.output.b.anything}", ""], // object-typed field
  ["list(str)", "${step.output}", ""], // whole value
])("reads the type allows still load: %s %s", (stepOutput, ref, preamble) => {
  load(flow(stepOutput, `output:\n  o: ${ref}\n`, preamble));
});

test("untyped output reads still load", () => {
  load(
    "id: p\nname: p\ninput:\n  s: !str\n" +
      "nodes:\n" +
      "  step:\n" +
      "    kind: code\n" +
      "    input:\n      x: ${input.s}\n" +
      "    code: \"return {'a': 1}\"\n" +
      "output:\n  o: ${step.output.anything.deeper}\n",
  );
});

/** A flow whose CODE node `step` binds `v` to `ref`, over inputs of each kind. */
function inputFlow(ref: string): string {
  return (
    "id: p\nname: p\n" +
    "typedefs:\n  R: {a: str}\n" +
    "input:\n  s: !str\n  r: !R\n  tags: !list(str)\n  o: !object\n  d: !file\n" +
    "nodes:\n" +
    "  step:\n" +
    "    kind: code\n" +
    "    input:\n" +
    `      v: ${ref}\n` +
    "    output: str\n" +
    "    code: return 'x'\n" +
    "output: ${step.output}\n"
  );
}

test.each([
  ["${input.s.first}", "reads field 'first' on type str, which has no fields"],
  ["${input.tags.first}", "reads field 'first' on type list(str), which has no fields"],
  ["${input.r.b}", "reads unknown field 'b' on record"],
  ["${input.r.a.b}", "reads field 'b' on type str, which has no fields"],
])("flow input field reads are checked: %s", (ref, expected) => {
  expect(failure(inputFlow(ref))).toContain(`node 'step' input 'v' from: reference ${ref} ${expected}`);
});

test.each([["${input.r.a}"], ["${input.o.any.depth}"], ["${input.s}"], ["${input.d.uri}"]])(
  "valid flow input reads still load: %s",
  (ref) => {
    load(inputFlow(ref));
  },
);

test("defs body input field reads are checked", () => {
  // A named `defs:` body's own typed inputs are checked.
  const text =
    "id: p\nname: p\ninput:\n  s: !str\n" +
    "defs:\n" +
    "  child:\n" +
    "    input:\n      s: !str\n" +
    "    nodes:\n" +
    "      step:\n" +
    "        kind: code\n" +
    "        input:\n          v: ${input.s.first}\n" +
    "        output: str\n" +
    "        code: return 'x'\n" +
    "    output: ${step.output}\n" +
    "nodes:\n" +
    "  c:\n" +
    "    kind: call\n" +
    "    call: child\n" +
    "    input:\n      s: ${input.s}\n" +
    "output: ${c.output}\n";
  expect(failure(text)).toContain("reference ${input.s.first} reads field 'first' on type str");
});

test("inline body input field reads are checked", () => {
  const text =
    "id: p\nname: p\ninput:\n  s: !str\n" +
    "nodes:\n" +
    "  c:\n" +
    "    kind: call\n" +
    "    input:\n      v: ${input.s}\n" +
    "    nodes:\n" +
    "      inner:\n" +
    "        kind: code\n" +
    "        input:\n          w: ${input.v.z}\n" +
    "        output: str\n" +
    "        code: return 'x'\n" +
    "    output: ${inner.output}\n" +
    "output: ${c.output}\n";
  expect(failure(text)).toContain("reference ${input.v.z} reads field 'z' on type str");
});

/** A flow whose MAP `m` calls `each` over `xs: !<overInput>`, binding `v` to `itemRef`. */
function mapFlow(overInput: string, itemRef: string): string {
  return (
    "id: p\nname: p\n" +
    "typedefs:\n  Rec: {a: str}\n" +
    `input:\n  xs: !${overInput}\n` +
    EACH_DEFS +
    "nodes:\n" +
    "  m:\n" +
    "    kind: map\n" +
    "    call: each\n" +
    "    over: ${input.xs}\n" +
    `    input:\n      v: ${itemRef}\n` +
    "output: ${m.output}\n"
  );
}

test.each([
  ["list(str)", "${item.a}", "reads field 'a' on type str, which has no fields"],
  ["list(Rec)", "${item.b}", "reads unknown field 'b' on record"],
  ["Optional(list(Rec))", "${item.b}", "reads unknown field 'b' on record"],
])("map item field reads are checked: %s %s", (overInput, itemRef, expected) => {
  expect(failure(mapFlow(overInput, itemRef))).toContain(
    `node 'm' map input 'v' from: reference ${itemRef} ${expected}`,
  );
});

test.each([
  ["list(Rec)", "${item.a}"],
  ["list(str)", "${item}"],
  ["list(object)", "${item.anything}"],
  ["list(Any)", "${item.anything}"],
  ["Any", "${item.anything}"], // over: not a proven list
])("map item reads the type allows still load: %s %s", (overInput, itemRef) => {
  load(mapFlow(overInput, itemRef));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  ...[
    "field_on_str_in_a_case_on",
    "field_on_str_in_a_case_when",
    "field_on_str_in_a_flow_assert",
    "field_on_str_in_a_flow_output",
    "field_on_str_in_a_node_binding",
    "field_on_str_in_a_wait_until",
    "flow_input_reads_an_unknown_field",
    "inline_body_input_field_reads_are_checked",
    "node_assert_reads_a_field_of_a_str_field",
    "undeclared_record_field_keeps_its_message",
  ].map((name) => ({ id: `load-core/compose_field_reads__${name}`, call: assembled })),
  ...["field_on_str_in_a_map_over", "map_item_reads_an_unknown_field"].map((name) => ({
    id: `load-core/compose_field_reads__${name}`,
    call: assembled,
  })),
])("assemble: $id", runCase);

/** A flow whose `use` binds `${a.output.k}` from `a`, which runs only when `input.go` holds. */
const FLOW_WITH_A =
  "id: f\nname: f\n" +
  "typedefs:\n  Rec: {k: str}\n" +
  "input:\n  go: !bool\n" +
  "nodes:\n" +
  "  gate:\n    kind: case\n    cases:\n      - when: ${input.go}\n        then: a\n" +
  "  a:\n    kind: code\n    output: Rec\n    code: \"return {'k': 'v'}\"\n" +
  "  use:\n    kind: code\n    input:\n      v: ${a.output.k}\n    code: \"return inputs['v']\"\n" +
  "output: ${use.output}\n";

test("a ?. read keeps its edge, co-skip and field check", () => {
  const text = FLOW_WITH_A.replace("${a.output.k}", "${a.output?.k}");
  const { flow } = assembleText(text);
  expect(flow.compiled.edges.some((edge) => edge.from === "a" && edge.to === "use" && !edge.optional)).toBe(true);
  expect(failure(text.replace("a.output?.k", "a.output?.nope"))).toContain("nope");
});

test.each([
  ["${a?.output.k}", "a?.output.k", "a.output?.k"],
  ["${a?.output?.k}", "a?.output?.k", "a.output?.k"],
  ["${input?.go}", "input?.go", "input.go"],
])("?. inside a root is a load error (%s)", (source, read, fix) => {
  const message = failure(FLOW_WITH_A.replace("${a.output.k}", source));
  expect(message).toStartWith("flow has ?. inside a reference root:");
  expect(message).toContain(`reference \${${read}} puts ?. inside its root; ?. goes after the root, as in \${${fix}}`);
});

test("an unknown field read through ?. is named as written", () => {
  expect(failure(FLOW_WITH_A.replace("${a.output.k}", "${a.output?.nope}"))).toContain(
    "reference ${a.output?.nope} reads unknown field 'nope'",
  );
  expect(failure(FLOW_WITH_A.replace("${a.output.k}", "'${a.output?.k.x} y'"))).toContain(
    "reference ${a.output?.k.x} reads field 'x' on type str",
  );
});

test("?. after a root loads, and a comprehension variable is a one-name root", () => {
  assembleText(FLOW_WITH_A.replace("${a.output.k}", "${a.output?.k}"));
  assembleText(FLOW_WITH_A.replace("${a.output.k}", "${[x?.k for x in [a.output]]}"));
});

// The nullable field-read check over one expression (`nullableReadErrors`): a `.` step whose
// receiver may be null is an error, a `?.` step is not.

const str = Type.scalar(ValueKind.STRING);
const Rec = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["k", str],
    ["o", str.replace({ nullable: true })],
  ]),
  required: new Set(["k", "o"]),
});
const Nested = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([["r", Rec.replace({ nullable: true })]]),
  required: new Set(["r"]),
});
/** One root `x` of type `typ`, and an untyped root `u`. */
const roots = (typ: Type, nullable = false): ReadRoots => ({
  contracts: (root) => (root === "x" ? [new ValueContract(typ)] : root === "u" ? [null] : undefined),
  baseContract: (root) => (root === "x" ? new ValueContract(typ) : root === "u" ? null : undefined),
  rootLength: () => 1,
  nullable: () => nullable,
  why: () => "it is Optional",
  untyped: (root) => root === "u",
  routing: () => null,
});
const errors = (text: string, r: ReadRoots) => nullableReadErrors(parseCondition(text), r, new Map());

test.each([
  ["x.k", Rec.replace({ nullable: true }), "${x.k} reads field 'k' of x, which may be null (it is Optional)"],
  ["x.r.k", Nested, "${x.r.k} reads field 'k' of x.r, which may be null (field 'r' is Optional)"],
  // ?. short-circuits only a null x; x exists and x.r may be null, so .k follows rule 1.
  ["x?.r.k", Nested, "${x?.r.k} reads field 'k' of x?.r, which may be null (field 'r' is Optional)"],
])("a . step on a value that may be null is an error (%s)", (text, typ, message) => {
  expect(errors(text, roots(typ))[0]).toStartWith(message);
});

test.each(["x?.k", "x != null and x.k", "x == null or x.k", "x.r?.k", "x?.r?.k == null", "u.k.m"])(
  "no error (%s)",
  (text) => {
    const typ = text.includes(".r") ? Nested : Rec.replace({ nullable: true });
    expect(errors(text, roots(typ))).toEqual([]);
  },
);

test("a root that may be skipped needs ?. whatever its type", () => {
  expect(errors("x.k", roots(Rec, true))[0]).toBe(
    "${x.k} reads field 'k' of x, which may be null (it is Optional); write x?.k, or check x != null first",
  );
});

test("a read is an error only when every alternative rejects it", () => {
  const two: ReadRoots = {
    ...roots(Rec),
    contracts: () => [new ValueContract(Rec.replace({ nullable: true })), new ValueContract(Rec)],
  };
  expect(errors("x.k", two)).toEqual([]);
});

// Design 2.6.6: a skipped producer is a missing root whatever its type.
test.each([
  ["no contract", null],
  ["Any", new ValueContract(Type.scalar(ValueKind.ANY))],
  ["Optional(Any)", new ValueContract(Type.scalar(ValueKind.ANY).replace({ nullable: true }))],
])("a root that may be skipped needs ?. even when untyped (%s)", (_name, contract) => {
  const skippable: ReadRoots = {
    ...roots(Rec, true),
    contracts: (root) => (root === "x" ? [contract] : undefined),
    untyped: () => true,
  };
  expect(errors("x.k.m", skippable)[0]).toStartWith("${x.k} reads field 'k' of x, which may be null (it is Optional)");
  expect(errors("x?.k.m", skippable)).toEqual([]);
  expect(errors("x != null and x.k.m", skippable)).toEqual([]);
});

test("an untyped root that can't be skipped is never flagged", () => {
  const optionalAny: ReadRoots = {
    ...roots(Rec),
    contracts: () => [new ValueContract(Type.scalar(ValueKind.ANY).replace({ nullable: true }))],
    untyped: () => true,
  };
  expect(errors("x.k.m", optionalAny)).toEqual([]);
  expect(errors("u.k.m", roots(Rec))).toEqual([]);
});

test("an arithmetic operand gets the whole fix in one message", () => {
  const message = errors("x.k + 1 > 1", roots(Rec, true))[0]!;
  expect(message).toEndWith("write (x?.k :- <default>)");
  expect(message).not.toContain("check x != null");
  expect(errors("-(x.k) > 1", roots(Rec, true))[0]).toEndWith("write (x?.k :- <default>)");
});

test.each(["(x.k :- 0) + 1 > 1", "(x.k | 0) + 1 > 1"])("a defaulted arithmetic operand needs only ?. (%s)", (text) => {
  expect(errors(text, roots(Rec, true))[0]).toEndWith("which may be null (it is Optional); write x?.k");
});

test("a :- null arithmetic operand still needs a default", () => {
  expect(errors("(x.k :- null) + 1 > 1", roots(Rec, true))[0]).toEndWith("write (x?.k :- <default>)");
});

const types: TypeRegistry = new Map([
  [
    "T",
    {
      kind: "record",
      fields: new Map([
        ["f", "Optional(str)"],
        ["g", "str"],
      ]),
    },
  ],
]);
const T = typeFor("T", types);

test("an untyped value narrowed by isinstance needs ?. for a field T doesn't require", () => {
  const r = roots(Rec);
  expect(nullableReadErrors(parseCondition("isinstance(u, T) and u.f == 'a'"), r, types)[0]).toBe(
    "${u.f} reads field 'f', which T doesn't require, of u, which is untyped and only narrowed to T; " +
      "write u?.f, or type u as T where it is bound (`output: T` or `!T`)",
  );
  expect(nullableReadErrors(parseCondition("isinstance(u, T) and u?.f == 'a'"), r, types)).toEqual([]);
  expect(nullableReadErrors(parseCondition("isinstance(u, T) and u.g == 'a'"), r, types)).toEqual([]);
});

test("a root a CASE branch narrowed is untyped by its base, not its routed view", () => {
  // The routed view types m.output as T; its un-narrowed base is untyped (Any).
  const r: ReadRoots = {
    ...roots(Rec),
    contracts: (root) => (root === "m.output" ? [new ValueContract(T)] : undefined),
    rootLength: () => 2,
    untyped: (root) => root === "m.output",
  };
  expect(nullableReadErrors(parseCondition("m.output.f == 'a'"), r, types)[0]).toContain(
    "which is untyped and only narrowed to T; write m.output?.f",
  );
  expect(nullableReadErrors(parseCondition("m.output.g == 'a'"), r, types)).toEqual([]);
});

const TextTypes: TypeRegistry = new Map([
  ["Inner", { kind: "record", fields: new Map([["y", "str"]]) }],
  [
    "Text",
    {
      kind: "record",
      fields: new Map([
        ["body", "str"],
        ["t", "Optional(Inner)"],
      ]),
    },
  ],
]);
const Plain = new Type({ kind: ValueKind.OBJECT, fields: new Map([["k", str]]), required: new Set(["k"]) });

test("a read under a check that can never hold is not checked", () => {
  // No Plain value fits Text (it lacks the required body), so the isinstance is never true.
  expect(nullableReadErrors(parseCondition("isinstance(x, Text) and x.t.y == 'a'"), roots(Plain), TextTypes)).toEqual([]);
});

test("an impossible isinstance next to a comprehension doesn't break checkedRefs", () => {
  const tree = parseCondition("isinstance(x, Text) and len([m.k for m in xs]) > 0");
  expect(() => nullableReadErrors(tree, roots(Plain), TextTypes)).not.toThrow();
});

// Guards apply at every prefix of a read, and owner ruling 1 holds at any depth.

const NestedTypes: TypeRegistry = new Map([
  ["Inner", { kind: "record", fields: new Map([["y", "str"]]) }],
  ["Holder", { kind: "record", fields: new Map([["k", "Optional(Inner)"]]) }],
  ["W", { kind: "record", fields: new Map([["payload", "Any"]]) }],
  [
    "T",
    {
      kind: "record",
      fields: new Map([
        ["f", "Optional(str)"],
        ["g", "str"],
      ]),
    },
  ],
]);

/** One root `x` of type `typ`, untyped wherever its un-narrowed value is. */
const typedRoots = (typ: Type): ReadRoots => ({
  ...roots(typ),
  untyped: (root, fields) => root !== "x" || untypedAt(new ValueContract(typ), fields),
});

test.each([
  "x != null and x.k != null and x.k.y == 'a'",
  "x == null or x.k == null or x.k.y == 'a'",
  "x != null and isinstance(x.k, Inner) and x.k.y == 'a'",
  "x?.k != null and x.k.y == 'a'",
])("a guard on each prefix lifts each step (%s)", (text) => {
  const holder = typeFor("Optional(Holder)", NestedTypes);
  expect(nullableReadErrors(parseCondition(text), typedRoots(holder), NestedTypes)).toEqual([]);
});

test("a guard on the field alone still leaves the root step flagged", () => {
  const holder = typeFor("Optional(Holder)", NestedTypes);
  expect(nullableReadErrors(parseCondition("x.k != null and x.k.y == 'a'"), typedRoots(holder), NestedTypes)[0]).toStartWith(
    "${x.k} reads field 'k' of x, which may be null",
  );
});

test("an untyped field of a typed root narrowed to T needs ?. for a field T doesn't require", () => {
  const w = typeFor("W", NestedTypes);
  const errors = (text: string) => nullableReadErrors(parseCondition(text), typedRoots(w), NestedTypes);
  expect(errors("isinstance(x.payload, T) and x.payload.f == 'a'")[0]).toBe(
    "${x.payload.f} reads field 'f', which T doesn't require, of x.payload, which is untyped and only narrowed to T; " +
      "write x.payload?.f, or declare x.payload as T in the type that holds it",
  );
  expect(errors("isinstance(x.payload, T) and x.payload?.f == 'a'")).toEqual([]);
  expect(errors("isinstance(x.payload, T) and x.payload.g == 'a'")).toEqual([]);
});

/** A flow with `mk` an `Optional(Rec)` producer, `w` a `W` producer and `input.a` an `Optional(Rec)`. */
const NESTED_FLOW = `id: f
name: f
typedefs:
  Inner: {x: str}
  Rec: {k: Optional(Inner), m: str}
  T: {f: Optional(str), g: str}
  W: {payload: Any}
input:
  s: !str
  a: !Optional(Rec)
nodes:
  mk:
    kind: code
    input:
      s: \${input.s}
    output: Optional(Rec)
    code: |
      return None
  w:
    kind: code
    input:
      s: \${input.s}
    output: W
    code: |
      return {"payload": None}
NODES
output: OUTPUT
`;

/** Return `NESTED_FLOW` with `nodes` appended and `output` as the flow output. */
const nestedFlow = (nodes: string, output = "${mk.output}") =>
  NESTED_FLOW.replace("NODES", nodes).replace("OUTPUT", output);

/** A CODE node `id` with the input `v: value` and an optional assert. */
const codeNode = (id: string, value: string, assert = "") =>
  `  ${id}:
    kind: code
    input:
      v: ${value}
    output: Any${assert === "" ? "" : `\n    asserts:\n      - ${assert}`}
    code: |
      return 1`;

/** A searched CASE `route` with one `when:` to `go` (reading `read`), else `other`. */
const caseNodes = (when: string, read: string) =>
  `  route:
    kind: case
    cases:
      - when: ${when}
        then: go
    else: other
${codeNode("go", read)}
${codeNode("other", "${input.s}")}`;

test.each([
  ["binding", nestedFlow(codeNode("use", '${mk.output != null and mk.output.k != null and mk.output.k.x == "a"}'))],
  ["binding, == null or", nestedFlow(codeNode("use", '${mk.output == null or mk.output.k == null or mk.output.k.x == "a"}'))],
  ["binding, ?. guard", nestedFlow(codeNode("use", '${mk.output?.k != null and mk.output.k.x == "a"}'))],
  ["node assert", nestedFlow(codeNode("use", "${mk.output}", '${v != null and v.k != null and v.k.x == "a"}'))],
  ["flow output", nestedFlow("", '${mk.output != null and mk.output.k != null and mk.output.k.x == "a"}')],
  [
    "searched when",
    nestedFlow(caseNodes('mk.output != null and mk.output.k != null and mk.output.k.x == "a"', "${input.s}"), "${go.output | other.output}"),
  ],
  ["input, inline", nestedFlow(codeNode("use", '${input.a?.k != null and input.a.m == "p"}'))],
  [
    "input, CASE branch",
    nestedFlow(caseNodes('input.a?.k != null and input.a.m == "p"', "${input.a.m}"), "${go.output | other.output}"),
  ],
])("nested null guards load (%s)", (_label, text) => {
  load(text);
});

const PAYLOAD_ERROR =
  "${w.output.payload.f} reads field 'f', which T doesn't require, of w.output.payload, which is untyped and only " +
  "narrowed to T; write w.output.payload?.f, or declare w.output.payload as T in the type that holds it";

test.each([
  ["inline", nestedFlow(codeNode("use", '${isinstance(w.output.payload, T) and w.output.payload.f == "a"}'))],
  ["CASE branch", nestedFlow(caseNodes("isinstance(w.output.payload, T)", "${w.output.payload.f}"), "${go.output | other.output}")],
])("an untyped field narrowed to T needs ?. for a field T doesn't require (%s)", (_label, text) => {
  expect(failure(text)).toContain(PAYLOAD_ERROR);
  load(text.replace("w.output.payload.f", "w.output.payload?.f"));
  load(text.replace("w.output.payload.f", "w.output.payload.g"));
});

test("a case value read's error spells the read as written", () => {
  const text = `id: f
name: f
typedefs:
  Inner: {x: str}
  Rec: {opt: Optional(Inner)}
input:
  s: !str
nodes:
  gate:
    kind: case
    cases:
      - when: input.s == "a"
        then: pro
    else: con
  pro:
    kind: code
    output: Rec
    code: |
      return {"opt": None}
  con:
    kind: code
    output: Rec
    code: |
      return {"opt": None}
  use:
    kind: code
    input:
      v: \${gate.output.opt.x}
    output: Any
    code: |
      return 1
output: \${use.output}
`;
  expect(failure(text)).toContain(
    "node 'use' input 'v': ${gate.output.opt.x} reads field 'x' of gate.output.opt, which may be null " +
      "(field 'opt' is Optional); write gate.output.opt?.x, or check gate.output.opt != null first",
  );
  expect(failure(text.replace("v: ${gate.output.opt.x}", "v: ${input.s}").replace("${use.output}", "${gate.output.opt.x}"))).toContain(
    "output: ${gate.output.opt.x} reads field 'x' of gate.output.opt",
  );
});

test("a read with an unknown field gets no nullable-read message too", () => {
  const holder = typeFor("Holder", NestedTypes);
  const errors = (text: string) =>
    fieldReadErrors(parseCondition(text), new Map([["v", [new ValueContract(holder)]]]), NestedTypes).map(
      ({ message }) => message,
    );
  expect(errors("v.k.zz == 'a'")).toEqual(["reference ${v.k.zz} reads unknown field 'zz' on record"]);
  expect(errors("v.k.y.zz == 'a'")).toEqual(["reference ${v.k.y.zz} reads field 'zz' on type str, which has no fields"]);
  expect(errors("v.k.zz == 'a' and v.k.y == 'b'")).toEqual([
    "reference ${v.k.zz} reads unknown field 'zz' on record",
    "${v.k.y} reads field 'y' of v.k, which may be null (field 'k' is Optional); write v.k?.y, or check v.k != null first",
  ]);
});

test("an inline body's output error names the node that owns the body", () => {
  const text = `id: f
name: f
typedefs:
  Inner: {x: str}
  Rec: {k: Optional(Inner)}
input:
  s: !str
nodes:
  lp:
    kind: loop
    times: 1
    input:
      value: 0
    nodes:
      mk:
        kind: code
        output: Rec
        code: |
          return {"k": None}
    output: \${mk.output.k.x}
output: \${lp.output}
`;
  expect(failure(text)).toContain(
    "inline body of node 'lp' output: ${mk.output.k.x} reads field 'x' of mk.output.k, which may be null",
  );
});

// ---------- comprehension variables ----------

/** A flow whose `use` node binds `v` to `binding`, over inputs of Msg lists and Messages. */
function variableFlow(binding: string): string {
  return (
    "id: p\nname: p\n" +
    "typedefs:\n" +
    "  Msg: {role: str, text: str, opt: Optional(Inner)}\n" +
    "  Inner: {x: str}\n" +
    "  T: {f: Optional(str)}\n" +
    "input:\n" +
    "  ms: !list(Msg)\n" +
    "  maybe: !list(Optional(Msg))\n" +
    "  one: !Optional(Inner)\n" +
    "  hist: !list(Message)\n" +
    "  om: !Optional(list(Msg))\n" +
    "nodes:\n" +
    "  mk:\n" +
    "    kind: code\n" +
    "    code: return []\n" +
    "  use:\n" +
    "    kind: code\n" +
    "    input:\n" +
    `      v: ${binding}\n` +
    "      after: ${mk.output}\n" +
    "    output: Any\n" +
    "    code: return 1\n" +
    "output: ${use.output}\n"
  );
}

test.each([
  ["${[m.opt for m in input.ms]}"],
  ["${[m.opt?.x for m in input.ms]}"],
  ["${[m.text for m in input.maybe if m != null]}"],
  ["${[input.one.x for m in input.ms if input.one != null]}"],
  [`'\${[p.text for m in input.hist for p in m.parts if p.type == "text"]}'`],
  ["${[m.text for m in (input.om :- [])]}"],
])("a read through a comprehension variable loads: %s", (binding) => {
  expect(() => load(variableFlow(binding))).not.toThrow();
});

test.each([
  ["${[m.txt for m in input.ms]}", "compose.load", "reference ${m.txt} reads unknown field 'txt' on record"],
  ["${[m.opt.x for m in input.ms]}", NULLABLE_FIELD_READ, "${m.opt.x} reads field 'x' of m.opt, which may be null (field 'opt' is Optional)"],
  ["${[m.text for m in input.maybe]}", NULLABLE_FIELD_READ, "${m.text} reads field 'text' of m, which may be null (it is Optional)"],
  ["${[input.one.x for m in input.ms]}", NULLABLE_FIELD_READ, "which may be null (it is Optional)"],
  ["${[p.text for m in input.hist for p in m.parts]}", "compose.load", "reference ${p.text} reads field 'text'"],
  ["${[m.k for m in (input.om :- [])]}", "compose.load", "reference ${m.k} reads unknown field 'k' on record"],
  ["${[p.f for p in mk.output if isinstance(p, T)]}", NULLABLE_FIELD_READ, "only narrowed to T; write p?.f"],
])("a read through a comprehension variable is checked: %s", (binding, code, message) => {
  const error = loadError(() => load(variableFlow(binding)));
  expect(error.message).toStartWith("node 'use' input 'v': ");
  expect(error.message).toContain(message);
  expect(error.diagnostic.code).toBe(code);
});

/** A LOOP that carries `ms: list(Msg)`, driven by `predicate`. */
function variableLoop(predicate: string): string {
  return `id: p
name: p
typedefs:
  Msg: {role: str, text: str}
defs:
  body:
    input:
      ms: !list(Msg)
    nodes:
      keep:
        kind: code
        input:
          ms: \${input.ms}
        output:
          ms: list(Msg)
        code: "return ({ms: inputs['ms']})"
    output: \${keep.output}
input:
  ms: !list(Msg)
nodes:
  drive:
    kind: loop
    call: body
    while: ${predicate}
    max: 3
    input:
      ms: \${input.ms}
output: \${drive.output}
`;
}

test("a LOOP predicate checks reads through a comprehension variable", () => {
  expect(failure(variableLoop("${len([m.txt for m in ms]) < 3}"))).toContain(
    "loop node 'drive' `while:` reference ${m.txt} reads unknown field 'txt' on record",
  );
  expect(() => load(variableLoop("${len([m.text for m in ms]) < 3}"))).not.toThrow();
});

// ---------- isinstance on an untyped value, after a null check ----------

/** A flow whose `use` node binds `v` to `binding`; `u.output` is `Any`, `mk.output` an `Optional(W)` with `u: Any`. */
function untypedFlow(binding: string): string {
  return (
    "id: p\nname: p\n" +
    "typedefs:\n  T: {f: Optional(str), g: str}\n  W: {u: Any}\n" +
    "input:\n  s: !str\n" +
    "nodes:\n" +
    "  u:\n    kind: code\n    input:\n      s: ${input.s}\n    output: Any\n    code: return 1\n" +
    "  mk:\n    kind: code\n    input:\n      s: ${input.s}\n    output: Optional(W)\n    code: return None\n" +
    `  use:\n    kind: code\n    input:\n      v: ${binding}\n    output: Any\n    code: return 1\n` +
    "output: ${use.output}\n"
  );
}

test.each([
  ["${u.output != null and isinstance(u.output, T) and u.output.g}"],
  ["${u.output != null and isinstance(u.output, T) and u.output?.f}"],
  ["${mk.output != null and isinstance(mk.output.u, T) and mk.output.u.g}"],
])("an isinstance after a null check narrows an untyped value: %s loads", (binding) => {
  expect(() => load(untypedFlow(binding))).not.toThrow();
});

test.each([
  ["${u.output != null and isinstance(u.output, T) and u.output.f}", NULLABLE_FIELD_READ, "only narrowed to T; write u.output?.f"],
  ["${mk.output != null and isinstance(mk.output.u, T) and mk.output.u.f}", NULLABLE_FIELD_READ, "only narrowed to T; write mk.output.u?.f"],
  ["${isinstance(u.output, T) and u.output.zz}", "compose.load", "reference ${u.output.zz} reads field 'zz', which T doesn't have; u.output is T here"],
  ["${u.output != null and isinstance(u.output, T) and u.output.zz}", "compose.load", "reads field 'zz', which T doesn't have"],
  ["${mk.output != null and isinstance(mk.output.u, T) and mk.output.u.zz}", "compose.load", "reference ${mk.output.u.zz} reads field 'zz', which T doesn't have"],
])("an isinstance after a null check narrows an untyped value: %s", (binding, code, message) => {
  const error = loadError(() => load(untypedFlow(binding)));
  expect(error.message).toContain(message);
  expect(error.diagnostic.code).toBe(code);
});
