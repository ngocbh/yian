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

import { runFlow } from "../../src/run.ts";
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
  // A `when:` reading a field of an omitted `Optional(Rec)` input takes the else branch.
  const text =
    "id: p\nname: p\n" +
    "typedefs:\n  Rec: {a: str}\n" +
    "input:\n  s: !str\n  maybe: !Optional(Rec)\n" +
    "nodes:\n" +
    "  route:\n" +
    "    kind: case\n" +
    "    cases:\n" +
    "      - when: ${input.maybe.a == 'x'}\n" +
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
  ["Optional(Rec)", "${step.output.a}", "typedefs:\n  Rec: {a: str}\n"],
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
