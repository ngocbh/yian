/**
 * Load-time field-read checks in node asserts, and the type-table pass's builtin-result reads: a
 * port of the node-assert rows of `tests/compose/test_field_read_sites.py`:
 * `::test_node_assert_reads_unknown_input_field` (2),
 * `::test_node_assert_field_error_is_reported_once_per_ref`,
 * `::test_call_node_assert_reads_a_field_its_binding_lacks` (2),
 * `::test_loop_assert_reads_a_field_neither_seed_nor_carried_has` (2),
 * `::test_node_assert_literal_input_field_read`, `::test_node_assert_open_input_field_reads_load` (2),
 * `::test_node_assert_reads_the_narrowed_member_inside_a_branch`, the `untagged-assert` and
 * `shared-assert` rows of `::test_isinstance_branch_reads_the_narrowed_member`,
 * `::test_node_assert_reads_are_narrowed_by_the_checks_before_them` (9),
 * `::test_node_assert_reads_the_checks_do_not_cover_are_rejected` (3),
 * `::test_node_assert_read_under_two_guards_is_checked_under_each`,
 * `::test_agent_node_assert_reads_are_narrowed`, `::test_node_assert_output_reads_are_narrowed`,
 * `::test_partial_union_node_assert_reads_the_narrowed_member`,
 * `::test_impossible_node_assert_check_is_located_at_the_assert`,
 * `::test_call_node_assert_tag_checks_narrow_its_local`, `::test_assert_reading_many_typed_inputs_loads`,
 * `::test_node_assert_read_after_a_reported_check_is_not_checked` (2),
 * `::test_node_assert_read_after_an_always_true_negation_is_not_checked`,
 * `::test_node_assert_read_of_one_local_after_a_reported_check_on_another_is_not_checked`,
 * `::test_node_assert_check_impossible_under_every_alternative_is_reported_once`,
 * `::test_node_assert_read_dead_under_only_some_alternatives_is_checked`,
 * `::test_node_assert_impossible_output_check_is_reported` (3),
 * `::test_node_assert_output_read_after_a_reported_check_is_not_checked`,
 * `::test_node_assert_over_many_locals_with_two_alternatives_loads`,
 * `::test_call_node_assert_member_only_hint_compares_the_tag`,
 * `::test_call_node_assert_member_only_hint_on_an_untagged_union_names_a_case`,
 * `::test_node_assert_reads_a_field_on_a_builtin_result`, the `assert` row of
 * `::test_builtin_results_without_a_field_read_load`,
 * `::test_unparsable_condition_keeps_its_located_parse_error` and
 * `::test_loop_while_reads_a_field_on_a_builtin_result` (the type-table pass); and the load halves of
 * `::test_loop_assert_reads_the_carried_field` (2), `::test_node_assert_known_input_field_loads_and_runs`,
 * `::test_loop_node_assert_tag_checks_narrow_its_local` (with its two load errors) and
 * `::test_partial_record_local_root_reads` (with its load error).
 *
 * The flows assemble with `assembleText` where Python calls `load_flow`. Each `defs:` entry a CALL
 * or LOOP calls is built by hand (`handChild`, since def bodies load in T3-22): `child` with the
 * def's `v` parameter, `body` with `st: !Rec3` returning `{st: Rec3}`, and the partial `child`
 * with a typed `a` and an opaque `b`. Inline LOOP bodies assemble through the harness's
 * private-child registry.
 *
 * Changed: the CODE bodies are TS's, as the harvested cases spell them. A compared tag and its
 * close match are spelled `"byu"` / `did you mean "buy"?` (Python `'byu'` / `did you mean 'buy'?`),
 * and the cycle's nodes `["b","c"]` (Python `['b', 'c']`).
 *
 * TS-only: the harvested node-assert cases and the type-table pass's builtin-result case, pinned
 * at the end.
 *
 * Later (T3-20, `lowerFlowAsserts`): the flow-assert rows
 * `test_flow_assert_reads_are_narrowed_by_the_checks_before_them`,
 * `test_flow_post_assert_producer_reads_are_narrowed`, `test_flow_assert_read_before_its_check_is_rejected`,
 * `test_impossible_flow_assert_check_is_located_at_the_assert`,
 * `test_flow_assert_check_impossible_even_on_null_is_reported`,
 * `test_flow_assert_read_error_names_the_declared_type`,
 * `test_flow_assert_shorthand_check_is_reported_as_written`, `test_flow_assert_reads_unknown_input_field`,
 * `test_flow_assert_readable_input_fields_load`, `test_flow_assert_reads_a_field_on_a_builtin_result`,
 * and the load halves of `test_flow_assert_on_a_skippable_producer_reads_null` and
 * `test_flow_assert_known_input_field_loads_and_runs`.
 *
 * Later (T3-18, the reference validator's prompt, binding, `case when`, flow output and scanned
 * sites): `test_agent_prompt_reads_unknown_field`, `test_agent_system_prompt_reads_unknown_field`,
 * `test_typed_input_field_read_errors_only_when_every_alternative_rejects`,
 * `test_typed_union_input_reads_the_narrowed_member_inside_a_branch`,
 * `test_typed_record_binding_reads_its_tagged_member`, `test_optional_binding_field_reads`,
 * `test_adaptive_brief_field_read_is_reported_on_the_gate`, `test_prompt_inside_a_def_reads_unknown_field`,
 * `test_literal_input_field_read`, `test_null_literal_input_field_read`,
 * `test_field_read_inside_a_builtin_argument`, `test_agent_variable_field_read`,
 * `test_human_input_prompt_field_read`, `test_human_input_question_field_read`,
 * `test_prompt_field_error_is_reported_once_per_ref`, `test_readable_fields_still_load`,
 * `test_narrowed_input_reads_the_branch_member`, `test_defaulted_union_binding_hint_names_the_bound_source`,
 * `test_transformed_union_binding_hint_names_no_bound_source`, the prompt rows of
 * `test_isinstance_branch_reads_the_narrowed_member`, `test_binding_reads_a_field_on_a_builtin_result`,
 * `test_flow_output_reads_a_field_on_a_builtin_result`, `test_prompt_reads_a_field_on_a_builtin_result`,
 * `test_case_when_reads_a_field_on_a_builtin_result`,
 * `test_scanned_sites_read_a_field_on_a_builtin_result_once`, the `upper` and `join` rows of
 * `test_builtin_results_without_a_field_read_load`, and `test_host_builtin_field_reads_load` (its
 * binding reads a field of the host builtin's result); and their cases.
 *
 * Later (T4): `test_guarded_node_assert_runs`, `test_guarded_flow_assert_runs`, and the run halves
 * of the rows above.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  type ChildFlows,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

const PREAMBLE =
  "id: p\nname: p\n" +
  "typedefs:\n" +
  "  Rec: {a: str}\n" +
  "  Rec3: {a: str, b: Optional(str)}\n" +
  "  Buy: {action: 'Literal(buy)', qty: int}\n" +
  "  Sell: {action: 'Literal(sell)'}\n" +
  "  Order: Union(Buy, Sell)\n";

const INPUTS = "input:\n  s: !str\n  r: !Rec\n";

const STEP =
  "  step:\n" +
  "    kind: code\n" +
  "    input:\n      x: ${input.s}\n" +
  "    output: Rec\n" +
  '    code: "return ({a: \\"x\\"})"\n';

const ASSERT_QTY_HINT =
  "reads field 'qty', which only union members with action buy have; " +
  "check the member first with `isinstance(v, <Type>) and ...`";

const UNION_TYPES =
  "  Short: {action: 'Literal(short)', qty: int}\n" +
  "  Hold: {action: 'Literal(hold)'}\n" +
  "  Trade: Union(Buy, Short, Hold)\n" +
  "  Sized: Union(Buy, Short)\n" +
  "  A: {x: int}\n" +
  "  B: {y: str}\n" +
  "  AB: Union(A, B)\n";

/** Every typedef the flows here declare, for the hand-built children's parameters. */
const REGISTRY: TypeRegistry = readTypedefs({
  Rec: { a: "str" },
  Rec3: { a: "str", b: "Optional(str)" },
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)" },
  Order: "Union(Buy, Sell)",
  A: { x: "int" },
  B: { y: "str" },
  AB: "Union(A, B)",
  Out: { st: "Rec3" },
});

/**
 * Return a flow built from the shared preamble and the given sections: the shared typedefs plus
 * `typedefs` lines, `defs`, the inputs (plus `inputs` lines), node `step`, then `nodes`, `output`
 * and `tail`.
 */
function flow(
  nodes: string,
  output: string,
  options: { typedefs?: string; defs?: string; inputs?: string; tail?: string } = {},
): string {
  const { typedefs = "", defs = "", inputs = "", tail = "" } = options;
  return PREAMBLE + typedefs + defs + INPUTS + inputs + "nodes:\n" + STEP + nodes + `output: ${output}\n` + tail;
}

/** An AGENT node with `inputs`, a quoted `prompt`, and extra YAML lines. */
function agent(
  inputs: Record<string, string>,
  prompt: string,
  options: { nodeId?: string; extra?: string } = {},
): string {
  const { nodeId = "ag", extra = "" } = options;
  let text = `  ${nodeId}:\n    kind: agent\n`;
  const entries = Object.entries(inputs);
  if (entries.length > 0) {
    text += "    input:\n";
    for (const [name, source] of entries) text += `      ${name}: ${source}\n`;
  }
  return text + `    prompt: "${prompt}"\n` + extra;
}

/** Assemble `text`, its `defs:` entries standing in as `children`. */
const load = (text: string, children: ChildFlows = {}) => assembleText(text, "flow.yaml", { children });

/** Return the message of the `LoadError` assembling `text` throws. */
const loadMessage = (text: string, children: ChildFlows = {}) => loadError(() => load(text, children)).message;

/** A CODE node typed `output` reading the flow input `s`. */
function producer(nodeId: string, output: string): string {
  return `  ${nodeId}:\n    kind: code\n    input:\n      s: \${input.s}\n    output: ${output}\n    code: "return null"\n`;
}

/** Producer `d` (Order), a case on its action routing buy to `ag`, sell to `other`. */
function routed(node: string): string {
  return (
    producer("d", "Order") +
    "  route:\n" +
    "    kind: case\n" +
    "    on: ${d.output.action}\n" +
    "    cases:\n" +
    "      - when: buy\n        then: ag\n" +
    "      - when: sell\n        then: other\n" +
    node +
    "  other:\n" +
    "    kind: code\n" +
    "    input:\n      s: ${input.s}\n" +
    "    output: str\n" +
    "    code: return 'x'\n"
  );
}

/** A CODE node returning `'x'` (typed `str`) with `inputs` and `asserts`. */
function code(inputs: Record<string, string>, asserts: readonly string[], nodeId = "c"): string {
  let text = `  ${nodeId}:\n    kind: code\n    input:\n`;
  for (const [name, source] of Object.entries(inputs)) text += `      ${name}: ${source}\n`;
  text += "    output: str\n    code: return 'x'\n    asserts:\n";
  return text + asserts.map((a) => `      - "${a}"\n`).join("");
}

/** The number of times `needle` occurs in `text`. */
const count = (text: string, needle: string) => text.split(needle).length - 1;

/** The 1-based line of `text` holding `needle`. */
const lineOf = (text: string, needle: string) => text.split("\n").findIndex((line) => line.includes(needle)) + 1;

/** A hand-built `child` def taking `v: !<typ>`, whose CODE node `inner` returns `str`. */
function childOf(typ: string): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml(`input:\n  v: !${typ}\n`, REGISTRY),
    producers: { inner: new ValueContract(Type.scalar(ValueKind.STRING)) },
    output: "${inner.output}",
  });
}

/** The `defs:` text of a `child` taking `v: !<typ>`, as Python's flows write it. */
function childText(typ: string): string {
  return (
    "defs:\n" +
    "  child:\n" +
    `    input:\n      v: !${typ}\n` +
    "    nodes:\n" +
    "      inner:\n" +
    "        kind: code\n" +
    "        input:\n          v: ${input.v}\n" +
    "        output: str\n" +
    "        code: return 'x'\n" +
    "    output: ${inner.output}\n"
  );
}

// ---------- node asserts ----------

test.each([
  ["pre", "v.b == 'x'"],
  ["post", "output == 'x' and v.b == 'x'"],
])("a node assert reads an unknown input field: %s", (_id, assertion) => {
  expect(loadMessage(flow(code({ v: "${step.output}" }, [assertion]), "${c.output}"))).toContain(
    "node 'c' assert: reference ${v.b} reads unknown field 'b' on record",
  );
});

test("a node assert field error is reported once per ref", () => {
  const message = loadMessage(flow(code({ v: "${step.output}" }, ["v.b == null or v.b == 'y'"]), "${c.output}"));
  expect(count(message, "reference ${v.b} reads unknown field 'b'")).toBe(1);
});

test.each(["Rec", "Rec3"])("a call node assert reads a field its binding lacks: %s", (childType) => {
  const nodes =
    "  c:\n    kind: call\n    call: child\n" +
    "    input:\n      v: ${step.output}\n" +
    "    asserts:\n      - \"v.b == 'x'\"\n";
  const text = flow(nodes, "${c.output}", { defs: childText(childType) });
  expect(loadMessage(text, { child: childOf(childType) })).toContain(
    "node 'c' assert: reference ${v.b} reads unknown field 'b' on record",
  );
});

const LOOP_NAMED_BODY =
  "defs:\n" +
  "  body:\n" +
  "    input:\n      st: !Rec3\n" +
  "    nodes:\n" +
  "      bump:\n" +
  "        kind: code\n" +
  "        input:\n          st: ${input.st}\n" +
  "        output:\n          st: Rec3\n" +
  '        code: "return ({st: {a: \\"x\\", b: \\"y\\"}})"\n' +
  "    output: ${bump.output}\n";

/** The hand-built `body` def: `st: !Rec3` into `bump`, whose `{st: Rec3}` it returns. */
const BODY = {
  body: handChild({
    inputs: parametersFromYaml("input:\n  st: !Rec3\n", REGISTRY),
    producers: { bump: new ValueContract(typeFor("Out", REGISTRY)) },
    output: "${bump.output}",
  }),
};

/** A LOOP over the named `body` (carries `Rec3`), seeded with `step`'s `Rec`. */
function namedLoop(assertion: string): string {
  const nodes =
    "  drive:\n    kind: loop\n    call: body\n    times: 3\n" +
    "    input:\n      st: ${step.output}\n" +
    `    asserts:\n      - "${assertion}"\n`;
  return flow(nodes, "${drive.output}", { defs: LOOP_NAMED_BODY });
}

/** A one-turn inline LOOP carrying `Rec3`, seeded with `step`'s `Rec`. */
function inlineLoop(assertion: string): string {
  const nodes =
    "  drive:\n    kind: loop\n    times: 1\n" +
    "    input:\n      st: ${step.output}\n" +
    `    asserts:\n      - "${assertion}"\n` +
    "    nodes:\n" +
    "      bump:\n" +
    "        kind: code\n" +
    "        input:\n          st: ${input.st}\n" +
    "        output:\n          st: Rec3\n" +
    '        code: "return ({st: {a: \\"x\\", b: \\"y\\"}})"\n' +
    "    output: ${bump.output}\n";
  return flow(nodes, "${drive.output}");
}

const LOOPS: [string, (assertion: string) => string][] = [
  ["named", namedLoop],
  ["inline", inlineLoop],
];

test.each(LOOPS)("a loop assert reads the carried field (load half): %s", (_id, build) => {
  load(build("st.b == null or st.b == 'y'"), BODY);
});

test.each(LOOPS)("a loop assert reads a field neither seed nor carried has: %s", (_id, build) => {
  expect(loadMessage(build("st.zz == 'x'"), BODY)).toContain(
    "node 'drive' assert: reference ${st.zz} reads unknown field 'zz'",
  );
});

test("a node assert literal input field read", () => {
  expect(loadMessage(flow(code({ n: "3" }, ["n.x == 1"]), "${c.output}"))).toContain(
    "node 'c' assert: reference ${n.x} reads field 'x' on type int, which has no fields",
  );
});

test.each([
  ["any", code({ v: "!Any ${step.output}" }, ["v.k == 1"])],
  ["object", producer("o", "object") + code({ v: "${o.output}" }, ["v.k == 1"])],
])("node assert open input field reads load: %s", (_id, nodes) => {
  load(flow(nodes, "${c.output}"));
});

test("a node assert known input field loads (load half)", () => {
  load(flow(code({ v: "${step.output}" }, ["v.a == 'x'"]), "${c.output}"));
});

test("a node assert reads the narrowed member inside a branch", () => {
  load(flow(routed(code({ v: "${d.output}" }, ["v.qty > 0"], "ag")), "${ag.output | other.output}"));
  const outside = producer("d", "Order") + code({ v: "${d.output}" }, ["v.qty > 0"]);
  expect(loadMessage(flow(outside, "${c.output}"))).toContain(`node 'c' assert: reference \${v.qty} ${ASSERT_QTY_HINT}`);
});

/**
 * Producer `d` typed `produced`, a case routing `isinstance(d.output, member)` to `ag` and anything
 * else to `other`, then `node` (id `ag`) and `other`.
 */
function isinstanceRouted(produced: string, member: string, node: string): string {
  return (
    producer("d", produced) +
    "  route:\n" +
    "    kind: case\n" +
    "    cases:\n" +
    `      - when: isinstance(d.output, ${member})\n        then: ag\n` +
    "    else: other\n" +
    node +
    "  other:\n" +
    "    kind: code\n" +
    "    input:\n      s: ${input.s}\n" +
    "    output: str\n" +
    "    code: return 'x'\n"
  );
}

test.each([
  ["untagged-assert", "AB", "A", code({ v: "${d.output}" }, ["v.x > 0"], "ag")],
  ["shared-assert", "Trade", "Sized", code({ v: "${d.output}" }, ["v.qty > 0"], "ag")],
])("an isinstance branch reads the narrowed member: %s", (_id, produced, member, node) => {
  load(flow(isinstanceRouted(produced, member, node), "${ag.output | other.output}", { typedefs: UNION_TYPES }));
});

/**
 * A flow whose node `c` asserts `assertion` on its local `v`, bound from `${d.output}` with `d`
 * typed `Order`; `c` is an AGENT node when `kind` is `agent`.
 */
function localAssert(assertion: string, kind = "code"): string {
  const node =
    kind === "agent"
      ? agent({ v: "${d.output}" }, "hi", { nodeId: "c", extra: `    asserts:\n      - "${assertion}"\n` })
      : code({ v: "${d.output}" }, [assertion]);
  return flow(producer("d", "Order") + node, "${c.output}");
}

test.each([
  "isinstance(v, Buy) and v.qty > 0",
  "not isinstance(v, Buy) or v.qty > 0",
  "isinstance(v, Sell) or v.qty > 0",
  "v.action == 'buy' and v.qty > 0",
  "v.action != 'buy' or v.qty > 0",
  "'sell' == v.action or v.qty > 0",
  "not (isinstance(v, Buy) and v.qty < 0)",
  "(isinstance(v, Buy) and v.qty > 0) or v.action == 'sell'",
  "isinstance(v, Buy) and (v.qty > 0 or v.qty < -10)",
])("node assert reads are narrowed by the checks before them: %s", (assertion) => {
  load(localAssert(assertion));
});

test.each([
  ["check-after-read", "v.qty > 0 and isinstance(v, Buy)", `reference \${v.qty} ${ASSERT_QTY_HINT}`],
  ["unguarded", "v.qty > 0", `reference \${v.qty} ${ASSERT_QTY_HINT}`],
  [
    "or-sees-the-other-member",
    "isinstance(v, Buy) or v.qty > 0",
    "reference ${v.qty} reads field 'qty', which Sell doesn't have; v is Sell here, by the checks before this read",
  ],
])("node assert reads the checks do not cover are rejected: %s", (_id, assertion, expected) => {
  expect(loadMessage(localAssert(assertion))).toContain(`node 'c' assert: ${expected}`);
});

test("a node assert read under two guards is checked under each", () => {
  const message = loadMessage(localAssert("v.qty > 0 or (isinstance(v, Buy) and v.qty > 1)"));
  expect(count(message, `reference \${v.qty} ${ASSERT_QTY_HINT}`)).toBe(1);
});

test("agent node assert reads are narrowed", () => {
  load(localAssert("isinstance(v, Sell) or v.qty > 0", "agent"));
});

/** A flow whose CODE node `c` returns an `Order` and asserts `assertion`. */
function outputAssert(assertion: string): string {
  const node =
    "  c:\n    kind: code\n    input:\n      s: ${input.s}\n" +
    '    output: Order\n    code: "return ({action: \\"sell\\"})"\n' +
    `    asserts:\n      - "${assertion}"\n`;
  return flow(node, "${c.output}");
}

test("node assert output reads are narrowed", () => {
  load(outputAssert("isinstance(output, Buy) and output.qty > 0"));
  expect(loadMessage(outputAssert("output.qty > 0"))).toContain(
    "node 'c' assert: reference ${output.qty} reads field 'qty', which only union members with action " +
      "buy have; check the member first with `isinstance(output, <Type>) and ...`",
  );
});

test("a partial-union node assert reads the narrowed member", () => {
  // On a cycle a member-only field is readable, below a narrowed member too; the cycle is what the
  // load reports.
  const typedefs =
    "  Limit: {type: 'Literal(limit)', price: int}\n" +
    "  Market: {type: 'Literal(market)'}\n" +
    "  LBuy: {action: 'Literal(buy)', leg: 'Union(Limit, Market)'}\n" +
    "  LOrder: Union(LBuy, Sell)\n";
  const nodes =
    producer("d", "LOrder") +
    "  b:\n    kind: code\n    input:\n      y: ${c.output}\n" +
    "    output: str\n    code: return 'x'\n" +
    code({ v: "${d.output}", y: "${b.output}" }, ["isinstance(v, LBuy) and v.leg.price > 0"]);
  const message = loadMessage(flow(nodes, "${c.output}", { typedefs }));
  expect(message).toContain('flow has a cycle involving ["b","c"]');
  expect(message).not.toContain("reads field");
});

test("an impossible node assert check is located at the assert", () => {
  // The read after the impossible check is reached by no value, so it is not checked.
  const text = localAssert("isinstance(v, Buy) and v.action == 'sell' and v.zz > 0");
  const error = loadError(() => load(text));
  expect(error.message).toContain(
    "node 'c' assert: `v.action == \"sell\"` can never be true here: v is Buy, by the checks before it",
  );
  expect(error.message).not.toContain("${v.zz}");
  expect(error.diagnostic.frames.at(-1)!.range!.startLine).toBe(lineOf(text, "v.action == 'sell'"));
});

test("a call node assert's tag checks narrow its local", () => {
  const nodes =
    producer("d", "Order") +
    "  c:\n    kind: call\n    call: child\n" +
    "    input:\n      v: ${d.output}\n" +
    "    asserts:\n      - \"v.action == 'buy' and v.qty > 0\"\n";
  load(flow(nodes, "${c.output}", { defs: childText("Order") }), { child: childOf("Order") });
});

test("an assert reading many typed inputs loads", () => {
  // Equal alternatives of a typed binding count once, so the checks stay linear.
  const names = Array.from({ length: 14 }, (_, i) => `a${i}`);
  const assertion = names.map((name) => `${name}.a == 'x'`).join(" and ");
  const node = code(Object.fromEntries(names.map((name) => [name, "!Rec ${step.output}"])), [assertion]);
  load(flow(node, "${c.output}"));
});

/**
 * A flow with CODE producers typed by `producers` and node `c` asserting `assertion` on its
 * `inputs`.
 */
function localsAssert(assertion: string, producers: Record<string, string>, inputs: Record<string, string>): string {
  const nodes = Object.entries(producers)
    .map(([nodeId, typ]) => producer(nodeId, typ))
    .join("");
  return flow(nodes + code(inputs, [assertion]), "${c.output}");
}

test.each([
  [
    "no-overlap",
    "isinstance(v, Rec) and v.zz > 0",
    "`isinstance(v, Rec)` can never be true: v is declared Order, and no Order value fits Rec",
  ],
  [
    "not-a-tag",
    "v.action == 'byu' and v.zz > 0",
    '`v.action == "byu"` compares v.action with "byu", which is not one of its tags (buy, sell); ' +
      'did you mean "buy"?',
  ],
])("a node assert read after a reported check is not checked: %s", (_id, assertion, expected) => {
  const message = loadMessage(localAssert(assertion));
  expect(message).toContain(`node 'c' assert: ${expected}`);
  expect(message).not.toContain("${v.zz}");
});

test("a node assert read after an always-true negation is not checked", () => {
  const message = loadMessage(localsAssert("not isinstance(v, Buy) and v.zz > 0", { d: "Buy" }, { v: "${d.output}" }));
  expect(message).toContain(
    "node 'c' assert: `not isinstance(v, Buy)` can never be true: v is declared Buy, so " +
      "`isinstance(v, Buy)` is always true",
  );
  expect(message).not.toContain("${v.zz}");
});

test("a node assert read of one local after a reported check on another is not checked", () => {
  const message = loadMessage(
    localsAssert(
      "isinstance(v, Buy) and v.action == 'sell' and w.zz > 0",
      { d: "Order" },
      { v: "${d.output}", w: "${step.output}" },
    ),
  );
  expect(message).toContain('`v.action == "sell"` can never be true here');
  expect(message).not.toContain("${w.zz}");
});

// `v` is declared Order but bound from a Buy, so it has two alternatives: Order and Buy.
const TWO_ALTERNATIVES = { v: "!Order ${d.output}", w: "!Order ${e.output}" };

test("a node assert check impossible under every alternative is reported once", () => {
  const message = loadMessage(localsAssert("isinstance(v, Rec) or v.zz > 0", { d: "Buy", e: "Order" }, TWO_ALTERNATIVES));
  expect(count(message, "`isinstance(v, Rec)` can never be true")).toBe(1);
  expect(count(message, "reference ${v.zz} reads unknown field 'zz'")).toBe(1);
});

test("a node assert read dead under only some alternatives is checked", () => {
  // Under Order the tag check rules `v` out, but under Buy it is no check at all, so neither the
  // check is reported nor the read after it skipped.
  const message = loadMessage(
    localsAssert(
      "isinstance(v, Buy) and v.action == 'sell' and w.zz > 0",
      { d: "Buy", e: "Order" },
      TWO_ALTERNATIVES,
    ),
  );
  expect(message).toContain("node 'c' assert: reference ${w.zz} reads unknown field 'zz' on record");
  expect(message).not.toContain("can never be true");
});

test.each([
  [
    "no-overlap",
    "isinstance(output, Rec)",
    "`isinstance(output, Rec)` can never be true: output is declared Order, and no Order value fits Rec",
  ],
  [
    "always",
    "not isinstance(output, Order)",
    "`not isinstance(output, Order)` can never be true: output is declared Order, so " +
      "`isinstance(output, Order)` is always true",
  ],
  [
    "context",
    "isinstance(output, Buy) and isinstance(output, Sell)",
    "`isinstance(output, Sell)` can never be true here: output is Buy, by the checks before it",
  ],
])("a node assert impossible output check is reported: %s", (_id, assertion, expected) => {
  expect(loadMessage(outputAssert(assertion))).toContain(`node 'c' assert: ${expected}`);
});

test("a node assert output read after a reported check is not checked", () => {
  const message = loadMessage(outputAssert("isinstance(output, Rec) and output.action == 'buy'"));
  expect(message).toContain("`isinstance(output, Rec)` can never be true");
  expect(message).not.toContain("${output.action}");
});

test("a node assert over many locals with two alternatives loads", () => {
  // Each local is checked against its own alternatives, not every combination.
  const names = Array.from({ length: 12 }, (_, i) => `v${i}`);
  const assertion = names.map((x) => `(isinstance(${x}, Buy) or ${x}.action == 'sell')`).join(" and ");
  load(localsAssert(assertion, { d: "Buy" }, Object.fromEntries(names.map((x) => [x, "!Order ${d.output}"]))));
});

test("a call node assert's member-only hint compares the tag", () => {
  // `isinstance` is not allowed in a CALL node's asserts, so the hint compares the tag.
  const nodes =
    producer("d", "Order") +
    "  c:\n    kind: call\n    call: child\n" +
    "    input:\n      v: ${d.output}\n" +
    '    asserts:\n      - "v.qty > 0"\n';
  const message = loadMessage(flow(nodes, "${c.output}", { defs: childText("Order") }), { child: childOf("Order") });
  expect(message).toContain(
    "node 'c' assert: reference ${v.qty} reads field 'qty', which only union members with action buy " +
      'have; check the member first with `v.action == "buy" and ...`',
  );
  expect(message).not.toContain("isinstance");
});

test("a call node assert's member-only hint on an untagged union names a case", () => {
  const nodes =
    producer("d", "AB") +
    "  c:\n    kind: call\n    call: child\n" +
    "    input:\n      v: ${d.output}\n" +
    '    asserts:\n      - "v.x > 0"\n';
  const text = flow(nodes, "${c.output}", { typedefs: UNION_TYPES, defs: childText("AB") });
  expect(loadMessage(text, { child: childOf("AB") })).toContain(
    "node 'c' assert: reference ${v.x} reads field 'x', which only 1 of 2 union members have; " +
      "check the member with isinstance(d.output, <Type>) in a case first",
  );
});

/** A one-turn inline LOOP carrying `o: Order`, seeded from the flow input `o`. */
function orderLoop(assertion: string): string {
  const nodes =
    "  drive:\n    kind: loop\n    times: 1\n" +
    "    input:\n      o: ${input.o}\n" +
    `    asserts:\n      - "${assertion}"\n` +
    "    nodes:\n" +
    "      bump:\n" +
    "        kind: code\n" +
    "        input:\n          o: ${input.o}\n" +
    "        output:\n          o: Order\n" +
    '        code: "return ({o: inputs.o})"\n' +
    "    output: ${bump.output}\n";
  return flow(nodes, "${drive.output}", { inputs: "  o: !Order\n" });
}

test("a loop node assert's tag checks narrow its local (load half)", () => {
  load(orderLoop("o.action != 'buy' or o.qty > 0"));
  expect(loadMessage(orderLoop("o.qty > 0"))).toContain(
    "node 'drive' assert: reference ${o.qty} reads field 'qty', which only union members with action buy " +
      'have; check the member first with `o.action == "buy" and ...`',
  );
  const message = loadMessage(orderLoop("o.action == 'sell' and o.action == 'buy' and o.zz > 0"));
  expect(message).toContain('`o.action == "buy"` can never be true here: o is Sell');
  expect(message).not.toContain("${o.zz}");
});

const PARTIAL_CHILD_TEXT =
  "defs:\n" +
  "  child:\n" +
  "    input:\n      x: !str\n" +
  "    nodes:\n" +
  "      typed:\n" +
  "        kind: code\n" +
  "        input:\n          x: ${input.x}\n" +
  "        output: str\n" +
  "        code: return 'a'\n" +
  "      untyped:\n" +
  "        kind: code\n" +
  "        input:\n          x: ${input.x}\n" +
  '        code: "return ({deep: {k: 1}})"\n' +
  "    output:\n" +
  "      a: ${typed.output}\n" +
  "      b: ${untyped.output}\n";

/** The hand-built partial `child`: a record of a typed `a` and an opaque `b`. */
const PARTIAL_CHILD = {
  child: handChild({
    inputs: parametersFromYaml("input:\n  x: !str\n", REGISTRY),
    producers: { typed: new ValueContract(Type.scalar(ValueKind.STRING)), untyped: null },
    output: { a: "${typed.output}", b: "${untyped.output}" },
  }),
};

/**
 * A CALL to `child` (a partial record: typed `a`, opaque `b`), and a CODE node asserting
 * `assertion` on it through the local name `v`.
 */
function partialRead(assertion: string): string {
  const nodes =
    "  call_child:\n    kind: call\n    call: child\n" +
    "    input:\n      x: ${input.s}\n" +
    code({ v: "${call_child.output}" }, [assertion]);
  return flow(nodes, "${c.output}", { defs: PARTIAL_CHILD_TEXT });
}

test("partial record local root reads (load half)", () => {
  expect(loadMessage(partialRead("v.zz == 1"), PARTIAL_CHILD)).toContain(
    "node 'c' assert: reference ${v.zz} reads unknown field 'zz' on record",
  );
  load(partialRead("v.a == 'a' and v.b.deep.k == 1"), PARTIAL_CHILD);
});

// ---------- builtin results ----------

const TEXT_FIELD = (name: string) => `reference \${${name}(...).f} reads field 'f' on type str, which has no fields`;

test("a node assert reads a field on a builtin result", () => {
  expect(loadMessage(flow(code({ v: "${step.output}" }, ["upper(v.a).f == 'x'"]), "${c.output}"))).toContain(
    `node 'c' assert "upper(v.a).f == 'x'": ${TEXT_FIELD("upper")}`,
  );
});

test("a loop while reads a field on a builtin result", () => {
  const nodes =
    "  drive:\n    kind: loop\n    call: body\n" +
    "    while: ${upper(st.a).f == 'x'}\n    max: 3\n" +
    "    input:\n      st: ${step.output}\n";
  expect(loadMessage(flow(nodes, "${drive.output}", { defs: LOOP_NAMED_BODY }), BODY)).toContain(
    `loop node 'drive' while: ${TEXT_FIELD("upper")}`,
  );
});

test("a builtin result without a field read loads: assert", () => {
  load(flow(code({ v: "${input.s}" }, ["upper(v) == 'X'"]), "${c.output}"));
});

test("an unparsable condition keeps its located parse error", () => {
  expect(loadMessage(flow(code({ v: "${step.output}" }, ["upper(v.a).f =="]), "${c.output}"))).toContain(
    "node 'c' assert 'upper(v.a).f ==': not a valid boolean expression",
  );
});

// ---------- harvested conformance cases ----------

const CASES = "load-core/compose_field_read_sites__";

test.each<CaseRow>([
  { id: `${CASES}call_node_assert_reads_a_field_its_binding_lacks`, call: assembledWith({ child: childOf("Rec") }) },
  { id: `${CASES}impossible_node_assert_check`, call: assembled },
  { id: `${CASES}loop_assert_reads_an_unknown_carried_field`, call: assembled },
  { id: `${CASES}member_only_hint_on_an_untagged_union`, call: assembledWith({ child: childOf("AB") }) },
  { id: `${CASES}node_assert_check_no_value_fits`, call: assembled },
  { id: `${CASES}node_assert_compares_with_a_non_tag`, call: assembled },
  { id: `${CASES}node_assert_negated_check_always_true`, call: assembled },
  { id: `${CASES}node_assert_open_input_field_reads_load`, call: assembled },
  { id: `${CASES}node_assert_read_before_its_check`, call: assembled },
  { id: `${CASES}node_assert_read_on_the_other_member`, call: assembled },
  { id: `${CASES}node_assert_reads_unknown_input_field`, call: assembled },
  { id: `${CASES}loop_while_reads_a_field_on_a_builtin_result`, call: assembledWith(BODY), full: true },
])("assemble: $id", runCase);
