/**
 * Union narrowing at load: a CASE branch reads the members its tag or isinstance check selects. A
 * port of `tests/compose/test_union_narrowing.py`, loading each flow with `assembleText` where
 * Python calls `load_flow`; every flow carries the Python `defs:` block, whose entries are built by
 * hand (`handChild`, each def's input read with the flow's typedefs, returning its node `p`), since
 * def bodies load in T3-22.
 *
 * Ported: `test_a_when_check_that_can_never_be_true_is_a_located_error` (5),
 * `test_a_when_check_the_routing_rules_out_is_an_error`,
 * `test_a_cycle_through_a_branch_reports_the_cycle`,
 * `test_a_member_only_field_on_a_cycle_reports_the_cycle`,
 * `test_an_unknown_depends_on_still_reports_the_unknown_node`,
 * `test_an_inline_body_branch_target_cannot_infer_a_member_field` and
 * `test_an_unknown_isinstance_type_keeps_its_own_error`; the rows the `when:` value and
 * exhaustiveness checks decide, `test_a_nested_on_case_over_a_union_field_narrows_outer_then_inner`
 * (its exhaustiveness half), `test_an_inner_on_case_is_exhaustive_over_its_narrowed_tag`,
 * `test_an_inner_on_case_over_a_flow_input_is_exhaustive_over_its_narrowed_tag`,
 * `test_a_when_that_is_not_a_tag_is_a_located_error`,
 * `test_a_non_exhaustive_on_over_a_literal_input_errors`,
 * `test_a_non_exhaustive_on_over_a_union_node_output_still_errors` and
 * `test_an_on_inside_a_partly_shared_union_branch_covers_the_shared_tags`; the harvested cases
 * `load-types/compose_union_narrowing__a_when_comparing_the_tag_with_a_non_tag_suggests_one`,
 * `…__an_isinstance_that_can_never_be_true`, `…__a_defensive_else_after_every_tag_loads` and
 * `…__a_nullable_union_defensive_else_sees_none` (its `place_rest` def built by hand). The
 * rows the CALL type check decides, each error pinned whole where Python matches fragments:
 * `test_a_call_target_under_buy_passes_a_buy_order`,
 * `test_a_call_target_under_sell_fails_the_call_type_check`,
 * `test_an_unrouted_call_of_the_whole_union_still_fails`,
 * `test_a_nested_inline_call_in_a_branch_target_is_narrowed`,
 * `test_an_inline_call_in_a_routed_node_input_is_narrowed`,
 * `test_else_passes_the_remaining_union_to_a_call`,
 * `test_a_nullable_union_else_target_sees_an_optional_union`,
 * `test_a_nullable_union_narrows_to_a_non_nullable_member_under_when`,
 * `test_a_path_base_through_an_optional_record_keeps_nullability_in_else`,
 * `test_a_nullable_union_defensive_else_sees_none`,
 * `test_isinstance_narrows_an_opaque_object_to_the_checked_type`,
 * `test_isinstance_cases_on_unrelated_types_into_one_target_do_not_narrow`,
 * `test_an_opaque_object_outside_isinstance_does_not_fit_a_record`,
 * `test_an_isinstance_branch_narrows_a_nullable_base_to_the_member`,
 * `test_a_partly_shared_union_branch_passes_the_shared_members_to_a_call` and the call half of
 * `test_a_member_that_only_overlaps_the_checked_union_keeps_the_checked_type`.
 *
 * Changed: a compared tag and its close match are spelled `"biy"` / `did you mean "buy"?` (Python
 * `'biy'` / `did you mean 'buy'?`), and the cycle's nodes and the uncovered tags as JSON text,
 * `["b","c"]` and `["hold","sell"]` (Python `['b', 'c']` and `['hold', 'sell']`).
 *
 * Later (T3-18): the rows whose error the reference validator raises, or that load only once it
 * reads the narrowed views: `test_a_buy_branch_reads_a_member_field`,
 * `test_a_hold_branch_reading_qty_reads_an_unknown_field_on_hold`,
 * `test_a_plain_reference_chain_is_narrowed`, `test_a_join_of_buy_and_sell_branches_narrows_to_both`,
 * `test_depends_on_propagates_the_branch`, `test_runs_after_does_not_propagate_the_branch`,
 * `test_an_optional_group_does_not_propagate_the_branch`,
 * `test_two_whens_to_one_node_give_a_two_member_union`, `test_else_gives_the_remaining_member`,
 * `test_else_gives_the_remaining_union`,
 * `test_a_node_routed_by_two_narrowing_cases_on_different_bases_is_not_narrowed`,
 * `test_a_node_routed_by_a_narrowing_case_and_a_searched_case_is_not_narrowed`,
 * `test_an_inner_searched_case_in_a_buy_branch_keeps_the_narrowing`,
 * `test_a_nested_on_case_outside_the_outer_branch_is_a_rule_one_error`,
 * `test_a_flow_input_base_narrows`, `test_a_path_base_narrows`,
 * `test_on_over_a_coalesce_does_not_narrow`, `test_on_over_embedded_text_does_not_narrow`,
 * `test_nodes_outside_the_branch_and_the_flow_output_are_not_narrowed`,
 * `test_a_field_no_member_has_on_a_cycle_is_still_an_unknown_field`,
 * `test_a_defensive_else_after_every_tag_loads_and_is_not_narrowed`,
 * `test_a_contradictory_nested_case_target_is_not_narrowed`,
 * `test_required_groups_from_buy_and_sell_branches_meet_empty`,
 * `test_a_target_under_isinstance_reads_a_member_field`, `test_an_isinstance_and_conjunct_narrows`,
 * `test_isinstance_under_or_narrows_only_what_the_when_proves`,
 * `test_a_branch_sees_a_tag_comparison`,
 * `test_a_branch_reads_the_fields_the_members_its_when_proves_share`,
 * `test_a_read_in_a_when_is_narrowed_through_and_or_and_not`,
 * `test_a_read_in_a_when_of_a_field_the_narrowed_members_lack_is_an_error`,
 * `test_a_misspelled_read_in_a_when_names_the_narrowed_type`,
 * `test_a_branch_of_a_negated_partly_shared_check_keeps_the_overlapping_member`,
 * `test_a_read_after_a_check_the_routing_decides_is_checked_against_the_routing`,
 * `test_a_when_read_narrowed_only_by_the_routing_credits_the_routing`,
 * `test_a_when_read_past_a_check_the_checks_before_it_decide_is_checked_under_them`,
 * `test_overlapping_checks_narrow_each_branch_by_its_own_check`,
 * `test_a_conjunct_after_isinstance_reads_the_narrowed_base`,
 * `test_a_conjunct_before_isinstance_is_a_rule_one_error`,
 * `test_a_read_guarded_in_one_case_and_unguarded_in_another_is_an_error`,
 * `test_an_unguarded_read_error_points_at_its_own_when`,
 * `test_a_read_unguarded_first_points_at_the_first_when`,
 * `test_isinstance_of_a_type_the_base_already_fits_does_not_narrow`,
 * `test_a_read_guarded_by_different_members_reads_their_shared_fields`,
 * `test_isinstance_narrows_an_untagged_union_to_the_member`,
 * `test_a_target_routed_by_two_isinstance_cases_gets_their_union`,
 * `test_a_target_routed_by_isinstance_and_a_plain_case_is_not_narrowed`,
 * `test_a_searched_else_is_not_narrowed`, `test_isinstance_on_a_flow_input_narrows`,
 * `test_isinstance_inside_an_on_branch_composes`,
 * `test_isinstance_of_a_partly_shared_union_reads_the_shared_fields`,
 * `test_a_partly_shared_union_check_lines_up_with_an_on_else` and
 * `test_a_conjunct_after_a_partly_shared_union_check_reads_the_shared_fields`; the member-field
 * read of `test_a_nested_on_case_over_a_union_field_narrows_outer_then_inner` (its
 * `${d.output.leg.price}` loads only under both cases' narrowing; until the reference validator
 * runs, the same read loads unrouted too, so the TS load proves nothing about it); and the
 * reference half of `test_a_member_that_only_overlaps_the_checked_union_keeps_the_checked_type`.
 *
 * Later (T4): `test_a_searched_when_with_a_tag_check_and_a_member_read_runs` and
 * `test_a_partly_shared_union_branch_runs` (runs).
 */

import { afterEach, describe, expect, test } from "bun:test";

import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  caseRegistry,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

afterEach(() => forget());

const TYPEDEFS =
  "typedefs:\n" +
  "  Limit: {type: 'Literal(limit)', price: float}\n" +
  "  Market: {type: 'Literal(market)'}\n" +
  "  Leg: Union(Limit, Market)\n" +
  "  Buy: {action: 'Literal(buy)', qty: int, limit: float, leg: Leg}\n" +
  "  Hold: {action: 'Literal(hold)', note: str}\n" +
  "  Sell: {action: 'Literal(sell)', qty: int, price: float, note: str}\n" +
  "  Decision: Union(Buy, Hold, Sell)\n" +
  "  HoldOrSell: Union(Hold, Sell)\n" +
  "  Wrap: {decision: Decision}\n" +
  "  BuyOrSell: Union(Buy, Sell)\n" +
  "  Text: {text: str}\n" +
  "  Num: {num: int}\n" +
  "  Reply: Union(Text, Num)\n" +
  "  Mode: Literal(a, b)\n" +
  "  Cancel: {action: 'Literal(cancel)', reason: str}\n" +
  "  Rest: Union(Hold, Sell, Cancel)\n" +
  "  Narrow: {action: 'Literal(buy)', qty: int, limit: float, leg: Limit}\n" +
  "  SellOrNarrow: Union(Sell, Narrow)\n";

// Fields: `qty` (buy, sell), `limit` and `leg` (buy), `price` (sell), `note` (hold, sell).
const QTY_ERROR =
  "reference ${d.output.qty} reads field 'qty', which only union members with action " +
  "buy, sell have; read it inside a case on ${d.output.action}";

const DEFS =
  "defs:\n" +
  "  place:\n" +
  "    input:\n      order: !Buy\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          o: ${input.order}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  place_rest:\n" +
  "    input:\n      order: !HoldOrSell\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          o: ${input.order}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  place_sell_or_narrow:\n" +
  "    input:\n      order: !SellOrNarrow\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          o: ${input.order}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  place_maybe_rest:\n" +
  "    input:\n      order: !Optional(HoldOrSell)\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          o: ${input.order}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  outer:\n" +
  "    input:\n      x: !int\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          x: ${input.x}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  inner:\n" +
  "    input:\n      y: !int\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          y: ${input.y}\n" +
  "        code: return 1\n" +
  "    output: 1\n";

/** A CODE node typed `output` that reads the flow input `s`. */
function producer(nodeId: string, output: string): string {
  return (
    `  ${nodeId}:\n    kind: code\n    input:\n      s: \${input.s}\n` + `    output: ${output}\n    code: return None\n`
  );
}

/** An untyped CODE node with `inputs` and extra top-level YAML keys. */
function code(nodeId: string, inputs: Record<string, string> = {}, extra: Record<string, string> = {}): string {
  let text = `  ${nodeId}:\n    kind: code\n`;
  for (const [key, value] of Object.entries(extra)) text += `    ${key}: ${value}\n`;
  if (Object.keys(inputs).length > 0) {
    text += "    input:\n";
    for (const [name, source] of Object.entries(inputs)) text += `      ${name}: ${source}\n`;
  }
  return text + "    code: return 1\n";
}

/** A CASE node; `on: null` is the searched form. */
function caseNode(
  nodeId: string,
  on: string | null,
  cases: readonly (readonly [string, string])[],
  else_: string | null = null,
): string {
  let text = `  ${nodeId}:\n    kind: case\n`;
  if (on !== null) text += `    on: ${on}\n`;
  text += "    cases:\n";
  for (const [when, then] of cases) text += `      - when: ${when}\n        then: ${then}\n`;
  if (else_ !== null) text += `    else: ${else_}\n`;
  return text;
}

/** A flow with producer `d` (a `Decision`) reading the input `s`, then `nodes`, the defs, and an output. */
function flow(...nodes: string[]): string {
  return flowWith({}, ...nodes);
}

/**
 * A flow as `flow` builds it, with producer `d` typed `dType` (default `Decision`) and an `input:`
 * section holding `inputs` (default `s: !str`).
 */
function flowWith(options: { inputs?: string; dType?: string }, ...nodes: string[]): string {
  const { inputs = "  s: !str\n", dType = "Decision" } = options;
  return (
    "id: p\nname: p\n" +
    TYPEDEFS +
    `input:\n${inputs}` +
    "nodes:\n" +
    producer("d", dType) +
    nodes.join("") +
    "output: ${d.output.action}\n" +
    DEFS
  );
}

const REGISTRY = caseRegistry(SourceDocument.parse(TYPEDEFS));

/**
 * A `defs:` entry built by hand: input `<name>: !<type>`, returning its CODE node `p` (typed
 * `output`, else untyped).
 */
const def = (name: string, type: string, output: ValueContract | null = null) =>
  handChild({
    inputs: parametersFromYaml(`input:\n  ${name}: !${type}\n`, REGISTRY),
    producers: { p: output },
    output: "${p.output}",
  });

/** The `DEFS` entries built by hand (`inner` returns the int `1`). */
const CHILDREN = {
  place: def("order", "Buy"),
  place_rest: def("order", "HoldOrSell"),
  place_sell_or_narrow: def("order", "SellOrNarrow"),
  place_maybe_rest: def("order", "Optional(HoldOrSell)"),
  outer: def("x", "int"),
  inner: def("y", "int", new ValueContract(Type.scalar(ValueKind.INTEGER))),
};

// The canonical texts of `Buy`, `Hold`, `Sell` and `HoldOrSell`, as the call type check names them.
const BUY =
  '{"action":Literal(buy),"leg":Union({"price":float,"type":Literal(limit)},{"type":Literal(market)}),' +
  '"limit":float,"qty":int}';
const HOLD = '{"action":Literal(hold),"note":str}';
const SELL = '{"action":Literal(sell),"note":str,"price":float,"qty":int}';
const HOLD_OR_SELL = `Union(${HOLD},${SELL})`;
const PLACE_EXPECTS = `call(place) argument 'order' — child expects '${BUY}'`;
const PLACE_REST_EXPECTS = `call(place_rest) argument 'order' — child expects '${HOLD_OR_SELL}'`;

/** Assemble `text` with `CHILDREN` as its defs. */
const load = (text: string) => assembleText(text, "flow.yaml", { children: CHILDREN });

const ON = "${d.output.action}";

/** A searched CASE `route` over `cases`, else to `e`. */
function searched(...cases: (readonly [string, string])[]): string {
  return caseNode("route", null, cases, "e");
}

/** The message of the `LoadError` assembling `text` throws. */
const assembleError = (text: string): string => loadError(() => load(text)).message;

// --- robustness --- //

test("a cycle through a branch reports the cycle", () => {
  const message = assembleError(
    flow(
      caseNode("route", ON, [["buy", "b"]], "e"),
      code("b", { y: "${c.output}", l: "${d.output.limit}" }),
      code("c", { y: "${b.output}", l: "${d.output.limit}" }),
      code("e"),
    ),
  );
  expect(message).toContain('flow has a cycle involving ["b","c"]');
});

test("a member-only field on a cycle reports the cycle", () => {
  const message = assembleError(
    flow(code("b", { y: "${c.output}", l: "${d.output.limit}" }), code("c", { y: "${b.output}" })),
  );
  expect(message).toContain('flow has a cycle involving ["b","c"]');
});

test("an unknown depends_on still reports the unknown node", () => {
  const message = assembleError(
    flow(
      caseNode("route", ON, [["buy", "b"]], "e"),
      code("b", { l: "${d.output.limit}" }, { depends_on: "[nope]" }),
      code("e"),
    ),
  );
  expect(message).toContain("node 'b': `depends_on` references unknown node 'nope'");
});

test("an inline body branch target cannot infer a member field", () => {
  const target =
    "  b:\n" +
    "    kind: call\n" +
    "    input:\n      q: ${d.output.qty}\n" +
    "    nodes:\n" +
    "      inner:\n        kind: code\n        input:\n          q: ${input.q}\n" +
    "        code: return 1\n" +
    "    output: ${inner.output}\n";
  const message = assembleError(flow(caseNode("route", ON, [["buy", "b"]], "e"), target, code("e")));
  expect(message).toContain(`cannot infer type for inline binding 'b.input.q': ${QTY_ERROR}`);
});

// --- impossible checks in a searched `when:` --- //

describe("a when check that can never be true is a located error", () => {
  test.each<[string, string]>([
    [
      "isinstance(d.output, Buy) and d.output.action == 'hold'",
      '`d.output.action == "hold"` can never be true here: d.output is Buy, by the checks before it',
    ],
    [
      "isinstance(d.output, Buy) and d.output.action != 'buy' and d.output.limit > 0",
      '`d.output.action != "buy"` can never be true here: d.output is Buy, by the checks ' +
        'before it, so `d.output.action == "buy"` is always true',
    ],
    [
      "d.output.action == 'biy'",
      '`d.output.action == "biy"` compares d.output.action with "biy", which is not one ' +
        'of its tags (buy, hold, sell); did you mean "buy"?',
    ],
    [
      "isinstance(d.output, Text)",
      "`isinstance(d.output, Text)` can never be true: d.output is declared Decision, " +
        "and no Decision value fits Text",
    ],
    [
      "not not isinstance(d.output, Text)",
      "`isinstance(d.output, Text)` can never be true: d.output is declared Decision, " +
        "and no Decision value fits Text",
    ],
  ])("%s", (when, message) => {
    const text = flow(searched(["input.s == 'x'", "b"], [`"${when}"`, "b"]), code("b"), code("e"));
    const error = loadError(() => assembleText(text));
    expect(error.message).toContain(`case node 'route': \`cases[1].when\`: ${message}`);
    // A read after the impossible check is not checked on top of it.
    expect(error.message).not.toContain("'limit'");
    const line = text.split("\n").findIndex((row) => row.includes(when)) + 1;
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
  });
});

test("a when check the routing rules out is an error", () => {
  const message = assembleError(
    flow(
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", null, [[`"d.output.action == 'hold'"`, "t"]], "m"),
      code("t"),
      code("m"),
      code("e"),
    ),
  );
  expect(message).toContain(
    "case node 'inner': `cases[0].when`: `d.output.action == \"hold\"` can never be true " +
      "here: d.output is Buy, by the case branch that routes this node",
  );
});

// --- calls --- //

test("a call target under buy passes a buy order", () => {
  load(flow(caseNode("route", ON, [["buy", "call(place, order=${d.output})"]], "e"), code("e")));
});

test("a call target under sell fails the call type check", () => {
  const message = assembleError(
    flow(caseNode("route", ON, [["sell", "call(place, order=${d.output})"]], "e"), code("e")),
  );
  expect(message).toBe(`node 'route' cases[0].then ${PLACE_EXPECTS}, source is '${SELL}'`);
});

test("an unrouted call of the whole union still fails", () => {
  const message = assembleError(flow(code("c", { r: "call(place, order=${d.output})" })));
  expect(message).toBe(`node 'c' input 'r' ${PLACE_EXPECTS}, source is 'Union(${BUY},${HOLD},${SELL})'`);
});

test("a nested inline call in a branch target is narrowed", () => {
  // `inner` is lifted out of the routed `outer`, so it carries the same route.
  load(flow(caseNode("route", ON, [["buy", "call(outer, x=call(inner, y=${d.output.qty}))"]], "e"), code("e")));
});

test("an inline call in a routed node input is narrowed", () => {
  // The call is lifted out of `b`, so it carries `b`'s route.
  load(flow(caseNode("route", ON, [["buy", "b"]], "e"), code("b", { r: "call(place, order=${d.output})" }), code("e")));
});

test("else passes the remaining union to a call", () => {
  load(flow(caseNode("route", ON, [["buy", "b"]], "call(place_rest, order=${d.output})"), code("b")));
});

test("a nullable union else target sees an optional union", () => {
  const nullable = (callee: string) =>
    flowWith(
      { dType: "Optional(Decision)" },
      caseNode("route", ON, [["buy", "b"]], `call(${callee}, order=\${d.output})`),
      code("b"),
    );

  load(nullable("place_maybe_rest"));
  const message = assembleError(nullable("place_rest"));
  expect(message).toBe(`node 'route' else ${PLACE_REST_EXPECTS}, source is 'Optional(${HOLD_OR_SELL})'`);
});

test("a nullable union narrows to a non-nullable member under when", () => {
  load(
    flowWith(
      { dType: "Optional(Decision)" },
      caseNode("route", ON, [["buy", "call(place, order=${d.output})"]], "e"),
      code("e"),
    ),
  );
});

test("a nullable union defensive else sees None", () => {
  const message = assembleError(
    flowWith(
      { dType: "Optional(Decision)" },
      caseNode(
        "route",
        ON,
        [
          ["buy", "b"],
          ["sell", "b"],
          ["hold", "b"],
        ],
        "call(place_rest, order=${d.output})",
      ),
      code("b"),
    ),
  );
  expect(message).toBe(`node 'route' else ${PLACE_REST_EXPECTS}, source is 'None'`);
});

// --- on: cases over a narrowed tag --- //

test("a nested on case over a union field narrows outer then inner", () => {
  assembleText(
    flow(
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", "${d.output.leg.type}", [
        ["limit", "t"],
        ["market", "m"],
      ]),
      code("t", { p: "${d.output.leg.price}" }),
      code("m"),
      code("e"),
    ),
  );
});

test("an inner on case is exhaustive over its narrowed tag", () => {
  assembleText(
    flow(
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", ON, [["buy", "t"]]),
      code("t", { l: "${d.output.limit}" }),
      code("e"),
    ),
  );
});

test("an inner on case over a flow input is exhaustive over its narrowed tag", () => {
  assembleText(
    flowWith(
      { inputs: "  s: !str\n  d: !Decision\n" },
      caseNode("route", "${input.d.action}", [["buy", "inner"]], "e"),
      caseNode("inner", "${input.d.action}", [["buy", "t"]]),
      code("t", { l: "${input.d.limit}" }),
      code("e"),
    ),
  );
});

// --- other bases --- //

test("a path base through an optional record keeps nullability in else", () => {
  const wrapped = (callee: string) =>
    flow(
      producer("r", "Optional(Wrap)"),
      caseNode("route", "${r.output.decision.action}", [["buy", "b"]], `call(${callee}, order=\${r.output.decision})`),
      code("b", { l: "${r.output.decision.limit}" }),
    );

  load(wrapped("place_maybe_rest"));
  const message = assembleError(wrapped("place_rest"));
  expect(message).toBe(`node 'route' else ${PLACE_REST_EXPECTS}, source is 'Optional(${HOLD_OR_SELL})'`);
});

test("a when that is not a tag is a located error", () => {
  const text = flow(caseNode("route", ON, [["buyy", "b"]], "e"), code("b"), code("e"));

  const error = loadError(() => assembleText(text));

  expect(error.message).toStartWith("case node 'route': `when: buyy` is not a tag of `on: ${d.output.action}` (");
  expect(error.message).toEndWith("); use one of: buy, hold, sell");
  const line = text.split("\n").findIndex((row) => row.includes("when: buyy")) + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

test("a non-exhaustive on over a Literal input errors", () => {
  const message = assembleError(
    flowWith({ inputs: "  s: !str\n  mode: !Mode\n" }, caseNode("route", "${input.mode}", [["a", "b"]]), code("b")),
  );
  expect(message).toContain('enum tag(s) ["b"] not covered');
});

test("a non-exhaustive on over a union node output still errors", () => {
  const message = assembleError(flow(caseNode("route", ON, [["buy", "b"]]), code("b")));
  expect(message).toContain('enum tag(s) ["hold","sell"] not covered');
});

// --- isinstance in a searched CASE --- //

test("an on inside a partly shared union branch covers the shared tags", () => {
  assembleText(
    flow(
      searched(["isinstance(d.output, Rest)", "inner"]),
      caseNode("inner", ON, [
        ["hold", "h"],
        ["sell", "s"],
      ]),
      code("h"),
      code("s"),
      code("e"),
    ),
  );
});

test("a partly shared union branch passes the shared members to a call", () => {
  const branch = searched(["isinstance(d.output, Rest)", "call(place_rest, order=${d.output})"]);
  load(flow(branch, code("e")));
  load(flowWith({ dType: "Optional(Decision)" }, branch, code("e")));
});

test("a member that only overlaps the checked union keeps the checked type (call half)", () => {
  const call = "call(place_sell_or_narrow, order=${d.output})";
  load(flow(searched(["isinstance(d.output, SellOrNarrow)", call]), code("e")));
});

// --- isinstance --- //

test("isinstance narrows an opaque object to the checked type", () => {
  const opaque = (when: string) =>
    flow(producer("x", "object"), searched([when, "call(place, order=${x.output})"]), code("e"));

  load(opaque("isinstance(x.output, Buy)"));
  const message = assembleError(opaque("isinstance(x.output, Sell)"));
  expect(message).toBe(`node 'route' cases[0].then ${PLACE_EXPECTS}, source is '${SELL}'`);
});

test("isinstance cases on unrelated types into one target do not narrow", () => {
  const routed = (...whens: string[]) =>
    flow(
      producer("x", "object"),
      searched(...whens.map((when) => [when, "b"] as const)),
      "  b:\n    kind: call\n    call: place\n    input:\n      order: ${x.output}\n",
      code("e"),
    );

  load(routed("isinstance(x.output, Buy)"));
  const message = assembleError(routed("isinstance(x.output, Buy)", "isinstance(x.output, Text)"));
  expect(message).toBe(`call node 'b': binding 'order' — child expects '${BUY}', source is 'object'`);
});

test("an opaque object outside isinstance does not fit a record", () => {
  const message = assembleError(
    flow(producer("x", "object"), searched(["input.s == 'x'", "call(place, order=${x.output})"]), code("e")),
  );
  expect(message).toBe(`node 'route' cases[0].then ${PLACE_EXPECTS}, source is 'object'`);
});

test("an isinstance branch narrows a nullable base to the member", () => {
  load(
    flowWith(
      { dType: "Optional(Decision)" },
      searched(["isinstance(d.output, Buy)", "call(place, order=${d.output})"]),
      code("e"),
    ),
  );
});

test("an unknown isinstance type keeps its own error", () => {
  const message = assembleError(flow(searched(["isinstance(d.output, Nope)", "b"]), code("b"), code("e")));
  expect(message).toContain("unknown type 'Nope'");
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "a_when_comparing_the_tag_with_a_non_tag_suggests_one",
    "an_isinstance_that_can_never_be_true",
    "a_defensive_else_after_every_tag_loads",
  ].map((name) => ({ id: `load-types/compose_union_narrowing__${name}`, call: assembled })),
)("assemble: $id", runCase);

test("assembledWith: load-types/compose_union_narrowing__a_nullable_union_defensive_else_sees_none", () => {
  // Its `source is 'None'` is the CASE narrowing's: the defensive `else:` sees only null.
  runCase({
    id: "load-types/compose_union_narrowing__a_nullable_union_defensive_else_sees_none",
    call: (document, registry, diagnostics, builder) => {
      const placeRest = handChild({
        inputs: parametersFromYaml("input:\n  order: !HoldOrSell\n", registry),
        producers: { p: null },
        output: "${p.output}",
      });
      return assembledWith({ place_rest: placeRest })(document, registry, diagnostics, builder);
    },
  });
});
