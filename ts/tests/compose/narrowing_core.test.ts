/**
 * Union narrowing: the fact domain, guards into facts, and narrowed reads. A port of
 * `tests/compose/test_narrowing_core.py::test_union_of_members_is_the_set_union_and_or_of_nullability`,
 * `::test_union_of_exact_facts_keeps_only_an_equal_type`,
 * `::test_intersection_of_members_is_the_set_intersection`,
 * `::test_intersection_of_exact_with_members`, `::test_isinstance_fact_against_the_base_union`,
 * `::test_isinstance_fact_against_a_partly_shared_union`,
 * `::test_not_isinstance_fact_keeps_the_members_the_check_cannot_have_passed`,
 * `::test_not_isinstance_fact_nullability`,
 * `::test_not_isinstance_fact_gives_nothing_without_a_union_or_an_exclusion`,
 * `::test_tag_fact_selects_members_by_tag`, `::test_tag_fact_nullability`,
 * `::test_tag_fact_gives_nothing_off_a_tag_field_or_a_union`,
 * `::test_check_fact_dispatches_on_the_check`, `::test_check_fact_on_an_unproven_base`,
 * `::test_check_fact_rejects_an_unknown_type_name`, `::test_guard_env_of_one_check`,
 * `::test_guard_env_meets_and_joins_on_one_base`, `::test_guard_env_over_two_bases`,
 * `::test_guard_env_skips_unknown_types_and_bases_it_cannot_narrow`,
 * `::test_guard_env_types_a_member_only_base_through_the_checks_before_it`,
 * `::test_env_unsatisfiable`, `::test_guarded_read_types_reads_below_the_narrowed_base`,
 * `::test_guarded_read_types_takes_the_longest_base`,
 * `::test_guarded_read_types_without_a_fact_for_the_read`,
 * `::test_guarded_read_types_keeps_each_conjunct_s_candidate`, `::test_env_read_type`,
 * `::test_narrowed_type_label_names_members`, `::test_narrowed_read_message`,
 * `::test_isinstance_members_keeps_the_shared_members_only_when_nothing_else_can_pass`,
 * `::test_narrowed_read_error_names_the_narrowed_type_only_for_a_field_it_lacks`,
 * `::test_checked_refs_keeps_a_guard_some_value_satisfies`,
 * `::test_a_read_after_a_reported_check_is_not_checked`,
 * `::test_a_read_that_needs_a_reported_check_to_hold_is_not_checked`,
 * `::test_a_read_that_does_not_need_a_reported_check_is_checked`,
 * `::test_a_read_after_a_check_the_known_type_decides_is_checked_without_it`,
 * `::test_a_check_the_checks_before_it_rule_out_is_impossible`,
 * `::test_a_check_the_checks_before_it_make_always_true_says_so`,
 * `::test_a_tag_value_no_member_has_is_reported`,
 * `::test_an_isinstance_the_declared_type_rules_out_is_impossible`,
 * `::test_an_isinstance_on_a_nested_base_the_type_rules_out_is_impossible`,
 * `::test_a_negated_isinstance_the_declared_type_always_passes_is_impossible`,
 * `::test_a_negated_isinstance_of_a_nullable_typedef`, `::test_possible_checks_are_not_reported`,
 * `::test_unproven_bases_and_lost_facts_prove_nothing_impossible`,
 * `::test_each_impossible_check_is_reported_once_and_only_first`,
 * `::test_code_after_an_always_true_check_is_not_reported`,
 * `::test_a_check_the_routing_rules_out_is_impossible`,
 * `::test_routing_and_the_checks_together_use_the_checks_wording`, and the `check_text`,
 * `check_fact`, `unresolved_type`, `declared_impossible` and `never_value` sites of
 * `::test_an_unknown_check_kind_fails_loudly`.
 *
 * TS-only: the pinned-case table, which makes the call a site makes on the conformance cases a
 * narrowing message decides (an impossible check, or a narrowed read's error) and checks that the
 * case's message, or its last line for a multi-line one, is the row's site prefix plus the
 * produced text. The `compose_null_narrowing__*` and `compose_null_narrowing_case__*` rows sit
 * here too, in the one harness, though their Python tests are `test_null_narrowing.py` and
 * `test_null_narrowing_case.py`. The sites' locations are not pinned here.
 *
 * Changed: `Exact`, `Members`, `intersectFacts`, `isinstanceFact` and `unionFacts` are imported
 * from `narrowing_facts.ts`; Python imports them through `compose.validate.narrowing`'s
 * re-export. Facts, environments and guards compare with `toStrictEqual`, which also compares
 * classes; `Members` takes an array. `types` takes dotted base paths as keys (Python spells the
 * dots `__`). A compared tag and its close match are spelled `"biy"` / `did you mean "buy"?`
 * (Python `'biy'` / `did you mean 'buy'?`). `declaredImpossible` and `neverValue` are exported
 * for these tests (Python's `_declared_impossible` and `_never_value`).
 *
 * Later (T3-16): `test_searched_param_guards_lists_the_checks_before_each_read`.
 *
 * Later (T3-20): `test_the_fact_algebra_lives_in_helpers_and_narrowing_re_exports_it`, the
 * `narrows` site of `test_an_unknown_check_kind_fails_loudly`,
 * `test_an_isinstance_case_narrows_its_target_to_the_member_or_sub_union`,
 * `test_a_case_branch_sees_what_its_when_proves`,
 * `test_a_case_branch_under_a_check_that_may_not_hold_is_not_narrowed`,
 * `test_a_case_param_after_an_impossible_check_is_not_checked`,
 * `test_a_case_param_after_a_check_the_type_decides_sees_the_case_s_view`,
 * `test_an_isinstance_case_on_a_non_union_base_gives_the_exact_type`,
 * `test_isinstance_facts_use_the_original_base_union`,
 * `test_an_isinstance_of_a_type_the_base_fits_gives_no_fact`,
 * `test_two_isinstance_cases_into_one_target_join_by_union`,
 * `test_two_isinstance_cases_on_unrelated_types_into_one_target_drop_the_fact`,
 * `test_a_plain_case_or_else_into_the_target_drops_the_fact`,
 * `test_an_unknown_isinstance_type_gives_no_fact`, `test_a_guarded_case_param_gets_the_checked_view`,
 * `test_on_case_narrows_each_branch_and_else_to_the_unlisted_members`,
 * `test_views_keep_origins_and_unnarrowed_nodes_get_the_same_maps`,
 * `test_else_keeps_the_base_nullability_and_when_branches_drop_it`,
 * `test_a_when_branch_and_else_into_one_target_join_to_every_member`,
 * `test_a_then_target_named_default_is_narrowed_like_any_node`,
 * `test_a_then_target_named_default_leaves_the_else_target_the_unlisted_members`,
 * `test_else_keeps_nullability_from_a_nullable_path_segment`,
 * `test_control_edges_join_by_union_and_an_edge_without_facts_unconstrains`,
 * `test_a_required_data_group_joins_its_producers_by_union`,
 * `test_depends_on_carries_the_source_environment_and_runs_after_does_not`,
 * `test_contributions_intersect`, `test_two_required_groups_from_disjoint_branches_meet_empty`,
 * `test_nested_case_composes_with_its_own_environment`,
 * `test_a_contradictory_nested_case_branch_is_empty`, `test_outer_bases_narrow_before_inner_ones`,
 * `test_flow_input_bases_narrow_the_input_view`,
 * `test_partial_record_producer_narrows_a_proven_field`,
 * `test_bases_that_are_not_tagged_unions_give_no_facts`,
 * `test_defensive_else_after_every_tag_is_empty_and_unnarrowed`,
 * `test_defensive_else_over_a_nullable_base_sees_none`,
 * `test_a_when_value_that_is_not_a_tag_narrows_its_branch_to_nothing`,
 * `test_a_when_value_that_is_not_a_tag_is_a_load_error`,
 * `test_the_tag_error_points_at_the_when_value`,
 * `test_a_tag_error_on_a_base_only_the_branch_view_has`,
 * `test_nodes_kahn_cannot_order_stay_unnarrowed`, `test_unknown_ids_are_ignored`,
 * `test_narrow_contract_at_the_root_and_a_nested_record_field`,
 * `test_narrow_contract_at_does_not_cross_unions_or_missing_fields`,
 * `test_narrow_contract_at_inside_a_partial_record` and
 * `test_narrow_contract_at_the_root_of_a_partial_record_rebases_field_origins`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import {
  AllOf,
  type Check,
  CheckLiteral,
  checkText,
  type Guard,
  Holds,
  IsinstanceCheck,
  NOTHING,
  proves,
  TagCheck,
} from "../../src/compose/helpers/condition_guards.ts";
import { isinstanceMembers } from "../../src/compose/helpers/isinstance_types.ts";
import {
  type BaseTypeOf,
  checkedRefs,
  checkFact,
  declaredImpossible,
  type Env,
  envReadType,
  envUnsatisfiable,
  Exact,
  type Fact,
  guardedReadTypes,
  guardEnv,
  ImpossibleCheck,
  impossibleCheckMessage,
  impossibleChecks,
  ImpossibleReason,
  intersectFacts,
  isinstanceFact,
  Members,
  NOT_NARROWABLE,
  type NarrowedRead,
  narrowedReadError,
  narrowedReadMessage,
  narrowedTypeLabel,
  neverValue,
  notIsinstanceFact,
  tagFact,
  unionFacts,
  unresolvedType,
} from "../../src/compose/helpers/narrowing_facts.ts";
import { projectOutputContract } from "../../src/compose/helpers/output_contracts.ts";
import { readType } from "../../src/compose/helpers/types.ts";
import { forget } from "../../src/defaults.ts";
import { parseCondition } from "../../src/expr/index.ts";
import { isExactly, ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import { canonicalTypeLabel, Type, TypeCheckError, typesOverlap, ValueKind } from "../../src/typesys/values.ts";
import type { Case } from "../conformance.ts";
import { caseDocument, caseRegistry, cursorAt } from "./declarations.ts";

// `caseDocument` reloads the defaults from each case's `env:`.
afterEach(forget);

const TYPEDEFS = {
  Limit: { type: "Literal(limit)", price: "float" },
  Market: { type: "Literal(market)" },
  Leg: "Union(Limit, Market)",
  Buy: { action: "Literal(buy)", qty: "int", leg: "Leg" },
  Hold: { action: "Literal(hold)" },
  Sell: { action: "Literal(sell)", qty: "int", price: "float" },
  Decision: "Union(Buy, Hold, Sell)",
  BuyOrSell: "Union(Buy, Sell)",
  Text: { text: "str" },
  Cancel: { action: "Literal(cancel)", reason: "str" },
  Rest: "Union(Hold, Sell, Cancel)",
  Wide: { action: "Literal(buy, cancel)", qty: "int", leg: "Leg" },
  SellOrWide: "Union(Sell, Wide)",
  TextOrWide: "Union(Text, Wide)",
};
const REGISTRY = readTypedefs(TYPEDEFS);

/** Resolve one type expression against the test typedefs. */
function t(name: string): Type {
  return readType(name, REGISTRY);
}

const [BUY, HOLD, SELL] = [t("Buy"), t("Hold"), t("Sell")];
const DECISION = t("Decision");
const BUY_OR_SELL = t("BuyOrSell");
const LIMIT = t("Limit");
const TEXT = t("Text");
const MARKET = t("Market");
const OBJECT = Type.scalar(ValueKind.OBJECT);
const BOOL = t("bool");

// --------------------------------------------------------------------------- //
// The fact domain
// --------------------------------------------------------------------------- //

test("union of members is the set union and or of nullability", () => {
  const joined = unionFacts(new Members([BUY]), new Members([SELL, BUY], true));
  expect(joined).toStrictEqual(new Members([BUY, SELL], true));
});

test("union of exact facts keeps only an equal type", () => {
  expect(unionFacts(new Exact(TEXT), new Exact(TEXT))).toStrictEqual(new Exact(TEXT));
  expect(unionFacts(new Exact(TEXT), new Exact(BUY))).toBeNull();
  expect(unionFacts(new Exact(BUY), new Members([BUY]))).toBeNull();
  expect(unionFacts(new Members([BUY]), null)).toBeNull();
});

test("intersection of members is the set intersection", () => {
  const met = intersectFacts(new Members([BUY, SELL], true), new Members([SELL, HOLD], true));
  expect(met).toStrictEqual(new Members([SELL], true));
  expect(intersectFacts(new Members([BUY]), new Members([SELL]))).toStrictEqual(new Members([]));
  expect(intersectFacts(new Members([BUY]), null)).toStrictEqual(new Members([BUY]));
});

test("intersection of exact with members", () => {
  expect(intersectFacts(new Exact(BUY), new Members([BUY, SELL]))).toStrictEqual(new Members([BUY]));
  expect(intersectFacts(new Members([BUY, SELL]), new Exact(TEXT))).toStrictEqual(new Exact(TEXT));
  expect(intersectFacts(new Exact(TEXT), new Exact(TEXT))).toStrictEqual(new Exact(TEXT));
  expect(intersectFacts(new Exact(TEXT), new Exact(BUY))).toBeNull();
});

test("isinstanceFact against the base union", () => {
  expect(isinstanceFact(DECISION, BUY)).toStrictEqual(new Members([BUY]));
  expect(isinstanceFact(DECISION, BUY_OR_SELL)).toStrictEqual(new Members([BUY, SELL]));
  expect(isinstanceFact(DECISION, TEXT)).toStrictEqual(new Exact(TEXT));
  expect(isinstanceFact(TEXT, TEXT)).toBeNull();
  expect(isinstanceFact(DECISION, OBJECT)).toBeNull();
  expect(isinstanceFact(DECISION, DECISION)).toBeNull();
  expect(isinstanceFact(DECISION.replace({ nullable: true }), BUY)).toStrictEqual(new Members([BUY]));
  expect(isinstanceFact(OBJECT, TEXT)).toStrictEqual(new Exact(TEXT));
  expect(isinstanceFact(DECISION, BUY.replace({ nullable: true }))).toStrictEqual(new Members([BUY], true));
});

test("isinstanceFact against a partly shared union", () => {
  expect(isinstanceFact(DECISION, t("Rest"))).toStrictEqual(new Members([HOLD, SELL]));
  expect(isinstanceFact(DECISION, t("Optional(Rest)"))).toStrictEqual(new Members([HOLD, SELL], true));
  expect(isinstanceFact(DECISION.replace({ nullable: true }), t("Rest"))).toStrictEqual(
    new Members([HOLD, SELL]),
  );
  // `Buy` values can pass `Wide`, so keeping only `Sell` would be unsound.
  expect(typesOverlap(BUY, t("Wide"))).toBe(true);
  expect(isinstanceFact(DECISION, t("SellOrWide"))).toStrictEqual(new Exact(t("SellOrWide")));
  expect(isinstanceFact(DECISION, t("TextOrWide"))).toStrictEqual(new Exact(t("TextOrWide")));
});

test("notIsinstanceFact keeps the members the check cannot have passed", () => {
  expect(notIsinstanceFact(DECISION, BUY)).toStrictEqual(new Members([HOLD, SELL]));
  expect(notIsinstanceFact(DECISION, t("Rest"))).toStrictEqual(new Members([BUY]));
  // `Buy` only overlaps `Wide`, so a `Buy` value may fail the check: it is kept.
  expect(notIsinstanceFact(DECISION, t("SellOrWide"))).toStrictEqual(new Members([BUY, HOLD]));
  expect(notIsinstanceFact(DECISION, DECISION)).toStrictEqual(new Members([]));
});

test("notIsinstanceFact nullability", () => {
  const nullable = DECISION.replace({ nullable: true });
  expect(notIsinstanceFact(nullable, BUY)).toStrictEqual(new Members([HOLD, SELL], true));
  expect(notIsinstanceFact(nullable, t("Optional(Buy)"))).toStrictEqual(new Members([HOLD, SELL]));
  expect(notIsinstanceFact(nullable, t("Optional(Text)"))).toStrictEqual(new Members([BUY, HOLD, SELL]));
});

test("notIsinstanceFact gives nothing without a union or an exclusion", () => {
  expect(notIsinstanceFact(BUY, BUY)).toBeNull();
  expect(notIsinstanceFact(OBJECT, BUY)).toBeNull();
  expect(notIsinstanceFact(DECISION, TEXT)).toBeNull();
  expect(notIsinstanceFact(DECISION.replace({ nullable: true }), TEXT)).toBeNull();
});

test("tagFact selects members by tag", () => {
  expect(tagFact(DECISION, "action", "buy", true)).toStrictEqual(new Members([BUY]));
  expect(tagFact(DECISION, "action", "buy", false)).toStrictEqual(new Members([HOLD, SELL]));
  expect(tagFact(DECISION, "action", "zzz", true)).toStrictEqual(new Members([]));
  expect(tagFact(DECISION, "action", "zzz", false)).toStrictEqual(new Members([BUY, HOLD, SELL]));
});

test("tagFact nullability", () => {
  const nullable = DECISION.replace({ nullable: true });
  expect(tagFact(nullable, "action", "buy", true)).toStrictEqual(new Members([BUY]));
  expect(tagFact(nullable, "action", "buy", false)).toStrictEqual(new Members([HOLD, SELL], true));
});

test("tagFact gives nothing off a tag field or a union", () => {
  expect(tagFact(DECISION, "qty", "1", true)).toBeNull();
  expect(tagFact(BUY, "action", "buy", true)).toBeNull();
  expect(tagFact(TEXT, "text", "x", false)).toBeNull();
});

test("checkFact dispatches on the check", () => {
  const buy = new IsinstanceCheck("v", "Buy");
  expect(checkFact(buy, DECISION, true, REGISTRY)).toStrictEqual(new Members([BUY]));
  expect(checkFact(buy, DECISION, false, REGISTRY)).toStrictEqual(new Members([HOLD, SELL]));
  expect(checkFact(buy, BUY, true, REGISTRY)).toBeNull();
  const tag = new TagCheck("v", "action", "sell");
  expect(checkFact(tag, DECISION, true, REGISTRY)).toStrictEqual(new Members([SELL]));
  expect(checkFact(tag, DECISION, false, REGISTRY)).toStrictEqual(new Members([BUY, HOLD]));
});

test("checkFact on an unproven base", () => {
  expect(checkFact(new IsinstanceCheck("v", "Buy"), null, true, REGISTRY)).toStrictEqual(new Exact(BUY));
  expect(checkFact(new IsinstanceCheck("v", "Buy"), null, false, REGISTRY)).toBeNull();
  expect(checkFact(new TagCheck("v", "action", "buy"), null, true, REGISTRY)).toBeNull();
});

test("checkFact rejects an unknown type name", () => {
  expect(() => checkFact(new IsinstanceCheck("v", "Nope"), DECISION, true, REGISTRY)).toThrow(TypeCheckError);
});

// --------------------------------------------------------------------------- //
// Guards into facts
// --------------------------------------------------------------------------- //

/** What `condition` proves when it is true. */
function g(condition: string): Guard {
  return proves(parseCondition(condition), true);
}

/** A `BaseTypeOf` over `declared`, keyed by dotted base path; other bases are unproven. */
function types(declared: Record<string, Type> = {}): BaseTypeOf {
  const byBase = new Map(Object.entries(declared));
  return (base) => byBase.get(base) ?? null;
}

/** The facts `condition` proves when true, with the bases typed by `declared`. */
function env(condition: string, declared: Record<string, Type> = {}): Env {
  return guardEnv(g(condition), types(declared), REGISTRY);
}

test("guardEnv of one check", () => {
  expect(env("isinstance(v, Buy)", { v: DECISION })).toStrictEqual(new Map([["v", new Members([BUY])]]));
  expect(env("not isinstance(v, Buy)", { v: DECISION })).toStrictEqual(
    new Map([["v", new Members([HOLD, SELL])]]),
  );
  expect(env("v.action == 'sell'", { v: DECISION })).toStrictEqual(new Map([["v", new Members([SELL])]]));
  expect(env("isinstance(v, Buy)", { v: BUY })).toStrictEqual(new Map());
  expect(env("isinstance(v, Buy)")).toStrictEqual(new Map([["v", new Exact(BUY)]]));
  expect(env("v.action == 'sell'")).toStrictEqual(new Map());
  expect(guardEnv(NOTHING, types(), REGISTRY)).toStrictEqual(new Map());
});

test("guardEnv meets and joins on one base", () => {
  expect(env("isinstance(v, BuyOrSell) and v.action != 'buy'", { v: DECISION })).toStrictEqual(
    new Map([["v", new Members([SELL])]]),
  );
  expect(env("isinstance(v, Buy) or isinstance(v, Sell)", { v: DECISION })).toStrictEqual(
    new Map([["v", new Members([BUY, SELL])]]),
  );
  expect(env("not (isinstance(v, Hold) or v.action == 'sell')", { v: DECISION })).toStrictEqual(
    new Map([["v", new Members([BUY])]]),
  );
});

test("guardEnv over two bases", () => {
  expect(env("isinstance(v, Buy) and isinstance(w, Sell)", { v: DECISION, w: DECISION })).toStrictEqual(
    new Map([
      ["v", new Members([BUY])],
      ["w", new Members([SELL])],
    ]),
  );
  // `w` is unconstrained when `v` is a `Buy`, so only the joined facts on `v` remain.
  expect(env("isinstance(v, Buy) or isinstance(w, Sell)", { v: DECISION, w: DECISION })).toStrictEqual(
    new Map([["v", new Members([BUY, HOLD, SELL])]]),
  );
  // Each way through the `or` proves something about `v`.
  expect(
    env("isinstance(v, Buy) and isinstance(w, Sell) or isinstance(v, Hold)", { v: DECISION, w: DECISION }),
  ).toStrictEqual(new Map([["v", new Members([BUY, HOLD])]]));
});

test("guardEnv skips unknown types and bases it cannot narrow", () => {
  expect(env("isinstance(v, Nope) and isinstance(w, Buy)", { v: DECISION, w: DECISION })).toStrictEqual(
    new Map([["w", new Members([BUY])]]),
  );
  expect(unresolvedType(g("isinstance(v, Nope) and isinstance(w, Buy)"), REGISTRY)).toBe(true);
  expect(unresolvedType(g("isinstance(w, Buy) and v.action == 'x'"), REGISTRY)).toBe(false);
  expect(unresolvedType(g("isinstance(w, Buy) or isinstance(v, int)"), REGISTRY)).toBe(false);

  const baseTypeOf: BaseTypeOf = (base) => (base === "v" ? DECISION : NOT_NARROWABLE);
  const guard = g("isinstance(v, Buy) and isinstance(x, Sell)");
  expect(guardEnv(guard, baseTypeOf, REGISTRY)).toStrictEqual(new Map([["v", new Members([BUY])]]));
});

test("guardEnv types a member-only base through the checks before it", () => {
  expect(env("v.action == 'buy' and v.leg.type == 'limit'", { v: DECISION })).toStrictEqual(
    new Map([
      ["v", new Members([BUY])],
      ["v.leg", new Members([LIMIT])],
    ]),
  );
  expect(env("isinstance(v, Buy) and not isinstance(v.leg, Limit)", { v: DECISION })).toStrictEqual(
    new Map([
      ["v", new Members([BUY])],
      ["v.leg", new Members([MARKET])],
    ]),
  );
  // Into the parts of a nested `or`, and through the `and` inside it.
  expect(
    env("v.action == 'buy' and (v.leg.type == 'limit' or v.leg.type == 'market')", { v: DECISION }),
  ).toStrictEqual(
    new Map([
      ["v", new Members([BUY])],
      ["v.leg", new Members([LIMIT, MARKET])],
    ]),
  );
  // With no check before it, the member-only base stays unproven.
  expect(env("v.leg.type == 'limit' and v.action == 'buy'", { v: DECISION })).toStrictEqual(
    new Map([["v", new Members([BUY])]]),
  );
  expect(env("isinstance(v.leg, Limit) and v.action == 'buy'", { v: DECISION })).toStrictEqual(
    new Map<string, Fact>([
      ["v", new Members([BUY])],
      ["v.leg", new Exact(LIMIT)],
    ]),
  );
});

test("envUnsatisfiable", () => {
  expect(envUnsatisfiable(new Map([["v", new Members([])]]))).toBe(true);
  expect(
    envUnsatisfiable(
      new Map<string, Fact>([
        ["w", new Exact(BUY)],
        ["v", new Members([])],
      ]),
    ),
  ).toBe(true);
  expect(envUnsatisfiable(new Map([["v", new Members([], true)]]))).toBe(false);
  expect(envUnsatisfiable(new Map([["v", new Exact(BUY)]]))).toBe(false);
  expect(envUnsatisfiable(new Map())).toBe(false);
  expect(envUnsatisfiable(env("isinstance(v, Buy) and v.action == 'hold'", { v: DECISION }))).toBe(true);
});

/** The candidate types of `read` under what `condition` proves. */
function reads(read: string, condition: string, declared: Record<string, Type> = {}): NarrowedRead[] {
  return guardedReadTypes(read, g(condition), types(declared), REGISTRY);
}

test("guardedReadTypes reads below the narrowed base", () => {
  expect(reads("v.qty", "isinstance(v, Buy)", { v: DECISION })).toStrictEqual([[BUY, ["qty"], "v"]]);
  expect(reads("v.leg.price", "isinstance(v, Buy)", { v: DECISION })).toStrictEqual([
    [BUY, ["leg", "price"], "v"],
  ]);
  expect(reads("v.qty", "isinstance(v, Buy) or isinstance(v, Sell)", { v: DECISION })).toStrictEqual([
    [BUY_OR_SELL, ["qty"], "v"],
  ]);
});

test("guardedReadTypes takes the longest base", () => {
  const condition = "isinstance(v, Buy) and isinstance(v.leg, Limit)";
  expect(reads("v.leg.price", condition, { v: DECISION })).toStrictEqual([
    [LIMIT, ["price"], "v.leg"],
    [BUY, ["leg", "price"], "v"],
  ]);
});

test("guardedReadTypes without a fact for the read", () => {
  expect(reads("v", "isinstance(v, Buy)", { v: DECISION })).toStrictEqual([]);
  expect(reads("w.qty", "isinstance(v, Buy)", { v: DECISION })).toStrictEqual([]);
  expect(reads("vv.qty", "isinstance(v, Buy)", { v: DECISION })).toStrictEqual([]);
  expect(reads("v.qty", "n > 1", { v: DECISION })).toStrictEqual([]);
  // `isinstance(v, Buy) and n > 1` may be false through `n > 1`, so `v` keeps every member.
  expect(reads("v.qty", "not (isinstance(v, Buy) and n > 1)", { v: DECISION })).toStrictEqual([]);
});

test("guardedReadTypes keeps each conjunct's candidate", () => {
  // Two unrelated exact types lose their meet, but each still narrows the read.
  expect(reads("v.qty", "isinstance(v, Buy) and isinstance(v, Text)", { v: OBJECT })).toStrictEqual([
    [BUY, ["qty"], "v"],
    [TEXT, ["qty"], "v"],
  ]);
});

test("envReadType", () => {
  const facts: Env = new Map([
    ["v", new Members([BUY])],
    ["v.leg", new Members([LIMIT])],
  ]);
  expect(envReadType("v.leg.price", facts)).toStrictEqual([LIMIT, ["price"], "v.leg"]);
  expect(envReadType("v.qty", facts)).toStrictEqual([BUY, ["qty"], "v"]);
  expect(envReadType("v", facts)).toBeNull();
  expect(envReadType("w.qty", facts)).toBeNull();
  expect(envReadType("v.qty", new Map([["v", new Members([])]]))).toBeNull();
});

/** The guard each read in `condition` is checked under, with the bases typed by `declared`. */
function checked(
  condition: string,
  declared: Record<string, Type> = {},
  routing: Env | null = null,
): [string, Guard | null][] {
  return checkedRefs(parseCondition(condition), types(declared), REGISTRY, { routing });
}

test("checkedRefs keeps a guard some value satisfies", () => {
  expect(checked("isinstance(v, Buy) and v.qty > 0", { v: DECISION })).toStrictEqual([
    ["v", NOTHING],
    ["v.qty", new Holds(new IsinstanceCheck("v", "Buy"), true)],
  ]);
});

test("a read after a reported check is not checked", () => {
  expect(checked("isinstance(v, Buy) and v.action == 'hold' and v.qty > 0", { v: DECISION }).at(-1)).toStrictEqual([
    "v.qty",
    null,
  ]);
  expect(checked("v.action != 'biy' or v.qty > 0", { v: DECISION }).at(-1)).toStrictEqual(["v.qty", null]);
  expect(checked("v.action == 'zzz' and v.qty > 0", { v: DECISION }).at(-1)).toStrictEqual(["v.qty", null]);
  // A negated check the declared type always passes is reported, so what follows is dead.
  expect(checked("not isinstance(v, Decision) and v.qty > 0", { v: DECISION }).at(-1)).toStrictEqual([
    "v.qty",
    null,
  ]);
  const routing: Env = new Map([["v", new Members([BUY])]]);
  expect(checked("isinstance(v, Sell) and v.qty > 0", { v: DECISION }, routing).at(-1)).toStrictEqual([
    "v.qty",
    null,
  ]);
});

describe("a read that needs a reported check to hold is not checked", () => {
  const rows: [string, Type][] = [
    // No overlap: the check would narrow `v` to `Text`, a type it can never be.
    ["isinstance(v, Text) and v.qty > 0", DECISION],
    ["(isinstance(v, Text) or isinstance(v, Text)) and v.qty > 0", DECISION],
    // A negated check a non-union type always passes.
    ["not isinstance(v, Buy) and v.qty > 0", BUY],
    ["not isinstance(v, Decision) and v.qty > 0", BUY],
    // No overlap, on a record and on a union.
    ["isinstance(v, Hold) and v.qty > 0", BUY],
    ["isinstance(v, str) and v.qty > 0", DECISION],
    // A text that is no tag never equals the tag field.
    ["v.action == 'biy' and v.qty > 0", DECISION],
    ["not (v.action != 'biy') and v.qty > 0", DECISION],
  ];
  for (const [condition, declared] of rows) {
    test(condition, () => {
      expect(checked(condition, { v: declared }).at(-1)).toStrictEqual(["v.qty", null]);
    });
  }
});

test("a read that does not need a reported check is checked", () => {
  // A `!=` with a text that is no tag is always true, so the read after it runs.
  expect(checked("v.action != 'biy' and v.qty > 0", { v: DECISION }).at(-1)![1]).not.toBeNull();
  // The read runs when the reported check is false, as it always is.
  expect(checked("isinstance(v, Text) or v.qty > 0", { v: DECISION }).at(-1)![1]).not.toBeNull();
  // Reporting one occurrence of a check leaves another occurrence of it alone.
  const condition = "(isinstance(v, Buy) and v.action == 'hold') or (v.action == 'hold' and v.qty > 0)";
  expect(checked(condition, { v: DECISION }).at(-1)![1]).not.toBeNull();
});

test("a read after a check the known type decides is checked without it", () => {
  expect(checked("isinstance(v, Decision) or v.nosuch > 0", { v: DECISION }).at(-1)).toStrictEqual([
    "v.nosuch",
    NOTHING,
  ]);
  // A false check on a non-union base proves nothing, so its guard is kept.
  const [read, guard] = checked("isinstance(v, Buy) or v.qty > 0", { v: BUY }).at(-1)!;
  expect(guard).not.toBeNull();
  expect(guardedReadTypes(read, guard!, types({ v: BUY }), REGISTRY)).toStrictEqual([]);
  // The other checks before the read still narrow it.
  expect(
    checked("v.action == 'buy' or isinstance(v, Decision) or v.qty > 0", { v: DECISION }).at(-1),
  ).toStrictEqual(["v.qty", new Holds(new TagCheck("v", "action", "buy"), false)]);
  const routing: Env = new Map([["v", new Members([BUY])]]);
  expect(checked("isinstance(v, Buy) or v.qty > 0", { v: DECISION }, routing).at(-1)).toStrictEqual([
    "v.qty",
    NOTHING,
  ]);
  expect(checked("isinstance(v, Buy) or v.qty > 0", { v: DECISION }).at(-1)).toStrictEqual([
    "v.qty",
    new Holds(new IsinstanceCheck("v", "Buy"), false),
  ]);
  // A check the checks before it decide proves nothing either, so the read is checked under the
  // checks before it.
  expect(
    checked("isinstance(v, Buy) and (isinstance(v, Buy) or v.qty > 0)", { v: DECISION }).at(-1),
  ).toStrictEqual(["v.qty", new Holds(new IsinstanceCheck("v", "Buy"), true)]);
  const condition = "isinstance(v, Buy) or isinstance(v, Hold) or isinstance(v, Sell) or v.qty > 0";
  expect(checked(condition, { v: DECISION }).at(-1)).toStrictEqual([
    "v.qty",
    new AllOf([new Holds(new IsinstanceCheck("v", "Buy"), false), new Holds(new IsinstanceCheck("v", "Hold"), false)]),
  ]);
});

// --------------------------------------------------------------------------- //
// Impossible checks
// --------------------------------------------------------------------------- //

const MORE_REGISTRY = readTypedefs({ ...TYPEDEFS, MaybeText: "Optional(Text)", OptBuy: "Optional(Buy)" });

/** The impossible-check messages of `condition`, with the bases typed by `declared`. */
function impossible(
  condition: string,
  declared: Record<string, Type> = {},
  options: { registry?: TypeRegistry; routing?: Env } = {},
): string[] {
  const { registry = REGISTRY, routing } = options;
  const found = impossibleChecks(parseCondition(condition), types(declared), registry, { routing });
  return found.map(impossibleCheckMessage);
}

test("a check the checks before it rule out is impossible", () => {
  const found = impossibleChecks(
    parseCondition("isinstance(v, Buy) and v.action == 'hold'"),
    types({ v: DECISION }),
    REGISTRY,
  );
  expect(found.map((c) => [c.text, c.base, c.reason])).toEqual([['v.action == "hold"', "v", ImpossibleReason.CONTEXT]]);
  expect(impossibleCheckMessage(found[0]!)).toBe(
    '`v.action == "hold"` can never be true here: v is Buy, by the checks before it',
  );
  expect(impossible("not isinstance(v, Buy) or v.action == 'hold'", { v: DECISION })).toEqual([
    '`v.action == "hold"` can never be true here: v is Buy, by the checks before it',
  ]);
  expect(impossible("v.action != 'buy' and isinstance(v, Buy)", { v: DECISION })).toEqual([
    "`isinstance(v, Buy)` can never be true here: v is Hold | Sell, by the checks before it",
  ]);
});

test("a check the checks before it make always true says so", () => {
  expect(impossible("isinstance(v, Buy) and not isinstance(v, Buy)", { v: DECISION })).toEqual([
    "`not isinstance(v, Buy)` can never be true here: v is Buy, by the checks before it, " +
      "so `isinstance(v, Buy)` is always true",
  ]);
  expect(impossible("isinstance(v, Buy) and v.action != 'buy'", { v: DECISION })).toEqual([
    '`v.action != "buy"` can never be true here: v is Buy, by the checks before it, ' +
      'so `v.action == "buy"` is always true',
  ]);
});

test("a tag value no member has is reported", () => {
  const message =
    'compares v.action with "biy", which is not one of its tags (buy, hold, sell); did you mean "buy"?';
  expect(impossible("v.action == 'biy'", { v: DECISION })).toEqual([`\`v.action == "biy"\` ${message}`]);
  expect(impossible("v.action != 'biy'", { v: DECISION })).toEqual([`\`v.action != "biy"\` ${message}`]);
  expect(impossible("v.action == 'zzzz'", { v: DECISION })).toEqual([
    '`v.action == "zzzz"` compares v.action with "zzzz", which is not one of its tags (buy, hold, sell)',
  ]);
  // A field only `Buy` has is typed through the check before it.
  expect(impossible("v.action == 'buy' and v.leg.type == 'lmit'", { v: DECISION })).toEqual([
    '`v.leg.type == "lmit"` compares v.leg.type with "lmit", which is not one of its ' +
      'tags (limit, market); did you mean "limit"?',
  ]);
});

test("an isinstance the declared type rules out is impossible", () => {
  const message =
    "`isinstance(v, Text)` can never be true: v is declared Decision, and no Decision value fits Text";
  expect(impossible("isinstance(v, Text)", { v: DECISION })).toEqual([message]);
  expect(impossible("not not isinstance(v, Text)", { v: DECISION })).toEqual([message]);
  expect(impossible("not isinstance(v, Text)", { v: DECISION })).toEqual([]);
  // A missing value is checked as null, which a nullable type admits.
  expect(impossible("isinstance(v, MaybeText)", { v: DECISION }, { registry: MORE_REGISTRY })).toEqual([]);
  expect(impossible("isinstance(v, int)", { v: BOOL })).toEqual([
    "`isinstance(v, int)` can never be true: v is declared bool, and no bool value fits int",
  ]);
});

test("an isinstance on a nested base the type rules out is impossible", () => {
  expect(impossible("isinstance(v.leg, Text)", { v: BUY, "v.leg": t("Leg") })).toEqual([
    "`isinstance(v.leg, Text)` can never be true: v.leg is declared Leg, and no Leg value fits Text",
  ]);
  // `leg` is only on `Buy`, so its type comes from the check before it.
  expect(impossible("isinstance(v, Buy) and isinstance(v.leg, Text)", { v: DECISION })).toEqual([
    "`isinstance(v.leg, Text)` can never be true: v.leg is Leg here, " +
      "by the checks before it, and no Leg value fits Text",
  ]);
});

test("a negated isinstance the declared type always passes is impossible", () => {
  expect(impossible("not isinstance(v, Buy)", { v: BUY })).toEqual([
    "`not isinstance(v, Buy)` can never be true: v is declared Buy, so `isinstance(v, Buy)` is always true",
  ]);
  expect(impossible("not isinstance(v, BuyOrSell)", { v: BUY })).toEqual([
    "`not isinstance(v, BuyOrSell)` can never be true: v is declared Buy, " +
      "so `isinstance(v, BuyOrSell)` is always true",
  ]);
  expect(impossible("not isinstance(v, Decision)", { v: DECISION })).toEqual([
    "`not isinstance(v, Decision)` can never be true: v is declared Decision, " +
      "so `isinstance(v, Decision)` is always true",
  ]);
  expect(impossible("not isinstance(v, Buy)", { v: BUY.replace({ nullable: true }) })).toEqual([]);
  expect(impossible("isinstance(v, Buy)", { v: BUY })).toEqual([]);
});

test("a negated isinstance of a nullable typedef", () => {
  const nullableBuy = BUY.replace({ nullable: true });
  const expected = "so `isinstance(v, OptBuy)` is always true";
  expect(impossible("not isinstance(v, OptBuy)", { v: nullableBuy }, { registry: MORE_REGISTRY })).toEqual([
    `\`not isinstance(v, OptBuy)\` can never be true: v is declared Optional(Buy), ${expected}`,
  ]);
  expect(impossible("not isinstance(v, OptBuy)", { v: BUY }, { registry: MORE_REGISTRY })).toEqual([
    `\`not isinstance(v, OptBuy)\` can never be true: v is declared Buy, ${expected}`,
  ]);
});

test("possible checks are not reported", () => {
  expect(impossible("isinstance(v, Buy) or isinstance(v, Sell)", { v: DECISION })).toEqual([]);
  expect(impossible("isinstance(v, Buy) and v.action == 'buy'", { v: DECISION })).toEqual([]);
  expect(impossible("(v.action == 'biy') == true", { v: DECISION })).toEqual([]);
  expect(impossible("v.qty == 'x'", { v: DECISION })).toEqual([]);
  // An always-true check is no error on its own.
  expect(impossible("isinstance(v, Decision)", { v: DECISION })).toEqual([]);
  expect(impossible("isinstance(v, Decision) and v.qty > 0", { v: DECISION })).toEqual([]);
});

test("unproven bases and lost facts prove nothing impossible", () => {
  expect(impossible("isinstance(v, Buy) and v.action == 'hold'", { v: OBJECT })).toEqual([]);
  expect(impossible("isinstance(v, Buy) and v.action == 'hold'")).toEqual([]);
  expect(impossible("isinstance(v, Buy) and isinstance(v, Text)", { v: OBJECT })).toEqual([]);
  expect(impossible("isinstance(v, Buy) and isinstance(v, Text)")).toEqual([]);
  expect(impossible("isinstance(v, Nope)", { v: DECISION })).toEqual([]);
  expect(impossible("isinstance(v, Nope) and not isinstance(v, Nope)", { v: DECISION })).toEqual([]);

  const tree = parseCondition("isinstance(v, Text) and v.action == 'biy'");
  expect(impossibleChecks(tree, () => NOT_NARROWABLE, REGISTRY)).toEqual([]);
});

test("each impossible check is reported once and only first", () => {
  expect(impossible("v.action == 'biy' or v.action == 'biy'", { v: DECISION })).toEqual([
    '`v.action == "biy"` compares v.action with "biy", which is not one of its tags ' +
      '(buy, hold, sell); did you mean "buy"?',
  ]);
  // Once the checks before a literal can't all hold, it is not reported again.
  expect(
    impossible("isinstance(v, Buy) and v.action == 'hold' and v.action == 'sell'", { v: DECISION }),
  ).toEqual(['`v.action == "hold"` can never be true here: v is Buy, by the checks before it']);
});

test("code after an always-true check is not reported", () => {
  // The not-a-tag text is wrong whatever runs before it.
  expect(impossible("isinstance(v, Decision) or v.action == 'biy'", { v: DECISION })).toEqual([
    '`v.action == "biy"` compares v.action with "biy", which is not one of its tags ' +
      '(buy, hold, sell); did you mean "buy"?',
  ]);
  // A read after a check the declared type decides is checked by its site, unnarrowed.
  expect(impossible("isinstance(v, Decision) or v.nosuch > 0", { v: DECISION })).toEqual([]);
  expect(impossible("v.action == 'buy' or isinstance(v, Decision) or v.qty > w", { v: DECISION })).toEqual([]);
  expect(impossible("isinstance(v, Text) or isinstance(v, Decision) or v.x", { v: DECISION })).toEqual([
    "`isinstance(v, Text)` can never be true: v is declared Decision, and no Decision value fits Text",
  ]);
});

test("a check the routing rules out is impossible", () => {
  expect(impossible("v.action == 'hold'", { v: DECISION }, { routing: new Map([["v", new Members([BUY])]]) })).toEqual(
    ['`v.action == "hold"` can never be true here: v is Buy, by the case branch that routes this node'],
  );
  expect(
    impossible("v.action == 'buy'", { v: DECISION }, { routing: new Map([["v", new Members([BUY, SELL])]]) }),
  ).toEqual([]);
  expect(
    impossible(
      "not isinstance(v, Sell) and isinstance(v, Hold)",
      { v: DECISION },
      { routing: new Map([["v", new Members([BUY, SELL])]]) },
    ),
  ).toEqual([
    "`isinstance(v, Hold)` can never be true here: v is Buy | Sell, by the case branch that routes this node",
  ]);
  // A node the routing never runs gets no impossible checks.
  expect(impossible("v.action == 'hold'", { v: DECISION }, { routing: new Map([["v", new Members([])]]) })).toEqual(
    [],
  );
});

test("routing and the checks together use the checks' wording", () => {
  expect(
    impossible(
      "not isinstance(v, Buy) and v.action != 'sell'",
      { v: DECISION },
      { routing: new Map([["v", new Members([BUY, SELL])]]) },
    ),
  ).toEqual([
    '`v.action != "sell"` can never be true here: v is Sell, by the checks before it, ' +
      'so `v.action == "sell"` is always true',
  ]);
  expect(
    impossible("isinstance(v, Buy) or v.qty > 0", { v: DECISION }, { routing: new Map([["v", new Members([BUY])]]) }),
  ).toEqual([]);
});

// --------------------------------------------------------------------------- //
// Messages
// --------------------------------------------------------------------------- //

test("narrowedTypeLabel names members", () => {
  expect(narrowedTypeLabel(BUY, REGISTRY)).toBe("Buy");
  expect(narrowedTypeLabel(BUY_OR_SELL, REGISTRY)).toBe("Buy | Sell");
  expect(narrowedTypeLabel(BUY_OR_SELL.replace({ nullable: true }), REGISTRY)).toBe("Optional(Buy | Sell)");
  expect(narrowedTypeLabel(BUY.replace({ nullable: true }), REGISTRY)).toBe("Optional(Buy)");
  const anonymous = TEXT.replace({ fields: new Map([["note", Type.scalar(ValueKind.STRING)]]) });
  expect(narrowedTypeLabel(anonymous, REGISTRY)).toBe(canonicalTypeLabel(anonymous));
  expect(narrowedTypeLabel(t("SellOrWide"), REGISTRY)).toBe("Wide | Sell");
});

test("narrowedReadMessage", () => {
  expect(narrowedReadMessage("r.answr", "answr", "Done", "r", ["answer"])).toBe(
    "reference ${r.answr} reads field 'answr', which Done doesn't have; r is Done here, " +
      "by the checks before this read; did you mean 'answer'?",
  );
  expect(narrowedReadMessage("v.qty", "qty", "Hold", "v", ["action"])).toBe(
    "reference ${v.qty} reads field 'qty', which Hold doesn't have; v is Hold here, " +
      "by the checks before this read",
  );
});

test("isinstanceMembers keeps the shared members only when nothing else can pass", () => {
  expect(isinstanceMembers(DECISION, t("Rest"))).toStrictEqual([HOLD, SELL]);
  expect(isinstanceMembers(DECISION, BUY)).toStrictEqual([BUY]);
  expect(isinstanceMembers(DECISION, t("SellOrWide"))).toBeNull();
  expect(isinstanceMembers(DECISION, TEXT)).toBeNull();
  expect(isinstanceMembers(TEXT, TEXT)).toBeNull();
});

test("narrowedReadError names the narrowed type only for a field it lacks", () => {
  expect(narrowedReadError("v.qtty", [BUY, ["qtty"], "v"], "plain", REGISTRY)).toBe(
    "reference ${v.qtty} reads field 'qtty', which Buy doesn't have; v is Buy here, " +
      "by the checks before this read; did you mean 'qty'?",
  );
  expect(narrowedReadError("v.price", [BUY_OR_SELL, ["price"], "v"], "plain", REGISTRY)).toBe("plain");
  expect(narrowedReadError("v.nope", [BUY_OR_SELL, ["nope"], "v"], "plain", REGISTRY)).toBe(
    "reference ${v.nope} reads field 'nope', which Buy | Sell doesn't have; " +
      "v is Buy | Sell here, by the checks before this read",
  );
  expect(narrowedReadError("v.leg.prise", [BUY, ["leg", "prise"], "v"], "plain", REGISTRY)).toBe("plain");
  expect(narrowedReadError("v.x", [BOOL, ["x"], "v"], "plain", REGISTRY)).toBe("plain");
});

// --------------------------------------------------------------------------- //
// Unknown check kinds
// --------------------------------------------------------------------------- //

/** A check kind no dispatch knows, as a new form would be before it is wired in. */
class OtherCheck {
  readonly base: string;

  constructor(base: string) {
    this.base = base;
    Object.freeze(this);
  }
}

describe("an unknown check kind fails loudly", () => {
  const other = new OtherCheck("v") as unknown as Check;
  const literal = new CheckLiteral(NOTHING, other, true, false);
  const reported = new ImpossibleCheck("x", "v", ImpossibleReason.CONTEXT);
  const sites: [string, () => unknown][] = [
    ["checkFact", () => checkFact(other, DECISION, true, REGISTRY)],
    ["checkText", () => checkText(other)],
    ["declaredImpossible", () => declaredImpossible(literal, DECISION, true, REGISTRY)],
    ["neverValue", () => neverValue(literal, reported)],
    ["unresolvedType", () => unresolvedType(new Holds(other, true), REGISTRY)],
  ];
  for (const [site, call] of sites) {
    test(site, () => {
      expect(call).toThrow(TypeError);
      expect(call).toThrow(/OtherCheck/);
    });
  }
});

// --------------------------------------------------------------------------- //
// Harvested conformance cases
// --------------------------------------------------------------------------- //

/** One harvested case whose message a narrowing call decides. */
interface NarrowingRow {
  /** The case, `group/name`. */
  id: string;
  /** The path from the document root to the condition. */
  at: (string | number)[];
  /** The site's prefix on the expected message (on its last line, for a multi-line one). */
  prefix: string;
  /** The declared type text of each base path the site types. */
  bases: Record<string, string>;
  /** The condition of the case branch that routes the site; for an `on:` route, its tag check. */
  routing?: string;
  /** The field read whose narrowed error the case pins; without it, the first impossible check. */
  read?: string;
}

/** A `BaseTypeOf` over `bases`: the longest declared prefix of a base, projected to the rest. */
function caseBaseTypeOf(bases: Record<string, string>, registry: TypeRegistry): BaseTypeOf {
  return (base) => {
    const parts = base.split(".");
    for (let cut = parts.length; cut > 0; cut--) {
      const text = bases[parts.slice(0, cut).join(".")];
      if (text === undefined) continue;
      const projected = projectOutputContract(new ValueContract(readType(text, registry)), parts.slice(cut));
      return isExactly(projected, ValueContract) ? projected.type : null;
    }
    return null;
  };
}

/** The narrowing text the site produces for `row`: its first impossible check, or its read's error. */
function producedText(row: NarrowingRow, condition: string, registry: TypeRegistry): string {
  const tree = parseCondition(condition);
  const baseTypeOf = caseBaseTypeOf(row.bases, registry);
  const routing: Env =
    row.routing === undefined ? new Map() : guardEnv(proves(parseCondition(row.routing), true), baseTypeOf, registry);
  if (row.read === undefined) {
    const found = impossibleChecks(tree, baseTypeOf, registry, { routing });
    expect(found.length).toBeGreaterThan(0);
    return impossibleCheckMessage(found[0]!);
  }
  const byRouting = row.routing !== undefined;
  let candidate: NarrowedRead | null | undefined;
  if (byRouting) {
    candidate = envReadType(row.read, routing);
  } else {
    const guard = checkedRefs(tree, baseTypeOf, registry, { routing }).find(([ref]) => ref === row.read)?.[1];
    expect(guard).toBeDefined();
    expect(guard).not.toBeNull();
    candidate = guardedReadTypes(row.read, guard!, baseTypeOf, registry)[0];
  }
  expect(candidate).toBeDefined();
  expect(candidate).not.toBeNull();
  return narrowedReadError(row.read, candidate!, "<unused>", registry, { byRouting });
}

/** Check that the case's message, or its last line, is the row's prefix plus the produced text. */
function runNarrowingRow(row: NarrowingRow): void {
  const { case_, document } = caseDocument(row.id);
  const registry = caseRegistry(document);
  const condition = cursorAt(document, row.at).value;
  expect(typeof condition).toBe("string");
  const produced = producedText(row, condition as string, registry);
  const expected = (case_["expect"] as Case)["load"] as Case;
  const message = (expected["diagnostics"] as Case[])[0]!["message"] as string;
  expect(message.split("\n").at(-1)).toBe(row.prefix + produced);
}

const FIELD_READ_SITES = "load-core/compose_field_read_sites__";
const LOOP_BUILD = "load-loop/engine_loop_build__";
const NULL_NARROWING = "load-types/compose_null_narrowing__";
const UNION_NARROWING = "load-types/compose_union_narrowing__";

test.each<NarrowingRow>([
  {
    id: `${FIELD_READ_SITES}flow_assert_compares_with_a_non_tag`,
    at: ["asserts", 0],
    prefix: `assert "input.o.action == 'hold'": `,
    bases: { "input.o": "Order" },
  },
  {
    id: `${FIELD_READ_SITES}flow_assert_shorthand_check_is_reported_as_written`,
    at: ["asserts", 0],
    prefix: "assert 'isinstance(o, Rec)': ",
    bases: { o: "Order" },
  },
  {
    id: `${FIELD_READ_SITES}impossible_flow_assert_check`,
    at: ["asserts", 0],
    prefix: "assert 'isinstance(d.output, Buy) and isinstance(d.output, Sell)': ",
    bases: { "d.output": "Order" },
  },
  {
    id: `${FIELD_READ_SITES}impossible_node_assert_check`,
    at: ["nodes", "c", "asserts", 0],
    prefix: "  node 'c' assert: ",
    bases: { v: "Order" },
  },
  {
    id: `${FIELD_READ_SITES}node_assert_check_no_value_fits`,
    at: ["nodes", "c", "asserts", 0],
    prefix: "  node 'c' assert: ",
    bases: { output: "Order" },
  },
  {
    id: `${FIELD_READ_SITES}node_assert_compares_with_a_non_tag`,
    at: ["nodes", "c", "asserts", 0],
    prefix: "  node 'c' assert: ",
    bases: { v: "Order" },
  },
  {
    id: `${FIELD_READ_SITES}node_assert_negated_check_always_true`,
    at: ["nodes", "c", "asserts", 0],
    prefix: "  node 'c' assert: ",
    bases: { output: "Order" },
  },
  {
    id: `${FIELD_READ_SITES}node_assert_read_on_the_other_member`,
    at: ["nodes", "c", "asserts", 0],
    prefix: "  node 'c' assert: ",
    bases: { v: "Order" },
    read: "v.qty",
  },
  {
    id: `${LOOP_BUILD}predicate_isinstance_of_a_disjoint_type_is_rejected`,
    at: ["nodes", "ask", "until"],
    prefix: "loop node 'ask': `until:` ",
    bases: { r: "Done" },
  },
  {
    id: `${LOOP_BUILD}predicate_isinstance_ruled_out_by_an_earlier_check`,
    at: ["nodes", "ask", "until"],
    prefix: "loop node 'ask': `until:` ",
    bases: { r: "Step" },
  },
  {
    id: `${LOOP_BUILD}predicate_negated_isinstance_of_the_declared_type_is_rejected`,
    at: ["nodes", "ask", "until"],
    prefix: "loop node 'ask': `until:` ",
    bases: { r: "Done" },
  },
  {
    id: `${LOOP_BUILD}predicate_tag_comparison_with_a_non_tag`,
    at: ["nodes", "ask", "until"],
    prefix: "loop node 'ask': `until:` ",
    bases: { r: "Step" },
  },
  {
    id: `${NULL_NARROWING}a_node_assert_check_made_impossible_is_reported`,
    at: ["nodes", "c", "asserts", 0],
    prefix: "  node 'c' assert: ",
    bases: { x: "Optional(AB)" },
  },
  {
    id: `${NULL_NARROWING}a_null_check_after_isinstance_is_impossible`,
    at: ["nodes", "route", "cases", 0, "when"],
    prefix: "case node 'route': `cases[0].when`: ",
    bases: { "input.x": "Optional(AB)" },
  },
  {
    id: `${NULL_NARROWING}a_null_check_the_routing_rules_out_is_reported`,
    at: ["nodes", "inner", "cases", 0, "when"],
    prefix: "case node 'inner': `cases[0].when`: ",
    bases: { "input.x": "Optional(AB)" },
    routing: "not isinstance(input.x, A) and not isinstance(input.x, B)",
  },
  {
    id: `${NULL_NARROWING}a_tag_check_after_a_null_check_is_impossible`,
    at: ["nodes", "route", "cases", 0, "when"],
    prefix: "case node 'route': `cases[0].when`: ",
    bases: { "input.x": "Optional(AB)" },
  },
  {
    id: "load-types/compose_null_narrowing_case__a_node_assert_null_check_narrows_its_read",
    at: ["nodes", "x", "asserts", 0],
    prefix: "  node 'x' assert: ",
    bases: { output: "Optional(Item)" },
    read: "output.nosuch",
  },
  {
    id: `${UNION_NARROWING}a_misspelled_read_in_a_when_names_the_narrowed_type`,
    at: ["nodes", "route", "cases", 0, "when"],
    prefix: "  node 'route' cases[0].when from: ",
    bases: { "d.output": "Decision" },
    read: "d.output.limt",
  },
  {
    id: `${UNION_NARROWING}a_read_in_a_when_of_a_field_the_narrowed_members_lack`,
    at: ["nodes", "route", "cases", 0, "when"],
    prefix: "  node 'route' cases[0].when from: ",
    bases: { "d.output": "Decision" },
    read: "d.output.limit",
  },
  {
    id: `${UNION_NARROWING}a_when_comparing_the_tag_with_a_non_tag_suggests_one`,
    at: ["nodes", "route", "cases", 1, "when"],
    prefix: "case node 'route': `cases[1].when`: ",
    bases: { "d.output": "Decision" },
  },
  {
    id: `${UNION_NARROWING}a_when_read_narrowed_only_by_the_routing`,
    at: ["nodes", "inner", "cases", 0, "when"],
    prefix: "  node 'inner' cases[0].when from: ",
    bases: { "d.output": "Decision", "input.o": "Reply" },
    routing: 'd.output.action == "buy"',
    read: "d.output.nosuch",
  },
  {
    id: `${UNION_NARROWING}an_isinstance_that_can_never_be_true`,
    at: ["nodes", "route", "cases", 1, "when"],
    prefix: "case node 'route': `cases[1].when`: ",
    bases: { "d.output": "Decision" },
  },
])("narrowing message: $id", runNarrowingRow);
