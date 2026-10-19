/**
 * Null checks narrow: `x != null` proves `x` is not null, as `isinstance` proves its type. A port
 * of `tests/compose/test_null_narrowing.py::test_check_in_recognizes_a_null_check`,
 * `::test_a_null_check_proves_like_any_check`, `::test_check_text_spells_a_null_check_by_value`,
 * `::test_a_true_null_check_proves_the_base_is_not_null`, `::test_a_null_check_proves_nothing_else`,
 * `::test_a_null_check_meets_an_isinstance_on_the_same_base`,
 * `::test_a_null_check_keeps_an_isinstance_narrowing`,
 * `::test_a_null_check_keeps_a_nullable_isinstance_narrowing`,
 * `::test_two_exact_facts_meet_to_the_narrower`,
 * `::test_a_non_null_exact_fact_meets_a_nullable_one_with_null_removed`,
 * `::test_two_nullable_exact_facts_meet_to_the_narrower_still_nullable`,
 * `::test_two_unrelated_exact_facts_give_no_fact`,
 * `::test_two_isinstance_checks_on_one_base_meet_to_the_narrower` and
 * `::test_a_null_check_names_no_type`.
 *
 * Changed: facts and environments compare with `toStrictEqual`, which also compares classes;
 * `Members` takes an array.
 *
 * Later (T3-20): `test_a_non_null_field_is_present_in_every_record_around_it`,
 * `test_a_nullable_field_leaves_the_records_around_it_alone` and
 * `test_a_field_narrowed_to_null_leaves_the_records_around_it_alone`.
 *
 * Later (T3-22): `test_a_when_null_check_impossible_with_the_checks_around_it_is_reported`,
 * `test_a_node_assert_check_a_null_check_makes_impossible_is_reported`,
 * `test_a_null_check_case_under_a_branch_that_rules_out_every_member_is_reported`,
 * `test_a_narrowed_optional_field_fills_a_required_input`,
 * `test_a_narrowed_field_makes_its_parent_non_null`,
 * `test_an_unnarrowed_optional_field_still_cannot_fill_a_required_input`,
 * `test_a_null_when_is_no_on_value_even_on_a_narrowed_field`,
 * `test_a_null_check_on_a_field_or_its_parent_makes_a_sibling_tag_exhaustive` and
 * `test_a_field_of_an_unchecked_optional_parent_may_still_be_null`.
 */

import { describe, expect, test } from "bun:test";

import {
  checkIn,
  checkText,
  guardedRefs,
  Holds,
  NOTHING,
  NotNullCheck,
  proves,
} from "../../src/compose/helpers/condition_guards.ts";
import {
  checkFact,
  type Env,
  Exact,
  type Fact,
  guardEnv,
  intersectFacts,
  Members,
  unresolvedType,
} from "../../src/compose/helpers/narrowing_facts.ts";
import { readType } from "../../src/compose/helpers/types.ts";
import { parseCondition } from "../../src/expr/index.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

const REGISTRY = readTypedefs({
  A: { kind: "Literal(a)", n: "int" },
  B: { kind: "Literal(b)", s: "str" },
  AB: "Union(A, B)",
  Item: { name: "str", qty: "int" },
  Wide: { name: "str" },
  Other: { flag: "bool" },
  MaybeA: "Optional(A)",
  MaybeItem: "Optional(Item)",
  Holder: { b: "Optional(str)" },
  Outer: { h: "Optional(Holder)" },
});

/** Resolve one type expression against the test typedefs. */
function t(name: string): Type {
  return readType(name, REGISTRY);
}

const [A, B, ITEM, WIDE, OTHER] = [t("A"), t("B"), t("Item"), t("Wide"), t("Other")];
const STR = t("str");
const ANY = t("Any");
const NULL = Type.scalar(ValueKind.NONE);
const X = new NotNullCheck("x");

/** The facts `condition` proves when true, with the bases typed by `declared`; other bases are unproven. */
function env(condition: string, declared: Record<string, Type> = {}): Env {
  const byBase = new Map(Object.entries(declared));
  return guardEnv(proves(parseCondition(condition), true), (base) => byBase.get(base) ?? null, REGISTRY);
}

// --------------------------------------------------------------------------- //
// Recognition
// --------------------------------------------------------------------------- //

describe("checkIn recognizes a null check", () => {
  const rows: [string, [NotNullCheck, boolean] | null][] = [
    ["x != null", [X, true]],
    ["null != x", [X, true]],
    ["x != none", [X, true]],
    ["input.a.b != null", [new NotNullCheck("input.a.b"), true]],
    ["${x != null}", [X, true]],
    ["x == null", [X, false]],
    ["null == x", [X, false]],
    ["x == none", [X, false]],
    ["x != 1", null],
    ["x.y != z", null],
    ["len(x) != null", null],
    ["x != None", null],
    ["null == null", null],
    ["x > null", null],
  ];
  for (const [text, expected] of rows) {
    test(text, () => {
      expect(checkIn(parseCondition(text))).toStrictEqual(expected);
    });
  }
});

test("a null check proves like any check", () => {
  expect(proves(parseCondition("not (x == null)"), true)).toStrictEqual(new Holds(X, true));
  expect(proves(parseCondition("x != null or y > 1"), true)).toStrictEqual(NOTHING);
  const reads = new Map(guardedRefs(parseCondition("x == null or x.f > 0")));
  expect(reads.get("x.f")).toStrictEqual(new Holds(X, true));
});

test("checkText spells a null check by value", () => {
  expect(checkText(X)).toBe("x != null");
  expect(checkText(X, false)).toBe("x == null");
  for (const text of ["x != null", "x == null"]) {
    const [check, value] = checkIn(parseCondition(text))!;
    expect(checkText(check, value)).toBe(text);
  }
});

// --------------------------------------------------------------------------- //
// Facts
// --------------------------------------------------------------------------- //

describe("a true null check proves the base is not null", () => {
  const rows: [string, Fact | null][] = [
    ["Optional(str)", new Exact(STR)],
    ["Optional(AB)", new Members([A, B], false)],
    ["Optional(Any)", new Exact(ANY)],
    ["Optional(A)", new Exact(A)],
    ["str", null],
    ["AB", null],
  ];
  for (const [base, expected] of rows) {
    test(base, () => {
      expect(checkFact(X, t(base), true, REGISTRY)).toStrictEqual(expected);
    });
  }
});

test("a null check proves nothing else", () => {
  expect(checkFact(X, NULL, true, REGISTRY)).toBeNull();
  expect(checkFact(X, null, true, REGISTRY)).toBeNull();
  expect(checkFact(X, t("Optional(str)"), false, REGISTRY)).toBeNull();
  expect(checkFact(X, t("Optional(AB)"), false, REGISTRY)).toBeNull();
});

test("a null check meets an isinstance on the same base", () => {
  const maybeAB = t("Optional(AB)");
  expect(env("x != null and isinstance(x, A)", { x: maybeAB })).toStrictEqual(new Map([["x", new Members([A])]]));
  expect(env("isinstance(x, MaybeA) and x != null", { x: maybeAB })).toStrictEqual(
    new Map([["x", new Members([A])]]),
  );
  expect(env("x != null", { x: maybeAB })).toStrictEqual(new Map([["x", new Members([A, B])]]));
  expect(env("x == null", { x: maybeAB })).toStrictEqual(new Map());
});

describe("a null check keeps an isinstance narrowing", () => {
  for (const condition of ["x != null and isinstance(x, Item)", "isinstance(x, Item) and x != null"]) {
    for (const base of ["Optional(Any)", "Optional(Wide)"]) {
      test(`${condition} on ${base}`, () => {
        expect(env(condition, { x: t(base) })).toStrictEqual(new Map([["x", new Exact(ITEM)]]));
      });
    }
  }
});

describe("a null check keeps a nullable isinstance narrowing", () => {
  for (const condition of ["x != null and isinstance(x, MaybeItem)", "isinstance(x, MaybeItem) and x != null"]) {
    for (const base of ["Optional(Any)", "Optional(Wide)"]) {
      test(`${condition} on ${base}`, () => {
        expect(env(condition, { x: t(base) })).toStrictEqual(new Map([["x", new Exact(ITEM)]]));
      });
    }
  }
});

// --------------------------------------------------------------------------- //
// Meeting two exact facts
// --------------------------------------------------------------------------- //

test("two exact facts meet to the narrower", () => {
  expect(intersectFacts(new Exact(ANY), new Exact(ITEM))).toStrictEqual(new Exact(ITEM));
  expect(intersectFacts(new Exact(ITEM), new Exact(ANY))).toStrictEqual(new Exact(ITEM));
  expect(intersectFacts(new Exact(WIDE), new Exact(ITEM))).toStrictEqual(new Exact(ITEM));
  expect(intersectFacts(new Exact(ITEM), new Exact(WIDE))).toStrictEqual(new Exact(ITEM));
  expect(intersectFacts(new Exact(ITEM), new Exact(ITEM))).toStrictEqual(new Exact(ITEM));
});

describe("a non-null exact fact meets a nullable one with null removed", () => {
  for (const [name, other] of [
    ["Any", ANY],
    ["Wide", WIDE],
  ] as const) {
    test(name, () => {
      const maybeItem = t("Optional(Item)");
      expect(intersectFacts(new Exact(other), new Exact(maybeItem))).toStrictEqual(new Exact(ITEM));
      expect(intersectFacts(new Exact(maybeItem), new Exact(other))).toStrictEqual(new Exact(ITEM));
      expect(intersectFacts(new Exact(ITEM), new Exact(maybeItem))).toStrictEqual(new Exact(ITEM));
      expect(intersectFacts(new Exact(maybeItem), new Exact(ITEM))).toStrictEqual(new Exact(ITEM));
    });
  }
});

test("two nullable exact facts meet to the narrower, still nullable", () => {
  const [maybeItem, maybeWide] = [t("Optional(Item)"), t("Optional(Wide)")];
  expect(intersectFacts(new Exact(maybeWide), new Exact(maybeItem))).toStrictEqual(new Exact(maybeItem));
  expect(intersectFacts(new Exact(maybeItem), new Exact(maybeWide))).toStrictEqual(new Exact(maybeItem));
});

test("two unrelated exact facts give no fact", () => {
  expect(intersectFacts(new Exact(ITEM), new Exact(OTHER))).toBeNull();
  expect(intersectFacts(new Exact(OTHER), new Exact(ITEM))).toBeNull();
  expect(intersectFacts(new Exact(OTHER), new Exact(t("Optional(Item)")))).toBeNull();
});

test("two isinstance checks on one base meet to the narrower", () => {
  expect(env("isinstance(x, Wide) and isinstance(x, Item)", { x: ANY })).toStrictEqual(
    new Map([["x", new Exact(ITEM)]]),
  );
});

// --------------------------------------------------------------------------- //
// Impossible checks
// --------------------------------------------------------------------------- //

test("a null check names no type", () => {
  expect(unresolvedType(new Holds(X, true), REGISTRY)).toBe(false);
  expect(unresolvedType(new Holds(X, false), REGISTRY)).toBe(false);
});
