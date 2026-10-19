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
 * `::test_two_isinstance_checks_on_one_base_meet_to_the_narrower`,
 * `::test_a_null_check_names_no_type`,
 * `::test_a_null_check_is_never_impossible_by_its_declared_type`,
 * `::test_a_reported_null_check_literal_is_never_true`,
 * `::test_a_null_check_on_any_value_is_not_reported`,
 * `::test_an_isinstance_check_is_still_reported_after_a_null_check`,
 * `::test_a_null_check_the_checks_before_it_prove_null_is_reported`,
 * `::test_a_check_a_null_check_before_it_makes_dead_is_reported` and
 * `::test_a_null_check_the_routing_proves_null_is_reported`.
 *
 * Changed: facts and environments compare with `toStrictEqual`, which also compares classes;
 * `Members` takes an array. `declaredImpossible` and `neverValue` are exported for these tests
 * (Python's `_declared_impossible` and `_never_value`). The pinned `compose_null_narrowing__*`
 * cases sit in `narrowing_core.test.ts`'s table, except
 * `compose_null_narrowing__a_null_when_is_no_on_value_even_on_a_narrowed_field`, whose error the
 * CASE build raises: it is pinned at the end (`assemble`), for
 * `test_a_null_when_is_no_on_value_even_on_a_narrowed_field`.
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
 * `test_a_null_check_on_a_field_or_its_parent_makes_a_sibling_tag_exhaustive` and
 * `test_a_field_of_an_unchecked_optional_parent_may_still_be_null`.
 */

import { describe, expect, test } from "bun:test";

import {
  checkIn,
  checkText,
  guardedRefs,
  Holds,
  literalsInContext,
  NOTHING,
  NotNullCheck,
  proves,
} from "../../src/compose/helpers/condition_guards.ts";
import {
  type BaseTypeOf,
  checkFact,
  declaredImpossible,
  type Env,
  Exact,
  type Fact,
  guardEnv,
  ImpossibleCheck,
  impossibleCheckMessage,
  impossibleChecks,
  ImpossibleReason,
  intersectFacts,
  Members,
  neverValue,
  unresolvedType,
} from "../../src/compose/helpers/narrowing_facts.ts";
import { readType } from "../../src/compose/helpers/types.ts";
import { parseCondition } from "../../src/expr/index.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { assembled, runCase } from "./declarations.ts";

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

/** A `BaseTypeOf` over `declared`; other bases are unproven. */
function types(declared: Record<string, Type> = {}): BaseTypeOf {
  const byBase = new Map(Object.entries(declared));
  return (base) => byBase.get(base) ?? null;
}

/** The facts `condition` proves when true, with the bases typed by `declared`. */
function env(condition: string, declared: Record<string, Type> = {}): Env {
  return guardEnv(proves(parseCondition(condition), true), types(declared), REGISTRY);
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

describe("a null check is never impossible by its declared type", () => {
  for (const text of ["x != null", "x == null", "not x != null"]) {
    for (const base of ["str", "Optional(str)", "AB", "Any", "None"]) {
      test(`${text} on ${base}`, () => {
        const literals = literalsInContext(parseCondition(text));
        expect(literals).toHaveLength(1);
        expect(declaredImpossible(literals[0]!, t(base), true, REGISTRY)).toBeNull();
      });
    }
  }
});

describe("a reported null check literal is never true", () => {
  const rows: [string, boolean][] = [
    ["x != null", true],
    ["x == null", true],
    ["not x != null", false],
  ];
  for (const [text, never] of rows) {
    test(text, () => {
      // The value is that of the comparison without the `not` around it.
      const reported = new ImpossibleCheck("x", "x", ImpossibleReason.CONTEXT);
      const literals = literalsInContext(parseCondition(text));
      expect(literals).toHaveLength(1);
      expect(neverValue(literals[0]!, reported)).toBe(never);
    });
  }
});

describe("a null check on any value is not reported", () => {
  for (const condition of [
    "x != null and x.name != ''",
    "x == null or x.name != ''",
    "x != null and x != null",
    "x == null and x == null",
  ]) {
    for (const base of ["Item", "Optional(Item)"]) {
      test(`${condition} on ${base}`, () => {
        expect(impossibleChecks(parseCondition(condition), types({ x: t(base) }), REGISTRY)).toEqual([]);
      });
    }
  }
});

test("an isinstance check is still reported after a null check", () => {
  const tree = parseCondition("x != null and isinstance(x, Other)");
  const found = impossibleChecks(tree, types({ x: t("Optional(Item)") }), REGISTRY);
  expect(found.map((check) => [check.text, check.reason])).toEqual([
    ["isinstance(x, Other)", ImpossibleReason.NO_OVERLAP],
  ]);
});

/** The impossible-check messages of `condition`, with `x` an `Optional(AB)`. */
function impossible(condition: string, routing: Env | null = null): string[] {
  const found = impossibleChecks(parseCondition(condition), types({ x: t("Optional(AB)") }), REGISTRY, { routing });
  return found.map(impossibleCheckMessage);
}

test("a null check the checks before it prove null is reported", () => {
  expect(impossible("not isinstance(x, A) and not isinstance(x, B) and x != null")).toEqual([
    "`x != null` can never be true here: x is None, by the checks before it",
  ]);
});

test("a check a null check before it makes dead is reported", () => {
  expect(impossible("x != null and not isinstance(x, A) and not isinstance(x, B)")).toEqual([
    "`not isinstance(x, B)` can never be true here: x is B, by the checks before it, " +
      "so `isinstance(x, B)` is always true",
  ]);
});

test("a null check the routing proves null is reported", () => {
  expect(impossible("x != null", new Map([["x", new Members([], true)]]))).toEqual([
    "`x != null` can never be true here: x is None, by the case branch that routes this node",
  ]);
  expect(impossible("x == null", new Map([["x", new Members([], true)]]))).toEqual([]);
});

// ---------- harvested conformance cases ----------

// The flow's searched `route` builds before the `on:` case `inner`.
test("assembled: load-types/compose_null_narrowing__a_null_when_is_no_on_value_even_on_a_narrowed_field", () => {
  // YAML `null` is no `when:` value at all; the build rejects it before any typing.
  runCase({
    id: "load-types/compose_null_narrowing__a_null_when_is_no_on_value_even_on_a_narrowed_field",
    call: assembled,
  });
});
