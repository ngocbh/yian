/**
 * Null checks narrow: `x != null` proves `x` is not null, as `isinstance` proves its type. A port
 * of `tests/compose/test_null_narrowing.py::test_check_in_recognizes_a_null_check`,
 * `::test_a_null_check_proves_like_any_check` and
 * `::test_check_text_spells_a_null_check_by_value`.
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
import { parseCondition } from "../../src/expr/index.ts";

const X = new NotNullCheck("x");

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
