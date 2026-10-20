/**
 * Null checks in CASE conditions: what a read after a null check is checked under. A port of
 * `tests/compose/test_null_narrowing_case.py::test_a_read_after_an_is_null_check_in_an_or_is_guarded`
 * and `::test_a_case_param_read_after_a_null_check_sees_the_value_as_not_null` (`engine/case.test.ts`
 * pins its allocation order too).
 *
 * Changed: the guard and the facts compare with `toStrictEqual`, which also compares classes, and
 * types with `typeEquals`. The pinned `compose_null_narrowing_case__*` case sits in
 * `narrowing_core.test.ts`'s table.
 *
 * Later (T3-20, the asserts validators): `test_an_assert_read_after_a_null_check_is_guarded_by_it`
 * (`narrows` and root base types), `test_a_node_assert_read_after_a_null_check_names_the_non_null_type`
 * and `test_a_flow_assert_null_check_narrows_a_skippable_producer_of_a_non_null_type`.
 *
 * Later (T3-19, the call/map type check decides them): `test_a_branch_under_a_null_check_passes_the_value_as_not_null`,
 * `test_a_branch_under_a_null_check_on_an_input_passes_it_as_not_null`,
 * `test_a_negated_is_null_check_narrows_the_branch`,
 * `test_a_null_check_that_may_not_hold_does_not_narrow_the_branch`,
 * `test_the_else_of_a_null_check_is_not_narrowed`,
 * `test_a_node_two_null_checking_cases_route_is_narrowed`,
 * `test_a_node_that_only_one_of_two_cases_proves_not_null_is_not_narrowed`,
 * `test_an_on_case_over_an_optional_union_narrows_its_branch_as_before`,
 * `test_a_null_check_keeps_an_isinstance_narrowing_of_the_branch`,
 * `test_a_null_check_keeps_a_nullable_isinstance_narrowing_of_the_branch`,
 * `test_a_null_check_case_under_an_isinstance_branch_keeps_its_narrowing` and
 * `test_a_null_check_alone_does_not_prove_a_type`.
 */

import { expect, test } from "bun:test";

import { Holds, NOTHING, NotNullCheck } from "../../src/compose/helpers/condition_guards.ts";
import { type BaseTypeOf, checkedRefs, Exact, guardEnv } from "../../src/compose/helpers/narrowing_facts.ts";
import { readType } from "../../src/compose/helpers/types.ts";
import { BuildContext, NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { computeNarrowing } from "../../src/compose/validate/narrowing.ts";
import { parseCondition } from "../../src/expr/index.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { typeEquals } from "../../src/typesys/values.ts";

test("a read after an is-null check in an or is guarded", () => {
  const registry = readTypedefs({});
  const tree = parseCondition("x.output == null or x.output > 0");
  const declared = readType("Optional(int)", registry);
  const baseTypeOf: BaseTypeOf = (base) => (base === "x.output" ? declared : null);

  const refs = checkedRefs(tree, baseTypeOf, registry);
  const guard = new Holds(new NotNullCheck("x.output"), true);
  expect(refs).toStrictEqual([
    ["x.output", NOTHING],
    ["x.output", guard],
  ]);
  const env = guardEnv(guard, baseTypeOf, registry);
  expect(env).toStrictEqual(new Map([["x.output", new Exact(readType("int", registry))]]));
});

const PARAM_REGISTRY = readTypedefs({ Rec: { a: "str", b: "str" } });

test("a case param read after a null check sees the value as not null", () => {
  const rec = readType("Optional(Rec)", PARAM_REGISTRY);
  const gate = new CaseDescriptor({
    id: "gate",
    on: null,
    cases: [{ when: "x.output.a == 'p' and x.output != null and x.output.b == 'q'", then: "use" }],
    else_: null,
  });
  const descriptors = new Map<string, NodeDescriptor>([
    ["x", new NodeDescriptor({ id: "x" })],
    ["gate", gate],
    ["use", new NodeDescriptor({ id: "use" })],
  ]);
  const edges = gate.build(new BuildContext({ registry: new Map() })).edges;
  const narrowing = computeNarrowing(descriptors, edges, new Map([["x", new ValueContract(rec)]]), new Map(), {
    registry: PARAM_REGISTRY,
  });
  // The params follow the reads: `x.output.a`, `x.output`, `x.output.b`.
  const [before, after] = ["__r0", "__r2"];
  const seen = (param: string) => (narrowing.forParam("gate", param)[0].get("x") as ValueContract).type;
  expect(seen(after).nullable).toBe(false);
  expect(typeEquals(seen(before), rec)).toBe(true);
  const notNull = new Exact(readType("Rec", PARAM_REGISTRY));
  expect(narrowing.paramEnv("gate", after)).toStrictEqual(new Map([["x.output", notNull]]));
  expect(narrowing.paramEnv("gate", before)).toStrictEqual(new Map());
});
