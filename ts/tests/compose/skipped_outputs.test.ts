/**
 * Output contracts for bindings that default a maybe-skipped reference to null: a port of the
 * direct tests of `tests/compose/test_skipped_outputs.py`
 * (`test_null_default_over_a_reference_is_optional_and_keeps_origins`,
 * `test_null_default_over_a_dotted_field_is_optional`,
 * `test_null_default_over_a_coalesce_of_equal_types_is_optional`,
 * `test_null_default_over_a_coalesce_of_different_types_is_untyped`,
 * `test_null_default_over_a_nullable_head_stays_optional`,
 * `test_null_default_over_a_partial_or_unknown_producer_is_untyped`,
 * `test_other_defaults_stay_untyped` and `test_null_default_over_a_flow_input_is_optional`).
 *
 * Changed: producer and flow-input tables are `Map`s.
 *
 * Later (T3-22): `test_on_over_a_null_default_is_checked_for_exhaustiveness`, which loads a flow.
 */

import { expect, test } from "bun:test";

import { resolveOutputBindingContract } from "../../src/compose/helpers/output_contracts.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import {
  ContractOrigins,
  isExactly,
  type OutputContract,
  PartialRecordContract,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";

const STR = Type.scalar(ValueKind.STRING);
const INT = Type.scalar(ValueKind.INTEGER);
const OPTIONAL_STR = STR.replace({ nullable: true });

/** Return a closed record type whose fields are all required. */
function record(fields: Record<string, Type>): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
  });
}

/** Resolve `source` against `producers` and no flow input. */
function resolve(source: string, producers: Record<string, OutputContract>): OutputContract | null {
  return resolveOutputBindingContract(source, new Map(Object.entries(producers)), new Map());
}

/** Return the type of `contract`, requiring it to be an exact `ValueContract`. */
function exactType(contract: OutputContract | null): Type {
  if (!isExactly(contract, ValueContract)) throw new Error("expected an exact ValueContract");
  return contract.type;
}

test("a null default over a reference is optional and keeps its origins", () => {
  const origins = new ContractOrigins({ default: new OriginRef("producer-output") });

  const resolved = resolve("${p.output:-null}", { p: new ValueContract(STR, origins) });

  expect(isExactly(resolved, ValueContract)).toBe(true);
  expect(typeEquals(exactType(resolved), OPTIONAL_STR)).toBe(true);
  expect((resolved as ValueContract).origins).toBe(origins);
});

test("a null default over a dotted field is optional", () => {
  const resolved = resolve("${p.output.f :- null}", { p: new ValueContract(record({ f: INT })) });

  expect(typeEquals(exactType(resolved), INT.replace({ nullable: true }))).toBe(true);
});

test("a null default over a coalesce of equal types is optional", () => {
  const resolved = resolve("${(a.output | b.output) :- null}", {
    a: new ValueContract(STR),
    b: new ValueContract(STR),
  });

  expect(resolved).toEqual(new ValueContract(OPTIONAL_STR));
});

test("a null default over a coalesce of different types is untyped", () => {
  const producers = { a: new ValueContract(STR), b: new ValueContract(INT) };

  expect(resolve("${(a.output | b.output) :- null}", producers)).toBe(null);
});

test("a null default over a nullable head stays optional", () => {
  const resolved = resolve("${p.output :- null}", { p: new ValueContract(OPTIONAL_STR) });

  expect(typeEquals(exactType(resolved), OPTIONAL_STR)).toBe(true);
});

test("a null default over a partial or unknown producer is untyped", () => {
  const partial = new PartialRecordContract([
    ["n", new ValueContract(INT)],
    ["m", null],
  ]);

  expect(resolve("${p.output :- null}", { p: partial })).toBe(null);
  expect(resolve("${q.output :- null}", {})).toBe(null);
});

test("other defaults stay untyped", () => {
  const producers = { p: new ValueContract(STR), a: new ValueContract(STR), b: new ValueContract(STR) };

  expect(resolve('${p.output :- "x"}', producers)).toBe(null);
  expect(resolve("${(a.output :- b.output) :- null}", producers)).toBe(null);
});

test("a null default over a flow input is optional", () => {
  const inputs = new Map([["x", new ValueContract(STR)]]);

  const resolved = resolveOutputBindingContract("${input.x :- null}", new Map(), inputs);

  expect(typeEquals(exactType(resolved), OPTIONAL_STR)).toBe(true);
});
