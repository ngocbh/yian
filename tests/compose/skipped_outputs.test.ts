/**
 * Output contracts for bindings that default a maybe-skipped reference to null: a port of the
 * direct tests of `tests/compose/test_skipped_outputs.py`
 * (`test_null_default_over_a_reference_is_optional_and_keeps_origins`,
 * `test_null_default_over_a_dotted_field_is_optional`,
 * `test_null_default_over_a_coalesce_of_equal_types_is_optional`,
 * `test_null_default_over_a_coalesce_of_different_types_is_untyped`,
 * `test_null_default_over_a_nullable_head_stays_optional`,
 * `test_null_default_over_a_partial_or_unknown_producer_is_untyped`,
 * `test_other_defaults_stay_untyped` and `test_null_default_over_a_flow_input_is_optional`), and
 * `test_on_over_a_null_default_is_checked_for_exhaustiveness`, whose harvested case is pinned at the
 * end.
 *
 * Changed: producer and flow-input tables are `Map`s. The flow assembles (`assembleText`), where
 * Python loads it.
 */

import { expect, test } from "bun:test";

import { Edge, FlowOutput } from "../../src/compile/model.ts";
import { resolveOutputBindingContract } from "../../src/compile/output_contracts.ts";
import { nullSkippedOutputs, SkipModel } from "../../src/compose/validate/skips.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import {
  bindingSource,
  ContractOrigins,
  isExactly,
  type OutputContract,
  PartialRecordContract,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "./declarations.ts";

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

/** Return `contract`, requiring it to be an exact `ValueContract`. */
function exact(contract: OutputContract | null): ValueContract {
  if (!isExactly(contract, ValueContract)) throw new Error("expected an exact ValueContract");
  return contract;
}

test("a null default over a reference is optional and keeps its origins", () => {
  const origins = new ContractOrigins({ default: new OriginRef("producer-output") });

  const resolved = resolve("${p.output:-null}", { p: new ValueContract(STR, origins) });

  const contract = exact(resolved);
  expect(typeEquals(contract.type, OPTIONAL_STR)).toBe(true);
  expect(contract.origins).toBe(origins);
});

test("a null default over a dotted field is optional", () => {
  const resolved = resolve("${p.output.f :- null}", { p: new ValueContract(record({ f: INT })) });

  expect(typeEquals(exact(resolved).type, INT.replace({ nullable: true }))).toBe(true);
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

  expect(typeEquals(exact(resolved).type, OPTIONAL_STR)).toBe(true);
});

test("a null default over a partial or unknown producer is untyped", () => {
  const partial = new PartialRecordContract([
    ["n", new ValueContract(INT)],
    ["m", null],
  ]);

  expect(resolve("${p.output :- null}", { p: partial })).toBe(null);
  expect(resolve("${q.output :- null}", {})).toBe(null);
});

test("a default of the head's type has that type; a null default over a default stays untyped", () => {
  const producers = { p: new ValueContract(STR), a: new ValueContract(STR), b: new ValueContract(STR) };

  expect(typeEquals(exact(resolve('${p.output :- "x"}', producers)).type, STR)).toBe(true);
  expect(resolve("${(a.output :- b.output) :- null}", producers)).toBe(null);
});

test("a null default over a flow input is optional", () => {
  const inputs = new Map([["x", new ValueContract(STR)]]);

  const resolved = resolveOutputBindingContract("${input.x :- null}", new Map(), inputs);

  expect(typeEquals(exact(resolved).type, OPTIONAL_STR)).toBe(true);
});

const NULL_DEFAULT_ON = `
id: f
name: f
nodes:
  cls:
    kind: code
    output: Literal(a, b, c)
    code: |
      return "a"
  route:
    kind: case
    on: \${cls.output :- null}
    cases:
      - when: a
        then: na
      - when: b
        then: nb
  na:
    kind: code
    output: str
    code: |
      return "A"
  nb:
    kind: code
    output: str
    code: |
      return "B"
output: \${na.output | nb.output}
`;

test("an on over a null default is checked for exhaustiveness", () => {
  // `on: ${cls.output :- null}` has the enum's type, so a missing tag is a load error.
  expect(loadError(() => assembleText(NULL_DEFAULT_ON)).message).toContain("non-exhaustive");
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-case/compose_skipped_outputs__on_over_a_null_default_is_checked_for_exhaustiveness", call: assembled },
])("assemble: $id", runCase);

// `c` isn't exhaustive, so it may take no branch and skip both `a` and `b`.
const MODEL = new SkipModel(
  ["c", "a", "b"],
  [new Edge("t", "c", "a", { sourceHandle: "then" }), new Edge("e", "c", "b", { sourceHandle: "else" })],
  new Set(),
);
const rewritten = (source: string) =>
  bindingSource(nullSkippedOutputs([new FlowOutput("v", source)], MODEL)[0]!.binding);

test("a co-skipping output whose producers may all skip reads null through ?.", () => {
  expect(rewritten("${a.output.qty}")).toBe("${a.output?.qty:-null}");
  expect(rewritten("${a.output?.qty}")).toBe("${a.output?.qty:-null}");
  expect(rewritten("${a.output}")).toBe("${a.output:-null}");
  expect(rewritten("${a.output.x | b.output.x}")).toBe("${(a.output?.x | b.output?.x) :- null}");
});
