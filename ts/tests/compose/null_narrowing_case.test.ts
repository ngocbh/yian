/**
 * Null checks in CASE conditions: what a branch and a param read after a null check see. A port of
 * `tests/compose/test_null_narrowing_case.py::test_a_read_after_an_is_null_check_in_an_or_is_guarded`
 * and `::test_a_case_param_read_after_a_null_check_sees_the_value_as_not_null` (`engine/case.test.ts`
 * pins its allocation order too); through `assembleText`, the rows the CALL/MAP type check decides,
 * each error pinned whole where Python matches a fragment:
 * `::test_a_branch_under_a_null_check_passes_the_value_as_not_null` (2),
 * `::test_a_branch_under_a_null_check_on_an_input_passes_it_as_not_null` (4),
 * `::test_a_negated_is_null_check_narrows_the_branch`,
 * `::test_a_null_check_that_may_not_hold_does_not_narrow_the_branch`,
 * `::test_the_else_of_a_null_check_is_not_narrowed`,
 * `::test_a_node_two_null_checking_cases_route_is_narrowed`,
 * `::test_a_node_that_only_one_of_two_cases_proves_not_null_is_not_narrowed`,
 * `::test_an_on_case_over_an_optional_union_narrows_its_branch_as_before` (3),
 * `::test_a_null_check_keeps_an_isinstance_narrowing_of_the_branch` (4),
 * `::test_a_null_check_keeps_a_nullable_isinstance_narrowing_of_the_branch` (8),
 * `::test_a_null_check_case_under_an_isinstance_branch_keeps_its_narrowing` (2) and
 * `::test_a_null_check_alone_does_not_prove_a_type`.
 *
 * Changed: the guard and the facts compare with `toStrictEqual`, which also compares classes, and
 * types with `typeEquals`. The `take` and `take_each` defs are built by hand (`handChild`: the def's
 * parameters read from its `input:` text with the flow's typedefs, its node `d` with its output
 * contract), since def bodies load in T3-22; the CODE references are TS's
 * (`./fixtures/compose_codefns.ts:echo`). The pinned `compose_null_narrowing_case__*` case sits in
 * `narrowing_core.test.ts`'s table.
 *
 * Later (T3-20, the asserts validators): `test_an_assert_read_after_a_null_check_is_guarded_by_it`
 * (`narrows` and root base types), `test_a_node_assert_read_after_a_null_check_names_the_non_null_type`
 * and `test_a_flow_assert_null_check_narrows_a_skippable_producer_of_a_non_null_type`.
 */

import { describe, expect, test } from "bun:test";

import { Holds, NOTHING, NotNullCheck } from "../../src/compose/helpers/condition_guards.ts";
import { type BaseTypeOf, checkedRefs, Exact, guardEnv } from "../../src/compose/helpers/narrowing_facts.ts";
import { readType } from "../../src/compose/helpers/types.ts";
import { BuildContext, NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { computeNarrowing } from "../../src/compose/validate/narrowing.ts";
import { parseCondition } from "../../src/expr/index.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import { typeEquals } from "../../src/typesys/values.ts";
import {
  assembleText,
  caseRegistry,
  type ChildFlows,
  handChild,
  loadError,
  parametersFromYaml,
} from "./declarations.ts";

const ECHO = "./fixtures/compose_codefns.ts:echo";
const CHILD_EXPECTS = "call node 'use': binding 'v' — child expects 'str', source is 'Optional(str)'";

/** A CODE node `nodeId` that outputs `0`, for a branch that passes nothing on. */
function skip(nodeId: string): string {
  return `  ${nodeId}:\n    kind: code\n    code: ${ECHO}\n    input:\n      topic: 0\n    output: int\n`;
}

/** The `use` node: a CALL of `take`, or a MAP of `take_each`, passing `${x.output}`. */
function sink(target: "call" | "map"): string {
  if (target === "call") return "  use:\n    kind: call\n    call: take\n    input:\n      v: ${x.output}\n";
  return (
    "  use:\n    kind: map\n    call: take_each\n    over: ${input.items}\n" +
    "    input:\n      i: ${item}\n      v: ${x.output}\n"
  );
}

/**
 * A flow whose searched CASE `gate` routes on `x`, a node with an `Optional(str)` output: one
 * `(when, then)` per case, `else:` to `skip`; `target` `"call"` passes `${x.output}` to a def's
 * `str` input, `"map"` as a MAP's per-item argument.
 */
function producerFlow(cases: readonly (readonly [string, string])[], target: "call" | "map" = "call"): string {
  const whens = cases.map(([when, then]) => `      - when: "${when}"\n        then: ${then}\n`).join("");
  return (
    "id: p\nname: p\n" +
    "input:\n  s: !Optional(str)\n  items: !list(int)\n  y: !bool\n" +
    "nodes:\n" +
    `  x:\n    kind: code\n    code: ${ECHO}\n` +
    "    input:\n      topic: ${input.s}\n    output: Optional(str)\n" +
    "  gate:\n    kind: case\n    cases:\n" +
    whens +
    "    else: skip\n" +
    sink(target) +
    `  skip:\n    kind: code\n    code: ${ECHO}\n` +
    "    input:\n      topic: skipped\n    output: str\n" +
    "output: ${use.output :- skip.output}\n" +
    "defs:\n" +
    "  take:\n    input:\n      v: !str\n    nodes:\n" +
    `      d:\n        kind: code\n        code: ${ECHO}\n` +
    "        input:\n          topic: ${input.v}\n        output: str\n" +
    "    output: ${d.output}\n" +
    "  take_each:\n    input:\n      i: !int\n      v: !str\n    nodes:\n" +
    `      d:\n        kind: code\n        code: ${ECHO}\n` +
    "        input:\n          topic: ${input.v}\n        output: str\n" +
    "    output: ${d.output}\n"
  );
}

const TYPEDEFS =
  "typedefs:\n" +
  "  Item:\n    qty: int\n" +
  "  Wide:\n    qty: int\n    note: Optional(str)\n" +
  "  A:\n    b: Optional(str)\n" +
  "  MaybeItem: Optional(Item)\n";

/**
 * A flow whose searched CASE passes `source` to a def's `!<sinkType>` input under `when`. `Item` is
 * `{qty: int}`; `Wide` is `{qty: int, note: Optional(str)}`, which `Item` fits; `A` is
 * `{b: Optional(str)}`; `MaybeItem` is `Optional(Item)`.
 */
function inputFlow(when: string, xType: string, sinkType = "Item", source = "${input.x}"): string {
  return (
    `id: p\nname: p\n${TYPEDEFS}` +
    `input:\n  x: !${xType}\n` +
    "nodes:\n" +
    "  gate:\n    kind: case\n    cases:\n" +
    `      - when: "${when}"\n        then: use\n` +
    "    else: skip\n" +
    `  use:\n    kind: call\n    call: take\n    input:\n      v: ${source}\n` +
    skip("skip") +
    "output: ${use.output :- skip.output}\n" +
    "defs:\n" +
    `  take:\n    input:\n      v: !${sinkType}\n    nodes:\n` +
    `      d:\n        kind: code\n        code: ${ECHO}\n` +
    "        input:\n          topic: ${input.v}\n        output: Any\n" +
    "    output: ${d.output}\n"
  );
}

/** A flow where CASE `outer` routes to CASE `inner`, which routes `${input.x}` to `!Item`. */
function crossNodeFlow(outer: string, inner: string, xType: string): string {
  return (
    `id: p\nname: p\n${TYPEDEFS}` +
    `input:\n  x: !${xType}\n` +
    "nodes:\n" +
    "  outer:\n    kind: case\n    cases:\n" +
    `      - when: "${outer}"\n        then: inner\n` +
    "    else: skip\n" +
    "  inner:\n    kind: case\n    cases:\n" +
    `      - when: "${inner}"\n        then: use\n` +
    "    else: skip2\n" +
    "  use:\n    kind: call\n    call: take\n    input:\n      v: ${input.x}\n" +
    skip("skip") +
    skip("skip2") +
    "output: ${use.output :- skip.output}\n" +
    "defs:\n" +
    "  take:\n    input:\n      v: !Item\n    nodes:\n" +
    `      d:\n        kind: code\n        code: ${ECHO}\n` +
    "        input:\n          topic: ${input.v.qty}\n        output: int\n" +
    "    output: ${d.output}\n"
  );
}

const INPUT_REGISTRY = caseRegistry(SourceDocument.parse(TYPEDEFS));

/**
 * A `take` def built by hand: the def's `input:` lines read with `registry`, its node `d` typed
 * `output`, returned.
 */
const take = (input: string, output: string, registry: TypeRegistry = INPUT_REGISTRY) =>
  handChild({
    inputs: parametersFromYaml(`input:\n${input}`, registry),
    producers: { d: new ValueContract(readType(output, registry)) },
    output: "${d.output}",
  });

/** `producerFlow`'s defs built by hand. */
const PRODUCER_DEFS = { take: take("  v: !str\n", "str"), take_each: take("  i: !int\n  v: !str\n", "str") };

/** Assemble `text` with `children` as its defs. */
const load = (text: string, children: ChildFlows = PRODUCER_DEFS) => assembleText(text, "flow.yaml", { children });

/** The message of the `LoadError` assembling `text` with `children` throws. */
const loadMessage = (text: string, children: ChildFlows = PRODUCER_DEFS) =>
  loadError(() => load(text, children)).message;

// --------------------------------------------------------------------------- //
// A branch under a null check
// --------------------------------------------------------------------------- //

test.each(["call", "map"] as const)("a branch under a null check passes the value as not null: %s", (target) => {
  load(producerFlow([["x.output != null", "use"]], target));
});

test.each([
  ["input.x != null", "Optional(Item)", "Item", "${input.x}"],
  ["input.x.b != null", "A", "str", "${input.x.b}"],
  ["input.x.b != null", "Optional(A)", "str", "${input.x.b}"],
  ["input.x != null", "Optional(Any)", "Any", "${input.x}"],
])("a branch under a null check on an input passes it as not null: %s on %s", (when, xType, sinkType, source) => {
  load(inputFlow(when, xType, sinkType, source), { take: take(`  v: !${sinkType}\n`, "Any") });
});

test("a negated is-null check narrows the branch", () => {
  load(producerFlow([["not (x.output == null)", "use"]]));
});

test("a null check that may not hold does not narrow the branch", () => {
  expect(loadMessage(producerFlow([["x.output != null or input.y", "use"]]))).toBe(CHILD_EXPECTS);
});

test("the else of a null check is not narrowed", () => {
  const text = producerFlow([["x.output != null", "skip"]]).replace("else: skip", "else: use");
  expect(loadMessage(text)).toBe(CHILD_EXPECTS);
});

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

// --------------------------------------------------------------------------- //
// Joins and `on:` cases
// --------------------------------------------------------------------------- //

test("a node two null-checking cases route is narrowed", () => {
  load(
    producerFlow([
      ["x.output != null and input.y", "use"],
      ["x.output != null", "use"],
    ]),
  );
});

test("a node that only one of two cases proves not null is not narrowed", () => {
  const text = producerFlow([
    ["x.output != null", "use"],
    ["input.y", "use"],
  ]);
  expect(loadMessage(text)).toBe(CHILD_EXPECTS);
});

const ON_TYPEDEFS =
  "typedefs:\n" +
  "  A:\n    kind: Literal(a)\n    n: int\n" +
  "  B:\n    kind: Literal(b)\n    s: str\n" +
  "  AB: Union(A, B)\n";

/** A flow whose `on:` CASE over an `Optional(Union(A, B))` tag passes the value to `!<sinkType>`. */
function onFlow(sinkType: string): string {
  return (
    "id: p\nname: p\n" +
    ON_TYPEDEFS +
    "input:\n  v: !Optional(AB)\n" +
    "nodes:\n" +
    "  gate:\n    kind: case\n    on: ${input.v.kind}\n    cases:\n" +
    "      - when: a\n        then: use\n" +
    "    else: skip\n" +
    "  use:\n    kind: call\n    call: take\n    input:\n      v: ${input.v}\n" +
    skip("skip") +
    "output: ${use.output :- skip.output}\n" +
    "defs:\n" +
    `  take:\n    input:\n      v: !${sinkType}\n    nodes:\n` +
    `      d:\n        kind: code\n        code: ${ECHO}\n` +
    "        input:\n          topic: 1\n        output: int\n" +
    "    output: ${d.output}\n"
  );
}

test.each(["A", "Optional(A)", "AB"])(
  "an on case over an optional union narrows its branch as before: %s",
  (sinkType) => {
    const registry = caseRegistry(SourceDocument.parse(ON_TYPEDEFS));
    load(onFlow(sinkType), { take: take(`  v: !${sinkType}\n`, "int", registry) });
  },
);

// --------------------------------------------------------------------------- //
// Null checks meet isinstance checks
// --------------------------------------------------------------------------- //

const ITEM_TAKE = { take: take("  v: !Item\n", "Any") };

describe("a null check keeps an isinstance narrowing of the branch", () => {
  for (const when of [
    "input.x != null and isinstance(input.x, Item)",
    "isinstance(input.x, Item) and input.x != null",
  ]) {
    for (const xType of ["Optional(Any)", "Optional(Wide)"]) {
      test(`${when} on ${xType}`, () => {
        load(inputFlow(when, xType), ITEM_TAKE);
      });
    }
  }
});

describe("a null check keeps a nullable isinstance narrowing of the branch", () => {
  for (const when of [
    "input.x != null and isinstance(input.x, MaybeItem)",
    "isinstance(input.x, MaybeItem) and input.x != null",
  ]) {
    for (const xType of ["Optional(Any)", "Optional(Wide)"]) {
      for (const sinkType of ["Optional(Item)", "Item"]) {
        test(`${when} on ${xType} into ${sinkType}`, () => {
          load(inputFlow(when, xType, sinkType), { take: take(`  v: !${sinkType}\n`, "Any") });
        });
      }
    }
  }
});

const CROSS_TAKE = { take: take("  v: !Item\n", "int") };

test.each(["Optional(Any)", "Optional(Wide)"])(
  "a null check case under an isinstance branch keeps its narrowing: %s",
  (xType) => {
    load(crossNodeFlow("isinstance(input.x, Item)", "input.x != null", xType), CROSS_TAKE);
  },
);

test("a null check alone does not prove a type", () => {
  const text = crossNodeFlow("input.x != null", "input.x != null", "Optional(Any)");
  expect(loadMessage(text, CROSS_TAKE)).toBe(
    `call node 'use': binding 'v' — child expects '{"qty":int}', source is 'Any'; tag the slot ` +
      "(`v: !Item ${...}`) to check it at runtime, or narrow it with `isinstance` in a `case`",
  );
});

// --------------------------------------------------------------------------- //
// Param views
// --------------------------------------------------------------------------- //

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
