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
 * `::test_a_check_a_null_check_before_it_makes_dead_is_reported`,
 * `::test_a_null_check_the_routing_proves_null_is_reported`,
 * `::test_a_when_null_check_impossible_with_the_checks_around_it_is_reported`,
 * `::test_a_null_check_case_under_a_branch_that_rules_out_every_member_is_reported`,
 * `::test_a_non_null_field_is_present_in_every_record_around_it`,
 * `::test_a_nullable_field_leaves_the_records_around_it_alone`,
 * `::test_a_field_narrowed_to_null_leaves_the_records_around_it_alone`,
 * `::test_a_null_check_on_a_field_or_its_parent_makes_a_sibling_tag_exhaustive` and
 * `::test_a_field_of_an_unchecked_optional_parent_may_still_be_null`. Through `assembleText`, the
 * rows the CALL type check decides, the `need` def built by hand (`handChild`, since def bodies
 * load in T3-22): `::test_a_narrowed_optional_field_fills_a_required_input` (4),
 * `::test_a_narrowed_field_makes_its_parent_non_null` (2) and
 * `::test_an_unnarrowed_optional_field_still_cannot_fill_a_required_input`, pinning the whole
 * message where Python matches a fragment.
 *
 * Changed: facts and environments compare with `toStrictEqual`, which also compares classes;
 * `Members` takes an array. `declaredImpossible` and `neverValue` are exported for these tests
 * (Python's `_declared_impossible` and `_never_value`), and `narrowType` from `narrowing.ts`
 * (Python's `_narrow_type`); its types compare with `typeEquals` and its required sets with
 * `toStrictEqual`. The flows load with `assembleText` where Python calls `load_flow`. The
 * pinned `compose_null_narrowing__*` cases are at the end: `…__a_null_when_is_no_on_value_even_on_a_narrowed_field`,
 * whose error the CASE build raises, for `test_a_null_when_is_no_on_value_even_on_a_narrowed_field`;
 * `…__an_unnarrowed_optional_field_into_a_required_input`, whose error the cross-flow binding
 * check raises (`assembledWith`, the `need` def built by hand); the three `checkImpossibleWhens`
 * raises; and `…__a_field_of_an_unchecked_optional_parent_may_be_null`, whose error the
 * exhaustiveness check raises; and `…__a_node_assert_check_made_impossible_is_reported`, whose
 * error the node-assert check raises, for
 * `test_a_node_assert_check_a_null_check_makes_impossible_is_reported` (ported above it, pinning
 * the whole message).
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
import { narrowType } from "../../src/compose/validate/narrowing.ts";
import { parseCondition } from "../../src/expr/index.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

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

// ---------- typed CALL arguments ----------

/**
 * A flow whose searched CASE passes `source` (default `${input.a.b}`) to a def `need` taking
 * `!<sink>` (default `str`) when `when`; the input `a` is `aType` (default `A`).
 */
const nestedFlow = (when: string, aType = "A", source = "${input.a.b}", sink = "str") =>
  "id: p\nname: p\n" +
  "typedefs:\n  A: {b: Optional(str)}\n" +
  `input:\n  a: !${aType}\n` +
  "nodes:\n" +
  "  route:\n    kind: case\n    cases:\n" +
  `      - when: "${when}"\n` +
  `        then: call(need, v=${source})\n` +
  "    else: e\n" +
  "  e:\n    kind: code\n    code: return 1\n" +
  "output: ${route.output}\n" +
  "defs:\n  need:\n" +
  `    input:\n      v: !${sink}\n` +
  "    nodes:\n      p:\n        kind: code\n" +
  "        input:\n          o: ${input.v}\n        code: return 1\n" +
  "    output: ${p.output}\n";

/** The `need` def built by hand: input `v: !<sink>` (default `str`), an untyped `p`, returned. */
const need = (sink = "str") =>
  handChild({
    inputs: parametersFromYaml(`input:\n  v: !${sink}\n`, readTypedefs({ A: { b: "Optional(str)" } })),
    producers: { p: null },
    output: "${p.output}",
  });

const NEED = need();

describe("a narrowed optional field fills a required input", () => {
  for (const when of ["isinstance(input.a.b, str)", "input.a.b != null"]) {
    for (const aType of ["A", "Optional(A)"]) {
      test(`${when} on ${aType}`, () => {
        assembleText(nestedFlow(when, aType), "flow.yaml", { children: { need: NEED } });
      });
    }
  }
});

test.each(["isinstance(input.a.b, str)", "input.a.b != null"])(
  "a narrowed field makes its parent non-null: %s",
  (when) => {
    assembleText(nestedFlow(when, "Optional(A)", "${input.a}", "A"), "flow.yaml", { children: { need: need("A") } });
  },
);

test("an unnarrowed optional field still cannot fill a required input", () => {
  const error = loadError(() =>
    assembleText(nestedFlow("input.a.b == null"), "flow.yaml", { children: { need: NEED } }),
  );

  expect(error.message).toBe(
    "node 'route' cases[0].then call(need) argument 'v' — child expects 'str', source is 'Optional(str)'",
  );
});

// --------------------------------------------------------------------------- //
// Nested optional fields
// --------------------------------------------------------------------------- //

test("a non-null field is present in every record around it", () => {
  const narrowed = narrowType(t("Optional(Outer)"), ["h", "b"], STR)!;
  expect(narrowed.nullable).toBe(false);
  expect(narrowed.required!.has("h")).toBe(true);
  const holder = narrowed.fields!.get("h")!;
  expect(holder.nullable).toBe(false);
  expect(holder.required!.has("b")).toBe(true);
  expect(typeEquals(holder.fields!.get("b")!, STR)).toBe(true);
});

test("a nullable field leaves the records around it alone", () => {
  const outer = t("Optional(Outer)");
  const narrowed = narrowType(outer, ["h", "b"], t("Optional(str)"))!;
  expect(narrowed.nullable).toBe(true);
  expect(narrowed.required).toStrictEqual(outer.required);
  expect(narrowed.fields!.get("h")!.required).toStrictEqual(outer.fields!.get("h")!.required);
  expect(narrowed.fields!.get("h")!.nullable).toBe(true);
});

test("a field narrowed to null leaves the records around it alone", () => {
  const holder = t("Optional(Holder)");
  const narrowed = narrowType(holder, ["b"], NULL)!;
  expect(narrowed.nullable).toBe(true);
  expect(narrowed.required).toStrictEqual(holder.required);
});

// --------------------------------------------------------------------------- //
// Impossible null checks in a flow
// --------------------------------------------------------------------------- //

const AB_FLOW_HEAD =
  "id: p\nname: p\n" +
  "typedefs:\n" +
  "  A:\n    kind: Literal(a)\n    n: int\n" +
  "  B:\n    kind: Literal(b)\n    s: str\n" +
  "  AB: Union(A, B)\n" +
  "input:\n  x: !Optional(AB)\n" +
  "nodes:\n";

/** A CODE node `nodeId` returning `1`, with `extra` lines added to its body. */
function codeNode(nodeId: string, extra = ""): string {
  return `  ${nodeId}:\n    kind: code\n    code: return 1\n${extra}`;
}

/** A searched CASE `nodeId` routing to `then` under `when`, else to `else_`. */
function searched(nodeId: string, when: string, then: string, else_: string): string {
  return (
    `  ${nodeId}:\n    kind: case\n    cases:\n` +
    `      - when: "${when}"\n        then: ${then}\n` +
    `    else: ${else_}\n`
  );
}

/** The message of the `LoadError` assembling `text` throws. */
const assembleError = (text: string): string => loadError(() => assembleText(text)).message;

describe("a when null check impossible with the checks around it is reported", () => {
  test.each<[string, string, string]>([
    [
      "tags-after-null-check",
      "input.x != null and input.x.kind != 'a' and input.x.kind != 'b'",
      '`input.x.kind != "b"` can never be true here: input.x is B, by the checks ' +
        'before it, so `input.x.kind == "b"` is always true',
    ],
    [
      "null-check-after-isinstance",
      "isinstance(input.x, A) or isinstance(input.x, B) or input.x != null",
      "`input.x != null` can never be true here: input.x is None, by the checks before it",
    ],
    [
      "null-check-after-tags",
      "input.x.kind == 'a' or input.x.kind == 'b' or input.x != null",
      "`input.x != null` can never be true here: input.x is None, by the checks before it",
    ],
  ])("%s", (_id, when, message) => {
    const text =
      AB_FLOW_HEAD + searched("route", when, "t", "e") + codeNode("t") + codeNode("e") + "output: ${input.x}\n";
    expect(assembleError(text)).toContain(`case node 'route': \`cases[0].when\`: ${message}`);
  });
});

test("a null check case under a branch that rules out every member is reported", () => {
  const text =
    AB_FLOW_HEAD +
    searched("outer", "not isinstance(input.x, A) and not isinstance(input.x, B)", "inner", "e") +
    searched("inner", "input.x != null", "t", "e2") +
    codeNode("t") +
    codeNode("e") +
    codeNode("e2") +
    "output: ${input.x}\n";
  expect(assembleError(text)).toContain(
    "case node 'inner': `cases[0].when`: `input.x != null` can never be true here: " +
      "input.x is None, by the case branch that routes this node",
  );
});

test("a node assert check a null check makes impossible is reported", () => {
  const assertion = "x != null and not isinstance(x, A) and not isinstance(x, B)";
  const body = `    input:\n      x: \${input.x}\n    asserts:\n      - "${assertion}"\n`;
  const text = AB_FLOW_HEAD + codeNode("c", body) + "output: ${c.output}\n";
  expect(assembleError(text)).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'c' assert: `not isinstance(x, B)` can never be true here: x is B, by the checks " +
      "before it, so `isinstance(x, B)` is always true",
  );
});

/**
 * A flow whose read CASE routes, under `when`, to an `on:` CASE over the tag `input.a.m` that
 * lists every tag and has no `else:`; `input.a` is `Optional(A)`.
 */
function exhaustiveOnFlow(when: string): string {
  return (
    "id: p\nname: p\n" +
    "typedefs:\n  A:\n    b: Optional(str)\n    m: Literal(p, q)\n" +
    "input:\n  a: !Optional(A)\n" +
    "nodes:\n" +
    "  route:\n    kind: case\n    cases:\n" +
    `      - when: "${when}"\n        then: inner\n` +
    "    else: e\n" +
    "  inner:\n    kind: case\n    on: ${input.a.m}\n    cases:\n" +
    "      - when: p\n        then: np\n" +
    "      - when: q\n        then: nq\n" +
    "  np:\n    kind: code\n    code: return 1\n" +
    "  nq:\n    kind: code\n    code: return 2\n" +
    "  e:\n    kind: code\n    code: return 3\n" +
    "output: ${route.output}\n"
  );
}

test.each(["input.a.b != null", "input.a != null"])(
  "a null check on a field or its parent makes a sibling tag exhaustive: %s",
  (when) => {
    // A non-null `input.a.b` means `input.a` is not null, so its required tag `m` is not.
    assembleText(exhaustiveOnFlow(when));
  },
);

test("a field of an unchecked optional parent may still be null", () => {
  expect(assembleError(exhaustiveOnFlow("input.a == null"))).toContain("${input.a.m} may be null");
});

// ---------- harvested conformance cases ----------

test("assembled: load-types/compose_null_narrowing__a_null_when_is_no_on_value_even_on_a_narrowed_field", () => {
  // YAML `null` is no `when:` value at all; the build rejects it before any typing.
  runCase({
    id: "load-types/compose_null_narrowing__a_null_when_is_no_on_value_even_on_a_narrowed_field",
    call: assembled,
  });
});

test("assembledWith: load-types/compose_null_narrowing__an_unnarrowed_optional_field_into_a_required_input", () => {
  runCase({
    id: "load-types/compose_null_narrowing__an_unnarrowed_optional_field_into_a_required_input",
    call: assembledWith({ need: NEED }),
  });
});

test.each<CaseRow>(
  [
    "a_null_check_after_isinstance_is_impossible",
    "a_null_check_the_routing_rules_out_is_reported",
    "a_tag_check_after_a_null_check_is_impossible",
    "a_field_of_an_unchecked_optional_parent_may_be_null",
    "a_node_assert_check_made_impossible_is_reported",
  ].map((name) => ({ id: `load-types/compose_null_narrowing__${name}`, call: assembled })),
)("assemble: $id", runCase);
