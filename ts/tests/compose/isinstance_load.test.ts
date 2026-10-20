/**
 * The type names `isinstance(x, T)` and `T(...)` constructors name, and their resolution: a port
 * of the name-collecting test of `tests/compose/test_isinstance_load.py`.
 *
 * Ported: `test_collect_isinstance_names`; through `assembleText`,
 * `test_an_inline_body_cannot_take_a_whole_value_isinstance_argument` (pinning the whole message)
 * and its case
 * `load-types/compose_isinstance_load__an_inline_body_cannot_take_a_whole_value_isinstance`.
 *
 * TS-only: `collectConstructorNames`, `expressionTrees`, `resolveIsinstanceType`,
 * `resolveConstructorType`, `constructorTypeFn` and `isinstanceMembers` rows (Python's own
 * `isinstanceMembers` test is ported with the narrowing tests), and the pinned-case table, which
 * makes the resolution call a site makes on the conformance cases it decides and checks that the
 * case's message is the row's site prefix plus the resolver's message.
 *
 * Later (T3-22): the `load_flow` errors (`test_an_unknown_type_name_is_a_located_load_error`,
 * `test_a_flow_assert_error_is_located_by_its_own_assert`,
 * `test_list_is_not_an_isinstance_type_name`,
 * `test_a_parent_cannot_use_a_childs_typedef`,
 * `test_isinstance_in_wait_until_is_a_located_load_error`,
 * `test_isinstance_in_map_over_is_a_located_load_error`,
 * `test_isinstance_in_a_call_nodes_asserts_is_a_located_load_error`,
 * `test_isinstance_in_a_loop_nodes_asserts_is_a_load_error`,
 * `test_isinstance_in_an_agent_prompt_is_a_located_load_error`,
 * `test_isinstance_in_a_human_input_question_is_a_located_load_error`).
 *
 * Later (T4): the type tables and runs (`test_isinstance_in_a_node_input`,
 * `test_isinstance_in_a_searched_case_is_first_true_wins`, `test_isinstance_in_a_node_assert`,
 * `test_isinstance_in_flow_asserts_fills_start_and_end`,
 * `test_a_failing_flow_input_assert_fails_at_start`, `test_isinstance_in_a_flow_output`,
 * `test_isinstance_in_a_loop_while_runs_several_iterations`,
 * `test_a_run_that_used_isinstance_replays`, `test_isinstance_in_defs_and_nested_inline_children`,
 * `test_a_uses_child_resolves_names_against_its_own_typedefs`,
 * `test_isinstance_in_a_map_per_item_input`, `test_isinstance_in_a_call_directive_argument`,
 * `test_isinstance_in_a_loop_seed_and_until`, `test_isinstance_in_a_case_on`).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import {
  collectConstructorNames,
  collectIsinstanceNames,
  constructorTypeFn,
  expressionTrees,
  isinstanceMembers,
  resolveConstructorType,
  resolveIsinstanceType,
} from "../../src/compose/helpers/isinstance_types.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ExpressionError } from "../../src/expr/index.ts";
import { Binding, ValueTemplate } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { TypeCheckError, typeEquals } from "../../src/typesys/values.ts";
import type { Case } from "../conformance.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  caseDocument,
  caseRegistry,
  cursorAt,
  loadError,
  runCase,
} from "./declarations.ts";

// Pinned before `REG` is read too, so neither depends on the shell's `YIAN_*` limits.
install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

const REG = readTypedefs({
  Buy: { action: "Literal(buy)", qty: "int" },
  Hold: { action: "Literal(hold)" },
  Sell: { action: "Literal(sell)", price: "float" },
  Order: "Union(Buy, Sell)",
  Decision: "Union(Buy, Hold, Sell)",
});
const BUY = typeFor("Buy", REG);

/** Return the `TypeCheckError` `run` throws. */
function typeCheckError(run: () => unknown): TypeCheckError {
  try {
    run();
  } catch (error) {
    if (error instanceof TypeCheckError) return error;
    throw error;
  }
  throw new Error("expected a TypeCheckError");
}

describe("collecting names", () => {
  test.each<[unknown, boolean, string[]]>([
    ["${isinstance(d.output, Buy)}", false, ["Buy"]],
    ["a ${isinstance(x, Buy) and isinstance(y, int)} b ${isinstance(z, Buy)}", false, ["Buy", "int"]],
    ["isinstance(x, Buy)", false, []],
    ["isinstance(x, Buy) or not isinstance(x, Sell)", true, ["Buy", "Sell"]],
    ["${isinstance(x, Order)}", true, ["Order"]],
    ["plain text", false, []],
    [3n, false, []],
  ])("collectIsinstanceNames(%p, condition=%p)", (source, condition, names) => {
    expect(collectIsinstanceNames(source, { condition })).toEqual(names);
  });

  test("collectConstructorNames keeps first-seen order without duplicates", () => {
    expect(collectConstructorNames("${Buy(d.output).conviction}")).toEqual(["Buy"]);
    const condition = "Sell(x).price > 0 and Buy(x).qty > Sell(y).price";
    expect(collectConstructorNames(condition, { condition: true })).toEqual(["Sell", "Buy"]);
  });
});

describe("expressionTrees", () => {
  test("a condition is one expression; a template has one per span", () => {
    expect(expressionTrees("a or b", { condition: true })).toHaveLength(1);
    expect(expressionTrees("a or b")).toEqual([]);
    expect(expressionTrees("${a} and ${b}")).toHaveLength(2);
  });

  test("a Binding-wrapped source is unwrapped", () => {
    expect(expressionTrees(new Binding("${a}"))).toHaveLength(1);
    expect(expressionTrees(new Binding("a"), { condition: true })).toHaveLength(1);
  });

  test("a ValueTemplate's nested strings are scanned", () => {
    const template = new ValueTemplate({ x: "${a}", y: ["${b} ${c}", 1n] });
    expect(expressionTrees(template)).toHaveLength(3);
  });

  test("a non-string condition has no expressions", () => {
    expect(expressionTrees(true, { condition: true })).toEqual([]);
  });

  test("a parse error throws ExpressionError", () => {
    expect(() => expressionTrees("a or", { condition: true })).toThrow(ExpressionError);
    expect(() => expressionTrees("${a or}")).toThrow(ExpressionError);
  });
});

describe("resolution", () => {
  test("resolveIsinstanceType takes a builtin or a typedef", () => {
    expect(typeEquals(resolveIsinstanceType("int", REG), typeFor("int", REG))).toBe(true);
    expect(typeEquals(resolveIsinstanceType("Buy", REG), BUY)).toBe(true);
  });

  test("resolveIsinstanceType refuses an unknown name", () => {
    expect(typeCheckError(() => resolveIsinstanceType("list", REG)).message).toBe(
      "unknown type 'list'; isinstance takes a typedefs: name or one of str, int, float, bool, date, datetime, object",
    );
  });

  test("resolveConstructorType takes a typedef and refuses a builtin with the hint", () => {
    expect(typeEquals(resolveConstructorType("Buy", REG), BUY)).toBe(true);
    expect(typeCheckError(() => resolveConstructorType("int", REG)).message).toBe(
      "unknown expression function 'int'; types in expressions are typedef names; `int(...)` is not a conversion",
    );
  });

  test("constructorTypeFn maps a typedef to its type, once, and other names to null", () => {
    const typeFn = constructorTypeFn(REG);
    const first = typeFn("Buy");
    expect(first).not.toBeNull();
    expect(typeEquals(first!, BUY)).toBe(true);
    expect(typeFn("Buy")).toBe(first);
    expect(typeFn("int")).toBeNull();
  });

  test("isinstanceMembers narrows a union to the named member", () => {
    const members = isinstanceMembers(typeFor("Decision", REG), BUY);
    expect(members).not.toBeNull();
    expect(members!.length).toBe(1);
    expect(typeEquals(members![0]!, BUY)).toBe(true);
    expect(isinstanceMembers(BUY, BUY)).toBeNull();
  });
});

// ---------- harvested conformance cases ----------

/** One harvested case whose names one site resolves. */
interface NameRow {
  /** The case, `group/name`. */
  id: string;
  /** The path from the document root to the source. */
  at: (string | number)[];
  /** The site's prefix on the expected message. */
  prefix: string;
  /** The source is one condition expression. */
  condition?: boolean;
  /** The source is a record or list read as a value template. */
  template?: boolean;
  /** Which resolver the site calls. */
  kind: "isinstance" | "constructor";
}

/**
 * Collect the names at `row.at` as the site does and check that resolving them throws the
 * `TypeCheckError` whose message, after the row's prefix, is the case's.
 */
function runNameRow(row: NameRow): void {
  const { case_, document } = caseDocument(row.id);
  const registry = caseRegistry(document);
  const raw = cursorAt(document, row.at).value;
  const source = row.template ? new ValueTemplate(raw as Record<string, unknown>) : raw;
  const condition = row.condition ?? false;
  const isinstance = row.kind === "isinstance";
  const collect = isinstance ? collectIsinstanceNames : collectConstructorNames;
  const names = collect(source, { condition });
  expect(names.length).toBeGreaterThan(0);
  const resolve = isinstance ? resolveIsinstanceType : resolveConstructorType;
  const error = typeCheckError(() => names.map((name) => resolve(name, registry)));
  const expected = (case_["expect"] as Case)["load"] as Case;
  const caseMessage = (expected["diagnostics"] as Case[])[0]!["message"] as string;
  expect(caseMessage).toBe(row.prefix + error.message);
}

test.each<NameRow>([
  {
    id: "load-types/compose_isinstance_load__an_unknown_type_name_in_a_case_when",
    at: ["nodes", "route", "cases", 1, "when"],
    prefix: "case node 'route' cases[1].when: ",
    condition: true,
    kind: "isinstance",
  },
  {
    id: "load-types/compose_isinstance_load__an_unknown_type_name_in_a_flow_assert",
    at: ["asserts", 1],
    prefix: "flow assert 'isinstance(d.output, Byu)': ",
    condition: true,
    kind: "isinstance",
  },
  {
    id: "load-types/compose_isinstance_load__an_unknown_type_name_in_a_flow_output",
    at: ["output", "is_buy"],
    prefix: "flow output 'is_buy': ",
    kind: "isinstance",
  },
  {
    id: "load-types/compose_isinstance_load__an_unknown_type_name_in_an_inline_body",
    at: ["nodes", "inline", "nodes", "outer", "nodes", "inner", "input", "is_buy"],
    prefix: "node 'inner' input 'is_buy': ",
    kind: "isinstance",
  },
  {
    id: "load-core/compose_nested_template_walkers__unknown_nested_isinstance_name_fails",
    at: ["nodes", "c", "input", "x"],
    prefix: "node 'c' input 'x': ",
    template: true,
    kind: "isinstance",
  },
  {
    id: "load-core/engine_binding_labels_flows__unknown_type_in_a_single_flow_output",
    at: ["output"],
    prefix: "flow output: ",
    kind: "isinstance",
  },
  {
    id: "load-loop/engine_loop_build__predicate_isinstance_with_an_unknown_type",
    at: ["nodes", "ask", "until"],
    prefix: "loop node 'ask' until: ",
    condition: true,
    kind: "isinstance",
  },
  {
    id: "load-types/compose_constructor_load__an_unknown_name_in_a_case_when_is_an_unknown_function",
    at: ["nodes", "route", "cases", 0, "when"],
    prefix: "case node 'route' cases[0].when: ",
    condition: true,
    kind: "constructor",
  },
  {
    id: "load-types/compose_constructor_load__a_builtin_type_name_in_a_case_when_is_not_a_conversion",
    at: ["nodes", "route", "cases", 0, "when"],
    prefix: "case node 'route' cases[0].when: ",
    condition: true,
    kind: "constructor",
  },
  {
    id: "load-case/engine_condition_calls__unknown_function_in_a_node_assert",
    at: ["nodes", "x", "asserts", 0],
    prefix: "node 'x' assert 'nosuchfn(n) > 0': ",
    condition: true,
    kind: "constructor",
  },
  {
    id: "load-case/engine_condition_calls__unknown_function_in_a_flow_assert",
    at: ["asserts", 0],
    prefix: "flow assert 'nosuchfn(input.v) > 0': ",
    condition: true,
    kind: "constructor",
  },
  {
    id: "load-case/engine_condition_calls__unknown_function_in_a_loop_while",
    at: ["nodes", "repeat", "while"],
    prefix: "loop node 'repeat' while: ",
    condition: true,
    kind: "constructor",
  },
])("resolve names: $id", runNameRow);

// ---------- an inline body ----------

test("an inline body cannot take a whole-value isinstance argument", () => {
  // Inline bodies infer each argument's type while the parent builds, and a whole-value
  // expression has no static type there; a named `defs:` body declares it instead.
  const text = `id: p
name: p
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', price: float}
  Order: Union(Buy, Sell)
  Running: Literal(running)
input:
  order: !object
nodes:
  d:
    kind: code
    input:
      order: \${input.order}
    output: Order
    code: |
      return inputs["order"]
  c:
    kind: call
    input:
      is_buy: \${isinstance(d.output, Buy)}
    nodes:
      e:
        kind: code
        input:
          is_buy: \${input.is_buy}
        code: |
          return inputs["is_buy"]
    output: \${e.output}
output: \${c.output}
`;

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    "cannot infer type for inline binding 'c.input.is_buy': a whole-value expression has no supported " +
      "static scalar type; declare its type with a tag (`is_buy: !<type> ...`) or use an explicitly typed " +
      "named `defs:` body",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(
    text.split("\n").findIndex((line) => line.includes("is_buy: ${isinstance")) + 1,
  );
});

test.each<CaseRow>([
  { id: "load-types/compose_isinstance_load__an_inline_body_cannot_take_a_whole_value_isinstance", call: assembled },
])("assemble: $id", runCase);
