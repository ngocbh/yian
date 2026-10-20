/**
 * `${T(...)}` constructors resolve at load into each node's type table: a port of
 * `tests/compose/test_constructor_load.py`.
 *
 * A constructor name is a `typedefs:` name; `T(x)` checks a value (exactly `!T ${x}`), and
 * `T(a=..., b=...)` builds a record from exactly the given fields and checks it.
 *
 * Ported: `test_a_constructor_resolves_into_the_node_type_table`,
 * `test_a_span_inside_a_constructor_shows_the_rewrite`,
 * `test_isinstance_with_a_builtin_name_does_not_make_it_a_constructor`,
 * `test_a_flow_call_with_several_arguments_keeps_the_flow_call_message`,
 * `test_a_malformed_constructor_call_is_a_load_error` (the `case when` site),
 * `test_a_builtin_type_name_in_a_binding_is_not_a_conversion`,
 * `test_a_builtin_type_name_in_a_case_when_is_not_a_conversion`,
 * `test_an_unknown_name_in_a_case_when_is_an_unknown_function`,
 * `test_a_constructor_in_wait_until_is_a_located_load_error`,
 * `test_a_field_read_on_a_constructor_in_a_case_when_type_checks` and
 * `test_keywords_missing_a_field_in_a_case_when_are_a_load_error`; and the load halves of
 * `test_a_constructor_in_a_case_when` (its type table), `test_a_constructor_in_node_and_flow_asserts`,
 * `test_a_constructor_in_a_flow_output` and `test_a_constructor_in_a_map_per_item_input` (each
 * loads). The `one` def the map calls is a hand-built child (`handChild`) with the def's signature.
 * `engine/inline_calls.test.ts` pins the cases of the binding sites the inline-call desugar raises.
 *
 * TS-only: one row per constructor message the type-table pass reports at a `case when` (a union,
 * a scalar type, an unknown field, a disjoint value, a disjoint field, a builtin field read), which
 * Python checks only at binding sites.
 *
 * Later (T3-18 PR 3): the reference validator's binding and prompt sites:
 * `test_a_malformed_constructor_call_is_a_load_error` (the `binding` site),
 * `test_a_constructor_in_an_agent_prompt_is_a_located_load_error`,
 * `test_a_field_read_on_a_constructor_result_type_checks`, `test_a_disjoint_argument_is_a_load_error`,
 * `test_a_narrower_or_unknown_argument_loads`, `test_keywords_missing_a_required_field_are_a_load_error`,
 * `test_an_unknown_keyword_field_is_a_load_error`, `test_a_disjoint_keyword_field_is_a_load_error`,
 * `test_keywords_cannot_build_a_union` and `test_keywords_cannot_build_a_scalar_type`.
 *
 * Later (T3-19): the typed-binding and call-type rows
 * (`test_a_value_lacking_a_field_fails_with_binding_type_naming_it`,
 * `test_a_bad_built_field_fails_with_binding_type`, `test_a_nested_constructor_failure_is_binding_type`,
 * `test_downstream_sees_the_constructed_type`, `test_a_call_argument_is_checked_with_the_constructed_type`,
 * `test_an_inline_body_parameter_takes_the_constructed_type`).
 *
 * Later (T3-20): `test_a_constructor_in_a_call_nodes_asserts_is_a_located_load_error` (the
 * node-assert check).
 *
 * Later (T3-22): `test_a_caller_sees_a_childs_constructed_output_type` (the child's own assembly),
 * `test_a_uses_child_resolves_constructors_against_its_own_typedefs` and
 * `test_a_parent_cannot_construct_a_childs_typedef` (`uses:`).
 *
 * Later (T4): the run halves of the rows above, `test_a_value_that_fits_passes_unchanged`,
 * `test_keywords_build_the_record`, `test_a_union_typedef_and_an_enum_alias_check` and
 * `test_constructors_in_a_list_literal_and_a_nested_value`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import type { Node } from "../../src/nodes/base.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { type Type, typeEquals } from "../../src/typesys/values.ts";
import { assembled, assembleText, type CaseRow, handChild, loadError, parametersFromYaml, runCase } from "./declarations.ts";

install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

const REG = readTypedefs({
  Buy: { action: "Literal(buy)", conviction: "float" },
  Sell: { action: "Literal(sell)", price: "float" },
  Decision: "Union(Buy, Sell)",
  Size: "Literal(s, m, l)",
});
const BUY = typeFor("Buy", REG);

const TYPEDEFS = `typedefs:
  Buy: {action: Literal(buy), conviction: float}
  Sell: {action: Literal(sell), price: float}
  Decision: Union(Buy, Sell)
  Size: Literal(s, m, l)
`;

// One CODE node `d` that returns the untyped `order` input, so only a check can narrow it.
const HEAD =
  "id: p\nname: p\n" +
  TYPEDEFS +
  `input:
  order: !object
nodes:
  d:
    kind: code
    input:
      order: \${input.order}
    output: object
    code: |
      return inputs["order"]
`;

const flow = (rest: string) => HEAD + rest;

/** A flow whose node `c` binds `x: <binding>` and returns it. */
const consumer = (binding: string) =>
  flow(`  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs["x"]
output: \${c.output}
`);

/** Return the 1-based line of the first line of `text` containing `marker`. */
function lineOf(text: string, marker: string): number {
  const index = text.split("\n").findIndex((line) => line.includes(marker));
  if (index < 0) throw new Error(`${JSON.stringify(marker)} not in text`);
  return index + 1;
}

/** Return the start line of the error's last frame. */
const errorLine = (error: { diagnostic: { frames: readonly { range: { startLine: number } }[] } }) =>
  error.diagnostic.frames.at(-1)!.range.startLine;

/** Assemble `text` as the root flow, its `defs:` entries standing in as `children`; return its nodes. */
const loadedNodes = (text: string, children: Readonly<Record<string, ComposedFlow>> = {}) =>
  assembleText(text, "flow.yaml", { children }).flow.compiled.nodes;

/** Check that `node`'s type table holds exactly `expected`, in any order. */
function expectTypes(node: Node | undefined, expected: Record<string, Type>): void {
  const types = node!.types;
  expect([...types.keys()].sort()).toEqual(Object.keys(expected).sort());
  for (const [name, typ] of Object.entries(expected)) expect(typeEquals(types.get(name)!, typ)).toBe(true);
}

// ---------- resolution ----------

test("a constructor resolves into the node type table", () => {
  const nodes = loadedNodes(consumer("${Buy(d.output)}"));

  expectTypes(nodes.get("c"), { Buy: BUY });
  expectTypes(nodes.get("d"), {});
});

test.each([
  ["${Buy(${d.output})}", "${Buy(d.output)}"],
  ['${Buy(action="buy", conviction=${d.output.conviction})}', '${Buy(action="buy", conviction=d.output.conviction)}'],
])("a span inside a constructor shows the rewrite: %s", (binding, rewrite) => {
  const error = loadError(() => loadedNodes(consumer(`'${binding}'`)));

  expect(error.diagnostic.message).toBe(`c: \`\${...}\` can't appear inside an expression; write \`${rewrite}\``);
});

test("isinstance with a builtin name does not make it a constructor", () => {
  expectTypes(loadedNodes(consumer("${isinstance(d.output, int)}")).get("c"), { int: typeFor("int", new Map()) });
});

// ---------- positions ----------

const CASE_FLOW = flow(`  route:
    kind: case
    cases:
      - when: Buy(d.output).conviction > 0.4
        then: strong
    else: weak
  strong:
    kind: code
    code: return "strong"
  weak:
    kind: code
    code: return "weak"
output: \${strong.output | weak.output}
`);

test("a constructor in a case when (load half)", () => {
  expectTypes(loadedNodes(CASE_FLOW).get("route"), { Buy: BUY });
});

test("a constructor in node and flow asserts (load half)", () => {
  const text = flow(`  c:
    kind: code
    input:
      x: \${d.output}
    asserts:
      - Buy(x).conviction > 0.1
    code: |
      return inputs["x"]
output: \${c.output}
asserts:
  - \${Buy(c.output).conviction < 0.9}
`);

  expectTypes(loadedNodes(text).get("c"), { Buy: BUY });
});

test("a constructor in a flow output (load half)", () => {
  loadedNodes(flow("output:\n  decision: ${Buy(d.output)}\n"));
});

test("a constructor in a map per-item input (load half)", () => {
  const text =
    "id: m\nname: m\n" +
    TYPEDEFS +
    `input:
  orders: !list(object)
defs:
  one:
    input:
      x: !object
    nodes:
      e:
        kind: code
        input:
          x: \${input.x}
        code: |
          return inputs["x"]["conviction"]
    output: \${e.output}
nodes:
  each:
    kind: map
    call: one
    over: \${input.orders}
    input:
      x: \${Buy(item)}
output: \${each.output}
`;
  const one = handChild({
    inputs: parametersFromYaml("input:\n  x: !object\n", new Map()),
    producers: { e: null },
    output: "${e.output}",
  });

  expectTypes(loadedNodes(text, { one }).get("each"), { Buy: BUY });
});

test("a flow call with several arguments keeps the flow call message", () => {
  const error = loadError(() => loadedNodes(consumer("${review(d.output, d.output)}")));

  expect(error.message).toContain("a flow call 'review' inside ${...} is not supported");
});

test("a malformed constructor call is a load error: case when", () => {
  const text = CASE_FLOW.replace("Buy(d.output)", "Buy(d.output, d.output)");

  expect(loadError(() => loadedNodes(text)).message).toContain(
    "Buy(...) takes keyword fields, as in Buy(name=value, ...), or one value to check, as in Buy(x)",
  );
});

// ---------- names that are not constructors ----------

test("a builtin type name in a binding is not a conversion", () => {
  const text = consumer("${int(d.output)}");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain(
    "unknown expression function 'int'; types in expressions are typedef names; `int(...)` is not a conversion",
  );
  expect(errorLine(error)).toBe(lineOf(text, "x: ${int"));
});

test("a builtin type name in a case when is not a conversion", () => {
  const text = CASE_FLOW.replace("Buy(d.output).conviction > 0.4", "int(d.output) > 0");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("case node 'route' cases[0].when: unknown expression function 'int'; types");
  expect(errorLine(error)).toBe(lineOf(text, "int(d.output)"));
});

test("an unknown name in a case when is an unknown function", () => {
  const text = CASE_FLOW.replace("Buy(d.output)", "Hold(d.output)");

  expect(loadError(() => loadedNodes(text)).message.split("\n")[0]).toMatch(/unknown expression function 'Hold'$/);
});

test("a constructor in wait until is a located load error", () => {
  const text =
    "id: w\nname: w\n" +
    TYPEDEFS +
    `input:
  when: !date
nodes:
  w:
    kind: wait
    until: \${Size(input.when)}
output: \${w.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("Size(...) is not supported in a wait node's until:");
  expect(errorLine(error)).toBe(lineOf(text, "until:"));
});

// ---------- static types ----------

test("a field read on a constructor in a case when type checks", () => {
  const text = CASE_FLOW.replace("Buy(d.output).conviction", "Buy(d.output).size");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("reads unknown field 'size' on record");
  expect(errorLine(error)).toBe(lineOf(text, "Buy(d.output).size"));
});

test("keywords missing a field in a case when are a load error", () => {
  const text = CASE_FLOW.replace("Buy(d.output)", "Buy(conviction=d.output.conviction)");

  expect(loadError(() => loadedNodes(text)).message).toContain("Buy(...) is missing required field `action`");
});

describe("constructor messages at a case when", () => {
  test.each([
    [
      "Decision(conviction=1).conviction > 0.4",
      "Decision is a union; construct a member (Buy(...) or Sell(...)), or check a value with Decision(x)",
    ],
    ['Size(name="m") == "m"', "Size is not a record, so Size(...) takes one value to check, as in Size(x)"],
    [
      'Buy(action="buy", conviction=1, size="m").conviction > 0.4',
      "Buy has no field 'size' (fields: action, conviction)",
    ],
    ['Buy("x").conviction > 0.4', "Buy(...): the value is Literal(x), which can never be a Buy"],
    [
      'Buy(d.output).conviction > 0.4 and Size(3) == "m"',
      "Size(...): the value is int(3..3), which can never be a Size",
    ],
    [
      'Buy(action="sell", conviction=1).conviction > 0.4',
      "Buy(...) field 'action': declared Literal(buy), but the value is Literal(sell)",
    ],
    [
      'Buy(conviction=true, action="buy").conviction > 0.4',
      "Buy(...) field 'conviction': declared float, but the value is bool",
    ],
    ["upper(d.output).size > 0", "reference ${upper(...).size} reads field 'size' on type str, which has no fields"],
  ])("%s", (when, message) => {
    const text = CASE_FLOW.replace("Buy(d.output).conviction > 0.4", when);

    const error = loadError(() => loadedNodes(text));

    expect(error.message).toBe(`case node 'route' cases[0].when: ${message}`);
    expect(errorLine(error)).toBe(lineOf(text, "- when:"));
  });
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "load-types/compose_constructor_load__a_builtin_type_name_in_a_case_when_is_not_a_conversion",
    "load-types/compose_constructor_load__a_constructor_in_wait_until",
    "load-types/compose_constructor_load__a_field_read_on_a_constructor_in_a_case_when",
    "load-types/compose_constructor_load__a_malformed_constructor_call_in_a_case_when",
    "load-types/compose_constructor_load__an_unknown_name_in_a_case_when_is_an_unknown_function",
    "load-types/compose_constructor_load__keywords_missing_a_field_in_a_case_when",
  ].map((id) => ({ id, call: assembled })),
)("assembled: $id", runCase);
