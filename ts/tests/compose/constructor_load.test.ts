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
 * `test_a_malformed_constructor_call_is_a_load_error`,
 * `test_a_builtin_type_name_in_a_binding_is_not_a_conversion`,
 * `test_a_builtin_type_name_in_a_case_when_is_not_a_conversion`,
 * `test_an_unknown_name_in_a_case_when_is_an_unknown_function`,
 * `test_a_constructor_in_wait_until_is_a_located_load_error`,
 * `test_a_constructor_in_an_agent_prompt_is_a_located_load_error`,
 * `test_a_field_read_on_a_constructor_result_type_checks`,
 * `test_a_field_read_on_a_constructor_in_a_case_when_type_checks`,
 * `test_a_disjoint_argument_is_a_load_error`, `test_a_narrower_or_unknown_argument_loads`,
 * `test_keywords_missing_a_required_field_are_a_load_error`,
 * `test_keywords_missing_a_field_in_a_case_when_are_a_load_error`,
 * `test_an_unknown_keyword_field_is_a_load_error`, `test_a_disjoint_keyword_field_is_a_load_error`,
 * `test_keywords_cannot_build_a_union`, `test_keywords_cannot_build_a_scalar_type`,
 * `test_a_constructor_in_a_call_nodes_asserts_is_a_located_load_error` (the node-assert check,
 * pinning the whole message), the typed-binding row `test_downstream_sees_the_constructed_type`,
 * the call-type row `test_a_call_argument_is_checked_with_the_constructed_type` and
 * `test_a_parent_cannot_construct_a_childs_typedef`. `engine/inline_calls.test.ts` pins the cases
 * of the binding sites the inline-call desugar raises.
 *
 * Ported through `loadFlow` and `runFlow`: `test_a_constructor_in_node_and_flow_asserts`,
 * `test_a_constructor_in_a_flow_output`, `test_a_constructor_in_a_map_per_item_input`,
 * `test_a_uses_child_resolves_constructors_against_its_own_typedefs`,
 * `test_a_caller_sees_a_childs_constructed_output_type`,
 * `test_an_inline_body_parameter_takes_the_constructed_type`, the `weak` row of
 * `test_a_constructor_in_a_case_when` and the passing run of
 * `test_a_union_typedef_and_an_enum_alias_check`.
 *
 * Covered by `run-core` cases: `test_a_value_that_fits_passes_unchanged`
 * (`compose_constructor_load__a_value_that_fits_passes_unchanged`), the run of
 * `test_a_value_lacking_a_field_fails_with_binding_type_naming_it`
 * (`compose_constructor_load__a_value_lacking_a_field_fails`), `test_keywords_build_the_record`
 * (`compose_constructor_load__keywords_build_the_record`), the run of
 * `test_a_bad_built_field_fails_with_binding_type`
 * (`compose_constructor_load__a_bad_built_field_fails_with_binding_type`), the failing run of
 * `test_a_union_typedef_and_an_enum_alias_check`
 * (`compose_constructor_load__an_enum_alias_constructor_rejects_a_non_member`),
 * `test_constructors_in_a_list_literal_and_a_nested_value`
 * (`compose_constructor_load__constructors_in_a_list_and_a_nested_value`), the run of
 * `test_a_nested_constructor_failure_is_binding_type`
 * (`compose_constructor_load__a_nested_constructor_failure_is_binding_type`) and the `strong` row
 * of `test_a_constructor_in_a_case_when` (`compose_constructor_load__a_constructor_in_a_case_when`).
 * Their load halves stay here.
 *
 * TS-only: one row per constructor message the type-table pass reports at a `case when` (a union,
 * a scalar type, an unknown field, a disjoint value, a disjoint field, a builtin field read); Python
 * reports them there too, but its tests pin them only at binding sites.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import type { Node } from "../../src/nodes/base.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { runFlow } from "../../src/run.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { type Type, typeEquals } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadError,
  loadErrorAsync,
  runCase,
} from "./declarations.ts";

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

const A_BUY = { action: "buy", conviction: 0.5 };
const A_SELL = { action: "sell", price: 1.5 };

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

/** Assemble `text` as the root flow; return its nodes. */
const loadedNodes = (text: string) => assembleText(text).flow.compiled.nodes;

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

test("a union typedef and an enum alias check", async () => {
  const text = flow(`  c:
    kind: code
    input:
      x: \${Decision(d.output.decision)}
      size: \${Size(d.output.size)}
    code: |
      return [inputs["x"]["action"], inputs["size"]]
output: \${c.output}
`);

  const result = await runFlow(await loadFlow(text), { order: { decision: A_SELL, size: "m" } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual(["sell", "m"]);
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

test("a constructor in a case when", async () => {
  const loaded = await loadFlow(CASE_FLOW);
  expectTypes(loaded.compiled.nodes.get("route"), { Buy: BUY });

  const result = await runFlow(loaded, { order: { action: "buy", conviction: 0.1 } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("weak");
});

test("a constructor in node and flow asserts", async () => {
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

  const loaded = await loadFlow(text);
  expectTypes(loaded.compiled.nodes.get("c"), { Buy: BUY });

  expect((await runFlow(loaded, { order: A_BUY })).status).toBe("succeeded");
  expect((await runFlow(loaded, { order: { action: "buy", conviction: 0.95 } })).status).toBe("failed");
  expect((await runFlow(loaded, { order: { action: "buy", conviction: 0.05 } })).status).toBe("failed");
});

test("a constructor in a flow output", async () => {
  const result = await runFlow(await loadFlow(flow("output:\n  decision: ${Buy(d.output)}\n")), { order: A_BUY });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ decision: A_BUY });
});

test("a constructor in a map per-item input", async () => {
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
  const loaded = await loadFlow(text);
  expectTypes(loaded.compiled.nodes.get("each"), { Buy: BUY });

  const result = await runFlow(loaded, { orders: [A_BUY, A_BUY] });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual([0.5, 0.5]);
});

// ---------- typed bindings and call types over a constructor ----------

test.each([
  // `test_a_value_lacking_a_field_fails_with_binding_type_naming_it`
  "${Buy(d.output)}",
  // `test_a_bad_built_field_fails_with_binding_type`
  '${Buy(action="buy", conviction=d.output.price)}',
])("a binding whose run fails with binding.type loads: %s", (binding) => {
  loadedNodes(consumer(binding));
});

test("a nested constructor whose run fails with binding.type loads", () => {
  loadedNodes(
    flow(`  c:
    kind: code
    input:
      nested:
        first: \${Buy(d.output)}
    code: |
      return inputs["nested"]
output: \${c.output}
`),
  );
});

test("downstream sees the constructed type", () => {
  // A typed binding over a constructor compares the declared type with the constructor's.
  const error = loadError(() => loadedNodes(consumer("!Sell ${Buy(d.output)}")));

  expect(error.message).toContain('declared Sell, but the value is {"action":Literal(buy),"conviction":float}');
});

test("a call argument is checked with the constructed type", () => {
  const text = flow(`  run:
    kind: call
    call: sell
    input:
      order: \${Buy(d.output)}
output: \${run.output}
`).replace(
    "nodes:\n",
    `defs:
  sell:
    input:
      order: !Sell
    nodes:
      e:
        kind: code
        input:
          order: \${input.order}
        code: |
          return inputs["order"]
    output: \${e.output}
nodes:
`,
  );

  expect(loadError(() => loadedNodes(text)).message).toContain("call node 'run': binding 'order'");
});

const PARENT =
  "id: p\nname: p\n" +
  TYPEDEFS +
  `input:
  order: !object
defs:
  decide:
    input:
      order: !object
    nodes:
      e:
        kind: code
        input:
          order: \${input.order}
        code: |
          return inputs["order"]
    output: \${Buy(e.output)}
nodes:
  run:
    kind: call
    call: decide
    input:
      order: \${input.order}
  use:
    kind: code
    input:
      conviction: \${run.output.conviction}
    code: |
      return inputs["conviction"]
output: \${use.output}
`;

test("a caller sees a child's constructed output type", async () => {
  const result = await runFlow(await loadFlow(PARENT), { order: A_BUY });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(0.5);
  const text = PARENT.replace("${run.output.conviction}", "${run.output.size}");

  expect((await loadErrorAsync(loadFlow(text))).message).toContain("reads unknown field 'size' on record");
});

test("an inline body parameter takes the constructed type", async () => {
  const text = flow(`  inline:
    kind: call
    input:
      decision: \${Buy(d.output)}
    nodes:
      e:
        kind: code
        input:
          c: \${input.decision.conviction}
        code: |
          return inputs["c"]
    output: \${e.output}
output: \${inline.output}
`);

  const loaded = await loadFlow(text);
  const child = (loaded.compiled.nodes.get("inline") as CallNode).child as CompiledFlow;

  expect(typeEquals(child.nodes.get(child.startId)!.params![0]!.contract!.type, BUY)).toBe(true);
  const result = await runFlow(loaded, { order: A_BUY });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(0.5);
});

const USES_CHILD = `id: child
name: child
typedefs:
  Hold: {action: Literal(hold)}
input:
  order: !object
nodes:
  c:
    kind: code
    input:
      x: \${Hold(input.order)}
    code: |
      return inputs["x"]
output: \${c.output}
`;

const USES_PARENT = `id: parent
name: parent
input:
  order: !object
uses:
  child: child
nodes:
  run:
    kind: call
    call: child
    input:
      order: \${input.order}
output: \${run.output}
`;

/** A resolver that hands back the loaded `USES_CHILD` for `child`. */
async function usesResolver(): Promise<(flowId: string) => ComposedFlow> {
  const loaded: Record<string, ComposedFlow> = { child: await loadFlow(USES_CHILD) };
  return (flowId) => loaded[flowId]!;
}

test("a uses child resolves constructors against its own typedefs", async () => {
  const loaded = await loadFlow(USES_PARENT, { childResolver: await usesResolver() });

  const child = (loaded.compiled.nodes.get("run") as CallNode).child as CompiledFlow;
  expect(child.nodes.get("c")!.types.has("Hold")).toBe(true);
  const result = await runFlow(loaded, { order: { action: "hold" } });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ action: "hold" });
});

test("a flow call with several arguments keeps the flow call message", () => {
  const error = loadError(() => loadedNodes(consumer("${review(d.output, d.output)}")));

  expect(error.message).toContain("a flow call 'review' inside ${...} is not supported");
});

test.each([
  ["binding", consumer('${Buy(d.output, action="buy")}')],
  ["case when", CASE_FLOW.replace("Buy(d.output)", "Buy(d.output, d.output)")],
])("a malformed constructor call is a load error: %s", (_, text) => {
  expect(loadError(() => loadedNodes(text)).message).toContain(
    "Buy(...) takes keyword fields, as in Buy(name=value, ...), or one value to check, as in Buy(x)",
  );
});

test("a parent cannot construct a child's typedef", async () => {
  const text = USES_PARENT.replace(
    "      order: ${input.order}\n",
    "      order: ${input.order}\n  probe:\n    kind: code\n    input:\n" +
      "      x: ${Hold(input.order)}\n    code: |\n      return 1\n",
  );

  const error = await loadErrorAsync(loadFlow(text, { childResolver: await usesResolver() }));

  expect(error.message).toContain("a flow call 'Hold' inside ${...} is not supported");
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

test("a constructor in a call node's asserts is a located load error", () => {
  const text =
    "id: p\nname: p\n" +
    TYPEDEFS +
    `input:
  order: !object
defs:
  check:
    input:
      order: !object
    nodes:
      e:
        kind: code
        input:
          order: \${input.order}
        code: |
          return inputs["order"]
    output: \${e.output}
nodes:
  run:
    kind: call
    call: check
    input:
      order: \${input.order}
    asserts:
      - Buy(output).conviction > 0
output: \${run.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'run' assert 'Buy(output).conviction > 0': Buy(...) in a call node's asserts is not " +
      "supported; assert inside the child flow",
  );
  expect(errorLine(error)).toBe(lineOf(text, "Buy(output)"));
});

test("a constructor in an agent prompt is a located load error", () => {
  const text =
    "id: a\nname: a\n" +
    TYPEDEFS +
    `input:
  order: !object
nodes:
  ask:
    kind: agent
    input:
      order: \${input.order}
    prompt: "Is it a buy? \${Buy(order)}"
    output: str
output: \${ask.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("unknown prompt function 'Buy'");
  expect(error.message).toContain(
    "build or check a typed value with a `typedefs:` constructor `T(...)` in an input binding",
  );
  expect(errorLine(error)).toBe(lineOf(text, "prompt:"));
});

// ---------- static types ----------

/** Like `HEAD`, with `d` declaring `dOutput`. */
const typedHead = (dOutput: string) => HEAD.replace("    output: object\n", `    output: ${dOutput}\n`);

test("a field read on a constructor result type checks", () => {
  loadedNodes(consumer("${Buy(d.output).conviction}"));
  const text = consumer("${Buy(d.output).size}");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("reference ${Buy(...).size} reads unknown field 'size' on record");
  expect(errorLine(error)).toBe(lineOf(text, "x: ${Buy"));
});

test("a field read on a constructor in a case when type checks", () => {
  const text = CASE_FLOW.replace("Buy(d.output).conviction", "Buy(d.output).size");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("reads unknown field 'size' on record");
  expect(errorLine(error)).toBe(lineOf(text, "Buy(d.output).size"));
});

test("a disjoint argument is a load error", () => {
  const text =
    typedHead("Sell") +
    `  c:
    kind: code
    input:
      x: \${Buy(d.output)}
    code: |
      return inputs["x"]
output: \${c.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain(
    'Buy(...): the value is {"action":Literal(sell),"price":float}, which can never be a Buy',
  );
  expect(errorLine(error)).toBe(lineOf(text, "x: ${Buy"));
});

test("a narrower or unknown argument loads", () => {
  loadedNodes(typedHead("Decision") + "output: ${Buy(d.output)}\n");
  loadedNodes(typedHead("Optional(Buy)") + "output: ${Buy(d.output)}\n");
});

test("keywords missing a required field are a load error", () => {
  const text = consumer("${Buy(conviction=d.output.conviction)}");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("Buy(...) is missing required field `action`");
  expect(errorLine(error)).toBe(lineOf(text, "x: ${Buy"));
});

test("keywords missing a field in a case when are a load error", () => {
  const text = CASE_FLOW.replace("Buy(d.output)", "Buy(conviction=d.output.conviction)");

  expect(loadError(() => loadedNodes(text)).message).toContain("Buy(...) is missing required field `action`");
});

test("an unknown keyword field is a load error", () => {
  const text = consumer('${Buy(action="buy", conviction=1, size="m")}');

  expect(loadError(() => loadedNodes(text)).message).toContain("Buy has no field 'size' (fields: action, conviction)");
});

test.each([
  [
    '${Buy(action="sell", conviction=1)}',
    "Buy(...) field 'action': declared Literal(buy), but the value is Literal(sell)",
  ],
  ['${Buy(action="buy", conviction=d.output)}', "Buy(...) field 'conviction': declared float, but the value is str"],
])("a disjoint keyword field is a load error: %s", (binding, message) => {
  const text =
    typedHead("str") +
    `  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs["x"]
output: \${c.output}
`;

  expect(loadError(() => loadedNodes(text)).message).toContain(message);
});

test("keywords cannot build a union", () => {
  const error = loadError(() => loadedNodes(consumer("${Decision(conviction=1)}")));

  expect(error.message).toContain(
    "Decision is a union; construct a member (Buy(...) or Sell(...)), or check a value with Decision(x)",
  );
});

test("keywords cannot build a scalar type", () => {
  const error = loadError(() => loadedNodes(consumer('${Size(name="m")}')));

  expect(error.message).toContain("Size is not a record, so Size(...) takes one value to check, as in Size(x)");
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
    "a_builtin_type_name_in_a_case_when_is_not_a_conversion",
    "a_constructor_in_wait_until",
    "a_field_read_on_a_constructor_in_a_case_when",
    "a_malformed_constructor_call_in_a_case_when",
    "an_unknown_name_in_a_case_when_is_an_unknown_function",
    "keywords_missing_a_field_in_a_case_when",
    "a_constructor_in_an_agent_prompt",
    "a_disjoint_argument_is_a_load_error",
    "a_disjoint_keyword_field_is_a_load_error",
    "a_field_read_on_a_constructor_result_type_checks",
    "a_malformed_constructor_call_in_a_binding",
    "an_unknown_keyword_field_is_a_load_error",
    "keywords_cannot_build_a_scalar_type",
    "keywords_cannot_build_a_union",
    "keywords_missing_a_required_field",
    "downstream_sees_the_constructed_type",
  ].map((name) => ({ id: `load-types/compose_constructor_load__${name}`, call: assembled })),
)("assembled: $id", runCase);

test("assembled: load-types/compose_constructor_load__a_constructor_in_a_call_nodes_asserts", () => {
  runCase({ id: "load-types/compose_constructor_load__a_constructor_in_a_call_nodes_asserts", call: assembled });
});
