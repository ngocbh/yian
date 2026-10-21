/**
 * The type names `isinstance(x, T)` and `T(...)` constructors name, their resolution, and the
 * type tables load fills from them: a port of `tests/compose/test_isinstance_load.py`.
 *
 * Ported: `test_collect_isinstance_names`;
 * `test_an_inline_body_cannot_take_a_whole_value_isinstance_argument` (pinning the whole message);
 * `test_an_unknown_type_name_is_a_located_load_error`,
 * `test_a_flow_assert_error_is_located_by_its_own_assert`,
 * `test_list_is_not_an_isinstance_type_name`,
 * `test_isinstance_in_wait_until_is_a_located_load_error`,
 * `test_isinstance_in_map_over_is_a_located_load_error`,
 * `test_isinstance_in_an_agent_prompt_is_a_located_load_error` and
 * `test_isinstance_in_a_human_input_question_is_a_located_load_error`;
 * `test_isinstance_in_a_call_nodes_asserts_is_a_located_load_error` and
 * `test_isinstance_in_a_loop_nodes_asserts_is_a_load_error` (the node-assert check); and, loaded
 * with `loadFlow`, `test_a_parent_cannot_use_a_childs_typedef`.
 *
 * Ported through `loadFlow` and `runFlow` (and `resumeFlow`): the type-table rows
 * `test_isinstance_in_a_node_input`, `test_isinstance_in_a_searched_case_is_first_true_wins`,
 * `test_isinstance_in_a_node_assert`, `test_isinstance_in_flow_asserts_fills_start_and_end`,
 * `test_isinstance_in_a_flow_output`, `test_isinstance_in_a_loop_while_runs_several_iterations`,
 * `test_isinstance_in_defs_and_nested_inline_children`,
 * `test_isinstance_in_a_map_per_item_input`, `test_isinstance_in_a_call_directive_argument`,
 * `test_isinstance_in_a_loop_seed_and_until` and `test_isinstance_in_a_case_on`;
 * `test_a_failing_flow_input_assert_fails_at_start`, `test_a_run_that_used_isinstance_replays` and
 * `test_a_uses_child_resolves_names_against_its_own_typedefs`. The sell rows of the first two are
 * covered by `run-core/compose_isinstance_load__isinstance_in_a_node_input` and
 * `run-core/compose_isinstance_load__isinstance_in_a_searched_case_first_true_wins`. The loop
 * body's CODE is TS's; a flow's int output is a `bigint`.
 *
 * TS-only: `collectConstructorNames`, `expressionTrees`, `resolveIsinstanceType`,
 * `resolveConstructorType`, `constructorTypeFn` and `isinstanceMembers` rows (Python's own
 * `isinstanceMembers` test is ported with the narrowing tests).
 */

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { type CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import {
  collectConstructorNames,
  collectIsinstanceNames,
  constructorTypeFn,
  expressionTrees,
  isinstanceMembers,
  resolveConstructorType,
  resolveIsinstanceType,
} from "../../src/compose/helpers/isinstance_types.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ExpressionError } from "../../src/expr/index.ts";
import type { Node } from "../../src/nodes/base.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
import { Binding, ValueTemplate } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { type Type, TypeCheckError, typeEquals } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadError,
  loadErrorAsync,
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

// ---------- type tables ----------

const FLOW_REG = readTypedefs({
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", price: "float" },
  Order: "Union(Buy, Sell)",
  Running: "Literal(running)",
});
const ORDER = typeFor("Order", FLOW_REG);
const SELL = typeFor("Sell", FLOW_REG);
const RUNNING = typeFor("Running", FLOW_REG);
const OBJECT = typeFor("object", new Map());
const BOOL = typeFor("bool", new Map());

const A_BUY = { action: "buy", qty: 3 };
const A_SELL = { action: "sell", price: 1.5 };

const TYPEDEFS = `typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', price: float}
  Order: Union(Buy, Sell)
  Running: Literal(running)
`;

// One CODE node `d` that returns the `order` input typed as `Order`.
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
    output: Order
    code: |
      return inputs["order"]
`;

const flow = (rest: string) => HEAD + rest;

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

/** Return the child flow nodes CALL node `node` was baked with. */
const childNodes = (node: Node | undefined) => ((node as CallNode).child as CompiledFlow).nodes;


const INPUT_FLOW = flow(`  check:
    kind: code
    input:
      ok: \${isinstance(d.output, Buy)}
      known: \${isinstance(input.order, object)}
    code: |
      return [inputs["ok"], inputs["known"]]
output: \${check.output}
`);

const CASE_FLOW = flow(`  route:
    kind: case
    cases:
      - when: isinstance(d.output, Buy)
        then: b
      - when: isinstance(d.output, Order)
        then: o
    else: n
  b:
    kind: code
    code: |
      return "buy"
  o:
    kind: code
    code: |
      return "order"
  n:
    kind: code
    code: |
      return "none"
output: \${b.output | o.output | n.output}
`);

const NODE_ASSERT_FLOW =
  "id: p\nname: p\n" +
  TYPEDEFS +
  `input:
  order: !object
nodes:
  step:
    kind: code
    input:
      order: \${input.order}
    code: |
      return inputs["order"]
    asserts:
      - isinstance(order, Order)
      - isinstance(output, Buy)
output: \${step.output}
`;

const FLOW_ASSERT_FLOW = flow(`output: \${d.output}
asserts:
  - isinstance(input.order, Order)
  - isinstance(d.output, Buy)
`);

const FLOW_OUTPUT_FLOW = flow(`output:
  order: \${d.output}
  is_buy: \${isinstance(d.output, Buy)}
`);

const LOOP_FLOW =
  "id: counter\nname: counter\n" +
  TYPEDEFS +
  `defs:
  bump:
    input:
      n: !int
      status: !str
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
          status: \${input.status}
        code: |
          const n = inputs["n"] + 1
          return {n, status: n >= 4 ? "done" : "running"}
        output:
          n: int
          status: str
    output: \${step.output}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
      status: running
    while: isinstance(status, Running)
    max: 10
output: \${loop.output}
`;

const CHILD_FLOWS =
  "id: p\nname: p\n" +
  TYPEDEFS +
  `input:
  order: !object
defs:
  classify:
    input:
      order: !object
    nodes:
      inner:
        kind: code
        input:
          is_sell: \${isinstance(input.order, Sell)}
        code: |
          return inputs["is_sell"]
    output: \${inner.output}
nodes:
  named:
    kind: call
    call: classify
    input:
      order: \${input.order}
  inline:
    kind: call
    input:
      order: \${input.order}
    nodes:
      outer:
        kind: call
        input:
          order: \${input.order}
        nodes:
          inner:
            kind: code
            input:
              is_buy: \${isinstance(input.order, Buy)}
            code: |
              return inputs["is_buy"]
        output: \${inner.output}
    output: \${outer.output}
output:
  sell: \${named.output}
  buy: \${inline.output}
`;

const FLAG_DEF = `defs:
  flag:
    input:
      is_buy: !bool
    nodes:
      e:
        kind: code
        input:
          is_buy: \${input.is_buy}
        code: |
          return inputs["is_buy"]
    output: \${e.output}
`;

const MAP_FLOW =
  "id: p\nname: p\n" +
  TYPEDEFS +
  FLAG_DEF +
  `input:
  orders: !list(object)
nodes:
  each:
    kind: map
    call: flag
    over: \${input.orders}
    input:
      is_buy: \${isinstance(item, Buy)}
output: \${each.output}
`;

const CALL_ARG_FLOW =
  flow(`  c:
    kind: code
    input:
      res: call(flag, is_buy=\${isinstance(d.output, Buy)})
    code: |
      return inputs["res"]
`) +
  FLAG_DEF +
  "output: ${c.output}\n";

const LOOP_SEED_FLOW =
  "id: p\nname: p\n" +
  TYPEDEFS +
  `input:
  order: !object
defs:
  bump:
    input:
      n: !int
      buy: !bool
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
          buy: \${input.buy}
        code: |
          return {"n": inputs["n"] + 1, "buy": inputs["buy"]}
        output:
          n: int
          buy: bool
    output: \${step.output}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
      buy: \${isinstance(input.order, Buy)}
    until: n >= 3 and isinstance(buy, bool)
    max: 10
output: \${loop.output}
`;

const ON_FLOW = flow(`  route:
    kind: case
    on: \${isinstance(d.output, Buy)}
    cases:
      - when: true
        then: b
    else: o
  b:
    kind: code
    code: |
      return "buy"
  o:
    kind: code
    code: |
      return "other"
output: \${b.output | o.output}
`);

describe("type tables and runs", () => {
  test("isinstance in a node input", async () => {
    const loaded = await loadFlow(INPUT_FLOW);
    expectTypes(loaded.compiled.nodes.get("check"), { Buy: BUY, object: OBJECT });
    expectTypes(loaded.compiled.nodes.get("d"), {});

    const result = await runFlow(loaded, { order: A_BUY });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual([true, true]);
  });

  test("isinstance in a searched case is first true wins", async () => {
    const loaded = await loadFlow(CASE_FLOW);
    expectTypes(loaded.compiled.nodes.get("route"), { Buy: BUY, Order: ORDER });

    const result = await runFlow(loaded, { order: A_BUY });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe("buy");
  });

  test.each<[object, "succeeded" | "failed"]>([
    [A_BUY, "succeeded"],
    [A_SELL, "failed"],
  ])("isinstance in a node assert: %j", async (order, status) => {
    const loaded = await loadFlow(NODE_ASSERT_FLOW);
    expectTypes(loaded.compiled.nodes.get("step"), { Buy: BUY, Order: ORDER });

    const result = await runFlow(loaded, { order });

    expect(result.status, result.failure?.message).toBe(status);
    if (status === "failed") expect(result.failure!.code).toBe("assertion.failed");
  });

  test("isinstance in flow asserts fills start and end", async () => {
    const loaded = await loadFlow(FLOW_ASSERT_FLOW);
    expectTypes(loaded.compiled.nodes.get(START_ID), { Order: ORDER });
    expectTypes(loaded.compiled.nodes.get(END_ID), { Buy: BUY });

    expect((await runFlow(loaded, { order: A_BUY })).status).toBe("succeeded");
    const failed = await runFlow(loaded, { order: A_SELL });
    expect(failed.status).toBe("failed");
    expect(failed.failure!.code).toBe("assertion.failed");
  });

  test("a failing flow input assert fails at start", async () => {
    const failed = await runFlow(await loadFlow(FLOW_ASSERT_FLOW), { order: { action: "hold" } });

    expect(failed.status).toBe("failed");
    expect(failed.failure!.nodeId).toBe(START_ID);
  });

  test.each<[object, boolean, object]>([
    [A_BUY, true, { action: "buy", qty: 3n }],
    [A_SELL, false, A_SELL],
  ])("isinstance in a flow output: %j", async (order, isBuy, output) => {
    const loaded = await loadFlow(FLOW_OUTPUT_FLOW);
    expectTypes(loaded.compiled.nodes.get(END_ID), { Buy: BUY });

    const result = await runFlow(loaded, { order });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual({ order: output, is_buy: isBuy });
  });

  test("isinstance in a loop while runs several iterations", async () => {
    const loaded = await loadFlow(LOOP_FLOW);
    expectTypes(loaded.compiled.nodes.get("loop"), { Running: RUNNING });

    const result = await runFlow(loaded, {});

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual({ n: 4n, status: "done" });
  });

  test("a run that used isinstance replays", async () => {
    const dir = mkdtempSync(join(tmpdir(), "yian-isinstance-"));
    try {
      const log = join(dir, "run.jsonl");
      const first = await runFlow(await loadFlow(LOOP_FLOW), {}, { log });
      expect(first.status, first.failure?.message).toBe("succeeded");
      const replayed = await resumeFlow(await loadFlow(LOOP_FLOW), { log });
      expect(replayed.status, replayed.failure?.message).toBe("succeeded");
      expect(replayed.output).toEqual(first.output);
      // Cut after the second iteration, so resume replays two iterations and runs two live.
      const kept = cutLogAfter(log, "loop#1/__end__");
      expect(kept).toContain("loop~1");
      expect(kept.some((nodeId) => nodeId?.startsWith("loop~2") || nodeId?.startsWith("loop#2/"))).toBe(false);
      const resumed = await resumeFlow(await loadFlow(LOOP_FLOW), { log });
      expect(resumed.status, resumed.failure?.message).toBe("succeeded");
      expect(resumed.output).toEqual(first.output);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test.each<[object, object]>([
    [A_BUY, { sell: false, buy: true }],
    [A_SELL, { sell: true, buy: false }],
  ])("isinstance in defs and nested inline children: %j", async (order, expected) => {
    const loaded = await loadFlow(CHILD_FLOWS);
    const nodes = loaded.compiled.nodes;
    expectTypes(childNodes(nodes.get("named")).get("inner"), { Sell: SELL });
    const outer = childNodes(nodes.get("inline")).get("outer");
    expectTypes(childNodes(outer).get("inner"), { Buy: BUY });
    expectTypes(nodes.get("inline"), {});

    const result = await runFlow(loaded, { order });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual(expected);
  });

  test("isinstance in a map per-item input", async () => {
    const loaded = await loadFlow(MAP_FLOW);
    expectTypes(loaded.compiled.nodes.get("each"), { Buy: BUY });

    const result = await runFlow(loaded, { orders: [A_BUY, A_SELL, A_BUY] });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual([true, false, true]);
  });

  test.each([
    [A_BUY, true],
    [A_SELL, false],
  ])("isinstance in a call directive argument: %j", async (order, expected) => {
    const loaded = await loadFlow(CALL_ARG_FLOW);
    const nodes = loaded.compiled.nodes;
    const synths = [...nodes].filter(([nodeId]) => nodeId.startsWith("__call")).map(([, node]) => node);
    expect(synths).toHaveLength(1);
    expectTypes(synths[0], { Buy: BUY });
    expectTypes(nodes.get("c"), {});

    const result = await runFlow(loaded, { order });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe(expected);
  });

  test.each([
    [A_BUY, true],
    [A_SELL, false],
  ])("isinstance in a loop seed and until: %j", async (order, buy) => {
    const loaded = await loadFlow(LOOP_SEED_FLOW);
    expectTypes(loaded.compiled.nodes.get("loop"), { Buy: BUY, bool: BOOL });

    const result = await runFlow(loaded, { order });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual({ n: 3n, buy });
  });

  test("isinstance in a case on", async () => {
    // The bool `on:` value matches `when: true`, so a buy routes to `b`; the run evaluates the
    // `on:` binding with the table.
    const loaded = await loadFlow(ON_FLOW);
    const route = loaded.compiled.nodes.get("route")!;
    expectTypes(route, { Buy: BUY, bool: BOOL });

    const result = await runFlow(loaded, { order: A_BUY });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe("buy");

    route.types = new Map();
    const failed = await runFlow(loaded, { order: A_BUY });
    expect(failed.status).toBe("failed");
    expect(failed.failure!.nodeId).toBe("route");
  });
});

/** Keep the log up to `nodeId`'s success record; return the kept records' node ids. */
function cutLogAfter(log: string, nodeId: string): (string | undefined)[] {
  const lines = readFileSync(log, "utf8").split(/(?<=\n)/);
  const records = lines.map((line) => JSON.parse(line) as { kind: string; payload: { node_id?: string } });
  const end = records.findIndex((record) => record.kind === "node_succeeded" && record.payload.node_id === nodeId);
  expect(end).toBeGreaterThanOrEqual(0);
  writeFileSync(log, lines.slice(0, end + 1).join(""));
  return records.slice(0, end + 1).map((record) => record.payload.node_id);
}

// ---------- an inline body ----------

test("an inline body cannot take a whole-value isinstance argument", () => {
  // Inline bodies infer each argument's type while the parent builds, and a whole-value
  // expression has no static type there; a named `defs:` body declares it instead.
  const text = flow(`  c:
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
`);

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    "cannot infer type for inline binding 'c.input.is_buy': a whole-value expression has no supported " +
      "static scalar type; declare its type with a tag (`is_buy: !<type> ...`) or use an explicitly typed " +
      "named `defs:` body",
  );
  expect(errorLine(error)).toBe(lineOf(text, "is_buy: ${isinstance"));
});

// ---------- `uses:` children ----------

const USES_CHILD = `id: child
name: child
typedefs:
  Ticket: {kind: 'Literal(ticket)', id: int}
input:
  value: !object
nodes:
  check:
    kind: code
    input:
      ok: \${isinstance(input.value, Ticket)}
    code: |
      return inputs["ok"]
output: \${check.output}
`;

const USES_PARENT = `id: parent
name: parent
input:
  value: !object
uses:
  child: child
nodes:
  run:
    kind: call
    call: child
    input:
      value: \${input.value}
output: \${run.output}
`;

/** A resolver over the loaded `USES_CHILD`; any other name is a load error. */
async function usesResolver(): Promise<(flowId: string) => ComposedFlow> {
  const loaded: Record<string, ComposedFlow> = { child: await loadFlow(USES_CHILD) };
  return (flowId) => {
    if (Object.hasOwn(loaded, flowId)) return loaded[flowId]!;
    throw new LoadDiagnostics().error(`unknown child ${JSON.stringify(flowId)}`);
  };
}

test("a uses child resolves names against its own typedefs", async () => {
  const flow = await loadFlow(USES_PARENT, { childResolver: await usesResolver() });

  const ticket = childNodes(flow.compiled.nodes.get("run")).get("check")!.types.get("Ticket")!;
  expect(typeEquals(ticket.fields!.get("id")!, typeFor("int", new Map()))).toBe(true);
  const result = await runFlow(flow, { value: { kind: "ticket", id: 7 } });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(true);
});

test("a parent cannot use a child's typedef", async () => {
  const text = USES_PARENT.replace(
    "      value: ${input.value}\n",
    "      value: ${input.value}\n  probe:\n    kind: code\n    input:\n" +
      "      ok: ${isinstance(input.value, Ticket)}\n    code: |\n      return 1\n",
  );

  const error = await loadErrorAsync(loadFlow(text, { childResolver: await usesResolver() }));

  expect(error.message).toContain("unknown type 'Ticket'");
  expect(errorLine(error)).toBe(lineOf(text, "isinstance(input.value, Ticket)"));
});

// ---------- located load errors ----------

/** One site of an unknown type name: the flow, its written text, the misspelling, its line's marker. */
const UNKNOWN_SITES: Record<string, [string, string, string, string]> = {
  "node input": [INPUT_FLOW, "d.output, Buy", "Buyy", "ok:"],
  "case when": [CASE_FLOW, "d.output, Order", "Ordr", "Ordr"],
  "node assert": [NODE_ASSERT_FLOW, "output, Buy", "Bu", "isinstance(output, Bu)"],
  "flow assert": [FLOW_ASSERT_FLOW, "d.output, Buy", "Byu", "Byu"],
  "flow output": [FLOW_OUTPUT_FLOW, "d.output, Buy", "Bye", "is_buy:"],
  "loop while": [LOOP_FLOW, "status, Running", "Runing", "while:"],
  "defs child": [CHILD_FLOWS, "input.order, Sell", "Sel", "is_sell:"],
  "inline child": [CHILD_FLOWS, "input.order, Buy", "By", "is_buy:"],
  "map per-item input": [MAP_FLOW, "item, Buy", "Buyy", "is_buy: ${isinstance(item"],
  "call argument": [CALL_ARG_FLOW, "d.output, Buy", "Buyy", "res: call"],
  "loop seed": [LOOP_SEED_FLOW, "input.order, Buy", "Buyy", "buy: ${isinstance"],
  "loop until": [LOOP_SEED_FLOW, "buy, bool", "boo", "until:"],
  "case on": [ON_FLOW, "d.output, Buy", "Buyy", "on: ${isinstance"],
};

test.each(Object.keys(UNKNOWN_SITES).sort())("an unknown type name is a located load error: %s", (site) => {
  const [source, written, name, marker] = UNKNOWN_SITES[site]!;
  const text = source.replace(written, `${written.split(", ")[0]}, ${name}`);
  expect(text).not.toBe(source);

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain(`unknown type '${name}'`);
  expect(error.message).toContain("typedefs");
  expect(error.message).not.toContain(":: ");
  expect(errorLine(error)).toBe(lineOf(text, marker));
});

test("a flow assert error is located by its own assert", () => {
  const text = flow(`output: \${d.output}
asserts:
  - isinstance(d.output, Buy)
  - isinstance(d.output, Buy) or isinstance(d.output, Nope)
`);

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("unknown type 'Nope'");
  expect(errorLine(error)).toBe(lineOf(text, "Nope"));
});

test("list is not an isinstance type name", () => {
  const text = INPUT_FLOW.replace("d.output, Buy", "d.output, list");

  expect(loadError(() => loadedNodes(text)).message).toContain("unknown type 'list'");
});

test("isinstance in wait until is a located load error", () => {
  const text = `id: w
name: w
input:
  when: !date
nodes:
  w:
    kind: wait
    until: \${isinstance(input.when, date)}
output: \${w.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("isinstance is not supported in a wait node's until:");
  expect(errorLine(error)).toBe(lineOf(text, "until:"));
});

test("isinstance in map over is a located load error", () => {
  const text =
    "id: m\nname: m\n" +
    TYPEDEFS +
    `input:
  items: !list(int)
defs:
  one:
    input:
      x: !int
    nodes:
      e:
        kind: code
        input:
          x: \${input.x}
        code: |
          return inputs["x"]
    output: \${e.output}
nodes:
  each:
    kind: map
    call: one
    over: \${isinstance(input.items, Buy)}
    input:
      x: \${item}
output: \${each.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("isinstance is not supported in a map node's over:");
  expect(errorLine(error)).toBe(lineOf(text, "over:"));
});

test("isinstance in a call node's asserts is a located load error", () => {
  const text = CHILD_FLOWS.replace("    call: classify\n", "    call: classify\n    asserts:\n      - isinstance(output, bool)\n");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'named' assert 'isinstance(output, bool)': isinstance in a call node's asserts is not " +
      "supported; assert inside the child flow",
  );
  expect(errorLine(error)).toBe(lineOf(text, "isinstance(output, bool)"));
});

test("isinstance in a loop node's asserts is a load error", () => {
  const text = LOOP_FLOW.replace("    max: 10\n", "    max: 10\n    asserts:\n      - isinstance(output, Buy)\n");

  expect(loadError(() => loadedNodes(text)).message).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'loop' assert 'isinstance(output, Buy)': isinstance in a loop node's asserts is not " +
      "supported; assert inside the child flow",
  );
});

test("isinstance in an agent prompt is a located load error", () => {
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
    prompt: "Is it a buy? \${isinstance(order, Buy)}"
    output: str
output: \${ask.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("isinstance is not a prompt function; compute it in an input binding");
  expect(errorLine(error)).toBe(lineOf(text, "prompt:"));
});

test("isinstance in a human input question is a located load error", () => {
  // The human-input node's own build refuses the question before any type table is filled.
  const text =
    "id: h\nname: h\n" +
    TYPEDEFS +
    `input:
  order: !object
nodes:
  ask:
    kind: human_input
    input:
      order: \${input.order}
    questions: '\${isinstance(order, Buy)}'
output: \${ask.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("exactly one `${name}`");
  expect(errorLine(error)).toBe(lineOf(text, "questions:"));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-types/compose_isinstance_load__an_inline_body_cannot_take_a_whole_value_isinstance", call: assembled },
  { id: "load-types/compose_isinstance_load__an_unknown_type_name_in_a_case_when", call: assembled },
  { id: "load-types/compose_isinstance_load__an_unknown_type_name_in_a_flow_assert", call: assembled },
  { id: "load-types/compose_isinstance_load__an_unknown_type_name_in_a_flow_output", call: assembled },
  // The inline body assembles through the harness's private-child registry.
  { id: "load-types/compose_isinstance_load__an_unknown_type_name_in_an_inline_body", call: assembled },
  { id: "load-types/compose_isinstance_load__isinstance_in_wait_until", call: assembled },
  { id: "load-types/compose_isinstance_load__isinstance_in_an_agent_prompt", call: assembled },
  {
    id: "load-types/compose_isinstance_load__isinstance_in_map_over",
    call: assembled,
  },
  { id: "load-types/compose_isinstance_load__isinstance_in_a_call_nodes_asserts", call: assembled },
  { id: "load-types/compose_isinstance_load__isinstance_in_a_loop_nodes_asserts", call: assembled },
])("assemble: $id", runCase);
