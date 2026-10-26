/**
 * `isinstance(x, T)` in the expression language: the evaluation and parse tests of
 * `tests/engine/test_isinstance.py`, `test_reference_walks_skip_the_type_name`,
 * `test_rewrites_leave_the_type_name_alone`, `test_isinstance_subject_is_the_tested_expression`,
 * `test_infer_expr_type_gives_bool_without_looking_up_the_type_name`,
 * `test_eval_binding_and_eval_template_use_the_type_table` (an embedded `false` is written
 * `false`), the co-skip, flow-callee and prompt-function tests, and an `isinstanceTypeNames`
 * test.
 *
 * `isinstance` is a special form: the first argument is an expression, the second a bare type
 * name looked up in the type table (`types`, a `Map`), never a reference.
 *
 * Ported: `test_a_node_starts_with_an_empty_type_table` (the loaded flow through `assembleText`)
 * and `test_case_forward_routes_with_its_type_table`. Ported through `loadFlow` and `runFlow` (and
 * `advance`), a loaded node's type table set by hand as Python sets it:
 * `test_bindings_and_case_conditions_use_the_node_type_table`,
 * `test_a_node_without_the_type_name_fails_naming_it` (failing `binding.evaluate`),
 * `test_node_asserts_use_the_node_type_table`, `test_loop_respawn_copies_the_type_table`,
 * `test_loop_while_uses_the_type_table_across_iterations`,
 * `test_loop_with_a_type_table_resumes_mid_loop`,
 * `test_a_read_guarded_by_isinstance_routes_a_non_member_to_else` and
 * `test_a_then_target_named_default_is_narrowed_by_a_searched_isinstance`.
 *
 * Changed: the CODE bodies are JavaScript, and a flow's int output is a `bigint`. The guarded
 * rows' limits `9.0` and `2.0` reach the run as ints (a whole host number reads as an int,
 * decisions 9A and 45), which a `float` field accepts (decision 38).
 */

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import {
  conditionRefs,
  ExpressionError,
  evalExpr,
  evaluateWhen,
  evaluateWhenRecord,
  exprRefs,
  inferExprType,
  isinstanceSubject,
  isinstanceTypeNames,
  ResolveMode,
  type RootResolver,
  rewriteConditionRefs,
  rewriteExprRefs,
} from "../../src/expr/expressions.ts";
import { type ExprNode, parseExpr, Token } from "../../src/expr/grammar.ts";
import { SourceRef } from "../../src/expr/lineage.ts";
import {
  bindingCoSkips,
  evalBinding,
  evalTemplate,
  exprRefsOf,
  flowCallCalleesInSpans,
  promptRefs,
  rewriteTemplateRefs,
  scanTemplate,
} from "../../src/expr/template.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import { advance, runFlow } from "../../src/run.ts";
import { Type, type Value, type ValueRecord, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import { fieldOf } from "../../src/typesys/values.ts";
import { assembleText } from "../compose/declarations.ts";
import { startedIds } from "./run_state_fixtures.ts";

/** Build a closed record tagged by a single-value `action` Literal. */
function member(tag: string, fields: Record<string, ValueKind>): Type {
  const recordFields = new Map<string, Type>([["action", new Type({ kind: ValueKind.STRING, tags: new Set([tag]) })]]);
  for (const [name, kind] of Object.entries(fields)) recordFields.set(name, Type.scalar(kind));
  return new Type({ kind: ValueKind.OBJECT, fields: recordFields, required: new Set(recordFields.keys()) });
}

const BUY = member("buy", { qty: ValueKind.INTEGER });
const SELL = member("sell", { price: ValueKind.NUMBER });
const TYPES = new Map([
  ["Buy", BUY],
  ["Sell", SELL],
  ["Order", new Type({ kind: ValueKind.OBJECT, variants: [BUY, SELL] })],
]);

const A_BUY: ValueRecord = { action: "buy", qty: 3n };
const A_SELL: ValueRecord = { action: "sell", price: 1.5 };

/** Return a `resolve` over `values`: the head, then `fieldOf` per dotted step; a miss is `null`. */
function resolver(values: ValueRecord = {}): RootResolver {
  return (names) => {
    const [present, value] = fieldOf(values, names[0]!);
    return { used: 1, present, value, why: null };
  };
}

/** Evaluate `text` against `resolve` with the type table `types`. */
function evaluate(
  text: string,
  resolve: RootResolver,
  types: ReadonlyMap<string, Type> | null = TYPES,
): Value {
  return evalExpr(parseExpr(text), resolve, { types });
}

/** Expect `run` to throw an `ExpressionError` whose message is exactly `message`. */
function expectError(run: () => unknown, message: string): void {
  let error: unknown;
  try {
    run();
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe(message);
}

// --- evaluation --- //

test("evalExpr checks the value against the named type", () => {
  const resolve = resolver({ d: { output: A_BUY } });
  expect(evaluate("isinstance(d.output, Buy)", resolve)).toBe(true);
  expect(evaluate("isinstance(d.output, Sell)", resolve)).toBe(false);
  expect(evaluate("isinstance(d.output, Order)", resolve)).toBe(true);
});

test("a missing value is not an instance of a record", () => {
  expect(evaluate("isinstance(d, Buy)", resolver())).toBe(false);
  expect(evaluate("isinstance(d?.output, Buy)", resolver())).toBe(false);
});

test("isinstance composes with other operators", () => {
  const resolve = resolver({ d: { output: A_BUY } });
  expect(evaluate("isinstance(d.output, Buy) and d.output.qty > 2", resolve)).toBe(true);
  expect(evaluate("not isinstance(d.output, Buy)", resolve)).toBe(false);
});

test("evaluateWhenRecord uses the type table", () => {
  expect(evaluateWhenRecord("isinstance(order, Sell)", { order: A_SELL }, TYPES)).toBe(true);
  expect(evaluateWhenRecord("${isinstance(order, Buy)}", { order: A_SELL }, TYPES)).toBe(false);
});

test("evaluateWhen uses the type table", () => {
  const pool = new VariablePool();
  pool.set("d", A_SELL);
  expect(evaluateWhen("isinstance(d.output, Sell)", pool, TYPES)).toBe(true);
  expect(evaluateWhen("isinstance(d.output, Buy)", pool, TYPES)).toBe(false);
});

test("evalBinding and evalTemplate use the type table", () => {
  const resolve = resolver({ d: { output: A_BUY } });
  expect(evalBinding("${isinstance(d.output, Buy)}", resolve, { types: TYPES })).toBe(true);
  expect(evalBinding("buy? ${isinstance(d.output, Sell)}", resolve, { types: TYPES })).toBe("buy? false");
  const segments = scanTemplate("${isinstance(d.output, Buy)}");
  expect(evalTemplate(segments, resolve, { types: TYPES })).toBe(true);
});

test("scalar type names check scalars", () => {
  const types = new Map([
    ["int", Type.scalar(ValueKind.INTEGER)],
    ["str", Type.scalar(ValueKind.STRING)],
    ["float", Type.scalar(ValueKind.NUMBER)],
  ]);
  const resolve = resolver({ n: 3n });
  expect(evaluate("isinstance(n, int)", resolve, types)).toBe(true);
  expect(evaluate("isinstance(n, str)", resolve, types)).toBe(false);
  // `inf` is a float, never an int.
  expect(evaluate("isinstance(inf, float)", resolve, types)).toBe(true);
  expect(evaluate("isinstance(inf, int)", resolve, types)).toBe(false);
});

test.each([
  ["no table", null],
  ["an empty table", new Map()],
  ["a table without it", new Map([["Sell", SELL]])],
] as [string, ReadonlyMap<string, Type> | null][])(
  "an unknown type name is an error naming it: %s",
  (_label, types) => {
    const resolve = resolver({ d: { output: A_BUY } });
    expectError(() => evaluate("isinstance(d.output, Buy)", resolve, types), "isinstance: unknown type 'Buy'");
  },
);

test("record evaluation without types names the type", () => {
  expectError(
    () => evaluateWhenRecord("isinstance(order, Buy)", { order: A_BUY }),
    "isinstance: unknown type 'Buy'",
  );
});

test("a missing value raises under STRICT_RAISE", () => {
  const tree = parseExpr("isinstance(d, Buy)");
  expectError(
    () => evalExpr(tree, resolver(), { mode: ResolveMode.STRICT_RAISE, types: TYPES }),
    "unresolved reference ${d} in expression",
  );
});

test("an Optional alias accepts a missing value", () => {
  const types = new Map([["MaybeBuy", BUY.replace({ nullable: true })]]);
  expect(evaluate("isinstance(d?.output, MaybeBuy)", resolver(), types)).toBe(true);
  expect(evaluate("isinstance(d.output, MaybeBuy)", resolver({ d: { output: A_BUY } }), types)).toBe(true);
  expect(evaluate("isinstance(d.output, MaybeBuy)", resolver({ d: { output: A_SELL } }), types)).toBe(false);
});

// --- parsing --- //

/** Expect `parseExpr(text)` to fail with exactly `message`. */
function expectMessage(text: string, message: string): void {
  let error: unknown;
  try {
    parseExpr(text);
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe(message);
}

const TYPE_NAME = "isinstance's second argument must be a bare type name, e.g. isinstance(x, Buy)";

test("isinstance takes exactly two arguments", () => {
  const cases: [string, number][] = [
    ["isinstance(x)", 1],
    ["isinstance()", 0],
    ["isinstance(x, Buy, Sell)", 3],
  ];
  for (const [text, count] of cases) {
    expectMessage(text, `isinstance takes exactly two arguments (a value and a type name), got ${count}`);
  }
});

test("the second argument must be a bare type name", () => {
  for (const text of [
    'isinstance(x, "Buy")',
    "isinstance(x, a.b)",
    "isinstance(x, f(Buy))",
    "isinstance(x, 3)",
    "isinstance(x, [Buy])",
    "isinstance(x, in)",
  ]) {
    expectMessage(text, TYPE_NAME);
  }
});

test("a not as the type name is a parse error", () => {
  expectMessage(
    "isinstance(x, not)",
    "could not parse expression 'isinstance(x, not)': unexpected ')' at line 1, column 18",
  );
});

test("a wrapped type name shows the bare one", () => {
  expectMessage("isinstance(x, ${Buy})", "`${...}` can't appear inside an expression; write `isinstance(x, Buy)`");
});

test("isinstance has no keyword form", () => {
  for (const text of ["isinstance(value=x, type=Buy)", "isinstance(x, type=Buy)"]) {
    expectMessage(text, "isinstance takes positional arguments, not keyword arguments");
  }
});

test("isinstance result has no fields to read", () => {
  expectMessage("isinstance(x, Buy).ok", "isinstance(...) gives a bool; it has no fields to read");
});

test("isinstance nested in a larger expression is validated", () => {
  expectMessage("a > 1 and isinstance(x)", "isinstance takes exactly two arguments (a value and a type name), got 1");
  expectMessage("upper(isinstance(x, a.b))", TYPE_NAME);
});

test("reference walks skip the type name", () => {
  expect(exprRefs(parseExpr("isinstance(d.output, Buy)"))).toEqual(["d.output"]);
  expect(conditionRefs("isinstance(d.output, Buy) and d.output.qty > 1")).toEqual(["d.output", "d.output.qty"]);
  expect(conditionRefs("${isinstance(d.output, Buy)}")).toEqual(["d.output"]);
  expect(exprRefsOf("ok: ${isinstance(d.output, Buy)}")).toEqual(["d.output"]);
});

test("rewrites leave the type name alone", () => {
  const rename = (path: string) => ({ text: "x." + path, used: path.split(".").length });
  expect(rewriteExprRefs("isinstance(d.output, Buy)", rename)).toBe("isinstance(x.d.output, Buy)");
  expect(rewriteConditionRefs("${isinstance(d.output, Buy)}", rename)).toBe("${isinstance(x.d.output, Buy)}");
  expect(rewriteTemplateRefs("ok ${isinstance(d.output, Buy)}", rename)).toBe("ok ${isinstance(x.d.output, Buy)}");
});

/** A position-blind form of `node`, as Lark's `Tree`/`Token` equality ignores positions. */
function plain(node: ExprNode): unknown {
  if (node instanceof Token) return { type: node.type, value: node.value };
  return { data: node.data, children: node.children.map(plain) };
}

test("isinstanceSubject is the tested expression", () => {
  const cases: [string, string | null][] = [
    ["isinstance(d.output, Buy)", "d.output"],
    ["isinstance(upper(x), str)", "upper(x)"],
    ["isinstance(d.output, Buy) and x", null],
    ["not isinstance(d.output, Buy)", null],
    ["upper(x)", null],
    ["x", null],
  ];
  for (const [text, subject] of cases) {
    const found = isinstanceSubject(parseExpr(text));
    if (subject === null) expect(found).toBeNull();
    else expect(plain(found!)).toEqual(plain(parseExpr(subject)));
  }
});

test("inferExprType gives bool without looking up the type name", () => {
  const seen: string[] = [];
  const inferred = inferExprType(parseExpr("isinstance(d.output, Buy)"), (path) => {
    seen.push(path);
    return null;
  });
  expect(inferred).toEqual(Type.scalar(ValueKind.BOOLEAN));
  expect(seen).not.toContain("Buy");
});

test("isinstanceTypeNames lists each type name once, in source order", () => {
  expect(isinstanceTypeNames(parseExpr("isinstance(x, Buy) or isinstance(y, int)"))).toEqual(["Buy", "int"]);
  expect(isinstanceTypeNames(parseExpr("isinstance(x, Buy) and not isinstance(y.z, Buy)"))).toEqual(["Buy"]);
  expect(isinstanceTypeNames(parseExpr("x"))).toEqual([]);
});

test("an isinstance binding does not co-skip", () => {
  expect(bindingCoSkips("${isinstance(d.output, Buy)}")).toBe(false);
  expect(bindingCoSkips("${upper(d.output)}")).toBe(false);
});

// --- callee and prompt handling --- //

test("isinstance is not a flow callee", () => {
  expect(flowCallCalleesInSpans("${isinstance(x, Buy)}")).toEqual([]);
  expect(flowCallCalleesInSpans("${isinstance(x, Buy) and flow(a=1)}")).toEqual(["flow"]);
});

test("isinstance is not a prompt function", () => {
  const message = "isinstance is not a prompt function; compute it in an input binding";
  expectError(() => promptRefs("${isinstance(x, Buy)}"), message);
  expectError(() => promptRefs("Order: ${render_as_json(x)} ${upper(isinstance(x, Buy))}"), message);
});

// --- tracing --- //

test("traced isinstance records one route over the first argument", () => {
  const source = new SourceRef("run", 0, ["output"]);
  const resolve = () => ({ used: 2, present: true, value: A_BUY, why: null, source });
  const traced = evalBinding("${isinstance(d.output, Buy)}", resolve, { trace: true, types: TYPES });
  expect(traced.value).toBe(true);
  const route = traced.route;
  expect(route.kind).toBe("opaque");
  expect(route.expression).toBe("isinstance(d.output, Buy)");
  expect(route.children).toHaveLength(1);
  expect(route.names).toEqual([""]);
  const child = route.children[0]!;
  expect(child.kind).toBe("source");
  expect(child.source).toEqual(new SourceRef("run", 0, ["output"]));
});

// --- nodes --- //

const ORDER_FLOW = `
id: orders
name: orders
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', price: float}
  Order: Union(Buy, Sell)
input:
  order: !object
nodes:
  d:
    kind: code
    input:
      order: \${input.order}
    code: |
      return inputs["order"]
  check:
    kind: code
    input:
      ok: \${isinstance(d.output, Buy)}
    code: |
      return inputs["ok"]
  route:
    kind: case
    cases:
      - when: isinstance(d.output, Sell)
        then: s
    else: b
  s:
    kind: code
    code: |
      return "sell"
  b:
    kind: code
    code: |
      return "buy"
output:
  ok: \${check.output}
  side: \${s.output | b.output}
`;

/** `ORDER_FLOW` loaded, its `check` and `route` given the type table by hand. */
async function ordersFlow(): Promise<ComposedFlow> {
  const flow = await loadFlow(ORDER_FLOW);
  flow.compiled.nodes.get("check")!.types = new Map(TYPES);
  flow.compiled.nodes.get("route")!.types = new Map(TYPES);
  return flow;
}

test("a node starts with an empty type table", () => {
  const node = new CaseNode("route", [new Case("s", { when: "isinstance(order, Sell)" })]);
  expect(node.types).toEqual(new Map());
  expect(assembleText(ORDER_FLOW).flow.compiled.nodes.get("d")!.types).toEqual(new Map());
});

test.each([
  ["buy", A_BUY, { ok: true, side: "buy" }],
  ["sell", A_SELL, { ok: false, side: "sell" }],
])("bindings and case conditions use the node type table: %s", async (_name, order, expected) => {
  const result = await runFlow(await ordersFlow(), { order });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual(expected);
});

test("a node without the type name fails naming it", async () => {
  const flow = await ordersFlow();
  flow.compiled.nodes.get("check")!.types = new Map();
  const result = await runFlow(flow, { order: A_BUY });
  expect(result.status).toBe("failed");
  expect(result.failure?.code).toBe("binding.evaluate");
  expect(result.failure?.nodeId).toBe("check");
  expect(result.failure?.message).toContain("'Buy'");
});

test("case forward routes with its type table", () => {
  const node = new CaseNode("route", [new Case("s", { when: "isinstance(order, Sell)" })]);
  node.types = new Map(TYPES);
  expect(node.forward({ order: A_SELL }).handle).toBe("s");
  expect(node.forward({ order: A_BUY }).handle).toBe(ELSE_HANDLE);
});

const ASSERT_FLOW = `
id: asserted
name: asserted
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', price: float}
  Order: Union(Buy, Sell)
input:
  order: !object
nodes:
  step:
    kind: code
    input:
      order: \${input.order}
    code: |
      return inputs["order"]
    asserts:
      - isinstance(order, Buy)
      - isinstance(output, Order)
output: \${step.output}
`;

test.each([
  ["buy", A_BUY, "succeeded"],
  ["sell", A_SELL, "failed"],
] as const)("node asserts use the node type table: %s", async (_name, order, status) => {
  const flow = await loadFlow(ASSERT_FLOW);
  flow.compiled.nodes.get("step")!.types = new Map(TYPES);
  const result = await runFlow(flow, { order });
  expect(result.status, result.failure?.message).toBe(status);
  if (status === "failed") expect(result.failure?.code).toBe("assertion.failed");
});

const LOOP_FLOW = `
id: counter
name: counter
typedefs:
  Running: Literal(running)
defs:
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

const LOOP_TYPES = new Map([["Running", new Type({ kind: ValueKind.STRING, tags: new Set(["running"]) })]]);

/** `LOOP_FLOW` loaded, its loop driver given the type table by hand. */
async function loopFlow(): Promise<ComposedFlow> {
  const flow = await loadFlow(LOOP_FLOW);
  flow.compiled.nodes.get("loop")!.types = new Map(LOOP_TYPES);
  return flow;
}

test("loop respawn copies the type table", async () => {
  const driver = (await loopFlow()).compiled.nodes.get("loop") as LoopNode;
  expect(driver.respawn(1).types).toEqual(LOOP_TYPES);
});

test("loop while uses the type table across iterations", async () => {
  const result = await runFlow(await loopFlow(), {});
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ n: 4n, status: "done" });
});

/** Keep `log` up to `nodeId`'s success record; return the kept records' node ids. */
function cutLogAfter(log: string, nodeId: string): (string | undefined)[] {
  const lines = readFileSync(log, "utf8").split(/(?<=\n)/);
  const records = lines.map((line) => JSON.parse(line) as { kind: string; payload: { node_id?: string } });
  const end = records.findIndex((record) => record.kind === "node_succeeded" && record.payload.node_id === nodeId);
  expect(end).not.toBe(-1);
  writeFileSync(log, lines.slice(0, end + 1).join(""));
  return records.slice(0, end + 1).map((record) => record.payload.node_id);
}

test("a loop with a type table resumes mid-loop", async () => {
  const dir = mkdtempSync(join(tmpdir(), "isinstance-"));
  try {
    const log = join(dir, "run.jsonl");
    const first = await runFlow(await loopFlow(), {}, { log });
    expect(first.status, first.failure?.message).toBe("succeeded");
    // Cut after the second iteration, so resume replays two iterations and runs two live.
    const kept = cutLogAfter(log, "loop#1/__end__");
    expect(kept).toContain("loop~1");
    expect(kept.some((nodeId) => nodeId?.startsWith("loop~2") || nodeId?.startsWith("loop#2/"))).toBe(false);
    const resumed = await advance({ log });
    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
    expect(resumed.output).toEqual(first.output);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

const GUARDED_FLOW = `
id: guarded
name: guarded
typedefs:
  Buy: {action: 'Literal(buy)', qty: int, limit: float}
  Hold: {action: 'Literal(hold)', note: str}
  Decision: Union(Buy, Hold)
input:
  decision: !object
nodes:
  d:
    kind: code
    input:
      decision: \${input.decision}
    code: |
      return inputs["decision"]
    output: Decision
  route:
    kind: case
    cases:
      - when: isinstance(d.output, Buy) and d.output.limit > 5
        then: big
    else: other
  big:
    kind: code
    code: |
      return "big"
  other:
    kind: code
    code: |
      return "other"
output: \${big.output | other.output}
`;

test.each([
  ["a big buy", { action: "buy", qty: 1, limit: 9.0 }, "big"],
  ["a small buy", { action: "buy", qty: 1, limit: 2.0 }, "other"],
  ["a hold", { action: "hold", note: "wait" }, "other"],
])("a read guarded by isinstance routes a non-member to else: %s", async (_name, decision, expected) => {
  // The load check reads `d.output.limit` as a `Buy` field; on a `Hold` value the read gives
  // null and the `and` has already short-circuited, so the run routes to `else:`.
  const result = await runFlow(await loadFlow(GUARDED_FLOW), { decision });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(expected);
});

const NARROWED_DEFAULT_FLOW = `
id: narrowed-default
name: narrowed_default
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Hold: {action: 'Literal(hold)', note: str}
  Decision: Union(Buy, Hold)
input:
  decision: !object
nodes:
  d:
    kind: code
    input:
      decision: \${input.decision}
    code: |
      return inputs["decision"]
    output: Decision
  route:
    kind: case
    cases:
      - when: isinstance(d.output, Buy)
        then: default
    else: other
  default:
    kind: code
    input:
      qty: \${d.output.qty}
    code: |
      return inputs["qty"]
    output: int
  other:
    kind: code
    code: |
      return 0
    output: int
output: \${default.output | other.output}
`;

test("a then target named default is narrowed by a searched isinstance", async () => {
  // `default` is an ordinary node name, so its branch carries the `Buy` fact and may read the
  // `Buy`-only `qty`.
  const result = await runFlow(await loadFlow(NARROWED_DEFAULT_FLOW), { decision: { action: "buy", qty: 3 } });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(3n);
  expect(startedIds(result)).toContain("default");
  expect(startedIds(result)).not.toContain("other");
});
