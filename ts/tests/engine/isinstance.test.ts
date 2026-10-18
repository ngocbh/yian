/**
 * `isinstance(x, T)` in the expression language: the evaluation and parse tests of
 * `tests/engine/test_isinstance.py`, the `expr_refs`/`condition_refs`/`rewrite_*` parts of
 * `test_reference_walks_skip_the_type_name` and `test_rewrites_leave_the_type_name_alone`,
 * `test_isinstance_subject_is_the_tested_expression`,
 * `test_infer_expr_type_gives_bool_without_looking_up_the_type_name`, and an
 * `isinstanceTypeNames` test.
 *
 * `isinstance` is a special form: the first argument is an expression, the second a bare type
 * name looked up in the type table (`types`, a `Map`), never a reference.
 *
 * Left for later: `test_eval_binding_and_eval_template_use_the_type_table`, the co-skip,
 * flow-callee and prompt-function tests and the template parts of the walk and rewrite tests
 * land with `template.ts`; the tracing test with `lineage.ts`; the node, case and loop tests
 * (from `test_a_node_starts_with_an_empty_type_table` on) with compose and `run.ts`.
 */

import { expect, test } from "bun:test";

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
  rewriteConditionRefs,
  rewriteExprRefs,
} from "../../src/expr/expressions.ts";
import { type ExprNode, parseExpr, Token } from "../../src/expr/grammar.ts";
import { Type, type Value, type ValueRecord, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import { fieldOf } from "../../src/typesys/values.ts";

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
function resolver(values: ValueRecord = {}): (path: string) => Value {
  return (path) => {
    const [head, ...parts] = path.split(".");
    let value = fieldOf(values, head!)[1];
    for (const part of parts) value = fieldOf(value, part)[1];
    return value;
  };
}

/** Evaluate `text` against `resolve` with the type table `types`. */
function evaluate(
  text: string,
  resolve: (path: string) => Value,
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
  expect(evaluate("isinstance(d.output, Buy)", resolver())).toBe(false);
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
  const tree = parseExpr("isinstance(d.output, Buy)");
  expectError(
    () => evalExpr(tree, resolver(), { mode: ResolveMode.STRICT_RAISE, types: TYPES }),
    "unresolved reference ${d.output} in expression",
  );
});

test("an Optional alias accepts a missing value", () => {
  const types = new Map([["MaybeBuy", BUY.replace({ nullable: true })]]);
  expect(evaluate("isinstance(d.output, MaybeBuy)", resolver(), types)).toBe(true);
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
});

test("rewrites leave the type name alone", () => {
  const rename = (path: string) => "x." + path;
  expect(rewriteExprRefs("isinstance(d.output, Buy)", rename)).toBe("isinstance(x.d.output, Buy)");
  expect(rewriteConditionRefs("${isinstance(d.output, Buy)}", rename)).toBe("${isinstance(x.d.output, Buy)}");
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
