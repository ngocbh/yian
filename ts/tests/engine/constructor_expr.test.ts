/**
 * `T(...)` constructors in the expression language: the parse and evaluation tests of
 * `tests/engine/test_constructor_expr.py` (call shape, the positional and keyword forms, larger
 * expressions, names that are not constructors), its walker tests (`expr_refs`,
 * `rewrite_expr_refs`, `constructor_names`) and its static inference tests.
 * `test_a_builtin_still_mixes_positional_and_keyword_arguments` calls a `join` stand-in.
 *
 * Left for later: `test_a_constructor_in_an_embedded_template_span`, the co-skip, flow-callee
 * (`test_a_typedef_name_is_not_a_flow_callee`) and prompt tests land with `template.ts`; the
 * route and feedback tests with `lineage.ts`; the binder tests with `nodes/binding.ts`.
 */

import { afterEach, expect, test } from "bun:test";

import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import {
  ConstructorTypeError,
  constructorNames,
  ExpressionError,
  evalExpr,
  evaluateWhenRecord,
  exprRefs,
  inferExprType,
  parseCondition,
  rewriteExprRefs,
} from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { Type, typeFor, type Value, type ValueRecord, ValueKind } from "../../src/typesys/index.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { fieldOf } from "../../src/typesys/values.ts";

const REGISTRY = readTypedefs({
  Buy: { action: "Literal(buy)", conviction: "float" },
  Sell: { action: "Literal(sell)", price: "float" },
  Decision: "Union(Buy, Sell)",
  Size: "Literal(s, m, l)",
});
const TYPES = new Map(["Buy", "Sell", "Decision", "Size"].map((name) => [name, typeFor(name, REGISTRY)]));

const A_BUY: ValueRecord = { action: "buy", conviction: 0.5 };
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

/** Evaluate `text` against `values` with `TYPES`. */
function evaluate(text: string, values: ValueRecord = {}, types: ReadonlyMap<string, Type> = TYPES): Value {
  return evalExpr(parseExpr(text), resolver(values), { types });
}

/** Return the error `run` throws, failing if it returns. */
function errorOf(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Expect `run` to throw a `cls` whose message is exactly `message`. */
function expectError(run: () => unknown, message: string, cls: typeof ExpressionError = ExpressionError): void {
  const error = errorOf(run);
  expect(error).toBeInstanceOf(cls);
  expect(error.message).toBe(message);
}

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in builtin `name` taking `params` and running `call`, for this test only. */
function standIn(name: string, params: TemplateParam[], call: (...args: Value[]) => unknown = () => null): void {
  saved.push([name, TEMPLATE_FNS.get(name)]);
  registerTemplateFn(name, { params, call });
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

// --- call shape --- //

test("a constructor takes keywords or one value", () => {
  for (const text of ["Buy(a, b)", 'Buy(a, action="buy")', 'Buy(action="buy", a)']) {
    // The callee isn't known to be a constructor until it resolves, so each of these parses.
    expect(() => parseExpr(text)).not.toThrow();
    expectError(
      () => evaluate(text, { a: A_BUY, b: A_BUY }),
      "Buy(...) takes keyword fields, as in Buy(name=value, ...), or one value to check, as in Buy(x)",
    );
  }
});

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

test("a call repeating a keyword is rejected", () => {
  const cases: [string, string][] = [
    ["Buy(action=1, action=2)", "Buy(...) repeats the keyword 'action'"],
    ['join(xs, sep=",", sep=";")', "join(...) repeats the keyword 'sep'"],
  ];
  for (const [text, message] of cases) expectMessage(text, message);
});

test("a builtin still mixes positional and keyword arguments", () => {
  standIn("join", [{ name: "value" }, { name: "sep", default: "\n" }], (value, sep) =>
    (value as string[]).join(sep as string),
  );
  expect(evaluate('join(xs, sep="-")', { xs: ["a", "b"] })).toBe("a-b");
});

// --- positional form: check and narrow --- //

test("a value that fits is given back unchanged", () => {
  expect(evaluate("Buy(d.output)", { d: { output: A_BUY } })).toEqual(A_BUY);
});

test("an int checked as a float field stays an int", () => {
  const value = evaluate("Buy(d.output)", { d: { output: { action: "buy", conviction: 3n } } }) as ValueRecord;
  expect(value.conviction).toBe(3n);
});

test("a value missing a required field fails naming it", () => {
  expectError(
    () => evaluate("Buy(d.output)", { d: { output: { conviction: 0.5 } } }),
    "Buy(...): record missing required fields: ['action']",
    ConstructorTypeError,
  );
});

test("a missing value fails the check", () => {
  const error = errorOf(() => evaluate("Buy(d.output)"));
  expect(error).toBeInstanceOf(ConstructorTypeError);
  expect(error.message.startsWith("Buy(...): ")).toBe(true);
});

test("a union typedef checks against every member", () => {
  expect(evaluate("Decision(d.output)", { d: { output: A_SELL } })).toEqual(A_SELL);
  expect(errorOf(() => evaluate("Decision(d.output)", { d: { output: { action: "hold" } } }))).toBeInstanceOf(
    ConstructorTypeError,
  );
});

test("an enum alias checks a scalar", () => {
  expect(evaluate("Size(x)", { x: "m" })).toBe("m");
  const error = errorOf(() => evaluate("Size(x)", { x: "xl" }));
  expect(error).toBeInstanceOf(ConstructorTypeError);
  expect(error.message.startsWith("Size(...): ")).toBe(true);
});

test("a field read on the result", () => {
  expect(evaluate("Buy(d.output).conviction", { d: { output: A_BUY } })).toBe(0.5);
});

test("a constructor error is an expression error", () => {
  expect(new ConstructorTypeError("x")).toBeInstanceOf(ExpressionError);
});

// --- keyword form: build a record --- //

test("keywords build a record from exactly the given fields", () => {
  const value = evaluate('Buy(action="buy", conviction=s.output)', { s: { output: 0.7 } });
  expect(value).toEqual({ action: "buy", conviction: 0.7 });
});

test("a built record missing a field fails naming it", () => {
  expectError(
    () => evaluate("Buy(conviction=s.output)", { s: { output: 0.7 } }),
    "Buy(...): record missing required fields: ['action']",
    ConstructorTypeError,
  );
});

test("a built record with a wrong field fails", () => {
  const error = errorOf(() => evaluate('Buy(action="buy", conviction="high")'));
  expect(error).toBeInstanceOf(ConstructorTypeError);
  expect(error.message).toContain("does not match declared type float");
});

test("a keyword named __proto__ is a field, never the prototype", () => {
  const types = new Map([["Bag", new Type({ kind: ValueKind.OBJECT })]]);
  const value = evaluate("Bag(__proto__=1)", {}, types) as ValueRecord;
  expect(Object.hasOwn(value, "__proto__")).toBe(true);
  expect(Object.getPrototypeOf(value)).toBe(Object.prototype);
});

// --- inside larger expressions --- //

test("constructors in a list literal", () => {
  const value = evaluate("[Buy(a.output), Sell(b.output)]", { a: { output: A_BUY }, b: { output: A_SELL } });
  expect(value).toEqual([A_BUY, A_SELL]);
});

test("a constructor in a condition", () => {
  const record = { d: A_BUY };
  expect(evaluateWhenRecord("Buy(d).conviction > 0.1", record, TYPES)).toBe(true);
  expect(errorOf(() => evaluateWhenRecord("Sell(d).price > 0.1", record, TYPES))).toBeInstanceOf(
    ConstructorTypeError,
  );
});

// --- names that are not constructors --- //

/** Expect `parseCondition(text, TYPES)` to fail with exactly `message`. */
function expectConditionMessage(text: string, message: string): void {
  let error: unknown;
  try {
    parseCondition(text, TYPES);
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe(message);
}

test("a builtin type name is not a conversion", () => {
  for (const name of ["int", "str", "list", "Optional"]) {
    const message =
      `unknown expression function '${name}'; types in expressions are typedef names; ` +
      `\`${name}(...)\` is not a conversion`;
    expectConditionMessage(`${name}(x)`, message);
    // Even with the name in the type table, evaluation never treats it as a constructor.
    const types = new Map([...TYPES, [name, Type.scalar(ValueKind.INTEGER)]]);
    expectError(() => evaluate(`${name}(x)`, { x: 1n }, types), message);
  }
});

test("a name absent from the type table is an unknown function", () => {
  expectConditionMessage("Hold(x)", "unknown expression function 'Hold'");
  expectError(() => evaluate("Hold(x)", { x: 1n }), "unknown expression function 'Hold'");
});

test("reference walks read every argument and skip the names", () => {
  const tree = parseExpr('Buy(action="buy", conviction=s.output) == Buy(d.output)');
  expect(exprRefs(tree)).toEqual(["s.output", "d.output"]);
});

test("rewrites leave the type and field names alone", () => {
  const renamed = rewriteExprRefs("Buy(action=a, conviction=s.output).action", (path) => "x." + path);
  expect(renamed).toBe("Buy(action=x.a, conviction=x.s.output).action");
});

test("constructor names are the callees that are no builtin", () => {
  standIn("upper", [{ name: "s" }]);
  const tree = parseExpr("[Buy(d.output), upper(x), isinstance(y, Sell), Sell(price=p), Buy(z)]");
  expect(constructorNames(tree)).toEqual(["Buy", "Sell"]);
  expect(constructorNames(parseExpr("x"))).toEqual([]);
});

// --- static inference --- //

/** Return the type a constructor name names, else `null`. */
function typeFn(name: string): Type | null {
  return TYPES.get(name) ?? null;
}

/** Resolve no reference. */
function noRefs(): Type | null {
  return null;
}

test("inference gives a constructor its type", () => {
  expect(inferExprType(parseExpr("Buy(d.output)"), noRefs, typeFn)).toBe(TYPES.get("Buy")!);
  expect(inferExprType(parseExpr('Buy(action="buy", conviction=1)'), noRefs, typeFn)).toBe(TYPES.get("Buy")!);
});

test("inference reads fields of a constructor result", () => {
  expect(inferExprType(parseExpr("Buy(d.output).conviction"), noRefs, typeFn)).toEqual(
    Type.scalar(ValueKind.NUMBER),
  );
  expect(inferExprType(parseExpr("Buy(d.output).size"), noRefs, typeFn)).toBeNull();
});

test("inference reads a field every union member has", () => {
  const inferred = inferExprType(parseExpr("Decision(d.output).action"), noRefs, typeFn);
  expect(inferred).toEqual(new Type({ kind: ValueKind.STRING, tags: new Set(["buy", "sell"]) }));
});

test("inference without types leaves a call unknown", () => {
  expect(inferExprType(parseExpr("Buy(d.output)"), noRefs)).toBeNull();
  standIn("upper", [{ name: "s" }]);
  expect(inferExprType(parseExpr("upper(x)"), noRefs, typeFn)).toBeNull();
});
