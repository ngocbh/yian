/**
 * `T(...)` constructors in the expression language: the parse halves of
 * `tests/engine/test_constructor_expr.py`, its walker tests (`expr_refs`, `rewrite_expr_refs`,
 * `constructor_names`), and the `parse_condition` halves of
 * `test_a_name_absent_from_the_type_table_is_an_unknown_function` and
 * `test_a_builtin_type_name_is_not_a_conversion`.
 *
 * Left for later: the evaluation tests (call shape, positional and keyword forms, larger
 * expressions, the `eval_expr` halves of the two tests above) and the static inference tests land
 * with the evaluator; the route and feedback tests with `lineage.ts`; the embedded-span, co-skip,
 * flow-callee (`test_a_typedef_name_is_not_a_flow_callee`) and prompt tests with `template.ts`;
 * the binder tests with `nodes/binding.ts`.
 */

import { afterEach, expect, test } from "bun:test";

import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn } from "../../src/expr/builtins.ts";
import {
  constructorNames,
  ExpressionError,
  exprRefs,
  parseCondition,
  rewriteExprRefs,
} from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { typeFor } from "../../src/typesys/index.ts";
import { readTypedefs } from "../../src/typesys/types.ts";

test("a constructor takes keywords or one value", () => {
  // The callee isn't known to be a constructor until it resolves, so each of these parses.
  for (const text of ["Buy(a, b)", 'Buy(a, action="buy")', 'Buy(action="buy", a)']) {
    expect(() => parseExpr(text)).not.toThrow();
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

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in for the shipped `upper(s)` for this test only. */
function upperStandIn(): void {
  saved.push(["upper", TEMPLATE_FNS.get("upper")]);
  registerTemplateFn("upper", { params: [{ name: "s" }], call: () => null });
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

const REGISTRY = readTypedefs({ Buy: { conviction: "float" }, Sell: { price: "float" } });
const TYPES = new Map(["Buy", "Sell"].map((name) => [name, typeFor(name, REGISTRY)]));

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
    expectConditionMessage(
      `${name}(x)`,
      `unknown expression function '${name}'; types in expressions are typedef names; \`${name}(...)\` is not a conversion`,
    );
  }
});

test("a name absent from the type table is an unknown function", () => {
  expectConditionMessage("Hold(x)", "unknown expression function 'Hold'");
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
  upperStandIn();
  const tree = parseExpr("[Buy(d.output), upper(x), isinstance(y, Sell), Sell(price=p), Buy(z)]");
  expect(constructorNames(tree)).toEqual(["Buy", "Sell"]);
  expect(constructorNames(parseExpr("x"))).toEqual([]);
});
