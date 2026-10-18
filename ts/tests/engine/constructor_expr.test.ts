/**
 * `T(...)` constructors in the expression language: the parse halves of
 * `tests/engine/test_constructor_expr.py`. The evaluation tests land with the evaluator.
 */

import { expect, test } from "bun:test";

import { ExpressionError } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";

test("a constructor takes keywords or one value", () => {
  // The callee isn't known to be a constructor until it resolves, so each of these parses.
  for (const text of ["Buy(a, b)", 'Buy(a, action="buy")', 'Buy(action="buy", a)']) {
    expect(() => parseExpr(text)).not.toThrow();
  }
});

test("a call repeating a keyword is rejected", () => {
  const cases: [string, string][] = [
    ["Buy(action=1, action=2)", "Buy(...) repeats the keyword 'action'"],
    ['join(xs, sep=",", sep=";")', "join(...) repeats the keyword 'sep'"],
  ];
  for (const [text, message] of cases) {
    expect(() => parseExpr(text)).toThrow(ExpressionError);
    expect(() => parseExpr(text)).toThrow(message);
  }
});
