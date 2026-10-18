/**
 * `isinstance(x, T)` in the expression language: the parse tests of
 * `tests/engine/test_isinstance.py`.
 *
 * `isinstance` is a special form: the first argument is an expression, the second a bare type
 * name, never a reference. The evaluation, reference and rewrite tests land with the evaluator.
 */

import { expect, test } from "bun:test";

import { ExpressionError } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";

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
