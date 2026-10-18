/**
 * `isinstance(x, T)` in the expression language: the parse tests of
 * `tests/engine/test_isinstance.py`, the `expr_refs`/`condition_refs`/`rewrite_*` parts of
 * `test_reference_walks_skip_the_type_name` and `test_rewrites_leave_the_type_name_alone`, and
 * `test_isinstance_subject_is_the_tested_expression`.
 *
 * `isinstance` is a special form: the first argument is an expression, the second a bare type
 * name, never a reference. The evaluation and inference tests land with the evaluator, and the
 * template parts of the walk and rewrite tests with `template.ts` (T2-5).
 */

import { expect, test } from "bun:test";

import {
  conditionRefs,
  ExpressionError,
  exprRefs,
  isinstanceSubject,
  rewriteConditionRefs,
  rewriteExprRefs,
} from "../../src/expr/expressions.ts";
import { type ExprNode, parseExpr, Token } from "../../src/expr/grammar.ts";

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
