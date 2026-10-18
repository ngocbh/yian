/**
 * A `${...}` marks an expression inside a YAML string; it can't appear inside one: the
 * `parseExpr` and condition tests of `tests/engine/test_nested_span_rejected.py`, which check
 * that a whole-span condition reads as its bare spelling, and the error a `${...}` nested inside
 * an expression raises and the rewrite it shows.
 *
 * Left for later: the template-span tests land with `template.ts`; the binding, case, assert,
 * loop and prompt load tests and the run tests with compose and `run.ts`.
 */

import { expect, test } from "bun:test";

import {
  conditionRefs,
  ExpressionError,
  parseCondition,
  rewriteConditionRefs,
} from "../../src/expr/expressions.ts";
import { type ExprNode, parseExpr, rejectNestedSpans, Token } from "../../src/expr/grammar.ts";

const NESTED = "`${...}` can't appear inside an expression";

/** A position-blind form of `node`, as Lark's `Tree`/`Token` equality ignores positions. */
function plain(node: ExprNode): unknown {
  if (node instanceof Token) return { type: node.type, value: node.value };
  return { data: node.data, children: node.children.map(plain) };
}

// --- whole-span conditions: one `${...}` around the whole value --- //

test("whole-span condition matches its bare spelling", () => {
  expect(plain(parseCondition("${a > 5}"))).toEqual(plain(parseCondition("a > 5")));
});

test("whole-span condition with a brace in a string literal", () => {
  expect(conditionRefs('${x == "}"}')).toEqual(["x"]);
});

test("whole-span rewrite with a brace in a string literal", () => {
  const rename = (path: string) => (path === "x" ? "y" : null);
  expect(rewriteConditionRefs('${x == "}"}', rename)).toBe('${y == "}"}');
});

// --- a `${...}` inside an expression is an error that shows the rewrite --- //

/** Return the error `fn` throws, which must be an `ExpressionError`. */
function caught(fn: () => unknown): Error {
  let error: unknown;
  try {
    fn();
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  return error as Error;
}

/** Expect `parseExpr(text)` to fail with the nested-span error and no rewrite. */
function expectNoRewrite(text: string): void {
  expect(caught(() => parseExpr(text)).message).toBe(NESTED);
}

test("nested span shows the rewrite", () => {
  const cases: [string, string][] = [
    ["input.a :- ${input.b}", "input.a :- input.b"],
    ["${a.output} > 5", "a.output > 5"],
    ["${a.output:-0} >= 1", "(a.output:-0) >= 1"],
    ["x :- ${ y }", "x :- y"],
    ["x :- ${y + 1}", "x :- (y + 1)"],
    ["a :- ${b :- ${c}}", "a :- (b :- c)"],
    ["isinstance(x, ${Buy})", "isinstance(x, Buy)"],
    ["render_as_json(${briefs}, 4)", "render_as_json(briefs, 4)"],
    ["not${x}", "not x"],
    ["x :- ${and}", "x :- and"],
  ];
  for (const [text, rewrite] of cases) {
    expect(caught(() => parseExpr(text)).message).toBe(`${NESTED}; write \`${rewrite}\``);
  }
});

test("nested span without a parsable rewrite shows none", () => {
  for (const text of ["a :- ${b", "a :- ${b +}", "${}"]) expectNoRewrite(text);
});

test("a span inside a string literal is text", () => {
  for (const text of ['x == "${y}"', "x == '${y}'"]) expect(() => parseExpr(text)).not.toThrow();
});

test("a whole-span condition holding a nested span shows the bare rewrite", () => {
  expect(caught(() => parseCondition("${a > ${b}}")).message).toBe(`${NESTED}; write \`a > b\``);
});

test("deeply nested spans are rejected without a rewrite", () => {
  expectNoRewrite("${".repeat(3000) + "x" + "}".repeat(3000));
});

test("a rewrite past the nesting limit is not suggested", () => {
  // The rewrite `not (not ... true)` would be 101 levels deep, so it would not parse.
  expectNoRewrite("not ${" + "not ".repeat(100) + "true}");
});

test("a rewrite at the nesting limit is suggested", () => {
  expect(caught(() => parseExpr("not ${" + "not ".repeat(99) + "true}")).message).toBe(
    `${NESTED}; write \`not (${"not ".repeat(99)}true)\``,
  );
});

test("a span interior's rewrite is the whole fixed span", () => {
  expect(caught(() => rejectNestedSpans("a :- ${b}", { inSpan: true })).message).toBe(
    `${NESTED}; write \`\${a :- b}\``,
  );
});

test("the rewrite only needs to parse, not pass the later checks", () => {
  expect(caught(() => rejectNestedSpans("${a.b(x)}")).message).toBe(`${NESTED}; write \`(a.b(x))\``);
  expect(caught(() => rejectNestedSpans("${isinstance(x)}")).message).toBe(
    `${NESTED}; write \`(isinstance(x))\``,
  );
});
