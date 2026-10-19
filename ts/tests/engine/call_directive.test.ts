/**
 * The directive recognizer of `tests/engine/test_call_directive.py`, as direct rows.
 *
 * `matchCallDirective(value)` recognizes a whole-value `call(<flow-id>, kw=...)` directive and
 * returns the flow id with the raw argument text after its comma, `null` when the trimmed text
 * doesn't start with `call(`, or throws an `ExpressionError` naming the problem.
 *
 * Later (T3-17): every test, which drives `desugar_inline_calls`.
 *
 * Later (T3-22): the `load_flow` tests.
 *
 * TS-only: the recognizer rows those tests reach through `desugar_inline_calls`, each matching the
 * whole message, with the harvested
 * `load-types/compose_tagged_bindings__malformed_unquoted_call_after_a_tag` message, and direct
 * rows for `findParenEnd`, `splitCallsAware` and `InlineCall`.
 */

import { expect, test } from "bun:test";

import { CALL_HEAD, matchCallDirective } from "../../src/compose/helpers/directives.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import { findParenEnd, InlineCall, splitCallsAware } from "../../src/expr/template.ts";

test.each([
  ["call(summarize, messages=${messages})", ["summarize", " messages=${messages}"]],
  ["call(f)", ["f", ""]],
  ["  call (f, x=1)  ", ["f", " x=1"]],
  ["call(my-flow, a=1)", ["my-flow", " a=1"]],
  ["call(f, 30)", ["f", " 30"]],
] as const)("a whole-value directive %p is recognized", (value, expected) => {
  expect(matchCallDirective(value)).toEqual([...expected]);
});

test.each(["call-me(x)", "pre call(x)", "recall(x)", "plain text"])(
  "%p is not a directive",
  (value) => {
    expect(matchCallDirective(value)).toBeNull();
  },
);

test("CALL_HEAD anchors at the start", () => {
  expect(CALL_HEAD.test("call (f)")).toBe(true);
  expect(CALL_HEAD.test("pre call(f)")).toBe(false);
});

/** Assert `matchCallDirective(value)` throws an `ExpressionError` with exactly `message`. */
function expectThrows(value: string, message: string): void {
  let caught: unknown;
  try {
    matchCallDirective(value);
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(ExpressionError);
  expect((caught as ExpressionError).message).toBe(message);
}

const NEVER_CLOSED = "`call(` is never closed (check its parentheses and quotes)";
const notFirst = (first: string) =>
  `a call(...) must start with a flow id (a bare name) and a comma, got ${first}`;

test.each([
  ["call(f) tail", "text follows the closing `)`: 'tail'; a call(...) must be the whole value"],
  ["call(f, x=1", NEVER_CLOSED],
  ['call(f, x="a)', NEVER_CLOSED],
  ["call(f, x=${g)", NEVER_CLOSED],
  ["call(f x=1)", notFirst("'f x=1'")],
  ["call(f.g)", notFirst("'f.g'")],
  ["call()", notFirst("''")],
  [
    "call(f) and more",
    "text follows the closing `)`: 'and more'; a call(...) must be the whole value",
  ],
])("a malformed directive %p throws its message", (value, message) => {
  expectThrows(value, message);
});

// The outer call keeps a malformed nested call in its remainder; the error comes from the nested
// call when the desugarer recurses into it.
test("a nested call's trailing text is the nested call's error", () => {
  expect(matchCallDirective("call(f, x=call(g) tail)")).toEqual(["f", " x=call(g) tail"]);
  expectThrows(
    "call(g) tail",
    "text follows the closing `)`: 'tail'; a call(...) must be the whole value",
  );
});

test.each([
  ["call(f, x=${g(1)})", 17],
  ['call(f, x=")")', 13],
  ["call(f, x=call(g, y=1))", 22],
  ["call(f)", 6],
  ["call(f, x=${g(1)", null],
  ["call(f, x=${g)", null],
  ["call(f, x=(1)", null],
] as const)("findParenEnd(%p) is %p", (text, expected) => {
  expect(findParenEnd(text, 5)).toBe(expected);
});

test.each([
  [
    'a=1, b=f(1, 2), c="x,y", d=${h(1, 2)}',
    ",",
    ["a=1", " b=f(1, 2)", ' c="x,y"', " d=${h(1, 2)}"],
  ],
  ["a=${[1, 2]}, b=1", ",", ["a=${[1, 2]}", " b=1"]],
  ["x=a=b", "=", ["x", "a", "b"]],
  ["a), b", ",", ["a)", " b"]],
  ["c='a=b'", "=", ["c", "'a=b'"]],
  ["", ",", [""]],
] as const)("splitCallsAware(%p, %p) splits at top level only", (text, sep, expected) => {
  expect(splitCallsAware(text, sep)).toEqual([...expected]);
});

test("an InlineCall is frozen and keeps its fields", () => {
  const call = new InlineCall("__call_0", "f", { a: 1n });
  expect(Object.isFrozen(call)).toBe(true);
  expect(call.id).toBe("__call_0");
  expect(call.callee).toBe("f");
  expect(call.args).toEqual({ a: 1n });
});
