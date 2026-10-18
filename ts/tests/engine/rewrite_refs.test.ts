/**
 * The position-splicing reference rewriters, the write-side analog of the `exprRefs` walk, a port
 * of `tests/engine/test_rewrite_refs.py`.
 *
 * `rewriteExprRefs(text, rename)` reparses one unified `${...}` expression and splices each
 * reference-leaf path through `rename(path)` (a new path, or `null` to keep), leaving every
 * operator, literal, builtin callee and whitespace verbatim. `rewriteConditionRefs` wraps it for
 * the two condition spellings (bare and whole-span); `rewriteTemplateRefs` wraps it for a
 * multi-span binding or template value.
 */

import { expect, test } from "bun:test";

import { ExpressionError, rewriteConditionRefs, rewriteExprRefs } from "../../src/expr/expressions.ts";
import { rewriteTemplateRefs } from "../../src/expr/template.ts";

/** Upper-case the head segment (`a.b` -> `A.b`); keep `keep.*`. */
function bump(path: string): string | null {
  if (path.startsWith("keep")) return null;
  const dot = path.indexOf(".");
  return dot < 0 ? path.toUpperCase() : path.slice(0, dot).toUpperCase() + path.slice(dot);
}

// --- rewriteExprRefs: leaves spliced, everything else verbatim --- //

test("rewrite a bare refcall leaf", () => {
  expect(rewriteExprRefs("a > 5", bump)).toBe("A > 5");
});

test("rewrite a dotted refcall, keeping its tail", () => {
  expect(rewriteExprRefs("a.b.c >= foo.output", bump)).toBe("A.b.c >= FOO.output");
});

test("rewrite leaves operators and literals verbatim", () => {
  expect(rewriteExprRefs('a in ["x", "y"] and b < 3', bump)).toBe('A in ["x", "y"] and B < 3');
});

test("rewrite to null keeps the reference", () => {
  expect(rewriteExprRefs("keep.x > a", bump)).toBe("keep.x > A");
});

test("rewrite leaves a builtin callee but renames its args", () => {
  expect(rewriteExprRefs("upper(name) == foo", bump)).toBe("upper(NAME) == FOO");
});

test("rewrite a default's right-hand side", () => {
  expect(rewriteExprRefs("x :- y", bump)).toBe("X :- Y");
});

test("rewrite hash and slash segments", () => {
  expect(rewriteExprRefs("node#0.output", (p) => "ns/" + p)).toBe("ns/node#0.output");
});

// --- rewriteConditionRefs: both spellings rewrite identically --- //

test("condition, bare", () => {
  expect(rewriteConditionRefs("a > 5", bump)).toBe("A > 5");
});

test("condition with a nested span is rejected", () => {
  expect(() => rewriteConditionRefs("${a} > 5", bump)).toThrow(ExpressionError);
  expect(() => rewriteConditionRefs("${a} > 5", bump)).toThrow("write `a > 5`");
});

test("condition, whole span keeps the outer braces", () => {
  expect(rewriteConditionRefs("${a > 5}", bump)).toBe("${A > 5}");
});

test("condition, whole span with several refs", () => {
  expect(rewriteConditionRefs("${a > 5 and b < 3}", bump)).toBe("${A > 5 and B < 3}");
});

// --- rewriteTemplateRefs: spans rewritten, literal text and $$ kept --- //

test("template, whole span", () => {
  expect(rewriteTemplateRefs("${a.output}", bump)).toBe("${A.output}");
});

test("template, embedded text kept", () => {
  expect(rewriteTemplateRefs("pre ${a} mid ${b.x} post", bump)).toBe("pre ${A} mid ${B.x} post");
});

test("template, the dollar escape kept", () => {
  // `$$` is a literal `$`: it must survive verbatim (not be read as a span start).
  expect(rewriteTemplateRefs("cost is $$${a}", bump)).toBe("cost is $$${A}");
});

test("template with no span comes back unchanged", () => {
  expect(rewriteTemplateRefs("plain text", bump)).toBe("plain text");
});

test("a template value that isn't text passes through", () => {
  expect(rewriteTemplateRefs(5, bump)).toBe(5);
});

test("template, default span", () => {
  expect(rewriteTemplateRefs("${x :- y}", bump)).toBe("${X :- Y}");
});
