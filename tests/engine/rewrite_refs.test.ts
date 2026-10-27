/**
 * The position-splicing reference rewriters, the write-side analog of the `exprRefs` walk.
 */

import { expect, test } from "bun:test";

import { rensInternal } from "../../src/compile/expand.ts";
import {
  ExpressionError,
  type RefRenamed,
  rewriteConditionRefs,
  rewriteExprRefs,
} from "../../src/expr/expressions.ts";
import { rewriteTemplateRefs } from "../../src/expr/template.ts";
import { Binding, bindingSource } from "../../src/nodes/contracts.ts";

/** Upper-case the head (`a.b` -> `A.b`); keep `keep.*`. */
function bump(path: string): RefRenamed | null {
  if (path.startsWith("keep")) return null;
  const head = path.split(".")[0]!;
  return { text: head.toUpperCase(), used: 1 };
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
  expect(rewriteExprRefs("node#0.output", (p) => ({ text: "ns/" + p, used: p.split(".").length }))).toBe("ns/node#0.output");
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

test("a rename replaces only the names it used, keeping ?. after them", () => {
  const rename = (path: string) => (path.startsWith("a.") ? { text: "cs/a.output", used: 2 } : null);
  expect(rewriteExprRefs("a.output?.k.m + b", rename)).toBe("cs/a.output?.k.m + b");
  expect(rewriteTemplateRefs("x ${a.output?.k} y", rename)).toBe("x ${cs/a.output?.k} y");
  expect(rewriteConditionRefs("${a.output?.k > 1}", rename)).toBe("${cs/a.output?.k > 1}");
});

test("a rename that uses no name or more names than the read has is a bug", () => {
  expect(() => rewriteExprRefs("a.b", () => ({ text: "x", used: 0 }))).toThrow(RangeError);
  expect(() => rewriteExprRefs("a.b", () => ({ text: "x", used: 3 }))).toThrow(RangeError);
});

test("rensInternal keeps ?. after each renamed root", () => {
  const renamed = rensInternal(new Binding("${input.q?.k :- 1} ${p.output?.k}"), "cs");
  expect(bindingSource(renamed)).toBe("${cs/__start__.output.q?.k :- 1} ${cs/p.output?.k}");
});
