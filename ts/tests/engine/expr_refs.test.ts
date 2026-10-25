/**
 * The one compile-time reference walk (`exprRefs`), a port of `tests/engine/test_expr_refs.py`.
 *
 * `exprRefs(parseExpr(text))` collects every reference-leaf path a unified `${...}` expression
 * reads, in source order (no dedupe). A `refcall` without a `call_suffix` is a reference (its
 * dotted path is collected); one with a `call_suffix` is a call: the callee name contributes
 * nothing, but each argument value's references are collected. Literals contribute nothing.
 * `item`-headed references are collected here; the caller skips `item` when minting edges.
 *
 * TS-only: comprehension variables are not reads, and `rewriteExprRefs` leaves them alone.
 */

import { expect, test } from "bun:test";

import { exprRefs, referenceText, rewriteExprRefs, splitReference } from "../../src/expr/expressions.ts";
import { parseExpr, Tree } from "../../src/expr/grammar.ts";

/** Return the reference paths `parseExpr(text)` reads. */
function refs(text: string): string[] {
  return exprRefs(parseExpr(text));
}

test("two heads in arithmetic", () => {
  expect(refs("node_a.x + node_b.y")).toEqual(["node_a.x", "node_b.y"]);
});

test("item-headed ref is collected", () => {
  // The `call(...)` `${item}` guard needs to see them; the caller skips `item` when minting edges.
  expect(refs("item.x")).toEqual(["item.x"]);
});

test("hash and slash segments preserved", () => {
  expect(refs("node#0.output")).toEqual(["node#0.output"]);
  expect(refs("def/child.output")).toEqual(["def/child.output"]);
});

test("builtin call collects arg refs, not the callee or a literal", () => {
  expect(refs('join(items, ", ")')).toEqual(["items"]);
});

test("default collects the head and the default ref", () => {
  expect(refs("a :- b")).toEqual(["a", "b"]);
});

test("coalesce collects all operands", () => {
  expect(refs("x | y | z")).toEqual(["x", "y", "z"]);
});

test("required collects the head", () => {
  // The string message contributes nothing.
  expect(refs('p :? "msg"')).toEqual(["p"]);
});

test("call with only literal args has no refs", () => {
  expect(refs('upper("hi")')).toEqual([]);
});

test("ref inside a list literal", () => {
  expect(refs("[a, b]")).toEqual(["a", "b"]);
});

test("ref inside parens", () => {
  expect(refs("(a + b)")).toEqual(["a", "b"]);
});

test("nested call collects the inner ref", () => {
  expect(refs("join(upper(x))")).toEqual(["x"]);
});

test("comprehension variables are not reads; the iterable and outer names are", () => {
  expect(refs("[m.text for t in input.history for m in t if m.role == r]")).toEqual(["input.history", "r"]);
});

test("rewriteExprRefs leaves comprehension variables alone", () => {
  expect(rewriteExprRefs("[x + y for x in xs]", (p) => (p === "x" || p === "xs" ? { text: "z", used: 1 } : null))).toBe(
    "[x + y for x in z]",
  );
});

test("a ?. read is the same flat reference as a . read", () => {
  expect(exprRefs(parseExpr("a.output?.k.m | b?.c"))).toEqual(["a.output.k.m", "b.c"]);
});

test("splitReference keeps each step's ?.", () => {
  const read = parseExpr("a.output?.k") as Tree;
  expect(splitReference(read)).toEqual({
    head: "a",
    steps: [
      { name: "output", optional: false },
      { name: "k", optional: true },
    ],
  });
  expect(referenceText(read)).toBe("a.output?.k");
});
