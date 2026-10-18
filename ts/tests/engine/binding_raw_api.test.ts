/**
 * The raw-string binding API on the unified engine, a port of
 * `tests/engine/test_binding_raw_api.py`: `evalBinding(source, resolve, {item})` and
 * `exprRefsOf(source)`, which take a raw template and scan it themselves.
 *
 * These cover the binding semantics (a typed whole span, an embedded span written as text, `$$`,
 * coalesce `|`, default `:-`, required `:?`, a default reference `${x :- y}`, the `${item}`
 * scope) and full expressions inside a binding span (arithmetic, text and list operations). The
 * values follow the TS number model (ints are `bigint`s).
 *
 * Left for later: `test_capture_marks_actual_template_evaluation_in_materialized_seed` lands
 * with `nodes/binding.ts` and `lineage.ts`.
 */

import { expect, test } from "bun:test";

import { RequiredError } from "../../src/expr/expressions.ts";
import { evalBinding, exprRefsOf } from "../../src/expr/template.ts";
import type { Value, ValueRecord } from "../../src/typesys/index.ts";

/** Return a `resolve` over a flat record: a path it lacks is `null`, the binding miss. */
function from(values: ValueRecord): (path: string) => Value {
  return (path) => (Object.hasOwn(values, path) ? values[path]! : null);
}

// --- binding value semantics --- //

test("a whole span is typed", () => {
  expect(evalBinding("${a}", from({ a: 0.7 }))).toBe(0.7);
  expect(evalBinding("${a}", from({ a: ["x", "y"] }))).toEqual(["x", "y"]);
});

test("an embedded span is written as text", () => {
  expect(evalBinding("pe=${a}", from({ a: 0.7 }))).toBe("pe=0.7");
});

test("$$ is a literal $", () => {
  expect(evalBinding("cost is $$5", from({}))).toBe("cost is $5");
});

test("coalesce takes the first non-null", () => {
  expect(evalBinding("${a | b}", from({ b: "second" }))).toBe("second");
  // A present falsy value wins: only null falls through.
  expect(evalBinding("${a | b}", from({ a: 0n, b: "second" }))).toBe(0n);
});

test("default and required", () => {
  expect(evalBinding('${a :- "d"}', from({}))).toBe("d");
  expect(evalBinding("${a :- 30}", from({}))).toBe(30n);
  // The `:?` message is a full expression, so a plain-text message is quoted.
  expect(() => evalBinding('${a :? "need a topic"}', from({}))).toThrow(RequiredError);
  expect(() => evalBinding('${a :? "need a topic"}', from({}))).toThrow("need a topic");
});

test("a default reference resolves when the head misses", () => {
  expect(evalBinding("${a :- b}", from({ b: "fb" }))).toBe("fb");
});

test("item scope", () => {
  expect(evalBinding("${item.title}", from({}), { item: { title: "T" } })).toBe("T");
  expect(evalBinding("${item}", from({}), { item: "whole" })).toBe("whole");
});

// --- full expressions inside a binding span --- //

test("arithmetic in a binding", () => {
  expect(evalBinding("${x + 1}", from({ x: 2n }))).toBe(3n);
});

test("text and list operations in a binding", () => {
  expect(evalBinding("${a + b}", from({ a: "foo", b: "bar" }))).toBe("foobar");
  expect(evalBinding("${xs + [item]}", from({ xs: [1n, 2n] }), { item: 3n })).toEqual([1n, 2n, 3n]);
});

// --- exprRefsOf: the raw-string reference walk --- //

test("exprRefsOf collects span references in order", () => {
  expect(exprRefsOf("${a.output | b.output}")).toEqual(["a.output", "b.output"]);
});

test("exprRefsOf: literal text contributes nothing", () => {
  expect(exprRefsOf("just text, no spans")).toEqual([]);
});

test("exprRefsOf collects embedded, default and arithmetic references", () => {
  expect(exprRefsOf("n=${a :- b} m=${x + y}")).toEqual(["a", "b", "x", "y"]);
});

test("exprRefsOf collects item references", () => {
  // The caller skips them when minting edges.
  expect(exprRefsOf("${item.x}")).toEqual(["item.x"]);
});
