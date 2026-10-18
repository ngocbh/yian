/**
 * Condition evaluation against a variable pool and a node's input record: the expression tests
 * of `tests/engine/test_expr_case.py` (`resolve_reference`, `evaluate_when`,
 * `evaluate_when_record`, `first_failing_assert`, and conditions with arithmetic, strings, lists
 * and builtins).
 *
 * The values follow the TS number model: pool ints are `bigint`s, and a membership test against
 * `null` fails with the operand-kinds message. The builtin test calls an `upper` stand-in.
 *
 * Left for later: the `render_template_record` tests and the `eval_binding` half of
 * `test_builtins_evaluate_in_conditions_and_bindings` land with `template.ts`; the CASE node
 * tests with the CASE node and the runtime.
 */

import { afterEach, expect, test } from "bun:test";

import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import {
  ExpressionError,
  evaluateWhen,
  evaluateWhenRecord,
  firstFailingAssert,
  resolveInRecord,
  resolveReference,
} from "../../src/expr/expressions.ts";
import { type Value, VariablePool } from "../../src/typesys/index.ts";

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in builtin `name` taking `params` and running `call`, for this test only. */
function standIn(name: string, params: TemplateParam[], call: (...args: Value[]) => unknown): void {
  saved.push([name, TEMPLATE_FNS.get(name)]);
  registerTemplateFn(name, { params, call });
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

/** Return a pool holding each node's value; `${<id>.output}` reads it. */
function poolWith(outputs: Record<string, unknown>): VariablePool {
  const pool = new VariablePool();
  // `.output` is a syntactic discriminator the resolver skips, so store each value directly.
  for (const [nodeId, value] of Object.entries(outputs)) pool.set(nodeId, value);
  return pool;
}

/** Expect `run` to throw an `ExpressionError` whose message is exactly `message`. */
function expectError(run: () => unknown, message: string): void {
  let error: unknown;
  try {
    run();
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe(message);
}

// --- expression evaluation --- //

test("resolveReference walks a structured value", () => {
  // `.output` is skipped, not a stored key: `a.output.pe` reads store["a"]["pe"].
  const pool = poolWith({ a: { pe: 21.0 } });
  expect(resolveReference("a.output.pe", pool)).toBe(21.0);
  expect(resolveReference("a.output", pool)).toEqual({ pe: 21.0 });
});

test("resolveReference and resolveInRecord reject an empty path segment", () => {
  for (const path of ["a..b", "a.", ".a", ""]) {
    expectError(() => resolveReference(path, new VariablePool()), `invalid variable path: '\${${path}}'`);
    expectError(() => resolveInRecord(path, {}), `invalid variable path: '\${${path}}'`);
  }
});

test("resolveInRecord reads own fields only, segments trimmed", () => {
  expect(resolveInRecord(" a . b ", { a: { b: 1n } })).toBe(1n);
  expect(resolveInRecord("toString", {})).toBeNull();
  expect(resolveInRecord("a.constructor", { a: {} })).toBeNull();
  expect(resolveInRecord("a.b", { a: null })).toBeNull();
});

test("when: numeric and boolean logic", () => {
  const pool = poolWith({ score: 0.8 });
  expect(evaluateWhen("score.output > 0.5", pool)).toBe(true);
  expect(evaluateWhen("score.output > 0.5 and score.output < 1", pool)).toBe(true);
  expect(evaluateWhen("score.output > 0.9 or score.output == 0.8", pool)).toBe(true);
  expect(evaluateWhen("not score.output > 0.9", pool)).toBe(true);
});

test("when: strings and in", () => {
  const pool = poolWith({ label: "approve" });
  expect(evaluateWhen("label.output == 'approve'", pool)).toBe(true);
  expect(evaluateWhen("label.output in 'approve/reject'", pool)).toBe(true);
});

test("when: a missing reference is falsy, not an error", () => {
  expect(evaluateWhen("ghost.output > 5", new VariablePool())).toBe(false);
});

test("when: a bare reference is a truthiness test", () => {
  expect(evaluateWhen("${x.output}", poolWith({ x: 1n }))).toBe(true);
  expect(evaluateWhen("${x.output}", poolWith({ x: 0n }))).toBe(false);
  expect(evaluateWhen("${missing.output}", new VariablePool())).toBe(false);
});

// --- the when: spellings evaluate identically --- //

test("when: both spellings evaluate identically", () => {
  const pool = poolWith({ a: 8n });
  expect(evaluateWhen("a.output > 5", pool)).toBe(true);
  expect(evaluateWhen("${a.output > 5}", pool)).toBe(true);
  const low = poolWith({ a: 3n });
  expect(evaluateWhen("a.output > 5", low)).toBe(false);
  expect(evaluateWhen("${a.output > 5}", low)).toBe(false);
});

test("when: braces on a reference inside the condition are rejected", () => {
  expectError(
    () => evaluateWhen("${a.output} > 5", poolWith({ a: 8n })),
    "`${...}` can't appear inside an expression; write `a.output > 5`",
  );
});

test("when: a bare arithmetic condition", () => {
  const pool = poolWith({ a: 5n });
  expect(evaluateWhen("a.output + 1 > 5", pool)).toBe(true);
  expect(evaluateWhen("a.output + 1 > 10", pool)).toBe(false);
});

test("when: record conditions in both spellings agree", () => {
  const record = { a: 8n };
  expect(evaluateWhenRecord("a > 5", record)).toBe(true);
  expect(evaluateWhenRecord("${a > 5}", record)).toBe(true);
});

test("firstFailingAssert in both spellings", () => {
  const pool = poolWith({ a: 8n });
  // All hold: null.
  expect(firstFailingAssert(["a.output > 5", "${a.output > 5}"], pool)).toBeNull();
  // The failing one comes back verbatim, and both spellings agree.
  expect(firstFailingAssert(["a.output < 5"], pool)).toBe("a.output < 5");
  expect(firstFailingAssert(["${a.output < 5}"], pool)).toBe("${a.output < 5}");
  expect(firstFailingAssert(["a.output > 5", "a.output > 9", "a.output > 10"], pool)).toBe("a.output > 9");
});

// --- arithmetic in when:/asserts: --- //

test("when: arithmetic and precedence", () => {
  const pool = poolWith({ a: 3n, b: 2n });
  expect(evaluateWhen("a.output + b.output == 5", pool)).toBe(true);
  // `*` before `+`
  expect(evaluateWhen("a.output + b.output * 2 == 7", pool)).toBe(true);
  // parentheses override
  expect(evaluateWhen("(a.output + b.output) * 2 == 10", pool)).toBe(true);
});

test("when: division, remainder and unary minus", () => {
  const pool = poolWith({ n: 6n, d: 4n });
  expect(evaluateWhen("n.output / 3 == 2", pool)).toBe(true);
  expect(evaluateWhen("d.output % 2 == 0", pool)).toBe(true);
  expect(evaluateWhen("-n.output < 0", pool)).toBe(true);
  // arithmetic on both sides of the comparison
  expect(evaluateWhen("n.output - 1 >= d.output + 1", pool)).toBe(true);
  // `/` always gives a float, and `%` takes the sign of the left
  expect(evaluateWhen("7 / 2 == 3.5", pool)).toBe(true);
  expect(evaluateWhen("-7 % 2 == -1", pool)).toBe(true);
});

test("when: arithmetic over text raises", () => {
  expectError(
    () => evaluateWhen("s.output + 1 == 2", poolWith({ s: "x" })),
    'type error evaluating "x" + 1: unsupported operand kinds for `+`: `str` and `int`',
  );
});

test("when: arithmetic in a record condition", () => {
  // The same grammar serves the strict-CASE record path.
  expect(evaluateWhenRecord("a * b >= 10", { a: 4n, b: 3n })).toBe(true);
  expect(evaluateWhenRecord("a % 2 == 1", { a: 5n })).toBe(true);
});

test("when: string equality and membership", () => {
  const pool = poolWith({ label: "approve" });
  expect(evaluateWhen("label.output == 'approve'", pool)).toBe(true);
  expect(evaluateWhen("label.output in 'approve/reject'", pool)).toBe(true);
});

// --- list literals in when:/asserts: --- //

test("when: list literal membership and equality", () => {
  const pool = poolWith({ style: "relevance", bundle: ["ACME", "BETA"], empty: [] });
  expect(evaluateWhen('style.output in ["relevance", "value"]', pool)).toBe(true);
  expect(evaluateWhen('style.output not in ["value", "quality"]', pool)).toBe(true);
  expect(evaluateWhen("bundle.output != []", pool)).toBe(true);
  expect(evaluateWhen("empty.output == []", pool)).toBe(true);
  // numeric and nested arithmetic elements resolve as values too
  expect(evaluateWhen("2 in [1, 2, 3]", pool)).toBe(true);
  expect(evaluateWhen("5 in [2 + 3, 10]", pool)).toBe(true);
});

test("a condition treats a missing reference as falsy", () => {
  expect(evaluateWhen("ghost.output > 5", new VariablePool())).toBe(false);
});

test("a record condition compares a miss as null", () => {
  // A miss is null compared as a plain value: `==` is false and `!=` is true.
  expect(evaluateWhenRecord('x == "done"', {})).toBe(false);
  expect(evaluateWhenRecord('x != "done"', {})).toBe(true);
  expect(evaluateWhenRecord('x != "done"', { x: null })).toBe(true);
  expect(evaluateWhenRecord("${x}", {})).toBe(false);
  // A null left side of `in` is just a value that isn't in the container; only a null right
  // side (the container itself) raises.
  expect(evaluateWhenRecord('x in ["a"]', {})).toBe(false);
  expectError(
    () => evaluateWhenRecord('"a" in x', {}),
    'type error evaluating "a" in null: unsupported operand kinds for `in`: `str` and `null`',
  );
});

test("builtins evaluate in conditions", () => {
  standIn("upper", [{ name: "s" }], (s) => (s as string).toUpperCase());
  expect(evaluateWhenRecord('${upper(s) == "AB"}', { s: "ab" })).toBe(true);
});

// --- record-scoped CASE condition evaluation --- //

test("evaluateWhenRecord resolves declared inputs", () => {
  const record = { label: "POSITIVE", rating: { value: 0.8 } };
  expect(evaluateWhenRecord("label == 'POSITIVE'", record)).toBe(true);
  expect(evaluateWhenRecord("rating.value > 0.7", record)).toBe(true);
  expect(evaluateWhenRecord("label == 'NEGATIVE'", record)).toBe(false);
});

test("evaluateWhenRecord: a miss is falsy, not an error", () => {
  // A declared input holding null, or a dotted miss, is falsy rather than an error.
  expect(evaluateWhenRecord("label == 'POSITIVE'", { label: null })).toBe(false);
  expect(evaluateWhenRecord("rating.value > 0.7", { rating: null })).toBe(false);
});

test("evaluateWhenRecord: in with a null operand raises", () => {
  // Membership with null fails the node, while equality and ordered comparisons stay falsy.
  expectError(
    () => evaluateWhenRecord("'x' in opts", { opts: null }),
    'type error evaluating "x" in null: unsupported operand kinds for `in`: `str` and `null`',
  );
  expectError(
    () => evaluateWhenRecord("x in 'abc'", {}),
    'type error evaluating null in "abc": unsupported operand kinds for `in`: `null` and `str`',
  );
  expectError(
    () => evaluateWhenRecord("x not in 'abc'", {}),
    'type error evaluating null not in "abc": unsupported operand kinds for `not in`: `null` and `str`',
  );
});
