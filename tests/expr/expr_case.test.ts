/**
 * Condition evaluation against a variable pool and a node's input record: references, `when:`
 * conditions with arithmetic, strings, lists and builtins, builtins in bindings, record-scoped
 * strict prompt rendering (`renderTemplateRecord`), and CASE routing through the engine.
 *
 * The values follow the TS number model: pool ints are `bigint`s, and a membership test against
 * `null` fails with the operand-kinds message. An embedded `false` is written `false`.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { NodeFailed } from "../../src/events.ts";
import {
  ExpressionError,
  evaluateWhen,
  evaluateWhenRecord,
  resolveInRecord,
  resolveReference,
} from "../../src/expr/expressions.ts";
import { evalBinding, renderTemplateRecord } from "../../src/expr/template.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { VariablePool } from "../../src/typesys/index.ts";
import { engine, FuncNode, graph, RecordNode } from "../helpers/run_state_fixtures.ts";

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

test("resolveReference reads a structured value's root; the evaluator walks its fields", () => {
  // `.output` is part of the root, not a stored key: `a.output.pe` reads store["a"], then `pe`.
  const pool = poolWith({ a: { pe: 21.0 } });
  const read = resolveReference(["a", "output", "pe"], pool);
  expect(read).toEqual({ used: 2, present: true, value: { pe: 21.0 }, why: null });
  expect(evaluateWhen("a.output.pe == 21.0", pool)).toBe(true);
});

test("resolveReference and resolveInRecord reject an empty name", () => {
  for (const names of [["a", "", "b"], ["a", ""], ["", "a"], [""]]) {
    const path = names.join(".");
    expectError(() => resolveReference(names, new VariablePool()), `invalid variable path: '\${${path}}'`);
    expectError(() => resolveInRecord(names, {}), `invalid variable path: '\${${path}}'`);
  }
});

test("resolveInRecord reads own fields only", () => {
  const read = resolveInRecord(["a", "b"], { a: { b: 1n } });
  expect(read).toEqual({ used: 1, present: true, value: { b: 1n }, why: null });
  expect(resolveInRecord(["toString"], {})).toEqual({
    used: 1,
    present: false,
    value: null,
    why: "input 'toString' is not bound",
  });
  expect(() => evaluateWhenRecord("a.constructor", { a: {} })).toThrow("a has no field 'constructor'");
  expect(evaluateWhenRecord("a?.constructor", { a: {} })).toBe(false);
  expect(() => evaluateWhenRecord("a.b == null", { a: null })).toThrow("a is null");
  expect(evaluateWhenRecord("a?.b == null", { a: null })).toBe(true);
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

test("builtins evaluate in conditions and bindings", () => {
  expect(evaluateWhenRecord('${upper(s) == "AB"}', { s: "ab" })).toBe(true);
  expect(evalBinding('${join(items, ",")}', (path) => resolveInRecord(path, { items: ["a", "b"] }))).toBe("a,b");
});

// --- record-scoped strict prompt rendering --- //

test("renderTemplateRecord substitutes declared inputs", () => {
  const record = { topic: "ZETA", rating: { value: 0.8 } };
  expect(renderTemplateRecord("Assess ${topic} v=${rating.value}", record)).toBe("Assess ZETA v=0.8");
});

test("renderTemplateRecord raises on an unknown or null input", () => {
  expect(() => renderTemplateRecord("hi ${ghost}", { topic: "X" })).toThrow(ExpressionError);
  // A present null raises: the strict floor.
  expect(() => renderTemplateRecord("hi ${t}", { t: null })).toThrow(ExpressionError);
});

test("renderTemplateRecord with no references is the identity", () => {
  expect(renderTemplateRecord("plain", {})).toBe("plain");
});

test("renderTemplateRecord renders falsy present values", () => {
  // 0, "" and false are present, not missing, so they render.
  const record = { w: 0n, s: { on: false }, name: "" };
  expect(renderTemplateRecord("w=${w} flag=${s.on} n='${name}'", record)).toBe("w=0 flag=false n=''");
});

// --- record-scoped CASE condition evaluation --- //

test("evaluateWhenRecord resolves declared inputs", () => {
  const record = { label: "POSITIVE", rating: { value: 0.8 } };
  expect(evaluateWhenRecord("label == 'POSITIVE'", record)).toBe(true);
  expect(evaluateWhenRecord("rating.value > 0.7", record)).toBe(true);
  expect(evaluateWhenRecord("label == 'NEGATIVE'", record)).toBe(false);
});

test("evaluateWhenRecord: a miss is falsy, not an error", () => {
  // A declared input holding null is falsy rather than an error; a field of it needs `?.`.
  expect(evaluateWhenRecord("label == 'POSITIVE'", { label: null })).toBe(false);
  expect(evaluateWhenRecord("rating?.value > 0.7", { rating: null })).toBe(false);
  expect(() => evaluateWhenRecord("rating.value > 0.7", { rating: null })).toThrow(
    "cannot read ${rating.value}: rating is null",
  );
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

// --- the CASE node --- //

/**
 * Return `START -> <name> -> cond`, `<name>` producing `value` and `cond` routing on it by bare
 * name: `is_high` to `high`, the else handle to `low`; each of those logs its id.
 */
function branchGraph(cases: readonly Case[], name = "score", value: unknown = 0.8) {
  const log: string[] = [];
  const cond = new CaseNode("cond", cases);
  cond.params = [new Parameter(name)];
  const flow = graph(
    [new FuncNode(name, () => value), cond, new RecordNode("high", log), new RecordNode("low", log)],
    [
      [START_ID, name],
      [name, "cond"],
      ["cond", "high", "is_high"],
      ["cond", "low", ELSE_HANDLE],
      ["high", END_ID],
      ["low", END_ID],
    ],
    { wiring: { [name]: {}, cond: { [name]: `\${${name}.output}` }, high: {}, low: {} } },
  );
  return { flow, log };
}

test("case when routes high", async () => {
  const { flow, log } = branchGraph([new Case("is_high", { when: "score >= 0.7" })]);
  await engine(flow).run();
  expect(log).toEqual(["high"]);
});

test("case when routes default", async () => {
  const { flow, log } = branchGraph([new Case("is_high", { when: "score >= 0.9" })]);
  await engine(flow).run();
  expect(log).toEqual(["low"]);
});

test("case string equality routes", async () => {
  // The canonical pattern: an upstream node writes a label; the router declares it as an input
  // and does a pure string comparison on the bare `label`.
  const { flow, log } = branchGraph([new Case("is_high", { when: "label == 'POSITIVE'" })], "label", "POSITIVE");
  await engine(flow).run();
  expect(log).toEqual(["high"]);
});

test("case case without when errors", async () => {
  // A validated compiled CASE always has a string predicate. Missing one is malformed setup, not
  // expected node execution rejection.
  const { flow } = branchGraph([new Case("is_high")]);
  await expect(engine(flow).run()).rejects.toThrow("has no `when` expression");
});

test("a failing predicate fails the run with condition.evaluate at its when: origin", async () => {
  const origin = new OriginRef("cond/when");
  const { flow, log } = branchGraph([new Case("is_high", { when: "score / 0 > 0", origin })]);

  const events = await engine(flow).run();

  const failed = events.filter((event) => event instanceof NodeFailed);
  expect(failed).toHaveLength(1);
  const { failure } = failed[0] as NodeFailed;
  expect(failure.code).toBe("condition.evaluate");
  expect(failure.causeType).toBe("ExpressionError");
  expect(failure.trace.at(-1)).toEqual(origin);
  expect(log).toEqual([]);
});
