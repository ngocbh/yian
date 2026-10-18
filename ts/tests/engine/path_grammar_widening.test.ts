/**
 * References with the graph-expansion separators `#` and `/`: the `parseExpr` tests of
 * `tests/engine/test_path_grammar_widening.py`.
 *
 * The case-output regex test lands with compose's desugar.
 */

import { expect, test } from "bun:test";

import { ExpressionError } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";

test("namespaced path parses", () => {
  // A namespaced expansion reference like `${each#0/score.output.output}` parses.
  for (const text of ["each#0/score.output.output", "a/b.output.output", "each#0/score.output", "a/b.output"]) {
    expect(() => parseExpr(text)).not.toThrow();
  }
});

test("existing dotted paths still parse", () => {
  for (const text of ["research.output.report", "outputs.research.report"]) {
    expect(() => parseExpr(text)).not.toThrow();
  }
});

test("a genuinely bad path is still rejected", () => {
  let error: unknown;
  try {
    parseExpr("a..b"); // an empty segment
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toStartWith("could not parse expression");
});
