/**
 * References with the graph-expansion separators `#` and `/`, a port of
 * `tests/engine/test_path_grammar_widening.py`.
 *
 * `test_case_output_interior_accepts_separators` checks the private case-output pattern through
 * `expandCaseOutputs`: a read matching it expands to the case's branch targets.
 */

import { expect, test } from "bun:test";

import { expandCaseOutputs } from "../../src/compose/desugar/case_outputs.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
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
  expect((error as Error).message).toBe("could not parse expression 'a..b': unexpected '.' at line 1, column 3");
});

test("case output interior accepts separators", () => {
  // The pattern accepts the node-first `<id>.output[.<seg>…]` shape with namespaced ids.
  for (const [caseId, read, expanded] of [
    ["each#0/leaf", "each#0/leaf.output", "${left.output | right.output}"],
    // A branch the route didn't take is skipped, so each is read through `?.`.
    ["research", "research.output.report", "${left.output?.report | right.output?.report}"],
  ] as const) {
    const gate = new CaseDescriptor({ id: caseId, cases: [{ when: "yes", then: "left" }], else_: "right" });
    const [, outputs] = expandCaseOutputs(new Map([[caseId, gate]]), `\${${read}}`, []);
    expect(outputs).toBe(expanded);
  }
});
