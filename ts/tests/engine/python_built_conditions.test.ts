/**
 * A CASE `when` set in code that doesn't parse is a setup error: a port of
 * `tests/engine/test_python_built_conditions.py`.
 *
 * Ported: `test_unparseable_case_when_raises_its_parse_error_from_setup` and
 * `test_validate_conditions_names_the_given_path`.
 *
 * Changed: `run` is async, so the setup error is a rejection.
 *
 * TS-only: a CASE whose `when` parses rejects from `run` as not ported yet.
 *
 * Later (T3-8, LOOP): `test_unparseable_loop_predicate_raises_before_the_first_body` and
 * `test_times_loop_has_no_predicate_to_check`.
 *
 * Later (T4-9): `test_case_when_evaluation_error_still_fails_in_forward`.
 *
 * Later (T4-8): `test_python_set_case_when_fails_before_the_run_starts` and
 * `test_python_set_loop_predicate_fails_before_the_run_starts`.
 */

import { expect, test } from "bun:test";

import { ExpressionError } from "../../src/expr/index.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";

/** Return the error `promise` rejects with; fail when it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

test("unparseable case when raises its parse error from setup", async () => {
  // `run` names the node, case, and field instead of failing inside `forward`.
  const node = new CaseNode("gate", [new Case("ok", { when: "x > 0" }), new Case("pos", { when: "${x} > 0" })]);

  const error = await rejection(node.run({ x: 1n }));

  expect(error).toBeInstanceOf(ExpressionError);
  const message = (error as Error).message;
  expect(message).toStartWith("node 'gate' case 'pos' `when:` '${x} > 0' is not a valid boolean expression: ");
  expect(message).toContain("write `x > 0`");
});

test("validate conditions names the given path", () => {
  // The pre-run check names a node by its static path.
  const node = new CaseNode("gate", [new Case("pos", { when: "${x} > 0" })]);

  expect(() => node.validateConditions({ path: "loop/gate" })).toThrow(ExpressionError);
  expect(() => node.validateConditions({ path: "loop/gate" })).toThrow(/^node 'loop\/gate' case 'pos' `when:`/);
});

test("a case whose when parses rejects from run as not ported yet", async () => {
  const node = new CaseNode("gate", [new Case("pos", { when: "x > 0" })]);

  await expect(node.run({ x: 1n })).rejects.toThrow("node 'gate' (CaseNode) forward is not ported yet (T4-9)");
});
