/**
 * A CASE `when` or LOOP predicate set in code that doesn't parse is a setup error: a port of
 * `tests/engine/test_python_built_conditions.py`.
 *
 * Ported: `test_unparseable_case_when_raises_its_parse_error_from_setup`,
 * `test_unparseable_loop_predicate_raises_before_the_first_body`,
 * `test_times_loop_has_no_predicate_to_check` and `test_validate_conditions_names_the_given_path`.
 *
 * Ported (adapted): `test_case_when_evaluation_error_still_fails_in_forward` (the rejection's stack
 * names `forward`, where Python reads the traceback's frames).
 *
 * Changed: `run` is async, so the setup error is a rejection. The LOOP's child is a stub object.
 *
 * TS-only: LOOP's `validateConditions` names the given path.
 *
 * Later (T4-8): `test_python_set_case_when_fails_before_the_run_starts` and
 * `test_python_set_loop_predicate_fails_before_the_run_starts`.
 */

import { expect, test } from "bun:test";

import { ExpressionError } from "../../src/expr/index.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";

/** Return the error `promise` rejects with; fail when it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

test("case when evaluation error still fails in forward", async () => {
  // A `when` that parses but can't evaluate is unchanged: it rejects from `forward`.
  const node = new CaseNode("gate", [new Case("pos", { when: "x > 0" })]);

  const error = await rejection(node.run({ x: { a: 1n } }));

  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).not.toContain("is not a valid boolean expression");
  expect((error as Error).stack).toContain("forward");
});

test("unparseable case when raises its parse error from setup", async () => {
  // `run` names the node, case, and field instead of failing inside `forward`.
  const node = new CaseNode("gate", [new Case("ok", { when: "x > 0" }), new Case("pos", { when: "${x} > 0" })]);

  const error = await rejection(node.run({ x: 1n }));

  expect(error).toBeInstanceOf(ExpressionError);
  const message = (error as Error).message;
  expect(message).toStartWith("node 'gate' case 'pos' `when:` '${x} > 0' is not a valid boolean expression: ");
  expect(message).toContain("write `x > 0`");
});

for (const kind of ["while", "until"] as const) {
  test(`unparseable loop predicate raises before the first body (${kind})`, async () => {
    // Both predicate kinds are checked at setup, including `until` at iteration 0.
    const node = new LoopNode("loop", {
      flowId: "f",
      child: {},
      predicateKind: kind,
      predicate: "not ${exited}",
      maxIters: 10,
    });

    const error = await rejection(node.run({ exited: false }));

    expect(error).toBeInstanceOf(ExpressionError);
    const message = (error as Error).message;
    expect(message).toStartWith(`node 'loop' \`${kind}:\` 'not \${exited}' is not a valid boolean expression: `);
    expect(message).toContain("write `not exited`");
  });
}

test("times loop has no predicate to check", () => {
  // A `times` loop carries no predicate; setup accepts it.
  const node = new LoopNode("loop", {
    flowId: "f",
    child: {},
    predicateKind: "times",
    predicate: null,
    times: 2,
    maxIters: 2,
  });

  expect(() => node.validateConditions()).not.toThrow();
});

test("validate conditions names the given path", () => {
  // The pre-run check names a node by its static path.
  const node = new CaseNode("gate", [new Case("pos", { when: "${x} > 0" })]);

  expect(() => node.validateConditions({ path: "loop/gate" })).toThrow(ExpressionError);
  expect(() => node.validateConditions({ path: "loop/gate" })).toThrow(/^node 'loop\/gate' case 'pos' `when:`/);
});

test("loop validate conditions names the given path", () => {
  const node = new LoopNode("loop", {
    flowId: "f",
    child: {},
    predicateKind: "while",
    predicate: "not ${exited}",
    maxIters: 10,
  });

  expect(() => node.validateConditions({ path: "outer/loop" })).toThrow(ExpressionError);
  expect(() => node.validateConditions({ path: "outer/loop" })).toThrow(/^node 'outer\/loop' `while:`/);
});

