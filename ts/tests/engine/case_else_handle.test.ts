/**
 * CASE's else route has its own handle: the node-level and reducer rows of
 * `tests/engine/test_case_else_handle.py`.
 *
 * Ported: `test_a_case_may_not_take_the_else_handle_as_a_branch`,
 * `test_a_route_to_an_unknown_handle_named_default_is_rejected`.
 *
 * Changed: `Case` is frozen, so the node-level half can't set `handle` after construction as
 * Python does; it builds the `Case` holding `#else` without the constructor. `RangeError` stands
 * for Python's `ValueError`. The unknown-handle rejection is node data: a `NodeDataError` whose
 * `original` is the `RangeError`, with the full message where Python matches a part of it.
 *
 * Note: `CaseNode.rewriteLocalOrigins` rebuilds each `Case` through its constructor, so it
 * re-checks the handle there; Python's rewrite assigns the origin in place and checks the handle
 * only in `validate_execution_setup`. No path reaches a rewrite with such a `Case`.
 *
 * Later (compose, RunState and the run boundary):
 * `test_a_then_target_named_default_runs_only_on_its_own_branch`,
 * `test_a_then_target_named_default_does_not_run_on_no_match_without_else`,
 * `test_an_on_case_target_named_default_runs_only_on_its_own_branch`,
 * `test_the_else_handle_is_no_possible_node_id`,
 * `test_the_else_handle_without_an_edge_skips_every_branch` and
 * `test_a_log_routed_to_the_old_default_handle_does_not_resume`.
 */

import { expect, test } from "bun:test";

import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { NodeDataError, prepareNodeRouted } from "../../src/runtime/state.ts";
import { callRedirectFlow, routerInFlight } from "./run_state_fixtures.ts";

test("a case may not take the else handle as a branch", () => {
  expect(() => new Case(ELSE_HANDLE, { when: "True" })).toThrow(RangeError);
  expect(() => new Case(ELSE_HANDLE, { when: "True" })).toThrow(
    "CASE handle '#else' is reserved for the else route; name the branch by its target node id",
  );

  const held = Object.assign(Object.create(Case.prototype) as Case, { handle: ELSE_HANDLE, when: "x", origin: null });
  const node = new CaseNode("g", [held]);
  expect(() => node.validateExecutionSetup()).toThrow(RangeError);
  expect(() => node.validateExecutionSetup()).toThrow(
    "node 'g' case handle '#else' is reserved for the else route; name the branch by its target node id",
  );
});

test("a route to an unknown handle named default is rejected", () => {
  const { state } = routerInFlight(callRedirectFlow());

  let caught: unknown;
  try {
    prepareNodeRouted(state, "router", "default");
  } catch (error) {
    caught = error;
  }

  expect(caught).toBeInstanceOf(NodeDataError);
  const original = (caught as NodeDataError).original;
  expect((original as Error).constructor).toBe(RangeError);
  expect((original as Error).message).toBe("NodeRouted handle 'default' matches no outgoing edge of 'router'");
});
