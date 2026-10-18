/**
 * The durable steps a node records: `test_node_steps_checks_its_fields` of
 * `tests/engine/test_node_steps_engine.py` (`NodeSteps` checks `starts`, `prior` and `record`).
 *
 * Changed: `ValueError` is `RangeError`; `True` for `starts` is `true`.
 *
 * Left for later: the other thirteen tests (`eval_node`'s steps capability, resume, the run log)
 * need the runtime.
 *
 * Not ported: the `prior=[{}]` row, a list where Python wants a tuple: TS has only arrays.
 */

import { expect, test } from "bun:test";

import { NodeSteps } from "../../src/nodes/contracts.ts";

test("NodeSteps checks its fields", () => {
  const record = (): void => {};

  expect(() => new NodeSteps(true as never, [], record)).toThrow(TypeError);
  expect(() => new NodeSteps(0, [], record)).toThrow(RangeError);
  expect(() => new NodeSteps(1, ["x"] as never, record)).toThrow(TypeError);
  expect(() => new NodeSteps(1, ["x"] as never, record)).toThrow(/^prior must be an array of records$/);
  expect(() => new NodeSteps(1, [], null as never)).toThrow(TypeError);
});
