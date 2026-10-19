/**
 * Execution scopes: the two `ExecutionScope`/`ExecutionPosition` tests of
 * `tests/engine/test_backward_contracts.py` (root-relative structured positions, and the rejected
 * positions and paths).
 *
 * Changed: `FrozenInstanceError` is the `TypeError` an assignment to a frozen object throws in a
 * module; dataclass `!=` is `not.toEqual`; each rejected row names its one error class, where
 * Python accepts either `TypeError` or `ValueError` (`RangeError`).
 *
 * Left for later: the other fifteen tests (`BackwardResult`, the backward contexts, memory and the
 * run contracts) need `nodes/base` and the runtime.
 *
 * Not ported: the rows that pass a list where Python wants a tuple (`static_path=["node"]`,
 * `occurrence_path=[]`): TS has only arrays.
 */

import { expect, test } from "bun:test";

import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";

test("execution scopes keep root-relative structured positions", () => {
  const root = new ExecutionScope([], { flowPath: [] });
  expect(root.occurrencePath).toEqual([]);
  expect(root.boundary).toBeNull();
  const positions = [
    new ExecutionPosition("call", ["child"]),
    new ExecutionPosition("map_item", ["child", "each"], 2),
    new ExecutionPosition("loop_body", ["child", "each", "loop"], 3),
  ];
  const scope = new ExecutionScope(["child", "each", "loop", "END"], {
    flowPath: ["child", "each", "loop"],
    occurrencePath: positions,
    boundary: "end",
  });
  expect(scope.occurrencePath).toEqual(positions);
  expect(positions[2]).not.toEqual(new ExecutionPosition("loop_driver", positions[2]!.staticPath, 3));
  expect(() => {
    (scope as { boundary: string | null }).boundary = "start";
  }).toThrow(TypeError);
});

test.each<[string, () => unknown, typeof TypeError | typeof RangeError]>([
  ["a boolean path component", () => new ExecutionScope([true] as never), RangeError],
  ["an unknown boundary", () => new ExecutionScope(["node"], { boundary: "other" as never }), RangeError],
  ["a boolean index", () => new ExecutionPosition("map_item", ["each"], true as never), TypeError],
  ["a negative index", () => new ExecutionPosition("loop_driver", ["loop"], -1), RangeError],
  ["a missing index", () => new ExecutionPosition("loop_body", ["loop"]), TypeError],
  ["an index on a call", () => new ExecutionPosition("call", ["child"], 2), RangeError],
])("execution scopes reject %s", (_name, factory, cls) => {
  expect(factory).toThrow(cls);
});
