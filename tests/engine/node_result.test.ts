/**
 * The `NodeResult` closed sum `Output | Route | Grow` of `nodes/base.ts`, a port of
 * `tests/engine/test_node_result.py`. A failing node rejects; there is no failure result variant.
 * `Route` is tested in `route.test.ts`.
 *
 * Changed: Python's `frozenset()` is an empty `Set`, and `hasattr` is the `in` operator.
 */

import { expect, test } from "bun:test";

import { Grow, Output } from "../../src/nodes/base.ts";

test("output carries value", () => {
  expect(new Output(3n).value).toBe(3n);
});

test("grow is defined", () => {
  const g = new Grow({ x: 1n });
  expect(g.seed).toEqual({ x: 1n });
  expect(g.prune).toEqual(new Set());
  expect("subgraph" in g).toBe(false);
});
