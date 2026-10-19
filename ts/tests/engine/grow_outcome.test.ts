/**
 * The seed-first `Grow` outcome of `nodes/base.ts`, a port of
 * `tests/engine/test_grow_outcome.py`.
 *
 * Changed: Python's `frozenset()` is an empty `Set`, and `hasattr` is the `in` operator.
 *
 * TS-only: `Grow` copies `prune`, so changing the passed set doesn't change it, and `Output`,
 * `Route` and `Grow` are frozen.
 *
 * Later (T3-10): `test_flow_holds_fragment_and_start_id`.
 *
 * Not ported: `test_grow_is_a_noderesult_member` (the union is a type).
 */

import { expect, test } from "bun:test";

import { Grow, Output, Route } from "../../src/nodes/base.ts";

test("grow defaults to empty prune", () => {
  const g = new Grow(null);
  expect(g.prune).toEqual(new Set());
  expect(g.seed).toBeNull();
  expect("subgraph" in g).toBe(false);
});

test("grow carries seed", () => {
  const g = new Grow({ x: 1n });
  expect(g.seed).toEqual({ x: 1n });
});

test("grow copies prune", () => {
  const prune = new Set(["a"]);
  const g = new Grow(null, prune);
  prune.add("b");
  expect(g.prune).toEqual(new Set(["a"]));
  expect(g.prune).not.toBe(prune);
});

test("output, route and grow are frozen", () => {
  expect(Object.isFrozen(new Output(1n, { commitAs: "t" }))).toBe(true);
  expect(Object.isFrozen(new Route("x"))).toBe(true);
  expect(Object.isFrozen(new Grow(null, ["a"]))).toBe(true);
});
