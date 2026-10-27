/**
 * The seed-first `Grow` outcome of `nodes/base.ts` and the compile-owned `Flow` value type; `Grow`
 * copies `prune`, and `Output`, `Route` and `Grow` are frozen.
 */

import { expect, test } from "bun:test";

import { Flow } from "../../src/compile/model.ts";
import { Grow, Output, Route } from "../../src/nodes/base.ts";

test("flow holds fragment and start id", () => {
  const sg = new Flow(new Map(), [], { wiring: new Map(), startId: "r", endId: "r" });
  expect(sg.startId).toBe("r");
  expect(sg.nodes).toEqual(new Map());
  expect(sg.edges).toEqual([]);
  expect(sg.wiring).toEqual(new Map());
});

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
