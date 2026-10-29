/** Kahn's order over a node list and its edges, shared by the cycle check and the load passes. */

import { expect, test } from "bun:test";

import { topologicalOrder } from "../../src/compile/validation.ts";

const edge = (from: string, to: string) => ({ from, to });

test("ready nodes are taken first in, first out, starting in node order", () => {
  const { order, rest } = topologicalOrder(["c", "a", "b", "d"], [edge("c", "d"), edge("a", "b")]);
  expect(order).toEqual(["c", "a", "d", "b"]);
  expect(rest).toEqual([]);
});

test("a node waits for every edge into it, repeated edges included", () => {
  const { order } = topologicalOrder(["x", "y", "z"], [edge("x", "z"), edge("x", "z"), edge("y", "z")]);
  expect(order).toEqual(["x", "y", "z"]);
});

test("nodes on or behind a cycle are left out of the order, in node order", () => {
  const { order, rest } = topologicalOrder(
    ["a", "b", "c", "d"],
    [edge("a", "b"), edge("b", "c"), edge("c", "b"), edge("c", "d")],
  );
  expect(order).toEqual(["a"]);
  expect(rest).toEqual(["b", "c", "d"]);
});
