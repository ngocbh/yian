/**
 * `NodeKind.MAP` and `MapNode`, the re-split MAP driver. `MapNode` discriminates by kind, not an
 * `over` flag: it carries no `over` field, and the `over` source rides `flow.wiring[id].over`.
 */

import { expect, test } from "bun:test";

import { Grow, NodeKind } from "../../src/nodes/base.ts";
import { MapNode } from "../../src/nodes/map/node.ts";

test("map node kind value", () => {
  expect(NodeKind.MAP).toBe("map");
});

test("map node carries no over attr", () => {
  const n = new MapNode("m", { flowId: "child", child: {}, parallel: true });
  expect(n.parallel).toBe(true);
  expect(n.child).not.toBeNull();
  expect("over" in n).toBe(false); // the discriminator is the kind, not an `over` flag
  expect(new MapNode("d", { flowId: "child", child: {} }).parallel).toBe(false);
});

test("map node run builds grow fan in per element", async () => {
  const n = new MapNode("m", { flowId: "child", child: {}, childInputs: [] });
  const grow = await n.run({ over: [1n, 2n, 3n] }, { bindItem: (element) => ({ x: element }) });
  expect(grow).toBeInstanceOf(Grow);
  expect((grow as Grow).seed).toEqual([{ x: 1n }, { x: 2n }, { x: 3n }]);
  expect((grow as Grow).prune).toEqual(new Set());
  expect("subgraph" in grow).toBe(false);
});

test("map node run empty over builds lone list end", async () => {
  const n = new MapNode("m", { flowId: "child", child: {}, childInputs: [] });
  const grow = await n.run({ over: [] }, { bindItem: (element) => ({ x: element }) });
  expect(grow).toBeInstanceOf(Grow);
  expect((grow as Grow).seed).toEqual([]);
  expect((grow as Grow).prune).toEqual(new Set());
  expect("subgraph" in grow).toBe(false);
});

test("bindItem receives each element and its index in order, and none for an empty over", async () => {
  const calls: [unknown, number][] = [];
  const bindItem = (element: unknown, index: number) => {
    calls.push([element, index]);
    return {};
  };
  const n = new MapNode("m", { flowId: "child", child: {} });
  await n.run({ over: ["a", "b", "c"] }, { bindItem });
  expect(calls).toEqual([
    ["a", 0],
    ["b", 1],
    ["c", 2],
  ]);
  calls.length = 0;
  await n.run({ over: [] }, { bindItem });
  expect(calls).toEqual([]);
});

test("map node kind exists and package imports", async () => {
  expect("MAP" in NodeKind).toBe(true);
  const module = await import("../../src/nodes/map/node.ts");
  expect(module.MapNode).toBe(MapNode);
});
