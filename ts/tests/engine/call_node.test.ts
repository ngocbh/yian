/**
 * The CALL and MAP drivers' shells: a port of `tests/engine/test_call_node.py`.
 *
 * Ported: `test_call_node_has_no_map_fields`, `test_map_node_kind_and_parallel_fields`,
 * `test_call_returns_grow`, `test_map_returns_grow`, `test_map_empty_returns_grow_with_lone_list_end`,
 * `test_call_unbaked_child_raises` and `test_map_unbaked_child_raises`.
 *
 * Changed: the baked child is a stub object (Python's growth rows use `test_expand`'s
 * `_child_flow`, which `run` never reads); `hasattr` is the `in` operator; `run` is async, so the
 * not-baked setup fault is a rejection, and MAP's `bindItem` capability is passed in `caps`.
 *
 * TS-only: the traits read on instances (`isSpawner`, `growDepthDelta`, MAP's `bindsPerItem` and
 * `itemSourceKey`); CALL's seed record is a copy (changing the input record after `run` leaves the
 * seed alone) and keeps an input named `__proto__` as an own field; MAP without `bindItem` throws
 * a `TypeError`, also for an empty `over`.
 */

import { expect, test } from "bun:test";

import { Grow, type NodeCaps, NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";

/** A stub baked child: the shell only reads `child` for the not-baked guard. */
function child(): object {
  return {};
}

test("call node has no map fields", () => {
  const call = new CallNode("c", { flowId: "child", child: child() });
  expect(call.kind).toBe(NodeKind.CALL);
  expect("over" in call).toBe(false);
  expect("parallel" in call).toBe(false);
});

test("map node kind and parallel fields", () => {
  const mp = new MapNode("m", { flowId: "child", child: child(), parallel: true });
  expect(mp.kind).toBe(NodeKind.MAP);
  expect(mp.parallel).toBe(true);
  expect("over" in mp).toBe(false); // the source rides flow.wiring, not the node
  expect(new MapNode("d", { flowId: "child", child: child() }).parallel).toBe(false);
});

test("call unbaked child raises", async () => {
  const node = new CallNode("c", { flowId: "child", child: null });
  await expect(node.run({ topic: "ACME" })).rejects.toThrow("CALL node 'c': child flow 'child' not baked");
});

test("map unbaked child raises", async () => {
  const node = new MapNode("m", { flowId: "child", child: null });
  const caps: NodeCaps = { bindItem: (element) => ({ topic: element }) };
  await expect(node.run({ over: ["ACME"] }, caps)).rejects.toThrow("MAP node 'm': child flow 'child' not baked");
});

test("call returns grow", async () => {
  // CALL returns only the raw call-arg record; compile owns fragment construction.
  const node = new CallNode("c", { flowId: "child", child: child(), childInputs: [] });
  const out = await node.run({ topic: "ACME" });
  expect(out).toBeInstanceOf(Grow);
  expect((out as Grow).seed).toEqual({ topic: "ACME" });
  expect((out as Grow).prune).toEqual(new Set());
  expect("subgraph" in out).toBe(false);
});

test("map returns grow", async () => {
  // MAP returns only the raw per-element records; compile owns fan-in construction.
  const node = new MapNode("m", { flowId: "child", child: child(), childInputs: [] });
  const out = await node.run({ over: ["ACME", "BETA"] }, { bindItem: (element) => ({ x: element }) });
  expect(out).toBeInstanceOf(Grow);
  expect((out as Grow).seed).toEqual([{ x: "ACME" }, { x: "BETA" }]);
  expect((out as Grow).prune).toEqual(new Set());
  expect("subgraph" in out).toBe(false);
});

test("map empty returns grow with lone list end", async () => {
  const node = new MapNode("m", { flowId: "child", child: child(), childInputs: [] });
  const out = await node.run({ over: [] }, { bindItem: (element) => ({ x: element }) });
  expect(out).toBeInstanceOf(Grow);
  expect((out as Grow).seed).toEqual([]);
  expect((out as Grow).prune).toEqual(new Set());
  expect("subgraph" in out).toBe(false);
});

test("the CALL seed is a copy of the input record", async () => {
  const inputs: Record<string, unknown> = { topic: "ACME" };
  const out = (await new CallNode("c", { flowId: "child", child: child() }).run(inputs)) as Grow;
  inputs.topic = "BETA";
  inputs.extra = 1n;
  expect(out.seed).toEqual({ topic: "ACME" });
});

test("an input named __proto__ stays an own field of the CALL seed", async () => {
  const inputs = JSON.parse('{"__proto__": "x", "topic": "ACME"}') as Record<string, unknown>;
  const out = (await new CallNode("c", { flowId: "child", child: child() }).run(inputs)) as Grow;
  const seed = out.seed as Record<string, unknown>;
  expect(Object.hasOwn(seed, "__proto__")).toBe(true);
  expect(seed["__proto__"]).toBe("x");
  expect(Object.getPrototypeOf(seed)).toBe(Object.prototype);
});

test("MAP without bindItem throws a TypeError, also for an empty over", async () => {
  for (const over of [["ACME"], []]) {
    const error = await new MapNode("m", { flowId: "child", child: child() }).run({ over }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toBe("MAP node 'm' was run without the bindItem capability");
  }
});

test("CALL and MAP are spawners one call level deep, and MAP binds per item from over", () => {
  const call = new CallNode("c", { flowId: "child" });
  expect(call.isSpawner).toBe(true);
  expect(call.growDepthDelta).toBe(1);
  expect(call.bindsPerItem).toBe(false);
  expect(call.itemSourceKey).toBeNull();

  const mp = new MapNode("m", { flowId: "child" });
  expect(mp.isSpawner).toBe(true);
  expect(mp.growDepthDelta).toBe(1);
  expect(mp.bindsPerItem).toBe(true);
  expect(mp.itemSourceKey).toBe("over");
});
