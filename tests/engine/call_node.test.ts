/**
 * The CALL and MAP drivers' shells: their fields and traits, `Grow` results, seed records, and the
 * not-baked setup fault.
 */

import { expect, test } from "bun:test";

import { Grow, type NodeCaps, NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";

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

test("the CALL and MAP seeds copy their records", () => {
  // Through `forward` itself: `run`'s default filling already copies the input record.
  const inputs: Record<string, unknown> = { topic: "ACME" };
  const call = new CallNode("c", { flowId: "child", child: child() }).forward(inputs);
  inputs.topic = "BETA";
  inputs.extra = 1n;
  expect(call.seed).toEqual({ topic: "ACME" });

  const record: ValueRecord = { x: "ACME" };
  const map = new MapNode("m", { flowId: "child", child: child() });
  const grow = map.forward({ over: [null] }, { bindItem: () => record });
  record.x = "BETA";
  expect(grow.seed).toEqual([{ x: "ACME" }]);
});

test("an input named __proto__ stays an own field of the CALL and MAP seeds", () => {
  const record = JSON.parse('{"__proto__": "x", "topic": "ACME"}') as ValueRecord;
  const call = new CallNode("c", { flowId: "child", child: child() }).forward(record);
  const map = new MapNode("m", { flowId: "child", child: child() });
  const grow = map.forward({ over: [null] }, { bindItem: () => record });
  for (const seed of [call.seed, (grow.seed as unknown[])[0]] as Record<string, unknown>[]) {
    expect(Object.hasOwn(seed, "__proto__")).toBe(true);
    expect(seed["__proto__"]).toBe("x");
    expect(Object.getPrototypeOf(seed)).toBe(Object.prototype);
  }
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
