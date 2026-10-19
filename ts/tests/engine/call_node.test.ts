/**
 * The CALL and MAP drivers' shells: a port of `tests/engine/test_call_node.py`.
 *
 * Ported: `test_call_node_has_no_map_fields`, `test_map_node_kind_and_parallel_fields`,
 * `test_call_unbaked_child_raises` and `test_map_unbaked_child_raises`.
 *
 * Changed: the baked child is a stub object; `hasattr` is the `in` operator; `run` is async, so
 * the not-baked setup fault is a rejection, and MAP's `bindItem` capability is passed in `caps`.
 *
 * TS-only: `forward` rejects as not ported yet for a baked CALL and MAP; the traits read on
 * instances (`isSpawner`, `growDepthDelta`, MAP's `bindsPerItem` and `itemSourceKey`).
 *
 * Later (T4-10): `test_call_returns_grow`, `test_map_returns_grow` and
 * `test_map_empty_returns_grow_with_lone_list_end`.
 */

import { expect, test } from "bun:test";

import { type NodeCaps, NodeKind } from "../../src/nodes/base.ts";
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
  await expect(node.run({ topic: "ACME" })).rejects.toThrow("not baked");
  await expect(node.run({ topic: "ACME" })).rejects.toThrow("CALL node 'c': child flow 'child' not baked");
});

test("map unbaked child raises", async () => {
  const node = new MapNode("m", { flowId: "child", child: null });
  const caps: NodeCaps = { bindItem: (element) => ({ topic: element }) };
  await expect(node.run({ over: ["ACME"] }, caps)).rejects.toThrow("not baked");
  await expect(node.run({ over: ["ACME"] }, caps)).rejects.toThrow("MAP node 'm': child flow 'child' not baked");
});

test("a baked CALL or MAP forward is not ported yet", async () => {
  await expect(new CallNode("c", { flowId: "child", child: child() }).run({})).rejects.toThrow(
    "node 'c' (CallNode) forward is not ported yet (T4-10)",
  );
  await expect(new MapNode("m", { flowId: "child", child: child() }).run({ over: [] })).rejects.toThrow(
    "node 'm' (MapNode) forward is not ported yet (T4-10)",
  );
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
