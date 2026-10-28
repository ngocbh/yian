/**
 * CALL and MAP are distinct typed drivers: the retired ref kind and package stay absent;
 * `CallNode` carries no `over` or `parallel`, while `MapNode` has its own kind and keeps the
 * `over` source in flow wiring.
 */

import { expect, test } from "bun:test";

import { NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";

test("retired ref kind is absent and call map are distinct", () => {
  expect("REF" in NodeKind).toBe(false);
  expect(NodeKind.CALL).toBe("call");
  expect(NodeKind.MAP).toBe("map"); // MAP re-split into its own kind
  expect(NodeKind.CALL).not.toBe(NodeKind.MAP);
});

test("retired ref package is absent", async () => {
  const retired = "../../src/nodes/ref/" + "node.ts";
  await expect(import(retired)).rejects.toMatchObject({ code: "ERR_MODULE_NOT_FOUND" });
  await import("../../src/nodes/call/node.ts");
  await import("../../src/nodes/map/node.ts");
});

test("call node excludes map fields", () => {
  const call = new CallNode("c", { flowId: "child", child: {} });
  expect(call.kind).toBe(NodeKind.CALL);
  expect("over" in call || "parallel" in call).toBe(false);
  const mp = new MapNode("m", { flowId: "child", child: {}, parallel: true });
  expect(mp.kind).toBe(NodeKind.MAP);
  expect(mp.parallel).toBe(true);
  expect("over" in mp).toBe(false); // the source rides flow.wiring, not the node
});
