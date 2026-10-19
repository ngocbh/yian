/**
 * CALL and MAP are distinct typed drivers, a port of `tests/engine/test_map_resplit.py`: the
 * retired ref kind and package stay absent; `CallNode` carries no `over` or `parallel`, while
 * `MapNode` has its own kind and keeps the `over` source in flow wiring.
 *
 * Changed: `NodeKind` is a const object of strings, so a member is its value and has no
 * `.value`; `hasattr` is the `in` operator. In `test_retired_ref_package_is_absent` a package
 * import is a dynamic `import()` of the kind's `node.ts`; the missing module's specifier is
 * computed, because a literal one fails the typecheck.
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
  await expect(import(retired)).rejects.toThrow();
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
