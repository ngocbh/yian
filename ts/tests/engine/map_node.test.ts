/**
 * `NodeKind.MAP` and `MapNode`, the re-split MAP driver: a port of `tests/engine/test_map_node.py`.
 * `MapNode` discriminates by kind, not an `over` flag: it carries no `over` field, and the `over`
 * source rides `flow.wiring[id].over`.
 *
 * Ported: `test_map_node_kind_value`, `test_map_node_carries_no_over_attr` and
 * `test_map_node_kind_exists_and_package_imports`.
 *
 * Changed: `NodeKind` is a const object of strings, so a member is its value and has no `.value`;
 * `hasattr` is the `in` operator; the package import is a dynamic `import()` of the module.
 *
 * Later (T4-10): `test_map_node_run_builds_grow_fan_in_per_element` and
 * `test_map_node_run_empty_over_builds_lone_list_end`.
 */

import { expect, test } from "bun:test";

import { NodeKind } from "../../src/nodes/base.ts";
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

test("map node kind exists and package imports", async () => {
  expect("MAP" in NodeKind).toBe(true);
  const module = await import("../../src/nodes/map/node.ts");
  expect(module.MapNode).toBe(MapNode);
});
