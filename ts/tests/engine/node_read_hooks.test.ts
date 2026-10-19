/**
 * The node-owned read hooks of the `Node` base: a port of `tests/engine/test_node_read_hooks.py`.
 * Ported: `test_plain_node_defaults`.
 *
 * TS-only: the default `reservedKeyLabel` and `wiringRefLabel` phrases.
 *
 * Later (T3-8): WAIT and MAP (`test_wait_bind_reserved_resolves_until`,
 * `test_map_binds_per_item_and_resolves_over`, `test_map_over_not_a_list_raises`).
 */

import { expect, test } from "bun:test";

import { Node, NodeKind } from "../../src/nodes/base.ts";
import { VariablePool } from "../../src/typesys/index.ts";

/** A minimal ordinary node (no reserved keys, binds up front) for the default-hook check. */
class PlainNode extends Node {
  readonly kind = NodeKind.CODE;
}

test("plain node defaults", () => {
  // An ordinary node reserves no keys and binds up front (not per item).
  const node = new PlainNode("n");
  expect(node.bindsPerItem).toBe(false);
  expect(node.bindReserved(new Map(), new VariablePool())).toEqual({});
  expect(node.reservedWiringKeys()).toEqual(new Set());
});

test("a reserved key is named after the node's kind", () => {
  expect(new PlainNode("n").reservedKeyLabel("until")).toBe("code until");
});

test("a wiring param is named by its recorded label, else as an input", () => {
  const node = new PlainNode("n");
  node.wiringLabels = new Map([["__on", "on:"]]);

  expect(node.wiringRefLabel("__on")).toBe("on:");
  expect(node.wiringRefLabel("p")).toBe("input 'p'");
});
