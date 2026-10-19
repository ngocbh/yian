/**
 * The node-owned read hooks of the `Node` base: a port of `tests/engine/test_node_read_hooks.py`.
 * Ported: `test_plain_node_defaults`, `test_map_binds_per_item_and_resolves_over` and
 * `test_map_over_not_a_list_raises`.
 *
 * Changed: the node wiring is a `Map`; the `binds_per_item` trait is read on an instance.
 *
 * TS-only: the base reserves no wiring keys (`reservedWiringKeys`), and the default
 * `reservedKeyLabel` and `wiringRefLabel` phrases; MAP reserves `over`, labelled `map over`; a MAP
 * wiring without `over` throws a plain `Error` (Python's `KeyError`), not a `BindingError`.
 *
 * Later (T3-8, WAIT): `test_wait_bind_reserved_resolves_until`.
 */

import { expect, test } from "bun:test";

import { Node, NodeKind } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
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
});

test("the base reserves no wiring keys and names a reserved key after its kind", () => {
  const node = new PlainNode("n");
  expect(node.reservedWiringKeys()).toEqual(new Set());
  expect(node.reservedKeyLabel("until")).toBe("code until");
});

test("a wiring param is named by its recorded label, else as an input", () => {
  const node = new PlainNode("n");
  node.wiringLabels = new Map([["__on", "on:"]]);

  expect(node.wiringRefLabel("__on")).toBe("on:");
  expect(node.wiringRefLabel("p")).toBe("input 'p'");
});

test("map binds per item and resolves over", () => {
  // MAP declares per-element binding and resolves `over` to the list run() maps over.
  const pool = new VariablePool();
  pool.set("items", [1n, 2n, 3n]);
  const node = new MapNode("m", { flowId: "child" });
  expect(node.bindsPerItem).toBe(true);
  expect(node.bindReserved(new Map([["over", "${items.output}"]]), pool)).toEqual({ over: [1n, 2n, 3n] });
});

test("map over not a list raises", () => {
  // A non-list `over` source is an expected input-contract rejection at the read seam.
  const pool = new VariablePool();
  pool.set("scalar", 7n);
  const node = new MapNode("m", { flowId: "child" });
  let caught: unknown = null;
  try {
    node.bindReserved(new Map([["over", "${scalar.output}"]]), pool);
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(BindingError);
  expect((caught as BindingError).message).toContain("did not resolve to a list");
  expect((caught as BindingError).failureCode).toBe("contract.input");
});

test("MAP reserves over, labelled map over, and a wiring without it is a setup fault", () => {
  const node = new MapNode("m", { flowId: "child" });
  expect(node.reservedWiringKeys()).toEqual(new Set(["over"]));
  expect(node.reservedKeyLabel("over")).toBe("map over");

  let caught: unknown = null;
  try {
    node.bindReserved(new Map(), new VariablePool());
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(Error);
  expect(caught).not.toBeInstanceOf(BindingError);
  expect((caught as Error).message).toBe("MAP node 'm' has no `over` wiring");
});
