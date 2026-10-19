/**
 * The node-owned read hooks of the `Node` base: a port of `tests/engine/test_node_read_hooks.py`.
 * Ported: `test_wait_bind_reserved_resolves_until`, `test_plain_node_defaults`,
 * `test_map_binds_per_item_and_resolves_over` and `test_map_over_not_a_list_raises`.
 *
 * Changed: the node wiring is a `Map`; the `binds_per_item` trait is read on an instance.
 *
 * TS-only: the base reserves no wiring keys (`reservedWiringKeys`), and the default
 * `reservedKeyLabel` and `wiringRefLabel` phrases; MAP reserves `over`, labelled `map over`; a MAP
 * wiring without `over` throws a plain `Error` (Python's `KeyError`), not a `BindingError`; a
 * `bind` capability gets `over`, its source and the normalizer, whose result comes back as `over`;
 * a resolved list holding NaN is a `binding.evaluate` `BindingError`. WAIT's `until` is labelled
 * `wait until`; a WAIT wiring without `until` throws a plain `Error`; a `bind` capability gets
 * `until`, its source and the normalizer, whose result comes back as `until`, and the normalizer
 * still refuses a non-date under `bind`.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { Node, NodeKind } from "../../src/nodes/base.ts";
import { BindingError, type ReservedBinder } from "../../src/nodes/binding.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";
import { VariablePool } from "../../src/typesys/index.ts";

/** A minimal ordinary node (no reserved keys, binds up front) for the default-hook check. */
class PlainNode extends Node {
  readonly kind = NodeKind.CODE;
}

test("wait bind reserved resolves until", () => {
  // WAIT pre-resolves its `until` source (from nodeWiring) into a concrete ISO deadline.
  const pool = new VariablePool();
  pool.set("deadline", "2026-01-01T00:00:00");
  const node = new WaitNode("w");
  const got = node.bindReserved(new Map([["until", "${deadline.output}"]]), pool);
  expect(got).toEqual({ until: "2026-01-01T00:00:00" });
});

test("WAIT labels until as wait until, and a wiring without it is a setup fault", () => {
  const node = new WaitNode("w");
  expect(node.reservedKeyLabel("until")).toBe("wait until");

  let caught: unknown = null;
  try {
    node.bindReserved(new Map(), new VariablePool());
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(Error);
  expect(caught).not.toBeInstanceOf(BindingError);
  expect((caught as Error).message).toBe("WAIT node 'w' has no `until` wiring");
});

test("WAIT hands until, its source and its normalizer to a bind capability", () => {
  const node = new WaitNode("w");
  const calls: [string, unknown][] = [];
  const bind: ReservedBinder = (name, source, normalize) => {
    calls.push([name, source]);
    return normalize("2026-01-01");
  };

  expect(node.bindReserved(new Map([["until", "${deadline.output}"]]), new VariablePool(), { bind })).toEqual({
    until: "2026-01-01",
  });
  expect(calls).toEqual([["until", "${deadline.output}"]]);
});

test("WAIT's normalizer refuses a non-date under a bind capability", () => {
  const node = new WaitNode("w");
  const bind: ReservedBinder = (_name, _source, normalize) => normalize(5n);
  let caught: unknown = null;
  try {
    node.bindReserved(new Map([["until", "${deadline.output}"]]), new VariablePool(), { bind });
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(BindingError);
  expect((caught as BindingError).failureCode).toBe("contract.input");
});

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

test("MAP hands over, its source and its normalizer to a bind capability", () => {
  const node = new MapNode("m", { flowId: "child" });
  const calls: [string, unknown][] = [];
  const bind: ReservedBinder = (name, source, normalize) => {
    calls.push([name, source]);
    return normalize([1n, 2n]);
  };

  expect(node.bindReserved(new Map([["over", "${items.output}"]]), new VariablePool(), { bind })).toEqual({
    over: [1n, 2n],
  });
  expect(calls).toEqual([["over", "${items.output}"]]);
});

test("MAP over holding NaN is a binding evaluate rejection", () => {
  const node = new MapNode("m", { flowId: "child" });
  const origin = new OriginRef("map-over");
  let caught: unknown = null;
  try {
    node.bindReserved(new Map([["over", new Binding([NaN], { origin })]]), new VariablePool());
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(BindingError);
  expect((caught as BindingError).message).toContain("at [0]: NaN is not a yian value");
  expect((caught as BindingError).failureCode).toBe("binding.evaluate");
  expect((caught as BindingError).origin).toBe(origin);
});
