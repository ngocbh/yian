/**
 * The node-level halves of `tests/engine/test_eval_node.py` rows that need node kinds, called on
 * the node directly.
 *
 * Ported (adapted): `test_map_over_accepts_a_binding_carrier`,
 * `test_unbaked_composite_is_a_setup_fault` (the CALL and MAP rows) and
 * `test_map_over_not_a_list_message_is_byte_identical`.
 *
 * Changed: Python drives each node through `eval_node`; here MAP's reserved source goes through
 * `bindReserved` and the setup fault through `run`, which reject or throw the node's own error.
 * `test_map_over_accepts_a_binding_carrier` checks the resolved `over` list, not the `Grow` the
 * MAP body builds from it. The not-a-list message is pinned as the golden format's text.
 *
 * TS-only: `test_map_over_not_a_list_message_is_byte_identical` also checks the error's `origin`
 * (the carrier's, else `null`).
 *
 * Later (T3-8, LOOP): the LOOP row of `test_unbaked_composite_is_a_setup_fault`.
 *
 * Later (T4-6): the evaluator and the other rows.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import type { Node } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Assertion, Binding } from "../../src/nodes/contracts.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

test("map over accepts a binding carrier", () => {
  // MAP's reserved iteration source uses the ordinary binding read seam.
  const node = new MapNode("m", { flowId: "child", child: {} });
  const wiring = new Map([["over", new Binding(["first", "second"], { origin: new OriginRef("map-over") })]]);

  expect(node.bindReserved(wiring, new VariablePool())).toEqual({ over: ["first", "second"] });
});

for (const [label, make, inputs] of [
  ["call", () => new CallNode("call", { flowId: "child" }), {}],
  ["map", () => new MapNode("map", { flowId: "child" }), { over: [] }],
] as const) {
  test(`unbaked composite is a setup fault (${label})`, async () => {
    // A missing compiled child is malformed setup, not expected node rejection.
    const node: Node = make();
    node.preAsserts = [new Assertion("false", "must not mask malformed setup")];

    await expect(node.run(inputs)).rejects.toThrow("baked");
  });
}

test("map over not a list message is byte identical", () => {
  // Raw and carrier-wrapped sources produce the same origin-neutral error text.
  const pool = new VariablePool();
  pool.set("__start__", { bad: "notalist" });
  const rawSource = "${input.bad}";
  const origin = new OriginRef("map-over");
  for (const source of [rawSource, new Binding(rawSource, { origin })]) {
    const node = new MapNode("m", { flowId: "f" });
    let caught: unknown = null;
    try {
      node.bindReserved(new Map([["over", source]]), pool);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(BindingError);
    expect((caught as BindingError).message).toBe("MAP node 'm': `over` (${input.bad}) did not resolve to a list");
    expect((caught as BindingError).origin).toBe(source instanceof Binding ? origin : null);
  }
});
