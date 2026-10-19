/**
 * The node-level halves of `tests/engine/test_eval_node.py` rows that need node kinds, called on
 * the node directly.
 *
 * Ported: `test_case_rejects_a_non_origin_predicate_carrier`.
 *
 * Ported (adapted): `test_map_over_accepts_a_binding_carrier`,
 * `test_wait_until_accepts_a_binding_carrier`,
 * `test_unbaked_composite_is_a_setup_fault`,
 * `test_case_without_predicate_is_a_setup_fault`,
 * `test_case_with_non_string_predicate_is_a_setup_fault` and
 * `test_map_over_not_a_list_message_is_byte_identical`.
 *
 * Changed: Python drives each node through `eval_node`; here MAP's and WAIT's reserved sources go
 * through `bindReserved` and the setup faults through `run`, which reject or throw the node's own
 * error. The non-string `when` is a cast. `RangeError` stands for Python's `ValueError`.
 * `test_map_over_accepts_a_binding_carrier` checks the resolved `over` list, not the `Grow` the
 * MAP body builds from it, and `test_wait_until_accepts_a_binding_carrier` the resolved `until`,
 * not the `Output` the WAIT body returns. The not-a-list message is pinned as the golden format's
 * text.
 *
 * TS-only: `test_map_over_not_a_list_message_is_byte_identical` also checks the error's `origin`
 * (the carrier's, else `null`).
 *
 * Later (T4-9): `test_case_predicate_failure_selects_the_active_ordered_origin` and
 * `test_source_free_case_predicate_failure_falls_back_to_the_node_origin`.
 *
 * Later (T3-8, WAIT): `test_wait_until_accepts_a_binding_carrier`.
 *
 * Later (T4-6): the evaluator and the other rows.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import type { Node } from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { Assertion, Binding } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { WaitNode } from "../../src/nodes/wait/node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";

test("map over accepts a binding carrier", () => {
  // MAP's reserved iteration source uses the ordinary binding read seam.
  const node = new MapNode("m", { flowId: "child", child: {} });
  const wiring = new Map([["over", new Binding(["first", "second"], { origin: new OriginRef("map-over") })]]);

  expect(node.bindReserved(wiring, new VariablePool())).toEqual({ over: ["first", "second"] });
});

test("wait until accepts a binding carrier", () => {
  // WAIT's dedicated resolver accepts the carrier shape it bypasses elsewhere.
  const node = new WaitNode("wait");
  const source = new Binding("2000-01-01T00:00:00+00:00", { origin: new OriginRef("wait-until") });
  const wiring = new Map([["until", source]]);

  expect(node.bindReserved(wiring, new VariablePool())).toEqual({ until: "2000-01-01T00:00:00+00:00" });
});

for (const [label, make, inputs] of [
  ["call", () => new CallNode("call", { flowId: "child" }), {}],
  ["map", () => new MapNode("map", { flowId: "child" }), { over: [] }],
  ["loop", () => new LoopNode("loop", { flowId: "child", predicate: "true", maxIters: 1 }), {}],
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

test("case without predicate is a setup fault", async () => {
  // A compiled CASE branch must have its validated predicate.
  const node = new CaseNode("case", [new Case("branch")]);

  await expect(node.run({})).rejects.toThrow(RangeError);
  await expect(node.run({})).rejects.toThrow("node 'case' case 'branch' has no `when` expression");
});

test("case with non string predicate is a setup fault", async () => {
  // Malformed code-built CASE state must not look like node execution.
  const malformed = new Case("branch", { when: 42 as unknown as string });
  const node = new CaseNode("case", [malformed]);

  await expect(node.run({})).rejects.toThrow(TypeError);
  await expect(node.run({})).rejects.toThrow("node 'case' case 'branch' `when` must be a string");
});

test("case rejects a non origin predicate carrier", () => {
  // Code-built CASE provenance must stay source-neutral.
  const make = () => new Case("branch", { when: "true", origin: "flow.yaml:12" as unknown as OriginRef });

  expect(make).toThrow(TypeError);
  expect(make).toThrow("CASE predicate origin");
});
