/**
 * Provenance-bearing semantic IR in the compiled flow model and the node base: rows of
 * `tests/engine/test_semantic_ir.py`. Ported: the model rows
 * `test_flow_construction_normalizes_direct_wiring_values`,
 * `test_compiled_flow_from_parts_normalizes_direct_wiring_values`,
 * `test_flow_construction_rejects_non_string_wiring_keys`,
 * `test_add_subgraph_normalizes_wiring_and_preserves_existing_binding`,
 * `test_add_subgraph_rejects_bad_wiring_before_mutating_topology` and the binding half of
 * `test_flow_output_normalizes_one_binding_and_keeps_a_source_view`; and the node rows
 * `test_node_origin_is_optional_validated_and_distinct_from_loop_attribution`,
 * `test_node_failure_trace_selects_local_origin_after_generated_context`,
 * `test_node_failure_trace_preserves_source_free_context_and_duplicates` and
 * `test_node_failure_trace_rejects_malformed_generated_origins`.
 *
 * Changed:
 * - Python's bare `Node` is a minimal subclass; node and wiring tables are `Map`s, and a non-string
 *   wiring key is a `Map` key such as `1`. Bindings compare by `source`, `origin` and `contract`.
 * - `FlowOutput` has no `from_` source view, so the view and keyword rows become a check that the
 *   output is frozen.
 * - The origin context is an array: the malformed-context row sets a non-array where Python sets a
 *   list, and a forged origin is made with `Object.create(OriginRef.prototype)`, its key error a
 *   `RangeError`. The messages read `node origin_context ...`.
 * - The bad-splice row also checks the adjacency index is unchanged.
 *
 * Later (T3-14…T3-22): the compose/load rows
 * (`test_loaded_compose_flow_has_only_canonical_binding_storage`,
 * `test_boundary_reads_wrapped_output_but_preserves_it_in_end_wiring` and
 * `test_reference_validation_checks_a_wrapped_flow_output`); the output-contract and
 * child-signature rows (`test_output_contract_*` and `test_child_signature_*`); the CALL/LOOP
 * type-check and composite-builder rows (`test_call_type_*`, `test_loop_type_check_*` and
 * `test_composite_builders_preserve_or_deliberately_wrap_child_contract`); inline binding type
 * inference (`test_inline_binding_type_inference_reads_one_carrier_layer`); and edge inference
 * (`test_data_edge_inference_is_identical_for_raw_and_wrapped_sources`,
 * `test_producer_scan_unwraps_exactly_one_carrier_layer` and
 * `test_case_edge_inference_reads_wrapped_lowered_wiring`).
 *
 * Later (T3-21): the identity and manifest rows (`test_*_identity_*`, `test_*manifest*` and
 * `test_assertion_*`) and `test_wrapped_wiring_keeps_its_source_text_and_identity`.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, Flow, FlowOutput } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { Binding } from "../../src/nodes/contracts.ts";

/** A minimal concrete node; the base fields are the subject. */
class BareNode extends Node {
  readonly kind = NodeKind.CODE;

  override forward(): NodeResult {
    return new Output(null);
  }
}

/** Return one short opaque provenance key for semantic-IR tests. */
function origin(key: string): OriginRef {
  return new OriginRef(key);
}

/** Return an `OriginRef` that bypassed the constructor, so its key was never checked. */
function forgedOrigin(): OriginRef {
  return Object.create(OriginRef.prototype) as OriginRef;
}

test("node origin is optional, validated and distinct from loop attribution", () => {
  const sourceOrigin = origin("node/header");

  const sourceFree = new BareNode("source-free");
  const authored = new BareNode("authored", { origin: sourceOrigin });

  expect(sourceFree.origin).toBeNull();
  expect(authored.origin).toBe(sourceOrigin);
  expect(authored.originId).toBeNull();
  expect(() => new BareNode("invalid", { origin: "node/header" as never })).toThrow(TypeError);
  expect(() => new BareNode("invalid", { origin: "node/header" as never })).toThrow(
    "node origin must be an OriginRef or null",
  );
});

test("node failure trace selects the local origin after generated context", () => {
  const outer = origin("outer/call");
  const nodeOrigin = origin("child/node");
  const operation = origin("child/input/value");
  const node = new BareNode("child", { origin: nodeOrigin });

  expect(node.originContext).toEqual([]);
  expect(node.failureTrace()).toEqual([nodeOrigin]);
  expect(node.failureTrace(operation)).toEqual([operation]);

  node.originContext = [outer];
  expect(node.failureTrace()).toEqual([outer, nodeOrigin]);
  expect(node.failureTrace(operation)).toEqual([outer, operation]);
});

test("node failure trace preserves source-free context and duplicates", () => {
  const repeated = origin("shared/site");
  const sourceFree = new BareNode("source-free");

  expect(sourceFree.failureTrace()).toEqual([]);

  sourceFree.originContext = [repeated, repeated];
  expect(sourceFree.failureTrace()).toEqual([repeated, repeated]);
  expect(sourceFree.failureTrace(repeated)).toEqual([repeated, repeated, repeated]);
});

test("node failure trace rejects malformed generated origins", () => {
  const node = new BareNode("node");
  (node as { originContext: unknown }).originContext = new Set();
  expect(() => node.failureTrace()).toThrow(TypeError);
  expect(() => node.failureTrace()).toThrow("node origin_context must be an array");

  (node as { originContext: unknown }).originContext = ["outer"];
  expect(() => node.failureTrace()).toThrow(TypeError);
  expect(() => node.failureTrace()).toThrow("node origin_context item must be an OriginRef");

  node.originContext = [forgedOrigin()];
  expect(() => node.failureTrace()).toThrow(RangeError);
  expect(() => node.failureTrace()).toThrow("origin key must be a non-empty string");

  node.originContext = [];
  expect(() => node.failureTrace("operation" as never)).toThrow(TypeError);
  expect(() => node.failureTrace("operation" as never)).toThrow("failure operation origin must be an OriginRef");
  expect(() => node.failureTrace(forgedOrigin())).toThrow(RangeError);
  expect(() => node.failureTrace(forgedOrigin())).toThrow("origin key must be a non-empty string");
});

test("flow construction normalizes direct wiring values", () => {
  // Stored flow wiring is canonical without copying semantic payloads.
  const payload = { literal: ["kept-by-identity"] };
  const existing = new Binding("${producer.output}", { origin: origin("existing") });
  const nested = new Binding(existing, { origin: origin("outer") });
  const sink = new Map<string, unknown>([
    ["raw", payload],
    ["existing", existing],
    ["nested", nested],
  ]);
  const authored = new Map([["sink", sink]]);

  const flow = new Flow(new Map([["sink", new BareNode("sink")]]), [], { wiring: authored });

  expect(flow.wiring).not.toBe(authored);
  expect(flow.wiring.get("sink")).not.toBe(sink);
  expect(flow.wiring.get("sink")!.get("raw")).toEqual(new Binding(payload));
  expect(flow.wiring.get("sink")!.get("raw")!.source).toBe(payload);
  expect(flow.wiring.get("sink")!.get("existing")).toBe(existing);
  expect(flow.wiring.get("sink")!.get("nested")).toBe(nested);
});

test("compiled flow fromParts normalizes direct wiring values", () => {
  const existing = new Binding(7n, { origin: origin("count") });

  const flow = CompiledFlow.fromParts(new Map([["sink", new BareNode("sink")]]), [], {
    wiring: new Map([
      [
        "sink",
        new Map<string, unknown>([
          ["raw", 3n],
          ["existing", existing],
        ]),
      ],
    ]),
  });

  expect(flow.wiring.get("sink")!.get("raw")).toEqual(new Binding(3n));
  expect(flow.wiring.get("sink")!.get("existing")).toBe(existing);
});

const BAD_WIRING: [string, (sink: string) => unknown, (sink: string) => string][] = [
  [
    "node id",
    () => new Map([[1, new Map([["value", "raw"]])]]),
    () => "flow wiring node id must be a string, got number",
  ],
  [
    "parameter name",
    (sink) => new Map([[sink, new Map([[1, "raw"]])]]),
    (sink) => `flow wiring parameter name for node '${sink}' must be a string, got number`,
  ],
];

for (const constructor of ["flow", "compiled"]) {
  test.each(BAD_WIRING)(`${constructor} construction rejects a non-string wiring %s`, (_name, wiring, message) => {
    const nodes = new Map<string, Node>([["sink", new BareNode("sink")]]);
    const build = () =>
      constructor === "flow"
        ? new Flow(nodes, [], { wiring: wiring("sink") as never })
        : CompiledFlow.fromParts(nodes, [], { wiring: wiring("sink") as never });
    expect(build).toThrow(TypeError);
    expect(build).toThrow(message("sink"));
  });
}

test("addSubgraph normalizes wiring and preserves an existing binding", () => {
  // Programmatic growth enters the same canonical stored representation.
  const flow = CompiledFlow.fromParts(new Map([["base", new BareNode("base")]]), []);
  const existing = new Binding("${base.output}", { origin: origin("child-value") });

  flow.addSubgraph(
    new Map([["child", new BareNode("child")]]),
    [new Edge("base->child", "base", "child", { inputGroup: "value" })],
    new Map([
      [
        "child",
        new Map<string, unknown>([
          ["raw", 4n],
          ["existing", existing],
        ]),
      ],
    ]),
  );

  expect(flow.wiring.get("child")!.get("raw")).toEqual(new Binding(4n));
  expect(flow.wiring.get("child")!.get("existing")).toBe(existing);
});

test.each(BAD_WIRING)("addSubgraph rejects a bad wiring %s before changing topology", (_name, wiring, message) => {
  // An invalid splice can't leave nodes, edges, wiring or adjacency partially installed.
  const flow = CompiledFlow.fromParts(new Map([["base", new BareNode("base")]]), [], {
    wiring: new Map([["base", new Map()]]),
  });
  const beforeNodes = new Map(flow.nodes);
  const beforeEdges = [...flow.edges];
  const beforeWiring = new Map(flow.wiring);
  const beforeIncoming = new Map(flow.incomingEdges);
  const beforeOutgoing = new Map(flow.outgoingEdges);

  const splice = () =>
    flow.addSubgraph(
      new Map([["child", new BareNode("child")]]),
      [new Edge("base->child", "base", "child")],
      wiring("child") as never,
    );
  expect(splice).toThrow(TypeError);
  expect(splice).toThrow(message("child"));

  expect(flow.nodes).toEqual(beforeNodes);
  expect(flow.edges).toEqual(beforeEdges);
  expect(flow.wiring).toEqual(beforeWiring);
  expect(flow.incomingEdges).toEqual(beforeIncoming);
  expect(flow.outgoingEdges).toEqual(beforeOutgoing);
});

test("flow output normalizes one binding", () => {
  // Declared outputs own carriers.
  const payload = { literal: ["kept-by-identity"] };
  const raw = new FlowOutput("result", payload);
  const existing = new Binding("${producer.output}", { origin: origin("declared-output") });
  const wrapped = new FlowOutput("result", existing);

  expect(raw.binding).toEqual(new Binding(payload));
  expect(raw.binding.source).toBe(payload);
  expect(wrapped.binding).toBe(existing);
  expect(() => {
    (wrapped as { binding: unknown }).binding = new Binding("changed");
  }).toThrow(TypeError);
});
