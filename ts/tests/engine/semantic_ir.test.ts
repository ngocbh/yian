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
 * `test_node_failure_trace_rejects_malformed_generated_origins`; and the identity and manifest rows
 * `test_wiring_identity_uses_source_and_ignores_binding_origin`,
 * `test_partial_output_contract_is_manifested_without_origins`,
 * `test_flow_output_identity_uses_source_and_ignores_binding_origin`,
 * `test_manifest_binding_policy_rejects_unknown_fields`,
 * `test_manifest_unwraps_exactly_one_binding_layer`,
 * `test_manifest_rejects_raw_values_that_bypass_construction`,
 * `test_assertion_identity_uses_expression_and_display_but_ignores_origin`,
 * `test_assertion_manifest_preserves_order_and_duplicate_count`,
 * `test_assertion_manifest_rejects_unknown_fields_and_raw_bypass` and
 * `test_wrapped_wiring_keeps_its_source_text_and_identity`.
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
 * - Binding, Assertion and FlowOutput are frozen, so `object.__setattr__(carrier, ...)` is a forged
 *   carrier (`Object.assign(Object.create(prototype), carrier, {futureSemanticField: true})`); the
 *   raw-output row sets a forged `FlowOutput` whose binding is a string. Python's `ValueError` is a
 *   `RangeError`, and each error row checks the exact message (a nested binding is refused as an
 *   unsupported non-JSON value of type `Binding`). Python's `repr(manifest)` is `JSON.stringify`.
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
 */

import { expect, test } from "bun:test";

import { expansionDigest, flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import {
  Assertion,
  Binding,
  ContractOrigins,
  PartialRecordContract,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { Type, ValueKind } from "../../src/typesys/index.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

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

/** Check `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** Return a topology-only flow whose terminal wiring contains `value`. */
function manifestFragment(value: unknown, key = "result"): Flow {
  const start = new StartNode("entry", { params: [] });
  const end = EndNode.value("terminal", { outputName: key });
  return new Flow(
    new Map<string, Node>([
      ["entry", start],
      ["terminal", end],
    ]),
    [new Edge("entry->terminal", "entry", "terminal", { inputGroup: key })],
    {
      wiring: new Map([
        ["entry", new Map()],
        ["terminal", new Map([[key, value]])],
      ]),
      startId: "entry",
      endId: "terminal",
    },
  );
}

/** Return a compiled flow whose declared output metadata contains `value`. */
function manifestCompiledOutput(value: unknown): CompiledFlow {
  const start = new StartNode(START_ID, { params: [] });
  const end = EndNode.value(END_ID, { outputName: "result" });
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, start],
      [END_ID, end],
    ]),
    [new Edge("boundary", START_ID, END_ID, { ordering: true })],
    {
      wiring: new Map([
        [START_ID, new Map()],
        [END_ID, new Map([["result", "fixed"]])],
      ]),
      outputs: [new FlowOutput("result", value)],
    },
  );
}

/** Return a manifestable fragment whose END owns the supplied assertion storage. */
function manifestAssertions(assertions: unknown[]): Flow {
  const flow = manifestFragment("value");
  flow.nodes.get("terminal")!.postAsserts = assertions as Assertion[];
  return flow;
}

/** Return `carrier` forged with an extra own field, as a frozen carrier can't take one. */
function withFutureField<T extends object>(carrier: T): T {
  return Object.assign(Object.create(Object.getPrototypeOf(carrier)), carrier, { futureSemanticField: true });
}

test("wiring identity uses the source and ignores the binding origin", () => {
  // Manifest, fingerprint and expansion identity exclude provenance only.
  const source = "${entry.output}";
  const flows = [
    manifestFragment(source),
    manifestFragment(new Binding(source, { origin: origin("first") })),
    manifestFragment(new Binding(source, { origin: origin("second") })),
  ];
  const changed = manifestFragment(new Binding("changed", { origin: origin("first") }));

  const manifests = flows.map(flowManifest);
  const fingerprints = flows.map(flowFingerprint);
  const digests = flows.map(expansionDigest);

  expect(manifests[1]).toEqual(manifests[0]!);
  expect(manifests[2]).toEqual(manifests[0]!);
  expect(new Set(fingerprints).size).toBe(1);
  expect(new Set(digests).size).toBe(1);
  expect(flowManifest(changed)).not.toEqual(manifests[0]!);
  expect(flowFingerprint(changed)).not.toBe(fingerprints[0]!);
  expect(expansionDigest(changed)).not.toBe(digests[0]!);
});

test("a partial output contract is manifested without origins", () => {
  // Field-local acceptance semantics affect identity; diagnostics do not.
  const partial = (kind: ValueKind, site: string) =>
    new PartialRecordContract([
      ["known", new ValueContract(Type.scalar(kind), new ContractOrigins({ default: origin(site) }))],
      ["opaque", null],
    ]);

  const [first, sameSemantics, changedType] = [
    partial(ValueKind.INTEGER, "first"),
    partial(ValueKind.INTEGER, "second"),
    partial(ValueKind.STRING, "first"),
  ].map((contract) => {
    const flow = manifestFragment("value");
    flow.nodes.get("entry")!.outputContract = contract;
    return flow;
  }) as [Flow, Flow, Flow];

  const firstManifest = flowManifest(first);
  expect(((firstManifest.nodes as JsonObject).entry as JsonObject).output_type).toEqual({
    partial_record: [
      {
        name: "known",
        type: { element: null, fields: null, kind: "int", nullable: false, required: null, tags: null },
      },
      { name: "opaque", type: null },
    ],
  });
  expect(flowManifest(sameSemantics)).toEqual(firstManifest);
  expect(flowFingerprint(sameSemantics)).toBe(flowFingerprint(first));
  expect(flowManifest(changedType)).not.toEqual(firstManifest);
  expect(flowFingerprint(changedType)).not.toBe(flowFingerprint(first));
});

test("flow output identity uses the source and ignores the binding origin", () => {
  // Declared-output metadata applies the same source-only identity projection.
  const source = "${__start__.output}";
  const flows = [
    manifestCompiledOutput(source),
    manifestCompiledOutput(new Binding(source, { origin: origin("first") })),
    manifestCompiledOutput(new Binding(source, { origin: origin("second") })),
  ];
  const changed = manifestCompiledOutput(new Binding("changed", { origin: origin("first") }));

  const manifests = flows.map(flowManifest);
  const fingerprints = flows.map(flowFingerprint);

  expect(manifests[1]).toEqual(manifests[0]!);
  expect(manifests[2]).toEqual(manifests[0]!);
  expect(new Set(fingerprints).size).toBe(1);
  expect(flowManifest(changed)).not.toEqual(manifests[0]!);
  expect(flowFingerprint(changed)).not.toBe(fingerprints[0]!);
});

test.each([
  ["wiring", "flow.wiring.terminal.values[0]"],
  ["output", "flow.outputs[0].binding"],
])("the manifest binding policy rejects unknown fields (%s)", (site, path) => {
  // A future carrier field must receive an explicit canonical/ignored policy.
  const binding = withFutureField(new Binding("value", { origin: origin("site") }));
  const flow = site === "wiring" ? manifestFragment(binding) : manifestCompiledOutput(binding);

  expectError(() => flowManifest(flow), RangeError, `${path}: unknown noncanonical field(s): ["futureSemanticField"]`);
});

test.each([
  ["wiring", "flow.wiring.terminal.values[0]"],
  ["output", "flow.outputs[0].binding.source"],
])("the manifest unwraps exactly one binding layer (%s)", (site, path) => {
  // A nested carrier is unsupported literal data, not a second IR layer.
  const nested = new Binding(new Binding("value", { origin: origin("inner") }), { origin: origin("outer") });
  const flow = site === "wiring" ? manifestFragment(nested) : manifestCompiledOutput(nested);

  expectError(() => flowManifest(flow), TypeError, `${path}: unsupported non-JSON value of type Binding`);
});

test.each([
  ["wiring", "flow.wiring.terminal.values[0]"],
  ["output", "flow.outputs[0].binding"],
])("the manifest rejects raw values that bypass construction (%s)", (site, path) => {
  // Identity serialization fails loudly when mutation violates canonical IR.
  let flow: Flow;
  if (site === "wiring") {
    flow = manifestFragment("value");
    (flow.wiring.get("terminal") as Map<string, unknown>).set("result", "bypass");
  } else {
    const output = Object.assign(Object.create(FlowOutput.prototype), { name: "result", binding: "bypass" });
    flow = manifestCompiledOutput("value");
    (flow as CompiledFlow).outputs = [output];
  }

  expectError(() => flowManifest(flow), TypeError, `${path}: expected Binding, got string`);
});

test("assertion identity uses the expression and display but ignores the origin", () => {
  // All identity surfaces retain behavior/presentation and exclude provenance only.
  const base = new Assertion("output > 0", "output must be positive", { origin: origin("first") });
  const otherOrigin = new Assertion(base.expression, base.display, { origin: origin("second") });
  const changedExpression = new Assertion("output >= 0", base.display, { origin: base.origin });
  const changedDisplay = new Assertion(base.expression, "nonnegative output", { origin: base.origin });

  const same = [manifestAssertions([base]), manifestAssertions([otherOrigin])] as const;
  const changed = [manifestAssertions([changedExpression]), manifestAssertions([changedDisplay])];

  expect(flowManifest(same[1])).toEqual(flowManifest(same[0]));
  expect(flowFingerprint(same[1])).toBe(flowFingerprint(same[0]));
  expect(expansionDigest(same[1])).toBe(expansionDigest(same[0]));
  for (const candidate of changed) {
    expect(flowManifest(candidate)).not.toEqual(flowManifest(same[0]));
    expect(flowFingerprint(candidate)).not.toBe(flowFingerprint(same[0]));
    expect(expansionDigest(candidate)).not.toBe(expansionDigest(same[0]));
  }
});

test("the assertion manifest preserves order and duplicate count", () => {
  // An ordered contract sequence is behavioral identity, including repeated checks.
  const first = new Assertion("a", "first");
  const second = new Assertion("b", "second");

  const ordered = manifestAssertions([first, second]);
  const reversed = manifestAssertions([second, first]);
  const duplicated = manifestAssertions([first, second, second]);

  expect(flowFingerprint(ordered)).not.toBe(flowFingerprint(reversed));
  expect(flowFingerprint(ordered)).not.toBe(flowFingerprint(duplicated));
});

test("the assertion manifest rejects unknown fields and a raw bypass", () => {
  // Every carrier field needs an explicit policy and strings cannot re-enter Node IR.
  const assertion = withFutureField(new Assertion("true", "true"));

  expectError(
    () => flowManifest(manifestAssertions([assertion])),
    RangeError,
    'flow.nodes.terminal.postAsserts[0]: unknown noncanonical field(s): ["futureSemanticField"]',
  );
  expectError(
    () => flowManifest(manifestAssertions(["true"])),
    TypeError,
    "flow.nodes.terminal.postAsserts[0]: expected Assertion, got string",
  );
});

test("wrapped wiring keeps its source text and identity", () => {
  // Projecting a carrier keeps its authored source and drops only its origin.
  const firstSource = '${input.primary_token:-"first-value"}';
  const secondSource = '${input.backup_token:-"second-value"}';
  const raw = manifestFragment(firstSource, "authorization");
  const wrapped = manifestFragment(new Binding(firstSource, { origin: origin("credential") }), "authorization");
  const changed = manifestFragment(new Binding(secondSource, { origin: origin("credential") }), "authorization");

  const rawManifest = flowManifest(raw);
  const wrappedManifest = flowManifest(wrapped);
  const rendered = JSON.stringify(wrappedManifest);

  expect(wrappedManifest).toEqual(rawManifest);
  expect(flowFingerprint(wrapped)).toBe(flowFingerprint(raw));
  expect(rendered).toContain(JSON.stringify(firstSource).slice(1, -1));
  expect(rendered).not.toContain("credential");
  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(wrapped));
});
