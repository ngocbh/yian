/**
 * Provenance-bearing semantic IR in the compiled flow model and the node base: wiring
 * normalization, node origins and failure traces, identity and manifests, output-contract
 * resolution, child signatures, and the type and edge checks that read through one `Binding` layer.
 */

import { expect, test } from "bun:test";

import { expansionDigest, flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import {
  combineOutputContracts,
  directReferenceParts,
  directReferencePath,
  literalContract,
  literalType,
  literalValue,
  localFieldError,
  NOT_LITERAL,
  projectOutputContract,
  recordFieldOrigins,
  referenceContract,
  resolveOutputBindingContract,
  templateContract,
} from "../../src/compile/output_contracts.ts";
import { synthesizeBoundaryGraph } from "../../src/compose/build/boundary.ts";
import { inferInlineBindingType } from "../../src/compose/build/inline_inputs.ts";
import { inferDataEdges } from "../../src/compose/edges/data.ts";
import { bindingProducers } from "../../src/compose/edges/producers.ts";
import { BuildContext, BuildResult, NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { ChildSignature, childSignature } from "../../src/compose/nodes/resolver.ts";
import { checkCallMapTypes } from "../../src/compose/validate/call_types.ts";
import { checkLoopTypeContract } from "../../src/compose/validate/loop_types.ts";
import { validateReferences } from "../../src/compose/validate/references.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { parseExpr } from "../../src/expr/index.ts";
import { Token, Tree } from "../../src/expr/grammar.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import {
  Assertion,
  Binding,
  ContractOrigins,
  isExactly,
  type OutputContract,
  Parameter,
  PartialRecordContract,
  ValueContract,
  ValueTemplate,
} from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { Type, ValueKind } from "../../src/typesys/index.ts";
import { typeEquals } from "../../src/typesys/values.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { assembleText, handChild, loadError } from "../helpers/declarations.ts";

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

test("inline binding type inference reads one carrier layer", () => {
  // A generated inline parameter keeps the type of a wrapped parent reference.
  const parentType = Type.scalar(ValueKind.INTEGER);
  const options = {
    parentInputTypes: new Map([["count", parentType]]),
    producerType: () => Type.scalar(ValueKind.NONE),
    bindingLabel: "call.input.count",
  };

  const wrapped = new Binding("${input.count}", { origin: origin("inline-count") });
  expect(inferInlineBindingType(wrapped, options)).toEqual(parentType);

  const inner = new Binding("${input.count}", { origin: origin("inner") });
  const error = loadError(() => inferInlineBindingType(new Binding(inner, { origin: origin("outer") }), options));
  expect(error.message).toBe(
    "cannot infer type for inline binding 'call.input.count': literal at call.input.count has unsupported " +
      "type 'Binding'; declare its type with a tag (`count: !<type> ...`) or use an explicitly typed named " +
      "`defs:` body",
  );
});

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

  expect(manifests[1]).toEqual(manifests[0]);
  expect(manifests[2]).toEqual(manifests[0]);
  expect(new Set(fingerprints).size).toBe(1);
  expect(new Set(digests).size).toBe(1);
  expect(flowManifest(changed)).not.toEqual(manifests[0]);
  expect(flowFingerprint(changed)).not.toBe(fingerprints[0]);
  expect(expansionDigest(changed)).not.toBe(digests[0]);
});

test("a partial output contract is manifested without origins", () => {
  // Field-local acceptance semantics affect identity; diagnostics do not.
  const partial = (kind: ValueKind, site: string) =>
    new PartialRecordContract([
      ["known", new ValueContract(Type.scalar(kind), new ContractOrigins({ default: origin(site) }))],
      ["opaque", null],
    ]);

  const withContract = (contract: PartialRecordContract) => {
    const flow = manifestFragment("value");
    flow.nodes.get("entry")!.outputContract = contract;
    return flow;
  };
  const first = withContract(partial(ValueKind.INTEGER, "first"));
  const sameSemantics = withContract(partial(ValueKind.INTEGER, "second"));
  const changedType = withContract(partial(ValueKind.STRING, "first"));

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

  expect(manifests[1]).toEqual(manifests[0]);
  expect(manifests[2]).toEqual(manifests[0]);
  expect(new Set(fingerprints).size).toBe(1);
  expect(flowManifest(changed)).not.toEqual(manifests[0]);
  expect(flowFingerprint(changed)).not.toBe(fingerprints[0]);
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
    const compiled = manifestCompiledOutput("value");
    compiled.outputs = [output];
    flow = compiled;
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
  // The manifest text JSON-escapes the source's quotes.
  expect(rendered).toContain(JSON.stringify(firstSource).slice(1, -1));
  expect(rendered).not.toContain("credential");
  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(wrapped));
});

// --------------------------------------------------------------------------------------- //
// Output-binding contracts (compile/output_contracts.ts)
// --------------------------------------------------------------------------------------- //

const STR = Type.scalar(ValueKind.STRING);
const INT = Type.scalar(ValueKind.INTEGER);

/** Return a closed record type whose fields are all required, unless `required` names them. */
function recordType(fields: Record<string, Type>, required: readonly string[] = Object.keys(fields)): Type {
  return new Type({ kind: ValueKind.OBJECT, fields: new Map(Object.entries(fields)), required: new Set(required) });
}

/** Return a `Map` of a record's entries, as producer and flow-input tables are. */
function table<T>(entries: Record<string, T>): Map<string, T> {
  return new Map(Object.entries(entries));
}

/** Resolve `binding` against `producers` and `flowInputs`, given as records. */
function resolveContract(
  binding: unknown,
  producers: Record<string, OutputContract>,
  flowInputs: Record<string, ValueContract> = {},
): OutputContract | null {
  return resolveOutputBindingContract(binding, table(producers), table(flowInputs));
}

/** Return `contract`, requiring it to be an exact `ValueContract`. */
function exact(contract: OutputContract | null): ValueContract {
  if (!isExactly(contract, ValueContract)) throw new Error("expected an exact ValueContract");
  return contract;
}

test("reference validation checks a wrapped flow output", () => {
  // A carrier cannot hide unknown node or output namespaces from validation.
  const bad = new Binding("${missing.output}", { origin: origin("bad-reference") });
  const sink = new BuildResult({ node: new BareNode("sink"), wiring: new Map() });

  const nodeError = loadError(() =>
    validateReferences(new Map([["sink", sink]]), new Set(), new Map(), null, table({ sink: table({ value: bad }) })),
  );
  expect(nodeError.message).toContain("unknown namespace");
  const outputError = loadError(() => validateReferences(new Map(), new Set(), new Map(), bad, new Map()));
  expect(outputError.message).toContain("unknown namespace");

  validateReferences(
    new Map([["sink", sink]]),
    new Set(),
    new Map(),
    null,
    table({ sink: table({ value: new Binding(bad, { origin: origin("outer") }) }) }),
  );
});

test("output contract resolution retains whole-value identity", () => {
  // Raw and wrapped references resolve to the producer's exact carrier.
  const contract = new ValueContract(STR, new ContractOrigins({ default: origin("producer-output") }));
  const source = "${producer.output}";
  const producers = { producer: contract };

  expect(resolveContract(source, producers)).toBe(contract);
  expect(resolveContract(new Binding(source, { origin: origin("typed-output") }), producers)).toBe(contract);
  const doubleWrapped = new Binding(new Binding(source, { origin: origin("inner") }), { origin: origin("outer") });
  expect(resolveContract(doubleWrapped, producers)).toBe(null);
  expect(resolveContract("${input.topic}", {}, { topic: contract })).toBe(contract);

  const computed = exact(resolveContract('${producer.output == "/exit"}', { producer: contract }));
  expect(computed).not.toBe(contract);
  expect(computed.type).toEqual(Type.scalar(ValueKind.BOOLEAN));
  expect(computed.origins).toEqual(new ContractOrigins());

  for (const [expression, expressionProducers] of [
    ["${false}", {}],
    ["${true and false}", {}],
    ["${left.output == right.output}", { left: contract, right: contract }],
  ] as const) {
    const inferred = exact(resolveContract(expression, expressionProducers));
    expect(inferred.type).toEqual(Type.scalar(ValueKind.BOOLEAN));
    expect(inferred.origins).toEqual(new ContractOrigins());
  }

  expect(resolveContract("${producer.output + 1}", { producer: contract })).toBe(null);

  expectError(
    () => resolveContract("${input.topic}", {}, { topic: STR as unknown as ValueContract }),
    TypeError,
    "flow input contracts must be exact ValueContract values, got Type for 'topic'",
  );
  expectError(
    () => resolveContract('${producer.output == "/exit"}', { producer: STR as unknown as ValueContract }),
    TypeError,
    "output producer contracts must be ValueContract or PartialRecordContract values, got Type for 'producer'",
  );
});

test.each([
  ["${42}", ValueKind.INTEGER],
  ["${1.5}", ValueKind.NUMBER],
  ['${"text"}', ValueKind.STRING],
  ["${null}", ValueKind.NONE],
])("output contract resolution infers the whole-span literal %s", (source, kind) => {
  // Compose turns a proven scalar literal span into a source-free contract.
  expect(resolveContract(source, {})).toEqual(new ValueContract(Type.scalar(kind)));
});

test.each([
  [
    "prefix ${producer.output}",
    { producer: STR as unknown as ValueContract },
    {},
    "output producer contracts must be ValueContract or PartialRecordContract values, got Type for 'producer'",
  ],
  [
    "prefix ${input.topic}",
    {},
    { topic: STR as unknown as ValueContract },
    "flow input contracts must be exact ValueContract values, got Type for 'topic'",
  ],
])("output contract resolution validates embedded reference contracts: %s", (source, producers, inputs, message) => {
  // Template shape must not bypass exact semantic-carrier validation.
  expectError(() => resolveContract(source, producers, inputs), TypeError, message);
});

test("output contract resolution rebases nested origins", () => {
  // Selecting a record field makes its origin root and strips its path prefix.
  const nestedType = recordType({ answer: INT, note: STR });
  const rootType = recordType({ result: nestedType, ignored: Type.scalar(ValueKind.BOOLEAN) });
  const contract = new ValueContract(
    rootType,
    new ContractOrigins({
      default: origin("root"),
      entries: [
        [["ignored"], origin("ignored")],
        [["result"], origin("result")],
        [["result", "answer"], origin("answer")],
        [["result", "note"], origin("note")],
      ],
    }),
  );

  const projected = exact(resolveContract("${producer.output.result}", { producer: contract }));
  expect(projected.type).toBe(nestedType);
  expect(projected.origins).toEqual(
    new ContractOrigins({
      default: origin("result"),
      entries: [
        [["answer"], origin("answer")],
        [["note"], origin("note")],
      ],
    }),
  );

  const inputProjected = exact(resolveContract("${input.payload.result}", {}, { payload: contract }));
  expect(inputProjected.type).toBe(nestedType);
  expect(inputProjected.origins).toEqual(projected.origins);

  const fallbackContract = new ValueContract(
    rootType,
    new ContractOrigins({ default: origin("root"), entries: [[["result"], origin("result")]] }),
  );
  const leaf = exact(resolveContract("${producer.output.result.answer}", { producer: fallbackContract }));
  expect(leaf.origins.default).toEqual(origin("result"));
  expect(leaf.origins.entries).toEqual([]);
});

test("output contract projection propagates an optional field path", () => {
  // A missing optional ancestor makes a strict descendant value nullable.
  const optionalRecord = recordType({ value: INT });
  const root = recordType({ payload: optionalRecord }, []);

  const projected = exact(resolveContract("${producer.output.payload.value}", { producer: new ValueContract(root) }));

  expect(projected.type.kind).toBe(ValueKind.INTEGER);
  expect(projected.type.nullable).toBe(true);
});

test("output contract resolution projects proven partial record fields", () => {
  // Known fields remain typed when an opaque sibling prevents an exact record.
  const known = new ValueContract(recordType({ value: INT }), new ContractOrigins({ default: origin("known") }));
  const partial = new PartialRecordContract([
    ["known", known],
    ["opaque", null],
  ]);
  const producers = { producer: partial };

  const projected = exact(resolveContract("${producer.output.known.value}", producers));
  expect(projected.type).toEqual(INT);
  expect(projected.origins.default).toEqual(origin("known"));
  expect(resolveContract("${producer.output.opaque}", producers)).toBe(null);
  expect(resolveContract("${producer.output}", producers)).toBe(partial);
  expect(exact(resolveContract("${producer.output == null}", producers)).type).toEqual(
    Type.scalar(ValueKind.BOOLEAN),
  );
});

test("output contract resolution types embedded text over a partial record as a string", () => {
  const partial = new PartialRecordContract([["opaque", null]]);

  expect(exact(resolveContract("record: ${producer.output}", { producer: partial })).type).toEqual(STR);
});

test.each<[string, unknown, Type | null]>([
  ["an int", 3n, new Type({ kind: ValueKind.INTEGER, bounds: [3n, 3n] })],
  ["a float", 1.5, new Type({ kind: ValueKind.NUMBER, bounds: [1.5, 1.5] })],
  ["Infinity", Infinity, null],
  ["-Infinity", -Infinity, null],
  ["text", "buy", new Type({ kind: ValueKind.STRING, tags: new Set(["buy"]) })],
  ["empty text", "", STR],
  ["a bool", true, Type.scalar(ValueKind.BOOLEAN)],
  ["null", null, Type.scalar(ValueKind.NONE)],
  ["a record", { a: 1n }, Type.scalar(ValueKind.OBJECT)],
  ["a value with no type", new Date(0), null],
])("literalType of %s", (_name, value, expected) => {
  const typ = literalType(value);
  if (expected === null) expect(typ).toBe(null);
  else expect(typeEquals(typ!, expected)).toBe(true);
});

test("literalValue unescapes text and marks templates", () => {
  const value = { a: 1n };
  expect(literalValue("costs $$5")).toBe("costs $5");
  expect(literalValue("hi ${a.output}")).toBe(NOT_LITERAL);
  expect(literalValue(new ValueTemplate({ a: "x" }))).toBe(NOT_LITERAL);
  expect(literalValue("${")).toBe(NOT_LITERAL);
  expect(literalValue(value)).toBe(value);
  expect(literalValue(7n)).toBe(7n);
});

test("literalContract types a nested record, joins a list, and leaves an unjoinable list unproven", () => {
  const nested = exact(literalContract({ a: { b: "x" }, n: 1n }));
  expect(typeEquals(nested.type, recordType({ a: recordType({ b: STR }), n: INT }))).toBe(true);
  const exactNested = exact(literalContract({ a: { b: "x" } }, { exactLiterals: true }));
  const tagX = new Type({ kind: ValueKind.STRING, tags: new Set(["x"]) });
  expect(typeEquals(exactNested.type, recordType({ a: recordType({ b: tagX }) }))).toBe(true);

  // Equal exact items join as they are; differing literals join once widened.
  const same = exact(literalContract(["x", "x"], { exactLiterals: true }));
  expect(typeEquals(same.type, new Type({ kind: ValueKind.LIST_STRING, element: tagX }))).toBe(true);
  const differing = exact(literalContract(["x", "y"], { exactLiterals: true }));
  expect(typeEquals(differing.type, new Type({ kind: ValueKind.LIST_STRING, element: STR }))).toBe(true);
  const ints = exact(literalContract([1n, 2n]));
  expect(typeEquals(ints.type, new Type({ kind: ValueKind.LIST_INTEGER, element: INT }))).toBe(true);

  expect(literalContract([1n, "a"])).toBe(null);
  expect(literalContract([])).toBe(null);
  expect(literalContract(Infinity)).toBe(null);
  // A nested list that doesn't join leaves its field opaque, so the record is partial.
  expect(literalContract({ items: [1n, "a"], n: 1n })).toEqual(
    new PartialRecordContract([
      ["items", null],
      ["n", new ValueContract(INT)],
    ]),
  );
});

test("templateContract infers a record from its fields", () => {
  const declared = new ValueContract(INT, new ContractOrigins({ default: origin("declared") }));
  const template = new ValueTemplate({
    a: "${p.output}",
    b: "lit",
    n: 1n,
    c: new Binding("${q.output}", { contract: declared }),
  });

  const seen: string[] = [];
  const contract = templateContract(template, (text) => {
    seen.push(text);
    return null;
  });

  expect(seen).toEqual(["${p.output}"]);
  expect(contract).toEqual(
    new PartialRecordContract([
      ["a", null],
      ["b", new ValueContract(STR)],
      ["n", new ValueContract(INT)],
      ["c", declared],
    ]),
  );

  const exactRecord = exact(templateContract(new ValueTemplate({ b: "lit" }), () => null, { exactLiterals: true }));
  const tagLit = new Type({ kind: ValueKind.STRING, tags: new Set(["lit"]) });
  expect(typeEquals(exactRecord.type, recordType({ b: tagLit }))).toBe(true);
});

test("combineOutputContracts builds an exact record, a partial record, or rejects a non-contract", () => {
  const a = new ValueContract(STR, new ContractOrigins({ default: origin("a") }));
  const b = new ValueContract(INT);
  const combined = exact(combineOutputContracts(table({ a, b })));
  expect(typeEquals(combined.type, recordType({ a: STR, b: INT }))).toBe(true);
  expect(combined.origins).toEqual(new ContractOrigins({ entries: [[["a"], origin("a")]] }));

  const nestedPartial = new PartialRecordContract([
    ["x", b],
    ["y", null],
  ]);
  expect(combineOutputContracts(table<OutputContract | null>({ a, opaque: null, nested: nestedPartial }))).toEqual(
    new PartialRecordContract([
      ["a", a],
      ["opaque", null],
      ["nested", null],
    ]),
  );

  expectError(
    () => combineOutputContracts(table({ a: STR as unknown as ValueContract })),
    TypeError,
    "output fields must contain ValueContract, PartialRecordContract, or null values, got Type for 'a'",
  );
});

test("recordFieldOrigins indexes field origins in canonical order", () => {
  const b = new ValueContract(
    recordType({ x: STR }),
    new ContractOrigins({ default: origin("b"), entries: [[["x"], origin("b.x")]] }),
  );
  const a = new ValueContract(STR, new ContractOrigins({ default: origin("a") }));

  expect(
    recordFieldOrigins([
      ["b", b],
      ["a", a],
      ["c", null],
    ]),
  ).toEqual(
    new ContractOrigins({
      entries: [
        [["a"], origin("a")],
        [["b"], origin("b")],
        [["b", "x"], origin("b.x")],
      ],
    }),
  );
});

test("referenceContract projects flow inputs and producer outputs", () => {
  const record = new ValueContract(recordType({ f: INT }));
  const producers = table<OutputContract>({ p: record });
  const inputs = table({ x: record });

  expect(exact(referenceContract("input.x.f", producers, inputs)).type).toEqual(INT);
  expect(exact(referenceContract("p.output.f", producers, inputs)).type).toEqual(INT);
  expect(referenceContract("input.x", producers, inputs)).toBe(record);
  expect(referenceContract("q.output", producers, inputs)).toBe(null);
  expect(referenceContract("input.y", producers, inputs)).toBe(null);
  expect(referenceContract("p.other", producers, inputs)).toBe(null);
  expect(referenceContract("p", producers, inputs)).toBe(null);

  expectError(
    () => referenceContract("input.x", producers, table({ x: STR as unknown as ValueContract })),
    TypeError,
    "flow input contracts must be exact ValueContract values, got Type for 'x'",
  );
  expectError(
    () => referenceContract("p.output", table({ p: STR as unknown as ValueContract }), inputs),
    TypeError,
    "output producer contracts must be ValueContract or PartialRecordContract values, got Type for 'p'",
  );
});

test("localFieldError accepts a read any alternative accepts, else reports the first error", () => {
  const alternatives = [new ValueContract(recordType({ a: STR })), new ValueContract(recordType({ b: STR }))];

  expect(localFieldError(alternatives, ["b"], "v.b")).toBe(null);
  expect(localFieldError([alternatives[0]!, new ValueContract(STR)], ["c"], "v.c")).toBe(
    "reference ${v.c} reads unknown field 'c' on record",
  );
});

test("directReferencePath and directReferenceParts read only a lone plain reference", () => {
  expect(directReferencePath("${a.output.f}")).toBe("a.output.f");
  expect(directReferencePath(new Binding("${item}"))).toBe("item");
  expect(directReferencePath("${len(a.output)}")).toBe(null);
  expect(directReferencePath("${a.output :- null}")).toBe(null);
  expect(directReferencePath("text ${a.output}")).toBe(null);
  expect(directReferencePath("${")).toBe(null);
  expect(directReferencePath(5n)).toBe(null);

  expect(directReferenceParts(parseExpr("d.output.action"))).toEqual(["d", "output", "action"]);
  expect(directReferenceParts(parseExpr("len(a.output)"))).toBe(null);
  expect(directReferenceParts(parseExpr("a.output :- null"))).toBe(null);
  expect(directReferenceParts(parseExpr('"text"'))).toBe(null);
  // The grammar has no index; a hand-built `a.0` stands in for one.
  const indexed = new Tree("refcall", [new Token("NAME", "a"), new Tree("trailer", [new Token("NUMBER", "0")])]);
  expect(directReferenceParts(indexed)).toBe(null);
});

test("projectOutputContract rejects a non-contract", () => {
  expectError(
    () => projectOutputContract(STR as unknown as ValueContract, ["f"]),
    TypeError,
    "output contract must be an exact ValueContract or PartialRecordContract, got Type",
  );
});

test("CASE edge inference reads wrapped lowered wiring", () => {
  // CASE-owned data dependencies can't disappear behind a carrier.
  const desc = new CaseDescriptor({
    id: "gate",
    on: "${producer.output}",
    onOrigin: origin("case-on"),
    cases: [{ when: "x", then: "a" }],
  });

  const result = desc.build(new BuildContext({ registry: new Map() }));

  expect(result.wiring.get("__on")).toStrictEqual(new Binding("${producer.output}", { origin: origin("case-on") }));
  const data = result.edges.filter((edge) => edge.sourceHandle === null);
  expect(data.map((edge) => [edge.from, edge.to, edge.inputGroup, edge.optional])).toEqual([
    ["producer", "gate", "__on", false],
  ]);
});

test("childSignature prefixes multi-output contract origins", () => {
  // A record codomain retains each field contract under its output name.
  const alpha = new ValueContract(
    new Type({ kind: ValueKind.OBJECT, fields: new Map([["leaf", STR]]), required: new Set(["leaf"]) }),
    new ContractOrigins({ default: origin("alpha-root"), entries: [[["leaf"], origin("alpha-leaf")]] }),
  );
  const beta = new ValueContract(INT, new ContractOrigins({ default: origin("beta-root") }));
  const loaded = handChild({
    producers: { alpha_node: alpha, beta_node: beta },
    output: { beta: "${beta_node.output}", alpha: "${alpha_node.output}" },
  });

  const output = childSignature(loaded).output;

  expect(isExactly(output, ValueContract)).toBe(true);
  const contract = output as ValueContract;
  expect(contract.type.kind).toBe(ValueKind.OBJECT);
  expect([...contract.type.fields!.keys()]).toEqual(["beta", "alpha"]);
  expect(typeEquals(contract.type.fields!.get("beta")!, beta.type)).toBe(true);
  expect(typeEquals(contract.type.fields!.get("alpha")!, alpha.type)).toBe(true);
  expect(contract.origins).toStrictEqual(
    new ContractOrigins({
      entries: [
        [["alpha"], origin("alpha-root")],
        [["alpha", "leaf"], origin("alpha-leaf")],
        [["beta"], origin("beta-root")],
      ],
    }),
  );
});

test("childSignature retains a partial contract when one field is opaque", () => {
  // Declared output shape and proven siblings form one partial codomain.
  const loaded = handChild({
    producers: { step: new ValueContract(INT) },
    output: { n: "${step.output + 1}", done: "${false}" },
  });

  const signature = childSignature(loaded);

  expect(isExactly(signature.output, PartialRecordContract)).toBe(true);
  const output = signature.output as PartialRecordContract;
  expect(output.names).toEqual(["n", "done"]);
  expect(output.fields[0]).toEqual(["n", null]);
  expect(output.fields[1]![1]).not.toBeNull();
  expect(signature.outputNames).toEqual(["n", "done"]);
  expect(signature.declaredOutputs[0]).toEqual(["n", null]);
  expect(signature.declaredOutputs[1]![0]).toBe("done");
  expect(typeEquals((signature.declaredOutputs[1]![1] as ValueContract).type, Type.scalar(ValueKind.BOOLEAN))).toBe(
    true,
  );
});

test("childSignature keeps a nested partial output opaque at a record field", () => {
  // A root partial can't falsely become an exact nested field contract.
  const nested = new PartialRecordContract([
    ["n", new ValueContract(INT)],
    ["note", null],
  ]);
  const loaded = handChild({
    producers: { inner: nested },
    output: { nested: "${inner.output}", done: "${false}" },
  });

  const signature = childSignature(loaded);

  expect(signature.declaredOutputs[0]![0]).toBe("nested");
  expect(signature.declaredOutputs[0]![1]).toBe(nested);
  expect(isExactly(signature.output, PartialRecordContract)).toBe(true);
  const output = signature.output as PartialRecordContract;
  expect(output.fields[0]).toEqual(["nested", null]);
  expect(output.fields[1]![0]).toBe("done");
  expect(output.fields[1]![1]).not.toBeNull();
});

test("childSignature rejects contradictory declared output evidence", () => {
  // The exported signature can't carry two incompatible output authorities.
  const boolean = new ValueContract(Type.scalar(ValueKind.BOOLEAN));

  expectError(
    () => new ChildSignature({ inputs: [], output: boolean, declaredOutputs: [["result", null]] }),
    RangeError,
    "child output contract contradicts its declared output evidence",
  );
});

test("composite builders preserve or deliberately wrap the child contract", () => {
  // CALL and LOOP keep the child's contract; MAP makes a new list contract with only the default
  // origin.
  const contract = new ValueContract(
    new Type({ kind: ValueKind.OBJECT, fields: new Map([["state", INT]]), required: new Set(["state"]) }),
    new ContractOrigins({ default: origin("child-root"), entries: [[["state"], origin("child-state")]] }),
  );
  const child = handChild({ producers: { producer: contract }, output: "${producer.output}" });
  const ctx = new BuildContext({ registry: new Map(), resolver: () => child });

  const call = new CallDescriptor({ id: "call", call: "child" }).build(ctx).node;
  const loop = new LoopDescriptor({
    id: "loop",
    call: "child",
    inputs: { state: "${input.state}" },
    times: 1n,
  }).build(ctx).node;
  const mapped = new MapDescriptor({ id: "mapped", call: "child", over: "${input.items}" }).build(ctx).node;

  expect(call.outputContract).toBe(contract);
  expect(loop.outputContract).toBe(contract);
  expect(isExactly(mapped.outputContract, ValueContract)).toBe(true);
  const listed = mapped.outputContract as ValueContract;
  expect(listed).not.toBe(contract);
  expect(listed.type.element).toBe(contract.type);
  expect(listed.origins).toStrictEqual(new ContractOrigins({ default: origin("child-root") }));
});

/** A CALL node `call` over one required child input `topic: str`, built with no wiring. */
function topicCall(): BuildResult {
  const childInput = new Parameter("topic", {
    contract: new ValueContract(STR),
    required: true,
    coerceStrings: true,
  });
  const node = new CallNode("call", { flowId: "child", childInputs: [childInput] });
  return new BuildResult({ node, wiring: new Map() });
}

test("the call type error is identical for raw and wrapped sources", () => {
  // Nullable-call compatibility reads the source expression, not the carrier.
  const nullableParent = new Type({ kind: ValueKind.STRING, nullable: true });
  const source = "${input.x}";

  const errors = [source, new Binding(source, { origin: origin("call-topic") })].map(
    (value) =>
      loadError(() =>
        checkCallMapTypes(
          new Map([["call", topicCall()]]),
          new Map(),
          new Map([["x", new ValueContract(nullableParent)]]),
          new Map([["call", new Map([["topic", value]])]]),
        ),
      ).message,
  );

  expect(errors[0]).toBe(errors[1]!);
  expect(errors[0]).toContain("child expects 'str', source is 'Optional(str)'");
});

test("the call type check projects a known partial producer field", () => {
  // A partial sibling cannot hide a proven source mismatch from a child call.
  const partial = new PartialRecordContract([
    ["n", new ValueContract(INT)],
    ["note", null],
  ]);

  const error = loadError(() =>
    checkCallMapTypes(
      new Map([["call", topicCall()]]),
      new Map([["producer", partial]]),
      new Map(),
      new Map([["call", new Map([["topic", "${producer.output.n}"]])]]),
    ),
  );

  expect(error.message).toContain("child expects 'str', source is 'int'");
});

test("the loop type check projects a known partial seed field", () => {
  // A partial sibling cannot hide a proven carried-field mismatch.
  const loopContract = new PartialRecordContract([
    ["n", new ValueContract(STR)],
    ["note", null],
  ]);
  const producerContract = new PartialRecordContract([
    ["n", new ValueContract(INT)],
    ["note", null],
  ]);
  const node = new LoopNode("loop", { flowId: "body" });
  node.outputContract = loopContract;
  const result = new BuildResult({ node, wiring: new Map() });

  const error = loadError(() =>
    checkLoopTypeContract(
      new Map([["loop", result]]),
      new Map([["producer", producerContract]]),
      new Map(),
      new Map([
        [
          "loop",
          new Map([
            ["n", "${producer.output.n}"],
            ["note", "opaque literal"],
          ]),
        ],
      ]),
    ),
  );

  expect(error.message).toMatch(/body output is 'str'.*source is 'int'/);
});

test("a loaded compose flow has only canonical binding storage", () => {
  // The YAML front end crosses into one uniform compiled representation.
  const { flow } = assembleText(`
id: canonical-bindings
name: Canonical bindings
input:
  topic: !str
nodes:
  emit:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: \${input.topic}
    output: str
output: \${emit.output}
`);

  const wiring = [...flow.compiled.wiring.values()].flatMap((nodeWiring) => [...nodeWiring.values()]);
  expect(flow.compiled.wiring.has(START_ID) && flow.compiled.wiring.has(END_ID)).toBe(true);
  expect(wiring.every((binding) => isExactly(binding, Binding))).toBe(true);
  expect(flow.compiled.outputs.every((output) => isExactly(output.binding, Binding))).toBe(true);
});

test("data edge inference is identical for raw and wrapped sources", () => {
  // Carrier provenance must not change producers, groups, or optionality.
  const source = "${producer.output}";
  const descriptors = new Map([["sink", new NodeDescriptor({ id: "sink" })]]);
  const raw = inferDataEdges(descriptors, new Map([["sink", new Map([["value", source]])]]));
  const wrapped = inferDataEdges(
    descriptors,
    new Map([["sink", new Map([["value", new Binding(source, { origin: origin("sink-value") })]])]]),
  );

  expect(wrapped).toStrictEqual(raw);
  expect(raw.map((edge) => [edge.from, edge.to, edge.inputGroup, edge.optional])).toEqual([
    ["producer", "sink", "value", false],
  ]);
});

test("the producer scan unwraps exactly one carrier layer", () => {
  // A nested carrier remains an opaque payload rather than executable wiring.
  const inner = new Binding("${producer.output}", { origin: origin("inner") });

  expect(bindingProducers(new Binding(inner, { origin: origin("outer") }))).toEqual([]);
});

test("the boundary reads a wrapped output but preserves it in END's wiring", () => {
  // Boundary dependencies use source semantics without discarding provenance.
  const wrapped = new Binding("${producer.output}", { origin: origin("flow-output") });
  const output = new FlowOutput("result", wrapped);

  const [, edges, wiring] = synthesizeBoundaryGraph([], [output], new Set(["producer"]), [], null, { record: false });

  expect(edges).toContainEqual(new Edge(`producer->${END_ID}#0`, "producer", END_ID, { inputGroup: "result" }));
  expect(wiring.get(END_ID)!.get("result")).toBe(wrapped);
});
