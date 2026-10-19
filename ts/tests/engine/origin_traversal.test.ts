/**
 * Origin traversal, rewrite and rebase over compiled flows (`compile/origins.ts`) and the local
 * origin hooks of the nodes: rows of `tests/engine/test_origin_traversal.py`.
 *
 * Ported: `test_iter_flow_origins_reaches_every_slot_once_and_is_repeatable`,
 * `test_parameter_origins_follow_the_carriers_canonical_order`,
 * `test_partial_output_contract_origins_traverse_and_rewrite`,
 * `test_case_predicate_origins_traverse_and_rewrite_by_arm_identity`,
 * `test_source_free_memory_decl_adds_no_origin`,
 * `test_each_composite_exposes_its_signature_and_baked_child`,
 * `test_rewrite_flow_origins_is_a_pure_alias_preserving_graph_copy`,
 * `test_traversal_and_rewrite_terminate_on_a_deliberate_child_cycle`,
 * `test_source_free_python_graph_ignores_origin_refs_in_semantic_payloads`,
 * `test_generated_origin_context_is_not_local_authored_provenance`,
 * `test_invalid_mutable_node_origin_fails_before_iteration_or_mapping`,
 * `test_rewrite_flow_origins_rejects_invalid_mapper_results`,
 * `test_rewrite_flow_origins_propagates_mapper_exceptions`,
 * `test_rebase_maps_equal_origins_once_and_routes_the_exact_old_identity`,
 * `test_rebase_rejects_missing_resolution_and_distinct_origin_collisions`,
 * `test_rebase_rejects_invalid_mapper_results`,
 * `test_rebase_source_free_graph_returns_an_independent_copy_and_empty_table` and
 * `test_rebase_retains_a_deliberate_child_cycle`.
 *
 * Changed:
 * - Python's bare `Node` is a minimal subclass, `Flow(...)` is `new Flow(nodes, [], {wiring,
 *   startId, endId})` over `Map`s, and keyword arguments are option objects. Python's
 *   `ValueError` is a `RangeError`.
 * - The generated-context, invalid-origin, composite and CASE rows also keep a local half that
 *   calls the node's own `iterLocalOrigins` / `rewriteLocalOrigins` / `originChildren`, rewriting
 *   a shallow copy and checking the original is unchanged.
 * - `test_shared_memory_decl_origin_traverses_and_rewrites_once` and
 *   `test_memory_decl_origin_rebase_routes_the_single_shared_site`: AGENT isn't ported, so the
 *   shared declaration sits in the `memories` of a root `CompiledFlow` and of its CALL child (flow
 *   `DeclSite`s), which exercises the same graph-wide carrier identity. A rebuilt declaration
 *   keeps the mapper's origin object, so the row checks its `origin.key`, not that its class is
 *   exactly `OriginRef`.
 * - `test_invalid_child_is_rejected_before_deepcopying_payloads`: the invalid child is a plain
 *   object (a TS copy runs no user hook); both APIs throw "origin child must be a Flow" and the
 *   mapper is never called.
 * - The equal-origins rebase row's `DerivedOrigin` is a subclass of `OriginRef`, and
 *   `table.origins == {x}` reads `table.size === 1 && table.has(x)`.
 * - `test_rebase_is_chainable_and_preserves_graph_identity_and_manifests`: the chaining half, over
 *   a hand-built CODE flow.
 * - `test_rebase_revalidates_origin_subclass_keys`: the invalid origin is forged with
 *   `Object.create(OriginRef.prototype)` and an empty `key`, since a subclass can't skip the base
 *   constructor's check.
 *
 * TS-only: the local traversal order with shared carriers yielded once; `rewriteLocalOrigins` on
 * a shallow node copy leaves the original unchanged and keeps shared carriers shared; a shallow
 * copy validates its assertions and runs; a payload is shared by identity between the original
 * and the rewritten copy; `rewriteFlowOrigins` on a `CompiledFlow` keeps `flowId` and
 * `flowLlmValueTypes` and rewrites a `Binding` `flowLlmConfig`'s origin; and the flow APIs'
 * argument checks (`iterFlowOrigins` rejects a non-flow eagerly).
 *
 * Later (T3-9): `test_agent_variable_origins_rewrite_shared_carriers_once`.
 *
 * Later (T3-21): `test_rewriting_origins_preserves_manifest_and_stable_identity`, and the
 * manifest and fingerprint lines of the chaining rebase row.
 *
 * Not ported: `test_rewrite_preserves_payload_back_references_to_frozen_graph_objects` (payloads
 * holding carriers must be acyclic, and payloads are shared, not copied) and
 * `test_rewrite_rebuilds_payload_hash_containers_after_origin_changes` (there are no hash
 * containers keyed by carriers, and the rewrite makes no second copy).
 */

import { expect, test } from "bun:test";

import { CompiledFlow, END_ID, Edge, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { iterFlowOrigins, rebaseFlowOrigins, rewriteFlowOrigins } from "../../src/compile/origins.ts";
import { DiagnosticFrame, OriginRef, ResolverTable, SourceRange } from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import {
  Assertion,
  Binding,
  ContractOrigins,
  Parameter,
  PartialRecordContract,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

/** A minimal concrete node; the base hooks are the subject. Its body lets a copy run. */
class BareNode extends Node {
  readonly kind = NodeKind.CODE;

  override forward(): NodeResult {
    return new Output("done");
  }
}

/** Return one opaque source identity with a readable test key. */
function origin(key: string): OriginRef {
  return new OriginRef(key);
}

/** Return a node whose output contract and param share one contract, and whose asserts share one. */
function richNode(): { node: BareNode; contract: ValueContract; shared: Assertion } {
  const contract = new ValueContract(
    Type.scalar(ValueKind.STRING),
    new ContractOrigins({ default: origin("contract"), entries: [[["leaf"], origin("contract/leaf")]] }),
  );
  const node = new BareNode("n", { outputContract: contract, origin: origin("node") });
  const shared = new Assertion("x > 0", "x is positive", { origin: origin("shared") });
  node.params = [
    new Parameter("x", {
      contract,
      origin: origin("param"),
      hasDefault: true,
      default: 1n,
      defaultOrigin: origin("param/default"),
    }),
  ];
  node.preAsserts = [new Assertion("x < 9", "x is small", { origin: origin("pre") }), shared];
  node.postAsserts = [shared, new Assertion("output != ''", "output is set", { origin: origin("post") })];
  return { node, contract, shared };
}

/** Return a shallow copy of `node`: the same prototype, its own fields copied. */
function shallowCopy<N extends Node>(node: N): N {
  return Object.assign(Object.create(Object.getPrototypeOf(node)) as N, node);
}

/** One CALL, MAP and LOOP factory each over a baked child and its cached signature. */
const COMPOSITES = [
  [
    "call",
    (child: unknown, signature: Parameter) =>
      new CallNode("call", { flowId: "child", child, childInputs: [signature] }),
  ],
  [
    "map",
    (child: unknown, signature: Parameter) => new MapNode("map", { flowId: "child", child, childInputs: [signature] }),
  ],
  [
    "loop",
    (child: unknown, signature: Parameter) =>
      new LoopNode("loop", {
        flowId: "child",
        child,
        childInputs: [signature],
        predicateKind: "times",
        times: 1,
        maxIters: 1,
      }),
  ],
] as const;

test("generated origin context is not local authored provenance (local half)", () => {
  const node = new BareNode("child", { origin: origin("child/node") });
  const ancestry = [origin("outer/call")];
  node.originContext = ancestry;
  const visited: OriginRef[] = [];

  expect([...node.iterLocalOrigins()]).toEqual([origin("child/node")]);

  node.rewriteLocalOrigins((o) => {
    visited.push(o);
    return origin(`mapped/${o.key}`);
  });

  expect(visited).toEqual([origin("child/node")]);
  expect(node.origin).toEqual(origin("mapped/child/node"));
  expect(node.originContext).toBe(ancestry);
  expect(ancestry).toEqual([origin("outer/call")]);
});

test("invalid mutable node origin fails before iteration or mapping (local half)", () => {
  const node = new BareNode("invalid");
  (node as { origin: unknown }).origin = "not-an-origin";

  expect(() => [...node.iterLocalOrigins()]).toThrow(TypeError);
  expect(() => [...node.iterLocalOrigins()]).toThrow(/node origin must be an OriginRef/);
  expect(() => node.rewriteLocalOrigins((o) => o)).toThrow(TypeError);
  expect(() => node.rewriteLocalOrigins((o) => o)).toThrow(/node origin must be an OriginRef/);
});

test("local origins come in structural order, each shared carrier once", () => {
  const { node } = richNode();

  expect([...node.iterLocalOrigins()].map((o) => o.key)).toEqual([
    "node",
    "contract",
    "contract/leaf",
    "param",
    "param/default",
    "pre",
    "shared",
    "post",
  ]);
});

test("a graph traversal's seen set skips carriers it already visited", () => {
  const { node, contract, shared } = richNode();

  expect([...node.iterLocalOrigins(new Set([contract, shared]))].map((o) => o.key)).toEqual([
    "node",
    "param",
    "param/default",
    "pre",
    "post",
  ]);
});

test("rewriting a shallow copy leaves the original and keeps shared carriers shared", () => {
  const { node, contract, shared } = richNode();
  const { params, preAsserts, postAsserts } = node;
  const copy = shallowCopy(node);

  copy.rewriteLocalOrigins((o) => origin(`mapped/${o.key}`));

  expect([...node.iterLocalOrigins()].map((o) => o.key)).toEqual([
    "node",
    "contract",
    "contract/leaf",
    "param",
    "param/default",
    "pre",
    "shared",
    "post",
  ]);
  expect(node.outputContract).toBe(contract);
  expect(node.params).toBe(params);
  expect(node.preAsserts).toBe(preAsserts);
  expect(node.postAsserts).toBe(postAsserts);
  expect(params![0]!.contract).toBe(contract);
  expect(contract.origins.default).toEqual(origin("contract"));
  expect(shared.origin).toEqual(origin("shared"));

  expect([...copy.iterLocalOrigins()].map((o) => o.key)).toEqual([
    "mapped/node",
    "mapped/contract",
    "mapped/contract/leaf",
    "mapped/param",
    "mapped/param/default",
    "mapped/pre",
    "mapped/shared",
    "mapped/post",
  ]);
  expect(copy.outputContract).not.toBe(contract);
  expect(copy.params![0]!.contract).toBe(copy.outputContract as ValueContract);
  expect(copy.preAsserts[1]).toBe(copy.postAsserts[0]!);
  expect(copy.preAsserts[1]).not.toBe(shared);
});

test("a shallow copy validates its assertions and runs", async () => {
  const { node } = richNode();
  const copy = shallowCopy(node);

  expect(copy.validateAssertions()).toBeUndefined();
  expect(await copy.run({ x: 5n })).toEqual(new Output("done"));
  copy.preAsserts = [new Assertion("x >", "broken")];
  expect(() => copy.validateAssertions()).toThrow(/^node 'n' pre-assert 'broken' \(`x >`\) is not/);
});

for (const [label, make] of COMPOSITES) {
  test(`each composite exposes its signature and baked child (${label}, local half)`, () => {
    const signature = new Parameter("item", { origin: origin("signature") });
    const child = {};
    const driver = make(child, signature);
    driver.origin = origin("driver");

    expect([...driver.iterLocalOrigins()].map((o) => o.key)).toEqual(["driver", "signature"]);
    expect(driver.originChildren()).toEqual([child]);
    expect(driver.originChildren()[0]).toBe(child);

    const copy = shallowCopy(driver);
    copy.rewriteLocalOrigins((o) => origin(`mapped/${o.key}`));

    expect([...copy.iterLocalOrigins()].map((o) => o.key)).toEqual(["mapped/driver", "mapped/signature"]);
    expect(copy.childInputs[0]).not.toBe(signature);
    expect([...driver.iterLocalOrigins()].map((o) => o.key)).toEqual(["driver", "signature"]);
    expect(driver.childInputs[0]).toBe(signature);
    expect(signature.origin).toEqual(origin("signature"));
  });
}

test("case predicate origins traverse and rewrite by arm identity (local half)", () => {
  // Ordered CASE sites participate in rewrites without double-mapping aliases.
  const shared = new Case("first", { when: "false", origin: origin("case/shared") });
  const second = new Case("second", { when: "true", origin: origin("case/second") });
  const node = new CaseNode("case", [shared, shared, second]);
  const cases = node.cases;

  expect([...node.iterLocalOrigins()].map((o) => o.key)).toEqual(["case/shared", "case/second"]);

  const calls: string[] = [];
  const copy = shallowCopy(node);
  copy.rewriteLocalOrigins((o) => {
    calls.push(o.key);
    return origin(`mapped/${o.key}`);
  });

  const rewritten = copy.cases;
  expect(rewritten[0]).toBe(rewritten[1]);
  expect(rewritten[0]!.origin).toEqual(origin("mapped/case/shared"));
  expect(rewritten[2]!.origin).toEqual(origin("mapped/case/second"));
  expect(calls).toEqual(["case/shared", "case/second"]);
  expect(shared.origin).toEqual(origin("case/shared"));
  expect(node.cases).toBe(cases);
  expect(node.cases[0]).toBe(shared);
  expect(node.cases[1]).toBe(shared);
  expect(node.cases[2]).toBe(second);
});

// The flow API: `iterFlowOrigins`, `rewriteFlowOrigins` and `rebaseFlowOrigins`.

/** Return the keys of `origins`, in order. */
function keys(origins: Iterable<OriginRef>): string[] {
  return [...origins].map((o) => o.key);
}

/** Return a flow of one node that is both its entry and its result, with that node's wiring. */
function soloFlow(node: Node, wiring: ReadonlyMap<string, unknown> = new Map()): Flow {
  return new Flow(new Map([[node.id, node]]), [], {
    wiring: new Map([[node.id, wiring]]),
    startId: node.id,
    endId: node.id,
  });
}

/** Return a resolver that resolves nothing. */
function nullResolver(): { resolve(origin: OriginRef): DiagnosticFrame | null } {
  return { resolve: () => null };
}

/** Return a contract carrying both root and nested provenance. */
function contractOf(prefix: string): ValueContract {
  return new ValueContract(
    Type.scalar(ValueKind.STRING),
    new ContractOrigins({ default: origin(prefix), entries: [[["leaf"], origin(`${prefix}/leaf`)]] }),
  );
}

/** Return a parameter carrying declaration, default, and contract origins. */
function parameterOf(
  name: string,
  prefix: string,
  options: { contract?: ValueContract; default?: unknown } = {},
): Parameter {
  const { contract = contractOf(`${prefix}/contract`), default: fallback = "fallback" } = options;
  return new Parameter(name, {
    contract,
    hasDefault: true,
    default: fallback,
    origin: origin(prefix),
    defaultOrigin: origin(`${prefix}/default`),
  });
}

/** Build all current origin slots with deliberate carrier and child aliases. */
function richFlow(): { flow: CompiledFlow; expected: Set<string>; payloadOrigin: OriginRef; payload: object } {
  const payloadOrigin = origin("payload/not-provenance");
  const payload = { opaque: payloadOrigin };

  const body = new BareNode("body", { origin: origin("root/body/node") });
  body.params = [parameterOf("value", "root/body/parameter", { default: payload })];
  body.preAsserts = [new Assertion("true", "pre", { origin: origin("root/body/pre") })];
  body.postAsserts = [new Assertion("true", "post", { origin: origin("root/body/post") })];
  body.outputContract = contractOf("root/body/output");
  body.env = { opaque: payloadOrigin };

  const signatureContract = contractOf("child/signature/contract");
  const signature = parameterOf("item", "child/signature/parameter", { contract: signatureContract });
  const childStart = new StartNode(START_ID, { params: [signature] });
  childStart.origin = origin("child/start/node");
  const childBody = new BareNode("child_body", {
    outputContract: signatureContract,
    origin: origin("child/body/node"),
  });
  const childEnd = EndNode.value(END_ID, { outputName: "result" });
  childEnd.origin = origin("child/end/node");
  const childOutput = new Binding("${child_body.output}", { origin: origin("child/output") });
  const child = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, childStart],
      ["child_body", childBody],
      [END_ID, childEnd],
    ]),
    [],
    {
      outputs: [new FlowOutput("result", childOutput)],
      wiring: new Map([
        [START_ID, new Map()],
        [
          "child_body",
          new Map([["item", new Binding("${__start__.output.item}", { origin: origin("child/wiring") })]]),
        ],
        [END_ID, new Map([["result", childOutput]])],
      ]),
    },
  );

  const [call, mapped, loop] = COMPOSITES.map(([, make]) => make(child, signature));
  call!.origin = origin("root/call/node");
  mapped!.origin = origin("root/map/node");
  loop!.origin = origin("root/loop/node");

  const rootOutput = new Binding("${body.output}", { origin: origin("root/output") });
  const flow = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: [] })],
      ["body", body],
      ["call", call!],
      ["map", mapped!],
      ["loop", loop!],
      [END_ID, EndNode.value(END_ID, { outputName: "result" })],
    ]),
    [],
    {
      outputs: [new FlowOutput("result", rootOutput)],
      wiring: new Map<string, ReadonlyMap<string, unknown>>([
        [START_ID, new Map()],
        ["body", new Map([["value", new Binding(payload, { origin: origin("root/wiring") })]])],
        ["call", new Map()],
        ["map", new Map()],
        ["loop", new Map()],
        [END_ID, new Map([["result", rootOutput]])],
      ]),
    },
  );

  const expected = new Set([
    "root/wiring",
    "root/output",
    "root/body/node",
    "root/body/parameter/contract",
    "root/body/parameter/contract/leaf",
    "root/body/parameter",
    "root/body/parameter/default",
    "root/body/pre",
    "root/body/post",
    "root/body/output",
    "root/body/output/leaf",
    "root/call/node",
    "root/map/node",
    "root/loop/node",
    "child/signature/contract",
    "child/signature/contract/leaf",
    "child/signature/parameter",
    "child/signature/parameter/default",
    "child/start/node",
    "child/body/node",
    "child/end/node",
    "child/wiring",
    "child/output",
  ]);
  return { flow, expected, payloadOrigin, payload };
}

test("iter flow origins reaches every slot once and is repeatable", () => {
  // Traversal is structural, deterministic, opaque to payloads, and alias-safe.
  const { flow, expected } = richFlow();

  const first = [...iterFlowOrigins(flow)];
  const second = [...iterFlowOrigins(flow)];

  expect(first).toEqual(second);
  expect(first.every((o, i) => o === second[i])).toBe(true);
  expect(new Set(keys(first))).toEqual(expected);
  expect(first.length).toBe(expected.size);
  expect(keys(first)).not.toContain("payload/not-provenance");
});

test("parameter origins follow the carrier's canonical order", () => {
  // A parameter visits contract root/entries before its own declaration sites.
  const parameter = new Parameter("value", {
    contract: new ValueContract(
      Type.scalar(ValueKind.STRING),
      new ContractOrigins({
        default: origin("contract/root"),
        entries: [
          [["a"], origin("contract/a")],
          [["z"], origin("contract/z")],
        ],
      }),
    ),
    hasDefault: true,
    default: "fallback",
    origin: origin("parameter"),
    defaultOrigin: origin("parameter/default"),
  });
  const node = new BareNode("only");
  node.params = [parameter];

  expect(keys(iterFlowOrigins(soloFlow(node)))).toEqual([
    "contract/root",
    "contract/a",
    "contract/z",
    "parameter",
    "parameter/default",
  ]);
});

test("partial output contract origins traverse and rewrite", () => {
  // Only proven fields contribute origins through the common node seam.
  const node = new BareNode("only", {
    outputContract: new PartialRecordContract([
      [
        "known",
        new ValueContract(Type.scalar(ValueKind.INTEGER), new ContractOrigins({ default: origin("output/known") })),
      ],
      ["opaque", null],
    ]),
  });
  const flow = soloFlow(node);

  expect(keys(iterFlowOrigins(flow))).toEqual(["output/known"]);
  const rewritten = rewriteFlowOrigins(flow, (o) => origin(`mapped/${o.key}`));
  const contract = rewritten.nodes.get("only")!.outputContract as PartialRecordContract;
  expect(contract.constructor).toBe(PartialRecordContract);
  expect(contract.fields[0]![1]!.origins.default).toEqual(origin("mapped/output/known"));
  expect((node.outputContract as PartialRecordContract).fields[0]![1]!.origins.default).toEqual(origin("output/known"));
});

test("case predicate origins traverse and rewrite by arm identity (flow half)", () => {
  // Ordered CASE sites participate in graph rewrites without double-mapping aliases.
  const shared = new Case("first", { when: "false", origin: origin("case/shared") });
  const second = new Case("second", { when: "true", origin: origin("case/second") });
  const node = new CaseNode("case", [shared, shared, second]);
  const flow = soloFlow(node);

  expect(keys(iterFlowOrigins(flow))).toEqual(["case/shared", "case/second"]);

  const calls: string[] = [];
  const rewritten = rewriteFlowOrigins(flow, (o) => {
    calls.push(o.key);
    return origin(`mapped/${o.key}`);
  });

  const rewrittenCases = (rewritten.nodes.get("case") as CaseNode).cases;
  expect(rewrittenCases[0]).toBe(rewrittenCases[1]!);
  expect(rewrittenCases[0]!.origin).toEqual(origin("mapped/case/shared"));
  expect(rewrittenCases[2]!.origin).toEqual(origin("mapped/case/second"));
  expect(calls).toEqual(["case/shared", "case/second"]);
  expect(shared.origin).toEqual(origin("case/shared"));
});

/** Return a flow-site reflective declaration with `origin`. */
function sharedDecl(declOrigin: OriginRef | null): MemoryDecl {
  return new MemoryDecl("reflective", "reflective", { max_items: 4n }, new DeclSite([], ["root"]), declOrigin);
}

/** Return a root compiled flow and its CALL child that both declare `shared`. */
function aliasedFlow(shared: MemoryDecl): CompiledFlow {
  const child = new CompiledFlow(new Map(), [], { memories: [shared] });
  const call = new CallNode("call", { flowId: "child", child });
  return new CompiledFlow(new Map([["call", call]]), [], {
    wiring: new Map([["call", new Map()]]),
    memories: [shared],
  });
}

/** Return the memories of `flow` and of its CALL child. */
function aliasedMemories(flow: CompiledFlow): [MemoryDecl, MemoryDecl] {
  const child = (flow.nodes.get("call") as CallNode).child as CompiledFlow;
  return [flow.memories[0]!, child.memories[0]!];
}

test("shared memory decl origin traverses and rewrites once", () => {
  // Two flows may deliberately alias one immutable declaration carrier.
  const shared = sharedDecl(origin("policy/shared"));
  const flow = aliasedFlow(shared);

  expect([...iterFlowOrigins(flow)]).toEqual([origin("policy/shared")]);
  const calls: OriginRef[] = [];

  class DerivedOrigin extends OriginRef {}

  const rewritten = rewriteFlowOrigins(flow, (o) => {
    calls.push(o);
    return new DerivedOrigin(`mapped/${o.key}`);
  });

  expect(calls).toEqual([origin("policy/shared")]);
  const [root, child] = aliasedMemories(rewritten);
  expect(root).toBe(child);
  expect(root).not.toBe(shared);
  expect(root.origin!.key).toBe("mapped/policy/shared");
  expect(aliasedMemories(flow)).toEqual([shared, shared]);
  expect(aliasedMemories(flow)[0]).toBe(shared);
  expect(shared.origin).toEqual(origin("policy/shared"));
});

test("source-free memory decl adds no origin", () => {
  expect([...iterFlowOrigins(aliasedFlow(sharedDecl(null)))]).toEqual([]);
});

test("memory decl origin rebase routes the single shared site", () => {
  const shared = sharedDecl(origin("policy/shared"));
  const flow = aliasedFlow(shared);
  const frame = new DiagnosticFrame({
    label: "memory.yaml",
    text: "long_term_memory: reflective\n",
    range: new SourceRange(1, 19, 1, 29),
  });
  const resolver = { resolve: (o: OriginRef) => (o === shared.origin ? frame : null) };

  const mapped: OriginRef[] = [];
  const [rebased, table] = rebaseFlowOrigins(
    flow,
    (o) => {
      mapped.push(o);
      return origin(`alias/${o.key}`);
    },
    { resolver },
  );

  expect(mapped).toEqual([origin("policy/shared")]);
  const [root, child] = aliasedMemories(rebased);
  expect(root).toBe(child);
  expect(table.resolve(origin("alias/policy/shared"))).toBe(frame);
});

for (const [label, make] of COMPOSITES) {
  test(`each composite exposes its signature and baked child (${label}, flow half)`, () => {
    // Every higher-order node delegates both local caches and recursive children.
    const signature = new Parameter("item", { origin: origin("signature") });
    const childStart = new StartNode(START_ID, { params: [signature] });
    childStart.origin = origin("child");
    const child = CompiledFlow.fromParts(new Map([[START_ID, childStart]]), [], {
      wiring: new Map([[START_ID, new Map()]]),
    });
    const driver = make(child, signature);
    const flow = soloFlow(driver);

    expect(new Set(keys(iterFlowOrigins(flow)))).toEqual(new Set(["signature", "child"]));

    const rewritten = rewriteFlowOrigins(flow, (o) => origin(`mapped/${o.key}`));
    const rewrittenDriver = rewritten.nodes.get(driver.id) as typeof driver;
    const rewrittenChild = rewrittenDriver.child as CompiledFlow;
    const rewrittenSignature = rewrittenChild.nodes.get(START_ID)!.params![0];
    expect(rewrittenDriver.childInputs[0]).toBe(rewrittenSignature!);
    expect(new Set(keys(iterFlowOrigins(rewritten)))).toEqual(new Set(["mapped/signature", "mapped/child"]));
  });
}

test("rewrite flow origins is a pure alias-preserving graph copy", () => {
  // One rewrite maps shared carriers once while preserving every copied alias.
  const { flow, expected, payloadOrigin } = richFlow();
  const before = [...iterFlowOrigins(flow)];
  const calls: OriginRef[] = [];

  const rewritten = rewriteFlowOrigins(flow, (o) => {
    calls.push(o);
    return origin(`mapped/${o.key}`);
  });

  expect(rewritten).not.toBe(flow);
  expect([...iterFlowOrigins(flow)]).toEqual(before);
  expect(new Set(keys(calls))).toEqual(expected);
  expect(calls.length).toBe(expected.size);
  expect(new Set(keys(iterFlowOrigins(rewritten)))).toEqual(new Set([...expected].map((k) => `mapped/${k}`)));

  expect(rewritten.outputs[0]!.binding).toBe(rewritten.wiring.get(END_ID)!.get("result")!);
  expect(flow.outputs[0]!.binding).toBe(flow.wiring.get(END_ID)!.get("result")!);

  const childOf = (f: Flow, id: string) => (f.nodes.get(id) as CallNode).child as CompiledFlow;
  const rewrittenChild = childOf(rewritten, "call");
  expect(rewrittenChild).not.toBe(childOf(flow, "call"));
  expect(childOf(rewritten, "map")).toBe(rewrittenChild);
  expect(childOf(rewritten, "loop")).toBe(rewrittenChild);
  const rewrittenSignature = rewrittenChild.nodes.get(START_ID)!.params![0]!;
  for (const id of ["call", "map", "loop"]) {
    expect((rewritten.nodes.get(id) as CallNode).childInputs[0]).toBe(rewrittenSignature);
  }
  expect(rewrittenSignature.contract).toBe(rewrittenChild.nodes.get("child_body")!.outputContract as ValueContract);
  expect(rewrittenChild.outputs[0]!.binding).toBe(rewrittenChild.wiring.get(END_ID)!.get("result")!);

  const rewrittenPayload = rewritten.wiring.get("body")!.get("value")!.source as { opaque: OriginRef };
  expect(rewrittenPayload).toBe(rewritten.nodes.get("body")!.params![0]!.default as { opaque: OriginRef });
  expect(rewrittenPayload.opaque).toBe(payloadOrigin);
  expect(rewrittenPayload.opaque.key).toBe("payload/not-provenance");
  expect((rewritten.nodes.get("body")!.env["opaque"] as OriginRef).key).toBe("payload/not-provenance");
});

test("a rewrite shares payloads by identity with the original", () => {
  const { flow, payload } = richFlow();

  const rewritten = rewriteFlowOrigins(flow, (o) => origin(`mapped/${o.key}`));

  expect(rewritten.wiring.get("body")!.get("value")!.source).toBe(payload);
  expect(rewritten.nodes.get("body")!.params![0]!.default).toBe(payload);
  expect(rewritten.nodes.get("body")!.env).toBe(flow.nodes.get("body")!.env);
});

test("rewriting a compiled flow keeps its identity and value types and rewrites a binding llm config", () => {
  const llmConfig = new Binding("${__start__.output.llm}", { origin: origin("llm") });
  const flow = new CompiledFlow(new Map(), [], { flowId: "f", flowLlmConfig: llmConfig });
  const valueTypes = [[[], null]] as const;
  flow.flowLlmValueTypes = valueTypes;

  expect(keys(iterFlowOrigins(flow))).toEqual(["llm"]);
  const rewritten = rewriteFlowOrigins(flow, (o) => origin(`mapped/${o.key}`));

  expect(rewritten).toBeInstanceOf(CompiledFlow);
  expect(rewritten.flowId).toBe("f");
  expect(rewritten.flowLlmValueTypes).toBe(valueTypes);
  expect(rewritten.flowLlmConfig).toBeInstanceOf(Binding);
  expect((rewritten.flowLlmConfig as Binding).origin).toEqual(origin("mapped/llm"));
  expect((rewritten.flowLlmConfig as Binding).source).toBe(llmConfig.source);
  expect(flow.flowLlmConfig).toBe(llmConfig);
  expect(llmConfig.origin).toEqual(origin("llm"));
});

/** Return a flow whose only node is a CALL whose child is the flow itself. */
function cycleFlow(): Flow {
  const cycle = new Flow(new Map(), [], { startId: "self", endId: "self" });
  const driver = new CallNode("self", { flowId: "self", child: cycle });
  driver.origin = origin("cycle/node");
  cycle.nodes.set("self", driver);
  cycle.wiring.set("self", new Map());
  return cycle;
}

test("traversal and rewrite terminate on a deliberate child cycle", () => {
  // Identity guards visit a recursive object graph once and retain its cycle.
  const cycle = cycleFlow();

  expect([...iterFlowOrigins(cycle)]).toEqual([origin("cycle/node")]);

  const calls: OriginRef[] = [];
  const rewritten = rewriteFlowOrigins(cycle, (o) => {
    calls.push(o);
    return origin(`mapped/${o.key}`);
  });

  expect(calls).toEqual([origin("cycle/node")]);
  expect(rewritten).not.toBe(cycle);
  const driver = rewritten.nodes.get("self") as CallNode;
  expect(driver.child).toBe(rewritten);
  expect(driver.origin).toEqual(origin("mapped/cycle/node"));
  expect((cycle.nodes.get("self") as CallNode).child).toBe(cycle);
  expect(cycle.nodes.get("self")!.origin).toEqual(origin("cycle/node"));
});

test("source-free graph ignores origin refs in semantic payloads", () => {
  // Only declared provenance slots are visited; arbitrary values remain opaque.
  const payloadOrigin = origin("ordinary/value");
  const payload = { "origin-shaped": payloadOrigin };
  const node = new BareNode("python");
  node.params = [new Parameter("value", { hasDefault: true, default: payload })];
  node.env = { ordinary: payloadOrigin };
  const flow = CompiledFlow.fromParts(new Map([["python", node]]), [], {
    outputs: [new FlowOutput("result", payloadOrigin)],
    wiring: new Map([["python", new Map([["value", payload]])]]),
  });

  expect([...iterFlowOrigins(flow)]).toEqual([]);

  const rewritten = rewriteFlowOrigins(flow, () => {
    throw new Error("semantic payload was mistaken for provenance");
  });

  expect(rewritten).not.toBe(flow);
  expect(rewritten.nodes.get("python")).not.toBe(node);
  const source = rewritten.wiring.get("python")!.get("value")!.source as typeof payload;
  expect(source["origin-shaped"]).toBe(payloadOrigin);
  expect(rewritten.outputs[0]!.binding.source).toBe(payloadOrigin);
  expect((rewritten.nodes.get("python")!.params![0]!.default as typeof payload)["origin-shaped"]).toBe(payloadOrigin);
  expect(rewritten.nodes.get("python")!.env["ordinary"]).toBe(payloadOrigin);
});

test("generated origin context is not local authored provenance (flow half)", () => {
  // Authored-origin traversal neither visits nor rewrites generated ancestry.
  const node = new BareNode("child", { origin: origin("child/node") });
  node.originContext = [origin("outer/call")];
  const flow = soloFlow(node);
  const visited: OriginRef[] = [];

  expect([...iterFlowOrigins(flow)]).toEqual([origin("child/node")]);

  const rewritten = rewriteFlowOrigins(flow, (o) => {
    visited.push(o);
    return origin(`mapped/${o.key}`);
  });

  expect(visited).toEqual([origin("child/node")]);
  expect(rewritten.nodes.get("child")!.origin).toEqual(origin("mapped/child/node"));
  expect(rewritten.nodes.get("child")!.originContext).toEqual([origin("outer/call")]);
  expect(node.originContext).toEqual([origin("outer/call")]);
});

test("invalid mutable node origin fails before iteration or mapping (flow half)", () => {
  // A directly corrupted node cannot leak a non-origin through the API.
  const node = new BareNode("invalid");
  (node as { origin: unknown }).origin = "not-an-origin";
  const flow = soloFlow(node);

  expect(() => [...iterFlowOrigins(flow)]).toThrow(TypeError);
  expect(() => [...iterFlowOrigins(flow)]).toThrow(/node origin must be an OriginRef/);
  expect(() => rewriteFlowOrigins(flow, (o) => o)).toThrow(TypeError);
  expect(() => rewriteFlowOrigins(flow, (o) => o)).toThrow(/node origin must be an OriginRef/);
});

test("invalid child is rejected before copying or mapping", () => {
  // Structural child errors win over any copy or mapper call.
  const flow = soloFlow(new CallNode("call", { flowId: "invalid", child: {} }));
  const calls: OriginRef[] = [];
  const mapper = (o: OriginRef) => {
    calls.push(o);
    return o;
  };

  expect(() => [...iterFlowOrigins(flow)]).toThrow(TypeError);
  expect(() => [...iterFlowOrigins(flow)]).toThrow("node 'call' origin child must be a Flow, got Object");
  expect(() => rewriteFlowOrigins(flow, mapper)).toThrow(TypeError);
  expect(() => rewriteFlowOrigins(flow, mapper)).toThrow(/origin child must be a Flow/);
  expect(calls).toEqual([]);
});

test("the flow APIs reject a non-flow eagerly and a non-callable mapper", () => {
  expect(() => iterFlowOrigins({} as Flow)).toThrow(new TypeError("flow must be a Flow, got Object"));
  expect(() => rewriteFlowOrigins(null as unknown as Flow, (o) => o)).toThrow("flow must be a Flow, got object");
  const flow = soloFlow(new BareNode("only"));
  expect(() => rewriteFlowOrigins(flow, "x" as unknown as (o: OriginRef) => OriginRef)).toThrow(
    new TypeError("origin mapper must be callable, got string"),
  );
  expect(() => rebaseFlowOrigins(flow, (o) => o, { resolver: {} as ResolverTable })).toThrow(
    new TypeError("origin resolver must implement resolve"),
  );
});

for (const invalid of [null, "not-an-origin"]) {
  test(`rewrite flow origins rejects invalid mapper results (${String(invalid)})`, () => {
    // A present origin cannot be erased or replaced with another value type.
    const binding = new Binding(1n, { origin: origin("binding") });
    const flow = soloFlow(new BareNode("only"), new Map([["value", binding]]));

    expect(() => rewriteFlowOrigins(flow, () => invalid as unknown as OriginRef)).toThrow(TypeError);
    expect(() => rewriteFlowOrigins(flow, () => invalid as unknown as OriginRef)).toThrow(
      /origin mapper must return an OriginRef/,
    );

    expect(flow.wiring.get("only")!.get("value")).toBe(binding);
    expect(binding.origin).toEqual(origin("binding"));
  });
}

test("rewrite flow origins propagates mapper exceptions", () => {
  // Mapper failures are loud and cannot partially mutate the authored graph.
  class MappingError extends Error {}
  const binding = new Binding(1n, { origin: origin("binding") });
  const flow = soloFlow(new BareNode("only"), new Map([["value", binding]]));

  expect(() =>
    rewriteFlowOrigins(flow, () => {
      throw new MappingError("cannot rewrite");
    }),
  ).toThrow(MappingError);

  expect(flow.wiring.get("only")!.get("value")).toBe(binding);
  expect(binding.origin).toEqual(origin("binding"));
});

test("rebase maps equal origins once and routes the exact old identity", () => {
  // Equal keys share one new identity and route across distinct carrier classes.
  const old = origin("opaque/~and/slash");
  const binding = new Binding(1n, { origin: old });

  class DerivedOrigin extends OriginRef {}

  const node = new BareNode("only", { origin: new DerivedOrigin("opaque/~and/slash") });
  const flow = soloFlow(node, new Map([["value", binding]]));
  const frame = new DiagnosticFrame({ label: "memory", text: "value\n", range: new SourceRange(1, 1, 1, 6) });
  const resolved: OriginRef[] = [];
  const resolver = {
    resolve(o: OriginRef): DiagnosticFrame | null {
      resolved.push(o);
      return o === old ? frame : null;
    },
  };
  const mapped: OriginRef[] = [];

  const [rebased, table] = rebaseFlowOrigins(
    flow,
    (o) => {
      mapped.push(o);
      return origin("scope/~opaque/key");
    },
    { resolver },
  );

  expect(mapped.length).toBe(1);
  expect(mapped[0]).toBe(old);
  const before = [...iterFlowOrigins(flow)];
  expect(before[0]).toBe(old);
  expect(before[1]).toBe(node.origin!);
  expect(keys(iterFlowOrigins(rebased))).toEqual(["scope/~opaque/key", "scope/~opaque/key"]);
  expect(table.size === 1 && table.has(origin("scope/~opaque/key"))).toBe(true);
  expect(table.resolve(origin("scope/~opaque/key"))).toBe(frame);
  expect(resolved.length).toBe(1);
  expect(resolved[0]).toBe(old);
});

/** Return a hand-built CODE flow whose origins cover every carrier. */
function codeFlow(): CompiledFlow {
  const inputContract = contractOf("code/input/contract");
  const start = new StartNode(START_ID, {
    params: [new Parameter("value", { contract: inputContract, required: true, origin: origin("code/input") })],
  });
  start.origin = origin("code/start");
  const body = new CodeNode("body", { code: 'return inputs["value"]' });
  body.origin = origin("code/body");
  body.params = [
    new Parameter("value", { contract: inputContract, required: true, origin: origin("code/body/input") }),
  ];
  body.outputContract = contractOf("code/output/contract");
  body.preAsserts = [new Assertion("value >= 0", "value >= 0", { origin: origin("code/pre") })];
  body.postAsserts = [new Assertion("output >= 0", "output >= 0", { origin: origin("code/post") })];
  const end = EndNode.value(END_ID, { outputName: "result" });
  end.origin = origin("code/end");
  const output = new Binding("${body.output}", { origin: origin("code/output") });
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, start],
      ["body", body],
      [END_ID, end],
    ]),
    [
      new Edge("start->body", START_ID, "body", { inputGroup: "value" }),
      new Edge("body->end", "body", END_ID, { inputGroup: "result" }),
    ],
    {
      outputs: [new FlowOutput("result", output)],
      wiring: new Map<string, ReadonlyMap<string, unknown>>([
        [START_ID, new Map([["value", new Binding(1n, { origin: origin("code/start/value") })]])],
        ["body", new Map([["value", new Binding("${__start__.output.value}", { origin: origin("code/body/value") })]])],
        [END_ID, new Map([["result", output]])],
      ]),
    },
  );
}

test("rebase is chainable and preserves graph identity", () => {
  // A later namespace delegates through the earlier table without graph mutation.
  const flow = codeFlow();
  const originalOrigins = [...iterFlowOrigins(flow)];
  const resolver = {
    resolve: (o: OriginRef) =>
      new DiagnosticFrame({ label: o.key, text: "value\n", range: new SourceRange(1, 1, 1, 6) }),
  };

  const [first, firstTable] = rebaseFlowOrigins(flow, (o) => origin(`first/${o.key}`), { resolver });
  const firstOrigins = [...iterFlowOrigins(first)];
  const [second, secondTable] = rebaseFlowOrigins(first, (o) => origin(`second/${o.key}`), { resolver: firstTable });

  expect([...iterFlowOrigins(flow)]).toEqual(originalOrigins);
  expect([...iterFlowOrigins(first)]).toEqual(firstOrigins);
  expect(firstOrigins.every((o) => o.key.startsWith("first/"))).toBe(true);
  expect(keys(iterFlowOrigins(second)).every((k) => k.startsWith("second/first/"))).toBe(true);
  const sample = originalOrigins[0]!;
  expect(secondTable.resolve(origin(`second/first/${sample.key}`))!.label).toBe(sample.key);
});

test("rebase rejects missing resolution and distinct origin collisions", () => {
  // Present origins require a delegate and distinct old keys cannot collapse.
  const flow = soloFlow(
    new BareNode("only", { origin: origin("node") }),
    new Map([["value", new Binding(1n, { origin: origin("binding") })]]),
  );

  expect(() => rebaseFlowOrigins(flow, (o) => o, { resolver: null })).toThrow(
    new RangeError("originful flow requires an origin resolver"),
  );
  expect(() => rebaseFlowOrigins(flow, () => origin("same"), { resolver: nullResolver() })).toThrow(
    new RangeError("mapped origin collision 'same'"),
  );

  expect(keys(iterFlowOrigins(flow))).toEqual(["binding", "node"]);
});

for (const invalid of [null, "not-an-origin"]) {
  test(`rebase rejects invalid mapper results (${String(invalid)})`, () => {
    // A namespace mapper must return an `OriginRef` for every distinct old ref.
    const flow = soloFlow(new BareNode("only", { origin: origin("node") }));

    expect(() =>
      rebaseFlowOrigins(flow, () => invalid as unknown as OriginRef, { resolver: new ResolverTable() }),
    ).toThrow(new TypeError(`origin mapper must return an OriginRef, got ${invalid === null ? "object" : "string"}`));
  });
}

test("rebase revalidates origin subclass keys", () => {
  // A forged origin cannot introduce an empty old or mapped opaque key.
  const invalid = Object.assign(Object.create(OriginRef.prototype) as OriginRef, { key: "" });
  const flow = soloFlow(new BareNode("only", { origin: invalid }));

  expect(() => rebaseFlowOrigins(flow, (o) => o, { resolver: nullResolver() })).toThrow(
    new RangeError("origin key must be a non-empty string"),
  );

  flow.nodes.get("only")!.origin = origin("valid");
  expect(() => rebaseFlowOrigins(flow, () => invalid, { resolver: nullResolver() })).toThrow(
    new RangeError("origin key must be a non-empty string"),
  );
});

test("rebase source-free graph returns an independent copy and empty table", () => {
  // Hand-built graphs need no front-end resolver when no origins are present.
  const node = new BareNode("python");
  const flow = soloFlow(node);
  const unexpected = (): OriginRef => {
    throw new Error("source-free graph called the namespace mapper");
  };

  const [rebased, table] = rebaseFlowOrigins(flow, unexpected, { resolver: null });

  expect(rebased).not.toBe(flow);
  expect(rebased.nodes.get("python")).not.toBe(node);
  expect([...iterFlowOrigins(rebased)]).toEqual([]);
  expect(table.size).toBe(0);

  expect(() => rebaseFlowOrigins(flow, unexpected, { resolver: new ResolverTable() })).toThrow(
    new RangeError("source-free flow cannot have an origin resolver"),
  );
});

test("rebase retains a deliberate child cycle", () => {
  // Atomic rebasing inherits traversal's identity guards and cycle-safe copying.
  const cycle = cycleFlow();

  const [rebased, table] = rebaseFlowOrigins(cycle, (o) => origin(`mapped/${o.key}`), { resolver: nullResolver() });

  const driver = rebased.nodes.get("self") as CallNode;
  expect(driver.child).toBe(rebased);
  expect(driver.origin).toEqual(origin("mapped/cycle/node"));
  expect((cycle.nodes.get("self") as CallNode).child).toBe(cycle);
  expect(table.size === 1 && table.has(origin("mapped/cycle/node"))).toBe(true);
});
