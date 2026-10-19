/**
 * The compiled flow topology model of `compile/model.ts`: a port of `tests/engine/test_graph.py`.
 * Ported: `test_adjacency_and_root`, `test_diamond_incoming`, `test_edge_input_group`,
 * `test_compiled_flow_wiring_field_threads_and_defaults`,
 * `test_flow_base_normalizes_wiring_with_boundary_defaults` and
 * `test_edge_optional_defaults_false_and_roundtrips`.
 *
 * Changed: Python's `FuncNode` is a minimal `Node` subclass; node and wiring tables are `Map`s;
 * bindings compare by `source`, `origin` and `contract`.
 *
 * TS-only: `GrowthRejected` keeps `original` and refuses a non-`Error`; the `CompiledFlow`
 * `flowId` and `memories` checks; `terminalId` is `END_ID` on a hand-built flow holding an END
 * node and `null` once that node is removed (the model half of
 * `test_compiled_flow_is_flow_and_keeps_boundary_and_terminal`); `copyGraph`, the shared graph
 * copier that stands in for Python's `copy.deepcopy` of a flow or node: a copied flow and its
 * nodes are distinct instances of their classes with fresh containers, and splicing or pruning the
 * copy leaves the original; a node or child flow reached twice is copied once; a cycle copies to a
 * cycle; a node copy takes its child flow along; each call gets a fresh memo; a reassigned field
 * on a copy leaves the original.
 *
 * Later (T3-17/T3-22): `test_compiled_flow_is_flow_and_keeps_boundary_and_terminal` and the
 * compose rows (`test_data_edge_optional_reflects_binding_escape`, the CASE escape rows, the
 * branch gate rows, `test_item_head_ref_yields_no_edge_but_producer_ref_does` and
 * `test_binding_coskip_sets_edge_optional_per_stance`): they load YAML.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, copyGraph, Edge, END_ID, Flow, GrowthRejected, START_ID } from "../../src/compile/model.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Binding, ExecutionScope, Parameter } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";

/** A minimal concrete node standing in for Python's `FuncNode`. */
class FuncNode extends Node {
  readonly kind = NodeKind.CODE;

  override forward(): NodeResult {
    return new Output({});
  }
}

/** Return one `FuncNode` per id. */
function nodesOf(...ids: string[]): Map<string, Node> {
  return new Map(ids.map((id) => [id, new FuncNode(id)]));
}

/** Return `nodes` plus the real START and END boundary nodes. */
function withBoundary(nodes: Map<string, Node>): Map<string, Node> {
  const out = new Map(nodes);
  if (!out.has(START_ID)) out.set(START_ID, new StartNode(START_ID, { params: [] }));
  if (!out.has(END_ID)) out.set(END_ID, EndNode.value(END_ID, { outputName: null }));
  return out;
}

test("adjacency and root", () => {
  const nodes = withBoundary(nodesOf("a", "b", "c"));
  const edges = [
    new Edge("e0", START_ID, "a"),
    new Edge("e1", "a", "b"),
    new Edge("e2", "a", "c"),
    new Edge("e3", "b", END_ID),
  ];
  const g = CompiledFlow.fromParts(nodes, edges);
  // The single root is the synthesized START_ID; `a` is its out-edge target.
  expect(g.startId).toBe(START_ID);
  expect(new Set(g.outgoing(START_ID).map((e) => e.to))).toEqual(new Set(["a"]));
  expect(new Set(g.outgoing("a").map((e) => e.to))).toEqual(new Set(["b", "c"]));
  expect(g.incoming("b").map((e) => e.from)).toEqual(["a"]);
  expect(g.terminalId).toBe(END_ID);
});

test("diamond incoming", () => {
  const edges = [
    new Edge("e0", START_ID, "a"),
    new Edge("e1", "a", "b"),
    new Edge("e2", "a", "c"),
    new Edge("e3", "b", "d"),
    new Edge("e4", "c", "d"),
    new Edge("e5", "d", END_ID),
  ];
  const g = CompiledFlow.fromParts(nodesOf("a", "b", "c", "d"), edges);
  expect(new Set(g.incoming("d").map((e) => e.from))).toEqual(new Set(["b", "c"]));
});

test("edge input group", () => {
  expect(new Edge("e0", "a", "b", { inputGroup: "x" }).inputGroup).toBe("x");
  expect(new Edge("e1", "a", "b").inputGroup).toBeNull();
  expect(new Edge("e2", START_ID, "a").inputGroup).toBeNull();
});

test("compiled flow wiring field threads and defaults", () => {
  // CompiledFlow carries flow-owned wiring (node id -> param -> source); empty by default.
  const nodes = nodesOf("a");
  const edges = [new Edge("e0", START_ID, "a"), new Edge("e1", "a", END_ID)];
  const g = CompiledFlow.fromParts(nodes, edges, { wiring: new Map([["a", new Map([["x", "${input.x}"]])]]) });
  expect(g.wiring).toEqual(new Map([["a", new Map([["x", new Binding("${input.x}")]])]]));
  expect(CompiledFlow.fromParts(nodes, edges).wiring).toEqual(new Map());
});

test("flow base normalizes wiring with boundary defaults", () => {
  // A bare Flow keeps topology and normalizes wiring; boundary ids use reserved defaults.
  const nodes = nodesOf("a");
  const edges = [new Edge("e0", START_ID, "a"), new Edge("e1", "a", END_ID)];
  const f = new Flow(nodes, edges, { wiring: new Map([["a", new Map([["x", "${input.x}"]])]]) });
  expect(f.nodes).toBe(nodes);
  expect(f.edges).toBe(edges);
  expect(f.wiring).toEqual(new Map([["a", new Map([["x", new Binding("${input.x}")]])]]));
  expect(f.startId).toBe(START_ID);
  expect(f.endId).toBe(END_ID);
  // Spliced flows pass their own namespaced boundary ids.
  const spliced = new Flow(nodes, edges, { startId: "call#1/__start__", endId: "call#1/__end__" });
  expect(spliced.startId).toBe("call#1/__start__");
  expect(spliced.endId).toBe("call#1/__end__");
  expect(spliced.wiring).toEqual(new Map());
});

test("edge optional defaults false and roundtrips", () => {
  expect(new Edge("a->b#0", "a", "b", { inputGroup: "x" }).optional).toBe(false);
  expect(new Edge("a->b#0", "a", "b", { inputGroup: "x", optional: true }).optional).toBe(true);
});

test("growth rejected keeps the original error and refuses a non-Error", () => {
  const original = new RangeError("bad growth");
  const rejected = new GrowthRejected(original);
  expect(rejected.name).toBe("GrowthRejected");
  expect(rejected.message).toBe("bad growth");
  expect(rejected.original).toBe(original);
  expect(rejected.cause).toBe(original);
  expect(() => new GrowthRejected("bad growth" as never)).toThrow(TypeError);
  expect(() => new GrowthRejected("bad growth" as never)).toThrow("GrowthRejected original must be an Error");
});

test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string]>([
  [
    "a non-string flow id",
    () => new CompiledFlow(new Map(), [], { flowId: 1 as never }),
    TypeError,
    "flow_id must be an exact string or null",
  ],
  ["an empty flow id", () => new CompiledFlow(new Map(), [], { flowId: "" }), RangeError, "flow_id must be non-empty"],
  [
    "memories that aren't MemoryDecls",
    () => new CompiledFlow(new Map(), [], { memories: [{}] as never }),
    TypeError,
    "memories must be an array of exact MemoryDecl values",
  ],
  [
    "a memory declared on a node site",
    () =>
      new CompiledFlow(new Map(), [], {
        memories: [new MemoryDecl("notes", "episodic", {}, new DeclSite([], ["root"], "agent"))],
      }),
    RangeError,
    "flow memories must carry flow DeclSites (node=null)",
  ],
])("compiled flow refuses %s", (_name, factory, cls, message) => {
  expect(factory).toThrow(cls);
  expect(factory).toThrow(message);
});

test("compiled flow keeps its flow id and flow memories", () => {
  const decl = new MemoryDecl("notes", "episodic", {}, new DeclSite([], ["root"]));
  const g = new CompiledFlow(new Map(), [], { flowId: "flow", memories: [decl] });
  expect(g.flowId).toBe("flow");
  expect(g.memories).toEqual([decl]);
  expect(g.flowLlmConfig).toEqual({});
  expect(g.flowLlmValueTypes).toEqual([]);
  expect(g.outputs).toEqual([]);
});

test("terminal id is END_ID while the END node is present and null once it is removed", () => {
  const edges = [new Edge("e0", START_ID, "a"), new Edge("e1", "a", END_ID)];
  const g = CompiledFlow.fromParts(withBoundary(nodesOf("a")), edges);
  expect(g).toBeInstanceOf(Flow);
  expect(g.startId).toBe(START_ID);
  expect(g.endId).toBe(END_ID);
  expect(g.terminalId).toBe(END_ID);
  g.nodes.delete(END_ID);
  expect(g.terminalId).toBeNull();
});

/** Return a flow holding `nodes` (by id) with no edges. */
function flowOf(...nodes: Node[]): Flow {
  return new Flow(new Map(nodes.map((node) => [node.id, node])), []);
}

test("a copied flow has fresh containers and nodes, and changing it leaves the original", () => {
  const edges = [new Edge("e0", START_ID, "a"), new Edge("e1", "a", "b"), new Edge("e2", "b", END_ID)];
  const wiring = new Map([
    ["a", new Map([["x", "${input.x}"]])],
    ["b", new Map([["y", "${a.output}"]])],
  ]);
  const g = CompiledFlow.fromParts(withBoundary(nodesOf("a", "b")), edges, { wiring, flowId: "flow" });
  const before = {
    nodes: [...g.nodes],
    edges: [...g.edges],
    wiring: [...g.wiring],
    incoming: [...g.incomingEdges].map(([id, list]) => [id, [...list]]),
    outgoing: [...g.outgoingEdges].map(([id, list]) => [id, [...list]]),
  };

  const copy = copyGraph(g);

  expect(copy).not.toBe(g);
  expect(Object.getPrototypeOf(copy)).toBe(CompiledFlow.prototype);
  expect(copy.flowId).toBe("flow");
  expect(copy.nodes).not.toBe(g.nodes);
  expect([...copy.nodes.keys()]).toEqual([...g.nodes.keys()]);
  for (const [id, node] of g.nodes) {
    const copied = copy.nodes.get(id);
    expect(copied).not.toBe(node);
    expect(Object.getPrototypeOf(copied)).toBe(Object.getPrototypeOf(node));
    expect(copied).toStrictEqual(node);
  }
  expect(copy.edges).not.toBe(g.edges);
  expect(copy.edges).toEqual(g.edges);
  expect(copy.wiring).not.toBe(g.wiring);
  expect([...copy.wiring.keys()]).toEqual([...g.wiring.keys()]);
  for (const [id, nodeWiring] of g.wiring) expect(copy.wiring.get(id)).toBe(nodeWiring);
  expect(copy.incomingEdges).not.toBe(g.incomingEdges);
  expect(copy.incoming("b")).not.toBe(g.incoming("b"));
  expect(copy.incoming("b")).toEqual(g.incoming("b"));
  expect(copy.outgoing("a")).not.toBe(g.outgoing("a"));
  expect(copy.outgoing("a")).toEqual(g.outgoing("a"));

  copy.addSubgraph(
    nodesOf("c"),
    [new Edge("e3", "b", "c"), new Edge("e4", "a", "c")],
    new Map([["c", new Map([["z", "${b.output}"]])]]),
  );
  copy.removeSubgraph(new Set(["b"]));
  expect([...g.nodes]).toEqual(before.nodes);
  expect(g.edges).toEqual(before.edges);
  expect([...g.wiring]).toEqual(before.wiring);
  expect([...g.incomingEdges].map(([id, list]) => [id, [...list]])).toEqual(before.incoming);
  expect([...g.outgoingEdges].map(([id, list]) => [id, [...list]])).toEqual(before.outgoing);
});

test("a node shared by a root flow and its child is copied once", () => {
  const shared = new FuncNode("shared");
  const child = flowOf(shared);
  const root = flowOf(shared, new CallNode("call", { flowId: "child", child }));

  const copy = copyGraph(root);
  const copiedChild = (copy.nodes.get("call") as CallNode).child as Flow;

  expect(copy.nodes.get("shared")).not.toBe(shared);
  expect(copiedChild.nodes.get("shared")).toBe(copy.nodes.get("shared"));
});

test("a child flow shared by CALL, MAP and LOOP drivers is copied once", () => {
  const child = flowOf(new FuncNode("leaf"));
  const root = flowOf(
    new CallNode("call", { flowId: "child", child }),
    new MapNode("map", { flowId: "child", child }),
    new LoopNode("loop", { flowId: "child", child, predicateKind: "times", times: 1, maxIters: 1 }),
  );

  const copy = copyGraph(root);
  const children = ["call", "map", "loop"].map((id) => (copy.nodes.get(id) as CallNode | MapNode | LoopNode).child);

  expect(children[0]).toBeInstanceOf(Flow);
  expect(children[0]).not.toBe(child);
  expect(children[1]).toBe(children[0]);
  expect(children[2]).toBe(children[0]);
  expect((children[0] as Flow).nodes.get("leaf")).not.toBe(child.nodes.get("leaf"));
});

test("a cyclic flow copies to a cycle", () => {
  const flow = flowOf();
  flow.nodes.set("call", new CallNode("call", { flowId: "self", child: flow }));

  const copy = copyGraph(flow);
  const call = copy.nodes.get("call") as CallNode;

  expect(copy).not.toBe(flow);
  expect(call).not.toBe(flow.nodes.get("call"));
  expect(call.child).toBe(copy);
});

test("copying a node copies its child flow and leaves the original's child", () => {
  const child = flowOf(new FuncNode("leaf"));
  const call = new CallNode("call", { flowId: "child", child });

  const copy = copyGraph(call);

  expect(copy).not.toBe(call);
  expect(copy).toBeInstanceOf(CallNode);
  expect(copy.flowId).toBe("child");
  expect(copy.child).toBeInstanceOf(Flow);
  expect(copy.child).not.toBe(child);
  expect((copy.child as Flow).nodes.get("leaf")).not.toBe(child.nodes.get("leaf"));
  expect(call.child).toBe(child);
});

test("two copies of one flow share no graph-owned objects", () => {
  const child = flowOf(new FuncNode("leaf"));
  const root = flowOf(new CallNode("call", { flowId: "child", child }));

  const first = copyGraph(root);
  const second = copyGraph(root);
  const firstCall = first.nodes.get("call") as CallNode;
  const secondCall = second.nodes.get("call") as CallNode;

  expect(second).not.toBe(first);
  expect(second.nodes).not.toBe(first.nodes);
  expect(secondCall).not.toBe(firstCall);
  expect(secondCall.child).not.toBe(firstCall.child);
  expect((secondCall.child as Flow).nodes.get("leaf")).not.toBe((firstCall.child as Flow).nodes.get("leaf"));
});

test("a field reassigned on a node copy leaves the original", () => {
  const params = [new Parameter("x")];
  const node = new FuncNode("a");
  node.params = params;
  const copy = copyGraph(node);

  copy.params = [new Parameter("y")];
  copy.executionScope = new ExecutionScope(["a"]);

  expect(node.params).toBe(params);
  expect(node.executionScope).toBeNull();
});
