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
 * cycle; a node copy takes its child flow along, with fresh containers; each call gets a fresh
 * memo; a copy shares its carriers, and a field reassigned on it leaves the original;
 * `validateGrowthMetadata` (Python tests it through the expansion builders): a small valid
 * fragment passes, and one row per refusal message.
 *
 * Later (T3-17/T3-22): `test_compiled_flow_is_flow_and_keeps_boundary_and_terminal` and the
 * compose rows (`test_data_edge_optional_reflects_binding_escape`, the CASE escape rows, the
 * branch gate rows, `test_item_head_ref_yields_no_edge_but_producer_ref_does` and
 * `test_binding_coskip_sets_edge_optional_per_stance`): they load YAML.
 */

import { expect, test } from "bun:test";

import {
  CompiledFlow,
  copyGraph,
  Edge,
  END_ID,
  Flow,
  GrowthRejected,
  START_ID,
  validateGrowthMetadata,
} from "../../src/compile/model.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import {
  Binding,
  ChildFlowBoundary,
  ExecutionPosition,
  ExecutionScope,
  Parameter,
  SeedConnection,
} from "../../src/nodes/contracts.ts";
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
  expect((copy.child as Flow).edges).not.toBe(child.edges);
  expect((copy.child as Flow).wiring).not.toBe(child.wiring);
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
  expect(copy.params).toBe(params);

  copy.params = [new Parameter("y")];
  copy.executionScope = new ExecutionScope(["a"]);

  expect(node.params).toBe(params);
  expect(node.executionScope).toBeNull();
});

/** Return a boundary scope for node `id` of child flow `call`, at item `index` of a MAP. */
function boundaryScope(id: string, boundary: "start" | "end", index = 0): ExecutionScope {
  return new ExecutionScope(["call", id], {
    flowPath: ["call"],
    occurrencePath: [new ExecutionPosition("map_item", ["call"], index)],
    boundary,
  });
}

/**
 * Return a valid growth fragment: child START `s` (input `x`, wired) and END `e`, each with its own
 * boundary scope, and node `h` with a defaulted unwired input `p`; both inputs are seeded unless
 * `recipes` replaces the recipes.
 */
function growthFragment(recipes: { inputConnections?: unknown; childBoundaries?: unknown } = {}): Flow {
  const {
    inputConnections = [new SeedConnection("s", "x", ["x"]), new SeedConnection("h", "p", ["p"])],
    childBoundaries = [new ChildFlowBoundary("s", "e")],
  } = recipes;
  const nodes = nodesOf("s", "e", "h");
  nodes.get("s")!.params = [new Parameter("x")];
  nodes.get("s")!.executionScope = boundaryScope("s", "start");
  nodes.get("e")!.executionScope = boundaryScope("e", "end");
  nodes.get("h")!.params = [new Parameter("p", { hasDefault: true, default: 1n })];
  return new Flow(nodes, [], {
    wiring: new Map([["s", new Map([["x", "${input.x}"]])]]),
    inputConnections: inputConnections as readonly SeedConnection[],
    childBoundaries: childBoundaries as readonly ChildFlowBoundary[],
  });
}

/** Return an instance of `cls` holding `fields`, made without running its constructor. */
function forged<T>(cls: abstract new (...args: never[]) => T, fields: object): T {
  return Object.assign(Object.create(cls.prototype) as object, fields) as T;
}

test("growth metadata accepts a seeded child START and a seeded defaulted unwired input", () => {
  const flow = growthFragment();
  expect(flow.nodes.get("s")!.executionScope!.occurrencePath[0]).not.toBe(
    flow.nodes.get("e")!.executionScope!.occurrencePath[0],
  );

  expect(() => validateGrowthMetadata(flow)).not.toThrow();
});

test.each([
  [
    "a non-array recipe",
    () => growthFragment({ inputConnections: "s" }),
    TypeError,
    "input_connections must be an array of exact SeedConnection values",
  ],
  [
    "a recipe item of another class",
    () => growthFragment({ childBoundaries: [new SeedConnection("s", "x", ["x"])] }),
    TypeError,
    "child_boundaries must be an array of exact ChildFlowBoundary values",
  ],
  [
    "a forged boundary with equal ids",
    () => growthFragment({ childBoundaries: [forged(ChildFlowBoundary, { startId: "s", endId: "s" })] }),
    RangeError,
    "child START and END must be distinct",
  ],
  [
    "a forged seed connection with an empty name",
    () =>
      growthFragment({
        inputConnections: [forged(SeedConnection, { childStartId: "s", inputName: "", seedPath: ["x"] })],
      }),
    RangeError,
    "seed connection identifiers must be non-empty strings",
  ],
  [
    "a boundary naming a missing node",
    () => growthFragment({ childBoundaries: [new ChildFlowBoundary("s", "absent")] }),
    RangeError,
    "child boundaries must name distinct fragment nodes",
  ],
  [
    "a boundary reusing an endpoint",
    () => growthFragment({ childBoundaries: [new ChildFlowBoundary("s", "e"), new ChildFlowBoundary("h", "e")] }),
    RangeError,
    "child boundaries must name distinct fragment nodes",
  ],
  [
    "a boundary endpoint that isn't a Node",
    () => {
      const flow = growthFragment();
      flow.nodes.set("e", { executionScope: null } as unknown as Node);
      return flow;
    },
    TypeError,
    "child boundaries must name Node values",
  ],
  [
    "a boundary scope of another class",
    () => {
      const flow = growthFragment();
      const node = flow.nodes.get("s")!;
      node.executionScope = { ...node.executionScope } as ExecutionScope;
      return flow;
    },
    TypeError,
    "child boundary scope must be an ExecutionScope",
  ],
  [
    "a boundary role that disagrees with its scope",
    () => {
      const flow = growthFragment();
      flow.nodes.get("s")!.executionScope = boundaryScope("s", "end");
      return flow;
    },
    RangeError,
    "child boundary role disagrees with its execution scope",
  ],
  [
    "a boundary scope whose static path isn't in its flow path",
    () => {
      const flow = growthFragment();
      flow.nodes.get("s")!.executionScope = new ExecutionScope(["other", "s"], {
        flowPath: ["call"],
        occurrencePath: [new ExecutionPosition("map_item", ["call"], 0)],
        boundary: "start",
      });
      return flow;
    },
    RangeError,
    "child boundary location must belong to its containing flow",
  ],
  [
    "boundaries whose occurrences differ in one position index",
    () => {
      const flow = growthFragment();
      flow.nodes.get("e")!.executionScope = boundaryScope("e", "end", 1);
      return flow;
    },
    RangeError,
    "child boundaries must belong to the same flow occurrence",
  ],
  [
    "a seed connection naming a missing node",
    () => growthFragment({ inputConnections: [new SeedConnection("absent", "x", ["x"])] }),
    RangeError,
    "seed connections must name fragment nodes",
  ],
  [
    "a duplicate seed target",
    () =>
      growthFragment({
        inputConnections: [new SeedConnection("s", "x", ["x"]), new SeedConnection("s", "x", [0, "x"])],
      }),
    RangeError,
    "seed connections must name unique target inputs",
  ],
  [
    "a seed target that isn't a Node",
    () => {
      const flow = growthFragment();
      flow.nodes.set("h", { params: [new Parameter("p", { hasDefault: true })] } as unknown as Node);
      return flow;
    },
    TypeError,
    "seed connection targets must be Node values",
  ],
  [
    "target parameters that aren't Parameters",
    () => {
      const flow = growthFragment();
      flow.nodes.get("s")!.params = [{ name: "x", hasDefault: false } as Parameter];
      return flow;
    },
    TypeError,
    "seed connection target parameters must be Parameter values",
  ],
  [
    "an undeclared seed input",
    () => growthFragment({ inputConnections: [new SeedConnection("s", "absent", ["absent"])] }),
    RangeError,
    "seed connection target input must be declared",
  ],
  [
    "an unwired child START input",
    () => {
      const flow = growthFragment();
      flow.wiring.delete("s");
      return flow;
    },
    RangeError,
    "seed connection child START input must be wired",
  ],
  [
    "a non-START target without a default",
    () => {
      const flow = growthFragment();
      flow.nodes.get("h")!.params = [new Parameter("p")];
      return flow;
    },
    RangeError,
    "seed connection non-START input must declare a default",
  ],
  [
    "a wired non-START target",
    () => {
      const flow = growthFragment();
      flow.wiring.set("h", new Map([["p", new Binding("${input.x}")]]));
      return flow;
    },
    RangeError,
    "seed connection non-START input must not be wired",
  ],
] as const)("growth metadata refuses %s", (_case, build, error, message) => {
  const flow = build();
  expect(() => validateGrowthMetadata(flow)).toThrow(error);
  expect(() => validateGrowthMetadata(flow)).toThrow(message);
});
