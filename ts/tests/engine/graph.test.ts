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
 * `test_compiled_flow_is_flow_and_keeps_boundary_and_terminal`).
 *
 * Later (T3-17/T3-22): `test_compiled_flow_is_flow_and_keeps_boundary_and_terminal` and the
 * compose rows (`test_data_edge_optional_reflects_binding_escape`, the CASE escape rows, the
 * branch gate rows, `test_item_head_ref_yields_no_edge_but_producer_ref_does` and
 * `test_binding_coskip_sets_edge_optional_per_stance`): they load YAML.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, Flow, GrowthRejected, START_ID } from "../../src/compile/model.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
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
