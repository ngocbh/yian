/**
 * The compiled flow topology model of `compile/model.ts`: a port of `tests/engine/test_graph.py`.
 * Ported: `test_adjacency_and_root`, `test_diamond_incoming`, `test_edge_input_group`,
 * `test_compiled_flow_wiring_field_threads_and_defaults`,
 * `test_flow_base_normalizes_wiring_with_boundary_defaults`,
 * `test_compiled_flow_is_flow_and_keeps_boundary_and_terminal` and
 * `test_edge_optional_defaults_false_and_roundtrips`; the compose rows
 * `test_data_edge_optional_reflects_binding_escape`, `test_case_on_escape_marks_optional`, the
 * searched-case condition rows (`test_searched_case_condition_default_marks_optional`,
 * `test_searched_case_plain_reference_stays_required`,
 * `test_searched_case_literal_coalesce_and_required_mark_optional`,
 * `test_searched_case_default_to_another_reference_also_joins`,
 * `test_searched_case_plain_and_default_reads_of_one_output`,
 * `test_searched_case_unparenthesized_default_covers_the_whole_condition`),
 * `test_searched_case_condition_with_a_nested_span_shows_the_rewrite`,
 * `test_item_head_ref_yields_no_edge_but_producer_ref_does` and
 * `test_binding_coskip_sets_edge_optional_per_stance`.
 *
 * Changed: Python's `FuncNode` is a minimal `Node` subclass; node and wiring tables are `Map`s;
 * bindings compare by `source`, `origin` and `contract`. The compose rows assemble their flows
 * through `assembleText`, where Python calls `load_flow`, and name TS's CODE functions
 * (`./fixtures/compose_codefns.ts:<fn>`); the item-head row's consumer is a CODE descriptor whose
 * wiring the row hands to `inferDataEdges`, as Python's does.
 *
 * TS-only: `GrowthRejected` keeps `original` and refuses a non-`Error`; the `CompiledFlow`
 * `flowId` and `memories` checks; `terminalId` is `END_ID` on a hand-built flow holding an END
 * node and `null` once that node is removed; `copyGraph`, the shared graph
 * copier that stands in for Python's `copy.deepcopy` of a flow or node: a copied flow and its
 * nodes are distinct instances of their classes with fresh containers, and splicing or pruning the
 * copy leaves the original; a node or child flow reached twice is copied once; a cycle copies to a
 * cycle; a node copy takes its child flow along, with fresh containers; each call gets a fresh
 * memo; a copy shares its carriers, and a field reassigned on it leaves the original;
 * `validateGrowthMetadata` (Python tests it through the expansion builders): a small valid
 * fragment passes, and one row per refusal message; `compile/index.ts` exports exactly Python's
 * `yian.compile.__all__`.
 *
 * Later (T3-20): the branch gate rows, which need the narrowing and the skipped-output checks
 * (`test_a_plain_read_of_a_branch_keeps_the_gate_in_the_branch`,
 * `test_a_default_read_of_a_branch_lets_the_gate_run_outside_it`,
 * `test_a_default_read_of_a_branch_rejects_arithmetic_on_it_downstream`,
 * `test_a_guarded_read_inside_a_default_loads`,
 * `test_a_co_skipping_default_keeps_the_gate_in_the_branch`).
 */

import { expect, test } from "bun:test";

import * as layoutModule from "../../src/compile/execution_layout.ts";
import * as compilePackage from "../../src/compile/index.ts";
import * as modelModule from "../../src/compile/model.ts";
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
import * as originsModule from "../../src/compile/origins.ts";
import { inferDataEdges } from "../../src/compose/edges/data.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCursor } from "../../src/compose/source/index.ts";
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
import { assembleText, loadError } from "../compose/declarations.ts";

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

test("a compiled flow is a Flow and keeps its boundary and terminal", () => {
  const compiled = assembleText(OPTIONAL_EDGE_FLOW).flow.compiled;
  expect(compiled).toBeInstanceOf(Flow);
  expect(compiled.startId).toBe(START_ID);
  expect(compiled.endId).toBe(END_ID);
  // terminalId resolves to END_ID while the synthesized END node is present, null once absent.
  expect(compiled.terminalId).toBe(END_ID);
  compiled.nodes.delete(END_ID);
  expect(compiled.terminalId).toBeNull();
});

test("edge optional defaults false and roundtrips", () => {
  expect(new Edge("a->b#0", "a", "b", { inputGroup: "x" }).optional).toBe(false);
  expect(new Edge("a->b#0", "a", "b", { inputGroup: "x", optional: true }).optional).toBe(true);
});

const OPTIONAL_EDGE_FLOW = `
id: opt
name: opt
input:
  seed: !float
nodes:
  a:
    kind: code
    code: ./fixtures/compose_codefns.ts:score
    input:
      seed: \${input.seed}
    output: float
  hard:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      v: \${a.output}
    output: str
  soft:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      v: \${a.output:-null}
    output: str
output: \${hard.output}
`;

/** Return the assembled flow's edges in input group `v`, by consumer. */
function vEdges(text: string): Map<string, Edge> {
  const edges = assembleText(text).flow.compiled.edges;
  return new Map(edges.filter((edge) => edge.inputGroup === "v").map((edge) => [edge.to, edge]));
}

test("a data edge's optional reflects the binding's escape", () => {
  const byTo = vEdges(OPTIONAL_EDGE_FLOW);
  expect(byTo.get("hard")!.optional).toBe(false); // plain ref -> required
  expect(byTo.get("soft")!.optional).toBe(true); // `:-null` escape -> optional
});

const CASE_ESCAPE_FLOW = `
id: ce
name: ce
input:
  seed: !float
nodes:
  a:
    kind: code
    code: ./fixtures/compose_codefns.ts:score
    input:
      seed: \${input.seed}
    output: float
  gate:
    kind: case
<gate>
    else: cold
  hot:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      seed: \${input.seed}
    output: str
  cold:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      seed: \${input.seed}
    output: str
output: \${gate.output}
`;

/** Return the `a -> gate` edges of `CASE_ESCAPE_FLOW` with `gate`'s body. */
function aToGateEdges(gate: string): Edge[] {
  const edges = assembleText(CASE_ESCAPE_FLOW.replace("<gate>", gate)).flow.compiled.edges;
  return edges.filter((edge) => edge.from === "a" && edge.to === "gate");
}

/** Return the one `a -> gate` edge of `CASE_ESCAPE_FLOW` with `gate`'s body. */
function aToGate(gate: string): Edge {
  const edges = aToGateEdges(gate);
  expect(edges).toHaveLength(1);
  return edges[0]!;
}

/** The `gate` body of `CASE_ESCAPE_FLOW` for one searched case reading `when`. */
const searchedGate = (when: string) => `    cases:\n      - when: ${when}\n        then: hot`;

test("an on: escape marks the gate input optional", () => {
  // An `on:` value is a whole binding, so a `:-` escape makes the gate input optional.
  const gate = "    on: ${a.output :- 0}\n    cases:\n      - when: 1\n        then: hot";
  expect(aToGate(gate).optional).toBe(true);
});

test("a searched case condition's default marks the edge optional", () => {
  // A reference under `:-` takes the edge a binding of that default would: optional.
  expect(aToGate(searchedGate('"(a.output :- 0) >= 1"')).optional).toBe(true);
});

test("a searched case's plain reference stays required", () => {
  expect(aToGate(searchedGate('"a.output >= 1"')).optional).toBe(false);
});

test.each(['"(a.output | 0) >= 1"', `'(a.output :? "need a") >= 1'`])(
  "a searched case's literal coalesce or required mark the edge optional: %s",
  (when) => {
    expect(aToGate(searchedGate(when)).optional).toBe(true);
  },
);

test("a searched case's default to another reference also joins", () => {
  // `a.output :- input.seed` co-skips as a binding: the value is read optionally, and a required
  // join `${a.output}` skips the gate with `a`.
  const edges = aToGateEdges(searchedGate('"(a.output :- input.seed) >= 1"'));
  expect(edges.map((edge) => edge.optional).sort()).toEqual([false, true]);
});

test("a searched case's plain and default reads of one output", () => {
  const edges = aToGateEdges(searchedGate('"a.output > 5 or (a.output :- 0) < 1"'));
  expect(edges.map((edge) => edge.optional).sort()).toEqual([false, true]);
});

test("a searched case's unparenthesized default covers the whole condition", () => {
  // `:-` binds looser than `and`, so this is `(input.seed > 0 and a.output) :- false`: one escape
  // around the whole condition, whose literal default does not co-skip.
  const edges = aToGateEdges(searchedGate('"input.seed > 0 and a.output :- false"'));
  expect(edges.length > 0 && edges.every((edge) => edge.optional)).toBe(true);
});

test("a searched case condition with a nested span shows the rewrite", () => {
  const gate = '    cases:\n      - when: "${a.output:-0} >= 1"\n        then: hot';
  expect(loadError(() => aToGate(gate)).message).toMatch(/write `\(a.output:-0\) >= 1`/);
});

// An item-headed ref (`${item[.field]}`) in a MAP/LOOP body binding is body-local and yields no
// data edge (there is no node named `item`), while `${producer.output}` in the same body does.

test("an item-head ref yields no edge but a producer ref does", () => {
  // A CODE `producer` and a `mapbody` consumer whose wiring binds both `${item.foo}` (body-local)
  // and `${producer.output}`. `inferDataEdges` gets the hand-built wiring for the consumer (the
  // leaf/MAP branch reads it verbatim), the producer ref walk a map/loop body input goes through.
  const descriptors = parseNodes(
    SourceCursor.virtual({
      producer: {
        kind: "code",
        code: "./fixtures/compose_codefns.ts:score",
        input: { seed: "${input.seed}" },
        output: "float",
      },
      mapbody: {
        kind: "code",
        code: "./fixtures/compose_codefns.ts:cautious",
        input: { v: "${producer.output}" },
        output: "str",
      },
    }),
  );
  const flowWiring = new Map([
    // `item`-headed refs are body-local (no edge); `producer.output` is a producer ref (an edge).
    [
      "mapbody",
      new Map([
        ["a", "${item}"],
        ["b", "${item.foo}"],
        ["c", "${producer.output}"],
      ]),
    ],
  ]);
  const edges = inferDataEdges(descriptors, flowWiring);
  expect(edges.some((edge) => edge.from === "producer" && edge.to === "mapbody")).toBe(true);
  expect(edges.some((edge) => edge.from === "item")).toBe(false);
  expect(edges.some((edge) => edge.to === "item")).toBe(false);
});

// The co-skip contract end to end: a binding whose fallback is a ref (a bare ref, a `.field` walk,
// or a `:-<ref>` default) may be absent without failing, so it co-skips: a required edge. A binding
// that pins a concrete value or requires presence (`:-"literal"`, `:-null`, `:?`) does not
// co-skip: an optional edge.

const COSKIP_STANCE_FLOW = `
id: coskip_stance
name: coskip_stance
input:
  seed: !float
nodes:
  a:
    kind: code
    code: ./fixtures/compose_codefns.ts:score
    input:
      seed: \${input.seed}
    output: float
  bare:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      v: \${a.output}
    output: str
  ref_default:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      v: \${a.output:-a.output}
    output: str
  lit_default:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      v: \${a.output:-"fallback"}
    output: str
  required:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      v: \${a.output:?"required"}
    output: str
output: \${bare.output}
`;

test("a binding's co-skip sets the edge's optional per stance", () => {
  const byTo = vEdges(COSKIP_STANCE_FLOW);
  // Co-skipping bindings (ref fallback) are required:
  expect(byTo.get("bare")!.optional).toBe(false); // a bare ref
  expect(byTo.get("ref_default")!.optional).toBe(false); // `:-<ref>` default (ref fallback)
  // Non-co-skipping bindings (concrete or required) are optional:
  expect(byTo.get("lit_default")!.optional).toBe(true); // `:-"literal"` default
  expect(byTo.get("required")!.optional).toBe(true); // `:?` required
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
    "a forged boundary scope with a non-array occurrence path",
    () => {
      const flow = growthFragment();
      flow.nodes.get("s")!.executionScope = forged(ExecutionScope, {
        staticPath: ["call", "s"],
        flowPath: ["call"],
        occurrencePath: "x",
        boundary: "start",
      });
      return flow;
    },
    TypeError,
    "occurrence_path must be an array",
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

test("compile/index.ts exports exactly Python's yian.compile.__all__", () => {
  // Each name and the module that defines it; a namespace lists its keys code-unit sorted.
  const sources: Record<string, Record<string, unknown>> = {
    CompiledFlow: modelModule,
    END_ID: modelModule,
    Edge: modelModule,
    START_ID: modelModule,
    applyExecutionLayout: layoutModule,
    iterFlowOrigins: originsModule,
    rebaseFlowOrigins: originsModule,
    resolveExecutionLayout: layoutModule,
    rewriteFlowOrigins: originsModule,
  };
  expect(Object.keys(compilePackage)).toEqual(Object.keys(sources));
  const exported: Record<string, unknown> = compilePackage;
  for (const [name, source] of Object.entries(sources)) expect(exported[name]).toBe(source[name]);
});
