/** Splicing and pruning one run's private topology: its nodes, edges, wiring and adjacency. */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { Node, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { Binding } from "../../src/nodes/contracts.ts";

/** A minimal concrete node. */
class FuncNode extends Node {
  readonly kind = NodeKind.CODE;

  override forward(): NodeResult {
    return new Output(1n);
  }
}

/** Return a flow `START -> a -> END`. */
function base(): CompiledFlow {
  return new CompiledFlow(new Map([["a", new FuncNode("a")]]), [
    new Edge("__start__->a", START_ID, "a"),
    new Edge("a->__end__#0", "a", END_ID),
  ]);
}

test("addSubgraph extends nodes, edges and wiring", () => {
  const flow = base();
  flow.addSubgraph(
    new Map([["ns/b", new FuncNode("ns/b")]]),
    [new Edge("a->ns/b#0", "a", "ns/b", { inputGroup: "x" })],
    new Map([["ns/b", new Map([["x", "${a.output}"]])]]),
  );
  expect(flow.nodes.has("ns/b")).toBe(true);
  expect(flow.wiring.get("ns/b")).toEqual(new Map([["x", new Binding("${a.output}")]]));
  // Adjacency is updated in both directions.
  expect(flow.incoming("ns/b").some((e) => e.to === "ns/b")).toBe(true);
  expect(flow.outgoing("a").some((e) => e.to === "ns/b")).toBe(true);
});

test("removeSubgraph drops nodes, edges, wiring and adjacency", () => {
  // Two sibling namespaces spawned onto the base graph; remove one and the other survives.
  const flow = base();
  for (const ns of ["iter0", "iter1"]) {
    flow.addSubgraph(
      new Map([[`${ns}/b`, new FuncNode(`${ns}/b`)]]),
      [new Edge(`a->${ns}/b`, "a", `${ns}/b`, { inputGroup: "x" }), new Edge(`${ns}/b->__end__`, `${ns}/b`, END_ID)],
      new Map([[`${ns}/b`, new Map([["x", "${a.output}"]])]]),
    );
  }

  flow.removeSubgraph(new Set(["iter0/b"]));

  // The removed node is gone from nodes and wiring.
  expect(flow.nodes.has("iter0/b")).toBe(false);
  expect(flow.wiring.has("iter0/b")).toBe(false);
  // No edge references the removed id (as producer or consumer).
  expect(flow.edges.every((e) => e.from !== "iter0/b" && e.to !== "iter0/b")).toBe(true);
  // Adjacency: no key for the removed id, and no surviving edge references it.
  expect(flow.outgoingEdges.has("iter0/b")).toBe(false);
  expect(flow.incomingEdges.has("iter0/b")).toBe(false);
  for (const edges of flow.outgoingEdges.values()) expect(edges.every((e) => e.to !== "iter0/b")).toBe(true);
  for (const edges of flow.incomingEdges.values()) expect(edges.every((e) => e.from !== "iter0/b")).toBe(true);

  // The surviving sibling namespace and the base node are untouched.
  expect(flow.nodes.has("iter1/b")).toBe(true);
  expect(flow.wiring.get("iter1/b")).toEqual(new Map([["x", new Binding("${a.output}")]]));
  expect(flow.incoming("iter1/b").some((e) => e.to === "iter1/b")).toBe(true);
  expect(flow.outgoing("a").some((e) => e.to === "iter1/b")).toBe(true);
  expect(flow.nodes.has("a")).toBe(true);
  // `a` no longer routes to the removed node, and still routes to the survivor.
  expect(flow.outgoing("a").every((e) => e.to !== "iter0/b")).toBe(true);
});
