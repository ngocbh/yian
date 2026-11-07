/**
 * The pure LOOP continue builder `loopContinueSubgraph`, which pins the body and next-driver shape.
 *
 * `loopContinueSubgraph(child, origin, carried, k, driver)` splices body `k` (no `commitAs`: its
 * Output feeds the next driver by plain wiring), the fresh `L~(k+1)` driver and the producer edge
 * `body_k.END -> L~(k+1)`. Bodies are keyed on the origin, not the running driver id, so a live
 * run and durable reconstruction build the same `L#k/…` namespace.
 */

import { expect, test } from "bun:test";

import { loopContinueSubgraph } from "../../src/compile/expand.ts";
import { CompiledFlow, Edge, END_ID, Flow, START_ID, validateGrowthMetadata } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import type { Node } from "../../src/nodes/base.ts";
import { Binding, ChildFlowBoundary, Parameter, SeedConnection } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { mapCallsite, ns } from "../../src/nodes/ids.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { childFlow } from "../helpers/run_state_fixtures.ts";

/** Return the running driver `origin~k` over `child`, reading the child START's params. */
function driver(origin: string, k: number, child: Flow): LoopNode {
  const d = new LoopNode(`${origin}~${k}`, {
    flowId: "f",
    child,
    predicateKind: "while",
    predicate: "${x}",
    maxIters: 10,
    iteration: k,
    originId: origin,
  });
  d.params = child.nodes.get(child.startId)!.params;
  return d;
}

/** Return a valid zero-field loop body. */
function emptyChild(): CompiledFlow {
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: [] })],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("boundary", START_ID, END_ID, { ordering: true })],
    {
      wiring: new Map([
        [START_ID, new Map()],
        [END_ID, new Map()],
      ]),
    },
  );
}

test("continue subgraph body keyed on origin not driver", () => {
  const child = childFlow();
  // The origin is lp; the body must key on lp, not on lp~0.
  const d = driver("lp", 1, child);
  const sg = loopContinueSubgraph(child, "lp", { x: 1n }, 0, d);
  expect(sg).toBeInstanceOf(Flow);
  // "lp#0": origin-keyed, not lp~0#0.
  const callsite = mapCallsite("lp", 0);
  expect(sg.startId).toBe(ns(callsite, child.startId));
  expect(sg.endId).toBe(ns(callsite, END_ID));
  expect([...sg.nodes.keys()].every((nid) => nid.startsWith(`${callsite}/`) || nid === "lp~1")).toBe(true);
});

test("continue body end has no commit as", () => {
  const child = childFlow();
  const sg = loopContinueSubgraph(child, "lp", { x: 1n }, 0, driver("lp", 1, child));
  expect(sg.nodes.get(ns(mapCallsite("lp", 0), END_ID))!.commitAs).toBeNull();
});

test("continue splices next driver and edge", () => {
  const child = childFlow();
  const d = driver("lp", 1, child);
  const sg = loopContinueSubgraph(child, "lp", { x: 1n }, 0, d);
  expect(sg.nodes.has("lp~1")).toBe(true);
  const bodyEnd = ns(mapCallsite("lp", 0), END_ID);
  expect(sg.edges.some((e) => e.from === bodyEnd && e.to === "lp~1")).toBe(true);
  expect(sg.wiring.get("lp~1")).toEqual(new Map([["x", new Binding(`\${${bodyEnd}.output.x}`)]]));
});

test("continue context prefixes body but not driver local origin", () => {
  const child = childFlow();
  const outer = new OriginRef("root/nodes/call");
  const loopOrigin = new OriginRef("child/nodes/loop");
  const d = driver("lp", 1, child);
  d.origin = loopOrigin;
  d.originContext = [outer];

  const sg = loopContinueSubgraph(child, "lp", { x: 1n }, 0, d, { originContext: [outer, loopOrigin] });

  const bodyIds = [...sg.nodes.keys()].filter((nodeId) => nodeId !== d.id);
  expect(bodyIds.length).toBeGreaterThan(0);
  for (const nodeId of bodyIds) expect(sg.nodes.get(nodeId)!.originContext).toEqual([outer, loopOrigin]);
  expect(sg.nodes.get(d.id)!.originContext).toEqual([outer]);
  expect(sg.nodes.get(d.id)!.failureTrace()).toEqual([outer, loopOrigin]);
});

test("continue seeds only fields declared by the body start", () => {
  // Carry loop-only state forward without adding undeclared child START wiring.
  const child = childFlow();
  const d = driver("lp", 1, child);
  d.params = [new Parameter("x"), new Parameter("exited")];

  const sg = loopContinueSubgraph(child, "lp", { x: 1n, exited: false }, 0, d);

  expect(sg.wiring.get(sg.startId)).toEqual(new Map([["x", new Binding(1n)]]));
  const bodyEnd = ns(mapCallsite("lp", 0), END_ID);
  expect(sg.wiring.get(d.id)).toEqual(
    new Map([
      ["x", new Binding(`\${${bodyEnd}.output.x}`)],
      ["exited", new Binding(`\${${bodyEnd}.output.exited}`)],
    ]),
  );
  expect(sg.inputConnections).toEqual([new SeedConnection(sg.startId, "x", ["x"])]);
  expect(sg.childBoundaries).toEqual([new ChildFlowBoundary(sg.startId, bodyEnd)]);
});

/** Return a compiled LOOP origin carrying `x` and loop-only `exited`. */
function recordOrigin(returnedFields: readonly string[] | null): LoopNode {
  const origin = new LoopNode("lp", {
    flowId: "f",
    child: childFlow(),
    predicateKind: "while",
    predicate: "${x}",
    maxIters: 10,
    returnedFields,
  });
  origin.params = [new Parameter("x"), new Parameter("exited")];
  return origin;
}

test("continue wires only returned fields and connects passthrough", () => {
  // Omitted fields reach the next driver through its default and a passthrough seed recipe.
  const child = childFlow();
  const carried = { x: 1n, exited: false };
  const d = recordOrigin(["x"]).respawn(1, carried);

  const sg = loopContinueSubgraph(child, "lp", carried, 0, d);

  const bodyEnd = ns(mapCallsite("lp", 0), END_ID);
  expect(sg.wiring.get(d.id)).toEqual(new Map([["x", new Binding(`\${${bodyEnd}.output.x}`)]]));
  const driverEdges = sg.edges.filter((e) => e.from === bodyEnd && e.to === d.id);
  expect(driverEdges).toEqual([new Edge(`${bodyEnd}->${d.id}#0`, bodyEnd, d.id, { inputGroup: "x" })]);
  expect(sg.inputConnections).toEqual([
    new SeedConnection(sg.startId, "x", ["x"]),
    new SeedConnection(d.id, "exited", ["exited"]),
  ]);
  expect(() => validateGrowthMetadata(sg)).not.toThrow();
});

test("continue total driver wires every carried field", () => {
  const child = childFlow();
  const carried = { x: 1n, exited: false };
  const d = recordOrigin(null).respawn(1, carried);

  const sg = loopContinueSubgraph(child, "lp", carried, 0, d);

  const bodyEnd = ns(mapCallsite("lp", 0), END_ID);
  expect(sg.wiring.get(sg.startId)).toEqual(new Map([["x", new Binding(1n)]]));
  expect(sg.wiring.get(d.id)).toEqual(
    new Map([
      ["x", new Binding(`\${${bodyEnd}.output.x}`)],
      ["exited", new Binding(`\${${bodyEnd}.output.exited}`)],
    ]),
  );
  expect(sg.edges.filter((e) => e.to === d.id)).toEqual([
    new Edge(`${bodyEnd}->${d.id}#0`, bodyEnd, d.id, { inputGroup: "x" }),
    new Edge(`${bodyEnd}->${d.id}#1`, bodyEnd, d.id, { inputGroup: "exited" }),
  ]);
  expect(sg.inputConnections).toEqual([new SeedConnection(sg.startId, "x", ["x"])]);
  expect(sg.childBoundaries).toEqual([new ChildFlowBoundary(sg.startId, bodyEnd)]);
  expect(() => validateGrowthMetadata(sg)).not.toThrow();
});

test("continue orders zero field body before next driver", () => {
  const child = emptyChild();
  const d = new LoopNode("lp~1", {
    flowId: "f",
    child,
    childInputs: [],
    predicateKind: "times",
    times: 2,
    maxIters: 2,
    iteration: 1,
    originId: "lp",
  });
  d.params = [];

  const sg = loopContinueSubgraph(child, "lp", {}, 0, d);
  const bodyEnd = ns(mapCallsite("lp", 0), END_ID);

  expect(sg.edges).toContainEqual(new Edge(`${bodyEnd}->lp~1#0`, bodyEnd, "lp~1", { ordering: true }));
  expect(sg.wiring.get("lp~1")).toEqual(new Map());
  expect(sg.inputConnections).toEqual([]);
  expect(sg.childBoundaries).toEqual([new ChildFlowBoundary(sg.startId, bodyEnd)]);
});

test("continue keys body on origin when driver id differs", () => {
  // A running driver `lp~2` with origin `lp` must still splice bodies under `lp#k`, never `lp~2#k`.
  const child = childFlow();
  const d = driver("lp", 3, child);
  const sg = loopContinueSubgraph(child, "lp", { x: 5n }, 2, d);
  expect(sg.nodes.has(ns(mapCallsite("lp", 2), child.startId))).toBe(true);
  expect([...sg.nodes.keys()].some((nid) => nid.startsWith("lp~2#"))).toBe(false);
  expect(sg.nodes.has("lp~3")).toBe(true);
});

