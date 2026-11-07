/**
 * The pure CALL builder `callSubgraph`, which pins its output to the CALL residual's clone shape.
 *
 * `callSubgraph(child, callsite, record)` wraps `cloneChild`, bakes `commitAs = callsite` on the
 * cloned child END, and returns a `Flow` (the self-describing fragment a CALL spawner grows into).
 * Its `startId` is the namespaced child START, its `endId` the namespaced child END carrying
 * `commitAs === callsite`, and every node id is `ns`-prefixed under the callsite.
 */

import { expect, test } from "bun:test";

import { callSubgraph, cloneChild, mapSubgraph } from "../../src/compile/expand.ts";
import { END_ID, Flow, START_ID, validateGrowthMetadata } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { MemoryOccurrenceKind, MemoryOccurrenceSegment, MemoryScope } from "../../src/memory/contracts.ts";
import { Binding, ChildFlowBoundary, Parameter, SeedConnection } from "../../src/nodes/contracts.ts";
import { mapCallsite, ns } from "../../src/nodes/ids.ts";
import { childFlow } from "../helpers/run_state_fixtures.ts";

/** Return an instance of `cls` holding `fields`, made without running its constructor. */
function forged<T>(cls: abstract new (...args: never[]) => T, fields: object): T {
  return Object.assign(Object.create(cls.prototype) as object, fields) as T;
}

/** Reassign `flow`'s readonly growth recipes. */
function setRecipes(flow: Flow, recipes: Partial<Pick<Flow, "inputConnections" | "childBoundaries">>): void {
  Object.assign(flow, recipes);
}

test("call subgraph matches grow call type", () => {
  const child = childFlow();
  const sg = callSubgraph(child, "c0", { x: 1n });

  expect(sg).toBeInstanceOf(Flow);
  // startId is the namespaced child START (the sole seed point).
  expect(sg.startId).toBe(ns("c0", child.startId));
  // The terminal (endId) is the namespaced child END; its Output commits under the callsite.
  expect(sg.endId).toBe(ns("c0", END_ID));
  const terminal = sg.nodes.get(ns("c0", END_ID))!;
  expect(terminal.commitAs).toBe("c0");
  // Every cloned node id is namespaced under the callsite.
  expect([...sg.nodes.keys()].every((nid) => nid.startsWith("c0/"))).toBe(true);
  expect(new Set(sg.nodes.keys())).toEqual(new Set([...child.nodes.keys()].map((nid) => ns("c0", nid))));
  expect(sg.inputConnections).toEqual([new SeedConnection(sg.startId, "x", ["x"])]);
  expect(sg.childBoundaries).toEqual([new ChildFlowBoundary(sg.startId, sg.endId)]);
});

test.each([
  ["boolean-index", RangeError],
  ["negative-index", RangeError],
  ["deep-selector", RangeError],
  ["duplicate-target", RangeError],
  ["missing-start", RangeError],
  ["undeclared-input", RangeError],
  ["unwired-input", RangeError],
  ["invalid-parameters", TypeError],
  ["duplicate-boundary", RangeError],
  ["missing-end", RangeError],
] as const)("growth metadata revalidates exact targets and nested fields: %s", (malformation, error) => {
  const fragment = callSubgraph(childFlow(), "call", { x: 1n });
  const start = fragment.nodes.get(fragment.startId)!;
  const connection = fragment.inputConnections[0]!;
  switch (malformation) {
    case "boolean-index":
    case "negative-index":
    case "deep-selector": {
      const seedPath = {
        "boolean-index": [true, "x"],
        "negative-index": [-1, "x"],
        "deep-selector": [0, "x", "nested"],
      }[malformation];
      setRecipes(fragment, {
        inputConnections: [
          forged(SeedConnection, { childStartId: connection.childStartId, inputName: connection.inputName, seedPath }),
        ],
      });
      break;
    }
    case "duplicate-target":
      setRecipes(fragment, { inputConnections: [...fragment.inputConnections, connection] });
      break;
    case "missing-start":
      setRecipes(fragment, {
        inputConnections: [new SeedConnection("absent", connection.inputName, connection.seedPath)],
      });
      break;
    case "undeclared-input":
      setRecipes(fragment, {
        inputConnections: [new SeedConnection(connection.childStartId, "absent", connection.seedPath)],
      });
      break;
    case "unwired-input":
      fragment.wiring.set(fragment.startId, new Map());
      break;
    case "invalid-parameters":
      start.params = { length: 0 } as unknown as Parameter[];
      break;
    case "duplicate-boundary":
      setRecipes(fragment, { childBoundaries: [...fragment.childBoundaries, ...fragment.childBoundaries] });
      break;
    case "missing-end":
      setRecipes(fragment, { childBoundaries: [new ChildFlowBoundary(fragment.startId, "absent")] });
      break;
  }

  expect(() => validateGrowthMetadata(fragment)).toThrow(error);
});

/** Return a CALL fragment whose non-START node declares an extra input `p`, and that node's id. */
function fragmentWithDefaultedInput(options: { default?: boolean; wired?: boolean } = {}): [Flow, string] {
  const { default: hasDefault = true, wired = false } = options;
  const fragment = callSubgraph(childFlow(), "call", { x: 1n });
  const nodeId = ns("call", "head");
  const node = fragment.nodes.get(nodeId)!;
  node.params = [...node.params!, new Parameter("p", { hasDefault, default: hasDefault ? 1n : null })];
  if (wired) {
    fragment.wiring.set(nodeId, new Map([...fragment.wiring.get(nodeId)!, ["p", new Binding("${input.x}")]]));
  }
  return [fragment, nodeId];
}

test("growth metadata accepts unwired defaulted target", () => {
  const [fragment, nodeId] = fragmentWithDefaultedInput();
  setRecipes(fragment, { inputConnections: [...fragment.inputConnections, new SeedConnection(nodeId, "p", ["p"])] });

  expect(() => validateGrowthMetadata(fragment)).not.toThrow();
});

test.each([
  ["no-default", "must declare a default"],
  ["wired", "must not be wired"],
  ["undeclared-input", "target input must be declared"],
  ["unknown-target", "must name fragment nodes"],
] as const)("growth metadata rejects invalid non start target: %s", (kase, message) => {
  const [fragment, nodeId] = fragmentWithDefaultedInput({ default: kase !== "no-default", wired: kase === "wired" });
  const connection = {
    "no-default": new SeedConnection(nodeId, "p", ["p"]),
    wired: new SeedConnection(nodeId, "p", ["p"]),
    "undeclared-input": new SeedConnection(nodeId, "absent", ["p"]),
    "unknown-target": new SeedConnection("absent", "p", ["p"]),
  }[kase];
  setRecipes(fragment, { inputConnections: [...fragment.inputConnections, connection] });

  expect(() => validateGrowthMetadata(fragment)).toThrow(RangeError);
  expect(() => validateGrowthMetadata(fragment)).toThrow(message);
});

test.each(["call", "map"] as const)(
  "clone preserves nested recipes without reusing runtime target ids: %s",
  (kind) => {
    const child = childFlow();
    const nested = callSubgraph(childFlow(), "inner", { x: 2n });
    for (const [id, node] of nested.nodes) child.nodes.set(id, node);
    for (const [id, nodeWiring] of nested.wiring) child.wiring.set(id, nodeWiring);
    child.edges.push(...nested.edges);
    setRecipes(child, { inputConnections: nested.inputConnections, childBoundaries: nested.childBoundaries });
    const before = [[...child.inputConnections], [...child.childBoundaries]];

    let fragment: Flow;
    let prefix: string;
    let selector: (string | number)[];
    if (kind === "call") {
      fragment = callSubgraph(child, "outer", { x: 1n });
      [prefix, selector] = ["outer", ["x"]];
    } else {
      fragment = mapSubgraph(child, "outer", [{ x: 1n }]);
      [prefix, selector] = [mapCallsite("outer", 0), [0, "x"]];
    }

    expect(fragment.inputConnections).toEqual([
      new SeedConnection(ns(prefix, child.startId), "x", selector),
      new SeedConnection(ns(prefix, nested.startId), "x", selector),
    ]);
    expect(fragment.childBoundaries).toEqual([
      new ChildFlowBoundary(ns(prefix, child.startId), ns(prefix, child.endId)),
      new ChildFlowBoundary(ns(prefix, nested.startId), ns(prefix, nested.endId)),
    ]);
    expect(() => validateGrowthMetadata(fragment)).not.toThrow();
    expect([[...child.inputConnections], [...child.childBoundaries]]).toEqual(before);
  },
);

test("map clone rejects a preserved recipe that would need two item indices", () => {
  const child = childFlow();
  const nested = callSubgraph(childFlow(), "inner", { x: 2n });
  for (const [id, node] of nested.nodes) child.nodes.set(id, node);
  for (const [id, nodeWiring] of nested.wiring) child.wiring.set(id, nodeWiring);
  child.edges.push(...nested.edges);
  const inner = nested.inputConnections[0]!;
  setRecipes(child, {
    inputConnections: [new SeedConnection(inner.childStartId, inner.inputName, [2, "x"])],
    childBoundaries: nested.childBoundaries,
  });

  expect(() => cloneChild(child, "outer", { x: 1n }, { seedPrefix: [0] })).toThrow(RangeError);
  expect(() => cloneChild(child, "outer", { x: 1n }, { seedPrefix: [0] })).toThrow(
    "seed_path must be a name or an index/name tuple",
  );

  expect(child.inputConnections[0]!.seedPath).toEqual([2, "x"]);
});

test("call subgraph prefixes every cloned node including boundaries", () => {
  const child = childFlow();
  const context = [new OriginRef("root/nodes/call")];

  const sg = callSubgraph(child, "c0", { x: 1n }, { originContext: context });

  for (const node of sg.nodes.values()) expect(node.originContext).toEqual(context);
  expect(sg.nodes.get(sg.startId)!.originContext).toEqual(context);
  expect(sg.nodes.get(sg.endId)!.originContext).toEqual(context);
});

test("call subgraph inherits occurrence and keeps child keys", () => {
  const child = childFlow();
  for (const [nodeId, node] of child.nodes) {
    node.memoryScope = new MemoryScope(["outer_map", "call", nodeId], { memories: ["child"] });
  }
  const outer = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["outer_map"], 2);
  const spawnerScope = new MemoryScope(["outer_map", "call"], { occurrencePath: [outer] });

  const fragment = callSubgraph(child, "call", { x: 1n }, { memoryScope: spawnerScope });

  for (const node of fragment.nodes.values()) expect(node.memoryScope!.occurrencePath).toEqual([outer]);
  for (const [nodeId, fragmentId] of [
    [START_ID, fragment.startId],
    [END_ID, fragment.endId],
  ] as const) {
    expect(fragment.nodes.get(fragmentId)!.memoryScope).toEqual(
      new MemoryScope(["outer_map", "call", nodeId], { memories: ["child"], occurrencePath: [outer] }),
    );
  }
});

