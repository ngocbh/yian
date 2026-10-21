/**
 * Seed-first growth, nodes describe expansion and compile constructs topology: a port of
 * `tests/engine/test_seed_first_growth.py`.
 *
 * Ported: `test_call_map_and_loop_return_seed_only`,
 * `test_loop_respawn_preserves_diagnostic_origin_separately_from_origin_id`,
 * `test_compile_builder_derives_prefix_from_each_spawner_failure_trace`,
 * `test_source_free_spawners_do_not_invent_generated_context`,
 * `test_compile_builder_propagates_each_spawner_memory_occurrence`,
 * `test_nested_map_then_call_extends_context_only_when_inner_expands`,
 * `test_compile_builder_reconstructs_every_growth_kind`,
 * `test_compile_builder_marks_only_explicit_legal_seed_rejection`,
 * `test_compile_builder_keeps_an_unsupported_kind_outside_growth_rejection` and
 * `test_compile_builder_preserves_helper_fault_identity_after_seed_validation`.
 *
 * Changed:
 * - `growthProducers` is async, as `run` is; ints are `bigint`s, as the reducer's canonical seed
 *   holds them, the LOOP iteration included (`[{ x: 1n }, 0n]`, where Python's seed is a tuple);
 * - Python's `ValueError` is a `RangeError`;
 * - `test_compile_builder_keeps_an_unsupported_kind_outside_growth_rejection`: the spawner is a
 *   `FuncNode` (kind `code`), since `Node` is abstract, and the message is pinned whole;
 * - `test_compile_builder_preserves_helper_fault_identity_after_seed_validation`: no monkeypatched
 *   `call_subgraph`; the CALL's child throws the fault from its `inputConnections` getter, which
 *   the builder reads after seed validation.
 *
 * Not ported: the `call-key-type` row of `test_compile_builder_marks_only_explicit_legal_seed_rejection`,
 * since JS record keys are always text.
 *
 * TS-only:
 * - a float LOOP iteration and one above 2^53-1 are rejected, beside Python's boolean and negative
 *   rows;
 * - two `buildGrow` calls on one spawner give equal fragments and leave its child template
 *   unchanged.
 */

import { expect, test } from "bun:test";

import { buildGrow, callSubgraph, loopContinueSubgraph, mapSubgraph } from "../../src/compile/expand.ts";
import { expansionDigest } from "../../src/compile/manifest.ts";
import {
  CompiledFlow,
  copyGraph,
  Edge,
  END_ID,
  Flow,
  FlowOutput,
  GrowthRejected,
  START_ID,
} from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { MemoryOccurrenceKind, MemoryOccurrenceSegment, MemoryScope } from "../../src/memory/contracts.ts";
import { Grow, type Node, NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Assertion, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { FuncNode } from "./run_state_fixtures.ts";

/** A required int parameter that coerces supplied strings. */
function intParameter(name: string): Parameter {
  return new Parameter(name, {
    contract: new ValueContract(Type.scalar(ValueKind.INTEGER)),
    required: true,
    coerceStrings: true,
  });
}

/** A child START -> head (reads `${input.x}`) -> tail (reads `${head.output}`) -> END. */
function childFlow(): CompiledFlow {
  const head = new CodeNode("head", { code: 'return inputs["x"] + 1' });
  head.params = [new Parameter("x")];
  const tail = new CodeNode("tail", { code: 'return inputs["r"] * 10' });
  tail.params = [new Parameter("r")];
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: [intParameter("x")] })],
    ["head", head],
    ["tail", tail],
    [END_ID, EndNode.value(END_ID, { outputName: "out" })],
  ]);
  const edges = [
    new Edge("__start__->head#0", START_ID, "head", { inputGroup: "x" }),
    new Edge("head->tail#0", "head", "tail", { inputGroup: "r" }),
    new Edge("tail->__end__#0", "tail", END_ID, { inputGroup: "out" }),
  ];
  const wiring = new Map([
    ["head", new Map([["x", "${input.x}"]])],
    ["tail", new Map([["r", "${head.output}"]])],
    [START_ID, new Map([["x", "${input.x}"]])],
    [END_ID, new Map([["out", "${tail.output}"]])],
  ]);
  return CompiledFlow.fromParts(nodes, edges, { outputs: [new FlowOutput("out", "${tail.output}")], wiring });
}

/** A parent whose CALL `sp` of `child` reads `${input.x}` and whose `after` reads `${sp.output}`. */
function parentWithSpawner(child: CompiledFlow): CompiledFlow {
  const sp = new CallNode("sp", { flowId: "child", child, childInputs: child.nodes.get(child.startId)!.params });
  sp.params = [new Parameter("x")];
  const after = new FuncNode("after", (inputs) => inputs["v"], { reads: { v: "${sp.output}" } });
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: [intParameter("x")] })],
    ["sp", sp],
    ["after", after],
    [END_ID, EndNode.value(END_ID, { outputName: "out" })],
  ]);
  const edges = [
    new Edge("__start__->sp", START_ID, "sp"),
    new Edge("sp->after#0", "sp", "after", { inputGroup: "v" }),
    new Edge("after->__end__#0", "after", END_ID, { inputGroup: "out" }),
  ];
  const wiring = new Map([
    ["sp", new Map([["x", "${input.x}"]])],
    ["after", new Map([["v", "${sp.output}"]])],
    [START_ID, new Map([["x", "${input.x}"]])],
    [END_ID, new Map([["out", "${after.output}"]])],
  ]);
  return CompiledFlow.fromParts(nodes, edges, { outputs: [new FlowOutput("out", "${after.output}")], wiring });
}

/** One live seed-producing spawner of every legal grow kind, with the `Grow` its `run` gives. */
async function growthProducers(): Promise<[[CallNode, Grow], [MapNode, Grow], [LoopNode, Grow]]> {
  const child = childFlow();

  const call = new CallNode("call", { flowId: "child", child });
  call.params = [new Parameter("x")];
  call.postAsserts = [new Assertion("output == x", "output == x")];
  const callGrow = await call.run({ x: 1n });

  const mapNode = new MapNode("map", { flowId: "child", child });
  const mapGrow = await mapNode.run({ over: [1n, 2n] }, { bindItem: (item) => ({ x: item }) });

  const loop = new LoopNode("loop", {
    flowId: "child",
    child,
    predicateKind: "while",
    predicate: "x > 0",
    maxIters: 3,
  });
  loop.params = child.nodes.get(child.startId)!.params;
  const loopGrow = await loop.run({ x: 1n });

  return [
    [call, callGrow as Grow],
    [mapNode, mapGrow as Grow],
    [loop, loopGrow as Grow],
  ];
}

/** Compare construction-relevant topology and node-owned execution metadata. */
function expectSameFragment(actual: Flow, expected: Flow): void {
  expect(actual).toBeInstanceOf(Flow);
  expect(actual.startId).toBe(expected.startId);
  expect(actual.endId).toBe(expected.endId);
  expect(new Set(actual.nodes.keys())).toEqual(new Set(expected.nodes.keys()));
  expect(actual.edges).toEqual(expected.edges);
  expect(actual.wiring).toEqual(expected.wiring);
  for (const [nodeId, left] of actual.nodes) {
    const right = expected.nodes.get(nodeId)!;
    expect(left.constructor).toBe(right.constructor);
    expect(left.commitAs).toBe(right.commitAs);
    expect(left.originId).toBe(right.originId);
    expect(left.originContext).toEqual(right.originContext);
    expect(left.memoryScope).toEqual(right.memoryScope);
    expect(left.postAsserts).toEqual(right.postAsserts);
  }
}

test("call map and loop return seed only", async () => {
  const producers = await growthProducers();

  expect(producers.every(([, grow]) => grow instanceof Grow)).toBe(true);
  expect(producers.every(([, grow]) => !("subgraph" in grow))).toBe(true);
  expect(producers[0][1].seed).toEqual({ x: 1n });
  expect(producers[1][1].seed).toEqual([{ x: 1n }, { x: 2n }]);
  expect(producers[2][1].seed).toEqual([{ x: 1n }, 0n]);
});

test("loop respawn preserves diagnostic origin separately from origin id", () => {
  const loop = new LoopNode("loop", { flowId: "child", originId: "runtime-loop" });
  const authoredOrigin = new OriginRef("root/node/loop");
  const outerOrigin = new OriginRef("root/node/call");
  loop.origin = authoredOrigin;
  loop.originContext = [outerOrigin];

  const clone = loop.respawn(2);

  expect(clone.origin).toBe(authoredOrigin);
  expect(clone.originContext).toEqual([outerOrigin]);
  expect(clone.failureTrace()).toEqual([outerOrigin, authoredOrigin]);
  expect(clone.originId).toBe("runtime-loop");
});

test("compile builder derives prefix from each spawner failure trace", async () => {
  const [[call, callGrow], [mapNode, mapGrow], [loop, loopGrow]] = await growthProducers();
  const outer = new OriginRef("root/nodes/outer");
  const localOrigins = [
    new OriginRef("root/nodes/call"),
    new OriginRef("root/nodes/map"),
    new OriginRef("root/nodes/loop"),
  ] as const;
  for (const [i, spawner] of [call, mapNode, loop].entries()) {
    spawner.originContext = [outer];
    spawner.origin = localOrigins[i]!;
  }

  const callFragment = buildGrow(call, callGrow.seed as JsonValue);
  for (const node of callFragment.nodes.values()) expect(node.originContext).toEqual([outer, localOrigins[0]]);

  const mapFragment = buildGrow(mapNode, mapGrow.seed as JsonValue);
  for (const node of mapFragment.nodes.values()) expect(node.originContext).toEqual([outer, localOrigins[1]]);

  const loopFragment = buildGrow(loop, loopGrow.seed as JsonValue);
  const nextDriver = loopFragment.nodes.get("loop~1")!;
  for (const [nodeId, node] of loopFragment.nodes) {
    if (nodeId !== nextDriver.id) expect(node.originContext).toEqual([outer, localOrigins[2]]);
  }
  expect(nextDriver.originContext).toEqual([outer]);
  expect(nextDriver.failureTrace()).toEqual([outer, localOrigins[2]]);
});

test("source free spawners do not invent generated context", async () => {
  for (const [spawner, grow] of await growthProducers()) {
    const fragment = buildGrow(spawner, grow.seed as JsonValue);
    expect([...fragment.nodes.values()].every((node) => node.originContext.length === 0)).toBe(true);
  }
});

test("compile builder propagates each spawner memory occurrence", async () => {
  const [[call, callGrow], [mapNode, mapGrow], [loop, loopGrow]] = await growthProducers();
  const outer = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["outer"], 3);
  for (const spawner of [call, mapNode, loop]) {
    const child = copyGraph(spawner.child as Flow);
    spawner.child = child;
    spawner.memoryScope = new MemoryScope(["outer", spawner.id], { occurrencePath: [outer] });
    for (const [childId, childNode] of child.nodes) {
      childNode.memoryScope = new MemoryScope(["outer", spawner.id, childId]);
    }
  }

  const callFragment = buildGrow(call, callGrow.seed as JsonValue);
  for (const node of callFragment.nodes.values()) expect(node.memoryScope!.occurrencePath).toEqual([outer]);

  const mapFragment = buildGrow(mapNode, mapGrow.seed as JsonValue);
  const mapItems = [...mapFragment.nodes]
    .filter(([nodeId]) => nodeId.includes("#"))
    .map(([, node]) => node.memoryScope!.occurrencePath.at(-1)!.index);
  expect(new Set(mapItems)).toEqual(new Set([0, 1]));
  for (const [nodeId, node] of mapFragment.nodes) {
    if (!nodeId.includes("#")) continue;
    const item = node.memoryScope!.occurrencePath.at(-1)!;
    expect([item.kind, item.staticPath]).toEqual([MemoryOccurrenceKind.MAP_ITEM, ["outer", "map"]]);
  }

  const loopFragment = buildGrow(loop, loopGrow.seed as JsonValue);
  const nextDriver = loopFragment.nodes.get("loop~1")!;
  expect(nextDriver.memoryScope!.occurrencePath).toEqual([outer]);
  for (const [nodeId, node] of loopFragment.nodes) {
    if (nodeId === nextDriver.id) continue;
    expect(node.memoryScope!.occurrencePath.at(-1)).toEqual(
      new MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ["outer", "loop"], 0),
    );
  }
});

test("nested map then call extends context only when inner expands", () => {
  // A nested spawner carries outer ancestry without pre-stamping its baked child.
  const leaf = childFlow();
  const middle = parentWithSpawner(leaf);
  const outerOrigin = new OriginRef("root/nodes/each");
  const innerOrigin = new OriginRef("middle/nodes/inner");
  middle.nodes.get("sp")!.origin = innerOrigin;
  const outer = new MapNode("each", { flowId: "middle", child: middle });
  outer.origin = outerOrigin;

  const mapFragment = buildGrow(outer, [{ x: 1n }]);
  const inner = mapFragment.nodes.get("each#0/sp") as CallNode;

  expect(inner.originContext).toEqual([outerOrigin]);
  expect([...(inner.child as Flow).nodes.values()].every((node) => node.originContext.length === 0)).toBe(true);

  const callFragment = buildGrow(inner, { x: 1n });

  expect(callFragment.nodes.get("each#0/sp/head")!.originContext).toEqual([outerOrigin, innerOrigin]);
});

test("compile builder reconstructs every growth kind", async () => {
  const [[call, callGrow], [mapNode, mapGrow], [loop, loopGrow]] = await growthProducers();
  const callSeed = callGrow.seed as JsonObject;
  const mapSeed = mapGrow.seed as JsonValue[];
  const loopSeed = loopGrow.seed as JsonValue[];

  const callExpected = callSubgraph(call.child as Flow, call.id, { ...callSeed }, {
    postAsserts: call.postAsserts,
    inputNames: new Set((call.params ?? []).map((param) => param.name)),
  });
  expectSameFragment(buildGrow(call, callSeed), callExpected);

  const mapExpected = mapSubgraph(
    mapNode.child as Flow,
    mapNode.id,
    mapSeed.map((record) => ({ ...(record as JsonObject) })),
  );
  expectSameFragment(buildGrow(mapNode, mapSeed), mapExpected);

  const [carried, iteration] = loopSeed as [JsonObject, bigint];
  const loopExpected = loopContinueSubgraph(
    loop.child as Flow,
    loop.originId!,
    { ...carried },
    Number(iteration),
    loop.respawn(Number(iteration) + 1),
  );
  expectSameFragment(buildGrow(loop, loopSeed), loopExpected);
});

test.each([
  ["call-container", 0, [], TypeError, "CALL growth seed must be a record"],
  ["call-empty-key", 0, { "": "value" }, RangeError, "CALL growth seed record keys must be non-empty"],
  ["map-container", 1, {}, TypeError, "MAP growth seed must be a list of records"],
  ["map-item", 1, [[]], TypeError, "MAP item 0 growth seed must be a record"],
  ["loop-arity", 2, [{}], RangeError, "LOOP growth seed must be a two-item (record, iteration) pair"],
  ["loop-record", 2, [[], 0n], TypeError, "LOOP carried growth seed must be a record"],
  ["loop-boolean-index", 2, [{}, true], TypeError, "LOOP growth seed iteration must be an integer"],
  ["loop-negative-index", 2, [{}, -1n], RangeError, "LOOP growth seed iteration must be non-negative"],
  ["loop-float-index", 2, [{}, 1], TypeError, "LOOP growth seed iteration must be an integer"],
  [
    "loop-unsafe-index",
    2,
    [{}, 2n ** 53n],
    RangeError,
    "LOOP growth seed iteration must be at most 9007199254740991, got 9007199254740992",
  ],
] as const)("compile builder marks only explicit legal seed rejection: %s", async (_id, index, seed, cls, message) => {
  // Malformed legal-kind seeds keep their typed rejection in one marker.
  const spawner = (await growthProducers())[index][0];

  let error: unknown;
  try {
    buildGrow(spawner, seed as JsonValue);
  } catch (caught) {
    error = caught;
  }

  expect((error as Error).constructor).toBe(GrowthRejected);
  const original = (error as GrowthRejected).original;
  expect(original.constructor).toBe(cls);
  expect(original.message).toBe(message);
  expect((error as Error).cause).toBe(original);
});

test("compile builder keeps an unsupported kind outside growth rejection", () => {
  // A trait/kind compiler mismatch is not node-produced legal-kind seed data.
  const spawner = new FuncNode("unsupported", () => null);
  expect(spawner.kind).toBe(NodeKind.CODE);

  let error: unknown;
  try {
    buildGrow(spawner, {});
  } catch (caught) {
    error = caught;
  }

  expect((error as Error).constructor).toBe(RangeError);
  expect((error as Error).message).toBe("node 'unsupported' (code) cannot build graph growth");
});

test("compile builder preserves helper fault identity after seed validation", async () => {
  // Only seed validation marks rejection; compiler construction faults stay raw.
  const call = (await growthProducers())[0][0];
  const expected = new TypeError("compiler helper fault");
  Object.defineProperty(call.child, "inputConnections", {
    get() {
      throw expected;
    },
  });

  let error: unknown;
  try {
    buildGrow(call, { x: 1n });
  } catch (caught) {
    error = caught;
  }

  expect(error).toBe(expected);
});

test("two builds of one spawner give equal fragments and leave its template unchanged", async () => {
  const [call, grow] = (await growthProducers())[0];
  const seed = grow.seed as JsonValue;
  const template = (call.child as Flow).nodes.get(END_ID)!;

  const first = buildGrow(call, seed);
  const second = buildGrow(call, seed);

  expect(expansionDigest(second)).toBe(expansionDigest(first));
  for (const fragment of [first, second]) {
    expect(fragment.nodes.get("call/__end__")!.params!.map((param) => param.name)).toEqual(["out", "__assert_input_x"]);
  }
  expect(template.params!.map((param) => param.name)).toEqual(["out"]);
  expect(template.postAsserts).toEqual([]);
  expect((call.child as Flow).wiring.get(END_ID)!.has("__assert_input_x")).toBe(false);
});
