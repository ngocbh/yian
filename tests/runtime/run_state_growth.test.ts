/**
 * Transactional flat graph expansion in the run-state reducer: growth checks, pruning, the growth
 * index, and live and replayed growth through `buildGrow`. A live rejection is a `GrowthRejected`
 * whose `original` has the exact class and message.
 *
 * `snapshot` holds the flow by reference, so a growth's topology change needs its own detached copy
 * (`topologySnapshot`).
 */

import { expect, spyOn, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { expansionDigest } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, Flow, GrowthRejected, START_ID } from "../../src/compile/model.ts";
import { current } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import { NodeExpanded, type RunEvent } from "../../src/events.ts";
import { Grow, Node, NodeKind } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import {
  Binding,
  ContractOrigins,
  type OutputContract,
  Parameter,
  PartialRecordContract,
  ValueContract,
  ValueTemplate,
} from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import { MemoryRunLog, NodeExpandedRecord, type RecordEnvelope, type RunRecord } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  type GrowBuilder,
  growthIndex,
  planExpansion,
  type PreparedTransition,
  prepareExpanded,
  prepareExpandedFromPlan,
  prepareExpansionCommit,
  prepareNodeStarted,
  prepareNodeStep,
  prepareRecord,
  prepareRunStopped,
  RunState,
} from "../../src/runtime/state.ts";
import type { JsonValue } from "../../src/typesys/json.ts";
import { Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";
import {
  canonicalChild,
  expectIndexMatchesRebuild,
  expectUnchanged,
  FailingLog,
  fragment,
  fragmentBuilder,
  commitExpansion,
  fragmentIds,
  initializedSpawner,
  reducerWrite,
  parentFlow,
  prepareGrowth,
  prepareOutput,
  reducerSnapshot,
  RUN_ID,
  snapshot,
  Spawner,
} from "../helpers/run_state_fixtures.ts";
import { YianRangeError } from "../../src/typesys/errors.ts";
import { thrownClass } from "../helpers/own_errors.ts";

/** A `Parameter` subclass, near-canonical, proving growth requires the exact carrier. */
class ParameterSubclass extends Parameter {}

/** An ordinary dynamic-overlay node. */
class OverlayNode extends Node {
  readonly kind = NodeKind.CODE;
}

/**
 * Return the topology coordinates `snapshot` holds by reference: each node's class, id, `commitAs` and `originId`, the edges, the wiring, the adjacency
 * and the initial static ids, as detached copies.
 */
function topologySnapshot(state: RunState): Record<string, unknown> {
  const flow = state.flow;
  return {
    nodes: [...flow.nodes].map(([nodeId, node]) => [nodeId, node.constructor, node.id, node.commitAs, node.originId]),
    edges: [...flow.edges],
    wiring: [...flow.wiring].map(([nodeId, wiring]) => [nodeId, [...wiring]]),
    incoming: [...flow.incomingEdges].map(([nodeId, edges]) => [nodeId, [...edges]]),
    outgoing: [...flow.outgoingEdges].map(([nodeId, edges]) => [nodeId, [...edges]]),
    initialStaticIds: [...state.initialStaticIds],
  };
}

/**
 * Install settled dynamic nodes without changing `initialStaticIds`; return their ids. With
 * `residue`, also install a settled `<prefix>/child/residue` the ids don't name.
 */
function installSettledOverlay(state: RunState, options: { prefix?: string; residue?: boolean } = {}): string[] {
  const prefix = options.prefix ?? "old";
  const built = fragment(prefix);
  state.flow.addSubgraph(built.nodes, built.edges, built.wiring);
  for (const nodeId of built.nodes.keys()) {
    state.nodeState.set(nodeId, NodeState.TAKEN);
    state.pool.set(nodeId, { retained: nodeId });
    state.depth.set(nodeId, 4);
    state.wiringOverrides.set(nodeId, { copy: [nodeId] });
  }
  for (const edge of built.edges) state.edgeState.set(edge.id, NodeState.TAKEN);
  if (options.residue) {
    addLiveNode(state, new OverlayNode(`${prefix}/child/residue`));
    state.nodeState.set(`${prefix}/child/residue`, NodeState.TAKEN);
  }
  return [...built.nodes.keys()];
}

/** Assert `fn`'s growth rejection leaves the state, the topology and the log unchanged. */
function expectAtomic(state: RunState, log: MemoryRunLog, fn: () => void): void {
  const before = snapshot(state);
  const beforeTopology = topologySnapshot(state);
  const sequence = log.nextSequence;

  fn();

  expectUnchanged(state, before);
  expect(topologySnapshot(state)).toEqual(beforeTopology);
  expect(log.nextSequence).toBe(sequence);
}

/** Add one live node of `state`'s flow with no edges and no wiring, as an `UNKNOWN` node. */
function addLiveNode(state: RunState, node: Node): void {
  state.flow.addSubgraph(new Map([[node.id, node]]), [], new Map([[node.id, new Map()]]));
  state.nodeState.set(node.id, NodeState.UNKNOWN);
}

/** Return what `fn` throws. */
function caught(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected an error");
}

/** Assert `fn` throws an error of exactly class `cls` whose message matches `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: RegExp): void {
  const error = caught(fn) as Error;
  expect(error.constructor).toBe(thrownClass(cls));
  expect(error.message).toMatch(message);
}

/**
 * Assert `fn` throws a `GrowthRejected` whose `original` has exactly class `cls` and a message
 * matching `message` (a string matches exactly); return that original.
 */
function expectRejected(fn: () => unknown, cls: new (...args: never[]) => Error, message: RegExp | string): Error {
  const error = caught(fn) as GrowthRejected;
  expect(error.constructor).toBe(GrowthRejected);
  expect(error.original.constructor).toBe(thrownClass(cls));
  if (typeof message === "string") expect(error.original.message).toBe(message);
  else expect(error.original.message).toMatch(message);
  expect(error.cause).toBe(error.original);
  return error.original;
}

test("a spawner that leaves flight with an unadmitted expansion commit is refused before the next append", () => {
  const { state, log } = initializedSpawner();
  expect(commitExpansion(log, state, "spawn", new Grow({}), fragmentBuilder)).toBeGreaterThan(0);
  expect([...state.pendingCommits.values()].some((pending) => pending.spawner === "spawn" && !pending.admitted)).toBe(true);
  // Preparing checks the nodes the commits touched; only the spawner's write below is left.
  const stop = prepareRunStopped(state, "signal", "");
  reducerWrite(state, "spawn", () => {
    state.inFlight.delete("spawn");
    state.acceptedStarts.delete("spawn");
  });
  const before = log.nextSequence;

  expect(() => commitPrepared(log, state, stop)).toThrow(
    "node starts, steps and pending commits must match the unsettled invocations",
  );
  expect(log.nextSequence).toBe(before);
});

test("growth canonicalizes the seed and isolates the builder, record and observers", () => {
  const { state, log } = initializedSpawner();
  const originalSeed = { items: [{ label: "stable" }], nested: { values: [1, 2] } };
  const builderInputs: unknown[] = [];
  const mutatingBuilder: GrowBuilder = (spawner, seed) => {
    builderInputs.push(structuredClone(seed));
    (seed as { nested: { values: number[] } }).nested.values.push(99);
    return fragment(spawner.id);
  };
  commitExpansion(log, state, "spawn", new Grow(originalSeed), fragmentBuilder);
  const before = snapshot(state);
  const beforeTopology = topologySnapshot(state);

  const prepared = prepareExpanded(state, "spawn", new Grow(originalSeed), mutatingBuilder);

  expectUnchanged(state, before);
  expect(topologySnapshot(state)).toEqual(beforeTopology);
  expect(builderInputs).toEqual([{ items: [{ label: "stable" }], nested: { values: [1, 2] } }]);
  expect(prepared.record).toEqual(
    new NodeExpandedRecord(
      "spawn",
      [],
      expansionDigest(fragment("spawn")),
    ),
  );
  // The record holds no seed (a replay rebuilds it); the plan keeps a copy apart from the
  // observer's and the caller's.
  expect(Object.hasOwn(prepared.record, "seed")).toBe(false);
  ((prepared.observers[0] as NodeExpanded).grow.seed as { nested: { values: number[] } }).nested.values.push(88);
  originalSeed.nested.values.push(77);
  expect((prepared._plan as { seed: unknown }).seed).toEqual({
    items: [{ label: "stable" }],
    nested: { values: [1, 2] },
  });

  const observers = commitPrepared(log, state, prepared);

  expect(observers).toEqual([
    new NodeExpanded({
      nodeId: "spawn",
      grow: new Grow({ items: [{ label: "stable" }], nested: { values: [1, 2] } }),
    }),
  ]);
  expect(observers[0]).not.toBe(prepared.observers[0]);
  expect(state.nodeState.get("spawn")).toBe(NodeState.EXPANDED);
  expect(state.inFlight.has("spawn")).toBe(false);
  expect(state.ready).toEqual(["spawn/child/start"]);
  expect(state.depth.get("spawn/child/end")).toBe(1);
  // The child START's expansion commit, then the expansion.
  expect(state.nextSequence).toBe(9);
});

test("a planned expansion is refused on another state", () => {
  const { state } = initializedSpawner();
  const planned = planExpansion(state, "spawn", new Grow({}), fragmentBuilder);
  const other = initializedSpawner();
  const before = snapshot(other.state);

  expectError(
    () => prepareExpandedFromPlan(other.state, planned),
    RangeError,
    /^the planned expansion of 'spawn' belongs to another run state$/,
  );

  expectUnchanged(other.state, before);
});

test("a planned expansion is refused once another record follows its plan", () => {
  const { state, log } = initializedSpawner();
  const planned = planExpansion(state, "spawn", new Grow({}), fragmentBuilder);
  for (const [key] of planned.commits) commitPrepared(log, state, prepareExpansionCommit(state, "spawn", key, new Map()));
  commitPrepared(log, state, prepareNodeStep(state, "spawn", { kind: "test" }));

  expectError(() => prepareExpandedFromPlan(state, planned), RangeError, /^the planned expansion of 'spawn' is stale$/);
});

test("growth rejects a cyclic seed before the builder", () => {
  const { state } = initializedSpawner();
  const seed: unknown[] = [];
  seed.push(seed);

  expectRejected(
    () => prepareExpanded(state, "spawn", new Grow(seed), fragmentBuilder),
    TypeCheckError,
    /^value contains itself$/,
  );
});

test("growth rejects a nonfinite seed before the builder", () => {
  const { state } = initializedSpawner();
  let builderCalled = false;
  const builder: GrowBuilder = () => {
    builderCalled = true;
    return fragment("spawn");
  };

  const original = expectRejected(
    () => prepareExpanded(state, "spawn", new Grow({ x: Number.NaN }), builder),
    TypeCheckError,
    /^NaN is not a yian value$/,
  );

  expect((original as TypeCheckError).path).toEqual(["x"]);
  expect(builderCalled).toBe(false);
});

test("growth rejects an excessively nested seed before the builder", () => {
  const limit = current().maxValueNesting;
  const nested = (levels: number): unknown => {
    let seed: unknown = "leaf";
    for (let level = 0; level < levels; level += 1) seed = [seed];
    return seed;
  };
  let builderCalled = false;
  const builder: GrowBuilder = () => {
    builderCalled = true;
    return fragment("spawn");
  };

  const spawned = initializedSpawner();
  prepareGrowth(spawned.log, spawned.state, "spawn", new Grow(nested(limit + 2)), builder);
  expect(builderCalled).toBe(true);

  builderCalled = false;
  expectRejected(
    () => prepareExpanded(initializedSpawner().state, "spawn", new Grow(nested(limit + 3)), builder),
    TypeCheckError,
    new RegExp(`^growth seed at (\\[0\\]){${limit + 2}}: value nests deeper than ${limit} levels, `),
  );
  expect(builderCalled).toBe(false);
});

test("the growth builder receives a detached spawner and its failure is pure", () => {
  const { state, log } = initializedSpawner();
  const liveSpawner = state.flow.nodes.get("spawn")!;
  liveSpawner.title = "live";
  const original = new Error("builder failed");
  let received: Node | null = null;
  const mutatingFailingBuilder: GrowBuilder = (spawner) => {
    received = spawner;
    spawner.title = "builder-mutated";
    throw original;
  };

  let error: unknown;
  expectAtomic(state, log, () => {
    error = caught(() => prepareExpanded(state, "spawn", new Grow({ value: 1 }), mutatingFailingBuilder));
  });

  expect(error).toBe(original);
  expect(received).not.toBeNull();
  expect(received).not.toBe(liveSpawner);
  expect(state.flow.nodes.get("spawn")).toBe(liveSpawner);
  expect(liveSpawner.title).toBe("live");
});

test.each([
  ["commit_as", "absent-target", RangeError, "commit_as target 'absent-target' does not exist"],
  ["commit_as", "", RangeError, "commit_as must be non-empty"],
  ["commit_as", 7, TypeError, "commit_as must be a string or null"],
  ["origin_id", "absent-target", RangeError, "origin_id target 'absent-target' does not exist"],
  ["origin_id", "", RangeError, "origin_id must be non-empty"],
  ["origin_id", 7, TypeError, "origin_id must be a string or null"],
] as const)("growth rejects an invalid node-owned topology target: %s %p", (attribute, target, cls, message) => {
  const { state, log } = initializedSpawner();
  const invalidTargetBuilder: GrowBuilder = (spawner) => {
    const built = fragment(spawner.id);
    const end = built.nodes.get(built.endId)! as unknown as Record<string, unknown>;
    end[attribute === "commit_as" ? "commitAs" : "originId"] = target;
    return built;
  };

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}), invalidTargetBuilder),
      cls,
      new RegExp(`^growth fragment node 'spawn/child/end' ${message}$`),
    ),
  );
});

test("growth rejects an invalid skipped-output hook target", () => {
  // A future hook override can't hide a missing logical output target.
  const { state, log } = initializedSpawner();
  const invalidTargetBuilder: GrowBuilder = (spawner) => {
    const built = fragment(spawner.id);
    built.nodes.get(built.endId)!.skippedOutputTarget = () => "missing";
    return built;
  };

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}), invalidTargetBuilder),
      RangeError,
      /^node 'spawn\/child\/end' skipped output target 'missing' does not exist$/,
    ),
  );
});

test("growth rejects a duplicate redirect owner in the final topology", () => {
  const { state, log } = initializedSpawner();
  const existing = new OverlayNode("existing-owner");
  existing.commitAs = "spawn";
  addLiveNode(state, existing);

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}), fragmentBuilder),
      RangeError,
      `logical output target 'spawn' has multiple skipped output owners: ["existing-owner","spawn/child/end"]`,
    ),
  );
});

test("growth rejects an executable redirect target in the final topology", () => {
  const { state, log } = initializedSpawner();
  const executableTargetBuilder: GrowBuilder = (spawner) => {
    const built = fragment(spawner.id);
    built.nodes.get(built.endId)!.commitAs = END_ID;
    return built;
  };

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}), executableTargetBuilder),
      RangeError,
      /^logical output target '__end__' is also executable$/,
    ),
  );
});

test("growth accepts a disconnected inert redirect placeholder", () => {
  const { state, log } = initializedSpawner();
  const anchor = new OverlayNode("anchor");
  addLiveNode(state, anchor);
  const before = snapshot(state);
  const beforeTopology = topologySnapshot(state);
  const inertTargetBuilder: GrowBuilder = (spawner) => {
    const built = fragment(spawner.id);
    built.nodes.get(built.endId)!.commitAs = anchor.id;
    return built;
  };

  commitExpansion(log, state, "spawn", new Grow({}), inertTargetBuilder);
  const committed = snapshot(state);
  const prepared = prepareExpanded(state, "spawn", new Grow({}), inertTargetBuilder);

  expect(prepared.record).toBeInstanceOf(NodeExpandedRecord);
  expectUnchanged(state, committed);
  expect(snapshot(state)).not.toEqual(before);
  expect(topologySnapshot(state)).toEqual(beforeTopology);
  // The expansion commit for the child START is the only record written.
  expect(log.nextSequence).toBe(8);
});

const END_EDGE = "'spawn/child/start->spawn/child/end'";

test.each([
  ["sourceHandle", 7, TypeError, "source_handle must be a string or null"],
  ["sourceHandle", "", RangeError, "source_handle must be non-empty"],
  ["inputGroup", 7, TypeError, "input_group must be a string or null"],
  ["inputGroup", "", RangeError, "input_group must be non-empty"],
  ["optional", 1, TypeError, "optional must be a bool"],
  ["ordering", 1, TypeError, "ordering must be a bool"],
] as const)("growth rejects an invalid edge field type: %s %p", (attribute, value, cls, message) => {
  const { state, log } = initializedSpawner();
  const invalidEdgeBuilder: GrowBuilder = (spawner) => {
    const built = fragment(spawner.id);
    const edge = built.edges[0]!;
    const fields: Record<string, unknown> = {
      sourceHandle: edge.sourceHandle,
      inputGroup: edge.inputGroup,
      optional: edge.optional,
      ordering: edge.ordering,
    };
    fields[attribute] = value;
    // The TS `Edge` constructor doesn't validate its fields.
    built.edges[0] = new Edge(edge.id, edge.from, edge.to, fields as ConstructorParameters<typeof Edge>[3]);
    return built;
  };

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}), invalidEdgeBuilder),
      cls,
      new RegExp(`^growth fragment edge ${END_EDGE} ${message}$`),
    ),
  );
});

test("a growth append failure leaves the topology and every state coordinate unchanged", () => {
  const { state, log: prior } = initializedSpawner();
  commitExpansion(prior, state, "spawn", new Grow({ value: "stable" }), fragmentBuilder);
  const log = new FailingLog(RUN_ID, prior.records());
  const prepared = prepareExpanded(state, "spawn", new Grow({ value: "stable" }), fragmentBuilder);

  expectAtomic(state, log, () => expectError(() => commitPrepared(log, state, prepared), Error, /^disk full$/));
  expect(log.nextSequence).toBe(prior.nextSequence);
});

test("growth replay rejects a changed but valid builder digest", () => {
  const { state: live, log: liveLog } = initializedSpawner();
  const prepared = prepareGrowth(liveLog, live, "spawn", new Grow({ value: "stable" }), fragmentBuilder);
  const { state: replay, log: replayLog } = initializedSpawner();
  commitExpansion(replayLog, replay, "spawn", new Grow({ value: "stable" }), fragmentBuilder);
  const recorded = (prepared.record as NodeExpandedRecord).topologyDigest;
  const rebuilt = expansionDigest(fragment("spawn", "changed"));

  expectAtomic(replay, replayLog, () =>
    expectError(
      () => prepareRecord(replay, prepared.record, { growBuilder: (spawner) => fragment(spawner.id, "changed") }),
      RangeError,
      new RegExp(`^NodeExpandedRecord topology digest mismatch: recorded '${recorded}', rebuilt '${rebuilt}'$`),
    ),
  );
  expect(replay.flow.nodes.has("spawn/changed/start")).toBe(false);
});

test("growth replay unwraps a fragment rejection to its original", () => {
  const { state: live, log: liveLog } = initializedSpawner();
  const prepared = prepareGrowth(liveLog, live, "spawn", new Grow({}), fragmentBuilder);
  const { state: replay, log } = initializedSpawner();
  commitExpansion(log, replay, "spawn", new Grow({}), fragmentBuilder);
  const cyclicBuilder: GrowBuilder = (spawner) => {
    const built = fragment(spawner.id);
    built.edges.push(new Edge("cycle-back", built.endId, built.startId, { ordering: true }));
    return built;
  };

  let error: unknown;
  expectAtomic(replay, log, () => {
    error = caught(() => prepareRecord(replay, prepared.record, { growBuilder: cyclicBuilder }));
  });

  expect((error as Error).constructor).toBe(YianRangeError);
  expect((error as Error).message).toBe(
    'growth fragment creates a cycle involving ["spawn/child/end","spawn/child/start"]',
  );
  expect((error as Error).cause).toBeUndefined();
});

test("live growth and its record replay produce equal state", () => {
  const { state: live, log: liveLog } = initializedSpawner();
  const { state: replay, log: replayLog } = initializedSpawner();

  // The replay rebuilds the seed from the spawner's forward.
  (replay.flow.nodes.get("spawn") as Spawner).grow = new Grow({ labels: ["a", "b"] });
  const livePrepared = prepareGrowth(liveLog, live, "spawn", new Grow({ labels: ["a", "b"] }), fragmentBuilder);
  const liveObservers = commitPrepared(liveLog, live, livePrepared);
  for (const record of liveLog.records().slice(replayLog.nextSequence, -1)) {
    commitPrepared(replayLog, replay, prepareRecord(replay, record.record));
  }
  const replayPrepared = prepareRecord(replay, livePrepared.record, { growBuilder: fragmentBuilder });
  const replayObservers = commitPrepared(replayLog, replay, replayPrepared);

  expect(replayObservers).toEqual(liveObservers);
  expect(reducerSnapshot(replay)).toEqual(reducerSnapshot(live));
  expect(topologySnapshot(replay)).toEqual(topologySnapshot(live));
  expect(replayLog.records()).toEqual(liveLog.records());
});

test("growth record replay requires the injected builder", () => {
  const { state } = initializedSpawner();
  const record = new NodeExpandedRecord("spawn", [], "a".repeat(64));

  expectError(() => prepareRecord(state, record), TypeError, /grow_builder/);
});

test.each(["call", "map", "loop"])("actual buildGrow splices %s with FIFO order and depth", (kind) => {
  const child = canonicalChild();
  let spawner: Node;
  let grow: Grow;
  let expectedDepth: Record<string, number>;
  if (kind === "call") {
    spawner = new CallNode("spawn", { flowId: "child", child });
    grow = new Grow({});
    expectedDepth = { "spawn/__end__": 3 };
  } else if (kind === "map") {
    spawner = new MapNode("spawn", { flowId: "child", child });
    grow = new Grow([{}, {}]);
    expectedDepth = { "spawn/__end__": 3 };
  } else {
    spawner = new LoopNode("spawn", {
      flowId: "child",
      child,
      childInputs: [],
      predicateKind: "times",
      times: 3,
      maxIters: 3,
      originId: "spawn",
    });
    grow = new Grow([{}, 0n]);
    expectedDepth = { "spawn~1": 2 };
  }
  spawner.params = [];
  const { state, log } = initializedSpawner(parentFlow({ spawner, sibling: true }));
  state.depth.set("spawn", 2);
  const priorPool = state.pool.toWire();

  const prepared = prepareGrowth(log, state, "spawn", grow, buildGrow);
  const observers = commitPrepared(log, state, prepared);

  const record = prepared.record as NodeExpandedRecord;
  expect(record.spawnerId).toBe("spawn");
  expect(record.topologyDigest).toBe(expansionDigest(buildGrow(spawner, (prepared._plan as { seed: JsonValue }).seed)));
  expect(observers).toEqual(prepared.observers);
  expect(state.nodeState.get("spawn")).toBe(NodeState.EXPANDED);
  expect(state.inFlight.has("spawn")).toBe(false);
  expect(state.edgeState.get("spawn-end")).toBe(NodeState.UNKNOWN);
  expect(state.ready[0]).toBe("side");
  expect(state.ready.at(-1)).toBe(fragmentIds(prepared).fragmentStartId);
  expect(state.pool.toWire()).toEqual(priorPool);
  expect(state.result).toBeNull();
  for (const [nodeId, depth] of Object.entries(expectedDepth)) expect(state.depth.get(nodeId)).toBe(depth);
});

test("actual CALL growth allows an omitted optional defaulted child input", () => {
  const spawner = new CallNode("spawn", { flowId: "child", child: canonicalChild({ optionalInput: true }) });
  spawner.params = [];
  const { state, log } = initializedSpawner(parentFlow({ spawner }));

  commitPrepared(log, state, prepareGrowth(log, state, "spawn", new Grow({}), buildGrow));

  const clonedStart = state.flow.nodes.get("spawn/__start__")!;
  expect(clonedStart.params!.map((param) => param.name)).toEqual(["optional"]);
  expect(state.flow.wiring.get("spawn/__start__")).toEqual(new Map());
});

test("a zero-field LOOP growth schedules its next driver after the body", () => {
  const loop = new LoopNode("spawn", {
    flowId: "child",
    child: canonicalChild(),
    childInputs: [],
    predicateKind: "times",
    times: 2,
    maxIters: 2,
    originId: "spawn",
  });
  loop.params = [];
  const { state, log } = initializedSpawner(parentFlow({ spawner: loop }));

  const expansion = prepareGrowth(log, state, "spawn", new Grow([{}, 0n]), buildGrow);
  commitPrepared(log, state, expansion);
  const { fragmentStartId: bodyStart, fragmentEndId: bodyEnd } = fragmentIds(expansion);

  commitPrepared(log, state, prepareNodeStarted(state, bodyStart));
  commitPrepared(log, state, prepareOutput(log, state, bodyStart, {}));
  commitPrepared(log, state, prepareNodeStarted(state, bodyEnd));
  commitPrepared(log, state, prepareOutput(log, state, bodyEnd, null));

  expect(state.ready).toEqual(["spawn~1"]);
});

test.each(["today", "now", "run_id"])("growth accepts the closed system ambient %s", (ambient) => {
  const { state, log } = initializedSpawner();
  const systemReferenceBuilder: GrowBuilder = (spawner) => {
    const built = fragment(spawner.id);
    const end = EndNode.value(built.endId, { outputName: "value" });
    end.commitAs = spawner.id;
    built.nodes.set(built.endId, end);
    built.wiring.set(built.endId, new Map([["value", new Binding(`\${system.${ambient}}`)]]));
    return built;
  };

  expect(prepareGrowth(log, state, "spawn", new Grow({}), systemReferenceBuilder).record).toBeInstanceOf(
    NodeExpandedRecord,
  );
});

test("growth accepts typed fields and a nested MAP item scope", () => {
  const sourceType = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["known", Type.scalar(ValueKind.STRING)]]),
    required: new Set(["known"]),
  });
  const spawner = new Spawner("spawn", {
    outputContract: new ValueContract(
      sourceType,
      new ContractOrigins({
        default: new OriginRef("spawn/output"),
        entries: [[["known"], new OriginRef("spawn/output/known")]],
      }),
    ),
  });
  const { state } = initializedSpawner(parentFlow({ spawner }));
  const typedMapBuilder: GrowBuilder = (parent) => {
    const nodeId = `${parent.id}/nested-map`;
    const node = new MapNode(nodeId, { flowId: "child", child: canonicalChild() });
    node.params = [new Parameter("value")];
    return new Flow(new Map([[nodeId, node]]), [], {
      wiring: new Map([
        [
          nodeId,
          new Map([
            ["over", `\${${parent.id}.output.known}`],
            ["value", "${item.known}"],
          ]),
        ],
      ]),
      startId: nodeId,
      endId: nodeId,
    });
  };

  expect(prepareExpanded(state, "spawn", new Grow({}), typedMapBuilder).record).toBeInstanceOf(NodeExpandedRecord);
});

/** Return a fragment whose required START param has no wiring source. */
function missingRequiredFragment(spawnerId: string): Flow {
  const startId = `${spawnerId}/required/start`;
  const endId = `${spawnerId}/required/end`;
  const start = new StartNode(startId, {
    params: [
      new Parameter("required", {
        contract: new ValueContract(Type.scalar(ValueKind.STRING)),
        required: true,
        coerceStrings: true,
      }),
    ],
  });
  return new Flow(
    new Map<string, Node>([
      [startId, start],
      [endId, EndNode.value(endId, { outputName: null })],
    ]),
    [new Edge("required-edge", startId, endId, { ordering: true })],
    {
      wiring: new Map([
        [startId, new Map()],
        [endId, new Map()],
      ]),
      startId,
      endId,
    },
  );
}

/** Return a one-node MAP fragment omitting its reserved `over` wiring. */
function missingReservedFragment(spawnerId: string): Flow {
  const nodeId = `${spawnerId}/nested-map`;
  const node = new MapNode(nodeId, { flowId: "child", child: canonicalChild() });
  node.params = [];
  return new Flow(new Map([[nodeId, node]]), [], {
    wiring: new Map([[nodeId, new Map()]]),
    startId: nodeId,
    endId: nodeId,
  });
}

/** Return a fragment with a valid param whose source head is absent. */
function unknownReferenceFragment(spawnerId: string): Flow {
  const startId = `${spawnerId}/ref/start`;
  const endId = `${spawnerId}/ref/end`;
  return new Flow(
    new Map<string, Node>([
      [startId, new StartNode(startId, { params: [] })],
      [endId, EndNode.value(endId, { outputName: "value" })],
    ]),
    [new Edge("ref-edge", startId, endId, { inputGroup: "value" })],
    {
      wiring: new Map<string, Map<string, unknown>>([
        [startId, new Map()],
        [endId, new Map([["value", "${absent.output}"]])],
      ]),
      startId,
      endId,
    },
  );
}

/**
 * Replace `built`'s end with a value (or `record`) END committing under `spawn` that wires `value`
 * from `source`, and its edge with one feeding `value`, unless `keepEdge`.
 */
function valueEnd(built: Flow, source: string, options: { record?: boolean; keepEdge?: boolean } = {}): void {
  const end = options.record
    ? EndNode.record(built.endId, { outputNames: ["value"] })
    : EndNode.value(built.endId, { outputName: "value" });
  end.commitAs = "spawn";
  built.nodes.set(built.endId, end);
  built.wiring.set(built.endId, new Map([["value", new Binding(source)]]));
  if (options.keepEdge) return;
  built.edges[0] = new Edge(built.edges[0]!.id, built.startId, built.endId, { inputGroup: "value" });
}

/** Return `built`'s wiring of `nodeId`, writable and open to raw sources. */
function wiringOf(built: Flow, nodeId: string): Map<string, unknown> {
  return built.wiring.get(nodeId) as Map<string, unknown>;
}

/** Build one isolated malformed builder result for the validation table. */
function invalidFragment(name: string): Flow {
  if (name === "missing-required") return missingRequiredFragment("spawn");
  if (name === "missing-reserved") return missingReservedFragment("spawn");
  if (name === "unknown-reference") return unknownReferenceFragment("spawn");
  const built = fragment("spawn");
  const start = built.nodes.get(built.startId)!;
  const edge = built.edges[0]!;
  switch (name) {
    case "missing-start":
      Object.assign(built, { startId: "absent" });
      break;
    case "missing-end":
      Object.assign(built, { endId: "absent" });
      break;
    case "key-id-mismatch":
      built.nodes.delete(built.startId);
      built.nodes.set("wrong-key", start);
      built.wiring.set("wrong-key", built.wiring.get(built.startId)!);
      built.wiring.delete(built.startId);
      break;
    case "empty-node-id":
      built.nodes.delete(built.startId);
      built.wiring.delete(built.startId);
      start.id = "";
      built.nodes.set("", start);
      built.wiring.set("", new Map());
      Object.assign(built, { startId: "" });
      break;
    case "missing-wiring-entry":
      built.wiring.delete(built.endId);
      break;
    case "extra-wiring-key":
      wiringOf(built, built.startId).set("extra", new Binding(1));
      break;
    case "raw-param":
      start.params = ["raw"] as unknown as Parameter[];
      break;
    case "parameter-subclass":
      start.params = [new ParameterSubclass("x")];
      break;
    case "raw-output-contract":
      start.outputContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;
      break;
    case "duplicate-edge-id":
      built.edges.push(new Edge(edge.id, edge.from, edge.to, { ordering: true }));
      break;
    case "empty-edge-id":
      built.edges[0] = new Edge("", edge.from, edge.to, { ordering: true });
      break;
    case "node-collision":
      built.nodes.delete(built.startId);
      built.wiring.delete(built.startId);
      start.id = "spawn";
      built.nodes.set("spawn", start);
      built.wiring.set("spawn", new Map());
      Object.assign(built, { startId: "spawn" });
      built.edges[0] = new Edge(edge.id, "spawn", edge.to, { ordering: true });
      break;
    case "edge-collision":
      built.edges[0] = new Edge("start-spawn", edge.from, edge.to, { ordering: true });
      break;
    case "unknown-endpoint":
      built.edges[0] = new Edge(edge.id, "absent", edge.to, { ordering: true });
      break;
    case "unreachable-node":
      built.nodes.set("spawn/child/orphan", new StartNode("spawn/child/orphan", { params: [] }));
      built.wiring.set("spawn/child/orphan", new Map());
      break;
    case "unknown-system-ambient":
      valueEnd(built, "${system.not_declared}", { keepEdge: true });
      break;
    case "unknown-typed-field":
      start.outputContract = new ValueContract(
        new Type({
          kind: ValueKind.OBJECT,
          fields: new Map([["known", Type.scalar(ValueKind.STRING)]]),
          required: new Set(["known"]),
        }),
      );
      valueEnd(built, `\${${built.startId}.output.absent}`);
      break;
    case "field-on-scalar":
      start.outputContract = new ValueContract(Type.scalar(ValueKind.STRING));
      valueEnd(built, `\${${built.startId}.output.order}`, { record: true });
      break;
    case "unknown-partial-field":
      start.outputContract = new PartialRecordContract([
        ["known", new ValueContract(Type.scalar(ValueKind.STRING))],
        ["opaque", null],
      ]);
      valueEnd(built, `\${${built.startId}.output.absent}`);
      break;
    case "cycle":
      built.edges.push(new Edge("cycle-back", built.endId, built.startId, { ordering: true }));
      break;
    default:
      throw new Error(name);
  }
  return built;
}

const WIRING = "growth fragment node 'spawn/child/end' wiring 'value': reference";

test.each([
  ["missing-start", RangeError, "growth fragment start_id is missing from its nodes"],
  ["missing-end", RangeError, "growth fragment end_id is missing from its nodes"],
  ["key-id-mismatch", RangeError, "growth fragment start_id is missing from its nodes"],
  ["empty-node-id", RangeError, "growth fragment start_id must be a non-empty string"],
  ["missing-wiring-entry", RangeError, "growth fragment wiring keys must exactly equal fragment nodes"],
  [
    "extra-wiring-key",
    RangeError,
    "growth fragment node 'spawn/child/start' wiring parity failed: " +
      `extra=["extra"], missing_reserved=[], missing_required=[]`,
  ],
  [
    "raw-param",
    TypeError,
    "growth fragment node 'spawn/child/start' params must contain Parameter values, got string",
  ],
  [
    "parameter-subclass",
    TypeError,
    "growth fragment node 'spawn/child/start' params must contain Parameter values, got ParameterSubclass",
  ],
  [
    "raw-output-contract",
    TypeError,
    "output contract must be an exact ValueContract, PartialRecordContract, or null, got Type",
  ],
  [
    "missing-required",
    RangeError,
    "growth fragment node 'spawn/required/start' wiring parity failed: " +
      `extra=[], missing_reserved=[], missing_required=["required"]`,
  ],
  [
    "missing-reserved",
    RangeError,
    "growth fragment node 'spawn/nested-map' wiring parity failed: " +
      `extra=[], missing_reserved=["over"], missing_required=[]`,
  ],
  ["duplicate-edge-id", RangeError, `growth fragment edge ID ${END_EDGE} is duplicated`],
  ["empty-edge-id", RangeError, "growth fragment edge IDs must be non-empty strings"],
  ["node-collision", RangeError, "growth fragment node 'spawn' collides with live topology"],
  ["edge-collision", RangeError, "growth fragment edge 'start-spawn' collides with live topology"],
  ["unknown-endpoint", RangeError, `growth fragment edge ${END_EDGE} has unknown source 'absent'`],
  ["unreachable-node", RangeError, `growth fragment nodes are unreachable from its start: ["spawn/child/orphan"]`],
  [
    "unknown-system-ambient",
    RangeError,
    `${WIRING} \${system.not_declared} uses unknown system ambient 'not_declared' (only today/now/run_id)`,
  ],
  [
    "unknown-typed-field",
    RangeError,
    `${WIRING} \${spawn/child/start.output.absent} reads unknown field 'absent' on record`,
  ],
  [
    "unknown-partial-field",
    RangeError,
    `${WIRING} \${spawn/child/start.output.absent} reads unknown field 'absent' on record`,
  ],
  [
    "field-on-scalar",
    RangeError,
    `${WIRING} \${spawn/child/start.output.order} reads field 'order' on type str, which has no fields`,
  ],
  [
    "unknown-reference",
    RangeError,
    "growth fragment node 'spawn/ref/end' wiring 'value': reference ${absent.output} uses unknown namespace 'absent'",
  ],
  ["cycle", RangeError, `growth fragment creates a cycle involving ["spawn/child/end","spawn/child/start"]`],
] as const)("growth rejects an invalid fragment atomically: %s", (name, cls, message) => {
  const { state, log } = initializedSpawner();

  expectAtomic(state, log, () =>
    expectRejected(() => prepareExpanded(state, "spawn", new Grow({}), () => invalidFragment(name)), cls, message),
  );
});

test("growth rejects a field read on a scalar contract", () => {
  const { state, log } = initializedSpawner();

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}), () => invalidFragment("field-on-scalar")),
      RangeError,
      /reads field 'order' on type str, which has no fields$/,
    ),
  );
});

test("a growth builder's non-Flow return is a raw host fault", () => {
  const { state } = initializedSpawner();

  expectError(
    () => prepareExpanded(state, "spawn", new Grow({}), (() => ({})) as unknown as GrowBuilder),
    TypeError,
    /^grow_builder must return a Flow$/,
  );
});

test("growth reference validation reads a Binding source", () => {
  // A valid wrapped reference reaches the same closed-reference scanner as raw text.
  const { state, log } = initializedSpawner();
  const built = unknownReferenceFragment("spawn");
  const origin = new OriginRef("growth-reference");
  wiringOf(built, built.endId).set("value", new Binding(`\${${built.startId}.output}`, { origin }));

  const prepared = prepareGrowth(log, state, "spawn", new Grow({}), () => built);

  const delta = (prepared._plan as { delta: { newNodeIds: readonly string[]; newEdgeIds: readonly string[] } }).delta;
  expect(delta.newNodeIds).toEqual([built.startId, built.endId]);
  expect(delta.newEdgeIds).toEqual(["ref-edge"]);
});

test("growth rejects a Binding-wrapped unknown reference before digesting", () => {
  // A wrapped bad reference fails as node data before manifest identity is needed.
  const { state, log } = initializedSpawner();
  const built = unknownReferenceFragment("spawn");
  const origin = new OriginRef("unknown-growth-reference");
  wiringOf(built, built.endId).set("value", new Binding("${absent.output}", { origin }));

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}), () => built),
      RangeError,
      "growth fragment node 'spawn/ref/end' wiring 'value': reference ${absent.output} uses unknown namespace 'absent'",
    ),
  );
});

test("growth rejects an unknown reference nested in a template", () => {
  // A value template can't hide a bad reference in one of its leaf strings.
  const { state, log } = initializedSpawner();
  const built = unknownReferenceFragment("spawn");
  wiringOf(built, built.endId).set("value", new Binding(new ValueTemplate({ ok: "fine", bad: ["${absent.output}"] })));

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}), () => built),
      RangeError,
      /uses unknown namespace 'absent'$/,
    ),
  );
});

test.each([
  {
    name: "empty prune id",
    grow: () => new Grow({}, [""]),
    original: RangeError as new (...args: never[]) => Error,
    message: /^Grow\.prune node IDs must be non-empty$/,
  },
  {
    name: "non-text prune id",
    grow: () => new Grow({}, [1] as unknown as string[]),
    original: TypeError,
    message: /^Grow\.prune must contain only strings$/,
  },
  {
    name: "Set in the seed",
    grow: () => new Grow({ bad: new Set([1, 2]) }),
    original: TypeCheckError,
    message: /^Set is not a yian value$/,
  },
])("growth wraps a node-produced carrier rejection: $name", ({ grow, original, message }) => {
  const { state } = initializedSpawner();

  expectRejected(() => prepareExpanded(state, "spawn", grow(), fragmentBuilder), original, message);
});

test("growth requires a callable builder, a raw host fault", () => {
  const { state } = initializedSpawner();

  expectError(
    () => prepareExpanded(state, "spawn", new Grow({}), null as unknown as GrowBuilder),
    TypeError,
    /^grow_builder must be callable$/,
  );
});

test("growth rejects a malformed static contract as a host invariant", () => {
  // A corrupt live contract the fragment reads stays outside the fragment boundary.
  const { state, log } = initializedSpawner();
  state.flow.nodes.get("spawn")!.outputContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;
  const readingBuilder: GrowBuilder = (spawner) => {
    const built = unknownReferenceFragment(spawner.id);
    wiringOf(built, built.endId).set("value", `\${${spawner.id}.output}`);
    return built;
  };

  expectAtomic(state, log, () =>
    expectError(
      () => prepareExpanded(state, "spawn", new Grow({}), readingBuilder),
      TypeError,
      /^output contract must be an exact ValueContract, PartialRecordContract, or null, got Type$/,
    ),
  );
});

test("growth enforces the peak node budget before same-transition pruning", () => {
  const { state, log } = initializedSpawner();
  const oldIds = installSettledOverlay(state);
  const newCount = fragment("spawn").nodes.size;
  const finalCount = state.flow.nodes.size - oldIds.length + newCount;
  const peakCount = state.flow.nodes.size + newCount;
  expect(finalCount).toBeLessThan(peakCount);
  state.runDefaults = { ...state.runDefaults, maxTotalNodes: peakCount - 1 };

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}, oldIds), fragmentBuilder),
      RangeError,
      new RegExp(`^expansion exceeded node budget \\(${peakCount - 1}\\) at spawner 'spawn'$`),
    ),
  );
});

test("growth accepts a peak node count equal to the budget", () => {
  const { state, log } = initializedSpawner();
  const oldIds = installSettledOverlay(state);
  const newCount = fragment("spawn").nodes.size;
  state.runDefaults = { ...state.runDefaults, maxTotalNodes: state.flow.nodes.size + newCount };

  commitPrepared(log, state, prepareGrowth(log, state, "spawn", new Grow({}, oldIds), fragmentBuilder));

  expect(oldIds.filter((nodeId) => state.flow.nodes.has(nodeId))).toEqual([]);
  expect(state.flow.nodes.has("spawn/child/start")).toBe(true);
  expectIndexMatchesRebuild(state);
});

test.each([true, -1, 1.5, "1"])("growth rejects an invalid exact depth delta %p", (delta) => {
  const spawner = new Spawner("spawn");
  (spawner as unknown as { growDepthDelta: unknown }).growDepthDelta = delta;
  const { state, log } = initializedSpawner(parentFlow({ spawner }));

  expectAtomic(state, log, () =>
    expectError(
      () => prepareExpanded(state, "spawn", new Grow({}), fragmentBuilder),
      RangeError,
      /^spawner 'spawn' grow_depth_delta must be null, 0, or a positive integer$/,
    ),
  );
});

test("growth treats a corrupt stored depth as a host invariant error", () => {
  const { state, log } = initializedSpawner();
  state.depth.set("spawn", "corrupt" as unknown as number);

  expectAtomic(state, log, () =>
    expectError(
      () => prepareExpanded(state, "spawn", new Grow({}), fragmentBuilder),
      RangeError,
      /^spawner 'spawn' has an invalid parent depth$/,
    ),
  );
});

test("growth stamps a positive reference depth on the boundary", () => {
  // Depth is bookkeeping, not a cap: a spawner already 7 levels deep grows an 8th level.
  const { state, log } = initializedSpawner();
  state.depth.set("spawn", 7);

  commitPrepared(log, state, prepareGrowth(log, state, "spawn", new Grow({}), fragmentBuilder));

  expect(state.depth.get("spawn/child/end")).toBe(8);
});

test("growth prunes only the named settled dynamic overlay coordinates", () => {
  const { state, log } = initializedSpawner();
  const oldIds = installSettledOverlay(state, { residue: true });
  const old = new Set(oldIds);
  const oldEdgeIds = new Set(
    state.flow.edges.filter((edge) => old.has(edge.from) || old.has(edge.to)).map((edge) => edge.id),
  );
  const residueId = "old/child/residue";

  commitPrepared(log, state, prepareGrowth(log, state, "spawn", new Grow({}, oldIds), fragmentBuilder));

  for (const nodeId of oldIds) {
    expect(state.flow.nodes.has(nodeId)).toBe(false);
    expect(state.flow.wiring.has(nodeId)).toBe(false);
    expect(state.nodeState.has(nodeId)).toBe(false);
    expect(state.pool.store.has(nodeId)).toBe(false);
    expect(state.depth.has(nodeId)).toBe(false);
    expect(state.wiringOverrides.has(nodeId)).toBe(false);
    expect(state.ready.includes(nodeId)).toBe(false);
    expect(state.inFlight.has(nodeId)).toBe(false);
    expect(state.flow.incomingEdges.has(nodeId)).toBe(false);
    expect(state.flow.outgoingEdges.has(nodeId)).toBe(false);
  }
  expect([...state.edgeState.keys()].filter((edgeId) => oldEdgeIds.has(edgeId))).toEqual([]);
  expect(state.flow.edges.some((edge) => old.has(edge.from) || old.has(edge.to))).toBe(false);
  expect(state.flow.nodes.has(residueId)).toBe(true);
  expect(state.nodeState.get(residueId)).toBe(NodeState.TAKEN);
  expectIndexMatchesRebuild(state);
});

test("growth preserves unrelated in-flight siblings and FIFO order", () => {
  const { state, log } = initializedSpawner(parentFlow({ sibling: true }));
  expect(state.ready).toEqual(["side"]);
  commitPrepared(log, state, prepareNodeStarted(state, "side"));
  const sideStart = state.acceptedStarts.get("side");

  commitPrepared(log, state, prepareGrowth(log, state, "spawn", new Grow({}), fragmentBuilder));

  expect([...state.inFlight]).toEqual(["side"]);
  expect(state.acceptedStarts).toEqual(new Map([["side", sideStart!]]));
  expect(state.ready).toEqual(["spawn/child/start"]);
  expect(state.nodeState.get("side")).toBe(NodeState.UNKNOWN);
});

test("growth rejects a live node collision even when the id is pruned", () => {
  const { state, log } = initializedSpawner();
  const collidingId = "spawn/child/start";
  addLiveNode(state, new OverlayNode(collidingId));
  state.nodeState.set(collidingId, NodeState.TAKEN);

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}, [collidingId]), fragmentBuilder),
      RangeError,
      /^growth fragment node 'spawn\/child\/start' collides with live topology$/,
    ),
  );
});

/** Add a LOOP clone `cloneId` of `originId` to `state`, put it in flight and return its id. */
function startLoopClone(state: RunState, log: MemoryRunLog, cloneId: string, originId: string): string {
  const clone = new LoopNode(cloneId, {
    flowId: "child",
    child: canonicalChild(),
    childInputs: [],
    predicateKind: "times",
    times: 3,
    maxIters: 3,
    iteration: 1,
    originId,
  });
  clone.params = [];
  addLiveNode(state, clone);
  state.ready.unshift(cloneId);
  commitPrepared(log, state, prepareNodeStarted(state, cloneId));
  return cloneId;
}

test.each([
  ["missing", "growth prune node 'absent' does not exist"],
  ["top-start", "growth cannot prune top-level boundary '__start__'"],
  ["top-end", "growth cannot prune top-level boundary '__end__'"],
  ["initial-static", "growth cannot prune initial static node 'spawn'"],
  ["non-loop-self", "growth spawner 'clone' is not a self-prunable LOOP clone"],
  ["loop-missing-origin", "growth spawner 'loop~1' is not a self-prunable LOOP clone"],
  ["loop-pruned-origin", "growth spawner 'loop-origin~1' is not a self-prunable LOOP clone"],
  ["edge-dependency", "surviving node '__end__' depends on pruned edge source 'old/child/start'"],
  ["wiring-dependency", "surviving node '__end__' wiring depends on pruned node"],
  ["commit-dependency", "surviving node '__end__' commits to pruned node 'old/child/start'"],
  ["origin-dependency", "surviving node '__end__' has pruned origin 'old/child/start'"],
  ["skipped-output-dependency", "surviving node '__end__' has pruned skipped output target 'old/child/start'"],
])("growth rejects unsafe pruning atomically: %s", (name, message) => {
  const { state, log } = initializedSpawner();
  const oldId = installSettledOverlay(state)[0]!;
  let pruneIds: string[] = [oldId];
  let executing = "spawn";
  const dependent = state.flow.nodes.get(END_ID)!;
  switch (name) {
    case "missing":
      pruneIds = ["absent"];
      break;
    case "top-start":
      pruneIds = [START_ID];
      break;
    case "top-end":
      pruneIds = [END_ID];
      break;
    case "initial-static":
      pruneIds = ["spawn"];
      break;
    case "non-loop-self":
      addLiveNode(state, new Spawner("clone"));
      state.ready.unshift("clone");
      commitPrepared(log, state, prepareNodeStarted(state, "clone"));
      executing = "clone";
      pruneIds = ["clone"];
      break;
    case "loop-missing-origin":
      executing = startLoopClone(state, log, "loop~1", "missing-origin");
      pruneIds = [executing];
      break;
    case "loop-pruned-origin":
      addLiveNode(state, new OverlayNode("loop-origin"));
      state.nodeState.set("loop-origin", NodeState.TAKEN);
      executing = startLoopClone(state, log, "loop-origin~1", "loop-origin");
      pruneIds = ["loop-origin", executing];
      break;
    case "edge-dependency": {
      const edge = new Edge("old-to-end", oldId, END_ID, { ordering: true });
      state.flow.addSubgraph(new Map(), [edge], new Map());
      state.edgeState.set(edge.id, NodeState.UNKNOWN);
      break;
    }
    case "wiring-dependency":
      state.flow.wiring.set(END_ID, new Map([["value", new Binding(`\${${oldId}.output}`)]]));
      break;
    case "commit-dependency":
      dependent.commitAs = oldId;
      break;
    case "origin-dependency":
      dependent.originId = oldId;
      break;
    case "skipped-output-dependency":
      dependent.skippedOutputTarget = () => oldId;
      break;
    default:
      throw new Error(name);
  }

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, executing, new Grow({}, pruneIds), fragmentBuilder),
      RangeError,
      message,
    ),
  );
});

test("growth rejects a Binding-wrapped dependency on a pruned node", () => {
  // A wrapped reference can't hide a live dependency on removed topology.
  const { state, log } = initializedSpawner();
  const oldId = installSettledOverlay(state)[0]!;
  state.flow.wiring.set(
    END_ID,
    new Map([["value", new Binding(`\${${oldId}.output}`, { origin: new OriginRef("pruned-dependency") })]]),
  );

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}, [oldId]), fragmentBuilder),
      RangeError,
      "surviving node '__end__' wiring depends on pruned node",
    ),
  );
  expect(log.nextSequence).toBe(7);
});

test("growth rejects a template dependency on a pruned node", () => {
  // A reference nested in a value template keeps its producer from being pruned.
  const { state, log } = initializedSpawner();
  const oldId = installSettledOverlay(state)[0]!;
  state.flow.wiring.set(
    END_ID,
    new Map([["value", new Binding(new ValueTemplate({ seen: [`\${${oldId}.output}`] }))]]),
  );

  expectAtomic(state, log, () =>
    expectRejected(
      () => prepareExpanded(state, "spawn", new Grow({}, [oldId]), fragmentBuilder),
      RangeError,
      "surviving node '__end__' wiring depends on pruned node",
    ),
  );
  expect(log.nextSequence).toBe(7);
});

test("an unsettled reader keeps a settled prune id it reads only through a template", () => {
  const setup = () => {
    const { state, log } = initializedSpawner();
    const oldId = installSettledOverlay(state)[0]!;
    const reader = new OverlayNode("reader");
    state.flow.addSubgraph(
      new Map([[reader.id, reader]]),
      [],
      new Map([[reader.id, new Map([["value", new Binding(new ValueTemplate({ seen: `saw \${${oldId}.output}` }))]])]]),
    );
    state.nodeState.set(reader.id, NodeState.UNKNOWN);
    return { state, log, oldId, readerId: reader.id };
  };
  const { state, log, oldId, readerId } = setup();

  const prepared = prepareGrowth(log, state, "spawn", new Grow({}, [oldId, readerId]), fragmentBuilder);
  const observers = commitPrepared(log, state, prepared);

  expect(state.flow.nodes.has(oldId)).toBe(true);
  expect([...state.deferredPrune].sort()).toEqual([oldId, readerId].sort());
  expectIndexMatchesRebuild(state);
  assertReplaysEqual(state, prepared, observers, setup);
});

test.each([
  ["at this growth", false, /^node '__end__' has invalid wiring: /],
  ["before this growth", true, /^surviving node '__end__' has invalid wiring: /],
])("the prune scan rejects malformed surviving wiring as a raw invariant, index built %s", (_name, built, message) => {
  const { state } = initializedSpawner();
  const oldId = installSettledOverlay(state)[0]!;
  if (built) {
    // With the index already built, the malformed wiring is first read by the prune scan.
    state.flow.wiring.set(END_ID, new Map([["value", new Binding(`\${${oldId}.output}`)]]));
    growthIndex(state);
  }
  state.flow.wiring.set(END_ID, new Map([["value", new Binding(`\${${oldId}.output`)]]));

  const error = caught(() => prepareExpanded(state, "spawn", new Grow({}, [oldId]), fragmentBuilder)) as Error;

  expect(error.constructor).toBe(YianRangeError);
  expect(error.message).toMatch(message);
  expect(error.cause).toBeInstanceOf(ExpressionError);
});

test("a LOOP clone record uses the actual id, self-prunes and inherits depth zero", async () => {
  const child = canonicalChild();
  const loop = new LoopNode("spawn", {
    flowId: "child",
    child,
    childInputs: [],
    predicateKind: "times",
    times: 3,
    maxIters: 3,
    originId: "spawn",
  });
  loop.params = [];
  const { state, log } = initializedSpawner(parentFlow({ spawner: loop }));
  state.depth.set("spawn", 2);

  commitPrepared(log, state, prepareGrowth(log, state, "spawn", new Grow([{}, 0n]), buildGrow));
  const firstDynamic = [...state.flow.nodes.keys()].filter((nodeId) => !state.initialStaticIds.has(nodeId));
  expect(firstDynamic).toContain("spawn~1");
  expect(state.depth.get("spawn~1")).toBe(2);
  for (const nodeId of firstDynamic) {
    state.nodeState.set(nodeId, nodeId === "spawn~1" ? NodeState.UNKNOWN : NodeState.TAKEN);
  }
  // The body settled by fiat, so its START's expansion commit is no longer waiting.
  state.pendingCommits.clear();
  for (const edgeId of state.edgeState.keys()) {
    if (edgeId !== "start-spawn" && edgeId !== "spawn-end") state.edgeState.set(edgeId, NodeState.TAKEN);
  }
  state.ready.length = 0;
  state.ready.push("spawn~1");
  commitPrepared(log, state, prepareNodeStarted(state, "spawn~1"));

  const clone = state.flow.nodes.get("spawn~1")!;
  const grow = await clone.run({});
  expect(grow.constructor).toBe(Grow);
  const prepared = prepareGrowth(log, state, "spawn~1", grow as Grow, buildGrow);

  const record = prepared.record as NodeExpandedRecord;
  expect(record.spawnerId).toBe("spawn~1");
  expect(record.pruneIds).toContain("spawn~1");
  const observers = commitPrepared(log, state, prepared);

  expect((observers[0] as NodeExpanded).nodeId).toBe("spawn~1");
  expect(state.flow.nodes.has("spawn~1")).toBe(false);
  expect(state.acceptedStarts.has("spawn~1")).toBe(false);
  expect(state.flow.nodes.has("spawn")).toBe(true);
  expect(state.nodeState.get("spawn")).toBe(NodeState.EXPANDED);
  expect(state.flow.nodes.has("spawn~2")).toBe(true);
  expect(state.depth.get("spawn~2")).toBe(2);
  expect([...state.flow.nodes.keys()].some((nodeId) => nodeId.startsWith("spawn#0/"))).toBe(false);
});

test("a growth commit does no builder or graph work after append", () => {
  const { state, log: prior } = initializedSpawner();
  commitExpansion(prior, state, "spawn", new Grow({ nested: { items: ["stable"] } }), fragmentBuilder);
  const built = fragment("spawn");
  const builderSeeds: unknown[] = [];
  let builderCalls = 0;
  const builder: GrowBuilder = (_spawner, seed) => {
    builderCalls += 1;
    builderSeeds.push(seed);
    return built;
  };
  const rawSeed = { nested: { items: ["stable"] } };
  const prepared = prepareExpanded(state, "spawn", new Grow(rawSeed), builder);
  const expectedStart = (prepared._plan as { delta: { fragmentStartId: string } }).delta.fragmentStartId;

  // Every object the caller or the builder still owns may change after preparation.
  rawSeed.nested.items.push("raw-mutated");
  (builderSeeds[0] as { nested: { items: string[] } }).nested.items.push("builder-mutated");
  built.nodes.clear();
  built.edges.length = 0;
  built.wiring.clear();
  ((prepared.observers[0] as NodeExpanded).grow.seed as { nested: { items: string[] } }).nested.items.push(
    "observer-mutated",
  );

  let appended = false;
  class GuardedLog extends MemoryRunLog {
    override append(record: RunRecord): RecordEnvelope {
      const envelope = super.append(record);
      if (record instanceof NodeExpandedRecord) appended = true;
      return envelope;
    }
  }
  const log = new GuardedLog(RUN_ID);
  for (const envelope of prior.records()) log.append(envelope.record);
  const addSubgraph = spyOn(CompiledFlow.prototype, "addSubgraph").mockImplementation(() => {
    throw new Error("addSubgraph ran during commit");
  });
  const removeSubgraph = spyOn(CompiledFlow.prototype, "removeSubgraph").mockImplementation(() => {
    throw new Error("removeSubgraph ran during commit");
  });
  let observers: readonly RunEvent[];
  try {
    observers = commitPrepared(log, state, prepared);
  } finally {
    addSubgraph.mockRestore();
    removeSubgraph.mockRestore();
  }

  expect(appended).toBe(true);
  expect(builderCalls).toBe(1);
  expect(state.flow.nodes.has(expectedStart)).toBe(true);
  expect(state.ready.at(-1)).toBe(expectedStart);
  expect((observers[0] as NodeExpanded).grow.seed).toEqual({ nested: { items: ["stable"] } });
});

test("growth records the spawner as the owner of its fragment", () => {
  const { state, log } = initializedSpawner();
  const staticIds = new Set(state.flow.nodes.keys());

  commitPrepared(log, state, prepareGrowth(log, state, "spawn", new Grow({}), fragmentBuilder));

  const newIds = [...state.flow.nodes.keys()].filter((nodeId) => !staticIds.has(nodeId));
  expect(newIds.length).toBeGreaterThan(0);
  expect(state.growthOwner).toEqual(new Map(newIds.map((nodeId) => [nodeId, "spawn"])));
});

/** Return a state whose prunable `x/child/start` owns settled nodes and `side`. */
function grownUnderRequested(sideState: NodeState): {
  state: RunState;
  log: MemoryRunLog;
  grown: string[];
  side: string;
} {
  const { state, log } = initializedSpawner();
  const ids = installSettledOverlay(state, { prefix: "x" });
  const grown = installSettledOverlay(state, { prefix: "x/child/start" });
  const side = "x/child/start/side";
  addLiveNode(state, new OverlayNode(side));
  state.nodeState.set(side, sideState);
  state.growthOwner = new Map([
    ...ids.map((nodeId) => [nodeId, "spawn"] as const),
    ...[...grown, side].map((nodeId) => [nodeId, "x/child/start"] as const),
  ]);
  return { state, log, grown, side };
}

/** Prepare the growth of `spawn` that requests pruning `x/child/start`. */
function pruneRequested(state: RunState, log: MemoryRunLog): PreparedTransition {
  return prepareGrowth(log, state, "spawn", new Grow({}, ["x/child/start"]), fragmentBuilder);
}

/**
 * Replay `prepared.record` on a fresh identical `setup()` and expect the observers, the state and
 * the topology to equal the live ones, and the replay's growth index to match a rebuild.
 */
function assertReplaysEqual(
  state: RunState,
  prepared: PreparedTransition,
  observers: readonly RunEvent[],
  setup: () => { state: RunState; log: MemoryRunLog },
): void {
  const { state: replay, log: replayLog } = setup();
  const record = prepared.record as NodeExpandedRecord;
  const seed = (prepared._plan as { seed: JsonValue }).seed;
  // The replay rebuilds the seed from the spawner's forward.
  (replay.flow.nodes.get(record.spawnerId) as Spawner).grow = new Grow(seed, record.pruneIds);
  commitExpansion(replayLog, replay, record.spawnerId, new Grow(seed, record.pruneIds), fragmentBuilder);
  const replayObservers = commitPrepared(
    replayLog,
    replay,
    prepareRecord(replay, prepared.record, { growBuilder: fragmentBuilder }),
  );
  expect(replayObservers).toEqual(observers);
  expect(reducerSnapshot(replay)).toEqual(reducerSnapshot(state));
  expect(topologySnapshot(replay)).toEqual(topologySnapshot(state));
  expectIndexMatchesRebuild(replay);
}

/** Expect `ids` gone from every topology and reducer container. */
function assertRetired(state: RunState, ids: readonly string[]): void {
  for (const nodeId of ids) {
    expect(state.flow.nodes.has(nodeId)).toBe(false);
    expect(state.flow.wiring.has(nodeId)).toBe(false);
    expect(state.nodeState.has(nodeId)).toBe(false);
    expect(state.pool.store.has(nodeId)).toBe(false);
    expect(state.depth.has(nodeId)).toBe(false);
    expect(state.wiringOverrides.has(nodeId)).toBe(false);
    expect(state.growthOwner.has(nodeId)).toBe(false);
    expect(state.flow.incomingEdges.has(nodeId)).toBe(false);
    expect(state.flow.outgoingEdges.has(nodeId)).toBe(false);
  }
  expect(state.flow.edges.some((edge) => ids.includes(edge.from) || ids.includes(edge.to))).toBe(false);
  expect([...state.edgeState.keys()].sort()).toEqual(state.flow.edges.map((edge) => edge.id).sort());
}

test("pruning keeps a grown subtree with unsettled work", () => {
  const { state, log, grown, side } = grownUnderRequested(NodeState.UNKNOWN);

  const prepared = pruneRequested(state, log);
  const observers = commitPrepared(log, state, prepared);

  expect([...state.deferredPrune]).toEqual(["x/child/start"]);
  expect(state.growthOwner.get("x/child/start")).toBe("spawn");
  for (const nodeId of [...grown, side]) {
    expect(state.flow.nodes.has(nodeId)).toBe(true);
    expect(state.growthOwner.get(nodeId)).toBe("x/child/start");
  }
  expectIndexMatchesRebuild(state);
  assertReplaysEqual(state, prepared, observers, () => grownUnderRequested(NodeState.UNKNOWN));
});

test("pruning retires a grown subtree once settled", () => {
  const { state, log, grown, side } = grownUnderRequested(NodeState.TAKEN);

  const prepared = pruneRequested(state, log);
  const observers = commitPrepared(log, state, prepared);

  expect((prepared.record as NodeExpandedRecord).pruneIds).toEqual(["x/child/start"]);
  assertRetired(state, ["x/child/start", ...grown, side]);
  expect(new Set(state.growthOwner.values())).toEqual(new Set(["spawn"]));
  expectIndexMatchesRebuild(state);
  assertReplaysEqual(state, prepared, observers, () => grownUnderRequested(NodeState.TAKEN));
});

/** Make the settled overlay node `nodeId` waiting, ready, or running. */
function unsettle(
  state: RunState,
  log: MemoryRunLog,
  nodeId: string,
  how: "waiting" | "ready" | "in-flight",
): void {
  state.nodeState.set(nodeId, NodeState.UNKNOWN);
  if (how === "ready" || how === "in-flight") state.ready.unshift(nodeId);
  if (how === "in-flight") commitPrepared(log, state, prepareNodeStarted(state, nodeId));
}

test.each(["waiting", "ready", "in-flight"] as const)(
  "growth keeps an unsettled prune id live and deferred: %s",
  (how) => {
    const setup = () => {
      const { state, log } = initializedSpawner();
      const oldId = installSettledOverlay(state)[0]!;
      unsettle(state, log, oldId, how);
      return { state, log, oldId };
    };
    const { state, log, oldId } = setup();

    const prepared = prepareGrowth(log, state, "spawn", new Grow({}, [oldId]), fragmentBuilder);
    const observers = commitPrepared(log, state, prepared);

    expect((prepared.record as NodeExpandedRecord).pruneIds).toEqual([oldId]);
    expect(state.flow.nodes.has(oldId)).toBe(true);
    expect([...state.deferredPrune]).toEqual([oldId]);
    expect(state.inFlight.has(oldId)).toBe(how === "in-flight");
    expectIndexMatchesRebuild(state);
    assertReplaysEqual(state, prepared, observers, setup);
  },
);

/** Install settled overlay nodes `a -> b` whose `b` reads `a` by edge and wiring. */
function sideChain(state: RunState): [string, string] {
  const edge = new Edge("a->b", "a", "b", { ordering: true });
  state.flow.addSubgraph(
    new Map([
      ["a", new OverlayNode("a")],
      ["b", new OverlayNode("b")],
    ]),
    [edge],
    new Map([
      ["a", new Map()],
      ["b", new Map([["value", new Binding("${a.output}")]])],
    ]),
  );
  state.nodeState.set("a", NodeState.TAKEN);
  state.nodeState.set("b", NodeState.TAKEN);
  state.edgeState.set(edge.id, NodeState.TAKEN);
  return ["a", "b"];
}

/** Prepare the growth of `spawn` that requests pruning `a` and `b`. */
function pruneSideChain(state: RunState, log: MemoryRunLog): PreparedTransition {
  return prepareGrowth(log, state, "spawn", new Grow({}, ["a", "b"]), fragmentBuilder);
}

test("growth keeps a settled prune id that kept work reads", () => {
  const setup = () => {
    const { state, log } = initializedSpawner();
    const [, b] = sideChain(state);
    unsettle(state, log, b, "waiting");
    return { state, log };
  };
  const { state, log } = setup();

  const prepared = pruneSideChain(state, log);
  const observers = commitPrepared(log, state, prepared);

  expect(["a", "b"].filter((nodeId) => !state.flow.nodes.has(nodeId))).toEqual([]);
  expect([...state.deferredPrune].sort()).toEqual(["a", "b"]);
  expectIndexMatchesRebuild(state);
  assertReplaysEqual(state, prepared, observers, setup);
});

test("growth retires a settled prune id the kept work doesn't read", () => {
  const setup = () => {
    const { state, log } = initializedSpawner();
    const [a, b] = sideChain(state);
    unsettle(state, log, a, "waiting");
    state.nodeState.set(b, NodeState.SKIPPED);
    return { state, log };
  };
  const { state, log } = setup();

  const prepared = pruneSideChain(state, log);
  const observers = commitPrepared(log, state, prepared);

  assertRetired(state, ["b"]);
  expect([...state.deferredPrune]).toEqual(["a"]);
  expectIndexMatchesRebuild(state);
  assertReplaysEqual(state, prepared, observers, setup);
});

test("a growth that retires a counted node keeps its crashed-attempt count", () => {
  const setup = () => {
    const { state, log } = initializedSpawner();
    const oldIds = installSettledOverlay(state);
    state.crashedAttempts.set(oldIds[0]!, 2);
    return { state, log, oldIds };
  };
  const { state, log, oldIds } = setup();

  const prepared = prepareGrowth(log, state, "spawn", new Grow({}, oldIds), fragmentBuilder);
  const observers = commitPrepared(log, state, prepared);

  assertRetired(state, oldIds);
  expect(state.crashedAttempts.get(oldIds[0]!)).toBe(2);
  expectIndexMatchesRebuild(state);
  assertReplaysEqual(state, prepared, observers, setup);
});
