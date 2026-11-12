/**
 * Skip propagation through node-owned logical output redirects. Each rejected row names its one
 * class and full message, and each test that commits a route pins `state.ready` after it.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import { RunFailed, RunSucceeded } from "../../src/events.ts";
import { ELSE_HANDLE, type Node } from "../../src/nodes/base.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import {
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  PoolCommitRecord,
  RunStartedRecord,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  prepareNodeRouted,
  prepareNodeStarted,
} from "../../src/runtime/state.ts";
import {
  callRedirectFlow,
  CHILD_END,
  FuncNode,
  prepareOutput,
  reducerSnapshot,
  replayOutcomeLog,
  routeInFlight,
} from "../helpers/run_state_fixtures.ts";
import { thrownClass } from "../helpers/own_errors.ts";

/** Assert `fn` throws an error of exactly class `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(thrownClass(cls));
  expect((caught as Error).message).toBe(message);
}

/** A pass-through node. */
function func(nodeId: string): FuncNode {
  return new FuncNode(nodeId, (inputs) => inputs);
}

/** The run failure of a route from `router` that skips END's `result` group. */
const RESULT_UNREACHABLE = new RunFailed({
  failure: new Failure({
    code: "graph.terminal_unreachable",
    message: "terminal output 'result' skipped",
    nodeId: "router",
  }),
});

/** Return one MAP item's child END feeding a list collector redirected to the MAP spawner. */
function mapRedirectFlow(): CompiledFlow {
  const childEndId = "map#0/__end__";
  const collectorId = "map/__end__";
  const collector = EndNode.list(collectorId, { n: 1 });
  collector.commitAs = "map";
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["router", func("router")],
      [childEndId, EndNode.value(childEndId, { outputName: null })],
      [collectorId, collector],
      ["map", new MapNode("map", { flowId: "child" })],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-router", START_ID, "router"),
      new Edge("router-map-child-end", "router", childEndId, { sourceHandle: "item" }),
      new Edge("map-child-collector", childEndId, collectorId, { inputGroup: "e0" }),
      new Edge("map-end", "map", END_ID, { inputGroup: "result" }),
    ],
  );
}

/** Return one LOOP body END feeding a respawned driver whose output goes to the loop origin. */
function loopRedirectFlow(): CompiledFlow {
  const bodyEndId = "loop#0/__end__";
  const driverId = "loop~1";
  const loop = { flowId: "body", predicateKind: "times", times: 2, maxIters: 2, originId: "loop" } as const;
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["router", func("router")],
      [bodyEndId, EndNode.value(bodyEndId, { outputName: null })],
      [driverId, new LoopNode(driverId, { ...loop, iteration: 1 })],
      ["loop", new LoopNode("loop", loop)],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-router", START_ID, "router"),
      new Edge("router-loop-body-end", "router", bodyEndId, { sourceHandle: "body" }),
      new Edge("loop-body-driver", bodyEndId, driverId, { inputGroup: "carried" }),
      new Edge("loop-end", "loop", END_ID, { inputGroup: "result" }),
    ],
  );
}

/** Return two CALL output redirects separated by ordinary child wiring. */
function nestedCallRedirectFlow(): CompiledFlow {
  const innerEndId = "outer/inner/__end__";
  const outerEndId = "outer/__end__";
  const innerEnd = EndNode.value(innerEndId, { outputName: null });
  innerEnd.commitAs = "outer/inner";
  const outerEnd = EndNode.value(outerEndId, { outputName: null });
  outerEnd.commitAs = "outer";
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["router", func("router")],
      [innerEndId, innerEnd],
      ["outer/inner", new CallNode("outer/inner", { flowId: "inner" })],
      [outerEndId, outerEnd],
      ["outer", new CallNode("outer", { flowId: "outer" })],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-router", START_ID, "router"),
      new Edge("router-inner-end", "router", innerEndId, { sourceHandle: "inner" }),
      new Edge("inner-call-outer-end", "outer/inner", outerEndId, { inputGroup: "inner-result" }),
      new Edge("outer-call-end", "outer", END_ID, { inputGroup: "result" }),
    ],
  );
}

test("a required CALL child END redirect skips the parent's output path", () => {
  const { state, log } = routeInFlight(callRedirectFlow());

  commitPrepared(log, state, prepareNodeRouted(state, "router", ELSE_HANDLE));

  expect(state.nodeState.get(CHILD_END)).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("call")).toBe(NodeState.UNKNOWN);
  expect(state.edgeState.get("call-end")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get(END_ID)).toBe(NodeState.SKIPPED);
  expect(state.result).toStrictEqual(RESULT_UNREACHABLE);
  expect(state.ready).toEqual([]);
});

test("a redirected skip completes a coalesced fallback and the run", () => {
  const { state, log } = routeInFlight(callRedirectFlow({ fallback: true }));
  commitPrepared(log, state, prepareNodeStarted(state, "fallback"));
  commitPrepared(log, state, prepareOutput(log, state, "fallback", "fallback-value"));

  commitPrepared(log, state, prepareNodeRouted(state, "router", ELSE_HANDLE));

  expect(state.edgeState.get("call-end")).toBe(NodeState.SKIPPED);
  expect(state.edgeState.get("fallback-end")).toBe(NodeState.TAKEN);
  expect(state.nodeState.get(END_ID)).toBe(NodeState.UNKNOWN);
  expect(state.ready).toEqual([END_ID]);
  expect(state.result).toBeNull();

  commitPrepared(log, state, prepareNodeStarted(state, END_ID));
  commitPrepared(log, state, prepareOutput(log, state, END_ID, "fallback-value"));

  expect(state.result).toStrictEqual(new RunSucceeded({ output: "fallback-value" }));
  expect(state.ready).toEqual([]);
});

test.each<[unknown, new (...args: never[]) => Error, string]>([
  ["missing", RangeError, "node 'call/__end__' skipped output target 'missing' does not exist"],
  ["", RangeError, "node 'call/__end__' skipped output target must be non-empty"],
  [0, TypeError, "node 'call/__end__' skipped output target must be a string"],
])("a malformed skipped output target leaves the state unchanged: %p", (target, cls, message) => {
  const { state, log } = routeInFlight(callRedirectFlow());
  state.flow.nodes.get(CHILD_END)!.commitAs = target as string;
  const before = reducerSnapshot(state);
  const sequence = log.nextSequence;

  expectError(() => prepareNodeRouted(state, "router", ELSE_HANDLE), cls, message);

  expect(reducerSnapshot(state)).toEqual(before);
  expect(log.nextSequence).toBe(sequence);
});

test("several redirect owners are refused before the skip", () => {
  // One skipped alias can't settle a target another alias may still produce.
  const first = func("first");
  first.commitAs = "target";
  const second = func("second");
  second.commitAs = "target";
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["router", func("router")],
      ["first", first],
      ["second", second],
      ["target", func("target")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-router", START_ID, "router"),
      new Edge("router-first", "router", "first", { sourceHandle: "first" }),
      new Edge("router-second", "router", "second", { sourceHandle: "second" }),
      new Edge("target-end", "target", END_ID, { inputGroup: "result" }),
    ],
  );
  const { state, log } = routeInFlight(flow);
  const before = reducerSnapshot(state);
  const sequence = log.nextSequence;

  expectError(
    () => prepareNodeRouted(state, "router", "second"),
    RangeError,
    `logical output target 'target' has multiple skipped output owners: ["first","second"]`,
  );

  expect(reducerSnapshot(state)).toEqual(before);
  expect(log.nextSequence).toBe(sequence);
});

test("a live redirect target is refused before the skip", () => {
  // A logical anchor can't race a redirected producer for the same output.
  const owner = func("owner");
  owner.commitAs = "target";
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["router", func("router")],
      ["owner", owner],
      ["target", func("target")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-router", START_ID, "router"),
      new Edge("start-target", START_ID, "target"),
      new Edge("router-owner", "router", "owner", { sourceHandle: "owner" }),
      new Edge("target-end", "target", END_ID, { inputGroup: "result" }),
    ],
  );
  const { state, log } = routeInFlight(flow);
  expect(state.ready).toEqual(["target"]);
  const before = reducerSnapshot(state);
  const sequence = log.nextSequence;

  expectError(
    () => prepareNodeRouted(state, "router", ELSE_HANDLE),
    RangeError,
    "logical output target 'target' is also executable",
  );

  expect(reducerSnapshot(state)).toEqual(before);
  expect(log.nextSequence).toBe(sequence);
});

test("settled redirect residue doesn't own a future output", () => {
  // Pruned nested-growth residue is inert even when its old anchor is gone.
  const flow = callRedirectFlow();
  const residue = func("settled-residue");
  residue.commitAs = "pruned-anchor";
  flow.nodes.set(residue.id, residue);
  const { state, log } = routeInFlight(flow);
  state.nodeState.set(residue.id, NodeState.TAKEN);

  commitPrepared(log, state, prepareNodeRouted(state, "router", ELSE_HANDLE));

  expect(state.edgeState.get("call-end")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get(END_ID)).toBe(NodeState.SKIPPED);
  expect(state.result).toStrictEqual(RESULT_UNREACHABLE);
  expect(state.ready).toEqual([]);
});

test("a MAP collector redirects its skipped output to the spawner", () => {
  const { state, log } = routeInFlight(mapRedirectFlow());

  commitPrepared(log, state, prepareNodeRouted(state, "router", ELSE_HANDLE));

  expect(state.nodeState.get("map#0/__end__")).toBe(NodeState.SKIPPED);
  expect(state.edgeState.get("map-child-collector")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("map/__end__")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("map")).toBe(NodeState.UNKNOWN);
  expect(state.edgeState.get("map-end")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get(END_ID)).toBe(NodeState.SKIPPED);
  expect(state.ready).toEqual([]);
});

test("a LOOP driver redirects its skipped output to the origin", () => {
  const { state, log } = routeInFlight(loopRedirectFlow());

  commitPrepared(log, state, prepareNodeRouted(state, "router", ELSE_HANDLE));

  expect(state.nodeState.get("loop#0/__end__")).toBe(NodeState.SKIPPED);
  expect(state.edgeState.get("loop-body-driver")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("loop~1")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("loop")).toBe(NodeState.UNKNOWN);
  expect(state.edgeState.get("loop-end")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get(END_ID)).toBe(NodeState.SKIPPED);
  expect(state.ready).toEqual([]);
});

test("nested redirects propagate to the top-level END", () => {
  const { state, log } = routeInFlight(nestedCallRedirectFlow());

  commitPrepared(log, state, prepareNodeRouted(state, "router", ELSE_HANDLE));

  expect(state.nodeState.get("outer/inner/__end__")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("outer/inner")).toBe(NodeState.UNKNOWN);
  expect(state.edgeState.get("inner-call-outer-end")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("outer/__end__")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("outer")).toBe(NodeState.UNKNOWN);
  expect(state.edgeState.get("outer-call-end")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get(END_ID)).toBe(NodeState.SKIPPED);
  expect(state.ready).toEqual([]);
});

test("a redirected skip replays without a terminal record", () => {
  const live = routeInFlight(nestedCallRedirectFlow());
  commitPrepared(live.log, live.state, prepareNodeRouted(live.state, "router", ELSE_HANDLE));

  expect(live.log.records().map((envelope) => envelope.record.constructor)).toEqual([
    RunStartedRecord,
    PoolCommitRecord,
    PoolCommitRecord,
    NodeStartedRecord,
    PoolCommitRecord,
    NodeSucceededRecord,
    NodeStartedRecord,
    NodeRoutedRecord,
  ]);

  const replay = replayOutcomeLog(nestedCallRedirectFlow(), live.log);

  expect(reducerSnapshot(replay.state)).toEqual(reducerSnapshot(live.state));
  expect(replay.log.records()).toEqual(live.log.records());
  expect(replay.state.ready).toEqual([]);
});
