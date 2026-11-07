/**
 * Transactional node-outcome tests for the run-state reducer: routes, successes (a `pool_commit`
 * and the `node_succeeded` naming it), failures, terminal derivation and replay. Each rejected row
 * pins its class and full message; the failing log is the shared `FailingLog`.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, Flow, START_ID } from "../../src/compile/model.ts";
import { Failure, OriginRef } from "../../src/diagnostics.ts";
import {
  NodeFailed,
  NodeRouted,
  NodeSkipped,
  NodeSucceeded,
  type RunEvent,
  RunFailed,
  RunSucceeded,
} from "../../src/events.ts";
import { ELSE_HANDLE, Grow, Node, NodeKind } from "../../src/nodes/base.ts";
import { type OutputContract, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import {
  MemoryRunLog,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  PoolCommitRecord,
  type RecordEnvelope,
  type RunRecord,
  RunStartedRecord,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  type GrowBuilder,
  NodeDataError,
  type PreparedTransition,
  prepareNodeFailed,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareNodeSucceeded,
  preparePoolCommit,
  prepareRecord,
  RunState,
} from "../../src/runtime/state.ts";
import { Type, TypeCheckError, type Value, ValueKind } from "../../src/typesys/values.ts";
import {
  commitStartSuccess,
  expectUnchanged,
  FailingLog,
  fanoutFlow,
  FuncNode,
  initializedAndStarted,
  nodeFailure,
  prepareOutput,
  reducerSnapshot,
  replayOutcomeLog,
  roots,
  routeFlow,
  routeInFlight,
  RUN_ID,
  snapshot,
  terminalSkipRouteFlow,
  prepareGrowth,
} from "../helpers/run_state_fixtures.ts";

/** Assert `fn` throws an error of exactly class `cls` with exactly `message`; return it. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): Error {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
  return caught as Error;
}

/** Assert `fn` throws a `NodeDataError` whose `original` is exactly of class `cls` with `message`. */
function expectNodeData(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): NodeDataError {
  const caught = expectError(fn, NodeDataError, message) as NodeDataError;
  expect((caught.original as Error).constructor).toBe(cls);
  expect((caught.original as Error).message).toBe(message);
  expect(caught.cause).toBe(caught.original);
  return caught;
}

/** Return `prepared`'s plan narrowed to a success plan; throw unless it is one. */
function successPlan(prepared: PreparedTransition) {
  const plan = prepared._plan;
  if (plan.kind !== "success") throw new Error(`expected a success plan, found ${plan.kind}`);
  return plan;
}

/** Return the node `nodeId` of `flow`. */
function node(flow: CompiledFlow, nodeId: string): Node {
  return flow.nodes.get(nodeId)!;
}

/** A pass-through node. */
function func(nodeId: string, options: { outputContract?: OutputContract | null } = {}): FuncNode {
  return new FuncNode(nodeId, (inputs) => inputs, options);
}

/** Return a graph with fan-out, a two-edge join and an existing ready node. */
function joinedFanoutFlow(): CompiledFlow {
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["first", func("first")],
      ["second", func("second")],
      ["existing", func("existing")],
      ["after-a", func("after-a")],
      ["after-b", func("after-b")],
      ["join", func("join")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-first", START_ID, "first"),
      new Edge("start-second", START_ID, "second"),
      new Edge("start-existing", START_ID, "existing"),
      new Edge("first-join", "first", "join", { inputGroup: "both" }),
      new Edge("first-after-a", "first", "after-a"),
      new Edge("first-after-b", "first", "after-b"),
      new Edge("second-join", "second", "join", { inputGroup: "both" }),
    ],
  );
}

/** Return a graph whose executing filler has a baked and an explicit target, both `list[str]`. */
function redirectFlow(edges?: Edge[]): CompiledFlow {
  const listType = Type.scalar(ValueKind.LIST_STRING);
  const filler = func("filler");
  filler.commitAs = "baked";
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["filler", filler],
      ["baked", func("baked", { outputContract: new ValueContract(listType) })],
      ["explicit", func("explicit", { outputContract: new ValueContract(listType) })],
      ["after-baked", func("after-baked")],
      ["after-explicit", func("after-explicit")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    edges ?? [
      new Edge("start-filler", START_ID, "filler"),
      new Edge("baked-after", "baked", "after-baked"),
      new Edge("explicit-after", "explicit", "after-explicit"),
    ],
  );
}

/** Return a replay-valid state with the redirect filler in flight. */
function redirectReady(flow: CompiledFlow = redirectFlow()): { state: RunState; log: MemoryRunLog } {
  const { state, log } = initializedAndStarted(flow);
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "filler"));
  return { state, log };
}

/** Return the redirect graph with two target edges into the same successor. */
function duplicateSuccessorFlow(): CompiledFlow {
  return redirectFlow([
    new Edge("start-filler", START_ID, "filler"),
    new Edge("explicit-after", "explicit", "after-explicit"),
    new Edge("baked-after-1", "baked", "after-baked"),
    new Edge("baked-after-2", "baked", "after-baked"),
  ]);
}

/** Return target edges ordered A1, B, A2, so B becomes ready before A. */
function interleavedSuccessorFlow(): CompiledFlow {
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["target", func("target")],
      ["after-a", func("after-a")],
      ["after-b", func("after-b")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-target", START_ID, "target"),
      new Edge("target-a-1", "target", "after-a", { inputGroup: "a" }),
      new Edge("target-b", "target", "after-b"),
      new Edge("target-a-2", "target", "after-a", { inputGroup: "a" }),
    ],
  );
}

/** Return two wide joins fed by interleaved edges from one target. */
function parallelSuccessorFlow(width: number): CompiledFlow {
  const edges = [new Edge("start-target", START_ID, "target")];
  for (let index = 0; index < width; index++) {
    edges.push(
      new Edge(`target-a-${index}`, "target", "after-a", { inputGroup: "a" }),
      new Edge(`target-b-${index}`, "target", "after-b", { inputGroup: "b" }),
    );
  }
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["target", func("target")],
      ["after-a", func("after-a")],
      ["after-b", func("after-b")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    edges,
  );
}

/**
 * Run `START -> leaf-i -> END` for `width` leaves through each leaf's success and return how many
 * incoming edges the reducer read through `flow.incoming`, from START's success on.
 */
function wideJoinVisits(width: number): number {
  const nodes = new Map<string, Node>([[START_ID, new StartNode(START_ID)]]);
  const edges: Edge[] = [];
  for (let index = 0; index < width; index++) {
    nodes.set(`leaf-${index}`, func(`leaf-${index}`));
    edges.push(
      new Edge(`start-leaf-${index}`, START_ID, `leaf-${index}`),
      new Edge(`leaf-${index}-end`, `leaf-${index}`, END_ID),
    );
  }
  nodes.set(END_ID, EndNode.value(END_ID, { outputName: null }));
  const { state, log } = initializedAndStarted(new CompiledFlow(nodes, edges));
  const incoming = state.flow.incoming.bind(state.flow);
  let visits = 0;
  state.flow.incoming = (nodeId) => {
    const found = incoming(nodeId);
    visits += found.length;
    return found;
  };
  commitStartSuccess(state, log);
  for (let index = 0; index < width; index++) {
    commitPrepared(log, state, prepareNodeStarted(state, `leaf-${index}`));
    commitPrepared(log, state, prepareOutput(log, state, `leaf-${index}`, { done: true }));
  }
  expect(state.ready).toEqual([END_ID]);
  return visits;
}

/** Return `START -> work -> END` where `work` declares a partial-record output, `work` in flight. */
function partialRecordWorking(): { state: RunState; log: MemoryRunLog } {
  const work = func("work", {
    outputContract: new PartialRecordContract([
      ["items", new ValueContract(Type.scalar(ValueKind.LIST_INTEGER))],
      ["score", new ValueContract(Type.scalar(ValueKind.NUMBER))],
      ["note", null],
    ]),
  });
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["work", work],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-work", START_ID, "work"), new Edge("work-end", "work", END_ID)],
  );
  const { state, log } = initializedAndStarted(flow);
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "work"));
  return { state, log };
}

/** Return a graph whose END, sibling and queued leaf run concurrently. */
function endAndSiblingFlow(): CompiledFlow {
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["sibling", func("sibling")],
      ["queued", func("queued")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-end", START_ID, END_ID),
      new Edge("start-sibling", START_ID, "sibling"),
      new Edge("start-queued", START_ID, "queued"),
    ],
  );
}

/** Return a live run whose END value exists while its last sibling runs. */
function endCommittedWithSiblingInFlight(): { state: RunState; log: MemoryRunLog } {
  const { state, log } = initializedAndStarted(endAndSiblingFlow());
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, END_ID));
  commitPrepared(log, state, prepareNodeStarted(state, "sibling"));
  commitPrepared(log, state, prepareOutput(log, state, END_ID, { terminal: ["stable"] }));
  return { state, log };
}

/** Return `START -> END`. */
function startEndFlow(): CompiledFlow {
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-end", START_ID, END_ID)],
  );
}

/** Return a live run with END the sole in-flight node and no ready work. */
function endOnlyInFlight(): { state: RunState; log: MemoryRunLog } {
  const { state, log } = initializedAndStarted(startEndFlow());
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, END_ID));
  return { state, log };
}

test("a success commits the canonical value before its FIFO successors", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());

  const prepared = prepareOutput(log, state, START_ID, { items: ["alpha", "beta"] });

  expect(log.records().at(-1)!.record).toStrictEqual(
    new PoolCommitRecord(START_ID, START_ID, { kind: "object", value: { items: ["alpha", "beta"] } }),
  );
  expect(prepared.record).toStrictEqual(new NodeSucceededRecord(START_ID, log.nextSequence - 1));
  expect(prepared.observers).toStrictEqual([
    new NodeSucceeded({ nodeId: START_ID, output: { items: ["alpha", "beta"] }, commitAs: null }),
  ]);

  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual(prepared.observers);
  expect(state.pool.get(START_ID)).toEqual({ items: ["alpha", "beta"] });
  expect(state.nodeState.get(START_ID)).toBe(NodeState.TAKEN);
  expect(state.edgeState).toEqual(
    new Map([
      ["start-first", NodeState.TAKEN],
      ["start-second", NodeState.TAKEN],
    ]),
  );
  expect(state.ready).toEqual(["first", "second"]);
  expect([...state.inFlight]).toEqual([]);
  expect(state.nextSequence).toBe(6);
  expect(log.records().map((envelope) => envelope.record.constructor)).toEqual([
    RunStartedRecord,
    PoolCommitRecord,
    PoolCommitRecord,
    NodeStartedRecord,
    PoolCommitRecord,
    NodeSucceededRecord,
  ]);
});

test("a success replay revalidates the partial-record contract", () => {
  const flow = fanoutFlow();
  node(flow, START_ID).outputContract = new PartialRecordContract([
    ["items", new ValueContract(Type.scalar(ValueKind.LIST_INTEGER))],
    ["note", null],
  ]);
  const { state } = initializedAndStarted(flow);
  const record = new PoolCommitRecord(START_ID, START_ID, {
    kind: "object",
    value: { items: ["bad"], note: "opaque" },
  });

  expectError(() => prepareRecord(state, record), TypeCheckError, 'list element "bad" is not int (declared list(int))');
});

test("a success keeps its sibling in flight and appends the new readiness FIFO", () => {
  const { state, log } = initializedAndStarted(joinedFanoutFlow());
  commitStartSuccess(state, log);
  expect(state.ready).toEqual(["first", "second", "existing"]);
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  commitPrepared(log, state, prepareNodeStarted(state, "second"));

  commitPrepared(log, state, prepareOutput(log, state, "first", { done: "first" }));

  expect([...state.inFlight]).toEqual(["second"]);
  expect(state.ready).toEqual(["existing", "after-a", "after-b"]);
  expect(state.edgeState.get("first-join")).toBe(NodeState.TAKEN);
  expect(state.edgeState.get("second-join")).toBe(NodeState.UNKNOWN);

  commitPrepared(log, state, prepareOutput(log, state, "second", { done: "second" }));

  expect([...state.inFlight]).toEqual([]);
  expect(state.ready).toEqual(["existing", "after-a", "after-b", "join"]);
});

test.each([
  ["explicit", "explicit", "after-explicit"],
  [null, "baked", "after-baked"],
] as const)(
  "a success resolves the redirect precedence and keeps the target's disposition: %s",
  (suppliedTarget, expectedTarget, expectedSuccessor) => {
    const { state, log } = redirectReady();
    const otherTarget = expectedTarget === "explicit" ? "baked" : "explicit";

    const prepared = prepareOutput(log, state, "filler", ["alpha", "beta"], { commitAs: suppliedTarget });
    const commit = log.records().at(-1)!.record as PoolCommitRecord;
    const observers = commitPrepared(log, state, prepared);

    expect(commit.key).toBe(expectedTarget);
    expect(commit.value).toEqual({ kind: "list[str]", value: ["alpha", "beta"] });
    expect(observers).toStrictEqual([
      new NodeSucceeded({ nodeId: "filler", output: ["alpha", "beta"], commitAs: expectedTarget }),
    ]);
    expect(state.pool.get(expectedTarget)).toEqual(["alpha", "beta"]);
    expect(state.pool.getValue(otherTarget)).toBeNull();
    expect(state.nodeState.get("filler")).toBe(NodeState.TAKEN);
    expect(state.nodeState.get(expectedTarget)).toBe(NodeState.UNKNOWN);
    expect(state.edgeState.get(`${expectedTarget}-after`)).toBe(NodeState.TAKEN);
    expect(state.edgeState.get(`${otherTarget}-after`)).toBe(NodeState.UNKNOWN);
    expect(state.ready).toEqual([expectedSuccessor]);
  },
);

test("a success checks an ordinary target against its output contract's type", () => {
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["filler", func("filler", { outputContract: new ValueContract(Type.scalar(ValueKind.NUMBER)) })],
      ["after", func("after")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-filler", START_ID, "filler"), new Edge("filler-after", "filler", "after")],
  );
  const { state, log } = initializedAndStarted(flow);
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "filler"));

  // A `float` slot keeps an int an int, so a mismatch shows the contract applies.
  expectNodeData(
    () => prepareOutput(log, state, "filler", "three"),
    TypeCheckError,
    '"three" does not match declared type float',
  );
  commitPrepared(log, state, prepareOutput(log, state, "filler", 3n));
  const commit = log.records().at(-2)!.record as PoolCommitRecord;

  expect(commit.key).toBe("filler");
  expect(commit.value).toEqual({ kind: "int", value: 3n });
  expect(state.pool.get("filler")).toBe(3n);
  expect(state.ready).toEqual(["after"]);
});

/** Return `START -> filler -> END` with `filler` in flight under `outputContract`. */
function fillerWorking(outputContract: OutputContract | null): { state: RunState; log: MemoryRunLog } {
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["filler", func("filler", { outputContract })],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-filler", START_ID, "filler"), new Edge("filler-end", "filler", END_ID)],
  );
  const { state, log } = initializedAndStarted(flow);
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "filler"));
  return { state, log };
}

const N_RECORD = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([["n", Type.scalar(ValueKind.INTEGER)]]),
  required: new Set(["n"]),
});

test.each<[string, OutputContract, unknown, string, (string | number)[]]>([
  [
    "a list element's field",
    new ValueContract(new Type({ kind: ValueKind.LIST_OBJECT, element: N_RECORD })),
    [{ n: 1n }, { n: "x" }],
    'output at [1].n: "x" does not match declared type int',
    [1, "n"],
  ],
  [
    "a root record's missing field",
    new ValueContract(N_RECORD),
    {},
    'output at n: record missing required fields: ["n"]',
    ["n"],
  ],
])("a contract failure inside the output names where: %s", (_name, contract, output, message, path) => {
  const { state, log } = fillerWorking(contract);

  const caught = expectNodeData(() => prepareOutput(log, state, "filler", output), TypeCheckError, message);

  expect((caught.original as TypeCheckError).path).toEqual(path);
});

test("an output nested too deep names its place once", () => {
  const { state, log } = fillerWorking(null);
  let output: unknown = 1n;
  for (let level = 0; level < 102; level++) output = [output];

  let caught: unknown;
  try {
    prepareOutput(log, state, "filler", output);
  } catch (error) {
    caught = error;
  }

  expect(caught).toBeInstanceOf(NodeDataError);
  expect(((caught as NodeDataError).original as Error).message).toMatch(/^output at (\[0\])+: value nests deeper than 100 levels, /);
});

const MALFORMED_CONTRACT =
  "output contract must be an exact ValueContract, PartialRecordContract, or null, got Type";

test("a success refuses a malformed target contract as a host invariant", () => {
  const { state, log } = redirectReady();
  node(state.flow, "baked").outputContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;
  const before = snapshot(state);
  const sequence = log.nextSequence;

  expectError(() => prepareOutput(log, state, "filler", ["valid"]), TypeError, MALFORMED_CONTRACT);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(sequence);
});

test("a success replay revalidates the target's output contract type", () => {
  const live = redirectReady();
  commitPrepared(live.log, live.state, prepareOutput(live.log, live.state, "filler", ["valid"]));

  const replay = replayOutcomeLog(redirectFlow(), live.log);
  expect(reducerSnapshot(replay.state)).toEqual(reducerSnapshot(live.state));
  expect(replay.log.records()).toEqual(live.log.records());
  expect(replay.state.ready).toEqual(["after-baked"]);

  const incompatible = redirectFlow();
  node(incompatible, "baked").outputContract = new ValueContract(Type.scalar(ValueKind.INTEGER));
  expectError(
    () => replayOutcomeLog(incompatible, live.log),
    TypeCheckError,
    '["valid"] does not match declared type int',
  );

  const malformed = redirectFlow();
  node(malformed, "baked").outputContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;
  expectError(() => replayOutcomeLog(malformed, live.log), TypeError, MALFORMED_CONTRACT);
});

test.each([
  ["a type mismatch", "not-a-list", null, TypeCheckError, '"not-a-list" is not a list for declared list(str)'],
  ["a missing target", ["valid"], "missing", RangeError, "success commit target 'missing' does not exist in the flow"],
] as const)(
  "a success refuses %s as node data before append",
  (_name, output, commitAs, cls, message) => {
    const { state, log } = redirectReady();
    const before = snapshot(state);
    const sequence = log.nextSequence;

    expectNodeData(() => prepareOutput(log, state, "filler", output, { commitAs }), cls, message);

    expectUnchanged(state, before);
    expect(log.nextSequence).toBe(sequence);
  },
);

test("a success refuses a previously settled target edge before append", () => {
  const { state, log } = redirectReady();
  state.edgeState.set("baked-after", NodeState.TAKEN);
  commitPrepared(log, state, preparePoolCommit(state, "filler", ["valid"]));
  const before = snapshot(state);
  const sequence = log.nextSequence;

  expectError(
    () => prepareNodeSucceeded(state, "filler"),
    RangeError,
    "success target outgoing edge 'baked-after' must be UNKNOWN",
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(sequence);
});

test("a success treats an invalid baked target as a host invariant error", () => {
  const { state, log } = redirectReady();
  node(state.flow, "filler").commitAs = "missing";
  const before = snapshot(state);

  expectError(
    () => prepareOutput(log, state, "filler", ["valid"]),
    RangeError,
    "success commit target 'missing' does not exist in the flow",
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(7);
});

test.each([
  ["holds from, though its node isn't derived", false, "pool_commit of 'filler' holds from, but its node isn't derived"],
  ["holds no from, though its node is derived", true, "pool_commit of 'filler' must hold from: its node is derived"],
] as const)("a live commit that %s is refused before append, as replay refuses it", (_label, derived, message) => {
  const { state, log } = redirectReady();
  Object.defineProperty(node(state.flow, "filler"), "derived", { value: derived });
  const before = snapshot(state);
  const sequence = log.nextSequence;
  const from = derived ? null : new Map(state.producers);

  expectError(() => preparePoolCommit(state, "filler", ["valid"], { from }), RangeError, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(sequence);
});

test("a success refuses a settled newly ready successor before append", () => {
  const { state, log } = redirectReady();
  state.nodeState.set("after-baked", NodeState.TAKEN);
  commitPrepared(log, state, preparePoolCommit(state, "filler", ["valid"]));
  const before = snapshot(state);
  const sequence = log.nextSequence;

  expectError(
    () => prepareNodeSucceeded(state, "filler"),
    RangeError,
    "newly-ready successor 'after-baked' must have UNKNOWN disposition",
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(sequence);
});

test("several edges to one successor enqueue it once", () => {
  const { state, log } = redirectReady(duplicateSuccessorFlow());

  commitPrepared(log, state, prepareOutput(log, state, "filler", ["valid"]));

  expect(state.edgeState.get("baked-after-1")).toBe(NodeState.TAKEN);
  expect(state.edgeState.get("baked-after-2")).toBe(NodeState.TAKEN);
  expect(state.ready).toEqual(["after-baked"]);
});

test("a success keeps the sequential readiness of interleaved successors", () => {
  const { state, log } = initializedAndStarted(interleavedSuccessorFlow());
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "target"));

  commitPrepared(log, state, prepareOutput(log, state, "target", { done: true }));

  expect(state.ready).toEqual(["after-b", "after-a"]);
});

test("a success scans each distinct affected join once", () => {
  const { state, log } = initializedAndStarted(parallelSuccessorFlow(64));
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "target"));
  const incoming = state.flow.incoming.bind(state.flow);
  const scanned: string[] = [];
  state.flow.incoming = (nodeId) => {
    scanned.push(nodeId);
    return incoming(nodeId);
  };

  const prepared = prepareOutput(log, state, "target", { done: true });

  expect(scanned).toEqual(["after-a", "after-b"]);
  expect(successPlan(prepared).delta.newlyReadyIds).toEqual(["after-a", "after-b"]);
});

test("the successes into a wide join visit its incoming edges linearly", () => {
  const visits = wideJoinVisits(2_000);
  const doubled = wideJoinVisits(4_000);

  // A rescan of the whole join on every success would quadruple the visits.
  expect(doubled).toBeLessThan(2.5 * visits);
});

test.each([
  ["missing", "NodeSucceeded node 'missing' does not exist in the flow"],
  ["first", "NodeSucceeded node 'first' is not in-flight"],
] as const)("a success requires an existing unknown in-flight node: %s", (nodeId, message) => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const before = snapshot(state);

  expectError(() => prepareNodeSucceeded(state, nodeId), RangeError, message);
  expectError(() => preparePoolCommit(state, nodeId, { value: 1n }), RangeError, `pool_commit node '${nodeId}' is not in flight`);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(4);
});

test("a success append failure leaves every state coordinate unchanged", () => {
  const { state, log: prior } = initializedAndStarted(fanoutFlow());
  const prepared = prepareOutput(prior, state, START_ID, { value: [1n, 2n] });
  const before = snapshot(state);
  const log = new FailingLog(RUN_ID, prior.records());

  expectError(() => commitPrepared(log, state, prepared), Error, "disk full");

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(prior.nextSequence);
});

test("a success delta keeps every reducer container's identity", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const prepared = prepareOutput(log, state, START_ID, { value: "done" });
  const before = roots(state);

  commitPrepared(log, state, prepared);

  roots(state).forEach((root, index) => expect(root).toBe(before[index]));
  expect(state.ready).toEqual(["first", "second"]);
});

test("a success plan holds only an immutable sparse delta", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const delta = successPlan(prepareOutput(log, state, START_ID, { value: "done" })).delta;

  expect(typeof delta.valueJson).toBe("string");
  expect(delta.outgoingEdgeIds).toEqual(["start-first", "start-second"]);
  expect(delta.newlyReadyIds).toEqual(["first", "second"]);
  expect(() => {
    (delta as { newlyReadyIds: unknown }).newlyReadyIds = [];
  }).toThrow(TypeError);
  expect(Object.isFrozen(delta.newlyReadyIds)).toBe(true);
  expect(Object.isFrozen(delta.outgoingEdgeIds)).toBe(true);
});

test("a success commit uses only the prevalidated delta", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const prepared = prepareOutput(log, state, START_ID, { value: "done" });
  const unexpected = (): never => {
    throw new Error("commit repeated success preparation work");
  };
  state.flow.incoming = unexpected;
  state.flow.outgoing = unexpected;

  commitPrepared(log, state, prepared);

  expect(state.pool.get(START_ID)).toEqual({ value: "done" });
  expect(state.ready).toEqual(["first", "second"]);
});

test.each([
  ["edge-state", "success target outgoing edge 'baked-after' has no disposition"],
  ["successor", "success target outgoing edge 'baked-after' has unknown successor 'after-baked'"],
  ["incoming", "successor 'after-baked' has no incoming adjacency"],
] as const)("a success refuses a broken touched topology: %s", (broken, message) => {
  const { state, log } = redirectReady();
  if (broken === "edge-state") state.edgeState.delete("baked-after");
  else if (broken === "successor") state.nodeState.delete("after-baked");
  else state.flow.incomingEdges.set("after-baked", []);
  commitPrepared(log, state, preparePoolCommit(state, "filler", ["valid"]));

  expectError(() => prepareNodeSucceeded(state, "filler"), RangeError, message);

  expect(log.nextSequence).toBe(8);
});

test("a success's canonical copies isolate the input, the record, the pool and the observers", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const raw = { nested: [["original"]] };
  const expected = { nested: [["original"]] };

  const prepared = prepareOutput(log, state, START_ID, raw);
  raw.nested[0]!.push("raw mutation");
  ((prepared.observers[0] as NodeSucceeded).output as typeof raw).nested[0]!.push("prepared mutation");

  const observers = commitPrepared(log, state, prepared);
  const commitRecord = log.records().at(-2)!.record;
  expect(commitRecord).toBeInstanceOf(PoolCommitRecord);
  const recorded = (commitRecord as PoolCommitRecord).value!;
  expect(recorded.value).toEqual(expected);
  expect(state.pool.get(START_ID)).toEqual(expected);
  expect(observers).toStrictEqual([new NodeSucceeded({ nodeId: START_ID, output: expected, commitAs: null })]);

  ((observers[0] as NodeSucceeded).output as typeof raw).nested[0]!.push("returned mutation");
  expect(state.pool.get(START_ID)).toEqual(expected);
  expect(recorded.value).toEqual(expected);
  (state.pool.get(START_ID) as typeof raw).nested[0]!.push("pool mutation");
  expect(recorded.value).toEqual(expected);
  expect(state.ready).toEqual(["first", "second"]);
});

test("competing success plans refuse the second sequence before append", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const first = prepareOutput(log, state, START_ID, { value: "done" });
  const second = prepareNodeSucceeded(state, START_ID);

  commitPrepared(log, state, first);
  const sequence = log.nextSequence;
  expectError(() => commitPrepared(log, state, second), RangeError, "run-state sequence mismatch: prepared 5, state 6");

  expect(log.nextSequence).toBe(sequence);
  expect(state.ready).toEqual(["first", "second"]);
});

test("a success record replay reconstructs an identical state", () => {
  const live = initializedAndStarted(fanoutFlow());
  commitPrepared(live.log, live.state, prepareOutput(live.log, live.state, START_ID, { items: ["alpha", "beta"] }));

  const replay = replayOutcomeLog(fanoutFlow(), live.log);

  expect(reducerSnapshot(replay.state)).toEqual(reducerSnapshot(live.state));
  expect(replay.log.records()).toEqual(live.log.records());
  expect(replay.state.ready).toEqual(["first", "second"]);
});

test.each([
  ["a typed field", { items: [1n], score: Number.NaN, note: "x" }, ["score"]],
  ["an opaque field", { items: [1n], score: 1.5, note: { deep: Number.NaN } }, ["note", "deep"]],
] as const)("NaN in %s of a partial-record output is node data", (_name, output, path) => {
  const { state, log } = partialRecordWorking();
  const before = snapshot(state);

  const caught = expectNodeData(
    () => prepareOutput(log, state, "work", output),
    TypeCheckError,
    `output at ${path.join(".")}: NaN is not a yian value`,
  );

  expect((caught.original as TypeCheckError).path).toEqual(path);
  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(7);
});

test.each<[string, Value]>([
  ["Infinity", { items: [1n], score: Number.POSITIVE_INFINITY, note: { low: Number.NEGATIVE_INFINITY } }],
  ["a long int", { items: [1n, 10n ** 5000n], score: 1.5, note: { deep: 10n ** 5000n } }],
])("%s in a partial-record output is accepted", (_name, output) => {
  const { state, log } = partialRecordWorking();

  commitPrepared(log, state, prepareOutput(log, state, "work", output));

  expect(state.pool.get("work")).toEqual(output);
  expect(state.ready).toEqual([END_ID]);
});

test("a whole float output stays a float", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "first"));

  commitPrepared(log, state, prepareOutput(log, state, "first", 2));

  expect((log.records().at(-2)!.record as PoolCommitRecord).value).toEqual({ kind: "float", value: 2 });
  expect(state.pool.getValue("first")!.kind).toBe("float");
  expect(state.pool.get("first")).toBe(2);
});

test("the END output waits for full quiescence", () => {
  const { state, log } = endCommittedWithSiblingInFlight();

  expect(state.pool.get(END_ID)).toEqual({ terminal: ["stable"] });
  expect(state.result).toBeNull();
  expect(state.ready).toEqual(["queued"]);
  expect([...state.inFlight]).toEqual(["sibling"]);

  let observers = commitPrepared(log, state, prepareOutput(log, state, "sibling", { done: true }));

  expect(observers).toStrictEqual([new NodeSucceeded({ nodeId: "sibling", output: { done: true } })]);
  expect(state.result).toBeNull();
  expect(state.ready).toEqual(["queued"]);
  expect([...state.inFlight]).toEqual([]);

  commitPrepared(log, state, prepareNodeStarted(state, "queued"));
  observers = commitPrepared(log, state, prepareOutput(log, state, "queued", { done: "last" }));

  expect(observers).toStrictEqual([
    new NodeSucceeded({ nodeId: "queued", output: { done: "last" } }),
    new RunSucceeded({ output: { terminal: ["stable"] } }),
  ]);
  expect(state.result).toStrictEqual(new RunSucceeded({ output: { terminal: ["stable"] } }));
  expect(state.ready).toEqual([]);
  expect([...state.inFlight]).toEqual([]);
});

test("a newly ready successor prevents quiescence", () => {
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["worker", func("worker")],
      ["after", func("after")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-end", START_ID, END_ID),
      new Edge("start-worker", START_ID, "worker"),
      new Edge("worker-after", "worker", "after"),
    ],
  );
  const { state, log } = initializedAndStarted(flow);
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, END_ID));
  commitPrepared(log, state, prepareNodeStarted(state, "worker"));
  commitPrepared(log, state, prepareOutput(log, state, END_ID, "done"));

  const observers = commitPrepared(log, state, prepareOutput(log, state, "worker", "worker-done"));

  expect(observers).toStrictEqual([new NodeSucceeded({ nodeId: "worker", output: "worker-done" })]);
  expect(state.result).toBeNull();
  expect(state.ready).toEqual(["after"]);
  expect([...state.inFlight]).toEqual([]);
});

test("a late sibling failure wins", () => {
  const { state, log } = endCommittedWithSiblingInFlight();
  const failure = nodeFailure("sibling", "late boom", { causeType: "RuntimeError" });

  const observers = commitPrepared(log, state, prepareNodeFailed(state, failure));

  expect(observers).toStrictEqual([new NodeFailed({ failure }), new RunFailed({ failure })]);
  expect(state.pool.get(END_ID)).toEqual({ terminal: ["stable"] });
  expect(state.result).toStrictEqual(new RunFailed({ failure }));
  expect(state.ready).toEqual(["queued"]);
});

test("a typed null END membership succeeds at quiescence", () => {
  const { state, log } = endOnlyInFlight();

  const prepared = prepareOutput(log, state, END_ID, null);

  const terminal = successPlan(prepared).terminal!;
  expect(typeof terminal.outputJson).toBe("string");
  expect(() => {
    (terminal as { outputJson: unknown }).outputJson = "changed";
  }).toThrow(TypeError);
  expect(prepared.observers).toStrictEqual([
    new NodeSucceeded({ nodeId: END_ID, output: null }),
    new RunSucceeded({ output: null }),
  ]);
  const observers = commitPrepared(log, state, prepared);

  expect(state.pool.store.has(END_ID)).toBe(true);
  expect(state.pool.getValue(END_ID)).not.toBeNull();
  expect(state.pool.get(END_ID)).toBeNull();
  expect(state.result).toStrictEqual(new RunSucceeded({ output: null }));
  expect(observers).toStrictEqual(prepared.observers);
  expect(state.ready).toEqual([]);
});

test("a terminal derivation replays from the causal records only (success)", () => {
  const live = endOnlyInFlight();
  const expected = commitPrepared(live.log, live.state, prepareOutput(live.log, live.state, END_ID, { value: [1n, 2n] }));

  const replay = replayOutcomeLog(startEndFlow(), live.log);

  expect(replay.observers.at(-1)).toStrictEqual(expected);
  expect(reducerSnapshot(replay.state)).toEqual(reducerSnapshot(live.state));
  expect(replay.log.records()).toEqual(live.log.records());
  expect(replay.state.ready).toEqual([]);
  expect(
    replay.log
      .records()
      .every(
        (envelope) =>
          !envelope.record.constructor.name.startsWith("RunSucceeded") &&
          !envelope.record.constructor.name.startsWith("RunFailed"),
      ),
  ).toBe(true);
});

test("a terminal append failure leaves the result, the frontier and the pool unchanged (success)", () => {
  const { state, log: prior } = endOnlyInFlight();
  const prepared = prepareOutput(prior, state, END_ID, { value: [1n, 2n] });
  const before = snapshot(state);
  const log = new FailingLog(RUN_ID, prior.records());

  expectError(() => commitPrepared(log, state, prepared), Error, "disk full");

  expectUnchanged(state, before);
  expect(state.result).toBeNull();
  expect(log.nextSequence).toBe(prior.nextSequence);
});

test("terminal success values are mutation isolated", () => {
  const { state, log } = endOnlyInFlight();
  type Nested = { nested: string[][] };
  const raw: Nested = { nested: [["original"]] };
  const prepared = prepareOutput(log, state, END_ID, raw);
  raw.nested[0]!.push("raw mutation");
  ((prepared.observers[0] as NodeSucceeded).output as Nested).nested[0]!.push("prepared node mutation");
  ((prepared.observers[1] as RunSucceeded).output as Nested).nested[0]!.push("prepared run mutation");

  const observers = commitPrepared(log, state, prepared);

  const expected = { nested: [["original"]] };
  expect(state.pool.get(END_ID)).toEqual(expected);
  expect(state.result).toStrictEqual(new RunSucceeded({ output: expected }));
  expect(observers).toStrictEqual([
    new NodeSucceeded({ nodeId: END_ID, output: expected }),
    new RunSucceeded({ output: expected }),
  ]);
  const nodeOutput = (observers[0] as NodeSucceeded).output as Nested;
  const runOutput = (observers[1] as RunSucceeded).output as Nested;
  const resultOutput = (state.result as RunSucceeded).output as Nested;
  expect(nodeOutput).not.toBe(runOutput);
  expect(runOutput).not.toBe(resultOutput);
  nodeOutput.nested[0]!.push("returned node mutation");
  runOutput.nested[0]!.push("returned run mutation");
  resultOutput.nested[0]!.push("state mutation");
  expect(state.pool.get(END_ID)).toEqual(expected);
  expect(state.ready).toEqual([]);
});

/** A memory log holding `source`'s records that makes `flow`'s adjacency throw once it appends. */
class PoisonAfterAppendLog extends MemoryRunLog {
  readonly #flow: CompiledFlow;

  constructor(source: MemoryRunLog, flow: CompiledFlow) {
    super(source.runId);
    for (const envelope of source.records()) super.append(envelope.record);
    this.#flow = flow;
  }

  override append(record: RunRecord): RecordEnvelope {
    const envelope = super.append(record);
    const unexpected = (): never => {
      throw new Error("terminal commit materialized data after append");
    };
    this.#flow.incoming = unexpected;
    this.#flow.outgoing = unexpected;
    return envelope;
  }
}

test("a terminal commit does no materialization after append (success)", () => {
  const { state, log: prior } = endOnlyInFlight();
  const prepared = prepareOutput(prior, state, END_ID, { value: [1n, 2n] });
  const log = new PoisonAfterAppendLog(prior, state.flow);

  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual([
    new NodeSucceeded({ nodeId: END_ID, output: { value: [1n, 2n] } }),
    new RunSucceeded({ output: { value: [1n, 2n] } }),
  ]);
  expect(state.result).toStrictEqual(new RunSucceeded({ output: { value: [1n, 2n] } }));
  expect(state.ready).toEqual([]);
});

/** Return `prepared`'s plan narrowed to a route plan; throw unless it is one. */
function routePlan(prepared: PreparedTransition) {
  const plan = prepared._plan;
  if (plan.kind !== "route") throw new Error(`expected a route plan, found ${plan.kind}`);
  return plan;
}

/** Return the route and skip observers of one route. */
function routed(nodeId: string, handle: string, ...skipped: string[]): RunEvent[] {
  return [new NodeRouted({ nodeId, handle }), ...skipped.map((id) => new NodeSkipped({ nodeId: id }))];
}

test("a route derives its skip closure without a skip record", () => {
  const { state, log } = routeInFlight();
  const before = snapshot(state);

  const prepared = prepareNodeRouted(state, "router", "yes");

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(7);
  expect(prepared.record).toStrictEqual(new NodeRoutedRecord("router", "yes"));
  expect(prepared.observers).toStrictEqual(routed("router", "yes", "dead-parent", "dead-child"));
  const delta = routePlan(prepared).delta;
  expect(delta.edgeUpdates).toEqual([
    ["route-dead", NodeState.SKIPPED],
    ["dead-child", NodeState.SKIPPED],
    ["route-chosen", NodeState.TAKEN],
  ]);
  expect(() => {
    (delta as { edgeUpdates: unknown }).edgeUpdates = [];
  }).toThrow(TypeError);

  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual(prepared.observers);
  expect(state.nodeState.get("router")).toBe(NodeState.TAKEN);
  expect(state.nodeState.get("dead-parent")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("dead-child")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("chosen")).toBe(NodeState.UNKNOWN);
  expect(state.ready).toEqual(["chosen"]);
  expect([...state.inFlight]).toEqual([]);
  expect(log.records().map((envelope) => envelope.record.constructor)).toEqual([
    RunStartedRecord,
    PoolCommitRecord,
    PoolCommitRecord,
    NodeStartedRecord,
    PoolCommitRecord,
    NodeSucceededRecord,
    NodeStartedRecord,
    NodeRoutedRecord,
  ]);
});

test.each([ELSE_HANDLE, "go"])("a route takes every edge matching the else or a named handle: %s", (handle) => {
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["router", func("router")],
      ["left", func("left")],
      ["right", func("right")],
      ["dead", func("dead")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-router", START_ID, "router"),
      new Edge("match-left", "router", "left", { sourceHandle: handle }),
      new Edge("match-right", "router", "right", { sourceHandle: handle }),
      new Edge("other", "router", "dead", { sourceHandle: "other" }),
    ],
  );
  const { state, log } = routeInFlight(flow);

  commitPrepared(log, state, prepareNodeRouted(state, "router", handle));

  expect(state.edgeState.get("match-left")).toBe(NodeState.TAKEN);
  expect(state.edgeState.get("match-right")).toBe(NodeState.TAKEN);
  expect(state.edgeState.get("other")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("dead")).toBe(NodeState.SKIPPED);
  expect(state.ready).toEqual(["left", "right"]);
});

test("a route keeps DFS observers and the existing FIFO readiness", () => {
  const ids = ["router", "existing", "dead-a", "ready-a", "chosen", "dead-b", "ready-b"];
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ...ids.map((id) => [id, func(id)] as const),
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-router", START_ID, "router"),
      new Edge("start-existing", START_ID, "existing"),
      new Edge("route-a", "router", "dead-a", { sourceHandle: "no-a" }),
      new Edge("route-chosen", "router", "chosen", { sourceHandle: "yes" }),
      new Edge("route-b", "router", "dead-b", { sourceHandle: "no-b" }),
      new Edge("a-ready", "dead-a", "ready-a", { optional: true }),
      new Edge("b-ready", "dead-b", "ready-b", { optional: true }),
    ],
  );
  const { state, log } = routeInFlight(flow);

  const observers = commitPrepared(log, state, prepareNodeRouted(state, "router", "yes"));

  expect(observers).toStrictEqual(routed("router", "yes", "dead-a", "dead-b"));
  expect(state.ready).toEqual(["existing", "ready-a", "chosen", "ready-b"]);
});

test.each([
  ["missing", "NodeRouted node 'missing' does not exist in the flow"],
  ["chosen", "NodeRouted node 'chosen' is not in-flight"],
] as const)("a route refuses an unknown router or frontier: %s", (nodeId, message) => {
  const { state, log } = routeInFlight();
  const before = snapshot(state);

  expectError(() => prepareNodeRouted(state, nodeId, "yes"), RangeError, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(7);
});

test("a route refuses an unknown handle as node data", () => {
  const { state, log } = routeInFlight();
  const before = snapshot(state);

  expectNodeData(
    () => prepareNodeRouted(state, "router", "missing"),
    RangeError,
    "NodeRouted handle 'missing' matches no outgoing edge of 'router'",
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(7);
});

test.each([NodeState.TAKEN, NodeState.SKIPPED])("a route refuses a previously settled outgoing edge: %s", (settled) => {
  const { state, log } = routeInFlight();
  state.edgeState.set("route-dead", settled);
  const before = snapshot(state);

  expectError(() => prepareNodeRouted(state, "router", "yes"), RangeError, "route edge 'route-dead' must be UNKNOWN");

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(7);
});

test.each<[unknown, new (...args: never[]) => Error, string]>([
  ["", RangeError, "route edge 'route-chosen' source_handle must be non-empty"],
  [7, TypeError, "route edge 'route-chosen' source_handle must be a string or null"],
  [false, TypeError, "route edge 'route-chosen' source_handle must be a string or null"],
])("a route refuses a malformed source handle as a host invariant: %p", (sourceHandle, cls, message) => {
  const { state, log } = routeInFlight();
  const outgoing = state.flow.outgoingEdges.get("router")!;
  const selected = outgoing.find((edge) => edge.sourceHandle === "yes")!;
  // The Edge constructor doesn't check its fields, so a malformed handle reaches the reducer.
  const malformed = new Edge(selected.id, selected.from, selected.to, { sourceHandle: sourceHandle as string });
  outgoing[outgoing.indexOf(selected)] = malformed;
  state.flow.edges[state.flow.edges.indexOf(selected)] = malformed;
  const before = snapshot(state);

  expectError(() => prepareNodeRouted(state, "router", "yes"), cls, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(7);
});

test("a route requires an exact non-empty node id", () => {
  const { state } = routeInFlight();

  expectError(() => prepareNodeRouted(state, "", "yes"), RangeError, "node_id must be non-empty");
  expectError(() => prepareNodeRouted(state, 1 as unknown as string, "yes"), TypeError, "node_id must be a string");
});

test("a route requires an exact non-empty handle, as node data", () => {
  const { state } = routeInFlight();

  expectNodeData(() => prepareNodeRouted(state, "router", ""), RangeError, "handle must be non-empty");
  expectNodeData(
    () => prepareNodeRouted(state, "router", 1 as unknown as string),
    TypeError,
    "handle must be a string",
  );
});

test("a route append failure leaves every state coordinate unchanged", () => {
  const { state, log: prior } = routeInFlight();
  const prepared = prepareNodeRouted(state, "router", "yes");
  const before = snapshot(state);
  const log = new FailingLog(RUN_ID, prior.records());

  expectError(() => commitPrepared(log, state, prepared), Error, "disk full");

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(prior.nextSequence);
});

test("a route delta keeps every reducer container's identity", () => {
  const { state, log } = routeInFlight();
  const before = roots(state);

  commitPrepared(log, state, prepareNodeRouted(state, "router", "yes"));

  roots(state).forEach((root, index) => expect(root).toBe(before[index]));
  expect(state.ready).toEqual(["chosen"]);
});

test("competing route plans refuse the second sequence before append", () => {
  const { state, log } = routeInFlight();
  const first = prepareNodeRouted(state, "router", "yes");
  const second = prepareNodeRouted(state, "router", "yes");

  commitPrepared(log, state, first);
  const sequence = log.nextSequence;
  expectError(() => commitPrepared(log, state, second), RangeError, "run-state sequence mismatch: prepared 7, state 8");

  expect(log.nextSequence).toBe(sequence);
  expect(state.ready).toEqual(["chosen"]);
});

test("route observers are fresh and isolated from a prepared mutation", () => {
  const { state, log } = routeInFlight();
  const prepared = prepareNodeRouted(state, "router", "yes");
  (prepared.observers[0] as { handle: string }).handle = "mutated";
  (prepared.observers[1] as { nodeId: string }).nodeId = "mutated";

  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual(routed("router", "yes", "dead-parent", "dead-child"));
  expect(state.ready).toEqual(["chosen"]);
});

test("a route record replay reconstructs an identical state", () => {
  const live = routeInFlight();
  commitPrepared(live.log, live.state, prepareNodeRouted(live.state, "router", "yes"));

  const replay = replayOutcomeLog(routeFlow(), live.log);

  expect(reducerSnapshot(replay.state)).toEqual(reducerSnapshot(live.state));
  expect(replay.log.records()).toEqual(live.log.records());
  expect(replay.state.ready).toEqual(["chosen"]);
});

test("a route commit uses only the materialized delta after append", () => {
  const { state, log: prior } = routeInFlight();
  const prepared = prepareNodeRouted(state, "router", "yes");
  const poolBefore = state.pool.toWire();
  const unexpected = (): never => {
    throw new Error("route commit repeated preparation work");
  };
  state.flow.incoming = unexpected;
  state.flow.outgoing = unexpected;

  const observers = commitPrepared(prior, state, prepared);

  expect(observers).toStrictEqual(routed("router", "yes", "dead-parent", "dead-child"));
  expect(state.pool.toWire()).toEqual(poolBefore);
  expect(state.ready).toEqual(["chosen"]);
});

test("a route skip flood is iterative for a deep chain", () => {
  const depth = 100_000;
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["router", func("router")],
    ["chosen", func("chosen")],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  for (let index = 0; index < depth; index++) nodes.set(`dead-${index}`, func(`dead-${index}`));
  const edges = [
    new Edge("start-router", START_ID, "router"),
    new Edge("route-dead", "router", "dead-0", { sourceHandle: "no" }),
    new Edge("route-live", "router", "chosen", { sourceHandle: "yes" }),
  ];
  for (let index = 0; index < depth - 1; index++) {
    edges.push(new Edge(`dead-edge-${index}`, `dead-${index}`, `dead-${index + 1}`));
  }
  const { state, log } = routeInFlight(new CompiledFlow(nodes, edges));

  const observers = commitPrepared(log, state, prepareNodeRouted(state, "router", "yes"));

  expect(observers.length).toBe(depth + 1);
  expect(observers[1]).toStrictEqual(new NodeSkipped({ nodeId: "dead-0" }));
  expect(observers.at(-1)).toStrictEqual(new NodeSkipped({ nodeId: `dead-${depth - 1}` }));
  expect(state.ready).toEqual(["chosen"]);
});

test("a route skip join matches the control, data, ordering and wait semantics", () => {
  const ids = [
    "router",
    "dead-parent",
    "control-dead",
    "data-dead",
    "ordering-dead",
    "waiting",
    "optional-ready",
    "runs-after-ready",
    "unused",
  ];
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ...ids.map((id) => [id, func(id)] as const),
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-router", START_ID, "router"),
      new Edge("start-control-data", START_ID, "control-dead", { inputGroup: "value" }),
      new Edge("route-parent", "router", "dead-parent", { sourceHandle: "no" }),
      new Edge("route-live", "router", "unused", { sourceHandle: "yes" }),
      new Edge("parent-control", "dead-parent", "control-dead", { sourceHandle: "gate" }),
      new Edge("parent-data", "dead-parent", "data-dead", { inputGroup: "required" }),
      new Edge("parent-ordering", "dead-parent", "ordering-dead", { ordering: true }),
      new Edge("parent-ordering-again", "dead-parent", "ordering-dead", { ordering: true }),
      new Edge("parent-wait", "dead-parent", "waiting", { inputGroup: "both" }),
      new Edge("unused-wait", "unused", "waiting", { inputGroup: "both" }),
      new Edge("parent-optional", "dead-parent", "optional-ready", { optional: true }),
      new Edge("parent-runs-after", "dead-parent", "runs-after-ready", { optional: true, ordering: true }),
    ],
  );
  const { state, log } = routeInFlight(flow);

  const observers = commitPrepared(log, state, prepareNodeRouted(state, "router", "yes"));

  expect(observers).toStrictEqual(
    routed("router", "yes", "dead-parent", "control-dead", "data-dead", "ordering-dead"),
  );
  expect(state.nodeState.get("waiting")).toBe(NodeState.UNKNOWN);
  expect(state.edgeState.get("start-control-data")).toBe(NodeState.TAKEN);
  expect(state.edgeState.get("unused-wait")).toBe(NodeState.UNKNOWN);
  expect(state.edgeState.get("parent-ordering-again")).toBe(NodeState.SKIPPED);
  expect(state.ready).toEqual(["optional-ready", "runs-after-ready", "unused"]);
});

test("a route scans each touched head's incoming adjacency once", () => {
  const edges = [new Edge("start-router", START_ID, "router")];
  for (let index = 0; index < 64; index++) {
    edges.push(new Edge(`route-${index}`, "router", "head", { sourceHandle: `no-${index}` }));
  }
  edges.push(new Edge("route-live", "router", "head", { sourceHandle: "yes" }));
  const flow = new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["router", func("router")],
      ["head", func("head")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    edges,
  );
  const { state } = routeInFlight(flow);
  const incoming = flow.incoming.bind(flow);
  const scans = new Map<string, number>();
  flow.incoming = (nodeId) => {
    scans.set(nodeId, (scans.get(nodeId) ?? 0) + 1);
    return incoming(nodeId);
  };

  const prepared = prepareNodeRouted(state, "router", "yes");

  expect([...scans]).toEqual([["head", 1]]);
  expect(routePlan(prepared).delta.newlyReadyIds).toEqual(["head"]);
});

/** A valid front-end-specific origin, for testing durable normalization. */
class SpecialOrigin extends OriginRef {}

/** Return a live run with the terminal-skipping router as its sole work. */
function terminalSkipRouterInFlight(): { state: RunState; log: MemoryRunLog } {
  return routeInFlight(terminalSkipRouteFlow());
}

/** Return a live run with the causal router (origin `router-site`) and `unrelated` in flight. */
function terminalSkipWithSiblingInFlight(): { state: RunState; log: MemoryRunLog } {
  const { state, log } = routeInFlight(
    terminalSkipRouteFlow({ routerOrigin: new OriginRef("router-site"), sibling: true }),
  );
  commitPrepared(log, state, prepareNodeStarted(state, "unrelated"));
  return { state, log };
}

/** The diagnosis of a route that first skips END's `zeta` group. */
const ZETA = "terminal output 'zeta' skipped";

/** Return the derived failure of a route from `router` that first skips END's `zeta` group. */
function zetaUnreachable(trace: readonly OriginRef[] = []): Failure {
  return new Failure({
    code: "graph.terminal_unreachable",
    message: ZETA,
    nodeId: "router",
    trace,
  });
}

test("a route's terminal skip names the first required END group and adds no record", () => {
  const { state, log } = terminalSkipRouterInFlight();

  const prepared = prepareNodeRouted(state, "router", "go");

  const expectedFailure = zetaUnreachable();
  const failure = (prepared.record as NodeRoutedRecord).terminalFailure;
  expect(failure).toBe(routePlan(prepared).delta.terminalUnreachableFailure);
  expect(failure).toStrictEqual(expectedFailure);
  const expectedTerminal = new RunFailed({ failure: expectedFailure });
  expect(prepared.observers).toStrictEqual([
    ...routed("router", "go", "dead-optional", "dead-z", END_ID, "dead-a"),
    expectedTerminal,
  ]);
  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual(prepared.observers);
  expect(state.result).toStrictEqual(expectedTerminal);
  expect(state.terminalUnreachableFailure).toStrictEqual(expectedFailure);
  expect((state.result as RunFailed).failure).toBe(state.terminalUnreachableFailure!);
  expect(state.ready).toEqual([]);
  expect([...state.inFlight]).toEqual([]);
  const records = log.records();
  expect(records.at(-1)!.record.constructor).toBe(NodeRoutedRecord);
  expect((records.at(-1)!.record as NodeRoutedRecord).terminalFailure).toStrictEqual(expectedFailure);
  expect(records.length).toBe(8);
});

test("a route latches its causal terminal failure until sibling quiescence", () => {
  const { state, log } = terminalSkipWithSiblingInFlight();
  const outer = new OriginRef("outer-site");
  node(state.flow, "router").originContext = [outer];

  const prepared = prepareNodeRouted(state, "router", "go");
  const failure = routePlan(prepared).delta.terminalUnreachableFailure!;

  expect(failure).toStrictEqual(zetaUnreachable([outer, new OriginRef("router-site")]));
  expect(failure.trace).toEqual([outer, node(state.flow, "router").origin!]);
  expect((prepared.record as NodeRoutedRecord).terminalFailure).toBe(failure);
  expect(routePlan(prepared).terminal).toBeNull();
  expect(prepared.observers.some((event) => event instanceof RunFailed)).toBe(false);
  expect(state.terminalUnreachableFailure).toBeNull();

  const routeObservers = commitPrepared(log, state, prepared);

  expect(routeObservers).toStrictEqual(prepared.observers);
  expect(state.terminalUnreachableFailure).toBe(failure);
  expect(state.result).toBeNull();
  expect(state.ready).toEqual([]);

  const final = prepareOutput(log, state, "unrelated", { done: true });
  expect(successPlan(final).terminal!.failure).toBe(failure);
  expect(final.observers).toStrictEqual([
    new NodeSucceeded({ nodeId: "unrelated", output: { done: true } }),
    new RunFailed({ failure }),
  ]);

  const observers = commitPrepared(log, state, final);

  expect((observers.at(-1) as RunFailed).failure).toBe(failure);
  expect((state.result as RunFailed | null)?.failure).toBe(failure);
  expect(state.terminalUnreachableFailure).toBe(failure);
  expect(log.records().some((envelope) => envelope.record instanceof NodeFailedRecord)).toBe(false);
  expect(log.records().slice(-3).map((envelope) => envelope.record.constructor)).toEqual([
    NodeRoutedRecord,
    PoolCommitRecord,
    NodeSucceededRecord,
  ]);
  expect(state.ready).toEqual([]);
});

test("a causal route canonicalizes an origin subclass for the wire", () => {
  const { state, log } = routeInFlight(
    terminalSkipRouteFlow({ routerOrigin: new SpecialOrigin("special-router-site") }),
  );

  const prepared = prepareNodeRouted(state, "router", "go");
  const failure = (prepared.record as NodeRoutedRecord).terminalFailure!;

  expect(failure.trace).toEqual([new OriginRef("special-router-site")]);
  expect(failure.trace[0]!.constructor).toBe(OriginRef);
  expect(failure).toBe(routePlan(prepared).delta.terminalUnreachableFailure!);

  commitPrepared(log, state, prepared);

  expect((state.result as RunFailed | null)?.failure).toBe(failure);
  const logged = (log.records().at(-1)!.record as NodeRoutedRecord).terminalFailure!;
  expect(logged).toStrictEqual(failure);
  expect(logged.trace[0]!.constructor).toBe(OriginRef);
  expect(state.ready).toEqual([]);
});

test("an explicit failure wins after the terminal-unreachable latch is set", () => {
  const { state, log } = terminalSkipWithSiblingInFlight();
  const route = prepareNodeRouted(state, "router", "go");
  const routeFailure = routePlan(route).delta.terminalUnreachableFailure;
  commitPrepared(log, state, route);
  const explicit = nodeFailure("unrelated", "explicit failure", { causeType: "RuntimeError" });

  const prepared = prepareNodeFailed(state, explicit);
  const failure = (prepared.record as NodeFailedRecord).failure;
  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual([new NodeFailed({ failure }), new RunFailed({ failure })]);
  expect((observers[0] as NodeFailed).failure).toBe(failure);
  expect((observers[1] as RunFailed).failure).toBe(failure);
  expect((state.result as RunFailed | null)?.failure).toBe(failure);
  expect(failure).toStrictEqual(explicit);
  expect(failure).not.toBe(routeFailure);
  expect(state.terminalUnreachableFailure).toBe(routeFailure);
  expect(state.ready).toEqual([]);
});

test("the terminal-unreachable latch is installed only after the route's append", () => {
  const { state, log: prior } = terminalSkipWithSiblingInFlight();
  const prepared = prepareNodeRouted(state, "router", "go");
  const before = snapshot(state);
  const log = new FailingLog(RUN_ID, prior.records());

  expectError(() => commitPrepared(log, state, prepared), Error, "disk full");

  expectUnchanged(state, before);
  expect(state.terminalUnreachableFailure).toBeNull();
  expect(log.nextSequence).toBe(prior.nextSequence);
});

test("a causal route replay requires its persisted terminal failure", () => {
  const { state, log } = terminalSkipRouterInFlight();
  const before = snapshot(state);

  expectError(
    () => prepareRecord(state, new NodeRoutedRecord("router", "go")),
    RangeError,
    "causal NodeRoutedRecord requires a persisted terminal failure",
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(state.nextSequence);
});

test("a noncausal route replay refuses a terminal failure", () => {
  const { state, log } = routeInFlight();
  const failure = new Failure({
    code: "graph.terminal_unreachable",
    message: "terminal output null skipped",
    nodeId: "router",
  });
  const before = snapshot(state);

  expectError(
    () => prepareRecord(state, new NodeRoutedRecord("router", "yes", { terminalFailure: failure })),
    RangeError,
    "NodeRoutedRecord terminal failure is present on a noncausal route",
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(state.nextSequence);
});

test.each<[string, Failure, string]>([
  [
    "code",
    new Failure({ code: "node.execute", message: ZETA, nodeId: "router" }),
    "route terminal failure has the wrong failure code",
  ],
  [
    "node",
    new Failure({ code: "graph.terminal_unreachable", message: ZETA, nodeId: "other" }),
    "route terminal failure has the wrong causal node ID",
  ],
  [
    "message",
    new Failure({ code: "graph.terminal_unreachable", message: "wrong", nodeId: "router" }),
    "route terminal failure has the wrong message",
  ],
  [
    "cause",
    new Failure({ code: "graph.terminal_unreachable", message: ZETA, nodeId: "router", causeType: "RuntimeError" }),
    "route terminal failure cannot carry exception diagnostics",
  ],
  [
    "traceback",
    new Failure({ code: "graph.terminal_unreachable", message: ZETA, nodeId: "router", traceback: "trace" }),
    "route terminal failure cannot carry exception diagnostics",
  ],
])("a causal route replay refuses a mismatched terminal failure: %s", (_name, failure, message) => {
  const { state, log } = terminalSkipRouterInFlight();
  const before = snapshot(state);

  expectError(
    () => prepareRecord(state, new NodeRoutedRecord("router", "go", { terminalFailure: failure })),
    RangeError,
    message,
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(state.nextSequence);
});

test("quiescence without a retained causal failure is a graph fault", () => {
  const { state, log } = terminalSkipWithSiblingInFlight();
  commitPrepared(log, state, prepareNodeRouted(state, "router", "go"));
  state.terminalUnreachableFailure = null;
  commitPrepared(log, state, preparePoolCommit(state, "unrelated", { done: true }));
  const before = snapshot(state);

  expectError(
    () => prepareNodeSucceeded(state, "unrelated"),
    RangeError,
    "run quiesced without END output or a causal terminal failure",
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(state.nextSequence);
});

/** Return the sibling terminal-skip flow with a rebased router origin and ancestry. */
function rebasedTerminalSkipFlow(): CompiledFlow {
  const flow = terminalSkipRouteFlow({ routerOrigin: new OriginRef("rebased-router-site"), sibling: true });
  node(flow, "router").originContext = [new OriginRef("rebased-outer-site")];
  return flow;
}

test("the terminal-unreachable latch replays before and at quiescence", () => {
  const live = terminalSkipWithSiblingInFlight();
  node(live.state.flow, "router").originContext = [new OriginRef("outer-site")];
  commitPrepared(live.log, live.state, prepareNodeRouted(live.state, "router", "go"));

  const prefix = replayOutcomeLog(rebasedTerminalSkipFlow(), live.log);

  expect(prefix.observers.at(-1)!.at(-1)).toStrictEqual(new NodeSkipped({ nodeId: "dead-a" }));
  expect(prefix.state.result).toBeNull();
  expect(prefix.state.terminalUnreachableFailure).toStrictEqual(live.state.terminalUnreachableFailure);
  expect(reducerSnapshot(prefix.state)).toEqual(reducerSnapshot(live.state));
  expect(prefix.log.records()).toEqual(live.log.records());
  expect(prefix.state.ready).toEqual([]);

  const liveObservers = commitPrepared(
    live.log,
    live.state,
    prepareOutput(live.log, live.state, "unrelated", { done: true }),
  );
  const final = replayOutcomeLog(rebasedTerminalSkipFlow(), live.log);

  expect(final.observers.at(-1)).toStrictEqual(liveObservers);
  expect(reducerSnapshot(final.state)).toEqual(reducerSnapshot(live.state));
  expect(final.log.records()).toEqual(live.log.records());
  const failure = (final.state.result as RunFailed).failure;
  expect(failure).toBe(final.state.terminalUnreachableFailure!);
  expect(failure.nodeId).toBe("router");
  expect(failure.trace).toEqual([new OriginRef("outer-site"), new OriginRef("router-site")]);
  expect(final.state.ready).toEqual([]);
});

/** A minimal spawner for the terminal-latch pruning test. */
class OutcomeSpawner extends Node {
  readonly kind = NodeKind.CALL;
  override readonly isSpawner = true;
  override readonly growDepthDelta = 0;
}

/** Return static `creator` and `pruner` spawners after START, and END's one required `result`. */
function terminalPruneParentFlow(): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["creator", new OutcomeSpawner("creator")],
    ["pruner", new OutcomeSpawner("pruner")],
    [END_ID, EndNode.value(END_ID, { outputName: "result" })],
  ]);
  const edges = [
    new Edge("start-creator", START_ID, "creator"),
    new Edge("start-pruner", START_ID, "pruner"),
    new Edge("creator-end", "creator", END_ID, { inputGroup: "result" }),
  ];
  return new CompiledFlow(nodes, edges, { wiring: new Map([...nodes.keys()].map((id) => [id, new Map()])) });
}

/** Return `nodes` and `edges` as a fragment from `startId` to `endId`, with empty wiring. */
function fragmentOf(nodes: Map<string, Node>, edges: Edge[], startId: string, endId: string): Flow {
  return new Flow(nodes, edges, { wiring: new Map([...nodes.keys()].map((id) => [id, new Map()])), startId, endId });
}

/**
 * Return a dynamic fragment whose router's `go` route skips `dead`, which skips the fragment END's
 * `zeta` and so global END's `result`.
 */
function terminalRouterFragment(spawnerId: string): Flow {
  const startId = `${spawnerId}/child/start`;
  const routerId = `${spawnerId}/child/router`;
  const deadId = `${spawnerId}/child/dead`;
  const endId = `${spawnerId}/child/end`;
  const end = EndNode.value(endId, { outputName: "zeta" });
  end.commitAs = spawnerId;
  const router = new CodeNode(routerId, { code: "return inputs" });
  router.origin = new OriginRef("dynamic-router-site");
  const nodes = new Map<string, Node>([
    [startId, new StartNode(startId)],
    [routerId, router],
    [deadId, new CodeNode(deadId, { code: "return inputs" })],
    [endId, end],
  ]);
  const edges = [
    new Edge(`${startId}-router`, startId, routerId),
    new Edge(`${routerId}-end`, routerId, endId, { sourceHandle: "go" }),
    new Edge(`${routerId}-dead`, routerId, deadId, { sourceHandle: "skip" }),
    new Edge(`${deadId}-end`, deadId, endId, { inputGroup: "zeta" }),
  ];
  return fragmentOf(nodes, edges, startId, endId);
}

/** Return a minimal START -> END fragment for the unrelated pruner. */
function terminalPrunerFragment(spawnerId: string): Flow {
  const startId = `${spawnerId}/child/start`;
  const endId = `${spawnerId}/child/end`;
  const end = EndNode.value(endId, { outputName: null });
  end.commitAs = spawnerId;
  const nodes = new Map<string, Node>([
    [startId, new StartNode(startId)],
    [endId, end],
  ]);
  return fragmentOf(nodes, [new Edge(`${startId}-end`, startId, endId, { ordering: true })], startId, endId);
}

test("the terminal-unreachable latch survives pruning its dynamic cause", () => {
  const { state, log } = initializedAndStarted(terminalPruneParentFlow());
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "creator"));
  commitPrepared(log, state, prepareNodeStarted(state, "pruner"));
  const routerBuilder: GrowBuilder = (spawner) => terminalRouterFragment(spawner.id);
  commitPrepared(log, state, prepareGrowth(log, state, "creator", new Grow({}), routerBuilder));
  const routerId = "creator/child/router";
  commitPrepared(log, state, prepareNodeStarted(state, "creator/child/start"));
  commitPrepared(log, state, prepareOutput(log, state, "creator/child/start", {}));
  commitPrepared(log, state, prepareNodeStarted(state, routerId));

  const route = prepareNodeRouted(state, routerId, "go");
  const failure = routePlan(route).delta.terminalUnreachableFailure!;
  commitPrepared(log, state, route);

  expect(failure).toStrictEqual(
    new Failure({
      code: "graph.terminal_unreachable",
      message: "terminal output 'result' skipped",
      nodeId: routerId,
      trace: [new OriginRef("dynamic-router-site")],
    }),
  );
  expect(state.result).toBeNull();
  expect(state.terminalUnreachableFailure).toBe(failure);
  expect(state.initialStaticIds.has(routerId)).toBe(false);
  expect(state.ready).toEqual([]);

  const prunerBuilder: GrowBuilder = (spawner) => terminalPrunerFragment(spawner.id);
  commitPrepared(log, state, prepareGrowth(log, state, "pruner", new Grow({}, [routerId]), prunerBuilder));

  expect(state.flow.nodes.has(routerId)).toBe(false);
  expect(state.terminalUnreachableFailure).toBe(failure);
  expect(state.result).toBeNull();
  expect(state.ready).toEqual(["pruner/child/start"]);

  commitPrepared(log, state, prepareNodeStarted(state, "pruner/child/start"));
  commitPrepared(log, state, prepareOutput(log, state, "pruner/child/start", {}));
  commitPrepared(log, state, prepareNodeStarted(state, "pruner/child/end"));
  const observers = commitPrepared(log, state, prepareOutput(log, state, "pruner/child/end", {}));

  expect(observers.at(-1)).toStrictEqual(new RunFailed({ failure }));
  expect((observers.at(-1) as RunFailed).failure).toBe(failure);
  expect((state.result as RunFailed).failure).toBe(failure);
  expect(state.terminalUnreachableFailure).toBe(failure);
  expect(state.ready).toEqual([]);
});

test("a terminal-skip diagnosis scans only END beyond the touched route joins", () => {
  const { state } = terminalSkipRouterInFlight();
  const incoming = state.flow.incoming.bind(state.flow);
  const scanned: string[] = [];
  state.flow.incoming = (nodeId) => {
    scanned.push(nodeId);
    return incoming(nodeId);
  };

  prepareNodeRouted(state, "router", "go");

  expect(scanned).toEqual([END_ID, "dead-optional", "dead-z", "dead-a", END_ID]);
});

test("a terminal derivation replays from the causal records only (route)", () => {
  const live = terminalSkipRouterInFlight();
  const expected = commitPrepared(live.log, live.state, prepareNodeRouted(live.state, "router", "go"));

  const replay = replayOutcomeLog(terminalSkipRouteFlow(), live.log);

  expect(replay.observers.at(-1)).toStrictEqual(expected);
  expect(reducerSnapshot(replay.state)).toEqual(reducerSnapshot(live.state));
  expect(replay.log.records()).toEqual(live.log.records());
  expect(replay.state.ready).toEqual([]);
  expect(
    replay.log
      .records()
      .every(
        (envelope) =>
          !envelope.record.constructor.name.startsWith("RunSucceeded") &&
          !envelope.record.constructor.name.startsWith("RunFailed"),
      ),
  ).toBe(true);
});

test("a terminal append failure leaves the result, the frontier and the pool unchanged (route)", () => {
  const { state, log: prior } = terminalSkipRouterInFlight();
  const prepared = prepareNodeRouted(state, "router", "go");
  const before = snapshot(state);
  const log = new FailingLog(RUN_ID, prior.records());

  expectError(() => commitPrepared(log, state, prepared), Error, "disk full");

  expectUnchanged(state, before);
  expect(state.result).toBeNull();
  expect(log.nextSequence).toBe(prior.nextSequence);
});

test("a terminal commit does no materialization after append (route)", () => {
  const { state, log: prior } = terminalSkipRouterInFlight();
  const prepared = prepareNodeRouted(state, "router", "go");
  const log = new PoisonAfterAppendLog(prior, state.flow);

  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual([
    ...routed("router", "go", "dead-optional", "dead-z", END_ID, "dead-a"),
    new RunFailed({ failure: zetaUnreachable() }),
  ]);
  expect(state.result).toStrictEqual(new RunFailed({ failure: zetaUnreachable() }));
  expect(state.ready).toEqual([]);
});

/** Return END and two ordinary nodes concurrently in flight, plus a queued leaf. */
function failureFlow(): CompiledFlow {
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["failing", func("failing")],
      ["sibling", func("sibling")],
      ["queued", func("queued")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [
      new Edge("start-end", START_ID, END_ID),
      new Edge("start-failing", START_ID, "failing"),
      new Edge("start-sibling", START_ID, "sibling"),
      new Edge("start-queued", START_ID, "queued"),
    ],
  );
}

/** Return a live run whose END has its output while `failing` and its sibling run. */
function failureInFlight(): { state: RunState; log: MemoryRunLog } {
  const { state, log } = initializedAndStarted(failureFlow());
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, END_ID));
  commitPrepared(log, state, prepareNodeStarted(state, "failing"));
  commitPrepared(log, state, prepareNodeStarted(state, "sibling"));
  commitPrepared(log, state, prepareOutput(log, state, END_ID, {}));
  return { state, log };
}

/** Return `prepared`'s plan narrowed to a failure plan; throw unless it is one. */
function failurePlan(prepared: PreparedTransition) {
  const plan = prepared._plan;
  if (plan.kind !== "failure") throw new Error(`expected a failure plan, found ${plan.kind}`);
  return plan;
}

test("a failure is terminal and keeps its diagnostics", () => {
  const { state, log } = failureInFlight();
  const failure = nodeFailure("failing", "value must be positive", {
    causeType: "NodeAssertFailed",
    traceback: "Traceback (most recent call last):\n  boom",
  });
  const poolBefore = state.pool.toWire();
  const edgesBefore = new Map(state.edgeState);
  const readyBefore = [...state.ready];
  const sequenceBefore = state.nextSequence;

  const prepared = prepareNodeFailed(state, failure);

  const expectedRecord = new NodeFailedRecord(failure);
  const authority = (prepared.record as NodeFailedRecord).failure;
  const expectedResult = new RunFailed({ failure: authority });
  expect(prepared.record).toStrictEqual(expectedRecord);
  expect(failurePlan(prepared).failure).toBe(authority);
  expect(prepared.observers).toStrictEqual([new NodeFailed({ failure: authority }), expectedResult]);
  expect((prepared.observers[0] as NodeFailed).failure).toBe(authority);
  expect((prepared.observers[1] as RunFailed).failure).toBe(authority);
  expect(state.result).toBeNull();
  expect(state.pool.get(END_ID)).toEqual({});

  (prepared.observers[0] as { failure: Failure }).failure = nodeFailure("failing", "mutated");
  (prepared.observers[1] as { failure: Failure }).failure = nodeFailure("failing", "mutated");
  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual([new NodeFailed({ failure: authority }), expectedResult]);
  expect(state.result).toStrictEqual(expectedResult);
  expect(observers[1]).not.toBe(state.result);
  expect((observers[0] as NodeFailed).failure).toBe(authority);
  expect((observers[1] as RunFailed).failure).toBe(authority);
  expect((state.result as RunFailed | null)?.failure).toBe(authority);
  (observers[1] as { failure: Failure }).failure = nodeFailure("failing", "mutated returned observer");
  expect((state.result as RunFailed | null)?.failure.message).toBe("value must be positive");
  expect(state.nodeState.get("failing")).toBe(NodeState.TAKEN);
  expect(state.nodeState.get("sibling")).toBe(NodeState.UNKNOWN);
  expect([...state.inFlight]).toEqual(["sibling"]);
  expect(state.ready).toEqual(readyBefore);
  expect(state.ready).toEqual(["queued"]);
  expect(state.pool.toWire()).toEqual(poolBefore);
  expect(state.edgeState).toEqual(edgesBefore);
  expect(state.nextSequence).toBe(sequenceBefore + 1);
  expect(log.records().at(-1)!.record).toStrictEqual(expectedRecord);
});

test("a failure transition canonicalizes an origin subclass for the wire", () => {
  const { state, log } = failureInFlight();
  const supplied = new Failure({
    code: "node.execute",
    message: "boom",
    nodeId: "failing",
    trace: [new SpecialOrigin("special-failure-site")],
  });

  const prepared = prepareNodeFailed(state, supplied);
  const failure = (prepared.record as NodeFailedRecord).failure;

  expect(failure.trace).toEqual([new OriginRef("special-failure-site")]);
  expect(failure.trace[0]!.constructor).toBe(OriginRef);
  expect(failure).toBe(failurePlan(prepared).failure);
  expect((prepared.observers[0] as NodeFailed).failure).toBe(failure);
  expect((prepared.observers[1] as RunFailed).failure).toBe(failure);

  commitPrepared(log, state, prepared);

  expect((state.result as RunFailed | null)?.failure).toBe(failure);
  const logged = (log.records().at(-1)!.record as NodeFailedRecord).failure;
  expect(logged).toStrictEqual(failure);
  expect(logged.trace[0]!.constructor).toBe(OriginRef);
  expect(state.ready).toEqual(["queued"]);
});

test("a failure record replay reconstructs an identical state", () => {
  const live = failureInFlight();
  commitPrepared(
    live.log,
    live.state,
    prepareNodeFailed(live.state, nodeFailure("failing", "boom", { causeType: "ValueError", traceback: "trace" })),
  );

  const replay = replayOutcomeLog(failureFlow(), live.log);

  expect(reducerSnapshot(replay.state)).toEqual(reducerSnapshot(live.state));
  expect(replay.log.records()).toEqual(live.log.records());
  expect(replay.state.ready).toEqual(["queued"]);
});

test("a failure append failure leaves every state coordinate unchanged", () => {
  const { state, log: prior } = failureInFlight();
  const prepared = prepareNodeFailed(state, nodeFailure("failing"));
  const before = snapshot(state);
  const log = new FailingLog(RUN_ID, prior.records());

  expectError(() => commitPrepared(log, state, prepared), Error, "disk full");

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(prior.nextSequence);
});

test.each<[string, () => Failure, new (...args: never[]) => Error, string]>([
  ["not a Failure", () => 1 as unknown as Failure, TypeError, "prepareNodeFailed expects a Failure"],
  [
    "no node id",
    () => new Failure({ code: "node.execute", message: "boom", nodeId: null }),
    RangeError,
    "NodeFailedRecord failure node_id must be a non-empty string",
  ],
])("a failure requires exact diagnostic carriers before append: %s", (_name, failure, cls, message) => {
  const { state, log } = failureInFlight();
  const before = snapshot(state);

  expectError(() => prepareNodeFailed(state, failure()), cls, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(state.nextSequence);
});

test.each([
  ["missing", "NodeFailed node 'missing' does not exist in the flow"],
  ["queued", "NodeFailed node 'queued' is not in-flight"],
] as const)("a failure requires an existing unknown in-flight node: %s", (nodeId, message) => {
  const { state, log } = failureInFlight();
  const before = snapshot(state);

  expectError(() => prepareNodeFailed(state, nodeFailure(nodeId)), RangeError, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(state.nextSequence);
});

test("a failure requires an initialized nonterminal state and an UNKNOWN node", () => {
  const pristine = new RunState(failureFlow());
  expectError(() => prepareNodeFailed(pristine, nodeFailure("failing")), RangeError, "run state is not initialized");

  const { state, log } = failureInFlight();
  state.nodeState.set("failing", NodeState.TAKEN);
  let before = snapshot(state);
  expectError(
    () => prepareNodeFailed(state, nodeFailure("failing")),
    RangeError,
    "NodeFailed node 'failing' must have UNKNOWN disposition",
  );
  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(state.nextSequence);

  state.nodeState.set("failing", NodeState.UNKNOWN);
  state.result = new RunFailed({ failure: nodeFailure("failing", "already terminal") });
  before = snapshot(state);
  expectError(
    () => prepareNodeFailed(state, nodeFailure("failing")),
    RangeError,
    "terminal run state cannot prepare another graph transition",
  );
  expectUnchanged(state, before);
});

test("a failure record refuses a non-Failure authority at construction", () => {
  expectError(() => new NodeFailedRecord({} as Failure), TypeError, "NodeFailedRecord failure must be a Failure");
});

test("a failure record refuses the derived terminal-unreachable code", () => {
  const { state, log } = failureInFlight();
  const failure = new Failure({
    code: "graph.terminal_unreachable",
    message: "terminal output 'result' skipped",
    nodeId: "failing",
  });
  const before = snapshot(state);

  for (const call of [
    () => prepareNodeFailed(state, failure),
    () => prepareRecord(state, new NodeFailedRecord(failure)),
  ]) {
    expectError(call, RangeError, "graph.terminal_unreachable must be derived from a route record");
  }

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(state.nextSequence);
});

test("a competing failure plan is refused after another same-sequence outcome", () => {
  const { state, log } = failureInFlight();
  const failure = prepareNodeFailed(state, nodeFailure("failing"));
  const siblingSuccess = prepareOutput(log, state, "sibling", { ok: true });

  commitPrepared(log, state, siblingSuccess);
  const sequence = log.nextSequence;
  expectError(() => commitPrepared(log, state, failure), RangeError, "run-state sequence mismatch: prepared 11, state 13");

  expect(log.nextSequence).toBe(sequence);
  expect(state.result).toBeNull();
  expect([...state.inFlight]).toEqual(["failing"]);
  expect(state.ready).toEqual(["queued"]);
});

test("a failure commit uses only the materialized data after append", () => {
  const { state, log: prior } = failureInFlight();
  const failure = nodeFailure("failing", "boom", { causeType: "RuntimeError" });
  const prepared = prepareNodeFailed(state, failure);
  const unexpected = (): never => {
    throw new Error("failure commit repeated preparation work");
  };
  state.flow.incoming = unexpected;
  state.flow.outgoing = unexpected;
  const log = new PoisonAfterAppendLog(prior, state.flow);

  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual([new NodeFailed({ failure }), new RunFailed({ failure })]);
  expect(state.result).toStrictEqual(new RunFailed({ failure }));
  expect(state.ready).toEqual(["queued"]);
});

/** Return a route whose `no` branch feeds END through one data edge with no input group. */
function unnamedEndInputFlow(): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["router", new FuncNode("router", () => null)],
    ["yes", new FuncNode("yes", (inputs) => inputs)],
    ["no", new FuncNode("no", (inputs) => inputs)],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  return new CompiledFlow(nodes, [
    new Edge("start-router", START_ID, "router"),
    new Edge("router-yes", "router", "yes", { sourceHandle: "yes" }),
    new Edge("router-no", "router", "no", { sourceHandle: "no" }),
    new Edge("no-end", "no", END_ID),
  ]);
}

test("a skipped END input with no group name derives a failure naming null, and replays", () => {
  const live = routeInFlight(unnamedEndInputFlow());
  commitPrepared(live.log, live.state, prepareNodeRouted(live.state, "router", "yes"));
  expect(live.state.terminalUnreachableFailure?.message).toBe("terminal output null skipped");
  const replay = replayOutcomeLog(unnamedEndInputFlow(), live.log);
  expect(replay.state.terminalUnreachableFailure?.message).toBe("terminal output null skipped");
});
