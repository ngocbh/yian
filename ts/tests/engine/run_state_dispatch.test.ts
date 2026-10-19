/**
 * Transactional dispatch and recovery-frontier tests for the run-state reducer: a port of
 * `tests/engine/test_run_state_dispatch.py`.
 *
 * Ported: `test_node_started_requires_and_removes_the_exact_ready_head`,
 * `test_node_started_rejects_an_empty_wrong_or_unknown_ready_head` (3 rows),
 * `test_node_started_requires_a_string_node_id` (4 rows),
 * `test_node_started_requires_unknown_disposition_and_no_existing_flight`,
 * `test_node_started_preparation_keeps_existing_frontier_live`,
 * `test_node_started_appends_before_applying_the_frontier_delta`,
 * `test_prepare_record_requires_a_builder_for_expansion_replay`,
 * `test_non_start_commit_requires_exact_log_identity_before_append` (2 rows),
 * `test_dispatch_rejects_adopted_flow_or_boundary_replacement_before_append`,
 * `test_commit_returns_fresh_observers_derived_from_the_canonical_plan` and
 * `test_non_start_append_return_mismatch_leaves_state_unapplied`. Python's `ValueError` is a
 * `RangeError` here; each rejected row names its one class and full message, where Python
 * matches a part of the message. `RUN_ID` is the shared fixture's, not `dispatch-run`.
 *
 * Changed:
 * - `test_dispatch_and_resume_require_initialized_nonterminal_state`: its dispatch half.
 * - `test_frontier_plan_is_frozen_and_contains_only_an_immutable_delta`: the envelope is canonical
 *   text, not bytes; assigning a plan field is a `TypeError`, and the delta is frozen too.
 * - `test_frontier_commit_rejects_public_record_mutation_before_append`: the record is frozen, so a
 *   forged `PreparedTransition` pairs a different `NodeStartedRecord` with the original plan.
 * - `test_commit_returns_fresh_observers_derived_from_the_canonical_plan`: TS events aren't
 *   frozen, so the caller's change goes through a cast.
 * - The state snapshot keeps the root containers by reference and asserts each with `toBe`, for
 *   Python's `id()` values, and compares the pool as its wire object.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import { NodeStarted, RunFailed } from "../../src/events.ts";
import type { Node } from "../../src/nodes/base.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import {
  MemoryRunLog,
  NodeExpandedRecord,
  NodeStartedRecord,
  RecordEnvelope,
  type RunRecord,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  PreparedTransition,
  prepareNodeStarted,
  prepareRecord,
  prepareRunStarted,
  RunState,
} from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { FINGERPRINT, FuncNode, ReturningLog, RUN_ID, stateSnapshot } from "./run_state_fixtures.ts";

/** Assert `fn` throws an error of exactly class `cls` whose message matches `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: RegExp): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toMatch(message);
}

/** Return a graph with several nodes available for isolated frontier tests. */
function dispatchFlow(): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["first", new FuncNode("first", (inputs) => inputs)],
    ["second", new FuncNode("second", (inputs) => inputs)],
    ["ordinary", new FuncNode("ordinary", (inputs) => inputs)],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  return CompiledFlow.fromParts(nodes, [
    new Edge("start-first", START_ID, "first"),
    new Edge("first-second", "first", "second"),
    new Edge("second-ordinary", "second", "ordinary"),
    new Edge("ordinary-end", "ordinary", END_ID),
  ]);
}

/** Return state and log after their shared authoritative RunStarted record. */
function initialized(
  flow: CompiledFlow = dispatchFlow(),
  options: { numWorkers?: number } = {},
): { state: RunState; log: MemoryRunLog } {
  const state = new RunState(flow);
  const pool = new VariablePool(flow.startId);
  pool.set(flow.startId, { labels: ["one", "two"] });
  pool.addSystem("run_id", RUN_ID);
  const log = new MemoryRunLog(RUN_ID);
  const prepared = prepareRunStarted(state, {
    runId: RUN_ID,
    flowFingerprint: FINGERPRINT,
    initialPool: pool,
    startWiringOverride: { metadata: { labels: ["one", "two"] } },
    effectiveLlmByStaticPath: { agent: { options: { stop: ["END"] } } },
    numWorkers: options.numWorkers ?? 2,
  });
  commitPrepared(log, state, prepared);
  expectCommittedPath(log);
  return { state, log };
}

/** Assert one replay-shaped prefix with RunStarted followed by `tail`. */
function expectCommittedPath(log: MemoryRunLog, ...tail: RunRecord[]): void {
  const envelopes = log.records();
  expect(envelopes.map((envelope) => envelope.seq)).toEqual(envelopes.map((_, index) => index));
  expect(envelopes.every((envelope) => envelope.runId === RUN_ID)).toBe(true);
  expect(envelopes[0]!.record).toBeInstanceOf(RunStartedRecord);
  expect(envelopes.slice(1).map((envelope) => envelope.record)).toStrictEqual(tail);
}

/** Accept starts from an isolated test frontier through the real reducer. */
function startFrontier(state: RunState, log: MemoryRunLog, ...nodeIds: string[]): void {
  state.ready = [...nodeIds];
  for (const nodeId of nodeIds) commitPrepared(log, state, prepareNodeStarted(state, nodeId));
}

/** The root containers whose identity a frontier commit keeps. */
function roots(state: RunState): unknown[] {
  return [
    state.pool,
    state.nodeState,
    state.edgeState,
    state.ready,
    state.inFlight,
    state.acceptedStarts,
    state.depth,
    state.wiringOverrides,
    state.input,
    state.effectiveLlmByStaticPath,
  ];
}

/** Every reducer coordinate in a detached form, plus the root containers by reference. */
interface Snapshot {
  values: Record<string, unknown>;
  roots: unknown[];
}

/** Return `state`'s snapshot, with its start counts and steps as well. */
function snapshot(state: RunState): Snapshot {
  return {
    values: { ...stateSnapshot(state), nodeStarts: [...state.nodeStarts], nodeSteps: [...state.nodeSteps] },
    roots: roots(state),
  };
}

/** Assert `state` still has `before`'s values and root containers. */
function expectUnchanged(state: RunState, before: Snapshot): void {
  const after = snapshot(state);
  expect(after.values).toEqual(before.values);
  after.roots.forEach((root, index) => expect(root).toBe(before.roots[index]));
}

test("NodeStarted requires and removes the exact ready head", () => {
  const { state, log } = initialized();
  const before = snapshot(state);

  const prepared = prepareNodeStarted(state, START_ID);

  expectUnchanged(state, before);
  expect(prepared.record).toStrictEqual(new NodeStartedRecord(START_ID));
  expect(prepared.observers).toStrictEqual([new NodeStarted({ nodeId: START_ID })]);

  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual([new NodeStarted({ nodeId: START_ID })]);
  expect(state.ready).toEqual([]);
  expect([...state.inFlight]).toEqual([START_ID]);
  expect(state.acceptedStarts).toEqual(new Map([[START_ID, 1]]));
  expect(state.nodeStarts).toEqual(new Map([[START_ID, 1]]));
  expect(state.nodeState.get(START_ID)).toBe(NodeState.UNKNOWN);
  expect(state.nextSequence).toBe(2);
  expectCommittedPath(log, new NodeStartedRecord(START_ID));
});

test.each([
  ["first", [], /^NodeStarted requires ready head 'first', found null$/],
  ["second", ["first", "second"], /^NodeStarted requires ready head 'second', found 'first'$/],
  ["missing", ["missing"], /^NodeStarted node 'missing' does not exist in the flow$/],
] as const)("NodeStarted rejects an empty, wrong or unknown ready head: %s", (nodeId, ready, message) => {
  const { state, log } = initialized();
  state.ready = [...ready];
  const before = snapshot(state);

  expectError(() => prepareNodeStarted(state, nodeId), RangeError, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(1);
});

test.each([["true", true], ["1", 1], ["null", null], ["an object", {}]] as const)(
  "NodeStarted requires a string node id: %s",
  (_name, invalid) => {
    const { state, log } = initialized();

    expectError(() => prepareNodeStarted(state, invalid as unknown as string), TypeError, /^node_id must be a string$/);

    expect(log.nextSequence).toBe(1);
  },
);

test("NodeStarted requires UNKNOWN disposition and no existing flight", () => {
  const { state, log } = initialized();
  state.ready = ["first"];
  state.nodeState.set("first", NodeState.TAKEN);

  expectError(
    () => prepareNodeStarted(state, "first"),
    RangeError,
    /^NodeStarted node 'first' must have UNKNOWN disposition$/,
  );

  state.nodeState.set("first", NodeState.UNKNOWN);
  startFrontier(state, log, "first");
  state.ready.push("first");
  expectError(() => prepareNodeStarted(state, "first"), RangeError, /^NodeStarted node 'first' is already in-flight$/);

  expect(log.nextSequence).toBe(2);
});

test("NodeStarted preparation keeps the existing frontier live", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first", "second");
  state.ready = ["ordinary"];

  const prepared = prepareNodeStarted(state, "ordinary");

  expect(prepared.record).toStrictEqual(new NodeStartedRecord("ordinary"));
  expect(prepared._plan.kind === "frontier" && prepared._plan.delta).toEqual({ kind: "started", nodeId: "ordinary" });
  expect(state.ready).toEqual(["ordinary"]);
  expect([...state.inFlight]).toEqual(["first", "second"]);
  expectCommittedPath(log, new NodeStartedRecord("first"), new NodeStartedRecord("second"));
});

/** A memory log holding `prior`'s records that asserts the frontier is still unapplied at append. */
class InspectingLog extends MemoryRunLog {
  readonly #state: RunState;
  readonly #ready: readonly string[];
  readonly #inFlight: readonly string[];
  readonly #acceptedStarts: [string, number][];
  sawAppend = false;

  constructor(state: RunState, prior: MemoryRunLog, ready: readonly string[], inFlight: readonly string[]) {
    super(RUN_ID);
    for (const envelope of prior) super.append(envelope.record);
    this.#state = state;
    this.#ready = ready;
    this.#inFlight = inFlight;
    this.#acceptedStarts = [...state.acceptedStarts];
  }

  override append(record: RunRecord): RecordEnvelope {
    expect(this.#state.ready).toEqual([...this.#ready]);
    expect([...this.#state.inFlight]).toEqual([...this.#inFlight]);
    expect([...this.#state.acceptedStarts]).toEqual(this.#acceptedStarts);
    this.sawAppend = true;
    return super.append(record);
  }
}

test("NodeStarted appends before applying the frontier delta", () => {
  const { state, log: prior } = initialized();
  const log = new InspectingLog(state, prior, [START_ID], []);

  commitPrepared(log, state, prepareNodeStarted(state, START_ID));

  expect(log.sawAppend).toBe(true);
  expect(state.ready).toEqual([]);
  expect([...state.inFlight]).toEqual([START_ID]);
  expectCommittedPath(log, new NodeStartedRecord(START_ID));
});

test("prepareRecord requires a builder for expansion replay", () => {
  const { state } = initialized();

  expectError(
    () => prepareRecord(state, new NodeExpandedRecord("first", {}, [], "0".repeat(64))),
    TypeError,
    /^NodeExpandedRecord replay requires a callable grow_builder$/,
  );
});

test("dispatch requires an initialized nonterminal state", () => {
  const pristine = new RunState(dispatchFlow());
  expectError(() => prepareNodeStarted(pristine, START_ID), RangeError, /^run state is not initialized$/);

  const { state } = initialized();
  state.result = new RunFailed({ failure: new Failure({ code: "node.execute", message: "done", nodeId: "first" }) });
  expectError(
    () => prepareNodeStarted(state, START_ID),
    RangeError,
    /^terminal run state cannot prepare another graph transition$/,
  );
});

test.each([
  ["log run id", "different", 1, /^run-log run_id mismatch: prepared 'run-123', log 'different'$/],
  ["log sequence", RUN_ID, 2, /^run-log sequence mismatch: prepared 1, log 2$/],
] as const)("a non-start commit requires the exact log identity before append: %s", (_name, runId, next, message) => {
  const { state, log: prior } = initialized();
  const prepared = prepareNodeStarted(state, START_ID);
  const log = new ReturningLog(
    () => {
      throw new Error("coordinate mismatch must reject before append");
    },
    { runId, nextSequence: next },
  );
  const before = snapshot(state);

  expectError(() => commitPrepared(log, state, prepared), RangeError, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(next);
  expectCommittedPath(prior);
});

test("dispatch rejects an adopted flow or boundary replacement before append", () => {
  const { state, log } = initialized();
  state.ready = ["first"];
  const prepared = prepareNodeStarted(state, "first");
  state.flow.nodes.set(START_ID, new StartNode(START_ID));

  expectError(
    () => commitPrepared(log, state, prepared),
    RangeError,
    /^transition requires its originating flow and boundary identities$/,
  );

  expect(log.nextSequence).toBe(1);
});

test("a frontier commit rejects a changed public record before append", () => {
  const { state, log } = initialized();
  const prepared = prepareNodeStarted(state, START_ID);
  const forged = new PreparedTransition(
    new NodeStartedRecord("changed"),
    prepared.expectedRunId,
    prepared.expectedSequence,
    prepared._plan,
    prepared.observers,
  );

  expectError(() => commitPrepared(log, state, forged), RangeError, /^prepared record changed after validation$/);

  expect(log.nextSequence).toBe(1);
  expect(state.ready).toEqual([START_ID]);
});

test("commit returns fresh observers derived from the canonical plan", () => {
  const { state, log } = initialized();
  const prepared = prepareNodeStarted(state, START_ID);
  (prepared.observers[0] as { nodeId: string }).nodeId = "caller-mutated";

  const observers = commitPrepared(log, state, prepared);

  expect(observers).toStrictEqual([new NodeStarted({ nodeId: START_ID })]);
  expect(observers[0]).not.toBe(prepared.observers[0]);
  expect([...state.inFlight]).toEqual([START_ID]);
  expectCommittedPath(log, new NodeStartedRecord(START_ID));
});

test("a frontier plan is frozen and holds only an immutable delta", () => {
  const { state } = initialized();
  const flow = state.flow;
  state.ready = ["first", "second"];
  const prepared = prepareNodeStarted(state, "first");
  const plan = prepared._plan;
  if (plan.kind !== "frontier") throw new Error("expected a frontier plan");

  expect(plan.originatingState).toBe(state);
  expect(plan.originatingFlow).toBe(flow);
  expect(plan.delta).toEqual({ kind: "started", nodeId: "first" });
  expect(typeof plan.envelopeJson).toBe("string");
  expect("expectedReady" in plan).toBe(false);
  expect("expectedInFlight" in plan).toBe(false);
  expect("authoritySignature" in plan).toBe(false);
  expect(Object.isFrozen(plan)).toBe(true);
  expect(Object.isFrozen(plan.delta)).toBe(true);
  expect(() => {
    (plan as { delta: unknown }).delta = null;
  }).toThrow(TypeError);
});

test("a non-start append return mismatch leaves the state unapplied", () => {
  const { state } = initialized();
  const prepared = prepareNodeStarted(state, START_ID);
  const before = snapshot(state);
  const returned = new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 2, prepared.record);
  const log = new ReturningLog(() => returned, { nextSequence: 1 });

  expectError(
    () => commitPrepared(log, state, prepared),
    RangeError,
    /^run-log append result does not match prepared transition$/,
  );

  expectUnchanged(state, before);
});
