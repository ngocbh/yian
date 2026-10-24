/**
 * Transactional dispatch and recovery-frontier tests for the run-state reducer: a port of
 * `tests/engine/test_run_state_dispatch.py`.
 *
 * Ported: `test_node_started_requires_and_removes_the_exact_ready_head`,
 * `test_node_started_rejects_an_empty_wrong_or_unknown_ready_head` (3 rows),
 * `test_node_started_requires_a_string_node_id` (4 rows),
 * `test_node_started_requires_unknown_disposition_and_no_existing_flight`,
 * `test_retried_node_can_start_again_in_a_valid_record_sequence`,
 * `test_node_started_preparation_keeps_existing_frontier_live`,
 * `test_node_started_appends_before_applying_the_frontier_delta`,
 * `test_run_resumed_appends_before_applying_the_frontier_delta`,
 * `test_run_resumed_requires_an_exact_nonnegative_worker_count` (5 rows),
 * `test_run_resumed_allows_an_empty_retry_segment_and_changes_workers`,
 * `test_run_resumed_preserves_wide_ready_work_without_copying_it`,
 * `test_run_resumed_rejects_invalid_retry_frontiers` (2 rows),
 * `test_prepare_record_requires_the_exact_ordered_resume_retry_list` (4 rows),
 * `test_prepare_record_accepts_exact_dispatch_and_resume_records`,
 * `test_prepare_record_requires_a_builder_for_expansion_replay`,
 * `test_dispatch_and_resume_require_initialized_nonterminal_state`,
 * `test_competing_frontier_plans_reject_the_second_sequence_before_append` (2 rows),
 * `test_non_start_commit_requires_exact_log_identity_before_append` (2 rows),
 * `test_resume_rejects_cross_state_before_append`,
 * `test_dispatch_rejects_adopted_flow_or_boundary_replacement_before_append`,
 * `test_append_failure_leaves_dispatch_or_resume_state_exactly_unchanged` (2 rows),
 * `test_committed_frontier_deltas_preserve_container_identities` (2 rows) and
 * `test_non_start_append_return_mismatch_leaves_state_unapplied`. Python's `ValueError` is a
 * `RangeError` here; each rejected row names its one class and full message, where Python
 * matches a part of the message or accepts either class. `RUN_ID` is the shared fixture's, not
 * `dispatch-run`; Python's `_CoordinateLog` is a `ReturningLog` whose append throws. Each test
 * pins `state.ready` after the commits it makes.
 *
 * Changed:
 * - `test_run_resumed_preparation_preserves_retry_order_ahead_of_ready_work`: it also commits the
 *   resume and pins the retries ahead of the ready work.
 * - `test_run_resumed_validation_does_not_read_the_ready_frontier`: Python walks the function's
 *   AST; here `validateRunResumed`'s body is brace-matched in the blanked source (`source_scan.ts`)
 *   and holds no `state.ready`.
 * - `test_frontier_commit_uses_only_the_prevalidated_delta` (2 rows): only the instance patches,
 *   throwing `flow.incoming` and `flow.outgoing`.
 * - `test_frontier_commit_rejects_public_record_mutation_before_append`: the record is frozen, so a
 *   forged `PreparedTransition` pairs a different `NodeStartedRecord` with the original plan.
 * - `test_commit_returns_fresh_observers_derived_from_the_canonical_plan`: TS events aren't
 *   frozen, so the caller's change goes through a cast.
 * - `test_frontier_plan_is_frozen_and_contains_only_an_immutable_delta`: the envelope is canonical
 *   text, not bytes; assigning a plan field is a `TypeError`, and the delta is frozen too.
 * - The state snapshot keeps the root containers by reference and asserts each with `toBe`, for
 *   Python's `id()` values; it compares the pool as its wire object, keeps the terminal-unreachable
 *   latch as Python does, and adds the start counts, steps, crashed-attempt counts and last
 *   stop sequence.
 *
 * TS-only (RunStopped): `prepareRunStopped` lists the in-flight nodes in dispatch order; its
 * commit changes only the sequence and the last stop sequence, and returns no observers; a record
 * whose ids differ in order, miss one or add one is refused; a plan whose stop delta disagrees with
 * its record is refused; nothing in flight records no ids; a terminal state refuses it; after a
 * stop, a step and a resume still commit and the resume puts the stopped nodes first; an in-flight
 * node missing from the flow or not `UNKNOWN` is refused. Also: a frontier plan holding an outcome
 * record is refused; `prepareRecord` refuses a `RunStartedRecord`, hands a backward record to the
 * backward reducer (whose terminal check, not the graph check, refuses it on a nonterminal state),
 * and an expansion with a builder to the growth reducer, which checks the record's spawner before
 * anything else.
 *
 * TS-only (crashed-attempt counts, which Python doesn't keep): each resume counts every retried
 * start no stop covered; a stop covers the starts before it but not a later one, and two stops
 * cover everything before the second; a count survives the node's outcome; a resume clears the
 * last stop; `replayRecords` over the live log rebuilds the counts and the last stop.
 *
 * Not ported (ESM): the module patches of `test_frontier_commit_uses_only_the_prevalidated_delta`
 * (`_validate_node_started`, `_validate_run_resumed`, `encode_pool`, `decode_pool`).
 */

import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import { NodeStarted, RunFailed, RunResumed } from "../../src/events.ts";
import type { Node } from "../../src/nodes/base.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import {
  BackwardStartedRecord,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  RecordEnvelope,
  RunResumedRecord,
  type RunRecord,
  PoolCommitRecord,
  RunStartedRecord,
  RunStoppedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitInitialPool,
  commitPrepared,
  PreparedTransition,
  prepareNodeStarted,
  prepareNodeStep,
  prepareRecord,
  prepareRunResumed,
  prepareRunStarted,
  prepareRunStopped,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { blank, closing } from "./source_scan.ts";
import {
  commitStartSuccess,
  expectUnchanged,
  FailingLog,
  fanoutFlow,
  FINGERPRINT,
  fragmentBuilder,
  FuncNode,
  initializedAndStarted,
  prepareOutput,
  reducerSnapshot,
  ReturningLog,
  roots,
  RUN_ID,
  snapshot,
} from "./run_state_fixtures.ts";

/** Assert `fn` throws an error of exactly class `cls` whose message matches, or is, `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: RegExp | string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  if (typeof message === "string") expect((caught as Error).message).toBe(message);
  else expect((caught as Error).message).toMatch(message);
}

const TERMINAL = /^terminal run state cannot prepare another graph transition$/;

/** Return the frontier delta of `prepared`'s plan; throw unless the plan is a frontier plan. */
function frontierDelta(prepared: PreparedTransition) {
  const plan = prepared._plan;
  if (plan.kind !== "frontier") throw new Error(`expected a frontier plan, found ${plan.kind}`);
  return plan.delta;
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
    startWiringOverride: { metadata: { labels: ["one", "two"] } },
    effectiveLlmByStaticPath: { agent: { options: { stop: ["END"] } } },
    numWorkers: options.numWorkers ?? 2,
  });
  commitPrepared(log, state, prepared);
  commitInitialPool(log, state, pool);
  expectCommittedPath(log);
  return { state, log };
}

/** Assert one replay-shaped prefix with RunStarted and the run start's two commits, followed by `tail`. */
function expectCommittedPath(log: MemoryRunLog, ...tail: RunRecord[]): void {
  const envelopes = log.records();
  expect(envelopes.map((envelope) => envelope.seq)).toEqual(envelopes.map((_, index) => index));
  expect(envelopes.every((envelope) => envelope.runId === RUN_ID)).toBe(true);
  expect(envelopes[0]!.record).toBeInstanceOf(RunStartedRecord);
  expect(envelopes.slice(1, 3).map((envelope) => envelope.record.constructor)).toEqual([PoolCommitRecord, PoolCommitRecord]);
  expect(envelopes.slice(3).map((envelope) => envelope.record)).toStrictEqual(tail);
}

/** Accept starts from an isolated test frontier through the real reducer. */
function startFrontier(state: RunState, log: MemoryRunLog, ...nodeIds: string[]): void {
  state.ready = [...nodeIds];
  for (const nodeId of nodeIds) commitPrepared(log, state, prepareNodeStarted(state, nodeId));
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
  expect(state.acceptedStarts).toEqual(new Map([[START_ID, 3]]));
  expect(state.nodeStarts).toEqual(new Map([[START_ID, 1]]));
  expect(state.nodeState.get(START_ID)).toBe(NodeState.UNKNOWN);
  expect(state.nextSequence).toBe(4);
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
  expect(log.nextSequence).toBe(3);
});

test.each([["true", true], ["1", 1], ["null", null], ["an object", {}]] as const)(
  "NodeStarted requires a string node id: %s",
  (_name, invalid) => {
    const { state, log } = initialized();

    expectError(() => prepareNodeStarted(state, invalid as unknown as string), TypeError, /^node_id must be a string$/);

    expect(log.nextSequence).toBe(3);
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

  expect(log.nextSequence).toBe(4);
});

test("a retried node can start again in a valid record sequence", () => {
  const { state, log } = initialized();

  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  expect(state.ready).toEqual([]);
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 3 }));
  expect(state.ready).toEqual([START_ID]);
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));

  expect(state.ready).toEqual([]);
  expect([...state.inFlight]).toEqual([START_ID]);
  expect(state.acceptedStarts).toEqual(new Map([[START_ID, 5]]));
  expect(state.nodeStarts).toEqual(new Map([[START_ID, 2]]));
  expectCommittedPath(
    log,
    new NodeStartedRecord(START_ID),
    new RunResumedRecord([START_ID], 3),
    new NodeStartedRecord(START_ID),
  );
});

test("NodeStarted preparation keeps the existing frontier live", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first", "second");
  state.ready = ["ordinary"];

  const prepared = prepareNodeStarted(state, "ordinary");

  expect(prepared.record).toStrictEqual(new NodeStartedRecord("ordinary"));
  expect(frontierDelta(prepared)).toEqual({ kind: "started", nodeId: "ordinary" });
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

test("RunResumed appends before applying the frontier delta", () => {
  const { state, log: prior } = initialized(dispatchFlow(), { numWorkers: 2 });
  commitPrepared(prior, state, prepareNodeStarted(state, START_ID));
  const log = new InspectingLog(state, prior, [], [START_ID]);

  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 6 }));

  expect(log.sawAppend).toBe(true);
  expect(state.ready).toEqual([START_ID]);
  expect(state.inFlight).toEqual(new Set());
  expect(state.acceptedStarts).toEqual(new Map());
  expect(state.nodeStarts).toEqual(new Map([[START_ID, 1]]));
  expect(state.numWorkers).toBe(6);
  expectCommittedPath(log, new NodeStartedRecord(START_ID), new RunResumedRecord([START_ID], 6));
});

test("RunResumed preparation keeps the retry order ahead of ready work", () => {
  const { state, log } = initialized(dispatchFlow(), { numWorkers: 2 });
  startFrontier(state, log, "first", "second");
  state.ready = ["ordinary", END_ID];
  const before = snapshot(state);

  const prepared = prepareRunResumed(state, { numWorkers: 5 });

  expectUnchanged(state, before);
  expect(prepared.record).toStrictEqual(new RunResumedRecord(["first", "second"], 5));
  expect(prepared.observers).toStrictEqual([new RunResumed()]);
  expect(frontierDelta(prepared)).toEqual({
    kind: "resumed",
    retryNodeIds: ["first", "second"],
    numWorkers: 5,
  });
  expect(state.ready).toEqual(["ordinary", END_ID]);
  expect([...state.inFlight]).toEqual(["first", "second"]);
  expect(state.numWorkers).toBe(2);
  expect(state.nodeState.get("first")).toBe(NodeState.UNKNOWN);
  expect(state.nodeState.get("second")).toBe(NodeState.UNKNOWN);
  expectCommittedPath(log, new NodeStartedRecord("first"), new NodeStartedRecord("second"));

  commitPrepared(log, state, prepared);
  expect(state.ready).toEqual(["first", "second", "ordinary", END_ID]);
});

test.each([
  ["-1", -1, RangeError],
  ["true", true, TypeError],
  ["1.5", 1.5, TypeError],
  ['"3"', "3", TypeError],
  ["null", null, TypeError],
] as const)("RunResumed requires an exact non-negative worker count: %s", (_name, numWorkers, cls) => {
  const { state, log } = initialized();
  const before = snapshot(state);

  expectError(
    () => prepareRunResumed(state, { numWorkers: numWorkers as number }),
    cls,
    /^num_workers must be a non-negative integer$/,
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(3);
});

/** Return a START fan-out whose first successor can be retried over wide ready work. */
function wideResumeFlow(width: number): CompiledFlow {
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID)],
    ["retry", new FuncNode("retry", (inputs) => inputs)],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  const edges = [new Edge("start-retry", START_ID, "retry")];
  for (let index = 0; index < width; index++) {
    nodes.set(`queued-${index}`, new FuncNode(`queued-${index}`, (inputs) => inputs));
    edges.push(new Edge(`start-queued-${index}`, START_ID, `queued-${index}`));
  }
  return CompiledFlow.fromParts(nodes, edges);
}

test("RunResumed keeps wide ready work without copying it", () => {
  const { state, log } = initialized(wideResumeFlow(512));
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(log, state, prepareOutput(log, state, START_ID, { started: true }));
  commitPrepared(log, state, prepareNodeStarted(state, "retry"));

  const prepared = prepareRunResumed(state, { numWorkers: 3 });
  const ready = state.ready;
  commitPrepared(log, state, prepared);

  expect(state.ready).toBe(ready);
  expect(state.ready.length).toBe(513);
  expect(state.ready[0]).toBe("retry");
  expect(state.ready.slice(1)).toEqual(Array.from({ length: 512 }, (_, index) => `queued-${index}`));
  expect(state.inFlight).toEqual(new Set());
  expect(prepared.record).toStrictEqual(new RunResumedRecord(["retry"], 3));
});

test("RunResumed allows an empty retry segment and changes the workers", () => {
  const { state, log } = initialized(dispatchFlow(), { numWorkers: 4 });

  const prepared = prepareRunResumed(state, { numWorkers: 0 });
  commitPrepared(log, state, prepared);

  expect(prepared.record).toStrictEqual(new RunResumedRecord([], 0));
  expect(state.ready).toEqual([START_ID]);
  expect(state.inFlight).toEqual(new Set());
  expect(state.numWorkers).toBe(0);
  expectCommittedPath(log, new RunResumedRecord([], 0));
});

test("RunResumed validation doesn't read the ready frontier", () => {
  const source = blank(readFileSync(new URL("../../src/runtime/state.ts", import.meta.url), "utf8"));
  const start = source.indexOf("function validateRunResumed(");
  expect(start).toBeGreaterThan(-1);
  // Brace-match the body: strings and comments are blanked, so every brace is code.
  const open = source.indexOf("{", source.indexOf(")", start));
  const body = source.slice(open, closing(source, open) + 1);

  expect(body).toContain("state.inFlight");
  expect(body).not.toMatch(/\bstate\s*\.\s*ready\b/);
});

test.each([
  ["missing", /^RunResumed retry node 'first' does not exist in the flow$/],
  ["taken", /^RunResumed retry node 'first' must have UNKNOWN disposition$/],
] as const)("RunResumed rejects an invalid retry frontier: %s", (kind, message) => {
  const { state, log } = initialized();
  startFrontier(state, log, "first");
  if (kind === "missing") state.flow.nodes.delete("first");
  else state.nodeState.set("first", NodeState.TAKEN);
  const before = snapshot(state);

  expectError(() => prepareRunResumed(state, { numWorkers: 3 }), RangeError, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(4);
});

test.each([[[]], [["first"]], [["second", "first"]], [["first", "second", "ordinary"]]] as const)(
  "prepareRecord requires the exact ordered resume retry list: %j",
  (retryNodeIds) => {
    const { state, log } = initialized();
    startFrontier(state, log, "first", "second");
    const before = snapshot(state);

    expectError(
      () => prepareRecord(state, new RunResumedRecord(retryNodeIds, 3)),
      RangeError,
      "RunResumedRecord retry_node_ids must exactly equal ordered in-flight nodes: " +
        `expected ["first","second"], found ${JSON.stringify(retryNodeIds)}`,
    );

    expectUnchanged(state, before);
    expect(log.nextSequence).toBe(5);
  },
);

test("prepareRecord accepts exact dispatch and resume records", () => {
  const { state, log } = initialized();

  commitPrepared(log, state, prepareRecord(state, new NodeStartedRecord(START_ID)));
  expect(state.ready).toEqual([]);
  commitPrepared(log, state, prepareRecord(state, new RunResumedRecord([START_ID], 7)));

  expect(state.ready).toEqual([START_ID]);
  expect(state.inFlight).toEqual(new Set());
  expect(state.numWorkers).toBe(7);
  expectCommittedPath(log, new NodeStartedRecord(START_ID), new RunResumedRecord([START_ID], 7));
});

test("prepareRecord requires a builder for expansion replay", () => {
  const { state } = initialized();

  expectError(
    () => prepareRecord(state, new NodeExpandedRecord("first", {}, [], "0".repeat(64))),
    TypeError,
    /^NodeExpandedRecord replay requires a callable grow_builder$/,
  );
});

test("prepareRecord refuses a RunStartedRecord", () => {
  const { state, log } = initialized();
  const started = log.records()[0]!.record;

  expectError(
    () => prepareRecord(state, started),
    TypeError,
    "unsupported run record for state preparation: RunStartedRecord",
  );
});

test("prepareRecord hands a backward record to the backward reducer, which requires a terminal", () => {
  const { state } = initialized();

  expectError(
    () => prepareRecord(state, new BackwardStartedRecord(1, null)),
    RangeError,
    "backward records require an accepted terminal result",
  );
});

test("prepareRecord hands an expansion with a builder to the growth reducer", () => {
  const { state } = initialized();
  const record = new NodeExpandedRecord("first", {}, [], "0".repeat(64));

  expectError(
    () => prepareRecord(state, record, { growBuilder: () => state.flow }),
    RangeError,
    /^NodeExpanded node 'first' is not in-flight$/,
  );
});

test("dispatch and resume require an initialized nonterminal state", () => {
  const pristine = new RunState(dispatchFlow());
  expectError(() => prepareNodeStarted(pristine, START_ID), RangeError, /^run state is not initialized$/);
  expectError(() => prepareRunResumed(pristine, { numWorkers: 1 }), RangeError, /^run state is not initialized$/);

  const { state } = initialized();
  state.result = new RunFailed({ failure: new Failure({ code: "node.execute", message: "done", nodeId: "first" }) });
  expectError(() => prepareNodeStarted(state, START_ID), RangeError, TERMINAL);
  expectError(() => prepareRunResumed(state, { numWorkers: 1 }), RangeError, TERMINAL);
});

test.each(["start", "resume"] as const)(
  "competing frontier plans reject the second sequence before append: %s",
  (transition) => {
    const { state, log } = initialized();
    let first: PreparedTransition;
    let second: PreparedTransition;
    if (transition === "start") {
      first = prepareNodeStarted(state, START_ID);
      second = prepareNodeStarted(state, START_ID);
    } else {
      commitPrepared(log, state, prepareNodeStarted(state, START_ID));
      first = prepareRunResumed(state, { numWorkers: 3 });
      second = prepareRunResumed(state, { numWorkers: 3 });
    }

    commitPrepared(log, state, first);
    const sequence = log.nextSequence;
    expectError(
      () => commitPrepared(log, state, second),
      RangeError,
      `run-state sequence mismatch: prepared ${sequence - 1}, state ${sequence}`,
    );

    expect(log.nextSequence).toBe(sequence);
  },
);

test.each([
  ["log run id", "different", 3, /^run-log run_id mismatch: prepared 'run-123', log 'different'$/],
  ["log sequence", RUN_ID, 4, /^run-log sequence mismatch: prepared 3, log 4$/],
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

test("resume rejects a different state before append", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first", "second");
  state.ready = ["ordinary"];
  const prepared = prepareRunResumed(state, { numWorkers: 4 });

  const other = new RunState(state.flow);
  expectError(
    () => commitPrepared(log, other, prepared),
    RangeError,
    /^prepared transition belongs to a different originating RunState$/,
  );
  expect(log.nextSequence).toBe(5);
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

  expect(log.nextSequence).toBe(3);
});

/** Return state, log and a prepared start of START, or a resume after START's start. */
function preparedFrontier(transition: "start" | "resume"): {
  state: RunState;
  log: MemoryRunLog;
  prepared: PreparedTransition;
} {
  const { state, log } = initialized(dispatchFlow(), { numWorkers: 2 });
  if (transition === "start") return { state, log, prepared: prepareNodeStarted(state, START_ID) };
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  return { state, log, prepared: prepareRunResumed(state, { numWorkers: 6 }) };
}

test.each(["start", "resume"] as const)(
  "an append failure leaves the dispatch or resume state exactly unchanged: %s",
  (transition) => {
    const { state, log: prior, prepared } = preparedFrontier(transition);
    const before = snapshot(state);

    expectError(() => commitPrepared(new FailingLog(RUN_ID, prior), state, prepared), Error, /^disk full$/);

    expectUnchanged(state, before);
  },
);

test.each(["start", "resume"] as const)(
  "committed frontier deltas keep the container identities: %s",
  (transition) => {
    const { state, log, prepared } = preparedFrontier(transition);
    const before = roots(state);

    commitPrepared(log, state, prepared);

    roots(state).forEach((root, index) => expect(root).toBe(before[index]));
  },
);

test.each(["start", "resume"] as const)("a frontier commit uses only the prevalidated delta: %s", (transition) => {
  const { state, log, prepared } = preparedFrontier(transition);
  const unexpected = (): never => {
    throw new Error("commit repeated preparation work");
  };
  Object.assign(state.flow, { incoming: unexpected, outgoing: unexpected });

  commitPrepared(log, state, prepared);

  expect(state.nextSequence).toBe(log.nextSequence);
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

  expect(log.nextSequence).toBe(3);
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
  const returned = new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 4, prepared.record);
  const log = new ReturningLog(() => returned, { nextSequence: 3 });

  expectError(
    () => commitPrepared(log, state, prepared),
    RangeError,
    /^run-log append result does not match prepared transition$/,
  );

  expectUnchanged(state, before);
});

// TS-only: RunStopped.

test("prepareRunStopped lists the in-flight nodes in dispatch order", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "second", "first");
  const before = snapshot(state);

  const prepared = prepareRunStopped(state, "signal", "SIGINT");

  expectUnchanged(state, before);
  expect(prepared.record).toStrictEqual(new RunStoppedRecord("signal", "SIGINT", ["second", "first"]));
  expect(prepared.observers).toEqual([]);
  expect(frontierDelta(prepared)).toEqual({
    kind: "stopped",
    nodeIds: ["second", "first"],
  });
});

test("a RunStopped commit changes only the sequence and the last stop, and returns no observers", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first", "second");
  state.ready = ["ordinary"];
  const before = snapshot(state);

  const observers = commitPrepared(log, state, prepareRunStopped(state, "callback", "boom"));

  expect(observers).toEqual([]);
  expect(state.nextSequence).toBe(6);
  expectUnchanged(state, { ...before, values: { ...before.values, nextSequence: 6, lastStopSequence: 5 } });
  expect(state.ready).toEqual(["ordinary"]);
  expectCommittedPath(
    log,
    new NodeStartedRecord("first"),
    new NodeStartedRecord("second"),
    new RunStoppedRecord("callback", "boom", ["first", "second"]),
  );
});

test.each([
  ["order", ["second", "first"]],
  ["a missing node", ["first"]],
  ["an extra node", ["first", "second", "ordinary"]],
] as const)("RunStopped refuses node ids that differ from the in-flight nodes: %s", (_name, nodeIds) => {
  const { state, log } = initialized();
  startFrontier(state, log, "first", "second");
  const before = snapshot(state);

  expectError(
    () => prepareRecord(state, new RunStoppedRecord("signal", "", nodeIds)),
    RangeError,
    "RunStoppedRecord node_ids must exactly equal ordered in-flight nodes: " +
      `expected ["first","second"], found ${JSON.stringify(nodeIds)}`,
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(5);
});

test.each([
  ["missing", /^RunStopped node 'first' does not exist in the flow$/],
  ["taken", /^RunStopped node 'first' must have UNKNOWN disposition$/],
] as const)("RunStopped rejects an invalid in-flight node: %s", (kind, message) => {
  const { state, log } = initialized();
  startFrontier(state, log, "first");
  if (kind === "missing") state.flow.nodes.delete("first");
  else state.nodeState.set("first", NodeState.TAKEN);
  const before = snapshot(state);

  expectError(() => prepareRecord(state, new RunStoppedRecord("signal", "", ["first"])), RangeError, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(4);
});

test("a RunStopped plan whose delta disagrees with its record is refused", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first", "second");
  const prepared = prepareRunStopped(state, "signal", "");
  const delta = Object.freeze({ kind: "stopped", nodeIds: ["second", "first"] });
  const plan = Object.freeze({ ...prepared._plan, delta });
  const forged = new PreparedTransition(prepared.record, RUN_ID, prepared.expectedSequence, plan as never, []);

  expectError(() => commitPrepared(log, state, forged), RangeError, /^frontier stop delta disagrees with its record$/);

  expect(log.nextSequence).toBe(5);
});

test("RunStopped with nothing in flight records no node ids", () => {
  const { state, log } = initialized();

  const prepared = prepareRunStopped(state, "signal", "");
  commitPrepared(log, state, prepared);

  expect(prepared.record).toStrictEqual(new RunStoppedRecord("signal", "", []));
  expect(state.ready).toEqual([START_ID]);
  expect(state.nextSequence).toBe(4);
});

test("a terminal state refuses RunStopped", () => {
  const { state } = initialized();
  state.result = new RunFailed({ failure: new Failure({ code: "node.execute", message: "done", nodeId: "first" }) });

  expectError(() => prepareRunStopped(state, "signal", ""), RangeError, TERMINAL);
});

test("after a stop, a step and a resume still commit, and the resume puts the stopped nodes first", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first", "second");
  state.ready = ["ordinary"];

  commitPrepared(log, state, prepareRunStopped(state, "signal", ""));
  expect(state.ready).toEqual(["ordinary"]);
  commitPrepared(log, state, prepareNodeStep(state, "second", { kind: "test", event: "late", attempt: 1n }));
  expect(state.ready).toEqual(["ordinary"]);
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 1 }));

  expect(state.ready).toEqual(["first", "second", "ordinary"]);
  expect(state.inFlight).toEqual(new Set());
  expect(state.nodeStarts).toEqual(
    new Map([
      ["first", 1],
      ["second", 1],
    ]),
  );
  expect(state.nodeSteps).toEqual(new Map([["second", [{ attempt: 1n, event: "late", kind: "test" }]]]));
  expectCommittedPath(
    log,
    new NodeStartedRecord("first"),
    new NodeStartedRecord("second"),
    new RunStoppedRecord("signal", "", ["first", "second"]),
    new NodeStepRecord("second", 1, { kind: "test", event: "late", attempt: 1n }),
    new RunResumedRecord(["first", "second"], 1),
  );
});

test("a frontier plan holding an outcome record is refused", () => {
  const { state, log } = initialized();
  const prepared = prepareNodeStarted(state, START_ID);
  const failure = new Failure({ code: "node.execute", message: "done", nodeId: START_ID });
  const forged = new PreparedTransition(new NodeFailedRecord(failure), RUN_ID, 1, prepared._plan, []);

  expectError(
    () => commitPrepared(log, state, forged),
    TypeError,
    /^frontier plan requires a dispatch, resume, step or stop record$/,
  );

  expect(log.nextSequence).toBe(3);
});

// TS-only: crashed-attempt counts.

test("a resume counts each start it retries, once per resume", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first");

  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));
  expect(state.crashedAttempts).toEqual(new Map([["first", 1]]));
  expect(state.ready).toEqual(["first"]);
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));

  expect(state.crashedAttempts).toEqual(new Map([["first", 2]]));
  expect(state.ready).toEqual(["first"]);
});

test("a stop covers the starts before it, so their resume counts nothing", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first");

  commitPrepared(log, state, prepareRunStopped(state, "signal", ""));
  expect(state.lastStopSequence).toBe(4);
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));

  expect(state.crashedAttempts).toEqual(new Map());
  expect(state.lastStopSequence).toBeNull();
  expect(state.ready).toEqual(["first"]);
});

test("a start after a stop still counts, and the stopped one doesn't", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first");
  commitPrepared(log, state, prepareRunStopped(state, "signal", ""));
  startFrontier(state, log, "second");

  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));

  expect(state.crashedAttempts).toEqual(new Map([["second", 1]]));
  expect(state.ready).toEqual(["first", "second"]);
});

test("two stops in a row cover everything started before the second", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first");
  commitPrepared(log, state, prepareRunStopped(state, "signal", ""));
  startFrontier(state, log, "second");
  commitPrepared(log, state, prepareRunStopped(state, "callback", "boom"));
  expect(state.lastStopSequence).toBe(6);

  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));

  expect(state.crashedAttempts).toEqual(new Map());
  expect(state.ready).toEqual(["first", "second"]);
});

test("a crashed-attempt count survives the node's outcome, while its start count goes", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first");
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));
  commitPrepared(log, state, prepareNodeStarted(state, "first"));

  commitPrepared(log, state, prepareOutput(log, state, "first", {}));

  expect(state.nodeStarts.has("first")).toBe(false);
  expect(state.crashedAttempts).toEqual(new Map([["first", 1]]));
  expect(state.ready).toEqual(["second"]);
});

test("a resume clears the last stop, so an earlier segment's stop doesn't cover a later crash", () => {
  const { state, log } = initialized();
  startFrontier(state, log, "first");
  commitPrepared(log, state, prepareRunStopped(state, "signal", ""));
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));
  expect(state.lastStopSequence).toBeNull();
  commitPrepared(log, state, prepareNodeStarted(state, "first"));

  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));

  expect(state.crashedAttempts).toEqual(new Map([["first", 1]]));
  expect(state.lastStopSequence).toBeNull();
  expect(state.ready).toEqual(["first"]);
});

test("replayRecords rebuilds the crashed-attempt counts and the last stop of a live log", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  commitPrepared(log, state, prepareRunStopped(state, "signal", ""));
  commitPrepared(log, state, prepareNodeStarted(state, "second"));
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 2 }));
  commitPrepared(log, state, prepareNodeStarted(state, "first"));
  commitPrepared(log, state, prepareRunStopped(state, "signal", ""));
  expect(state.crashedAttempts).toEqual(
    new Map([
      ["first", 1],
      ["second", 1],
    ]),
  );
  expect(state.lastStopSequence).toBe(13);
  expect(state.ready).toEqual(["second"]);

  const replayed = replayRecords(new RunState(fanoutFlow()), log.records(), { growBuilder: fragmentBuilder });

  expect(reducerSnapshot(replayed)).toEqual(reducerSnapshot(state));
});
