/**
 * Transactional node-outcome tests for the run-state reducer: a port of
 * `tests/engine/test_run_state_outcomes.py`.
 *
 * Ported: `test_route_derives_skip_closure_without_skip_record`,
 * `test_route_takes_every_edge_matching_else_or_named_handle` (2 rows),
 * `test_route_preserves_dfs_observers_and_existing_fifo_readiness`,
 * `test_route_skip_join_matches_control_data_ordering_and_wait_semantics`,
 * `test_route_rejects_a_previously_settled_outgoing_edge` (2 rows),
 * `test_route_append_failure_leaves_every_state_coordinate_unchanged`,
 * `test_route_delta_preserves_all_reducer_container_identities`,
 * `test_competing_route_plans_reject_the_second_sequence_before_append`,
 * `test_route_observers_are_fresh_and_isolated_from_prepared_mutation`,
 * `test_route_record_replay_reconstructs_identical_state`,
 * `test_route_scans_each_touched_head_incoming_adjacency_once`,
 * `test_route_skip_flood_is_iterative_for_a_deep_chain`,
 * `test_success_commits_canonical_value_before_fifo_successors`,
 * `test_success_replay_revalidates_partial_record_contract`,
 * `test_success_preserves_sibling_flight_and_appends_new_readiness_fifo`,
 * `test_success_resolves_redirect_precedence_and_preserves_target_disposition` (2 rows),
 * `test_success_rejects_malformed_target_contract_as_a_host_invariant`,
 * `test_success_replay_revalidates_target_output_contract_type`,
 * `test_success_rejects_type_or_missing_target_before_append` (2 rows),
 * `test_success_rejects_a_previously_settled_target_edge_before_append`,
 * `test_success_treats_an_invalid_baked_target_as_a_host_invariant_error`,
 * `test_success_rejects_a_settled_newly_ready_successor_before_append`,
 * `test_success_multiple_edges_to_one_successor_enqueue_it_once`,
 * `test_success_preserves_sequential_readiness_for_interleaved_successors`,
 * `test_success_scans_each_distinct_affected_join_once`,
 * `test_success_requires_an_existing_unknown_in_flight_node` (2 rows),
 * `test_success_append_failure_leaves_every_state_coordinate_unchanged`,
 * `test_success_delta_preserves_all_reducer_container_identities`,
 * `test_success_canonical_copies_isolate_input_record_pool_and_observers`,
 * `test_competing_success_plans_reject_the_second_sequence_before_append`,
 * `test_success_record_replay_reconstructs_identical_state`,
 * `test_end_output_waits_for_full_quiescence`, `test_newly_ready_successor_prevents_quiescence`,
 * `test_route_terminal_skip_uses_first_required_end_group_and_no_extra_record`,
 * `test_route_latches_causal_terminal_failure_until_sibling_quiescence`,
 * `test_causal_route_canonicalizes_origin_subclass_for_wire` (`SpecialOrigin extends OriginRef`,
 * checked through `constructor`),
 * `test_terminal_unreachable_latch_is_installed_only_after_route_append`,
 * `test_causal_route_replay_requires_its_persisted_terminal_failure`,
 * `test_noncausal_route_replay_rejects_a_terminal_failure` (its unused message reads `null`),
 * `test_causal_route_replay_rejects_mismatched_terminal_failure` (5 rows),
 * `test_quiescence_without_retained_causal_failure_is_a_graph_fault`,
 * `test_terminal_unreachable_latch_replays_before_and_at_quiescence`,
 * `test_terminal_skip_diagnosis_scans_only_end_beyond_touched_route_joins` (an instance patch of
 * `flow.incoming`), `test_terminal_success_values_are_mutation_isolated`, and both rows of
 * `test_terminal_derivation_replays_from_causal_records_only` (the record class names, not
 * Python's type names) and `test_terminal_append_failure_leaves_result_frontier_and_pool_unchanged`,
 * one test per row.
 * Python's `ValueError` is a `RangeError` here, and a type mismatch the value layer's
 * `TypeCheckError`; each rejected row names its one class and full message, where Python matches a
 * part of the message. A rejection Python raises as `_NodeDataError` is a `NodeDataError` whose
 * `original` is the `RangeError` or `TypeCheckError`. `RUN_ID` and `FINGERPRINT` are the shared
 * fixture's; the failing log is the shared `FailingLog` (its error is `disk full`, Python's
 * `append failed`); tuple outputs are arrays and ints `bigint`s. Each test that commits an outcome
 * pins `state.ready` after it.
 *
 * Changed:
 * - `test_route_rejects_unknown_router_frontier_or_handle` (3 rows): two tests, the router and
 *   frontier rows (`RangeError`) and the handle row (`NodeDataError`).
 * - `test_route_requires_exact_nonempty_strings` (2 rows): two tests, the node id (`RangeError`,
 *   `TypeError`) and the handle (`NodeDataError` over each); the non-string rows pass `1` by cast.
 * - `test_route_rejects_malformed_source_handle_as_a_host_invariant` (3 rows): the `Edge`
 *   constructor doesn't check its fields, so each row builds the malformed edge with it (`7` and
 *   `false` by cast) and puts it in the public `flow.outgoingEdges` list and `flow.edges`.
 * - `test_route_commit_uses_only_materialized_delta_after_append`: only the instance patches,
 *   throwing `flow.incoming` and `flow.outgoing` before the commit.
 * - `test_route_derives_skip_closure_without_skip_record`: assigning a delta field is a
 *   `TypeError`.
 * - `test_success_validates_ordinary_target_against_output_contract_type`: a `float` slot keeps
 *   the int `3n` an int (the value layer's rule), so the test also shows a text output refused
 *   against the target's `float` contract.
 * - `test_success_plan_contains_only_an_immutable_sparse_delta`: the value is canonical text, not
 *   bytes; assigning a delta field is a `TypeError`, and the id arrays are frozen.
 * - `test_success_commit_uses_only_the_prevalidated_delta`: only the instance patches, throwing
 *   `flow.incoming` and `flow.outgoing`.
 * - `test_typed_none_end_membership_succeeds_at_quiescence`: the terminal output is canonical
 *   text, not bytes, and assigning it is a `TypeError`.
 * - `test_terminal_commit_performs_no_materialization_after_append` (2 rows, one test each): after
 *   the append the log patches only the instance's `flow.incoming` and `flow.outgoing` to throw;
 *   each test pins the full observers where Python counts them.
 * - `test_success_rejects_touched_topology_invariant_breaks` (3 rows): the `incoming` row clears the
 *   public `flow.incomingEdges` entry of `after-baked`.
 * - The state snapshot keeps the root containers by reference and asserts each with `toBe`, for
 *   Python's `id()` values; it compares the pool as its wire object and adds the start counts and
 *   steps.
 *
 * TS-only: NaN in a typed and in an opaque field of a partial-record output each give a
 * `NodeDataError` whose `original` is the value layer's `TypeCheckError` at the field; `Infinity`
 * in both is accepted (D34); a long int in a declared and in an opaque field of a partial-record
 * output is accepted (the flip of `test_long_int_boundaries.py::
 * test_a_long_int_in_a_partial_record_output_names_the_limit`, since a `bigint` has no digit
 * limit); a whole float output stays a float.
 *
 * Not ported (no allocation probe): `test_route_reducer_allocation_is_independent_of_wide_ready_queue`,
 * `test_success_reducer_allocation_is_independent_of_total_run_size`.
 *
 * Not ported (ESM): the module patches of `test_route_commit_uses_only_materialized_delta_after_append`
 * (`_prepare_route_delta`, `_scan_join`, `_settle_join_edge`, `_classify_join`, and `NodeRouted`
 * and `NodeSkipped` after the append), of `test_success_commit_uses_only_the_prevalidated_delta`
 * (`_validate_success_value`, `_prepare_success_delta`, `canonical_typed`, `encode_pool`,
 * `decode_pool`) and of `test_terminal_commit_performs_no_materialization_after_append`
 * (`RunSucceeded`, `RunFailed`, `_materialize_terminal`).
 *
 * Later (T4-4 PR 2), failures: `test_late_sibling_failure_wins`,
 * `test_explicit_failure_wins_after_terminal_unreachable_is_latched`,
 * `test_failure_is_terminal_and_preserves_diagnostics`,
 * `test_failure_transition_canonicalizes_origin_subclass_for_wire`,
 * `test_failure_record_replay_reconstructs_identical_state`,
 * `test_failure_append_failure_leaves_every_state_coordinate_unchanged`,
 * `test_failure_requires_exact_diagnostic_carriers_before_append`,
 * `test_failure_requires_an_existing_unknown_in_flight_node`,
 * `test_failure_requires_initialized_nonterminal_unknown_state`,
 * `test_failure_replay_rejects_a_non_failure_authority`,
 * `test_failure_record_rejects_derived_terminal_unreachable_code`,
 * `test_competing_failure_plan_rejects_after_another_same_sequence_outcome` and
 * `test_failure_commit_uses_only_materialized_data_after_append`.
 *
 * Later (T4-5): `test_terminal_unreachable_latch_survives_pruning_its_dynamic_cause` (growth).
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure, OriginRef } from "../../src/diagnostics.ts";
import { NodeRouted, NodeSkipped, NodeSucceeded, type RunEvent, RunFailed, RunSucceeded } from "../../src/events.ts";
import { ELSE_HANDLE, type Node } from "../../src/nodes/base.ts";
import { type OutputContract, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import {
  encodePool,
  MemoryRunLog,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  type RecordEnvelope,
  type RunRecord,
  RunStartedRecord,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  NodeDataError,
  type PreparedTransition,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareNodeSucceeded,
  prepareRecord,
  type RunState,
} from "../../src/runtime/state.ts";
import { Type, TypeCheckError, type Value, ValueKind } from "../../src/typesys/values.ts";
import {
  commitStartSuccess,
  expectUnchanged,
  FailingLog,
  fanoutFlow,
  FuncNode,
  initializedAndStarted,
  reducerSnapshot,
  replayOutcomeLog,
  roots,
  routeFlow,
  routeInFlight,
  RUN_ID,
  snapshot,
  terminalSkipRouteFlow,
} from "./run_state_fixtures.ts";

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
  return CompiledFlow.fromParts(
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
  return CompiledFlow.fromParts(
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
  return CompiledFlow.fromParts(
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
  return CompiledFlow.fromParts(
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

/** Return `START -> work -> END` where `work` declares a partial-record output, `work` in flight. */
function partialRecordWorking(): { state: RunState; log: MemoryRunLog } {
  const work = func("work", {
    outputContract: new PartialRecordContract([
      ["items", new ValueContract(Type.scalar(ValueKind.LIST_INTEGER))],
      ["score", new ValueContract(Type.scalar(ValueKind.NUMBER))],
      ["note", null],
    ]),
  });
  const flow = CompiledFlow.fromParts(
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
  return CompiledFlow.fromParts(
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
  commitPrepared(log, state, prepareNodeSucceeded(state, END_ID, { terminal: ["stable"] }));
  return { state, log };
}

/** Return `START -> END`. */
function startEndFlow(): CompiledFlow {
  return CompiledFlow.fromParts(
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

  const prepared = prepareNodeSucceeded(state, START_ID, { items: ["alpha", "beta"] });

  expect(prepared.record).toStrictEqual(
    new NodeSucceededRecord(START_ID, START_ID, { kind: "object", value: { items: ["alpha", "beta"] } }),
  );
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
  expect(state.nextSequence).toBe(3);
  expect(log.records().map((envelope) => envelope.record.constructor)).toEqual([
    RunStartedRecord,
    NodeStartedRecord,
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
  const record = new NodeSucceededRecord(START_ID, START_ID, {
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

  commitPrepared(log, state, prepareNodeSucceeded(state, "first", { done: "first" }));

  expect([...state.inFlight]).toEqual(["second"]);
  expect(state.ready).toEqual(["existing", "after-a", "after-b"]);
  expect(state.edgeState.get("first-join")).toBe(NodeState.TAKEN);
  expect(state.edgeState.get("second-join")).toBe(NodeState.UNKNOWN);

  commitPrepared(log, state, prepareNodeSucceeded(state, "second", { done: "second" }));

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

    const prepared = prepareNodeSucceeded(state, "filler", ["alpha", "beta"], { commitAs: suppliedTarget });
    const observers = commitPrepared(log, state, prepared);

    expect((prepared.record as NodeSucceededRecord).commitTarget).toBe(expectedTarget);
    expect((prepared.record as NodeSucceededRecord).value).toEqual({ kind: "list[str]", value: ["alpha", "beta"] });
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
  const flow = CompiledFlow.fromParts(
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
    () => prepareNodeSucceeded(state, "filler", "three"),
    TypeCheckError,
    '"three" does not match declared type float',
  );
  const prepared = prepareNodeSucceeded(state, "filler", 3n);
  commitPrepared(log, state, prepared);

  expect((prepared.record as NodeSucceededRecord).commitTarget).toBe("filler");
  expect((prepared.record as NodeSucceededRecord).value).toEqual({ kind: "int", value: 3n });
  expect(state.pool.get("filler")).toBe(3n);
  expect(state.ready).toEqual(["after"]);
});

const MALFORMED_CONTRACT =
  "output contract must be an exact ValueContract, PartialRecordContract, or null, got Type";

test("a success refuses a malformed target contract as a host invariant", () => {
  const { state, log } = redirectReady();
  node(state.flow, "baked").outputContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;
  const before = snapshot(state);
  const sequence = log.nextSequence;

  expectError(() => prepareNodeSucceeded(state, "filler", ["valid"]), TypeError, MALFORMED_CONTRACT);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(sequence);
});

test("a success replay revalidates the target's output contract type", () => {
  const live = redirectReady();
  commitPrepared(live.log, live.state, prepareNodeSucceeded(live.state, "filler", ["valid"]));

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

    expectNodeData(() => prepareNodeSucceeded(state, "filler", output, { commitAs }), cls, message);

    expectUnchanged(state, before);
    expect(log.nextSequence).toBe(sequence);
  },
);

test("a success refuses a previously settled target edge before append", () => {
  const { state, log } = redirectReady();
  state.edgeState.set("baked-after", NodeState.TAKEN);
  const before = snapshot(state);
  const sequence = log.nextSequence;

  expectError(
    () => prepareNodeSucceeded(state, "filler", ["valid"]),
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
    () => prepareNodeSucceeded(state, "filler", ["valid"]),
    RangeError,
    "success commit target 'missing' does not exist in the flow",
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(4);
});

test("a success refuses a settled newly ready successor before append", () => {
  const { state, log } = redirectReady();
  state.nodeState.set("after-baked", NodeState.TAKEN);
  const before = snapshot(state);
  const sequence = log.nextSequence;

  expectError(
    () => prepareNodeSucceeded(state, "filler", ["valid"]),
    RangeError,
    "newly-ready successor 'after-baked' must have UNKNOWN disposition",
  );

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(sequence);
});

test("several edges to one successor enqueue it once", () => {
  const { state, log } = redirectReady(duplicateSuccessorFlow());

  commitPrepared(log, state, prepareNodeSucceeded(state, "filler", ["valid"]));

  expect(state.edgeState.get("baked-after-1")).toBe(NodeState.TAKEN);
  expect(state.edgeState.get("baked-after-2")).toBe(NodeState.TAKEN);
  expect(state.ready).toEqual(["after-baked"]);
});

test("a success keeps the sequential readiness of interleaved successors", () => {
  const { state, log } = initializedAndStarted(interleavedSuccessorFlow());
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "target"));

  commitPrepared(log, state, prepareNodeSucceeded(state, "target", { done: true }));

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

  const prepared = prepareNodeSucceeded(state, "target", { done: true });

  expect(scanned).toEqual(["after-a", "after-b"]);
  expect(successPlan(prepared).delta.newlyReadyIds).toEqual(["after-a", "after-b"]);
});

test.each([
  ["missing", "NodeSucceeded node 'missing' does not exist in the flow"],
  ["first", "NodeSucceeded node 'first' is not in-flight"],
] as const)("a success requires an existing unknown in-flight node: %s", (nodeId, message) => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const before = snapshot(state);

  expectError(() => prepareNodeSucceeded(state, nodeId, { value: 1n }), RangeError, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(2);
});

test("a success append failure leaves every state coordinate unchanged", () => {
  const { state, log: prior } = initializedAndStarted(fanoutFlow());
  const prepared = prepareNodeSucceeded(state, START_ID, { value: [1n, 2n] });
  const before = snapshot(state);
  const log = new FailingLog(RUN_ID, prior.records());

  expectError(() => commitPrepared(log, state, prepared), Error, "disk full");

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(prior.nextSequence);
});

test("a success delta keeps every reducer container's identity", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const prepared = prepareNodeSucceeded(state, START_ID, { value: "done" });
  const before = roots(state);

  commitPrepared(log, state, prepared);

  roots(state).forEach((root, index) => expect(root).toBe(before[index]));
  expect(state.ready).toEqual(["first", "second"]);
});

test("a success plan holds only an immutable sparse delta", () => {
  const { state } = initializedAndStarted(fanoutFlow());
  const delta = successPlan(prepareNodeSucceeded(state, START_ID, { value: "done" })).delta;

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
  const prepared = prepareNodeSucceeded(state, START_ID, { value: "done" });
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
  ["incoming", `success edges ["baked-after"] are absent from successor 'after-baked' incoming adjacency`],
] as const)("a success refuses a broken touched topology: %s", (broken, message) => {
  const { state, log } = redirectReady();
  if (broken === "edge-state") state.edgeState.delete("baked-after");
  else if (broken === "successor") state.nodeState.delete("after-baked");
  else state.flow.incomingEdges.set("after-baked", []);

  expectError(() => prepareNodeSucceeded(state, "filler", ["valid"]), RangeError, message);

  expect(log.nextSequence).toBe(4);
});

test("a success's canonical copies isolate the input, the record, the pool and the observers", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const raw = { nested: [["original"]] };
  const expected = { nested: [["original"]] };

  const prepared = prepareNodeSucceeded(state, START_ID, raw);
  raw.nested[0]!.push("raw mutation");
  ((prepared.observers[0] as NodeSucceeded).output as typeof raw).nested[0]!.push("prepared mutation");

  const observers = commitPrepared(log, state, prepared);
  const successRecord = log.records().at(-1)!.record;
  expect(successRecord).toBeInstanceOf(NodeSucceededRecord);
  const recorded = (successRecord as NodeSucceededRecord).value;
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
  const first = prepareNodeSucceeded(state, START_ID, { value: "done" });
  const second = prepareNodeSucceeded(state, START_ID, { value: "done" });

  commitPrepared(log, state, first);
  const sequence = log.nextSequence;
  expectError(() => commitPrepared(log, state, second), RangeError, "run-state sequence mismatch: prepared 2, state 3");

  expect(log.nextSequence).toBe(sequence);
  expect(state.ready).toEqual(["first", "second"]);
});

test("a success record replay reconstructs an identical state", () => {
  const live = initializedAndStarted(fanoutFlow());
  commitPrepared(live.log, live.state, prepareNodeSucceeded(live.state, START_ID, { items: ["alpha", "beta"] }));

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
    () => prepareNodeSucceeded(state, "work", output),
    TypeCheckError,
    "NaN is not a yian value",
  );

  expect((caught.original as TypeCheckError).path).toEqual(path);
  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(4);
});

test.each<[string, Value]>([
  ["Infinity", { items: [1n], score: Number.POSITIVE_INFINITY, note: { low: Number.NEGATIVE_INFINITY } }],
  ["a long int", { items: [1n, 10n ** 5000n], score: 1.5, note: { deep: 10n ** 5000n } }],
])("%s in a partial-record output is accepted", (_name, output) => {
  const { state, log } = partialRecordWorking();

  commitPrepared(log, state, prepareNodeSucceeded(state, "work", output));

  expect(state.pool.get("work")).toEqual(output);
  expect(state.ready).toEqual([END_ID]);
});

test("a whole float output stays a float", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  commitStartSuccess(state, log);
  commitPrepared(log, state, prepareNodeStarted(state, "first"));

  const prepared = prepareNodeSucceeded(state, "first", 2);
  commitPrepared(log, state, prepared);

  expect((prepared.record as NodeSucceededRecord).value).toEqual({ kind: "float", value: 2 });
  expect(state.pool.getValue("first")!.kind).toBe("float");
  expect(state.pool.get("first")).toBe(2);
});

test("the END output waits for full quiescence", () => {
  const { state, log } = endCommittedWithSiblingInFlight();

  expect(state.pool.get(END_ID)).toEqual({ terminal: ["stable"] });
  expect(state.result).toBeNull();
  expect(state.ready).toEqual(["queued"]);
  expect([...state.inFlight]).toEqual(["sibling"]);

  let observers = commitPrepared(log, state, prepareNodeSucceeded(state, "sibling", { done: true }));

  expect(observers).toStrictEqual([new NodeSucceeded({ nodeId: "sibling", output: { done: true } })]);
  expect(state.result).toBeNull();
  expect(state.ready).toEqual(["queued"]);
  expect([...state.inFlight]).toEqual([]);

  commitPrepared(log, state, prepareNodeStarted(state, "queued"));
  observers = commitPrepared(log, state, prepareNodeSucceeded(state, "queued", { done: "last" }));

  expect(observers).toStrictEqual([
    new NodeSucceeded({ nodeId: "queued", output: { done: "last" } }),
    new RunSucceeded({ output: { terminal: ["stable"] } }),
  ]);
  expect(state.result).toStrictEqual(new RunSucceeded({ output: { terminal: ["stable"] } }));
  expect(state.ready).toEqual([]);
  expect([...state.inFlight]).toEqual([]);
});

test("a newly ready successor prevents quiescence", () => {
  const flow = CompiledFlow.fromParts(
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
  commitPrepared(log, state, prepareNodeSucceeded(state, END_ID, "done"));

  const observers = commitPrepared(log, state, prepareNodeSucceeded(state, "worker", "worker-done"));

  expect(observers).toStrictEqual([new NodeSucceeded({ nodeId: "worker", output: "worker-done" })]);
  expect(state.result).toBeNull();
  expect(state.ready).toEqual(["after"]);
  expect([...state.inFlight]).toEqual([]);
});

test("a typed null END membership succeeds at quiescence", () => {
  const { state, log } = endOnlyInFlight();

  const prepared = prepareNodeSucceeded(state, END_ID, null);

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
  const expected = commitPrepared(live.log, live.state, prepareNodeSucceeded(live.state, END_ID, { value: [1n, 2n] }));

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
  const prepared = prepareNodeSucceeded(state, END_ID, { value: [1n, 2n] });
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
  const prepared = prepareNodeSucceeded(state, END_ID, raw);
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
  const prepared = prepareNodeSucceeded(state, END_ID, { value: [1n, 2n] });
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
  expect(log.nextSequence).toBe(4);
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
    NodeStartedRecord,
    NodeSucceededRecord,
    NodeStartedRecord,
    NodeRoutedRecord,
  ]);
});

test.each([
  [ELSE_HANDLE, ELSE_HANDLE],
  ["go", "go"],
])("a route takes every edge matching the else or a named handle: %s", (selected, matching) => {
  const flow = CompiledFlow.fromParts(
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
      new Edge("match-left", "router", "left", { sourceHandle: matching }),
      new Edge("match-right", "router", "right", { sourceHandle: matching }),
      new Edge("other", "router", "dead", { sourceHandle: "other" }),
    ],
  );
  const { state, log } = routeInFlight(flow);

  commitPrepared(log, state, prepareNodeRouted(state, "router", selected));

  expect(state.edgeState.get("match-left")).toBe(NodeState.TAKEN);
  expect(state.edgeState.get("match-right")).toBe(NodeState.TAKEN);
  expect(state.edgeState.get("other")).toBe(NodeState.SKIPPED);
  expect(state.nodeState.get("dead")).toBe(NodeState.SKIPPED);
  expect(state.ready).toEqual(["left", "right"]);
});

test("a route keeps DFS observers and the existing FIFO readiness", () => {
  const ids = ["router", "existing", "dead-a", "ready-a", "chosen", "dead-b", "ready-b"];
  const flow = CompiledFlow.fromParts(
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
  ["missing", "yes", "NodeRouted node 'missing' does not exist in the flow"],
  ["chosen", "yes", "NodeRouted node 'chosen' is not in-flight"],
] as const)("a route refuses an unknown router or frontier: %s", (nodeId, handle, message) => {
  const { state, log } = routeInFlight();
  const before = snapshot(state);

  expectError(() => prepareNodeRouted(state, nodeId, handle), RangeError, message);

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(4);
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
  expect(log.nextSequence).toBe(4);
});

test.each([NodeState.TAKEN, NodeState.SKIPPED])("a route refuses a previously settled outgoing edge: %s", (settled) => {
  const { state, log } = routeInFlight();
  state.edgeState.set("route-dead", settled);
  const before = snapshot(state);

  expectError(() => prepareNodeRouted(state, "router", "yes"), RangeError, "route edge 'route-dead' must be UNKNOWN");

  expectUnchanged(state, before);
  expect(log.nextSequence).toBe(4);
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
  expect(log.nextSequence).toBe(4);
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
  expectError(() => commitPrepared(log, state, second), RangeError, "run-state sequence mismatch: prepared 4, state 5");

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
  const poolBefore = encodePool(state.pool);
  const unexpected = (): never => {
    throw new Error("route commit repeated preparation work");
  };
  state.flow.incoming = unexpected;
  state.flow.outgoing = unexpected;

  const observers = commitPrepared(prior, state, prepared);

  expect(observers).toStrictEqual(routed("router", "yes", "dead-parent", "dead-child"));
  expect(encodePool(state.pool)).toEqual(poolBefore);
  expect(state.ready).toEqual(["chosen"]);
});

test("a route skip flood is iterative for a deep chain", () => {
  const depth = 1500;
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
  const { state, log } = routeInFlight(CompiledFlow.fromParts(nodes, edges));

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
  const flow = CompiledFlow.fromParts(
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
  const flow = CompiledFlow.fromParts(
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

/** Return the derived failure of a route from `router` that first skips END's `zeta` group. */
function zetaUnreachable(trace: readonly OriginRef[] = []): Failure {
  return new Failure({
    code: "graph.terminal_unreachable",
    message: "terminal output 'zeta' skipped",
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
  expect(records.length).toBe(5);
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

  const final = prepareNodeSucceeded(state, "unrelated", { done: true });
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
  expect(log.records().slice(-2).map((envelope) => envelope.record.constructor)).toEqual([
    NodeRoutedRecord,
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

const ZETA = "terminal output 'zeta' skipped";

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
  const before = snapshot(state);

  expectError(
    () => prepareNodeSucceeded(state, "unrelated", { done: true }),
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
    prepareNodeSucceeded(live.state, "unrelated", { done: true }),
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
