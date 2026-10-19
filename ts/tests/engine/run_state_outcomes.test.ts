/**
 * Transactional node-outcome tests for the run-state reducer: a port of
 * `tests/engine/test_run_state_outcomes.py`.
 *
 * Ported: `test_success_commits_canonical_value_before_fifo_successors`,
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
 * `test_competing_success_plans_reject_the_second_sequence_before_append` and
 * `test_success_record_replay_reconstructs_identical_state`. Python's `ValueError` is a
 * `RangeError` here, and a type mismatch the value layer's `TypeCheckError`; each rejected row
 * names its one class and full message, where Python matches a part of the message. A rejection
 * Python raises as `_NodeDataError` is a `NodeDataError` whose `original` is the `RangeError` or
 * `TypeCheckError`. `RUN_ID` and `FINGERPRINT` are the shared fixture's; the failing log is the
 * shared `FailingLog` (its error is `disk full`, Python's `append failed`); tuple outputs are arrays
 * and ints `bigint`s. Each test pins `state.ready` after its commits.
 *
 * Changed:
 * - `test_success_validates_ordinary_target_against_output_contract_type`: a `float` slot keeps
 *   the int `3n` an int (the value layer's rule), so the test also shows a text output refused
 *   against the target's `float` contract.
 * - `test_success_plan_contains_only_an_immutable_sparse_delta`: the value is canonical text, not
 *   bytes; assigning a delta field is a `TypeError`, and the id arrays are frozen.
 * - `test_success_commit_uses_only_the_prevalidated_delta`: only the instance patches, throwing
 *   `flow.incoming` and `flow.outgoing`.
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
 * Not ported (ESM): the module patches of `test_success_commit_uses_only_the_prevalidated_delta`
 * (`_validate_success_value`, `_prepare_success_delta`, `canonical_typed`, `encode_pool`,
 * `decode_pool`).
 *
 * Later (T4-4 task 3b): `test_end_output_waits_for_full_quiescence`,
 * `test_newly_ready_successor_prevents_quiescence`,
 * `test_typed_none_end_membership_succeeds_at_quiescence`,
 * `test_terminal_success_values_are_mutation_isolated`, and the `success` rows of
 * `test_terminal_derivation_replays_from_causal_records_only`,
 * `test_terminal_append_failure_leaves_result_frontier_and_pool_unchanged` and
 * `test_terminal_commit_performs_no_materialization_after_append`.
 *
 * Later (T4-4 PR 2), routes and failures: `test_route_derives_skip_closure_without_skip_record`,
 * `test_route_takes_every_edge_matching_else_or_named_handle`,
 * `test_route_preserves_dfs_observers_and_existing_fifo_readiness`,
 * `test_route_skip_join_matches_control_data_ordering_and_wait_semantics`,
 * `test_route_rejects_unknown_router_frontier_or_handle`,
 * `test_route_rejects_a_previously_settled_outgoing_edge`,
 * `test_route_rejects_malformed_source_handle_as_a_host_invariant`,
 * `test_route_requires_exact_nonempty_strings`,
 * `test_route_append_failure_leaves_every_state_coordinate_unchanged`,
 * `test_route_delta_preserves_all_reducer_container_identities`,
 * `test_competing_route_plans_reject_the_second_sequence_before_append`,
 * `test_route_observers_are_fresh_and_isolated_from_prepared_mutation`,
 * `test_route_record_replay_reconstructs_identical_state`,
 * `test_route_commit_uses_only_materialized_delta_after_append`,
 * `test_route_scans_each_touched_head_incoming_adjacency_once`,
 * `test_route_skip_flood_is_iterative_for_a_deep_chain`, `test_late_sibling_failure_wins`,
 * `test_route_terminal_skip_uses_first_required_end_group_and_no_extra_record`,
 * `test_route_latches_causal_terminal_failure_until_sibling_quiescence`,
 * `test_causal_route_canonicalizes_origin_subclass_for_wire`,
 * `test_explicit_failure_wins_after_terminal_unreachable_is_latched`,
 * `test_terminal_unreachable_latch_is_installed_only_after_route_append`,
 * `test_causal_route_replay_requires_its_persisted_terminal_failure`,
 * `test_noncausal_route_replay_rejects_a_terminal_failure`,
 * `test_causal_route_replay_rejects_mismatched_terminal_failure`,
 * `test_quiescence_without_retained_causal_failure_is_a_graph_fault`,
 * `test_terminal_unreachable_latch_replays_before_and_at_quiescence`,
 * `test_terminal_skip_diagnosis_scans_only_end_beyond_touched_route_joins`, the `route` rows of
 * the three terminal tests above, `test_failure_is_terminal_and_preserves_diagnostics`,
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
import { NodeSucceeded } from "../../src/events.ts";
import type { Node } from "../../src/nodes/base.ts";
import { type OutputContract, PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import {
  MemoryRunLog,
  NodeStartedRecord,
  NodeSucceededRecord,
  RunStartedRecord,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  NodeDataError,
  type PreparedTransition,
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
  FuncNode,
  initializedAndStarted,
  reducerSnapshot,
  replayOutcomeLog,
  roots,
  RUN_ID,
  snapshot,
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

/** Return the success delta of `prepared`'s plan; throw unless the plan is a success plan. */
function successDelta(prepared: PreparedTransition) {
  const plan = prepared._plan;
  if (plan.kind !== "success") throw new Error(`expected a success plan, found ${plan.kind}`);
  return plan.delta;
}

/** Return the node `nodeId` of `flow`. */
function node(flow: CompiledFlow, nodeId: string): Node {
  return flow.nodes.get(nodeId)!;
}

/** A pass-through node. */
function func(nodeId: string, options: { outputContract?: OutputContract | null } = {}): FuncNode {
  return new FuncNode(nodeId, (inputs) => inputs, options);
}

/** Return a complete graph whose START success makes two nodes ready in order. */
function fanoutFlow(): CompiledFlow {
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["first", func("first")],
      ["second", func("second")],
      [END_ID, EndNode.value(END_ID, { outputName: null })],
    ]),
    [new Edge("start-first", START_ID, "first"), new Edge("start-second", START_ID, "second")],
  );
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
  expect(successDelta(prepared).newlyReadyIds).toEqual(["after-a", "after-b"]);
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
  const delta = successDelta(prepareNodeSucceeded(state, START_ID, { value: "done" }));

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
});

test("competing success plans refuse the second sequence before append", () => {
  const { state, log } = initializedAndStarted(fanoutFlow());
  const first = prepareNodeSucceeded(state, START_ID, { value: "done" });
  const second = prepareNodeSucceeded(state, START_ID, { value: "done" });

  commitPrepared(log, state, first);
  const sequence = log.nextSequence;
  expectError(() => commitPrepared(log, state, second), RangeError, "run-state sequence mismatch: prepared 2, state 3");

  expect(log.nextSequence).toBe(sequence);
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
