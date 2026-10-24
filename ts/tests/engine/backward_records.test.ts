/**
 * Backward proposal and record contracts: a port of `tests/engine/test_backward_records.py`.
 * Ported: `test_memory_proposal_owns_detached_mutations_in_supplied_order`,
 * `test_backward_record_constructors_detach_values_without_reducer_inference`,
 * `test_commit_attempt_ids_are_required_keyword_only_positive_integers`,
 * `test_backward_records_reject_subclasses_of_exact_nested_carriers` (the 4 record rows; the
 * `MutationSubclass` row is its own test) and
 * `test_backward_started_rejects_nonordinary_feedback_before_recording` (3 rows). Messages and
 * classes are pinned where Python accepts either class.
 *
 * Changed: a `MemoryProposal` keeps the frozen mutations it was given, where Python copies them:
 * the test asserts the same mutation (`is not` becomes `toBe`), and a reader's payload edit throws
 * `TypeError` where Python changes a private copy. A mutation batch is an array, so the rejected
 * container is a `Set` (Python passes a list); assigning a field throws `TypeError` where Python
 * raises `FrozenInstanceError`, and Python's `ValueError` is `RangeError`.
 * - The backward records hold their frozen carriers by reference (`toBe`), where Python copies
 *   them; the started record's feedback is a deep-frozen copy read without copying, so a reader's
 *   edit throws `TypeError`. Array arguments are copied and frozen; the rejected containers are
 *   `Set`s.
 * - Attempt ids are safe-integer `number`s: the invalid ids are `0, -1, true, "1", 1n` (`1n`
 *   replaces Python's `1.0`); the commit records take `attemptId` as their last positional
 *   argument (TS has no keyword-only arguments), so a missing one throws `TypeError` and
 *   Python's positional row is dropped; the invalid commit ids are `null, 0, -1, true, "2", 2n`.
 * - The non-ordinary feedback rows are `1n`, `true` and `[0n]`; `{"field": ()}` and
 *   `{1: "Review."}` have no TS counterpart (no tuples; object keys are always text).
 *
 * Not ported: the rows that forge carriers with `object.__setattr__`
 * (`test_backward_codec_revalidates_nested_carriers_forged_after_construction`,
 * `test_backward_codec_rejects_forged_dictionary_subclasses_before_copying`): frozen TS carriers
 * refuse the forgery.
 *
 * Ported, the wire codec: `test_backward_record_wire_round_trip_preserves_all_channels`,
 * `test_commit_wire_preserves_supplied_attempt_ids`,
 * `test_failed_backward_wire_keeps_causes_completed_feedback_and_policy_dispositions` (12 rows),
 * `test_wire_preserves_transfer_coordinates_unreached_inventory_and_empty_defaults`,
 * `test_backward_record_feedback_keeps_the_existing_wider_depth_boundary` (140 levels),
 * `test_memory_and_mutation_proposal_order_survives_wire_round_trip`,
 * `test_backward_wire_nested_records_have_closed_required_fields` (42 rows),
 * `test_backward_wire_rejects_malformed_nested_values` (17 rows),
 * `test_forward_cause_wire_rejects_hybrid_and_extra_origin_fields`,
 * `test_backward_failure_cause_round_trips_optional_execution_attribution` (3 rows),
 * `test_backward_failure_cause_wire_rejects_nonstring_execution_attribution` (4 rows) and
 * `test_node_failure_wire_keeps_required_execution_attribution` (4 rows). Each rejection is
 * pinned with its class and message where Python accepts either class or asserts only
 * `ValueError`: a field of the wrong JSON type is a `TypeError`, a missing or extra field, an
 * empty string or a value a carrier refuses is a `RangeError`, and a carrier's own message
 * applies.
 *
 * Changed (wire): a wire int is a `bigint`, so the malformed rows write `1n` where Python writes
 * `1`; `true` isn't an int, as in Python. A cause `node_id` that isn't text (`1n`, `true`, `[]`,
 * `{}`) and a node failure's `null` `node_id` are `TypeError`s where Python raises `ValueError`.
 *
 * Changed (log): `test_backward_records_isolate_payload_views_before_and_after_log_round_trip`
 * with the memory run log: the records, the envelope's payload and the replayed record are
 * frozen, so each edit throws `TypeError` where Python edits a private copy.
 *
 * TS-only: `sameBackwardAttribution`, `sameAttributedBackwardResult`, `sameBackwardFailure` and
 * `sameBackwardOutcome` are true for separately built equal carriers (including two backward
 * failures built from one `Failure` cause, whose retained causes are different objects) and false
 * for each differing field, with a both-null and a one-null row for each nullable field (an
 * attribution's nullable fields are null together, on a root seed; an outcome's failure is null
 * exactly on success).
 *
 * Ported, the trace delta:
 * `test_postterminal_trace_delta_validation_covers_the_closed_field_inventory`. Changed: it reads
 * the own keys of a constructed `TraceDelta`, in camelCase, where Python reads the dataclass
 * fields.
 *
 * Ported, after rewrite: `test_backward_state_revalidates_exact_memory_order` builds
 * `RuntimeBackwardState` directly (Python replaces an engine's state): the unsorted order is a
 * `RangeError`, and the `list_order` row refuses a memory order that isn't an array (a `Set`),
 * since the order is an array in TS. Not ported: the `string_subclass` row (TS text has no
 * subclasses).
 *
 * Ported, the run state's backward records, on a local `retainedTerminal()` (Python's
 * `_retained_backward` without `compute_backward`: `learnableStarted`, with its `topic` and
 * memory `keys` options, and reducer-direct successes of START, `work` and END):
 * `test_backward_started_requires_initialized_learned_terminal_authority` (4 rows),
 * `test_backward_prepared_plans_reject_stale_forged_or_mutated_authority_before_append` (6 rows),
 * `test_postterminal_trace_delta_rejects_equal_but_noncanonical_fields_before_append` (9 of 16
 * rows: `initialized`, `started`, `occurrence`, `writer`, `root_start_source`, `root_writer`,
 * `causal_route_source`, `terminal`, `handler`),
 * `test_backward_append_failure_does_not_install_attempt_or_irreversible_request` (the `started`
 * row), `test_interrupted_attempt_requires_explicit_finish_before_a_fresh_start` (the
 * `ready=False` row),
 * `test_failed_backward_computation_has_no_completed_feedback_and_keeps_unrequested_learning`,
 * `test_failed_computation_finished_rejects_foreign_context_or_fabricated_results` (the `source`,
 * `node`, `scope` and `run` rows) and
 * `test_computation_failure_requires_effective_feedback_and_a_driver_failure_code` (3 rows).
 *
 * Changed (run state): each refusal pins its `RangeError` and message where Python accepts
 * `TypeError` or `ValueError`. The gate test's `learn_false` terminal is a non-learnable
 * `initializedAndStarted` run driven to END, and `missing_terminal` clears the trace's terminal
 * on a reducer-built terminal. A forged plan is a frozen copy in a new `PreparedTransition`, and
 * the `payload` row pairs a changed record with the original plan (the record is frozen, so it
 * can't be changed in place). The log's `append` is patched on the instance, its error an
 * `Error`, Python's `OSError`.
 *
 * TS-only (run state): `retainedTerminal()` reaches a succeeded result and a succeeded trace
 * terminal at sequence 7; attempt state replaced after preparation (trace and sequence unchanged)
 * reaches the commit's `prepared backward transition has stale attempt authority`, which prepare
 * and commit alone can't reach, and the log is unchanged.
 *
 * Not ported (TS types): 7 rows of the post-terminal delta test, as forward_trace's frontier rows:
 * `sequence` (a `float`), `resumed` (`0`), `prune_ids`, `connections` and `boundaries` (tuple
 * subclasses), `consumed_connection` and `boundary_update` (`__eq__` aliases of `None`).
 *
 * Ported, the run state's backward records through `computeBackward`, on a local async
 * `retainedBackward()` (Python's `_retained_backward`: the engine runs `START -> work -> END`, `work`
 * a `RecordingBackwardNode` that sends its feedback back to `x`, then `computeBackward`) and
 * `proposalTable()` (Python's `_proposal_table`):
 * `test_backward_attempt_records_prepare_append_then_reduce_without_forward_effects`,
 * `test_backward_policy_batches_and_unfinished_prefixes_follow_accepted_layout_order`, the
 * `ready=True` row of `test_interrupted_attempt_requires_explicit_finish_before_a_fresh_start`,
 * `test_ready_rejects_inventory_local_contract_transfer_and_root_disagreement_before_append` (14
 * rows), `test_ready_proposals_require_the_complete_table_in_order` (4 rows),
 * `test_mutation_ids_may_repeat_across_different_recorded_memories`,
 * `test_learner_failed_finished_keeps_completed_feedback_and_all_unrequested_policies` (4 rows),
 * `test_known_commit_failure_stops_later_requests_and_admits_a_later_attempt` (4 rows),
 * `test_any_finished_attempt_admits_a_later_start_with_fresh_proposals` (3 rows),
 * `test_every_committing_attempt_uses_its_own_attempt_in_the_commit_id`,
 * `test_first_committing_attempt_uses_its_own_attempt_id_after_precommit_attempts`,
 * `test_started_feedback_shape_and_attempt_identity_validate_before_acceptance`,
 * `test_backward_commit_request_rejects_wrong_order_batch_identity_and_attempt` (5 rows),
 * `test_backward_commit_observation_requires_the_current_pending_request` (6 rows), the `ready`,
 * `request` and `finished` rows of
 * `test_backward_append_failure_does_not_install_attempt_or_irreversible_request`, the `feedback`
 * and `partial_inventory` rows of
 * `test_failed_computation_finished_rejects_foreign_context_or_fabricated_results`,
 * `test_successful_finished_requires_ready_and_every_enabled_policy_confirmation` (4 rows),
 * `test_explicit_interruption_keeps_unknown_request_and_ready_attribution_without_resuming` and
 * `test_disabled_memory_ready_and_finished_reject_fabricated_memory_material`.
 *
 * Changed (computeBackward): each refusal pins its `RangeError` and message, as above. The run
 * state keeps the log's canonical copy of each accepted record, so accepted records, outcomes and
 * results compare with `toEqual` (Python's `==`). A `topic` of `[1, 2]` is `[1n, 2n]`. The log's
 * `append` is patched on the instance, as above.
 * - `test_attributed_validation_reuses_recorded_transfers_without_callbacks_or_materialization`
 *   drops the `_materialize_context` and `_output_context` patches (TS can't patch a module's
 *   functions) and shadows the trace's `backwardHandlers` getter on the instance with one that
 *   throws.
 * - `test_finished_must_preserve_ready_results_and_exact_confirmed_policy_inventory` (6 rows): the
 *   `commit_id` row rebuilds the first policy outcome with its constructor, where Python uses
 *   `dataclasses.replace`.
 * - `test_interrupted_finished_must_exactly_project_the_prefix` (4 rows) rebuilds the outcome and,
 *   for the `message` row, its failure with their constructors, where Python uses
 *   `dataclasses.replace`; that failure is no longer the interruption, so, as in Python, the
 *   commit-failure rule refuses it.
 * - `test_backward_replay_matches_every_accepted_prefix_without_callbacks_or_context_formatting`
 *   drops the module patches, as above; `forward`, `backward` and `captureBackwardContext` throw
 *   on the nodes of a `copyGraph` copy of the run's flow, each prefix replays over a fresh
 *   `copyGraph` of that copy with `noGrowth`, and `sameRuntimeBackwardState` is Python's `==`.
 *
 * The reducer's own rules are tested TS-only in `runtime_backward.test.ts`.
 *
 * Ported, through `run.ts`: `test_historical_backward_prepares_recorded_schema_bytes` (7 rows).
 * Changed: Python reads recorded schema-9 lines; schema 13 is the only TS schema, so a live
 * learnable `runFlow` of the source test's flow (memory `test` served by `FakeMemory`) and its
 * `RunResult.backward` write the lines (the failed rows' commit throws a `MemoryOperationError`);
 * each prefix replays through `replayRunLog` from a `MemoryRunLog`, and the next line is prepared
 * and committed through the internal `HistoricalLogCursor`.
 */

import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { copyGraph, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { Failure, OriginRef } from "../../src/diagnostics.ts";
import { RunSucceeded } from "../../src/events.ts";
import { type FeedbackValue, SourceRef } from "../../src/expr/lineage.ts";
import {
  DeclSite,
  LearningFailure,
  LearningResult,
  MemoryMutation,
  MemoryOperationError,
  MemoryOperationFailure,
  MemorySnapshotEnvelope,
  MemorySpec,
  PolicyCommitOutcome,
} from "../../src/memory/contracts.ts";
import { BackwardResult } from "../../src/nodes/base.ts";
import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";
import { replayRunLog, runFlow } from "../../src/run.ts";
import { computeBackward } from "../../src/runtime/backward.ts";
import {
  AttributedBackwardResult,
  BackwardAttribution,
  BackwardFailure,
  BackwardOutcome,
  MemoryProposal,
  sameAttributedBackwardResult,
  sameBackwardAttribution,
  sameBackwardFailure,
  sameBackwardOutcome,
} from "../../src/runtime/contracts.ts";
import {
  memoryCommitId,
  RuntimeBackwardState,
  sameRuntimeBackwardState,
  validateAttributedBackward,
} from "../../src/runtime/learning.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  decodeRecord,
  encodeRecord,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  MemoryRunLog,
  NodeFailedRecord,
  NodeRoutedRecord,
  RecordEnvelope,
  type RunLog,
  type RunRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  HistoricalLogCursor,
  PreparedTransition,
  prepareNodeStarted,
  prepareRecord,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { TraceDelta, type TraceDeltaFields } from "../../src/runtime/trace.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { FakeMemory } from "./memory_fakes.ts";
import {
  commitStartSuccess,
  engine,
  forwardCapture,
  FuncNode,
  graph,
  initializedAndStarted,
  learnableStarted,
  noGrowth,
  prepareOutput,
  RecordingBackwardNode,
  untimed,
} from "./run_state_fixtures.ts";

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

test("memory proposal owns detached mutations in supplied order", () => {
  const payload = { values: [1n] };
  const mutations = [new MemoryMutation("second", payload), new MemoryMutation("first", {})];
  const proposal = new MemoryProposal("agent", mutations);
  payload.values.push(2n);
  expect(() => {
    (proposal.mutations[0]!.payload as { values: bigint[] }).values.length = 0;
  }).toThrow(TypeError);
  expect(proposal.mutations.map((mutation) => mutation.mutationId)).toEqual(["second", "first"]);
  expect(proposal.mutations[0]!.payload).toEqual({ values: [1n] });
  expect(proposal.mutations[0]).toBe(mutations[0]!);
  expect(proposal.memory).toBe("agent");
  expect(Object.isFrozen(proposal)).toBe(true);
  expect(Object.isFrozen(proposal.mutations)).toBe(true);
  expect(() => {
    (proposal as { mutations: readonly MemoryMutation[] }).mutations = [];
  }).toThrow(TypeError);
  expect(() => new MemoryProposal("agent", new Set(mutations) as never)).toThrow(TypeError);
  const repeated = [mutations[0]!, mutations[0]!];
  expectError(() => new MemoryProposal("agent", repeated), RangeError, "mutation IDs must be unique");
  expect(new MemoryProposal("agent", []).mutations).toEqual([]);
});

// test_backward_records_reject_subclasses_of_exact_nested_carriers (the MutationSubclass row)
test("a memory proposal rejects a MemoryMutation subclass", () => {
  class MutationSubclass extends MemoryMutation {}
  expect(() => new MemoryProposal("agent", [new MutationSubclass("m1", {})])).toThrow(TypeError);
});

/** Return the three-channel result the record tests use. */
function result(): BackwardResult {
  return new BackwardResult({
    inputFeedback: { topic: [[null, { uri: "Review." }]] },
    selfFeedback: ["Behavior."],
    variableFeedback: { tone: ["Concise."] },
  });
}

/** Return one invoked occurrence result with a root attribution. */
function attributed(): AttributedBackwardResult[] {
  return [
    new AttributedBackwardResult(
      new SourceRef("run", 1),
      ["Review."],
      [new BackwardAttribution(null, null, 0, null, [])],
      result(),
      true,
    ),
  ];
}

const ATTEMPT_INTEGER = "attempt_id must be an integer";
const ATTEMPT_POSITIVE = "attempt_id must be positive";
const INVALID_ATTEMPT_IDS: [unknown, typeof TypeError | typeof RangeError, string][] = [
  [0, RangeError, ATTEMPT_POSITIVE],
  [-1, RangeError, ATTEMPT_POSITIVE],
  [true, TypeError, ATTEMPT_INTEGER],
  ["1", TypeError, ATTEMPT_INTEGER],
  [1n, TypeError, ATTEMPT_INTEGER],
];

const INVALID_COMMIT_ATTEMPT_IDS: [unknown, typeof TypeError | typeof RangeError, string][] = [
  [null, TypeError, ATTEMPT_INTEGER],
  [0, RangeError, ATTEMPT_POSITIVE],
  [-1, RangeError, ATTEMPT_POSITIVE],
  [true, TypeError, ATTEMPT_INTEGER],
  ["2", TypeError, ATTEMPT_INTEGER],
  [2n, TypeError, ATTEMPT_INTEGER],
];

test("backward record constructors detach values without reducer inference", () => {
  const feedback = { parts: [null, "Review."] };
  const started = new BackwardStartedRecord(1, feedback);
  feedback.parts.length = 0;
  expect(() => (started.feedback as { parts: unknown[] }).parts.splice(0)).toThrow(TypeError);
  expect(started.feedback).toEqual({ parts: [null, "Review."] });

  const proposal = new MemoryProposal("agent", [new MemoryMutation("m1", { notes: ["one"] })]);
  const root = result();
  const occurrences = attributed();
  const proposals = [proposal];
  const ready = new BackwardReadyRecord(1, root, occurrences, proposals);
  occurrences.pop();
  proposals.pop();
  expect(() => (ready.rootResult.inputFeedback["topic"]![0] as unknown[]).splice(0)).toThrow(TypeError);
  expect(ready.rootResult).toBe(root);
  expect(ready.rootResult).toEqual(result());
  expect(ready.attributedResults).toEqual(attributed());
  expect(ready.proposals[0]).toBe(proposal);

  const outcome = new BackwardOutcome("succeeded", { feedback: result() });
  const finished = new BackwardFinishedRecord(1, outcome);
  expect(finished.attributedResults).toEqual([]);
  expect(finished.outcome).toBe(outcome);

  for (const record of [started, ready, finished]) {
    expect(() => {
      (record as { attemptId: number }).attemptId = 2;
    }).toThrow(TypeError);
  }
  for (const [invalid, cls, message] of INVALID_ATTEMPT_IDS) {
    expectError(() => new BackwardStartedRecord(invalid as never, null), cls, message);
  }
  expectError(
    () => new BackwardReadyRecord(1, result(), new Set(attributed()) as never, []),
    TypeError,
    "attributed_results must be an array",
  );
  expectError(
    () => new BackwardReadyRecord(1, result(), [], new Set([proposal]) as never),
    TypeError,
    "proposals must be an array",
  );
});

test("commit attempt ids are required positive integers", () => {
  const failure = new MemoryOperationFailure("backend", "failed", "Failed.", true);
  // A missing attempt id reads as `undefined`, so `build(undefined)` omits it.
  const carriers: ((attemptId: unknown) => { attemptId: number })[] = [
    (attemptId) => new MemoryCommitRequestedRecord("history", "commit-1", [], attemptId as number),
    (attemptId) => new MemoryCommitSucceededRecord("history", "commit-1", attemptId as number),
    (attemptId) => new MemoryCommitFailedRecord("history", "commit-1", failure, attemptId as number),
  ];
  for (const build of carriers) {
    expectError(() => build(undefined), TypeError, ATTEMPT_INTEGER);
    expect(build(2).attemptId).toBe(2);
    for (const [invalid, cls, message] of INVALID_COMMIT_ATTEMPT_IDS) {
      expectError(() => build(invalid), cls, message);
    }
  }
});

class ResultSubclass extends BackwardResult {}
class AttributionSubclass extends AttributedBackwardResult {}
class ProposalSubclass extends MemoryProposal {}
class OutcomeSubclass extends BackwardOutcome {}

// test_backward_records_reject_subclasses_of_exact_nested_carriers (the record rows)
test.each<[string, () => unknown, string]>([
  [
    "a BackwardResult subclass",
    () => new BackwardReadyRecord(1, new ResultSubclass(), [], []),
    "result must be an exact BackwardResult",
  ],
  [
    "an AttributedBackwardResult subclass",
    () => {
      const item = attributed()[0]!;
      const forged = new AttributionSubclass(item.source, item.feedback, item.attribution, item.result, true);
      return new BackwardReadyRecord(1, result(), [forged], []);
    },
    "attributed_results require exact AttributedBackwardResult values",
  ],
  [
    "a MemoryProposal subclass",
    () => new BackwardReadyRecord(1, result(), [], [new ProposalSubclass("agent", [])]),
    "proposals require exact MemoryProposal values",
  ],
  [
    "a BackwardOutcome subclass",
    () => new BackwardFinishedRecord(1, new OutcomeSubclass("succeeded")),
    "outcome must be an exact BackwardOutcome",
  ],
])("backward records reject %s", (_name, build, message) => {
  expectError(build, TypeError, message);
});

test.each<[string, unknown, string]>([
  ["1n", 1n, "feedback must be a string, object, list, or None"],
  ["true", true, "feedback must be a string, object, list, or None"],
  ["[0n]", [0n], "feedback[0] must be a string, object, list, or None"],
])("backward started rejects non-ordinary feedback before recording: %s", (_name, feedback, message) => {
  expectError(() => new BackwardStartedRecord(1, feedback as never), TypeError, message);
});

/** Return the started, ready and finished records of the wire tests. */
function backwardRecords(): [BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord] {
  const proposal = new MemoryProposal("agent", [new MemoryMutation("m1", { notes: ["one"] })]);
  const learning = new LearningResult("succeeded", {
    policies: [new PolicyCommitOutcome("history", "succeeded", { commitId: "commit-1" })],
  });
  const outcome = new BackwardOutcome("succeeded", { feedback: result(), learning });
  return [
    new BackwardStartedRecord(1, [null, { uri: "Review." }]),
    new BackwardReadyRecord(1, result(), attributed(), [proposal]),
    new BackwardFinishedRecord(1, outcome, attributed()),
  ];
}

/** Encode `record` at seq 0 and decode it; the envelope and its line must survive. */
function roundTrip<T extends RunRecord>(record: T): T {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 0, record);
  const encoded = encodeRecord(envelope);
  const decoded = decodeRecord(encoded);
  expect(decoded).toEqual(envelope);
  expect(decoded.record.constructor).toBe(record.constructor);
  expect(encodeRecord(decoded)).toBe(encoded);
  return decoded.record as T;
}

/** The decoded JSON of `record`'s line at seq 0, ints as `bigint`s, as a mutable object. */
function wireOf(record: RunRecord): Record<string, any> {
  return parseJson(encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "run", 0, record)), "codec") as Record<string, any>;
}

/** Return the value at `path` inside `value`. */
function atPath(value: any, path: readonly (string | number)[]): any {
  for (const key of path) value = value[key];
  return value;
}

type CauseKind = "none" | "forward" | "memory";

const MEMORY_CAUSE = new MemoryOperationFailure("backend-stage", "backend.failed", "Retry later.", true);

/** Return a failed finished record with a cause of `causeKind`, a scope and four policy dispositions. */
function failedRecord(
  causeKind: CauseKind = "memory",
  options: { completed?: boolean; learningStatus?: "failed" | "incomplete" } = {},
): BackwardFinishedRecord {
  const { completed = true, learningStatus = "failed" } = options;
  const causes = {
    none: null,
    memory: MEMORY_CAUSE,
    forward: new Failure({
      code: "node.execute",
      message: "Failed.",
      nodeId: "agent",
      trace: [new OriginRef("agent.prompt")],
      causeType: "ValueError",
      traceback: "forward traceback",
    }),
  };
  const scope = new ExecutionScope(["each", "agent"], {
    flowPath: ["each"],
    occurrencePath: [new ExecutionPosition("map_item", ["each"], 2)],
  });
  const failure = new BackwardFailure("learn", "memory.learn", "Could not propose.", {
    scope,
    runId: "run",
    nodeId: "each/2/agent",
    nodeStartedSeq: 7,
    memory: "history",
    cause: causes[causeKind],
    causeType: "MemoryOperationError",
    traceback: "backward traceback",
  });
  const learning = new LearningResult(learningStatus, {
    failures: [
      new LearningFailure("history", "learn", MEMORY_CAUSE),
      new LearningFailure("store", "commit", MEMORY_CAUSE),
    ],
    policies: [
      new PolicyCommitOutcome("later", "unrequested"),
      new PolicyCommitOutcome("pending", "unknown", { commitId: "commit-unknown" }),
      new PolicyCommitOutcome("confirmed", "succeeded", { commitId: "commit-ok" }),
      new PolicyCommitOutcome("store", "failed", { commitId: "commit-failed", failure: MEMORY_CAUSE }),
    ],
  });
  const outcome = new BackwardOutcome("failed", { feedback: completed ? result() : null, failure, learning });
  return new BackwardFinishedRecord(2, outcome, completed ? attributed() : []);
}

test.each(backwardRecords().map((record) => [record.constructor.name, record] as const))(
  "a backward record's wire round trip preserves all channels: %s",
  (_name, record) => {
    expect(roundTrip(record)).toEqual(record);
  },
);

test("the commit wire preserves supplied attempt ids", () => {
  const failure = new MemoryOperationFailure("backend", "failed", "Failed.", false);
  for (const attemptId of [1, 3]) {
    for (const record of [
      new MemoryCommitRequestedRecord("history", "commit-1", [], attemptId),
      new MemoryCommitSucceededRecord("history", "commit-1", attemptId),
      new MemoryCommitFailedRecord("history", "commit-1", failure, attemptId),
    ]) {
      expect(roundTrip(record).attemptId).toBe(attemptId);
    }
  }
});

const FAILED_ROWS: [CauseKind, boolean, "failed" | "incomplete"][] = [];
for (const causeKind of ["none", "forward", "memory"] as const) {
  for (const completed of [false, true]) {
    for (const learningStatus of ["failed", "incomplete"] as const) {
      FAILED_ROWS.push([causeKind, completed, learningStatus]);
    }
  }
}

test.each(FAILED_ROWS)(
  "a failed backward's wire keeps causes, completed feedback and policy dispositions: %s, %p, %s",
  (causeKind, completed, learningStatus) => {
    const record = failedRecord(causeKind, { completed, learningStatus });
    const decoded = roundTrip(record);
    expect(decoded).toEqual(record);
    expect(decoded.outcome.feedback !== null).toBe(completed);
    expect(decoded.attributedResults.length > 0).toBe(completed);
    expect(decoded.outcome.failure!.scope!.occurrencePath[0]!.index).toBe(2);
    expect(decoded.outcome.learning!.policies.map((policy) => policy.status)).toEqual([
      "unrequested",
      "unknown",
      "succeeded",
      "failed",
    ]);
    expect(decoded.outcome.learning!.failures[0]!.failure.retryable).toBe(true);
    const cause = decoded.outcome.failure!.cause;
    expect(cause === null ? null : cause.constructor).toBe(
      { none: null, forward: Failure, memory: MemoryOperationFailure }[causeKind],
    );
  },
);

test("the wire preserves transfer coordinates, the unreached inventory and empty defaults", () => {
  const transfer = new BackwardAttribution(new SourceRef("run", 9), "topic", 3, 2, [0, 2]);
  const occurrences = [
    new AttributedBackwardResult(new SourceRef("run", 7), [""], [transfer], result(), true),
    new AttributedBackwardResult(new SourceRef("run", 5), [], [], new BackwardResult(), false),
  ];
  const ready = roundTrip(new BackwardReadyRecord(1, new BackwardResult(), occurrences, []));
  expect(ready.attributedResults).toEqual(occurrences);
  expect(ready.attributedResults[0]!.attribution[0]).toEqual(transfer);
  expect(roundTrip(new BackwardFinishedRecord(1, new BackwardOutcome("succeeded"))).attributedResults).toEqual([]);
  for (const feedback of [null, "", [], {}, [null, { empty: null }]]) {
    expect(roundTrip(new BackwardStartedRecord(1, feedback)).feedback).toEqual(feedback);
  }
});

test("a backward record's feedback keeps the existing wider depth boundary", () => {
  let feedback: unknown = "Review.";
  for (let level = 0; level < 140; level += 1) feedback = [feedback];
  const deep = feedback as string[];
  const deepResult = new BackwardResult({ inputFeedback: { deep: [deep] } });
  const occurrences = [
    new AttributedBackwardResult(
      new SourceRef("run", 1),
      [deep],
      [new BackwardAttribution(null, null, 0, null, [])],
      deepResult,
      true,
    ),
  ];
  for (const record of [
    new BackwardStartedRecord(1, deep),
    new BackwardReadyRecord(1, deepResult, occurrences, []),
    new BackwardFinishedRecord(1, new BackwardOutcome("succeeded", { feedback: deepResult }), occurrences),
  ]) {
    expect(roundTrip(record)).toEqual(record);
  }
});

test("backward records isolate payload views before and after a log round trip", () => {
  const ready = backwardRecords()[1];
  const log = new MemoryRunLog("run");
  const accepted = log.append(ready);
  const edits: (() => unknown)[] = [
    () => (ready.rootResult.inputFeedback["topic"]![0] as any)[1].uri = "changed",
    () => (ready.proposals[0]!.mutations[0]!.payload as any).notes.length = 0,
    () => ((accepted.record as BackwardReadyRecord).attributedResults[0]!.result.variableFeedback as any).tone = [],
    () => (accepted.payload as any).proposals[0].mutations[0].payload.notes.length = 0,
  ];
  const replayed = [...log][0]!.record as BackwardReadyRecord;
  edits.push(
    () => ((replayed.rootResult.inputFeedback["topic"]![0] as any).length = 0),
    () => (replayed.proposals[0]!.mutations[0]!.payload as any).notes.push("changed"),
  );
  for (const edit of edits) expect(edit).toThrow(TypeError);
  expect([...log][0]!.record).toEqual(backwardRecords()[1]);
});

test("memory and mutation proposal order survives the wire round trip", () => {
  const proposals = ["later", "earlier"].map(
    (name) =>
      new MemoryProposal(
        name,
        ["second", "first"].map((mutationId) => new MemoryMutation(mutationId, { step: mutationId })),
      ),
  );
  const decoded = roundTrip(new BackwardReadyRecord(1, result(), [], proposals));
  expect(decoded.proposals.map((proposal) => proposal.memory)).toEqual(["later", "earlier"]);
  expect(decoded.proposals[0]!.mutations.map((mutation) => mutation.mutationId)).toEqual(["second", "first"]);
});

type RecordKind = "started" | "ready" | "finished";

/** Return the wire-test record of `kind`: the started and ready records, or the failed finished one. */
function wireRecord(kind: RecordKind): RunRecord {
  return kind === "started" ? backwardRecords()[0] : kind === "ready" ? backwardRecords()[1] : failedRecord();
}

const READY = "backward_ready payload";
const OCCURRENCE = `${READY} attributed_results item`;
const OUTCOME = "backward_finished payload outcome";
const CLOSED_OBJECTS: [RecordKind, (string | number)[], string][] = [
  ["started", [], "backward_started payload"],
  ["ready", [], READY],
  ["finished", [], "backward_finished payload"],
  ["ready", ["root_result"], `${READY} root_result`],
  ["ready", ["attributed_results", 0], OCCURRENCE],
  ["ready", ["attributed_results", 0, "source"], `${OCCURRENCE} source`],
  ["ready", ["attributed_results", 0, "attribution", 0], `${OCCURRENCE} attribution item`],
  ["ready", ["attributed_results", 0, "result"], `${OCCURRENCE} result`],
  ["ready", ["proposals", 0], `${READY} proposals item`],
  ["ready", ["proposals", 0, "mutations", 0], `${READY} proposals item mutations item`],
  ["finished", ["outcome"], OUTCOME],
  ["finished", ["outcome", "feedback"], `${OUTCOME} feedback`],
  ["finished", ["outcome", "failure"], `${OUTCOME} failure`],
  ["finished", ["outcome", "failure", "scope"], `${OUTCOME} failure scope`],
  ["finished", ["outcome", "failure", "scope", "occurrence_path", 0], `${OUTCOME} failure scope occurrence_path item`],
  ["finished", ["outcome", "failure", "cause"], `${OUTCOME} failure cause`],
  ["finished", ["outcome", "learning"], `${OUTCOME} learning`],
  ["finished", ["outcome", "learning", "failures", 0], `${OUTCOME} learning failures item`],
  ["finished", ["outcome", "learning", "failures", 0, "failure"], `${OUTCOME} learning failures item failure`],
  ["finished", ["outcome", "learning", "policies", 0], `${OUTCOME} learning policies item`],
  ["finished", ["outcome", "learning", "policies", 3, "failure"], `${OUTCOME} learning policies item failure`],
];
const CLOSED_ROWS = CLOSED_OBJECTS.flatMap(([kind, path, label]) =>
  (["extra", "missing"] as const).map((change) => [kind, path.join("."), change, path, label] as const),
);

test.each(CLOSED_ROWS)("backward wire nested records have closed required fields: %s %s, %s", (...row) => {
  const [kind, , change, path, label] = row;
  const wire = wireOf(wireRecord(kind));
  const target = atPath(wire["payload"], path);
  let message: string;
  if (change === "extra") {
    target["unexpected"] = null;
    message = `invalid ${label} fields; missing=[], extra=["unexpected"]`;
  } else {
    const first = Object.keys(target)[0]!;
    delete target[first];
    message = `invalid ${label} fields; missing=["${first}"], extra=[]`;
  }
  expectError(() => decodeRecord(logJson(wire)), RangeError, message);
});

const MALFORMED: [RecordKind, (string | number)[], unknown, typeof TypeError | typeof RangeError, string][] = [
  ["ready", ["attempt_id"], true, TypeError, `${READY} attempt_id must be an integer`],
  [
    "ready",
    ["root_result", "input_feedback", "topic", 0],
    1n,
    TypeError,
    'feedback["topic"] must be a string, object, list, or None',
  ],
  ["ready", ["root_result", "self_feedback"], [null], TypeError, "self_feedback messages must be strings"],
  [
    "ready",
    ["root_result", "variable_feedback", "tone"],
    [{}],
    TypeError,
    'variable_feedback["tone"] messages must be strings',
  ],
  [
    "ready",
    ["attributed_results", 0, "source", "execution_seq"],
    true,
    TypeError,
    `${OCCURRENCE} source execution_seq must be an integer`,
  ],
  [
    "ready",
    ["attributed_results", 0, "source", "path"],
    [false],
    RangeError,
    "route path components must be strings or non-negative integers",
  ],
  [
    "ready",
    ["attributed_results", 0, "attribution", 0, "parents"],
    [0n],
    RangeError,
    "root attribution cannot carry sender coordinates",
  ],
  [
    "ready",
    ["attributed_results", 0, "attribution", 0, "message_index"],
    1n,
    RangeError,
    "root attribution cannot carry sender coordinates",
  ],
  ["ready", ["attributed_results", 0, "feedback"], [], RangeError, "feedback and attribution must align"],
  ["ready", ["attributed_results", 0, "invoked"], 1n, TypeError, `${OCCURRENCE} invoked must be a boolean`],
  ["ready", ["proposals", 0, "memory"], "", RangeError, `${READY} proposals item memory must be a non-empty string`],
  [
    "finished",
    ["outcome", "failure", "scope", "occurrence_path", 0, "index"],
    true,
    TypeError,
    `${OUTCOME} failure scope occurrence_path item index must be an integer`,
  ],
  ["finished", ["outcome", "failure", "node_started_seq"], 0n, RangeError, "node_started_seq must be positive"],
  [
    "finished",
    ["outcome", "failure", "cause", "retryable"],
    1n,
    TypeError,
    `${OUTCOME} failure cause retryable must be a boolean`,
  ],
  [
    "finished",
    ["outcome", "learning", "failures", 0, "memory"],
    "",
    RangeError,
    `${OUTCOME} learning failures item memory must be a non-empty string`,
  ],
  [
    "finished",
    ["outcome", "learning", "policies", 1, "commit_id"],
    null,
    RangeError,
    "requested policy requires commit_id",
  ],
  [
    "finished",
    ["outcome", "learning", "policies", 3, "failure"],
    null,
    RangeError,
    "only failed policy commits require a failure",
  ],
];

test.each(MALFORMED.map(([kind, path, ...rest]) => [kind, path.join("."), path, ...rest] as const))(
  "backward wire rejects malformed nested values: %s %s",
  (kind, _at, path, invalid, cls, message) => {
    const wire = wireOf(wireRecord(kind));
    atPath(wire["payload"], path.slice(0, -1))[path.at(-1)!] = invalid;
    expectError(() => decodeRecord(logJson(wire)), cls, message);
  },
);

test("a forward cause's wire rejects hybrid and extra origin fields", () => {
  const wire = wireOf(failedRecord("forward"));
  const cause = wire["payload"]["outcome"]["failure"]["cause"];
  cause["stage"] = "backend";
  expectError(
    () => decodeRecord(logJson(wire)),
    RangeError,
    `invalid ${OUTCOME} failure cause fields; missing=["retryable"], ` +
      'extra=["cause_type","node_id","trace","traceback"]',
  );
  delete cause["stage"];
  cause["trace"][0]["file"] = "/private/source.yaml";
  expectError(
    () => decodeRecord(logJson(wire)),
    RangeError,
    `invalid ${OUTCOME} failure cause trace item fields; missing=[], extra=["file"]`,
  );
});

test.each([null, "", "agent"])(
  "a backward failure's cause round trips its optional execution attribution: %p",
  (nodeId) => {
    const cause = new Failure({
      code: "graph.terminal_unreachable",
      message: "No executing producer.",
      nodeId,
      trace: [new OriginRef("root.boundary")],
    });
    const failure = new BackwardFailure("backward", "backward.execute", "Could not finish.", { cause });
    const record = new BackwardFinishedRecord(1, new BackwardOutcome("failed", { failure }));
    const decoded = roundTrip(record).outcome.failure!.cause as Failure;
    expect(decoded).toEqual(cause);
    expect(decoded.constructor).toBe(Failure);
    expect(decoded.trace).toEqual([new OriginRef("root.boundary")]);
  },
);

test.each<[string, unknown]>([
  ["1n", 1n],
  ["true", true],
  ["[]", []],
  ["{}", {}],
])("a backward failure's cause wire rejects a non-text execution attribution: %s", (_name, invalid) => {
  const wire = wireOf(failedRecord("forward"));
  wire["payload"]["outcome"]["failure"]["cause"]["node_id"] = invalid;
  const message = `${OUTCOME} failure cause node_id must be a string or null`;
  expectError(() => decodeRecord(logJson(wire)), TypeError, message);
});

test.each<["failed" | "routed", string | null, typeof TypeError | typeof RangeError]>([
  ["failed", null, TypeError],
  ["failed", "", RangeError],
  ["routed", null, TypeError],
  ["routed", "", RangeError],
])("a node failure's wire keeps its required execution attribution: %s, %p", (kind, nodeId, cls) => {
  const failure = new Failure({
    code: "graph.terminal_unreachable",
    message: "Terminal unreachable.",
    nodeId: "agent",
  });
  const record =
    kind === "failed"
      ? new NodeFailedRecord(failure)
      : new NodeRoutedRecord("agent", "other", { terminalFailure: failure });
  const wire = wireOf(record);
  const field = kind === "failed" ? "failure" : "terminal_failure";
  wire["payload"][field]["node_id"] = nodeId;
  const message = `node_${kind} payload ${field} node_id must be a non-empty string`;
  expectError(() => decodeRecord(logJson(wire)), cls, message);
});

test("sameBackwardAttribution compares sender by key, input, indexes and parents", () => {
  const fields = {
    sender: new SourceRef("run", 3),
    inputName: "text",
    messageIndex: 1,
    contributionIndex: 2,
    parents: [0, 2],
  };
  const transfer = (changed: Partial<typeof fields> = {}) => {
    const { sender, inputName, messageIndex, contributionIndex, parents } = { ...fields, ...changed };
    return new BackwardAttribution(sender, inputName, messageIndex, contributionIndex, parents);
  };
  const root = () => new BackwardAttribution(null, null, 0, null, []);
  const base = transfer();

  expect(sameBackwardAttribution(base, transfer())).toBe(true);
  // Both null: two root seeds have no sender, input name or contribution index.
  expect(sameBackwardAttribution(root(), root())).toBe(true);
  // One null: a root seed against a transfer, for the sender, input name and contribution index.
  expect(sameBackwardAttribution(root(), base)).toBe(false);
  expect(sameBackwardAttribution(base, root())).toBe(false);
  expect(sameBackwardAttribution(base, transfer({ sender: new SourceRef("run", 4) }))).toBe(false);
  expect(sameBackwardAttribution(base, transfer({ sender: new SourceRef("other", 3) }))).toBe(false);
  expect(sameBackwardAttribution(base, transfer({ inputName: "other" }))).toBe(false);
  expect(sameBackwardAttribution(base, transfer({ messageIndex: 0 }))).toBe(false);
  expect(sameBackwardAttribution(base, transfer({ contributionIndex: 3 }))).toBe(false);
  expect(sameBackwardAttribution(base, transfer({ parents: [0] }))).toBe(false);
  expect(sameBackwardAttribution(base, transfer({ parents: [0, 3] }))).toBe(false);
  expect(sameBackwardAttribution(base, transfer({ parents: [0, 2, 3] }))).toBe(false);
});

test("sameAttributedBackwardResult compares source by key, feedback, attributions, result and invoked", () => {
  const transfer = (parents = [0]) => new BackwardAttribution(new SourceRef("run", 3), "text", 0, 0, parents);
  const fields = {
    source: new SourceRef("run", 1),
    feedback: ["Review.", { title: "Shorter." }] as FeedbackValue[],
    attribution: [new BackwardAttribution(null, null, 0, null, []), transfer()],
    result: result(),
  };
  const build = (changed: Partial<typeof fields> = {}) => {
    const { source, feedback, attribution, result } = { ...fields, ...changed };
    return new AttributedBackwardResult(source, feedback, attribution, result, true);
  };
  const base = build();
  const unreached = (source = new SourceRef("run", 1)) =>
    new AttributedBackwardResult(source, [], [], new BackwardResult(), false);

  expect(
    sameAttributedBackwardResult(
      base,
      build({
        source: new SourceRef("run", 1),
        feedback: ["Review.", { title: "Shorter." }],
        attribution: [new BackwardAttribution(null, null, 0, null, []), transfer()],
        result: result(),
      }),
    ),
  ).toBe(true);
  expect(sameAttributedBackwardResult(unreached(), unreached())).toBe(true);
  expect(sameAttributedBackwardResult(base, build({ source: new SourceRef("run", 2) }))).toBe(false);
  expect(sameAttributedBackwardResult(base, build({ source: new SourceRef("other", 1) }))).toBe(false);
  expect(sameAttributedBackwardResult(base, build({ feedback: ["Review.", { title: "Longer." }] }))).toBe(false);
  expect(sameAttributedBackwardResult(base, build({ feedback: [{ title: "Shorter." }, "Review."] }))).toBe(false);
  expect(sameAttributedBackwardResult(base, build({ attribution: [fields.attribution[0]!, transfer([1])] }))).toBe(
    false,
  );
  expect(sameAttributedBackwardResult(base, build({ attribution: [transfer(), fields.attribution[0]!] }))).toBe(false);
  expect(sameAttributedBackwardResult(base, build({ result: new BackwardResult({ selfFeedback: ["Other."] }) }))).toBe(
    false,
  );
  // An unreached occurrence has no messages, so invoked can't differ alone.
  expect(sameAttributedBackwardResult(unreached(), base)).toBe(false);
  expect(sameAttributedBackwardResult(unreached(), unreached(new SourceRef("run", 2)))).toBe(false);
});

test("sameBackwardFailure compares every field and the cause by value", () => {
  type Options = NonNullable<ConstructorParameters<typeof BackwardFailure>[3]>;
  const scope = (index = 2) =>
    new ExecutionScope(["each", "agent"], {
      flowPath: ["each"],
      occurrencePath: [new ExecutionPosition("map_item", ["each"], index)],
    });
  const forward = (message = "Failed.") =>
    new Failure({
      code: "node.execute",
      message,
      nodeId: "agent",
      trace: [new OriginRef("agent.prompt")],
      causeType: "Error",
      traceback: "forward traceback",
    });
  const memoryCause = (retryable = true) =>
    new MemoryOperationFailure("backend-stage", "backend.failed", "Retry later.", retryable);
  const options = (): Options => ({
    scope: scope(),
    runId: "run",
    nodeId: "each/2/agent",
    nodeStartedSeq: 7,
    memory: "history",
    cause: forward(),
    causeType: "MemoryOperationError",
    traceback: "backward traceback",
  });
  type Fields = Options & { stage?: string; code?: string; message?: string };
  const failure = (changed: Fields = {}) => {
    const { stage = "learn", code = "memory.learn", message = "Could not propose.", ...rest } = changed;
    return new BackwardFailure(stage, code, message, { ...options(), ...rest });
  };
  const base = failure();

  const cause = forward();
  const [first, second] = [failure({ cause }), failure({ cause })];
  expect(first.cause).not.toBe(second.cause);
  expect(sameBackwardFailure(first, second)).toBe(true);
  expect(sameBackwardFailure(base, failure())).toBe(true);
  expect(sameBackwardFailure(failure({ cause: memoryCause() }), failure({ cause: memoryCause() }))).toBe(true);
  const bare = new BackwardFailure("learn", "memory.learn", "Could not propose.");
  expect(sameBackwardFailure(bare, new BackwardFailure("learn", "memory.learn", "Could not propose."))).toBe(true);

  expect(sameBackwardFailure(base, failure({ stage: "commit" }))).toBe(false);
  expect(sameBackwardFailure(base, failure({ code: "memory.commit" }))).toBe(false);
  expect(sameBackwardFailure(base, failure({ message: "" }))).toBe(false);
  expect(sameBackwardFailure(base, failure({ scope: scope(3) }))).toBe(false);
  expect(sameBackwardFailure(base, failure({ cause: forward("Other.") }))).toBe(false);
  expect(sameBackwardFailure(failure({ cause: memoryCause() }), failure({ cause: memoryCause(false) }))).toBe(false);
  expect(sameBackwardFailure(base, failure({ cause: memoryCause() }))).toBe(false);
  expect(sameBackwardFailure(failure({ cause: memoryCause() }), base)).toBe(false);
  const changed: [keyof Options, Options[keyof Options]][] = [
    ["runId", "other"],
    ["nodeId", "each/3/agent"],
    ["nodeStartedSeq", 8],
    ["memory", "store"],
    ["causeType", "Error"],
    ["traceback", "other traceback"],
  ];
  for (const [field, value] of changed) {
    expect(sameBackwardFailure(base, failure({ [field]: value }))).toBe(false);
  }
  const nullable: (keyof Options)[] = [
    "scope",
    "runId",
    "nodeId",
    "nodeStartedSeq",
    "memory",
    "cause",
    "causeType",
    "traceback",
  ];
  for (const field of nullable) {
    expect(sameBackwardFailure(failure({ [field]: null }), failure({ [field]: null }))).toBe(true);
    expect(sameBackwardFailure(base, failure({ [field]: null }))).toBe(false);
    expect(sameBackwardFailure(failure({ [field]: null }), base)).toBe(false);
  }
});

test("sameBackwardOutcome compares status, feedback, failure and learning", () => {
  const learning = (commitId = "commit-1") =>
    new LearningResult("succeeded", { policies: [new PolicyCommitOutcome("history", "succeeded", { commitId })] });
  const failure = (message = "Could not propose.") =>
    new BackwardFailure("learn", "memory.learn", message, { cause: MEMORY_CAUSE });
  type Options = NonNullable<ConstructorParameters<typeof BackwardOutcome>[1]>;
  const succeeded = (changed: Options = {}) =>
    new BackwardOutcome("succeeded", { feedback: result(), learning: learning(), ...changed });
  const failed = (changed: Options = {}) =>
    new BackwardOutcome("failed", { feedback: result(), failure: failure(), learning: learning(), ...changed });
  const base = succeeded();

  expect(sameBackwardOutcome(base, succeeded())).toBe(true);
  expect(sameBackwardOutcome(failed(), failed())).toBe(true);
  expect(sameBackwardOutcome(base, failed())).toBe(false);
  expect(sameBackwardOutcome(base, succeeded({ feedback: new BackwardResult() }))).toBe(false);
  expect(sameBackwardOutcome(succeeded({ feedback: null }), succeeded({ feedback: null }))).toBe(true);
  expect(sameBackwardOutcome(base, succeeded({ feedback: null }))).toBe(false);
  expect(sameBackwardOutcome(succeeded({ feedback: null }), base)).toBe(false);
  expect(sameBackwardOutcome(failed(), failed({ failure: failure("Other.") }))).toBe(false);
  // Both null on success; a failure is present exactly when the status is failed, so one null
  // comes with a status change.
  expect(sameBackwardOutcome(failed(), base)).toBe(false);
  expect(sameBackwardOutcome(base, succeeded({ learning: learning("commit-2") }))).toBe(false);
  expect(sameBackwardOutcome(succeeded({ learning: null }), succeeded({ learning: null }))).toBe(true);
  expect(sameBackwardOutcome(base, succeeded({ learning: null }))).toBe(false);
  expect(sameBackwardOutcome(succeeded({ learning: null }), base)).toBe(false);
  expect(sameBackwardOutcome(failed(), failed({ learning: new LearningResult("incomplete") }))).toBe(false);
});

test("backward state revalidates the exact memory order", () => {
  expectError(
    () => new RuntimeBackwardState(new Set(["alpha", "beta"]) as never),
    TypeError,
    "backward memory order must be an array of non-empty strings",
  );
  expectError(
    () => new RuntimeBackwardState(["beta", "alpha"]),
    RangeError,
    "backward memory order must be sorted and unique",
  );
  expect(new RuntimeBackwardState(["alpha", "beta"]).memoryOrder).toEqual(["alpha", "beta"]);
});

test("the post-terminal trace delta validation covers the closed field inventory", () => {
  expect(new Set(Object.keys(new TraceDelta(null, 0)))).toEqual(
    new Set([
      "originatingTrace",
      "sequence",
      "initialized",
      "started",
      "resumed",
      "occurrence",
      "handler",
      "writer",
      "systemValue",
      "pruneIds",
      "connections",
      "consumedConnection",
      "boundaries",
      "boundaryUpdate",
      "rootStartSource",
      "rootWriter",
      "causalRouteSource",
      "terminal",
    ]),
  );
});

// The run state's backward records, on a terminal built through the reducer.

/** Start and succeed each `[nodeId, value, inputs]` in order, capturing `inputs` as bindings. */
function succeedEach(state: RunState, log: MemoryRunLog, outcomes: [string, unknown, Record<string, unknown>][]): void {
  for (const [nodeId, value, inputs] of outcomes) {
    commitPrepared(log, state, prepareNodeStarted(state, nodeId));
    const capture = forwardCapture(state.flow.nodes.get(nodeId)!, inputs);
    commitPrepared(log, state, prepareOutput(log, state, nodeId, value, { forwardCapture: capture }));
  }
}

/**
 * Return a learnable state and log after `START -> work -> END` succeeded, `work` reading
 * `${input.topic}` and END's `answer` reading `work`, with a root-owned memory per key in `keys`
 * (Python's `_retained_backward`, without `compute_backward`).
 */
function retainedTerminal(
  options: { topic?: unknown; keys?: readonly string[] | null } = {},
): { state: RunState; log: MemoryRunLog } {
  const { topic = "reefs", keys = null } = options;
  const work = new FuncNode("work", (inputs) => inputs.x, { reads: { x: "${input.topic}" } });
  const owned = graph([work], [[START_ID, "work"]], { outputs: [new FlowOutput("answer", "${work.output}")] });
  const root = new DeclSite([], ["root"]);
  const { state, log } = learnableStarted(owned, {
    inputs: { topic },
    memories: keys === null ? null : keys.map((key) => new MemorySpec(key, key, {}, [root])),
    memoryStacks: keys === null ? null : [],
    memorySnapshots: keys === null ? null : keys.map((key) => new MemorySnapshotEnvelope(key, {})),
  });
  succeedEach(state, log, [
    [START_ID, { topic }, {}],
    ["work", topic, { x: topic }],
    [END_ID, topic, { answer: topic }],
  ]);
  return { state, log };
}

/** Prepare and commit `record` (Python's `_accept_backward`). */
function acceptBackward(state: RunState, log: RunLog, record: RunRecord): void {
  commitPrepared(log, state, prepareRecord(state, record));
}

/** Return `prepared` with its plan's `changes`, for Python's `dataclasses.replace`. */
function changedPlan(prepared: PreparedTransition, changes: Record<string, unknown>): PreparedTransition {
  const plan = Object.freeze({ ...prepared._plan, ...changes });
  return new PreparedTransition(prepared.record, prepared.expectedRunId, prepared.expectedSequence, plan as never, []);
}

/** Return a copy of `delta` with `changes`. */
function changedDelta(delta: TraceDelta, changes: TraceDeltaFields): TraceDelta {
  return new TraceDelta(delta.originatingTrace, delta.sequence, { ...delta, ...changes });
}

test("the reducer-built terminal records learning and terminal trace authority", () => {
  const { state } = retainedTerminal();
  expect(state.result).toBeInstanceOf(RunSucceeded);
  expect(state.trace!.terminal!.succeeded).toBe(true);
  // RunStarted and the run start's two commits, then three successes of three records each.
  expect(state.nextSequence).toBe(12);
});

test.each(["pristine", "nonterminal", "learn_false", "missing_terminal"] as const)(
  "a backward start requires initialized, learned terminal authority: %s",
  (gate) => {
    const owned = () => graph([], [[START_ID, END_ID]]);
    let state: RunState;
    let log: MemoryRunLog;
    if (gate === "learn_false") {
      ({ state, log } = initializedAndStarted(owned()));
      commitStartSuccess(state, log, {});
      commitPrepared(log, state, prepareNodeStarted(state, END_ID));
      commitPrepared(log, state, prepareOutput(log, state, END_ID, null));
    } else {
      ({ state, log } = learnableStarted(owned()));
      if (gate === "missing_terminal") {
        succeedEach(state, log, [
          [START_ID, {}, {}],
          [END_ID, null, {}],
        ]);
        state.trace!.terminal = null;
      }
      if (gate === "pristine") state = new RunState(owned());
    }
    const messages = {
      pristine: "run state is not initialized",
      nonterminal: "backward records require an accepted terminal result",
      learn_false: "backward records require recorded learning and terminal trace authority",
      missing_terminal: "backward records require recorded learning and terminal trace authority",
    };
    const prior = state.backwardState;
    const sequence = log.nextSequence;
    expectError(() => prepareRecord(state, new BackwardStartedRecord(1, "Review.")), RangeError, messages[gate]);
    expect(state.backwardState).toBe(prior);
    expect(log.nextSequence).toBe(sequence);
  },
);

test.each([
  ["stale", "postterminal trace delta must only advance the accepted sequence"],
  ["next_state", "backward plan state disagrees with its canonical record"],
  ["trace", "postterminal trace delta must only advance the accepted sequence"],
  ["missing_trace", "postterminal trace delta must only advance the accepted sequence"],
  ["payload", "prepared record changed after validation"],
  ["origin", "prepared transition belongs to a different originating RunState"],
] as const)("a prepared backward plan rejects stale, forged or changed authority: %s", (change, message) => {
  const { state, log } = retainedTerminal();
  let prepared = prepareRecord(state, new BackwardStartedRecord(1, "Review."));
  const delta = prepared._plan.traceDelta!;
  if (change === "stale") acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  else if (change === "next_state") prepared = changedPlan(prepared, { nextBackwardState: state.backwardState });
  else if (change === "trace") prepared = changedPlan(prepared, { traceDelta: changedDelta(delta, { resumed: true }) });
  else if (change === "missing_trace") prepared = changedPlan(prepared, { traceDelta: null });
  else if (change === "payload") {
    const changed = new BackwardStartedRecord(1, "changed");
    prepared = new PreparedTransition(changed, prepared.expectedRunId, prepared.expectedSequence, prepared._plan, []);
  } else prepared = changedPlan(prepared, { originatingState: retainedTerminal().state });
  const prior = state.backwardState;
  const sequence = log.nextSequence;
  expectError(() => commitPrepared(log, state, prepared), RangeError, message);
  expect(state.backwardState).toBe(prior);
  expect(log.nextSequence).toBe(sequence);
});

const POSTTERMINAL_FIELDS = [
  "initialized",
  "started",
  "occurrence",
  "writer",
  "rootStartSource",
  "rootWriter",
  "causalRouteSource",
  "terminal",
  "handler",
] as const;

test.each(POSTTERMINAL_FIELDS)("a post-terminal trace delta rejects a forged %s before append", (field) => {
  const { state, log } = retainedTerminal();
  const prepared = prepareRecord(state, new BackwardStartedRecord(1, "Review."));
  const trace = state.trace!;
  const delta = prepared._plan.traceDelta!;
  const writer = [...trace.currentWriters][0]!;
  const changes: Record<(typeof POSTTERMINAL_FIELDS)[number], TraceDeltaFields> = {
    initialized: { initialized: trace },
    started: { started: ["work", delta.sequence] },
    occurrence: { occurrence: trace.occurrences.at(-1)! },
    writer: { writer },
    rootStartSource: { rootStartSource: new SourceRef(state.runId!, 0) },
    rootWriter: { rootWriter: writer[1] },
    causalRouteSource: { causalRouteSource: new SourceRef(state.runId!, 0) },
    terminal: { terminal: trace.terminal },
    handler: {
      handler: () => {
        throw new Error("invoked a forged trace handler");
      },
    },
  };
  const forged = changedPlan(prepared, { traceDelta: changedDelta(delta, changes[field]) });
  const writers = [...trace.currentWriters];
  const sources = [...trace.sources];
  const backward = state.backwardState;
  const sequence = state.nextSequence;
  expectError(
    () => commitPrepared(log, state, forged),
    RangeError,
    "postterminal trace delta must only advance the accepted sequence",
  );
  expect(state.backwardState).toBe(backward);
  expect(state.trace).toBe(trace);
  expect([...trace.currentWriters]).toEqual(writers);
  expect([...trace.sources]).toEqual(sources);
  expect(state.nextSequence).toBe(sequence);
  expect(trace.nextSequence).toBe(sequence);
  expect(log.nextSequence).toBe(sequence);
});

test("a backward start's append failure installs no attempt", () => {
  const { state, log } = retainedTerminal();
  const prepared = prepareRecord(state, new BackwardStartedRecord(1, "Review."));
  const prior = state.backwardState;
  const sequence = state.nextSequence;
  const result = state.result;
  log.append = () => {
    throw new Error("append failed");
  };
  expect(() => commitPrepared(log, state, prepared)).toThrow("append failed");
  expect(state.backwardState).toBe(prior);
  expect(prior.everRequested).toBe(false);
  expect(state.result).toBe(result);
  expect(state.nextSequence).toBe(sequence);
  expect(state.trace!.nextSequence).toBe(sequence);
  expect(log.nextSequence).toBe(sequence);
});

test("an interrupted attempt before Ready requires an explicit finish before a fresh start", () => {
  const { state, log } = retainedTerminal();
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  const prior = state.backwardState;
  const sequence = log.nextSequence;
  const invalid: [BackwardStartedRecord, string][] = [
    // The feedback check runs first: object feedback doesn't fit the text answer.
    [new BackwardStartedRecord(2, {}), "feedback requires a recorded object"],
    [new BackwardStartedRecord(2, "Review."), "backward attempt cannot start from the accepted prefix"],
  ];
  for (const [record, message] of invalid) {
    expectError(() => prepareRecord(state, record), RangeError, message);
    expect(state.backwardState).toBe(prior);
    expect(log.nextSequence).toBe(sequence);
  }
  acceptBackward(state, log, new BackwardFinishedRecord(1, prior.outcome!));
  expect(state.backwardState.canStart).toBe(true);
  acceptBackward(state, log, new BackwardStartedRecord(2, "New feedback."));
  expect(state.backwardState.attempts).toHaveLength(2);
  expect(state.backwardState.ready).toBeNull();
  expect(state.backwardState.outcome!.feedback).toBeNull();
});

test("a prepared backward plan rejects attempt state replaced after preparation", () => {
  const { state, log } = retainedTerminal();
  const prepared = prepareRecord(state, new BackwardStartedRecord(1, "Review."));
  const plan = prepared._plan as { nextBackwardState: RuntimeBackwardState };
  const records = log.records();
  state.backwardState = plan.nextBackwardState;
  expectError(
    () => commitPrepared(log, state, prepared),
    RangeError,
    "prepared backward transition has stale attempt authority",
  );
  expect(log.records()).toEqual(records);
});

/** The `work` occurrence's failed computation, with `changes` to its attribution. */
function computationFailure(
  state: RunState,
  changes: { scope?: ExecutionScope; runId?: string; nodeId?: string; nodeStartedSeq?: number } = {},
  options: { nodeId?: string; code?: string; causeType?: string; traceback?: string } = {},
): BackwardFailure {
  const { nodeId = "work", code = "backward.execute", ...debug } = options;
  const occurrence = state.trace!.occurrences.find((item) => item.nodeId === nodeId)!;
  const context = state.trace!.backwardContext(occurrence.source);
  return new BackwardFailure("backward", code, "Callback failed.", {
    scope: context.scope,
    runId: context.runId,
    nodeId: context.nodeId,
    nodeStartedSeq: context.nodeStartedSeq,
    ...changes,
    ...debug,
  });
}

test("a failed backward computation has no completed feedback and keeps unrequested learning", () => {
  const { state, log } = retainedTerminal({ keys: ["alpha", "beta"] });
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  const failure = computationFailure(state, {}, { causeType: "RuntimeError", traceback: "debug trace" });
  const outcome = new BackwardOutcome("failed", { failure, learning: state.backwardState.outcome!.learning });
  acceptBackward(state, log, new BackwardFinishedRecord(1, outcome));
  expect(sameBackwardOutcome(state.backwardState.outcome!, outcome)).toBe(true);
  expect(state.backwardState.finished!.attributedResults).toEqual([]);
  expect(outcome.feedback).toBeNull();
  expect(outcome.learning!.status).toBe("incomplete");
  expect(outcome.learning!.policies.map((policy) => policy.status)).toEqual(["unrequested", "unrequested"]);
  expect(state.backwardState.canStart).toBe(true);
});

const INVOCATION = "backward computation failure must match its recorded invocation";

test.each([
  ["source", { nodeStartedSeq: 9999 }, "backward computation failure requires an accepted occurrence"],
  ["node", { nodeId: "foreign" }, INVOCATION],
  ["scope", { scope: new ExecutionScope(["foreign"]) }, INVOCATION],
  ["run", { runId: "foreign" }, "backward computation failure requires accepted occurrence attribution"],
] as const)("a failed computation's Finished rejects a foreign context: %s", (_change, changes, message) => {
  const { state, log } = retainedTerminal();
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  const failure = computationFailure(state, changes);
  const record = new BackwardFinishedRecord(1, new BackwardOutcome("failed", { failure }));
  const sequence = log.nextSequence;
  expectError(() => prepareRecord(state, record), RangeError, message);
  expect(log.nextSequence).toBe(sequence);
});

test.each<[FeedbackValue, string]>([
  [null, "backward.execute"],
  [[null, null], "backward.result"],
  ["Review.", "backward.unknown"],
])("a computation failure requires effective feedback and a driver failure code: %j, %s", (feedback, code) => {
  const { state, log } = retainedTerminal({ topic: [1n, 2n] });
  acceptBackward(state, log, new BackwardStartedRecord(1, feedback));
  const failure = computationFailure(state, {}, { nodeId: END_ID, code });
  const record = new BackwardFinishedRecord(1, new BackwardOutcome("failed", { failure }));
  const sequence = log.nextSequence;
  expectError(
    () => prepareRecord(state, record),
    RangeError,
    "backward computation failure requires effective feedback and a driver failure code",
  );
  expect(log.nextSequence).toBe(sequence);
});

// The run state's backward records, on a terminal the engine ran, with `computeBackward`'s results.

/**
 * Run `START -> work -> END`, `work` reading `${input.topic}` and sending its feedback back to
 * `x`, with a root-owned memory per key in `keys`, then compute backward for `feedback` (Python's
 * `_retained_backward`). Returns the engine's state and log, and the root and attributed results.
 */
async function retainedBackward(
  options: { keys?: readonly string[] | null; feedback?: FeedbackValue; topic?: unknown } = {},
): Promise<{ state: RunState; log: RunLog; root: BackwardResult; attributed: readonly AttributedBackwardResult[] }> {
  const { keys = null, feedback = "Review.", topic = "reefs" } = options;
  const work = new RecordingBackwardNode(
    "work",
    (inputs) => inputs.x,
    (_context, messages) => new BackwardResult({ inputFeedback: { x: messages } }),
    { reads: { x: "${input.topic}" } },
  );
  const owned = graph([work], [[START_ID, "work"]], { outputs: [new FlowOutput("answer", "${work.output}")] });
  const root = new DeclSite([], ["root"]);
  const built = engine(owned, {
    inputs: { topic },
    learnable: true,
    memories: keys === null ? null : keys.map((key) => new MemorySpec(key, key, {}, [root])),
    memoryStacks: keys === null ? null : [],
    memorySnapshots: keys === null ? null : keys.map((key) => new MemorySnapshotEnvelope(key, {})),
  });
  await built.run();
  const [rootResult, attributedResults] = await computeBackward(built.state.trace!, feedback);
  return { state: built.state, log: built.log, root: rootResult, attributed: attributedResults };
}

/** The recorded memory keys and their complete proposal table (Python's `_proposal_table`). */
function proposalTable(): [string[], MemoryProposal[]] {
  const proposals = [
    new MemoryProposal("alpha", [
      new MemoryMutation("second", { text: "A" }),
      new MemoryMutation("first", { text: "B" }),
    ]),
    new MemoryProposal("beta", []),
  ];
  return [["alpha", "beta"], proposals];
}

/** Request and confirm each memory's batch of `proposals` in table order under `attemptId`. */
function commitEach(state: RunState, log: RunLog, proposals: readonly MemoryProposal[], attemptId = 1): void {
  for (const { memory, mutations } of proposals) {
    const commitId = memoryCommitId(state.runId!, memory, mutations, attemptId);
    acceptBackward(state, log, new MemoryCommitRequestedRecord(memory, commitId, mutations, attemptId));
    acceptBackward(state, log, new MemoryCommitSucceededRecord(memory, commitId, attemptId));
  }
}

test("attributed validation reuses recorded transfers without callbacks", async () => {
  const { state, root, attributed } = await retainedBackward();
  const trace = state.trace!;
  Object.defineProperty(trace, "backwardHandlers", {
    get() {
      throw new Error("read backward callbacks");
    },
  });
  expect(validateAttributedBackward(trace, "Review.", root, attributed)).toBeUndefined();
  expect(root.inputFeedback).toEqual({ topic: ["Review."] });
});

test("backward attempt records prepare, append, then reduce without forward effects", async () => {
  const { state, log, root, attributed } = await retainedBackward();
  const forward = () => [
    state.result,
    state.pool.toWire(),
    [...state.nodeState],
    [...state.edgeState],
    [...state.ready],
    [...state.inFlight],
    state.trace!.occurrences,
  ];
  const before = forward();
  const initial = state.backwardState;
  const records = [
    new BackwardStartedRecord(1, "Review."),
    new BackwardReadyRecord(1, root, attributed, []),
    new BackwardFinishedRecord(1, new BackwardOutcome("succeeded", { feedback: root })),
  ] as const;
  for (const record of records) {
    const prior = state.backwardState;
    const sequence = state.nextSequence;
    const prepared = prepareRecord(state, record);
    expect(state.backwardState).toBe(prior);
    expect(log.nextSequence).toBe(sequence);
    expect(commitPrepared(log, state, prepared)).toEqual([]);
    expect(state.nextSequence).toBe(sequence + 1);
    expect(state.trace!.nextSequence).toBe(sequence + 1);
    expect(log.nextSequence).toBe(sequence + 1);
  }
  expect(state.backwardState).not.toBe(initial);
  expect(state.backwardState.started).toEqual(records[0]);
  expect(state.backwardState.ready).toEqual(records[1]);
  expect(state.backwardState.finished).toEqual(records[2]);
  expect(state.backwardState.outcome).toEqual(records[2].outcome);
  expect(state.backwardState.canStart).toBe(true);
  expect(forward()).toEqual(before);
});

test("backward policy batches and unfinished prefixes follow the accepted layout order", async () => {
  const [keys, proposals] = proposalTable();
  const { state, log, root, attributed } = await retainedBackward({ keys });
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  expect(state.backwardState.outcome!.feedback).toBeNull();
  expect(state.backwardState.outcome!.learning!.status).toBe("incomplete");
  expect(state.backwardState.policyOutcomes.map((policy) => policy.status)).toEqual(["unrequested", "unrequested"]);
  acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
  expect(state.backwardState.outcome!.feedback).toEqual(root);
  for (const { memory, mutations } of proposals) {
    const commitId = memoryCommitId(state.runId!, memory, mutations, 1);
    acceptBackward(state, log, new MemoryCommitRequestedRecord(memory, commitId, mutations, 1));
    expect(state.backwardState.everRequested).toBe(true);
    expect(state.backwardState.policyOutcomes.find((policy) => policy.memory === memory)!.status).toBe("unknown");
    expect(state.backwardState.canStart).toBe(false);
    acceptBackward(state, log, new MemoryCommitSucceededRecord(memory, commitId, 1));
  }
  const interrupted = state.backwardState.outcome!;
  expect(interrupted.status).toBe("failed");
  expect(interrupted.failure!.code).toBe("backward.unfinished");
  expect(interrupted.learning!.status).toBe("incomplete");
  const policies = state.backwardState.policyOutcomes;
  expect(policies.map((policy) => policy.status)).toEqual(["succeeded", "succeeded"]);
  const learning = new LearningResult("succeeded", { policies });
  const outcome = new BackwardOutcome("succeeded", { feedback: root, learning });
  acceptBackward(state, log, new BackwardFinishedRecord(1, outcome));
  expect(state.backwardState.outcome).toEqual(outcome);
  expect(state.backwardState.current!.requests[0]!.mutations).toEqual(proposals[0]!.mutations);
  expect(state.backwardState.current!.requests[1]!.mutations).toEqual([]);
});

test("an interrupted attempt after Ready requires an explicit finish before a fresh start", async () => {
  const { state, log, root, attributed } = await retainedBackward();
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, []));
  const prior = state.backwardState;
  const sequence = log.nextSequence;
  const invalid: [BackwardStartedRecord, string][] = [
    [new BackwardStartedRecord(2, {}), "feedback requires a recorded object"],
    [new BackwardStartedRecord(2, "Review."), "backward attempt cannot start from the accepted prefix"],
  ];
  for (const [record, message] of invalid) {
    expectError(() => prepareRecord(state, record), RangeError, message);
    expect(state.backwardState).toBe(prior);
    expect(log.nextSequence).toBe(sequence);
  }
  acceptBackward(state, log, new BackwardFinishedRecord(1, prior.outcome!));
  expect(state.backwardState.canStart).toBe(true);
  acceptBackward(state, log, new BackwardStartedRecord(2, "New feedback."));
  expect(state.backwardState.attempts).toHaveLength(2);
  expect(state.backwardState.ready).toBeNull();
  expect(state.backwardState.outcome!.feedback).toBeNull();
});

const READY_INVENTORY = "backward results must match the complete accepted occurrence order";
const READY_TRANSFER = "backward received feedback and attribution must match exact transfers";

test.each<[string, string]>([
  ["missing", READY_INVENTORY],
  ["duplicate", READY_INVENTORY],
  ["reordered", READY_INVENTORY],
  ["foreign", READY_INVENTORY],
  ["feedback", READY_TRANSFER],
  ["input_coordinate", READY_TRANSFER],
  ["message_coordinate", READY_TRANSFER],
  ["contribution_coordinate", READY_TRANSFER],
  ["parent_coordinate", READY_TRANSFER],
  ["unreached", READY_TRANSFER],
  ["unknown_input", "input_feedback selects undeclared input 'foreign'"],
  ["unknown_variable", "variable_feedback selects undeclared variable 'foreign'"],
  ["short_list", 'feedback["x"] must have exactly 2 items; got 1'],
  ["root", "backward root result must match the accepted START and source-zero projection"],
])(
  "Ready rejects an inventory, local contract, transfer or root disagreement before append: %s",
  async (change, message) => {
    const built = await retainedBackward({ topic: [1n, 2n] });
    const { state, log, attributed } = built;
    let root = built.root;
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    const submitted = [...attributed];
    const workIndex = state.trace!.occurrences.findIndex((occurrence) => occurrence.nodeId === "work");
    const item = submitted[workIndex]!;
    let { source, feedback, attribution, result, invoked } = item;
    if (change === "missing") submitted.pop();
    else if (change === "duplicate") submitted[submitted.length - 1] = submitted[0]!;
    else if (change === "reordered") submitted.reverse();
    else if (change === "root") root = new BackwardResult({ inputFeedback: { topic: ["different"] } });
    else {
      if (change === "foreign") source = new SourceRef("foreign-run", source.executionSeq);
      else if (change === "feedback") feedback = ["different"];
      else if (change === "unreached") [feedback, attribution, result, invoked] = [[], [], new BackwardResult(), false];
      else if (change === "unknown_input") result = new BackwardResult({ inputFeedback: { foreign: ["Review."] } });
      else if (change === "unknown_variable") {
        result = new BackwardResult({ variableFeedback: { foreign: ["Review."] } });
      } else if (change === "short_list") result = new BackwardResult({ inputFeedback: { x: [[null]] } });
      else {
        const original = attribution[0]!;
        attribution = [
          new BackwardAttribution(
            original.sender,
            change === "input_coordinate" ? "foreign" : original.inputName,
            change === "message_coordinate" ? 1 : original.messageIndex,
            change === "contribution_coordinate" ? 1 : original.contributionIndex,
            change === "parent_coordinate" ? [0, 1] : original.parents,
          ),
        ];
      }
      submitted[workIndex] = new AttributedBackwardResult(source, feedback, attribution, result, invoked);
    }
    const prior = state.backwardState;
    const sequence = log.nextSequence;
    expectError(() => prepareRecord(state, new BackwardReadyRecord(1, root, submitted, [])), RangeError, message);
    expect(state.backwardState).toBe(prior);
    expect(log.nextSequence).toBe(sequence);
    expect(state.trace!.nextSequence).toBe(sequence);
  },
);

test.each(["missing", "reordered", "duplicate", "foreign"] as const)(
  "Ready proposals require the complete table in order: %s",
  async (change) => {
    const [keys, proposals] = proposalTable();
    const { state, log, root, attributed } = await retainedBackward({ keys });
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    const submitted = [...proposals];
    if (change === "missing") submitted.pop();
    else if (change === "reordered") submitted.reverse();
    else if (change === "duplicate") submitted[submitted.length - 1] = submitted[0]!;
    else submitted[submitted.length - 1] = new MemoryProposal("foreign", []);
    expectError(
      () => prepareRecord(state, new BackwardReadyRecord(1, root, attributed, submitted)),
      RangeError,
      "memory proposals must cover the complete recorded memory table in order",
    );
    expect(state.backwardState.ready).toBeNull();
  },
);

test("mutation ids may repeat across different recorded memories", async () => {
  const [keys, table] = proposalTable();
  const proposals = [table[0]!, new MemoryProposal("beta", [new MemoryMutation("second", {})])];
  const { state, log, root, attributed } = await retainedBackward({ keys });
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
  expect(state.backwardState.ready!.proposals).toEqual(proposals);
});

/** Each code with `retryable` false, then true. */
const retryableRows = (codes: readonly string[]): [boolean, string][] =>
  codes.flatMap((code) => [false, true].map((retryable): [boolean, string] => [retryable, code]));

test.each(retryableRows(["learn.failed", "backward.unfinished"]))(
  "a learner-failed Finished keeps completed feedback and all unrequested policies: retryable %p, %s",
  async (retryable, code) => {
    const [keys] = proposalTable();
    const { state, log, root, attributed } = await retainedBackward({ keys });
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    const cause = new MemoryOperationFailure("provider-stage", code, "No proposal.", retryable);
    const failure = new BackwardFailure("learn", cause.code, cause.message, { memory: "alpha", cause });
    const learning = new LearningResult("failed", {
      failures: [new LearningFailure("alpha", "learn", cause)],
      policies: state.backwardState.policyOutcomes,
    });
    const outcome = new BackwardOutcome("failed", { feedback: root, failure, learning });
    const finished = new BackwardFinishedRecord(1, outcome, attributed);
    acceptBackward(state, log, finished);
    expect(state.backwardState.outcome).toEqual(finished.outcome);
    expect(state.backwardState.finished!.attributedResults).toEqual(attributed);
    expect(state.backwardState.ready).toBeNull();
    expect(state.backwardState.canStart).toBe(true);
    acceptBackward(state, log, new BackwardStartedRecord(2, "Different."));
    expect(state.backwardState.ready).toBeNull();
    expect(state.backwardState.outcome!.feedback).toBeNull();
  },
);

test.each(retryableRows(["commit.failed", "backward.unfinished"]))(
  "a known commit failure stops later requests and admits a later attempt: retryable %p, %s",
  async (retryable, code) => {
    const [keys, proposals] = proposalTable();
    const { state, log, root, attributed } = await retainedBackward({ keys });
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
    const batch = proposals[0]!.mutations;
    const commitId = memoryCommitId(state.runId!, "alpha", batch, 1);
    acceptBackward(state, log, new MemoryCommitRequestedRecord("alpha", commitId, batch, 1));
    const cause = new MemoryOperationFailure("backend-stage", code, "Rejected.", retryable);
    acceptBackward(state, log, new MemoryCommitFailedRecord("alpha", commitId, cause, 1));
    expect(state.backwardState.outcome!.learning!.status).toBe("failed");
    expect(state.backwardState.policyOutcomes.map((policy) => policy.status)).toEqual(["failed", "unrequested"]);
    const beta = new MemoryCommitRequestedRecord("beta", memoryCommitId(state.runId!, "beta", [], 1), [], 1);
    expectError(
      () => prepareRecord(state, beta),
      RangeError,
      "backward commit request requires Ready and no pending or failed memory",
    );
    const learning = new LearningResult("failed", {
      failures: [new LearningFailure("alpha", "commit", cause)],
      policies: state.backwardState.policyOutcomes,
    });
    const failure = new BackwardFailure("commit", cause.code, cause.message, { memory: "alpha", cause });
    acceptBackward(
      state,
      log,
      new BackwardFinishedRecord(1, new BackwardOutcome("failed", { feedback: root, failure, learning })),
    );
    expect(state.backwardState.everRequested).toBe(true);
    expect(state.backwardState.canStart).toBe(true);
    acceptBackward(state, log, new BackwardStartedRecord(2, "Different."));
    expect(state.backwardState.current!.started.attemptId).toBe(2);
  },
);

test.each(["succeeded", "commit_failed", "interrupted_after_request"] as const)(
  "any finished attempt admits a later start with fresh proposals: %s",
  async (closing) => {
    const [keys, proposals] = proposalTable();
    const { state, log, root, attributed } = await retainedBackward({ keys });
    const batch = proposals[0]!.mutations;
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
    const commitId = memoryCommitId(state.runId!, "alpha", batch, 1);
    acceptBackward(state, log, new MemoryCommitRequestedRecord("alpha", commitId, batch, 1));
    let outcome: BackwardOutcome;
    if (closing === "succeeded") {
      acceptBackward(state, log, new MemoryCommitSucceededRecord("alpha", commitId, 1));
      commitEach(state, log, proposals.slice(1));
      const learning = new LearningResult("succeeded", { policies: state.backwardState.policyOutcomes });
      outcome = new BackwardOutcome("succeeded", { feedback: root, learning });
    } else if (closing === "commit_failed") {
      const cause = new MemoryOperationFailure("backend-stage", "commit.failed", "Rejected.", true);
      acceptBackward(state, log, new MemoryCommitFailedRecord("alpha", commitId, cause, 1));
      const learning = new LearningResult("failed", {
        failures: [new LearningFailure("alpha", "commit", cause)],
        policies: state.backwardState.policyOutcomes,
      });
      const failure = new BackwardFailure("commit", cause.code, cause.message, { memory: "alpha", cause });
      outcome = new BackwardOutcome("failed", { feedback: root, failure, learning });
    } else {
      outcome = state.backwardState.outcome!;
      expect(outcome.failure!.code).toBe("backward.unfinished");
    }
    expect(state.backwardState.canStart).toBe(false);
    expectError(
      () => prepareRecord(state, new BackwardStartedRecord(2, "Later review.")),
      RangeError,
      "backward attempt cannot start from the accepted prefix",
    );
    acceptBackward(state, log, new BackwardFinishedRecord(1, outcome));
    expect(state.backwardState.canStart).toBe(true);
    acceptBackward(state, log, new BackwardStartedRecord(2, "Later review."));
    const backward = state.backwardState;
    expect(backward.attempts.map((attempt) => attempt.started.attemptId)).toEqual([1, 2]);
    expect(backward.attempts[0]!.finished!.outcome).toEqual(outcome);
    expect(backward.ready).toBeNull();
    expect(backward.current!.requests).toEqual([]);
    expect(backward.policyOutcomes.map((policy) => policy.status)).toEqual(["unrequested", "unrequested"]);
    const [laterRoot, laterAttributed] = await computeBackward(state.trace!, "Later review.");
    acceptBackward(state, log, new BackwardReadyRecord(2, laterRoot, laterAttributed, proposals));
    expect(state.backwardState.ready!.rootResult).toEqual(laterRoot);
  },
);

test("every committing attempt uses its own attempt in the commit id", async () => {
  const [keys, proposals] = proposalTable();
  const { state, log } = await retainedBackward({ keys });
  const runId = state.runId!;
  for (const attempt of [1, 2]) {
    const feedback = `Review ${attempt}.`;
    const [root, attributed] = await computeBackward(state.trace!, feedback);
    acceptBackward(state, log, new BackwardStartedRecord(attempt, feedback));
    acceptBackward(state, log, new BackwardReadyRecord(attempt, root, attributed, proposals));
    expect(state.backwardState.commitAttemptKey).toBe(attempt);
    for (const { memory, mutations } of proposals) {
      if (attempt === 2) {
        const earlier = memoryCommitId(runId, memory, mutations, 1);
        expectError(
          () => prepareRecord(state, new MemoryCommitRequestedRecord(memory, earlier, mutations, 2)),
          RangeError,
          "backward commit batch and identity must match the ordered Ready proposals",
        );
      }
      commitEach(state, log, [new MemoryProposal(memory, mutations)], attempt);
    }
    const learning = new LearningResult("succeeded", { policies: state.backwardState.policyOutcomes });
    const outcome = new BackwardOutcome("succeeded", { feedback: root, learning });
    acceptBackward(state, log, new BackwardFinishedRecord(attempt, outcome));
  }
  const ids = state.backwardState.attempts.flatMap((attempt) => attempt.requests.map((request) => request.commitId));
  expect(new Set(ids).size).toBe(4);
  expect(ids.map((commitId) => commitId.split(":")[0])).toEqual(Array(4).fill("ac-memory-v3"));
});

test("the first committing attempt uses its own attempt id after pre-commit attempts", async () => {
  const [keys, proposals] = proposalTable();
  const { state, log, root, attributed } = await retainedBackward({ keys });
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  acceptBackward(state, log, new BackwardFinishedRecord(1, state.backwardState.outcome!));
  acceptBackward(state, log, new BackwardStartedRecord(2, "Review."));
  acceptBackward(state, log, new BackwardReadyRecord(2, root, attributed, proposals));
  expect(state.backwardState.commitAttemptKey).toBe(2);
  const batch = proposals[0]!.mutations;
  const commitId = memoryCommitId(state.runId!, "alpha", batch, 2);
  acceptBackward(state, log, new MemoryCommitRequestedRecord("alpha", commitId, batch, 2));
  expect(state.backwardState.current!.requests[0]!.commitId.startsWith("ac-memory-v3:")).toBe(true);
});

const COMMIT_BATCH = "backward commit batch and identity must match the ordered Ready proposals";
const MATCHING_ATTEMPT = "backward record requires its matching unfinished attempt";
const PENDING_REQUEST = "backward commit result requires a matching pending request";
const PENDING_MATCH = "backward commit result must match its pending memory and identity";
const NEXT_START = "backward attempt_id must be the next accepted start";
const FINISHED_FEEDBACK = "Finished feedback must match accepted Ready";
const FINISHED_ATTRIBUTION = "Finished attribution must match accepted Ready";
const FINISHED_LEARNING = "successful Finished learning must match accepted memory confirmations";

test("Started feedback shape and attempt identity validate before acceptance", async () => {
  const { state, log, root, attributed } = await retainedBackward({ topic: [1n, 2n], feedback: [null, null] });
  const invalid: [BackwardStartedRecord, string][] = [
    [new BackwardStartedRecord(1, [null]), "feedback must have exactly 2 items; got 1"],
    [new BackwardStartedRecord(2, [null, null]), NEXT_START],
  ];
  for (const [record, message] of invalid) expectError(() => prepareRecord(state, record), RangeError, message);
  acceptBackward(state, log, new BackwardStartedRecord(1, [null, null]));
  acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, []));
  expectError(
    () => prepareRecord(state, new BackwardReadyRecord(1, root, attributed, [])),
    RangeError,
    "backward attempt already has a Ready",
  );
  acceptBackward(state, log, new BackwardFinishedRecord(1, state.backwardState.outcome!));
  expectError(() => prepareRecord(state, new BackwardStartedRecord(1, "")), RangeError, NEXT_START);
  acceptBackward(state, log, new BackwardStartedRecord(2, ""));
  expect(state.backwardState.started!.feedback).toBe("");
});

test.each([
  ["policy", "backward request must name the next recorded memory"],
  ["batch_order", COMMIT_BATCH],
  ["batch_content", COMMIT_BATCH],
  ["identity", COMMIT_BATCH],
  ["attempt", MATCHING_ATTEMPT],
] as const)("a backward commit request rejects a wrong order, batch, identity or attempt: %s", async (...row) => {
  const [change, message] = row;
  const [keys, proposals] = proposalTable();
  const { state, log, root, attributed } = await retainedBackward({ keys });
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
  let batch = proposals[0]!.mutations;
  let policy = "alpha";
  let attempt = 1;
  if (change === "policy") [policy, batch] = ["beta", []];
  else if (change === "batch_order") batch = [...batch].reverse();
  else if (change === "batch_content") batch = [new MemoryMutation("second", { changed: true }), ...batch.slice(1)];
  else if (change === "attempt") attempt = 2;
  const commitId = change === "identity" ? "commit-1" : memoryCommitId(state.runId!, policy, batch, 1);
  const record = new MemoryCommitRequestedRecord(policy, commitId, batch, attempt);
  expectError(() => prepareRecord(state, record), RangeError, message);
  expect(state.backwardState.everRequested).toBe(false);
});

test.each([
  ["no_request", PENDING_REQUEST],
  ["policy", PENDING_MATCH],
  ["identity", PENDING_MATCH],
  ["attempt", MATCHING_ATTEMPT],
  ["duplicate", PENDING_REQUEST],
  ["pending_request", "backward commit request requires Ready and no pending or failed memory"],
] as const)("a backward commit observation requires the current pending request: %s", async (change, message) => {
  const [keys, proposals] = proposalTable();
  const { state, log, root, attributed } = await retainedBackward({ keys });
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
  const batch = proposals[0]!.mutations;
  const commitId = memoryCommitId(state.runId!, "alpha", batch, 1);
  const request = new MemoryCommitRequestedRecord("alpha", commitId, batch, 1);
  if (change !== "no_request") acceptBackward(state, log, request);
  if (change === "duplicate") acceptBackward(state, log, new MemoryCommitSucceededRecord("alpha", commitId, 1));
  const record =
    change === "pending_request"
      ? request
      : new MemoryCommitSucceededRecord(
          change === "policy" ? "beta" : "alpha",
          change === "identity" ? "other" : commitId,
          change === "attempt" ? 2 : 1,
        );
  expectError(() => prepareRecord(state, record), RangeError, message);
});

test.each(["ready", "request", "finished"] as const)(
  "a backward %s append failure installs no attempt or irreversible request",
  async (phase) => {
    const [keys, proposals] = phase === "request" ? proposalTable() : [null, []];
    const { state, log, root, attributed } = await retainedBackward({ keys });
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    if (phase !== "ready") acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
    let record: RunRecord;
    if (phase === "ready") record = new BackwardReadyRecord(1, root, attributed, []);
    else if (phase === "request") {
      const batch = proposals[0]!.mutations;
      record = new MemoryCommitRequestedRecord("alpha", memoryCommitId(state.runId!, "alpha", batch, 1), batch, 1);
    } else record = new BackwardFinishedRecord(1, new BackwardOutcome("succeeded", { feedback: root }));
    const prepared = prepareRecord(state, record);
    const prior = state.backwardState;
    const sequence = state.nextSequence;
    const result = state.result;
    log.append = () => {
      throw new Error("append failed");
    };
    expect(() => commitPrepared(log, state, prepared)).toThrow("append failed");
    expect(state.backwardState).toBe(prior);
    expect(prior.everRequested).toBe(false);
    expect(state.result).toBe(result);
    expect(state.nextSequence).toBe(sequence);
    expect(state.trace!.nextSequence).toBe(sequence);
    expect(log.nextSequence).toBe(sequence);
  },
);

test.each(["feedback", "partial_inventory"] as const)(
  "a failed computation's Finished rejects fabricated results: %s",
  async (change) => {
    const { state, log, root, attributed } = await retainedBackward();
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    const failure = computationFailure(state);
    const outcome = new BackwardOutcome("failed", { feedback: change === "feedback" ? root : null, failure });
    const record = new BackwardFinishedRecord(1, outcome, change === "partial_inventory" ? attributed.slice(0, 1) : []);
    const message = "incomplete backward computation cannot carry completed or partial results";
    expectError(() => prepareRecord(state, record), RangeError, message);
  },
);

test.each(["before_ready", "before_request", "pending", "one_confirmed"] as const)(
  "a successful Finished requires Ready and every enabled policy's confirmation: %s",
  async (phase) => {
    const [keys, proposals] = proposalTable();
    const { state, log, root, attributed } = await retainedBackward({ keys });
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    if (phase !== "before_ready") acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
    if (phase === "pending" || phase === "one_confirmed") {
      const batch = proposals[0]!.mutations;
      const identity = memoryCommitId(state.runId!, "alpha", batch, 1);
      acceptBackward(state, log, new MemoryCommitRequestedRecord("alpha", identity, batch, 1));
      if (phase === "one_confirmed") acceptBackward(state, log, new MemoryCommitSucceededRecord("alpha", identity, 1));
    }
    const outcome = new BackwardOutcome("succeeded", { feedback: root, learning: new LearningResult("succeeded") });
    expectError(
      () => prepareRecord(state, new BackwardFinishedRecord(1, outcome)),
      RangeError,
      "successful Finished requires Ready and every memory confirmation",
    );
  },
);

test.each([
  ["root", FINISHED_FEEDBACK],
  ["attribution", FINISHED_ATTRIBUTION],
  ["missing_memory", FINISHED_LEARNING],
  ["memory_order", FINISHED_LEARNING],
  ["commit_id", FINISHED_LEARNING],
  ["attempt", MATCHING_ATTEMPT],
] as const)(
  "Finished must preserve Ready's results and the exact confirmed policy inventory: %s",
  async (change, message) => {
    const [keys, proposals] = proposalTable();
    const built = await retainedBackward({ keys });
    const { state, log, attributed } = built;
    let root = built.root;
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
    commitEach(state, log, proposals);
    let policies = state.backwardState.policyOutcomes;
    if (change === "root") root = new BackwardResult({ inputFeedback: { topic: ["Different."] } });
    else if (change === "missing_memory") policies = policies.slice(0, 1);
    else if (change === "memory_order") policies = [...policies].reverse();
    else if (change === "commit_id") {
      const [first, ...rest] = policies;
      const options = { commitId: "other", failure: first!.failure };
      policies = [new PolicyCommitOutcome(first!.memory, first!.status, options), ...rest];
    }
    const learning = new LearningResult("succeeded", { policies });
    const outcome = new BackwardOutcome("succeeded", { feedback: root, learning });
    const record = new BackwardFinishedRecord(
      change === "attempt" ? 2 : 1,
      outcome,
      change === "attribution" ? [...attributed].reverse() : [],
    );
    expectError(() => prepareRecord(state, record), RangeError, message);
  },
);

test("an explicit interruption keeps the unknown request and Ready's attribution without resuming", async () => {
  const [keys, proposals] = proposalTable();
  const { state, log, root, attributed } = await retainedBackward({ keys });
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
  const batch = proposals[0]!.mutations;
  const identity = memoryCommitId(state.runId!, "alpha", batch, 1);
  acceptBackward(state, log, new MemoryCommitRequestedRecord("alpha", identity, batch, 1));
  const projected = state.backwardState.outcome!;
  acceptBackward(state, log, new BackwardFinishedRecord(1, projected, attributed));
  expect(state.backwardState.outcome).toEqual(projected);
  expect(projected.feedback).toEqual(root);
  expect(projected.learning!.status).toBe("incomplete");
  expect(projected.learning!.policies.map((item) => item.status)).toEqual(["unknown", "unrequested"]);
  acceptBackward(state, log, new BackwardStartedRecord(2, "Review again."));
  expect(state.backwardState.attempts[0]!.finished!.outcome).toEqual(projected);
  expect(state.backwardState.ready).toBeNull();
});

test.each([
  // A changed message is no longer the interruption, so the commit-failure rule refuses it.
  ["message", "failed Finished after Ready requires a recorded commit failure"],
  ["feedback", FINISHED_FEEDBACK],
  ["learning", "the Finished of an unfinished attempt must exactly project the accepted prefix"],
  ["attribution", FINISHED_ATTRIBUTION],
] as const)(
  "an interrupted Finished must exactly project the prefix: %s",
  async (change, message) => {
    const [keys, proposals] = proposalTable();
    const { state, log, root, attributed } = await retainedBackward({ keys });
    acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
    acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, proposals));
    const projected = state.backwardState.outcome!;
    let { feedback, failure, learning } = projected;
    if (change === "message") {
      const { stage, code, scope, runId, nodeId, nodeStartedSeq, memory, cause, causeType, traceback } = failure!;
      const options = { scope, runId, nodeId, nodeStartedSeq, memory, cause, causeType, traceback };
      failure = new BackwardFailure(stage, code, "different", options);
    } else if (change === "feedback") feedback = new BackwardResult();
    else if (change === "learning") learning = new LearningResult("incomplete");
    const outcome = new BackwardOutcome(projected.status, { feedback, failure, learning });
    const record = new BackwardFinishedRecord(1, outcome, change === "attribution" ? attributed.slice(0, 1) : []);
    expectError(() => prepareRecord(state, record), RangeError, message);
  },
);

test("disabled memory Ready and Finished reject fabricated memory material", async () => {
  const { state, log, root, attributed } = await retainedBackward();
  acceptBackward(state, log, new BackwardStartedRecord(1, "Review."));
  expectError(
    () => prepareRecord(state, new BackwardReadyRecord(1, root, attributed, [new MemoryProposal("fake", [])])),
    RangeError,
    "memory proposals must cover the complete recorded memory table in order",
  );
  acceptBackward(state, log, new BackwardReadyRecord(1, root, attributed, []));
  const outcome = new BackwardOutcome("succeeded", { feedback: root, learning: new LearningResult("succeeded") });
  expectError(() => prepareRecord(state, new BackwardFinishedRecord(1, outcome)), RangeError, FINISHED_LEARNING);
});

test("backward replay matches every accepted prefix without callbacks", async () => {
  const [keys, proposals] = proposalTable();
  const { state, log, root, attributed } = await retainedBackward({ keys });
  const original = copyGraph(state.flow);
  for (const node of original.nodes.values()) {
    node.forward = () => {
      throw new Error("replayed forward");
    };
    node.backward = () => {
      throw new Error("replayed backward callback");
    };
    node.captureBackwardContext = () => {
      throw new Error("recaptured backward context");
    };
  }
  const prefixes: [number, RuntimeBackwardState][] = [[log.records().length, state.backwardState]];
  const records: RunRecord[] = [
    new BackwardStartedRecord(1, "Review."),
    new BackwardReadyRecord(1, root, attributed, proposals),
  ];
  for (const { memory, mutations } of proposals) {
    const commitId = memoryCommitId(state.runId!, memory, mutations, 1);
    records.push(
      new MemoryCommitRequestedRecord(memory, commitId, mutations, 1),
      new MemoryCommitSucceededRecord(memory, commitId, 1),
    );
  }
  for (const record of records) {
    acceptBackward(state, log, record);
    prefixes.push([log.records().length, state.backwardState]);
  }
  const learning = new LearningResult("succeeded", { policies: state.backwardState.policyOutcomes });
  const outcome = new BackwardOutcome("succeeded", { feedback: root, learning });
  acceptBackward(state, log, new BackwardFinishedRecord(1, outcome));
  prefixes.push([log.records().length, state.backwardState]);
  const accepted = log.records();
  for (const [length, expected] of prefixes) {
    const prefix = accepted.slice(0, length);
    const replay = replayRecords(new RunState(copyGraph(original)), prefix, { growBuilder: noGrowth });
    expect(sameRuntimeBackwardState(replay.backwardState, expected)).toBe(true);
    expect(replay.backwardState.outcome).toEqual(expected.outcome);
    expect(replay.trace!.nextSequence).toBe(length);
    expect(replay.nextSequence).toBe(length);
    expect(replay.result).toEqual(state.result);
  }
});

const HISTORICAL_FLOW = `id: historical-backward
name: historical_backward
long_term_memory: test
nodes:
  work:
    kind: code
    code: return 3
    output: int
output: \${work.output}
`;

// The forward run is RunStarted, the run start's four commits and three successes (start, commit,
// success each), so the backward records begin at 14.
test.each([
  ["started", false, 14, "backward_started"],
  ["ready", false, 15, "backward_ready"],
  ["requested", false, 16, "memory_commit_requested"],
  ["succeeded", false, 17, "memory_commit_succeeded"],
  ["finished", false, 18, "backward_finished"],
  ["failed", true, 17, "memory_commit_failed"],
  ["failed-finished", true, 18, "backward_finished"],
] as const)("a historical backward record prepares its recorded line: %s", async (_id, failed, index, kind) => {
  const dir = mkdtempSync(join(tmpdir(), "yian-backward-records-"));
  try {
    const flow = await loadFlow(HISTORICAL_FLOW);
    const path = join(dir, "run.jsonl");
    const backend = new FakeMemory({
      commitOutcomes: failed ? [new MemoryOperationError("commit", "declined", "Not now.", true)] : [],
    });
    const result = await runFlow(flow, {}, {
      log: path,
      learnable: true,
      numWorkers: 0,
      memory: new Map([["test", backend]]),
      env: hostEnv(dir),
    });
    await result.backward("Review.");
    const wires = readFileSync(path, "utf8").split("\n").slice(0, index + 1);
    const last = wires.at(-1)!;
    expect(decodeRecord(last).kind).toBe(kind);

    const state = replayRunLog(flow, MemoryRunLog.fromRecords(wires.slice(0, -1).map((wire) => decodeRecord(wire))));
    const before = state.backwardState;
    const prepared = prepareRecord(state, decodeRecord(last).record);
    expect(prepared._plan.envelopeJson).toBe(untimed(last));
    expect(state.backwardState).toBe(before);
    const cursor = new HistoricalLogCursor(wires);
    for (const wire of wires.slice(0, -1)) cursor.append(decodeRecord(wire).record);
    commitPrepared(cursor, state, prepared);
    expect(state.nextSequence).toBe(index + 1);
    expect(state.backwardState).not.toBe(before);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
