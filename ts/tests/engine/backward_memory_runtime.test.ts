/**
 * Explicit backward proposals and ordered, append-authorized memory effects: a port of
 * `tests/engine/test_backward_memory_runtime.py`, over the shared fakes of `memory_fakes.ts`.
 *
 * Ported: `test_proposal_phase_visits_every_memory_without_committing` and
 * `test_proposal_contexts_validate_completely_before_any_learner` (6 rows), with the `_table` and
 * `_open` helpers; the `backend` and `duplicate` rows of
 * `test_late_proposal_failure_discards_all_staged_mutations`,
 * `test_policy_failure_preserves_prior_success_and_stops_later_policies` (2 rows; `{}` for Python's
 * `object()`), `test_empty_feedback_still_learns_empty_memories_and_commits_empty_batches`,
 * `test_no_policy_backward_still_records_ready_and_finished_without_opening_resources`,
 * `test_computation_failure_finishes_without_partial_feedback_or_learning` (2 rows),
 * `test_append_fault_preserves_exact_prefix_and_closes_resources` (8 rows; the log's `append`
 * instance-patched to throw an `Error`, Python's `OSError`),
 * `test_manual_precommit_retry_reopens_snapshots_and_recomputes_every_memory` (2 rows),
 * `test_known_precommit_failure_allows_fresh_complete_pass_without_prior_proposals`,
 * `test_failed_resource_setup_does_not_close_an_interrupted_attempt` (3 rows; the opener throws an
 * `Error`, Python's `OSError`),
 * `test_later_attempt_follows_or_closes_any_prior_attempt_and_recomputes_everything` (3 rows),
 * `test_repeated_driven_attempts_request_distinct_commit_ids_for_identical_batches`,
 * `test_opened_memory_runtime_subclass_is_closed_by_the_driver` and
 * `test_submission_gates_reject_before_opening_or_appending` (3 rows), with the `_terminal`,
 * `_records` and `_drive` helpers.
 *
 * Changed: Python's tuples are arrays and its dictionaries are `Map`s; `proposeLearning`,
 * `MemoryRuntime.open` and `driveBackward` return promises; each learner receives the very context
 * passed in (`toBe`, where Python compares a revalidated copy with `==`); each refusal pins its
 * class and full message (Python accepts either `TypeError` or `ValueError`), and the `list` row
 * passes a `Set` (Python's list for a tuple). Python's `type(record) is X` is
 * `record.constructor === X`, and its `copy.deepcopy(pool.store)` is `encodePool(state.pool)`.
 * `test_backward_finishes_all_callbacks_before_ready_and_policy_requests` wraps the session's
 * commit with `spyOn(FakeSession.prototype, "commit")` (Python's class-level `monkeypatch`),
 * restored with `mockRestore()`.
 * - `test_late_proposal_failure_discards_all_staged_mutations`: the `tuple` row's learner returns a
 *   `Set` (not an array), and the `forged` row returns a plain object shaped like a mutation with
 *   `mutationId: true`, since a `MemoryMutation` is frozen (Python forges with `object.__setattr__`).
 * - `test_context_depth_failure_keeps_completed_backward_without_calling_learners` and
 *   `test_replayed_context_failure_must_name_an_accepted_receiving_occurrence` (7 rows): TS has no
 *   memory JSON nesting bound, so a spy on `LearnContext.fromFrozen` throws
 *   `RangeError("context too deep")` on its first call (START's occurrence, seq 1, under
 *   `_terminal`'s every-node scope), restored with `mockRestore()`; the six changes rebuild the
 *   failure through the `BackwardFailure` and `MemoryOperationFailure` constructors (Python's
 *   `dataclasses.replace`), and each refusal pins its `RangeError` and full message (Python's
 *   `ValueError`).
 * - `test_computation_failure_finishes_without_partial_feedback_or_learning`: the callback throws a
 *   plain `Error` (Python's `ValueError`).
 * - `test_backend_host_or_unexpected_error_escapes_with_exact_prefix`: the `RuntimeError` and
 *   `RecursionError` cells, × learn and commit, throw a plain `Error` and a `RangeError`.
 * - `test_host_capability_identity_and_cleanup_error_preserve_primary_exception`: the learner's
 *   `CapabilityFault` escapes as that same object (TS unwraps no capability fault), with both
 *   sessions' close failures noted on it; the close is wrapped with
 *   `spyOn(FakeSession.prototype, "close")` and throws an `Error` (Python's `OSError`).
 * - `test_cleanup_error_after_finished_preserves_success_and_admits_a_later_attempt`: the close is
 *   wrapped with `spyOn(FakeSession.prototype, "close")` and throws an `Error` (Python's `OSError`).
 * - `test_backward_and_memory_replay_calls_no_callbacks_or_materialization`: the module-private
 *   `materializeContext` can't be patched, so only the node hooks and
 *   `spyOn(MemoryRuntime.prototype, "proposeLearning"/"commitLearning")` fail if called, and the
 *   replayed backward state is compared with `sameRuntimeBackwardState`.
 * - `test_prepare_failure_is_not_sanitized_as_a_learner_failure`: `prepareRecord` is a module
 *   function that can't be patched, so the log's `append` throws the `MemoryOperationError` for the
 *   Ready record instead.
 * - `test_learner_mutation_of_its_arguments_and_returned_values_cannot_change_authority`: the
 *   context and the mutation are frozen, so each learner's edit throws a `TypeError` (asserted)
 *   before the same authority checks.
 * - `test_wrapped_host_error_during_partial_setup_closes_prior_sessions`: TS unwraps no cause, so
 *   B's `openRun` throwing a `MemoryOperationError` rejects the drive with the `MemorySetupError`
 *   whose `cause` is that error.
 * - `test_cleanup_unwraps_host_failure_after_closing_all_sessions`: the two `with_primary` cells,
 *   with plain errors: the learner throws an `Error` and B's close a `MemoryOperationError`, which
 *   escapes itself without a primary and is noted on the primary with one.
 * - `test_commit_batch_is_detached_from_every_accepted_authority`: the `mutation` and
 *   `public_payload` targets × the three completions (6 cells); the mutation and its payload are
 *   frozen, so the backend's edit throws a `TypeError` (asserted), and the `host_failure` commit
 *   throws a plain `Error` (Python's `MemoryError`) that escapes unchanged. TS also asserts that
 *   `commitLearning` receives the accepted request itself and the session its own commit id and
 *   mutations (`toBe`).
 *
 * TS only: driving with an inexact `RunState` or a non-log refuses before opening; feedback that
 * passes the feedback check but can't be prepared (a lone surrogate) is refused before the opener
 * runs; an async opener resolves before the first record; a drive overtaken by another drive while
 * it awaits its opener (fresh and interrupted prior attempt) or its handler (the handler then
 * throws, or its learner fails) is refused by the reducer, appends nothing and closes its runtime,
 * and one overtaken while its commit is pending is refused the same way and appends nothing
 * further; a drive whose handler throws or whose learner fails while a newer drive is mid-attempt
 * can't finish the newer attempt, which then finishes itself (choice 11); an opener result that
 * isn't a `MemoryRuntime` and a runtime whose earlier close failed are refused; a
 * `MemoryOperationError` with an `AbortError` or `CapabilityFault` cause is a learner or commit
 * failure; a thrown `undefined` still closes the runtime and is rethrown; a drive hands its
 * `models` and `onEvent` to a node's backward hook as its third argument.
 *
 * Not ported:
 * - `test_typed_backend_wrappers_preserve_original_host_error`: it walks the cause chain for the
 *   original host error, and TS unwraps no host control (choice 1): a `MemoryOperationError` from
 *   learn or commit is a learning failure whatever its `cause`, and any other thrown value escapes
 *   unchanged.
 * - `test_each_attempt_checks_only_its_own_records_not_earlier_attempts`: it counts the calls of a
 *   module-private function, which TS can't patch.
 * - `test_nested_capability_faults_never_escape_private_wrappers`,
 *   `test_capability_originals_are_checked_for_wrapped_host_controls` and
 *   `test_cyclic_capability_fault_is_a_host_error_without_exposing_private_carriers`: they unwrap
 *   capability faults (choice 1).
 * - The `BaseException`-type cells (`MemoryError`, both `CancelledError`s, `KeyboardInterrupt`,
 *   `SystemExit`) of `test_backend_host_or_unexpected_error_escapes_with_exact_prefix` and the
 *   `error_type` dimension (the same classes) of
 *   `test_cleanup_unwraps_host_failure_after_closing_all_sessions`: TS has no such classes.
 * - The `inner_payload` cells of `test_commit_batch_is_detached_from_every_accepted_authority`: TS
 *   has no private payload tree.
 */

import { expect, spyOn, test } from "bun:test";

import { copyGraph, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { type FeedbackValue, SourceRef } from "../../src/expr/lineage.ts";
import {
  DeclSite,
  LearnContext,
  LearningFailure,
  MemoryMutation,
  MemoryOperationError,
  MemoryOperationFailure,
  MemoryScope,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
} from "../../src/memory/contracts.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { type BackwardContext, type BackwardOptions, BackwardResult, CapabilityFault } from "../../src/nodes/base.ts";
import { computeBackward, driveBackward } from "../../src/runtime/backward.ts";
import { BackwardFailure, BackwardOutcome, MemoryProposal } from "../../src/runtime/contracts.ts";
import type { FlowEngine } from "../../src/runtime/engine.ts";
import { memoryCommitId, prefixLearning, sameRuntimeBackwardState } from "../../src/runtime/learning.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  encodePool,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  type RunLog,
  type RunRecord,
} from "../../src/runtime/log.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import { commitPrepared, prepareRecord, replayRecords, RunState } from "../../src/runtime/state.ts";
import {
  type BackwardCallback,
  engine,
  graph,
  NO_ENV,
  noGrowth,
  RecordingBackwardNode,
  rejected,
  stacks,
} from "./run_state_fixtures.ts";
import { FakeMemory, FakeSession } from "./memory_fakes.ts";

const ROOT = new DeclSite([], ["root"]);

/** Return a sorted key table; the first key is root-owned, the rest are owned by unused defs. */
function table(...keys: string[]): MemorySpec[] {
  return [...new Set(keys)]
    .sort()
    .map(
      (key, index) =>
        new MemorySpec(key, key, {}, [index === 0 ? ROOT : new DeclSite([], ["def", `unused${index}`])]),
    );
}

/** Open a runtime over `memories` with empty snapshots. */
function open(memories: readonly MemorySpec[], memory: ReadonlyMap<string, FakeMemory>): Promise<MemoryRuntime> {
  return MemoryRuntime.open({
    memories,
    origins: new Map(memories.map((spec) => [spec.key, null])),
    memory,
    snapshots: memories.map((spec) => new MemorySnapshotEnvelope(spec.key, {})),
  });
}

test("proposal phase visits every memory without committing", async () => {
  const memories = table("A", "B", "C");
  const contexts = memories.map((spec) => new LearnContext("run", spec.key, []));
  const order: string[] = [];

  const propose = (context: LearnContext) => {
    order.push(context.memory);
    return [new MemoryMutation("same-id", { notes: ["one"] })];
  };

  const memory = new Map([
    ["A", new FakeMemory({ learnOutcomes: [propose] })],
    ["B", new FakeMemory({ learnOutcomes: [[]] })],
    ["C", new FakeMemory({ learnOutcomes: [propose] })],
  ]);
  const runtime = await open(memories, memory);
  let proposals: readonly MemoryProposal[];
  try {
    proposals = await runtime.proposeLearning(contexts);
  } finally {
    await runtime.close();
  }
  expect(proposals.map((proposal) => proposal.memory)).toEqual(["A", "B", "C"]);
  expect(proposals.every((proposal) => proposal.constructor === MemoryProposal)).toBe(true);
  expect(proposals.map((proposal) => proposal.mutations.length)).toEqual([1, 0, 1]);
  expect(order).toEqual(["A", "C"]);
  ["A", "B", "C"].forEach((key, index) => {
    const received = memory.get(key)!.learnContexts;
    expect(received).toHaveLength(1);
    expect(received[0]).toBe(contexts[index]!);
  });
  for (const backend of memory.values()) {
    expect(backend.commitCalls).toEqual([]);
    expect(backend.closeCount).toBe(1);
  }
});

test("proposal contexts validate completely before any learner", async () => {
  const coverage = "learning contexts must cover the complete recorded memory table in order";
  const rows: [string, new (...args: never[]) => Error, string][] = [
    ["missing", RangeError, coverage],
    ["reordered", RangeError, coverage],
    ["duplicate", RangeError, coverage],
    ["foreign", RangeError, coverage],
    ["run", RangeError, "learning contexts must belong to one run"],
    ["list", TypeError, "learning contexts must be an array of exact LearnContext values"],
  ];
  for (const [invalid, cls, message] of rows) {
    const memories = table("A", "B");
    const [a, b] = memories.map((spec) => new LearnContext("run", spec.key, [])) as [LearnContext, LearnContext];
    let contexts: unknown;
    if (invalid === "missing") contexts = [a];
    else if (invalid === "reordered") contexts = [b, a];
    else if (invalid === "duplicate") contexts = [a, a];
    else if (invalid === "foreign") contexts = [a, new LearnContext("run", "Z", [])];
    else if (invalid === "run") contexts = [a, new LearnContext("other", "B", [])];
    else contexts = new Set([a, b]);
    const memory = new FakeMemory();
    const runtime = await open(memories, new Map([["A", memory], ["B", new FakeMemory()]]));
    let caught: unknown;
    try {
      await runtime.proposeLearning(contexts as LearnContext[]).catch((error: unknown) => {
        caught = error;
      });
    } finally {
      await runtime.close();
    }
    expect((caught as Error | undefined)?.constructor).toBe(cls);
    expect((caught as Error).message).toBe(message);
    expect(memory.learnContexts).toEqual([]);
    expect(memory.commitCalls).toEqual([]);
  }
});

/**
 * Build one real retained CODE occurrence plus a complete enabled key table: every node's scope
 * holds the table's first key.
 */
async function terminal(
  keys: readonly string[] = ["A"],
  options: { value?: unknown; callback?: BackwardCallback | null; learnable?: boolean } = {},
): Promise<{ engine: FlowEngine; node: RecordingBackwardNode }> {
  const { value = "reefs", callback = null, learnable = true } = options;
  const node = new RecordingBackwardNode(
    "work",
    (inputs) => inputs.topic,
    callback ?? ((_context, feedback) => new BackwardResult({ inputFeedback: { topic: feedback } })),
    { reads: { topic: "${input.topic}" } },
  );
  const flow = graph([node], [[START_ID, "work"]], { outputs: [new FlowOutput("answer", "${work.output}")] });
  const memories = table(...keys);
  let memoryStacks = null;
  if (memories.length > 0) {
    const scopes = [...flow.nodes.values()].map((current) => {
      current.memoryScope = new MemoryScope([current.id], { memories: [memories[0]!.key] });
      return current.memoryScope;
    });
    memoryStacks = stacks(...scopes);
  }
  const built = engine(flow, {
    inputs: { topic: value },
    learnable,
    memories: memories.length > 0 ? memories : null,
    memoryStacks,
    memorySnapshots: memories.length > 0 ? memories.map((spec) => new MemorySnapshotEnvelope(spec.key, {})) : null,
  });
  await built.run();
  return { engine: built, node };
}

/** Return the records of the engine's log, in order. */
function records(built: FlowEngine): RunRecord[] {
  return built.log.records().map((envelope) => envelope.record);
}

/** Drive one attempt with `feedback`, opening `memory` over the recorded table. */
function drive(built: FlowEngine, memory: ReadonlyMap<string, FakeMemory>, feedback: FeedbackValue = "review") {
  return driveBackward(built.state, built.log, feedback, {
    openMemory: () => open(built.state.memories ?? [], memory),
  });
}

/** Return each record's class. */
function classes(items: readonly RunRecord[]): unknown[] {
  return items.map((record) => record.constructor);
}

test("backward finishes all callbacks before Ready and policy requests", async () => {
  const { engine: built, node } = await terminal(["A", "B", "C"]);
  const visited: string[] = [];
  const learn = (context: LearnContext) => {
    expect(node.calls).toHaveLength(1);
    expect(records(built).at(-1)!.constructor).toBe(BackwardStartedRecord);
    expect([...memory.values()].every((backend) => backend.commitCalls.length === 0)).toBe(true);
    visited.push(context.memory);
    return [new MemoryMutation(String(visited.length), { notes: ["one"] })];
  };
  const memory = new Map(["A", "B", "C"].map((key) => [key, new FakeMemory({ learnOutcomes: [learn] })]));
  const originalCommit = FakeSession.prototype.commit;
  const commit = spyOn(FakeSession.prototype, "commit").mockImplementation(function (
    this: FakeSession,
    commitId: string,
    mutations: readonly MemoryMutation[],
  ) {
    expect(visited).toEqual(["A", "B", "C"]);
    const last = records(built).at(-1)!;
    expect(last.constructor).toBe(MemoryCommitRequestedRecord);
    expect((last as MemoryCommitRequestedRecord).attemptId).toBe(1);
    expect(built.state.backwardState.ready).not.toBeNull();
    return originalCommit.call(this, commitId, mutations);
  });
  const forwardPool = encodePool(built.state.pool);
  let result;
  try {
    result = await drive(built, memory);
  } finally {
    commit.mockRestore();
  }
  expect(result.status).toBe("succeeded");
  expect(result.feedback!.inputFeedback).toEqual({ topic: ["review"] });
  expect(result.learning!.policies.map((policy) => policy.status)).toEqual(["succeeded", "succeeded", "succeeded"]);
  expect(["A", "B", "C"].map((key) => memory.get(key)!.commitCalls[0]![1].length)).toEqual([1, 1, 1]);
  expect(memory.get("A")!.learnContexts[0]!.episodes).toHaveLength(3);
  expect(memory.get("B")!.learnContexts[0]!.episodes).toEqual([]);
  expect(memory.get("C")!.learnContexts[0]!.episodes).toEqual([]);
  for (const backend of memory.values()) {
    expect(backend.openSnapshots).toEqual([{}]);
    expect(backend.closeCount).toBe(1);
  }
  expect(encodePool(built.state.pool)).toEqual(forwardPool);
  expect(built.state.trace!.nextSequence).toBe(built.state.nextSequence);
  expect(built.state.nextSequence).toBe(built.log.nextSequence);
  expect(records(built).at(-1)!.constructor).toBe(BackwardFinishedRecord);
});

test("a late proposal failure discards all staged mutations", async () => {
  for (const failureKind of ["backend", "duplicate", "tuple", "forged"]) {
    const { engine: built, node } = await terminal(["A", "B", "C"]);
    const prior = new MemoryMutation("same", { lesson: "first" });
    const fail = () => {
      if (failureKind === "backend") {
        throw new MemoryOperationError("provider", "backward.interrupted", "Backend declined.", true);
      }
      if (failureKind === "duplicate") return [new MemoryMutation("same", {}), new MemoryMutation("same", {})];
      if (failureKind === "tuple") return new Set();
      // A mutation can't be forged in place: it is frozen.
      return [{ mutationId: true, payload: {} }];
    };
    const memory = new Map([
      ["A", new FakeMemory({ learnOutcomes: [[prior]] })],
      ["B", new FakeMemory({ learnOutcomes: [fail] })],
      ["C", new FakeMemory()],
    ]);
    const outcome = await drive(built, memory);
    expect(outcome.status).toBe("failed");
    expect(outcome.failure!.stage).toBe("learn");
    expect(outcome.failure!.memory).toBe("B");
    expect(outcome.failure!.code).toBe(failureKind === "backend" ? "backward.interrupted" : "memory.invalid_result");
    expect(outcome.feedback!.inputFeedback).toEqual({ topic: ["review"] });
    expect(outcome.learning!.status).toBe("failed");
    expect(outcome.learning!.policies.map((policy) => policy.status)).toEqual(Array(3).fill("unrequested"));
    const backward = built.state.backwardState;
    expect(backward.finished!.attributedResults).toHaveLength(built.state.trace!.occurrences.length);
    expect(backward.ready).toBeNull();
    expect(backward.canStart).toBe(true);
    expect(node.calls).toHaveLength(1);
    expect(memory.get("A")!.learnContexts).toHaveLength(1);
    expect(memory.get("B")!.learnContexts).toHaveLength(1);
    expect(memory.get("C")!.learnContexts).toEqual([]);
    for (const backend of memory.values()) {
      expect(backend.commitCalls).toEqual([]);
      expect(backend.closeCount).toBe(1);
    }
  }
});

test("a policy failure preserves prior success and stops later policies", async () => {
  for (const invalidResult of [false, true]) {
    const { engine: built, node } = await terminal(["A", "B", "C"]);
    const failed = invalidResult
      ? {}
      : new MemoryOperationError("storage", "backward.interrupted", "Store refused.", true);
    const memory = new Map([
      ["A", new FakeMemory()],
      ["B", new FakeMemory({ commitOutcomes: [failed] })],
      ["C", new FakeMemory()],
    ]);
    const outcome = await drive(built, memory);
    expect(outcome.status).toBe("failed");
    expect(outcome.failure!.stage).toBe("commit");
    expect(outcome.failure!.memory).toBe("B");
    expect(outcome.failure!.code).toBe(invalidResult ? "memory.invalid_result" : "backward.interrupted");
    expect(outcome.learning!.status).toBe("failed");
    expect(outcome.learning!.policies.map((policy) => policy.status)).toEqual(["succeeded", "failed", "unrequested"]);
    expect(["A", "B", "C"].map((name) => memory.get(name)!.commitCalls.length)).toEqual([1, 1, 0]);
    for (const backend of memory.values()) {
      expect(backend.learnContexts).toHaveLength(1);
      expect(backend.closeCount).toBe(1);
    }
    expect(node.calls).toHaveLength(1);
    expect(built.state.backwardState.canStart).toBe(true);
    expect(built.state.backwardState.finished!.attributedResults).toEqual([]);
    expect(built.state.backwardState.ready!.attributedResults).toHaveLength(3);
  }
});

test("empty feedback still learns empty memories and commits empty batches", async () => {
  const { engine: built, node } = await terminal(["A", "B"]);
  const memory = new Map([
    ["A", new FakeMemory()],
    ["B", new FakeMemory()],
  ]);
  const outcome = await drive(built, memory, null);
  expect(node.calls).toEqual([]);
  expect(outcome.status).toBe("succeeded");
  expect(outcome.feedback).toEqual(new BackwardResult());
  expect(["A", "B"].map((name) => memory.get(name)!.learnContexts.length)).toEqual([1, 1]);
  for (const backend of memory.values()) {
    expect(backend.learnContexts.every((context) => context.feedback.length === 0)).toBe(true);
    expect(backend.commitCalls).toHaveLength(1);
    expect(backend.commitCalls[0]![1]).toEqual([]);
  }
});

test("a backward without policies still records Ready and Finished without opening resources", async () => {
  const { engine: built, node } = await terminal([]);
  const before = records(built).length;
  const outcome = await driveBackward(built.state, built.log, "review", {
    openMemory: () => {
      throw new Error("no enabled policy");
    },
  });
  expect(outcome.status).toBe("succeeded");
  expect(outcome.learning).toBeNull();
  expect(outcome.feedback!.inputFeedback).toEqual({ topic: ["review"] });
  expect(node.calls).toHaveLength(1);
  expect(classes(records(built).slice(before))).toEqual([
    BackwardStartedRecord,
    BackwardReadyRecord,
    BackwardFinishedRecord,
  ]);
  const again = await driveBackward(built.state, built.log, "again");
  expect(again.status).toBe("succeeded");
  expect(again.feedback!.inputFeedback).toEqual({ topic: ["again"] });
  expect(node.calls).toHaveLength(2);
});

test("a drive hands its models and onEvent to the backward hook", async () => {
  const received: (BackwardOptions | undefined)[] = [];
  class OptionsNode extends RecordingBackwardNode {
    override backward(context: BackwardContext, feedback: readonly FeedbackValue[], options?: BackwardOptions) {
      received.push(options);
      return super.backward(context, feedback);
    }
  }
  const node = new OptionsNode(
    "work",
    (inputs) => inputs.topic,
    (_context, feedback) => new BackwardResult({ inputFeedback: { topic: feedback } }),
    { reads: { topic: "${input.topic}" } },
  );
  const flow = graph([node], [[START_ID, "work"]], { outputs: [new FlowOutput("answer", "${work.output}")] });
  const built = engine(flow, { inputs: { topic: "reefs" }, learnable: true });
  await built.run();
  const models: ModelAccess = {
    env: NO_ENV,
    createModel: () => {
      throw new Error("the hook builds no model");
    },
  };
  const onEvent = () => {};

  const outcome = await driveBackward(built.state, built.log, "review", { models, onEvent });

  expect(outcome.status).toBe("succeeded");
  expect(received).toHaveLength(1);
  expect(received[0]!.models).toBe(models);
  expect(received[0]!.onEvent).toBe(onEvent);
});

/**
 * Spy on `LearnContext.fromFrozen`, throwing `RangeError("context too deep")` on its first call
 * (START's occurrence, seq 1, under `terminal`'s every-node scope) and passing the others through;
 * restore it with `mockRestore()`.
 */
function failFirstContext() {
  const fromFrozen = LearnContext.fromFrozen;
  let calls = 0;
  return spyOn(LearnContext, "fromFrozen").mockImplementation((...args) => {
    calls += 1;
    if (calls === 1) throw new RangeError("context too deep");
    return fromFrozen.apply(LearnContext, args);
  });
}

test("a context failure keeps the completed backward without calling learners", async () => {
  const { engine: built, node } = await terminal();
  const memory = new FakeMemory();
  const spy = failFirstContext();
  let outcome: BackwardOutcome;
  try {
    outcome = await drive(built, new Map([["A", memory]]));
  } finally {
    spy.mockRestore();
  }
  const failure = outcome.failure!;
  expect(failure.stage).toBe("learn");
  expect(failure.code).toBe("memory.invalid_context");
  expect(failure.memory).toBe(built.state.memories![0]!.key);
  const context = built.state.trace!.backwardContext(new SourceRef(built.state.runId!, 1));
  expect(failure.runId).toBe(built.state.runId);
  expect(failure.nodeId).toBe(context.nodeId);
  expect(failure.nodeStartedSeq).toBe(1);
  expect(failure.scope).toEqual(context.scope);
  expect(failure.message).toBe("memory learn context could not be constructed for node '__start__': context too deep");
  expect(failure.cause!.message).toBe(failure.message);
  expect(outcome.feedback!.inputFeedback).toEqual({ topic: ["review"] });
  expect(built.state.backwardState.finished!.attributedResults).toHaveLength(3);
  expect(node.calls).toHaveLength(1);
  expect(memory.learnContexts).toEqual([]);
  expect(memory.commitCalls).toEqual([]);
  expect(memory.closeCount).toBe(1);
  expect(built.state.backwardState.canStart).toBe(true);
});

/** Return a Finished record for a fresh attempt that ends with `failure`. */
function contextFailureRetry(built: FlowEngine, failure: BackwardFailure): BackwardFinishedRecord {
  const first = built.state.backwardState.finished!;
  commitPrepared(built.log, built.state, prepareRecord(built.state, new BackwardStartedRecord(2, "review")));
  const learning = prefixLearning(built.state.backwardState, {
    learnerFailure: new LearningFailure(failure.memory!, "learn", failure.cause as MemoryOperationFailure),
  });
  return new BackwardFinishedRecord(
    2,
    new BackwardOutcome("failed", { feedback: first.outcome.feedback, failure, learning }),
    first.attributedResults,
  );
}

test("a replayed context failure must name an accepted receiving occurrence", async () => {
  const mismatch = "a context failure must match its recorded occurrence";
  const foreign = "an attributed learner failure must be a context failure of this run";
  const messages: Record<string, string> = {
    node: mismatch,
    source: "a context failure must name an accepted occurrence",
    memory_scope: "a context failure must name an occurrence that receives its memory",
    scope: mismatch,
    run: foreign,
    code: foreign,
  };
  for (const change of [null, "node", "source", "memory_scope", "scope", "run", "code"]) {
    const { engine: built } = await terminal(["A", "B"]);
    const spy = failFirstContext();
    let failure: BackwardFailure;
    try {
      failure = (await drive(built, new Map([["A", new FakeMemory()], ["B", new FakeMemory()]]))).failure!;
    } finally {
      spy.mockRestore();
    }
    expect(failure.nodeStartedSeq).toBe(1);
    let cause = failure.cause as MemoryOperationFailure;
    let fields = {
      code: failure.code,
      scope: failure.scope,
      runId: failure.runId,
      nodeId: failure.nodeId,
      nodeStartedSeq: failure.nodeStartedSeq,
      memory: failure.memory,
    };
    if (change === "node") fields = { ...fields, nodeId: "other" };
    // Seq 2 is START's success record, not an accepted occurrence start.
    else if (change === "source") fields = { ...fields, nodeStartedSeq: 2 };
    // Every node's memory scope holds only "A".
    else if (change === "memory_scope") fields = { ...fields, memory: "B" };
    else if (change === "scope") {
      const later = built.state.trace!.backwardContext(new SourceRef(built.state.runId!, 3));
      fields = { ...fields, scope: later.scope };
    } else if (change === "run") fields = { ...fields, runId: "other" };
    else if (change === "code") {
      cause = new MemoryOperationFailure(cause.stage, "learn.failed", cause.message, cause.retryable);
      fields = { ...fields, code: cause.code };
    }
    const { code, ...options } = fields;
    const record = contextFailureRetry(
      built,
      new BackwardFailure(failure.stage, code, failure.message, { ...options, cause }),
    );
    if (change === null) {
      prepareRecord(built.state, record);
      continue;
    }
    const error = (() => {
      try {
        prepareRecord(built.state, record);
      } catch (caught) {
        return caught as Error;
      }
    })();
    expect(error?.constructor).toBe(RangeError);
    expect(error!.message).toBe(messages[change]!);
  }
});

test("a computation failure finishes without partial feedback or learning", async () => {
  for (const invalidResult of [false, true]) {
    const callback = () => {
      if (invalidResult) return new BackwardResult({ inputFeedback: { undeclared: ["no"] } });
      throw new Error("Callback failed.");
    };
    const { engine: built, node } = await terminal(["A"], { callback });
    const memory = new FakeMemory();
    const outcome = await drive(built, new Map([["A", memory]]));
    expect(outcome.failure!.code).toBe(invalidResult ? "backward.result" : "backward.execute");
    expect(outcome.feedback).toBeNull();
    expect(built.state.backwardState.finished!.attributedResults).toEqual([]);
    expect(outcome.learning!.status).toBe("incomplete");
    expect(outcome.learning!.policies[0]!.status).toBe("unrequested");
    expect(node.calls).toHaveLength(1);
    expect(memory.learnContexts).toEqual([]);
    expect(memory.commitCalls).toEqual([]);
    expect(memory.closeCount).toBe(1);
  }
});

test("an append fault preserves the exact prefix and closes resources", async () => {
  const phases: [unknown, string | null][] = [
    [BackwardStartedRecord, null],
    [BackwardReadyRecord, null],
    [MemoryCommitRequestedRecord, null],
    [MemoryCommitSucceededRecord, null],
    [MemoryCommitFailedRecord, "commit"],
    [BackwardFinishedRecord, null],
    [BackwardFinishedRecord, "learn"],
    [BackwardFinishedRecord, "commit"],
  ];
  for (const [recordType, knownFailure] of phases) {
    const { engine: built, node } = await terminal(["A", "B"]);
    const typed = new MemoryOperationError("backend", "declined", "Declined.", false);
    const memory = new Map([
      [
        "A",
        new FakeMemory({
          learnOutcomes: knownFailure === "learn" ? [typed] : [],
          commitOutcomes: knownFailure === "commit" ? [typed] : [],
        }),
      ],
      ["B", new FakeMemory()],
    ]);
    const originalAppend = built.log.append.bind(built.log);
    const error = new Error("append unavailable");
    const refused: RunRecord[] = [];
    const forwardResult = built.state.result;
    const forwardPool = encodePool(built.state.pool);
    const append = spyOn(built.log, "append").mockImplementation((record) => {
      if (record.constructor === recordType) {
        refused.push(record);
        throw error;
      }
      return originalAppend(record);
    });
    let caught: unknown;
    try {
      caught = await rejected(drive(built, memory));
    } finally {
      append.mockRestore();
    }
    expect(caught).toBe(error);
    expect(refused).toHaveLength(1);
    expect(built.state.result).toBe(forwardResult);
    expect(encodePool(built.state.pool)).toEqual(forwardPool);
    expect(built.state.nextSequence).toBe(built.state.trace!.nextSequence);
    expect(built.state.nextSequence).toBe(built.log.nextSequence);
    expect(built.state.backwardState.finished).toBeNull();
    for (const backend of memory.values()) expect(backend.closeCount).toBe(1);
    const current = built.state.backwardState;
    if (recordType === BackwardStartedRecord) {
      expect(current.started).toBeNull();
      expect(node.calls).toEqual([]);
      expect(memory.get("A")!.learnContexts).toEqual([]);
    } else {
      expect(current.started).not.toBeNull();
      expect(node.calls).toHaveLength(1);
    }
    if (recordType === BackwardStartedRecord || recordType === BackwardReadyRecord || knownFailure === "learn") {
      expect(current.ready).toBeNull();
      expect(memory.get("A")!.commitCalls).toEqual([]);
    } else if (recordType === MemoryCommitRequestedRecord) {
      expect(current.ready).not.toBeNull();
      expect(current.everRequested).toBe(false);
      expect(memory.get("A")!.commitCalls).toEqual([]);
    } else if (recordType === MemoryCommitSucceededRecord || recordType === MemoryCommitFailedRecord) {
      expect(current.policyOutcomes[0]!.status).toBe("unknown");
      expect(memory.get("A")!.commitCalls).toHaveLength(1);
      expect(memory.get("B")!.commitCalls).toEqual([]);
    } else {
      expect(current.outcome!.failure!.code).toBe("backward.interrupted");
      expect(current.policyOutcomes[0]!.status).toBe(knownFailure === "commit" ? "failed" : "succeeded");
    }
  }
});

test("a manual precommit retry reopens snapshots and recomputes every memory", async () => {
  for (const failureRecord of [BackwardReadyRecord, MemoryCommitRequestedRecord] as unknown[]) {
    const { engine: built, node } = await terminal(["A"]);
    const visits: string[] = [];
    const learn = (context: LearnContext) => {
      visits.push(context.memory);
      return [new MemoryMutation(String(visits.length), {})];
    };
    const memory = new FakeMemory({ learnOutcomes: [learn, learn] });
    const originalAppend = built.log.append.bind(built.log);
    const append = spyOn(built.log, "append").mockImplementation((record) => {
      if (record.constructor === failureRecord) throw new Error("interrupted before request");
      return originalAppend(record);
    });
    let caught: unknown;
    try {
      caught = await rejected(drive(built, new Map([["A", memory]]), "first"));
    } finally {
      append.mockRestore();
    }
    expect((caught as Error).message).toBe("interrupted before request");
    const old = built.state.backwardState;
    expect(old.finished).toBeNull();
    expect(old.everRequested).toBe(false);
    const before = records(built).length;
    const invalid = await rejected(drive(built, new Map([["A", memory]]), []));
    expect((invalid as Error).constructor).toBe(RangeError);
    expect((invalid as Error).message).toBe("feedback requires a recorded list");
    expect(built.state.backwardState).toBe(old);
    expect(records(built)).toHaveLength(before);
    expect(memory.openSnapshots).toEqual([{}]);
    expect(node.calls).toHaveLength(1);

    const outcome = await drive(built, new Map([["A", memory]]), "second");
    expect(outcome.status).toBe("succeeded");
    expect(outcome.feedback!.inputFeedback).toEqual({ topic: ["second"] });
    expect(memory.openSnapshots).toEqual([{}, {}]);
    expect(memory.closeCount).toBe(2);
    const order = built.state.backwardState.memoryOrder;
    expect(visits).toEqual([...order, ...order]);
    expect(node.calls).toHaveLength(2);
    expect(memory.commitCalls[0]![1].map((mutation) => mutation.mutationId)).toEqual(["2"]);
    expect(built.state.backwardState.attempts).toHaveLength(2);
    const first = built.state.backwardState.attempts[0]!.finished!.outcome;
    expect(first.failure!.code).toBe("backward.interrupted");
    expect(first.feedback).toEqual(old.ready !== null ? old.ready.rootResult : null);
    expect(classes(records(built).slice(before, before + 2))).toEqual([BackwardFinishedRecord, BackwardStartedRecord]);
    expect(built.state.backwardState.started!.attemptId).toBe(2);
  }
});

test("a known precommit failure allows a fresh complete pass without prior proposals", async () => {
  const { engine: built, node } = await terminal(["A", "B"]);
  const proposal = (name: string) => () => [new MemoryMutation(name, {})];
  const memory = new Map([
    ["A", new FakeMemory({ learnOutcomes: [proposal("old"), proposal("new-first")] })],
    [
      "B",
      new FakeMemory({
        learnOutcomes: [new MemoryOperationError("learn", "rejected", "Rejected.", false), proposal("new-second")],
      }),
    ],
  ]);
  expect((await drive(built, memory)).status).toBe("failed");
  expect(built.state.backwardState.canStart).toBe(true);
  expect((await drive(built, memory)).status).toBe("succeeded");
  expect(node.calls).toHaveLength(2);
  for (const backend of memory.values()) {
    expect(backend.learnContexts).toHaveLength(2);
    expect(backend.openSnapshots).toEqual([{}, {}]);
    expect(backend.closeCount).toBe(2);
  }
  expect(memory.get("A")!.commitCalls[0]![1].map((mutation) => mutation.mutationId)).toEqual(["new-first"]);
  expect(memory.get("B")!.commitCalls[0]![1].map((mutation) => mutation.mutationId)).toEqual(["new-second"]);
});

test("a failed resource setup does not close an interrupted attempt", async () => {
  const mismatch = "opened memory runtime does not match the recorded memory table and snapshots";
  const rows: [string, new (...args: never[]) => Error, string][] = [
    ["table", RangeError, mismatch],
    ["snapshots", RangeError, mismatch],
    ["opener", Error, "cannot reopen"],
  ];
  for (const [problem, cls, message] of rows) {
    const { engine: built, node } = await terminal();
    commitPrepared(built.log, built.state, prepareRecord(built.state, new BackwardStartedRecord(1, "first")));
    const old = built.state.backwardState;
    const sequence = built.state.nextSequence;
    const memory = new FakeMemory();
    const openMemory = () => {
      if (problem === "opener") throw new Error("cannot reopen");
      const memories = problem === "table" ? table("B") : built.state.memories!;
      return MemoryRuntime.open({
        memories,
        origins: new Map(memories.map((spec) => [spec.key, null])),
        memory: new Map(memories.map((spec) => [spec.key, memory])),
        snapshots: [new MemorySnapshotEnvelope(memories[0]!.key, problem === "snapshots" ? { changed: true } : {})],
      });
    };
    const error = await rejected(driveBackward(built.state, built.log, "second", { openMemory }));
    expect((error as Error).constructor).toBe(cls);
    expect((error as Error).message).toBe(message);
    expect(built.state.backwardState).toBe(old);
    expect(built.state.nextSequence).toBe(sequence);
    expect(node.calls).toEqual([]);
    expect(memory.learnContexts).toEqual([]);
    expect(memory.commitCalls).toEqual([]);
    expect(memory.closeCount).toBe(problem === "opener" ? 0 : 1);
  }
});

test("a backend host or unexpected error escapes with the exact prefix", async () => {
  for (const operation of ["learn", "commit"]) {
    for (const cls of [Error, RangeError]) {
      const { engine: built, node } = await terminal(["A", "B"]);
      const error = new cls("host stopped");
      const memory = new Map([
        [
          "A",
          new FakeMemory({
            learnOutcomes: operation === "learn" ? [error] : [],
            commitOutcomes: operation === "commit" ? [error] : [],
          }),
        ],
        ["B", new FakeMemory()],
      ]);
      expect(await rejected(drive(built, memory))).toBe(error);
      expect(node.calls).toHaveLength(1);
      const backward = built.state.backwardState;
      expect(backward.finished).toBeNull();
      expect(backward.policyOutcomes[0]!.status).toBe(operation === "learn" ? "unrequested" : "unknown");
      expect(memory.get("B")!.commitCalls).toEqual([]);
      for (const backend of memory.values()) expect(backend.closeCount).toBe(1);
      if (operation === "commit") {
        expect((await drive(built, memory, "after host stop")).status).toBe("succeeded");
        const finished = records(built).filter(
          (record) => record.constructor === BackwardFinishedRecord,
        ) as BackwardFinishedRecord[];
        expect(finished.map((record) => record.attemptId)).toEqual([1, 2]);
        expect(finished[0]!.outcome.failure!.code).toBe("backward.interrupted");
        expect(finished[0]!.outcome.learning!.policies.map((policy) => policy.status)).toEqual([
          "unknown",
          "unrequested",
        ]);
        for (const backend of memory.values()) expect(backend.openSnapshots).toEqual([{}, {}]);
      }
    }
  }
});

test("a host capability fault escapes as itself, with cleanup errors noted on it", async () => {
  const { engine: built } = await terminal(["A", "B"]);
  const fault = new CapabilityFault(new Error("capability failed"));
  const memory = new Map([
    ["A", new FakeMemory({ learnOutcomes: [fault] })],
    ["B", new FakeMemory()],
  ]);
  const originalClose = FakeSession.prototype.close;
  const close = spyOn(FakeSession.prototype, "close").mockImplementation(function (this: FakeSession) {
    originalClose.call(this);
    throw new Error("close failed");
  });
  let caught: unknown;
  try {
    caught = await rejected(drive(built, memory));
  } finally {
    close.mockRestore();
  }
  expect(caught).toBe(fault);
  expect((caught as { notes?: unknown }).notes).toEqual([
    "memory 'B' session close also raised Error: close failed",
    "memory 'A' session close also raised Error: close failed",
  ]);
  for (const backend of memory.values()) expect(backend.closeCount).toBe(1);
  expect(built.state.backwardState.finished).toBeNull();
  expect(built.state.backwardState.everRequested).toBe(false);
});

test("a later attempt follows or closes any prior attempt and recomputes everything", async () => {
  for (const prior of ["succeeded", "commit_failed", "requested_unfinished"]) {
    const { engine: built, node } = await terminal(["A"]);
    if (prior === "requested_unfinished") {
      const state = built.state;
      const [root, attributed] = await computeBackward(state.trace!, "review");
      const proposals = state.memories!.map((spec) => new MemoryProposal(spec.key, []));
      const identity = memoryCommitId(state.runId!, "A", [], 1);
      for (const record of [
        new BackwardStartedRecord(1, "review"),
        new BackwardReadyRecord(1, root, attributed, proposals),
        new MemoryCommitRequestedRecord("A", identity, [], 1),
      ]) {
        commitPrepared(built.log, state, prepareRecord(state, record));
      }
    } else {
      const first = new FakeMemory({
        commitOutcomes:
          prior === "commit_failed" ? [new MemoryOperationError("storage", "store.moved", "Moved.", true)] : [],
      });
      expect((await drive(built, new Map([["A", first]]))).status).toBe(prior === "succeeded" ? "succeeded" : "failed");
    }
    const calls = node.calls.length;
    const later = new FakeMemory();
    const outcome = await drive(built, new Map([["A", later]]), "later review");
    expect(outcome.status).toBe("succeeded");
    expect(outcome.learning!.status).toBe("succeeded");
    const all = records(built);
    const started = all.filter((record) => record.constructor === BackwardStartedRecord) as BackwardStartedRecord[];
    expect(started.map((record) => record.attemptId)).toEqual([1, 2]);
    const finished = all.filter((record) => record.constructor === BackwardFinishedRecord) as BackwardFinishedRecord[];
    expect(finished.map((record) => record.attemptId)).toEqual([1, 2]);
    if (prior === "requested_unfinished") {
      expect(finished[0]!.outcome.failure!.code).toBe("backward.interrupted");
      expect(finished[0]!.outcome.learning!.policies.map((policy) => policy.status)).toEqual(["unknown"]);
    }
    expect(node.calls).toHaveLength(calls + 1);
    expect(later.learnContexts).toHaveLength(1);
    expect(later.commitCalls).toHaveLength(1);
    expect(later.closeCount).toBe(1);
    expect(built.state.backwardState.canStart).toBe(true);
  }
});

test("repeated driven attempts request distinct commit ids for identical batches", async () => {
  const { engine: built } = await terminal(["A"]);
  const memory = new FakeMemory();
  for (const feedback of ["first", "second", "third"]) {
    expect((await drive(built, new Map([["A", memory]]), feedback)).status).toBe("succeeded");
  }
  const requests = records(built).filter(
    (record) => record.constructor === MemoryCommitRequestedRecord,
  ) as MemoryCommitRequestedRecord[];
  expect(requests.map((record) => record.attemptId)).toEqual([1, 2, 3]);
  expect(requests.every((record) => record.mutations.length === 0)).toBe(true);
  expect(memory.commitCalls.map(([commitId]) => commitId)).toEqual(requests.map((record) => record.commitId));
  expect(new Set(requests.map((record) => record.commitId)).size).toBe(3);
  expect(requests[0]!.commitId.startsWith("ac-memory-v3:")).toBe(true);
});

test("a cleanup error after Finished preserves success and admits a later attempt", async () => {
  const { engine: built } = await terminal();
  const memory = new FakeMemory();
  const originalClose = FakeSession.prototype.close;
  const close = spyOn(FakeSession.prototype, "close").mockImplementation(function (this: FakeSession) {
    originalClose.call(this);
    throw new Error("close failed");
  });
  let caught: unknown;
  try {
    caught = await rejected(drive(built, new Map([["A", memory]])));
  } finally {
    close.mockRestore();
  }
  expect((caught as Error).message).toBe("close failed");
  expect(built.state.backwardState.finished!.outcome.status).toBe("succeeded");
  expect(built.state.backwardState.canStart).toBe(true);
  expect(memory.closeCount).toBe(1);
});

test("backward and memory replay call no callbacks or materialization", async () => {
  const { engine: built } = await terminal(["A", "B"]);
  const original = copyGraph(built.state.flow);
  const memory = new Map([
    ["A", new FakeMemory()],
    [
      "B",
      new FakeMemory({ commitOutcomes: [new MemoryOperationError("commit", "not.available", "Unavailable.", false)] }),
    ],
  ]);
  await drive(built, memory);
  const fail = (what: string) => () => {
    throw new Error(`replayed ${what}`);
  };
  for (const node of original.nodes.values()) {
    Object.assign(node, {
      forward: fail("forward"),
      backward: fail("backward"),
      captureBackwardContext: fail("capture"),
    });
  }
  const propose = spyOn(MemoryRuntime.prototype, "proposeLearning").mockImplementation(fail("learner"));
  const commit = spyOn(MemoryRuntime.prototype, "commitLearning").mockImplementation(fail("commit"));
  let replayed: RunState;
  try {
    replayed = replayRecords(new RunState(original), built.log.records(), { growBuilder: noGrowth });
  } finally {
    propose.mockRestore();
    commit.mockRestore();
  }
  expect(sameRuntimeBackwardState(replayed.backwardState, built.state.backwardState)).toBe(true);
  expect(replayed.trace!.occurrences).toEqual(built.state.trace!.occurrences);
  expect(replayed.nextSequence).toBe(replayed.trace!.nextSequence);
  expect(replayed.nextSequence).toBe(built.state.nextSequence);
});

test("a prepare failure is not sanitized as a learner failure", async () => {
  const { engine: built } = await terminal();
  const memory = new FakeMemory();
  const error = new MemoryOperationError("host", "prepare.failed", "Prepare failed.", false);
  // `prepareRecord` is a module function that can't be patched, so the Ready record's append throws.
  const originalAppend = built.log.append.bind(built.log);
  const append = spyOn(built.log, "append").mockImplementation((record) => {
    if (record.constructor === BackwardReadyRecord) throw error;
    return originalAppend(record);
  });
  let caught: unknown;
  try {
    caught = await rejected(drive(built, new Map([["A", memory]])));
  } finally {
    append.mockRestore();
  }
  expect(caught).toBe(error);
  expect(built.state.backwardState.ready).toBeNull();
  expect(built.state.backwardState.finished).toBeNull();
  expect(memory.commitCalls).toEqual([]);
  expect(memory.closeCount).toBe(1);
});

test("a learner's edits to its arguments and returned values cannot change authority", async () => {
  const { engine: built } = await terminal(["A", "B"]);
  const returned = new MemoryMutation("first", { items: ["saved"] });
  const learnFirst = (context: LearnContext) => {
    expect(() => {
      (context as { memory: string }).memory = "wrong";
    }).toThrow(TypeError);
    return [returned];
  };
  const learnSecond = () => {
    expect(() => {
      (returned as { mutationId: string }).mutationId = "changed-later";
    }).toThrow(TypeError);
    return [];
  };
  const memory = new Map([
    ["A", new FakeMemory({ learnOutcomes: [learnFirst] })],
    ["B", new FakeMemory({ learnOutcomes: [learnSecond] })],
  ]);
  const outcome = await drive(built, memory);
  expect(outcome.status).toBe("succeeded");
  const ready = records(built).find((record) => record.constructor === BackwardReadyRecord) as BackwardReadyRecord;
  expect(ready.proposals.map((proposal) => proposal.memory)).toEqual(["A", "B"]);
  expect(memory.get("A")!.commitCalls[0]![1][0]!.mutationId).toBe("first");
  expect(memory.get("A")!.commitCalls[0]![1][0]!.payload).toEqual({ items: ["saved"] });
});

test("an opened MemoryRuntime subclass is closed by the driver", async () => {
  class SpecializedRuntime extends MemoryRuntime {}

  const { engine: built } = await terminal();
  const memory = new FakeMemory();
  const memories = built.state.memories!;
  const runtime = await SpecializedRuntime.open({
    memories,
    origins: new Map(memories.map((spec) => [spec.key, null])),
    memory: new Map([["A", memory]]),
    snapshots: built.state.memorySnapshots,
  });
  expect(runtime).toBeInstanceOf(SpecializedRuntime);
  const outcome = await driveBackward(built.state, built.log, "review", { openMemory: () => runtime });
  expect(outcome.status).toBe("succeeded");
  expect(memory.closeCount).toBe(1);
});

test("submission gates reject before opening or appending", async () => {
  const rows: [string, string][] = [
    ["learn_disabled", "backward requires a retained terminal trace"],
    ["missing_opener", "enabled backward memories require a fresh memory runtime"],
    ["invalid_feedback", "feedback requires a recorded list"],
  ];
  for (const [problem, message] of rows) {
    const { engine: built, node } = await terminal(["A"], { learnable: problem !== "learn_disabled" });
    const memory = new FakeMemory();
    const before = records(built);
    const error = await rejected(
      driveBackward(built.state, built.log, problem === "invalid_feedback" ? [] : "review", {
        openMemory:
          problem === "missing_opener" ? null : () => open(built.state.memories!, new Map([["A", memory]])),
      }),
    );
    expect((error as Error).constructor).toBe(RangeError);
    expect((error as Error).message).toBe(message);
    expect(records(built)).toEqual(before);
    expect(node.calls).toEqual([]);
    expect(memory.openSnapshots).toEqual([]);
  }
});

test("feedback the state can't prepare is refused before the opener runs", async () => {
  // A lone surrogate passes the feedback check but isn't canonical JSON, so only the gate refuses it.
  const { engine: built, node } = await terminal();
  const before = records(built);
  let opened = 0;
  const error = await rejected(
    driveBackward(built.state, built.log, "\uD800", {
      openMemory: () => {
        opened += 1;
        return open(built.state.memories!, new Map([["A", new FakeMemory()]]));
      },
    }),
  );
  expect((error as Error).constructor).toBe(TypeError);
  expect((error as Error).message).toBe("Unicode surrogate is not valid canonical JSON");
  expect(opened).toBe(0);
  expect(records(built)).toEqual(before);
  expect(node.calls).toEqual([]);
});

test("a wrapped host error during partial setup closes prior sessions", async () => {
  const { engine: built } = await terminal(["A", "B"]);
  const memory = new Map([
    ["A", new FakeMemory()],
    ["B", new FakeMemory()],
  ]);
  const wrapped = new MemoryOperationError("open", "wrapped", "Open failed.", false);
  memory.get("B")!.openRun = () => {
    throw wrapped;
  };
  const before = records(built);
  const error = await rejected(drive(built, memory));
  expect((error as Error).constructor).toBe(MemorySetupError);
  expect((error as Error).message).toBe("memory 'B' could not open its run session: Open failed.");
  expect((error as Error).cause).toBe(wrapped);
  expect(records(built)).toEqual(before);
  expect(memory.get("A")!.closeCount).toBe(1);
  expect(memory.get("B")!.closeCount).toBe(0);
});

test("cleanup closes every session and keeps the primary error or the close failure", async () => {
  for (const withPrimary of [false, true]) {
    const { engine: built } = await terminal(["A", "B", "C"]);
    const primary = new Error("learner failed unexpectedly");
    const memory = new Map([
      ["A", new FakeMemory({ learnOutcomes: withPrimary ? [primary] : [] })],
      ["B", new FakeMemory()],
      ["C", new FakeMemory()],
    ]);
    const failure = new MemoryOperationError("close", "wrapped", "Close failed.", false);
    const closed: string[] = [];
    const originalClose = FakeSession.prototype.close;
    const close = spyOn(FakeSession.prototype, "close").mockImplementation(function (this: FakeSession) {
      originalClose.call(this);
      const name = [...memory].find(([, backend]) => backend === this.memory)![0];
      closed.push(name);
      if (name === "B") throw failure;
    });
    let caught: unknown;
    try {
      caught = await rejected(drive(built, memory));
    } finally {
      close.mockRestore();
    }
    expect(caught).toBe(withPrimary ? primary : failure);
    expect(closed).toEqual(["C", "B", "A"]);
    for (const backend of memory.values()) expect(backend.closeCount).toBe(1);
    if (withPrimary) {
      expect((caught as { notes?: unknown }).notes).toEqual([
        "memory 'B' session close also raised MemoryOperationError: Close failed.",
      ]);
      expect(built.state.backwardState.finished).toBeNull();
      expect(built.state.backwardState.everRequested).toBe(false);
    } else {
      expect(built.state.backwardState.finished!.outcome.status).toBe("succeeded");
      expect(built.state.backwardState.canStart).toBe(true);
    }
  }
});

test("a commit batch is detached from every accepted authority", async () => {
  for (const target of ["mutation", "public_payload"]) {
    for (const completion of ["success", "known_failure", "host_failure"]) {
      const { engine: built } = await terminal();
      const known = new MemoryOperationError("commit", "refused", "Commit refused.", false);
      const host = new Error("commit interrupted");
      const memory = new FakeMemory({
        learnOutcomes: [() => [new MemoryMutation("proposal", { notes: ["saved"] })]],
        commitOutcomes: completion === "known_failure" ? [known] : completion === "host_failure" ? [host] : [],
      });
      const originalCommit = FakeSession.prototype.commit;
      const commit = spyOn(FakeSession.prototype, "commit").mockImplementation(function (
        this: FakeSession,
        commitId: string,
        mutations: readonly MemoryMutation[],
      ) {
        const accepted = built.state.backwardState.current!.requests.at(-1)!;
        const recorded = records(built).at(-1)!;
        expect(recorded.constructor).toBe(MemoryCommitRequestedRecord);
        expect(accepted).toEqual(recorded as MemoryCommitRequestedRecord);
        expect(commitLearning.mock.calls.at(-1)![0]).toBe(accepted);
        expect(commitId).toBe(accepted.commitId);
        expect(mutations).toBe(accepted.mutations);
        const mutation = mutations[0]!;
        // Python edits the mutation in place with `object.__setattr__`; a frozen carrier refuses both edits.
        if (target === "mutation") {
          expect(() => {
            (mutation as { mutationId: string }).mutationId = "backend-changed";
          }).toThrow(TypeError);
        } else {
          expect(() => (mutation.payload as { notes: unknown[] }).notes.push("backend-copy")).toThrow(TypeError);
        }
        expect(accepted).toEqual(recorded as MemoryCommitRequestedRecord);
        return originalCommit.call(this, commitId, mutations);
      });
      const commitLearning = spyOn(MemoryRuntime.prototype, "commitLearning");
      try {
        if (completion === "host_failure") {
          expect(await rejected(drive(built, new Map([["A", memory]])))).toBe(host);
          expect(built.state.backwardState.finished).toBeNull();
        } else {
          const outcome = await drive(built, new Map([["A", memory]]));
          expect(outcome.status).toBe(completion === "success" ? "succeeded" : "failed");
        }
      } finally {
        commit.mockRestore();
        commitLearning.mockRestore();
      }
      const backward = built.state.backwardState;
      const request = backward.current!.requests[0]!;
      const recorded = records(built);
      const requested = recorded.find((record) => record.constructor === MemoryCommitRequestedRecord);
      const ready = recorded.find((record) => record.constructor === BackwardReadyRecord);
      expect(request).toEqual(requested as MemoryCommitRequestedRecord);
      expect(backward.ready).toEqual(ready as BackwardReadyRecord);
      expect(request.mutations[0]!.mutationId).toBe("proposal");
      expect(request.mutations[0]!.payload).toEqual({ notes: ["saved"] });
      const statuses: Record<string, "succeeded" | "failed" | "unknown"> = {
        success: "succeeded",
        known_failure: "failed",
        host_failure: "unknown",
      };
      expect(backward.policyOutcomes[0]!.status).toBe(statuses[completion]!);
      expect(built.state.trace!.nextSequence).toBe(built.state.nextSequence);
      expect(built.state.nextSequence).toBe(built.log.nextSequence);
      expect(memory.commitCalls).toHaveLength(1);
      expect(memory.closeCount).toBe(1);
    }
  }
});

/** Let pending promise callbacks and timers run. */
function settle(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

test("driving requires a RunState and a RunLog", async () => {
  const { engine: built } = await terminal();
  const before = records(built);
  let opened = 0;
  const openMemory = () => {
    opened += 1;
    return open(built.state.memories!, new Map([["A", new FakeMemory()]]));
  };
  // A RunState-derived object isn't exactly a RunState.
  for (const [state, log] of [
    [Object.create(built.state), built.log],
    [built.state, {}],
  ]) {
    const error = await rejected(driveBackward(state as RunState, log as RunLog, "review", { openMemory }));
    expect((error as Error).constructor).toBe(TypeError);
    expect((error as Error).message).toBe("backward driving requires a RunState and RunLog");
  }
  expect(opened).toBe(0);
  expect(records(built)).toEqual(before);
});

test("an async opener resolves before the first record is appended", async () => {
  const { engine: built, node } = await terminal();
  const memory = new FakeMemory();
  const gate = Promise.withResolvers<void>();
  const before = records(built);
  const pending = driveBackward(built.state, built.log, "review", {
    openMemory: async () => {
      await gate.promise;
      return open(built.state.memories!, new Map([["A", memory]]));
    },
  });
  await settle();
  expect(records(built)).toEqual(before);
  expect(node.calls).toEqual([]);
  gate.resolve();
  expect((await pending).status).toBe("succeeded");
  expect(memory.closeCount).toBe(1);
});

test("a drive overtaken during its opener appends nothing and closes its runtime", async () => {
  const rows: [boolean, string][] = [
    [false, "backward attempt_id must be the next accepted start"],
    [true, "backward record requires its matching unfinished attempt"],
  ];
  for (const [interrupted, message] of rows) {
    const { engine: built } = await terminal();
    if (interrupted) {
      commitPrepared(built.log, built.state, prepareRecord(built.state, new BackwardStartedRecord(1, "first")));
    }
    const stale = new FakeMemory();
    const gate = Promise.withResolvers<void>();
    const pending = driveBackward(built.state, built.log, "stale", {
      openMemory: async () => {
        await gate.promise;
        return open(built.state.memories!, new Map([["A", stale]]));
      },
    });
    // The other drive completes the next attempt while the first awaits its opener.
    expect((await drive(built, new Map([["A", new FakeMemory()]]))).status).toBe("succeeded");
    const after = records(built);
    gate.resolve();
    const error = await rejected(pending);
    expect((error as Error).constructor).toBe(RangeError);
    expect((error as Error).message).toBe(message);
    expect(records(built)).toEqual(after);
    expect(stale.learnContexts).toEqual([]);
    expect(stale.closeCount).toBe(1);
  }
});

test("a drive overtaken during its handler is refused by the reducer", async () => {
  for (const failure of ["handler", "learner"]) {
    const gate = Promise.withResolvers<void>();
    const entered = Promise.withResolvers<void>();
    let calls = 0;
    const callback: BackwardCallback = async (_context, feedback) => {
      calls += 1;
      if (calls === 1) {
        entered.resolve();
        await gate.promise;
        if (failure === "handler") throw new Error("Callback failed.");
      }
      return new BackwardResult({ inputFeedback: { topic: feedback } });
    };
    const { engine: built } = await terminal(["A"], { callback });
    const stale = new FakeMemory({
      learnOutcomes: failure === "learner" ? [new MemoryOperationError("learn", "rejected", "Rejected.", false)] : [],
    });
    const pending = drive(built, new Map([["A", stale]]), "stale");
    await entered.promise;
    // The other drive closes the stale attempt as interrupted and completes its own.
    expect((await drive(built, new Map([["A", new FakeMemory()]]))).status).toBe("succeeded");
    const after = records(built);
    const finished = built.state.backwardState.finished!;
    gate.resolve();
    const error = await rejected(pending);
    expect((error as Error).constructor).toBe(RangeError);
    expect((error as Error).message).toBe("backward record requires its matching unfinished attempt");
    expect(records(built)).toEqual(after);
    expect(built.state.backwardState.finished).toBe(finished);
    expect(built.state.backwardState.finished!.outcome.status).toBe("succeeded");
    expect(stale.learnContexts).toHaveLength(failure === "learner" ? 1 : 0);
    expect(stale.commitCalls).toEqual([]);
    expect(stale.closeCount).toBe(1);
  }
});

test("a drive overtaken while its commit is pending appends nothing further", async () => {
  const { engine: built } = await terminal();
  const stale = new FakeMemory({ learnOutcomes: [() => [new MemoryMutation("stale", { notes: ["one"] })]] });
  const gate = Promise.withResolvers<void>();
  const entered = Promise.withResolvers<void>();
  const originalCommit = FakeSession.prototype.commit;
  const commit = spyOn(FakeSession.prototype, "commit").mockImplementation(async function (
    this: FakeSession,
    commitId: string,
    mutations: readonly MemoryMutation[],
  ) {
    if (this.memory === stale) {
      entered.resolve();
      await gate.promise;
    }
    return originalCommit.call(this, commitId, mutations);
  });
  try {
    const pending = drive(built, new Map([["A", stale]]), "stale");
    await entered.promise;
    expect(records(built).at(-1)!.constructor).toBe(MemoryCommitRequestedRecord);
    // The other drive closes the stale attempt as interrupted and completes its own.
    expect((await drive(built, new Map([["A", new FakeMemory()]]))).status).toBe("succeeded");
    const after = records(built);
    gate.resolve();
    const error = await rejected(pending);
    expect((error as Error).constructor).toBe(RangeError);
    expect((error as Error).message).toBe("backward record requires its matching unfinished attempt");
    expect(records(built)).toEqual(after);
    expect(stale.commitCalls).toHaveLength(1);
    expect(stale.closeCount).toBe(1);
  } finally {
    commit.mockRestore();
  }
});

test("a stale drive can't finish a newer drive's attempt", async () => {
  for (const failure of ["handler", "learner"]) {
    const gates = [Promise.withResolvers<void>(), Promise.withResolvers<void>()];
    const entered = [Promise.withResolvers<void>(), Promise.withResolvers<void>()];
    let calls = 0;
    const callback: BackwardCallback = async (_context, feedback) => {
      const call = calls++;
      entered[call]!.resolve();
      await gates[call]!.promise;
      if (call === 0 && failure === "handler") throw new Error("Callback failed.");
      return new BackwardResult({ inputFeedback: { topic: feedback } });
    };
    const { engine: built } = await terminal(["A"], { callback });
    const stale = new FakeMemory({
      learnOutcomes: failure === "learner" ? [new MemoryOperationError("learn", "rejected", "Rejected.", false)] : [],
    });
    const pending = drive(built, new Map([["A", stale]]), "stale");
    await entered[0]!.promise;
    // The newer drive closes attempt 1 as interrupted, starts attempt 2 and waits in its handler.
    const newer = drive(built, new Map([["A", new FakeMemory()]]));
    await entered[1]!.promise;
    const after = records(built);
    expect(built.state.backwardState.started!.attemptId).toBe(2);
    gates[0]!.resolve();
    const error = await rejected(pending);
    expect((error as Error).constructor).toBe(RangeError);
    expect((error as Error).message).toBe("backward record requires its matching unfinished attempt");
    expect(records(built)).toEqual(after);
    expect(stale.learnContexts).toHaveLength(failure === "learner" ? 1 : 0);
    expect(stale.closeCount).toBe(1);
    gates[1]!.resolve();
    expect((await newer).status).toBe("succeeded");
    const finished = records(built).at(-1)!;
    expect(finished.constructor).toBe(BackwardFinishedRecord);
    expect((finished as BackwardFinishedRecord).attemptId).toBe(2);
  }
});

test("an opener result that isn't a MemoryRuntime is refused before any record", async () => {
  const { engine: built, node } = await terminal();
  const before = records(built);
  const error = await rejected(
    driveBackward(built.state, built.log, "review", { openMemory: () => ({}) as MemoryRuntime }),
  );
  expect((error as Error).constructor).toBe(TypeError);
  expect((error as Error).message).toBe("backward memory opener must return a MemoryRuntime");
  expect(records(built)).toEqual(before);
  expect(node.calls).toEqual([]);
});

test("an opened runtime whose earlier close failed is refused as a mismatch", async () => {
  const { engine: built, node } = await terminal();
  const memory = new FakeMemory();
  const runtime = await open(built.state.memories!, new Map([["A", memory]]));
  const closeFailure = new Error("close failed");
  const close = spyOn(FakeSession.prototype, "close").mockImplementation(() => {
    throw closeFailure;
  });
  try {
    expect(await rejected(runtime.close())).toBe(closeFailure);
  } finally {
    close.mockRestore();
  }
  const before = records(built);
  const error = await rejected(driveBackward(built.state, built.log, "review", { openMemory: () => runtime }));
  expect((error as Error).constructor).toBe(RangeError);
  expect((error as Error).message).toBe(
    "opened memory runtime does not match the recorded memory table and snapshots",
  );
  expect(records(built)).toEqual(before);
  expect(node.calls).toEqual([]);
  expect(memory.learnContexts).toEqual([]);
});

test("a MemoryOperationError from learn or commit is a learning failure whatever its cause", async () => {
  for (const operation of ["learn", "commit"]) {
    const causes = [new DOMException("Aborted.", "AbortError"), new CapabilityFault(new Error("capability failed"))];
    for (const cause of causes) {
      const { engine: built } = await terminal();
      const error = new MemoryOperationError(operation, "wrapped", "Wrapped.", false);
      error.cause = cause;
      const memory = new FakeMemory({
        learnOutcomes: operation === "learn" ? [error] : [],
        commitOutcomes: operation === "commit" ? [error] : [],
      });
      const outcome = await drive(built, new Map([["A", memory]]));
      expect(outcome.status).toBe("failed");
      expect(outcome.failure!.stage).toBe(operation);
      expect(outcome.failure!.code).toBe("wrapped");
      expect(outcome.failure!.memory).toBe("A");
      expect(memory.closeCount).toBe(1);
    }
  }
});

test("a thrown undefined still closes the runtime and is rethrown", async () => {
  const { engine: built } = await terminal();
  const memory = new FakeMemory({
    learnOutcomes: [
      () => {
        throw undefined;
      },
    ],
  });
  // A close failure would escape in place of a lost primary.
  const originalClose = FakeSession.prototype.close;
  const close = spyOn(FakeSession.prototype, "close").mockImplementation(function (this: FakeSession) {
    originalClose.call(this);
    throw new Error("close failed");
  });
  let caught: unknown = null;
  try {
    caught = await rejected(drive(built, new Map([["A", memory]])));
  } finally {
    close.mockRestore();
  }
  expect(caught).toBeUndefined();
  expect(memory.closeCount).toBe(1);
  expect(built.state.backwardState.finished).toBeNull();
});
