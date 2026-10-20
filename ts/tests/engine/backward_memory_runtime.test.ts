/**
 * Explicit backward proposals and ordered, append-authorized memory effects: a port of
 * `tests/engine/test_backward_memory_runtime.py`, over the shared fakes of `memory_fakes.ts`.
 *
 * Ported: `test_proposal_phase_visits_every_memory_without_committing` and
 * `test_proposal_contexts_validate_completely_before_any_learner` (6 rows), with the `_table` and
 * `_open` helpers; `test_empty_feedback_still_learns_empty_memories_and_commits_empty_batches`,
 * `test_no_policy_backward_still_records_ready_and_finished_without_opening_resources`,
 * `test_computation_failure_finishes_without_partial_feedback_or_learning` (2 rows; the callback
 * throws a plain `Error`, Python's `ValueError`), the `succeeded` and `requested_unfinished` rows of
 * `test_later_attempt_follows_or_closes_any_prior_attempt_and_recomputes_everything`,
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
 *
 * Later (PR 3 of T4-7): `test_late_proposal_failure_discards_all_staged_mutations`,
 * `test_policy_failure_preserves_prior_success_and_stops_later_policies`,
 * `test_context_depth_failure_keeps_completed_backward_without_calling_learners`,
 * `test_replayed_context_failure_must_name_an_accepted_receiving_occurrence`,
 * `test_append_fault_preserves_exact_prefix_and_closes_resources`,
 * `test_manual_precommit_retry_reopens_snapshots_and_recomputes_every_memory`,
 * `test_known_precommit_failure_allows_fresh_complete_pass_without_prior_proposals`,
 * `test_failed_resource_setup_does_not_close_an_interrupted_attempt`, the `RuntimeError` and
 * `RecursionError` cells (a plain `Error` and a `RangeError`) of
 * `test_backend_host_or_unexpected_error_escapes_with_exact_prefix`, the two `with_primary` cells
 * of `test_cleanup_unwraps_host_failure_after_closing_all_sessions` (plain errors),
 * `test_host_capability_identity_and_cleanup_error_preserve_primary_exception`, the `commit_failed`
 * row of `test_later_attempt_follows_or_closes_any_prior_attempt_and_recomputes_everything`,
 * `test_cleanup_error_after_finished_preserves_success_and_admits_a_later_attempt`,
 * `test_backward_and_memory_replay_calls_no_callbacks_or_materialization`,
 * `test_prepare_failure_is_not_sanitized_as_a_learner_failure`,
 * `test_learner_mutation_of_its_arguments_and_returned_values_cannot_change_authority`,
 * `test_wrapped_host_error_during_partial_setup_closes_prior_sessions` and
 * `test_commit_batch_is_detached_from_every_accepted_authority`.
 *
 * Not ported: TS unwraps no host control from a `MemoryOperationError`'s cause chain, so a
 * `MemoryOperationError` from learn or commit is a learning failure whatever its `cause`, and any
 * other thrown value escapes unchanged. So the cause-walk tests
 * `test_typed_backend_wrappers_preserve_original_host_error`,
 * `test_nested_capability_faults_never_escape_private_wrappers`,
 * `test_capability_originals_are_checked_for_wrapped_host_controls` and
 * `test_cyclic_capability_fault_is_a_host_error_without_exposing_private_carriers`; the
 * `BaseException`-type cells (`MemoryError`, both `CancelledError`s, `KeyboardInterrupt`,
 * `SystemExit`) of `test_backend_host_or_unexpected_error_escapes_with_exact_prefix` and the
 * `error_type` dimension of `test_cleanup_unwraps_host_failure_after_closing_all_sessions`; and
 * `test_each_attempt_checks_only_its_own_records_not_earlier_attempts`, which counts the calls of a
 * module-private function.
 */

import { expect, spyOn, test } from "bun:test";

import { FlowOutput, START_ID } from "../../src/compile/model.ts";
import type { FeedbackValue } from "../../src/expr/lineage.ts";
import {
  DeclSite,
  LearnContext,
  MemoryMutation,
  MemoryScope,
  MemorySnapshotEnvelope,
  MemorySpec,
} from "../../src/memory/contracts.ts";
import { BackwardResult } from "../../src/nodes/base.ts";
import { computeBackward, driveBackward } from "../../src/runtime/backward.ts";
import { MemoryProposal } from "../../src/runtime/contracts.ts";
import type { FlowEngine } from "../../src/runtime/engine.ts";
import { memoryCommitId } from "../../src/runtime/learning.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  encodePool,
  MemoryCommitRequestedRecord,
  type RunRecord,
} from "../../src/runtime/log.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import { commitPrepared, prepareRecord } from "../../src/runtime/state.ts";
import { type BackwardCallback, engine, graph, RecordingBackwardNode, rejected, stacks } from "./run_state_fixtures.ts";
import { FakeMemory, FakeSession } from "./memory_fakes.ts";

const ROOT = new DeclSite([], ["root"]);

/** Return a sorted key table; the first key is root-owned, the rest are owned by unused defs. */
function table(...keys: string[]): MemorySpec[] {
  return [...new Set(keys)]
    .sort()
    .map(
      (key, index) =>
        new MemorySpec(key, key, {}, [index === 0 ? ROOT : new DeclSite([], ["def", `unused${index}`])], false),
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

test("a later attempt follows or closes any prior attempt and recomputes everything", async () => {
  for (const prior of ["succeeded", "requested_unfinished"]) {
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
      expect((await drive(built, new Map([["A", new FakeMemory()]]))).status).toBe("succeeded");
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
