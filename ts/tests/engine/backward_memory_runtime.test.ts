/**
 * Explicit backward proposals and ordered, append-authorized memory effects: a port of
 * `tests/engine/test_backward_memory_runtime.py`, over the shared fakes of `memory_fakes.ts`.
 *
 * Ported: `test_proposal_phase_visits_every_memory_without_committing` and
 * `test_proposal_contexts_validate_completely_before_any_learner` (6 rows), with the `_table` and
 * `_open` helpers.
 *
 * Changed: Python's tuples are arrays and its dictionaries are `Map`s; `proposeLearning` and
 * `MemoryRuntime.open` return promises; each learner receives the very context passed in (`toBe`,
 * where Python compares a revalidated copy with `==`); each refusal pins its class and full
 * message (Python accepts either `TypeError` or `ValueError`), and the `list` row passes a `Set`
 * (Python's list for a tuple).
 *
 * Later (PR 3 of T4-7): `test_backward_finishes_all_callbacks_before_ready_and_policy_requests`,
 * `test_late_proposal_failure_discards_all_staged_mutations`,
 * `test_policy_failure_preserves_prior_success_and_stops_later_policies`,
 * `test_empty_feedback_still_learns_empty_memories_and_commits_empty_batches`,
 * `test_no_policy_backward_still_records_ready_and_finished_without_opening_resources`,
 * `test_context_depth_failure_keeps_completed_backward_without_calling_learners`,
 * `test_replayed_context_failure_must_name_an_accepted_receiving_occurrence`,
 * `test_computation_failure_finishes_without_partial_feedback_or_learning`,
 * `test_append_fault_preserves_exact_prefix_and_closes_resources`,
 * `test_manual_precommit_retry_reopens_snapshots_and_recomputes_every_memory`,
 * `test_known_precommit_failure_allows_fresh_complete_pass_without_prior_proposals`,
 * `test_failed_resource_setup_does_not_close_an_interrupted_attempt`, the `RuntimeError` and
 * `RecursionError` cells (a plain `Error` and a `RangeError`) of
 * `test_backend_host_or_unexpected_error_escapes_with_exact_prefix`, the two `with_primary` cells
 * of `test_cleanup_unwraps_host_failure_after_closing_all_sessions` (plain errors),
 * `test_host_capability_identity_and_cleanup_error_preserve_primary_exception`,
 * `test_later_attempt_follows_or_closes_any_prior_attempt_and_recomputes_everything`,
 * `test_repeated_driven_attempts_request_distinct_commit_ids_for_identical_batches`,
 * `test_cleanup_error_after_finished_preserves_success_and_admits_a_later_attempt`,
 * `test_backward_and_memory_replay_calls_no_callbacks_or_materialization`,
 * `test_prepare_failure_is_not_sanitized_as_a_learner_failure`,
 * `test_learner_mutation_of_its_arguments_and_returned_values_cannot_change_authority`,
 * `test_opened_memory_runtime_subclass_is_closed_by_the_driver`,
 * `test_submission_gates_reject_before_opening_or_appending`,
 * `test_wrapped_host_error_during_partial_setup_closes_prior_sessions` and
 * `test_commit_batch_is_detached_from_every_accepted_authority`, with the `_terminal`, `_records`
 * and `_drive` helpers.
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

import { expect, test } from "bun:test";

import {
  DeclSite,
  LearnContext,
  MemoryMutation,
  MemorySnapshotEnvelope,
  MemorySpec,
} from "../../src/memory/contracts.ts";
import { MemoryProposal } from "../../src/runtime/contracts.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import { FakeMemory } from "./memory_fakes.ts";

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
