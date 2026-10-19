/**
 * Memory learning projections: a port of `tests/engine/test_runtime_memory_learning.py`.
 * Ported: `test_commit_id_has_one_documented_canonical_encoding` (both pinned digests),
 * `test_causal_route_episode_retains_the_accepted_failure` (with the `_scope` and `_evidence`
 * helpers) and `test_route_and_growth_projection_use_closed_source_neutral_envelopes`.
 *
 * Changed:
 * - `test_commit_id_requires_a_positive_attempt_and_separates_attempts`: the invalid attempt ids
 *   are `null, 0, -1, true, "2", 2.5, 2n` (`2.0` is the valid `2` in TS), each a `RangeError`
 *   naming `attempt_id` (Python's `ValueError`); the rejected mutation container is a `Set`
 *   (Python passes a list), since a batch is an array.
 * - Payload and seed ints are `bigint`s (`1n`), and the growth seed's tuple is an array.
 * - `_episode_outcome` is the module export `episodeOutcome`; its `is` on the status becomes
 *   `toBe` on the status string.
 *
 * TS-only: the run id, memory and mutation-item messages of `memoryCommitId`; a float payload
 * digests its `logJson` text; `failureJson` equals `toJSON()`, refuses a value that isn't exactly
 * a `Failure` and revalidates the failure's fields; `episodeOutcome` projects a success's canonical
 * value and a failed record's failure JSON, and refuses a record of another class and a forged
 * success without a canonical typed value.
 *
 * Later, in this item: the rewrites of
 * `test_owner_context_projects_synthetic_source_free_failure_only_to_root_flow` and
 * `test_root_fallback_reaches_only_keys_owned_by_the_root_flow_site`, and the export half of
 * `test_owner_contexts_are_the_only_episode_projection` (with the owner-context projection).
 *
 * Later (T4-4, T4-5, with the run state): `test_evidence_is_rejected_when_learning_is_not_enabled`,
 * `test_learning_requires_evidence_and_rejects_ineligible_or_unrecorded_scope`,
 * `test_learning_rejects_a_scope_without_a_recorded_stack`,
 * `test_empty_enabled_table_retains_scope_free_evidence_without_episodes`,
 * `test_learning_rejects_trace_scope_that_omits_a_recorded_memory`,
 * `test_graph_outcome_uses_constant_memory_roots_and_indexed_scope` and
 * `test_replay_rejects_an_eligible_node_scope_that_differs_from_its_recorded_stack`.
 *
 * Later (T4-7, with `computeBackward` and the engine): the owner-context and episode tests
 * (`test_owner_contexts_include_general_occurrences_and_empty_memories` through
 * `test_owner_variable_batches_follow_captured_inventory_across_wire_round_trip`, except
 * `test_owner_context_projects_synthetic_source_free_failure_only_to_root_flow`, listed above;
 * `test_runstarted_retains_defensive_memory_authority_and_projects_episodes`,
 * `test_engine_episodes_use_the_physical_start_sequence_for_every_outcome`,
 * `test_failure_episode_uses_full_canonical_durable_failure_json`,
 * `test_expanded_map_occurrence_projects_and_replays_from_live_node_scope` and the rest of
 * `test_owner_contexts_are_the_only_episode_projection`).
 */

import { createHash } from "node:crypto";

import { expect, test } from "bun:test";

import { Failure, OriginRef } from "../../src/diagnostics.ts";
import {
  EvaluationEvidence,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryScope,
  ModelTextInteraction,
} from "../../src/memory/index.ts";
import {
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
} from "../../src/runtime/log.ts";
import { episodeOutcome, failureJson, memoryCommitId } from "../../src/runtime/learning.ts";
import type { OutcomeRecord } from "../../src/runtime/trace.ts";

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

/** One exact trace scope for the node `agent`: an outer and a local memory, in one MAP item. */
function scope(): MemoryScope {
  return new MemoryScope(["flow", "agent"], {
    memories: ["outer", "local"],
    occurrencePath: [new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["flow"], 2)],
  });
}

/** Deterministic visible evidence for one accepted outcome. */
function evidence(memoryScope: MemoryScope | null): EvaluationEvidence {
  return new EvaluationEvidence(memoryScope, { question: "q" }, "system", "prompt", [], [
    new ModelTextInteraction("answer"),
  ]);
}

/** One mutation whose payload holds text, null and an int. */
const MUTATIONS = [new MemoryMutation("m", { z: 1n, a: ["é", null] })];

test("commit id has one documented canonical encoding", () => {
  expect(memoryCommitId("run", "notes", MUTATIONS, 1)).toBe(
    "ac-memory-v3:250e0a4c357acfd6ef4d37a84a5207bffdb3df2e343816df5e7219401a7bf47b",
  );
  expect(memoryCommitId("run", "notes", MUTATIONS, 2)).toBe(
    "ac-memory-v3:1c87e547bc72a30942158a6c0c0bc3dc4fa21245e45564ac7571175e48cdf64a",
  );
});

test("commit id requires a positive attempt and separates attempts", () => {
  expect(memoryCommitId("run", "notes", [], 2)).not.toBe(memoryCommitId("run", "notes", [], 3));
  for (const invalid of [null, 0, -1, true, "2", 2.5, 2n]) {
    expectError(
      () => memoryCommitId("run", "notes", MUTATIONS, invalid as number),
      RangeError,
      "attempt_id must be a positive integer",
    );
  }
  expectError(
    () => memoryCommitId("run", "notes", new Set(MUTATIONS) as unknown as MemoryMutation[], 1),
    TypeError,
    "mutations must be an array",
  );
});

test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string]>([
  ["an empty run id", () => memoryCommitId("", "notes", [], 1), RangeError, "run_id must be a non-empty string"],
  [
    "a run id that isn't text",
    () => memoryCommitId(null as unknown as string, "notes", [], 1),
    RangeError,
    "run_id must be a non-empty string",
  ],
  ["an empty memory", () => memoryCommitId("run", "", [], 1), RangeError, "memory must be a non-empty string"],
  [
    "a memory that isn't text",
    () => memoryCommitId("run", 1 as unknown as string, [], 1),
    RangeError,
    "memory must be a non-empty string",
  ],
  [
    "a mutation that isn't exactly a MemoryMutation",
    () => memoryCommitId("run", "notes", [{ mutationId: "m", payload: 1n } as unknown as MemoryMutation], 1),
    TypeError,
    "mutations items must be MemoryMutation values",
  ],
])("commit id refuses %s", (_label, fn, cls, message) => {
  expectError(fn, cls, message);
});

test("a float payload digests its log text", () => {
  const mutations = [new MemoryMutation("m", { x: 1.5, n: 2 })];
  const text = '["run",1,"notes",[["m",{"n":2.0,"x":1.5}]]]';
  const digest = createHash("sha256").update(text, "utf8").digest("hex");
  expect(memoryCommitId("run", "notes", mutations, 1)).toBe(`ac-memory-v3:${digest}`);
});

test("failure JSON equals the failure's JSON form", () => {
  const failure = new Failure({
    code: "node.execute",
    message: "Failed.",
    nodeId: "agent",
    trace: [new OriginRef("writer")],
    causeType: "Error",
    traceback: "stack",
  });
  expect(failureJson(failure)).toEqual(failure.toJSON());
});

test("failure JSON refuses a value that isn't exactly a Failure and revalidates its fields", () => {
  const failure = new Failure({ code: "node.execute", message: "Failed.", nodeId: "agent" });
  class SubFailure extends Failure {}
  for (const invalid of [failure.toJSON(), { ...failure }, new SubFailure(failure), null]) {
    expectError(() => failureJson(invalid as Failure), TypeError, "failure projection requires a Failure");
  }
  const edited = new Failure({ code: "node.execute", message: "Failed.", nodeId: "agent" });
  (edited as { code: string }).code = "bogus";
  expectError(() => failureJson(edited), RangeError, "unknown failure code 'bogus'");
});

test("causal route episode retains the accepted failure", () => {
  const failure = new Failure({ code: "graph.terminal_unreachable", message: "No terminal route.", nodeId: "agent" });
  const record = new NodeRoutedRecord("agent", "go", { evidence: evidence(scope()), terminalFailure: failure });
  const [status, output, failurePayload] = episodeOutcome(record);
  expect(status).toBe(MemoryEpisodeStatus.FAILED);
  expect(output).toBeNull();
  expect(failurePayload!.code).toBe(failure.code);
});

test("route and growth projection use closed source-neutral envelopes", () => {
  const [, route] = episodeOutcome(new NodeRoutedRecord("agent", "yes"));
  const [, growth] = episodeOutcome(new NodeExpandedRecord("agent", { value: [1n, 2n] }, ["a", "z"], "b".repeat(64)));

  expect(route).toEqual({ kind: "route", handle: "yes" });
  expect(growth).toEqual({ kind: "grow", seed: { value: [1n, 2n] }, prune_ids: ["a", "z"] });
});

test("an episode projects a success's canonical value and a failed record's failure JSON", () => {
  const record = new NodeSucceededRecord("agent", "agent", { kind: "object", value: { b: 1n, a: 2.5 } });
  const succeeded = episodeOutcome(record);
  expect(succeeded).toEqual([MemoryEpisodeStatus.SUCCEEDED, { a: 2.5, b: 1n }, null]);
  expect(Object.keys(succeeded[1] as object)).toEqual(["a", "b"]);

  const failure = new Failure({ code: "node.execute", message: "Failed.", nodeId: "agent" });
  expect(episodeOutcome(new NodeFailedRecord(failure))).toEqual([MemoryEpisodeStatus.FAILED, null, failure.toJSON()]);

  expectError(
    () => episodeOutcome(new NodeStartedRecord("agent") as unknown as OutcomeRecord),
    TypeError,
    "unsupported graph outcome for episode projection",
  );
});

test("an episode refuses a success without a canonical typed value", () => {
  // A constructed record always holds a typed wire, so the record is forged through its prototype.
  const forged = Object.assign(Object.create(NodeSucceededRecord.prototype) as NodeSucceededRecord, {
    nodeId: "agent",
    value: { kind: "int" },
  });
  expectError(() => episodeOutcome(forged), RangeError, "successful record has no canonical typed value");
});
