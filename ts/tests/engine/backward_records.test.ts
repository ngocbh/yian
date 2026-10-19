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
 * Later (T4-2): the wire round trips and rejections, with the record codec; the prepare, append
 * and replay tests, with the runtime log and state.
 */

import { expect, test } from "bun:test";

import { SourceRef } from "../../src/expr/lineage.ts";
import { MemoryMutation, MemoryOperationFailure } from "../../src/memory/contracts.ts";
import { BackwardResult } from "../../src/nodes/base.ts";
import {
  AttributedBackwardResult,
  BackwardAttribution,
  BackwardOutcome,
  MemoryProposal,
} from "../../src/runtime/contracts.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
} from "../../src/runtime/log.ts";

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).constructor).toBe(cls);
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
