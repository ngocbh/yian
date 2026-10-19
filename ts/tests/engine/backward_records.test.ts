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
 * Later (T4-3 to T4-8): `test_historical_backward_prepares_recorded_schema_bytes` and the
 * prepare, append, reduce and replay tests (with the reducer, the runtime log and the engine).
 */

import { expect, test } from "bun:test";

import { Failure, OriginRef } from "../../src/diagnostics.ts";
import { type FeedbackValue, SourceRef } from "../../src/expr/lineage.ts";
import {
  LearningFailure,
  LearningResult,
  MemoryMutation,
  MemoryOperationFailure,
  PolicyCommitOutcome,
} from "../../src/memory/contracts.ts";
import { BackwardResult } from "../../src/nodes/base.ts";
import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";
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
  type RunRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";

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
