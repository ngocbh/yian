/**
 * The conversation carriers, the memory context carriers and the conversation interfaces, from
 * `tests/engine/test_session_contracts.py`: `test_memory_protocol_has_no_frames`,
 * `test_invocation_rejects_boolean_start_sequence`, `test_request_detaches_compatibility`,
 * `test_item_payload_is_detached`, `test_stamp_requires_nonnegative_exact_counters`,
 * `test_conversation_json_rejects_noncanonical_values`, `test_conversation_json_rejects_cycles`,
 * `test_effect_state_preserves_invocation_and_status`,
 * `test_effect_state_requires_ordered_later_observations`,
 * `test_completion_preserves_actual_disposition`, `test_applied_entry_validates_digest_shape`,
 * `test_applied_entry_requires_canonical_operation_identity`,
 * `test_empty_view_has_exact_detached_shape`, `test_view_rejects_duplicate_item_ids`,
 * `test_view_requires_exact_tuple_collections`, `test_view_requires_one_applied_entry_per_revision`,
 * `test_zero_revision_view_cannot_carry_history`, `test_view_rejects_duplicate_ledger_identities`,
 * `test_admitted_contains_detached_baseline`, `test_requested_retains_only_new_items`,
 * `test_observed_rejects_invalid_terminal_failure`,
 * `test_observed_preserves_detached_payload_and_actual_status`,
 * `test_observed_requires_a_prior_effect`, `test_closed_accepts_only_turn_dispositions`,
 * `test_conversation_capabilities_are_optional_and_exact`,
 * `test_conversation_identifiers_are_exact_nonempty_unicode`, `test_effect_kind_is_closed`,
 * `test_terminal_effect_requires_observation`, `test_pending_effect_can_have_no_observations`,
 * `test_view_counters_are_exact_nonnegative_integers`, `test_nested_carrier_subclasses_are_rejected`,
 * the rows without forgery of `test_nested_view_carriers_are_revalidated_and_detached`,
 * `test_effect_checkpoints_reject_zero_ordinal_and_list_items`,
 * `test_memory_context_request_detaches_every_value_boundary`,
 * `test_memory_context_request_rejects_invalid_carriers`,
 * `test_memory_content_validates_explicit_transcript_sources`, the `contracts.ts` half of
 * `test_memory_context_carriers_are_contracts_only_exports` and the field-order half of
 * `test_conversation_carriers_have_no_owner_key`; plus JavaScript's own checks: a `-0` payload
 * number reads back `0`, counters must be safe integers, every carrier and its arrays are frozen,
 * `sameConversationInvocation`, ledger identities compared by value, a `RecallContext` subclass
 * refused by `MemoryContextRequest`, and `MemoryContext`'s section checks.
 *
 * From the same file, the `conversation.ts` fold, codecs and digests:
 * `test_content_digest_excludes_ledger`, `test_admission_advances_revision_once`,
 * `test_request_appends_new_items_and_pending_effect`,
 * `test_observation_changes_only_matching_effect`, `test_close_preserves_unresolved_effects`,
 * `test_transition_rejects_skipped_ordinal`, `test_admission_preserves_unresolved_or_unclosed_effects`,
 * `test_admission_can_follow_effect_free_unclosed_invocation`,
 * `test_conversation_codecs_round_trip_complete_history_and_detach`,
 * `test_view_codec_rejects_wrong_closed_wire_shape`, `test_view_codec_rejects_broken_digest_chain`,
 * `test_view_codec_rejects_impossible_history_even_with_matching_final_digest`,
 * `test_invalid_transitions_preserve_input_evidence`,
 * `test_later_invocation_reuses_local_ordinals_without_replacing_prior_effects`,
 * `test_checkpoint_codec_rejects_malformed_closed_wire`, `test_codecs_reject_non_json_nested_values`,
 * `test_codecs_reject_cycles_and_non_string_keys`,
 * `test_checkpoint_digest_includes_complete_admission_ledger`,
 * `test_view_codec_rejects_invalid_nested_values`,
 * `test_view_codec_accepts_admission_after_unclosed_effectful_invocation`,
 * `test_fold_does_not_mask_unexpected_host_fault` and
 * `test_checkpoint_codec_rejects_retired_native_state_without_rewriting`; plus JavaScript's own
 * checks: a 10 000-level payload through the checkpoint and view codecs and the fold, a `-0`
 * payload that round-trips and digests as `0`, a counter past 2^53 refused, and the exact-class
 * checks of the codecs and the fold.
 *
 * Then `summary.ts`, which Python tests only through `SessionMemory`
 * (`tests/engine/test_session_summaries.py`); these focused tests are TypeScript's own: admissions
 * include effect-free turns, the prefix digest equals one computed with Python yian and changes with
 * any earlier ledger entry, a prefix needs a next admission, a `Summary` is frozen and refuses
 * non-string or blank text (Python's `strip()` whitespace, so `\x1c` is blank and `\ufeff` isn't),
 * and `matches` is false for another namespace, generation or version, empty `covered`, a
 * non-prefix `prior` or a stale digest.
 *
 * Changed: nested carriers are kept by reference, so where Python asserts a detached copy `is not`
 * the carrier passed in, the test checks that it is the same frozen value. The JSON fields are
 * deep-frozen snapshots, so editing them throws `TypeError` (Python returns a fresh copy). A frozen
 * field throws `TypeError` on assignment where Python raises `FrozenInstanceError`; Python's
 * `ValueError` is `RangeError`, and each row checks its exact class and message. The JSON fields
 * follow the yian value number rule: `±Infinity` is accepted (Python refuses `inf`), NaN stays
 * refused, and `-0` is read as `0`; Python's ints are `bigint`s. The non-string-key row (`{1: "v"}`)
 * becomes a symbol-key row; the tuple and `object()` rows become a `Set` and a class instance.
 * Tuples are arrays, so the list-for-tuple rows pass a `Set`. JavaScript has no `1.0` distinct from
 * `1`, so the float counter rows pass `1.5` and a `bigint`. An operation id whose number is written as
 * a float (`0.0`, `1e3`) fails the canonical round trip instead of the counter's type check; one past
 * 2^53 is refused as a counter (Python accepts it); `NaN` isn't JSON, so it fails the encoding check
 * (Python raises `TypeError`). The identifier row built with `replace(_stamp(), namespace=...)`
 * constructs the stamp directly (Python's `replace` reruns the same checks). Dataclass `==` becomes
 * `toEqual` or `sameConversationInvocation`.
 *
 * Changed (memory contexts and interfaces): the fake memory is a plain object, as Python's `_Memory`
 * fake; Python's `isinstance` on a runtime-checkable protocol becomes `isMemory` or a probe of
 * `openTurn` with `typeof`. Python's `vars(MemorySession)`, `get_args`, `get_type_hints` and
 * `inspect.signature` rows become typecheck-only checks: `MemorySession`'s member set, the
 * checkpoint union, `openTurn`'s parameters (the request and an options object holding the stop
 * `signal`, with no checkpoint), and fake turns typed as `ConversationTurn` with and without the
 * optional `prepareHistory` and `interrupt`, whose `request` is called without `items`.
 * `MemoryContextRequest` keeps its frozen recall context and invocation by reference, so where
 * Python mutates the originals through `object.__setattr__`, the test checks that assigning to them
 * throws `TypeError` and that the request holds the same values. `MemoryContent` takes an options
 * object (`{ text, transcript }`); its rows check each class and message.
 *
 * Changed (conversation): wire JSON is frozen, so a row damages a mutable `structuredClone` copy
 * (Python edits the returned dict), and the detach check clears a copy and asserts that editing the
 * frozen wire throws `TypeError`. Wire counters are `bigint`s and checkpoint digests hash `logJson`
 * text (Python's canonical `json.dumps`). Python's `pytest.raises((TypeError, ValueError))` rows
 * check the exact class and message. Python's `1.0` is a `number`, its ints `bigint`s. The `inf`
 * row of `test_codecs_reject_non_json_nested_values` is accepted under the value number rule and
 * moves to its own round-trip row; the tuple and `object()` rows pass a `Set` and a class
 * instance, and the `{1: "invalid"}` row of `test_codecs_reject_cycles_and_non_string_keys` is a
 * symbol key. A conflict's message is checked too. Where Python compares the input view against a
 * `_copy_conversation_view` copy, the test compares its wire's `logJson` text before and after; the
 * host-fault row spies on `crypto.createHash` (Python patches `hashlib.sha256`). The deep row
 * compares `logJson` text, not `toEqual`. The view decoder's freeze paths read
 * `conversation wire.<field>`, where Python's read `conversation wire.baseline.<field>` (its
 * admission-wrapper reservation, dropped).
 *
 * Not ported (conversation): the wire-nesting tests
 * `test_wire_nesting_is_checked_before_fold_and_encoder_return` (`:1055`),
 * `test_publishable_view_can_complete_and_become_the_next_admission_baseline` (`:1091`) and
 * `test_view_codec_reserves_the_next_admission_baseline_wrapper` (`:1130`): TS memory JSON has no
 * nesting limit, so the 10 000-level row replaces them; and
 * `test_fold_and_codecs_revalidate_tampered_exact_carriers` (`:1239`): frozen carriers can't be
 * tampered with.
 *
 * Not ported: the `object.__setattr__` rows of
 * `test_nested_view_carriers_are_revalidated_and_detached` and all of
 * `test_nested_corrupted_identity_is_revalidated` (Python forges a frozen dataclass; a TS carrier is
 * frozen by its constructor and a subclass can't skip it, so none holds an unchecked field); the
 * `__enter__` and `__exit__` rows of `test_memory_protocol_has_no_frames` (a session has no `with`
 * form; its caller calls `close()`).
 *
 * Left for later: `test_conversation_contracts_are_exported_without_concrete_backend` (`:361`), the
 * `memory.__all__` half of `test_memory_context_carriers_are_contracts_only_exports` (`:1357`) and
 * the export half of `test_conversation_carriers_have_no_owner_key` (`:1367`), with the package
 * exports.
 */

import { expect, spyOn, test } from "bun:test";
import crypto from "node:crypto";

import * as contracts from "../../src/memory/contracts.ts";
import {
  ConversationAdmitted,
  ConversationApplied,
  type ConversationCheckpoint,
  ConversationClosed,
  ConversationCompletion,
  type ConversationEffectKind,
  ConversationEffectObserved,
  ConversationEffectRequested,
  ConversationEffectState,
  type ConversationEffectStatus,
  ConversationInvocation,
  ConversationItem,
  ConversationRequest,
  ConversationStamp,
  type ConversationTurn,
  type ConversationTurnStatus,
  ConversationView,
  isMemory,
  LearnContext,
  type Memory,
  MemoryContent,
  MemoryContext,
  MemoryContextRequest,
  MemoryOperationError,
  MemoryOperationFailure,
  MemorySection,
  type MemorySession,
  RecallContext,
  sameConversationInvocation,
  type TranscriptSource,
} from "../../src/memory/contracts.ts";
import {
  advanceConversation,
  checkpointDigest,
  conversationCheckpointFromJson,
  conversationCheckpointToJson,
  conversationContentDigest,
  conversationViewFromJson,
  conversationViewToJson,
} from "../../src/memory/conversation.ts";
import { admissions, prefixDigest, Summary } from "../../src/memory/summary.ts";
import { logJson } from "../../src/typesys/json.ts";

type ErrorClass = typeof TypeError | typeof RangeError;

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: ErrorClass, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).message).toBe(message);
}

function invocation(): ConversationInvocation {
  return new ConversationInvocation("run", 1);
}

function stamp(ordinal = 0, revision = 0): ConversationStamp {
  return new ConversationStamp(invocation(), "namespace", 0, revision, ordinal);
}

function failure(): MemoryOperationFailure {
  return new MemoryOperationFailure("effect", "fake.failed", "Failed", false);
}

function applied(ordinal = 0): ConversationApplied {
  return new ConversationApplied(`["run",1,${ordinal}]`, "a".repeat(64), "b".repeat(64), "c".repeat(64));
}

interface ViewFields {
  namespace: unknown;
  generation: unknown;
  revision: unknown;
  compatibility: unknown;
  items: unknown;
  effects: unknown;
  closedInvocations: unknown;
  applied: unknown;
}

function view(changes: Partial<ViewFields> = {}): ConversationView {
  const v: ViewFields = {
    namespace: "namespace",
    generation: 0,
    revision: 0,
    compatibility: { provider: "fake" },
    items: [],
    effects: [],
    closedInvocations: [],
    applied: [],
    ...changes,
  };
  return new ConversationView(
    v.namespace as never,
    v.generation as never,
    v.revision as never,
    v.compatibility,
    v.items as never,
    v.effects as never,
    v.closedInvocations as never,
    v.applied as never,
  );
}

class Point {
  x = 1n;
}

/** Whether two types are mutually assignable; a typecheck-only equality. */
type Same<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;

/** An observable plain-object memory, as Python's `_Memory` fake. */
interface ObservableMemory extends Memory {
  readonly recallContexts: (RecallContext | null)[];
  readonly learnContexts: LearnContext[];
}

/** Build a memory whose sessions recall `recallText`, learn nothing and record their contexts. */
function observableMemory(recallText = ""): ObservableMemory {
  const recallContexts: (RecallContext | null)[] = [];
  const learnContexts: LearnContext[] = [];
  return {
    recallContexts,
    learnContexts,
    openRun: () => ({
      snapshot: { namespace: "test" },
      close() {},
      recall: (context = null) => {
        recallContexts.push(context);
        return recallText;
      },
      learn: (context) => {
        learnContexts.push(context);
        return [];
      },
      commit() {},
    }),
    reset() {},
  };
}

/** Build a turn without the optional capabilities whose requests all get ordinal 1. */
function plainTurn(): ConversationTurn {
  return { view: view(), request: () => 1, observe() {}, close() {} };
}

// test_memory_protocol_has_no_frames
test("the memory protocol has no frames", async () => {
  const memory = observableMemory("notes");
  const session = await memory.openRun();
  expect(isMemory(memory)).toBe(true);
  // The typecheck fails if `MemorySession` gains or loses a member.
  const members: Record<keyof MemorySession, true> = {
    snapshot: true,
    close: true,
    recall: true,
    learn: true,
    commit: true,
  };
  expect(Object.keys(session).sort()).toEqual(Object.keys(members).sort());
  expect("openFrame" in session).toBe(false);
  expect(await session.recall()).toBe("notes");
  expect(await session.learn(new LearnContext("run", "notes", []))).toEqual([]);
  expect(memory.recallContexts).toEqual([null]);
  expect(memory.learnContexts.map((context) => context.memory)).toEqual(["notes"]);
  await session.close();
});

// test_invocation_rejects_boolean_start_sequence
test.each<[unknown, ErrorClass, string]>([
  [true, TypeError, "start_sequence must be an integer"],
  [false, TypeError, "start_sequence must be an integer"],
  [0, RangeError, "start_sequence must be positive"],
  [-1, RangeError, "start_sequence must be positive"],
  [1.5, TypeError, "start_sequence must be an integer"],
  [1n, TypeError, "start_sequence must be an integer"],
  [2 ** 53, TypeError, "start_sequence must be an integer"],
  ["1", TypeError, "start_sequence must be an integer"],
])("an invocation refuses start sequence %p", (sequence, cls, message) => {
  expectError(() => new ConversationInvocation("run", sequence as never), cls, message);
});

test("sameConversationInvocation compares the fields", () => {
  expect(sameConversationInvocation(invocation(), invocation())).toBe(true);
  expect(sameConversationInvocation(invocation(), new ConversationInvocation("run", 2))).toBe(false);
  expect(sameConversationInvocation(invocation(), new ConversationInvocation("other", 1))).toBe(false);
});

// test_request_detaches_compatibility
test("a request snapshots its compatibility and keeps its invocation", () => {
  const descriptor = { provider: "fake", options: [true, 1n, 1.0] as unknown[] };
  const path = ["root", "child", "agent"];
  const inv = invocation();
  const request = new ConversationRequest(path, inv, descriptor);
  descriptor.options.push("changed");
  path.push("changed");
  expect(() => (request.compatibility.options as unknown[]).push("changed")).toThrow(TypeError);
  expect(request.compatibility).toEqual({ provider: "fake", options: [true, 1n, 1] });
  expect(request.consumerStaticPath).toEqual(["root", "child", "agent"]);
  expect(Object.isFrozen(request.consumerStaticPath)).toBe(true);
  expect(request.invocation).toBe(inv);
  expect(request.compatibility).not.toEqual({ provider: "fake", options: [1n, 1n, 1] });
  expectError(() => new ConversationRequest([], inv, {}), RangeError, "consumer_static_path must be non-empty");
  expectError(
    () => new ConversationRequest(new Set(["root"]) as never, inv, {}),
    TypeError,
    "consumer_static_path must be an array",
  );
});

// test_item_payload_is_detached
test("an item snapshots its payload", () => {
  const payload = { text: ["hello"] };
  const item = new ConversationItem("message-1", "api_message", payload);
  payload.text.push("caller mutation");
  expect(() => (item.payload.text as string[]).push("reader mutation")).toThrow(TypeError);
  expect(item.payload).toEqual({ text: ["hello"] });
  expect(new ConversationItem("x", "y", { value: true }).payload).not.toEqual(
    new ConversationItem("x", "y", { value: 1n }).payload,
  );
});

test("an item payload reads -0 back as 0 and keeps Infinity", () => {
  const item = new ConversationItem("id", "format", { v: -0, w: [Number.NEGATIVE_INFINITY], x: Infinity });
  expect(Object.is(item.payload.v, 0)).toBe(true);
  expect(item.payload.w).toEqual([Number.NEGATIVE_INFINITY]);
  expect(item.payload.x).toBe(Number.POSITIVE_INFINITY);
  const request = new ConversationRequest(["root", "agent"], invocation(), { v: Infinity });
  expect(request.compatibility.v).toBe(Number.POSITIVE_INFINITY);
});

// test_stamp_requires_nonnegative_exact_counters
test.each(
  (["generation", "expected_revision", "ordinal"] as const).flatMap((field) =>
    (
      [
        [true, TypeError, "must be an integer"],
        [-1, RangeError, "must be non-negative"],
        [1.5, TypeError, "must be an integer"],
        [1n, TypeError, "must be an integer"],
        ["1", TypeError, "must be an integer"],
      ] as const
    ).map(([value, cls, text]) => [field, value, cls, text] as const),
  ),
)("a stamp refuses %s %p", (field, value, cls, text) => {
  const values: Record<string, unknown> = { generation: 0, expected_revision: 0, ordinal: 0 };
  values[field] = value;
  expectError(
    () =>
      new ConversationStamp(
        invocation(),
        "ns",
        values.generation as never,
        values.expected_revision as never,
        values.ordinal as never,
      ),
    cls,
    `${field} ${text}`,
  );
});

// test_conversation_json_rejects_noncanonical_values
test.each<[string, unknown, ErrorClass, string]>([
  ["a top-level array", [], TypeError, " must be an exact JSON object"],
  ["NaN", { v: Number.NaN }, RangeError, ".v JSON number must not be NaN"],
  ["a Set", { v: new Set() }, TypeError, ".v must contain exact JSON values"],
  ["a class instance", { v: new Point() }, TypeError, ".v must contain exact JSON values"],
  ["a lone surrogate", { v: "\ud800" }, RangeError, "Unicode surrogate is not valid canonical JSON"],
  ["a symbol key", { [Symbol("key")]: "v" }, TypeError, " JSON object keys must be strings"],
])("conversation JSON refuses %s", (_name, payload, cls, text) => {
  const message = (name: string) => (text.startsWith("Unicode") ? text : `${name}${text}`);
  expectError(() => new ConversationItem("id", "format", payload), cls, message("payload"));
  expectError(
    () => new ConversationRequest(["root", "agent"], invocation(), payload),
    cls,
    message("compatibility"),
  );
});

// test_conversation_json_rejects_cycles
test("conversation JSON refuses cycles", () => {
  const payload: Record<string, unknown> = {};
  payload.cycle = payload;
  expectError(
    () => new ConversationItem("id", "format", payload),
    RangeError,
    "payload.cycle contains a circular JSON value",
  );
});

// test_effect_state_preserves_invocation_and_status
test.each(
  (["model", "tool"] as const).flatMap((kind) =>
    (["pending", "completed", "failed", "interrupted"] as const).map((status) => [kind, status] as const),
  ),
)("a %s effect state keeps its invocation and %s status", (kind, status) => {
  const inv = invocation();
  const request = { arguments: ["hello"] };
  const ordinals = [2, 3];
  const known = status === "failed" ? failure() : null;
  const effect = new ConversationEffectState(inv, 1, kind, request, status, ordinals, known);
  request.arguments.length = 0;
  ordinals.push(4);
  expect(() => (effect.request.arguments as string[]).push("x")).toThrow(TypeError);
  expect(effect.request).toEqual({ arguments: ["hello"] });
  expect(effect.invocation).toBe(inv);
  expect(effect.effectOrdinal).toBe(1);
  expect(effect.kind).toBe(kind);
  expect(effect.status).toBe(status);
  expect(effect.observationOrdinals).toEqual([2, 3]);
  expect(effect.failure).toBe(known);
});

// test_effect_state_requires_ordered_later_observations
test.each<[string, unknown, ErrorClass, string]>([
  ["a Set", new Set([2]), TypeError, "observation_ordinals must be an array"],
  ["[true]", [true], TypeError, "observation ordinal must be an integer"],
  ["[1]", [1], RangeError, "observation ordinals must increase after the request"],
  ["[3, 2]", [3, 2], RangeError, "observation ordinals must increase after the request"],
  ["[2, 2]", [2, 2], RangeError, "observation ordinals must increase after the request"],
])("an effect state refuses observation ordinals %s", (_name, ordinals, cls, message) => {
  expectError(
    () => new ConversationEffectState(invocation(), 1, "model", {}, "pending", ordinals as never, null),
    cls,
    message,
  );
});

// test_completion_preserves_actual_disposition
test.each<ConversationTurnStatus>(["completed", "failed", "interrupted"])(
  "a completion keeps its %s disposition",
  (status) => {
    const inv = invocation();
    const completed = new ConversationCompletion(inv, 3, status);
    expect(completed.status).toBe(status);
    expect(completed.closedOrdinal).toBe(3);
    expect(completed.invocation).toBe(inv);
    expectError(
      () => new ConversationCompletion(invocation(), 3, "pending" as never),
      RangeError,
      "invalid conversation turn status",
    );
    expectError(
      () => new ConversationCompletion(invocation(), 0, status),
      RangeError,
      "closed_ordinal must be positive",
    );
  },
);

// test_applied_entry_validates_digest_shape
test.each(
  (["event_digest", "before_digest", "after_digest"] as const).flatMap((field) =>
    (
      [
        ["A".repeat(64), RangeError, "must be a lowercase SHA-256 digest"],
        ["a".repeat(63), RangeError, "must be a lowercase SHA-256 digest"],
        ["g".repeat(64), RangeError, "must be a lowercase SHA-256 digest"],
        ["", RangeError, "must be non-empty"],
        [null, TypeError, "must be a string"],
      ] as const
    ).map(([digest, cls, text]) => [field, digest, cls, text] as const),
  ),
)("an applied entry refuses %s %p", (field, digest, cls, text) => {
  const values: Record<string, unknown> = {
    event_digest: "a".repeat(64),
    before_digest: "b".repeat(64),
    after_digest: "c".repeat(64),
  };
  const build = () =>
    new ConversationApplied(
      '["run",1,0]',
      values.event_digest as never,
      values.before_digest as never,
      values.after_digest as never,
    );
  const entry = build();
  expect(entry.operationId).toBe('["run",1,0]');
  expect([entry.eventDigest, entry.beforeDigest, entry.afterDigest]).toEqual([
    "a".repeat(64),
    "b".repeat(64),
    "c".repeat(64),
  ]);
  values[field] = digest;
  expectError(build, cls, `${field} ${text}`);
});

// test_applied_entry_requires_canonical_operation_identity
test.each<[string, ErrorClass, string]>([
  ['["run",true,0]', TypeError, "start_sequence must be an integer"],
  ['["run",0,0]', RangeError, "start_sequence must be positive"],
  ['["run",1,-1]', RangeError, "ordinal must be non-negative"],
  ['["run",1,0.0]', RangeError, "operation_id must be canonical JSON"],
  ['["run",1e3,0]', RangeError, "operation_id must be canonical JSON"],
  ['["run",9007199254740993,0]', TypeError, "start_sequence must be an integer"],
  ['["run",NaN,0]', RangeError, "operation_id must encode [run_id, start_sequence, ordinal]"],
  ['["run", 1, 0]', RangeError, "operation_id must be canonical JSON"],
  ['["",1,0]', RangeError, "run_id must be non-empty"],
  ["{}", RangeError, "operation_id must encode [run_id, start_sequence, ordinal]"],
  ["not-json", RangeError, "operation_id must encode [run_id, start_sequence, ordinal]"],
  ['["run",1,0,4]', RangeError, "operation_id must encode [run_id, start_sequence, ordinal]"],
])("an applied entry refuses operation id %s", (operationId, cls, message) => {
  expectError(
    () => new ConversationApplied(operationId, "a".repeat(64), "b".repeat(64), "c".repeat(64)),
    cls,
    message,
  );
});

// test_empty_view_has_exact_detached_shape
test("an empty view has its exact snapshotted shape", () => {
  const descriptor = { options: [true] };
  const empty = view({ compatibility: descriptor });
  descriptor.options.length = 0;
  expect(() => (empty.compatibility.options as boolean[]).push(false)).toThrow(TypeError);
  expect(empty.compatibility).toEqual({ options: [true] });
  expect([empty.namespace, empty.generation, empty.revision]).toEqual(["namespace", 0, 0]);
  expect([empty.items, empty.effects, empty.closedInvocations, empty.applied]).toEqual([[], [], [], []]);
  expect(() => {
    (empty as { revision: number }).revision = 1;
  }).toThrow(TypeError);
});

// test_view_rejects_duplicate_item_ids
test("a view refuses repeated item ids", () => {
  const item = new ConversationItem("id", "text", {});
  expectError(
    () => view({ revision: 1, applied: [applied()], items: [item, item] }),
    RangeError,
    "conversation item IDs must be unique",
  );
  const kept = view({ revision: 1, applied: [applied()], items: [item] });
  expect(kept.items[0]).toBe(item);
});

// test_view_requires_exact_tuple_collections
test.each([
  ["items", "items"],
  ["effects", "effects"],
  ["closedInvocations", "closed_invocations"],
  ["applied", "applied"],
] as const)("a view requires %s to be an array", (field, name) => {
  expectError(() => view({ [field]: new Set() }), TypeError, `${name} must be an array`);
});

// test_view_requires_one_applied_entry_per_revision
test("a view requires one applied entry per revision", () => {
  const message = "conversation revision requires one applied entry per checkpoint";
  expectError(() => view({ revision: 1 }), RangeError, message);
  expectError(() => view({ applied: [applied()] }), RangeError, message);
});

// test_zero_revision_view_cannot_carry_history
test("a revision-zero view can't carry history", () => {
  expectError(
    () => view({ items: [new ConversationItem("id", "message", {})] }),
    RangeError,
    "revision-zero conversation must have empty history",
  );
});

// test_view_rejects_duplicate_ledger_identities
test.each([
  ["effects", "conversation effect identities must be unique"],
  ["closedInvocations", "closed invocation identities must be unique"],
  ["applied", "applied operation IDs must be unique"],
] as const)("a view refuses repeated %s identities", (field, message) => {
  const build = () =>
    ({
      effects: new ConversationEffectState(invocation(), 1, "model", {}, "pending", [], null),
      closedInvocations: new ConversationCompletion(invocation(), 2, "failed"),
      applied: applied(),
    })[field];
  const changes: Partial<ViewFields> = { revision: 2, applied: [applied(), applied(1)] };
  const entry = build();
  changes[field] = [entry, entry];
  expectError(() => view(changes), RangeError, message);
  // Equal values built apart share one identity, as Python's dataclass equality does.
  changes[field] = [build(), build()];
  expectError(() => view(changes), RangeError, message);
});

// test_admitted_contains_detached_baseline
test("an admission keeps its baseline and stamp", () => {
  const baseline = view();
  const admission = stamp();
  const event = new ConversationAdmitted(admission, baseline);
  expect(event.baseline).toBe(baseline);
  expect(event.stamp).toBe(admission);
  expectError(
    () => new ConversationAdmitted(stamp(1), baseline),
    RangeError,
    "only conversation admission has ordinal zero",
  );
  expectError(
    () => new ConversationAdmitted(stamp(0, 1), baseline),
    RangeError,
    "admission stamp must match its baseline",
  );
  expectError(() => new ConversationAdmitted({} as never, baseline), TypeError, "stamp must be a ConversationStamp");
});

// test_requested_retains_only_new_items
test("a request checkpoint retains only new items", () => {
  const item = new ConversationItem("id", "text", { text: "hello" });
  const payload = { args: [true] };
  const event = new ConversationEffectRequested(stamp(1, 1), "model", payload, [item]);
  payload.args.length = 0;
  expect(() => (event.payload.args as boolean[]).push(false)).toThrow(TypeError);
  expect(event.payload).toEqual({ args: [true] });
  expect(event.items).toEqual([item]);
  expect(event.items[0]).toBe(item);
  expectError(
    () => new ConversationEffectRequested(stamp(1, 1), "model", {}, [item, item]),
    RangeError,
    "conversation item IDs must be unique",
  );
});

// test_observed_rejects_invalid_terminal_failure
test.each<[string, boolean, string]>([
  ["completed", true, "pending or completed effect cannot carry a failure"],
  ["pending", true, "pending or completed effect cannot carry a failure"],
  ["failed", false, "failed effect requires a failure"],
  ["unknown", false, "invalid conversation effect status"],
])("an observation with status %s and failure %p is refused", (status, hasFailure, message) => {
  const known = hasFailure ? failure() : null;
  expectError(() => new ConversationEffectObserved(stamp(2, 2), 1, status as never, [], known), RangeError, message);
  expectError(
    () => new ConversationEffectState(invocation(), 1, "model", {}, status as never, [2], known),
    RangeError,
    message,
  );
});

test("an effect failure must be exactly a MemoryOperationFailure or null", () => {
  class SubFailure extends MemoryOperationFailure {}
  expectError(
    () => new ConversationEffectObserved(stamp(2, 2), 1, "failed", [], new SubFailure("a", "b", "c", false)),
    TypeError,
    "failure must be a MemoryOperationFailure or null",
  );
});

// test_observed_preserves_detached_payload_and_actual_status
test.each<ConversationEffectStatus>(["pending", "completed", "failed", "interrupted"])(
  "an observation keeps its items and %s status",
  (status) => {
    const known = status === "failed" || status === "interrupted" ? failure() : null;
    const item = new ConversationItem("id", "text", {});
    const observed = stamp(2, 2);
    const event = new ConversationEffectObserved(observed, 1, status, [item], known);
    expect(event.stamp).toBe(observed);
    expect(event.effectOrdinal).toBe(1);
    expect(event.status).toBe(status);
    expect(event.items).toEqual([item]);
    expect(event.items[0]).toBe(item);
    expect(event.failure).toBe(known);
  },
);

// test_observed_requires_a_prior_effect
test.each<[unknown, ErrorClass, string]>([
  [true, TypeError, "effect_ordinal must be an integer"],
  [0, RangeError, "effect_ordinal must be positive"],
  [2, RangeError, "observed effect ordinal must precede the observation"],
  [3, RangeError, "observed effect ordinal must precede the observation"],
])("an observation refuses effect ordinal %p", (ordinal, cls, message) => {
  expectError(() => new ConversationEffectObserved(stamp(2, 2), ordinal as never, "completed", [], null), cls, message);
});

// test_closed_accepts_only_turn_dispositions
test.each<ConversationTurnStatus>(["completed", "failed", "interrupted"])(
  "a close accepts the %s turn disposition",
  (status) => {
    expect(new ConversationClosed(stamp(3, 3), status).status).toBe(status);
    expectError(
      () => new ConversationClosed(stamp(3, 3), "pending" as never),
      RangeError,
      "invalid conversation turn status",
    );
    expectError(
      () => new ConversationClosed(stamp(), status),
      RangeError,
      "only conversation admission has ordinal zero",
    );
  },
);

// test_conversation_capabilities_are_optional_and_exact
test("conversation capabilities are optional and exact", async () => {
  const session = await observableMemory().openRun();
  // Probed as `MemoryContent` probes a transcript source.
  expect(typeof (session as Partial<TranscriptSource>).openTurn).toBe("undefined");
  const checkpoints: Same<
    ConversationCheckpoint,
    ConversationAdmitted | ConversationEffectRequested | ConversationEffectObserved | ConversationClosed
  > = true;
  // `openTurn` takes the request and the stop signal, with no checkpoint parameter.
  const openTurnParameters: Same<
    Parameters<TranscriptSource["openTurn"]>,
    [request: ConversationRequest, options: { signal: AbortSignal }]
  > = true;

  const plain = plainTurn();
  // `items` is optional, as Python's `()` default.
  expect(await plain.request("model", {})).toBe(1);
  expect(typeof plain.prepareHistory).toBe("undefined");
  expect(typeof plain.interrupt).toBe("undefined");
  const capable: ConversationTurn = {
    ...plain,
    prepareHistory: async ({ project }) => [null, await project(plain.view, [])],
    interrupt() {},
  };
  expect(typeof capable.prepareHistory).toBe("function");
  expect(typeof capable.interrupt).toBe("function");

  const source: TranscriptSource = {
    openTurn: (_request, { signal }) => {
      signal.throwIfAborted();
      return capable;
    },
  };
  const request = new ConversationRequest(["root", "agent"], invocation(), {});
  expect(await source.openTurn(request, { signal: new AbortController().signal })).toBe(capable);
});

// test_conversation_identifiers_are_exact_nonempty_unicode
test.each(
  (
    [
      ["run_id", (value: never) => new ConversationInvocation(value, 1)],
      ["item_id", (value: never) => new ConversationItem(value, "text", {})],
      ["format", (value: never) => new ConversationItem("id", value, {})],
      ["namespace", (value: never) => new ConversationStamp(invocation(), value, 0, 0, 0)],
      ["namespace", (value: never) => view({ namespace: value })],
    ] as const
  ).flatMap(([name, factory]) =>
    (
      [
        ["", RangeError, `${name} must be non-empty`],
        ["\ud800", RangeError, "Unicode surrogate is not valid canonical JSON"],
        [1, TypeError, `${name} must be a string`],
        [null, TypeError, `${name} must be a string`],
      ] as const
    ).map(([value, cls, message]) => [name, value, factory, cls, message] as const),
  ),
)("conversation %s refuses %p", (_name, value, factory, cls, message) => {
  expectError(() => factory(value as never), cls, message);
});

// test_effect_kind_is_closed
test.each<[unknown, ErrorClass, string]>([
  ["other", RangeError, "unsupported conversation effect kind"],
  ["", RangeError, "kind must be non-empty"],
  [true, TypeError, "kind must be a string"],
])("the effect kind refuses %p", (kind, cls, message) => {
  expectError(
    () => new ConversationEffectState(invocation(), 1, kind as ConversationEffectKind, {}, "pending", [], null),
    cls,
    message,
  );
  expectError(
    () => new ConversationEffectRequested(stamp(1, 1), kind as ConversationEffectKind, {}, []),
    cls,
    message,
  );
});

// test_terminal_effect_requires_observation
test.each<ConversationEffectStatus>(["completed", "failed", "interrupted"])(
  "a %s effect requires an observation",
  (status) => {
    expectError(
      () =>
        new ConversationEffectState(invocation(), 1, "model", {}, status, [], status === "failed" ? failure() : null),
      RangeError,
      "terminal effect requires an observation",
    );
  },
);

// test_pending_effect_can_have_no_observations
test("a pending effect can have no observations", () => {
  const effect = new ConversationEffectState(invocation(), 1, "model", {}, "pending", [], null);
  expect(effect.observationOrdinals).toEqual([]);
});

// test_view_counters_are_exact_nonnegative_integers
test.each(
  (["generation", "revision"] as const).flatMap((field) =>
    (
      [
        [true, TypeError, "must be an integer"],
        [-1, RangeError, "must be non-negative"],
        [1.5, TypeError, "must be an integer"],
      ] as const
    ).map(([value, cls, text]) => [field, value, cls, text] as const),
  ),
)("a view refuses %s %p", (field, value, cls, text) => {
  expectError(() => view({ [field]: value }), cls, `${field} ${text}`);
});

// test_nested_carrier_subclasses_are_rejected
test("nested carrier subclasses are refused", () => {
  class InvocationSubclass extends ConversationInvocation {}
  class ItemSubclass extends ConversationItem {}
  class ViewSubclass extends ConversationView {}
  expectError(
    () => new ConversationRequest(["root", "agent"], new InvocationSubclass("run", 1), {}),
    TypeError,
    "invocation must be a ConversationInvocation",
  );
  expectError(
    () => view({ revision: 1, applied: [applied()], items: [new ItemSubclass("id", "text", {})] }),
    TypeError,
    "items must contain ConversationItem values",
  );
  const baseline = view();
  expectError(
    () => new ConversationAdmitted(stamp(), new ViewSubclass(baseline.namespace, 0, 0, {}, [], [], [], [])),
    TypeError,
    "baseline must be a ConversationView",
  );
});

// test_nested_view_carriers_are_revalidated_and_detached, without its forged rows
test("a view keeps its nested carriers by reference", () => {
  const effect = new ConversationEffectState(invocation(), 1, "model", {}, "failed", [2], failure());
  const completion = new ConversationCompletion(invocation(), 3, "failed");
  const item = new ConversationItem("id", "text", {});
  const entry = applied();
  const items = [item];
  const kept = view({ revision: 1, items, effects: [effect], closedInvocations: [completion], applied: [entry] });
  items.push(new ConversationItem("other", "text", {}));
  expect(kept.items).toEqual([item]);
  expect(kept.items[0]).toBe(item);
  expect(kept.effects[0]).toBe(effect);
  expect(kept.closedInvocations[0]).toBe(completion);
  expect(kept.applied[0]).toBe(entry);
  expectError(
    () => view({ revision: 1, applied: [applied()], effects: [completion] }),
    TypeError,
    "effects must contain ConversationEffectState values",
  );
  expectError(
    () => view({ revision: 1, applied: [applied()], closedInvocations: [effect] }),
    TypeError,
    "closed_invocations must contain ConversationCompletion values",
  );
  expectError(
    () => view({ revision: 1, applied: [{ ...entry }] }),
    TypeError,
    "applied must contain ConversationApplied values",
  );
});

// test_effect_checkpoints_reject_zero_ordinal_and_list_items
test("effect checkpoints refuse ordinal zero and non-array items", () => {
  expectError(
    () => new ConversationEffectRequested(stamp(), "model", {}, []),
    RangeError,
    "only conversation admission has ordinal zero",
  );
  expectError(
    () => new ConversationEffectRequested(stamp(1, 1), "model", {}, new Set() as never),
    TypeError,
    "items must be an array",
  );
  expectError(
    () => new ConversationEffectObserved(stamp(2, 2), 1, "pending", new Set() as never, null),
    TypeError,
    "items must be an array",
  );
});

// test_memory_context_request_detaches_every_value_boundary
test("a memory context request snapshots its JSON and keeps its carriers", () => {
  const inputs = { nested: ["original"] };
  const compatibility = { provider: "test", format: { version: 1n } };
  const recall = new RecallContext("system", "prompt", inputs, ["root", "agent"]);
  const identity = new ConversationInvocation("run", 3);
  const request = new MemoryContextRequest(recall, identity, compatibility);
  inputs.nested.push("outside");
  compatibility.format.version = 2n;
  expect(() => {
    (request.recall.inputs as { nested: string[] }).nested.push("returned");
  }).toThrow(TypeError);
  expect(() => {
    (request.compatibility as { format: { version: bigint } }).format.version = 3n;
  }).toThrow(TypeError);
  expect(() => {
    (recall as { prompt: string }).prompt = "mutated";
  }).toThrow(TypeError);
  expect(() => {
    (identity as { runId: string }).runId = "mutated";
  }).toThrow(TypeError);
  expect(request.recall.inputs).toEqual({ nested: ["original"] });
  expect(request.recall.prompt).toBe("prompt");
  expect(request.compatibility).toEqual({ provider: "test", format: { version: 1n } });
  expect(request.recall).toBe(recall);
  expect(request.invocation).toBe(identity);
  expect(() => {
    (request as { recall: RecallContext }).recall = recall;
  }).toThrow(TypeError);
  expect(new MemoryContextRequest(request.recall, null, {}).invocation).toBeNull();
});

class SubRecall extends RecallContext {}

// test_memory_context_request_rejects_invalid_carriers
test.each<[string, string, unknown, ErrorClass, string]>([
  ["a null recall", "recall", null, TypeError, "recall must be a RecallContext"],
  ["a plain-object recall", "recall", {}, TypeError, "recall must be a RecallContext"],
  ["a recall subclass", "recall", new SubRecall("", "", {}, ["a"]), TypeError, "recall must be a RecallContext"],
  ["a plain-object invocation", "invocation", {}, TypeError, "invocation must be a ConversationInvocation"],
  ["a string invocation", "invocation", "run", TypeError, "invocation must be a ConversationInvocation"],
  ["an array compatibility", "compatibility", [], TypeError, "compatibility must be an exact JSON object"],
  ["NaN", "compatibility", { n: Number.NaN }, RangeError, "compatibility.n JSON number must not be NaN"],
])("a memory context request refuses %s", (_name, field, value, cls, message) => {
  const args: Record<string, unknown> = {
    recall: new RecallContext("", "", {}, ["root", "agent"]),
    invocation: invocation(),
    compatibility: {},
    [field]: value,
  };
  const build = () => new MemoryContextRequest(args.recall as never, args.invocation as never, args.compatibility);
  expectError(build, cls, message);
});

// test_memory_content_validates_explicit_transcript_sources
test("memory content validates explicit transcript sources", () => {
  const source: TranscriptSource = { openTurn: () => plainTurn() };
  const content = new MemoryContent({ text: "notes", transcript: source });
  expect(content.text).toBe("notes");
  expect(content.transcript).toBe(source);
  expect(new MemoryContent()).toEqual(new MemoryContent({ text: "", transcript: null }));
  expect([new MemoryContent().text, new MemoryContent().transcript]).toEqual(["", null]);
  for (const [text, cls, message] of [
    [null, TypeError, "text must be a string"],
    [7, TypeError, "text must be a string"],
    ["\ud800", RangeError, "Unicode surrogate is not valid canonical JSON"],
  ] as const) {
    expectError(() => new MemoryContent({ text: text as never }), cls, message);
  }
  for (const bad of [new Point(), { openTurn: null }]) {
    expectError(
      () => new MemoryContent({ transcript: bad as never }),
      TypeError,
      "transcript must supply openTurn or be null",
    );
  }
  expect(() => {
    (content as { text: string }).text = "changed";
  }).toThrow(TypeError);
});

test("a memory context requires exact sections and keeps its transcript", () => {
  const empty = new MemoryContext();
  expect([empty.sections, empty.transcript]).toEqual([[], null]);
  const section = new MemorySection("notes", "outer");
  const sections = [section];
  const turn = plainTurn();
  const context = new MemoryContext({ sections, transcript: turn });
  sections.push(new MemorySection("chat", "inner"));
  expect(context.sections).toEqual([section]);
  expect(context.sections[0]).toBe(section);
  expect(context.transcript).toBe(turn);
  expect(Object.isFrozen(context)).toBe(true);
  expect(Object.isFrozen(context.sections)).toBe(true);
  expectError(() => new MemoryContext({ sections: new Set() as never }), TypeError, "sections must be an array");
  expectError(
    () => new MemoryContext({ sections: [{ memory: "notes", text: "" }] as never }),
    TypeError,
    "sections must contain MemorySection values",
  );
});

// test_memory_context_carriers_are_contracts_only_exports, its contracts.ts half
test("the memory context carriers are contracts exports", () => {
  for (const name of ["MemoryContent", "MemoryContext", "MemoryContextRequest"]) expect(name in contracts).toBe(true);
  expect("ConversationFrame" in contracts).toBe(false);
  // `TranscriptSource` is an interface, so the typecheck pins its export.
  const source: contracts.TranscriptSource | null = null;
  // @ts-expect-error the stop check is the turn's AbortSignal, not an export
  const check: contracts.CheckStopped | null = null;
});

// test_conversation_carriers_have_no_owner_key, its field-order half
test("conversation carriers have no owner key", () => {
  expect("ConversationKey" in contracts).toBe(false);
  expect(Object.keys(stamp())).toEqual(["invocation", "namespace", "generation", "expectedRevision", "ordinal"]);
  expect(Object.keys(view())).not.toContain("key");
});

test("every conversation carrier and its arrays are frozen", () => {
  const item = new ConversationItem("id", "text", {});
  const effect = new ConversationEffectState(invocation(), 1, "model", {}, "completed", [2], null);
  const full = view({
    revision: 1,
    items: [item],
    effects: [effect],
    closedInvocations: [new ConversationCompletion(invocation(), 3, "completed")],
    applied: [applied()],
  });
  const carriers: object[] = [
    invocation(),
    new ConversationRequest(["root", "agent"], invocation(), {}),
    item,
    stamp(),
    effect,
    effect.observationOrdinals,
    full.closedInvocations[0] as object,
    full.applied[0] as object,
    full,
    full.items,
    full.effects,
    full.closedInvocations,
    full.applied,
    new ConversationAdmitted(stamp(), view()),
    new ConversationEffectRequested(stamp(1, 1), "tool", {}, [item]),
    new ConversationEffectObserved(stamp(2, 2), 1, "completed", [item], null),
    new ConversationClosed(stamp(3, 3), "completed"),
  ];
  for (const carrier of carriers) expect(Object.isFrozen(carrier)).toBe(true);
});

// The conversation fold, codecs and digests.

/** Wire JSON a row reaches into at any field. */
type Wire = any;

/** A mutable deep copy of frozen wire JSON, so a row can damage it as Python edits its dict. */
function thaw(value: unknown): Wire {
  return structuredClone(value);
}

/** The canonical text of a view's wire: equal text means an unchanged view. */
function viewText(value: ConversationView): string {
  return logJson(conversationViewToJson(value));
}

/** Assert that `fn` throws a non-retryable `MemoryOperationError` with `stage`, `code` and `message`. */
function expectOperationError(fn: () => unknown, stage: string, code: string, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(MemoryOperationError);
  const error = caught as MemoryOperationError;
  expect([error.stage, error.code, error.retryable, error.message]).toEqual([stage, code, false, message]);
}

function nextStamp(revision: number, ordinal = 0, runId = "next", startSequence = 1): ConversationStamp {
  return new ConversationStamp(new ConversationInvocation(runId, startSequence), "namespace", 0, revision, ordinal);
}

function admitted(): ConversationView {
  const baseline = view();
  return advanceConversation(baseline, new ConversationAdmitted(stamp(), baseline));
}

function requested(): ConversationView {
  return advanceConversation(admitted(), new ConversationEffectRequested(stamp(1, 1), "model", {}, []));
}

function completeHistory(): [ConversationView[], ConversationCheckpoint[]] {
  const baseline = view({ compatibility: { provider: "fake", options: [true, 1n, 1] } });
  const events: ConversationCheckpoint[] = [
    new ConversationAdmitted(stamp(), baseline),
    new ConversationEffectRequested(stamp(1, 1), "model", { arguments: [true, 1n, 1] }, [
      new ConversationItem("user", "message", { text: "héllo" }),
    ]),
    new ConversationEffectObserved(
      stamp(2, 2),
      1,
      "completed",
      [new ConversationItem("assistant", "message", { text: "hello" })],
      null,
    ),
    new ConversationClosed(stamp(3, 3), "completed"),
  ];
  const views = [baseline];
  for (const event of events) views.push(advanceConversation(views.at(-1)!, event));
  return [views, events];
}

function changedView(value: ConversationView, changes: Partial<ViewFields>): ConversationView {
  return view({
    namespace: value.namespace,
    generation: value.generation,
    revision: value.revision,
    compatibility: value.compatibility,
    items: value.items,
    effects: value.effects,
    closedInvocations: value.closedInvocations,
    applied: value.applied,
    ...changes,
  });
}

function withDigest(entry: ConversationApplied, fields: { event?: string; after?: string }): ConversationApplied {
  return new ConversationApplied(
    entry.operationId,
    fields.event ?? entry.eventDigest,
    entry.beforeDigest,
    fields.after ?? entry.afterDigest,
  );
}

function reseal(value: ConversationView): ConversationView {
  return changedView(value, {
    applied: [...value.applied.slice(0, -1), withDigest(value.applied.at(-1)!, { after: conversationContentDigest(value) })],
  });
}

// test_content_digest_excludes_ledger
test("the content digest excludes the ledger", () => {
  const first = view({ revision: 1, applied: [applied()] });
  const second = view({ revision: 1, applied: [withDigest(applied(), { event: "d".repeat(64) })] });
  expect(conversationContentDigest(first)).toBe(conversationContentDigest(second));
  expect(conversationContentDigest(first)).not.toBe(conversationContentDigest(view()));
  expect(conversationContentDigest(first)).toHaveLength(64);
  const variants = [true, 1n, 1].map((value) => view({ compatibility: { number: value } }));
  expect(new Set(variants.map(conversationContentDigest)).size).toBe(3);
});

// test_admission_advances_revision_once
test("admission advances the revision once", () => {
  const before = view();
  const checkpoint = new ConversationAdmitted(stamp(), before);
  const after = advanceConversation(before, checkpoint);
  expect(before).toEqual(view());
  expect(after.revision).toBe(1);
  expect([after.items, after.effects, after.closedInvocations]).toEqual([[], [], []]);
  expect(after.applied).toEqual([
    new ConversationApplied(
      '["run",1,0]',
      checkpointDigest(checkpoint),
      conversationContentDigest(before),
      conversationContentDigest(after),
    ),
  ]);
  expectOperationError(
    () => advanceConversation(after, checkpoint),
    "conversation",
    "session.conflict",
    "checkpoint does not name the current predecessor",
  );
});

// test_request_appends_new_items_and_pending_effect
test("a request appends new items and a pending effect", () => {
  const before = admitted();
  const item = new ConversationItem("user-1", "message", { role: "user", text: "Hi" });
  const after = advanceConversation(before, new ConversationEffectRequested(stamp(1, 1), "model", { model: "fake" }, [item]));
  expect([before.items, before.effects]).toEqual([[], []]);
  expect(after.items).toEqual([item]);
  expect(after.effects).toEqual([
    new ConversationEffectState(invocation(), 1, "model", { model: "fake" }, "pending", [], null),
  ]);
  expect(after.revision).toBe(2);
  const appended = advanceConversation(
    after,
    new ConversationEffectRequested(stamp(2, 2), "tool", { arguments: [1n] }, []),
  );
  expect(appended.items).toEqual(after.items);
  expect(appended.effects[0]).toEqual(after.effects[0]!);
  expect(appended.effects[1]!.effectOrdinal).toBe(2);
});

// test_observation_changes_only_matching_effect
test("an observation changes only the matching effect", () => {
  const before = advanceConversation(requested(), new ConversationEffectRequested(stamp(2, 2), "tool", {}, []));
  const item = new ConversationItem("assistant-1", "message", { text: "Hi" });
  const after = advanceConversation(before, new ConversationEffectObserved(stamp(3, 3), 1, "pending", [item], null));
  expect(before.items).toEqual([]);
  expect(before.effects[0]!.observationOrdinals).toEqual([]);
  expect(after.items).toEqual([item]);
  expect(after.effects[0]!.observationOrdinals).toEqual([3]);
  expect(after.effects[0]!.status).toBe("pending");
  expect(after.effects[1]).toEqual(before.effects[1]!);
  const complete = advanceConversation(after, new ConversationEffectObserved(stamp(4, 4), 1, "completed", [], null));
  expect(complete.effects[0]!.status).toBe("completed");
  expect(complete.effects[0]!.observationOrdinals).toEqual([3, 4]);
  expect(complete.items).toEqual([item]);
  expect(complete.effects[1]).toEqual(before.effects[1]!);
});

// test_close_preserves_unresolved_effects
test.each<ConversationTurnStatus>(["failed", "interrupted"])("closing as %s keeps unresolved effects", (status) => {
  const before = requested();
  const after = advanceConversation(before, new ConversationClosed(stamp(2, 2), status));
  expect(after.effects).toEqual(before.effects);
  expect(after.items).toEqual(before.items);
  expect(after.closedInvocations).toEqual([new ConversationCompletion(invocation(), 2, status)]);
  expectOperationError(
    () => advanceConversation(before, new ConversationClosed(stamp(2, 2), "completed")),
    "conversation",
    "session.conflict",
    "successful closure cannot leave pending effects",
  );
  expect(before.closedInvocations).toEqual([]);
  expect(before.revision).toBe(2);
});

// test_transition_rejects_skipped_ordinal
test("a transition refuses a skipped ordinal", () => {
  const before = admitted();
  const event = new ConversationEffectRequested(stamp(2, 1), "model", {}, []);
  const original = viewText(before);
  expectOperationError(
    () => advanceConversation(before, event),
    "conversation",
    "session.conflict",
    "checkpoint must be the current invocation's next ordinal",
  );
  expect(viewText(before)).toBe(original);
  expect(event.stamp.ordinal).toBe(2);
});

// test_admission_preserves_unresolved_or_unclosed_effects
test.each([null, "pending", "completed", "closed_failed"])(
  "admission keeps unresolved or unclosed effects (%p)",
  (status) => {
    let before = requested();
    if (status === "pending" || status === "completed") {
      before = advanceConversation(before, new ConversationEffectObserved(stamp(2, 2), 1, status, [], null));
    } else if (status === "closed_failed") {
      before = advanceConversation(before, new ConversationClosed(stamp(2, 2), "failed"));
    }
    const original = viewText(before);
    const after = advanceConversation(before, new ConversationAdmitted(nextStamp(before.revision), before));
    expect(after.effects).toEqual(before.effects);
    expect(after.closedInvocations).toEqual(before.closedInvocations);
    expect(after.revision).toBe(before.revision + 1);
    expect(viewText(before)).toBe(original);
  },
);

// test_admission_can_follow_effect_free_unclosed_invocation
test("admission can follow an effect-free unclosed invocation", () => {
  const before = admitted();
  const after = advanceConversation(before, new ConversationAdmitted(nextStamp(1), before));
  expect(after.revision).toBe(2);
  expect(after.closedInvocations).toEqual([]);
});

// test_conversation_codecs_round_trip_complete_history_and_detach
test("the conversation codecs round-trip a complete history and detach it", () => {
  const [views, events] = completeHistory();
  for (const value of views) {
    const encoded = conversationViewToJson(value);
    const decoded = conversationViewFromJson(encoded);
    expect(decoded).toEqual(value);
    expect(decoded).not.toBe(value);
    expect(conversationViewToJson(decoded)).toEqual(encoded);
    expect(() => (encoded.compatibility as Wire).options.splice(0)).toThrow(TypeError);
    const copy = thaw(encoded);
    const fromCopy = conversationViewFromJson(copy);
    copy.compatibility.options.length = 0;
    expect(fromCopy.compatibility.options).toEqual([true, 1n, 1]);
    expect(decoded.compatibility.options).toEqual([true, 1n, 1]);
  }
  for (const event of events) {
    const encoded = conversationCheckpointToJson(event);
    const decoded = conversationCheckpointFromJson(encoded);
    expect(decoded).toEqual(event);
    expect(decoded).not.toBe(event);
    expect(Object.isFrozen(encoded)).toBe(true);
    const digest = crypto.createHash("sha256").update(logJson(encoded)).digest("hex");
    expect(checkpointDigest(event)).toBe(digest);
  }
});

const VIEW_NAMES = "namespace generation revision compatibility items reference effects closed_invocations applied";

// test_view_codec_rejects_wrong_closed_wire_shape
test.each<[string, ErrorClass, string]>([
  ["missing", RangeError, `conversation wire record requires exactly: ${VIEW_NAMES}`],
  ["extra", RangeError, `conversation wire record requires exactly: ${VIEW_NAMES}`],
  ["tuple", TypeError, "conversation wire.items must contain exact JSON values"],
  ["bool", TypeError, "revision must be an integer"],
  ["key", RangeError, `conversation wire record requires exactly: ${VIEW_NAMES}`],
])("the view codec refuses a wrong closed wire shape (%s)", (damage, cls, message) => {
  const wire = thaw(conversationViewToJson(view()));
  if (damage === "missing") delete wire.reference;
  else if (damage === "extra") wire.extra = null;
  else if (damage === "tuple") wire.items = new Set();
  else if (damage === "bool") wire.revision = false;
  else wire.key = null;
  expectError(() => conversationViewFromJson(wire), cls, message);
});

// test_view_codec_rejects_broken_digest_chain
test.each([
  ["final", "ledger final digest does not match conversation content"],
  ["adjacency", "ledger digest chain does not match its predecessor"],
  ["anchor", "ledger digest chain does not match its predecessor"],
  ["ordinal", "ledger operation ordinals must be contiguous"],
  ["length", "conversation revision requires one applied entry per checkpoint"],
])("the view codec refuses a broken digest chain (%s)", (damage, message) => {
  const [views] = completeHistory();
  const wire = thaw(conversationViewToJson(views.at(-1)!));
  if (damage === "final") wire.applied.at(-1).after_digest = "0".repeat(64);
  else if (damage === "adjacency") wire.applied[1].before_digest = "0".repeat(64);
  else if (damage === "anchor") wire.applied[0].before_digest = "0".repeat(64);
  else if (damage === "ordinal") wire.applied[1].operation_id = '["run",1,8]';
  else wire.applied.pop();
  expectError(() => conversationViewFromJson(wire), RangeError, message);
  expectError(
    () => conversationViewToJson(view({ revision: 1, applied: [applied()] })),
    RangeError,
    "ledger digest chain does not match its predecessor",
  );
});

// test_view_codec_rejects_impossible_history_even_with_matching_final_digest
test.each([
  ["missing_effect", "every positive operation must have retained effect or closure evidence"],
  ["foreign_effect", "history must name unique accepted operation ordinals"],
  ["overlap", "history must name unique accepted operation ordinals"],
  ["early_close", "history must name unique accepted operation ordinals"],
  ["pending_success", "successful closure cannot retain pending effects"],
  ["foreign_close", "history must name unique accepted operation ordinals"],
])("the view codec refuses impossible history with a matching final digest (%s)", (damage, message) => {
  const [views] = completeHistory();
  let value = views.at(-1)!;
  if (damage === "missing_effect") {
    value = changedView(value, { effects: [] });
  } else if (damage === "foreign_effect") {
    const effect = new ConversationEffectState(
      new ConversationInvocation("foreign", 1),
      1,
      "model",
      {},
      "completed",
      [2],
      null,
    );
    value = changedView(value, { effects: [effect] });
  } else if (damage === "overlap") {
    value = changedView(value, {
      effects: [new ConversationEffectState(invocation(), 1, "model", {}, "completed", [3], null)],
    });
  } else if (damage === "early_close") {
    value = changedView(value, { closedInvocations: [new ConversationCompletion(invocation(), 2, "completed")] });
  } else if (damage === "pending_success") {
    value = changedView(value, {
      effects: [new ConversationEffectState(invocation(), 1, "model", {}, "pending", [2], null)],
    });
  } else {
    value = changedView(value, {
      closedInvocations: [new ConversationCompletion(new ConversationInvocation("foreign", 1), 3, "completed")],
    });
  }
  expectError(() => conversationViewToJson(reseal(value)), RangeError, message);
});

// test_invalid_transitions_preserve_input_evidence
test.each([
  ["before_admit", "checkpoint requires an admitted invocation"],
  ["terminal_effect", "a terminal effect cannot receive more observations"],
  ["closed", "invocation is already closed"],
  ["missing_request", "observation has no matching requested effect"],
  ["foreign_invocation", "checkpoint must be the current invocation's next ordinal"],
  ["namespace", "checkpoint does not name the current predecessor"],
  ["generation", "checkpoint does not name the current predecessor"],
  ["revision", "checkpoint does not name the current predecessor"],
  ["duplicate_request_item", "checkpoint repeats an existing conversation item ID"],
  ["duplicate_observed_item", "checkpoint repeats an existing conversation item ID"],
  ["baseline_descriptor", "admission baseline does not match the complete predecessor"],
])("an invalid transition keeps its input evidence (%s)", (scenario, message) => {
  const [views] = completeHistory();
  let before = views[1]!;
  let event: ConversationCheckpoint = new ConversationEffectRequested(stamp(1, 1), "model", {}, []);
  if (scenario === "before_admit") {
    before = views[0]!;
    event = new ConversationEffectRequested(stamp(1, 0), "model", {}, []);
  } else if (scenario === "terminal_effect") {
    before = views[3]!;
    event = new ConversationEffectObserved(stamp(3, 3), 1, "completed", [], null);
  } else if (scenario === "closed") {
    before = views[4]!;
    event = new ConversationEffectRequested(stamp(4, 4), "model", {}, []);
  } else if (scenario === "missing_request") {
    before = advanceConversation(requested(), new ConversationEffectObserved(stamp(2, 2), 1, "pending", [], null));
    event = new ConversationEffectObserved(stamp(3, 3), 2, "completed", [], null);
  } else if (scenario === "duplicate_request_item") {
    before = views[2]!;
    event = new ConversationEffectRequested(stamp(2, 2), "tool", {}, before.items);
  } else if (scenario === "duplicate_observed_item") {
    before = views[2]!;
    event = new ConversationEffectObserved(stamp(2, 2), 1, "pending", before.items, null);
  } else if (scenario === "baseline_descriptor") {
    before = view({ compatibility: { value: true } });
    event = new ConversationAdmitted(stamp(), view({ compatibility: { value: 1n } }));
  } else {
    const fields = { invocation: invocation(), namespace: "namespace", generation: 0, expectedRevision: 1 };
    if (scenario === "foreign_invocation") fields.invocation = new ConversationInvocation("foreign", 1);
    else if (scenario === "namespace") fields.namespace = "foreign";
    else if (scenario === "generation") fields.generation = 1;
    else fields.expectedRevision = 2;
    const changed = new ConversationStamp(fields.invocation, fields.namespace, fields.generation, fields.expectedRevision, 1);
    event = new ConversationEffectRequested(changed, "model", {}, []);
  }
  const original = viewText(before);
  const originalEvent = logJson(conversationCheckpointToJson(event));
  expectOperationError(() => advanceConversation(before, event), "conversation", "session.conflict", message);
  expect(viewText(before)).toBe(original);
  expect(logJson(conversationCheckpointToJson(event))).toBe(originalEvent);
});

// test_later_invocation_reuses_local_ordinals_without_replacing_prior_effects
test("a later invocation reuses local ordinals without replacing prior effects", () => {
  const [views] = completeHistory();
  const first = views.at(-1)!;
  const second = new ConversationInvocation("second", 2);
  const stamps = [0, 1, 2, 3].map((ordinal) => nextStamp(4 + ordinal, ordinal, "second", 2));
  const events: ConversationCheckpoint[] = [
    new ConversationAdmitted(stamps[0]!, first),
    new ConversationEffectRequested(stamps[1]!, "tool", { tool: "fake" }, []),
    new ConversationEffectObserved(stamps[2]!, 1, "failed", [], failure()),
    new ConversationClosed(stamps[3]!, "failed"),
  ];
  let after = first;
  for (const event of events) after = advanceConversation(after, event);
  expect(after.effects.map((effect) => effect.effectOrdinal)).toEqual([1, 1]);
  expect(after.effects[0]).toEqual(first.effects[0]!);
  expect(sameConversationInvocation(after.effects[1]!.invocation, second)).toBe(true);
  expect(after.items).toEqual(first.items);
  expect(conversationViewFromJson(conversationViewToJson(after))).toEqual(after);
});

const CHECKPOINT_NAMES = [
  "type stamp baseline",
  "type stamp kind payload items",
  "type stamp effect_ordinal status items reference failure",
  "type stamp status",
];

// test_checkpoint_codec_rejects_malformed_closed_wire
test.each(
  [0, 1, 2, 3].flatMap((index) =>
    (
      [
        ["extra", RangeError, `conversation wire record requires exactly: ${CHECKPOINT_NAMES[index]}`],
        ["missing", RangeError, `conversation wire record requires exactly: ${CHECKPOINT_NAMES[index]}`],
        ["discriminator", RangeError, "unsupported conversation checkpoint type"],
        ["stamp", RangeError, "conversation wire record requires exactly: run_id start_sequence"],
        ["bool", TypeError, "ordinal must be an integer"],
        ["array", TypeError, "namespace must be a string"],
      ] as const
    ).map(([damage, cls, message]) => [index, damage, cls, message] as const),
  ),
)("the checkpoint codec refuses malformed closed wire (event %p, %s)", (index, damage, cls, message) => {
  const [, events] = completeHistory();
  const wire = thaw(conversationCheckpointToJson(events[index]!));
  if (damage === "extra") wire.extra = null;
  else if (damage === "missing") delete wire.stamp;
  else if (damage === "discriminator") wire.type = "unknown";
  else if (damage === "stamp") wire.stamp.invocation.extra = null;
  else if (damage === "bool") wire.stamp.ordinal = true;
  else wire.stamp.namespace = ["namespace"];
  expectError(() => conversationCheckpointFromJson(wire), cls, message);
});

// test_codecs_reject_non_json_nested_values
test.each<[string, unknown, ErrorClass, string]>([
  ["NaN", Number.NaN, RangeError, "conversation wire.compatibility.bad JSON number must not be NaN"],
  ["a lone surrogate", "\ud800", RangeError, "Unicode surrogate is not valid canonical JSON"],
  ["a Set", new Set(), TypeError, "conversation wire.compatibility.bad must contain exact JSON values"],
  ["a class instance", new Point(), TypeError, "conversation wire.compatibility.bad must contain exact JSON values"],
])("the codecs refuse a non-JSON nested value (%s)", (_name, bad, cls, message) => {
  const wire = thaw(conversationViewToJson(view()));
  wire.compatibility.bad = bad;
  expectError(() => conversationViewFromJson(wire), cls, message);
});

test("the codecs keep an infinite nested value", () => {
  const wire = thaw(conversationViewToJson(view()));
  wire.compatibility.bad = Number.NEGATIVE_INFINITY;
  const decoded = conversationViewFromJson(wire);
  expect(decoded.compatibility.bad).toBe(Number.NEGATIVE_INFINITY);
  expect(conversationViewFromJson(conversationViewToJson(decoded))).toEqual(decoded);
});

// test_codecs_reject_cycles_and_non_string_keys
test("the codecs refuse cycles and symbol keys", () => {
  let wire = thaw(conversationViewToJson(view()));
  wire.compatibility.cycle = wire;
  expectError(
    () => conversationViewFromJson(wire),
    RangeError,
    "conversation wire.compatibility.cycle contains a circular JSON value",
  );
  wire = thaw(conversationViewToJson(view()));
  wire.compatibility[Symbol("key")] = "invalid";
  expectError(
    () => conversationViewFromJson(wire),
    TypeError,
    "conversation wire.compatibility JSON object keys must be strings",
  );
});

test("a 10 000-level payload goes through the view codec and the fold", () => {
  let nested: unknown = "value";
  for (let i = 0; i < 10_000; i++) nested = { nested };
  const item = new ConversationItem("deep", "message", nested);
  const event = new ConversationEffectRequested(stamp(1, 1), "model", {}, [item]);
  const wire = conversationCheckpointToJson(event);
  expect(logJson(conversationCheckpointToJson(conversationCheckpointFromJson(wire)))).toBe(logJson(wire));
  const after = advanceConversation(admitted(), event);
  const encoded = conversationViewToJson(after);
  const decoded = conversationViewFromJson(encoded);
  expect(logJson(conversationViewToJson(decoded))).toBe(logJson(encoded));
  expect(logJson(decoded.items[0]!.payload)).toBe(logJson(nested));
});

test("a view with a -0 payload round-trips and digests as 0", () => {
  const history = (zero: number) => {
    const baseline = view({ compatibility: { v: zero } });
    const before = advanceConversation(baseline, new ConversationAdmitted(stamp(), baseline));
    const item = new ConversationItem("item", "message", { v: [zero] });
    return advanceConversation(before, new ConversationEffectRequested(stamp(1, 1), "model", { v: zero }, [item]));
  };
  const negative = history(-0);
  const decoded = conversationViewFromJson(conversationViewToJson(negative));
  expect(decoded).toEqual(negative);
  expect(Object.is((decoded.items[0]!.payload.v as number[])[0], 0)).toBe(true);
  expect(conversationContentDigest(negative)).toBe(conversationContentDigest(history(0)));
  expect(viewText(negative)).toBe(viewText(history(0)));
  const wire = thaw(conversationViewToJson(view()));
  wire.compatibility.v = -0;
  expect(Object.is(conversationViewFromJson(wire).compatibility.v, 0)).toBe(true);
});

// test_checkpoint_digest_includes_complete_admission_ledger
test("the checkpoint digest includes the complete admission ledger", () => {
  const [views] = completeHistory();
  const baseline = views.at(-1)!;
  const changed = changedView(baseline, {
    applied: [withDigest(baseline.applied[0]!, { event: "0".repeat(64) }), ...baseline.applied.slice(1)],
  });
  expect(checkpointDigest(new ConversationAdmitted(nextStamp(4), baseline))).not.toBe(
    checkpointDigest(new ConversationAdmitted(nextStamp(4), changed)),
  );
});

// test_view_codec_rejects_invalid_nested_values
test.each<[string, ErrorClass, string]>([
  ["duplicate_item", RangeError, "conversation item IDs must be unique"],
  ["duplicate_operation", RangeError, "applied operation IDs must be unique"],
  ["bad_status", RangeError, "invalid conversation effect status"],
  ["failure_type", TypeError, "retryable must be a boolean"],
  ["key_extra", RangeError, `conversation wire record requires exactly: ${VIEW_NAMES}`],
  ["float_sequence", TypeError, "start_sequence must be an integer"],
])("the view codec refuses an invalid nested value (%s)", (damage, cls, message) => {
  const [views] = completeHistory();
  const wire = thaw(conversationViewToJson(views.at(-1)!));
  if (damage === "duplicate_item") wire.items.push(wire.items[0]);
  else if (damage === "duplicate_operation") wire.applied.at(-1).operation_id = wire.applied[0].operation_id;
  else if (damage === "bad_status") wire.effects[0].status = "unknown";
  else if (damage === "failure_type") {
    wire.effects[0].status = "failed";
    wire.effects[0].failure = { stage: "effect", code: "fake.failed", message: "Failed", retryable: 0n };
  } else if (damage === "key_extra") wire.key = {};
  else wire.effects[0].invocation.start_sequence = 1;
  const original = logJson(wire);
  expectError(() => conversationViewFromJson(wire), cls, message);
  expect(logJson(wire)).toBe(original);
});

test("a counter beyond the safe integer range is refused", () => {
  const wire = thaw(conversationViewToJson(view()));
  wire.generation = 2n ** 53n;
  expectError(() => conversationViewFromJson(wire), RangeError, "generation must be a safe integer");
});

// test_view_codec_accepts_admission_after_unclosed_effectful_invocation
test("the view codec accepts admission after an unclosed effectful invocation", () => {
  const [views] = completeHistory();
  const before = views[3]!;
  const after = advanceConversation(before, new ConversationAdmitted(nextStamp(before.revision), before));
  expect(conversationViewFromJson(conversationViewToJson(after))).toEqual(after);
});

test("the codecs and the fold require exact carriers", () => {
  class SubView extends ConversationView {}
  const sub = new SubView("namespace", 0, 0, {}, [], [], [], []);
  for (const value of [{ ...view() }, sub]) {
    expectError(() => conversationViewToJson(value as never), TypeError, "baseline must be a ConversationView");
    expectError(() => conversationContentDigest(value as never), TypeError, "baseline must be a ConversationView");
    expectError(
      () => advanceConversation(value as never, new ConversationAdmitted(stamp(), view())),
      TypeError,
      "baseline must be a ConversationView",
    );
  }
  class SubClosed extends ConversationClosed {}
  const message = "checkpoints must contain exact ConversationCheckpoint values";
  for (const value of [{ ...new ConversationClosed(stamp(1, 1), "failed") }, new SubClosed(stamp(1, 1), "failed")]) {
    expectError(() => conversationCheckpointToJson(value as never), TypeError, message);
    expectError(() => checkpointDigest(value as never), TypeError, message);
    expectError(() => advanceConversation(admitted(), value as never), TypeError, message);
  }
  expectError(() => conversationViewFromJson([] as never), TypeError, "conversation view wire must be an exact JSON object");
  expectError(() => conversationCheckpointFromJson({ type: 1n } as never), TypeError, "checkpoint type must be a string");
  expectError(
    () => conversationCheckpointFromJson({ type: "constructor" } as never),
    RangeError,
    "unsupported conversation checkpoint type",
  );
});

// test_fold_does_not_mask_unexpected_host_fault
test("the fold doesn't mask an unexpected host fault", () => {
  const before = admitted();
  const checkpoint = new ConversationEffectRequested(stamp(1, 1), "model", {}, []);
  const fault = new Error("host digest fault");
  const spy = spyOn(crypto, "createHash").mockImplementation(() => {
    throw fault;
  });
  let caught: unknown;
  try {
    advanceConversation(before, checkpoint);
  } catch (error) {
    caught = error;
  } finally {
    spy.mockRestore();
  }
  expect(caught).toBe(fault);
  expect(before.revision).toBe(1);
  expect(before.effects).toEqual([]);
});

// test_checkpoint_codec_rejects_retired_native_state_without_rewriting
test.each([0, 1, 2])("the checkpoint codec refuses retired native state without rewriting it (event %p)", (index) => {
  const [, events] = completeHistory();
  const wire = thaw(conversationCheckpointToJson(events[index]!));
  if (index === 0) wire.baseline.reference = { provider: "codex", session_id: "thread" };
  else if (index === 1) wire.kind = "native_session";
  else wire.reference = { provider: "codex", session_id: "thread" };
  const original = thaw(wire);
  expectOperationError(
    () => conversationCheckpointFromJson(wire),
    "snapshot",
    "session.unsupported_history",
    "Native session history is unsupported; use fresh SessionMemory.",
  );
  expect(wire).toEqual(original);
});

// The summary.ts whole-turn selection and summary prefix proofs.

function summaryStamp(runId: string, revision: number, ordinal: number): ConversationStamp {
  return new ConversationStamp(new ConversationInvocation(runId, 1), "namespace", 0, revision, ordinal);
}

/** One effectful closed turn, an effect-free unclosed turn, then the current admission. */
function summaryView(): ConversationView {
  const baseline = view();
  const events: ((before: ConversationView) => ConversationCheckpoint)[] = [
    (before) => new ConversationAdmitted(summaryStamp("run", 0, 0), before),
    () =>
      new ConversationEffectRequested(summaryStamp("run", 1, 1), "model", {}, [
        new ConversationItem("user", "message", { text: "héllo" }),
      ]),
    () =>
      new ConversationEffectObserved(
        summaryStamp("run", 2, 2),
        1,
        "completed",
        [new ConversationItem("assistant", "message", { text: "hi" })],
        null,
      ),
    () => new ConversationClosed(summaryStamp("run", 3, 3), "completed"),
    (before) => new ConversationAdmitted(summaryStamp("next", 4, 0), before),
    (before) => new ConversationAdmitted(summaryStamp("third", 5, 0), before),
  ];
  return events.reduce((before, event) => advanceConversation(before, event(before)), baseline);
}

function summaryInvocations(...runIds: string[]): ConversationInvocation[] {
  return runIds.map((runId) => new ConversationInvocation(runId, 1));
}

test("admissions read turn order, including effect-free turns", () => {
  const turns = admissions(summaryView());
  expect(turns).toEqual(summaryInvocations("run", "next", "third"));
  expect(admissions(view())).toEqual([]);
});

test("the prefix digest equals Python yian's and proves every earlier ledger entry", () => {
  const value = summaryView();
  // Computed with Python yian's `prefix_digest` over the same history.
  const original = "b76c35cd21ab71ef39c7f3eb541c6904f9498fd9195b16d9425a4e265c33b67c";
  expect(prefixDigest(value, 2)).toBe(original);
  expect(prefixDigest(value, 0)).toBe("4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945");
  for (const index of value.applied.keys()) {
    for (const field of ["event", "after"] as const) {
      const entries = value.applied.with(index, withDigest(value.applied[index]!, { [field]: "d".repeat(64) }));
      const changed = prefixDigest(changedView(value, { applied: entries }), 2);
      // The first five entries precede the third admission; the last is that admission.
      if (index < 5) expect(changed).not.toBe(original);
      else expect(changed).toBe(original);
    }
  }
});

test("the prefix digest requires a next admission", () => {
  const value = summaryView();
  expectError(() => prefixDigest(value, 3), RangeError, "a summary prefix must precede the current admission");
  expectError(() => prefixDigest(view(), 0), RangeError, "a summary prefix must precede the current admission");
});

test("a summary is frozen and copies its covered invocations", () => {
  const covered = summaryInvocations("run");
  const summary = new Summary("namespace", 0, "v1", covered, "a".repeat(64), "text");
  covered.push(new ConversationInvocation("next", 1));
  expect(summary.covered).toEqual(summaryInvocations("run"));
  expect(Object.isFrozen(summary)).toBe(true);
  expect(Object.isFrozen(summary.covered)).toBe(true);
  expect([summary.namespace, summary.generation, summary.projectionVersion, summary.digest, summary.text]).toEqual([
    "namespace",
    0,
    "v1",
    "a".repeat(64),
    "text",
  ]);
});

test.each([
  ["", "empty"],
  [" \t\n", "whitespace"],
  ["\x1c", "a file separator"],
  ["\u3000", "an ideographic space"],
  ["\x85", "NEL, whitespace only to Python"],
])("a summary of %p is blank (%s)", (text) => {
  expectError(
    () => new Summary("namespace", 0, "v1", [], "a".repeat(64), text),
    RangeError,
    "summary must contain nonempty text",
  );
});

test("a summary needs string text; a byte order mark isn't blank", () => {
  for (const text of [1, null, new String("text")]) {
    expectError(
      () => new Summary("namespace", 0, "v1", [], "a".repeat(64), text as never),
      TypeError,
      "summary must be a string",
    );
  }
  expect(new Summary("namespace", 0, "v1", [], "a".repeat(64), "\ufeff").text).toBe("\ufeff");
});

test("a summary matches only its own namespace, generation, version and complete prefix", () => {
  const value = summaryView();
  const prior = admissions(value);
  const covered = summaryInvocations("run", "next");
  const make = (changes: { namespace?: string; generation?: number; covered?: ConversationInvocation[]; digest?: string }) =>
    new Summary(
      changes.namespace ?? "namespace",
      changes.generation ?? 0,
      "v1",
      changes.covered ?? covered,
      changes.digest ?? prefixDigest(value, changes.covered?.length ?? 2),
      "text",
    );
  expect(make({}).matches(value, prior, "v1")).toBe(true);
  expect(make({ covered: summaryInvocations("run") }).matches(value, prior, "v1")).toBe(true);
  expect(make({}).matches(value, prior.slice(0, 2), "v1")).toBe(true);
  expect(make({ namespace: "other" }).matches(value, prior, "v1")).toBe(false);
  expect(make({ generation: 1 }).matches(value, prior, "v1")).toBe(false);
  expect(make({}).matches(value, prior, "v2")).toBe(false);
  expect(make({ covered: [] }).matches(value, prior, "v1")).toBe(false);
  expect(make({}).matches(value, summaryInvocations("next", "run", "third"), "v1")).toBe(false);
  expect(make({}).matches(value, prior.slice(0, 1), "v1")).toBe(false);
  expect(make({ digest: prefixDigest(value, 1) }).matches(value, prior, "v1")).toBe(false);
  expect(make({ digest: "a".repeat(64) }).matches(value, prior, "v1")).toBe(false);
  expectError(
    () => make({ covered: [...prior], digest: "a".repeat(64) }).matches(value, prior, "v1"),
    RangeError,
    "a summary prefix must precede the current admission",
  );
});
