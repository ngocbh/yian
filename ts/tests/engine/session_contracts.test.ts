/**
 * The conversation carriers, from `tests/engine/test_session_contracts.py`:
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
 * `test_conversation_identifiers_are_exact_nonempty_unicode`, `test_effect_kind_is_closed`,
 * `test_terminal_effect_requires_observation`, `test_pending_effect_can_have_no_observations`,
 * `test_view_counters_are_exact_nonnegative_integers`, `test_nested_carrier_subclasses_are_rejected`,
 * the rows without forgery of `test_nested_view_carriers_are_revalidated_and_detached`,
 * `test_effect_checkpoints_reject_zero_ordinal_and_list_items` and the field-order half of
 * `test_conversation_carriers_have_no_owner_key`; plus JavaScript's own checks: a `-0` payload
 * number reads back `0`, counters must be safe integers, every carrier and its arrays are frozen,
 * `sameConversationInvocation`, and ledger identities compared by value.
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
 * Not ported: the `object.__setattr__` rows of
 * `test_nested_view_carriers_are_revalidated_and_detached` and all of
 * `test_nested_corrupted_identity_is_revalidated` (Python forges a frozen dataclass; a TS carrier is
 * frozen by its constructor and a subclass can't skip it, so none holds an unchecked field).
 *
 * Left for later: `test_memory_protocol_has_no_frames` (`:17`) and the memory-context tests
 * (`:1302`–`:1357`), with the recall contracts; `test_conversation_capabilities_are_optional_and_exact`
 * (`:334`), with the conversation turn and transcript source contracts;
 * `test_conversation_contracts_are_exported_without_concrete_backend` (`:361`) and the export half
 * of `test_conversation_carriers_have_no_owner_key` (`:1367`), with the package exports; and the
 * fold and codec tests (`:523` onward), with `conversation.ts`.
 */

import { expect, test } from "bun:test";

import * as contracts from "../../src/memory/contracts.ts";
import {
  ConversationAdmitted,
  ConversationApplied,
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
  type ConversationTurnStatus,
  ConversationView,
  MemoryOperationFailure,
  sameConversationInvocation,
} from "../../src/memory/contracts.ts";

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
