/**
 * Process-local session memory with backend-owned atomic turns: a port of
 * `tests/engine/test_session_memory.py`.
 *
 * Ported: every test of the file: `test_turn_publishes_history_without_a_checkpoint_sink`,
 * `test_new_turn_preserves_incomplete_history`, `test_session_backend_implements_the_protocol`,
 * `test_snapshot_getter_is_detached`, `test_uncontended_lease_reservation_does_not_change_history`,
 * `test_first_turn_admission_uses_root_and_consumer`, `test_same_agent_reuses_conversation_across_runs`,
 * `test_distinct_consumer_sees_the_shared_history`, `test_open_session_snapshot_stays_pinned`,
 * `test_admission_reads_live_view_not_open_snapshot`, `test_pristine_backend_adopts_snapshot_namespace`,
 * `test_used_backend_rejects_foreign_snapshot`, `test_old_same_lineage_snapshot_stays_pinned_without_rewind`,
 * `test_session_recall_contributes_transcript_without_notes`,
 * `test_session_learn_and_commit_accept_empty_batches`, `test_incompatible_descriptor_requires_reset`,
 * `test_second_same_key_ticket_waits`, `test_third_ticket_cannot_overtake_second`,
 * `test_second_consumer_waits_for_the_first_turn`, `test_cancelled_ticket_is_removed`,
 * `test_reset_rejects_active_backend_without_mutation`, `test_reset_rejects_queued_backend_without_mutation`,
 * `test_idle_reset_clears_the_conversation`, `test_reset_invalidates_old_session_and_turn`,
 * `test_cleanup_annotation_cannot_replace_primary_error`,
 * `test_completed_observation_is_retained_before_checker_stop`,
 * `test_stop_after_publication_preserves_requested_effect`,
 * `test_body_failure_closes_truthfully_without_inventing_observations`,
 * `test_completed_close_with_pending_effect_is_rejected_without_fabrication`,
 * `test_active_invocation_cannot_be_shadowed_by_another_key`, `test_publication_requires_owned_stamp`,
 * `test_session_close_retains_active_owner_until_finalization`,
 * `test_session_close_removes_own_waiters_without_poisoning_other_owner`,
 * `test_stop_and_external_call_do_not_hold_backend_lock`,
 * `test_reset_rejects_queued_ticket_even_without_an_active_lease`,
 * `test_snapshot_envelope_adds_no_nesting_beyond_the_view`,
 * `test_snapshot_rejects_invalid_container_or_nonfinite_json`,
 * `test_stop_checker_cannot_close_session_then_grant_effect_permission`,
 * `test_observed_stop_cannot_be_caught_to_reenable_new_effects`,
 * `test_two_call_sites_share_the_instance_conversation`,
 * `test_used_backend_rejects_newer_same_namespace_snapshot`,
 * `test_invalid_request_wire_publishes_nothing_and_keeps_lease_usable`,
 * `test_session_close_at_reserved_admission_boundary_is_typed_stale`,
 * `test_session_close_never_overlaps_an_inflight_effect_with_the_next_turn`,
 * `test_candidate_snapshot_failure_publishes_nothing`,
 * `test_old_opening_snapshot_reads_latest_history_without_rewind`,
 * `test_malformed_request_never_publishes_partial_evidence`,
 * `test_malformed_observation_leaves_pending_evidence_unchanged`,
 * `test_alternate_same_namespace_history_is_not_a_valid_opening_snapshot`,
 * `test_an_invocation_identity_cannot_be_reused`,
 * `test_turn_exposes_only_message_effects_without_native_tracking` and
 * `test_every_consumer_of_an_instance_shares_one_conversation`.
 *
 * TS-only: the constructor's settings (`true`, `1.5` and `"1"` are `TypeError`s; `0` budget
 * characters, `-1` recent turns and a `bigint` beyond 2^53 − 1 are `RangeError`s; a non-boolean
 * `createParent` is a `TypeError`; `bigint` and `null` settings are accepted); a snapshot whose
 * `generation` is `2n ** 53n` is `session.invalid_snapshot`; `openTurn` rejects a request that
 * isn't exactly a `ConversationRequest` and a missing or non-`AbortSignal` signal with a
 * `TypeError`, leaving no ticket queued or active; `openTurn` with an already-aborted signal, or
 * aborted while queued, rejects with `signal.reason` and leaves no ticket; opening an active
 * invocation again on its session is `session.conflict` at once; `close(error)` never throws
 * `error` (a stop seen meanwhile is noted on it); `close()` after an abort records `interrupted`,
 * releases, then throws `signal.reason`; `interrupt()` closes a normal return as `interrupted`; a
 * second `close()` does nothing.
 *
 * Changed:
 * - `test_session_backend_implements_the_protocol`: `isMemory` and the session's methods stand in
 *   for the runtime-checkable protocol `isinstance` checks.
 * - `test_snapshot_getter_is_detached`: a snapshot is deep-frozen, not a detached mutable copy, so
 *   the assignment throws `TypeError` and the snapshot is unchanged.
 * - `test_pristine_backend_adopts_snapshot_namespace`: the restored value is a `structuredClone` of
 *   the frozen export, edited after the open.
 * - `test_session_learn_and_commit_accept_empty_batches`: an array is the valid batch type, so
 *   `commit("same", [])` commits; a non-array (a `Set`) is the `TypeError`.
 * - `test_snapshot_rejects_invalid_container_or_nonfinite_json`: the `tuple` row is a `Map`; the
 *   `cycle` row adds the cycle to a `structuredClone`d copy; the `deep` row still fails because a
 *   130-level object isn't a view wire (TS has no memory nesting limit).
 * - `test_malformed_request_never_publishes_partial_evidence` and
 *   `test_malformed_observation_leaves_pending_evidence_unchanged`: an array is the valid items
 *   type, so the list rows pass a `Set`; each row pins its error class and message.
 * - `test_invalid_request_wire_publishes_nothing_and_keeps_lease_usable`: TS has no memory nesting
 *   limit, so a 10 000-level payload publishes and the turn stays usable; a NaN payload is the
 *   invalid wire that publishes nothing and gets no note.
 * - `test_snapshot_envelope_adds_no_nesting_beyond_the_view`: TS has no memory nesting limit, so a
 *   200-level payload publishes and the snapshot holds it whole.
 * - `test_new_turn_preserves_incomplete_history`: `close(primary)` returns, where Python's
 *   `__exit__` re-raises; the late observation pins `conversation lease is no longer active`.
 * - `test_incompatible_descriptor_requires_reset`: `(true, 1n)` and `(1n, 1.0)` stay
 *   incompatible; `(0.0, -0.0)` is admitted, since `-0` reads as `0`.
 * - `test_cleanup_annotation_cannot_replace_primary_error`: the note is pushed onto the primary's
 *   own `notes` array (Python's `add_note`); the hostile rows are a cleanup whose `message` getter
 *   throws and a frozen primary, and both keep the primary without a note.
 * - `test_body_failure_closes_truthfully_without_inventing_observations`: the `KeyboardInterrupt`
 *   row is a thrown string, which records `failed`: TS has no `BaseException`.
 * - `test_observed_stop_cannot_be_caught_to_reenable_new_effects`: an abort can't be cleared, so the
 *   second request fails with the same reason, where Python clears its event.
 * - Thread rows (`test_second_same_key_ticket_waits`, `test_third_ticket_cannot_overtake_second`,
 *   `test_second_consumer_waits_for_the_first_turn`, `test_cancelled_ticket_is_removed`,
 *   `test_reset_rejects_queued_backend_without_mutation`,
 *   `test_session_close_removes_own_waiters_without_poisoning_other_owner`,
 *   `test_session_close_never_overlaps_an_inflight_effect_with_the_next_turn`) run concurrent
 *   promises: a waiter is `settle`d, "is waiting" is a macrotask later with the private `queue`
 *   length, and events are `Promise.withResolvers()`.
 * - `check_stopped` callables are an `AbortController`'s signal. The checkers with side effects are
 *   spies: `test_stop_after_publication_preserves_requested_effect` and
 *   `test_stop_checker_cannot_close_session_then_grant_effect_permission` wrap the session's private
 *   `candidateSnapshot` to abort or close the session once the effect is saved;
 *   `test_session_close_at_reserved_admission_boundary_is_typed_stale` wraps its private `reserve`
 *   to close the session once the lease is granted.
 * - `test_stop_and_external_call_do_not_hold_backend_lock`: TS has no backend lock; a snapshot read
 *   succeeds from a `candidateSnapshot` spy and while an effect is in flight.
 * - `test_reset_rejects_queued_ticket_even_without_an_active_lease`: a stop check can't block, so the
 *   row uses the window between a release and the waiter's wake-up: turn A closes, and before any
 *   await `reset()` is busy for B's queued ticket with no active lease.
 * - `test_reset_invalidates_old_session_and_turn`: `open_turn(...).__enter__()` is `openTurn`
 *   rejecting; each call pins its message.
 * - JSON ints in snapshots are `bigint`s; `with` blocks are `withHarness`/`withTurn`; private
 *   members are read and called by element access (`memory["view"]`, `session["reserve"]`).
 *
 * Not ported: none.
 */

import { describe, expect, spyOn, test } from "bun:test";

import { current } from "../../src/defaults.ts";
import * as contracts from "../../src/memory/contracts.ts";
import {
  ConversationEffectRequested,
  ConversationStamp,
  type ConversationView,
  isMemory,
  LearnContext,
  MemoryContent,
  MemoryMutation,
  MemoryOperationError,
  RecallContext,
} from "../../src/memory/contracts.ts";
import { SessionMemory, type Turn } from "../../src/memory/session.ts";
import {
  appendReply,
  caught,
  failure,
  type Harness,
  item,
  macrotask,
  type Mutable,
  NEVER,
  rejected,
  request,
  settle,
  snapshot,
  withHarness,
  withTurn,
} from "./session_fakes.ts";

const STALE_SESSION = "memory session is closed or belongs to an old generation";
const STALE_LEASE = "conversation lease is no longer active";
const RESET_BUSY = "cannot reset active or queued conversations";

/** Check that `fn` throws exactly a `cls` with `message`. */
function expectThrow(fn: () => unknown, cls: ErrorConstructor, message: string | RegExp): void {
  const error = caught(fn);
  expect(Object.getPrototypeOf(error)).toBe(cls.prototype);
  if (typeof message === "string") expect((error as Error).message).toBe(message);
  else expect((error as Error).message).toMatch(message);
}

/** The backend's live conversation view. */
function liveView(h: Harness): ConversationView {
  return h.memory["view"]!;
}

/** An object nested `depth` levels deep. */
function nested(depth: number): Mutable {
  let value: Mutable = {};
  for (let i = 0; i < depth; i++) value = { nested: value };
  return value;
}

test("turn publishes history without a checkpoint sink", async () => {
  const memory = new SessionMemory();
  const session = memory.openRun();
  try {
    await withTurn(session, request(), (turn) => {
      expect(turn.view.revision).toBe(1);
      const ordinal = turn.request("model", { prompt: "hi" }, { items: [item("u", "hi")] });
      expect((snapshot(memory) as Mutable).conversation.revision).toBe(2n);
      turn.observe(ordinal, "completed", { items: [item("a", "hello")] });
      expect(turn.view.effects[0]!.status).toBe("completed");
    });
    const saved: Mutable = snapshot(memory);
    expect(saved.conversation.revision).toBe(4n);
    expect(saved.conversation.items.map((entry: Mutable) => entry.item_id)).toEqual(["u", "a"]);
  } finally {
    session.close();
  }
});

test("session backend implements the protocol", () => {
  const memory = new SessionMemory();
  expect(isMemory(memory)).toBe(true);
  const session = memory.openRun();
  try {
    for (const name of ["recall", "learn", "commit", "close", "openTurn"] as const) {
      expect(typeof session[name]).toBe("function");
    }
    expect("snapshot" in session).toBe(true);
  } finally {
    session.close();
  }
});

test("snapshot getter is detached", () => {
  const session = new SessionMemory().openRun();
  try {
    const first: Mutable = session.snapshot;
    expect(() => {
      first.conversation = { bad: true };
    }).toThrow(TypeError);
    expect(session.snapshot.conversation).toBeNull();
  } finally {
    session.close();
  }
});

test("uncontended lease reservation does not change history", async () => {
  const memory = new SessionMemory();
  const session = memory.openRun();
  try {
    const pinned = session.snapshot;
    const lease = await session["reserve"](request(), NEVER);
    expect(lease.active).toBe(true);
    expect(session.snapshot).toEqual(pinned);
    expect(memory["view"]).toBeNull();
    session["release"](lease);
  } finally {
    session.close();
  }
});

test("first turn admission uses root and consumer", async () => {
  await withHarness(async (h) => {
    await withTurn(h.frame, request(), (turn) => {
      expect(turn.view.revision).toBe(1);
      appendReply(turn, "one", "hello");
    });
    expect(liveView(h).revision).toBe(4);
  });
});

test("same agent reuses conversation across runs", async () => {
  const memory = new SessionMemory();
  await withHarness((h) => withTurn(h.frame, request(), (turn) => appendReply(turn, "one", "hello")), memory);
  await withHarness(
    (h) =>
      withTurn(h.frame, request("b"), (turn) => {
        expect(turn.view.items).toEqual([item("one", "hello")]);
      }),
    memory,
  );
});

test("distinct consumer sees the shared history", async () => {
  await withHarness(async (h) => {
    await withTurn(h.frame, request(), (turn) => {
      const ordinal = turn.request("model", {});
      turn.observe(ordinal, "completed", { items: [item("one", "hello")] });
    });
    await withTurn(h.frame, request("b", 1, ["root", "other"]), (turn) => {
      expect(turn.view.items).toEqual([item("one", "hello")]);
    });
  });
});

test("open session snapshot stays pinned", async () => {
  await withHarness(async (h) => {
    const pinned = h.session.snapshot;
    await withTurn(h.frame, request(), (turn) => appendReply(turn, "one", "hello"));
    expect(h.session.snapshot).toEqual(pinned);
    expect((snapshot(h.memory) as Mutable).conversation.revision).toBe(4n);
  });
});

test("admission reads live view not open snapshot", async () => {
  await withHarness(async (first) => {
    await withHarness(async (second) => {
      expect(second.session.snapshot.conversation).toBeNull();
      await withTurn(first.frame, request(), (turn) => appendReply(turn, "one", "hello"));
      await withTurn(second.frame, request("b"), (turn) => {
        expect(turn.view.items).toEqual([item("one", "hello")]);
      });
    }, first.memory);
  });
});

test("pristine backend adopts snapshot namespace", async () => {
  const exported = await withHarness(async (h) => {
    await withTurn(h.frame, request(), (turn) => appendReply(turn, "one", "hello"));
    return snapshot(h.memory);
  });
  const value: Mutable = structuredClone(exported);
  const session = new SessionMemory().openRun({ snapshot: value });
  try {
    expect(session.snapshot).toEqual(exported);
    value.conversation = null;
    expect(session.snapshot.conversation).not.toBeNull();
  } finally {
    session.close();
  }
});

test("used backend rejects foreign snapshot", () => {
  const memory = new SessionMemory();
  const before = snapshot(memory);
  failure(
    () => memory.openRun({ snapshot: snapshot(new SessionMemory()) }),
    "session.invalid_snapshot",
    "snapshot is not proven history of this backend",
  );
  expect(snapshot(memory)).toEqual(before);
});

test("old same lineage snapshot stays pinned without rewind", async () => {
  await withHarness(async (h) => {
    const old = h.session.snapshot;
    await withTurn(h.frame, request(), (turn) => appendReply(turn, "one", "hello"));
    const latest = snapshot(h.memory);
    const replay = h.memory.openRun({ snapshot: old });
    try {
      expect(replay.snapshot).toEqual(old);
      expect(snapshot(h.memory)).toEqual(latest);
    } finally {
      replay.close();
    }
  });
});

test("session recall contributes transcript without notes", async () => {
  await withHarness((h) => {
    for (const content of [
      h.frame.recall(),
      h.frame.recall(new RecallContext("system", "prompt", {}, ["root", "agent"])),
    ]) {
      expect(Object.getPrototypeOf(content)).toBe(MemoryContent.prototype);
      expect(content.text).toBe("");
      expect(content.transcript).toBe(h.frame);
    }
    expectThrow(() => h.frame.recall({} as never), TypeError, "context must be a RecallContext or null");
  });
});

test("session learn and commit accept empty batches", async () => {
  await withHarness((h) => {
    const before = snapshot(h.memory);
    expect(h.frame.learn(new LearnContext("a", "session", []))).toEqual([]);
    h.session.commit("same", []);
    h.session.commit("same", []);
    failure(
      () => h.session.commit("same", [new MemoryMutation("m", {})]),
      "session.invalid_batch",
      "mutation batch is invalid",
    );
    expectThrow(() => h.session.commit("same", new Set() as never), TypeError, "mutations must be an array");
    expect(snapshot(h.memory)).toEqual(before);
  });
});

test("idle reset clears the conversation", async () => {
  await withHarness(async (h) => {
    for (const name of ["agent", "other"]) {
      await withTurn(h.frame, request(name, 1, ["root", name]), () => {});
    }
    const before: Mutable = snapshot(h.memory);
    h.memory.reset();
    expect(snapshot(h.memory)).toEqual({ ...before, generation: before.generation + 1n, conversation: null });
  });
});

describe("snapshot rejects invalid container or nonfinite json", () => {
  const rows: [string, (x: Mutable) => Mutable][] = [
    ["list", () => []],
    ["version_bool", (x) => ({ ...x, version: true })],
    ["version_unknown", (x) => ({ ...x, version: 2n })],
    ["generation_bool", (x) => ({ ...x, generation: false })],
    ["generation_negative", (x) => ({ ...x, generation: -1n })],
    ["namespace_empty", (x) => ({ ...x, namespace: "" })],
    ["namespace_surrogate", (x) => ({ ...x, namespace: "\ud800" })],
    ["extra", (x) => ({ ...x, extra: null })],
    ["missing", ({ version: _, ...rest }) => rest],
    ["tuple", (x) => ({ ...x, conversation: new Map(Object.entries(x.conversation)) })],
    ["nan", (x) => ({ ...x, version: Number.NaN })],
    ["infinity", (x) => ({ ...x, version: Number.POSITIVE_INFINITY })],
    [
      "cycle",
      (x) => {
        x.conversation.items.push(x);
        return x;
      },
    ],
    // Fails as a non-view wire: TS has no memory nesting limit.
    ["deep", (x) => ({ ...x, conversation: nested(130) })],
    ["duplicate", (x) => ({ ...x, conversation: [x.conversation, x.conversation] })],
    [
      "foreign_namespace",
      (x) => {
        x.conversation.namespace = "foreign";
        return x;
      },
    ],
    [
      "foreign_generation",
      (x) => {
        x.conversation.generation += 1n;
        return x;
      },
    ],
    [
      "bad_digest",
      (x) => {
        x.conversation.applied[0].after_digest = "0".repeat(64);
        return x;
      },
    ],
    [
      "counter_bool",
      (x) => {
        x.conversation.revision = true;
        return x;
      },
    ],
    ["bad_container", (x) => ({ ...x, conversation: [] })],
    // TS-only: a generation beyond 2^53 − 1 that a double holds exactly.
    ["generation 2**53", (x) => ({ ...x, generation: 2n ** 53n, conversation: null })],
  ];
  for (const [name, mutate] of rows) {
    test(name, async () => {
      const exported = await withHarness(async (source) => {
        await withTurn(source.frame, request(), () => {});
        return snapshot(source.memory);
      });
      const value = mutate(structuredClone(exported));
      const memory = new SessionMemory();
      const error = failure(
        () => memory.openRun({ snapshot: value }),
        "session.invalid_snapshot",
        "invalid SessionMemory snapshot",
      );
      expect([error.stage, error.retryable]).toEqual(["snapshot", false]);
      expect(memory["used"]).toBe(false);
      expect(memory["view"]).toBeNull();
    });
  }
});

test("two call sites share the instance conversation", async () => {
  await withHarness(async (h) => {
    for (const [run, site] of [
      ["a", "call_a"],
      ["b", "call_b"],
      ["c", "call_a"],
    ] as const) {
      await withTurn(h.frame, request(run, 1, ["root", site, "agent"]), (turn) => {
        expect(turn.view.items.length).toBe("abc".indexOf(run));
        appendReply(turn, run, run);
      });
    }
  });
});

test("used backend rejects newer same namespace snapshot", async () => {
  const [initial, newer] = await withHarness(async (source) => {
    const pinned = source.session.snapshot;
    await withTurn(source.frame, request(), () => {});
    return [pinned, snapshot(source.memory)];
  });
  const restored = new SessionMemory();
  restored.openRun({ snapshot: initial }).close();
  failure(
    () => restored.openRun({ snapshot: newer }),
    "session.invalid_snapshot",
    "snapshot is not proven history of this backend",
  );
  expect(snapshot(restored)).toEqual(initial);
});

test("invalid request wire publishes nothing and keeps lease usable", async () => {
  await withHarness(async (h) => {
    await withTurn(h.frame, request(), (turn) => {
      const before = snapshot(h.memory);
      const rejected = caught(() => turn.request("model", { value: Number.NaN }));
      expect(rejected).toBeInstanceOf(RangeError);
      expect((rejected as Error).message).toBe("payload.value JSON number must not be NaN");
      expect(Object.hasOwn(rejected as object, "notes")).toBe(false);
      expect(snapshot(h.memory)).toEqual(before);
      // No memory nesting limit: a deep payload publishes, and the lease stays usable.
      const ordinal = turn.request("model", nested(10_000));
      const revision = (before as Mutable).conversation.revision;
      expect((snapshot(h.memory) as Mutable).conversation.revision).toBe(revision + 1n);
      turn.observe(ordinal, "completed");
      appendReply(turn, "valid", "still usable");
    });
    expect(liveView(h).effects.map((effect) => effect.status)).toEqual(["completed", "completed"]);
  });
});

test("old opening snapshot reads latest history without rewind", async () => {
  for (const nonempty of [false, true]) {
    await withHarness(async (source) => {
      if (nonempty) await withTurn(source.frame, request("first"), (turn) => appendReply(turn, "first", "first"));
      const old = snapshot(source.memory);
      await withTurn(source.frame, request("second"), (turn) => appendReply(turn, "second", "second"));
      const latest = liveView(source);
      const opened = source.memory.openRun({ snapshot: old });
      try {
        expect(opened.snapshot).toEqual(old);
        await withTurn(opened, request("third"), (turn) => {
          expect(turn.view.items).toEqual(latest.items);
          expect(turn.view.revision).toBe(latest.revision + 1);
          appendReply(turn, "third", "third");
        });
        expect(opened.snapshot).toEqual(old);
      } finally {
        opened.close();
      }
    });
  }
});

describe("malformed request never publishes partial evidence", () => {
  const rows: [string, Mutable, Mutable, Mutable, ErrorConstructor, string][] = [
    ["kind", "unknown", {}, [], RangeError, "unsupported conversation effect kind"],
    ["payload_list", "model", [], [], TypeError, "payload must be an exact JSON object"],
    ["nan", "model", { value: Number.NaN }, [], RangeError, "payload.value JSON number must not be NaN"],
    ["items_set", "model", {}, new Set([item("x", "x")]), TypeError, "items must be an array"],
    [
      "duplicate_items",
      "model",
      {},
      [item("x", "x"), item("x", "y")],
      RangeError,
      "conversation item IDs must be unique",
    ],
  ];
  for (const [name, kind, payload, items, cls, message] of rows) {
    test(name, async () => {
      await withHarness(async (h) => {
        await withTurn(h.frame, request(), (turn) => {
          const before = snapshot(h.memory);
          expectThrow(() => turn.request(kind, payload, { items }), cls, message);
          expect(snapshot(h.memory)).toEqual(before);
          appendReply(turn, "valid", "retained");
        });
      });
    });
  }
});

describe("malformed observation leaves pending evidence unchanged", () => {
  const rows: [string, Mutable, Mutable, Mutable, ErrorConstructor, string][] = [
    ["ordinal", true, "completed", [], TypeError, "effect_ordinal must be an integer"],
    ["status", null, "unknown", [], RangeError, "invalid conversation effect status"],
    ["failure", null, "failed", [], RangeError, "failed effect requires a failure"],
    ["items", null, "completed", new Set([item("partial", "no")]), TypeError, "items must be an array"],
  ];
  for (const [name, ordinalValue, status, items, cls, message] of rows) {
    test(name, async () => {
      await withHarness(async (h) => {
        await withTurn(h.frame, request(), (turn) => {
          const ordinal = turn.request("model", {}, { items: [item("input", "question")] });
          const before = snapshot(h.memory);
          expectThrow(() => turn.observe(ordinalValue ?? ordinal, status, { items }), cls, message);
          expect(snapshot(h.memory)).toEqual(before);
          turn.observe(ordinal, "completed", { items: [item("answer", "actual")] });
        });
      });
    });
  }
});

test("alternate same namespace history is not a valid opening snapshot", async () => {
  await withHarness(async (live) => {
    const initial = live.session.snapshot;
    await withTurn(live.frame, request(), (turn) => appendReply(turn, "reply", "real"));
    const original = snapshot(live.memory);
    const fork = new SessionMemory();
    fork.openRun({ snapshot: initial }).close();
    await withHarness(
      (alternate) => withTurn(alternate.frame, request(), (turn) => appendReply(turn, "reply", "different")),
      fork,
    );
    failure(
      () => live.memory.openRun({ snapshot: snapshot(fork) }),
      "session.invalid_snapshot",
      "snapshot is not proven history of this backend",
    );
    expect(snapshot(live.memory)).toEqual(original);
  });
});

test("turn exposes only message effects without native tracking", async () => {
  expect("ConversationReference" in contracts).toBe(false);
  await withHarness(async (h) => {
    await withTurn(h.frame, request(), (turn) => {
      expect("nativeSession" in turn).toBe(false);
      expect("reference" in turn.view).toBe(false);
      const before = snapshot(h.memory);
      expectThrow(() => turn.request("native_session" as never, {}), RangeError, /unsupported/);
      expect(snapshot(h.memory)).toEqual(before);
    });
  });
});

test("every consumer of an instance shares one conversation", async () => {
  const exported: Mutable = await withHarness(async (h) => {
    for (const [run, consumer] of [
      ["a", ["root", "left"]],
      ["b", ["root", "right"]],
    ] as const) {
      await withTurn(h.frame, request(run, 1, consumer), (turn) => {
        expect(turn.view.items.length).toBe(run === "a" ? 0 : 1);
        appendReply(turn, run, run);
      });
    }
    return snapshot(h.memory);
  });
  expect(Object.keys(exported).toSorted()).toEqual(["conversation", "generation", "namespace", "notes", "version"]);
  expect(exported.version).toBe(3n);
  expect("key" in exported.conversation).toBe(false);
  expect(exported.conversation.items.map((entry: Mutable) => entry.item_id)).toEqual(["a", "b"]);
});

describe("publication requires owned stamp", () => {
  for (const field of ["invocation", "namespace", "generation"] as const) {
    test(field, async () => {
      await withHarness(async (h) => {
        await withTurn(h.frame, request(), (turn) => {
          const stamp: ConversationStamp = turn["stamp"]();
          const replaced = new ConversationStamp(
            field === "invocation" ? request("other").invocation : stamp.invocation,
            field === "namespace" ? "foreign" : stamp.namespace,
            field === "generation" ? 1 : stamp.generation,
            stamp.expectedRevision,
            stamp.ordinal,
          );
          const checkpoint = new ConversationEffectRequested(replaced, "model", {}, []);
          const before = snapshot(h.memory);
          failure(() => turn["publish"](checkpoint), "session.conflict", "checkpoint does not belong to this lease");
          expect(snapshot(h.memory)).toEqual(before);
        });
      });
    });
  }
});

describe("new turn preserves incomplete history", () => {
  for (const restore of [false, true]) {
    for (const closed of [false, true]) {
      test(`restore ${restore} closed ${closed}`, async () => {
        let memory = new SessionMemory();
        const session = memory.openRun();
        let turn: Turn;
        let ordinal: number;
        let prefix: Mutable;
        try {
          turn = await session.openTurn(request("a"), { signal: NEVER });
          ordinal = turn.request("model", {}, { items: [item("a-in", "first")] });
          turn.observe(ordinal, "pending", { items: [item("a-out", "partial")] });
          if (closed) {
            const primary = new Error("interrupted effect");
            expect(turn.close(primary)).toBeUndefined();
          } else {
            // A restored prefix can end before its owner recorded closure.
            session["release"](turn["lease"]!);
          }
          prefix = snapshot(memory);
        } finally {
          session.close();
        }
        if (restore) {
          memory = new SessionMemory();
          const restored = memory.openRun({ snapshot: prefix });
          expect(restored.snapshot).toEqual(prefix);
          restored.close();
        }
        const replay = memory.openRun({ snapshot: prefix });
        try {
          for (const name of ["b", "c"]) {
            await withTurn(replay, request(name), (later) => {
              expect(later.view.effects[0]!.status).toBe("pending");
              expect(later.view.items.slice(0, 2)).toEqual([item("a-in", "first"), item("a-out", "partial")]);
              appendReply(later, name, name);
            });
          }
          expect(replay.snapshot).toEqual(prefix);
        } finally {
          replay.close();
        }
        const final: Mutable = snapshot(memory);
        const original = prefix.conversation;
        expect(final.conversation.applied.slice(0, Number(original.revision))).toEqual(original.applied);
        expect(final.conversation.effects.map((effect: Mutable) => effect.status)).toEqual([
          "pending",
          "completed",
          "completed",
        ]);
        const reopened = new SessionMemory().openRun({ snapshot: final });
        expect(reopened.snapshot).toEqual(final);
        reopened.close();
        failure(() => turn.observe(ordinal, "completed"), "session.stale", STALE_LEASE);
      });
    }
  }
});

describe("incompatible descriptor requires reset", () => {
  const rows: [Mutable, Mutable, boolean][] = [
    [true, 1n, false],
    [1n, 1.0, false],
    // Changed: `-0` reads as `0`, so the descriptors are equal.
    [0.0, -0.0, true],
  ];
  for (const [old, next, compatible] of rows) {
    test(`${String(old)} then ${Object.is(next, -0) ? "-0" : String(next)}`, async () => {
      const memory = new SessionMemory();
      await withHarness(async (h) => {
        await withTurn(h.frame, request("a", 1, ["root", "agent"], { format: old }), () => {});
        const before = snapshot(h.memory);
        const opening = withTurn(h.frame, request("b", 1, ["root", "agent"], { format: next }), () => {});
        if (compatible) {
          await opening;
        } else {
          const error = (await rejected(opening)) as MemoryOperationError;
          expect(error).toBeInstanceOf(MemoryOperationError);
          expect([error.code, error.message]).toEqual([
            "session.incompatible",
            "conversation compatibility changed; reset is required",
          ]);
          expect(snapshot(h.memory)).toEqual(before);
        }
        h.memory.reset();
      }, memory);
      const fresh = request("c", 1, ["root", "agent"], { format: next });
      await withHarness((h) => withTurn(h.frame, fresh, () => {}), memory);
    });
  }
});

test("second same key ticket waits", async () => {
  await withHarness(async (h) => {
    let entered = false;
    let second: ReturnType<typeof settle<void>>;
    await withTurn(h.frame, request(), async () => {
      second = settle(
        withTurn(h.frame, request("b"), () => {
          entered = true;
        }),
      );
      await macrotask();
      expect(h.memory["queue"].length).toBe(1);
      expect(second.settled).toBe(false);
      expect(entered).toBe(false);
    });
    await second!.promise;
    expect(second!.error).toBeUndefined();
    expect(entered).toBe(true);
  });
});

test("third ticket cannot overtake second", async () => {
  await withHarness(async (h) => {
    const order: string[] = [];
    const waiters: ReturnType<typeof settle<void>>[] = [];
    await withTurn(h.frame, request(), async () => {
      for (const [index, name] of ["b", "c"].entries()) {
        waiters.push(settle(withTurn(h.frame, request(name), () => void order.push(name))));
        await macrotask();
        expect(h.memory["queue"].length).toBe(index + 1);
      }
    });
    for (const waiter of waiters) {
      await waiter.promise;
      expect(waiter.error).toBeUndefined();
    }
    expect(order).toEqual(["b", "c"]);
  });
});

test("second consumer waits for the first turn", async () => {
  await withHarness(async (h) => {
    let entered = false;
    let other: ReturnType<typeof settle<void>>;
    await withTurn(h.frame, request(), async () => {
      other = settle(
        withTurn(h.frame, request("b", 1, ["root", "other"]), () => {
          entered = true;
        }),
      );
      await macrotask();
      expect(entered).toBe(false);
    });
    await other!.promise;
    expect(other!.error).toBeUndefined();
    expect(entered).toBe(true);
  });
});

test("cancelled ticket is removed", async () => {
  await withHarness(async (h) => {
    const controller = new AbortController();
    const reason = new Error("stop waiting");
    await withTurn(h.frame, request(), async () => {
      const second = settle(
        withTurn(
          h.frame,
          request("b"),
          () => {
            throw new Error("cancelled waiter entered");
          },
          controller.signal,
        ),
      );
      await macrotask();
      expect(h.memory["queue"].length).toBe(1);
      controller.abort(reason);
      await second.promise.catch(() => {});
      expect(second.error).toBe(reason);
      expect(h.memory["queue"].length).toBe(0);
    });
  });
});

test("reset rejects active backend without mutation", async () => {
  await withHarness((h) =>
    withTurn(h.frame, request(), () => {
      const before = snapshot(h.memory);
      const error = failure(() => h.memory.reset(), "session.busy", RESET_BUSY);
      expect([error.stage, error.retryable]).toEqual(["reset", true]);
      expect(snapshot(h.memory)).toEqual(before);
    }),
  );
});

test("reset rejects queued backend without mutation", async () => {
  await withHarness(async (h) => {
    await withHarness(async (waitingSession) => {
      await withTurn(h.frame, request(), async () => {
        const worker = settle(
          withTurn(waitingSession.frame, request("b"), () => {
            throw new Error("closed waiter entered");
          }),
        );
        await macrotask();
        expect(h.memory["queue"].length).toBe(1);
        const before = snapshot(h.memory);
        try {
          failure(() => h.memory.reset(), "session.busy", RESET_BUSY);
          expect(snapshot(h.memory)).toEqual(before);
        } finally {
          waitingSession.session.close();
          await worker.promise.catch(() => {});
        }
        expect(worker.error).toBeInstanceOf(MemoryOperationError);
        expect((worker.error as MemoryOperationError).code).toBe("session.stale");
        expect((worker.error as MemoryOperationError).message).toBe(STALE_SESSION);
      });
    }, h.memory);
  });
});

test("reset invalidates old session and turn", async () => {
  await withHarness(async (h) => {
    let turn: Turn;
    await withTurn(h.frame, request(), (opened) => {
      turn = opened;
    });
    h.memory.reset();
    const calls: [() => unknown, string][] = [
      [() => h.session.snapshot, STALE_SESSION],
      [() => h.frame.recall(), STALE_SESSION],
      [() => h.frame.learn(new LearnContext("a", "session", [])), STALE_SESSION],
      [() => h.session.commit("id", []), STALE_SESSION],
      [() => turn.request("model", {}), STALE_SESSION],
      [() => turn.observe(1, "completed"), STALE_LEASE],
    ];
    for (const [call, message] of calls) failure(call, "session.stale", message);
    const error = (await rejected(h.frame.openTurn(request("b"), { signal: NEVER }))) as MemoryOperationError;
    expect([error.code, error.message]).toEqual(["session.stale", STALE_SESSION]);
    expect(snapshot(h.memory).conversation).toBeNull();
  });
});

describe("cleanup annotation cannot replace primary error", () => {
  for (const hostile of [null, "message", "frozen"]) {
    test(String(hostile), async () => {
      class Primary extends Error {}
      class Cleanup extends Error {}
      const primary = new Primary("primary");
      if (hostile === "frozen") Object.freeze(primary);
      const cleanup = new Cleanup();
      Object.defineProperty(cleanup, "message", {
        get() {
          if (hostile === "message") throw new Error("string formatting failed");
          return "cleanup";
        },
      });
      await withHarness(async (h) => {
        const reject = spyOn(h.session as Mutable, "candidateSnapshot");
        try {
          const error = await rejected(
            withTurn(h.frame, request(), () => {
              reject.mockImplementation(() => {
                throw cleanup;
              });
              throw primary;
            }),
          );
          expect(error).toBe(primary);
        } finally {
          reject.mockRestore();
        }
        if (hostile === null) {
          expect((primary as Mutable).notes).toEqual(["Conversation cleanup also failed: Cleanup: cleanup"]);
        } else {
          expect(Object.hasOwn(primary, "notes")).toBe(false);
        }
        expect(h.memory["active"]).toBeNull();
      });
    });
  }
});

test("completed observation is retained before checker stop", async () => {
  const reason = new Error("ordinary Exception used as host stop");
  const controller = new AbortController();
  await withHarness(async (h) => {
    const error = await rejected(
      withTurn(
        h.frame,
        request(),
        (turn) => {
          const ordinal = turn.request("tool", {});
          controller.abort(reason);
          turn.observe(ordinal, "completed", { items: [item("result", "already happened")] });
        },
        controller.signal,
      ),
    );
    expect(error).toBe(reason);
    const last = liveView(h);
    expect(last.effects[0]!.status).toBe("completed");
    expect(last.items).toEqual([item("result", "already happened")]);
    expect(last.closedInvocations.at(-1)!.status).toBe("interrupted");
  });
});

test("stop after publication preserves requested effect", async () => {
  const reason = new Error("stop before external call");
  const controller = new AbortController();
  await withHarness(async (h) => {
    const original = h.session["candidateSnapshot"].bind(h.session);
    const spy = spyOn(h.session as Mutable, "candidateSnapshot").mockImplementation((view: Mutable) => {
      original(view);
      if (view.effects.length > 0) controller.abort(reason);
    });
    try {
      const error = await rejected(
        withTurn(
          h.frame,
          request(),
          (turn) => {
            turn.request("tool", {});
            throw new Error("stopped request granted effect permission");
          },
          controller.signal,
        ),
      );
      expect(error).toBe(reason);
    } finally {
      spy.mockRestore();
    }
    expect(liveView(h).effects[0]!.status).toBe("pending");
    expect(liveView(h).closedInvocations.at(-1)!.status).toBe("interrupted");
  });
});

describe("body failure closes truthfully without inventing observations", () => {
  // Changed: Python's KeyboardInterrupt row is a thrown string, which records `failed`.
  for (const primary of [new Error("failed"), "interrupted"]) {
    test(String(primary), async () => {
      await withHarness(async (h) => {
        const error = await rejected(
          withTurn(h.frame, request(), (turn) => {
            turn.request("tool", {});
            throw primary;
          }),
        );
        expect(error).toBe(primary);
        const view = liveView(h);
        expect(view.effects[0]!.status).toBe("pending");
        expect(view.effects[0]!.observationOrdinals).toEqual([]);
        expect(view.closedInvocations.at(-1)!.status).toBe("failed");
        await withTurn(h.frame, request("b"), (turn) => {
          expect(turn.view.effects).toEqual(view.effects);
        });
      });
    });
  }
});

test("completed close with pending effect is rejected without fabrication", async () => {
  await withHarness(async (h) => {
    const error = (await rejected(
      withTurn(h.frame, request(), (turn) => {
        turn.request("model", {});
      }),
    )) as MemoryOperationError;
    expect(error).toBeInstanceOf(MemoryOperationError);
    expect([error.code, error.message]).toEqual([
      "session.conflict",
      "successful closure cannot leave pending effects",
    ]);
    expect(liveView(h).revision).toBe(2);
    expect(liveView(h).effects[0]!.status).toBe("pending");
    expect(liveView(h).closedInvocations).toEqual([]);
    expect(h.memory["active"]).toBeNull();
    await withTurn(h.frame, request("b"), () => {});
  });
});

test("active invocation cannot be shadowed by another key", async () => {
  await withHarness((h) =>
    withTurn(h.frame, request(), async () => {
      const error = (await rejected(
        withTurn(h.frame, request("a", 1, ["root", "other"]), () => {}),
      )) as MemoryOperationError;
      expect([error.code, error.message]).toEqual(["session.conflict", "invocation already has a session lease"]);
      expect(h.memory["active"]).not.toBeNull();
    }),
  );
});

test("session close retains active owner until finalization", async () => {
  await withHarness(async (one) => {
    await withHarness(async (two) => {
      await withTurn(one.frame, request(), (turn) => {
        const ordinal = turn.request("model", {});
        one.session.close();
        one.session.close();
        failure(() => turn.request("model", {}), "session.stale", STALE_SESSION);
        failure(() => one.memory.reset(), "session.busy", RESET_BUSY);
        turn.observe(ordinal, "completed", { items: [item("actual", "retained")] });
      });
      expect(liveView(one).effects[0]!.status).toBe("completed");
      expect(liveView(one).items).toEqual([item("actual", "retained")]);
      expect(liveView(one).closedInvocations.at(-1)!.status).toBe("interrupted");
      await withTurn(two.frame, request("b", 1, ["root", "other"]), () => {});
      expect(one.memory["active"]).toBeNull();
    }, one.memory);
  });
});

test("session close removes own waiters without poisoning other owner", async () => {
  await withHarness(async (one) => {
    await withHarness(async (two) => {
      await withTurn(one.frame, request(), async () => {
        const worker = settle(
          withTurn(two.frame, request("b"), () => {
            throw new Error("closed session was admitted");
          }),
        );
        await macrotask();
        expect(one.memory["queue"].length).toBe(1);
        two.session.close();
        await worker.promise.catch(() => {});
        expect(worker.error).toBeInstanceOf(MemoryOperationError);
        expect((worker.error as MemoryOperationError).code).toBe("session.stale");
        expect((worker.error as MemoryOperationError).message).toBe(STALE_SESSION);
        expect(one.memory["queue"].length).toBe(0);
        expect(one.memory["active"]).not.toBeNull();
      });
    }, one.memory);
  });
});

test("stop and external call do not hold backend lock", async () => {
  await withHarness(async (h) => {
    const reads: Mutable[] = [];
    const original = h.session["candidateSnapshot"].bind(h.session);
    const spy = spyOn(h.session as Mutable, "candidateSnapshot").mockImplementation((view: Mutable) => {
      original(view);
      reads.push(snapshot(h.memory));
    });
    try {
      await withTurn(h.frame, request(), async (turn) => {
        const ordinal = turn.request("model", {});
        await macrotask();
        expect((snapshot(h.memory) as Mutable).conversation.revision).toBe(2n);
        turn.observe(ordinal, "completed");
      });
    } finally {
      spy.mockRestore();
    }
    expect(reads.length).toBe(4);
  });
});

test("reset rejects queued ticket even without an active lease", async () => {
  await withHarness(async (h) => {
    const first = await h.frame.openTurn(request(), { signal: NEVER });
    const waiting = settle(withTurn(h.frame, request("b"), () => {}));
    await macrotask();
    expect(h.memory["queue"].length).toBe(1);
    // Before any await, the released lease has woken nobody yet: B is queued, nothing is active.
    first.close();
    expect(h.memory["active"]).toBeNull();
    expect(h.memory["queue"].length).toBe(1);
    const before = snapshot(h.memory);
    failure(() => h.memory.reset(), "session.busy", RESET_BUSY);
    expect(snapshot(h.memory)).toEqual(before);
    await waiting.promise;
    expect(waiting.error).toBeUndefined();
  });
});

test("snapshot envelope adds no nesting beyond the view", async () => {
  await withHarness(async (h) => {
    await withTurn(h.frame, request(), (turn) => {
      const ordinal = turn.request("model", nested(200));
      let value: Mutable = (snapshot(h.memory) as Mutable).conversation.effects[0].request;
      let depth = 0;
      while (value.nested !== undefined) {
        value = value.nested;
        depth += 1;
      }
      expect(depth).toBe(200);
      expect(turn["lease"]!.active).toBe(true);
      turn.observe(ordinal, "completed");
    });
  });
});

test("stop checker cannot close session then grant effect permission", async () => {
  await withHarness(async (h) => {
    const original = h.session["candidateSnapshot"].bind(h.session);
    const spy = spyOn(h.session as Mutable, "candidateSnapshot").mockImplementation((view: Mutable) => {
      original(view);
      if (view.effects.length > 0) h.session.close();
    });
    try {
      const error = (await rejected(
        withTurn(h.frame, request(), (turn) => {
          turn.request("tool", {});
          throw new Error("closed session granted effect permission");
        }),
      )) as MemoryOperationError;
      expect([error.code, error.message]).toEqual(["session.stale", STALE_SESSION]);
    } finally {
      spy.mockRestore();
    }
    expect(liveView(h).effects[0]!.status).toBe("pending");
    expect(liveView(h).closedInvocations.at(-1)!.status).toBe("interrupted");
    expect(h.memory["active"]).toBeNull();
  });
});

test("observed stop cannot be caught to reenable new effects", async () => {
  await withHarness(async (h) => {
    const controller = new AbortController();
    const reason = new Error("stop once");
    const error = await rejected(
      withTurn(
        h.frame,
        request(),
        (turn) => {
          controller.abort(reason);
          expect(caught(() => turn.request("tool", {}))).toBe(reason);
          turn.request("tool", {});
          throw new Error("observed stop was forgotten");
        },
        controller.signal,
      ),
    );
    expect(error).toBe(reason);
    expect(liveView(h).effects).toEqual([]);
  });
});

test("session close at reserved admission boundary is typed stale", async () => {
  await withHarness(async (h) => {
    const original = h.session["reserve"].bind(h.session);
    const spy = spyOn(h.session as Mutable, "reserve").mockImplementation(
      async (conversationRequest: Mutable, signal: Mutable) => {
        const lease = await original(conversationRequest, signal);
        expect(lease.before).not.toBeNull();
        h.session.close();
        expect(lease.before).not.toBeNull();
        expect(lease.active).toBe(true);
        return lease;
      },
    );
    let error: Mutable;
    try {
      error = await rejected(
        withTurn(h.frame, request(), () => {
          throw new Error("a closed session admitted its reserved turn");
        }),
      );
    } finally {
      spy.mockRestore();
    }
    expect(Object.getPrototypeOf(error)).toBe(MemoryOperationError.prototype);
    expect(Object.hasOwn(error, "notes")).toBe(false);
    expect([error.stage, error.code, error.retryable, error.message]).toEqual([
      "conversation",
      "session.stale",
      false,
      STALE_SESSION,
    ]);
    expect(h.memory["view"]).toBeNull();
    expect(h.memory["active"]).toBeNull();
    expect(h.memory["queue"].length).toBe(0);
  });
});

test("session close never overlaps an inflight effect with the next turn", async () => {
  await withHarness(async (owner) => {
    await withHarness(async (nextRun) => {
      const inEffect = Promise.withResolvers<void>();
      const finishEffect = Promise.withResolvers<void>();
      const order: string[] = [];
      const active = settle(
        withTurn(owner.frame, request("a"), async (turn) => {
          const ordinal = turn.request("model", {});
          order.push("effect-start");
          inEffect.resolve();
          await finishEffect.promise;
          order.push("effect-end");
          turn.observe(ordinal, "completed", { items: [item("result", "actual")] });
        }),
      );
      await inEffect.promise;
      const waiting = settle(
        withTurn(nextRun.frame, request("b"), (turn) => {
          order.push("next-enter");
          expect(turn.view.items).toEqual([item("result", "actual")]);
          expect(turn.view.closedInvocations.at(-1)!.status).toBe("interrupted");
        }),
      );
      try {
        await macrotask();
        expect(owner.memory["queue"].length).toBe(1);
        owner.session.close();
        await macrotask();
        expect(order).not.toContain("next-enter");
        expect(owner.memory["active"]).not.toBeNull();
        failure(() => owner.memory.reset(), "session.busy", RESET_BUSY);
        expect(order).not.toContain("next-enter");
      } finally {
        finishEffect.resolve();
        await active.promise.catch(() => {});
        await waiting.promise.catch(() => {});
      }
      expect(active.error).toBeUndefined();
      expect(waiting.error).toBeUndefined();
      expect(order).toEqual(["effect-start", "effect-end", "next-enter"]);
    }, owner.memory);
  });
});

describe("candidate snapshot failure publishes nothing", () => {
  for (const stage of ["admission", "request", "observation", "close"] as const) {
    test(stage, async () => {
      const reason = new Error("snapshot unavailable");
      await withHarness(async (h) => {
        let turn: Turn | null = null;
        let ordinal = 0;
        if (stage !== "admission") turn = await h.frame.openTurn(request(), { signal: NEVER });
        if (stage === "observation" || stage === "close") ordinal = turn!.request("model", {});
        if (stage === "close") turn!.observe(ordinal, "completed");
        const before = snapshot(h.memory);
        const spy = spyOn(h.session as Mutable, "candidateSnapshot").mockImplementation(() => {
          throw reason;
        });
        let error: unknown;
        try {
          if (stage === "admission") error = await rejected(h.frame.openTurn(request(), { signal: NEVER }));
          if (stage === "request") error = caught(() => turn!.request("model", {}, { items: [item("new", "no")] }));
          if (stage === "observation") {
            error = caught(() => turn!.observe(ordinal, "completed", { items: [item("new", "no")] }));
          }
          if (stage === "close") error = caught(() => turn!.close());
        } finally {
          spy.mockRestore();
        }
        expect(error).toBe(reason);
        expect(Object.hasOwn(reason, "notes")).toBe(false);
        expect(snapshot(h.memory)).toEqual(before);
        if (stage === "observation") turn!.observe(ordinal, "completed");
        if (stage === "request" || stage === "observation") turn!.close();
        await withTurn(h.frame, request("next"), () => {});
      });
    });
  }
});

describe("an invocation identity cannot be reused", () => {
  for (const newSession of [false, true]) {
    test(`new session ${newSession}`, async () => {
      await withHarness(async (h) => {
        await withTurn(h.frame, request(), () => {});
        const before = snapshot(h.memory);
        await withHarness(async (other) => {
          const frame = newSession ? other.frame : h.frame;
          const error = (await rejected(
            withTurn(frame, request(), () => {
              throw new Error("reused invocation entered");
            }),
          )) as MemoryOperationError;
          expect(error.code).toBe("session.conflict");
          expect(error.message).toBe(
            newSession ? "checkpoint operation has already been applied" : "invocation already has a session lease",
          );
        }, h.memory);
        expect(snapshot(h.memory)).toEqual(before);
      });
    });
  }
});

describe("TS-only", () => {
  test("openTurn with an aborted signal rejects with its reason and leaves no ticket", async () => {
    await withHarness(async (h) => {
      const controller = new AbortController();
      const reason = new Error("stopped");
      controller.abort(reason);
      expect(await rejected(h.frame.openTurn(request(), { signal: controller.signal }))).toBe(reason);
      expect(h.memory["queue"].length).toBe(0);
      expect(h.memory["active"]).toBeNull();
      await withTurn(h.frame, request("b"), async () => {
        const queued = new AbortController();
        const why = new Error("stop waiting");
        const waiting = settle(h.frame.openTurn(request("c"), { signal: queued.signal }));
        await macrotask();
        expect(h.memory["queue"].length).toBe(1);
        queued.abort(why);
        await waiting.promise.catch(() => {});
        expect(waiting.error).toBe(why);
        expect(h.memory["queue"].length).toBe(0);
      });
      expect(h.memory["active"]).toBeNull();
    });
  });

  test("opening an active invocation again on its session is a conflict at once", async () => {
    await withHarness((h) =>
      withTurn(h.frame, request(), async () => {
        const error = (await rejected(h.frame.openTurn(request(), { signal: NEVER }))) as MemoryOperationError;
        expect([error.code, error.message]).toEqual(["session.conflict", "invocation already has a session lease"]);
        expect(h.memory["queue"].length).toBe(0);
      }),
    );
  });

  test("close with an error never throws it and notes a stop seen meanwhile", async () => {
    await withHarness(async (h) => {
      const controller = new AbortController();
      const turn = await h.frame.openTurn(request(), { signal: controller.signal });
      turn.request("model", {});
      const reason = new Error("stop");
      controller.abort(reason);
      const body: Mutable = new Error("body");
      expect(turn.close(body)).toBeUndefined();
      expect(body.notes).toEqual(["Conversation cleanup also failed: Error: stop"]);
      expect(liveView(h).closedInvocations.at(-1)!.status).toBe("interrupted");
      expect(h.memory["active"]).toBeNull();
      expect(turn.close()).toBeUndefined();
    });
  });

  test("close after an abort records interrupted, releases, then throws the reason", async () => {
    await withHarness(async (h) => {
      const controller = new AbortController();
      const turn = await h.frame.openTurn(request(), { signal: controller.signal });
      appendReply(turn, "one", "hello");
      const reason = new Error("stop");
      controller.abort(reason);
      expect(caught(() => turn.close())).toBe(reason);
      expect(liveView(h).closedInvocations.at(-1)!.status).toBe("interrupted");
      expect(h.memory["active"]).toBeNull();
      expect(turn.close()).toBeUndefined();
    });
  });

  test("interrupt closes a normal return as interrupted", async () => {
    await withHarness(async (h) => {
      await withTurn(h.frame, request(), (turn) => {
        appendReply(turn, "one", "hello");
        turn.interrupt();
      });
      expect(liveView(h).closedInvocations.at(-1)!.status).toBe("interrupted");
    });
  });

  test("openTurn rejects a non-request or a non-signal before queuing", async () => {
    await withHarness(async (h) => {
      const rows: [unknown, unknown, string][] = [
        [{ ...request() }, { signal: NEVER }, "request must be a ConversationRequest"],
        [request(), undefined, "signal must be an AbortSignal"],
        [request(), {}, "signal must be an AbortSignal"],
        [request(), { signal: { aborted: false } }, "signal must be an AbortSignal"],
      ];
      for (const [value, options, message] of rows) {
        const error = await h.frame.openTurn(value as never, options as never).then(
          () => null,
          (rejected: unknown) => rejected,
        );
        expect(Object.getPrototypeOf(error)).toBe(TypeError.prototype);
        expect((error as Error).message).toBe(message);
        expect(h.memory["queue"].length).toBe(0);
        expect(h.memory["active"]).toBeNull();
        expect(h.frame["leases"].size).toBe(0);
      }
      await withTurn(h.frame, request(), () => {});
    });
  });

  test("constructor settings are exact integers in range", () => {
    for (const value of [true, 1.5, "1"]) {
      expectThrow(
        () => new SessionMemory({ historyBudgetChars: value as never }),
        TypeError,
        "history_budget_chars must be an exact integer",
      );
      expectThrow(
        () => new SessionMemory({ recentTurns: value as never }),
        TypeError,
        "recent_turns must be an exact integer",
      );
    }
    expectThrow(
      () => new SessionMemory({ historyBudgetChars: 0 }),
      RangeError,
      "history_budget_chars must be at least 1",
    );
    expectThrow(() => new SessionMemory({ recentTurns: -1 }), RangeError, "recent_turns must be at least 0");
    expectThrow(
      () => new SessionMemory({ recentTurns: 2n ** 53n }),
      RangeError,
      "recent_turns must be at most 9007199254740991, got 9007199254740992",
    );
    expectThrow(() => new SessionMemory({ createParent: "yes" as never }), TypeError, "create_parent must be a bool");
    const memory = new SessionMemory({ historyBudgetChars: 5n, recentTurns: 0, createParent: null });
    expect(memory["historyBudgetChars"]).toBe(5);
    expect(memory["recentTurns"]).toBe(0);
    const defaults = new SessionMemory({ historyBudgetChars: null, recentTurns: undefined });
    expect(defaults["historyBudgetChars"]).toBe(current().sessionHistoryBudgetChars);
    expect(defaults["recentTurns"]).toBe(current().sessionRecentTurns);
  });
});
