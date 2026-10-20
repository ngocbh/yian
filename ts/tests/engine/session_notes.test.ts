/**
 * Learned self-feedback remains separate from automatic session history: a port of
 * `tests/engine/test_session_notes.py`.
 *
 * Ported: `test_snapshot_adopts_detached_notes`,
 * `test_snapshot_rejects_invalid_or_unadmitted_notes_without_adoption`,
 * `test_historical_note_prefix_is_pinned_without_rewinding_live_state`,
 * `test_learn_is_pure_deterministic_self_feedback_projection`,
 * `test_learn_ignores_feedback_without_owning_admitted_leaf`,
 * `test_commit_publishes_notes_only_to_future_run_recall`,
 * `test_exact_commit_replay_and_stable_note_ids_are_idempotent`,
 * `test_invalid_last_mutation_publishes_no_notes_or_commit_tombstone`,
 * `test_empty_commit_identity_cannot_later_publish_a_different_batch`,
 * `test_reset_clears_notes_and_keeps_commit_replay_tombstones`,
 * `test_restored_notes_recall_and_accept_idempotent_note_republication`,
 * `test_malformed_attributed_self_feedback_fails_without_publication`,
 * `test_repeat_source_feedback_in_one_context_proposes_one_note`,
 * `test_candidate_failure_keeps_notes_and_commit_identity_unpublished`,
 * `test_exact_replay_from_another_run_handle_is_inert`,
 * `test_busy_reset_preserves_notes_and_forward_turn_updates_preserve_notes`,
 * `test_reset_keeps_empty_commit_tombstones`, `test_notes_belong_to_the_instance_and_reach_every_consumer`
 * and `test_session_is_the_transcript_source_and_learns_without_frames`.
 *
 * TS-only: none.
 *
 * Changed:
 * - `test_candidate_failure_keeps_notes_and_commit_identity_unpublished`: ESM bindings can't be
 *   patched, so a spy on the memory's private `saveState` throws once, where Python patches the
 *   module's `_snapshot`; the commit fails with that error unchanged (Python's patched `_snapshot`
 *   ran inside the validation `try`, so it got `session.invalid_batch`), nothing is published, and
 *   the retried commit succeeds.
 * - Note and feedback JSON ints are `bigint`s, so the `bool_sequence`, `bool_index` and
 *   `execution_seq`/`message_index` `true` rows fail on the boolean, not on the int type; mutable
 *   copies are `structuredClone`s; batches are arrays; `with` blocks are `withHarness`/`withTurn`;
 *   `hasattr` checks are `in` checks on the camelCase names.
 *
 * Not ported: none.
 */

import crypto from "node:crypto";

import { describe, expect, spyOn, test } from "bun:test";

import {
  LearnContext,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  RecallContext,
} from "../../src/memory/contracts.ts";
import { type Session, SessionMemory } from "../../src/memory/session.ts";
import { logJson } from "../../src/typesys/json.ts";
import { caught, failure, type Mutable, request, snapshot, withHarness, withTurn } from "./session_fakes.ts";

/** One closed note with its source identity. */
function note(
  options: { consumer?: string[]; runId?: string; sequence?: unknown; index?: unknown; text?: unknown } = {},
): Mutable {
  const { consumer = ["root", "agent"], runId = "a", sequence = 1n, index = 0n, text = "Check units." } = options;
  const value = {
    consumer_static_path: [...consumer],
    run_id: runId,
    execution_seq: sequence,
    message_index: index,
    text,
  };
  return { note_id: crypto.createHash("sha256").update(logJson(value)).digest("hex"), ...value };
}

/** A snapshot of one admitted, closed turn, holding `notes`. */
async function withNotes(...notes: Mutable[]): Promise<Mutable> {
  const value = await withHarness(async (h) => {
    await withTurn(h.frame, request(), () => {});
    return snapshot(h.memory);
  });
  return { ...structuredClone(value), notes: [...notes] };
}

/** Open one closed turn on `h`'s session. */
async function admit(h: { frame: Session }): Promise<void> {
  await withTurn(h.frame, request(), () => {});
}

/** A learning context with one episode and self-feedback items. */
function context(
  options: {
    consumer?: string[];
    runId?: string;
    sequence?: number | null;
    texts?: string[];
    channel?: string;
    feedbackSequence?: bigint | null;
  } = {},
): LearnContext {
  const {
    consumer = ["root", "agent"],
    runId = "a",
    sequence = 1,
    texts = ["Check units."],
    channel = "self",
    feedbackSequence = 1n,
  } = options;
  const episode = new MemoryEpisode(runId, [], consumer, MemoryEpisodeStatus.SUCCEEDED, {}, { executionSeq: sequence });
  const feedback = texts.map((text, index) => ({
    execution_seq: feedbackSequence,
    owner_static_path: consumer.slice(1),
    occurrence_path: [],
    execution_scope: { static_path: consumer.slice(1), occurrence_path: [] },
    channel,
    target_name: null,
    value: text,
    message_index: BigInt(index),
    contribution_context: {},
  }));
  return new LearnContext(runId, "session", [episode], feedback);
}

/** The append-note mutation for `value` (the default note when unset). */
function mutation(value: Mutable = note()): MemoryMutation {
  return new MemoryMutation(value.note_id, { operation: "append_note", note: value });
}

/** The notes text `frame` recalls for `consumer`. */
function recall(frame: Session, consumer: string[] = ["root", "agent"]): string {
  return frame.recall(new RecallContext("system", "prompt", {}, consumer)).text;
}

describe("snapshot adopts detached notes", () => {
  for (const text of ["Check units.", ""]) {
    test(JSON.stringify(text), async () => {
      const value = await withNotes(note({ text }));
      const expected = structuredClone(value);
      const memory = new SessionMemory();
      const opened = memory.openRun({ snapshot: value });
      try {
        value.notes[0].text = "changed outside";
        expect(opened.snapshot).toEqual(expected);
        expect(snapshot(memory)).toEqual(expected);
      } finally {
        opened.close();
      }
    });
  }
});

describe("snapshot rejects invalid or unadmitted notes without adoption", () => {
  const rows: [string, (value: Mutable, first: Mutable) => void][] = [
    ["duplicate", (value) => (value.notes = [...value.notes, ...value.notes])],
    ["extra", (_, first) => (first.extra = null)],
    ["missing", (_, first) => delete first.text],
    ["bad_id", (_, first) => (first.note_id = "wrong")],
    ["foreign_run", (value) => (value.notes = [note({ runId: "elsewhere" })])],
    ["foreign_sequence", (value) => (value.notes = [note({ sequence: 2n })])],
    ["bool_sequence", (value) => (value.notes = [note({ sequence: true })])],
    ["negative_index", (value) => (value.notes = [note({ index: -1n })])],
    ["bool_index", (value) => (value.notes = [note({ index: true })])],
    ["nontext", (value) => (value.notes = [note({ text: { text: "wrong" } })])],
    ["nan", (_, first) => (first.text = Number.NaN)],
    ["notes_object", (value) => (value.notes = {})],
    ["path_type", (_, first) => (first.consumer_static_path = "root/agent")],
  ];
  for (const [name, mutate] of rows) {
    test(name, async () => {
      const value = await withNotes(note());
      mutate(value, value.notes[0]);
      const memory = new SessionMemory();
      failure(() => memory.openRun({ snapshot: value }), "session.invalid_snapshot", "invalid SessionMemory snapshot");
      expect(memory["used"]).toBe(false);
    });
  }
});

test("historical note prefix is pinned without rewinding live state", async () => {
  const full = await withNotes(note(), note({ index: 1n, text: "Use exact arithmetic." }));
  const memory = new SessionMemory();
  memory.openRun({ snapshot: full }).close();
  const older = structuredClone(full);
  older.notes = older.notes.slice(0, 1);
  const opened = memory.openRun({ snapshot: older });
  try {
    expect(opened.snapshot).toEqual(older);
    expect(snapshot(memory)).toEqual(full);
  } finally {
    opened.close();
  }
  for (const notes of [full.notes.slice(1), full.notes.toReversed(), [note({ text: "Changed" })]]) {
    failure(
      () => memory.openRun({ snapshot: { ...full, notes } }),
      "session.invalid_snapshot",
      "snapshot is not proven history of this backend",
    );
  }
});

test("learn is pure deterministic self feedback projection", async () => {
  await withHarness(async (h) => {
    await admit(h);
    const before = snapshot(h.memory);
    const texts = ["Check units.", ""];
    const learning = context({ texts });
    const first = h.frame.learn(learning);
    expect(first.length).toBe(2);
    expect(first).toEqual(h.frame.learn(learning));
    expect(snapshot(h.memory)).toEqual(before);
    first.forEach((proposal, index) => {
      const expected = note({ index: BigInt(index), text: texts[index] });
      expect(proposal.mutationId).toBe(expected.note_id);
      expect(proposal.payload).toEqual({ operation: "append_note", note: expected });
    });
    expect(first[0]!.mutationId).not.toBe(h.frame.learn(context({ texts: ["Distinct"] }))[0]!.mutationId);
  });
});

describe("learn ignores feedback without owning admitted leaf", () => {
  const rows: [string, Parameters<typeof context>[0]][] = [
    ["no_feedback", { texts: [] }],
    ["other_channel", { channel: "input" }],
    ["root_only", {}],
    ["other_run", { runId: "another-run" }],
    ["unadmitted_start", { sequence: 2, feedbackSequence: 2n }],
    ["no_admission", {}],
    ["zero_sequence", { sequence: 0, feedbackSequence: 0n }],
    ["missing_sequence", { sequence: null, feedbackSequence: null }],
  ];
  for (const [name, options] of rows) {
    test(name, async () => {
      await withHarness(async (h) => {
        if (name !== "no_admission") await admit(h);
        let learning = context(options);
        if (name === "root_only") {
          learning = new LearnContext("a", "session", learning.episodes, [
            {
              execution_seq: null,
              owner_static_path: null,
              root: true,
              channel: "self",
              value: "Root only",
              message_index: 0n,
            },
          ]);
        }
        expect(h.frame.learn(learning)).toEqual([]);
      });
    });
  }
});

test("commit publishes notes only to future run recall", async () => {
  await withHarness(async (h) => {
    await admit(h);
    const history = (snapshot(h.memory) as Mutable).conversation;
    h.session.commit("learn-a", h.frame.learn(context()));
    expect((snapshot(h.memory) as Mutable).notes).toEqual([note()]);
    expect((snapshot(h.memory) as Mutable).conversation).toEqual(history);
    expect(recall(h.frame)).toBe("");
    await withHarness((later) => {
      expect(recall(later.frame)).toBe("Check units.");
      const content = later.frame.recall();
      expect(content.text).toBe("");
      expect(content.transcript).toBe(later.frame);
      expect(recall(later.frame, ["root", "sibling"])).toBe("Check units.");
    }, h.memory);
  });
});

test("exact commit replay and stable note ids are idempotent", async () => {
  await withHarness(async (h) => {
    await admit(h);
    const batch = [mutation()];
    h.session.commit("same", batch);
    h.session.commit("same", batch);
    h.session.commit("another", batch);
    expect((snapshot(h.memory) as Mutable).notes).toEqual([note()]);
    const distinct = mutation(note({ text: "A distinct observation" }));
    h.session.commit("new-feedback", [distinct]);
    const expected = snapshot(h.memory);
    failure(
      () => h.session.commit("same", [distinct]),
      "session.commit_collision",
      "commit id identifies a different batch",
    );
    expect(snapshot(h.memory)).toEqual(expected);
  });
});

describe("invalid last mutation publishes no notes or commit tombstone", () => {
  const rows: [string, (first: MemoryMutation, second: MemoryMutation) => unknown][] = [
    ["extra", (_, second) => new MemoryMutation(second.mutationId, { ...(second.payload as Mutable), extra: null })],
    [
      "operation",
      (_, second) => new MemoryMutation(second.mutationId, { ...(second.payload as Mutable), operation: "other" }),
    ],
    ["wrong_id", (_, second) => new MemoryMutation("wrong", second.payload)],
    ["unadmitted", () => mutation(note({ runId: "missing", index: 1n }))],
    ["duplicate", (first) => first],
    ["nonmutation", () => ({})],
  ];
  for (const [name, build] of rows) {
    test(name, async () => {
      await withHarness(async (h) => {
        await admit(h);
        const before = snapshot(h.memory);
        const first = mutation();
        const second = build(first, mutation(note({ index: 1n })));
        failure(
          () => h.session.commit("atomic", [first, second as MemoryMutation]),
          "session.invalid_batch",
          "mutation batch is invalid",
        );
        expect(snapshot(h.memory)).toEqual(before);
        h.session.commit("atomic", [first]);
        expect((snapshot(h.memory) as Mutable).notes).toEqual([note()]);
      });
    });
  }
});

test("empty commit identity cannot later publish a different batch", async () => {
  await withHarness(async (h) => {
    await admit(h);
    h.session.commit("empty", []);
    failure(
      () => h.session.commit("empty", [mutation()]),
      "session.commit_collision",
      "commit id identifies a different batch",
    );
    expect((snapshot(h.memory) as Mutable).notes).toEqual([]);
  });
});

test("reset clears notes and keeps commit replay tombstones", async () => {
  await withHarness(async (h) => {
    await admit(h);
    const batch = [mutation()];
    h.session.commit("before-reset", batch);
    const old: Mutable = snapshot(h.memory);
    h.memory.reset();
    failure(
      () => h.session.commit("before-reset", batch),
      "session.stale",
      "memory session is closed or belongs to an old generation",
    );
    failure(() => recall(h.frame), "session.stale", "memory session is closed or belongs to an old generation");
    await withHarness((later) => {
      later.session.commit("before-reset", batch);
      expect(recall(later.frame)).toBe("");
      const now: Mutable = snapshot(h.memory);
      expect(now.notes).toEqual([]);
      expect(now.conversation).toBeNull();
      expect(now.generation).toBe(old.generation + 1n);
      failure(() => later.session.commit("new-identity", batch), "session.invalid_batch", "mutation batch is invalid");
    }, h.memory);
  });
});

test("restored notes recall and accept idempotent note republication", async () => {
  const value = await withNotes(note(), note({ index: 1n, text: "" }));
  const memory = new SessionMemory();
  const opened = memory.openRun({ snapshot: value });
  try {
    expect(recall(opened)).toBe("Check units.\n");
    opened.commit("restored", value.notes.map(mutation));
    expect(snapshot(memory)).toEqual(value);
  } finally {
    opened.close();
  }
});

describe("malformed attributed self feedback fails without publication", () => {
  const rows: [string, unknown][] = [
    ["execution_seq", true],
    ["message_index", true],
    ["message_index", -1n],
    ["message_index", null],
    ["value", null],
    ["value", { text: "not text" }],
  ];
  for (const [field, value] of rows) {
    test(`${field} ${logJson(value)}`, async () => {
      await withHarness(async (h) => {
        await admit(h);
        const before = snapshot(h.memory);
        const learning = context();
        const feedback = { ...(learning.feedback[0] as Mutable), [field]: value };
        failure(
          () => h.frame.learn(new LearnContext("a", "session", learning.episodes, [feedback])),
          "session.invalid_context",
          "learning context is invalid",
        );
        expect(snapshot(h.memory)).toEqual(before);
      });
    });
  }
});

test("repeat source feedback in one context proposes one note", async () => {
  await withHarness(async (h) => {
    await admit(h);
    const learning = context();
    const feedback = [...learning.feedback, ...learning.feedback];
    const duplicated = new LearnContext("a", "session", learning.episodes, feedback);
    expect(h.frame.learn(duplicated)).toEqual([mutation()]);
  });
});

test("candidate failure keeps notes and commit identity unpublished", async () => {
  await withHarness(async (h) => {
    await admit(h);
    const before = snapshot(h.memory);
    const rejected = new Error("candidate validation failed");
    const save = spyOn(h.memory as Mutable, "saveState").mockImplementation(() => {
      throw rejected;
    });
    try {
      expect(caught(() => h.session.commit("candidate", [mutation()]))).toBe(rejected);
    } finally {
      save.mockRestore();
    }
    expect(snapshot(h.memory)).toEqual(before);
    h.session.commit("candidate", [mutation()]);
    expect((snapshot(h.memory) as Mutable).notes).toEqual([note()]);
  });
});

test("exact replay from another run handle is inert", async () => {
  await withHarness(async (h) => {
    await admit(h);
    h.session.commit("owned", [mutation()]);
    const unopened = h.memory.openRun();
    try {
      unopened.commit("owned", [mutation()]);
    } finally {
      unopened.close();
    }
    expect((snapshot(h.memory) as Mutable).notes).toEqual([note()]);
  });
});

test("busy reset preserves notes and forward turn updates preserve notes", async () => {
  await withHarness(async (h) => {
    await admit(h);
    h.session.commit("note", [mutation()]);
    await withTurn(h.frame, request("b"), () => {
      const before = snapshot(h.memory);
      failure(() => h.memory.reset(), "session.busy", "cannot reset active or queued conversations");
      expect(snapshot(h.memory)).toEqual(before);
    });
    expect((snapshot(h.memory) as Mutable).notes).toEqual([note()]);
  });
});

test("reset keeps empty commit tombstones", async () => {
  await withHarness(async (h) => {
    h.session.commit("empty", []);
    h.memory.reset();
    await withHarness(async (later) => {
      await admit(later);
      failure(
        () => later.session.commit("empty", [mutation()]),
        "session.commit_collision",
        "commit id identifies a different batch",
      );
    }, h.memory);
  });
});

test("notes belong to the instance and reach every consumer", async () => {
  const memory = new SessionMemory();
  await withHarness(async (h) => {
    await admit(h);
    const mutations = h.frame.learn(context());
    expect(Object.keys((mutations[0]!.payload as Mutable).note).toSorted()).toEqual([
      "consumer_static_path",
      "execution_seq",
      "message_index",
      "note_id",
      "run_id",
      "text",
    ]);
    h.session.commit("learn", mutations);
  }, memory);
  await withHarness((later) => {
    expect(recall(later.frame, ["root", "elsewhere"])).toBe("Check units.");
    expect(recall(later.session)).toBe("Check units.");
  }, memory);
});

test("session is the transcript source and learns without frames", async () => {
  const memory = new SessionMemory();
  const session = memory.openRun();
  let mutations: MemoryMutation[];
  try {
    expect("openFrame" in session).toBe(false);
    const content = session.recall();
    expect(content.text).toBe("");
    expect(content.transcript).toBe(session);
    await withTurn(session, request(), () => {});
    mutations = session.learn(context());
    expect(mutations.length).toBe(1);
    expect("frameRef" in mutations[0]!).toBe(false);
    session.commit("learn", mutations);
  } finally {
    session.close();
  }
  const replay = memory.openRun();
  try {
    replay.commit("learn", mutations);
    expect(replay.recall(new RecallContext("s", "p", {}, ["root", "x"])).text).toBe("Check units.");
  } finally {
    replay.close();
  }
});
