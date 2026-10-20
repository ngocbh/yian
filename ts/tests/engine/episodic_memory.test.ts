/**
 * Session and snapshot foundations for process-local episodic memory: a port of
 * `tests/engine/test_episodic_memory.py`.
 *
 * Ported: `test_constructor_and_protocol_are_exact`,
 * `test_session_operations_validate_context_and_generation`,
 * `test_keys_of_one_instance_share_one_history_and_one_limit`,
 * `test_unset_limits_adopt_recorded_ones_and_explicit_ones_must_match`,
 * `test_snapshot_is_closed_canonical_and_defensive`, `test_snapshot_restore_rules_and_recorded_defaults`,
 * `test_snapshot_format_is_strict`, `test_empty_commit_idempotency_collision_and_staleness`,
 * `test_commit_inputs_are_exact`, `test_closed_session_rejects_every_operation`,
 * `test_module_imports_only_stdlib_and_lower_memory_modules`,
 * `test_learning_is_deterministic_complete_and_round_trips_snapshot`,
 * `test_commit_is_atomic_rejects_malformed_and_duplicate_entries`,
 * `test_handcrafted_out_of_scope_commit_is_rejected_atomically`,
 * `test_retention_recall_order_relevance_and_pinned_history`,
 * `test_rewrite_query_is_overridable_and_must_return_text`,
 * `test_nonempty_snapshot_rejects_malformed_episode_and_duplicate_identity` and
 * `test_session_recalls_and_learns_under_the_memory_key`.
 *
 * TS-only: the constructor's `bigint` limits (`0n` refused, one past 2^53 − 1 refused), `null`
 * limits as unset and a non-boolean `createParent`; an episode holding `Infinity` in its inputs
 * fails `learn` as `episodic.invalid_context`; a snapshot holding `Infinity` in an episode is
 * `episodic.invalid_snapshot`, and one holding `-0` reads as `0`; a host mutation whose payload holds
 * `-0` is `episodic.invalid_batch` (not canonical JSON); the base `rewriteQuery` text and its
 * `TypeError`; an async `rewriteQuery` is awaited, a rejecting one is `episodic.invalid_query` with
 * its cause, and a re-entrant `memory.close()` from inside it is `episodic.busy`.
 *
 * Changed:
 * - `test_constructor_and_protocol_are_exact`: `1.0` is the int 1 in JS, so it's accepted; `1.5`
 *   and `"1"` stand in for Python's non-int rows. `isMemory` and the session's methods stand in for
 *   the runtime-checkable protocol `isinstance` checks.
 * - `test_snapshot_rejects_nesting_beyond_the_memory_limit`: TS has no memory nesting limit, so the
 *   200-level `config.extra` row fails only for its extra field, and a 10 000-level value inside an
 *   episode's inputs round-trips through learn, commit, snapshot and restore.
 * - `test_snapshot_is_closed_canonical_and_defensive`: a snapshot is deep-frozen, not a detached
 *   mutable copy, so the assignment throws `TypeError` and the snapshot is unchanged.
 * - `test_commit_inputs_are_exact`: an array is the valid batch type in TS, so the `[]` row commits;
 *   a non-array (`{}`, a `Set`) is `episodic.invalid_batch`.
 * - `test_closed_session_rejects_every_operation`: there is no `__enter__` row (no `with` form); the
 *   other operations are checked.
 * - JSON ints written into snapshots are `bigint`s; mutable copies are `structuredClone`s; `recall`
 *   is async; `with` blocks are `try`/`finally` around `close()`.
 *
 * Not ported: none.
 */

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import {
  isMemory,
  type JsonObject,
  LearnContext,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryOperationError,
  MemorySection,
  ModelTextInteraction,
  ModelValueInteraction,
  RecallContext,
  ToolCallInteraction,
  ToolResultInteraction,
} from "../../src/memory/contracts.ts";
import { EpisodicMemory, type EpisodicSession } from "../../src/memory/episodic.ts";
import { logJson } from "../../src/typesys/json.ts";
import { importStatements } from "./source_scan.ts";

// Cloned snapshots and nested test values are edited freely.
type Mutable = any;

/** A one-episode learning context for `memory`. */
function context(memory = "episodic"): LearnContext {
  const episode = new MemoryEpisode("run", [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, {}, { output: "ok" });
  return new LearnContext("run", memory, [episode]);
}

/** Return the `MemoryOperationError` `fn` throws. */
function failure(fn: () => unknown): MemoryOperationError {
  try {
    fn();
  } catch (error) {
    expect(error).toBeInstanceOf(MemoryOperationError);
    return error as MemoryOperationError;
  }
  throw new Error("expected a MemoryOperationError");
}

/** Return the `MemoryOperationError` `promise` rejects with. */
async function rejection(promise: Promise<unknown>): Promise<MemoryOperationError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(MemoryOperationError);
    return error as MemoryOperationError;
  }
  throw new Error("expected a MemoryOperationError");
}

/** Open a run, await `fn` with it and close it (Python's `with memory.open_run() as session`). */
async function withRun<T>(
  memory: EpisodicMemory,
  fn: (session: EpisodicSession) => T | Promise<T>,
  snapshot?: JsonObject,
): Promise<T> {
  const session = memory.openRun({ snapshot });
  try {
    return await fn(session);
  } finally {
    session.close();
  }
}

/** The pinned snapshot of a fresh run of `memory`. */
function snapshotOf(memory: EpisodicMemory): JsonObject {
  const session = memory.openRun();
  try {
    return session.snapshot;
  } finally {
    session.close();
  }
}

/** A complete loop-iteration episode labelled `label`. */
function episode(label: string, status: MemoryEpisodeStatus = MemoryEpisodeStatus.SUCCEEDED): MemoryEpisode {
  const failed = status === MemoryEpisodeStatus.FAILED;
  return new MemoryEpisode(
    `run-${label}`,
    [new MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ["loop"], 2)],
    ["loop", "agent"],
    status,
    { topic: label },
    {
      output: failed ? null : { answer: label },
      failure: failed ? { code: "failed", message: label } : null,
      interactions: [
        new ModelTextInteraction(`text ${label}`),
        new ModelValueInteraction({ value: label }),
        new ToolCallInteraction("lookup", { query: label }),
        new ToolResultInteraction("lookup", `result ${label}`),
      ],
      baseSystemPrompt: "system",
      prompt: `prompt ${label}`,
      memorySections: [new MemorySection("episodic", `memory ${label}`)],
      executionSeq: 7,
    },
  );
}

/** Learn and commit one episode in its own run. */
async function learnInto(memory: EpisodicMemory, item: MemoryEpisode, commitId: string): Promise<void> {
  await withRun(memory, (session) => {
    session.commit(commitId, session.learn(new LearnContext(item.runId, "episodic", [item])));
  });
}

const SNAPSHOT_FIELDS = ["config", "episodes", "epoch", "format", "namespace", "revision", "version"];

test("constructor and protocol are exact", async () => {
  for (const value of [true, 1.5, "1"]) {
    expect(() => new EpisodicMemory({ maxEpisodes: value as never })).toThrow(
      new TypeError("max_episodes must be an integer"),
    );
    expect(() => new EpisodicMemory({ recallLimit: value as never })).toThrow(
      new TypeError("recall_limit must be an integer"),
    );
  }
  for (const value of [0, -1, 0n]) {
    expect(() => new EpisodicMemory({ maxEpisodes: value })).toThrow(new RangeError("max_episodes must be positive"));
    expect(() => new EpisodicMemory({ recallLimit: value })).toThrow(new RangeError("recall_limit must be positive"));
  }
  expect(() => new EpisodicMemory({ maxEpisodes: 2n ** 53n })).toThrow(
    new RangeError("max_episodes must be at most 9007199254740991, got 9007199254740992"),
  );
  expect(() => new EpisodicMemory({ createParent: 1 as never })).toThrow(new TypeError("create_parent must be a bool"));
  expect(snapshotOf(new EpisodicMemory({ maxEpisodes: 1.0, recallLimit: 2n })).config).toEqual({
    max_episodes: 1n,
    recall_limit: 2n,
  });
  expect(snapshotOf(new EpisodicMemory({ maxEpisodes: null, recallLimit: null })).config).toEqual({
    max_episodes: 100n,
    recall_limit: 10n,
  });

  const memory = new EpisodicMemory({ maxEpisodes: 7, recallLimit: 3 });
  expect(isMemory(memory)).toBe(true);
  await withRun(memory, (session) => {
    for (const method of ["recall", "learn", "commit", "close"] as const) {
      expect(typeof session[method]).toBe("function");
    }
    expect(typeof session.snapshot).toBe("object");
    expect("openFrame" in session).toBe(false);
    expect("open_frame" in session).toBe(false);
  });
});

test("session operations validate context and generation", async () => {
  const memory = new EpisodicMemory();
  const session = memory.openRun();
  expect(await session.recall()).toBe("");
  expect(await session.recall(new RecallContext("agent", "prompt", {}, ["root"]))).toBe("");
  expect(session.learn(context()).length).toBe(1);
  expect((await rejection(session.recall({} as never))).code).toBe("episodic.invalid_context");
  expect(failure(() => session.learn({} as never)).code).toBe("episodic.invalid_context");
  memory.reset();
  expect(await session.recall()).toBe("");
  expect(failure(() => session.learn(context())).code).toBe("episodic.stale_generation");
  expect(failure(() => session.commit("stale", [])).code).toBe("episodic.stale_generation");
  session.close();
});

test("keys of one instance share one history and one limit", async () => {
  const memory = new EpisodicMemory({ maxEpisodes: 2, recallLimit: 1 });
  const batch = await withRun(memory, (session) => {
    const both = [...session.learn(context("first")), ...session.learn(context("second"))];
    session.commit("both", both);
    return both;
  });
  const third = await withRun(memory, async (session) => {
    const snapshot = session.snapshot;
    expect(snapshot.version).toBe(2n);
    expect(Object.keys(snapshot).toSorted()).toEqual(SNAPSHOT_FIELDS);
    expect(snapshot.config).toEqual({ max_episodes: 2n, recall_limit: 1n });
    expect((snapshot.episodes as readonly unknown[]).length).toBe(2);
    expect((await session.recall()).split("\n").length).toBe(2);
    const learned = session.learn(context("third"));
    session.commit("third", learned);
    return learned;
  });
  await withRun(memory, (session) => {
    const retained = (session.snapshot.episodes as readonly JsonObject[]).map((item) => item.entry_id);
    expect(retained).toEqual([batch[1]!.mutationId, third[0]!.mutationId]);
  });
});

test("unset limits adopt recorded ones and explicit ones must match", async () => {
  expect(snapshotOf(new EpisodicMemory()).config).toEqual({ max_episodes: 100n, recall_limit: 10n });
  const recorded = snapshotOf(new EpisodicMemory({ maxEpisodes: 4, recallLimit: 2 }));
  await withRun(new EpisodicMemory(), (session) => expect(session.snapshot).toEqual(recorded), recorded);
  await withRun(
    new EpisodicMemory({ maxEpisodes: 4 }),
    (session) => expect(session.snapshot).toEqual(recorded),
    recorded,
  );
  for (const options of [{ maxEpisodes: 5 }, { recallLimit: 3 }]) {
    const error = failure(() => new EpisodicMemory(options).openRun({ snapshot: recorded }));
    expect(error.code).toBe("episodic.config_conflict");
    expect(error.retryable).toBe(false);
  }
});

test("snapshot is closed canonical and defensive", async () => {
  const memory = new EpisodicMemory({ maxEpisodes: 4, recallLimit: 2 });
  await withRun(memory, (session) => {
    const snapshot = session.snapshot;
    expect(Object.keys(snapshot).toSorted()).toEqual(SNAPSHOT_FIELDS);
    expect(snapshot.format).toBe("yian.episodic_memory");
    expect(snapshot.version).toBe(2n);
    expect(snapshot.config).toEqual({ max_episodes: 4n, recall_limit: 2n });
    expect(snapshot.episodes).toEqual([]);
    expect(() => {
      (snapshot.config as Mutable).max_episodes = 99n;
    }).toThrow(TypeError);
    expect((session.snapshot.config as JsonObject).max_episodes).toBe(4n);
    logJson(session.snapshot);
  });
});

test("snapshot restore rules and recorded defaults", async () => {
  const original = new EpisodicMemory({ maxEpisodes: 4, recallLimit: 2 });
  await withRun(original, (session) => session.commit("declare", []));
  const recorded = snapshotOf(original);
  await withRun(original, (current) => expect(current.snapshot).toEqual(recorded), recorded);
  await withRun(new EpisodicMemory(), (session) => expect(session.snapshot).toEqual(recorded), recorded);
  const nonpristine = new EpisodicMemory();
  await withRun(nonpristine, () => {});
  failure(() => nonpristine.openRun({ snapshot: recorded }));
  for (const [field, value] of [
    ["revision", 2n],
    ["epoch", 1n],
  ] as const) {
    const changed: Mutable = structuredClone(recorded);
    changed[field] = value;
    failure(() => original.openRun({ snapshot: changed }));
  }
  const changed: Mutable = structuredClone(recorded);
  changed.config.recall_limit = 3n;
  failure(() => original.openRun({ snapshot: changed }));
  original.reset();
  failure(() => original.openRun({ snapshot: recorded }));
});

describe("snapshot format is strict", () => {
  const rows: [string, (x: Mutable) => void][] = [
    ["extra", (x) => (x.extra = true)],
    ["version true", (x) => (x.version = true)],
    ["format", (x) => (x.format = "other")],
    ["namespace", (x) => (x.namespace = "")],
    ["epoch true", (x) => (x.epoch = true)],
    ["revision -1", (x) => (x.revision = -1n)],
    ["version 1", (x) => (x.version = 1n)],
    ["frames", (x) => (x.frames = [])],
    ["episodes object", (x) => (x.episodes = {})],
    ["max_episodes 0", (x) => (x.config.max_episodes = 0n)],
    ["config extra", (x) => (x.config.extra = 1n)],
    ["empty episode", (x) => x.episodes.push({})],
  ];
  for (const [name, mutate] of rows) {
    test(name, () => {
      const malformed: Mutable = structuredClone(snapshotOf(new EpisodicMemory()));
      mutate(malformed);
      const error = failure(() => new EpisodicMemory().openRun({ snapshot: malformed }));
      expect(error.code).toBe("episodic.invalid_snapshot");
      expect(error.retryable).toBe(false);
    });
  }
});

// test_snapshot_rejects_nesting_beyond_the_memory_limit (Changed: no memory nesting limit)
test("a deep value round-trips and a deep extra config field is still refused", async () => {
  const malformed: Mutable = structuredClone(snapshotOf(new EpisodicMemory()));
  let nested: Mutable = {};
  for (let i = 0; i < 200; i++) nested = { nested };
  malformed.config.extra = nested;
  expect(failure(() => new EpisodicMemory().openRun({ snapshot: malformed })).code).toBe("episodic.invalid_snapshot");

  let deep: Mutable = {};
  for (let i = 0; i < 10_000; i++) deep = { nested: deep };
  const item = new MemoryEpisode("run", [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, { deep }, { output: "ok" });
  const memory = new EpisodicMemory();
  await learnInto(memory, item, "deep");
  const snapshot = snapshotOf(memory);
  const restored = new EpisodicMemory();
  await withRun(
    restored,
    async (session) => {
      expect(logJson(session.snapshot)).toBe(logJson(snapshot));
      expect(await session.recall()).toBe(logJson((snapshot.episodes as readonly JsonObject[])[0]));
    },
    snapshot,
  );
});

test("empty commit idempotency collision and staleness", async () => {
  const memory = new EpisodicMemory();
  const session = memory.openRun();
  session.commit("empty", []);
  session.commit("empty", []);
  await withRun(memory, (fresh) => expect(fresh.snapshot.revision).toBe(1n));
  expect(failure(() => session.commit("empty", [new MemoryMutation("m", {})])).code).toBe("episodic.commit_collision");
  expect(failure(() => memory.openRun().commit("nonempty", [new MemoryMutation("m", {})])).code).toBe(
    "episodic.invalid_batch",
  );
  const stale = memory.openRun();
  await withRun(memory, (current) => current.commit("next", []));
  expect(failure(() => stale.commit("stale", [])).code).toBe("episodic.stale_generation");
  stale.close();
  session.close();
});

test("commit inputs are exact", async () => {
  await withRun(new EpisodicMemory(), (session) => {
    for (const commitId of ["", true, 1, null]) {
      expect(failure(() => session.commit(commitId as never, [])).code).toBe("episodic.invalid_commit_id");
    }
    session.commit("id", []);
    expect(session.snapshot.revision).toBe(0n);
    for (const batch of [{}, new Set()]) {
      const error = failure(() => session.commit("other", batch as never));
      expect(error.code).toBe("episodic.invalid_batch");
      expect(error.message).toBe("mutations must be an exact array");
    }
  });
});

test("closed session rejects every operation", async () => {
  const session = new EpisodicMemory().openRun();
  session.close();
  session.close();
  for (const operation of [() => session.snapshot, () => session.commit("x", []), () => session.learn(context())]) {
    expect(failure(operation).code).toBe("episodic.session_closed");
  }
  expect((await rejection(session.recall())).code).toBe("episodic.session_closed");
});

test("module imports only stdlib and lower memory modules", () => {
  const source = readFileSync(new URL("../../src/memory/episodic.ts", import.meta.url), "utf8");
  const allowed = new Set([
    "node:crypto",
    "../defaults.ts",
    "../typesys/json.ts",
    "./contracts.ts",
    "./storage/sqlite.ts",
  ]);
  const imported = importStatements(source).map((statement) => statement.specifier);
  expect(imported.filter((specifier) => !allowed.has(specifier))).toEqual([]);
  expect(imported).not.toContain("./index.ts");
});

test("learning is deterministic complete and round trips snapshot", async () => {
  const memory = new EpisodicMemory();
  const item = episode("alpha");
  await withRun(memory, (session) => {
    expect(session.learn(new LearnContext("empty", "episodic", []))).toEqual([]);
    const learning = new LearnContext(item.runId, "episodic", [item]);
    const first = session.learn(learning);
    const second = session.learn(learning);
    expect(first.map((m) => m.mutationId)).toEqual(second.map((m) => m.mutationId));
    expect(first.map((m) => m.payload)).toEqual(second.map((m) => m.payload));
    expect((first[0]!.payload as JsonObject).operation).toBe("append");
    session.commit("learn-alpha", first);
    session.commit("learn-alpha", first);
  });
  const [snapshot, rendered] = await withRun(
    memory,
    async (session) => [session.snapshot, await session.recall()] as const,
  );
  expect(rendered).toBe(logJson((snapshot.episodes as readonly JsonObject[])[0]));
  expect(JSON.parse(rendered).episode.interactions).toEqual([
    { kind: "model_text", text: "text alpha" },
    { kind: "model_value", value: { value: "alpha" } },
    { arguments: { query: "alpha" }, kind: "tool_call", name: "lookup" },
    { kind: "tool_result", name: "lookup", text: "result alpha" },
  ]);
  await withRun(new EpisodicMemory(), async (session) => expect(await session.recall()).toBe(rendered), snapshot);

  const failedMemory = new EpisodicMemory();
  await learnInto(failedMemory, episode("failure", MemoryEpisodeStatus.FAILED), "learn-failure");
  const failed = await withRun(failedMemory, async (session) => JSON.parse(await session.recall()).episode);
  expect(failed.status).toBe("failed");
  expect(failed.failure).toEqual({ code: "failed", message: "failure" });
  expect(failed.output).toBeNull();
});

test("commit is atomic rejects malformed and duplicate entries", async () => {
  const memory = new EpisodicMemory();
  const item = episode("alpha");
  await withRun(memory, (session) => {
    const valid = session.learn(new LearnContext(item.runId, "episodic", [item]))[0]!;
    const malformed = new MemoryMutation("wrong", valid.payload);
    expect(failure(() => session.commit("bad", [valid, malformed])).code).toBe("episodic.invalid_batch");
    expect(session.snapshot.revision).toBe(0n);
    session.commit("good", [valid]);
  });
  await withRun(memory, (session) => {
    expect((session.snapshot.episodes as readonly unknown[]).length).toBe(1);
    const again = session.learn(new LearnContext(item.runId, "episodic", [item]));
    expect(failure(() => session.commit("duplicate", again)).code).toBe("episodic.invalid_batch");
  });
});

test("handcrafted out of scope commit is rejected atomically", async () => {
  const memory = new EpisodicMemory();
  const item = episode("alpha");
  await withRun(memory, (session) => {
    const valid = session.learn(new LearnContext(item.runId, "episodic", [item]))[0]!;
    const payload: Mutable = structuredClone(valid.payload);
    // The episode's occurrence path no longer contains its scoped node.
    payload.entry.episode.owner_static_path = ["root", "outside"];
    const forged = new MemoryMutation(valid.mutationId, payload);
    expect(failure(() => session.commit("forged-scope", [forged])).code).toBe("episodic.invalid_batch");
    expect(session.snapshot.revision).toBe(0n);
    expect(session.snapshot.episodes).toEqual([]);
    session.commit("valid-after-forgery", [valid]);
  });
  await withRun(memory, (session) => expect((session.snapshot.episodes as readonly unknown[]).length).toBe(1));
});

test("retention recall order relevance and pinned history", async () => {
  const memory = new EpisodicMemory({ maxEpisodes: 2, recallLimit: 1 });
  const pinned = memory.openRun();
  expect(await pinned.recall()).toBe("");
  await learnInto(memory, episode("alpha"), "one");
  await learnInto(memory, episode("beta alpha"), "two");
  await learnInto(memory, episode("gamma"), "three");
  expect(await pinned.recall()).toBe("");
  memory.reset();
  expect(await pinned.recall()).toBe("");
  pinned.close();

  const restored = new EpisodicMemory({ maxEpisodes: 2, recallLimit: 1 });
  await learnInto(restored, episode("alpha"), "one");
  await learnInto(restored, episode("beta alpha"), "two");
  await learnInto(restored, episode("gamma"), "three");
  await withRun(restored, async (session) => {
    const history = (await session.recall()).split("\n").map((line) => JSON.parse(line));
    expect(history.map((item) => item.episode.run_id)).toEqual(["run-beta alpha", "run-gamma"]);
    const relevant = JSON.parse(await session.recall(new RecallContext("", "alpha", {}, ["root", "agent"])));
    expect(relevant.episode.run_id).toBe("run-beta alpha");
    const recent = JSON.parse(await session.recall(new RecallContext("", "absent", {}, ["root", "agent"])));
    expect(recent.episode.run_id).toBe("run-gamma");
  });
});

test("rewrite query is overridable and must return text", async () => {
  class FixedQueryMemory extends EpisodicMemory {
    override rewriteQuery(): string {
      return "alpha";
    }
  }
  const memory = new FixedQueryMemory({ recallLimit: 1 });
  await learnInto(memory, episode("alpha"), "one");
  await learnInto(memory, episode("newer"), "two");
  await withRun(memory, async (session) => {
    const recalled = await session.recall(new RecallContext("", "ignored", {}, ["root", "agent"]));
    expect(JSON.parse(recalled).episode.run_id).toBe("run-alpha");
  });

  class InvalidQueryMemory extends EpisodicMemory {
    override rewriteQuery(): string {
      return 1 as never;
    }
  }
  await withRun(new InvalidQueryMemory(), async (session) => {
    const error = await rejection(session.recall(new RecallContext("", "", {}, ["root", "agent"])));
    expect(error.code).toBe("episodic.invalid_query");
    expect(error.message).toBe("query rewriting must return text");
  });
});

test("nonempty snapshot rejects malformed episode and duplicate identity", async () => {
  const memory = new EpisodicMemory();
  await learnInto(memory, episode("alpha"), "one");
  const snapshot = snapshotOf(memory);
  const malformed: Mutable = structuredClone(snapshot);
  malformed.episodes[0].episode.status = "unknown";
  failure(() => new EpisodicMemory().openRun({ snapshot: malformed }));
  const duplicate: Mutable = structuredClone(snapshot);
  duplicate.episodes.push(duplicate.episodes[0]);
  failure(() => new EpisodicMemory().openRun({ snapshot: duplicate }));
});

test("session recalls and learns under the memory key", async () => {
  const memory = new EpisodicMemory({ maxEpisodes: 3, recallLimit: 1 });
  const segment = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["fan"], 1);
  const item = new MemoryEpisode("run", [segment], ["fan", "agent"], MemoryEpisodeStatus.SUCCEEDED, {}, {
    output: "ok",
    memorySections: [new MemorySection("notes", "prior")],
  });
  await withRun(memory, async (session) => {
    expect("openFrame" in session).toBe(false);
    expect(await session.recall()).toBe("");
    const mutations = session.learn(new LearnContext("run", "notes", [item]));
    const again = session.learn(new LearnContext("run", "notes", [item]));
    const other = session.learn(new LearnContext("run", "other", [item]));
    expect(mutations.map((m) => m.mutationId)).toEqual(again.map((m) => m.mutationId));
    expect(mutations[0]!.mutationId).not.toBe(other[0]!.mutationId);
    const stored = ((mutations[0]!.payload as JsonObject).entry as JsonObject).episode as JsonObject;
    expect(stored.occurrence_path).toEqual([{ index: 1n, kind: "map_item", static_path: ["fan"] }]);
    expect(stored.memory_sections).toEqual([{ memory: "notes", text: "prior" }]);
    expect("occurrence" in stored).toBe(false);
    session.commit("one", mutations);
  });
  await withRun(memory, async (session) => {
    expect((await session.recall()).split("\n").length).toBe(1);
    const query = new RecallContext("base", "prompt", {}, ["fan", "agent"]);
    expect(await session.recall(query)).toBe(await session.recall());
  });
});

describe("TS-only", () => {
  test("an infinite episode input fails learning as an invalid context", async () => {
    const item = new MemoryEpisode("run", [], ["agent"], MemoryEpisodeStatus.SUCCEEDED, { x: Infinity }, {
      output: "ok",
    });
    await withRun(new EpisodicMemory(), (session) => {
      const error = failure(() => session.learn(new LearnContext("run", "episodic", [item])));
      expect(error.code).toBe("episodic.invalid_context");
      expect(error.message).toBe("learning context is invalid");
    });
  });

  test("a snapshot episode holding Infinity is invalid and one holding -0 reads as 0", async () => {
    const memory = new EpisodicMemory();
    await learnInto(memory, episode("alpha"), "one");
    const snapshot = snapshotOf(memory);
    const infinite: Mutable = structuredClone(snapshot);
    infinite.episodes[0].episode.inputs.topic = Infinity;
    expect(failure(() => new EpisodicMemory().openRun({ snapshot: infinite })).code).toBe("episodic.invalid_snapshot");
    const negativeZero: Mutable = structuredClone(snapshot);
    negativeZero.episodes[0].episode.inputs.topic = -0;
    await withRun(
      new EpisodicMemory(),
      (session) => {
        const topic = ((session.snapshot.episodes as Mutable)[0].episode.inputs as JsonObject).topic;
        expect(Object.is(topic, 0)).toBe(true);
      },
      negativeZero,
    );
  });

  test("a host mutation payload holding -0 is not canonical JSON", async () => {
    await withRun(new EpisodicMemory(), (session) => {
      const error = failure(() => session.commit("zero", [new MemoryMutation("m", { x: -0 })]));
      expect(error.code).toBe("episodic.invalid_batch");
      expect(error.message).toBe("mutation batch is not canonical JSON");
    });
  });

  test("the base rewriteQuery joins the prompts and the inputs' log JSON", () => {
    const memory = new EpisodicMemory();
    expect(memory.rewriteQuery(new RecallContext("base", "prompt", { b: 1n, a: 1.5 }, ["root"]))).toBe(
      'base\nprompt\n{"a":1.5,"b":1}',
    );
    expect(() => memory.rewriteQuery({} as never)).toThrow(new TypeError("context must be a RecallContext"));
  });

  test("an async rewriteQuery is awaited and a rejecting one is an invalid query", async () => {
    class AsyncQueryMemory extends EpisodicMemory {
      override async rewriteQuery(): Promise<string> {
        await new Promise((resolve) => setTimeout(resolve, 1));
        return "alpha";
      }
    }
    const memory = new AsyncQueryMemory({ recallLimit: 1 });
    await learnInto(memory, episode("alpha"), "one");
    await learnInto(memory, episode("newer"), "two");
    await withRun(memory, async (session) => {
      const recalled = await session.recall(new RecallContext("", "ignored", {}, ["root", "agent"]));
      expect(JSON.parse(recalled).episode.run_id).toBe("run-alpha");
    });

    const cause = new Error("model unavailable");
    class RejectingQueryMemory extends EpisodicMemory {
      override async rewriteQuery(): Promise<string> {
        throw cause;
      }
    }
    await withRun(new RejectingQueryMemory(), async (session) => {
      const error = await rejection(session.recall(new RecallContext("", "", {}, ["root", "agent"])));
      expect(error.code).toBe("episodic.invalid_query");
      expect(error.message).toBe("query rewriting failed");
      expect(error.cause).toBe(cause);
    });
  });

  test("a re-entrant close from inside an async rewriteQuery is busy", async () => {
    let closeError: unknown = null;
    class ClosingQueryMemory extends EpisodicMemory {
      override async rewriteQuery(): Promise<string> {
        await new Promise((resolve) => setTimeout(resolve, 1));
        try {
          this.close();
        } catch (error) {
          closeError = error;
        }
        return "query";
      }
    }
    const memory = new ClosingQueryMemory();
    await withRun(memory, async (session) => {
      expect(await session.recall(new RecallContext("", "", {}, ["root", "agent"]))).toBe("");
    });
    expect(closeError).toBeInstanceOf(MemoryOperationError);
    const busy = closeError as MemoryOperationError;
    expect([busy.stage, busy.code, busy.message, busy.retryable]).toEqual([
      "close",
      "episodic.busy",
      "memory backend is active",
      true,
    ]);
    memory.close();
    memory.close();
    expect(failure(() => memory.openRun()).code).toBe("episodic.closed");
  });
});
