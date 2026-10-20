/**
 * Process-local reflective-memory behavior: a port of `tests/engine/test_reflective_memory.py`.
 *
 * Ported: `test_constructor_and_protocol_are_exact`,
 * `test_snapshot_contains_only_public_pinned_model_configuration`,
 * `test_defaults_are_read_once_and_recorded_snapshot_never_rereads_them`,
 * `test_empty_ambient_model_is_rejected_and_open_session_config_is_pinned` (the open half),
 * `test_snapshot_is_defensive_and_foreign_restore_is_conservative`,
 * `test_pristine_restore_keeps_recorded_limits`,
 * `test_unset_limits_adopt_recorded_ones_and_explicit_ones_must_match`,
 * `test_same_object_restore_accepts_recorded_snapshot`, `test_populated_memory_rejects_foreign_snapshot`,
 * `test_snapshot_rejects_noncanonical_scalars_and_circular_json`,
 * `test_reflective_json_rejects_non_utf8_model_and_snapshot_strings`,
 * `test_snapshot_rejects_excessive_nesting_as_typed_invalid_snapshot`,
 * `test_snapshot_rejects_malformed_foreign_and_extra_lesson_sources`,
 * `test_session_idempotency_and_closed_session_operations` (the recall and commit rows),
 * `test_sessions_of_one_instance_share_one_lesson_list_and_limit`,
 * `test_recall_all_and_context_ranking_are_deterministic`, `test_subclass_controls_query_rewriting`,
 * `test_query_rewriting_failure_is_sanitized`,
 * `test_commit_is_atomic_idempotent_and_empty_batches_advance_revision`,
 * `test_revise_and_remove_are_learned_and_committed`,
 * `test_learn_and_commit_enforce_final_max_items` (the commit half),
 * `test_multi_mutation_commit_failure_has_no_partial_effect`,
 * `test_concurrent_same_generation_commits_have_one_winner`,
 * `test_reset_preserves_commit_ledger_ordering_and_old_recall` and
 * `test_old_reflective_snapshot_rejects_active_retired_options`. Also ported: the `organization`,
 * `project` and `headers` rows of `tests/engine/test_run_connection_settings.py::
 * test_reflective_memory_refuses_connection_settings`, in the constructor test.
 *
 * TS-only: `bigint` limits, `null` limits as unset, a limit one past 2^53 − 1 refused, a `number` int
 * in a host `llm_config` block accepted (host intake gives a `bigint`); in the constructor, a blank
 * `model`, the exact `api_key` message, a non-object `llmConfig` (`TypeError`) and `createParent: 1`
 * (`TypeError`); a snapshot whose `revision` is `2n ** 60n` (beyond 2^53 − 1, with its cause
 * message), whose `epoch` is `-1n` or whose `version` is `1n` is `reflective.invalid_snapshot`; one
 * row for each of commit's batch, commit-id and mutation messages; "recall rejects an invalid
 * context and a non-string query"; an async `rewriteQuery` override is awaited (in
 * `test_subclass_controls_query_rewriting`); a re-entrant `memory.close()` from inside it is
 * `reflective.busy`; a host mutation whose payload holds `-0` is `reflective.invalid_batch`
 * "mutation batch cannot be encoded canonically", with a `TypeError` cause; the closing
 * `memory.close()` in `test_session_idempotency_and_closed_session_operations` (a failed recall
 * releases its operation).
 *
 * Changed:
 * - Every backend gets `baseEnv()`, an environment map standing in for Python's `conftest.py`
 *   variables, and each test starts from the built-in defaults.
 * - `test_constructor_and_protocol_are_exact`: `1.0` is the int 1 in JS, so `true`, `1.5` and `"1"`
 *   stand in for the non-int rows; `isMemory` replaces the protocol check.
 * - `test_defaults_are_read_once_and_recorded_snapshot_never_rereads_them`: an `env` counting reads
 *   of `YIAN_DEFAULT_PROVIDER` stands in for patching `resolve_model_config`.
 * - `test_empty_ambient_model_is_rejected_and_open_session_config_is_pinned`: `fail_default` is an
 *   `env` that throws a sentinel on `YIAN_DEFAULT_PROVIDER`, so the cause is the sentinel.
 * - `test_snapshot_is_defensive_and_foreign_restore_is_conservative`: a snapshot is deep-frozen, so
 *   the assignment throws `TypeError` and the next snapshot is unchanged.
 * - `test_snapshot_rejects_noncanonical_scalars_and_circular_json`: the snapshot rows edit a
 *   `structuredClone`.
 * - `test_snapshot_rejects_excessive_nesting_as_typed_invalid_snapshot`: TS has no memory nesting
 *   limit, so the 1 500-level `extra` field fails only as an extra field.
 * - `test_old_reflective_snapshot_rejects_active_retired_options`: TS reads no Python-era snapshot,
 *   so every row fails on the `sandbox: null`/`cwd: null` keys before its own field; each row is
 *   still `reflective.invalid_snapshot`.
 * - `test_commit_is_atomic_idempotent_and_empty_batches_advance_revision`,
 *   `test_revise_and_remove_are_learned_and_committed`,
 *   `test_recall_all_and_context_ranking_are_deterministic`, `test_subclass_controls_query_rewriting`,
 *   `test_reset_preserves_commit_ledger_ordering_and_old_recall`,
 *   `test_concurrent_same_generation_commits_have_one_winner` and the replacement row of
 *   `test_learn_and_commit_enforce_final_max_items` commit hand-built mutations where Python learns
 *   them.
 * - `test_concurrent_same_generation_commits_have_one_winner`: JS commits are synchronous, so two
 *   sessions of one generation commit in turn and the second is stale.
 * - Recall's messages say "context must be a RecallContext or null" and "rewriteQuery() must return
 *   a string" where Python's say `None` and `rewrite_query()`.
 * - JSON ints written into snapshots are `bigint`s; `recall` is async; `with` blocks are
 *   `try`/`finally` around `close()`.
 *
 * Later (in this PR): `test_reflective_episode_material_retains_the_physical_execution_sequence`,
 * `test_learn_uses_pinned_config_and_returns_deterministic_proposals`,
 * `test_learn_rejects_malformed_or_conflicting_actions`,
 * `test_model_construction_failure_is_final_and_sanitized`,
 * `test_structured_binding_failure_is_final_and_sanitized`,
 * `test_learn_provider_failure_is_retryable_and_sanitized`,
 * `test_learn_errors_scrub_the_key_variable_and_saved_login`,
 * `test_structured_output_parser_failure_is_a_final_contract_error`,
 * `test_parsed_response_validation_failure_is_final_and_sanitized`,
 * `test_episode_material_is_capped_in_one_model_call`,
 * `test_huge_current_lessons_cannot_displace_new_episode_evidence`,
 * `test_learning_is_propose_only_and_rejects_a_stale_generation`,
 * `test_reflective_memory_implements_protocol_and_imports_only_lower_layers`,
 * `test_session_reflects_under_the_memory_key`, the learn half of
 * `test_empty_ambient_model_is_rejected_and_open_session_config_is_pinned`, the learn row of
 * `test_session_idempotency_and_closed_session_operations`, the learn half of
 * `test_learn_and_commit_enforce_final_max_items`, and the learned versions of the hand-built
 * commit tests above.
 *
 * Not ported: `test_old_api_snapshot_keeps_digest_and_api_semantics` (TS reads no Python-era
 * snapshot, so there is no retired-key branch).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { BUILTIN, type Env, install } from "../../src/defaults.ts";
import {
  isMemory,
  type JsonObject,
  MemoryMutation,
  MemoryOperationError,
  RecallContext,
} from "../../src/memory/contracts.ts";
import { ReflectiveMemory, type ReflectiveSession } from "../../src/memory/reflective.ts";

// Cloned snapshots and nested test values are edited freely.
type Mutable = any;

let home: string;

beforeEach(() => {
  install(BUILTIN);
  home = mkdtempSync(join(tmpdir(), "yian-reflective-"));
});

afterEach(() => {
  install(BUILTIN);
  rmSync(home, { recursive: true, force: true });
});

/** Python's `conftest.py` environment: an empty private store and an explicit model default. */
function baseEnv(overrides: Record<string, string> = {}): Env {
  const vars = new Map([
    ["YIAN_AUTH_FILE", join(home, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
    ...Object.entries(overrides),
  ]);
  return (name) => vars.get(name);
}

/** A reflective memory reading `baseEnv()` unless `options.env` is given. */
function newMemory(options: ConstructorParameters<typeof ReflectiveMemory>[0] = {}): ReflectiveMemory {
  return new ReflectiveMemory({ env: baseEnv(), ...options });
}

/** The pinned snapshot of a fresh run of `memory`, optionally restored from `snapshot`. */
function snapshotOf(memory: ReflectiveMemory, snapshot?: JsonObject): JsonObject {
  const session = memory.openRun({ snapshot });
  try {
    return session.snapshot;
  } finally {
    session.close();
  }
}

/** Open a run, await `fn` with it and close it (Python's `with memory.open_run() as session`). */
async function withRun<T>(
  memory: ReflectiveMemory,
  fn: (session: ReflectiveSession) => T | Promise<T>,
  snapshot?: JsonObject,
): Promise<T> {
  const session = memory.openRun({ snapshot });
  try {
    return await fn(session);
  } finally {
    session.close();
  }
}

/** A hand-built reflective mutation attributed to `run-1`. */
function mutation(mutationId: string, action: string, lessonId: string, text?: string): MemoryMutation {
  const payload: Record<string, unknown> = { action, lesson_id: lessonId, source: { run_id: "run-1" } };
  if (text !== undefined) payload.text = text;
  return new MemoryMutation(mutationId, payload);
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

const OLLAMA = { provider: "ollama", model: "qwen" };

test("constructor and protocol are exact", () => {
  for (const value of [true, 1.5, "1"]) {
    expect(() => newMemory({ maxItems: value as never })).toThrow(new TypeError("max_items must be an integer"));
    expect(() => newMemory({ recallLimit: value as never })).toThrow(new TypeError("recall_limit must be an integer"));
  }
  expect(() => newMemory({ recallLimit: 0 })).toThrow(new RangeError("recall_limit must be positive"));
  expect(() => newMemory({ maxItems: 0n })).toThrow(new RangeError("max_items must be positive"));
  expect(() => newMemory({ maxItems: 2n ** 53n })).toThrow(
    new RangeError("max_items must be at most 9007199254740991, got 9007199254740992"),
  );
  expect(() => newMemory({ llmConfig: { provider: "not-a-provider" } })).toThrow(RangeError);
  expect(() => newMemory({ llmConfig: { model: "" } })).toThrow(new RangeError("llm_config.model must be non-empty"));
  expect(() => newMemory({ llmConfig: { model: "  " } })).toThrow(new RangeError("llm_config.model must be non-empty"));
  for (const privateKey of ["api_key", "backend_token", "tenant_id"]) {
    expect(() => newMemory({ llmConfig: { ...OLLAMA, [privateKey]: "private" } })).toThrow(RangeError);
  }
  expect(() => newMemory({ llmConfig: { ...OLLAMA, api_key: "private" } })).toThrow(
    new RangeError(
      "llm_config.api_key is not supported by reflective memory, which records its model configuration",
    ),
  );
  for (const field of ["organization", "project", "headers"]) {
    expect(() => newMemory({ llmConfig: { provider: "openai", model: "m", [field]: "x" } })).toThrow(
      new RangeError(
        `llm_config.${field} is not supported by reflective memory, which records its model configuration`,
      ),
    );
  }
  expect(() => newMemory({ llmConfig: "config" as never })).toThrow(TypeError);
  expect(() => newMemory({ createParent: 1 as never })).toThrow(new TypeError("create_parent must be a bool"));

  const memory = newMemory({ maxItems: 7, recallLimit: 3 });
  expect(isMemory(memory)).toBe(true);
  expect(snapshotOf(memory).config).toEqual({ max_items: 7n, recall_limit: 3n });
  expect(snapshotOf(newMemory({ maxItems: 1.0, recallLimit: 2n })).config).toEqual({ max_items: 1n, recall_limit: 2n });
  expect(snapshotOf(newMemory({ maxItems: null, recallLimit: null })).config).toEqual({
    max_items: 100n,
    recall_limit: 20n,
  });
});

test("snapshot contains only public pinned model configuration", () => {
  const memory = newMemory({ llmConfig: { ...OLLAMA, ollama: { options: { temperature: 0.2 } } } });
  const llmConfig = snapshotOf(memory).llm_config as JsonObject;
  expect(Object.keys(llmConfig).toSorted()).toEqual(["model", "ollama", "provider"]);
  for (const key of ["api_key", "backend_token", "tenant_id"]) expect(Object.hasOwn(llmConfig, key)).toBe(false);
});

test("a number int in a host llm_config block is host intake", () => {
  const memory = newMemory({ llmConfig: { ...OLLAMA, ollama: { options: { num_ctx: 4096 } } } });
  const snapshot = snapshotOf(memory);
  expect(snapshot.llm_config).toEqual({ ...OLLAMA, ollama: { options: { num_ctx: 4096n } } });
  expect(snapshotOf(newMemory(), snapshot)).toEqual(snapshot);
});

test("defaults are read once and recorded snapshot never rereads them", () => {
  let reads = 0;
  const inner = baseEnv({ YIAN_DEFAULT_PROVIDER: "ollama", YIAN_DEFAULT_MODEL: "qwen" });
  const env: Env = (name) => {
    if (name === "YIAN_DEFAULT_PROVIDER") reads += 1;
    return inner(name);
  };
  const first = newMemory({ env, llmConfig: { ollama: { options: { temperature: 0.2 } } } });
  const snapshot = snapshotOf(first);
  expect(reads).toBe(1);
  expect(snapshot.llm_config).toEqual({ ...OLLAMA, ollama: { options: { temperature: 0.2 } } });

  reads = 0;
  expect(snapshotOf(newMemory({ env }), snapshot)).toEqual(snapshot);
  expect(reads).toBe(0);
});

test("empty ambient model is rejected and open session config is pinned", () => {
  const emptyEnv = baseEnv({ YIAN_DEFAULT_PROVIDER: "ollama", YIAN_DEFAULT_MODEL: "" });
  const empty = failure(() => newMemory({ env: emptyEnv }).openRun());
  expect(empty.code).toBe("reflective.config");
  expect(empty.retryable).toBe(false);

  const defaultFailure = new Error("default-resolution-secret");
  const inner = baseEnv();
  const failDefault: Env = (name) => {
    if (name === "YIAN_DEFAULT_PROVIDER") throw defaultFailure;
    return inner(name);
  };
  const defaultError = failure(() => newMemory({ env: failDefault }).openRun());
  expect(defaultError.code).toBe("reflective.config");
  expect(defaultError.message).not.toContain("default-resolution-secret");
  expect(defaultError.cause).toBe(defaultFailure);

  const secretEnv = baseEnv({ YIAN_DEFAULT_PROVIDER: "secret-provider-value", YIAN_DEFAULT_MODEL: "qwen" });
  const configError = failure(() => newMemory({ env: secretEnv }).openRun());
  expect(configError.code).toBe("reflective.config");
  expect(configError.message).not.toContain("secret-provider-value");
  expect(configError.cause).toBeDefined();

  const memory = newMemory({ env: baseEnv({ YIAN_DEFAULT_PROVIDER: "ollama", YIAN_DEFAULT_MODEL: "qwen" }) });
  const session = memory.openRun();
  expect(session.snapshot.llm_config).toEqual(OLLAMA);
  session.close();
});

test("snapshot is defensive and foreign restore is conservative", () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  const session = memory.openRun();
  const snapshot = session.snapshot;
  expect(() => {
    (snapshot as Mutable).namespace = "changed";
  }).toThrow(TypeError);
  const recorded = session.snapshot;
  expect(recorded.namespace).not.toBe("changed");
  session.close();

  const foreign = newMemory();
  snapshotOf(foreign);
  const foreignError = failure(() => foreign.openRun({ snapshot: recorded }));
  expect(foreignError.code).toBe("reflective.invalid_snapshot");
  expect(foreignError.message).toContain("namespace");

  const reset = newMemory();
  reset.reset();
  expect(failure(() => reset.openRun({ snapshot: recorded })).message).toContain("namespace");
});

test("pristine restore keeps recorded limits", async () => {
  const snapshot = snapshotOf(newMemory({ llmConfig: OLLAMA, maxItems: 4, recallLimit: 2 }));
  await withRun(
    newMemory(),
    async (session) => {
      expect(await session.recall()).toBe("");
      expect(session.snapshot).toEqual(snapshot);
    },
    snapshot,
  );
});

test("unset limits adopt recorded ones and explicit ones must match", () => {
  expect(snapshotOf(newMemory({ llmConfig: OLLAMA })).config).toEqual({ max_items: 100n, recall_limit: 20n });
  const recorded = snapshotOf(newMemory({ llmConfig: OLLAMA, maxItems: 4, recallLimit: 2 }));
  expect(snapshotOf(newMemory(), recorded)).toEqual(recorded);
  expect(snapshotOf(newMemory({ maxItems: 4 }), recorded)).toEqual(recorded);
  for (const options of [{ maxItems: 5 }, { recallLimit: 3 }]) {
    const error = failure(() => newMemory(options).openRun({ snapshot: recorded }));
    expect(error.code).toBe("reflective.config_conflict");
    expect(error.message).toBe("configured limits differ from the recorded limits");
    expect(error.retryable).toBe(false);
  }
});

test("snapshot rejects noncanonical scalars and circular json", () => {
  const circular: Mutable = {};
  circular.value = circular;
  expect(() => newMemory({ llmConfig: circular })).toThrow(
    new RangeError("llm_config.value contains a circular JSON value"),
  );

  const memory = newMemory({ llmConfig: OLLAMA });
  const version: Mutable = structuredClone(snapshotOf(memory));
  version.version = true;
  expect(failure(() => newMemory().openRun({ snapshot: version })).code).toBe("reflective.invalid_snapshot");

  const blankModel: Mutable = structuredClone(snapshotOf(memory));
  blankModel.llm_config.model = "   ";
  const error = failure(() => newMemory().openRun({ snapshot: blankModel }));
  expect(error.code).toBe("reflective.invalid_snapshot");
  expect(error.message).toBe("recorded memory snapshot is invalid");
  expect((error.cause as Error).message).toBe("snapshot.llm_config must pin nonblank provider and model");

  for (const [field, value] of [
    ["revision", 2n ** 60n],
    ["epoch", -1n],
    ["version", 1n],
  ] as const) {
    const changed: Mutable = structuredClone(snapshotOf(memory));
    changed[field] = value;
    const changedError = failure(() => newMemory().openRun({ snapshot: changed }));
    expect(changedError.code).toBe("reflective.invalid_snapshot");
    if (field === "revision") {
      expect((changedError.cause as Error).message).toBe("snapshot.revision must be at most 9007199254740991");
    }
  }
});

test("reflective json rejects non utf8 model and snapshot strings", () => {
  const surrogate = "\ud800";
  expect(() => newMemory({ llmConfig: { provider: "ollama", model: surrogate } })).toThrow(
    new RangeError("Unicode surrogate is not valid canonical JSON"),
  );
  const snapshot: Mutable = structuredClone(snapshotOf(newMemory({ llmConfig: OLLAMA })));
  snapshot.namespace = surrogate;
  expect(failure(() => newMemory().openRun({ snapshot })).code).toBe("reflective.invalid_snapshot");
});

test("snapshot rejects excessive nesting as typed invalid snapshot", () => {
  const snapshot: Mutable = structuredClone(snapshotOf(newMemory({ llmConfig: OLLAMA })));
  let nested: Mutable = {};
  for (let i = 0; i < 1_500; i++) nested = { nested };
  snapshot.extra = nested;
  const error = failure(() => newMemory().openRun({ snapshot }));
  expect(error.code).toBe("reflective.invalid_snapshot");
  expect((error.cause as Error).message).toBe("snapshot has invalid fields");
});

for (const [field, value] of [
  ["sandbox", "read-only"],
  ["cwd", "/tmp"],
  ["provider", "codex"],
] as const) {
  test(`old reflective snapshot rejects active retired options: ${field}`, () => {
    const memory = newMemory({ llmConfig: { provider: "openai", model: "old-model" } });
    const old: Mutable = structuredClone(snapshotOf(memory));
    delete old.llm_config._connection;
    Object.assign(old.llm_config, { sandbox: null, cwd: null });
    old.llm_config[field] = value;
    expect(failure(() => newMemory().openRun({ snapshot: old })).code).toBe("reflective.invalid_snapshot");
  });
}

test("same object restore accepts recorded snapshot", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  await withRun(memory, (session) => session.commit("seed", [mutation("seed", "add", "one", "first")]));
  const snapshot = snapshotOf(memory);
  await withRun(
    memory,
    async (restored) => {
      expect(await restored.recall()).toBe("[one] first");
      expect(restored.snapshot).toEqual(snapshot);
    },
    snapshot,
  );
});

test("populated memory rejects foreign snapshot", async () => {
  const snapshot = snapshotOf(newMemory({ llmConfig: OLLAMA }));
  const populated = newMemory({ llmConfig: OLLAMA });
  await withRun(populated, (session) => session.commit("populate", [mutation("populate", "add", "one", "first")]));
  expect(failure(() => populated.openRun({ snapshot })).message).toContain("namespace");
});

test("snapshot rejects malformed foreign and extra lesson sources", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  await withRun(memory, (session) => session.commit("seed", [mutation("seed", "add", "one", "first")]));
  const base = snapshotOf(memory);
  const validSource = { run_id: "run-1" };
  for (const malformed of [
    "not-an-object",
    {},
    { run_id: "" },
    { ...validSource, frame_ref: {} },
    { ...validSource, extra: "source-secret" },
  ]) {
    const snapshot: Mutable = structuredClone(base);
    snapshot.lessons[0].source = malformed;
    const error = failure(() => newMemory().openRun({ snapshot }));
    expect(error.code).toBe("reflective.invalid_snapshot");
    expect(error.message).not.toContain("source-secret");
  }
});

test("session idempotency and closed session operations", async () => {
  const memory = newMemory();
  const session = memory.openRun();
  session.close();
  session.close();
  expect((await rejection(session.recall())).message).toContain("closed");
  expect(failure(() => session.commit("commit", [])).message).toContain("closed");
  // The failed recall released its operation, so the backend isn't busy.
  memory.close();
});

test("sessions of one instance share one lesson list and limit", async () => {
  const memory = newMemory({ llmConfig: OLLAMA, maxItems: 2, recallLimit: 1 });
  await withRun(memory, (session) =>
    session.commit("seed", [
      mutation("one", "add", "one", "first lesson"),
      mutation("two", "add", "two", "second lesson"),
    ]),
  );
  await withRun(memory, async (session) => {
    const snapshot = session.snapshot;
    expect(snapshot.version).toBe(2n);
    expect(Object.keys(snapshot).toSorted()).toEqual([
      "config",
      "epoch",
      "format",
      "lessons",
      "llm_config",
      "namespace",
      "revision",
      "version",
    ]);
    expect(snapshot.config).toEqual({ max_items: 2n, recall_limit: 1n });
    expect((snapshot.lessons as JsonObject[]).map((lesson) => lesson.source)).toEqual([
      { run_id: "run-1" },
      { run_id: "run-1" },
    ]);
    expect(await session.recall()).toBe("[one] first lesson\n[two] second lesson");
    const error = failure(() => session.commit("overflow", [mutation("three", "add", "three", "third")]));
    expect(error.code).toBe("reflective.max_items");
  });
});

test("recall all and context ranking are deterministic", async () => {
  const memory = newMemory({ llmConfig: OLLAMA, recallLimit: 2 });
  await withRun(memory, (session) =>
    session.commit("seed", [
      mutation("b", "add", "b", "red apple"),
      mutation("a", "add", "a", "blue berry"),
      mutation("c", "add", "c", "red berry"),
    ]),
  );
  await withRun(memory, async (session) => {
    expect(await session.recall()).toBe("[b] red apple\n[a] blue berry\n[c] red berry");
    const recalled = await session.recall(new RecallContext("", "RED berry", { tone: "BLUE" }, ["root", "user"]));
    expect(recalled).toBe("[a] blue berry\n[c] red berry");
  });
});

test("subclass controls query rewriting", async () => {
  // TS-only: the async override is awaited.
  class AppleMemory extends ReflectiveMemory {
    override rewriteQuery(): string {
      return "apple";
    }
  }
  class AsyncAppleMemory extends ReflectiveMemory {
    override async rewriteQuery(): Promise<string> {
      await Promise.resolve();
      return "apple";
    }
  }
  for (const Memory of [AppleMemory, AsyncAppleMemory]) {
    const memory = new Memory({ env: baseEnv(), llmConfig: OLLAMA, recallLimit: 1 });
    await withRun(memory, (session) =>
      session.commit("seed", [
        mutation("berry", "add", "berry", "fresh berry"),
        mutation("apple", "add", "apple", "green apple"),
      ]),
    );
    await withRun(memory, async (session) => {
      expect(await session.recall(new RecallContext("", "berry", {}, ["root"]))).toBe("[apple] green apple");
    });
  }
});

test("query rewriting failure is sanitized", async () => {
  const cause = new Error("query-secret");
  class FailingMemory extends ReflectiveMemory {
    override rewriteQuery(): string {
      throw cause;
    }
  }
  const memory = new FailingMemory({ env: baseEnv(), llmConfig: OLLAMA });
  const error = await withRun(memory, (session) =>
    rejection(session.recall(new RecallContext("", "hello", {}, ["root"]))),
  );
  expect(error.code).toBe("reflective.query");
  expect(error.message).not.toContain("query-secret");
  expect(error.cause).toBe(cause);
});

test("recall rejects an invalid context and a non-string query", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  const context = await withRun(memory, (session) => rejection(session.recall({} as never)));
  expect([context.stage, context.code, context.message]).toEqual([
    "recall",
    "reflective.invalid_context",
    "context must be a RecallContext or null",
  ]);

  class NumberMemory extends ReflectiveMemory {
    override rewriteQuery(): string {
      return 1 as never;
    }
  }
  const numeric = new NumberMemory({ env: baseEnv(), llmConfig: OLLAMA });
  const query = await withRun(numeric, (session) =>
    rejection(session.recall(new RecallContext("", "hello", {}, ["root"]))),
  );
  expect([query.stage, query.code, query.message]).toEqual([
    "recall",
    "reflective.query",
    "rewriteQuery() must return a string",
  ]);
  // The failed recalls released their operations, so the backends aren't busy.
  memory.close();
  numeric.close();
});

test("commit is atomic idempotent and empty batches advance revision", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  const add = [mutation("one", "add", "one", "first")];
  await withRun(memory, (session) => {
    session.commit("add", add);
    session.commit("add", add);
    expect(failure(() => session.commit("add", [])).message).toContain("collision");
  });
  const before = snapshotOf(memory).revision as bigint;
  await withRun(memory, (session) => session.commit("empty", []));
  expect(snapshotOf(memory).revision).toBe(before + 1n);
});

test("revise and remove are learned and committed", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  await withRun(memory, (session) => session.commit("add", [mutation("add", "add", "one", "first")]));
  await withRun(memory, (session) => session.commit("revise", [mutation("revise", "revise", "one", "revised")]));
  await withRun(memory, async (session) => expect(await session.recall()).toBe("[one] revised"));
  await withRun(memory, (session) => session.commit("remove", [mutation("remove", "remove", "one")]));
  await withRun(memory, async (session) => expect(await session.recall()).toBe(""));
});

test("learn and commit enforce final max items (commit half)", async () => {
  const memory = newMemory({ llmConfig: OLLAMA, maxItems: 1 });
  await withRun(memory, (session) => session.commit("seed", [mutation("seed", "add", "old", "old lesson")]));
  await withRun(memory, (session) =>
    session.commit("replace", [mutation("new", "add", "new", "new lesson"), mutation("old", "remove", "old")]),
  );
  await withRun(memory, async (session) => expect(await session.recall()).toBe("[new] new lesson"));

  const empty = newMemory({ llmConfig: OLLAMA, maxItems: 1 });
  const error = await withRun(empty, async (session) => {
    const commitError = failure(() =>
      session.commit("too-many", [mutation("two", "add", "two", "two"), mutation("three", "add", "three", "three")]),
    );
    expect(await session.recall()).toBe("");
    expect(session.snapshot.revision).toBe(0n);
    return commitError;
  });
  expect([error.code, error.message]).toEqual(["reflective.max_items", "mutation batch exceeds max_items"]);
});

test("multi mutation commit failure has no partial effect", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  const error = await withRun(memory, async (session) => {
    const commitError = failure(() =>
      session.commit("atomic", [
        mutation("left", "add", "one", "first"),
        mutation("right", "revise", "missing", "bad"),
      ]),
    );
    expect(await session.recall()).toBe("");
    expect(session.snapshot.revision).toBe(0n);
    return commitError;
  });
  expect(error.code).toBe("reflective.invalid_mutation");
  await withRun(memory, async (session) => expect(await session.recall()).toBe(""));
});

test("concurrent same generation commits have one winner", () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  const left = memory.openRun();
  const right = memory.openRun();
  left.commit("left", [mutation("one", "add", "one", "first")]);
  const error = failure(() => right.commit("right", [mutation("two", "add", "two", "second")]));
  expect([error.stage, error.code, error.message, error.retryable]).toEqual([
    "commit",
    "reflective.stale_generation",
    "memory session is stale",
    false,
  ]);
  left.close();
  right.close();
});

test("reset preserves commit ledger ordering and old recall", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  const add = [mutation("one", "add", "one", "first")];
  await withRun(memory, (seed) => seed.commit("applied", add));
  const old = memory.openRun();
  expect(await old.recall()).toBe("[one] first");
  memory.reset();

  expect(await old.recall()).toBe("[one] first");
  old.commit("applied", add);
  expect(failure(() => old.commit("applied", [])).message).toContain("collision");
  expect(failure(() => old.commit("not-applied", add)).message).toContain("stale");
  await withRun(memory, async (fresh) => expect(await fresh.recall()).toBe(""));
  old.close();
});

test("commit rejects invalid batches, commit ids and mutations with Python's messages", async () => {
  const memory = newMemory({ llmConfig: OLLAMA });
  await withRun(memory, (session) => session.commit("seed", [mutation("seed", "add", "kept", "kept lesson")]));
  const source = { run_id: "run-1" };
  const rows: [string, unknown, unknown, string][] = [
    ["reflective.invalid_commit_id", "", [], "commit id must be a nonempty string"],
    ["reflective.invalid_commit_id", 1, [], "commit id must be a nonempty string"],
    ["reflective.invalid_batch", "c", {}, "mutations must be an exact array"],
    ["reflective.invalid_batch", "c", [{}], "mutation batch contains an invalid value"],
    [
      "reflective.invalid_batch",
      "c",
      [mutation("m", "add", "a", "x"), mutation("m", "add", "b", "y")],
      "mutation ids must be unique within a batch",
    ],
    ["reflective.invalid_mutation", "c", [new MemoryMutation("m", [])], "mutation payload must be an object"],
    [
      "reflective.invalid_mutation",
      "c",
      [new MemoryMutation("m", { action: "add", lesson_id: "a" })],
      "mutation payload has invalid fields",
    ],
    [
      "reflective.invalid_mutation",
      "c",
      [new MemoryMutation("m", { action: "add", lesson_id: "a", source, text: "x", extra: 1 })],
      "mutation payload has invalid fields",
    ],
    ["reflective.invalid_mutation", "c", [mutation("m", "merge", "a", "x")], "mutation action is invalid"],
    [
      "reflective.invalid_mutation",
      "c",
      [new MemoryMutation("m", { action: "add", lesson_id: "", source, text: "x" })],
      "mutation identity or source is invalid",
    ],
    [
      "reflective.invalid_mutation",
      "c",
      [new MemoryMutation("m", { action: "add", lesson_id: "a", source: { run_id: "r", extra: 1 }, text: "x" })],
      "mutation identity or source is invalid",
    ],
    ["reflective.invalid_mutation", "c", [mutation("m", "add", "a")], "mutation text is invalid"],
    ["reflective.invalid_mutation", "c", [mutation("m", "add", "a", "  ")], "text must not be blank"],
    ["reflective.invalid_mutation", "c", [mutation("m", "remove", "kept", "x")], "remove mutations cannot carry text"],
    [
      "reflective.invalid_mutation",
      "c",
      [mutation("m1", "add", "a", "x"), mutation("m2", "revise", "a", "y")],
      "a batch cannot target one lesson more than once",
    ],
    [
      "reflective.invalid_mutation",
      "c",
      [mutation("m", "add", "kept", "x")],
      "add mutation targets an existing lesson",
    ],
    [
      "reflective.invalid_mutation",
      "c",
      [mutation("m", "revise", "a", "x")],
      "revise mutation targets a missing lesson",
    ],
    ["reflective.invalid_mutation", "c", [mutation("m", "remove", "a")], "remove mutation targets a missing lesson"],
  ];
  await withRun(memory, async (session) => {
    for (const [code, commitId, mutations, message] of rows) {
      const error = failure(() => session.commit(commitId as never, mutations as never));
      expect([error.stage, error.code, error.message, error.retryable]).toEqual(["commit", code, message, false]);
    }
    expect(await session.recall()).toBe("[kept] kept lesson");
    expect(session.snapshot.revision).toBe(1n);
  });
});

test("a host mutation payload holding -0 cannot be encoded canonically", async () => {
  await withRun(newMemory({ llmConfig: OLLAMA }), (session) => {
    const zero = new MemoryMutation("m", { action: "add", lesson_id: "a", source: { run_id: "r" }, text: "x", n: -0 });
    const error = failure(() => session.commit("zero", [zero]));
    expect([error.code, error.message]).toEqual([
      "reflective.invalid_batch",
      "mutation batch cannot be encoded canonically",
    ]);
    expect(error.cause).toBeInstanceOf(TypeError);
  });
});

test("a re-entrant close from inside an async rewriteQuery is busy", async () => {
  let closeError: unknown = null;
  class ClosingQueryMemory extends ReflectiveMemory {
    override async rewriteQuery(): Promise<string> {
      await Promise.resolve();
      try {
        this.close();
      } catch (error) {
        closeError = error;
      }
      return "query";
    }
  }
  const memory = new ClosingQueryMemory({ env: baseEnv(), llmConfig: OLLAMA });
  await withRun(memory, async (session) => {
    expect(await session.recall(new RecallContext("", "", {}, ["root"]))).toBe("");
  });
  expect(closeError).toBeInstanceOf(MemoryOperationError);
  const busy = closeError as MemoryOperationError;
  expect([busy.stage, busy.code, busy.message, busy.retryable]).toEqual([
    "close",
    "reflective.busy",
    "memory backend is active",
    true,
  ]);
  memory.close();
  expect(failure(() => memory.openRun()).code).toBe("reflective.closed");
});
