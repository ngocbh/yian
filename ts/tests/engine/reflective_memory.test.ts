/**
 * Process-local reflective-memory behavior: a port of `tests/engine/test_reflective_memory.py`.
 *
 * Ported: `test_constructor_and_protocol_are_exact`,
 * `test_snapshot_contains_only_public_pinned_model_configuration`,
 * `test_defaults_are_read_once_and_recorded_snapshot_never_rereads_them`,
 * `test_empty_ambient_model_is_rejected_and_open_session_config_is_pinned` (the open half),
 * `test_snapshot_is_defensive_and_foreign_restore_is_conservative`,
 * `test_pristine_restore_keeps_recorded_limits` (the snapshot half),
 * `test_unset_limits_adopt_recorded_ones_and_explicit_ones_must_match`,
 * `test_snapshot_rejects_noncanonical_scalars_and_circular_json`,
 * `test_reflective_json_rejects_non_utf8_model_and_snapshot_strings`,
 * `test_snapshot_rejects_excessive_nesting_as_typed_invalid_snapshot` and
 * `test_old_reflective_snapshot_rejects_active_retired_options`. Also ported: the `organization`,
 * `project` and `headers` rows of `tests/engine/test_run_connection_settings.py::
 * test_reflective_memory_refuses_connection_settings`, in the constructor test.
 *
 * TS-only: `bigint` limits, `null` limits as unset, a limit one past 2^53 − 1 refused, a `number` int
 * in a host `llm_config` block accepted (host intake gives a `bigint`); in the constructor, a blank
 * `model`, the exact `api_key` message, a non-object `llmConfig` (`TypeError`) and `createParent: 1`
 * (`TypeError`); a snapshot whose `revision` is `2n ** 60n` (beyond 2^53 − 1, with its cause
 * message), whose `epoch` is `-1n` or whose `version` is `1n` is `reflective.invalid_snapshot`.
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
 * - JSON ints written into snapshots are `bigint`s; `with` blocks are `try`/`finally` around
 *   `close()`.
 *
 * Later (in this PR): `test_reflective_episode_material_retains_the_physical_execution_sequence`,
 * `test_same_object_restore_accepts_recorded_snapshot`, `test_populated_memory_rejects_foreign_snapshot`,
 * `test_snapshot_rejects_malformed_foreign_and_extra_lesson_sources`,
 * `test_session_idempotency_and_closed_session_operations`,
 * `test_sessions_of_one_instance_share_one_lesson_list_and_limit`,
 * `test_recall_all_and_context_ranking_are_deterministic`, `test_subclass_controls_query_rewriting`,
 * `test_query_rewriting_failure_is_sanitized`,
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
 * `test_commit_is_atomic_idempotent_and_empty_batches_advance_revision`,
 * `test_revise_and_remove_are_learned_and_committed`, `test_learn_and_commit_enforce_final_max_items`,
 * `test_multi_mutation_commit_failure_has_no_partial_effect`,
 * `test_concurrent_same_generation_commits_have_one_winner`,
 * `test_reset_preserves_commit_ledger_ordering_and_old_recall`,
 * `test_reflective_memory_implements_protocol_and_imports_only_lower_layers`,
 * `test_session_reflects_under_the_memory_key`, and the learn half of
 * `test_empty_ambient_model_is_rejected_and_open_session_config_is_pinned` and the recall half of
 * `test_pristine_restore_keeps_recorded_limits`.
 *
 * Not ported: `test_old_api_snapshot_keeps_digest_and_api_semantics` (TS reads no Python-era
 * snapshot, so there is no retired-key branch).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { BUILTIN, type Env, install } from "../../src/defaults.ts";
import { isMemory, type JsonObject, MemoryMutation, MemoryOperationError } from "../../src/memory/contracts.ts";
import { ReflectiveMemory } from "../../src/memory/reflective.ts";

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

test("pristine restore keeps recorded limits", () => {
  const snapshot = snapshotOf(newMemory({ llmConfig: OLLAMA, maxItems: 4, recallLimit: 2 }));
  expect(snapshotOf(newMemory(), snapshot)).toEqual(snapshot);
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
