/**
 * Memory identity values, the memory JSON snapshot, the learning result carriers, the recall
 * context and the host memory interfaces, from `tests/engine/test_memory_contracts.py`:
 * `test_identity_values_are_frozen_structured_products`, `test_memory_scope_has_no_boundary`,
 * `test_identity_values_reject_ambiguous_paths_and_indices`,
 * `test_scope_occurrence_path_must_strictly_contain_the_node`,
 * `test_memory_decls_preserve_origin_and_isolate_canonical_json`,
 * `test_json_contracts_reject_non_exact_or_non_finite_values`, `test_json_contracts_reject_cycles`,
 * `test_scope_carries_a_root_relative_path_and_memory_keys`, the declaration rows of
 * `test_memory_strings_must_have_a_canonical_utf8_representation`,
 * `test_json_scalar_identity_preserves_exact_types_at_every_depth`, the recall context and mutation
 * parts of `test_context_episode_and_mutation_keep_json_snapshots`, the `MemoryMutation`,
 * `MemoryProposal`, `LearningFailure` and `BackwardFailure` rows of
 * `test_episodes_contexts_mutations_and_failures_name_the_memory_key`,
 * `test_learning_result_keeps_primary_independent_failure_attribution`,
 * `test_policy_commit_outcome_keeps_protocol_identity_and_backend_hints`,
 * `test_learning_completion_is_independent_of_retryability_and_policy_confirmation`,
 * `test_policy_commit_outcome_rejects_inconsistent_optional_fields`,
 * `test_policy_commit_outcome_requires_exact_valid_unicode_identities`, the rows without forgery of
 * `test_learning_result_revalidates_exact_nested_policy_carriers`,
 * `test_memory_errors_expose_typed_fields_and_origin`,
 * `test_protocols_accept_narrow_structural_implementations` (with a synchronous and an async fake),
 * `test_session_protocol_recalls_and_learns_without_frames`,
 * `test_memory_recall_error_preserves_an_explicit_exception_cause`, the scope row of
 * `test_nested_memory_carriers_reject_forged_subclass_values` and the subclass row of
 * `test_memory_origins_revalidate_subclass_keys_without_losing_identity`, the memory package,
 * `contracts.ts` and `runtime/contracts.ts` rows of `test_frame_types_are_gone` and
 * `test_importing_memory_contracts_does_not_load_heavy_backends`; the episode and learn context
 * parts of `test_context_episode_and_mutation_keep_json_snapshots`,
 * `test_episode_status_enforces_output_or_failure_shape`, the rows 0 and 17 of
 * `test_episode_execution_sequence_is_optional_and_survives_context_copy`,
 * `test_episode_rejects_nonphysical_execution_sequences`,
 * `test_learn_context_requires_episodes_of_its_run`, the episode and context rows of
 * `test_episodes_contexts_mutations_and_failures_name_the_memory_key`,
 * `test_freeze_still_detects_cycles`, `test_freeze_shares_a_repeated_subtree`,
 * `test_learn_context_equal_feedback` and `test_propose_learning_shares_canonical_feedback` (as "the
 * learner receives the exact `LearnContext`"); plus JavaScript's own values (a `"__proto__"` key, a
 * null-prototype object, a repeated subtree, an array hole), the value nesting limit, and the
 * depth-first order of the first error. 100 000-level arrays and objects are copied through a
 * snapshot envelope, a mutation and learn feedback (compared as log JSON text, since Bun's
 * `toEqual` overflows the stack at that depth), and an error deep inside one names its full path.
 * It also covers the occurrence nesting order, the operation failure's field checks (Python pins
 * those through the learning outcome and backward records), `validateMutations`, a mutation's id
 * and payload checks (a payload deeper than the value nesting limit is kept), each learning
 * carrier's field checks in Python's order, the recall context's field checks, its value number
 * rule at any depth, and the recall error's field checks.
 *
 * TS-only (episodes and learn contexts): one row per `MemoryEpisode` and `LearnContext`
 * constructor message with its class, in Python's check order; the episode JSON and the learn
 * feedback keep `±Infinity`, read `-0` back as `0` and refuse NaN; `LearnContext.fromFrozen` stores
 * the given feedback (`toBe`) and checks run, memory and episodes; `freezeFeedback` names items from
 * `firstIndex` and shares a subtree across calls through its memo; each memory `same*` comparison
 * (`sameMemoryScope`, `sameMemoryOperationFailure`, `sameMemoryMutation`,
 * `sameMemorySnapshotEnvelope`, `sameMemorySpec`, `sameLearningFailure`, `samePolicyCommitOutcome`,
 * `sameLearningResult`) is true for separately built equal values and false for each differing
 * field; `sameMemorySnapshotEnvelope` also tells `1n` from `1` and `1` from `true` at depth; a
 * setup error keeps an explicit `cause` and its origin, and has no `cause` without one.
 *
 * Changed (episodes and learn contexts): the episode and the context hold deep-frozen JSON, so a
 * reader's edit of `inputs`, an interaction value or `feedback` throws `TypeError` where Python
 * edits a private copy. Python's keyword arguments are an options object (`{ output, failure,
 * interactions, baseSystemPrompt, prompt, memorySections, executionSeq }`); the 11-argument
 * positional "legacy" row passes `{ output: "answer" }`. The non-physical sequence rows are
 * `false, true, -1, 1.5, "1", 1n` (`1.0` is `1` in JS), each with its class and message.
 * `test_freeze_shares_a_repeated_subtree` checks the read feedback (`feedback[0].a` is
 * `feedback[0].b`), since the stored snapshot is what `feedback` returns. Python's dataclass `==`
 * of two contexts is `toEqual`; `copy.deepcopy` is `structuredClone`. `proposeLearning` doesn't
 * rebuild a context, so the canonical-feedback test checks that the learner receives the very
 * context and its very `feedback` (`toBe`, where Python checks the private canonical feedback is
 * shared); its rebuilt non-canonical scalar row has no counterpart (a frozen context can't be
 * forged).
 *
 * Changed: the JSON tests build a custom-type `MemoryDecl`, as Python's `_decl` does. The non-exact
 * rows are JavaScript's (a `Map`, a `Date`, an own symbol key, a class instance, an array subclass
 * for `_ListSubclass`, a top-level class instance for `_DictSubclass`), each with its exact class
 * and message. The scalar identity test checks that `true`, `1n` and `1` keep their exact JS types
 * at every depth of the snapshot. A memory config nests at most `maxValueNesting` levels (memory
 * JSON nesting doesn't exist here), with that limit's message. The ambiguous-index rows add `1.5`,
 * `1n` and a `"MAP_ITEM"` kind; the non-array paths are `Set`s (Python passes lists). A scope's
 * lack of a boundary is checked on the module's exports and the scope's fields (TS takes extra
 * constructor arguments silently). The scope row of the forged-carrier test passes a plain object
 * shaped like a segment. The memory protocols are checked on plain-object fakes: `Memory` with
 * `isMemory` (Python's `isinstance` on a runtime-checkable protocol), `MemorySession` by its type,
 * and its missing `openFrame` with `@ts-expect-error` and `in`. A host method may return a promise,
 * so the protocol test runs a synchronous and an async fake and awaits each call; a session has no
 * context-manager methods, so the test calls `close()`. The recall context's `inputs` follow the
 * yian value number rule and are a deep-frozen snapshot, so editing them throws `TypeError` (Python
 * returns a fresh copy). `MemoryRecallError` takes its cause as `{ cause }`, read back as
 * `error.cause` (Python's `raise ... from` and `__cause__`). The fake's `snapshot` is frozen, so a
 * reader's edit of it throws `TypeError` and leaves the snapshot unchanged (Python's fake returns a
 * copy). The backend-import test reads `memory/index.ts` and checks that it imports only
 * `./contracts.ts` (the layer-direction test pins `contracts.ts` to `diagnostics` and `defaults`),
 * where Python imports the package in a fresh interpreter. Python's `ValueError` is `RangeError`; a
 * frozen field throws `TypeError` where Python raises `FrozenInstanceError`. A mutation's payload
 * is its deep-frozen snapshot, so editing it throws `TypeError` (Python returns a fresh copy);
 * Python's int payloads are `bigint`s. The learning carriers keep their nested carriers by
 * reference: where Python asserts a copied policy `is not` the one passed in, the test checks that
 * it is the same frozen value. The policy outcome's optional fields are an options object
 * (`{ commitId, failure }`), and the learning result's arrays are `{ failures, policies }`. The
 * identity rows check only the error class, as Python does.
 *
 * Not ported: the tuple and bytes rows and the list-for-tuple path row (TS has only arrays and no
 * bytes in a config); `test_memory_json_rejects_nesting_beyond_the_memory_limit` (its carriers
 * aren't ported; the nesting test below replaces it for declarations); the forged rows of
 * `test_nested_memory_carriers_reject_forged_subclass_values` and
 * `test_memory_origins_revalidate_subclass_keys_without_losing_identity` (a JS subclass can't skip
 * its base constructor, so no carrier holds an unvalidated segment or origin);
 * `test_memory_operation_error_survives_copy_and_pickle` (no `copy` or `pickle`); dataclass `==`
 * and `hash` (Bun's `toEqual` replaces equality); the `MemorySession` row of
 * `test_frame_types_are_gone` (`MemorySession` is an interface, with no runtime members; the
 * session protocol test checks a session's missing `openFrame`); the forged rows of
 * `test_learning_result_revalidates_exact_nested_policy_carriers` (Python's `object.__setattr__` on
 * a frozen dataclass; a TS carrier is frozen by its constructor, so none holds an unchecked
 * `commitId` or `retryable`); `test_freeze_memo_keeps_the_nesting_limit` (TS has no memory nesting
 * limit); the forged row of `test_episode_execution_sequence_is_optional_and_survives_context_copy`
 * and the episode and context rows of `test_nested_memory_carriers_reject_forged_subclass_values`
 * (a frozen TS carrier can't be forged, so a context keeps its checked episodes by reference);
 * `test_episode_rejects_integer_subclass_execution_sequence` (no integer subclasses in JS);
 * `test_propose_learning_parity_on_deleted_slots` (`object.__delattr__` and
 * `LearnContext._revalidated`, which TS doesn't need) and
 * `test_propose_learning_rebuild_of_forged_feedback` (`object.__setattr__` on a frozen context; a
 * TS context is frozen by its constructor, so none holds forged feedback).
 */

import { afterEach, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import * as contracts from "../../src/memory/contracts.ts";
import {
  DeclSite,
  freezeFeedback,
  isMemory,
  type JsonObject,
  type JsonValue,
  LearnContext,
  LearningFailure,
  LearningResult,
  type Memory,
  MemoryDecl,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryOperationError,
  MemoryOperationFailure,
  MemoryRecallError,
  MemoryScope,
  MemorySection,
  type MemorySession,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
  ModelTextInteraction,
  ModelValueInteraction,
  PolicyCommitOutcome,
  type PolicyCommitStatus,
  RecallContext,
  sameLearningFailure,
  sameLearningResult,
  sameMemoryMutation,
  sameMemoryOperationFailure,
  sameMemoryScope,
  sameMemorySnapshotEnvelope,
  sameMemorySpec,
  samePolicyCommitOutcome,
  validateMutations,
} from "../../src/memory/contracts.ts";
import * as memoryPackage from "../../src/memory/index.ts";
import * as runtimeContracts from "../../src/runtime/contracts.ts";
import { BackwardFailure, MemoryProposal } from "../../src/runtime/contracts.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import { logJson } from "../../src/typesys/json.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { importStatements } from "./source_scan.ts";

/** Build a custom-type declaration, so the built-in key allowlist does not apply. */
function decl(config: unknown, origin: OriginRef | null = null): MemoryDecl {
  return new MemoryDecl("notes", "custom", config, new DeclSite([], ["root"]), origin);
}

/** Build a reflective declaration, whose built-in rule tells the scalar kinds apart. */
function reflective(config: unknown): MemoryDecl {
  return new MemoryDecl("notes", "reflective", config, new DeclSite([], ["root"]));
}

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).message).toBe(message);
}

afterEach(() => forget());

class Point {
  x = 1n;
}

class Items extends Array<unknown> {}

// test_json_contracts_reject_non_exact_or_non_finite_values
test.each<[string, unknown, typeof TypeError | typeof RangeError, string]>([
  ["a Map in an array", { array: [new Map()] }, TypeError, "config.array[0] must contain exact JSON values"],
  ["a Date", { date: new Date(0) }, TypeError, "config.date must contain exact JSON values"],
  ["NaN", { nan: Number.NaN }, RangeError, "config.nan JSON number must be finite"],
  ["Infinity", { infinity: Number.POSITIVE_INFINITY }, RangeError, "config.infinity JSON number must be finite"],
  ["a symbol key", { [Symbol("key")]: 1n }, TypeError, "config JSON object keys must be strings"],
  ["a class instance", { point: new Point() }, TypeError, "config.point must contain exact JSON values"],
  ["an array subclass", { a: Items.from([1n]) }, TypeError, "config.a must contain exact JSON values"],
  ["a top-level class instance", new Point(), TypeError, "config must be an exact JSON object"],
  ["a top-level array", [1n], TypeError, "config must be an exact JSON object"],
])("JSON contracts reject %s", (_name, config, cls, message) => {
  expectError(() => decl(config), cls, message);
});

test("a null-prototype object is a JSON object", () => {
  decl(Object.assign(Object.create(null), { a: 1n }));
  decl({ a: Object.assign(Object.create(null), { b: 1n }) });
});

// test_json_contracts_reject_cycles
test("JSON contracts reject cycles", () => {
  const config: Record<string, unknown> = {};
  config.self = config;
  expectError(() => decl(config), RangeError, "config.self contains a circular JSON value");
});

test("a subtree reached twice without a cycle passes", () => {
  const s = { x: 1n };
  decl({ a: s, b: s });
  decl({ a: [s, s] });
});

test("an array hole is refused", () => {
  expectError(() => decl({ a: [1n, , 2n] }), TypeError, "config.a[1] must contain exact JSON values");
});

// test_memory_strings_must_have_a_canonical_utf8_representation
test.each<[string, unknown]>([
  ["a value", { text: "\ud800" }],
  ["a key", { "\ud800": "text" }],
])("memory strings reject a lone surrogate in %s", (_name, config) => {
  expectError(() => decl(config), RangeError, "Unicode surrogate is not valid canonical JSON");
});

// test_json_scalar_identity_preserves_exact_types_at_every_depth (the declaration row)
test("JSON scalars keep their exact types at every depth", () => {
  for (const [value, type] of [
    [true, "boolean"],
    [1n, "bigint"],
    [1, "number"],
  ] as const) {
    const nested = decl({ nested: [value, { value }] }).config.nested as readonly [unknown, { value: unknown }];
    expect(typeof nested[0]).toBe(type);
    expect(typeof nested[1].value).toBe(type);
    expect(nested[0]).toBe(value);
    expect(nested[1].value).toBe(value);
  }
  reflective({ max_items: 1n });
  expect(() => reflective({ max_items: true })).toThrow("got true");
  expect(() => reflective({ max_items: 1 })).toThrow("got 1");
});

test("a __proto__ key stays a data key", () => {
  const config = { ["__proto__"]: 1n };
  expect(Object.hasOwn(config, "__proto__")).toBe(true);
  const snapshot = decl(config).config;
  expect(Object.getOwnPropertyDescriptor(snapshot, "__proto__")).toEqual({
    value: 1n,
    writable: false,
    enumerable: true,
    configurable: false,
  });
  expect(Object.getPrototypeOf(snapshot)).toBe(Object.prototype);
  // The snapshot's key list holds it, so a built-in type reports it as unknown.
  expect(() => new MemoryDecl("notes", "episodic", config, new DeclSite([], ["root"]))).toThrow(
    'unknown key(s) ["__proto__"]',
  );
});

test("a memory config nests at most maxValueNesting levels", () => {
  install({ ...BUILTIN, maxValueNesting: 3 });
  decl({ a: { a: {} } });
  decl({ a: [[]] });
  expectError(
    () => decl({ a: { a: { a: {} } } }),
    RangeError,
    "config.a.a.a nests deeper than 3 levels, the limit for a yian value (each record and list is a level)",
  );
});

test("the first error is the depth-first one, keys checked before any child", () => {
  expectError(
    () => decl({ a: { x: new Map() }, b: new Date(0) }),
    TypeError,
    "config.a.x must contain exact JSON values",
  );
  expectError(
    () => decl({ a: [new Map()], "\ud800": 1n }),
    RangeError,
    "Unicode surrogate is not valid canonical JSON",
  );
  expectError(
    () => decl({ b: [1n, new Map()], a: [Number.NaN] }),
    RangeError,
    "config.a[0] JSON number must be finite",
  );
});

const DEEP = 100_000;

/** Return `leaf` wrapped in `DEEP` arrays, or one-key objects under `key`, built with a loop. */
function deepValue(leaf: unknown, key: string | null): unknown {
  let value = leaf;
  for (let i = 0; i < DEEP; i++) value = key === null ? [value] : { [key]: value };
  return value;
}

// Compared as log JSON text: Bun's `toEqual` overflows the stack at this depth, and `===` keeps a
// failure from printing the whole 600 KB text.
test.each([
  ["array", null],
  ["object", "a"],
] as const)("a 100 000-level %s freezes through the memory carriers", (_kind, key) => {
  const value = { deep: deepValue(1n, key) };
  const text = logJson(value);
  expect(logJson(new MemorySnapshotEnvelope("notes", value).snapshot) === text).toBe(true);
  expect(logJson(new MemoryMutation("m", value).payload) === text).toBe(true);
  expect(logJson(new LearnContext("run", "notes", [], [value]).feedback[0]!) === text).toBe(true);
});

test("an error deep inside a 100 000-level value names its full path", () => {
  let value: unknown = new Map();
  const steps: string[] = [];
  for (let i = 0; i < DEEP; i++) {
    value = i % 2 === 0 ? [value] : { b: value };
    steps.push(i % 2 === 0 ? "[0]" : ".b");
  }
  const path = `payload${steps.reverse().join("")}`;
  expectError(() => new MemoryMutation("m", value), TypeError, `${path} must contain exact JSON values`);
});

// test_identity_values_are_frozen_structured_products
test("identity values are frozen structured products", () => {
  const segment = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["child"], 2);
  const scope = new MemoryScope(["child", "agent"], { memories: ["outer", "local"], occurrencePath: [segment] });
  expect(scope.memories).toEqual(["outer", "local"]);
  expect(scope.occurrencePath).toEqual([segment]);
  expect(scope.occurrencePath[0]).toBe(segment);
  expect(() => {
    (scope as { memories: readonly string[] }).memories = ["other"];
  }).toThrow(TypeError);
  expect(Object.isFrozen(scope.memories)).toBe(true);
  expect(Object.isFrozen(scope.occurrencePath)).toBe(true);
  expect(Object.isFrozen(segment)).toBe(true);
  expect(Object.isFrozen(segment.staticPath)).toBe(true);
});

// test_memory_scope_has_no_boundary
test("a memory scope has no boundary", () => {
  expect("MemoryBoundary" in contracts).toBe(false);
  expect("MemoryBoundaryKind" in contracts).toBe(false);
  expect(Object.keys(new MemoryScope(["root"]))).toEqual(["staticPath", "memories", "occurrencePath"]);
});

// test_identity_values_reject_ambiguous_paths_and_indices
test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string]>([
  [
    "an empty segment path",
    () => new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, [], 0),
    RangeError,
    "static_path must be non-empty",
  ],
  [
    "a boolean index",
    () => new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["root"], true as never),
    TypeError,
    "index must be an integer",
  ],
  [
    "a fractional index",
    () => new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["root"], 1.5),
    TypeError,
    "index must be an integer",
  ],
  [
    "a bigint index",
    () => new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["root"], 1n as never),
    TypeError,
    "index must be an integer",
  ],
  [
    "an index beyond 2^53",
    () => new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["root"], 2 ** 53),
    TypeError,
    "index must be an integer",
  ],
  [
    "a negative index",
    () => new MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ["root"], -1),
    RangeError,
    "index must be non-negative",
  ],
  [
    "a kind given by its member name",
    () => new MemoryOccurrenceSegment("MAP_ITEM" as never, ["root"], 0),
    TypeError,
    "kind must be a MemoryOccurrenceKind",
  ],
  [
    "an empty scope path component",
    () => new MemoryScope([""]),
    RangeError,
    "static_path components must be non-empty",
  ],
  [
    "a non-array occurrence path",
    () => new MemoryScope(["root"], { occurrencePath: new Set() as never }),
    TypeError,
    "occurrence_path must be an array",
  ],
  ["repeated keys", () => new MemoryScope(["root"], { memories: ["a", "a"] }), RangeError, "memories must be unique"],
])("identity values reject %s", (_name, factory, cls, message) => {
  expectError(factory, cls, message);
});

// test_scope_occurrence_path_must_strictly_contain_the_node
test.each([[["other"]], [["child", "agent"]], [["child", "agent", "inner"]]])(
  "an occurrence segment at %p doesn't contain the node",
  (segmentPath) => {
    const segment = new MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, segmentPath, 0);
    expectError(
      () => new MemoryScope(["child", "agent"], { occurrencePath: [segment] }),
      RangeError,
      "the innermost occurrence segment must contain the scoped node",
    );
  },
);

test("occurrence segments follow lexical nesting order", () => {
  const outer = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["a"], 0);
  const inner = new MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ["a", "b"], 1);
  const sibling = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["c"], 0);
  expect(new MemoryScope(["a", "b", "x"], { occurrencePath: [outer, inner] }).occurrencePath).toEqual([outer, inner]);
  const message = "occurrence segments must follow lexical nesting order";
  expectError(() => new MemoryScope(["a", "b", "x"], { occurrencePath: [inner, outer] }), RangeError, message);
  expectError(() => new MemoryScope(["a", "x"], { occurrencePath: [outer, outer] }), RangeError, message);
  expectError(() => new MemoryScope(["c", "x"], { occurrencePath: [outer, sibling] }), RangeError, message);
});

// test_nested_memory_carriers_reject_forged_subclass_values (the scope row)
test("a scope refuses an occurrence item that isn't a segment", () => {
  const shaped = { kind: MemoryOccurrenceKind.MAP_ITEM, staticPath: ["root", "mapped"], index: -1 };
  expectError(
    () => new MemoryScope(["mapped", "x"], { occurrencePath: [shaped] as never }),
    TypeError,
    "occurrence_path items must be MemoryOccurrenceSegment values",
  );
});

// test_memory_decls_preserve_origin_and_isolate_canonical_json
test("a declaration keeps its origin and isolates its canonical JSON", () => {
  const origin = new OriginRef("decl");
  const source = { z: [{ enabled: true }], a: { threshold: 0.5 } };
  const declaration = decl(source, origin);
  source.z[0]!.enabled = false;
  expect(() => {
    (declaration.config.a as { threshold: number }).threshold = 1.0;
  }).toThrow(TypeError);
  expect(declaration.name).toBe("notes");
  expect(declaration.origin).toBe(origin);
  expect(declaration.config).toEqual({ a: { threshold: 0.5 }, z: [{ enabled: true }] });
  expect(Object.keys(declaration.config)).toEqual(["a", "z"]);
  expect(declaration).toEqual(decl({ a: { threshold: 0.5 }, z: [{ enabled: true }] }, origin));
});

// test_memory_origins_revalidate_subclass_keys_without_losing_identity (the subclass row)
test("a declaration keeps an OriginRef subclass by identity", () => {
  class SubOrigin extends OriginRef {}
  const valid = new SubOrigin("valid");
  expect(decl({}, valid).origin).toBe(valid);
});

// test_memory_errors_expose_typed_fields_and_origin
test("a setup error exposes its origin", () => {
  const origin = new OriginRef("memory-policy");
  const setup = new MemorySetupError("unknown policy", { origin });
  expect(setup).toBeInstanceOf(RangeError);
  expect(setup.name).toBe("MemorySetupError");
  expect(setup.message).toBe("unknown policy");
  expect(setup.origin).toBe(origin);
  expect(setup.related).toEqual([]);
});

test("an operation error exposes its typed fields", () => {
  const operation = new MemoryOperationError("commit", "stale_revision", "the session is stale", false);
  expect(operation).toBeInstanceOf(Error);
  expect(operation.name).toBe("MemoryOperationError");
  expect(operation.message).toBe("the session is stale");
  expect([operation.stage, operation.code, operation.retryable]).toEqual(["commit", "stale_revision", false]);
  expectError(() => new MemoryOperationError("learn", "bad", "failed", 1 as never), TypeError, "retryable must be a boolean");
  expectError(() => new MemoryOperationError("", "bad", "failed", true), RangeError, "stage must be non-empty");
  expectError(() => new MemoryOperationError("learn", "", "failed", true), RangeError, "code must be non-empty");
  expectError(() => new MemoryOperationError("learn", "bad", "", true), RangeError, "message must be non-empty");
});

test("an operation failure checks its fields and freezes", () => {
  const failure = new MemoryOperationFailure("commit", "declined", "Declined.", true);
  expect([failure.stage, failure.code, failure.message, failure.retryable]).toEqual([
    "commit",
    "declined",
    "Declined.",
    true,
  ]);
  expect(Object.isFrozen(failure)).toBe(true);
  expectError(() => new MemoryOperationFailure("", "c", "m", true), RangeError, "stage must be non-empty");
  expectError(() => new MemoryOperationFailure("s", "", "m", true), RangeError, "code must be non-empty");
  expectError(() => new MemoryOperationFailure("s", "c", "", true), RangeError, "message must be non-empty");
  expectError(() => new MemoryOperationFailure(1n as never, "c", "m", true), TypeError, "stage must be a string");
  expectError(
    () => new MemoryOperationFailure("\uD800", "c", "m", true),
    RangeError,
    "Unicode surrogate is not valid canonical JSON",
  );
  expectError(() => new MemoryOperationFailure("s", "c", "m", 1 as never), TypeError, "retryable must be a boolean");
});

/** A fake memory's observations. */
interface FakeMemoryLog {
  resets: number;
  closes: number;
  opened: (JsonObject | null)[];
  commits: [string, readonly MemoryMutation[]][];
}

/**
 * Build a plain-object memory whose sessions record the snapshot they opened with and their
 * commits; with `async`, every method returns a promise.
 */
function fakeMemory(async = false): Memory & FakeMemoryLog {
  const settle = <T>(value: T): T | Promise<T> => (async ? Promise.resolve(value) : value);
  return {
    resets: 0,
    closes: 0,
    opened: [],
    commits: [],
    openRun({ snapshot = null } = {}) {
      this.opened.push(snapshot);
      const session: MemorySession = {
        snapshot: Object.freeze({ namespace: "test" }),
        close: () => {
          this.closes++;
          return settle(undefined);
        },
        recall: (context = null) => settle(context === null ? "all" : context.prompt),
        learn: (context) => settle([new MemoryMutation("learned", { run: context.runId })]),
        commit: (commitId, mutations) => {
          this.commits.push([commitId, mutations]);
          return settle(undefined);
        },
      };
      return settle(session);
    },
    reset() {
      this.resets++;
      return settle(undefined);
    },
  };
}

// test_protocols_accept_narrow_structural_implementations
test.each([
  ["synchronous", false],
  ["async", true],
] as const)("the memory protocols accept a narrow %s structural implementation", async (_kind, async) => {
  const memory = fakeMemory(async);
  expect(isMemory(memory)).toBe(true);
  for (const value of [{}, null, undefined, "memory", { openRun() {} }, { openRun: 1, reset() {} }]) {
    expect(isMemory(value)).toBe(false);
  }
  const snapshot = { namespace: "test" };
  const opening = memory.openRun({ snapshot });
  expect(opening instanceof Promise).toBe(async);
  const session = await opening;
  await memory.openRun();
  expect(memory.opened[0]).toBe(snapshot);
  expect(memory.opened[1]).toBeNull();
  expect(session.snapshot).toEqual({ namespace: "test" });
  expect(() => {
    (session.snapshot as Record<string, unknown>).namespace = "changed";
  }).toThrow(TypeError);
  expect(session.snapshot).toEqual({ namespace: "test" });
  expect(await session.recall()).toBe("all");
  expect(await session.recall(null)).toBe("all");
  expect(await session.recall(new RecallContext("base", "query", {}, ["root", "agent"]))).toBe("query");
  const learned = await session.learn(new LearnContext("run", "notes", []));
  expect(Array.isArray(learned)).toBe(true);
  expect(learned[0]!.payload).toEqual({ run: "run" });
  await session.commit("commit-1", learned);
  expect(memory.commits).toEqual([["commit-1", learned]]);
  expect(memory.commits[0]![1]).toBe(learned);
  await session.close();
  expect(memory.closes).toBe(1);
  await memory.reset();
  expect(memory.resets).toBe(1);
});

// test_session_protocol_recalls_and_learns_without_frames
test("the session protocol recalls and learns without frames", async () => {
  const session: MemorySession = {
    snapshot: {},
    close() {},
    recall: () => "",
    learn: (context) => [new MemoryMutation("learned", { memory: context.memory })],
    commit() {},
  };
  const learned = await session.learn(new LearnContext("run", "notes", []));
  expect(learned[0]!.payload).toEqual({ memory: "notes" });
  expect("openFrame" in session).toBe(false);
  // @ts-expect-error a session has no openFrame
  expect(session.openFrame).toBeUndefined();
});

// test_memory_recall_error_preserves_an_explicit_exception_cause
test("a recall error preserves an explicit cause", () => {
  const origin = new OriginRef("agent-policy");
  const cause = new Error("backend failed");
  const error = new MemoryRecallError("memory recall failed", { origin, cause });
  expect(error).toBeInstanceOf(Error);
  expect(error.name).toBe("MemoryRecallError");
  expect(error.message).toBe("memory recall failed");
  expect(error.origin).toBe(origin);
  expect(error.cause).toBe(cause);
  const bare = new MemoryRecallError("");
  expect(bare.origin).toBeNull();
  expect("cause" in bare).toBe(false);
  expectError(
    () => new MemoryRecallError("m", { origin: "site" as never }),
    TypeError,
    "origin must be an OriginRef or null",
  );
  expectError(() => new MemoryRecallError(1 as never), TypeError, "message must be a string");
});

// test_scope_carries_a_root_relative_path_and_memory_keys
test("a scope carries a root-relative path and memory keys", () => {
  const segment = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["child"], 2);
  const scope = new MemoryScope(["child", "agent"], { memories: ["notes", "chat"], occurrencePath: [segment] });
  expect([scope.staticPath, scope.memories, scope.occurrencePath]).toEqual([
    ["child", "agent"],
    ["notes", "chat"],
    [segment],
  ]);
  expect(new MemoryScope(["plain"]).memories).toEqual([]);
  expectError(
    () => new MemoryScope(["agent"], { memories: ["notes", "notes"] }),
    RangeError,
    "memories must be unique",
  );
  expectError(
    () => new MemoryScope(["agent"], { memories: [""] }),
    RangeError,
    "memories components must be non-empty",
  );
  expectError(
    () => new MemoryScope(["agent"], { memories: new Set(["notes"]) as never }),
    TypeError,
    "memories must be an array",
  );
  expectError(
    () => new MemoryScope(["child"], { occurrencePath: [segment] }),
    RangeError,
    "the innermost occurrence segment must contain the scoped node",
  );
});

// test_frame_types_are_gone (the memory package, contracts.ts and runtime/contracts.ts rows)
test("the frame types are gone", () => {
  for (const name of ["MemoryFrame", "MemoryFrameRef", "MemoryFrameSpec", "MemoryOccurrenceRef", "MemoryOwner"]) {
    expect(name in memoryPackage).toBe(false);
    expect(name in contracts).toBe(false);
  }
  expect("FrameProposal" in runtimeContracts).toBe(false);
});

// test_importing_memory_contracts_does_not_load_heavy_backends
test("the memory package imports only its contracts", () => {
  // The facade must stay a dependency-light leaf: contracts.ts's own imports are pinned by the
  // layer-direction test, so the facade importing nothing else keeps the whole package light.
  const source = readFileSync(new URL("../../src/memory/index.ts", import.meta.url), "utf8");
  expect(new Set(importStatements(source).map((statement) => statement.specifier))).toEqual(
    new Set(["./contracts.ts"]),
  );
});

// test_context_episode_and_mutation_keep_json_snapshots (the mutation part)
test("a mutation keeps a frozen JSON snapshot", () => {
  const payload: { add: { id: string; text: string }[] } = { add: [{ id: "lesson", text: "remember" }] };
  const mutation = new MemoryMutation("m1", payload);
  payload.add.length = 0;
  expect(() => {
    (mutation.payload as { add: unknown[] }).add.push("changed");
  }).toThrow(TypeError);
  expect(mutation.payload).toEqual({ add: [{ id: "lesson", text: "remember" }] });
  expect(Object.isFrozen(mutation)).toBe(true);
  expect(Object.isFrozen(mutation.payload)).toBe(true);
});

// test_json_scalar_identity_preserves_exact_types_at_every_depth (the mutation row)
test("mutation JSON scalars keep their exact types at every depth", () => {
  for (const [value, type] of [
    [true, "boolean"],
    [1n, "bigint"],
    [1, "number"],
  ] as const) {
    const payload = new MemoryMutation("same", [value, { value }]).payload as readonly [unknown, { value: unknown }];
    expect(typeof payload[0]).toBe(type);
    expect(typeof payload[1].value).toBe(type);
    expect(payload[0]).toBe(value);
    expect(payload[1].value).toBe(value);
  }
});

// test_json_scalar_identity_preserves_exact_types_at_every_depth (the recall context row)
test("recall context JSON scalars keep their exact types at every depth", () => {
  for (const [value, type] of [
    [true, "boolean"],
    [1n, "bigint"],
    [1, "number"],
  ] as const) {
    const inputs = new RecallContext("base", "prompt", { value, nested: [value, { value }] }, ["root", "agent"])
      .inputs as { value: unknown; nested: readonly [unknown, { value: unknown }] };
    expect(typeof inputs.value).toBe(type);
    expect(typeof inputs.nested[0]).toBe(type);
    expect(typeof inputs.nested[1].value).toBe(type);
    expect(inputs.nested[1].value).toBe(value);
  }
});

// test_context_episode_and_mutation_keep_json_snapshots (the recall context part)
test("a recall context keeps a frozen JSON snapshot", () => {
  const inputs = { items: [1n, 2n] };
  const path = ["root", "agent"];
  const context = new RecallContext("base", "prompt", inputs, path);
  inputs.items.push(3n);
  path.push("changed");
  expect(() => {
    (context.inputs as { items: bigint[] }).items.push(4n);
  }).toThrow(TypeError);
  expect(context.inputs).toEqual({ items: [1n, 2n] });
  expect([context.baseSystemPrompt, context.prompt, context.consumerStaticPath]).toEqual([
    "base",
    "prompt",
    ["root", "agent"],
  ]);
  for (const value of [context, context.inputs, context.consumerStaticPath]) {
    expect(Object.isFrozen(value)).toBe(true);
  }
});

test("recall context inputs follow the value number rule at any depth", () => {
  let deep: unknown = 1n;
  for (let i = 0; i < 150; i++) deep = { d: deep };
  const inputs = new RecallContext("", "", { inf: Number.NEGATIVE_INFINITY, zero: -0, deep }, ["agent"]).inputs;
  expect(inputs.inf).toBe(Number.NEGATIVE_INFINITY);
  expect(Object.is(inputs.zero, 0)).toBe(true);
  expect(inputs.deep).toEqual(deep as JsonObject);
  expectError(
    () => new RecallContext("", "", { nan: Number.NaN }, ["agent"]),
    RangeError,
    "inputs.nan JSON number must not be NaN",
  );
});

test.each<[string, unknown[], typeof TypeError | typeof RangeError, string]>([
  ["a non-string base prompt", [1, "", {}, ["a"]], TypeError, "base_system_prompt must be a string"],
  ["a non-string prompt", ["", null, {}, ["a"]], TypeError, "prompt must be a string"],
  ["a lone surrogate", ["", "\ud800", {}, ["a"]], RangeError, "Unicode surrogate is not valid canonical JSON"],
  ["array inputs", ["", "", [], ["a"]], TypeError, "inputs must be an exact JSON object"],
  ["an empty path", ["", "", {}, []], RangeError, "consumer_static_path must be non-empty"],
  ["an empty component", ["", "", {}, [""]], RangeError, "consumer_static_path components must be non-empty"],
  ["a non-array path", ["", "", {}, new Set(["a"])], TypeError, "consumer_static_path must be an array"],
])("a recall context refuses %s", (_name, args, cls, message) => {
  expectError(() => new RecallContext(...(args as [never, never, never, never])), cls, message);
});

test("a mutation checks its id and payload", () => {
  let deep: unknown = {};
  for (let i = 0; i < 150; i++) deep = { d: deep };
  expect(new MemoryMutation("deep", deep).payload).toEqual(deep as JsonObject);
  expect(new MemoryMutation("scalar", "text").payload).toBe("text");
  expectError(
    () => new MemoryMutation("m", { x: Number.POSITIVE_INFINITY }),
    RangeError,
    "payload.x JSON number must be finite",
  );
  expectError(() => new MemoryMutation("m", Number.NaN), RangeError, "payload JSON number must be finite");
  expectError(() => new MemoryMutation("", {}), RangeError, "mutation_id must be non-empty");
  expectError(() => new MemoryMutation("\ud800", {}), RangeError, "Unicode surrogate is not valid canonical JSON");
  expectError(() => new MemoryMutation(1 as never, {}), TypeError, "mutation_id must be a string");
  expectError(() => new MemoryMutation("m", new Map()), TypeError, "payload must contain exact JSON values");
});

test("validateMutations keeps an array of unique mutations by reference", () => {
  const a = new MemoryMutation("a", 1n);
  const b = new MemoryMutation("b", 2n);
  const input = [a, b];
  const checked = validateMutations(input);
  expect(checked).not.toBe(input);
  expect(Object.isFrozen(checked)).toBe(true);
  expect(checked[0]).toBe(a);
  expect(checked[1]).toBe(b);
  expect(validateMutations([])).toEqual([]);
  class SubMutation extends MemoryMutation {}
  expectError(() => validateMutations(new Set([a])), TypeError, "mutations must be an array");
  const item = "mutations items must be MemoryMutation values";
  expectError(() => validateMutations([a, { mutationId: "x", payload: 1n }]), TypeError, item);
  expectError(() => validateMutations([new SubMutation("s", 1n)]), TypeError, item);
  expectError(() => validateMutations([a, new MemoryMutation("a", {})]), RangeError, "mutation IDs must be unique");
});

// test_episodes_contexts_mutations_and_failures_name_the_memory_key (the mutation, proposal and
// failure rows)
test("mutations, proposals and failures name the memory key", () => {
  const mutation = new MemoryMutation("m1", { add: 1n });
  expect([mutation.mutationId, mutation.payload]).toEqual(["m1", { add: 1n }]);
  expect("frameRef" in mutation).toBe(false);
  const proposal = new MemoryProposal("notes", [mutation]);
  expect(proposal.memory).toBe("notes");
  expect(proposal.mutations).toEqual([mutation]);
  expectError(
    () => new MemoryProposal("notes", [mutation, new MemoryMutation("m1", {})]),
    RangeError,
    "mutation IDs must be unique",
  );
  const cause = new MemoryOperationFailure("learn", "notes.failed", "boom", false);
  expect(new LearningFailure("notes", "learn", cause).memory).toBe("notes");
  expect("frameRef" in new LearningFailure("notes", "commit", cause)).toBe(false);
  expect("frameRef" in new BackwardFailure("learn", "x", "y", { memory: "notes" })).toBe(false);
});

test("a learning failure checks its attribution and failure", () => {
  const cause = new MemoryOperationFailure("learn", "busy", "try later", true);
  const failure = new LearningFailure("notes", "learn", cause);
  expect([failure.memory, failure.operation]).toEqual(["notes", "learn"]);
  expect(failure.failure).toBe(cause);
  expect(Object.isFrozen(failure)).toBe(true);
  class SubFailure extends MemoryOperationFailure {}
  const operation = "operation must be 'learn' or 'commit'";
  expectError(() => new LearningFailure("notes", "other" as never, cause), RangeError, operation);
  expectError(() => new LearningFailure("", "learn", cause), RangeError, "memory must be non-empty");
  expectError(() => new LearningFailure(1 as never, "learn", cause), TypeError, "memory must be a string");
  expectError(() => new LearningFailure("notes", "" as never, cause), RangeError, "operation must be non-empty");
  expectError(() => new LearningFailure("notes", 1 as never, cause), TypeError, "operation must be a string");
  const message = "failure must be a MemoryOperationFailure";
  expectError(() => new LearningFailure("notes", "learn", { ...cause } as never), TypeError, message);
  expectError(() => new LearningFailure("notes", "learn", new SubFailure("learn", "c", "m", true)), TypeError, message);
});

// test_learning_result_keeps_primary_independent_failure_attribution
test("a learning result keeps primary, independent failure attribution", () => {
  const busy = new MemoryOperationFailure("learn", "busy", "try later", true);
  const rejected = new MemoryOperationFailure("commit", "rejected", "not applied", false);
  const retryable = new LearningFailure("reflective", "learn", busy);
  const final = new LearningFailure("profile", "commit", rejected);
  expect(new LearningResult("succeeded").failures).toEqual([]);
  expect(new LearningResult("failed", { failures: [final] }).status).toBe("failed");
  const result = new LearningResult("incomplete", { failures: [retryable, final] });
  expect(result.failures).toEqual([retryable, final]);
  expect(result.failures[0]).toBe(retryable);
  expect(new LearningResult("failed", { failures: [retryable] }).failures).toEqual([retryable]);
  expect(new LearningResult("incomplete", { failures: [final] }).failures).toEqual([final]);
});

// test_policy_commit_outcome_keeps_protocol_identity_and_backend_hints
test.each<[PolicyCommitStatus, string | null, boolean]>([
  ["unrequested", null, false],
  ["unknown", "commit-1", false],
  ["succeeded", "commit-1", false],
  ["failed", "commit-1", true],
])("a %s policy commit outcome keeps its identity and backend hints", (status, commitId, hasFailure) => {
  const failure = hasFailure ? new MemoryOperationFailure("backend-stage", "retry", "Try later.", true) : null;
  const outcome = new PolicyCommitOutcome("history", status, { commitId, failure });
  expect([outcome.memory, outcome.status, outcome.commitId, outcome.failure]).toEqual([
    "history",
    status,
    commitId,
    failure,
  ]);
  expect(outcome.failure).toBe(failure);
  expect(() => {
    (outcome as { status: string }).status = "succeeded";
  }).toThrow(TypeError);
});

// test_learning_completion_is_independent_of_retryability_and_policy_confirmation
test("learning completion is independent of retryability and policy confirmation", () => {
  const cause = new MemoryOperationFailure("backend", "busy", "Retry.", true);
  const failure = new LearningFailure("history", "commit", cause);
  const succeeded = new PolicyCommitOutcome("history", "succeeded", { commitId: "commit-1" });
  const pending = new PolicyCommitOutcome("other", "unrequested");
  expect(new LearningResult("failed", { failures: [failure] }).failures).toEqual([failure]);
  expect(new LearningResult("incomplete").policies).toEqual([]);
  expect(new LearningResult("incomplete", { policies: [succeeded, pending] }).policies).toEqual([succeeded, pending]);
  expect(new LearningResult("incomplete", { policies: [succeeded] }).status).toBe("incomplete");
  expect(new LearningResult("succeeded", { policies: [succeeded] }).policies).toEqual([succeeded]);
  expectError(
    () => new LearningResult("succeeded", { policies: [pending] }),
    RangeError,
    "succeeded learning requires succeeded policies",
  );
  expectError(() => new LearningResult("failed"), RangeError, "failed learning requires at least one failure");
  expectError(
    () => new LearningResult("incomplete", { policies: [succeeded, succeeded] }),
    RangeError,
    "learning memory keys must be unique",
  );
});

// test_policy_commit_outcome_rejects_inconsistent_optional_fields
test.each<[string, string | null, boolean, string]>([
  ["unrequested", "commit-1", false, "unrequested policy cannot carry commit data"],
  ["unrequested", null, true, "unrequested policy cannot carry commit data"],
  ["unknown", null, false, "requested policy requires commit_id"],
  ["unknown", "commit-1", true, "only failed policy commits require a failure"],
  ["succeeded", null, false, "requested policy requires commit_id"],
  ["succeeded", "commit-1", true, "only failed policy commits require a failure"],
  ["failed", null, true, "requested policy requires commit_id"],
  ["failed", "commit-1", false, "only failed policy commits require a failure"],
  ["other", null, false, "invalid policy commit status"],
])("a %s policy commit outcome with commit id %p and failure %p is refused", (status, commitId, withFailure, text) => {
  const failure = withFailure ? new MemoryOperationFailure("backend", "error", "Failed.", false) : null;
  expectError(() => new PolicyCommitOutcome("history", status as never, { commitId, failure }), RangeError, text);
});

// test_policy_commit_outcome_requires_exact_valid_unicode_identities
test.each(
  (["memory", "status", "commit_id"] as const).flatMap((field) =>
    (
      [
        ["", RangeError],
        ["\ud800", RangeError],
        [1, TypeError],
        [true, TypeError],
      ] as const
    ).map(([invalid, cls]) => [field, invalid, cls] as const),
  ),
)("a policy commit outcome refuses %s %p", (field, invalid, cls) => {
  const values: Record<string, unknown> = { memory: "history", status: "unknown", commit_id: "commit-1" };
  values[field] = invalid;
  const { memory, status, commit_id: commitId } = values as { memory: never; status: never; commit_id: never };
  expect(() => new PolicyCommitOutcome(memory, status, { commitId })).toThrow(cls);
});

test("a policy commit outcome refuses a failure that isn't a MemoryOperationFailure", () => {
  const failure = new MemoryOperationFailure("backend", "failed", "Failed.", true);
  expectError(
    () => new PolicyCommitOutcome("history", "failed", { commitId: "c", failure: { ...failure } as never }),
    TypeError,
    "failure must be a MemoryOperationFailure or null",
  );
});

// test_learning_result_revalidates_exact_nested_policy_carriers (the rows without forgery)
test("a learning result checks its exact nested policy carriers", () => {
  const first = new PolicyCommitOutcome("second", "unknown", { commitId: "commit-2" });
  const second = new PolicyCommitOutcome("first", "unrequested");
  const policies = [first, second];
  const result = new LearningResult("incomplete", { failures: [], policies });
  expect(result.policies).toEqual([first, second]);
  expect(result.policies).not.toBe(policies);
  expect(result.policies[0]).toBe(first);
  expect(Object.isFrozen(result.policies[0])).toBe(true);
  expect(Object.isFrozen(result.policies)).toBe(true);
  expect(Object.isFrozen(result.failures)).toBe(true);
  expect(Object.isFrozen(result)).toBe(true);
  expectError(
    () => new LearningResult("incomplete", { policies: new Set([first]) as never }),
    TypeError,
    "learning policies must be an array",
  );
  class PolicySubclass extends PolicyCommitOutcome {}
  expectError(
    () => new LearningResult("incomplete", { policies: [new PolicySubclass("history", "unrequested")] }),
    TypeError,
    "policies items must be PolicyCommitOutcome values",
  );
});

test("a learning result checks its status and failures in Python's order", () => {
  const cause = new MemoryOperationFailure("learn", "busy", "try later", true);
  const a = new LearningFailure("notes", "learn", cause);
  expectError(
    () => new LearningResult("other" as never),
    RangeError,
    "learning status must be 'succeeded', 'failed', or 'incomplete'",
  );
  expectError(() => new LearningResult(1 as never), TypeError, "status must be a string");
  expectError(
    () => new LearningResult("failed", { failures: new Set([a]) as never }),
    TypeError,
    "learning failures must be an array",
  );
  expectError(
    () => new LearningResult("failed", { failures: [{ ...a }] as never }),
    TypeError,
    "failures items must be LearningFailure values",
  );
  expectError(
    () => new LearningResult("failed", { failures: [a, new LearningFailure("notes", "learn", cause)] }),
    RangeError,
    "learning failure attribution must be unique",
  );
  expect(
    new LearningResult("failed", { failures: [a, new LearningFailure("notes", "commit", cause)] }).failures,
  ).toHaveLength(2);
  expectError(
    () => new LearningResult("succeeded", { failures: [a] }),
    RangeError,
    "succeeded learning cannot carry failures",
  );
  // One loop per array, class then uniqueness, so the repeat is found before the stranger.
  expectError(
    () => new LearningResult("failed", { failures: [a, a, { ...a }] as never }),
    RangeError,
    "learning failure attribution must be unique",
  );
  expectError(
    () => new LearningResult("incomplete", { policies: [new PolicyCommitOutcome("x", "unrequested"), {}] as never }),
    TypeError,
    "policies items must be PolicyCommitOutcome values",
  );
});

/** One authored AGENT locator for the episode tests. */
const AGENT_PATH = ["root", "child", "agent"];

/** Build a succeeded episode of `run` with empty inputs and the given options. */
function episodeOf(
  run = "run",
  options: ConstructorParameters<typeof MemoryEpisode>[5] = { output: "answer" },
): MemoryEpisode {
  return new MemoryEpisode(run, [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, {}, options);
}

// test_context_episode_and_mutation_keep_json_snapshots (the episode and learn context parts)
test("an episode and a learn context keep frozen JSON snapshots", () => {
  const inputs = { items: [1n, 2n] };
  const interactionValue = { text: "answer" };
  const interaction = new ModelValueInteraction(interactionValue);
  const output = { answer: ["yes"] };
  const episode = new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, inputs, {
    output,
    interactions: [interaction],
  });
  const learn = new LearnContext("run", "notes", [episode], [{ score: 1n }]);

  inputs.items.push(3n);
  interactionValue.text = "changed";
  output.answer.push("no");
  expect(() => {
    ((episode.interactions[0] as ModelValueInteraction).value as { text: string }).text = "changed again";
  }).toThrow(TypeError);
  expect(() => {
    (episode.inputs as { items: bigint[] }).items.push(4n);
  }).toThrow(TypeError);
  expect(() => {
    (learn.feedback[0] as { score: bigint }).score = 0n;
  }).toThrow(TypeError);

  expect(episode.inputs).toEqual({ items: [1n, 2n] });
  expect(episode.output).toEqual({ answer: ["yes"] });
  expect(episode.failure).toBeNull();
  expect(episode.interactions).toEqual([new ModelValueInteraction({ text: "answer" })]);
  expect(learn.episodes).toEqual([episode]);
  expect(learn.episodes[0]).toBe(episode);
  expect(learn.feedback).toEqual([{ score: 1n }]);
  for (const value of [episode, learn, episode.interactions, learn.episodes, learn.feedback]) {
    expect(Object.isFrozen(value)).toBe(true);
  }
});

// test_episode_status_enforces_output_or_failure_shape
test("an episode's status enforces its output or failure shape", () => {
  const failed = new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.FAILED, {}, {
    failure: { code: "node.execute", message: "failed" },
  });
  expect(failed.output).toBeNull();
  expect(failed.failure).toEqual({ code: "node.execute", message: "failed" });
  expectError(
    () => new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.FAILED, {}),
    RangeError,
    "a failed episode must carry failure data",
  );
  expectError(
    () =>
      new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.FAILED, {}, {
        output: "conflict",
        failure: { code: "node.execute" },
      }),
    RangeError,
    "a failed episode cannot carry output data",
  );
  expectError(
    () =>
      new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, {}, {
        output: null,
        failure: { code: "unexpected" },
      }),
    RangeError,
    "a succeeded episode cannot carry failure data",
  );
});

// test_episode_execution_sequence_is_optional_and_survives_context_copy (rows 0 and 17)
test.each([0, 17])("an episode's execution sequence %p is optional and survives a context", (sequence) => {
  const legacy = episodeOf();
  expect(legacy.executionSeq).toBeNull();
  const inputs = { items: [1n] };
  const episode = new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, inputs, {
    output: "answer",
    executionSeq: sequence,
  });
  const context = new LearnContext("run", "notes", [episode]);
  inputs.items.push(2n);
  expect(context.episodes[0]!.executionSeq).toBe(sequence);
  expect(context.episodes[0]!.inputs).toEqual({ items: [1n] });
});

// test_episode_rejects_nonphysical_execution_sequences
test.each<[string, unknown, typeof TypeError | typeof RangeError, string]>([
  ["false", false, TypeError, "execution_seq must be an integer or null"],
  ["true", true, TypeError, "execution_seq must be an integer or null"],
  ["-1", -1, RangeError, "execution_seq must be non-negative"],
  ["1.5", 1.5, TypeError, "execution_seq must be an integer or null"],
  ['"1"', "1", TypeError, "execution_seq must be an integer or null"],
  ["1n", 1n, TypeError, "execution_seq must be an integer or null"],
])("an episode refuses the execution sequence %s", (_name, sequence, cls, message) => {
  expectError(() => episodeOf("run", { executionSeq: sequence as number }), cls, message);
});

// test_learn_context_requires_episodes_of_its_run
test("a learn context requires episodes of its run", () => {
  expectError(
    () => new LearnContext("run", "notes", [episodeOf("other-run")]),
    RangeError,
    "episode run_id must match the learning context",
  );
});

// test_episodes_contexts_mutations_and_failures_name_the_memory_key (the episode and context rows)
test("episodes and learn contexts name the memory key", () => {
  const segment = new MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ["loop"], 0);
  const episode = new MemoryEpisode("run", [segment], ["loop", "agent"], MemoryEpisodeStatus.SUCCEEDED, {}, {
    output: "answer",
    memorySections: [new MemorySection("notes", "outer")],
  });
  expect(episode.occurrencePath).toEqual([segment]);
  expect(episode.occurrencePath[0]).toBe(segment);
  expectError(
    () => new MemoryEpisode("run", [segment], ["other"], MemoryEpisodeStatus.SUCCEEDED, {}),
    RangeError,
    "the innermost occurrence segment must contain the scoped node",
  );
  const context = new LearnContext("run", "notes", [episode], ["good"]);
  expect([context.memory, context.episodes, context.feedback]).toEqual(["notes", [episode], ["good"]]);
  expectError(() => new LearnContext("run", "", []), RangeError, "memory must be non-empty");
});

// test_freeze_still_detects_cycles
test("freezing feedback still detects cycles", () => {
  const loop: unknown[] = [];
  loop.push(loop);
  expectError(
    () => new LearnContext("run", "notes", [], [loop]),
    RangeError,
    "feedback[0][0] contains a circular JSON value",
  );
  const node: Record<string, unknown> = {};
  node.self = node;
  expectError(
    () => new LearnContext("run", "notes", [], [{ x: [node] }]),
    RangeError,
    "feedback[0].x[0].self contains a circular JSON value",
  );
});

// test_freeze_shares_a_repeated_subtree
test("freezing feedback shares a repeated subtree", () => {
  const shared = { note: ["kept", 1n] };
  const context = new LearnContext("run", "notes", [], [{ a: shared, b: shared }]);
  const item = context.feedback[0] as { a: JsonValue; b: JsonValue };
  expect(item.a).toBe(item.b);
  expect(context.feedback).toEqual([{ a: shared, b: shared }]);
});

// test_learn_context_equal_feedback
test("learn contexts over aliased and copied feedback are equal", () => {
  const shared = { note: ["kept", 1n] };
  const feedback = [{ a: shared, b: [shared] }, shared];
  const aliased = new LearnContext("run", "notes", [], feedback);
  const copied = new LearnContext("run", "notes", [], structuredClone(feedback));
  expect(aliased.feedback).toEqual(copied.feedback);
  expect(copied.feedback).toEqual(feedback);
  expect(aliased).toEqual(copied);
});

// test_propose_learning_shares_canonical_feedback
test("the learner receives the exact LearnContext", async () => {
  const memory = new FakeMemory();
  const runtime = await MemoryRuntime.open({
    memories: [new MemorySpec("notes", "notes", {}, [new DeclSite([], ["root"])], false)],
    origins: new Map([["notes", null]]),
    memory: new Map([["notes", memory]]),
  });
  const context = new LearnContext("run", "notes", [], [{ b: [1n, "x", null, 1.5, true], a: {} }]);
  try {
    await runtime.proposeLearning([context]);
  } finally {
    await runtime.close();
  }
  expect(memory.learnContexts).toHaveLength(1);
  expect(memory.learnContexts[0]).toBe(context);
  expect(memory.learnContexts[0]!.feedback).toBe(context.feedback);
});

class SubSection extends MemorySection {}
class SubEpisode extends MemoryEpisode {}

test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string]>([
  ["an empty run id", () => episodeOf(""), RangeError, "run_id must be non-empty"],
  ["a run id that isn't a string", () => episodeOf(1 as never), TypeError, "run_id must be a string"],
  [
    "a bad run id before a bad sequence",
    () => episodeOf("", { executionSeq: -1 }),
    RangeError,
    "run_id must be non-empty",
  ],
  [
    "a bad sequence before a bad occurrence path",
    () =>
      new MemoryEpisode("run", new Set() as never, AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, {}, { executionSeq: -1 }),
    RangeError,
    "execution_seq must be non-negative",
  ],
  [
    "an occurrence path that isn't an array",
    () => new MemoryEpisode("run", new Set() as never, AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, {}),
    TypeError,
    "occurrence_path must be an array",
  ],
  [
    "an occurrence item that isn't a segment",
    () => new MemoryEpisode("run", [{}] as never, AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, {}),
    TypeError,
    "occurrence_path items must be MemoryOccurrenceSegment values",
  ],
  [
    "an owner path that isn't an array",
    () => new MemoryEpisode("run", [], new Set() as never, MemoryEpisodeStatus.SUCCEEDED, {}),
    TypeError,
    "owner_static_path must be an array",
  ],
  [
    "an empty owner path",
    () => new MemoryEpisode("run", [], [], MemoryEpisodeStatus.SUCCEEDED, {}),
    RangeError,
    "owner_static_path must be non-empty",
  ],
  [
    "occurrence segments out of lexical nesting order",
    () =>
      new MemoryEpisode(
        "run",
        [
          new MemoryOccurrenceSegment(MemoryOccurrenceKind.LOOP_ITERATION, ["root", "child"], 0),
          new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["root"], 0),
        ],
        AGENT_PATH,
        MemoryEpisodeStatus.SUCCEEDED,
        {},
      ),
    RangeError,
    "occurrence segments must follow lexical nesting order",
  ],
  [
    "an owner path component that isn't a string",
    () => new MemoryEpisode("run", [], ["root", 1] as never, MemoryEpisodeStatus.SUCCEEDED, {}),
    TypeError,
    "owner_static_path components must be a string",
  ],
  [
    "an empty owner path component",
    () => new MemoryEpisode("run", [], ["root", ""], MemoryEpisodeStatus.SUCCEEDED, {}),
    RangeError,
    "owner_static_path components must be non-empty",
  ],
  [
    "an unknown status",
    () => new MemoryEpisode("run", [], AGENT_PATH, "SUCCEEDED" as never, []),
    TypeError,
    "status must be a MemoryEpisodeStatus",
  ],
  [
    "a shape error before bad inputs",
    () => new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.FAILED, []),
    RangeError,
    "a failed episode must carry failure data",
  ],
  [
    "inputs that are an array",
    () => new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, []),
    TypeError,
    "inputs must be an exact JSON object or null",
  ],
  [
    "inputs that are a Map",
    () => new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, new Map()),
    TypeError,
    "inputs must be an exact JSON object or null",
  ],
  [
    "a bad inputs value before a bad output",
    () =>
      new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, { a: new Map() }, { output: new Map() }),
    TypeError,
    "inputs.a must contain exact JSON values",
  ],
  [
    "an output that isn't JSON",
    () => episodeOf("run", { output: new Map() }),
    TypeError,
    "output must contain exact JSON values",
  ],
  [
    "a failure that isn't an object",
    () => new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.FAILED, {}, { failure: "boom" }),
    TypeError,
    "failure must be an exact JSON object",
  ],
  [
    "a bad failure before a bad prompt pair",
    () => new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.FAILED, {}, { failure: [], prompt: "p" }),
    TypeError,
    "failure must be an exact JSON object",
  ],
  [
    "a prompt without a base prompt",
    () => episodeOf("run", { prompt: "p" }),
    RangeError,
    "base_system_prompt and prompt must both be present or absent",
  ],
  [
    "a base prompt without a prompt",
    () => episodeOf("run", { baseSystemPrompt: "b", prompt: null }),
    RangeError,
    "base_system_prompt and prompt must both be present or absent",
  ],
  [
    "a base prompt that isn't a string",
    () => episodeOf("run", { baseSystemPrompt: 1 as never, prompt: "p" }),
    TypeError,
    "base_system_prompt must be a string",
  ],
  [
    "a prompt that isn't a string",
    () => episodeOf("run", { baseSystemPrompt: "b", prompt: 1 as never }),
    TypeError,
    "prompt must be a string",
  ],
  [
    "a bad prompt before bad sections",
    () => episodeOf("run", { baseSystemPrompt: "b", prompt: 1 as never, memorySections: new Set() as never }),
    TypeError,
    "prompt must be a string",
  ],
  [
    "sections that aren't an array",
    () => episodeOf("run", { memorySections: new Set() as never }),
    TypeError,
    "memory_sections must be an array",
  ],
  [
    "a section that isn't a MemorySection",
    () => episodeOf("run", { memorySections: [{ memory: "notes", text: "x" }] as never }),
    TypeError,
    "memory_sections items must be MemorySection values",
  ],
  [
    "a section subclass",
    () => episodeOf("run", { memorySections: [new SubSection("notes", "x")] }),
    TypeError,
    "memory_sections items must be MemorySection values",
  ],
  [
    "bad sections before bad interactions",
    () => episodeOf("run", { memorySections: [{}] as never, interactions: new Set() as never }),
    TypeError,
    "memory_sections items must be MemorySection values",
  ],
  [
    "interactions that aren't an array",
    () => episodeOf("run", { interactions: new Set() as never }),
    TypeError,
    "interactions must be an array",
  ],
  [
    "an interaction that isn't a visible interaction",
    () => episodeOf("run", { interactions: [{ text: "x" }] as never }),
    TypeError,
    "interactions items must be VisibleInteraction values",
  ],
])("an episode refuses %s", (_name, build, cls, message) => {
  expectError(build, cls, message);
});

test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string]>([
  ["an empty run id", () => new LearnContext("", "notes", []), RangeError, "run_id must be non-empty"],
  [
    "a run id that isn't a string",
    () => new LearnContext(1 as never, "notes", []),
    TypeError,
    "run_id must be a string",
  ],
  ["a memory that isn't a string", () => new LearnContext("run", 1 as never, []), TypeError, "memory must be a string"],
  [
    "a bad memory before bad episodes",
    () => new LearnContext("run", "", new Set() as never),
    RangeError,
    "memory must be non-empty",
  ],
  [
    "episodes that aren't an array",
    () => new LearnContext("run", "notes", new Set() as never),
    TypeError,
    "episodes must be an array",
  ],
  [
    "an episode that isn't a MemoryEpisode",
    () => new LearnContext("run", "notes", [{ ...episodeOf() }] as never),
    TypeError,
    "episodes items must be MemoryEpisode values",
  ],
  [
    "an episode subclass",
    () => new LearnContext("run", "notes", [new SubEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, {})]),
    TypeError,
    "episodes items must be MemoryEpisode values",
  ],
  [
    "an episode of another run before bad feedback",
    () => new LearnContext("run", "notes", [episodeOf("other")], new Set() as never),
    RangeError,
    "episode run_id must match the learning context",
  ],
  [
    "feedback that isn't an array",
    () => new LearnContext("run", "notes", [], new Set() as never),
    TypeError,
    "feedback must be an array",
  ],
  [
    "a feedback item that isn't JSON",
    () => new LearnContext("run", "notes", [], [1n, { a: new Map() }]),
    TypeError,
    "feedback[1].a must contain exact JSON values",
  ],
  [
    "a NaN feedback item",
    () => new LearnContext("run", "notes", [], [Number.NaN]),
    RangeError,
    "feedback[0] JSON number must not be NaN",
  ],
])("a learn context refuses %s", (_name, build, cls, message) => {
  expectError(build, cls, message);
});

test("episode JSON and learn feedback keep infinities, read -0 as 0 and refuse NaN", () => {
  const inf = Number.POSITIVE_INFINITY;
  const succeeded = new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, { x: inf, z: -0 }, {
    output: [-inf, -0],
  });
  expect(succeeded.inputs).toEqual({ x: inf, z: 0 });
  expect(Object.is((succeeded.inputs as { z: number }).z, 0)).toBe(true);
  expect(succeeded.output).toEqual([-inf, 0]);
  expect(Object.is((succeeded.output as number[])[1], 0)).toBe(true);
  const failed = new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.FAILED, null, { failure: { y: inf } });
  expect(failed.inputs).toBeNull();
  expect(failed.failure).toEqual({ y: inf });
  expect(new LearnContext("run", "notes", [], [inf, { n: -inf }]).feedback).toEqual([inf, { n: -inf }]);
  const nan = Number.NaN;
  expectError(() => episodeOf("run", { output: { v: nan } }), RangeError, "output.v JSON number must not be NaN");
  expectError(
    () => new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.SUCCEEDED, { x: nan }),
    RangeError,
    "inputs.x JSON number must not be NaN",
  );
  expectError(
    () => new MemoryEpisode("run", [], AGENT_PATH, MemoryEpisodeStatus.FAILED, {}, { failure: { y: [nan] } }),
    RangeError,
    "failure.y[0] JSON number must not be NaN",
  );
});

test("an episode keeps its fields, copies its arrays and keeps their carriers by reference", () => {
  const segment = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["root", "child"], 3);
  const occurrencePath = [segment];
  const ownerPath = [...AGENT_PATH];
  const section = new MemorySection("notes", "recalled");
  const sections = [section];
  const text = new ModelTextInteraction("visible");
  const interactions = [text];
  const episode = new MemoryEpisode("run", occurrencePath, ownerPath, MemoryEpisodeStatus.SUCCEEDED, null, {
    output: 1n,
    baseSystemPrompt: "",
    prompt: "",
    memorySections: sections,
    interactions,
    executionSeq: 4,
  });
  occurrencePath.length = 0;
  ownerPath.length = 0;
  sections.length = 0;
  interactions.length = 0;
  expect(episode.runId).toBe("run");
  expect(episode.occurrencePath).toEqual([segment]);
  expect(episode.ownerStaticPath).toEqual(AGENT_PATH);
  expect(episode.status).toBe("succeeded");
  expect(episode.inputs).toBeNull();
  expect(episode.output).toBe(1n);
  expect([episode.baseSystemPrompt, episode.prompt]).toEqual(["", ""]);
  expect(episode.memorySections[0]).toBe(section);
  expect(episode.interactions[0]).toBe(text);
  expect(episode.executionSeq).toBe(4);
  for (const value of [episode.occurrencePath, episode.ownerStaticPath, episode.memorySections, episode.interactions]) {
    expect(Object.isFrozen(value)).toBe(true);
  }
  expect(() => {
    (episode as { runId: string }).runId = "other";
  }).toThrow(TypeError);
  expect(Object.values(MemoryEpisodeStatus)).toEqual(["succeeded", "failed"]);
});

test("fromFrozen stores the given feedback and checks run, memory and episodes", () => {
  const frozen = freezeFeedback([{ a: 1n }, "two"], new Map());
  const episode = episodeOf();
  const context = LearnContext.fromFrozen("run", "notes", [episode], frozen);
  expect(context.feedback).toBe(frozen);
  expect(context.feedback[0]).toBe(frozen[0]);
  expect(context.episodes).toEqual([episode]);
  expect([context.runId, context.memory]).toEqual(["run", "notes"]);
  expect(context).toBeInstanceOf(LearnContext);
  expect(Object.isFrozen(context)).toBe(true);
  expectError(() => LearnContext.fromFrozen("", "notes", [], frozen), RangeError, "run_id must be non-empty");
  expectError(() => LearnContext.fromFrozen("run", 1 as never, [], frozen), TypeError, "memory must be a string");
  expectError(
    () => LearnContext.fromFrozen("run", "notes", [episodeOf("other")], frozen),
    RangeError,
    "episode run_id must match the learning context",
  );
});

test("freezeFeedback names items from firstIndex and shares a subtree through its memo", () => {
  expectError(
    () => freezeFeedback([1n, new Map()], new Map(), 3),
    TypeError,
    "feedback[4] must contain exact JSON values",
  );
  const shared = { note: ["kept"] };
  const memo = new Map<object, JsonValue>();
  const [first] = freezeFeedback([shared], memo);
  const [second] = freezeFeedback(new Set([shared]), memo, 1);
  expect(first).toBe(second);
  expect(Object.isFrozen(first)).toBe(true);
  expect(freezeFeedback([shared], new Map())[0]).not.toBe(first);
  expect(Object.isFrozen(freezeFeedback([], new Map()))).toBe(true);
});

test("sameMemoryScope compares static path, memories and occurrence segments", () => {
  const segment = (kind: MemoryOccurrenceKind = "map_item", path = ["flow"], index = 0) =>
    new MemoryOccurrenceSegment(kind, path, index);
  const scope = (staticPath = ["flow", "agent"], memories = ["notes"], occurrencePath = [segment()]) =>
    new MemoryScope(staticPath, { memories, occurrencePath });
  expect(sameMemoryScope(scope(), scope())).toBe(true);
  expect(sameMemoryScope(scope(["flow", "other"]), scope())).toBe(false);
  expect(sameMemoryScope(scope(undefined, ["other"]), scope())).toBe(false);
  expect(sameMemoryScope(scope(undefined, undefined, []), scope())).toBe(false);
  expect(sameMemoryScope(scope(undefined, undefined, [segment("loop_iteration")]), scope())).toBe(false);
  const inner = ["flow", "x", "agent"];
  expect(sameMemoryScope(scope(inner, undefined, [segment(undefined, ["flow", "x"])]), scope(inner))).toBe(false);
  expect(sameMemoryScope(scope(undefined, undefined, [segment(undefined, undefined, 1)]), scope())).toBe(false);
});

test("sameMemoryOperationFailure compares all four fields", () => {
  const failure = (stage = "learn", code = "c", message = "m", retryable = true) =>
    new MemoryOperationFailure(stage, code, message, retryable);
  expect(sameMemoryOperationFailure(failure(), failure())).toBe(true);
  expect(sameMemoryOperationFailure(failure("commit"), failure())).toBe(false);
  expect(sameMemoryOperationFailure(failure(undefined, "d"), failure())).toBe(false);
  expect(sameMemoryOperationFailure(failure(undefined, undefined, "n"), failure())).toBe(false);
  expect(sameMemoryOperationFailure(failure(undefined, undefined, undefined, false), failure())).toBe(false);
});

test("sameMemoryMutation compares the id and the payload by JSON value", () => {
  const payload = { add: [{ id: "lesson", n: 1n, x: 1.5, ok: true, none: null }] };
  expect(sameMemoryMutation(new MemoryMutation("m", payload), new MemoryMutation("m", structuredClone(payload)))).toBe(
    true,
  );
  expect(sameMemoryMutation(new MemoryMutation("m", payload), new MemoryMutation("n", payload))).toBe(false);
  for (const [a, b] of [
    [1n, 1],
    [{ a: 1n }, { a: 1n, b: 1n }],
    [{ a: 1n }, { b: 1n }],
    [[1n], [1n, 2n]],
    [[1n], { 0: 1n }],
    [null, {}],
    ["1", 1n],
    [true, 1n],
    [{ a: [1n] }, { a: [2n] }],
  ] as const) {
    expect(sameMemoryMutation(new MemoryMutation("m", a), new MemoryMutation("m", b))).toBe(false);
  }
});

test("sameMemorySnapshotEnvelope compares the memory key and the snapshot by JSON value", () => {
  const snapshot = () => ({ notes: [{ id: "lesson", n: 1n, x: 1.5, ok: true, none: null }], meta: { v: 1 } });
  const envelope = (memory = "notes", value: JsonObject = snapshot()) => new MemorySnapshotEnvelope(memory, value);
  expect(sameMemorySnapshotEnvelope(envelope(), envelope())).toBe(true);
  expect(sameMemorySnapshotEnvelope(envelope("other"), envelope())).toBe(false);
  for (const [a, b] of [
    [{ deep: { list: [{ n: 1n }] } }, { deep: { list: [{ n: 2n }] } }],
    [{ deep: { list: [{ n: 1n }] } }, { deep: { list: [{ n: 1 }] } }],
    [{ deep: { list: [1] } }, { deep: { list: [true] } }],
    [{ a: 1 }, { a: 1, b: 1 }],
  ] as const) {
    expect(sameMemorySnapshotEnvelope(envelope("notes", a), envelope("notes", b))).toBe(false);
  }
});

test("sameMemorySpec compares key, type, config, owners and eligibility by value", () => {
  const site = (owner = "root", node: string | null = null) => new DeclSite([], [owner], node);
  const spec = (
    key = "notes",
    type = "summary",
    config: JsonObject = { limit: 3, tags: ["a"] },
    owners: DeclSite[] = [site(), site("tool")],
    eligible = false,
  ) => new MemorySpec(key, type, config, owners, eligible);
  expect(sameMemorySpec(spec(), spec())).toBe(true);
  expect(sameMemorySpec(spec("other"), spec())).toBe(false);
  expect(sameMemorySpec(spec(undefined, "episodic"), spec())).toBe(false);
  expect(sameMemorySpec(spec(undefined, undefined, { limit: 4, tags: ["a"] }), spec())).toBe(false);
  expect(sameMemorySpec(spec(undefined, undefined, { limit: 3n, tags: ["a"] }), spec())).toBe(false);
  expect(sameMemorySpec(spec(undefined, undefined, undefined, [site()]), spec())).toBe(false);
  expect(sameMemorySpec(spec(undefined, undefined, undefined, [site(), site("zeta")]), spec())).toBe(false);
  // Eligibility follows the owners, so it differs only with them: one AGENT site against one flow site.
  const agent = spec(undefined, undefined, undefined, [site("root", "writer")], true);
  expect(sameMemorySpec(agent, spec(undefined, undefined, undefined, [site("root", "writer")], true))).toBe(true);
  expect(sameMemorySpec(agent, spec(undefined, undefined, undefined, [site()], false))).toBe(false);
});

test("a setup error keeps an explicit cause and its origin", () => {
  const origin = new OriginRef("memory-policy");
  const cause = new Error("backend failed");
  const error = new MemorySetupError("m", { origin, cause });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe("m");
  expect(error.cause).toBe(cause);
  expect(error.origin).toBe(origin);
  expect("cause" in new MemorySetupError("m", { origin })).toBe(false);
});

test("sameLearningFailure compares memory, operation and failure by value", () => {
  const cause = (code = "c") => new MemoryOperationFailure("learn", code, "m", true);
  const base = new LearningFailure("notes", "learn", cause());
  expect(sameLearningFailure(new LearningFailure("notes", "learn", cause()), base)).toBe(true);
  expect(sameLearningFailure(new LearningFailure("other", "learn", cause()), base)).toBe(false);
  expect(sameLearningFailure(new LearningFailure("notes", "commit", cause()), base)).toBe(false);
  expect(sameLearningFailure(new LearningFailure("notes", "learn", cause("d")), base)).toBe(false);
});

test("samePolicyCommitOutcome compares memory, status, commit id and failure", () => {
  const cause = (code = "c") => new MemoryOperationFailure("commit", code, "m", false);
  const failed = (memory = "notes", commitId = "c1", code = "c") =>
    new PolicyCommitOutcome(memory, "failed", { commitId, failure: cause(code) });
  expect(samePolicyCommitOutcome(failed(), failed())).toBe(true);
  expect(samePolicyCommitOutcome(failed("other"), failed())).toBe(false);
  expect(samePolicyCommitOutcome(failed(undefined, "c2"), failed())).toBe(false);
  expect(samePolicyCommitOutcome(failed(undefined, undefined, "d"), failed())).toBe(false);
  const unknown = new PolicyCommitOutcome("notes", "unknown", { commitId: "c1" });
  expect(samePolicyCommitOutcome(unknown, new PolicyCommitOutcome("notes", "unknown", { commitId: "c1" }))).toBe(true);
  const succeeded = new PolicyCommitOutcome("notes", "succeeded", { commitId: "c1" });
  expect(samePolicyCommitOutcome(unknown, succeeded)).toBe(false);
  expect(samePolicyCommitOutcome(new PolicyCommitOutcome("notes", "unrequested"), unknown)).toBe(false);
  expect(samePolicyCommitOutcome(unknown, failed())).toBe(false);
  expect(samePolicyCommitOutcome(failed(), unknown)).toBe(false);
});

test("sameLearningResult compares status, failures and policies in order", () => {
  const cause = new MemoryOperationFailure("learn", "c", "m", true);
  const failure = (memory: string) => new LearningFailure(memory, "learn", cause);
  const policy = (memory: string) => new PolicyCommitOutcome(memory, "unrequested");
  const result = (status: "failed" | "incomplete" = "failed", failures = ["a", "b"], policies = ["p", "q"]) =>
    new LearningResult(status, { failures: failures.map(failure), policies: policies.map(policy) });
  expect(sameLearningResult(result(), result())).toBe(true);
  expect(sameLearningResult(result("incomplete"), result())).toBe(false);
  expect(sameLearningResult(result(undefined, ["b", "a"]), result())).toBe(false);
  expect(sameLearningResult(result(undefined, ["a"]), result())).toBe(false);
  expect(sameLearningResult(result(undefined, undefined, ["q", "p"]), result())).toBe(false);
  expect(sameLearningResult(result(undefined, undefined, ["p"]), result())).toBe(false);
});
