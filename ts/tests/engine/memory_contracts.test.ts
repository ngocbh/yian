/**
 * Memory identity values, the memory JSON snapshot and the learning result carriers, from
 * `tests/engine/test_memory_contracts.py`:
 * `test_identity_values_are_frozen_structured_products`, `test_memory_scope_has_no_boundary`,
 * `test_identity_values_reject_ambiguous_paths_and_indices`,
 * `test_scope_occurrence_path_must_strictly_contain_the_node`,
 * `test_memory_decls_preserve_origin_and_isolate_canonical_json`,
 * `test_json_contracts_reject_non_exact_or_non_finite_values`, `test_json_contracts_reject_cycles`,
 * `test_scope_carries_a_root_relative_path_and_memory_keys`, the declaration rows of
 * `test_memory_strings_must_have_a_canonical_utf8_representation`, the declaration and mutation
 * thirds of `test_json_scalar_identity_preserves_exact_types_at_every_depth`, the mutation half of
 * `test_context_episode_and_mutation_keep_json_snapshots`, the `MemoryMutation`, `MemoryProposal`,
 * `LearningFailure` and `BackwardFailure` rows of
 * `test_episodes_contexts_mutations_and_failures_name_the_memory_key`,
 * `test_learning_result_keeps_primary_independent_failure_attribution`,
 * `test_policy_commit_outcome_keeps_protocol_identity_and_backend_hints`,
 * `test_learning_completion_is_independent_of_retryability_and_policy_confirmation`,
 * `test_policy_commit_outcome_rejects_inconsistent_optional_fields`,
 * `test_policy_commit_outcome_requires_exact_valid_unicode_identities`, the rows without forgery of
 * `test_learning_result_revalidates_exact_nested_policy_carriers`,
 * `test_memory_errors_expose_typed_fields_and_origin`, the `Memory` half of
 * `test_protocols_accept_narrow_structural_implementations`, the scope row of
 * `test_nested_memory_carriers_reject_forged_subclass_values` and the subclass row of
 * `test_memory_origins_revalidate_subclass_keys_without_losing_identity`, the memory package,
 * `contracts.ts` and `runtime/contracts.ts` rows of `test_frame_types_are_gone` and
 * `test_importing_memory_contracts_does_not_load_heavy_backends`; plus JavaScript's own
 * values (a `"__proto__"` key, a null-prototype object, a repeated subtree, an array hole), the
 * value nesting limit, the occurrence nesting order, the operation failure's field checks
 * (Python pins those through the learning outcome and backward records), `validateMutations`, a
 * mutation's id and payload checks (a payload deeper than the value nesting limit is kept), and
 * each learning carrier's field checks in Python's order.
 *
 * Changed: the JSON tests build a custom-type `MemoryDecl`, as Python's `_decl` does. The non-exact
 * rows are JavaScript's (a `Map`, a `Date`, an own symbol key, a class instance, an array subclass
 * for `_ListSubclass`, a top-level class instance for `_DictSubclass`), each with its exact class
 * and message. The scalar identity test checks that `true`, `1n` and `1` keep their exact JS types
 * at every depth of the snapshot. A memory config nests at most `maxValueNesting` levels (memory
 * JSON nesting doesn't exist here), with that limit's message. The ambiguous-index rows add `1.5`,
 * `1n` and a `"MAP_ITEM"` kind; the non-array paths are `Set`s (Python passes lists). A
 * scope's lack of a boundary is checked on the module's exports and the scope's fields (TS takes
 * extra constructor arguments silently). The scope row of the forged-carrier test passes a plain
 * object shaped like a segment. The memory protocol is checked with `isMemory` on plain-object
 * fakes (Python's `isinstance` on a runtime-checkable protocol); a session has no context-manager
 * methods, so the test calls `close()`. The backend-import test reads `memory/index.ts` and checks
 * that it imports only `./contracts.ts` (the layer-direction test pins `contracts.ts` to
 * `diagnostics` and `defaults`), where Python imports the package in a fresh interpreter. Python's
 * `ValueError` is `RangeError`; a frozen field throws `TypeError` where Python raises
 * `FrozenInstanceError`. A mutation's payload is its deep-frozen snapshot, so editing it throws
 * `TypeError` (Python returns a fresh copy); Python's int payloads are `bigint`s. The learning
 * carriers keep their nested carriers by reference: where Python asserts a copied policy `is not`
 * the one passed in, the test checks that it is the same frozen value. The policy outcome's
 * optional fields are an options object (`{ commitId, failure }`), and the learning result's arrays
 * are `{ failures, policies }`. The identity rows check only the error class, as Python does.
 *
 * Not ported: the tuple and bytes rows and the list-for-tuple path row (TS has only arrays and no
 * bytes in a config); `test_memory_json_rejects_nesting_beyond_the_memory_limit` (its carriers
 * aren't ported; the nesting test below replaces it for declarations); the forged rows of
 * `test_nested_memory_carriers_reject_forged_subclass_values` and
 * `test_memory_origins_revalidate_subclass_keys_without_losing_identity` (a JS subclass can't skip
 * its base constructor, so no carrier holds an unvalidated segment or origin);
 * `test_memory_operation_error_survives_copy_and_pickle` (no `copy` or `pickle`); dataclass `==`
 * and `hash` (Bun's `toEqual` replaces equality); the `MemorySession` row of
 * `test_frame_types_are_gone` (`MemorySession` is an interface, with no runtime members to check);
 * the forged rows of
 * `test_learning_result_revalidates_exact_nested_policy_carriers` (Python's `object.__setattr__`
 * on a frozen dataclass; a TS carrier is frozen by its constructor, so none holds an unchecked
 * `commitId` or `retryable`).
 *
 * Left for later: the recall, learning evidence and conversation tests, with their carriers
 * (`MemoryEpisode`, `LearnContext` and the rest); and the `recall`, `learn` and `commit` rows of
 * `test_protocols_accept_narrow_structural_implementations`.
 */

import { afterEach, expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import * as contracts from "../../src/memory/contracts.ts";
import {
  DeclSite,
  isMemory,
  type JsonObject,
  LearningFailure,
  LearningResult,
  type Memory,
  MemoryDecl,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryOperationError,
  MemoryOperationFailure,
  MemoryScope,
  type MemorySession,
  MemorySetupError,
  PolicyCommitOutcome,
  type PolicyCommitStatus,
  validateMutations,
} from "../../src/memory/contracts.ts";
import * as memoryPackage from "../../src/memory/index.ts";
import * as runtimeContracts from "../../src/runtime/contracts.ts";
import { BackwardFailure, MemoryProposal } from "../../src/runtime/contracts.ts";
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

// test_json_scalar_identity_preserves_exact_types_at_every_depth (the declaration half)
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

/** Build a plain-object memory whose sessions record the snapshot they opened with. */
function fakeMemory(): Memory & { resets: number; closes: number; opened: (JsonObject | null)[] } {
  return {
    resets: 0,
    closes: 0,
    opened: [],
    openRun({ snapshot = null } = {}) {
      this.opened.push(snapshot);
      return {
        snapshot: { namespace: "test" },
        close: () => {
          this.closes++;
        },
      };
    },
    reset() {
      this.resets++;
    },
  };
}

// test_protocols_accept_narrow_structural_implementations (the Memory half)
test("the memory protocol accepts a narrow structural implementation", () => {
  const memory = fakeMemory();
  expect(isMemory(memory)).toBe(true);
  for (const value of [{}, null, undefined, "memory", { openRun() {} }, { openRun: 1, reset() {} }]) {
    expect(isMemory(value)).toBe(false);
  }
  const snapshot = { namespace: "test" };
  const session: MemorySession = memory.openRun({ snapshot });
  memory.openRun();
  expect(memory.opened[0]).toBe(snapshot);
  expect(memory.opened[1]).toBeNull();
  expect(session.snapshot).toEqual({ namespace: "test" });
  session.close();
  expect(memory.closes).toBe(1);
  memory.reset();
  expect(memory.resets).toBe(1);
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

// test_context_episode_and_mutation_keep_json_snapshots (the mutation half)
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

// test_json_scalar_identity_preserves_exact_types_at_every_depth (the mutation third)
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
