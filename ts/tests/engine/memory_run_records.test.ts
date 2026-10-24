/**
 * Run-record contracts for source-neutral memory evidence: a port of
 * `tests/engine/test_memory_run_records.py`. Ported: the evidence carriers,
 * `test_source_neutral_carriers_are_defensive_and_frozen` and
 * `test_evidence_requires_coherent_prompts_and_ordered_scope_sections`, the evidence line of
 * `test_memory_scope_requires_unique_keys_and_nested_occurrences`, and
 * `test_episode_accepts_binding_failure_and_typed_visible_material` (its keyword arguments are the
 * episode's options object). Python's `ValueError` is a
 * `RangeError` here; a frozen field throws `TypeError` where Python raises `FrozenInstanceError`.
 *
 * TS-only: one row per carrier and evidence message with its class, in Python's check order; an
 * exact-class row for each nested carrier (a subclass is refused); the number rule of the evidence
 * JSON (`value`, `arguments`, `boundInputs`): `±Infinity` kept, `-0` read back as `0`, NaN refused;
 * bound inputs deeper than the value nesting limit are kept; evidence keeps its fields, and the
 * section and interaction arrays are copied, frozen and keep their order and their carriers by
 * reference.
 *
 * Changed (TS yian):
 * - The carriers hold deep-frozen copies, with no copy on read: a reader's edit of `value`,
 *   `arguments`, `boundInputs` or a snapshot throws a `TypeError` where Python changes a private
 *   copy. The caller's later edits don't show, as in Python.
 * - The evidence JSON keeps `±Infinity` (decision 34: bound inputs are yian values) and has no
 *   memory nesting limit (bound inputs are values, bounded where they're made).
 * - Sections and interactions are arrays, not tuples; a missing value is `null`, not `None`.
 *
 * Ported, the run-record constructors: `test_run_started_distinguishes_disabled_and_enabled_empty_memory`,
 * `test_run_started_requires_a_canonical_table_and_matching_snapshots` and the constructor rows of
 * `test_run_started_records_a_key_table_and_stacks` (each without its wire round trip),
 * `test_records_reject_forged_nested_carrier_subclasses`.
 *
 * TS-only (run records): the memory table, stacks and snapshots are held as frozen arrays whose
 * items are the given carriers, unchanged by the caller's later edits; a table, stack or snapshot
 * item of another class (a subclass too) is refused with its message; an unknown stack key is
 * named in the message (`["missing"]`).
 *
 * Changed (run records):
 * - The memory table, stacks and snapshots are arrays, so the rejected container is a `Set`
 *   (Python passes a list) and the message says "arrays" where Python says "tuples".
 * - `test_records_reject_forged_nested_carrier_subclasses` constructs subclass instances (TS has
 *   no forged dataclasses); the evidence message says `…or null` where Python says `…or None`.
 *
 * Covered elsewhere: the scope rows of `test_memory_scope_requires_unique_keys_and_nested_occurrences`
 * by `memory_contracts.test.ts`; `test_memory_spec_config_rejects_private_keys` by
 * `memory_decl_shape.test.ts`.
 *
 * Not ported: `test_nested_frozen_carriers_are_revalidated_before_recording`,
 * `test_codec_revalidates_carriers_mutated_after_record_construction` and
 * `test_codec_replays_outer_record_invariants_after_forging` (they forge frozen carriers with
 * `object.__setattr__`, which a frozen TS carrier refuses).
 *
 * Ported, the outcome wire: `test_all_outcome_records_round_trip_closed_evidence`,
 * `test_existing_outcome_records_round_trip_with_no_evidence`,
 * `test_memory_scope_wire_has_no_boundary_field`, the evidence rows of
 * `test_schema_four_rejects_old_schema_and_nested_extra_or_unknown_fields` (an extra
 * `evidence.scope` field, the unknown interaction kind `provider_trace`, an extra `provider`) and
 * the evidence half of `test_memory_payloads_name_keys_on_the_wire`, each rejection with its
 * class and message.
 *
 * Ported, the run-start and memory-commit wire: `test_public_token_policy_options_round_trip_durably`,
 * `test_snapshot_envelope_preserves_opaque_public_memory_state`,
 * `test_run_started_uses_learnable_as_its_only_wire_retention_field`, the wire rows of
 * `test_run_started_records_a_key_table_and_stacks` (with the snapshots' wire and a round trip),
 * `test_every_memory_commit_record_round_trips`,
 * `test_memory_commit_wire_requires_an_explicit_positive_attempt_slot`, the mutation half of
 * `test_proposals_preserve_order_and_reject_duplicate_ids`, the schema row of
 * `test_schema_four_rejects_old_schema_and_nested_extra_or_unknown_fields`,
 * `test_schema_four_rejects_origin_and_extra_memory_operation_fields`,
 * `test_schema_four_rejects_plural_private_memory_config` and the commit row of
 * `test_memory_payloads_name_keys_on_the_wire`, each rejection with its class and message (the
 * private-key messages are `MemorySpec`'s).
 *
 * TS-only (run-start and commit wire): rows for a memory key, config, owner, eligibility flag,
 * snapshot, stack, mutation and memory operation failure of the wrong shape, each with its message;
 * memory JSON holding `-0` (a snapshot, a spec config and a mutation payload, which memory's carriers
 * keep) is written as `0.0` and read back as `0`.
 *
 * Changed (run-start and commit wire): `test_run_log_copying_isolated_from_new_carrier_accessors`:
 * a decoded mutation payload is frozen, so a reader's edit throws `TypeError` (Python edits a
 * private copy).
 *
 * Ported, the backward wire: the backward-ready half of `test_memory_payloads_name_keys_on_the_wire`
 * (its extra `frame_ref` row with its class and message) and the `MemoryProposal` line of
 * `test_proposals_preserve_order_and_reject_duplicate_ids`.
 */

import { expect, test } from "bun:test";

import { current } from "../../src/defaults.ts";
import { Failure } from "../../src/diagnostics.ts";
import {
  DeclSite,
  EvaluationEvidence,
  type JsonObject,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOccurrenceSegment,
  MemoryOperationFailure,
  MemoryScope,
  MemorySection,
  MemorySnapshotEnvelope,
  MemorySpec,
  MemoryStack,
  ModelTextInteraction,
  ModelValueInteraction,
  ToolCallInteraction,
  ToolResultInteraction,
} from "../../src/memory/contracts.ts";
import { BackwardResult } from "../../src/nodes/base.ts";
import { MemoryProposal } from "../../src/runtime/contracts.ts";
import {
  BackwardReadyRecord,
  decodeRecord,
  encodeRecord,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeSucceededRecord,
  RecordEnvelope,
  RunLogSchemaError,
  type RunRecord,
  type RunStartedFields,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";

type ErrorClass = typeof TypeError | typeof RangeError | typeof RunLogSchemaError;

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: ErrorClass, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

const SCOPE = new MemoryScope(["flow", "agent"], { memories: ["notes", "reflective"] });

test("source-neutral carriers are defensive and frozen", () => {
  const snapshotSource = { lessons: [{ text: "keep" }] };
  const valueSource = { nested: [1n] };
  const argumentsSource = { query: ["x"] };
  const inputsSource = { question: ["hello"] };
  const snapshot = new MemorySnapshotEnvelope("reflective", snapshotSource);
  const value = new ModelValueInteraction(valueSource);
  const call = new ToolCallInteraction("search", argumentsSource);
  const evidence = new EvaluationEvidence(
    new MemoryScope(["flow", "agent"], { memories: ["reflective"] }),
    inputsSource,
    "base",
    "prompt",
    [new MemorySection("reflective", "memory")],
    [value, call],
  );

  snapshotSource.lessons[0]!.text = "changed";
  valueSource.nested.push(2n);
  argumentsSource.query.push("y");
  inputsSource.question.push("changed");
  expect(() => (snapshot.snapshot.lessons as unknown[]).splice(0)).toThrow(TypeError);
  expect(() => ((value.value as JsonObject).nested as unknown[]).push(3n)).toThrow(TypeError);
  expect(() => (call.arguments.query as unknown[]).push("z")).toThrow(TypeError);
  expect(() => (evidence.boundInputs!.question as unknown[]).push("again")).toThrow(TypeError);

  expect(snapshot.snapshot).toEqual({ lessons: [{ text: "keep" }] });
  expect(value.value).toEqual({ nested: [1n] });
  expect(call.arguments).toEqual({ query: ["x"] });
  expect(evidence.boundInputs).toEqual({ question: ["hello"] });
  expect(() => {
    (evidence as { prompt: string | null }).prompt = "changed";
  }).toThrow(TypeError);
  expect(evidence.prompt).toBe("prompt");
});

test("evidence requires coherent prompts and ordered scope sections", () => {
  expectError(
    () => new EvaluationEvidence(SCOPE, {}, "base", null),
    RangeError,
    "base_system_prompt and prompt must both be present or absent",
  );
  expectError(
    () => new EvaluationEvidence(SCOPE, {}, "base", "prompt", [new MemorySection("other", "x")]),
    RangeError,
    "memory_sections memories must appear in scope.memories",
  );
  expectError(
    () =>
      new EvaluationEvidence(SCOPE, {}, "base", "prompt", [
        new MemorySection("notes", "a"),
        new MemorySection("notes", "b"),
      ]),
    RangeError,
    "memory_sections memory values must be unique",
  );
  expectError(
    () =>
      new EvaluationEvidence(SCOPE, {}, "base", "prompt", [
        new MemorySection("reflective", "a"),
        new MemorySection("notes", "b"),
      ]),
    RangeError,
    "memory_sections must follow scope.memories order",
  );
  expectError(
    () => new EvaluationEvidence(SCOPE, {}, "base", "prompt", [], [{} as never]),
    TypeError,
    "interactions items must be VisibleInteraction values",
  );

  const nonAgent = new EvaluationEvidence(SCOPE, null, null, null);
  expect(nonAgent.boundInputs).toBeNull();
  expect(nonAgent.memorySections).toEqual([]);
  expect(nonAgent.interactions).toEqual([]);
});

// The evidence line of test_memory_scope_requires_unique_keys_and_nested_occurrences.
test("evidence for a scope without memories has no sections", () => {
  expect(new EvaluationEvidence(new MemoryScope(["neutral"]), {}, "base", "prompt").memorySections).toEqual([]);
});

test("evidence keeps its fields and the sections and interactions in order", () => {
  const sections = [new MemorySection("notes", "a"), new MemorySection("reflective", "")];
  const interactions = [
    new ModelTextInteraction("hi"),
    new ModelValueInteraction(null),
    new ToolCallInteraction("search", {}),
    new ToolResultInteraction("search", ""),
  ];
  const evidence = new EvaluationEvidence(SCOPE, {}, "", "", sections, interactions);
  const kept = [...interactions];
  sections.pop();
  interactions.pop();
  expect(evidence.scope).toBe(SCOPE);
  expect([evidence.baseSystemPrompt, evidence.prompt]).toEqual(["", ""]);
  expect(evidence.memorySections.map((section) => section.memory)).toEqual(["notes", "reflective"]);
  expect(evidence.interactions).toEqual(kept);
  expect(evidence.interactions[3]).toBe(kept[3]!);
  expect(Object.isFrozen(evidence.memorySections)).toBe(true);
  expect(Object.isFrozen(evidence.interactions)).toBe(true);
  expect(new EvaluationEvidence(null, null, null, null).scope).toBeNull();
});

class SubScope extends MemoryScope {}
class SubSection extends MemorySection {}
class SubText extends ModelTextInteraction {}

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "a section memory that isn't a string",
    () => new MemorySection(1 as never, "x"),
    TypeError,
    "memory must be a string",
  ],
  ["an empty section memory", () => new MemorySection("", "x"), RangeError, "memory must be non-empty"],
  [
    "a section text that isn't a string",
    () => new MemorySection("notes", null as never),
    TypeError,
    "text must be a string",
  ],
  ["a model text that isn't a string", () => new ModelTextInteraction(1 as never), TypeError, "text must be a string"],
  [
    "a model value that isn't JSON",
    () => new ModelValueInteraction(undefined),
    TypeError,
    "value must contain exact JSON values",
  ],
  [
    "a model value holding NaN",
    () => new ModelValueInteraction([NaN]),
    RangeError,
    "value[0] JSON number must not be NaN",
  ],
  [
    "a tool call name that isn't a string",
    () => new ToolCallInteraction(1 as never, {}),
    TypeError,
    "name must be a string",
  ],
  ["an empty tool call name", () => new ToolCallInteraction("", {}), RangeError, "name must be non-empty"],
  [
    "tool call arguments that aren't an object",
    () => new ToolCallInteraction("search", [] as never),
    TypeError,
    "arguments must be an exact JSON object",
  ],
  [
    "tool call arguments holding NaN",
    () => new ToolCallInteraction("search", { x: NaN }),
    RangeError,
    "arguments.x JSON number must not be NaN",
  ],
  ["an empty tool result name", () => new ToolResultInteraction("", "x"), RangeError, "name must be non-empty"],
  [
    "a tool result text that isn't a string",
    () => new ToolResultInteraction("search", 1 as never),
    TypeError,
    "text must be a string",
  ],
  [
    "a scope that isn't a MemoryScope",
    () => new EvaluationEvidence({ staticPath: ["a"], memories: [] } as never, null, null, null),
    TypeError,
    "scope must be a MemoryScope or null",
  ],
  [
    "a MemoryScope subclass",
    () => new EvaluationEvidence(new SubScope(["a"]), null, null, null),
    TypeError,
    "scope must be a MemoryScope or null",
  ],
  [
    "bound inputs that aren't an object",
    () => new EvaluationEvidence(null, [] as never, null, null),
    TypeError,
    "bound_inputs must be an exact JSON object or null",
  ],
  [
    "bound inputs holding a non-JSON value",
    () => new EvaluationEvidence(null, { x: undefined }, null, null),
    TypeError,
    "bound_inputs.x must contain exact JSON values",
  ],
  [
    "bound inputs holding NaN",
    () => new EvaluationEvidence(null, { x: [NaN] }, null, null),
    RangeError,
    "bound_inputs.x[0] JSON number must not be NaN",
  ],
  [
    "a prompt without a base system prompt",
    () => new EvaluationEvidence(null, null, null, "prompt"),
    RangeError,
    "base_system_prompt and prompt must both be present or absent",
  ],
  [
    "a base system prompt that isn't a string",
    () => new EvaluationEvidence(null, null, 1 as never, "prompt"),
    TypeError,
    "base_system_prompt must be a string",
  ],
  [
    "a prompt that isn't a string",
    () => new EvaluationEvidence(null, null, "base", 1 as never),
    TypeError,
    "prompt must be a string",
  ],
  [
    "sections that aren't an array",
    () => new EvaluationEvidence(SCOPE, null, null, null, new Set() as never),
    TypeError,
    "memory_sections must be an array",
  ],
  [
    "a section that isn't a MemorySection",
    () => new EvaluationEvidence(SCOPE, null, null, null, [{ memory: "notes", text: "x" } as never]),
    TypeError,
    "memory_sections items must be MemorySection values",
  ],
  [
    "a MemorySection subclass",
    () => new EvaluationEvidence(SCOPE, null, null, null, [new SubSection("notes", "x")]),
    TypeError,
    "memory_sections items must be MemorySection values",
  ],
  [
    "a repeated section memory",
    () =>
      new EvaluationEvidence(SCOPE, null, null, null, [
        new MemorySection("notes", "a"),
        new MemorySection("notes", "b"),
      ]),
    RangeError,
    "memory_sections memory values must be unique",
  ],
  [
    "a section outside the scope",
    () => new EvaluationEvidence(SCOPE, null, null, null, [new MemorySection("other", "x")]),
    RangeError,
    "memory_sections memories must appear in scope.memories",
  ],
  [
    "sections out of scope order",
    () =>
      new EvaluationEvidence(SCOPE, null, null, null, [
        new MemorySection("reflective", "a"),
        new MemorySection("notes", "b"),
      ]),
    RangeError,
    "memory_sections must follow scope.memories order",
  ],
  [
    "interactions that aren't an array",
    () => new EvaluationEvidence(SCOPE, null, null, null, [], "x" as never),
    TypeError,
    "interactions must be an array",
  ],
  [
    "an interaction that isn't a VisibleInteraction",
    () => new EvaluationEvidence(SCOPE, null, null, null, [], [new MemorySection("notes", "x") as never]),
    TypeError,
    "interactions items must be VisibleInteraction values",
  ],
  [
    "a visible interaction subclass",
    () => new EvaluationEvidence(SCOPE, null, null, null, [], [new SubText("x")]),
    TypeError,
    "interactions items must be VisibleInteraction values",
  ],
])("the evidence carriers reject %s", (_label, build, cls, message) => {
  expectError(build, cls, message);
});

test("the evidence JSON keeps infinities and reads negative zero back as zero", () => {
  const evidence = new EvaluationEvidence(null, { up: Infinity, down: [-Infinity], zero: -0 }, null, null);
  expect(evidence.boundInputs).toEqual({ down: [-Infinity], up: Infinity, zero: 0 });
  expect(Object.is(evidence.boundInputs!.zero, 0)).toBe(true);
  const value = new ModelValueInteraction([Infinity, -Infinity, -0]);
  expect(value.value).toEqual([Infinity, -Infinity, 0]);
  expect(Object.is((value.value as number[])[2], 0)).toBe(true);
  expect(Object.is(new ModelValueInteraction(-0).value, 0)).toBe(true);
  const call = new ToolCallInteraction("search", { limit: Infinity, offset: -0 });
  expect(call.arguments).toEqual({ limit: Infinity, offset: 0 });
  expect(Object.is(call.arguments.offset, 0)).toBe(true);
});

test("bound inputs deeper than the value nesting limit are kept", () => {
  let deep: JsonObject = {};
  for (let i = 0; i <= current().maxValueNesting; i++) deep = { d: deep };
  expect(new EvaluationEvidence(null, deep, null, null).boundInputs).toEqual(deep);
});

// test_episode_accepts_binding_failure_and_typed_visible_material
test("an episode accepts a binding failure and typed visible material", () => {
  const episode = new MemoryEpisode("run", [], ["flow", "agent"], MemoryEpisodeStatus.FAILED, null, {
    failure: { code: "binding.evaluate" },
    baseSystemPrompt: null,
    prompt: null,
    memorySections: [new MemorySection("reflective", "memory")],
    interactions: [new ModelTextInteraction("visible")],
  });
  expect(episode.inputs).toBeNull();
  expect(episode.memorySections).toEqual([new MemorySection("reflective", "memory")]);
  expect(episode.interactions).toEqual([new ModelTextInteraction("visible")]);
});

const ROOT = new DeclSite([], ["root"], null);
const AGENT = new DeclSite([], ["root"], "agent");

/** Return one key-table entry; an AGENT owner makes it conversation-eligible. */
function spec(key = "reflective", config: JsonObject = {}, owner: DeclSite = AGENT): MemorySpec {
  return new MemorySpec(key, key, config, [owner]);
}

/** Return a run-start record with the given memory fields. */
function started(
  fields: Pick<Partial<RunStartedFields>, "memorySnapshots" | "learnable" | "memories" | "memoryStacks"> = {},
): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    ...fields,
  });
}

const ALL_OR_NONE = "memories, memory_stacks and memory_snapshots must all be recorded or all be absent";

test("run started distinguishes disabled and enabled empty memory", () => {
  const disabled = started();
  const enabled = started({ memories: [], memoryStacks: [], memorySnapshots: [] });
  const learning = started({ memories: [], memoryStacks: [], memorySnapshots: [], learnable: true });

  expect(disabled.memories).toBeNull();
  expect(disabled.memoryStacks).toBeNull();
  expect(disabled.memorySnapshots).toBeNull();
  expect(enabled.memories).toEqual([]);
  expect(enabled.memoryStacks).toEqual([]);
  expect(enabled.memorySnapshots).toEqual([]);
  expect(learning.learnable).toBe(true);
  expect("memoryLayout" in enabled).toBe(false);

  for (const partial of [
    { memories: [], memoryStacks: [], memorySnapshots: null },
    { memories: [], memoryStacks: null, memorySnapshots: [] },
    { memories: null, memoryStacks: [], memorySnapshots: [] },
  ]) {
    expectError(() => started(partial), RangeError, ALL_OR_NONE);
  }
  expectError(
    () => started({ memories: new Set() as never, memoryStacks: [], memorySnapshots: [] }),
    TypeError,
    "memories, memory_stacks and memory_snapshots must be arrays when recorded",
  );
  expect(started({ learnable: true }).learnable).toBe(true);
  expectError(
    () => started({ memories: [], memoryStacks: [], memorySnapshots: [], learnable: 1 as never }),
    TypeError,
    "learnable must be a boolean",
  );
});

test("run started requires a canonical table and matching snapshots", () => {
  const memories = [spec("profile", { mode: "flow" }, ROOT), spec("reflective", { mode: "agent" })];
  const snapshots = [
    new MemorySnapshotEnvelope("profile", { state: 1n }),
    new MemorySnapshotEnvelope("reflective", { state: 2n }),
  ];
  const stacks = [new MemoryStack(["agent"], ["profile", "reflective"]), new MemoryStack(["plain"], ["profile"])];
  const record = started({ memories, memoryStacks: stacks, memorySnapshots: snapshots });

  expect(record.memories).toEqual(memories);
  expect(record.memoryStacks).toEqual(stacks);
  expect(record.memorySnapshots).toEqual(snapshots);

  expectError(
    () => started({ memories: [...memories].reverse(), memoryStacks: stacks, memorySnapshots: snapshots }),
    RangeError,
    "memories keys must be sorted and unique",
  );
  expectError(
    () => started({ memories: [memories[0]!, memories[0]!], memoryStacks: [], memorySnapshots: snapshots.slice(0, 1) }),
    RangeError,
    "memories keys must be sorted and unique",
  );
  expectError(
    () => started({ memories, memoryStacks: stacks, memorySnapshots: [...snapshots].reverse() }),
    RangeError,
    "memory_snapshots memory keys must be sorted and unique",
  );
  expectError(
    () => started({ memories, memoryStacks: stacks, memorySnapshots: snapshots.slice(0, 1) }),
    RangeError,
    "memory_snapshots memory keys must equal memories keys",
  );
  expectError(
    () => started({ memories, memoryStacks: [...stacks].reverse(), memorySnapshots: snapshots }),
    RangeError,
    "memory_stacks locators must be sorted and unique",
  );
  expectError(
    () => started({ memories, memoryStacks: [new MemoryStack(["agent"], ["missing"])], memorySnapshots: snapshots }),
    RangeError,
    'memory_stacks name unknown memory keys: ["missing"]',
  );
});

// The constructor rows of test_run_started_records_a_key_table_and_stacks.
test("run started records a key table and stacks", () => {
  const memories = [spec("profile", { mode: "flow" }, ROOT), spec("reflective", { mode: "agent" })];
  const stacks = [new MemoryStack(["agent"], ["profile", "reflective"])];
  const snapshots = [
    new MemorySnapshotEnvelope("profile", { state: 1n }),
    new MemorySnapshotEnvelope("reflective", { state: 2n }),
  ];
  const record = started({ memories, memoryStacks: stacks, memorySnapshots: snapshots });
  memories.pop();
  stacks.pop();
  snapshots.pop();

  expect(record.memories!.map((item) => [item.key, item.config])).toEqual([
    ["profile", { mode: "flow" }],
    ["reflective", { mode: "agent" }],
  ]);
  expect(record.memories![0]).toBe(memories[0]!);
  expect(record.memoryStacks!.map((stack) => [stack.locator, stack.memories])).toEqual([
    [["agent"], ["profile", "reflective"]],
  ]);
  expect(record.memorySnapshots!.map((snapshot) => snapshot.memory)).toEqual(["profile", "reflective"]);
  expect(Object.isFrozen(record.memories)).toBe(true);
  expect(Object.isFrozen(record.memoryStacks)).toBe(true);
  expect(Object.isFrozen(record.memorySnapshots)).toBe(true);
});

class SubSpec extends MemorySpec {}
class SubStack extends MemoryStack {}
class SubSnapshot extends MemorySnapshotEnvelope {}

test.each<[string, Pick<RunStartedFields, "memories" | "memoryStacks" | "memorySnapshots">, string]>([
  [
    "a table entry that isn't a MemorySpec",
    { memories: [{ key: "notes" } as never], memoryStacks: [], memorySnapshots: [] },
    "memories items must be MemorySpec values",
  ],
  [
    "a MemorySpec subclass",
    { memories: [new SubSpec("notes", "notes", {}, [ROOT])], memoryStacks: [], memorySnapshots: [] },
    "memories items must be MemorySpec values",
  ],
  [
    "a MemorySnapshotEnvelope subclass",
    { memories: [], memoryStacks: [], memorySnapshots: [new SubSnapshot("notes", {})] },
    "memory_snapshots items must be MemorySnapshotEnvelope values",
  ],
  [
    "a MemoryStack subclass",
    { memories: [], memoryStacks: [new SubStack(["agent"], [])], memorySnapshots: [] },
    "memory_stacks items must be MemoryStack values",
  ],
])("run started rejects %s", (_label, fields, message) => {
  expectError(() => started(fields), TypeError, message);
});

class ForgedEvidence extends EvaluationEvidence {}
class ForgedMutation extends MemoryMutation {}

test("records reject forged nested carrier subclasses", () => {
  const forgedEvidence = new ForgedEvidence(SCOPE, { question: ["hello"] }, "base", "answer the question");

  expectError(
    () => new NodeSucceededRecord("agent", 1, { evidence: forgedEvidence }),
    TypeError,
    "evidence must be an EvaluationEvidence or null",
  );
  expectError(
    () => new MemoryCommitRequestedRecord("reflective", "commit", [new ForgedMutation("m1", {})], 1),
    TypeError,
    "mutations items must be MemoryMutation values",
  );
});

// --------------------------------------------------------------------------------------- //
// Outcome evidence on the run log
// --------------------------------------------------------------------------------------- //

const DIGEST = "b".repeat(64);

/** Closed evidence with a scope, prompts, two sections and one interaction of each kind. */
function evidence(): EvaluationEvidence {
  return new EvaluationEvidence(
    new MemoryScope(["flow", "agent"], {
      memories: ["notes", "reflective"],
      occurrencePath: [new MemoryOccurrenceSegment("map_item", ["flow"], 0)],
    }),
    { question: ["hello"] },
    "base",
    "answer the question",
    [new MemorySection("notes", "flow memory"), new MemorySection("reflective", "agent memory")],
    [
      new ModelTextInteraction("draft"),
      new ModelValueInteraction({ answer: "done" }),
      new ToolCallInteraction("lookup", { key: "value" }),
      new ToolResultInteraction("lookup", "result"),
    ],
  );
}

/** Encode `record` at seq 0 and decode it; the line must be stable. Return the decoded record. */
function roundTrip(record: RunRecord): RunRecord {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 0, record);
  const encoded = encodeRecord(envelope);
  const decoded = decodeRecord(encoded);
  expect(encodeRecord(decoded)).toBe(encoded);
  expect(decoded).toEqual(envelope);
  return decoded.record;
}

/** The decoded JSON of `record`'s line at seq 0, ints as `bigint`s, as a mutable object. */
function wireOf(record: RunRecord): Record<string, any> {
  return parseJson(encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "run", 0, record)), "codec") as Record<string, any>;
}

const FAILED = new Failure({ code: "node.execute", message: "failed", nodeId: "agent" });

test.each<[string, () => RunRecord]>([
  ["NodeSucceededRecord", () => new NodeSucceededRecord("agent", 1, { evidence: evidence() })],
  ["NodeRoutedRecord", () => new NodeRoutedRecord("agent", "yes", { evidence: evidence() })],
  ["NodeExpandedRecord", () => new NodeExpandedRecord("agent", [], DIGEST, { evidence: evidence() })],
  ["NodeFailedRecord", () => new NodeFailedRecord(FAILED, { evidence: evidence() })],
])("all outcome records round trip closed evidence: %s", (_name, build) => {
  const record = build();
  expect(roundTrip(record)).toEqual(record);
});

test("existing outcome records round trip with no evidence", () => {
  for (const record of [
    new NodeSucceededRecord("agent", 1),
    new NodeRoutedRecord("agent", "yes"),
    new NodeExpandedRecord("agent", [], DIGEST),
    new NodeFailedRecord(FAILED),
  ]) {
    expect((roundTrip(record) as NodeSucceededRecord).evidence).toBeNull();
  }
});

const EVIDENCE_LABEL = "node_succeeded payload evidence";

test.each<[string, (evidence: Record<string, any>) => void, string]>([
  [
    "an extra scope field",
    (wire) => {
      wire.scope.extra = true;
    },
    `invalid ${EVIDENCE_LABEL} scope fields; missing=[], extra=["extra"]`,
  ],
  [
    "an unknown interaction kind",
    (wire) => {
      wire.interactions[0].kind = "provider_trace";
    },
    "unknown visible interaction kind: 'provider_trace'",
  ],
  [
    "an extra interaction field",
    (wire) => {
      wire.interactions[0].provider = "x";
    },
    `invalid ${EVIDENCE_LABEL} interactions item fields; missing=[], extra=["provider"]`,
  ],
])("the evidence wire rejects %s", (_name, mutate, message) => {
  const wire = wireOf(new NodeSucceededRecord("agent", 1, { evidence: evidence() }));
  mutate(wire.payload.evidence);
  expectError(() => decodeRecord(logJson(wire)), RangeError, message);
});

test("the memory scope wire has no boundary field", () => {
  const wire = wireOf(new NodeSucceededRecord("agent", 1, { evidence: evidence() }));
  const scope = wire.payload.evidence.scope;
  expect(Object.keys(scope).sort()).toEqual(["memories", "occurrence_path", "static_path"]);
  scope.boundary = null;
  expectError(
    () => decodeRecord(logJson(wire)),
    RangeError,
    `invalid ${EVIDENCE_LABEL} scope fields; missing=[], extra=["boundary"]`,
  );
});

test("memory payloads name keys on the wire", () => {
  const record = new NodeSucceededRecord("agent", 1, { evidence: evidence() });
  const wire = JSON.parse(encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "run", 0, record))).payload.evidence;
  expect(wire.scope).toEqual({
    static_path: ["flow", "agent"],
    memories: ["notes", "reflective"],
    occurrence_path: [{ kind: "map_item", static_path: ["flow"], index: 0 }],
  });
  expect(wire.memory_sections).toEqual([
    { memory: "notes", text: "flow memory" },
    { memory: "reflective", text: "agent memory" },
  ]);
  expect(roundTrip(record)).toEqual(record);
});

// --------------------------------------------------------------------------------------- //
// Run start and memory commits on the run log
// --------------------------------------------------------------------------------------- //

test("public token policy options round trip durably", () => {
  const config = { stop_token: "semantic-stop-marker", continuationToken: "semantic-continuation-marker" };
  const record = started({
    memories: [spec("reflective", config)],
    memoryStacks: [],
    memorySnapshots: [new MemorySnapshotEnvelope("reflective", { state: [] })],
  });

  const encoded = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "run", 0, record));
  const decoded = decodeRecord(encoded).record;

  expect(encoded).toContain("semantic-stop-marker");
  expect(encoded).toContain("semantic-continuation-marker");
  expect(decoded.constructor).toBe(RunStartedRecord);
  expect((decoded as RunStartedRecord).memories![0]!.config).toEqual(config);
});

test("a snapshot envelope preserves opaque public memory state", () => {
  const publicState = {
    lessons: [
      { text: "Never place an api_key in durable memory configuration.", tags: ["security", "configuration"] },
    ],
    profile: { preferred_runtime: "local" },
  };
  expect(new MemorySnapshotEnvelope("reflective", publicState).snapshot).toEqual(publicState);
});

test("run started uses learnable as its only wire retention field", () => {
  const wire = wireOf(started({ learnable: true }));
  expect(wire.payload.learnable).toBe(true);
  expect("learn" in wire.payload).toBe(false);

  wire.payload.learn = wire.payload.learnable;
  delete wire.payload.learnable;
  expectError(
    () => decodeRecord(logJson(wire)),
    RangeError,
    'invalid run_started payload fields; missing=["learnable"], extra=["learn"]',
  );
});

// The wire rows of test_run_started_records_a_key_table_and_stacks.
test("run started writes its key table and stacks on the wire", () => {
  const memories = [spec("profile", { mode: "flow" }, ROOT), spec("reflective", { mode: "agent" })];
  const record = started({
    memories,
    memoryStacks: [new MemoryStack(["agent"], ["profile", "reflective"])],
    memorySnapshots: [
      new MemorySnapshotEnvelope("profile", { state: 1n }),
      new MemorySnapshotEnvelope("reflective", { state: 2n }),
    ],
  });
  const wire = wireOf(record);

  expect("memory_layout" in wire.payload).toBe(false);
  expect(wire.payload.memories).toEqual([
    {
      key: "profile",
      type: "profile",
      config: { mode: "flow" },
      owners: [{ scope: [], owner: ["root"], node: null }],
    },
    {
      key: "reflective",
      type: "reflective",
      config: { mode: "agent" },
      owners: [{ scope: [], owner: ["root"], node: "agent" }],
    },
  ]);
  expect(wire.payload.memory_stacks).toEqual([{ locator: ["agent"], memories: ["profile", "reflective"] }]);
  expect(wire.payload.memory_snapshots).toEqual([
    { memory: "profile", snapshot: { state: 1n } },
    { memory: "reflective", snapshot: { state: 2n } },
  ]);
  expect(roundTrip(record)).toEqual(record);
  const disabled = wireOf(started());
  expect([disabled.payload.memories, disabled.payload.memory_stacks, disabled.payload.memory_snapshots]).toEqual([
    null,
    null,
    null,
  ]);

  for (const field of ["memory_stacks", "memories"]) {
    const missing = wireOf(record);
    delete missing.payload[field];
    expectError(
      () => decodeRecord(logJson(missing)),
      RangeError,
      `invalid run_started payload fields; missing=["${field}"], extra=[]`,
    );
  }
  const legacy = wireOf(record);
  legacy.payload.memory_layout = null;
  expectError(
    () => decodeRecord(logJson(legacy)),
    RangeError,
    'invalid run_started payload fields; missing=[], extra=["memory_layout"]',
  );
});

const COMMIT_FAILURE = new MemoryOperationFailure("learn", "provider.error", "failed", true);

test("every memory commit record round trips", () => {
  const mutations = [new MemoryMutation("m1", { action: "add" }), new MemoryMutation("m2", { action: "revise" })];
  for (const record of [
    new MemoryCommitRequestedRecord("reflective", "commit-1", mutations, 1),
    new MemoryCommitSucceededRecord("reflective", "commit-1", 1),
    new MemoryCommitFailedRecord("reflective", "commit-1", COMMIT_FAILURE, 1),
  ]) {
    expect(roundTrip(record)).toEqual(record);
  }
});

test.each([1, 2])("the memory commit wire requires an explicit positive attempt slot: %d", (attemptId) => {
  const failure = new MemoryOperationFailure("backend", "failed", "Failed.", false);
  for (const record of [
    new MemoryCommitRequestedRecord("history", "commit-1", [], attemptId),
    new MemoryCommitSucceededRecord("history", "commit-1", attemptId),
    new MemoryCommitFailedRecord("history", "commit-1", failure, attemptId),
  ]) {
    const wire = wireOf(record);
    expect(wire.payload.attempt_id).toBe(BigInt(attemptId));
    delete wire.payload.attempt_id;
    expectError(
      () => decodeRecord(logJson(wire)),
      RangeError,
      `invalid ${wire.kind} payload fields; missing=["attempt_id"], extra=[]`,
    );
  }
});

// test_proposals_preserve_order_and_reject_duplicate_ids
test("proposals preserve order and reject duplicate ids", () => {
  const ordered = [new MemoryMutation("m2", { sequence: 1n }), new MemoryMutation("m1", { sequence: 2n })];
  expect(new MemoryProposal("reflective", ordered).mutations).toEqual(ordered);
  expect(new MemoryCommitRequestedRecord("reflective", "commit", ordered, 1).mutations).toEqual(ordered);
  expectError(
    () => {
      const same = [new MemoryMutation("same", {}), new MemoryMutation("same", {})];
      return new MemoryCommitRequestedRecord("reflective", "commit", same, 1);
    },
    RangeError,
    "mutation IDs must be unique",
  );
});

// The schema row of test_schema_four_rejects_old_schema_and_nested_extra_or_unknown_fields.
test("the wire rejects an old schema", () => {
  const wire = wireOf(new NodeSucceededRecord("agent", 1, { evidence: evidence() }));
  wire.schema = 3n;
  expectError(() => decodeRecord(logJson(wire)), RunLogSchemaError, "unsupported run-record schema: 3");
});

/** A memory-enabled run start with one flow-owned key, `reflective`. */
function flowMemoryStarted(): RunStartedRecord {
  return started({
    memories: [spec("reflective", {}, ROOT)],
    memoryStacks: [],
    memorySnapshots: [new MemorySnapshotEnvelope("reflective", {})],
  });
}

test("the wire rejects an origin, a private config and extra memory operation fields", () => {
  const withOrigin = wireOf(flowMemoryStarted());
  withOrigin.payload.memories[0].origin = { key: "source" };
  expectError(
    () => decodeRecord(logJson(withOrigin)),
    RangeError,
    'invalid run_started payload memories item fields; missing=[], extra=["origin"]',
  );

  const privateConfig = wireOf(flowMemoryStarted());
  privateConfig.payload.memories[0].config = { nested: { api_key: "private" } };
  expectError(
    () => decodeRecord(logJson(privateConfig)),
    RangeError,
    "config.nested contains forbidden private key 'api_key'",
  );

  const failed = wireOf(
    new MemoryCommitFailedRecord(
      "reflective",
      "commit",
      new MemoryOperationFailure("commit", "provider.error", "sanitized", false),
      1,
    ),
  );
  failed.payload.failure.credentials = "secret";
  expectError(
    () => decodeRecord(logJson(failed)),
    RangeError,
    'invalid memory_commit_failed payload failure fields; missing=[], extra=["credentials"]',
  );
});

test.each<[Record<string, unknown>, string]>([
  [{ nested: { secretTokens: ["private"] } }, "secretTokens"],
  [{ nested: { apiKeys: ["private"] } }, "apiKeys"],
  [{ nested: { backendPaths: ["/private/backend"] } }, "backendPaths"],
  [{ nested: { pythonModules: ["package.policy"] } }, "pythonModules"],
])("the wire rejects a plural private memory config: %o", (config, key) => {
  const wire = wireOf(flowMemoryStarted());
  wire.payload.memories[0].config = config;
  expectError(() => decodeRecord(logJson(wire)), RangeError, `config.nested contains forbidden private key '${key}'`);
});

test("a decoded commit's mutation payloads are frozen", () => {
  const record = roundTrip(
    new MemoryCommitRequestedRecord("reflective", "commit", [new MemoryMutation("m1", { items: [1n] })], 1),
  ) as MemoryCommitRequestedRecord;

  expect(() => ((record.mutations[0]!.payload as { items: unknown[] }).items as unknown[]).push(2n)).toThrow(TypeError);
  expect(record.mutations[0]!.payload).toEqual({ items: [1n] });
});

// The commit half of test_memory_payloads_name_keys_on_the_wire.
test("a commit request names its mutations on the wire", () => {
  const commit = new MemoryCommitRequestedRecord("reflective", "c1", [new MemoryMutation("m1", { add: 1n })], 1);
  expect(wireOf(commit).payload.mutations).toEqual([{ mutation_id: "m1", payload: { add: 1n } }]);
});

// The backward-ready half of test_memory_payloads_name_keys_on_the_wire.
test("a backward-ready record names its proposals' memory keys on the wire", () => {
  const proposal = new MemoryProposal("reflective", [new MemoryMutation("m1", { add: 1n })]);
  const ready = new BackwardReadyRecord(1, new BackwardResult(), [], [proposal]);
  const wire = wireOf(ready);
  expect(wire.payload.proposals).toEqual([
    { memory: "reflective", mutations: [{ mutation_id: "m1", payload: { add: 1n } }] },
  ]);
  expect(Object.hasOwn(wire.payload, "frame_proposals")).toBe(false);
  expect((roundTrip(ready) as BackwardReadyRecord).proposals).toEqual([proposal]);

  wire.payload.proposals[0].frame_ref = null;
  expectError(
    () => decodeRecord(logJson(wire)),
    RangeError,
    'invalid backward_ready payload proposals item fields; missing=[], extra=["frame_ref"]',
  );
});

const MEMORY_LABEL = "run_started payload memories item";

test.each<[string, (payload: Record<string, any>) => void, ErrorClass, string]>([
  [
    "an empty memory key",
    (payload) => (payload.memories[0].key = ""),
    RangeError,
    `${MEMORY_LABEL} key must be a non-empty string`,
  ],
  [
    "a config that isn't an object",
    (payload) => (payload.memories[0].config = []),
    TypeError,
    `${MEMORY_LABEL} config must be an object`,
  ],
  [
    "no owner",
    (payload) => (payload.memories[0].owners = []),
    RangeError,
    "owners must be non-empty",
  ],
  [
    "an owner with an extra field",
    (payload) => (payload.memories[0].owners[0].extra = null),
    RangeError,
    `invalid ${MEMORY_LABEL} owners item fields; missing=[], extra=["extra"]`,
  ],
  [
    "an empty owner path",
    (payload) => (payload.memories[0].owners[0].owner = []),
    RangeError,
    "owner must be non-empty",
  ],
  [
    "a non-text owner node",
    (payload) => (payload.memories[0].owners[0].node = 1n),
    TypeError,
    `${MEMORY_LABEL} owners item node must be a string or null`,
  ],
  [
    "a retired eligibility flag",
    (payload) => (payload.memories[0].conversation_eligible = false),
    RangeError,
    `invalid ${MEMORY_LABEL} fields; missing=[], extra=["conversation_eligible"]`,
  ],
  [
    "a snapshot that isn't an object",
    (payload) => (payload.memory_snapshots[0].snapshot = []),
    TypeError,
    "run_started payload memory_snapshots item snapshot must be an object",
  ],
  [
    "a snapshot with an extra field",
    (payload) => (payload.memory_snapshots[0].extra = true),
    RangeError,
    'invalid run_started payload memory_snapshots item fields; missing=[], extra=["extra"]',
  ],
  [
    "an empty stack locator",
    (payload) => (payload.memory_stacks = [{ locator: [], memories: [] }]),
    RangeError,
    "locator must be non-empty",
  ],
  [
    "a non-text stack key",
    (payload) => (payload.memory_stacks = [{ locator: ["agent"], memories: [1n] }]),
    TypeError,
    "run_started payload memory_stacks item memories must be an array of strings",
  ],
])("decode rejects a run_started memory payload with %s (TS-only)", (_name, edit, cls, message) => {
  const wire = wireOf(flowMemoryStarted());
  edit(wire.payload);
  expectError(() => decodeRecord(logJson(wire)), cls, message);
});

test.each<[string, (payload: Record<string, any>) => void, ErrorClass, string]>([
  [
    "an empty mutation id",
    (payload) => (payload.mutations[0].mutation_id = ""),
    RangeError,
    "memory_commit_requested payload mutations item mutation_id must be a non-empty string",
  ],
  [
    "a mutation with an extra field",
    (payload) => (payload.mutations[0].extra = null),
    RangeError,
    'invalid memory_commit_requested payload mutations item fields; missing=[], extra=["extra"]',
  ],
  [
    "mutations that aren't an array",
    (payload) => (payload.mutations = {}),
    TypeError,
    "memory_commit_requested payload mutations must be an array",
  ],
  [
    "repeated mutation ids",
    (payload) => payload.mutations.push(payload.mutations[0]),
    RangeError,
    "mutation IDs must be unique",
  ],
])("decode rejects a memory_commit_requested payload with %s (TS-only)", (_name, edit, cls, message) => {
  const wire = wireOf(new MemoryCommitRequestedRecord("notes", "c1", [new MemoryMutation("m1", {})], 1));
  edit(wire.payload);
  expectError(() => decodeRecord(logJson(wire)), cls, message);
});

test.each<[string, (failure: Record<string, any>) => void, ErrorClass, string]>([
  [
    "an empty stage",
    (failure) => (failure.stage = ""),
    RangeError,
    "memory_commit_failed payload failure stage must be a non-empty string",
  ],
  [
    "a non-boolean retry flag",
    (failure) => (failure.retryable = "yes"),
    TypeError,
    "memory_commit_failed payload failure retryable must be a boolean",
  ],
])("decode rejects a memory_commit_failed failure with %s (TS-only)", (_name, edit, cls, message) => {
  const wire = wireOf(new MemoryCommitFailedRecord("notes", "c1", COMMIT_FAILURE, 1));
  edit(wire.payload.failure);
  expectError(() => decodeRecord(logJson(wire)), cls, message);
});

test("memory JSON holding -0 is written as 0 (TS-only)", () => {
  const snapshot = new MemorySnapshotEnvelope("reflective", { score: -0 });
  const memory = spec("reflective", { weight: -0 });
  const mutation = new MemoryMutation("m1", [-0]);
  expect(Object.is(snapshot.snapshot["score"], -0)).toBe(true);
  expect(Object.is(memory.config["weight"], -0)).toBe(true);
  expect(Object.is((mutation.payload as number[])[0], -0)).toBe(true);

  const startedLine = encodeRecord(
    new RecordEnvelope(
      SCHEMA_VERSION,
      "run",
      0,
      started({ memories: [memory], memoryStacks: [], memorySnapshots: [snapshot] }),
    ),
  );
  const commitLine = encodeRecord(
    new RecordEnvelope(SCHEMA_VERSION, "run", 0, new MemoryCommitRequestedRecord("reflective", "c", [mutation], 1)),
  );

  expect(startedLine).toContain('"config":{"weight":0.0}');
  expect(startedLine).toContain('"snapshot":{"score":0.0}');
  expect(commitLine).toContain('"payload":[0.0]');
  const decoded = decodeRecord(startedLine).record as RunStartedRecord;
  expect(Object.is(decoded.memorySnapshots![0]!.snapshot["score"], 0)).toBe(true);
  expect(Object.is(decoded.memories![0]!.config["weight"], 0)).toBe(true);
  const commit = decodeRecord(commitLine).record as MemoryCommitRequestedRecord;
  expect(Object.is((commit.mutations[0]!.payload as number[])[0], 0)).toBe(true);
  expect(encodeRecord(decodeRecord(commitLine))).toBe(commitLine);
});
