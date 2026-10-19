/**
 * Run-record contracts for source-neutral memory evidence: a port of
 * `tests/engine/test_memory_run_records.py`. Ported: the evidence carriers,
 * `test_source_neutral_carriers_are_defensive_and_frozen` and
 * `test_evidence_requires_coherent_prompts_and_ordered_scope_sections`, and the evidence line of
 * `test_memory_scope_requires_unique_keys_and_nested_occurrences`. Python's `ValueError` is a
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
 * Later: the other wire tests of the module (the round trips of the run-start tests above, the
 * commit and backward records, their schema and payload rejections), with the record codec.
 */

import { expect, test } from "bun:test";

import { current } from "../../src/defaults.ts";
import { Failure } from "../../src/diagnostics.ts";
import {
  DeclSite,
  EvaluationEvidence,
  type JsonObject,
  MemoryMutation,
  MemoryOccurrenceSegment,
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
import {
  decodeRecord,
  encodePool,
  encodeRecord,
  MemoryCommitRequestedRecord,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeSucceededRecord,
  RecordEnvelope,
  type RunRecord,
  type RunStartedFields,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { VariablePool } from "../../src/typesys/index.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";

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

const ROOT = new DeclSite([], ["root"], null);
const AGENT = new DeclSite([], ["root"], "agent");

/** Return one key-table entry; an AGENT owner makes it conversation-eligible. */
function spec(key = "reflective", config: JsonObject = {}, owner: DeclSite = AGENT): MemorySpec {
  return new MemorySpec(key, key, config, [owner], owner.node !== null);
}

/** Return a run-start record with an empty pool and the given memory fields. */
function started(
  fields: Pick<Partial<RunStartedFields>, "memorySnapshots" | "learnable" | "memories" | "memoryStacks"> = {},
): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    initialPool: encodePool(new VariablePool("start")),
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

  expect(record.memories!.map((item) => [item.key, item.config, item.conversationEligible])).toEqual([
    ["profile", { mode: "flow" }, false],
    ["reflective", { mode: "agent" }, true],
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
    { memories: [new SubSpec("notes", "notes", {}, [ROOT], false)], memoryStacks: [], memorySnapshots: [] },
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
    () => new NodeSucceededRecord("agent", "agent", { kind: "str", value: "done" }, { evidence: forgedEvidence }),
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
const DONE = { kind: "str", value: "done" } as const;

test.each<[string, () => RunRecord]>([
  ["NodeSucceededRecord", () => new NodeSucceededRecord("agent", "agent", DONE, { evidence: evidence() })],
  ["NodeRoutedRecord", () => new NodeRoutedRecord("agent", "yes", { evidence: evidence() })],
  ["NodeExpandedRecord", () => new NodeExpandedRecord("agent", {}, [], DIGEST, { evidence: evidence() })],
  ["NodeFailedRecord", () => new NodeFailedRecord(FAILED, { evidence: evidence() })],
])("all outcome records round trip closed evidence: %s", (_name, build) => {
  const record = build();
  expect(roundTrip(record)).toEqual(record);
});

test("existing outcome records round trip with no evidence", () => {
  for (const record of [
    new NodeSucceededRecord("agent", "agent", DONE),
    new NodeRoutedRecord("agent", "yes"),
    new NodeExpandedRecord("agent", {}, [], DIGEST),
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
  const wire = wireOf(new NodeSucceededRecord("agent", "agent", DONE, { evidence: evidence() }));
  mutate(wire.payload.evidence);
  expectError(() => decodeRecord(logJson(wire)), RangeError, message);
});

test("the memory scope wire has no boundary field", () => {
  const wire = wireOf(new NodeSucceededRecord("agent", "agent", DONE, { evidence: evidence() }));
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
  const record = new NodeSucceededRecord("agent", "agent", DONE, { evidence: evidence() });
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
