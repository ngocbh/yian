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
 * Covered elsewhere: the scope rows of `test_memory_scope_requires_unique_keys_and_nested_occurrences`
 * by `memory_contracts.test.ts`; `test_memory_spec_config_rejects_private_keys` by
 * `memory_decl_shape.test.ts`.
 *
 * Later: the run-record tests of the module, with the runtime log.
 */

import { expect, test } from "bun:test";

import { current } from "../../src/defaults.ts";
import {
  EvaluationEvidence,
  type JsonObject,
  MemoryScope,
  MemorySection,
  MemorySnapshotEnvelope,
  ModelTextInteraction,
  ModelValueInteraction,
  ToolCallInteraction,
  ToolResultInteraction,
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
