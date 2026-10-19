/**
 * Backward contracts: a port of `tests/engine/test_backward_contracts.py`. Ported: the feedback
 * value checks, `BackwardResult`, `validateBackwardResult`, the `ExecutionScope` and
 * `ExecutionPosition` tests,
 * `test_declared_variables_supply_local_backward_targets_without_activating_hooks` (2 rows),
 * `test_attributed_backward_result_detaches_messages_and_keeps_reference_coordinates`,
 * `test_backward_attribution_rejects_malformed_root_coordinates`,
 * `test_backward_outcome_does_not_normalize_its_caller_owned_failure`,
 * `test_backward_outcome_can_succeed_without_memory`,
 * `test_backward_outcome_copy_preserves_ordered_policy_dispositions`,
 * `test_backward_context_and_outcome_do_not_expose_mutable_authority`,
 * `test_retained_failure_origins_drop_mutable_subclass_state` and
 * `test_backward_contracts_reject_invalid_snapshots_and_outcomes`.
 *
 * TS-only: `copyBackwardValue` keeps a `__proto__` key as data, writes `-0` as `0`, keeps a
 * shared subtree shared, names the nested value it refuses, and with a reused memo names a failing
 * container by its own path in each call; `validateInputOutputPaths` checks each path; a `null`
 * channel is an empty one;
 * `projectStructuralFeedback` selects through `input_output_paths`; record membership reads own
 * keys only (`constructor`, `toString`); an `ExecutionScope` rebuilds each occurrence position, so
 * a forged one is refused; `sameExecutionPosition` and `sameExecutionScope` compare by value, field
 * by field; `sameOptional`'s three cases (both null, one null, both present); `sameBackwardResult`
 * is true for separately built equal results and false for each differing channel. One row per
 * check, with its class and message, for the transfer half of `BackwardAttribution`,
 * `AttributedBackwardResult`, `MemoryProposal`'s memory key,
 * `BackwardFailure` (stage before code) and `BackwardOutcome`; a `MemoryOperationFailure` cause is
 * kept by identity; a transfer attribution keeps a frozen copy of its parents and a root seed is
 * valid; an unreached occurrence with no material is valid; a backward failure keeps each field it
 * was given; parents with their own `Symbol.iterator` are read by index, so the stored parents are
 * the checked elements; the ported invalid-snapshot table also checks the message of its outcome
 * and failure rows.
 *
 * Changed:
 * - Infinity is a backward JSON value (decision 34), so the `output=float("inf")` row is accepted;
 *   an `output: NaN` row is refused instead.
 * - `BackwardResult`, `BackwardContext` and `AttributedBackwardResult` hold frozen copies, with no
 *   copy on read: a reader's edit throws a `TypeError` where Python changes a private copy.
 *   `FrozenInstanceError` is the `TypeError` an assignment to a frozen object throws in a module.
 * - `BackwardOutcome` keeps the frozen feedback, failure and learning result it was given, where
 *   Python copies them: where Python asserts `is not`, the test asserts the same frozen value.
 *   With no `replace`, the outcome copy is a second outcome built from the first's fields.
 * - Message batches, declared names and parents are arrays, not tuples. Dataclass `!=` is
 *   `not.toEqual`; each rejected row names its one error class, where Python accepts either
 *   `TypeError` or `ValueError` (`RangeError`).
 * - Changed (rule `messages`, decision 43): a quoted key in a value path is spelled in the text
 *   form, `value["a"][0]`, `input_feedback["text"]`, never Python's `value['a'][0]`.
 * - The declared-variables row builds the AGENT's `tone` variable by hand where Python reads it
 *   from YAML.
 *
 * Not ported: the rows that pass a list where Python wants a tuple (`static_path=["node"]`,
 * `occurrence_path=[]`, `input_names=["text"]`, a list batch, `{"path": (1,)}`) and the tuple
 * type checks: TS has only arrays. The non-text-key rows (`({1: "Bad."}, {1: "text"})`,
 * `inputs={1: "text"}`): a record key is always text. The feedback row `(("Bad.",), ["text"])` is
 * a valid positional message in TS.
 */

import { expect, test } from "bun:test";

import { Failure, OriginRef } from "../../src/diagnostics.ts";
import { type FeedbackValue, hasFeedback, validateFeedback } from "../../src/expr/index.ts";
import { SourceRef } from "../../src/expr/lineage.ts";
import {
  LearningFailure,
  LearningResult,
  MemoryOperationFailure,
  PolicyCommitOutcome,
} from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import {
  BackwardContext,
  BackwardResult,
  copyBackwardValue,
  projectStructuralFeedback,
  sameBackwardResult,
  validateBackwardResult,
  validateInputOutputPaths,
} from "../../src/nodes/base.ts";
import {
  ExecutionPosition,
  ExecutionScope,
  Parameter,
  sameExecutionPosition,
  sameExecutionScope,
  sameOptional,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import {
  AttributedBackwardResult,
  BackwardAttribution,
  BackwardFailure,
  BackwardOutcome,
  MemoryProposal,
} from "../../src/runtime/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

test.each([
  ["inputs", { topic: "reefs" }],
  ["no inputs", null],
] as const)("declared variables supply local backward targets without activating hooks (%s)", (_label, inputs) => {
  const tone = new Parameter("tone", {
    contract: new ValueContract(Type.scalar(ValueKind.STRING)),
    hasDefault: true,
    default: "concise",
  });
  const node = new AgentNode("writer", { variables: [tone] });
  const context = new BackwardContext(inputs, {
    inputNames: ["topic"],
    variableNames: node.declaredVariables().map((parameter) => parameter.name),
  });
  validateBackwardResult(new BackwardResult({ variableFeedback: { tone: ["Use a technical tone."] } }), context);
  for (const name of ["topic", "other_instruction", "other.tone"]) {
    const result = new BackwardResult({ variableFeedback: { [name]: ["Be clearer."] } });
    expect(() => validateBackwardResult(result, context)).toThrow(RangeError);
    expect(() => validateBackwardResult(result, context)).toThrow(name);
  }
});

test("execution scopes keep root-relative structured positions", () => {
  const root = new ExecutionScope([], { flowPath: [] });
  expect(root.occurrencePath).toEqual([]);
  expect(root.boundary).toBeNull();
  const positions = [
    new ExecutionPosition("call", ["child"]),
    new ExecutionPosition("map_item", ["child", "each"], 2),
    new ExecutionPosition("loop_body", ["child", "each", "loop"], 3),
  ];
  const scope = new ExecutionScope(["child", "each", "loop", "END"], {
    flowPath: ["child", "each", "loop"],
    occurrencePath: positions,
    boundary: "end",
  });
  expect(scope.occurrencePath).toEqual(positions);
  expect(positions[2]).not.toEqual(new ExecutionPosition("loop_driver", positions[2]!.staticPath, 3));
  expect(() => {
    (scope as { boundary: string | null }).boundary = "start";
  }).toThrow(TypeError);
});

test.each<[string, () => unknown, typeof TypeError | typeof RangeError]>([
  ["a boolean path component", () => new ExecutionScope([true] as never), RangeError],
  ["an unknown boundary", () => new ExecutionScope(["node"], { boundary: "other" as never }), RangeError],
  ["a boolean index", () => new ExecutionPosition("map_item", ["each"], true as never), TypeError],
  ["an index beyond 2^53", () => new ExecutionPosition("map_item", ["each"], 2 ** 53), TypeError],
  ["a negative index", () => new ExecutionPosition("loop_driver", ["loop"], -1), RangeError],
  ["a missing index", () => new ExecutionPosition("loop_body", ["loop"]), TypeError],
  ["an index on a call", () => new ExecutionPosition("call", ["child"], 2), RangeError],
])("execution scopes reject %s", (_name, factory, cls) => {
  expect(factory).toThrow(cls);
});

test.each<[unknown, typeof TypeError | typeof RangeError, string]>([
  [true, TypeError, "execution position index must be an integer"],
  [2 ** 53, TypeError, "execution position index must be an integer"],
  [-1, RangeError, "execution position index must be non-negative"],
])("an execution scope rebuilds a forged position with index %p", (index, cls, message) => {
  const forged = Object.assign(Object.create(ExecutionPosition.prototype) as ExecutionPosition, {
    kind: "map_item",
    staticPath: ["each"],
    index,
  });
  const build = () => new ExecutionScope(["each", "END"], { flowPath: ["each"], occurrencePath: [forged] });
  expect(build).toThrow(cls);
  expect(build).toThrow(message);
});

test("execution positions compare by kind, index and static path", () => {
  const position = new ExecutionPosition("map_item", ["child", "each"], 2);
  expect(sameExecutionPosition(position, new ExecutionPosition("map_item", ["child", "each"], 2))).toBe(true);
  expect(sameExecutionPosition(position, new ExecutionPosition("loop_body", ["child", "each"], 2))).toBe(false);
  expect(sameExecutionPosition(position, new ExecutionPosition("map_item", ["child", "each"], 3))).toBe(false);
  expect(sameExecutionPosition(position, new ExecutionPosition("map_item", ["child", "other"], 2))).toBe(false);
  expect(sameExecutionPosition(position, new ExecutionPosition("map_item", ["child"], 2))).toBe(false);
});

test("execution scopes compare by static path, flow path, positions and boundary", () => {
  const call = new ExecutionPosition("call", ["child"]);
  const item = (index: number) => new ExecutionPosition("map_item", ["child", "each"], index);
  const fields = {
    staticPath: ["child", "each", "END"],
    flowPath: ["child", "each"],
    occurrencePath: [call, item(1)],
    boundary: "end" as "start" | "end" | null,
  };
  const scope = (changed: Partial<typeof fields> = {}) => {
    const { staticPath, ...options } = { ...fields, ...changed };
    return new ExecutionScope(staticPath, options);
  };
  const base = scope();

  expect(sameExecutionScope(base, scope())).toBe(true);
  expect(sameExecutionScope(base, scope({ staticPath: ["child", "each", "START"] }))).toBe(false);
  expect(sameExecutionScope(base, scope({ flowPath: ["child"] }))).toBe(false);
  expect(sameExecutionScope(base, scope({ occurrencePath: [call, item(2)] }))).toBe(false);
  expect(sameExecutionScope(base, scope({ occurrencePath: [call] }))).toBe(false);
  expect(sameExecutionScope(base, scope({ occurrencePath: [call, item(1), item(1)] }))).toBe(false);
  expect(sameExecutionScope(base, scope({ boundary: "start" }))).toBe(false);
  expect(sameExecutionScope(base, scope({ boundary: null }))).toBe(false);
});

test("sameOptional is true for two nulls, false for one null, and otherwise asks same", () => {
  const same = (x: string, y: string) => x.toLowerCase() === y.toLowerCase();
  expect(sameOptional<string>(null, null, same)).toBe(true);
  expect(sameOptional("a", null, same)).toBe(false);
  expect(sameOptional(null, "a", same)).toBe(false);
  expect(sameOptional("a", "A", same)).toBe(true);
  expect(sameOptional("a", "b", same)).toBe(false);
});

test("sameBackwardResult compares the three channels by feedback value", () => {
  const inputs = () => ({ text: ["Shorter.", { title: "Bad.", body: null }], items: [["One.", null]] });
  const variables = () => ({ tone: ["Warmer."] });
  const result = (changed: ConstructorParameters<typeof BackwardResult>[0] = {}) =>
    new BackwardResult({ inputFeedback: inputs(), selfFeedback: ["Slow."], variableFeedback: variables(), ...changed });
  const base = result();

  expect(sameBackwardResult(base, result())).toBe(true);
  expect(sameBackwardResult(new BackwardResult(), new BackwardResult({ inputFeedback: null }))).toBe(true);
  expect(sameBackwardResult(base, result({ inputFeedback: { ...inputs(), text: ["Shorter."] } }))).toBe(false);
  expect(sameBackwardResult(base, result({ inputFeedback: { ...inputs(), extra: ["More."] } }))).toBe(false);
  expect(sameBackwardResult(base, result({ inputFeedback: { items: [["One.", "Two."]] } }))).toBe(false);
  expect(sameBackwardResult(base, result({ inputFeedback: {} }))).toBe(false);
  expect(sameBackwardResult(base, result({ selfFeedback: ["Fast."] }))).toBe(false);
  expect(sameBackwardResult(base, result({ selfFeedback: [] }))).toBe(false);
  expect(sameBackwardResult(base, result({ selfFeedback: ["Slow.", "Slow."] }))).toBe(false);
  expect(sameBackwardResult(base, result({ variableFeedback: { tone: ["Cooler."] } }))).toBe(false);
  expect(sameBackwardResult(base, result({ variableFeedback: { style: ["Warmer."] } }))).toBe(false);
  expect(sameBackwardResult(base, result({ variableFeedback: {} }))).toBe(false);
});

test("feedback uses exact nested list lengths", () => {
  const value = { summary: "text", tags: ["a", "b"] };

  validateFeedback("Review the whole result.", value);
  validateFeedback({ tags: "Review these tags." }, value);
  validateFeedback({ tags: [null, "Irrelevant."] }, value);
  validateFeedback({ tags: [null, null] }, value);
  validateFeedback([], []);
  for (const tags of [[], ["Irrelevant."], [null, null, null]]) {
    expect(() => validateFeedback({ tags }, value)).toThrow(/tags/);
  }

  const nested = { groups: [{ tags: ["a", "b"] }, { tags: [] }] };
  validateFeedback({ groups: [{ tags: ["", null] }, { tags: [] }] }, nested);
  expect(() => validateFeedback({ groups: [{ tags: [null] }, null] }, nested)).toThrow(/groups.*0.*tags/);
});

test.each<[unknown, unknown, typeof TypeError | typeof RangeError]>([
  [7n, { summary: "text" }, TypeError],
  [true, { summary: "text" }, TypeError],
  [{ missing: "Bad." }, { summary: "text" }, RangeError],
  [1.5, "text", TypeError],
  [{ summary: false }, { summary: "text" }, TypeError],
  [{ summary: [null] }, { summary: "text" }, RangeError],
  [{ summary: {} }, { summary: [] }, RangeError],
  [[null], { summary: "text" }, RangeError],
  [{}, null, RangeError],
])("feedback rejects invalid values and unknown fields (row %#)", (feedback, value, cls) => {
  expect(() => validateFeedback(feedback, value)).toThrow(cls);
  expect(() => validateFeedback(feedback, value)).toThrow(/feedback/);
});

test.each<[FeedbackValue, boolean]>([
  [null, false],
  [{}, false],
  [[], false],
  [[null, null], false],
  [{ tags: [null, { nested: null }] }, false],
  [{ tags: [], summary: {} }, false],
  ["", true],
  ["Good.", true],
  [{ tags: [null, ""] }, true],
  [{ summary: "Good." }, true],
])("null and empty selections have no effective feedback (row %#)", (feedback, expected) => {
  expect(hasFeedback(feedback)).toBe(expected);
});

test("backward result keeps message tuples distinct from item lists", () => {
  const itemFeedback: FeedbackValue[] = [null, "Irrelevant."];
  const messages: Record<string, FeedbackValue[]> = {
    tags: [itemFeedback, "Review the complete list."],
    summary: [""],
  };
  const result = new BackwardResult({
    inputFeedback: messages,
    selfFeedback: ["Be concise.", "Preserve sources."],
    variableFeedback: { instruction: ["Be explicit.", ""] },
  });

  expect(result.inputFeedback).toEqual({
    tags: [[null, "Irrelevant."], "Review the complete list."],
    summary: [""],
  });
  expect(result.selfFeedback).toEqual(["Be concise.", "Preserve sources."]);
  expect(result.variableFeedback).toEqual({ instruction: ["Be explicit.", ""] });
  const empty = new BackwardResult();
  expect(empty.inputFeedback).toEqual({});
  expect(empty.selfFeedback).toEqual([]);
  expect(empty.variableFeedback).toEqual({});

  itemFeedback[1] = "Changed by the producer.";
  delete messages.tags;
  expect(() => {
    (result.inputFeedback.tags![0] as FeedbackValue[])[1] = "Changed by the reader.";
  }).toThrow(TypeError);
  expect(() => {
    delete (result.inputFeedback as Record<string, unknown>).tags;
  }).toThrow(TypeError);
  expect(() => {
    (result.variableFeedback as Record<string, unknown>).other = [];
  }).toThrow(TypeError);
  expect(result.inputFeedback.tags![0]).toEqual([null, "Irrelevant."]);
  expect(result.variableFeedback).toEqual({ instruction: ["Be explicit.", ""] });
  expect(() => {
    (result as { selfFeedback: readonly string[] }).selfFeedback = [];
  }).toThrow(TypeError);
});

test("a null channel is an empty channel", () => {
  expect(new BackwardResult({ inputFeedback: null, variableFeedback: null })).toEqual(new BackwardResult());
});

test.each<[unknown]>([[null], [7n], [true], [{}], [["Bad."]]])(
  "self and variable feedback require text messages (row %#)",
  (message) => {
    expect(() => new BackwardResult({ selfFeedback: [message] as never })).toThrow(TypeError);
    expect(() => new BackwardResult({ selfFeedback: [message] as never })).toThrow(/self_feedback/);
    expect(() => new BackwardResult({ variableFeedback: { instruction: [message] as never } })).toThrow(TypeError);
    expect(() => new BackwardResult({ variableFeedback: { instruction: [message] as never } })).toThrow(
      /instruction/,
    );
  },
);

test("result channel errors spell a quoted key in the text form", () => {
  expect(() => new BackwardResult({ inputFeedback: { "my key": "Bad." as never } })).toThrow(
    new TypeError('input_feedback["my key"] must be an array'),
  );
  expect(() => new BackwardResult({ variableFeedback: { v: [7] as never } })).toThrow(
    new TypeError('variable_feedback["v"] messages must be strings'),
  );
});

test("copyBackwardValue keeps a __proto__ key as data", () => {
  const copy = copyBackwardValue(JSON.parse('{"__proto__": {"x": "y"}}'), "value") as Record<string, unknown>;
  expect(Object.getPrototypeOf(copy)).toBe(Object.prototype);
  expect(Object.keys(copy)).toEqual(["__proto__"]);
  expect(Object.getOwnPropertyDescriptor(copy, "__proto__")?.value).toEqual({ x: "y" });
});

test("copyBackwardValue writes -0 as 0", () => {
  expect(Object.is(copyBackwardValue(-0, "value"), 0)).toBe(true);
  expect(Object.is((copyBackwardValue([-0], "value") as number[])[0], 0)).toBe(true);
});

test("copyBackwardValue keeps a shared subtree shared and freezes every copy", () => {
  const shared = { note: ["kept"] };
  const copy = copyBackwardValue({ a: shared, b: shared }, "value", new Map()) as Record<string, typeof shared>;
  expect(copy).toEqual({ a: shared, b: shared });
  expect(copy.a).toBe(copy.b!);
  expect(copy.a).not.toBe(shared);
  expect(Object.isFrozen(copy)).toBe(true);
  expect(Object.isFrozen(copy.a)).toBe(true);
  expect(Object.isFrozen(copy.a!.note)).toBe(true);
  const unshared = copyBackwardValue({ a: shared, b: shared }, "value") as Record<string, typeof shared>;
  expect(unshared.a).not.toBe(unshared.b!);
});

class Point {
  x = 1;
}

test.each<[string, unknown, typeof TypeError | typeof RangeError, string]>([
  ["NaN", { a: [NaN] }, RangeError, 'value["a"][0] must not contain NaN'],
  ["a Map", [new Map()], TypeError, "value[0] must contain only JSON values"],
  ["a class instance", { p: new Point() }, TypeError, 'value["p"] must contain only JSON values'],
  ["undefined", [{ u: undefined }], TypeError, 'value[0]["u"] must contain only JSON values'],
])("copyBackwardValue refuses %s with the nested name", (_name, value, cls, message) => {
  expect(() => copyBackwardValue(value, "value")).toThrow(cls);
  expect(() => copyBackwardValue(value, "value")).toThrow(message);
});

test("result validates input shapes and declared variable names", () => {
  const context = new BackwardContext(
    { tags: ["a", "b"], summary: "text", reserved: 7n },
    { inputNames: ["tags", "summary"], variableNames: ["instruction"] },
  );
  validateBackwardResult(
    new BackwardResult({
      inputFeedback: { tags: [[null, "Irrelevant."], "Review the list."] },
      selfFeedback: ["Be concise."],
      variableFeedback: { instruction: ["Make the task explicit."] },
    }),
    context,
  );
  for (const name of ["missing", "reserved"]) {
    const result = new BackwardResult({ inputFeedback: { [name]: ["Bad."] } });
    expect(() => validateBackwardResult(result, context)).toThrow(RangeError);
    expect(() => validateBackwardResult(result, context)).toThrow(name);
  }
  const tags = new BackwardResult({ inputFeedback: { tags: [["Bad."]] } });
  expect(() => validateBackwardResult(tags, context)).toThrow(RangeError);
  expect(() => validateBackwardResult(tags, context)).toThrow(/tags/);
  const parent = new BackwardResult({ variableFeedback: { parent_instruction: ["Bad."] } });
  expect(() => validateBackwardResult(parent, context)).toThrow(RangeError);
  expect(() => validateBackwardResult(parent, context)).toThrow(/parent_instruction/);
  const instruction = new BackwardResult({ variableFeedback: { instruction: ["Bad."] } });
  expect(() => validateBackwardResult(instruction, new BackwardContext({}))).toThrow(RangeError);
  expect(() => validateBackwardResult(instruction, new BackwardContext({}))).toThrow(/instruction/);

  const earlyFailure = new BackwardContext(null, { inputNames: ["tags"] });
  validateBackwardResult(new BackwardResult({ selfFeedback: ["Binding failed."] }), earlyFailure);
  for (const batch of [[], [null], ["Bad."]]) {
    const result = new BackwardResult({ inputFeedback: { tags: batch } });
    expect(() => validateBackwardResult(result, earlyFailure)).toThrow(RangeError);
    expect(() => validateBackwardResult(result, earlyFailure)).toThrow(/inputs/);
  }
});

test("backward context does not expose mutable authority", () => {
  const inputs = { tags: ["a", "b"] };
  const inputNames = ["tags"];
  const output = { summary: ["result"] };
  const data = { prompt: ["Summarize."] };
  const downstream = [{ source_path: ["summary", 0n] }];
  const forwardFailure = new Failure({ code: "node.execute", message: "Forward failed.", nodeId: "writer" });
  const scope = new ExecutionScope(["writer"]);
  const context = new BackwardContext(inputs, {
    inputNames,
    variableNames: ["instruction"],
    output,
    failure: forwardFailure,
    scope,
    runId: "run-1",
    nodeId: "writer",
    nodeStartedSeq: 17,
    nodeData: data,
    downstreamContext: downstream,
  });
  inputs.tags.length = 0;
  inputNames.length = 0;
  output.summary.length = 0;
  data.prompt.length = 0;
  downstream[0]!.source_path.length = 0;
  for (const array of [
    context.inputs!.tags,
    context.inputNames,
    (context.output as typeof output).summary,
    context.nodeData.prompt,
    context.downstreamContext[0]!.source_path,
  ]) {
    expect(() => {
      (array as unknown[]).length = 0;
    }).toThrow(TypeError);
  }
  expect(context.inputs).toEqual({ tags: ["a", "b"] });
  expect(context.inputNames).toEqual(["tags"]);
  expect(context.output).toEqual({ summary: ["result"] });
  expect(context.nodeData).toEqual({ prompt: ["Summarize."] });
  expect(context.downstreamContext).toEqual([{ source_path: ["summary", 0n] }]);
  expect([context.runId, context.nodeStartedSeq]).toEqual(["run-1", 17]);
  expect(context.scope).toEqual(scope);
  expect(context.failure).toStrictEqual(forwardFailure);
  expect(context.failure).not.toBe(forwardFailure);
  expect(Object.isFrozen(context.failure)).toBe(true);
  expect(Object.isFrozen(context.failure!.trace)).toBe(true);
  expect(() => {
    (context as { inputNames: readonly string[] }).inputNames = [];
  }).toThrow(TypeError);
});

class AnnotatedOrigin extends OriginRef {
  notes: string[];

  constructor(key: string, notes: string[]) {
    super(key);
    this.notes = notes;
  }
}

test.each([
  ["context", "argument"],
  ["context", "public view"],
  ["failure", "argument"],
  ["failure", "public view"],
  ["outcome", "argument"],
  ["outcome", "public view"],
])("retained failure origins drop mutable subclass state (%s, %s)", (carrier, mutation) => {
  const origin = new AnnotatedOrigin("writer.prompt", ["authored note"]);
  const failure = new Failure({
    code: "node.execute",
    message: "Forward failed.",
    nodeId: "writer",
    trace: [origin, new OriginRef("writer"), origin],
    causeType: "LookupError",
    traceback: "original debugging detail",
  });
  const context = new BackwardContext(null, { failure });
  const backwardFailure = new BackwardFailure("backward", "node.backward", "Reverse failed.", { cause: failure });
  const outcome = new BackwardOutcome("failed", { failure: backwardFailure });
  const retained = {
    context: context.failure,
    failure: backwardFailure.cause,
    outcome: outcome.failure!.cause,
  }[carrier] as Failure;

  if (mutation === "argument") {
    origin.notes.push("changed through the constructor argument");
  } else {
    for (const item of retained.trace) {
      if (item instanceof AnnotatedOrigin) item.notes.push("changed through a public diagnostic view");
    }
  }

  expect(retained).toStrictEqual(
    new Failure({
      code: "node.execute",
      message: "Forward failed.",
      nodeId: "writer",
      trace: [new OriginRef("writer.prompt"), new OriginRef("writer"), new OriginRef("writer.prompt")],
      causeType: "LookupError",
      traceback: "original debugging detail",
    }),
  );
  expect(retained.trace.every((item) => Object.getPrototypeOf(item) === OriginRef.prototype)).toBe(true);
  expect(retained.trace.every((item) => Object.isFrozen(item))).toBe(true);
  expect(Object.isFrozen(retained)).toBe(true);
  expect(failure.trace[0]).toBe(origin);
});

test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string | null]>([
  ["array inputs", () => new BackwardContext([] as never), TypeError, null],
  ["a class instance in the inputs", () => new BackwardContext({ text: new Point() }), TypeError, null],
  ["a NaN output", () => new BackwardContext({}, { output: NaN }), RangeError, null],
  ["a duplicate input name", () => new BackwardContext({}, { inputNames: ["text", "text"] }), RangeError, null],
  ["a non-text variable name", () => new BackwardContext({}, { variableNames: [7] as never }), TypeError, null],
  ["a boolean start sequence", () => new BackwardContext({}, { nodeStartedSeq: true as never }), TypeError, null],
  ["a zero start sequence", () => new BackwardContext({}, { nodeStartedSeq: 0 }), RangeError, null],
  ["array node data", () => new BackwardContext({}, { nodeData: [] as never }), TypeError, null],
  [
    "an unknown backward status",
    () => new BackwardOutcome("incomplete" as never),
    RangeError,
    "backward status must be 'succeeded' or 'failed'",
  ],
  [
    "a failed outcome without a failure",
    () => new BackwardOutcome("failed"),
    RangeError,
    "failed backward requires a failure",
  ],
  [
    "record feedback",
    () => new BackwardOutcome("succeeded", { feedback: {} as never }),
    TypeError,
    "feedback must be a BackwardResult or null",
  ],
  [
    "a succeeded outcome with a failure",
    () => new BackwardOutcome("succeeded", { failure: new BackwardFailure("learn", "x", "x") }),
    RangeError,
    "succeeded backward cannot carry a failure",
  ],
  [
    "an empty failure code",
    () => new BackwardFailure("learn", "", "detail"),
    RangeError,
    "backward failure code must be non-empty",
  ],
  [
    "a boolean failure start sequence",
    () => new BackwardFailure("learn", "x", "detail", { nodeStartedSeq: true as never }),
    TypeError,
    "node_started_seq must be an integer or null",
  ],
  [
    "an error as a failure cause",
    () => new BackwardFailure("learn", "x", "detail", { cause: new RangeError("bad") as never }),
    TypeError,
    "cause must be a Failure, MemoryOperationFailure, or null",
  ],
])("backward contracts reject invalid snapshots and outcomes (%s)", (_name, factory, cls, message) => {
  if (message === null) expect(factory).toThrow(cls);
  else expectError(factory, cls, message);
});

test.each<[string, unknown, string]>([
  ["a trace that isn't an array", "x", "failure trace must be an array"],
  ["a plain {key} item", [{ key: "writer" }], "failure trace item must be an OriginRef"],
])("backward context checks the failure's trace before copying it (%s)", (_name, trace, message) => {
  const failure = new Failure({ code: "node.execute", message: "Forward failed.", nodeId: "writer" });
  (failure as { trace: unknown }).trace = trace;
  expect(() => new BackwardContext(null, { failure })).toThrow(new TypeError(message));
});

test("copyBackwardValue with a reused memo names a failing container by its own path", () => {
  const memo = new Map<object, unknown>();
  const failing = { x: [NaN] };
  expect(() => copyBackwardValue({ a: failing }, "first", memo)).toThrow('first["a"]["x"][0] must not contain NaN');
  expect(() => copyBackwardValue([failing], "second", memo)).toThrow('second[0]["x"][0] must not contain NaN');
});

const BAD_PART = "output path components must be strings or non-negative integers";

test.each<[string, unknown, typeof TypeError | typeof RangeError, string]>([
  ["paths that aren't a record", [], TypeError, "input_output_paths must be a record"],
  ["an undeclared input", { other: [] }, RangeError, "input_output_paths must name declared inputs"],
  ["a path that isn't an array", { topic: "summary" }, TypeError, "input_output_paths values must be arrays"],
  ["a negative bigint", { topic: [-1n] }, RangeError, BAD_PART],
  ["a negative number", { topic: [-1] }, RangeError, BAD_PART],
  ["a fractional number", { topic: [1.5] }, RangeError, BAD_PART],
  ["a boolean", { topic: [true] }, RangeError, BAD_PART],
])("validateInputOutputPaths refuses %s", (_name, paths, cls, message) => {
  expect(() => validateInputOutputPaths({ input_output_paths: paths }, ["topic"])).toThrow(cls);
  expect(() => validateInputOutputPaths({ input_output_paths: paths }, ["topic"])).toThrow(message);
});

test("validateInputOutputPaths accepts strings and non-negative integers", () => {
  validateInputOutputPaths({ input_output_paths: { topic: ["items", 0n, 2, Number.MAX_SAFE_INTEGER] } }, ["topic"]);
  validateInputOutputPaths({}, ["topic"]);
});

test("backward context accepts an infinite output", () => {
  expect(new BackwardContext({}, { output: Infinity }).output).toBe(Infinity);
});

test("projectStructuralFeedback selects through input_output_paths and skips inputs with no path", () => {
  const context = new BackwardContext(
    { topic: "reefs", item: "b", extra: "x" },
    {
      inputNames: ["topic", "item", "extra"],
      output: { summary: "text", items: ["a", "b"] },
      nodeData: { input_output_paths: { topic: ["summary"], item: ["items", 1n] } },
    },
  );
  const result = projectStructuralFeedback(context, [
    { summary: "Shorter.", items: [null, "Wrong item."] },
    "Whole output.",
  ]);
  expect(result.inputFeedback).toEqual({
    topic: ["Shorter.", "Whole output."],
    item: ["Wrong item.", "Whole output."],
  });
});

test("projectStructuralFeedback refuses a missing path and missing input_output_paths", () => {
  const missingPath = new BackwardContext(
    { topic: "reefs" },
    { inputNames: ["topic"], output: { summary: "text" }, nodeData: { input_output_paths: { topic: ["other"] } } },
  );
  expect(() => projectStructuralFeedback(missingPath, ["Bad."])).toThrow(RangeError);
  expect(() => projectStructuralFeedback(missingPath, ["Bad."])).toThrow(
    "captured output path does not exist in the accepted output",
  );
  const noPaths = new BackwardContext({ topic: "reefs" }, { inputNames: ["topic"], output: "text" });
  expect(() => projectStructuralFeedback(noPaths, ["Bad."])).toThrow(RangeError);
  expect(() => projectStructuralFeedback(noPaths, ["Bad."])).toThrow(
    "structural backward requires captured input_output_paths",
  );
});

test("projectStructuralFeedback gives no feedback to a path a partial message omits", () => {
  const context = new BackwardContext(
    { topic: "reefs" },
    { inputNames: ["topic"], output: { a: "1", b: "2" }, nodeData: { input_output_paths: { topic: ["b"] } } },
  );
  expect(projectStructuralFeedback(context, [{ a: "x" }]).inputFeedback).toEqual({});
});

test("projectStructuralFeedback skips an input named constructor with no path", () => {
  const context = new BackwardContext(
    { constructor: "x" },
    { inputNames: ["constructor"], output: "text", nodeData: { input_output_paths: {} } },
  );
  expect(projectStructuralFeedback(context, ["Note."]).inputFeedback).toEqual({});
});

test("input feedback for an unrecorded toString input is refused", () => {
  const result = new BackwardResult({ inputFeedback: { toString: ["Bad."] } });
  const context = new BackwardContext({}, { inputNames: ["toString"] });
  expect(() => validateBackwardResult(result, context)).toThrow(RangeError);
  expect(() => validateBackwardResult(result, context)).toThrow("input_feedback has no recorded input 'toString'");
});

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
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

const SENDER = new SourceRef("run", 4);

test("attributed backward result detaches messages and keeps reference coordinates", () => {
  const source = new SourceRef("run", 1);
  const attribution = new BackwardAttribution(SENDER, "items", 0, 1, [0, 2]);
  const message: FeedbackValue[] = ["first", null];
  const result = new AttributedBackwardResult(
    source,
    [message],
    [attribution],
    new BackwardResult({ selfFeedback: ["note"] }),
    true,
  );
  message[0] = "changed";
  expect(() => {
    (result.feedback[0] as FeedbackValue[])[0] = "also changed";
  }).toThrow(TypeError);
  expect(result.feedback).toEqual([["first", null]]);
  expect(result.attribution).toEqual([attribution]);
  expect(result.attribution[0]).toBe(attribution);
  expect(result.result.selfFeedback).toEqual(["note"]);
  expect("order" in attribution).toBe(false);
  for (const value of [result, result.feedback, result.attribution, attribution, attribution.parents]) {
    expect(Object.isFrozen(value)).toBe(true);
  }
});

test("a backward attribution copies its parents", () => {
  const parents = [0, 2];
  const attribution = new BackwardAttribution(SENDER, "items", 0, 1, parents);
  parents.push(3);
  expect(attribution.parents).toEqual([0, 2]);
  const root = new BackwardAttribution(null, null, 0, null, []);
  expect([root.sender, root.inputName, root.messageIndex, root.contributionIndex, root.parents]).toEqual([
    null,
    null,
    0,
    null,
    [],
  ]);
});

test("a backward attribution stores the parents it checked, not what their iterator yields", () => {
  const parents = [1, 2];
  Object.defineProperty(parents, Symbol.iterator, { value: () => [5, 3][Symbol.iterator]() });
  expect(new BackwardAttribution(SENDER, "items", 0, 1, parents).parents).toEqual([1, 2]);
});

/** Build a `BackwardAttribution` from untyped arguments. */
function attribute(args: unknown[]): BackwardAttribution {
  return new BackwardAttribution(...(args as ConstructorParameters<typeof BackwardAttribution>));
}

const ROOT_COORDINATES = "root attribution cannot carry sender coordinates";

test.each<[unknown[], typeof TypeError | typeof RangeError, string]>([
  [[null, "input", 0, null, []], RangeError, ROOT_COORDINATES],
  [[null, null, 1, null, []], RangeError, ROOT_COORDINATES],
  [[null, null, 0, 0, []], RangeError, ROOT_COORDINATES],
  [[null, null, 0, null, [0]], RangeError, ROOT_COORDINATES],
  [["sender", "input", 0, 0, []], TypeError, "source must be an exact SourceRef"],
])("backward attribution rejects malformed root coordinates (row %#)", (args, cls, message) => {
  expectError(() => attribute(args), cls, message);
});

const PARENTS_TYPE = "parents must be an array of non-negative integer positions";
const PARENTS_ORDER = "parents must be distinct ordered positions";
const ACCEPTED_SENDER = "attribution sender must identify an accepted occurrence";
const CONTRIBUTION = "transfer attribution requires a non-negative contribution index";

test.each<[string, unknown[], typeof TypeError | typeof RangeError, string]>([
  [
    "a negative message index",
    [SENDER, "items", -1, 0, [0]],
    RangeError,
    "message_index must be a non-negative integer",
  ],
  ["fractional parents", [SENDER, "items", 0, 0, [1.5]], TypeError, PARENTS_TYPE],
  ["sparse parents", [SENDER, "items", 0, 0, [, 1]], TypeError, PARENTS_TYPE],
  ["duplicate parents", [SENDER, "items", 0, 0, [0, 0]], RangeError, PARENTS_ORDER],
  ["misordered parents", [SENDER, "items", 0, 0, [2, 0]], RangeError, PARENTS_ORDER],
  ["a run-input sender", [new SourceRef("run", 0), "items", 0, 0, [0]], RangeError, ACCEPTED_SENDER],
  ["a sender with a path", [new SourceRef("run", 4, ["a"]), "items", 0, 0, [0]], RangeError, ACCEPTED_SENDER],
  ["an empty input name", [SENDER, "", 0, 0, [0]], RangeError, "transfer attribution requires an input name"],
  ["a missing contribution index", [SENDER, "items", 0, null, [0]], RangeError, CONTRIBUTION],
  ["a negative contribution index", [SENDER, "items", 0, -1, [0]], RangeError, CONTRIBUTION],
  ["empty parents", [SENDER, "items", 0, 0, []], RangeError, "transfer attribution requires incoming parent positions"],
])("a transfer attribution rejects %s", (_name, args, cls, message) => {
  expectError(() => attribute(args), cls, message);
});

const OCCURRENCE = new SourceRef("run", 1);
const LINK = new BackwardAttribution(SENDER, "items", 0, 0, [0]);
const ACCEPTED_SOURCE = "backward result source must identify an accepted occurrence";
const ARRAYS = "feedback and attribution must be arrays";
const RESULT_AND_INVOKED = "result and invoked must be a BackwardResult and boolean";
const UNREACHED = "unreached occurrences cannot carry backward material";

test.each<[string, unknown[], typeof TypeError | typeof RangeError, string]>([
  [
    "a source that isn't a SourceRef",
    [{}, ["x"], [LINK], new BackwardResult(), true],
    TypeError,
    "source must be an exact SourceRef",
  ],
  [
    "a run-input source",
    [new SourceRef("run", 0), ["x"], [LINK], new BackwardResult(), true],
    RangeError,
    ACCEPTED_SOURCE,
  ],
  [
    "a source with a path",
    [new SourceRef("run", 1, [0]), ["x"], [LINK], new BackwardResult(), true],
    RangeError,
    ACCEPTED_SOURCE,
  ],
  ["feedback that isn't an array", [OCCURRENCE, new Set(["x"]), [LINK], new BackwardResult(), true], TypeError, ARRAYS],
  ["attribution that isn't an array", [OCCURRENCE, ["x"], LINK, new BackwardResult(), true], TypeError, ARRAYS],
  [
    "misaligned batches",
    [OCCURRENCE, ["x", "y"], [LINK], new BackwardResult(), true],
    RangeError,
    "feedback and attribution must align",
  ],
  [
    "a number message",
    [OCCURRENCE, [7], [LINK], new BackwardResult(), true],
    TypeError,
    "feedback must be a string, object, list, or None",
  ],
  [
    "a plain attribution",
    [OCCURRENCE, ["x"], [{ ...LINK }], new BackwardResult(), true],
    TypeError,
    "attribution requires exact BackwardAttribution values",
  ],
  ["a record result", [OCCURRENCE, ["x"], [LINK], {}, true], TypeError, RESULT_AND_INVOKED],
  ["a number invoked flag", [OCCURRENCE, ["x"], [LINK], new BackwardResult(), 1], TypeError, RESULT_AND_INVOKED],
  [
    "an unreached occurrence with messages",
    [OCCURRENCE, ["x"], [LINK], new BackwardResult(), false],
    RangeError,
    UNREACHED,
  ],
  [
    "an unreached occurrence with self feedback",
    [OCCURRENCE, [], [], new BackwardResult({ selfFeedback: ["x"] }), false],
    RangeError,
    UNREACHED,
  ],
  [
    "an unreached occurrence with an empty input batch",
    [OCCURRENCE, [], [], new BackwardResult({ inputFeedback: { topic: [] } }), false],
    RangeError,
    UNREACHED,
  ],
  [
    "an unreached occurrence with an empty variable batch",
    [OCCURRENCE, [], [], new BackwardResult({ variableFeedback: { tone: [] } }), false],
    RangeError,
    UNREACHED,
  ],
  [
    "an invoked occurrence without messages",
    [OCCURRENCE, [], [], new BackwardResult(), true],
    RangeError,
    "invoked occurrences require received feedback",
  ],
])("an attributed backward result rejects %s", (_name, args, cls, message) => {
  expectError(
    () => new AttributedBackwardResult(...(args as ConstructorParameters<typeof AttributedBackwardResult>)),
    cls,
    message,
  );
});

test("an unreached occurrence carries no material", () => {
  const result = new AttributedBackwardResult(OCCURRENCE, [], [], new BackwardResult(), false);
  expect([result.feedback, result.attribution, result.invoked]).toEqual([[], [], false]);
});

test.each<[unknown, typeof TypeError | typeof RangeError, string]>([
  [1, TypeError, "memory must be a string"],
  ["", RangeError, "memory must be non-empty"],
])("a memory proposal rejects the memory key %p", (memory, cls, message) => {
  expectError(() => new MemoryProposal(memory as string, []), cls, message);
});

test("backward outcome does not expose mutable authority", () => {
  const scope = new ExecutionScope(["writer"]);
  const cause = new MemoryOperationFailure("learn", "model.failed", "Model failed.", true);
  const failure = new BackwardFailure("learn", "memory.learn", "Could not compute a learning proposal.", {
    scope,
    memory: "history",
    cause,
    causeType: "MemoryOperationError",
    traceback: "debugging detail",
  });
  const learning = new LearningResult("failed", { failures: [new LearningFailure("history", "learn", cause)] });
  const feedback = new BackwardResult({ inputFeedback: { tags: [[null, "Irrelevant."]] } });
  const outcome = new BackwardOutcome("failed", { feedback, failure, learning });
  expect(() => {
    (outcome.feedback!.inputFeedback.tags![0] as FeedbackValue[]).length = 0;
  }).toThrow(TypeError);
  expect(outcome.feedback!.inputFeedback).toEqual({ tags: [[null, "Irrelevant."]] });
  expect(outcome.status).toBe("failed");
  expect(outcome.failure!.cause).toBe(cause);
  expect((outcome.failure!.cause as MemoryOperationFailure).retryable).toBe(true);
  expect(outcome.failure!.scope).toBe(scope);
  expect(outcome.failure!.traceback).toBe("debugging detail");
  expect(outcome.learning).toBe(learning);
  expect(Object.isFrozen(failure)).toBe(true);
  expect(() => {
    (outcome as { status: string }).status = "succeeded";
  }).toThrow(TypeError);
  expect(() => {
    (failure as { memory: string | null }).memory = "other";
  }).toThrow(TypeError);
});

test("backward outcome does not normalize its caller-owned failure", () => {
  const cause = new Failure({
    code: "node.execute",
    message: "Forward failed.",
    nodeId: "writer",
    trace: [new OriginRef("writer")],
  });
  const trace = cause.trace;
  const origin = cause.trace[0];
  const failure = new BackwardFailure("backward", "node.backward", "Reverse failed.", { cause });
  const originalCause = failure.cause;

  const outcome = new BackwardOutcome("failed", { failure });

  expect(failure.cause).toBe(originalCause);
  expect(failure.cause).not.toBe(cause);
  expect(cause.trace).toBe(trace);
  expect(cause.trace[0]).toBe(origin!);
  expect(Object.isFrozen(cause)).toBe(false);
  // Changed: the outcome keeps the same frozen failure, where Python copies it.
  expect(outcome.failure).toBe(failure);
  expect(Object.isFrozen(outcome.failure)).toBe(true);
});

test("backward outcome can succeed without memory", () => {
  const result = new BackwardResult();
  const outcome = new BackwardOutcome("succeeded", { feedback: result });
  expect(outcome.feedback).toBe(result);
  expect(outcome.failure).toBeNull();
  expect(outcome.learning).toBeNull();
});

test("backward outcome copy preserves ordered policy dispositions", () => {
  const policies = [
    new PolicyCommitOutcome("second", "unknown", { commitId: "commit-2" }),
    new PolicyCommitOutcome("first", "unrequested"),
  ];
  const learning = new LearningResult("incomplete", { policies });
  const feedback = new BackwardResult();
  const outcome = new BackwardOutcome("failed", {
    feedback,
    failure: new BackwardFailure("commit", "interrupted", ""),
    learning,
  });
  const copied = new BackwardOutcome(outcome.status, {
    feedback: outcome.feedback,
    failure: outcome.failure,
    learning: outcome.learning,
  });
  expect(copied).toEqual(outcome);
  expect(copied.feedback).toBe(feedback);
  // Changed: the copy shares the frozen learning result and its policies, where Python copies them.
  expect(copied.learning).toBe(learning);
  expect(copied.learning!.policies).toEqual(policies);
  expect(copied.learning!.policies[0]).toBe(policies[0]!);
  expect(Object.isFrozen(copied.learning!.policies)).toBe(true);
});

class SubFailure extends Failure {}

const FAILURE_SEQ_TYPE = "node_started_seq must be an integer or null";
const FAILURE_CAUSE = "cause must be a Failure, MemoryOperationFailure, or null";

test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string]>([
  [
    "a number stage",
    () => new BackwardFailure(1 as never, "x", "y"),
    TypeError,
    "backward failure stage must be a string",
  ],
  ["an empty stage", () => new BackwardFailure("", "x", "y"), RangeError, "backward failure stage must be non-empty"],
  [
    "a number code",
    () => new BackwardFailure("learn", 1 as never, "y"),
    TypeError,
    "backward failure code must be a string",
  ],
  [
    "an empty stage and a number code",
    () => new BackwardFailure("", 1 as never, "y"),
    RangeError,
    "backward failure stage must be non-empty",
  ],
  [
    "a number message",
    () => new BackwardFailure("learn", "x", 1 as never),
    TypeError,
    "backward failure message must be a string",
  ],
  [
    "a number run id",
    () => new BackwardFailure("learn", "x", "y", { runId: 1 as never }),
    TypeError,
    "backward failure run_id must be a string or null",
  ],
  [
    "a number node id",
    () => new BackwardFailure("learn", "x", "y", { nodeId: 1 as never }),
    TypeError,
    "backward failure node_id must be a string or null",
  ],
  [
    "a number memory",
    () => new BackwardFailure("learn", "x", "y", { memory: 1 as never }),
    TypeError,
    "backward failure memory must be a string or null",
  ],
  [
    "a number cause type",
    () => new BackwardFailure("learn", "x", "y", { causeType: 1 as never }),
    TypeError,
    "backward failure cause_type must be a string or null",
  ],
  [
    "a number traceback",
    () => new BackwardFailure("learn", "x", "y", { traceback: 1 as never }),
    TypeError,
    "backward failure traceback must be a string or null",
  ],
  [
    "a fractional start sequence",
    () => new BackwardFailure("learn", "x", "y", { nodeStartedSeq: 1.5 }),
    TypeError,
    FAILURE_SEQ_TYPE,
  ],
  [
    "a bigint start sequence",
    () => new BackwardFailure("learn", "x", "y", { nodeStartedSeq: 1n as never }),
    TypeError,
    FAILURE_SEQ_TYPE,
  ],
  [
    "a zero start sequence",
    () => new BackwardFailure("learn", "x", "y", { nodeStartedSeq: 0 }),
    RangeError,
    "node_started_seq must be positive",
  ],
  [
    "a plain scope",
    () => new BackwardFailure("learn", "x", "y", { scope: { staticPath: [] } as never }),
    TypeError,
    "scope must be an ExecutionScope or null",
  ],
  [
    "a plain cause",
    () => new BackwardFailure("learn", "x", "y", { cause: { code: "node.execute" } as never }),
    TypeError,
    FAILURE_CAUSE,
  ],
  [
    "a Failure subclass cause",
    () => {
      const cause = new SubFailure({ code: "node.execute", message: "", nodeId: null });
      return new BackwardFailure("learn", "x", "y", { cause });
    },
    TypeError,
    FAILURE_CAUSE,
  ],
])("a backward failure rejects %s", (_name, factory, cls, message) => {
  expectError(factory, cls, message);
});

test("a backward failure keeps its attribution", () => {
  const scope = new ExecutionScope(["writer"]);
  const failure = new BackwardFailure("backward", "node.backward", "", {
    scope,
    runId: "run-1",
    nodeId: "writer",
    nodeStartedSeq: 3,
    memory: "notes",
  });
  expect([failure.stage, failure.code, failure.message]).toEqual(["backward", "node.backward", ""]);
  expect([failure.scope, failure.runId, failure.nodeId, failure.nodeStartedSeq, failure.memory]).toEqual([
    scope,
    "run-1",
    "writer",
    3,
    "notes",
  ]);
  expect([failure.cause, failure.causeType, failure.traceback]).toEqual([null, null, null]);
});

test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string]>([
  [
    "a plain failure",
    () => new BackwardOutcome("failed", { failure: {} as never }),
    TypeError,
    "failure must be a BackwardFailure or null",
  ],
  [
    "a plain learning result",
    () => new BackwardOutcome("succeeded", { learning: {} as never }),
    TypeError,
    "learning must be a LearningResult or null",
  ],
  [
    "a succeeded outcome with incomplete learning",
    () => new BackwardOutcome("succeeded", { learning: new LearningResult("incomplete") }),
    RangeError,
    "succeeded backward requires succeeded learning",
  ],
])("a backward outcome rejects %s", (_name, factory, cls, message) => {
  expectError(factory, cls, message);
});
