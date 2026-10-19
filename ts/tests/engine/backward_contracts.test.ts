/**
 * Backward contracts: a port of `tests/engine/test_backward_contracts.py`. Ported: the feedback
 * value checks, `BackwardResult`, `validateBackwardResult`, the `ExecutionScope` and
 * `ExecutionPosition` tests,
 * `test_declared_variables_supply_local_backward_targets_without_activating_hooks` (2 rows), and
 * the `BackwardContext` halves of
 * `test_backward_context_and_outcome_do_not_expose_mutable_authority`,
 * `test_retained_failure_origins_drop_mutable_subclass_state` (the two `context` cases) and
 * `test_backward_contracts_reject_invalid_snapshots_and_outcomes`.
 *
 * TS-only: `copyBackwardValue` keeps a `__proto__` key as data, writes `-0` as `0`, keeps a
 * shared subtree shared, names the nested value it refuses, and with a reused memo names a failing
 * container by its own path in each call; `validateInputOutputPaths` checks each path; a `null`
 * channel is an empty one;
 * `projectStructuralFeedback` selects through `input_output_paths`; record membership reads own
 * keys only (`constructor`, `toString`); an `ExecutionScope` rebuilds each occurrence position, so
 * a forged one is refused; `sameExecutionPosition` and `sameExecutionScope` compare by value, field
 * by field.
 *
 * Changed:
 * - Infinity is a backward JSON value (decision 34), so the `output=float("inf")` row is accepted;
 *   an `output: NaN` row is refused instead.
 * - `BackwardResult` and `BackwardContext` hold frozen copies, with no copy on read: a reader's
 *   edit throws a `TypeError` where Python changes a private copy. `FrozenInstanceError` is the
 *   `TypeError` an assignment to a frozen object throws in a module.
 * - Message batches and declared names are arrays, not tuples. Dataclass `!=` is `not.toEqual`;
 *   each rejected row names its one error class, where Python accepts either `TypeError` or
 *   `ValueError` (`RangeError`).
 * - Changed (rule `messages`, decision 43): a quoted key in a value path is spelled in the text
 *   form, `value["a"][0]`, `input_feedback["text"]`, never Python's `value['a'][0]`.
 * - The declared-variables row builds the AGENT's `tone` variable by hand where Python reads it
 *   from YAML.
 *
 * Later (T4-1): `test_attributed_backward_result_detaches_messages_and_keeps_reference_coordinates`,
 * `test_backward_attribution_rejects_malformed_root_coordinates`,
 * `test_backward_outcome_does_not_normalize_its_caller_owned_failure`,
 * `test_backward_outcome_can_succeed_without_memory` and
 * `test_backward_outcome_copy_preserves_ordered_policy_dispositions` need the runtime contracts,
 * as do the outcome and failure halves of the three partly ported tests.
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
import { AgentNode } from "../../src/nodes/agent/node.ts";
import {
  BackwardContext,
  BackwardResult,
  copyBackwardValue,
  projectStructuralFeedback,
  validateBackwardResult,
  validateInputOutputPaths,
} from "../../src/nodes/base.ts";
import {
  ExecutionPosition,
  ExecutionScope,
  Parameter,
  sameExecutionPosition,
  sameExecutionScope,
  ValueContract,
} from "../../src/nodes/contracts.ts";
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
  ["a negative index", () => new ExecutionPosition("loop_driver", ["loop"], -1), RangeError],
  ["a missing index", () => new ExecutionPosition("loop_body", ["loop"]), TypeError],
  ["an index on a call", () => new ExecutionPosition("call", ["child"], 2), RangeError],
])("execution scopes reject %s", (_name, factory, cls) => {
  expect(factory).toThrow(cls);
});

test.each<[unknown, typeof TypeError | typeof RangeError, string]>([
  [true, TypeError, "execution position index must be an integer"],
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

test.each(["argument", "public view"])(
  "retained failure origins drop mutable subclass state (context, %s)",
  (mutation) => {
    const origin = new AnnotatedOrigin("writer.prompt", ["authored note"]);
    const failure = new Failure({
      code: "node.execute",
      message: "Forward failed.",
      nodeId: "writer",
      trace: [origin, new OriginRef("writer"), origin],
      causeType: "LookupError",
      traceback: "original debugging detail",
    });
    const retained = new BackwardContext(null, { failure }).failure!;

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
    expect(failure.trace[0]).toBe(origin);
  },
);

test.each<[string, () => unknown, typeof TypeError | typeof RangeError]>([
  ["array inputs", () => new BackwardContext([] as never), TypeError],
  ["a class instance in the inputs", () => new BackwardContext({ text: new Point() }), TypeError],
  ["a NaN output", () => new BackwardContext({}, { output: NaN }), RangeError],
  ["a duplicate input name", () => new BackwardContext({}, { inputNames: ["text", "text"] }), RangeError],
  ["a non-text variable name", () => new BackwardContext({}, { variableNames: [7] as never }), TypeError],
  ["a boolean start sequence", () => new BackwardContext({}, { nodeStartedSeq: true as never }), TypeError],
  ["a zero start sequence", () => new BackwardContext({}, { nodeStartedSeq: 0 }), RangeError],
  ["array node data", () => new BackwardContext({}, { nodeData: [] as never }), TypeError],
])("backward contracts reject invalid snapshots (%s)", (_name, factory, cls) => {
  expect(factory).toThrow(cls);
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
