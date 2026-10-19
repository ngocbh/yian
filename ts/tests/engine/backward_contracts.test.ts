/**
 * Backward results and execution scopes: a port of `tests/engine/test_backward_contracts.py`
 * (the feedback value checks, `BackwardResult`, and the two `ExecutionScope`/`ExecutionPosition`
 * tests).
 *
 * TS-only: `copyBackwardValue` keeps a `__proto__` key as data, writes `-0` as `0`, keeps a
 * shared subtree shared, and names the nested value it refuses; a `null` channel is an empty one.
 *
 * Changed: `FrozenInstanceError` is the `TypeError` an assignment to a frozen object throws in a
 * module; dataclass `!=` is `not.toEqual`; each rejected row names its one error class, where
 * Python accepts either `TypeError` or `ValueError` (`RangeError`); a `BackwardResult` holds
 * frozen values, so a reader's edit throws instead of changing a private copy.
 *
 * Left for later: the backward contexts and the result check need `BackwardContext`; memory and
 * the run contracts need the runtime.
 *
 * Not ported: the rows that pass a list where Python wants a tuple (`static_path=["node"]`,
 * `occurrence_path=[]`, a list batch) and the tuple type checks: TS has only arrays. The
 * feedback rows `(("Bad.",), ["text"])` and `({1: "Bad."}, {1: "text"})` are valid in TS.
 */

import { expect, test } from "bun:test";

import { type FeedbackValue, hasFeedback, validateFeedback } from "../../src/expr/index.ts";
import { BackwardResult, copyBackwardValue } from "../../src/nodes/base.ts";
import { ExecutionPosition, ExecutionScope } from "../../src/nodes/contracts.ts";

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
  ["NaN", { a: [NaN] }, RangeError, "value['a'][0] must not contain NaN"],
  ["a Map", [new Map()], TypeError, "value[0] must contain only JSON values"],
  ["a class instance", { p: new Point() }, TypeError, "value['p'] must contain only JSON values"],
  ["undefined", [{ u: undefined }], TypeError, "value[0]['u'] must contain only JSON values"],
])("copyBackwardValue refuses %s with the nested name", (_name, value, cls, message) => {
  expect(() => copyBackwardValue(value, "value")).toThrow(cls);
  expect(() => copyBackwardValue(value, "value")).toThrow(message);
});
