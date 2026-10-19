/**
 * Node steps: the log half of `tests/engine/test_node_steps_log.py`, the `NodeStepRecord` a
 * running node appends. Python's `ValueError` is a `RangeError` here.
 *
 * Ported: `test_node_step_record_detaches_the_callers_step`,
 * `test_node_step_record_rejects_invalid_values` (list, text, null, NaN, a class instance, a
 * `Set`, an empty and a non-text node id; each with its exact error).
 *
 * TS-only: a step keeps `±Infinity` (a TS yian value) and writes `-0` as `0`; a `FileRef` in a
 * step is refused.
 *
 * Changed: `test_node_step_too_deep_for_python_is_a_value_error`: a 5000-deep step is accepted
 * (the TS codec is iterative).
 *
 * Not ported: the `infinity` and `non-text-key` rows (a TS float may be infinite; JS object keys
 * are always text); the wire-payload tests and the reducer tests (they need the codec and the
 * run state).
 */

import { expect, test } from "bun:test";

import { NodeStepRecord } from "../../src/runtime/log.ts";
import { FileRef, TypeCheckError } from "../../src/typesys/index.ts";

/** Assert `fn` throws an error of exactly class `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): unknown {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
  return caught;
}

/** Return a JSON object nested `levels` objects deep, itself included. */
function nested(levels: number): Record<string, unknown> {
  let value: Record<string, unknown> = {};
  for (let i = 0; i < levels - 1; i++) value = { inner: value };
  return value;
}

class Thing {}

test("a node step record detaches the caller's step", () => {
  const step = { event: "started", attempt: 2n, tags: ["x", "y"] };
  const record = new NodeStepRecord("a", step);
  step.attempt = 99n;
  step.tags.push("z");

  expect(record.step).toEqual({ attempt: 2n, event: "started", tags: ["x", "y"] });
  expect(Object.keys(record.step)).toEqual(["attempt", "event", "tags"]);
});

test.each<[string, unknown, unknown, new (...args: never[]) => Error, string, (string | number)[] | null]>([
  ["list", "a", ["not", "an", "object"], TypeError, "node step must be a JSON object", null],
  ["text", "a", "text", TypeError, "node step must be a JSON object", null],
  ["null", "a", null, TypeError, "node step must be a JSON object", null],
  ["NaN", "a", { retry_in: NaN }, TypeCheckError, "NaN is not a yian value", ["retry_in"]],
  ["a class instance", "a", { when: new Thing() }, TypeCheckError, "<Thing object> is not a JSON value", ["when"]],
  ["a Set", "a", { items: new Set([1, 2]) }, TypeCheckError, "<Set object> is not a JSON value", ["items"]],
  [
    "a FileRef",
    "a",
    { file: new FileRef({ uri: "memory://answer" }) },
    TypeCheckError,
    '{"uri":"memory://answer","mime":null,"name":null} is not a JSON value',
    ["file"],
  ],
  ["an empty node id", "", { event: "started" }, RangeError, "node_id must be non-empty", null],
  ["a non-text node id", 1, { event: "started" }, TypeError, "node_id must be a string", null],
])("a node step record rejects %s", (_name, nodeId, step, cls, message, path) => {
  const caught = expectError(() => new NodeStepRecord(nodeId as never, step as never), cls, message);
  if (path !== null) expect((caught as TypeCheckError).path).toEqual(path);
});

test("a node step record keeps infinite floats and writes -0 as 0", () => {
  const record = new NodeStepRecord("a", { high: Infinity, low: [-Infinity], zero: -0 });

  expect(record.step).toEqual({ high: Infinity, low: [-Infinity], zero: 0 });
  expect(Object.is(record.step["zero"], 0)).toBe(true);
});

test("a node step too deep for Python is accepted", () => {
  const record = new NodeStepRecord("a", nested(5000));

  let value: unknown = record.step;
  let depth = 1;
  while (Object.keys(value as object).length > 0) {
    expect(Object.isFrozen(value)).toBe(true);
    value = (value as Record<string, unknown>)["inner"];
    depth++;
  }
  expect(depth).toBe(5000);
});
