/**
 * Node steps: the log half of `tests/engine/test_node_steps_log.py`, the `NodeStepRecord` a
 * running node appends. Python's `ValueError` is a `RangeError` here.
 *
 * Ported: `test_node_step_record_detaches_the_callers_step`,
 * `test_node_step_record_rejects_invalid_values` (list, text, null, NaN, a class instance, a
 * `Set`, an empty and a non-text node id; each with its exact error).
 *
 * Ported, the wire tests: `test_node_step_round_trips_under_its_wire_kind` (schema 10),
 * `test_node_step_wire_payload_rejects_invalid_shapes` (list, text, empty node id, missing step,
 * extra field; each with its exact class and message),
 * `test_node_step_wire_payload_rejects_non_finite_numbers` (`invalid run-record JSON`, caused by
 * the reader's `NaN is not JSON`).
 *
 * TS-only: a step writes `-0` as `0`; a `FileRef` in a step is refused; a wire step keeps
 * yian's infinity tag as `Infinity`.
 *
 * Changed:
 * - the `infinity` row of `test_node_step_record_rejects_invalid_values`: a step keeps
 *   `±Infinity` (a TS yian value).
 * - `test_node_step_too_deep_for_python_is_a_value_error`: a 5000-deep step is accepted (the TS
 *   codec is iterative).
 *
 * Not ported: the `non-text-key` row (JS object keys are always text); the reducer tests (they
 * need the run state).
 */

import { expect, test } from "bun:test";

import { decodeRecord, encodeRecord, NodeStepRecord, RecordEnvelope, SCHEMA_VERSION } from "../../src/runtime/log.ts";
import { FileRef, TypeCheckError } from "../../src/typesys/index.ts";
import { JSONDecodeError, parseJson } from "../../src/typesys/json.ts";

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

const FAILED = {
  event: "failed",
  attempt: 1n,
  error: { type: "TimeoutError", message: "slow" },
  retry_in: 1.5,
};

test("a node step round trips under its wire kind", () => {
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "run-1", 3, new NodeStepRecord("a", FAILED));

  const encoded = encodeRecord(envelope);
  const decoded = decodeRecord(encoded);

  expect(decoded).toEqual(envelope);
  expect(encodeRecord(decoded)).toBe(encoded);
  const wire = parseJson(encoded, "codec") as Record<string, unknown>;
  expect(wire["kind"]).toBe("node_step");
  expect(wire["schema"]).toBe(10n);
  expect(wire["payload"]).toEqual({ node_id: "a", step: FAILED });
});

test("a wire step keeps yian's infinity tag", () => {
  const text =
    '{"kind":"node_step","payload":{"node_id":"a","step":{"high":{"$float":"inf"}}},' +
    '"run_id":"run-1","schema":10,"seq":0}';
  const record = decodeRecord(text).record as NodeStepRecord;
  expect(record.step).toEqual({ high: Infinity });
  expect(encodeRecord(decodeRecord(text))).toBe(text);
});

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  ["list", { node_id: "a", step: [1] }, TypeError, "node_step payload step must be an object"],
  ["text", { node_id: "a", step: "text" }, TypeError, "node_step payload step must be an object"],
  ["empty-node-id", { node_id: "", step: {} }, RangeError, "node_step payload node_id must be a non-empty string"],
  ["missing-step", { node_id: "a" }, RangeError, 'invalid node_step payload fields; missing=["step"], extra=[]'],
  [
    "extra-field",
    { node_id: "a", step: {}, extra: true },
    RangeError,
    'invalid node_step payload fields; missing=[], extra=["extra"]',
  ],
])("a node step wire payload rejects an invalid shape: %s", (_name, payload, cls, message) => {
  const text = JSON.stringify({ schema: SCHEMA_VERSION, run_id: "run-1", seq: 0, kind: "node_step", payload });
  expectError(() => decodeRecord(text), cls, message);
});

test("a node step wire payload rejects non-finite numbers", () => {
  const text =
    '{"schema":10,"run_id":"run-1","seq":0,"kind":"node_step","payload":{"node_id":"a","step":{"retry_in":NaN}}}';
  const caught = expectError(() => decodeRecord(text), RangeError, "invalid run-record JSON") as Error;
  expect((caught.cause as Error).constructor).toBe(JSONDecodeError);
  expect((caught.cause as Error).message).toStartWith("NaN is not JSON");
});

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
