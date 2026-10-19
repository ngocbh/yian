/**
 * Recorded defaults: the log half of `tests/engine/test_defaults_recorded.py`, the
 * `RunStartedRecord.defaults` a run starts with. Python's `ValueError` is a `RangeError` here.
 *
 * Ported: `test_record_rejects_a_bad_recorded_value` (string, bool, float `5000`, empty name; each
 * with its exact error), `test_record_rejects_a_list_of_pairs_as_a_type_error`,
 * `test_record_keeps_its_own_copy_of_the_values`. The record is built directly, not from the
 * decoded `_RECORDED_ROUTE` line: TS has no `_RECORDED_ROUTE` golden fixture.
 *
 * TS-only: the values are `bigint`s, so a `number` (even a whole one) is refused; a `Map` is
 * refused like the list of pairs; omitted defaults are `{}`; the copy is frozen.
 *
 * Not ported: the `int-name` row (JS object keys are always text);
 * `test_append_revalidates_the_recorded_values` (it forges the frozen record with
 * `object.__setattr__`).
 *
 * Ported, the codec: `test_recorded_values_encode_and_decode` and
 * `test_recorded_values_are_sorted_by_name` (also pinned as bytes). The decoded lines are built
 * from records, since TS has no `_RECORDED_ROUTE` golden fixture.
 *
 * Changed (TS yian):
 * - The log half of `test_default_run_log_is_unchanged`: schema 10 always writes `defaults`, so
 *   an all-default run writes `"defaults":{}` and reads it back as `{}` (Python omits the field
 *   for byte compatibility within schema 9).
 * - `test_decode_rejects_a_bad_recorded_value`: each row with its class and message; the string,
 *   bool and float rows are `TypeError`s where Python raised a `ValidationError` (a `ValueError`),
 *   and the empty-name row is a `RangeError`.
 *
 * Later: the run, resume and mismatch tests, with the engine.
 */

import { expect, test } from "bun:test";

import {
  decodeRecord,
  encodePool,
  encodeRecord,
  RecordEnvelope,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { VariablePool } from "../../src/typesys/index.ts";
import { type JsonObject, logJson, parseJson } from "../../src/typesys/json.ts";

/** Assert `fn` throws an error of exactly class `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** Return a run-start record with an empty pool and the given recorded defaults. */
function runStarted(defaults?: unknown): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    initialPool: encodePool(new VariablePool("start")),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    defaults: defaults as Record<string, bigint>,
  });
}

const NOT_AN_INT = "RunStarted defaults value for 'MAX_TOTAL_NODES' must be an int";

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  ["string", { MAX_TOTAL_NODES: "5000" }, TypeError, NOT_AN_INT],
  ["bool", { MAX_TOTAL_NODES: true }, TypeError, NOT_AN_INT],
  ["float", { MAX_TOTAL_NODES: 5000 }, TypeError, NOT_AN_INT],
  ["empty-name", { "": 1n }, RangeError, "RunStarted defaults names must be non-empty strings"],
])("the record rejects a bad recorded value: %s", (_name, recorded, cls, message) => {
  expectError(() => runStarted(recorded), cls, message);
});

test("the record rejects a list of pairs as a type error", () => {
  expectError(() => runStarted([["MAX_TOTAL_NODES", 5000n]]), TypeError, "RunStarted defaults must be a mapping");
  expectError(
    () => runStarted(new Map([["MAX_TOTAL_NODES", 5000n]])),
    TypeError,
    "RunStarted defaults must be a mapping",
  );
});

test("the record keeps its own copy of the values", () => {
  const recorded = { MAX_TOTAL_NODES: 5000n };
  const record = runStarted(recorded);

  recorded.MAX_TOTAL_NODES = 6000n;

  expect(record.defaults).toEqual({ MAX_TOTAL_NODES: 5000n });
  expect(Object.isFrozen(record.defaults)).toBe(true);
  expect(runStarted().defaults).toEqual({});
});

/** Encode `record` as the first line of run `direct-engine-test`. */
function encode(record: RunStartedRecord): string {
  return encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "direct-engine-test", 0, record));
}

/** The decoded payload of `line`, ints as `bigint`s. */
function payloadOf(line: string): JsonObject {
  return (parseJson(line, "codec") as JsonObject)["payload"] as JsonObject;
}

test("recorded values encode and decode", () => {
  const encoded = encode(runStarted({ MAX_TOTAL_NODES: 5000n }));

  expect(payloadOf(encoded)["defaults"]).toEqual({ MAX_TOTAL_NODES: 5000n });
  expect((decodeRecord(encoded).record as RunStartedRecord).defaults).toEqual({ MAX_TOTAL_NODES: 5000n });
});

test("recorded values are sorted by name", () => {
  const encoded = encode(runStarted({ MAX_TOTAL_NODES: 5000n, AGENT_RETRIES: 0n }));

  expect(Object.keys(payloadOf(encoded)["defaults"] as JsonObject)).toEqual(["AGENT_RETRIES", "MAX_TOTAL_NODES"]);
  expect(encoded).toContain('"defaults":{"AGENT_RETRIES":0,"MAX_TOTAL_NODES":5000}');
});

test.each<[string, unknown, new (...args: never[]) => Error, string]>([
  ["string", { MAX_TOTAL_NODES: "5000" }, TypeError, NOT_AN_INT],
  ["bool", { MAX_TOTAL_NODES: true }, TypeError, NOT_AN_INT],
  // A JS number is written as a float token, `5.0`.
  ["float", { MAX_TOTAL_NODES: 5 }, TypeError, NOT_AN_INT],
  ["empty-name", { "": 1n }, RangeError, "RunStarted defaults names must be non-empty strings"],
])("decode rejects a bad recorded value: %s", (_name, recorded, cls, message) => {
  const envelope = parseJson(encode(runStarted()), "codec") as { payload: Record<string, unknown> };
  envelope.payload["defaults"] = recorded;
  expectError(() => decodeRecord(logJson(envelope)), cls, message);
});

test("the default run log writes empty defaults", () => {
  const encoded = encode(runStarted());
  const decoded = decodeRecord(encoded);

  expect(encoded).toContain('"defaults":{}');
  expect((decoded.record as RunStartedRecord).defaults).toEqual({});
  expect(encodeRecord(decoded)).toBe(encoded);
});
