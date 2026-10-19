/**
 * Recorded defaults: the log half of `tests/engine/test_defaults_recorded.py`, the
 * `RunStartedRecord.defaults` a run starts with. Python's `ValueError` is a `RangeError` here.
 *
 * Ported: `test_record_rejects_a_bad_recorded_value` (string, bool, float `5000`, empty name; each
 * with its exact error), `test_record_rejects_a_list_of_pairs_as_a_type_error`,
 * `test_record_keeps_its_own_copy_of_the_values`. The record is built directly, not from the
 * decoded `_RECORDED_ROUTE` line (the codec comes later).
 *
 * TS-only: the values are `bigint`s, so a `number` (even a whole one) is refused; a `Map` is
 * refused like the list of pairs; omitted defaults are `{}`; the copy is frozen.
 *
 * Not ported: the `int-name` row (JS object keys are always text);
 * `test_append_revalidates_the_recorded_values` (it forges the frozen record with
 * `object.__setattr__`).
 *
 * Later: the encode and decode tests, with the record codec; the run, resume and mismatch tests,
 * with the engine.
 */

import { expect, test } from "bun:test";

import { encodePool, RunStartedRecord } from "../../src/runtime/log.ts";
import { VariablePool } from "../../src/typesys/index.ts";

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
