/**
 * Transactional run-state initialization: a port of `tests/engine/test_run_state_initialization.py`.
 *
 * Ported: `test_run_state_starts_pristine_around_the_owned_graph`,
 * `test_historical_start_prepares_recorded_schema_bytes`,
 * `test_prepare_run_started_requires_an_exact_nonnegative_worker_count` (3 rows),
 * `test_prepare_run_started_requires_a_pristine_state_with_a_start_node`,
 * `test_prepare_run_started_rejects_pool_identity_and_start_priming_mismatches` and
 * `test_prepare_run_started_validates_the_public_envelope_contract` (3 rows). Python's
 * `ValueError` is a `RangeError` here; each rejected row names its one class and message, where
 * Python accepts either `TypeError` or `ValueError` or matches a part of the message.
 *
 * Changed: `test_private_plan_contains_no_mutable_transition_payload`: the plan is a frozen plain
 * object; each own field is a primitive (the envelope is canonical text, not bytes), a `Node`, the
 * originating state or flow, `null`, or a frozen value, and assigning a field is a `TypeError`.
 *
 * Not ported: `test_historical_start_requires_an_exact_supported_schema` (`prepareRunStarted` has
 * no schema parameter: schema 10 is the only one); the `1.0` row of the worker-count test (TS
 * types: `1.0` is `1`).
 */

import { expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import { Node } from "../../src/nodes/base.ts";
import { NodeState } from "../../src/runtime/common.ts";
import { decodeRecord, encodeRecord, RecordEnvelope, SCHEMA_VERSION } from "../../src/runtime/log.ts";
import { prepareRunStarted, RunState } from "../../src/runtime/state.ts";
import { flow, pool, RUN_ID, startData, stateSnapshot } from "./run_state_fixtures.ts";

/** Assert `fn` throws an error of exactly class `cls` whose message matches `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: RegExp): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toMatch(message);
}

test("run state starts pristine around the owned graph", () => {
  const owned = flow();

  const state = new RunState(owned);

  expect(state.flow).toBe(owned);
  expect(state.pool.startId).toBe(owned.startId);
  expect(state.pool.store.size).toBe(0);
  expect(state.pool.system.size).toBe(0);
  expect(state.nodeState).toEqual(new Map([...owned.nodes.keys()].map((id) => [id, NodeState.UNKNOWN])));
  expect(state.edgeState).toEqual(new Map(owned.edges.map((edge) => [edge.id, NodeState.UNKNOWN])));
  expect(state.ready).toEqual([]);
  expect(state.inFlight).toEqual(new Set());
  expect(state.acceptedStarts).toEqual(new Map());
  expect("conversationState" in state).toBe(false);
  expect(state.result).toBeNull();
  expect(state.depth).toEqual(new Map());
  expect(state.wiringOverrides).toEqual(new Map());
  expect(state.runId).toBeNull();
  expect(state.flowFingerprint).toBeNull();
  expect(state.nextSequence).toBe(0);
  expect(state.numWorkers).toBe(0);
  expect(state.input).toEqual({});
  expect(state.effectiveLlmByStaticPath).toEqual({});
});

test("historical start prepares recorded schema bytes", () => {
  const state = new RunState(flow());
  const before = stateSnapshot(state);
  const prepared = prepareRunStarted(state, startData(state.flow));
  const expected = new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 0, prepared.record);
  expect(prepared._plan.envelopeJson).toBe(encodeRecord(expected));
  expect(decodeRecord(prepared._plan.envelopeJson).schema).toBe(SCHEMA_VERSION);
  expect(stateSnapshot(state)).toEqual(before);
});

test.each([
  ["-1", -1, RangeError],
  ["true", true, TypeError],
  ['"2"', "2", TypeError],
] as const)("prepareRunStarted requires an exact non-negative worker count: %s", (_name, numWorkers, cls) => {
  const owned = flow();
  const state = new RunState(owned);
  const before = stateSnapshot(state);

  expectError(
    () => prepareRunStarted(state, startData(owned, { numWorkers: numWorkers as number })),
    cls,
    /^num_workers must be a non-negative integer$/,
  );

  expect(stateSnapshot(state)).toEqual(before);
});

test("prepareRunStarted requires a pristine state with a START node", () => {
  const owned = flow();
  const state = new RunState(owned);
  state.ready.push("work");

  expectError(() => prepareRunStarted(state, startData(owned)), RangeError, /pristine/);

  const missingStart = flow();
  missingStart.nodes.delete(missingStart.startId);
  const missingState = new RunState(missingStart);
  expectError(
    () => prepareRunStarted(missingState, startData(owned)),
    RangeError,
    /^flow START node '__start__' is missing$/,
  );
});

test("prepareRunStarted rejects pool identity and START priming mismatches", () => {
  const owned = flow();

  const wrongStart = pool(owned);
  (wrongStart as { startId: string }).startId = "other";
  expectError(
    () => prepareRunStarted(new RunState(owned), startData(owned, { pool: wrongStart })),
    RangeError,
    /^initial pool start_id mismatch: expected '__start__', found 'other'$/,
  );

  const wrongRun = pool(owned, "different");
  expectError(
    () => prepareRunStarted(new RunState(owned), startData(owned, { pool: wrongRun })),
    RangeError,
    /^initial pool system\.run_id must equal RunStartData\.run_id$/,
  );

  const scalarStart = pool(owned);
  scalarStart.set(owned.startId, "not a record");
  expectError(
    () => prepareRunStarted(new RunState(owned), startData(owned, { pool: scalarStart })),
    RangeError,
    /^initial pool primed START value must be a dict$/,
  );
});

test.each([
  [
    "a fingerprint that isn't a digest",
    (owned: CompiledFlow) => ({ ...startData(owned), flowFingerprint: "not-a-digest" }),
    RangeError,
    /^flow_fingerprint must be 64 lowercase hex digits$/,
  ],
  [
    "an empty run id",
    (owned: CompiledFlow) => startData(owned, { runId: "" }),
    RangeError,
    /^run-record envelope run_id must be a non-empty string$/,
  ],
  [
    "an agent configuration that isn't an object",
    (owned: CompiledFlow) => ({ ...startData(owned), effectiveLlmByStaticPath: { agent: "bad" } }),
    TypeError,
    /^effective_llm_by_static_path value for 'agent' must be a mapping$/,
  ],
] as const)("prepareRunStarted validates the public envelope contract: %s", (_name, dataOf, cls, message) => {
  const owned = flow();
  const data = dataOf(owned) as Parameters<typeof prepareRunStarted>[1];

  expectError(() => prepareRunStarted(new RunState(owned), data), cls, message);
});

test("the private plan contains no mutable transition payload", () => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  const plan = prepared._plan;

  expect(plan.originatingState).toBe(state);
  expect(plan.originatingFlow).toBe(owned);
  expect(typeof plan.envelopeJson).toBe("string");
  expect(Object.isFrozen(plan)).toBe(true);
  for (const value of Object.values(plan)) {
    const immutable =
      value === null ||
      (typeof value !== "object" && typeof value !== "function") ||
      value instanceof Node ||
      value === state ||
      value === owned ||
      Object.isFrozen(value);
    expect(immutable).toBe(true);
  }
  expect(() => {
    (plan as { envelopeJson: string }).envelopeJson = "changed";
  }).toThrow(TypeError);
});
