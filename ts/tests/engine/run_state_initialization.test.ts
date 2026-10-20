/**
 * Transactional run-state initialization: a port of `tests/engine/test_run_state_initialization.py`.
 *
 * Ported: `test_run_state_starts_pristine_around_the_owned_graph`,
 * `test_historical_start_prepares_recorded_schema_bytes`,
 * `test_prepare_run_started_requires_an_exact_nonnegative_worker_count` (3 rows),
 * `test_prepare_run_started_requires_a_pristine_state_with_a_start_node`,
 * `test_prepare_run_started_rejects_pool_identity_and_start_priming_mismatches`,
 * `test_prepare_run_started_validates_the_public_envelope_contract` (3 rows),
 * `test_commit_prepared_appends_before_applying_the_total_plan`,
 * `test_commit_prepared_append_failure_leaves_state_and_observers_untouched`,
 * `test_commit_prepared_rejects_a_different_pristine_state_before_append`,
 * `test_commit_prepared_rejects_flow_or_boundary_replacement_before_append` (4 rows),
 * `test_commit_prepared_rejects_state_and_log_identity_mismatches_before_append` (4 rows),
 * `test_commit_prepared_rejects_a_custom_log_return_mismatch_without_applying` (4 rows) and
 * `test_mismatched_append_return_leaves_the_advanced_log_as_recovery_authority`. Python's
 * `ValueError` is a `RangeError` here; each rejected row names its one class and message, where
 * Python accepts either `TypeError` or `ValueError` or matches a part of the message.
 *
 * Changed:
 * - `test_private_plan_contains_no_mutable_transition_payload`: the plan is a frozen plain object;
 *   each own field is a primitive (the envelope is canonical text, not bytes), a `Node`, the
 *   originating state or flow, `null`, or a frozen value; assigning a field, or writing into the
 *   envelope text's characters, is a `TypeError`.
 * - `test_run_started_preparation_is_immutable_and_commit_restores_complete_state`: `RunStartData`
 *   is an interface, so its freeze half has no counterpart; assigning a `PreparedTransition` field
 *   is a `TypeError`; the tuple START value is an array, and its int is a `bigint`.
 * - `test_append_result_must_preserve_the_complete_validated_envelope`: TS envelopes and records
 *   are frozen, so the log returns a different envelope: one built on the envelope prototype with
 *   schema `true`, or one whose record holds the float `1` where the prepared record holds the int
 *   `1n`.
 * - `test_commit_prepared_rejects_mutation_of_the_public_prepared_record`: the record is frozen,
 *   so a forged `PreparedTransition` pairs a changed record with the original plan.
 * - `test_commit_prepared_rejects_every_stale_state_mutation_before_append`: the 7 rows that change
 *   values (`deferredPrune` is a `Set`).
 * - `test_committed_state_does_not_alias_old_containers_or_prepared_payloads`: the prepared record
 *   is deep-frozen, so each change to its payload is a `TypeError`.
 *
 * TS-only, for the part-2 entry points: `prepareExpanded` refuses an uninitialized state and a
 * node that isn't a graph spawner, and `replayRecords` throws `not ported yet (T4-5)`; a hand-built
 * backward plan with no trace change is refused by the post-terminal trace check before anything
 * else, and one with a sequence-only trace change reaches its commit stub; a hand-built growth plan
 * holding a RunStarted record fails the growth commit's record check. The log stays empty.
 *
 * TS-only, for the interrupted-attempt counts Python doesn't keep: `prepareRunStarted` refuses a
 * count or a last stop sequence on the fresh state, and an initialized state refuses a wrong
 * container, a count that isn't a positive integer by node id, or a stop sequence that isn't an
 * integer in `[1, nextSequence)`.
 *
 * Not ported:
 * - `test_historical_start_requires_an_exact_supported_schema`: `prepareRunStarted` has no schema
 *   parameter, since schema 10 is the only one.
 * - TS types: the `1.0` row of the worker-count test (`1.0` is `1`), and the `ready-container`,
 *   `in-flight-container`, `node-state-container` (dict subclass) and `node-disposition-type`
 *   (`NodeState.UNKNOWN` is the text `"unknown"`) rows of the stale-mutation test.
 */

import { expect, test } from "bun:test";

import { type CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { RunStarted } from "../../src/events.ts";
import { Grow, Node } from "../../src/nodes/base.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { NodeState } from "../../src/runtime/common.ts";
import {
  decodeRecord,
  encodePool,
  encodeRecord,
  MemoryRunLog,
  NodeStartedRecord,
  RecordEnvelope,
  type RunRecord,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  PreparedTransition,
  prepareExpanded,
  prepareRunStarted,
  prepareRunStopped,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { TraceDelta } from "../../src/runtime/trace.ts";
import {
  FailingLog,
  FINGERPRINT,
  flow,
  initializedAndStarted,
  pool,
  ReturningLog,
  RUN_ID,
  startData,
  stateSnapshot,
} from "./run_state_fixtures.ts";

const APPEND_MISMATCH = /^run-log append result does not match prepared transition$/;

/** Return a run-start record with `record`'s fields and `wiring` as its START wiring override. */
function withWiring(record: RunStartedRecord, wiring: Record<string, unknown>): RunStartedRecord {
  return new RunStartedRecord({ ...record, startWiringOverride: wiring });
}

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

  expectError(
    () => prepareRunStarted(state, startData(owned)),
    RangeError,
    /^RunStarted requires a pristine RunState at sequence 0$/,
  );

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
  // Strict-mode code can't write into a string's characters, as Python can't into bytes.
  expect(() => {
    (plan.envelopeJson as unknown as string[])[0] = "x";
  }).toThrow(TypeError);
});

test.each(["schema", "numeric payload"])(
  "the append result must preserve the complete validated envelope: %s",
  (corruption) => {
    /** Return an envelope with an invalid schema, or a float where the record holds an int. */
    class CorruptingLog extends MemoryRunLog {
      override append(record: RunRecord): RecordEnvelope {
        const envelope = super.append(record);
        if (corruption === "schema") {
          const forged = Object.create(RecordEnvelope.prototype) as RecordEnvelope;
          return Object.assign(forged, { ...envelope, schema: true });
        }
        const started = envelope.record as RunStartedRecord;
        const changed = withWiring(started, { ...started.startWiringOverride, numeric: 1 });
        return new RecordEnvelope(envelope.schema, envelope.runId, envelope.seq, changed);
      }
    }

    const state = new RunState(flow());
    const data = startData(state.flow);
    data.startWiringOverride = { ...data.startWiringOverride, numeric: 1n };
    const prepared = prepareRunStarted(state, data);
    const before = stateSnapshot(state);
    const log = new CorruptingLog(RUN_ID);
    expectError(() => commitPrepared(log, state, prepared), RangeError, APPEND_MISMATCH);
    expect(stateSnapshot(state)).toEqual(before);
    expect(log.nextSequence).toBe(1);
  },
);

test("RunStarted preparation is immutable and commit restores the complete state", () => {
  const owned = flow();
  const state = new RunState(owned);
  const given = pool(owned);
  const data = startData(owned, { pool: given });
  const before = stateSnapshot(state);

  const prepared = prepareRunStarted(state, data);

  expect(stateSnapshot(state)).toEqual(before);
  expect(prepared.record).toBeInstanceOf(RunStartedRecord);
  expect(prepared.expectedRunId).toBe(RUN_ID);
  expect(prepared.expectedSequence).toBe(0);
  expect(prepared.observers).toEqual([new RunStarted()]);
  expect(() => {
    (prepared as { expectedSequence: number }).expectedSequence = 1;
  }).toThrow(TypeError);

  // Preparation owns canonical copies rather than keeping caller-owned containers.
  given.set(owned.startId, { name: "mutated" });
  given.addSystem("today", "never");
  ((data.startWiringOverride["metadata"] as Record<string, unknown>)["labels"] as string[]).push("mutated");
  (data.effectiveLlmByStaticPath["agent"]!["options"] as Record<string, unknown>)["temperature"] = 1.0;

  const log = new MemoryRunLog(RUN_ID);
  const observers = commitPrepared(log, state, prepared);

  expect(observers).toEqual([new RunStarted()]);
  expect(log.records()).toEqual([new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 0, prepared.record)]);
  expect(state.runId).toBe(RUN_ID);
  expect(state.flowFingerprint).toBe(FINGERPRINT);
  expect(state.nextSequence).toBe(1);
  expect(state.numWorkers).toBe(3);
  expect(state.pool).not.toBe(given);
  expect(state.pool.startId).toBe(owned.startId);
  expect(state.pool.get(owned.startId)).toEqual({ name: "Ada", count: 2n, aliases: ["A", "B"] });
  expect(state.pool.get("preseeded")).toEqual({ retained: true });
  expect(state.pool.resolve("system", ["run_id"])).toBe(RUN_ID);
  expect(state.pool.resolve("system", ["today"])).toBe("2026-08-13");
  expect(state.pool.resolve("system", ["now"])).toBe("2026-08-13T12:34:56+00:00");
  expect(state.pool.resolve("system", ["tenant"])).toBe("tests");
  expect(encodePool(state.pool)).toEqual((prepared.record as RunStartedRecord).initialPool);
  expect(state.input).toEqual({ name: "Ada", count: 2n, aliases: ["A", "B"] });
  expect(state.wiringOverrides).toEqual(
    new Map([[owned.startId, { name: "${input.name}", metadata: { labels: ["one", "two"] } }]]),
  );
  expect(state.effectiveLlmByStaticPath).toEqual({
    agent: { provider: "openai", model: "gpt-test", options: { temperature: 0.2 } },
  });
  expect(state.ready).toEqual([owned.startId]);
  expect(state.nodeState.get(owned.startId)).toBe(NodeState.UNKNOWN);
  expect(state.inFlight).toEqual(new Set());
  state.pool.set(owned.startId, { name: "coerced later" });
  expect(state.input).toEqual({ name: "Ada", count: 2n, aliases: ["A", "B"] });
});

/** A memory log that proves the reducer hasn't applied before append. */
class InspectingLog extends MemoryRunLog {
  readonly #state: RunState;
  sawAppend = false;

  constructor(state: RunState) {
    super(RUN_ID);
    this.#state = state;
  }

  override append(record: RunRecord): RecordEnvelope {
    expect(this.#state.runId).toBeNull();
    expect(this.#state.nextSequence).toBe(0);
    expect(this.#state.ready).toEqual([]);
    expect(this.#state.input).toEqual({});
    this.sawAppend = true;
    return super.append(record);
  }
}

test("commitPrepared appends before applying the total plan", () => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  const log = new InspectingLog(state);

  const observers = commitPrepared(log, state, prepared);

  expect(log.sawAppend).toBe(true);
  expect(state.runId).toBe(RUN_ID);
  expect(observers).toEqual([new RunStarted()]);
});

test("commitPrepared append failure leaves state and observers untouched", () => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  const before = stateSnapshot(state);

  expectError(() => commitPrepared(new FailingLog(RUN_ID), state, prepared), Error, /^disk full$/);

  expect(stateSnapshot(state)).toEqual(before);
  expect(prepared.observers).toEqual([new RunStarted()]);
});

test("commitPrepared rejects a changed public prepared record", () => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  const record = prepared.record as RunStartedRecord;
  const forged = new PreparedTransition(
    withWiring(record, { ...record.startWiringOverride, name: "changed" }),
    prepared.expectedRunId,
    prepared.expectedSequence,
    prepared._plan,
    prepared.observers,
  );
  const before = stateSnapshot(state);
  const log = new MemoryRunLog(RUN_ID);

  expectError(() => commitPrepared(log, state, forged), RangeError, /^prepared record changed after validation$/);

  expect(stateSnapshot(state)).toEqual(before);
  expect(log.nextSequence).toBe(0);
});

test("commitPrepared rejects a different pristine state before append", () => {
  const owned = flow();
  const originatingState = new RunState(owned);
  const otherState = new RunState(owned);
  const prepared = prepareRunStarted(originatingState, startData(owned));
  const log = new MemoryRunLog(RUN_ID);
  const before = stateSnapshot(otherState);

  expectError(
    () => commitPrepared(log, otherState, prepared),
    RangeError,
    /^prepared transition belongs to a different originating RunState$/,
  );

  expect(stateSnapshot(otherState)).toEqual(before);
  expect(log.nextSequence).toBe(0);
});

test.each([
  [
    "flow object",
    (state: RunState): void => {
      state.flow = flow();
    },
  ],
  [
    "start id",
    (state: RunState): void => {
      (state.flow as { startId: string }).startId = "work";
    },
  ],
  ["start node", (state: RunState): void => void state.flow.nodes.set(START_ID, new StartNode(START_ID))],
  [
    "end node",
    (state: RunState): void => void state.flow.nodes.set(END_ID, EndNode.value(END_ID, { outputName: null })),
  ],
] as const)("commitPrepared rejects flow or boundary replacement before append: %s", (_name, replace) => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  replace(state);
  const log = new MemoryRunLog(RUN_ID);

  expectError(
    () => commitPrepared(log, state, prepared),
    RangeError,
    /^transition requires its originating flow and boundary identities$/,
  );

  expect(log.nextSequence).toBe(0);
});

test.each([
  ["ready frontier", (state: RunState) => state.ready.push("work")],
  ["node disposition", (state: RunState) => state.nodeState.set("work", NodeState.TAKEN)],
  ["edge disposition", (state: RunState) => state.edgeState.set("start-work", NodeState.TAKEN)],
  ["wiring", (state: RunState) => state.wiringOverrides.set("work", {})],
  ["pool", (state: RunState) => state.pool.addSystem("stale", true)],
  ["growth owner", (state: RunState) => state.growthOwner.set("work", "start")],
  ["deferred prune", (state: RunState) => (state.deferredPrune = new Set(["work"]))],
] as const)("commitPrepared rejects every stale state mutation before append: %s", (_name, mutate) => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  mutate(state);
  const before = stateSnapshot(state);
  const log = new MemoryRunLog(RUN_ID);

  expectError(
    () => commitPrepared(log, state, prepared),
    RangeError,
    /^RunStarted requires a pristine RunState at sequence 0$/,
  );

  expect(stateSnapshot(state)).toEqual(before);
  expect(log.nextSequence).toBe(0);
});

test("committed state doesn't alias old containers or prepared payloads", () => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  const record = prepared.record as RunStartedRecord;
  const oldReady = state.ready;
  const oldPool = state.pool;
  const oldInFlight = state.inFlight;
  const oldNodeState = state.nodeState;
  const oldEdgeState = state.edgeState;
  const oldDepth = state.depth;
  const oldWiring = state.wiringOverrides;
  const oldInput = state.input;
  const oldEffective = state.effectiveLlmByStaticPath;

  commitPrepared(new MemoryRunLog(RUN_ID), state, prepared);
  const committed = stateSnapshot(state);

  oldReady.push("old-ready");
  oldPool.addSystem("old", true);
  oldInFlight.add("old-flight");
  oldNodeState.set("work", NodeState.TAKEN);
  oldEdgeState.set("start-work", NodeState.TAKEN);
  oldDepth.set("work", 9);
  oldWiring.set("work", { x: 1n });
  oldInput["old"] = true;
  oldEffective["old"] = { model: "wrong" };
  // The prepared record is deep-frozen, so its payload can't be changed at all.
  expect(() => {
    delete (record.initialPool.system as Record<string, unknown>)["run_id"];
  }).toThrow(TypeError);
  expect(() => {
    ((record.startWiringOverride["metadata"] as Record<string, unknown>)["labels"] as string[]).push("wrong");
  }).toThrow(TypeError);
  expect(() => {
    (record.effectiveLlmByStaticPath["agent"] as Record<string, unknown>)["model"] = "wrong";
  }).toThrow(TypeError);

  expect(stateSnapshot(state)).toEqual(committed);
  expect(state.ready).not.toBe(oldReady);
  expect(state.pool).not.toBe(oldPool);
  expect(state.inFlight).not.toBe(oldInFlight);
  expect(state.nodeState).not.toBe(oldNodeState);
  expect(state.edgeState).not.toBe(oldEdgeState);
  expect(state.depth).not.toBe(oldDepth);
  expect(state.wiringOverrides).not.toBe(oldWiring);
  expect(state.input).not.toBe(oldInput);
  expect(state.effectiveLlmByStaticPath).not.toBe(oldEffective);
});

test.each([
  ["log run id", /^run-log run_id mismatch: prepared 'run-123', log 'other'$/],
  ["log sequence", /^run-log sequence mismatch: prepared 0, log 1$/],
  ["state sequence", /^RunStarted requires a pristine RunState at sequence 0$/],
  ["state run id", /^RunStarted requires a pristine RunState at sequence 0$/],
] as const)("commitPrepared rejects state and log identity mismatches before append: %s", (mismatch, message) => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  const log = new MemoryRunLog(mismatch === "log run id" ? "other" : RUN_ID);
  if (mismatch === "log sequence") log.append(new NodeStartedRecord("already-there"));
  else if (mismatch === "state sequence") state.nextSequence = 1;
  else if (mismatch === "state run id") state.runId = "already-started";
  const before = stateSnapshot(state);
  const beforeSequence = log.nextSequence;

  expectError(() => commitPrepared(log, state, prepared), RangeError, message);

  expect(stateSnapshot(state)).toEqual(before);
  expect(log.nextSequence).toBe(beforeSequence);
});

test.each([
  ["run id", (record: RunRecord) => new RecordEnvelope(SCHEMA_VERSION, "other", 0, record)],
  ["sequence", (record: RunRecord) => new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 1, record)],
  ["record", () => new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 0, new NodeStartedRecord("different"))],
  ["log frontier", (record: RunRecord) => new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 0, record)],
] as const)("commitPrepared rejects a custom log return mismatch without applying: %s", (_name, result) => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  const before = stateSnapshot(state);

  expectError(() => commitPrepared(new ReturningLog(result), state, prepared), RangeError, APPEND_MISMATCH);

  expect(stateSnapshot(state)).toEqual(before);
});

/** A memory log that retains the right record but breaks the append-return protocol. */
class RetainingMismatchedLog extends MemoryRunLog {
  override append(record: RunRecord): RecordEnvelope {
    const retained = super.append(record);
    return new RecordEnvelope(retained.schema, retained.runId, retained.seq + 1, retained.record);
  }
}

test("a mismatched append return leaves the advanced log as recovery authority", () => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  const before = stateSnapshot(state);
  const log = new RetainingMismatchedLog(RUN_ID);

  expectError(() => commitPrepared(log, state, prepared), RangeError, APPEND_MISMATCH);

  expect(stateSnapshot(state)).toEqual(before);
  expect(log.nextSequence).toBe(1);
  const retained = log.records();
  expect(retained).toEqual([new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 0, prepared.record)]);
  expect(retained[0]!.record).toBeInstanceOf(RunStartedRecord);
});

test("growth refuses an uninitialized state and a non-spawner; the replay entry point isn't ported yet", () => {
  const owned = flow();
  const state = new RunState(owned);
  const builder = () => owned;
  expectError(() => prepareExpanded(state, "work", new Grow({}), builder), RangeError, /^run state is not initialized$/);
  expectError(() => replayRecords(state, [], { growBuilder: builder }), Error, /^not ported yet \(T4-5\)$/);

  const { state: started } = initializedAndStarted(flow());
  expectError(
    () => prepareExpanded(started, START_ID, new Grow({}), builder),
    RangeError,
    /^NodeExpanded node '__start__' is not a graph spawner$/,
  );
});

/** Return a prepared RunStarted whose plan is replaced by `kind`'s, with `traceDelta`. */
function handBuilt(kind: "backward" | "growth", traceDelta: TraceDelta | null): [RunState, PreparedTransition] {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData(owned));
  const extra =
    kind === "backward"
      ? { originatingBackwardState: state.backwardState, nextBackwardState: state.backwardState }
      : { delta: Object.freeze({}), topology: Object.freeze({}) };
  const plan = Object.freeze({ ...prepared._plan, kind, traceDelta, ...extra });
  return [state, new PreparedTransition(prepared.record, prepared.expectedRunId, 0, plan as never, [])];
}

test("a backward plan needs a sequence-only trace change before anything else", () => {
  const [state, prepared] = handBuilt("backward", null);
  const log = new MemoryRunLog(RUN_ID);
  expectError(
    () => commitPrepared(log, state, prepared),
    RangeError,
    /^postterminal trace delta must only advance the accepted sequence$/,
  );
  expect(log.nextSequence).toBe(0);
});

test("a backward plan reaches its commit, not ported yet", () => {
  const [state, prepared] = handBuilt("backward", new TraceDelta(null, 0));
  const log = new MemoryRunLog(RUN_ID);
  expectError(() => commitPrepared(log, state, prepared), Error, /^not ported yet \(T4-5\)$/);
  expect(log.nextSequence).toBe(0);
});

test("a growth plan requires its NodeExpandedRecord at commit", () => {
  const [state, prepared] = handBuilt("growth", null);
  const log = new MemoryRunLog(RUN_ID);
  expectError(() => commitPrepared(log, state, prepared), TypeError, /^growth plan requires a NodeExpandedRecord$/);
  expect(log.nextSequence).toBe(0);
});

test.each([
  ["interrupted attempts", (state: RunState) => state.interruptedAttempts.set("work", 1)],
  [
    "last stop sequence",
    (state: RunState): void => {
      state.lastStopSequence = 1;
    },
  ],
] as const)("prepareRunStarted refuses a state with interrupted-attempt history: %s", (_name, mutate) => {
  const owned = flow();
  const state = new RunState(owned);
  mutate(state);

  expectError(
    () => prepareRunStarted(state, startData(owned)),
    RangeError,
    /^RunStarted requires a pristine RunState at sequence 0$/,
  );
});

const COUNTS = /^interrupted attempt counts must be positive integers by node$/;
const LAST_STOP = /^last stop sequence must be null or an earlier accepted sequence$/;

test.each([
  [
    "a plain-object container",
    (state: RunState): void => {
      state.interruptedAttempts = {} as never;
    },
    /^transition requires initialized reducer containers$/,
  ],
  ["a zero count", (state: RunState) => state.interruptedAttempts.set("work", 0), COUNTS],
  ["a negative count", (state: RunState) => state.interruptedAttempts.set("work", -1), COUNTS],
  ["a fractional count", (state: RunState) => state.interruptedAttempts.set("work", 1.5), COUNTS],
  ["a non-string node id", (state: RunState) => state.interruptedAttempts.set(1 as never, 1), COUNTS],
  [
    "a zero stop sequence",
    (state: RunState): void => {
      state.lastStopSequence = 0;
    },
    LAST_STOP,
  ],
  [
    "the next sequence as the stop",
    (state: RunState): void => {
      state.lastStopSequence = state.nextSequence;
    },
    LAST_STOP,
  ],
  [
    "a fractional stop sequence",
    (state: RunState): void => {
      state.lastStopSequence = 1.5;
    },
    LAST_STOP,
  ],
] as const)("an initialized state refuses broken interrupted-attempt history: %s", (_name, mutate, message) => {
  const { state } = initializedAndStarted(flow());
  mutate(state);

  expectError(() => prepareRunStopped(state, "signal", ""), RangeError, message);
});

test("an initialized state accepts counts by node and an earlier stop sequence", () => {
  const { state } = initializedAndStarted(flow());
  state.interruptedAttempts.set("work", 2);
  state.lastStopSequence = state.nextSequence - 1;

  expect(prepareRunStopped(state, "signal", "").expectedSequence).toBe(state.nextSequence);
});
