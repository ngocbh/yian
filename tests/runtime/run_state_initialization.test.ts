/**
 * Transactional run-state initialization: `prepareRunStarted`, `commitPrepared` and the run
 * start's pool commits, with append failures and log-return mismatches. Each rejected row pins its
 * class and message.
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
  encodeRecord,
  MemoryRunLog,
  NodeStartedRecord,
  PoolCommitRecord,
  RecordEnvelope,
  type RunRecord,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitInitialPool,
  commitPrepared,
  PreparedTransition,
  prepareExpanded,
  prepareInitialCommit,
  prepareNodeStarted,
  prepareRecord,
  prepareRunStarted,
  prepareRunStopped,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import { TraceDelta } from "../../src/runtime/trace.ts";
import {
  FailingLog,
  FINGERPRINT,
  FIXED_AT,
  FIXED_CLOCK,
  flow,
  initializedAndStarted,
  pool,
  ReturningLog,
  RUN_ID,
  startData,
  stateSnapshot,
} from "../helpers/run_state_fixtures.ts";

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
  const prepared = prepareRunStarted(state, startData());
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
    () => prepareRunStarted(state, startData({ numWorkers: numWorkers as number })),
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
    () => prepareRunStarted(state, startData()),
    RangeError,
    /^RunStarted requires a pristine RunState at sequence 0$/,
  );

  const missingStart = flow();
  missingStart.nodes.delete(missingStart.startId);
  const missingState = new RunState(missingStart);
  expectError(
    () => prepareRunStarted(missingState, startData()),
    RangeError,
    /^flow START node '__start__' is missing$/,
  );
});

/** Return a state of the fixture flow and its log, after `RunStarted` and before its commits. */
function runStartedOnly(): { state: RunState; log: MemoryRunLog } {
  const state = new RunState(flow());
  const log = new MemoryRunLog(RUN_ID, { clock: FIXED_CLOCK });
  commitPrepared(log, state, prepareRunStarted(state, startData()));
  return { state, log };
}

test.each([
  [
    "a pool of another START",
    (state: RunState, log: MemoryRunLog) => {
      const wrong = pool(state.flow);
      (wrong as { startId: string }).startId = "other";
      commitInitialPool(log, state, wrong);
    },
    "the initial pool must store only the START inputs, under '__start__'",
  ],
  [
    "a pool storing another value",
    (state: RunState, log: MemoryRunLog) => {
      const wrong = pool(state.flow);
      wrong.set("preseeded", { retained: true });
      commitInitialPool(log, state, wrong);
    },
    "the initial pool must store only the START inputs, under '__start__'",
  ],
  [
    "a system.run_id of another run",
    (state: RunState, log: MemoryRunLog) => commitInitialPool(log, state, pool(state.flow, "different")),
    "system.run_id must equal the run id",
  ],
  [
    "START inputs that aren't a record",
    (state: RunState, log: MemoryRunLog) => {
      const scalar = pool(state.flow);
      scalar.set(state.flow.startId, "not a record");
      commitInitialPool(log, state, scalar);
    },
    "the START inputs' pool_commit must hold a record",
  ],
  [
    "START inputs before system.run_id",
    (state: RunState) => prepareInitialCommit(state, START_ID, {}),
    "the run start commits system.run_id, equal to the run id, before the START inputs",
  ],
  [
    "a key of neither form",
    (state: RunState) => prepareInitialCommit(state, "work", {}),
    "a run-start pool_commit key must be system.<name> or the START '__start__', got 'work'",
  ],
  [
    "an empty system name",
    (state: RunState) => prepareInitialCommit(state, "system.", 1n),
    "a run-start pool_commit key must be system.<name> or the START '__start__', got 'system.'",
  ],
  [
    "another node",
    (state: RunState) => prepareRecord(state, new PoolCommitRecord("work", "system.x", { kind: "int", value: 1n })),
    "a run-start pool_commit must name the START '__start__', not 'work'",
  ],
  [
    "a system value committed twice",
    (state: RunState, log: MemoryRunLog) => {
      commitPrepared(log, state, prepareInitialCommit(state, "system.tenant", "a"));
      prepareInitialCommit(state, "system.tenant", "b");
    },
    "the run start commits 'system.tenant' twice",
  ],
  [
    "a node start before the START inputs",
    (state: RunState) => prepareNodeStarted(state, START_ID),
    "the run start is incomplete: only its pool_commit records may follow RunStarted",
  ],
] as const)("the run start's commits check the run's identity and the START inputs: %s", (_name, act, message) => {
  const { state, log } = runStartedOnly();
  expectError(() => act(state, log), RangeError, new RegExp(`^${message.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`));
});

test("the run start's last commit, the START inputs, makes START ready", () => {
  const { state, log } = runStartedOnly();
  expect(state.ready).toEqual([]);
  expect(state.inputsCommitted).toBe(false);

  commitInitialPool(log, state, pool(state.flow));

  expect(log.records().slice(1).map(({ record }) => (record as PoolCommitRecord).key)).toEqual([
    "system.now",
    "system.run_id",
    "system.tenant",
    "system.today",
    START_ID,
  ]);
  expect(state.inputsCommitted).toBe(true);
  expect(state.ready).toEqual([START_ID]);
  expectError(
    () => prepareInitialCommit(state, "system.late", 1n),
    RangeError,
    /^pool_commit node '__start__' is not in flight$/,
  );
});

test.each([
  [
    "a fingerprint that isn't a digest",
    (owned: CompiledFlow) => ({ ...startData(), flowFingerprint: "not-a-digest" }),
    RangeError,
    /^flow_fingerprint must be 64 lowercase hex digits$/,
  ],
  [
    "an empty run id",
    (owned: CompiledFlow) => startData({ runId: "" }),
    RangeError,
    /^run log run_id must be a non-empty string$/,
  ],
  [
    "an agent configuration that isn't an object",
    (owned: CompiledFlow) => ({ ...startData(), effectiveLlmByStaticPath: { agent: "bad" } }),
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
  const prepared = prepareRunStarted(state, startData());
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
  // Strict-mode code can't write into a string's characters.
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
    const data = startData();
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
  const data = startData();
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
  ((data.startWiringOverride["metadata"] as Record<string, unknown>)["labels"] as string[]).push("mutated");
  (data.effectiveLlmByStaticPath["agent"]!["options"] as Record<string, unknown>)["temperature"] = 1.0;

  const log = new MemoryRunLog(RUN_ID, { clock: FIXED_CLOCK });
  const observers = commitPrepared(log, state, prepared);

  expect(observers).toEqual([new RunStarted()]);
  expect(log.records()).toEqual([new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 0, prepared.record, FIXED_AT)]);
  expect(state.runId).toBe(RUN_ID);
  expect(state.flowFingerprint).toBe(FINGERPRINT);
  expect(state.nextSequence).toBe(1);
  expect(state.numWorkers).toBe(3);
  expect(state.ready).toEqual([]);

  // The run start's commits own canonical copies too.
  const given = pool(owned);
  commitInitialPool(log, state, given);
  given.set(owned.startId, { name: "mutated" });
  given.addSystem("today", "never");
  expect(state.pool).not.toBe(given);
  expect(state.pool.startId).toBe(owned.startId);
  expect(state.pool.get(owned.startId)).toEqual({ name: "Ada", count: 2n, aliases: ["A", "B"] });
  expect(state.pool.readRoot(["system", "run_id"]).value).toBe(RUN_ID);
  expect(state.pool.readRoot(["system", "today"]).value).toBe("2026-08-13");
  expect(state.pool.readRoot(["system", "now"]).value).toBe("2026-08-13T12:34:56+00:00");
  expect(state.pool.readRoot(["system", "tenant"]).value).toBe("tests");
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
  const prepared = prepareRunStarted(state, startData());
  const log = new InspectingLog(state);

  const observers = commitPrepared(log, state, prepared);

  expect(log.sawAppend).toBe(true);
  expect(state.runId).toBe(RUN_ID);
  expect(observers).toEqual([new RunStarted()]);
});

test("commitPrepared append failure leaves state and observers untouched", () => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData());
  const before = stateSnapshot(state);

  expectError(() => commitPrepared(new FailingLog(RUN_ID), state, prepared), Error, /^disk full$/);

  expect(stateSnapshot(state)).toEqual(before);
  expect(prepared.observers).toEqual([new RunStarted()]);
});

test("commitPrepared rejects a changed public prepared record", () => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData());
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
  const prepared = prepareRunStarted(originatingState, startData());
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
  const prepared = prepareRunStarted(state, startData());
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
  const prepared = prepareRunStarted(state, startData());
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
  const prepared = prepareRunStarted(state, startData());
  const record = prepared.record as RunStartedRecord;
  const oldReady = state.ready;
  const oldInFlight = state.inFlight;
  const oldNodeState = state.nodeState;
  const oldEdgeState = state.edgeState;
  const oldDepth = state.depth;
  const oldWiring = state.wiringOverrides;
  const oldEffective = state.effectiveLlmByStaticPath;

  commitPrepared(new MemoryRunLog(RUN_ID), state, prepared);
  const committed = stateSnapshot(state);

  oldReady.push("old-ready");
  oldInFlight.add("old-flight");
  oldNodeState.set("work", NodeState.TAKEN);
  oldEdgeState.set("start-work", NodeState.TAKEN);
  oldDepth.set("work", 9);
  oldWiring.set("work", { x: 1n });
  oldEffective["old"] = { model: "wrong" };
  // The prepared record is deep-frozen, so its payload can't be changed at all.
  expect(() => {
    ((record.startWiringOverride["metadata"] as Record<string, unknown>)["labels"] as string[]).push("wrong");
  }).toThrow(TypeError);
  expect(() => {
    (record.effectiveLlmByStaticPath["agent"] as Record<string, unknown>)["model"] = "wrong";
  }).toThrow(TypeError);

  expect(stateSnapshot(state)).toEqual(committed);
  expect(state.ready).not.toBe(oldReady);
  expect(state.inFlight).not.toBe(oldInFlight);
  expect(state.nodeState).not.toBe(oldNodeState);
  expect(state.edgeState).not.toBe(oldEdgeState);
  expect(state.depth).not.toBe(oldDepth);
  expect(state.wiringOverrides).not.toBe(oldWiring);
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
  const prepared = prepareRunStarted(state, startData());
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
  const prepared = prepareRunStarted(state, startData());
  const before = stateSnapshot(state);

  expectError(() => commitPrepared(new ReturningLog(result), state, prepared), RangeError, APPEND_MISMATCH);

  expect(stateSnapshot(state)).toEqual(before);
});

/** A memory log that retains the record but returns it without the time it stamped. */
class UntimedReturnLog extends MemoryRunLog {
  override append(record: RunRecord): RecordEnvelope {
    const retained = super.append(record);
    return new RecordEnvelope(retained.schema, retained.runId, retained.seq, retained.record);
  }
}

test("commitPrepared rejects an append result without a time, since a log stamps every record", () => {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData());
  const before = stateSnapshot(state);

  expectError(() => commitPrepared(new UntimedReturnLog(RUN_ID), state, prepared), RangeError, APPEND_MISMATCH);

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
  const prepared = prepareRunStarted(state, startData());
  const before = stateSnapshot(state);
  const log = new RetainingMismatchedLog(RUN_ID, { clock: FIXED_CLOCK });

  expectError(() => commitPrepared(log, state, prepared), RangeError, APPEND_MISMATCH);

  expect(stateSnapshot(state)).toEqual(before);
  expect(log.nextSequence).toBe(1);
  const retained = log.records();
  expect(retained).toEqual([new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 0, prepared.record, FIXED_AT)]);
  expect(retained[0]!.record).toBeInstanceOf(RunStartedRecord);
});

test("growth refuses an uninitialized state and a non-spawner", () => {
  const owned = flow();
  const state = new RunState(owned);
  const builder = () => owned;
  expectError(
    () => prepareExpanded(state, "work", new Grow({}), builder),
    RangeError,
    /^run state is not initialized$/,
  );

  const { state: started } = initializedAndStarted(flow());
  expectError(
    () => prepareExpanded(started, START_ID, new Grow({}), builder),
    RangeError,
    /^NodeExpanded node '__start__' is not a graph spawner$/,
  );
});

test("replay checks its arguments' types before the state is pristine", () => {
  const owned = flow();
  const state = new RunState(owned);
  state.ready.push(START_ID);
  const builder = () => owned;
  const { log } = initializedAndStarted(flow());
  const envelopes = log.records();

  expectError(
    () => replayRecords({} as RunState, envelopes, { growBuilder: builder }),
    TypeError,
    /^replayRecords expects a RunState$/,
  );
  for (const bad of [new Set(envelopes), [...envelopes, envelopes[0]!.record]]) {
    expectError(
      () => replayRecords(state, bad as never, { growBuilder: builder }),
      TypeError,
      /^replay envelopes must be an array of RecordEnvelope values$/,
    );
  }
  expectError(
    () => replayRecords(state, envelopes, { growBuilder: null as never }),
    TypeError,
    /^grow_builder must be callable$/,
  );
  expectError(
    () => replayRecords(state, [], { growBuilder: builder }),
    RangeError,
    /^RunStarted requires a pristine RunState at sequence 0$/,
  );
});

/** Return a prepared RunStarted whose plan is replaced by `kind`'s, with `traceDelta`. */
function handBuilt(kind: "backward" | "growth", traceDelta: TraceDelta | null): [RunState, PreparedTransition] {
  const owned = flow();
  const state = new RunState(owned);
  const prepared = prepareRunStarted(state, startData());
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

test("a backward plan reaches its commit, which requires an initialized state", () => {
  const [state, prepared] = handBuilt("backward", new TraceDelta(null, 0));
  const log = new MemoryRunLog(RUN_ID);
  expectError(() => commitPrepared(log, state, prepared), RangeError, /^run state is not initialized$/);
  expect(log.nextSequence).toBe(0);
});

test("a growth plan requires its NodeExpandedRecord at commit", () => {
  const [state, prepared] = handBuilt("growth", null);
  const log = new MemoryRunLog(RUN_ID);
  expectError(() => commitPrepared(log, state, prepared), TypeError, /^growth plan requires a NodeExpandedRecord$/);
  expect(log.nextSequence).toBe(0);
});

test.each([
  ["crashed attempts", (state: RunState) => state.crashedAttempts.set("work", 1)],
  [
    "last stop sequence",
    (state: RunState): void => {
      state.lastStopSequence = 1;
    },
  ],
] as const)("prepareRunStarted refuses a state with crashed-attempt history: %s", (_name, mutate) => {
  const owned = flow();
  const state = new RunState(owned);
  mutate(state);

  expectError(
    () => prepareRunStarted(state, startData()),
    RangeError,
    /^RunStarted requires a pristine RunState at sequence 0$/,
  );
});

const COUNTS = /^crashed attempt counts must be positive integers by node$/;
const LAST_STOP = /^last stop sequence must be null or an earlier accepted sequence$/;

test.each([
  [
    "a plain-object container",
    (state: RunState): void => {
      state.crashedAttempts = {} as never;
    },
    /^transition requires initialized reducer containers$/,
  ],
  ["a zero count", (state: RunState) => state.crashedAttempts.set("work", 0), COUNTS],
  ["a negative count", (state: RunState) => state.crashedAttempts.set("work", -1), COUNTS],
  ["a fractional count", (state: RunState) => state.crashedAttempts.set("work", 1.5), COUNTS],
  ["a non-string node id", (state: RunState) => state.crashedAttempts.set(1 as never, 1), COUNTS],
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
] as const)("an initialized state refuses broken crashed-attempt history: %s", (_name, mutate, message) => {
  const { state } = initializedAndStarted(flow());
  mutate(state);

  expectError(() => prepareRunStopped(state, "signal", ""), RangeError, message);
});

test("an initialized state accepts counts by node and an earlier stop sequence", () => {
  const { state } = initializedAndStarted(flow());
  state.crashedAttempts.set("work", 2);
  state.lastStopSequence = state.nextSequence - 1;

  expect(prepareRunStopped(state, "signal", "").expectedSequence).toBe(state.nextSequence);
});
