/**
 * Silent replay of a historical log prefix through the live reducer: a port of
 * `tests/engine/test_run_recovery_prefixes.py`. Python's `ValueError` is a `RangeError` here, and
 * each rejection names its one class and full message.
 *
 * Ported: none unchanged.
 *
 * Changed:
 * - `test_replay_requires_pristine_state`: the log is a live run of the fixtures' `flow()`
 *   (`load_flow` belongs to T4-8).
 * - `test_historical_cursor_matches_each_envelopes_schema`: the cursor holds live TS log lines,
 *   not Python's recorded schema-9 bytes, since schema 10 is the only TS schema. It also pins the
 *   append past the prefix and the consumed envelopes `records()` and the iterator return.
 *
 * TS-only, the reducer cores of the T4-8 tests below, through `replayRecords`: every prefix of a
 * live log replays to the live state after it; a replayed RunStarted keeps the recorded defaults
 * and call-time LLM config; an empty prefix, a first record that isn't RunStarted, a first
 * sequence that isn't 0, a sequence gap, a run-id change and a record after the terminal are
 * refused; a route to a handle that matches no edge throws the `RangeError` itself, not the live
 * `NodeDataError`.
 *
 * Not ported: none.
 *
 * Later (T4-8), since each needs `load_flow` and `run._replay_run_log` (or `resume_flow`):
 * `test_replay_every_prefix_matches_live_state_and_uses_commit_path`,
 * `test_replay_preserves_ordered_unfinished_starts_for_later_resume`,
 * `test_replay_applies_logged_llm_map_before_rebuilding_growth`,
 * `test_recorded_public_resume_keeps_completed_code_and_wire` (2 rows),
 * `test_replay_rejects_changed_but_valid_growth_digest`,
 * `test_replay_rejects_invalid_prefix_roots` (3 rows),
 * `test_replay_rejects_log_identity_and_length_disagreement`,
 * `test_fingerprint_mismatch_fails_before_copy_or_llm_apply`,
 * `test_replay_rejects_records_after_terminal_without_touching_source` and
 * `test_run_started_replay_rejects_pool_and_envelope_identity_drift`.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { BUILTIN } from "../../src/defaults.ts";
import {
  decodeRecord,
  encodeRecord,
  MemoryRunLog,
  NodeRoutedRecord,
  NodeStartedRecord,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  HistoricalLogCursor,
  prepareNodeStarted,
  prepareNodeSucceeded,
  prepareRunStarted,
  replayRecords,
  RunState,
} from "../../src/runtime/state.ts";
import {
  commitStartSuccess,
  flow,
  fragmentBuilder,
  reducerSnapshot,
  routeFlow,
  routeInFlight,
  RUN_ID,
  startData,
} from "./run_state_fixtures.ts";

/** Assert `fn` throws an error of exactly class `cls` whose message is `message`. */
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

/**
 * Return the log of a terminal live run of `flow()` and its reducer snapshot after each prefix,
 * from its RunStarted on.
 */
function linearHistory(): { log: MemoryRunLog; snapshots: Record<string, unknown>[] } {
  const state = new RunState(flow());
  const log = new MemoryRunLog(RUN_ID);
  commitPrepared(log, state, prepareRunStarted(state, startData(state.flow)));
  const snapshots = [reducerSnapshot(state)];
  const transitions = [
    () => commitPrepared(log, state, prepareNodeStarted(state, START_ID)),
    () => commitStartSuccess(state, log),
    () => commitPrepared(log, state, prepareNodeStarted(state, "work")),
    () => commitPrepared(log, state, prepareNodeSucceeded(state, "work", "Ada")),
    () => commitPrepared(log, state, prepareNodeStarted(state, END_ID)),
    () => commitPrepared(log, state, prepareNodeSucceeded(state, END_ID, "Ada")),
  ];
  for (const transition of transitions) {
    transition();
    snapshots.push(reducerSnapshot(state));
  }
  expect(state.result).not.toBeNull();
  return { log, snapshots };
}

/** Replay `envelopes` over a fresh state of `flow()`. */
function replay(envelopes: readonly RecordEnvelope[]): RunState {
  return replayRecords(new RunState(flow()), envelopes, { growBuilder: fragmentBuilder });
}

test("every prefix of a live log replays to the live state after that prefix", () => {
  const { log, snapshots } = linearHistory();
  const envelopes = log.records();
  expect(snapshots).toHaveLength(envelopes.length);

  snapshots.forEach((expected, index) => {
    const length = index + 1;
    const recovered = replay(envelopes.slice(0, length));
    expect(reducerSnapshot(recovered)).toEqual(expected);
    expect(recovered.nextSequence).toBe(length);
  });
});

test("a replayed RunStarted keeps the recorded defaults and call-time LLM config", () => {
  const live = new RunState(flow());
  const log = new MemoryRunLog(RUN_ID);
  const data = {
    ...startData(live.flow),
    defaults: { MAX_TOTAL_NODES: 5000n },
    callTimeLlm: { selection: {}, agents: { chat: {} } },
  };
  commitPrepared(log, live, prepareRunStarted(live, data));
  expect(live.runDefaults).not.toEqual(BUILTIN);
  expect(live.callTimeLlm).not.toBeNull();

  const recovered = replay(log.records());

  expect(recovered.runDefaults).toEqual(live.runDefaults);
  expect(recovered.callTimeLlm).toEqual(live.callTimeLlm);
  expect(recovered.nextSequence).toBe(live.nextSequence);
});

test("the historical cursor matches each envelope's text", () => {
  const { log } = linearHistory();
  const wires = log.records().map(encodeRecord);
  const cursor = new HistoricalLogCursor(wires);
  expect(cursor.runId).toBe(RUN_ID);
  expect(cursor.path).toBeNull();

  for (const wire of wires) {
    const expected = decodeRecord(wire);
    const before = cursor.nextSequence;
    expectError(
      () => cursor.append(new NodeStartedRecord("wrong")),
      RangeError,
      "prepared transition does not match the historical run record",
    );
    expect(cursor.nextSequence).toBe(before);
    const accepted = cursor.append(expected.record);
    expect(accepted.schema).toBe(SCHEMA_VERSION);
    expect(encodeRecord(accepted)).toBe(wire);
  }
  expect(cursor.nextSequence).toBe(wires.length);
  expectError(
    () => cursor.append(new NodeStartedRecord("wrong")),
    RangeError,
    "replay attempted to append beyond the historical prefix",
  );
  expect(cursor.records().map(encodeRecord)).toEqual(wires);
  expect([...cursor].map(encodeRecord)).toEqual(wires);
});

test("replay refuses a malformed prefix root", () => {
  const { log } = linearHistory();
  const envelopes = log.records();
  const started = new NodeStartedRecord(START_ID);

  expectError(() => replay([]), RangeError, "cannot replay an empty run-record prefix");
  expectError(
    () => replay([new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 0, started)]),
    RangeError,
    "the first historical run record must be RunStarted",
  );
  expectError(
    () => replay([new RecordEnvelope(SCHEMA_VERSION, RUN_ID, 1, started)]),
    RangeError,
    "replay sequence mismatch: expected 0, found 1",
  );
  expectError(
    () => replay([...envelopes.slice(0, 2), ...envelopes.slice(3)]),
    RangeError,
    "replay sequence mismatch: expected 2, found 3",
  );
  const last = envelopes.at(-1)!;
  expectError(
    () => replay([...envelopes.slice(0, -1), new RecordEnvelope(last.schema, "other-run", last.seq, last.record)]),
    RangeError,
    "replay run_id mismatch: expected 'run-123', found 'other-run'",
  );
});

test("replay requires a pristine state", () => {
  const { log } = linearHistory();
  const state = new RunState(flow());
  state.ready.push(START_ID);

  expectError(
    () => replayRecords(state, log.records(), { growBuilder: fragmentBuilder }),
    RangeError,
    "RunStarted requires a pristine RunState at sequence 0",
  );
});

test("replay refuses a record after the terminal", () => {
  const { log } = linearHistory();
  const trailing = MemoryRunLog.fromRecords(log.records());
  trailing.append(new NodeStartedRecord(START_ID));
  const envelopes = trailing.records();

  expectError(() => replay(envelopes), RangeError, "terminal run state cannot prepare another graph transition");
});

test("a replayed route to a handle that matches no edge throws the RangeError itself", () => {
  const { log } = routeInFlight();
  const envelopes = [
    ...log.records(),
    new RecordEnvelope(SCHEMA_VERSION, RUN_ID, log.nextSequence, new NodeRoutedRecord("router", "missing")),
  ];

  expectError(
    () => replayRecords(new RunState(routeFlow()), envelopes, { growBuilder: fragmentBuilder }),
    RangeError,
    "NodeRouted handle 'missing' matches no outgoing edge of 'router'",
  );
});
