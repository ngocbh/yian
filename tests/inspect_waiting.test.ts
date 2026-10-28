/**
 * `inspectRun` without a flow on logs whose nodes wait: `waiting`, `ready`, `wakeAt` and the
 * waits; crashes while nodes wait; `cancelled` and `run_failed` endings.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { END_ID, START_ID } from "../src/compile/model.ts";
import { Failure } from "../src/diagnostics.ts";
import { inspectRun } from "../src/run.ts";
import {
  JsonlRunLog,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  RunCancelledRecord,
  RunFailedRecord,
  type RunRecord,
  RunStartedRecord,
  RunWaitingRecord,
} from "../src/runtime/log.ts";
import { FIXED_AT, FIXED_CLOCK } from "./helpers/run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-inspect-waiting-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const DEADLINE = "2026-10-24T09:00:00.000Z";
const unreachable = new Failure({ code: "graph.terminal_unreachable", message: "END is unreachable", nodeId: "gate" });

const takeWait = (node: string, deadline: string | null = null) =>
  new NodeWaitingRecord(node, [`r/${node}`], { waits: [{ callId: `r/${node}`, tool: "take", inbox: "msgs", mode: "one", deadline }] });
const sleepWait = (node: string) =>
  new NodeWaitingRecord(node, [`r/${node}`], { waits: [{ callId: `r/${node}`, tool: "sleep", inbox: null, mode: null, deadline: DEADLINE }] });
const send = (value: string) => new PoolCommitRecord(START_ID, `${START_ID}.msgs`, { kind: "str", value });
const endSuccess = (commit: number) => [
  new NodeStartedRecord(END_ID),
  new PoolCommitRecord(END_ID, END_ID, { kind: "str", value: "x" }),
  new NodeSucceededRecord(END_ID, commit),
];

let logs = 0;
/** Write a log of the run start (seqs 0 and 1) then `records`; return its path. */
function logOf(...records: RunRecord[]): string {
  const path = join(tmp, `run-${++logs}.jsonl`);
  const log = JsonlRunLog.create(path, "r", { clock: FIXED_CLOCK });
  log.append(
    new RunStartedRecord({ flowFingerprint: "a".repeat(64), startWiringOverride: {}, effectiveLlmByStaticPath: {}, numWorkers: 0 }),
  );
  log.append(new PoolCommitRecord(START_ID, START_ID, { kind: "object", value: {} }));
  for (const record of records) log.append(record);
  log.close();
  return path;
}

/** The parts of an inspection a wait changes. */
function progress(path: string) {
  const { status, inFlight, crashedAttempts, ready, wakeAt, waits, endedAt } = inspectRun(path);
  return { status, inFlight, crashedAttempts: Object.fromEntries(crashedAttempts), ready, wakeAt, waits, endedAt };
}

const WAITING = [new NodeStartedRecord("heard"), takeWait("heard"), new RunWaitingRecord(["heard"])];

test("a run whose node waits on a take is waiting, not ready", () => {
  expect(progress(logOf(...WAITING))).toEqual({
    status: "waiting",
    inFlight: [],
    crashedAttempts: {},
    ready: false,
    wakeAt: null,
    waits: [{ node: "heard", callId: "r/heard", tool: "take", inbox: "msgs", deadline: null, settled: false }],
    endedAt: null,
  });
});

test("a send to the waiting take makes the run ready", () => {
  const inspection = progress(logOf(...WAITING, send("hi")));
  expect([inspection.status, inspection.ready, inspection.waits[0]!.settled]).toEqual(["waiting", true, true]);
});

test("a sleeping node sets wakeAt to its deadline", () => {
  const inspection = progress(logOf(new NodeStartedRecord("nap"), sleepWait("nap"), new RunWaitingRecord(["nap"])));
  expect([inspection.status, inspection.ready, inspection.wakeAt]).toEqual(["waiting", false, DEADLINE]);
});

test("a wake then the log ends: crashed, the woken node in flight with one crashed attempt", () => {
  expect(progress(logOf(...WAITING, send("hi"), new NodeWokenRecord("heard")))).toEqual({
    status: "crashed",
    inFlight: ["heard"],
    crashedAttempts: { heard: 1 },
    ready: false,
    wakeAt: null,
    waits: [],
    endedAt: null,
  });
});

test("a crash with one node running and one waiting: only the running one is in flight", () => {
  const inspection = progress(logOf(new NodeStartedRecord("heard"), new NodeStartedRecord("work"), takeWait("heard")));
  expect([inspection.status, inspection.inFlight, inspection.crashedAttempts]).toEqual(["crashed", ["work"], { work: 1 }]);
});

test("a cancel ends the run: cancelled, nothing in flight, no crashed attempt added, an end time", () => {
  expect(progress(logOf(new NodeStartedRecord("work"), new NodeStartedRecord("heard"), takeWait("heard"), new RunCancelledRecord()))).toEqual({
    status: "cancelled",
    inFlight: [],
    crashedAttempts: {},
    ready: false,
    wakeAt: null,
    waits: [],
    endedAt: FIXED_AT,
  });
});

test("run_failed ends the run failed with its failure", () => {
  const path = logOf(
    new NodeStartedRecord("heard"),
    takeWait("heard"),
    new NodeStartedRecord("gate"),
    new NodeRoutedRecord("gate", "no", { terminalFailure: unreachable }),
    new RunFailedRecord(unreachable),
  );
  const inspection = inspectRun(path);
  expect([inspection.status, inspection.failure, inspection.endedAt, inspection.inFlight]).toEqual(["failed", unreachable, FIXED_AT, []]);
});

test("an unreachable END with a node waiting isn't failed until run_failed", () => {
  const path = logOf(
    new NodeStartedRecord("heard"),
    takeWait("heard"),
    new NodeStartedRecord("gate"),
    new NodeRoutedRecord("gate", "no", { terminalFailure: unreachable }),
    new RunWaitingRecord(["heard"]),
  );
  expect([inspectRun(path).status, inspectRun(path).failure]).toEqual(["waiting", null]);
});

test("END succeeded while a node waits: waiting after run_waiting", () => {
  const inspection = progress(logOf(new NodeStartedRecord("heard"), takeWait("heard"), ...endSuccess(5), new RunWaitingRecord(["heard"])));
  expect([inspection.status, inspection.endedAt]).toEqual(["waiting", null]);
});

test("END succeeded after a route proved it unreachable, nothing in flight: succeeded", () => {
  const path = logOf(new NodeStartedRecord("gate"), new NodeRoutedRecord("gate", "no", { terminalFailure: unreachable }), ...endSuccess(5));
  expect([inspectRun(path).status, inspectRun(path).failure]).toEqual(["succeeded", null]);
});
