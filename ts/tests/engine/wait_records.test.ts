/**
 * The inbox and wait record kinds (`node_waiting`, `node_woken`, `run_waiting`, `inbox_closed`,
 * `run_cancelled`, `run_failed`), the take form of `pool_commit`, and the record-time exports:
 * their constructors' checks and their canonical wire; the waits fold's empty state on a fresh
 * `RunState`, the reducer refusing the new kinds until it applies them, and `eventBytes`, an
 * event's size. TS-only: Python yian has no inbox or suspension records.
 */
import { expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Failure } from "../../src/diagnostics.ts";
import {
  decodeRecord,
  encodeRecord,
  formatRecordTime,
  InboxClosedRecord,
  iterRunRecords,
  JsonlRunLog,
  NodeStartedRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  RecordEnvelope,
  requireRecordTime,
  RunCancelledRecord,
  RunFailedRecord,
  type RunRecord,
  RunResumedRecord,
  RunWaitingRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { prepareRecord, prepareRunStarted, RunState } from "../../src/runtime/state.ts";
import { WaitsState } from "../../src/runtime/waits.ts";
import { eventBytes, logJson } from "../../src/typesys/json.ts";
import { commitStartSuccess, flow, initializedAndStarted, startData } from "./run_state_fixtures.ts";
import { StartNode } from "../../src/nodes/start/node.ts";

const DEADLINE = "2026-10-24T09:00:00.000Z";
const take = { callId: "r/heard", tool: "take", inbox: "msgs", mode: "one", deadline: null } as const;
const sleep = { callId: "r/nap", tool: "sleep", inbox: null, mode: null, deadline: DEADLINE } as const;
const FAILURE = new Failure({ code: "graph.terminal_unreachable", message: "END is unreachable", nodeId: "choice" });

/** Assert `build` throws exactly `cls` with exactly `message`. */
function expectError(build: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    build();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** The wire line of `kind` with `payload` at seq 3. */
function line(kind: string, payload: Record<string, unknown>): string {
  return logJson({ schema: BigInt(SCHEMA_VERSION), run_id: "r", seq: 3n, kind, payload });
}

test("node_waiting holds unique call ids and at most one wait entry; frozen", () => {
  const record = new NodeWaitingRecord("heard", ["r/heard"], { waits: [take] });
  expect(record.waits).toEqual([take]);
  expect(record.steerInbox).toBeNull();
  expect(Object.isFrozen(record) && Object.isFrozen(record.waits) && Object.isFrozen(record.waits[0])).toBe(true);
  expect(Object.isFrozen(record.callIds)).toBe(true);
  expect(new NodeWaitingRecord("ask", ["r/ask"]).waits).toEqual([]);
  expect(new NodeWaitingRecord("nap", ["r/nap"], { waits: [sleep] }).waits).toEqual([sleep]);
  expect(new NodeWaitingRecord("n", ["r/n"], { steerInbox: "steer" }).steerInbox).toBe("steer");
});

test.each<[string, () => unknown, string | RegExp]>([
  ["no call ids", () => new NodeWaitingRecord("n", []), "call_ids must be non-empty"],
  ["duplicate ids", () => new NodeWaitingRecord("n", ["a", "a"]), /call_ids/],
  [
    "two entries",
    () => new NodeWaitingRecord("n", ["r/heard", "r/nap"], { waits: [take, sleep] }),
    "waits holds at most one entry",
  ],
  ["entry not named", () => new NodeWaitingRecord("n", ["x"], { waits: [take] }), "wait entry call_id 'r/heard' is not in call_ids"],
  [
    "bad tool",
    () => new NodeWaitingRecord("n", ["r/heard"], { waits: [{ ...take, tool: "ask" as never }] }),
    "wait entry tool must be take or sleep",
  ],
  [
    "take without mode",
    () => new NodeWaitingRecord("n", ["r/heard"], { waits: [{ ...take, mode: null }] }),
    "a take wait entry needs inbox and mode",
  ],
  [
    "bad mode",
    () => new NodeWaitingRecord("n", ["r/heard"], { waits: [{ ...take, mode: "some" as never }] }),
    "wait entry mode must be one or all",
  ],
  [
    "sleep with inbox",
    () => new NodeWaitingRecord("n", ["r/nap"], { waits: [{ ...sleep, inbox: "msgs" }] }),
    "a sleep wait entry takes no inbox or mode",
  ],
  [
    "sleep without deadline",
    () => new NodeWaitingRecord("n", ["r/nap"], { waits: [{ ...sleep, deadline: null }] }),
    "a sleep wait entry needs a deadline",
  ],
  [
    "bad deadline",
    () => new NodeWaitingRecord("n", ["r/nap"], { waits: [{ ...sleep, deadline: "tomorrow" }] }),
    /wait entry deadline must be a UTC time/,
  ],
  ["an empty node id", () => new NodeWaitingRecord("", ["r/n"]), "node_id must be non-empty"],
  ["an empty steer inbox", () => new NodeWaitingRecord("n", ["r/n"], { steerInbox: "" }), "steer_inbox must be non-empty"],
])("node_waiting refuses %s", (_name, build, message) => expect(build).toThrow(message));

test("node_woken, run_waiting, inbox_closed, run_cancelled and run_failed", () => {
  expect(new NodeWokenRecord("heard").timedOut).toEqual([]);
  expect(Object.isFrozen(new NodeWokenRecord("nap", ["r/nap"]).timedOut)).toBe(true);
  expect(() => new NodeWokenRecord("nap", ["r/nap", "r/nap"])).toThrow(/timed_out/);
  expect(new RunWaitingRecord(["heard", "nap"]).nodeIds).toEqual(["heard", "nap"]);
  expectError(() => new RunWaitingRecord([]), RangeError, "node_ids must be non-empty");
  expect(new InboxClosedRecord("msgs").inbox).toBe("msgs");
  expectError(() => new InboxClosedRecord(""), RangeError, "inbox must be non-empty");
  const cancelled = new RunCancelledRecord();
  expect(Object.keys(cancelled)).toEqual([]);
  expect(Object.isFrozen(cancelled)).toBe(true);
  expect(new RunFailedRecord(FAILURE).failure).toEqual(FAILURE);
  expectError(() => new RunFailedRecord({ ...FAILURE } as never), TypeError, "RunFailedRecord failure must be a Failure");
  class OtherFailure extends Failure {}
  expectError(
    () => new RunFailedRecord(new OtherFailure({ code: "node.execute", message: "x", nodeId: "a" })),
    TypeError,
    "RunFailedRecord failure must be a Failure",
  );
});

test.each([
  [
    "node_waiting",
    new NodeWaitingRecord("heard", ["r/heard"], { waits: [take] }),
    {
      node_id: "heard",
      call_ids: ["r/heard"],
      waits: [{ call_id: "r/heard", tool: "take", inbox: "msgs", mode: "one", deadline: null }],
      steer_inbox: null,
    },
  ],
  [
    "node_waiting",
    new NodeWaitingRecord("nap", ["r/nap"], { waits: [sleep] }),
    {
      node_id: "nap",
      call_ids: ["r/nap"],
      waits: [{ call_id: "r/nap", tool: "sleep", inbox: null, mode: null, deadline: DEADLINE }],
      steer_inbox: null,
    },
  ],
  ["node_waiting", new NodeWaitingRecord("ask", ["r/ask"]), { node_id: "ask", call_ids: ["r/ask"], waits: [], steer_inbox: null }],
  ["node_woken", new NodeWokenRecord("nap", ["r/nap"]), { node_id: "nap", timed_out: ["r/nap"] }],
  ["node_woken", new NodeWokenRecord("heard"), { node_id: "heard", timed_out: [] }],
  ["run_waiting", new RunWaitingRecord(["heard"]), { node_ids: ["heard"] }],
  ["inbox_closed", new InboxClosedRecord("msgs"), { inbox: "msgs" }],
  ["run_cancelled", new RunCancelledRecord(), {}],
  ["run_failed", new RunFailedRecord(FAILURE), { failure: FAILURE.toJSON() }],
])("%s round trips under its wire kind", (kind, record, payload) => {
  const encoded = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "r", 7, record));
  const decoded = decodeRecord(encoded);
  expect(decoded.record).toEqual(record);
  expect(decoded.record.constructor).toBe(record.constructor);
  expect(encodeRecord(decoded)).toBe(encoded);
  expect(decoded.kind).toBe(kind);
  expect(decoded.payload).toEqual(payload);
});

test.each<[string, string, Record<string, unknown>, typeof TypeError | typeof RangeError, string]>([
  [
    "a missing field",
    "node_waiting",
    { node_id: "n", call_ids: ["r/n"], waits: [] },
    RangeError,
    'invalid node_waiting payload fields; missing=["steer_inbox"], extra=[]',
  ],
  [
    "an extra field",
    "node_woken",
    { node_id: "n", timed_out: [], seed: null },
    RangeError,
    'invalid node_woken payload fields; missing=[], extra=["seed"]',
  ],
  [
    "a wait entry missing a key",
    "node_waiting",
    { node_id: "n", call_ids: ["r/n"], waits: [{ call_id: "r/n", tool: "sleep", inbox: null, mode: null }], steer_inbox: null },
    RangeError,
    'invalid node_waiting payload waits item fields; missing=["deadline"], extra=[]',
  ],
  [
    "a wait entry that isn't an object",
    "node_waiting",
    { node_id: "n", call_ids: ["r/n"], waits: ["r/n"], steer_inbox: null },
    TypeError,
    "node_waiting payload waits item must be an object",
  ],
  ["text node ids", "run_waiting", { node_ids: "heard" }, TypeError, "run_waiting payload node_ids must be an array of strings"],
  ["a field in run_cancelled", "run_cancelled", { reason: "host" }, RangeError, 'invalid run_cancelled payload fields; missing=[], extra=["reason"]'],
  ["no failure", "run_failed", {}, RangeError, 'invalid run_failed payload fields; missing=["failure"], extra=[]'],
])("decode refuses %s", (_name, kind, payload, cls, message) => {
  expectError(() => decodeRecord(line(kind, payload)), cls, message);
});

test("the log reader names the node of node_waiting and node_woken lines", () => {
  const dir = mkdtempSync(join(tmpdir(), "wait-records-"));
  try {
    const path = join(dir, "run.jsonl");
    const log = JsonlRunLog.create(path, "r");
    log.append(new NodeStartedRecord("nap"));
    log.append(new NodeWaitingRecord("nap", ["r/nap"], { waits: [sleep] }));
    log.append(new RunWaitingRecord(["nap"]));
    log.append(new NodeWokenRecord("nap", ["r/nap"]));
    log.close();
    const entries = [...iterRunRecords(path, { payloads: false })];
    expect(entries.map((entry) => [entry.kind, entry.nodeId])).toEqual([
      ["node_started", "nap"],
      ["node_waiting", "nap"],
      ["run_waiting", null],
      ["node_woken", "nap"],
    ]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the take commit form holds the inbox, the consumed send seqs and closed", () => {
  const record = new PoolCommitRecord("heard", "heard", null, { take: { inbox: "msgs", events: [9, 12], closed: false } });
  expect(record.take).toEqual({ inbox: "msgs", events: [9, 12], closed: false });
  expect(Object.isFrozen(record.take) && Object.isFrozen(record.take!.events)).toBe(true);
  expect(record.value).toBeNull();
  expect(record.from).toBeNull();
  const envelope = new RecordEnvelope(SCHEMA_VERSION, "r", 13, record);
  const encoded = encodeRecord(envelope);
  expect(envelope.payload).toEqual({ node_id: "heard", key: "heard", inbox: "msgs", events: [9n, 12n], closed: false });
  expect(encoded).toContain('"payload":{"closed":false,"events":[9,12],"inbox":"msgs","key":"heard","node_id":"heard"}');
  const decoded = decodeRecord(encoded);
  expect(decoded).toEqual(envelope);
  expect(encodeRecord(decoded)).toBe(encoded);
  expect(new PoolCommitRecord("heard", "heard", null, { take: { inbox: "msgs", events: [], closed: true } }).take!.events).toEqual([]);
  expect(new PoolCommitRecord("e", "e", { kind: "int", value: 2n }).take).toBeNull();
  expect(new PoolCommitRecord("e", "e", null, { from: new Map() }).take).toBeNull();
});

test.each<[string, () => unknown, typeof TypeError | typeof RangeError, string]>([
  [
    "a value and a take",
    () => new PoolCommitRecord("e", "e", { kind: "int", value: 2n }, { take: { inbox: "m", events: [], closed: false } }),
    RangeError,
    "a pool_commit holds exactly one of value, from and take",
  ],
  [
    "from and a take",
    () => new PoolCommitRecord("e", "e", null, { from: new Map(), take: { inbox: "m", events: [], closed: false } }),
    RangeError,
    "a pool_commit holds exactly one of value, from and take",
  ],
  ["none", () => new PoolCommitRecord("e", "e", null), RangeError, "a pool_commit holds exactly one of value, from and take"],
  [
    "events out of order",
    () => new PoolCommitRecord("e", "e", null, { take: { inbox: "m", events: [12, 9], closed: false } }),
    RangeError,
    "take events must be strictly increasing",
  ],
  [
    "a repeated event",
    () => new PoolCommitRecord("e", "e", null, { take: { inbox: "m", events: [9, 9], closed: false } }),
    RangeError,
    "take events must be strictly increasing",
  ],
  [
    "a negative event",
    () => new PoolCommitRecord("e", "e", null, { take: { inbox: "m", events: [-1], closed: false } }),
    RangeError,
    "take events must be positive integers",
  ],
  [
    "event seq 0, the run start",
    () => new PoolCommitRecord("e", "e", null, { take: { inbox: "m", events: [0, 4], closed: false } }),
    RangeError,
    "take events must be positive integers",
  ],
  [
    "an empty inbox",
    () => new PoolCommitRecord("e", "e", null, { take: { inbox: "", events: [], closed: false } }),
    RangeError,
    "take inbox must be non-empty",
  ],
  [
    "a non-boolean closed",
    () => new PoolCommitRecord("e", "e", null, { take: { inbox: "m", events: [], closed: 0 as never } }),
    TypeError,
    "take closed must be a boolean",
  ],
])("the take commit refuses %s", (_name, build, cls, message) => expectError(build, cls, message));

test.each<[string, Record<string, unknown>, string]>([
  [
    "a take without closed",
    { node_id: "h", key: "h", inbox: "msgs", events: [9n] },
    'invalid pool_commit payload fields; missing=["closed"], extra=[]',
  ],
  [
    "a value mixed with an inbox",
    { node_id: "h", key: "h", kind: "int", value: 1n, inbox: "msgs", events: [], closed: false },
    'invalid pool_commit payload fields; missing=[], extra=["kind","value"]',
  ],
])("decode refuses a pool_commit with %s", (_name, payload, message) => {
  expectError(() => decodeRecord(line("pool_commit", payload)), RangeError, message);
});

test("the record-time helpers are exported", () => {
  expect(formatRecordTime(Date.UTC(2026, 9, 24, 9))).toBe("2026-10-24T09:00:00.000Z");
  expect(requireRecordTime(DEADLINE, "deadline")).toBe(DEADLINE);
  expectError(
    () => requireRecordTime("2026-02-30T00:00:00.000Z", "deadline"),
    RangeError,
    "deadline must be a UTC time such as 2026-10-20T14:02:11.123Z, found '2026-02-30T00:00:00.000Z'",
  );
  expectError(() => requireRecordTime(5, "deadline"), TypeError, "deadline must be a string");
  expectError(
    () => requireRecordTime("tomorrow"),
    RangeError,
    "run-record envelope at must be a UTC time such as 2026-10-20T14:02:11.123Z, found 'tomorrow'",
  );
});

test("a fresh RunState holds an empty waits fold and no inbox events", () => {
  const state = new RunState(flow());
  expect(state.waits).toBeInstanceOf(WaitsState);
  for (const map of [state.waits.waiting, state.waits.calls, state.waits.inboxes, state.waits.interrupts]) {
    expect(map.size).toBe(0);
  }
  expect(state.waits.lastFrontier).toBeNull();
  expect(state.inboxEvents.size).toBe(0);
});

test.each<[string, () => RunRecord]>([
  ["RunCancelledRecord", () => new RunCancelledRecord()],
  ["RunFailedRecord", () => new RunFailedRecord(FAILURE)],
])("the reducer doesn't prepare a %s yet", (name, build) => {
  const { state, log } = initializedAndStarted(flow());
  commitStartSuccess(state, log);
  expectError(() => prepareRecord(state, build()), TypeError, `unsupported run record for state preparation: ${name}`);
});

test("a state whose waits fold was replaced is refused", () => {
  const { state, log } = initializedAndStarted(flow());
  commitStartSuccess(state, log);
  (state as { waits: unknown }).waits = {};
  expectError(
    () => prepareRecord(state, new NodeStartedRecord("work")),
    RangeError,
    "transition requires initialized reducer containers",
  );
});

test.each<[string, unknown, number]>([
  ["text", "hello", 7],
  ["a record, keys sorted", { b: 1n, a: "é" }, 16],
  ["a float", 1.5, 3],
  ["a whole float", { x: 2.0 }, 9],
  ["an int past 2^53", 2n ** 60n, 19],
])("eventBytes measures %s as UTF-8 bytes of its run-log JSON", (_name, value, size) => {
  expect(eventBytes(value)).toBe(size);
});

test("eventBytes is the same whatever the key order", () => {
  expect(eventBytes({ a: "é", b: 1n })).toBe(eventBytes({ b: 1n, a: "é" }));
});

test("run_resumed names the calls it closes, none by default", () => {
  expect(new RunResumedRecord(["a"], 2).closedCalls).toEqual([]);
  expect(new RunResumedRecord(["a"], 2, ["r/a/1/0"]).closedCalls).toEqual(["r/a/1/0"]);
  expectError(() => new RunResumedRecord(["a"], 2, ["c", "c"]), RangeError, "closed_calls must contain unique IDs");
  const record = new RunResumedRecord(["a"], 2, ["r/a/1/0"]);
  const encoded = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "r", 4, record));
  expect(encoded).toContain('"payload":{"closed_calls":["r/a/1/0"],"num_workers":2,"retry_node_ids":["a"]}');
  expect(decodeRecord(encoded).record).toEqual(record);
  expectError(
    () => decodeRecord(line("run_resumed", { retry_node_ids: ["a"], num_workers: 2n })),
    RangeError,
    'invalid run_resumed payload fields; missing=["closed_calls"], extra=[]',
  );
});

test("the reducer refuses a resume closing a call no record opened", () => {
  const { state } = initializedAndStarted(flow());
  expectError(
    () => prepareRecord(state, new RunResumedRecord([StartNode.ID], 2, ["r/x/1"])),
    RangeError,
    `RunResumedRecord closed_calls must exactly equal the retried nodes' open calls: expected [], found ["r/x/1"]`,
  );
  expect(prepareRecord(state, new RunResumedRecord([StartNode.ID], 2)).record).toEqual(new RunResumedRecord([StartNode.ID], 2));
});

test.each<[string, (state: RunState) => void]>([
  ["a waiting node", (state) => state.waits.waiting.set("work", { seq: 3, callIds: ["r/work"], steerInbox: null })],
  ["a call", (state) => state.waits.calls.set("r/work", {} as never)],
  ["an inbox", (state) => state.waits.inboxes.set("msgs", { unread: [], closed: false, absorbing: [] })],
  ["an interrupt", (state) => state.waits.interrupts.set("msgs", 4)],
  ["a frontier kind", (state) => (state.waits.lastFrontier = "node_started")],
  ["an inbox event", (state) => state.inboxEvents.set(4, null as never)],
])("a run doesn't start on a state holding %s", (_name, dirty) => {
  const state = new RunState(flow());
  dirty(state);
  expectError(() => prepareRunStarted(state, startData()), RangeError, "RunStarted requires a pristine RunState at sequence 0");
});
