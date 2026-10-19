/**
 * In-memory run logs, the run-log reader and the schema checks: a port of
 * `tests/engine/test_run_log.py`.
 *
 * Ported: `test_memory_log_orders_records_and_retains_a_canonical_copy`,
 * `test_memory_log_reopening_rejects_invalid_prefix_coordinates` (the empty, gap, duplicate and
 * foreign rows), `test_memory_log_rejects_invalid_run_ids`, the memory rows of
 * `test_prefix_identity_rejects_corruption` and
 * `test_schema_check_does_not_weaken_payload_validation`. Each rejection is pinned with its class
 * and message; Python's `ValueError` is a `RangeError` and its `RuntimeError` an `Error`, and a
 * run id that isn't text is a `RangeError` as in Python.
 *
 * TS-only: a run id holding a lone surrogate; a record appended while iterating doesn't show in
 * that iteration, and `records()` is a fresh frozen array; disposing a memory log closes it; the
 * reader `readRunLog` (a complete prefix read without the lock, an incomplete tail ignored and
 * left in place, an empty file, a corrupt line named by path, line and seq, a line that isn't
 * UTF-8).
 *
 * Changed:
 * - `MemoryRunLog._from_records` is `MemoryRunLog.fromRecords`, and Python's "tuple" reads
 *   "array".
 * - `test_memory_log_reopens_a_detached_prefix_at_the_next_sequence` and
 *   `test_memory_log_never_exposes_its_retained_mutable_payload`: records are frozen, so a
 *   reader's edit throws `TypeError` where Python edits a private copy.
 * - `test_memory_log_reopening_revalidates_exact_envelope_authority`: the `record` row is the
 *   `NodeStartedRecord` constructor's rejection, and the `envelope` row (a `RecordEnvelope`
 *   subclass) is a `TypeError`.
 * - `test_retired_schema_record_is_rejected_by_name`: every schema below 10 is Python's, so 7 and
 *   8 read `unsupported run-record schema: N` and 9 names Python yian.
 * - `test_schema_check_does_not_weaken_validation`: Python's 10 and 9.0 rows are 11 and the float
 *   token `10.0` (the current schema written as a float); the float row is wire-only, since a JS
 *   `10.0` is the number 10.
 *
 * Not ported: `test_schema_version_is_nine` (`SCHEMA_VERSION` is 10, pinned in
 * `run_records.test.ts`); the list and tuple-subclass rows of the reopening-coordinates test (a
 * TS prefix is an array); the forgery rows of the envelope-authority and schema-check tests
 * (`object.__setattr__`; a frozen TS envelope refuses the edit). The JSONL tests come later in
 * this file.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { lockFile } from "../../src/platform.ts";
import {
  decodeRecord,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeStartedRecord,
  readRunLog,
  RecordEnvelope,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";

const DIGEST = "d".repeat(64);
const PYTHON_SCHEMA = "run log schema 9 was written by Python yian; resume it with Python yian";
const NOT_CONTIGUOUS = "reopened prefix must have one run and contiguous sequences";

/** Assert `fn` throws an error of exactly class `cls` with exactly `message`, and return it. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): Error {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
  return caught as Error;
}

/**
 * Return one `node_started` line without its newline; `schema` is literal JSON text, so a float
 * or boolean token can be written.
 */
function wireLine({ runId = "run", seq = 0, schema = String(SCHEMA_VERSION), nodeId = "work" } = {}): string {
  const payload = JSON.stringify({ node_id: nodeId });
  const id = JSON.stringify(runId);
  return `{"kind":"node_started","payload":${payload},"run_id":${id},"schema":${schema},"seq":${seq}}`;
}

/** The record's seed, typed for reading and for the edits a frozen seed refuses. */
function seedOf(envelope: RecordEnvelope): any {
  return (envelope.record as NodeExpandedRecord).seed;
}

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-log-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

test.each([7, 8])("retired schema %d record is rejected by name", (schema) => {
  const message = `unsupported run-record schema: ${schema}`;
  expectError(() => decodeRecord(wireLine({ schema: String(schema) })), RangeError, message);
  expectError(() => new RecordEnvelope(schema, "run", 0, new NodeStartedRecord("work")), RangeError, message);
});

test("a schema 9 record names Python yian", () => {
  expectError(() => decodeRecord(wireLine({ schema: "9" })), RangeError, PYTHON_SCHEMA);
  expectError(() => new RecordEnvelope(9, "run", 0, new NodeStartedRecord("work")), RangeError, PYTHON_SCHEMA);
});

test.each<[string, unknown, string]>([
  ["6", 6, "6"],
  ["7", 7, "7"],
  ["8", 8, "8"],
  ["11", 11, "11"],
  ["true", true, "true"],
  ["false", false, "false"],
  ["10.0", undefined, "10.0"],
])("the schema check doesn't weaken validation: %s", (token, value, shown) => {
  const message = `unsupported run-record schema: ${shown}`;
  if (value !== undefined) {
    expectError(
      () => new RecordEnvelope(value as number, "run", 0, new NodeStartedRecord("work")),
      RangeError,
      message,
    );
  }
  expectError(() => decodeRecord(wireLine({ schema: token })), RangeError, message);
});

test.each<[string, (wire: any) => void, new (...args: never[]) => Error, string]>([
  [
    "extra",
    (wire) => {
      wire.payload.extra = null;
    },
    RangeError,
    'invalid node_started payload fields; missing=[], extra=["extra"]',
  ],
  [
    "node_id",
    (wire) => {
      wire.payload.node_id = true;
    },
    TypeError,
    "node_started payload node_id must be a non-empty string",
  ],
  [
    "seq",
    (wire) => {
      wire.seq = false;
    },
    TypeError,
    "run-record envelope seq must be an integer",
  ],
  [
    "kind",
    (wire) => {
      wire.kind = "unsupported";
    },
    RangeError,
    "run-record kind 'unsupported' is unsupported in schema 10",
  ],
])("the schema check doesn't weaken payload validation: %s", (_name, corrupt, cls, message) => {
  const wire = JSON.parse(wireLine());
  corrupt(wire);
  expectError(() => decodeRecord(JSON.stringify(wire)), cls, message);
});

test.each(["run_id", "gap"])("prefix identity rejects corruption: memory, %s", (corruption) => {
  const suffix = wireLine({
    seq: corruption === "gap" ? 2 : 1,
    runId: corruption === "run_id" ? "foreign" : "run",
  });
  const records = [wireLine(), suffix].map((line) => decodeRecord(line));
  expectError(() => MemoryRunLog.fromRecords(records), RangeError, NOT_CONTIGUOUS);
});

test("a memory log orders records and retains a canonical copy", () => {
  const seed = { items: [{ value: 1n }] };
  const first = new NodeExpandedRecord("each", seed, [], DIGEST);
  const log = new MemoryRunLog("memory-run");

  const firstEnvelope = log.append(first);
  const secondEnvelope = log.append(new NodeStartedRecord("next"));
  seed.items[0]!.value = 99n;
  seed.items.push({ value: 2n });

  expect(log.path).toBeNull();
  expect(firstEnvelope.seq).toBe(0);
  expect(secondEnvelope.seq).toBe(1);
  expect(log.nextSequence).toBe(2);
  expect([...log]).toEqual([firstEnvelope, secondEnvelope]);
  expect(log.records()).toEqual([firstEnvelope, secondEnvelope]);
  expect(seedOf(firstEnvelope)).toEqual({ items: [{ value: 1n }] });

  log.close();
  log.close();
  expect([...log]).toEqual([firstEnvelope, secondEnvelope]);
  expectError(() => log.append(new NodeStartedRecord("late")), Error, "run log is closed");
});

test("a memory log reopens a detached prefix at the next sequence", () => {
  const original = new MemoryRunLog("retained");
  original.append(new NodeExpandedRecord("each", { items: [{ value: 1n }] }, [], DIGEST));
  original.close();
  const prefix = original.records();
  const reopened = MemoryRunLog.fromRecords(prefix);
  try {
    expect(() => {
      seedOf(prefix[0]!).items[0].value = 2n;
    }).toThrow(TypeError);
    const exported = reopened.records();
    expect(() => seedOf(exported[0]!).items.push({ value: 3n })).toThrow(TypeError);
    expect(seedOf([...reopened][0]!)).toEqual({ items: [{ value: 1n }] });
    expect(seedOf(original.records()[0]!)).toEqual({ items: [{ value: 1n }] });
    expect(exported[0]).not.toBe(prefix[0]);
    const accepted = reopened.append(new NodeStartedRecord("next"));
    expect(accepted.seq).toBe(1);
    expect(reopened.nextSequence).toBe(2);
    expect(original.records()).toHaveLength(1);
    expectError(() => original.append(new NodeStartedRecord("late")), Error, "run log is closed");
  } finally {
    reopened.close();
  }
});

test.each<[string, (records: RecordEnvelope[]) => RecordEnvelope[], string]>([
  ["empty", () => [], "reopening requires a nonempty record array"],
  ["gap", () => [new RecordEnvelope(SCHEMA_VERSION, "run", 1, new NodeStartedRecord("first"))], NOT_CONTIGUOUS],
  ["duplicate", (records) => [...records, ...records], NOT_CONTIGUOUS],
  [
    "foreign",
    (records) => [...records, new RecordEnvelope(SCHEMA_VERSION, "other", 1, new NodeStartedRecord("second"))],
    NOT_CONTIGUOUS,
  ],
])("memory log reopening rejects invalid prefix coordinates: %s", (_name, change, message) => {
  const records = change([new RecordEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("first"))]);
  expectError(() => MemoryRunLog.fromRecords(records), RangeError, message);
});

test("memory log reopening revalidates exact envelope authority: record", () => {
  expectError(() => new NodeStartedRecord(1 as never), TypeError, "node_id must be a string");
});

test("memory log reopening revalidates exact envelope authority: envelope", () => {
  class OtherEnvelope extends RecordEnvelope {}
  const envelope = new OtherEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("first"));
  expectError(() => MemoryRunLog.fromRecords([envelope]), TypeError, "reopening requires exact RecordEnvelope values");
});

test("a memory log iterates a snapshot and returns a fresh frozen records array", () => {
  const log = new MemoryRunLog("run");
  log.append(new NodeStartedRecord("first"));
  const seen: number[] = [];
  for (const envelope of log) {
    seen.push(envelope.seq);
    if (envelope.seq === 0) log.append(new NodeStartedRecord("second"));
  }
  expect(seen).toEqual([0]);
  const records = log.records();
  expect(records.map((envelope) => envelope.seq)).toEqual([0, 1]);
  expect(Object.isFrozen(records)).toBe(true);
  expect(log.records()).not.toBe(records);
});

test("disposing a memory log closes it and keeps its records", () => {
  let log: MemoryRunLog;
  {
    using scoped = new MemoryRunLog("run");
    scoped.append(new NodeStartedRecord("work"));
    log = scoped;
  }
  expectError(() => log.append(new NodeStartedRecord("late")), Error, "run log is closed");
  expect(log.records()).toHaveLength(1);
});

test("a memory log never exposes its retained mutable payload", () => {
  const log = new MemoryRunLog("memory-run");
  const appended = log.append(new NodeExpandedRecord("each", { items: [{ value: 1n }] }, [], DIGEST));

  expect(() => seedOf(appended).items.push({ from: "append" })).toThrow(TypeError);
  const iterated = log[Symbol.iterator]().next().value!;
  expect(() => {
    seedOf(iterated).items[0].value = 99n;
  }).toThrow(TypeError);
  const snapshotted = log.records()[0]!;
  expect(() => seedOf(snapshotted).items.push({ from: "records" })).toThrow(TypeError);

  expect(seedOf(log[Symbol.iterator]().next().value!)).toEqual({ items: [{ value: 1n }] });
  expect(seedOf(log.records()[0]!)).toEqual({ items: [{ value: 1n }] });
});

test.each<[string, unknown, string]>([
  ["empty", "", "run log run_id must be a non-empty string"],
  ["a number", 7, "run log run_id must be a non-empty string"],
  ["null", null, "run log run_id must be a non-empty string"],
  ["a lone surrogate (TS-only)", "\ud800", "Unicode surrogate is not valid canonical JSON"],
])("a memory log rejects an invalid run id: %s", (_name, runId, message) => {
  expectError(() => new MemoryRunLog(runId as string), RangeError, message);
});

test("readRunLog reads the complete prefix without the lock", () => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${wireLine()}\n${wireLine({ seq: 1, nodeId: "next" })}\n`);
  const nodeIds = (records: readonly RecordEnvelope[]) =>
    records.map((envelope) => (envelope.record as NodeStartedRecord).nodeId);

  expect(nodeIds(readRunLog(file))).toEqual(["work", "next"]);
  expect(fs.existsSync(`${file}.lock`)).toBe(false);
  using _lock = lockFile(`${file}.lock`);
  expect(nodeIds(readRunLog(file))).toEqual(["work", "next"]);
});

test("readRunLog ignores an incomplete tail and leaves it in place", () => {
  const file = path.join(dir, "run.jsonl");
  const bytes = `${wireLine()}\n${wireLine({ seq: 1 }).slice(0, 20)}`;
  fs.writeFileSync(file, bytes);
  expect(readRunLog(file).map((envelope) => envelope.seq)).toEqual([0]);
  expect(fs.readFileSync(file, "utf8")).toBe(bytes);
});

test.each([
  ["an empty file", ""],
  ["no complete line", wireLine().slice(0, 20)],
])("readRunLog reads %s as no records", (_name, bytes) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, bytes);
  expect(readRunLog(file)).toEqual([]);
});

test.each<[string, string, string, (new (...args: never[]) => Error) | null]>([
  ["invalid JSON", "{", "invalid run-record JSON", RangeError],
  [
    "a wrong field type",
    wireLine({ seq: 1 }).replace('"work"', "true"),
    "node_started payload node_id must be a non-empty string",
    TypeError,
  ],
  ["a sequence gap", wireLine({ seq: 2 }), "run-log sequence mismatch: expected 1, found 2", null],
  [
    "a foreign run id",
    wireLine({ seq: 1, runId: "foreign" }),
    "run-log run_id mismatch: expected 'run', found 'foreign'",
    null,
  ],
])("readRunLog names the path, line and seq of %s", (_name, line, message, causeClass) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${wireLine()}\n${line}\n`);
  const error = expectError(() => readRunLog(file), RangeError, `run log ${file} line 2 (seq 1): ${message}`);
  if (causeClass === null) expect(error.cause).toBeUndefined();
  else expect((error.cause as Error).constructor).toBe(causeClass);
});

test("readRunLog reads a line that isn't UTF-8 as invalid record JSON", () => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, new Uint8Array([0xff, 0x0a]));
  const message = `run log ${file} line 1 (seq 0): invalid run-record JSON`;
  const error = expectError(() => readRunLog(file), RangeError, message);
  expect((error.cause as Error).message).toBe("invalid run-record JSON");
});
