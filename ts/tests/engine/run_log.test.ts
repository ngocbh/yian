/**
 * In-memory and locked JSONL run logs, the run-log reader and the schema checks: a port of
 * `tests/engine/test_run_log.py`.
 *
 * Ported: `test_schema_version_is_nine`, `test_memory_log_orders_records_and_retains_a_canonical_copy`,
 * `test_memory_log_reopening_rejects_invalid_prefix_coordinates` (the empty, gap, duplicate and
 * foreign rows), `test_memory_log_rejects_invalid_run_ids`, `test_prefix_identity_rejects_corruption`,
 * `test_schema_check_does_not_weaken_payload_validation`,
 * `test_jsonl_create_is_exclusive_and_preserves_an_existing_file`,
 * `test_jsonl_create_sets_owner_read_write_permissions`, `test_jsonl_excludes_a_second_writer`,
 * `test_jsonl_resume_truncates_and_syncs_every_incomplete_tail`,
 * `test_jsonl_resume_rejects_a_corrupt_complete_line_and_releases_lock`,
 * `test_jsonl_resume_errors_name_the_log_line_and_seq`, `test_jsonl_resume_empty_error_names_the_log`,
 * `test_jsonl_resume_rejects_identity_mismatches_and_releases_lock`,
 * `test_jsonl_resume_leaves_a_rejected_file_unchanged`,
 * `test_jsonl_resume_rejects_an_empty_file_and_releases_lock`,
 * `test_jsonl_close_is_idempotent_keeps_cache_and_releases_lock`,
 * `test_jsonl_context_releases_lock_when_the_body_raises` (`Symbol.dispose` in a `finally`),
 * `test_jsonl_append_uses_one_complete_write_flush_and_fsync`,
 * `test_jsonl_append_and_cache_are_isolated_from_caller_mutation`,
 * `test_jsonl_never_exposes_its_retained_mutable_payload`, `test_append_rollback_preserves_log`,
 * `test_invalid_append_retains_the_prior_complete_prefix`,
 * `test_write_failure_rolls_back_to_the_prior_complete_prefix`,
 * `test_fsync_failure_rolls_back_and_syncs_the_prior_complete_prefix`,
 * `test_short_write_is_an_append_failure_and_rolls_back` and
 * `test_tail_repair_failure_releases_the_lock`. Each rejection is pinned with its class and
 * message; Python's `ValueError` is a `RangeError` and its `RuntimeError` an `Error`, and a run id
 * that isn't text is a `RangeError` as in Python. A free lock is probed with
 * `isLocked(<log>.lock)`, where Python takes the `flock` itself. A failed write or sync is
 * injected with a spy on the `fs` operation, filtered to the log's descriptor, where Python wraps
 * the file object.
 *
 * TS-only: a run id holding a lone surrogate (also refused by `create` before any file exists); a
 * record appended while iterating doesn't show in that iteration, and `records()` is a fresh
 * frozen array; disposing a memory log closes it; appending a value that isn't a record leaves a
 * memory log unchanged; resuming a missing log throws `ENOENT` and creates no lock file;
 * `isRunLogLocked`; a symlink to a log shares the real log's lock; the reader `readRunLog` (a complete prefix read without the lock, an
 * incomplete tail ignored and left in place, an empty file, a corrupt line named by path, line and
 * seq, a line that isn't UTF-8); the streaming reader `iterRunRecords` (each record's seq, offsets,
 * time, kind, node and envelope, the time also with `payloads: false`; a log a writer holds; a torn last line skipped, then read from its
 * offset once complete; a log another process appends to, followed from saved offsets without
 * blocking the writer; a log that shrank before, at the end-of-line probe, or while it reads, as
 * `RunLogChangedError`; another run's log renamed over the path, caught by `runId` and `afterSeq`,
 * and their option checks; bad options and a missing file thrown on the first `next()`; offsets
 * that aren't the end of a line; seq checked from the start and after an offset, and the run id; `payloads: false`, and
 * what it still refuses; `kinds`; schemas 9, 10 and 12 as `RunLogSchemaError`; bounded reads that go
 * only as far as asked); a failed resume whose descriptor close also fails throws the
 * original error; a write that fails and whose rollback truncate also fails; closing a poisoned
 * log, after which an append throws `run log is closed`; with `YIAN_TEST_NODE` set to a Node
 * binary, the built log on Node through a failed sync, a failed write, a short write and a write
 * whose rollback truncate also fails.
 *
 * TS-only, record times (schema 11 on): each log stamps `at` from its clock, as read, even when the
 * clock moves back, and drops a fraction of a millisecond; the default clock is the wall clock; a
 * reading that isn't finite or falls outside the years 0000 to 9999 fails the append before any
 * write, without poisoning the log; a clock that isn't a function is refused before any file is
 * opened; a reopened log keeps its records' times and stamps new ones from its own clock; the
 * envelope's and the decoder's time checks, and the first and last times they take; an envelope
 * without a time encodes without `at`, decodes back with `null`, and every log reader refuses it;
 * a schema 10 record, from before record times, and a schema 11 one, from before `run_started` held
 * the flow files, are retired.
 *
 * Changed:
 * - `test_schema_version_is_nine`: `SCHEMA_VERSION` is 12.
 * - `MemoryRunLog._from_records` is `MemoryRunLog.fromRecords`, and Python's "tuple" reads
 *   "array".
 * - `test_memory_log_reopens_a_detached_prefix_at_the_next_sequence`,
 *   `test_memory_log_never_exposes_its_retained_mutable_payload` and
 *   `test_jsonl_never_exposes_its_retained_mutable_payload`: records are frozen, so a reader's
 *   edit throws `TypeError` where Python edits a private copy.
 * - `test_memory_log_reopening_revalidates_exact_envelope_authority`: the `record` row is the
 *   `NodeStartedRecord` constructor's rejection, and the `envelope` row (a `RecordEnvelope`
 *   subclass) is a `TypeError`.
 * - `test_retired_schema_record_is_rejected_by_name` and `test_retired_schema_log_is_rejected_on_resume`:
 *   every schema below 10 is Python's, so 7 and 8 read `unsupported run-record schema: N` and 9
 *   names Python yian; TS's own schemas 10 and 11 are retired as Python names its retired
 *   schemas. Every schema rejection is a `RunLogSchemaError` (Python's `ValueError`).
 * - `test_schema_check_does_not_weaken_validation`: Python's 10 and 9.0 rows are 13 and the float
 *   token `18.0` (the current schema written as a float); the float row is wire-only, since a JS
 *   `18.0` is the number 18.
 * - The writer's lock is a SQLite lock on the sibling `<log>.lock`, not a `flock` on the log:
 *   `test_jsonl_resume_locks_before_its_first_read` checks the lock is held when the log is read;
 *   `test_jsonl_lock_contention_names_the_log_and_the_likely_cause` pins a busy `FileLockError` as
 *   the cause, not an errno; `test_jsonl_excludes_a_writer_in_a_separate_process` runs the child
 *   in the test's runtime and matches the busy message.
 * - `test_jsonl_create_is_exclusive_and_preserves_an_existing_file`: the error's `code` is
 *   `EEXIST` (Python's `FileExistsError`).
 * - `test_jsonl_resume_empty_error_names_the_log` and
 *   `test_jsonl_resume_rejects_an_empty_file_and_releases_lock` are one test.
 * - `test_jsonl_resume_truncates_and_syncs_every_incomplete_tail`: one `ftruncateSync` and one
 *   `fsyncSync`; TS has no buffered flush to count.
 * - `test_jsonl_create_releases_lock_when_file_adoption_fails`: TS adopts no file object, so the
 *   rows are an `fchmodSync` failure and a lock held before `create`; each closes the descriptor
 *   and leaves the empty file.
 * - Python refuses entering a closed log with `with`; TS disposal (`Symbol.dispose`) has no enter
 *   step, so there is nothing to refuse.
 * - `test_jsonl_append_uses_one_complete_write_flush_and_fsync`: one `writeSync` and one
 *   `fsyncSync`; there is no flush.
 * - A failed append poisons the log: every later append throws `run log <path> refuses appends
 *   after a failed append` with the original error as `cause`, while the lock stays held until
 *   `close()`. Python retries in place, so `test_append_rollback_preserves_log` appends its third
 *   record after a fresh resume, and the failure tests also check the poisoned error.
 * - `test_append_rollback_preserves_log`: every sync fails, so the rollback's sync fails too and
 *   the error is an `AggregateError` of both, where Python adds a note.
 * - `test_invalid_append_retains_the_prior_complete_prefix`: the record is refused by
 *   `RecordEnvelope`, before any write, and doesn't poison the log.
 * - `test_short_write_is_an_append_failure_and_rolls_back`: a short write is completed by writing
 *   the rest, so the append succeeds; a write of 0 bytes after a partial one is the failure.
 *
 * Not ported: the list and tuple-subclass rows of the reopening-coordinates test (a TS prefix is
 * an array); the forgery rows of the envelope-authority and schema-check tests
 * (`object.__setattr__`; a frozen TS envelope refuses the edit);
 * `test_flush_failure_rolls_back_to_the_prior_complete_prefix` (there is no flush).
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import { execFileSync, spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { FileLockError, isLocked, lockFile } from "../../src/platform.ts";
import {
  decodeRecord,
  encodeRecord,
  isRunLogLocked,
  iterRunRecords,
  JsonlRunLog,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeStepRecord,
  NodeStartedRecord,
  readRunLog,
  RecordEnvelope,
  RunLogChangedError,
  RunLogSchemaError,
  type RunStoppedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";

const DIGEST = "d".repeat(64);
/** A record time, as a log writes one, and a clock that always reads it. */
const AT = "2026-10-20T14:02:11.123Z";
const CLOCK = () => Date.UTC(2026, 9, 20, 14, 2, 11, 123);
/** The error text for the retired TS schema `schema`. */
const retired = (schema: number) => `run log schema ${schema} predates schema 18 and is not supported`;
const RETIRED_SCHEMA = retired(17);
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
  return `{"at":"${AT}","kind":"node_started","payload":${payload},"run_id":${id},"schema":${schema},"seq":${seq}}`;
}

/** A step record's `items`, typed for reading and for the edits a frozen step refuses. */
function seedOf(envelope: RecordEnvelope): any {
  return { items: ((envelope.record as NodeStepRecord).step as any).items };
}

/** A step record holding the nested `items` that `seedOf` reads. */
const nestedStep = (items: unknown = [{ value: 1n }]) => new NodeStepRecord("each", 1, { kind: "test", items } as never);

/** Assert no writer holds the lock of the log at `file`. */
function expectUnlocked(file: string): void {
  expect(isLocked(`${file}.lock`)).toBe(false);
}

/** Create a log at `file` holding one `node_started` record for `first`. */
function baselineLog(file: string): JsonlRunLog {
  const log = JsonlRunLog.create(file, "run", { clock: CLOCK });
  log.append(new NodeStartedRecord("first"));
  return log;
}

/** Return whether `fd` is a descriptor open on the file at `file`, so a spy skips the lock's own calls. */
function isDescriptorOf(fd: unknown, file: string): boolean {
  if (typeof fd !== "number" || !fs.existsSync(file)) return false;
  const open = fs.fstatSync(fd);
  const named = fs.statSync(file);
  return open.dev === named.dev && open.ino === named.ino;
}

type FsOperation = "writeSync" | "fsyncSync" | "ftruncateSync" | "closeSync";

/**
 * Replace `fs[name]`: a call on a descriptor of `file` runs `onLog(real, args)`, where `real`
 * calls the original operation; every other call (the lock's own) goes to the original.
 */
function spyOnLog(name: FsOperation, file: string, onLog: (real: (...args: any[]) => any, args: any[]) => unknown) {
  const original = fs[name] as (...args: unknown[]) => unknown;
  const real = (...args: unknown[]) => Reflect.apply(original, fs, args);
  return spyOn(fs as any, name).mockImplementation((...args: any[]) =>
    isDescriptorOf(args[0], file) ? onLog(real, args) : real(...args),
  );
}

/**
 * Assert `append` throws, leaving the file's bytes, the records and `nextSequence` at the prior
 * complete prefix, and return the thrown error.
 */
function expectAppendFailure(log: JsonlRunLog, file: string, append: () => unknown): unknown {
  const bytes = fs.readFileSync(file);
  const records = log.records();
  const sequence = log.nextSequence;
  let caught: unknown;
  try {
    append();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeDefined();
  expect(fs.readFileSync(file)).toEqual(bytes);
  expect(log.records()).toEqual(records);
  expect(log.nextSequence).toBe(sequence);
  return caught;
}

/** Assert `log` refuses appends with `cause` as the reason, and still holds its lock. */
function expectPoisoned(log: JsonlRunLog, file: string, cause: unknown): void {
  const message = `run log ${file} refuses appends after a failed append`;
  const error = expectError(() => log.append(new NodeStartedRecord("late")), Error, message);
  expect(error.cause).toBe(cause);
  expect(isRunLogLocked(file)).toBe(true);
}

/** Return the node ids of `records`, which are all `node_started`. */
function nodeIds(records: Iterable<RecordEnvelope>): string[] {
  return [...records].map((envelope) => (envelope.record as NodeStartedRecord).nodeId);
}

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-log-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

test("the schema version is 18", () => {
  expect(SCHEMA_VERSION).toBe(18);
});

test.each([7, 8])("retired schema %d record is rejected by name", (schema) => {
  const message = `unsupported run-record schema: ${schema}`;
  expectError(() => decodeRecord(wireLine({ schema: String(schema) })), RunLogSchemaError, message);
  expectError(() => new RecordEnvelope(schema, "run", 0, new NodeStartedRecord("work")), RunLogSchemaError, message);
});

test("a schema 9 record names Python yian", () => {
  expectError(() => decodeRecord(wireLine({ schema: "9" })), RunLogSchemaError, PYTHON_SCHEMA);
  expectError(() => new RecordEnvelope(9, "run", 0, new NodeStartedRecord("work")), RunLogSchemaError, PYTHON_SCHEMA);
});

test("a schema 17 record, from before the inbox and wait records, is retired", () => {
  const line = `{"kind":"node_started","payload":{"node_id":"work"},"run_id":"run","schema":17,"seq":0}`;
  const error = expectError(() => decodeRecord(line), RunLogSchemaError, RETIRED_SCHEMA) as RunLogSchemaError;
  expect(error.schema).toBe(17);
  expectError(() => new RecordEnvelope(17, "run", 0, new NodeStartedRecord("work")), RunLogSchemaError, RETIRED_SCHEMA);
});

test.each<[number, string]>([
  [7, "unsupported run-record schema: 7"],
  [8, "unsupported run-record schema: 8"],
  [9, PYTHON_SCHEMA],
  [10, retired(10)],
  [11, retired(11)],
  [12, retired(12)],
  [13, retired(13)],
  [14, retired(14)],
  [15, retired(15)],
  [16, retired(16)],
  [17, RETIRED_SCHEMA],
])("a retired schema %d log is rejected on resume", (schema, message) => {
  const file = path.join(dir, "retired.jsonl");
  const bytes = `${wireLine({ schema: String(schema) })}\n`;
  fs.writeFileSync(file, bytes);
  const error = expectError(() => JsonlRunLog.resume(file), RunLogSchemaError, `run log ${file} line 1 (seq 0): ${message}`);
  expect((error as RunLogSchemaError).schema).toBe(schema);
  expect(fs.readFileSync(file, "utf8")).toBe(bytes);
  expectUnlocked(file);
});

test.each<[string, unknown, string]>([
  ["6", 6, "6"],
  ["7", 7, "7"],
  ["8", 8, "8"],
  ["19", 19, "19"],
  ["true", true, "true"],
  ["false", false, "false"],
  ["18.0", undefined, "18.0"],
])("the schema check doesn't weaken validation: %s", (token, value, shown) => {
  const message = `unsupported run-record schema: ${shown}`;
  if (value !== undefined) {
    expectError(
      () => new RecordEnvelope(value as number, "run", 0, new NodeStartedRecord("work")),
      RunLogSchemaError,
      message,
    );
  }
  expectError(() => decodeRecord(wireLine({ schema: token })), RunLogSchemaError, message);
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
    "run-record kind 'unsupported' is unsupported in schema 18",
  ],
])("the schema check doesn't weaken payload validation: %s", (_name, corrupt, cls, message) => {
  const wire = JSON.parse(wireLine());
  corrupt(wire);
  expectError(() => decodeRecord(JSON.stringify(wire)), cls, message);
});

test.each<[string, string]>([
  ["run_id", "run-log run_id mismatch: expected 'run', found 'foreign'"],
  ["gap", "run-log sequence mismatch: expected 1, found 2"],
])("prefix identity rejects corruption: jsonl, %s", (corruption, message) => {
  const suffix = wireLine({
    seq: corruption === "gap" ? 2 : 1,
    runId: corruption === "run_id" ? "foreign" : "run",
  });
  const file = path.join(dir, "invalid-prefix.jsonl");
  const bytes = `${wireLine()}\n${suffix}\n`;
  fs.writeFileSync(file, bytes);
  expectError(() => JsonlRunLog.resume(file), RangeError, `run log ${file} line 2 (seq 1): ${message}`);
  expect(fs.readFileSync(file, "utf8")).toBe(bytes);
  expectUnlocked(file);
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
  const first = nestedStep(seed.items);
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

/** A clock that reads each of `times` in turn, in epoch milliseconds. */
function clockOf(...times: number[]): () => number {
  return () => times.shift()!;
}

test.each(["memory", "jsonl"])("a %s log stamps each record with its clock's time, as read", (kind) => {
  const file = path.join(dir, "run.jsonl");
  // The second reading is earlier: a clock can move back, and `seq` stays the order. Its fraction
  // of a millisecond is dropped.
  const clock = clockOf(Date.UTC(2026, 9, 20, 14, 2, 11, 123), Date.UTC(2026, 9, 20, 14, 2, 10, 0) + 0.75);
  const log = kind === "memory" ? new MemoryRunLog("run", { clock }) : JsonlRunLog.create(file, "run", { clock });
  try {
    const first = log.append(new NodeStartedRecord("work"));
    const second = log.append(new NodeStartedRecord("next"));
    expect([first.at, second.at]).toEqual([AT, "2026-10-20T14:02:10.000Z"]);
    expect([first.seq, second.seq]).toEqual([0, 1]);
    expect(encodeRecord(first)).toBe(wireLine());
    expect(log.records().map((envelope) => envelope.at)).toEqual([AT, "2026-10-20T14:02:10.000Z"]);
  } finally {
    log.close();
  }
  if (kind === "jsonl") {
    expect(fs.readFileSync(file, "utf8").split("\n")[0]).toBe(wireLine());
    expect(readRunLog(file).map((envelope) => envelope.at)).toEqual([AT, "2026-10-20T14:02:10.000Z"]);
  }
});

test("a log reads the wall clock by default", () => {
  const log = new MemoryRunLog("run");
  const before = Date.now();
  const { at } = log.append(new NodeStartedRecord("work"));
  const after = Date.now();
  const ms = new Date(at!).getTime();
  expect(ms).toBeGreaterThanOrEqual(before);
  expect(ms).toBeLessThanOrEqual(after);
});

test.each<[string, unknown, string]>([
  ["NaN", Number.NaN, "run-log clock must return finite epoch milliseconds, got NaN"],
  ["infinity", Infinity, "run-log clock must return finite epoch milliseconds, got Infinity"],
  ["text", "now", 'run-log clock must return finite epoch milliseconds, got "now"'],
  ["a year past 9999", Date.UTC(10000, 0, 1), "run-log clock time 253402300800000 is outside the years 0000 to 9999"],
  ["a year before 0000", Date.UTC(-1, 0, 1), "run-log clock time -62198755200000 is outside the years 0000 to 9999"],
  ["past the Date range", 8.64e15 + 1, "run-log clock time 8640000000000001 is outside the years 0000 to 9999"],
])("an append whose clock reads %s throws before writing and doesn't poison the log", (_name, reading, message) => {
  const file = path.join(dir, "run.jsonl");
  let next: unknown = reading;
  const clock = () => next as number;
  const memory = new MemoryRunLog("run", { clock });
  try {
    const jsonl = JsonlRunLog.create(file, "run", { clock });
    try {
      for (const log of [memory, jsonl]) {
        expectError(() => log.append(new NodeStartedRecord("work")), RangeError, message);
        expect(log.nextSequence).toBe(0);
      }
      expect(fs.readFileSync(file, "utf8")).toBe("");
      next = CLOCK();
      for (const log of [memory, jsonl]) expect(log.append(new NodeStartedRecord("work")).at).toBe(AT);
      expect(fs.readFileSync(file, "utf8")).toBe(`${wireLine()}\n`);
    } finally {
      jsonl[Symbol.dispose]();
    }
  } finally {
    memory[Symbol.dispose]();
  }
});

test("a log refuses a clock that isn't a function, before any file is opened", () => {
  const message = "run log clock must be a function or null";
  const clock = 0 as never;
  expectError(() => new MemoryRunLog("run", { clock }), TypeError, message);
  const file = path.join(dir, "run.jsonl");
  expectError(() => JsonlRunLog.create(file, "run", { clock }), TypeError, message);
  expect(fs.existsSync(file)).toBe(false);
  baselineLog(file).close();
  expectError(() => JsonlRunLog.resume(file, { clock }), TypeError, message);
  expectUnlocked(file);
  expectError(() => MemoryRunLog.fromRecords(readRunLog(file), { clock }), TypeError, message);
});

test.each(["memory", "jsonl"])("a reopened %s log keeps its records' times and stamps new ones from its own clock", (kind) => {
  const file = path.join(dir, "run.jsonl");
  const original = baselineLog(file);
  original.close();
  const later = "2026-10-21T09:00:00.000Z";
  const clock = clockOf(Date.UTC(2026, 9, 21, 9, 0, 0, 0));
  const log = kind === "memory" ? MemoryRunLog.fromRecords(original.records(), { clock }) : JsonlRunLog.resume(file, { clock });
  try {
    log.append(new NodeStartedRecord("second"));
    expect(log.records().map((envelope) => envelope.at)).toEqual([AT, later]);
  } finally {
    log.close();
  }
});

test.each<[string, unknown, new (...args: never[]) => Error]>([
  ["without milliseconds", "2026-10-20T14:02:11Z", RangeError],
  ["with microseconds", "2026-10-20T14:02:11.123456Z", RangeError],
  ["with an offset", "2026-10-20T14:02:11.123+00:00", RangeError],
  ["in lower case", "2026-10-20t14:02:11.123z", RangeError],
  ["on a day the month lacks", "2026-02-30T14:02:11.123Z", RangeError],
  ["at hour 24", "2026-10-20T24:00:00.000Z", RangeError],
  ["with a six-digit year", "+010000-01-01T00:00:00.000Z", RangeError],
  ["empty", "", RangeError],
  ["a number", 1760968931123, TypeError],
])("the envelope refuses a time %s", (_name, at, cls) => {
  const message =
    cls === TypeError
      ? "run-record envelope at must be a string"
      : `run-record envelope at must be a UTC time such as 2026-10-20T14:02:11.123Z, found ${JSON.stringify(at).replaceAll('"', "'")}`;
  expectError(() => new RecordEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("work"), at as string), cls, message);
  const line = wireLine().replace(`"at":"${AT}"`, `"at":${JSON.stringify(at)}`);
  expectError(() => decodeRecord(line), cls, message);
});

test("the envelope takes the first and last times of the years 0000 to 9999", () => {
  for (const at of ["0000-01-01T00:00:00.000Z", "9999-12-31T23:59:59.999Z"]) {
    const envelope = new RecordEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("work"), at);
    expect(decodeRecord(encodeRecord(envelope)).at).toBe(at);
  }
  const first = new Date("0000-01-01T00:00:00.000Z").getTime();
  const log = new MemoryRunLog("run", { clock: clockOf(first, first - 1) });
  expect(log.append(new NodeStartedRecord("work")).at).toBe("0000-01-01T00:00:00.000Z");
  expectError(
    () => log.append(new NodeStartedRecord("next")),
    RangeError,
    `run-log clock time ${first - 1} is outside the years 0000 to 9999`,
  );
});

test("an envelope no log appended has no time and encodes without one, which every log reader refuses", () => {
  const prepared = new RecordEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("work"));
  expect(prepared.at).toBeNull();
  const line = encodeRecord(prepared);
  expect(line).toBe(wireLine().replace(`"at":"${AT}",`, ""));
  expect(decodeRecord(line).at).toBeNull();

  const missing = 'invalid run-record envelope fields; missing=["at"], extra=[]';
  expectError(() => MemoryRunLog.fromRecords([prepared]), RangeError, missing);
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${line}\n`);
  const where = `run log ${file} line 1 (seq 0): ${missing}`;
  expectError(() => JsonlRunLog.resume(file), RangeError, where);
  expectError(() => readRunLog(file), RangeError, where);
  expectError(() => [...iterRunRecords(file, { payloads: false })], RangeError, where);
  expectUnlocked(file);
});

test("a memory log reopens a detached prefix at the next sequence", () => {
  const original = new MemoryRunLog("retained");
  original.append(nestedStep());
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
  ["gap", () => [new RecordEnvelope(SCHEMA_VERSION, "run", 1, new NodeStartedRecord("first"), AT)], NOT_CONTIGUOUS],
  ["duplicate", (records) => [...records, ...records], NOT_CONTIGUOUS],
  [
    "foreign",
    (records) => [...records, new RecordEnvelope(SCHEMA_VERSION, "other", 1, new NodeStartedRecord("second"), AT)],
    NOT_CONTIGUOUS,
  ],
])("memory log reopening rejects invalid prefix coordinates: %s", (_name, change, message) => {
  const records = change([new RecordEnvelope(SCHEMA_VERSION, "run", 0, new NodeStartedRecord("first"), AT)]);
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
    const scoped = new MemoryRunLog("run");
    try {
      scoped.append(new NodeStartedRecord("work"));
      log = scoped;
    } finally {
      scoped[Symbol.dispose]();
    }
  }
  expectError(() => log.append(new NodeStartedRecord("late")), Error, "run log is closed");
  expect(log.records()).toHaveLength(1);
});

test("appending a value that isn't a record leaves a memory log unchanged", () => {
  const log = new MemoryRunLog("run");
  log.append(new NodeStartedRecord("work"));
  const before = log.records();
  expectError(() => log.append({} as never), TypeError, "unsupported run record: Object");
  expect(log.nextSequence).toBe(1);
  expect(log.records()).toEqual(before);
});

test("a memory log never exposes its retained mutable payload", () => {
  const log = new MemoryRunLog("memory-run");
  const appended = log.append(nestedStep());

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

  expect(nodeIds(readRunLog(file))).toEqual(["work", "next"]);
  expect(fs.existsSync(`${file}.lock`)).toBe(false);
  const lock = lockFile(`${file}.lock`);
  try {
    expect(nodeIds(readRunLog(file))).toEqual(["work", "next"]);
  } finally {
    lock[Symbol.dispose]();
  }
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

/** Return a wire line of `kind` with `payload` (JSON text), without its newline. */
function kindLine(kind: string, payload: string, { seq = 0, runId = "run", schema = String(SCHEMA_VERSION) } = {}): string {
  return `{"at":"${AT}","kind":"${kind}","payload":${payload},"run_id":${JSON.stringify(runId)},"schema":${schema},"seq":${seq}}`;
}

/** Return the byte length of `text` in UTF-8. */
function byteLength(text: string): number {
  return Buffer.byteLength(text, "utf8");
}

test("iterRunRecords yields each record's seq, offsets, time, kind, node and envelope", () => {
  const file = path.join(dir, "run.jsonl");
  const lines = [wireLine(), wireLine({ seq: 1, nodeId: "next" })];
  fs.writeFileSync(file, `${lines.join("\n")}\n`);

  const entries = [...iterRunRecords(file)];
  const first = byteLength(lines[0]!) + 1;
  const fields = entries.map(({ seq, offset, end, runId, at, kind, nodeId }) => ({ seq, offset, end, runId, at, kind, nodeId }));
  expect(fields).toEqual([
    { seq: 0, offset: 0, end: first, runId: "run", at: AT, kind: "node_started", nodeId: "work" },
    { seq: 1, offset: first, end: first + byteLength(lines[1]!) + 1, runId: "run", at: AT, kind: "node_started", nodeId: "next" },
  ]);
  expect(entries.map((entry) => entry.envelope)).toEqual(lines.map((line) => decodeRecord(line)));
  expect(entries.map((entry) => entry.envelope.at)).toEqual([AT, AT]);
  expect(fs.existsSync(`${file}.lock`)).toBe(false);
});

test("iterRunRecords reads a log a writer holds, without its lock", () => {
  const file = path.join(dir, "run.jsonl");
  const log = baselineLog(file);
  try {
    expect(isRunLogLocked(file)).toBe(true);
    expect(nodeIds([...iterRunRecords(file)].map((entry) => entry.envelope))).toEqual(["first"]);
    log.append(new NodeStartedRecord("second"));
    expect(log.nextSequence).toBe(2);
  } finally {
    log[Symbol.dispose]();
  }
});

test("iterRunRecords stops at a torn last line and reads it from its offset once it's complete", () => {
  const file = path.join(dir, "run.jsonl");
  const second = wireLine({ seq: 1, nodeId: "next" });
  fs.writeFileSync(file, `${wireLine()}\n${second.slice(0, 20)}`);
  const before = fs.readFileSync(file);

  const read = [...iterRunRecords(file)];
  expect(read.map((entry) => entry.seq)).toEqual([0]);
  expect(fs.readFileSync(file)).toEqual(before);

  fs.appendFileSync(file, `${second.slice(20)}\n`);
  const after = [...iterRunRecords(file, { afterOffset: read[0]!.end })];
  expect(after.map(({ seq, offset, nodeId }) => ({ seq, offset, nodeId }))).toEqual([
    { seq: 1, offset: read[0]!.end, nodeId: "next" },
  ]);
  expect([...iterRunRecords(file, { afterOffset: after[0]!.end })]).toEqual([]);
});

test("iterRunRecords follows a log another process appends to, without blocking it", async () => {
  const file = path.join(dir, "run.jsonl");
  const count = 40;
  const writer = `
import { JsonlRunLog, NodeStartedRecord } from ${JSON.stringify(LOG_URL)};
const log = JsonlRunLog.create(${JSON.stringify(file)}, "run");
for (let i = 0; i < ${count}; i++) {
  log.append(new NodeStartedRecord("n" + i));
  await new Promise((resolve) => setTimeout(resolve, 5));
}
log.close();
`;
  const child = spawn(process.execPath, ["-e", writer], { stdio: ["ignore", "ignore", "pipe"], timeout: 20_000 });
  let stderr = "";
  child.stderr!.on("data", (data) => (stderr += data));
  const exited = new Promise<number | null>((resolve) => child.on("close", (code) => resolve(code)));
  try {
    const seen: string[] = [];
    let offset = 0;
    let readWhileWriting = 0;
    const deadline = Date.now() + 15_000;
    while (seen.length < count && Date.now() < deadline) {
      if (fs.existsSync(file)) {
        for (const entry of iterRunRecords(file, { afterOffset: offset })) {
          seen.push(entry.nodeId!);
          offset = entry.end;
          if (child.exitCode === null) readWhileWriting++;
        }
      }
      await new Promise((resolve) => setTimeout(resolve, 2));
    }
    expect(await exited).toBe(0);
    expect(stderr).toBe("");
    expect(seen).toEqual(Array.from({ length: count }, (_, i) => `n${i}`));
    expect(readWhileWriting).toBeGreaterThan(0);
    expect(offset).toBe(fs.statSync(file).size);
  } finally {
    if (child.exitCode === null) child.kill("SIGKILL");
  }
}, 30_000);

test("iterRunRecords refuses an offset past the end of a log that shrank", () => {
  const file = path.join(dir, "run.jsonl");
  const first = wireLine();
  fs.writeFileSync(file, `${first}\n${wireLine({ seq: 1 })}\n`);
  const end = [...iterRunRecords(file)].at(-1)!.end;
  fs.truncateSync(file, byteLength(first) + 1);

  const message = `run log ${file} shrank to ${byteLength(first) + 1} bytes under its reader at byte ${end}; it was truncated or replaced`;
  const error = expectError(() => [...iterRunRecords(file, { afterOffset: end })], RunLogChangedError, message);
  expect(error.name).toBe("RunLogChangedError");
  expect(error).toBeInstanceOf(RangeError);
});

test("iterRunRecords says a log shrank when it shrinks before the end-of-line probe", () => {
  const file = path.join(dir, "run.jsonl");
  const first = wireLine();
  fs.writeFileSync(file, `${first}\n${wireLine({ seq: 1 })}\n`);
  const end = byteLength(first) + 1;
  const realFstat = fs.fstatSync;
  let cut = false;
  // The size check sees the whole file; the probe after it finds the file cut to nothing.
  const fstatSpy = spyOn(fs, "fstatSync").mockImplementation(((...args: any[]) => {
    const stats = Reflect.apply(realFstat, fs, args) as fs.Stats;
    if (!cut) fs.truncateSync(file, 0);
    cut = true;
    return stats;
  }) as typeof fs.fstatSync);
  try {
    const message = `run log ${file} shrank to 0 bytes under its reader at byte ${end}; it was truncated or replaced`;
    expectError(() => [...iterRunRecords(file, { afterOffset: end })], RunLogChangedError, message);
  } finally {
    fstatSpy.mockRestore();
  }
});

test("iterRunRecords refuses another run's log renamed over the path when given the run id and seq read before", () => {
  const file = path.join(dir, "run.jsonl");
  const lines = (runId: string) => [0, 1, 2].map((seq) => wireLine({ runId, seq, nodeId: `n${seq}` }));
  fs.writeFileSync(file, `${lines("run-a").slice(0, 2).join("\n")}\n`);
  const last = [...iterRunRecords(file)].at(-1)!;
  const saved = { afterOffset: last.end, runId: last.runId, afterSeq: last.seq };
  expect(saved).toEqual({ afterOffset: last.end, runId: "run-a", afterSeq: 1 });

  // Another run's log, line for line the same length, replaces the file.
  const other = path.join(dir, "other.jsonl");
  fs.writeFileSync(other, `${lines("run-b").join("\n")}\n`);
  fs.renameSync(other, file);
  // Without the run id and seq, the offset alone reads on into the other run.
  expect([...iterRunRecords(file, { afterOffset: saved.afterOffset })].map((entry) => entry.runId)).toEqual(["run-b"]);
  const changed = `run log ${file} doesn't continue the records read before byte ${saved.afterOffset}`;
  expectError(
    () => [...iterRunRecords(file, saved)],
    RunLogChangedError,
    `${changed}: expected seq 2 of run 'run-a', found seq 2 of run 'run-b'; it was truncated or replaced`,
  );

  // The same run, continued, reads on.
  fs.writeFileSync(file, `${lines("run-a").join("\n")}\n`);
  expect([...iterRunRecords(file, saved)].map((entry) => entry.nodeId)).toEqual(["n2"]);
  expectError(
    () => [...iterRunRecords(file, { ...saved, afterSeq: 0 })],
    RunLogChangedError,
    `${changed}: expected seq 1 of run 'run-a', found seq 2 of run 'run-a'; it was truncated or replaced`,
  );
  expectError(
    () => [...iterRunRecords(file, { afterOffset: saved.afterOffset, runId: "run-b" })],
    RunLogChangedError,
    `${changed}: expected run 'run-b', found seq 2 of run 'run-a'; it was truncated or replaced`,
  );

  // A line end that moved is a change too, not a bad offset.
  const shifted = [wireLine({ runId: "run-a", nodeId: "n0x" }), wireLine({ runId: "run-a", seq: 1, nodeId: "n1" })];
  fs.writeFileSync(file, `${shifted.join("\n")}\n`);
  expectError(
    () => [...iterRunRecords(file, saved)],
    RunLogChangedError,
    `${changed}: it isn't the end of a line; it was truncated or replaced`,
  );
});

test.each<[string, Record<string, unknown>, string]>([
  ["afterSeq without an offset", { afterSeq: 0 }, "runId and afterSeq describe the record before afterOffset, so they need one"],
  ["runId without an offset", { runId: "run" }, "runId and afterSeq describe the record before afterOffset, so they need one"],
  ["a negative afterSeq", { afterOffset: 1, afterSeq: -1 }, "afterSeq must be a non-negative integer, got -1"],
  ["an empty runId", { afterOffset: 1, runId: "" }, "runId must be a non-empty string, got ''"],
])("iterRunRecords refuses %s", (_name, options, message) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${wireLine()}\n`);
  expectError(() => [...iterRunRecords(file, options)], RangeError, message);
});

test("iterRunRecords throws a bad option or a missing file on the first next(), not at the call", () => {
  const reader = iterRunRecords(path.join(dir, "missing.jsonl"), { afterOffset: -1 });
  expectError(() => reader.next(), RangeError, "afterOffset must be a non-negative integer, got -1");
  const missing = iterRunRecords(path.join(dir, "missing.jsonl"));
  let caught: NodeJS.ErrnoException | undefined;
  try {
    missing.next();
  } catch (error) {
    caught = error as NodeJS.ErrnoException;
  }
  expect(caught?.code).toBe("ENOENT");
});

test("iterRunRecords detects a log that shrinks while it reads", () => {
  const file = path.join(dir, "run.jsonl");
  const first = wireLine();
  fs.writeFileSync(file, `${first}\n${wireLine({ seq: 1 })}\n`);
  const reader = iterRunRecords(file);
  expect(reader.next().value!.seq).toBe(0);
  fs.truncateSync(file, 0);
  // The second line was read before the truncation.
  const second = reader.next().value!;
  expect(second.seq).toBe(1);
  const message = `run log ${file} shrank to 0 bytes under its reader at byte ${second.end}; it was truncated or replaced`;
  expectError(() => reader.next(), RunLogChangedError, message);
});

test.each<[string, number, string]>([
  ["inside a line", 5, "afterOffset 5 isn't the end of a line in run log {file}"],
  ["negative", -1, "afterOffset must be a non-negative integer, got -1"],
  ["fractional", 1.5, "afterOffset must be a non-negative integer, got 1.5"],
])("iterRunRecords refuses an offset %s", (_name, afterOffset, message) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${wireLine()}\n`);
  expectError(() => [...iterRunRecords(file, { afterOffset })], RangeError, message.replace("{file}", file));
});

test("iterRunRecords checks seq against the line from the start, and after an offset from the first record", () => {
  const file = path.join(dir, "run.jsonl");
  const lines = [wireLine(), wireLine({ seq: 1 }), wireLine({ seq: 3 })];
  fs.writeFileSync(file, `${lines.join("\n")}\n`);
  const mismatch = "run-log sequence mismatch: expected 2, found 3";
  expectError(() => [...iterRunRecords(file)], RangeError, `run log ${file} line 3 (seq 2): ${mismatch}`);

  const offset = byteLength(lines[0]!) + 1;
  const third = offset + byteLength(lines[1]!) + 1;
  expectError(
    () => [...iterRunRecords(file, { afterOffset: offset })],
    RangeError,
    `run log ${file} byte ${third} (seq 2): ${mismatch}`,
  );

  fs.writeFileSync(file, `${lines[0]}\n${wireLine({ seq: 1, runId: "foreign" })}\n`);
  expectError(
    () => [...iterRunRecords(file, { payloads: false })],
    RangeError,
    `run log ${file} line 2 (seq 1): run-log run_id mismatch: expected 'run', found 'foreign'`,
  );
});

test("iterRunRecords with payloads: false reads the envelope and node id without validating payloads", () => {
  const file = path.join(dir, "run.jsonl");
  const lines = [
    kindLine("node_started", '{"node_id":"work","extra":1}'),
    kindLine("node_failed", '{"failure":{"node_id":"boom"}}', { seq: 1 }),
    kindLine("node_expanded", '{"spawner_id":"fan"}', { seq: 2 }),
    kindLine("run_stopped", "{}", { seq: 3 }),
  ];
  fs.writeFileSync(file, `${lines.join("\n")}\n`);

  const entries = [...iterRunRecords(file, { payloads: false })];
  expect(entries.every((entry) => entry.at === AT)).toBe(true);
  expect(entries.map(({ seq, kind, nodeId, envelope }) => ({ seq, kind, nodeId, envelope }))).toEqual([
    { seq: 0, kind: "node_started", nodeId: "work", envelope: null },
    { seq: 1, kind: "node_failed", nodeId: "boom", envelope: null },
    { seq: 2, kind: "node_expanded", nodeId: "fan", envelope: null },
    { seq: 3, kind: "run_stopped", nodeId: null, envelope: null },
  ]);
  const error = expectError(
    () => [...iterRunRecords(file)],
    RangeError,
    `run log ${file} line 1 (seq 0): invalid node_started payload fields; missing=[], extra=["extra"]`,
  );
  expect((error.cause as Error).constructor).toBe(RangeError);
});

test.each<[string, string, new (...args: never[]) => Error, string]>([
  ["an empty node id", kindLine("node_started", '{"node_id":""}'), RangeError, "node_started payload node_id must be a non-empty string"],
  ["a missing node id", kindLine("node_step", "{}"), TypeError, "node_step payload node_id must be a non-empty string"],
  ["an unknown kind", kindLine("node_paused", "{}"), RangeError, "run-record kind 'node_paused' is unsupported in schema 18"],
  ["invalid JSON", "{", RangeError, "invalid run-record JSON"],
])("iterRunRecords with payloads: false still refuses %s", (_name, line, causeClass, message) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${line}\n`);
  const error = expectError(
    () => [...iterRunRecords(file, { payloads: false })],
    RangeError,
    `run log ${file} line 1 (seq 0): ${message}`,
  );
  expect((error.cause as Error).constructor).toBe(causeClass);
});

test("iterRunRecords yields only the kinds asked for, still checking every line's envelope and seq", () => {
  const file = path.join(dir, "run.jsonl");
  const lines = [
    kindLine("node_started", '{"node_id":"work","extra":1}'),
    kindLine("run_stopped", '{"reason":"signal","message":"stop","node_ids":["work"]}', { seq: 1 }),
    kindLine("node_started", '{"node_id":"next"}', { seq: 3 }),
  ];
  fs.writeFileSync(file, `${lines.slice(0, 2).join("\n")}\n`);
  const stops = [...iterRunRecords(file, { kinds: ["run_stopped"] })];
  expect(stops.map((entry) => [entry.seq, entry.kind])).toEqual([[1, "run_stopped"]]);
  expect((stops[0]!.envelope.record as RunStoppedRecord).nodeIds).toEqual(["work"]);

  fs.appendFileSync(file, `${lines[2]}\n`);
  expectError(
    () => [...iterRunRecords(file, { kinds: ["run_stopped"] })],
    RangeError,
    `run log ${file} line 3 (seq 2): run-log sequence mismatch: expected 2, found 3`,
  );
  expectError(
    () => [...iterRunRecords(file, { kinds: ["run_stopped", "node_paused"] })],
    RangeError,
    `unknown run-record kinds: ["node_paused"]`,
  );
});

test.each<[string, string, number]>([
  ["9", PYTHON_SCHEMA, 9],
  ["10", retired(10), 10],
  ["11", retired(11), 11],
  ["12", retired(12), 12],
  ["13", retired(13), 13],
  ["14", retired(14), 14],
  ["15", retired(15), 15],
  ["16", retired(16), 16],
  ["17", RETIRED_SCHEMA, 17],
  ["19", "unsupported run-record schema: 19", 19],
])("iterRunRecords refuses schema %s with a RunLogSchemaError naming the line", (schema, message, value) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${wireLine({ schema })}\n`);
  for (const payloads of [true, false]) {
    const error = expectError(
      () => [...iterRunRecords(file, { payloads })],
      RunLogSchemaError,
      `run log ${file} line 1 (seq 0): ${message}`,
    ) as RunLogSchemaError;
    expect(error.name).toBe("RunLogSchemaError");
    expect(error.schema).toBe(value);
    expect(error).toBeInstanceOf(RangeError);
  }
});

test("iterRunRecords reads in bounded chunks, and reads only as far as it is asked", () => {
  const file = path.join(dir, "run.jsonl");
  const big = "x".repeat(1_000_000);
  const lines = [
    wireLine(),
    kindLine("run_stopped", `{"reason":"signal","message":"${big}","node_ids":[]}`, { seq: 1 }),
    ...Array.from({ length: 3000 }, (_, i) => wireLine({ seq: i + 2, nodeId: `n${i}` })),
  ];
  fs.writeFileSync(file, `${lines.join("\n")}\n`);
  const reads: number[] = [];
  const realRead = fs.readSync;
  const readSpy = spyOn(fs, "readSync").mockImplementation(((fd: number, buffer: Uint8Array, ...rest: any[]) => {
    const count = Reflect.apply(realRead, fs, [fd, buffer, ...rest]) as number;
    if (isDescriptorOf(fd, file)) reads.push(rest[1] as number);
    return count;
  }) as typeof fs.readSync);
  const wholeFile = spyOn(fs, "readFileSync");
  try {
    const reader = iterRunRecords(file);
    expect(reader.next().value!.seq).toBe(0);
    expect(reads.length).toBe(1);
    reader.return(undefined);

    let last = -1;
    for (const entry of iterRunRecords(file)) last = entry.seq;
    expect(last).toBe(3001);
    expect(Math.max(...reads)).toBeLessThanOrEqual(1 << 16);
    expect(wholeFile).not.toHaveBeenCalled();
  } finally {
    readSpy.mockRestore();
    wholeFile.mockRestore();
  }
});

const LOG_URL = new URL("../../src/runtime/log.ts", import.meta.url).href;

/** The message `create` and `resume` throw when another writer holds the log at `file`. */
function busyMessage(file: string): string {
  return `run log ${file} is locked by another writer (is the run still active?)`;
}

test("resuming a missing log throws ENOENT and creates no lock file", () => {
  const file = path.join(dir, "missing.jsonl");
  let caught: NodeJS.ErrnoException | undefined;
  try {
    JsonlRunLog.resume(file);
  } catch (error) {
    caught = error as NodeJS.ErrnoException;
  }
  expect(caught?.code).toBe("ENOENT");
  expect(caught?.path).toBe(file);
  expect(fs.readdirSync(dir)).toEqual([]);
});

test("jsonl create is exclusive and preserves an existing file", () => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, "already here\n");
  let caught: NodeJS.ErrnoException | undefined;
  try {
    JsonlRunLog.create(file, "run");
  } catch (error) {
    caught = error as NodeJS.ErrnoException;
  }
  expect(caught?.code).toBe("EEXIST");
  expect(fs.readFileSync(file, "utf8")).toBe("already here\n");
  expect(fs.existsSync(`${file}.lock`)).toBe(false);
});

test("jsonl create sets owner read-write permissions", () => {
  const file = path.join(dir, "run.jsonl");
  {
    const log = JsonlRunLog.create(file, "run");
    try {
      expect(log.path).toBe(file);
      expect(log.runId).toBe("run");
      expect(log.nextSequence).toBe(0);
    } finally {
      log[Symbol.dispose]();
    }
  }
  expect(fs.statSync(file).mode & 0o777).toBe(0o600);
});

test("jsonl create refuses a lone-surrogate run id before any file exists", () => {
  const file = path.join(dir, "run.jsonl");
  expectError(() => JsonlRunLog.create(file, "\ud800"), RangeError, "Unicode surrogate is not valid canonical JSON");
  expect(fs.readdirSync(dir)).toEqual([]);
});

test("jsonl resume locks before its first read", () => {
  const file = path.join(dir, "run.jsonl");
  {
    const created = JsonlRunLog.create(file, "run");
    try {
      created.append(new NodeStartedRecord("work"));
    } finally {
      created[Symbol.dispose]();
    }
  }
  const real = fs.readFileSync;
  const lockedAtRead: boolean[] = [];
  const spy = spyOn(fs, "readFileSync").mockImplementation(((target: unknown, ...rest: unknown[]) => {
    if (isDescriptorOf(target, file)) lockedAtRead.push(isLocked(`${file}.lock`));
    return Reflect.apply(real, fs, [target, ...rest]);
  }) as typeof fs.readFileSync);
  try {
    const resumed = JsonlRunLog.resume(file);
    try {
      expect(nodeIds(resumed)).toEqual(["work"]);
    } finally {
      resumed[Symbol.dispose]();
    }
  } finally {
    spy.mockRestore();
  }
  expect(lockedAtRead).toEqual([true]);
});

test("jsonl excludes a second writer", () => {
  const file = path.join(dir, "run.jsonl");
  const first = baselineLog(file);
  try {
    expectError(() => JsonlRunLog.resume(file), Error, busyMessage(file));
  } finally {
    first.close();
  }
  const reopened = JsonlRunLog.resume(file);
  try {
    expect(reopened.runId).toBe("run");
  } finally {
    reopened[Symbol.dispose]();
  }
});

test("jsonl lock contention names the log and the likely cause", () => {
  const file = path.join(dir, "run.jsonl");
  const first = baselineLog(file);
  let error: Error;
  try {
    error = expectError(() => JsonlRunLog.resume(file), Error, busyMessage(file));
  } finally {
    first.close();
  }
  expect(error.cause).toBeInstanceOf(FileLockError);
  expect((error.cause as FileLockError).reason).toBe("busy");
  expectUnlocked(file);
});

test("jsonl excludes a writer in a separate process", () => {
  const file = path.join(dir, "run.jsonl");
  const first = baselineLog(file);
  const child = `
import { JsonlRunLog } from ${JSON.stringify(LOG_URL)};
const file = ${JSON.stringify(file)};
try {
  JsonlRunLog.resume(file).close();
} catch (error) {
  if (error.message === ${JSON.stringify(busyMessage(file))}) process.exit(0);
  throw error;
}
console.error("child unexpectedly acquired the writer lock");
process.exit(1);
`;
  let result: ReturnType<typeof spawnSync>;
  try {
    result = spawnSync(process.execPath, ["-e", child], { encoding: "utf8", timeout: 10_000 });
  } finally {
    first.close();
  }
  expect(result.stderr).toBe("");
  expect(result.status).toBe(0);
});

test.each([
  ["partial JSON", '{"partial":'],
  [
    "valid JSON without a newline",
    encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "run", 1, new NodeStartedRecord("valid-but-incomplete"))),
  ],
])("jsonl resume truncates and syncs every incomplete tail: %s", (_name, tail) => {
  const file = path.join(dir, "run.jsonl");
  baselineLog(file).close();
  const completePrefix = fs.readFileSync(file, "utf8");
  fs.appendFileSync(file, tail);

  const realTruncate = fs.ftruncateSync;
  const realFsync = fs.fsyncSync;
  let truncates = 0;
  let fsyncs = 0;
  const truncateSpy = spyOn(fs, "ftruncateSync").mockImplementation(((fd: unknown, ...rest: unknown[]) => {
    if (isDescriptorOf(fd, file)) truncates++;
    return Reflect.apply(realTruncate, fs, [fd, ...rest]);
  }) as typeof fs.ftruncateSync);
  const fsyncSpy = spyOn(fs, "fsyncSync").mockImplementation(((fd: unknown) => {
    if (isDescriptorOf(fd, file)) fsyncs++;
    return Reflect.apply(realFsync, fs, [fd]);
  }) as typeof fs.fsyncSync);
  try {
    const resumed = JsonlRunLog.resume(file);
    try {
      expect(nodeIds(resumed)).toEqual(["first"]);
      expect(resumed.nextSequence).toBe(1);
    } finally {
      resumed[Symbol.dispose]();
    }
  } finally {
    truncateSpy.mockRestore();
    fsyncSpy.mockRestore();
  }
  expect(fs.readFileSync(file, "utf8")).toBe(completePrefix);
  expect(truncates).toBe(1);
  expect(fsyncs).toBe(1);
});

test("jsonl resume rejects a corrupt complete line and releases the lock", () => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${wireLine()}\n{not-json}\n`);
  const error = expectError(
    () => JsonlRunLog.resume(file),
    RangeError,
    `run log ${file} line 2 (seq 1): invalid run-record JSON`,
  );
  expect(error.cause).toBeInstanceOf(RangeError);
  expectUnlocked(file);
});

test.each<[string, string, string]>([
  ["json", "{not-json}", "invalid run-record JSON"],
  ["blank", "", "invalid run-record JSON"],
  ["payload", wireLine({ seq: 2, nodeId: "" }), "node_started payload node_id must be a non-empty string"],
  [
    "kind",
    wireLine({ seq: 2 }).replace("node_started", "xnode_started"),
    "run-record kind 'xnode_started' is unsupported in schema 18",
  ],
  ["sequence", wireLine({ seq: 5 }), "run-log sequence mismatch: expected 2, found 5"],
  ["run id", wireLine({ seq: 2, runId: "other" }), "run-log run_id mismatch: expected 'run', found 'other'"],
])("jsonl resume errors name the log, line and seq: %s", (_name, damaged, message) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${wireLine({ seq: 0 })}\n${wireLine({ seq: 1 })}\n${damaged}\n`);
  expectError(() => JsonlRunLog.resume(file), RangeError, `run log ${file} line 3 (seq 2): ${message}`);
  expectUnlocked(file);
});

test("jsonl resume of an empty file names the log and releases the lock", () => {
  const file = path.join(dir, "empty.jsonl");
  fs.writeFileSync(file, "");
  expectError(() => JsonlRunLog.resume(file), RangeError, `cannot resume an empty run log: ${file}`);
  expect(fs.readFileSync(file, "utf8")).toBe("");
  expectUnlocked(file);
});

test.each<[string, string, string, new (...args: never[]) => Error]>([
  ["schema 1", `${wireLine({ schema: "1" })}\n`, "line 1 (seq 0): unsupported run-record schema: 1", RunLogSchemaError],
  ["schema 2", `${wireLine({ schema: "2" })}\n`, "line 1 (seq 0): unsupported run-record schema: 2", RunLogSchemaError],
  ["sequence", `${wireLine({ seq: 1 })}\n`, "line 1 (seq 0): run-log sequence mismatch: expected 0, found 1", RangeError],
  [
    "run id",
    `${wireLine({ runId: "first" })}\n${wireLine({ runId: "second", seq: 1, nodeId: "other" })}\n`,
    "line 2 (seq 1): run-log run_id mismatch: expected 'first', found 'second'",
    RangeError,
  ],
])("jsonl resume rejects identity mismatches and releases the lock: %s", (_name, contents, where, cls) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, contents);
  expectError(() => JsonlRunLog.resume(file), cls, `run log ${file} ${where}`);
  expectUnlocked(file);
});

test.each<[string, string, string | null]>([
  ["not a log", "id: echo\nname: echo", "line 1 (seq 0): invalid run-record JSON"],
  ["no complete line", '{"a": 1}', null],
  ["retired schema", `${wireLine({ schema: "2" })}\n{"partial":`, "line 1 (seq 0): unsupported run-record schema: 2"],
  ["corrupt line", `${wireLine()}\n{not-json}\n{"partial":`, "line 2 (seq 1): invalid run-record JSON"],
])("jsonl resume leaves a rejected file unchanged: %s", (name, contents, where) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, contents);
  const message = where === null ? `cannot resume an empty run log: ${file}` : `run log ${file} ${where}`;
  expectError(() => JsonlRunLog.resume(file), name === "retired schema" ? RunLogSchemaError : RangeError, message);
  expect(fs.readFileSync(file, "utf8")).toBe(contents);
  expectUnlocked(file);
});

test("jsonl create releases its descriptor when adoption fails: fchmod", () => {
  const file = path.join(dir, "run.jsonl");
  const realFchmod = fs.fchmodSync;
  const realClose = fs.closeSync;
  const failed: number[] = [];
  const closed: number[] = [];
  const fchmodSpy = spyOn(fs, "fchmodSync").mockImplementation(((fd: unknown, ...rest: unknown[]) => {
    if (isDescriptorOf(fd, file)) {
      failed.push(fd as number);
      throw new Error("fchmod failed");
    }
    return Reflect.apply(realFchmod, fs, [fd, ...rest]);
  }) as typeof fs.fchmodSync);
  const closeSpy = spyOn(fs, "closeSync").mockImplementation(((fd: unknown) => {
    if (isDescriptorOf(fd, file)) closed.push(fd as number);
    return Reflect.apply(realClose, fs, [fd]);
  }) as typeof fs.closeSync);
  try {
    expectError(() => JsonlRunLog.create(file, "run"), Error, "fchmod failed");
  } finally {
    fchmodSpy.mockRestore();
    closeSpy.mockRestore();
  }
  expect(failed).toHaveLength(1);
  expect(closed).toEqual(failed);
  expect(fs.readFileSync(file, "utf8")).toBe("");
  expectUnlocked(file);
});

test("jsonl create releases its descriptor when adoption fails: a busy lock", () => {
  const file = path.join(dir, "run.jsonl");
  const realClose = fs.closeSync;
  const closed: number[] = [];
  const holder = lockFile(`${file}.lock`);
  const closeSpy = spyOn(fs, "closeSync").mockImplementation(((fd: unknown) => {
    if (isDescriptorOf(fd, file)) closed.push(fd as number);
    return Reflect.apply(realClose, fs, [fd]);
  }) as typeof fs.closeSync);
  let error: Error;
  try {
    error = expectError(() => JsonlRunLog.create(file, "run"), Error, busyMessage(file));
  } finally {
    closeSpy.mockRestore();
    holder.release();
  }
  expect(error.cause).toBeInstanceOf(FileLockError);
  expect((error.cause as FileLockError).reason).toBe("busy");
  expect(closed).toHaveLength(1);
  expect(fs.readFileSync(file, "utf8")).toBe("");
  expectUnlocked(file);
});

test("jsonl close is idempotent, keeps its records and releases the lock", () => {
  const file = path.join(dir, "run.jsonl");
  const log = baselineLog(file);
  const records = [...log];

  log.close();
  log.close();

  expect([...log]).toEqual(records);
  expect(log.records()).toEqual(records);
  expectError(() => log.append(new NodeStartedRecord("late")), Error, "run log is closed");
  const reopened = JsonlRunLog.resume(file);
  try {
    expect([...reopened]).toEqual(records);
  } finally {
    reopened[Symbol.dispose]();
  }
});

test("disposing a jsonl log releases the lock when the body throws", () => {
  const file = path.join(dir, "run.jsonl");
  expectError(
    () => {
      const log = JsonlRunLog.create(file, "run");
      try {
        log.append(new NodeStartedRecord("work"));
        throw new Error("body failed");
      } finally {
        log[Symbol.dispose]();
      }
    },
    Error,
    "body failed",
  );
  const reopened = JsonlRunLog.resume(file);
  try {
    expect(reopened.nextSequence).toBe(1);
  } finally {
    reopened[Symbol.dispose]();
  }
});

test("isRunLogLocked reports whether a writer holds the log", () => {
  const file = path.join(dir, "run.jsonl");
  expect(isRunLogLocked(file)).toBe(false);
  const log = baselineLog(file);
  try {
    expect(isRunLogLocked(file)).toBe(true);
  } finally {
    log.close();
  }
  expect(isRunLogLocked(file)).toBe(false);
  expect(fs.existsSync(`${file}.lock`)).toBe(true);
});

test("a symlink to a log shares the real log's lock", () => {
  const real = path.join(dir, "real.jsonl");
  const link = path.join(dir, "link.jsonl");
  const first = baselineLog(real);
  fs.symlinkSync(real, link);
  try {
    expectError(() => JsonlRunLog.resume(link), Error, busyMessage(link));
    expect(isRunLogLocked(link)).toBe(true);
  } finally {
    first.close();
  }
  expect(isRunLogLocked(link)).toBe(false);
  {
    const resumed = JsonlRunLog.resume(link);
    try {
      expect(resumed.path).toBe(link);
      expect(isRunLogLocked(real)).toBe(true);
      resumed.append(new NodeStartedRecord("second"));
    } finally {
      resumed[Symbol.dispose]();
    }
  }
  expect(fs.lstatSync(link).isSymbolicLink()).toBe(true);
  expect(nodeIds(readRunLog(real))).toEqual(["first", "second"]);
});

test("a failed resume whose descriptor close also fails throws the original error", () => {
  const file = path.join(dir, "empty.jsonl");
  fs.writeFileSync(file, "");
  const spy = spyOnLog("closeSync", file, (real, args) => {
    real(...args);
    throw new Error("close failed");
  });
  try {
    expectError(() => JsonlRunLog.resume(file), RangeError, `cannot resume an empty run log: ${file}`);
  } finally {
    spy.mockRestore();
  }
  expectUnlocked(file);
});

test("jsonl append uses one complete write and one fsync", () => {
  const file = path.join(dir, "run.jsonl");
  const log = JsonlRunLog.create(file, "run", { clock: CLOCK });
  const writes: string[] = [];
  let fsyncs = 0;
  const writeSpy = spyOnLog("writeSync", file, (real, args) => {
    const [, buffer, offset, length] = args as [number, Uint8Array, number, number];
    writes.push(Buffer.from(buffer.subarray(offset, offset + length)).toString("utf8"));
    return real(...args);
  });
  const fsyncSpy = spyOnLog("fsyncSync", file, (real, args) => {
    fsyncs++;
    return real(...args);
  });
  let envelope: RecordEnvelope;
  try {
    envelope = log.append(new NodeStartedRecord("work"));
  } finally {
    writeSpy.mockRestore();
    fsyncSpy.mockRestore();
    log.close();
  }
  expect(envelope.seq).toBe(0);
  expect(writes).toEqual([`${wireLine()}\n`]);
  expect(fsyncs).toBe(1);
});

test("jsonl append and cache are isolated from caller mutation", () => {
  const file = path.join(dir, "run.jsonl");
  const seed = { items: [1n] };
  {
    const log = JsonlRunLog.create(file, "run");
    try {
      const envelope = log.append(nestedStep(seed.items));
      seed.items.push(2n);
      expect(seedOf(envelope)).toEqual({ items: [1n] });
      expect(seedOf([...log][0]!)).toEqual({ items: [1n] });
    } finally {
      log[Symbol.dispose]();
    }
  }
  const resumed = JsonlRunLog.resume(file);
  try {
    expect(seedOf([...resumed][0]!)).toEqual({ items: [1n] });
  } finally {
    resumed[Symbol.dispose]();
  }
});

test("a jsonl log never exposes its retained mutable payload", () => {
  const file = path.join(dir, "run.jsonl");
  const log = JsonlRunLog.create(file, "run");
  const appended = log.append(nestedStep());
  const durable = fs.readFileSync(file, "utf8");

  expect(() => seedOf(appended).items.push({ from: "append" })).toThrow(TypeError);
  const iterated = log[Symbol.iterator]().next().value!;
  expect(() => {
    seedOf(iterated).items[0].value = 99n;
  }).toThrow(TypeError);
  expect(() => seedOf(log.records()[0]!).items.push({ from: "records" })).toThrow(TypeError);

  expect(seedOf(log[Symbol.iterator]().next().value!)).toEqual({ items: [{ value: 1n }] });
  expect(seedOf(log.records()[0]!)).toEqual({ items: [{ value: 1n }] });
  expect(fs.readFileSync(file, "utf8")).toBe(durable);
  log.close();

  const resumed = JsonlRunLog.resume(file);
  try {
    expect(seedOf(resumed[Symbol.iterator]().next().value!)).toEqual({ items: [{ value: 1n }] });
    expect(fs.readFileSync(file, "utf8")).toBe(durable);
  } finally {
    resumed[Symbol.dispose]();
  }
});

test("an append rollback preserves the log", () => {
  const file = path.join(dir, "rollback.jsonl");
  const prefix = `${wireLine()}\n`;
  fs.writeFileSync(file, prefix);
  const log = JsonlRunLog.resume(file);
  try {
    log.append(new NodeStartedRecord("second"));
    const failure = new Error("fsync failed");
    const spy = spyOnLog("fsyncSync", file, () => {
      throw failure;
    });
    try {
      const error = expectAppendFailure(log, file, () => log.append(new NodeStartedRecord("third"))) as AggregateError;
      expect(error.constructor).toBe(AggregateError);
      expect(error.message).toBe(`run log ${file} append failed and its rollback also failed`);
      expect(error.errors).toHaveLength(2);
      expect(error.errors[0]).toBe(failure);
      expect(error.errors[1]).toBe(failure);
      expectPoisoned(log, file, failure);
    } finally {
      spy.mockRestore();
    }
  } finally {
    log.close();
  }
  expect(fs.readFileSync(file, "utf8").startsWith(prefix)).toBe(true);
  const resumed = JsonlRunLog.resume(file);
  try {
    expect(nodeIds(resumed)).toEqual(["work", "second"]);
    expect(resumed.append(new NodeStartedRecord("third")).seq).toBe(2);
  } finally {
    resumed[Symbol.dispose]();
  }
});

test("an invalid append retains the prior complete prefix and doesn't poison the log", () => {
  const file = path.join(dir, "run.jsonl");
  const log = baselineLog(file);
  try {
    const error = expectAppendFailure(log, file, () => log.append({ nodeId: "x" } as never));
    expect((error as Error).constructor).toBe(TypeError);
    expect((error as Error).message).toBe("unsupported run record: Object");
    expect(log.append(new NodeStartedRecord("second")).seq).toBe(1);
  } finally {
    log[Symbol.dispose]();
  }
});

test("a write failure rolls back to the prior complete prefix", () => {
  const file = path.join(dir, "run.jsonl");
  const log = baselineLog(file);
  try {
    const failure = new Error("write failed");
    const spy = spyOnLog("writeSync", file, (real, args) => {
      const [fd, buffer, offset, length, position] = args;
      real(fd, buffer, offset, Math.max(1, Math.floor(length / 2)), position);
      throw failure;
    });
    try {
      expect(expectAppendFailure(log, file, () => log.append(new NodeStartedRecord("second")))).toBe(failure);
      expectPoisoned(log, file, failure);
    } finally {
      spy.mockRestore();
    }
  } finally {
    log[Symbol.dispose]();
  }
});

test("an fsync failure rolls back and syncs the prior complete prefix", () => {
  const file = path.join(dir, "run.jsonl");
  const log = baselineLog(file);
  try {
    const failure = new Error("fsync failed");
    let calls = 0;
    const spy = spyOnLog("fsyncSync", file, (real, args) => {
      calls++;
      if (calls === 1) throw failure;
      return real(...args);
    });
    try {
      expect(expectAppendFailure(log, file, () => log.append(new NodeStartedRecord("second")))).toBe(failure);
      expectPoisoned(log, file, failure);
    } finally {
      spy.mockRestore();
    }
    expect(calls).toBe(2);
  } finally {
    log[Symbol.dispose]();
  }
});

test("a short write is completed by writing the rest", () => {
  const file = path.join(dir, "run.jsonl");
  const log = baselineLog(file);
  try {
    const prefix = fs.readFileSync(file, "utf8");
    let calls = 0;
    const spy = spyOnLog("writeSync", file, (real, args) => {
      calls++;
      const [fd, buffer, offset, length, position] = args;
      return real(fd, buffer, offset, calls === 1 ? length - 1 : length, position);
    });
    try {
      expect(log.append(new NodeStartedRecord("second")).seq).toBe(1);
    } finally {
      spy.mockRestore();
    }
    expect(calls).toBe(2);
    expect(log.nextSequence).toBe(2);
    expect(fs.readFileSync(file, "utf8")).toBe(`${prefix}${wireLine({ seq: 1, nodeId: "second" })}\n`);
  } finally {
    log[Symbol.dispose]();
  }
});

test("a write of 0 bytes after a partial write is an append failure and rolls back", () => {
  const file = path.join(dir, "run.jsonl");
  const log = baselineLog(file);
  try {
    const total = wireLine({ seq: 1, nodeId: "second" }).length + 1;
    const half = Math.floor(total / 2);
    let calls = 0;
    const spy = spyOnLog("writeSync", file, (real, args) => {
      calls++;
      const [fd, buffer, offset, , position] = args;
      return calls === 1 ? real(fd, buffer, offset, half, position) : 0;
    });
    try {
      const error = expectAppendFailure(log, file, () => log.append(new NodeStartedRecord("second"))) as Error;
      expect(error.constructor).toBe(Error);
      expect(error.message).toBe(`short run-log write: wrote ${half} of ${total} bytes`);
      expectPoisoned(log, file, error);
    } finally {
      spy.mockRestore();
    }
    expect(calls).toBe(2);
  } finally {
    log[Symbol.dispose]();
  }
});

test("a write failure whose rollback truncate also fails is an AggregateError and poisons the log", () => {
  const file = path.join(dir, "run.jsonl");
  const log = baselineLog(file);
  try {
    const records = log.records();
    const writeFailure = new Error("write failed");
    const truncateFailure = new Error("ftruncate failed");
    const writeSpy = spyOnLog("writeSync", file, (real, args) => {
      const [fd, buffer, offset, length, position] = args;
      real(fd, buffer, offset, Math.floor(length / 2), position);
      throw writeFailure;
    });
    const truncateSpy = spyOnLog("ftruncateSync", file, () => {
      throw truncateFailure;
    });
    let fsyncs = 0;
    const fsyncSpy = spyOnLog("fsyncSync", file, (real, args) => {
      fsyncs++;
      return real(...args);
    });
    try {
      const error = expectError(
        () => log.append(new NodeStartedRecord("second")),
        AggregateError,
        `run log ${file} append failed and its rollback also failed`,
      ) as AggregateError;
      expect(error.errors).toHaveLength(2);
      expect(error.errors[0]).toBe(writeFailure);
      expect(error.errors[1]).toBe(truncateFailure);
      expect(log.records()).toEqual(records);
      expect(log.nextSequence).toBe(1);
      expectPoisoned(log, file, writeFailure);
    } finally {
      writeSpy.mockRestore();
      truncateSpy.mockRestore();
      fsyncSpy.mockRestore();
    }
    // The rollback still syncs after its truncate failed.
    expect(fsyncs).toBe(1);
  } finally {
    log.close();
  }
  expectUnlocked(file);
});

test("a poisoned log that is closed refuses appends as closed", () => {
  const file = path.join(dir, "run.jsonl");
  const log = baselineLog(file);
  try {
    const failure = new Error("fsync failed");
    const spy = spyOnLog("fsyncSync", file, () => {
      throw failure;
    });
    try {
      // The rollback's fsync fails too, so the append throws an AggregateError.
      expect(() => log.append(new NodeStartedRecord("second"))).toThrow(AggregateError);
    } finally {
      spy.mockRestore();
    }
    expectPoisoned(log, file, failure);
  } finally {
    log.close();
  }
  expectError(() => log.append(new NodeStartedRecord("late")), Error, "run log is closed");
  expectUnlocked(file);
});

test("a tail repair failure releases the lock", () => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, `${wireLine()}\npartial`);
  const failure = new Error("repair fsync failed");
  const spy = spyOnLog("fsyncSync", file, () => {
    throw failure;
  });
  let caught: unknown;
  try {
    JsonlRunLog.resume(file);
  } catch (error) {
    caught = error;
  } finally {
    spy.mockRestore();
  }
  expect(caught).toBe(failure);
  expectUnlocked(file);
});

/** The `ts/` directory. */
const ROOT = path.resolve(import.meta.dirname, "../..");
const BUILT = path.join(ROOT, "dist/src/runtime/log.js");

/**
 * The Node binary named by `YIAN_TEST_NODE`, or `undefined` when it's unset. A set
 * value that can't run, a Node older than the `engines` floor, or a missing build
 * fails the whole file, so a typo can't pass as a run with the Node case skipped.
 * A copy of `tests/engine/platform.test.ts`'s: importing a test file would run its tests.
 */
const NODE = ((): string | undefined => {
  const node = process.env.YIAN_TEST_NODE;
  if (!node) return undefined;
  let version: string;
  try {
    version = execFileSync(node, ["--version"], { encoding: "utf8" }).trim();
  } catch (error) {
    throw new Error(`YIAN_TEST_NODE=${node} doesn't run`, { cause: error });
  }
  const [major = 0, minor = 0] = version.replace(/^v/, "").split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    throw new Error(`YIAN_TEST_NODE=${node} is Node ${version}; the Node checks need 22.18 or later`);
  }
  if (!fs.existsSync(BUILT)) throw new Error(`${BUILT} is missing; run bun run build first`);
  return node;
})();

describe.skipIf(!NODE)("the built jsonl log on Node", () => {
  test(
    "rollback and poisoning survive failed writes and syncs",
    () => {
      const url = (file: string) => JSON.stringify(pathToFileURL(path.join(ROOT, file)).href);
      const script = String.raw`
        import assert from "node:assert/strict";
        import { mkdtempSync, readFileSync, rmSync } from "node:fs";
        import { tmpdir } from "node:os";
        import { join } from "node:path";

        // log.js calls the operations on this default export, so replacing them here is seen.
        const fs = (await import("node:fs")).default;
        const { JsonlRunLog, NodeStartedRecord } = await import(${url("dist/src/runtime/log.js")});
        const { isLocked } = await import(${url("dist/src/platform.js")});

        const tmp = mkdtempSync(join(tmpdir(), "yian-log-node-"));
        const isDescriptorOf = (fd, file) => {
          if (typeof fd !== "number") return false;
          const open = fs.fstatSync(fd);
          const named = fs.statSync(file);
          return open.dev === named.dev && open.ino === named.ino;
        };
        // Run fn with fs[name] replaced: a call on a descriptor of file runs onLog(real, args),
        // every other call (the lock's own) goes to the original.
        const patched = (name, file, onLog, fn) => {
          const original = fs[name];
          const real = (...args) => Reflect.apply(original, fs, args);
          fs[name] = (...args) => (isDescriptorOf(args[0], file) ? onLog(real, args) : real(...args));
          try {
            return fn();
          } finally {
            fs[name] = original;
          }
        };
        const caught = (fn) => {
          try {
            fn();
          } catch (error) {
            return error;
          }
          assert.fail("expected a throw");
        };
        const appendSecond = (log) => caught(() => log.append(new NodeStartedRecord("second")));
        const baseline = (name) => {
          const file = join(tmp, name + ".jsonl");
          const log = JsonlRunLog.create(file, "run", { clock: () => ${CLOCK()} });
          log.append(new NodeStartedRecord("first"));
          return { file, log, prefix: readFileSync(file) };
        };
        const expectPoisoned = (log, file, cause) => {
          const error = caught(() => log.append(new NodeStartedRecord("late")));
          assert.equal(error.constructor, Error);
          assert.equal(error.message, "run log " + file + " refuses appends after a failed append");
          assert.equal(error.cause, cause);
          assert.equal(isLocked(file + ".lock"), true);
        };
        const expectRolledBack = (log, file, prefix) => {
          assert.deepEqual(readFileSync(file), prefix);
          assert.equal(log.nextSequence, 1);
          assert.equal(log.records().length, 1);
        };
        // Close the log, check the lock is free, and resume to read nodeIds.
        const expectClean = (log, file, nodeIds) => {
          log.close();
          assert.equal(isLocked(file + ".lock"), false);
          const resumed = JsonlRunLog.resume(file);
          try {
            assert.deepEqual([...resumed].map((envelope) => envelope.record.nodeId), nodeIds);
          } finally {
            resumed.close();
          }
        };

        try {
          {
            const { file, log, prefix } = baseline("fsync");
            const failure = new Error("fsync failed");
            let calls = 0;
            const onLog = (real, args) => {
              calls++;
              if (calls === 1) throw failure;
              return real(...args);
            };
            const error = patched("fsyncSync", file, onLog, () => appendSecond(log));
            assert.equal(error, failure);
            assert.equal(calls, 2);
            expectRolledBack(log, file, prefix);
            expectPoisoned(log, file, failure);
            expectClean(log, file, ["first"]);
            console.log("ok fsync fails once");
          }
          {
            const { file, log, prefix } = baseline("write");
            const failure = new Error("write failed");
            const onLog = (real, [fd, buffer, offset, length, position]) => {
              real(fd, buffer, offset, Math.max(1, Math.floor(length / 2)), position);
              throw failure;
            };
            const error = patched("writeSync", file, onLog, () => appendSecond(log));
            assert.equal(error, failure);
            expectRolledBack(log, file, prefix);
            expectPoisoned(log, file, failure);
            expectClean(log, file, ["first"]);
            console.log("ok write fails halfway");
          }
          {
            const { file, log, prefix } = baseline("short");
            let calls = 0;
            const onLog = (real, [fd, buffer, offset, length, position]) => {
              calls++;
              return real(fd, buffer, offset, calls === 1 ? length - 1 : length, position);
            };
            const envelope = patched("writeSync", file, onLog, () => log.append(new NodeStartedRecord("second")));
            assert.equal(envelope.seq, 1);
            assert.equal(calls, 2);
            const line = ${JSON.stringify(`${wireLine({ seq: 1, nodeId: "second" })}\n`)};
            assert.equal(readFileSync(file, "utf8"), prefix.toString("utf8") + line);
            expectClean(log, file, ["first", "second"]);
            console.log("ok short write completes");
          }
          {
            const { file, log, prefix } = baseline("truncate");
            const writeFailure = new Error("write failed");
            const truncateFailure = new Error("ftruncate failed");
            const onWrite = (real, [fd, buffer, offset, length, position]) => {
              real(fd, buffer, offset, Math.floor(length / 2), position);
              throw writeFailure;
            };
            const onTruncate = () => {
              throw truncateFailure;
            };
            const error = patched("writeSync", file, onWrite, () =>
              patched("ftruncateSync", file, onTruncate, () => appendSecond(log)),
            );
            assert.equal(error.constructor, AggregateError);
            assert.equal(error.message, "run log " + file + " append failed and its rollback also failed");
            assert.equal(error.errors.length, 2);
            assert.equal(error.errors[0], writeFailure);
            assert.equal(error.errors[1], truncateFailure);
            // The truncate failed, so the half line stays until a resume repairs it.
            const data = readFileSync(file);
            assert.ok(data.length > prefix.length);
            assert.deepEqual(data.subarray(0, prefix.length), prefix);
            assert.equal(log.nextSequence, 1);
            assert.equal(log.records().length, 1);
            expectPoisoned(log, file, writeFailure);
            expectClean(log, file, ["first"]);
            assert.deepEqual(readFileSync(file), prefix);
            console.log("ok rollback truncate fails");
          }
        } finally {
          rmSync(tmp, { recursive: true, force: true });
        }
      `;
      expect(execFileSync(NODE!, ["--input-type=module", "-e", script], { encoding: "utf8" })).toBe(
        "ok fsync fails once\nok write fails halfway\nok short write completes\nok rollback truncate fails\n",
      );
    },
    30_000,
  );
});
