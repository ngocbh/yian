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
 * `test_jsonl_context_releases_lock_when_the_body_raises` (`using`),
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
 * seq, a line that isn't UTF-8); a failed resume whose descriptor close also fails throws the
 * original error; a write that fails and whose rollback truncate also fails; closing a poisoned
 * log, after which an append throws `run log is closed`; with `YIAN_TEST_NODE` set to a Node
 * binary, the built log on Node through a failed sync, a failed write, a short write and a write
 * whose rollback truncate also fails.
 *
 * Changed:
 * - `test_schema_version_is_nine`: `SCHEMA_VERSION` is 10.
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
 *   names Python yian.
 * - `test_schema_check_does_not_weaken_validation`: Python's 10 and 9.0 rows are 11 and the float
 *   token `10.0` (the current schema written as a float); the float row is wire-only, since a JS
 *   `10.0` is the number 10.
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
 * - Python refuses entering a closed log with `with`; TS `using` has no enter step, so there is
 *   nothing to refuse.
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
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { FileLockError, isLocked, lockFile } from "../../src/platform.ts";
import {
  decodeRecord,
  encodeRecord,
  isRunLogLocked,
  JsonlRunLog,
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

/** Assert no writer holds the lock of the log at `file`. */
function expectUnlocked(file: string): void {
  expect(isLocked(`${file}.lock`)).toBe(false);
}

/** Create a log at `file` holding one `node_started` record for `first`. */
function baselineLog(file: string): JsonlRunLog {
  const log = JsonlRunLog.create(file, "run");
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

test("the schema version is 10", () => {
  expect(SCHEMA_VERSION).toBe(10);
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

test.each<[number, string]>([
  [7, "unsupported run-record schema: 7"],
  [8, "unsupported run-record schema: 8"],
  [9, PYTHON_SCHEMA],
])("a retired schema %d log is rejected on resume", (schema, message) => {
  const file = path.join(dir, "retired.jsonl");
  const bytes = `${wireLine({ schema: String(schema) })}\n`;
  fs.writeFileSync(file, bytes);
  expectError(() => JsonlRunLog.resume(file), RangeError, `run log ${file} line 1 (seq 0): ${message}`);
  expect(fs.readFileSync(file, "utf8")).toBe(bytes);
  expectUnlocked(file);
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
    using log = JsonlRunLog.create(file, "run");
    expect(log.path).toBe(file);
    expect(log.runId).toBe("run");
    expect(log.nextSequence).toBe(0);
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
    using created = JsonlRunLog.create(file, "run");
    created.append(new NodeStartedRecord("work"));
  }
  const real = fs.readFileSync;
  const lockedAtRead: boolean[] = [];
  const spy = spyOn(fs, "readFileSync").mockImplementation(((target: unknown, ...rest: unknown[]) => {
    if (isDescriptorOf(target, file)) lockedAtRead.push(isLocked(`${file}.lock`));
    return Reflect.apply(real, fs, [target, ...rest]);
  }) as typeof fs.readFileSync);
  try {
    using resumed = JsonlRunLog.resume(file);
    expect(nodeIds(resumed)).toEqual(["work"]);
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
  using reopened = JsonlRunLog.resume(file);
  expect(reopened.runId).toBe("run");
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
    using resumed = JsonlRunLog.resume(file);
    expect(nodeIds(resumed)).toEqual(["first"]);
    expect(resumed.nextSequence).toBe(1);
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
    "run-record kind 'xnode_started' is unsupported in schema 10",
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

test.each<[string, string, string]>([
  ["schema 1", `${wireLine({ schema: "1" })}\n`, "line 1 (seq 0): unsupported run-record schema: 1"],
  ["schema 2", `${wireLine({ schema: "2" })}\n`, "line 1 (seq 0): unsupported run-record schema: 2"],
  ["sequence", `${wireLine({ seq: 1 })}\n`, "line 1 (seq 0): run-log sequence mismatch: expected 0, found 1"],
  [
    "run id",
    `${wireLine({ runId: "first" })}\n${wireLine({ runId: "second", seq: 1, nodeId: "other" })}\n`,
    "line 2 (seq 1): run-log run_id mismatch: expected 'first', found 'second'",
  ],
])("jsonl resume rejects identity mismatches and releases the lock: %s", (_name, contents, where) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, contents);
  expectError(() => JsonlRunLog.resume(file), RangeError, `run log ${file} ${where}`);
  expectUnlocked(file);
});

test.each<[string, string, string | null]>([
  ["not a log", "id: echo\nname: echo", "line 1 (seq 0): invalid run-record JSON"],
  ["no complete line", '{"a": 1}', null],
  ["retired schema", `${wireLine({ schema: "2" })}\n{"partial":`, "line 1 (seq 0): unsupported run-record schema: 2"],
  ["corrupt line", `${wireLine()}\n{not-json}\n{"partial":`, "line 2 (seq 1): invalid run-record JSON"],
])("jsonl resume leaves a rejected file unchanged: %s", (_name, contents, where) => {
  const file = path.join(dir, "run.jsonl");
  fs.writeFileSync(file, contents);
  const message = where === null ? `cannot resume an empty run log: ${file}` : `run log ${file} ${where}`;
  expectError(() => JsonlRunLog.resume(file), RangeError, message);
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
  using reopened = JsonlRunLog.resume(file);
  expect([...reopened]).toEqual(records);
});

test("disposing a jsonl log releases the lock when the body throws", () => {
  const file = path.join(dir, "run.jsonl");
  expectError(
    () => {
      using log = JsonlRunLog.create(file, "run");
      log.append(new NodeStartedRecord("work"));
      throw new Error("body failed");
    },
    Error,
    "body failed",
  );
  using reopened = JsonlRunLog.resume(file);
  expect(reopened.nextSequence).toBe(1);
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
    using resumed = JsonlRunLog.resume(link);
    expect(resumed.path).toBe(link);
    expect(isRunLogLocked(real)).toBe(true);
    resumed.append(new NodeStartedRecord("second"));
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
  const log = JsonlRunLog.create(file, "run");
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
    using log = JsonlRunLog.create(file, "run");
    const envelope = log.append(new NodeExpandedRecord("each", seed, [], DIGEST));
    seed.items.push(2n);
    expect(seedOf(envelope)).toEqual({ items: [1n] });
    expect(seedOf([...log][0]!)).toEqual({ items: [1n] });
  }
  using resumed = JsonlRunLog.resume(file);
  expect(seedOf([...resumed][0]!)).toEqual({ items: [1n] });
});

test("a jsonl log never exposes its retained mutable payload", () => {
  const file = path.join(dir, "run.jsonl");
  const log = JsonlRunLog.create(file, "run");
  const appended = log.append(new NodeExpandedRecord("each", { items: [{ value: 1n }] }, [], DIGEST));
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

  using resumed = JsonlRunLog.resume(file);
  expect(seedOf(resumed[Symbol.iterator]().next().value!)).toEqual({ items: [{ value: 1n }] });
  expect(fs.readFileSync(file, "utf8")).toBe(durable);
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
  using resumed = JsonlRunLog.resume(file);
  expect(nodeIds(resumed)).toEqual(["work", "second"]);
  expect(resumed.append(new NodeStartedRecord("third")).seq).toBe(2);
});

test("an invalid append retains the prior complete prefix and doesn't poison the log", () => {
  const file = path.join(dir, "run.jsonl");
  using log = baselineLog(file);
  const error = expectAppendFailure(log, file, () => log.append({ nodeId: "x" } as never));
  expect((error as Error).constructor).toBe(TypeError);
  expect((error as Error).message).toBe("unsupported run record: Object");
  expect(log.append(new NodeStartedRecord("second")).seq).toBe(1);
});

test("a write failure rolls back to the prior complete prefix", () => {
  const file = path.join(dir, "run.jsonl");
  using log = baselineLog(file);
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
});

test("an fsync failure rolls back and syncs the prior complete prefix", () => {
  const file = path.join(dir, "run.jsonl");
  using log = baselineLog(file);
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
});

test("a short write is completed by writing the rest", () => {
  const file = path.join(dir, "run.jsonl");
  using log = baselineLog(file);
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
});

test("a write of 0 bytes after a partial write is an append failure and rolls back", () => {
  const file = path.join(dir, "run.jsonl");
  using log = baselineLog(file);
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
const ROOT = path.resolve(import.meta.dir, "../..");
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
          const log = JsonlRunLog.create(file, "run");
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
