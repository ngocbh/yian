/**
 * The run-log readers that start anywhere but the front: `iterRunRecordsBackwards`, which reads
 * from the end, and `runRecordAt`, which bisects to one seq. Each reads what `iterRunRecords`
 * reads, under its crash rules.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  iterRunRecords,
  iterRunRecordsBackwards,
  type IterRunRecordsBackwardsOptions,
  type IterRunRecordsOptions,
  RunLogChangedError,
  RunLogLineTooLongError,
  runRecordAt,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import { thrownClass } from "../helpers/own_errors.ts";

const AT = "2026-10-20T14:02:11.123Z";
/** The bytes the readers ask for in one read. */
const CHUNK = 1 << 16;
const CHANGED = "it was truncated or replaced";

/** One record's line, without its newline: a `node_started`, or a `run_stopped`, padded by `pad` bytes. */
function line(seq: number, { runId = "run", pad = 0, stop = false } = {}): string {
  const kind = stop ? "run_stopped" : "node_started";
  const payload = stop
    ? `{"message":"${"m".repeat(pad)}","node_ids":[],"reason":"signal"}`
    : `{"node_id":"n${seq}${"x".repeat(pad)}"}`;
  return `{"at":"${AT}","kind":"${kind}","payload":${payload},"run_id":${JSON.stringify(runId)},"schema":${SCHEMA_VERSION},"seq":${seq}}`;
}

/** `count` lines of varying length, every fifth a stop, so lines straddle the chunks' ends. */
function lines(count: number, runId = "run"): string[] {
  return Array.from({ length: count }, (_, seq) => line(seq, { runId, pad: (seq * 37) % 300, stop: seq % 5 === 4 }));
}

/** Write `records` as a log at `file`, each line ended, then `tail`. */
function writeLog(file: string, records: readonly string[], tail = ""): void {
  fs.writeFileSync(file, `${records.map((record) => `${record}\n`).join("")}${tail}`);
}

/** Where each of `records` starts in a log of them. */
function offsetsOf(records: readonly string[]): number[] {
  let offset = 0;
  return records.map((record) => {
    const start = offset;
    offset += Buffer.byteLength(record) + 1;
    return start;
  });
}

/** The seqs `entries` hold, in order. */
function seqs(entries: Iterable<{ readonly seq: number }>): number[] {
  return Array.from(entries, (entry) => entry.seq);
}

/** Assert `fn` throws an error of exactly class `cls` with exactly `message`, and return it. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): Error {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error | undefined)?.constructor).toBe(thrownClass(cls));
  expect((caught as Error).message).toBe(message);
  return caught as Error;
}

/** The options both directions take, as a forward and a backward read of each are compared. */
const SHARED_OPTIONS: readonly Pick<IterRunRecordsOptions, "kinds" | "payloads">[] = [
  {},
  { payloads: false },
  { kinds: ["run_stopped", "pool_commit", "node_step"] },
  { kinds: ["node_started"], payloads: false },
];

/** Assert the log at `file` reads backwards as forwards reversed, and by seq as forwards, with each option. */
function expectSameEveryWay(file: string): void {
  for (const options of SHARED_OPTIONS) {
    expect([...iterRunRecordsBackwards(file, options)]).toEqual([...iterRunRecords(file, options)].reverse());
  }
  const forward = [...iterRunRecords(file)];
  for (const entry of forward) expect(runRecordAt(file, entry.seq)).toEqual(entry);
  expect(runRecordAt(file, forward.length)).toBeNull();
}

/** The conformance cases' golden logs. */
function goldenLogs(folder = path.join(import.meta.dirname, "..", "conformance", "cases")): string[] {
  return fs.readdirSync(folder, { withFileTypes: true }).flatMap((entry) => {
    const at = path.join(folder, entry.name);
    if (entry.isDirectory()) return goldenLogs(at);
    return entry.name.endsWith(".log.jsonl") ? [at] : [];
  });
}

/** A golden log's text with the times and clock values it masks filled in, so it reads as a log. */
function unmasked(text: string): string {
  return text
    .replaceAll('"at":"<masked>"', `"at":"${AT}"`)
    .replace(/("kind":"datetime","node_id":"[^"]*","value":)"<masked>"/g, `$1"${AT}"`)
    .replace(/("kind":"date","node_id":"[^"]*","value":)"<masked>"/g, '$1"2026-10-20"');
}

/** Read the first `count` entries of a backwards read from `cursor`, then stop. */
function page(file: string, cursor: IterRunRecordsBackwardsOptions, count: number) {
  const entries = [];
  for (const entry of iterRunRecordsBackwards(file, cursor)) {
    entries.push(entry);
    if (entries.length === count) break;
  }
  return entries;
}

/** The cursor for the page before `entries`, the last page read. */
function before(entries: readonly { offset: number; runId: string; seq: number }[]): IterRunRecordsBackwardsOptions {
  const oldest = entries.at(-1)!;
  return { beforeOffset: oldest.offset, runId: oldest.runId, beforeSeq: oldest.seq };
}

/** Count the reads of the file at `file`, and the bytes asked for, while `fn` runs. */
function countReads<T>(file: string, fn: () => T): { result: T; reads: number; bytes: number } {
  const target = fs.statSync(file).ino;
  const realRead = fs.readSync;
  let reads = 0;
  let bytes = 0;
  const spy = spyOn(fs, "readSync").mockImplementation(((fd: number, buffer: Uint8Array, ...rest: any[]) => {
    if (fs.fstatSync(fd).ino === target) {
      reads++;
      bytes += rest[1] as number;
    }
    return Reflect.apply(realRead, fs, [fd, buffer, ...rest]) as number;
  }) as typeof fs.readSync);
  try {
    return { result: fn(), reads, bytes };
  } finally {
    spy.mockRestore();
  }
}

let dir: string;

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-log-back-"));
});

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

test("every golden log reads backwards as forwards reversed, and each seq by bisection, with every option", () => {
  const logs = goldenLogs();
  expect(logs.length).toBeGreaterThan(30);
  const file = path.join(dir, "golden.jsonl");
  for (const golden of logs) {
    fs.writeFileSync(file, unmasked(fs.readFileSync(golden, "utf8")));
    expectSameEveryWay(file);
  }
});

test.each([-1, 0, 1])("a log many chunks long reads the same every way, its newest lines a chunk %d bytes long", (shift) => {
  const file = path.join(dir, "run.jsonl");
  // The newest 200 lines fill a chunk, give or take `shift`: the first read back starts at the
  // newline before them, at their first byte, or a byte into it.
  const records = lines(3000);
  records[2800] = line(2800);
  const newest = records.slice(2800).reduce((sum, record) => sum + Buffer.byteLength(record) + 1, 0);
  records[2800] = line(2800, { pad: CHUNK + shift - newest });
  writeLog(file, records);
  expect(fs.statSync(file).size - [...iterRunRecords(file)][2800]!.offset).toBe(CHUNK + shift);
  expectSameEveryWay(file);
});

test("paging back with each page's oldest entry as the cursor reads every record once, and reads only as far as asked", () => {
  const file = path.join(dir, "run.jsonl");
  writeLog(file, lines(3000));
  const read: number[] = [];
  let cursor: IterRunRecordsBackwardsOptions = {};
  for (let entries = page(file, cursor, 7); entries.length > 0; entries = page(file, cursor, 7)) {
    read.push(...seqs(entries));
    cursor = before(entries);
  }
  expect(read).toEqual(Array.from({ length: 3000 }, (_, i) => 2999 - i));
  expect(cursor).toEqual({ beforeOffset: 0, runId: "run", beforeSeq: 0 });

  const stops: number[] = [];
  cursor = { kinds: ["run_stopped"] };
  for (let entries = page(file, cursor, 9); entries.length > 0; entries = page(file, cursor, 9)) {
    stops.push(...seqs(entries));
    cursor = { ...before(entries), kinds: ["run_stopped"] };
  }
  expect(stops).toEqual(Array.from({ length: 600 }, (_, i) => 2999 - 5 * i));

  const first = countReads(file, () => page(file, {}, 20));
  expect(seqs(first.result)).toEqual(Array.from({ length: 20 }, (_, i) => 2999 - i));
  expect(first.reads).toBe(1);
  expect(first.bytes).toBe(CHUNK);
});

test("runRecordAt bisects: a few small reads find any seq of a long log, or none past its end", () => {
  const file = path.join(dir, "run.jsonl");
  writeLog(file, lines(12_000));
  const size = fs.statSync(file).size;
  for (const seq of [0, 1, 4321, 11_998, 11_999]) {
    const { result, reads, bytes } = countReads(file, () => runRecordAt(file, seq));
    expect(result!.seq).toBe(seq);
    expect(reads).toBeLessThanOrEqual(40);
    expect(bytes).toBeLessThan(size / 20);
  }
  expect(runRecordAt(file, 12_000)).toBeNull();
  expect(runRecordAt(file, 1e9)).toBeNull();
  for (const seq of [-1, 1.5, Number.NaN]) {
    expectError(() => runRecordAt(file, seq), RangeError, `seq must be a non-negative integer, got ${seq}`);
  }
});

test("both readers leave out an unterminated last line, and read it once it's complete", () => {
  const file = path.join(dir, "run.jsonl");
  const torn = line(3, { pad: 2 * CHUNK });
  writeLog(file, lines(3), torn.slice(0, CHUNK + 5));
  expect(seqs(iterRunRecordsBackwards(file))).toEqual([2, 1, 0]);
  expect(seqs(iterRunRecordsBackwards(file, { maxLineBytes: 1000 }))).toEqual([2, 1, 0]);
  expect(runRecordAt(file, 3)).toBeNull();
  expect(runRecordAt(file, 2)!.seq).toBe(2);

  fs.appendFileSync(file, `${torn.slice(CHUNK + 5)}\n`);
  expect(seqs(iterRunRecordsBackwards(file))).toEqual([3, 2, 1, 0]);
  expect(runRecordAt(file, 3)!.end).toBe(fs.statSync(file).size);

  for (const text of ["", line(0).slice(0, 10)]) {
    fs.writeFileSync(file, text);
    expect([...iterRunRecordsBackwards(file)]).toEqual([]);
    expect(runRecordAt(file, 0)).toBeNull();
  }
});

test("paging refuses a log cut below its cursor or replaced by another run's, and reads on after appends", () => {
  const file = path.join(dir, "run.jsonl");
  writeLog(file, lines(10, "run-a"));
  const cursor = before(page(file, {}, 4));
  expect(cursor).toMatchObject({ runId: "run-a", beforeSeq: 6 });

  fs.appendFileSync(file, `${line(10, { runId: "run-a" })}\n`);
  expect(seqs(page(file, cursor, 2))).toEqual([5, 4]);

  const at = cursor.beforeOffset!;
  const changed = (reason: string) => `run log ${file} doesn't lead to the records read from byte ${at}: ${reason}; ${CHANGED}`;
  // Another run's log, line for line the same length, replaces the file.
  const other = path.join(dir, "other.jsonl");
  writeLog(other, lines(10, "run-b"));
  fs.renameSync(other, file);
  const replaced = changed("expected seq 5 of run 'run-a', found seq 5 of run 'run-b'");
  expectError(() => page(file, cursor, 2), RunLogChangedError, replaced);
  // Without the run id and seq, the offset alone reads on into the other run.
  expect(page(file, { beforeOffset: at }, 1)[0]!.runId).toBe("run-b");

  writeLog(file, lines(10, "run-a"));
  const skipped = changed("expected seq 6 of run 'run-a', found seq 5 of run 'run-a'");
  expectError(() => page(file, { ...cursor, beforeSeq: 7 }, 2), RunLogChangedError, skipped);
  writeLog(file, [line(0, { runId: "run-a", pad: 1 }), ...lines(10, "run-a").slice(1)]);
  expectError(() => page(file, cursor, 2), RunLogChangedError, changed("it isn't the start of a line"));
  const notStart = `beforeOffset ${at} isn't the start of a line in run log ${file}`;
  expectError(() => page(file, { beforeOffset: at }, 2), RangeError, notStart);

  fs.truncateSync(file, at - 1);
  const shrank = `run log ${file} shrank to ${at - 1} bytes under its reader at byte ${at}; ${CHANGED}`;
  expectError(() => page(file, cursor, 2), RunLogChangedError, shrank);
});

test("a backwards read refuses a log that shrinks while it reads, or that loses its newest line before it ends", () => {
  const file = path.join(dir, "run.jsonl");
  writeLog(file, lines(1000));
  const size = fs.statSync(file).size;
  const reader = iterRunRecordsBackwards(file);
  expect(reader.next().value!.seq).toBe(999);
  fs.truncateSync(file, 0);
  const message = (at: number) => `run log ${file} shrank to 0 bytes under its reader at byte ${at}; ${CHANGED}`;
  // The rest of the first chunk was read before the cut.
  const left = [];
  expectError(() => {
    for (const entry of reader) left.push(entry);
  }, RunLogChangedError, message(size - CHUNK));
  expect(left.length).toBeGreaterThan(100);

  writeLog(file, lines(3));
  const whole = fs.statSync(file).size;
  const short = iterRunRecordsBackwards(file);
  expect(seqs([short.next().value!, short.next().value!, short.next().value!])).toEqual([2, 1, 0]);
  fs.truncateSync(file, 0);
  expectError(() => short.next(), RunLogChangedError, message(whole));
});

test("lines longer than a chunk read the same every way, and over maxLineBytes throw as a forward read throws them", () => {
  const file = path.join(dir, "run.jsonl");
  const records = lines(6);
  records[0] = line(0, { pad: 3 * CHUNK });
  records[3] = line(3, { stop: true, pad: 2 * CHUNK + 7 });
  records[5] = line(5, { pad: CHUNK });
  writeLog(file, records);
  expectSameEveryWay(file);

  const cap = 1000;
  const offsets = offsetsOf(records);
  const tooLong = (seq: number, where: string) =>
    `run log ${file} ${where}: the line is ${Buffer.byteLength(records[seq]!)} bytes, over maxLineBytes ${cap}`;
  const fields = (error: Error) => {
    const { offset, end, length, limit, seq, runId, kind, nodeId } = error as RunLogLineTooLongError;
    return { offset, end, length, limit, seq, runId, kind, nodeId };
  };
  /** The error a forward read from line `seq` throws for it. */
  const forward = (seq: number) => {
    const read = () => [...iterRunRecords(file, { afterOffset: offsets[seq]!, maxLineBytes: cap })];
    return expectError(read, RunLogLineTooLongError, tooLong(seq, `byte ${offsets[seq]}`));
  };
  const sizes: number[] = [];
  const realConcat = Buffer.concat;
  const concatSpy = spyOn(Buffer, "concat").mockImplementation(((list: readonly Uint8Array[], ...rest: any[]) => {
    const joined = Reflect.apply(realConcat, Buffer, [list, ...rest]) as Buffer;
    sizes.push(joined.length);
    return joined;
  }) as typeof Buffer.concat);
  try {
    // From the end, the newest line is too long; a host reads on before it, as the error names it.
    const fromEnd = () => [...iterRunRecordsBackwards(file, { maxLineBytes: cap })];
    const newest = expectError(fromEnd, RunLogLineTooLongError, tooLong(5, `byte ${offsets[5]}`));
    expect(fields(newest)).toEqual(fields(forward(5)));
    const { offset, runId, seq } = newest as RunLogLineTooLongError;
    const older = iterRunRecordsBackwards(file, { beforeOffset: offset, runId, beforeSeq: seq, maxLineBytes: cap });
    expect(older.next().value!.seq).toBe(4);
    const third = expectError(() => older.next(), RunLogLineTooLongError, tooLong(3, `byte ${offsets[3]} (seq 3)`));
    expect(fields(third)).toEqual(fields(forward(3)));
    expect(fields(third)).toMatchObject({ seq: 3, runId: "run", kind: "run_stopped", nodeId: null });

    const bySeq = () => runRecordAt(file, 3, { maxLineBytes: cap });
    const at = expectError(bySeq, RunLogLineTooLongError, tooLong(3, `byte ${offsets[3]}`));
    expect(fields(at)).toEqual(fields(third));
    // A long line on the way is bisected by the seq its scan reads.
    expect(seqs([1, 2, 4].map((seq) => runRecordAt(file, seq, { maxLineBytes: cap })!))).toEqual([1, 2, 4]);
  } finally {
    concatSpy.mockRestore();
  }
  expect(Math.max(0, ...sizes)).toBeLessThanOrEqual(cap);
});

test.each([
  ["a short line", true, (records: string[]) => [records[0], line(1), records[2], "x".repeat(5000)]],
  ["a longer one", false, (records: string[]) => [records[0], line(1, { pad: 6000 }), records[2]]],
] as const)("a backwards read refuses an over-long line rewritten as %s before its scan reads it again", (_name, shorter, rewrite) => {
  const file = path.join(dir, "run.jsonl");
  const records = [line(0), line(1, { pad: 5000 }), line(2)];
  writeLog(file, records);
  const long = offsetsOf(records)[1]!;
  const realRead = fs.readSync;
  let rewritten = false;
  // The scan's first read of the long line, after the read back found it, sees it rewritten in place.
  const readSpy = spyOn(fs, "readSync").mockImplementation(((fd: number, buffer: Uint8Array, ...rest: any[]) => {
    if (!rewritten && rest[2] === long && (rest[1] as number) <= 4096) {
      rewritten = true;
      fs.writeFileSync(file, rewrite(records).map((record) => `${record}\n`).join(""));
    }
    return Reflect.apply(realRead, fs, [fd, buffer, ...rest]) as number;
  }) as typeof fs.readSync);
  try {
    const end = offsetsOf(records)[2]!;
    const moved = shorter ? `the line at byte ${long} no longer ends at byte ${end}` : `no line ends between bytes ${long} and ${end}`;
    const message = `run log ${file} changed under its reader: ${moved}; ${CHANGED}`;
    expectError(() => [...iterRunRecordsBackwards(file, { maxLineBytes: 1000 })], RunLogChangedError, message);
  } finally {
    readSpy.mockRestore();
  }
  expect(rewritten).toBe(true);
});

test("kinds and payloads: false check every line's envelope, as a forward read does", () => {
  const file = path.join(dir, "run.jsonl");
  const second = Buffer.byteLength(line(0)) + 1;
  /** The message a forward read throws for line 2, as a read that doesn't start at the front names the line. */
  const forwardMessage = (options: IterRunRecordsOptions) => {
    try {
      [...iterRunRecords(file, options)];
    } catch (error) {
      return (error as Error).message.replace("line 2 (seq 1)", `byte ${second} (seq 1)`);
    }
    throw new Error("expected the forward read to throw");
  };
  const extra = line(1).replace('{"node_id"', '{"extra":1,"node_id"');
  writeLog(file, [line(0), extra, line(2, { stop: true })]);
  expect(seqs(iterRunRecordsBackwards(file, { payloads: false }))).toEqual([2, 1, 0]);
  expect(seqs(iterRunRecordsBackwards(file, { kinds: ["run_stopped"] }))).toEqual([2]);
  const invalid = forwardMessage({});
  expect(invalid).toEndWith(`invalid node_started payload fields; missing=[], extra=["extra"]`);
  expectError(() => [...iterRunRecordsBackwards(file)], RangeError, invalid);
  expectError(() => runRecordAt(file, 1), RangeError, invalid.replace(" (seq 1)", ""));

  writeLog(file, [line(0), `{"kind":"node_started"}`, line(2, { stop: true })]);
  const kinds = ["run_stopped"];
  expectError(() => [...iterRunRecordsBackwards(file, { kinds })], RangeError, forwardMessage({ kinds }));
  const unknown = () => [...iterRunRecordsBackwards(file, { kinds: ["node_paused"] })];
  expectError(unknown, RangeError, `unknown run-record kinds: ["node_paused"]`);
});

test("a backwards read checks that seqs count down by one in one run, to seq 0 at the start", () => {
  const file = path.join(dir, "run.jsonl");
  const read = (records: string[]) => {
    writeLog(file, records);
    return () => [...iterRunRecordsBackwards(file)];
  };
  const mismatch = "run-log sequence mismatch";
  const gap = [line(0), line(1), line(3)];
  expectError(read(gap), RangeError, `run log ${file} byte ${offsetsOf(gap)[1]} (seq 2): ${mismatch}: expected 2, found 1`);
  const foreign = `run log ${file} byte 0 (seq 0): run-log run_id mismatch: expected 'other', found 'run'`;
  expectError(read([line(0), line(1, { runId: "other" })]), RangeError, foreign);
  expectError(read([line(1), line(2)]), RangeError, `run log ${file} line 1 (seq 0): ${mismatch}: expected 0, found 1`);
  const twice = [line(0), line(0)];
  expectError(read(twice), RangeError, `run log ${file} byte ${offsetsOf(twice)[1]}: ${mismatch}: seq 0 isn't the first line`);
});

test.each<[string, Record<string, unknown>, string]>([
  ["a negative beforeOffset", { beforeOffset: -1 }, "beforeOffset must be a non-negative integer, got -1"],
  ["a fractional beforeOffset", { beforeOffset: 1.5 }, "beforeOffset must be a non-negative integer, got 1.5"],
  ["a negative beforeSeq", { beforeOffset: 1, beforeSeq: -1 }, "beforeSeq must be a non-negative integer, got -1"],
  ["an empty runId", { beforeOffset: 1, runId: "" }, "runId must be a non-empty string, got ''"],
  ["beforeSeq without an offset", { beforeSeq: 1 }, "runId and beforeSeq describe the record at beforeOffset, so they need one"],
  ["runId without an offset", { runId: "run" }, "runId and beforeSeq describe the record at beforeOffset, so they need one"],
  [
    "beforeSeq 0 past the start",
    { beforeOffset: 5, beforeSeq: 0 },
    "a log's first record, seq 0, starts at byte 0: beforeSeq 0 can't be at beforeOffset 5",
  ],
  [
    "a later beforeSeq at the start",
    { beforeOffset: 0, beforeSeq: 5 },
    "a log's first record, seq 0, starts at byte 0: beforeSeq 5 can't be at beforeOffset 0",
  ],
  ["a maxLineBytes of 0", { maxLineBytes: 0 }, "maxLineBytes must be a positive integer, got 0"],
])("a backwards read refuses %s, on its first next()", (_name, options, message) => {
  const file = path.join(dir, "run.jsonl");
  writeLog(file, [line(0)]);
  const reader = iterRunRecordsBackwards(file, options);
  expectError(() => reader.next(), RangeError, message);
});

test("both readers throw ENOENT for a missing log, a backwards read on its first next(), and read nothing before byte 0", () => {
  const missing = path.join(dir, "missing.jsonl");
  const reader = iterRunRecordsBackwards(missing);
  for (const read of [() => reader.next(), () => runRecordAt(missing, 0)]) {
    let caught: NodeJS.ErrnoException | undefined;
    try {
      read();
    } catch (error) {
      caught = error as NodeJS.ErrnoException;
    }
    expect(caught?.code).toBe("ENOENT");
  }
  const file = path.join(dir, "run.jsonl");
  writeLog(file, lines(3));
  expect([...iterRunRecordsBackwards(file, { beforeOffset: 0, runId: "run", beforeSeq: 0 })]).toEqual([]);
});
