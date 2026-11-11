/**
 * The run index, `<runs>/index.sqlite`. `updateRunIndex` folds each run folder's log into
 * one row from the byte offset it reached last time, rebuilds a row whose log was truncated or
 * replaced, and serializes builders in one SQLite transaction that waits a bounded time for the
 * lock. `readRunIndex` puts `running` on top of a stored status while the writer's lock is held.
 * The index may lag the logs and never leads them; deleting it and building again gives the same
 * rows.
 */

import { execFileSync, spawn, spawnSync } from "node:child_process";
import fs, {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  truncateSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { afterAll, describe, expect, spyOn, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { Failure } from "../../src/diagnostics.ts";
import { openSqlite, withWarnings } from "../../src/platform.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { FileMemory } from "../../src/memory/file.ts";
import {
  inspectRun,
  readRunIndex,
  RunIndexBusyError,
  type RunIndexRow,
  runFlow,
  runLogPath,
  updateRunIndex,
} from "../../src/run.ts";
import { emptyRunScan, foldRunScan, indexRows, type RunScan, runScanFromJson, scanStatus } from "../../src/runtime/run_index.ts";
import { WaitsState } from "../../src/runtime/waits.ts";
import { DEADLINE, runStart, sleepWait, send as sendTo, takeWait as takeWaitOn, WAIT_LOGS } from "./wait_logs.ts";
import {
  InboxClosedRecord,
  JsonlRunLog,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  RunCancelledRecord,
  RunFailedRecord,
  type RunRecord,
  RunResumedRecord,
  RunStartedRecord,
  RunStoppedRecord,
  RunWaitingRecord,
  readRunLog,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import type { TypedValue } from "../../src/typesys/values.ts";

type TypedWire = ReturnType<TypedValue["toWire"]>;

const TS_ROOT = fileURLToPath(new URL("../../", import.meta.url));
const tmp = mkdtempSync(join(tmpdir(), "yian-run-index-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

let dirs = 0;
/** A fresh, empty runs folder. */
function freshRuns(): string {
  const dir = join(tmp, `runs-${++dirs}`);
  mkdirSync(dir);
  return dir;
}

/** A clock that reads 2026-10-23T00:00:00.000Z, then one second later on each call. */
function steppingClock(): () => number {
  let ms = Date.parse("2026-10-23T00:00:00.000Z");
  return () => (ms += 1000) - 1000;
}

/** The record time `seconds` after 2026-10-23T00:00:00.000Z. */
const timeAt = (seconds: number) => new Date(Date.parse("2026-10-23T00:00:00.000Z") + seconds * 1000).toISOString();

/** A run-start record of the flow `<dir>/flow.yaml`, with `parent` when given. */
function runStarted(parent: { runId: string; node: string | null } | null = null): RunStartedRecord {
  return new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    flow: { root: "flow.yaml", dir: "/flows", codeRoot: null, files: { "flow.yaml": { text: "id: f\n", refs: {} } } },
    parent,
  });
}

/** A `model_call` step with these token counts; `error` makes it a failed attempt. */
function modelCall(
  usage: [number, number, number | null, number | null, number | null] | null,
  error = false,
): Record<string, unknown> {
  const count = (n: number | null) => (n === null ? null : BigInt(n));
  return {
    kind: "model_call",
    model: "gpt-test",
    usage:
      usage === null
        ? null
        : {
            input_tokens: count(usage[0]),
            output_tokens: count(usage[1]),
            cache_read_tokens: count(usage[2]),
            cache_creation_tokens: count(usage[3]),
            reasoning_tokens: count(usage[4]),
          },
    ...(error ? { error: { kind: "server", message: "overloaded" } } : {}),
  };
}

/** Append a node's commit of `wire` and its success naming it; return the commit's seq. */
function succeed(log: JsonlRunLog, node: string, wire: TypedWire): number {
  const commit = log.append(new PoolCommitRecord(node, node, wire)).seq;
  log.append(new NodeSucceededRecord(node, commit));
  return commit;
}

/** Create the run `runId` under `runs`, append what `write` appends, and close it unless `keep`. */
function writeRun(runs: string, runId: string, write: (log: JsonlRunLog) => void, { keep = false } = {}): JsonlRunLog {
  const log = JsonlRunLog.create(runLogPath(runs, runId), runId, { clock: steppingClock() });
  try {
    write(log);
  } finally {
    if (!keep) log.close();
  }
  return log;
}

/**
 * Run, as `runId` in `runs`, a flow that appends to its memory `lessons`, bound to a FileMemory
 * of identity `t/lessons`; its Optional memory `notes` is left out.
 */
async function memoryRun(runs: string, runId: string): Promise<void> {
  const flow = await loadFlow(
    "id: m\nname: m\ninput:\n  lessons: !FileMemory\n  notes: !Optional(FileMemory)\nnodes:\n  add:\n" +
      "    kind: tool\n    tool_id: {lessons: append}\n    input: {path: MEMORY.md, text: hi}\n" +
      "output: ${add.output.path}\n",
  );
  const lessons = new FileMemory({ root: join(runs, "..", `${runId}-memory`), id: "t/lessons" });
  const result = await runFlow(flow, { lessons }, { log: runLogPath(runs, runId), runId, env: () => undefined });
  expect(result.status).toBe("succeeded");
}

/** The start of every test run: `run_started`, then the START inputs' commit. */
function start(log: JsonlRunLog, parent: { runId: string; node: string | null } | null = null): void {
  log.append(runStarted(parent));
  log.append(new PoolCommitRecord(START_ID, START_ID, { kind: "object", value: { topic: "clouds" } }));
}

/** A run whose agent failed one call, succeeded the next, and whose END succeeded. Ends at seq 9. */
function succeededRun(log: JsonlRunLog): void {
  start(log, { runId: "parent-run", node: "spawner" });
  log.append(new NodeStartedRecord("agent"));
  log.append(new NodeStepRecord("agent", 1, modelCall([100, 3, null, 4, null], true)));
  log.append(new NodeStepRecord("agent", 2, modelCall([120, 30, 64, 16, 8])));
  succeed(log, "agent", { kind: "str", value: "rain" });
  log.append(new NodeStartedRecord(END_ID));
  succeed(log, END_ID, { kind: "object", value: { answer: "rain", n: 2n } });
}

/** A run whose only node failed after one call. */
function failedRun(log: JsonlRunLog): void {
  start(log);
  log.append(new NodeStartedRecord("agent"));
  log.append(new NodeStepRecord("agent", 1, modelCall([10, 1, null, null, null], true)));
  log.append(new NodeFailedRecord(new Failure({ code: "node.execute", message: "boom", nodeId: "agent" })));
}

/** A run the host stopped while its node was in flight. */
function stoppedRun(log: JsonlRunLog): void {
  start(log);
  log.append(new NodeStartedRecord("agent"));
  log.append(new RunStoppedRecord("signal", "SIGINT", ["agent"]));
}

/** A run whose node is in flight. */
function runningRun(log: JsonlRunLog): void {
  start(log);
  log.append(new NodeStartedRecord("agent"));
  log.append(new NodeStepRecord("agent", 1, modelCall([7, 2, null, null, 1])));
}

/** The rows by run id. */
function byId(rows: readonly RunIndexRow[]): Record<string, RunIndexRow> {
  return Object.fromEntries(rows.map((row) => [row.runId, row]));
}

/** The four runs of the first test, the running one's log still held. */
function fourRuns(runs: string): JsonlRunLog {
  writeRun(runs, "run-succeeded", succeededRun);
  writeRun(runs, "run-failed", failedRun);
  writeRun(runs, "run-stopped", stoppedRun);
  return writeRun(runs, "run-running", runningRun, { keep: true });
}

describe("updateRunIndex and readRunIndex", () => {
  test("one row per run: status, times, parent, result and usage sums, failed calls included", () => {
    const runs = freshRuns();
    const held = fourRuns(runs);
    try {
      // A run folder whose log isn't there yet has no row.
      mkdirSync(join(runs, "run-new"));
      updateRunIndex(runs);
      const rows = byId(readRunIndex(runs));
      expect(Object.keys(rows).sort()).toEqual(["run-failed", "run-running", "run-stopped", "run-succeeded"]);
      expect(rows["run-succeeded"]).toEqual({
        runId: "run-succeeded",
        flow: "/flows/flow.yaml",
        status: "succeeded",
        startedAt: timeAt(0),
        endedAt: timeAt(9),
        parentRunId: "parent-run",
        memories: [],
        lastSeq: 9,
        result: '{"answer":"rain","n":2}',
        inputTokens: 220,
        outputTokens: 33,
        cacheReadTokens: 64,
        cacheCreationTokens: 20,
        reasoningTokens: 8,
        offset: readFileSync(join(runs, "run-succeeded", "log.jsonl")).length,
        waits: [],
        ready: false,
        wakeAt: null,
        unreadable: null,
      });
      expect(rows["run-failed"]).toMatchObject({
        status: "failed",
        endedAt: timeAt(4),
        parentRunId: null,
        lastSeq: 4,
        result: null,
        inputTokens: 10,
        outputTokens: 1,
        cacheReadTokens: 0,
        cacheCreationTokens: 0,
        reasoningTokens: 0,
      });
      expect(rows["run-stopped"]).toMatchObject({ status: "stopped", endedAt: null, lastSeq: 3, result: null });
      expect(rows["run-running"]).toMatchObject({ status: "running", endedAt: null, lastSeq: 3, inputTokens: 7, reasoningTokens: 1 });
      // The stored status is the one `inspectRun` reads from the records.
      for (const row of Object.values(rows)) {
        expect(inspectRun(join(runs, row.runId, "log.jsonl")).status).toBe(row.status);
      }
    } finally {
      held.close();
    }
    // The writer died: its lock is gone and nothing was appended. No update in between.
    const crashed = byId(readRunIndex(runs))["run-running"]!;
    expect(crashed.status).toBe("crashed");
    expect(inspectRun(join(runs, "run-running", "log.jsonl")).status).toBe("crashed");
  });

  test("the result is right when the END's commit was read in an earlier update than its success", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", succeededRun);
    const path = join(runs, "run-1", "log.jsonl");
    const full = readFileSync(path);
    const lines = full.toString("utf8").split("\n");
    // Cut the last line, the END's `node_succeeded`, keeping its `pool_commit`.
    writeFileSync(path, `${lines.slice(0, -2).join("\n")}\n`);
    updateRunIndex(runs);
    expect(readRunIndex(runs)[0]).toMatchObject({ status: "crashed", lastSeq: 8, result: null, endedAt: null });
    writeFileSync(path, full);
    updateRunIndex(runs);
    expect(readRunIndex(runs)[0]).toMatchObject({
      status: "succeeded",
      lastSeq: 9,
      result: '{"answer":"rain","n":2}',
      endedAt: timeAt(9),
    });
  });

  test("an update reads only what was appended since the last one", () => {
    const runs = freshRuns();
    writeRun(runs, "run-a", succeededRun);
    const growing = writeRun(runs, "run-b", runningRun, { keep: true });
    try {
      updateRunIndex(runs);
      const before = byId(readRunIndex(runs));
      const path = join(runs, "run-b", "log.jsonl");
      const size = readFileSync(path).length;
      growing.append(new NodeStepRecord("agent", 2, modelCall([5, 5, 5, 5, 5])));
      succeed(growing, "agent", { kind: "str", value: "ok" });
      const appended = readFileSync(path).length - size;
      const reads = spyOn(fs, "readSync");
      let bytes: number;
      try {
        updateRunIndex(runs);
        bytes = reads.mock.results.reduce((sum, result) => sum + (result.value as number), 0);
      } finally {
        reads.mockRestore();
      }
      // The appended lines, plus the byte before the saved offset that proves it ends a line.
      expect(bytes).toBe(appended + 1);
      const after = byId(readRunIndex(runs));
      expect(after["run-a"]).toEqual(before["run-a"]!);
      expect(after["run-b"]!.offset).toBe(size + appended);
      expect(after["run-b"]).toMatchObject({ status: "running", lastSeq: 6, inputTokens: 12, outputTokens: 7 });
    } finally {
      growing.close();
    }
  });

  test("a truncated or replaced log is rebuilt from 0, an emptied one loses its row; a deleted index rebuilds the same rows", () => {
    const runs = freshRuns();
    fourRuns(runs).close();
    updateRunIndex(runs);
    // Truncate below the saved offset: the run is back to its agent's first call.
    const succeeded = join(runs, "run-succeeded", "log.jsonl");
    const lines = readFileSync(succeeded, "utf8").split("\n");
    truncateSync(succeeded, Buffer.byteLength(`${lines.slice(0, 4).join("\n")}\n`));
    // Replace a log with a longer one of the same run whose records don't continue the old ones.
    const stopped = join(runs, "run-stopped", "log.jsonl");
    rmSync(stopped);
    writeRun(runs, "run-stopped-new", (log) => {
      start(log);
      log.append(new NodeStepRecord("x", 1, { kind: "note", text: "x".repeat(400) }));
      log.append(new NodeStartedRecord("other"));
      log.append(new NodeStartedRecord("third"));
    });
    const replacement = readFileSync(join(runs, "run-stopped-new", "log.jsonl"), "utf8").replaceAll(
      '"run-stopped-new"',
      '"run-stopped"',
    );
    rmSync(join(runs, "run-stopped-new"), { recursive: true });
    writeFileSync(stopped, replacement);
    // An emptied log has no row.
    truncateSync(join(runs, "run-failed", "log.jsonl"), 0);
    updateRunIndex(runs);
    const rows = byId(readRunIndex(runs));
    expect(Object.keys(rows).sort()).toEqual(["run-running", "run-stopped", "run-succeeded"]);
    expect(rows["run-succeeded"]).toMatchObject({ status: "crashed", lastSeq: 3, inputTokens: 100, endedAt: null, result: null });
    expect(rows["run-stopped"]).toMatchObject({ status: "crashed", lastSeq: 4 });
    rmSync(join(runs, "index.sqlite"));
    updateRunIndex(runs);
    expect(byId(readRunIndex(runs))).toEqual(rows);
  });

  test("a status agrees with inspectRun's record rules when nodes are still in flight", () => {
    const runs = freshRuns();
    // END succeeded while a side node is still in flight: not finished yet.
    writeRun(runs, "end-early", (log) => {
      start(log);
      log.append(new NodeStartedRecord("side"));
      log.append(new NodeStartedRecord(END_ID));
      succeed(log, END_ID, { kind: "str", value: "done" });
    });
    // A route proved END unreachable: failed once nothing else is in flight.
    const unreachable = new Failure({ code: "graph.terminal_unreachable", message: "", nodeId: "gate" });
    writeRun(runs, "unreachable", (log) => {
      start(log);
      log.append(new NodeStartedRecord("gate"));
      log.append(new NodeRoutedRecord("gate", "no", { terminalFailure: unreachable }));
    });
    // A stopped run that was resumed and crashed again.
    writeRun(runs, "resumed", (log) => {
      start(log);
      log.append(new NodeStartedRecord("agent"));
      log.append(new RunStoppedRecord("signal", "SIGINT", ["agent"]));
      log.append(new RunResumedRecord(["agent"], 0));
      log.append(new NodeStartedRecord("agent"));
    });
    updateRunIndex(runs);
    const rows = byId(readRunIndex(runs));
    expect([rows["end-early"]!.status, rows["unreachable"]!.status, rows["resumed"]!.status]).toEqual([
      "crashed",
      "failed",
      "crashed",
    ]);
    for (const row of Object.values(rows)) {
      expect(inspectRun(join(runs, row.runId, "log.jsonl")).status).toBe(row.status);
    }
    // The side node finishes in a later update: the END's earlier success now finishes the run.
    const log = JsonlRunLog.resume(join(runs, "end-early", "log.jsonl"));
    succeed(log, "side", { kind: "str", value: "x" });
    log.close();
    updateRunIndex(runs);
    expect(byId(readRunIndex(runs))["end-early"]).toMatchObject({ status: "succeeded", result: '"done"' });
  });

  test("a resumed run shows running while its writer holds the lock, and stopped once it lets go", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", stoppedRun);
    updateRunIndex(runs);
    const log = JsonlRunLog.resume(join(runs, "run-1", "log.jsonl"));
    try {
      expect(readRunIndex(runs)[0]!.status).toBe("running");
    } finally {
      log.close();
    }
    expect(readRunIndex(runs)[0]!.status).toBe("stopped");
  });

  test("a log of unreadable content keeps a row marked unreadable, a file error is skipped, each with a warning; a gone run loses its row; no index reads as []", () => {
    const runs = freshRuns();
    expect(readRunIndex(runs)).toEqual([]);
    writeRun(runs, "good", succeededRun);
    writeRun(runs, "gone", failedRun);
    mkdirSync(join(runs, "damaged"));
    writeFileSync(join(runs, "damaged", "log.jsonl"), "not a run log\n");
    // A file error is skipped, not marked.
    mkdirSync(join(runs, "folder-log", "log.jsonl"), { recursive: true });
    // A run folder reached through a symlink is a run folder.
    const elsewhere = freshRuns();
    writeRun(elsewhere, "linked", failedRun);
    symlinkSync(join(elsewhere, "linked"), join(runs, "linked"));
    const warnings: string[] = [];
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    const rows = byId(readRunIndex(runs));
    expect(Object.keys(rows).sort()).toEqual(["damaged", "gone", "good", "linked"]);
    expect(warnings).toHaveLength(2);
    expect(warnings[0]).toStartWith(`run index skipped ${join(runs, "damaged", "log.jsonl")}: run log `);
    expect(warnings[1]).toStartWith(`run index skipped ${join(runs, "folder-log", "log.jsonl")}: `);
    // No record was read, so the row knows nothing but where the log stops being readable.
    expect(rows["damaged"]).toMatchObject({ flow: null, startedAt: null, endedAt: null, lastSeq: null, offset: 0, result: null });
    expect(rows["damaged"]!.unreadable).toEqual({ seq: 0, reason: warnings[0]!.slice(warnings[0]!.indexOf(": ") + 2) });
    expect(rows["good"]!.unreadable).toBeNull();
    // The index holds every run's result, so it is as private as the run folders.
    expect(statSync(join(runs, "index.sqlite")).mode & 0o777).toBe(0o600);
    rmSync(join(runs, "gone"), { recursive: true });
    warnings.length = 0;
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    // An unreadable log that hasn't changed isn't read or warned about again; a file error is tried again.
    expect(warnings).toEqual([expect.stringContaining(join(runs, "folder-log", "log.jsonl"))]);
    expect(readRunIndex(runs).map((row) => row.runId)).toEqual(["damaged", "good", "linked"]);
    // A log fixed since is read again.
    rmSync(join(runs, "damaged"), { recursive: true });
    writeRun(runs, "damaged", failedRun);
    updateRunIndex(runs);
    expect(byId(readRunIndex(runs))["damaged"]).toMatchObject({ status: "failed", unreadable: null, lastSeq: 4 });
  });

  test("a usage count that isn't an int counts as 0, so one odd step can't break the index", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", (log) => {
      start(log);
      log.append(new NodeStartedRecord("agent"));
      const odd = {
        input_tokens: "12",
        output_tokens: 1.5,
        cache_read_tokens: Infinity,
        cache_creation_tokens: -0.5,
        reasoning_tokens: 2n ** 60n,
      };
      log.append(new NodeStepRecord("agent", 1, { kind: "model_call", usage: odd }));
      log.append(new NodeStepRecord("agent", 2, modelCall([5, 6, 7, 9, 8])));
    });
    updateRunIndex(runs);
    expect(readRunIndex(runs)[0]).toMatchObject({
      inputTokens: 5,
      outputTokens: 6,
      cacheReadTokens: 7,
      cacheCreationTokens: 9,
      reasoningTokens: 8,
    });
  });

  test("maxLineBytes marks a log with a longer line unreadable at it, reads it again only when it or the limit grows, and a bad one throws", () => {
    const runs = freshRuns();
    writeRun(runs, "good", succeededRun);
    writeRun(runs, "big", (log) => {
      start(log);
      log.append(new NodeStartedRecord("agent"));
      log.append(new NodeStepRecord("agent", 1, { kind: "note", text: "x".repeat(10_000) }));
    });
    const bigLog = join(runs, "big", "log.jsonl");
    const warnings: string[] = [];
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs, { maxLineBytes: 4096 }));
    const tooLong = /^run log .* line 4 \(seq 3\): the line is \d+ bytes, over maxLineBytes 4096$/;
    expect(warnings).toEqual([expect.stringMatching(/^run index skipped .*big\/log\.jsonl: run log /)]);
    // The row holds the records before the long line, and where it stopped.
    const big = byId(readRunIndex(runs))["big"]!;
    expect(big).toMatchObject({ status: "crashed", startedAt: timeAt(0), lastSeq: 2, endedAt: null });
    expect(big.unreadable).toEqual({ seq: 3, reason: expect.stringMatching(tooLong) });
    expect(byId(readRunIndex(runs))["good"]!.unreadable).toBeNull();
    // Unchanged, the log isn't read again.
    warnings.length = 0;
    const reads = spyOn(fs, "readSync");
    try {
      withWarnings((message) => warnings.push(message), () => updateRunIndex(runs, { maxLineBytes: 4096 }));
      expect(reads).not.toHaveBeenCalled();
    } finally {
      reads.mockRestore();
    }
    expect(warnings).toEqual([]);
    expect(byId(readRunIndex(runs))["big"]).toEqual(big);
    // Grown, it is read again, and still stops at the long line.
    fs.appendFileSync(bigLog, "more\n");
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs, { maxLineBytes: 4096 }));
    expect(warnings).toHaveLength(1);
    // Read on from the saved row, the reason names the line by its byte offset.
    expect(byId(readRunIndex(runs))["big"]!.unreadable).toEqual({ seq: 3, reason: expect.stringMatching(/\(seq 3\): the line is \d+ bytes/) });
    // A lower limit can't read more, so it doesn't read again; a higher one reads to the damaged line.
    warnings.length = 0;
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs, { maxLineBytes: 1024 }));
    expect(warnings).toEqual([]);
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    expect(warnings).toHaveLength(1);
    expect(byId(readRunIndex(runs))["big"]).toMatchObject({ lastSeq: 3, unreadable: { seq: 4 } });
    expect(() => updateRunIndex(runs, { maxLineBytes: 0 })).toThrow("maxLineBytes must be a positive integer, got 0");
  });

  test.skipIf(process.getuid?.() === 0)("a log yian may not open keeps the row it had, and is read once it may, with its size and mtime unchanged", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", stoppedRun);
    updateRunIndex(runs);
    const [before] = readRunIndex(runs);
    const path = join(runs, "run-1", "log.jsonl");
    const reopened = JsonlRunLog.resume(path, { clock: steppingClock() });
    reopened.append(new RunResumedRecord(["agent"], 0));
    reopened.close();
    fs.chmodSync(path, 0o000);
    const warnings: string[] = [];
    try {
      withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
      expect(warnings).toEqual([expect.stringMatching(new RegExp(`^run index skipped ${path}: .*EACCES`))]);
      expect(readRunIndex(runs)).toEqual([before!]);
    } finally {
      // A chmod changes neither the size nor the mtime.
      fs.chmodSync(path, 0o600);
    }
    warnings.length = 0;
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    expect(warnings).toEqual([]);
    expect(readRunIndex(runs)[0]).toMatchObject({ lastSeq: 4, unreadable: null });
  });

  test.skipIf(process.platform === "win32")("a FIFO or a /dev/zero symlink at a run's log is skipped with one warning each, and the other runs are listed", () => {
    const runs = freshRuns();
    writeRun(runs, "good", succeededRun);
    const fifo = join(runs, "fifo", "log.jsonl");
    mkdirSync(join(runs, "fifo"));
    execFileSync("mkfifo", [fifo]);
    const zero = join(runs, "zero", "log.jsonl");
    mkdirSync(join(runs, "zero"));
    symlinkSync("/dev/zero", zero);
    const url = (file: string) => JSON.stringify(pathToFileURL(join(TS_ROOT, file)).href);
    // In a child, so a scan that hangs fails at the timeout instead of stalling this process.
    const child = `
const { readRunIndex, updateRunIndex } = await import(${url("src/run.ts")});
const { withWarnings } = await import(${url("src/platform.ts")});
const warnings = [];
withWarnings((message) => warnings.push(message), () => updateRunIndex(${JSON.stringify(runs)}));
console.log(JSON.stringify({ warnings, runs: readRunIndex(${JSON.stringify(runs)}).map((row) => row.runId) }));
`;
    const result = spawnSync(process.execPath, ["-e", child], { cwd: TS_ROOT, encoding: "utf8", timeout: 10_000, killSignal: "SIGKILL" });
    expect([result.status, result.signal, result.stderr]).toEqual([0, null, ""]);
    expect(JSON.parse(result.stdout)).toEqual({
      warnings: [
        `run index skipped ${fifo}: EINVAL: not a regular file, open '${fifo}'`,
        `run index skipped ${zero}: EINVAL: not a regular file, open '${zero}'`,
      ],
      runs: ["good"],
    });
  }, 15_000);

  test.skipIf(process.platform === "win32")("a symlink at a run's log, even to a valid log of that run, is skipped with a warning and has no row", () => {
    const runs = freshRuns();
    writeRun(runs, "good", succeededRun);
    const elsewhere = freshRuns();
    writeRun(elsewhere, "linked", failedRun);
    const link = join(runs, "linked", "log.jsonl");
    mkdirSync(join(runs, "linked"));
    symlinkSync(join(elsewhere, "linked", "log.jsonl"), link);
    const warnings: string[] = [];
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    expect(warnings).toEqual([`run index skipped ${link}: EINVAL: not a regular file, open '${link}'`]);
    expect(readRunIndex(runs).map((row) => row.runId)).toEqual(["good"]);
  });

  test.skipIf(process.platform === "win32")("a log replaced by a FIFO keeps the row it had, and the scan never opens it", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", stoppedRun);
    updateRunIndex(runs);
    const before = readRunIndex(runs);
    const path = join(runs, "run-1", "log.jsonl");
    rmSync(path);
    execFileSync("mkfifo", [path]);
    const realOpen = fs.openSync;
    const opened: string[] = [];
    // Refused here rather than opened, so a scan that would open the FIFO fails instead of hanging.
    const openSpy = spyOn(fs, "openSync").mockImplementation(((file: fs.PathLike, ...rest: unknown[]) => {
      if (file === path) {
        opened.push(String(file));
        throw new Error("the scan opened the FIFO");
      }
      return Reflect.apply(realOpen, fs, [file, ...rest]);
    }) as typeof fs.openSync);
    const warnings: string[] = [];
    try {
      withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    } finally {
      openSpy.mockRestore();
    }
    expect(opened).toEqual([]);
    expect(warnings).toEqual([`run index skipped ${path}: EINVAL: not a regular file, open '${path}'`]);
    expect(readRunIndex(runs)).toEqual(before);
  });

  test("the reason stored for an unreadable log is cut to 200 characters", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", stoppedRun);
    const path = join(runs, "run-1", "log.jsonl");
    // A kind of 300 characters, which the error message quotes.
    writeFileSync(path, readFileSync(path, "utf8").replace('"kind":"run_started"', `"kind":"${"k".repeat(300)}"`));
    const warnings: string[] = [];
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    const { reason } = readRunIndex(runs)[0]!.unreadable!;
    expect([...reason]).toHaveLength(200);
    expect(reason).toStartWith(`run log ${path} line 1 (seq 0): run-record kind 'kkk`);
    expect(reason).toEndWith("k…");
    expect(warnings).toEqual([`run index skipped ${path}: ${reason}`]);
  });

  test("a log that turns unreadable keeps the row it had, marked unreadable at the bad line", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", stoppedRun);
    updateRunIndex(runs);
    const [before] = readRunIndex(runs);
    fs.appendFileSync(join(runs, "run-1", "log.jsonl"), "not a run record\n");
    const warnings: string[] = [];
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    expect(warnings).toHaveLength(1);
    const [after] = readRunIndex(runs);
    expect(after).toEqual({ ...before!, unreadable: { seq: 4, reason: expect.stringContaining("(seq 4)") } });
  });

  test("an index of another version is rebuilt", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", failedRun);
    const db = openSqlite(join(runs, "index.sqlite"));
    db.exec("CREATE TABLE runs (run_id TEXT PRIMARY KEY, junk TEXT); PRAGMA user_version = 99");
    db.close();
    expect(readRunIndex(runs)).toEqual([]);
    updateRunIndex(runs);
    expect(readRunIndex(runs).map((row) => [row.runId, row.status])).toEqual([["run-1", "failed"]]);
  });

  test("an index of this version whose runs table is missing or misshaped reads as [] and is rebuilt", () => {
    for (const damage of ["DROP TABLE runs", "ALTER TABLE runs DROP COLUMN memories"]) {
      const runs = freshRuns();
      writeRun(runs, "run-1", failedRun);
      updateRunIndex(runs);
      const db = openSqlite(join(runs, "index.sqlite"));
      db.exec(damage);
      db.close();
      expect(readRunIndex(runs)).toEqual([]);
      updateRunIndex(runs);
      expect(readRunIndex(runs).map((row) => [row.runId, row.status])).toEqual([["run-1", "failed"]]);
    }
  });

  test("an index built under another run-log schema is rebuilt, so its rows are a fresh build's", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", (log) => {
      start(log);
      log.append(new RunCancelledRecord());
    });
    updateRunIndex(runs);
    expect(readRunIndex(runs).map((row) => [row.runId, row.status])).toEqual([["run-1", "cancelled"]]);
    // The log now reads as an older schema, its size unchanged, so a plain update skips it.
    const path = join(runs, "run-1", "log.jsonl");
    const old = `"schema":${SCHEMA_VERSION - 1}`;
    writeFileSync(path, readFileSync(path, "utf8").replaceAll(`"schema":${SCHEMA_VERSION}`, old));
    updateRunIndex(runs);
    expect(readRunIndex(runs).map((row) => row.status)).toEqual(["cancelled"]);
    // The index as a yian of the older schema left it is rebuilt, as deleting it would rebuild it.
    const db = openSqlite(join(runs, "index.sqlite"));
    db.exec(`PRAGMA user_version = ${7 * 1000 + SCHEMA_VERSION - 1}`);
    db.close();
    expect(readRunIndex(runs)).toEqual([]);
    withWarnings(() => {}, () => updateRunIndex(runs));
    const rebuilt = readRunIndex(runs);
    rmSync(join(runs, "index.sqlite"));
    withWarnings(() => {}, () => updateRunIndex(runs));
    expect(readRunIndex(runs)).toEqual(rebuilt);
    expect(rebuilt.map((row) => row.unreadable?.reason)).toEqual([expect.stringContaining("predates schema")]);
  });

  test("an index that isn't a database reads as [] with a warning; an update warns once and rebuilds it private", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", failedRun);
    const index = join(runs, "index.sqlite");
    writeFileSync(index, "garbage");
    const warnings: string[] = [];
    expect(withWarnings((message) => warnings.push(message), () => readRunIndex(runs))).toEqual([]);
    expect(warnings).toEqual([`run index ${index} is damaged (file is not a database); the next update rebuilds it`]);
    warnings.length = 0;
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    expect(warnings).toEqual([`run index ${index} was damaged (file is not a database); rebuilt it from the logs`]);
    expect(readRunIndex(runs).map((row) => [row.runId, row.status])).toEqual([["run-1", "failed"]]);
    expect(statSync(index).mode & 0o777).toBe(0o600);
    warnings.length = 0;
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    expect(warnings).toEqual([]);
  });

  test("an index cut short is rebuilt with the rows it had, and a stale journal beside it is removed", () => {
    const runs = freshRuns();
    for (let i = 0; i < 40; i++) writeRun(runs, `run-${String(i).padStart(2, "0")}`, succeededRun);
    updateRunIndex(runs);
    const before = readRunIndex(runs);
    const index = join(runs, "index.sqlite");
    expect(statSync(index).size).toBeGreaterThanOrEqual(8 * 4096);
    truncateSync(index, Math.floor(statSync(index).size / 2 / 4096) * 4096);
    writeFileSync(`${index}-journal`, "stale journal");
    const warnings: string[] = [];
    withWarnings((message) => warnings.push(message), () => updateRunIndex(runs));
    expect(warnings).toEqual([expect.stringMatching(/^run index .* was damaged \(.*\); rebuilt it from the logs$/)]);
    expect(readRunIndex(runs)).toEqual(before);
    expect(existsSync(`${index}-journal`)).toBe(false);
  });

  test("an index of version 3, whose scans hold no remembered Stops, is rebuilt", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", failedRun);
    updateRunIndex(runs);
    const db = openSqlite(join(runs, "index.sqlite"));
    db.exec("UPDATE runs SET scan = json_remove(scan, '$.waits.stops'); PRAGMA user_version = 3");
    db.close();
    // Another layout version reads as no rows until an update rebuilds it.
    expect(readRunIndex(runs)).toEqual([]);
    updateRunIndex(runs);
    expect(readRunIndex(runs).map((row) => [row.runId, row.status])).toEqual([["run-1", "failed"]]);
    const reader = openSqlite(join(runs, "index.sqlite"), { readonly: true });
    try {
      const { scan } = reader.prepare("SELECT scan FROM runs").get() as { scan: string };
      expect(JSON.parse(scan).waits.stops).toEqual([]);
      expect(reader.prepare("PRAGMA user_version").get()).toEqual({ user_version: 7 * 1000 + SCHEMA_VERSION });
    } finally {
      reader.close();
    }
  });

  test("an index of version 4, which has no cache-creation column, is rebuilt with the count", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", succeededRun);
    updateRunIndex(runs);
    const db = openSqlite(join(runs, "index.sqlite"));
    db.exec("ALTER TABLE runs DROP COLUMN cache_creation_tokens; PRAGMA user_version = 4");
    db.close();
    expect(readRunIndex(runs)).toEqual([]);
    updateRunIndex(runs);
    expect(readRunIndex(runs).map((row) => [row.runId, row.cacheCreationTokens])).toEqual([["run-1", 20]]);
  });

  test("a run's memory identities come from its START inputs; a run without memories has none", async () => {
    const runs = freshRuns();
    await memoryRun(runs, "run-m");
    writeRun(runs, "run-1", failedRun);
    updateRunIndex(runs);
    const memories = Object.fromEntries(readRunIndex(runs).map((row) => [row.runId, row.memories]));
    // The Optional memory left out recorded none.
    expect(memories).toEqual({
      "run-1": [],
      "run-m": [{ input: "lessons", identity: "t/lessons", type: "FileMemory" }],
    });
  });

  test("only inputs the run's flow declares as memories are listed, whatever another input holds", async () => {
    const runs = freshRuns();
    const flow = await loadFlow(
      "id: m\nname: m\ntypedefs:\n  Cfg: {memory: object}\ninput:\n  lessons: !FileMemory\n" +
        "  notes: !Optional(FileMemory)\n  cfg: !object\n  typed: !Cfg\n  memory: !object\nnodes:\n" +
        "  add:\n    kind: tool\n" +
        "    tool_id: {lessons: append}\n    input: {path: MEMORY.md, text: hi}\noutput: ${add.output.path}\n",
    );
    // A memory value's form, given as plain data to inputs that aren't memories.
    const forged = {
      memory: {
        identity: "victim/answer/lessons",
        type: "FileMemory",
        settings: {},
        version: "v",
        recall: { text: "", cut: null, hint: null },
        tools: [],
      },
    };
    const lessons = new FileMemory({ root: join(runs, "..", "forged-lessons"), id: "t/lessons" });
    const notes = new FileMemory({ root: join(runs, "..", "forged-notes"), id: "t/notes" });
    const inputs = { lessons, notes, cfg: forged, typed: forged, memory: forged };
    const log = runLogPath(runs, "run-f");
    const result = await runFlow(flow, inputs, { log, runId: "run-f", env: () => undefined });
    expect(result.status).toBe("succeeded");
    updateRunIndex(runs);
    expect(readRunIndex(runs).map((row) => row.memories)).toEqual([
      [
        { input: "lessons", identity: "t/lessons", type: "FileMemory" },
        { input: "notes", identity: "t/notes", type: "FileMemory" },
      ],
    ]);
  });

  test("a START commit read by a later update than the run's start is read against the recorded flow", () => {
    const runs = freshRuns();
    const form = {
      memory: {
        identity: "t/lessons",
        type: "FileMemory",
        settings: {},
        version: "v",
        recall: { text: "", cut: null, hint: null },
        tools: [],
      },
    };
    const flow = "id: f\ninput:\n  lessons: !FileMemory\n  cfg: !object\nnodes: {}\n";
    const log = writeRun(
      runs,
      "run-1",
      (log) =>
        log.append(
          new RunStartedRecord({
            flowFingerprint: "a".repeat(64),
            startWiringOverride: {},
            effectiveLlmByStaticPath: {},
            numWorkers: 0,
            flow: {
              root: "flow.yaml",
              dir: "/flows",
              codeRoot: null,
              files: { "flow.yaml": { text: flow, refs: {} } },
            },
          }),
        ),
      { keep: true },
    );
    try {
      updateRunIndex(runs);
      expect(readRunIndex(runs)[0]!.memories).toEqual([]);
      log.append(new PoolCommitRecord(START_ID, START_ID, { kind: "object", value: { lessons: form, cfg: form } }));
    } finally {
      log.close();
    }
    updateRunIndex(runs);
    expect(readRunIndex(runs)[0]!.memories).toEqual([{ input: "lessons", identity: "t/lessons", type: "FileMemory" }]);
  });

  test("an index of version 6, which has no memories column, is rebuilt with them", async () => {
    const runs = freshRuns();
    await memoryRun(runs, "run-m");
    updateRunIndex(runs);
    const db = openSqlite(join(runs, "index.sqlite"));
    db.exec("ALTER TABLE runs DROP COLUMN memories; PRAGMA user_version = 6");
    db.close();
    expect(readRunIndex(runs)).toEqual([]);
    updateRunIndex(runs);
    expect(readRunIndex(runs).map((row) => row.memories)).toEqual([
      [{ input: "lessons", identity: "t/lessons", type: "FileMemory" }],
    ]);
  });

  test("a read takes the layout version and the rows from one snapshot, so a rebuild between them can't mix layouts", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", failedRun);
    updateRunIndex(runs);
    const path = join(runs, "index.sqlite");
    const reader = openSqlite(path, { readonly: true });
    try {
      // Another yian version rebuilds the index between the version check and the rows' read.
      const rebuild = () => {
        const writer = openSqlite(path);
        try {
          writer.exec("PRAGMA busy_timeout = 0");
          writer.exec("DROP TABLE runs; CREATE TABLE runs (run_id TEXT PRIMARY KEY, junk TEXT); PRAGMA user_version = 99");
        } catch {
          // The read's snapshot holds the file, so the rebuild waits for it.
        } finally {
          writer.close();
        }
      };
      const racing: typeof reader = {
        ...reader,
        prepare: (sql) => {
          if (sql.startsWith("SELECT")) rebuild();
          return reader.prepare(sql);
        },
      };
      expect(indexRows(racing).map((row) => row["run_id"])).toEqual(["run-1"]);
    } finally {
      reader.close();
    }
  });

  test("an update waits a bounded time for a held lock, then throws; reads go on meanwhile", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", failedRun);
    updateRunIndex(runs);
    const path = join(runs, "index.sqlite");
    const holder = openSqlite(path);
    try {
      holder.exec("BEGIN IMMEDIATE");
      const began = Date.now();
      let error: unknown;
      try {
        updateRunIndex(runs, { timeoutMs: 300 });
      } catch (caught) {
        error = caught;
      }
      const waited = Date.now() - began;
      expect(error).toBeInstanceOf(RunIndexBusyError);
      expect((error as Error).name).toBe("RunIndexBusyError");
      expect((error as Error).message).toBe(`run index ${path} is locked by another update; gave up after 300 ms`);
      expect(waited).toBeGreaterThanOrEqual(250);
      expect(waited).toBeLessThan(5000);
      expect(readRunIndex(runs).map((row) => row.runId)).toEqual(["run-1"]);
    } finally {
      holder.close();
    }
    updateRunIndex(runs, { timeoutMs: 300 });
    expect(() => updateRunIndex(runs, { timeoutMs: -1 })).toThrow(RangeError);
  });

  test("two builders in two processes at once both finish and leave the right rows", async () => {
    const runs = freshRuns();
    const count = 24;
    for (let i = 0; i < count; i++) {
      const id = `run-${String(i).padStart(2, "0")}`;
      writeRun(runs, id, [succeededRun, failedRun, stoppedRun][i % 3]!);
    }
    const script = join(tmp, "builder.mjs");
    writeFileSync(
      script,
      `const { updateRunIndex } = await import(${JSON.stringify(pathToFileURL(join(TS_ROOT, "src/run.ts")).href)});
for (let i = 0; i < 5; i++) updateRunIndex(process.argv[2], { timeoutMs: 20000 });
console.log("done");
`,
    );
    const builder = () =>
      new Promise<{ code: number | null; output: string }>((resolve, reject) => {
        const child = spawn(process.execPath, [script, runs], { cwd: TS_ROOT, stdio: ["ignore", "pipe", "pipe"] });
        let output = "";
        child.stdout.setEncoding("utf8").on("data", (chunk: string) => (output += chunk));
        child.stderr.setEncoding("utf8").on("data", (chunk: string) => (output += chunk));
        const timer = setTimeout(() => child.kill("SIGKILL"), 60_000);
        child.on("error", reject);
        child.on("close", (code) => {
          clearTimeout(timer);
          resolve({ code, output });
        });
      });
    const results = await Promise.all([builder(), builder()]);
    for (const { code, output } of results) expect([code, output.trim()]).toEqual([0, "done"]);
    const db = openSqlite(join(runs, "index.sqlite"), { readonly: true });
    try {
      expect(db.prepare("PRAGMA integrity_check").get()).toEqual({ integrity_check: "ok" });
    } finally {
      db.close();
    }
    const rows = readRunIndex(runs);
    expect(rows).toHaveLength(count);
    const statuses: RunIndexRow["status"][] = ["succeeded", "failed", "stopped"];
    for (const row of rows) expect(row.status).toBe(statuses[Number(row.runId.slice(4)) % 3]!);
    expect(existsSync(join(runs, "index.sqlite"))).toBe(true);
  }, 90_000);
});

describe("the run scan, shared by inspectRun and the index", () => {
  /** Fold every record of the log at `path` into a fresh scan. */
  function scanOf(path: string) {
    const scan = emptyRunScan();
    for (const envelope of readRunLog(path)) foldRunScan(scan, envelope);
    return scan;
  }

  test("plain data: nodes in flight by start seq, the last stop, crashed counts and deciding seqs", () => {
    const runs = freshRuns();
    const unreachable = new Failure({ code: "graph.terminal_unreachable", message: "", nodeId: "gate" });
    writeRun(runs, "run-1", (log) => {
      start(log);
      log.append(new NodeStartedRecord("a"));
      log.append(new NodeStartedRecord("b"));
      log.append(new RunStoppedRecord("signal", "SIGINT", ["a", "b"]));
      log.append(new RunResumedRecord(["a", "b"], 0));
      log.append(new NodeStartedRecord("b"));
      log.append(new NodeStartedRecord("gate"));
      // A node started again while in flight keeps its place, as a Map key does.
      log.append(new NodeStartedRecord("b"));
      log.append(new NodeRoutedRecord("gate", "no", { terminalFailure: unreachable }));
    });
    const path = join(runs, "run-1", "log.jsonl");
    const scan = scanOf(path);
    // The resume came after a stop that covered both starts, so it counts no crashed attempt.
    expect({ ...scan, waits: null }).toEqual({
      inFlight: [["b", 8]],
      lastStop: null,
      endSucceeded: false,
      crashed: [],
      nodeFailureSeq: null,
      unreachableSeq: 9,
      runFailureSeq: null,
      cancelled: false,
      waits: null,
    });
    expect(scan.waits.lastFrontier).toBe("node_routed");
    expect(runScanFromJson(JSON.stringify(scan))).toEqual(scan);
    expect(scanStatus(scan)).toBe("crashed");
    const inspection = inspectRun(path);
    expect([inspection.status, inspection.inFlight]).toEqual(["crashed", ["b"]]);
    expect([...inspection.crashedAttempts]).toEqual([["b", 1]]);
  });

  test("a resume after a crash counts the attempts it retries, as inspectRun does", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", (log) => {
      start(log);
      log.append(new NodeStartedRecord("a"));
      log.append(new NodeStartedRecord("b"));
      log.append(new RunResumedRecord(["a", "b"], 0));
      log.append(new NodeStartedRecord("a"));
      log.append(new NodeFailedRecord(new Failure({ code: "node.execute", message: "boom", nodeId: "a" })));
    });
    const path = join(runs, "run-1", "log.jsonl");
    const scan = scanOf(path);
    expect(scan.crashed).toEqual([
      ["a", 1],
      ["b", 1],
    ]);
    expect([scan.nodeFailureSeq, scanStatus(scan)]).toEqual([6, "failed"]);
    const inspection = inspectRun(path);
    expect([...inspection.crashedAttempts]).toEqual(scan.crashed);
    expect(inspection.failure?.message).toBe("boom");
  });

  test("the index stores the scan, not a status column", () => {
    const runs = freshRuns();
    writeRun(runs, "run-1", stoppedRun);
    updateRunIndex(runs);
    const db = openSqlite(join(runs, "index.sqlite"), { readonly: true });
    try {
      const columns = db.prepare("PRAGMA table_info(runs)").all().map((row) => row["name"]);
      expect(columns).not.toContain("status");
      const stored = db.prepare("SELECT scan FROM runs").get()!;
      expect(runScanFromJson(stored["scan"] as string)).toEqual(scanOf(join(runs, "run-1", "log.jsonl")));
    } finally {
      db.close();
    }
  });

  /** Fold `records` after the run start (seqs 0 and 1) into a fresh scan. */
  function folded(...records: RunRecord[]): RunScan {
    const scan = emptyRunScan();
    const all = [runStarted(), new PoolCommitRecord(START_ID, START_ID, { kind: "object", value: {} }), ...records];
    all.forEach((record, seq) => foldRunScan(scan, { seq, record }));
    return scan;
  }
  const takeWait = (node: string) =>
    new NodeWaitingRecord(node, [`r/${node}`], { waits: [{ callId: `r/${node}`, tool: "take", inbox: "msgs", mode: "one", deadline: null }] });
  const send = (value: string) => new PoolCommitRecord(START_ID, `${START_ID}.msgs`, { kind: "str", value });
  const unreachable = new Failure({ code: "graph.terminal_unreachable", message: "", nodeId: "gate" });
  const endSuccess = [new NodeStartedRecord(END_ID), new PoolCommitRecord(END_ID, END_ID, { kind: "str", value: "x" })];

  test("a waiting node leaves the nodes in flight; its wake puts it back at the wake's seq", () => {
    const waiting = folded(new NodeStartedRecord("heard"), takeWait("heard"), new RunWaitingRecord(["heard"]));
    expect(waiting.inFlight).toEqual([]);
    expect(scanStatus(waiting)).toBe("waiting");
    const woken = folded(new NodeStartedRecord("heard"), takeWait("heard"), new RunWaitingRecord(["heard"]), send("x"), new NodeWokenRecord("heard"));
    expect(woken.inFlight).toEqual([["heard", 6]]);
    expect(scanStatus(woken)).toBe("crashed");
    // A crash while nodes wait retries only the running ones.
    const resumed = folded(new NodeStartedRecord("a"), new NodeStartedRecord("heard"), takeWait("heard"), new RunResumedRecord(["a"], 0));
    expect(resumed.crashed).toEqual([["a", 1]]);
  });

  test.each<[string, RunRecord[], ReturnType<typeof scanStatus>]>([
    ["run_waiting after a wait", [new NodeStartedRecord("heard"), takeWait("heard"), new RunWaitingRecord(["heard"])], "waiting"],
    ["run_waiting, then a resume", [new NodeStartedRecord("heard"), takeWait("heard"), new RunWaitingRecord(["heard"]), new RunResumedRecord([], 2)], "crashed"],
    ["a stop after a wait", [new NodeStartedRecord("a"), new RunStoppedRecord("signal", "SIGINT", ["a"])], "stopped"],
    [
      "END succeeded while a node waits",
      [new NodeStartedRecord("heard"), takeWait("heard"), ...endSuccess, new NodeSucceededRecord(END_ID, 5), new RunWaitingRecord(["heard"])],
      "waiting",
    ],
    [
      "END succeeded while a node waits, then a wake and the log ends",
      [new NodeStartedRecord("heard"), takeWait("heard"), ...endSuccess, new NodeSucceededRecord(END_ID, 5), new RunWaitingRecord(["heard"]), send("x"), new NodeWokenRecord("heard")],
      "crashed",
    ],
    [
      "a route proves END unreachable while a node waits",
      [new NodeStartedRecord("heard"), takeWait("heard"), new NodeStartedRecord("gate"), new NodeRoutedRecord("gate", "no", { terminalFailure: unreachable }), new RunWaitingRecord(["heard"])],
      "waiting",
    ],
    [
      "then run_failed",
      [new NodeStartedRecord("heard"), takeWait("heard"), new NodeStartedRecord("gate"), new NodeRoutedRecord("gate", "no", { terminalFailure: unreachable }), new RunFailedRecord(unreachable)],
      "failed",
    ],
    [
      "a node failure, whatever else",
      [new NodeStartedRecord("heard"), takeWait("heard"), new NodeStartedRecord("a"), new NodeFailedRecord(new Failure({ code: "node.execute", message: "boom", nodeId: "a" }))],
      "failed",
    ],
    ["a cancel while a node waits", [new NodeStartedRecord("heard"), takeWait("heard"), new RunCancelledRecord()], "cancelled"],
    ["a cancel after a close", [new NodeStartedRecord("heard"), takeWait("heard"), new InboxClosedRecord("msgs"), new RunCancelledRecord()], "cancelled"],
    [
      "END succeeded and END proved unreachable, nothing in flight",
      [new NodeStartedRecord("gate"), new NodeRoutedRecord("gate", "no", { terminalFailure: unreachable }), ...endSuccess, new NodeSucceededRecord(END_ID, 5)],
      "succeeded",
    ],
  ])("scanStatus: %s", (_name, records, status) => {
    expect(scanStatus(folded(...records))).toBe(status);
  });

  test("the scan's waits fold round-trips through JSON", () => {
    const scan = folded(new NodeStartedRecord("heard"), takeWait("heard"), send("x"), new RunWaitingRecord(["heard"]), new RunCancelledRecord());
    expect(scan.waits).toBeInstanceOf(WaitsState);
    const copy = runScanFromJson(JSON.stringify(scan));
    expect(copy).toEqual(scan);
    expect(copy.waits).toBeInstanceOf(WaitsState);
    expect(copy.waits.calls.get("r/heard")).toMatchObject({ state: "answered", assigned: [4], closedReason: null });
    expect(copy.waits.lastFrontier).toBe("run_waiting");
  });
});

describe("waiting runs in the index", () => {
  /** Write the run `runId` of the run start then `records`; keep it open when `keep`. */
  function waitRun(runs: string, runId: string, records: readonly RunRecord[], { keep = false } = {}): JsonlRunLog {
    return writeRun(
      runs,
      runId,
      (log) => {
        for (const record of [...runStart(), ...records]) log.append(record);
      },
      { keep },
    );
  }
  const WAITING = [new NodeStartedRecord("heard"), takeWaitOn("heard"), new RunWaitingRecord(["heard"])];

  test("a waiting take: waiting, not ready, no wake time; a send makes only its row ready, reading only the send", () => {
    const runs = freshRuns();
    writeRun(runs, "run-a", succeededRun);
    const log = waitRun(runs, "run-w", WAITING, { keep: true });
    try {
      log.close();
      updateRunIndex(runs);
      const before = byId(readRunIndex(runs));
      expect(before["run-w"]).toMatchObject({
        status: "waiting",
        endedAt: null,
        ready: false,
        wakeAt: null,
        waits: [{ node: "heard", callId: "r/heard", tool: "take", inbox: "msgs", deadline: null, settled: false }],
      });
      const path = join(runs, "run-w", "log.jsonl");
      const size = readFileSync(path).length;
      const reopened = JsonlRunLog.resume(path, { clock: steppingClock() });
      reopened.append(sendTo("hi"));
      reopened.close();
      const appended = readFileSync(path).length - size;
      const reads = spyOn(fs, "readSync");
      let bytes: number;
      try {
        updateRunIndex(runs);
        bytes = reads.mock.results.reduce((sum, result) => sum + (result.value as number), 0);
      } finally {
        reads.mockRestore();
      }
      expect(bytes).toBe(appended + 1);
      const after = byId(readRunIndex(runs));
      expect(after["run-a"]).toEqual(before["run-a"]!);
      expect(after["run-w"]).toMatchObject({ status: "waiting", ready: true, wakeAt: null, lastSeq: 5 });
      expect(after["run-w"]!.waits[0]!.settled).toBe(true);
    } finally {
      log.close();
    }
  });

  test("a sleep gives a wake time; outside waiting there is none and nothing is ready", () => {
    const runs = freshRuns();
    waitRun(runs, "run-nap", [new NodeStartedRecord("nap"), sleepWait("nap"), new RunWaitingRecord(["nap"])]);
    waitRun(runs, "run-woken", [new NodeStartedRecord("nap"), sleepWait("nap"), new RunWaitingRecord(["nap"]), new NodeWokenRecord("nap", ["r/nap"])]);
    updateRunIndex(runs);
    const rows = byId(readRunIndex(runs));
    expect(rows["run-nap"]).toMatchObject({ status: "waiting", ready: false, wakeAt: DEADLINE });
    expect(rows["run-woken"]).toMatchObject({ status: "crashed", ready: false, wakeAt: null, waits: [] });
  });

  test("cancelled and run_failed runs end with an end time; a woken take after END's success reads crashed", () => {
    const runs = freshRuns();
    for (const [name, records] of Object.entries(WAIT_LOGS)) waitRun(runs, name.toLowerCase().replaceAll(/[^a-z]+/g, "-"), records);
    updateRunIndex(runs);
    const rows = byId(readRunIndex(runs));
    expect(rows["two-takes-a-close-then-a-cancel"]).toMatchObject({ status: "cancelled", ready: false, waits: [] });
    expect(rows["two-takes-a-close-then-a-cancel"]!.endedAt).not.toBeNull();
    expect(rows["end-unreachable-while-a-take-waits-then-run-failed"]).toMatchObject({ status: "failed" });
    expect(rows["end-unreachable-while-a-take-waits-then-run-failed"]!.endedAt).not.toBeNull();
    expect(rows["end-succeeds-while-a-take-waits-the-woken-take-crashes"]).toMatchObject({ status: "crashed", endedAt: null });
    // Deleting the index and building it again gives the same rows.
    rmSync(join(runs, "index.sqlite"));
    updateRunIndex(runs);
    expect(byId(readRunIndex(runs))).toEqual(rows);
  });

  test("while a writer holds the lock, a waiting row reads running and a terminal one keeps its status; inspectRun reads both running", () => {
    const runs = freshRuns();
    const waiting = waitRun(runs, "run-w", WAITING, { keep: true });
    const ended = waitRun(runs, "run-x", [...WAITING, new RunCancelledRecord()], { keep: true });
    try {
      updateRunIndex(runs);
      const rows = byId(readRunIndex(runs));
      expect(rows["run-w"]).toMatchObject({ status: "running", ready: false, wakeAt: null, waits: [] });
      expect(rows["run-x"]!.status).toBe("cancelled");
      // The intended difference: inspectRun reads any locked log as running, since its writer may
      // still be appending; the index overlays only runs that haven't ended.
      expect(inspectRun(join(runs, "run-x", "log.jsonl")).status).toBe("running");
    } finally {
      waiting.close();
      ended.close();
    }
    const released = byId(readRunIndex(runs));
    expect([released["run-w"]!.status, released["run-x"]!.status]).toEqual(["waiting", "cancelled"]);
  });

  test("lockProbe: false reads each row's recorded status, never running, even while a writer holds it", () => {
    const runs = freshRuns();
    writeRun(runs, "run-s", stoppedRun);
    const waiting = waitRun(runs, "run-w", WAITING, { keep: true });
    try {
      updateRunIndex(runs);
      const held = byId(readRunIndex(runs, { lockProbe: false }));
      expect([held["run-s"]!.status, held["run-w"]!.status]).toEqual(["stopped", "waiting"]);
      expect(held["run-w"]).toMatchObject({ waits: [expect.objectContaining({ node: "heard" })] });
      expect(byId(readRunIndex(runs, { lockProbe: true }))["run-w"]!.status).toBe("running");
    } finally {
      waiting.close();
    }
    expect(() => readRunIndex(runs, { lockProbe: "no" as never })).toThrow(
      new RangeError('lockProbe must be a boolean, got "no"'),
    );
  });

  test.each(Object.entries(WAIT_LOGS))("the index and inspectRun agree at every cut: %s", (_name, scenario) => {
    const records = [...runStart(), ...scenario];
    for (let k = 1; k <= records.length; k++) {
      const runs = freshRuns();
      const log = JsonlRunLog.create(runLogPath(runs, "run-1"), "run-1", { clock: steppingClock() });
      for (const record of records.slice(0, k)) log.append(record);
      log.close();
      updateRunIndex(runs);
      const [row] = readRunIndex(runs);
      const inspection = inspectRun(join(runs, "run-1", "log.jsonl"));
      const parts = ({ status, ready, wakeAt, waits, endedAt }: typeof row & {}) => ({ status, ready, wakeAt, waits, endedAt });
      expect(parts(row!)).toEqual(parts(inspection as never));
    }
  });

  test("a stored scan that can't be read is rebuilt from 0", () => {
    const runs = freshRuns();
    waitRun(runs, "run-w", WAITING);
    updateRunIndex(runs);
    const before = readRunIndex(runs);
    const db = openSqlite(join(runs, "index.sqlite"));
    try {
      db.exec(`UPDATE runs SET scan = '{"inFlight": "junk"}', "offset" = 0`);
    } finally {
      db.close();
    }
    const warnings: string[] = [];
    expect(withWarnings((message) => warnings.push(message), () => readRunIndex(runs))).toEqual([]);
    expect(warnings).toEqual([expect.stringContaining("run index skipped run-w: run scan inFlight must be")]);
    updateRunIndex(runs);
    expect(readRunIndex(runs)).toEqual(before);
  });
});

