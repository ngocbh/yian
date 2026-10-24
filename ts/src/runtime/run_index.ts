/**
 * The run index: `<runs>/index.sqlite`, one row per run folder `<runs>/<run_id>/log.jsonl`.
 *
 * The logs are the only source of truth; the index is a view built from them. `updateRunIndex`
 * reads each log on from the byte offset its row reached, so it may lag the logs but never leads
 * them, and deleting the index and building it again gives the same rows. The stored status comes
 * from the records alone; `readRunIndex` puts `running` on top while a writer holds a run's lock.
 *
 * Knows about: compile (the END id), diagnostics, platform, runtime/log, typesys.
 */

import fs from "node:fs";
import { join } from "node:path";

import { END_ID } from "../compile/model.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { openSqlite, type Sqlite, type SqliteRow, sqliteErrorCode, warn } from "../platform.ts";
import { logJson, parseJson } from "../typesys/json.ts";
import { isRecord, valueRepr } from "../typesys/values.ts";
import {
  isRunLogLocked,
  iterRunRecords,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  PoolCommitRecord,
  RunLogChangedError,
  runRecordOffsets,
  RunResumedRecord,
  RunStartedRecord,
  RunStoppedRecord,
} from "./log.ts";

/** The index file's name in the runs folder. */
const INDEX_FILE = "index.sqlite";

/** The index layout's version, kept in `PRAGMA user_version`; an index of another is rebuilt. */
const INDEX_VERSION = 1;

/** How long an update or read waits for SQLite's lock by default, in milliseconds. */
const DEFAULT_TIMEOUT_MS = 5000;

// `scan` holds the builder's carried state (see `RunScan`), so an update goes on where the last
// one stopped and the stored status follows the same rules as `inspectRun`'s record scan.
const SCHEMA = `CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  flow TEXT,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  parent_run_id TEXT,
  last_seq INTEGER NOT NULL,
  result TEXT,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  cache_read_tokens INTEGER NOT NULL,
  reasoning_tokens INTEGER NOT NULL,
  "offset" INTEGER NOT NULL,
  scan TEXT NOT NULL
)`;

const COLUMNS = [
  "run_id",
  "flow",
  "status",
  "started_at",
  "ended_at",
  "parent_run_id",
  "last_seq",
  "result",
  "input_tokens",
  "output_tokens",
  "cache_read_tokens",
  "reasoning_tokens",
  '"offset"',
  "scan",
];

const SQLITE_BUSY = 5;

/** A run's status in the index; `running` is only ever put on top at read time. */
export type RunIndexStatus = "running" | "succeeded" | "failed" | "stopped" | "crashed";

/** One run as the index holds it. */
export interface RunIndexRow {
  /** The run's id, its folder's name. */
  readonly runId: string;
  /** The path of the flow file the run started from, or `null` for a flow built in code. */
  readonly flow: string | null;
  /**
   * `succeeded` or `failed` once the records show the run finished; `stopped` when the host
   * stopped the last segment; `crashed` otherwise. `running` replaces `stopped` or `crashed` while
   * a writer holds the log's lock.
   */
  readonly status: RunIndexStatus;
  /** The time of the run's `run_started` record. */
  readonly startedAt: string;
  /** For a `succeeded` or `failed` run, the time of the log's last record; otherwise `null`. */
  readonly endedAt: string | null;
  /** The id of the run that started this one (`run_started.parent`), or `null`. */
  readonly parentRunId: string | null;
  /** The seq of the last record the index read. */
  readonly lastSeq: number;
  /**
   * The run's result as a run-log line (`logJson`): the value of the `pool_commit` the top-level
   * END's success names; `null` until the END succeeded.
   */
  readonly result: string | null;
  /** The sums of every `model_call` step's token counts, failed attempts included. */
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly cacheReadTokens: number;
  readonly reasoningTokens: number;
  /** The byte offset the row was built up to: the end of the record at `lastSeq`. */
  readonly offset: number;
}

/**
 * `updateRunIndex` gave up waiting for the index's lock: another update held it for longer than
 * the timeout.
 */
export class RunIndexBusyError extends Error {
  override name = "RunIndexBusyError";
}

/** What the builder carries from one update to the next, beyond the row's columns. */
interface RunScan {
  /** The nodes started and not settled, in start order. */
  inFlight: string[];
  /** Whether the top-level END succeeded. */
  endSucceeded: boolean;
  /** Whether a node failed. */
  nodeFailed: boolean;
  /** Whether a route proved END unreachable. */
  unreachable: boolean;
  /** Whether the last segment was stopped. */
  stopped: boolean;
}

/** A row while it is built. */
interface Building {
  runId: string;
  flow: string | null;
  startedAt: string;
  endedAt: string | null;
  parentRunId: string | null;
  lastSeq: number;
  result: string | null;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  reasoningTokens: number;
  offset: number;
  scan: RunScan;
}

/** Return `timeoutMs` when it is a non-negative safe integer. */
function checkedTimeout(timeoutMs: unknown): number {
  if (typeof timeoutMs !== "number" || !Number.isSafeInteger(timeoutMs) || timeoutMs < 0) {
    throw new RangeError(`timeoutMs must be a non-negative integer, got ${valueRepr(timeoutMs)}`);
  }
  return timeoutMs;
}

/**
 * Bring `<runsDir>/index.sqlite` up to date with every run log `<runsDir>/<run_id>/log.jsonl`,
 * creating the index when it is missing.
 *
 * Each log is read on from the byte offset its row reached (`iterRunRecords` with `afterOffset`,
 * `runId` and `afterSeq`); a log that hasn't grown isn't opened. A log that was truncated or
 * replaced (`RunLogChangedError`) is read again from the start. A run folder with no log, or a log
 * with no complete record yet, has no row; a row whose log is gone is deleted. A log that can't be
 * read (an old schema, a damaged line, a first record that isn't `run_started` of the folder's run)
 * raises a warning through `warn` and keeps its row as it was. An index of another layout version
 * is dropped and built again.
 *
 * The update is one SQLite transaction (`BEGIN IMMEDIATE`), so builders in any process take turns.
 * It waits at most `timeoutMs` for another builder's lock, then throws. Synchronous: it blocks its
 * thread while it waits and reads.
 *
 * @param runsDir The runs folder.
 * @param options.timeoutMs How long to wait for the index's lock, in milliseconds; 5000 by default.
 * @throws RunIndexBusyError when another update holds the lock longer than `timeoutMs`.
 * @throws RangeError when `timeoutMs` isn't a non-negative integer.
 * @throws Error (code `ENOENT`) when `runsDir` doesn't exist, and SQLite's error when the index
 *   file isn't a database.
 */
export function updateRunIndex(runsDir: string, options: { timeoutMs?: number } = {}): void {
  const timeoutMs = checkedTimeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const path = join(runsDir, INDEX_FILE);
  const runIds = runFolders(runsDir);
  const db = openSqlite(path);
  try {
    db.exec(`PRAGMA busy_timeout = ${timeoutMs}`);
    try {
      db.exec("BEGIN IMMEDIATE");
    } catch (error) {
      throw busyOr(error, path, timeoutMs);
    }
    try {
      prepareSchema(db);
      const select = db.prepare(`SELECT * FROM runs WHERE run_id = ?`);
      const upsert = db.prepare(
        `INSERT OR REPLACE INTO runs (${COLUMNS.join(", ")}) VALUES (${COLUMNS.map(() => "?").join(", ")})`,
      );
      const kept = new Set<string>();
      for (const runId of runIds) {
        const log = join(runsDir, runId, "log.jsonl");
        const size = fs.statSync(log, { throwIfNoEntry: false })?.size;
        if (size === undefined) continue;
        const stored = select.get(runId);
        if (stored !== undefined) kept.add(runId);
        const saved = stored === undefined ? null : fromColumns(stored);
        if (saved !== null && saved.offset === size) continue;
        let built: Building | null;
        try {
          try {
            built = readRun(log, runId, saved);
          } catch (error) {
            if (!(error instanceof RunLogChangedError) || saved === null) throw error;
            built = readRun(log, runId, null);
          }
        } catch (error) {
          // Removed since it was listed: the run is gone.
          if ((error as NodeJS.ErrnoException | null)?.code === "ENOENT") {
            kept.delete(runId);
            continue;
          }
          if (!(error instanceof RangeError)) throw error;
          warn(`run index skipped ${log}: ${error.message}`);
          continue;
        }
        // A log emptied since its last update has no row.
        if (built === null) {
          kept.delete(runId);
          continue;
        }
        kept.add(runId);
        upsert.run(...toColumns(built));
      }
      const remove = db.prepare("DELETE FROM runs WHERE run_id = ?");
      for (const row of db.prepare("SELECT run_id FROM runs").all()) {
        if (!kept.has(row["run_id"] as string)) remove.run(row["run_id"] as string);
      }
      db.exec("COMMIT");
    } catch (error) {
      try {
        db.exec("ROLLBACK");
      } catch {
        // A failed COMMIT may have rolled back already; the first error is the one to report.
      }
      throw busyOr(error, path, timeoutMs);
    }
  } finally {
    db.close();
  }
}

/**
 * Return the rows of `<runsDir>/index.sqlite`, by start time then run id, as the last update left
 * them, with `running` put on top of a `stopped` or `crashed` run whose log a writer holds now
 * (`isRunLogLocked`, probed on every read). A missing index, or one of another layout version, has
 * no rows. Reading doesn't update the index; it reads while a builder works, and waits only while
 * one commits.
 *
 * @param runsDir The runs folder.
 * @param options.timeoutMs How long to wait while a builder commits, in milliseconds; 5000 by default.
 * @throws RangeError when `timeoutMs` isn't a non-negative integer.
 * @throws SQLite's error when the index file isn't a database or stays locked past `timeoutMs`.
 */
export function readRunIndex(runsDir: string, options: { timeoutMs?: number } = {}): RunIndexRow[] {
  const timeoutMs = checkedTimeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const path = join(runsDir, INDEX_FILE);
  if (!fs.existsSync(path)) return [];
  const db = openSqlite(path, { readonly: true });
  let rows: SqliteRow[];
  try {
    db.exec(`PRAGMA busy_timeout = ${timeoutMs}`);
    if (db.prepare("PRAGMA user_version").get()?.["user_version"] !== INDEX_VERSION) return [];
    rows = db.prepare("SELECT * FROM runs ORDER BY started_at, run_id").all();
  } finally {
    db.close();
  }
  return rows.map((columns) => {
    const { scan, ...row } = fromColumns(columns);
    let status = statusOf(scan);
    if ((status === "stopped" || status === "crashed") && isRunLogLocked(join(runsDir, row.runId, "log.jsonl"))) {
      status = "running";
    }
    return { ...row, status };
  });
}

/** Return the names of the folders in `runsDir`, sorted. */
function runFolders(runsDir: string): string[] {
  return fs
    .readdirSync(runsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort(compareCodeUnits);
}

/** Create the `runs` table, dropping one of another layout version first. */
function prepareSchema(db: Sqlite): void {
  const version = db.prepare("PRAGMA user_version").get()?.["user_version"];
  if (version === INDEX_VERSION) return;
  db.exec(`DROP TABLE IF EXISTS runs; ${SCHEMA}; PRAGMA user_version = ${INDEX_VERSION}`);
}

/** Return a busy SQLite `error` as a `RunIndexBusyError`, and any other error as it is. */
function busyOr(error: unknown, path: string, timeoutMs: number): unknown {
  if (((sqliteErrorCode(error) ?? -1) & 0xff) !== SQLITE_BUSY) return error;
  return new RunIndexBusyError(`run index ${path} is locked by another update; gave up after ${timeoutMs} ms`, {
    cause: error,
  });
}

/**
 * Fold the records of `log` after `saved` (from the start without it) into a copy of the row;
 * return `saved` when nothing new was read, and `null` for a log with no complete record.
 *
 * @throws RunLogChangedError when the log no longer continues `saved`.
 * @throws RangeError when the log can't be read or isn't the run `runId`.
 */
function readRun(log: string, runId: string, saved: Building | null): Building | null {
  const continuing = saved === null ? {} : { afterOffset: saved.offset, runId, afterSeq: saved.lastSeq };
  let row: Building | null = saved === null ? null : { ...saved, scan: { ...saved.scan, inFlight: [...saved.scan.inFlight] } };
  let lastAt: string | null = null;
  // The values of the top-level END's commits read in this pass, by seq; one read earlier is
  // found again through the offsets.
  const endCommits = new Map<number, string | null>();
  for (const entry of iterRunRecords(log, continuing)) {
    const record = entry.envelope.record;
    if (row === null) {
      if (!(record instanceof RunStartedRecord)) throw new RangeError("the first run log record must be RunStarted");
      if (entry.runId !== runId) {
        throw new RangeError(`the log is run ${reprText(entry.runId)}, not its folder's ${reprText(runId)}`);
      }
      row = {
        runId,
        flow: record.flow === null ? null : join(record.flow.dir, record.flow.root),
        startedAt: entry.at,
        endedAt: null,
        parentRunId: record.parent?.runId ?? null,
        lastSeq: 0,
        result: null,
        inputTokens: 0,
        outputTokens: 0,
        cacheReadTokens: 0,
        reasoningTokens: 0,
        offset: 0,
        scan: { inFlight: [], endSucceeded: false, nodeFailed: false, unreachable: false, stopped: false },
      };
    }
    row.lastSeq = entry.seq;
    row.offset = entry.end;
    lastAt = entry.at;
    const { scan } = row;
    const settle = (node: string) => {
      scan.inFlight = scan.inFlight.filter((id) => id !== node);
    };
    if (record instanceof NodeStartedRecord) {
      settle(record.nodeId);
      scan.inFlight.push(record.nodeId);
    } else if (record instanceof NodeStepRecord) {
      addUsage(row, record.step);
    } else if (record instanceof PoolCommitRecord) {
      if (record.nodeId === END_ID) endCommits.set(entry.seq, record.value === null ? null : logJson(record.value.value));
    } else if (record instanceof NodeSucceededRecord) {
      settle(record.nodeId);
      if (record.nodeId === END_ID) {
        scan.endSucceeded = true;
        row.result = endCommits.has(record.commit) ? endCommits.get(record.commit)! : committedValue(log, record.commit);
      }
    } else if (record instanceof NodeRoutedRecord) {
      settle(record.nodeId);
      if (record.terminalFailure !== null) scan.unreachable = true;
    } else if (record instanceof NodeExpandedRecord) {
      settle(record.spawnerId);
    } else if (record instanceof NodeFailedRecord) {
      settle(record.failure.nodeId!);
      scan.nodeFailed = true;
    } else if (record instanceof RunStoppedRecord) {
      scan.stopped = true;
    } else if (record instanceof RunResumedRecord) {
      // The resume retries exactly the nodes in flight.
      scan.inFlight = [];
      scan.stopped = false;
    }
  }
  if (row !== null && lastAt !== null) {
    const status = statusOf(row.scan);
    row.endedAt = status === "succeeded" || status === "failed" ? lastAt : null;
  }
  return row;
}

/** Add a `model_call` step's token counts to the row's sums. */
function addUsage(row: Building, step: Readonly<Record<string, unknown>>): void {
  if (step["kind"] !== "model_call" || !isRecord(step["usage"])) return;
  const usage = step["usage"];
  const count = (name: string) => (usage[name] == null ? 0 : Number(usage[name]));
  row.inputTokens += count("input_tokens");
  row.outputTokens += count("output_tokens");
  row.cacheReadTokens += count("cache_read_tokens");
  row.reasoningTokens += count("reasoning_tokens");
}

/** Return the value (`logJson`) of the `pool_commit` at `seq` in `log`, `null` for a derived one. */
function committedValue(log: string, seq: number): string | null {
  const offset = runRecordOffsets(log)[seq];
  if (offset === undefined) throw new RangeError(`the log has no record at seq ${seq}`);
  for (const entry of iterRunRecords(log, { afterOffset: offset })) {
    const record = entry.envelope.record;
    if (!(record instanceof PoolCommitRecord)) throw new RangeError(`the record at seq ${seq} isn't a pool_commit`);
    return record.value === null ? null : logJson(record.value.value);
  }
  throw new RangeError(`the log has no record at seq ${seq}`);
}

/**
 * Return the status the records show, as `inspectRun` reads it without a flow: a node failure, or
 * a route that proved END unreachable with nothing in flight, is `failed`; the top-level END's
 * success with nothing in flight is `succeeded`; then `stopped` or `crashed`.
 */
function statusOf(scan: RunScan): RunIndexStatus {
  const idle = scan.inFlight.length === 0;
  if (scan.nodeFailed || (idle && scan.unreachable)) return "failed";
  if (idle && scan.endSucceeded) return "succeeded";
  return scan.stopped ? "stopped" : "crashed";
}

/** Return a row's column values, in `COLUMNS` order. */
function toColumns(row: Building): (string | number | null)[] {
  const { scan } = row;
  return [
    row.runId,
    row.flow,
    statusOf(scan),
    row.startedAt,
    row.endedAt,
    row.parentRunId,
    row.lastSeq,
    row.result,
    row.inputTokens,
    row.outputTokens,
    row.cacheReadTokens,
    row.reasoningTokens,
    row.offset,
    logJson({
      in_flight: scan.inFlight,
      end_succeeded: scan.endSucceeded,
      node_failed: scan.nodeFailed,
      unreachable: scan.unreachable,
      stopped: scan.stopped,
    }),
  ];
}

/** Return the row a `SELECT *` gave. */
function fromColumns(columns: SqliteRow): Building {
  const scan = parseJson(columns["scan"] as string, "codec") as Record<string, unknown>;
  return {
    runId: columns["run_id"] as string,
    flow: columns["flow"] as string | null,
    startedAt: columns["started_at"] as string,
    endedAt: columns["ended_at"] as string | null,
    parentRunId: columns["parent_run_id"] as string | null,
    lastSeq: columns["last_seq"] as number,
    result: columns["result"] as string | null,
    inputTokens: columns["input_tokens"] as number,
    outputTokens: columns["output_tokens"] as number,
    cacheReadTokens: columns["cache_read_tokens"] as number,
    reasoningTokens: columns["reasoning_tokens"] as number,
    offset: columns["offset"] as number,
    scan: {
      inFlight: scan["in_flight"] as string[],
      endSucceeded: scan["end_succeeded"] === true,
      nodeFailed: scan["node_failed"] === true,
      unreachable: scan["unreachable"] === true,
      stopped: scan["stopped"] === true,
    },
  };
}
