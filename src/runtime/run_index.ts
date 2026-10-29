/**
 * The run index: `<runs>/index.sqlite`, one row per run folder `<runs>/<run_id>/log.jsonl`.
 *
 * The logs are the only source of truth; the index is a view built from them. `updateRunIndex`
 * reads each log on from the byte offset its row reached, so it may lag the logs but never leads
 * them, and deleting the index and building it again gives the same rows. A row keeps the run
 * scan (`RunScan`, folded by `foldRunScan`, read by `scanStatus`), which `inspectRun` folds too, so
 * both read a status from the records by one set of rules; `readRunIndex` puts `running` on top
 * while a writer holds a run's lock.
 *
 * Knows about: compile (the END id), diagnostics, platform, runtime/log, runtime/state
 * (`addCrashedAttempts`), runtime/waits (the waits fold), typesys.
 */

import fs from "node:fs";
import { join } from "node:path";

import { END_ID } from "../compile/model.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { openSqlite, type Sqlite, type SqliteRow, sqliteErrorCode, warn } from "../platform.ts";
import { logJson } from "../typesys/json.ts";
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
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  RunCancelledRecord,
  RunFailedRecord,
  RunLogChangedError,
  runRecordOffsets,
  RunResumedRecord,
  RunStartedRecord,
  type RunRecord,
  RunStoppedRecord,
} from "./log.ts";
import { addCrashedAttempts } from "./state.ts";
import { applyWaitsRecord, type WaitInfo, WaitsState, wakeableNodes, wakeAt, waitInfos } from "./waits.ts";

/** The index file's name in the runs folder. */
const INDEX_FILE = "index.sqlite";

/** The index layout's version, kept in `PRAGMA user_version`; an index of another is rebuilt. */
const INDEX_VERSION = 4;

/** How long an update or read waits for SQLite's lock by default, in milliseconds. */
const DEFAULT_TIMEOUT_MS = 5000;

// `scan` holds the record scan (`RunScan`, as JSON), so an update goes on where the
// last one stopped. The status is read from it (`scanStatus`), the rules `inspectRun` uses, so no
// column holds a status that could drift from them.
const SCHEMA = `CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  flow TEXT,
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
export type RunIndexStatus = "running" | ReturnType<typeof scanStatus>;

/** One run as the index holds it. */
export interface RunIndexRow {
  /** The run's id, its folder's name. */
  readonly runId: string;
  /** The path of the flow file the run started from, or `null` for a flow built in code. */
  readonly flow: string | null;
  /**
   * `succeeded`, `failed` or `cancelled` once the records show the run ended; `waiting` when it
   * ended its last segment with every unsettled node waiting; `stopped` when the host stopped the
   * last segment; `crashed` otherwise. `running` replaces `waiting`, `stopped` or `crashed` while a
   * writer holds the log's lock.
   */
  readonly status: RunIndexStatus;
  /** The time of the run's `run_started` record. */
  readonly startedAt: string;
  /** For a `succeeded`, `failed` or `cancelled` run, the time of the log's last record; otherwise `null`. */
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
  /** The sum of every `model_call` step's `input_tokens`, failed attempts included. */
  readonly inputTokens: number;
  /** The sum of every `model_call` step's `output_tokens`, failed attempts included. */
  readonly outputTokens: number;
  /** The sum of every `model_call` step's `cache_read_tokens`, failed attempts included. */
  readonly cacheReadTokens: number;
  /** The sum of every `model_call` step's `reasoning_tokens`, failed attempts included. */
  readonly reasoningTokens: number;
  /** The byte offset the row was built up to: the end of the record at `lastSeq`. */
  readonly offset: number;
  /** For a `waiting` run, the calls of each waiting node, oldest wait first; `[]` otherwise. */
  readonly waits: readonly WaitInfo[];
  /** Whether a `waiting` run has a waiting node it can wake now; `false` otherwise. */
  readonly ready: boolean;
  /** For a `waiting` run, the earliest deadline of an open call of a waiting node; `null` otherwise. */
  readonly wakeAt: string | null;
}

/**
 * `updateRunIndex` gave up waiting for the index's lock: another update held it for longer than
 * the timeout.
 */
export class RunIndexBusyError extends Error {
  override name = "RunIndexBusyError";
}

/**
 * What the records of a run log show about where it stands, folded one record at a time
 * (`foldRunScan`) by `inspectRun` and by the run index, which stores it between updates. JSON
 * data but for `waits`, which writes its own JSON form; `runScanFromJson` reads the stored text
 * back. Numbers are record seqs or counts.
 */
export interface RunScan {
  /**
   * `[node id, start seq]` of each node in flight (started or woken, and neither settled nor
   * waiting), in first-start order.
   */
  inFlight: [string, number][];
  /** The seq of the last `RunStopped` since the last `RunResumed`, or `null`. */
  lastStop: number | null;
  /** Whether the top-level END succeeded. */
  endSucceeded: boolean;
  /** `[node id, count]`: the attempts the run's resumes retried after a crash, in first-count order. */
  crashed: [string, number][];
  /** The seq of the first `NodeFailed`, whose failure ends the run, or `null`. */
  nodeFailureSeq: number | null;
  /** The seq of the first `NodeRouted` that proved END unreachable, or `null`. */
  unreachableSeq: number | null;
  /** The seq of the `RunFailed` that ended the run, or `null`. */
  runFailureSeq: number | null;
  /** Whether a `RunCancelled` ended the run. */
  cancelled: boolean;
  /** The inbox and call fold (`applyWaitsRecord`): the waiting nodes, their calls and the inboxes. */
  waits: WaitsState;
}

/** Return the scan of a log with no record. */
export function emptyRunScan(): RunScan {
  return {
    inFlight: [],
    lastStop: null,
    endSucceeded: false,
    crashed: [],
    nodeFailureSeq: null,
    unreachableSeq: null,
    runFailureSeq: null,
    cancelled: false,
    waits: new WaitsState(),
  };
}

/**
 * Return the scan `JSON.stringify(scan)` wrote as `text`.
 *
 * @throws RangeError when `text` isn't such a scan.
 */
export function runScanFromJson(text: string): RunScan {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (error) {
    throw new RangeError(`run scan isn't JSON: ${(error as Error).message}`, { cause: error });
  }
  const bad = (field: string, what: string) => new RangeError(`run scan ${field} must be ${what}`);
  if (!isRecord(raw)) throw new RangeError("run scan must be an object");
  const seqOrNull = (field: string): number | null => {
    const value = raw[field];
    if (value !== null && !Number.isSafeInteger(value)) throw bad(field, "an integer or null");
    return value as number | null;
  };
  const pairs = (field: string): [string, number][] => {
    const value = raw[field];
    const ok =
      Array.isArray(value) &&
      value.every((pair) => Array.isArray(pair) && pair.length === 2 && typeof pair[0] === "string" && Number.isSafeInteger(pair[1]));
    if (!ok) throw bad(field, "an array of [node id, integer] pairs");
    return (value as [string, number][]).map(([node, n]) => [node, n]);
  };
  const flag = (field: string): boolean => {
    if (typeof raw[field] !== "boolean") throw bad(field, "a boolean");
    return raw[field] as boolean;
  };
  return {
    inFlight: pairs("inFlight"),
    lastStop: seqOrNull("lastStop"),
    endSucceeded: flag("endSucceeded"),
    crashed: pairs("crashed"),
    nodeFailureSeq: seqOrNull("nodeFailureSeq"),
    unreachableSeq: seqOrNull("unreachableSeq"),
    runFailureSeq: seqOrNull("runFailureSeq"),
    cancelled: flag("cancelled"),
    waits: WaitsState.fromJSON(raw["waits"]),
  };
}

/**
 * Fold the record `envelope` holds into `scan`, in place, as the reducer derives starts, outcomes,
 * stops and resumes: a start or a wake puts its node in flight (a node already in flight keeps its
 * place), a wait takes it out, an outcome settles it, a stop is remembered until the next resume,
 * and a resume counts the attempts no stop covers (`addCrashedAttempts`) and retries every node in
 * flight. Every record is folded into `scan.waits` too.
 */
export function foldRunScan(scan: RunScan, envelope: { readonly seq: number; readonly record: RunRecord }): void {
  const { seq, record } = envelope;
  const settle = (node: string) => {
    scan.inFlight = scan.inFlight.filter(([id]) => id !== node);
  };
  applyWaitsRecord(scan.waits, envelope);
  if (record instanceof NodeStartedRecord || record instanceof NodeWokenRecord) {
    const entry = scan.inFlight.find(([id]) => id === record.nodeId);
    if (entry === undefined) scan.inFlight.push([record.nodeId, seq]);
    else entry[1] = seq;
  } else if (record instanceof NodeWaitingRecord) {
    settle(record.nodeId);
  } else if (record instanceof NodeSucceededRecord) {
    settle(record.nodeId);
    if (record.nodeId === END_ID) scan.endSucceeded = true;
  } else if (record instanceof NodeRoutedRecord) {
    settle(record.nodeId);
    if (record.terminalFailure !== null) scan.unreachableSeq ??= seq;
  } else if (record instanceof NodeExpandedRecord) {
    settle(record.spawnerId);
  } else if (record instanceof NodeFailedRecord) {
    settle(record.failure.nodeId!);
    scan.nodeFailureSeq ??= seq;
  } else if (record instanceof RunStoppedRecord) {
    scan.lastStop = seq;
  } else if (record instanceof RunResumedRecord) {
    // The resume retries exactly the nodes in flight.
    const counts = new Map(scan.crashed);
    addCrashedAttempts(counts, new Map(scan.inFlight), scan.lastStop);
    scan.crashed = [...counts];
    scan.inFlight = [];
    scan.lastStop = null;
  } else if (record instanceof RunFailedRecord) {
    scan.runFailureSeq ??= seq;
  } else if (record instanceof RunCancelledRecord) {
    scan.cancelled = true;
  }
}

/**
 * Return where the scanned records leave the run, as `inspectRun` reads it without a flow: a
 * cancel is `cancelled`; a node failure or `RunFailed` is `failed`; with nothing in flight and
 * nothing waiting, the top-level END's success is `succeeded`, else a route that proved END
 * unreachable is `failed`; otherwise `unfinishedStatus`. The records don't hold the ready nodes,
 * so a node still ready is missed.
 */
export function scanStatus(scan: RunScan): "succeeded" | "failed" | "cancelled" | ReturnType<typeof unfinishedStatus> {
  if (scan.cancelled) return "cancelled";
  // A node failure ends the run at once, ahead of a pending unreachable END.
  if (scan.nodeFailureSeq !== null || scan.runFailureSeq !== null) return "failed";
  const idle = scan.inFlight.length === 0 && scan.waits.waiting.size === 0;
  // END's stored output wins over a later unreachable proof, as the reducer reads it.
  if (idle && scan.endSucceeded) return "succeeded";
  if (idle && scan.unreachableSeq !== null) return "failed";
  return unfinishedStatus(scan.waits, scan.lastStop);
}

/**
 * Return where an unfinished run stands: `waiting` when its last frontier record is `run_waiting`
 * (every unsettled node waits, and no worker runs), `stopped` when the host stopped the last
 * segment (`lastStop`, the seq of the last `RunStopped` since the last `RunResumed`), else
 * `crashed`.
 */
export function unfinishedStatus(waits: WaitsState, lastStop: number | null): "waiting" | "stopped" | "crashed" {
  if (waits.lastFrontier === "run_waiting") return "waiting";
  return lastStop === null ? "crashed" : "stopped";
}

/** A row as the index stores it, and while an update builds it. */
interface StoredRow {
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
 * creating the index (mode 0600) when it is missing.
 *
 * Each log is read on from the byte offset its row reached (`iterRunRecords` with `afterOffset`,
 * `runId` and `afterSeq`); a log that hasn't grown isn't opened. A log that was truncated or
 * replaced (`RunLogChangedError`) is read again from the start. A run folder with no log, or a log
 * with no complete record yet, has no row; a row whose log is gone is deleted. A log that can't be
 * read (an old schema, a damaged line, a first record that isn't `run_started` of the folder's run,
 * or a file error such as no permission) raises a warning through `warn` and keeps its row as it
 * was. A row whose stored scan can't be read is built again from the start of its log. An index
 * of another layout version is dropped and built again.
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
  createPrivate(path);
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
      const select = db.prepare("SELECT * FROM runs WHERE run_id = ?");
      const upsert = db.prepare(
        `INSERT OR REPLACE INTO runs (${COLUMNS.join(", ")}) VALUES (${COLUMNS.map(() => "?").join(", ")})`,
      );
      const kept = new Set<string>();
      // Listed under the lock, so a run another builder indexed since is never dropped as gone.
      for (const runId of fs.readdirSync(runsDir).sort(compareCodeUnits)) {
        const log = join(runsDir, runId, "log.jsonl");
        const stored = select.get(runId);
        let saved: StoredRow | null = null;
        try {
          saved = stored === undefined ? null : fromColumns(stored);
        } catch (error) {
          // A row this version can't read is built again from the log.
          if (!(error instanceof RangeError)) throw error;
        }
        let built: StoredRow | null;
        try {
          const size = fs.statSync(log).size;
          if (saved !== null && saved.offset === size) {
            kept.add(runId);
            continue;
          }
          try {
            built = readRun(log, runId, saved);
          } catch (error) {
            if (!(error instanceof RunLogChangedError) || saved === null) throw error;
            built = readRun(log, runId, null);
          }
        } catch (error) {
          const code = (error as NodeJS.ErrnoException | null)?.code;
          // No log: a file such as the index, a folder whose run hasn't started, or a run removed since.
          if (code === "ENOENT" || code === "ENOTDIR") continue;
          if (!(error instanceof RangeError || typeof code === "string")) throw error;
          // One log yian can't read must not stop the others; its row stays as it was.
          warn(`run index skipped ${log}: ${(error as Error).message}`);
          if (saved !== null) kept.add(runId);
          continue;
        }
        // A log with no complete record (one emptied since its last update included) has no row.
        if (built === null) continue;
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
 * them, with `running` put on top of a `waiting`, `stopped` or `crashed` run whose log a writer holds now
 * (`isRunLogLocked`, probed on every read; a lock that can't be probed counts as free). A missing
 * index, or one of another layout version, has no rows. A row whose stored scan can't be read is
 * left out with a warning through `warn`; the next update builds it again. A `waiting` row's
 * `waits`, `ready` and `wakeAt` are read from its scan's waits fold, as `inspectRun` reads them.
 * Reading doesn't update the index; it reads while a builder works, and waits only while one
 * commits.
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
    rows = indexRows(db);
  } finally {
    db.close();
  }
  return rows.flatMap((columns) => {
    let stored: StoredRow;
    try {
      stored = fromColumns(columns);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      // The next update builds the row again; until then it isn't listed.
      warn(`run index skipped ${columns["run_id"] as string}: ${error.message}`);
      return [];
    }
    const { scan, ...row } = stored;
    let status: RunIndexStatus = scanStatus(scan);
    if ((status === "waiting" || status === "stopped" || status === "crashed") && locked(join(runsDir, row.runId, "log.jsonl"))) {
      status = "running";
    }
    const waiting = status === "waiting";
    return [
      {
        ...row,
        status,
        waits: waiting ? waitInfos(scan.waits) : [],
        ready: waiting && wakeableNodes(scan.waits).length > 0,
        wakeAt: waiting ? wakeAt(scan.waits) : null,
      },
    ];
  });
}

/**
 * @internal Return the index's rows in run order, or `[]` for an index of another layout version.
 * One read transaction holds the version check and the rows, so a rebuild by another yian version
 * between the two can't hand back rows of its layout.
 */
export function indexRows(db: Sqlite): SqliteRow[] {
  db.exec("BEGIN");
  try {
    if (db.prepare("PRAGMA user_version").get()?.["user_version"] !== INDEX_VERSION) return [];
    return db.prepare("SELECT * FROM runs ORDER BY started_at, run_id").all();
  } finally {
    db.exec("COMMIT");
  }
}

/** Return whether a writer holds the lock of `log`; a lock that can't be probed counts as free. */
function locked(log: string): boolean {
  try {
    return isRunLogLocked(log);
  } catch {
    // The stored status is what the records show; a lock that can't be read adds nothing to it.
    return false;
  }
}

/**
 * Create the empty file `path` with mode 0600 when it is missing, so the index, which holds every
 * run's result, is as private as the run folders; SQLite gives its journal the same mode.
 */
function createPrivate(path: string): void {
  try {
    fs.closeSync(fs.openSync(path, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY, 0o600));
  } catch (error) {
    if ((error as NodeJS.ErrnoException | null)?.code !== "EEXIST") throw error;
  }
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
 * Fold the records of `log` after `saved` (from the start without it) into `saved` itself, a row
 * fresh from the index that a failed read leaves to be discarded, or into a new row; return the
 * row (`saved` unchanged when nothing new was read), or `null` for a log with no complete record.
 *
 * @throws RunLogChangedError when the log no longer continues `saved`.
 * @throws RangeError when the log can't be read or isn't the run `runId`.
 */
function readRun(log: string, runId: string, saved: StoredRow | null): StoredRow | null {
  const continuing = saved === null ? {} : { afterOffset: saved.offset, runId, afterSeq: saved.lastSeq };
  let row = saved;
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
        scan: emptyRunScan(),
      };
    }
    row.lastSeq = entry.seq;
    row.offset = entry.end;
    lastAt = entry.at;
    foldRunScan(row.scan, entry.envelope);
    if (record instanceof NodeStepRecord) {
      addUsage(row, record.step);
    } else if (record instanceof PoolCommitRecord) {
      if (record.nodeId === END_ID) endCommits.set(entry.seq, record.value === null ? null : logJson(record.value.value));
    } else if (record instanceof NodeSucceededRecord && record.nodeId === END_ID) {
      row.result = endCommits.has(record.commit) ? endCommits.get(record.commit)! : committedValue(log, record.commit);
    }
  }
  if (row !== null && lastAt !== null) {
    const status = scanStatus(row.scan);
    row.endedAt = status === "succeeded" || status === "failed" || status === "cancelled" ? lastAt : null;
  }
  return row;
}

/**
 * Add a `model_call` step's token counts to the row's sums. Only a safe integer counts: the step
 * isn't validated, and a count of another form (text, a fraction, a huge int) adds 0.
 */
function addUsage(row: StoredRow, step: Readonly<Record<string, unknown>>): void {
  if (step["kind"] !== "model_call" || !isRecord(step["usage"])) return;
  const usage = step["usage"];
  const count = (name: string) => {
    const value = usage[name];
    const n = typeof value === "bigint" ? Number(value) : value;
    return typeof n === "number" && Number.isSafeInteger(n) ? n : 0;
  };
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

/** Return a row's column values, in `COLUMNS` order. */
function toColumns(row: StoredRow): (string | number | null)[] {
  return [
    row.runId,
    row.flow,
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
    // JSON data: ids, seqs, counts, flags, and the waits fold's own JSON form.
    JSON.stringify(row.scan),
  ];
}

/** Return the row a `SELECT *` gave. */
function fromColumns(columns: SqliteRow): StoredRow {
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
    scan: runScanFromJson(columns["scan"] as string),
  };
}
