/**
 * The run index: `<runs>/index.sqlite`, one row per run folder `<runs>/<run_id>/log.jsonl`.
 *
 * The logs are the only source of truth; the index is a view built from them. `updateRunIndex`
 * reads each log on from the byte offset its row reached, so it may lag the logs but never leads
 * them, and deleting the index and building it again gives the same rows. A row keeps the run
 * scan (`RunScan`, folded by `foldRunScan`, read by `scanStatus`), which `inspectRun` folds too, so
 * both read a status from the records by one set of rules; `readRunIndex` puts `running` on top
 * while a writer holds a run's lock. A run whose log can't be read keeps a row marked unreadable
 * at the first record yian couldn't read, holding the records before it. An index of another
 * layout or run-log schema is rebuilt, and so is a damaged one, as deleting it would.
 *
 * Knows about: compile (the START and END ids), compose's source parser (a run's declared memory
 * inputs), diagnostics, platform, runtime/log, runtime/state (`addCrashedAttempts`), runtime/waits
 * (the waits fold), typesys.
 */

import fs from "node:fs";
import { join } from "node:path";

import { END_ID, START_ID } from "../compile/model.ts";
import { SourceDocument } from "../compose/source/document.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { openSqlite, type Sqlite, type SqliteRow, sqliteErrorCode, warn } from "../platform.ts";
import { logJson } from "../typesys/json.ts";
import { parseType, type TypeExpr } from "../typesys/types.ts";
import { isRecord, type MemoryRef, shownText, TypedValue, valueRepr } from "../typesys/values.ts";
import {
  checkMaxLineBytes,
  type FlowSource,
  hardLinked,
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
  notRegularFile,
  PoolCommitRecord,
  RunCancelledRecord,
  RunFailedRecord,
  RunLogChangedError,
  runRecordOffsets,
  RunResumedRecord,
  RunStartedRecord,
  type RunRecord,
  RunStoppedRecord,
  SCHEMA_VERSION,
} from "./log.ts";
import { addCrashedAttempts } from "./state.ts";
import { applyWaitsRecord, type WaitInfo, WaitsState, wakeableNodes, wakeAt, waitInfos } from "./waits.ts";

/** The index file's name in the runs folder. */
const INDEX_FILE = "index.sqlite";

/** The index layout's version. */
const INDEX_VERSION = 7;

/**
 * What `PRAGMA user_version` holds: the layout version and the run-log schema the stored scans
 * were read under. An index of another is rebuilt, so a schema bump gives the rows a fresh build
 * gives.
 */
const STORED_VERSION = INDEX_VERSION * 1000 + SCHEMA_VERSION;

/** How long an update or read waits for SQLite's lock by default, in milliseconds. */
const DEFAULT_TIMEOUT_MS = 5000;

// `scan` holds the record scan (`RunScan`, as JSON), so an update goes on where the
// last one stopped. The status is read from it (`scanStatus`), the rules `inspectRun` uses, so no
// column holds a status that could drift from them. The `unreadable_*` columns are set for a log
// that can't be read; `log_size`, `log_mtime` and `max_line_bytes` are what it was read with, so a
// later update reads it again only when the log changed or a higher limit may read further.
// `memories` holds the run's memory inputs as JSON (`RunIndexMemory`s): identities, never a
// location, which no log holds either.
const SCHEMA = `CREATE TABLE runs (
  run_id TEXT PRIMARY KEY,
  flow TEXT,
  started_at TEXT,
  ended_at TEXT,
  parent_run_id TEXT,
  last_seq INTEGER,
  result TEXT,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  cache_read_tokens INTEGER NOT NULL,
  cache_creation_tokens INTEGER NOT NULL,
  reasoning_tokens INTEGER NOT NULL,
  "offset" INTEGER NOT NULL,
  scan TEXT NOT NULL,
  unreadable_seq INTEGER,
  unreadable_reason TEXT,
  log_size INTEGER,
  log_mtime REAL,
  max_line_bytes INTEGER,
  memories TEXT NOT NULL
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
  "cache_creation_tokens",
  "reasoning_tokens",
  '"offset"',
  "scan",
  "unreadable_seq",
  "unreadable_reason",
  "log_size",
  "log_mtime",
  "max_line_bytes",
  "memories",
];

/** The `runs` table's column names, in order, as `PRAGMA table_info` gives them. */
const COLUMN_NAMES = COLUMNS.map((column) => column.replaceAll('"', ""));

const SQLITE_BUSY = 5;
const SQLITE_CORRUPT = 11;
const SQLITE_NOTADB = 26;

/** A memory input of a run, as its START inputs recorded it. */
export interface RunIndexMemory {
  /** The memory input's name. */
  readonly input: string;
  /** The memory's identity, fixed when it was created; never its location. */
  readonly identity: string;
  /** Its memory type, such as `FileMemory`. */
  readonly type: string;
}

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
  /** The time of the run's `run_started` record, or `null` when that record can't be read. */
  readonly startedAt: string | null;
  /** For a `succeeded`, `failed` or `cancelled` run, the time of the log's last record; otherwise `null`. */
  readonly endedAt: string | null;
  /** The id of the run that started this one (`run_started.parent`), or `null`. */
  readonly parentRunId: string | null;
  /**
   * The memories the run was bound to, as its START inputs recorded them, in input-name order; an
   * Optional memory left out recorded none. `yian memory` finds the runs that used a memory here.
   */
  readonly memories: readonly RunIndexMemory[];
  /** The seq of the last record the index read, or `null` when it read none (see `unreadable`). */
  readonly lastSeq: number | null;
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
  /** The sum of every `model_call` step's `cache_creation_tokens`, failed attempts included. */
  readonly cacheCreationTokens: number;
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
  /**
   * For a log yian can't read, the seq of the first record it couldn't read and why; `null`
   * otherwise. The other fields hold the records before `seq`, so the status is where they leave
   * the run.
   */
  readonly unreadable: { readonly seq: number; readonly reason: string } | null;
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
    // A flow call's scope leaves its agent in flight.
    if (record.callId === null) settle(record.spawnerId);
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

/**
 * Where a log stops being readable (`RunIndexRow.unreadable`), and the log's size and mtime and the
 * `maxLineBytes` it was read with.
 */
interface Unreadable {
  seq: number;
  reason: string;
  size: number;
  mtimeMs: number;
  maxLineBytes: number | null;
}

/** A row as the index stores it, and while an update builds it. */
interface StoredRow {
  runId: string;
  flow: string | null;
  startedAt: string | null;
  endedAt: string | null;
  parentRunId: string | null;
  memories: RunIndexMemory[];
  lastSeq: number | null;
  result: string | null;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  reasoningTokens: number;
  offset: number;
  scan: RunScan;
  unreadable: Unreadable | null;
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
 * a line over `maxLineBytes`, or a file error such as no permission, or a log that is a symlink
 * or isn't a regular file, `EINVAL: not a regular file, open '<log>'`, which is never opened, or
 * has another link, `EINVAL: has more than one link, open '<log>'`) raises a
 * warning through `warn`. A log whose content can't be read gets a row of the records before the
 * one that failed, marked unreadable at it (`RunIndexRow.unreadable`), and is read again only once
 * its size or mtime changed, or with a higher `maxLineBytes` than it was read with. A file error keeps the row
 * as it was, and the next update tries again. A row whose stored scan can't be read is built
 * again from the start of its log. An index of another layout version or run-log schema, or whose
 * `runs` table has other columns, is dropped and built again.
 *
 * An index that is damaged or isn't a database (SQLite's CORRUPT or NOTADB) is deleted, with its
 * journal, and built again from the logs, with a warning.
 *
 * The update is one SQLite transaction (`BEGIN IMMEDIATE`), so builders in any process take turns.
 * It waits at most `timeoutMs` for another builder's lock, then throws. Synchronous: it blocks its
 * thread while it waits and reads.
 *
 * @param runsDir The runs folder.
 * @param options.timeoutMs How long to wait for the index's lock, in milliseconds; 5000 by default.
 * @param options.maxLineBytes The most bytes a log line may hold, as `iterRunRecords` takes it;
 *   no limit by default.
 * @throws RunIndexBusyError when another update holds the lock longer than `timeoutMs`.
 * @throws RangeError when `timeoutMs` isn't a non-negative integer, or `maxLineBytes` isn't a
 *   positive integer.
 * @throws Error (code `ENOENT`) when `runsDir` doesn't exist, and SQLite's other errors, such as
 *   an index rebuilt after damage that is still damaged.
 */
export function updateRunIndex(
  runsDir: string,
  options: { timeoutMs?: number; maxLineBytes?: number | null } = {},
): void {
  const timeoutMs = checkedTimeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const maxLineBytes = options.maxLineBytes ?? null;
  // Checked here: a log's read errors become warnings, and a bad option must not.
  checkMaxLineBytes(maxLineBytes);
  const path = join(runsDir, INDEX_FILE);
  try {
    updateIndexFile(runsDir, path, timeoutMs, maxLineBytes);
  } catch (error) {
    if (!damaged(error)) throw error;
    // The index is a cache of the logs: build it again, as deleting it by hand would. Its journal
    // goes too, so SQLite can't play a stale one into the new file.
    warn(`run index ${path} was damaged (${(error as Error).message}); rebuilt it from the logs`);
    for (const file of [path, `${path}-journal`]) fs.rmSync(file, { force: true });
    updateIndexFile(runsDir, path, timeoutMs, maxLineBytes);
  }
}

/** Run `updateRunIndex`'s transaction on the index file `path`, creating the file (0600) when missing. */
function updateIndexFile(runsDir: string, path: string, timeoutMs: number, maxLineBytes: number | null): void {
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
          // lstat, so a symlink at the log is skipped as the readers refuse it, not read as its target.
          const stat = fs.lstatSync(log);
          // Before `readUpTo`: a FIFO or a device reads as size 0, and is never opened here; a hard
          // link, which the readers refuse, would otherwise keep a row read through it current.
          if (!stat.isFile()) throw notRegularFile(log);
          if (stat.nlink > 1) throw hardLinked(log);
          if (saved !== null && readUpTo(saved, stat, maxLineBytes)) {
            kept.add(runId);
            continue;
          }
          try {
            built = readRun(log, runId, saved, stat, maxLineBytes);
          } catch (error) {
            if (!(error instanceof RunLogChangedError) || saved === null) throw error;
            built = readRun(log, runId, null, stat, maxLineBytes);
          }
        } catch (error) {
          const code = (error as NodeJS.ErrnoException | null)?.code;
          // No log: a file such as the index, a folder whose run hasn't started, or a run removed since.
          if (code === "ENOENT" || code === "ENOTDIR") continue;
          if (typeof code !== "string") throw error;
          // One log yian can't read as a file must not stop the others; its row stays as it was, and
          // the next update tries again.
          warn(`run index skipped ${log}: ${(error as Error).message}`);
          if (saved !== null) kept.add(runId);
          continue;
        }
        // A log with no complete record (one emptied since its last update included) has no row.
        if (built === null) continue;
        if (built.unreadable !== null) warn(`run index skipped ${log}: ${built.unreadable.reason}`);
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
 * Return the rows of `<runsDir>/index.sqlite`, by start time then run id (a row with no start time
 * first), as the last update left them, with `running` put on top of a `waiting`, `stopped` or `crashed` run whose log a writer holds now
 * (`isRunLogLocked`, probed on every read unless `lockProbe` is `false`; a lock that can't be
 * probed counts as free). A missing
 * index, or one of another layout, has no rows; so has a damaged one, with a warning through `warn`
 * (the next update rebuilds it). A row whose stored scan can't be read is
 * left out with a warning through `warn`; the next update builds it again. A `waiting` row's
 * `waits`, `ready` and `wakeAt` are read from its scan's waits fold, as `inspectRun` reads them.
 * Reading doesn't update the index; it reads while a builder works, and waits only while one
 * commits.
 *
 * @param runsDir The runs folder.
 * @param options.timeoutMs How long to wait while a builder commits, in milliseconds; 5000 by default.
 * @param options.lockProbe `false` skips the lock probes, which cost a file open per such row, so
 *   each row has the status its records show and none is `running`; `true` by default.
 * @throws RangeError when `timeoutMs` isn't a non-negative integer, or `lockProbe` isn't a boolean.
 * @throws SQLite's error when the index file stays locked past `timeoutMs`, or can't be read for
 *   another reason than damage.
 */
export function readRunIndex(runsDir: string, options: { timeoutMs?: number; lockProbe?: boolean } = {}): RunIndexRow[] {
  const timeoutMs = checkedTimeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const lockProbe = options.lockProbe ?? true;
  if (typeof lockProbe !== "boolean") throw new RangeError(`lockProbe must be a boolean, got ${valueRepr(lockProbe)}`);
  const path = join(runsDir, INDEX_FILE);
  if (!fs.existsSync(path)) return [];
  let db: Sqlite | undefined;
  let rows: SqliteRow[];
  try {
    db = openSqlite(path, { readonly: true });
    db.exec(`PRAGMA busy_timeout = ${timeoutMs}`);
    rows = indexRows(db);
  } catch (error) {
    if (!damaged(error)) throw error;
    warn(`run index ${path} is damaged (${(error as Error).message}); the next update rebuilds it`);
    return [];
  } finally {
    db?.close();
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
    const { scan, unreadable, ...row } = stored;
    let status: RunIndexStatus = scanStatus(scan);
    if (
      lockProbe &&
      (status === "waiting" || status === "stopped" || status === "crashed") &&
      locked(join(runsDir, row.runId, "log.jsonl"))
    ) {
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
        unreadable: unreadable === null ? null : { seq: unreadable.seq, reason: unreadable.reason },
      },
    ];
  });
}

/**
 * @internal Return the index's rows in run order, or `[]` for an index of another layout.
 * One read transaction holds the layout check and the rows, so a rebuild by another yian version
 * between the two can't hand back rows of its layout.
 */
export function indexRows(db: Sqlite): SqliteRow[] {
  db.exec("BEGIN");
  try {
    if (!currentLayout(db)) return [];
    return db.prepare("SELECT * FROM runs ORDER BY started_at, run_id").all();
  } finally {
    db.exec("COMMIT");
  }
}

/**
 * Return whether the index holds this version's layout: its `user_version`, and a `runs` table
 * with this version's columns (a hand edit or a partial restore can leave either alone).
 */
function currentLayout(db: Sqlite): boolean {
  if (db.prepare("PRAGMA user_version").get()?.["user_version"] !== STORED_VERSION) return false;
  const names = db.prepare("PRAGMA table_info(runs)").all().map((column) => column["name"]);
  return names.length === COLUMN_NAMES.length && names.every((name, i) => name === COLUMN_NAMES[i]);
}

/**
 * @internal Return why the run folder named `folder` can't hold the log of run `runId` (the folder
 * of a run is named by its id), or `null` when it can.
 */
export function folderMismatch(runId: string, folder: string): string | null {
  return runId === folder ? null : `the log is run ${reprText(runId)}, not its folder's ${reprText(folder)}`;
}

/** Return whether `error` is SQLite's report that the index file is damaged or isn't a database. */
function damaged(error: unknown): boolean {
  const code = (sqliteErrorCode(error) ?? -1) & 0xff;
  return code === SQLITE_CORRUPT || code === SQLITE_NOTADB;
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

/** Create the `runs` table, dropping one of another layout first. */
function prepareSchema(db: Sqlite): void {
  if (currentLayout(db)) return;
  db.exec(`DROP TABLE IF EXISTS runs; ${SCHEMA}; PRAGMA user_version = ${STORED_VERSION}`);
}

/** Return a busy SQLite `error` as a `RunIndexBusyError`, and any other error as it is. */
function busyOr(error: unknown, path: string, timeoutMs: number): unknown {
  if (((sqliteErrorCode(error) ?? -1) & 0xff) !== SQLITE_BUSY) return error;
  return new RunIndexBusyError(`run index ${path} is locked by another update; gave up after ${timeoutMs} ms`, {
    cause: error,
  });
}

/**
 * Return whether `saved` holds all an update can read of the log `stat` describes: every byte, or,
 * for a log marked unreadable, the same size and mtime read with a `maxLineBytes` at least as high.
 */
function readUpTo(saved: StoredRow, stat: fs.Stats, maxLineBytes: number | null): boolean {
  if (saved.unreadable === null) return saved.offset === stat.size;
  const { size, mtimeMs, maxLineBytes: readWith } = saved.unreadable;
  // A higher limit may read the line that was too long.
  const higher = readWith !== null && (maxLineBytes === null || maxLineBytes > readWith);
  return size === stat.size && mtimeMs === stat.mtimeMs && !higher;
}

/**
 * Fold the records of `log` after `saved` (from the start without it, or when it holds none) into
 * `saved` itself, a row fresh from the index that a thrown error leaves to be discarded, or into a
 * new row; return the row (`saved` unchanged when nothing new was read), or `null` for a log with
 * no complete record. Lines are read with `maxLineBytes`.
 *
 * A log whose content can't be read past some record (a `RangeError`) gives the row of the records
 * before it, marked `unreadable` at it with `stat`, the log as it was before the read, and the
 * error's message cut by `shownText`.
 *
 * @throws RunLogChangedError when the log no longer continues `saved`.
 * @throws Error (a file error, such as `ENOENT` or `EACCES`) when the log can't be read as a file.
 */
function readRun(log: string, runId: string, saved: StoredRow | null, stat: fs.Stats, maxLineBytes: number | null): StoredRow | null {
  const continuing = saved !== null && saved.lastSeq !== null;
  const options = continuing ? { afterOffset: saved.offset, runId, afterSeq: saved.lastSeq!, maxLineBytes } : { maxLineBytes };
  let row = continuing ? saved : null;
  if (row !== null) row.unreadable = null;
  let lastAt: string | null = null;
  // The values of the top-level END's commits read in this pass, by seq; one read earlier is
  // found again through the offsets.
  const endCommits = new Map<number, string | null>();
  // The memory inputs the run's flow declares, once this pass has read them (`undefined` until).
  let declared: ReadonlySet<string> | null | undefined;
  try {
    for (const entry of iterRunRecords(log, options)) {
      const record = entry.envelope.record;
      if (row === null) {
        if (!(record instanceof RunStartedRecord)) throw new RangeError("the first run log record must be RunStarted");
        const mismatch = folderMismatch(entry.runId, runId);
        if (mismatch !== null) throw new RangeError(mismatch);
        row = emptyRow(runId);
        row.flow = record.flow === null ? null : join(record.flow.dir, record.flow.root);
        declared = declaredMemories(record.flow);
        row.startedAt = entry.at;
        row.parentRunId = record.parent?.runId ?? null;
      }
      // Read before the row changes, so a failure leaves it at the record before this one.
      let result = row.result;
      let memories = row.memories;
      if (record instanceof PoolCommitRecord) {
        if (record.nodeId === END_ID) endCommits.set(entry.seq, record.value === null ? null : logJson(record.value.value));
        else if (record.nodeId === START_ID && record.key === START_ID) {
          // An earlier pass read the run's start; its flow is read again from the log's first record.
          if (declared === undefined) declared = declaredMemories(startedFlow(log, maxLineBytes));
          memories = startMemories(record.value?.value, declared);
        }
      } else if (record instanceof NodeSucceededRecord && record.nodeId === END_ID) {
        result = endCommits.has(record.commit) ? endCommits.get(record.commit)! : committedValue(log, record.commit, maxLineBytes);
      }
      row.lastSeq = entry.seq;
      row.offset = entry.end;
      row.result = result;
      row.memories = memories;
      lastAt = entry.at;
      foldRunScan(row.scan, entry.envelope);
      if (record instanceof NodeStepRecord) addUsage(row, record.step);
    }
  } catch (error) {
    // The caller reads a replaced log again from the start. A file error isn't marked: a change
    // that fixes it (a chmod) may leave the size and mtime as they were.
    if (!(error instanceof RangeError) || (error instanceof RunLogChangedError && continuing)) throw error;
    row ??= emptyRow(runId);
    const seq = row.lastSeq === null ? 0 : row.lastSeq + 1;
    // The message quotes the line's fields, which may be of any length.
    row.unreadable = { seq, reason: shownText(error.message), size: stat.size, mtimeMs: stat.mtimeMs, maxLineBytes };
  }
  if (row !== null && lastAt !== null) {
    const status = scanStatus(row.scan);
    row.endedAt = status === "succeeded" || status === "failed" || status === "cancelled" ? lastAt : null;
  }
  return row;
}

/**
 * Return the memory inputs the top-level START inputs `inputs` recorded, by input name: each input
 * the run's flow declares a memory (`declared`) whose value is a memory value's form. Another input
 * may hold plain data of that form, so the declaration decides. A flow built in code recorded no
 * declarations (`null`); then each field of that form counts.
 */
function startMemories(inputs: unknown, declared: ReadonlySet<string> | null): RunIndexMemory[] {
  if (!isRecord(inputs)) return [];
  const memories: RunIndexMemory[] = [];
  for (const input of Object.keys(inputs).sort(compareCodeUnits)) {
    if (declared !== null && !declared.has(input)) continue;
    let memory: MemoryRef;
    try {
      memory = TypedValue.fromWire({ kind: "memory", value: inputs[input] }).value as MemoryRef;
    } catch (error) {
      if (error instanceof RangeError) continue;
      throw error;
    }
    memories.push({ input, identity: memory.identity, type: memory.type });
  }
  return memories;
}

/**
 * Return the names of the memory inputs the flow `source` declares, read from the tags of its root
 * file's `input:` without loading the flow: a memory type, alone or in `Optional(...)`, or a name
 * that is none of the file's typedefs, which only a host's memory type can be. `null` for a flow
 * built in code, which has no recorded source, or a root file that doesn't parse.
 */
function declaredMemories(source: FlowSource | null): ReadonlySet<string> | null {
  const text = source?.files[source.root]?.text;
  if (text === undefined) return null;
  let document: SourceDocument;
  try {
    document = SourceDocument.parse(text);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    return null;
  }
  const top = document.root.value;
  const typedefs = new Set(isRecord(top) && isRecord(top["typedefs"]) ? Object.keys(top["typedefs"]) : []);
  const names = new Set<string>();
  for (const node of document.taggedNodes({ skipTopLevel: (key) => key !== "input" })) {
    const [section, name, ...deeper] = node.keyPath;
    if (node.isKey || section !== "input" || typeof name !== "string" || deeper.length > 0) continue;
    let typ: TypeExpr;
    try {
      typ = parseType(node.tag);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      continue;
    }
    while (typ.kind === "optional") typ = typ.inner;
    if (typ.kind === "memory" || (typ.kind === "ref" && !typedefs.has(typ.name))) names.add(name);
  }
  return names;
}

/**
 * Return the flow the run in `log` started from, as its first record recorded it, or `null` when it
 * records none.
 */
function startedFlow(log: string, maxLineBytes: number | null): FlowSource | null {
  for (const entry of iterRunRecords(log, { maxLineBytes })) {
    const record = entry.envelope.record;
    return record instanceof RunStartedRecord ? record.flow : null;
  }
  return null;
}

/**
 * Return the stored `memories` column.
 *
 * @throws RangeError when it isn't a JSON array of `{input, identity, type}` texts.
 */
function memoriesFromJson(text: unknown): RunIndexMemory[] {
  let parsed: unknown;
  try {
    parsed = typeof text === "string" ? JSON.parse(text) : null;
  } catch {
    parsed = null;
  }
  const valid =
    Array.isArray(parsed) &&
    parsed.every(
      (each) =>
        isRecord(each) && ["input", "identity", "type"].every((field) => typeof each[field] === "string"),
    );
  if (!valid) throw new RangeError("the stored memories aren't a list of memory inputs");
  return (parsed as RunIndexMemory[]).map(({ input, identity, type }) => ({ input, identity, type }));
}

/** Return the row of the run `runId` before any record is read. */
function emptyRow(runId: string): StoredRow {
  return {
    runId,
    flow: null,
    startedAt: null,
    endedAt: null,
    parentRunId: null,
    memories: [],
    lastSeq: null,
    result: null,
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheCreationTokens: 0,
    reasoningTokens: 0,
    offset: 0,
    scan: emptyRunScan(),
    unreadable: null,
  };
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
  row.cacheCreationTokens += count("cache_creation_tokens");
  row.reasoningTokens += count("reasoning_tokens");
}

/**
 * Return the value (`logJson`) of the `pool_commit` at `seq` in `log`, `null` for a derived one,
 * reading lines with `maxLineBytes`.
 */
function committedValue(log: string, seq: number, maxLineBytes: number | null): string | null {
  const offset = runRecordOffsets(log, { maxLineBytes })[seq];
  if (offset === undefined) throw new RangeError(`the log has no record at seq ${seq}`);
  for (const entry of iterRunRecords(log, { afterOffset: offset, maxLineBytes })) {
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
    row.cacheCreationTokens,
    row.reasoningTokens,
    row.offset,
    // JSON data: ids, seqs, counts, flags, and the waits fold's own JSON form.
    JSON.stringify(row.scan),
    row.unreadable?.seq ?? null,
    row.unreadable?.reason ?? null,
    row.unreadable?.size ?? null,
    row.unreadable?.mtimeMs ?? null,
    row.unreadable?.maxLineBytes ?? null,
    JSON.stringify(row.memories),
  ];
}

/** Return the row a `SELECT *` gave. */
function fromColumns(columns: SqliteRow): StoredRow {
  return {
    runId: columns["run_id"] as string,
    flow: columns["flow"] as string | null,
    startedAt: columns["started_at"] as string | null,
    endedAt: columns["ended_at"] as string | null,
    parentRunId: columns["parent_run_id"] as string | null,
    memories: memoriesFromJson(columns["memories"]),
    lastSeq: columns["last_seq"] as number | null,
    result: columns["result"] as string | null,
    inputTokens: columns["input_tokens"] as number,
    outputTokens: columns["output_tokens"] as number,
    cacheReadTokens: columns["cache_read_tokens"] as number,
    cacheCreationTokens: columns["cache_creation_tokens"] as number,
    reasoningTokens: columns["reasoning_tokens"] as number,
    offset: columns["offset"] as number,
    scan: runScanFromJson(columns["scan"] as string),
    unreadable:
      columns["unreadable_seq"] === null
        ? null
        : {
            seq: columns["unreadable_seq"] as number,
            reason: columns["unreadable_reason"] as string,
            size: columns["log_size"] as number,
            mtimeMs: columns["log_mtime"] as number,
            maxLineBytes: columns["max_line_bytes"] as number | null,
          },
  };
}
