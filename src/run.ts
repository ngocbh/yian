/**
 * The package-root run boundary over one authoritative append-only log.
 *
 * `runFlow` initializes a private copy of a loaded flow, commits its complete canonical start
 * state, and drives only post-durable observer notifications. `advance` replays a durable log
 * silently onto a private copy and drives the unfinished rest the same way.
 *
 * The runtime borrows an already-open log. This module owns log creation and closure after the
 * engine's structured drain. It also owns the compile/runtime inversion seam by injecting
 * `compile/expand.ts`'s `buildGrow` into the kind-blind engine.
 *
 * This is the `yian/run` module: it exports `runFlow`, `advance` and `RunResult`. Hosts reach a
 * run by its log through `send`, `withdraw`, `complete`, `close`, `interrupt`, `cancel` and
 * `pendingCalls`; their errors are `RunBusyError` and the re-exported `DeliveryRefused`
 * (`runtime/state.ts`). For hosts that show runs it also exports `inspectRun`, which reads
 * where a run log stands, and `staticNodes`, which names each node of a flow, and re-exports
 * `staticPathOf` (`nodes/ids.ts`), the lock-free reader `iterRunRecords`, `RunLogChangedError`,
 * `RunLogLineTooLongError` and `RunLogSchemaError` (`runtime/log.ts`). `runLogPath` gives a fresh
 * run a folder of its own under the host's runs folder and names its log there, and
 * `updateRunIndex` and `readRunIndex` (`runtime/run_index.ts`) keep and read the runs folder's
 * rebuildable `index.sqlite`, one row per run. For hosts that read run-log values it re-exports
 * yian's JSON form of them: `TypedValue` (the `{kind, value}` wire of a `pool_commit`,
 * `toWire`/`fromWire`), `FileRef`, `InboxRef`, the handle an `Inbox(T)` input holds, and
 * `MemoryRef`, the value a memory input records, with its tools' `RecordedMemoryTool` shape
 * (`typesys/values.ts`, `typesys/memory.ts`), and `logJson`, `parseJson` and
 * `JSONDecodeError` (`typesys/json.ts`), the codec the run log itself writes and reads with, and
 * `foldRunLog`, which replays a log and gives each accepted `pool_commit`'s value, a derived one
 * rebuilt from its `from` pointers, a take's from the sends it consumed and an AGENT's new messages
 * from its steps, with `PoolCommitRecord` (`runtime/log.ts`) to read the records
 * it returns, and the built-in message value (`Message`, `MessagePart`, `checkMessage`, which reads
 * a decoded `Message` back with its files, `checkConversation` and `ConversationError`, from
 * `nodes/agent/messages.ts`). For hosts that show an AGENT's model calls it exports `agentCalls`,
 * which lists a log's calls from its steps, and `agentCall`, which rebuilds one call's request and
 * reads its attempts, reply and tool results.
 *
 * A run binds each memory input to the instance its host passes (`runtime/memories.ts`): before
 * `RunStarted` it opens and snapshots each into the value START's inputs record, and a resume
 * binds the same memories again; `RunResult.memoryDrift` and the `MemoryDrift` type report a
 * memory that changed since the run started (including the run's own writes), as a crash resume
 * finds it.
 *
 * Knows about:   compile, compose, defaults, diagnostics, events, expr, llm_clients, memory,
 *                nodes, platform, runtime, skills, tools, typesys.
 * Never imports: the CLI.
 */

import { randomUUID } from "node:crypto";
import { chmodSync, mkdirSync, realpathSync, rmdirSync, statSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { types } from "node:util";

import { localTypes } from "./compile/declared_types.ts";
import { buildGrow } from "./compile/expand.ts";
import {
  applyEffectiveLlmMap,
  bindHostCredentials,
  callTimeSelection,
  hostKeyProviders,
  inFlightLlmConfigs,
  knownCallTimeProviders,
  resolveRunLlm,
  type RunLLM,
  staticAgentLabels,
  valueConfigSubjects,
} from "./compile/llm_cascade.ts";
import { flowFingerprint } from "./compile/manifest.ts";
import { type CompiledFlow, copyGraph, type Flow } from "./compile/model.ts";
import { typeRelation } from "./compile/validation.ts";
import { ComposedFlow } from "./compose/build/pipeline.ts";
import { CLOSE_MATCH_CUTOFF } from "./compose/helpers/tags.ts";
import { memoryToolNames, memoryTypeTools } from "./compose/helpers/memory.ts";
import { loadFlowSource } from "./compose/loader.ts";
import { childSignature } from "./compose/nodes/resolver.ts";
import { checkRecorded, current, type Defaults, type Env, nonDefaults, processEnv, withEnv } from "./defaults.ts";
import { compareCodeUnits, type Failure, getCloseMatches, noteCleanup, reprText } from "./diagnostics.ts";
import { CallClosed, RunCancelled, type RunEvent, RunFailed, RunSucceeded, type StreamChunk } from "./events.ts";
import { ExpressionError } from "./expr/expressions.ts";
import { checkTemplateCalls } from "./expr/template.ts";
import { LLMConfig, popHostCredentials } from "./llm_clients/config.ts";
import {
  type AgentConfig,
  blockProblem,
  checkModelConfig,
  type ConnectionSnapshot,
  connectionSnapshot,
  HEADER_NAMES_KEY,
  HOST_KEY_MARKER,
  headerNameDiff,
  legacyConnectionConfig,
  legacyConnectionMap,
  refusedBlock,
  type SelectionRecord,
  splitModelConfig,
} from "./llm_clients/connection.ts";
import { validateHeaders } from "./llm_clients/connection_settings.ts";
import { defaultModelAccess, type ModelAccess } from "./llm_clients/factory.ts";
import { Message, NamedTool, type Request, ToolCallPart, ToolSpec } from "./llm/messages.ts";
import { isMemory, type MemoryBase } from "./memory/base.ts";
import { MEMORY_SCOPE_LABEL } from "./memory/tools.ts";
import {
  type FoldedSteps,
  foldSteps,
  fromModelMessage,
  type Message as MessageValue,
  type NumberedStep,
  type StepExchange,
} from "./nodes/agent/messages.ts";
import { staticPathOf } from "./nodes/ids.ts";
import { typeToSchema } from "./nodes/agent/structured.ts";
import { type CompleteResult, type LocalToolSpec, type Node, NodeKind, type RebuildCaps } from "./nodes/base.ts";
import { applyDefaults, BindingError, checkInputNames, coerceInputs, copyValue } from "./nodes/binding.ts";
import {
  type HostToolCall,
  isExactly,
  iterSourceStrings,
  type OutputContract,
  type Parameter,
} from "./nodes/contracts.ts";
import { FileLockError, isMissingPath, realScheduler, type Scheduler, syncFolder, VERSION, warn } from "./platform.ts";
import { type DeliveryOp, FlowEngine, prepareDelivery, resumedAttemptEvents } from "./runtime/engine.ts";
import { LiveChannel } from "./runtime/live.ts";
import { type MemoryBinding, type MemoryDrift, RunMemories } from "./runtime/memories.ts";
import { bindStarted } from "./runtime/eval_node.ts";
import {
  decodeRecord,
  encodeRecord,
  isRunLogLocked,
  InterruptedRecord,
  iterRunRecords,
  JsonlRunLog,
  RunLogChangedError,
  MemoryRunLog,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  readRunLog,
  type RecordedSkill,
  type RecordedTool,
  type RecordEnvelope,
  type RunLog,
  type RunLogEntry,
  RunFailedRecord,
  type RunRecord,
  RunResumedRecord,
  RunStartedRecord,
} from "./runtime/log.ts";
import { emptyRunScan, foldRunScan, scanStatus, unfinishedStatus } from "./runtime/run_index.ts";
import {
  applyWaitsRecord,
  type Call,
  dueTimeouts,
  unreadEvents,
  type WaitInfo,
  WaitsState,
  wakeableNodes,
  wakeAt,
  waitInfos,
  withdrawableEvents,
} from "./runtime/waits.ts";
import {
  acceptedCommits,
  addCrashedAttempts,
  commitInitialPool,
  commitPrepared,
  DeliveryRefused,
  type PreparedTransition,
  prepareRunResumed,
  prepareRunStarted,
  RunState,
  replayRecords,
  stepIndexAt,
  stepsWithSteerValues,
} from "./runtime/state.ts";
import { Skill } from "./skills/catalog.ts";
import { isAlreadyOpenError, isMcpServer, type McpServer } from "./tools/mcp.ts";
import { ASK, ASK_USER, BUILTIN_TOOL_IDS, type BuiltinToolId, SLEEP, TAKE } from "./tools/context.ts";
import {
  checkVisibleNames,
  expandGrants,
  GrantError,
  type GrantedTool,
  type GrantScope,
  localGroupScope,
  reachedGroups,
  requestToolSpecs,
  type ToolGrant,
  visibleToolName,
} from "./tools/grants.ts";
import { RESERVED_TOOL_NAMES, type ToolEntry, type ToolRef, ToolSet } from "./tools/registry.ts";
import { toolTypes } from "./tools/types.ts";
import { type JsonObject, logJson } from "./typesys/json.ts";
import { VariablePool } from "./typesys/pool.ts";
import { defaultRunId, seedSystemClock } from "./typesys/seeding.ts";
import {
  canonicalTypeLabel,
  fromHost,
  isInboxType,
  isMemoryType,
  isRecord,
  MemoryRef,
  rejectDeepNesting,
  setField,
  type Type,
  TypeCheckError,
  type TypedValue,
  type Value,
  ValueKind,
  type ValueRecord,
  writeAsText,
} from "./typesys/values.ts";

export {
  checkConversation,
  checkMessage,
  ConversationError,
  type Message,
  type MessagePart,
} from "./nodes/agent/messages.ts";
export { staticPathOf } from "./nodes/ids.ts";
export {
  LiveChannel,
  type LiveCall,
  type LiveChannelOptions,
  type LiveFollowFrame,
  type LiveFrame,
  type LiveSnapshot,
  LiveView,
  type LiveViewCall,
} from "./runtime/live.ts";
export {
  type IterRunRecordsOptions,
  iterRunRecords,
  PoolCommitRecord,
  RunLogChangedError,
  type RunLogEntry,
  RunLogLineTooLongError,
  RunLogSchemaError,
  runRecordOffsets,
} from "./runtime/log.ts";
export {
  type JsonMode,
  type JsonObject,
  JSONDecodeError,
  type JsonValue,
  logJson,
  parseJson,
} from "./typesys/json.ts";
export {
  FileRef,
  InboxRef,
  MemoryRef,
  type StorageKind,
  TypedValue,
  type Value,
  type ValueRecord,
} from "./typesys/values.ts";
export type { RecordedMemoryTool } from "./typesys/memory.ts";
export type { WaitInfo } from "./runtime/waits.ts";
export type { MemoryDrift } from "./runtime/memories.ts";
export type { CompleteResult } from "./nodes/base.ts";
export type { DeliveryOp } from "./runtime/engine.ts";
export { DeliveryRefused, type DeliveryRefusal } from "./runtime/state.ts";
export {
  readRunIndex,
  RunIndexBusyError,
  type RunIndexMemory,
  type RunIndexRow,
  type RunIndexStatus,
  updateRunIndex,
} from "./runtime/run_index.ts";

/** Sees each durable observer, and each in-loop model and tool event. */
export type EventHook = (event: RunEvent) => unknown;
/** Sees each transient stream chunk. */
export type StreamHook = (chunk: StreamChunk) => unknown;

/** Where a run stands when a run call returns: a terminal result, or waiting on calls. */
export type RunResultStatus = "succeeded" | "failed" | "waiting" | "cancelled";

/**
 * The surface-independent result of a run call: a terminal result, or a run waiting on calls.
 *
 * `events` holds only the observers committed during this call.
 */
export class RunResult {
  /** The canonical caller-supplied inputs `RunStarted` retained. */
  readonly input: Record<string, unknown>;
  /**
   * `"succeeded"`, `"failed"` or `"cancelled"` at a terminal result; `"waiting"` when the run
   * stopped at quiescence with nodes waiting on calls, which a delivery or `advance` moves on.
   */
  readonly status: RunResultStatus;
  /** The durable identity shared by the state and its run log. */
  readonly runId: string;
  /** The JSONL log path as given to `runFlow` or `advance`, or `null` for a process-local run. */
  readonly log: string | null;
  /** The terminal flow value on success; otherwise `null`. */
  readonly output: unknown;
  /** The canonical terminal failure on failure; otherwise `null`. */
  readonly failure: Failure | null;
  /** The post-durable observers of this call only. */
  readonly events: readonly RunEvent[];
  /**
   * The number of accepted starts of each node that a resume retried with no recorded stop
   * (crashed attempts), by node id; empty for a run never resumed.
   */
  readonly crashedAttempts: ReadonlyMap<string, number>;
  /**
   * The memories that changed since the run started (including the run's own writes), as this
   * call's crash resume found them: one per memory whose version isn't the one recorded at the
   * run's start; frozen, and empty for a fresh run and any other call.
   */
  readonly memoryDrift: readonly MemoryDrift[];
  /**
   * The calls a waiting run waits on, by waiting node in `node_waiting` order; empty otherwise. A
   * call `settled` as the run stopped wakes its node at the next `advance`.
   */
  readonly waits: readonly WaitInfo[];
  /**
   * The send seqs no take read, by inbox, at a terminal status; empty while the run waits (they
   * may still be read) and for an inbox with none.
   */
  readonly unread: ReadonlyMap<string, readonly number[]>;

  /**
   * Hold the given fields; `output` and `failure` default to `null`, `memoryDrift` and `waits` to
   * none and `unread` to an empty map.
   */
  constructor(fields: {
    input: Record<string, unknown>;
    status: RunResultStatus;
    runId: string;
    log: string | null;
    output?: unknown;
    failure?: Failure | null;
    events?: readonly RunEvent[];
    crashedAttempts?: ReadonlyMap<string, number>;
    memoryDrift?: readonly MemoryDrift[];
    waits?: readonly WaitInfo[];
    unread?: ReadonlyMap<string, readonly number[]>;
  }) {
    this.input = fields.input;
    this.status = fields.status;
    this.runId = fields.runId;
    this.log = fields.log;
    this.output = fields.output ?? null;
    this.failure = fields.failure ?? null;
    this.events = fields.events ?? [];
    this.crashedAttempts = fields.crashedAttempts ?? new Map();
    this.memoryDrift = Object.freeze([...(fields.memoryDrift ?? [])]);
    this.waits = fields.waits ?? [];
    this.unread = fields.unread ?? new Map();
  }
}

/** A writer in this process holds the run's lock and no engine runs yet. */
interface HeldRun {
  readonly kind: "held";
  /** Resolves when the holder goes live or releases; a delivery then reads `RUNS` again. */
  readonly settled: Promise<void>;
  /** Resolves when the holder releases; an advance then reads `RUNS` again. */
  readonly done: Promise<void>;
}

/** A run this process is moving: deliveries go to its engine while its loop runs. */
interface LiveRun {
  readonly kind: "live";
  readonly engine: FlowEngine;
  /** Resolves when the holder releases. */
  readonly done: Promise<void>;
}

/** A waiting run with no writer: its state, kept while the log stays the file it left. */
interface IdleRun {
  readonly kind: "idle";
  readonly state: RunState;
  readonly flow: ComposedFlow;
  /** The code-built flow the run was given, or `null` for one recompiled from its recorded files. */
  readonly hostFlow: ComposedFlow | null;
  readonly apiKey: string | null;
  readonly headers: Readonly<Record<string, string>> | null;
  /** The seq of the log's last record when the run was left. */
  readonly lastSeq: number;
  /** The log file's size, device and inode when the run was left. */
  readonly size: bigint;
  readonly dev: bigint;
  readonly ino: bigint;
}

/**
 * What this process holds of each durable run, by its log's real path. Only waiting runs stay idle
 * here, each with its whole `RunState`: an entry goes when a call here sees the run move on, but a
 * run another process moved on keeps its stale entry until a call here reads the log again.
 * Dropping one costs a rebuild.
 */
const RUNS = new Map<string, HeldRun | LiveRun | IdleRun>();

/** This call's hold on a run: `live` hands it to an engine, `release` (once, always) ends it. */
interface RunHold {
  /** The idle entry this hold replaced, or `null`. */
  readonly idle: IdleRun | null;
  live(engine: FlowEngine): void;
  release(next: IdleRun | null): void;
}

/**
 * Make this call the holder of `key`, replacing `idle` or nothing. Read `RUNS` and call this in one
 * tick. `live` and `release` change `RUNS` only while the entry there is still this hold's, so a
 * stale hold can't remove a newer one.
 */
function hold(key: string, idle: IdleRun | null): RunHold {
  const settled = Promise.withResolvers<void>();
  const done = Promise.withResolvers<void>();
  let mine: HeldRun | LiveRun = { kind: "held", settled: settled.promise, done: done.promise };
  RUNS.set(key, mine);
  let released = false;
  return {
    idle,
    live(engine) {
      if (RUNS.get(key) === mine) {
        mine = { kind: "live", engine, done: done.promise };
        RUNS.set(key, mine);
      }
      settled.resolve();
    },
    release(next) {
      if (released) return;
      released = true;
      if (RUNS.get(key) === mine) {
        if (next === null) RUNS.delete(key);
        else RUNS.set(key, next);
      }
      settled.resolve();
      done.resolve();
    },
  };
}

/**
 * Another writer holds the run's log. `cause` is the busy `FileLockError` when another process (or
 * an outside holder) has the lock: route the call to it. It is absent for an `advance` of a run
 * this process's engine is moving: advance again once that run call returns.
 */
export class RunBusyError extends Error {
  override name = "RunBusyError";
  /** The log's absolute path. */
  readonly log: string;

  /** Name the log at the absolute path `log` in the lock's busy message. */
  constructor(log: string, options?: ErrorOptions) {
    super(`run log ${log} is locked by another writer (is the run still active?)`, options);
    this.log = log;
  }
}

/**
 * Open the log at `locator` under its lock.
 *
 * @throws RunBusyError when another writer holds the lock; the log's other open errors.
 */
function openLog(locator: string, clock: () => number): JsonlRunLog {
  try {
    return JsonlRunLog.resume(locator, { clock });
  } catch (error) {
    const cause = (error as Error | null)?.cause;
    if (cause instanceof FileLockError && cause.reason === "busy") throw new RunBusyError(locator, { cause });
    throw error;
  }
}

/**
 * Wait until no other call in this process holds `key`, then hold it.
 *
 * @throws RunBusyError when this process's engine is moving the run (its loop runs).
 */
async function holdForAdvance(key: string, locator: string): Promise<RunHold> {
  for (;;) {
    const entry = RUNS.get(key);
    if (entry?.kind === "live" && entry.engine.looping) throw new RunBusyError(locator);
    if (entry !== undefined && entry.kind !== "idle") {
      // A holder setting the run up, or one whose loop has ended and is closing it.
      await entry.done;
      continue;
    }
    return hold(key, entry ?? null);
  }
}

/**
 * Hand the run at `key` to `onLive` when this process's engine is moving it, in the same tick as
 * the check, else hold it. A run another call is setting up is waited for until it goes live or
 * releases; one whose loop has ended is waited for until its holder releases it, since a delivery
 * the ended loop can't read must be appended where the next `advance` finds it.
 */
async function liveOrHold<T>(
  key: string,
  onLive: (engine: FlowEngine) => T,
): Promise<{ readonly delivered: T } | { readonly hold: RunHold }> {
  for (;;) {
    const entry = RUNS.get(key);
    if (entry?.kind === "live") {
      if (entry.engine.looping) return { delivered: onLive(entry.engine) };
      await entry.done;
      continue;
    }
    if (entry?.kind === "held") {
      await entry.settled;
      continue;
    }
    return { hold: hold(key, entry ?? null) };
  }
}

/**
 * Return `idle` when the log is the same file, ends where it left it, and this call would rebuild
 * the same state; else `null`. Credentials matter only when given: `advance` always gives them,
 * and a delivery, which calls no model, only when its caller does.
 */
function reusable(
  idle: IdleRun | null,
  authority: JsonlRunLog,
  key: string,
  want: {
    hostFlow: ComposedFlow | null;
    credentials?: { apiKey: string | null; headers: Readonly<Record<string, string>> | null };
  },
): IdleRun | null {
  if (idle === null) return null;
  const stat = statSync(key, { bigint: true });
  if (idle.state.runId !== authority.runId || idle.lastSeq !== authority.nextSequence - 1) return null;
  if (idle.size !== stat.size || idle.dev !== stat.dev || idle.ino !== stat.ino) return null;
  if (idle.hostFlow !== want.hostFlow) return null;
  const credentials = want.credentials;
  if (
    credentials !== undefined &&
    (idle.apiKey !== credentials.apiKey || !sameHeaders(idle.headers, credentials.headers))
  ) {
    return null;
  }
  return idle;
}

/** Whether two header maps hold the same names and values. */
function sameHeaders(a: Readonly<Record<string, string>> | null, b: Readonly<Record<string, string>> | null): boolean {
  return logJson(a ?? {}) === logJson(b ?? {});
}

/** The idle entry for a waiting `state`, read while the lock is held; `null` for any other state. */
function idleRun(
  state: RunState,
  flow: ComposedFlow,
  key: string,
  authority: JsonlRunLog,
  keep: Pick<IdleRun, "hostFlow" | "apiKey" | "headers">,
): IdleRun | null {
  if (state.result !== null || state.waits.waiting.size === 0) return null;
  const stat = statSync(key, { bigint: true });
  return {
    kind: "idle",
    state,
    flow,
    ...keep,
    lastSeq: authority.nextSequence - 1,
    size: stat.size,
    dev: stat.dev,
    ino: stat.ino,
  };
}

/**
 * Close what a run call opened and hand the run on, whatever fails.
 *
 * 1. Servers close, then memory sessions. Neither rejects (`closeRunServers`, `RunMemories.close`):
 *    a memory's close error is noted on `primary`, else warned. Only log close errors are noted
 *    below.
 * 2. Then, in one synchronous step: the next idle entry is built (after every await, so it holds
 *    every record appended under this hold), the log closes, and the hold is released.
 *
 * The entry is built only with a hold, and only when nothing failed or `keepAfter(primary)` says a
 * failure changed nothing (a refused delivery). Cleanup errors are noted on `primary`; with no
 * `primary`, the first one is thrown.
 */
async function closeAndRelease(parts: {
  servers: readonly McpServer[];
  memories: RunMemories | null;
  authority: RunLog | null;
  primary: { error: unknown } | null;
  hold: RunHold | null;
  keep: () => IdleRun | null;
  keepAfter?: (primary: { error: unknown }) => boolean;
}): Promise<void> {
  const { primary, hold: runHold } = parts;
  let pending = primary;
  const note = (operation: string, error: unknown): void => {
    if (pending === null) pending = { error };
    else noteCleanup(pending.error, error, `${operation} also raised`);
  };
  await closeRunServers(parts.servers);
  // A memory that fails to close can't change what the run recorded: it is noted on the call's
  // error, or else warned, as a host's other failed cleanups are, and never thrown.
  for (const { identity, error } of (await parts.memories?.close()) ?? []) {
    if (primary !== null) noteCleanup(primary.error, error, `closing memory \`${identity}\` also raised`);
    else warn(`memory \`${identity}\` didn't close: ${errorMessage(error)}`);
  }
  let next: IdleRun | null = null;
  try {
    // Cleanup errors after a refusal are only noted on it, so the entry is still kept; only a
    // failed log close below drops it.
    const keepable = pending === null || (primary !== null && parts.keepAfter?.(primary) === true);
    if (runHold !== null && keepable) next = parts.keep();
    if (parts.authority !== null) {
      try {
        parts.authority.close();
      } catch (error) {
        next = null;
        note("run-log close", error);
      }
    }
  } catch (error) {
    next = null;
    note("run registry", error);
  } finally {
    runHold?.release(next);
  }
  if (primary === null && pending !== null) throw pending.error;
}

/** The options of one `runFlow` call. */
export interface RunFlowOptions {
  /**
   * A new JSONL log path. `null` (the default) uses a process-local `MemoryRunLog`; an existing
   * path is never overwritten.
   */
  log?: string | null;
  /**
   * The non-empty identity exposed as `system.run_id` and retained by every record. `null` (the
   * default) mints a fresh one.
   */
  runId?: string | null;
  /** The non-negative worker bound recorded at initialization (default 8); zero runs serially. */
  numWorkers?: number;
  /**
   * Sees each observer after its record is appended and reduced, plus the transient in-loop AGENT
   * events. Called synchronously; a throw, or a rejection of the promise it returns, rejects the run
   * with that error.
   */
  onEvent?: EventHook | null;
  /** Sees each transient stream chunk; called as `onEvent` is. */
  onStream?: StreamHook | null;
  /**
   * The run's live channel: gets each record once it is appended, and each AGENT model call's
   * start, chunks and end, for a viewer to follow the run (`LiveChannel`). A subscriber that throws
   * is dropped and reported by the channel; it never affects the run.
   */
  live?: LiveChannel | null;
  /**
   * The per-run outer model configuration layer, taken through host intake (`{max_tokens: 1000}`
   * holds the int `1000n`). Its `api_key` goes only to the agents of its `provider`, with its
   * `base_url`, `organization`, `project` and `headers`. The key and the header values are never
   * recorded; `base_url`, `organization`, `project` and the header names are, on the agents that
   * use the key. A key no agent uses is dropped with its headers.
   */
  llmConfig?: Readonly<Record<string, unknown>> | null;
  /**
   * The tools this run sees, keyed by tool id; `null` (the default) uses the registered tools.
   * A record, even an empty one, is the whole set. An `McpServer` that isn't open is opened for
   * this run and closed when it ends; one the host opened stays open.
   */
  tools?: Readonly<Record<string, ToolEntry>> | null;
  /** The skills this run can grant, keyed by `Skill.name`. */
  skills?: ReadonlyMap<string, Skill> | null;
  /** The host's app object a tool receives as `ctx.context`; never recorded. */
  context?: unknown;
  /**
   * The host's stop: aborting it stops the run, which rejects with its reason. An already-aborted
   * signal rejects before anything is created.
   */
  signal?: AbortSignal | null;
  /**
   * The environment for connection selection, credentials, model building and the run's `YIAN_*`
   * values. Without one, those read `process.env`, and the `YIAN_*` values are the ones already in
   * effect: an enclosing `withEnv` scope's, else this process's.
   */
  env?: Env | null;
  /** For tests: the clock the system clock is seeded from and node waits run on (default `realScheduler`). */
  scheduler?: Scheduler | null;
  /** For tests: a draw in [0, 1) for backoff jitter (default `Math.random`). */
  random?: (() => number) | null;
  /** For tests: the model builder AGENT nodes use (default `defaultModelAccess(env)`). */
  models?: ModelAccess | null;
  /**
   * The run that started this one, recorded in `RunStarted`: a host-started child run, and the
   * parent's node that started it when one did. `null` (the default) when
   * none did.
   */
  parent?: { runId: string; node?: string | null } | null;
}

/**
 * Start and drive one flow through an in-memory or durable run log.
 *
 * Flow failures resolve as a `RunResult` with `status` `"failed"`. Host, callback, log and
 * reducer errors reject after any running nodes have settled and the log has been closed.
 *
 * @param flow The loaded flow to copy and execute.
 * @param inputs Caller values keyed by declared input name; any other key is an error. Each is
 *   taken through host intake (an own data property, `{n: 3}` becomes the int `3n`), and
 *   supplied text is coerced to a declared scalar; START applies the declared defaults and the
 *   full boundary type checks as its recorded execution. A memory input takes a `MemoryBase`
 *   instance (an Optional one may be left out or `null`): before `RunStarted` each instance is
 *   opened once and snapshotted, its value is recorded in START's inputs, and its session closes
 *   when the run's segment ends. A session that fails to close doesn't change the result: its
 *   error is noted on the call's error, if it failed, or else reported as a warning (`warn`, which
 *   `withWarnings` receives).
 * @param options See `RunFlowOptions`.
 * @returns The terminal success or failure, with this call's post-durable observers.
 * @throws TypeError when `flow` isn't a `ComposedFlow`, `signal` isn't an `AbortSignal`, `live`
 *   isn't a `LiveChannel`, `onEvent` or `onStream` isn't a function, `inputs` isn't a record (a
 *   Proxy included), `llmConfig` is a primitive or an array, a `tools` entry is malformed, `skills`
 *   isn't a `Map` of `Skill`s each keyed by its name, `scheduler` lacks `now` or `sleep`, `random`
 *   isn't a function, or `models` lacks `env` or `createModel`.
 * @throws RangeError when `learnable` or `memory`, a removed option, is given.
 * @throws ExpressionError when a condition, binding or prompt set in code can't run.
 * @throws TypeCheckError when an input holds an accessor, NaN, a lone surrogate, a cycle or
 *   another non-yian value, nests deeper than `YIAN_MAX_VALUE_NESTING` levels, or has text that
 *   can't be read; or when `llmConfig` is or holds a non-yian value (a `Map`, a class instance, a
 *   Proxy).
 * @throws RangeError when an input key names no declared input, or a memory input isn't given a
 *   `MemoryBase` instance; two instances hold one memory identity, or an instance's type isn't
 *   its input's; a memory grant's input is also a host tool or group of the run, or its tool
 *   shows under a name the node's host grants show; an AGENT grants a tool, group or
 *   skill this run lacks; a TOOL node names a tool this run lacks, or its inputs or output can't
 *   fit that tool's types; `llmConfig` or an agent's resolved model config is invalid, an agent
 *   has no provider or model, or `llmConfig.api_key` has no `provider`; or the identity, worker
 *   count or prepared start state breaks its contract; or a `tools` server was opened by
 *   another run in progress. A message never echoes the key.
 * @throws McpConnectError when a `tools` server doesn't start or doesn't list its tools.
 * @throws AuthError when the credential store can't be read.
 * @throws DefaultsError when a `YIAN_<NAME>` variable in `env` holds a value its entry doesn't accept.
 * @throws MemoryBusyError when a memory's lock stays held past its timeout as it is opened.
 * @throws Error naming the input when a memory fails to open or snapshot (`RunMemories.open`).
 * @throws The signal's reason when it aborts.
 * @throws Error (code `EEXIST`) when `log` names an existing path, and the `fs` error when the log
 *   can't be created, locked, written, synced or closed.
 */
export async function runFlow(
  flow: ComposedFlow,
  inputs: Readonly<Record<string, unknown>>,
  options: RunFlowOptions = {},
): Promise<RunResult> {
  // The scope makes everything the run reads, the engine's work included, see the run's values;
  // a bad YIAN_<NAME> value fails here, before any work.
  const env = options.env ?? processEnv;
  return withEnv(env, async () => {
    const runDefaults = current();
    if (!(flow instanceof ComposedFlow)) throw new TypeError("runFlow expects a ComposedFlow");
    const unopened = checkDriveOptions(options);
    const skills = checkFlowBeforeServers(flow.compiled, options.skills ?? null);
    return await startRun(flow, inputs, { ...options, skills }, unopened, runDefaults, env);
  });
}

/** The options `runFlow` and `advance` share, which drive the engine. */
type DriveOptions = Pick<
  RunFlowOptions,
  "tools" | "context" | "signal" | "live" | "onEvent" | "onStream" | "scheduler" | "random" | "models"
>;

/**
 * Refuse, by name, an option yian removed: `learnable` (learning) and, unless `takesMemory` (a
 * resume, which binds its run's memories again), `memory` (memories are flow inputs now).
 *
 * @throws RangeError when `options` holds either.
 */
function refuseRemovedOptions(options: object, takesMemory = false): void {
  if (Object.hasOwn(options, "learnable")) {
    throw new RangeError("learnable was removed: yian no longer learns from runs; drop the option");
  }
  if (!takesMemory && Object.hasOwn(options, "memory")) {
    throw new RangeError("the memory option was removed: pass each memory as a flow input");
  }
}

/**
 * Check the options `runFlow` and `advance` share before anything is opened or created, and split
 * off the servers the call opens.
 *
 * @returns The unopened `tools` servers, as `splitRunServers` returns them.
 * @throws TypeError when `signal` isn't an `AbortSignal`, `live` isn't a `LiveChannel`, `onEvent`
 *   or `onStream` isn't a function, `scheduler` lacks `now` or `sleep`, `random` isn't a function,
 *   `models` lacks `env` or `createModel`, or a `tools` entry is malformed.
 * @throws RangeError when a `tools` key is reserved, or a server was opened by another run in
 *   progress, or a removed option is given (`memory` unless `takesMemory`).
 * @throws The signal's reason when it has aborted.
 */
function checkDriveOptions(options: DriveOptions, takesMemory = false): [string, McpServer][] {
  refuseRemovedOptions(options, takesMemory);
  const { tools = null, context, signal = null, live = null, onEvent = null, onStream = null } = options;
  const { scheduler = null, random = null, models = null } = options;
  if (signal !== null && !(signal instanceof AbortSignal)) throw new TypeError("signal must be an AbortSignal or null");
  if (live !== null && !(live instanceof LiveChannel)) throw new TypeError("live must be a LiveChannel or null");
  if (onEvent !== null && typeof onEvent !== "function") throw new TypeError("onEvent must be a function or null");
  if (onStream !== null && typeof onStream !== "function") throw new TypeError("onStream must be a function or null");
  if (
    scheduler !== null &&
    (typeof scheduler !== "object" || typeof scheduler.now !== "function" || typeof scheduler.sleep !== "function")
  ) {
    throw new TypeError("scheduler must be a Scheduler or null");
  }
  if (random !== null && typeof random !== "function") throw new TypeError("random must be a function or null");
  requireModels(models);
  signal?.throwIfAborted();
  const [otherTools, unopened] = splitRunServers(tools);
  // The other entries' errors come before anything is opened.
  new ToolSet(otherTools, { context });
  return unopened;
}

/**
 * Check a host model access's shape.
 *
 * @throws TypeError when `models` isn't `null` and lacks a function `env` and `createModel`.
 */
function requireModels(models: unknown): void {
  const access = models as Partial<ModelAccess> | null | undefined;
  if (models !== null && !(typeof access?.env === "function" && typeof access.createModel === "function")) {
    throw new TypeError("models must provide env(name) and createModel(config, options), or be null");
  }
}

/**
 * Run the checks of `flow` that need no tools, so they fail before any server starts.
 *
 * @returns The skills the run grants, as `checkSkills` returns them.
 * @throws What `checkConditions`, `checkTemplates` and `checkSkills` throw.
 */
function checkFlowBeforeServers(
  flow: CompiledFlow,
  skills: ReadonlyMap<string, Skill> | null,
): Map<string, Skill> | null {
  checkConditions(flow);
  checkTemplates(flow);
  return checkSkills(flow, skills);
}

/**
 * @internal Take each supplied input through host intake, keyed as given. A memory input is left
 * out: `memoryBindings` reads its instance.
 *
 * @throws TypeError when `inputs` isn't a record or is a Proxy.
 * @throws RangeError when a key names no declared input, names an inbox, which takes events
 *   through `send` instead, or gives a memory input anything but a `MemoryBase` instance (or null,
 *   for an Optional one).
 * @throws TypeCheckError when an input is an accessor property or isn't a yian value.
 */
export function hostInputs(params: readonly Parameter[], inputs: unknown): Record<string, Value> {
  // Checked first: reading a Proxy's prototype or keys would run its traps.
  if (types.isProxy(inputs)) throw new TypeError("inputs must be a mapping of input names to values, got Proxy");
  checkInputNames(params, inputs);
  const record = inputs as Readonly<Record<string, unknown>>;
  const memories = new Set<string>();
  for (const param of params) {
    if (isInboxType(param.contract?.type) && Object.hasOwn(record, param.name)) {
      throw new RangeError(`inbox \`${param.name}\` receives events through send(); it takes no run input`);
    }
    const typ = param.contract?.type;
    if (!isMemoryType(typ) || !Object.hasOwn(record, param.name)) continue;
    const value = suppliedValue(record, param.name);
    // An Optional memory given null is left out of the START inputs, as one left out is: never
    // recorded as null.
    if (!isMemory(value) && !(value === null && typ.nullable)) {
      throw new RangeError(`memory \`${param.name}\` is bound to a MemoryBase instance, not a value`);
    }
    memories.add(param.name);
  }
  const taken: ValueRecord = {};
  for (const name of Object.keys(record)) {
    if (memories.has(name)) continue;
    setField(taken, name, fromHost(suppliedValue(record, name), `input ${reprText(name)}`));
  }
  return taken;
}

/**
 * Return the supplied input `name`, read from its data property.
 *
 * @throws TypeCheckError when it is an accessor property, whose getter isn't run.
 */
function suppliedValue(record: Readonly<Record<string, unknown>>, name: string): unknown {
  const descriptor = Object.getOwnPropertyDescriptor(record, name)!;
  if (!("value" in descriptor)) throw new TypeCheckError(`input ${reprText(name)}: accessor property is not a yian value`);
  return descriptor.value;
}

/**
 * Return the memory inputs `inputs` binds to `MemoryBase` instances, in declaration order, each
 * with its input's memory type and item type. `hostInputs` checked them first.
 */
function memoryBindings(params: readonly Parameter[], inputs: Readonly<Record<string, unknown>>): MemoryBinding[] {
  const bindings: MemoryBinding[] = [];
  for (const param of params) {
    const typ = param.contract?.type;
    if (!isMemoryType(typ) || !Object.hasOwn(inputs, param.name)) continue;
    const instance = suppliedValue(inputs, param.name);
    if (!isMemory(instance)) continue;
    const itemType = typ.element === undefined ? null : typeToSchema(typ.element);
    bindings.push({ input: param.name, instance, type: typ.memory, itemType });
  }
  return bindings;
}

/**
 * Return the value each memory input recorded in the run's START inputs, as `state`'s pool read
 * them back, by name; an Optional memory left out recorded none.
 */
function startMemories(params: readonly Parameter[], state: RunState): Map<string, MemoryRef> {
  const inputs = state.pool.store.get(state.flow.startId)?.value;
  const recorded = new Map<string, MemoryRef>();
  if (!isRecord(inputs)) return recorded;
  for (const param of params) {
    if (!isMemoryType(param.contract?.type) || !Object.hasOwn(inputs, param.name)) continue;
    const value = inputs[param.name];
    if (MemoryRef.isMemoryRef(value)) recorded.set(param.name, value);
  }
  return recorded;
}

/**
 * Return the bindings of a resume's `memory` option, checked against the run's memory inputs and
 * the values they recorded. An Optional memory given `null` is left out, as `runFlow` leaves it.
 *
 * @throws TypeError when `memory` isn't a record, or gives an input something other than a
 *   `MemoryBase` instance (or `null`, for an Optional one).
 * @throws RangeError when it names an input that isn't a memory input of the flow, or gives an
 *   instance to one recorded null when the run started (an Optional memory left out).
 */
function resumeBindings(
  params: readonly Parameter[],
  memory: unknown,
  recorded: ReadonlyMap<string, MemoryRef>,
): MemoryBinding[] {
  if (memory === null) return [];
  if (!isRecord(memory) || types.isProxy(memory)) {
    throw new TypeError("memory must be a record of memory input names to MemoryBase instances");
  }
  for (const name of Object.keys(memory)) {
    const param = params.find((each) => each.name === name);
    const typ = param?.contract?.type;
    if (!isMemoryType(typ)) throw new RangeError(`memory \`${name}\` isn't a memory input of this run's flow`);
    const value = suppliedValue(memory, name);
    if (value === null && typ.nullable) continue;
    if (!isMemory(value)) throw new TypeError(`memory \`${name}\` takes a MemoryBase instance`);
    if (!recorded.has(name)) throw new RangeError(`memory \`${name}\` was null when the run started; drop it`);
  }
  return memoryBindings(params, memory);
}

/** A run's `RunStarted.call_time_llm`: the call-time agents' bases and the selection they bind from. */
type CallTimeLlm = {
  selection: SelectionRecord;
  agents: Record<string, AgentConfig>;
};

/**
 * Return whether a run's recorded agents may use the host key: a static agent is marked for it, or
 * the call-time agents' selection names its provider.
 *
 * @param effective The recorded effective model map, keyed by static agent path.
 * @param callTime The recorded `RunStarted.call_time_llm`, or `null`.
 */
function usesHostKey(
  effective: Readonly<Record<string, Readonly<AgentConfig>>>,
  callTime: Readonly<CallTimeLlm> | null,
): boolean {
  return hostKeyProviders(effective).size > 0 || (callTime !== null && callTime.selection.host !== null);
}

/**
 * Return the host key the call-time agents' selection names, for a resume check.
 *
 * A call-time agent that has not started may still resolve to that key's provider, so a resume
 * that can't supply the key is refused up front, as for a marked agent.
 *
 * @param callTime The recorded `RunStarted.call_time_llm`, or `null`.
 * @returns `{provider, header_names}` (header names only when recorded), or `null` when the
 *   selection names no host key.
 */
function selectionHost(callTime: Readonly<CallTimeLlm> | null): Record<string, unknown> | null {
  const host = callTime === null ? null : callTime.selection.host;
  if (host === null) return null;
  const selected: Record<string, unknown> = { provider: host["provider"] };
  const names = host["header_names"] as readonly string[] | undefined;
  if (names !== undefined && names.length > 0) selected["header_names"] = [...names];
  return selected;
}

/**
 * Check that the host's headers have the names recorded for its key's agents; names compare
 * without case, as HTTP compares them.
 *
 * @param effective The recorded effective model map, keyed by static agent path.
 * @param headers The headers given for the host key; `null` gives none.
 * @param callTime The recorded `RunStarted.call_time_llm`; the header names its selection records
 *   for the host key are expected too.
 * @throws RangeError when the names differ; the message names the missing and unrecorded header
 *   names, never a value.
 */
function checkHostHeaderNames(
  effective: Readonly<Record<string, Readonly<AgentConfig>>>,
  headers: Readonly<Record<string, unknown>> | null,
  callTime: Readonly<CallTimeLlm> | null,
): void {
  const expected: string[] = [];
  for (const config of Object.values(effective)) {
    if (config[HOST_KEY_MARKER] === true) expected.push(...((config[HEADER_NAMES_KEY] as string[] | undefined) ?? []));
  }
  const host = callTime === null ? null : callTime.selection.host;
  if (host !== null) expected.push(...((host["header_names"] as string[] | undefined) ?? []));
  const diff = headerNameDiff(expected, Object.keys(headers ?? {}));
  if (diff !== null) {
    throw new RangeError(`headers must have the header names this run recorded for its host key: ${diff}`);
  }
}

/** Validate a host key without echoing it in errors. */
function secretKey(raw: string): string {
  return new LLMConfig({ api_key: raw }).apiKey!;
}

/** Validate a host key and the headers sent with it, without echoing a value; a missing one is `null`. */
function hostCredentials(
  apiKey: string | null | undefined,
  headers: Readonly<Record<string, string>> | null | undefined,
): { apiKey: string | null; headers: Record<string, string> | null } {
  return {
    apiKey: apiKey == null ? null : secretKey(apiKey),
    headers: headers == null ? null : validateHeaders(headers, null),
  };
}

/**
 * Reject any agent config its model call would reject, before the run starts.
 *
 * @param effectiveLlm The effective model map, keyed by static agent path.
 * @param apiKey The host key, supplied to each config marked for it.
 * @param headers The header values sent with the host key, supplied beside it.
 * @param labels The authored site of each synthesized agent's `llm_config`, keyed by static agent
 *   path; a message names it instead of the path.
 * @param located The `llm_config` subject of each agent whose layers carry values
 *   (`valueConfigSubjects`), keyed by static agent path; its message names the subject, and the
 *   provider's block when the block is what is refused.
 * @throws RangeError when a config would be rejected by its model call; the message names its agent
 *   path, or the authored site of a synthesized agent.
 */
function checkEffectiveLlm(
  effectiveLlm: Readonly<Record<string, Readonly<AgentConfig>>>,
  apiKey: string | null,
  headers: Readonly<Record<string, string>> | null,
  labels: ReadonlyMap<string, string>,
  located: ReadonlyMap<string, string> | null = null,
): void {
  for (const [path, config] of Object.entries(effectiveLlm)) {
    // The host key and header values stay out of the map; supply them where the marker expects
    // them.
    let probe = config;
    if (HOST_KEY_MARKER in config) {
      const supplied: AgentConfig = { ...config, api_key: apiKey };
      if (headers !== null && Object.keys(headers).length > 0) supplied["headers"] = headers;
      probe = supplied;
    }
    try {
      checkModelConfig(probe);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      const subject = located?.get(path);
      if (subject !== undefined) {
        throw new RangeError(blockProblem(subject, refusedBlock(probe).slice(1), error.message), { cause: error });
      }
      const where = labels.get(path) ?? `Agent ${reprText(path)}`;
      throw new RangeError(`${where}: ${error.message}`, { cause: error });
    }
  }
}

/**
 * @internal Return the top-level START's wiring for a run: each supplied input read from the run's
 * input record, and each inbox input bound to its handle. An omitted input is left out, so START
 * fills its default.
 *
 * @param params The flow's declared inputs.
 * @param coerced The supplied inputs, already coerced at the run boundary.
 */
export function startWiring(
  params: readonly Parameter[],
  coerced: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  const wiring: Record<string, unknown> = Object.fromEntries(
    Object.keys(coerced).map((name) => [name, `\${input.${name}}`]),
  );
  for (const param of params) if (isInboxType(param.contract?.type)) wiring[param.name] = { inbox: param.name };
  return wiring;
}

/**
 * @internal Return the run's inputs as run start resolves against them: the supplied inputs plus
 * each omitted input's valid default.
 *
 * START still binds and checks the inputs, so a missing required input or a default that fails its
 * check stays START's logged failure: such a default is left out here rather than thrown before the
 * log exists.
 *
 * @param params The flow's declared inputs.
 * @param coerced The supplied inputs, already coerced at the run boundary.
 */
export function runStartInputs(
  params: readonly Parameter[],
  coerced: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  let runInputs: Record<string, unknown> = { ...coerced };
  for (const parameter of params) {
    try {
      runInputs = applyDefaults([parameter], runInputs);
    } catch (error) {
      if (!(error instanceof BindingError)) throw error;
    }
  }
  return runInputs;
}

/**
 * @internal Resolve a fresh run's llm cascade over `compiled`, in place, with its inputs.
 *
 * Values that read only the run's inputs resolve against the supplied inputs, with omitted inputs'
 * defaults filled.
 *
 * @param compiled The run's private flow copy; its agents get their effective configs.
 * @param coerced The supplied inputs, already coerced at the run boundary.
 * @param llmConfig The host layer, without the host key and its settings.
 * @param snapshot The one selection snapshot that determines every agent's defaults.
 * @returns The resolved configurations, before any connection is bound.
 * @throws RangeError as `resolveRunLlm` throws it.
 */
export function runStartLlm(
  compiled: CompiledFlow,
  coerced: Readonly<Record<string, unknown>>,
  llmConfig: Readonly<Record<string, unknown>>,
  snapshot: ConnectionSnapshot,
): RunLLM {
  return resolveRunLlm(compiled, llmConfig, {
    defaults: snapshot.defaults,
    providerModels: snapshot.providerModels,
    strayModel: snapshot.strayModel,
    runInputs: runStartInputs(compiled.nodes.get(compiled.startId)!.params ?? [], coerced),
  });
}

/**
 * Open the run's MCP servers, check the tools, then record and drive a fresh run. What this opened
 * closes before the run is released, whatever fails.
 */
async function startRun(
  flow: ComposedFlow,
  inputs: Readonly<Record<string, unknown>>,
  options: RunFlowOptions,
  unopened: readonly [string, McpServer][],
  runDefaults: Defaults,
  env: Env,
): Promise<RunResult> {
  const { log = null, runId = null, numWorkers = 8 } = options;
  const live = options.live ?? null;
  const { tools = null, skills = null, context, signal = null } = options;
  const scheduler = options.scheduler ?? realScheduler;
  const models = options.models ?? defaultModelAccess(env);
  let runServers: McpServer[] = [];
  let memories: RunMemories | null = null;
  let authority: RunLog | null = null;
  let primary: { error: unknown } | null = null;
  let runHold: RunHold | null = null;
  let keep: () => IdleRun | null = () => null;
  try {
    runServers = await openRunServers(unopened, signal);
    const runTools = new ToolSet(tools, { context });
    const hostTypes = memoryTypeTools(flow.memoryTypes);
    checkToolGrants(flow.compiled, runTools);
    checkMemoryGrantNames(flow.compiled, runTools, hostTypes);
    checkToolTypes(flow.compiled, runTools);

    // Identity belongs to the authored program, before cascade resolution or any per-run copy can
    // change object topology.
    const authoredFingerprint = flowFingerprint(flow.compiled);
    const privateFlow = copyGraph(flow.compiled);

    // Convert supplied boundary values before RunStarted persists its input. This is the only
    // string conversion: a value crossing a CALL/MAP/LOOP edge is validated as is, and START stays
    // the authority for required/default handling and contract validation.
    const coerced = coerceInputs(flow.input, hostInputs(flow.input, inputs));
    for (const [name, value] of Object.entries(coerced)) rejectDeepNesting(value, `input ${reprText(name)}`);

    // The key stays out of the cascade so RunStarted never records it. Its base_url and connection
    // settings stay out too, so no authored layer can pair the key with another. Host intake makes
    // the copy they are taken from, with an integral number as an int.
    const runLlmConfig = fromHost(options.llmConfig ?? {}, "llm_config");
    if (!isRecord(runLlmConfig)) throw new TypeError("llmConfig must be a record or null");
    const host = popHostCredentials(runLlmConfig);
    let apiKey = host.apiKey;
    let headers = host.headers;
    // The idle entry keeps the given credentials, as `advance`'s does, before a flow that sends no
    // host key drops them below: a later call passing the same ones then reuses it.
    const credentials = { apiKey, headers };

    // One selection snapshot determines every static agent before RunStarted, and what each agent
    // resolved only when it runs binds from.
    const snapshot = connectionSnapshot({ env });
    const hostKeyProvider = apiKey === null ? null : (runLlmConfig["provider"] as string);
    const runLlm = runStartLlm(privateFlow, coerced, runLlmConfig, snapshot);
    const effectiveLlm = snapshot.bind(runLlm.effective, {
      hostKeyProvider,
      hostBaseUrl: host.baseUrl,
      hostSettings: host,
    });
    // Each call-time agent's provider when run start already knows it: it limits the recorded
    // selection.
    const known = knownCallTimeProviders(privateFlow, runLlm.callTime, snapshot.defaults);
    let callTimeLlm: CallTimeLlm | null = null;
    if (Object.keys(runLlm.callTime).length > 0) {
      callTimeLlm = {
        selection: callTimeSelection(snapshot, { known, hostKeyProvider, hostBaseUrl: host.baseUrl, hostSettings: host }),
        agents: runLlm.callTime,
      };
    }
    applyEffectiveLlmMap(privateFlow, effectiveLlm, callTimeLlm);
    // A key that neither marks an agent nor is in the call-time agents' selection has nothing to
    // authenticate, so the run keeps neither it nor its headers.
    if (apiKey !== null && !usesHostKey(effectiveLlm, callTimeLlm)) {
      apiKey = null;
      headers = null;
    }
    if (apiKey !== null) bindHostCredentials(privateFlow, apiKey, headers);
    checkEffectiveLlm(effectiveLlm, apiKey, headers, staticAgentLabels(privateFlow), valueConfigSubjects(privateFlow));

    // Last before the log: a memory read now is each memory's recall for the whole run.
    memories = await RunMemories.open(memoryBindings(flow.input, inputs), {
      create: true,
      signal: signal ?? new AbortController().signal,
      hostTypes,
    });
    for (const [name, value] of memories.values) setField(coerced as ValueRecord, name, value);

    const canonicalRunId = runId ?? defaultRunId();
    const pool = new VariablePool(privateFlow.startId);
    pool.set(privateFlow.startId, coerced);
    seedSystemClock(pool, scheduler.now());
    pool.addSystem("run_id", canonicalRunId);
    const startWiringOverride = startWiring(flow.input, coerced);

    // The result reads its files through the contract a CALL of this flow gets.
    const state = new RunState(privateFlow, { resultContract: childSignature(flow).output });
    const preparedStart = prepareRunStarted(state, {
      runId: canonicalRunId,
      flowFingerprint: authoredFingerprint,
      startWiringOverride,
      effectiveLlmByStaticPath: effectiveLlm,
      callTimeLlm,
      numWorkers,
      defaults: nonDefaults(runDefaults),
      flow: flow.source,
      skills: recordedSkills(flow.compiled, skills),
      tools: recordedTools(flow.compiled, runTools),
      toolGroups: recordedToolGroups(flow.compiled, runTools),
      yianVersion: VERSION,
      parent: options.parent == null ? null : { runId: options.parent.runId, node: options.parent.node ?? null },
    });

    // Taken once, so a later directory change can't move the log.
    const locator = log === null ? null : resolve(log);
    // The log stamps its records from the run's clock, as the system clock is seeded from it.
    const clock = () => scheduler.now();
    // Preparation precedes durable-file creation, so invalid initialization cannot leave an empty
    // path behind.
    authority =
      locator === null
        ? new MemoryRunLog(canonicalRunId, { clock })
        : JsonlRunLog.create(locator, canonicalRunId, { clock });
    if (locator !== null) {
      // The file's real path, as `advance` resolves it: `create` made a new regular file, so only
      // its folder can be a link.
      const key = join(realpathSync(dirname(locator)), basename(locator));
      // `create` was exclusive, so an entry under this path is stale.
      runHold = hold(key, null);
      const opened = authority as JsonlRunLog;
      const hostFlow = flow.source === null ? flow : null;
      keep = () => idleRun(state, flow, key, opened, { hostFlow, ...credentials });
    }
    // Every append goes through the live channel's wrapper; the authority itself is kept and closed.
    const appender = live === null ? authority : live.wrap(authority);
    const observers = commitPrepared(appender, state, preparedStart);
    // The pool's commits are validated as they are written, after `run_started` is on disk: input
    // intake has already rejected bad values, before any file exists, so a failure here is not an
    // input error.
    commitInitialPool(appender, state, pool);
    const events = await driveSegment(state, appender, observers, {
      options,
      numWorkers,
      tools: runTools,
      memories,
      skills,
      models,
      scheduler,
      runDir: locator === null ? null : dirname(locator),
      hold: runHold,
    });
    return resultFromState(state, log, events);
  } catch (error) {
    primary = { error };
    throw error;
  } finally {
    await closeAndRelease({ servers: runServers, memories, authority, primary, hold: runHold, keep });
  }
}

/**
 * Return, by record name, each tool an AGENT of `flow` grants and each tool of a group one reaches,
 * as the model is told about it, with its own name. A tool reached twice is recorded once.
 *
 * @throws RangeError when a grant can't be resolved (`checkToolGrants` reports it first).
 */
function recordedTools(flow: CompiledFlow, tools: ToolSet): Record<string, RecordedTool> {
  const recorded = new Map<string, RecordedTool>();
  const record = (ref: ToolRef) => {
    const key = tools.recordName(ref);
    if (recorded.has(key)) return;
    const tool = tools.resolve([ref])[0]!;
    const { name, description } = tool;
    // A tool built by an older copy of yian has no `resumable` field, and isn't resumable.
    recorded.set(key, { name, description, parameters: tool.parametersJsonSchema, resumable: tool.resumable === true });
  };
  for (const [, node] of staticNodes(flow)) {
    const grants = node.grantedTools();
    for (const granted of expandGrants(grants, tools.scopes())) {
      record({ group: granted.group, name: granted.tool });
    }
    // `tool_groups` lists every tool of a group, so each one is recorded too.
    for (const { group } of reachedGroups(grants, tools.scopes())) {
      for (const name of tools.scope().group(group)!) record({ group, name });
    }
  }
  // `Object.fromEntries` defines own data properties, so a `__proto__` name stays a key.
  return Object.fromEntries(recorded);
}

/**
 * Return each tool group an AGENT of `flow` reaches, by name: its tools' record names in group
 * order. A reopened run expands its grants with these, since it has no `ToolSet` of its own.
 */
function recordedToolGroups(flow: CompiledFlow, tools: ToolSet): Record<string, readonly string[]> {
  const recorded = new Map<string, readonly string[]>();
  for (const [, node] of staticNodes(flow)) {
    for (const { group } of reachedGroups(node.grantedTools(), tools.scopes())) {
      const members = tools.scope().group(group)!;
      recorded.set(group, members.map((name) => tools.recordName({ group, name })));
    }
  }
  return Object.fromEntries(recorded);
}

/** Return each skill an AGENT of `flow` grants from `skills`, by name. */
function recordedSkills(flow: CompiledFlow, skills: ReadonlyMap<string, Skill> | null): Record<string, RecordedSkill> {
  const recorded = new Map<string, RecordedSkill>();
  for (const [, grants] of skillGrants(flow)) {
    for (const name of grants === "all" ? [...(skills?.keys() ?? [])] : grants) {
      const skill = skills?.get(name);
      if (skill !== undefined) recorded.set(name, { description: skill.description, instructions: skill.instructions });
    }
  }
  return Object.fromEntries(recorded);
}

/**
 * @internal A check of the recorded model configs before a resume appends anything (the CLI's
 * credential check). It gets a copy of the recorded effective map plus each started call-time
 * agent's recorded config keyed by its node id (old OpenAI records adapted to their legacy
 * connection), and as `callTimeHost` the host key the call-time agents' selection names. A throw,
 * or a rejection of the promise it returns, stops the resume and appends nothing.
 */
export type LlmConfigsCheck = (
  configs: Record<string, AgentConfig>,
  options: { callTimeHost: Record<string, unknown> | null },
) => unknown;

/** The options of one `advance` call. */
export interface AdvanceOptions {
  /** The existing JSONL run log to lock, repair to its last complete line, check and replay. */
  log: string;
  /**
   * The flow of a run whose flow was built in code (`RunStarted.flow` is `null`), whose
   * fingerprint must match the recorded one. A run with recorded flow files compiles them and
   * refuses this.
   */
  flow?: ComposedFlow | null;
  /**
   * The code root the run recorded: the `codeRoot` its flow was loaded under, which the recorded
   * flow files' CODE references must still resolve inside (a reference outside it rejects with
   * `LoadError` before anything is appended). Given, it must name the same folder once symlinks
   * are resolved, or the call is refused; `null` (the default) uses the recorded one. Not used for
   * a flow built in code: the host confines it when it loads it.
   */
  codeRoot?: string | null;
  /**
   * The non-negative worker bound of the new segment; `null` (the default) keeps the last recorded
   * one.
   */
  numWorkers?: number | null;
  /**
   * Sees each observer committed by this call only, plus the transient in-loop AGENT events; the
   * replayed history and a finished log publish nothing. Called as `runFlow`'s is.
   */
  onEvent?: EventHook | null;
  /** Sees each new transient stream chunk; chunks are never replayed. */
  onStream?: StreamHook | null;
  /**
   * The live channel of this segment, as `runFlow`'s: it gets the records this call appends and
   * the model calls it makes, never the replayed ones.
   */
  live?: LiveChannel | null;
  /**
   * The tools the resumed segment sees, with `runFlow`'s rules. Only the specs of the tools AGENTs
   * grant are recorded, so pass the tools the run used (`ask_user` included); for an unfinished
   * run, each granted tool's description and parameters must be the recorded ones. An unopened
   * `McpServer` is opened only when the run is unfinished, after the log is replayed, and closed
   * when the resume ends; a finished log opens none.
   */
  tools?: Readonly<Record<string, ToolEntry>> | null;
  /**
   * The skills the run was given; an unfinished run of a flow that grants skills needs them, each
   * granted one with its recorded description and instructions.
   */
  skills?: ReadonlyMap<string, Skill> | null;
  /**
   * The provider key of a run started with `llmConfig.api_key`, for the agents the log marks for it
   * (and a call-time agent resolving to its provider); never recorded. A log that marks no agent
   * ignores it.
   */
  apiKey?: string | null;
  /**
   * The header values a run started with `llmConfig.headers` sent with its key; their names must
   * be the recorded ones, compared without case. Never recorded; a log that marks no agent ignores
   * them.
   */
  headers?: Readonly<Record<string, string>> | null;
  /** The host's app object a tool receives as `ctx.context`; the log doesn't hold it. */
  context?: unknown;
  /**
   * The instance of each memory input the run started with, by input name: the same memories,
   * bound again. An unfinished run opens each with `create: false` and needs every one it
   * recorded; an Optional memory left out takes none, or `null`, as `runFlow` takes it.
   */
  memory?: Readonly<Record<string, MemoryBase | null>> | null;
  /**
   * The host's stop: aborting it stops the resumed segment, which rejects with its reason. An
   * already-aborted signal rejects before the log is locked.
   */
  signal?: AbortSignal | null;
  /** The environment, as `runFlow`'s `env`; the run's recorded `YIAN_*` values must match it. */
  env?: Env | null;
  /** For tests: as `runFlow`'s `scheduler`. */
  scheduler?: Scheduler | null;
  /** For tests: as `runFlow`'s `random`. */
  random?: (() => number) | null;
  /** For tests: as `runFlow`'s `models`. */
  models?: ModelAccess | null;
  /** @internal See `LlmConfigsCheck`; not called for a finished log. */
  checkLlmConfigs?: LlmConfigsCheck | null;
}

/**
 * Move a durable run on: recover a crashed or stopped one, wake a waiting one, or read a finished
 * one.
 *
 * The run's flow is the one `RunStarted` recorded: its flow files are compiled again, never read
 * from disk (`loadRecordedFlow`), and a flow built in code is the `flow` option. Its fingerprint
 * must match the recorded one. The recorded history is replayed silently on a private copy of the
 * flow; recovery never reads the ambient model defaults, mints a new identity or takes new inputs
 * or configuration. A finished log returns its result with no events, opening no server. Otherwise
 * a log another yian version started is resumed with a warning naming both versions, and the host's
 * specs of the tools and skills the AGENTs grant must be the recorded ones (the skills checked
 * before any server starts, the tools once the servers are open).
 *
 * An unfinished run binds the memories it started with again (`memory`): each is opened with
 * `create: false` and must match the identity, type, settings and tool specs its run recorded. The
 * recorded recall stays the run's. A crash resume reads each memory's version and reports, in
 * `RunResult.memoryDrift`, one that changed since the run started (including the run's own
 * writes); advancing a waiting run doesn't read them. The sessions close
 * when the segment ends, a failed close reported as `runFlow` reports it.
 *
 * `RunResumed` is written only when the last segment crashed or stopped (work in flight, a
 * recorded stop, or no `run_waiting` as its last frontier record) or `numWorkers` changes; then the
 * starts the log left unfinished are retried first, in their recorded order, after one
 * `RunResumed` observer and one `NodeAttemptResumed` per retried node. A waiting run with a node
 * that is wakeable (its calls settled by a delivery) or due (a deadline past by the scheduler's
 * clock) goes straight to `node_woken`. A waiting run with neither appends nothing and returns
 * `waiting`.
 *
 * When the run ends waiting, the log's lock is released and its state stays in an in-process
 * registry keyed by the log's real path, so a later `advance` (or delivery) in this process reuses
 * it while the log is the file it left, with the same code-built flow and credentials.
 * An `advance` of a run this process's engine is moving (its loop running) rejects with
 * `RunBusyError`; one of a run another call is setting up, or closing after its loop ended, waits
 * for it.
 *
 * @param options See `AdvanceOptions`.
 * @returns The recorded terminal result, or the terminal or waiting result of the new segment with
 *   its post-durable observers; a waiting run that couldn't move returns `waiting` with none.
 * @throws TypeError when `flow` isn't a `ComposedFlow` or `null`, `signal` isn't an `AbortSignal`,
 *   `live` isn't a `LiveChannel`, `onEvent` or `onStream` isn't a function, a `tools` entry is
 *   malformed, `skills`, `scheduler`, `random` or `models` is malformed, a worker count isn't an
 *   integer, a recorded carrier has the wrong type, or `memory` isn't a record of `MemoryBase`
 *   instances (or `null`, for an Optional memory).
 * @throws ExpressionError as `runFlow`, for an unfinished run.
 * @throws RangeError when the log's prefix is invalid, the run's flow was built in code and no
 *   `flow` was passed, or its flow files were recorded and one was, `codeRoot` isn't the recorded
 *   code root, the flow fingerprint differs, the worker count is negative, `apiKey` or `headers`
 *   is refused or their header names aren't the recorded ones (the message never echoes a value),
 *   a recorded agent config has an invalid field, or, for an unfinished run, a tool or skill
 *   check of `runFlow` fails, a granted tool's spec or skill's text differs from the recorded one
 *   (`resume refused: tools.<name> description differs from the recorded one`), a `tools` server
 *   was opened by another run in progress, or a started call-time agent resolves to another config
 *   than the one it recorded (`node '<path>' llm_config: recorded config differs from the
 *   recomputed one; ...`), before its model call. Also when `learnable`, a removed option, is
 *   given; when `memory` names an input that isn't a memory input, or gives an instance to one
 *   recorded null (checked for any run, finished or not); or, for an unfinished run, a memory it
 *   recorded isn't bound or isn't at its location, or differs from the recorded one.
 * @throws AuthError when a call-time agent resolves to the provider of the run's host key and
 *   `apiKey` (or a recorded header) wasn't given, before its model call, or the credential store
 *   can't be read.
 * @throws McpConnectError when a `tools` server doesn't start or doesn't list its tools.
 * @throws DefaultsError when a `YIAN_<NAME>` variable in `env` holds a value its entry doesn't accept.
 * @throws DefaultsMismatchError when the run recorded `YIAN_*` values that differ from `env`'s.
 * @throws LoadError when the recorded flow files no longer load (a CODE reference's file gone, or
 *   resolving outside the recorded code root), before anything is appended.
 * @throws The signal's reason when it aborts.
 * @throws RunBusyError when another writer holds the log: another process or outside holder (the
 *   busy `FileLockError` is its cause), or a run this process's engine is moving (no cause).
 * @throws Error (code `ENOENT`) when `log` doesn't exist (from resolving its real path), and the
 *   lock, repair, append, sync or close error of the log.
 */
export async function advance(options: AdvanceOptions): Promise<RunResult> {
  // As in `runFlow`, the scope covers everything the resume reads; a bad YIAN_<NAME> value fails
  // here, before the log is opened.
  const env = options.env ?? processEnv;
  return withEnv(env, async () => {
    current();
    const { log, tools = null, context, signal = null, flow: hostFlow = null } = options;
    if (hostFlow !== null && !(hostFlow instanceof ComposedFlow)) {
      throw new TypeError("advance's flow must be a ComposedFlow or null");
    }
    const unopened = checkDriveOptions(options, true);
    let { apiKey, headers } = hostCredentials(options.apiKey, options.headers);
    // The absolute path is taken once, so a later directory change can't move the log.
    const locator = resolve(log);
    // The registry's key: the file itself, however the path names it (throws ENOENT when missing).
    const key = realpathSync(locator);
    const scheduler = options.scheduler ?? realScheduler;
    const clock = () => scheduler.now();
    // Resolved before the hold is awaited, like the log.
    const root = resolvedRoot(options.codeRoot);
    const runHold = await holdForAdvance(key, locator);
    let authority: JsonlRunLog | null = null;
    let runServers: McpServer[] = [];
    let memories: RunMemories | null = null;
    let primary: { error: unknown } | null = null;
    let keep: () => IdleRun | null = () => null;
    try {
      authority = openLog(locator, clock);
      const opened = authority;
      const start = startRecord(authority[Symbol.iterator]().next().value);
      // Values like AGENT_RETRIES change the loaded flow, so they are checked before it loads.
      checkRecorded(start.defaults);
      checkCodeRoot(log, start, root);
      const credentials = { apiKey, headers };
      const kept = reusable(runHold.idle, authority, key, { hostFlow, credentials });
      const flow = kept?.flow ?? (await resumedFlow(log, start, hostFlow));
      const state = kept?.state ?? replayRunLog(flow, authority, credentials);
      // The caller's credentials, which the replay used, before a flow that sends no host key
      // drops them below: a later call passing the same ones then reuses the entry.
      keep = () => idleRun(state, flow, key, opened, { hostFlow, ...credentials });
      // Checked before any early return, so a bad binding is never silently ignored.
      const recordedMemories = startMemories(flow.input, state);
      const memoryBound = resumeBindings(flow.input, options.memory ?? null, recordedMemories);
      const callTime = state.callTimeLlm as CallTimeLlm | null;
      if (!usesHostKey(state.effectiveLlmByStaticPath, callTime)) {
        apiKey = null;
        headers = null;
      } else if (apiKey !== null || headers !== null) {
        checkHostHeaderNames(state.effectiveLlmByStaticPath, headers, callTime);
      }
      // A finished run opens no tools.
      if (state.result !== null) return resultFromState(state, log, []);
      const models = options.models ?? defaultModelAccess(env);
      const numWorkers = options.numWorkers ?? state.numWorkers;
      // A crash or a stop left work unfinished, or the last segment ended before its run_waiting.
      const crashed =
        state.inFlight.size > 0 || state.lastStopSequence !== null || state.waits.lastFrontier !== "run_waiting";
      const resume = crashed || numWorkers !== state.numWorkers;
      if (
        !resume &&
        state.ready.length === 0 &&
        wakeableNodes(state.waits).length === 0 &&
        dueTimeouts(state.waits, scheduler.now()).size === 0
      ) {
        // Nothing can move: the run still waits, and nothing is appended.
        return resultFromState(state, log, []);
      }
      if (start.yianVersion !== VERSION) {
        warn(`run log ${log} was started by yian ${start.yianVersion}; resuming it with yian ${VERSION}`);
      }
      const skills = checkFlowBeforeServers(flow.compiled, options.skills ?? null);
      // A started call-time agent resumes with the config it recorded; one not yet started is
      // checked when it resolves.
      const inFlight = inFlightLlmConfigs(state.flow.nodes, state.nodeSteps);
      const recordedLlm = { ...state.effectiveLlmByStaticPath, ...inFlight };
      checkRecordedLlmFields(
        recordedLlm,
        staticAgentLabels(flow.compiled),
        staticPathKeys(Object.keys(inFlight)),
      );
      if (options.checkLlmConfigs != null) {
        await options.checkLlmConfigs(legacyConnectionMap(recordedLlm), { callTimeHost: selectionHost(callTime) });
      }
      // Skills need no server, so a changed one is refused before any server starts.
      checkRecordedSpecs("skills", start.skills, recordedSkills(flow.compiled, skills));
      runServers = await openRunServers(unopened, signal);
      const runTools = new ToolSet(tools, { context });
      const hostTypes = memoryTypeTools(flow.memoryTypes);
      checkToolGrants(flow.compiled, runTools);
      checkMemoryGrantNames(flow.compiled, runTools, hostTypes);
      checkToolTypes(flow.compiled, runTools);
      checkRecordedSpecs("tools", start.tools, recordedTools(flow.compiled, runTools));
      checkRecordedToolGroups(start.toolGroups, recordedToolGroups(flow.compiled, runTools));
      const reopened = await RunMemories.reopen(
        memoryBound,
        recordedMemories,
        { signal: signal ?? new AbortController().signal, hostTypes, drift: crashed },
      );
      memories = reopened.memories;

      const prepared = resume ? prepareRunResumed(state, { numWorkers }) : null;
      // This segment's appends go through the live channel's wrapper.
      const live = options.live ?? null;
      const appender = live === null ? authority : live.wrap(authority);
      const observers = prepared === null ? [] : commitPrepared(appender, state, prepared);
      const lead =
        prepared === null ? [] : [...observers, ...resumedAttemptEvents(state, prepared.record as RunResumedRecord)];
      const events = await driveSegment(state, appender, lead, {
        options,
        numWorkers,
        tools: runTools,
        memories,
        skills,
        models,
        scheduler,
        runDir: dirname(locator),
        hold: runHold,
      });
      return resultFromState(state, log, events, reopened.drift);
    } catch (error) {
      primary = { error };
      throw error;
    } finally {
      await closeAndRelease({ servers: runServers, memories, authority, primary, hold: runHold, keep });
    }
  });
}

/**
 * Return the `RunStarted` record of `first`, a log's first envelope.
 *
 * @throws RangeError when there is none, or it is another kind.
 */
function startRecord(first: RecordEnvelope | undefined): RunStartedRecord {
  if (first === undefined) throw new RangeError("cannot replay an empty run log");
  if (!(first.record instanceof RunStartedRecord)) throw new RangeError("the first run log record must be RunStarted");
  return first.record;
}

/**
 * Return the flow a resume of the log at `log` runs: the recorded one, or for a flow built in
 * code, the host's.
 *
 * @throws RangeError when the run's flow was built in code and `hostFlow` is `null`, or its files
 *   were recorded and `hostFlow` isn't `null`.
 * @throws LoadError when the recorded files don't load, a CODE reference resolving outside the
 *   recorded code root included.
 */
async function resumedFlow(log: string, start: RunStartedRecord, hostFlow: ComposedFlow | null): Promise<ComposedFlow> {
  if (start.flow === null) {
    if (hostFlow === null) {
      throw new RangeError(`run log ${log}: this run's flow was built in code; resume it with advance({log, flow})`);
    }
    return hostFlow;
  }
  if (hostFlow !== null) {
    throw new RangeError(`run log ${log} holds its flow files, which resume compiles; leave out flow`);
  }
  return loadFlowSource(start.flow);
}

/** Return `codeRoot` made absolute, or `null` when it is absent. */
function resolvedRoot(codeRoot: string | null | undefined): string | null {
  return codeRoot == null ? null : resolve(codeRoot);
}

/**
 * Refuse `root`, a call's absolute `codeRoot`, unless it names the code root the run recorded once
 * symlinks are resolved; `null` asks for the recorded one. A run built in code records no flow
 * files, so any root passes: its host confined the flow when it loaded it.
 *
 * @throws RangeError when the run recorded flow files under another code root, or under none.
 */
function checkCodeRoot(log: string, start: RunStartedRecord, root: string | null): void {
  if (root === null || start.flow === null) return;
  const recorded = start.flow.codeRoot;
  let real = root;
  try {
    real = realpathSync(root);
  } catch (error) {
    if (!isMissingPath(error)) throw error;
  }
  if (real === recorded) return;
  const was = recorded === null ? "the run recorded none" : `the run recorded ${reprText(recorded)}`;
  throw new RangeError(
    `run log ${log}: codeRoot ${reprText(real)} isn't the run's code root (${was}); leave out codeRoot`,
  );
}

/**
 * Compile the flow the run log at `log` recorded, as `advance` does: its flow files as the
 * loader first read them, never the files on disk now, under `env`'s `YIAN_*` values, which must
 * be the ones the run recorded. The log is read without its lock.
 *
 * @param log The JSONL run log.
 * @param options.env The environment the load reads the `YIAN_*` values from (default
 *   `processEnv`).
 * @param options.codeRoot As `advance`'s: the recorded code root, or `null` (the default) for it.
 * @param options.maxLineBytes The most bytes a line may hold, as `iterRunRecords` takes it; a longer
 *   line throws `RunLogLineTooLongError`.
 * @returns The recorded flow, or `null` when the run's flow was built in code.
 * @throws RangeError when the log holds no complete line or doesn't start with `RunStarted`, or
 *   `codeRoot` isn't the recorded code root, besides `iterRunRecords`' errors.
 * @throws LoadError when the files no longer load (a CODE reference's file gone, or outside the
 *   recorded code root).
 * @throws DefaultsError when a `YIAN_<NAME>` variable in `env` holds a value its entry doesn't accept.
 * @throws DefaultsMismatchError when the run recorded `YIAN_*` values that differ from `env`'s.
 */
export async function loadRecordedFlow(
  log: string,
  options: { env?: Env | null; codeRoot?: string | null; maxLineBytes?: number | null } = {},
): Promise<ComposedFlow | null> {
  const env = options.env ?? processEnv;
  const root = resolvedRoot(options.codeRoot);
  const start = readStart(log, options.maxLineBytes ?? null);
  return withEnv(env, () => {
    checkRecorded(start.defaults);
    checkCodeRoot(log, start, root);
    return start.flow === null ? null : loadFlowSource(start.flow);
  });
}

/**
 * Return the `RunStarted` record of the log at `log`, reading only its first line, without the
 * lock: the first line of an append-only log never changes. Lines are read with `maxLineBytes`.
 *
 * @throws RangeError when the log holds no complete line or doesn't start with `RunStarted`,
 *   besides `iterRunRecords`' errors.
 */
function readStart(log: string, maxLineBytes: number | null): RunStartedRecord {
  const records = iterRunRecords(log, { maxLineBytes });
  let first: RecordEnvelope | undefined;
  try {
    first = records.next().value?.envelope;
  } finally {
    records.return();
  }
  if (first === undefined) throw new RangeError(`cannot resume an empty run log: ${log}`);
  return startRecord(first);
}

/**
 * Check the host's specs of the granted tools or skills (`kind`) against the ones the run recorded.
 *
 * @throws RangeError naming the first one, by name, that is missing, added, or has a field that
 *   differs.
 */
function checkRecordedSpecs(
  kind: "tools" | "skills",
  recorded: Readonly<Record<string, Readonly<Record<string, unknown>>>>,
  current: Readonly<Record<string, Readonly<Record<string, unknown>>>>,
): void {
  const names = [...new Set([...Object.keys(recorded), ...Object.keys(current)])].sort(compareCodeUnits);
  for (const name of names) {
    const subject = `resume refused: ${kind}.${name}`;
    if (!Object.hasOwn(current, name)) throw new RangeError(`${subject} is recorded, but this run's ${kind} lack it`);
    if (!Object.hasOwn(recorded, name)) throw new RangeError(`${subject} isn't recorded; the run started without it`);
    const was = recorded[name]!;
    const now = current[name]!;
    for (const field of Object.keys(was)) {
      if (logJson(was[field]) !== logJson(now[field])) {
        throw new RangeError(`${subject} ${field} differs from the recorded one`);
      }
    }
  }
}

/**
 * Refuse a resume whose granted tool groups hold other tools, or other tools in another order,
 * than the run recorded: the request rebuild expands the run's grants with the recorded membership.
 *
 * @throws RangeError naming the first group, by name, that differs.
 */
function checkRecordedToolGroups(
  recorded: Readonly<Record<string, readonly string[]>>,
  current: Readonly<Record<string, readonly string[]>>,
): void {
  const names = [...new Set([...Object.keys(recorded), ...Object.keys(current)])].sort(compareCodeUnits);
  const members = (groups: Readonly<Record<string, readonly string[]>>, name: string) =>
    Object.hasOwn(groups, name) ? logJson([...groups[name]!]) : null;
  for (const name of names) {
    if (members(recorded, name) !== members(current, name)) {
      throw new RangeError(`resume refused: tool_groups.${name} differs from the recorded one`);
    }
  }
}

/**
 * The options of a host call on a run log. A live delivery ignores them, except `codeRoot`'s check
 * and the checks of `apiKey` and `headers`.
 */
export interface DeliveryOptions {
  /**
   * As `advance`'s `env`: the environment the call reads `YIAN_*` values from, which must be the
   * ones the run recorded (default `processEnv`).
   */
  env?: Env | null;
  /**
   * As `advance`'s: the recorded code root, refused when it names another folder, a live delivery
   * included; `null` (the default) uses the recorded one. Recompiling the recorded flow refuses a
   * CODE file outside it, before any append.
   */
  codeRoot?: string | null;
  /** The flow of a run built in code (`RunStarted` recorded no flow files), as `advance`'s. */
  flow?: ComposedFlow | null;
  /**
   * As `advance`'s, never recorded. A delivery calls no model, but the waiting state it leaves for
   * the next `advance` holds them: pass the ones that `advance` will, and it reuses that state
   * instead of replaying the log. With both `null` (the default), the call reuses this process's
   * state whatever credentials it holds, and keeps them.
   */
  apiKey?: string | null;
  /** As `advance`'s, never recorded; sent with `apiKey`, as above. */
  headers?: Readonly<Record<string, string>> | null;
}

/**
 * Deliver `op` to the run at `log`: through this process's engine when it is moving the run,
 * else under the log's lock, which is taken without blocking and released before this returns.
 * Offline, a waiting run's registry entry is reused while the log is the file it left; otherwise
 * the recorded flow is compiled and the log replayed, as `advance` does. A refused delivery keeps
 * the entry. `before` sees the state just before the delivery, synchronously.
 *
 * @returns The appended record and its seq (both `null` for a no-op), and its observers.
 */
async function deliverTo(
  log: string,
  op: DeliveryOp,
  options: DeliveryOptions,
  before: (state: RunState) => void = () => {},
): Promise<{ readonly seq: number | null; readonly events: readonly RunEvent[]; readonly record: RunRecord | null }> {
  const env = options.env ?? processEnv;
  return withEnv(env, async () => {
    current();
    const { flow: hostFlow = null, apiKey = null, headers = null } = options;
    if (hostFlow !== null && !(hostFlow instanceof ComposedFlow)) {
      throw new TypeError("a host call's flow must be a ComposedFlow or null");
    }
    const given = apiKey === null && headers === null ? null : hostCredentials(apiKey, headers);
    // The absolute path is taken once, so a later directory change can't move the log.
    const locator = resolve(log);
    // The registry's key: the file itself, however the path names it (throws ENOENT when missing).
    const key = realpathSync(locator);
    const root = resolvedRoot(options.codeRoot);
    // Checked before a live engine can take the delivery, since it reads no log.
    if (root !== null) checkCodeRoot(log, readStart(locator, null), root);
    const found = await liveOrHold(key, (engine) => {
      before(engine.state);
      return engine.deliver(op);
    });
    if ("delivered" in found) return found.delivered;
    const runHold = found.hold;
    let authority: JsonlRunLog | null = null;
    let primary: { error: unknown } | null = null;
    let keep: () => IdleRun | null = () => null;
    try {
      // A delivery carries no deadline, so its record time is the wall clock's.
      authority = openLog(locator, () => Date.now());
      const opened = authority;
      const start = startRecord(authority[Symbol.iterator]().next().value);
      checkRecorded(start.defaults);
      const want = given === null ? { hostFlow } : { hostFlow, credentials: given };
      const kept = reusable(runHold.idle, authority, key, want);
      const flow = kept?.flow ?? (await resumedFlow(log, start, hostFlow));
      const state = kept?.state ?? replayRunLog(flow, authority, given ?? {});
      // Set before preparing: a refused delivery changes nothing, so the state stays reusable.
      const credentials = given ?? { apiKey: kept?.apiKey ?? null, headers: kept?.headers ?? null };
      keep = () => idleRun(state, flow, key, opened, { hostFlow, ...credentials });
      before(state);
      const prepared = prepareDelivery(state, op);
      if (prepared === null) return { seq: null, events: [], record: null };
      const events = commitPrepared(authority, state, prepared);
      return { seq: prepared.expectedSequence, events, record: prepared.record };
    } catch (error) {
      primary = { error };
      throw error;
    } finally {
      await closeAndRelease({
        servers: [],
        memories: null,
        authority,
        primary,
        hold: runHold,
        keep,
        keepAfter: (failed) => failed.error instanceof DeliveryRefused,
      });
    }
  });
}

/**
 * Send `event` to the top-level inbox `inbox` of the run at `log`: a take reads it, now or when
 * it next takes from the inbox.
 *
 * Through this process's engine when it is moving the run (the options are ignored then, though a
 * `codeRoot` other than the recorded one, or a refused `apiKey` or `headers`, is still refused);
 * else under the log's lock, taken without blocking, reusing the state of a waiting run this process left while the log is
 * unchanged, or compiling the recorded flow and replaying the log as `advance` does. A send doesn't move the run: `advance` does (or the live engine).
 *
 * @param log The run's JSONL log.
 * @param inbox The name of one of the flow's `Inbox(T)` inputs.
 * @param event The value, checked against `T`.
 * @param options See `DeliveryOptions`.
 * @returns The send's seq.
 * @throws TypeError when `inbox` isn't a string, or `options.flow` isn't a `ComposedFlow` or `null`.
 * @throws DeliveryRefused (nothing appended) with reason `ended` once the run has a terminal
 *   result, `unknown_inbox`, `inbox_closed`, `invalid_value` (the `TypeCheckError` is the cause)
 *   or `too_large` (over the run's `YIAN_MAX_EVENT_BYTES`).
 * @throws RunBusyError when another writer holds the log.
 * @throws RangeError when the log is empty or doesn't start with `RunStarted`, the run's flow was
 *   built in code and `options.flow` is missing (or was recorded and it is given),
 *   `options.codeRoot` isn't the recorded code root, or `options.apiKey` or `options.headers` is
 *   refused (the message never echoes a value).
 * @throws LoadError when the recorded flow no longer loads (a CODE file gone, or outside the
 *   recorded code root), before anything is appended.
 * @throws DefaultsMismatchError when the run recorded `YIAN_*` values that differ from `env`'s.
 * @throws Error (code `ENOENT`) when `log` doesn't exist; the log's open, append and close errors.
 */
export async function send(log: string, inbox: string, event: unknown, options: DeliveryOptions = {}): Promise<number> {
  if (typeof inbox !== "string") throw new TypeError("inbox must be a string");
  return (await deliverTo(log, { kind: "send", inbox, event }, options)).seq!;
}

/**
 * Answer the pending host call `callId` of the run at `log` with `result`: its node wakes to it
 * when the run next moves. Delivered as `send` is.
 *
 * @param log The run's JSONL log.
 * @param callId The call's id, as `pendingCalls` lists it.
 * @param result `{output}`, checked against the tool's output type (an `ask`'s answer can't be
 *   null), or `{error}`, final for the call.
 * @param options See `DeliveryOptions`.
 * @returns The answer's seq, or `null` when the call is already answered with exactly this result.
 * @throws TypeError when `callId` isn't a string, `result` isn't `{output}` (output not
 *   `undefined`) or `{error: string}`,
 *   or `options.flow` isn't a `ComposedFlow` or `null`.
 * @throws DeliveryRefused (nothing appended) with reason `ended` once the run has a terminal
 *   result, `unknown_call` (no such call, its node settled, or its handler hasn't returned
 *   `pending()`), `not_host_call` for a take or sleep, `call_settled` (with `callState`) for an
 *   answered, timed-out or closed call, `invalid_value` (the `TypeCheckError` is the cause) or
 *   `too_large`.
 * @throws RunBusyError, RangeError, LoadError, DefaultsMismatchError and `ENOENT`, as `send`.
 */
export async function complete(
  log: string,
  callId: string,
  result: CompleteResult,
  options: DeliveryOptions = {},
): Promise<number | null> {
  if (typeof callId !== "string") throw new TypeError("callId must be a string");
  const fields: Record<string, unknown> | null = isRecord(result) ? result : null;
  const shaped =
    fields !== null &&
    Object.keys(fields).length === 1 &&
    ((Object.hasOwn(fields, "output") && fields.output !== undefined) ||
      (Object.hasOwn(fields, "error") && typeof fields.error === "string"));
  if (!shaped) throw new TypeError("a complete's result must be {output} or {error: string}");
  return (await deliverTo(log, { kind: "complete", callId, result }, options)).seq;
}

/**
 * Close the top-level inbox `inbox` of the run at `log`: later sends are refused, takes still read
 * what is queued, and a take waiting on it with nothing to read gets `{events: [], closed: true}`
 * when the run next moves. Delivered as `send` is.
 *
 * @param log The run's JSONL log.
 * @param inbox The name of one of the flow's `Inbox(T)` inputs.
 * @param options See `DeliveryOptions`.
 * @returns The close's seq, or `null` when the inbox is already closed.
 * @throws TypeError when `inbox` isn't a string, or `options.flow` isn't a `ComposedFlow` or `null`.
 * @throws DeliveryRefused (nothing appended) with reason `ended` once the run has a terminal
 *   result, or `unknown_inbox`.
 * @throws RunBusyError, RangeError, LoadError, DefaultsMismatchError and `ENOENT`, as `send`.
 */
export async function close(log: string, inbox: string, options: DeliveryOptions = {}): Promise<number | null> {
  if (typeof inbox !== "string") throw new TypeError("inbox must be a string");
  return (await deliverTo(log, { kind: "close", inbox }, options)).seq;
}

/** One send `withdraw` took back: its seq and value. */
export interface WithdrawnEvent {
  /** The send's seq. */
  readonly seq: number;
  /** The value it sent. */
  readonly value: unknown;
}

/**
 * Take back every unread send of the top-level inbox `inbox` of the run at `log`: those no take or
 * steer read has taken, including sends assigned to a take whose node hasn't woken, which then
 * waits again (or, on a closed inbox, gets `{events: [], closed: true}`). Withdrawn sends get no
 * reply, take output or late steer. Delivered as `send` is; a closed inbox is allowed. No event
 * is published.
 *
 * @param log The run's JSONL log.
 * @param inbox The name of one of the flow's `Inbox(T)` inputs.
 * @param options See `DeliveryOptions`.
 * @returns The sends taken back, in send order; `[]` when nothing was unread (nothing is appended).
 * @throws TypeError when `inbox` isn't a string, or `options.flow` isn't a `ComposedFlow` or `null`.
 * @throws DeliveryRefused (nothing appended) with reason `ended` once the run has a terminal
 *   result, `not_started`, or `unknown_inbox`.
 * @throws RunBusyError, RangeError, LoadError, DefaultsMismatchError and `ENOENT`, as `send`.
 */
export async function withdraw(log: string, inbox: string, options: DeliveryOptions = {}): Promise<readonly WithdrawnEvent[]> {
  if (typeof inbox !== "string") throw new TypeError("inbox must be a string");
  let taken: WithdrawnEvent[] = [];
  const { seq } = await deliverTo(log, { kind: "withdraw", inbox }, options, (state) => {
    // Read before the record drops the values, in the delivery's synchronous section.
    taken = withdrawableEvents(state.waits, inbox).map((event) => ({
      seq: event,
      value: copyValue(state.inboxEvents.get(event)!.toObject()),
    }));
  });
  return seq === null ? [] : taken;
}

/**
 * Cancel the run at `log`: `run_cancelled` ends it, closing its open calls, with no output. A live
 * engine aborts its running nodes and publishes `CallClosed {reason: "cancelled"}` for each open
 * host call. Delivered as `send` is.
 *
 * @param log The run's JSONL log.
 * @param options See `DeliveryOptions`.
 * @returns The open host calls the cancel closed, as `pendingCalls` listed them.
 * @throws TypeError when `options.flow` isn't a `ComposedFlow` or `null`.
 * @throws DeliveryRefused (nothing appended) with reason `ended` once the run has a terminal
 *   result (a second cancel among them).
 * @throws RunBusyError, RangeError, LoadError, DefaultsMismatchError and `ENOENT`, as `send`.
 */
export async function cancel(log: string, options: DeliveryOptions = {}): Promise<readonly PendingCall[]> {
  let open: readonly PendingCall[] = [];
  await deliverTo(log, { kind: "cancel" }, options, (state) => {
    open = openHostCalls(state.waits, (call) => stateArguments(state, call));
  });
  return open;
}

/** What `interrupt` did. */
export interface InterruptResult {
  /** The `interrupted` record's seq; `null` when nothing was appended. */
  readonly seq: number | null;
  /** The agents the record named. */
  readonly nodes: readonly string[];
  /** The open host calls it closed, as `pendingCalls` listed them. */
  readonly calls: readonly PendingCall[];
}

/**
 * Stop the turn on the top-level inbox `inbox`: interrupt the agents reading it as their steer
 * inbox, and remember the Stop for one that opens it soon. Each named agent's current reply is cut
 * and its result is null, which its declared output decides (`Optional(...)` succeeds with null;
 * anything else, no `output:` included, fails `agent.interrupted`). The interrupt closes their open
 * calls (`CallClosed {reason: "interrupted"}`), so a later `complete` of one is refused. A live
 * engine cuts a running reply at once; a waiting agent honors it when the run next moves. Queued
 * steers stay in the inbox: `withdraw` takes them back.
 *
 * It appends nothing (`seq: null`) when the run waits for the next message on `inbox` (a take of
 * it waits with nothing unread and no reader is open), or when it is already stopping (a Stop on
 * `inbox` stands, or every current reader already holds or has honored an interrupt). Otherwise
 * it records `interrupted {inbox, nodes}`, and `nodes` may be empty. The Stop is remembered until
 * the next pickup from `inbox` (a take of it waits, a waiting take is answered or times out, an
 * inline take commits, a withdraw reopens a take), the first `run_waiting` after it stopped an
 * agent, or run end: a reader that opens `inbox` meanwhile is stopped before its first model
 * call, and only its own `interrupted` record and `CallClosed` events report it. Delivered as
 * `send` is; offline only waiting agents are named, and a crashed run's agents stop when they
 * reopen.
 *
 * @param log The run's JSONL log.
 * @param inbox The name of one of the flow's `Inbox(T)` inputs.
 * @param options See `DeliveryOptions`.
 * @returns The record's seq, the agents it named and the open host calls it closed.
 * @throws TypeError when `inbox` isn't a string, or `options.flow` isn't a `ComposedFlow` or `null`.
 * @throws DeliveryRefused (nothing appended) with reason `ended` once the run has a terminal
 *   result, `not_started`, or `unknown_inbox`.
 * @throws RunBusyError, RangeError, LoadError, DefaultsMismatchError and `ENOENT`, as `send`.
 */
export async function interrupt(log: string, inbox: string, options: DeliveryOptions = {}): Promise<InterruptResult> {
  if (typeof inbox !== "string") throw new TypeError("inbox must be a string");
  let open: readonly PendingCall[] = [];
  const { seq, events, record } = await deliverTo(log, { kind: "interrupt", inbox }, options, (state) => {
    open = openHostCalls(state.waits, (call) => stateArguments(state, call));
  });
  const closed = new Set(events.flatMap((event) => (event instanceof CallClosed ? [event.callId] : [])));
  const nodes = record instanceof InterruptedRecord ? [...record.nodes] : [];
  return { seq, nodes, calls: open.filter((call) => closed.has(call.callId)) };
}

/** One open host call of a run, as `pendingCalls` lists it. */
export interface PendingCall {
  /** The id `complete` answers it by. */
  readonly callId: string;
  /** The id of the node that made it. */
  readonly node: string;
  /** The tool's id (`"ask"`, or a host tool's). */
  readonly tool: string;
  /**
   * The arguments its `pending` step recorded, or for an agent's call those of the reply's tool
   * call; `null` when none.
   */
  readonly arguments: JsonObject | null;
  /** The seq of its `pending` step. */
  readonly pendingSeq: number;
  /** When it times out, as a record time, or `null` when it doesn't. */
  readonly deadline: string | null;
}

/**
 * List the open host calls (not takes or sleeps) of the run at `log`, in the order they were made,
 * from one pass over its records. It takes no lock, so it reads a log another writer holds; a
 * call leaves the list once it is answered, timed out or closed. Hosts rebuild their open asks
 * from it.
 *
 * @param options.maxLineBytes The most bytes a line may hold, as `iterRunRecords` takes it; a longer
 *   line throws `RunLogLineTooLongError`.
 * @throws RangeError when the log is empty or doesn't start with `RunStarted`, besides
 *   `iterRunRecords`' errors.
 * @throws Error (code `ENOENT`) when `log` doesn't exist.
 */
export function pendingCalls(log: string, options: { maxLineBytes?: number | null } = {}): readonly PendingCall[] {
  const waits = new WaitsState();
  // Each unsettled call's recorded arguments; a settled node's calls leave the fold, so theirs go.
  const args = new Map<string, JsonObject | null>();
  // Each running node's step count and latest `model_call` step with its number: an agent call's
  // `reply` must name that step, so the earlier ones needn't be kept.
  const nodes = new Map<string, { count: number; replyN: number; reply: Readonly<JsonObject> | undefined }>();
  let first = true;
  for (const { envelope } of iterRunRecords(log, { maxLineBytes: options.maxLineBytes ?? null })) {
    if (first && !(envelope.record instanceof RunStartedRecord)) {
      throw new RangeError(`run log ${log} doesn't start with run_started`);
    }
    first = false;
    applyWaitsRecord(waits, envelope);
    const record = envelope.record;
    if (record instanceof NodeStepRecord) {
      const node = nodes.get(record.nodeId) ?? { count: 0, replyN: 0, reply: undefined };
      nodes.set(record.nodeId, node);
      node.count += 1;
      const step = record.step;
      if (step["kind"] === "model_call") {
        node.replyN = node.count;
        node.reply = step;
      } else if (step["kind"] === "pending") {
        const replyAt = (n: number) => (n === node.replyN ? node.reply : undefined);
        args.set(String(step["call_id"]), pendingArguments(replyAt, step));
      }
    } else {
      const settled = settledNodeId(record);
      if (settled === null) continue;
      nodes.delete(settled);
      for (const callId of args.keys()) if (!waits.calls.has(callId)) args.delete(callId);
    }
  }
  if (first) throw new RangeError(`cannot read an empty run log: ${log}`);
  return openHostCalls(waits, (call) => args.get(call.callId) ?? null);
}

/**
 * The node `record` settles (its success, route, expansion or failure), or `null` for any other
 * record. A failure record's node id is never `null`.
 */
function settledNodeId(record: RunRecord): string | null {
  if (record instanceof NodeSucceededRecord || record instanceof NodeRoutedRecord) return record.nodeId;
  if (record instanceof NodeExpandedRecord) return record.spawnerId;
  if (record instanceof NodeFailedRecord) return record.failure.nodeId;
  return null;
}

/** The open host calls of `waits`, in registration order, with the arguments `argumentsOf` reads. */
function openHostCalls(waits: WaitsState, argumentsOf: (call: Call) => JsonObject | null): PendingCall[] {
  return [...waits.calls.values()]
    .filter((call) => call.state === "open" && call.pendingSeq !== null && call.tool !== TAKE && call.tool !== SLEEP)
    .map((call) => ({
      callId: call.callId,
      node: call.nodeId,
      tool: call.tool,
      arguments: argumentsOf(call),
      pendingSeq: call.pendingSeq!,
      deadline: call.deadline,
    }));
}

/**
 * The arguments of `call`'s `pending` step among its node's steps in `state` (for an agent's call,
 * those of its reply's tool call), or `null`.
 */
function stateArguments(state: RunState, call: Call): JsonObject | null {
  const steps = state.nodeSteps.get(call.nodeId) ?? [];
  const step = call.pendingSeq === null ? undefined : steps[stepIndexAt(state, call.nodeId, call.pendingSeq)];
  return step === undefined ? null : pendingArguments((n) => steps[n - 1], step);
}

/**
 * The arguments `pending` step `step` registered (for an agent's call, those of the tool call in
 * the reply its `reply` names, which `stepAt` returns by step number), or `null`.
 */
function pendingArguments(
  stepAt: (n: number) => Readonly<JsonObject> | undefined,
  step: Readonly<JsonObject>,
): JsonObject | null {
  if (Object.hasOwn(step, "reply")) return agentCallArguments(stepAt(Number(step["reply"])), Number(step["index"]));
  return (step["arguments"] as JsonObject | undefined) ?? null;
}

/** The arguments of runnable tool call `index` of the reply a `model_call` step recorded, or `null`. */
function agentCallArguments(reply: Readonly<JsonObject> | undefined, index: number): JsonObject | null {
  if (reply === undefined || reply["kind"] !== "model_call") return null;
  // A call whose arguments didn't parse never ran, so it has no index.
  const calls = Message.fromJSON(reply["output"]).parts.filter(
    (part): part is ToolCallPart => part instanceof ToolCallPart && part.arguments !== null,
  );
  return (calls[index]?.arguments as JsonObject | undefined) ?? null;
}

/**
 * Create a fresh run's folder, `<runsDir>/<runId>`, and return its log path, `log.jsonl` inside it.
 *
 * A run's folder is the folder that holds its log, and the run's other files go only there, so
 * each run gets a folder of its own. `runsDir` is made when missing. The run folder must not
 * exist yet; it is made with mode 0700. The folder above each one made is synced, so the run
 * folder survives a power loss; when a sync fails, the run folder is removed if it can be. The log
 * file is not created: the run log creates it exclusively, owner-only.
 *
 * @throws {RangeError} When `runId` isn't one folder name: empty, `.`, `..`, or holding `/`, `\`
 *   or NUL.
 * @throws {Error} `EEXIST` when anything (a folder, a file, a symlink) already holds the run
 *   folder's path, or another error when a folder can't be made, changed or synced.
 */
export function runLogPath(runsDir: string, runId: string): string {
  if (runId === "" || runId === "." || runId === ".." || /[/\\\0]/.test(runId)) {
    throw new RangeError(`run id is not one folder name: ${reprText(runId)}`);
  }
  // Resolved, so the first folder `mkdirSync` reports making is `runsDir` or above it.
  const absoluteRunsDir = resolve(runsDir);
  const first = mkdirSync(absoluteRunsDir, { recursive: true, mode: 0o700 });
  const runDir = join(absoluteRunsDir, runId);
  mkdirSync(runDir, { mode: 0o700 });
  // The umask can narrow mkdir's mode.
  chmodSync(runDir, 0o700);
  try {
    // Each new folder's entry lives in its parent: sync the run folder's and, when `runsDir` was
    // made, each made folder's up to the first. The walk ends at the root whatever `first` holds.
    const stop = first === undefined ? absoluteRunsDir : dirname(resolve(first));
    for (let made = runDir; made !== stop && made !== dirname(made); made = dirname(made)) {
      syncFolder(dirname(made));
    }
  } catch (error) {
    try {
      rmdirSync(runDir);
    } catch {
      // The original error is the one to report.
    }
    throw error;
  }
  return join(runsDir, runId, "log.jsonl");
}

/** Where a run log stands, as `inspectRun` reads it. */
export type RunStatus = "running" | "succeeded" | "failed" | "cancelled" | "waiting" | "stopped" | "crashed";

/** What `inspectRun` reads from one run log. */
export interface RunInspection {
  /**
   * `running` while a writer holds the log's lock; `succeeded`, `failed` or `cancelled` once the
   * run ended; `waiting` when its last segment ended with every unsettled node waiting (on a take,
   * a sleep or a host call); `stopped` when the host stopped the last segment; otherwise `crashed`
   * (the process ended without a stop). A run that isn't running or ended can be advanced.
   */
  readonly status: RunStatus;
  /**
   * The nodes started (or woken) and neither settled nor waiting, in start order; `[]` for an
   * ended run.
   */
  readonly inFlight: readonly string[];
  /**
   * By node id, the starts a resume retried with no stop covering them, plus, for a run that is
   * neither running nor finished, one for each node in flight that no stop covers: the attempts a
   * resume would retry after a crash.
   */
  readonly crashedAttempts: ReadonlyMap<string, number>;
  /** The run's id; `null` only for a running log with no record yet, as are the next five. */
  readonly runId: string | null;
  /** The run-record schema the log is written in, `SCHEMA_VERSION`. */
  readonly schema: number | null;
  /** The fingerprint of the flow the run started with (`RunStarted`). */
  readonly flowFingerprint: string | null;
  /** The seq of the log's last complete record. */
  readonly lastSeq: number | null;
  /**
   * When the run started: the time of its `RunStarted` record, a UTC RFC 3339 time with
   * milliseconds such as `2026-10-20T14:02:11.123Z`.
   */
  readonly startedAt: string | null;
  /**
   * When a `succeeded`, `failed` or `cancelled` run ended: the time of the log's last record; `null`
   * for any other status. Times come from the writer's clock, so they aren't guaranteed to increase.
   */
  readonly endedAt: string | null;
  /**
   * The failure that ended a `failed` run: the node's failure, or the `graph.terminal_unreachable`
   * failure of a route that proved END unreachable (or of the `RunFailed` that ended a run whose
   * nodes waited); `null` for any other status.
   */
  readonly failure: Failure | null;
  /**
   * For a `waiting` run, the calls of each waiting node, oldest wait first; `[]` for any other
   * status.
   */
  readonly waits: readonly WaitInfo[];
  /**
   * Whether a `waiting` run has a waiting node it can wake now (every call it waits on is
   * settled); `false` for any other status. A host also advances a waiting run once `wakeAt` is
   * due, though `ready` is still `false`.
   */
  readonly ready: boolean;
  /**
   * For a `waiting` run, the earliest deadline of an open call of a waiting node (a record time),
   * when `advance` times it out; `null` when none has a deadline, and for any other status.
   */
  readonly wakeAt: string | null;
}

/**
 * Read where the run in `log` stands without changing the file or taking its lock.
 *
 * In-flight nodes, crashed counts and the waits come from the records. Without `flow`, the status
 * is read from the records too: `cancelled` after a cancel; `failed` after a node failure or a
 * `RunFailed`; with nothing in flight and nothing waiting, `succeeded` after the top-level END
 * succeeded, else `failed` after a route proved END unreachable; then `waiting` when the last
 * segment ended with every node waiting (`RunWaiting`). The records don't hold the ready nodes, so a run whose process ended after END succeeded (or after
 * that proof) but before a node already ready started reads as finished, though `advance` would
 * still run that node. With `flow`, the log is replayed as `advance` replays it, so the status
 * is exact; pass the flow when you have it. The lock is probed before and after the records are
 * read, so a writer that holds it at either probe makes the run `running`. Without `flow`, the
 * records are streamed (`iterRunRecords`), so memory is bounded by the largest record. When a
 * writer holding the lock rolls back a failed append under the read, the log is read once more.
 *
 * @param log The JSONL run log's path.
 * @param options.flow The loaded flow the run was started with, for an exact status.
 * @param options.env The environment whose `YIAN_*` values the replay with `flow` checks the
 *   recorded ones against, as `advance`'s `env`; without one, the values already in effect.
 *   Reading without `flow` doesn't use it.
 * @param options.maxLineBytes The most bytes a line may hold, as `iterRunRecords` takes it; a longer
 *   line throws `RunLogLineTooLongError`.
 * @returns The status, the nodes in flight, the crashed counts, the run's id, schema, flow
 *   fingerprint, last seq, start time and (once ended) end time, a failed run's
 *   failure, and a waiting run's waits, readiness and wake time.
 * @throws Error (code `ENOENT`) when `log` doesn't exist, and the `fs` error of a failed read.
 * @throws RunLogSchemaError when a record's schema isn't one this version reads.
 * @throws RunLogChangedError when the log shrank while it was read and no writer holds it.
 * @throws RangeError when a complete line isn't a valid record, the first record isn't
 *   `RunStarted`, the log holds no record and no writer, or, with `flow`, the replay refuses the
 *   log (`replayRunLog`).
 * @throws DefaultsError when a `YIAN_<NAME>` variable in `env` holds a value its entry doesn't accept.
 * @throws DefaultsMismatchError with `flow`, when the run recorded `YIAN_*` values that differ
 *   from `env`'s.
 */
export function inspectRun(
  log: string,
  options: { flow?: ComposedFlow | null; env?: Env | null; maxLineBytes?: number | null } = {},
): RunInspection {
  const lockedBefore = isRunLogLocked(log);
  const flow = options.flow ?? null;
  const maxLineBytes = options.maxLineBytes ?? null;
  // The replay needs every record at once; reading without it streams them.
  const read = () => {
    const envelopes = flow === null ? null : readRunLog(log, { maxLineBytes });
    return { envelopes, scan: scanRunRecords(envelopes ?? streamedEnvelopes(log, maxLineBytes)) };
  };
  let records: ReturnType<typeof read>;
  try {
    records = read();
  } catch (error) {
    if (!(error instanceof RunLogChangedError && (lockedBefore || isRunLogLocked(log)))) throw error;
    // The writer rolled back a failed append under the read; that poisons it, so it appends and
    // cuts nothing more, and a second read sees its final prefix.
    records = read();
  }
  const { envelopes, scan } = records;
  // A writer that took or released the lock while the records were read is caught by one probe.
  const running = isRunLogLocked(log) || lockedBefore;
  if (scan.started === null && !running) throw new RangeError(`cannot inspect an empty run log: ${log}`);
  // A running log's status needs no replay.
  const source =
    running || envelopes === null
      ? scan
      : withEnv(options.env ?? processEnv, () => replayedScan(flow!, envelopes));
  const { starts, crashed, lastStop, status: recorded, waits } = source;
  const status: RunStatus = running ? "running" : recorded;
  // A running run's attempts are still running, and an ended run's are abandoned.
  if (status === "crashed" || status === "stopped") addCrashedAttempts(crashed, starts, lastStop);
  const ended = status === "succeeded" || status === "failed" || status === "cancelled";
  const inFlight = ended ? [] : [...starts.keys()];
  const waiting = status === "waiting";
  const { started } = scan;
  return {
    status,
    inFlight,
    crashedAttempts: crashed,
    runId: started?.runId ?? null,
    schema: started?.schema ?? null,
    flowFingerprint: started === null ? null : (started.record as RunStartedRecord).flowFingerprint,
    lastSeq: scan.lastSeq,
    startedAt: started?.at ?? null,
    endedAt: ended ? scan.lastAt : null,
    failure: status === "failed" ? source.failure : null,
    waits: waiting ? waitInfos(waits) : [],
    ready: waiting && wakeableNodes(waits).length > 0,
    wakeAt: waiting ? wakeAt(waits) : null,
  };
}

/** A run log replayed through the run reducer (see `foldRunLog`). */
export interface FoldedRun {
  /** The flow the replay ran: the recorded one, or the host's for a flow built in code. */
  readonly flow: ComposedFlow;
  /** Every record, in log order, with its accepted value. */
  readonly records: readonly FoldedRecord[];
}

/** One record of a folded run log. */
export interface FoldedRecord {
  /** The record as `iterRunRecords` reads it. */
  readonly entry: RunLogEntry;
  /**
   * For a `pool_commit` the run accepted, the value it writes: the one it holds (a send's too), for
   * a derived commit the one rebuilt from its `from`, for a take commit `{events, closed}` built
   * from the sends it consumed, or for a messages commit the `list(Message)` rebuilt from its
   * steps. `null` for an orphan commit and any other record.
   */
  readonly value: TypedValue | null;
}

/**
 * Replay the run log at `log` through the run reducer, as resume does, and return each record with
 * the value of each accepted `pool_commit`: hosts read derived values this way, such as a LOOP
 * iteration's `<loop>#k/__end__`, which the log records only as `from` pointers, and a take's
 * events, which its commit records as the seqs of the sends it consumed. The recorded flow
 * is compiled first, under `env`'s `YIAN_*` values, which must be the run's. No seq-to-value map
 * is kept beside the reducer: a derived value is rebuilt from the pool when its commit is accepted.
 *
 * @param options.env The environment the load reads the `YIAN_*` values from (default
 *   `processEnv`).
 * @param options.flow The flow, required when the run's flow was built in code (`flow: null`),
 *   refused otherwise.
 * @param options.codeRoot As `advance`'s: the recorded code root, or `null` (the default) for it. A
 *   recorded CODE reference whose file's real path is outside it rejects with `LoadError` before
 *   the replay; not used for a flow built in code.
 * @param options.maxLineBytes The most bytes a line may hold, as `iterRunRecords` takes it; a longer
 *   line throws `RunLogLineTooLongError`.
 * @throws RangeError when the log is empty, ends inside its run start, or the replay refuses it;
 *   when the run's flow was built in code and `options.flow` is missing, or the log holds its
 *   flow files and `options.flow` is given, or `codeRoot` isn't the recorded code root.
 * @throws LoadError when the recorded files no longer load (a CODE reference's file gone, or
 *   outside the recorded code root).
 * @throws DefaultsError, DefaultsMismatchError as `loadRecordedFlow`, and `iterRunRecords`' errors.
 */
export async function foldRunLog(
  log: string,
  options: { env?: Env | null; flow?: ComposedFlow | null; codeRoot?: string | null; maxLineBytes?: number | null } = {},
): Promise<FoldedRun> {
  const env = options.env ?? processEnv;
  const root = resolvedRoot(options.codeRoot);
  const entries = [...iterRunRecords(log, { maxLineBytes: options.maxLineBytes ?? null })];
  const start = startRecord(entries[0]?.envelope);
  const flow = await withEnv(env, () => {
    checkRecorded(start.defaults);
    checkCodeRoot(log, start, root);
    return resumedFlow(log, start, options.flow ?? null);
  });
  const values = new Map<number, TypedValue>();
  withEnv(env, () =>
    replayRunLog(flow, MemoryRunLog.fromRecords(entries.map((entry) => entry.envelope)), {
      onPrepared: (prepared) => {
        for (const { seq, value } of acceptedCommits(prepared)) values.set(seq, value);
      },
    }),
  );
  return { flow, records: entries.map((entry, seq) => ({ entry, value: values.get(seq) ?? null })) };
}

/** One AGENT model call of a run log, as `agentCalls` lists it. */
export interface AgentCallEntry {
  /** The node that made the call, as the log names it (a grown node's own id). */
  readonly nodeId: string;
  /** The `n` of the call's last attempt's `model_call` step, which its tool call ids use. */
  readonly n: number;
  /** `chat`, `structured_native` or `structured_fallback`. */
  readonly purpose: string;
  /** The model the call reached, as its step records it. */
  readonly model: string;
  /** When its first attempt started (a record time). */
  readonly startedAt: string;
  /** How many attempts it made: 1, plus one per retry. */
  readonly attempts: number;
  /** The last attempt's finish reason (`interrupted` for a reply a host interrupt cut), `null` when it failed. */
  readonly finish: string | null;
}

/** The token counts a `model_call` step records, each `null` when the provider didn't report it. */
export interface AgentCallUsage {
  readonly inputTokens: number | null;
  readonly outputTokens: number | null;
  readonly cacheReadTokens: number | null;
  /** `null` too for a step an older version wrote, which didn't record it. */
  readonly cacheCreationTokens: number | null;
  readonly reasoningTokens: number | null;
}

/** One attempt of an AGENT model call, as its `model_call` step records it. */
export interface AgentCallAttempt {
  readonly n: number;
  readonly startedAt: string;
  /** When its first event arrived; `null` when none did. */
  readonly firstTokenAt: string | null;
  /** When its step was appended (the record's `at`), right after it returned or failed. */
  readonly finishedAt: string;
  /** A failed attempt's `{kind, message}`, the message scrubbed of credentials; `null` for a reply. */
  readonly error: { readonly kind: string; readonly message: string } | null;
  /** Seconds a failed attempt waited before its retry; `null` when it wasn't retried, or replied. */
  readonly retryIn: number | null;
  readonly usage: AgentCallUsage | null;
}

/** The request an AGENT model call sent, rebuilt (see `agentCall`). */
export interface AgentCallRequest {
  /** The model the call reached, as its step records it. */
  readonly model: string;
  /** The model config the call used, without the host's key, headers or connection. */
  readonly params: JsonObject;
  /** The system message's text; `null` without one. */
  readonly system: string | null;
  /** The other messages, in order, as `Message` values. */
  readonly messages: readonly MessageValue[];
  /** The tools the model could call. */
  readonly tools: readonly {
    readonly name: string;
    readonly description: string;
    readonly parameters: JsonObject;
    readonly strict: boolean | null;
  }[];
  /** `auto`, `required`, `none`, `{name}` for a named tool, or `null` for the provider's default. */
  readonly toolChoice: "auto" | "required" | "none" | { readonly name: string } | null;
  readonly parallelToolCalls: boolean | null;
  /** The schema a structured reply's text must match, or `null`. */
  readonly outputFormat: { readonly name: string; readonly schema: JsonObject; readonly strict: boolean | null } | null;
}

/** One tool call of an AGENT reply and its recorded result. */
export interface AgentToolResult {
  /** The id of the reply's `tool_call` part. */
  readonly callId: string;
  readonly name: string;
  /** The parsed arguments; `null` when they didn't parse (the call never ran). */
  readonly arguments: JsonObject | null;
  /** The text the model read, or `null` while no result is recorded (pending, cut off, or the call threw). */
  readonly output: string | null;
  /** Whether the result is an error; `null` without a result. */
  readonly isError: boolean | null;
  /**
   * `skipped`, `interrupted`, `timed_out` or `crashed` for a result yian wrote in place of running
   * the call, else `null`.
   */
  readonly closed: string | null;
  /** When the call started; `null` for one that never ran, or without a result. */
  readonly startedAt: string | null;
}

/** One AGENT model call as `agentCall` rebuilds it. */
export interface AgentCallView {
  readonly nodeId: string;
  /** The `n` of the call's last attempt. */
  readonly n: number;
  readonly purpose: string;
  /**
   * The request, or `null` when it couldn't be rebuilt: a structured call whose model can't be
   * built to read its plan off, or a granted tool the recorded specs lack.
   */
  readonly request: AgentCallRequest | null;
  /**
   * Whether the request was rebuilt and an attempt's recorded purpose or `request_sha256` differs
   * from it; `false` when `request` is `null`.
   */
  readonly digestMismatch: boolean;
  /** Every attempt, oldest first. */
  readonly attempts: readonly AgentCallAttempt[];
  /** The last attempt's reply as recorded (what arrived before a failure), or `null` when nothing did. */
  readonly reply: MessageValue | null;
  /** Each tool call of the reply, in call order, with its result. */
  readonly toolResults: readonly AgentToolResult[];
  /** The last attempt's usage. */
  readonly usage: AgentCallUsage | null;
  /** The last attempt's finish reason, `null` when it failed. */
  readonly finish: string | null;
  /**
   * Whether a host interrupt cut the reply (`finish: "interrupted"`), or the node's honor of one
   * (its `interrupted` step) follows the call, so the node ended before acting on all of the reply.
   */
  readonly interrupted: boolean;
}

/** A node's logical model calls (`foldSteps`), a call not over yet (retried attempts only) last. */
function agentStepCalls(steps: readonly ValueRecord[]): { calls: StepExchange[]; folded: FoldedSteps } {
  const folded = foldSteps(steps);
  return { calls: folded.open.attempts.length > 0 ? [...folded.exchanges, folded.open] : [...folded.exchanges], folded };
}

/**
 * List the AGENT model calls the run log at `log` records, in the order their first attempts
 * were written. A call is one `model_call` step with the failed attempts before it that were
 * retried, as a resumed AGENT groups them; retried attempts with no attempt after them yet are a
 * call too. Reads only the `node_step` records, with `iterRunRecords`; no flow is loaded.
 *
 * @param log The JSONL run log.
 * @param options.maxLineBytes The most bytes a line may hold, as `iterRunRecords` takes it; a longer
 *   line throws `RunLogLineTooLongError`.
 * @returns Each call's node, `n`, purpose, model, start, attempt count and finish.
 * @throws `iterRunRecords`' errors.
 *
 * @example
 * for (const call of agentCalls(log)) show(call.nodeId, call.n, call.model, call.finish);
 */
export function agentCalls(log: string, options: { maxLineBytes?: number | null } = {}): AgentCallEntry[] {
  // Each node's steps in order, and the seq of each.
  const nodes = new Map<string, { steps: ValueRecord[]; seqs: number[] }>();
  for (const { envelope, seq } of iterRunRecords(log, { kinds: ["node_step"], maxLineBytes: options.maxLineBytes ?? null })) {
    const { nodeId, step } = envelope!.record as NodeStepRecord;
    let node = nodes.get(nodeId);
    if (node === undefined) nodes.set(nodeId, (node = { steps: [], seqs: [] }));
    node.steps.push(step as ValueRecord);
    node.seqs.push(seq);
  }
  const listed: { seq: number; entry: AgentCallEntry }[] = [];
  for (const [nodeId, { steps, seqs }] of nodes) {
    for (const { attempts } of agentStepCalls(steps).calls) {
      const last = attempts.at(-1)!;
      const entry = {
        nodeId,
        n: last.n,
        purpose: last.step["purpose"] as string,
        model: last.step["model"] as string,
        startedAt: attempts[0]!.step["started_at"] as string,
        attempts: attempts.length,
        finish: (last.step["finish"] as string | undefined) ?? null,
      };
      listed.push({ seq: seqs[attempts[0]!.n - 1]!, entry });
    }
  }
  return listed.sort((a, b) => a.seq - b.seq).map(({ entry }) => entry);
}

/**
 * Rebuild one AGENT model call of the run log at `log`: the request it sent, and what it got back.
 *
 * The log is replayed as `foldRunLog` replays it, up to the node's last step, so the recorded
 * flow is used whatever the files on disk hold now. The node is bound from the replayed pool when
 * it first starts, as the run bound it, and rendered with the skills and tool specs the run
 * recorded; the steps before the call give the history, steers and tool results its request
 * carried. The model config comes from `run_started` (or the call's `llm_config` step for a
 * call-time agent), without the host's key, headers or connection. The reply, tool results,
 * attempts and usage are read from the steps as recorded.
 *
 * A rebuilt request is checked against each attempt's purpose and `request_sha256`: a mismatch
 * returns it with `digestMismatch: true` rather than throwing. A request that can't be rebuilt (a
 * structured call whose model can't be built, a granted tool the recorded specs lack) is `null`,
 * with `digestMismatch: false`.
 *
 * @param log The JSONL run log.
 * @param call `nodeId` and the `n` of any attempt of the call, such as an `agentCalls` entry.
 * @param options.env The environment the load reads `YIAN_*` values from (default `processEnv`);
 *   they must be the ones the run recorded.
 * @param options.flow The flow, required when the run's flow was built in code, refused otherwise.
 * @param options.codeRoot As `foldRunLog`'s.
 * @param options.apiKey As `advance`'s: the host key a run started with `llmConfig.api_key` used.
 *   A call-time agent of such a run binds its config with it, and a structured call's model is
 *   built with it; it never reaches the view.
 * @param options.headers As `advance`'s: the header values sent with `apiKey`.
 * @param options.models Builds a structured call's model to read its plan off; never called
 *   (default `defaultModelAccess(env)`).
 * @param options.maxLineBytes The most bytes a line may hold, as `iterRunRecords` takes it; a longer
 *   line throws `RunLogLineTooLongError`.
 * @returns The call's view.
 * @throws RangeError when the log has no model call `n` of `nodeId`, or the node's binding, which
 *   finished in the run, no longer does (the `BindingError` is its cause); `foldRunLog`'s errors.
 * @throws RangeError as `advance`'s, when `apiKey` or `headers` is refused, or `headers` differ
 *   from the header names a run that uses the host key recorded.
 * @throws AuthError when a call-time agent's run used a host key and `apiKey` isn't given, or
 *   `headers` lack the names it recorded.
 * @throws RangeError when `learnable` or `memory`, a removed option, is given.
 *
 * @example
 * const [first] = agentCalls(log);
 * const view = await agentCall(log, first);
 * show(view.request?.system, view.request?.messages, view.reply, view.toolResults);
 */
export async function agentCall(
  log: string,
  call: { readonly nodeId: string; readonly n: number },
  options: {
    env?: Env | null;
    flow?: ComposedFlow | null;
    codeRoot?: string | null;
    apiKey?: string | null;
    headers?: Readonly<Record<string, string>> | null;
    models?: ModelAccess | null;
    maxLineBytes?: number | null;
  } = {},
): Promise<AgentCallView> {
  refuseRemovedOptions(options);
  const { nodeId, n } = call;
  const missing = () => new RangeError(`run log ${log} has no model call ${n} of node ${reprText(nodeId)}`);
  const { apiKey, headers } = hostCredentials(options.apiKey, options.headers);
  const env = options.env ?? processEnv;
  const root = resolvedRoot(options.codeRoot);
  const entries = [...iterRunRecords(log, { maxLineBytes: options.maxLineBytes ?? null })];
  const start = startRecord(entries[0]?.envelope);
  const flow = await withEnv(env, () => {
    checkRecorded(start.defaults);
    checkCodeRoot(log, start, root);
    return resumedFlow(log, start, options.flow ?? null);
  });
  const isStep = (record: RunRecord): record is NodeStepRecord =>
    record instanceof NodeStepRecord && record.nodeId === nodeId;
  // Replayed up to the node's last step, so the state still holds its steps (a settled node's
  // are dropped) and the values its steer steps consumed.
  const end = entries.findLastIndex(({ envelope }) => isStep(envelope.record));
  if (end === -1) throw missing();
  let node: Node | null = null;
  let inputs: Record<string, unknown> | null = null;
  const state = withEnv(env, () =>
    replayRunLog(flow, MemoryRunLog.fromRecords(entries.slice(0, end + 1).map((entry) => entry.envelope)), {
      apiKey,
      headers,
      onPrepared: ({ record }, state) => {
        if (!(record instanceof NodeStartedRecord) || record.nodeId !== nodeId || node !== null) return;
        node = state.flow.nodes.get(nodeId)!;
        try {
          inputs = bindStarted(node, state.flow, state.pool, state.wiringOverrides.get(nodeId) ?? null);
        } catch (error) {
          // A node that failed at binding in the run has no steps, so it never gets here: the
          // binding finished then and fails now (an edited log, or a yian that binds differently).
          if (!(error instanceof BindingError)) throw error;
          throw new RangeError(
            `run log ${log}: node ${reprText(nodeId)}'s binding no longer finishes, so no request is rebuilt`,
            { cause: error },
          );
        }
      },
    }),
  );
  const callTime = state.callTimeLlm as CallTimeLlm | null;
  if ((apiKey !== null || headers !== null) && usesHostKey(state.effectiveLlmByStaticPath, callTime)) {
    checkHostHeaderNames(state.effectiveLlmByStaticPath, headers, callTime);
  }
  const steps = stepsWithSteerValues(state, nodeId);
  const { calls, folded } = agentStepCalls(steps);
  const exchange = calls.find((each) => each.attempts.some((attempt) => attempt.n === n));
  if (exchange === undefined || node === null) throw missing();
  const { attempts } = exchange;
  const agent: Node = node;
  const appended = new Map(
    entries.flatMap(({ envelope, at }) => (isStep(envelope.record) ? [[envelope.record.n, at]] : [])),
  );

  const caps: RebuildCaps = {
    ...recordedGrants(state),
    models: options.models ?? defaultModelAccess(env),
    localTypes: localTypes(state.flow, agent),
  };
  const rebuiltRequests = await withEnv(env, () => agent.rebuildRequests(inputs, steps, caps));
  const rebuilt = (attempt: NumberedStep) => rebuiltRequests?.requests.get(attempt.n - 1) ?? null;
  const last = attempts.at(-1)!;
  const request = rebuilt(last)?.request ?? null;
  // A call-time agent's config is its latest `llm_config` step before the call.
  const recorded = inFlightLlmConfigs(state.flow.nodes, new Map([[nodeId, steps.slice(0, attempts[0]!.n - 1)]]));
  const config = recorded[nodeId] ?? state.effectiveLlmByStaticPath[staticPathOf(nodeId)] ?? {};
  const output = last.step["output"];
  return {
    nodeId,
    n: last.n,
    purpose: last.step["purpose"] as string,
    request: request === null ? null : requestView(request, last.step["model"] as string, withoutCredentials(config)),
    digestMismatch: request !== null && attempts.some((attempt) => rebuilt(attempt)?.digestOk !== true),
    attempts: attempts.map((attempt) => attemptView(attempt, appended.get(attempt.n)!)),
    reply: output == null ? null : fromModelMessage(Message.fromJSON(output)),
    toolResults: output == null ? [] : toolResults(Message.fromJSON(output), exchange.results),
    usage: usageView(last.step["usage"]),
    finish: (last.step["finish"] as string | undefined) ?? null,
    // The honor is the node's last step, so it follows its last call.
    interrupted: last.step["finish"] === "interrupted" || (folded.honored !== null && exchange === calls.at(-1)),
  };
}

/**
 * Return what the run recorded of its grants, as a `RebuildCaps` reads them: the skills by name,
 * and a request's tool list for some grants, expanded as the live loop expands them but over the
 * node's own groups (its memories' recorded tools) and the groups and tools the run recorded
 * (`null` when a grant names no recorded tool).
 */
function recordedGrants(state: RunState): Pick<RebuildCaps, "skills" | "toolSpecs"> {
  const skills = new Map(
    Object.entries(state.recordedSkills).map(([name, { description, instructions }]) => [
      name,
      new Skill({ name, description, instructions }),
    ]),
  );
  const scope = recordedScope(state);
  const toolSpecs = (
    grants: readonly ToolGrant[],
    loadSkill: ToolSpec | null,
    local: ReadonlyMap<string, readonly LocalToolSpec[]> = new Map(),
  ) => {
    // The node's own groups come first, as `ToolSet.withLocalGroups` puts them live.
    const names = new Map([...local].map(([group, specs]) => [group, specs.map((spec) => spec.name)]));
    const scopes = local.size === 0 ? [scope] : [localGroupScope(MEMORY_SCOPE_LABEL, names), scope];
    let granted: GrantedTool[];
    try {
      granted = expandGrants(grants, scopes);
    } catch (error) {
      if (error instanceof GrantError) return null;
      throw error;
    }
    const specOf = (tool: GrantedTool) => {
      if (tool.scope < scopes.length - 1) {
        const { name, description, parameters } = local.get(tool.group!)!.find((spec) => spec.name === tool.tool)!;
        return new ToolSpec({ name, description, parameters });
      }
      if (!Object.hasOwn(state.recordedTools, tool.key)) return null;
      const { name, description, parameters } = state.recordedTools[tool.key]!;
      return new ToolSpec({ name, description, parameters });
    };
    return requestToolSpecs(granted, specOf, loadSkill);
  };
  return { skills, toolSpecs };
}

/**
 * Return the run's recorded tools as the grant scope its `ToolSet` was: the recorded groups by
 * name, each member by its own name; a top-level tool is a recorded tool no group lists; and a
 * tool's identity is its record name, so grants expand and dedupe as they did live.
 */
function recordedScope(state: RunState): GrantScope {
  const groups = state.recordedToolGroups;
  const tools = state.recordedTools;
  const members = (group: string): readonly string[] =>
    Object.hasOwn(groups, group) ? groups[group]! : [];
  const nameOf = (key: string) => (Object.hasOwn(tools, key) ? tools[key]!.name : null);
  const listed = new Set(Object.values(groups).flat());
  return {
    label: "this run",
    bareMembers: true,
    group: (name) =>
      Object.hasOwn(groups, name) ? members(name).map((key) => nameOf(key) ?? key) : null,
    hasTool: (name) => Object.hasOwn(tools, name) && !listed.has(name),
    groupsHolding: (name) => {
      const seen = new Set<string>();
      const holding: string[] = [];
      for (const group of Object.keys(groups)) {
        const key = members(group).find((member) => nameOf(member) === name);
        if (key === undefined || seen.has(key)) continue;
        seen.add(key);
        holding.push(group);
      }
      return holding;
    },
    groupNames: () => Object.keys(groups),
    identity: (group, tool) =>
      group === null ? tool : (members(group).find((key) => nameOf(key) === tool) ?? tool),
  };
}

/**
 * `config` without what says how the host authenticated: `api_key`, `headers`, and yian's
 * `_`-prefixed `_host_key`, `_header_names` and `_connection`.
 */
function withoutCredentials(config: Readonly<Record<string, unknown>>): JsonObject {
  return Object.fromEntries(
    Object.entries(config).filter(([key]) => key !== "api_key" && key !== "headers" && !key.startsWith("_")),
  ) as JsonObject;
}

/** `request` as an `AgentCallRequest` of `model` with `params`. */
function requestView(request: Request, model: string, params: JsonObject): AgentCallRequest {
  const [first, ...rest] = request.messages;
  const system = first?.role === "system" ? first : null;
  const choice = request.toolChoice;
  const format = request.outputFormat;
  return {
    model,
    params,
    system: system === null ? null : system.text(),
    messages: (system === null ? request.messages : rest).map(fromModelMessage),
    tools: request.tools.map(({ name, description, parameters, strict }) => ({
      name,
      description,
      parameters,
      strict,
    })),
    toolChoice: choice instanceof NamedTool ? { name: choice.name } : choice,
    parallelToolCalls: request.parallelToolCalls,
    outputFormat: format === null ? null : { name: format.name, schema: format.schema, strict: format.strict },
  };
}

/** One attempt's `model_call` step, its record appended `at`, as an `AgentCallAttempt`. */
function attemptView({ n, step }: NumberedStep, at: string): AgentCallAttempt {
  const error = step["error"] as { kind: string; message: string } | undefined;
  return {
    n,
    startedAt: step["started_at"] as string,
    firstTokenAt: (step["first_token_at"] as string | null) ?? null,
    finishedAt: at,
    error: error === undefined ? null : { kind: error.kind, message: error.message },
    retryIn: step["retry_in"] == null ? null : Number(step["retry_in"]),
    usage: usageView(step["usage"]),
  };
}

/** A step's recorded `usage` as an `AgentCallUsage`, or `null`. */
function usageView(usage: unknown): AgentCallUsage | null {
  if (usage == null) return null;
  const counts = usage as Record<string, unknown>;
  const count = (key: string) => (counts[key] == null ? null : Number(counts[key]));
  return {
    inputTokens: count("input_tokens"),
    outputTokens: count("output_tokens"),
    cacheReadTokens: count("cache_read_tokens"),
    cacheCreationTokens: count("cache_creation_tokens"),
    reasoningTokens: count("reasoning_tokens"),
  };
}

/** Each tool call of `reply`, in call order, with its `tool_result` step from `results` (by call id). */
function toolResults(reply: Message, results: StepExchange["results"]): AgentToolResult[] {
  return reply.parts
    .filter((part) => part instanceof ToolCallPart)
    .map((part) => {
      const result = results.get(part.id)?.step;
      return {
        callId: part.id,
        name: part.name,
        arguments: (part.arguments as JsonObject | null) ?? null,
        output: result === undefined ? null : (result["output"] as string),
        isError: result === undefined ? null : result["is_error"] === true,
        closed: (result?.["closed"] as string | undefined) ?? null,
        startedAt: (result?.["started_at"] as string | null | undefined) ?? null,
      };
    });
}

/** Yield the envelope of each record of the log at `log`, read by `iterRunRecords` with `maxLineBytes`. */
function* streamedEnvelopes(log: string, maxLineBytes: number | null): Generator<RecordEnvelope, void, undefined> {
  for (const entry of iterRunRecords(log, { maxLineBytes })) yield entry.envelope;
}

/**
 * Prove every static AGENT grant and TOOL `tool_id` names a tool this run supplies.
 *
 * An AGENT's grants expand as its request will: a group gives its tools, a group grant the tools it
 * lists, and a bare name a top-level tool or the one group tool by that name. A TOOL node names
 * one tool, bare or as `{group: tool}`. `take` and `sleep` TOOL nodes need no tool: the engine
 * runs them. Every static node is checked, including nodes that may never run, so an unresolvable
 * id rejects the run before any record is written instead of failing its node when the node
 * starts.
 *
 * @throws RangeError when an AGENT grants a built-in (`take`, `sleep`, `ask`), a tool or group
 *   this run lacks, a tool of two groups by its bare name, a tool its group lacks, two tools that
 *   would show one name, or `ask_user` with no handler; or a TOOL node names a tool or group this
 *   run lacks, a tool of two groups by its bare name, a tool group, or `ask` or `ask_user` with no
 *   handler.
 */
function checkToolGrants(flow: CompiledFlow, tools: ToolSet): void {
  const memories = memoryInputNames(flow);
  for (const [nodeId, node] of staticNodes(flow)) {
    const grants = node.grantedTools();
    for (const grant of grants) {
      if (grant.kind !== "name") continue;
      const toolId = grant.name;
      if (BUILTIN_TOOL_IDS.has(toolId as BuiltinToolId)) {
        const reason =
          toolId === ASK
            ? "grant ask_user; ask is a TOOL-node built-in in this version"
            : `an engine built-in a TOOL node reaches with tool_id: ${toolId}`;
        throw new RangeError(`AGENT node ${reprText(nodeId)} grants ${reprText(toolId)}: ${reason}`);
      }
      if (toolId !== ASK_USER) continue;
      try {
        tools.resolve([ASK_USER]);
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        throw new RangeError(`AGENT node ${reprText(nodeId)} grants 'ask_user': ${error.message}`);
      }
    }
    try {
      // A provider refuses a name it can't call only at the first request; refuse it here.
      checkVisibleNames(expandGrants(grants, tools.scopes()));
    } catch (error) {
      if (!(error instanceof GrantError)) throw error;
      // A grant naming another flow's memory input is a host grant here.
      const named = grants.some((grant) => memories.has(grant.kind === "name" ? grant.name : grant.group));
      const hint = named ? "; memory grants name the node's own flow inputs" : "";
      throw new RangeError(`AGENT node ${reprText(nodeId)} ${error.message}${hint}`);
    }
    const call = node.hostToolCall();
    if (call === null || (call.group === null && isEngineBuiltin(call.toolId))) continue;
    try {
      tools.resolve([callRef(call)]);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      let reason: string;
      if (call.group !== null) {
        reason = `{${call.group}: ${call.toolId}}: ${error.message}`;
      } else if (
        error instanceof GrantError ||
        call.toolId === ASK ||
        call.toolId === ASK_USER ||
        tools.groups.includes(call.toolId)
      ) {
        reason = `${reprText(call.toolId)}: ${error.message}`;
      } else {
        const known = tools.toolIds;
        reason =
          `tool ${reprText(call.toolId)}, which isn't in this run's tools; ` +
          `known tools: ${writeAsText([...known])}`;
        // A group's tool is matched by the name a model sees it under, and offered as written:
        // `{group: tool}`.
        const scope = tools.scope();
        const shown = new Map<string, string>();
        for (const group of scope.groupNames()) {
          for (const name of scope.group(group)!) {
            shown.set(`{${group}: ${name}}`, visibleToolName(group, name));
          }
        }
        const candidates = new Map(
          known.map((id) => [shown.get(id) ?? id, shown.has(id) ? `\`${id}\`` : reprText(id)]),
        );
        const [close] = getCloseMatches(call.toolId, [...candidates.keys()], 1, CLOSE_MATCH_CUTOFF);
        if (close !== undefined) reason += `; did you mean ${candidates.get(close)!}?`;
      }
      throw new RangeError(`TOOL node ${reprText(nodeId)} names ${reason}`);
    }
  }
}

/**
 * Refuse a memory grant this run's host tools clash with, before any record is written: one whose
 * memory input is also a host tool or tool group of the run, and one whose tool would show to a
 * model under the name of a host tool the same node grants. Every static node is checked, child
 * flows' included, each against its own flow's inputs.
 *
 * @param hostTypes The tool names of the host memory types the flow was loaded with, by type.
 * @throws RangeError naming the node, the input and the fix.
 */
function checkMemoryGrantNames(
  flow: CompiledFlow,
  tools: ToolSet,
  hostTypes: ReadonlyMap<string, readonly string[]>,
): void {
  const scope = tools.scope();
  const table = { tools: hostTypes, opaque: false, clashes: new Set<string>() };
  for (const [nodeId, node, inputs] of nodesWithFlowInputs(flow)) {
    const grants = node.memoryGrants();
    if (grants.length === 0) continue;
    const where = `${node.kind.toUpperCase()} node ${reprText(nodeId)}`;
    const hostNames = new Set(expandGrants(node.grantedTools(), tools.scopes()).map((tool) => tool.visible));
    for (const { input, tools: listed } of grants) {
      const host = scope.group(input) !== null ? "a tool group" : scope.hasTool(input) ? "a tool" : null;
      if (host !== null) {
        throw new RangeError(`${where}: \`${input}\` is both a memory input and ${host} of this run; rename one`);
      }
      const typ = inputs.get(input);
      const names = listed ?? (isMemoryType(typ) ? memoryToolNames(typ, table) : null) ?? [];
      const shown = listed === null ? input : `{${input}: [${listed.join(", ")}]}`;
      for (const name of names) {
        const visible = visibleToolName(input, name);
        if (!hostNames.has(visible)) continue;
        throw new RangeError(
          `${where}: the memory tool \`${visible}\` of \`${shown}\` has the name of a host tool it grants; rename one`,
        );
      }
    }
  }
}

/** Return each static node of `flow` by its path, with the input types of the flow it is declared in. */
function nodesWithFlowInputs(flow: Flow): [string, Node, ReadonlyMap<string, Type | undefined>][] {
  const found: [string, Node, ReadonlyMap<string, Type | undefined>][] = [];
  const collect = (graph: Flow, path: readonly string[]): void => {
    const start = graph.nodes.get(graph.startId);
    const inputs = new Map((start?.params ?? []).map((param) => [param.name, param.contract?.type]));
    for (const [nodeId, node] of graph.nodes) {
      const nodePath = [...path, nodeId];
      found.push([nodePath.join("/"), node, inputs]);
      for (const child of node.originChildren()) collect(child as Flow, nodePath);
    }
  };
  collect(flow, []);
  return found;
}

/** Whether a TOOL node's `toolId` is `take` or `sleep`, which the engine runs, not a tool. */
function isEngineBuiltin(toolId: string): boolean {
  return toolId === TAKE || toolId === SLEEP;
}

/** Return the tool a TOOL node's call names: its bare id, or its group's tool. */
function callRef(call: HostToolCall): string | ToolRef {
  return call.group === null ? call.toolId : { group: call.group, name: call.toolId };
}

/**
 * Reject a condition that doesn't parse on any static node, before any record is written.
 *
 * A condition is an assertion, a CASE `when`, or a LOOP predicate. A YAML flow can't hold one that
 * doesn't parse (Compose rejects it at load); this catches conditions set in code.
 *
 * @throws ExpressionError when a condition doesn't parse; the message names the node's path.
 */
function checkConditions(flow: CompiledFlow): void {
  for (const [nodePath, node] of staticNodes(flow)) node.validateConditions({ path: nodePath });
}

/**
 * Reject a binding or prompt span that can't run, before any record is written.
 *
 * Checks every static flow level: each template string of a wiring source (END's is the flow's
 * output) must parse, pass each builtin arguments it takes, and name only constructors in the
 * type table the binder evaluates it with (the reading node's; none for a reserved key such as
 * MAP `over`); each node checks its prompts. A YAML flow can't hold such a span
 * (Compose rejects it at load); this catches values set in code.
 *
 * @param flow A compiled flow or a static child flow.
 * @param path The static path of the node that owns `flow`; empty for the top level.
 * @throws ExpressionError when a span can't run; the message names the node and binding, the flow
 *   output, or the prompt field.
 */
function checkTemplates(flow: Flow, path: readonly string[] = []): void {
  const owner = path.length > 0 ? `node ${reprText(path.join("/"))} ` : "";
  for (const [nodeId, node] of flow.nodes) {
    const nodePath = [...path, nodeId].join("/");
    const reserved = node.reservedWiringKeys();
    for (const [key, binding] of flow.wiring.get(nodeId) ?? []) {
      let subject: string;
      if (nodeId === flow.endId) {
        subject = `${owner}${node.bindingLabels.get(key) ?? `flow output ${reprText(key)}`}`;
      } else if (node.bindingLabels.has(key) && !node.wiringLabels.has(key)) {
        // A desugared param (an inline call's argument, a value entry's input) reads as its
        // authored site; a CASE param keeps its wiring site name.
        subject = `${owner}${node.bindingLabels.get(key)}`;
      } else if (reserved.has(key)) {
        subject = `node ${reprText(nodePath)} ${node.reservedKeyLabel(key)}`;
      } else if (node.bindsPerItem) {
        subject = `node ${reprText(nodePath)} map input ${reprText(key)}`;
      } else {
        subject = `node ${reprText(nodePath)} ${node.wiringRefLabel(key).replace(/:+$/, "")}`;
      }
      // The binder resolves a reserved key with no type table.
      checkTemplateSource(binding, subject, reserved.has(key) ? new Map() : node.types);
    }
    node.validatePrompts({ path: nodePath });
    for (const child of node.originChildren()) checkTemplates(child as Flow, [...path, nodeId]);
  }
}

/**
 * Check each template string of one wiring source, prefixing an error with `subject`.
 *
 * @throws ExpressionError when a string's span can't run.
 */
function checkTemplateSource(binding: unknown, subject: string, types: ReadonlyMap<string, Type>): void {
  for (const [, text] of iterSourceStrings(binding)) {
    try {
      checkTemplateCalls(text, types);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      throw new ExpressionError(`${subject}: ${error.message}`, { cause: error });
    }
  }
}

/**
 * Prove every static TOOL node's inputs and output agree with its tool's types. Every static node
 * is checked, including nodes that may never run, before any record is written. `take` and
 * `sleep` aren't tools and are skipped; an `ask` node is checked against yian's ask surface plus
 * the extras the run's handler declares.
 *
 * @throws RangeError when a TOOL node passes an input its tool doesn't take, omits one the tool
 *   requires, binds an input whose type can never fit the tool's, or declares an output the tool
 *   can never return.
 */
function checkToolTypes(flow: CompiledFlow, tools: ToolSet): void {
  for (const [nodeId, node] of staticNodes(flow)) {
    const call = node.hostToolCall();
    if (call === null || (call.group === null && isEngineBuiltin(call.toolId))) continue;
    const declared = toolTypes(tools.resolve([callRef(call)])[0]!);
    if (declared.inputs !== null) checkToolInputs(nodeId, call, declared.inputs, declared.required, declared.extra);
    if (declared.output !== null) checkToolOutput(nodeId, call, declared.output);
  }
}

/**
 * Prove one TOOL node's inputs match the names and types its tool declares.
 *
 * @param nodeId The node's path, for the message.
 * @param call The node's tool call.
 * @param inputs The tool's declared input types by name; `null` leaves an input unchecked.
 * @param required The names of the inputs the tool requires.
 * @param extra The type of an input not in `inputs`; `undefined` when the tool takes none, and
 *   `null` leaves such inputs unchecked.
 * @throws RangeError when an input isn't one the tool takes, a required input is missing (all
 *   missing inputs are named at once), or an input's binding type is disjoint from the tool's.
 */
function checkToolInputs(
  nodeId: string,
  call: HostToolCall,
  inputs: ReadonlyMap<string, Type | null>,
  required: ReadonlySet<string>,
  extra: Type | null | undefined,
): void {
  const tool = reprText(visibleToolName(call.group, call.toolId));
  for (const name of call.inputs) {
    if (!inputs.has(name) && extra === undefined) {
      const taken = [...inputs.keys()].sort(compareCodeUnits);
      throw new RangeError(
        `TOOL node ${reprText(nodeId)} input ${reprText(name)}: tool ${tool} takes no such ` +
          `input; it takes ${writeAsText(taken)}`,
      );
    }
  }
  const missing = [...required].filter((name) => !call.inputs.includes(name)).sort(compareCodeUnits);
  if (missing.length > 0) {
    const named = missing.length === 1 ? `input ${reprText(missing[0]!)}` : `inputs ${writeAsText(missing)}`;
    throw new RangeError(
      `TOOL node ${reprText(nodeId)} is missing ${named} that tool ${tool} requires`,
    );
  }
  for (const [name, inferred] of call.inputTypes) {
    const expected = (inputs.has(name) ? inputs.get(name) : extra) ?? null;
    // Disjoint is yian's non-coercing relation, so a binding a tool's validator would coerce (a
    // string of digits into an int) is rejected too.
    if (expected !== null && typeRelation(inferred, expected) === "disjoint") {
      throw new RangeError(
        `TOOL node ${reprText(nodeId)} input ${reprText(name)}: tool ${tool} takes ` +
          `${canonicalTypeLabel(expected)}, but the binding is ${bindingLabel(inferred)}`,
      );
    }
  }
}

/**
 * Prove one TOOL node's declared output can hold a value its tool returns; a call with no output
 * type isn't checked.
 *
 * @throws RangeError when the tool's return type is disjoint from the node's output type.
 */
function checkToolOutput(nodeId: string, call: HostToolCall, declaredOutput: Type): void {
  if (call.outputType === null) return;
  if (typeRelation(declaredOutput, call.outputType) === "disjoint") {
    throw new RangeError(
      `TOOL node ${reprText(nodeId)} output: declared ${canonicalTypeLabel(call.outputType)}, ` +
        `but tool ${reprText(visibleToolName(call.group, call.toolId))} returns ` +
        canonicalTypeLabel(declaredOutput),
    );
  }
}

/** The most characters of a one-value string `bindingLabel` shows, `...` included. */
const LITERAL_LABEL_WIDTH = 40;

/** Return a binding type's label, showing a one-value string or int by its value. */
function bindingLabel(typ: Type): string {
  const [low, high] = typ.bounds ?? [null, null];
  if (typ.kind === ValueKind.INTEGER && low !== null && low === high && !typ.nullable) return `${low} (int)`;
  if (typ.kind === ValueKind.STRING && typ.tags?.size === 1 && typ.members === undefined && !typ.nullable) {
    // Code points, so a cut never splits a surrogate pair.
    let chars = [...[...typ.tags][0]!];
    if (chars.length > LITERAL_LABEL_WIDTH) chars = [...chars.slice(0, LITERAL_LABEL_WIDTH - 3), "..."];
    return `"${chars.join("")}" (str)`;
  }
  return canonicalTypeLabel(typ);
}

/** Map each runtime node id to its static path (`staticPathOf`). */
function staticPathKeys(nodeIds: Iterable<string>): Map<string, string> {
  return new Map([...nodeIds].map((nodeId) => [nodeId, staticPathOf(nodeId)]));
}

/**
 * @internal Check the public fields of each recorded agent config before a resume appends.
 *
 * A record written before a key was retired may carry it set; resuming would then run with a config
 * the current carrier refuses. No key is required here, so the host-key marker is left out, and
 * with it the base_url and connection settings bound to that key (replay has already checked them;
 * the model call checks the URL against its connection and the header names against the headers
 * given).
 *
 * @param effective The recorded configs: the effective model map keyed by static agent path, and
 *   the config of each started call-time agent keyed by its runtime id.
 * @param labels The authored site of each synthesized agent's `llm_config`, keyed by static agent
 *   path; a message names it instead of the path.
 * @param staticPaths The static path of each runtime id in `effective`, under which its label is
 *   looked up; other keys are static paths already.
 * @throws RangeError when a config has an invalid or retired field; the message names its agent
 *   path or runtime id, or the authored site of a synthesized agent, and never echoes values.
 */
export function checkRecordedLlmFields(
  effective: Readonly<Record<string, Readonly<AgentConfig>>>,
  labels: ReadonlyMap<string, string>,
  staticPaths: ReadonlyMap<string, string> = new Map(),
): void {
  for (const [path, config] of Object.entries(effective)) {
    const hostOnly: readonly string[] =
      HOST_KEY_MARKER in config ? [HOST_KEY_MARKER, "base_url", "organization", "project", HEADER_NAMES_KEY] : [];
    const publicFields = Object.fromEntries(Object.entries(config).filter(([key]) => !hostOnly.includes(key)));
    try {
      splitModelConfig(legacyConnectionConfig(publicFields));
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      const where = labels.get(staticPaths.get(path) ?? path) ?? `Agent ${reprText(path)}`;
      throw new RangeError(`${where}: ${error.message}`, { cause: error });
    }
  }
}

/**
 * Copy the host's skills and prove every AGENT grant can be served.
 *
 * Walks top-level and static child-flow agents before any record is written. A flow that grants
 * skills needs a `skills` map (an empty one grants none), so a run never silently drops them. A
 * listed skill must be in the map.
 *
 * @returns A copy of `skills`, or `null` when the host passed none.
 * @throws TypeError when `skills` isn't a `Map` of `Skill`s each keyed by its name.
 * @throws RangeError when a flow grants skills and no map was passed, or a listed skill is missing.
 */
function checkSkills(flow: CompiledFlow, skills: ReadonlyMap<string, Skill> | null): Map<string, Skill> | null {
  let copied: Map<string, Skill> | null = null;
  if (skills !== null) {
    if (!(skills instanceof Map)) throw new TypeError("skills must be a Map of name to Skill, or null");
    for (const [name, skill] of skills as ReadonlyMap<unknown, unknown>) {
      if (!isExactly(skill, Skill) || skill.name !== name) {
        const shown = reprText(String(name));
        throw new TypeError(`skills[${shown}] must be a Skill named ${shown}`);
      }
    }
    copied = new Map(skills);
  }
  for (const [nodeId, grants] of skillGrants(flow)) {
    if (copied === null) {
      throw new RangeError(
        `AGENT node ${reprText(nodeId)} grants skills, but no skills= mapping was passed; ` +
          "pass the run's skills (or an empty Map for none)",
      );
    }
    const missing = grants === "all" ? [] : grants.filter((name) => !copied.has(name));
    if (missing.length > 0) {
      throw new RangeError(
        `AGENT node ${reprText(nodeId)} grants skills not supplied to this run: ${writeAsText(missing)}`,
      );
    }
  }
  return copied;
}

/**
 * @internal Return each static node's non-empty skill grant, keyed by its node path.
 */
export function skillGrants(flow: Flow): [string, readonly string[] | "all"][] {
  const found: [string, readonly string[] | "all"][] = [];
  for (const [nodeId, node] of staticNodes(flow)) {
    // Read kind-blind: only a node that declares `needsSkills` carries a grant.
    const grants = (node as { skills?: readonly string[] | "all" }).skills;
    if (node.needsSkills && grants !== undefined && grants.length > 0) found.push([nodeId, grants]);
  }
  return found;
}

/**
 * Return every static node of `flow` and its child flows, keyed by node path, parents before
 * their children.
 *
 * A path joins node ids with `/`: `call/write` is node `write` of the child flow that node `call`
 * runs. These are the paths `RunStarted` keys its effective model map by, and a run log's node
 * ids for nodes the graph holds from the start, so a host reads each logged node's kind as
 * `node.kind`. The nodes are the flow's own: read them, don't
 * change them.
 *
 * Kind-blind: child flows come from `Node.originChildren()`, so the run boundary imports no node
 * class.
 *
 * A node a run grows has an id of its own (`each#1/write` for MAP element 1, `node~2` for a LOOP's
 * third iteration); `staticPathOf` maps it onto the path here.
 *
 * @param flow A loaded flow, or its compiled graph.
 * @returns `[path, node]` pairs in the flow's node order, each node followed by its child flows'.
 */
export function staticNodes(flow: ComposedFlow | Flow): [string, Node][] {
  const found: [string, Node][] = [];
  collectStaticNodes(flow instanceof ComposedFlow ? flow.compiled : flow, [], found);
  return found;
}

/** Return the names of the memory inputs `flow` and its child flows declare. */
function memoryInputNames(flow: CompiledFlow): Set<string> {
  const names = new Set<string>();
  for (const [, node] of staticNodes(flow)) {
    if (node.kind !== NodeKind.START) continue;
    for (const param of node.params ?? []) if (isMemoryType(param.contract?.type)) names.add(param.name);
  }
  return names;
}

/** Append each node of `graph` under `path`, then its child flows', to `found`. */
function collectStaticNodes(graph: Flow, path: readonly string[], found: [string, Node][]): void {
  for (const [nodeId, node] of graph.nodes) {
    const nodePath = [...path, nodeId];
    found.push([nodePath.join("/"), node]);
    for (const child of node.originChildren()) collectStaticNodes(child as Flow, nodePath, found);
  }
}

/**
 * @internal Silently rebuild the state of one borrowed, locked log prefix.
 *
 * The authored fingerprint is checked before copying. The private graph gets only the complete
 * effective model map and the call-time agents `RunStarted` recorded; recovery never reads the
 * current model defaults. Recorded transitions go through the live reducer and growth builder.
 * A host `apiKey`, and the `headers` sent with it, go to the agents the map marks for it and to
 * the call-time agents. The run result reads its files through the flow's output contract, as
 * `runFlow`'s does.
 *
 * @throws TypeError when `flow` isn't a `ComposedFlow`.
 * @throws RangeError, and the other errors of `replayCompiledRunLog`.
 */
export function replayRunLog(
  flow: ComposedFlow,
  log: RunLog,
  options: {
    apiKey?: string | null;
    headers?: Readonly<Record<string, string>> | null;
    onPrepared?: ((prepared: PreparedTransition, state: RunState) => void) | null;
  } = {},
): RunState {
  if (!(flow instanceof ComposedFlow)) throw new TypeError("replayRunLog expects a ComposedFlow");
  return replayCompiledRunLog(flow.compiled, log, { ...options, resultContract: childSignature(flow).output });
}

/**
 * @internal Replay a log prefix over a private copy of an authored compiled program.
 *
 * `resultContract` is the contract the terminal output is read through to restore its files
 * (`RunState.resultContract`); a finished log produces its terminal during replay, so it is set
 * before any record is folded.
 *
 * @throws RangeError when the log's run id or sequence is malformed or changes while it is read,
 *   the prefix is empty, an envelope isn't in canonical form or breaks the run id or the sequence,
 *   the first record isn't `RunStarted`, the flow fingerprint differs, a record isn't the exact
 *   next transition, or the log ends inside its run start (before the START inputs' commit).
 * @throws DefaultsMismatchError when the run recorded `YIAN_*` values that differ from the ones in
 *   effect.
 */
export function replayCompiledRunLog(
  flow: CompiledFlow,
  log: RunLog,
  options: {
    apiKey?: string | null;
    headers?: Readonly<Record<string, string>> | null;
    resultContract?: OutputContract | null;
    onPrepared?: ((prepared: PreparedTransition, state: RunState) => void) | null;
  } = {},
): RunState {
  const { apiKey = null, headers = null, resultContract = null, onPrepared = null } = options;
  const sourceRunId = log.runId;
  const sourceSequence = log.nextSequence;
  if (typeof sourceRunId !== "string" || sourceRunId === "") {
    throw new RangeError("run log run_id must be a non-empty string");
  }
  if (!Number.isSafeInteger(sourceSequence) || sourceSequence < 0) {
    throw new RangeError("run log next_sequence must be a non-negative integer");
  }

  const supplied = [...log];
  if (log.runId !== sourceRunId || log.nextSequence !== sourceSequence) {
    throw new RangeError("run log changed while its historical prefix was read");
  }
  if (sourceSequence !== supplied.length) {
    throw new RangeError(
      `run log sequence mismatch: next_sequence is ${sourceSequence}, prefix length is ${supplied.length}`,
    );
  }
  if (supplied.length === 0) throw new RangeError("cannot replay an empty run log");

  const canonical: RecordEnvelope[] = [];
  for (const [expectedSequence, suppliedEnvelope] of supplied.entries()) {
    const line = encodeRecord(suppliedEnvelope);
    const envelope = decodeRecord(line);
    // Envelopes are frozen and compared by their canonical lines.
    if (encodeRecord(envelope) !== line) {
      throw new RangeError(`run log envelope ${expectedSequence} is not in canonical wire form`);
    }
    if (envelope.seq !== expectedSequence) {
      throw new RangeError(`run log sequence mismatch: expected ${expectedSequence}, found ${envelope.seq}`);
    }
    if (envelope.runId !== sourceRunId) {
      throw new RangeError(
        `run log run_id mismatch: expected ${reprText(sourceRunId)}, found ${reprText(envelope.runId)}`,
      );
    }
    canonical.push(envelope);
  }

  const first = canonical[0]!.record;
  if (!(first instanceof RunStartedRecord)) throw new RangeError("the first run log record must be RunStarted");
  // Values like AGENT_RETRIES change the loaded flow, so a mismatch checked after the fingerprint
  // would surface as a fingerprint error instead of naming them.
  checkRecorded(first.defaults);
  const actualFingerprint = flowFingerprint(flow);
  if (actualFingerprint !== first.flowFingerprint) {
    throw new RangeError(
      `flow fingerprint mismatch: recorded ${reprText(first.flowFingerprint)}, supplied ${reprText(actualFingerprint)}`,
    );
  }

  const privateFlow = copyGraph(flow);
  applyEffectiveLlmMap(privateFlow, legacyConnectionMap(first.effectiveLlmByStaticPath), first.callTimeLlm);
  if (apiKey !== null) {
    // A run that marked no agent ignores the key and headers: no static agent is marked, and a
    // call-time agent gets them only when its selection names the key's provider, and uses them
    // only when its resolved config is marked.
    bindHostCredentials(privateFlow, apiKey, headers);
  }
  const state = replayRecords(new RunState(privateFlow, { resultContract }), canonical, {
    growBuilder: buildGrow,
    onPrepared,
  });
  if (log.runId !== sourceRunId || log.nextSequence !== sourceSequence) {
    throw new RangeError("run log changed while its historical prefix was replayed");
  }
  // The run's inputs aren't all in the log yet, so nothing could go on from it.
  if (!state.inputsCommitted) {
    throw new RangeError(`run log ${log.path ?? reprText(sourceRunId)} ends inside its run start; start the run again`);
  }
  return state;
}

/** What `scanRunRecords` reads from a log's records. */
type RecordScan = {
  /** The start sequence of each node in flight, in start order. */
  starts: Map<string, number>;
  /** The crashed counts the run's resumes recorded, as `RunState.crashedAttempts`. */
  crashed: Map<string, number>;
  /** The sequence of the last `RunStopped` since the last `RunResumed`, or `null`. */
  lastStop: number | null;
  /** Where the records leave the run (`scanStatus`; with a replay, the reducer's result first). */
  status: Exclude<RunStatus, "running">;
  /** The failure that ends the run when `status` is `failed`. */
  failure: Failure | null;
  /** The inbox and call fold. */
  waits: WaitsState;
  /** The log's `RunStarted` envelope, or `null` for a log with no record. */
  started: RecordEnvelope | null;
  /** The last record's seq, or `null` for a log with no record. */
  lastSeq: number | null;
  /** The last record's time, or `null` for a log with no record. */
  lastAt: string | null;
};

/**
 * Read the in-flight nodes, crashed counts, waits and status from a log's records with the run
 * scan the run index shares (`foldRunScan`, `scanStatus`), keeping the deciding failure itself.
 */
function scanRunRecords(envelopes: Iterable<RecordEnvelope>): RecordScan {
  const scan = emptyRunScan();
  let nodeFailure: Failure | null = null;
  let runFailure: Failure | null = null;
  let unreachable: Failure | null = null;
  let started: RecordEnvelope | null = null;
  let lastSeq: number | null = null;
  let lastAt: string | null = null;
  for (const envelope of envelopes) {
    const { seq, record } = envelope;
    lastAt = envelope.at;
    if (seq === 0) {
      if (!(record instanceof RunStartedRecord)) throw new RangeError("the first run log record must be RunStarted");
      started = envelope;
    }
    lastSeq = seq;
    foldRunScan(scan, envelope);
    // The first of each, as the scan keeps their seqs.
    if (record instanceof NodeFailedRecord) nodeFailure ??= record.failure;
    else if (record instanceof NodeRoutedRecord) unreachable ??= record.terminalFailure;
    else if (record instanceof RunFailedRecord) runFailure ??= record.failure;
  }
  const status = scanStatus(scan);
  const failure = status === "failed" ? (nodeFailure ?? runFailure ?? unreachable) : null;
  return {
    starts: new Map(scan.inFlight),
    crashed: new Map(scan.crashed),
    lastStop: scan.lastStop,
    status,
    failure,
    waits: scan.waits,
    started,
    lastSeq,
    lastAt,
  };
}

/**
 * Replay `envelopes` read-only over a private copy of `flow`, as `advance` replays its log, and
 * read the same parts as `scanRunRecords` from the state.
 *
 * @throws RangeError, and the other errors of `replayRunLog`.
 */
function replayedScan(flow: ComposedFlow, envelopes: readonly RecordEnvelope[]): RecordScan {
  const state = replayRunLog(flow, MemoryRunLog.fromRecords(envelopes));
  let status: RecordScan["status"];
  if (state.result instanceof RunSucceeded) status = "succeeded";
  else if (state.result instanceof RunFailed) status = "failed";
  else if (state.result instanceof RunCancelled) status = "cancelled";
  else status = unfinishedStatus(state.waits, state.lastStopSequence);
  return {
    starts: new Map(state.acceptedStarts),
    crashed: new Map(state.crashedAttempts),
    lastStop: state.lastStopSequence,
    status,
    failure: state.result instanceof RunFailed ? state.result.failure : null,
    waits: state.waits,
    started: envelopes[0] ?? null,
    lastSeq: envelopes.at(-1)?.seq ?? null,
    lastAt: envelopes.at(-1)?.at ?? null,
  };
}

/**
 * Drive one segment of a run whose log is open and whose lead is committed: build the engine,
 * publish the lead, and run it. Calls waiting on `hold` reach the engine once it runs.
 *
 * @param lead The segment's post-durable observers, committed before the engine runs.
 * @returns The segment's observers, the lead's first.
 * @throws What the engine's constructor or loop throws, and the error of an `onEvent` that throws
 *   on the lead.
 */
async function driveSegment(
  state: RunState,
  appender: RunLog,
  lead: readonly RunEvent[],
  parts: {
    options: Pick<RunFlowOptions, "onEvent" | "onStream" | "live" | "signal" | "random">;
    numWorkers: number;
    tools: ToolSet;
    /** The segment's memory sessions, or `null` for a run that binds none. */
    memories: RunMemories | null;
    skills: ReadonlyMap<string, Skill> | null;
    models: ModelAccess;
    scheduler: Scheduler;
    /** The log's folder, or `null` for a process-local log. */
    runDir: string | null;
    hold: RunHold | null;
  },
): Promise<RunEvent[]> {
  const { options } = parts;
  const onEvent = options.onEvent ?? null;
  // Built before publishing so a callback's rejected promise can stop it.
  const engine = new FlowEngine(state, appender, {
    growBuilder: buildGrow,
    numWorkers: parts.numWorkers,
    onEvent,
    onStream: options.onStream ?? null,
    live: options.live ?? null,
    tools: parts.tools,
    memories: parts.memories,
    skills: parts.skills,
    models: parts.models,
    signal: options.signal ?? null,
    scheduler: parts.scheduler,
    random: options.random ?? null,
    runDir: parts.runDir,
    // Nothing could answer a pending call in a run with no log.
    durable: parts.runDir !== null,
  });
  const events: RunEvent[] = [];
  publishBoundaryObservers(lead, events, onEvent, (error) => engine.stopForCallback(error));
  const running = engine.run();
  // Deliveries waiting on the hold reach the engine from here on.
  parts.hold?.live(engine);
  events.push(...(await running));
  return events;
}

/**
 * Publish an already-applied segment lead before retaining it in the result. Nothing is in flight
 * yet, so a throwing callback rejects the run with no stop record; a promise the callback returns
 * that rejects goes to `stop`, which stops the run with its error.
 */
function publishBoundaryObservers(
  observers: readonly RunEvent[],
  events: RunEvent[],
  onEvent: EventHook | null,
  stop: (error: unknown) => void,
): void {
  for (const observer of observers) {
    const returned = onEvent?.(observer);
    if (returned instanceof Promise) returned.then(undefined, stop);
    events.push(observer);
  }
}

/**
 * Project one terminal or waiting reducer state into the public result.
 *
 * @throws Error when the state has no terminal result and no node waits.
 */
function resultFromState(
  state: RunState,
  path: string | null,
  events: readonly RunEvent[],
  memoryDrift: readonly MemoryDrift[] = [],
): RunResult {
  const terminal = state.result;
  const common = {
    input: copyValue(state.input) as Record<string, unknown>,
    runId: requiredStateRunId(state),
    log: path,
    events,
    crashedAttempts: new Map(state.crashedAttempts),
    memoryDrift,
  };
  const unread = new Map(unreadEvents(state.waits));
  if (terminal instanceof RunSucceeded) {
    return new RunResult({ ...common, status: "succeeded", output: copyValue(terminal.output as Value), unread });
  }
  if (terminal instanceof RunFailed) {
    return new RunResult({ ...common, status: "failed", failure: terminal.failure, unread });
  }
  if (terminal instanceof RunCancelled) return new RunResult({ ...common, status: "cancelled", unread });
  if (terminal === null && state.waits.waiting.size > 0) {
    return new RunResult({ ...common, status: "waiting", waits: waitInfos(state.waits) });
  }
  throw new Error("engine returned without deriving a terminal run result");
}

/**
 * Return the initialized non-empty run identity a `RunResult` carries.
 *
 * @throws RangeError when the state has no valid run id.
 */
function requiredStateRunId(state: RunState): string {
  if (typeof state.runId !== "string" || state.runId === "") {
    throw new RangeError("terminal run state has no valid run_id");
  }
  return state.runId;
}

/** Return an error's message, or the thrown value as text, for a warning; never throws. */
function errorMessage(error: unknown): string {
  try {
    return error instanceof Error ? error.message : String(error);
  } catch {
    return "an error that can't be shown";
  }
}

/** Return the error for the server under `key`, which another run has open or is opening. */
function otherRunError(key: string, options?: ErrorOptions): RangeError {
  return new RangeError(
    `tools[${JSON.stringify(key)}]: this MCP server was opened by another run; ` +
      'open it with "await server.open()" to share it between runs',
    options,
  );
}

/**
 * Split the run's tools into the entries that need no opening and the unopened servers.
 *
 * An unopened server under a key `ToolSet` rejects (empty, or a reserved name: `take`, `sleep`,
 * `ask`, `ask_user`, `load_skill`) stays with the other entries, so `ToolSet` throws its usual error
 * before any server starts.
 *
 * @param tools The `tools` option of `runFlow`. Anything other than a record is returned as is,
 *   for `ToolSet` to reject.
 * @returns `tools` without its unopened `McpServer` entries, and those servers in key order with
 *   the first key each sits under, each once even when it sits under several keys.
 * @throws RangeError when a server entry is open because another run in progress opened it.
 */
function splitRunServers(
  tools: Readonly<Record<string, ToolEntry>> | null,
): [Readonly<Record<string, ToolEntry>> | null, [string, McpServer][]] {
  if (!isRecord(tools)) return [tools, []];
  // No prototype, so a `__proto__` key stays an entry.
  const others: Record<string, ToolEntry> = Object.create(null);
  const unopened: [string, McpServer][] = [];
  for (const key of Object.keys(tools)) {
    const entry = tools[key]!;
    if (!isMcpServer(entry) || key === "" || RESERVED_TOOL_NAMES.has(key)) {
      others[key] = entry;
    } else if (entry.isOpen) {
      if (entry.openedByRun) throw otherRunError(key);
      others[key] = entry;
    } else if (!unopened.some(([, seen]) => seen === entry)) {
      unopened.push([key, entry]);
    }
  }
  return [others, unopened];
}

/**
 * Open the servers for one run, in order; if one fails, close the ones already open.
 *
 * Once `signal` aborts, no further server opens: the ones already open close and the run rejects
 * with its reason, before anything is recorded.
 *
 * @param servers Unopened servers, each listed once with the `tools` key it sits under.
 * @param signal The run's stop.
 * @returns The opened servers, owned by a token unique to this run.
 * @throws RangeError when another caller opened or is opening a server since it was found
 *   unopened.
 * @throws Error when the MCP SDK is not installed.
 * @throws McpConnectError when a server doesn't start or doesn't list its tools.
 * @throws The signal's reason when it aborts before every server is open.
 */
async function openRunServers(
  servers: readonly [string, McpServer][],
  signal: AbortSignal | null,
): Promise<McpServer[]> {
  const token = `run-${randomUUID().replaceAll("-", "")}`;
  const opened: McpServer[] = [];
  try {
    for (const [key, server] of servers) {
      signal?.throwIfAborted();
      try {
        await server.openForRun(token, signal === null ? {} : { signal });
      } catch (error) {
        if (isAlreadyOpenError(error)) throw otherRunError(key, { cause: error });
        throw error;
      }
      opened.push(server);
    }
  } catch (error) {
    await closeRunServers(opened);
    throw error;
  }
  return opened;
}

/**
 * Close the run's servers in order.
 *
 * `McpServer.close()` never rejects, so there is no close error to note on the run's.
 */
async function closeRunServers(servers: readonly McpServer[]): Promise<void> {
  for (const server of servers) await server.close();
}
