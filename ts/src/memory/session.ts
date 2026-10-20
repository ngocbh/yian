/**
 * Conversations and learned notes with optional single-owner SQLite storage.
 *
 * This backend owns FIFO turn persistence and atomic learned-note commits. It accepts an injected
 * cooperative-stop signal but owns no provider, scheduler, run log or model call. Durable values
 * use the lower private storage adapter.
 *
 * Knows about: `memory/contracts`, `memory/conversation`, `memory/summary`, `memory/storage/sqlite`,
 * `defaults`, `typesys/json`.
 * Never imports: the `memory` facade (Python's import cycle), nodes, runtime, llm.
 */

// The default import keeps calls patchable by test spies.
import crypto from "node:crypto";

import { current } from "../defaults.ts";
import { logJson } from "../typesys/json.ts";
import {
  ConversationAdmitted,
  ConversationClosed,
  type ConversationCheckpoint,
  type ConversationEffectKind,
  ConversationEffectObserved,
  ConversationEffectRequested,
  type ConversationEffectStatus,
  type ConversationInvocation,
  type ConversationItem,
  ConversationRequest,
  ConversationStamp,
  type ConversationTurn,
  ConversationView,
  freezeJsonObject,
  type JsonObject,
  type JsonValue,
  LearnContext,
  type Memory,
  MemoryContent,
  type MemoryEpisode,
  MemoryMutation,
  MemoryOperationError,
  type MemoryOperationFailure,
  type MemorySession,
  RecallContext,
  sameConversationInvocation,
  type TranscriptSource,
  validateMutations,
} from "./contracts.ts";
import {
  advanceConversation,
  conversationContentDigest,
  conversationViewFromJson,
  conversationViewToJson,
  invocationJson,
} from "./conversation.ts";
import { checkStoragePath, type SQLiteStorage } from "./storage/sqlite.ts";
import type { Summary } from "./summary.ts";

const NOTE_FIELDS: readonly string[] = [
  "consumer_static_path",
  "execution_seq",
  "message_index",
  "note_id",
  "run_id",
  "text",
];
const SNAPSHOT_FIELDS: readonly string[] = ["conversation", "generation", "namespace", "notes", "version"];

/** One FIFO ticket, and once granted the lease, of an invocation on the instance conversation. */
interface Lease {
  readonly session: Session;
  readonly request: ConversationRequest;
  readonly ticket: number;
  active: boolean;
  released: boolean;
  /** The live view the lease was granted over; `null` until granted and after release. */
  before: ConversationView | null;
}

/** The inventory fields a publication replaces. */
interface StateChanges {
  used?: boolean;
  commits?: ReadonlyMap<string, string>;
  summaries?: ReadonlyMap<string, Summary>;
}

/** Build one stable backend failure; only `busy` is retryable. */
function error(
  code: string,
  message: string,
  options: { stage?: string; cause?: unknown } = {},
): MemoryOperationError {
  const stage = options.stage ?? "conversation";
  const failure = new MemoryOperationError(stage, "session." + code, message, code === "busy");
  if ("cause" in options) failure.cause = options.cause;
  return failure;
}

/**
 * Append the note `text()` to `error`'s own `notes` array, creating it when absent (Python's
 * `add_note`). A failure, such as a frozen error, a non-array `notes`, a primitive or a throwing
 * `text`, loses the note, as Python swallows a failing `add_note`.
 */
function addNote(error: unknown, text: () => string): void {
  try {
    if (typeof error !== "object" || error === null) return;
    const target = error as { notes?: unknown };
    const note = text();
    if (!Object.hasOwn(target, "notes")) target.notes = [];
    if (Array.isArray(target.notes)) target.notes.push(note);
  } catch {}
}

// Local copy of memory/episodic.ts's typeName.
/** Python's `type(value).__name__`: the constructor's name, else a string `name`, else `Error`. */
function typeName(value: unknown): string {
  const constructorName = (value as { constructor?: { name?: unknown } } | null | undefined)?.constructor?.name;
  if (typeof constructorName === "string" && constructorName) return constructorName;
  const name = (value as { name?: unknown } | null | undefined)?.name;
  return typeof name === "string" ? name : "Error";
}

/** Preserve a primary failure while noting a secondary cleanup fault on it (Python's `_secondary`). */
function secondary(primary: unknown, cleanup: unknown): void {
  addNote(primary, () => {
    const message = (cleanup as { message?: unknown } | null | undefined)?.message;
    const text = typeof message === "string" ? message : String(cleanup);
    return `Conversation cleanup also failed: ${typeName(cleanup)}: ${text}`;
  });
}

// Local copy of memory/contracts.ts's isExactly, which that module keeps private.
/** Whether `value` is an instance of exactly `cls`, not of a subclass. */
function isExactly<T>(value: unknown, cls: abstract new (...args: never[]) => T): value is T {
  return typeof value === "object" && value !== null && Object.getPrototypeOf(value) === cls.prototype;
}

/** Whether `value` is a JSON object (not an array or `null`). */
function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Whether `data` has exactly the keys `fields` (Python's `set(data) == {...}`). */
function hasFields(data: JsonObject, fields: readonly string[]): boolean {
  return Object.keys(data).length === fields.length && fields.every((field) => Object.hasOwn(data, field));
}

// Local copy of memory/contracts.ts's requireString, which that module keeps private.
/**
 * Require one exact string with a canonical UTF-8 representation.
 *
 * @throws TypeError when `value` isn't a string.
 * @throws RangeError when it's empty and `nonempty` is set, or holds a lone surrogate.
 */
function requireString(value: unknown, name: string, nonempty = true): string {
  if (typeof value !== "string") throw new TypeError(`${name} must be a string`);
  if (nonempty && !value) throw new RangeError(`${name} must be non-empty`);
  if (!value.isWellFormed()) throw new RangeError("Unicode surrogate is not valid canonical JSON");
  return value;
}

// Local copy of memory/contracts.ts's requirePath, which that module keeps private.
/**
 * Require one non-empty array path of non-empty strings.
 *
 * @throws TypeError when `value` isn't an array or a component isn't a string.
 * @throws RangeError when the path or a component is empty, or a component holds a lone surrogate.
 */
function requirePath(value: unknown, name: string): void {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) {
    throw new TypeError(`${name} must be an array`);
  }
  if (value.length === 0) throw new RangeError(`${name} must be non-empty`);
  for (const part of value) requireString(part, `${name} components`);
}

// Local copy of memory/contracts.ts's requireCounter for JSON ints, which arrive as `bigint`s.
/**
 * Require one non-negative JSON int counter.
 *
 * @throws TypeError when `value` isn't a `bigint`.
 * @throws RangeError when it's negative or beyond 2^53 − 1.
 */
function jsonCounter(value: unknown, name: string): bigint {
  if (typeof value !== "bigint") throw new TypeError(`${name} must be an integer`);
  if (value < 0n) throw new RangeError(`${name} must be non-negative`);
  // Counters become `number`s, so one past 2^53 − 1 would lose its increment.
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError(`${name} must be at most ${Number.MAX_SAFE_INTEGER}`);
  }
  return value;
}

/**
 * Require one history setting: a safe-integer `number` or a `bigint` of at least `minimum`,
 * returned as a `number`.
 *
 * @throws TypeError when `value` isn't an integer.
 * @throws RangeError when it's below `minimum` or beyond 2^53 − 1.
 */
function setting(value: unknown, name: string, minimum: number): number {
  if (typeof value !== "bigint" && !(typeof value === "number" && Number.isInteger(value))) {
    throw new TypeError(`${name} must be an exact integer`);
  }
  if (value < minimum) throw new RangeError(`${name} must be at least ${minimum}`);
  if (value > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(`${name} must be at most ${Number.MAX_SAFE_INTEGER}, got ${value}`);
  }
  return Number(value);
}

/**
 * Return a detached frozen copy of one snapshot JSON object, at any depth, under the yian value
 * number rule.
 *
 * @throws TypeError when `value` isn't a plain object or holds a non-JSON value.
 * @throws RangeError for NaN, a lone surrogate or a cycle.
 */
function wire(value: unknown): JsonObject {
  return freezeJsonObject(value, "session snapshot", { limitNesting: false, valueNumbers: true });
}

/** Whether two views have the same complete wire text. */
function same(left: ConversationView, right: ConversationView): boolean {
  return logJson(conversationViewToJson(left)) === logJson(conversationViewToJson(right));
}

/** Return a note's identity: the SHA-256 of its source's log JSON. */
function noteId(source: JsonObject): string {
  return crypto.createHash("sha256").update(logJson(source)).digest("hex");
}

// Local copy of memory/conversation.ts's invocationKey, which that module keeps private.
/** The lease key of one invocation. */
function invocationKey(invocation: ConversationInvocation): string {
  return JSON.stringify([invocation.runId, invocation.startSequence]);
}

/** Whether `view` admitted the invocation `[runId, executionSeq]`. */
function admitted(view: ConversationView | null, runId: string, executionSeq: number): boolean {
  // Local copy of memory/conversation.ts's operationKey for the admission, ordinal 0.
  const identity = JSON.stringify([runId, executionSeq, 0]);
  return view !== null && view.applied.some((entry) => entry.operationId === identity);
}

/**
 * Detach a closed note and prove its source identity.
 *
 * @throws TypeError or RangeError when `value` isn't a valid note.
 */
function validateNote(value: unknown): JsonObject {
  const note = wire(value);
  if (!hasFields(note, NOTE_FIELDS)) throw new RangeError("note fields do not match the supported shape");
  if (!Array.isArray(note.consumer_static_path)) throw new TypeError("note consumer path must be an array");
  requirePath(note.consumer_static_path, "consumer_static_path");
  requireString(note.run_id, "run_id");
  jsonCounter(note.execution_seq, "execution_seq");
  jsonCounter(note.message_index, "message_index");
  requireString(note.text, "text", false);
  requireString(note.note_id, "note_id");
  const { note_id: identity, ...source } = note;
  if (identity !== noteId(source)) throw new RangeError("note identity does not match its source");
  return note;
}

/** Build one deep-frozen version 3 snapshot, ints as `bigint`s. */
function snapshotJson(
  namespace: string,
  generation: number,
  view: ConversationView | null,
  notes: readonly JsonObject[],
): JsonObject {
  // The view wire and the notes are frozen already, so the envelope shares them; keys in
  // code-unit order, as a frozen JSON copy has them.
  return Object.freeze({
    conversation: view === null ? null : conversationViewToJson(view),
    generation: BigInt(generation),
    namespace,
    notes: Object.freeze([...notes]),
    version: 3n,
  });
}

/** An empty view of one generation with `compatibility`. */
function zero(namespace: string, generation: number, compatibility: JsonObject): ConversationView {
  return new ConversationView(namespace, generation, 0, compatibility, [], [], [], []);
}

/** Prove a complete historical view against a newer ledger without guessing. */
function prefix(earlier: ConversationView, later: ConversationView): boolean {
  if (
    earlier.namespace !== later.namespace ||
    earlier.generation !== later.generation ||
    logJson(earlier.compatibility) !== logJson(later.compatibility)
  ) {
    return false;
  }
  if (earlier.revision > later.revision) return false;
  if (earlier.revision === later.revision) return same(earlier, later);
  // Python compares the ledger entries as dataclasses: field by field, item by item.
  const ledgerMatches = earlier.applied.every((entry, index) => {
    const other = later.applied[index]!;
    return (
      entry.operationId === other.operationId &&
      entry.eventDigest === other.eventDigest &&
      entry.beforeDigest === other.beforeDigest &&
      entry.afterDigest === other.afterDigest
    );
  });
  if (!ledgerMatches) return false;
  const expected = earlier.revision
    ? later.applied[earlier.revision - 1]!.afterDigest
    : later.applied[0]!.beforeDigest;
  return conversationContentDigest(earlier) === expected;
}

/**
 * Decode the closed version 3 envelope without normalizing note order.
 *
 * @returns The frozen envelope, its view (or `null`) and its notes.
 * @throws MemoryOperationError `session.invalid_snapshot` for any defect;
 *   `session.unsupported_history` for retired provider-native history.
 */
function parseSnapshot(value: unknown): [JsonObject, ConversationView | null, readonly JsonObject[]] {
  try {
    const data = wire(value);
    if (data.version !== 3n) throw new RangeError("snapshot version must be 3");
    if (!hasFields(data, SNAPSHOT_FIELDS)) throw new RangeError("snapshot fields do not match the supported version");
    requireString(data.namespace, "namespace");
    const generation = jsonCounter(data.generation, "generation");
    let view: ConversationView | null = null;
    if (data.conversation !== null) {
      view = conversationViewFromJson(data.conversation as JsonObject);
      if (view.namespace !== data.namespace || BigInt(view.generation) !== generation) {
        throw new RangeError("snapshot contains a foreign conversation");
      }
    }
    if (!Array.isArray(data.notes)) throw new TypeError("notes must be an array");
    const notes: JsonObject[] = [];
    const ids = new Set<string>();
    for (const raw of data.notes as readonly JsonValue[]) {
      const note = validateNote(raw);
      const id = note.note_id as string;
      if (ids.has(id) || !admitted(view, note.run_id as string, Number(note.execution_seq))) {
        throw new RangeError("snapshot contains duplicate or unadmitted notes");
      }
      ids.add(id);
      notes.push(note);
    }
    return [data, view, Object.freeze(notes)];
  } catch (cause) {
    if (!(cause instanceof TypeError || cause instanceof RangeError)) throw cause;
    throw error("invalid_snapshot", "invalid SessionMemory snapshot", { stage: "snapshot", cause });
  }
}

/** Encode one cached summary as its inventory record. */
function summaryJson(summary: Summary): JsonObject {
  return {
    namespace: summary.namespace,
    generation: BigInt(summary.generation),
    projection_version: summary.projectionVersion,
    covered: summary.covered.map(invocationJson),
    digest: summary.digest,
    text: summary.text,
  };
}

/**
 * Own one conversation and its notes, optionally in local SQLite.
 *
 * `new SessionMemory()` creates an isolated namespace. Injected conversation-capable callers use
 * `memory.openRun()`; runtime and provider integration are separate from this backend. With
 * `storage`, values survive restarts; the file is opened at the first operation, not at
 * construction, and `close()` releases the single file owner.
 *
 * @example
 * ```ts
 * const memory = new SessionMemory({ storage: "memory.sqlite" });
 * try {
 *   const session = memory.openRun();
 *   // ...
 *   session.close();
 * } finally {
 *   memory.close();
 * }
 * ```
 */
export class SessionMemory implements Memory {
  // TS-private, not `#`, so tests can spy on and read them, as Python's tests read `_view`.
  private readonly historyBudgetChars: number;
  private readonly recentTurns: number;
  private summaries: ReadonlyMap<string, Summary> = new Map();
  private closed = false;
  private storage: SQLiteStorage | null = null;
  private readonly storagePath: string | null;
  private readonly createParent: boolean;
  private namespace: string;
  private generation = 0;
  private used = false;
  private view: ConversationView | null = null;
  private notes: readonly JsonObject[] = Object.freeze([]);
  private appliedCommits: ReadonlyMap<string, string> = new Map();
  private readonly queue: Lease[] = [];
  private active: Lease | null = null;
  private nextTicket = 0;
  /** Wake-ups of reservations waiting for the lease; `notifyAll` resolves and clears them. */
  private readonly waiters = new Set<() => void>();

  /**
   * Create an empty namespace with an optional API context budget. An option that is `undefined`
   * or `null` is unset.
   *
   * @param options.historyBudgetChars Positive soft context character budget; unset uses
   *   `YIAN_SESSION_HISTORY_BUDGET_CHARS` (32,000). Raw history is never removed.
   * @param options.recentTurns Non-negative number of whole prior turns kept unsummarized; unset
   *   uses `YIAN_SESSION_RECENT_TURNS` (4). The current turn is always kept.
   * @param options.storage Optional local SQLite file; unset is process-local. Nothing is opened
   *   until the first operation; this backend then owns the file until close.
   * @param options.createParent Create the storage file's parent directory at first open.
   * @throws TypeError when a setting isn't an integer or `createParent` isn't a boolean.
   * @throws RangeError when a setting is outside its range or beyond 2^53 − 1.
   * @throws MemoryOperationError `storage.path` when `storage` isn't a local file path.
   * @throws DefaultsError when a setting is unset and a `YIAN_` variable holds a value it doesn't
   *   accept.
   */
  constructor(
    options: {
      historyBudgetChars?: number | bigint | null;
      recentTurns?: number | bigint | null;
      storage?: string | null;
      createParent?: boolean | null;
    } = {},
  ) {
    const historyBudgetChars = options.historyBudgetChars ?? current().sessionHistoryBudgetChars;
    const recentTurns = options.recentTurns ?? current().sessionRecentTurns;
    this.historyBudgetChars = setting(historyBudgetChars, "history_budget_chars", 1);
    this.recentTurns = setting(recentTurns, "recent_turns", 0);
    this.namespace = crypto.randomUUID().replaceAll("-", "");
    const createParent = options.createParent ?? false;
    if (typeof createParent !== "boolean") throw new TypeError("create_parent must be a bool");
    const storage = options.storage ?? null;
    this.storagePath = storage === null ? null : checkStoragePath(storage);
    this.createParent = createParent;
  }

  /**
   * Guard a value operation (Python's `_locked()`); cleanup paths don't call it.
   *
   * @throws MemoryOperationError `session.closed` after `close()`.
   */
  private guard(): void {
    if (this.closed) throw error("closed", "memory backend is closed", { stage: "session" });
  }

  /** Wake every waiting reservation so it re-checks its ticket. */
  private notifyAll(): void {
    for (const wake of this.waiters) wake();
    this.waiters.clear();
  }

  /** Wait until the next `notifyAll` or until `signal` aborts, whichever comes first. */
  private wait(signal: AbortSignal): Promise<void> {
    return new Promise((resolve) => {
      const wake = (): void => {
        signal.removeEventListener("abort", wake);
        this.waiters.delete(wake);
        resolve();
      };
      this.waiters.add(wake);
      signal.addEventListener("abort", wake);
    });
  }

  /** Save a complete validated candidate before publishing it in RAM; nothing without storage. */
  private saveState(snapshot: JsonObject, changes: StateChanges = {}): void {
    if (this.storage === null) return;
    this.storage.save({
      version: 1n,
      snapshot,
      used: changes.used ?? this.used,
      applied_commits: Object.fromEntries(changes.commits ?? this.appliedCommits),
      summaries: [...(changes.summaries ?? this.summaries).values()].map(summaryJson),
    });
  }

  /**
   * Release storage and invalidate idle handles without interrupting work. Repeated closes are
   * inert; active turns and queued tickets keep their ownership even after their run handle closes.
   *
   * @throws MemoryOperationError `session.busy` (retryable) while a turn is active or queued; a
   *   storage cleanup failure.
   */
  close(): void {
    if (this.closed) return;
    if (this.active !== null || this.queue.length > 0) {
      throw error("busy", "cannot close active or queued conversations", { stage: "close" });
    }
    this.storage?.close();
    this.closed = true;
  }

  /**
   * Open a run with a pinned, deep-frozen public snapshot. Only a pristine backend may adopt a
   * snapshot's namespace and history; a used one accepts only proven history of its own.
   *
   * @param options.snapshot Exact recorded snapshot to restore or replay; `null` or unset pins the
   *   current live state.
   * @returns A run-scoped memory session.
   * @throws MemoryOperationError `session.invalid_snapshot` when the snapshot's shape or lineage is
   *   invalid; `session.closed` after `close()`.
   */
  openRun(options: { snapshot?: JsonObject | null } = {}): Session {
    const recorded = options.snapshot ?? null;
    const parsed = recorded === null ? null : parseSnapshot(recorded);
    this.guard();
    let namespace = this.namespace;
    let generation = this.generation;
    let liveView = this.view;
    let liveNotes = this.notes;
    let pinned: JsonObject;
    if (parsed === null) {
      pinned = snapshotJson(this.namespace, this.generation, this.view, this.notes);
    } else {
      const [data, view, notes] = parsed;
      pinned = data;
      if (!this.used) {
        namespace = data.namespace as string;
        generation = Number(data.generation);
        liveView = view;
        liveNotes = notes;
      } else if (
        data.namespace !== this.namespace ||
        data.generation !== BigInt(this.generation) ||
        (view !== null && (this.view === null || !prefix(view, this.view))) ||
        logJson(notes) !== logJson(this.notes.slice(0, notes.length))
      ) {
        throw error("invalid_snapshot", "snapshot is not proven history of this backend", { stage: "snapshot" });
      }
    }
    if (!this.used) this.saveState(snapshotJson(namespace, generation, liveView, liveNotes), { used: true });
    this.namespace = namespace;
    this.generation = generation;
    this.view = liveView;
    this.notes = liveNotes;
    this.used = true;
    return new Session(this, pinned);
  }

  /**
   * Clear the conversation and notes and invalidate older generation handles. Commit tombstones
   * remain, so exact retries can't resurrect cleared notes. Provider resources aren't deleted.
   *
   * @throws MemoryOperationError `session.busy` (retryable), without changing state, while a turn
   *   is active or queued; `session.closed` after `close()`.
   */
  reset(): void {
    this.guard();
    if (this.active !== null || this.queue.length > 0) {
      throw error("busy", "cannot reset active or queued conversations", { stage: "reset" });
    }
    this.saveState(snapshotJson(this.namespace, this.generation + 1, null, []), { used: true, summaries: new Map() });
    this.view = null;
    this.notes = Object.freeze([]);
    this.summaries = new Map();
    this.generation += 1;
    this.used = true;
    this.notifyAll();
  }
}

/**
 * One run handle over a pinned snapshot and live conversation capabilities (Python's `_Session`).
 *
 * It reads and publishes its backend's private state through element access (`memory["view"]`),
 * TypeScript's escape for private members, as Python's session reads `self._memory._view`.
 */
export class Session implements MemorySession, TranscriptSource {
  private readonly memory: SessionMemory;
  private readonly pinned: JsonObject;
  private readonly namespace: string;
  private readonly generation: number;
  private closed = false;
  /** Every lease this session reserved, by invocation; released entries stay, as in Python. */
  private readonly leases = new Map<string, Lease>();

  /** @internal Pin one snapshot; `SessionMemory.openRun` builds sessions. */
  constructor(memory: SessionMemory, pinned: JsonObject) {
    this.memory = memory;
    this.pinned = pinned;
    this.namespace = memory["namespace"];
    this.generation = memory["generation"];
  }

  /**
   * Require this handle to be open and of the current generation.
   *
   * @throws MemoryOperationError `session.stale` otherwise.
   */
  private ensureLive(): void {
    if (
      this.closed ||
      this.namespace !== this.memory["namespace"] ||
      this.generation !== this.memory["generation"]
    ) {
      throw error("stale", "memory session is closed or belongs to an old generation");
    }
  }

  /**
   * The run's original snapshot, deep-frozen: the exact pinned public envelope, unaffected by later
   * conversation checkpoints.
   *
   * @throws MemoryOperationError `session.stale` when this handle is closed or reset-stale;
   *   `session.closed` after the backend closes.
   */
  get snapshot(): JsonObject {
    this.memory["guard"]();
    this.ensureLive();
    return this.pinned;
  }

  /**
   * Contribute pinned notes and the live transcript source for one consumer. Raw conversation
   * history is never rendered as recall text.
   *
   * @param context Optional exact visible-context carrier; `null` or unset yields no notes.
   * @returns The instance's pinned notes in retained order, one per line, and this session as the
   *   transcript source.
   * @throws TypeError when a supplied context isn't exactly a `RecallContext`.
   * @throws MemoryOperationError `session.stale` when this handle is closed or reset-stale.
   */
  recall(context: RecallContext | null = null): MemoryContent {
    if (context !== null && !isExactly(context, RecallContext)) {
      throw new TypeError("context must be a RecallContext or null");
    }
    this.memory["guard"]();
    this.ensureLive();
    if (context === null) return new MemoryContent({ transcript: this });
    const notes = (this.pinned.notes as readonly JsonObject[]).map((note) => note.text as string).join("\n");
    return new MemoryContent({ text: notes, transcript: this });
  }

  /**
   * Propose notes from self-feedback for the instance's admitted invocations: one append-note
   * mutation per distinct source, in feedback order. Automatic history is unchanged, and no provider
   * or model is called.
   *
   * @throws TypeError when `context` isn't exactly a `LearnContext`.
   * @throws MemoryOperationError `session.invalid_context` when the context is invalid for this
   *   instance; `session.stale` when this handle is closed or reset-stale.
   */
  learn(context: LearnContext): MemoryMutation[] {
    if (!isExactly(context, LearnContext)) throw new TypeError("context must be a LearnContext");
    this.memory["guard"]();
    this.ensureLive();
    try {
      const episodes = new Map<number, MemoryEpisode>();
      for (const episode of context.episodes) {
        if (episode.executionSeq !== null) {
          if (episodes.has(episode.executionSeq)) throw new RangeError("episode execution coordinates must be unique");
          episodes.set(episode.executionSeq, episode);
        }
      }
      const notes = new Map<string, JsonObject>();
      for (const feedback of context.feedback) {
        if (!isObject(feedback) || feedback.channel !== "self" || feedback.root === true) continue;
        const sequence = feedback.execution_seq ?? null;
        if (sequence === null) continue;
        const executionSeq = Number(jsonCounter(sequence, "execution_seq"));
        const episode = episodes.get(executionSeq);
        if (episode === undefined) continue;
        if (!admitted(this.memory["view"], context.runId, executionSeq)) continue;
        const source: JsonObject = {
          consumer_static_path: [...episode.ownerStaticPath],
          run_id: context.runId,
          execution_seq: sequence,
          message_index: feedback.message_index ?? null,
          text: feedback.value ?? null,
        };
        const note = validateNote({ note_id: noteId(source), ...source });
        notes.set(note.note_id as string, note);
      }
      return [...notes].map(([id, note]) => new MemoryMutation(id, { operation: "append_note", note }));
    } catch (cause) {
      if (!(cause instanceof TypeError || cause instanceof RangeError)) throw cause;
      throw error("invalid_context", "learning context is invalid", { stage: "learn", cause });
    }
  }

  /**
   * Atomically append one exact idempotent batch of learned notes. Exact replays stay inert even
   * after reset; a different batch under the same id is a collision. Empty batches are allowed.
   *
   * @param commitId Non-empty caller-owned batch identity.
   * @param mutations Exact append-note proposals for this instance.
   * @throws TypeError when `commitId` isn't a string or `mutations` isn't an array.
   * @throws RangeError when `commitId` is empty or holds a lone surrogate.
   * @throws MemoryOperationError `session.invalid_batch` for a malformed, foreign or unadmitted
   *   batch; `session.commit_collision`; `session.stale` when this handle is closed or reset-stale.
   */
  commit(commitId: string, mutations: readonly MemoryMutation[]): void {
    requireString(commitId, "commit_id");
    if (!Array.isArray(mutations)) throw new TypeError("mutations must be an array");
    const memory = this.memory;
    memory["guard"]();
    this.ensureLive();
    const additions: JsonObject[] = [];
    let digest: string;
    try {
      const batch: JsonObject[] = [];
      for (const mutation of validateMutations(mutations)) {
        const payload = mutation.payload;
        if (!isObject(payload) || !hasFields(payload, ["note", "operation"]) || payload.operation !== "append_note") {
          throw new RangeError("mutation must be a closed append-note operation");
        }
        const note = validateNote(payload.note);
        if (note.note_id !== mutation.mutationId) throw new RangeError("mutation and note identities must match");
        additions.push(note);
        batch.push({ mutation_id: mutation.mutationId, payload });
      }
      digest = crypto.createHash("sha256").update(logJson(batch)).digest("hex");
    } catch (cause) {
      if (!(cause instanceof TypeError || cause instanceof RangeError)) throw cause;
      throw error("invalid_batch", "mutation batch is invalid", { stage: "commit", cause });
    }
    const previous = memory["appliedCommits"].get(commitId);
    if (previous !== undefined) {
      if (previous === digest) return;
      throw error("commit_collision", "commit id identifies a different batch", { stage: "commit" });
    }
    let candidate: JsonObject;
    let notes: readonly JsonObject[];
    try {
      const retained = new Map(memory["notes"].map((note) => [note.note_id as string, note]));
      for (const note of additions) {
        if (!admitted(memory["view"], note.run_id as string, Number(note.execution_seq))) {
          throw new RangeError("note has no matching conversation admission");
        }
        const existing = retained.get(note.note_id as string);
        if (existing !== undefined && logJson(existing) !== logJson(note)) {
          throw new RangeError("note id identifies different content");
        }
        retained.set(note.note_id as string, note);
      }
      notes = Object.freeze([...retained.values()]);
      candidate = snapshotJson(this.namespace, this.generation, memory["view"], notes);
    } catch (cause) {
      if (!(cause instanceof TypeError || cause instanceof RangeError)) throw cause;
      throw error("invalid_batch", "mutation batch is invalid", { stage: "commit", cause });
    }
    const commits = new Map(memory["appliedCommits"]).set(commitId, digest);
    memory["saveState"](candidate, { commits });
    memory["notes"] = notes;
    memory["appliedCommits"] = commits;
  }

  /**
   * Open one turn for an invocation: wait in FIFO order for the instance conversation, persist the
   * admission and resolve to the admitted turn. If opening fails or the signal aborts after the
   * lease is reserved, it records the close of a published admission (`interrupted` for an abort),
   * releases the lease and rejects with that failure unchanged; the caller then has no turn to
   * close.
   *
   * @param request The authored consumer, unique invocation and exact compatibility descriptor.
   * @param options.signal Aborted when a cooperative stop is requested; checked while queuing and
   *   between effects, never before recording an actual observation.
   * @throws TypeError when `request` isn't exactly a `ConversationRequest` or the signal isn't an
   *   `AbortSignal`, before any waiting.
   * @throws MemoryOperationError `session.stale` when this handle is closed or reset-stale;
   *   `session.conflict` when the invocation already has a lease or conflicts with retained
   *   history; `session.incompatible` when the compatibility descriptor changed. The signal's
   *   reason when it aborts.
   */
  async openTurn(request: ConversationRequest, options: { signal: AbortSignal }): Promise<Turn> {
    if (!isExactly(request, ConversationRequest)) throw new TypeError("request must be a ConversationRequest");
    const signal = options?.signal;
    if (!(signal instanceof AbortSignal)) throw new TypeError("signal must be an AbortSignal");
    const turn = new Turn(this, request, signal);
    await turn["enter"]();
    return turn;
  }

  /**
   * Close this run handle idempotently and remove its queued tickets. Active owners keep their
   * leases until their turn closes, so closing a session never overlaps an in-flight external effect
   * with another turn.
   */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const lease of this.leases.values()) {
      if (!lease.active && !lease.released) this.release(lease);
    }
    this.memory["notifyAll"]();
  }

  /** Release a lease or ticket. */
  private release(lease: Lease): void {
    const memory = this.memory;
    if (memory["active"] === lease) memory["active"] = null;
    const queue = memory["queue"];
    const index = queue.indexOf(lease);
    if (index !== -1) queue.splice(index, 1);
    lease.active = false;
    lease.released = true;
    lease.before = null;
    memory["notifyAll"]();
  }

  /**
   * Reserve a FIFO position, wait in ticket order until no lease is active, then grant the lease
   * over the current live baseline. A wait ends at `notifyAll` (release, session close, reset) or
   * at the signal's abort, and the checks run again.
   *
   * @throws MemoryOperationError `session.stale`, `session.conflict` or `session.incompatible`; the
   *   signal's reason when it aborted. The ticket is released on any failure.
   */
  private async reserve(request: ConversationRequest, signal: AbortSignal): Promise<Lease> {
    const memory = this.memory;
    memory["guard"]();
    this.ensureLive();
    const key = invocationKey(request.invocation);
    if (this.leases.has(key)) throw error("conflict", "invocation already has a session lease");
    memory["nextTicket"] += 1;
    const lease: Lease = {
      session: this,
      request,
      ticket: memory["nextTicket"],
      active: false,
      released: false,
      before: null,
    };
    this.leases.set(key, lease);
    memory["queue"].push(lease);
    memory["notifyAll"]();
    try {
      for (;;) {
        signal.throwIfAborted();
        memory["guard"]();
        this.ensureLive();
        if (lease.released) throw error("stale", "conversation ticket was released");
        if (memory["queue"][0] === lease && memory["active"] === null) break;
        await memory["wait"](signal);
      }
      memory["queue"].shift();
      lease.active = true;
      memory["active"] = lease;
      let before = memory["view"];
      if (before === null) {
        before = zero(this.namespace, this.generation, request.compatibility);
      } else if (logJson(before.compatibility) !== logJson(request.compatibility)) {
        throw error("incompatible", "conversation compatibility changed; reset is required");
      }
      const stamp = new ConversationStamp(request.invocation, this.namespace, this.generation, before.revision, 0);
      // A dry run: it proves the admission applies before the lease is handed out.
      advanceConversation(before, new ConversationAdmitted(stamp, before));
      lease.before = before;
      return lease;
    } catch (primary) {
      try {
        this.release(lease);
      } catch (cleanup) {
        secondary(primary, cleanup);
      }
      throw primary;
    }
  }

  /** Validate and save a complete candidate snapshot over `view` before publishing it in RAM. */
  private candidateSnapshot(view: ConversationView): void {
    this.memory["saveState"](snapshotJson(this.namespace, this.generation, view, this.memory["notes"]));
  }

  /**
   * Require `lease` to be this session's active lease of the current generation.
   *
   * @param options.allowClosed Accept a closed handle, so actual evidence is still recorded.
   * @throws MemoryOperationError `session.stale` otherwise.
   */
  private ensureLease(lease: Lease | null, options: { allowClosed?: boolean } = {}): void {
    if (!options.allowClosed) this.ensureLive();
    const memory = this.memory;
    if (
      this.namespace !== memory["namespace"] ||
      this.generation !== memory["generation"] ||
      lease === null ||
      lease.session !== this ||
      !lease.active ||
      lease.released ||
      memory["active"] !== lease ||
      this.leases.get(invocationKey(lease.request.invocation)) !== lease
    ) {
      throw error("stale", "conversation lease is no longer active");
    }
  }
}

/**
 * One FIFO lease that persists requests and actual observations directly (Python's `_Turn`).
 *
 * `Session.openTurn` returns it admitted; the caller calls `close()` when the turn's body ends.
 */
export class Turn implements ConversationTurn {
  private readonly session: Session;
  private readonly turnRequest: ConversationRequest;
  private readonly signal: AbortSignal;
  private lease: Lease | null = null;
  private turnView: ConversationView | null = null;
  private ordinal = 0;
  private interrupted = false;

  /** @internal Build an unadmitted turn; `Session.openTurn` builds and admits turns. */
  constructor(session: Session, request: ConversationRequest, signal: AbortSignal) {
    this.session = session;
    this.turnRequest = request;
    this.signal = signal;
  }

  /** Acquire the instance conversation and persist its admission (Python's `__enter__`). */
  private async enter(): Promise<void> {
    const session = this.session;
    const memory = session["memory"];
    const lease = await session["reserve"](this.turnRequest, this.signal);
    this.lease = lease;
    try {
      memory["guard"]();
      session["ensureLease"](lease);
      this.turnView = lease.before;
      this.publish(new ConversationAdmitted(this.stamp(), this.turnView!));
      this.signal.throwIfAborted();
      memory["guard"]();
      session["ensureLease"](lease);
    } catch (failure) {
      this.finish(failure, true);
      throw failure;
    }
  }

  /** The next operation's stamp over the turn's latest view. */
  private stamp(): ConversationStamp {
    const view = this.turnView!;
    const invocation = this.turnRequest.invocation;
    return new ConversationStamp(invocation, view.namespace, view.generation, view.revision, this.ordinal);
  }

  /**
   * Validate the whole candidate snapshot, then publish one checkpoint atomically.
   *
   * @throws MemoryOperationError `session.stale` when the lease isn't active; `session.conflict`
   *   when the checkpoint isn't this lease's or the live view moved; `session.incompatible`.
   */
  private publish(checkpoint: ConversationCheckpoint): void {
    const session = this.session;
    const memory = session["memory"];
    memory["guard"]();
    session["ensureLease"](this.lease, {
      allowClosed: isExactly(checkpoint, ConversationEffectObserved) || isExactly(checkpoint, ConversationClosed),
    });
    const lease = this.lease!;
    const view = this.turnView!;
    const stamp = checkpoint.stamp;
    if (
      !sameConversationInvocation(stamp.invocation, lease.request.invocation) ||
      stamp.namespace !== session["namespace"] ||
      stamp.generation !== session["generation"]
    ) {
      throw error("conflict", "checkpoint does not belong to this lease");
    }
    if (logJson(view.compatibility) !== logJson(lease.request.compatibility)) {
      throw error("incompatible", "turn compatibility changed");
    }
    const live = memory["view"] ?? lease.before!;
    if (!same(view, live)) throw error("conflict", "turn predecessor is not the current live view");
    const after = advanceConversation(view, checkpoint);
    session["candidateSnapshot"](after);
    memory["view"] = after;
    this.turnView = after;
    this.ordinal += 1;
  }

  /**
   * The last persisted view, frozen, while the lease is live.
   *
   * @throws MemoryOperationError `session.stale` when the turn isn't active or its run is stale.
   */
  get view(): ConversationView {
    this.session["memory"]["guard"]();
    this.session["ensureLease"](this.lease);
    return this.turnView!;
  }

  /**
   * Persist an effect request before allowing its external execution.
   *
   * @param kind `model` or `tool`.
   * @param payload Exact JSON describing the external request.
   * @param options.items Newly retained items with unique ids; none by default.
   * @returns The positive effect ordinal, only after persistence, live-lease and stop checks.
   * @throws TypeError or RangeError when an argument has the wrong carrier, kind, item identity or
   *   JSON value.
   * @throws MemoryOperationError when the lease is stale or its history conflicts. A stop throws the
   *   signal's reason unchanged.
   */
  request(
    kind: ConversationEffectKind,
    payload: JsonObject,
    options: { items?: readonly ConversationItem[] } = {},
  ): number {
    const session = this.session;
    const memory = session["memory"];
    memory["guard"]();
    session["ensureLease"](this.lease);
    this.signal.throwIfAborted();
    memory["guard"]();
    session["ensureLease"](this.lease);
    const checkpoint = new ConversationEffectRequested(this.stamp(), kind, payload, options.items ?? []);
    this.publish(checkpoint);
    this.signal.throwIfAborted();
    memory["guard"]();
    session["ensureLease"](this.lease);
    return checkpoint.stamp.ordinal;
  }

  /**
   * Record actual evidence before honoring a stop that arrived during the effect.
   *
   * @param effectOrdinal Positive ordinal of a request of this turn.
   * @param status The actual `pending`, `completed`, `failed` or `interrupted` disposition.
   * @param options.items New evidence with unique ids; none by default.
   * @param options.failure Required for `failed`, optional for `interrupted`, absent otherwise.
   * @throws TypeError or RangeError when the ordinal, status, items or failure are invalid.
   * @throws MemoryOperationError when the lease is stale or the effect transition is invalid. A stop
   *   throws the signal's reason after the evidence is retained.
   */
  observe(
    effectOrdinal: number,
    status: ConversationEffectStatus,
    options: { items?: readonly ConversationItem[]; failure?: MemoryOperationFailure | null } = {},
  ): void {
    const session = this.session;
    session["memory"]["guard"]();
    session["ensureLease"](this.lease, { allowClosed: true });
    const checkpoint = new ConversationEffectObserved(
      this.stamp(),
      effectOrdinal,
      status,
      options.items ?? [],
      options.failure ?? null,
    );
    this.publish(checkpoint);
    this.signal.throwIfAborted();
  }

  /** Close this turn as `interrupted` when its body later returns normally. */
  interrupt(): void {
    this.interrupted = true;
  }

  /**
   * Record the turn's terminal disposition and release its lease (Python's `__exit__`).
   *
   * `error` is undefined after the body returns normally, else the body's thrown value. The turn
   * records `interrupted` when its signal aborted, its session was closed, or `interrupt()` was
   * called before a normal return; `failed` when `error` is given; else `completed`. A turn records
   * a close only once its admission was published, and releases its lease even when recording
   * fails. It never throws `error`: a stop or close failure seen while `error` is pending is noted on
   * `error` (`Conversation cleanup also failed: <Type>: <message>`). With `error` undefined, the
   * first failure is thrown and later ones are noted on it; after an abort that is `signal.reason`,
   * thrown once the close is recorded and the lease released. A second call does nothing.
   *
   * @throws MemoryOperationError when recording the close fails, such as a pending effect, and
   *   `error` is undefined; the signal's reason after an abort.
   */
  close(error?: unknown): void {
    this.finish(error);
  }

  /**
   * Python's `_finish`, without re-raising a pending `error`; `pending` says whether `error` is a
   * body failure, so a failed open that threw `undefined` still records `failed`.
   */
  private finish(error: unknown, pending = error !== undefined): void {
    const lease = this.lease;
    if (lease === null || lease.released) return;
    let hasPrimary = pending;
    let primary = error;
    const fail = (value: unknown): void => {
      if (!hasPrimary) {
        hasPrimary = true;
        primary = value;
      } else if (primary !== value) {
        secondary(primary, value);
      }
    };
    const session = this.session;
    try {
      if (this.signal.aborted) fail(this.signal.reason);
      if (this.ordinal) {
        session["memory"]["guard"]();
        const stopped = this.signal.aborted || session["closed"] || (!pending && this.interrupted);
        const status = stopped ? "interrupted" : pending ? "failed" : "completed";
        this.publish(new ConversationClosed(this.stamp(), status));
      }
    } catch (cleanup) {
      fail(cleanup);
    } finally {
      try {
        session["release"](lease);
      } catch (cleanup) {
        fail(cleanup);
      }
    }
    if (hasPrimary && !pending) throw primary;
  }
}
