/**
 * Reflective memory with snapshot-isolated sessions and optional SQLite storage.
 *
 * The backend keeps atomic lessons in memory, retrieves them deterministically, and asks one
 * tool-free structured model call to propose mutations for this memory. It deliberately knows
 * nothing about compiled graphs, nodes, or the scheduler.
 *
 * Knows about: `memory/contracts`, `memory/storage/sqlite`, `defaults`, `typesys/json`,
 * `typesys/values`, `llm`, `llm_clients`.
 * Never imports: the `memory` facade, nodes, runtime.
 */

// The default import keeps calls patchable by test spies.
import crypto from "node:crypto";

import { current, type Env, processEnv } from "../defaults.ts";
import type { ChatModel } from "../llm/events.ts";
import type { RetryInfo } from "../llm/executor.ts";
import { LLMConfig } from "../llm_clients/config.ts";
import { resolveModelConfig, splitModelConfig } from "../llm_clients/connection.ts";
import { chatModelFromConfig } from "../llm_clients/factory.ts";
import { logJson } from "../typesys/json.ts";
import { fromHost } from "../typesys/values.ts";
import {
  freezeJsonObject,
  type JsonObject,
  type JsonValue,
  LearnContext,
  type Memory,
  MemoryMutation,
  MemoryOperationError,
  type MemorySession,
  RecallContext,
} from "./contracts.ts";
import { checkStoragePath, SQLiteStorage } from "./storage/sqlite.ts";

const SNAPSHOT_FORMAT = "yian.reflective_memory";
const SNAPSHOT_VERSION = 2n;
const POLICY_NAME = "reflective";

/** One immutable atomic lesson in insertion order. Frozen. */
interface Lesson {
  readonly lessonId: string;
  readonly text: string;
  readonly source: JsonObject;
}

/** Effective lesson and recall limits of one reflective instance. Frozen. */
interface Limits {
  readonly maxItems: number;
  readonly recallLimit: number;
}

/** The limits and ordered lessons of one reflective instance. Frozen. */
interface State {
  readonly limits: Limits;
  readonly lessons: readonly Lesson[];
}

/** Build the learner's model from the config Python passes to `model_from_config`. */
type CreateModel = (config: Readonly<Record<string, unknown>>) => ChatModel | Promise<ChatModel>;

/** Observe one retried learner model call. */
type OnRetry = (retry: RetryInfo) => void | Promise<void>;

/** The inventory flags and commit ledger a publication replaces. */
interface StateChanges {
  appliedCommits?: ReadonlyMap<string, string>;
  everOpened?: boolean;
  everReset?: boolean;
  everCommitted?: boolean;
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

/** Build one stable typed backend failure, with `cause` (Python's `raise ... from`) when given. */
function operationError(
  stage: string,
  code: string,
  message: string,
  options: { retryable?: boolean; cause?: unknown } = {},
): MemoryOperationError {
  const error = new MemoryOperationError(stage, code, message, options.retryable ?? false);
  if ("cause" in options) error.cause = options.cause;
  return error;
}

/**
 * Append the note `text()` to `error`'s own `notes` array, creating it when absent (Python's
 * `add_note`). A failure, such as a frozen error, a non-array `notes` or a primitive, loses the
 * note, as Python swallows a failing `add_note`.
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

/** Python's `type(value).__name__`: the constructor's name, else a string `name`, else `Error`. */
function typeName(value: unknown): string {
  const constructorName = (value as { constructor?: { name?: unknown } } | null | undefined)?.constructor?.name;
  if (typeof constructorName === "string" && constructorName) return constructorName;
  const name = (value as { name?: unknown } | null | undefined)?.name;
  return typeof name === "string" ? name : "Error";
}

/**
 * Require one exact positive integer: a safe-integer `number` or a `bigint`, returned as a `number`.
 *
 * @throws TypeError when `value` isn't an integer.
 * @throws RangeError when it isn't positive or is beyond 2^53 − 1.
 */
function positiveInt(value: unknown, name: string): number {
  if (typeof value !== "bigint" && !(typeof value === "number" && Number.isInteger(value))) {
    throw new TypeError(`${name} must be an integer`);
  }
  if (value <= 0) throw new RangeError(`${name} must be positive`);
  if (value > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(`${name} must be at most ${Number.MAX_SAFE_INTEGER}, got ${value}`);
  }
  return Number(value);
}

/**
 * Fill unset limits from recorded ones or defaults; reject explicit conflicts.
 *
 * @throws MemoryOperationError `reflective.config_conflict` when a requested limit differs from the
 *   recorded one.
 */
function resolveLimits(
  requested: readonly [number | null, number | null],
  recorded: Limits | null,
  stage: string,
): Limits {
  const [maxItems, recallLimit] = requested;
  if (
    recorded !== null &&
    ((maxItems !== null && maxItems !== recorded.maxItems) ||
      (recallLimit !== null && recallLimit !== recorded.recallLimit))
  ) {
    throw operationError(stage, "reflective.config_conflict", "configured limits differ from the recorded limits");
  }
  let base = recorded;
  if (base === null) {
    const values = current();
    base = { maxItems: values.reflectiveMaxItems, recallLimit: values.reflectiveRecallLimit };
  }
  return Object.freeze({ maxItems: maxItems ?? base.maxItems, recallLimit: recallLimit ?? base.recallLimit });
}

// Local copy of memory/contracts.ts's requireString, which that module keeps private.
/**
 * Require one exact string with a canonical UTF-8 representation.
 *
 * @throws TypeError when `value` isn't a string.
 * @throws RangeError when it's empty and `nonempty` is set, or holds a lone surrogate.
 */
function jsonString(value: unknown, name: string, nonempty = false): string {
  if (typeof value !== "string") throw new TypeError(`${name} must be a string`);
  if (nonempty && !value) throw new RangeError(`${name} must be non-empty`);
  if (!value.isWellFormed()) throw new RangeError("Unicode surrogate is not valid canonical JSON");
  return value;
}

/**
 * Require one exact non-empty string (Python's `_nonempty_string`).
 *
 * @throws TypeError when `value` isn't a string.
 * @throws RangeError when it's empty or holds a lone surrogate.
 */
function nonemptyString(value: unknown, name: string): string {
  return jsonString(value, name, true);
}

/**
 * Validate and copy one exact finite JSON object into frozen containers, at any depth.
 *
 * @throws TypeError when `value` isn't a plain object or holds a non-JSON value.
 * @throws RangeError for NaN, `±Infinity`, a lone surrogate or a cycle.
 */
function copyObject(value: unknown, name: string): JsonObject {
  // The value number rule reads `-0` as `0`; reflective JSON then refuses `±Infinity`, as Python's
  // `_copy_json` refuses non-finite floats. A walk over the copy, not recursion, so depth is free.
  const copy = freezeJsonObject(value, name, { limitNesting: false, valueNumbers: true });
  const stack: [JsonValue, string][] = [[copy, name]];
  while (stack.length > 0) {
    const [item, path] = stack.pop()!;
    if (typeof item === "number" && !Number.isFinite(item)) throw new RangeError(`${path} JSON number must be finite`);
    // Children pushed last to first, so they're visited depth-first in order.
    if (Array.isArray(item)) {
      for (let i = item.length - 1; i >= 0; i--) stack.push([item[i]!, `${path}[${i}]`]);
    } else if (isObject(item)) {
      const keys = Object.keys(item);
      for (let i = keys.length - 1; i >= 0; i--) stack.push([item[keys[i]!]!, `${path}.${keys[i]!}`]);
    }
  }
  return copy;
}

/** Return stable source metadata for one proposed lesson mutation, frozen. */
function sourceJson(runId: string): JsonObject {
  return Object.freeze({ run_id: runId });
}

/**
 * Decode one exact canonical lesson source.
 *
 * @throws TypeError or RangeError when `value` isn't exactly `{run_id}` with a non-empty id.
 */
function sourceFromJson(value: unknown, name: string): JsonObject {
  const source = copyObject(value, name);
  if (!hasFields(source, ["run_id"])) throw new RangeError(`${name} has invalid fields`);
  const canonical = sourceJson(nonemptyString(source.run_id, `${name}.run_id`));
  if (logJson(source) !== logJson(canonical)) throw new RangeError(`${name} must be canonical`);
  return canonical;
}

/** Encode canonical lessons and identity without host model settings, deep-frozen, ints as `bigint`s. */
function stateJson(namespace: string, epoch: number, revision: number, state: State): JsonObject {
  return Object.freeze({
    config: Object.freeze({
      max_items: BigInt(state.limits.maxItems),
      recall_limit: BigInt(state.limits.recallLimit),
    }),
    epoch: BigInt(epoch),
    // Lesson sources are frozen, so the encoding shares them (Python copies each).
    lessons: Object.freeze(
      state.lessons.map((lesson) => Object.freeze({ id: lesson.lessonId, source: lesson.source, text: lesson.text })),
    ),
    namespace,
    revision: BigInt(revision),
  });
}

/** Encode one canonical session snapshot with its pinned (frozen) model settings, deep-frozen. */
function snapshotJson(
  namespace: string,
  epoch: number,
  revision: number,
  state: State,
  llmConfig: JsonObject,
): JsonObject {
  return Object.freeze({
    ...stateJson(namespace, epoch, revision, state),
    format: SNAPSHOT_FORMAT,
    llm_config: llmConfig,
    version: SNAPSHOT_VERSION,
  });
}

/** Return a stable final snapshot-validation failure, with `cause` when given. */
function snapshotError(message: string, cause?: unknown): MemoryOperationError {
  return operationError("open_run", "reflective.invalid_snapshot", message, cause === undefined ? {} : { cause });
}

/**
 * Require one recorded positive integer limit: a JSON int is a `bigint`; a `number` is Python's
 * float, which `_positive_int` refuses.
 *
 * @throws TypeError when `value` isn't a `bigint`.
 * @throws RangeError when it isn't positive or is beyond 2^53 − 1.
 */
function intLimit(value: JsonValue | undefined, name: string): number {
  if (typeof value !== "bigint") throw new TypeError(`${name} must be an integer`);
  return positiveInt(value, name);
}

/**
 * Decode the shared semantic state without requiring a model configuration.
 *
 * @returns The namespace, epoch, revision and state.
 * @throws TypeError or RangeError when `value` isn't a valid state.
 */
function parseState(value: unknown): [string, number, number, State] {
  const data = copyObject(value, "state");
  if (!hasFields(data, ["config", "epoch", "lessons", "namespace", "revision"])) {
    throw new RangeError("state has invalid fields");
  }
  const namespace = nonemptyString(data.namespace, "snapshot.namespace");
  const epoch = data.epoch;
  const revision = data.revision;
  if (typeof epoch !== "bigint" || epoch < 0n) throw new RangeError("snapshot.epoch must be a non-negative integer");
  if (typeof revision !== "bigint" || revision < 0n) {
    throw new RangeError("snapshot.revision must be a non-negative integer");
  }
  // Counters are `number`s, so one past 2^53 − 1 would lose its increment.
  if (epoch > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError(`snapshot.epoch must be at most ${Number.MAX_SAFE_INTEGER}`);
  }
  if (revision > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError(`snapshot.revision must be at most ${Number.MAX_SAFE_INTEGER}`);
  }

  const config = copyObject(data.config, "snapshot.config");
  if (!hasFields(config, ["max_items", "recall_limit"])) throw new RangeError("snapshot.config has invalid fields");
  const limits: Limits = Object.freeze({
    maxItems: intLimit(config.max_items, "snapshot.config.max_items"),
    recallLimit: intLimit(config.recall_limit, "snapshot.config.recall_limit"),
  });
  const lessonsRaw = data.lessons;
  if (!Array.isArray(lessonsRaw)) throw new TypeError("snapshot.lessons must be an array");
  const lessons: Lesson[] = [];
  const lessonIds = new Set<string>();
  for (const [lessonIndex, rawLesson] of (lessonsRaw as readonly JsonValue[]).entries()) {
    const lesson = copyObject(rawLesson, `snapshot.lessons[${lessonIndex}]`);
    if (!hasFields(lesson, ["id", "source", "text"])) throw new RangeError("snapshot lesson has invalid fields");
    const lessonId = nonemptyString(lesson.id, "snapshot lesson id");
    const text = nonemptyString(lesson.text, "snapshot lesson text");
    if (!text.trim()) throw new RangeError("snapshot lesson text must not be blank");
    if (lessonIds.has(lessonId)) throw new RangeError("snapshot lesson ids must be unique");
    lessonIds.add(lessonId);
    const source = sourceFromJson(lesson.source, "snapshot lesson source");
    lessons.push(Object.freeze({ lessonId, text, source }));
  }
  if (lessons.length > limits.maxItems) throw new RangeError("snapshot exceeds max_items");
  return [namespace, Number(epoch), Number(revision), Object.freeze({ limits, lessons: Object.freeze(lessons) })];
}

/**
 * Strictly decode one canonical reflective-memory snapshot.
 *
 * @returns The namespace, epoch, revision, state and the pinned model configuration (frozen).
 * @throws MemoryOperationError `reflective.invalid_snapshot` for anything but the canonical form.
 * @throws AuthError when `llm_config` carries the host key marker without a key, as Python's
 *   `RuntimeError` escapes.
 */
function parseSnapshot(value: unknown): [string, number, number, State, JsonObject] {
  try {
    const data = copyObject(value, "snapshot");
    if (
      !hasFields(data, ["config", "epoch", "format", "lessons", "llm_config", "namespace", "revision", "version"])
    ) {
      throw new RangeError("snapshot has invalid fields");
    }
    if (data.format !== SNAPSHOT_FORMAT) throw new RangeError("snapshot format is unsupported");
    if (data.version !== SNAPSHOT_VERSION) throw new RangeError("snapshot version is unsupported");
    const llmConfig = copyObject(data.llm_config, "snapshot.llm_config");
    const [validated, connection] = splitModelConfig(llmConfig);
    const complete: Record<string, unknown> = validated.dump();
    if (connection !== null) complete._connection = connection;
    if (logJson(llmConfig) !== logJson(complete)) {
      throw new RangeError("snapshot.llm_config must be complete and canonical");
    }
    const provider = nonemptyString(complete.provider, "snapshot.llm_config.provider");
    const model = nonemptyString(complete.model, "snapshot.llm_config.model");
    if (!provider.trim() || !model.trim()) {
      throw new RangeError("snapshot.llm_config must pin nonblank provider and model");
    }
    const [namespace, epoch, revision, state] = parseState({
      config: data.config,
      epoch: data.epoch,
      lessons: data.lessons,
      namespace: data.namespace,
      revision: data.revision,
    });
    // The frozen copy has the complete configuration's exact text, so it stands in for it.
    return [namespace, epoch, revision, state, llmConfig];
  } catch (error) {
    if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
    throw snapshotError("recorded memory snapshot is invalid", error);
  }
}

/** One decoded private storage inventory: the state, the commit ledger and the flags. */
type Inventory = [
  state: [string, number, number, State],
  appliedCommits: ReadonlyMap<string, string>,
  everOpened: boolean,
  everReset: boolean,
  everCommitted: boolean,
];

/**
 * Validate the complete private inventory before restoring any RAM state.
 *
 * @throws MemoryOperationError `reflective.invalid_storage` for any defect, with the defect as its
 *   cause.
 */
function parseInventory(value: unknown): Inventory {
  try {
    const data = copyObject(value, "inventory");
    if (!hasFields(data, ["version", "state", "applied_commits", "ever_opened", "ever_reset", "ever_committed"])) {
      throw new RangeError("invalid inventory fields");
    }
    if (data.version !== 1n) throw new RangeError("unsupported inventory version");
    const state = parseState(data.state);
    const [, epoch, revision, instance] = state;
    const opened = data.ever_opened;
    const reset = data.ever_reset;
    const committed = data.ever_committed;
    if (typeof opened !== "boolean" || typeof reset !== "boolean" || typeof committed !== "boolean") {
      throw new RangeError("invalid inventory flags");
    }
    const commits = new Map<string, string>();
    for (const [commitId, digest] of Object.entries(copyObject(data.applied_commits, "applied_commits"))) {
      nonemptyString(commitId, "commit_id");
      const text = nonemptyString(digest, "digest");
      if (!/^[0-9a-f]{64}$/.test(text)) throw new RangeError("invalid commit digest");
      commits.set(commitId, text);
    }
    const hasLessons = instance.lessons.length > 0;
    if (
      (!opened && (hasLessons || revision !== 0 || committed || commits.size > 0)) ||
      (!committed && (revision !== 0 || commits.size > 0 || hasLessons)) ||
      (reset && epoch === 0) ||
      (!opened && !reset && epoch !== 0) ||
      (committed && revision === 0 && !reset && !hasLessons)
    ) {
      throw new RangeError("inconsistent inventory flags");
    }
    return [state, commits, opened, reset, committed];
  } catch (error) {
    if (!(error instanceof TypeError || error instanceof RangeError || error instanceof MemoryOperationError)) {
      throw error;
    }
    throw operationError("storage", "reflective.invalid_storage", "stored reflective state is invalid", {
      cause: error,
    });
  }
}

/**
 * Learn atomic lessons through reflection, optionally in a local SQLite file.
 *
 * @example
 * Pass `storage: "memory.sqlite"` to retain lessons across restarts, and call `close()` in
 * `finally` to release the single file owner. The file is opened at the first operation, not at
 * construction. Idle run handles do not need to close before the backend.
 *
 * ```ts
 * const memory = new ReflectiveMemory({ storage: "memory.sqlite" });
 * try {
 *   const result = await runFlow(flow, inputs, { memory: { reflective: memory }, learnable: true });
 *   await result.backward();
 * } finally {
 *   memory.close();
 * }
 * ```
 */
export class ReflectiveMemory implements Memory {
  // TS-private, not `#`, so the session and tests can read them, as Python's read `_state`.
  private readonly requested: readonly [number | null, number | null];
  private readonly llmConfig: JsonObject;
  private readonly env: Env;
  private readonly createModel: CreateModel;
  private readonly onRetry: OnRetry | null;
  private storage: SQLiteStorage | null = null;
  private readonly storagePath: string | null;
  private readonly createParent: boolean;
  private closed = false;
  private activeOperations = 0;
  private namespace: string;
  private epoch = 0;
  private revision = 0;
  private state: State;
  private appliedCommits: ReadonlyMap<string, string> = new Map();
  private everOpened = false;
  private everReset = false;
  private everCommitted = false;

  /**
   * Create or reopen one reflective-memory namespace. An option that is `undefined` or `null` is
   * unset.
   *
   * @param options.llmConfig Public model settings for new sessions, filled from ambient defaults
   *   when absent. Recorded sessions retain their exact pinned model configuration. `api_key` is
   *   rejected, so the learner always authenticates with environment or saved-connection
   *   credentials; `organization`, `project` and `headers`, which go with a key, are rejected too.
   *   `base_url` is accepted only beside `provider: azure`, as the resource endpoint.
   * @param options.maxItems Maximum retained lessons. Unset adopts the stored or snapshot value,
   *   else `YIAN_REFLECTIVE_MAX_ITEMS` (100).
   * @param options.recallLimit Maximum lessons returned by contextual recall. Unset adopts the
   *   stored or snapshot value, else `YIAN_REFLECTIVE_RECALL_LIMIT` (20).
   * @param options.storage Optional local SQLite file. Nothing is opened until the first
   *   operation; this backend then owns the file until close. Unset keeps state process-local.
   * @param options.createParent Create the storage file's parent directory at first open.
   * @param options.env The environment model defaults and credentials are read from; default
   *   `processEnv`.
   * @param options.createModel Build the learner's model from the config Python passes to
   *   `model_from_config`; default `chatModelFromConfig(config, {env})`.
   * @param options.onRetry Called before the learner's model call is retried; unset publishes
   *   nothing.
   * @throws TypeError when a limit or `llmConfig` has an invalid type, or `createParent` isn't a
   *   boolean.
   * @throws RangeError when a limit or the model settings are invalid.
   * @throws MemoryOperationError `storage.path` when `storage` isn't a local file path.
   */
  constructor(
    options: {
      llmConfig?: JsonObject | null;
      maxItems?: number | bigint | null;
      recallLimit?: number | bigint | null;
      storage?: string | null;
      createParent?: boolean | null;
      env?: Env | null;
      createModel?: CreateModel | null;
      onRetry?: OnRetry | null;
    } = {},
  ) {
    const maxItems = options.maxItems ?? null;
    const recallLimit = options.recallLimit ?? null;
    this.requested = [
      maxItems === null ? null : positiveInt(maxItems, "max_items"),
      recallLimit === null ? null : positiveInt(recallLimit, "recall_limit"),
    ];
    const llmConfig = options.llmConfig ?? null;
    const rawConfig = llmConfig === null ? {} : freezeJsonObject(llmConfig, "llm_config");
    for (const field of ["organization", "project", "headers"]) {
      // Checked before validation, which would otherwise ask for an api_key that this backend
      // refuses too.
      if (Object.hasOwn(rawConfig, field)) {
        throw new RangeError(
          `llm_config.${field} is not supported by reflective memory, which records its model configuration`,
        );
      }
    }
    const validated = new LLMConfig(fromHost(rawConfig, "llm_config"));
    if (validated.apiKey !== null) {
      // Snapshots record this config; a key would be dropped or persisted.
      throw new RangeError(
        "llm_config.api_key is not supported by reflective memory, which records its model configuration",
      );
    }
    if (validated.model !== null && !validated.model.trim()) throw new RangeError("llm_config.model must be non-empty");
    this.llmConfig = copyObject(
      Object.fromEntries(Object.entries(validated.dump()).filter(([, value]) => value !== null)),
      "llm_config",
    );
    this.namespace = crypto.randomUUID().replaceAll("-", "");
    this.state = Object.freeze({ limits: resolveLimits(this.requested, null, "storage"), lessons: Object.freeze([]) });
    const createParent = options.createParent ?? false;
    if (typeof createParent !== "boolean") throw new TypeError("create_parent must be a bool");
    const storage = options.storage ?? null;
    this.storagePath = storage === null ? null : checkStoragePath(storage);
    this.createParent = createParent;
    const env = options.env ?? processEnv;
    this.env = env;
    this.createModel = options.createModel ?? ((config) => chatModelFromConfig(config, { env }));
    this.onRetry = options.onRetry ?? null;
  }

  /**
   * Open and load durable state once. On failure the storage is closed and forgotten, so a later
   * operation retries the open, and the primary failure is rethrown unchanged.
   *
   * @throws MemoryOperationError The file can't be opened, read or initialized;
   *   `reflective.invalid_storage` for invalid stored state; `reflective.config_conflict` when a
   *   configured limit differs from the stored one.
   */
  private openStorage(): void {
    if (this.storage !== null || this.storagePath === null) return;
    this.storage = new SQLiteStorage(this.storagePath, POLICY_NAME, { createParent: this.createParent });
    try {
      const saved = this.storage.load();
      if (saved === null) {
        this.saveState(stateJson(this.namespace, this.epoch, this.revision, this.state));
      } else {
        const [state, commits, opened, reset, committed] = parseInventory(saved);
        resolveLimits(this.requested, state[3].limits, "storage");
        [this.namespace, this.epoch, this.revision, this.state] = state;
        this.appliedCommits = commits;
        this.everOpened = opened;
        this.everReset = reset;
        this.everCommitted = committed;
      }
    } catch (primary) {
      try {
        this.storage.close();
      } catch (cleanup) {
        addNote(primary, () => `ReflectiveMemory cleanup failed: ${typeName(cleanup)}`);
      }
      // Forget the failed handle so a later operation may retry the open.
      this.storage = null;
      throw primary;
    }
  }

  /**
   * Enter one operation, independently of run-handle lifetime (Python's `_operation()` entry):
   * open storage at the first operation and reject an unusable handle.
   *
   * @throws MemoryOperationError `reflective.closed` after `close()`; the storage can't be opened,
   *   or is closed or poisoned.
   */
  private begin(): void {
    if (this.closed) throw operationError("session", "reflective.closed", "memory backend is closed");
    this.openStorage();
    this.storage?.ensureOpen();
    this.activeOperations += 1;
  }

  /** Leave the operation `begin()` entered. */
  private end(): void {
    this.activeOperations -= 1;
  }

  /** Run `fn` as one operation. */
  private operation<T>(fn: () => T): T {
    this.begin();
    try {
      return fn();
    } finally {
      this.end();
    }
  }

  /** Validate and persist a candidate before changing the live backend; nothing without storage. */
  private saveState(state: JsonObject, changes: StateChanges = {}): void {
    if (this.storage === null) return;
    const inventory = {
      version: 1n,
      state,
      applied_commits: Object.fromEntries(changes.appliedCommits ?? this.appliedCommits),
      ever_opened: changes.everOpened ?? this.everOpened,
      ever_reset: changes.everReset ?? this.everReset,
      ever_committed: changes.everCommitted ?? this.everCommitted,
    };
    parseInventory(inventory);
    this.storage.save(inventory);
  }

  /**
   * Release storage and invalidate idle handles without waiting for work; repeated closes are inert.
   *
   * @throws MemoryOperationError `reflective.busy` (retryable) while an operation is active,
   *   including a re-entrant close from an awaited callback; a storage cleanup failure.
   */
  close(): void {
    if (this.closed) return;
    if (this.activeOperations > 0) {
      throw operationError("close", "reflective.busy", "memory backend is active", { retryable: true });
    }
    this.storage?.close();
    this.closed = true;
  }

  /**
   * Build the default lexical query from the visible prompt and inputs:
   * ``Task: <prompt>\nInputs: <inputs as log JSON>``. A subclass may override it, also with an async
   * method.
   *
   * @throws TypeError when `context` isn't exactly a `RecallContext`.
   * @throws MemoryOperationError `reflective.closed` after `close()`; durable state can't be opened.
   */
  rewriteQuery(context: RecallContext): string | Promise<string> {
    return this.operation(() => {
      if (!isExactly(context, RecallContext)) throw new TypeError("context must be a RecallContext");
      return `Task: ${context.prompt}\nInputs: ${logJson(context.inputs)}`;
    });
  }

  /**
   * Resolve ambient defaults exactly once for one fresh session.
   *
   * @returns The complete pinned configuration, frozen.
   * @throws MemoryOperationError `reflective.config` when no valid provider and model resolve, with
   *   the failure as its cause.
   */
  private effectiveLlmConfig(): JsonObject {
    try {
      const resolved = resolveModelConfig(this.llmConfig, { env: this.env });
      const [validated, connection] = splitModelConfig(resolved);
      const complete: Record<string, unknown> = validated.dump();
      if (connection !== null) complete._connection = connection;
      nonemptyString(complete.provider, "llm_config.provider");
      const model = nonemptyString(complete.model, "llm_config.model");
      if (!model.trim()) throw new RangeError("llm_config.model must not be blank");
      return copyObject(complete, "llm_config");
    } catch (error) {
      throw operationError("open_run", "reflective.config", "reflective model configuration is invalid", {
        cause: error,
      });
    }
  }

  /**
   * Open one immutable-read session, optionally from a recorded snapshot.
   *
   * @param options.snapshot Recorded public snapshot to replay, including its pinned model
   *   settings. `null` or unset takes current lessons and resolves current host model settings.
   * @returns An isolated run session.
   * @throws MemoryOperationError The snapshot or model configuration is invalid, the backend is
   *   closed, or durable state can't be opened or saved.
   */
  openRun(options: { snapshot?: JsonObject | null } = {}): ReflectiveSession {
    return this.operation(() => {
      const snapshot = options.snapshot ?? null;
      const parsed = snapshot === null ? null : parseSnapshot(snapshot);
      let adopt = false;
      // The state is frozen, so a session pins it without Python's defensive copy.
      const [namespace, epoch, revision, state, llmConfig] = parsed ?? [
        this.namespace,
        this.epoch,
        this.revision,
        this.state,
        this.effectiveLlmConfig(),
      ];
      if (parsed !== null) {
        resolveLimits(this.requested, state.limits, "open_run");
        if (namespace !== this.namespace) {
          const pristine =
            !this.everOpened &&
            !this.everReset &&
            !this.everCommitted &&
            this.state.lessons.length === 0 &&
            this.epoch === 0 &&
            this.revision === 0;
          if (!pristine) throw snapshotError("snapshot namespace cannot replace a non-pristine memory");
          adopt = true;
        } else {
          if (epoch !== this.epoch) throw snapshotError("snapshot generation is incompatible after reset");
          if (revision > this.revision) throw snapshotError("snapshot revision is ahead of the memory state");
          if (
            revision === this.revision &&
            logJson(stateJson(namespace, epoch, revision, state)) !==
              logJson(stateJson(namespace, epoch, revision, this.state))
          ) {
            throw snapshotError("snapshot state does not match the current generation");
          }
        }
      }
      if (adopt || !this.everOpened) {
        const committed = adopt ? revision > 0 || state.lessons.length > 0 : this.everCommitted;
        const candidate = adopt
          ? stateJson(namespace, epoch, revision, state)
          : stateJson(this.namespace, this.epoch, this.revision, this.state);
        this.saveState(candidate, { everOpened: true, everCommitted: committed });
        if (adopt) {
          this.namespace = namespace;
          this.epoch = epoch;
          this.revision = revision;
          this.state = state;
          this.everCommitted = committed;
        }
        this.everOpened = true;
      }
      return new ReflectiveSession(this, namespace, epoch, revision, state, llmConfig);
    });
  }

  /**
   * Clear lessons, start a new generation, and retain commit idempotency.
   *
   * @throws MemoryOperationError The backend is closed or durable state can't be opened or saved.
   *   A failed save leaves live lessons unchanged.
   */
  reset(): void {
    this.operation(() => {
      const cleared: State = Object.freeze({ limits: this.state.limits, lessons: Object.freeze([]) });
      this.saveState(stateJson(this.namespace, this.epoch + 1, 0, cleared), { everReset: true });
      this.state = cleared;
      this.epoch += 1;
      this.revision = 0;
      this.everReset = true;
    });
  }
}

/**
 * One run-scoped view over a pinned reflective-memory generation (Python's `_ReflectiveSession`).
 *
 * It reads and publishes its backend's private state through element access (`memory["state"]`),
 * TypeScript's escape for private members, as Python's session reads `self._memory._state`.
 */
export class ReflectiveSession implements MemorySession {
  private readonly memory: ReflectiveMemory;
  private readonly namespace: string;
  private readonly epoch: number;
  private readonly revision: number;
  private readonly state: State;
  private readonly llmConfig: JsonObject;
  private closed = false;

  /** @internal Pin one generation and its model configuration; `ReflectiveMemory.openRun` builds sessions. */
  constructor(
    memory: ReflectiveMemory,
    namespace: string,
    epoch: number,
    revision: number,
    state: State,
    llmConfig: JsonObject,
  ) {
    this.memory = memory;
    this.namespace = namespace;
    this.epoch = epoch;
    this.revision = revision;
    this.state = state;
    this.llmConfig = llmConfig;
  }

  /**
   * Reject every operation once the run boundary has closed.
   *
   * @throws MemoryOperationError `reflective.session_closed` after `close()`.
   */
  private ensureOpen(): void {
    if (this.closed) throw operationError("session", "reflective.session_closed", "memory session is closed");
  }

  /**
   * The public durable source-neutral state pinned for this run, deep-frozen.
   *
   * @throws MemoryOperationError The session or backend is closed.
   */
  get snapshot(): JsonObject {
    return this.memory["operation"](() => {
      this.ensureOpen();
      return snapshotJson(this.namespace, this.epoch, this.revision, this.state, this.llmConfig);
    });
  }

  /** Idempotently close this process-local session, without closing its backend. */
  close(): void {
    this.closed = true;
  }

  /**
   * Require this proposal session to remain the current generation.
   *
   * @throws MemoryOperationError `reflective.stale_generation` otherwise.
   */
  private assertCurrent(stage: string): void {
    if (
      this.namespace !== this.memory["namespace"] ||
      this.epoch !== this.memory["epoch"] ||
      this.revision !== this.memory["revision"]
    ) {
      throw operationError(stage, "reflective.stale_generation", "memory session is stale");
    }
  }

  /** Return all pinned lessons or a deterministic lexical ranking. */
  async recall(context: RecallContext | null = null): Promise<string> {
    throw new Error("not implemented");
  }

  /** Propose, but do not apply, one exact array of lesson mutations. */
  async learn(context: LearnContext): Promise<MemoryMutation[]> {
    throw new Error("not implemented");
  }

  /** Atomically apply one exact batch, with ledger lookup before generation. */
  commit(commitId: string, mutations: readonly MemoryMutation[]): void {
    throw new Error("not implemented");
  }
}
