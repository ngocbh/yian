/**
 * Episodic memory with snapshot-isolated runs and optional SQLite persistence.
 *
 * The backend owns retained episode history behind the source-neutral memory interfaces. This
 * module establishes the session, instance, snapshot and commit boundaries without depending on
 * the workflow engine or providers. Optional durability uses the private single-owner storage
 * adapter below this backend.
 *
 * Knows about: `memory/contracts`, `memory/storage/sqlite`, `defaults`, `typesys/json`.
 * Never imports: the `memory` facade (Python's import cycle), nodes, runtime, llm.
 */

// The default import keeps calls patchable by test spies.
import crypto from "node:crypto";

import { current } from "../defaults.ts";
import { logJson } from "../typesys/json.ts";
import {
  freezeJsonObject,
  type JsonObject,
  type JsonValue,
  LearnContext,
  type Memory,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryOperationError,
  MemorySection,
  type MemorySession,
  ModelTextInteraction,
  ModelValueInteraction,
  RecallContext,
  ToolCallInteraction,
  ToolResultInteraction,
  type VisibleInteraction,
} from "./contracts.ts";
import { checkStoragePath, SQLiteStorage } from "./storage/sqlite.ts";

const POLICY_NAME = "episodic";
const SNAPSHOT_FORMAT = "yian.episodic_memory";
const SNAPSHOT_VERSION = 2n;

const EPISODE_STATUSES: readonly unknown[] = Object.values(MemoryEpisodeStatus);
const OCCURRENCE_KINDS: readonly unknown[] = Object.values(MemoryOccurrenceKind);
const EPISODE_FIELDS: readonly string[] = [
  "base_system_prompt",
  "execution_seq",
  "failure",
  "inputs",
  "interactions",
  "memory_sections",
  "occurrence_path",
  "output",
  "owner_static_path",
  "prompt",
  "run_id",
  "status",
];

/** Effective retention and recall limits of one episodic instance. Frozen. */
interface Limits {
  readonly maxEpisodes: number;
  readonly recallLimit: number;
}

/** The limits and ordered retained entries of one episodic instance. Frozen. */
interface State {
  readonly limits: Limits;
  readonly episodes: readonly JsonObject[];
}

/** The inventory flags and tombstones a publication replaces. */
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

/** Build one stable nonretryable backend failure, with `cause` (Python's `raise ... from`) when given. */
function operationError(
  stage: string,
  code: string,
  message: string,
  options: { cause?: unknown } = {},
): MemoryOperationError {
  const error = new MemoryOperationError(stage, code, message, false);
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
 * @throws MemoryOperationError `episodic.config_conflict` when a requested limit differs from the
 *   recorded one.
 */
function resolveLimits(
  requested: readonly [number | null, number | null],
  recorded: Limits | null,
  stage: string,
): Limits {
  const [maxEpisodes, recallLimit] = requested;
  if (
    recorded !== null &&
    ((maxEpisodes !== null && maxEpisodes !== recorded.maxEpisodes) ||
      (recallLimit !== null && recallLimit !== recorded.recallLimit))
  ) {
    throw operationError(stage, "episodic.config_conflict", "configured limits differ from the recorded limits");
  }
  let base = recorded;
  if (base === null) {
    const values = current();
    base = { maxEpisodes: values.episodicMaxEpisodes, recallLimit: values.episodicRecallLimit };
  }
  return Object.freeze({
    maxEpisodes: maxEpisodes ?? base.maxEpisodes,
    recallLimit: recallLimit ?? base.recallLimit,
  });
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
 * Validate and copy one exact finite JSON object into frozen containers, at any depth.
 *
 * @throws TypeError when `value` isn't a plain object or holds a non-JSON value.
 * @throws RangeError for NaN, `±Infinity`, a lone surrogate or a cycle.
 */
function copyObject(value: unknown, name: string): JsonObject {
  // The value number rule reads `-0` as `0`; episodic JSON then refuses `±Infinity`, as Python's
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

/** Split text into case-insensitive alphanumeric retrieval terms. */
function lexicalTerms(value: string): Set<string> {
  // Upper then lower case is the design's stand-in for Python's `casefold`; `\p{L}\p{N}` is
  // `isalnum`.
  return new Set(
    value
      .toUpperCase()
      .toLowerCase()
      .split(/[^\p{L}\p{N}]+/u)
      .filter((term) => term !== ""),
  );
}

/** Encode one root-relative occurrence path. */
function occurrencePathJson(path: readonly MemoryOccurrenceSegment[]): JsonValue[] {
  return path.map((segment) => ({
    index: BigInt(segment.index),
    kind: segment.kind,
    static_path: [...segment.staticPath],
  }));
}

/**
 * Decode one exact root-relative occurrence path.
 *
 * @throws TypeError or RangeError when the path isn't valid.
 */
function occurrencePathFromJson(value: JsonValue | undefined, name: string): MemoryOccurrenceSegment[] {
  if (!Array.isArray(value)) throw new TypeError(`${name} must be an array`);
  return (value as readonly JsonValue[]).map((raw, index) => {
    const itemName = `${name}[${index}]`;
    const item = copyObject(raw, itemName);
    if (!hasFields(item, ["index", "kind", "static_path"])) throw new RangeError(`${itemName} has invalid fields`);
    const kind = jsonString(item.kind, `${itemName}.kind`, true);
    if (!OCCURRENCE_KINDS.includes(kind)) throw new RangeError(`${itemName}.kind is invalid`);
    const staticRaw = item.static_path;
    if (!Array.isArray(staticRaw) || staticRaw.length === 0) {
      throw new RangeError(`${itemName}.static_path must be a non-empty array`);
    }
    const staticPath = (staticRaw as readonly JsonValue[]).map((part, partIndex) =>
      jsonString(part, `${itemName}.static_path[${partIndex}]`, true),
    );
    const sequence = item.index;
    if (typeof sequence !== "bigint" || sequence < 0n) throw new RangeError(`${itemName}.index is invalid`);
    return new MemoryOccurrenceSegment(kind as MemoryOccurrenceKind, staticPath, Number(sequence));
  });
}

/**
 * Encode one closed visible-interaction variant.
 *
 * @throws TypeError when `interaction` isn't exactly one of the four variants.
 */
function interactionJson(interaction: unknown): JsonObject {
  if (isExactly(interaction, ModelTextInteraction)) return { kind: "model_text", text: interaction.text };
  if (isExactly(interaction, ModelValueInteraction)) return { kind: "model_value", value: interaction.value };
  if (isExactly(interaction, ToolCallInteraction)) {
    return { arguments: interaction.arguments, kind: "tool_call", name: interaction.name };
  }
  if (isExactly(interaction, ToolResultInteraction)) {
    return { kind: "tool_result", name: interaction.name, text: interaction.text };
  }
  throw new TypeError("episode contains an invalid interaction");
}

/**
 * Decode one closed visible-interaction variant.
 *
 * @throws TypeError or RangeError when `value` isn't a valid variant.
 */
function interactionFromJson(value: JsonValue, name: string): VisibleInteraction {
  const data = copyObject(value, name);
  const kind = data.kind;
  if (kind === "model_text" && hasFields(data, ["kind", "text"])) {
    return new ModelTextInteraction(jsonString(data.text, `${name}.text`));
  }
  if (kind === "model_value" && hasFields(data, ["kind", "value"])) return new ModelValueInteraction(data.value);
  if (kind === "tool_call" && hasFields(data, ["arguments", "kind", "name"])) {
    return new ToolCallInteraction(
      jsonString(data.name, `${name}.name`, true),
      copyObject(data.arguments, `${name}.arguments`),
    );
  }
  if (kind === "tool_result" && hasFields(data, ["kind", "name", "text"])) {
    return new ToolResultInteraction(
      jsonString(data.name, `${name}.name`, true),
      jsonString(data.text, `${name}.text`),
    );
  }
  throw new RangeError(`${name} has an invalid interaction variant`);
}

/** Encode every source-neutral field of one accepted episode. */
function episodeJson(episode: MemoryEpisode): JsonObject {
  return {
    base_system_prompt: episode.baseSystemPrompt,
    execution_seq: episode.executionSeq === null ? null : BigInt(episode.executionSeq),
    failure: episode.failure,
    inputs: episode.inputs,
    interactions: episode.interactions.map(interactionJson),
    memory_sections: episode.memorySections.map((section) => ({ memory: section.memory, text: section.text })),
    occurrence_path: occurrencePathJson(episode.occurrencePath),
    output: episode.output,
    owner_static_path: [...episode.ownerStaticPath],
    prompt: episode.prompt,
    run_id: episode.runId,
    status: episode.status,
  };
}

/**
 * Decode one exact complete episode through its public carrier.
 *
 * @throws TypeError or RangeError when `value` isn't a valid episode.
 */
function episodeFromJson(value: JsonValue | undefined, name: string): MemoryEpisode {
  const data = copyObject(value, name);
  if (!hasFields(data, EPISODE_FIELDS)) throw new RangeError(`${name} has invalid fields`);
  const ownerRaw = data.owner_static_path;
  if (!Array.isArray(ownerRaw) || ownerRaw.length === 0) {
    throw new RangeError(`${name}.owner_static_path must be a non-empty array`);
  }
  const ownerPath = (ownerRaw as readonly JsonValue[]).map((part, index) =>
    jsonString(part, `${name}.owner_static_path[${index}]`, true),
  );
  const status = jsonString(data.status, `${name}.status`, true);
  if (!EPISODE_STATUSES.includes(status)) throw new RangeError(`${name}.status is invalid`);
  const sectionsRaw = data.memory_sections;
  const interactionsRaw = data.interactions;
  if (!Array.isArray(sectionsRaw) || !Array.isArray(interactionsRaw)) {
    throw new TypeError(`${name} collections must be arrays`);
  }
  const sections = (sectionsRaw as readonly JsonValue[]).map((raw, index) => {
    const sectionName = `${name}.memory_sections[${index}]`;
    const section = copyObject(raw, sectionName);
    if (!hasFields(section, ["memory", "text"])) throw new RangeError(`${sectionName} has invalid fields`);
    return new MemorySection(
      jsonString(section.memory, `${sectionName}.memory`, true),
      jsonString(section.text, `${sectionName}.text`),
    );
  });
  const inputs = data.inputs!;
  const failure = data.failure!;
  if (inputs !== null && !isObject(inputs)) throw new TypeError(`${name}.inputs must be an object or null`);
  if (failure !== null && !isObject(failure)) throw new TypeError(`${name}.failure must be an object or null`);
  const executionSeq = data.execution_seq;
  if (executionSeq !== null && (typeof executionSeq !== "bigint" || executionSeq < 0n)) {
    throw new RangeError(`${name}.execution_seq is invalid`);
  }
  const basePrompt =
    data.base_system_prompt === null ? null : jsonString(data.base_system_prompt, `${name}.base_system_prompt`);
  const prompt = data.prompt === null ? null : jsonString(data.prompt, `${name}.prompt`);
  return new MemoryEpisode(
    jsonString(data.run_id, `${name}.run_id`, true),
    occurrencePathFromJson(data.occurrence_path, `${name}.occurrence_path`),
    ownerPath,
    status as MemoryEpisodeStatus,
    inputs,
    {
      output: data.output,
      failure,
      interactions: (interactionsRaw as readonly JsonValue[]).map((item, index) =>
        interactionFromJson(item, `${name}.interactions[${index}]`),
      ),
      baseSystemPrompt: basePrompt,
      prompt,
      memorySections: sections,
      executionSeq: executionSeq === null ? null : Number(executionSeq),
    },
  );
}

/** Build one attributed canonical retained entry. */
function storedEntry(entryId: string, episode: MemoryEpisode): JsonObject {
  return { entry_id: entryId, episode: episodeJson(episode) };
}

/**
 * Validate one retained entry and return its canonical defensive copy.
 *
 * @throws TypeError or RangeError when `value` isn't a canonical entry.
 */
function validateEntry(value: JsonValue | undefined, name: string): JsonObject {
  const data = copyObject(value, name);
  if (!hasFields(data, ["entry_id", "episode"])) throw new RangeError(`${name} has invalid fields`);
  const entryId = jsonString(data.entry_id, `${name}.entry_id`, true);
  const episode = episodeFromJson(data.episode, `${name}.episode`);
  if (logJson(data) !== logJson(storedEntry(entryId, episode))) throw new RangeError(`${name} must be canonical`);
  // The frozen copy has the canonical entry's exact text, so it stands in for the entry.
  return data;
}

/** Encode one canonical episodic-memory snapshot, deep-frozen, ints as `bigint`s. */
function snapshotJson(namespace: string, epoch: number, revision: number, state: State): JsonObject {
  return Object.freeze({
    config: Object.freeze({
      max_episodes: BigInt(state.limits.maxEpisodes),
      recall_limit: BigInt(state.limits.recallLimit),
    }),
    // Retained entries are deep-frozen, so the snapshot shares them (Python copies each).
    episodes: state.episodes,
    epoch: BigInt(epoch),
    format: SNAPSHOT_FORMAT,
    namespace,
    revision: BigInt(revision),
    version: SNAPSHOT_VERSION,
  });
}

/** Return the stable snapshot-validation failure. */
function snapshotError(options: { cause?: unknown } = {}): MemoryOperationError {
  return operationError("open_run", "episodic.invalid_snapshot", "recorded memory snapshot is invalid", options);
}

/**
 * Strictly decode one canonical episodic-memory snapshot.
 *
 * @returns The namespace, epoch, revision and state.
 * @throws MemoryOperationError `episodic.invalid_snapshot` for anything but the canonical form.
 */
function parseSnapshot(value: unknown): [string, number, number, State] {
  try {
    const data = copyObject(value, "snapshot");
    if (!hasFields(data, ["config", "episodes", "epoch", "format", "namespace", "revision", "version"])) {
      throw new RangeError("snapshot has invalid fields");
    }
    if (data.format !== SNAPSHOT_FORMAT) throw new RangeError("snapshot format is unsupported");
    if (data.version !== SNAPSHOT_VERSION) throw new RangeError("snapshot version is unsupported");
    const namespace = jsonString(data.namespace, "snapshot.namespace", true);
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
    if (!hasFields(config, ["max_episodes", "recall_limit"])) {
      throw new RangeError("snapshot config has invalid fields");
    }
    const limits: Limits = Object.freeze({
      maxEpisodes: positiveInt(config.max_episodes, "max_episodes"),
      recallLimit: positiveInt(config.recall_limit, "recall_limit"),
    });
    const episodes = data.episodes;
    if (!Array.isArray(episodes)) throw new RangeError("snapshot episodes must be an array");
    if (episodes.length > limits.maxEpisodes) throw new RangeError("snapshot exceeds its retention limit");
    const decoded = (episodes as readonly JsonValue[]).map((item, index) =>
      validateEntry(item, `snapshot.episodes[${index}]`),
    );
    if (new Set(decoded.map((item) => item.entry_id)).size !== decoded.length) {
      throw new RangeError("snapshot episode IDs must be unique");
    }
    const state: State = Object.freeze({ limits, episodes: Object.freeze(decoded) });
    const result: [string, number, number, State] = [namespace, Number(epoch), Number(revision), state];
    if (logJson(data) !== logJson(snapshotJson(...result))) throw new RangeError("snapshot must be canonical");
    return result;
  } catch (error) {
    if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
    throw snapshotError({ cause: error });
  }
}

/** One decoded private storage inventory: the snapshot, the commit tombstones and the flags. */
type Inventory = [
  snapshot: [string, number, number, State],
  appliedCommits: ReadonlyMap<string, string>,
  everOpened: boolean,
  everReset: boolean,
  everCommitted: boolean,
];

/**
 * Validate complete private storage state before adopting any of it.
 *
 * @throws MemoryOperationError `episodic.invalid_storage` for any defect.
 */
function parseInventory(value: unknown): Inventory {
  try {
    const data = copyObject(value, "inventory");
    if (!hasFields(data, ["applied_commits", "ever_committed", "ever_opened", "ever_reset", "snapshot", "version"])) {
      throw new RangeError("invalid inventory fields");
    }
    if (data.version !== 1n) throw new RangeError("unsupported inventory version");
    const snapshot = parseSnapshot(data.snapshot);
    const [, epoch, revision, state] = snapshot;
    const opened = data.ever_opened;
    const reset = data.ever_reset;
    const committed = data.ever_committed;
    if (typeof opened !== "boolean" || typeof reset !== "boolean" || typeof committed !== "boolean") {
      throw new RangeError("invalid inventory flags");
    }
    const commits = new Map<string, string>();
    for (const [commitId, digest] of Object.entries(copyObject(data.applied_commits, "applied_commits"))) {
      jsonString(commitId, "commit_id", true);
      const text = jsonString(digest, "digest", true);
      if (!/^[0-9a-f]{64}$/.test(text)) throw new RangeError("invalid commit digest");
      commits.set(commitId, text);
    }
    const episodes = state.episodes.length > 0;
    if (
      (!opened && (episodes || revision !== 0 || committed || commits.size > 0)) ||
      (!committed && (revision !== 0 || commits.size > 0)) ||
      (reset && epoch === 0) ||
      (!opened && !reset && epoch !== 0) ||
      (committed && revision === 0 && !reset)
    ) {
      throw new RangeError("inconsistent inventory flags");
    }
    return [snapshot, commits, opened, reset, committed];
  } catch (error) {
    // Python's `raise ... from None`: the stored defect isn't chained.
    if (!(error instanceof TypeError || error instanceof RangeError || error instanceof MemoryOperationError)) {
      throw error;
    }
    throw operationError("storage", "episodic.invalid_storage", "stored episodic state is invalid");
  }
}

/**
 * Retain bounded canonical episodes, optionally in a local SQLite file.
 *
 * @example
 * Inject one instance under the name an authored `long_term_memory: episodic` declaration selects.
 * Forward execution only retains history; the explicit backward call commits it for later runs.
 *
 * ```ts
 * const memory = new EpisodicMemory({ maxEpisodes: 100, recallLimit: 10 });
 * const result = await runFlow(flow, inputs, { memory: { episodic: memory }, learnable: true });
 * await result.backward();
 * ```
 *
 * Pass `storage: "memory.sqlite"` to retain episodes across restarts, and call `close()` in
 * `finally` to release the single file owner. The file is opened at the first operation, not at
 * construction. Closing idle run handles is not required before closing this backend.
 */
export class EpisodicMemory implements Memory {
  // TS-private, not `#`, so tests can spy on and read them, as Python's tests patch `_save_state`.
  private readonly requested: readonly [number | null, number | null];
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
   * Create one isolated episodic-memory namespace. An option that is `undefined` or `null` is
   * unset.
   *
   * @param options.maxEpisodes Maximum retained episodes. Unset adopts the stored or snapshot
   *   value, else `YIAN_EPISODIC_MAX_EPISODES` (100).
   * @param options.recallLimit Maximum episodes returned by contextual recall. Unset adopts the
   *   stored or snapshot value, else `YIAN_EPISODIC_RECALL_LIMIT` (10).
   * @param options.storage Optional local SQLite file. Nothing is opened until the first
   *   operation; this backend then owns the file until close. Unset is process-local.
   * @param options.createParent Create the storage file's parent directory at first open.
   * @throws TypeError when a limit isn't an integer or `createParent` isn't a boolean.
   * @throws RangeError when a limit isn't positive or is beyond 2^53 − 1.
   * @throws MemoryOperationError `storage.path` when `storage` isn't a local file path.
   */
  constructor(
    options: {
      maxEpisodes?: number | bigint | null;
      recallLimit?: number | bigint | null;
      storage?: string | null;
      createParent?: boolean | null;
    } = {},
  ) {
    const maxEpisodes = options.maxEpisodes ?? null;
    const recallLimit = options.recallLimit ?? null;
    this.requested = [
      maxEpisodes === null ? null : positiveInt(maxEpisodes, "max_episodes"),
      recallLimit === null ? null : positiveInt(recallLimit, "recall_limit"),
    ];
    this.namespace = crypto.randomUUID().replaceAll("-", "");
    this.state = Object.freeze({ limits: resolveLimits(this.requested, null, "storage"), episodes: Object.freeze([]) });
    const createParent = options.createParent ?? false;
    if (typeof createParent !== "boolean") throw new TypeError("create_parent must be a bool");
    const storage = options.storage ?? null;
    this.storagePath = storage === null ? null : checkStoragePath(storage);
    this.createParent = createParent;
  }

  /**
   * Open and load durable state once. On failure the storage is closed and forgotten, so a later
   * operation retries the open, and the primary failure is rethrown unchanged.
   *
   * @throws MemoryOperationError The file can't be opened, read or initialized;
   *   `episodic.invalid_storage` for invalid stored state; `episodic.config_conflict` when a
   *   configured limit differs from the stored one.
   */
  private openStorage(): void {
    if (this.storage !== null || this.storagePath === null) return;
    this.storage = new SQLiteStorage(this.storagePath, POLICY_NAME, { createParent: this.createParent });
    try {
      const saved = this.storage.load();
      if (saved === null) {
        this.saveState(snapshotJson(this.namespace, this.epoch, this.revision, this.state));
      } else {
        const [snapshot, commits, opened, reset, committed] = parseInventory(saved);
        resolveLimits(this.requested, snapshot[3].limits, "storage");
        [this.namespace, this.epoch, this.revision, this.state] = snapshot;
        this.appliedCommits = commits;
        this.everOpened = opened;
        this.everReset = reset;
        this.everCommitted = committed;
      }
    } catch (primary) {
      try {
        this.storage.close();
      } catch (cleanup) {
        addNote(primary, () => `EpisodicMemory cleanup failed: ${typeName(cleanup)}`);
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
   * @throws MemoryOperationError `episodic.closed` after `close()`; the storage can't be opened, or
   *   is closed or poisoned.
   */
  private begin(): void {
    if (this.closed) throw operationError("session", "episodic.closed", "memory backend is closed");
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

  /** Validate and persist a complete candidate before live RAM publication; nothing without storage. */
  private saveState(snapshot: JsonObject, changes: StateChanges = {}): void {
    if (this.storage === null) return;
    const inventory = {
      version: 1n,
      snapshot,
      applied_commits: Object.fromEntries(changes.appliedCommits ?? this.appliedCommits),
      ever_opened: changes.everOpened ?? this.everOpened,
      ever_reset: changes.everReset ?? this.everReset,
      ever_committed: changes.everCommitted ?? this.everCommitted,
    };
    parseInventory(inventory);
    this.storage.save(inventory);
  }

  /**
   * Release storage and invalidate idle handles; repeated closes are inert.
   *
   * @throws MemoryOperationError `episodic.busy` (retryable) while an operation is active,
   *   including a re-entrant close from an awaited `rewriteQuery`; a storage cleanup failure.
   */
  close(): void {
    if (this.closed) return;
    if (this.activeOperations > 0) {
      throw new MemoryOperationError("close", "episodic.busy", "memory backend is active", true);
    }
    this.storage?.close();
    this.closed = true;
  }

  /**
   * Open a run session over one pinned memory snapshot.
   *
   * @param options.snapshot Recorded episodic snapshot to restore or replay; `null` or unset pins
   *   the current state.
   * @returns The isolated run-scoped session.
   * @throws MemoryOperationError The snapshot is invalid or incompatible, the backend is closed,
   *   or durable state can't be opened, read or saved.
   */
  openRun(options: { snapshot?: JsonObject | null } = {}): EpisodicSession {
    return this.operation(() => {
      const snapshot = options.snapshot ?? null;
      const parsed = snapshot === null ? null : parseSnapshot(snapshot);
      if (parsed !== null) resolveLimits(this.requested, parsed[3].limits, "open_run");
      // The state is frozen, so a session pins it without Python's defensive copy.
      const [namespace, epoch, revision, state] = parsed ?? [this.namespace, this.epoch, this.revision, this.state];
      let adopt = false;
      if (parsed !== null) {
        if (namespace !== this.namespace) {
          const pristine =
            !this.everOpened &&
            !this.everReset &&
            !this.everCommitted &&
            this.state.episodes.length === 0 &&
            this.epoch === 0 &&
            this.revision === 0;
          if (!pristine) throw snapshotError();
          adopt = true;
        } else {
          if (epoch !== this.epoch || revision > this.revision) throw snapshotError();
          if (
            revision === this.revision &&
            logJson(snapshotJson(namespace, epoch, revision, state)) !==
              logJson(snapshotJson(namespace, epoch, revision, this.state))
          ) {
            throw snapshotError();
          }
        }
      }
      if (adopt || !this.everOpened) {
        const candidate = adopt
          ? snapshotJson(namespace, epoch, revision, state)
          : snapshotJson(this.namespace, this.epoch, this.revision, this.state);
        this.saveState(candidate, {
          everOpened: true,
          everCommitted: adopt ? revision > 0 : this.everCommitted,
        });
        if (adopt) {
          this.namespace = namespace;
          this.epoch = epoch;
          this.revision = revision;
          this.state = state;
          this.everCommitted = revision > 0;
        }
        this.everOpened = true;
      }
      return new EpisodicSession(this, namespace, epoch, revision, state);
    });
  }

  /**
   * Clear episodes and begin a new generation, retaining commit tombstones.
   *
   * @throws MemoryOperationError The backend is closed, or durable state can't be opened or saved.
   * @throws RangeError, without changing state, when the epoch is already 2^53 − 1.
   */
  reset(): void {
    this.operation(() => {
      // The epoch is a `number`, so one past 2^53 − 1 would lose its increment.
      if (this.epoch >= Number.MAX_SAFE_INTEGER) {
        throw new RangeError(`epoch must be at most ${Number.MAX_SAFE_INTEGER}`);
      }
      const cleared: State = Object.freeze({ limits: this.state.limits, episodes: Object.freeze([]) });
      this.saveState(snapshotJson(this.namespace, this.epoch + 1, 0, cleared), { everReset: true });
      this.state = cleared;
      this.epoch += 1;
      this.revision = 0;
      this.everReset = true;
    });
  }

  /**
   * Derive deterministic lexical retrieval text from visible context: the base system prompt, the
   * prompt and the inputs' log JSON, one per line. A subclass may override it, also with an async
   * method.
   *
   * @throws TypeError when `context` isn't exactly a `RecallContext`.
   * @throws MemoryOperationError `episodic.closed` after `close()`; durable state can't be opened.
   */
  rewriteQuery(context: RecallContext): string | Promise<string> {
    return this.operation(() => {
      if (!isExactly(context, RecallContext)) throw new TypeError("context must be a RecallContext");
      return [context.baseSystemPrompt, context.prompt, logJson(context.inputs)].join("\n");
    });
  }
}

/**
 * One run-scoped view over a pinned episodic-memory generation (Python's `_EpisodicSession`).
 *
 * It reads and publishes its backend's private state through element access (`memory["state"]`),
 * TypeScript's escape for private members, as Python's session reads `self._memory._state`.
 */
export class EpisodicSession implements MemorySession {
  private readonly memory: EpisodicMemory;
  private readonly namespace: string;
  private readonly epoch: number;
  private readonly revision: number;
  private readonly state: State;
  private closed = false;

  /** @internal Pin one generation; `EpisodicMemory.openRun` builds sessions. */
  constructor(memory: EpisodicMemory, namespace: string, epoch: number, revision: number, state: State) {
    this.memory = memory;
    this.namespace = namespace;
    this.epoch = epoch;
    this.revision = revision;
    this.state = state;
  }

  /**
   * Reject every operation after the run boundary closes.
   *
   * @throws MemoryOperationError `episodic.session_closed` after `close()`.
   */
  private ensureOpen(): void {
    if (this.closed) throw operationError("session", "episodic.session_closed", "memory session is closed");
  }

  /**
   * The public snapshot pinned for this run, deep-frozen.
   *
   * @throws MemoryOperationError The session or backend is closed.
   */
  get snapshot(): JsonObject {
    return this.memory["operation"](() => {
      this.ensureOpen();
      return snapshotJson(this.namespace, this.epoch, this.revision, this.state);
    });
  }

  /**
   * Render pinned retained history, optionally ranked by lexical relevance: no context gives every
   * entry, oldest first; a context gives the `recallLimit` entries sharing the most terms with
   * `memory.rewriteQuery(context)`, newest first among equals. Each entry is one log-JSON line.
   *
   * @throws MemoryOperationError `episodic.invalid_context` for a context that isn't exactly a
   *   `RecallContext`; `episodic.invalid_query` when rewriting fails or returns a non-string; the
   *   session or backend is closed.
   */
  async recall(context: RecallContext | null = null): Promise<string> {
    // The operation spans the awaited rewrite, so `close()` stays busy until recall returns.
    this.memory["begin"]();
    try {
      this.ensureOpen();
      if (context !== null && !isExactly(context, RecallContext)) {
        throw operationError("recall", "episodic.invalid_context", "context is invalid");
      }
      let entries = this.state.episodes;
      if (context !== null) {
        let query: unknown;
        try {
          query = await this.memory.rewriteQuery(context);
        } catch (error) {
          throw operationError("recall", "episodic.invalid_query", "query rewriting failed", { cause: error });
        }
        // No session check after the await: recall reads only this session's pinned state.
        if (typeof query !== "string") {
          throw operationError("recall", "episodic.invalid_query", "query rewriting must return text");
        }
        const queryTerms = lexicalTerms(query);
        entries = entries
          .map((entry, index) => {
            let overlap = 0;
            for (const term of lexicalTerms(logJson(entry))) if (queryTerms.has(term)) overlap += 1;
            return { entry, index, overlap };
          })
          .toSorted((a, b) => b.overlap - a.overlap || b.index - a.index)
          .slice(0, this.state.limits.recallLimit)
          .map((ranked) => ranked.entry);
      }
      return entries.map((entry) => logJson(entry)).join("\n");
    } finally {
      this.memory["end"]();
    }
  }

  /**
   * Propose one deterministic append for every episode in the batch; an entry id is the SHA-256 of
   * the log JSON of the episode, memory key, ordinal and run id.
   *
   * @throws MemoryOperationError `episodic.invalid_context` for a context that isn't exactly a
   *   `LearnContext` or an episode that can't be a mutation payload (`±Infinity`);
   *   `episodic.stale_generation` when the backend moved on; the session or backend is closed.
   */
  learn(context: LearnContext): MemoryMutation[] {
    return this.memory["operation"](() => {
      this.ensureOpen();
      if (!isExactly(context, LearnContext)) {
        throw operationError("learn", "episodic.invalid_context", "learning context is invalid");
      }
      this.assertCurrent("learn");
      return context.episodes.map((episode, ordinal) => {
        const identity = {
          episode: episodeJson(episode),
          memory: context.memory,
          ordinal: BigInt(ordinal),
          run_id: context.runId,
        };
        const entryId = crypto.createHash("sha256").update(logJson(identity)).digest("hex");
        // An episode holding `±Infinity` can't be a mutation payload, so it fails as a learning
        // context Python can't build.
        try {
          return new MemoryMutation(entryId, { entry: storedEntry(entryId, episode), operation: "append" });
        } catch (error) {
          if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
          throw operationError("learn", "episodic.invalid_context", "learning context is invalid", { cause: error });
        }
      });
    });
  }

  /** Idempotently close this session, without closing its backend. */
  close(): void {
    this.closed = true;
  }

  /**
   * Require this session to remain the current memory generation.
   *
   * @throws MemoryOperationError `episodic.stale_generation` otherwise.
   */
  private assertCurrent(stage: string): void {
    if (
      this.namespace !== this.memory["namespace"] ||
      this.epoch !== this.memory["epoch"] ||
      this.revision !== this.memory["revision"]
    ) {
      throw operationError(stage, "episodic.stale_generation", "memory session is stale");
    }
  }

  /**
   * Compute a canonical digest for one exact mutation array: the SHA-256 of the log JSON of each
   * mutation's id and payload.
   *
   * @throws MemoryOperationError `episodic.invalid_batch` for a non-array, an item that isn't
   *   exactly a `MemoryMutation`, or a payload log JSON refuses (`-0`).
   */
  private batchDigest(mutations: unknown): string {
    if (!Array.isArray(mutations) || Object.getPrototypeOf(mutations) !== Array.prototype) {
      throw operationError("commit", "episodic.invalid_batch", "mutations must be an exact array");
    }
    const encoded = (mutations as unknown[]).map((mutation) => {
      if (!isExactly(mutation, MemoryMutation)) {
        throw operationError("commit", "episodic.invalid_batch", "mutation batch contains an invalid value");
      }
      return { mutation_id: mutation.mutationId, payload: mutation.payload };
    });
    let material: string;
    try {
      material = logJson(encoded);
    } catch (error) {
      if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
      throw operationError("commit", "episodic.invalid_batch", "mutation batch is not canonical JSON", {
        cause: error,
      });
    }
    return crypto.createHash("sha256").update(material).digest("hex");
  }

  /**
   * Atomically append one exact idempotent mutation batch; a repeated commit id with the same
   * batch is a no-op, and retention keeps the newest `maxEpisodes` entries.
   *
   * @throws MemoryOperationError `episodic.invalid_commit_id`, `episodic.invalid_batch`,
   *   `episodic.commit_collision`, `episodic.stale_generation`, `episodic.config_conflict`; the
   *   session or backend is closed; durable state can't be saved.
   */
  commit(commitId: string, mutations: readonly MemoryMutation[]): void {
    const memory = this.memory;
    memory["operation"](() => {
      this.ensureOpen();
      let id: string;
      try {
        id = jsonString(commitId, "commit_id", true);
      } catch (error) {
        if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
        throw operationError("commit", "episodic.invalid_commit_id", "commit id is invalid", { cause: error });
      }
      const digest = this.batchDigest(mutations);
      const applied = memory["appliedCommits"].get(id);
      if (applied !== undefined) {
        if (applied === digest) return;
        throw operationError("commit", "episodic.commit_collision", "commit id collision with a different batch");
      }
      this.assertCurrent("commit");
      let state = memory["state"];
      if (
        state.limits.maxEpisodes !== this.state.limits.maxEpisodes ||
        state.limits.recallLimit !== this.state.limits.recallLimit
      ) {
        throw operationError("commit", "episodic.config_conflict", "current memory limits changed");
      }
      try {
        const seen = new Set<string>();
        const additions = mutations.map((mutation, index) => {
          if (seen.has(mutation.mutationId)) throw new RangeError("mutation IDs must be unique");
          seen.add(mutation.mutationId);
          const payload = copyObject(mutation.payload, `mutations[${index}].payload`);
          if (!hasFields(payload, ["entry", "operation"]) || payload.operation !== "append") {
            throw new RangeError("mutation payload must be a closed append operation");
          }
          const entry = validateEntry(payload.entry, `mutations[${index}].payload.entry`);
          if (entry.entry_id !== mutation.mutationId) throw new RangeError("mutation and entry IDs must match");
          return entry;
        });
        const existing = new Set(state.episodes.map((entry) => entry.entry_id));
        if (additions.some((entry) => existing.has(entry.entry_id))) {
          throw new RangeError("episode entry is already retained");
        }
        const retained = [...state.episodes, ...additions].slice(-state.limits.maxEpisodes);
        state = Object.freeze({ limits: state.limits, episodes: Object.freeze(retained) });
      } catch (error) {
        if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
        throw operationError("commit", "episodic.invalid_batch", "mutation batch is invalid", { cause: error });
      }
      const commits = new Map(memory["appliedCommits"]).set(id, digest);
      const revision = memory["revision"] + 1;
      memory["saveState"](snapshotJson(memory["namespace"], memory["epoch"], revision, state), {
        appliedCommits: commits,
        everCommitted: true,
      });
      memory["state"] = state;
      memory["appliedCommits"] = commits;
      memory["revision"] = revision;
      memory["everCommitted"] = true;
    });
  }
}
