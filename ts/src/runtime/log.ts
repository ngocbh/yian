/**
 * Closed run-record vocabulary and canonical JSON codecs.
 *
 * The dispatcher is the sole producer of these durable records. Records contain only transition
 * data; their enclosing `RecordEnvelope` owns run identity, schema version and sequence. This
 * module deliberately knows nothing about the scheduler or reducer, so the same validated
 * carriers can back memory and JSONL logs without creating an authority cycle.
 *
 * Knows about:   diagnostics, expr/lineage, memory/contracts, nodes/base, nodes/contracts,
 *                platform, runtime/contracts, typesys.
 * Never imports: state, trace, engine, run, compile, compose, or the CLI.
 */

import fs from "node:fs";

import { compareCodeUnits, Failure, OriginRef, reprText } from "../diagnostics.ts";
import {
  type FeedbackValue,
  type RouteKind,
  type RoutePath,
  SourceRef,
  ValueRoute,
  validateFeedback,
} from "../expr/lineage.ts";
import {
  DeclSite,
  EvaluationEvidence,
  LearningFailure,
  type LearningOperation,
  LearningResult,
  type LearningStatus,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryOperationFailure,
  MemoryScope,
  MemorySection,
  MemorySnapshotEnvelope,
  MemorySpec,
  MemoryStack,
  ModelTextInteraction,
  ModelValueInteraction,
  PolicyCommitOutcome,
  type PolicyCommitStatus,
  ToolCallInteraction,
  ToolResultInteraction,
  validateMutations,
  type VisibleInteraction,
} from "../memory/contracts.ts";
import { BackwardResult, copyBackwardValue, retainFailure } from "../nodes/base.ts";
import { type FileLock, FileLockError, isLocked, lockFile } from "../platform.ts";
import {
  comparePaths,
  ExecutionPosition,
  ExecutionScope,
  isExactly,
  type PositionKind,
  sameItems,
} from "../nodes/contracts.ts";
import { JSONDecodeError, type JsonObject, type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import {
  buildValue,
  buildValueAs,
  type PoolWire,
  type Type,
  TypedValue,
  type ValueKind,
  VariablePool,
} from "../typesys/index.ts";
import { isList, isRecord, jsonPart, requireFields, valueRepr, writeAsText } from "../typesys/values.ts";
import {
  AttributedBackwardResult,
  BackwardAttribution,
  BackwardFailure,
  BackwardOutcome,
  type BackwardStatus,
  ForwardCapture,
  MemoryProposal,
  validateRoute,
  validateSource,
} from "./contracts.ts";

/** The only run-record schema that this version writes and reads. */
export const SCHEMA_VERSION = 10;

/** A typed value's wire object `{kind, value}`. */
type TypedWire = ReturnType<TypedValue["toWire"]>;

/**
 * Canonicalize a node value through its durable typed JSON representation.
 *
 * `value` is a live node output, or an existing `TypedValue`. `declared` is the target output
 * type (checked with `buildValueAs`), or `null` to infer the natural storage type
 * (`buildValue`).
 *
 * Returns the canonical typed wire object (a fresh JSON copy, keys in code-unit order, each
 * `FileRef` as its mapping) and the value decoded from an independent copy of that exact
 * object.
 *
 * @throws TypeCheckError at the rejected value's path when the value doesn't satisfy
 *   `declared`, or isn't made of JSON values and `FileRef`s (NaN, text holding a lone surrogate,
 *   a container inside itself, a `Date`, a `Set`, ...).
 */
export function canonicalTyped(value: unknown, declared: ValueKind | Type | null): [TypedWire, TypedValue] {
  const typed = declared === null ? buildValue(value) : buildValueAs(declared, value);
  const text = logJson(typed.toWire());
  return [parseJson(text, "codec") as TypedWire, TypedValue.fromWire(parseJson(text, "codec"))];
}

/**
 * Canonicalize a seed value (builder input or invocation data) as JSON.
 *
 * Returns independent, mutable canonical JSON copies, the wire and the decoded value, keys in
 * code-unit order; each `FileRef` becomes its mapping and `±Infinity` stays a float. Live
 * expansion building must consume the decoded second value.
 *
 * @throws TypeCheckError at the rejected value's path when the value isn't a JSON value (see
 *   `jsonPart`): NaN, text holding a lone surrogate, a container inside itself, a `Date`, ...
 */
export function canonicalSeed(value: unknown): [JsonValue, JsonValue] {
  const text = logJson(jsonPart(value, [], "mapping"));
  return [parseJson(text, "codec"), parseJson(text, "codec")];
}

/**
 * Encode a variable pool to its canonical JSON object, a `RunStartedRecord`'s `initial_pool`.
 *
 * @throws TypeCheckError when a stored value isn't a JSON value (see `TypedValue.toWire`).
 * @throws RangeError when a stored typed value doesn't hold its kind (`TypedValue.fromWire`).
 */
export function encodePool(pool: VariablePool): PoolWire {
  const wire = canonicalCopy(pool.toWire()) as unknown as PoolWire;
  // The unchecked `TypedValue` constructor can hold a value of the wrong kind.
  decodePool(wire);
  return wire;
}

/**
 * Decode and strictly validate a canonical variable-pool wire object.
 *
 * `wire` must have exactly `store`, `system` and `start_id` (see `VariablePool.fromWire`). The
 * pool's values are decoded from a fresh copy, so they share nothing with `wire`.
 *
 * @throws RangeError when the object has missing or extra fields or invalid typed values.
 * @throws TypeError when it holds data outside canonical JSON (the writer's error).
 */
export function decodePool(wire: PoolWire): VariablePool {
  return VariablePool.fromWire(canonicalCopy(wire));
}

/**
 * Return a fresh canonical copy of `value`, keys in code-unit order, as the run log reads it back.
 *
 * @throws TypeError when `value` isn't canonical JSON (the writer's error).
 */
function canonicalCopy(value: unknown): JsonValue {
  return parseJson(logJson(value), "codec");
}

/**
 * Freeze `root` and every container in it, in place, and return it.
 *
 * `root` must be a fresh value `parseJson` returned, so it is a tree of plain containers; nothing
 * is checked. An explicit stack walks it, so any depth works.
 */
function freezeParsed<T extends JsonValue>(root: T): T {
  const stack: JsonValue[] = [root];
  while (stack.length > 0) {
    const value = stack.pop()!;
    if (typeof value !== "object" || value === null) continue;
    Object.freeze(value);
    for (const item of Object.values(value)) stack.push(item);
  }
  return root;
}

/**
 * Why a run stopped before it finished: `signal` when the host's abort signal fired (the CLI's
 * SIGINT, SIGTERM and SIGHUP included), `callback` when a host callback threw or rejected.
 */
export const RUN_STOP_REASONS = Object.freeze(["signal", "callback"] as const);

/** One of `RUN_STOP_REASONS`. */
export type RunStopReason = (typeof RUN_STOP_REASONS)[number];

/** The optional carriers accepted with a node outcome, both frozen and held by reference. */
export interface OutcomeCarriers {
  /** Visible evidence accepted with the outcome; defaults to `null`. */
  evidence?: EvaluationEvidence | null;
  /** The binding capture accepted with the outcome; defaults to `null`. */
  forwardCapture?: ForwardCapture | null;
}

/** The fields of a `RunStartedRecord`. */
export interface RunStartedFields {
  /** The authored flow's SHA-256 identity, 64 lowercase hex digits. */
  flowFingerprint: string;
  /** The full initial variable pool's canonical wire object (see `encodePool`). */
  initialPool: PoolWire;
  /** The per-run wiring override of the synthesized START node. */
  startWiringOverride: Readonly<Record<string, unknown>>;
  /** The complete resolved agent configuration, keyed by stable static node path. */
  effectiveLlmByStaticPath: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  /** The initial dispatcher worker count; zero selects serial execution. */
  numWorkers: number;
  /** Recorded snapshots, one per memory key in key order; `null` (the default) when memory is disabled. */
  memorySnapshots?: readonly MemorySnapshotEnvelope[] | null;
  /** Whether the run retains learning evidence; defaults to `false`. */
  learnable?: boolean;
  /** The memory key table, sorted by key; `null` (the default) when memory is disabled. */
  memories?: readonly MemorySpec[] | null;
  /** Each target's ordered key stack, sorted by locator; `null` (the default) when memory is disabled. */
  memoryStacks?: readonly MemoryStack[] | null;
  /** The `yian.defaults` values that differ from the built-in ones, by entry name; defaults to `{}`. */
  defaults?: Readonly<Record<string, bigint>>;
  /**
   * The agents whose configuration resolves only when they run, `{selection, agents}`: `agents`
   * maps each one's static path to its base layer, and `selection` holds the connection selection
   * they bind from. `null` (the default) when the run has no such agent.
   */
  callTimeLlm?: Readonly<Record<string, unknown>> | null;
}

/** Initialize a run from its complete canonical inputs and resolved configuration. Frozen. */
export class RunStartedRecord {
  /** The authored flow's SHA-256 identity, 64 lowercase hex digits. */
  readonly flowFingerprint: string;
  /** The full initial variable pool's deep-frozen canonical wire object. */
  readonly initialPool: Readonly<PoolWire>;
  /** The deep-frozen canonical START wiring override; a `FileRef` is held as its mapping. */
  readonly startWiringOverride: Readonly<JsonObject>;
  /** The deep-frozen canonical agent configuration by static path; a `FileRef` is held as its mapping. */
  readonly effectiveLlmByStaticPath: Readonly<Record<string, Readonly<JsonObject>>>;
  /** The initial dispatcher worker count; zero selects serial execution. */
  readonly numWorkers: number;
  /** Recorded snapshots, one per memory key in key order, or `null` when memory is disabled. */
  readonly memorySnapshots: readonly MemorySnapshotEnvelope[] | null;
  /** Whether the run retains learning evidence. */
  readonly learnable: boolean;
  /** The memory key table, sorted by key, or `null` when memory is disabled. */
  readonly memories: readonly MemorySpec[] | null;
  /** Each scoped target's ordered key stack, sorted by locator, or `null` when memory is disabled. */
  readonly memoryStacks: readonly MemoryStack[] | null;
  /** The non-built-in `yian.defaults` values by entry name; empty means all built-in values. */
  readonly defaults: Readonly<Record<string, bigint>>;
  /** The deep-frozen canonical call-time agents `{selection, agents}`, or `null` when there are none. */
  readonly callTimeLlm: Readonly<JsonObject> | null;

  /**
   * The memory table, stacks and snapshots are all recorded or all `null`; recorded, they are
   * copied into frozen arrays that hold the given carriers. JSON fields keep `±Infinity` and read
   * `-0` back as `0`.
   *
   * @throws TypeError when the fingerprint isn't a string, the wiring override, the agent
   *   configuration or one agent's configuration isn't a JSON object, `numWorkers` isn't a safe
   *   integer, `learnable` isn't a boolean, the memory fields aren't arrays of exactly their carrier
   *   class, `defaults` isn't a plain object of `bigint`s, or `callTimeLlm` or one of its parts
   *   isn't a plain object.
   * @throws RangeError when the fingerprint isn't 64 lowercase hex digits, `numWorkers` is
   *   negative, the initial pool isn't a pool wire object (see `decodePool`), the memory fields
   *   are only partly recorded, aren't sorted and unique, disagree on their keys or a stack names
   *   an unknown key, a defaults name is empty, or `callTimeLlm` has missing or extra fields, no
   *   agent, an empty agent path or a non-JSON value.
   * @throws TypeCheckError at the bad part's path when the wiring override or the agent
   *   configuration holds NaN or another non-JSON value (see `jsonPart`).
   */
  constructor(fields: RunStartedFields) {
    const {
      memorySnapshots = null,
      learnable = false,
      memories = null,
      memoryStacks = null,
      defaults = {},
      callTimeLlm = null,
    } = fields;
    this.flowFingerprint = requireSha256(fields.flowFingerprint, "flow_fingerprint");
    // Decoding only validates: a decoded pool writes back exactly this canonical copy.
    const pool = canonicalCopy(fields.initialPool);
    VariablePool.fromWire(pool);
    this.initialPool = freezeParsed(pool) as unknown as PoolWire;
    const wiring = canonicalCopy(jsonPart(fields.startWiringOverride, [], "mapping"));
    if (!isRecord(wiring)) throw new TypeError("start_wiring_override must be a mapping");
    this.startWiringOverride = freezeParsed(wiring);
    const llm = canonicalCopy(jsonPart(fields.effectiveLlmByStaticPath, [], "mapping"));
    if (!isRecord(llm)) throw new TypeError("effective_llm_by_static_path must be a mapping");
    for (const [path, config] of Object.entries(llm)) {
      if (!isRecord(config)) {
        throw new TypeError(`effective_llm_by_static_path value for ${reprText(path)} must be a mapping`);
      }
    }
    this.effectiveLlmByStaticPath = freezeParsed(llm) as Readonly<Record<string, Readonly<JsonObject>>>;
    this.numWorkers = validatedCount(fields.numWorkers, "num_workers");
    [this.memories, this.memoryStacks, this.memorySnapshots] = validatedRunMemoryConfiguration(
      memories,
      memoryStacks,
      memorySnapshots,
      learnable,
    );
    this.learnable = learnable;
    this.defaults = validatedDefaults(defaults);
    this.callTimeLlm = callTimeLlm === null ? null : validatedCallTimeLlm(callTimeLlm);
    Object.freeze(this);
  }
}

/**
 * Return a deep-frozen canonical copy of a `RunStarted.callTimeLlm` value after checking its
 * shape: `{selection: {...}, agents: {path: {...}, ...}}` with at least one agent. The contents
 * are checked when the run applies them.
 *
 * @throws TypeError when the value or one of its parts isn't a plain object.
 * @throws RangeError when a field is missing or extra, `agents` is empty or has an empty path,
 *   or a value isn't JSON (NaN, a `FileRef`, ...).
 */
function validatedCallTimeLlm(value: unknown): Readonly<JsonObject> {
  if (!isRecord(value)) throw new TypeError("RunStarted call_time_llm must be a dict");
  const keys = Object.keys(value);
  if (keys.length !== 2 || !Object.hasOwn(value, "selection") || !Object.hasOwn(value, "agents")) {
    throw new RangeError("RunStarted call_time_llm must have exactly selection and agents");
  }
  if (!isRecord(value["selection"])) throw new TypeError("RunStarted call_time_llm selection must be a dict");
  const agents = value["agents"];
  if (!isRecord(agents)) throw new TypeError("RunStarted call_time_llm agents must be a dict");
  if (Object.keys(agents).length === 0) throw new RangeError("RunStarted call_time_llm agents must not be empty");
  for (const [path, base] of Object.entries(agents)) {
    if (!path) throw new RangeError("RunStarted call_time_llm agent paths must be non-empty strings");
    if (!isRecord(base)) throw new TypeError(`RunStarted call_time_llm base for ${reprText(path)} must be a dict`);
  }
  let copy: JsonObject;
  try {
    copy = canonicalCopy(jsonPart(value, [], "refuse")) as JsonObject;
  } catch (error) {
    throw new RangeError(`RunStarted call_time_llm: ${(error as Error).message}`, { cause: error });
  }
  return freezeParsed(copy);
}

/**
 * Return a frozen copy of recorded defaults: non-empty entry names mapped to `bigint`s.
 *
 * @throws TypeError when `recorded` isn't a plain object or a value isn't a `bigint`.
 * @throws RangeError when a name is empty.
 */
function validatedDefaults(recorded: unknown): Readonly<Record<string, bigint>> {
  if (!isRecord(recorded)) throw new TypeError("RunStarted defaults must be a mapping");
  const entries = Object.entries(recorded);
  for (const [name, value] of entries) {
    if (!name) throw new RangeError("RunStarted defaults names must be non-empty strings");
    if (typeof value !== "bigint") {
      throw new TypeError(`RunStarted defaults value for ${reprText(name)} must be an int`);
    }
  }
  // `Object.fromEntries` defines own data properties, so a `__proto__` name stays a key.
  return Object.freeze(Object.fromEntries(entries) as Record<string, bigint>);
}

/** Start a recovery segment with ordered unfinished work and a worker count. Frozen. */
export class RunResumedRecord {
  /** Unfinished node ids to retry, in order; unique and non-empty. */
  readonly retryNodeIds: readonly string[];
  /** The worker count of the recovery segment. */
  readonly numWorkers: number;

  /**
   * @throws RangeError when `retryNodeIds` isn't an array of unique non-empty strings, or
   *   `numWorkers` is negative.
   * @throws TypeError when `numWorkers` isn't a safe integer.
   */
  constructor(retryNodeIds: readonly string[], numWorkers: number) {
    this.retryNodeIds = validatedIdArray(retryNodeIds, "retry_node_ids");
    this.numWorkers = validatedCount(numWorkers, "num_workers");
    Object.freeze(this);
  }
}

/** Record that the host stopped the run while nodes were in flight. Not terminal. Frozen. */
export class RunStoppedRecord {
  /** Why the run stopped. */
  readonly reason: RunStopReason;
  /** The reason's text, such as the callback's error message; any string. */
  readonly message: string;
  /** The in-flight node ids in dispatcher order; unique and non-empty, possibly none. */
  readonly nodeIds: readonly string[];

  /**
   * @throws TypeError when `reason` or `message` isn't a string.
   * @throws RangeError for an unknown reason, or node ids that aren't an array of unique
   *   non-empty strings.
   */
  constructor(reason: RunStopReason, message: string, nodeIds: readonly string[]) {
    if (typeof reason !== "string") throw new TypeError("reason must be a string");
    if (!(RUN_STOP_REASONS as readonly string[]).includes(reason)) {
      throw new RangeError(`reason must be one of ${writeAsText([...RUN_STOP_REASONS])}, got ${reprText(reason)}`);
    }
    if (typeof message !== "string") throw new TypeError("message must be a string");
    this.reason = reason;
    this.message = message;
    this.nodeIds = validatedIdArray(nodeIds, "node_ids");
    Object.freeze(this);
  }
}

/** Move the exact ready-head node into the ordered in-flight set. Frozen. */
export class NodeStartedRecord {
  /** The started node's non-empty id. */
  readonly nodeId: string;

  /**
   * @throws TypeError when `nodeId` isn't a string.
   * @throws RangeError when it is empty.
   */
  constructor(nodeId: string) {
    this.nodeId = requireNonemptyString(nodeId, "node_id");
    Object.freeze(this);
  }
}

/**
 * Append one step an in-flight node recorded while it runs. Frozen.
 *
 * The step is opaque to the runtime: the node that writes it gives it meaning, and a later
 * evaluation of the same node reads its recorded steps back.
 */
export class NodeStepRecord {
  /** Non-empty id of the in-flight node that recorded the step. */
  readonly nodeId: string;
  /** A deep-frozen canonical copy of the step, keys in code-unit order. */
  readonly step: Readonly<JsonObject>;

  /**
   * @param step A JSON object; `-0` is kept as `0` and `±Infinity` as is.
   * @throws TypeError when `nodeId` isn't a string or `step` isn't a plain object.
   * @throws RangeError when `nodeId` is empty.
   * @throws TypeCheckError at the bad part's path when the step holds NaN, a `FileRef` or
   *   another non-JSON value (see `jsonPart`).
   */
  constructor(nodeId: string, step: Readonly<Record<string, unknown>>) {
    this.nodeId = requireNonemptyString(nodeId, "node_id");
    if (!isRecord(step)) throw new TypeError("node step must be a JSON object");
    this.step = freezeParsed(canonicalCopy(jsonPart(step, [], "refuse")) as JsonObject);
    Object.freeze(this);
  }
}

/** Commit one canonical typed value under an already-resolved target. Frozen. */
export class NodeSucceededRecord {
  /** The succeeded node's non-empty id. */
  readonly nodeId: string;
  /** The non-empty pool key the value is committed under. */
  readonly commitTarget: string;
  /** The value's deep-frozen canonical typed wire `{kind, value}`. */
  readonly value: Readonly<TypedWire>;
  /** Visible evidence accepted with the outcome, if any. */
  readonly evidence: EvaluationEvidence | null;
  /** The binding capture accepted with the outcome, if any. */
  readonly forwardCapture: ForwardCapture | null;

  /**
   * @param value A typed value's wire form (see `TypedValue.fromWire`).
   * @throws TypeError when an id isn't a string, or a carrier is of another class.
   * @throws RangeError when an id is empty or `value` isn't a typed-value wire form.
   * @throws TypeCheckError when the value holds NaN or another non-JSON value.
   */
  constructor(
    nodeId: string,
    commitTarget: string,
    value: TypedWire,
    options: OutcomeCarriers = {},
  ) {
    const { evidence = null, forwardCapture = null } = options;
    this.nodeId = requireNonemptyString(nodeId, "node_id");
    this.commitTarget = requireNonemptyString(commitTarget, "commit_target");
    this.value = freezeParsed(canonicalCopy(TypedValue.fromWire(value).toWire())) as unknown as TypedWire;
    this.evidence = validatedEvidence(evidence);
    this.forwardCapture = validatedCapture(forwardCapture);
    Object.freeze(this);
  }
}

/** Select one route and retain any terminal failure it first proves. Frozen. */
export class NodeRoutedRecord {
  /** The routing node's non-empty id. */
  readonly nodeId: string;
  /** The selected non-empty route handle. */
  readonly handle: string;
  /** The terminal-unreachable failure this route first proves, if any. */
  readonly terminalFailure: Failure | null;
  /** Visible evidence accepted with the outcome, if any. */
  readonly evidence: EvaluationEvidence | null;
  /** The binding capture accepted with the outcome, if any. */
  readonly forwardCapture: ForwardCapture | null;

  /**
   * A terminal failure is kept through `retainFailure`.
   *
   * @throws TypeError when an id isn't a string, or a failure or carrier is of another class.
   * @throws RangeError when an id is empty, or the failure has no node id.
   */
  constructor(
    nodeId: string,
    handle: string,
    options: OutcomeCarriers & { terminalFailure?: Failure | null } = {},
  ) {
    const { terminalFailure = null, evidence = null, forwardCapture = null } = options;
    this.nodeId = requireNonemptyString(nodeId, "node_id");
    this.handle = requireNonemptyString(handle, "handle");
    this.terminalFailure = terminalFailure === null ? null : retainedFailure(terminalFailure, "NodeRoutedRecord");
    this.evidence = validatedEvidence(evidence);
    this.forwardCapture = validatedCapture(forwardCapture);
    Object.freeze(this);
  }
}

/** Splice one deterministic expansion from its canonical seed and digest. Frozen. */
export class NodeExpandedRecord {
  /** The spawner's non-empty id. */
  readonly spawnerId: string;
  /** The deep-frozen canonical seed; a `FileRef` is held as its mapping. */
  readonly seed: JsonValue;
  /** Node ids the expansion prunes: sorted, unique and non-empty. */
  readonly pruneIds: readonly string[];
  /** The expanded topology's SHA-256 digest, 64 lowercase hex digits. */
  readonly topologyDigest: string;
  /** Visible evidence accepted with the outcome, if any. */
  readonly evidence: EvaluationEvidence | null;
  /** The binding capture accepted with the outcome, if any. */
  readonly forwardCapture: ForwardCapture | null;

  /**
   * @throws TypeError when an id or the digest isn't a string, or a carrier is of another class.
   * @throws RangeError when the spawner id is empty, the prune ids aren't a sorted array of
   *   unique non-empty strings, or the digest isn't 64 lowercase hex digits.
   * @throws TypeCheckError when the seed isn't a JSON value (see `jsonPart`).
   */
  constructor(
    spawnerId: string,
    seed: unknown,
    pruneIds: readonly string[],
    topologyDigest: string,
    options: OutcomeCarriers = {},
  ) {
    const { evidence = null, forwardCapture = null } = options;
    this.spawnerId = requireNonemptyString(spawnerId, "spawner_id");
    this.seed = freezeParsed(canonicalCopy(jsonPart(seed, [], "mapping")));
    this.pruneIds = validatedIdArray(pruneIds, "prune_ids", true);
    this.topologyDigest = requireSha256(topologyDigest, "topology_digest");
    this.evidence = validatedEvidence(evidence);
    this.forwardCapture = validatedCapture(forwardCapture);
    Object.freeze(this);
  }
}

/** Make a run terminal-failed with the captured semantic failure. Frozen. */
export class NodeFailedRecord {
  /** The failure, kept through `retainFailure`; its node id is non-empty. */
  readonly failure: Failure;
  /** Visible evidence accepted with the outcome, if any. */
  readonly evidence: EvaluationEvidence | null;
  /** The binding capture accepted with the outcome, if any. */
  readonly forwardCapture: ForwardCapture | null;

  /**
   * @throws TypeError when the failure or a carrier is of another class.
   * @throws RangeError when the failure has no node id.
   */
  constructor(
    failure: Failure,
    options: OutcomeCarriers = {},
  ) {
    const { evidence = null, forwardCapture = null } = options;
    this.failure = retainedFailure(failure, "NodeFailedRecord");
    this.evidence = validatedEvidence(evidence);
    this.forwardCapture = validatedCapture(forwardCapture);
    Object.freeze(this);
  }
}

/**
 * Return `attemptId` when it is a positive safe integer, an accepted backward attempt's identity.
 *
 * @throws TypeError when it isn't a safe integer `number`.
 * @throws RangeError when it isn't positive.
 */
function validatedAttemptId(attemptId: unknown): number {
  if (typeof attemptId !== "number" || !Number.isSafeInteger(attemptId)) {
    throw new TypeError("attempt_id must be an integer");
  }
  if (attemptId <= 0) throw new RangeError("attempt_id must be positive");
  return attemptId;
}

/**
 * Return a frozen copy of an array of exact `AttributedBackwardResult`s, kept by reference.
 *
 * @throws TypeError when `results` isn't an array or an item is of another class.
 */
function validatedAttributedResults(results: unknown): readonly AttributedBackwardResult[] {
  if (!isList(results)) throw new TypeError("attributed_results must be an array");
  const message = "attributed_results require exact AttributedBackwardResult values";
  return exactItems(results, AttributedBackwardResult, message);
}

/** Start one backward attempt with detached ordinary root feedback. Frozen. */
export class BackwardStartedRecord {
  /** The attempt's positive identity within the run. */
  readonly attemptId: number;
  /** A deep-frozen copy of the attempt's root message. */
  readonly feedback: FeedbackValue;

  /**
   * @param feedback One ordinary root message; agreement with the output's shape is checked by
   *   the reducer.
   * @throws TypeError when `attemptId` isn't a safe integer or `feedback` isn't ordinary
   *   feedback.
   * @throws RangeError when `attemptId` isn't positive.
   */
  constructor(attemptId: number, feedback: FeedbackValue) {
    this.attemptId = validatedAttemptId(attemptId);
    validateFeedback(feedback, feedback);
    this.feedback = copyBackwardValue(feedback, "feedback") as FeedbackValue;
    Object.freeze(this);
  }
}

/** Retain a computed root result, occurrence inventory and memory proposals. Frozen. */
export class BackwardReadyRecord {
  /** The attempt's positive identity within the run. */
  readonly attemptId: number;
  /** The completed three-channel root feedback. */
  readonly rootResult: BackwardResult;
  /** The ordered accepted occurrence results, unreached occurrences included. */
  readonly attributedResults: readonly AttributedBackwardResult[];
  /**
   * The ordered proposals, one per recorded memory key. Table membership and attempt agreement
   * belong to reduction.
   */
  readonly proposals: readonly MemoryProposal[];

  /**
   * The carriers are held by reference; the arrays are copied and frozen.
   *
   * @throws TypeError when `attemptId` isn't a safe integer, a carrier isn't exactly its class,
   *   or an array argument isn't an array.
   * @throws RangeError when `attemptId` isn't positive.
   */
  constructor(
    attemptId: number,
    rootResult: BackwardResult,
    attributedResults: readonly AttributedBackwardResult[],
    proposals: readonly MemoryProposal[],
  ) {
    this.attemptId = validatedAttemptId(attemptId);
    if (!isExactly(rootResult, BackwardResult)) throw new TypeError("result must be an exact BackwardResult");
    this.rootResult = rootResult;
    this.attributedResults = validatedAttributedResults(attributedResults);
    if (!isList(proposals)) throw new TypeError("proposals must be an array");
    this.proposals = exactItems(proposals, MemoryProposal, "proposals require exact MemoryProposal values");
    Object.freeze(this);
  }
}

/** Settle one backward attempt with its outcome and completed inventory. Frozen. */
export class BackwardFinishedRecord {
  /** The attempt's positive identity within the run. */
  readonly attemptId: number;
  /** The backward and learning dispositions, keeping completed root feedback when a later operation fails. */
  readonly outcome: BackwardOutcome;
  /**
   * The completed occurrence inventory, possibly empty; its agreement with the attempt's accepted
   * computation is checked by the reducer.
   */
  readonly attributedResults: readonly AttributedBackwardResult[];

  /**
   * The carriers are held by reference; the array is copied and frozen.
   *
   * @throws TypeError when `attemptId` isn't a safe integer, a carrier isn't exactly its class,
   *   or `attributedResults` isn't an array.
   * @throws RangeError when `attemptId` isn't positive.
   */
  constructor(
    attemptId: number,
    outcome: BackwardOutcome,
    attributedResults: readonly AttributedBackwardResult[] = [],
  ) {
    this.attemptId = validatedAttemptId(attemptId);
    if (!isExactly(outcome, BackwardOutcome)) throw new TypeError("outcome must be an exact BackwardOutcome");
    this.outcome = outcome;
    this.attributedResults = validatedAttributedResults(attributedResults);
    Object.freeze(this);
  }
}

/** Request one memory's atomic idempotent mutation commit. Frozen. */
export class MemoryCommitRequestedRecord {
  /** The non-empty memory key. */
  readonly memory: string;
  /** The non-empty commit id. */
  readonly commitId: string;
  /** The mutations with unique ids, in order, kept by reference. */
  readonly mutations: readonly MemoryMutation[];
  /** The positive identity of the accepted backward attempt. */
  readonly attemptId: number;

  /**
   * @throws TypeError when `attemptId` is missing or isn't a safe integer, an identity isn't a
   *   string, `mutations` isn't an array or an item isn't exactly a `MemoryMutation`.
   * @throws RangeError when `attemptId` isn't positive, an identity is empty or two mutations
   *   share an id.
   */
  constructor(memory: string, commitId: string, mutations: readonly MemoryMutation[], attemptId: number) {
    this.attemptId = validatedAttemptId(attemptId);
    this.memory = requireNonemptyString(memory, "memory");
    this.commitId = requireNonemptyString(commitId, "commit_id");
    this.mutations = validateMutations(mutations);
    Object.freeze(this);
  }
}

/** Confirm that one memory commit completed. Frozen. */
export class MemoryCommitSucceededRecord {
  /** The non-empty memory key. */
  readonly memory: string;
  /** The non-empty commit id. */
  readonly commitId: string;
  /** The positive identity of the accepted backward attempt whose request this settles. */
  readonly attemptId: number;

  /**
   * @throws TypeError when `attemptId` is missing or isn't a safe integer, or an identity isn't
   *   a string.
   * @throws RangeError when `attemptId` isn't positive or an identity is empty.
   */
  constructor(memory: string, commitId: string, attemptId: number) {
    this.attemptId = validatedAttemptId(attemptId);
    this.memory = requireNonemptyString(memory, "memory");
    this.commitId = requireNonemptyString(commitId, "commit_id");
    Object.freeze(this);
  }
}

/** Retain one observed memory commit failure. Frozen. */
export class MemoryCommitFailedRecord {
  /** The non-empty memory key. */
  readonly memory: string;
  /** The non-empty commit id. */
  readonly commitId: string;
  /** The failure, held by reference. */
  readonly failure: MemoryOperationFailure;
  /** The positive identity of the accepted backward attempt whose request this settles. */
  readonly attemptId: number;

  /**
   * @throws TypeError when `attemptId` is missing or isn't a safe integer, an identity isn't a
   *   string, or `failure` isn't exactly a `MemoryOperationFailure`.
   * @throws RangeError when `attemptId` isn't positive or an identity is empty.
   */
  constructor(memory: string, commitId: string, failure: MemoryOperationFailure, attemptId: number) {
    this.attemptId = validatedAttemptId(attemptId);
    this.memory = requireNonemptyString(memory, "memory");
    this.commitId = requireNonemptyString(commitId, "commit_id");
    if (!isExactly(failure, MemoryOperationFailure)) throw new TypeError("failure must be a MemoryOperationFailure");
    this.failure = failure;
    Object.freeze(this);
  }
}

/** One durable run transition. */
export type RunRecord =
  | RunStartedRecord
  | RunResumedRecord
  | RunStoppedRecord
  | NodeStartedRecord
  | NodeStepRecord
  | NodeSucceededRecord
  | NodeRoutedRecord
  | NodeExpandedRecord
  | NodeFailedRecord
  | MemoryCommitRequestedRecord
  | MemoryCommitSucceededRecord
  | MemoryCommitFailedRecord
  | BackwardStartedRecord
  | BackwardReadyRecord
  | BackwardFinishedRecord;

/** A run-record class. */
type RunRecordClass = new (...args: never[]) => RunRecord;

/** Each record class and its snake_case wire kind. */
const RECORD_KINDS: ReadonlyMap<RunRecordClass, string> = new Map<RunRecordClass, string>([
  [RunStartedRecord, "run_started"],
  [RunResumedRecord, "run_resumed"],
  [RunStoppedRecord, "run_stopped"],
  [NodeStartedRecord, "node_started"],
  [NodeStepRecord, "node_step"],
  [NodeSucceededRecord, "node_succeeded"],
  [NodeRoutedRecord, "node_routed"],
  [NodeExpandedRecord, "node_expanded"],
  [NodeFailedRecord, "node_failed"],
  [MemoryCommitRequestedRecord, "memory_commit_requested"],
  [MemoryCommitSucceededRecord, "memory_commit_succeeded"],
  [MemoryCommitFailedRecord, "memory_commit_failed"],
  [BackwardStartedRecord, "backward_started"],
  [BackwardReadyRecord, "backward_ready"],
  [BackwardFinishedRecord, "backward_finished"],
]);

/** Every run-record class. */
export const RUN_RECORD_TYPES: readonly RunRecordClass[] = Object.freeze([...RECORD_KINDS.keys()]);

/** The closed set of wire kinds this schema reads. */
const WIRE_KINDS: ReadonlySet<string> = new Set(RECORD_KINDS.values());

/**
 * Return `value` when it is a non-empty string.
 *
 * @throws TypeError when it isn't a string.
 * @throws RangeError when it is empty.
 */
function requireNonemptyString(value: unknown, name: string): string {
  if (typeof value !== "string") throw new TypeError(`${name} must be a string`);
  if (!value) throw new RangeError(`${name} must be non-empty`);
  return value;
}

/**
 * Return `value` when it is a SHA-256 hex digest: 64 lowercase hex digits.
 *
 * @throws TypeError when it isn't a string.
 * @throws RangeError when it isn't such a digest.
 */
function requireSha256(value: unknown, name: string): string {
  if (typeof value !== "string") throw new TypeError(`${name} must be a string`);
  if (!/^[0-9a-f]{64}$/.test(value)) throw new RangeError(`${name} must be 64 lowercase hex digits`);
  return value;
}

/**
 * Return `value` when it is a non-negative safe integer.
 *
 * @throws TypeError when it isn't a safe integer `number`.
 * @throws RangeError when it is negative.
 */
function validatedCount(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) throw new TypeError(`${name} must be an integer`);
  if (value < 0) throw new RangeError(`${name} must be non-negative`);
  return value;
}

/**
 * Return `evidence` when it is `null` or exactly an `EvaluationEvidence`.
 *
 * @throws TypeError otherwise.
 */
function validatedEvidence(evidence: unknown): EvaluationEvidence | null {
  if (evidence !== null && !isExactly(evidence, EvaluationEvidence)) {
    throw new TypeError("evidence must be an EvaluationEvidence or null");
  }
  return evidence;
}

/**
 * Return `capture` when it is `null` or exactly a `ForwardCapture`.
 *
 * @throws TypeError otherwise.
 */
function validatedCapture(capture: unknown): ForwardCapture | null {
  if (capture !== null && !isExactly(capture, ForwardCapture)) {
    throw new TypeError("forward_capture must be a ForwardCapture or null");
  }
  return capture;
}

/**
 * Return a frozen copy of `items` when each is exactly a `cls` instance, kept by reference.
 *
 * @throws TypeError with `message` otherwise.
 */
function exactItems<T>(
  items: readonly unknown[],
  cls: abstract new (...args: never[]) => T,
  message: string,
): readonly T[] {
  // Spread, so a hole of a sparse array reads as `undefined` and is refused.
  const copy = [...items];
  if (!copy.every((item) => isExactly(item, cls))) throw new TypeError(message);
  return Object.freeze(copy as T[]);
}

/** Whether `items` are in strictly increasing order by `compare`: sorted and unique. */
function strictlyIncreasing<T>(items: readonly T[], compare: (a: T, b: T) => number): boolean {
  return items.every((item, index) => index === 0 || compare(items[index - 1]!, item) < 0);
}

/**
 * Validate one `RunStarted` memory table, its stacks and its snapshots, and return frozen copies
 * of the three arrays (each `null` when memory is disabled), holding the given carriers.
 *
 * @throws TypeError when `learnable` isn't a boolean, a recorded field isn't an array, or an item
 *   isn't exactly its carrier class.
 * @throws RangeError when the fields are only partly recorded, keys or locators aren't sorted and
 *   unique in code-unit order, the snapshot keys differ from the table's, or a stack names an
 *   unknown key.
 */
function validatedRunMemoryConfiguration(
  memories: unknown,
  stacks: unknown,
  snapshots: unknown,
  learnable: unknown,
): [readonly MemorySpec[] | null, readonly MemoryStack[] | null, readonly MemorySnapshotEnvelope[] | null] {
  if (typeof learnable !== "boolean") throw new TypeError("learnable must be a boolean");
  if (new Set([memories === null, stacks === null, snapshots === null]).size !== 1) {
    throw new RangeError("memories, memory_stacks and memory_snapshots must all be recorded or all be absent");
  }
  if (memories === null) return [null, null, null];
  if (!isList(memories) || !isList(stacks) || !isList(snapshots)) {
    throw new TypeError("memories, memory_stacks and memory_snapshots must be arrays when recorded");
  }
  const specs = exactItems(memories, MemorySpec, "memories items must be MemorySpec values");
  const keys = specs.map((spec) => spec.key);
  if (!strictlyIncreasing(keys, compareCodeUnits)) throw new RangeError("memories keys must be sorted and unique");
  const envelopes = exactItems(
    snapshots,
    MemorySnapshotEnvelope,
    "memory_snapshots items must be MemorySnapshotEnvelope values",
  );
  const snapshotKeys = envelopes.map((snapshot) => snapshot.memory);
  if (!strictlyIncreasing(snapshotKeys, compareCodeUnits)) {
    throw new RangeError("memory_snapshots memory keys must be sorted and unique");
  }
  if (!sameItems(snapshotKeys, keys)) {
    throw new RangeError("memory_snapshots memory keys must equal memories keys");
  }
  const keyStacks = exactItems(stacks, MemoryStack, "memory_stacks items must be MemoryStack values");
  if (!strictlyIncreasing(keyStacks.map((stack) => stack.locator), comparePaths)) {
    throw new RangeError("memory_stacks locators must be sorted and unique");
  }
  const known = new Set(keys);
  const unknownKeys = [...new Set(keyStacks.flatMap((stack) => stack.memories))].filter((key) => !known.has(key));
  if (unknownKeys.length > 0) {
    throw new RangeError(`memory_stacks name unknown memory keys: ${writeAsText(unknownKeys.sort(compareCodeUnits))}`);
  }
  return [specs, keyStacks, envelopes];
}

/**
 * Return a frozen copy of a record's failure (see `retainFailure`); `carrier` names the record.
 *
 * @throws TypeError when `failure` isn't exactly a `Failure`, or its fields have wrong types.
 * @throws RangeError when its fields break the failure contract, or its node id is null or empty.
 */
function retainedFailure(failure: unknown, carrier: string): Failure {
  if (!isExactly(failure, Failure)) throw new TypeError(`${carrier} failure must be a Failure`);
  const retained = retainFailure(failure);
  if (!retained.nodeId) throw new RangeError(`${carrier} failure node_id must be a non-empty string`);
  return retained;
}

/**
 * Return a frozen copy of an ordered id array: unique non-empty strings, and sorted in
 * code-unit order when `requireSorted` is set.
 *
 * @throws RangeError when `value` isn't such an array.
 */
function validatedIdArray(value: unknown, label: string, requireSorted = false): readonly string[] {
  if (!isList(value)) throw new RangeError(`${label} must be an array of strings`);
  // Spread, so a hole of a sparse array reads as `undefined` and is refused.
  const ids = [...value];
  if (!ids.every((id): id is string => typeof id === "string" && id !== "")) {
    throw new RangeError(`${label} must contain only non-empty strings`);
  }
  if (new Set(ids).size !== ids.length) throw new RangeError(`${label} must contain unique IDs`);
  if (requireSorted && ids.some((id, index) => index > 0 && compareCodeUnits(ids[index - 1]!, id) > 0)) {
    throw new RangeError(`${label} must be sorted`);
  }
  return Object.freeze(ids);
}

/**
 * Check an envelope's schema version.
 *
 * @throws RangeError when it isn't `SCHEMA_VERSION`, naming Python yian for schema 9.
 */
function requireSchema(schema: unknown): void {
  if (schema === 9) {
    throw new RangeError("run log schema 9 was written by Python yian; resume it with Python yian");
  }
  if (schema === SCHEMA_VERSION) return;
  // `valueRepr` prints a bigint like a number, so `10n` would read as the supported schema.
  const shown = typeof schema === "bigint" ? `${schema}n` : valueRepr(schema);
  throw new RangeError(`unsupported run-record schema: ${shown}`);
}

/**
 * Check a decoded envelope's `schema` member.
 *
 * @throws RangeError when it isn't the int token `SCHEMA_VERSION`, naming Python yian for 9.
 */
function requireWireSchema(schema: unknown): void {
  if (schema === BigInt(SCHEMA_VERSION)) return;
  if (schema === 9n) requireSchema(9);
  // Only an int token (a `bigint`) names a schema; a float token reads as a number, shown as the
  // log writes floats (`10.0`), so it doesn't read as a supported schema.
  const shown = typeof schema === "number" ? logJson(schema) : valueRepr(schema);
  throw new RangeError(`unsupported run-record schema: ${shown}`);
}

/**
 * Return a record's wire kind.
 *
 * @throws TypeError when `record` isn't exactly one of `RUN_RECORD_TYPES`.
 */
function kindForRecord(record: unknown): string {
  for (const [cls, kind] of RECORD_KINDS) if (isExactly(record, cls)) return kind;
  const name = record === null || record === undefined ? String(record) : Object(record).constructor?.name;
  throw new TypeError(`unsupported run record: ${name}`);
}

/**
 * Bind one validated run record to its schema, run id and zero-based sequence. Frozen.
 *
 * The envelope holds the frozen record; its `kind` is the record's stable snake_case wire name.
 */
export class RecordEnvelope {
  /** The run-record schema version, `SCHEMA_VERSION`. */
  readonly schema: number;
  /** The run's non-empty id. */
  readonly runId: string;
  /** The record's zero-based position in the log. */
  readonly seq: number;
  /** The record. */
  readonly record: RunRecord;

  /**
   * The schema is checked first, so a Python log's schema is named whatever its other fields
   * hold.
   *
   * @throws RangeError when the schema isn't `SCHEMA_VERSION`, the run id is empty or holds a
   *   lone surrogate, or `seq` isn't a non-negative safe integer.
   * @throws TypeError when `record` isn't exactly one of `RUN_RECORD_TYPES`.
   */
  constructor(schema: number, runId: string, seq: number, record: RunRecord) {
    requireSchema(schema);
    if (typeof runId !== "string" || !runId) {
      throw new RangeError("run-record envelope run_id must be a non-empty string");
    }
    if (!runId.isWellFormed()) throw new RangeError("Unicode surrogate is not valid canonical JSON");
    if (!Number.isSafeInteger(seq) || seq < 0) {
      throw new RangeError("run-record envelope seq must be a non-negative integer");
    }
    kindForRecord(record);
    this.schema = schema;
    this.runId = runId;
    this.seq = seq;
    this.record = record;
    Object.freeze(this);
  }

  /** The record's stable snake_case wire kind, such as `node_started`. */
  get kind(): string {
    return kindForRecord(this.record);
  }

  /** The record's deep-frozen canonical JSON payload, counters as `bigint`s. */
  get payload(): Readonly<JsonObject> {
    return freezeParsed(canonicalCopy(payloadForRecord(this.record)) as JsonObject);
  }
}

/**
 * Encode one envelope as its canonical run-log line: compact JSON, keys in code-unit order, no
 * trailing newline (a JSONL log adds it). The frozen record is trusted as constructed.
 *
 * @throws TypeError when `envelope` isn't exactly a `RecordEnvelope`.
 */
export function encodeRecord(envelope: RecordEnvelope): string {
  if (!isExactly(envelope, RecordEnvelope)) throw new TypeError("encodeRecord expects a RecordEnvelope");
  return logJson({
    schema: BigInt(envelope.schema),
    run_id: envelope.runId,
    seq: BigInt(envelope.seq),
    kind: envelope.kind,
    payload: payloadForRecord(envelope.record),
  });
}

const UTF8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

/**
 * Decode one run-log line into a validated envelope.
 *
 * `data` is one complete JSON object, as text or UTF-8 bytes, without a trailing newline. The
 * envelope's `schema` is checked before anything else, so a Python log is named whatever else
 * the line holds; the record is built through its constructor, so its checks apply.
 *
 * @throws RangeError when the line isn't JSON yian reads (`invalid run-record JSON`, with the
 *   reader's error as `cause`), the schema isn't `SCHEMA_VERSION`, the kind is unknown, an object
 *   has missing or extra fields, a string is empty, or a counter isn't a safe integer.
 * @throws TypeError when `data` is neither text nor bytes, or a field has the wrong JSON type.
 * @throws JSONDecodeError, the reader's error unchanged, when a route's embedded `value_json`
 *   isn't JSON yian reads.
 */
export function decodeRecord(data: string | Uint8Array): RecordEnvelope {
  let text: string;
  if (typeof data === "string") {
    text = data;
  } else if (data instanceof Uint8Array) {
    try {
      // The BOM is kept, so the reader refuses it.
      text = UTF8.decode(data);
    } catch (error) {
      throw new RangeError("invalid run-record JSON", { cause: error });
    }
  } else {
    throw new TypeError("decodeRecord expects a string or a Uint8Array");
  }
  let raw: JsonValue;
  try {
    raw = parseJson(text, "codec");
  } catch (error) {
    if (error instanceof JSONDecodeError) throw new RangeError("invalid run-record JSON", { cause: error });
    throw error;
  }
  const envelope = wireObject(raw, "run-record envelope");
  if (Object.hasOwn(envelope, "schema")) requireWireSchema(envelope["schema"]);
  requireFields(envelope, ["schema", "run_id", "seq", "kind", "payload"], "run-record envelope");
  const runId = wireName(envelope["run_id"], "run-record envelope run_id");
  const seq = wireInt(envelope["seq"], "run-record envelope seq");
  const kind = wireText(envelope["kind"], "run-record envelope kind");
  if (!WIRE_KINDS.has(kind)) {
    throw new RangeError(`run-record kind ${reprText(kind)} is unsupported in schema ${SCHEMA_VERSION}`);
  }
  const payload = wireObject(envelope["payload"], "run-record envelope payload");
  return new RecordEnvelope(SCHEMA_VERSION, runId, seq, recordFromPayload(kind, payload));
}

/**
 * An ordered, append-only owner of validated record envelopes for one run.
 *
 * `close` must be idempotent. An engine borrows a log and never closes it; the run boundary or
 * the direct caller owns it (`using`).
 */
export interface RunLog {
  /** The sole non-empty run id this log accepts. */
  readonly runId: string;
  /** The durable JSONL path, or `null` for a process-local log. */
  readonly path: string | null;
  /** The zero-based sequence the next successful append receives. */
  readonly nextSequence: number;
  /** Iterate the validated envelopes in contiguous sequence order. */
  [Symbol.iterator](): Iterator<RecordEnvelope>;
  /** Return the validated envelopes in sequence order, as a fresh frozen array. */
  records(): readonly RecordEnvelope[];
  /** Validate `record`, append it under `nextSequence`, and return its retained envelope. */
  append(record: RunRecord): RecordEnvelope;
  /** Release owned resources; repeated calls do nothing. */
  close(): void;
  /** Close the log. */
  [Symbol.dispose](): void;
}

/**
 * A process-local, ordered authority for one run's canonical records.
 *
 * The log retains the codec round trip of every appended record. Closing refuses later appends
 * but keeps the records iterable, so a caller can inspect a completed result after disposing it.
 */
export class MemoryRunLog implements RunLog {
  /** The sole run id this log accepts. */
  readonly runId: string;
  /** Always `null`: this log has no durable file. */
  readonly path = null;
  #records: RecordEnvelope[] = [];
  #closed = false;

  /**
   * Create an empty, zero-based log for one run id.
   *
   * @throws RangeError when `runId` isn't a non-empty string or holds a lone surrogate.
   */
  constructor(runId: string) {
    this.runId = validatedRunId(runId);
  }

  /**
   * Reopen a detached canonical prefix without renumbering its records; the next append gets
   * the sequence after the last record.
   *
   * @throws RangeError when `records` isn't a non-empty array, or its envelopes don't share one
   *   run id with contiguous sequences from 0.
   * @throws TypeError when an item isn't exactly a `RecordEnvelope`.
   */
  static fromRecords(records: readonly RecordEnvelope[]): MemoryRunLog {
    if (!Array.isArray(records) || records.length === 0) {
      throw new RangeError("reopening requires a nonempty record array");
    }
    if (!records.every((envelope) => isExactly(envelope, RecordEnvelope))) {
      throw new TypeError("reopening requires exact RecordEnvelope values");
    }
    const canonical = records.map((envelope) => decodeRecord(encodeRecord(envelope)));
    const runId = canonical[0]!.runId;
    if (canonical.some((envelope, sequence) => envelope.runId !== runId || envelope.seq !== sequence)) {
      throw new RangeError("reopened prefix must have one run and contiguous sequences");
    }
    const log = new MemoryRunLog(runId);
    log.#records = canonical;
    return log;
  }

  /** The sequence the next successful append receives. */
  get nextSequence(): number {
    return this.#records.length;
  }

  /** Iterate the records retained when the iteration starts. */
  [Symbol.iterator](): Iterator<RecordEnvelope> {
    return [...this.#records][Symbol.iterator]();
  }

  records(): readonly RecordEnvelope[] {
    return Object.freeze([...this.#records]);
  }

  /**
   * Append `record` under `nextSequence` and return its retained envelope.
   *
   * @throws Error when the log is closed.
   * @throws RangeError, TypeError the envelope's and the codec's errors; the log is unchanged.
   */
  append(record: RunRecord): RecordEnvelope {
    if (this.#closed) throw new Error("run log is closed");
    const [, retained] = canonicalEnvelope(record, this.runId, this.nextSequence);
    this.#records.push(retained);
    return retained;
  }

  close(): void {
    this.#closed = true;
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

const UTF8_ENCODER = new TextEncoder();

/**
 * An exclusively locked, append-and-fsync JSONL log for one run.
 *
 * Built only by `create` or `resume`. The log holds the non-blocking exclusive lock on its
 * sibling `<path>.lock` for its whole life, and keeps its validated records in memory, so they
 * stay iterable after `close`.
 */
export class JsonlRunLog implements RunLog {
  /** The sole run id this log accepts. */
  readonly runId: string;
  /** The JSONL path, as given to `create` or `resume`. */
  readonly path: string;
  readonly #fd: number;
  readonly #lock: FileLock;
  readonly #records: RecordEnvelope[];
  #prefixSize: number;
  #closed = false;
  /**
   * The error of the failed append that poisoned the log, or `null`. The error is wrapped as
   * `{ error }` so even a thrown `undefined` poisons the log.
   */
  #failure: { error: unknown } | null = null;

  /** Adopt a locked descriptor and its validated complete prefix of `prefixSize` bytes. */
  private constructor(
    path: string,
    runId: string,
    fd: number,
    lock: FileLock,
    records: RecordEnvelope[],
    prefixSize: number,
  ) {
    this.path = path;
    this.runId = runId;
    this.#fd = fd;
    this.#lock = lock;
    this.#records = records;
    this.#prefixSize = prefixSize;
  }

  /**
   * Create a new owner-only (0600) log at `path` and lock it. An existing path is never replaced.
   * A failure after the file is created closes it and releases the lock; the empty file stays.
   *
   * @throws RangeError when `runId` isn't a non-empty string or holds a lone surrogate; nothing
   *   is created then.
   * @throws Error `run log <path> is locked by another writer (is the run still active?)`, with
   *   the busy `FileLockError` as `cause`.
   * @throws the `fs` error (`code` `EEXIST` for an existing path) and other `lockFile` errors.
   */
  static create(path: string, runId: string): JsonlRunLog {
    const canonicalRunId = validatedRunId(runId);
    const { O_CREAT, O_EXCL, O_RDWR } = fs.constants;
    const fd = fs.openSync(path, O_RDWR | O_CREAT | O_EXCL, 0o600);
    let lock: FileLock | undefined;
    try {
      // The open's mode is masked by the umask.
      fs.fchmodSync(fd, 0o600);
      lock = lockRunLog(path);
      return new JsonlRunLog(path, canonicalRunId, fd, lock, [], 0);
    } catch (error) {
      releaseAfterFailure(fd, lock);
      throw error;
    }
  }

  /**
   * Lock, repair and reopen the existing log at `path`.
   *
   * The lock is taken before the first read. Bytes after the last newline are never records,
   * even when they form valid JSON; they are truncated and synced only after the complete lines
   * decode as a non-empty log, so a rejected file is left unchanged. Any failure closes the file
   * and releases the lock.
   *
   * @throws RangeError when the file holds no complete line (`cannot resume an empty run log:
   *   <path>`) or a complete line isn't a valid record, or the records don't share one run id with
   *   contiguous sequences from 0 (the message names `path`, the line and its seq).
   * @throws Error the busy message, as `create`.
   * @throws the `fs` error (`code` `ENOENT` for a missing log, which creates no lock file) and
   *   other `lockFile` errors.
   */
  static resume(path: string): JsonlRunLog {
    const fd = fs.openSync(path, fs.constants.O_RDWR);
    let lock: FileLock | undefined;
    try {
      lock = lockRunLog(path);
      const data = fs.readFileSync(fd);
      const completeEnd = data.lastIndexOf(0x0a) + 1;
      const records = decodeCompletePrefix(data.subarray(0, completeEnd), path);
      if (records.length === 0) throw new RangeError(`cannot resume an empty run log: ${path}`);
      if (completeEnd !== data.length) {
        fs.ftruncateSync(fd, completeEnd);
        fs.fsyncSync(fd);
      }
      return new JsonlRunLog(path, records[0]!.runId, fd, lock, records, completeEnd);
    } catch (error) {
      releaseAfterFailure(fd, lock);
      throw error;
    }
  }

  /** The sequence the next successful append receives. */
  get nextSequence(): number {
    return this.#records.length;
  }

  /** Iterate the records retained when the iteration starts. */
  [Symbol.iterator](): Iterator<RecordEnvelope> {
    return [...this.#records][Symbol.iterator]();
  }

  records(): readonly RecordEnvelope[] {
    return Object.freeze([...this.#records]);
  }

  /**
   * Append `record` as one complete line under `nextSequence`, and retain it only after the
   * file is synced.
   *
   * A failed write or sync truncates the file back to the prior complete prefix and syncs it,
   * and poisons the log: every later append throws, and the lock stays held until `close`.
   *
   * @throws Error `run log is closed`; `run log <path> refuses appends after a failed append`,
   *   with the failed append's error as `cause`; `short run-log write: wrote K of N bytes` when
   *   a write makes no progress.
   * @throws RangeError, TypeError the envelope's and the codec's errors, before anything is
   *   written; they don't poison the log.
   * @throws the `fs` error of a failed write or sync, or, when the rollback fails too,
   *   `AggregateError` `run log <path> append failed and its rollback also failed` holding that
   *   error and the rollback's errors.
   */
  append(record: RunRecord): RecordEnvelope {
    if (this.#closed) throw new Error("run log is closed");
    if (this.#failure !== null) {
      throw new Error(`run log ${this.path} refuses appends after a failed append`, { cause: this.#failure.error });
    }
    const [line, retained] = canonicalEnvelope(record, this.runId, this.nextSequence);
    const bytes = UTF8_ENCODER.encode(`${line}\n`);
    try {
      for (let written = 0; written < bytes.length; ) {
        const count = fs.writeSync(this.#fd, bytes, written, bytes.length - written, this.#prefixSize + written);
        // A write that makes no progress would loop forever.
        if (count === 0) throw new Error(`short run-log write: wrote ${written} of ${bytes.length} bytes`);
        written += count;
      }
      fs.fsyncSync(this.#fd);
    } catch (error) {
      // Poisoned even after a clean rollback: the failed append ends the run as a host error,
      // and a failed rollback leaves a tail only a resume repairs.
      this.#failure = { error };
      const rollbackErrors = rollbackAppend(this.#fd, this.#prefixSize);
      if (rollbackErrors.length > 0) {
        throw new AggregateError(
          [error, ...rollbackErrors],
          `run log ${this.path} append failed and its rollback also failed`,
        );
      }
      throw error;
    }
    this.#records.push(retained);
    this.#prefixSize += bytes.length;
    return retained;
  }

  /** Close the file and release the lock once; the records stay iterable. */
  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    release(this.#fd, this.#lock);
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

/**
 * Read a run log's complete prefix: every newline-terminated record, without taking the writer's
 * lock and without changing the file. An incomplete last line is ignored; an empty file or one
 * without a complete line gives no records.
 *
 * @throws RangeError when a complete line isn't a valid record, or the records don't share one
 *   run id with contiguous sequences from 0; the message names `path`, the line and its seq.
 */
export function readRunLog(path: string): RecordEnvelope[] {
  const data = fs.readFileSync(path);
  return decodeCompletePrefix(data.subarray(0, data.lastIndexOf(0x0a) + 1), path);
}

/** Return whether a writer (in any process) holds the lock of the log at `path`. */
export function isRunLogLocked(path: string): boolean {
  return isLocked(`${path}.lock`);
}

/** Return a failure's wire object, shared by the failure-bearing records. */
function failurePayload(failure: Failure): JsonObject {
  return failure.toJSON();
}

/**
 * Rebuild a `Failure` from its wire object; `label` names it in errors. Its node id must be a
 * non-empty string, or any string or `null` when `optionalNodeId` is set (a backward failure's
 * cause).
 *
 * @throws RangeError for missing or extra fields, an empty required node id or origin key, or an
 *   unknown code.
 * @throws TypeError when a field has the wrong JSON type.
 */
function failureFromPayload(value: unknown, label: string, optionalNodeId = false): Failure {
  const wire = wireObject(value, label, ["code", "message", "node_id", "trace", "cause_type", "traceback"]);
  const code = wireText(wire["code"], `${label} code`);
  const message = wireText(wire["message"], `${label} message`);
  const nodeLabel = `${label} node_id`;
  const nodeId = optionalNodeId ? wireNullableText(wire["node_id"], nodeLabel) : wireName(wire["node_id"], nodeLabel);
  const trace = wireArray(wire["trace"], `${label} trace`).map((item) => {
    const origin = wireObject(item, `${label} trace item`, ["key"]);
    return new OriginRef(wireName(origin["key"], `${label} trace item key`));
  });
  return new Failure({
    code: code as Failure["code"],
    message,
    nodeId,
    trace,
    causeType: wireNullableText(wire["cause_type"], `${label} cause_type`),
    traceback: wireNullableText(wire["traceback"], `${label} traceback`),
  });
}

/** Encode one dynamic occurrence segment of a memory scope. */
function occurrenceSegmentPayload(segment: MemoryOccurrenceSegment): JsonObject {
  return { kind: segment.kind, static_path: [...segment.staticPath], index: BigInt(segment.index) };
}

/** The closed occurrence segment kinds. */
const OCCURRENCE_KINDS: readonly string[] = Object.values(MemoryOccurrenceKind);

/**
 * Decode one occurrence segment; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields, an unknown kind or an index beyond ±(2^53 - 1),
 *   besides the segment's own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function occurrenceSegmentFromPayload(value: unknown, label: string): MemoryOccurrenceSegment {
  const wire = wireObject(value, label, ["kind", "static_path", "index"]);
  const kind = wireText(wire["kind"], `${label} kind`);
  // The segment refuses an unknown kind as a TypeError; on the wire it is a RangeError.
  if (!OCCURRENCE_KINDS.includes(kind)) {
    throw new RangeError(`${label} kind must be one of ${writeAsText([...OCCURRENCE_KINDS])}, got ${reprText(kind)}`);
  }
  return new MemoryOccurrenceSegment(
    kind as MemoryOccurrenceKind,
    wireTextArray(wire["static_path"], `${label} static_path`),
    wireInt(wire["index"], `${label} index`),
  );
}

/** Encode one compiled node's memory scope. */
function scopePayload(scope: MemoryScope): JsonObject {
  return {
    static_path: [...scope.staticPath],
    memories: [...scope.memories],
    occurrence_path: scope.occurrencePath.map(occurrenceSegmentPayload),
  };
}

/**
 * Decode one memory scope; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields, besides the segments' and scope's own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function scopeFromPayload(value: unknown, label: string): MemoryScope {
  const wire = wireObject(value, label, ["static_path", "memories", "occurrence_path"]);
  return new MemoryScope(wireTextArray(wire["static_path"], `${label} static_path`), {
    memories: wireTextArray(wire["memories"], `${label} memories`),
    occurrencePath: wireItems(wire["occurrence_path"], `${label} occurrence_path`, occurrenceSegmentFromPayload),
  });
}

/**
 * Return memory JSON as the log writes it. Memory's carriers keep `-0`, which canonical JSON
 * refuses, so it is written as `0`.
 */
function memoryJson(value: unknown): JsonValue {
  return jsonPart(value, [], "refuse") as JsonValue;
}

/** Encode one named memory snapshot. */
function snapshotPayload(snapshot: MemorySnapshotEnvelope): JsonObject {
  return { memory: snapshot.memory, snapshot: memoryJson(snapshot.snapshot) };
}

/**
 * Decode one named memory snapshot; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields or an empty memory key, besides the envelope's
 *   own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function snapshotFromPayload(value: unknown, label: string): MemorySnapshotEnvelope {
  const wire = wireObject(value, label, ["memory", "snapshot"]);
  return new MemorySnapshotEnvelope(
    wireName(wire["memory"], `${label} memory`),
    wireObject(wire["snapshot"], `${label} snapshot`),
  );
}

/** Encode one memory key-table entry. */
function memorySpecPayload(spec: MemorySpec): JsonObject {
  return {
    key: spec.key,
    type: spec.type,
    config: memoryJson(spec.config),
    owners: spec.owners.map((site) => ({ scope: [...site.scope], owner: [...site.owner], node: site.node })),
    conversation_eligible: spec.conversationEligible,
  };
}

/**
 * Decode one memory key-table entry; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields or an empty key or type, besides the sites' and
 *   spec's own errors (a private config key among them).
 * @throws TypeError when a field has the wrong JSON type.
 */
function memorySpecFromPayload(value: unknown, label: string): MemorySpec {
  const wire = wireObject(value, label, ["key", "type", "config", "owners", "conversation_eligible"]);
  const owners = wireArray(wire["owners"], `${label} owners`).map((item) => {
    const siteLabel = `${label} owners item`;
    const site = wireObject(item, siteLabel, ["scope", "owner", "node"]);
    return new DeclSite(
      wireTextArray(site["scope"], `${siteLabel} scope`),
      wireTextArray(site["owner"], `${siteLabel} owner`),
      wireNullableText(site["node"], `${siteLabel} node`),
    );
  });
  return new MemorySpec(
    wireName(wire["key"], `${label} key`),
    wireName(wire["type"], `${label} type`),
    wireObject(wire["config"], `${label} config`),
    owners,
    wireBool(wire["conversation_eligible"], `${label} conversation_eligible`),
  );
}

/** Encode one target's memory key stack. */
function memoryStackPayload(stack: MemoryStack): JsonObject {
  return { locator: [...stack.locator], memories: [...stack.memories] };
}

/**
 * Decode one target's memory key stack; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields, besides the stack's own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function memoryStackFromPayload(value: unknown, label: string): MemoryStack {
  const wire = wireObject(value, label, ["locator", "memories"]);
  return new MemoryStack(
    wireTextArray(wire["locator"], `${label} locator`),
    wireTextArray(wire["memories"], `${label} memories`),
  );
}

/** Encode one recalled-memory section. */
function memorySectionPayload(section: MemorySection): JsonObject {
  return { memory: section.memory, text: section.text };
}

/**
 * Decode one recalled-memory section; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields or an empty memory key.
 * @throws TypeError when a field has the wrong JSON type.
 */
function memorySectionFromPayload(value: unknown, label: string): MemorySection {
  const wire = wireObject(value, label, ["memory", "text"]);
  return new MemorySection(wireName(wire["memory"], `${label} memory`), wireText(wire["text"], `${label} text`));
}

/** Encode one member of the closed visible-interaction vocabulary. */
function interactionPayload(interaction: VisibleInteraction): JsonObject {
  // Memory's JSON types are the read-only form of the same frozen values.
  if (interaction instanceof ModelTextInteraction) return { kind: "model_text", text: interaction.text };
  if (interaction instanceof ModelValueInteraction) {
    return { kind: "model_value", value: interaction.value as JsonValue };
  }
  if (interaction instanceof ToolCallInteraction) {
    return { kind: "tool_call", name: interaction.name, arguments: interaction.arguments as JsonObject };
  }
  return { kind: "tool_result", name: interaction.name, text: interaction.text };
}

/**
 * Decode one visible interaction of a closed kind; `label` names it in errors.
 *
 * @throws RangeError for an unknown kind, missing or extra fields or an empty tool name, besides
 *   the interaction's own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function interactionFromPayload(value: unknown, label: string): VisibleInteraction {
  const wire = wireObject(value, label);
  // The other fields depend on the kind, so a missing kind is named before anything else.
  if (!Object.hasOwn(wire, "kind")) requireFields(wire, ["kind"], label);
  const kind = wireText(wire["kind"], `${label} kind`);
  switch (kind) {
    case "model_text":
      requireFields(wire, ["kind", "text"], label);
      return new ModelTextInteraction(wireText(wire["text"], `${label} text`));
    case "model_value":
      requireFields(wire, ["kind", "value"], label);
      return new ModelValueInteraction(wire["value"]);
    case "tool_call":
      requireFields(wire, ["kind", "name", "arguments"], label);
      return new ToolCallInteraction(
        wireName(wire["name"], `${label} name`),
        wireObject(wire["arguments"], `${label} arguments`),
      );
    case "tool_result":
      requireFields(wire, ["kind", "name", "text"], label);
      return new ToolResultInteraction(
        wireName(wire["name"], `${label} name`),
        wireText(wire["text"], `${label} text`),
      );
    default:
      throw new RangeError(`unknown visible interaction kind: ${reprText(kind)}`);
  }
}

/** Encode a route path: field names, and indexes as int tokens. */
function routePathPayload(path: RoutePath): JsonValue[] {
  return path.map((part) => (typeof part === "string" ? part : BigInt(part)));
}

/**
 * Decode a route or source path; `label` names it in errors. Each index is checked by
 * `validateRoute` or `validateSource`.
 *
 * @throws TypeError when `value` isn't an array.
 * @throws RangeError when a part is neither a string nor an int token.
 */
function routePathFromPayload(value: unknown, label: string): (string | number)[] {
  return wireArray(value, label).map((part) => {
    if (typeof part === "string") return part;
    if (typeof part === "bigint") return Number(part);
    throw new RangeError("route path components must be strings or non-negative integers");
  });
}

/** Encode a source identity without its produced value. */
function sourcePayload(source: SourceRef): JsonObject {
  return { run_id: source.runId, execution_seq: BigInt(source.executionSeq), path: routePathPayload(source.path) };
}

/**
 * Decode and check one source identity; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields, besides `validateSource`'s errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function sourceFromPayload(value: unknown, label: string): SourceRef {
  const wire = wireObject(value, label, ["run_id", "execution_seq", "path"]);
  const source = new SourceRef(
    wireText(wire["run_id"], `${label} run_id`),
    wireInt(wire["execution_seq"], `${label} execution_seq`),
    routePathFromPayload(wire["path"], `${label} path`),
  );
  validateSource(source);
  return source;
}

/**
 * Encode an authored execution scope independent of memory, or `null` for none.
 *
 * @throws TypeError when `scope` isn't `null` or exactly an `ExecutionScope`.
 */
export function executionScopePayload(scope: ExecutionScope | null): JsonObject | null {
  if (scope === null) return null;
  if (!isExactly(scope, ExecutionScope)) throw new TypeError("scope must be an exact ExecutionScope");
  return {
    static_path: [...scope.staticPath],
    flow_path: [...scope.flowPath],
    occurrence_path: scope.occurrencePath.map((position) => ({
      kind: position.kind,
      static_path: [...position.staticPath],
      index: position.index === null ? null : BigInt(position.index),
    })),
    boundary: scope.boundary,
  };
}

/**
 * Decode an execution scope, or `null`; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields or an index beyond ±(2^53 - 1), besides the
 *   positions' and scope's own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function executionScopeFromPayload(value: unknown, label: string): ExecutionScope | null {
  if (value === null) return null;
  const wire = wireObject(value, label, ["static_path", "flow_path", "occurrence_path", "boundary"]);
  const occurrencePath = wireArray(wire["occurrence_path"], `${label} occurrence_path`).map((item) => {
    const itemLabel = `${label} occurrence_path item`;
    const position = wireObject(item, itemLabel, ["kind", "static_path", "index"]);
    const index = position["index"];
    return new ExecutionPosition(
      wireText(position["kind"], `${itemLabel} kind`) as PositionKind,
      wireTextArray(position["static_path"], `${itemLabel} static_path`),
      index === null ? null : wireInt(index, `${itemLabel} index`),
    );
  });
  return new ExecutionScope(wireTextArray(wire["static_path"], `${label} static_path`), {
    flowPath: wireTextArray(wire["flow_path"], `${label} flow_path`),
    occurrencePath,
    boundary: wireNullableText(wire["boundary"], `${label} boundary`) as ExecutionScope["boundary"],
  });
}

/**
 * Encode a route after checking its whole tree once.
 *
 * @throws TypeError when a route field has the wrong type.
 * @throws RangeError when a route field or variant breaks the route contract.
 * @throws JSONDecodeError when a `valueJson` isn't JSON yian reads.
 */
export function routePayload(route: ValueRoute): JsonObject {
  validateRoute(route);
  return encodeRoute(route);
}

/** Encode a route that `validateRoute` already accepted, without checking it again. */
function encodeRoute(route: ValueRoute): JsonObject {
  return {
    kind: route.kind,
    source: route.source === null ? null : sourcePayload(route.source),
    value_json: route.valueJson,
    children: route.children.map(encodeRoute),
    expression: route.expression,
    names: [...route.names],
    sizes: route.sizes.map(BigInt),
    spans: route.spans.map(([start, end]) => [BigInt(start), BigInt(end)]),
    path: routePathPayload(route.path),
    result_type: route.resultType,
    snapshot: route.snapshot === null ? null : encodeRoute(route.snapshot),
  };
}

/** The fields of a route payload. */
const ROUTE_WIRE_FIELDS = [
  "kind",
  "source",
  "value_json",
  "children",
  "expression",
  "names",
  "sizes",
  "spans",
  "path",
  "result_type",
  "snapshot",
];

/**
 * Decode a route payload, then check its variant.
 *
 * Every field's JSON type is checked first: `ValueRoute`'s constructor would reshape a string or
 * a span of another length, and would turn a null `expression` into `""`; checking the scalar
 * fields here also reports a node's own shape errors before its children's. Then each child and
 * the snapshot are decoded and checked by their own call, so the new node is checked without its
 * subtree and each node once.
 *
 * @throws TypeError when a field has the wrong JSON type, or for `validateRoute`'s type errors.
 * @throws RangeError for missing or extra fields, or when a field or variant breaks the route
 *   contract.
 * @throws JSONDecodeError when a `value_json` isn't JSON yian reads.
 */
export function routeFromPayload(payload: unknown): ValueRoute {
  const wire = wireObject(payload, "route", ROUTE_WIRE_FIELDS);
  const { source, children, snapshot } = wire;
  if (source !== null && !isRecord(source)) throw new TypeError("route source must be an object or null");
  const valueJson = wire["value_json"];
  if (valueJson !== null && typeof valueJson !== "string") {
    throw new TypeError("route value_json must be a string or null");
  }
  const childWires = wireArray(children, "route children");
  const expression = wire["expression"];
  if (typeof expression !== "string") throw new TypeError("route expression must be a string");
  const names = wireArray(wire["names"], "route names");
  if (!names.every((name): name is string => typeof name === "string")) {
    throw new RangeError("route names must be strings");
  }
  const sizes = wireArray(wire["sizes"], "route sizes").map((size) => {
    if (typeof size !== "bigint") throw new RangeError("route sizes must be non-negative integers");
    return Number(size);
  });
  const spans = wireArray(wire["spans"], "route spans").map((span) => {
    if (!isList(span) || span.length !== 2 || !span.every((bound) => typeof bound === "bigint")) {
      throw new TypeError("interpolation spans must be integer pairs");
    }
    return [Number(span[0]), Number(span[1])] as const;
  });
  const path = routePathFromPayload(wire["path"], "route path");
  const resultType = wire["result_type"];
  if (resultType !== null && (typeof resultType !== "string" || !resultType)) {
    throw new RangeError("route result_type must be a non-empty string or null");
  }
  if (snapshot !== null && !isRecord(snapshot)) throw new TypeError("route snapshot must be an object or null");
  const route = new ValueRoute(wire["kind"] as RouteKind, {
    source: source === null ? null : sourceFromPayload(source, "route source"),
    valueJson,
    children: childWires.map(routeFromPayload),
    expression,
    names,
    sizes,
    spans,
    path,
    resultType,
    snapshot: snapshot === null ? null : routeFromPayload(snapshot),
  });
  validateRoute(route, { deep: false });
  return route;
}

/**
 * Encode a forward capture's binding data; its route is checked once.
 *
 * @throws RangeError when an evaluated key names no captured route, besides `routePayload`'s
 *   errors.
 */
function capturePayload(capture: ForwardCapture): JsonObject {
  const route = routePayload(capture.route);
  // `evaluatedKeys` is a `Set`, which stays mutable inside a frozen capture, so it is the one
  // field checked again.
  const keys = [...capture.evaluatedKeys];
  if (!keys.every((key) => capture.route.names.includes(key))) {
    throw new RangeError("evaluated_keys must name captured routes");
  }
  return {
    bound_inputs: capture.boundInputs as JsonObject | null,
    route,
    evaluated_keys: keys.sort(compareCodeUnits),
    items: capture.items.map(([index, item]) => ({ index: BigInt(index), capture: capturePayload(item) })),
    scope: executionScopePayload(capture.scope),
    node_data: capture.nodeData as JsonObject | null,
    input_names: [...capture.inputNames],
    variable_names: [...capture.variableNames],
  };
}

/**
 * Decode one forward capture; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields, evaluated keys that aren't sorted and unique,
 *   or an index beyond ±(2^53 - 1), besides the route's, scope's and capture's own errors.
 * @throws TypeError when a field has the wrong JSON type.
 * @throws JSONDecodeError when a route's `value_json` isn't JSON yian reads.
 */
function captureFromPayload(value: unknown, label: string): ForwardCapture {
  const wire = wireObject(value, label, [
    "bound_inputs",
    "route",
    "evaluated_keys",
    "items",
    "scope",
    "node_data",
    "input_names",
    "variable_names",
  ]);
  const keys = wireTextArray(wire["evaluated_keys"], `${label} evaluated_keys`);
  if (!strictlyIncreasing(keys, compareCodeUnits)) throw new RangeError("evaluated_keys must be sorted and unique");
  const items = wireArray(wire["items"], `${label} items`).map((item) => {
    const pair = wireObject(item, `${label} item`, ["index", "capture"]);
    const index = wireInt(pair["index"], `${label} item index`);
    return [index, captureFromPayload(pair["capture"], `${label} item capture`)] as const;
  });
  const boundInputs = wireNullableObject(wire["bound_inputs"], `${label} bound_inputs`);
  return new ForwardCapture(boundInputs, routeFromPayload(wire["route"]), {
    evaluatedKeys: new Set(keys),
    items,
    scope: executionScopeFromPayload(wire["scope"], `${label} scope`),
    nodeData: wireNullableObject(wire["node_data"], `${label} node_data`),
    inputNames: wireTextArray(wire["input_names"], `${label} input_names`),
    variableNames: wireTextArray(wire["variable_names"], `${label} variable_names`),
  });
}

/**
 * Encode optional evaluation evidence, or `null` for none.
 *
 * @throws TypeError when `evidence` isn't `null` or exactly an `EvaluationEvidence`.
 */
export function evidencePayload(evidence: EvaluationEvidence | null): JsonObject | null {
  if (evidence === null) return null;
  if (!isExactly(evidence, EvaluationEvidence)) throw new TypeError("evidence must be an EvaluationEvidence or null");
  return {
    scope: evidence.scope === null ? null : scopePayload(evidence.scope),
    bound_inputs: evidence.boundInputs as JsonObject | null,
    base_system_prompt: evidence.baseSystemPrompt,
    prompt: evidence.prompt,
    memory_sections: evidence.memorySections.map(memorySectionPayload),
    interactions: evidence.interactions.map(interactionPayload),
  };
}

/**
 * Decode one evaluation evidence; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields or an unknown interaction kind, besides the
 *   carriers' own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function evidenceFromPayload(value: unknown, label: string): EvaluationEvidence {
  const wire = wireObject(value, label, [
    "scope",
    "bound_inputs",
    "base_system_prompt",
    "prompt",
    "memory_sections",
    "interactions",
  ]);
  return new EvaluationEvidence(
    wire["scope"] === null ? null : scopeFromPayload(wire["scope"], `${label} scope`),
    wireNullableObject(wire["bound_inputs"], `${label} bound_inputs`),
    wireNullableText(wire["base_system_prompt"], `${label} base_system_prompt`),
    wireNullableText(wire["prompt"], `${label} prompt`),
    wireItems(wire["memory_sections"], `${label} memory_sections`, memorySectionFromPayload),
    wireItems(wire["interactions"], `${label} interactions`, interactionFromPayload),
  );
}

/** Encode one memory mutation. */
function mutationPayload(mutation: MemoryMutation): JsonObject {
  return { mutation_id: mutation.mutationId, payload: memoryJson(mutation.payload) };
}

/**
 * Decode one memory mutation; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields or an empty id, besides the mutation's own
 *   errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function mutationFromPayload(value: unknown, label: string): MemoryMutation {
  const wire = wireObject(value, label, ["mutation_id", "payload"]);
  return new MemoryMutation(wireName(wire["mutation_id"], `${label} mutation_id`), wire["payload"]);
}

/** Encode one sanitized memory-operation failure. */
function memoryFailurePayload(failure: MemoryOperationFailure): JsonObject {
  return { stage: failure.stage, code: failure.code, message: failure.message, retryable: failure.retryable };
}

/**
 * Decode one sanitized memory-operation failure; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields or an empty string field.
 * @throws TypeError when a field has the wrong JSON type.
 */
function memoryFailureFromPayload(value: unknown, label: string): MemoryOperationFailure {
  const wire = wireObject(value, label, ["stage", "code", "message", "retryable"]);
  return new MemoryOperationFailure(
    wireName(wire["stage"], `${label} stage`),
    wireName(wire["code"], `${label} code`),
    wireName(wire["message"], `${label} message`),
    wireBool(wire["retryable"], `${label} retryable`),
  );
}

/** Encode the three channels of a backward result, each batch in its order. */
function backwardResultPayload(result: BackwardResult): JsonObject {
  return {
    input_feedback: result.inputFeedback as JsonObject,
    self_feedback: [...result.selfFeedback],
    variable_feedback: result.variableFeedback as JsonObject,
  };
}

/**
 * Decode a backward result through its constructor; `label` names it in errors. Messages are
 * ordinary feedback, not memory JSON.
 *
 * @throws RangeError for missing or extra fields, besides the result's own errors.
 * @throws TypeError when a channel has the wrong JSON type, besides the result's own errors.
 */
function backwardResultFromPayload(value: unknown, label: string): BackwardResult {
  const wire = wireObject(value, label, ["input_feedback", "self_feedback", "variable_feedback"]);
  return new BackwardResult({
    inputFeedback: wireObject(wire["input_feedback"], `${label} input_feedback`) as Record<string, FeedbackValue[]>,
    selfFeedback: wireArray(wire["self_feedback"], `${label} self_feedback`) as string[],
    variableFeedback: wireObject(wire["variable_feedback"], `${label} variable_feedback`) as Record<string, string[]>,
  });
}

/** Encode one message's transfer coordinates, without the downstream context they point at. */
function attributionPayload(item: BackwardAttribution): JsonObject {
  return {
    sender: item.sender === null ? null : sourcePayload(item.sender),
    input_name: item.inputName,
    message_index: BigInt(item.messageIndex),
    contribution_index: item.contributionIndex === null ? null : BigInt(item.contributionIndex),
    parents: item.parents.map((parent) => BigInt(parent)),
  };
}

/**
 * Decode one message's coordinates through `BackwardAttribution`; `label` names it in errors.
 * Agreement with the trace belongs to the reducer.
 *
 * @throws RangeError for missing or extra fields or a counter beyond ±(2^53 - 1), besides the
 *   source's and the attribution's own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function attributionFromPayload(value: unknown, label: string): BackwardAttribution {
  const wire = wireObject(value, label, ["sender", "input_name", "message_index", "contribution_index", "parents"]);
  const sender = wire["sender"];
  const contribution = wire["contribution_index"];
  return new BackwardAttribution(
    sender === null ? null : sourceFromPayload(sender, `${label} sender`),
    wireNullableText(wire["input_name"], `${label} input_name`),
    wireInt(wire["message_index"], `${label} message_index`),
    contribution === null ? null : wireInt(contribution, `${label} contribution_index`),
    wireArray(wire["parents"], `${label} parents`).map((parent) => wireInt(parent, `${label} parents item`)),
  );
}

/** Encode one occurrence's received feedback, its attribution and its callback result. */
function attributedResultPayload(item: AttributedBackwardResult): JsonObject {
  return {
    source: sourcePayload(item.source),
    feedback: [...item.feedback] as JsonValue[],
    attribution: item.attribution.map(attributionPayload),
    result: backwardResultPayload(item.result),
    invoked: item.invoked,
  };
}

/**
 * Decode one occurrence's result through `AttributedBackwardResult`, without rebuilding an
 * invocation or a route; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields, besides the nested carriers' own errors.
 * @throws TypeError when a field has the wrong JSON type, besides the nested carriers' own errors.
 */
function attributedResultFromPayload(value: unknown, label: string): AttributedBackwardResult {
  const wire = wireObject(value, label, ["source", "feedback", "attribution", "result", "invoked"]);
  return new AttributedBackwardResult(
    sourceFromPayload(wire["source"], `${label} source`),
    wireArray(wire["feedback"], `${label} feedback`) as FeedbackValue[],
    wireItems(wire["attribution"], `${label} attribution`, attributionFromPayload),
    backwardResultFromPayload(wire["result"], `${label} result`),
    wireBool(wire["invoked"], `${label} invoked`),
  );
}

/** Encode one memory's proposed mutation batch, in order. */
function memoryProposalPayload(proposal: MemoryProposal): JsonObject {
  return { memory: proposal.memory, mutations: proposal.mutations.map(mutationPayload) };
}

/**
 * Decode one memory's proposal through `MemoryProposal`; `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields, an empty memory key or two mutations with one id.
 * @throws TypeError when a field has the wrong JSON type.
 */
function memoryProposalFromPayload(value: unknown, label: string): MemoryProposal {
  const wire = wireObject(value, label, ["memory", "mutations"]);
  return new MemoryProposal(
    wireName(wire["memory"], `${label} memory`),
    wireItems(wire["mutations"], `${label} mutations`, mutationFromPayload),
  );
}

/** Encode a learning result's failures and per-memory commit dispositions, in order. */
function learningResultPayload(result: LearningResult): JsonObject {
  return {
    status: result.status,
    failures: result.failures.map((item) => ({
      memory: item.memory,
      operation: item.operation,
      failure: memoryFailurePayload(item.failure),
    })),
    policies: result.policies.map((item) => ({
      memory: item.memory,
      status: item.status,
      commit_id: item.commitId,
      failure: item.failure === null ? null : memoryFailurePayload(item.failure),
    })),
  };
}

/**
 * Decode a learning result through its carriers, without inferring the attempt's completion;
 * `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields or an empty memory key, besides the carriers'
 *   own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function learningResultFromPayload(value: unknown, label: string): LearningResult {
  const wire = wireObject(value, label, ["status", "failures", "policies"]);
  const failures = wireArray(wire["failures"], `${label} failures`).map((item) => {
    const itemLabel = `${label} failures item`;
    const failure = wireObject(item, itemLabel, ["memory", "operation", "failure"]);
    return new LearningFailure(
      wireName(failure["memory"], `${itemLabel} memory`),
      wireText(failure["operation"], `${itemLabel} operation`) as LearningOperation,
      memoryFailureFromPayload(failure["failure"], `${itemLabel} failure`),
    );
  });
  const policies = wireArray(wire["policies"], `${label} policies`).map((item) => {
    const itemLabel = `${label} policies item`;
    const policy = wireObject(item, itemLabel, ["memory", "status", "commit_id", "failure"]);
    const failure = policy["failure"];
    return new PolicyCommitOutcome(
      wireName(policy["memory"], `${itemLabel} memory`),
      wireText(policy["status"], `${itemLabel} status`) as PolicyCommitStatus,
      {
        commitId: wireNullableText(policy["commit_id"], `${itemLabel} commit_id`),
        failure: failure === null ? null : memoryFailureFromPayload(failure, `${itemLabel} failure`),
      },
    );
  });
  return new LearningResult(wireText(wire["status"], `${label} status`) as LearningStatus, { failures, policies });
}

/** Encode one attributed backward diagnostic and its `Failure` or memory-operation cause. */
function backwardFailurePayload(failure: BackwardFailure): JsonObject {
  const { cause } = failure;
  return {
    stage: failure.stage,
    code: failure.code,
    message: failure.message,
    scope: executionScopePayload(failure.scope),
    run_id: failure.runId,
    node_id: failure.nodeId,
    node_started_seq: failure.nodeStartedSeq === null ? null : BigInt(failure.nodeStartedSeq),
    memory: failure.memory,
    cause: cause === null ? null : cause instanceof Failure ? failurePayload(cause) : memoryFailurePayload(cause),
    cause_type: failure.causeType,
    traceback: failure.traceback,
  };
}

/** The fields of a backward failure's wire object. */
const BACKWARD_FAILURE_FIELDS = [
  "stage",
  "code",
  "message",
  "scope",
  "run_id",
  "node_id",
  "node_started_seq",
  "memory",
  "cause",
  "cause_type",
  "traceback",
];

/**
 * Decode one backward diagnostic through `BackwardFailure`; `label` names it in errors. A cause
 * with a `stage` field is a memory-operation failure, any other a `Failure` (the two shapes are
 * disjoint, so nothing is inferred).
 *
 * @throws RangeError for missing or extra fields or a counter beyond ±(2^53 - 1), besides the
 *   nested carriers' own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function backwardFailureFromPayload(value: unknown, label: string): BackwardFailure {
  const wire = wireObject(value, label, BACKWARD_FAILURE_FIELDS);
  const causeLabel = `${label} cause`;
  const cause = wireNullableObject(wire["cause"], causeLabel);
  let decodedCause: Failure | MemoryOperationFailure | null = null;
  if (cause !== null) {
    decodedCause = Object.hasOwn(cause, "stage")
      ? memoryFailureFromPayload(cause, causeLabel)
      : failureFromPayload(cause, causeLabel, true);
  }
  const seq = wire["node_started_seq"];
  return new BackwardFailure(
    wireText(wire["stage"], `${label} stage`),
    wireText(wire["code"], `${label} code`),
    wireText(wire["message"], `${label} message`),
    {
      scope: executionScopeFromPayload(wire["scope"], `${label} scope`),
      runId: wireNullableText(wire["run_id"], `${label} run_id`),
      nodeId: wireNullableText(wire["node_id"], `${label} node_id`),
      nodeStartedSeq: seq === null ? null : wireInt(seq, `${label} node_started_seq`),
      memory: wireNullableText(wire["memory"], `${label} memory`),
      cause: decodedCause,
      causeType: wireNullableText(wire["cause_type"], `${label} cause_type`),
      traceback: wireNullableText(wire["traceback"], `${label} traceback`),
    },
  );
}

/** Encode a backward outcome with its optional feedback, failure and learning result. */
function backwardOutcomePayload(outcome: BackwardOutcome): JsonObject {
  return {
    status: outcome.status,
    feedback: outcome.feedback === null ? null : backwardResultPayload(outcome.feedback),
    failure: outcome.failure === null ? null : backwardFailurePayload(outcome.failure),
    learning: outcome.learning === null ? null : learningResultPayload(outcome.learning),
  };
}

/**
 * Decode a backward outcome through `BackwardOutcome`, without fabricating missing computation;
 * `label` names it in errors.
 *
 * @throws RangeError for missing or extra fields, besides the nested carriers' own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function backwardOutcomeFromPayload(value: unknown, label: string): BackwardOutcome {
  const wire = wireObject(value, label, ["status", "feedback", "failure", "learning"]);
  const { feedback, failure, learning } = wire;
  return new BackwardOutcome(wireText(wire["status"], `${label} status`) as BackwardStatus, {
    feedback: feedback === null ? null : backwardResultFromPayload(feedback, `${label} feedback`),
    failure: failure === null ? null : backwardFailureFromPayload(failure, `${label} failure`),
    learning: learning === null ? null : learningResultFromPayload(learning, `${label} learning`),
  });
}

/** Encode an outcome record's optional evidence and capture. */
function outcomePayload(record: OutcomeCarriers): JsonObject {
  const { evidence = null, forwardCapture = null } = record;
  return {
    evidence: evidencePayload(evidence),
    forward_capture: forwardCapture === null ? null : capturePayload(forwardCapture),
  };
}

/**
 * Decode an outcome payload's `evidence` and `forward_capture`; `label` names the payload.
 *
 * @throws RangeError or TypeError as `evidenceFromPayload` and `captureFromPayload`.
 */
function outcomeFromPayload(payload: JsonObject, label: string): OutcomeCarriers {
  const evidence = payload["evidence"];
  const capture = payload["forward_capture"];
  return {
    evidence: evidence === null ? null : evidenceFromPayload(evidence, `${label} evidence`),
    forwardCapture: capture === null ? null : captureFromPayload(capture, `${label} forward_capture`),
  };
}

/** The fields of a `run_started` payload; schema 10 writes every one. */
const RUN_STARTED_FIELDS = [
  "flow_fingerprint",
  "initial_pool",
  "start_wiring_override",
  "effective_llm_by_static_path",
  "num_workers",
  "memory_snapshots",
  "learnable",
  "memories",
  "memory_stacks",
  "defaults",
  "call_time_llm",
];

/** Build the canonical JSON payload of one record; counters are written as `bigint`s. */
function payloadForRecord(record: RunRecord): JsonObject {
  if (record instanceof RunStartedRecord) {
    return {
      flow_fingerprint: record.flowFingerprint,
      initial_pool: record.initialPool as unknown as JsonObject,
      start_wiring_override: record.startWiringOverride as JsonObject,
      effective_llm_by_static_path: record.effectiveLlmByStaticPath as JsonObject,
      num_workers: BigInt(record.numWorkers),
      memory_snapshots: record.memorySnapshots === null ? null : record.memorySnapshots.map(snapshotPayload),
      learnable: record.learnable,
      memories: record.memories === null ? null : record.memories.map(memorySpecPayload),
      memory_stacks: record.memoryStacks === null ? null : record.memoryStacks.map(memoryStackPayload),
      // The writer sorts every object's keys, so the defaults come out sorted by name.
      defaults: record.defaults as JsonObject,
      call_time_llm: record.callTimeLlm as JsonObject | null,
    };
  }
  if (record instanceof RunResumedRecord) {
    return { retry_node_ids: [...record.retryNodeIds], num_workers: BigInt(record.numWorkers) };
  }
  if (record instanceof RunStoppedRecord) {
    return { reason: record.reason, message: record.message, node_ids: [...record.nodeIds] };
  }
  if (record instanceof NodeStartedRecord) return { node_id: record.nodeId };
  if (record instanceof NodeStepRecord) return { node_id: record.nodeId, step: record.step as JsonObject };
  if (record instanceof NodeSucceededRecord) {
    return {
      node_id: record.nodeId,
      commit_target: record.commitTarget,
      value: record.value as unknown as JsonObject,
      ...outcomePayload(record),
    };
  }
  if (record instanceof NodeRoutedRecord) {
    return {
      node_id: record.nodeId,
      handle: record.handle,
      terminal_failure: record.terminalFailure === null ? null : failurePayload(record.terminalFailure),
      ...outcomePayload(record),
    };
  }
  if (record instanceof NodeExpandedRecord) {
    return {
      spawner_id: record.spawnerId,
      seed: record.seed,
      prune_ids: [...record.pruneIds],
      topology_digest: record.topologyDigest,
      ...outcomePayload(record),
    };
  }
  if (record instanceof NodeFailedRecord) return { failure: failurePayload(record.failure), ...outcomePayload(record) };
  if (record instanceof MemoryCommitRequestedRecord) {
    return {
      memory: record.memory,
      commit_id: record.commitId,
      attempt_id: BigInt(record.attemptId),
      mutations: record.mutations.map(mutationPayload),
    };
  }
  if (record instanceof MemoryCommitSucceededRecord) {
    return { memory: record.memory, commit_id: record.commitId, attempt_id: BigInt(record.attemptId) };
  }
  if (record instanceof MemoryCommitFailedRecord) {
    return {
      memory: record.memory,
      commit_id: record.commitId,
      attempt_id: BigInt(record.attemptId),
      failure: memoryFailurePayload(record.failure),
    };
  }
  if (record instanceof BackwardStartedRecord) {
    return { attempt_id: BigInt(record.attemptId), feedback: record.feedback as JsonValue };
  }
  if (record instanceof BackwardReadyRecord) {
    return {
      attempt_id: BigInt(record.attemptId),
      root_result: backwardResultPayload(record.rootResult),
      attributed_results: record.attributedResults.map(attributedResultPayload),
      proposals: record.proposals.map(memoryProposalPayload),
    };
  }
  // The one kind left is backward_finished.
  return {
    attempt_id: BigInt(record.attemptId),
    outcome: backwardOutcomePayload(record.outcome),
    attributed_results: record.attributedResults.map(attributedResultPayload),
  };
}

/**
 * Build the record of wire kind `kind` from its payload object, through the record's
 * constructor.
 *
 * @throws RangeError for an unknown kind, missing or extra fields, an empty id or a counter that
 *   isn't a safe integer, besides the constructor's own errors.
 * @throws TypeError when a field has the wrong JSON type.
 */
function recordFromPayload(kind: string, payload: JsonObject): RunRecord {
  const label = `${kind} payload`;
  switch (kind) {
    case "run_started":
      requireFields(payload, RUN_STARTED_FIELDS, label);
      // The record's constructor checks the fields cast here (the pool, the agents' configs, the
      // defaults and the call-time agents), so a bad value is named by its entry.
      return new RunStartedRecord({
        flowFingerprint: wireText(payload["flow_fingerprint"], `${label} flow_fingerprint`),
        initialPool: wireObject(payload["initial_pool"], `${label} initial_pool`) as unknown as PoolWire,
        startWiringOverride: wireObject(payload["start_wiring_override"], `${label} start_wiring_override`),
        effectiveLlmByStaticPath: wireObject(
          payload["effective_llm_by_static_path"],
          `${label} effective_llm_by_static_path`,
        ) as Record<string, JsonObject>,
        numWorkers: wireInt(payload["num_workers"], `${label} num_workers`),
        memorySnapshots: wireNullableItems(
          payload["memory_snapshots"],
          `${label} memory_snapshots`,
          snapshotFromPayload,
        ),
        learnable: wireBool(payload["learnable"], `${label} learnable`),
        memories: wireNullableItems(payload["memories"], `${label} memories`, memorySpecFromPayload),
        memoryStacks: wireNullableItems(payload["memory_stacks"], `${label} memory_stacks`, memoryStackFromPayload),
        defaults: wireObject(payload["defaults"], `${label} defaults`) as Record<string, bigint>,
        callTimeLlm: payload["call_time_llm"] as JsonObject | null,
      });
    case "run_resumed":
      requireFields(payload, ["retry_node_ids", "num_workers"], label);
      return new RunResumedRecord(
        wireTextArray(payload["retry_node_ids"], `${label} retry_node_ids`),
        wireInt(payload["num_workers"], `${label} num_workers`),
      );
    case "run_stopped":
      requireFields(payload, ["reason", "message", "node_ids"], label);
      return new RunStoppedRecord(
        wireText(payload["reason"], `${label} reason`) as RunStopReason,
        wireText(payload["message"], `${label} message`),
        wireTextArray(payload["node_ids"], `${label} node_ids`),
      );
    case "node_started":
      requireFields(payload, ["node_id"], label);
      return new NodeStartedRecord(wireName(payload["node_id"], `${label} node_id`));
    case "node_step":
      requireFields(payload, ["node_id", "step"], label);
      return new NodeStepRecord(
        wireName(payload["node_id"], `${label} node_id`),
        wireObject(payload["step"], `${label} step`),
      );
    case "node_succeeded":
      requireFields(payload, ["node_id", "commit_target", "value", "evidence", "forward_capture"], label);
      return new NodeSucceededRecord(
        wireName(payload["node_id"], `${label} node_id`),
        wireName(payload["commit_target"], `${label} commit_target`),
        wireObject(payload["value"], `${label} value`) as unknown as TypedWire,
        outcomeFromPayload(payload, label),
      );
    case "node_routed": {
      requireFields(payload, ["node_id", "handle", "terminal_failure", "evidence", "forward_capture"], label);
      const failure = payload["terminal_failure"];
      return new NodeRoutedRecord(
        wireName(payload["node_id"], `${label} node_id`),
        wireName(payload["handle"], `${label} handle`),
        {
          terminalFailure: failure === null ? null : failureFromPayload(failure, `${label} terminal_failure`),
          ...outcomeFromPayload(payload, label),
        },
      );
    }
    case "node_expanded":
      requireFields(
        payload,
        ["spawner_id", "seed", "prune_ids", "topology_digest", "evidence", "forward_capture"],
        label,
      );
      return new NodeExpandedRecord(
        wireName(payload["spawner_id"], `${label} spawner_id`),
        payload["seed"],
        wireTextArray(payload["prune_ids"], `${label} prune_ids`),
        wireText(payload["topology_digest"], `${label} topology_digest`),
        outcomeFromPayload(payload, label),
      );
    case "node_failed":
      requireFields(payload, ["failure", "evidence", "forward_capture"], label);
      return new NodeFailedRecord(
        failureFromPayload(payload["failure"], `${label} failure`),
        outcomeFromPayload(payload, label),
      );
    case "memory_commit_requested":
      requireFields(payload, ["memory", "commit_id", "attempt_id", "mutations"], label);
      return new MemoryCommitRequestedRecord(
        wireName(payload["memory"], `${label} memory`),
        wireName(payload["commit_id"], `${label} commit_id`),
        wireItems(payload["mutations"], `${label} mutations`, mutationFromPayload),
        wireInt(payload["attempt_id"], `${label} attempt_id`),
      );
    case "memory_commit_succeeded":
      requireFields(payload, ["memory", "commit_id", "attempt_id"], label);
      return new MemoryCommitSucceededRecord(
        wireName(payload["memory"], `${label} memory`),
        wireName(payload["commit_id"], `${label} commit_id`),
        wireInt(payload["attempt_id"], `${label} attempt_id`),
      );
    case "memory_commit_failed":
      requireFields(payload, ["memory", "commit_id", "attempt_id", "failure"], label);
      return new MemoryCommitFailedRecord(
        wireName(payload["memory"], `${label} memory`),
        wireName(payload["commit_id"], `${label} commit_id`),
        memoryFailureFromPayload(payload["failure"], `${label} failure`),
        wireInt(payload["attempt_id"], `${label} attempt_id`),
      );
    case "backward_started":
      requireFields(payload, ["attempt_id", "feedback"], label);
      return new BackwardStartedRecord(
        wireInt(payload["attempt_id"], `${label} attempt_id`),
        payload["feedback"] as FeedbackValue,
      );
    case "backward_ready":
      requireFields(payload, ["attempt_id", "root_result", "attributed_results", "proposals"], label);
      return new BackwardReadyRecord(
        wireInt(payload["attempt_id"], `${label} attempt_id`),
        backwardResultFromPayload(payload["root_result"], `${label} root_result`),
        wireItems(payload["attributed_results"], `${label} attributed_results`, attributedResultFromPayload),
        wireItems(payload["proposals"], `${label} proposals`, memoryProposalFromPayload),
      );
    case "backward_finished":
      requireFields(payload, ["attempt_id", "outcome", "attributed_results"], label);
      return new BackwardFinishedRecord(
        wireInt(payload["attempt_id"], `${label} attempt_id`),
        backwardOutcomeFromPayload(payload["outcome"], `${label} outcome`),
        wireItems(payload["attributed_results"], `${label} attributed_results`, attributedResultFromPayload),
      );
    default:
      throw new RangeError(`unknown run-record kind: ${reprText(kind)}`);
  }
}

/**
 * Return `value` when it is a JSON object, with exactly `fields` when they are given.
 *
 * @throws TypeError when it isn't an object.
 * @throws RangeError when it has missing or extra fields.
 */
function wireObject(value: unknown, label: string, fields?: readonly string[]): JsonObject {
  if (!isRecord(value)) throw new TypeError(`${label} must be an object`);
  if (fields !== undefined) requireFields(value, fields, label);
  return value as JsonObject;
}

/**
 * Return `value` when it is a JSON object or `null`.
 *
 * @throws TypeError otherwise.
 */
function wireNullableObject(value: unknown, label: string): JsonObject | null {
  if (value !== null && !isRecord(value)) throw new TypeError(`${label} must be an object or null`);
  return value as JsonObject | null;
}

/**
 * Return `value` when it is a JSON array.
 *
 * @throws TypeError otherwise.
 */
function wireArray(value: unknown, label: string): JsonValue[] {
  if (!isList(value)) throw new TypeError(`${label} must be an array`);
  return value as JsonValue[];
}

/**
 * Return `value` when it is an array of strings; emptiness, uniqueness and order are the record's checks.
 *
 * @throws TypeError otherwise.
 */
function wireTextArray(value: unknown, label: string): string[] {
  if (!isList(value) || !value.every((item) => typeof item === "string")) {
    throw new TypeError(`${label} must be an array of strings`);
  }
  return value as string[];
}

/**
 * Return each item of the JSON array `value` decoded by `decode` with the label `<label> item`.
 *
 * @throws TypeError when `value` isn't an array, besides `decode`'s errors.
 */
function wireItems<T>(value: unknown, label: string, decode: (item: unknown, label: string) => T): T[] {
  return wireArray(value, label).map((item) => decode(item, `${label} item`));
}

/**
 * Return `null` for `null`, else each item of the JSON array `value` decoded by `decode` with the
 * label `<label> item`.
 *
 * @throws TypeError when `value` is neither an array nor `null`, besides `decode`'s errors.
 */
function wireNullableItems<T>(value: unknown, label: string, decode: (item: unknown, label: string) => T): T[] | null {
  if (value === null) return null;
  if (!isList(value)) throw new TypeError(`${label} must be an array or null`);
  return value.map((item) => decode(item, `${label} item`));
}

/**
 * Return `value` when it is a boolean.
 *
 * @throws TypeError otherwise.
 */
function wireBool(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") throw new TypeError(`${label} must be a boolean`);
  return value;
}

/**
 * Return `value` when it is a string.
 *
 * @throws TypeError otherwise.
 */
function wireText(value: unknown, label: string): string {
  if (typeof value !== "string") throw new TypeError(`${label} must be a string`);
  return value;
}

/**
 * Return `value` when it is a string or `null`.
 *
 * @throws TypeError otherwise.
 */
function wireNullableText(value: unknown, label: string): string | null {
  if (value !== null && typeof value !== "string") throw new TypeError(`${label} must be a string or null`);
  return value;
}

/**
 * Return `value` when it is a non-empty string.
 *
 * @throws TypeError when it isn't a string.
 * @throws RangeError when it is empty.
 */
function wireName(value: unknown, label: string): string {
  if (typeof value !== "string") throw new TypeError(`${label} must be a non-empty string`);
  if (!value) throw new RangeError(`${label} must be a non-empty string`);
  return value;
}

/**
 * Return an int token (a `bigint`) as a `number`.
 *
 * @throws TypeError when `value` isn't an int token (a float token, a boolean, ...).
 * @throws RangeError when it is beyond ±(2^53 - 1).
 */
function wireInt(value: unknown, label: string): number {
  if (typeof value !== "bigint") throw new TypeError(`${label} must be an integer`);
  const number = Number(value);
  if (!Number.isSafeInteger(number)) throw new RangeError(`${label} must be a safe integer`);
  return number;
}

/**
 * Return `runId` when it is a non-empty string fit for an envelope.
 *
 * @throws RangeError when it isn't a non-empty string or holds a lone surrogate.
 */
function validatedRunId(runId: unknown): string {
  if (typeof runId !== "string" || !runId) throw new RangeError("run log run_id must be a non-empty string");
  if (!runId.isWellFormed()) throw new RangeError("Unicode surrogate is not valid canonical JSON");
  return runId;
}

/**
 * Decode every newline-terminated record in `data` and check their shared identity.
 *
 * Errors name `path`, the 1-based line and the expected seq, so a damaged log can be located
 * without bisecting it.
 *
 * @throws RangeError when `data` doesn't end in a newline, a line isn't a valid record (the
 *   decoder's error as `cause`), or a seq or run id doesn't match.
 */
function decodeCompletePrefix(data: Uint8Array, path: string): RecordEnvelope[] {
  if (data.length === 0) return [];
  // Callers pass bytes up to the last newline; the loop below relies on it.
  if (data[data.length - 1] !== 0x0a) throw new RangeError(`run log ${path} is not newline terminated`);
  const records: RecordEnvelope[] = [];
  let runId: string | null = null;
  for (let start = 0, sequence = 0; start < data.length; sequence++) {
    const end = data.indexOf(0x0a, start);
    const where = `run log ${path} line ${sequence + 1} (seq ${sequence})`;
    let envelope: RecordEnvelope;
    try {
      envelope = decodeRecord(data.subarray(start, end));
    } catch (error) {
      if (error instanceof JSONDecodeError || error instanceof TypeError || error instanceof RangeError) {
        throw new RangeError(`${where}: ${error.message}`, { cause: error });
      }
      throw error;
    }
    if (envelope.seq !== sequence) {
      throw new RangeError(`${where}: run-log sequence mismatch: expected ${sequence}, found ${envelope.seq}`);
    }
    runId ??= envelope.runId;
    if (envelope.runId !== runId) {
      throw new RangeError(
        `${where}: run-log run_id mismatch: expected ${reprText(runId)}, found ${reprText(envelope.runId)}`,
      );
    }
    records.push(envelope);
    start = end + 1;
  }
  return records;
}

/**
 * Take the writer's lock on the log at `path` (its sibling `<path>.lock`) without waiting.
 *
 * @throws Error naming the log and the likely cause when another writer holds it, with the busy
 *   `FileLockError` as `cause`; other `lockFile` errors propagate.
 */
function lockRunLog(path: string): FileLock {
  try {
    return lockFile(`${path}.lock`);
  } catch (error) {
    if (error instanceof FileLockError && error.reason === "busy") {
      throw new Error(`run log ${path} is locked by another writer (is the run still active?)`, { cause: error });
    }
    throw error;
  }
}

/**
 * Return a record's canonical line under `runId` and `seq`, and the envelope decoded from it.
 * A log retains the decoded line, so a live run sees exactly what a resumed run reads back.
 *
 * @throws RangeError, TypeError the envelope's and the codec's errors.
 */
function canonicalEnvelope(record: RunRecord, runId: string, seq: number): [string, RecordEnvelope] {
  const line = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, runId, seq, record));
  return [line, decodeRecord(line)];
}

/**
 * Truncate `fd` back to `prefixSize` bytes, then sync it, attempting both; return their errors.
 */
function rollbackAppend(fd: number, prefixSize: number): unknown[] {
  const errors: unknown[] = [];
  for (const operation of [() => fs.ftruncateSync(fd, prefixSize), () => fs.fsyncSync(fd)]) {
    try {
      operation();
    } catch (error) {
      errors.push(error);
    }
  }
  return errors;
}

/** Release `fd` and `lock` after a failed `create` or `resume`, ignoring release errors. */
function releaseAfterFailure(fd: number, lock: FileLock | undefined): void {
  try {
    release(fd, lock);
  } catch {
    // The caller rethrows the error that explains the failure; a release error would mask it.
  }
}

/** Release `lock` (when taken) and close `fd`; the close runs even when the release throws. */
function release(fd: number, lock: FileLock | undefined): void {
  try {
    lock?.release();
  } finally {
    fs.closeSync(fd);
  }
}
