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

import { compareCodeUnits, Failure, reprText } from "../diagnostics.ts";
import { EvaluationEvidence } from "../memory/contracts.ts";
import { retainFailure } from "../nodes/base.ts";
import { isExactly } from "../nodes/contracts.ts";
import { type JsonObject, type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import {
  buildValue,
  buildValueAs,
  type PoolWire,
  type Type,
  TypedValue,
  type ValueKind,
  VariablePool,
} from "../typesys/index.ts";
import { isList, isRecord, jsonPart, valueRepr, writeAsText } from "../typesys/values.ts";
import { ForwardCapture } from "./contracts.ts";

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
   * @throws TypeCheckError when the seed isn't a JSON value (see `canonicalSeed`).
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

/** One durable run transition. */
export type RunRecord =
  | RunResumedRecord
  | RunStoppedRecord
  | NodeStartedRecord
  | NodeStepRecord
  | NodeSucceededRecord
  | NodeRoutedRecord
  | NodeExpandedRecord
  | NodeFailedRecord;

/** A run-record class. */
type RunRecordClass = new (...args: never[]) => RunRecord;

/** Each record class and its snake_case wire kind. */
const RECORD_KINDS: ReadonlyMap<RunRecordClass, string> = new Map<RunRecordClass, string>([
  [RunResumedRecord, "run_resumed"],
  [RunStoppedRecord, "run_stopped"],
  [NodeStartedRecord, "node_started"],
  [NodeStepRecord, "node_step"],
  [NodeSucceededRecord, "node_succeeded"],
  [NodeRoutedRecord, "node_routed"],
  [NodeExpandedRecord, "node_expanded"],
  [NodeFailedRecord, "node_failed"],
]);

/** Every run-record class. */
export const RUN_RECORD_TYPES: readonly RunRecordClass[] = Object.freeze([...RECORD_KINDS.keys()]);

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
  if (schema !== SCHEMA_VERSION) throw new RangeError(`unsupported run-record schema: ${valueRepr(schema)}`);
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
}
