/**
 * Accepted execution history and paired reads over immutable producer entries.
 *
 * The state reducer prepares sparse deltas here before append and applies them only after
 * acceptance. Records own outcome data; private callable references live in a separate index and
 * never cross the wire. This module depends on runtime contracts and codecs plus lower
 * node/expression/value contracts, never the driver, public run boundary, or compile expansion
 * implementation.
 *
 * Accepted growth metadata installs pending child seed routes. Child outcomes keep their exact
 * recorded capture while deriving an effective input route, with actual second expression
 * evaluations taking precedence. Historical boundary associations survive consumption and live
 * graph pruning.
 *
 * Typed values (source entries, terminal output, system values) are held as their canonical log
 * text, and each read decodes a fresh value.
 *
 * Knows about:   diagnostics, expr/expressions, expr/lineage, memory, nodes/base, nodes/binding,
 *                nodes/contracts, runtime/contracts, runtime/log, typesys.
 * Never imports: runtime state, learning or backward, engine or run, compile, compose, or the CLI.
 */

import { compareCodeUnits, Failure } from "../diagnostics.ts";
import { ExpressionError } from "../expr/expressions.ts";
import { type FeedbackValue, ResolvedValue, SourceRef, sameSource, sourceKey, ValueRoute } from "../expr/lineage.ts";
import { MemoryScope } from "../memory/index.ts";
import {
  BackwardContext,
  type BackwardResult,
  type Node,
  retainFailure,
  validateDeclaredNames,
} from "../nodes/base.ts";
import { BindingCapture } from "../nodes/binding.ts";
import {
  type ChildFlowBoundary,
  isExactly,
  type SeedConnection,
  sameExecutionScope,
  sameItems,
  sameOptional,
} from "../nodes/contracts.ts";
import { type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import {
  fieldOf,
  isList,
  isRecord,
  normalizeTypedFiles,
  type Type,
  TypedValue,
  typeEquals,
  type Value,
} from "../typesys/values.ts";
import { BackwardCapture, ForwardCapture, validateSource } from "./contracts.ts";
import {
  canonicalSeed,
  decodeRecord,
  encodeRecord,
  evidencePayload,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  type RecordEnvelope,
  RunResumedRecord,
  RunStartedRecord,
} from "./log.ts";

/** An accepted outcome record, the record a trace occurrence holds. */
export type OutcomeRecord = NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord;

/** A node's bound backward handler, kept in the trace's private index. */
type BackwardHandler = (
  context: BackwardContext,
  feedback: readonly FeedbackValue[],
) => BackwardResult | Promise<BackwardResult>;

/** Decode a typed value from its canonical log text. */
function decodeTyped(text: string): TypedValue {
  return TypedValue.fromWire(parseJson(text, "codec"));
}

/**
 * Join detached binding and backward material through the existing seed codec.
 *
 * @param bindings Actual delivered values, routes and indexed item bindings frozen by the
 *   evaluator.
 * @param backward The same invocation's declared names, scope and captured node data. Item seeds
 *   have no separate node inventory.
 * @returns Canonical material ready for accepted-record validation.
 * @throws TypeError when either transient carrier isn't exactly its class.
 * @throws RangeError or TypeCheckError when captured data can't satisfy the canonical wire contract.
 */
export function captureBindings(bindings: BindingCapture, backward: BackwardCapture): ForwardCapture {
  if (!isExactly(bindings, BindingCapture)) throw new TypeError("retained evaluation requires a BindingCapture");
  if (!isExactly(backward, BackwardCapture)) throw new TypeError("retained evaluation requires a BackwardCapture");
  const boundInputs = canonicalSeed(bindings.boundInputs)[1] as Record<string, JsonValue> | null;
  return new ForwardCapture(boundInputs, bindings.route, {
    evaluatedKeys: bindings.evaluatedKeys,
    items: bindings.items.map(([index, item]) => [index, captureBindings(item, new BackwardCapture())] as const),
    scope: backward.scope,
    nodeData: backward.nodeData,
    inputNames: backward.inputNames,
    variableNames: backward.variableNames,
  });
}

/**
 * A typed accepted value and its producer identity read as one unit. Frozen.
 *
 * `files` is the type the commit target reads the value through to restore its files, as the pool
 * does; `null` reads the value as stored.
 */
export class SourceEntry {
  /** The producer's accepted start, or `null` for an initial pool value without one. */
  readonly source: SourceRef | null;
  /** The sequence of the record that accepted the value. */
  readonly outcomeSeq: number;
  /** The value's canonical typed wire `{kind, value}` as log text. */
  readonly valueJson: string;
  /** The type the value is read through to restore its files, or `null`. */
  readonly files: Type | null;

  /** Hold the entry's fields as given. */
  constructor(source: SourceRef | null, outcomeSeq: number, valueJson: string, files: Type | null = null) {
    this.source = source;
    this.outcomeSeq = outcomeSeq;
    this.valueJson = valueJson;
    this.files = files;
    Object.freeze(this);
  }

  /** A fresh decoded value with the pool's semantics: files restored through `files`. */
  get value(): Value {
    return normalizeTypedFiles(this.files, decodeTyped(this.valueJson)).toObject();
  }

  /** A fresh decoded value as stored, each nested file as its JSON form. */
  get storedValue(): Value {
    return decodeTyped(this.valueJson).toObject();
  }
}

/** Whether two entries have the same source (by key), outcome sequence, value text and files type. */
export function sameSourceEntry(a: SourceEntry, b: SourceEntry): boolean {
  return (
    sameOptional(a.source, b.source, sameSource) &&
    a.outcomeSeq === b.outcomeSeq &&
    a.valueJson === b.valueJson &&
    sameOptional(a.files, b.files, typeEquals)
  );
}

/** The reducer's terminal result and its actual accepted causal identity. Frozen. */
export class TraceTerminal {
  /** Whether the run succeeded. */
  readonly succeeded: boolean;
  /** The accepted occurrence or root writer the result comes from, if known. */
  readonly source: SourceRef | null;
  /** The output's canonical typed wire as log text, or `null` when the run failed. */
  readonly outputJson: string | null;
  /** The run's failure, kept through `retainFailure`, or `null` when it succeeded. */
  readonly failure: Failure | null;

  /**
   * Require distinct success and failure shapes with canonical values.
   *
   * @throws TypeError when `succeeded` isn't a boolean or `source` isn't exactly a `SourceRef`.
   * @throws RangeError when `source` has a path or is malformed, a success lacks output text or
   *   has a failure, a failure lacks a `Failure` or has output, or the output text isn't a typed
   *   value's wire.
   */
  constructor(succeeded: boolean, source: SourceRef | null, outputJson: string | null, failure: Failure | null) {
    if (typeof succeeded !== "boolean") throw new TypeError("terminal succeeded must be a bool");
    if (source !== null) {
      validateSource(source);
      if (source.path.length) throw new RangeError("terminal source must identify a root value");
    }
    if (succeeded) {
      if (typeof outputJson !== "string" || failure !== null) {
        throw new RangeError("successful terminal requires output text and no failure");
      }
      decodeTyped(outputJson);
    } else if (outputJson !== null || !isExactly(failure, Failure)) {
      throw new RangeError("failed terminal requires a failure and no output");
    }
    this.succeeded = succeeded;
    this.source = source;
    this.outputJson = outputJson;
    this.failure = failure === null ? null : retainFailure(failure);
    Object.freeze(this);
  }

  /** Fresh canonical output, or `null` when the run failed. */
  get output(): JsonValue {
    if (!this.succeeded) return null;
    return canonicalSeed(decodeTyped(this.outputJson!).toObject())[1];
  }
}

/**
 * Return a historical memory scope after checking its class. A scope is frozen, so it is held by
 * reference.
 *
 * @throws TypeError when `scope` is neither `null` nor exactly a `MemoryScope`.
 */
export function checkMemoryScope(scope: MemoryScope | null): MemoryScope | null {
  if (scope !== null && !isExactly(scope, MemoryScope)) {
    throw new TypeError("historical memory scope must be a MemoryScope or null");
  }
  return scope;
}

/**
 * One accepted outcome and its derived route, identified by its accepted start. Frozen.
 *
 * Private memory ownership is detached from the physical node before pruning. Replay reconstructs
 * it from that node; it never replaces execution scope or adds memory ownership to the accepted
 * outcome payload. `hiddenInputs` names the bound inputs that node keeps from models, so memory
 * episodes leave them out.
 */
export class TraceOccurrence {
  /** The occurrence's accepted start. */
  readonly source: SourceRef;
  /** The executing node's id. */
  readonly nodeId: string;
  /** The sequence of the accepted outcome record. */
  readonly outcomeSeq: number;
  /** The outcome envelope's canonical log line. */
  readonly recordJson: string;
  /** The envelope decoded from `recordJson`, a frozen copy independent of the caller's. */
  readonly envelope: RecordEnvelope;
  /** The effective input route: the captured route joined with pending seed routes. */
  readonly inputRoute: ValueRoute;
  /** The receiver's memory keys and occurrence path, if it has memory. */
  readonly memoryScope: MemoryScope | null;
  /** The bound inputs the node keeps from models; a private copy, read-only by contract. */
  readonly hiddenInputs: ReadonlySet<string>;

  /**
   * Encode `envelope` (which validates it, as an append does) and keep its decoded copy.
   *
   * @throws TypeError when `envelope` isn't exactly a `RecordEnvelope` or `memoryScope` isn't
   *   `null` or exactly a `MemoryScope`.
   */
  constructor(
    source: SourceRef,
    nodeId: string,
    outcomeSeq: number,
    envelope: RecordEnvelope,
    inputRoute: ValueRoute,
    memoryScope: MemoryScope | null = null,
    hiddenInputs: ReadonlySet<string> = new Set(),
  ) {
    this.source = source;
    this.nodeId = nodeId;
    this.outcomeSeq = outcomeSeq;
    this.recordJson = encodeRecord(envelope);
    this.envelope = decodeRecord(this.recordJson);
    this.inputRoute = inputRoute;
    this.memoryScope = checkMemoryScope(memoryScope);
    this.hiddenInputs = new Set(hiddenInputs);
    Object.freeze(this);
  }

  /** The accepted outcome record, read from the decoded envelope. */
  get record(): OutcomeRecord {
    // Occurrences are prepared only for outcome records.
    return this.envelope.record as OutcomeRecord;
  }
}

/** One accepted expansion's child flow and its actual accepted endpoints. Frozen. */
export class TraceBoundary {
  /** The expansion occurrence that owns the child flow. */
  readonly owner: SourceRef;
  /** The child flow's START id. */
  readonly startId: string;
  /** The child flow's END id. */
  readonly endId: string;
  /** The child START's accepted source, once known. */
  readonly startSource: SourceRef | null;
  /** The child END's accepted source, once known. */
  readonly endSource: SourceRef | null;

  /** Hold the boundary's fields as given. */
  constructor(
    owner: SourceRef,
    startId: string,
    endId: string,
    startSource: SourceRef | null = null,
    endSource: SourceRef | null = null,
  ) {
    this.owner = owner;
    this.startId = startId;
    this.endId = endId;
    this.startSource = startSource;
    this.endSource = endSource;
    Object.freeze(this);
  }
}

/** Candidate input routes owned by one accepted growth occurrence. Frozen. */
export class SeedRoutes {
  /** The growth occurrence that owns the routes. */
  readonly owner: SourceRef;
  /** The candidate `(input name, route)` pairs. */
  readonly fields: readonly (readonly [string, ValueRoute])[];

  /** Hold the owner and a frozen copy of its `(input name, route)` pairs. */
  constructor(owner: SourceRef, fields: readonly (readonly [string, ValueRoute])[]) {
    this.owner = owner;
    this.fields = Object.freeze(fields.map(([name, route]) => Object.freeze([name, route] as const)));
    Object.freeze(this);
  }
}

/**
 * Select one seed field route by explicit item/name metadata, preserving nulls.
 *
 * @param capture The owner's accepted forward capture.
 * @param connection The seed connection naming an item and/or field.
 * @param inputRoute The owner's effective input route; when given, a field-only path selects from
 *   it instead of the raw capture route.
 * @returns The selected route, or `null` when the item or field wasn't captured.
 */
export function selectSeedRoute(
  capture: ForwardCapture,
  connection: SeedConnection,
  inputRoute: ValueRoute | null = null,
): ValueRoute | null {
  let route = inputRoute ?? capture.route;
  const path = connection.seedPath;
  if (path.length === 2) {
    const binding = capture.items.find(([index]) => index === path[0]);
    if (binding === undefined) return null;
    route = binding[1].route;
  }
  const position = route.names.indexOf(path[path.length - 1] as string);
  return position < 0 ? null : route.children[position]!;
}

/**
 * Reducer-owned history with immutable entries and sparse mutable indexes.
 *
 * A worker reads one current entry at binding time. Pruning removes current names only;
 * historical values, accepted occurrences and private handlers stay available independently of
 * the live scheduling graph.
 *
 * The internal fields are plain properties that the trace delta and the state reducer write;
 * everyone else reads the views. An internal field behind a view of the same name has a leading
 * underscore. Maps of sources key by `sourceKey`.
 */
export class RetainedTrace {
  /** The run's id. */
  readonly runId: string;
  /** The root START node's id. */
  readonly startId: string;
  /** The root END node's id, if known. */
  readonly endId: string | null;
  /** The root flow's declared input names. */
  readonly rootInputNames: readonly string[];
  /** The accepted start of the root START node, once it has an outcome. */
  rootStartSource: SourceRef | null = null;
  /** The run's terminal result, once accepted. */
  terminal: TraceTerminal | null = null;
  /** The sequence the next accepted record must have. */
  nextSequence = 0;
  /** @internal The current writer of the root END target, if any. */
  rootWriter: SourceEntry | null = null;
  /** @internal The first accepted routed occurrence that proved a terminal failure. */
  causalRouteSource: SourceRef | null = null;
  /** @internal Accepted outcomes in record order. */
  _occurrences: TraceOccurrence[] = [];
  /** @internal Accepted outcomes by their start's `sourceKey`. */
  occurrenceBySource = new Map<string, TraceOccurrence>();
  /** @internal Accepted source entries by `sourceKey`. */
  _sources = new Map<string, SourceEntry>();
  /** @internal Each current pool name's producer entry. */
  _currentWriters = new Map<string, SourceEntry>();
  /** @internal Accepted unfinished starts: node id to start sequence. */
  _started = new Map<string, number>();
  /** @internal Bound backward handlers by occurrence `sourceKey`. */
  _backwardHandlers = new Map<string, BackwardHandler>();
  /** @internal System values as canonical typed log text, by key. */
  system = new Map<string, string>();
  /** @internal Unconsumed child seed routes by target node id. */
  _pendingConnections = new Map<string, SeedRoutes>();
  /** @internal Accepted child boundaries in record order. */
  _boundaries: TraceBoundary[] = [];
  /** @internal Each live boundary endpoint's boundary index and the field its source fills. */
  currentBoundaries = new Map<string, readonly [number, "startSource" | "endSource"]>();

  /**
   * Start an empty trace for one run's root START and END.
   *
   * @throws TypeError or RangeError when `rootInputNames` isn't an array of distinct strings.
   */
  constructor(runId: string, startId: string, endId: string | null = null, rootInputNames: readonly string[] = []) {
    validateDeclaredNames(rootInputNames, "root_input_names");
    this.runId = runId;
    this.startId = startId;
    this.endId = endId;
    this.rootInputNames = Object.freeze([...rootInputNames]);
  }

  /** Unconsumed child seed candidates by target node id, live and read-only. */
  get pendingConnections(): ReadonlyMap<string, SeedRoutes> {
    return this._pendingConnections;
  }

  /** Accepted child associations, kept after their live nodes are pruned; a fresh array. */
  get boundaries(): readonly TraceBoundary[] {
    return Object.freeze([...this._boundaries]);
  }

  /** Accepted outcomes in record order; a fresh array. */
  get occurrences(): readonly TraceOccurrence[] {
    return Object.freeze([...this._occurrences]);
  }

  /** Historical source entries by `sourceKey`, live and read-only. */
  get sources(): ReadonlyMap<string, SourceEntry> {
    return this._sources;
  }

  /** Current pool names and their physical producers, live and read-only. */
  get currentWriters(): ReadonlyMap<string, SourceEntry> {
    return this._currentWriters;
  }

  /** Accepted unfinished starts by node id, live and read-only. */
  get started(): ReadonlyMap<string, number> {
    return this._started;
  }

  /** Reconstructed node handlers by occurrence `sourceKey`, kept apart from records; live and read-only. */
  get backwardHandlers(): ReadonlyMap<string, BackwardHandler> {
    return this._backwardHandlers;
  }

  /**
   * Return fresh canonical JSON for an accepted producer's value, or a path inside it.
   *
   * @throws RangeError when no source is accepted at the root of `source`, or its path is absent.
   */
  lookupSource(source: SourceRef): JsonValue {
    const key = sourceKey(new SourceRef(source.runId, source.executionSeq));
    const entry = this._sources.get(key);
    if (entry === undefined) throw new RangeError(`no accepted source for ${key}`);
    let value = (parseJson(entry.valueJson, "codec") as { value: JsonValue }).value;
    for (const part of source.path) {
      if (typeof part === "string" && isRecord(value) && Object.hasOwn(value, part)) {
        value = value[part] as JsonValue;
      } else if (typeof part === "number" && isList(value) && part >= 0 && part < value.length) {
        value = value[part] as JsonValue;
      } else {
        throw new RangeError("source path is absent from its accepted value");
      }
    }
    return value;
  }

  /**
   * Look up an accepted occurrence by its physical start identity.
   *
   * @throws RangeError when no accepted outcome has this identity, including source zero.
   */
  occurrence(source: SourceRef): TraceOccurrence {
    const key = sourceKey(source);
    const occurrence = this.occurrenceBySource.get(key);
    if (occurrence === undefined) throw new RangeError(`no accepted occurrence for ${key}`);
    return occurrence;
  }

  /**
   * Build detached context from accepted invocation facts without calling hooks.
   *
   * The node data gains the reserved `_evaluation_evidence` entry, holding only the accepted
   * record's evidence.
   *
   * @param source The accepted start of an executed occurrence.
   * @param options.downstreamContext Transient attribution materialized by a caller, copied with
   *   the other context values; defaults to empty.
   * @throws RangeError when the occurrence or its accepted output source is absent, the captured
   *   node data already holds `_evaluation_evidence`, or a value breaks the context contract.
   * @throws TypeError when supplied context material breaks the context contract.
   */
  backwardContext(
    source: SourceRef,
    options: { downstreamContext?: readonly Readonly<Record<string, unknown>>[] } = {},
  ): BackwardContext {
    const { downstreamContext = [] } = options;
    const occurrence = this.occurrence(source);
    const record = occurrence.record;
    // An accepted outcome holds its forward capture while learning.
    const capture = record.forwardCapture!;
    const captured = capture.nodeData ?? {};
    if (Object.hasOwn(captured, "_evaluation_evidence")) {
      throw new RangeError("captured node data uses reserved _evaluation_evidence");
    }
    const failure = isExactly(record, NodeFailedRecord)
      ? record.failure
      : isExactly(record, NodeRoutedRecord)
        ? record.terminalFailure
        : null;
    return new BackwardContext(capture.boundInputs, {
      inputNames: capture.inputNames,
      variableNames: capture.variableNames,
      output: isExactly(record, NodeSucceededRecord) ? this.lookupSource(source) : null,
      failure,
      scope: capture.scope,
      runId: source.runId,
      nodeId: occurrence.nodeId,
      nodeStartedSeq: source.executionSeq,
      nodeData: { ...captured, _evaluation_evidence: evidencePayload(record.evidence) },
      downstreamContext,
    });
  }

  /**
   * Pair each actual lookup's value, presence and producer in one read.
   *
   * `<id>.output...` reads a current writer, `input...` the START writer and `system.<key>` a
   * system value; anything else is a miss.
   *
   * @throws ExpressionError when a dotted segment of `path` is empty.
   */
  resolve(path: string): ResolvedValue {
    const parts = path.trim().split(".");
    if (parts.some((part) => !part)) throw new ExpressionError(`invalid variable path: '\${${path}}'`);
    const [head, ...rest] = parts as [string, ...string[]];
    let entry = this._currentWriters.get(head);
    let projection: string[];
    if (entry !== undefined && rest[0] === "output") {
      projection = rest.slice(1);
    } else if (head === "input" && rest.length) {
      entry = this._currentWriters.get(this.startId);
      projection = rest;
    } else if (head === "system") {
      const encoded = rest.length ? this.system.get(rest[0]!) : undefined;
      if (encoded === undefined) return new ResolvedValue(null, false);
      return new ResolvedValue(decodeTyped(encoded).toObject(), true);
    } else {
      return new ResolvedValue(null, false);
    }
    if (entry === undefined) return new ResolvedValue(null, false);
    let value = entry.value;
    for (const part of projection) {
      const [present, next] = fieldOf(value, part);
      if (!present) return new ResolvedValue(null, false);
      value = next;
    }
    const producer = entry.source;
    return new ResolvedValue(
      value,
      true,
      producer === null ? null : new SourceRef(producer.runId, producer.executionSeq, projection),
    );
  }
}

/** The optional parts of a `TraceDelta`; each defaults to `null`, `false` or an empty array. */
export interface TraceDeltaFields {
  /** The trace a learnable RunStarted creates. */
  initialized?: RetainedTrace | null;
  /** A started node's id and start sequence. */
  started?: readonly [string, number] | null;
  /** Whether a resume clears the unfinished starts. */
  resumed?: boolean;
  /** The accepted outcome's occurrence. */
  occurrence?: TraceOccurrence | null;
  /** The occurrence node's bound backward handler. */
  handler?: BackwardHandler | null;
  /** The accepted output's commit target and its source entry. */
  writer?: readonly [string, SourceEntry] | null;
  /** The current names an expansion retires. */
  pruneIds?: readonly string[];
  /** The child seed routes the outcome installs, by target node id. */
  connections?: readonly (readonly [string, SeedRoutes])[];
  /** The pending seed routes the outcome consumes, with their target node id. */
  consumedConnection?: readonly [string, SeedRoutes] | null;
  /** The child boundaries the outcome installs. */
  boundaries?: readonly TraceBoundary[];
  /** A boundary index and the boundary with the outcome's endpoint source filled in. */
  boundaryUpdate?: readonly [number, TraceBoundary] | null;
  /** The root START's first accepted start. */
  rootStartSource?: SourceRef | null;
  /** The new writer of the root END target. */
  rootWriter?: SourceEntry | null;
  /** The first routed occurrence that proves a terminal failure. */
  causalRouteSource?: SourceRef | null;
  /** The run's terminal result. */
  terminal?: TraceTerminal | null;
}

/** A fully prepared sparse change, applied only after its record is accepted. Frozen. */
export class TraceDelta {
  /** The trace the delta was prepared against; `null` for a learnable RunStarted. */
  readonly originatingTrace: RetainedTrace | null;
  /** The accepted record's sequence. */
  readonly sequence: number;
  /** The trace a learnable RunStarted creates, or `null`. */
  readonly initialized: RetainedTrace | null;
  /** A started node's id and start sequence, or `null`. */
  readonly started: readonly [string, number] | null;
  /** Whether a resume clears the unfinished starts. */
  readonly resumed: boolean;
  /** The accepted outcome's occurrence, or `null`. */
  readonly occurrence: TraceOccurrence | null;
  /** The occurrence node's bound backward handler, or `null`. */
  readonly handler: BackwardHandler | null;
  /** The accepted output's commit target and its source entry, or `null`. */
  readonly writer: readonly [string, SourceEntry] | null;
  /** The current names an expansion retires. */
  readonly pruneIds: readonly string[];
  /** The child seed routes the outcome installs, by target node id. */
  readonly connections: readonly (readonly [string, SeedRoutes])[];
  /** The pending seed routes the outcome consumes, with their target node id, or `null`. */
  readonly consumedConnection: readonly [string, SeedRoutes] | null;
  /** The child boundaries the outcome installs. */
  readonly boundaries: readonly TraceBoundary[];
  /** A boundary index and its updated boundary, or `null`. */
  readonly boundaryUpdate: readonly [number, TraceBoundary] | null;
  /** The root START's first accepted start, or `null`. */
  readonly rootStartSource: SourceRef | null;
  /** The new writer of the root END target, or `null`. */
  readonly rootWriter: SourceEntry | null;
  /** The first routed occurrence that proves a terminal failure, or `null`. */
  readonly causalRouteSource: SourceRef | null;
  /** The run's terminal result, or `null`. */
  readonly terminal: TraceTerminal | null;

  /** Hold the fields as given, with frozen copies of the arrays. */
  constructor(originatingTrace: RetainedTrace | null, sequence: number, fields: TraceDeltaFields = {}) {
    this.originatingTrace = originatingTrace;
    this.sequence = sequence;
    this.initialized = fields.initialized ?? null;
    this.started = frozenPair(fields.started ?? null);
    this.resumed = fields.resumed ?? false;
    this.occurrence = fields.occurrence ?? null;
    this.handler = fields.handler ?? null;
    this.writer = frozenPair(fields.writer ?? null);
    this.pruneIds = Object.freeze([...(fields.pruneIds ?? [])]);
    const connections = fields.connections ?? [];
    this.connections = Object.freeze(connections.map(([nodeId, routes]) => Object.freeze([nodeId, routes] as const)));
    this.consumedConnection = frozenPair(fields.consumedConnection ?? null);
    this.boundaries = Object.freeze([...(fields.boundaries ?? [])]);
    this.boundaryUpdate = frozenPair(fields.boundaryUpdate ?? null);
    this.rootStartSource = fields.rootStartSource ?? null;
    this.rootWriter = fields.rootWriter ?? null;
    this.causalRouteSource = fields.causalRouteSource ?? null;
    this.terminal = fields.terminal ?? null;
    Object.freeze(this);
  }

  /**
   * Reject stale or replaced trace authority before durable append.
   *
   * @throws RangeError when `trace` isn't the originating trace, its sequence moved, or the
   *   consumed seed routes are no longer pending for their target.
   * @throws TypeError when the occurrence isn't exactly a `TraceOccurrence` or its memory scope
   *   isn't `null` or exactly a `MemoryScope`.
   */
  validateOrigin(trace: RetainedTrace | null): void {
    if (trace !== this.originatingTrace) throw new RangeError("trace transition requires its originating trace");
    if (trace !== null && trace.nextSequence !== this.sequence) {
      throw new RangeError("trace transition has a stale sequence");
    }
    if (this.occurrence !== null) {
      if (!isExactly(this.occurrence, TraceOccurrence)) throw new TypeError("trace delta requires an exact occurrence");
      checkMemoryScope(this.occurrence.memoryScope);
    }
    if (this.consumedConnection !== null) {
      const [nodeId, routes] = this.consumedConnection;
      if (trace === null || trace._pendingConnections.get(nodeId) !== routes) {
        throw new RangeError("seed connection no longer has its prepared owner");
      }
    }
  }

  /** Install only prevalidated index updates after the record is accepted, and return the trace. */
  apply(): RetainedTrace {
    // A delta always has its initialized or originating trace.
    const trace = (this.initialized ?? this.originatingTrace)!;
    if (this.started !== null) {
      const [nodeId, sequence] = this.started;
      trace._started.set(nodeId, sequence);
    }
    if (this.resumed) trace._started.clear();
    if (this.occurrence !== null) {
      const key = sourceKey(this.occurrence.source);
      trace._occurrences.push(this.occurrence);
      trace.occurrenceBySource.set(key, this.occurrence);
      trace._backwardHandlers.set(key, this.handler!);
      trace._started.delete(this.occurrence.nodeId);
      trace.currentBoundaries.delete(this.occurrence.nodeId);
    }
    if (this.consumedConnection !== null) trace._pendingConnections.delete(this.consumedConnection[0]);
    if (this.boundaryUpdate !== null) {
      const [index, boundary] = this.boundaryUpdate;
      trace._boundaries[index] = boundary;
    }
    for (const nodeId of this.pruneIds) {
      trace._currentWriters.delete(nodeId);
      trace._pendingConnections.delete(nodeId);
      trace.currentBoundaries.delete(nodeId);
    }
    for (const [nodeId, routes] of this.connections) trace._pendingConnections.set(nodeId, routes);
    for (const boundary of this.boundaries) {
      const index = trace._boundaries.length;
      trace._boundaries.push(boundary);
      trace.currentBoundaries.set(boundary.startId, Object.freeze([index, "startSource"] as const));
      trace.currentBoundaries.set(boundary.endId, Object.freeze([index, "endSource"] as const));
    }
    if (this.writer !== null) {
      const [target, entry] = this.writer;
      trace._currentWriters.set(target, entry);
      // An outcome's writer always has its occurrence's source.
      trace._sources.set(sourceKey(entry.source!), entry);
    }
    if (this.rootStartSource !== null) trace.rootStartSource = this.rootStartSource;
    if (this.rootWriter !== null) trace.rootWriter = this.rootWriter;
    if (this.causalRouteSource !== null) trace.causalRouteSource = this.causalRouteSource;
    if (this.terminal !== null) trace.terminal = this.terminal;
    trace.nextSequence = this.sequence + 1;
    return trace;
  }
}

/** A frozen copy of `pair`, or `null`. */
function frozenPair<A, B>(pair: readonly [A, B] | null): readonly [A, B] | null {
  return pair === null ? null : Object.freeze([pair[0], pair[1]] as const);
}

/** The record classes that accept a node outcome. */
const OUTCOME_RECORDS = [NodeSucceededRecord, NodeRoutedRecord, NodeExpandedRecord, NodeFailedRecord] as const;

/** The options of `prepareTrace`. */
export interface PrepareTraceOptions {
  /** The seed connections the outcome's growth installs; defaults to none. */
  inputConnections?: readonly SeedConnection[];
  /** The child flows the outcome's growth installs; defaults to none. */
  childBoundaries?: readonly ChildFlowBoundary[];
  /** The settled prune ids, earlier deferred ones and the nodes grown under them; defaults to none. */
  retiredIds?: readonly string[];
  /** The root END id, required for a learnable RunStarted. */
  rootEndId?: string | null;
  /** The root START's backward input names, required for a learnable RunStarted. */
  rootInputNames?: readonly string[] | null;
  /** The terminal disposition the record makes, if any. */
  terminalSucceeded?: boolean | null;
  /** A successful terminal's output as canonical typed log text. */
  terminalOutputJson?: string | null;
  /** A failed terminal's failure. */
  terminalFailure?: Failure | null;
  /** The type an accepted output's commit target reads it through to restore its files. */
  writerFiles?: Type | null;
}

/**
 * Validate capture authority and prepare a detached accepted-record delta.
 *
 * A learnable RunStarted creates the trace from the record's initial pool; a RunStarted that
 * doesn't learn, and any record before one, prepares `null`. Outcomes join pending seed routes
 * into their input route (evaluated keys keep their own routes) and install their writer, seed
 * connections, child boundaries and terminal.
 *
 * @throws RangeError when the root metadata is missing, a capture has no trace, the record's
 *   coordinates or terminal payload disagree with the trace, a resume disagrees with the
 *   unfinished starts, an outcome lacks its capture or its accepted start, the capture's scope
 *   differs from the node's, a route source isn't an earlier accepted producer or lacks its path,
 *   a seed connection isn't captured or its target or a boundary endpoint already has an owner,
 *   or a terminal disagrees with the root writer or has no causal occurrence.
 * @throws TypeError when the node's backward isn't a function or its memory scope isn't `null` or
 *   exactly a `MemoryScope`.
 */
export function prepareTrace(
  trace: RetainedTrace | null,
  envelope: RecordEnvelope,
  node: Node | null,
  options: PrepareTraceOptions = {},
): TraceDelta | null {
  const {
    inputConnections = [],
    childBoundaries = [],
    retiredIds = [],
    rootEndId = null,
    rootInputNames = null,
    terminalSucceeded = null,
    terminalOutputJson = null,
    terminalFailure = null,
    writerFiles = null,
  } = options;
  const record = envelope.record;
  if (isExactly(record, RunStartedRecord)) {
    if (!record.learnable) return null;
    if (typeof rootEndId !== "string" || rootInputNames === null) {
      throw new RangeError("retained RunStarted requires verified root metadata");
    }
    // The record validated its pool, so each stored wire is already canonical.
    const pool = record.initialPool;
    const initial = new RetainedTrace(envelope.runId, pool.start_id, rootEndId, rootInputNames);
    for (const [target, wire] of Object.entries(pool.store)) {
      const source = target === pool.start_id ? new SourceRef(envelope.runId, 0) : null;
      const entry = new SourceEntry(source, 0, logJson(wire));
      initial._currentWriters.set(target, entry);
      if (source !== null) initial._sources.set(sourceKey(source), entry);
    }
    initial.rootWriter = initial._currentWriters.get(rootEndId) ?? null;
    initial.system = new Map(Object.entries(pool.system).map(([key, wire]) => [key, logJson(wire)]));
    return new TraceDelta(null, envelope.seq, { initialized: initial });
  }
  const capture = (record as { forwardCapture?: ForwardCapture | null }).forwardCapture ?? null;
  if (trace === null) {
    if (capture !== null) throw new RangeError("forward capture requires recorded learning");
    return null;
  }
  if (trace.runId !== envelope.runId || trace.nextSequence !== envelope.seq) {
    throw new RangeError("trace disagrees with accepted record coordinates");
  }
  if (terminalSucceeded === null && (terminalOutputJson !== null || terminalFailure !== null)) {
    throw new RangeError("terminal payload requires a terminal disposition");
  }
  if (isExactly(record, NodeStartedRecord)) {
    return new TraceDelta(trace, envelope.seq, { started: [record.nodeId, envelope.seq] });
  }
  if (isExactly(record, RunResumedRecord)) {
    if (!sameItems([...trace._started.keys()], record.retryNodeIds)) {
      throw new RangeError("trace unfinished starts disagree with retry frontier");
    }
    return new TraceDelta(trace, envelope.seq, { resumed: true });
  }
  if (!OUTCOME_RECORDS.some((cls) => isExactly<OutcomeRecord>(record, cls))) return new TraceDelta(trace, envelope.seq);
  if (!isExactly(capture, ForwardCapture)) {
    throw new RangeError("accepted outcome requires forward capture while learning");
  }
  if (node === null || !trace._started.has(node.id)) throw new RangeError("trace outcome requires its accepted start");
  if (!sameOptional(capture.scope, node.executionScope, sameExecutionScope)) {
    throw new RangeError("forward capture scope does not match the executing node");
  }
  const pending = [capture];
  while (pending.length > 0) {
    const binding = pending.pop()!;
    validateSources(trace, binding.route, envelope.seq);
    pending.push(...binding.items.map(([, item]) => item));
  }
  const source = new SourceRef(envelope.runId, trace._started.get(node.id)!);
  const candidates = trace._pendingConnections.get(node.id) ?? null;
  let inputRoute = capture.route;
  if (candidates !== null) {
    const fields = new Map(candidates.fields);
    inputRoute = ValueRoute.record(
      capture.route.names.map((name, index) => {
        const route = capture.route.children[index]!;
        return [name, capture.evaluatedKeys.has(name) ? route : (fields.get(name) ?? route)] as const;
      }),
    );
  }
  validateSources(trace, inputRoute, envelope.seq);
  const occurrence = new TraceOccurrence(
    source,
    node.id,
    envelope.seq,
    envelope,
    inputRoute,
    node.memoryScope,
    node.modelHiddenInputs(),
  );
  let boundaryUpdate: readonly [number, TraceBoundary] | null = null;
  const boundaryTarget = trace.currentBoundaries.get(node.id);
  if (boundaryTarget !== undefined) {
    const [index, role] = boundaryTarget;
    const { owner, startId, endId, startSource, endSource } = trace._boundaries[index]!;
    boundaryUpdate = [
      index,
      role === "startSource"
        ? new TraceBoundary(owner, startId, endId, source, endSource)
        : new TraceBoundary(owner, startId, endId, startSource, source),
    ];
  }
  const connections = new Map<string, (readonly [string, ValueRoute])[]>();
  const sortedConnections = [...inputConnections].sort(
    (a, b) => compareCodeUnits(a.childStartId, b.childStartId) || compareCodeUnits(a.inputName, b.inputName),
  );
  for (const connection of sortedConnections) {
    // An owner that is itself a connection target (a LOOP driver whose omitted field binds a
    // default) forwards its joined route rather than the raw default, so lineage survives every
    // iteration.
    const route = selectSeedRoute(capture, connection, inputRoute);
    if (route === null) throw new RangeError("seed connection field or item was not captured");
    const fields = connections.get(connection.childStartId) ?? [];
    fields.push([connection.inputName, route]);
    connections.set(connection.childStartId, fields);
  }
  for (const nodeId of connections.keys()) {
    if (trace._pendingConnections.has(nodeId)) {
      throw new RangeError("seed connection target already has an accepted owner");
    }
  }
  const boundaries = [...childBoundaries]
    .sort((a, b) => compareCodeUnits(a.startId, b.startId) || compareCodeUnits(a.endId, b.endId))
    .map((boundary) => new TraceBoundary(source, boundary.startId, boundary.endId));
  if (boundaries.some((boundary) => [boundary.startId, boundary.endId].some((id) => trace.currentBoundaries.has(id)))) {
    throw new RangeError("child boundary target already has an accepted owner");
  }
  if (typeof node.backward !== "function") throw new TypeError("node backward handler must be callable");
  const handler: BackwardHandler = node.backward.bind(node);
  const writer = isExactly(record, NodeSucceededRecord)
    ? ([record.commitTarget, new SourceEntry(source, envelope.seq, logJson(record.value), writerFiles)] as const)
    : null;
  const rootWriter = writer !== null && writer[0] === trace.endId ? writer[1] : null;
  const causalRouteSource =
    isExactly(record, NodeRoutedRecord) && record.terminalFailure !== null && trace.causalRouteSource === null
      ? source
      : null;
  let terminal: TraceTerminal | null = null;
  if (terminalSucceeded !== null) {
    let terminalSource: SourceRef | null;
    if (terminalSucceeded) {
      const acceptedWriter = rootWriter ?? trace.rootWriter;
      // Python compares decoded typed values with `==`, which treats 1, 1.0 and True as equal
      // inside a value; equal canonical text is stricter, but a real terminal and its writer are
      // encoded by the same canonical encoder, so this never refuses one.
      // A missing output fails here, where Python fails inside pydantic.
      if (
        acceptedWriter === null ||
        terminalOutputJson === null ||
        canonicalText(acceptedWriter.valueJson) !== canonicalText(terminalOutputJson)
      ) {
        throw new RangeError("terminal output disagrees with the accepted root writer");
      }
      terminalSource = acceptedWriter.source;
    } else {
      terminalSource = isExactly(record, NodeFailedRecord)
        ? source
        : (causalRouteSource ?? trace.causalRouteSource);
      if (terminalSource === null) throw new RangeError("accepted failure requires an accepted causal occurrence");
    }
    terminal = new TraceTerminal(terminalSucceeded, terminalSource, terminalOutputJson, terminalFailure);
  }
  return new TraceDelta(trace, envelope.seq, {
    occurrence,
    handler,
    writer,
    pruneIds: isExactly(record, NodeExpandedRecord) ? retiredIds : [],
    connections: [...connections].map(([nodeId, fields]) => [nodeId, new SeedRoutes(source, fields)] as const),
    consumedConnection: candidates === null ? null : [node.id, candidates],
    boundaries,
    boundaryUpdate,
    rootStartSource: node.id === trace.startId && trace.rootStartSource === null ? source : null,
    rootWriter,
    causalRouteSource,
    terminal,
  });
}

/** Canonical log text of the JSON `text`. */
function canonicalText(text: string): string {
  return logJson(parseJson(text, "codec"));
}

/**
 * Require sources accepted before the consumer outcome, regardless of starts.
 *
 * @throws RangeError when a route source's root isn't accepted before `outcomeSeq`, or its path
 *   is absent from the accepted value (the lookup error as `cause`).
 */
function validateSources(trace: RetainedTrace, route: ValueRoute, outcomeSeq: number): void {
  const source = route.source;
  if (source !== null) {
    const entry = trace._sources.get(sourceKey(new SourceRef(source.runId, source.executionSeq)));
    if (entry === undefined || entry.outcomeSeq >= outcomeSeq) {
      throw new RangeError("route source must name an earlier accepted producer");
    }
    try {
      trace.lookupSource(source);
    } catch (error) {
      throw new RangeError("route source path is absent from its accepted value", { cause: error });
    }
  }
  for (const child of route.children) validateSources(trace, child, outcomeSeq);
}
