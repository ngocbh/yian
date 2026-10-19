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

import { Failure } from "../diagnostics.ts";
import { ExpressionError } from "../expr/expressions.ts";
import { ResolvedValue, SourceRef, sameSource, sourceKey, ValueRoute } from "../expr/lineage.ts";
import { MemoryScope } from "../memory/index.ts";
import { BackwardContext, type Node, retainFailure, validateDeclaredNames } from "../nodes/base.ts";
import { BindingCapture } from "../nodes/binding.ts";
import { isExactly, type SeedConnection, sameOptional } from "../nodes/contracts.ts";
import { type JsonValue, parseJson } from "../typesys/json.ts";
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
  type NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeSucceededRecord,
  type RecordEnvelope,
} from "./log.ts";

/** An accepted outcome record, the record a trace occurrence holds. */
export type OutcomeRecord = NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord;

/** A node's bound backward handler, kept in the trace's private index. */
type BackwardHandler = Node["backward"];

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
