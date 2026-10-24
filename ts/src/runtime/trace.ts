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

import { compareCodeUnits, Failure, reprText } from "../diagnostics.ts";
import { ExpressionError } from "../expr/expressions.ts";
import { type FeedbackValue, ResolvedValue, SourceRef, sameSource, sourceKey, ValueRoute } from "../expr/lineage.ts";
import {
  type MemoryOccurrenceSegment,
  MemoryScope,
  type MemorySection,
  ModelTextInteraction,
  ModelValueInteraction,
  ToolCallInteraction,
  type VisibleInteraction,
} from "../memory/index.ts";
import {
  BackwardContext,
  type BackwardOptions,
  type BackwardResult,
  type LearningMaterial,
  type MaterialCaps,
  type Node,
  retainFailure,
  validateDeclaredNames,
} from "../nodes/base.ts";
import { BINDING_FAILURE_CODES, BindingCapture, BindingError } from "../nodes/binding.ts";
import {
  type ChildFlowBoundary,
  isExactly,
  type SeedConnection,
  sameExecutionScope,
  sameItems,
  sameOptional,
} from "../nodes/contracts.ts";
import { type JsonObject, type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import type { VariablePool } from "../typesys/pool.ts";
import {
  fieldOf,
  isList,
  isRecord,
  normalizeTypedFiles,
  type Type,
  TypedValue,
  typeEquals,
  type Value,
  type ValueRecord,
} from "../typesys/values.ts";
import { BackwardCapture, ForwardCapture, validateSource } from "./contracts.ts";
import {
  canonicalSeed,
  decodeRecord,
  encodeRecord,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  type RecordEnvelope,
  RunFailedRecord,
  RunResumedRecord,
  RunStartedRecord,
} from "./log.ts";

/** An accepted outcome record, the record a trace occurrence holds. */
export type OutcomeRecord = NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord;

/** A node's bound backward handler, kept in the trace's private index. */
type BackwardHandler = (
  context: BackwardContext,
  feedback: readonly FeedbackValue[],
  options?: BackwardOptions,
) => BackwardResult | Promise<BackwardResult>;

/** A node's bound `learningMaterial`, kept in the trace's private index. */
type MaterialHandler = (
  inputs: Readonly<Record<string, unknown>> | null,
  steps: readonly ValueRecord[],
  caps: MaterialCaps,
) => Promise<LearningMaterial | null>;

/** One value `rebind` reads for a source: the typed value a pool name held, and the producer it names. */
export interface ReboundSource {
  /** The value the pool name held when the binding read it. */
  readonly value: TypedValue;
  /** The producer the input route names for it, or `null` for none. */
  readonly source: SourceRef | null;
}

/**
 * Bind an occurrence's node again from the values `resolve` gives for its sources (`rebind`, over
 * the node and wiring it ran with), returning the capture it collects (the bound inputs and
 * routes) and, for a binding that doesn't finish, its `BindingError`. `route` is the occurrence's
 * capture route, which holds whole any reserved value its hook resolved without the binder.
 */
export type Rebinder = (
  sources: ReadonlyMap<string, number>,
  resolve: (name: string, seq: number) => ReboundSource,
  route: ValueRoute,
) => { capture: BindingCapture; failure: BindingError | null };

/** What binding an occurrence's node again rebuilds (`TraceOccurrence`). */
export interface ReboundOccurrence {
  /** The canonical bound inputs, or `null` for a failure at a binding stage. */
  readonly inputs: JsonObject | null;
  /** The forward capture, each interpolation route's text rebuilt. */
  readonly capture: ForwardCapture;
  /** The effective input route over that capture. */
  readonly inputRoute: ValueRoute;
}

/** @internal Decode a typed value from its canonical log text, a fresh copy. */
export function decodeTyped(text: string): TypedValue {
  return TypedValue.fromWire(parseJson(text, "codec"));
}

/**
 * Join detached binding and backward material into the capture a record holds: the routes and
 * item routes, not the delivered values or the node data, which learning rebuilds.
 *
 * @param bindings Actual delivered values, routes and indexed item bindings frozen by the
 *   evaluator.
 * @param backward The same invocation's declared names and scope. Item seeds have no separate
 *   node inventory.
 * @returns Canonical material ready for accepted-record validation.
 * @throws TypeError when either transient carrier isn't exactly its class.
 * @throws RangeError or TypeCheckError when captured data can't satisfy the canonical wire contract.
 */
export function captureBindings(bindings: BindingCapture, backward: BackwardCapture): ForwardCapture {
  if (!isExactly(bindings, BindingCapture)) throw new TypeError("retained evaluation requires a BindingCapture");
  if (!isExactly(backward, BackwardCapture)) throw new TypeError("retained evaluation requires a BackwardCapture");
  return new ForwardCapture(bindings.route, {
    evaluatedKeys: bindings.evaluatedKeys,
    items: bindings.items.map(([index, item]) => [index, captureBindings(item, new BackwardCapture())] as const),
    scope: backward.scope,
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
 * episodes leave them out. `steps` are the steps the node recorded across its starts, which
 * learning reads (`occurrenceMaterial`). `capture` is the record's forward capture, or the one
 * rebuilt for a derived record. `boundInputs`, and the interpolation routes' text in `capture`
 * and `inputRoute`, are rebuilt by binding the node again on the first read.
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
  /** The effective input route as accepted; `inputRoute` reads it with its text rebuilt. */
  readonly #inputRoute: ValueRoute;
  /** The receiver's memory keys and occurrence path, if it has memory. */
  readonly memoryScope: MemoryScope | null;
  /** The bound inputs the node keeps from models; a private copy, read-only by contract. */
  readonly hiddenInputs: ReadonlySet<string>;
  /** An expansion's canonical seed, which its record leaves out; `null` for another outcome. */
  readonly seed: JsonValue | null;
  /** The node's recorded steps across its starts, in order; read-only by contract. */
  readonly steps: readonly ValueRecord[];
  /** The forward capture as accepted; `capture` reads it with its text rebuilt. */
  readonly #capture: ForwardCapture;
  /** Binds the node again, or `null` when none was given. */
  readonly #rebind: (() => ReboundOccurrence) | null;
  /** Whether binding again rebuilds the capture's interpolation text. */
  readonly #fillsText: boolean;
  /** What binding the node again gave, once read. */
  #rebound: ReboundOccurrence | null = null;
  /** Rebuilds the node data, or `null` when none was given. */
  readonly #rebuildNodeData: (() => JsonObject) | null;
  /** The rebuilt node data, once read. */
  #nodeData: JsonObject | null = null;

  /**
   * Encode `envelope` (which validates it, as an append does) and keep its decoded copy.
   *
   * @param inputRoute The effective input route: the captured route joined with pending seed
   *   routes.
   * @param options.memoryScope The receiver's memory scope; defaults to `null`.
   * @param options.hiddenInputs The bound inputs its node keeps from models; defaults to none.
   * @param options.seed An expansion's canonical seed; defaults to `null`.
   * @param options.steps The node's recorded steps; defaults to none.
   * @param options.capture The occurrence's forward capture; defaults to the record's.
   * @param options.rebind Binds the node again, which rebuilds the bound inputs; `null` (the
   *   default) leaves `boundInputs` unreadable.
   * @param options.fillsText Whether `rebind` also rebuilds the capture's interpolation text,
   *   which `capture` and `inputRoute` then read; defaults to `false` (they give it as held).
   * @param options.nodeDataRebuilder Rebuilds the node data from the node and its bound inputs;
   *   `null` (the default) leaves `nodeData` unreadable.
   * @throws TypeError when `envelope` isn't exactly a `RecordEnvelope`, `memoryScope` isn't
   *   `null` or exactly a `MemoryScope`, or there is no capture.
   * @throws RangeError when `fillsText` is given without `rebind`.
   */
  constructor(
    source: SourceRef,
    nodeId: string,
    outcomeSeq: number,
    envelope: RecordEnvelope,
    inputRoute: ValueRoute,
    options: {
      memoryScope?: MemoryScope | null;
      hiddenInputs?: ReadonlySet<string>;
      seed?: JsonValue | null;
      steps?: readonly ValueRecord[];
      capture?: ForwardCapture | null;
      rebind?: (() => ReboundOccurrence) | null;
      fillsText?: boolean;
      nodeDataRebuilder?: (() => JsonObject) | null;
    } = {},
  ) {
    const {
      memoryScope = null,
      hiddenInputs = new Set<string>(),
      seed = null,
      steps = [],
      capture = null,
      rebind = null,
      fillsText = false,
      nodeDataRebuilder = null,
    } = options;
    if (fillsText && rebind === null) throw new RangeError("rebuilding interpolation text requires a rebinder");
    this.source = source;
    this.nodeId = nodeId;
    this.outcomeSeq = outcomeSeq;
    this.recordJson = encodeRecord(envelope);
    this.envelope = decodeRecord(this.recordJson);
    const held = capture ?? (this.envelope.record as OutcomeRecord).forwardCapture;
    if (!isExactly(held, ForwardCapture)) throw new TypeError("a trace occurrence requires its forward capture");
    this.#capture = held;
    this.#rebind = rebind;
    this.#fillsText = fillsText;
    this.#rebuildNodeData = nodeDataRebuilder;
    this.#inputRoute = inputRoute;
    this.memoryScope = checkMemoryScope(memoryScope);
    this.hiddenInputs = new Set(hiddenInputs);
    this.seed = seed === null ? null : canonicalSeed(seed)[0];
    this.steps = Object.freeze([...steps]);
    Object.freeze(this);
  }

  /** The accepted outcome record, read from the decoded envelope. */
  get record(): OutcomeRecord {
    // Occurrences are prepared only for outcome records.
    return this.envelope.record as OutcomeRecord;
  }

  /**
   * The occurrence's forward capture: its record's, or the one rebuilt for a derived record. A
   * recorded capture holding an interpolation route gets each one's text by binding the node
   * again on the first read; any other capture is given as held.
   *
   * @throws RangeError, when it binds again, as `boundInputs`, or when binding again collects
   *   routes of another shape.
   */
  get capture(): ForwardCapture {
    return this.#fillsText ? this.#reboundOccurrence().capture : this.#capture;
  }

  /**
   * The effective input route: the captured route joined with pending seed routes, with each
   * interpolation route's text rebuilt as in `capture`.
   *
   * @throws RangeError as `capture`.
   */
  get inputRoute(): ValueRoute {
    return this.#fillsText ? this.#reboundOccurrence().inputRoute : this.#inputRoute;
  }

  /**
   * The bound inputs, canonical JSON rebuilt by binding the node again from the values its route
   * read (`reboundCapture`), or `null` for a failure at a binding stage, whose binding didn't
   * finish. Built on the first read.
   *
   * @throws RangeError when the occurrence was made without a rebinder, a route source has no
   *   accepted producer, or a finished binding no longer finishes (the `BindingError` as
   *   `cause`): the log and the flow disagree.
   */
  get boundInputs(): JsonObject | null {
    if (this.#rebind === null) throw new RangeError(`occurrence of ${reprText(this.nodeId)} can't rebuild its bound inputs`);
    return this.#reboundOccurrence().inputs;
  }

  /**
   * The node data backward reads (`BackwardContext.nodeData`), canonical JSON rebuilt from the
   * node and its bound inputs (`rebuildNodeData`): its static material and, for a node whose
   * `forward` rendered, the render and the model config it used. Built on the first read.
   *
   * @throws RangeError when the occurrence was made without a rebuilder, as `boundInputs` (a
   *   finished binding whose bound inputs don't rebuild included), or when the run recorded no
   *   model config for the node.
   */
  get nodeData(): JsonObject {
    if (this.#rebuildNodeData === null) {
      throw new RangeError(`occurrence of ${reprText(this.nodeId)} can't rebuild its node data`);
    }
    this.#nodeData ??= this.#rebuildNodeData();
    return this.#nodeData;
  }

  /** What binding the node again gives, bound on the first read. */
  #reboundOccurrence(): ReboundOccurrence {
    this.#rebound ??= this.#rebind!();
    return this.#rebound;
  }
}

/**
 * Bind an occurrence's node again (`rebind`) from the values its capture route read, each looked
 * up in the trace's history by the producer the route names (`traceResolver`), so it works after
 * the pool has moved on. The run's system values are added whole, since the route names no
 * producer for them.
 *
 * @returns What the rebinder returns: the capture binding again collects, and the `BindingError`
 *   of a binding that doesn't finish.
 * @throws RangeError when a route source has no accepted producer.
 */
export function reboundCapture(
  trace: RetainedTrace,
  route: ValueRoute,
  rebind: Rebinder,
): { capture: BindingCapture; failure: BindingError | null } {
  const sources = new Map<string, number>();
  // System values never change within a run, so their seq isn't checked (`traceResolver`).
  for (const key of trace.system.keys()) sources.set(`${SYSTEM_PREFIX}${key}`, 0);
  const pending = [route];
  while (pending.length > 0) {
    const node = pending.pop()!;
    pending.push(...node.children);
    if (node.source === null) continue;
    const seq = node.source.executionSeq;
    // A route source names its producer's start, not the pool name the binding read.
    const name = trace._sourceNames.get(sourceKey(new SourceRef(node.source.runId, seq)));
    if (name === undefined) throw new RangeError(`route source ${seq} names no accepted producer`);
    // One binding reads each pool name from its one current producer.
    sources.set(name, seq);
  }
  return rebind(sources, traceResolver(trace), route);
}

/**
 * Return the canonical bound inputs of `nodeId`'s finished binding, bound again as `rebound`.
 *
 * @throws RangeError when the binding no longer finishes (the `BindingError` as `cause`): the log
 *   and the flow disagree.
 */
function finishedInputs(nodeId: string, rebound: { capture: BindingCapture; failure: BindingError | null }): JsonObject {
  const { capture, failure } = rebound;
  if (failure !== null) {
    throw new RangeError(`the bound inputs of ${reprText(nodeId)} don't rebuild: ${failure.message}`, { cause: failure });
  }
  return canonicalSeed(capture.boundInputs)[1] as JsonObject;
}

/** Whether `capture`'s route, or an item's, holds an interpolation route. */
function holdsInterpolation(capture: ForwardCapture): boolean {
  const pending = [capture.route];
  while (pending.length > 0) {
    const route = pending.pop()!;
    if (route.kind === "interpolation") return true;
    pending.push(...route.children);
  }
  return capture.items.some(([, item]) => holdsInterpolation(item));
}

/**
 * Return `capture` with each interpolation route's text taken from `binding`, the capture binding
 * its node again collects: inputs matched by name, items by index, everything inside by position.
 * Everything else stays as recorded; `capture` itself when no route changed.
 *
 * @throws RangeError when `binding` lacks an input or item, or has a route of another shape.
 */
function withInterpolationText(capture: ForwardCapture, binding: BindingCapture): ForwardCapture {
  const { route } = capture;
  const again = binding.route;
  const fields = route.names.map((name, index) => {
    const at = again.names.indexOf(name);
    if (at < 0) throw new RangeError(`binding again collects no input ${reprText(name)}`);
    return [name, withRouteText(route.children[index]!, again.children[at]!)] as const;
  });
  const reboundItems = new Map(binding.items);
  const items = capture.items.map(([index, item]) => {
    const rebound = reboundItems.get(index);
    if (rebound === undefined) throw new RangeError(`binding again collects no item ${index}`);
    return [index, withInterpolationText(item, rebound)] as const;
  });
  const routeUnchanged = fields.every(([, child], index) => child === route.children[index]);
  if (routeUnchanged && items.every(([, item], index) => item === capture.items[index]![1])) return capture;
  return new ForwardCapture(routeUnchanged ? route : ValueRoute.record(fields), {
    evaluatedKeys: capture.evaluatedKeys,
    items,
    scope: capture.scope,
    inputNames: capture.inputNames,
    variableNames: capture.variableNames,
  });
}

/**
 * Return `route` with each interpolation route's text taken from the route at its position in
 * `again`; `route` itself when it holds none.
 *
 * @throws RangeError when `again` has another kind or number of children at some position.
 */
function withRouteText(route: ValueRoute, again: ValueRoute): ValueRoute {
  if (route.kind !== again.kind || route.children.length !== again.children.length) {
    throw new RangeError(`binding again collects a ${again.kind} route where the record has a ${route.kind} route`);
  }
  const children = route.children.map((child, index) => withRouteText(child, again.children[index]!));
  if (route.kind === "interpolation") return route.with({ valueJson: again.valueJson, children });
  return children.every((child, index) => child === route.children[index]) ? route : route.with({ children });
}

/**
 * Join `capture`'s route with an owner's pending seed routes: each field the capture didn't
 * evaluate here takes the seed route given for it.
 */
function joinSeedRoutes(capture: ForwardCapture, seeds: SeedRoutes | null): ValueRoute {
  if (seeds === null) return capture.route;
  const fields = new Map(seeds.fields);
  return ValueRoute.record(
    capture.route.names.map((name, index) => {
      const route = capture.route.children[index]!;
      return [name, capture.evaluatedKeys.has(name) ? route : (fields.get(name) ?? route)] as const;
    }),
  );
}

/**
 * Return a `rebind` resolver over the trace's history: pool name `name` as the producer whose
 * accepted start is `seq` wrote it, read through its files type as the pool reads it; a system
 * value is the run's, with no producer.
 *
 * @throws RangeError (from the resolver) when no accepted producer has start `seq`, or it wrote
 *   another name, or the run has no system value `name`.
 */
export function traceResolver(trace: RetainedTrace): (name: string, seq: number) => ReboundSource {
  return (name, seq) => {
    if (name.startsWith(SYSTEM_PREFIX)) {
      const text = trace.system.get(name.slice(SYSTEM_PREFIX.length));
      if (text === undefined) throw new RangeError(`the run has no system value ${reprText(name)}`);
      return { value: decodeTyped(text), source: null };
    }
    const key = sourceKey(new SourceRef(trace.runId, seq));
    const entry = trace._sources.get(key);
    if (entry === undefined || trace._sourceNames.get(key) !== name) {
      throw new RangeError(`no accepted producer at seq ${seq} wrote ${reprText(name)}`);
    }
    return { value: normalizeTypedFiles(entry.files, decodeTyped(entry.valueJson)), source: entry.source };
  };
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
  /** @internal The pool name each accepted source entry wrote, by `sourceKey`. */
  _sourceNames = new Map<string, string>();
  /** @internal Each current pool name's producer entry. */
  _currentWriters = new Map<string, SourceEntry>();
  /** @internal Accepted unfinished starts: node id to start sequence. */
  _started = new Map<string, number>();
  /**
   * @internal The starts of waiting nodes, moved out of `_started` while they wait (a resume
   * doesn't retry them) and back when they wake: node id to start sequence.
   */
  _parked = new Map<string, number>();
  /** @internal Bound backward handlers by occurrence `sourceKey`. */
  _backwardHandlers = new Map<string, BackwardHandler>();
  /** @internal Bound learning material builders by occurrence `sourceKey`. */
  _materialHandlers = new Map<string, MaterialHandler>();
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
   * Build detached context from accepted invocation facts; the bound inputs and node data are
   * rebuilt from the node (`TraceOccurrence`), and no other hook is called.
   *
   * The node data gains the reserved `_evaluation_evidence` entry: what the occurrence showed its
   * model, built from `options.material` (`evaluationEvidence`), or `null` without one.
   *
   * @param source The accepted start of an executed occurrence.
   * @param options.downstreamContext Transient attribution materialized by a caller, copied with
   *   the other context values; defaults to empty.
   * @param options.material The occurrence's learning material (`occurrenceMaterial`), if any.
   * @throws RangeError when the occurrence or its accepted output source is absent, the rebuilt
   *   node data already holds `_evaluation_evidence`, or a value breaks the context contract.
   * @throws TypeError when supplied context material breaks the context contract.
   */
  backwardContext(
    source: SourceRef,
    options: {
      downstreamContext?: readonly Readonly<Record<string, unknown>>[];
      material?: LearningMaterial | null;
    } = {},
  ): BackwardContext {
    const { downstreamContext = [], material = null } = options;
    const occurrence = this.occurrence(source);
    const record = occurrence.record;
    const capture = occurrence.capture;
    const data = occurrence.nodeData;
    if (Object.hasOwn(data, "_evaluation_evidence")) {
      throw new RangeError("node data uses reserved _evaluation_evidence");
    }
    const failure = isExactly(record, NodeFailedRecord)
      ? record.failure
      : isExactly(record, NodeRoutedRecord)
        ? record.terminalFailure
        : null;
    return new BackwardContext(occurrence.boundInputs, {
      inputNames: capture.inputNames,
      variableNames: capture.variableNames,
      output: isExactly(record, NodeSucceededRecord) ? this.lookupSource(source) : null,
      failure,
      scope: capture.scope,
      runId: source.runId,
      nodeId: occurrence.nodeId,
      nodeStartedSeq: source.executionSeq,
      nodeData: { ...data, _evaluation_evidence: evaluationEvidence(occurrence, material) },
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
    const read = sourcePath(path, this.startId, (name) => this._currentWriters.has(name));
    if (read === null) return new ResolvedValue(null, false);
    const { name, projection } = read;
    if (name.startsWith(SYSTEM_PREFIX)) {
      const encoded = this.system.get(name.slice(SYSTEM_PREFIX.length));
      if (encoded === undefined) return new ResolvedValue(null, false);
      return new ResolvedValue(decodeTyped(encoded).toObject(), true);
    }
    const entry = this._currentWriters.get(name);
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

/**
 * Return the bound inputs `occurrence` showed models: its bound inputs without the ones its node
 * keeps from models, or `null` when its binding didn't finish.
 */
export function visibleInputs(occurrence: TraceOccurrence): JsonObject | null {
  const inputs = occurrence.boundInputs;
  if (inputs === null || occurrence.hiddenInputs.size === 0) return inputs;
  return Object.fromEntries(Object.entries(inputs).filter(([name]) => !occurrence.hiddenInputs.has(name)));
}

/**
 * Return what AGENT backward reads as `_evaluation_evidence`: the occurrence's memory scope and
 * visible bound inputs, and `material`'s prompts, memory sections and interactions; `null` without
 * material.
 */
export function evaluationEvidence(occurrence: TraceOccurrence, material: LearningMaterial | null): JsonObject | null {
  if (material === null) return null;
  const scope = occurrence.memoryScope;
  return {
    scope: scope === null ? null : scopePayload(scope),
    bound_inputs: visibleInputs(occurrence),
    base_system_prompt: material.baseSystemPrompt,
    prompt: material.prompt,
    memory_sections: material.memorySections.map(memorySectionPayload),
    interactions: material.interactions.map(interactionPayload),
  };
}

/** Encode one memory scope as `_evaluation_evidence` holds it. */
function scopePayload(scope: MemoryScope): JsonObject {
  const segment = (item: MemoryOccurrenceSegment): JsonObject => ({
    kind: item.kind,
    static_path: [...item.staticPath],
    index: BigInt(item.index),
  });
  return {
    static_path: [...scope.staticPath],
    memories: [...scope.memories],
    occurrence_path: scope.occurrencePath.map(segment),
  };
}

/** Encode one recalled memory section as `_evaluation_evidence` holds it. */
function memorySectionPayload(section: MemorySection): JsonObject {
  return { memory: section.memory, text: section.text };
}

/** Encode one visible interaction as `_evaluation_evidence` holds it. */
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
 * Rebuild what learning reads about one accepted occurrence: its node's `learningMaterial` over
 * its bound inputs and recorded steps.
 *
 * @param source The accepted start of an executed occurrence.
 * @param caps The run's recorded skills and tool specs, model access and recall.
 * @returns The material, or `null` for a node that shows a learner none.
 * @throws RangeError when no accepted occurrence has this identity; what the node's hook throws.
 */
export async function occurrenceMaterial(
  trace: RetainedTrace,
  source: SourceRef,
  caps: MaterialCaps,
): Promise<LearningMaterial | null> {
  const occurrence = trace.occurrence(source);
  const build = trace._materialHandlers.get(sourceKey(source))!;
  return await build(occurrence.boundInputs, occurrence.steps, caps);
}

/** The optional parts of a `TraceDelta`; each defaults to `null`, `false` or an empty array. */
export interface TraceDeltaFields {
  /** The trace a learnable RunStarted creates. */
  initialized?: RetainedTrace | null;
  /** A started node's id and start sequence. */
  started?: readonly [string, number] | null;
  /** Whether a resume clears the unfinished starts. */
  resumed?: boolean;
  /** A node that starts waiting: its start moves from the unfinished starts to the parked ones. */
  parked?: string | null;
  /** A woken node: its start moves back from the parked starts to the unfinished ones. */
  unparked?: string | null;
  /** The accepted outcome's occurrence. */
  occurrence?: TraceOccurrence | null;
  /** The occurrence node's bound backward handler. */
  handler?: BackwardHandler | null;
  /** The occurrence node's bound learning material builder. */
  material?: MaterialHandler | null;
  /** The accepted output's commit target and its source entry. */
  writer?: readonly [string, SourceEntry] | null;
  /** A system value the run start commits: its name and canonical typed log text. */
  systemValue?: readonly [string, string] | null;
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
  /** A node that starts waiting, whose start is parked, or `null`. */
  readonly parked: string | null;
  /** A woken node, whose parked start comes back, or `null`. */
  readonly unparked: string | null;
  /** The accepted outcome's occurrence, or `null`. */
  readonly occurrence: TraceOccurrence | null;
  /** The occurrence node's bound backward handler, or `null`. */
  readonly handler: BackwardHandler | null;
  /** The occurrence node's bound learning material builder, or `null`. */
  readonly material: MaterialHandler | null;
  /** The accepted output's commit target and its source entry, or `null`. */
  readonly writer: readonly [string, SourceEntry] | null;
  /** A system value the run start commits: its name and canonical typed log text, or `null`. */
  readonly systemValue: readonly [string, string] | null;
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
    this.parked = fields.parked ?? null;
    this.unparked = fields.unparked ?? null;
    this.occurrence = fields.occurrence ?? null;
    this.handler = fields.handler ?? null;
    this.material = fields.material ?? null;
    this.writer = frozenPair(fields.writer ?? null);
    this.systemValue = frozenPair(fields.systemValue ?? null);
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
    // A resume retries only running nodes, so the parked starts stay.
    if (this.resumed) trace._started.clear();
    if (this.parked !== null) {
      trace._parked.set(this.parked, trace._started.get(this.parked)!);
      trace._started.delete(this.parked);
    }
    if (this.unparked !== null) {
      trace._started.set(this.unparked, trace._parked.get(this.unparked)!);
      trace._parked.delete(this.unparked);
    }
    if (this.occurrence !== null) {
      const key = sourceKey(this.occurrence.source);
      trace._occurrences.push(this.occurrence);
      trace.occurrenceBySource.set(key, this.occurrence);
      trace._backwardHandlers.set(key, this.handler!);
      trace._materialHandlers.set(key, this.material!);
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
      // An outcome's writer always has its occurrence's source, and the START inputs' source 0.
      trace._sources.set(sourceKey(entry.source!), entry);
      trace._sourceNames.set(sourceKey(entry.source!), target);
    }
    if (this.systemValue !== null) trace.system.set(this.systemValue[0], this.systemValue[1]);
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

/** The key prefix of a system value's run-start commit: `system.<name>`. */
export const SYSTEM_PREFIX = "system.";

/**
 * Parse a `${path}` lookup into the pool name it reads and the field path inside that value, or
 * `null` for a miss: `<id>.output...` reads `<id>` when `holds(id)`, `input...` the START's value
 * (`startId`) and `system.<key>` the system value `system.<key>`, which is read whole.
 *
 * @throws ExpressionError when a dotted segment of `path` is empty.
 */
export function sourcePath(
  path: string,
  startId: string,
  holds: (name: string) => boolean,
): { name: string; projection: string[] } | null {
  const parts = path.trim().split(".");
  if (parts.some((part) => !part)) throw new ExpressionError(`invalid variable path: '\${${path}}'`);
  const [head, ...rest] = parts as [string, ...string[]];
  if (holds(head) && rest[0] === "output") return { name: head, projection: rest.slice(1) };
  if (head === "input" && rest.length) return { name: startId, projection: rest };
  if (head === "system" && rest.length) return { name: `${SYSTEM_PREFIX}${rest[0]}`, projection: [] };
  return null;
}

/**
 * Pair one `${path}` lookup's value and presence in `pool` with the source `sourceOf` names for
 * the pool name it reads (`sourcePath`), projected to the lookup's field path; a miss, or a field
 * the value lacks, is absent.
 *
 * @throws ExpressionError when a dotted segment of `path` is empty.
 * @throws whatever `sourceOf` throws.
 */
export function resolvePoolPath(
  path: string,
  pool: VariablePool,
  sourceOf: (name: string) => SourceRef | null,
): ResolvedValue {
  const read = sourcePath(path, pool.startId, (name) => pool.store.has(name));
  if (read === null) return new ResolvedValue(null, false);
  const { name, projection } = read;
  const typed = name.startsWith(SYSTEM_PREFIX) ? pool.system.get(name.slice(SYSTEM_PREFIX.length)) : pool.store.get(name);
  if (typed === undefined) return new ResolvedValue(null, false);
  let value = typed.toObject();
  for (const part of projection) {
    const [present, next] = fieldOf(value, part);
    if (!present) return new ResolvedValue(null, false);
    value = next;
  }
  const source = sourceOf(name);
  return new ResolvedValue(value, true, source === null ? null : new SourceRef(source.runId, source.executionSeq, projection));
}

/** The options of `prepareTrace`. */
export interface PrepareTraceOptions {
  /** The seed connections the outcome's growth installs; defaults to none. */
  inputConnections?: readonly SeedConnection[];
  /** The child flows the outcome's growth installs; defaults to none. */
  childBoundaries?: readonly ChildFlowBoundary[];
  /** The settled prune ids, earlier deferred ones and the nodes grown under them; defaults to none. */
  retiredIds?: readonly string[];
  /** The root START id, required for a learnable RunStarted. */
  rootStartId?: string | null;
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
  /** An accepted success's commit: its pool key and canonical typed value text. */
  committed?: { readonly key: string; readonly valueJson: string } | null;
  /** Whether a `PoolCommitRecord` is one of the run start's commits, which apply at once. */
  initialCommit?: boolean;
  /** An accepted expansion's canonical seed, which its record leaves out. */
  seed?: JsonValue | null;
  /** The executing node's recorded steps across its starts, which a learnable occurrence keeps. */
  steps?: readonly ValueRecord[];
  /** A derived record's forward capture, rebuilt from its sources, which the record leaves out. */
  capture?: ForwardCapture | null;
  /** Binds the executing node again, which rebuilds the occurrence's bound inputs. */
  rebinder?: Rebinder | null;
  /**
   * Rebuilds the executing node's node data from its bound inputs (`null` for none); a finished
   * binding needs `rebinder` too.
   */
  nodeDataRebuilder?: ((inputs: Readonly<Record<string, unknown>> | null) => JsonObject) | null;
}

/**
 * Validate capture authority and prepare a detached accepted-record delta.
 *
 * A learnable RunStarted creates an empty trace, and the run start's commits install the system
 * values and the START inputs (as source 0); a RunStarted that doesn't learn, and any record
 * before one, prepares `null`. A node's commit, and a send, only advance the sequence. A node that
 * starts waiting parks its start, and its wake brings the start back. Outcomes join pending seed
 * routes into their input route (evaluated keys keep their own routes) and install their writer,
 * seed connections, child boundaries and terminal.
 *
 * @throws RangeError when the root metadata is missing, a capture has no trace, a derived record
 *   holds a capture (it is rebuilt from its sources), the record's
 *   coordinates or terminal payload disagree with the trace, a resume disagrees with the
 *   unfinished starts, a waiting node has no accepted start or a woken one no parked start, an
 *   outcome lacks its capture or its accepted start, the capture's scope
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
    rootStartId = null,
    rootEndId = null,
    rootInputNames = null,
    terminalSucceeded = null,
    terminalOutputJson = null,
    terminalFailure = null,
    writerFiles = null,
    committed = null,
    initialCommit = false,
    seed = null,
    steps = [],
    capture: rebuilt = null,
    rebinder = null,
    nodeDataRebuilder: rebuildNodeData = null,
  } = options;
  const record = envelope.record;
  if (isExactly(record, RunStartedRecord)) {
    if (!record.learnable) return null;
    if (typeof rootEndId !== "string" || rootInputNames === null) {
      throw new RangeError("retained RunStarted requires verified root metadata");
    }
    if (typeof rootStartId !== "string") throw new RangeError("retained RunStarted requires verified root metadata");
    const initial = new RetainedTrace(envelope.runId, rootStartId, rootEndId, rootInputNames);
    return new TraceDelta(null, envelope.seq, { initialized: initial });
  }
  const recorded = (record as { forwardCapture?: ForwardCapture | null }).forwardCapture ?? null;
  const capture = rebuilt ?? recorded;
  if (trace === null) {
    if (capture !== null) throw new RangeError("forward capture requires recorded learning");
    return null;
  }
  if (rebuilt !== null && recorded !== null) throw new RangeError("a derived record holds no forward capture");
  if (trace.runId !== envelope.runId || trace.nextSequence !== envelope.seq) {
    throw new RangeError("trace disagrees with accepted record coordinates");
  }
  if (terminalSucceeded === null && (terminalOutputJson !== null || terminalFailure !== null)) {
    throw new RangeError("terminal payload requires a terminal disposition");
  }
  if (isExactly(record, PoolCommitRecord)) {
    if (!initialCommit) return new TraceDelta(trace, envelope.seq);
    // The record holds a canonical wire.
    const valueJson = logJson(record.value);
    if (record.key !== trace.startId) {
      return new TraceDelta(trace, envelope.seq, { systemValue: [record.key.slice(SYSTEM_PREFIX.length), valueJson] });
    }
    // The run inputs keep their own source, seq 0: an internal number, not a record's seq.
    const entry = new SourceEntry(new SourceRef(envelope.runId, 0), 0, valueJson);
    // A flow with only a START has it as its END too, so the inputs are the root writer.
    return new TraceDelta(trace, envelope.seq, {
      writer: [record.key, entry],
      rootWriter: record.key === trace.endId ? entry : null,
    });
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
  if (isExactly(record, NodeWaitingRecord)) {
    if (!trace._started.has(record.nodeId)) throw new RangeError("trace waiting node requires its accepted start");
    return new TraceDelta(trace, envelope.seq, { parked: record.nodeId });
  }
  if (isExactly(record, NodeWokenRecord)) {
    if (!trace._parked.has(record.nodeId)) throw new RangeError("trace woken node requires its parked start");
    return new TraceDelta(trace, envelope.seq, { unparked: record.nodeId });
  }
  if (isExactly(record, RunFailedRecord)) {
    // No node completes a failure at quiescence: the route that proved END unreachable caused it.
    if (trace.causalRouteSource === null) throw new RangeError("trace run failure requires its causal route occurrence");
    return new TraceDelta(trace, envelope.seq, {
      terminal: new TraceTerminal(false, trace.causalRouteSource, null, record.failure),
    });
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
  const inputRoute = joinSeedRoutes(capture, candidates);
  validateSources(trace, inputRoute, envelope.seq);
  // A derived record's capture is rebuilt whole with its value; a recorded one's interpolation
  // routes get their text by binding the node again, which learning does on its first read. A
  // failure at a binding stage bound no inputs, so it is bound again only for that text.
  const fillsText = rebuilt === null && holdsInterpolation(capture);
  const bound = !unbound(record);
  if (bound && rebuildNodeData !== null && rebinder === null) {
    throw new RangeError("rebuilding the node data of a finished binding requires a rebinder");
  }
  let again: { capture: BindingCapture; failure: BindingError | null } | null = null;
  const bindAgain = () => (again ??= reboundCapture(trace, capture.route, rebinder!));
  const rebind = (): ReboundOccurrence => {
    if (!bound && !fillsText) return { inputs: null, capture, inputRoute };
    const rebound = bindAgain();
    const inputs = bound ? finishedInputs(node.id, rebound) : null;
    if (!fillsText) return { inputs, capture, inputRoute };
    const filled = withInterpolationText(capture, rebound.capture);
    return { inputs, capture: filled, inputRoute: filled === capture ? inputRoute : joinSeedRoutes(filled, candidates) };
  };
  const occurrence = new TraceOccurrence(source, node.id, envelope.seq, envelope, inputRoute, {
    memoryScope: node.memoryScope,
    hiddenInputs: node.modelHiddenInputs(),
    seed,
    steps,
    capture,
    rebind: rebinder !== null ? rebind : bound ? null : () => ({ inputs: null, capture, inputRoute }),
    fillsText: fillsText && rebinder !== null,
    nodeDataRebuilder:
      rebuildNodeData === null
        ? null
        : () => {
            if (!bound) return rebuildNodeData(null);
            const rebound = bindAgain();
            finishedInputs(node.id, rebound);
            // The typed values, not `finishedInputs`' canonical JSON: the render needs them as bound.
            return rebuildNodeData(rebound.capture.boundInputs);
          },
  });
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
  if (typeof node.learningMaterial !== "function") throw new TypeError("node learningMaterial must be callable");
  const material: MaterialHandler = node.learningMaterial.bind(node);
  if (isExactly(record, NodeSucceededRecord) && committed === null) {
    throw new RangeError("an accepted success requires its commit");
  }
  const writer = isExactly(record, NodeSucceededRecord)
    ? ([committed!.key, new SourceEntry(source, envelope.seq, committed!.valueJson, writerFiles)] as const)
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
    material,
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

/** Whether `record` is a failure at a binding stage, which bound no inputs. */
function unbound(record: RecordEnvelope["record"]): boolean {
  return isExactly(record, NodeFailedRecord) && (BINDING_FAILURE_CODES as readonly string[]).includes(record.failure.code);
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
