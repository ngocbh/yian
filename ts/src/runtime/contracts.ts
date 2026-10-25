/**
 * Immutable forward captures and backward outcomes shared by runtime consumers.
 *
 * These values depend only on lower node, expression, memory, and diagnostic contracts. They
 * never import the runtime log, state, trace, driver, or public run boundary.
 *
 * Knows about:   diagnostics, expr/lineage, memory/contracts, nodes/base, nodes/contracts,
 *                typesys/json, typesys/values.
 * Never imports: runtime log, state, trace, engine or run, compile, compose, or the CLI.
 */

import { Failure, sameFailure } from "../diagnostics.ts";
import {
  type FeedbackValue,
  type RouteKind,
  SourceRef,
  sameFeedback,
  sameSource,
  ValueRoute,
  validateFeedback,
} from "../expr/lineage.ts";
import {
  LearningResult,
  type MemoryMutation,
  MemoryOperationFailure,
  sameLearningResult,
  sameMemoryOperationFailure,
  validateMutations,
} from "../memory/contracts.ts";
import {
  BackwardResult,
  copyBackwardValue,
  retainFailure,
  sameBackwardResult,
  validateDeclaredNames,
} from "../nodes/base.ts";
import { ExecutionScope, isExactly, sameExecutionScope, sameItems, sameOptional } from "../nodes/contracts.ts";
import { type JsonValue, parseJson } from "../typesys/json.ts";
import { isList } from "../typesys/values.ts";

/** A route field besides `kind` and `expression`. */
type RouteField =
  | "source"
  | "valueJson"
  | "children"
  | "names"
  | "sizes"
  | "spans"
  | "path"
  | "resultType"
  | "snapshot";

/** Each route field with the snake_case name its messages use, in the order they are checked. */
const ROUTE_FIELDS: readonly (readonly [RouteField, string])[] = [
  ["source", "source"],
  ["valueJson", "value_json"],
  ["children", "children"],
  ["names", "names"],
  ["sizes", "sizes"],
  ["spans", "spans"],
  ["path", "path"],
  ["resultType", "result_type"],
  ["snapshot", "snapshot"],
];

/** The route fields whose default is `null`; every other field defaults to an empty array. */
const NULLABLE_FIELDS: ReadonlySet<RouteField> = new Set(["source", "valueJson", "resultType", "snapshot"]);

/** The fields each route kind may carry; every other field stays at its default. */
const ROUTE_VARIANTS: Readonly<Record<RouteKind, readonly RouteField[]>> = {
  literal: ["valueJson", "resultType", "snapshot"],
  source: ["source"],
  selected: ["children"],
  list: ["children"],
  record: ["children", "names"],
  concat: ["children", "sizes"],
  scalar: ["children", "valueJson", "resultType", "snapshot"],
  interpolation: ["children", "valueJson", "spans"],
  opaque: ["children", "names", "valueJson", "resultType", "snapshot"],
  project: ["children", "path"],
};

/** Whether `value` is a non-negative safe integer `number`. */
function isIndex(value: unknown): boolean {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

/**
 * Copy an array by index, so the checked copy is the stored one: a hole reads as `undefined`, and
 * an own `Symbol.iterator` is ignored.
 */
function copyList<T>(value: readonly T[]): T[] {
  return Array.from({ length: value.length }, (_, index) => value[index]!);
}

/**
 * Require a source identity, without normalizing forged carrier fields.
 *
 * @throws TypeError when `source` isn't exactly a `SourceRef`.
 * @throws RangeError for an empty run id, a sequence that isn't a non-negative integer, or a bad
 *   path part.
 */
export function validateSource(source: unknown): asserts source is SourceRef {
  if (!isExactly(source, SourceRef)) throw new TypeError("source must be an exact SourceRef");
  if (typeof source.runId !== "string" || !source.runId) throw new RangeError("source run_id must be non-empty");
  if (!isIndex(source.executionSeq)) {
    throw new RangeError("source execution_seq must be a non-negative integer");
  }
  validateRoutePath(source.path);
}

/**
 * Require field/index path parts that are strings or non-negative integers.
 *
 * @throws RangeError for any other part.
 */
function validateRoutePath(path: readonly unknown[]): void {
  if (path.some((part) => typeof part !== "string" && !isIndex(part))) {
    throw new RangeError("route path components must be strings or non-negative integers");
  }
}

/**
 * Require every route field and variant to keep the route contract, at the durable capture
 * boundary. Checks run in a fixed order and the first broken one throws.
 *
 * @param options.deep Whether to also check `route`'s children and snapshot (default `true`).
 *   `false` skips only those recursive checks; the snapshot kind walk still runs. It is for
 *   callers that already checked each child and the snapshot, so a tree built bottom-up checks
 *   each node once.
 * @throws TypeError when `route` isn't exactly a `ValueRoute` or a field has the wrong type.
 * @throws RangeError when a field or variant breaks the route contract.
 * @throws JSONDecodeError when `valueJson` isn't JSON yian reads.
 */
export function validateRoute(route: unknown, options: { deep?: boolean } = {}): asserts route is ValueRoute {
  const { deep = true } = options;
  if (!isExactly(route, ValueRoute)) throw new TypeError("route must be an exact ValueRoute");
  const kind = route.kind;
  if (typeof kind !== "string" || !Object.hasOwn(ROUTE_VARIANTS, kind)) throw new RangeError("unknown route kind");
  if (typeof route.expression !== "string") throw new TypeError("route expression must be a string");
  const variant = ROUTE_VARIANTS[kind];
  for (const [field, name] of ROUTE_FIELDS) {
    const value = route[field];
    const isDefault = NULLABLE_FIELDS.has(field) ? value === null : (value as readonly unknown[]).length === 0;
    if (!variant.includes(field) && !isDefault) throw new RangeError(`route ${kind} cannot carry ${name}`);
  }
  validateRoutePath(route.path);
  if (deep) {
    for (const child of route.children) validateRoute(child);
  }
  if (route.names.some((name) => typeof name !== "string")) throw new RangeError("route names must be strings");
  if (kind === "record" && new Set(route.names).size !== route.names.length) {
    throw new RangeError("record route names must be unique");
  }
  if (!route.sizes.every(isIndex)) throw new RangeError("route sizes must be non-negative integers");
  if (kind === "source") validateSource(route.source);
  if ((kind === "selected" || kind === "project") && route.children.length !== 1) {
    throw new RangeError("selected and project routes require one child");
  }
  if (kind === "project" && route.path.length === 0) throw new RangeError("project route requires a path");
  if ((kind === "record" || kind === "opaque") && route.names.length !== route.children.length) {
    throw new RangeError(`${kind} route names must match its children`);
  }
  if (kind === "concat" && (route.children.length !== 2 || route.sizes.length !== 2)) {
    throw new RangeError("concat route requires two children and sizes");
  }
  if (kind === "scalar" && route.children.length === 0) throw new RangeError("scalar route requires operands");
  let result: JsonValue | undefined;
  if (route.valueJson !== null) {
    if (typeof route.valueJson !== "string") throw new TypeError("route value_json must be a string or null");
    // The reader refuses duplicate keys, NaN tokens and lone surrogates, and yields only JSON
    // values, so the parse is the whole check.
    result = parseJson(route.valueJson, "codec");
  }
  if (route.resultType !== null && (typeof route.resultType !== "string" || !route.resultType)) {
    throw new RangeError("route result_type must be a non-empty string or null");
  }
  const hasResult = kind === "literal" || kind === "scalar" || kind === "opaque";
  if (hasResult && (route.valueJson === null) === (route.resultType === null)) {
    throw new RangeError("route requires either a JSON result or an unavailable type");
  }
  if (route.snapshot !== null) {
    if (route.resultType === null || route.valueJson !== null) {
      throw new RangeError("partial snapshot requires an unavailable result");
    }
    if (deep) validateRoute(route.snapshot);
    const pending = [route.snapshot];
    while (pending.length > 0) {
      const current = pending.pop()!;
      if (current.kind !== "literal" && current.kind !== "list" && current.kind !== "record") {
        throw new RangeError("partial snapshot may contain only literal, list and record routes");
      }
      for (const child of current.children) pending.push(child);
    }
  }
  if (kind === "interpolation") {
    // A recorded interpolation holds no text: learning rebuilds it (`TraceOccurrence.capture`).
    const textless = route.valueJson === null;
    if ((!textless && typeof result !== "string") || route.children.length === 0 || route.spans.length !== route.children.length) {
      throw new RangeError("interpolation requires a string result or none, and one span per child");
    }
    // Spans count code points, as interpolation contexts do.
    const length = textless ? Infinity : [...(result as string)].length;
    let previousEnd = 0;
    for (const [start, end] of route.spans) {
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) {
        throw new TypeError("interpolation spans must be integer pairs");
      }
      if (!(previousEnd <= start && start <= end && end <= length)) {
        throw new RangeError("interpolation spans must be ordered within the result");
      }
      previousEnd = end;
    }
  }
}

/**
 * Transient detached inventories and scope from one invocation. Frozen.
 *
 * This worker-owned snapshot becomes durable only through an accepted `ForwardCapture`. It
 * contains neither bound values, outcomes nor node material: learning rebuilds the node data
 * backward reads from the node (`TraceOccurrence.nodeData`).
 */
export class BackwardCapture {
  /** Distinct eligible input names, snapshotted before binding can fail. */
  readonly inputNames: readonly string[];
  /** Distinct local variable names in their declaration order. */
  readonly variableNames: readonly string[];
  /** This invocation's authored path and dynamic positions, if known. */
  readonly scope: ExecutionScope | null;

  /**
   * Validate and retain the declarations.
   *
   * @param options.inputNames Distinct eligible input names; defaults to empty.
   * @param options.variableNames Distinct local variable names; defaults to empty.
   * @param options.scope The invocation's execution scope; defaults to `null`.
   * @throws TypeError for a malformed inventory or scope.
   * @throws RangeError for a repeated name.
   */
  constructor(
    options: {
      inputNames?: readonly string[];
      variableNames?: readonly string[];
      scope?: ExecutionScope | null;
    } = {},
  ) {
    const { inputNames = [], variableNames = [], scope = null } = options;
    validateDeclaredNames(inputNames, "input_names");
    validateDeclaredNames(variableNames, "variable_names");
    // A scope is frozen by its constructor, so it is held without rebuilding, unlike Python.
    if (scope !== null && !isExactly(scope, ExecutionScope)) {
      throw new TypeError("scope must be an ExecutionScope or null");
    }
    this.inputNames = Object.freeze([...inputNames]);
    this.variableNames = Object.freeze([...variableNames]);
    this.scope = scope;
    Object.freeze(this);
  }
}

/**
 * Detached canonical routes from one actual binding operation. Frozen.
 *
 * The bound inputs aren't kept: learning rebuilds them by binding the node again from the values
 * its route read (`TraceOccurrence.boundInputs`). Its record holds no interpolation route's text
 * either; the same binding rebuilds it (`TraceOccurrence.capture`). Nor node data: learning
 * rebuilds it from the node (`TraceOccurrence.nodeData`). An unfinished binding keeps
 * its safe partial routes. Item captures retain explicit indices and actual second-template evaluation flags.
 * Outcomes and visible interactions stay on their accepted record. Scope may be absent for
 * synthetic nodes awaiting instantiation.
 */
export class ForwardCapture {
  /** Record-shaped routes for the captured fields. */
  readonly route: ValueRoute;
  /** Captured keys whose template expressions were evaluated here; a private copy, read-only by contract. */
  readonly evaluatedKeys: ReadonlySet<string>;
  /** Per-item bindings in increasing explicit index order. */
  readonly items: readonly (readonly [number, ForwardCapture])[];
  /** The captured authored location and occurrence, when available. */
  readonly scope: ExecutionScope | null;
  /** Distinct eligible input names retained for this invocation. */
  readonly inputNames: readonly string[];
  /** Distinct local variable names retained for this invocation. */
  readonly variableNames: readonly string[];

  /**
   * Validate the routes and retain frozen copies of the invocation's inventories.
   *
   * @param route Record-shaped routes for the captured fields.
   * @param options.evaluatedKeys Captured keys whose template expressions were evaluated here;
   *   copied, and defaults to empty.
   * @param options.items Per-item index/capture pairs in increasing index order; defaults to
   *   empty. Item captures are held by reference, since they are frozen.
   * @param options.scope The captured execution scope; defaults to `null`.
   * @param options.inputNames Distinct eligible input names; defaults to empty.
   * @param options.variableNames Distinct local variable names; defaults to empty.
   * @throws TypeError when a carrier or container has an invalid type.
   * @throws RangeError when a route, inventory name or item index breaks the capture contract.
   * @throws JSONDecodeError when a route's `valueJson` isn't JSON yian reads.
   */
  constructor(
    route: ValueRoute,
    options: {
      evaluatedKeys?: ReadonlySet<string>;
      items?: readonly (readonly [number, ForwardCapture])[];
      scope?: ExecutionScope | null;
      inputNames?: readonly string[];
      variableNames?: readonly string[];
    } = {},
  ) {
    const {
      evaluatedKeys = new Set<string>(),
      items = [],
      scope = null,
      inputNames = [],
      variableNames = [],
    } = options;
    validateRoute(route);
    if (route.kind !== "record") throw new RangeError("input route must be a record");
    // Copied once, so the checked keys are the stored ones even if a subclass iterates otherwise.
    const keys = evaluatedKeys instanceof Set ? new Set(evaluatedKeys) : null;
    if (keys === null || [...keys].some((key) => typeof key !== "string")) {
      throw new TypeError("evaluated_keys must be a set of strings");
    }
    for (const key of keys) {
      if (!route.names.includes(key)) throw new RangeError("evaluated_keys must name captured routes");
    }
    if (!isList(items)) throw new TypeError("items must be an array");
    const pairs = copyList<readonly [number, ForwardCapture]>(items);
    let previousIndex = -1;
    for (const pair of pairs as unknown[]) {
      if (!isList(pair) || pair.length !== 2) throw new TypeError("item captures must be index/capture pairs");
      const [index, capture] = pair;
      if (typeof index !== "number" || !Number.isSafeInteger(index) || index <= previousIndex) {
        throw new RangeError("item indices must be non-negative and increasing");
      }
      if (!isExactly(capture, ForwardCapture)) throw new TypeError("item capture must be a ForwardCapture");
      previousIndex = index;
    }
    if (scope !== null && !isExactly(scope, ExecutionScope)) {
      throw new TypeError("scope must be an ExecutionScope or null");
    }
    validateDeclaredNames(inputNames, "input_names");
    validateDeclaredNames(variableNames, "variable_names");
    this.route = route;
    this.evaluatedKeys = keys;
    this.items = Object.freeze(pairs.map(([index, capture]) => Object.freeze([index, capture] as const)));
    this.scope = scope;
    this.inputNames = Object.freeze([...inputNames]);
    this.variableNames = Object.freeze([...variableNames]);
    Object.freeze(this);
  }
}

/**
 * One input message transfer and its sender's incoming positions. Frozen.
 *
 * Sender, input and message identify a returned input batch slot; the contribution is its ordinal
 * in the existing pure route transfer. Parents index the sender's ordered received messages. A
 * root seed has no sender or parents. These coordinates keep expression stops and contexts
 * without copying source data.
 */
export class BackwardAttribution {
  /** The accepted occurrence that sent the message, or `null` for a root seed. */
  readonly sender: SourceRef | null;
  /** The sender's input whose batch holds the message, or `null` for a root seed. */
  readonly inputName: string | null;
  /** The message's position in that batch; `0` for a root seed. */
  readonly messageIndex: number;
  /** The message's ordinal in the route transfer, or `null` for a root seed. */
  readonly contributionIndex: number | null;
  /** Distinct increasing positions in the sender's received messages; empty for a root seed. */
  readonly parents: readonly number[];

  /**
   * Validate root or transfer coordinates and keep a frozen copy of `parents`.
   *
   * @throws TypeError when `parents` isn't an array of non-negative integers or `sender` isn't
   *   exactly a `SourceRef`.
   * @throws RangeError for a negative or fractional message index, repeated or misordered parents,
   *   a root seed with sender coordinates, or a transfer without an accepted sender, an input name,
   *   a contribution index or parents.
   */
  constructor(
    sender: SourceRef | null,
    inputName: string | null,
    messageIndex: number,
    contributionIndex: number | null,
    parents: readonly number[],
  ) {
    if (!isIndex(messageIndex)) throw new RangeError("message_index must be a non-negative integer");
    const positions = isList(parents) ? copyList<number>(parents) : null;
    if (positions === null || !positions.every(isIndex)) {
      throw new TypeError("parents must be an array of non-negative integer positions");
    }
    if (positions.some((parent, index) => index > 0 && parent <= positions[index - 1]!)) {
      throw new RangeError("parents must be distinct ordered positions");
    }
    if (sender === null) {
      if (inputName !== null || messageIndex !== 0 || contributionIndex !== null || positions.length > 0) {
        throw new RangeError("root attribution cannot carry sender coordinates");
      }
    } else {
      validateSource(sender);
      if (sender.executionSeq === 0 || sender.path.length > 0) {
        throw new RangeError("attribution sender must identify an accepted occurrence");
      }
      if (typeof inputName !== "string" || !inputName) {
        throw new RangeError("transfer attribution requires an input name");
      }
      if (!isIndex(contributionIndex)) {
        throw new RangeError("transfer attribution requires a non-negative contribution index");
      }
      if (positions.length === 0) throw new RangeError("transfer attribution requires incoming parent positions");
    }
    this.sender = sender;
    this.inputName = inputName;
    this.messageIndex = messageIndex;
    this.contributionIndex = contributionIndex;
    this.parents = Object.freeze(positions);
    Object.freeze(this);
  }
}

/** Whether two attributions have the same sender (by key), input name, indexes and parents. */
export function sameBackwardAttribution(a: BackwardAttribution, b: BackwardAttribution): boolean {
  return (
    sameOptional(a.sender, b.sender, sameSource) &&
    a.inputName === b.inputName &&
    a.messageIndex === b.messageIndex &&
    a.contributionIndex === b.contributionIndex &&
    sameItems(a.parents, b.parents)
  );
}

/**
 * Detached feedback and a result for one accepted forward occurrence. Frozen.
 *
 * The messages are deep-frozen copies made by the constructor, with no copy on read; copy one
 * before you modify it. The attributions and the result are frozen carriers, kept by reference.
 */
export class AttributedBackwardResult {
  /** The accepted forward occurrence, never source zero. */
  readonly source: SourceRef;
  /** Ordered ordinary messages delivered to its callback. */
  readonly feedback: readonly FeedbackValue[];
  /** One source-neutral transfer reference per message. */
  readonly attribution: readonly BackwardAttribution[];
  /** Its validated three-channel result, empty when unreached. */
  readonly result: BackwardResult;
  /** Whether the fold invoked this occurrence's callback. */
  readonly invoked: boolean;

  /**
   * Validate aligned attribution and keep frozen copies of the messages.
   *
   * @throws TypeError for a carrier of another class, a batch that isn't an array, or a message
   *   that isn't ordinary feedback.
   * @throws RangeError for an invalid source, misaligned batches, material on an unreached
   *   occurrence, or an invoked occurrence without messages.
   */
  constructor(
    source: SourceRef,
    feedback: readonly FeedbackValue[],
    attribution: readonly BackwardAttribution[],
    result: BackwardResult,
    invoked: boolean,
  ) {
    validateSource(source);
    if (source.executionSeq === 0 || source.path.length > 0) {
      throw new RangeError("backward result source must identify an accepted occurrence");
    }
    if (!isList(feedback) || !isList(attribution)) throw new TypeError("feedback and attribution must be arrays");
    const messages = copyList(feedback);
    if (messages.length !== attribution.length) throw new RangeError("feedback and attribution must align");
    for (const message of messages) validateFeedback(message, message);
    for (const item of attribution) {
      if (!isExactly(item, BackwardAttribution)) {
        throw new TypeError("attribution requires exact BackwardAttribution values");
      }
    }
    if (!isExactly(result, BackwardResult) || typeof invoked !== "boolean") {
      throw new TypeError("result and invoked must be a BackwardResult and boolean");
    }
    const material =
      Object.keys(result.inputFeedback).length > 0 ||
      result.selfFeedback.length > 0 ||
      Object.keys(result.variableFeedback).length > 0;
    if (!invoked && (messages.length > 0 || material)) {
      throw new RangeError("unreached occurrences cannot carry backward material");
    }
    if (invoked && messages.length === 0) throw new RangeError("invoked occurrences require received feedback");
    this.source = source;
    this.feedback = Object.freeze(messages.map((message) => copyBackwardValue(message, "feedback") as FeedbackValue));
    this.attribution = Object.freeze([...attribution]);
    this.result = result;
    this.invoked = invoked;
    Object.freeze(this);
  }
}

/** Whether two occurrence results have the same source key, feedback, attributions, result and invoked flag. */
export function sameAttributedBackwardResult(a: AttributedBackwardResult, b: AttributedBackwardResult): boolean {
  return (
    sameSource(a.source, b.source) &&
    sameItems(a.feedback, b.feedback, sameFeedback) &&
    sameItems(a.attribution, b.attribution, sameBackwardAttribution) &&
    sameBackwardResult(a.result, b.result) &&
    a.invoked === b.invoked
  );
}

/** The ordered mutations one memory proposed in a backward attempt. Frozen. */
export class MemoryProposal {
  /** The recorded memory key. */
  readonly memory: string;
  /** Mutations with unique ids, in proposal order, kept by reference; an empty batch is valid. */
  readonly mutations: readonly MemoryMutation[];

  /**
   * Validate the key and keep a frozen copy of the mutation batch.
   *
   * @throws TypeError when `memory` isn't a string, `mutations` isn't an array, or an item isn't
   *   exactly a `MemoryMutation`.
   * @throws RangeError for an empty `memory` or two mutations with one id.
   */
  constructor(memory: string, mutations: readonly MemoryMutation[]) {
    if (typeof memory !== "string") throw new TypeError("memory must be a string");
    if (!memory) throw new RangeError("memory must be non-empty");
    this.memory = memory;
    this.mutations = validateMutations(mutations);
    Object.freeze(this);
  }
}

/**
 * An attributed failure from one backward or memory operation. Frozen.
 *
 * Callers branch on `stage` and `code`; `causeType` and `traceback` are for debugging only.
 */
export class BackwardFailure {
  /** The non-empty operation stage, such as backward, learn, or commit. */
  readonly stage: string;
  /** The stable, non-empty semantic failure code, from the backward driver or a memory backend. */
  readonly code: string;
  /** Human-readable detail; it may be empty, and holds no lone surrogate. */
  readonly message: string;
  /** The responsible authored owner and child position. */
  readonly scope: ExecutionScope | null;
  /** The recorded run id, when available. */
  readonly runId: string | null;
  /** The executing node's display id, when applicable. */
  readonly nodeId: string | null;
  /** The positive accepted start sequence identifying a node occurrence with its run id. */
  readonly nodeStartedSeq: number | null;
  /** The responsible memory key, when applicable. */
  readonly memory: string | null;
  /**
   * The underlying source-neutral diagnostic, including a backend retryability hint when one
   * exists. A `Failure`'s origins keep only their keys.
   */
  readonly cause: Failure | MemoryOperationFailure | null;
  /** The thrown error's class name, kept for debugging; it holds no lone surrogate. */
  readonly causeType: string | null;
  /** Opaque debugging traceback text; it holds no lone surrogate. */
  readonly traceback: string | null;

  /**
   * Validate the attribution; a `Failure` cause is copied by `retainFailure`, and a scope or
   * `MemoryOperationFailure` cause is kept by reference. A lone surrogate in `message`,
   * `causeType` or `traceback` becomes U+FFFD, so a thrown error's text can always be logged.
   *
   * @param options Each field defaults to `null`.
   * @throws TypeError when a field has the wrong type, or the scope or cause is of another class.
   * @throws RangeError for an empty stage or code, a start sequence that isn't positive, or a
   *   `Failure` cause whose fields break its contract.
   */
  constructor(
    stage: string,
    code: string,
    message: string,
    options: {
      scope?: ExecutionScope | null;
      runId?: string | null;
      nodeId?: string | null;
      nodeStartedSeq?: number | null;
      memory?: string | null;
      cause?: Failure | MemoryOperationFailure | null;
      causeType?: string | null;
      traceback?: string | null;
    } = {},
  ) {
    const {
      scope = null,
      runId = null,
      nodeId = null,
      nodeStartedSeq = null,
      memory = null,
      cause = null,
      causeType = null,
      traceback = null,
    } = options;
    for (const [name, value] of [["stage", stage], ["code", code]] as const) {
      if (typeof value !== "string") throw new TypeError(`backward failure ${name} must be a string`);
      if (!value) throw new RangeError(`backward failure ${name} must be non-empty`);
    }
    if (typeof message !== "string") throw new TypeError("backward failure message must be a string");
    for (const [name, value] of [
      ["run_id", runId],
      ["node_id", nodeId],
      ["memory", memory],
      ["cause_type", causeType],
      ["traceback", traceback],
    ] as const) {
      if (value !== null && typeof value !== "string") {
        throw new TypeError(`backward failure ${name} must be a string or null`);
      }
    }
    if (nodeStartedSeq !== null) {
      if (!Number.isSafeInteger(nodeStartedSeq)) throw new TypeError("node_started_seq must be an integer or null");
      if (nodeStartedSeq <= 0) throw new RangeError("node_started_seq must be positive");
    }
    if (scope !== null && !isExactly(scope, ExecutionScope)) {
      throw new TypeError("scope must be an ExecutionScope or null");
    }
    if (cause !== null && !isExactly(cause, Failure) && !isExactly(cause, MemoryOperationFailure)) {
      throw new TypeError("cause must be a Failure, MemoryOperationFailure, or null");
    }
    this.stage = stage;
    this.code = code;
    this.message = message.toWellFormed();
    this.scope = scope;
    this.runId = runId;
    this.nodeId = nodeId;
    this.nodeStartedSeq = nodeStartedSeq;
    this.memory = memory;
    this.cause = isExactly(cause, Failure) ? retainFailure(cause) : cause;
    this.causeType = causeType?.toWellFormed() ?? null;
    this.traceback = traceback?.toWellFormed() ?? null;
    Object.freeze(this);
  }
}

/** Whether two causes are of one class and equal by value; a retained `Failure` is a fresh copy. */
function sameCause(a: Failure | MemoryOperationFailure, b: Failure | MemoryOperationFailure): boolean {
  if (isExactly(a, Failure)) return isExactly(b, Failure) && sameFailure(a, b);
  return isExactly(b, MemoryOperationFailure) && sameMemoryOperationFailure(a, b);
}

/** Whether two backward failures have equal fields: scopes, and causes, by value. */
export function sameBackwardFailure(a: BackwardFailure, b: BackwardFailure): boolean {
  return (
    a.stage === b.stage &&
    a.code === b.code &&
    a.message === b.message &&
    sameOptional(a.scope, b.scope, sameExecutionScope) &&
    a.runId === b.runId &&
    a.nodeId === b.nodeId &&
    a.nodeStartedSeq === b.nodeStartedSeq &&
    a.memory === b.memory &&
    sameOptional(a.cause, b.cause, sameCause) &&
    a.causeType === b.causeType &&
    a.traceback === b.traceback
  );
}

/** How a backward attempt ended. */
export type BackwardStatus = "succeeded" | "failed";

/**
 * A backward attempt's disposition, separate from its forward run result. Frozen.
 *
 * The feedback, failure and learning result are frozen carriers, kept by reference.
 */
export class BackwardOutcome {
  /** Whether the backward attempt succeeded. */
  readonly status: BackwardStatus;
  /** The completed flow-level feedback when available, even if a later learning operation failed. */
  readonly feedback: BackwardResult | null;
  /** The failed attempt's diagnostic; `null` on success. */
  readonly failure: BackwardFailure | null;
  /** The memory learning disposition, or `null` when no memory participates. */
  readonly learning: LearningResult | null;

  /**
   * Validate the disposition.
   *
   * @param options Each field defaults to `null`.
   * @throws TypeError when a carrier is of another class.
   * @throws RangeError for an unknown status, a failure on success or none on failure, or a
   *   succeeded outcome whose learning didn't succeed.
   */
  constructor(
    status: BackwardStatus,
    options: {
      feedback?: BackwardResult | null;
      failure?: BackwardFailure | null;
      learning?: LearningResult | null;
    } = {},
  ) {
    const { feedback = null, failure = null, learning = null } = options;
    if (status !== "succeeded" && status !== "failed") {
      throw new RangeError("backward status must be 'succeeded' or 'failed'");
    }
    if (feedback !== null && !isExactly(feedback, BackwardResult)) {
      throw new TypeError("feedback must be a BackwardResult or null");
    }
    if (failure !== null && !isExactly(failure, BackwardFailure)) {
      throw new TypeError("failure must be a BackwardFailure or null");
    }
    if (status === "succeeded" && failure !== null) throw new RangeError("succeeded backward cannot carry a failure");
    if (status === "failed" && failure === null) throw new RangeError("failed backward requires a failure");
    if (learning !== null) {
      if (!isExactly(learning, LearningResult)) throw new TypeError("learning must be a LearningResult or null");
      if (status === "succeeded" && learning.status !== "succeeded") {
        throw new RangeError("succeeded backward requires succeeded learning");
      }
    }
    this.status = status;
    this.feedback = feedback;
    this.failure = failure;
    this.learning = learning;
    Object.freeze(this);
  }
}

/** Whether two outcomes have the same status and equal feedback, failure and learning result. */
export function sameBackwardOutcome(a: BackwardOutcome, b: BackwardOutcome): boolean {
  return (
    a.status === b.status &&
    sameOptional(a.feedback, b.feedback, sameBackwardResult) &&
    sameOptional(a.failure, b.failure, sameBackwardFailure) &&
    sameOptional(a.learning, b.learning, sameLearningResult)
  );
}
