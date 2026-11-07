/**
 * Immutable forward captures and memory proposals shared by runtime consumers.
 *
 * These values depend only on lower node, expression, memory, and diagnostic contracts. They
 * never import the runtime log, state, trace, driver, or public run boundary.
 *
 * Knows about:   expr/lineage, memory/contracts, nodes/base, nodes/contracts, typesys/json,
 *                typesys/values.
 * Never imports: runtime log, state, trace, engine or run, compile, compose, or the CLI.
 */

import { type RouteKind, SourceRef, ValueRoute } from "../expr/lineage.ts";
import { type MemoryMutation, validateMutations } from "../memory/contracts.ts";
import { validateDeclaredNames } from "../nodes/base.ts";
import { ExecutionScope, isExactly } from "../nodes/contracts.ts";
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
 * contains neither bound values nor outcomes.
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
    // A scope is frozen by its constructor, so it is held without rebuilding.
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
 * either; the same binding rebuilds it (`TraceOccurrence.capture`). An unfinished binding keeps
 * its safe partial routes and says so (`bound`), since a failure's code doesn't tell whether its
 * bound inputs are usable. Item captures retain explicit indices and actual second-template evaluation
 * flags. Outcomes and visible interactions stay on their accepted record. Scope may be absent for
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
   * Whether the bound inputs are usable: `false` when this binding, or any of a MAP's item
   * bindings, failed; an item's flag follows its MAP's.
   */
  readonly bound: boolean;

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
   * @param options.bound Whether the bound inputs are usable (see `bound`); defaults to `true`.
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
      bound?: boolean;
    } = {},
  ) {
    const {
      evaluatedKeys = new Set<string>(),
      items = [],
      scope = null,
      inputNames = [],
      variableNames = [],
      bound = true,
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
    if (typeof bound !== "boolean") throw new TypeError("bound must be a boolean");
    this.route = route;
    this.evaluatedKeys = keys;
    this.items = Object.freeze(pairs.map(([index, capture]) => Object.freeze([index, capture] as const)));
    this.scope = scope;
    this.inputNames = Object.freeze([...inputNames]);
    this.variableNames = Object.freeze([...variableNames]);
    this.bound = bound;
    Object.freeze(this);
  }
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
