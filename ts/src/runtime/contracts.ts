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

import { type RouteKind, SourceRef, ValueRoute } from "../expr/lineage.ts";
import { isExactly } from "../nodes/contracts.ts";
import { type JsonValue, parseJson } from "../typesys/json.ts";

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
        throw new RangeError("partial snapshot cannot contain source routes");
      }
      for (const child of current.children) pending.push(child);
    }
  }
  if (kind === "interpolation") {
    if (typeof result !== "string" || route.children.length === 0 || route.spans.length !== route.children.length) {
      throw new RangeError("interpolation requires a string result and one span per child");
    }
    // Spans count code points, as interpolation contexts do.
    const length = [...result].length;
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
