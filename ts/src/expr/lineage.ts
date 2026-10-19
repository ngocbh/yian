/**
 * Recorded expression composition and source-relative feedback transfer.
 *
 * Feedback addresses recorded values without depending on nodes, memory, or the runtime. A list
 * is one positional selection; callers route each independent message separately. Routes are
 * immutable records from the same expression evaluation, with source values supplied by
 * identity when feedback is transferred. Registered function boundaries retain stopped feedback
 * for inspection.
 *
 * Snapshots are yian's codec JSON (`logJson`, read back with `parseJson(text, "codec")`): ints of
 * any size, floats marked `.0`, infinity as `{"$float":"inf"}`, and a record shaped like a tag
 * escaped with `$record`. Its keys are sorted, so a record read back from a snapshot may list its
 * fields in another order than the forward value; feedback selects fields by name.
 *
 * Knows about: `typesys` (`logJson`, `parseJson`, `buildValue`, `TypeCheckError`, `isRecord`),
 * `diagnostics` (`reprText`), `builtins` (`kindName`). Never imports: `expressions`,
 * `template`, `nodes`, `runtime`.
 */

import { reprText } from "../diagnostics.ts";
import { buildValue, isRecord, TypeCheckError, type Value } from "../typesys/index.ts";
import { logJson, parseJson } from "../typesys/json.ts";
import { kindName } from "./builtins.ts";

/** One feedback message: a string, absent feedback, a partial field selection, or a positional list. */
export type FeedbackValue = string | null | FeedbackValue[] | { [name: string]: FeedbackValue };

/** Field names and nonnegative list indices addressing a part of a value. */
export type RoutePath = readonly (string | number)[];

/**
 * One accepted execution's value, or a path inside that value.
 *
 * Sequence zero addresses run inputs; other sequences identify the accepted node-start record.
 * Values themselves remain in the caller's source store. A `SourceRef` is frozen and compares by
 * identity; match one by its `runId`, `executionSeq` and `path`, and key maps and sets by
 * `sourceKey(ref)`.
 */
export class SourceRef {
  readonly runId: string;
  readonly executionSeq: number;
  readonly path: RoutePath;

  /** Address `path` inside execution `executionSeq` of run `runId`; `path` is copied. */
  constructor(runId: string, executionSeq: number, path: RoutePath = []) {
    this.runId = runId;
    this.executionSeq = executionSeq;
    this.path = Object.freeze([...path]);
    Object.freeze(this);
  }
}

/** A text key equal for two `SourceRef`s exactly when their run id, sequence and path are equal. */
export function sourceKey(ref: SourceRef): string {
  return JSON.stringify([ref.runId, ref.executionSeq, ...ref.path]);
}

/** The kinds of recorded composition a `ValueRoute` holds. */
export type RouteKind =
  | "literal"
  | "source"
  | "selected"
  | "list"
  | "record"
  | "concat"
  | "scalar"
  | "interpolation"
  | "opaque"
  | "project";

/** The fields of a `ValueRoute` besides its kind; each one left out takes its default. */
export interface RouteFields {
  source?: SourceRef | null;
  valueJson?: string | null;
  children?: readonly ValueRoute[];
  expression?: string;
  names?: readonly string[];
  sizes?: readonly number[];
  spans?: readonly (readonly [number, number])[];
  path?: RoutePath;
  resultType?: string | null;
  snapshot?: ValueRoute | null;
}

/**
 * Immutable expression composition recorded alongside its forward value.
 *
 * Source leaves retain identity only. Literal and opaque results use codec JSON text to isolate
 * their snapshots from later mutation without mutable route fields. A temporary JSON can't hold
 * (a file) retains an unavailable-result type marker instead of the live value. Mixed lists and
 * records keep available shape in `snapshot`, using these same literal/list/record forms around
 * unavailable leaves. Ordinary JSON snapshots stay complete; a partial capture has bounded
 * nesting.
 *
 * A route and its arrays are frozen.
 */
export class ValueRoute {
  readonly kind: RouteKind;
  readonly source: SourceRef | null;
  readonly valueJson: string | null;
  readonly children: readonly ValueRoute[];
  readonly expression: string;
  readonly names: readonly string[];
  readonly sizes: readonly number[];
  readonly spans: readonly (readonly [number, number])[];
  readonly path: RoutePath;
  readonly resultType: string | null;
  readonly snapshot: ValueRoute | null;

  /** Build a route of `kind` from `fields`; arrays are copied. */
  constructor(kind: RouteKind, fields: RouteFields = {}) {
    this.kind = kind;
    this.source = fields.source ?? null;
    this.valueJson = fields.valueJson ?? null;
    this.children = Object.freeze([...(fields.children ?? [])]);
    this.expression = fields.expression ?? "";
    this.names = Object.freeze([...(fields.names ?? [])]);
    this.sizes = Object.freeze([...(fields.sizes ?? [])]);
    this.spans = Object.freeze((fields.spans ?? []).map(([start, end]) => Object.freeze([start, end] as const)));
    this.path = Object.freeze([...(fields.path ?? [])]);
    this.resultType = fields.resultType ?? null;
    this.snapshot = fields.snapshot ?? null;
    Object.freeze(this);
  }

  /**
   * Capture a literal snapshot or an unavailable-result type marker.
   *
   * @param value A value evaluated without a producer. A leaf JSON can't hold retains its kind
   *   name; ordinary containing lists and records preserve available structure.
   * @returns A literal snapshot with no upstream feedback route.
   */
  static literal(value: unknown): ValueRoute {
    // Only the root gets an unbounded snapshot; a partial capture is depth-bounded.
    const encoded = snapshotJson(value);
    if (encoded !== null) return new ValueRoute("literal", { valueJson: encoded, expression: encoded });
    return captureLiteral(value, 0);
  }

  /**
   * Retain the named child routes of a record assembled by a binder.
   *
   * @param fields Unique field names and their routes in construction order.
   * @returns The record constructor and its named child routes.
   */
  static record(fields: Iterable<readonly [string, ValueRoute]>): ValueRoute {
    const entries = [...fields];
    return new ValueRoute("record", {
      children: entries.map(([, route]) => route),
      names: entries.map(([name]) => name),
      expression: `{${entries.map(([name, route]) => `${name}: ${route.expression}`).join(", ")}}`,
    });
  }

  /**
   * Address an existing field or item of this composition's output.
   *
   * @param path Valid field names and nonnegative list indices, relative to this route. Empty
   *   selects the root.
   * @returns This route for an empty path; a source route extended by `path`; else a projection
   *   that retains this composition.
   */
  project(path: RoutePath): ValueRoute {
    if (path.length === 0) return this;
    const suffix = path.map((part) => (typeof part === "string" ? `.${part}` : `[${part}]`)).join("");
    if (this.kind === "source") {
      const source = this.source!;
      return new ValueRoute("source", {
        source: new SourceRef(source.runId, source.executionSeq, [...source.path, ...path]),
        expression: this.expression + suffix,
      });
    }
    return new ValueRoute("project", { children: [this], path, expression: this.expression + suffix });
  }

  /** Return a new route with this one's fields and `changes`. */
  with(changes: RouteFields & { kind?: RouteKind }): ValueRoute {
    const { kind = this.kind, ...fields } = changes;
    return new ValueRoute(kind, {
      source: this.source,
      valueJson: this.valueJson,
      children: this.children,
      expression: this.expression,
      names: this.names,
      sizes: this.sizes,
      spans: this.spans,
      path: this.path,
      resultType: this.resultType,
      snapshot: this.snapshot,
      ...fields,
    });
  }
}

/**
 * One read with actual presence and a producer or prior composition route.
 *
 * In traced evaluation this also represents the local MAP item scope. A present null keeps its
 * source even though forward's missing/null behavior still lets default and coalesce fall
 * through. Frozen.
 */
export class ResolvedValue {
  readonly value: Value;
  readonly present: boolean;
  readonly source: SourceRef | ValueRoute | null;

  /** Record `value`, whether it was `present`, and its producer or route. */
  constructor(value: Value, present: boolean, source: SourceRef | ValueRoute | null = null) {
    this.value = value;
    this.present = present;
    this.source = source;
    Object.freeze(this);
  }
}

/** The actual evaluated value together with its immutable routing recipe. */
export interface TracedValue {
  readonly value: Value;
  readonly route: ValueRoute;
}

/** The fields of a `RouteContext`; each optional one left out takes its default. */
export interface RouteContextFields {
  expression: string;
  resultJson: string | null;
  sources: readonly SourceRef[];
  role?: string;
  operand?: number | string | RoutePath | null;
  positions?: readonly (readonly number[])[];
  span?: readonly [number, number] | null;
  resultType?: string | null;
  snapshot?: ValueRoute | null;
}

/**
 * Interpretation data accompanying unchanged feedback text.
 *
 * `positions` maps each content item to its output positions. `span` is a half-open range of list
 * positions for concat, or code points for interpolation. A projection's `operand` is its selected
 * path; other operands are field names or input positions. Source contexts reference their result
 * through `sources` instead of copying accepted producer values. `sources` preserves evaluated
 * source occurrences, including duplicates; source values remain in the caller's accepted
 * snapshot store. Frozen.
 */
export class RouteContext {
  readonly expression: string;
  readonly resultJson: string | null;
  readonly sources: readonly SourceRef[];
  readonly role: string;
  readonly operand: number | string | RoutePath | null;
  readonly positions: readonly (readonly number[])[];
  readonly span: readonly [number, number] | null;
  readonly resultType: string | null;
  readonly snapshot: ValueRoute | null;

  /** Build a context from `fields`; role defaults to `operand`, the rest to null or empty. */
  constructor(fields: RouteContextFields) {
    const operand = fields.operand ?? null;
    const span = fields.span ?? null;
    this.expression = fields.expression;
    this.resultJson = fields.resultJson;
    this.sources = Object.freeze([...fields.sources]);
    this.role = fields.role ?? "operand";
    this.operand = Array.isArray(operand) ? Object.freeze([...operand]) : operand;
    this.positions = Object.freeze((fields.positions ?? []).map((items) => Object.freeze([...items])));
    this.span = span === null ? null : Object.freeze([span[0], span[1]] as const);
    this.resultType = fields.resultType ?? null;
    this.snapshot = fields.snapshot ?? null;
    Object.freeze(this);
  }

  /**
   * Return an isolated result, or null for a source or unavailable result.
   *
   * A recorded JSON null has `resultJson === "null"` and no `resultType`; an unavailable temporary
   * has no JSON snapshot and retains its type name. Partial container structure remains in
   * `snapshot`. Source contexts keep the exact accepted value reference in `sources` instead of a
   * JSON copy.
   */
  get result(): unknown {
    return this.resultJson !== null ? parseJson(this.resultJson, "codec") : null;
  }
}

/** One independent feedback value lifted to an execution's root output. */
export interface SourceFeedback {
  readonly source: SourceRef;
  readonly feedback: FeedbackValue;
  readonly context: readonly RouteContext[];
}

/** Feedback retained at an opaque expression instead of sent to its inputs. */
export interface StoppedFeedback {
  readonly feedback: FeedbackValue;
  readonly context: readonly RouteContext[];
}

/** Ordered source contributions and opaque stops from one transfer. */
export interface RoutedFeedback {
  readonly contributions: readonly SourceFeedback[];
  readonly stopped: readonly StoppedFeedback[];
}

/** Step into field or item `key` of `value`, reading own keys only. */
function stepInto(value: unknown, key: string | number): unknown {
  if (Array.isArray(value)) {
    if (typeof key === "string") throw new TypeError(`a list has no field ${reprText(key)}`);
    if (!(Number.isInteger(key) && key >= 0 && key < value.length)) {
      throw new RangeError(`list index ${key} is out of range`);
    }
    return value[key];
  }
  if (isRecord(value)) {
    if (typeof key === "number") throw new RangeError(`a record has no item [${key}]`);
    if (!Object.hasOwn(value, key)) throw new RangeError(`record has no field ${reprText(key)}`);
    return value[key];
  }
  throw new TypeError(`${typeName(value)} has no part ${typeof key === "string" ? reprText(key) : `[${key}]`}`);
}

function atPath(value: unknown, path: RoutePath): unknown {
  for (const step of path) value = stepInto(value, step);
  return value;
}

function liftFeedback(feedback: FeedbackValue, value: unknown, path: RoutePath): FeedbackValue {
  if (path.length === 0) return feedback;
  const head = path[0]!;
  const rest = path.slice(1);
  const lifted = liftFeedback(feedback, stepInto(value, head), rest);
  if (typeof head === "string") return { [head]: lifted };
  // `stepInto` accepted a numeric step, so `value` is a list.
  const result: FeedbackValue[] = Array.from({ length: (value as unknown[]).length }, () => null);
  result[head] = lifted;
  return result;
}

const UNAVAILABLE: unique symbol = Symbol("unavailable");
// Diagnostic retention must remain finite even when a later operation discards an arbitrarily
// deep temporary value.
const MAX_SNAPSHOT_DEPTH = 32;

/** Name `value`'s kind for an unavailable marker: `unavailable` for `UNAVAILABLE`, else its yian kind. */
function typeName(value: unknown): string {
  return value === UNAVAILABLE ? "unavailable" : kindName(value as Value);
}

/**
 * Return a snapshot's codec JSON text, or `null` when the value has no JSON snapshot: it holds a
 * file, an unavailable leaf, or an int the runtime can't write.
 */
function snapshotJson(value: unknown): string | null {
  try {
    // `buildValue` reads `-0` as `0` at any depth, which `logJson` would refuse.
    return logJson(buildValue(value).value);
  } catch (error) {
    if (error instanceof TypeError || error instanceof TypeCheckError) return null;
    throw error;
  }
}

/** Capture bounded list/record structure around leaves JSON can't hold. */
function captureLiteral(value: unknown, depth: number): ValueRoute {
  const resultType = typeName(value);
  const unavailable = new ValueRoute("literal", { resultType, expression: `<${resultType}>` });
  if (!Array.isArray(value) && !isRecord(value)) {
    const encoded = snapshotJson(value);
    if (encoded === null) return unavailable;
    return new ValueRoute("literal", { valueJson: encoded, expression: encoded });
  }
  if (depth >= MAX_SNAPSHOT_DEPTH) return unavailable;
  const snapshot = Array.isArray(value)
    ? new ValueRoute("list", { children: value.map((item) => captureLiteral(item, depth + 1)) })
    : ValueRoute.record(Object.keys(value).map((name) => [name, captureLiteral(value[name], depth + 1)] as const));
  const encoded = snapshot.children.every((child) => child.valueJson !== null) ? snapshotJson(value) : null;
  if (encoded === null) return unavailable.with({ snapshot });
  return new ValueRoute("literal", { valueJson: encoded, expression: encoded });
}

/** Select available snapshot structure without disguising ordinary path faults. */
function recordedAtPath(value: unknown, path: RoutePath): unknown {
  for (const step of path) {
    if (value === UNAVAILABLE) return UNAVAILABLE;
    value = stepInto(value, step);
  }
  return value;
}

/**
 * Return the value `route` recorded, reading source roots through `lookup`.
 *
 * May return a module-private unavailable marker, at any depth, for a value whose snapshot was
 * unavailable.
 *
 * @throws TypeError when a concat operand isn't a list, or a path steps into a value that isn't a
 *   list or record (or by field into a list).
 * @throws RangeError for an unknown route kind, or a path naming a missing field or index.
 */
export function routeValue(route: ValueRoute, lookup: (source: SourceRef) => unknown): unknown {
  switch (route.kind) {
    case "source": {
      const source = route.source!;
      return atPath(lookup(new SourceRef(source.runId, source.executionSeq)), source.path);
    }
    case "literal":
    case "scalar":
    case "interpolation":
    case "opaque":
      if (route.snapshot !== null) return routeValue(route.snapshot, lookup);
      return route.valueJson !== null ? parseJson(route.valueJson, "codec") : UNAVAILABLE;
    case "selected":
      return routeValue(route.children[0]!, lookup);
    case "project":
      return recordedAtPath(routeValue(route.children[0]!, lookup), route.path);
    case "list":
      return route.children.map((child) => routeValue(child, lookup));
    case "record":
      return Object.fromEntries(route.names.map((name, index) => [name, routeValue(route.children[index]!, lookup)]));
    case "concat": {
      const values: unknown[] = [];
      let unavailable = false;
      // Inspect known siblings too: unavailable snapshots must not mask real faults.
      for (const child of route.children) {
        const value = routeValue(child, lookup);
        if (value === UNAVAILABLE) unavailable = true;
        else if (!Array.isArray(value)) throw new TypeError("concat operand is not a list");
        else for (const item of value) values.push(item);
      }
      return unavailable ? UNAVAILABLE : values;
    }
  }
  throw new RangeError(`unknown route kind ${reprText(String(route.kind))}`);
}

function sourceRefs(route: ValueRoute): SourceRef[] {
  if (route.kind === "source") return [route.source!];
  return route.children.flatMap(sourceRefs);
}

/**
 * Transfer feedback using recorded composition and accepted source values.
 *
 * @param route The route retained during forward evaluation.
 * @param feedback One message selecting this route's result.
 * @param lookup Returns an accepted root value for an identity with an empty path. It must use
 *   the original snapshot.
 * @param options.recordedValue The accepted canonical result, when available (`undefined` means
 *   absent; `null` is a recorded null). It supplies complete feedback shape when a supported
 *   forward value had an unavailable temporary snapshot. Omission uses route snapshots. This
 *   value supplies no producer mappings and is propagated only through recorded structural
 *   selections.
 * @returns Ordered, independent source-shaped contributions and stopped feedback with
 *   expression/result/source interpretation data; it, its arrays and its entries are frozen.
 * @throws TypeError for unsupported feedback data.
 * @throws RangeError for feedback that does not match the result shape.
 */
export function routeFeedback(
  route: ValueRoute,
  feedback: FeedbackValue,
  lookup: (source: SourceRef) => unknown,
  options: { recordedValue?: unknown } = {},
): RoutedFeedback {
  const recordedValue = options.recordedValue === undefined ? UNAVAILABLE : options.recordedValue;
  if (recordedValue !== UNAVAILABLE) validateFeedback(feedback, recordedValue);
  const contributions: SourceFeedback[] = [];
  const stopped: StoppedFeedback[] = [];

  const visit = (
    current: ValueRoute,
    message: FeedbackValue,
    context: readonly RouteContext[] = [],
    knownValue: unknown = UNAVAILABLE,
    selection: RoutePath = [],
  ): void => {
    if (message === null) return;
    const value = routeValue(current, lookup);
    const selectedValue = recordedAtPath(value, selection);
    validateFeedback(message, knownValue === UNAVAILABLE ? selectedValue : knownValue);
    if (!hasFeedback(message)) return;
    if (current.kind === "source") {
      const source = current.source!;
      const root = new SourceRef(source.runId, source.executionSeq);
      const original = lookup(root);
      const path = [...source.path, ...selection];
      validateFeedback(message, atPath(original, path));
      const lifted = liftFeedback(message, original, path);
      validateFeedback(lifted, original);
      const selectedSource = new SourceRef(source.runId, source.executionSeq, path);
      const detail = new RouteContext({
        expression: current.expression,
        resultJson: null,
        sources: [selectedSource],
        role: path.length > 0 ? "projection" : "source",
        operand: path.length > 0 ? path : null,
      });
      const sourceContext = Object.freeze([...context, detail]);
      contributions.push(Object.freeze({ source: root, feedback: lifted, context: sourceContext }));
      return;
    }
    if (current.kind === "selected") {
      visit(current.children[0]!, message, context, knownValue, selection);
      return;
    }
    if (current.kind === "literal") return;

    const snapshot =
      current.kind === "scalar" || current.kind === "interpolation" || current.kind === "opaque"
        ? current
        : ValueRoute.literal(value);
    const sources = sourceRefs(current);

    const withContext = (
      role: string,
      operand: number | string | RoutePath | null = null,
      positions: readonly (readonly number[])[] = [],
      span: readonly [number, number] | null = null,
    ): readonly RouteContext[] =>
      Object.freeze([
        ...context,
        new RouteContext({
          expression: current.expression,
          resultJson: snapshot.valueJson,
          sources,
          role,
          operand,
          positions,
          span,
          resultType: snapshot.resultType,
          snapshot: snapshot.snapshot,
        }),
      ]);

    if (current.kind === "project") {
      visit(current.children[0]!, message, withContext("projection", current.path), knownValue, [
        ...current.path,
        ...selection,
      ]);
    } else if (current.kind === "list" || current.kind === "record") {
      current.children.forEach((child, index) => {
        const operand = current.kind === "list" ? index : current.names[index]!;
        if (selection.length > 0 && operand !== selection[0]) return;
        let selected: FeedbackValue;
        if (selection.length > 0 || typeof message === "string") selected = message;
        else if (typeof operand === "number") selected = (message as FeedbackValue[])[index]!;
        else {
          const fields = message as { [name: string]: FeedbackValue };
          selected = Object.hasOwn(fields, operand) ? fields[operand]! : null;
        }
        const childValue =
          selection.length > 0 || knownValue === UNAVAILABLE ? knownValue : stepInto(knownValue, operand);
        const positions = current.kind === "list" ? [[index]] : [];
        visit(
          child,
          selected,
          withContext(current.kind === "list" ? "item" : "field", operand, positions),
          childValue,
          selection.slice(1),
        );
      });
    } else if (current.kind === "concat") {
      const total = current.sizes.reduce((sum, size) => sum + size, 0);
      const first = selection[0];
      if (first !== undefined && !(typeof first === "number" && Number.isInteger(first) && 0 <= first && first < total)) {
        throw new RangeError("concat projection index is outside its recorded sizes");
      }
      let offset = 0;
      current.children.forEach((child, index) => {
        const size = current.sizes[index];
        if (size === undefined) return;
        if (first !== undefined && !(offset <= first && first < offset + size)) {
          offset += size;
          return;
        }
        const childValue =
          selection.length > 0 || knownValue === UNAVAILABLE
            ? knownValue
            : (knownValue as unknown[]).slice(offset, offset + size);
        visit(
          child,
          selection.length > 0 || typeof message === "string"
            ? message
            : (message as FeedbackValue[]).slice(offset, offset + size),
          withContext(
            "concat",
            index,
            Array.from({ length: size }, (_, item) => [offset + item]),
            [offset, offset + size],
          ),
          childValue,
          first !== undefined ? [first - offset, ...selection.slice(1)] : [],
        );
        offset += size;
      });
    } else if (current.kind === "scalar" || current.kind === "interpolation") {
      if (selection.length > 0) message = liftFeedback(message, value, selection);
      current.children.forEach((child, index) => {
        const span = current.kind === "interpolation" ? current.spans[index]! : null;
        visit(child, message, withContext(span !== null ? "interpolation" : "operand", index, [], span));
      });
    } else if (current.kind === "opaque") {
      if (selection.length > 0 && recordedAtPath(value, selection.slice(0, -1)) !== UNAVAILABLE) {
        message = liftFeedback(message, value, selection);
      }
      stopped.push(Object.freeze({ feedback: message, context: withContext("opaque") }));
    }
  };

  visit(route, feedback, [], recordedValue);
  return Object.freeze({ contributions: Object.freeze(contributions), stopped: Object.freeze(stopped) });
}

/**
 * Require feedback to select the exact shape of a recorded value.
 *
 * @param feedback A string, absent feedback, a partial field selection, or a positional list with
 *   the recorded list's length.
 * @param value The recorded JSON-like value addressed by the feedback.
 * @param path Address of this value, used to locate validation failures.
 * @throws TypeError for an unsupported feedback value (`<location> must be a string, object, list,
 *   or None`).
 * @throws RangeError for an unknown field or a mismatched container shape: `<location> requires a
 *   recorded object`, `<location> selects unknown field 'x'`, `<location> requires a recorded
 *   list`, `<location> must have exactly N items; got M`.
 */
export function validateFeedback(
  feedback: unknown,
  value: unknown,
  path: RoutePath = [],
): asserts feedback is FeedbackValue {
  if (feedback === null || typeof feedback === "string") return;

  const location = `feedback${path.map((part) => `[${typeof part === "string" ? JSON.stringify(part) : part}]`).join("")}`;
  if (isRecord(feedback)) {
    if (!isRecord(value)) throw new RangeError(`${location} requires a recorded object`);
    for (const name of Object.keys(feedback)) {
      if (!Object.hasOwn(value, name)) throw new RangeError(`${location} selects unknown field ${reprText(name)}`);
      validateFeedback(feedback[name], value[name], [...path, name]);
    }
    return;
  }

  if (Array.isArray(feedback)) {
    if (!Array.isArray(value)) throw new RangeError(`${location} requires a recorded list`);
    if (feedback.length !== value.length) {
      throw new RangeError(`${location} must have exactly ${value.length} items; got ${feedback.length}`);
    }
    feedback.forEach((message: unknown, index) => validateFeedback(message, value[index], [...path, index]));
    return;
  }

  throw new TypeError(`${location} must be a string, object, list, or None`);
}

/**
 * Return whether validated feedback contains any supplied string, including the empty string.
 *
 * @param feedback One ordinary feedback value. Its container shape must already have been
 *   validated against the recorded value.
 * @throws TypeError when a visited value has an unsupported type.
 */
export function hasFeedback(feedback: FeedbackValue): boolean {
  if (feedback === null) return false;
  if (typeof feedback === "string") return true;
  if (isRecord(feedback)) return Object.values(feedback).some((message) => hasFeedback(message as FeedbackValue));
  if (Array.isArray(feedback)) return feedback.some((message) => hasFeedback(message));
  throw new TypeError("feedback must be a string, object, list, or None");
}
