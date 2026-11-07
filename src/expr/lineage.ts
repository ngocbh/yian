/**
 * Recorded expression composition: the routes a traced evaluation records (`evalExpr`,
 * `evalTemplate` and `evalBinding` with `trace: true`).
 *
 * Routes are immutable records from one expression evaluation; a source leaf keeps only the
 * identity of the value it read. Snapshots are yian's codec JSON (`logJson`, read back with
 * `parseJson(text, "codec")`): ints of any size, floats marked `.0`, infinity as
 * `{"$float":"inf"}`, and a record shaped like a tag escaped with `$record`.
 *
 * Knows about: `typesys` (`logJson`, `buildValue`, `TypeCheckError`, `isRecord`, `RootValue`),
 * `builtins` (`kindName`). Never imports: `expressions`, `template`, `nodes`, `runtime`.
 */

import { buildValue, isRecord, TypeCheckError, type Value } from "../typesys/index.ts";
import { logJson } from "../typesys/json.ts";
import type { RootValue } from "../typesys/pool.ts";
import { kindName } from "./builtins.ts";

/** Field names and nonnegative list indices addressing a part of a value. */
export type RoutePath = readonly (string | number)[];

/**
 * One accepted execution's value, or a path inside that value.
 *
 * `executionSeq` is the seq of the producing `pool_commit` (the START inputs get their run-start
 * commit's seq). Values themselves remain in the caller's source store. A `SourceRef` is frozen
 * and compares by identity; match one by its `runId`, `executionSeq` and `path` (`sameSource`).
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

/** Whether two sources have the same run id, sequence and path. */
export function sameSource(a: SourceRef, b: SourceRef): boolean {
  return (
    a.runId === b.runId &&
    a.executionSeq === b.executionSeq &&
    a.path.length === b.path.length &&
    a.path.every((step, index) => step === b.path[index])
  );
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
   * @returns A literal snapshot with no source.
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

/** One reference root read for the evaluator: a `RootValue` and, when tracing, its producer or route. */
export interface RootRead extends RootValue {
  readonly source?: SourceRef | ValueRoute | null;
}

/** The actual evaluated value together with its immutable routing recipe. */
export interface TracedValue {
  readonly value: Value;
  readonly route: ValueRoute;
}

// Diagnostic retention must remain finite even when a later operation discards an arbitrarily
// deep temporary value.
const MAX_SNAPSHOT_DEPTH = 32;

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
  const resultType = kindName(value as Value);
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
