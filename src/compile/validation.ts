/**
 * Representation-neutral graph + reference leaf checkers.
 *
 * These operate on bare `Type`/edge/path-set inputs, not a built flow, so the Compose loader
 * reuses them directly:
 *
 * - `topologicalOrder`: Kahn's order over a node list and its edges.
 * - `rejectCycles`: a cycle check over a bare edge list + node-id set, through that order.
 * - `classifyPath`: head-resolution for one dotted `${...}` ref. Field reads are checked by the
 *   shared output-contract walk in `nodes/contracts.ts`.
 * - `typesCompatible`: whether every value of a source type fits a sink type (`typeRelation` fits).
 * - `typeRelation`: how an inferred type relates to a declared one: it fits, is narrower (only
 *   some values fit), or is disjoint (no value fits).
 *
 * `FlowValidationError` is the shared structural-error type the compose loader catches and
 * rethrows as a located `LoadError`.
 *
 * Knows about: compile/model (the boundary ids), diagnostics and typesys.
 */

import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { messagesKey, NEW_MESSAGES } from "../typesys/pool.ts";
import {
  type Bound,
  buildValueAs,
  hasFile,
  isListKind,
  listElementKind,
  requiresFile,
  Type,
  TypeCheckError,
  typeEquals,
  ValueKind,
  writeAsText,
} from "../typesys/values.ts";
import { END_ID, START_ID } from "./model.ts";

/**
 * The only valid `${system.X}` ambients, each with its scalar kind: `today` (date) and `now`
 * (datetime) read the host clock, `run_id` (str) is the run's id. The run seeds each once at its
 * boundary and a child inherits them.
 */
export const SYSTEM_AMBIENTS: ReadonlyMap<string, ValueKind> = new Map([
  ["today", ValueKind.DATE],
  ["now", ValueKind.DATETIME],
  ["run_id", ValueKind.STRING],
]);

/**
 * A flow graph is structurally invalid and cannot be compiled or run.
 *
 * Thrown by the graph/reference checkers and caught by the compose loader, which rethrows it as
 * a located `LoadError`.
 */
export class FlowValidationError extends RangeError {
  override name = "FlowValidationError";
  /**
   * All accumulated located problems. The reference-wiring pass collects every problem it
   * finds; fail-fast structural checks carry the single message.
   */
  readonly errors: readonly string[];

  /** Build the error; `errors` defaults to `[message]`. */
  constructor(message: string, errors?: readonly string[]) {
    super(message);
    this.errors = errors ?? [message];
  }
}

/**
 * Order `nodes` so each comes after every node with an edge into it (Kahn's algorithm).
 *
 * Ready nodes are taken first in, first out, starting in `nodes` order; a node waits for every
 * edge into it, a repeated edge included.
 *
 * @param nodes The nodes; every edge must name two of them.
 * @param edges The edges between them.
 * @returns `order`, the nodes in topological order, and `rest`, the nodes on or behind a cycle
 *   (in `nodes` order), which `order` leaves out.
 */
export function topologicalOrder(
  nodes: Iterable<string>,
  edges: Iterable<{ readonly from: string; readonly to: string }>,
): { order: string[]; rest: string[] } {
  const all = [...nodes];
  const indegree = new Map<string, number>(all.map((node) => [node, 0]));
  const successors = new Map<string, string[]>(all.map((node) => [node, []]));
  for (const edge of edges) {
    indegree.set(edge.to, indegree.get(edge.to)! + 1);
    successors.get(edge.from)!.push(edge.to);
  }
  const ready = all.filter((node) => indegree.get(node) === 0);
  // `ready` is a queue: nodes are taken from the front as they are appended at the back.
  for (let next = 0; next < ready.length; next++) {
    for (const successor of successors.get(ready[next]!)!) {
      const count = indegree.get(successor)! - 1;
      indegree.set(successor, count);
      if (count === 0) ready.push(successor);
    }
  }
  return { order: ready, rest: all.filter((node) => indegree.get(node)! > 0) };
}

/**
 * Throw if the real-node graph has a directed cycle (Kahn's algorithm).
 *
 * Edges touching `__start__`/`__end__` can't be part of a cycle (the sentinels have no
 * incoming/outgoing respectively), so only real-node edges count.
 *
 * Takes the bare `edges` (any object with `from`/`to`) + the node-id set, so the loader can
 * reuse it on its synthesized `Edge`s.
 *
 * @throws FlowValidationError naming the nodes left on or behind a cycle.
 */
export function rejectCycles(
  edges: Iterable<{ readonly from: string; readonly to: string }>,
  nodeIds: ReadonlySet<string>,
): void {
  const real: { readonly from: string; readonly to: string }[] = [];
  for (const edge of edges) {
    if (edge.from === START_ID || edge.from === END_ID) continue;
    if (edge.to === START_ID || edge.to === END_ID) continue;
    // An edge naming a node outside `nodeIds` is a caller defect.
    for (const end of [edge.from, edge.to]) {
      if (!nodeIds.has(end)) throw new Error(`edge names unknown node ${reprText(end)}`);
    }
    real.push(edge);
  }
  const { rest } = topologicalOrder(nodeIds, real);
  if (rest.length > 0) {
    const stuck = rest.sort(compareCodeUnits);
    throw new FlowValidationError(
      `flow has a cycle involving ${writeAsText(stuck)}; flows must be acyclic`,
    );
  }
}

/** Return whether low end `a` is at most low end `b`; `null` is −∞. */
function lowAtMost(a: Bound, b: Bound): boolean {
  if (a === null) return true;
  // `<=` compares a bigint with a number exactly.
  return b !== null && a <= b;
}

/** Return whether high end `a` is at most high end `b`; `null` is +∞. */
function highAtMost(a: Bound, b: Bound): boolean {
  if (b === null) return true;
  return a !== null && a <= b;
}

/** Return whether `source`'s numeric interval lies within `sink`'s; absent bounds are open. */
function intervalWithin(source: Type, sink: Type): boolean {
  const [low, high] = source.bounds ?? [null, null];
  const [sinkLow, sinkHigh] = sink.bounds ?? [null, null];
  return lowAtMost(sinkLow, low) && highAtMost(high, sinkHigh);
}

/**
 * Return whether every value of a source type fits a sink slot: `typeRelation(source, sink)` is
 * `"fits"`, so records are closed, as the write boundary checks them.
 *
 * @param source - Type guaranteed by the producer.
 * @param sink - Type required by the consumer slot.
 * @returns Whether every value admitted by `source` is admitted by `sink`.
 */
export function typesCompatible(source: Type, sink: Type): boolean {
  return typeRelation(source, sink) === "fits";
}

/** How an inferred type relates to a declared one (see `typeRelation`). */
export type TypeRelation = "fits" | "narrower" | "disjoint";

const RELATION_RANK: Readonly<Record<TypeRelation, number>> = {
  disjoint: 0,
  narrower: 1,
  fits: 2,
};

// Date and datetime values are stored as ISO strings, so each fits a `str` slot.
const STRING_STORED: ReadonlySet<ValueKind> = new Set([ValueKind.DATE, ValueKind.DATETIME]);

/**
 * Classify how an inferred value type relates to a declared type.
 *
 * - `"fits"`: every value of `inferred` is a value of `declared`; no runtime check is needed.
 * - `"narrower"`: some values of `inferred` are values of `declared` and some may not be; the
 *   binding is accepted and checked at runtime.
 * - `"disjoint"`: no non-null value of `inferred` is a value of `declared`; the binding can
 *   never succeed. Sharing only `None` does not count as overlap, so `Optional(str)` against
 *   `Optional(int)` is disjoint. This null-only rule applies to the whole value: a record whose
 *   differing field both sides may omit (or may both hold `None`) still shares non-null record
 *   values, so it is narrower, not disjoint.
 *
 * Records are treated as closed, as the write-boundary check treats them (unknown fields are
 * rejected), so a record fits only when its fields are a subset of the declared fields at every
 * level. `date` and `datetime` fit `str`, since they are stored as strings. A `Literal` meets `date` or `datetime` only through tags that are valid ISO strings of that
 * kind. Unions are compared member-wise: each inferred member takes its best relation against
 * the declared members, so a union member over the union is narrower and a record matching no
 * member is disjoint. A type without a `file` in it fits `Any` (narrower when nullable); one
 * whose every non-null value holds a `file` is disjoint from it, and one that only may hold a
 * `file` is narrower. `Any` is narrower than every other type except `None` and types whose
 * every non-null value holds a `file`, which are disjoint from it. Load treats `file` as
 * disjoint from `Any`, `object`, and a record shaped like a file's JSON form, so a typed `file`
 * slot needs a `file`-typed source; the JSON form is accepted at runtime where load can't tell
 * (an untyped producer, a value decoded from a run record). `Inbox(T)` relates to `Inbox(U)` as
 * `T` to `U`, and to nothing else, `Any` and `object` included.
 *
 * @param inferred - Type the producing expression is inferred to have.
 * @param declared - Type the binding declares.
 * @returns `"fits"`, `"narrower"`, or `"disjoint"`.
 */
export function typeRelation(inferred: Type, declared: Type): TypeRelation {
  if (inferred.kind === ValueKind.NONE) return acceptsNone(declared) ? "fits" : "disjoint";
  if (declared.kind === ValueKind.NONE) return "disjoint";
  const relation = relateNonNull(inferred, declared);
  if (relation === "fits" && inferred.nullable && !declared.nullable) return "narrower";
  return relation;
}

/** Return whether `typ` admits the `None` value. */
function acceptsNone(typ: Type): boolean {
  return typ.kind === ValueKind.NONE || typ.nullable;
}

/** Return the least permissive of `relations`. */
function weakest(...relations: TypeRelation[]): TypeRelation {
  return relations.reduce((a, b) => (RELATION_RANK[b] < RELATION_RANK[a] ? b : a));
}

/** Return the most permissive of `relations`. */
function strongest(...relations: TypeRelation[]): TypeRelation {
  return relations.reduce((a, b) => (RELATION_RANK[b] > RELATION_RANK[a] ? b : a));
}

/** Return whether every item of `a` is in `b`. */
function isSubset(a: Iterable<string>, b: { has(item: string): boolean }): boolean {
  for (const item of a) if (!b.has(item)) return false;
  return true;
}

/**
 * Return a list type's element contract.
 *
 * A flat list reads as its non-null scalar items; a `LIST_ANY` with no element (an unproven
 * list) reads as `Optional(Any)` items, since they may be any JSON value.
 */
function listElement(typ: Type): Type {
  if (typ.element !== undefined) return typ.element;
  const elementKind = listElementKind(typ.kind);
  if (elementKind !== null) return Type.scalar(elementKind);
  return new Type({ kind: ValueKind.ANY, nullable: true });
}

/**
 * Whether `a` and `b` are the same memory type: the same name, and equal item records or none.
 * Nullability is left to the caller.
 */
function sameMemory(a: Type, b: Type): boolean {
  if (a.kind !== ValueKind.MEMORY || b.kind !== ValueKind.MEMORY || a.memory !== b.memory) return false;
  if (a.element === undefined || b.element === undefined) return a.element === b.element;
  return typeEquals(a.element, b.element);
}

/** Classify the non-null values of `inferred` against `declared`. */
function relateNonNull(inferred: Type, declared: Type): TypeRelation {
  if (inferred.kind === ValueKind.INBOX || declared.kind === ValueKind.INBOX) {
    if (inferred.kind !== declared.kind) return "disjoint";
    return typeRelation(inferred.element!, declared.element!);
  }
  if (inferred.kind === ValueKind.MEMORY || declared.kind === ValueKind.MEMORY) {
    return sameMemory(inferred, declared) ? "fits" : "disjoint";
  }
  if (declared.kind === ValueKind.ANY) {
    if (!hasFile(inferred)) return "fits";
    return requiresFile(inferred) ? "disjoint" : "narrower";
  }
  if (inferred.kind === ValueKind.ANY) return requiresFile(declared) ? "disjoint" : "narrower";
  if (inferred.members !== undefined || declared.members !== undefined) {
    return relateUnion(inferred, declared);
  }
  const inferredIsList = isListKind(inferred.kind);
  const declaredIsList = isListKind(declared.kind);
  if (inferredIsList || declaredIsList) {
    if (!(inferredIsList && declaredIsList)) return "disjoint";
    // The empty list is shared by every pair of list types; it is not overlap.
    return typeRelation(listElement(inferred), listElement(declared));
  }

  const relation = relateKinds(inferred.kind, declared.kind);
  if (relation === "disjoint") return relation;
  return weakest(
    relation,
    relateBounds(inferred, declared),
    relateTags(inferred, declared),
    relateFields(inferred, declared),
  );
}

/**
 * Classify the non-null values of `inferred` against `declared` member-wise.
 *
 * Either side may be a union; a non-union side counts as a one-member union. Each inferred
 * member takes its best relation against the declared members, since a value fits the union
 * when it fits any member. The results then combine as alternatives: fits when every member
 * fits, disjoint when every member is disjoint, and narrower otherwise.
 */
function relateUnion(inferred: Type, declared: Type): TypeRelation {
  const inferredMembers = inferred.members ?? [inferred];
  const declaredMembers = declared.members ?? [declared];
  const relations = new Set(
    inferredMembers.map((member) =>
      strongest(...declaredMembers.map((option) => relateNonNull(member, option))),
    ),
  );
  if (relations.size === 1) return [...relations][0]!;
  return "narrower";
}

/** Classify two non-list storage kinds before refinements are compared. */
function relateKinds(inferred: ValueKind, declared: ValueKind): TypeRelation {
  if (inferred === declared) return "fits";
  if (inferred === ValueKind.INTEGER && declared === ValueKind.NUMBER) return "fits";
  if (inferred === ValueKind.NUMBER && declared === ValueKind.INTEGER) {
    // A float-typed value can still be an int: a typed binding never converts.
    return "narrower";
  }
  if (STRING_STORED.has(inferred) && declared === ValueKind.STRING) return "fits";
  if (inferred === ValueKind.STRING && STRING_STORED.has(declared)) return "narrower";
  return "disjoint";
}

/** Compare numeric intervals; an absent bound or open end is unbounded. */
function relateBounds(inferred: Type, declared: Type): TypeRelation {
  if (declared.bounds === undefined) return "fits";
  if (intervalWithin(inferred, declared)) return "fits";
  const [low, high] = inferred.bounds ?? [null, null];
  const [declaredLow, declaredHigh] = declared.bounds;
  // The shared range runs from the larger low end to the smaller high end.
  let sharedLow = lowAtMost(low, declaredLow) ? declaredLow : low;
  let sharedHigh = highAtMost(high, declaredHigh) ? high : declaredHigh;
  if (inferred.kind === ValueKind.INTEGER) {
    // An int interval overlaps only if a whole number lies in the shared range. Only a float
    // end is rounded; an int end is already whole.
    if (typeof sharedLow === "number") sharedLow = Math.ceil(sharedLow);
    if (typeof sharedHigh === "number") sharedHigh = Math.floor(sharedHigh);
  }
  const overlaps = sharedLow === null || sharedHigh === null || sharedLow <= sharedHigh;
  return overlaps ? "narrower" : "disjoint";
}

/**
 * Compare `Literal` tag sets; a tag-less source into a `Literal` is narrower.
 *
 * A `Literal` against `date` or `datetime` (either way round) overlaps only when some tag is a
 * valid ISO string of that kind.
 */
function relateTags(inferred: Type, declared: Type): TypeRelation {
  if (inferred.tags !== undefined && STRING_STORED.has(declared.kind)) {
    return tagsMeetKind(inferred.tags, declared.kind);
  }
  if (declared.tags === undefined) return "fits";
  if (STRING_STORED.has(inferred.kind)) return tagsMeetKind(declared.tags, inferred.kind);
  if (inferred.tags === undefined) return "narrower";
  if (isSubset(inferred.tags, declared.tags)) return "fits";
  const declaredTags = declared.tags;
  return [...inferred.tags].some((tag) => declaredTags.has(tag)) ? "narrower" : "disjoint";
}

/** Return narrower when some tag is a valid `kind` string, else disjoint. */
function tagsMeetKind(tags: ReadonlySet<string>, kind: ValueKind): TypeRelation {
  for (const tag of tags) {
    try {
      buildValueAs(kind, tag);
    } catch (error) {
      if (error instanceof TypeCheckError) continue;
      throw error;
    }
    return "narrower";
  }
  return "disjoint";
}

/** Compare record fields, treating both records as closed. */
function relateFields(inferred: Type, declared: Type): TypeRelation {
  const declaredFields = declared.fields;
  if (declaredFields === undefined) return "fits";
  const inferredFields = inferred.fields;
  if (inferredFields === undefined) return "narrower";
  const inferredRequired = inferred.required ?? new Set<string>();
  const declaredRequired = declared.required ?? new Set<string>();
  if (!isSubset(inferredRequired, declaredFields)) return "disjoint";
  if (!isSubset(declaredRequired, inferredFields)) return "disjoint";
  let relation: TypeRelation = "fits";
  if (!isSubset(inferredFields.keys(), declaredFields)) relation = "narrower";
  if (!isSubset(declaredRequired, inferredRequired)) relation = "narrower";
  for (const [name, inferredField] of inferredFields) {
    const declaredField = declaredFields.get(name);
    if (declaredField === undefined) continue;
    let fieldRelation = typeRelation(inferredField, declaredField);
    if (fieldRelation === "disjoint") {
      // Two records still share values when both may omit the field or both may hold `None`
      // there.
      const bothOmit = !inferredRequired.has(name) && !declaredRequired.has(name);
      const bothNone = acceptsNone(inferredField) && acceptsNone(declaredField);
      if (!(bothOmit || bothNone)) return "disjoint";
      fieldRelation = "narrower";
    }
    relation = weakest(relation, fieldRelation);
  }
  return relation;
}

/**
 * Return an error message for a bad reference path, or `null` if acceptable.
 *
 * `path` is a single dotted ref the binding actually reads (the template's reference scan
 * already collected the per-span reference paths). Empty path segments are rejected at parse,
 * so the head is always present here.
 *
 * - `${input.X}` closed-checks against the flow's declared inputs.
 * - `${<node>.output[.X]}` (node-first) checks against real node ids; the second segment MUST
 *   be the literal `output` (the syntactic discriminator), or `new_messages` for a node whose
 *   pool name `<node>.new_messages` the caller lists among `validTargets` (an AGENT, where its
 *   new messages may be read).
 * - `${system.X}` is the strict clock ambient namespace; each ambient is a scalar, so a field
 *   read below it is an error.
 * - `extraHeads` is a per-call set of body-local scopes (e.g. MAP's `${item}`).
 * - Legacy `${inputs.X}` / `${outputs.X}` return the retired-plural-head typo hint ("did you
 *   mean `input` / `<node>.output`?").
 */
export function classifyPath(
  path: string,
  validTargets: { has(id: string): boolean },
  flowInputs: ReadonlySet<string>,
  extraHeads: readonly string[] = [],
): string | null {
  const ref = "reference ${" + path + "}";
  const parts = path.split(".");
  const head = parts[0]!;
  if (head === "input") {
    if (parts.length < 2) return `${ref} must be input.<name>`;
    if (!flowInputs.has(parts[1]!)) {
      return `${ref} points to unknown flow input ${reprText(parts[1]!)}`;
    }
    return null;
  }
  if (head === "inputs" || head === "outputs") {
    // A transitional typo hint for the retired plural heads.
    const suggestion = head === "inputs" ? "input" : "<node>.output";
    return (
      `${ref} uses the retired plural head ${reprText(head)} ` +
      `(did you mean \`${suggestion}\`?)`
    );
  }
  // A body-local scope (e.g. ${item}) is lenient: dotted access is allowed.
  if (extraHeads.includes(head)) return null;
  if (head === "system") {
    // A strict ambient namespace: only the declared ambients are valid.
    const ambient = parts.length >= 2 ? SYSTEM_AMBIENTS.get(parts[1]!) : undefined;
    if (ambient === undefined) {
      const bad = parts.length >= 2 ? parts[1]! : "";
      return `${ref} uses unknown system ambient ${reprText(bad)} (only today/now/run_id)`;
    }
    if (parts.length > 2) {
      // Every ambient is a scalar, so a field read can only drop segments at run time.
      return `${ref} reads field ${reprText(parts[2]!)} on type ${ambient}, which has no fields`;
    }
    return null;
  }
  // A node-id head: `${<node>.output[.X]}`.
  if (validTargets.has(head)) {
    // The 1-segment form `${<node>}` is an error (must say `<node>.output[.path]`).
    if (parts.length < 2) return `${ref} must be ${head}.output[.path]`;
    // An AGENT's second value: the caller lists `<node>.new_messages` for each node it may be read from.
    if (parts[1] === NEW_MESSAGES) {
      if (validTargets.has(messagesKey(head))) return null;
      return (
        `${ref} reads new_messages, which only an AGENT node has, from a node input, case, flow ` +
        `output or flow assert; node ${reprText(head)} isn't one there`
      );
    }
    // The second segment MUST be the literal `output` (the discriminator).
    if (parts[1] !== "output") {
      return (
        `${ref} on node ${reprText(head)} must be ${head}.output[.field] ` +
        "(`.output` is the node-value selector)"
      );
    }
    return null;
  }
  return `${ref} uses unknown namespace ${reprText(head)}`;
}
