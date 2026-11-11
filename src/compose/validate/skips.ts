/**
 * The skip model: which nodes of one flow level CASE routing may skip, and together.
 *
 * A node is skipped when the runtime join rule marks it dead: it has control edges (from CASEs)
 * and every one is skipped, or a required data group has every producer skipped, or a required
 * ordering edge (`depends_on`) has its producer skipped. A control edge is skipped when its CASE is
 * skipped or picked another branch. A data or ordering edge is skipped only when its producer is
 * skipped: one from a CASE is not a branch, so it is taken whichever branch the CASE picks. A CASE
 * that runs picks one of its branches, or no branch at all unless it is exhaustive. CASE
 * conditions are treated as independent, so the model enumerates every combination of the choices
 * of the CASEs that can skip an ancestor of the queried nodes. The default side of `:-` is checked
 * as if it always runs: `${a.output :- (b.output + 1)}` is rejected even when `b` runs whenever `a`
 * is skipped, which is a false positive only when `a` cannot be null while it runs.
 *
 * The enumeration is bounded by `current().maxSkipWorlds` assignments (complete or cut short) per
 * query. A partial assignment that reaches a CASE in a state already ruled out there (the same
 * picks and skips among the nodes still to be read) is not explored again, and counts as one
 * assignment. Past the bound each query falls back to the answer that is safe for its caller:
 * `mayAllSkip` says true (the output gets a null default it may not need) and
 * `maySkipWhileRunning` says false (nothing is reported). A level whose edges have a cycle has no
 * skippable nodes here; the cycle check reports the cycle.
 *
 * `nullSkippedOutputs` gives a flow output that reads only nodes that may all be skipped the
 * explicit null default, so it returns null instead of leaving END unreachable.
 * `checkSkippedArithmetic` rejects arithmetic over a value that may be null because its producers
 * were skipped while the reading node (or END) still runs.
 *
 * Knows about: compiled edges and their topological order, flow outputs, node wiring, binding
 * producers, and expression trees.
 */

import { type Edge, FlowOutput } from "../../compile/model.ts";
import { topologicalOrder } from "../../compile/validation.ts";
import { current } from "../../defaults.ts";
import { compareCodeUnits, type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import {
  bindingCoSkips,
  ExpressionError,
  exprRefs,
  optionalAfterRoots,
  referencePath,
  referenceText,
  scanTemplate,
} from "../../expr/index.ts";
import { type ExprNode, iterSubtreesTopdown, Token, Tree } from "../../expr/grammar.ts";
import { Span } from "../../expr/template.ts";
import {
  asBinding,
  Binding,
  bindingSource,
  iterSourceStrings,
  mapSourceStrings,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import { poolRootLength } from "../../typesys/pool.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { TypeCheckError } from "../../typesys/values.ts";
import { bindingProducers, refProducer } from "../edges/producers.ts";
import {
  conjuncts,
  type Guard,
  guardedReads,
  Holds,
  IsinstanceCheck,
  NOTHING,
  NotNullCheck,
} from "../helpers/condition_guards.ts";
import { LoadDiagnostics, locatedError } from "../helpers/errors.ts";
import { resolveIsinstanceType } from "../helpers/isinstance_types.ts";
import { directReferenceParts, directReferencePath } from "../../compile/output_contracts.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import { parsedWhen } from "./narrowing.ts";
import type { ResolveOrigin } from "./typed_bindings.ts";

/** The grammar's arithmetic trees: the `+ -`, `* / %` and `**` chains and unary minus. */
export const ARITHMETIC: ReadonlySet<string> = new Set(["sum", "product", "power", "neg"]);

/** One searched CASE's authored `when:`: its case index, text, origin and range. */
export type SearchedWhen = readonly [index: number, when: unknown, origin: OriginRef | null, range: SourceRange | null];

/** The incoming edges of one node that can make it skip. */
class Join {
  /** `[case, handle]` per control edge. */
  readonly control: [string, string][] = [];
  /**
   * Input group to `[optional, producers]`; a group takes its first edge's `optional`, like the
   * runtime join.
   */
  readonly groups = new Map<string | null, [boolean, string[]]>();
  readonly requiredOrder: string[] = [];

  /** Return the producers of the edges that can make this node skip. */
  skipSources(): Set<string> {
    const sources = new Set([...this.control.map(([caseId]) => caseId), ...this.requiredOrder]);
    for (const [optional, producers] of this.groups.values()) {
      if (!optional) for (const producer of producers) sources.add(producer);
    }
    return sources;
  }
}

/**
 * The nodes the search has passed that are still read later and are skipped or picked a branch,
 * by topological position, newest first. Partial assignments share their common start.
 */
type Trail = { readonly index: number; readonly state: number; readonly previous: Trail | null };

/** A CASE the search is branching on, with the search state on arriving at it. */
type Frame = {
  /** The CASE's index in the walk's `nodes`. */
  readonly index: number;
  /** The next of the CASE's choices to try. */
  next: number;
  readonly hash: number;
  readonly live: number;
  readonly trail: Trail | null;
};

/** A `Join` over topological positions, keeping only edges that can skip, and its skip sources. */
type IndexedJoin = {
  /** `case, choice` pairs, one per control edge; empty when some control edge is always taken. */
  readonly control: Int32Array;
  readonly groups: readonly Int32Array[];
  readonly requiredOrder: Int32Array;
  /** The positions of its skip sources in the level, the `parents` entry. */
  readonly parents: Int32Array;
  /** The positions of the CASEs with choices it has a control edge from. */
  readonly cases: Int32Array;
};

/** A position's search flags. */
const IN_CLOSURE = 1;
const TARGET = 2;
const BRANCHING = 4;

/** A 32-bit hash of node `index` in `state`; the search XORs one per live node into a state hash. */
function stateHash(index: number, state: number): number {
  let h = Math.imul(index ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(state + 0x632be5ab, 0xc2b2ae35);
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d);
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b);
  return h ^ (h >>> 16);
}

/**
 * Which nodes of one flow level may be skipped by its CASE routing.
 *
 * @example
 * const edges = [new Edge("t", "c", "a", { sourceHandle: "then" }), new Edge("e", "c", "b", { sourceHandle: "else" })];
 * new SkipModel(["c", "a", "b"], edges, new Set(["c"])).mayAllSkip(["a", "b"]); // false
 */
export class SkipModel {
  private readonly nodeIds: string[];
  private readonly known: Set<string>;
  private readonly limit: number;
  private readonly joins = new Map<string, Join>();
  /** A CASE's choices: each branch, then "no branch" (`null`) unless it is exhaustive. */
  private readonly choices = new Map<string, readonly (string | null)[]>();
  private readonly parents = new Map<string, Set<string>>();
  /** Each node's index in topological order, or `null` when the level has a cycle. */
  private readonly position: Map<string, number> | null;
  private readonly order: string[] | null;
  /** Each position's join; empty when the level has a cycle or no CASE, so it is never searched. */
  private readonly indexed: readonly IndexedJoin[] = [];
  /** How many choices the CASE at each position has; 0 for any other node. */
  private readonly choiceCount: Int32Array;
  // Per-position scratch for one search. Between queries every entry is at rest: `state` 0,
  // `lastUse` -1, `flags` 0. `members` holds the closure's positions during a search.
  private readonly state: Int32Array;
  private readonly lastUse: Int32Array;
  private readonly flags: Uint8Array;
  private readonly members: Int32Array;

  /**
   * @param nodeIds The level's node ids, before the START/END boundary is synthesized. Any other
   *   id (START, a flow input) is a producer that never skips.
   * @param edges The level's edges. An edge into a node outside `nodeIds` is ignored.
   * @param exhaustive Ids of the CASEs that always take a branch when they run.
   * @param options.limit Most CASE assignments one query examines before it falls back; `null`
   *   (the default) reads `current().maxSkipWorlds`.
   */
  constructor(
    nodeIds: Iterable<string>,
    edges: Iterable<Edge>,
    exhaustive: ReadonlySet<string>,
    options: { limit?: number | null } = {},
  ) {
    this.nodeIds = [...new Set(nodeIds)];
    this.known = new Set(this.nodeIds);
    this.limit = options.limit ?? current().maxSkipWorlds;
    const graphParents = new Map<string, Set<string>>();
    for (const node of this.nodeIds) {
      this.joins.set(node, new Join());
      graphParents.set(node, new Set());
    }
    const handles = new Map<string, Set<string>>();
    for (const edge of edges) {
      if (!this.known.has(edge.to)) continue;
      const join = this.joins.get(edge.to)!;
      if (this.known.has(edge.from)) graphParents.get(edge.to)!.add(edge.from);
      if (edge.sourceHandle !== null) {
        join.control.push([edge.from, edge.sourceHandle]);
        if (this.known.has(edge.from)) {
          if (!handles.has(edge.from)) handles.set(edge.from, new Set());
          handles.get(edge.from)!.add(edge.sourceHandle);
        }
      } else if (edge.ordering) {
        if (!edge.optional) join.requiredOrder.push(edge.from);
      } else {
        if (!join.groups.has(edge.inputGroup)) join.groups.set(edge.inputGroup, [edge.optional, []]);
        join.groups.get(edge.inputGroup)![1].push(edge.from);
      }
    }
    for (const [caseId, names] of handles) {
      this.choices.set(caseId, exhaustive.has(caseId) ? [...names] : [...names, null]);
    }
    // Only these edges can skip a node, so only they decide which nodes a query must model.
    for (const node of this.nodeIds) {
      const sources = this.joins.get(node)!.skipSources();
      this.parents.set(node, new Set([...sources].filter((source) => this.known.has(source))));
    }
    const graphEdges = [...graphParents].flatMap(([node, sources]) => [...sources].map((from) => ({ from, to: node })));
    const { order, rest } = topologicalOrder(this.nodeIds, graphEdges);
    this.order = rest.length === 0 ? order : null;
    this.position = this.order === null ? null : new Map(this.order.map((node, index) => [node, index]));
    const size = this.order !== null && this.hasCases ? this.order.length : 0;
    this.choiceCount = new Int32Array(size);
    this.state = new Int32Array(size);
    this.lastUse = new Int32Array(size).fill(-1);
    this.flags = new Uint8Array(size);
    this.members = new Int32Array(size);
    if (size > 0) {
      for (const [caseId, choices] of this.choices) this.choiceCount[this.position!.get(caseId)!] = choices.length;
      this.indexed = this.order!.map((node) => this.indexedJoin(node));
    }
  }

  /** The level's node ids. */
  get nodes(): ReadonlySet<string> {
    return new Set(this.known);
  }

  /** Whether some CASE routes to a node of the level; without one nothing skips. */
  get hasCases(): boolean {
    return this.choices.size > 0;
  }

  /**
   * Return whether some CASE routing skips every node in `producers`.
   *
   * @param producers Node ids. An id outside the level never skips.
   * @returns `true` when one assignment of CASE choices skips them all, or when the search hits
   *   its bound. `false` for an empty set or a cyclic level.
   */
  mayAllSkip(producers: Iterable<string>): boolean {
    const targets = new Set(producers);
    if (!this.queryable(targets)) return false;
    return this.search(targets, null, true);
  }

  /**
   * Return whether some CASE routing skips every node in `producers` but runs `consumer`.
   *
   * @param producers Node ids. An id outside the level never skips.
   * @param consumer The node that must run. `null` (or an id outside the level) is a consumer
   *   that always runs, such as the flow's END.
   * @returns `true` when one assignment of CASE choices does so. `false` for an empty set, a
   *   cyclic level, or when the search hits its bound.
   */
  maySkipWhileRunning(producers: Iterable<string>, consumer: string | null): boolean {
    const targets = new Set(producers);
    if (!this.queryable(targets)) return false;
    return this.search(targets, consumer !== null && this.known.has(consumer) ? consumer : null, false);
  }

  /**
   * Return whether `targets` is a non-empty set of the level's nodes, and the level is acyclic and
   * has a CASE (without one nothing skips).
   */
  private queryable(targets: ReadonlySet<string>): boolean {
    return this.indexed.length > 0 && targets.size > 0 && [...targets].every((target) => this.known.has(target));
  }

  /**
   * Depth-first search for a CASE assignment that skips `targets` and runs `consumer`.
   *
   * It walks the queried nodes and their ancestors in topological order and branches on each
   * running CASE whose pick can skip one of them, trying its choices in order. Whether a later
   * node skips depends only on the nodes it reads, so arriving at a CASE with the same picks and
   * skips among the nodes still to be read as a state already ruled out there has the same
   * outcome: that state is ruled out at once and counts as one assignment. A ruled-out state
   * stands in for at least one failed assignment, so the search never counts more than a plain
   * enumeration and gives the same answer wherever that one stays within the bound.
   *
   * @param targets Node ids in the level that must all be skipped.
   * @param consumer A node id in the level that must run, or `null` for no such node.
   * @param onLimit The answer when the search examines `limit` assignments without finding one.
   * @returns Whether such an assignment exists, or `onLimit` past the bound.
   */
  private search(targets: ReadonlySet<string>, consumer: string | null, onLimit: boolean): boolean {
    const { indexed, flags, members } = this;
    const position = this.position!;
    let count = 0;
    for (const target of targets) {
      const at = position.get(target)!;
      flags[at] = IN_CLOSURE | TARGET;
      members[count++] = at;
    }
    const consumerAt = consumer === null ? -1 : position.get(consumer)!;
    if (consumerAt !== -1 && flags[consumerAt] === 0) {
      flags[consumerAt] = IN_CLOSURE;
      members[count++] = consumerAt;
    }
    let branches = false;
    for (let next = 0; next < count; next++) {
      const join = indexed[members[next]!]!;
      for (const parent of join.parents) {
        if ((flags[parent]! & IN_CLOSURE) === 0) {
          flags[parent] = flags[parent]! | IN_CLOSURE;
          members[count++] = parent;
        }
      }
      for (const caseAt of join.cases) {
        flags[caseAt] = flags[caseAt]! | BRANCHING;
        branches = true;
      }
    }
    const nodes = members.subarray(0, count).sort((a, b) => a - b);
    try {
      // With no CASE that can skip one of them, every node runs, so the first target fails.
      if (!branches) return false;
      if (this.limit < 1) return onLimit;
      return this.walk(nodes, consumerAt, onLimit);
    } finally {
      for (const at of nodes) {
        flags[at] = 0;
        this.state[at] = 0;
        this.lastUse[at] = -1;
      }
    }
  }

  /**
   * The search's depth-first walk over the CASE assignments of one closure, flagged in `flags`.
   *
   * @param nodes The closure's positions in topological order.
   * @param consumerAt The position of the node that must run, or -1 for none.
   * @param onLimit The answer past the bound.
   * @returns Whether an assignment skips every target and runs the consumer, or `onLimit`.
   */
  private walk(nodes: Int32Array, consumerAt: number, onLimit: boolean): boolean {
    const { indexed, flags, state, lastUse, choiceCount } = this;
    const count = nodes.length;
    // A node's state matters to the search until its last reader; past it, the node is dropped
    // from the state that decides whether two partial assignments have the same future.
    for (const at of nodes) for (const parent of indexed[at]!.parents) lastUse[parent] = at;

    // Each node's `state` is 0 when it runs (a CASE with one choice picks it), 1 when skipped, and
    // 2 + k for a CASE that runs and picks its choice k. `hash` is the XOR of `stateHash` over the
    // live nodes (passed, read later, state not 0), and `live` their count.
    let hash = 0;
    let live = 0;
    let trail: Trail | null = null;
    const frames: Frame[] = [];
    // The trails of the states ruled out at each CASE position, by state hash.
    const failed = new Map<number, Map<number, (Trail | null)[]>>();
    let worlds = 0;

    /** Whether the join rule marks the node at position `at` dead in the current state. */
    const dead = (at: number): boolean => {
      const { control, groups, requiredOrder } = indexed[at]!;
      if (control.length > 0) {
        let all = true;
        for (let edge = 0; edge < control.length; edge += 2) {
          const picked = state[control[edge]!]!;
          if (picked !== 1 && (picked === 0 ? 0 : picked - 2) === control[edge + 1]) {
            all = false;
            break;
          }
        }
        if (all) return true;
      }
      // A data or ordering edge carries no handle, so the runtime takes it on every route of a
      // CASE that runs; it is skipped only when its producer is.
      for (const producers of groups) {
        let all = true;
        for (const producer of producers) {
          if (state[producer] !== 1) {
            all = false;
            break;
          }
        }
        if (all) return true;
      }
      for (const producer of requiredOrder) if (state[producer] === 1) return true;
      return false;
    };
    /** Set the node at position `at` to `value`, and drop the nodes whose last reader it is. */
    const pass = (at: number, value: number): void => {
      state[at] = value;
      if (value !== 0 && lastUse[at]! > at) {
        hash ^= stateHash(at, value);
        live += 1;
        trail = { index: at, state: value, previous: trail };
      }
      for (const done of indexed[at]!.parents) {
        if (lastUse[done] === at && state[done] !== 0) {
          hash ^= stateHash(done, state[done]!);
          live -= 1;
        }
      }
    };
    /** Whether the live nodes at position `at` have the states `stored` recorded there. */
    const same = (stored: Trail | null, at: number): boolean => {
      let matched = 0;
      for (let entry = stored; entry !== null; entry = entry.previous) {
        if (lastUse[entry.index]! < at) continue;
        if (state[entry.index] !== entry.state) return false;
        matched += 1;
      }
      return matched === live;
    };

    let index = 0;
    for (;;) {
      let complete = true;
      for (; index < count; index++) {
        const at = nodes[index]!;
        const skipped = dead(at);
        if (((flags[at]! & TARGET) !== 0 && !skipped) || (at === consumerAt && skipped)) {
          complete = false;
          break;
        }
        if (!skipped && (flags[at]! & BRANCHING) !== 0 && choiceCount[at]! > 1) {
          if (failed.get(at)?.get(hash)?.some((stored) => same(stored, at))) {
            complete = false;
            break;
          }
          frames.push({ index, next: 1, hash, live, trail });
          pass(at, 2);
        } else {
          pass(at, skipped ? 1 : 0);
        }
      }
      if (complete) return true;
      worlds += 1;
      // Back to the newest CASE with a choice left; each CASE left behind rules out its state.
      for (;;) {
        const frame = frames.at(-1);
        if (frame === undefined) return false;
        for (let undo = frame.index; undo < index; undo++) state[nodes[undo]!] = 0;
        index = frame.index;
        hash = frame.hash;
        live = frame.live;
        trail = frame.trail;
        const at = nodes[index]!;
        if (frame.next < choiceCount[at]!) {
          if (worlds >= this.limit) return onLimit;
          pass(at, 2 + frame.next);
          frame.next += 1;
          index += 1;
          break;
        }
        frames.pop();
        let byHash = failed.get(at);
        if (byHash === undefined) failed.set(at, (byHash = new Map()));
        if (!byHash.has(hash)) byHash.set(hash, []);
        byHash.get(hash)!.push(trail);
      }
    }
  }

  /**
   * Return `node`'s join over topological positions.
   *
   * @param node A node of the level.
   * @returns Its edges that can make it skip. A control edge from an id outside the level is
   *   always taken, and a producer outside the level never skips.
   */
  private indexedJoin(node: string): IndexedJoin {
    const join = this.joins.get(node)!;
    const position = this.position!;
    const control: number[] = [];
    const cases: number[] = [];
    let taken = false;
    for (const [caseId, handle] of join.control) {
      const choices = this.choices.get(caseId);
      if (choices === undefined) {
        taken = true;
        continue;
      }
      const caseAt = position.get(caseId)!;
      cases.push(caseAt);
      control.push(caseAt, choices.indexOf(handle));
    }
    const groups: Int32Array[] = [];
    for (const [optional, producers] of join.groups.values()) {
      if (!optional && producers.every((producer) => this.known.has(producer))) {
        groups.push(Int32Array.from(producers, (producer) => position.get(producer)!));
      }
    }
    const requiredOrder = join.requiredOrder.filter((producer) => this.known.has(producer));
    return {
      control: Int32Array.from(taken ? [] : control),
      groups,
      requiredOrder: Int32Array.from(requiredOrder, (producer) => position.get(producer)!),
      parents: Int32Array.from(this.parents.get(node)!, (parent) => position.get(parent)!),
      cases: Int32Array.from(new Set(cases)),
    };
  }
}

/**
 * Give each flow output whose producers may all be skipped the explicit null default.
 *
 * @param flowOutputs The flow level's declared outputs.
 * @param model The skip model of the same level.
 * @returns The outputs in order. A co-skipping source `${E}` whose producers may all be skipped
 *   becomes `${E:-null}` when `E` is one reference, else `${(E) :- null}`, with `?.` on the first
 *   step after each producer root; its origin and declared contract are kept. Each string nested
 *   in a value template is rewritten on its own the same way. Every other source is kept as is.
 *
 * @example
 * const model = new SkipModel(["c", "a"], [new Edge("t", "c", "a", { sourceHandle: "then" })], new Set());
 * nullSkippedOutputs([new FlowOutput("v", "${a.output}")], model)[0].binding.source; // "${a.output:-null}"
 * nullSkippedOutputs([new FlowOutput("v", "${a.output.qty}")], model)[0].binding.source; // "${a.output?.qty:-null}"
 */
export function nullSkippedOutputs(flowOutputs: readonly FlowOutput[], model: SkipModel): FlowOutput[] {
  if (!model.hasCases) return [...flowOutputs];

  const nullDefault = (text: string): string => {
    if (!(bindingCoSkips(text) && model.mayAllSkip(bindingProducers(text)))) return text;
    // A co-skipping source is exactly one whole `${...}` span. A field of a skipped producer reads
    // null through `?.`, whatever the skip query's bound.
    const interior = optionalAfterRoots(text.slice(2, -1), poolRootLength, isProducerRoot);
    if (directReferencePath(text) !== null) return "${" + interior + ":-null}";
    return "${(" + interior + ") :- null}";
  };

  return flowOutputs.map(
    (output) => new FlowOutput(output.name, mapSourceStrings(output.binding, nullDefault)),
  );
}

/**
 * Reject arithmetic over a value that may be null because CASE routing skipped it.
 *
 * An operand of `+ - * / % **` or unary `-` is null from a skip when it is a reference to a node's
 * output, or a `|` / `:-` group of them, and every producer of one of its null alternatives is
 * skipped. That is an error when some routing skips them while the binding's node still runs; a
 * flow output's END always runs. Any other operand (a call, a nested operator, a literal) is never
 * null from a skip here. Each string nested in a value template is checked on its own and located
 * at its own site. A searched CASE's authored `when:` is checked the same way with the CASE as the
 * reader, since a reference under `:-`, `|` or `:?` there does not skip the CASE with its producer.
 *
 * @param flowWiring Each node's `param -> source` wiring, a CASE's `on:` and `when:` sources
 *   included.
 * @param flowOutputs The flow level's outputs, after `nullSkippedOutputs`.
 * @param model The skip model of the same level.
 * @param buildResults The level's built nodes keyed by id, for the binding site names and fallback
 *   ranges.
 * @param options.registry The flow's typedefs, which an `isinstance` guard's type name resolves
 *   against.
 * @param options.valueMode Whether the flow has a single `output:` value, named `output` in errors
 *   (default `false`).
 * @param options.outputsRange The authored `output:` range, the fallback location of an output
 *   error.
 * @param options.resolveOrigin Resolves a binding's registered origin to its source frame, the
 *   preferred location.
 * @param options.diagnostics Factory for located load errors.
 * @param options.searchedWhens Each searched CASE's id mapped to one `[case index, authored when,
 *   origin, range]` per case: the origin is the preferred location, the range the fallback. A
 *   `when:` is checked like a binding: arithmetic behind a guard that proves its root not null
 *   (`a.output?.n != null and a.output.n + 1 > 1`) loads.
 * @throws LoadError at the first such binding or `when:`, naming it, the operand and its producers,
 *   and asking for a `:-` default.
 */
export function checkSkippedArithmetic(
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  flowOutputs: readonly FlowOutput[],
  model: SkipModel,
  buildResults: ReadonlyMap<string, BuildResult>,
  options: {
    registry: TypeRegistry;
    valueMode?: boolean;
    outputsRange?: SourceRange | null;
    resolveOrigin?: ResolveOrigin | null;
    diagnostics?: LoadDiagnostics | null;
    searchedWhens?: ReadonlyMap<string, readonly SearchedWhen[]> | null;
  },
): void {
  if (!model.hasCases) return;
  const { registry, valueMode = false, outputsRange = null, resolveOrigin = null, searchedWhens = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const known = model.nodes;
  for (const [nodeId, wiring] of flowWiring) {
    const result = buildResults.get(nodeId) ?? null;
    for (const [param, binding] of wiring) {
      const found = skippedOperand(binding, model, nodeId, known, registry);
      if (found === null) continue;
      const [site, operand, producers] = found;
      const [where, fallback] = bindingSite(nodeId, param, result);
      throw locatedError(
        `${where}: ${arithmeticMessage(site, operand, producers)}`,
        site,
        fallback,
        resolveOrigin,
        diagnostics,
      );
    }
  }
  for (const [nodeId, whens] of searchedWhens ?? new Map<string, readonly SearchedWhen[]>()) {
    for (const [index, when, origin, whenRange] of whens) {
      // The CASE build reports a malformed condition.
      const tree = parsedWhen(when);
      if (tree === null) continue;
      const found = skippedOperandIn(tree, model, nodeId, known, registry);
      if (found === null) continue;
      throw locatedError(
        `node ${reprText(nodeId)} cases[${index}].when: ${arithmeticText(String(when).trim(), ...found)}`,
        new Binding(when, { origin }),
        whenRange,
        resolveOrigin,
        diagnostics,
      );
    }
  }
  for (const output of flowOutputs) {
    const found = skippedOperand(output.binding, model, null, known, registry);
    if (found === null) continue;
    const [site, operand, producers] = found;
    const label = valueMode ? "output" : `output ${reprText(output.name)}`;
    throw locatedError(
      `${label}: ${arithmeticMessage(site, operand, producers)}`,
      site,
      outputsRange,
      resolveOrigin,
      diagnostics,
    );
  }
}

/**
 * Return how a message names one node binding, and where it points when the binding's origin
 * doesn't resolve: the binding's own entry, else the node's `input:` mapping, else the node.
 *
 * A reserved key (`over`, `until`) is named and located as its kind field, any other param as an
 * input. Without a build result the binding is named by node id and param, with no location.
 *
 * @param nodeId The node's id.
 * @param param The binding's wiring key.
 * @param result The node's build, or `null` when there is none.
 * @returns The binding's name and its fallback range.
 */
export function bindingSite(
  nodeId: string,
  param: string,
  result: BuildResult | null,
): [where: string, fallback: SourceRange | null] {
  if (result === null) return [`node ${reprText(nodeId)} input ${reprText(param)}`, null];
  const isReserved = result.node.reservedWiringKeys().has(param);
  const label = isReserved ? result.node.reservedKeyLabel(param) : result.node.wiringRefLabel(param);
  const own = result.fieldRanges.get(siteKey(isReserved ? "field" : "input", param));
  const fallback = own ?? result.fieldRanges.get(siteKey("field", "inputs")) ?? result.nodeRange;
  // A synthetic param's label may already end in a colon (`on:`); the message adds one.
  return [result.siteName(nodeId, label.replace(/:+$/, "")), fallback];
}

/**
 * Return each template string of `binding` as a binding located at its own site.
 *
 * A string nested in a value template takes its registered origin, else the outer binding's.
 */
export function stringSites(binding: unknown): Binding[] {
  const outer = asBinding(binding);
  return [...iterSourceStrings(outer)].map(([path, text]) => {
    const origin = outer.source instanceof ValueTemplate ? outer.source.originAt(path) : null;
    return new Binding(text, { origin: origin ?? outer.origin });
  });
}

/** An arithmetic operand that may be null from a skip, and the producers whose skip makes it null. */
type SkippedOperand = [operand: ExprNode, producers: string[]];

/**
 * Return the first arithmetic operand of `binding` that may be null from a skip.
 *
 * @param binding A wiring or output binding; each of its template strings is scanned.
 * @param model The level's skip model.
 * @param consumer The node that evaluates the binding, or `null` for END.
 * @param known The level's node ids.
 * @param registry The flow's typedefs, for an `isinstance` guard's type name.
 * @returns The string that does the arithmetic (as a binding located at its own site), the
 *   operand, and the producers whose skip makes it null (in source order), or `null` when no
 *   operand may be null while `consumer` runs.
 */
function skippedOperand(
  binding: unknown,
  model: SkipModel,
  consumer: string | null,
  known: ReadonlySet<string>,
  registry: TypeRegistry,
): [site: Binding, ...SkippedOperand] | null {
  for (const site of stringSites(binding)) {
    let segments;
    try {
      segments = scanTemplate(site.source as string);
    } catch (error) {
      // Reference validation reports a malformed source.
      if (error instanceof ExpressionError) continue;
      throw error;
    }
    for (const segment of segments) {
      if (!(segment instanceof Span)) continue;
      const found = skippedOperandIn(segment.tree, model, consumer, known, registry);
      if (found !== null) return [site, ...found];
    }
  }
  return null;
}

/**
 * Return the first arithmetic operand of a parsed expression that may be null from a skip.
 *
 * @param expression A parsed expression.
 * @param model The level's skip model.
 * @param consumer The node that evaluates the expression, or `null` for END.
 * @param known The level's node ids.
 * @param registry The flow's typedefs, for an `isinstance` guard's type name.
 * @returns The operand and the producers whose skip makes it null (in source order), or `null`
 *   when no operand may be null while `consumer` runs.
 */
function skippedOperandIn(
  expression: ExprNode,
  model: SkipModel,
  consumer: string | null,
  known: ReadonlySet<string>,
  registry: TypeRegistry,
): SkippedOperand | null {
  // A lone token has no operator.
  if (!(expression instanceof Tree)) return null;
  // Matched by node, not by position: an operand that is no plain read has no guard here.
  const reads = guardedReads(expression);
  for (const tree of iterSubtreesTopdown(expression)) {
    if (!ARITHMETIC.has(tree.data)) continue;
    for (const operand of tree.children.flatMap(branchesOf)) {
      const guard = reads.find((read) => read.node === operand)?.guard ?? NOTHING;
      if (operand instanceof Tree && provesNotNull(guard, referencePath(operand), registry)) continue;
      for (const group of nullGroups(operand, known)) {
        if (group.size > 0 && model.maySkipWhileRunning(group, consumer)) {
          const producers = new Set(exprRefs(operand).map(refProducer));
          return [operand, [...producers].filter((producer): producer is string => producer !== null && group.has(producer))];
        }
      }
    }
  }
  return null;
}

/**
 * Return whether `guard` proves the root of the read `path` not null: a `!= null` check, or an
 * `isinstance` to a type that excludes null, on the root or a path below it holds whenever the
 * read runs. Such a check on a path below the root holds only when the root isn't null.
 */
function provesNotNull(guard: Guard, path: string, registry: TypeRegistry): boolean {
  const names = path.split(".");
  const root = names.slice(0, poolRootLength(names)).join(".");
  return conjuncts(guard).some(
    (part) =>
      part instanceof Holds &&
      part.value &&
      (part.check.base === root || part.check.base.startsWith(root + ".")) &&
      (part.check instanceof NotNullCheck ||
        (part.check instanceof IsinstanceCheck && excludesNull(part.check.typeName, registry))),
  );
}

/** Whether the type an `isinstance` names excludes null; an unknown name proves nothing. */
function excludesNull(typeName: string, registry: TypeRegistry): boolean {
  try {
    return !resolveIsinstanceType(typeName, registry).nullable;
  } catch (error) {
    // The expression checks report the unknown name.
    if (error instanceof TypeCheckError) return false;
    throw error;
  }
}

/**
 * Return an arithmetic operand as the operands it may evaluate to: each branch of a conditional,
 * whose value is the branch's, else the operand itself.
 */
function branchesOf(operand: ExprNode): ExprNode[] {
  if (!(operand instanceof Tree) || operand.data !== "if_expr") return [operand];
  return [operand.children[0]!, operand.children[2]!].flatMap(branchesOf);
}

/**
 * Return the producer sets whose joint skip makes `operand` null.
 *
 * @param operand One parsed operand of an arithmetic tree.
 * @param known The level's node ids; a reference to anything else (an input, `item`, `system`) is
 *   never skipped.
 * @returns The operand is null when every producer of one set is skipped. A `null` literal gives
 *   one empty set (always null); a slice gives its base's sets; an operand that is never null
 *   from a skip gives no sets.
 */
function nullGroups(operand: ExprNode, known: ReadonlySet<string>): Set<string>[] {
  if (operand instanceof Token) return operand.type === "NULL" ? [new Set()] : [];
  if (operand.data === "refcall") {
    const parts = directReferenceParts(operand);
    return parts !== null ? referenceGroups(parts.join("."), known) : [];
  }
  if (operand.data === "coalesce" || operand.data === "default_expr") {
    // Null only when every alternative is null: one set from each, unioned.
    let combined: Set<string>[] = [new Set()];
    for (const child of operand.children) {
      const groups = nullGroups(child, known);
      // Keep the first of equal sets, in order.
      const unique = new Map<string, Set<string>>();
      for (const a of combined) {
        for (const b of groups) {
          const union = new Set([...a, ...b]);
          const key = JSON.stringify([...union].sort(compareCodeUnits));
          if (!unique.has(key)) unique.set(key, union);
        }
      }
      combined = [...unique.values()];
    }
    return combined;
  }
  // A slice of null is a miss, so it is null whenever its base is.
  if (operand.data === "subscript") return nullGroups(operand.children[0]!, known);
  return [];
}

/**
 * Return `[{producer}]` for a reference to a level node's output, else `[]`.
 *
 * @param path A dotted reference path.
 * @param known The level's node ids.
 * @returns The one producer set that makes the reference null when skipped, or none.
 */
function referenceGroups(path: string, known: ReadonlySet<string>): Set<string>[] {
  const producer = refProducer(path);
  return producer !== null && known.has(producer) ? [new Set([producer])] : [];
}

/**
 * Return the error text for arithmetic on an operand that may be null from a skip.
 *
 * @param binding The binding whose source does the arithmetic.
 * @param operand The operand that may be null.
 * @param producers The nodes whose skip makes it null.
 * @returns The message, without the binding's name.
 */
function arithmeticMessage(binding: unknown, operand: ExprNode, producers: readonly string[]): string {
  return arithmeticText(String(bindingSource(binding)), operand, producers);
}

/**
 * Return the error text for arithmetic in `source` on an operand that may be null.
 *
 * @param source The authored text that does the arithmetic, shown as written.
 * @param operand The operand that may be null.
 * @param producers The nodes whose skip makes it null.
 * @returns The message, without the reader's name.
 */
function arithmeticText(source: string, operand: ExprNode, producers: readonly string[]): string {
  const text = operandText(operand);
  const hint = defaultHint(operand);
  const quoted = producers.map(reprText);
  const who =
    quoted.length === 1
      ? `node ${quoted[0]} is`
      : `nodes ${quoted.slice(0, -1).join(", ")} and ${quoted.at(-1)} are`;
  return `${source} does arithmetic on ${text}, which is null when ${who} skipped; give it a default with :-, e.g. ${hint}`;
}

/**
 * Return the operand with a `:- <default>` placeholder, replacing a trailing `:- null`; a field of a
 * producer is read through `?.` after its root, since `:-` sees only a null root.
 *
 * @param operand The operand that may be null.
 * @returns Parenthesized text such as `(order.output?.qty :- <default>)`.
 */
function defaultHint(operand: ExprNode): string {
  const read = (part: ExprNode) => optionalAfterRoots(operandText(part), poolRootLength, isProducerRoot);
  const last = operand instanceof Tree ? operand.children.at(-1) : undefined;
  if (operand instanceof Tree && operand.data === "default_expr" && last instanceof Token && last.type === "NULL") {
    const kept = operand.children.slice(0, -1).map(read);
    return "(" + [...kept, "<default>"].join(" :- ") + ")";
  }
  return `(${read(operand)} :- <default>)`;
}

/** Whether a reference root is a node's output or new messages, which a skip leaves null. */
function isProducerRoot(root: string): boolean {
  return /\.(output|new_messages)$/.test(root);
}

/**
 * Return source text for an operand that may be null: a reference or a `|`/`:-` group.
 *
 * @param operand A reference, a wrapped reference, `null`, a slice of one (printed as its base),
 *   or a coalesce or default of them.
 * @returns The operand's text, `?.` as written; a group is parenthesized.
 */
function operandText(operand: ExprNode): string {
  if (operand instanceof Token) return String(operand);
  // A slice prints its base, the value that may be null.
  if (operand.data === "subscript") return operandText(operand.children[0]!);
  if (directReferenceParts(operand) !== null) return referenceText(operand);
  const joiner = operand.data === "coalesce" ? " | " : " :- ";
  return "(" + operand.children.map(operandText).join(joiner) + ")";
}
