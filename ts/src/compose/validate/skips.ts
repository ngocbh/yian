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
 * query. Past the bound each query falls back to the answer that is safe for its caller:
 * `mayAllSkip` says true (the output gets a null default it may not need) and
 * `maySkipWhileRunning` says false (nothing is reported). A level whose edges have a cycle has no
 * skippable nodes here; the cycle check reports the cycle.
 *
 * `nullSkippedOutputs` gives a flow output that reads only nodes that may all be skipped the
 * explicit null default, so it returns null instead of leaving END unreachable. Python's
 * `check_skipped_arithmetic` (arithmetic over a value that may be null from a skip) is not ported
 * here yet.
 *
 * Knows about: compiled edges, flow outputs, binding producers, and expression templates.
 * Never imports: the runtime.
 */

import { type Edge, FlowOutput } from "../../compile/model.ts";
import { current } from "../../defaults.ts";
import type { OriginRef, SourceRange } from "../../diagnostics.ts";
import { bindingCoSkips } from "../../expr/index.ts";
import { mapSourceStrings } from "../../nodes/contracts.ts";
import { bindingProducers } from "../edges/producers.ts";
import { directReferencePath } from "../helpers/output_contracts.ts";

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

/** One pending search state: the next node index, the skipped nodes so far, and CASE choices. */
type SearchState = readonly [index: number, done: ReadonlySet<string>, choices: ReadonlyMap<string, string | null>];

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
  private readonly skipTargets = new Map<string, Set<string>>();
  private readonly order: string[] | null;

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
    // A CASE's choice matters only to the nodes its control edges reach; its data and ordering
    // edges are taken on every route.
    for (const caseId of this.choices.keys()) this.skipTargets.set(caseId, new Set());
    for (const node of this.nodeIds) {
      for (const [caseId] of this.joins.get(node)!.control) this.skipTargets.get(caseId)?.add(node);
    }
    this.order = this.topologicalOrder(graphParents);
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

  /** Return whether `targets` is a non-empty set of the level's nodes and the level is acyclic. */
  private queryable(targets: ReadonlySet<string>): boolean {
    return targets.size > 0 && [...targets].every((target) => this.known.has(target)) && this.order !== null;
  }

  /**
   * Return the level's nodes in topological order, or `null` when the edges cycle.
   *
   * @param parents Node id to the level's nodes with an edge into it.
   * @returns The nodes, each after all of its parents, or `null` for a cyclic level.
   */
  private topologicalOrder(parents: ReadonlyMap<string, ReadonlySet<string>>): string[] | null {
    const indegree = new Map(this.nodeIds.map((node) => [node, parents.get(node)!.size]));
    const children = new Map<string, string[]>(this.nodeIds.map((node) => [node, []]));
    for (const [node, sources] of parents) {
      for (const parent of sources) children.get(parent)!.push(node);
    }
    const ready = this.nodeIds.filter((node) => indegree.get(node) === 0);
    const order: string[] = [];
    // `ready` is a FIFO queue: `head` is its front.
    for (let head = 0; head < ready.length; head++) {
      const node = ready[head]!;
      order.push(node);
      for (const child of children.get(node)!) {
        const left = indegree.get(child)! - 1;
        indegree.set(child, left);
        if (left === 0) ready.push(child);
      }
    }
    return order.length === this.nodeIds.length ? order : null;
  }

  /**
   * Depth-first search for a CASE assignment that skips `targets` and runs `consumer`.
   *
   * @param targets Node ids in the level that must all be skipped.
   * @param consumer A node id in the level that must run, or `null` for no such node.
   * @param onLimit The answer when the search examines `limit` assignments without finding one.
   * @returns Whether such an assignment exists, or `onLimit` past the bound.
   */
  private search(targets: ReadonlySet<string>, consumer: string | null, onLimit: boolean): boolean {
    const queried = new Set(targets);
    if (consumer !== null) queried.add(consumer);
    const closure = new Set(queried);
    const pending = [...queried];
    while (pending.length > 0) {
      for (const parent of this.parents.get(pending.pop()!)!) {
        if (!closure.has(parent)) {
          closure.add(parent);
          pending.push(parent);
        }
      }
    }
    const nodes = this.order!.filter((node) => closure.has(node));
    const branching = new Set<string>();
    for (const [caseId, reached] of this.skipTargets) {
      if (closure.has(caseId) && [...reached].some((node) => closure.has(node))) branching.add(caseId);
    }
    const stack: SearchState[] = [[0, new Set(), new Map()]];
    let worlds = 0;
    while (stack.length > 0) {
      if (worlds >= this.limit) return onLimit;
      let [index, done, choices] = stack.pop()!;
      const skipped = new Set(done);
      let complete = true;
      for (const node of nodes.slice(index)) {
        index += 1;
        const dead = this.skips(node, skipped, choices);
        if (dead) skipped.add(node);
        if ((targets.has(node) && !dead) || (node === consumer && dead)) {
          worlds += 1;
          complete = false;
          break;
        }
        if (branching.has(node) && !dead) {
          const [first, ...rest] = this.choices.get(node)!;
          for (const choice of rest.reverse()) {
            stack.push([index, new Set(skipped), new Map([...choices, [node, choice]])]);
          }
          choices = new Map([...choices, [node, first!]]);
        }
      }
      if (complete) return true;
    }
    return false;
  }

  /**
   * Return whether `node` is skipped, given its skipped ancestors and CASE choices.
   *
   * @param node A node id in the level.
   * @param skipped The skipped nodes among the ones before `node` in topological order.
   * @param choices The branch each running CASE before `node` picked; `null` is no branch.
   * @returns Whether the join rule marks `node` dead.
   */
  private skips(node: string, skipped: ReadonlySet<string>, choices: ReadonlyMap<string, string | null>): boolean {
    const join = this.joins.get(node)!;
    if (
      join.control.length > 0 &&
      join.control.every(
        ([caseId, handle]) => skipped.has(caseId) || (this.choices.has(caseId) && choices.get(caseId) !== handle),
      )
    ) {
      return true;
    }
    // A data or ordering edge carries no handle, so the runtime takes it on every route of a CASE
    // that runs; it is skipped only when its producer is.
    for (const [optional, producers] of join.groups.values()) {
      if (!optional && producers.every((producer) => skipped.has(producer))) return true;
    }
    return join.requiredOrder.some((producer) => skipped.has(producer));
  }
}

/**
 * Give each flow output whose producers may all be skipped the explicit null default.
 *
 * @param flowOutputs The flow level's declared outputs.
 * @param model The skip model of the same level.
 * @returns The outputs in order. A co-skipping source `${E}` whose producers may all be skipped
 *   becomes `${E:-null}` when `E` is one reference, else `${(E) :- null}`; its origin and declared
 *   contract are kept. Each string nested in a value template is rewritten on its own the same
 *   way. Every other source is kept as is.
 *
 * @example
 * const model = new SkipModel(["c", "a"], [new Edge("t", "c", "a", { sourceHandle: "then" })], new Set());
 * nullSkippedOutputs([new FlowOutput("v", "${a.output}")], model)[0].binding.source; // "${a.output:-null}"
 */
export function nullSkippedOutputs(flowOutputs: readonly FlowOutput[], model: SkipModel): FlowOutput[] {
  if (!model.hasCases) return [...flowOutputs];

  const nullDefault = (text: string): string => {
    if (!(bindingCoSkips(text) && model.mayAllSkip(bindingProducers(text)))) return text;
    // A co-skipping source is exactly one whole `${...}` span.
    const interior = text.slice(2, -1);
    if (directReferencePath(text) !== null) return "${" + interior + ":-null}";
    return "${(" + interior + ") :- null}";
  };

  return flowOutputs.map(
    (output) => new FlowOutput(output.name, mapSourceStrings(output.binding, nullDefault)),
  );
}
