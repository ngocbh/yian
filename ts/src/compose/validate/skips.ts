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
 * explicit null default, so it returns null instead of leaving END unreachable.
 * `checkSkippedArithmetic` rejects arithmetic over a value that may be null because its producers
 * were skipped while the reading node (or END) still runs.
 *
 * Knows about: compiled edges, flow outputs, node wiring, binding producers, and expression trees.
 * Never imports: the runtime.
 */

import { type Edge, FlowOutput } from "../../compile/model.ts";
import { current } from "../../defaults.ts";
import { compareCodeUnits, type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import {
  bindingCoSkips,
  ExpressionError,
  exprRefs,
  optionalAfterRoots,
  parseCondition,
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
import { LoadDiagnostics } from "../helpers/errors.ts";
import { directReferenceParts, directReferencePath } from "../../compile/output_contracts.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import { locatedError, type ResolveOrigin } from "./typed_bindings.ts";

// The grammar's arithmetic trees: the `+ -`, `* / %` and `**` chains and unary minus.
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
 *   becomes `${E:-null}` when `E` is one reference, else `${(E) :- null}`, with `?.` on the first
 *   step after each producer root; its origin and declared contract are kept. Each string nested in a value template is rewritten on its own the same
 *   way. Every other source is kept as is.
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
 * @param options.valueMode Whether the flow has a single `output:` value, named `output` in errors
 *   (default `false`).
 * @param options.outputsRange The authored `output:` range, the fallback location of an output
 *   error.
 * @param options.resolveOrigin Resolves a binding's registered origin to its source frame, the
 *   preferred location.
 * @param options.diagnostics Factory for located load errors.
 * @param options.searchedWhens Each searched CASE's id mapped to one `[case index, authored when,
 *   origin, range]` per case: the origin is the preferred location, the range the fallback. A
 *   `when:` is checked like a binding, without following `and`/`or`: arithmetic behind a null guard
 *   is reported too.
 * @throws LoadError at the first such binding or `when:`, naming it, the operand and its producers,
 *   and asking for a `:-` default.
 */
export function checkSkippedArithmetic(
  flowWiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  flowOutputs: readonly FlowOutput[],
  model: SkipModel,
  buildResults: ReadonlyMap<string, BuildResult>,
  options: {
    valueMode?: boolean;
    outputsRange?: SourceRange | null;
    resolveOrigin?: ResolveOrigin | null;
    diagnostics?: LoadDiagnostics | null;
    searchedWhens?: ReadonlyMap<string, readonly SearchedWhen[]> | null;
  } = {},
): void {
  if (!model.hasCases) return;
  const { valueMode = false, outputsRange = null, resolveOrigin = null, searchedWhens = null } = options;
  const diagnostics = options.diagnostics ?? new LoadDiagnostics();
  const known = model.nodes;
  for (const [nodeId, wiring] of flowWiring) {
    const result = buildResults.get(nodeId) ?? null;
    const reserved = result !== null ? result.node.reservedWiringKeys() : new Set<string>();
    for (const [param, binding] of wiring) {
      const found = skippedOperand(binding, model, nodeId, known);
      if (found === null) continue;
      const [site, operand, producers] = found;
      let where = `node ${reprText(nodeId)} input ${reprText(param)}`;
      let fallback: SourceRange | null = null;
      if (result !== null) {
        let label: string;
        let own: SourceRange | undefined;
        if (reserved.has(param)) {
          // A reserved key (`over`, `until`) is a kind field, not an input.
          label = result.node.reservedKeyLabel(param);
          own = result.fieldRanges.get(siteKey("field", param));
        } else {
          label = result.node.wiringRefLabel(param);
          own = result.fieldRanges.get(siteKey("input", param));
        }
        fallback = own ?? result.fieldRanges.get(siteKey("field", "inputs")) ?? result.nodeRange;
        // A synthetic param's label may already end in a colon (`on:`); the message adds one.
        where = result.siteName(nodeId, label.replace(/:+$/, ""));
      }
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
      if (typeof when !== "string") continue;
      let tree: ExprNode;
      try {
        tree = parseCondition(when);
      } catch (error) {
        // The CASE build reports a malformed condition.
        if (error instanceof ExpressionError) continue;
        throw error;
      }
      const found = skippedOperandIn(tree, model, nodeId, known);
      if (found === null) continue;
      throw locatedError(
        `node ${reprText(nodeId)} cases[${index}].when: ${arithmeticText(when.trim(), ...found)}`,
        new Binding(when, { origin }),
        whenRange,
        resolveOrigin,
        diagnostics,
      );
    }
  }
  for (const output of flowOutputs) {
    const found = skippedOperand(output.binding, model, null, known);
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
 * @returns The string that does the arithmetic (as a binding located at its own site), the
 *   operand, and the producers whose skip makes it null (in source order), or `null` when no
 *   operand may be null while `consumer` runs.
 */
function skippedOperand(
  binding: unknown,
  model: SkipModel,
  consumer: string | null,
  known: ReadonlySet<string>,
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
      const found = skippedOperandIn(segment.tree, model, consumer, known);
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
 * @returns The operand and the producers whose skip makes it null (in source order), or `null`
 *   when no operand may be null while `consumer` runs.
 */
function skippedOperandIn(
  expression: ExprNode,
  model: SkipModel,
  consumer: string | null,
  known: ReadonlySet<string>,
): SkippedOperand | null {
  // A lone token has no operator.
  if (!(expression instanceof Tree)) return null;
  // Matched by node, not by position: an operand that is no plain read has no guard here.
  const reads = guardedReads(expression);
  for (const tree of iterSubtreesTopdown(expression)) {
    if (!ARITHMETIC.has(tree.data)) continue;
    for (const operand of tree.children) {
      const guard = reads.find((read) => read.node === operand)?.guard ?? NOTHING;
      if (operand instanceof Tree && provesNotNull(guard, referencePath(operand))) continue;
      for (const group of nullGroups(operand, known)) {
        if (group.size > 0 && model.maySkipWhileRunning(group, consumer)) {
          const producers = new Set(exprRefs(operand).map(refProducer));
          return [operand, [...producers].filter((producer) => producer !== null && group.has(producer)) as string[]];
        }
      }
    }
  }
  return null;
}

/**
 * Return whether `guard` proves the root of the read `path` not null: a `!= null` check on it, or
 * an `isinstance` of it to a type that excludes null, holds whenever the read runs.
 */
function provesNotNull(guard: Guard, path: string): boolean {
  const names = path.split(".");
  const root = names.slice(0, poolRootLength(names)).join(".");
  return conjuncts(guard).some(
    (part) =>
      part instanceof Holds &&
      part.value &&
      part.check.base === root &&
      (part.check instanceof NotNullCheck || (part.check instanceof IsinstanceCheck && !admitsNull(part.check.typeName))),
  );
}

/** Whether an `isinstance` type name admits null. */
function admitsNull(typeName: string): boolean {
  return typeName === "None" || typeName.startsWith("Optional(");
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
      // Keep the first of equal sets, in order, as Python's `dict.fromkeys` over frozensets does.
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
  const parts = directReferenceParts(operand);
  if (parts !== null) return referenceText(operand as Tree);
  const joiner = operand.data === "coalesce" ? " | " : " :- ";
  return "(" + operand.children.map(operandText).join(joiner) + ")";
}
