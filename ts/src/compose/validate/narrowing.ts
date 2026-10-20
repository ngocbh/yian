/**
 * Union narrowing: what each node may assume about union values, from the CASEs that route it.
 *
 * A narrowing fact says which part of a union a value at a base path can be. A base path is
 * `<node>.output[.field...]` or `input.<name>[.field...]`. Facts come from CASE branches: an
 * `on:` CASE over a tag field of a union gives each `when:` edge the member with that tag and the
 * `else:` edge the unlisted members. A searched CASE gives each case's edge what its `when:`
 * proves true (`isinstance` checks, tag comparisons and null checks through `and`, `or` and
 * `not`); its `else:` edge carries none.
 *
 * Facts flow along the graph in topological order. A node's control edges join by union (it ran
 * because one of them fired), each required data group joins its producers by union, and every
 * contribution (control, required data groups, `depends_on`) meets by intersection. Each node
 * then gets a view of the producer and flow-input contracts with its narrowed bases rebuilt, so
 * the reference and type checks can read member fields inside a branch. A searched CASE's own
 * param whose every read is guarded by the checks before it in its `when:` gets a view narrowed
 * by what they prove too; a param read only after a check that can never be true is not checked,
 * since `checkImpossibleWhens` reports that check.
 *
 * The pass reads built, desugared CASE descriptors and their edges, so it runs after
 * `desugarCaseCallTargets` and the node builds.
 *
 * The fact algebra (`Members`, `Exact`, their union and intersection) lives in
 * `helpers/narrowing_facts.ts`; this module re-exports it.
 *
 * Ported so far: `Narrowing`, `Views` and the re-exports; `computeNarrowing`,
 * `checkImpossibleWhens` and `narrowContractAt` are not ported yet (T3-20).
 *
 * Knows about: descriptors, compiled edges, output contracts, condition guards, narrowing facts,
 * and `typesys` unions.
 * Never imports: runtime.
 */

import type { SourceRange } from "../../diagnostics.ts";
import type { OutputContract, ValueContract } from "../../nodes/contracts.ts";
import type { Env } from "../helpers/narrowing_facts.ts";

export { Exact, intersectFacts, isinstanceFact, Members, unionFacts } from "../helpers/narrowing_facts.ts";
export type { Env, Fact } from "../helpers/narrowing_facts.ts";

/** The producer contracts and the flow-input contracts one node or param reads against. */
export type Views = readonly [
  producers: ReadonlyMap<string, OutputContract>,
  inputs: ReadonlyMap<string, ValueContract>,
];

/** Per-node narrowing results for one flow; frozen. */
export class Narrowing {
  /** The unnarrowed producer contracts. */
  readonly producers: ReadonlyMap<string, OutputContract>;
  /** The unnarrowed flow-input contracts. */
  readonly inputs: ReadonlyMap<string, ValueContract>;
  /** Each ordered node's facts, keyed by base path. */
  readonly environments: ReadonlyMap<string, Env>;
  /** The narrowed contract maps of nodes that narrow something. */
  readonly views: ReadonlyMap<string, Views>;
  /**
   * Per searched CASE, the narrowed contract maps of its params whose every read is guarded by
   * checks before it in its `when:`.
   */
  readonly paramViews: ReadonlyMap<string, ReadonlyMap<string, Views>>;
  /**
   * Per searched CASE, the label and range to report a param at when its first read is guarded
   * and a later one isn't: the first unguarded read's `when:`.
   */
  readonly paramSites: ReadonlyMap<string, ReadonlyMap<string, readonly [label: string, range: SourceRange | null]>>;
  /** Per searched CASE, the facts each param in `paramViews` is read under. */
  readonly paramEnvs: ReadonlyMap<string, ReadonlyMap<string, Env>>;
  /**
   * Per searched CASE, the bases of each param's `paramEnvs` that only the routing narrows, not
   * the checks before the read.
   */
  readonly paramRouted: ReadonlyMap<string, ReadonlyMap<string, ReadonlySet<string>>>;
  /**
   * Per searched CASE, the params whose every read follows a check `checkImpossibleWhens`
   * reports, so no value reaches them.
   */
  readonly uncheckedParams: ReadonlyMap<string, ReadonlySet<string>>;
  /** Nodes a topological order could not reach, on or after a cycle. They get no facts. */
  readonly unordered: ReadonlySet<string>;

  /** Build the results; every map and set left out is empty. */
  constructor(options: {
    producers: ReadonlyMap<string, OutputContract>;
    inputs: ReadonlyMap<string, ValueContract>;
    environments?: ReadonlyMap<string, Env>;
    views?: ReadonlyMap<string, Views>;
    paramViews?: ReadonlyMap<string, ReadonlyMap<string, Views>>;
    paramSites?: ReadonlyMap<string, ReadonlyMap<string, readonly [label: string, range: SourceRange | null]>>;
    paramEnvs?: ReadonlyMap<string, ReadonlyMap<string, Env>>;
    paramRouted?: ReadonlyMap<string, ReadonlyMap<string, ReadonlySet<string>>>;
    uncheckedParams?: ReadonlyMap<string, ReadonlySet<string>>;
    unordered?: ReadonlySet<string>;
  }) {
    this.producers = options.producers;
    this.inputs = options.inputs;
    this.environments = options.environments ?? new Map();
    this.views = options.views ?? new Map();
    this.paramViews = options.paramViews ?? new Map();
    this.paramSites = options.paramSites ?? new Map();
    this.paramEnvs = options.paramEnvs ?? new Map();
    this.paramRouted = options.paramRouted ?? new Map();
    this.uncheckedParams = options.uncheckedParams ?? new Map();
    this.unordered = options.unordered ?? new Set();
    Object.freeze(this);
  }

  /**
   * Return the producer and flow-input contracts node `nodeId` sees.
   *
   * @param nodeId A node id. An unknown or unnarrowed node gets the unchanged maps.
   * @returns The producer contracts and the flow-input contracts.
   */
  forNode(nodeId: string): Views {
    return this.views.get(nodeId) ?? [this.producers, this.inputs];
  }

  /**
   * Return the producer and flow-input contracts param `param` of `nodeId` reads against.
   *
   * @param nodeId A node id.
   * @param param One of the node's params. A searched CASE's guarded param gets its guarded view;
   *   any other param gets the node's view.
   * @returns The producer contracts and the flow-input contracts.
   */
  forParam(nodeId: string, param: string): Views {
    return this.paramViews.get(nodeId)?.get(param) ?? this.forNode(nodeId);
  }

  /**
   * Return the label and range to report param `param` of `nodeId` at, if moved.
   *
   * @param nodeId A node id.
   * @param param One of the node's params.
   * @returns The `cases[N].when` label and range of the param's first unguarded read when its
   *   first read is guarded and a later one isn't; otherwise `null`, and the param keeps the site
   *   of its first read.
   */
  paramSite(nodeId: string, param: string): readonly [label: string, range: SourceRange | null] | null {
    return this.paramSites.get(nodeId)?.get(param) ?? null;
  }

  /**
   * Return the facts a searched CASE's guarded param is read under; an empty map otherwise.
   *
   * @param nodeId A node id.
   * @param param One of the node's params.
   * @returns The facts behind `forParam`'s guarded view, by base path.
   */
  paramEnv(nodeId: string, param: string): Env {
    return this.paramEnvs.get(nodeId)?.get(param) ?? new Map();
  }

  /**
   * Return the bases of `paramEnv` that only the routing narrows.
   *
   * @param nodeId A node id.
   * @param param One of the node's params.
   * @returns The base paths whose facts come from the case branch that routes the node, not from
   *   the checks before the read.
   */
  paramRoutedBases(nodeId: string, param: string): ReadonlySet<string> {
    return this.paramRouted.get(nodeId)?.get(param) ?? new Set();
  }

  /**
   * Return whether no value reaches any read of a searched CASE's param.
   *
   * @param nodeId A node id.
   * @param param One of the node's params.
   * @returns `true` when every read of the param follows a check that can never be true, which
   *   `checkImpossibleWhens` reports; its fields are then not checked.
   */
  paramUnchecked(nodeId: string, param: string): boolean {
    return this.uncheckedParams.get(nodeId)?.has(param) ?? false;
  }

  /** Return node `nodeId`'s facts by base path; an empty map when it has none. */
  environment(nodeId: string): Env {
    return this.environments.get(nodeId) ?? new Map();
  }
}
