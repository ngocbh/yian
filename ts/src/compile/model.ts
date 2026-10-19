/**
 * The compiled flow graph: nodes, edges and adjacency.
 *
 * `CompiledFlow` carries authored topology: the nodes and how they connect. Mutable dispositions
 * live in the run state, and the run boundary copies the graph before a growth reducer splices or
 * prunes dynamic topology, so one authored flow can back many independent runs.
 *
 * Growth fragments also carry source-neutral seed connections and actual child boundary pairs.
 *
 * `START_ID` / `END_ID` are the reserved boundary-node ids. Every flow begins with a synthesized
 * START node (which binds the run inputs) and ends with a synthesized END node (whose value is the
 * flow's result). They are owned canonically by the node classes (`StartNode.ID` / `EndNode.ID`)
 * and re-exported here so runtime consumers can import them alongside the graph types.
 *
 * Knows about:   nodes, memory contracts, diagnostics and typesys.
 * Never imports: compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { reprText } from "../diagnostics.ts";
import { MemoryDecl } from "../memory/contracts.ts";
import type { Node } from "../nodes/base.ts";
import {
  asBinding,
  type Binding,
  type ChildFlowBoundary,
  isExactly,
  type SeedConnection,
  type ValuePath,
  type ValueTemplate,
} from "../nodes/contracts.ts";
import { EndNode } from "../nodes/end/node.ts";
import { StartNode } from "../nodes/start/node.ts";
import type { Type } from "../typesys/index.ts";
import { jsTypeName } from "../typesys/values.ts";

/** The synthesized START node id. Canonical source is the node class; re-exported here. */
export const START_ID = StartNode.ID;
/** The synthesized END node id. Canonical source is the node class; re-exported here. */
export const END_ID = EndNode.ID;

/**
 * Mark expected node-produced growth data rejected before durability.
 *
 * Compile and runtime throw this contract only around explicit growth-shape or growth-policy
 * rejection. Arbitrary builder, compiler and host errors stay outside it so the dispatcher can
 * preserve their identity.
 */
export class GrowthRejected extends Error {
  override name = "GrowthRejected";
  /** The exact typed rejection used for diagnostic data. */
  readonly original: Error;

  /**
   * Retain the exact growth rejection for semantic failure construction.
   *
   * @throws TypeError when `original` isn't an `Error`.
   */
  constructor(original: Error) {
    if (!(original instanceof Error)) throw new TypeError("GrowthRejected original must be an Error");
    super(String(original.message), { cause: original });
    this.original = original;
  }
}

/**
 * A directed edge `from -> to` in the compiled graph. Frozen.
 *
 * Most edges carry data (one producer output feeding one consumer input group); some carry only
 * branch-control or run-ordering and no data.
 */
export class Edge {
  /** Unique edge id within the flow. */
  readonly id: string;
  /** Producer node id. */
  readonly from: string;
  /** Consumer node id. */
  readonly to: string;
  /** Branch label on an edge leaving a CASE node; `null` otherwise. */
  readonly sourceHandle: string | null;
  /** The consumer input this data edge feeds; `null` for pure control/ordering edges. */
  readonly inputGroup: string | null;
  /**
   * When true, a skipped producer does not co-skip the consumer (a data group with a literal or
   * `:?` default, or an ordering edge).
   */
  readonly optional: boolean;
  /** When true, the edge carries no data: a `depends_on` / `runs_after` ordering edge. */
  readonly ordering: boolean;

  /** Build one edge; `sourceHandle` and `inputGroup` default to `null`, the flags to false. */
  constructor(
    id: string,
    from: string,
    to: string,
    options: {
      sourceHandle?: string | null;
      inputGroup?: string | null;
      optional?: boolean;
      ordering?: boolean;
    } = {},
  ) {
    const { sourceHandle = null, inputGroup = null, optional = false, ordering = false } = options;
    this.id = id;
    this.from = from;
    this.to = to;
    this.sourceHandle = sourceHandle;
    this.inputGroup = inputGroup;
    this.optional = optional;
    this.ordering = ordering;
    Object.freeze(this);
  }
}

/** One declared result name paired with its canonical semantic binding. Frozen. */
export class FlowOutput {
  /** Declared result name. */
  readonly name: string;
  /** Source or literal plus its optional authored origin. */
  readonly binding: Binding;

  /**
   * Pair `name` with `binding`; a raw `${...}` source or literal becomes an origin-free
   * `Binding`, and an existing `Binding` is kept.
   */
  constructor(name: string, binding: unknown) {
    this.name = name;
    this.binding = asBinding(binding);
    Object.freeze(this);
  }
}

/**
 * Copy flow-owned wiring into its canonical carrier representation.
 *
 * @param wiring Raw or already-normalized sources keyed first by node id and then parameter name;
 *   `null` is empty wiring.
 * @returns A fresh two-level table whose values are bindings.
 * @throws TypeError when a node id or parameter name isn't a string.
 */
function normalizeWiring(
  wiring: ReadonlyMap<string, ReadonlyMap<string, unknown>> | null,
): Map<string, ReadonlyMap<string, Binding>> {
  const normalized = new Map<string, ReadonlyMap<string, Binding>>();
  if (wiring === null) return normalized;
  for (const [nodeId, nodeWiring] of wiring) {
    if (typeof nodeId !== "string") {
      throw new TypeError(`flow wiring node id must be a string, got ${jsTypeName(nodeId)}`);
    }
    const normalizedNode = new Map<string, Binding>();
    for (const [parameter, source] of nodeWiring) {
      if (typeof parameter !== "string") {
        throw new TypeError(
          `flow wiring parameter name for node ${reprText(nodeId)} must be a string, ` +
            `got ${jsTypeName(parameter)}`,
        );
      }
      normalizedNode.set(parameter, asBinding(source));
    }
    normalized.set(nodeId, normalizedNode);
  }
  return normalized;
}

/** The wiring a flow constructor accepts: raw sources or bindings by node id, then parameter. */
type WiringInput = ReadonlyMap<string, ReadonlyMap<string, unknown>> | null;

/**
 * The shared topology core every flow carries: nodes, edges, wiring and its boundary ids.
 *
 * This is the one type that both the top-level compiled flow (`CompiledFlow`) and every spliced
 * subgraph share. A flow is a function: it has a single entry (`startId`) and a single result
 * node (`endId`), and its body is the `nodes`/`edges`/`wiring` triple. Spliced subgraphs pass
 * their namespaced boundary ids; the top-level case defaults to the reserved `START_ID`/`END_ID`.
 *
 * Topology only: a `Flow` holds no run state, so one instance can back many independent runs. A
 * compile-owned copy shares everything but its `nodes`, `edges` and `wiring` (and a compiled
 * flow's adjacency index), so change a copy by reassigning a field or through
 * `addSubgraph`/`removeSubgraph`, never by mutating a shared carrier.
 */
export class Flow {
  /** Node id to its `Node`, including the synthesized START/END boundary nodes. */
  nodes: Map<string, Node>;
  /** All directed edges: data edges and pure control/ordering edges alike. */
  edges: Edge[];
  /**
   * Authoritative input wiring `wiring.get(nodeId).get(param) -> Binding`. Each carrier owns a
   * `${...}` source or literal plus its optional origin; `edges` is the derived data-flow
   * projection.
   */
  wiring: Map<string, ReadonlyMap<string, Binding>>;
  /** The flow's single entry node id, which the engine seeds and advances at start. */
  readonly startId: string;
  /** The flow's result node id, whose value is the flow's output. */
  readonly endId: string;
  /**
   * Derived fragment-input bindings (child START inputs, or defaulted unwired inputs) selected
   * from a parent's materialized seed routes.
   */
  readonly inputConnections: readonly SeedConnection[];
  /** Actual authored child flow boundaries, excluding generated fan nodes. */
  readonly childBoundaries: readonly ChildFlowBoundary[];

  /**
   * Build a flow over `nodes` and `edges` (kept by identity) and a fresh normalized copy of
   * `wiring`. The boundary ids default to `START_ID`/`END_ID`.
   *
   * @throws TypeError when a wiring node id or parameter name isn't a string.
   */
  constructor(
    nodes: Map<string, Node>,
    edges: Edge[],
    options: {
      wiring?: WiringInput;
      startId?: string;
      endId?: string;
      inputConnections?: readonly SeedConnection[];
      childBoundaries?: readonly ChildFlowBoundary[];
    } = {},
  ) {
    const { wiring = null, startId = START_ID, endId = END_ID, inputConnections = [], childBoundaries = [] } =
      options;
    this.nodes = nodes;
    this.edges = edges;
    // Normalize once so every consumer below this construction boundary reads one shape.
    this.wiring = normalizeWiring(wiring);
    this.startId = startId;
    this.endId = endId;
    this.inputConnections = inputConnections;
    this.childBoundaries = childBoundaries;
  }
}

/** The options of `CompiledFlow`'s constructor and `fromParts`. */
type CompiledFlowOptions = {
  outputs?: readonly FlowOutput[] | null;
  wiring?: WiringInput;
  flowLlmConfig?: Readonly<Record<string, unknown>> | ValueTemplate | Binding | null;
  flowId?: string | null;
  memories?: readonly MemoryDecl[];
};

/** Append each of `edges` to its consumer's `incoming` list and its producer's `outgoing` list. */
function indexEdges(edges: readonly Edge[], incoming: Map<string, Edge[]>, outgoing: Map<string, Edge[]>): void {
  for (const edge of edges) {
    const into = incoming.get(edge.to);
    if (into === undefined) incoming.set(edge.to, [edge]);
    else into.push(edge);
    const out = outgoing.get(edge.from);
    if (out === undefined) outgoing.set(edge.from, [edge]);
    else out.push(edge);
  }
}

/**
 * The compiled graph: nodes, edges, adjacency and flow-owned input wiring.
 *
 * It holds no dispositions or variable values. The run boundary copies it for each execution;
 * only that private copy is changed when the reducer applies a prepared expansion or prunes a
 * retired loop iteration. The authored instance therefore remains reusable across independent
 * runs. The top-level compiled flow always roots at the synthesized START and returns END.
 */
export class CompiledFlow extends Flow {
  /**
   * The declared flow `output:` carriers END reads; empty for a flow without `output:` (END
   * returns `null`) or with `output: {}` (END returns `{}`).
   */
  outputs: readonly FlowOutput[];
  /**
   * The flow's authored `llm_config:` model-selection defaults (the cascade's flow layer); `{}`
   * when absent. A layer holding `${...}` values is kept as written: a `ValueTemplate`, or a
   * `Binding` for a whole record written as one value.
   */
  flowLlmConfig: Readonly<Record<string, unknown>> | ValueTemplate | Binding;
  /**
   * The static type Compose proved for each value in `flowLlmConfig`, as `[path, type]` entries;
   * `null` when unproven. Derived, so not part of the fingerprint; empty for a hand-built flow,
   * whose values answer from their tags.
   */
  flowLlmValueTypes: readonly (readonly [ValuePath, Type | null])[];
  /** Stable authored semantic identity; direct memoryless graphs may omit it. */
  readonly flowId: string | null;
  /** Flow-owned memory declarations in authored order. Runtime layout is derived later. */
  memories: readonly MemoryDecl[];
  /** Node id to the edges ending at it, in edge order. The runtime's reducer updates it. */
  incomingEdges: Map<string, Edge[]>;
  /** Node id to the edges leaving it, in edge order. The runtime's reducer updates it. */
  outgoingEdges: Map<string, Edge[]>;

  /**
   * Build a compiled flow rooted at `START_ID` and returning `END_ID`, and index its edges.
   *
   * @throws TypeError for a non-string wiring key, a `flowId` that isn't an exact string or
   *   `null`, or `memories` that isn't an array of exact `MemoryDecl` values.
   * @throws RangeError for an empty `flowId`, or a memory declared on a node site.
   */
  constructor(nodes: Map<string, Node>, edges: Edge[], options: CompiledFlowOptions = {}) {
    const { outputs = null, wiring = null, flowLlmConfig = null, flowId = null, memories = [] } = options;
    super(nodes, edges, { wiring, startId: START_ID, endId: END_ID });
    this.outputs = outputs ?? [];
    this.flowLlmConfig = flowLlmConfig ?? {};
    this.flowLlmValueTypes = [];
    if (flowId !== null) {
      if (typeof flowId !== "string") throw new TypeError("flow_id must be an exact string or null");
      if (!flowId) throw new RangeError("flow_id must be non-empty");
    }
    if (!Array.isArray(memories) || memories.some((decl) => !isExactly(decl, MemoryDecl))) {
      throw new TypeError("memories must be an array of exact MemoryDecl values");
    }
    if (memories.some((decl) => decl.site.node !== null)) {
      throw new RangeError("flow memories must carry flow DeclSites (node=null)");
    }
    this.flowId = flowId;
    this.memories = memories;
    this.incomingEdges = new Map();
    this.outgoingEdges = new Map();
    indexEdges(edges, this.incomingEdges, this.outgoingEdges);
  }

  /**
   * Build a compiled flow. The entry is always the synthesized START node (`StartNode.ID`), added
   * implicitly by the loader or graph builder and never authored; the engine seeds and advances
   * it at run start.
   */
  static fromParts(nodes: Map<string, Node>, edges: Edge[], options: CompiledFlowOptions = {}): CompiledFlow {
    return new CompiledFlow(nodes, edges, options);
  }

  /** The edges ending at `nodeId` (its inputs); empty when none. */
  incoming(nodeId: string): readonly Edge[] {
    return this.incomingEdges.get(nodeId) ?? [];
  }

  /** The edges leaving `nodeId` (its outputs); empty when none. */
  outgoing(nodeId: string): readonly Edge[] {
    return this.outgoingEdges.get(nodeId) ?? [];
  }

  /**
   * The return-boundary node id (`END_ID`) when the synthesized END node exists, which is every
   * loaded or built flow; `null` only for a hand-built graph that omits an END node.
   */
  get terminalId(): string | null {
    return this.nodes.has(END_ID) ? END_ID : null;
  }

  /**
   * Splice a deep-namespaced subgraph into one run's private topology.
   *
   * Called while the reducer applies a prepared expansion from CALL, MAP or LOOP. Adds the
   * fragment's nodes, edges and wiring and updates adjacency; the same reducer plan updates
   * dispositions and the ready frontier.
   *
   * @throws TypeError when a wiring node id or parameter name isn't a string, before any change.
   */
  addSubgraph(
    nodes: ReadonlyMap<string, Node>,
    edges: readonly Edge[],
    wiring: ReadonlyMap<string, ReadonlyMap<string, unknown>>,
  ): void {
    const normalizedWiring = normalizeWiring(wiring);
    for (const [id, node] of nodes) this.nodes.set(id, node);
    for (const edge of edges) this.edges.push(edge);
    for (const [id, nodeWiring] of normalizedWiring) this.wiring.set(id, nodeWiring);
    indexEdges(edges, this.incomingEdges, this.outgoingEdges);
  }

  /**
   * Prune a finished deep-namespaced fragment from one run's private topology.
   *
   * The growth reducer uses this inverse of `addSubgraph` for a committed loop iteration whose
   * ids are no longer referenced. It removes nodes, incident edges, wiring and adjacency, leaving
   * the flow as it would be had those nodes never been added: no surviving edge or adjacency
   * entry keys on or points at a removed id.
   */
  removeSubgraph(nodeIds: ReadonlySet<string>): void {
    for (const id of nodeIds) {
      this.nodes.delete(id);
      this.wiring.delete(id);
    }
    // Drop every edge incident to a removed node (as producer or consumer).
    this.edges = this.edges.filter((e) => !nodeIds.has(e.from) && !nodeIds.has(e.to));
    // Rebuild adjacency: drop keys for removed ids, and prune surviving lists of any edge that
    // touches a removed id.
    const outgoing = new Map<string, Edge[]>();
    for (const [id, list] of this.outgoingEdges) {
      if (!nodeIds.has(id)) outgoing.set(id, list.filter((e) => !nodeIds.has(e.to)));
    }
    const incoming = new Map<string, Edge[]>();
    for (const [id, list] of this.incomingEdges) {
      if (!nodeIds.has(id)) incoming.set(id, list.filter((e) => !nodeIds.has(e.from)));
    }
    this.outgoingEdges = outgoing;
    this.incomingEdges = incoming;
  }
}
