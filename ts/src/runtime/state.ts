/**
 * Authoritative mutable run state and append-before-apply transitions.
 *
 * Preparation validates and canonicalizes a transition without touching live state.
 * `commitPrepared` then makes its record durable before applying a frozen plan whose remaining
 * work is limited to total assignments and container operations. The dispatcher is the sole
 * caller that mutates this state while a run is live. The caller supplies a valid private graph
 * and relinquishes direct mutation of its execution fields; chronological reducer transitions are
 * the only topology writers. Everything here is synchronous.
 *
 * Knows about:   compile/model, defaults, diagnostics, events, memory, nodes, runtime/common,
 *                runtime/learning, runtime/log, runtime/trace, typesys.
 * Never imports: engine, run, backward, compose, or the CLI.
 */

import { CompiledFlow } from "../compile/model.ts";
import { BUILTIN, type Defaults } from "../defaults.ts";
import type { Failure } from "../diagnostics.ts";
import type { RunFailed, RunSucceeded } from "../events.ts";
import type { MemorySnapshotEnvelope, MemorySpec, MemoryStack } from "../memory/index.ts";
import { isExactly, type OutputContract } from "../nodes/contracts.ts";
import type { JsonObject } from "../typesys/json.ts";
import { VariablePool } from "../typesys/pool.ts";
import { isRecord } from "../typesys/values.ts";
import { NodeState } from "./common.ts";
import { RuntimeBackwardState } from "./learning.ts";
import type { RetainedTrace } from "./trace.ts";

/** @internal Immutable identity roots established by canonical RunStarted decoding. Frozen. */
export class MemoryAuthority {
  /** The recorded snapshots, or `null` when memory is disabled. */
  readonly snapshots: readonly MemorySnapshotEnvelope[] | null;
  /** Whether the run records learning evidence. */
  readonly learnable: boolean;
  /** The recorded memory key table, or `null` when memory is disabled. */
  readonly memories: readonly MemorySpec[] | null;
  /** The recorded key stacks, or `null` when memory is disabled. */
  readonly memoryStacks: readonly MemoryStack[] | null;
  /** The backward state's own memory order array. */
  readonly memoryOrder: readonly string[];

  /** Hold the roots by reference. */
  constructor(
    snapshots: readonly MemorySnapshotEnvelope[] | null,
    learnable: boolean,
    memories: readonly MemorySpec[] | null,
    memoryStacks: readonly MemoryStack[] | null,
    memoryOrder: readonly string[],
  ) {
    this.snapshots = snapshots;
    this.learnable = learnable;
    this.memories = memories;
    this.memoryStacks = memoryStacks;
    this.memoryOrder = memoryOrder;
    Object.freeze(this);
  }
}

/**
 * Reverse lookups over the live graph that growth checks read instead of scanning.
 *
 * The first growth builds them from the live state. Each growth commit then keeps them current,
 * since growth is the only transition that changes the topology or growth ownership.
 */
export interface GrowthIndex {
  /** Live node id to the node ids it names through its wiring, `commitAs`, `originId` or skipped output target. */
  names: Map<string, readonly string[]>;
  /** Node id to the live nodes that name it, in node order; the inverse of `names`. */
  referrers: Map<string, Set<string>>;
  /** Owner id to the live nodes its accepted growth owns; the inverse of `RunState.growthOwner`. */
  grown: Map<string, Set<string>>;
}

/**
 * All mutable reducer-owned state for one private compiled graph.
 *
 * The supplied graph is already the caller's per-run copy and is adopted rather than copied
 * again. The caller must not mutate that graph's nodes, wiring, edges, boundaries or adjacency
 * directly after adoption; reducer transitions own later topology changes. Initialization starts
 * with no durable sequence applied; `prepareRunStarted` builds the first transition, which
 * installs the canonical pool, identity, configuration and the sole ready START node.
 */
export class RunState {
  /** The private static graph plus any later chronological expansion overlay. */
  flow: CompiledFlow;
  /**
   * The caller-supplied contract the terminal output is read through to restore its files,
   * typically the flow's inferred output contract. Not recorded; `null` returns the output as
   * stored.
   */
  resultContract: OutputContract | null;
  /** Canonical typed values visible to node evaluation. */
  pool: VariablePool;
  /** The disposition of every live node, initially all `UNKNOWN`. */
  nodeState: Map<string, NodeState>;
  /** The disposition of every live edge, initially all `UNKNOWN`. */
  edgeState: Map<string, NodeState>;
  /** The dispatcher-owned FIFO of nodes eligible to start. */
  ready: string[] = [];
  /** Durably started nodes without a durable outcome, in start order. */
  inFlight = new Set<string>();
  /** The original accepted start sequence of each unfinished invocation. */
  acceptedStarts = new Map<string, number>();
  /**
   * The number of accepted `NodeStarted` records of each unsettled node, counting starts before
   * and after a resume; a node's entry goes when its outcome is applied.
   */
  nodeStarts = new Map<string, number>();
  /**
   * The canonical steps each unsettled node recorded, in record order. They survive a resume and
   * go when the node's outcome is applied.
   */
  nodeSteps = new Map<string, readonly JsonObject[]>();
  /** The derived terminal result, or `null` while the run is live. */
  result: RunSucceeded | RunFailed | null = null;
  /**
   * The first route-derived proof that the terminal output is unreachable, kept independently of
   * later frontier and topology changes.
   */
  terminalUnreachableFailure: Failure | null = null;
  /** Graph-growth nesting depth by live dynamic node id. */
  depth = new Map<string, number>();
  /**
   * Live dynamic node id to the live node whose accepted growth owns it. Pruning a node also
   * retires the nodes it owns, transitively, once it and all of them are settled. A self-pruning
   * LOOP driver's unsettled nodes move to its nearest surviving owner.
   */
  growthOwner = new Map<string, string>();
  /**
   * Live node ids a growth asked to prune while they or their grown nodes were unfinished; each
   * later growth retires those that have settled.
   */
  deferredPrune: ReadonlySet<string> = new Set();
  /** Per-run wiring by node id, beginning with START's narrowed inputs. */
  wiringOverrides = new Map<string, JsonObject>();
  /** The durable identity once `RunStarted` is applied. */
  runId: string | null = null;
  /** The stable authored-flow digest restored from `RunStarted`. */
  flowFingerprint: string | null = null;
  /** The sequence expected for the next authoritative record. */
  nextSequence = 0;
  /** The active dispatcher worker count; zero selects serial execution. */
  numWorkers = 0;
  /** The canonical caller-supplied START priming record, kept across START's commit. */
  input: JsonObject = {};
  /** Run-resolved agent configuration by stable static node path. */
  effectiveLlmByStaticPath: Record<string, JsonObject> = {};
  /**
   * The recorded agents whose configuration resolves only when they run, with the selection they
   * bind from (`RunStartedRecord.callTimeLlm`); `null` when there are none.
   */
  callTimeLlm: JsonObject | null = null;
  /** The canonical recorded snapshots, one per memory key. */
  memorySnapshots: readonly MemorySnapshotEnvelope[] | null = null;
  /** The recorded memory key table. */
  memories: readonly MemorySpec[] | null = null;
  /** The recorded key stacks. */
  memoryStacks: readonly MemoryStack[] | null = null;
  /** Whether accepted eligible outcomes must project durable learning evidence. */
  learnable = false;
  /**
   * The `yian.defaults` values `RunStarted` recorded, the built-in values overlaid with the
   * recorded ones; the reducer reads its limits here, never from the process.
   */
  runDefaults: Defaults = BUILTIN;
  /**
   * Accepted executions, immutable producer entries and private backward handlers, enabled by
   * recorded learning independently of memory scopes.
   */
  trace: RetainedTrace | null = null;
  /** Immutable accepted backward attempts and irreversible commit facts. */
  backwardState = new RuntimeBackwardState();
  /** @internal The memory identity roots `RunStarted` recorded. */
  memoryAuthority: MemoryAuthority;
  /**
   * The node ids adopted before any runtime expansion. Safe pruning may retire only later
   * dynamic-overlay nodes.
   */
  initialStaticIds: ReadonlySet<string>;
  /**
   * Reverse lookups derived from the live graph, built by the first growth and kept current by
   * growth commits, so a growth looks up the nodes it affects instead of scanning every node.
   */
  growthIndex: GrowthIndex | null = null;

  /**
   * Adopt `flow` and seed empty reducer structures for its current topology.
   *
   * @param flow The caller's private per-run graph.
   * @param options.resultContract The contract the terminal output is read through to restore
   *   its files; `null` (the default) returns the output as stored.
   * @throws TypeError when `flow` isn't a `CompiledFlow`.
   */
  constructor(flow: CompiledFlow, options: { resultContract?: OutputContract | null } = {}) {
    if (!(flow instanceof CompiledFlow)) throw new TypeError("RunState expects a CompiledFlow");
    this.flow = flow;
    this.resultContract = options.resultContract ?? null;
    this.pool = new VariablePool(flow.startId);
    this.nodeState = new Map([...flow.nodes.keys()].map((id) => [id, NodeState.UNKNOWN]));
    this.edgeState = new Map(flow.edges.map((edge) => [edge.id, NodeState.UNKNOWN]));
    this.memoryAuthority = new MemoryAuthority(null, false, null, null, this.backwardState.memoryOrder);
    this.initialStaticIds = new Set(flow.nodes.keys());
  }
}

/** Whether two sets of ids hold the same ids, in any order. */
function sameIds(a: ReadonlySet<string>, b: Iterable<string>): boolean {
  const other = new Set(b);
  return a.size === other.size && [...other].every((id) => a.has(id));
}

/** Reject any state that no longer has its constructor-created empty frontier. */
function requirePristine(state: RunState): void {
  const exactContainers =
    isExactly(state.pool, VariablePool) &&
    isExactly(state.nodeState, Map) &&
    isExactly(state.edgeState, Map) &&
    Array.isArray(state.ready) &&
    isExactly(state.inFlight, Set) &&
    isExactly(state.acceptedStarts, Map) &&
    isExactly(state.nodeStarts, Map) &&
    isExactly(state.nodeSteps, Map) &&
    isExactly(state.depth, Map) &&
    isExactly(state.growthOwner, Map) &&
    isExactly(state.deferredPrune, Set) &&
    isExactly(state.wiringOverrides, Map) &&
    isRecord(state.input) &&
    isRecord(state.effectiveLlmByStaticPath) &&
    isExactly(state.backwardState, RuntimeBackwardState) &&
    isExactly(state.memoryAuthority, MemoryAuthority) &&
    isExactly(state.initialStaticIds, Set);
  if (!exactContainers) throw new RangeError("RunStarted requires pristine reducer container types");
  const flow = state.flow;
  const authority = state.memoryAuthority;
  const pristine =
    state.pool.startId === flow.startId &&
    state.pool.store.size === 0 &&
    state.pool.system.size === 0 &&
    sameIds(new Set(state.nodeState.keys()), flow.nodes.keys()) &&
    [...state.nodeState.values()].every((value) => value === NodeState.UNKNOWN) &&
    sameIds(new Set(state.edgeState.keys()), flow.edges.map((edge) => edge.id)) &&
    [...state.edgeState.values()].every((value) => value === NodeState.UNKNOWN) &&
    state.ready.length === 0 &&
    state.inFlight.size === 0 &&
    state.acceptedStarts.size === 0 &&
    state.nodeStarts.size === 0 &&
    state.nodeSteps.size === 0 &&
    state.result === null &&
    state.terminalUnreachableFailure === null &&
    state.depth.size === 0 &&
    state.growthOwner.size === 0 &&
    state.deferredPrune.size === 0 &&
    state.wiringOverrides.size === 0 &&
    state.runId === null &&
    state.flowFingerprint === null &&
    state.nextSequence === 0 &&
    state.numWorkers === 0 &&
    Object.keys(state.input).length === 0 &&
    Object.keys(state.effectiveLlmByStaticPath).length === 0 &&
    state.callTimeLlm === null &&
    state.memorySnapshots === null &&
    state.memories === null &&
    state.memoryStacks === null &&
    state.learnable === false &&
    state.trace === null &&
    // An empty backward state: no attempt and no memory order.
    state.backwardState.attempts.length === 0 &&
    state.backwardState.memoryOrder.length === 0 &&
    authority.snapshots === null &&
    authority.memories === null &&
    authority.memoryStacks === null &&
    authority.learnable === false &&
    authority.memoryOrder === state.backwardState.memoryOrder &&
    sameIds(state.initialStaticIds, flow.nodes.keys());
  if (!pristine) throw new RangeError("RunStarted requires a pristine RunState at sequence 0");
}
