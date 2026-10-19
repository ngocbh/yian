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

import { CompiledFlow, type Edge } from "../compile/model.ts";
import { BUILTIN, type Defaults } from "../defaults.ts";
import { type Failure, reprText } from "../diagnostics.ts";
import { RunStarted, type RunEvent, type RunFailed, type RunSucceeded } from "../events.ts";
import type { MemorySnapshotEnvelope, MemorySpec, MemoryStack } from "../memory/index.ts";
import type { Node } from "../nodes/base.ts";
import { type Binding, isExactly, type OutputContract } from "../nodes/contracts.ts";
import type { JsonObject } from "../typesys/json.ts";
import { VariablePool } from "../typesys/pool.ts";
import { isRecord, type Type } from "../typesys/values.ts";
import { NodeState } from "./common.ts";
import { RuntimeBackwardState } from "./learning.ts";
import {
  decodePool,
  decodeRecord,
  encodePool,
  encodeRecord,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeSucceededRecord,
  RecordEnvelope,
  type RunRecord,
  RunStartedRecord,
  SCHEMA_VERSION,
} from "./log.ts";
import { type PrepareTraceOptions, prepareTrace, type RetainedTrace, type TraceDelta } from "./trace.ts";

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

/** The fully resolved inputs `prepareRunStarted` builds the first authoritative record from. */
export interface RunStartData {
  /** The non-empty identity the pool, the state and the enclosing log share. */
  runId: string;
  /** The authored flow's SHA-256 identity, 64 lowercase hex digits. */
  flowFingerprint: string;
  /** The complete pool, including the system values and the primed START record. */
  initialPool: VariablePool;
  /** START's narrowed per-run wiring. */
  startWiringOverride: Readonly<Record<string, unknown>>;
  /** The complete resolved agent configuration by stable static path. */
  effectiveLlmByStaticPath: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  /** The exact non-negative dispatcher worker count. */
  numWorkers: number;
  /** Canonical snapshots, one per memory key; `null` (the default) disables memory. */
  memorySnapshots?: readonly MemorySnapshotEnvelope[] | null;
  /** Whether eligible accepted outcomes must carry learning evidence; defaults to `false`. */
  learnable?: boolean;
  /** The sorted memory key table paired with `memorySnapshots`; defaults to `null`. */
  memories?: readonly MemorySpec[] | null;
  /** Each scoped target's key stack, sorted by locator; defaults to `null`. */
  memoryStacks?: readonly MemoryStack[] | null;
  /**
   * The `yian.defaults` values that differ from the built-in ones, by entry name; empty (the
   * default) means the run uses the built-in values.
   */
  defaults?: Readonly<Record<string, bigint>>;
  /**
   * The agents whose configuration resolves only when they run, as `RunStartedRecord.callTimeLlm`;
   * `null` (the default) when there are none.
   */
  callTimeLlm?: Readonly<Record<string, unknown>> | null;
}

/**
 * The authority every plan carries: the exact state and flow it was prepared against, the
 * canonical envelope text, the boundary nodes, and the trace change.
 */
interface PlanAnchors {
  readonly originatingState: RunState;
  readonly originatingFlow: CompiledFlow;
  readonly envelopeJson: string;
  readonly startId: string;
  readonly startNode: Node;
  readonly endId: string;
  readonly endNode: Node | null;
  readonly traceDelta: TraceDelta | null;
}

/** One initialization: the anchors alone, since the envelope text holds every payload. */
interface RunStartedPlan extends PlanAnchors {
  readonly kind: "runStarted";
}

/** Dispatching one ready head. */
interface NodeStartedDelta {
  readonly kind: "started";
  readonly nodeId: string;
}

/** Restoring one retry frontier. */
interface RunResumedDelta {
  readonly kind: "resumed";
  readonly retryNodeIds: readonly string[];
  readonly numWorkers: number;
}

/** Appending one canonical step to an in-flight node's steps. */
interface NodeStepDelta {
  readonly kind: "step";
  readonly nodeId: string;
  readonly step: Readonly<JsonObject>;
}

/** Recording a stop over the in-flight nodes; it changes nothing but the sequence. */
interface RunStoppedDelta {
  readonly kind: "stopped";
  readonly nodeIds: readonly string[];
}

/** One dispatch, resume, step or stop. */
interface FrontierPlan extends PlanAnchors {
  readonly kind: "frontier";
  readonly delta: NodeStartedDelta | RunResumedDelta | NodeStepDelta | RunStoppedDelta;
}

/** The touched coordinates of one successful output commit. */
interface SuccessDelta {
  readonly nodeId: string;
  readonly commitTarget: string;
  readonly valueJson: string;
  readonly outgoingEdgeIds: readonly string[];
  readonly newlyReadyIds: readonly string[];
  readonly observerCommitAs: string | null;
}

/**
 * The derived terminal attached to one causal outcome. `outputJson` is canonical typed log text
 * for success and `null` for failure; `succeeded` tells a typed null output from failure;
 * `failure` is the one source-neutral authority for an unreachable terminal. `files` is the type
 * the output is read through to restore its files, from `RunState.resultContract`.
 */
interface TerminalDelta {
  readonly succeeded: boolean;
  readonly outputJson: string | null;
  readonly failure: Failure | null;
  readonly files: Type | null;
}

/** One successful output; `files` is the type its commit target reads it through. */
interface SuccessPlan extends PlanAnchors {
  readonly kind: "success";
  readonly delta: SuccessDelta;
  readonly terminal: TerminalDelta | null;
  readonly files: Type | null;
}

/** The ordered operations of one route and its skip closure. */
interface RouteDelta {
  readonly nodeId: string;
  readonly handle: string;
  readonly edgeUpdates: readonly (readonly [string, NodeState])[];
  readonly skippedNodeIds: readonly string[];
  readonly newlyReadyIds: readonly string[];
  readonly terminalUnreachableFailure: Failure | null;
}

/** One route. */
interface RoutePlan extends PlanAnchors {
  readonly kind: "route";
  readonly delta: RouteDelta;
  readonly terminal: TerminalDelta | null;
}

/** One canonical failure. */
interface FailurePlan extends PlanAnchors {
  readonly kind: "failure";
  readonly failure: Failure;
}

/** The scalar authority of one flat deterministic expansion. */
interface GrowthDelta {
  readonly spawnerId: string;
  readonly pruneIds: readonly string[];
  readonly retiredIds: readonly string[];
  readonly topologyDigest: string;
  readonly fragmentStartId: string;
  readonly fragmentEndId: string;
  readonly newNodeIds: readonly string[];
  readonly newEdgeIds: readonly string[];
  readonly derivedDepth: number | null;
}

/**
 * A detached topology change: the fragment to add and what retires with it. Commit applies it in
 * place, so a growth costs the fragment plus its prunes rather than a copy of every container.
 */
interface GrowthTopology {
  readonly nodes: readonly (readonly [string, Node])[];
  readonly edges: readonly Edge[];
  readonly wiring: readonly (readonly [string, ReadonlyMap<string, Binding>])[];
  readonly retiredEdges: readonly Edge[];
  readonly depth: readonly (readonly [string, number])[];
  readonly growthOwner: readonly (readonly [string, string | null])[];
  readonly deferredPrune: ReadonlySet<string>;
  readonly names: readonly (readonly [string, readonly string[]])[];
}

/** One fully materialized expansion. */
interface GrowthPlan extends PlanAnchors {
  readonly kind: "growth";
  readonly delta: GrowthDelta;
  readonly topology: GrowthTopology;
}

/** One post-terminal backward or memory-commit record and the backward state it leads to. */
interface BackwardPlan extends PlanAnchors {
  readonly kind: "backward";
  readonly originatingBackwardState: RuntimeBackwardState;
  readonly nextBackwardState: RuntimeBackwardState;
}

/** Every plan kind a `PreparedTransition` carries. */
type PreparedPlan = RunStartedPlan | FrontierPlan | SuccessPlan | RoutePlan | FailurePlan | GrowthPlan | BackwardPlan;

/**
 * One canonical record plus its already validated total state-change plan. Frozen.
 *
 * `_plan` is internal: callers may inspect the durable record and the expected authority
 * coordinates, but the reducer's plan isn't a construction surface. The observers are returned
 * only after append and apply both succeed.
 */
export class PreparedTransition {
  /** The canonical record the log will retain. */
  readonly record: RunRecord;
  /** The run id the state and the log must have. */
  readonly expectedRunId: string;
  /** The sequence the state and the log must expect next. */
  readonly expectedSequence: number;
  /** @internal The frozen plan `commitPrepared` checks and applies. */
  readonly _plan: PreparedPlan;
  /** The events the transition publishes once committed. */
  readonly observers: readonly RunEvent[];

  /** Hold the fields as given, with a frozen copy of `observers`. */
  constructor(
    record: RunRecord,
    expectedRunId: string,
    expectedSequence: number,
    plan: PreparedPlan,
    observers: readonly RunEvent[],
  ) {
    this.record = record;
    this.expectedRunId = expectedRunId;
    this.expectedSequence = expectedSequence;
    this._plan = plan;
    this.observers = Object.freeze([...observers]);
    Object.freeze(this);
  }
}

/**
 * Prepare canonical initialization without mutating `state`.
 *
 * @param state A pristine state whose private graph holds its synthesized START node.
 * @param data The complete pool, identity, configuration and worker count of the run.
 * @returns The canonical `RunStartedRecord`, its frozen total plan and its observer.
 * @throws TypeError when `state` isn't exactly a `RunState`, or the worker count isn't a safe
 *   integer.
 * @throws RangeError when the state isn't pristine, identity disagrees across the inputs, START
 *   is missing or unprimed, or a field breaks the public run-record wire contract.
 */
export function prepareRunStarted(state: RunState, data: RunStartData): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareRunStarted expects a RunState");
  requirePristine(state);
  const startId = state.flow.startId;
  if (!state.flow.nodes.has(startId)) throw new RangeError(`flow START node ${reprText(startId)} is missing`);
  if (!Number.isSafeInteger(data.numWorkers)) throw new TypeError("num_workers must be a non-negative integer");
  if (data.numWorkers < 0) throw new RangeError("num_workers must be a non-negative integer");

  // Encode then decode before any identity check, so live initialization and replay both reason
  // about the same strict durable pool representation.
  const canonicalPool = decodePool(encodePool(data.initialPool));
  if (canonicalPool.startId !== startId) {
    throw new RangeError(
      `initial pool start_id mismatch: expected ${reprText(startId)}, found ${reprText(canonicalPool.startId)}`,
    );
  }
  if (canonicalPool.resolve("system", ["run_id"]) !== data.runId) {
    throw new RangeError("initial pool system.run_id must equal RunStartData.run_id");
  }
  if (!isRecord(canonicalPool.get(startId))) throw new RangeError("initial pool primed START value must be a dict");

  const candidate = new RunStartedRecord({
    flowFingerprint: data.flowFingerprint,
    initialPool: encodePool(canonicalPool),
    startWiringOverride: data.startWiringOverride,
    effectiveLlmByStaticPath: data.effectiveLlmByStaticPath,
    numWorkers: data.numWorkers,
    memorySnapshots: data.memorySnapshots ?? null,
    learnable: data.learnable ?? false,
    memories: data.memories ?? null,
    memoryStacks: data.memoryStacks ?? null,
    defaults: data.defaults ?? {},
    callTimeLlm: data.callTimeLlm ?? null,
  });
  // The public envelope codec is the sole payload authority. Decode its canonical text at once,
  // so the prepared record is exactly what either log will retain.
  const envelopeJson = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, data.runId, state.nextSequence, candidate));
  const envelope = decodeRecord(envelopeJson);
  const record = envelope.record;
  // An exhaustive codec invariant.
  if (!isExactly(record, RunStartedRecord)) {
    throw new TypeError("run-start envelope decoded to a different record kind");
  }

  const plan: RunStartedPlan = Object.freeze({
    kind: "runStarted",
    originatingState: state,
    originatingFlow: state.flow,
    envelopeJson,
    startId,
    startNode: state.flow.nodes.get(startId)!,
    endId: state.flow.endId,
    endNode: state.flow.nodes.get(state.flow.endId) ?? null,
    traceDelta: prepareRecordTrace(state, envelope),
  });
  return new PreparedTransition(record, envelope.runId, envelope.seq, plan, [new RunStarted()]);
}

/**
 * Prepare the accepted trace change while the executing node still exists.
 *
 * `options.writerFiles` is the type an accepted output's commit target reads it through to
 * restore its files.
 */
function prepareRecordTrace(
  state: RunState,
  envelope: RecordEnvelope,
  options: Omit<PrepareTraceOptions, "rootEndId" | "rootInputNames"> = {},
): TraceDelta | null {
  const record = envelope.record;
  const nodeId = outcomeNodeId(record);
  const started = isExactly(record, RunStartedRecord);
  return prepareTrace(state.trace, envelope, nodeId === null ? null : (state.flow.nodes.get(nodeId) ?? null), {
    ...options,
    rootEndId: started ? state.flow.endId : null,
    rootInputNames: started && record.learnable ? state.flow.nodes.get(state.flow.startId)!.backwardInputNames() : null,
  });
}

/** The physical executing node of a canonical outcome record, or `null` for another record. */
function outcomeNodeId(record: RunRecord): string | null {
  if (isExactly(record, NodeFailedRecord)) return record.failure.nodeId;
  if (isExactly(record, NodeExpandedRecord)) return record.spawnerId;
  if (isExactly(record, NodeSucceededRecord) || isExactly(record, NodeRoutedRecord)) return record.nodeId;
  return null;
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
