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
 * Knows about:   compile/model, defaults, diagnostics, events, expr/lineage, memory, nodes,
 *                runtime/common, runtime/contracts, runtime/learning, runtime/log, runtime/trace,
 *                typesys.
 * Never imports: engine, run, backward, compose, or the CLI.
 */

import { CompiledFlow, Edge, type Flow } from "../compile/model.ts";
import { BUILTIN, type Defaults, fromRecorded } from "../defaults.ts";
import { compareCodeUnits, Failure, type OriginRef, reprText, sameFailure } from "../diagnostics.ts";
import {
  NodeStarted,
  NodeSucceeded,
  RunFailed,
  RunResumed,
  RunStarted,
  type RunEvent,
  RunSucceeded,
} from "../events.ts";
import { SourceRef, sameSource } from "../expr/lineage.ts";
import {
  EvaluationEvidence,
  MemoryScope,
  type MemorySnapshotEnvelope,
  type MemorySpec,
  type MemoryStack,
} from "../memory/index.ts";
import { sameMemoryScope } from "../memory/contracts.ts";
import { type Grow, Node } from "../nodes/base.ts";
import {
  type Binding,
  comparePaths,
  fileFieldsType,
  isExactly,
  type OutputContract,
  outputContractFileType,
  outputContractOrigin,
  outputContractType,
  PartialRecordContract,
  sameItems,
  sameOptional,
  validateOutputContract,
} from "../nodes/contracts.ts";
import { type JsonObject, type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import { VariablePool } from "../typesys/pool.ts";
import {
  isRecord,
  normalizeTypedFiles,
  rejectDeepNesting,
  type Type,
  TypeCheckError,
  TypedValue,
  writeAsText,
} from "../typesys/values.ts";
import { NodeState } from "./common.ts";
import { validateSource } from "./contracts.ts";
import { RuntimeBackwardState } from "./learning.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  canonicalSeed,
  canonicalTyped,
  decodePool,
  decodeRecord,
  encodePool,
  encodeRecord,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  type OutcomeCarriers,
  RecordEnvelope,
  type RunLog,
  type RunRecord,
  RunResumedRecord,
  RunStartedRecord,
  type RunStopReason,
  RunStoppedRecord,
  SCHEMA_VERSION,
} from "./log.ts";
import {
  checkMemoryScope,
  type PrepareTraceOptions,
  prepareTrace,
  RetainedTrace,
  SourceEntry,
  sameSourceEntry,
  TraceDelta,
  TraceOccurrence,
  TraceTerminal,
} from "./trace.ts";

/** The injected kind-aware interpreter from a spawner and its canonical seed to a fragment. */
export type GrowBuilder = (spawner: Node, seed: JsonValue) => Flow;

/** A typed value's wire object `{kind, value}`, as `canonicalTyped` returns it. */
type TypedWire = ReturnType<typeof canonicalTyped>[0];

/** Project a whole-value type while validating the closed contract sum. */
function nodeOutputType(node: Node): Type | null {
  return outputContractType(node.outputContract);
}

/**
 * @internal Canonicalize a live or replayed output through the same contract path: a partial
 * record's contract runs first, then `canonicalTyped` against `declaredType`.
 *
 * @throws TypeCheckError when the output violates the contract or isn't a yian value.
 */
export function canonicalOutputValue(
  output: unknown,
  contract: OutputContract | null,
  declaredType: Type | null,
): [TypedWire, TypedValue] {
  if (isExactly(contract, PartialRecordContract)) output = validateOutputContract(contract, output);
  return canonicalTyped(output, declaredType);
}

/**
 * Return the type a value committed to `node` is read through to restore its files.
 *
 * The node's output contract gives it (`outputContractFileType`); a node without one that commits
 * its input record reads through its typed parameters. The run record stores each file as its
 * JSON form; reading through this type gives a reader a `FileRef` at each position typed `file`,
 * without touching the record text, so live and replayed runs read the same value.
 *
 * @returns The reading type, or `null` when no position may hold a file.
 */
function fileReadingType(node: Node): Type | null {
  if (node.outputContract === null && node.commitsInputRecord) {
    return fileFieldsType(
      (node.params ?? []).flatMap((param) =>
        param.contract === null ? [] : [[param.name, param.contract.type] as const],
      ),
    );
  }
  return outputContractFileType(node.outputContract);
}

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
 * @internal An outcome-owned value rejected before durability: it fails the node, not the run.
 *
 * It carries the original error's message; `original` is also its `cause`.
 */
export class NodeDataError extends Error {
  override name = "NodeDataError";
  /** The error the value check threw. */
  readonly original: unknown;
  /** The authored origin of the rejected value, if known. */
  readonly origin: OriginRef | null;

  /** Wrap `original`, keeping its message. */
  constructor(original: unknown, origin: OriginRef | null = null) {
    super(original instanceof Error ? original.message : String(original), { cause: original });
    this.original = original;
    this.origin = origin;
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
 * Prepare moving the exact ready-head node into ordered in-flight work.
 *
 * @param state An initialized nonterminal state whose ready head is being dispatched.
 * @param nodeId The exact non-empty id at the head of `state.ready`.
 * @returns The canonical `NodeStartedRecord` and a detached append-before-apply plan.
 * @throws TypeError when `state` or `nodeId` has the wrong type.
 * @throws RangeError when the state is uninitialized or terminal, or doesn't expose `nodeId` as
 *   one unknown, non-in-flight ready head.
 */
export function prepareNodeStarted(state: RunState, nodeId: string): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareNodeStarted expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  return prepareRecord(state, new NodeStartedRecord(nodeId));
}

/**
 * Prepare appending one step an in-flight node recorded while it runs.
 *
 * @param state An initialized nonterminal state in which `nodeId` is in flight.
 * @param nodeId The exact id of the in-flight node recording the step.
 * @param step A JSON object nested no deeper than the value nesting limit; only the node gives
 *   it meaning.
 * @returns The canonical `NodeStepRecord` and a detached append-before-apply plan, with no
 *   observers.
 * @throws TypeError when `state`, `nodeId` or `step` has the wrong type.
 * @throws RangeError when the state is uninitialized or terminal, or `nodeId` isn't in flight.
 * @throws TypeCheckError when `step` nests too deep or isn't a JSON value.
 */
export function prepareNodeStep(state: RunState, nodeId: string, step: unknown): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareNodeStep expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  if (!isRecord(step)) throw new TypeError("node step must be a JSON object");
  // Live steps only, like live outputs: replay folds recorded steps without the process limit,
  // so a log stays readable under a different limit.
  rejectDeepNesting(step, "node step");
  return prepareRecord(state, new NodeStepRecord(nodeId, step));
}

/**
 * Prepare one canonical output commit and its newly ready successors.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight.
 * @param nodeId The exact id of the executing node that produced `output`.
 * @param output The yian value (or `TypedValue`) to check against the effective target's output
 *   type; the checks and the stored value read it as given.
 * @param options.commitAs The explicit commit target; `null` (the default) falls back to the
 *   executing node's baked target, then to `nodeId` itself.
 * @param options.evidence Frozen visible evidence captured during this exact evaluation.
 * @param options.forwardCapture The canonical binding capture, required exactly when learning.
 * @returns The canonical `NodeSucceededRecord` and a detached append-before-apply plan.
 * @throws TypeError when a carrier has the wrong type or the target's contract is malformed.
 * @throws RangeError when the state, the executing node or its baked target is invalid.
 * @throws NodeDataError when `commitAs` or `output` is rejected: the output nests too deep,
 *   violates its target's type, or isn't a yian value. `original` holds the value layer's error.
 */
export function prepareNodeSucceeded(
  state: RunState,
  nodeId: string,
  output: unknown,
  options: { commitAs?: string | null } & OutcomeCarriers = {},
): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareNodeSucceeded expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  const commitAs: unknown = options.commitAs ?? null;
  if (commitAs !== null && typeof commitAs !== "string") {
    throw new NodeDataError(new TypeError("commit_as must be a string or null"));
  }
  if (commitAs === "") throw new NodeDataError(new RangeError("commit_as must be non-empty when supplied"));
  requireInitializedNonterminal(state);
  requireSucceedingNode(state, nodeId);

  const node = state.flow.nodes.get(nodeId)!;
  const nodeTarget: unknown = node.commitAs;
  if (nodeTarget !== null && (typeof nodeTarget !== "string" || !nodeTarget)) {
    throw new RangeError(`node ${reprText(nodeId)} has an invalid baked commit target`);
  }
  const target = commitAs ?? nodeTarget ?? nodeId;
  const targetNode = state.flow.nodes.get(target);
  if (targetNode === undefined) {
    const error = new RangeError(`success commit target ${reprText(target)} does not exist in the flow`);
    throw commitAs !== null ? new NodeDataError(error) : error;
  }
  const declaredType = nodeOutputType(targetNode);
  const outputContract = targetNode.outputContract;
  // Replay (`validateSuccessValue`) skips the nesting check on purpose, so logs recorded before
  // the limit still replay.
  const enclosingLevels = node.commitsInputRecord ? -1 : 0;
  let valueWire: TypedWire;
  let typed: TypedValue;
  try {
    rejectDeepNesting(output, "output", { enclosingLevels });
    [valueWire, typed] = canonicalOutputValue(output, outputContract, declaredType);
  } catch (error) {
    const path = error instanceof TypeCheckError ? error.path : [];
    throw new NodeDataError(error, outputContractOrigin(outputContract, path));
  }
  return prepareSuccessTransition(
    state,
    new NodeSucceededRecord(nodeId, target, valueWire, {
      evidence: options.evidence ?? null,
      forwardCapture: options.forwardCapture ?? null,
    }),
    typed,
  );
}

/**
 * Prepare a recovery segment from the state's ordered unfinished starts.
 *
 * The retry list comes only from the insertion-ordered `state.inFlight`; the caller chooses only
 * the worker count of the new segment.
 *
 * @param state An initialized nonterminal state rebuilt through a complete log prefix.
 * @param options.numWorkers The exact non-negative worker count; zero selects serial execution.
 * @returns The canonical `RunResumedRecord`, whose retries keep dispatch order.
 * @throws TypeError when `state` has the wrong type or the worker count isn't a safe integer.
 * @throws RangeError when the worker count is negative, or the state is uninitialized or terminal.
 */
export function prepareRunResumed(state: RunState, options: { numWorkers: number }): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareRunResumed expects a RunState");
  const { numWorkers } = options;
  if (!Number.isSafeInteger(numWorkers)) throw new TypeError("num_workers must be a non-negative integer");
  if (numWorkers < 0) throw new RangeError("num_workers must be a non-negative integer");
  requireInitializedNonterminal(state);
  return prepareRecord(state, new RunResumedRecord([...state.inFlight], numWorkers));
}

/**
 * Prepare recording that the host stopped the run, naming the in-flight nodes in dispatch order.
 *
 * The record isn't terminal and changes nothing but the sequence: steps may still follow it, and
 * a later `RunResumed` retries the nodes it names.
 *
 * @param state An initialized nonterminal state.
 * @param reason Why the run stopped.
 * @param message The reason's text, such as the callback's error message.
 * @returns The canonical `RunStoppedRecord` and its plan, with no observers.
 * @throws TypeError when `state` has the wrong type or `reason` or `message` isn't a string.
 * @throws RangeError when the state is uninitialized or terminal, or `reason` is unknown.
 */
export function prepareRunStopped(state: RunState, reason: RunStopReason, message: string): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareRunStopped expects a RunState");
  requireInitializedNonterminal(state);
  return prepareRecord(state, new RunStoppedRecord(reason, message, [...state.inFlight]));
}

/** The post-terminal records `prepareRecord` hands to the backward reducer. */
const BACKWARD_RECORDS = [
  BackwardStartedRecord,
  BackwardReadyRecord,
  BackwardFinishedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  MemoryCommitFailedRecord,
] as const;

/**
 * Validate one supported authoritative record against the current frontier.
 *
 * This is the shared reducer seam for live dispatch and historical replay. Graph records require
 * an initialized nonterminal state; backward and memory-commit records require an initialized
 * primary terminal with learning enabled.
 *
 * @param state The initialized reducer state immediately before `record`.
 * @param record One supported graph, frontier or backward-attempt record.
 * @param options.growBuilder Required only for `NodeExpandedRecord` replay; rebuilds and verifies
 *   the recorded fragment through the same injected path used live.
 * @returns The canonically copied record, its frozen plan and its observers.
 * @throws TypeError when `state` has the wrong type or the record is unsupported here.
 * @throws RangeError when the record isn't the exact next transition for this state.
 */
export function prepareRecord(
  state: RunState,
  record: RunRecord,
  options: { growBuilder?: GrowBuilder | null } = {},
): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareRecord expects a RunState");
  if (BACKWARD_RECORDS.some((cls) => isExactly<RunRecord>(record, cls))) {
    return prepareBackwardTransition(state, record);
  }

  requireInitializedNonterminal(state);

  if (isExactly(record, NodeSucceededRecord)) {
    const typed = validateSuccessValue(state, record);
    return prepareSuccessTransition(state, record, typed);
  }
  if (isExactly(record, NodeExpandedRecord)) {
    if (typeof options.growBuilder !== "function") {
      throw new TypeError("NodeExpandedRecord replay requires a callable grow_builder");
    }
    return prepareGrowthRecord(state, record, options.growBuilder);
  }
  if (isExactly(record, NodeStartedRecord)) {
    validateNodeStarted(state, record);
  } else if (isExactly(record, RunResumedRecord)) {
    validateRunResumed(state, record);
  } else if (isExactly(record, NodeStepRecord)) {
    validateNodeStep(state, record);
  } else if (isExactly(record, RunStoppedRecord)) {
    validateRunStopped(state, record);
  } else {
    const name = (record as { constructor?: { name?: unknown } } | null)?.constructor?.name;
    throw new TypeError(`unsupported run record for state preparation: ${String(name)}`);
  }

  const runId = state.runId;
  // Established by the initialized-state check.
  if (typeof runId !== "string") throw new RangeError("run state is not initialized");
  const envelopeJson = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, runId, state.nextSequence, record));
  const envelope = decodeRecord(envelopeJson);
  const canonical = envelope.record;
  let delta: FrontierPlan["delta"];
  let observers: RunEvent[];
  if (isExactly(canonical, NodeStartedRecord)) {
    validateNodeStarted(state, canonical);
    delta = { kind: "started", nodeId: canonical.nodeId };
    observers = [new NodeStarted({ nodeId: canonical.nodeId })];
  } else if (isExactly(canonical, RunResumedRecord)) {
    validateRunResumed(state, canonical);
    delta = { kind: "resumed", retryNodeIds: canonical.retryNodeIds, numWorkers: canonical.numWorkers };
    observers = [new RunResumed()];
  } else if (isExactly(canonical, NodeStepRecord)) {
    validateNodeStep(state, canonical);
    // The decoded copy keeps the applied step apart from the returned record.
    delta = { kind: "step", nodeId: canonical.nodeId, step: canonicalSeed(canonical.step)[1] as JsonObject };
    // A step is node-owned detail; no observer event announces it.
    observers = [];
  } else if (isExactly(canonical, RunStoppedRecord)) {
    validateRunStopped(state, canonical);
    delta = { kind: "stopped", nodeIds: canonical.nodeIds };
    observers = [];
  } else {
    // An exhaustive codec invariant.
    throw new TypeError("frontier envelope decoded to a different record kind");
  }

  const startId = state.flow.startId;
  const endId = state.flow.endId;
  const plan: FrontierPlan = Object.freeze({
    kind: "frontier",
    originatingState: state,
    originatingFlow: state.flow,
    envelopeJson,
    startId,
    startNode: state.flow.nodes.get(startId)!,
    endId,
    endNode: state.flow.nodes.get(endId) ?? null,
    traceDelta: prepareRecordTrace(state, envelope),
    delta: Object.freeze(delta),
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, observers);
}

/** Freeze one fully validated success delta and its canonical public record. */
function prepareSuccessTransition(state: RunState, record: NodeSucceededRecord, typed: TypedValue): PreparedTransition {
  validateNodeEvidence(state, record.nodeId, record.evidence);
  const delta = prepareSuccessDelta(state, record, typed);
  const terminal = prepareTerminalDelta(state, {
    completingNodeId: delta.nodeId,
    newlyReadyIds: delta.newlyReadyIds,
    pendingCommitTarget: delta.commitTarget,
    pendingValueJson: delta.valueJson,
    terminalUnreachableFailure: state.terminalUnreachableFailure,
  });
  const runId = state.runId;
  // Established by the initialized-state check.
  if (typeof runId !== "string") throw new RangeError("run state is not initialized");
  const envelopeJson = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, runId, state.nextSequence, record));
  const envelope = decodeRecord(envelopeJson);
  const canonical = envelope.record;
  if (!isExactly(canonical, NodeSucceededRecord)) {
    throw new TypeError("node-success envelope decoded to a different record kind");
  }
  if (encodeRecord(envelope) !== envelopeJson) {
    throw new RangeError("NodeSucceededRecord is not in canonical wire form");
  }
  const startId = state.flow.startId;
  const endId = state.flow.endId;
  const files = fileReadingType(state.flow.nodes.get(delta.commitTarget)!);
  const plan: SuccessPlan = Object.freeze({
    kind: "success",
    originatingState: state,
    originatingFlow: state.flow,
    envelopeJson,
    startId,
    startNode: state.flow.nodes.get(startId)!,
    endId,
    endNode: state.flow.nodes.get(endId) ?? null,
    traceDelta: prepareRecordTrace(state, envelope, {
      terminalSucceeded: terminal?.succeeded ?? null,
      terminalOutputJson: terminal?.outputJson ?? null,
      terminalFailure: terminal?.failure ?? null,
      writerFiles: files,
    }),
    delta,
    terminal,
    files,
  });
  const observers: RunEvent[] = [
    new NodeSucceeded({
      nodeId: delta.nodeId,
      output: normalizeTypedFiles(files, decodeTyped(delta.valueJson)).toObject(),
      commitAs: delta.observerCommitAs,
    }),
  ];
  if (terminal !== null) observers.push(terminalObserver(terminal));
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, observers);
}

/**
 * Prepare a live expansion: validate `grow` against the expanding node and build its fragment
 * with `growBuilder`, without mutating `state`.
 *
 * Not ported yet (T4-5): it always throws.
 */
export function prepareExpanded(
  state: RunState,
  nodeId: string,
  grow: Grow,
  growBuilder: GrowBuilder,
  options: OutcomeCarriers = {},
): PreparedTransition {
  throw new Error("not ported yet (T4-5)");
}

/** Prepare a recorded expansion by rebuilding its fragment. Not ported yet (T4-5): it always throws. */
function prepareGrowthRecord(
  state: RunState,
  record: NodeExpandedRecord,
  growBuilder: GrowBuilder,
): PreparedTransition {
  throw new Error("not ported yet (T4-5)");
}

/**
 * Prepare one backward or memory-commit record against the accepted terminal. Not ported yet
 * (T4-5): it always throws.
 */
function prepareBackwardTransition(state: RunState, record: RunRecord): PreparedTransition {
  throw new Error("not ported yet (T4-5)");
}

/** Every plan kind `commitPrepared` dispatches. */
const PLAN_KINDS: ReadonlySet<string> = new Set([
  "runStarted",
  "frontier",
  "success",
  "route",
  "failure",
  "growth",
  "backward",
]);

/**
 * Append one prepared record, verify the log's result, then apply its total plan.
 *
 * A log that durably appends but returns a different envelope has broken the `RunLog` protocol.
 * The live state stays unapplied; the caller must discard it and rebuild from the log's durable
 * prefix. This function can't and doesn't try to roll durable authority back.
 *
 * @param log The borrowed single-run authority; this function never closes it.
 * @param state The live state at the sequence and identity captured during preparation.
 * @param prepared The canonical record and frozen plan prepared against `state`.
 * @returns The observers, made visible only after durable append and total apply.
 * @throws RangeError when the state, the log, the preparation or the log's append result
 *   disagree on identity, sequence or the exact record.
 * @throws TypeError when the transition holds an unsupported plan.
 * @throws the log's append error, unchanged, before any state change.
 */
export function commitPrepared(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  if (!isExactly(state, RunState)) throw new TypeError("commitPrepared expects a RunState");
  if (!isExactly(prepared, PreparedTransition)) throw new TypeError("commitPrepared expects a PreparedTransition");
  const kind = (prepared._plan as { kind?: unknown } | null)?.kind;
  if (typeof kind !== "string" || !PLAN_KINDS.has(kind)) throw new TypeError("unsupported prepared transition plan");
  const traceDelta = prepared._plan.traceDelta;
  if (kind === "backward") requirePostterminalTraceDelta(state, traceDelta);
  else if (kind === "frontier") requireFrontierTraceDelta(state, prepared);
  if (traceDelta !== null) traceDelta.validateOrigin(state.trace);
  let observers: readonly RunEvent[];
  switch (prepared._plan.kind) {
    case "runStarted":
      observers = commitRunStarted(log, state, prepared);
      break;
    case "frontier":
      observers = commitFrontier(log, state, prepared);
      break;
    case "success":
      observers = commitSuccess(log, state, prepared);
      break;
    case "growth":
      observers = commitGrowth(log, state, prepared);
      break;
    case "backward":
      observers = commitBackward(log, state, prepared);
      break;
    default:
      throw new TypeError("unsupported prepared transition plan");
  }
  if (traceDelta !== null) state.trace = traceDelta.apply();
  return observers;
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

/**
 * Match the detached scope to the canonical physical node before append and prune.
 *
 * Callers have checked the record against its immutable prepared envelope. The adopted graph stays
 * the authority until apply; neither a commit target nor a substituted occurrence can select the
 * historical owner.
 */
function requireOutcomeTraceScope(state: RunState, prepared: PreparedTransition): void {
  const delta = prepared._plan.traceDelta;
  if (state.trace === null) {
    if (delta !== null) throw new RangeError("an unretained outcome cannot carry a trace delta");
    return;
  }
  if (!isExactly(delta, TraceDelta) || !isExactly(delta.occurrence, TraceOccurrence)) {
    throw new RangeError("a retained outcome requires its physical trace occurrence");
  }
  const nodeId = outcomeNodeId(prepared.record);
  const node = nodeId === null ? undefined : state.flow.nodes.get(nodeId);
  if (!(node instanceof Node)) throw new RangeError("outcome memory scope requires its physical executing node");
  if (!sameOptional(delta.occurrence.memoryScope, checkMemoryScope(node.memoryScope), sameMemoryScope)) {
    throw new RangeError("outcome memory scope disagrees with the physical executing node");
  }
}

/**
 * Match the physical occurrence, the terminal facts and the source latches to authority.
 *
 * The terminal disposition was already derived during preparation; this validates its trace
 * projection without repeating quiescence or invoking node hooks. Writer and route-latch updates
 * also constrain later terminal projections. Runs after `requireOutcomeTraceScope`.
 */
function requireOutcomeTraceTerminal(state: RunState, prepared: PreparedTransition): void {
  const trace = state.trace;
  if (trace === null) return;
  const plan = prepared._plan;
  const record = prepared.record;
  // Established by `requireOutcomeTraceScope`.
  const delta = plan.traceDelta!;
  if (
    !Number.isSafeInteger(delta.sequence) ||
    delta.sequence !== prepared.expectedSequence ||
    delta.initialized !== null ||
    delta.started !== null ||
    delta.resumed !== false
  ) {
    throw new RangeError("outcome trace cannot change initialization or dispatch authority");
  }
  const nodeId = outcomeNodeId(record)!;
  const occurrence = delta.occurrence!;
  const startSeq = trace.started.get(nodeId);
  if (startSeq === undefined) throw new RangeError("physical trace occurrence disagrees with the accepted outcome");
  const source = new SourceRef(state.runId!, startSeq);
  validateSource(occurrence.source);
  if (
    !sameSource(occurrence.source, source) ||
    typeof occurrence.nodeId !== "string" ||
    occurrence.nodeId !== nodeId ||
    !Number.isSafeInteger(occurrence.outcomeSeq) ||
    occurrence.outcomeSeq !== prepared.expectedSequence ||
    occurrence.recordJson !== plan.envelopeJson
  ) {
    throw new RangeError("physical trace occurrence disagrees with the accepted outcome");
  }
  const rootStartSource = nodeId === trace.startId && trace.rootStartSource === null ? source : null;
  if (delta.rootStartSource !== null) validateSource(delta.rootStartSource);
  if (!sameOptional(delta.rootStartSource, rootStartSource, sameSource)) {
    throw new RangeError("terminal trace root START source disagrees with the accepted outcome");
  }
  let writer: SourceEntry | null = null;
  let rootWriter: SourceEntry | null = null;
  if (plan.kind === "success" && isExactly(record, NodeSucceededRecord)) {
    const installed: unknown = delta.writer;
    if (
      !Array.isArray(installed) ||
      installed.length !== 2 ||
      typeof installed[0] !== "string" ||
      installed[0] !== record.commitTarget
    ) {
      throw new RangeError("physical trace writer disagrees with the accepted output target");
    }
    writer = new SourceEntry(source, prepared.expectedSequence, plan.delta.valueJson, plan.files);
    if (record.commitTarget === state.flow.endId) rootWriter = writer;
  } else if (delta.writer !== null) {
    throw new RangeError("only an accepted output can install a physical trace writer");
  }
  const entries = [
    [delta.writer?.[1] ?? null, writer, "physical trace writer"],
    [delta.rootWriter, rootWriter, "terminal trace root writer"],
  ] as const;
  for (const [actual, expected, label] of entries) {
    if (actual !== null) {
      if (
        !isExactly(actual, SourceEntry) ||
        !Number.isSafeInteger(actual.outcomeSeq) ||
        typeof actual.valueJson !== "string"
      ) {
        throw new RangeError(`${label} must be an exact accepted entry`);
      }
      if (actual.source !== null) validateSource(actual.source);
    }
    if (!sameOptional(actual, expected, sameSourceEntry)) {
      throw new RangeError(`${label} disagrees with the accepted outcome`);
    }
  }

  const causalSource =
    isExactly(record, NodeRoutedRecord) && record.terminalFailure !== null && trace.causalRouteSource === null
      ? source
      : null;
  if (delta.causalRouteSource !== null) validateSource(delta.causalRouteSource);
  if (!sameOptional(delta.causalRouteSource, causalSource, sameSource)) {
    throw new RangeError("terminal trace causal source disagrees with the accepted outcome");
  }

  const expected: TerminalDelta | null =
    plan.kind === "failure"
      ? { succeeded: false, outputJson: null, failure: plan.failure, files: null }
      : plan.kind === "success" || plan.kind === "route"
        ? plan.terminal
        : null;
  const terminal = delta.terminal;
  if (expected === null) {
    if (terminal !== null) throw new RangeError("nonterminal outcome cannot install a terminal trace");
    return;
  }
  // Python revalidates the frozen terminal; a TS `TraceTerminal` validated itself when built.
  if (!isExactly(terminal, TraceTerminal)) throw new RangeError("terminal outcome requires an exact terminal trace");
  let terminalSource: SourceRef | null;
  if (expected.succeeded) {
    const acceptedWriter = rootWriter ?? trace.rootWriter;
    if (acceptedWriter === null) throw new RangeError("terminal trace requires its accepted root writer");
    terminalSource = acceptedWriter.source;
  } else {
    terminalSource = plan.kind === "failure" ? source : (causalSource ?? trace.causalRouteSource);
    if (terminalSource === null) throw new RangeError("terminal trace failure requires its accepted causal occurrence");
  }
  if (
    terminal.succeeded !== expected.succeeded ||
    !sameOptional(terminal.source, terminalSource, sameSource) ||
    terminal.outputJson !== expected.outputJson ||
    !sameOptional(terminal.failure, expected.failure, sameFailure)
  ) {
    throw new RangeError("terminal trace disagrees with the prepared terminal outcome");
  }
}

/**
 * Silently fold one complete historical prefix through the live reducer path and return `state`.
 *
 * `state` is pristine around the graph that authored the run; `envelopes` is a non-empty
 * sequence-zero prefix starting with `RunStarted`; `options.growBuilder` rebuilds and verifies
 * growth. Not ported yet (T4-5): it always throws.
 */
export function replayRecords(
  state: RunState,
  envelopes: readonly RecordEnvelope[],
  options: { growBuilder: GrowBuilder },
): RunState {
  throw new Error("not ported yet (T4-5)");
}

/** Validate, append and install one initialization transition. */
function commitRunStarted(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "runStarted") throw new TypeError("RunStarted commit requires a RunStarted plan");
  if (!isExactly(prepared.record, RunStartedRecord)) throw new TypeError("RunStarted plan requires a RunStartedRecord");
  requirePreparedOrigin(state, plan);
  requirePristine(state);
  requireAuthorityCoordinates(log, state, prepared, false);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  const apply = materializeRunStarted(state, prepared, plan);

  appendAndVerify(log, prepared);
  applyRunStarted(state, apply);
  return apply.observers;
}

/** Validate, append and apply one sparse dispatch, recovery, step or stop delta. */
function commitFrontier(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "frontier") throw new TypeError("frontier commit requires a frontier plan");
  const record = prepared.record;
  if (
    !(
      isExactly(record, NodeStartedRecord) ||
      isExactly(record, RunResumedRecord) ||
      isExactly(record, NodeStepRecord) ||
      isExactly(record, RunStoppedRecord)
    )
  ) {
    throw new TypeError("frontier plan requires a dispatch, resume, step or stop record");
  }
  requirePreparedOrigin(state, plan);
  requireInitializedNonterminal(state);
  requireAuthorityCoordinates(log, state, prepared, true);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  const delta = plan.delta;
  let observers: readonly RunEvent[];
  if (isExactly(record, NodeStartedRecord) && delta.kind === "started") {
    if (delta.nodeId !== record.nodeId) throw new RangeError("frontier start delta disagrees with its record");
    observers = [new NodeStarted({ nodeId: delta.nodeId })];
  } else if (isExactly(record, RunResumedRecord) && delta.kind === "resumed") {
    if (
      !sameItems(delta.retryNodeIds, record.retryNodeIds) ||
      !Number.isSafeInteger(delta.numWorkers) ||
      delta.numWorkers !== record.numWorkers
    ) {
      throw new RangeError("frontier resume delta disagrees with its record");
    }
    observers = [new RunResumed()];
  } else if (isExactly(record, NodeStepRecord) && delta.kind === "step") {
    if (delta.nodeId !== record.nodeId || logJson(delta.step) !== logJson(record.step)) {
      throw new RangeError("frontier step delta disagrees with its record");
    }
    observers = [];
  } else if (isExactly(record, RunStoppedRecord) && delta.kind === "stopped") {
    if (!sameItems(delta.nodeIds, record.nodeIds)) {
      throw new RangeError("frontier stop delta disagrees with its record");
    }
    observers = [];
  } else {
    throw new TypeError("frontier record and prevalidated delta disagree");
  }

  appendAndVerify(log, prepared);
  switch (delta.kind) {
    case "started":
      state.ready.shift();
      state.inFlight.add(delta.nodeId);
      state.acceptedStarts.set(delta.nodeId, prepared.expectedSequence);
      state.nodeStarts.set(delta.nodeId, (state.nodeStarts.get(delta.nodeId) ?? 0) + 1);
      break;
    case "resumed":
      // Steps and start counts outlive the segment: a retried node reads them back. One `unshift`
      // per id, last first, puts the retries ahead in order without spreading a wide list.
      for (let index = delta.retryNodeIds.length - 1; index >= 0; index--) {
        state.ready.unshift(delta.retryNodeIds[index]!);
      }
      state.inFlight.clear();
      state.acceptedStarts.clear();
      state.numWorkers = delta.numWorkers;
      break;
    case "step":
      state.nodeSteps.set(
        delta.nodeId,
        Object.freeze([...(state.nodeSteps.get(delta.nodeId) ?? []), delta.step]),
      );
      break;
    case "stopped":
      break;
  }
  state.nextSequence = prepared.expectedSequence + 1;
  return observers;
}

/** Validate, append and install one successful outcome. */
function commitSuccess(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "success") throw new TypeError("success commit requires a success plan");
  const record = prepared.record;
  if (!isExactly(record, NodeSucceededRecord)) throw new TypeError("success plan requires a NodeSucceededRecord");
  requirePreparedOrigin(state, plan);
  requireInitializedNonterminal(state);
  requireAuthorityCoordinates(log, state, prepared, true);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  requireOutcomeTraceScope(state, prepared);
  requireOutcomeTraceTerminal(state, prepared);
  const delta = plan.delta;
  if (!(record.nodeId === delta.nodeId && record.commitTarget === delta.commitTarget)) {
    throw new RangeError("success record and prevalidated delta disagree");
  }
  const apply = materializeSuccess(plan);

  appendAndVerify(log, prepared);
  state.pool.store.set(delta.commitTarget, apply.value);
  state.nodeState.set(delta.nodeId, NodeState.TAKEN);
  for (const edgeId of delta.outgoingEdgeIds) state.edgeState.set(edgeId, NodeState.TAKEN);
  state.inFlight.delete(delta.nodeId);
  state.acceptedStarts.delete(delta.nodeId);
  forgetNodeSteps(state, delta.nodeId);
  for (const nodeId of delta.newlyReadyIds) state.ready.push(nodeId);
  if (apply.terminal !== null) state.result = apply.terminal.result;
  state.nextSequence = prepared.expectedSequence + 1;
  return apply.observers;
}

/** Append and apply one prepared expansion. Not ported yet (T4-5): it always throws. */
function commitGrowth(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  throw new Error("not ported yet (T4-5)");
}

/** Require the exact sequence-only trace change before accepting a post-terminal record. */
function requirePostterminalTraceDelta(state: RunState, delta: TraceDelta | null): void {
  const message = "postterminal trace delta must only advance the accepted sequence";
  if (
    !isExactly(delta, TraceDelta) ||
    delta.originatingTrace !== state.trace ||
    !Number.isSafeInteger(delta.sequence) ||
    delta.sequence !== state.nextSequence ||
    delta.resumed !== false
  ) {
    throw new RangeError(message);
  }
  const singles = [
    delta.initialized,
    delta.started,
    delta.occurrence,
    delta.handler,
    delta.writer,
    delta.consumedConnection,
    delta.boundaryUpdate,
    delta.rootStartSource,
    delta.rootWriter,
    delta.causalRouteSource,
    delta.terminal,
  ];
  const lists: unknown[] = [delta.pruneIds, delta.connections, delta.boundaries];
  if (
    singles.some((value) => value !== null) ||
    lists.some((values) => !Array.isArray(values) || values.length !== 0)
  ) {
    throw new RangeError(message);
  }
}

/** Permit only the canonical dispatch, resume, step or stop trace change before append. */
function requireFrontierTraceDelta(state: RunState, prepared: PreparedTransition): void {
  const delta = prepared._plan.traceDelta;
  const record = prepared.record;
  if (state.trace === null) {
    if (delta !== null) throw new RangeError("an unretained frontier cannot carry a frontier trace delta");
    return;
  }
  if (
    !isExactly(delta, TraceDelta) ||
    delta.originatingTrace !== state.trace ||
    !Number.isSafeInteger(delta.sequence) ||
    delta.sequence !== state.nextSequence
  ) {
    throw new RangeError("frontier trace delta requires its exact accepted sequence");
  }
  if (isExactly(record, NodeStartedRecord)) {
    const started: unknown = delta.started;
    if (
      !Array.isArray(started) ||
      started.length !== 2 ||
      started[0] !== record.nodeId ||
      started[1] !== prepared.expectedSequence ||
      delta.resumed !== false
    ) {
      throw new RangeError("frontier trace start disagrees with its canonical record");
    }
  } else if (isExactly(record, RunResumedRecord)) {
    if (delta.started !== null || delta.resumed !== true) {
      throw new RangeError("frontier trace resume disagrees with its canonical record");
    }
  } else if (isExactly(record, NodeStepRecord)) {
    if (delta.started !== null || delta.resumed !== false) {
      throw new RangeError("frontier trace step must only advance the accepted sequence");
    }
  } else if (isExactly(record, RunStoppedRecord)) {
    if (delta.started !== null || delta.resumed !== false) {
      throw new RangeError("frontier trace stop must only advance the accepted sequence");
    }
  } else {
    throw new RangeError("frontier trace requires a dispatch, resume, step or stop record");
  }
  const singles = [
    delta.initialized,
    delta.occurrence,
    delta.handler,
    delta.writer,
    delta.consumedConnection,
    delta.boundaryUpdate,
    delta.rootStartSource,
    delta.rootWriter,
    delta.causalRouteSource,
    delta.terminal,
  ];
  const lists: unknown[] = [delta.pruneIds, delta.connections, delta.boundaries];
  if (
    singles.some((value) => value !== null) ||
    lists.some((values) => !Array.isArray(values) || values.length !== 0)
  ) {
    throw new RangeError("frontier trace cannot change outcome or terminal history");
  }
}

/** Append and apply one post-terminal backward record. Not ported yet (T4-5): it always throws. */
function commitBackward(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  throw new Error("not ported yet (T4-5)");
}

/** Require the exact state, adopted flow and boundary nodes of the preparation. */
function requirePreparedOrigin(state: RunState, plan: PreparedPlan): void {
  if (state !== plan.originatingState) {
    throw new RangeError("prepared transition belongs to a different originating RunState");
  }
  requireOriginatingFlow(state, plan);
}

/** Require the state, the prepared transition and the log to share identity and sequence. */
function requireAuthorityCoordinates(
  log: RunLog,
  state: RunState,
  prepared: PreparedTransition,
  initialized: boolean,
): void {
  if (prepared.expectedSequence !== state.nextSequence) {
    throw new RangeError(
      `run-state sequence mismatch: prepared ${prepared.expectedSequence}, state ${state.nextSequence}`,
    );
  }
  if (prepared.expectedSequence !== log.nextSequence) {
    throw new RangeError(`run-log sequence mismatch: prepared ${prepared.expectedSequence}, log ${log.nextSequence}`);
  }
  if (prepared.expectedRunId !== log.runId) {
    throw new RangeError(
      `run-log run_id mismatch: prepared ${reprText(prepared.expectedRunId)}, log ${reprText(log.runId)}`,
    );
  }
  if (initialized) {
    if (state.runId !== prepared.expectedRunId) {
      const found = state.runId === null ? writeAsText(null) : reprText(state.runId);
      throw new RangeError(`run-state run_id mismatch: prepared ${reprText(prepared.expectedRunId)}, state ${found}`);
    }
  } else if (state.runId !== null) {
    throw new RangeError("RunStarted requires state.run_id to be None");
  }
}

/**
 * @internal Reject a changed or mismatched public record carrier: the record and coordinates must
 * encode to `envelopeJson`, the canonical text of the preparation.
 *
 * @throws RangeError when the coordinates or the record changed.
 */
export function requireUnchangedRecord(prepared: PreparedTransition, envelopeJson: string): void {
  const frozen = decodeRecord(envelopeJson);
  if (frozen.runId !== prepared.expectedRunId || frozen.seq !== prepared.expectedSequence) {
    throw new RangeError("prepared coordinates changed after validation");
  }
  const current = encodeRecord(
    new RecordEnvelope(frozen.schema, prepared.expectedRunId, prepared.expectedSequence, prepared.record),
  );
  if (current !== envelopeJson) throw new RangeError("prepared record changed after validation");
}

/** Append the canonical record and reject any protocol-breaking result. */
function appendAndVerify(log: RunLog, prepared: PreparedTransition): void {
  const envelope: unknown = log.append(prepared.record);
  // The record compare is the envelope text compare: the coordinates already match, and the
  // prepared record encodes to the plan's text (`requireUnchangedRecord`).
  if (
    !isExactly(envelope, RecordEnvelope) ||
    envelope.runId !== prepared.expectedRunId ||
    envelope.seq !== prepared.expectedSequence ||
    log.nextSequence !== prepared.expectedSequence + 1 ||
    encodeRecord(envelope) !== prepared._plan.envelopeJson
  ) {
    throw new RangeError("run-log append result does not match prepared transition");
  }
}

/** Check the initialized reducer roots and the recorded memory authority. */
function requireInitialized(state: RunState): void {
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
  if (!exactContainers) throw new RangeError("transition requires initialized reducer containers");
  if (typeof state.runId !== "string" || !state.runId) throw new RangeError("run state is not initialized");
  if (typeof state.flowFingerprint !== "string" || !state.flowFingerprint) {
    throw new RangeError("run state is not initialized with a flow fingerprint");
  }
  if (!Number.isSafeInteger(state.nextSequence) || state.nextSequence < 1) {
    throw new RangeError("run state is not initialized with a valid sequence");
  }
  const sequences = [...state.acceptedStarts.values()];
  if (
    !sameIds(state.acceptedStarts, state.inFlight) ||
    sequences.some((sequence) => !Number.isSafeInteger(sequence) || !(0 < sequence && sequence < state.nextSequence)) ||
    new Set(sequences).size !== sequences.length
  ) {
    throw new RangeError("accepted starts must match the unfinished invocation inventory");
  }
  // Every apply path indexes these maps by node, so a broken map must fail here, before the
  // append, rather than between append and apply.
  if (
    [...state.inFlight].some((nodeId) => !state.nodeStarts.has(nodeId)) ||
    [...state.nodeStarts.values()].some((count) => !Number.isSafeInteger(count) || count < 1) ||
    [...state.nodeSteps.keys()].some((nodeId) => !state.nodeStarts.has(nodeId)) ||
    [...state.nodeSteps.values()].some(
      (steps) => !Array.isArray(steps) || steps.length === 0 || steps.some((step) => !isRecord(step)),
    )
  ) {
    throw new RangeError("node starts and steps must match the unsettled invocations");
  }
  if (!Number.isSafeInteger(state.numWorkers) || state.numWorkers < 0) {
    throw new RangeError("run state has an invalid worker count");
  }
  const failure = state.terminalUnreachableFailure;
  if (failure !== null) {
    if (!isExactly(failure, Failure)) throw new TypeError("terminal-unreachable latch must be a Failure or null");
    if (failure.code !== "graph.terminal_unreachable") {
      throw new RangeError("terminal-unreachable latch has the wrong failure code");
    }
    if (typeof failure.nodeId !== "string" || !failure.nodeId) {
      throw new RangeError("terminal-unreachable latch requires a causal node ID");
    }
  }
  if (!state.flow.nodes.has(state.flow.startId)) {
    throw new RangeError("initialized run state is missing its START boundary");
  }
  if (state.pool.startId !== state.flow.startId) {
    throw new RangeError("initialized pool START must match the adopted flow");
  }
  if (!state.initialStaticIds.has(state.flow.startId)) {
    throw new RangeError("initialized run state START is not an initial static node");
  }
  if (state.learnable) {
    if (
      !isExactly(state.trace, RetainedTrace) ||
      state.trace.runId !== state.runId ||
      state.trace.nextSequence !== state.nextSequence
    ) {
      throw new RangeError("run-state trace disagrees with recorded authority");
    }
  } else if (state.trace !== null) {
    throw new RangeError("trace requires recorded learning");
  }
  validateRuntimeMemoryConfiguration(state);
}

/** Require an initialized graph state before its primary terminal outcome. */
function requireInitializedNonterminal(state: RunState): void {
  requireInitialized(state);
  if (state.result !== null) throw new RangeError("terminal run state cannot prepare another graph transition");
}

/** Require the identity roots of canonical RunStarted decoding. */
function validateRuntimeMemoryConfiguration(state: RunState): void {
  const authority = state.memoryAuthority;
  if (!isExactly(authority, MemoryAuthority)) throw new TypeError("run-state memory authority has the wrong type");
  if (
    !(
      state.memorySnapshots === authority.snapshots &&
      state.memories === authority.memories &&
      state.memoryStacks === authority.memoryStacks &&
      typeof state.learnable === "boolean" &&
      state.learnable === authority.learnable &&
      state.backwardState.memoryOrder === authority.memoryOrder
    )
  ) {
    throw new RangeError("run-state memory roots disagree with RunStarted authority");
  }
}

/** Require `record` to dispatch exactly one eligible ready-head node. */
function validateNodeStarted(state: RunState, record: NodeStartedRecord): void {
  const nodeId: unknown = record.nodeId;
  if (typeof nodeId !== "string") throw new TypeError("NodeStartedRecord node_id must be a string");
  if (state.ready.length === 0 || state.ready[0] !== nodeId) {
    const found = state.ready.length === 0 ? writeAsText(null) : reprText(state.ready[0]!);
    throw new RangeError(`NodeStarted requires ready head ${reprText(nodeId)}, found ${found}`);
  }
  if (!state.flow.nodes.has(nodeId)) {
    throw new RangeError(`NodeStarted node ${reprText(nodeId)} does not exist in the flow`);
  }
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`NodeStarted node ${reprText(nodeId)} must have UNKNOWN disposition`);
  }
  if (state.inFlight.has(nodeId)) throw new RangeError(`NodeStarted node ${reprText(nodeId)} is already in-flight`);
}

/** Require a resume record to name the exact ordered unfinished frontier. */
function validateRunResumed(state: RunState, record: RunResumedRecord): void {
  if (!Number.isSafeInteger(record.numWorkers)) throw new TypeError("num_workers must be a non-negative integer");
  if (record.numWorkers < 0) throw new RangeError("num_workers must be a non-negative integer");
  const retries: unknown = record.retryNodeIds;
  if (!Array.isArray(retries) || retries.some((nodeId) => typeof nodeId !== "string")) {
    throw new TypeError("RunResumedRecord retry_node_ids must be an array of strings");
  }
  const expected = [...state.inFlight];
  if (!sameItems(record.retryNodeIds, expected)) {
    throw new RangeError(
      "RunResumedRecord retry_node_ids must exactly equal ordered in-flight nodes: " +
        `expected ${writeAsText(expected)}, found ${writeAsText([...record.retryNodeIds])}`,
    );
  }
  for (const nodeId of record.retryNodeIds) {
    if (!state.flow.nodes.has(nodeId)) {
      throw new RangeError(`RunResumed retry node ${reprText(nodeId)} does not exist in the flow`);
    }
    if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
      throw new RangeError(`RunResumed retry node ${reprText(nodeId)} must have UNKNOWN disposition`);
    }
  }
}

/** Require a stop record to name the exact ordered in-flight nodes. */
function validateRunStopped(state: RunState, record: RunStoppedRecord): void {
  const nodeIds: unknown = record.nodeIds;
  if (!Array.isArray(nodeIds) || nodeIds.some((nodeId) => typeof nodeId !== "string")) {
    throw new TypeError("RunStoppedRecord node_ids must be an array of strings");
  }
  const expected = [...state.inFlight];
  if (!sameItems(record.nodeIds, expected)) {
    throw new RangeError(
      "RunStoppedRecord node_ids must exactly equal ordered in-flight nodes: " +
        `expected ${writeAsText(expected)}, found ${writeAsText([...record.nodeIds])}`,
    );
  }
  for (const nodeId of record.nodeIds) {
    if (!state.flow.nodes.has(nodeId)) {
      throw new RangeError(`RunStopped node ${reprText(nodeId)} does not exist in the flow`);
    }
    if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
      throw new RangeError(`RunStopped node ${reprText(nodeId)} must have UNKNOWN disposition`);
    }
  }
}

/** Require a step record to name an existing unknown in-flight node. */
function validateNodeStep(state: RunState, record: NodeStepRecord): void {
  const nodeId: unknown = record.nodeId;
  if (typeof nodeId !== "string") throw new TypeError("NodeStepRecord node_id must be a string");
  if (!state.flow.nodes.has(nodeId)) {
    throw new RangeError(`NodeStep node ${reprText(nodeId)} does not exist in the flow`);
  }
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`NodeStep node ${reprText(nodeId)} must have UNKNOWN disposition`);
  }
  if (!state.inFlight.has(nodeId)) throw new RangeError(`NodeStep node ${reprText(nodeId)} is not in-flight`);
}

/**
 * Drop a settled node's start count and recorded steps.
 *
 * @throws RangeError when the node has no start count, as Python's `del` raises `KeyError`.
 */
function forgetNodeSteps(state: RunState, nodeId: string): void {
  if (!state.nodeStarts.delete(nodeId)) throw new RangeError(`node ${reprText(nodeId)} has no start count`);
  state.nodeSteps.delete(nodeId);
}

/** Require an existing unknown node in the in-flight frontier. */
function requireSucceedingNode(state: RunState, nodeId: string): void {
  if (!state.flow.nodes.has(nodeId)) {
    throw new RangeError(`NodeSucceeded node ${reprText(nodeId)} does not exist in the flow`);
  }
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`NodeSucceeded node ${reprText(nodeId)} must have UNKNOWN disposition`);
  }
  if (!state.inFlight.has(nodeId)) throw new RangeError(`NodeSucceeded node ${reprText(nodeId)} is not in-flight`);
}

/** Whether two sets of ids hold the same ids, in any order. */
function sameIds(a: { readonly size: number; has(id: string): boolean }, b: Iterable<string>): boolean {
  const other = new Set(b);
  return a.size === other.size && [...other].every((id) => a.has(id));
}

/** Return the recorded stack at `locator` by binary search over the sorted stacks, or `null`. */
function recordedStack(stacks: readonly MemoryStack[], locator: readonly string[]): MemoryStack | null {
  let low = 0;
  let high = stacks.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (comparePaths(stacks[middle]!.locator, locator) < 0) low = middle + 1;
    else high = middle;
  }
  return low < stacks.length && sameItems(stacks[low]!.locator, locator) ? stacks[low]! : null;
}

/** Enforce recorded learning authority for one accepted graph outcome. */
function validateNodeEvidence(state: RunState, nodeId: string, evidence: EvaluationEvidence | null): void {
  const node = state.flow.nodes.get(nodeId);
  if (node === undefined) throw new RangeError(`evidence node ${reprText(nodeId)} does not exist in the flow`);
  const scope = node.memoryScope;

  if (state.learnable !== true || node.needsEvaluationTrace !== true) {
    if (evidence !== null) throw new RangeError(`node ${reprText(nodeId)} is ineligible for evaluation evidence`);
    return;
  }
  if (scope !== null) {
    // Python rebuilds the frozen scope to revalidate it; a TS scope validated itself when built.
    if (!isExactly(scope, MemoryScope)) {
      throw new RangeError(`node ${reprText(nodeId)} has no exact compiled memory scope`);
    }
    let expected: readonly string[] = [];
    if (state.memoryStacks !== null) {
      const recorded = recordedStack(state.memoryStacks, scope.staticPath);
      if (recorded === null) throw new RangeError(`node ${reprText(nodeId)} has no recorded memory stack`);
      expected = recorded.memories;
    }
    if (!sameItems(scope.memories, expected)) {
      throw new RangeError(`node ${reprText(nodeId)} memory scope must carry its exact recorded memory stack`);
    }
  } else if (state.memories !== null && state.memories.length > 0) {
    throw new RangeError(`node ${reprText(nodeId)} has no exact compiled memory scope`);
  }
  if (evidence === null) {
    throw new RangeError(`eligible node ${reprText(nodeId)} requires evaluation evidence while learning`);
  }
  if (!isExactly(evidence, EvaluationEvidence)) {
    throw new TypeError("node evidence must be an EvaluationEvidence or null");
  }
  if (!sameOptional(evidence.scope, scope, sameMemoryScope)) {
    throw new RangeError(`node ${reprText(nodeId)} evidence scope does not match its compiled scope`);
  }
}

/** Validate one replayed value against its explicit commit target exactly once. */
function validateSuccessValue(state: RunState, record: NodeSucceededRecord): TypedValue {
  if (typeof record.nodeId !== "string") throw new TypeError("NodeSucceededRecord node_id must be a string");
  if (typeof record.commitTarget !== "string") {
    throw new TypeError("NodeSucceededRecord commit_target must be a string");
  }
  requireSucceedingNode(state, record.nodeId);
  const targetNode = state.flow.nodes.get(record.commitTarget);
  if (targetNode === undefined) {
    throw new RangeError(`success commit target ${reprText(record.commitTarget)} does not exist in the flow`);
  }
  let typed: TypedValue;
  try {
    typed = TypedValue.fromWire(record.value);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw new RangeError("NodeSucceededRecord value is not a typed value", { cause: error });
  }
  const [wire, decoded] = canonicalOutputValue(typed, targetNode.outputContract, nodeOutputType(targetNode));
  if (logJson(wire) !== logJson(record.value)) {
    throw new RangeError("NodeSucceededRecord value is not canonical for its target");
  }
  return decoded;
}

/** Validate the touched topology and freeze one sparse successful-output delta. */
function prepareSuccessDelta(state: RunState, record: NodeSucceededRecord, typed: TypedValue): SuccessDelta {
  if (!isExactly(state.pool.store, Map)) throw new RangeError("success requires an exact variable-pool store");
  const target = record.commitTarget;
  const outgoingEdgeIds: string[] = [];
  const newlyReadyIds: string[] = [];
  const settledEdgeIds = new Set<string>();
  const seenOutgoingEdgeIds = new Set<string>();
  const remainingBySuccessor = new Map<string, number>();
  const edgeIdsBySuccessor = new Map<string, string[]>();
  const outgoing = state.flow.outgoing(target);

  for (const edge of outgoing) {
    if (!(edge instanceof Edge) || edge.from !== target) {
      throw new RangeError(`success target ${reprText(target)} has an invalid outgoing adjacency`);
    }
    const edgeId: unknown = edge.id;
    if (typeof edgeId !== "string" || !edgeId) throw new RangeError("success target has an invalid outgoing edge ID");
    if (seenOutgoingEdgeIds.has(edgeId)) {
      throw new RangeError(`success target outgoing edge ${reprText(edgeId)} is duplicated`);
    }
    if (!state.edgeState.has(edgeId)) {
      throw new RangeError(`success target outgoing edge ${reprText(edgeId)} has no disposition`);
    }
    if (state.edgeState.get(edgeId) !== NodeState.UNKNOWN) {
      throw new RangeError(`success target outgoing edge ${reprText(edgeId)} must be UNKNOWN`);
    }
    if (!state.flow.nodes.has(edge.to) || !state.nodeState.has(edge.to)) {
      throw new RangeError(
        `success target outgoing edge ${reprText(edgeId)} has unknown successor ${reprText(edge.to)}`,
      );
    }
    outgoingEdgeIds.push(edgeId);
    seenOutgoingEdgeIds.add(edgeId);
    remainingBySuccessor.set(edge.to, (remainingBySuccessor.get(edge.to) ?? 0) + 1);
    const ids = edgeIdsBySuccessor.get(edge.to);
    if (ids === undefined) edgeIdsBySuccessor.set(edge.to, [edgeId]);
    else ids.push(edgeId);
  }

  for (const edge of outgoing) {
    settledEdgeIds.add(edge.id);
    const successor = edge.to;
    const remaining = remainingBySuccessor.get(successor)! - 1;
    remainingBySuccessor.set(successor, remaining);
    if (remaining) continue;
    const required = new Set(edgeIdsBySuccessor.get(successor));
    if (sparseDisposition(state.flow, state.edgeState, settledEdgeIds, successor, required) !== "ready") continue;
    if (state.nodeState.get(successor) !== NodeState.UNKNOWN) {
      throw new RangeError(`newly-ready successor ${reprText(successor)} must have UNKNOWN disposition`);
    }
    newlyReadyIds.push(successor);
  }

  return Object.freeze({
    nodeId: record.nodeId,
    commitTarget: target,
    valueJson: logJson(typed.toWire()),
    outgoingEdgeIds: Object.freeze(outgoingEdgeIds),
    newlyReadyIds: Object.freeze(newlyReadyIds),
    observerCommitAs: target === record.nodeId ? null : target,
  });
}

/**
 * Derive an optional terminal result from one predicted post-outcome state.
 *
 * Quiescence depends only on the two frontier roots and the sparse additions of the causal
 * outcome. A successful END write is read from its pending typed text; an already committed END
 * value is re-encoded, so the plan never aliases the live pool.
 */
function prepareTerminalDelta(
  state: RunState,
  outcome: {
    completingNodeId: string;
    newlyReadyIds: readonly string[];
    pendingCommitTarget: string | null;
    pendingValueJson: string | null;
    terminalUnreachableFailure: Failure | null;
  },
): TerminalDelta | null {
  if (
    state.result !== null ||
    state.ready.length > 0 ||
    outcome.newlyReadyIds.length > 0 ||
    state.inFlight.size !== 1 ||
    !state.inFlight.has(outcome.completingNodeId)
  ) {
    return null;
  }

  const endId = state.flow.endId;
  const files = outputContractFileType(state.resultContract);
  if (outcome.pendingCommitTarget === endId) {
    if (typeof outcome.pendingValueJson !== "string") {
      throw new RangeError("pending END output must be canonical typed text");
    }
    return Object.freeze({ succeeded: true, outputJson: outcome.pendingValueJson, failure: null, files });
  }
  const stored = state.pool.store.get(endId);
  if (stored !== undefined) {
    return Object.freeze({ succeeded: true, outputJson: logJson(stored.toWire()), failure: null, files });
  }
  if (outcome.terminalUnreachableFailure === null) {
    throw new RangeError("run quiesced without END output or a causal terminal failure");
  }
  return Object.freeze({
    succeeded: false,
    outputJson: null,
    failure: outcome.terminalUnreachableFailure,
    files: null,
  });
}

/** Every `NodeState` value, for checking a disposition read from a state map. */
const NODE_STATES: ReadonlySet<unknown> = new Set(Object.values(NodeState));

/**
 * Classify one touched join over the live edge states plus the sparse TAKEN overlay
 * `takenEdgeIds`; `requiredEdgeIds` are the settling edges that must be among its incoming ones.
 */
function sparseDisposition(
  flow: CompiledFlow,
  edgeState: ReadonlyMap<string, NodeState>,
  takenEdgeIds: ReadonlySet<string>,
  nodeId: string,
  requiredEdgeIds: ReadonlySet<string>,
): "ready" | "wait" | "dead" {
  const absent = (ids: Iterable<string>) =>
    `success edges ${writeAsText([...ids].sort(compareCodeUnits))} are absent from successor ` +
    `${reprText(nodeId)} incoming adjacency`;
  const incoming = flow.incoming(nodeId);
  if (incoming.length === 0) throw new RangeError(absent(requiredEdgeIds));

  const foundRequired = new Set<string>();
  let sawControl = false;
  let allControlSkipped = true;
  let requiredOrderingSkipped = false;
  let anyUnknown = false;
  const dataGroups = new Map<string | null, readonly [boolean, boolean]>();
  const seenIncomingIds = new Set<string>();
  for (const edge of incoming) {
    if (!(edge instanceof Edge) || edge.to !== nodeId) {
      throw new RangeError(`successor ${reprText(nodeId)} has an invalid incoming adjacency`);
    }
    const edgeId: unknown = edge.id;
    if (typeof edgeId !== "string" || !edgeId || seenIncomingIds.has(edgeId)) {
      throw new RangeError(`successor ${reprText(nodeId)} has invalid incoming edge IDs`);
    }
    seenIncomingIds.add(edgeId);
    if (requiredEdgeIds.has(edgeId)) foundRequired.add(edgeId);
    if (!edgeState.has(edgeId)) throw new RangeError(`successor incoming edge ${reprText(edgeId)} has no disposition`);
    const disposition = takenEdgeIds.has(edgeId) ? NodeState.TAKEN : edgeState.get(edgeId);
    if (!NODE_STATES.has(disposition)) {
      throw new RangeError(`successor incoming edge ${reprText(edgeId)} has an invalid disposition`);
    }
    anyUnknown ||= disposition === NodeState.UNKNOWN;
    if (edge.sourceHandle !== null) {
      sawControl = true;
      allControlSkipped &&= disposition === NodeState.SKIPPED;
    } else if (edge.ordering) {
      requiredOrderingSkipped ||= !edge.optional && disposition === NodeState.SKIPPED;
    } else {
      const [optional, allSkipped] = dataGroups.get(edge.inputGroup) ?? [edge.optional, true];
      dataGroups.set(edge.inputGroup, [optional, allSkipped && disposition === NodeState.SKIPPED]);
    }
  }

  if (foundRequired.size !== requiredEdgeIds.size) {
    throw new RangeError(absent([...requiredEdgeIds].filter((id) => !foundRequired.has(id))));
  }
  if (sawControl && allControlSkipped) return "dead";
  for (const [optional, allSkipped] of dataGroups.values()) {
    if (!optional && allSkipped) return "dead";
  }
  if (requiredOrderingSkipped) return "dead";
  if (anyUnknown) return "wait";
  return "ready";
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
    sameIds(state.nodeState, flow.nodes.keys()) &&
    [...state.nodeState.values()].every((value) => value === NodeState.UNKNOWN) &&
    sameIds(state.edgeState, flow.edges.map((edge) => edge.id)) &&
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

/**
 * Reject a replaced adopted flow object or boundary node.
 *
 * The graph is a valid private `CompiledFlow` whose execution fields reducer transitions own. This
 * guard enforces that ownership boundary; it deliberately doesn't pretend to fingerprint arbitrary
 * in-place mutation by external code.
 */
function requireOriginatingFlow(state: RunState, plan: PreparedPlan): void {
  const flow = state.flow;
  if (
    !(
      flow === plan.originatingFlow &&
      flow.startId === plan.startId &&
      flow.nodes.has(plan.startId) &&
      flow.nodes.get(plan.startId) === plan.startNode &&
      flow.endId === plan.endId &&
      (flow.nodes.get(plan.endId) ?? null) === plan.endNode
    )
  ) {
    throw new RangeError("transition requires its originating flow and boundary identities");
  }
}

/** The detached values one initialization installs, all built before the durable append. */
interface RunStartedApply {
  pool: VariablePool;
  nodeState: Map<string, NodeState>;
  edgeState: Map<string, NodeState>;
  ready: string[];
  inFlight: Set<string>;
  acceptedStarts: Map<string, number>;
  terminalUnreachableFailure: Failure | null;
  depth: Map<string, number>;
  growthOwner: Map<string, string>;
  wiringOverrides: Map<string, JsonObject>;
  runId: string;
  flowFingerprint: string;
  nextSequence: number;
  numWorkers: number;
  input: JsonObject;
  effectiveLlmByStaticPath: Record<string, JsonObject>;
  callTimeLlm: JsonObject | null;
  memorySnapshots: readonly MemorySnapshotEnvelope[] | null;
  memories: readonly MemorySpec[] | null;
  memoryStacks: readonly MemoryStack[] | null;
  learnable: boolean;
  runDefaults: Defaults;
  backwardState: RuntimeBackwardState;
  memoryAuthority: MemoryAuthority;
  observers: readonly RunEvent[];
}

/** Decode and detach every apply value before the record becomes durable. */
function materializeRunStarted(state: RunState, prepared: PreparedTransition, plan: RunStartedPlan): RunStartedApply {
  const envelope = decodeRecord(plan.envelopeJson);
  const record = envelope.record;
  if (
    !(
      envelope.runId === prepared.expectedRunId &&
      envelope.seq === prepared.expectedSequence &&
      isExactly(record, RunStartedRecord)
    )
  ) {
    throw new RangeError("immutable RunStarted plan does not match its transition");
  }

  const pool = decodePool(record.initialPool);
  const startId = plan.startId;
  if (pool.startId !== startId) throw new RangeError("immutable RunStarted pool start_id mismatch");
  if (pool.resolve("system", ["run_id"]) !== envelope.runId) {
    throw new RangeError("immutable RunStarted pool system.run_id mismatch");
  }
  const primedStart = pool.get(startId);
  if (!isRecord(primedStart)) throw new RangeError("immutable RunStarted pool START value must be a dict");

  const wiring = canonicalSeed(record.startWiringOverride)[1];
  const effective = canonicalSeed(record.effectiveLlmByStaticPath)[1];
  const input = canonicalSeed(primedStart)[1];
  if (!isRecord(wiring)) throw new TypeError("canonical START wiring is not a dict");
  if (!isRecord(effective)) throw new TypeError("canonical effective LLM map is not a dict");
  if (!isRecord(input)) throw new TypeError("canonical START input is not a dict");

  const backwardState = new RuntimeBackwardState(
    record.memories === null ? [] : record.memories.map((spec) => spec.key),
  );
  return {
    pool,
    nodeState: new Map(state.nodeState),
    edgeState: new Map(state.edgeState),
    ready: [startId],
    inFlight: new Set(),
    acceptedStarts: new Map(),
    terminalUnreachableFailure: null,
    depth: new Map(),
    growthOwner: new Map(),
    wiringOverrides: new Map([[startId, wiring as JsonObject]]),
    runId: envelope.runId,
    flowFingerprint: record.flowFingerprint,
    nextSequence: envelope.seq + 1,
    numWorkers: record.numWorkers,
    input: input as JsonObject,
    effectiveLlmByStaticPath: effective as Record<string, JsonObject>,
    callTimeLlm: record.callTimeLlm === null ? null : (canonicalSeed(record.callTimeLlm)[1] as JsonObject),
    memorySnapshots: record.memorySnapshots,
    memories: record.memories,
    memoryStacks: record.memoryStacks,
    learnable: record.learnable,
    runDefaults: fromRecorded(record.defaults),
    backwardState,
    memoryAuthority: new MemoryAuthority(
      record.memorySnapshots,
      record.learnable,
      record.memories,
      record.memoryStacks,
      backwardState.memoryOrder,
    ),
    observers: [new RunStarted()],
  };
}

/** Install the already detached values with total replacement assignments only. */
function applyRunStarted(state: RunState, apply: RunStartedApply): void {
  state.pool = apply.pool;
  state.nodeState = apply.nodeState;
  state.edgeState = apply.edgeState;
  state.ready = apply.ready;
  state.inFlight = apply.inFlight;
  state.acceptedStarts = apply.acceptedStarts;
  state.result = null;
  state.terminalUnreachableFailure = apply.terminalUnreachableFailure;
  state.depth = apply.depth;
  state.growthOwner = apply.growthOwner;
  state.wiringOverrides = apply.wiringOverrides;
  state.runId = apply.runId;
  state.flowFingerprint = apply.flowFingerprint;
  state.numWorkers = apply.numWorkers;
  state.input = apply.input;
  state.effectiveLlmByStaticPath = apply.effectiveLlmByStaticPath;
  state.callTimeLlm = apply.callTimeLlm;
  state.memorySnapshots = apply.memorySnapshots;
  state.memories = apply.memories;
  state.memoryStacks = apply.memoryStacks;
  state.learnable = apply.learnable;
  state.runDefaults = apply.runDefaults;
  state.backwardState = apply.backwardState;
  state.memoryAuthority = apply.memoryAuthority;
  state.nextSequence = apply.nextSequence;
}

/** The detached values one success commit installs, all built before the durable append. */
interface SuccessApply {
  value: TypedValue;
  terminal: TerminalApply | null;
  observers: readonly RunEvent[];
}

/** Decode independent pool, observer and terminal copies before append. */
function materializeSuccess(plan: SuccessPlan): SuccessApply {
  const value = normalizeTypedFiles(plan.files, decodeTyped(plan.delta.valueJson));
  const observerValue = normalizeTypedFiles(plan.files, decodeTyped(plan.delta.valueJson)).toObject();
  const terminal = plan.terminal !== null ? materializeTerminal(plan.terminal) : null;
  const observers: RunEvent[] = [
    new NodeSucceeded({ nodeId: plan.delta.nodeId, output: observerValue, commitAs: plan.delta.observerCommitAs }),
  ];
  if (terminal !== null) observers.push(terminal.observer);
  return { value, terminal, observers };
}

/** Decode a typed value from its canonical log text, a fresh copy. */
function decodeTyped(text: string): TypedValue {
  return TypedValue.fromWire(parseJson(text, "codec"));
}

/**
 * Materialize one detached terminal event from immutable plan authority.
 *
 * A success output is read through `delta.files`, so each file the run's result contract types as
 * `file` is a `FileRef`.
 */
function terminalObserver(delta: TerminalDelta): RunSucceeded | RunFailed {
  if (delta.succeeded) {
    if (typeof delta.outputJson !== "string") {
      throw new RangeError("successful terminal delta requires typed output text");
    }
    if (delta.failure !== null) throw new RangeError("successful terminal delta cannot carry a failure");
    return new RunSucceeded({ output: normalizeTypedFiles(delta.files, decodeTyped(delta.outputJson)).toObject() });
  }
  if (delta.outputJson !== null) throw new RangeError("failed terminal delta cannot carry output text");
  if (!isExactly(delta.failure, Failure)) throw new RangeError("failed terminal delta requires one Failure");
  return new RunFailed({ failure: delta.failure });
}

/** Independent state and callback terminal values. */
interface TerminalApply {
  result: RunSucceeded | RunFailed;
  observer: RunSucceeded | RunFailed;
}

/** Build independent state and callback terminal values before the durable append. */
function materializeTerminal(delta: TerminalDelta): TerminalApply {
  return { result: terminalObserver(delta), observer: terminalObserver(delta) };
}
