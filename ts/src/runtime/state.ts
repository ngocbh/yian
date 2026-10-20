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
 * Knows about:   compile/manifest, compile/model, compile/validation, defaults, diagnostics,
 *                events, expr, memory, nodes, runtime/common, runtime/contracts, runtime/learning,
 *                runtime/log, runtime/trace, typesys.
 * Never imports: engine, run, backward, compose, or the CLI.
 */

import { expansionDigest } from "../compile/manifest.ts";
import { CompiledFlow, copyGraph, Edge, Flow, GrowthRejected, validateGrowthMetadata } from "../compile/model.ts";
import { classifyPath } from "../compile/validation.ts";
import { BUILTIN, type Defaults, fromRecorded } from "../defaults.ts";
import { compareCodeUnits, Failure, type OriginRef, reprText, sameFailure } from "../diagnostics.ts";
import {
  NodeExpanded,
  NodeFailed,
  NodeRouted,
  NodeSkipped,
  NodeStarted,
  NodeSucceeded,
  RunFailed,
  RunResumed,
  RunStarted,
  type RunEvent,
  RunSucceeded,
} from "../events.ts";
import { ExpressionError } from "../expr/expressions.ts";
import { SourceRef, sameSource } from "../expr/lineage.ts";
import { exprRefsOf } from "../expr/template.ts";
import {
  EvaluationEvidence,
  MemoryScope,
  type MemorySnapshotEnvelope,
  type MemorySpec,
  type MemoryStack,
} from "../memory/index.ts";
import { sameMemoryScope } from "../memory/contracts.ts";
import { ELSE_HANDLE, Grow, Node } from "../nodes/base.ts";
import {
  type Binding,
  comparePaths,
  fileFieldsType,
  isExactly,
  iterSourceStrings,
  type OutputContract,
  outputContractFieldError,
  outputContractFileType,
  outputContractOrigin,
  outputContractType,
  Parameter,
  PartialRecordContract,
  sameItems,
  sameOptional,
  validateOutputContract,
} from "../nodes/contracts.ts";
import { type JsonObject, type JsonValue, logJson } from "../typesys/json.ts";
import { VariablePool } from "../typesys/pool.ts";
import {
  isRecord,
  jsTypeName,
  normalizeTypedFiles,
  rejectDeepNesting,
  type Type,
  TypeCheckError,
  TypedValue,
  writeAsText,
} from "../typesys/values.ts";
import { NodeState } from "./common.ts";
import { ForwardCapture, validateSource } from "./contracts.ts";
import {
  type BackwardRecord,
  reduceBackwardRecord,
  RuntimeBackwardState,
  sameRuntimeBackwardState,
} from "./learning.ts";
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
  decodeTyped,
  type PrepareTraceOptions,
  prepareTrace,
  RetainedTrace,
  selectSeedRoute,
  SourceEntry,
  sameSourceEntry,
  TraceDelta,
  TraceOccurrence,
  TraceTerminal,
} from "./trace.ts";

/** The injected kind-aware interpreter from a spawner and its canonical seed to a fragment. */
export type GrowBuilder = (spawner: Node, seed: JsonValue) => Flow;

/** A typed value's wire object `{kind, value}`. */
type TypedWire = ReturnType<TypedValue["toWire"]>;

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

/** Mark one explicit live node-produced growth rejection. */
function raiseGrowthRejected(original: TypeError | RangeError): never {
  throw new GrowthRejected(original);
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
  /**
   * The number of accepted starts of each node that a later `RunResumed` retried with no
   * `RunStopped` covering them. Log history by node id: never decremented, and kept after the
   * node's outcome and after a prune.
   */
  interruptedAttempts = new Map<string, number>();
  /** The sequence of the last `RunStopped` since the last `RunResumed`, or `null` when none. */
  lastStopSequence: number | null = null;
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
 * @internal The read-only historical authority presented to the live commit reducer.
 *
 * The cursor owns canonical envelope texts. Its `append` advances only when the reducer's exact
 * next envelope matches that text; it never writes to or closes the log the prefix came from.
 */
export class HistoricalLogCursor implements RunLog {
  /** The sole historical run id this cursor accepts. */
  readonly runId: string;
  /** Always `null`: the cursor borrows already-read authority. */
  readonly path = null;
  readonly #envelopeJsons: readonly string[];
  readonly #consumed: RecordEnvelope[] = [];

  /** Adopt one validated non-empty canonical prefix at sequence zero. */
  constructor(envelopeJsons: readonly string[]) {
    this.#envelopeJsons = envelopeJsons;
    this.runId = decodeRecord(envelopeJsons[0]!).runId;
  }

  /** The sequence of the historical envelope the next reducer commit must match. */
  get nextSequence(): number {
    return this.#consumed.length;
  }

  /** Iterate the envelopes consumed so far. */
  [Symbol.iterator](): Iterator<RecordEnvelope> {
    return this.records()[Symbol.iterator]();
  }

  /** Return the envelopes consumed so far, as a fresh frozen array. */
  records(): readonly RecordEnvelope[] {
    return Object.freeze([...this.#consumed]);
  }

  /**
   * Match and consume one historical envelope without writing it again.
   *
   * @throws RangeError when the prefix is exhausted, or `record` isn't the next historical record.
   */
  append(record: RunRecord): RecordEnvelope {
    const sequence = this.nextSequence;
    if (sequence >= this.#envelopeJsons.length) {
      throw new RangeError("replay attempted to append beyond the historical prefix");
    }
    const expected = this.#envelopeJsons[sequence]!;
    const envelope = decodeRecord(expected);
    const candidate = encodeRecord(new RecordEnvelope(envelope.schema, this.runId, sequence, record));
    if (candidate !== expected) throw new RangeError("prepared transition does not match the historical run record");
    this.#consumed.push(envelope);
    return envelope;
  }

  /** Do nothing: the cursor owns no resource. */
  close(): void {}

  /** Do nothing: the cursor owns no resource. */
  [Symbol.dispose](): void {}
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
 * Prepare one selected route and its derived recursive skip closure.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight.
 * @param nodeId The exact non-empty id of the executing router.
 * @param handle The exact non-empty handle the router selected.
 * @param options.evidence Frozen visible evidence captured during this exact evaluation.
 * @param options.forwardCapture The canonical binding capture, required exactly when learning.
 * @returns The canonical `NodeRoutedRecord` and a detached append-before-apply plan.
 * @throws TypeError when a carrier has the wrong type.
 * @throws RangeError when the state, the router or the frontier is invalid.
 * @throws NodeDataError when `handle` isn't a non-empty string or matches no outgoing edge;
 *   `original` holds the `TypeError` or `RangeError`.
 */
export function prepareNodeRouted(
  state: RunState,
  nodeId: string,
  handle: string,
  options: OutcomeCarriers = {},
): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareNodeRouted expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  if (!nodeId) throw new RangeError("node_id must be non-empty");
  if (typeof handle !== "string") throw new NodeDataError(new TypeError("handle must be a string"));
  if (!handle) throw new NodeDataError(new RangeError("handle must be non-empty"));
  requireInitializedNonterminal(state);
  requireRoutingNode(state, nodeId);
  return prepareRouteTransition(
    state,
    new NodeRoutedRecord(nodeId, handle, {
      evidence: options.evidence ?? null,
      forwardCapture: options.forwardCapture ?? null,
    }),
    true,
  );
}

/**
 * Prepare one terminal node failure with its canonical durable authority.
 *
 * @param state An initialized nonterminal state with the failing node in flight.
 * @param failure The semantic failure whose non-empty `nodeId` names the executing node.
 * @param options.evidence Frozen visible evidence captured before the evaluation failed.
 * @param options.forwardCapture The canonical binding capture, including unavailable inputs on an
 *   early failure.
 * @returns The canonical `NodeFailedRecord` and a detached append-before-apply plan.
 * @throws TypeError when `state` or `failure` isn't of its exact class, or a carrier has the wrong
 *   type.
 * @throws RangeError when the failure has no node id, or the state or the executing node isn't
 *   eligible to fail.
 */
export function prepareNodeFailed(
  state: RunState,
  failure: Failure,
  options: OutcomeCarriers = {},
): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareNodeFailed expects a RunState");
  if (!isExactly(failure, Failure)) throw new TypeError("prepareNodeFailed expects a Failure");
  const record = new NodeFailedRecord(failure, {
    evidence: options.evidence ?? null,
    forwardCapture: options.forwardCapture ?? null,
  });
  requireInitializedNonterminal(state);
  return prepareFailureTransition(state, record);
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
  if (isExactly(record, NodeRoutedRecord)) {
    validateRouteRecord(state, record);
    return prepareRouteTransition(state, record, false);
  }
  if (isExactly(record, NodeExpandedRecord)) {
    if (typeof options.growBuilder !== "function") {
      throw new TypeError("NodeExpandedRecord replay requires a callable grow_builder");
    }
    return prepareGrowthRecord(state, record, options.growBuilder);
  }
  if (isExactly(record, NodeFailedRecord)) return prepareFailureTransition(state, record);
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
  // An exhaustive codec invariant: the record was decoded from this very text.
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
 * Freeze one fully validated route delta and its canonical public record.
 *
 * `deriveTerminalFailure` is set for a live route, which derives the terminal failure its skip
 * closure first proves; a replayed route checks the one its record carries instead.
 */
function prepareRouteTransition(
  state: RunState,
  record: NodeRoutedRecord,
  deriveTerminalFailure: boolean,
): PreparedTransition {
  validateNodeEvidence(state, record.nodeId, record.evidence);
  const route = prepareRouteDelta(state, record, deriveTerminalFailure);
  const causalFailure = route.causalFailure;
  let delta = route.delta;
  // The record keeps its terminal failure through `retainFailure`, which rebuilds each origin as a
  // plain `OriginRef`: the wire's single-key identity shape.
  const recordForWire = new NodeRoutedRecord(record.nodeId, record.handle, {
    terminalFailure: causalFailure,
    evidence: record.evidence,
    forwardCapture: record.forwardCapture,
  });
  const runId = state.runId;
  // Established by the initialized-state check.
  if (typeof runId !== "string") throw new RangeError("run state is not initialized");
  const envelopeJson = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, runId, state.nextSequence, recordForWire));
  const envelope = decodeRecord(envelopeJson);
  const canonical = envelope.record;
  if (!isExactly(canonical, NodeRoutedRecord)) {
    throw new TypeError("node-route envelope decoded to a different record kind");
  }
  // An exhaustive codec invariant: the record was decoded from this very text.
  if (encodeRecord(envelope) !== envelopeJson) throw new RangeError("NodeRoutedRecord is not in canonical wire form");
  if (causalFailure !== null) {
    if (canonical.terminalFailure === null) throw new RangeError("causal route lost its terminal failure");
    delta = Object.freeze({ ...delta, terminalUnreachableFailure: canonical.terminalFailure });
  }
  const terminal = prepareTerminalDelta(state, {
    completingNodeId: delta.nodeId,
    newlyReadyIds: delta.newlyReadyIds,
    pendingCommitTarget: null,
    pendingValueJson: null,
    terminalUnreachableFailure: delta.terminalUnreachableFailure,
  });

  const startId = state.flow.startId;
  const endId = state.flow.endId;
  const plan: RoutePlan = Object.freeze({
    kind: "route",
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
    }),
    delta,
    terminal,
  });
  const observers: RunEvent[] = [
    new NodeRouted({ nodeId: delta.nodeId, handle: delta.handle }),
    ...delta.skippedNodeIds.map((nodeId) => new NodeSkipped({ nodeId })),
  ];
  if (terminal !== null) observers.push(terminalObserver(terminal));
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, observers);
}

/** Freeze one validated terminal failure and its canonical public record. */
function prepareFailureTransition(state: RunState, record: NodeFailedRecord): PreparedTransition {
  validateFailureRecord(state, record);
  const runId = state.runId;
  // Established by the initialized-state check.
  if (typeof runId !== "string") throw new RangeError("run state is not initialized");
  // The record already holds its failure through `retainFailure`, which rebuilds each origin as a
  // plain `OriginRef`: the wire's single-key identity shape.
  const envelopeJson = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, runId, state.nextSequence, record));
  const envelope = decodeRecord(envelopeJson);
  const canonical = envelope.record;
  if (!isExactly(canonical, NodeFailedRecord)) {
    throw new TypeError("node-failure envelope decoded to a different record kind");
  }
  // An exhaustive codec invariant: the record was decoded from this very text.
  if (encodeRecord(envelope) !== envelopeJson) throw new RangeError("NodeFailedRecord is not in canonical wire form");

  const failure = canonical.failure;
  const startId = state.flow.startId;
  const endId = state.flow.endId;
  const plan: FailurePlan = Object.freeze({
    kind: "failure",
    originatingState: state,
    originatingFlow: state.flow,
    envelopeJson,
    startId,
    startNode: state.flow.nodes.get(startId)!,
    endId,
    endNode: state.flow.nodes.get(endId) ?? null,
    traceDelta: prepareRecordTrace(state, envelope, { terminalSucceeded: false, terminalFailure: failure }),
    failure,
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, failureObservers(failure));
}

/**
 * Prepare one canonical flat graph expansion without mutating live state.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight.
 * @param nodeId The exact executing spawner id, including a dynamic LOOP driver clone id.
 * @param grow The seed-first expansion description and its dynamic-overlay prune set.
 * @param growBuilder The injected compile-owned interpreter from a copy of the spawner and the
 *   canonical seed to one self-contained `Flow` fragment.
 * @param options.evidence Frozen visible evidence captured during this exact evaluation.
 * @param options.forwardCapture The canonical binding capture, required exactly when learning.
 * @returns The canonical `NodeExpandedRecord` and its detached append-before-apply plan.
 * @throws GrowthRejected when the node-produced prune set or seed, the returned fragment, the
 *   growth budget or the requested prune policy is invalid; `original` holds the exact error.
 * @throws TypeError when a carrier or the builder has the wrong type, or the builder doesn't
 *   return a `Flow`.
 * @throws RangeError when the live state, the spawner, the topology or another host invariant is
 *   invalid.
 */
export function prepareExpanded(
  state: RunState,
  nodeId: string,
  grow: Grow,
  growBuilder: GrowBuilder,
  options: OutcomeCarriers = {},
): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareExpanded expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  if (!nodeId) throw new RangeError("node_id must be non-empty");
  if (!isExactly(grow, Grow)) throw new TypeError("grow must be a Grow");
  if (typeof growBuilder !== "function") throw new TypeError("grow_builder must be callable");
  // Python's frozenset check has no counterpart: `Grow` always copies `prune` into a new `Set`.
  for (const pruneId of grow.prune as ReadonlySet<unknown>) {
    if (typeof pruneId !== "string") raiseGrowthRejected(new TypeError("Grow.prune must contain only strings"));
  }
  for (const pruneId of grow.prune) {
    if (!pruneId) raiseGrowthRejected(new RangeError("Grow.prune node IDs must be non-empty"));
  }
  requireInitializedNonterminal(state);
  requireExpandingNode(state, nodeId);

  let seedWire: JsonValue;
  let builderSeed: JsonValue;
  try {
    // Live seeds only, like steps: replay folds a recorded seed without the process limit.
    rejectDeepNesting(grow.seed, "growth seed");
    [seedWire, builderSeed] = canonicalSeed(grow.seed);
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) raiseGrowthRejected(error);
    throw error;
  }
  const fragment = buildGrowthFragment(state.flow.nodes.get(nodeId)!, builderSeed, growBuilder);
  const validatedFragment = validateGrowthFragment(state, fragment);
  const digest = expansionDigest(fragment);
  return prepareGrowthTransition(
    state,
    new NodeExpandedRecord(nodeId, seedWire, [...grow.prune].sort(compareCodeUnits), digest, {
      evidence: options.evidence ?? null,
      forwardCapture: options.forwardCapture ?? null,
    }),
    fragment,
    { actualDigest: digest, validatedFragment },
  );
}

/** The new node ids and the new edge ids of a validated fragment, in fragment order. */
type ValidatedFragment = readonly [readonly string[], readonly string[]];

/** Rebuild and verify one historical expansion through the live builder path. */
function prepareGrowthRecord(
  state: RunState,
  record: NodeExpandedRecord,
  growBuilder: GrowBuilder,
): PreparedTransition {
  validateGrowthRecord(state, record);
  try {
    const [seedWire, builderSeed] = canonicalSeed(record.seed);
    if (logJson(seedWire) !== logJson(record.seed)) {
      throw new RangeError("NodeExpandedRecord seed is not in canonical wire form");
    }
    const fragment = buildGrowthFragment(state.flow.nodes.get(record.spawnerId)!, builderSeed, growBuilder);
    const validatedFragment = validateGrowthFragment(state, fragment);
    const actualDigest = expansionDigest(fragment);
    if (actualDigest !== record.topologyDigest) {
      throw new RangeError(
        "NodeExpandedRecord topology digest mismatch: " +
          `recorded ${reprText(record.topologyDigest)}, rebuilt ${reprText(actualDigest)}`,
      );
    }
    return prepareGrowthTransition(state, record, fragment, { actualDigest, validatedFragment });
  } catch (error) {
    // A historical record is not a live node outcome: its rejection is a plain corrupt-log error.
    if (error instanceof GrowthRejected) throw error.original;
    throw error;
  }
}

/** Freeze one validated fragment, its final topology and the canonical public record. */
function prepareGrowthTransition(
  state: RunState,
  record: NodeExpandedRecord,
  fragment: Flow,
  options: { actualDigest: string; validatedFragment: ValidatedFragment },
): PreparedTransition {
  validateGrowthRecord(state, record);
  const { actualDigest: digest, validatedFragment: validated } = options;
  if (digest !== record.topologyDigest) {
    throw new RangeError("NodeExpandedRecord topology digest disagrees with its fragment");
  }
  const { delta, topology } = prepareGrowthTopology(state, record, fragment, validated);

  const runId = state.runId;
  // Established by the initialized-state check.
  if (typeof runId !== "string") throw new RangeError("run state is not initialized");
  const envelopeJson = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, runId, state.nextSequence, record));
  const envelope = decodeRecord(envelopeJson);
  const canonical = envelope.record;
  if (!isExactly(canonical, NodeExpandedRecord)) {
    throw new TypeError("node-expansion envelope decoded to a different record kind");
  }
  if (encodeRecord(envelope) !== envelopeJson) throw new RangeError("NodeExpandedRecord is not in canonical wire form");
  if (state.learnable && isExactly(canonical.forwardCapture, ForwardCapture)) {
    for (const connection of fragment.inputConnections) {
      if (selectSeedRoute(canonical.forwardCapture, connection) === null) {
        raiseGrowthRejected(
          new RangeError(`seed connection ${writeAsText([...connection.seedPath])} was not captured`),
        );
      }
    }
  }
  const startId = state.flow.startId;
  const endId = state.flow.endId;
  const plan: GrowthPlan = Object.freeze({
    kind: "growth",
    originatingState: state,
    originatingFlow: state.flow,
    envelopeJson,
    startId,
    startNode: state.flow.nodes.get(startId)!,
    endId,
    endNode: state.flow.nodes.get(endId) ?? null,
    traceDelta: prepareRecordTrace(state, envelope, {
      inputConnections: fragment.inputConnections,
      childBoundaries: fragment.childBoundaries,
      retiredIds: delta.retiredIds,
    }),
    delta,
    topology,
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, [growthObserver(canonical)]);
}

/** Validate all backward data and prepare its detached state before append. */
function prepareBackwardTransition(state: RunState, record: RunRecord): PreparedTransition {
  requireInitializedTerminalBackward(state);
  const envelopeJson = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, state.runId!, state.nextSequence, record));
  const envelope = decodeRecord(envelopeJson);
  if (encodeRecord(envelope) !== envelopeJson) throw new RangeError("backward record is not in canonical wire form");
  // `prepareRecord` routes only the exact backward record classes here.
  const nextBackwardState = reduceBackwardRecord(state.trace!, state.backwardState, envelope.record as BackwardRecord);
  const startId = state.flow.startId;
  const endId = state.flow.endId;
  const plan: BackwardPlan = Object.freeze({
    kind: "backward",
    originatingState: state,
    originatingFlow: state.flow,
    envelopeJson,
    startId,
    startNode: state.flow.nodes.get(startId)!,
    endId,
    endNode: state.flow.nodes.get(endId) ?? null,
    traceDelta: prepareRecordTrace(state, envelope),
    originatingBackwardState: state.backwardState,
    nextBackwardState,
  });
  return new PreparedTransition(envelope.record, envelope.runId, envelope.seq, plan, []);
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
    case "route":
      observers = commitRoute(log, state, prepared);
      break;
    case "failure":
      observers = commitFailure(log, state, prepared);
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
 * Silently fold one complete historical prefix through the live reducer path.
 *
 * Every envelope is canonicalized and checked for one contiguous run before any transition is
 * reduced. `commitPrepared` then receives a read-only cursor, so replay runs the same apply
 * functions as live dispatch without appending to historical authority or exposing observers.
 *
 * @param state A pristine state around the graph that authored the run.
 * @param envelopes A non-empty sequence-zero historical prefix beginning with `RunStarted`.
 * @param options.growBuilder The injected builder that rebuilds and verifies growth.
 * @returns `state`, after every historical record has been applied.
 * @throws TypeError when an argument has the wrong type.
 * @throws RangeError when the state isn't pristine, the prefix is empty, not contiguous, not one
 *   run, doesn't start with `RunStarted`, or a record isn't the exact next transition; a node-data
 *   rejection throws its original error.
 */
export function replayRecords(
  state: RunState,
  envelopes: readonly RecordEnvelope[],
  options: { growBuilder: GrowBuilder },
): RunState {
  if (!isExactly(state, RunState)) throw new TypeError("replayRecords expects a RunState");
  if (!Array.isArray(envelopes) || !envelopes.every((envelope) => envelope instanceof RecordEnvelope)) {
    throw new TypeError("replay envelopes must be an array of RecordEnvelope values");
  }
  const growBuilder = options.growBuilder;
  if (typeof growBuilder !== "function") throw new TypeError("grow_builder must be callable");
  requirePristine(state);
  if (envelopes.length === 0) throw new RangeError("cannot replay an empty run-record prefix");

  const canonical: string[] = [];
  let runId: string | null = null;
  for (const [expectedSequence, supplied] of envelopes.entries()) {
    const encoded = encodeRecord(supplied);
    const envelope = decodeRecord(encoded);
    if (envelope.seq !== expectedSequence) {
      throw new RangeError(`replay sequence mismatch: expected ${expectedSequence}, found ${envelope.seq}`);
    }
    if (runId === null) {
      runId = envelope.runId;
    } else if (envelope.runId !== runId) {
      throw new RangeError(`replay run_id mismatch: expected ${reprText(runId)}, found ${reprText(envelope.runId)}`);
    }
    canonical.push(encoded);
  }

  const first = decodeRecord(canonical[0]!);
  const start = first.record;
  if (!(start instanceof RunStartedRecord)) throw new RangeError("the first historical run record must be RunStarted");
  const cursor = new HistoricalLogCursor(canonical);
  const prepared = prepareRunStarted(state, {
    runId: first.runId,
    flowFingerprint: start.flowFingerprint,
    initialPool: decodePool(start.initialPool),
    startWiringOverride: start.startWiringOverride,
    effectiveLlmByStaticPath: start.effectiveLlmByStaticPath,
    numWorkers: start.numWorkers,
    memorySnapshots: start.memorySnapshots,
    learnable: start.learnable,
    memories: start.memories,
    memoryStacks: start.memoryStacks,
    defaults: start.defaults,
    callTimeLlm: start.callTimeLlm,
  });
  commitPrepared(cursor, state, prepared);

  for (const encoded of canonical.slice(1)) {
    const envelope = decodeRecord(encoded);
    let next: PreparedTransition;
    try {
      next = prepareRecord(state, envelope.record, { growBuilder });
    } catch (error) {
      // Python's node-data marker subclasses the original error's class, so a corrupt log reads
      // as that error there; replay reports the original here too.
      if (error instanceof NodeDataError) throw error.original;
      throw error;
    }
    commitPrepared(cursor, state, next);
  }

  if (cursor.nextSequence !== canonical.length) {
    throw new RangeError("replay did not consume the complete historical prefix");
  }
  return state;
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
      // A stop names every in-flight node, and an attempt stays in flight until its outcome or the
      // next resume, so an attempt retried here was covered exactly when it started before the
      // segment's last stop.
      for (const nodeId of delta.retryNodeIds) {
        if (state.lastStopSequence === null || state.acceptedStarts.get(nodeId)! >= state.lastStopSequence) {
          state.interruptedAttempts.set(nodeId, (state.interruptedAttempts.get(nodeId) ?? 0) + 1);
        }
      }
      state.lastStopSequence = null;
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
      state.lastStopSequence = prepared.expectedSequence;
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

/** Validate, append and install one route and its derived skip closure. */
function commitRoute(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "route") throw new TypeError("route commit requires a route plan");
  const record = prepared.record;
  if (!isExactly(record, NodeRoutedRecord)) throw new TypeError("route plan requires a NodeRoutedRecord");
  requirePreparedOrigin(state, plan);
  requireInitializedNonterminal(state);
  requireAuthorityCoordinates(log, state, prepared, true);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  requireOutcomeTraceScope(state, prepared);
  requireOutcomeTraceTerminal(state, prepared);
  const delta = plan.delta;
  if (!(record.nodeId === delta.nodeId && record.handle === delta.handle)) {
    throw new RangeError("route record and prevalidated delta disagree");
  }
  if (record.terminalFailure !== null) {
    if (delta.terminalUnreachableFailure !== record.terminalFailure) {
      throw new RangeError("route record and terminal-failure delta disagree");
    }
  } else if (delta.terminalUnreachableFailure !== state.terminalUnreachableFailure) {
    throw new RangeError("noncausal route changed terminal-failure authority");
  }
  const apply = materializeRoute(plan);

  appendAndVerify(log, prepared);
  state.nodeState.set(delta.nodeId, NodeState.TAKEN);
  for (const [edgeId, disposition] of delta.edgeUpdates) state.edgeState.set(edgeId, disposition);
  for (const nodeId of delta.skippedNodeIds) state.nodeState.set(nodeId, NodeState.SKIPPED);
  state.inFlight.delete(delta.nodeId);
  state.acceptedStarts.delete(delta.nodeId);
  forgetNodeSteps(state, delta.nodeId);
  for (const nodeId of delta.newlyReadyIds) state.ready.push(nodeId);
  state.terminalUnreachableFailure = delta.terminalUnreachableFailure;
  if (apply.terminal !== null) state.result = apply.terminal.result;
  state.nextSequence = prepared.expectedSequence + 1;
  return apply.observers;
}

/** Validate, append and install one immediately terminal node failure. */
function commitFailure(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "failure") throw new TypeError("failure commit requires a failure plan");
  const record = prepared.record;
  if (!isExactly(record, NodeFailedRecord)) throw new TypeError("failure plan requires a NodeFailedRecord");
  requirePreparedOrigin(state, plan);
  requireInitializedNonterminal(state);
  requireAuthorityCoordinates(log, state, prepared, true);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  requireOutcomeTraceScope(state, prepared);
  requireOutcomeTraceTerminal(state, prepared);
  if (record.failure !== plan.failure) throw new RangeError("failure record and prevalidated authority disagree");
  const nodeId: unknown = plan.failure.nodeId;
  if (typeof nodeId !== "string" || !nodeId) throw new RangeError("failure plan has no executing node");
  const apply = materializeFailure(plan.failure);

  appendAndVerify(log, prepared);
  state.nodeState.set(nodeId, NodeState.TAKEN);
  state.inFlight.delete(nodeId);
  state.acceptedStarts.delete(nodeId);
  forgetNodeSteps(state, nodeId);
  state.result = apply.result;
  state.nextSequence = prepared.expectedSequence + 1;
  return apply.observers;
}

/** Validate, append and install one fully materialized flat expansion. */
function commitGrowth(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "growth") throw new TypeError("growth commit requires a growth plan");
  const record = prepared.record;
  if (!isExactly(record, NodeExpandedRecord)) throw new TypeError("growth plan requires a NodeExpandedRecord");
  requirePreparedOrigin(state, plan);
  requireInitializedNonterminal(state);
  requireAuthorityCoordinates(log, state, prepared, true);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  requireOutcomeTraceScope(state, prepared);
  requireOutcomeTraceTerminal(state, prepared);
  const delta = plan.delta;
  if (
    !(
      record.spawnerId === delta.spawnerId &&
      sameItems(record.pruneIds, delta.pruneIds) &&
      record.topologyDigest === delta.topologyDigest
    )
  ) {
    throw new RangeError("growth record and prevalidated delta disagree");
  }
  const apply = materializeGrowth(plan);

  appendAndVerify(log, prepared);
  applyGrowth(state, delta, plan.topology, apply.wiring);
  state.nextSequence = prepared.expectedSequence + 1;
  return apply.observers;
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

/** Verify the complete prepared authority, append, then replace the attempt state. */
function commitBackward(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "backward") throw new TypeError("backward commit requires its exact plan");
  requirePreparedOrigin(state, plan);
  if (state.backwardState !== plan.originatingBackwardState) {
    throw new RangeError("prepared backward transition has stale attempt authority");
  }
  requireInitializedTerminalBackward(state);
  requireAuthorityCoordinates(log, state, prepared, true);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  const envelope = decodeRecord(plan.envelopeJson);
  // The text is the plan's own backward record, and the reducer refuses any other class.
  const expected = reduceBackwardRecord(state.trace!, state.backwardState, envelope.record as BackwardRecord);
  if (!isExactly(plan.nextBackwardState, RuntimeBackwardState)) {
    throw new TypeError("backward plan requires an exact next state");
  }
  // The reducer just validated `expected`, and only `expected` is kept, so equality is enough;
  // checking all history here would cost O(N) per record.
  if (!sameRuntimeBackwardState(expected, plan.nextBackwardState)) {
    throw new RangeError("backward plan state disagrees with its canonical record");
  }
  appendAndVerify(log, prepared);
  state.backwardState = expected;
  state.nextSequence = prepared.expectedSequence + 1;
  return [];
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
    throw new RangeError("RunStarted requires state.run_id to be null");
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
    isExactly(state.interruptedAttempts, Map) &&
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
  if (
    [...state.interruptedAttempts].some(
      ([nodeId, count]) => typeof nodeId !== "string" || !Number.isSafeInteger(count) || count < 1,
    )
  ) {
    throw new RangeError("interrupted attempt counts must be positive integers by node");
  }
  const lastStop = state.lastStopSequence;
  if (lastStop !== null && !(Number.isSafeInteger(lastStop) && 1 <= lastStop && lastStop < state.nextSequence)) {
    throw new RangeError("last stop sequence must be null or an earlier accepted sequence");
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

/** Require the accepted retained terminal, independently from memory declarations. */
function requireInitializedTerminalBackward(state: RunState): void {
  requireInitialized(state);
  if (!(state.result instanceof RunSucceeded || state.result instanceof RunFailed)) {
    throw new RangeError("backward records require an accepted terminal result");
  }
  if (state.learnable !== true || state.trace === null || state.trace.terminal === null) {
    throw new RangeError("backward records require recorded learning and terminal trace authority");
  }
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

/** Require an existing unknown router in the ordered in-flight frontier. */
function requireRoutingNode(state: RunState, nodeId: string): void {
  if (!nodeId) throw new RangeError("NodeRouted node_id must be non-empty");
  if (!state.flow.nodes.has(nodeId)) {
    throw new RangeError(`NodeRouted node ${reprText(nodeId)} does not exist in the flow`);
  }
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`NodeRouted node ${reprText(nodeId)} must have UNKNOWN disposition`);
  }
  if (!state.inFlight.has(nodeId)) throw new RangeError(`NodeRouted node ${reprText(nodeId)} is not in-flight`);
}

/** Require an existing unknown node in the ordered in-flight frontier. */
function requireFailingNode(state: RunState, nodeId: string): void {
  if (!nodeId) throw new RangeError("NodeFailed node_id must be non-empty");
  if (!state.flow.nodes.has(nodeId)) {
    throw new RangeError(`NodeFailed node ${reprText(nodeId)} does not exist in the flow`);
  }
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`NodeFailed node ${reprText(nodeId)} must have UNKNOWN disposition`);
  }
  if (!state.inFlight.has(nodeId)) throw new RangeError(`NodeFailed node ${reprText(nodeId)} is not in-flight`);
}

/** Require an existing unknown spawner in the ordered in-flight frontier. */
function requireExpandingNode(state: RunState, nodeId: string): void {
  if (!nodeId) throw new RangeError("NodeExpanded node_id must be non-empty");
  const node = state.flow.nodes.get(nodeId);
  if (node === undefined) throw new RangeError(`NodeExpanded node ${reprText(nodeId)} does not exist in the flow`);
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`NodeExpanded node ${reprText(nodeId)} must have UNKNOWN disposition`);
  }
  if (!state.inFlight.has(nodeId)) throw new RangeError(`NodeExpanded node ${reprText(nodeId)} is not in-flight`);
  if (node.isSpawner !== true) throw new RangeError(`NodeExpanded node ${reprText(nodeId)} is not a graph spawner`);
}

/** Validate a durable expansion's exact carriers and its executing spawner. */
function validateGrowthRecord(state: RunState, record: NodeExpandedRecord): void {
  const spawnerId: unknown = record.spawnerId;
  const pruneIds: unknown = record.pruneIds;
  const digest: unknown = record.topologyDigest;
  if (typeof spawnerId !== "string") throw new TypeError("NodeExpandedRecord spawner_id must be a string");
  if (!spawnerId) throw new RangeError("NodeExpandedRecord spawner_id must be non-empty");
  if (!Array.isArray(pruneIds) || pruneIds.some((nodeId) => typeof nodeId !== "string")) {
    throw new TypeError("NodeExpandedRecord prune_ids must be an array of strings");
  }
  const ids = pruneIds as string[];
  if (ids.some((nodeId) => !nodeId)) throw new RangeError("NodeExpandedRecord prune IDs must be non-empty");
  if (!sameItems(ids, [...new Set(ids)].sort(compareCodeUnits))) {
    throw new RangeError("NodeExpandedRecord prune IDs must be sorted and unique");
  }
  if (typeof digest !== "string") throw new TypeError("NodeExpandedRecord topology_digest must be a string");
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new RangeError("NodeExpandedRecord topology_digest must be lowercase SHA-256");
  }
  requireExpandingNode(state, spawnerId);
  validateNodeEvidence(state, spawnerId, record.evidence);
}

/** Call the injected builder with a copy of the spawner and return a copy of its fragment. */
function buildGrowthFragment(spawner: Node, builderSeed: JsonValue, growBuilder: GrowBuilder): Flow {
  const built: unknown = growBuilder(copyGraph(spawner), builderSeed);
  if (!(built instanceof Flow)) throw new TypeError("grow_builder must return a Flow");
  return copyGraph(built);
}

/** Validate one growth and build the detached change commit applies in place. */
function prepareGrowthTopology(
  state: RunState,
  record: NodeExpandedRecord,
  fragment: Flow,
  validatedFragment: ValidatedFragment,
): { delta: GrowthDelta; topology: GrowthTopology } {
  const [newNodeIds, newEdgeIds] = validatedFragment;
  const budget = state.runDefaults.maxTotalNodes;
  if (state.flow.nodes.size + newNodeIds.length > budget) {
    raiseGrowthRejected(
      new RangeError(`expansion exceeded node budget (${budget}) at spawner ${reprText(record.spawnerId)}`),
    );
  }
  const derivedDepth = validatedGrowthDepth(state, record.spawnerId);
  const index = growthIndex(state);
  validateGrowthPrune(state, record.spawnerId, record.pruneIds, new Set(newNodeIds));
  const [retired, deferredPrune] = retiredIds(state, record.spawnerId, record.pruneIds, index.grown);
  validatePruneDependencies(state, fragment, retired, index);

  const sortedRetiredIds = [...retired].sort(compareCodeUnits);
  const retiredEdges = new Map<string, Edge>();
  for (const nodeId of sortedRetiredIds) {
    for (const edge of [...state.flow.outgoing(nodeId), ...state.flow.incoming(nodeId)]) {
      if (!retiredEdges.has(edge.id)) retiredEdges.set(edge.id, edge);
    }
  }
  validateGrowthRedirects(state, fragment, record.spawnerId, retired, new Set(retiredEdges.keys()), index);

  /** Return the nearest owner at or above `owner` that this growth keeps. */
  const survivingOwner = (owner: string | null): string | null => {
    while (owner !== null && retired.has(owner)) owner = state.growthOwner.get(owner) ?? null;
    return owner;
  };

  // A kept subtree under a retired node, and a self-pruning spawner's new nodes, move to the
  // nearest owner that survives this transition.
  const ownerChanges: (readonly [string, string | null])[] = [];
  for (const retiredId of sortedRetiredIds) {
    for (const nodeId of index.grown.get(retiredId) ?? []) {
      if (!retired.has(nodeId)) ownerChanges.push([nodeId, survivingOwner(retiredId)]);
    }
  }
  const newOwner = survivingOwner(record.spawnerId);
  for (const nodeId of newNodeIds) ownerChanges.push([nodeId, newOwner]);

  const depth: (readonly [string, number])[] = [];
  if (derivedDepth !== null) {
    for (const [nodeId, node] of fragment.nodes) {
      if (node.isSpawner === true || node.commitAs === record.spawnerId) depth.push([nodeId, derivedDepth]);
    }
  }

  const delta: GrowthDelta = Object.freeze({
    spawnerId: record.spawnerId,
    pruneIds: record.pruneIds,
    retiredIds: Object.freeze(sortedRetiredIds),
    topologyDigest: record.topologyDigest,
    fragmentStartId: fragment.startId,
    fragmentEndId: fragment.endId,
    newNodeIds,
    newEdgeIds,
    derivedDepth,
  });
  const topology: GrowthTopology = Object.freeze({
    nodes: Object.freeze([...fragment.nodes]),
    edges: Object.freeze([...fragment.edges]),
    wiring: Object.freeze([...fragment.wiring].map(([nodeId, wiring]) => [nodeId, new Map(wiring)] as const)),
    retiredEdges: Object.freeze([...retiredEdges.values()]),
    depth: Object.freeze(depth),
    growthOwner: Object.freeze(ownerChanges),
    deferredPrune,
    names: Object.freeze(
      [...fragment.nodes].map(
        ([nodeId, node]) => [nodeId, nodeNames(nodeId, node, fragment.wiring.get(nodeId)!)] as const,
      ),
    ),
  });
  return { delta, topology };
}

/**
 * Require a collision-free, self-contained, acyclic fragment; return its new node and edge ids.
 *
 * The work is proportional to the fragment: live nodes are looked up, not rescanned, and only the
 * live producers the fragment's wiring reads have their output contracts checked. A malformed live
 * contract stays a host invariant error outside `GrowthRejected`.
 */
function validateGrowthFragment(state: RunState, fragment: Flow): ValidatedFragment {
  if (!isExactly(fragment.nodes, Map)) raiseGrowthRejected(new TypeError("growth fragment nodes must be a Map"));
  if (!Array.isArray(fragment.edges)) raiseGrowthRejected(new TypeError("growth fragment edges must be an array"));
  if (!isExactly(fragment.wiring, Map)) raiseGrowthRejected(new TypeError("growth fragment wiring must be a Map"));
  const fragmentStartId: unknown = fragment.startId;
  const fragmentEndId: unknown = fragment.endId;
  if (typeof fragmentStartId !== "string" || !fragmentStartId) {
    raiseGrowthRejected(new RangeError("growth fragment start_id must be a non-empty string"));
  }
  if (typeof fragmentEndId !== "string" || !fragmentEndId) {
    raiseGrowthRejected(new RangeError("growth fragment end_id must be a non-empty string"));
  }
  if (!fragment.nodes.has(fragmentStartId)) {
    raiseGrowthRejected(new RangeError("growth fragment start_id is missing from its nodes"));
  }
  if (!fragment.nodes.has(fragmentEndId)) {
    raiseGrowthRejected(new RangeError("growth fragment end_id is missing from its nodes"));
  }

  const newNodeIds: string[] = [];
  for (const [nodeId, node] of fragment.nodes as Map<unknown, unknown>) {
    if (typeof nodeId !== "string") raiseGrowthRejected(new TypeError("growth fragment node IDs must be strings"));
    if (!nodeId) raiseGrowthRejected(new RangeError("growth fragment node IDs must be non-empty"));
    if (!(node instanceof Node)) {
      raiseGrowthRejected(new TypeError(`growth fragment node ${reprText(nodeId)} must be a Node`));
    }
    if (node.id !== nodeId) {
      raiseGrowthRejected(
        new RangeError(`growth fragment node key ${reprText(nodeId)} does not equal node.id ${reprValue(node.id)}`),
      );
    }
    if (state.flow.nodes.has(nodeId)) {
      raiseGrowthRejected(new RangeError(`growth fragment node ${reprText(nodeId)} collides with live topology`));
    }
    newNodeIds.push(nodeId);
  }

  if (!sameIds(fragment.nodes, fragment.wiring.keys())) {
    raiseGrowthRejected(new RangeError("growth fragment wiring keys must exactly equal fragment nodes"));
  }
  try {
    validateGrowthMetadata(fragment);
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) raiseGrowthRejected(error);
    throw error;
  }
  // Python's `ChainMap(fragment.nodes, state.flow.nodes)`: a lookup, not a merged copy.
  const combinedNodeIds = { has: (id: string): boolean => fragment.nodes.has(id) || state.flow.nodes.has(id) };
  const fragmentProducerContracts = new Map<string, OutputContract>();
  for (const [nodeId, node] of fragment.nodes) {
    const contract = node.outputContract;
    if (contract === null) continue;
    try {
      outputContractType(contract);
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) raiseGrowthRejected(error);
      throw error;
    }
    fragmentProducerContracts.set(nodeId, contract);
  }
  const liveProducerContracts = new Map<string, OutputContract | null>();

  /** Return a producer's output contract, checking a live one on first read. */
  const producerContract = (nodeId: string): OutputContract | null => {
    const fragmentContract = fragmentProducerContracts.get(nodeId);
    if (fragmentContract !== undefined) return fragmentContract;
    const live = state.flow.nodes.get(nodeId);
    if (live === undefined) return null;
    if (!liveProducerContracts.has(nodeId)) {
      const contract = live.outputContract;
      outputContractType(contract);
      liveProducerContracts.set(nodeId, contract);
    }
    return liveProducerContracts.get(nodeId)!;
  };

  for (const [nodeId, node] of fragment.nodes) {
    for (const [attribute, target] of [
      ["commit_as", node.commitAs],
      ["origin_id", node.originId],
    ] as [string, unknown][]) {
      if (target !== null && typeof target !== "string") {
        raiseGrowthRejected(
          new TypeError(`growth fragment node ${reprText(nodeId)} ${attribute} must be a string or null`),
        );
      }
      if (target === "") {
        raiseGrowthRejected(new RangeError(`growth fragment node ${reprText(nodeId)} ${attribute} must be non-empty`));
      }
      if (target !== null && !combinedNodeIds.has(target)) {
        raiseGrowthRejected(
          new RangeError(
            `growth fragment node ${reprText(nodeId)} ${attribute} target ${reprText(target)} does not exist`,
          ),
        );
      }
    }
    const skippedTarget: unknown = node.skippedOutputTarget();
    try {
      validateSkippedOutputTarget(combinedNodeIds, nodeId, skippedTarget);
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) raiseGrowthRejected(error);
      throw error;
    }
    const wiring: unknown = fragment.wiring.get(nodeId);
    if (!isExactly(wiring, Map) || [...wiring.keys()].some((key) => typeof key !== "string")) {
      raiseGrowthRejected(new TypeError(`growth fragment wiring for ${reprText(nodeId)} must be a string-keyed Map`));
    }
    const params: unknown = node.params ?? [];
    if (!Array.isArray(params)) {
      raiseGrowthRejected(new TypeError(`growth fragment node ${reprText(nodeId)} params must be an array`));
    }
    const declared = new Map<string, boolean>();
    for (const param of params as unknown[]) {
      if (!isExactly(param, Parameter)) {
        raiseGrowthRejected(
          new TypeError(
            `growth fragment node ${reprText(nodeId)} params must contain Parameter values, got ${jsTypeName(param)}`,
          ),
        );
      }
      const name: unknown = param.name;
      if (typeof name !== "string" || !name) {
        raiseGrowthRejected(new RangeError(`growth fragment node ${reprText(nodeId)} has an invalid param name`));
      }
      if (declared.has(name)) {
        raiseGrowthRejected(
          new RangeError(`growth fragment node ${reprText(nodeId)} has duplicate param ${reprText(name)}`),
        );
      }
      declared.set(name, Boolean(param.required));
    }
    const reserved: unknown = node.reservedWiringKeys();
    if (!isExactly(reserved, Set) || [...reserved].some((key) => typeof key !== "string" || !key)) {
      raiseGrowthRejected(new TypeError(`growth fragment node ${reprText(nodeId)} reserved wiring keys are invalid`));
    }
    const reservedKeys = reserved as Set<string>;
    const actual = new Set((wiring as Map<string, unknown>).keys());
    const extras = [...actual].filter((key) => !declared.has(key) && !reservedKeys.has(key));
    const missingReserved = [...reservedKeys].filter((key) => !actual.has(key));
    const missingRequired = [...declared]
      .filter(([name, required]) => required && !actual.has(name))
      .map(([name]) => name);
    if (extras.length > 0 || missingReserved.length > 0 || missingRequired.length > 0) {
      raiseGrowthRejected(
        new RangeError(
          `growth fragment node ${reprText(nodeId)} wiring parity failed: ` +
            `extra=${writeAsText(extras.sort(compareCodeUnits))}, ` +
            `missing_reserved=${writeAsText(missingReserved.sort(compareCodeUnits))}, ` +
            `missing_required=${writeAsText(missingRequired.sort(compareCodeUnits))}`,
        ),
      );
    }
    validateGrowthWiringRefs(
      nodeId,
      node,
      wiring as Map<string, unknown>,
      combinedNodeIds,
      producerContract,
      reservedKeys,
    );
  }

  // Every live edge has a disposition, so the edge-state keys are the live edge ids.
  const newEdgeIds: string[] = [];
  const seenEdgeIds = new Set<string>();
  for (const edge of fragment.edges as unknown[]) {
    if (!isExactly(edge, Edge)) {
      raiseGrowthRejected(new TypeError("growth fragment edges must contain only Edge values"));
    }
    const edgeId: unknown = edge.id;
    if (typeof edgeId !== "string" || !edgeId) {
      raiseGrowthRejected(new RangeError("growth fragment edge IDs must be non-empty strings"));
    }
    if (seenEdgeIds.has(edgeId)) {
      raiseGrowthRejected(new RangeError(`growth fragment edge ID ${reprText(edgeId)} is duplicated`));
    }
    if (state.edgeState.has(edgeId)) {
      raiseGrowthRejected(new RangeError(`growth fragment edge ${reprText(edgeId)} collides with live topology`));
    }
    for (const [attribute, value] of [
      ["source_handle", edge.sourceHandle],
      ["input_group", edge.inputGroup],
    ] as [string, unknown][]) {
      if (value !== null && typeof value !== "string") {
        raiseGrowthRejected(
          new TypeError(`growth fragment edge ${reprText(edgeId)} ${attribute} must be a string or null`),
        );
      }
      if (value === "") {
        raiseGrowthRejected(new RangeError(`growth fragment edge ${reprText(edgeId)} ${attribute} must be non-empty`));
      }
    }
    for (const [attribute, value] of [
      ["optional", edge.optional],
      ["ordering", edge.ordering],
    ] as [string, unknown][]) {
      if (typeof value !== "boolean") {
        raiseGrowthRejected(new TypeError(`growth fragment edge ${reprText(edgeId)} ${attribute} must be a bool`));
      }
    }
    const from: unknown = edge.from;
    const to: unknown = edge.to;
    if (typeof from !== "string" || !combinedNodeIds.has(from)) {
      raiseGrowthRejected(
        new RangeError(`growth fragment edge ${reprText(edgeId)} has unknown source ${reprValue(from)}`),
      );
    }
    if (typeof to !== "string" || !fragment.nodes.has(to)) {
      raiseGrowthRejected(
        new RangeError(`growth fragment edge ${reprText(edgeId)} must target a fragment node, found ${reprValue(to)}`),
      );
    }
    seenEdgeIds.add(edgeId);
    newEdgeIds.push(edgeId);
  }

  validateGrowthReachability(fragment);
  // Fragment edges target only fresh fragment nodes and live edges can't reach them, so any new
  // cycle lies wholly inside the fragment.
  rejectGrowthCycle(new Set(fragment.nodes.keys()), fragment.edges.filter((edge) => fragment.nodes.has(edge.from)));
  return Object.freeze([Object.freeze(newNodeIds), Object.freeze(newEdgeIds)] as const);
}

/** Python's `repr` of a value a message names: quoted text, else the value's string form. */
function reprValue(value: unknown): string {
  return typeof value === "string" ? reprText(value) : String(value);
}

/** Require every fragment node to be driven from its sole scheduled entry. */
function validateGrowthReachability(fragment: Flow): void {
  const adjacency = new Map<string, string[]>([...fragment.nodes.keys()].map((nodeId) => [nodeId, []]));
  for (const edge of fragment.edges) adjacency.get(edge.from)?.push(edge.to);
  const reachable = new Set<string>();
  const pending = [fragment.startId];
  while (pending.length > 0) {
    const nodeId = pending.pop()!;
    if (reachable.has(nodeId)) continue;
    reachable.add(nodeId);
    for (const next of adjacency.get(nodeId)!) pending.push(next);
  }
  const unreachable = [...fragment.nodes.keys()].filter((nodeId) => !reachable.has(nodeId)).sort(compareCodeUnits);
  if (unreachable.length > 0) {
    raiseGrowthRejected(
      new RangeError(`growth fragment nodes are unreachable from its start: ${writeAsText(unreachable)}`),
    );
  }
}

/** Apply the closed reference rules to raw or carrier-wrapped fragment wiring. */
function validateGrowthWiringRefs(
  nodeId: string,
  node: Node,
  wiring: ReadonlyMap<string, unknown>,
  combinedNodeIds: { has(id: string): boolean },
  producerContract: (nodeId: string) => OutputContract | null,
  reserved: ReadonlySet<string>,
): void {
  for (const [param, source] of wiring) {
    const paths: string[] = [];
    for (const [, text] of iterSourceStrings(source)) {
      try {
        for (const path of exprRefsOf(text)) paths.push(path);
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        raiseGrowthRejected(
          new RangeError(
            `growth fragment node ${reprText(nodeId)} wiring ${reprText(param)} is invalid: ${error.message}`,
            { cause: error },
          ),
        );
      }
    }
    for (const path of paths) {
      const extraHeads = node.bindsPerItem === true && !reserved.has(param) ? ["item"] : [];
      let error = classifyPath(path, combinedNodeIds, new Set(), extraHeads);
      const parts = path.split(".");
      const contract =
        error === null && parts.length >= 2 && parts[1] === "output" ? producerContract(parts[0]!) : null;
      if (contract !== null) {
        // Fragments come from flows the loader already checked, where a CASE branch may narrow a
        // union so that a member-only field read is valid. That narrowing isn't recorded here, so
        // such reads are allowed.
        error = outputContractFieldError(contract, parts.slice(2), path, { partialUnionFields: true });
      }
      if (error !== null) {
        raiseGrowthRejected(
          new RangeError(`growth fragment node ${reprText(nodeId)} wiring ${reprText(param)}: ${error}`),
        );
      }
    }
  }
}

/** Reject a cycle among `nodeIds` as node-produced fragment data. */
function rejectGrowthCycle(nodeIds: ReadonlySet<string>, edges: readonly Edge[]): void {
  const adjacency = new Map<string, string[]>([...nodeIds].map((nodeId) => [nodeId, []]));
  const inDegree = new Map<string, number>([...nodeIds].map((nodeId) => [nodeId, 0]));
  for (const edge of edges) {
    if (!nodeIds.has(edge.from) || !nodeIds.has(edge.to)) {
      throw new RangeError(`growth edge ${reprText(edge.id)} has an unknown endpoint`);
    }
    adjacency.get(edge.from)!.push(edge.to);
    inDegree.set(edge.to, inDegree.get(edge.to)! + 1);
  }
  const ready = [...inDegree].filter(([, degree]) => degree === 0).map(([nodeId]) => nodeId);
  let visited = 0;
  while (ready.length > 0) {
    const nodeId = ready.pop()!;
    visited += 1;
    for (const successor of adjacency.get(nodeId)!) {
      const degree = inDegree.get(successor)! - 1;
      inDegree.set(successor, degree);
      if (degree === 0) ready.push(successor);
    }
  }
  if (visited !== nodeIds.size) {
    const stuck = [...inDegree].filter(([, degree]) => degree !== 0).map(([nodeId]) => nodeId).sort(compareCodeUnits);
    raiseGrowthRejected(new RangeError(`growth fragment creates a cycle involving ${writeAsText(stuck)}`));
  }
}

/** Derive the exact trait-driven growth depth for one expansion. */
function validatedGrowthDepth(state: RunState, spawnerId: string): number | null {
  const delta: unknown = state.flow.nodes.get(spawnerId)!.growDepthDelta;
  if (delta === null) return null;
  if (typeof delta !== "number" || !Number.isSafeInteger(delta) || delta < 0) {
    throw new RangeError(`spawner ${reprText(spawnerId)} grow_depth_delta must be null, 0, or a positive integer`);
  }
  const parentDepth: unknown = state.depth.has(spawnerId) ? state.depth.get(spawnerId) : 0;
  if (typeof parentDepth !== "number" || !Number.isSafeInteger(parentDepth) || parentDepth < 0) {
    throw new RangeError(`spawner ${reprText(spawnerId)} has an invalid parent depth`);
  }
  return parentDepth + delta;
}

/** Return the node ids one node names through its wiring and redirect fields. */
function nodeNames(nodeId: string, node: Node, wiring: ReadonlyMap<string, unknown>): readonly string[] {
  const names = new Set<string>();
  for (const target of [node.commitAs, node.originId, node.skippedOutputTarget()] as unknown[]) {
    if (typeof target === "string" && target !== nodeId) names.add(target);
  }
  for (const source of wiring.values()) {
    for (const [, text] of iterSourceStrings(source)) {
      let paths: string[];
      try {
        paths = exprRefsOf(text);
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        throw new RangeError(`node ${reprText(nodeId)} has invalid wiring: ${error.message}`, { cause: error });
      }
      for (const path of paths) names.add(path.split(".", 1)[0]!);
    }
  }
  return Object.freeze([...names]);
}

/**
 * Return `state.growthIndex`, building it from the live graph on first use.
 *
 * @internal Exported so tests can compare the kept index with a rebuild.
 */
export function growthIndex(state: RunState): GrowthIndex {
  if (state.growthIndex === null) {
    const names = new Map<string, readonly string[]>();
    for (const [nodeId, node] of state.flow.nodes) {
      names.set(nodeId, nodeNames(nodeId, node, state.flow.wiring.get(nodeId) ?? new Map()));
    }
    const referrers = new Map<string, Set<string>>();
    for (const [nodeId, named] of names) {
      for (const name of named) setDefault(referrers, name, () => new Set()).add(nodeId);
    }
    const grown = new Map<string, Set<string>>();
    for (const [nodeId, owner] of state.growthOwner) setDefault(grown, owner, () => new Set()).add(nodeId);
    state.growthIndex = { names, referrers, grown };
  }
  return state.growthIndex;
}

/** Python's `dict.setdefault`: return `map`'s value at `key`, first storing `make()` when absent. */
function setDefault<V>(map: Map<string, V>, key: string, make: () => V): V {
  let value = map.get(key);
  if (value === undefined) {
    value = make();
    map.set(key, value);
  }
  return value;
}

/**
 * Split the requested and deferred prune ids into those retired now and those kept.
 *
 * Each candidate is a requested prune id or one an earlier growth deferred. It is retired with its
 * whole grown subtree when every node in it is settled (taken, skipped, or expanded; none ready or
 * running). Otherwise it stays live with that subtree and is deferred, so a later growth retires it
 * once settled; a kept node also keeps every candidate it reads. A spawner that asks to prune
 * itself is always retired; its grown nodes go with it only if all are settled, else they move to
 * the nearest surviving owner.
 *
 * The candidates are `pruneIds`, then `state.deferredPrune`, without repeats; the deferred ids keep
 * the order they were found to be kept in.
 *
 * @param state The pre-transition state whose `deferredPrune` names earlier deferred ids.
 * @param spawnerId The spawner whose growth record asks for `pruneIds`.
 * @param pruneIds The node ids the growth record asks to prune.
 * @param grown Each owner's directly grown live nodes.
 * @returns The node ids retired by this growth, and the candidates kept live and deferred to a
 *   later growth.
 */
function retiredIds(
  state: RunState,
  spawnerId: string,
  pruneIds: readonly string[],
  grown: ReadonlyMap<string, ReadonlySet<string>>,
): readonly [ReadonlySet<string>, ReadonlySet<string>] {
  if (pruneIds.length === 0 && state.deferredPrune.size === 0) return [new Set(), new Set()];
  const settled: ReadonlySet<NodeState | undefined> = new Set([
    NodeState.TAKEN,
    NodeState.SKIPPED,
    NodeState.EXPANDED,
  ]);
  const readyIds = new Set(state.ready);
  const isSettled = (nodeId: string): boolean =>
    settled.has(state.nodeState.get(nodeId)) && !state.inFlight.has(nodeId) && !readyIds.has(nodeId);

  const units = new Map<string, string[]>();
  for (const candidate of new Set([...pruneIds, ...state.deferredPrune])) {
    const unit = [candidate];
    const pending = [candidate];
    while (pending.length > 0) {
      for (const nodeId of grown.get(pending.pop()!) ?? []) {
        unit.push(nodeId);
        pending.push(nodeId);
      }
    }
    units.set(candidate, unit);
  }

  const retired = new Set<string>();
  // A deferred spawner growing now is unsettled and kept like any other candidate.
  if (pruneIds.includes(spawnerId)) {
    const spawnerUnit = units.get(spawnerId)!;
    units.delete(spawnerId);
    retired.add(spawnerId);
    const grownIds = spawnerUnit.slice(1);
    if (grownIds.every(isSettled)) for (const nodeId of grownIds) retired.add(nodeId);
  }
  const kept = new Set<string>();
  for (const [candidate, unit] of units) if (!unit.every(isSettled)) kept.add(candidate);
  // A settled node that kept work still reads must stay with it. Units overlap when a deferred id
  // lies under another candidate, so a node can be in several.
  const unitsOf = new Map<string, string[]>();
  for (const [candidate, unit] of units) {
    for (const nodeId of unit) setDefault(unitsOf, nodeId, () => []).push(candidate);
  }
  const pending: string[] = [];
  for (const candidate of kept) for (const nodeId of units.get(candidate)!) pending.push(nodeId);
  while (pending.length > 0) {
    for (const read of readIds(state, pending.pop()!)) {
      for (const candidate of unitsOf.get(read) ?? []) {
        if (!kept.has(candidate)) {
          kept.add(candidate);
          for (const nodeId of units.get(candidate)!) pending.push(nodeId);
        }
      }
    }
  }
  for (const [candidate, unit] of units) {
    if (!kept.has(candidate)) for (const nodeId of unit) retired.add(nodeId);
  }
  return [retired, new Set([...kept].filter((candidate) => !retired.has(candidate)))];
}

/** Return the node ids a live node reads through edges, wiring, or output targets. */
function readIds(state: RunState, nodeId: string): Set<string> {
  const node = state.flow.nodes.get(nodeId)!;
  const reads = new Set(state.flow.incoming(nodeId).map((edge) => edge.from));
  for (const target of [node.commitAs, node.originId, node.skippedOutputTarget()] as unknown[]) {
    if (typeof target === "string") reads.add(target);
  }
  for (const source of (state.flow.wiring.get(nodeId) ?? new Map<string, unknown>()).values()) {
    for (const [, text] of iterSourceStrings(source)) {
      let paths: string[];
      try {
        paths = exprRefsOf(text);
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        continue; // the prune dependency check reports invalid wiring
      }
      for (const path of paths) reads.add(path.split(".", 1)[0]!);
    }
  }
  return reads;
}

/**
 * Reject requested prune ids that are not dynamic overlay or a self-prunable LOOP clone.
 *
 * Unsettled requested ids are not errors: `retiredIds` keeps them live.
 */
function validateGrowthPrune(
  state: RunState,
  spawnerId: string,
  pruneIds: readonly string[],
  newNodeIds: ReadonlySet<string>,
): void {
  const spawner = state.flow.nodes.get(spawnerId)!;
  for (const nodeId of pruneIds) {
    if (!state.flow.nodes.has(nodeId)) {
      raiseGrowthRejected(new RangeError(`growth prune node ${reprText(nodeId)} does not exist`));
    }
    if (nodeId === state.flow.startId || nodeId === state.flow.endId) {
      raiseGrowthRejected(new RangeError(`growth cannot prune top-level boundary ${reprText(nodeId)}`));
    }
    if (newNodeIds.has(nodeId)) {
      raiseGrowthRejected(new RangeError(`growth cannot prune new fragment node ${reprText(nodeId)}`));
    }
    if (state.initialStaticIds.has(nodeId)) {
      raiseGrowthRejected(new RangeError(`growth cannot prune initial static node ${reprText(nodeId)}`));
    }
    if (nodeId === spawnerId) {
      const originId: unknown = spawner.originId;
      if (
        !(
          spawner.isLoop === true &&
          typeof originId === "string" &&
          originId &&
          originId !== spawnerId &&
          state.flow.nodes.has(originId) &&
          !pruneIds.includes(originId)
        )
      ) {
        raiseGrowthRejected(new RangeError(`growth spawner ${reprText(spawnerId)} is not a self-prunable LOOP clone`));
      }
    }
  }
}

/**
 * Reject future work whose edge, wiring, commit, or origin reads a pruned id.
 *
 * The future work checked is every fragment node plus each unfinished live node that names a
 * pruned id or has an edge from one, found through the growth index and adjacency rather than a
 * scan of every node.
 */
function validatePruneDependencies(
  state: RunState,
  fragment: Flow,
  pruneIds: ReadonlySet<string>,
  index: GrowthIndex,
): void {
  if (pruneIds.size === 0) return;
  const sortedPruneIds = [...pruneIds].sort(compareCodeUnits);
  const candidates = new Set(fragment.nodes.keys());
  for (const nodeId of sortedPruneIds) {
    for (const referrer of index.referrers.get(nodeId) ?? []) candidates.add(referrer);
    for (const edge of state.flow.outgoing(nodeId)) candidates.add(edge.to);
  }
  const dependentIds = [...candidates].filter(
    (nodeId) =>
      fragment.nodes.has(nodeId) || (!pruneIds.has(nodeId) && state.nodeState.get(nodeId) === NodeState.UNKNOWN),
  );
  const dependents = new Set(dependentIds);
  const combinedNode = (nodeId: string): Node => fragment.nodes.get(nodeId) ?? state.flow.nodes.get(nodeId)!;
  const combinedNodeIds = { has: (nodeId: string) => fragment.nodes.has(nodeId) || state.flow.nodes.has(nodeId) };
  const prunedEdges: Edge[] = [];
  for (const nodeId of sortedPruneIds) for (const edge of state.flow.outgoing(nodeId)) prunedEdges.push(edge);
  for (const edge of fragment.edges) if (pruneIds.has(edge.from)) prunedEdges.push(edge);
  for (const edge of prunedEdges) {
    if (dependents.has(edge.to)) {
      raiseGrowthRejected(
        new RangeError(`surviving node ${reprText(edge.to)} depends on pruned edge source ${reprText(edge.from)}`),
      );
    }
  }
  for (const nodeId of dependentIds) {
    const node = combinedNode(nodeId);
    if (node.commitAs !== null && pruneIds.has(node.commitAs)) {
      raiseGrowthRejected(
        new RangeError(`surviving node ${reprText(nodeId)} commits to pruned node ${reprText(node.commitAs)}`),
      );
    }
    if (node.originId !== null && pruneIds.has(node.originId)) {
      raiseGrowthRejected(
        new RangeError(`surviving node ${reprText(nodeId)} has pruned origin ${reprText(node.originId)}`),
      );
    }
    const target: unknown = node.skippedOutputTarget();
    let skippedTarget: string;
    try {
      skippedTarget = validateSkippedOutputTarget(combinedNodeIds, nodeId, target);
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) raiseGrowthRejected(error);
      throw error;
    }
    if (pruneIds.has(skippedTarget)) {
      raiseGrowthRejected(
        new RangeError(
          `surviving node ${reprText(nodeId)} has pruned skipped output target ${reprText(skippedTarget)}`,
        ),
      );
    }
    const wiring = fragment.wiring.get(nodeId) ?? state.flow.wiring.get(nodeId) ?? new Map<string, unknown>();
    for (const source of wiring.values()) {
      for (const [, text] of iterSourceStrings(source)) {
        let paths: string[];
        try {
          paths = exprRefsOf(text);
        } catch (error) {
          if (!(error instanceof ExpressionError)) throw error;
          throw new RangeError(`surviving node ${reprText(nodeId)} has invalid wiring: ${error.message}`, {
            cause: error,
          });
        }
        for (const path of paths) {
          if (pruneIds.has(path.split(".", 1)[0]!)) {
            raiseGrowthRejected(new RangeError(`surviving node ${reprText(nodeId)} wiring depends on pruned node`));
          }
        }
      }
    }
  }
}

/** Validate a replayed route's exact scalar carriers and its executing router. */
function validateRouteRecord(state: RunState, record: NodeRoutedRecord): void {
  const nodeId: unknown = record.nodeId;
  const handle: unknown = record.handle;
  if (typeof nodeId !== "string") throw new TypeError("NodeRoutedRecord node_id must be a string");
  if (!nodeId) throw new RangeError("NodeRoutedRecord node_id must be non-empty");
  if (typeof handle !== "string") throw new TypeError("NodeRoutedRecord handle must be a string");
  if (!handle) throw new RangeError("NodeRoutedRecord handle must be non-empty");
  if (record.terminalFailure !== null && !isExactly(record.terminalFailure, Failure)) {
    throw new TypeError("NodeRoutedRecord terminal failure must be a Failure or null");
  }
  requireRoutingNode(state, nodeId);
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

/** Validate a failure's exact diagnostic carriers and its executing node. */
function validateFailureRecord(state: RunState, record: NodeFailedRecord): void {
  if (!isExactly(record.failure, Failure)) throw new TypeError("NodeFailedRecord failure must be a Failure");
  if (record.failure.code === "graph.terminal_unreachable") {
    throw new RangeError("graph.terminal_unreachable must be derived from a route record");
  }
  const nodeId: unknown = record.failure.nodeId;
  if (typeof nodeId !== "string") throw new TypeError("NodeFailedRecord failure node_id must be a string");
  if (!nodeId) throw new RangeError("NodeFailedRecord failure node_id must be non-empty");
  requireFailingNode(state, nodeId);
  validateNodeEvidence(state, nodeId, record.evidence);
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

/** Incremental counts for one data-input group of a touched join. */
interface DataGroupScratch {
  readonly optional: boolean;
  total: number;
  skipped: number;
}

/** How one incoming edge counts toward its join's gates, with its data group. */
type EdgeClass = readonly ["control" | "ordering" | "required_ordering" | "data", string | null];

/** One incoming-adjacency scan reduced to constant-time edge updates. */
interface JoinScratch {
  readonly states: Map<string, NodeState>;
  readonly classes: Map<string, EdgeClass>;
  unknown: number;
  controlTotal: number;
  controlSkipped: number;
  requiredDataDead: number;
  requiredOrderingSkipped: number;
  readonly dataGroups: Map<string | null, DataGroupScratch>;
}

/**
 * Validate one route and return its frozen skip closure, plus the terminal failure it first
 * proves (`causalFailure`), if any.
 */
function prepareRouteDelta(
  state: RunState,
  record: NodeRoutedRecord,
  deriveTerminalFailure: boolean,
): { delta: RouteDelta; causalFailure: Failure | null } {
  validateRouteRecord(state, record);
  const edgeUpdates = new Map<string, NodeState>();
  const skippedNodeIds: string[] = [];
  const skippedNodes = new Set<string>();
  const propagatedTargets = new Set<string>([record.nodeId]);
  let redirectOwners: Map<string, string[]> | null = null;
  const newlyReadyIds: string[] = [];
  const newlyReady = new Set<string>();
  const joins = new Map<string, JoinScratch>();

  const rootEdges = routeOutgoing(state, record.nodeId, edgeUpdates);
  // Only branch edges carry a handle. A handle-less edge (a `depends_on`/`runs_after` ordering
  // edge) isn't a branch: the router settled by routing, so it is TAKEN on every route rather than
  // read as the else branch. The reserved else route may have no authored edge: a CASE without
  // `else` selects no branch and skip-floods every branch edge. Any other unknown handle is invalid
  // node-produced data.
  if (!rootEdges.some((edge) => edge.sourceHandle === record.handle) && record.handle !== ELSE_HANDLE) {
    throw new NodeDataError(
      new RangeError(
        `NodeRouted handle ${reprText(record.handle)} matches no outgoing edge of ${reprText(record.nodeId)}`,
      ),
    );
  }

  // A LIFO stack with reversed pushes reproduces recursive stored-order DFS exactly, without a
  // recursion limit for deep skip chains.
  const work: (readonly [Edge, NodeState])[] = [];
  for (let index = rootEdges.length - 1; index >= 0; index--) {
    const edge = rootEdges[index]!;
    const taken = edge.sourceHandle === null || edge.sourceHandle === record.handle;
    work.push([edge, taken ? NodeState.TAKEN : NodeState.SKIPPED]);
  }
  while (work.length > 0) {
    const [edge, disposition] = work.pop()!;
    requireRouteEdge(state, edge, edgeUpdates);
    let scratch = joins.get(edge.to);
    if (scratch === undefined) {
      scratch = scanJoin(state, edge.to);
      joins.set(edge.to, scratch);
    }
    settleJoinEdge(scratch, edge.id, disposition, edge.to);
    edgeUpdates.set(edge.id, disposition);

    const head = edge.to;
    const headDisposition = state.nodeState.get(head);
    if (headDisposition !== NodeState.UNKNOWN && headDisposition !== NodeState.SKIPPED) {
      throw new RangeError(`route successor ${reprText(head)} must have UNKNOWN or SKIPPED disposition`);
    }
    if (skippedNodes.has(head) || headDisposition === NodeState.SKIPPED) continue;
    const joinDisposition = classifyJoin(scratch);
    if (joinDisposition === "wait") continue;
    if (joinDisposition === "ready") {
      if (newlyReady.has(head)) throw new RangeError(`route successor ${reprText(head)} became ready more than once`);
      newlyReady.add(head);
      newlyReadyIds.push(head);
      continue;
    }

    skippedNodes.add(head);
    skippedNodeIds.push(head);
    const target = skippedOutputTarget(state, head);
    if (target !== head) {
      redirectOwners ??= skippedOutputOwners(state);
      requireUnambiguousRedirect({
        ownerId: head,
        target,
        owners: redirectOwners.get(target) ?? [],
        hasCommittedValue: state.pool.store.has(target),
        disposition: state.nodeState.get(target),
        isExecutable: state.ready.includes(target) || state.inFlight.has(target),
        hasIncoming: state.flow.incoming(target).length > 0,
      });
    }
    if (propagatedTargets.has(target)) continue;
    propagatedTargets.add(target);
    const children = routeOutgoing(state, target, edgeUpdates);
    for (let index = children.length - 1; index >= 0; index--) work.push([children[index]!, NodeState.SKIPPED]);
  }

  let causalFailure: Failure | null = null;
  let terminalUnreachableFailure = state.terminalUnreachableFailure;
  if (terminalUnreachableFailure === null && skippedNodes.has(state.flow.endId)) {
    if (deriveTerminalFailure) {
      if (record.terminalFailure !== null) {
        throw new RangeError("live route preparation cannot supply a terminal failure");
      }
      causalFailure = makeTerminalUnreachableFailure(state, record.nodeId, edgeUpdates);
    } else {
      causalFailure = validateRecordedTerminalFailure(state, record, edgeUpdates);
    }
    terminalUnreachableFailure = causalFailure;
  } else if (record.terminalFailure !== null) {
    throw new RangeError("NodeRoutedRecord terminal failure is present on a noncausal route");
  }

  const delta: RouteDelta = Object.freeze({
    nodeId: record.nodeId,
    handle: record.handle,
    edgeUpdates: Object.freeze([...edgeUpdates].map((entry) => Object.freeze(entry))),
    skippedNodeIds: Object.freeze(skippedNodeIds),
    newlyReadyIds: Object.freeze(newlyReadyIds),
    terminalUnreachableFailure,
  });
  return { delta, causalFailure };
}

/** Build the source-neutral failure of one causal terminal skip. */
function makeTerminalUnreachableFailure(
  state: RunState,
  causalNodeId: string,
  edgeUpdates: ReadonlyMap<string, NodeState>,
): Failure {
  return new Failure({
    code: "graph.terminal_unreachable",
    message: terminalUnreachableMessage(state, edgeUpdates),
    nodeId: causalNodeId,
    trace: state.flow.nodes.get(causalNodeId)!.failureTrace(),
  });
}

/** Validate the persisted authority of one structurally causal route, field by field. */
function validateRecordedTerminalFailure(
  state: RunState,
  record: NodeRoutedRecord,
  edgeUpdates: ReadonlyMap<string, NodeState>,
): Failure {
  const failure = record.terminalFailure;
  if (failure === null) throw new RangeError("causal NodeRoutedRecord requires a persisted terminal failure");
  if (!isExactly(failure, Failure)) throw new TypeError("NodeRoutedRecord terminal failure must be a Failure");
  const expectedMessage = terminalUnreachableMessage(state, edgeUpdates);
  if (failure.code !== "graph.terminal_unreachable") {
    throw new RangeError("route terminal failure has the wrong failure code");
  }
  if (failure.nodeId !== record.nodeId) throw new RangeError("route terminal failure has the wrong causal node ID");
  if (failure.message !== expectedMessage) throw new RangeError("route terminal failure has the wrong message");
  if (failure.causeType !== null || failure.traceback !== null) {
    throw new RangeError("route terminal failure cannot carry exception diagnostics");
  }
  return failure;
}

/** The diagnosis of a skipped END group, naming it quoted, or `null`. */
function terminalUnreachableMessage(state: RunState, edgeUpdates: ReadonlyMap<string, NodeState>): string {
  const name = terminalSkippedOutputName(state, edgeUpdates);
  return `terminal output ${name === null ? writeAsText(null) : reprText(name)} skipped`;
}

/**
 * Return the first required END group, in stored order, that is fully skipped.
 *
 * `edgeUpdates` is the pending route overlay; an edge absent from it reads its live disposition.
 * Group insertion order keeps the first-occurrence rule, including `null` when that is the
 * diagnosed input group's name.
 */
function terminalSkippedOutputName(state: RunState, edgeUpdates: ReadonlyMap<string, NodeState>): string | null {
  const groups = new Map<string | null, readonly [boolean, boolean]>();
  for (const edge of state.flow.incoming(state.flow.endId)) {
    if (edge.sourceHandle !== null || edge.ordering) continue;
    const skipped = (edgeUpdates.get(edge.id) ?? state.edgeState.get(edge.id)) === NodeState.SKIPPED;
    const [optional, allSkipped] = groups.get(edge.inputGroup) ?? [edge.optional, true];
    groups.set(edge.inputGroup, [optional, allSkipped && skipped]);
  }
  for (const [name, [optional, allSkipped]] of groups) {
    if (!optional && allSkipped) return name;
  }
  return null;
}

/**
 * Return one skipped node's validated logical output target.
 *
 * The hook is node-owned, so CALL/MAP's baked completion redirect and LOOP's origin redirect stay
 * outside the kind-blind reducer. Its result is topology authority, not node-produced data, so a
 * malformed setup escapes as a plain host or graph error.
 */
function skippedOutputTarget(state: RunState, nodeId: string): string {
  const target: unknown = state.flow.nodes.get(nodeId)!.skippedOutputTarget();
  return validateSkippedOutputTarget(state.flow.nodes, nodeId, target);
}

/** Validate one hook result against an exact topology node lookup. */
function validateSkippedOutputTarget(nodes: { has(id: string): boolean }, nodeId: string, target: unknown): string {
  if (typeof target !== "string") {
    throw new TypeError(`node ${reprText(nodeId)} skipped output target must be a string`);
  }
  if (!target) throw new RangeError(`node ${reprText(nodeId)} skipped output target must be non-empty`);
  if (!nodes.has(target)) {
    throw new RangeError(`node ${reprText(nodeId)} skipped output target ${reprText(target)} does not exist`);
  }
  return target;
}

/**
 * Require every redirect a growth adds to have one inert logical anchor.
 *
 * Only targets of fragment nodes gain an owner, and the growth only relaxes every other target,
 * so those are the targets checked. Other targets aren't rechecked here: success and route
 * commits only settle nodes and edges, and a route checks each redirect again when it skips
 * through it. Each target is read in the post-growth topology: the live state plus the fragment
 * minus the retired nodes and edges.
 */
function validateGrowthRedirects(
  state: RunState,
  fragment: Flow,
  spawnerId: string,
  retiredIds: ReadonlySet<string>,
  retiredEdgeIds: ReadonlySet<string>,
  index: GrowthIndex,
): void {
  // Python's `_GrowthNodeIds`: membership after the growth, without copying the ids.
  const finalIds = {
    has: (id: string): boolean => fragment.nodes.has(id) || (state.flow.nodes.has(id) && !retiredIds.has(id)),
  };
  const newOwners = new Map<string, string[]>();
  for (const [nodeId, node] of fragment.nodes) {
    let target: string;
    try {
      target = validateSkippedOutputTarget(finalIds, nodeId, node.skippedOutputTarget());
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) raiseGrowthRejected(error);
      throw error;
    }
    if (target === nodeId) continue;
    setDefault(newOwners, target, () => []).push(nodeId);
  }
  if (newOwners.size === 0) return;
  const readyIds = new Set(state.ready);
  const fragmentHeads = new Set(fragment.edges.map((edge) => edge.to));
  for (const [target, fragmentOwnerIds] of newOwners) {
    const ownerIds: string[] = [];
    for (const nodeId of index.referrers.get(target) ?? []) {
      if (
        nodeId !== spawnerId &&
        !retiredIds.has(nodeId) &&
        state.nodeState.get(nodeId) === NodeState.UNKNOWN &&
        state.flow.nodes.get(nodeId)!.skippedOutputTarget() === target
      ) {
        ownerIds.push(nodeId);
      }
    }
    for (const nodeId of fragmentOwnerIds) ownerIds.push(nodeId);
    let disposition: unknown;
    let isExecutable: boolean;
    let hasIncoming: boolean;
    if (fragment.nodes.has(target)) {
      disposition = NodeState.UNKNOWN;
      isExecutable = target === fragment.startId;
      hasIncoming = fragmentHeads.has(target);
    } else {
      disposition = target === spawnerId ? NodeState.EXPANDED : state.nodeState.get(target);
      isExecutable = readyIds.has(target) || (state.inFlight.has(target) && target !== spawnerId);
      hasIncoming = state.flow.incoming(target).some((edge) => !retiredEdgeIds.has(edge.id));
    }
    try {
      requireUnambiguousRedirect({
        ownerId: ownerIds[0]!,
        target,
        owners: ownerIds,
        hasCommittedValue: state.pool.store.has(target),
        disposition,
        isExecutable,
        hasIncoming,
      });
    } catch (error) {
      if (error instanceof RangeError) raiseGrowthRejected(error);
      throw error;
    }
    // New fragment edges start UNKNOWN; only kept live edges can be settled.
    for (const edge of state.flow.outgoing(target)) {
      if (retiredEdgeIds.has(edge.id)) continue;
      if (state.edgeState.get(edge.id) !== NodeState.UNKNOWN) {
        raiseGrowthRejected(
          new RangeError(`logical output target ${reprText(target)} has a settled outgoing edge ${reprText(edge.id)}`),
        );
      }
    }
  }
}

/**
 * Index the unfinished logical-output redirects in stable node order.
 *
 * Settled dynamic residue can't produce another value and is ignored. Ordinary nodes target
 * themselves and aren't redirect owners. The scan is paid only when a route skips a redirected
 * completion; it keeps the common sparse skip path unchanged while refusing ambiguous
 * multi-writer topology before any edge settles.
 */
function skippedOutputOwners(state: RunState): Map<string, string[]> {
  const owners = new Map<string, string[]>();
  for (const nodeId of state.flow.nodes.keys()) {
    if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) continue;
    const target = skippedOutputTarget(state, nodeId);
    if (target === nodeId) continue;
    const ids = owners.get(target);
    if (ids === undefined) owners.set(target, [nodeId]);
    else ids.push(nodeId);
  }
  return owners;
}

/**
 * Require one inert logical anchor to have one unfinished redirect owner.
 *
 * A skip means the logical output is absent. That conclusion is sound only when no sibling
 * physical completion or executable anchor can still produce the same value. Supporting several
 * producers would need a distinct OR join and a multi-writer contract; `commitAs` stays a
 * single-owner redirect instead.
 */
function requireUnambiguousRedirect(options: {
  ownerId: string;
  target: string;
  owners: readonly string[];
  hasCommittedValue: boolean;
  disposition: unknown;
  isExecutable: boolean;
  hasIncoming: boolean;
}): void {
  const { ownerId, target, owners, hasCommittedValue, disposition, isExecutable, hasIncoming } = options;
  if (owners.length !== 1 || owners[0] !== ownerId) {
    throw new RangeError(
      `logical output target ${reprText(target)} has multiple skipped output owners: ${writeAsText([...owners])}`,
    );
  }
  if (hasCommittedValue) {
    throw new RangeError(`logical output target ${reprText(target)} already has a committed value`);
  }
  if (!NODE_STATES.has(disposition)) {
    throw new RangeError(`logical output target ${reprText(target)} has an invalid disposition`);
  }
  if (isExecutable) throw new RangeError(`logical output target ${reprText(target)} is also executable`);
  if (disposition === NodeState.UNKNOWN) {
    if (hasIncoming) throw new RangeError(`logical output target ${reprText(target)} is also executable`);
    return;
  }
  if (disposition !== NodeState.EXPANDED) {
    throw new RangeError(
      `logical output target ${reprText(target)} must be an expanded spawner or inert placeholder`,
    );
  }
}

/**
 * Return one node's validated outgoing adjacency in stored order.
 *
 * The adjacency index is the authority here; compile builds it and only growth commits change it,
 * so a route reads its own edges, not every edge in the flow.
 */
function routeOutgoing(state: RunState, nodeId: string, edgeUpdates: ReadonlyMap<string, NodeState>): readonly Edge[] {
  const outgoing: unknown = state.flow.outgoing(nodeId);
  if (!Array.isArray(outgoing)) throw new TypeError("compiled flow adjacency must be lists");
  const seen = new Set<string>();
  for (const edge of outgoing as unknown[]) {
    if (!(edge instanceof Edge) || edge.from !== nodeId) {
      throw new RangeError(`route node ${reprText(nodeId)} has an invalid outgoing adjacency`);
    }
    const sourceHandle: unknown = edge.sourceHandle;
    if (sourceHandle !== null && typeof sourceHandle !== "string") {
      throw new TypeError(`route edge ${reprText(edge.id)} source_handle must be a string or null`);
    }
    if (sourceHandle === "") throw new RangeError(`route edge ${reprText(edge.id)} source_handle must be non-empty`);
    const edgeId: unknown = edge.id;
    if (typeof edgeId !== "string" || !edgeId || seen.has(edgeId)) {
      throw new RangeError(`route node ${reprText(nodeId)} has invalid outgoing edge IDs`);
    }
    seen.add(edgeId);
    requireRouteEdge(state, edge, edgeUpdates);
    if (!state.flow.nodes.has(edge.to) || !state.nodeState.has(edge.to)) {
      throw new RangeError(`route edge ${reprText(edgeId)} has unknown successor ${reprText(edge.to)}`);
    }
  }
  return outgoing as readonly Edge[];
}

/** Recheck a queued edge without scanning any adjacency or the global frontier. */
function requireRouteEdge(state: RunState, edge: Edge, edgeUpdates: ReadonlyMap<string, NodeState>): void {
  if (edgeUpdates.has(edge.id)) throw new RangeError(`route edge ${reprText(edge.id)} is reached more than once`);
  if (!state.edgeState.has(edge.id)) throw new RangeError(`route edge ${reprText(edge.id)} has no disposition`);
  if (state.edgeState.get(edge.id) !== NodeState.UNKNOWN) {
    throw new RangeError(`route edge ${reprText(edge.id)} must be UNKNOWN`);
  }
}

/** Scan one touched head's incoming adjacency into incremental gate counts. */
function scanJoin(state: RunState, nodeId: string): JoinScratch {
  const incoming = state.flow.incoming(nodeId);
  if (incoming.length === 0) throw new RangeError(`route successor ${reprText(nodeId)} has no incoming adjacency`);
  const scratch: JoinScratch = {
    states: new Map(),
    classes: new Map(),
    unknown: 0,
    controlTotal: 0,
    controlSkipped: 0,
    requiredDataDead: 0,
    requiredOrderingSkipped: 0,
    dataGroups: new Map(),
  };
  for (const edge of incoming) {
    if (!(edge instanceof Edge) || edge.to !== nodeId) {
      throw new RangeError(`route successor ${reprText(nodeId)} has invalid incoming adjacency`);
    }
    const edgeId: unknown = edge.id;
    if (typeof edgeId !== "string" || !edgeId || scratch.states.has(edgeId)) {
      throw new RangeError(`route successor ${reprText(nodeId)} has invalid incoming edge IDs`);
    }
    if (!state.edgeState.has(edgeId)) {
      throw new RangeError(`route incoming edge ${reprText(edgeId)} has no disposition`);
    }
    const disposition = state.edgeState.get(edgeId)!;
    if (!NODE_STATES.has(disposition)) {
      throw new RangeError(`route incoming edge ${reprText(edgeId)} has invalid disposition`);
    }
    scratch.states.set(edgeId, disposition);
    if (disposition === NodeState.UNKNOWN) scratch.unknown += 1;
    const skipped = disposition === NodeState.SKIPPED;
    if (edge.sourceHandle !== null) {
      scratch.classes.set(edgeId, ["control", null]);
      scratch.controlTotal += 1;
      if (skipped) scratch.controlSkipped += 1;
    } else if (edge.ordering) {
      scratch.classes.set(edgeId, [edge.optional ? "ordering" : "required_ordering", null]);
      if (!edge.optional && skipped) scratch.requiredOrderingSkipped += 1;
    } else {
      let group = scratch.dataGroups.get(edge.inputGroup);
      if (group === undefined) {
        group = { optional: edge.optional, total: 0, skipped: 0 };
        scratch.dataGroups.set(edge.inputGroup, group);
      }
      group.total += 1;
      if (skipped) group.skipped += 1;
      scratch.classes.set(edgeId, ["data", edge.inputGroup]);
    }
  }
  for (const group of scratch.dataGroups.values()) {
    if (!group.optional && group.skipped === group.total) scratch.requiredDataDead += 1;
  }
  return scratch;
}

/** Apply one UNKNOWN-to-settled update to a cached join in constant time. */
function settleJoinEdge(scratch: JoinScratch, edgeId: string, disposition: NodeState, nodeId: string): void {
  if (!scratch.states.has(edgeId)) {
    throw new RangeError(
      `route edge ${reprText(edgeId)} is absent from successor ${reprText(nodeId)} incoming adjacency`,
    );
  }
  if (scratch.states.get(edgeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`route edge ${reprText(edgeId)} must be UNKNOWN in its incoming join`);
  }
  scratch.states.set(edgeId, disposition);
  scratch.unknown -= 1;
  if (disposition !== NodeState.SKIPPED) return;
  const [edgeClass, groupName] = scratch.classes.get(edgeId)!;
  if (edgeClass === "control") {
    scratch.controlSkipped += 1;
  } else if (edgeClass === "required_ordering") {
    scratch.requiredOrderingSkipped += 1;
  } else if (edgeClass === "data") {
    const group = scratch.dataGroups.get(groupName)!;
    group.skipped += 1;
    if (!group.optional && group.skipped === group.total) scratch.requiredDataDead += 1;
  }
}

/** Classify cached gate counts, an eager "dead" before an unresolved "wait". */
function classifyJoin(scratch: JoinScratch): "ready" | "wait" | "dead" {
  if (scratch.controlTotal && scratch.controlSkipped === scratch.controlTotal) return "dead";
  if (scratch.requiredDataDead) return "dead";
  if (scratch.requiredOrderingSkipped) return "dead";
  if (scratch.unknown) return "wait";
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
    isExactly(state.interruptedAttempts, Map) &&
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
    state.interruptedAttempts.size === 0 &&
    state.lastStopSequence === null &&
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

/** The detached values one route commit installs, all built before the durable append. */
interface RouteApply {
  terminal: TerminalApply | null;
  observers: readonly RunEvent[];
}

/** Materialize the route observers and independent terminal copies before append. */
function materializeRoute(plan: RoutePlan): RouteApply {
  const terminal = plan.terminal !== null ? materializeTerminal(plan.terminal) : null;
  const observers: RunEvent[] = [
    new NodeRouted({ nodeId: plan.delta.nodeId, handle: plan.delta.handle }),
    ...plan.delta.skippedNodeIds.map((nodeId) => new NodeSkipped({ nodeId })),
  ];
  if (terminal !== null) observers.push(terminal.observer);
  return { terminal, observers };
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

/** Build the ordered observers, sharing one immutable failure. */
function failureObservers(failure: Failure): RunEvent[] {
  return [new NodeFailed({ failure }), new RunFailed({ failure })];
}

/** The detached values one failure commit installs, all built before the durable append. */
interface FailureApply {
  result: RunFailed;
  observers: readonly RunEvent[];
}

/** Build independent state and callback values before the durable append. */
function materializeFailure(failure: Failure): FailureApply {
  return { result: new RunFailed({ failure }), observers: failureObservers(failure) };
}

/** Build one independent canonical expansion observer. */
function growthObserver(record: NodeExpandedRecord): NodeExpanded {
  const [, seed] = canonicalSeed(record.seed);
  return new NodeExpanded({ nodeId: record.spawnerId, grow: new Grow(seed, record.pruneIds) });
}

/** The detached values one growth commit installs, all built before the durable append. */
interface GrowthApply {
  wiring: Map<string, ReadonlyMap<string, Binding>>;
  observers: readonly RunEvent[];
}

/** Build fresh fragment wiring and the observer before the durable append. */
function materializeGrowth(plan: GrowthPlan): GrowthApply {
  const record = decodeRecord(plan.envelopeJson).record;
  if (!isExactly(record, NodeExpandedRecord)) {
    throw new TypeError("immutable growth plan does not contain a NodeExpandedRecord");
  }
  if (
    !(
      record.spawnerId === plan.delta.spawnerId &&
      sameItems(record.pruneIds, plan.delta.pruneIds) &&
      record.topologyDigest === plan.delta.topologyDigest
    )
  ) {
    throw new RangeError("immutable growth plan record and delta disagree");
  }
  return {
    wiring: new Map(plan.topology.wiring.map(([nodeId, wiring]) => [nodeId, new Map(wiring)])),
    observers: [growthObserver(record)],
  };
}

/** Carry one growth into `state.growthIndex` before ownership changes. */
function applyGrowthIndex(state: RunState, delta: GrowthDelta, topology: GrowthTopology): void {
  const index = state.growthIndex;
  // Preparation builds it; there is nothing to keep current otherwise.
  if (index === null) return;
  for (const nodeId of delta.retiredIds) {
    const named = index.names.get(nodeId) ?? [];
    index.names.delete(nodeId);
    for (const name of named) {
      const referrers = index.referrers.get(name);
      if (referrers !== undefined) {
        referrers.delete(nodeId);
        if (referrers.size === 0) index.referrers.delete(name);
      }
    }
    const owner = state.growthOwner.get(nodeId);
    if (owner !== undefined) index.grown.get(owner)?.delete(nodeId);
    index.grown.delete(nodeId);
  }
  for (const [nodeId, named] of topology.names) {
    index.names.set(nodeId, named);
    for (const name of named) setDefault(index.referrers, name, () => new Set()).add(nodeId);
  }
  for (const [nodeId, owner] of topology.growthOwner) {
    const previous = state.growthOwner.get(nodeId);
    if (previous !== undefined) index.grown.get(previous)?.delete(nodeId);
    if (owner !== null) setDefault(index.grown, owner, () => new Set()).add(nodeId);
  }
}

/**
 * Install one prevalidated growth into the live containers in place.
 *
 * Runs after append, so it only does the map and array updates the preparation already proved
 * valid.
 */
function applyGrowth(
  state: RunState,
  delta: GrowthDelta,
  topology: GrowthTopology,
  wiring: ReadonlyMap<string, ReadonlyMap<string, Binding>>,
): void {
  const flow = state.flow;
  const retired = new Set(delta.retiredIds);
  applyGrowthIndex(state, delta, topology);
  if (retired.size > 0) {
    // Edges are appended in growth order and a prune usually retires recent growth, so scan back
    // from the end and stop once every retired edge is found. Sibling growth (other MAP items,
    // say) can sit in between, so the scan may still pass edges the growth doesn't retire.
    const pending = new Set(topology.retiredEdges.map((edge) => edge.id));
    let position = flow.edges.length;
    while (pending.size > 0 && position > 0) {
      position -= 1;
      const edgeId = flow.edges[position]!.id;
      if (pending.has(edgeId)) {
        pending.delete(edgeId);
        flow.edges.splice(position, 1);
      }
    }
    for (const edge of topology.retiredEdges) {
      state.edgeState.delete(edge.id);
      for (const [adjacency, nodeId] of [
        [flow.outgoingEdges, edge.from],
        [flow.incomingEdges, edge.to],
      ] as const) {
        const edges = adjacency.get(nodeId);
        if (retired.has(nodeId) || edges === undefined) continue;
        const kept = edges.filter((other) => other !== edge);
        if (kept.length > 0) adjacency.set(nodeId, kept);
        else adjacency.delete(nodeId);
      }
    }
    for (const nodeId of delta.retiredIds) {
      flow.nodes.delete(nodeId);
      flow.wiring.delete(nodeId);
      flow.incomingEdges.delete(nodeId);
      flow.outgoingEdges.delete(nodeId);
      state.nodeState.delete(nodeId);
      state.pool.store.delete(nodeId);
      state.depth.delete(nodeId);
      state.wiringOverrides.delete(nodeId);
      state.growthOwner.delete(nodeId);
    }
  }

  for (const [nodeId, node] of topology.nodes) flow.nodes.set(nodeId, node);
  for (const edge of topology.edges) flow.edges.push(edge);
  for (const [nodeId, nodeWiring] of wiring) flow.wiring.set(nodeId, nodeWiring);
  for (const edge of topology.edges) {
    setDefault(flow.outgoingEdges, edge.from, () => []).push(edge);
    setDefault(flow.incomingEdges, edge.to, () => []).push(edge);
  }
  for (const nodeId of delta.newNodeIds) state.nodeState.set(nodeId, NodeState.UNKNOWN);
  if (!retired.has(delta.spawnerId)) state.nodeState.set(delta.spawnerId, NodeState.EXPANDED);
  for (const edgeId of delta.newEdgeIds) state.edgeState.set(edgeId, NodeState.UNKNOWN);
  state.inFlight.delete(delta.spawnerId);
  state.acceptedStarts.delete(delta.spawnerId);
  forgetNodeSteps(state, delta.spawnerId);
  state.ready.push(delta.fragmentStartId);
  for (const [nodeId, depth] of topology.depth) state.depth.set(nodeId, depth);
  state.deferredPrune = topology.deferredPrune;
  for (const [nodeId, owner] of topology.growthOwner) {
    if (owner === null) state.growthOwner.delete(nodeId);
    else state.growthOwner.set(nodeId, owner);
  }
}
