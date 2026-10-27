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
 *                events, expr, memory, nodes, runtime/common, runtime/contracts, runtime/eval_node,
 *                runtime/learning, runtime/log, runtime/trace, typesys.
 * Never imports: engine, run, backward, compose, or the CLI.
 */

import { localTypes } from "../compile/declared_types.ts";
import { expansionDigest } from "../compile/manifest.ts";
import { CompiledFlow, copyGraph, Edge, Flow, GrowthRejected, validateGrowthMetadata } from "../compile/model.ts";
import { classifyPath } from "../compile/validation.ts";
import { BUILTIN, type Defaults, fromRecorded } from "../defaults.ts";
import { compareCodeUnits, Failure, type OriginRef, reprText, sameFailure } from "../diagnostics.ts";
import {
  CallClosed,
  NodeExpanded,
  NodeFailed,
  NodeRouted,
  NodeSkipped,
  NodeStarted,
  NodeSucceeded,
  NodeWaiting,
  NodeWoken,
  RunCancelled,
  RunFailed,
  RunResumed,
  RunStarted,
  type CloseReason,
  type RunEvent,
  RunSucceeded,
  RunWaiting,
} from "../events.ts";
import { ExpressionError } from "../expr/expressions.ts";
import { type RootRead, SourceRef, sameSource, type ValueRoute } from "../expr/lineage.ts";
import { exprRefsOf } from "../expr/template.ts";
import {
  MemoryScope,
  type MemorySnapshotEnvelope,
  type MemorySpec,
  type MemoryStack,
} from "../memory/index.ts";
import { sameMemoryScope } from "../memory/contracts.ts";
import { type CompleteResult, ELSE_HANDLE, Grow, Node, Output } from "../nodes/base.ts";
import { BindingError, copyValue } from "../nodes/binding.ts";
import {
  type Binding,
  type CallView,
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
import { eventBytes, type JsonObject, type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import { messagesKey, NEW_MESSAGES, VariablePool } from "../typesys/pool.ts";
import { newMessagesType } from "../typesys/types.ts";
import {
  isInboxType,
  isRecord,
  jsTypeName,
  normalizeTypedFiles,
  rejectDeepNesting,
  type Type,
  TypeCheckError,
  TypedValue,
  type ValueRecord,
  writeAsText,
} from "../typesys/values.ts";
import { NodeState } from "./common.ts";
import { ForwardCapture, validateSource } from "./contracts.ts";
import { nodeBackwardCapture, rebind, rebuildDerived, rebuildNodeData } from "./eval_node.ts";
import { reduceBackwardRecord, RuntimeBackwardState } from "./learning.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  canonicalSeed,
  canonicalTyped,
  decodeRecord,
  encodeRecord,
  InboxClosedRecord,
  InboxWithdrawnRecord,
  InterruptedRecord,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  type OutcomeCarriers,
  PoolCommitRecord,
  RecordEnvelope,
  type FlowSource,
  type RecordedSkill,
  type RecordedTool,
  type RunLog,
  type RunParent,
  RunCancelledRecord,
  RunFailedRecord,
  type RunRecord,
  RunResumedRecord,
  RunStartedRecord,
  type RunStopReason,
  RunStoppedRecord,
  RunWaitingRecord,
  requireRecordTime,
  SCHEMA_VERSION,
  type TakeCommit,
  type WaitEntry,
} from "./log.ts";
import {
  checkMemoryScope,
  captureBindings,
  decodeTyped,
  type PrepareTraceOptions,
  type ReboundSource,
  type Rebinder,
  prepareTrace,
  RetainedTrace,
  selectSeedRoute,
  SourceEntry,
  sameSourceEntry,
  SYSTEM_PREFIX,
  resolvePoolPath,
  TraceDelta,
  type TraceDeltaFields,
  TraceOccurrence,
  TraceTerminal,
} from "./trace.ts";
import { ASK, ASK_USER, SLEEP, TAKE, toolCallId } from "../tools/context.ts";
import {
  applyWaitsRecord,
  availableUnread,
  awaitsNextMessage,
  type Call,
  type CallStateKind,
  nodeCalls,
  seqList,
  WaitsState,
  wakeableNodes,
  withdrawableEvents,
} from "./waits.ts";

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

/** The memory configuration `RunStarted` recorded. */
interface RunMemory {
  /** The recorded snapshots, or `null` when memory is disabled. */
  readonly snapshots: readonly MemorySnapshotEnvelope[] | null;
  /** The recorded memory key table, or `null` when memory is disabled. */
  readonly memories: readonly MemorySpec[] | null;
  /** The recorded key stacks, or `null` when memory is disabled. */
  readonly memoryStacks: readonly MemoryStack[] | null;
  /** Whether the run retains what learning reads (forward capture and steps). */
  readonly learnable: boolean;
}

/** A pristine state's memory configuration: memory and learning off. */
const NO_MEMORY: RunMemory = Object.freeze({ snapshots: null, memories: null, memoryStacks: null, learnable: false });

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

/** Why a host delivery (a send, withdraw, complete, close, interrupt or cancel) was refused. */
export type DeliveryRefusal =
  | "unknown_inbox"
  | "not_started"
  | "ended"
  | "inbox_closed"
  | "invalid_value"
  | "too_large"
  | "unknown_call"
  | "call_settled"
  | "not_host_call";

/** A send, withdraw, complete, close, interrupt or cancel the run refuses; nothing was appended. */
export class DeliveryRefused extends RangeError {
  override name = "DeliveryRefused";
  /** Why the delivery was refused. */
  readonly reason: DeliveryRefusal;
  /** The call's state for `call_settled`; `null` otherwise. */
  readonly callState: CallStateKind | null;

  /** Hold `reason` and the message; `options.cause` becomes the error's cause. */
  constructor(
    reason: DeliveryRefusal,
    message: string,
    options: { callState?: CallStateKind | null; cause?: unknown } = {},
  ) {
    super(message, Object.hasOwn(options, "cause") ? { cause: options.cause } : {});
    this.reason = reason;
    this.callState = options.callState ?? null;
  }
}

/**
 * Refuse a delivery to a run that has ended (`ended`) or whose run start is incomplete
 * (`not_started`), in that order.
 */
function requireDeliverable(state: RunState): void {
  if (state.result !== null) throw new DeliveryRefused("ended", "the run has ended");
  if (!state.inputsCommitted) throw new DeliveryRefused("not_started", "the run hasn't started");
}

/** Require a dispatcher worker count: a non-negative safe integer. */
function requireWorkerCount(numWorkers: number): void {
  if (!Number.isSafeInteger(numWorkers)) throw new TypeError("num_workers must be a non-negative integer");
  if (numWorkers < 0) throw new RangeError("num_workers must be a non-negative integer");
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
  /**
   * The number of `UNKNOWN` incoming edges of each live node that has any. Every commit that
   * settles, adds or retires an edge keeps it in step with `edgeState`, so a replay rebuilds it.
   */
  unknownIncoming = new Map<string, number>();
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
  /** The seq of each step in `nodeSteps`, by node, index for index. */
  nodeStepSeqs = new Map<string, readonly number[]>();
  /**
   * The number of accepted starts of each node that a later `RunResumed` retried with no
   * `RunStopped` covering them. Log history by node id: never decremented, and kept after the
   * node's outcome and after a prune.
   */
  crashedAttempts = new Map<string, number>();
  /** The sequence of the last `RunStopped` since the last `RunResumed`, or `null` when none. */
  lastStopSequence: number | null = null;
  /** The derived terminal result, or `null` while the run is live. */
  result: RunSucceeded | RunFailed | RunCancelled | null = null;
  /** The inbox and call fold: the waiting set, registered calls, inboxes and records-only status facts. */
  readonly waits = new WaitsState();
  /** Each send's value by its seq, kept until its consumer settles. */
  readonly inboxEvents = new Map<number, TypedValue>();
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
  /**
   * Whether the run start is complete: its last commit, the START inputs', is applied. Until then
   * only the run start's commits may follow `RunStarted`, and no node is ready.
   */
  inputsCommitted = false;
  /**
   * The output commit of each in-flight node that no success has accepted yet. The node's
   * `NodeSucceeded` applies it to the pool; a resume drops it, and the node writes a new one.
   */
  pendingCommits = new Map<string, PendingCommit>();
  /**
   * The messages commit of each in-flight node that no success has accepted yet, with its rebuilt
   * value. The node's `NodeSucceeded` applies it with its output; a resume drops it.
   */
  pendingMessages = new Map<string, PendingMessages>();
  /**
   * The seq of the accepted `PoolCommitRecord` that last wrote each pool name: a store key, or
   * `system.<name>` for a system value. Set when a commit is applied, deleted with its pool entry,
   * so it holds exactly the pool's names and never grows past them.
   */
  producers = new Map<string, number>();
  /** Run-resolved agent configuration by stable static node path. */
  effectiveLlmByStaticPath: Record<string, JsonObject> = {};
  /**
   * The recorded agents whose configuration resolves only when they run, with the selection they
   * bind from (`RunStartedRecord.callTimeLlm`); `null` when there are none.
   */
  callTimeLlm: JsonObject | null = null;
  // The memory roots, in one frozen object `RunStarted` installs once (`installMemory`) and the
  // getters read, so no root can be reassigned apart from the others.
  #memory: RunMemory = NO_MEMORY;
  /** The skills `RunStarted` recorded, by name, which learning renders prompts with. */
  recordedSkills: Readonly<Record<string, RecordedSkill>> = {};
  /** The tool specs `RunStarted` recorded, by name, which learning rebuilds requests with. */
  recordedTools: Readonly<Record<string, RecordedTool>> = {};
  /** The tool groups `RunStarted` recorded, by name, which learning expands grants with. */
  recordedToolGroups: Readonly<Record<string, readonly string[]>> = {};
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
    for (const edge of flow.edges) addUnknownIncoming(this, edge.to);
    this.initialStaticIds = new Set(flow.nodes.keys());
  }

  /** The canonical recorded snapshots, one per memory key. */
  get memorySnapshots(): readonly MemorySnapshotEnvelope[] | null {
    return this.#memory.snapshots;
  }

  /** The recorded memory key table. */
  get memories(): readonly MemorySpec[] | null {
    return this.#memory.memories;
  }

  /** The recorded key stacks. */
  get memoryStacks(): readonly MemoryStack[] | null {
    return this.#memory.memoryStacks;
  }

  /** Whether accepted outcomes carry forward capture and keep their steps for learning. */
  get learnable(): boolean {
    return this.#memory.learnable;
  }

  /**
   * @internal Install the memory configuration `RunStarted` recorded, frozen.
   *
   * @throws RangeError when one is already installed.
   */
  installMemory(memory: RunMemory): void {
    if (this.#memory !== NO_MEMORY) throw new RangeError("the run's memory configuration is already installed");
    this.#memory = Object.freeze({ ...memory });
  }
}

/**
 * A commit waiting for the success that accepts it, kept under that node's id: a node's own output
 * commit, or an expansion's commit for a child START (kept under the child START's id).
 */
export interface PendingCommit {
  /** The seq of the `PoolCommitRecord`. */
  readonly seq: number;
  /** The pool key the value is written under. */
  readonly key: string;
  /**
   * The value's canonical typed wire text; `null` for a derived or take commit until the success
   * that accepts it (for an expansion commit, its spawner's `NodeExpanded`) rebuilds the value.
   */
  readonly valueJson: string | null;
  /** A derived commit's sources (`PoolCommitRecord.from`), else `null`. */
  readonly from: ReadonlyMap<string, number> | null;
  /** A take commit's inbox and consumed sends (`PoolCommitRecord.take`), else `null`. */
  readonly take: TakeCommit | null;
  /** The spawner whose expansion wrote this commit for a child START, else `null`. */
  readonly spawner: string | null;
  /**
   * Whether the spawner's `NodeExpanded` admitted this expansion commit; until then the child START
   * doesn't exist yet, and a resume drops the commit with its spawner's retry.
   */
  readonly admitted: boolean;
}

/** A messages commit waiting for its node's success, kept under the node's id. */
export interface PendingMessages {
  /** The seq of the messages `PoolCommitRecord`. */
  readonly seq: number;
  /** The pool key the messages are written under, `messagesKey(<node>)`. */
  readonly key: string;
  /** The rebuilt `list(Message)` value's canonical typed wire text. */
  readonly valueJson: string;
}

/**
 * Read a reference's root in `pool` with the commit that wrote it, as `RetainedTrace.resolve`
 * reads it from the retained trace, but with the seq of the name's accepted `PoolCommitRecord`
 * (`producers`) as its source, so a run without a trace can build a node's input route. Both read
 * the same pool names (`referenceRoot`). The one difference: a system value gets its run-start
 * commit as its source, where the trace gives it none.
 *
 * @throws RangeError when a name the pool holds has no entry in `producers`, which always names
 *   every pool name.
 */
export function resolveFromPool(
  names: readonly string[],
  pool: VariablePool,
  producers: ReadonlyMap<string, number>,
  runId: string,
): RootRead {
  return resolvePoolPath(names, pool, (name) => {
    const seq = producers.get(name);
    if (seq === undefined) throw new RangeError(`pool name ${reprText(name)} has no pool_commit in the producer index`);
    return new SourceRef(runId, seq);
  });
}

/**
 * Return the resolver the reducer rebuilds derived values with: the value pool name `name` holds
 * now, named by its producer index seq. The source it gives is the one the run's routes name: in
 * a learnable run the trace's current writer (none for a system value), as `RetainedTrace.resolve`
 * gives it, else the commit at `seq`.
 *
 * Sound because records apply in log order: when a derived commit is accepted, the pool holds
 * exactly what its binding read.
 *
 * @throws RangeError (from the resolver) when `seq` isn't the seq of the `pool_commit` that last
 *   wrote `name`: the log and the flow disagree.
 */
export function poolResolver(state: RunState): (name: string, seq: number) => ReboundSource {
  return (name, seq) => {
    const producer = state.producers.get(name);
    if (producer !== seq) {
      throw new RangeError(
        `from names ${reprText(name)} at seq ${seq}, but its pool_commit is ${producer === undefined ? "(none)" : `seq ${producer}`}`,
      );
    }
    const value = name.startsWith(SYSTEM_PREFIX) ? state.pool.system.get(name.slice(SYSTEM_PREFIX.length)) : state.pool.store.get(name);
    // The producer index names exactly the pool's names.
    if (value === undefined) throw new RangeError(`pool name ${reprText(name)} has no value`);
    const trace = state.trace;
    if (trace === null) return { value, source: new SourceRef(state.runId!, seq) };
    if (name.startsWith(SYSTEM_PREFIX)) return { value, source: null };
    // The trace's current writers name exactly the pool's names.
    return { value, source: trace.currentWriters.get(name)!.source };
  };
}

/**
 * Rebuild `nodeId`'s derived outcome from `from` (`rebuildDerived`), reading the pool through the
 * producer index as its binding read it, and in a learnable run the forward capture its record
 * leaves out: the route and bound inputs the binding collects, and the backward material the node
 * gives, as the evaluator captures them (`null` in a run that doesn't learn).
 *
 * With `from` `null` (an expansion that wrote no commits) it binds from the whole pool.
 *
 * @throws as `rebuildDerived`.
 */
function rebuildDerivedRecord(
  state: RunState,
  nodeId: string,
  from: ReadonlyMap<string, number> | null,
): { outcome: Output | Grow; capture: ForwardCapture | null } {
  const node = state.flow.nodes.get(nodeId)!;
  const wiring = state.wiringOverrides.get(nodeId) ?? null;
  const learnable = state.trace !== null;
  const { outcome, capture } = rebuildDerived(node, state.flow, from ?? state.producers, poolResolver(state), wiring, {
    capture: learnable,
  });
  if (!learnable) return { outcome, capture: null };
  return { outcome, capture: captureBindings(capture!, nodeBackwardCapture(node)) };
}

/**
 * Rebuild the value of `nodeId`'s derived commit under `key` from its `from`, reading the pool
 * through the producer index (`poolResolver`), as its binding read it when the commit was accepted,
 * with the forward capture of the success that accepts the commit (`rebuildDerivedRecord`).
 *
 * @returns The canonical typed wire text of the value, and the capture (`null` in a run that
 *   doesn't learn).
 * @throws RangeError when a `from` seq isn't its name's producer, or the node rebuilds to a growth
 *   or under another key; a `BindingError` when its binding no longer evaluates.
 */
export function rebuildCommit(
  state: RunState,
  nodeId: string,
  key: string,
  from: ReadonlyMap<string, number>,
): { valueJson: string; capture: ForwardCapture | null } {
  const node = state.flow.nodes.get(nodeId)!;
  const { outcome, capture } = rebuildDerivedRecord(state, nodeId, from);
  if (!(outcome instanceof Output)) {
    throw new RangeError(`the derived commit of ${reprText(nodeId)} rebuilds to a growth, not a value`);
  }
  const target = outcome.commitAs ?? node.commitAs ?? nodeId;
  if (target !== key) {
    throw new RangeError(`the derived commit of ${reprText(nodeId)} is under ${reprText(key)}, but rebuilds under ${reprText(target)}`);
  }
  return { valueJson: committedText(state, key, outcome.value), capture };
}

/**
 * The canonical typed wire text `value` commits as under `key`, read through its target's
 * contract as a node's commit is.
 *
 * @throws as `canonicalOutputValue` when the target refuses the value.
 */
function committedText(state: RunState, key: string, value: unknown): string {
  const target = state.flow.nodes.get(key)!;
  return logJson(canonicalOutputValue(value, target.outputContract, nodeOutputType(target))[0]);
}

/**
 * Rebuild spawner `spawnerId`'s growth from its expansion commits' `from` (see
 * `rebuildCommit`): its canonical seed and sorted prune ids, and in a learnable run the
 * expansion's forward capture (`rebuildDerivedRecord`).
 *
 * With `from` `null` (every child START of the expansion fails to bind, so it wrote no commit),
 * the spawner binds from the whole pool. That is sound for the same reason the commits are: the
 * growth's record applies right after its spawner's evaluation, so the pool holds exactly what the
 * binding read; the topology digest still pins the fragment the seed builds.
 *
 * @throws RangeError when a `from` seq isn't its name's producer, or the spawner rebuilds to a
 *   value; a `BindingError` when its binding no longer evaluates.
 */
export function rebuildGrowth(
  state: RunState,
  spawnerId: string,
  from: ReadonlyMap<string, number> | null,
): { seed: JsonValue; pruneIds: readonly string[]; capture: ForwardCapture | null } {
  const { outcome, capture } = rebuildDerivedRecord(state, spawnerId, from);
  if (!(outcome instanceof Grow)) throw new RangeError(`spawner ${reprText(spawnerId)} rebuilds to a value, not a growth`);
  return { seed: canonicalSeed(outcome.seed)[0], pruneIds: [...outcome.prune].sort(compareCodeUnits), capture };
}

/**
 * The values a prepared record accepts, for a reader folding a log (`foldRunLog`), each with the
 * seq of the commit that holds it: a run-start commit's or a send's own value; for a success, its
 * commit's value (rebuilt for a derived or take commit), then its messages commit's value (rebuilt
 * from the node's steps) when it accepts one; none for any other record.
 */
export function acceptedCommits(prepared: PreparedTransition): readonly { seq: number; value: TypedValue }[] {
  const plan = prepared._plan;
  // A run-start commit and a send always hold their values, and apply at once.
  if (plan.kind === "commit" && (plan.commitKind === "initial" || plan.commitKind === "send")) {
    return [{ seq: prepared.expectedSequence, value: decodeTyped(plan.valueJson!) }];
  }
  if (plan.kind !== "success") return [];
  const seq = (prepared.record as NodeSucceededRecord).commit;
  const accepted = [{ seq, value: normalizeTypedFiles(plan.files, decodeTyped(plan.delta.valueJson)) }];
  const messages = plan.delta.messages;
  if (messages !== null) {
    accepted.push({ seq: messages.seq, value: normalizeTypedFiles(newMessagesType(), decodeTyped(messages.valueJson)) });
  }
  return accepted;
}

/**
 * Whether `nodeId`'s own commit is derived and holds `from`: its node is `derived` and is neither
 * the top-level START nor END, whose commits keep their values.
 */
export function isDerivedCommit(state: RunState, nodeId: string): boolean {
  return state.flow.nodes.get(nodeId)?.derived === true && nodeId !== state.flow.startId && nodeId !== state.flow.endId;
}

/**
 * Require a child START's evaluated `output` to equal the expansion commit its success names.
 *
 * @throws RangeError naming the node and the commit's seq when they differ.
 */
export function requireExpansionOutput(state: RunState, nodeId: string, output: unknown): void {
  const pending = state.pendingCommits.get(nodeId);
  if (pending === undefined || !pending.admitted) throw new RangeError(`node ${reprText(nodeId)} has no expansion commit`);
  requireEvaluated(state, `child START ${reprText(nodeId)}`, pending, pending.valueJson!, output);
}

/**
 * Require `output`, the value a node evaluated to, to commit as `valueJson`, the value `pending`
 * holds or rebuilds to.
 *
 * @throws RangeError naming the node (`what`) and the commit's seq when they differ.
 */
function requireEvaluated(state: RunState, what: string, pending: PendingCommit, valueJson: string, output: unknown): void {
  if (committedText(state, pending.key, output) !== valueJson) {
    throw new RangeError(`${what} evaluated to another value than its commit at seq ${pending.seq} holds`);
  }
}

/**
 * The fully resolved configuration `prepareRunStarted` builds the first authoritative record from.
 * The run's inputs and system values follow it as commits (`prepareInitialCommit`).
 */
export interface RunStartData {
  /** The non-empty identity the state and the enclosing log share. */
  runId: string;
  /** The authored flow's SHA-256 identity, 64 lowercase hex digits. */
  flowFingerprint: string;
  /** START's narrowed per-run wiring. */
  startWiringOverride: Readonly<Record<string, unknown>>;
  /** The complete resolved agent configuration by stable static path. */
  effectiveLlmByStaticPath: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  /** The exact non-negative dispatcher worker count. */
  numWorkers: number;
  /** Canonical snapshots, one per memory key; `null` (the default) disables memory. */
  memorySnapshots?: readonly MemorySnapshotEnvelope[] | null;
  /** Whether accepted outcomes must carry forward capture for learning; defaults to `false`. */
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
  /** The flow files the run compiled; `null` (the default) for a flow built in code. */
  flow?: FlowSource | null;
  /** Each skill an AGENT grants, by name; defaults to `{}`. */
  skills?: Readonly<Record<string, RecordedSkill>>;
  /** Each tool an AGENT grants, groups expanded, by name; defaults to `{}`. */
  tools?: Readonly<Record<string, RecordedTool>>;
  /** Each tool group an AGENT grants, by name: its tools' names; defaults to `{}`. */
  toolGroups?: Readonly<Record<string, readonly string[]>>;
  /** The yian version that starts the run; defaults to this one's. */
  yianVersion?: string;
  /** The run that started this one; `null` (the default) when none did. */
  parent?: RunParent | null;
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
  readonly closedCalls: readonly string[];
}

/**
 * Appending one canonical step to an in-flight node's steps; `closedCalls` is the host call a
 * `pending` step registers already closed, on a node holding a pending interrupt.
 */
interface NodeStepDelta {
  readonly kind: "step";
  readonly nodeId: string;
  readonly step: Readonly<JsonObject>;
  readonly closedCalls: readonly string[];
}

/** Recording a stop over the in-flight nodes; it changes nothing but the sequence. */
interface RunStoppedDelta {
  readonly kind: "stopped";
  readonly nodeIds: readonly string[];
}

/** Moving an in-flight node to the waiting set. */
interface NodeWaitingDelta {
  readonly kind: "waiting";
  readonly nodeId: string;
}

/** Moving a waiting node back in flight; `closedCalls` are the timed-out host calls it closes. */
interface NodeWokenDelta {
  readonly kind: "woken";
  readonly nodeId: string;
  readonly closedCalls: readonly string[];
}

/** Recording that every unsettled node waits; it changes nothing but the sequence. */
interface RunWaitingDelta {
  readonly kind: "runWaiting";
  readonly nodeIds: readonly string[];
}

/** Closing an inbox; the waits fold applies it. */
interface InboxClosedDelta {
  readonly kind: "inboxClosed";
  readonly inbox: string;
}

/** Withdrawing unread sends from an inbox; the waits fold applies it, and the commit drops their values. */
interface InboxWithdrawnDelta {
  readonly kind: "withdrawn";
  readonly inbox: string;
  readonly events: readonly number[];
}

/** Interrupting steer-inbox readers; the waits fold applies it. `closedCalls` are the host calls it closes. */
interface InterruptedDelta {
  readonly kind: "interrupted";
  readonly nodeIds: readonly string[];
  readonly closedCalls: readonly string[];
}

/**
 * What a pool commit is: a run-start commit (`initial`), which applies at once; a node's own output
 * commit (`node`), which waits for its success; a spawner's commit for a child START (`expansion`),
 * which waits under the START's key; a send to a START inbox (`send`), which applies at once to the
 * inbox store; or a node's messages commit (`messages`), which waits for its success.
 */
type CommitKind = "initial" | "node" | "expansion" | "send" | "messages";

/** One pool commit; `valueJson` is the value's canonical typed wire text. */
interface CommitPlan extends PlanAnchors {
  readonly kind: "commit";
  readonly nodeId: string;
  readonly key: string;
  /** The value's canonical typed wire text, or `null` for a derived or take commit. */
  readonly valueJson: string | null;
  readonly from: ReadonlyMap<string, number> | null;
  readonly take: TakeCommit | null;
  readonly commitKind: CommitKind;
  /** A messages commit's rebuilt value, as canonical typed wire text; `null` for any other commit. */
  readonly messagesJson: string | null;
}

/** One dispatch, resume, step, stop, wait, wake, run wait, inbox close, withdraw or interrupt. */
interface FrontierPlan extends PlanAnchors {
  readonly kind: "frontier";
  readonly delta:
    | NodeStartedDelta
    | RunResumedDelta
    | NodeStepDelta
    | RunStoppedDelta
    | NodeWaitingDelta
    | NodeWokenDelta
    | RunWaitingDelta
    | InboxClosedDelta
    | InboxWithdrawnDelta
    | InterruptedDelta;
}

/** The touched coordinates of one successful output commit. */
interface SuccessDelta {
  readonly nodeId: string;
  readonly commitTarget: string;
  readonly valueJson: string;
  readonly outgoingEdgeIds: readonly string[];
  /** The head of each of `outgoingEdgeIds`, in the same order. */
  readonly edgeHeads: readonly string[];
  readonly newlyReadyIds: readonly string[];
  readonly observerCommitAs: string | null;
  /** Whether the node honored an interrupt (its `interrupted` step) before this success. */
  readonly interrupted: boolean;
  /** The sends a take commit consumed, whose values leave the inbox store. */
  readonly consumedEvents: readonly number[];
  /** The node's messages commit this success also accepts, or `null`. */
  readonly messages: PendingMessages | null;
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
  /** The head of each of `edgeUpdates`, in the same order. */
  readonly edgeHeads: readonly string[];
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

/** One canonical failure; `closedCalls` are the open host calls it closes. */
interface FailurePlan extends PlanAnchors {
  readonly kind: "failure";
  readonly failure: Failure;
  readonly closedCalls: readonly string[];
}

/**
 * A run ending with no completing node: a cancel, or a failure at quiescence with nodes waiting
 * (`failure` set). `closedCalls` are the open host calls it closes.
 */
interface EndingPlan extends PlanAnchors {
  readonly kind: "ending";
  readonly failure: Failure | null;
  readonly closedCalls: readonly string[];
}

/** The scalar authority of one flat deterministic expansion. */
interface GrowthDelta {
  readonly spawnerId: string;
  /** The spawner's expansion commits, admitted: each now waits on its child START's success. */
  readonly admitted: readonly PendingCommit[];
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
  /** The canonical seed, live or rebuilt; the record holds none. */
  readonly seed: JsonValue;
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
type PreparedPlan =
  | RunStartedPlan
  | CommitPlan
  | FrontierPlan
  | SuccessPlan
  | RoutePlan
  | FailurePlan
  | EndingPlan
  | GrowthPlan
  | BackwardPlan;

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
    // The historical time is kept: replay compares records, not when they were appended.
    const candidate = encodeRecord(new RecordEnvelope(envelope.schema, this.runId, sequence, record, envelope.at));
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
 * Prepare canonical initialization without mutating `state`. The run start then commits the
 * system values and the START inputs (`prepareInitialCommit`); no node is ready before.
 *
 * @param state A pristine state whose private graph holds its synthesized START node.
 * @param data The identity, configuration and worker count of the run.
 * @returns The canonical `RunStartedRecord`, its frozen total plan and its observer.
 * @throws TypeError when `state` isn't exactly a `RunState`, or the worker count isn't a safe
 *   integer.
 * @throws RangeError when the state isn't pristine, START is missing, or a field breaks the
 *   public run-record wire contract.
 */
export function prepareRunStarted(state: RunState, data: RunStartData): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareRunStarted expects a RunState");
  requirePristine(state);
  const startId = state.flow.startId;
  if (!state.flow.nodes.has(startId)) throw new RangeError(`flow START node ${reprText(startId)} is missing`);
  requireWorkerCount(data.numWorkers);

  const candidate = new RunStartedRecord({
    flowFingerprint: data.flowFingerprint,
    startWiringOverride: data.startWiringOverride,
    effectiveLlmByStaticPath: data.effectiveLlmByStaticPath,
    numWorkers: data.numWorkers,
    memorySnapshots: data.memorySnapshots ?? null,
    learnable: data.learnable ?? false,
    memories: data.memories ?? null,
    memoryStacks: data.memoryStacks ?? null,
    defaults: data.defaults ?? {},
    callTimeLlm: data.callTimeLlm ?? null,
    flow: data.flow ?? null,
    skills: data.skills ?? {},
    tools: data.tools ?? {},
    toolGroups: data.toolGroups ?? {},
    yianVersion: data.yianVersion,
    parent: data.parent ?? null,
  });
  const { envelopeJson, envelope, canonical } = canonicalEnvelope(state, candidate, [RunStartedRecord], "run-start", data.runId);
  const plan: RunStartedPlan = Object.freeze({ kind: "runStarted", ...planAnchors(state, envelopeJson, envelope) });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, [new RunStarted()]);
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
 * Prepare appending one step an in-flight node recorded while it runs, numbered as the node's
 * next step (`n`, counting the steps of its earlier starts).
 *
 * @param state An initialized nonterminal state in which `nodeId` is in flight.
 * @param nodeId The exact id of the in-flight node recording the step.
 * @param step A JSON object with a string `kind`, at any depth; only the node gives it meaning.
 * @returns The canonical `NodeStepRecord` and a detached append-before-apply plan, with no
 *   observers.
 * @throws TypeError when `state`, `nodeId` or `step` has the wrong type, or `step` has no kind.
 * @throws RangeError when the state is uninitialized or terminal, or `nodeId` isn't in flight.
 * @throws TypeCheckError when `step` isn't a JSON value.
 */
export function prepareNodeStep(state: RunState, nodeId: string, step: unknown): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareNodeStep expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  if (!isRecord(step)) throw new TypeError("node step must be a JSON object");
  // No nesting limit: a step is node detail, not a yian value, and an AGENT's model call step holds
  // a reply whose tool call arguments may themselves sit at the limit.
  // Steps outlive a resume, so `n` keeps counting across the node's starts.
  const n = (state.nodeSteps.get(nodeId)?.length ?? 0) + 1;
  return prepareRecord(state, new NodeStepRecord(nodeId, n, step));
}

/**
 * Prepare one commit of the run start: a system value (`key` `system.<name>`) or, last, the START
 * inputs (`key` the START's id). It applies at once, and the START inputs' commit makes START
 * ready.
 *
 * @param state An initialized state whose run start is incomplete.
 * @param key `system.<name>`, or the START's id.
 * @param value The value; the START inputs must be a record.
 * @returns The canonical `PoolCommitRecord`, which names the START, and its plan, with no
 *   observers.
 * @throws TypeError when `state` or `key` has the wrong type.
 * @throws RangeError when the run start is complete, `key` is neither form, the system value is
 *   already committed, `system.run_id` isn't the run id, or the START inputs aren't a record or
 *   come before `system.run_id`.
 * @throws TypeCheckError when `value` isn't a yian value.
 */
export function prepareInitialCommit(state: RunState, key: string, value: unknown): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareInitialCommit expects a RunState");
  if (typeof key !== "string") throw new TypeError("key must be a string");
  const [wire] = canonicalTyped(value, null);
  return prepareRecord(state, new PoolCommitRecord(state.flow.startId, key, wire));
}

/**
 * Prepare a host's send of `event` to the top-level inbox `inbox`: a commit by the START under
 * `<START>.<inbox>`, applied at once to the inbox store, never to the pool.
 *
 * @param state The run's state.
 * @param inbox The name of one of the START's `Inbox(T)` params.
 * @param event The value, checked against `T`.
 * @returns The canonical `PoolCommitRecord` and its plan, with no observers.
 * @throws TypeError when `state` or `inbox` has the wrong type.
 * @throws DeliveryRefused, in this order: `ended` once the run has a terminal result;
 *   `not_started` before its run start is complete; `unknown_inbox` when the START declares no
 *   such inbox; `inbox_closed` once it is closed; `invalid_value` when `event` fails `T` (the
 *   `TypeCheckError` is the cause); `too_large` when its run-log JSON is over the run's
 *   `MAX_EVENT_BYTES`.
 */
export function prepareSend(state: RunState, inbox: string, event: unknown): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareSend expects a RunState");
  if (typeof inbox !== "string") throw new TypeError("inbox must be a string");
  requireDeliverable(state);
  const element = requireDeclaredInbox(state, inbox);
  if (state.waits.inboxes.get(inbox)?.closed === true) {
    throw new DeliveryRefused("inbox_closed", `inbox ${reprText(inbox)} is closed`);
  }
  let wire: TypedWire;
  try {
    [wire] = canonicalTyped(event, element);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    throw new DeliveryRefused("invalid_value", `inbox ${reprText(inbox)} refuses the event: ${error.message}`, {
      cause: error,
    });
  }
  const size = eventBytes(wire.value);
  if (size > state.runDefaults.maxEventBytes) {
    throw new DeliveryRefused("too_large", eventTooLarge(`the event to inbox ${reprText(inbox)}`, size, state));
  }
  return prepareRecord(state, new PoolCommitRecord(state.flow.startId, sendKey(state, inbox), wire));
}

/**
 * Return the element type of the START's top-level inbox `inbox`.
 *
 * @throws DeliveryRefused `unknown_inbox` when the START declares no such inbox.
 */
function requireDeclaredInbox(state: RunState, inbox: string): Type {
  const element = inboxElementType(state, inbox);
  if (element === null) throw new DeliveryRefused("unknown_inbox", `the run has no inbox ${reprText(inbox)}`);
  return element;
}

/** The message for a value of `size` bytes over the run's `MAX_EVENT_BYTES`; `what` names it. */
function eventTooLarge(what: string, size: number, state: RunState): string {
  return `${what} is ${size} bytes, over the run's limit of ${state.runDefaults.maxEventBytes} (YIAN_MAX_EVENT_BYTES)`;
}

/** The pool key a send to `inbox` commits under: the START's id, a dot, then the inbox's name. */
function sendKey(state: RunState, inbox: string): string {
  return `${state.flow.startId}.${inbox}`;
}

/** The element type `T` of the START's top-level `Inbox(T)` param `name`, or `null` when it declares none. */
export function inboxElementType(state: RunState, name: string): Type | null {
  const start = state.flow.nodes.get(state.flow.startId);
  const type = (start?.params ?? []).find((param) => param.name === name)?.contract?.type;
  return type !== undefined && isInboxType(type) ? type.element! : null;
}

/**
 * Prepare a host's close of the top-level inbox `inbox`: later sends are refused, takes still read
 * what is queued, and a take already waiting on it with nothing assigned is answered.
 *
 * @param state The run's state.
 * @param inbox The name of one of the START's `Inbox(T)` params.
 * @returns The canonical `InboxClosedRecord` and its plan, with no observers; `null` when the
 *   inbox is already closed.
 * @throws TypeError when `state` or `inbox` has the wrong type.
 * @throws DeliveryRefused, in this order: `ended` once the run has a terminal result;
 *   `not_started` before its run start is complete; `unknown_inbox` when the START declares no
 *   such inbox.
 */
export function prepareInboxClosed(state: RunState, inbox: string): PreparedTransition | null {
  if (!isExactly(state, RunState)) throw new TypeError("prepareInboxClosed expects a RunState");
  if (typeof inbox !== "string") throw new TypeError("inbox must be a string");
  requireDeliverable(state);
  requireDeclaredInbox(state, inbox);
  if (state.waits.inboxes.get(inbox)?.closed === true) return null;
  return prepareRecord(state, new InboxClosedRecord(inbox));
}

/**
 * Prepare a host's withdraw of every unread send of the top-level inbox `inbox`: its unread sends
 * and those assigned to a take whose node hasn't woken, which then waits again (or, on a closed
 * inbox, is answered by the close). The withdrawn sends' values are dropped.
 *
 * @param state The run's state.
 * @param inbox The name of one of the START's `Inbox(T)` params.
 * @returns The canonical `InboxWithdrawnRecord` and its plan, with no observers; `null` when
 *   nothing of the inbox is unread.
 * @throws TypeError when `state` or `inbox` has the wrong type.
 * @throws DeliveryRefused, in this order: `ended` once the run has a terminal result;
 *   `not_started` before its run start is complete; `unknown_inbox` when the START declares no
 *   such inbox.
 */
export function prepareWithdraw(state: RunState, inbox: string): PreparedTransition | null {
  if (!isExactly(state, RunState)) throw new TypeError("prepareWithdraw expects a RunState");
  if (typeof inbox !== "string") throw new TypeError("inbox must be a string");
  requireDeliverable(state);
  requireDeclaredInbox(state, inbox);
  const events = withdrawableEvents(state.waits, inbox);
  if (events.length === 0) return null;
  return prepareRecord(state, new InboxWithdrawnRecord(inbox, events));
}

/**
 * Prepare a host's cancel: `run_cancelled` ends the run, closing its open calls; every later record
 * is refused.
 *
 * @param state The run's state.
 * @returns The canonical `RunCancelledRecord` and its plan; its observers are `RunCancelled`, then
 *   a `CallClosed` for each open host call (takes and sleeps get none).
 * @throws TypeError when `state` has the wrong type.
 * @throws DeliveryRefused `ended` once the run has a terminal result (a second cancel among them),
 *   then `not_started` before its run start is complete.
 */
export function prepareRunCancelled(state: RunState): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareRunCancelled expects a RunState");
  requireDeliverable(state);
  return prepareRecord(state, new RunCancelledRecord());
}

/**
 * Prepare a host's interrupt (a Stop) of `inbox`: an `interrupted` record naming the agents
 * reading it as their steer inbox, `running` (the engine's in-flight readers, in flight order)
 * then each waiting node whose `node_waiting` named `inbox`, leaving out nodes already being
 * stopped (holding a pending interrupt, or running after honoring one). With no reader it names
 * none and only records the Stop, which a reader opening the inbox while it stands honors. The
 * record closes the named nodes' open calls (`interrupted`); a closed inbox can still be
 * interrupted, since a reply may still run on what it already read.
 *
 * @param state The run's state.
 * @param inbox The name of one of the START's `Inbox(T)` params.
 * @param running The in-flight nodes reading `inbox` as their steer inbox, which only the engine
 *   knows; `null` offline, where any in-flight node counts as an open reader for the idle rule,
 *   and any in-flight node that reads a steer inbox as a current reader for "already stopping".
 * @returns The canonical `InterruptedRecord`, its plan and a `CallClosed {reason: "interrupted"}`
 *   per open host call of the named nodes; `null` when the run waits for the inbox's next message
 *   (a take of it waits with nothing unread and no reader is open), or when it is already
 *   stopping (a Stop on the inbox stands, or every current reader holds or has honored an
 *   interrupt).
 * @throws TypeError when an argument has the wrong type.
 * @throws DeliveryRefused, in this order: `ended` once the run has a terminal result;
 *   `not_started` before its run start is complete; `unknown_inbox` when the START declares no
 *   such inbox.
 * @throws RangeError when a running node can't be interrupted (see `validateInterrupted`).
 */
export function prepareInterrupt(state: RunState, inbox: string, running: readonly string[] | null = null): PreparedTransition | null {
  if (!isExactly(state, RunState)) throw new TypeError("prepareInterrupt expects a RunState");
  if (typeof inbox !== "string") throw new TypeError("inbox must be a string");
  if (running !== null && (!Array.isArray(running) || running.some((nodeId) => typeof nodeId !== "string"))) {
    throw new TypeError("running must be null or an array of node ids");
  }
  requireDeliverable(state);
  requireDeclaredInbox(state, inbox);
  const w = state.waits;
  // Offline no engine says which in-flight node opened the inbox, so any counts as a reader.
  const readersOpen = running === null ? state.inFlight.size > 0 : running.length > 0;
  if (!readersOpen && awaitsNextMessage(w, inbox)) return null;
  // Already stopping: a second Esc never carries a Stop into the next reply.
  if (w.stops.has(inbox)) return null;
  // Offline every in-flight node that reads a steer inbox may have opened this one, so each counts
  // as a current reader, but none is named. Only an interrupt of this inbox makes one stopping
  // (a crashed reader of it still holding its interrupt); one of another inbox leaves it a
  // possible reader of this one.
  const runningReaders = running ?? [...state.inFlight].filter((nodeId) => state.flow.nodes.get(nodeId)?.readsSteerInbox === true);
  const stopping = (nodeId: string): boolean =>
    w._interruptInboxes.get(nodeId) === inbox &&
    (w.interrupts.has(nodeId) || (runningReaders.includes(nodeId) && w._honored.has(nodeId)));
  const waiting = waitingReaders(state, inbox);
  const readers = [...runningReaders, ...waiting];
  if (readers.length > 0 && readers.every(stopping)) return null;
  const named = running === null ? waiting : readers;
  return prepareRecord(state, new InterruptedRecord(inbox, named.filter((nodeId) => !stopping(nodeId))));
}

/**
 * Prepare the remembered Stop on `inbox` for `nodeId`, which has just opened it as its steer
 * inbox: an `interrupted` record naming it and each waiting reader without a pending interrupt.
 * The caller checks that a Stop stands and that the node holds and has honored none.
 *
 * @param state The run's state.
 * @param inbox The name of one of the START's `Inbox(T)` params.
 * @param nodeId The in-flight node that opened `inbox`.
 * @returns The canonical `InterruptedRecord`, its plan and a `CallClosed {reason: "interrupted"}`
 *   per open host call of the named nodes.
 * @throws TypeError when an argument has the wrong type.
 * @throws DeliveryRefused as `prepareInterrupt`.
 * @throws RangeError when the node can't be interrupted (see `validateInterrupted`).
 */
export function prepareAppliedStop(state: RunState, inbox: string, nodeId: string): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareAppliedStop expects a RunState");
  if (typeof inbox !== "string") throw new TypeError("inbox must be a string");
  if (typeof nodeId !== "string") throw new TypeError("nodeId must be a string");
  requireDeliverable(state);
  requireDeclaredInbox(state, inbox);
  const waiting = waitingReaders(state, inbox).filter((id) => id !== nodeId && !state.waits.interrupts.has(id));
  return prepareRecord(state, new InterruptedRecord(inbox, [nodeId, ...waiting]));
}

/** The waiting nodes whose `node_waiting` named `inbox` as their steer inbox, in waiting order. */
function waitingReaders(state: RunState, inbox: string): string[] {
  return [...state.waits.waiting].filter(([, node]) => node.steerInbox === inbox).map(([nodeId]) => nodeId);
}

/**
 * Prepare ending the run failed at quiescence with nodes waiting, once a route proved END
 * unreachable: whatever the waiting nodes do, the run can't succeed.
 *
 * @param state The run's state.
 * @returns The canonical `RunFailedRecord` holding the terminal-unreachable failure, and its plan;
 *   its observers are `RunFailed`, then a `CallClosed` for each open host call.
 * @throws TypeError when `state` has the wrong type.
 * @throws RangeError when no route proved END unreachable, or the run isn't quiescent with nodes
 *   waiting and none wakeable.
 */
export function prepareRunFailed(state: RunState): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareRunFailed expects a RunState");
  requireInitializedNonterminal(state);
  const failure = state.terminalUnreachableFailure;
  if (failure === null) throw new RangeError("RunFailed requires a route that proved END unreachable");
  return prepareRecord(state, new RunFailedRecord(failure));
}

/**
 * Prepare a host's answer to the pending host call `callId`: the `tool_result` step its node
 * shapes (`Node.waitResultStep`), on that node, whether it waits or still runs.
 *
 * @param state The run's state.
 * @param callId The call's id, as its `pending` step names it.
 * @param result The answer: `{output}` or `{error}`.
 * @returns The canonical `NodeStepRecord` and its plan, with no observers; `null` when the call
 *   is already answered with exactly this step (a host's repeat).
 * @throws TypeError when `state` or `callId` has the wrong type.
 * @throws DeliveryRefused, in this order: `ended` once the run has a terminal result;
 *   `not_started` before its run start is complete; `unknown_call` when no open or settled call
 *   has the id (its node settled, or its handler hasn't returned `pending()`); `not_host_call` for
 *   a take or sleep; `call_settled` (with the call's state) for an answered, timed-out or closed
 *   call, unless it is an identical repeat; `invalid_value` when the node refuses the answer (the
 *   `TypeCheckError` is the cause); `too_large` when the step is over the run's
 *   `MAX_EVENT_BYTES`.
 */
export function prepareComplete(state: RunState, callId: string, result: CompleteResult): PreparedTransition | null {
  if (!isExactly(state, RunState)) throw new TypeError("prepareComplete expects a RunState");
  if (typeof callId !== "string") throw new TypeError("call_id must be a string");
  requireDeliverable(state);
  const call = state.waits.calls.get(callId);
  if (call === undefined) throw new DeliveryRefused("unknown_call", `the run has no pending call ${reprText(callId)}`);
  if (call.tool === TAKE || call.tool === SLEEP) {
    throw new DeliveryRefused("not_host_call", `call ${reprText(callId)} is a ${call.tool}, which the run settles itself`);
  }
  const node = state.flow.nodes.get(call.nodeId)!;
  const view = callView(state, call);
  if (call.state !== "open") {
    if (call.state === "answered" && sameStep(shapedRepeat(node, view, result), view.result)) return null;
    throw new DeliveryRefused("call_settled", `call ${reprText(callId)} is already ${settledText(call)}`, { callState: call.state });
  }
  let step: JsonObject;
  try {
    step = node.waitResultStep(view, result);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    throw new DeliveryRefused("invalid_value", `call ${reprText(callId)} refuses the answer: ${error.message}`, { cause: error });
  }
  const size = eventBytes(step);
  if (size > state.runDefaults.maxEventBytes) {
    throw new DeliveryRefused("too_large", eventTooLarge(`the answer to ${reprText(callId)}`, size, state));
  }
  return prepareNodeStep(state, call.nodeId, step);
}

/** How a settled call's state reads in a message: `closed (<reason>)`, `answered` or `timed out`. */
function settledText(call: Call): string {
  return call.state === "closed" ? `closed (${call.closedReason})` : call.state === "answered" ? "answered" : "timed out";
}

/** The step `node` shapes for a repeated answer, or `null` when it refuses the answer. */
function shapedRepeat(node: Node, view: CallView, result: CompleteResult): JsonObject | null {
  try {
    return node.waitResultStep(view, result);
  } catch (error) {
    if (error instanceof TypeCheckError) return null;
    throw error;
  }
}

/** Whether two steps have the same canonical JSON; `null` matches nothing. */
function sameStep(a: Readonly<JsonObject> | null, b: Readonly<JsonObject> | null): boolean {
  return a !== null && b !== null && logJson(canonicalSeed(a)[0]) === logJson(canonicalSeed(b)[0]);
}

/** `call` as its node reads it: its state, closing reason, answer and deadline. */
function callView(state: RunState, call: Call): CallView {
  return Object.freeze({
    callId: call.callId,
    tool: call.tool,
    resultKey: call.resultKey,
    state: call.state,
    closedReason: call.closedReason,
    result: callResultStep(state, call.callId),
    deadline: call.deadline,
  });
}

/**
 * Return the `tool_result` step that answered the registered call `callId`, as its node recorded
 * it; `null` when the call is unknown or no step answered it (it is open, closed, timed out, or a
 * take answered by its events).
 */
export function callResultStep(state: RunState, callId: string): Readonly<JsonObject> | null {
  const call = state.waits.calls.get(callId);
  if (call === undefined || call.state !== "answered" || call.resultSeq === null) return null;
  // The waits fold recorded the answering step's seq; `nodeStepSeqs` is index-aligned with `nodeSteps`.
  const at = state.nodeStepSeqs.get(call.nodeId)?.indexOf(call.resultSeq);
  return at === undefined || at < 0 ? null : state.nodeSteps.get(call.nodeId)![at]!;
}

/**
 * Commit the run start's commits from `pool`, right after `RunStarted`: its system values in key
 * order, then its START inputs.
 *
 * @param log The run's log, which holds `RunStarted` and nothing after it.
 * @param state The state `RunStarted` initialized.
 * @param pool The run's pool: the system values, and the START inputs as its only stored value.
 * @throws RangeError when the pool's START isn't the flow's, it stores another value, or a commit
 *   is refused (see `prepareInitialCommit`).
 */
export function commitInitialPool(log: RunLog, state: RunState, pool: VariablePool): void {
  const startId = state.flow.startId;
  if (pool.startId !== startId || [...pool.store.keys()].some((key) => key !== startId)) {
    throw new RangeError(`the initial pool must store only the START inputs, under ${reprText(startId)}`);
  }
  for (const name of [...pool.system.keys()].sort(compareCodeUnits)) {
    commitPrepared(log, state, prepareInitialCommit(state, `${SYSTEM_PREFIX}${name}`, pool.system.get(name)));
  }
  commitPrepared(log, state, prepareInitialCommit(state, startId, pool.store.get(startId) ?? null));
}

/**
 * Prepare one in-flight node's output commit. It waits on the node until the node's success
 * (`prepareNodeSucceeded`) accepts it.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight and no pending commit.
 * @param nodeId The exact id of the executing node that produced `output`.
 * @param output The yian value (or `TypedValue`) to check against the effective target's output
 *   type; the checks and the stored value read it as given.
 * @param options.commitAs The explicit commit target; `null` (the default) falls back to the
 *   executing node's baked target, then to `nodeId` itself.
 * @param options.from A derived commit's sources (see `isDerivedCommit`); `null` by default.
 * @returns The canonical `PoolCommitRecord` and a detached append-before-apply plan, with no
 *   observers.
 * @throws TypeError when `state` or `nodeId` has the wrong type, or the target's contract is
 *   malformed.
 * @throws RangeError when the state, the executing node or its baked target is invalid, the node
 *   already has a pending commit, or `options.from` is set when the node isn't derived or missing
 *   when it is.
 * @throws NodeDataError when `commitAs` or `output` is rejected: the output nests too deep,
 *   violates its target's type, or isn't a yian value. `original` holds the value layer's error.
 */
export function preparePoolCommit(
  state: RunState,
  nodeId: string,
  output: unknown,
  options: { commitAs?: string | null; from?: ReadonlyMap<string, number> | null } = {},
): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("preparePoolCommit expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  const commitAs: unknown = options.commitAs ?? null;
  if (commitAs !== null && typeof commitAs !== "string") {
    throw new NodeDataError(new TypeError("commit_as must be a string or null"));
  }
  if (commitAs === "") throw new NodeDataError(new RangeError("commit_as must be non-empty when supplied"));
  requireInitializedNonterminal(state);
  requireCommittingNode(state, nodeId);
  requireValueCommitNode(state, nodeId);
  const from = options.from ?? null;
  requireCommitFrom(state, nodeId, from);

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
  // Replay (`validateNodeCommit`) skips the nesting check on purpose, so logs recorded before the
  // limit still replay.
  const enclosingLevels = node.commitsInputRecord ? -1 : 0;
  let valueWire: TypedWire;
  try {
    rejectDeepNesting(output, "output", { enclosingLevels });
    [valueWire] = canonicalOutputValue(output, outputContract, declaredType);
  } catch (error) {
    const path = error instanceof TypeCheckError ? error.path : [];
    throw new NodeDataError(error, outputContractOrigin(outputContract, path));
  }
  // A derived commit records only its sources; its success rebuilds the value, live as on replay.
  const record = from === null ? new PoolCommitRecord(nodeId, target, valueWire) : new PoolCommitRecord(nodeId, target, null, { from });
  return prepareCommitTransition(state, record, "node");
}

/**
 * Prepare an in-flight take node's commit of the events it takes, `commit`, under its target. Its
 * success (`prepareNodeSucceeded`, with no `output`) rebuilds the value `{events, closed}` from
 * the inbox store.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight and no pending commit.
 * @param nodeId The take node.
 * @param commit The inbox, the send seqs it consumes in order, and whether it saw the inbox closed.
 * @returns The canonical `PoolCommitRecord` and its plan, with no observers.
 * @throws TypeError when `state` or `nodeId` has the wrong type.
 * @throws RangeError when the node isn't an in-flight take, or the commit isn't the one the take's
 *   call or the inbox allows (see `validateTakeCommit`).
 */
export function prepareTakeCommit(state: RunState, nodeId: string, commit: TakeCommit): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareTakeCommit expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  requireInitializedNonterminal(state);
  const node = state.flow.nodes.get(nodeId);
  if (node === undefined) throw new RangeError(`take node ${reprText(nodeId)} does not exist in the flow`);
  return prepareRecord(state, new PoolCommitRecord(nodeId, node.commitAs ?? nodeId, null, { take: commit }));
}

/**
 * Prepare accepting an in-flight node's pending commit and its newly ready successors.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight and its commit pending.
 * @param nodeId The exact id of the executing node.
 * @param options.forwardCapture The canonical binding capture, required exactly when learning. A
 *   success naming a derived or expansion commit doesn't record it: the capture is rebuilt with
 *   the commit.
 * @param options.output The value the node evaluated to; required for a derived commit, whose
 *   value rebuilt from its `from` must equal it.
 * @returns The canonical `NodeSucceededRecord`, naming the commit's seq, and a detached
 *   append-before-apply plan.
 * @throws TypeError when a carrier has the wrong type, or a derived commit's `output` is missing.
 * @throws RangeError when the state or the executing node is invalid, the node has no pending
 *   commit, or its derived value doesn't rebuild to `options.output`.
 */
export function prepareNodeSucceeded(
  state: RunState,
  nodeId: string,
  options: OutcomeCarriers & { output?: unknown } = {},
): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareNodeSucceeded expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  requireInitializedNonterminal(state);
  requireOutcomeNode(state, nodeId, "NodeSucceeded");
  const pending = state.pendingCommits.get(nodeId);
  if (pending === undefined) throw new RangeError(`NodeSucceeded node ${reprText(nodeId)} has no pending pool_commit`);
  if (pending.valueJson === null && pending.take === null && !Object.hasOwn(options, "output")) {
    throw new TypeError(`NodeSucceeded of derived node ${reprText(nodeId)} needs the value it evaluated to (output)`);
  }
  // A take commit holds no value either, but its node isn't derived: its capture is recorded.
  const derived = (pending.valueJson === null && pending.take === null) || pending.spawner !== null;
  return prepareSuccessTransition(
    state,
    new NodeSucceededRecord(nodeId, pending.seq, {
      forwardCapture: derived ? null : (options.forwardCapture ?? null),
      messages: state.pendingMessages.get(nodeId)?.seq ?? null,
    }),
    Object.hasOwn(options, "output") ? { output: options.output } : null,
  );
}

/**
 * Prepare a messages commit: the seqs of the steps `nodeId`'s new messages are made of
 * (`Node.messageSteps`), written after its output commit and before the success that accepts both.
 * The reducer rebuilds the value (`Node.rebuildMessages`) now, so a rebuild that fails appends
 * nothing.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight and its output commit
 *   pending.
 * @param nodeId The exact id of the executing node; its kind produces messages.
 * @returns The canonical `PoolCommitRecord` under `messagesKey(nodeId)` holding `steps`, and its
 *   plan, with no observers.
 * @throws TypeError when `state` or `nodeId` has the wrong type.
 * @throws RangeError as `validateMessagesCommit`.
 */
export function prepareMessagesCommit(state: RunState, nodeId: string): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareMessagesCommit expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  requireInitializedNonterminal(state);
  if (state.flow.nodes.get(nodeId)?.producesMessages !== true) {
    throw new RangeError(`a messages commit by node ${reprText(nodeId)}, which produces no messages`);
  }
  return prepareRecord(state, new PoolCommitRecord(nodeId, messagesKey(nodeId), null, { steps: messageStepSeqs(state, nodeId) }));
}

/**
 * Prepare one selected route and its derived recursive skip closure.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight.
 * @param nodeId The exact non-empty id of the executing router.
 * @param handle The exact non-empty handle the router selected.
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
  return prepareRouteTransition(
    state,
    new NodeRoutedRecord(nodeId, handle, {
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
    forwardCapture: options.forwardCapture ?? null,
  });
  requireInitializedNonterminal(state);
  return prepareFailureTransition(state, record);
}

/**
 * Prepare a recovery segment from the state's ordered unfinished starts.
 *
 * The retry list comes only from the insertion-ordered `state.inFlight` (a waiting node isn't in
 * flight, so it isn't retried), and the closed calls are the retried agents' open calls; the
 * caller chooses only the worker count of the new segment.
 *
 * @param state An initialized nonterminal state rebuilt through a complete log prefix.
 * @param options.numWorkers The exact non-negative worker count; zero selects serial execution.
 * @returns The canonical `RunResumedRecord`, whose retries keep dispatch order, and its observers:
 *   `RunResumed`, then a `CallClosed` for each closed call.
 * @throws TypeError when `state` has the wrong type or the worker count isn't a safe integer.
 * @throws RangeError when the worker count is negative, or the state is uninitialized or terminal.
 */
export function prepareRunResumed(state: RunState, options: { numWorkers: number }): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareRunResumed expects a RunState");
  const { numWorkers } = options;
  requireWorkerCount(numWorkers);
  requireInitializedNonterminal(state);
  return prepareRecord(state, new RunResumedRecord([...state.inFlight], numWorkers, retriedAgentCalls(state)));
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

/**
 * Prepare moving an in-flight node to the waiting set: it holds no worker until `node_woken`.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight and no pending commit.
 * @param nodeId The node whose evaluation ended waiting.
 * @param options.callIds Every call the node waits on: each open call it registered since its
 *   last wake or that was still open at it (a timeout wake leaves the others open), and the call
 *   `waits` registers.
 * @param options.waits The take or sleep call this record registers, at most one; none by default.
 * @param options.steerInbox The steer inbox the node opened, which an interrupt of that inbox
 *   reaches while it waits; none by default.
 * @returns The canonical `NodeWaitingRecord`, its plan and its `NodeWaiting` observer.
 * @throws TypeError when an argument has the wrong type.
 * @throws RangeError when the record breaks a wait rule (see `validateNodeWaiting`).
 */
export function prepareNodeWaiting(
  state: RunState,
  nodeId: string,
  options: { callIds: readonly string[]; waits?: readonly WaitEntry[]; steerInbox?: string | null },
): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareNodeWaiting expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  const record = new NodeWaitingRecord(nodeId, options.callIds, {
    waits: options.waits ?? [],
    steerInbox: options.steerInbox ?? null,
  });
  return prepareRecord(state, record);
}

/**
 * Prepare moving a waiting node back in flight, timing out `timedOut` (calls whose deadline the
 * engine saw pass; the reducer reads no clock).
 *
 * @param state An initialized nonterminal state in which `nodeId` waits.
 * @param nodeId The waiting node.
 * @param timedOut Open calls of the node with a deadline; none by default.
 * @returns The canonical `NodeWokenRecord`, its plan and its observers: `NodeWoken`, then a
 *   `CallClosed` for each timed-out host call.
 * @throws TypeError when an argument has the wrong type.
 * @throws RangeError when the node isn't waiting, a timed-out id isn't an open call of it with a
 *   deadline, or, with no `timedOut` and no interrupt, it still waits on an open call.
 */
export function prepareNodeWoken(state: RunState, nodeId: string, timedOut: readonly string[] = []): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareNodeWoken expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  return prepareRecord(state, new NodeWokenRecord(nodeId, timedOut));
}

/**
 * Prepare recording that the run waits: nothing runs or is ready, and every unsettled node waits
 * on an open call.
 *
 * @returns The canonical `RunWaitingRecord` naming the waiting nodes in `node_waiting` order, its
 *   plan and its `RunWaiting` observer.
 * @throws TypeError when `state` has the wrong type.
 * @throws RangeError when a node is in flight or ready, none waits, one is wakeable, or the last
 *   frontier record is already a `run_waiting`.
 */
export function prepareRunWaiting(state: RunState): PreparedTransition {
  if (!isExactly(state, RunState)) throw new TypeError("prepareRunWaiting expects a RunState");
  requireInitializedNonterminal(state);
  requireWaitingQuiescence(state, "RunWaiting");
  return prepareRecord(state, new RunWaitingRecord([...state.waits.waiting.keys()]));
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

/** The frontier records `prepareRecord` validates and plans itself. */
const FRONTIER_RECORDS = [
  NodeStartedRecord,
  RunResumedRecord,
  NodeStepRecord,
  RunStoppedRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  RunWaitingRecord,
  InboxClosedRecord,
  InboxWithdrawnRecord,
  InterruptedRecord,
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

  if (isExactly(record, PoolCommitRecord)) {
    if (!state.inputsCommitted) {
      validateInitialCommit(state, record);
      return prepareCommitTransition(state, record, "initial");
    }
    if (record.steps !== null) {
      return prepareCommitTransition(state, record, "messages", validateMessagesCommit(state, record));
    }
    if (record.take !== null) {
      validateTakeCommit(state, record);
      return prepareCommitTransition(state, record, "node");
    }
    // `<START>.<name>` can't be a node id (ids hold no dot), so the prefix alone picks a send.
    if (record.key.startsWith(sendKey(state, ""))) {
      validateSend(state, record);
      return prepareCommitTransition(state, record, "send");
    }
    // A key the flow doesn't hold yet is a child START the spawner's expansion is about to add.
    const expansion = !state.flow.nodes.has(record.key) && state.flow.nodes.get(record.nodeId)?.isSpawner === true;
    if (expansion) validateExpansionCommit(state, record);
    else validateNodeCommit(state, record);
    return prepareCommitTransition(state, record, expansion ? "expansion" : "node");
  }
  if (!state.inputsCommitted) {
    throw new RangeError("the run start is incomplete: only its pool_commit records may follow RunStarted");
  }
  if (isExactly(record, RunCancelledRecord) || isExactly(record, RunFailedRecord)) {
    return prepareEndingTransition(state, record);
  }
  if (isExactly(record, NodeSucceededRecord)) {
    validateSuccessCommit(state, record);
    return prepareSuccessTransition(state, record);
  }
  if (isExactly(record, NodeRoutedRecord)) {
    return prepareRouteTransition(state, record, false);
  }
  if (isExactly(record, NodeExpandedRecord)) {
    if (typeof options.growBuilder !== "function") {
      throw new TypeError("NodeExpandedRecord replay requires a callable grow_builder");
    }
    return prepareGrowthRecord(state, record, options.growBuilder);
  }
  if (isExactly(record, NodeFailedRecord)) return prepareFailureTransition(state, record);
  if (!FRONTIER_RECORDS.some((cls) => isExactly<RunRecord>(record, cls))) {
    const name = (record as { constructor?: { name?: unknown } } | null)?.constructor?.name;
    throw new TypeError(`unsupported run record for state preparation: ${String(name)}`);
  }

  const { envelopeJson, envelope, canonical } = canonicalEnvelope(state, record, FRONTIER_RECORDS, "frontier");
  let delta: FrontierPlan["delta"];
  if (isExactly(canonical, NodeStartedRecord)) {
    validateNodeStarted(state, canonical);
    delta = { kind: "started", nodeId: canonical.nodeId };
  } else if (isExactly(canonical, RunResumedRecord)) {
    validateRunResumed(state, canonical);
    delta = {
      kind: "resumed",
      retryNodeIds: canonical.retryNodeIds,
      numWorkers: canonical.numWorkers,
      closedCalls: canonical.closedCalls,
    };
  } else if (isExactly(canonical, NodeStepRecord)) {
    validateNodeStep(state, canonical);
    // The decoded copy keeps the applied step apart from the returned record.
    delta = {
      kind: "step",
      nodeId: canonical.nodeId,
      step: canonicalSeed(canonical.step)[1] as JsonObject,
      closedCalls: bornClosedCalls(state, canonical),
    };
  } else if (isExactly(canonical, RunStoppedRecord)) {
    validateRunStopped(state, canonical);
    delta = { kind: "stopped", nodeIds: canonical.nodeIds };
  } else if (isExactly(canonical, NodeWaitingRecord)) {
    validateNodeWaiting(state, canonical);
    delta = { kind: "waiting", nodeId: canonical.nodeId };
  } else if (isExactly(canonical, NodeWokenRecord)) {
    validateNodeWoken(state, canonical);
    delta = { kind: "woken", nodeId: canonical.nodeId, closedCalls: hostCalls(state, canonical.timedOut) };
  } else if (isExactly(canonical, RunWaitingRecord)) {
    validateRunWaiting(state, canonical);
    delta = { kind: "runWaiting", nodeIds: canonical.nodeIds };
  } else if (isExactly(canonical, InboxWithdrawnRecord)) {
    // Before the close: it has every field a close has, so the close's guard would narrow it away.
    validateInboxWithdrawn(state, canonical);
    delta = { kind: "withdrawn", inbox: canonical.inbox, events: canonical.events };
  } else if (isExactly(canonical, InterruptedRecord)) {
    // Before the close: an interrupt has every field a close has, so the close's guard would narrow it away.
    validateInterrupted(state, canonical);
    const open = canonical.nodes.flatMap((nodeId) =>
      nodeCalls(state.waits, nodeId).filter((call) => call.state === "open").map((call) => call.callId),
    );
    delta = { kind: "interrupted", nodeIds: canonical.nodes, closedCalls: hostCalls(state, open) };
  } else if (isExactly(canonical, InboxClosedRecord)) {
    validateInboxClosed(state, canonical);
    delta = { kind: "inboxClosed", inbox: canonical.inbox };
  } else {
    // `canonicalEnvelope` refused every other class.
    throw new TypeError("frontier envelope decoded to a different record kind");
  }

  const plan: FrontierPlan = Object.freeze({
    kind: "frontier",
    ...planAnchors(state, envelopeJson, envelope),
    delta: Object.freeze(delta),
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, frontierObservers(canonical, delta));
}

/**
 * The events one frontier record publishes. `record` is the canonical record `delta` was planned
 * from, of the class its kind names.
 */
function frontierObservers(record: RunRecord, delta: FrontierPlan["delta"]): RunEvent[] {
  const closed = (reason: CloseReason) =>
    "closedCalls" in delta ? delta.closedCalls.map((callId) => new CallClosed({ callId, reason })) : [];
  switch (delta.kind) {
    case "started":
      return [new NodeStarted({ nodeId: delta.nodeId })];
    case "resumed":
      return [new RunResumed(), ...closed("crashed")];
    case "step":
      // A step is node-owned detail; no observer event announces it, except that a host call
      // registered closed is announced closed.
      return closed("interrupted");
    case "waiting": {
      const { nodeId, callIds } = record as NodeWaitingRecord;
      return [new NodeWaiting({ nodeId, callIds })];
    }
    case "woken": {
      const { nodeId, timedOut } = record as NodeWokenRecord;
      return [new NodeWoken({ nodeId, timedOut }), ...closed("timed_out")];
    }
    case "runWaiting":
      return [new RunWaiting({ nodeIds: delta.nodeIds })];
    case "interrupted":
      return closed("interrupted");
    case "stopped":
    case "inboxClosed":
    case "withdrawn":
      return [];
  }
}

/**
 * Encode `record` as `state`'s next envelope under `runId` (the state's own by default) and decode
 * it back, so the prepared record is exactly what either log will retain. The public envelope
 * codec is the sole payload authority.
 *
 * @param classes The record classes the caller plans; `what` names them in the error.
 * @returns The canonical envelope text, the decoded envelope and its record.
 * @throws RangeError when the state has no run id.
 */
function canonicalEnvelope<C extends abstract new (...args: never[]) => RunRecord>(
  state: RunState,
  record: RunRecord,
  classes: readonly C[],
  what: string,
  runId: string | null = state.runId,
): { envelopeJson: string; envelope: RecordEnvelope; canonical: InstanceType<C> } {
  // Established by the initialized-state check, or by `prepareRunStarted`'s data.
  if (typeof runId !== "string") throw new RangeError("run state is not initialized");
  const envelopeJson = encodeRecord(new RecordEnvelope(SCHEMA_VERSION, runId, state.nextSequence, record));
  const envelope = decodeRecord(envelopeJson);
  // An exhaustive codec invariant.
  if (!classes.some((cls) => isExactly(envelope.record, cls))) {
    throw new TypeError(`${what} envelope decoded to a different record kind`);
  }
  return { envelopeJson, envelope, canonical: envelope.record as InstanceType<C> };
}

/** The anchors of a plan prepared against `state` for `envelope`, with its trace change. */
function planAnchors(
  state: RunState,
  envelopeJson: string,
  envelope: RecordEnvelope,
  traceOptions: Parameters<typeof prepareRecordTrace>[2] = {},
): PlanAnchors {
  const { startId, endId } = state.flow;
  return {
    originatingState: state,
    originatingFlow: state.flow,
    envelopeJson,
    startId,
    startNode: state.flow.nodes.get(startId)!,
    endId,
    endNode: state.flow.nodes.get(endId) ?? null,
    traceDelta: prepareRecordTrace(state, envelope, traceOptions),
  };
}

/** Freeze one commit's plan and its canonical public record; `messagesJson` is a messages commit's value. */
function prepareCommitTransition(
  state: RunState,
  record: PoolCommitRecord,
  commitKind: CommitKind,
  messagesJson: string | null = null,
): PreparedTransition {
  const { envelopeJson, envelope, canonical } = canonicalEnvelope(state, record, [PoolCommitRecord], "pool-commit");
  const plan: CommitPlan = Object.freeze({
    kind: "commit",
    ...planAnchors(state, envelopeJson, envelope, { initialCommit: commitKind === "initial" }),
    nodeId: canonical.nodeId,
    key: canonical.key,
    valueJson: canonical.value === null ? null : logJson(canonical.value),
    from: canonical.from,
    take: canonical.take,
    commitKind,
    messagesJson,
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, []);
}

/** Freeze one fully validated success delta and its canonical public record. */
function prepareSuccessTransition(
  state: RunState,
  record: NodeSucceededRecord,
  live: { readonly output: unknown } | null = null,
): PreparedTransition {
  validateLearnedScope(state, record.nodeId);
  if (state.flow.nodes.get(record.nodeId)!.producesMessages && !state.pendingMessages.has(record.nodeId)) {
    throw new RangeError(`node_succeeded of ${reprText(record.nodeId)} has no messages commit to accept`);
  }
  // The caller checked that the record names the node's pending commit.
  let pending = state.pendingCommits.get(record.nodeId)!;
  // A derived record's capture is rebuilt with its value; a child START's, from its constant wiring,
  // only for a run that learns, since its value was admitted with the expansion.
  let capture =
    pending.spawner !== null && state.trace !== null ? rebuildDerivedRecord(state, record.nodeId, new Map()).capture : null;
  if (pending.take !== null) {
    // A take commit holds pointers; its value is the events it consumed, rebuilt from the inbox store.
    pending = { ...pending, valueJson: takeValueJson(state, pending) };
  } else if (pending.valueJson === null) {
    // A derived commit: the pool now holds what its binding read.
    const rebuilt = rebuildCommit(state, record.nodeId, pending.key, pending.from!);
    const valueJson = rebuilt.valueJson;
    capture = rebuilt.capture;
    // Live, a binding that read a name its `from` missed rebuilds to another value.
    if (live !== null) requireEvaluated(state, `derived node ${reprText(record.nodeId)}`, pending, valueJson, live.output);
    pending = { ...pending, valueJson };
  }
  const delta = prepareSuccessDelta(state, record, pending);
  const terminal = prepareTerminalDelta(state, {
    completingNodeId: delta.nodeId,
    newlyReadyIds: delta.newlyReadyIds,
    pendingCommitTarget: delta.commitTarget,
    pendingValueJson: delta.valueJson,
    terminalUnreachableFailure: state.terminalUnreachableFailure,
  });
  const { envelopeJson, envelope, canonical } = canonicalEnvelope(state, record, [NodeSucceededRecord], "node-success");
  const files = fileReadingType(state.flow.nodes.get(delta.commitTarget)!);
  const plan: SuccessPlan = Object.freeze({
    kind: "success",
    ...planAnchors(state, envelopeJson, envelope, {
      terminalSucceeded: terminal?.succeeded ?? null,
      terminalOutputJson: terminal?.outputJson ?? null,
      terminalFailure: terminal?.failure ?? null,
      writerFiles: files,
      committed: { key: delta.commitTarget, valueJson: delta.valueJson },
      messages: delta.messages,
      capture,
    }),
    delta,
    terminal,
    files,
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, materializeSuccess(plan).observers);
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
  validateRouteRecord(state, record);
  validateLearnedScope(state, record.nodeId);
  const route = prepareRouteDelta(state, record, deriveTerminalFailure);
  const causalFailure = route.causalFailure;
  let delta = route.delta;
  // The record keeps its terminal failure through `retainFailure`, which rebuilds each origin as a
  // plain `OriginRef`: the wire's single-key identity shape.
  const recordForWire = new NodeRoutedRecord(record.nodeId, record.handle, {
    terminalFailure: causalFailure,
    forwardCapture: record.forwardCapture,
  });
  const { envelopeJson, envelope, canonical } = canonicalEnvelope(state, recordForWire, [NodeRoutedRecord], "node-route");
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
  const plan: RoutePlan = Object.freeze({
    kind: "route",
    ...planAnchors(state, envelopeJson, envelope, {
      terminalSucceeded: terminal?.succeeded ?? null,
      terminalOutputJson: terminal?.outputJson ?? null,
      terminalFailure: terminal?.failure ?? null,
    }),
    delta,
    terminal,
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, materializeRoute(plan).observers);
}

/** Freeze one validated terminal failure and its canonical public record. */
function prepareFailureTransition(state: RunState, record: NodeFailedRecord): PreparedTransition {
  validateFailureRecord(state, record);
  // The record already holds its failure through `retainFailure`, which rebuilds each origin as a
  // plain `OriginRef`: the wire's single-key identity shape.
  const { envelopeJson, envelope, canonical } = canonicalEnvelope(state, record, [NodeFailedRecord], "node-failure");
  const failure = canonical.failure;
  const closedCalls = openHostCalls(state);
  const plan: FailurePlan = Object.freeze({
    kind: "failure",
    ...planAnchors(state, envelopeJson, envelope, { terminalSucceeded: false, terminalFailure: failure }),
    failure,
    closedCalls,
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, failureObservers(failure, closedCalls));
}

/**
 * Freeze one ending record, a cancel or a failure at quiescence with nodes waiting, and its
 * canonical public record.
 */
function prepareEndingTransition(state: RunState, record: RunCancelledRecord | RunFailedRecord): PreparedTransition {
  if (isExactly(record, RunFailedRecord)) validateRunFailed(state, record);
  const { envelopeJson, envelope, canonical } = canonicalEnvelope(state, record, [RunCancelledRecord, RunFailedRecord], "ending");
  const failure = isExactly(canonical, RunFailedRecord) ? canonical.failure : null;
  const closedCalls = openHostCalls(state);
  const plan: EndingPlan = Object.freeze({
    kind: "ending",
    ...planAnchors(state, envelopeJson, envelope),
    failure,
    closedCalls,
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, materializeEnding(plan).observers);
}

/** The open host calls of the run, in registration order: those a terminal record closes. */
function openHostCalls(state: RunState): string[] {
  return hostCalls(
    state,
    [...state.waits.calls.values()].filter((call) => call.state === "open").map((call) => call.callId),
  );
}

/**
 * Prepare one canonical flat graph expansion without mutating live state, unplanned: the engine
 * plans every growth (`planExpansion`, then `prepareExpandedFromPlan`); this entry point serves
 * direct reducer callers and tests.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight.
 * @param nodeId The exact executing spawner id, including a dynamic LOOP driver clone id.
 * @param grow The seed-first expansion description and its dynamic-overlay prune set.
 * @param growBuilder The injected compile-owned interpreter from a copy of the spawner and the
 *   canonical seed to one self-contained `Flow` fragment.
 * @param options.forwardCapture The canonical binding capture, required exactly when learning. The
 *   record doesn't hold it: an expansion's capture is rebuilt with its seed.
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
  const { seedWire, fragment, validatedFragment } = liveGrowth(state, nodeId, grow, growBuilder);
  // `liveGrowth` checked the spawner, and the record's constructor its carriers.
  validateLearnedScope(state, nodeId);
  const pruneIds = [...grow.prune].sort(compareCodeUnits);
  const digest = expansionDigest(fragment);
  return prepareGrowthTransition(state, new NodeExpandedRecord(nodeId, pruneIds, digest), fragment, {
    actualDigest: digest,
    validatedFragment,
    grown: null,
    seed: seedWire,
    commits: null,
    // Unplanned, the growth isn't checked against its rebuild, so it keeps the evaluated capture.
    capture: options.forwardCapture ?? null,
  });
}

/**
 * Prepare a planned expansion (`planExpansion`) once its expansion commits are written: the growth
 * must equal the one its commits' `from` rebuilds, and carries that rebuild's capture.
 *
 * @param state The state the plan was made against.
 * @param planned The plan.
 * @returns The canonical `NodeExpandedRecord` and its detached append-before-apply plan.
 * @throws RangeError when `state` or its flow isn't the plan's, records other than its expansion
 *   commits followed the plan, the growth differs from its rebuild, or as `prepareExpanded`.
 */
export function prepareExpandedFromPlan(state: RunState, planned: PlannedExpansion): PreparedTransition {
  if (state !== planned.state || state.flow !== planned.flow) {
    throw new RangeError(`the planned expansion of ${reprText(planned.spawnerId)} belongs to another run state`);
  }
  // Only its expansion commits may follow the plan, so the spawner and the topology it checked
  // still hold: they change neither.
  if (state.nextSequence !== planned.sequence + planned.commits.length) {
    throw new RangeError(`the planned expansion of ${reprText(planned.spawnerId)} is stale`);
  }
  validateLearnedScope(state, planned.spawnerId);
  const capture = requireRebuiltGrowth(state, planned);
  const { spawnerId, seed, pruneIds, fragment, validatedFragment, grown, digest, commits } = planned;
  return prepareGrowthTransition(state, new NodeExpandedRecord(spawnerId, pruneIds, digest), fragment, {
    actualDigest: digest,
    validatedFragment,
    grown,
    seed,
    commits,
    capture,
  });
}

/** A live spawner's growth, validated and built once for its expansion commits and its `NodeExpanded`. */
export interface PlannedExpansion {
  /** The state the plan was made against, and its flow; the plan is refused on any other. */
  readonly state: RunState;
  readonly flow: CompiledFlow;
  /** The state's next sequence when planned; only the expansion commits may follow. */
  readonly sequence: number;
  readonly spawnerId: string;
  /** The canonical seed. */
  readonly seed: JsonValue;
  /** The sorted prune ids. */
  readonly pruneIds: readonly string[];
  readonly fragment: Flow;
  readonly validatedFragment: ValidatedFragment;
  /** The topology change the plan checked, which the expansion applies. */
  readonly grown: GrownTopology;
  readonly digest: string;
  /**
   * The commits the expansion writes before its `NodeExpanded`: each START of the fragment (its
   * own, then each child boundary's) whose constant wiring binds, with the value it binds. A START
   * whose wiring doesn't bind gets none; it fails when it runs.
   */
  readonly commits: readonly (readonly [string, TypedWire])[];
}

/**
 * Validate and build a live spawner's growth once, running every check its `NodeExpanded` makes,
 * so a growth rejected after its commits are written can't leave the spawner with commits only its
 * expansion admits. Write `commits` (`prepareExpansionCommit`), then pass the plan to
 * `prepareExpandedFromPlan`.
 *
 * @param state An initialized nonterminal state with `nodeId` in flight.
 * @param options.forwardCapture The capture the expansion will carry, checked for its seed routes.
 * @throws GrowthRejected, TypeError or RangeError as `prepareExpanded`.
 */
export function planExpansion(
  state: RunState,
  nodeId: string,
  grow: Grow,
  growBuilder: GrowBuilder,
  options: OutcomeCarriers = {},
): PlannedExpansion {
  const { seedWire, fragment, validatedFragment } = liveGrowth(state, nodeId, grow, growBuilder);
  const pruneIds = [...grow.prune].sort(compareCodeUnits);
  const digest = expansionDigest(fragment);
  const grown = prepareGrowthTopology(state, new NodeExpandedRecord(nodeId, pruneIds, digest), fragment, validatedFragment);
  requireSeedRoutes(state, options.forwardCapture ?? null, fragment);
  return Object.freeze({
    state,
    flow: state.flow,
    sequence: state.nextSequence,
    grown,
    spawnerId: nodeId,
    seed: seedWire,
    pruneIds: Object.freeze(pruneIds),
    fragment,
    validatedFragment,
    digest,
    commits: Object.freeze(expansionValues(fragment)),
  });
}

/**
 * Require a live growth to equal the one its expansion commits' shared `from` rebuilds (the
 * spawner's binding and pure `forward`, as a replay runs them): its canonical seed and prune ids.
 * A binding that read a name its `from` missed fails here, before anything replays the log.
 * Returns the rebuilt forward capture (`rebuildGrowth`).
 *
 * @throws RangeError naming the spawner when they differ.
 */
function requireRebuiltGrowth(state: RunState, planned: PlannedExpansion): ForwardCapture | null {
  const { spawnerId } = planned;
  const { seed, pruneIds, capture } = rebuildGrowth(state, spawnerId, pendingExpansion(state, spawnerId)[0]?.from ?? null);
  if (logJson(seed) !== logJson(planned.seed) || !sameItems(pruneIds, planned.pruneIds)) {
    throw new RangeError(`spawner ${reprText(spawnerId)} grew another seed than its expansion commits' from rebuilds`);
  }
  return capture;
}

/**
 * Prepare one of a spawner's expansion commits (see `planExpansion`): it holds the spawner's
 * `from` and waits under the child START `key` until the spawner's `NodeExpanded` admits it with
 * the START's value, and the child START's success names it.
 *
 * @param from The spawner's sources, as a derived commit holds them.
 * @throws RangeError when the spawner isn't in flight, `key` is already in the flow or has a
 *   pending commit (see `prepareRecord`).
 */
export function prepareExpansionCommit(
  state: RunState,
  spawnerId: string,
  key: string,
  from: ReadonlyMap<string, number>,
): PreparedTransition {
  // The record leaves the value out; the spawner's NodeExpanded rebuilds it, live as on replay.
  return prepareRecord(state, new PoolCommitRecord(spawnerId, key, null, { from }));
}

/** Validate a live spawner's growth and build its fragment, before any record is written. */
function liveGrowth(
  state: RunState,
  nodeId: string,
  grow: Grow,
  growBuilder: GrowBuilder,
): { seedWire: JsonValue; fragment: Flow; validatedFragment: ValidatedFragment } {
  if (!isExactly(state, RunState)) throw new TypeError("prepareExpanded expects a RunState");
  if (typeof nodeId !== "string") throw new TypeError("node_id must be a string");
  if (!nodeId) throw new RangeError("node_id must be non-empty");
  if (!isExactly(grow, Grow)) throw new TypeError("grow must be a Grow");
  if (typeof growBuilder !== "function") throw new TypeError("grow_builder must be callable");
  // `Grow` always copies `prune` into a new `Set`, so only its items need checking.
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
    // Live seeds only, like steps: replay folds a recorded seed without the process limit. The
    // bound values a seed carries were checked at binding, and a seed wraps them in at most two
    // containers, so those two levels don't count.
    rejectDeepNesting(grow.seed, "growth seed", { enclosingLevels: -2 });
    [seedWire, builderSeed] = canonicalSeed(grow.seed);
  } catch (error) {
    if (error instanceof TypeError || error instanceof RangeError) raiseGrowthRejected(error);
    throw error;
  }
  const fragment = buildGrowthFragment(state.flow.nodes.get(nodeId)!, builderSeed, growBuilder);
  return { seedWire, fragment, validatedFragment: validateGrowthFragment(state, fragment) };
}

/**
 * Require a learnable growth's capture to hold each seed connection's route.
 *
 * @throws GrowthRejected when a connection's field or item wasn't captured.
 */
function requireSeedRoutes(state: RunState, capture: ForwardCapture | null, fragment: Flow): void {
  if (!state.learnable || !isExactly(capture, ForwardCapture)) return;
  for (const connection of fragment.inputConnections) {
    if (selectSeedRoute(capture, connection) === null) {
      raiseGrowthRejected(new RangeError(`seed connection ${writeAsText([...connection.seedPath])} was not captured`));
    }
  }
}

/** Refuse a lookup: a child START binds only its constant wiring. */
function noSources(name: string): ReboundSource {
  throw new RangeError(`a child START binds only its constant wiring, not ${reprText(name)}`);
}

/**
 * The expansion commits of `fragment`: each derived START of it (its own, then each child
 * boundary's) with the canonical value its constant wiring binds to. A START whose wiring doesn't
 * bind, or whose value its contract refuses, gets none: it runs and fails as any node does.
 */
function expansionValues(fragment: Flow): (readonly [string, TypedWire])[] {
  const values: (readonly [string, TypedWire])[] = [];
  for (const startId of new Set([fragment.startId, ...fragment.childBoundaries.map((boundary) => boundary.startId)])) {
    const node = fragment.nodes.get(startId);
    if (node === undefined || node.derived !== true) continue;
    let outcome: Output | Grow;
    try {
      ({ outcome } = rebuildDerived(node, fragment, new Map(), noSources, null));
    } catch (error) {
      if (error instanceof BindingError) continue;
      throw error;
    }
    if (!(outcome instanceof Output)) continue;
    try {
      // The checks a live commit makes (`preparePoolCommit`).
      rejectDeepNesting(outcome.value, "output", { enclosingLevels: node.commitsInputRecord ? -1 : 0 });
      values.push([startId, canonicalOutputValue(outcome.value, node.outputContract, nodeOutputType(node))[0]]);
    } catch {
      // The live commit refuses this value as a node data error, and so does the START when it runs.
    }
  }
  return values;
}

/**
 * Admit the spawner's pending expansion commits for `fragment`: their keys must be exactly its
 * `expansionValues`' STARTs, in order, with one shared `from`. Each gets the value its START binds
 * and then waits on that START's success.
 *
 * @throws RangeError when the commits and the fragment's STARTs disagree.
 */
function admitExpansion(
  state: RunState,
  spawnerId: string,
  fragment: Flow,
  values: PlannedExpansion["commits"] | null,
): readonly PendingCommit[] {
  const expected = values ?? expansionValues(fragment);
  const pending = pendingExpansion(state, spawnerId);
  const keys = pending.map((commit) => commit.key);
  if (!sameItems(keys, expected.map(([key]) => key))) {
    throw new RangeError(
      `NodeExpanded of ${reprText(spawnerId)} needs expansion commits for ${writeAsText(expected.map(([key]) => key))}, ` +
        `found ${writeAsText(keys)}`,
    );
  }
  const from = pending.length === 0 ? "" : fromText(pending[0]!.from);
  return Object.freeze(
    pending.map((commit, index) => {
      if (fromText(commit.from) !== from) {
        throw new RangeError(`the expansion commits of ${reprText(spawnerId)} must hold the same from`);
      }
      return Object.freeze({ ...commit, valueJson: logJson(expected[index]![1]), admitted: true });
    }),
  );
}

/** A `from` as comparable text. */
function fromText(from: ReadonlyMap<string, number> | null): string {
  return from === null ? "null" : logJson(Object.fromEntries([...from].map(([name, seq]) => [name, BigInt(seq)])));
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
    // The seed is rebuilt from the expansion commits' shared `from` (checked when they're
    // admitted) before the prune applies, so the names it read are still in the pool.
    const from = pendingExpansion(state, record.spawnerId)[0]?.from ?? null;
    const { seed: seedWire, pruneIds, capture } = rebuildGrowth(state, record.spawnerId, from);
    const fragment = buildGrowthFragment(state.flow.nodes.get(record.spawnerId)!, canonicalSeed(seedWire)[1], growBuilder);
    if (!sameItems(pruneIds, record.pruneIds)) {
      throw new RangeError(
        `NodeExpandedRecord prune ids ${writeAsText([...record.pruneIds])} differ from the rebuilt ${writeAsText([...pruneIds])}`,
      );
    }
    const validatedFragment = validateGrowthFragment(state, fragment);
    const actualDigest = expansionDigest(fragment);
    if (actualDigest !== record.topologyDigest) {
      throw new RangeError(
        "NodeExpandedRecord topology digest mismatch: " +
          `recorded ${reprText(record.topologyDigest)}, rebuilt ${reprText(actualDigest)}`,
      );
    }
    return prepareGrowthTransition(state, record, fragment, {
      actualDigest,
      validatedFragment,
      grown: null,
      seed: seedWire,
      commits: null,
      capture,
    });
  } catch (error) {
    // A historical record is not a live node outcome: its rejection is a plain corrupt-log error.
    if (error instanceof GrowthRejected) throw error.original;
    throw error;
  }
}

/**
 * Freeze one validated fragment, its final topology and the canonical public record. The caller
 * validated the record and its spawner.
 */
function prepareGrowthTransition(
  state: RunState,
  record: NodeExpandedRecord,
  fragment: Flow,
  options: {
    actualDigest: string;
    validatedFragment: ValidatedFragment;
    /** A live plan's checked topology change; `null` checks and builds it here. */
    grown: GrownTopology | null;
    seed: JsonValue;
    /** A live plan's child START values, so they aren't bound again; `null` binds them here. */
    commits: PlannedExpansion["commits"] | null;
    /** The expansion's forward capture, rebuilt with its seed (`rebuildGrowth`). */
    capture: ForwardCapture | null;
  },
): PreparedTransition {
  const { actualDigest: digest, validatedFragment: validated, seed, commits, capture } = options;
  if (digest !== record.topologyDigest) {
    throw new RangeError("NodeExpandedRecord topology digest disagrees with its fragment");
  }
  const { delta: grown, topology } = options.grown ?? prepareGrowthTopology(state, record, fragment, validated);
  const delta: GrowthDelta = Object.freeze({ ...grown, admitted: admitExpansion(state, record.spawnerId, fragment, commits) });

  const { envelopeJson, envelope, canonical } = canonicalEnvelope(state, record, [NodeExpandedRecord], "node-expansion");
  requireSeedRoutes(state, capture, fragment);
  const plan: GrowthPlan = Object.freeze({
    kind: "growth",
    ...planAnchors(state, envelopeJson, envelope, {
      inputConnections: fragment.inputConnections,
      childBoundaries: fragment.childBoundaries,
      retiredIds: delta.retiredIds,
      seed,
      capture,
    }),
    seed,
    delta,
    topology,
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, [growthObserver(canonical, seed)]);
}

/** Validate all backward data and prepare its detached state before append. */
function prepareBackwardTransition(state: RunState, record: RunRecord): PreparedTransition {
  requireInitializedTerminalBackward(state);
  const { envelopeJson, envelope, canonical } = canonicalEnvelope(state, record, BACKWARD_RECORDS, "backward");
  const nextBackwardState = reduceBackwardRecord(state.trace!, state.backwardState, canonical);
  const plan: BackwardPlan = Object.freeze({
    kind: "backward",
    ...planAnchors(state, envelopeJson, envelope),
    originatingBackwardState: state.backwardState,
    nextBackwardState,
  });
  return new PreparedTransition(canonical, envelope.runId, envelope.seq, plan, []);
}

/** Every plan kind `commitPrepared` dispatches. */
const PLAN_KINDS: ReadonlySet<string> = new Set([
  "runStarted",
  "commit",
  "frontier",
  "success",
  "route",
  "failure",
  "ending",
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
  else if (kind === "ending") requireEndingTraceDelta(state, prepared);
  if (traceDelta !== null) traceDelta.validateOrigin(state.trace);
  let observers: readonly RunEvent[];
  switch (prepared._plan.kind) {
    case "runStarted":
      observers = commitRunStarted(log, state, prepared);
      break;
    case "commit":
      observers = commitPoolCommit(log, state, prepared);
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
    case "ending":
      observers = commitEnding(log, state, prepared);
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
  // Backward records come after the terminal result and change no wait.
  if (kind !== "backward") applyWaitsRecord(state.waits, { seq: prepared.expectedSequence, record: prepared.record });
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
  options: Omit<PrepareTraceOptions, "rootStartId" | "rootEndId" | "rootInputNames"> = {},
): TraceDelta | null {
  const record = envelope.record;
  const nodeId = outcomeNodeId(record);
  const started = isExactly(record, RunStartedRecord);
  const node = nodeId === null ? null : (state.flow.nodes.get(nodeId) ?? null);
  return prepareTrace(state.trace, envelope, node, {
    ...options,
    rebinder: state.trace === null || node === null ? null : nodeRebinder(state, node),
    nodeDataRebuilder: state.trace === null || node === null ? null : nodeDataRebuilder(state, node),
    steps: nodeId === null ? [] : stepsWithSteerValues(state, nodeId),
    rootStartId: started ? state.flow.startId : null,
    rootEndId: started ? state.flow.endId : null,
    rootInputNames: started && record.learnable ? state.flow.nodes.get(state.flow.startId)!.backwardInputNames() : null,
  });
}

/**
 * Return what binds `node` again with the wiring it runs with now (`rebind`), kept for learning,
 * which runs after an expansion may have pruned the node from the flow.
 */
function nodeRebinder(state: RunState, node: Node): Rebinder {
  const flow = state.flow;
  // Copied now: the flow drops a pruned node's wiring.
  const wiring = (state.wiringOverrides.get(node.id) ??
    Object.fromEntries(flow.wiring.get(node.id) ?? new Map())) as JsonObject;
  return (sources, resolve, route) => rebind(node, flow, sources, resolve, wiring, { reserved: recordedReserved(node, route) });
}

/**
 * Return what rebuilds `node`'s node data from its bound inputs (`rebuildNodeData`), with the
 * model configs the run recorded.
 *
 * It holds the node itself, since learning runs after an expansion may have pruned the node from
 * the flow.
 */
function nodeDataRebuilder(
  state: RunState,
  node: Node,
): (inputs: Readonly<Record<string, unknown>> | null) => JsonObject {
  const llm = state.effectiveLlmByStaticPath;
  // Typed now, while the node is in the flow; learning may prune it later.
  const types = localTypes(state.flow, node);
  return (inputs) => rebuildNodeData(node, inputs, llm, types);
}

/**
 * The reserved values a capture route holds whole (a node kind has at most one reserved field,
 * such as AGENT's `__llm_config` or MAP's `over`): each route field that isn't one of `node`'s
 * params, when every such field is a literal with its value (a hook that resolved it without the
 * binder, such as a call-time AGENT's `__llm_config`, which reads names the route doesn't name);
 * `null` when there are none or one was bound through the binder, so `rebind` binds them again.
 */
function recordedReserved(node: Node, route: ValueRoute): Record<string, unknown> | null {
  const params = new Set((node.params ?? []).map((parameter) => parameter.name));
  const fields = route.names.map((name, index) => [name, route.children[index]!] as const).filter(([name]) => !params.has(name));
  if (fields.length === 0 || fields.some(([, child]) => child.kind !== "literal" || child.valueJson === null)) return null;
  return Object.fromEntries(fields.map(([name, child]) => [name, parseJson(child.valueJson!, "codec")]));
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
    delta.resumed !== false ||
    delta.parked !== null ||
    delta.unparked !== null
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
      installed[0] !== plan.delta.commitTarget
    ) {
      throw new RangeError("physical trace writer disagrees with the accepted output target");
    }
    writer = new SourceEntry(source, prepared.expectedSequence, plan.delta.valueJson, plan.files);
    if (plan.delta.commitTarget === state.flow.endId) rootWriter = writer;
  } else if (delta.writer !== null) {
    throw new RangeError("only an accepted output can install a physical trace writer");
  }
  // An accepted messages commit is a second writer, sourced at that commit.
  const messages = plan.kind === "success" ? plan.delta.messages : null;
  const messagesSource = messages === null ? null : new SourceRef(state.runId!, messages.seq);
  const messagesWriter =
    messages === null ? null : new SourceEntry(messagesSource, prepared.expectedSequence, messages.valueJson, newMessagesType());
  if ((delta.messagesWriter?.[0] ?? null) !== (messages?.key ?? null)) {
    throw new RangeError("physical trace messages writer disagrees with the accepted messages commit");
  }
  if (occurrence.messagesSource !== null) validateSource(occurrence.messagesSource);
  if (!sameOptional(occurrence.messagesSource, messagesSource, sameSource)) {
    throw new RangeError("physical trace occurrence disagrees with the accepted messages commit");
  }
  const entries = [
    [delta.writer?.[1] ?? null, writer, "physical trace writer"],
    [delta.messagesWriter?.[1] ?? null, messagesWriter, "physical trace messages writer"],
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
  // A `TraceTerminal` validated itself when built.
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
 * @param options.onPrepared Sees each record after its run-start record, prepared and not yet
 *   applied, with the state it applies to (a reader of derived values, a test's check).
 * @returns `state`, after every historical record has been applied.
 * @throws TypeError when an argument has the wrong type.
 * @throws RangeError when the state isn't pristine, the prefix is empty, not contiguous, not one
 *   run, doesn't start with `RunStarted`, or a record isn't the exact next transition; a node-data
 *   rejection throws its original error.
 */
export function replayRecords(
  state: RunState,
  envelopes: readonly RecordEnvelope[],
  options: { growBuilder: GrowBuilder; onPrepared?: ((prepared: PreparedTransition, state: RunState) => void) | null },
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
    startWiringOverride: start.startWiringOverride,
    effectiveLlmByStaticPath: start.effectiveLlmByStaticPath,
    numWorkers: start.numWorkers,
    memorySnapshots: start.memorySnapshots,
    learnable: start.learnable,
    memories: start.memories,
    memoryStacks: start.memoryStacks,
    defaults: start.defaults,
    callTimeLlm: start.callTimeLlm,
    flow: start.flow,
    skills: start.skills,
    tools: start.tools,
    toolGroups: start.toolGroups,
    yianVersion: start.yianVersion,
    parent: start.parent,
  });
  commitPrepared(cursor, state, prepared);

  for (const encoded of canonical.slice(1)) {
    const envelope = decodeRecord(encoded);
    let next: PreparedTransition;
    try {
      next = prepareRecord(state, envelope.record, { growBuilder });
    } catch (error) {
      // A historical record is not a live node outcome: its rejection is a plain corrupt-log error.
      if (error instanceof NodeDataError) throw error.original;
      throw error;
    }
    options.onPrepared?.(next, state);
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

/**
 * Validate, append and apply one pool commit: a run-start commit writes the pool at once (the
 * START inputs' also make START ready); a send enters the inbox store at once; a node's waits on
 * the node for its success.
 */
function commitPoolCommit(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "commit") throw new TypeError("pool-commit commit requires a commit plan");
  if (!isExactly(prepared.record, PoolCommitRecord)) throw new TypeError("commit plan requires a PoolCommitRecord");
  requirePreparedOrigin(state, plan);
  requireInitializedNonterminal(state);
  requireAuthorityCoordinates(log, state, prepared, true);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  const initial = plan.commitKind === "initial";
  if (initial === state.inputsCommitted) throw new RangeError("pool-commit plan disagrees with the run start");
  if ((state.trace === null) !== (plan.traceDelta === null)) {
    throw new RangeError("pool-commit trace delta disagrees with recorded learning");
  }
  // A run-start commit always holds its value.
  const value = initial ? decodeTyped(plan.valueJson!) : null;
  const startInputs = initial && plan.key === plan.startId;
  const input = startInputs ? (canonicalSeed(value!.toObject())[1] as JsonObject) : null;

  appendAndVerify(log, prepared);
  if (plan.commitKind === "send") {
    state.inboxEvents.set(prepared.expectedSequence, decodeTyped(plan.valueJson!));
  } else if (plan.commitKind === "messages") {
    const pending: PendingMessages = { seq: prepared.expectedSequence, key: plan.key, valueJson: plan.messagesJson! };
    state.pendingMessages.set(plan.nodeId, Object.freeze(pending));
  } else if (!initial) {
    const expansion = plan.commitKind === "expansion";
    const pending: PendingCommit = {
      seq: prepared.expectedSequence,
      key: plan.key,
      valueJson: plan.valueJson,
      from: plan.from,
      take: plan.take,
      spawner: expansion ? plan.nodeId : null,
      admitted: false,
    };
    // An expansion commit waits under the child START it fills, whose success accepts it.
    state.pendingCommits.set(expansion ? plan.key : plan.nodeId, Object.freeze(pending));
  } else if (input === null) {
    state.pool.system.set(plan.key.slice(SYSTEM_PREFIX.length), value!);
    state.producers.set(plan.key, prepared.expectedSequence);
  } else {
    state.pool.store.set(plan.key, value!);
    state.producers.set(plan.key, prepared.expectedSequence);
    state.input = input;
    state.ready = [plan.startId];
    state.inputsCommitted = true;
  }
  state.nextSequence = prepared.expectedSequence + 1;
  return [];
}

/** Validate, append and apply one sparse dispatch, recovery, step, stop, wait, wake, run-wait or close delta. */
function commitFrontier(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "frontier") throw new TypeError("frontier commit requires a frontier plan");
  const record = prepared.record;
  requirePreparedOrigin(state, plan);
  requireInitializedNonterminal(state);
  requireAuthorityCoordinates(log, state, prepared, true);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  const delta = plan.delta;
  if (isExactly(record, NodeStartedRecord) && delta.kind === "started") {
    if (delta.nodeId !== record.nodeId) throw new RangeError("frontier start delta disagrees with its record");
  } else if (isExactly(record, RunResumedRecord) && delta.kind === "resumed") {
    if (
      !sameItems(delta.retryNodeIds, record.retryNodeIds) ||
      !Number.isSafeInteger(delta.numWorkers) ||
      delta.numWorkers !== record.numWorkers
    ) {
      throw new RangeError("frontier resume delta disagrees with its record");
    }
  } else if (isExactly(record, NodeStepRecord) && delta.kind === "step") {
    if (delta.nodeId !== record.nodeId || logJson(delta.step) !== logJson(record.step)) {
      throw new RangeError("frontier step delta disagrees with its record");
    }
  } else if (isExactly(record, RunStoppedRecord) && delta.kind === "stopped") {
    if (!sameItems(delta.nodeIds, record.nodeIds)) {
      throw new RangeError("frontier stop delta disagrees with its record");
    }
  } else if (isExactly(record, NodeWaitingRecord) && delta.kind === "waiting") {
    if (delta.nodeId !== record.nodeId) throw new RangeError("frontier wait delta disagrees with its record");
  } else if (isExactly(record, NodeWokenRecord) && delta.kind === "woken") {
    if (delta.nodeId !== record.nodeId) throw new RangeError("frontier wake delta disagrees with its record");
  } else if (isExactly(record, RunWaitingRecord) && delta.kind === "runWaiting") {
    if (!sameItems(delta.nodeIds, record.nodeIds)) throw new RangeError("frontier run-wait delta disagrees with its record");
  } else if (isExactly(record, InboxClosedRecord) && delta.kind === "inboxClosed") {
    if (delta.inbox !== record.inbox) throw new RangeError("frontier close delta disagrees with its record");
  } else if (isExactly(record, InboxWithdrawnRecord) && delta.kind === "withdrawn") {
    if (delta.inbox !== record.inbox || !sameItems(delta.events, record.events)) {
      throw new RangeError("frontier withdraw delta disagrees with its record");
    }
  } else if (isExactly(record, InterruptedRecord) && delta.kind === "interrupted") {
    if (!sameItems(delta.nodeIds, record.nodes)) throw new RangeError("frontier interrupt delta disagrees with its record");
  } else {
    throw new TypeError("frontier record and prevalidated delta disagree");
  }
  const observers = frontierObservers(record, delta);

  appendAndVerify(log, prepared);
  switch (delta.kind) {
    case "started":
      state.ready.shift();
      state.inFlight.add(delta.nodeId);
      state.acceptedStarts.set(delta.nodeId, prepared.expectedSequence);
      state.nodeStarts.set(delta.nodeId, (state.nodeStarts.get(delta.nodeId) ?? 0) + 1);
      break;
    case "resumed":
      // The retries are exactly the in-flight nodes, which `acceptedStarts` holds in start order.
      addCrashedAttempts(state.crashedAttempts, state.acceptedStarts, state.lastStopSequence);
      state.lastStopSequence = null;
      // Steps and start counts outlive the segment: a retried node reads them back. One `unshift`
      // per id, last first, puts the retries ahead in order without spreading a wide list.
      for (let index = delta.retryNodeIds.length - 1; index >= 0; index--) {
        state.ready.unshift(delta.retryNodeIds[index]!);
      }
      state.inFlight.clear();
      state.acceptedStarts.clear();
      // A retried node runs again and writes a new commit; its orphan stays in the log, unread. An
      // admitted expansion commit stays: its spawner settled, and the child START still names it.
      for (const [nodeId, pending] of state.pendingCommits) {
        if (!pending.admitted) state.pendingCommits.delete(nodeId);
      }
      // A retried node writes its messages commit again; the orphan stays in the log, unread.
      state.pendingMessages.clear();
      state.numWorkers = delta.numWorkers;
      break;
    case "step":
      state.nodeSteps.set(
        delta.nodeId,
        Object.freeze([...(state.nodeSteps.get(delta.nodeId) ?? []), delta.step]),
      );
      state.nodeStepSeqs.set(
        delta.nodeId,
        Object.freeze([...(state.nodeStepSeqs.get(delta.nodeId) ?? []), prepared.expectedSequence]),
      );
      break;
    case "stopped":
      state.lastStopSequence = prepared.expectedSequence;
      break;
    case "waiting":
      // The waits fold moves it to the waiting set; its start count, steps and calls stay.
      state.inFlight.delete(delta.nodeId);
      state.acceptedStarts.delete(delta.nodeId);
      break;
    case "woken":
      // The wake is the invocation's start from here on, so a later crash counts; the start count
      // doesn't move, since no new start was recorded.
      state.inFlight.add(delta.nodeId);
      state.acceptedStarts.set(delta.nodeId, prepared.expectedSequence);
      break;
    case "withdrawn":
      // Withdrawn sends get no reply; nothing reads their values again.
      for (const seq of delta.events) state.inboxEvents.delete(seq);
      break;
    case "runWaiting":
    case "inboxClosed":
    case "interrupted":
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
  if (
    record.nodeId !== delta.nodeId ||
    state.pendingCommits.get(record.nodeId)?.seq !== record.commit ||
    (state.pendingMessages.get(record.nodeId)?.seq ?? null) !== record.messages ||
    (delta.messages?.seq ?? null) !== record.messages
  ) {
    throw new RangeError("success record and prevalidated delta disagree");
  }
  const apply = materializeSuccess(plan);

  appendAndVerify(log, prepared);
  state.pool.store.set(delta.commitTarget, apply.value);
  state.producers.set(delta.commitTarget, record.commit);
  state.pendingCommits.delete(delta.nodeId);
  if (delta.messages !== null) {
    state.pool.store.set(delta.messages.key, apply.messagesValue!);
    state.producers.set(delta.messages.key, delta.messages.seq);
    state.pendingMessages.delete(delta.nodeId);
  }
  state.nodeState.set(delta.nodeId, NodeState.TAKEN);
  for (const edgeId of delta.outgoingEdgeIds) state.edgeState.set(edgeId, NodeState.TAKEN);
  for (const nodeId of delta.edgeHeads) dropUnknownIncoming(state, nodeId);
  state.inFlight.delete(delta.nodeId);
  state.acceptedStarts.delete(delta.nodeId);
  forgetNodeSteps(state, delta.nodeId);
  for (const seq of delta.consumedEvents) state.inboxEvents.delete(seq);
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
  for (const nodeId of delta.edgeHeads) dropUnknownIncoming(state, nodeId);
  for (const nodeId of delta.skippedNodeIds) {
    state.nodeState.set(nodeId, NodeState.SKIPPED);
    // A skipped child START never runs, so no success names its expansion commit.
    state.pendingCommits.delete(nodeId);
  }
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
  const apply = materializeFailure(plan.failure, plan.closedCalls);

  appendAndVerify(log, prepared);
  state.nodeState.set(nodeId, NodeState.TAKEN);
  state.inFlight.delete(nodeId);
  state.acceptedStarts.delete(nodeId);
  state.pendingCommits.delete(nodeId);
  forgetNodeSteps(state, nodeId);
  state.result = apply.result;
  state.nextSequence = prepared.expectedSequence + 1;
  return apply.observers;
}

/** Validate, append and install one ending record: the run's terminal result. */
function commitEnding(log: RunLog, state: RunState, prepared: PreparedTransition): readonly RunEvent[] {
  const plan = prepared._plan;
  if (plan.kind !== "ending") throw new TypeError("ending commit requires an ending plan");
  const record = prepared.record;
  if (!isExactly(record, RunCancelledRecord) && !isExactly(record, RunFailedRecord)) {
    throw new TypeError("ending plan requires a RunCancelledRecord or RunFailedRecord");
  }
  requirePreparedOrigin(state, plan);
  requireInitializedNonterminal(state);
  requireAuthorityCoordinates(log, state, prepared, true);
  requireUnchangedRecord(prepared, plan.envelopeJson);
  if ((isExactly(record, RunFailedRecord) ? record.failure : null) !== plan.failure) {
    throw new RangeError("ending record and prevalidated failure disagree");
  }
  const apply = materializeEnding(plan);

  appendAndVerify(log, prepared);
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

/** Every `TraceDelta` field that changes the trace: all but its origin and sequence. */
const TRACE_CHANGES = [
  "initialized",
  "started",
  "resumed",
  "parked",
  "unparked",
  "occurrence",
  "handler",
  "material",
  "writer",
  "messagesWriter",
  "systemValue",
  "pruneIds",
  "connections",
  "consumedConnection",
  "boundaries",
  "boundaryUpdate",
  "rootStartSource",
  "rootWriter",
  "causalRouteSource",
  "terminal",
] as const satisfies readonly (keyof TraceDeltaFields)[];

type TraceChange = (typeof TRACE_CHANGES)[number];

// Fails to compile when `TraceDeltaFields` gains a field `TRACE_CHANGES` doesn't list.
const TRACE_CHANGES_COVER_FIELDS: Exclude<keyof TraceDeltaFields, TraceChange> extends never ? true : never = true;
void TRACE_CHANGES_COVER_FIELDS;

/** The `TraceDelta` change fields that hold lists; empty means unchanged. */
const TRACE_LIST_CHANGES: ReadonlySet<TraceChange> = new Set(["pruneIds", "connections", "boundaries"]);

/**
 * Whether `delta` leaves every change field but `allowed` unchanged: `resumed` false, a list
 * empty, any other field `null`.
 */
function changesOnly(delta: TraceDelta, allowed: readonly TraceChange[] = []): boolean {
  return TRACE_CHANGES.every((field) => {
    if (allowed.includes(field)) return true;
    const value: unknown = delta[field];
    if (field === "resumed") return value === false;
    if (TRACE_LIST_CHANGES.has(field)) return Array.isArray(value) && value.length === 0;
    return value === null;
  });
}

/** Require the exact sequence-only trace change before accepting a post-terminal record. */
function requirePostterminalTraceDelta(state: RunState, delta: TraceDelta | null): void {
  if (
    !isExactly(delta, TraceDelta) ||
    delta.originatingTrace !== state.trace ||
    !Number.isSafeInteger(delta.sequence) ||
    delta.sequence !== state.nextSequence ||
    !changesOnly(delta)
  ) {
    throw new RangeError("postterminal trace delta must only advance the accepted sequence");
  }
}

/** Permit only the canonical dispatch, resume, step, stop, wait, wake, run-wait or close trace change before append. */
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
  } else if (
    isExactly(record, RunStoppedRecord) ||
    isExactly(record, RunWaitingRecord) ||
    isExactly(record, InboxClosedRecord) ||
    isExactly(record, InboxWithdrawnRecord) ||
    isExactly(record, InterruptedRecord)
  ) {
    if (delta.started !== null || delta.resumed !== false) {
      throw new RangeError("frontier trace stop must only advance the accepted sequence");
    }
  } else if (isExactly(record, NodeWaitingRecord)) {
    if (delta.started !== null || delta.resumed !== false || delta.parked !== record.nodeId) {
      throw new RangeError("frontier trace wait must park its node's start");
    }
  } else if (isExactly(record, NodeWokenRecord)) {
    if (delta.started !== null || delta.resumed !== false || delta.unparked !== record.nodeId) {
      throw new RangeError("frontier trace wake must bring back its node's parked start");
    }
  } else {
    throw new RangeError("frontier trace requires a dispatch, resume, step, stop, wait, wake, run-wait, close, withdraw or interrupt record");
  }
  if (!isExactly(record, NodeWaitingRecord) && delta.parked !== null) {
    throw new RangeError("only a wait can park a start");
  }
  if (!isExactly(record, NodeWokenRecord) && delta.unparked !== null) {
    throw new RangeError("only a wake can bring back a parked start");
  }
  if (!changesOnly(delta, ["started", "resumed", "parked", "unparked"])) {
    throw new RangeError("frontier trace cannot change outcome or terminal history");
  }
}

/**
 * Permit only an ending's trace change before append: the sequence, plus for a `run_failed` the
 * failed terminal from the causal route occurrence.
 */
function requireEndingTraceDelta(state: RunState, prepared: PreparedTransition): void {
  const delta = prepared._plan.traceDelta;
  const trace = state.trace;
  if (trace === null) {
    if (delta !== null) throw new RangeError("an unretained ending cannot carry a trace delta");
    return;
  }
  if (!isExactly(delta, TraceDelta) || delta.originatingTrace !== trace || delta.sequence !== state.nextSequence) {
    throw new RangeError("ending trace delta requires its exact accepted sequence");
  }
  if (!changesOnly(delta, ["terminal"])) throw new RangeError("ending trace can only install its terminal");
  const record = prepared.record;
  if (!isExactly(record, RunFailedRecord)) {
    if (delta.terminal !== null) throw new RangeError("a cancel installs no terminal trace");
    return;
  }
  const terminal = delta.terminal;
  if (
    !isExactly(terminal, TraceTerminal) ||
    terminal.succeeded ||
    trace.causalRouteSource === null ||
    !sameOptional(terminal.source, trace.causalRouteSource, sameSource) ||
    !sameOptional(terminal.failure, record.failure, sameFailure)
  ) {
    throw new RangeError("a run failure's terminal trace disagrees with its causal route occurrence");
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
  // The plan reduced this record from this very backward state, as the identity check above shows.
  if (!isExactly(plan.nextBackwardState, RuntimeBackwardState)) {
    throw new TypeError("backward plan requires an exact next state");
  }
  appendAndVerify(log, prepared);
  state.backwardState = plan.nextBackwardState;
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
 * @internal Add one to `counts` for each in-flight attempt no stop covers: the count a resume
 * records for the nodes it retries, and what a run log reader adds for a run that a resume would
 * pick up. A stop names every in-flight node, and an attempt stays in flight until its outcome or
 * the next resume, so an attempt is covered exactly when it started before the segment's last stop.
 *
 * @param counts The crashed counts by node id, updated in place.
 * @param starts The accepted start sequence of each in-flight node.
 * @param lastStop The sequence of the segment's last `RunStopped`, or `null`.
 */
export function addCrashedAttempts(
  counts: Map<string, number>,
  starts: ReadonlyMap<string, number>,
  lastStop: number | null,
): void {
  for (const [nodeId, start] of starts) {
    if (lastStop === null || start >= lastStop) counts.set(nodeId, (counts.get(nodeId) ?? 0) + 1);
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
    new RecordEnvelope(frozen.schema, prepared.expectedRunId, prepared.expectedSequence, prepared.record, frozen.at),
  );
  if (current !== envelopeJson) throw new RangeError("prepared record changed after validation");
}

/**
 * Append the canonical record and reject any protocol-breaking result. The log stamps the record
 * with its time, which the plan's text doesn't hold, so the compare leaves it out.
 */
function appendAndVerify(log: RunLog, prepared: PreparedTransition): void {
  const envelope: unknown = log.append(prepared.record);
  // The record compare is the envelope text compare: the coordinates already match, and the
  // prepared record encodes to the plan's text (`requireUnchangedRecord`).
  if (
    !isExactly(envelope, RecordEnvelope) ||
    envelope.runId !== prepared.expectedRunId ||
    envelope.seq !== prepared.expectedSequence ||
    envelope.schema !== SCHEMA_VERSION ||
    envelope.at === null ||
    log.nextSequence !== prepared.expectedSequence + 1 ||
    encodeRecord(new RecordEnvelope(envelope.schema, envelope.runId, envelope.seq, envelope.record)) !==
      prepared._plan.envelopeJson
  ) {
    throw new RangeError("run-log append result does not match prepared transition");
  }
}

/** Check the initialized reducer roots. */
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
    isExactly(state.nodeStepSeqs, Map) &&
    isExactly(state.pendingCommits, Map) &&
    isExactly(state.pendingMessages, Map) &&
    isExactly(state.producers, Map) &&
    isExactly(state.crashedAttempts, Map) &&
    isExactly(state.depth, Map) &&
    isExactly(state.growthOwner, Map) &&
    isExactly(state.deferredPrune, Set) &&
    isExactly(state.wiringOverrides, Map) &&
    isExactly(state.waits, WaitsState) &&
    isExactly(state.inboxEvents, Map) &&
    isRecord(state.input) &&
    isRecord(state.effectiveLlmByStaticPath) &&
    isExactly(state.backwardState, RuntimeBackwardState) &&
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
  // append, rather than between append and apply. The checks are per node, not per step: a step is
  // checked once, when its record is validated, so a node's S steps don't cost O(S) per record.
  if (
    [...state.inFlight].some((nodeId) => !state.nodeStarts.has(nodeId)) ||
    [...state.nodeStarts.values()].some((count) => !Number.isSafeInteger(count) || count < 1) ||
    [...state.nodeSteps.keys()].some((nodeId) => !state.nodeStarts.has(nodeId)) ||
    [...state.nodeSteps.values()].some((steps) => !Array.isArray(steps) || steps.length === 0) ||
    [...state.pendingCommits].some(([nodeId, pending]) =>
      pending.admitted ? state.nodeState.get(nodeId) !== NodeState.UNKNOWN : !state.inFlight.has(pending.spawner ?? nodeId),
    ) ||
    state.nodeStepSeqs.size !== state.nodeSteps.size ||
    [...state.nodeStepSeqs].some(([nodeId, seqs]) => state.nodeSteps.get(nodeId)?.length !== seqs.length) ||
    // A messages commit follows its node's own output commit, which its success accepts with it.
    [...state.pendingMessages.keys()].some((nodeId) => {
      const output = state.pendingCommits.get(nodeId);
      return !state.inFlight.has(nodeId) || output === undefined || output.spawner !== null;
    })
  ) {
    throw new RangeError("node starts, steps and pending commits must match the unsettled invocations");
  }
  if ([...state.waits.waiting.keys()].some((nodeId) => state.inFlight.has(nodeId) || !state.nodeStarts.has(nodeId))) {
    throw new RangeError("a waiting node must have a start count and not be in flight");
  }
  if (
    [...state.crashedAttempts].some(
      ([nodeId, count]) => typeof nodeId !== "string" || !Number.isSafeInteger(count) || count < 1,
    )
  ) {
    throw new RangeError("crashed attempt counts must be positive integers by node");
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
    if (!sameIds(state.trace._parked, state.waits.waiting.keys())) {
      throw new RangeError("run-state trace parks other starts than the waiting nodes");
    }
  } else if (state.trace !== null) {
    throw new RangeError("trace requires recorded learning");
  }
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
  requireWorkerCount(record.numWorkers);
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
  const closedCalls = retriedAgentCalls(state);
  if (!sameItems(record.closedCalls, closedCalls)) {
    throw new RangeError(
      "RunResumedRecord closed_calls must exactly equal the retried nodes' open calls: " +
        `expected ${writeAsText(closedCalls)}, found ${writeAsText([...record.closedCalls])}`,
    );
  }
}

/**
 * The open calls a resume closes, in registration order: those of retried (in-flight) nodes that
 * make no direct host tool call, the agents'. A TOOL node's re-run reuses its call instead. An
 * agent's call registered before its last wake and still open (a timeout wake left it open) is
 * still pending at the host, not an unknown outcome: the replay waits on it again.
 */
function retriedAgentCalls(state: RunState): string[] {
  const w = state.waits;
  return [...w.calls.values()]
    .filter((call) => {
      if (call.state !== "open" || !state.inFlight.has(call.nodeId)) return false;
      if (state.flow.nodes.get(call.nodeId)?.hostToolCall() !== null) return false;
      const lastWoken = w._lastWoken.get(call.nodeId);
      return lastWoken === undefined || call.seq > lastWoken;
    })
    .map((call) => call.callId);
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

/** The host calls among `callIds`: the registered calls whose tool is neither take nor sleep, in order. */
function hostCalls(state: RunState, callIds: Iterable<string>): string[] {
  return [...callIds].filter((callId) => {
    const tool = state.waits.calls.get(callId)?.tool;
    return tool !== undefined && tool !== TAKE && tool !== SLEEP;
  });
}

/**
 * Require a wait record to move an in-flight node with no pending commit to the waiting set: its
 * `waits` entry (a take or sleep TOOL node's own call, not yet registered; a take's inbox declared,
 * open and with nothing unread a take may read), and `call_ids` naming exactly calls of the node
 * registered since its last wake or still open at it, not closed, including every open one. A
 * `steer_inbox` names a declared top-level inbox, on a node that reads a steer inbox.
 */
function validateNodeWaiting(state: RunState, record: NodeWaitingRecord): void {
  const { nodeId } = record;
  const node = state.flow.nodes.get(nodeId);
  if (node === undefined) throw new RangeError(`NodeWaiting node ${reprText(nodeId)} does not exist in the flow`);
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`NodeWaiting node ${reprText(nodeId)} must have UNKNOWN disposition`);
  }
  if (!state.inFlight.has(nodeId)) throw new RangeError(`NodeWaiting node ${reprText(nodeId)} is not in-flight`);
  requireNoPendingCommit(state, nodeId, "NodeWaiting");
  if (record.steerInbox !== null) {
    if (!node.readsSteerInbox) {
      throw new RangeError(`NodeWaiting steer_inbox on node ${reprText(nodeId)}, which reads no steer inbox`);
    }
    if (inboxElementType(state, record.steerInbox) === null) {
      throw new RangeError(`NodeWaiting steer_inbox ${reprText(record.steerInbox)} is no inbox the START declares`);
    }
  }
  const w = state.waits;
  const toolId = node.hostToolCall()?.toolId ?? null;
  for (const entry of record.waits) {
    if (toolId !== TAKE && toolId !== SLEEP) {
      throw new RangeError(`NodeWaiting node ${reprText(nodeId)} registers a wait but is neither a take nor a sleep`);
    }
    if (entry.tool !== toolId) {
      throw new RangeError(`NodeWaiting wait entry tool ${reprText(entry.tool)} isn't the node's tool ${reprText(toolId)}`);
    }
    if (entry.callId !== toolCallId(state.runId!, nodeId)) {
      throw new RangeError(`NodeWaiting wait entry call_id ${reprText(entry.callId)} isn't the node's call id`);
    }
    if (w.calls.has(entry.callId)) {
      throw new RangeError(`call ${reprText(entry.callId)} is already registered; it belongs to its node until the node settles`);
    }
    if (entry.inbox !== null) {
      if (inboxElementType(state, entry.inbox) === null) {
        throw new RangeError(`NodeWaiting wait entry inbox ${reprText(entry.inbox)} is no inbox the START declares`);
      }
      if (w.inboxes.get(entry.inbox)?.closed === true) {
        throw new RangeError(`a take can't wait on closed inbox ${reprText(entry.inbox)}`);
      }
      if (availableUnread(w, entry.inbox).length > 0) {
        throw new RangeError(`a take can't wait while inbox ${reprText(entry.inbox)} holds unread events it may read`);
      }
    }
  }
  const registering = new Set(record.waits.map((entry) => entry.callId));
  const lastWoken = w._lastWoken.get(nodeId);
  for (const callId of record.callIds) {
    if (registering.has(callId)) continue;
    const call = w.calls.get(callId);
    if (call === undefined) throw new RangeError(`NodeWaiting call ${reprText(callId)} is not registered`);
    if (call.nodeId !== nodeId) {
      throw new RangeError(`NodeWaiting call ${reprText(callId)} belongs to node ${reprText(call.nodeId)}`);
    }
    // A call registered before the last wake may be waited on again only if it was still open at
    // that wake (a timeout wake leaves the node's other calls open): open now, or answered by a
    // result after the wake. A closed one is refused below.
    const settledByWake =
      lastWoken !== undefined &&
      call.seq < lastWoken &&
      call.state !== "open" &&
      call.state !== "closed" &&
      !(call.state === "answered" && call.resultSeq !== null && call.resultSeq > lastWoken);
    if (settledByWake) {
      throw new RangeError(
        `NodeWaiting call ${reprText(callId)} was registered before the node's last node_woken and had settled by then`,
      );
    }
    // An interrupt can land between an agent's `Pending` and the engine's `node_waiting`; the node
    // is wakeable at once and honors it.
    const interrupted = call.closedReason === "interrupted" && w.interrupts.has(nodeId);
    if (call.state === "closed" && !interrupted) {
      throw new RangeError(`NodeWaiting call ${reprText(callId)} was closed (${call.closedReason})`);
    }
  }
  for (const call of nodeCalls(w, nodeId)) {
    if (call.state === "open" && !record.callIds.includes(call.callId)) {
      throw new RangeError(`NodeWaiting must name the node's open call ${reprText(call.callId)}`);
    }
  }
}

/**
 * Require a wake record to name a waiting node whose `timed_out` calls are open calls of the node
 * with a deadline. A wake with no `timed_out` (by answers) needs every call it waits on settled,
 * unless an interrupt names the node; a timeout wake may leave the node's other calls open (the
 * reducer reads no clock: the engine decides which calls are due).
 */
function validateNodeWoken(state: RunState, record: NodeWokenRecord): void {
  const { nodeId } = record;
  const w = state.waits;
  const waiting = w.waiting.get(nodeId);
  if (waiting === undefined) throw new RangeError(`NodeWoken node ${reprText(nodeId)} is not waiting`);
  for (const callId of record.timedOut) {
    const call = w.calls.get(callId);
    if (call === undefined || call.nodeId !== nodeId || call.state !== "open" || call.deadline === null) {
      throw new RangeError(`NodeWoken timed_out ${reprText(callId)} isn't an open call of the node with a deadline`);
    }
  }
  // A wake by answers needs every call settled; a timeout wake may leave the node's other calls open.
  const open = waiting.callIds.find((callId) => w.calls.get(callId)?.state === "open");
  if (record.timedOut.length === 0 && open !== undefined && !w.interrupts.has(nodeId)) {
    throw new RangeError(`NodeWoken node ${reprText(nodeId)} still waits on open call ${reprText(open)}`);
  }
}

/** Require a close record to name an open inbox the START declares. */
function validateInboxClosed(state: RunState, record: InboxClosedRecord): void {
  if (inboxElementType(state, record.inbox) === null) {
    throw new RangeError(`InboxClosed inbox ${reprText(record.inbox)} is no inbox the START declares`);
  }
  if (state.waits.inboxes.get(record.inbox)?.closed === true) {
    throw new RangeError(`InboxClosed inbox ${reprText(record.inbox)} is already closed`);
  }
}

/**
 * Require a withdraw record to name a declared inbox and its whole unread set (as
 * `withdrawableEvents` reads it), in seq order, as a steer step names its whole queue.
 */
function validateInboxWithdrawn(state: RunState, record: InboxWithdrawnRecord): void {
  if (inboxElementType(state, record.inbox) === null) {
    throw new RangeError(`InboxWithdrawn inbox ${reprText(record.inbox)} is no inbox the START declares`);
  }
  const unread = withdrawableEvents(state.waits, record.inbox);
  if (!sameItems(record.events, unread)) {
    throw new RangeError(
      `InboxWithdrawn must name the whole unread set [${unread.join(", ")}] of inbox ${reprText(record.inbox)}, found [${record.events.join(", ")}]`,
    );
  }
}

/**
 * Require an interrupt record to name, once each, nodes reading its declared inbox as their steer
 * inbox: in flight with `readsSteerInbox` (which inbox a running node opened is the engine's word,
 * bound at run time), or waiting with `node_waiting.steer_inbox` equal to it; none holding a
 * pending interrupt already; and every waiting reader of the inbox without one among them. It may
 * name none: the Stop is recorded for a reader that opens the inbox later. When a host may record
 * one (the idle and already-stopping rules) is `prepareInterrupt`'s.
 */
function validateInterrupted(state: RunState, record: InterruptedRecord): void {
  const { inbox, nodes } = record;
  if (inboxElementType(state, inbox) === null) {
    throw new RangeError(`Interrupted inbox ${reprText(inbox)} is no inbox the START declares`);
  }
  if (new Set(nodes).size !== nodes.length) throw new RangeError("Interrupted names each node once");
  const w = state.waits;
  for (const nodeId of nodes) {
    const node = state.flow.nodes.get(nodeId);
    if (node === undefined) throw new RangeError(`Interrupted node ${reprText(nodeId)} does not exist in the flow`);
    const waiting = w.waiting.get(nodeId);
    if (waiting !== undefined) {
      if (waiting.steerInbox !== inbox) {
        throw new RangeError(`Interrupted node ${reprText(nodeId)} waits without reading steer inbox ${reprText(inbox)}`);
      }
    } else if (!state.inFlight.has(nodeId)) {
      throw new RangeError(`Interrupted node ${reprText(nodeId)} is neither in flight nor waiting`);
    } else if (!node.readsSteerInbox) {
      throw new RangeError(`Interrupted node ${reprText(nodeId)} reads no steer inbox`);
    }
    if (w.interrupts.has(nodeId)) throw new RangeError(`Interrupted node ${reprText(nodeId)} already holds a pending interrupt`);
  }
  for (const [nodeId, waiting] of w.waiting) {
    if (waiting.steerInbox === inbox && !w.interrupts.has(nodeId) && !nodes.includes(nodeId)) {
      throw new RangeError(`Interrupted must name waiting node ${reprText(nodeId)}, which reads steer inbox ${reprText(inbox)}`);
    }
  }
}

/**
 * Require a run failure at quiescence with nodes waiting and none wakeable, holding the failure of
 * the route that proved END unreachable.
 */
function validateRunFailed(state: RunState, record: RunFailedRecord): void {
  requireWaitingQuiescence(state, "RunFailed");
  const latch = state.terminalUnreachableFailure;
  if (latch === null) throw new RangeError("RunFailed requires a route that proved END unreachable");
  if (!sameFailure(record.failure, latch)) {
    throw new RangeError("RunFailedRecord failure must be the failure of the route that proved END unreachable");
  }
}

/** Require quiescence with nodes waiting: none in flight or ready, at least one waiting, none wakeable. */
function requireWaitingQuiescence(state: RunState, kind: string): void {
  if (state.inFlight.size > 0 || state.ready.length > 0) throw new RangeError(`${kind} requires no node in flight or ready`);
  if (state.waits.waiting.size === 0) throw new RangeError(`${kind} requires a waiting node`);
  const wakeable = wakeableNodes(state.waits);
  if (wakeable.length > 0) throw new RangeError(`${kind} while node ${reprText(wakeable[0]!)} is wakeable`);
}

/**
 * Require a run-wait record at quiescence: nothing in flight or ready, waiting nodes named in
 * `node_waiting` order, none wakeable, and no `run_waiting` as the last frontier record (nothing
 * could have changed since).
 */
function validateRunWaiting(state: RunState, record: RunWaitingRecord): void {
  const w = state.waits;
  requireWaitingQuiescence(state, "RunWaiting");
  const expected = [...w.waiting.keys()];
  if (!sameItems(record.nodeIds, expected)) {
    throw new RangeError(
      "RunWaitingRecord node_ids must exactly equal the waiting nodes in order: " +
        `expected ${writeAsText(expected)}, found ${writeAsText([...record.nodeIds])}`,
    );
  }
  if (w.lastFrontier === "run_waiting") throw new RangeError("RunWaiting follows a RunWaiting with nothing changed");
}

/**
 * Require a step record to name an existing unknown node, in flight or waiting, and be its next
 * step. A `pending` step registers a host call; a `tool_result` for a registered call answers it.
 * A waiting node records only a `tool_result` answering one of its open calls.
 */
function validateNodeStep(state: RunState, record: NodeStepRecord): void {
  const nodeId: unknown = record.nodeId;
  if (typeof nodeId !== "string") throw new TypeError("NodeStepRecord node_id must be a string");
  if (!state.flow.nodes.has(nodeId)) {
    throw new RangeError(`NodeStep node ${reprText(nodeId)} does not exist in the flow`);
  }
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`NodeStep node ${reprText(nodeId)} must have UNKNOWN disposition`);
  }
  const waiting = state.waits.waiting.has(nodeId);
  if (!state.inFlight.has(nodeId) && !waiting) throw new RangeError(`NodeStep node ${reprText(nodeId)} is not in-flight`);
  const expected = (state.nodeSteps.get(nodeId)?.length ?? 0) + 1;
  if (record.n !== expected) {
    throw new RangeError(`node step n is not dense: node ${reprText(nodeId)} step ${record.n}, expected ${expected}`);
  }
  const step = record.step;
  if (step["kind"] === "tool_result") {
    validateToolResultStep(state, nodeId, step, waiting);
  } else if (waiting) {
    throw new RangeError(`waiting node ${reprText(nodeId)} records only a tool_result answering one of its open calls`);
  } else if (step["kind"] === "pending") {
    validatePendingStep(state, nodeId, step);
  } else if (step["kind"] === "steer") {
    validateSteerStep(state, nodeId, step);
  } else if (step["kind"] === "interrupted") {
    validateInterruptedStep(state, nodeId, step);
  }
}

/**
 * Require a running node's `interrupted` step, its honor of a host interrupt, to hold only `kind`
 * and `by`, the seq of the `interrupted` record the node holds pending.
 */
function validateInterruptedStep(state: RunState, nodeId: string, step: Readonly<JsonObject>): void {
  const keys = Object.keys(step);
  if (keys.length !== 2 || !Object.hasOwn(step, "by")) throw new RangeError("an interrupted step holds only kind and by");
  const pending = state.waits.interrupts.get(nodeId);
  if (pending === undefined) throw new RangeError(`an interrupted step on node ${reprText(nodeId)}, which holds no pending interrupt`);
  // An int, so the log writes it as an int.
  const by = typeof step["by"] === "bigint" ? stepInt(step["by"]) : null;
  if (by !== pending) {
    throw new RangeError(`node ${reprText(nodeId)}'s interrupted step must name its pending interrupt ${pending}, got ${writeAsText(step["by"] ?? null)}`);
  }
}

/**
 * The host call a `pending` step registers already closed: the node holds a pending interrupt,
 * whose record closed its calls before a tool already running pended. None for any other step.
 */
function bornClosedCalls(state: RunState, record: NodeStepRecord): string[] {
  const { step } = record;
  if (step["kind"] !== "pending" || !state.waits.interrupts.has(record.nodeId) || typeof step["call_id"] !== "string") return [];
  return [step["call_id"]];
}

/** The keys a `steer` step holds. */
const STEER_STEP_KEYS = ["kind", "inbox", "from"] as const;

/**
 * Require a running node's `steer` step to consume the whole unread queue of its one steer inbox:
 * the node reads a steer inbox (`readsSteerInbox`); exactly `kind`, `inbox` and `from`; `inbox` a
 * declared top-level inbox, the one any earlier steer step of the node named; `from` the inbox's
 * unread events a read may take (`availableUnread`), in order, at least one.
 */
function validateSteerStep(state: RunState, nodeId: string, step: Readonly<JsonObject>): void {
  if (!state.flow.nodes.get(nodeId)!.readsSteerInbox) {
    throw new RangeError(`a steer step on node ${reprText(nodeId)}, which reads no steer inbox`);
  }
  const keys = Object.keys(step);
  if (keys.length !== STEER_STEP_KEYS.length || !STEER_STEP_KEYS.every((key) => Object.hasOwn(step, key))) {
    throw new RangeError("a steer step holds only kind, inbox and from");
  }
  const inbox = step["inbox"];
  if (typeof inbox !== "string" || inboxElementType(state, inbox) === null) {
    throw new RangeError(`steer inbox ${typeof inbox === "string" ? reprText(inbox) : writeAsText(inbox ?? null)} is no inbox the START declares`);
  }
  const earlier = (state.nodeSteps.get(nodeId) ?? []).find((prior) => prior["kind"] === "steer");
  if (earlier !== undefined && earlier["inbox"] !== inbox) {
    throw new RangeError(
      `node ${reprText(nodeId)} reads steer inbox ${reprText(String(earlier["inbox"]))}, not ${reprText(inbox)}`,
    );
  }
  // Ints, so the log writes them as ints.
  const raw = step["from"];
  const from = Array.isArray(raw) && raw.every((seq) => typeof seq === "bigint") ? seqList(raw) : null;
  if (from === null) throw new RangeError("a steer step's from must be a list of send seqs (ints)");
  if (from.length === 0) throw new RangeError("a steer step consumes at least one event");
  const queue = availableUnread(state.waits, inbox);
  if (from.length !== queue.length || from.some((seq, index) => seq !== queue[index])) {
    throw new RangeError(
      `a steer step must consume the whole unread queue [${queue.join(", ")}], found [${from.join(", ")}]`,
    );
  }
}

/** The steps' fields only an AGENT's `pending` step holds. */
const AGENT_PENDING_FIELDS = ["tool_call_id", "reply", "index"] as const;

/**
 * Require a running node's `pending` step to register its own host call. An AGENT's is checked by
 * `validateAgentPendingStep`. Otherwise the node makes a host tool call that isn't a take or
 * sleep, `tool` is its tool, `call_id` its TOOL call id, not yet registered; no agent fields; a
 * `deadline` only on an `ask`, as a record time; `arguments` an object; and the whole step within
 * the run's `MAX_EVENT_BYTES`.
 */
function validatePendingStep(state: RunState, nodeId: string, step: Readonly<JsonObject>): void {
  const node = state.flow.nodes.get(nodeId)!;
  if (node.hostToolCall() === null && node.grantedTools().length > 0) {
    validateAgentPendingStep(state, nodeId, step);
    return;
  }
  const toolId = node.hostToolCall()?.toolId ?? null;
  if (toolId === null) throw new RangeError(`a pending step on node ${reprText(nodeId)}, which calls no host tool`);
  if (toolId === TAKE || toolId === SLEEP) {
    throw new RangeError(`a pending step on node ${reprText(nodeId)}, whose ${toolId} the engine decides`);
  }
  const { call_id: callId, tool } = step;
  if (tool !== toolId) throw new RangeError(`pending tool ${writeAsText(tool ?? null)} isn't node ${reprText(nodeId)}'s tool ${reprText(toolId)}`);
  if (callId !== toolCallId(state.runId!, nodeId)) {
    throw new RangeError(`pending call_id ${writeAsText(callId ?? null)} isn't node ${reprText(nodeId)}'s call id`);
  }
  for (const field of AGENT_PENDING_FIELDS) {
    if (Object.hasOwn(step, field)) throw new RangeError(`a TOOL node's pending step holds no ${field}`);
  }
  if (Object.hasOwn(step, "deadline")) {
    if (tool !== ASK) throw new RangeError(`a pending ${reprText(tool)} call has no deadline; only ask times out`);
    requirePendingDeadline(step);
  }
  if (!isRecord(step["arguments"])) throw new RangeError("pending arguments must be an object");
  requireNewPendingCall(state, callId, step);
}

/**
 * Require an AGENT's `pending` step to register a call of its latest reply: `tool` a granted
 * tool's name, neither take nor sleep; `tool_call_id` the provider's id; `reply` the `n` of the
 * node's latest `model_call` step and `index` the call's place among its runnable calls, which
 * `call_id` (`<run>/<node>/<reply>/<index>`) names; no `arguments` (they are in the reply); a
 * `deadline` only on `ask_user`, as a record time; not yet registered, and within the run's
 * `MAX_EVENT_BYTES`.
 */
function validateAgentPendingStep(state: RunState, nodeId: string, step: Readonly<JsonObject>): void {
  const { call_id: callId, tool, tool_call_id: toolCallIdOf } = step;
  if (typeof tool !== "string" || tool === "" || tool === TAKE || tool === SLEEP) {
    throw new RangeError(`an agent's pending tool must name a host tool, got ${writeAsText(tool ?? null)}`);
  }
  if (typeof toolCallIdOf !== "string" || toolCallIdOf === "") {
    throw new RangeError("an agent's pending tool_call_id must be a non-empty string");
  }
  const index = stepInt(step["index"]);
  if (index === null || index < 0) throw new RangeError("an agent's pending index must be an int of at least 0");
  const reply = stepInt(step["reply"]);
  const latest = latestReply(state, nodeId);
  if (reply === null || reply !== latest) {
    throw new RangeError(
      `an agent's pending reply must be node ${reprText(nodeId)}'s latest model_call step ${latest ?? "(none)"}, ` +
        `got ${writeAsText(step["reply"] ?? null)}`,
    );
  }
  if (callId !== `${state.runId}/${nodeId}/${reply}/${index}`) {
    const which = `call ${index} of node ${reprText(nodeId)}'s reply ${reply}`;
    throw new RangeError(`pending call_id ${writeAsText(callId ?? null)} isn't ${which}`);
  }
  if (Object.hasOwn(step, "arguments")) {
    throw new RangeError("an agent's pending step holds no arguments; they are in its reply");
  }
  if (Object.hasOwn(step, "deadline")) {
    if (tool !== ASK_USER) throw new RangeError(`an agent's pending ${reprText(tool)} call has no deadline; only ask_user times out`);
    requirePendingDeadline(step);
  }
  requireNewPendingCall(state, callId, step);
}

/** Require a `pending` step's `deadline` to be a record time. */
function requirePendingDeadline(step: Readonly<JsonObject>): void {
  const deadline = step["deadline"];
  if (typeof deadline !== "string") throw new RangeError("pending deadline must be a record time");
  requireRecordTime(deadline, "pending deadline");
}

/** Require a `pending` step within the run's `MAX_EVENT_BYTES` whose call isn't registered yet. */
function requireNewPendingCall(state: RunState, callId: string, step: Readonly<JsonObject>): void {
  const size = eventBytes(step);
  if (size > state.runDefaults.maxEventBytes) throw new RangeError(eventTooLarge(`the pending step of ${reprText(callId)}`, size, state));
  if (state.waits.calls.has(callId)) {
    throw new RangeError(`call ${reprText(callId)} is already registered; it belongs to its node until the node settles`);
  }
}

/** A step field's int (a recorded int reads back as a bigint), or `null` when it isn't a safe int. */
function stepInt(value: unknown): number | null {
  const n = typeof value === "bigint" ? Number(value) : value;
  return typeof n === "number" && Number.isSafeInteger(n) ? n : null;
}

/** The `n` of `nodeId`'s latest `model_call` step, or `null` before any. */
function latestReply(state: RunState, nodeId: string): number | null {
  const at = (state.nodeSteps.get(nodeId) ?? []).findLastIndex((step) => step["kind"] === "model_call");
  return at < 0 ? null : at + 1;
}

/**
 * Require a `tool_result` step for a registered call (its `tool_call_id ?? call_id` is the call's
 * result key) to answer an open host call of this node (never a take or sleep, which the run
 * settles itself), within the run's `MAX_EVENT_BYTES`. A result no call registered (an agent's
 * inline call) is the node's own detail, refused only on a waiting node.
 *
 * An AGENT's result refers to a registered call only when that call is its own node's and of its
 * latest reply (`resultCall`), since provider ids repeat across replies and nodes. A call closed
 * by record takes one more result on a running AGENT, its closer: `closed` equal to the call's
 * reason, once. No other result carries `closed` for a registered call.
 */
function validateToolResultStep(state: RunState, nodeId: string, step: Readonly<JsonObject>, waiting: boolean): void {
  const key = typeof step["tool_call_id"] === "string" ? step["tool_call_id"] : step["call_id"];
  const call = resultCall(state, nodeId, key);
  if (call === undefined) {
    if (waiting) {
      throw new RangeError(`waiting node ${reprText(nodeId)} records only a tool_result answering one of its open calls`);
    }
    return;
  }
  if (call.nodeId !== nodeId) {
    throw new RangeError(`tool_result for call ${reprText(call.callId)} of node ${reprText(call.nodeId)}, not ${reprText(nodeId)}`);
  }
  if (call.tool === TAKE || call.tool === SLEEP) {
    throw new RangeError(`tool_result for call ${reprText(call.callId)}, a ${call.tool}, which the run settles itself`);
  }
  const agent = state.flow.nodes.get(nodeId)!.hostToolCall() === null;
  const closed = step["closed"];
  if (closed !== undefined && call.state === "open") {
    throw new RangeError(`tool_result for open call ${reprText(call.callId)} can't be a closer; only a record closes a call`);
  }
  if (agent && !waiting && call.state === "closed" && closed === call.closedReason) {
    if (call.resultSeq !== null) throw new RangeError(`call ${reprText(call.callId)} already has its closer`);
  } else if (call.state !== "open") {
    throw new RangeError(`tool_result for call ${reprText(call.callId)}, which is already ${settledText(call)}`);
  }
  const size = eventBytes(step);
  if (size > state.runDefaults.maxEventBytes) throw new RangeError(eventTooLarge(`the answer to ${reprText(call.callId)}`, size, state));
}

/**
 * The registered call a `tool_result` step on `nodeId` with result key `key` refers to, or
 * `undefined`. A TOOL node's key is a call id, unique in the run, so any node's call may match (and
 * the caller refuses another node's). An AGENT's key is a provider id, which repeats across
 * replies and nodes, so only its own calls of its latest reply match.
 *
 * @throws RangeError when two calls of the latest reply share the key.
 */
function resultCall(state: RunState, nodeId: string, key: unknown): Call | undefined {
  if (state.flow.nodes.get(nodeId)!.hostToolCall() !== null) {
    return [...state.waits.calls.values()].find((each) => each.resultKey === key);
  }
  const latest = latestReply(state, nodeId);
  const calls = nodeCalls(state.waits, nodeId).filter(
    (each) => each.resultKey === key && stepInt(pendingStepOf(state, each)?.["reply"]) === latest,
  );
  if (calls.length > 1) {
    const ids = writeAsText(calls.map((each) => each.callId));
    throw new RangeError(`calls ${ids} of one reply share the tool call id ${reprText(String(key))}`);
  }
  return calls[0];
}

/** The `pending` step that registered `call` on its node, or `undefined` (a take or sleep). */
function pendingStepOf(state: RunState, call: Call): Readonly<JsonObject> | undefined {
  const steps = state.nodeSteps.get(call.nodeId) ?? [];
  return steps.find((step) => step["kind"] === "pending" && step["call_id"] === call.callId);
}

/**
 * Drop a settled node's start count and recorded steps, and the values its steer reads consumed.
 *
 * @throws RangeError when the node has no start count.
 */
function forgetNodeSteps(state: RunState, nodeId: string): void {
  if (!state.nodeStarts.delete(nodeId)) throw new RangeError(`node ${reprText(nodeId)} has no start count`);
  for (const step of state.nodeSteps.get(nodeId) ?? []) {
    if (step["kind"] === "steer") for (const seq of seqList(step["from"]) ?? []) state.inboxEvents.delete(seq);
  }
  state.nodeSteps.delete(nodeId);
  state.nodeStepSeqs.delete(nodeId);
}

/**
 * Return copies of `nodeId`'s recorded steps, each `steer` step's copy given `values`: the
 * consumed sends' values (copies, `copyValue`), in `from` order. The steps the state holds don't
 * change.
 *
 * @throws RangeError when a consumed send's value is gone (the log and the state disagree).
 */
export function stepsWithSteerValues(state: RunState, nodeId: string): ValueRecord[] {
  return (state.nodeSteps.get(nodeId) ?? []).map((step) => stepWithSteerValues(state, nodeId, step));
}

/** A copy of one of `nodeId`'s recorded steps, a `steer` step's given its values (`stepsWithSteerValues`). */
function stepWithSteerValues(state: RunState, nodeId: string, step: Readonly<JsonObject>): ValueRecord {
  const copy = structuredClone(step) as ValueRecord;
  if (step["kind"] !== "steer") return copy;
  copy["values"] = (seqList(step["from"]) ?? []).map((seq) => {
    const value = state.inboxEvents.get(seq);
    if (value === undefined) {
      throw new RangeError(`node ${reprText(nodeId)}'s steer step names send ${seq}, which the run doesn't hold`);
    }
    return copyValue(value.toObject());
  });
  return copy;
}

/** Require an existing unknown node in the in-flight frontier, with no pending commit. */
function requireCommittingNode(state: RunState, nodeId: string): void {
  if (!state.flow.nodes.has(nodeId) || state.nodeState.get(nodeId) !== NodeState.UNKNOWN || !state.inFlight.has(nodeId)) {
    throw new RangeError(`pool_commit node ${reprText(nodeId)} is not in flight`);
  }
  const pending = state.pendingCommits.get(nodeId);
  if (pending !== undefined) {
    throw new RangeError(`pool_commit node ${reprText(nodeId)} already has a pending commit at seq ${pending.seq}`);
  }
  const expansion = pendingExpansion(state, nodeId);
  if (expansion.length > 0) {
    throw new RangeError(`pool_commit node ${reprText(nodeId)} has pending expansion commits only its expansion admits`);
  }
}

/** Refuse a value commit by a take node, which commits the events it took (a take commit). */
function requireValueCommitNode(state: RunState, nodeId: string): void {
  if (state.flow.nodes.get(nodeId)!.hostToolCall()?.toolId === TAKE) {
    throw new RangeError(`take node ${reprText(nodeId)} commits the events it took, not a value`);
  }
}

/**
 * Validate a take commit by an in-flight take node with no pending commit, under its target. A node
 * holding its take call commits what the call settled: its assigned events, closed when its inbox is
 * closed with nothing left a take may read; or, timed out, nothing. A node holding none commits
 * inline: the oldest event a take may read (`availableUnread`) or all of them, closed when its
 * inbox is closed and it takes them all; with none available, nothing, closed as its inbox is. An
 * inline commit of nothing on an open inbox is a timeout the reducer can't check, so it is accepted
 * on the engine's word.
 *
 * @throws RangeError otherwise.
 */
function validateTakeCommit(state: RunState, record: PoolCommitRecord): void {
  const { nodeId } = record;
  const take = record.take!;
  requireCommittingNode(state, nodeId);
  const node = state.flow.nodes.get(nodeId)!;
  if (node.hostToolCall()?.toolId !== TAKE) throw new RangeError(`a take commit by node ${reprText(nodeId)}, which is no take`);
  const target = node.commitAs ?? nodeId;
  if (record.key !== target) {
    throw new RangeError(`the take commit of ${reprText(nodeId)} is under ${reprText(record.key)}, not ${reprText(target)}`);
  }
  if (inboxElementType(state, take.inbox) === null) {
    throw new RangeError(`the take commit of ${reprText(nodeId)} reads ${reprText(take.inbox)}, no inbox the START declares`);
  }
  const w = state.waits;
  const closed = w.inboxes.get(take.inbox)?.closed ?? false;
  const held = w.calls.get(toolCallId(state.runId!, nodeId));
  let events: readonly number[];
  let expectedClosed: boolean;
  if (held !== undefined) {
    if (held.inbox !== take.inbox) {
      throw new RangeError(`the take commit of ${reprText(nodeId)} reads ${reprText(take.inbox)}, but its call reads ${writeAsText(held.inbox)}`);
    }
    if (held.state === "open" || held.state === "closed") {
      throw new RangeError(`take node ${reprText(nodeId)} commits from its call, which is ${held.state}`);
    }
    events = held.state === "timed_out" ? [] : held.assigned;
    expectedClosed = held.state === "answered" && closed && availableUnread(w, take.inbox).length === 0;
  } else {
    const available = availableUnread(w, take.inbox);
    const k = take.events.length;
    const counted = available.length === 0 ? k === 0 : k === 1 || k === available.length;
    events = counted ? available.slice(0, k) : available;
    expectedClosed = closed && k === available.length;
  }
  if (take.events.length !== events.length || take.events.some((seq, index) => seq !== events[index])) {
    throw new RangeError(
      `the take commit of ${reprText(nodeId)} must consume ${writeAsText([...events])}` +
        `${held === undefined ? " (the oldest event a take may read, or all of them)" : ""}, found ${writeAsText([...take.events])}`,
    );
  }
  if (take.closed !== expectedClosed) {
    throw new RangeError(`the take commit of ${reprText(nodeId)} must hold closed ${expectedClosed}`);
  }
}

/**
 * Rebuild a take commit's value, `{events, closed}`, from the inbox store, checked against its
 * target's contract.
 *
 * @throws RangeError when a consumed send's value is gone, or the contract refuses the value
 *   (sends are checked against their inbox's type, so this means the flow and the log disagree).
 */
function takeValueJson(state: RunState, pending: PendingCommit): string {
  const take = pending.take!;
  const events = take.events.map((seq) => {
    const value = state.inboxEvents.get(seq);
    if (value === undefined) throw new RangeError(`the take commit at seq ${pending.seq} names send ${seq}, which the run doesn't hold`);
    return value.toObject();
  });
  try {
    return committedText(state, pending.key, { events, closed: take.closed });
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    throw new RangeError(`the take commit at seq ${pending.seq} rebuilds to a value its target refuses: ${error.message}`, {
      cause: error,
    });
  }
}

/** The expansion commits `spawnerId` wrote that its `NodeExpanded` hasn't admitted yet, in seq order. */
function pendingExpansion(state: RunState, spawnerId: string): PendingCommit[] {
  return [...state.pendingCommits.values()]
    .filter((pending) => pending.spawner === spawnerId && !pending.admitted)
    .sort((a, b) => a.seq - b.seq);
}

/** Reject an outcome other than success for a node whose commit only its success accepts. */
function requireNoPendingCommit(state: RunState, nodeId: string, kind: string): void {
  // A child START that fails drops its expansion commit: no success will name it.
  if (state.pendingCommits.has(nodeId) && !(kind === "NodeFailed" && state.pendingCommits.get(nodeId)!.admitted)) {
    throw new RangeError(`${kind} node ${reprText(nodeId)} has a pending pool_commit only its success accepts`);
  }
  if (kind !== "NodeExpanded" && pendingExpansion(state, nodeId).length > 0) {
    throw new RangeError(`${kind} node ${reprText(nodeId)} has pending expansion commits only its expansion admits`);
  }
}

/**
 * Require `nodeId` to name an existing node of `UNKNOWN` disposition in the in-flight frontier, as
 * a `kind` outcome (`NodeSucceeded`, `NodeRouted`, `NodeFailed` or `NodeExpanded`) needs.
 *
 * @returns The node.
 */
function requireOutcomeNode(state: RunState, nodeId: string, kind: string): Node {
  if (!nodeId) throw new RangeError(`${kind} node_id must be non-empty`);
  const node = state.flow.nodes.get(nodeId);
  if (node === undefined) throw new RangeError(`${kind} node ${reprText(nodeId)} does not exist in the flow`);
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN) {
    throw new RangeError(`${kind} node ${reprText(nodeId)} must have UNKNOWN disposition`);
  }
  if (!state.inFlight.has(nodeId)) throw new RangeError(`${kind} node ${reprText(nodeId)} is not in-flight`);
  return node;
}

/** Require an existing unknown spawner in the in-flight frontier, with no commit pending. */
function requireExpandingNode(state: RunState, nodeId: string): void {
  const node = requireOutcomeNode(state, nodeId, "NodeExpanded");
  if (node.isSpawner !== true) throw new RangeError(`NodeExpanded node ${reprText(nodeId)} is not a graph spawner`);
  requireNoPendingCommit(state, nodeId, "NodeExpanded");
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
  validateLearnedScope(state, spawnerId);
}

/** Call the injected builder with a copy of the spawner and return a copy of its fragment. */
function buildGrowthFragment(spawner: Node, builderSeed: JsonValue, growBuilder: GrowBuilder): Flow {
  const built: unknown = growBuilder(copyGraph(spawner), builderSeed);
  if (!(built instanceof Flow)) throw new TypeError("grow_builder must return a Flow");
  return copyGraph(built);
}

/** A checked growth's delta, short of its admitted commits, and its topology change. */
type GrownTopology = { readonly delta: Omit<GrowthDelta, "admitted">; readonly topology: GrowthTopology };

/** Validate one growth and build the detached change commit applies in place. */
function prepareGrowthTopology(
  state: RunState,
  record: NodeExpandedRecord,
  fragment: Flow,
  validatedFragment: ValidatedFragment,
): GrownTopology {
  const [newNodeIds, newEdgeIds] = validatedFragment;
  const budget = state.runDefaults.maxTotalNodes;
  if (state.flow.nodes.size + newNodeIds.length > budget) {
    raiseGrowthRejected(
      new RangeError(`expansion exceeded node budget (${budget}) at spawner ${reprText(record.spawnerId)}`),
    );
  }
  const derivedDepth = validatedGrowthDepth(state, record.spawnerId);
  const index = growthIndex(state);
  validateGrowthPrune(state, record.spawnerId, record.pruneIds);
  const [retired, deferredPrune] = retiredIds(state, record.spawnerId, record.pruneIds, index);
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

  const delta = Object.freeze({
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
  // A lookup over the fragment and live nodes, not a merged copy.
  const combinedNodeIds = { has: (id: string): boolean => fragment.nodes.has(id) || state.flow.nodes.has(id) };
  // A reference may also read an AGENT's new messages, under their pool name `<id>.new_messages`.
  const readableNames = {
    has: (name: string): boolean => {
      if (combinedNodeIds.has(name)) return true;
      if (!name.endsWith(`.${NEW_MESSAGES}`)) return false;
      const owner = name.slice(0, -NEW_MESSAGES.length - 1);
      return (fragment.nodes.get(owner) ?? state.flow.nodes.get(owner))?.producesMessages === true;
    },
  };
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
      readableNames,
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

/** A value as a message names it: quoted text, else the value's string form. */
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

/**
 * Apply the closed reference rules to raw or carrier-wrapped fragment wiring. `readableNames` holds
 * the fragment and live node ids, plus `<id>.new_messages` for each AGENT that produces messages.
 */
function validateGrowthWiringRefs(
  nodeId: string,
  node: Node,
  wiring: ReadonlyMap<string, unknown>,
  readableNames: { has(name: string): boolean },
  producerContract: (nodeId: string) => OutputContract | null,
  reserved: ReadonlySet<string>,
): void {
  for (const [param, source] of wiring) {
    let paths: string[];
    try {
      paths = [...sourceRefs(source)];
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      raiseGrowthRejected(
        new RangeError(
          `growth fragment node ${reprText(nodeId)} wiring ${reprText(param)} is invalid: ${error.message}`,
          { cause: error },
        ),
      );
    }
    for (const path of paths) {
      const extraHeads = node.bindsPerItem === true && !reserved.has(param) ? ["item"] : [];
      let error = classifyPath(path, readableNames, new Set(), extraHeads);
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
    try {
      for (const path of sourceRefs(source)) names.add(refHead(path));
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      throw new RangeError(`node ${reprText(nodeId)} has invalid wiring: ${error.message}`, { cause: error });
    }
  }
  return Object.freeze([...names]);
}

/**
 * Yield each reference path the expressions in one raw or carrier-wrapped wiring source read.
 *
 * @throws ExpressionError when an expression doesn't parse.
 */
function* sourceRefs(source: unknown): Generator<string> {
  for (const [, text] of iterSourceStrings(source)) yield* exprRefsOf(text);
}

/** The name a reference path starts with: a node id, or a reserved head such as `item`. */
function refHead(path: string): string {
  return path.split(".", 1)[0]!;
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

/** Return `map`'s value at `key`, first storing `make()` when absent. */
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
 * @param index The growth index: each owner's directly grown live nodes, and what each node names.
 * @returns The node ids retired by this growth, and the candidates kept live and deferred to a
 *   later growth.
 */
function retiredIds(
  state: RunState,
  spawnerId: string,
  pruneIds: readonly string[],
  index: GrowthIndex,
): readonly [ReadonlySet<string>, ReadonlySet<string>] {
  const { grown } = index;
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
    for (const read of readIds(state, index, pending.pop()!)) {
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

/**
 * Return the node ids a live node reads: its incoming edges' sources, plus the names it names
 * through its wiring or output targets (`GrowthIndex.names`).
 */
function readIds(state: RunState, index: GrowthIndex, nodeId: string): Set<string> {
  const reads = new Set(state.flow.incoming(nodeId).map((edge) => edge.from));
  for (const name of index.names.get(nodeId) ?? []) reads.add(name);
  return reads;
}

/**
 * Reject requested prune ids that are not dynamic overlay or a self-prunable LOOP clone.
 *
 * Unsettled requested ids are not errors: `retiredIds` keeps them live. A prune id must be live,
 * so it can't name a new fragment node, whose id `validateGrowthFragment` refused if live.
 */
function validateGrowthPrune(
  state: RunState,
  spawnerId: string,
  pruneIds: readonly string[],
): void {
  const spawner = state.flow.nodes.get(spawnerId)!;
  for (const nodeId of pruneIds) {
    if (!state.flow.nodes.has(nodeId)) {
      raiseGrowthRejected(new RangeError(`growth prune node ${reprText(nodeId)} does not exist`));
    }
    if (nodeId === state.flow.startId || nodeId === state.flow.endId) {
      raiseGrowthRejected(new RangeError(`growth cannot prune top-level boundary ${reprText(nodeId)}`));
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
      let paths: string[];
      try {
        paths = [...sourceRefs(source)];
      } catch (error) {
        if (!(error instanceof ExpressionError)) throw error;
        throw new RangeError(`surviving node ${reprText(nodeId)} has invalid wiring: ${error.message}`, {
          cause: error,
        });
      }
      if (paths.some((path) => pruneIds.has(refHead(path)))) {
        raiseGrowthRejected(new RangeError(`surviving node ${reprText(nodeId)} wiring depends on pruned node`));
      }
    }
  }
}

/** Validate a route's exact scalar carriers and its executing router. */
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
  requireOutcomeNode(state, nodeId, "NodeRouted");
  requireNoPendingCommit(state, nodeId, "NodeRouted");
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

/**
 * Require a learnable run's node that recalls memory to carry its exact recorded memory stack, so
 * learning recalls and attributes against the stack the run recorded.
 */
function validateLearnedScope(state: RunState, nodeId: string): void {
  const node = state.flow.nodes.get(nodeId);
  if (node === undefined) throw new RangeError(`node ${reprText(nodeId)} does not exist in the flow`);
  if (state.learnable !== true || node.needsMemoryContext !== true) return;
  const scope = node.memoryScope;
  if (scope !== null) {
    // A `MemoryScope` validated itself when built.
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
  requireOutcomeNode(state, nodeId, "NodeFailed");
  requireNoPendingCommit(state, nodeId, "NodeFailed");
  validateLearnedScope(state, nodeId);
}

/**
 * Validate one run-start commit: a system value, or the START inputs after `system.run_id`.
 *
 * @throws RangeError when it names another node than the START, its key is neither form, the
 *   system value is already committed, `system.run_id` isn't the run id, or the START inputs
 *   aren't a record or come before `system.run_id`.
 */
function validateInitialCommit(state: RunState, record: PoolCommitRecord): void {
  const startId = state.flow.startId;
  const { key } = record;
  if (record.nodeId !== startId) {
    throw new RangeError(`a run-start pool_commit must name the START ${reprText(startId)}, not ${reprText(record.nodeId)}`);
  }
  if (record.value === null) throw new RangeError("a run-start pool_commit holds its value, not from");
  const value = TypedValue.fromWire(record.value).toObject();
  if (key === startId) {
    if (state.pool.readRoot(["system", "run_id"]).value !== state.runId) {
      throw new RangeError("the run start commits system.run_id, equal to the run id, before the START inputs");
    }
    if (!isRecord(value)) throw new RangeError("the START inputs' pool_commit must hold a record");
    return;
  }
  const name = key.startsWith(SYSTEM_PREFIX) ? key.slice(SYSTEM_PREFIX.length) : "";
  if (!name) {
    throw new RangeError(
      `a run-start pool_commit key must be system.<name> or the START ${reprText(startId)}, got ${reprText(key)}`,
    );
  }
  if (state.pool.system.has(name)) throw new RangeError(`the run start commits ${reprText(key)} twice`);
  if (name === "run_id" && value !== state.runId) throw new RangeError("system.run_id must equal the run id");
}

/**
 * Validate one replayed node commit against its target exactly once.
 *
 * @throws RangeError when the node isn't in flight or already has a pending commit, the target
 *   doesn't exist, the commit holds `from` when its node isn't derived (`isDerivedCommit`) or no
 *   `from` when it is, or the value isn't canonical for it.
 */
function validateNodeCommit(state: RunState, record: PoolCommitRecord): void {
  requireCommittingNode(state, record.nodeId);
  requireValueCommitNode(state, record.nodeId);
  const targetNode = state.flow.nodes.get(record.key);
  if (targetNode === undefined) {
    throw new RangeError(`success commit target ${reprText(record.key)} does not exist in the flow`);
  }
  requireCommitFrom(state, record.nodeId, record.from);
  // A derived commit's value is rebuilt, and checked, when its success accepts it.
  if (record.value === null) return;
  const typed = TypedValue.fromWire(record.value);
  const [wire] = canonicalOutputValue(typed, targetNode.outputContract, nodeOutputType(targetNode));
  if (logJson(wire) !== logJson(record.value)) {
    throw new RangeError("PoolCommitRecord value is not canonical for its target");
  }
}

/**
 * Require a node's own commit to hold `from` exactly when its node is derived (`isDerivedCommit`).
 *
 * @throws RangeError otherwise.
 */
function requireCommitFrom(state: RunState, nodeId: string, from: ReadonlyMap<string, number> | null): void {
  const derived = isDerivedCommit(state, nodeId);
  if (derived && from === null) {
    throw new RangeError(`pool_commit of ${reprText(nodeId)} must hold from: its node is derived`);
  }
  if (!derived && from !== null) {
    throw new RangeError(`pool_commit of ${reprText(nodeId)} holds from, but its node isn't derived`);
  }
}

/**
 * The `n`s of the steps `nodeId`'s new messages are made of, as its node picks them from its
 * recorded steps (`Node.messageSteps`).
 *
 * @throws RangeError when the node picks a step it didn't record.
 */
function messageStepNumbers(state: RunState, nodeId: string): number[] {
  const steps = state.nodeSteps.get(nodeId) ?? [];
  return state.flow.nodes.get(nodeId)!.messageSteps(steps).map((n) => {
    if (!(Number.isSafeInteger(n) && 1 <= n && n <= steps.length)) {
      throw new RangeError(`node ${reprText(nodeId)} picked step ${n}, which it didn't record`);
    }
    return n;
  });
}

/** The seqs of the steps `nodeId`'s new messages are made of (`messageStepNumbers`). */
function messageStepSeqs(state: RunState, nodeId: string): number[] {
  const seqs = state.nodeStepSeqs.get(nodeId) ?? [];
  return messageStepNumbers(state, nodeId).map((n) => seqs[n - 1]!);
}

/**
 * Validate a messages commit and rebuild its value: its node produces messages and is in flight,
 * the key is `messagesKey(<node>)`, it follows the node's own output commit and is the first since,
 * and `steps` are exactly the seqs the node picks (`messageStepSeqs`).
 *
 * @returns The rebuilt `list(Message)` value's canonical typed wire text.
 * @throws RangeError otherwise, or when the picked steps rebuild to a value `list(Message)`
 *   refuses.
 */
function validateMessagesCommit(state: RunState, record: PoolCommitRecord): string {
  const { nodeId } = record;
  const node = state.flow.nodes.get(nodeId);
  if (node?.producesMessages !== true) {
    throw new RangeError(`a messages commit by node ${reprText(nodeId)}, which produces no messages`);
  }
  if (state.nodeState.get(nodeId) !== NodeState.UNKNOWN || !state.inFlight.has(nodeId)) {
    throw new RangeError(`pool_commit node ${reprText(nodeId)} is not in flight`);
  }
  if (record.key !== messagesKey(nodeId)) {
    throw new RangeError(
      `the messages commit of ${reprText(nodeId)} is under ${reprText(record.key)}, not ${reprText(messagesKey(nodeId))}`,
    );
  }
  const output = state.pendingCommits.get(nodeId);
  if (output === undefined || output.spawner !== null || output.take !== null || output.valueJson === null) {
    throw new RangeError(`the messages commit of ${reprText(nodeId)} must follow its output commit`);
  }
  const pending = state.pendingMessages.get(nodeId);
  if (pending !== undefined) {
    throw new RangeError(`node ${reprText(nodeId)} already has a pending messages commit at seq ${pending.seq}`);
  }
  const numbers = messageStepNumbers(state, nodeId);
  const seqs = state.nodeStepSeqs.get(nodeId) ?? [];
  const expected = numbers.map((n) => seqs[n - 1]!);
  if (!sameItems(record.steps!, expected)) {
    throw new RangeError(
      `the messages commit of ${reprText(nodeId)} must name steps [${expected.join(", ")}], found [${record.steps!.join(", ")}]`,
    );
  }
  const steps = state.nodeSteps.get(nodeId) ?? [];
  // A steer step's events are rebuilt into messages, so it needs its values.
  const picked = numbers.map((n) => stepWithSteerValues(state, nodeId, steps[n - 1]!) as unknown as Readonly<JsonObject>);
  try {
    const value = node.rebuildMessages(picked, decodeTyped(output.valueJson), localTypes(state.flow, node));
    const [wire] = canonicalTyped(value, newMessagesType());
    return logJson(wire);
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    throw new RangeError(`the messages of ${reprText(nodeId)} rebuild to a value list(Message) refuses: ${error.message}`, {
      cause: error,
    });
  }
}

/**
 * Validate one of a spawner's expansion commits, a commit by a spawner under a key the flow doesn't
 * hold yet: the spawner is in flight with no commit of its own pending, the commit holds `from`,
 * and its child START `key` has no commit pending yet.
 *
 * @throws RangeError otherwise.
 */
function validateExpansionCommit(state: RunState, record: PoolCommitRecord): void {
  const spawnerId = record.nodeId;
  if (state.nodeState.get(spawnerId) !== NodeState.UNKNOWN || !state.inFlight.has(spawnerId)) {
    throw new RangeError(`pool_commit node ${reprText(spawnerId)} is not in flight`);
  }
  if (state.pendingCommits.has(spawnerId)) {
    throw new RangeError(`pool_commit node ${reprText(spawnerId)} already has a pending commit`);
  }
  if (state.pendingCommits.has(record.key)) {
    throw new RangeError(`child START ${reprText(record.key)} already has a pending expansion commit`);
  }
  if (record.from === null) throw new RangeError(`the expansion commit for ${reprText(record.key)} holds no from`);
}

/**
 * Validate a send: a commit by the START under `<START>.<name>` for a top-level `Inbox(T)` param,
 * holding a value canonical for `T` and within the run's `MAX_EVENT_BYTES`, to an open inbox.
 *
 * @throws RangeError otherwise (a `TypeCheckError` when the value fails `T`).
 */
function validateSend(state: RunState, record: PoolCommitRecord): void {
  const name = record.key.slice(sendKey(state, "").length);
  if (record.nodeId !== state.flow.startId) {
    throw new RangeError(`a send to ${reprText(record.key)} must name the START, not ${reprText(record.nodeId)}`);
  }
  const element = inboxElementType(state, name);
  if (element === null) throw new RangeError(`send key ${reprText(record.key)} names no inbox the START declares`);
  if (record.value === null) throw new RangeError(`a send to ${reprText(name)} holds its value`);
  if (state.waits.inboxes.get(name)?.closed === true) throw new RangeError(`inbox ${reprText(name)} is closed`);
  const [wire] = canonicalTyped(TypedValue.fromWire(record.value), element);
  if (logJson(wire) !== logJson(record.value)) {
    throw new RangeError(`the send to ${reprText(name)} isn't canonical for the inbox's type`);
  }
  const size = eventBytes(record.value.value);
  if (size > state.runDefaults.maxEventBytes) throw new RangeError(eventTooLarge(`the send to ${reprText(name)}`, size, state));
}

/**
 * Require a replayed success to name its node's pending commit.
 *
 * @throws RangeError when the node isn't in flight, or `commit` isn't the seq of its pending
 *   commit.
 */
function validateSuccessCommit(state: RunState, record: NodeSucceededRecord): void {
  requireOutcomeNode(state, record.nodeId, "NodeSucceeded");
  const pending = state.pendingCommits.get(record.nodeId);
  if (pending?.seq !== record.commit) {
    throw new RangeError(
      `node_succeeded of ${reprText(record.nodeId)} names commit ${record.commit}, ` +
        `not its pending pool_commit ${pending === undefined ? "(none)" : pending.seq}`,
    );
  }
  const messages = state.pendingMessages.get(record.nodeId)?.seq ?? null;
  if (record.messages !== messages) {
    throw new RangeError(
      `node_succeeded of ${reprText(record.nodeId)} names messages ${record.messages ?? "(none)"}, ` +
        `not its pending messages commit ${messages ?? "(none)"}`,
    );
  }
}

/** Validate the touched topology and freeze one sparse successful-output delta. */
function prepareSuccessDelta(state: RunState, record: NodeSucceededRecord, pending: PendingCommit): SuccessDelta {
  if (!isExactly(state.pool.store, Map)) throw new RangeError("success requires an exact variable-pool store");
  const target = pending.key;
  const outgoingEdgeIds: string[] = [];
  const edgeHeads: string[] = [];
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
    edgeHeads.push(edge.to);
    seenOutgoingEdgeIds.add(edgeId);
    remainingBySuccessor.set(edge.to, (remainingBySuccessor.get(edge.to) ?? 0) + 1);
    setDefault(edgeIdsBySuccessor, edge.to, () => []).push(edgeId);
  }

  for (const edge of outgoing) {
    settledEdgeIds.add(edge.id);
    const successor = edge.to;
    const remaining = remainingBySuccessor.get(successor)! - 1;
    remainingBySuccessor.set(successor, remaining);
    if (remaining) continue;
    const required = new Set(edgeIdsBySuccessor.get(successor));
    // Another UNKNOWN incoming edge means the scan can only return "wait" or "dead", never
    // "ready", and a success skips both. So the running count can skip the scan, which would be
    // quadratic for a wide join. The scan still classifies and validates the join once it can be
    // ready.
    if ((state.unknownIncoming.get(successor) ?? 0) > required.size) continue;
    if (sparseDisposition(state.flow, state.edgeState, settledEdgeIds, successor, required) !== "ready") continue;
    if (state.nodeState.get(successor) !== NodeState.UNKNOWN) {
      throw new RangeError(`newly-ready successor ${reprText(successor)} must have UNKNOWN disposition`);
    }
    newlyReadyIds.push(successor);
  }

  return Object.freeze({
    nodeId: record.nodeId,
    commitTarget: target,
    // The caller rebuilt a replayed derived value.
    valueJson: pending.valueJson!,
    outgoingEdgeIds: Object.freeze(outgoingEdgeIds),
    edgeHeads: Object.freeze(edgeHeads),
    newlyReadyIds: Object.freeze(newlyReadyIds),
    observerCommitAs: target === record.nodeId ? null : target,
    interrupted: state.waits._honored.has(record.nodeId),
    consumedEvents: pending.take?.events ?? [],
    messages: state.pendingMessages.get(record.nodeId) ?? null,
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
  // A waiting node blocks quiescence as an in-flight one does.
  if (
    state.result !== null ||
    state.ready.length > 0 ||
    outcome.newlyReadyIds.length > 0 ||
    state.inFlight.size !== 1 ||
    !state.inFlight.has(outcome.completingNodeId) ||
    state.waits.waiting.size > 0
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
 * Return one validated route's frozen skip closure, plus the terminal failure it first
 * proves (`causalFailure`), if any.
 */
function prepareRouteDelta(
  state: RunState,
  record: NodeRoutedRecord,
  deriveTerminalFailure: boolean,
): { delta: RouteDelta; causalFailure: Failure | null } {
  const edgeUpdates = new Map<string, NodeState>();
  const edgeHeads: string[] = [];
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
    edgeHeads.push(edge.to);

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
    edgeHeads: Object.freeze(edgeHeads),
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
  // Membership after the growth, without copying the ids.
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
    setDefault(owners, target, () => []).push(nodeId);
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

/** Count one more `UNKNOWN` incoming edge of `nodeId`. */
function addUnknownIncoming(state: RunState, nodeId: string): void {
  state.unknownIncoming.set(nodeId, (state.unknownIncoming.get(nodeId) ?? 0) + 1);
}

/** Count one fewer `UNKNOWN` incoming edge of `nodeId`, as it settles or retires. */
function dropUnknownIncoming(state: RunState, nodeId: string): void {
  const count = (state.unknownIncoming.get(nodeId) ?? 0) - 1;
  if (count > 0) state.unknownIncoming.set(nodeId, count);
  else state.unknownIncoming.delete(nodeId);
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
    isExactly(state.pendingCommits, Map) &&
    isExactly(state.producers, Map) &&
    isExactly(state.crashedAttempts, Map) &&
    isExactly(state.depth, Map) &&
    isExactly(state.growthOwner, Map) &&
    isExactly(state.deferredPrune, Set) &&
    isExactly(state.wiringOverrides, Map) &&
    isExactly(state.waits, WaitsState) &&
    isExactly(state.inboxEvents, Map) &&
    isRecord(state.input) &&
    isRecord(state.effectiveLlmByStaticPath) &&
    isExactly(state.backwardState, RuntimeBackwardState) &&
    isExactly(state.initialStaticIds, Set);
  if (!exactContainers) throw new RangeError("RunStarted requires pristine reducer container types");
  const flow = state.flow;
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
    state.nodeStepSeqs.size === 0 &&
    state.pendingCommits.size === 0 &&
    state.pendingMessages.size === 0 &&
    state.producers.size === 0 &&
    state.inputsCommitted === false &&
    state.crashedAttempts.size === 0 &&
    state.lastStopSequence === null &&
    state.result === null &&
    state.terminalUnreachableFailure === null &&
    state.depth.size === 0 &&
    state.growthOwner.size === 0 &&
    state.deferredPrune.size === 0 &&
    state.wiringOverrides.size === 0 &&
    state.inboxEvents.size === 0 &&
    state.waits.waiting.size === 0 &&
    state.waits.calls.size === 0 &&
    state.waits.inboxes.size === 0 &&
    state.waits.interrupts.size === 0 &&
    state.waits.stops.size === 0 &&
    state.waits.lastFrontier === null &&
    !state.waits._inputsCommitted &&
    state.waits._lastWoken.size === 0 &&
    state.waits._pendingTakes.size === 0 &&
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
  effectiveLlmByStaticPath: Record<string, JsonObject>;
  callTimeLlm: JsonObject | null;
  memory: RunMemory;
  recordedSkills: Readonly<Record<string, RecordedSkill>>;
  recordedTools: Readonly<Record<string, RecordedTool>>;
  recordedToolGroups: Readonly<Record<string, readonly string[]>>;
  runDefaults: Defaults;
  backwardState: RuntimeBackwardState;
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

  const startId = plan.startId;
  const wiring = canonicalSeed(record.startWiringOverride)[1];
  const effective = canonicalSeed(record.effectiveLlmByStaticPath)[1];
  if (!isRecord(wiring)) throw new TypeError("canonical START wiring is not a dict");
  if (!isRecord(effective)) throw new TypeError("canonical effective LLM map is not a dict");

  const backwardState = new RuntimeBackwardState(
    record.memories === null ? [] : record.memories.map((spec) => spec.key),
  );
  return {
    nodeState: new Map(state.nodeState),
    edgeState: new Map(state.edgeState),
    // START becomes ready with the run start's last commit, its inputs.
    ready: [],
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
    effectiveLlmByStaticPath: effective as Record<string, JsonObject>,
    callTimeLlm: record.callTimeLlm === null ? null : (canonicalSeed(record.callTimeLlm)[1] as JsonObject),
    memory: {
      snapshots: record.memorySnapshots,
      memories: record.memories,
      memoryStacks: record.memoryStacks,
      learnable: record.learnable,
    },
    recordedSkills: record.skills,
    recordedTools: record.tools,
    recordedToolGroups: record.toolGroups,
    runDefaults: fromRecorded(record.defaults),
    backwardState,
    observers: [new RunStarted()],
  };
}

/** Install the already detached values with total replacement assignments only. */
function applyRunStarted(state: RunState, apply: RunStartedApply): void {
  state.nodeState = apply.nodeState;
  // The constructor's `unknownIncoming` stays valid: RunStarted requires the all-UNKNOWN edge
  // states it was seeded from.
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
  state.effectiveLlmByStaticPath = apply.effectiveLlmByStaticPath;
  state.callTimeLlm = apply.callTimeLlm;
  state.installMemory(apply.memory);
  state.recordedSkills = apply.recordedSkills;
  state.recordedTools = apply.recordedTools;
  state.recordedToolGroups = apply.recordedToolGroups;
  state.runDefaults = apply.runDefaults;
  state.backwardState = apply.backwardState;
  state.nextSequence = apply.nextSequence;
}

/** The detached values one success commit installs, all built before the durable append. */
interface SuccessApply {
  value: TypedValue;
  /** The node's new messages, when the success accepts a messages commit. */
  messagesValue: TypedValue | null;
  terminal: TerminalApply | null;
  observers: readonly RunEvent[];
}

/** Decode independent pool, observer and terminal copies before append. */
function materializeSuccess(plan: SuccessPlan): SuccessApply {
  const value = normalizeTypedFiles(plan.files, decodeTyped(plan.delta.valueJson));
  const messages = plan.delta.messages;
  const messagesValue = messages === null ? null : normalizeTypedFiles(newMessagesType(), decodeTyped(messages.valueJson));
  const observerValue = normalizeTypedFiles(plan.files, decodeTyped(plan.delta.valueJson)).toObject();
  const terminal = plan.terminal !== null ? materializeTerminal(plan.terminal) : null;
  const observers: RunEvent[] = [
    new NodeSucceeded({
      nodeId: plan.delta.nodeId,
      output: observerValue,
      commitAs: plan.delta.observerCommitAs,
      interrupted: plan.delta.interrupted,
    }),
  ];
  if (terminal !== null) observers.push(terminal.observer);
  return { value, messagesValue, terminal, observers };
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

/** Build the ordered observers, sharing one immutable failure, then one per closed host call. */
function failureObservers(failure: Failure, closedCalls: readonly string[]): RunEvent[] {
  return [
    new NodeFailed({ failure }),
    new RunFailed({ failure }),
    ...closedCalls.map((callId) => new CallClosed({ callId, reason: "failed" })),
  ];
}

/** The detached values one failure commit installs, all built before the durable append. */
interface FailureApply {
  result: RunFailed;
  observers: readonly RunEvent[];
}

/** Build independent state and callback values before the durable append. */
function materializeFailure(failure: Failure, closedCalls: readonly string[]): FailureApply {
  return { result: new RunFailed({ failure }), observers: failureObservers(failure, closedCalls) };
}

/** The detached values one ending installs, all built before the durable append. */
interface EndingApply {
  result: RunCancelled | RunFailed;
  observers: readonly RunEvent[];
}

/** Build the ending's state result and its observers: the result event, then one per closed host call. */
function materializeEnding(plan: EndingPlan): EndingApply {
  const failure = plan.failure;
  const reason = failure === null ? "cancelled" : "failed";
  const event = () => (failure === null ? new RunCancelled() : new RunFailed({ failure }));
  return {
    result: event(),
    observers: [event(), ...plan.closedCalls.map((callId) => new CallClosed({ callId, reason }))],
  };
}

/** Build one independent canonical expansion observer. */
function growthObserver(record: NodeExpandedRecord, seedWire: JsonValue): NodeExpanded {
  const [, seed] = canonicalSeed(seedWire);
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
    observers: [growthObserver(record, plan.seed)],
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
      if (state.edgeState.get(edge.id) === NodeState.UNKNOWN) dropUnknownIncoming(state, edge.to);
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
      state.unknownIncoming.delete(nodeId);
      state.pool.store.delete(nodeId);
      state.producers.delete(nodeId);
      state.pool.store.delete(messagesKey(nodeId));
      state.producers.delete(messagesKey(nodeId));
      state.depth.delete(nodeId);
      state.wiringOverrides.delete(nodeId);
      state.growthOwner.delete(nodeId);
      state.pendingCommits.delete(nodeId);
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
  // `topology.edges` are the edges of `newEdgeIds`, which the delta carries as ids only.
  for (const edge of topology.edges) addUnknownIncoming(state, edge.to);
  for (const commit of delta.admitted) state.pendingCommits.set(commit.key, commit);
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
