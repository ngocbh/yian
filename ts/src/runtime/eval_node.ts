/**
 * Node evaluation at the engine's pure read/execution boundary.
 *
 * A node is a pure function of its bound input record; the engine owns both boundaries: the read
 * (bind the inputs from the pool) and the write (the dispatcher stores `Output`). `evalNode` is
 * the append-only dispatcher's evaluation seam: it binds the record, pre-resolves reserved keys,
 * builds the trait-gated capabilities, runs the node through its assert-owning `run` template,
 * and resolves to an `EvaluatedOutcome` holding exactly `Output | Route | Grow | Pending |
 * EvaluationFailure` plus optional input capture. It publishes no observers; durable
 * observer order belongs to the dispatcher.
 *
 * An expected rejection is classified where its operation occurs: binding evaluation, supplied
 * input contracts, template assertions, node execution, or node-result validation. A stop (the
 * node's aborted signal), host capability faults, unexpected binder faults, and malformed compiled
 * or setup state reject `evalNode` instead of becoming an `EvaluationFailure`.
 *
 * Knows about:   compile/model (types), diagnostics, expr lineage, llm_clients (types), memory
 *                contracts, nodes, runtime contracts and log, skills, tools, typesys.
 * Never imports: runtime state or engine, the run boundary, compose, or the CLI.
 */

import type { CompiledFlow, Flow } from "../compile/model.ts";
import { Failure, type FailureCode, type OriginRef, reprText } from "../diagnostics.ts";
import { ResolvedValue, type SourceRef, ValueRoute } from "../expr/lineage.ts";
import type { ModelAccess } from "../llm_clients/factory.ts";
import {
  type JsonObject,
  MemoryContext,
  MemoryRecallError,
  MemoryScope,
  MemorySection,
  RecallContext,
} from "../memory/contracts.ts";
import {
  AssertionStageError,
  BindingStageError,
  CapabilityFault,
  DeferredBindingFault,
  Grow,
  markerMessage,
  Node,
  NodeBodyError,
  type NodeCaps,
  type NodeResult,
  Output,
  Pending,
  Route,
  type SteerInbox,
  tracebackText,
} from "../nodes/base.ts";
import {
  applyDefaults,
  BindingCapture,
  BindingCollector,
  BindingError,
  bindParams,
  type ReservedBinder,
  resolveSource,
  type SourceResolver,
} from "../nodes/binding.ts";
import { isExactly, NodeSteps } from "../nodes/contracts.ts";
import type { Scheduler } from "../platform.ts";
import type { Skill } from "../skills/catalog.ts";
import { ToolSet } from "../tools/registry.ts";
import type { JsonObject as WireObject, JsonValue } from "../typesys/json.ts";
import { VariablePool } from "../typesys/pool.ts";
import { isRecord, jsTypeName, setField, type TypedValue, type Value, type ValueRecord } from "../typesys/values.ts";
import { BackwardCapture } from "./contracts.ts";
import { canonicalSeed } from "./log.ts";
import { type ReboundSource, resolvePoolPath, SYSTEM_PREFIX } from "./trace.ts";

/**
 * One raw node-evaluation failure for durable dispatcher preparation. Frozen.
 *
 * `failure` is the sole semantic and presentation authority.
 */
export class EvaluationFailure {
  readonly failure: Failure;

  constructor(failure: Failure) {
    this.failure = failure;
    Object.freeze(this);
  }
}

/**
 * @internal Build one failure of `node` from a thrown value: its message, the `Error`'s `name`
 * (else the value's type name) as the cause type, and the `Error`'s stack followed by its string
 * `notes`, one per line, as the traceback. The engine uses it for its contract failures.
 *
 * @param origin The failing operation's authored origin; `null` falls back to the node's origin.
 */
export function evaluationFailure(
  node: Node,
  code: FailureCode,
  error: unknown,
  origin: OriginRef | null = null,
): EvaluationFailure {
  return new EvaluationFailure(
    new Failure({
      code,
      message: markerMessage(error),
      nodeId: node.id,
      trace: node.failureTrace(origin),
      causeType: error instanceof Error ? error.name : jsTypeName(error),
      traceback: tracebackText(error),
    }),
  );
}

/** Return `error` as a capability fault: a `CapabilityFault` as it is, anything else wrapped. */
function capabilityFault(error: unknown): CapabilityFault {
  return error instanceof CapabilityFault ? error : new CapabilityFault(error);
}

/**
 * Carry a host observer fault out of a node body as a capability fault, so a failing
 * `onStream`/`onEvent` stays a host fault, never a node failure. `onFault` sees each fault before
 * it is thrown. Once `isClosed()` is true the sink drops what it gets.
 */
function observerSink<T>(
  sink: (item: T) => unknown,
  onFault: (fault: CapabilityFault) => void,
  isClosed: () => boolean,
): (item: T) => void {
  return (item) => {
    if (isClosed()) return;
    try {
      sink(item);
    } catch (error) {
      const fault = capabilityFault(error);
      onFault(fault);
      throw fault;
    }
  };
}

/**
 * Wrap the engine's steer inbox opener for one evaluation: once the evaluation ended, opening and
 * each `queued`, `read` or `interruptedBy` throw a `CapabilityFault`; a `CapabilityFault` the engine throws reaches
 * `onFault` before it is rethrown, so a node that catches it can't hide it.
 */
function steerInboxCapability(
  open: (inbox: string) => SteerInbox,
  onFault: (fault: CapabilityFault) => void,
  isClosed: () => boolean,
): (inbox: string) => SteerInbox {
  const guarded = <T>(run: () => T): T => {
    if (isClosed()) throw new CapabilityFault(new Error("steer inbox capability is closed"));
    try {
      return run();
    } catch (error) {
      if (error instanceof CapabilityFault) onFault(error);
      throw error;
    }
  };
  return (inbox) => {
    const opened = guarded(() => open(inbox));
    return {
      inbox: opened.inbox,
      queued: () => guarded(() => opened.queued()),
      read: () => guarded(() => opened.read()),
      interrupt: opened.interrupt,
      interruptedBy: () => guarded(() => opened.interruptedBy()),
    };
  };
}

/**
 * Return a host model access after checking its shape, or `null`.
 *
 * @throws TypeError when `models` isn't `null` and lacks a function `env` and `createModel`.
 */
export function requireModels(models: unknown): ModelAccess | null {
  const access = models as Partial<ModelAccess> | null | undefined;
  if (models !== null && !(typeof access?.env === "function" && typeof access.createModel === "function")) {
    throw new TypeError("models must provide env(name) and createModel(config, options), or be null");
  }
  return models as ModelAccess | null;
}

/** The closed raw worker result the append-only dispatcher consumes. */
export type WorkerOutcome = Output | Route | Grow | Pending | EvaluationFailure;

/**
 * One raw worker outcome with its optional binding and backward capture. Frozen.
 *
 * Binding values and routes travel with the common inventories and static material, as separate
 * lower binding and runtime carriers. Neither is accepted history until the dispatcher records
 * this outcome.
 */
export class EvaluatedOutcome {
  readonly outcome: WorkerOutcome;
  readonly bindings: BindingCapture | null;
  readonly backwardCapture: BackwardCapture | null;

  /**
   * Require one closed raw outcome and exact optional captures; the backward capture is checked
   * again by rebuilding it.
   *
   * @throws TypeError for an outcome outside `WorkerOutcome`, or a capture that isn't exactly its
   *   class.
   * @throws RangeError when only one of `bindings` and `backwardCapture` is present, or as
   *   `BackwardCapture` refuses the capture.
   */
  constructor(
    outcome: WorkerOutcome,
    bindings: BindingCapture | null = null,
    backwardCapture: BackwardCapture | null = null,
  ) {
    if (
      !(
        outcome instanceof Output ||
        outcome instanceof Route ||
        outcome instanceof Grow ||
        outcome instanceof Pending ||
        outcome instanceof EvaluationFailure
      )
    ) {
      throw new TypeError("outcome must be a WorkerOutcome");
    }
    if (bindings !== null && !isExactly(bindings, BindingCapture)) {
      throw new TypeError("bindings must be a BindingCapture or null");
    }
    if (backwardCapture !== null) {
      if (!isExactly(backwardCapture, BackwardCapture)) {
        throw new TypeError("backward_capture must be a BackwardCapture or null");
      }
      const { inputNames, variableNames, scope } = backwardCapture;
      new BackwardCapture({ inputNames, variableNames, scope });
    }
    if ((bindings === null) !== (backwardCapture === null)) {
      throw new RangeError("binding capture and backward invocation metadata must be retained together");
    }
    this.outcome = outcome;
    this.bindings = bindings;
    this.backwardCapture = backwardCapture;
    Object.freeze(this);
  }
}

/**
 * Host-injected admission of recalled notes: selects one memory context for a scoped node's
 * `recall`, under the node's `signal`.
 */
export type MemoryContextOpener = (
  scope: MemoryScope,
  recall: RecallContext,
  options: { signal: AbortSignal },
) => Promise<MemoryContext>;

/**
 * Check one opener's sections are an ordered subsequence of the scope's memories, each with text.
 *
 * @throws TypeError when an item isn't exactly a `MemorySection`.
 * @throws RangeError for an empty text, or a section outside the rest of the scope's memories.
 */
function validateMemorySections(scope: MemoryScope, sections: readonly MemorySection[]): void {
  let next = 0;
  for (const section of sections) {
    if (!isExactly(section, MemorySection)) throw new TypeError("memory recall items must be MemorySection values");
    if (section.text === "") throw new RangeError("memory recall sections must contain non-empty text");
    const index = scope.memories.indexOf(section.memory, next);
    if (index < 0) throw new RangeError("memory recall sections must follow the consumer memory stack");
    next = index + 1;
  }
}

/** A lazy single-use memory context capability bound to one evaluation's lifetime. */
interface MemoryContextCapability {
  /** The node's `openMemoryContext`. */
  readonly open: NonNullable<NodeCaps["openMemoryContext"]>;
  /** End the lifetime: later opens reject. */
  close(): void;
}

/**
 * Bind a memory context capability to one evaluation.
 *
 * @param options.scope The node's memory scope; `null` or no memories opens an empty context.
 * @param options.opener The host's opener, present for a scoped node.
 * @param options.canonicalInputs The visible bound inputs for the recall context; `null` when
 *   canonicalization refused them.
 * @param options.onFault Sees each capability fault before the open rejects with it.
 * @param options.pending Receives the open's promise, for the evaluator to await.
 */
function memoryContextCapability(options: {
  scope: MemoryScope | null;
  opener: MemoryContextOpener | null;
  signal: AbortSignal;
  canonicalInputs: JsonObject | null;
  onFault: (fault: CapabilityFault) => void;
  pending: Promise<unknown>[];
}): MemoryContextCapability {
  const { scope, opener, signal, canonicalInputs, onFault, pending } = options;
  let closed = false;
  let used = false;

  /**
   * Open the context once.
   *
   * @throws MemoryRecallError for invalid recall inputs, request or opener result, or as the opener
   *   throws it.
   * @throws CapabilityFault carrying a repeated open or any other opener fault.
   */
  const enter = async (baseSystemPrompt: string, renderedPrompt: string): Promise<MemoryContext> => {
    try {
      if (used) throw new Error("memory context capability was already entered");
      used = true;
      if (scope === null || scope.memories.length === 0) return new MemoryContext();
      if (canonicalInputs === null) throw new MemoryRecallError("memory recall context is invalid");
      let recall: RecallContext;
      try {
        recall = new RecallContext(baseSystemPrompt, renderedPrompt, canonicalInputs, scope.staticPath);
      } catch (error) {
        if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
        throw new MemoryRecallError("memory recall context is invalid", { cause: error });
      }
      // The evaluator refuses a scoped node without an opener before the node runs.
      const context: unknown = await opener!(scope, recall, { signal });
      try {
        if (!isExactly(context, MemoryContext)) throw new TypeError("memory context must be an exact MemoryContext");
        validateMemorySections(scope, context.sections);
      } catch (error) {
        if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
        throw new MemoryRecallError("memory recall returned invalid context", { cause: error });
      }
      return context;
    } catch (error) {
      if (error instanceof MemoryRecallError) throw error;
      const fault = capabilityFault(error);
      onFault(fault);
      throw fault;
    }
  };

  const open = (baseSystemPrompt: string, renderedPrompt: string) => {
    const opened: Promise<MemoryContext> = closed
      ? Promise.reject(new Error("memory context capability is closed"))
      : enter(baseSystemPrompt, renderedPrompt);
    // A node may drop the promise: its rejection must not go unhandled, and the evaluator awaits
    // it, so a fault still reaches `onFault`.
    opened.catch(() => {});
    if (!closed) pending.push(opened);
    return opened;
  };

  return {
    open,
    close: () => {
      closed = true;
    },
  };
}

/** The optional inputs of one `evalNode` call; each defaults to `null` or `false`. */
export interface EvalNodeOptions {
  /**
   * The per-run wiring for this node in place of the compiled `flow.wiring.get(node.id)`: the
   * engine narrows top-level START's `${input.X}` sources to the passed run arguments, so an
   * omitted input binds absent and START fills its declared default, without changing the shared
   * compiled flow.
   */
  wiringOverride?: JsonObject | null;
  /** The node's own abort signal, given to it as `caps.signal`; a fresh un-aborted one when omitted. */
  signal?: AbortSignal | null;
  /** The clock and cancellable sleep node waits run on; every node gets it as `caps.scheduler`. */
  scheduler?: Scheduler | null;
  /** A draw in [0, 1) for backoff jitter; every node gets it as `caps.random`. */
  random?: (() => number) | null;
  /** The run's folder, where the run's files go; every node gets it as `caps.runDir`. */
  runDir?: string | null;
  /**
   * The transient stream sink a node declaring `needsLiveEvents` receives; closed, dropping what
   * it gets, once the outcome is decided.
   */
  emitStream?: NodeCaps["emitStream"] | null;
  /** The transient model and tool call observer sink, received and closed as `emitStream` is. */
  emitEvent?: NodeCaps["emitEvent"] | null;
  /** Whether to retain the bound input values, their routes and the backward capture. */
  captureBindings?: boolean;
  /**
   * The memory context source a node declaring `needsMemoryContext` reaches through
   * `caps.openMemoryContext`; required when the node's scope has memories.
   */
  memoryContextOpener?: MemoryContextOpener | null;
  /** The run's host-supplied skills, delivered only to a node declaring `needsSkills`. */
  skills?: ReadonlyMap<string, Skill> | null;
  /** The paired value and source lookup used while capturing bindings. */
  sourceResolver?: SourceResolver | null;
  /** The run's tool set, delivered only to a node declaring `needsTools`. */
  tools?: ToolSet | null;
  /** The run's environment and model builder, delivered only to a node declaring `needsModels`. */
  models?: ModelAccess | null;
  /**
   * This start's steps from the engine, delivered only to a node declaring `needsSteps` as a
   * `NodeSteps` over the same `starts`, `prior`, `calls` and `maxEventBytes`; a `CapabilityFault`
   * its `record` throws is a capability fault of this evaluation. The node's `record` closes when the outcome is decided.
   */
  steps?: NodeSteps | null;
  /**
   * Opens the node's steer inbox, from the engine, delivered only to a node declaring
   * `readsSteerInbox` as `caps.openSteerInbox`; a `CapabilityFault` it or a read throws is a
   * capability fault of this evaluation. It closes when the outcome is decided.
   */
  steerInbox?: ((inbox: string) => SteerInbox) | null;
  /**
   * Sees the original error of the evaluation's first capability fault when it is raised, before
   * the node settles, so a host can end its run even when the node catches the fault.
   */
  onFault?: ((original: unknown) => void) | null;
}

/**
 * Evaluate one node into a wrapped worker result.
 *
 * Binding precedes node-owned setup validation, so an unreachable MAP with a bad `over` reports
 * that binding rejection before its missing compiled child. An expected rejection resolves to an
 * `EvaluationFailure`; a stop, host fault or malformed setup rejects.
 *
 * @param node The node to evaluate.
 * @param flow The flow whose wiring holds the node's sources; `null` is empty wiring.
 * @param pool The pool the sources read.
 * @param options The capabilities and captures; see `EvalNodeOptions`.
 * @returns The outcome, with captures when asked for.
 * @throws TypeError (as a rejection) for a malformed option.
 * @throws The original error of the first capability fault (a failing stream or event sink,
 *   a repeated memory context open or a failing opener, a
 *   `CapabilityFault` from `steps.record` or the steer inbox), whatever the node then returned or
 *   threw.
 * @throws TypeError when a node declaring `needsMemoryContext` has a `memoryScope` that isn't
 *   exactly a `MemoryScope`; Error when a
 *   memory context node's scope has memories and no opener was injected.
 * @throws TypeError, RangeError or the hook's own error when capturing bindings and the input
 *   inventory fails; the body hasn't run.
 * @throws The signal's reason when the signal is already aborted, before binding, or when the node
 *   rejects after it aborted.
 * @throws RangeError when a node reading its steps fails to bind its reserved keys after an earlier
 *   start recorded steps.
 * @throws The original error of an unexpected per-item binder fault, or malformed node setup (a
 *   missing reserved wiring key, malformed `params`, no `forward`).
 */
export async function evalNode(
  node: Node,
  flow: CompiledFlow | null,
  pool: VariablePool,
  options: EvalNodeOptions = {},
): Promise<EvaluatedOutcome> {
  const {
    wiringOverride = null,
    emitStream = null,
    emitEvent = null,
    captureBindings = false,
    memoryContextOpener = null,
    skills = null,
    sourceResolver = null,
    tools = null,
    models = null,
    steps = null,
    steerInbox = null,
    onFault = null,
    scheduler = null,
    random = null,
    runDir = null,
  } = options;
  requireModels(models);
  if (typeof captureBindings !== "boolean") throw new TypeError("capture_bindings must be a bool");
  if (emitStream !== null && typeof emitStream !== "function") {
    throw new TypeError("emit_stream must be callable or null");
  }
  if (emitEvent !== null && typeof emitEvent !== "function") {
    throw new TypeError("emit_event must be callable or null");
  }
  if (sourceResolver !== null && typeof sourceResolver !== "function") {
    throw new TypeError("source_resolver must be callable or null");
  }
  if (memoryContextOpener !== null && typeof memoryContextOpener !== "function") {
    throw new TypeError("memory_context_opener must be callable or null");
  }
  if (tools !== null && !isExactly(tools, ToolSet)) throw new TypeError("tools must be a ToolSet or null");
  if (skills !== null && !(skills instanceof Map)) throw new TypeError("skills must be a map or null");
  if (steps !== null && !isExactly(steps, NodeSteps)) throw new TypeError("steps must be a NodeSteps or null");
  if (steerInbox !== null && typeof steerInbox !== "function") throw new TypeError("steer_inbox must be callable or null");
  if (onFault !== null && typeof onFault !== "function") throw new TypeError("on_fault must be callable or null");
  // Read once, so a node replacing its scope while it runs doesn't change what it recalls.
  const memoryScope = node.needsMemoryContext === true ? node.memoryScope : null;
  if (memoryScope !== null && !isExactly(memoryScope, MemoryScope)) {
    throw new TypeError("node memory_scope must be an exact MemoryScope or null");
  }
  if (options.signal != null && !(options.signal instanceof AbortSignal)) {
    throw new TypeError("signal must be an AbortSignal or null");
  }
  if (
    scheduler !== null &&
    (typeof scheduler !== "object" || typeof scheduler.now !== "function" || typeof scheduler.sleep !== "function")
  ) {
    throw new TypeError("scheduler must be a Scheduler or null");
  }
  if (random !== null && typeof random !== "function") throw new TypeError("random must be a function or null");
  if (runDir !== null && typeof runDir !== "string") throw new TypeError("run_dir must be a string or null");
  const signal = options.signal ?? new AbortController().signal;
  signal.throwIfAborted();
  const collector = captureBindings ? new BindingCollector() : null;
  const backwardCapture = captureBindings ? nodeBackwardCapture(node) : null;

  /** Snapshot the completed inputs before the body can change them. */
  const captureInputs = (inputs: Record<string, unknown>): void => {
    collector!.finish(inputs as ValueRecord);
  };

  const outcome = await evaluateNode(node, flow, pool, {
    wiringOverride,
    signal,
    scheduler,
    random,
    runDir,
    collector,
    captureInputs: captureBindings ? captureInputs : null,
    memoryScope,
    memoryContextOpener,
    sourceResolver,
    emitStream,
    emitEvent,
    skills,
    tools,
    models,
    steps,
    steerInbox,
    onFault,
  });
  return new EvaluatedOutcome(outcome, collector?.freeze() ?? null, backwardCapture);
}

/** Bind and run one node, resolving to one normalized result. */
async function evaluateNode(
  node: Node,
  flow: CompiledFlow | null,
  pool: VariablePool,
  context: {
    wiringOverride: JsonObject | null;
    signal: AbortSignal;
    scheduler: Scheduler | null;
    random: (() => number) | null;
    runDir: string | null;
    /** Retains the bound values and routes, when capturing bindings. */
    collector: BindingCollector | null;
    /** Receives the completed input record before the body runs, when capturing bindings. */
    captureInputs: ((inputs: Record<string, unknown>) => void) | null;
    /** The memory scope of a node declaring `needsMemoryContext`, else `null`. */
    memoryScope: MemoryScope | null;
    memoryContextOpener: MemoryContextOpener | null;
    sourceResolver: SourceResolver | null;
    emitStream: NonNullable<NodeCaps["emitStream"]> | null;
    emitEvent: NonNullable<NodeCaps["emitEvent"]> | null;
    skills: ReadonlyMap<string, Skill> | null;
    tools: ToolSet | null;
    models: ModelAccess | null;
    steps: NodeSteps | null;
    steerInbox: ((inbox: string) => SteerInbox) | null;
    /** Sees the original of the first capability fault when it is raised. */
    onFault: ((original: unknown) => void) | null;
  },
): Promise<WorkerOutcome> {
  const { wiringOverride, signal, collector, captureInputs, sourceResolver } = context;
  const { memoryScope, memoryContextOpener } = context;
  const { emitStream, emitEvent, skills, tools, models, steps, steerInbox, scheduler, random, runDir } = context;
  // The flow owns every source. A missing trait-declared reserved key is a malformed compiled
  // graph and deliberately escapes the expected binder catch.
  const nodeWiring: ReadonlyMap<string, unknown> =
    wiringOverride !== null ? new Map(Object.entries(wiringOverride)) : (flow?.wiring.get(node.id) ?? new Map());
  const usesBaseTemplate = node.run === Node.prototype.run;
  const reservedSources = new Set<string>();
  const reservedBinder: ReservedBinder | null =
    collector === null
      ? null
      : (name, source, normalize) => {
          // Retain the route of the value the node delivers, after its normalization.
          const value = normalize(resolveSource(source, pool, { collector, name, sourceResolver }));
          reservedSources.add(name);
          return value;
        };

  let record: ValueRecord;
  try {
    record = node.bindsPerItem
      ? {}
      : bindParams(node.params ?? [], nodeWiring, pool, {
          collector,
          sourceResolver,
          types: node.types,
          labels: node.bindingLabels,
          finishCapture: false,
        });
    let reserved: Record<string, unknown>;
    try {
      reserved = node.bindReserved(nodeWiring, pool, { bind: reservedBinder });
    } catch (error) {
      if (error instanceof BindingError && steps !== null && node.needsSteps === true && steps.prior.length > 0) {
        // A start records steps only after binding, and resume restores the pool, so the code
        // that resolves these keys changed since then.
        throw new RangeError(
          `${error.message}; an earlier start of this node resolved it, so resume the run with the flow and ` +
            "yian version that started it",
        );
      }
      throw error;
    }
    for (const name of Object.keys(reserved)) {
      const value = reserved[name] as Value;
      setField(record, name, value);
      // A hook that materialized its value without the binder has no evaluated source.
      if (collector !== null && !reservedSources.has(name)) {
        collector.recordSource(name, ValueRoute.literal(value), false);
      }
    }
  } catch (error) {
    if (!(error instanceof BindingError)) throw error;
    collector?.invalidate();
    return evaluationFailure(node, error.failureCode, error, error.origin);
  }

  if (captureInputs !== null && !usesBaseTemplate) captureInputs(record);

  const scoped = memoryScope !== null && memoryScope.memories.length > 0;
  if (scoped && memoryContextOpener === null) {
    throw new Error(`scoped node ${reprText(node.id)} requires an injected memory context callback`);
  }
  // The visible bound inputs, for the recall context.
  let canonicalInputs: JsonObject | null = null;
  if (scoped) {
    const hidden = node.modelHiddenInputs();
    const visible = Object.fromEntries(Object.entries(record).filter(([name]) => !hidden.has(name)));
    try {
      const value = canonicalSeed(visible)[0];
      if (!isRecord(value)) throw new TypeError("bound node inputs did not canonicalize to an exact record");
      canonicalInputs = value as JsonObject;
    } catch (error) {
      // The recall capability owns the rejection of its query context.
      if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
    }
  }

  // The first fault any capability raised. A node can catch the fault it was thrown, so the
  // evaluator keeps it to throw when the node settles.
  const faults: { first: CapabilityFault | null } = { first: null };
  const onFault = (fault: CapabilityFault): void => {
    if (faults.first !== null) return;
    faults.first = fault;
    context.onFault?.(fault.original);
  };
  // The capability promises handed to the node. A node may settle without awaiting one, so the
  // evaluator awaits their settlement before it settles, and a late fault is still the first.
  const pending: Promise<unknown>[] = [];
  const memory = memoryContextCapability({
    scope: memoryScope,
    opener: memoryContextOpener,
    signal,
    canonicalInputs,
    onFault,
    pending,
  });

  // Every node gets its signal and the given scheduler, random draw and run folder; the rest are
  // node-owned and trait-gated. The deferred per-item binder throws the same binder-owned marker from inside the
  // body, and the classification below keeps it.
  const caps: { -readonly [K in keyof NodeCaps]: NodeCaps[K] } = { signal };
  if (scheduler !== null) caps.scheduler = scheduler;
  if (random !== null) caps.random = random;
  if (runDir !== null) caps.runDir = runDir;
  if (node.bindsPerItem) {
    const params = node.params ?? [];
    caps.bindItem = (element, index) => {
      try {
        let item: Value | ResolvedValue = element;
        let itemCollector: BindingCollector | null = null;
        if (collector !== null) {
          itemCollector = collector.forItem(index);
          const key = node.itemSourceKey;
          item = new ResolvedValue(element, true, key !== null ? collector.routeFor(key).project([index]) : null);
        }
        return bindParams(params, nodeWiring, pool, {
          item,
          collector: itemCollector,
          sourceResolver,
          types: node.types,
          labels: node.bindingLabels,
        });
      } catch (error) {
        if (error instanceof BindingError) {
          collector?.invalidate();
          throw new BindingStageError(error);
        }
        throw new DeferredBindingFault(error);
      }
    };
  }
  if (node.needsMemoryContext === true) caps.openMemoryContext = memory.open;
  if (node.needsTools === true) caps.tools = tools ?? new ToolSet();
  // Closed once the node settles, so a step, event or steer read the node's dropped code makes after
  // the outcome is decided is refused or dropped rather than reaching the run with no evaluation
  // left to fail.
  let closed = false;
  const isClosed = (): boolean => closed;
  if (node.needsLiveEvents === true) {
    if (emitStream !== null) caps.emitStream = observerSink(emitStream, onFault, isClosed);
    if (emitEvent !== null) caps.emitEvent = observerSink(emitEvent, onFault, isClosed);
  }
  if (skills !== null && node.needsSkills === true) caps.skills = skills;
  if (models !== null && node.needsModels === true) caps.models = models;
  if (steps !== null && node.needsSteps === true) {
    // The engine's recorder throws its faults as capability faults; a node can catch one, so the
    // evaluator keeps it as it keeps the other capabilities' faults.
    const recordStep: NodeSteps["record"] = (step, options) => {
      if (closed) throw new CapabilityFault(new Error("node steps capability is closed"));
      try {
        return steps.record(step, options);
      } catch (error) {
        if (error instanceof CapabilityFault) onFault(error);
        throw error;
      }
    };
    caps.steps = new NodeSteps(steps.starts, steps.prior, recordStep, steps.calls, steps.maxEventBytes, steps.durable, () =>
      steps.currentCalls(),
    );
  }
  if (steerInbox !== null && node.readsSteerInbox === true) {
    caps.openSteerInbox = steerInboxCapability(steerInbox, onFault, isClosed);
  }

  let result: unknown;
  let rejection: { error: unknown } | null = null;
  try {
    result = usesBaseTemplate
      ? await node.runForEvaluator(record, caps, captureInputs)
      : await node.run(record, caps);
  } catch (error) {
    rejection = { error };
  }
  // The capabilities stay open while the evaluator waits, so a node's dropped code may start another
  // capability call meanwhile; one wait would miss it, and a late open's fault would never be
  // checked.
  for (let settled = 0; settled < pending.length; ) {
    const batch = pending.slice(settled);
    settled = pending.length;
    await Promise.allSettled(batch);
  }
  closed = true;
  memory.close();

  // A host capability fault, then a stop, is control flow outside the node-result sum, whatever
  // the node returned or rejected with.
  if (faults.first !== null) throw faults.first.original;
  if (rejection !== null && signal.aborted) throw signal.reason;
  if (rejection !== null) {
    const { error } = rejection;
    // A template default that fails to bind leaves no usable input record.
    if (usesBaseTemplate && error instanceof BindingStageError) collector?.invalidate();
    return stageFailure(node, error, usesBaseTemplate);
  }
  const contractError = resultContractError(node, result);
  if (contractError !== null) return evaluationFailure(node, "contract.node_result", contractError);
  return result as NodeResult;
}

/** Return the error of a result outside `NodeResult` or a `Grow` from a non-spawner, else `null`. */
function resultContractError(node: Node, result: unknown): Error | null {
  if (!(result instanceof Output || result instanceof Route || result instanceof Grow || result instanceof Pending)) {
    return new Error(`node ${reprText(node.id)} run() returned ${jsTypeName(result)}, not a NodeResult`);
  }
  if (result instanceof Grow && !node.isSpawner) {
    // The kind is diagnostic text only; the spawner trait owns this gate.
    return new Error(
      `node ${reprText(node.id)} (kind ${node.kind}) returned a Grow but is not a spawner (CALL/MAP/LOOP); ` +
        "only spawner kinds may grow the graph",
    );
  }
  return null;
}

/**
 * Classify one rejection of a node's run by the stage that threw it.
 *
 * @throws The original of a capability or deferred binder fault, and, for the base template, any
 *   error outside its stages (malformed setup).
 */
function stageFailure(node: Node, error: unknown, usesBaseTemplate: boolean): EvaluationFailure {
  if (error instanceof DeferredBindingFault || error instanceof CapabilityFault) throw error.original;
  if (error instanceof BindingStageError) {
    const { original } = error;
    // A custom run is one undivided node-body stage.
    if (!usesBaseTemplate) return evaluationFailure(node, "node.execute", original, node.exceptionOrigin(original));
    return evaluationFailure(node, original.failureCode, original, original.origin);
  }
  if (error instanceof AssertionStageError) {
    const { original } = error;
    if (!usesBaseTemplate) return evaluationFailure(node, "node.execute", original, node.exceptionOrigin(original));
    return evaluationFailure(node, original.failureCode, original, original.assertion?.origin ?? null);
  }
  if (error instanceof NodeBodyError) {
    const { original } = error;
    return evaluationFailure(node, node.exceptionCode(original), original, node.exceptionOrigin(original));
  }
  if (usesBaseTemplate) throw error;
  return evaluationFailure(node, node.exceptionCode(error), error, node.exceptionOrigin(error));
}

/**
 * The backward material `node` gives an evaluation: its backward input names, its variable names
 * and its execution scope.
 */
export function nodeBackwardCapture(node: Node): BackwardCapture {
  return new BackwardCapture({
    inputNames: node.backwardInputNames(),
    variableNames: node.declaredVariables().map((parameter) => parameter.name),
    scope: node.executionScope,
  });
}

/**
 * Rebuild the node data backward reads about one evaluation of `node`, which its record leaves
 * out: none for a binding that didn't finish (`inputs` `null`); else the node's static material
 * (`captureBackwardContext`), and, when the node renders (`renderCapture`) and its pre-asserts
 * hold for `inputs`, so its `forward` ran, that render with the model config the invocation used
 * as `effective_llm_config`: the config bound with its inputs (`boundLlmConfig`), else the run's
 * entry for its static path.
 *
 * @param inputs The bound inputs, as binding the node again gives them.
 * @param effectiveLlmByStaticPath The run's recorded model config by static path.
 * @returns Canonical JSON.
 * @throws TypeError when the static material isn't a record, TypeCheckError when it or the render
 *   isn't JSON, and the hooks' own errors.
 * @throws RangeError when the node renders and the run's map has no entry for its static path, or
 *   its render holds `effective_llm_config` or a key of its static material.
 */
export function rebuildNodeData(
  node: Node,
  inputs: Readonly<Record<string, unknown>> | null,
  effectiveLlmByStaticPath: Readonly<Record<string, unknown>>,
): WireObject {
  if (inputs === null) return {};
  const captured: unknown = node.captureBackwardContext();
  if (!isRecord(captured)) throw new TypeError("capture_backward_context must return a record");
  const data: Record<string, unknown> = { ...captured };
  const render = node.renderCapture(inputs);
  if (render !== null && node.preAssertsHold(inputs)) {
    if (Object.hasOwn(render, "effective_llm_config")) {
      throw new RangeError("effective_llm_config belongs to the run's model config");
    }
    const replaced = Object.keys(render).filter((key) => Object.hasOwn(data, key));
    if (replaced.length > 0) throw new RangeError(`the render cannot replace static node data: ${replaced.join(", ")}`);
    Object.assign(data, render, { effective_llm_config: invocationLlmConfig(node, inputs, effectiveLlmByStaticPath) });
  }
  return canonicalSeed(data)[1] as WireObject;
}

/**
 * The model config one invocation of `node` used: the config bound with `inputs`, else the run's
 * entry for the node's static path; `null` for a node with no scope.
 *
 * @throws RangeError when the run's map has no entry for the static path.
 */
function invocationLlmConfig(
  node: Node,
  inputs: Readonly<Record<string, unknown>>,
  effectiveLlmByStaticPath: Readonly<Record<string, unknown>>,
): JsonValue | null {
  const scope = node.executionScope;
  if (scope === null) return null;
  const bound = node.boundLlmConfig(inputs);
  if (bound !== null) return canonicalSeed(bound)[1];
  const key = scope.staticPath.join("/");
  if (!Object.hasOwn(effectiveLlmByStaticPath, key)) {
    throw new RangeError(`recorded effective model configuration is missing static path ${reprText(key)}`);
  }
  return canonicalSeed(effectiveLlmByStaticPath[key])[1];
}

/** The bound inputs and capabilities `rebind` and `rebuildDerived` share. */
interface Rebound {
  inputs: Record<string, unknown>;
  caps: NodeCaps;
}

/**
 * Bind `node` from `sources` as the evaluator does, with defaults filled; with a `collector`,
 * collect the route and bound inputs into it too. `reserved`, when given, are the node's reserved values, used in
 * place of its `bindReserved`.
 */
function bindFrom(
  node: Node,
  flow: Flow,
  sources: ReadonlyMap<string, number>,
  resolve: (name: string, seq: number) => ReboundSource,
  wiringOverride: JsonObject | null,
  collector: BindingCollector | null,
  reserved: Readonly<Record<string, unknown>> | null = null,
): Rebound {
  const pool = new VariablePool(flow.startId);
  const entries = new Map<string, ReboundSource>();
  for (const [name, seq] of sources) {
    const entry = resolve(name, seq);
    entries.set(name, entry);
    if (name.startsWith(SYSTEM_PREFIX)) pool.system.set(name.slice(SYSTEM_PREFIX.length), entry.value);
    else pool.store.set(name, entry.value);
  }
  const sourceResolver: SourceResolver = (names) => resolvePoolPath(names, pool, (name) => entries.get(name)!.source);
  return bindPool(node, flow, pool, sourceResolver, wiringOverride, collector, reserved);
}

/**
 * Bind `node` from the run's `pool` as the evaluator binds it when the node starts, with defaults
 * filled, capturing nothing.
 *
 * @param wiringOverride The wiring the run gave the node in place of the flow's (a growth's), or
 *   `null`.
 * @returns The bound inputs.
 * @throws BindingError when a source doesn't evaluate, and the node's `bindReserved` errors.
 */
export function bindStarted(
  node: Node,
  flow: Flow,
  pool: VariablePool,
  wiringOverride: JsonObject | null,
): Record<string, unknown> {
  return bindPool(node, flow, pool, null, wiringOverride, null, null).inputs;
}

/** `bindFrom`'s binding over a built `pool`. */
function bindPool(
  node: Node,
  flow: Flow,
  pool: VariablePool,
  sourceResolver: SourceResolver | null,
  wiringOverride: JsonObject | null,
  collector: BindingCollector | null,
  reserved: Readonly<Record<string, unknown>> | null,
): Rebound {
  const nodeWiring: ReadonlyMap<string, unknown> =
    wiringOverride !== null ? new Map(Object.entries(wiringOverride)) : (flow.wiring.get(node.id) ?? new Map());
  const params = node.params ?? [];
  const bindOptions = { sourceResolver, types: node.types, labels: node.bindingLabels };
  const record: Record<string, unknown> = node.bindsPerItem
    ? {}
    : bindParams(params, nodeWiring, pool, { ...bindOptions, collector, finishCapture: false });
  const reservedSources = new Set<string>();
  const bound =
    reserved ??
    node.bindReserved(nodeWiring, pool, {
      bind:
        collector === null
          ? null
          : (name, source, normalize) => {
              reservedSources.add(name);
              return normalize(resolveSource(source, pool, { collector, name, sourceResolver }));
            },
    });
  for (const name of Object.keys(bound)) {
    const value = bound[name] as Value;
    setField(record as ValueRecord, name, value);
    if (collector !== null && !reservedSources.has(name)) collector.recordSource(name, ValueRoute.literal(value), false);
  }
  const inputs = applyDefaults(params, record);
  collector?.finish(inputs as ValueRecord);
  const caps: { -readonly [K in keyof NodeCaps]: NodeCaps[K] } = { signal: new AbortController().signal };
  if (node.bindsPerItem) {
    caps.bindItem = (element, index) => {
      if (collector === null) return bindParams(params, nodeWiring, pool, { ...bindOptions, item: element });
      const key = node.itemSourceKey;
      const item = new ResolvedValue(element, true, key !== null ? collector.routeFor(key).project([index]) : null);
      return bindParams(params, nodeWiring, pool, { ...bindOptions, item, collector: collector.forItem(index) });
    };
  }
  return { inputs, caps };
}

/**
 * Bind a node's params from the values `resolve` gives for its sources, plus its wiring (or
 * `wiringOverride`), collecting the route and bound inputs in the same pass, as the evaluator does.
 *
 * `resolve(name, seq)` returns the value pool name `name` held when the `pool_commit` at `seq` wrote
 * it, and the producer the route names. Inside the run reducer it reads the run's pool and checks
 * the producer index, which is sound because records apply in log order: the pool when a derived
 * commit is accepted holds exactly what its binding read.
 *
 * `options.reserved` gives the node's reserved values instead of its `bindReserved`: a hook that
 * resolves its value without the binder records it whole, as a literal route, and reads names
 * the route doesn't name.
 *
 * A node that binds per item (MAP) binds its items in its pure `forward`, as the evaluator ran it,
 * so the capture holds them. A binding that doesn't finish, its items' included, gives the
 * `BindingError` as `failure` and, as the evaluator keeps it, the capture of the routes collected
 * before it failed, with no bound inputs.
 *
 * @throws whatever `resolve` or a per-item node's `forward` throws besides a `BindingError`.
 */
export function rebind(
  node: Node,
  flow: Flow,
  sources: ReadonlyMap<string, number>,
  resolve: (name: string, seq: number) => ReboundSource,
  wiringOverride: JsonObject | null,
  options: { reserved?: Readonly<Record<string, unknown>> | null } = {},
): { capture: BindingCapture; failure: BindingError | null } {
  const collector = new BindingCollector();
  try {
    const { inputs, caps } = bindFrom(node, flow, sources, resolve, wiringOverride, collector, options.reserved ?? null);
    if (node.bindsPerItem) node.forward(inputs, caps);
    return { capture: collector.freeze(), failure: null };
  } catch (error) {
    if (!(error instanceof BindingError)) throw error;
    collector.invalidate();
    return { capture: collector.freeze(), failure: error };
  }
}

/**
 * Rebuild a derived node's outcome: `rebind`, then its pure, synchronous `forward`. With `from`
 * empty (a child START over its constant wiring), it binds from constants only.
 *
 * @param options.capture Whether to collect the route and bound inputs too, with each item a MAP's
 *   `forward` binds; defaults to `false`.
 * @returns The outcome, and the capture (`null` unless `options.capture`).
 * @throws TypeError when `node` isn't derived, or its `forward` returns a promise or a route.
 * @throws BindingError when a source doesn't evaluate, and whatever `resolve` throws.
 */
export function rebuildDerived(
  node: Node,
  flow: Flow,
  from: ReadonlyMap<string, number>,
  resolve: (name: string, seq: number) => ReboundSource,
  wiringOverride: JsonObject | null,
  options: { capture?: boolean } = {},
): { outcome: Output | Grow; capture: BindingCapture | null } {
  if (node.derived !== true) throw new TypeError(`node ${reprText(node.id)} is not derived`);
  const collector = options.capture === true ? new BindingCollector() : null;
  const { inputs, caps } = bindFrom(node, flow, from, resolve, wiringOverride, collector);
  const outcome: unknown = node.forward(inputs, caps);
  if (!(outcome instanceof Output || outcome instanceof Grow)) {
    throw new TypeError(`derived node ${reprText(node.id)} forward returned ${jsTypeName(outcome)}, not an Output or Grow`);
  }
  // Frozen after `forward`, which binds a MAP's items into the capture.
  return { outcome, capture: collector === null ? null : collector.freeze() };
}
