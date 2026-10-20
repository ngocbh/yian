/**
 * Node evaluation at the engine's pure read/execution boundary.
 *
 * A node is a pure function of its bound input record; the engine owns both boundaries: the read
 * (bind the inputs from the pool) and the write (the dispatcher stores `Output`). `evalNode` is
 * the append-only dispatcher's evaluation seam: it binds the record, pre-resolves reserved keys,
 * builds the trait-gated capabilities, runs the node through its assert-owning `run` template,
 * and resolves to an `EvaluatedOutcome` holding exactly `Output | Route | Grow |
 * EvaluationFailure` plus optional evidence and input capture. It publishes no observers; durable
 * observer order belongs to the dispatcher.
 *
 * An expected rejection is classified where its operation occurs: binding evaluation, supplied
 * input contracts, template assertions, node execution, or node-result validation. A stop (the
 * node's aborted signal), host capability faults, unexpected binder faults, and malformed compiled
 * or setup state reject `evalNode` instead of becoming an `EvaluationFailure`.
 *
 * Knows about:   compile/model (types), diagnostics, events, memory contracts, nodes, runtime
 *                contracts and log, skills, tools, typesys.
 * Never imports: runtime state or engine, the run boundary, compose, or the CLI.
 */

import type { CompiledFlow } from "../compile/model.ts";
import { Failure, type FailureCode, type OriginRef, reprText } from "../diagnostics.ts";
import {
  ConversationInvocation,
  EvaluationEvidence,
  type JsonObject,
  type MemoryContext,
  type MemoryContextRequest,
  type MemoryScope,
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
  Output,
  Route,
  type Steering,
} from "../nodes/base.ts";
import { BindingCapture, BindingError, bindParams, type SourceResolver } from "../nodes/binding.ts";
import { isExactly, NodeSteps } from "../nodes/contracts.ts";
import type { Skill } from "../skills/catalog.ts";
import { ToolSet } from "../tools/registry.ts";
import type { VariablePool } from "../typesys/pool.ts";
import { isList, isRecord, jsTypeName, setField, type Value, type ValueRecord } from "../typesys/values.ts";
import { BackwardCapture } from "./contracts.ts";

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
 * (else the value's type name) as the cause type, and the `Error`'s stack as the traceback. The
 * engine uses it for its contract failures.
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
      traceback: error instanceof Error ? (error.stack ?? null) : null,
    }),
  );
}

/** Return `error` as a capability fault: a `CapabilityFault` as it is, anything else wrapped. */
function capabilityFault(error: unknown): CapabilityFault {
  return error instanceof CapabilityFault ? error : new CapabilityFault(error);
}

/**
 * Carry a host observer fault out of a node body as a capability fault, so a failing
 * `onStream`/`onEvent` stays a host fault, never a node failure.
 */
function observerSink<T>(sink: (item: T) => unknown): (item: T) => void {
  return (item) => {
    try {
      sink(item);
    } catch (error) {
      throw capabilityFault(error);
    }
  };
}

/**
 * Return a host steering inbox after checking its shape, or `null`.
 *
 * @throws TypeError when `steering` isn't `null` and lacks the `interruptSignal` and `drain`
 *   methods.
 */
export function requireSteering(steering: unknown): Steering | null {
  const inbox = steering as Partial<Steering> | null | undefined;
  if (steering !== null && !(typeof inbox?.interruptSignal === "function" && typeof inbox.drain === "function")) {
    throw new TypeError("steering must provide interruptSignal(nodeId) and drain(nodeId), or be null");
  }
  return steering as Steering | null;
}

/**
 * @internal The steering a node receives: it carries host inbox faults and malformed answers out
 * of the node body as capability faults, so a failing inbox stays a host fault, never a node
 * failure. The evaluator wraps the run's inbox in it.
 */
export class SteeringCapability implements Steering {
  readonly #inbox: Steering;

  constructor(inbox: Steering) {
    this.#inbox = inbox;
  }

  /**
   * Return the host's interrupt signal for `nodeId`.
   *
   * @throws CapabilityFault carrying the host's error, or a `TypeError` when the answer isn't an
   *   `AbortSignal`.
   */
  interruptSignal(nodeId: string): AbortSignal {
    let signal: unknown;
    try {
      signal = this.#inbox.interruptSignal(nodeId);
    } catch (error) {
      throw capabilityFault(error);
    }
    if (!(signal instanceof AbortSignal)) {
      throw new CapabilityFault(new TypeError("steering.interruptSignal must return an AbortSignal"));
    }
    return signal;
  }

  /**
   * Return the node's queued steer messages as a fresh array of strings, or a promise of one when
   * the host answers with a promise.
   *
   * @throws CapabilityFault (or rejects with one) carrying the host's error, or a `TypeError` when
   *   the answer isn't an array of strings.
   */
  drain(nodeId: string): readonly string[] | Promise<readonly string[]> {
    let messages: unknown;
    try {
      messages = this.#inbox.drain(nodeId);
    } catch (error) {
      throw capabilityFault(error);
    }
    if (!(messages instanceof Promise)) return checkedMessages(messages);
    const checked = messages.then(checkedMessages, (error: unknown) => {
      throw capabilityFault(error);
    });
    // A node may drop the promise; its fault must not become an unhandled rejection.
    checked.catch(() => {});
    return checked;
  }
}

/** Return a fresh copy of drained steer messages after checking they are strings. */
function checkedMessages(messages: unknown): string[] {
  if (!isList(messages) || messages.some((message) => typeof message !== "string")) {
    throw new CapabilityFault(new TypeError("steering.drain must return an array of strings"));
  }
  return [...(messages as string[])];
}

/** The closed raw worker result the append-only dispatcher consumes. */
export type WorkerOutcome = Output | Route | Grow | EvaluationFailure;

/**
 * One raw worker outcome with frozen evidence and optional backward capture. Frozen.
 *
 * Binding values and routes travel with the common inventories and static material, as separate
 * lower binding and runtime carriers. Neither is accepted history until the dispatcher records
 * this outcome.
 */
export class EvaluatedOutcome {
  readonly outcome: WorkerOutcome;
  readonly evidence: EvaluationEvidence | null;
  readonly bindings: BindingCapture | null;
  readonly backwardCapture: BackwardCapture | null;

  /**
   * Require one closed raw outcome and exact optional evidence and captures; the backward capture
   * is checked again by rebuilding it.
   *
   * @throws TypeError for an outcome outside `WorkerOutcome`, or evidence or a capture that isn't
   *   exactly its class.
   * @throws RangeError when only one of `bindings` and `backwardCapture` is present, or as
   *   `BackwardCapture` refuses the capture.
   */
  constructor(
    outcome: WorkerOutcome,
    evidence: EvaluationEvidence | null,
    bindings: BindingCapture | null = null,
    backwardCapture: BackwardCapture | null = null,
  ) {
    if (
      !(
        outcome instanceof Output ||
        outcome instanceof Route ||
        outcome instanceof Grow ||
        outcome instanceof EvaluationFailure
      )
    ) {
      throw new TypeError("outcome must be a WorkerOutcome");
    }
    if (evidence !== null && !isExactly(evidence, EvaluationEvidence)) {
      throw new TypeError("evidence must be an EvaluationEvidence or null");
    }
    if (bindings !== null && !isExactly(bindings, BindingCapture)) {
      throw new TypeError("bindings must be a BindingCapture or null");
    }
    if (backwardCapture !== null) {
      if (!isExactly(backwardCapture, BackwardCapture)) {
        throw new TypeError("backward_capture must be a BackwardCapture or null");
      }
      const { inputNames, variableNames, scope, nodeData } = backwardCapture;
      new BackwardCapture({ inputNames, variableNames, scope, nodeData });
    }
    if ((bindings === null) !== (backwardCapture === null)) {
      throw new RangeError("binding capture and backward invocation metadata must be retained together");
    }
    this.outcome = outcome;
    this.evidence = evidence;
    this.bindings = bindings;
    this.backwardCapture = backwardCapture;
    Object.freeze(this);
  }
}

/**
 * Host-injected admission of notes and an optional structured transcript: selects one memory
 * context for a scoped node's `request`, under the node's `signal`.
 */
export type MemoryContextOpener = (
  scope: MemoryScope,
  request: MemoryContextRequest,
  options: { signal: AbortSignal },
) => Promise<MemoryContext>;

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
  /** The transient stream sink a node declaring `needsLiveEvents` receives. */
  emitStream?: NodeCaps["emitStream"] | null;
  /** The transient model and tool call observer sink a node declaring `needsLiveEvents` receives. */
  emitEvent?: NodeCaps["emitEvent"] | null;
  /** Whether to capture visible evidence from a node declaring `needsEvaluationTrace`. */
  captureEvidence?: boolean;
  /** Whether to retain the bound input values, their routes and the backward capture. */
  captureBindings?: boolean;
  /** The memory context source a node declaring `needsMemoryContext` reaches. */
  memoryContextOpener?: MemoryContextOpener | null;
  /** The accepted node-start identity, needed only to admit a transcript. */
  memoryInvocation?: ConversationInvocation | null;
  /** The run's host-supplied skills, delivered only to a node declaring `needsSkills`. */
  skills?: ReadonlyMap<string, Skill> | null;
  /** The paired value and source lookup used while capturing bindings. */
  sourceResolver?: SourceResolver | null;
  /** The accepted run's model configuration by root-relative static path, for render capture. */
  effectiveLlmByStaticPath?: Readonly<Record<string, JsonObject>> | null;
  /** The run's tool set, delivered only to a node declaring `needsTools`. */
  tools?: ToolSet | null;
  /** The run's host inbox, delivered only to a node declaring `needsSteering`. */
  steering?: Steering | null;
  /** This start's steps from the engine, delivered only to a node declaring `needsSteps`. */
  steps?: NodeSteps | null;
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
 * @returns The outcome, with evidence and captures when asked for.
 * @throws TypeError (as a rejection) for a malformed option.
 * @throws The signal's reason when the signal is already aborted, before binding, or when the node
 *   rejects after it aborted.
 * @throws RangeError when a node reading its steps fails to bind its reserved keys after an earlier
 *   start recorded steps.
 * @throws The original error of a capability fault, an unexpected per-item binder fault, or
 *   malformed node setup (a missing reserved wiring key, malformed `params`, no `forward`).
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
    captureEvidence = false,
    captureBindings = false,
    memoryContextOpener = null,
    memoryInvocation = null,
    skills = null,
    sourceResolver = null,
    effectiveLlmByStaticPath = null,
    tools = null,
    steering = null,
    steps = null,
  } = options;
  requireSteering(steering);
  if (typeof captureEvidence !== "boolean") throw new TypeError("capture_evidence must be a bool");
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
  if (memoryInvocation !== null && !isExactly(memoryInvocation, ConversationInvocation)) {
    throw new TypeError("memory_invocation must be a ConversationInvocation or null");
  }
  if (effectiveLlmByStaticPath !== null && !isRecord(effectiveLlmByStaticPath)) {
    throw new TypeError("effective_llm_by_static_path must be a record or null");
  }
  if (options.signal != null && !(options.signal instanceof AbortSignal)) {
    throw new TypeError("signal must be an AbortSignal or null");
  }
  const signal = options.signal ?? new AbortController().signal;
  signal.throwIfAborted();
  const outcome = await evaluateNode(node, flow, pool, {
    wiringOverride,
    signal,
    sourceResolver,
    emitStream,
    emitEvent,
    skills,
    tools,
    steering,
    steps,
  });
  return new EvaluatedOutcome(outcome, null);
}

/** Bind and run one node, resolving to one normalized result. */
async function evaluateNode(
  node: Node,
  flow: CompiledFlow | null,
  pool: VariablePool,
  context: {
    wiringOverride: JsonObject | null;
    signal: AbortSignal;
    sourceResolver: SourceResolver | null;
    emitStream: NonNullable<NodeCaps["emitStream"]> | null;
    emitEvent: NonNullable<NodeCaps["emitEvent"]> | null;
    skills: ReadonlyMap<string, Skill> | null;
    tools: ToolSet | null;
    steering: Steering | null;
    steps: NodeSteps | null;
  },
): Promise<WorkerOutcome> {
  const { wiringOverride, signal, sourceResolver, emitStream, emitEvent, skills, tools, steering, steps } = context;
  // The flow owns every source. A missing trait-declared reserved key is a malformed compiled
  // graph and deliberately escapes the expected binder catch.
  const nodeWiring: ReadonlyMap<string, unknown> =
    wiringOverride !== null ? new Map(Object.entries(wiringOverride)) : (flow?.wiring.get(node.id) ?? new Map());
  const usesBaseTemplate = node.run === Node.prototype.run;

  let record: ValueRecord;
  try {
    record = node.bindsPerItem
      ? {}
      : bindParams(node.params ?? [], nodeWiring, pool, {
          sourceResolver,
          types: node.types,
          labels: node.bindingLabels,
          finishCapture: false,
        });
    let reserved: Record<string, unknown>;
    try {
      reserved = node.bindReserved(nodeWiring, pool, { bind: null });
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
    for (const name of Object.keys(reserved)) setField(record, name, reserved[name] as Value);
  } catch (error) {
    if (!(error instanceof BindingError)) throw error;
    return evaluationFailure(node, error.failureCode, error, error.origin);
  }

  // Build only the node-owned, trait-gated capabilities. The deferred per-item binder throws the
  // same binder-owned marker from inside the body, and the classification below keeps it.
  const caps: { -readonly [K in keyof NodeCaps]: NodeCaps[K] } = { signal };
  if (node.bindsPerItem) {
    const params = node.params ?? [];
    caps.bindItem = (element) => {
      try {
        return bindParams(params, nodeWiring, pool, {
          item: element,
          sourceResolver,
          types: node.types,
          labels: node.bindingLabels,
        });
      } catch (error) {
        if (error instanceof BindingError) throw new BindingStageError(error);
        throw new DeferredBindingFault(error);
      }
    };
  }
  if (node.needsTools === true) caps.tools = tools ?? new ToolSet();
  if (node.needsLiveEvents === true) {
    if (emitStream !== null) caps.emitStream = observerSink(emitStream);
    if (emitEvent !== null) caps.emitEvent = observerSink(emitEvent);
  }
  if (skills !== null && node.needsSkills === true) caps.skills = skills;
  if (steering !== null && node.needsSteering === true) caps.steering = new SteeringCapability(steering);
  if (steps !== null && node.needsSteps === true) caps.steps = steps;

  let result: unknown;
  try {
    result = usesBaseTemplate ? await node.runForEvaluator(record, caps) : await node.run(record, caps);
  } catch (error) {
    // A stop is control flow outside the node-result sum, whatever the node rejected with.
    if (signal.aborted) throw signal.reason;
    return stageFailure(node, error, usesBaseTemplate);
  }
  if (!(result instanceof Output || result instanceof Route || result instanceof Grow)) {
    return evaluationFailure(
      node,
      "contract.node_result",
      new Error(`node ${reprText(node.id)} run() returned ${jsTypeName(result)}, not a NodeResult`),
    );
  }
  if (result instanceof Grow && !node.isSpawner) {
    // The kind is diagnostic text only; the spawner trait owns this gate.
    return evaluationFailure(
      node,
      "contract.node_result",
      new Error(
        `node ${reprText(node.id)} (kind ${node.kind}) returned a Grow but is not a spawner (CALL/MAP/LOOP); ` +
          "only spawner kinds may grow the graph",
      ),
    );
  }
  return result;
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
