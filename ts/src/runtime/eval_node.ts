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
 * Knows about:   compile/model (types), diagnostics, events, expr lineage, memory contracts, nodes,
 *                runtime contracts and log, skills, tools, typesys.
 * Never imports: runtime state or engine, the run boundary, compose, or the CLI.
 */

import type { CompiledFlow } from "../compile/model.ts";
import { Failure, type FailureCode, type OriginRef, reprText } from "../diagnostics.ts";
import { ResolvedValue, ValueRoute } from "../expr/lineage.ts";
import {
  ConversationInvocation,
  EvaluationEvidence,
  type EvaluationTrace,
  type JsonObject,
  MemoryContext,
  MemoryContextRequest,
  MemoryRecallError,
  MemoryScope,
  MemorySection,
  RecallContext,
  type VisibleInteraction,
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
  Route,
  type Steering,
} from "../nodes/base.ts";
import {
  BindingCapture,
  BindingCollector,
  BindingError,
  bindParams,
  type ReservedBinder,
  resolveSource,
  type SourceResolver,
} from "../nodes/binding.ts";
import { isExactly, NodeSteps } from "../nodes/contracts.ts";
import type { Skill } from "../skills/catalog.ts";
import { ToolSet } from "../tools/registry.ts";
import type { JsonValue } from "../typesys/json.ts";
import type { VariablePool } from "../typesys/pool.ts";
import { isList, isRecord, jsTypeName, setField, type Value, type ValueRecord } from "../typesys/values.ts";
import { BackwardCapture } from "./contracts.ts";
import { canonicalSeed } from "./log.ts";

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

/**
 * Return an `Error`'s stack followed by its string `notes`, one per line, else `null`; Python's
 * traceback text prints `__notes__` the same way, so a cleanup note stays durable.
 */
function tracebackText(error: unknown): string | null {
  if (!(error instanceof Error)) return null;
  const notes: unknown = (error as { notes?: unknown }).notes;
  const lines = [error.stack, ...(Array.isArray(notes) ? notes : [])].filter((line) => typeof line === "string");
  return lines.length > 0 ? lines.join("\n") : null;
}

/** Return `error` as a capability fault: a `CapabilityFault` as it is, anything else wrapped. */
function capabilityFault(error: unknown): CapabilityFault {
  return error instanceof CapabilityFault ? error : new CapabilityFault(error);
}

/**
 * Append the note `text()` to `error`'s own `notes` array, creating it when absent (Python's
 * `add_note`). A failure, such as a frozen error, a non-array `notes`, a primitive or a throwing
 * `text`, is ignored. Local copy of `memory/session.ts`'s `addNote`.
 */
function addNote(error: unknown, text: () => string): void {
  try {
    if (typeof error !== "object" || error === null) return;
    const target = error as { notes?: unknown };
    const note = text();
    if (!Object.hasOwn(target, "notes")) target.notes = [];
    if (Array.isArray(target.notes)) target.notes.push(note);
  } catch {}
}

/**
 * Carry a host observer fault out of a node body as a capability fault, so a failing
 * `onStream`/`onEvent` stays a host fault, never a node failure. `onFault` sees each fault before
 * it is thrown.
 */
function observerSink<T>(
  sink: (item: T) => unknown,
  onFault: (fault: CapabilityFault) => void,
): (item: T) => void {
  return (item) => {
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
  readonly #onFault: (fault: CapabilityFault) => void;
  readonly #pending: Promise<unknown>[];

  /**
   * @param onFault Sees each capability fault before it is thrown or rejected with.
   * @param pending Receives each promise `drain` hands out, for the evaluator to await.
   */
  constructor(inbox: Steering, onFault: (fault: CapabilityFault) => void, pending: Promise<unknown>[]) {
    this.#inbox = inbox;
    this.#onFault = onFault;
    this.#pending = pending;
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
      throw this.#fault(error);
    }
    if (!(signal instanceof AbortSignal)) {
      throw this.#fault(new TypeError("steering.interruptSignal must return an AbortSignal"));
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
      throw this.#fault(error);
    }
    if (!(messages instanceof Promise)) return this.#checked(messages);
    const checked = messages.then(
      (answer: unknown) => this.#checked(answer),
      (error: unknown) => {
        throw this.#fault(error);
      },
    );
    // A node may drop the promise: its fault must not become an unhandled rejection, and the
    // evaluator awaits it so the fault still reaches `onFault` before the evaluation settles.
    checked.catch(() => {});
    this.#pending.push(checked);
    return checked;
  }

  /** Return a fresh copy of drained steer messages after checking they are strings. */
  #checked(messages: unknown): string[] {
    if (!isList(messages) || messages.some((message) => typeof message !== "string")) {
      throw this.#fault(new TypeError("steering.drain must return an array of strings"));
    }
    return [...(messages as string[])];
  }

  /** Return `error` as a capability fault after reporting it. */
  #fault(error: unknown): CapabilityFault {
    const fault = capabilityFault(error);
    this.#onFault(fault);
    return fault;
  }
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

/** The invocation's write-only trace, frozen once into evidence when the evaluation settles. */
class EvaluationTraceCollector implements EvaluationTrace {
  readonly #scope: MemoryScope | null;
  #boundInputs: JsonObject | null = null;
  #baseSystemPrompt: string | null = null;
  #prompt: string | null = null;
  #memorySections: readonly MemorySection[] = [];
  readonly #interactions: VisibleInteraction[] = [];
  #promptRecorded = false;
  #frozen = false;

  constructor(scope: MemoryScope | null) {
    this.#scope = scope;
  }

  /**
   * Keep the evaluator's canonical input snapshot taken before the node runs; inputs evidence
   * refuses are dropped.
   *
   * @throws Error once the trace is frozen.
   */
  setBoundInputs(boundInputs: JsonObject): void {
    this.#requireOpen();
    try {
      this.#boundInputs = new EvaluationEvidence(this.#scope, boundInputs, null, null).boundInputs;
    } catch (error) {
      if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
      // Evidence is observational: the run log can carry values the memory JSON boundary refuses.
      this.#boundInputs = null;
    }
  }

  /**
   * Record the one visible prompt and its ordered memory sections; a prompt evidence refuses is
   * ignored, but still counts as the one prompt.
   *
   * @throws Error once the trace is frozen, or for a second prompt.
   */
  recordPrompt(baseSystemPrompt: string, prompt: string, memorySections: readonly MemorySection[] = []): void {
    this.#requireOpen();
    if (this.#promptRecorded) throw new Error("evaluation trace prompt was already recorded");
    this.#promptRecorded = true;
    let evidence: EvaluationEvidence;
    try {
      evidence = new EvaluationEvidence(this.#scope, null, baseSystemPrompt, prompt, memorySections);
    } catch (error) {
      if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
      // The model already received this prompt, so a trace refusal mustn't fail the node.
      return;
    }
    this.#baseSystemPrompt = evidence.baseSystemPrompt;
    this.#prompt = evidence.prompt;
    this.#memorySections = evidence.memorySections;
  }

  /**
   * Append one exact visible interaction.
   *
   * @throws Error once the trace is frozen.
   * @throws TypeError or RangeError as `EvaluationEvidence` refuses the interaction.
   */
  recordInteraction(interaction: VisibleInteraction): void {
    this.#requireOpen();
    this.#interactions.push(new EvaluationEvidence(this.#scope, null, null, null, [], [interaction]).interactions[0]!);
  }

  /**
   * Close the trace and return its evidence.
   *
   * @throws Error when the trace is already frozen.
   */
  freeze(): EvaluationEvidence {
    this.#requireOpen();
    this.#frozen = true;
    return new EvaluationEvidence(
      this.#scope,
      this.#boundInputs,
      this.#baseSystemPrompt,
      this.#prompt,
      this.#memorySections,
      this.#interactions,
    );
  }

  /** Refuse a write once the evaluator froze the trace. */
  #requireOpen(): void {
    if (this.#frozen) throw new Error("evaluation trace is already frozen");
  }
}

/**
 * Return a trace for a node declaring `needsEvaluationTrace` while evidence capture is on, else
 * `null`.
 *
 * @throws TypeError when the node's `memoryScope` is neither `null` nor exactly a `MemoryScope`.
 */
function newEvaluationTrace(node: Node, captureEvidence: boolean): EvaluationTraceCollector | null {
  if (!captureEvidence || node.needsEvaluationTrace !== true) return null;
  const scope = node.memoryScope;
  if (scope !== null && !isExactly(scope, MemoryScope)) {
    throw new TypeError("node memory_scope must be an exact MemoryScope or null");
  }
  return new EvaluationTraceCollector(scope);
}

/**
 * Host-injected admission of notes and an optional structured transcript: selects one memory
 * context for a scoped node's `request`, under the node's `signal`. The evaluator closes the
 * context's transcript when the evaluation ends.
 */
export type MemoryContextOpener = (
  scope: MemoryScope,
  request: MemoryContextRequest,
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
  /**
   * End the lifetime: later opens reject, and the admitted transcript closes with `ending.error`,
   * or as a normal return when `ending` is `null`. Await it once the opener's promise settled.
   *
   * @throws The transcript's close failure after a normal return; after an error, the failure is
   *   a note on that error instead.
   */
  close(ending: { error: unknown } | null): Promise<void>;
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
  invocation: ConversationInvocation | null;
  signal: AbortSignal;
  canonicalInputs: JsonObject | null;
  onFault: (fault: CapabilityFault) => void;
  pending: Promise<unknown>[];
}): MemoryContextCapability {
  const { scope, opener, invocation, signal, canonicalInputs, onFault, pending } = options;
  let closed = false;
  let used = false;
  let admitted: MemoryContext | null = null;

  /**
   * Open the context once.
   *
   * @throws MemoryRecallError for invalid recall inputs, request or opener result, or as the opener
   *   throws it.
   * @throws CapabilityFault carrying a repeated open or any other opener fault.
   */
  const enter = async (
    baseSystemPrompt: string,
    renderedPrompt: string,
    compatibility: JsonObject,
  ): Promise<MemoryContext> => {
    try {
      if (used) throw new Error("memory context capability was already entered");
      used = true;
      if (scope === null || scope.memories.length === 0) return new MemoryContext();
      if (canonicalInputs === null) throw new MemoryRecallError("memory recall context is invalid");
      let request: MemoryContextRequest;
      try {
        const recall = new RecallContext(baseSystemPrompt, renderedPrompt, canonicalInputs, scope.staticPath);
        request = new MemoryContextRequest(recall, invocation, compatibility);
      } catch (error) {
        if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
        throw new MemoryRecallError("memory recall context is invalid", { cause: error });
      }
      // The evaluator refuses a scoped node without an opener before the node runs.
      const context: unknown = await opener!(scope, request, { signal });
      try {
        if (!isExactly(context, MemoryContext)) throw new TypeError("memory context must be an exact MemoryContext");
        // Admitted before its sections are checked, so a refused context still closes.
        admitted = context;
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

  const open = (baseSystemPrompt: string, renderedPrompt: string, compatibility: JsonObject) => {
    const opened: Promise<MemoryContext> = closed
      ? Promise.reject(new Error("memory context capability is closed"))
      : enter(baseSystemPrompt, renderedPrompt, compatibility);
    // A node may drop the promise: its rejection must not go unhandled, and the evaluator awaits
    // it, so a fault still reaches `onFault` and an admitted transcript still closes.
    opened.catch(() => {});
    if (!closed) pending.push(opened);
    return opened;
  };

  const close = async (ending: { error: unknown } | null): Promise<void> => {
    closed = true;
    const transcript = admitted?.transcript ?? null;
    if (transcript === null) return;
    // The turn reads `undefined` as a normal return, so a thrown `undefined` needs a stand-in.
    let error = ending?.error;
    if (ending !== null && error === undefined) error = new Error(markerMessage(undefined));
    try {
      await transcript.close(error);
    } catch (cleanup) {
      if (ending === null) throw cleanup;
      if (cleanup !== error) {
        addNote(error, () => {
          const name = cleanup instanceof Error ? cleanup.name : jsTypeName(cleanup);
          return `Memory context cleanup failed: ${name}: ${markerMessage(cleanup)}`;
        });
      }
    }
  };

  return { open, close };
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
  /** The transient stream sink a node declaring `needsLiveEvents` receives. */
  emitStream?: NodeCaps["emitStream"] | null;
  /** The transient model and tool call observer sink a node declaring `needsLiveEvents` receives. */
  emitEvent?: NodeCaps["emitEvent"] | null;
  /** Whether to capture visible evidence from a node declaring `needsEvaluationTrace`. */
  captureEvidence?: boolean;
  /** Whether to retain the bound input values, their routes and the backward capture. */
  captureBindings?: boolean;
  /**
   * The memory context source a node declaring `needsMemoryContext` reaches through
   * `caps.openMemoryContext`; required when the node's scope has memories.
   */
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
  /**
   * This start's steps from the engine, delivered only to a node declaring `needsSteps` as a
   * `NodeSteps` over the same `starts` and `prior`; a `CapabilityFault` its `record` throws is a
   * capability fault of this evaluation. The node's `record` closes when the outcome is decided.
   */
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
 * @throws The original error of the first capability fault (a failing stream or event sink,
 *   steering inbox, including a `drain` promise the node dropped, a refused or failing render
 *   capture, a repeated memory context open or a failing opener, a `CapabilityFault` from
 *   `steps.record`), whatever the node then returned or threw.
 * @throws TypeError when a node declaring `needsMemoryContext`, or `needsEvaluationTrace` while
 *   capturing evidence, has a `memoryScope` that isn't exactly a `MemoryScope`; Error when a
 *   memory context node's scope has memories and no opener was injected.
 * @throws The memory transcript's close failure after the node succeeded; after a failure or
 *   host error it is a note on that error.
 * @throws TypeError, RangeError or the value's own error when evidence capture can't canonicalize
 *   the visible bound inputs of an unscoped node; the body hasn't run.
 * @throws TypeError, RangeError or the hook's own error when capturing bindings and the input
 *   inventory, `boundLlmConfig` or `captureBackwardContext` fails, or their material isn't JSON;
 *   the body hasn't run.
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
  // Read once, so a node replacing its scope while it runs doesn't change what it recalls.
  const memoryScope = node.needsMemoryContext === true ? node.memoryScope : null;
  if (memoryScope !== null && !isExactly(memoryScope, MemoryScope)) {
    throw new TypeError("node memory_scope must be an exact MemoryScope or null");
  }
  if (effectiveLlmByStaticPath !== null && !isRecord(effectiveLlmByStaticPath)) {
    throw new TypeError("effective_llm_by_static_path must be a record or null");
  }
  if (options.signal != null && !(options.signal instanceof AbortSignal)) {
    throw new TypeError("signal must be an AbortSignal or null");
  }
  const signal = options.signal ?? new AbortController().signal;
  signal.throwIfAborted();
  const trace = newEvaluationTrace(node, captureEvidence);
  let evidence: EvaluationEvidence | null = null;
  const collector = captureBindings ? new BindingCollector() : null;
  let backwardCapture = captureBindings
    ? new BackwardCapture({
        inputNames: node.backwardInputNames(),
        variableNames: node.declaredVariables().map((parameter) => parameter.name),
        scope: node.executionScope,
      })
    : null;
  let capturesClosed = false;
  let renderRecorded = false;
  let boundConfig: JsonValue | null = null;

  /** Return the config backward records: the bound one, else the run's entry for the static path. */
  const recordedModelConfig = (): JsonValue | null => {
    const scope = backwardCapture!.scope;
    if (scope === null || effectiveLlmByStaticPath === null) return null;
    // A config resolved only when the node ran is bound with its inputs, so `captureInputs` has
    // snapshotted it.
    if (boundConfig !== null) return boundConfig;
    const key = scope.staticPath.join("/");
    if (!Object.hasOwn(effectiveLlmByStaticPath, key)) {
      throw new RangeError(`recorded effective model configuration is missing static path ${reprText(key)}`);
    }
    const config: unknown = effectiveLlmByStaticPath[key];
    if (!isRecord(config)) throw new TypeError("recorded effective model configuration must be a record");
    return canonicalSeed(config)[1];
  };

  /** Snapshot the completed inputs and static material before the body can change them. */
  const captureInputs = (inputs: Record<string, unknown>): void => {
    const config = node.boundLlmConfig(inputs);
    if (config !== null) {
      if (!isRecord(config)) throw new TypeError("bound model configuration must be a record");
      boundConfig = canonicalSeed(config)[1];
    }
    collector!.finish(inputs as ValueRecord);
    const data: unknown = node.captureBackwardContext();
    if (!isRecord(data)) throw new TypeError("capture_backward_context must return a record");
    const { inputNames, variableNames, scope } = backwardCapture!;
    const captured = new BackwardCapture({ inputNames, variableNames, scope, nodeData: data });
    canonicalSeed(captured.nodeData);
    backwardCapture = captured;
  };

  /**
   * Freeze the node's one actual render publication, apart from its visible evidence.
   *
   * @throws Error once the node and its capability promises have settled.
   * @throws CapabilityFault carrying any other refusal or fault.
   */
  const captureRender = (data: Readonly<Record<string, unknown>>): void => {
    if (capturesClosed) throw new Error("render capture is already closed");
    try {
      if (renderRecorded) throw new Error("render capture was already recorded");
      if (!isRecord(data)) throw new TypeError("render capture must be a record");
      if (Object.hasOwn(data, "effective_llm_config")) {
        throw new RangeError("effective_llm_config belongs to recorded runtime authority");
      }
      const snapshot = canonicalSeed(data)[1] as Record<string, unknown>;
      const { inputNames, variableNames, scope, nodeData } = backwardCapture!;
      if (Object.keys(snapshot).some((key) => Object.hasOwn(nodeData, key))) {
        throw new RangeError("render capture cannot replace static node data");
      }
      backwardCapture = new BackwardCapture({
        inputNames,
        variableNames,
        scope,
        nodeData: { ...nodeData, ...snapshot, effective_llm_config: recordedModelConfig() },
      });
      renderRecorded = true;
    } catch (error) {
      throw capabilityFault(error);
    }
  };

  /**
   * Close the render capture and freeze the trace once the outcome is decided, so a later write is
   * refused.
   */
  const closeCaptures = (): void => {
    if (capturesClosed) return;
    capturesClosed = true;
    evidence = trace?.freeze() ?? null;
  };

  let outcome: WorkerOutcome;
  try {
    outcome = await evaluateNode(node, flow, pool, {
      wiringOverride,
      signal,
      collector,
      captureInputs: captureBindings ? captureInputs : null,
      captureRender: captureBindings && node.needsRenderCapture === true ? captureRender : null,
      closeCaptures,
      trace,
      memoryScope,
      memoryContextOpener,
      memoryInvocation,
      sourceResolver,
      emitStream,
      emitEvent,
      skills,
      tools,
      steering,
      steps,
    });
  } finally {
    // The escape paths (a binder fault, a failing input capture) never reach the evaluator's close.
    closeCaptures();
  }
  return new EvaluatedOutcome(outcome, evidence, collector?.freeze() ?? null, backwardCapture);
}

/** Bind and run one node, resolving to one normalized result. */
async function evaluateNode(
  node: Node,
  flow: CompiledFlow | null,
  pool: VariablePool,
  context: {
    wiringOverride: JsonObject | null;
    signal: AbortSignal;
    /** Retains the bound values and routes, when capturing bindings. */
    collector: BindingCollector | null;
    /** Receives the completed input record before the body runs, when capturing bindings. */
    captureInputs: ((inputs: Record<string, unknown>) => void) | null;
    /** The render capture a node declaring `needsRenderCapture` receives, when capturing bindings. */
    captureRender: NonNullable<NodeCaps["captureRender"]> | null;
    /** Closes the captures once the node and its capability promises have settled. */
    closeCaptures: () => void;
    /** The trace a node declaring `needsEvaluationTrace` receives, when capturing evidence. */
    trace: EvaluationTraceCollector | null;
    /** The memory scope of a node declaring `needsMemoryContext`, else `null`. */
    memoryScope: MemoryScope | null;
    memoryContextOpener: MemoryContextOpener | null;
    memoryInvocation: ConversationInvocation | null;
    sourceResolver: SourceResolver | null;
    emitStream: NonNullable<NodeCaps["emitStream"]> | null;
    emitEvent: NonNullable<NodeCaps["emitEvent"]> | null;
    skills: ReadonlyMap<string, Skill> | null;
    tools: ToolSet | null;
    steering: Steering | null;
    steps: NodeSteps | null;
  },
): Promise<WorkerOutcome> {
  const { wiringOverride, signal, collector, captureInputs, captureRender, closeCaptures, trace, sourceResolver } =
    context;
  const { memoryScope, memoryContextOpener, memoryInvocation } = context;
  const { emitStream, emitEvent, skills, tools, steering, steps } = context;
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
  // The visible bound inputs, for the recall context and the evidence snapshot.
  let canonicalInputs: JsonObject | null = null;
  if (scoped || trace !== null) {
    const hidden = node.modelHiddenInputs();
    const visible = Object.fromEntries(Object.entries(record).filter(([name]) => !hidden.has(name)));
    try {
      const value = canonicalSeed(visible)[0];
      if (!isRecord(value)) throw new TypeError("bound node inputs did not canonicalize to an exact record");
      canonicalInputs = value as JsonObject;
    } catch (error) {
      // The recall capability owns the rejection of its query context; evidence just omits the
      // inputs.
      if (!scoped || !(error instanceof TypeError || error instanceof RangeError)) throw error;
    }
    if (canonicalInputs !== null) trace?.setBoundInputs(canonicalInputs);
  }

  // The first fault any capability raised. A node can catch the fault it was thrown, so the
  // evaluator keeps it to throw when the node settles.
  const faults: { first: CapabilityFault | null } = { first: null };
  const onFault = (fault: CapabilityFault): void => {
    faults.first ??= fault;
  };
  // The capability promises handed to the node. A node may settle without awaiting one, so the
  // evaluator awaits their settlement before it settles, and a late fault is still the first.
  const pending: Promise<unknown>[] = [];
  const memory = memoryContextCapability({
    scope: memoryScope,
    opener: memoryContextOpener,
    invocation: memoryInvocation,
    signal,
    canonicalInputs,
    onFault,
    pending,
  });

  // Build only the node-owned, trait-gated capabilities. The deferred per-item binder throws the
  // same binder-owned marker from inside the body, and the classification below keeps it.
  const caps: { -readonly [K in keyof NodeCaps]: NodeCaps[K] } = { signal };
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
  if (captureRender !== null) {
    // The capture throws its refusals and faults as capability faults; a write after the
    // evaluation settled is a plain error with no evaluation left to fail.
    caps.captureRender = (captured) => {
      try {
        captureRender(captured);
      } catch (error) {
        if (error instanceof CapabilityFault) onFault(error);
        throw error;
      }
    };
  }
  if (trace !== null) {
    // A view, so node code can't freeze the collector or replace its inputs snapshot.
    caps.evaluationTrace = Object.freeze({
      recordPrompt: (baseSystemPrompt: string, prompt: string, memorySections?: readonly MemorySection[]) =>
        trace.recordPrompt(baseSystemPrompt, prompt, memorySections),
      recordInteraction: (interaction: VisibleInteraction) => trace.recordInteraction(interaction),
    });
  }
  if (node.needsMemoryContext === true) caps.openMemoryContext = memory.open;
  if (node.needsTools === true) caps.tools = tools ?? new ToolSet();
  if (node.needsLiveEvents === true) {
    if (emitStream !== null) caps.emitStream = observerSink(emitStream, onFault);
    if (emitEvent !== null) caps.emitEvent = observerSink(emitEvent, onFault);
  }
  if (skills !== null && node.needsSkills === true) caps.skills = skills;
  if (steering !== null && node.needsSteering === true) caps.steering = new SteeringCapability(steering, onFault, pending);
  // Closed with the captures, so a step the node's dropped code records after the outcome is
  // decided is refused rather than committed with no evaluation left to fail.
  let stepsClosed = false;
  if (steps !== null && node.needsSteps === true) {
    // The engine's recorder throws its faults as capability faults; a node can catch one, so the
    // evaluator keeps it as it keeps the other capabilities' faults.
    caps.steps = new NodeSteps(steps.starts, steps.prior, (step) => {
      if (stepsClosed) throw new CapabilityFault(new Error("node steps capability is closed"));
      try {
        steps.record(step);
      } catch (error) {
        if (error instanceof CapabilityFault) onFault(error);
        throw error;
      }
    });
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
  // The captures stay open while the evaluator waits, so a node's dropped code may start another
  // capability call meanwhile; one wait would miss it, and a late open's transcript would be
  // admitted after `memory.close` and never closed, and its fault never checked.
  for (let settled = 0; settled < pending.length; ) {
    const batch = pending.slice(settled);
    settled = pending.length;
    await Promise.allSettled(batch);
  }
  stepsClosed = true;
  closeCaptures();

  // A host capability fault, then a stop, is control flow outside the node-result sum, whatever
  // the node returned or rejected with.
  let ending: { error: unknown } | null = null;
  if (faults.first !== null) ending = { error: faults.first.original };
  else if (rejection !== null && signal.aborted) ending = { error: signal.reason };
  if (ending !== null) {
    await memory.close(ending);
    throw ending.error;
  }
  if (rejection !== null) {
    const { error } = rejection;
    // A template default that fails to bind leaves no usable input record.
    if (usesBaseTemplate && error instanceof BindingStageError) collector?.invalidate();
    // The memory context sees the real failure, not the evaluator's stage marker.
    await memory.close({ error: stageOriginal(error) });
    return stageFailure(node, error, usesBaseTemplate);
  }
  const contractError = resultContractError(node, result);
  if (contractError !== null) {
    await memory.close({ error: contractError });
    return evaluationFailure(node, "contract.node_result", contractError);
  }
  await memory.close(null);
  return result as NodeResult;
}

/** Return the error of a result outside `NodeResult` or a `Grow` from a non-spawner, else `null`. */
function resultContractError(node: Node, result: unknown): Error | null {
  if (!(result instanceof Output || result instanceof Route || result instanceof Grow)) {
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

/** Return the original a stage marker or fault carries, else `error` itself. */
function stageOriginal(error: unknown): unknown {
  return error instanceof DeferredBindingFault ||
    error instanceof CapabilityFault ||
    error instanceof BindingStageError ||
    error instanceof AssertionStageError ||
    error instanceof NodeBodyError
    ? error.original
    : error;
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
