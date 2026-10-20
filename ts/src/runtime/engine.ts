/**
 * Append-only flow execution over one prepared run-state/log authority.
 *
 * `FlowEngine` is the sole ready-frontier consumer and state writer. Serial mode (no workers)
 * evaluates durably started nodes inline. Pooled mode keeps at most `numWorkers` evaluations in
 * flight and commits their outcomes in settle order, read from a FIFO completion queue. In both,
 * every `prepare* -> commitPrepared` runs with no `await` between. Every observer it returns
 * follows its record's append and reducer application. Stream chunks are transient callback-only
 * messages and never enter the log.
 *
 * Knows about:   compile/model, diagnostics, events, memory contracts, nodes, runtime eval_node,
 *                log, state and trace, skills, tools, typesys.
 * Never imports: the run boundary, compose, or the CLI.
 */

import { AsyncResource } from "node:async_hooks";

import { GrowthRejected } from "../compile/model.ts";
import { type Failure, type FailureCode, type OriginRef, reprText } from "../diagnostics.ts";
import { NodeAttemptResumed, type RunEvent, RunFailed, type StreamChunk, ToolCallFinished } from "../events.ts";
import { ConversationInvocation } from "../memory/contracts.ts";
import { CapabilityFault, Grow, markerMessage, Output, Route, type Steering } from "../nodes/base.ts";
import { isExactly, NodeSteps } from "../nodes/contracts.ts";
import type { Skill } from "../skills/catalog.ts";
import { ToolSet } from "../tools/registry.ts";
import { jsTypeName, type ValueRecord } from "../typesys/values.ts";
import {
  addNote,
  EvaluatedOutcome,
  EvaluationFailure,
  evalNode,
  evaluationFailure,
  type MemoryContextOpener,
  requireSteering,
} from "./eval_node.ts";
import type { OutcomeCarriers, RunLog, RunResumedRecord, RunStopReason } from "./log.ts";
import {
  commitPrepared,
  type GrowBuilder,
  NodeDataError,
  type PreparedTransition,
  prepareExpanded,
  prepareNodeFailed,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeSucceeded,
  prepareRunStopped,
  RunState,
} from "./state.ts";
import { captureBindings } from "./trace.ts";

/** The dependencies and host callbacks of one `FlowEngine`. */
export interface FlowEngineOptions {
  /** The compile-owned builder of a spawner's growth fragment. */
  growBuilder: GrowBuilder;
  /** The worker count `RunStarted` (or the last `RunResumed`) recorded; zero selects serial mode. */
  numWorkers: number;
  /**
   * Sees each durable observer before `run()` returns it, and each in-loop model and tool event.
   * Called synchronously and never awaited; a throw, or a promise it returns that rejects, stops
   * the run with that error; after the terminal result a throw only rejects `run()` and a rejection
   * is ignored.
   */
  onEvent?: ((event: RunEvent) => unknown) | null;
  /** Sees each transient stream chunk; called as `onEvent` is. */
  onStream?: ((chunk: StreamChunk) => unknown) | null;
  /** Admits memory context for a node declaring `needsMemoryContext`. */
  memoryContextOpener?: MemoryContextOpener | null;
  /** The run's tool set, bound to the run id and signal; the global registry when omitted. */
  tools?: ToolSet | null;
  /** The run's host-supplied skills, passed to nodes declaring `needsSkills`. */
  skills?: ReadonlyMap<string, Skill> | null;
  /**
   * The run's host inbox, passed to nodes declaring `needsSteering`; a method that throws, or
   * returns a promise that rejects, stops the run with that error.
   */
  steering?: Steering | null;
  /** The host's stop: aborting it stops the run with its reason. */
  signal?: AbortSignal | null;
}

/** One settled worker: the outcome its evaluation resolved to, or the error it rejected with. */
type Completion =
  | { readonly nodeId: string; readonly outcome: EvaluatedOutcome }
  | { readonly nodeId: string; readonly error: unknown };

/** The settled workers in settle order, read by one dispatcher. */
class CompletionQueue {
  readonly #entries: Completion[] = [];
  /** The next unread entry; read entries are dropped in batches so taking never shifts the array. */
  #head = 0;
  #waiter: ((entry: Completion) => void) | null = null;

  /** Add one settled worker, handing it straight to a waiting `take`. */
  push(entry: Completion): void {
    const waiter = this.#waiter;
    if (waiter === null) {
      this.#entries.push(entry);
      return;
    }
    this.#waiter = null;
    waiter(entry);
  }

  /**
   * Return the oldest unread entry, or a promise of the next one to arrive. At most one `take` may
   * be pending: the queue has a single reader, the dispatcher.
   *
   * @throws Error when another `take` is still pending.
   */
  take(): Completion | Promise<Completion> {
    if (this.#waiter !== null) throw new Error("CompletionQueue has a take pending already");
    if (this.#head === this.#entries.length) {
      return new Promise((resolve) => {
        this.#waiter = resolve;
      });
    }
    const entry = this.#entries[this.#head]!;
    this.#head += 1;
    // Dropping the read half costs at most as many moves as the takes that preceded it.
    if (2 * this.#head >= this.#entries.length) {
      this.#entries.splice(0, this.#head);
      this.#head = 0;
    }
    return entry;
  }
}

/** Whether `log` has the `RunLog` members the engine reads. */
function isRunLog(log: unknown): log is RunLog {
  const candidate = log as Partial<RunLog> | null;
  return (
    typeof candidate === "object" &&
    candidate !== null &&
    typeof candidate.append === "function" &&
    typeof candidate.runId === "string" &&
    typeof candidate.nextSequence === "number"
  );
}

/**
 * Return one `NodeAttemptResumed` per retried node of a committed `RunResumed`, in the record's
 * `retryNodeIds` order, each counting the node's attempts interrupted with no `RunStopped`. The run
 * boundary publishes them right after the `RunResumed` observers.
 *
 * @param state The state `record` was committed to.
 * @param record The committed `RunResumed` record.
 */
export function resumedAttemptEvents(state: RunState, record: RunResumedRecord): NodeAttemptResumed[] {
  return record.retryNodeIds.map(
    (nodeId) => new NodeAttemptResumed({ nodeId, interrupted: state.interruptedAttempts.get(nodeId) ?? 0 }),
  );
}

/**
 * Drive a prepared append-only run frontier without owning its log.
 *
 * `run()` resolves to the observers returned after each record was appended and its reducer plan
 * applied; `onEvent` sees each first. Stream chunks reach only `onStream`. Evidence and binding
 * capture follow the recorded `RunState.learnable`. A node declaring `needsSteps` gets a
 * `NodeSteps` snapshot taken right after its start; each step it records is appended and applied
 * before `record` returns, with no observer, and the capability closes once the evaluation's
 * outcome is decided, as do the node's event sinks. Once the run stops, `record` throws the node
 * signal's reason and appends nothing, and the sinks drop what they get, except a settled step
 * (`{ settled: true }`) and a `ToolCallFinished` while the run has no terminal result: the node
 * vouches that the call settled while its signal was live (see `NodeSteps.record`).
 *
 * Each node runs under its own signal (its `caps.signal`, also the memory opener's), aborted with
 * the run signal's reason and unlinked from it when the node settles. A host stop (the `signal`
 * option aborting) or a failing host callback is recorded as one `RunStopped` naming the
 * in-flight nodes, appended when the stop is requested and before the drain; any other host
 * fault, and a `RunFailed`, ends the run with no such record. A capability fault (a refused step,
 * an opener fault) ends it at once, even when the node catches it.
 *
 * Each evaluation runs in its own async resource, started in the `run()` caller's async context: an
 * `AsyncLocalStorage` store a node sets stays with that node. Host callbacks the dispatcher calls
 * (`onEvent` for durable observers) run in the dispatcher's context, so a store a host sets there
 * reaches later nodes.
 *
 * A `RunLog`'s `append` must not call back into the run: the engine appends synchronously, so a
 * callback from inside an append would interleave records.
 */
export class FlowEngine {
  /** The borrowed state; the engine is its only writer while it runs. */
  readonly state: RunState;
  /** The borrowed log; the engine never closes it. */
  readonly log: RunLog;
  readonly growBuilder: GrowBuilder;
  readonly numWorkers: number;
  readonly onEvent: ((event: RunEvent) => unknown) | null;
  readonly onStream: ((chunk: StreamChunk) => unknown) | null;
  readonly memoryContextOpener: MemoryContextOpener | null;
  /** The run's tool set bound to the state's run id and the run signal. */
  readonly tools: ToolSet;
  readonly skills: ReadonlyMap<string, Skill> | null;
  readonly steering: Steering | null;
  /** The run's stop; single-use, so `run()` runs once. */
  readonly #controller = new AbortController();
  /** The host's stop, linked to the run's while `run()` runs. */
  readonly #hostSignal: AbortSignal | null;
  /** The steering nodes receive: the host inbox, whose failures stop the run. */
  readonly #steering: Steering | null;
  /** The error that ended the run, once one did; the first wins. */
  #failure: { readonly error: unknown } | null = null;
  #ran = false;

  /**
   * Borrow a prepared state and log and configure one drive.
   *
   * @throws TypeError when `state` isn't exactly a `RunState`, `log` lacks `append`, `runId` or
   *   `nextSequence`, `growBuilder` or a given callback isn't a function, `numWorkers` isn't a safe
   *   integer, `tools` isn't exactly a `ToolSet`, `skills` isn't a `Map`, `steering` lacks its
   *   methods, or `signal` isn't an `AbortSignal`.
   * @throws RangeError when `numWorkers` is negative.
   */
  constructor(state: RunState, log: RunLog, options: FlowEngineOptions) {
    const { growBuilder, numWorkers, onEvent = null, onStream = null, memoryContextOpener = null } = options;
    const { tools = null, skills = null, steering = null, signal = null } = options;
    if (!isExactly(state, RunState)) throw new TypeError("FlowEngine expects a RunState");
    if (!isRunLog(log)) throw new TypeError("FlowEngine expects a RunLog");
    if (typeof growBuilder !== "function") throw new TypeError("grow_builder must be callable");
    if (typeof numWorkers !== "number" || !Number.isSafeInteger(numWorkers)) {
      throw new TypeError("num_workers must be a non-negative integer");
    }
    if (numWorkers < 0) throw new RangeError("num_workers must be a non-negative integer");
    if (onEvent !== null && typeof onEvent !== "function") throw new TypeError("on_event must be callable or null");
    if (onStream !== null && typeof onStream !== "function") throw new TypeError("on_stream must be callable or null");
    if (memoryContextOpener !== null && typeof memoryContextOpener !== "function") {
      throw new TypeError("memory_context_opener must be callable or null");
    }
    if (tools !== null && !isExactly(tools, ToolSet)) throw new TypeError("tools must be a ToolSet or null");
    if (skills !== null && !(skills instanceof Map)) throw new TypeError("skills must be a map or null");
    if (signal !== null && !(signal instanceof AbortSignal)) throw new TypeError("signal must be an AbortSignal or null");
    this.state = state;
    this.log = log;
    this.growBuilder = growBuilder;
    this.numWorkers = numWorkers;
    this.onEvent = onEvent;
    this.onStream = onStream;
    this.memoryContextOpener = memoryContextOpener;
    this.tools = (tools ?? new ToolSet()).forRun(state.runId ?? "", { signal: this.#controller.signal });
    this.skills = skills;
    this.steering = requireSteering(steering);
    this.#hostSignal = signal;
    const inbox = this.steering;
    this.#steering =
      inbox === null
        ? null
        : {
            interruptSignal: (nodeId) => this.#callHost((id: string) => inbox.interruptSignal(id), nodeId),
            drain: (nodeId) => this.#callHost((id: string) => inbox.drain(id), nodeId),
          };
  }

  /** @internal The run signal: aborted when the run fails, stops, or a host fault ends it. */
  get signal(): AbortSignal {
    return this.#controller.signal;
  }

  /**
   * Drive newly ready work to a terminal result.
   *
   * A node failure is a result: `run()` resolves with its `RunFailed`, whatever stops arrive while
   * it drains. A stop or any other error aborts the run signal, waits for every running node, and
   * rejects `run()` with the error that initiated it, never a sibling's derived abort.
   *
   * @returns The new post-durable observers, in order; none when the state is already terminal.
   * @throws Error (as a rejection) on a second call, or when nothing is ready and nothing is in
   *   flight before a terminal result.
   * @throws RangeError when the state is uninitialized, its run id, sequence or worker count
   *   differs from the log's or the engine's, or it has unfinished starts.
   * @throws The host signal's reason when it aborts (before any start when it is already aborted).
   * @throws The error of a failing host callback (`onEvent`, `onStream` or a steering method
   *   throwing, or a promise one returned rejecting), or of a host fault: a log or reducer refusal,
   *   or a fault `evalNode` rejects with.
   */
  async run(): Promise<readonly RunEvent[]> {
    if (this.#ran) throw new Error("FlowEngine.run() was already called");
    this.#ran = true;
    const host = this.#hostSignal;
    // Nothing is in flight yet, so there is no stop to record.
    if (host?.aborted) throw host.reason;
    this.#validateAuthority();
    if (this.state.result !== null) return [];
    if (this.state.inFlight.size > 0) {
      throw new RangeError("FlowEngine requires unfinished starts to be requeued with RunResumed before execution");
    }
    const onHostAbort = (): void => this.#stop("signal", host!.reason);
    host?.addEventListener("abort", onHostAbort, { once: true });
    try {
      return this.numWorkers === 0 ? await this.#runSerial() : await this.#runPooled();
    } finally {
      host?.removeEventListener("abort", onHostAbort);
    }
  }

  /** Execute each durably started node inline. */
  async #runSerial(): Promise<readonly RunEvent[]> {
    const { state, log } = this;
    const signal = this.#controller.signal;
    const events: RunEvent[] = [];
    try {
      while (state.result === null) {
        signal.throwIfAborted();
        if (state.ready.length === 0) throw new Error("FlowEngine reached nonterminal quiescence");
        const nodeId = state.ready[0]!;
        const started = commitPrepared(log, state, prepareNodeStarted(state, nodeId));
        const invocation = new ConversationInvocation(state.runId!, state.acceptedStarts.get(nodeId)!);
        this.#publish(started, events);
        signal.throwIfAborted();
        const evaluated = await this.#startEvaluation(nodeId, invocation);
        signal.throwIfAborted();
        const settled = this.#commitOutcome(nodeId, evaluated);
        // The commit may have set the result, which the loop condition narrowed to `null`.
        if ((state.result as RunState["result"]) instanceof RunFailed) this.#controller.abort();
        this.#publish(settled, events);
      }
    } catch (error) {
      throw this.#fail(error);
    }
    return events;
  }

  /**
   * Keep at most `numWorkers` evaluations in flight and commit their outcomes in settle order.
   *
   * A `RunFailed` is published before the drain, so a host learns of it while siblings still
   * block. Any other error aborts the run signal, waits for every active worker and rethrows the
   * initiating error.
   */
  async #runPooled(): Promise<readonly RunEvent[]> {
    const { state, log } = this;
    const signal = this.#controller.signal;
    const events: RunEvent[] = [];
    const completions = new CompletionQueue();
    const active = new Map<string, Promise<void>>();
    try {
      while (state.result === null) {
        signal.throwIfAborted();
        while (state.ready.length > 0 && active.size < this.numWorkers) {
          // A callback failing during the previous node's synchronous start stops the run.
          signal.throwIfAborted();
          const nodeId = state.ready[0]!;
          const started = commitPrepared(log, state, prepareNodeStarted(state, nodeId));
          const invocation = new ConversationInvocation(state.runId!, state.acceptedStarts.get(nodeId)!);
          // The start is durable and published before its worker starts.
          this.#publish(started, events);
          signal.throwIfAborted();
          const worker = this.#startEvaluation(nodeId, invocation).then(
            (outcome) => completions.push({ nodeId, outcome }),
            (error: unknown) => completions.push({ nodeId, error }),
          );
          active.set(nodeId, worker);
        }
        if (active.size === 0) throw new Error("FlowEngine reached nonterminal quiescence");
        const completion = await completions.take();
        active.delete(completion.nodeId);
        // An entry taken after the abort is discarded; the abort's error is the run's.
        signal.throwIfAborted();
        if ("error" in completion) throw completion.error;
        const settled = this.#commitOutcome(completion.nodeId, completion.outcome);
        if ((state.result as RunState["result"]) instanceof RunFailed) {
          this.#controller.abort();
          this.#publish(settled, events);
          await Promise.all(active.values());
          return events;
        }
        this.#publish(settled, events);
      }
    } catch (error) {
      const initiating = this.#fail(error);
      // Workers never reject, and the initiating error is kept first, so the drain can't replace it.
      await Promise.all(active.values());
      throw initiating;
    }
    // A success result means nothing else is ready or in flight, so there is nothing to drain.
    return events;
  }

  /**
   * Return the steps of a just-started node declaring `needsSteps`, under its `signal`, else
   * `null`. The evaluator closes them when the evaluation's outcome is decided.
   */
  #nodeSteps(nodeId: string, signal: AbortSignal): NodeSteps | null {
    const { state, log } = this;
    if (state.flow.nodes.get(nodeId)!.needsSteps !== true) return null;
    // The recorded steps are frozen; the node gets copies it may change.
    const prior = structuredClone(state.nodeSteps.get(nodeId) ?? []) as ValueRecord[];
    return new NodeSteps(state.nodeStarts.get(nodeId)!, prior, (step, options) => {
      // A stopping run takes only a settled call's step, until it has a result; the stop is
      // control flow, not a host fault.
      if (signal.aborted && !(options?.settled === true && state.result === null)) throw signal.reason;
      try {
        commitPrepared(log, state, prepareNodeStep(state, nodeId, step));
      } catch (error) {
        // A refused or unrecorded step is a host fault, never the node's failure.
        throw new CapabilityFault(error);
      }
    });
  }

  /** Prepare one outcome's record, append it and apply its plan, returning its observers. */
  #commitOutcome(nodeId: string, evaluated: EvaluatedOutcome): readonly RunEvent[] {
    if (!isExactly(evaluated, EvaluatedOutcome)) throw new TypeError("FlowEngine outcome must be an EvaluatedOutcome");
    const { state } = this;
    const { outcome } = evaluated;
    const carriers: OutcomeCarriers = {
      evidence: evaluated.evidence,
      // A learnable state evaluated with binding capture on, so both captures are present.
      forwardCapture: state.learnable ? captureBindings(evaluated.bindings!, evaluated.backwardCapture!) : null,
    };
    let prepared: PreparedTransition;
    if (outcome instanceof EvaluationFailure) {
      prepared = this.#prepareFailure(nodeId, outcome.failure, carriers);
    } else if (outcome instanceof Output) {
      try {
        prepared = prepareNodeSucceeded(state, nodeId, outcome.value, { commitAs: outcome.commitAs, ...carriers });
      } catch (error) {
        if (!(error instanceof NodeDataError)) throw error;
        prepared = this.#prepareContractFailure(nodeId, "contract.output", error.original, error.origin, carriers);
      }
    } else if (outcome instanceof Route) {
      try {
        prepared = prepareNodeRouted(state, nodeId, outcome.handle, carriers);
      } catch (error) {
        if (!(error instanceof NodeDataError)) throw error;
        prepared = this.#prepareContractFailure(nodeId, "contract.route", error.original, error.origin, carriers);
      }
    } else if (outcome instanceof Grow) {
      try {
        prepared = prepareExpanded(state, nodeId, outcome, this.growBuilder, carriers);
      } catch (error) {
        if (!(error instanceof GrowthRejected)) throw error;
        prepared = this.#prepareContractFailure(nodeId, "contract.growth", error.original, null, carriers);
      }
    } else {
      throw new TypeError(`raw evaluator returned an unsupported outcome ${jsTypeName(outcome)}`);
    }
    return commitPrepared(this.log, state, prepared);
  }

  /**
   * Prepare one canonical failure of `nodeId`.
   *
   * @throws Error when the failure names another node.
   */
  #prepareFailure(nodeId: string, failure: Failure, carriers: OutcomeCarriers): PreparedTransition {
    if (failure.nodeId !== nodeId) {
      const found = failure.nodeId === null ? "null" : reprText(failure.nodeId);
      throw new Error(`raw evaluation failure node identity mismatch: expected ${reprText(nodeId)}, found ${found}`);
    }
    return prepareNodeFailed(this.state, failure, carriers);
  }

  /** Prepare the failure of one outcome the reducer's contract refused. */
  #prepareContractFailure(
    nodeId: string,
    code: FailureCode,
    original: unknown,
    origin: OriginRef | null,
    carriers: OutcomeCarriers,
  ): PreparedTransition {
    const node = this.state.flow.nodes.get(nodeId)!;
    return this.#prepareFailure(nodeId, evaluationFailure(node, code, original, origin).failure, carriers);
  }

  /**
   * Start one started node's evaluation in a fresh async resource, so it begins in the `run()`
   * caller's async context and a store the node sets (`AsyncLocalStorage.enterWith`, even in its
   * synchronous part) stays in its own chain, never reaching the dispatcher or another node.
   */
  #startEvaluation(nodeId: string, invocation: ConversationInvocation): Promise<EvaluatedOutcome> {
    return new AsyncResource("yian.node").runInAsyncScope(() => this.#evaluate(nodeId, invocation));
  }

  /**
   * Evaluate one started node under its own signal, aborted with the run's reason when the run
   * signal aborts; the run state changes only through its recorded steps.
   */
  async #evaluate(nodeId: string, invocation: ConversationInvocation): Promise<EvaluatedOutcome> {
    const { state, onEvent, onStream } = this;
    const trace = state.trace;
    // A per-node controller, linked while the node runs, so the listeners node code leaves on its
    // signal die with the node instead of piling up on the long-lived run signal.
    const runSignal = this.#controller.signal;
    const nodeController = new AbortController();
    // Both callers check the run signal right before, so it is not aborted yet.
    const relay = (): void => nodeController.abort(runSignal.reason);
    runSignal.addEventListener("abort", relay, { once: true });
    try {
      return await evalNode(state.flow.nodes.get(nodeId)!, state.flow, state.pool, {
        wiringOverride: state.wiringOverrides.get(nodeId) ?? null,
        signal: nodeController.signal,
        emitStream: onStream === null ? null : (chunk) => this.#emit(onStream, chunk),
        emitEvent: onEvent === null ? null : (event) => this.#emit(onEvent, event),
        captureEvidence: state.learnable,
        captureBindings: state.learnable,
        effectiveLlmByStaticPath: state.effectiveLlmByStaticPath,
        sourceResolver: trace === null ? null : (path) => trace.resolve(path),
        memoryContextOpener: this.memoryContextOpener,
        memoryInvocation: invocation,
        tools: this.tools,
        skills: this.skills,
        steering: this.#steering,
        steps: this.#nodeSteps(nodeId, nodeController.signal),
        // A capability fault ends the run now, not when the node settles, so a node that catches
        // it can't let a later stop replace it. A failing host callback has stopped the run
        // already, so this is a no-op for it.
        onFault: (original) => void this.#fail(original),
      });
    } finally {
      runSignal.removeEventListener("abort", relay);
    }
  }

  /**
   * Pass one transient chunk or in-loop event to its callback unless the run is stopping; a
   * stopping run with no result still passes a settled call's `ToolCallFinished`.
   */
  #emit<T>(callback: (item: T) => unknown, item: T): void {
    if (this.#controller.signal.aborted && !(item instanceof ToolCallFinished && this.state.result === null)) return;
    this.#callHost(callback, item);
  }

  /** Pass each post-durable observer to `onEvent`, then add it to the run's events. */
  #publish(observers: readonly RunEvent[], events: RunEvent[]): void {
    for (const observer of observers) {
      if (this.onEvent !== null) this.#callHost(this.onEvent, observer);
      events.push(observer);
    }
  }

  /**
   * Call one host callback and return its answer. A throw, or a rejection of the promise it
   * returned, stops the run with that error; the throw is rethrown.
   */
  #callHost<A, R>(callback: (arg: A) => R, arg: A): R {
    let returned: R;
    try {
      returned = callback(arg);
    } catch (error) {
      this.#stop("callback", error);
      throw error;
    }
    if (returned instanceof Promise) returned.then(undefined, (error: unknown) => this.#stop("callback", error));
    return returned;
  }

  /**
   * Stop the run for a host abort or a failing host callback: keep `error` as the initiating
   * error, append one `RunStopped` naming the in-flight nodes unless this segment recorded one
   * already, then abort the run signal with `error`. Does nothing once the run signal is aborted,
   * for any cause, or once the run has a result. Never throws: a refused record is noted on
   * `error`.
   */
  #stop(reason: RunStopReason, error: unknown): void {
    const { state } = this;
    if (this.#failure !== null || this.#controller.signal.aborted || state.result !== null) return;
    // Kept before the append, so the error is first-wins and a call re-entering during it is a no-op.
    this.#failure = { error };
    if (state.lastStopSequence === null) {
      try {
        commitPrepared(this.log, state, prepareRunStopped(state, reason, markerMessage(error)));
      } catch (refusal) {
        addNote(error, () => `RunStopped was not recorded: ${markerMessage(refusal)}`);
      }
    }
    this.#controller.abort(error);
  }

  /**
   * End the run for `error` with no stop record: keep it unless an earlier error initiated the
   * end, abort the run signal, and return the initiating error.
   */
  #fail(error: unknown): unknown {
    this.#failure ??= { error };
    this.#controller.abort(this.#failure.error);
    return this.#failure.error;
  }

  /** Refuse a mismatched identity, sequence or worker authority before execution. */
  #validateAuthority(): void {
    const { state, log } = this;
    if (typeof state.runId !== "string" || !state.runId) {
      throw new RangeError("FlowEngine requires an initialized RunState");
    }
    if (state.runId !== log.runId) {
      throw new RangeError(`run-state/log run_id mismatch: state ${reprText(state.runId)}, log ${reprText(log.runId)}`);
    }
    if (state.nextSequence !== log.nextSequence) {
      throw new RangeError(`run-state/log sequence mismatch: state ${state.nextSequence}, log ${log.nextSequence}`);
    }
    if (state.numWorkers !== this.numWorkers) {
      throw new RangeError(`run-state/engine worker mismatch: state ${state.numWorkers}, engine ${this.numWorkers}`);
    }
  }
}
