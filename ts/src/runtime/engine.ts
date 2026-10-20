/**
 * Append-only flow execution over one prepared run-state/log authority.
 *
 * `FlowEngine` is the sole ready-frontier consumer and state writer. Serial mode evaluates
 * durably started nodes inline, and every `prepare* -> commitPrepared` runs with no `await`
 * between. Every observer it returns follows its record's append and reducer application. Stream
 * chunks are transient callback-only messages and never enter the log.
 *
 * Knows about:   compile/model, diagnostics, events, memory contracts, nodes, runtime eval_node,
 *                log, state and trace, skills, tools.
 * Never imports: the run boundary, compose, or the CLI.
 */

import { GrowthRejected } from "../compile/model.ts";
import { type Failure, type FailureCode, type OriginRef, reprText } from "../diagnostics.ts";
import { type RunEvent, RunFailed, type StreamChunk } from "../events.ts";
import { ConversationInvocation } from "../memory/contracts.ts";
import { CapabilityFault, Grow, Output, Route, type Steering } from "../nodes/base.ts";
import { isExactly, NodeSteps } from "../nodes/contracts.ts";
import type { Skill } from "../skills/catalog.ts";
import { ToolSet } from "../tools/registry.ts";
import { jsTypeName, type ValueRecord } from "../typesys/values.ts";
import {
  EvaluatedOutcome,
  EvaluationFailure,
  evalNode,
  evaluationFailure,
  type MemoryContextOpener,
  requireSteering,
} from "./eval_node.ts";
import type { OutcomeCarriers, RunLog } from "./log.ts";
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
   * Called synchronously and never awaited; a promise it returns that rejects ends the run with
   * that error.
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
  /** The run's host inbox, passed to nodes declaring `needsSteering`. */
  steering?: Steering | null;
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
 * One start's node steps: `steps.record` commits through the engine until `close`, after which a
 * call throws without reaching the log, so a node that keeps its `NodeSteps` can't write later.
 */
class StepRecorder {
  readonly steps: NodeSteps;
  #closed = false;

  /** @param commit Appends and applies one step, throwing its faults as capability faults. */
  constructor(starts: number, prior: readonly ValueRecord[], commit: (step: ValueRecord) => void) {
    this.steps = new NodeSteps(starts, prior, (step) => {
      if (this.#closed) throw new CapabilityFault(new Error("node steps capability is closed"));
      commit(step);
    });
  }

  /** End the capability; every later call throws. */
  close(): void {
    this.#closed = true;
  }
}

/**
 * Drive a prepared append-only run frontier without owning its log.
 *
 * `run()` resolves to the observers returned after each record was appended and its reducer plan
 * applied; `onEvent` sees each first. Stream chunks reach only `onStream`. Evidence and binding
 * capture follow the recorded `RunState.learnable`. A node declaring `needsSteps` gets a
 * `NodeSteps` snapshot taken right after its start; each step it records is appended and applied
 * before `record` returns, with no observer, and its recorder closes when the evaluation ends.
 * Once the run stops, `record` throws the run signal's reason and appends nothing.
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
  #ran = false;

  /**
   * Borrow a prepared state and log and configure one drive.
   *
   * @throws TypeError when `state` isn't exactly a `RunState`, `log` lacks `append`, `runId` or
   *   `nextSequence`, `growBuilder` or a given callback isn't a function, `numWorkers` isn't a safe
   *   integer, `tools` isn't exactly a `ToolSet`, `skills` isn't a `Map`, or `steering` lacks its
   *   methods.
   * @throws RangeError when `numWorkers` is negative.
   */
  constructor(state: RunState, log: RunLog, options: FlowEngineOptions) {
    const { growBuilder, numWorkers, onEvent = null, onStream = null, memoryContextOpener = null } = options;
    const { tools = null, skills = null, steering = null } = options;
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
  }

  /** @internal The run signal: aborted when the run fails or an error ends it. */
  get signal(): AbortSignal {
    return this.#controller.signal;
  }

  /**
   * Drive newly ready work to a terminal result.
   *
   * A node failure is a result: `run()` resolves with its `RunFailed`. Any other error aborts the
   * run signal and rejects `run()`.
   *
   * @returns The new post-durable observers, in order; none when the state is already terminal.
   * @throws Error (as a rejection) on a second call, when nothing is ready and nothing is in
   *   flight before a terminal result, or for a pooled engine (`numWorkers` above zero), which is
   *   not ported yet.
   * @throws RangeError when the state is uninitialized, its run id, sequence or worker count
   *   differs from the log's or the engine's, or it has unfinished starts.
   * @throws The error of a host fault: a throwing `onEvent`, a callback's rejected promise, a log
   *   or reducer refusal, or a fault `evalNode` rejects with.
   */
  async run(): Promise<readonly RunEvent[]> {
    if (this.#ran) throw new Error("FlowEngine.run() was already called");
    this.#ran = true;
    this.#validateAuthority();
    if (this.state.result !== null) return [];
    if (this.state.inFlight.size > 0) {
      throw new RangeError("FlowEngine requires unfinished starts to be requeued with RunResumed before execution");
    }
    if (this.numWorkers > 0) throw new Error("FlowEngine pooled execution is not ported yet");
    return await this.#runSerial();
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
        const recorder = this.#stepRecorder(nodeId);
        this.#publish(started, events);
        signal.throwIfAborted();
        const evaluated = await this.#evaluate(nodeId, invocation, recorder);
        signal.throwIfAborted();
        const settled = this.#commitOutcome(nodeId, evaluated);
        // The commit may have set the result, which the loop condition narrowed to `null`.
        if ((state.result as RunState["result"]) instanceof RunFailed) this.#controller.abort();
        this.#publish(settled, events);
      }
    } catch (error) {
      this.#controller.abort(error);
      throw error;
    }
    return events;
  }

  /** Return the steps recorder of a just-started node declaring `needsSteps`, else `null`. */
  #stepRecorder(nodeId: string): StepRecorder | null {
    const { state, log } = this;
    if (state.flow.nodes.get(nodeId)!.needsSteps !== true) return null;
    // The recorded steps are frozen; the node gets copies it may change.
    const prior = structuredClone(state.nodeSteps.get(nodeId) ?? []) as ValueRecord[];
    return new StepRecorder(state.nodeStarts.get(nodeId)!, prior, (step) => {
      // A stopping run takes no more steps; the stop is control flow, not a host fault.
      this.#controller.signal.throwIfAborted();
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
   * Evaluate one started node; the run state changes only through its recorded steps. `recorder`
   * closes when the evaluation ends.
   */
  async #evaluate(
    nodeId: string,
    invocation: ConversationInvocation,
    recorder: StepRecorder | null,
  ): Promise<EvaluatedOutcome> {
    const { state, onEvent, onStream } = this;
    const trace = state.trace;
    try {
      return await evalNode(state.flow.nodes.get(nodeId)!, state.flow, state.pool, {
        wiringOverride: state.wiringOverrides.get(nodeId) ?? null,
        signal: this.#controller.signal,
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
        steering: this.steering,
        steps: recorder?.steps ?? null,
      });
    } finally {
      recorder?.close();
    }
  }

  /** Pass one transient chunk or in-loop event to its callback unless the run is stopping. */
  #emit<T>(callback: (item: T) => unknown, item: T): void {
    if (!this.#controller.signal.aborted) this.#callHost(callback, item);
  }

  /** Pass each post-durable observer to `onEvent`, then add it to the run's events. */
  #publish(observers: readonly RunEvent[], events: RunEvent[]): void {
    for (const observer of observers) {
      if (this.onEvent !== null) this.#callHost(this.onEvent, observer);
      events.push(observer);
    }
  }

  /** Call one host callback; a promise it returns that rejects ends the run with that error. */
  #callHost<T>(callback: (item: T) => unknown, item: T): void {
    const returned = callback(item);
    if (returned instanceof Promise) {
      returned.then(undefined, (error: unknown) => {
        // Nothing is left to stop once the run has a result; after an error the run is already
        // aborted, and a second abort is a no-op.
        if (this.state.result === null) this.#controller.abort(error);
      });
    }
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
