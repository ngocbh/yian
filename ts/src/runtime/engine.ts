/**
 * Append-only flow execution over one prepared run-state/log authority.
 *
 * `FlowEngine` is the sole ready-frontier consumer and state writer. Serial mode evaluates
 * durably started nodes inline. Every observer it returns follows its record's append and reducer
 * application. Stream chunks are transient callback-only messages and never enter the log.
 *
 * Knows about:   compile/model, diagnostics, events, memory contracts, nodes, runtime eval_node,
 *                log, state and trace, skills, tools.
 * Never imports: the run boundary, compose, or the CLI.
 */

import { reprText } from "../diagnostics.ts";
import type { RunEvent, StreamChunk } from "../events.ts";
import type { Steering } from "../nodes/base.ts";
import { isExactly } from "../nodes/contracts.ts";
import type { Skill } from "../skills/catalog.ts";
import { ToolSet } from "../tools/registry.ts";
import { type MemoryContextOpener, requireSteering } from "./eval_node.ts";
import type { RunLog } from "./log.ts";
import { type GrowBuilder, RunState } from "./state.ts";

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
 * Drive a prepared append-only run frontier without owning its log.
 *
 * `run()` resolves to the observers returned after each record was appended and its reducer plan
 * applied; `onEvent` sees each first. Stream chunks reach only `onStream`. Evidence and binding
 * capture follow the recorded `RunState.learnable`.
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
   * @returns The new post-durable observers, in order; none when the state is already terminal.
   * @throws Error (as a rejection) on a second call.
   * @throws RangeError when the state is uninitialized, its run id, sequence or worker count
   *   differs from the log's or the engine's, or it has unfinished starts.
   */
  async run(): Promise<readonly RunEvent[]> {
    if (this.#ran) throw new Error("FlowEngine.run() was already called");
    this.#ran = true;
    this.#validateAuthority();
    if (this.state.result !== null) return [];
    if (this.state.inFlight.size > 0) {
      throw new RangeError("FlowEngine requires unfinished starts to be requeued with RunResumed before execution");
    }
    throw new Error("FlowEngine execution is not ported yet");
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
