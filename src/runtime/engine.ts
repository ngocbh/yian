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
 * A node can end its evaluation waiting (`Pending`): on host calls, or on a take or sleep the
 * engine decides when it commits the outcome. The dispatcher wakes a waiting node whose calls are
 * settled or due, oldest first, before it starts a ready one; pooled mode keeps one scheduler
 * timer for the earliest deadline. At quiescence with nodes waiting it records `run_waiting` (or
 * `run_failed` once END can't be reached) and `run()` resolves. A host's send, complete, close or
 * cancel lands synchronously between two awaits through `deliver`.
 *
 * Knows about: compile/model, diagnostics, events, llm_clients (types), nodes, platform (the
 *              scheduler), runtime eval_node, live, log, state and waits, skills, tools, typesys.
 */

import { AsyncResource } from "node:async_hooks";

import { GrowthRejected } from "../compile/model.ts";
import { addNote, type Failure, type FailureCode, type OriginRef, reprText } from "../diagnostics.ts";
import {
  type ConsoleLine,
  ModelCallFinished,
  ModelCallStarted,
  NodeAttemptResumed,
  RunCancelled,
  type RunEvent,
  RunFailed,
  type StreamChunk,
  ToolCallFinished,
} from "../events.ts";
import type { ModelAccess } from "../llm_clients/factory.ts";
import {
  CapabilityFault,
  type CompleteResult,
  Grow,
  markerMessage,
  type NodeMemories,
  Output,
  type Pending,
  Route,
  type SteerInbox,
} from "../nodes/base.ts";
import { copyValue, type SourceResolver } from "../nodes/binding.ts";
import { type CallView, isExactly, NodeSteps } from "../nodes/contracts.ts";
import { realScheduler, type Scheduler } from "../platform.ts";
import type { Skill } from "../skills/catalog.ts";
import { toolCallId } from "../tools/context.ts";
import { ToolSet } from "../tools/registry.ts";
import { YianRangeError, YianTypeError } from "../typesys/errors.ts";
import { referenceRoot } from "../typesys/pool.ts";
import {
  type EvaluatedOutcome,
  EvaluationFailure,
  evalNode,
  evaluationFailure,
} from "./eval_node.ts";
import type { LiveChannel } from "./live.ts";
import {
  InterruptedRecord,
  isRunLog,
  type NodeStepRecord,
  recordTimeMs,
  type RunLog,
  type RunRecord,
  type RunResumedRecord,
  type RunStopReason,
} from "./log.ts";
import {
  callResultStep,
  commitPrepared,
  type GrowBuilder,
  isDerivedCommit,
  NodeDataError,
  PreparedTransition,
  type RepeatedSend,
  planExpansion,
  prepareExpandedFromPlan,
  prepareExpansionCommit,
  prepareAppliedStop,
  prepareComplete,
  prepareInboxClosed,
  prepareInterrupt,
  prepareNodeFailed,
  prepareNodeRouted,
  prepareNodeStarted,
  prepareNodeStep,
  prepareMessagesCommit,
  prepareNodeSucceeded,
  prepareNodeWaiting,
  prepareNodeWoken,
  preparePoolCommit,
  prepareRunCancelled,
  prepareRunFailed,
  prepareRunStopped,
  prepareRunWaiting,
  prepareSend,
  prepareTakeCommit,
  prepareWithdraw,
  RunState,
  inboxElementType,
  requireExpansionOutput,
  resolveFromPool,
  stepsWithSteerValues,
} from "./state.ts";
import { availableUnread, decideWait, dueTimeouts, isWakeable, nodeCalls, wakeAt } from "./waits.ts";

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
  /**
   * Takes each line a node's code prints, in place of this process's `console`; called as
   * `onEvent` is, also for a line printed after its node settled.
   */
  onConsole?: ((line: ConsoleLine) => unknown) | null;
  /** The run's tool set, bound to the run id and signal; the global registry when omitted. */
  tools?: ToolSet | null;
  /** The run's memory sessions, passed to nodes declaring `needsMemories`; none when omitted. */
  memories?: NodeMemories | null;
  /** The run's host-supplied skills, passed to nodes declaring `needsSkills`. */
  skills?: ReadonlyMap<string, Skill> | null;
  /** The run's environment and model builder, passed to nodes declaring `needsModels`. */
  models?: ModelAccess | null;
  /** The host's stop: aborting it stops the run with its reason. */
  signal?: AbortSignal | null;
  /** The clock and sleep node waits run on, passed to every node; `realScheduler` when omitted. */
  scheduler?: Scheduler | null;
  /** A draw in [0, 1) for backoff jitter, passed to every node; `Math.random` when omitted. */
  random?: (() => number) | null;
  /** The run's folder (its log's folder), passed to every node; `null` for a run without a log. */
  runDir?: string | null;
  /**
   * Whether the run has a durable log a host can answer pending calls in (`NodeSteps.durable`);
   * `true` when omitted. A run without a log passes `false`, so a node's `pending()` fails it.
   */
  durable?: boolean;
  /**
   * Gets each AGENT model call's start, chunks and end, and drops a node's calls when the node
   * settles; the run boundary hands it the appended records itself (`LiveChannel.wrap`).
   */
  live?: LiveChannel | null;
}

/** One settled worker: the outcome its evaluation resolved to, or the error it rejected with. */
type Completion =
  | { readonly nodeId: string; readonly outcome: EvaluatedOutcome }
  | { readonly nodeId: string; readonly error: unknown };

/** A wake-up for the dispatcher with no settled worker: a delivery or a timer changed the state. */
type Nudge = { readonly nudge: true };
const NUDGE: Nudge = Object.freeze({ nudge: true });

/** The settled workers in settle order, read by one dispatcher. */
class CompletionQueue {
  readonly #entries: Completion[] = [];
  /** The next unread entry; read entries are dropped in batches so taking never shifts the array. */
  #head = 0;
  #waiter: ((entry: Completion | Nudge) => void) | null = null;
  /** Whether a nudge waits for the next `take`; at most one is queued. */
  #nudged = false;

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

  /** Wake the dispatcher: hand a waiting `take` a nudge, or queue one for the next. */
  nudge(): void {
    const waiter = this.#waiter;
    if (waiter === null) {
      this.#nudged = true;
      return;
    }
    this.#waiter = null;
    waiter(NUDGE);
  }

  /**
   * Return a queued nudge, else the oldest unread entry, or a promise of the next entry or nudge
   * to arrive. At most one `take` may be pending: the queue has a single reader, the dispatcher.
   *
   * @throws Error when another `take` is still pending.
   */
  take(): Completion | Nudge | Promise<Completion | Nudge> {
    if (this.#waiter !== null) throw new Error("CompletionQueue has a take pending already");
    if (this.#nudged) {
      this.#nudged = false;
      return NUDGE;
    }
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

/**
 * Return one `NodeAttemptResumed` per retried node of a committed `RunResumed`, in the record's
 * `retryNodeIds` order, each counting the node's crashed attempts (no `RunStopped` covering them).
 * The run boundary publishes them right after the `RunResumed` observers.
 *
 * @param state The state `record` was committed to.
 * @param record The committed `RunResumed` record.
 */
export function resumedAttemptEvents(state: RunState, record: RunResumedRecord): NodeAttemptResumed[] {
  return record.retryNodeIds.map(
    (nodeId) => new NodeAttemptResumed({ nodeId, crashed: state.crashedAttempts.get(nodeId) ?? 0 }),
  );
}

/**
 * One host delivery: a send, a withdraw of an inbox's unread sends, a completed call, an inbox
 * close, an interrupt or a cancel. An interrupt's `running` are the in-flight readers of its
 * inbox, which only the engine knows; absent offline: then any in-flight node counts as an open
 * reader for the idle rule.
 */
export type DeliveryOp =
  | { readonly kind: "send"; readonly inbox: string; readonly event: unknown; readonly key?: string | null }
  | { readonly kind: "withdraw"; readonly inbox: string }
  | { readonly kind: "complete"; readonly callId: string; readonly result: CompleteResult }
  | { readonly kind: "close"; readonly inbox: string }
  | { readonly kind: "interrupt"; readonly inbox: string; readonly running?: readonly string[] }
  | { readonly kind: "cancel" };

/**
 * @internal Prepare `op` on `state`; the engine and the run boundary share it.
 *
 * @returns null for a no-op (nothing to withdraw, an identical repeat complete, closing a closed
 *   inbox, or an interrupt while the run waits for the inbox's next message or is already
 *   stopping), and the seq of the first send for a repeat of a keyed send.
 * @throws DeliveryRefused when the run refuses it.
 * @throws TypeError when `op` is no delivery.
 */
export function prepareDelivery(state: RunState, op: DeliveryOp): PreparedTransition | RepeatedSend | null {
  switch (op.kind) {
    case "send":
      return prepareSend(state, op.inbox, op.event, op.key ?? null);
    case "withdraw":
      return prepareWithdraw(state, op.inbox);
    case "complete":
      return prepareComplete(state, op.callId, op.result);
    case "close":
      return prepareInboxClosed(state, op.inbox);
    case "interrupt":
      return prepareInterrupt(state, op.inbox, op.running ?? null);
    case "cancel":
      return prepareRunCancelled(state);
    default: {
      const unknown: never = op;
      const kind = (unknown as { readonly kind: unknown }).kind;
      throw new YianTypeError(`a delivery's kind must be send, withdraw, complete, close, interrupt or cancel, got ${reprText(String(kind))}`);
    }
  }
}

/** The node's calls as its evaluation sees them: state, closing reason, deadline and answer. */
function callViews(state: RunState, nodeId: string): CallView[] {
  return nodeCalls(state.waits, nodeId).map((call) => {
    const answer = callResultStep(state, call.callId);
    return {
      callId: call.callId,
      tool: call.tool,
      resultKey: call.resultKey,
      state: call.state,
      closedReason: call.closedReason,
      result: answer === null ? null : structuredClone(answer),
      deadline: call.deadline,
    };
  });
}

/**
 * Drive a prepared append-only run frontier without owning its log.
 *
 * `run()` resolves to the observers returned after each record was appended and its reducer plan
 * applied; `onEvent` sees each first. Stream chunks reach only `onStream`. A `live` channel gets
 * each AGENT model call's start (from `ModelCallStarted`), text chunks and end (from
 * `ModelCallFinished`), none once the run stops, and a node's calls are dropped from it when the
 * node settles; the appended records reach it through the log the run boundary wraps. A derived
 * node's bindings are captured, for the sources its outcome points at. A node declaring
 * `needsSteps` gets a `NodeSteps` snapshot taken right after its start; each step it records is
 * appended and applied before `record` returns, with no observer but the `CallClosed` of a call a
 * `pending` step registers already closed (its node holds an interrupt), and the capability closes
 * once the evaluation's outcome is decided, as do the node's event sinks. Once the run stops,
 * `record` throws the node signal's reason and appends nothing, and the sinks drop what they get,
 * except a settled step (`{ settled: true }`) and a `ToolCallFinished` while the run has no
 * terminal result: the node vouches that the call settled while its signal was live (see
 * `NodeSteps.record`).
 *
 * Each node runs under its own signal (its `caps.signal`), aborted with the run signal's reason and
 * unlinked from it when the node settles. A host stop (the `signal` option aborting) or a failing
 * host callback is recorded as one `RunStopped` naming the in-flight nodes, appended when the stop
 * is requested and before the drain; any other host fault, and a `RunFailed`, ends the run with no
 * such record. A capability fault (a refused step, an opener fault) ends it at once, even when the
 * node catches it.
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
  readonly onConsole: ((line: ConsoleLine) => unknown) | null;
  /** The run's tool set bound to the state's run id and the run signal. */
  readonly tools: ToolSet;
  /** The run's sessions on its memories, which nodes declaring `needsMemories` get; `null` for none. */
  readonly memories: NodeMemories | null;
  readonly skills: ReadonlyMap<string, Skill> | null;
  readonly scheduler: Scheduler | null;
  readonly random: (() => number) | null;
  readonly models: ModelAccess | null;
  readonly runDir: string | null;
  readonly live: LiveChannel | null;
  /** The run's stop; single-use, so `run()` runs once. */
  readonly #controller = new AbortController();
  /** The host's stop, linked to the run's while `run()` runs. */
  readonly #hostSignal: AbortSignal | null;
  /** Whether the run has a durable log a host can answer pending calls in. */
  readonly #durable: boolean;
  /** The error that ended the run, once one did; the first wins. */
  #failure: { readonly error: unknown } | null = null;
  /**
   * The pool names each derived node in flight read, to the seq of the `pool_commit` that wrote
   * each: its commit's `from`. Taken when the node's outcome is committed.
   */
  readonly #reads = new Map<string, Map<string, number>>();
  /** The steer inbox each running node opened and the controller of its interrupt signal, until its outcome commits. */
  readonly #steerInboxes = new Map<string, { readonly inbox: string; readonly controller: AbortController }>();
  #ran = false;
  /** Whether a loop is driving the run; a delivery made after it ended publishes without it. */
  #looping = false;
  /** The run's post-durable observers so far, in order. */
  readonly #events: RunEvent[] = [];
  /** The pooled dispatcher's queue while it runs, so a delivery can wake it. */
  #queue: CompletionQueue | null = null;
  /** The run signal's reason once a host cancels the run. */
  readonly #cancelled = new Error("run cancelled");
  /** The pooled dispatcher's one scheduler sleep, until the earliest open deadline. */
  #timer: { readonly at: string; readonly controller: AbortController } | null = null;

  /**
   * Borrow a prepared state and log and configure one drive. The run boundary checks the options.
   *
   * @throws TypeError when `state` isn't exactly a `RunState`, or `log` lacks `append`, `runId` or
   *   `nextSequence`.
   */
  constructor(state: RunState, log: RunLog, options: FlowEngineOptions) {
    const { growBuilder, numWorkers, onEvent = null, onStream = null, onConsole = null } = options;
    const { tools = null, skills = null, models = null, signal = null, scheduler = null, random = null } =
      options;
    const { memories = null } = options;
    const { runDir = null, live = null } = options;
    if (!isExactly(state, RunState)) throw new YianTypeError("FlowEngine expects a RunState");
    if (!isRunLog(log)) throw new YianTypeError("FlowEngine expects a RunLog");
    this.state = state;
    this.log = log;
    this.growBuilder = growBuilder;
    this.numWorkers = numWorkers;
    this.onEvent = onEvent;
    this.onStream = onStream;
    this.onConsole = onConsole;
    this.tools = (tools ?? new ToolSet()).forRun(state.runId ?? "", { signal: this.#controller.signal });
    this.memories = memories;
    this.skills = skills;
    this.scheduler = scheduler;
    this.random = random;
    this.models = models;
    this.runDir = runDir;
    this.live = live;
    this.#hostSignal = signal;
    this.#durable = options.durable ?? true;
  }

  /**
   * @internal Whether the loop is driving the run, so a delivery now is read by it: from `run()`
   * until the loop decides to end (at a terminal result or quiescence) or rejects.
   */
  get looping(): boolean {
    return this.#looping;
  }

  /** @internal The run signal: aborted when the run fails, stops, or a host fault ends it. */
  get signal(): AbortSignal {
    return this.#controller.signal;
  }

  /**
   * Commit one host delivery synchronously, between two awaits of the loop. A cancel then aborts
   * the run signal: `run()` resolves with the cancelled state once running nodes have drained. An
   * interrupt names the running nodes that opened its inbox as their steer inbox, then the waiting
   * ones, and aborts the running ones' interrupt signals. Then nudge the dispatcher and publish
   * the observers.
   *
   * @returns The appended seq and record (both null for a no-op; for a repeat of a keyed send,
   *   the first send's seq and no record) and the observers.
   * @throws DeliveryRefused when the run refuses it; nothing is appended.
   * @throws Error before `run()` was called.
   * @throws The error of an `onEvent` that throws on these observers. The record is durable by
   *   then. While the loop runs, the throw (or a rejection of the promise it returned) stops the
   *   run; after the loop has ended, a throw is only rethrown and a rejection is ignored.
   */
  deliver(op: DeliveryOp): { readonly seq: number | null; readonly events: readonly RunEvent[]; readonly record: RunRecord | null } {
    if (!this.#ran) throw new Error("FlowEngine.deliver needs run() first");
    const { state, log } = this;
    // Which inbox a running node opened only the engine knows.
    const delivery: DeliveryOp =
      op?.kind === "interrupt"
        ? { ...op, running: [...state.inFlight].filter((nodeId) => this.#steerInboxes.get(nodeId)?.inbox === op.inbox) }
        : op;
    const prepared = prepareDelivery(state, delivery);
    if (prepared === null) return { seq: null, events: [], record: null };
    if (!(prepared instanceof PreparedTransition)) return { seq: prepared.repeatOf, events: [], record: null };
    const seq = prepared.expectedSequence;
    const observers = commitPrepared(log, state, prepared);
    if (state.result instanceof RunCancelled) this.#controller.abort(this.#cancelled);
    // In the same synchronous section as the record, so a running reader sees both at once.
    if (prepared.record instanceof InterruptedRecord) {
      for (const nodeId of prepared.record.nodes) this.#steerInboxes.get(nodeId)?.controller.abort();
    }
    this.#queue?.nudge();
    if (this.#looping) {
      this.#publish(observers, this.#events);
    } else {
      // Nothing reads the run's events once the loop has ended; only the host sees these.
      for (const observer of observers) {
        const returned = this.onEvent?.(observer);
        // The loop has ended, so a rejection has no run to stop.
        if (returned instanceof Promise) returned.then(undefined, () => {});
      }
    }
    return { seq, events: observers, record: prepared.record };
  }

  /**
   * Drive newly ready and wakeable work to a terminal result, or to quiescence with nodes waiting.
   *
   * A node failure is a result: `run()` resolves with its `RunFailed`, whatever stops arrive while
   * it drains. A host cancel (`deliver`) is one too: `run()` resolves once running nodes have
   * drained. At quiescence with waiting nodes it records `run_waiting` (unless that is already the
   * last frontier record) or, when END can't be reached, `run_failed`, and resolves. A stop or any
   * other error aborts the run signal, waits for every running node, and rejects `run()` with the
   * error that initiated it, never a sibling's derived abort.
   *
   * @returns A copy of the run's post-durable observers, in order, including those of deliveries
   *   made while it ran; none when the state is already terminal.
   * @throws Error (as a rejection) on a second call, or when nothing is ready, in flight or
   *   waiting before a terminal result.
   * @throws RangeError when the state is uninitialized, its run id, sequence or worker count
   *   differs from the log's or the engine's, or it has unfinished starts.
   * @throws The host signal's reason when it aborts (before any start when it is already aborted).
   * @throws The error of a failing host callback (`onEvent` or `onStream` throwing, or a promise
   *   one returned rejecting), or of a host fault: a log or reducer refusal,
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
      throw new YianRangeError("FlowEngine requires unfinished starts to be requeued with RunResumed before execution");
    }
    const onHostAbort = (): void => this.#stop("signal", host!.reason);
    host?.addEventListener("abort", onHostAbort, { once: true });
    this.#looping = true;
    try {
      await (this.numWorkers === 0 ? this.#runSerial() : this.#runPooled());
      return [...this.#events];
    } finally {
      this.#looping = false;
      host?.removeEventListener("abort", onHostAbort);
    }
  }

  /** Execute each durably started node inline. */
  async #runSerial(): Promise<void> {
    const { state, log } = this;
    const signal = this.#controller.signal;
    const events = this.#events;
    try {
      while (state.result === null) {
        signal.throwIfAborted();
        const woken = this.#nextWake();
        let nodeId: string;
        if (woken !== null) {
          nodeId = woken.nodeId;
          this.#publish(commitPrepared(log, state, prepareNodeWoken(state, nodeId, woken.timedOut)), events);
        } else if (state.ready.length > 0) {
          nodeId = state.ready[0]!;
          this.#publish(commitPrepared(log, state, prepareNodeStarted(state, nodeId)), events);
        } else {
          if (this.#quiesceEnds(events)) break;
          continue;
        }
        signal.throwIfAborted();
        const evaluated = await this.#startEvaluation(nodeId);
        signal.throwIfAborted();
        const settled = this.#commitOutcome(nodeId, evaluated);
        // The commit may have set the result, which the loop condition narrowed to `null`.
        if ((state.result as RunState["result"]) instanceof RunFailed) this.#controller.abort();
        this.#publish(settled, events);
      }
    } catch (error) {
      // A cancel aborted the evaluation; the run has its result, and nothing else runs.
      if (this.#isCancelled()) return;
      throw this.#fail(error);
    }
  }

  /** Whether a host cancel ended the run: its result is `RunCancelled` and the run signal says so. */
  #isCancelled(): boolean {
    return this.state.result instanceof RunCancelled && this.#controller.signal.reason === this.#cancelled;
  }

  /**
   * Keep at most `numWorkers` evaluations in flight and commit their outcomes in settle order.
   *
   * A `RunFailed` is published before the drain, so a host learns of it while siblings still
   * block. Any other error aborts the run signal, waits for every active worker and rethrows the
   * initiating error.
   */
  async #runPooled(): Promise<void> {
    const { state, log } = this;
    const signal = this.#controller.signal;
    const events = this.#events;
    const completions = new CompletionQueue();
    const active = new Map<string, Promise<void>>();
    // The woken nodes among the active ones.
    const wakes = new Set<string>();
    this.#queue = completions;
    const startWorker = (nodeId: string): void => {
      const worker = this.#startEvaluation(nodeId).then(
        (outcome) => completions.push({ nodeId, outcome }),
        (error: unknown) => completions.push({ nodeId, error }),
      );
      active.set(nodeId, worker);
    };
    try {
      while (state.result === null) {
        signal.throwIfAborted();
        // Whether a wakeable node waits for a worker slot; then waking stops for this turn.
        let blocked = false;
        for (let woken = this.#nextWake(); woken !== null; woken = this.#nextWake()) {
          const { nodeId } = woken;
          const free = state.flow.nodes.get(nodeId)!.wakesWithoutWorker;
          // A woken agent calls a model, so it takes a slot as a start does; waking stays oldest first.
          if (!free && active.size - wakes.size >= this.numWorkers) {
            blocked = true;
            break;
          }
          this.#publish(commitPrepared(log, state, prepareNodeWoken(state, nodeId, woken.timedOut)), events);
          signal.throwIfAborted();
          startWorker(nodeId);
          if (free) wakes.add(nodeId);
        }
        while (state.ready.length > 0 && active.size - wakes.size < this.numWorkers) {
          // A callback failing during the previous node's synchronous start stops the run.
          signal.throwIfAborted();
          const nodeId = state.ready[0]!;
          const started = commitPrepared(log, state, prepareNodeStarted(state, nodeId));
          // The start is durable and published before its worker starts.
          this.#publish(started, events);
          signal.throwIfAborted();
          startWorker(nodeId);
        }
        if (active.size === 0) {
          if (this.#quiesceEnds(events)) break;
          continue;
        }
        this.#armTimer(blocked);
        const completion = await completions.take();
        if ("nudge" in completion) continue;
        active.delete(completion.nodeId);
        wakes.delete(completion.nodeId);
        // An entry taken after the abort is discarded; the abort's error is the run's.
        signal.throwIfAborted();
        if ("error" in completion) throw completion.error;
        const settled = this.#commitOutcome(completion.nodeId, completion.outcome);
        if ((state.result as RunState["result"]) instanceof RunFailed) {
          this.#controller.abort();
          this.#publish(settled, events);
          await Promise.all(active.values());
          return;
        }
        this.#publish(settled, events);
      }
    } catch (error) {
      if (this.#isCancelled()) {
        await Promise.all(active.values());
        return;
      }
      const initiating = this.#fail(error);
      // Workers never reject, and the initiating error is kept first, so the drain can't replace it.
      await Promise.all(active.values());
      throw initiating;
    } finally {
      this.#queue = null;
      this.#disarmTimer();
    }
    // A cancel taken through a nudge ends the loop with workers still running; they drain here.
    await Promise.all(active.values());
  }

  /**
   * Quiesce, and end the loop unless a delivery made from the run_waiting observer left work.
   *
   * @returns Whether the loop ends.
   */
  #quiesceEnds(events: RunEvent[]): boolean {
    const { state } = this;
    this.#quiesce(events);
    if (state.result === null && (state.ready.length > 0 || this.#nextWake() !== null)) return false;
    // Decided with no await, so no delivery can land after this check and before the flag.
    this.#looping = false;
    return true;
  }

  /**
   * End a quiescent segment: `run_failed` when END can't be reached whatever the waiting nodes do,
   * else `run_waiting` unless it is already the last frontier record.
   *
   * @throws Error when no node waits (nonterminal quiescence).
   */
  #quiesce(events: RunEvent[]): void {
    const { state, log } = this;
    if (state.waits.waiting.size === 0) throw new Error("FlowEngine reached nonterminal quiescence");
    if (state.terminalUnreachableFailure !== null) {
      this.#publish(commitPrepared(log, state, prepareRunFailed(state)), events);
      this.#controller.abort();
      return;
    }
    if (state.waits.lastFrontier !== "run_waiting") this.#publish(commitPrepared(log, state, prepareRunWaiting(state)), events);
  }

  /**
   * Return the steps of a just-started node declaring `needsSteps`, under its `signal`, else
   * `null`. The evaluator closes them when the evaluation's outcome is decided.
   */
  #nodeSteps(nodeId: string, signal: AbortSignal): NodeSteps | null {
    const { state, log } = this;
    if (state.flow.nodes.get(nodeId)!.needsSteps !== true) return null;
    // The state owns the recorded steps and appends to them in place; the node gets copies it may
    // change, each steer step's with the values it consumed.
    const prior = stepsWithSteerValues(state, nodeId);
    const record: NodeSteps["record"] = (step, options) => {
      // A stopping run takes only a settled call's step, until it has a result; the stop is
      // control flow, not a host fault.
      if (signal.aborted && !(options?.settled === true && state.result === null)) throw signal.reason;
      let prepared: PreparedTransition;
      let observers: readonly RunEvent[];
      try {
        prepared = prepareNodeStep(state, nodeId, step);
        observers = commitPrepared(log, state, prepared);
      } catch (error) {
        // A refused or unrecorded step is a host fault, never the node's failure.
        throw new CapabilityFault(error);
      }
      // A pending step on a node an interrupt names registers its call closed; the host dismisses it.
      this.#publish(observers, this.#events);
      return (prepared.record as NodeStepRecord).n;
    };
    // The node reads its calls on every evaluation, so a node holding a call reuses it instead of
    // calling its handler again after a wake or a crash.
    const calls = callViews(state, nodeId);
    const stepSeq = (n: number): number => {
      const seq = state.nodeStepSeqs.get(nodeId)?.[n - 1];
      if (seq === undefined) throw new YianRangeError(`node ${reprText(nodeId)} has no step ${String(n)}`);
      return seq;
    };
    return new NodeSteps(
      state.nodeStarts.get(nodeId)!,
      prior,
      record,
      calls,
      state.runDefaults.maxEventBytes,
      this.#durable,
      () => callViews(state, nodeId),
      // A steer read and a host's answer append the node's steps without `record`.
      () => (state.nodeSteps.get(nodeId)?.length ?? 0) + 1,
      // Only a resume that found the node cut off (by a crash or a stop) starts it again; a wake
      // records no start.
      (n) => {
        const seq = stepSeq(n);
        return (state.nodeStartSeqs.get(nodeId) ?? []).filter((start) => start > seq).length;
      },
      stepSeq,
    );
  }

  /**
   * Return the opener of a just-started node's steer inbox when the node declares
   * `readsSteerInbox`, else `null`. Opening registers the node as reading that inbox until its
   * outcome commits, with an interrupt signal that an interrupt of the inbox aborts (aborted at
   * once when the node holds a pending interrupt). A node that opens an inbox while a Stop on it
   * stands is interrupted at once by its own `interrupted` record. A read is zero-wait: in one
   * synchronous section it takes every event a read may take now, appends them as one `steer` step
   * and returns their values, so no delivery lands between reading the queue and the append.
   */
  #steerInbox(nodeId: string, signal: AbortSignal): ((inbox: string) => SteerInbox) | null {
    const { state, log } = this;
    if (state.flow.nodes.get(nodeId)!.readsSteerInbox !== true) return null;
    return (inbox) => {
      if (typeof inbox !== "string" || inboxElementType(state, inbox) === null) {
        throw new CapabilityFault(new YianRangeError(`steer inbox ${reprText(String(inbox))} is no inbox the START declares`));
      }
      this.#applyStop(nodeId, inbox);
      const controller = new AbortController();
      // An interrupt recorded before this evaluation (it waited, or crashed), or the Stop just
      // applied, stands.
      if (state.waits.interrupts.has(nodeId)) controller.abort();
      this.#steerInboxes.set(nodeId, { inbox, controller });
      return {
        inbox,
        interrupt: controller.signal,
        interruptedBy: () => state.waits.interrupts.get(nodeId) ?? null,
        queued: () => availableUnread(state.waits, inbox).length > 0,
        read: () => {
          if (signal.aborted) throw signal.reason;
          const from = [...availableUnread(state.waits, inbox)];
          if (from.length === 0) return null;
          let n: number;
          try {
            // Seqs are ints, which the log keeps apart from floats only as bigints.
            const prepared = prepareNodeStep(state, nodeId, { kind: "steer", inbox, from: from.map(BigInt) });
            commitPrepared(log, state, prepared);
            n = (prepared.record as NodeStepRecord).n;
          } catch (error) {
            // A refused or unrecorded step is a host fault, never the node's failure.
            throw new CapabilityFault(error);
          }
          return { n, from, values: from.map((seq) => copyValue(state.inboxEvents.get(seq)!.toObject())) };
        },
      };
    };
  }

  /**
   * Append the remembered Stop on `inbox` as an `interrupted` record naming `nodeId`, which has just
   * opened it as its steer inbox, and publish what it closes. Nothing when no Stop stands, the run
   * has a result or its signal is aborted (a stop or cancel drains; the window outlives a stop, so
   * the rerun applies it), or the node already holds or honored an interrupt.
   */
  #applyStop(nodeId: string, inbox: string): void {
    const { state, log } = this;
    const w = state.waits;
    if (!w.stops.has(inbox) || state.result !== null || this.#controller.signal.aborted) return;
    if (w.interrupts.has(nodeId) || w._honored.has(nodeId)) return;
    let observers: readonly RunEvent[];
    try {
      // Not prepareInterrupt: the host's rules record nothing while a Stop stands.
      observers = commitPrepared(log, state, prepareAppliedStop(state, inbox, nodeId));
    } catch (error) {
      throw new CapabilityFault(error);
    }
    this.#publish(observers, this.#events);
    // A waiting reader it named is wakeable now.
    this.#queue?.nudge();
  }

  /** Prepare one outcome's record, append it and apply its plan, returning its observers. */
  #commitOutcome(nodeId: string, evaluated: EvaluatedOutcome): readonly RunEvent[] {
    const { state } = this;
    const { outcome } = evaluated;
    const steerInbox = this.#steerInboxes.get(nodeId)?.inbox ?? null;
    this.#steerInboxes.delete(nodeId);
    const reads = this.#reads.get(nodeId) ?? new Map<string, number>();
    this.#reads.delete(nodeId);
    let prepared: PreparedTransition;
    if (outcome instanceof EvaluationFailure) {
      prepared = this.#prepareFailure(nodeId, outcome.failure);
    } else if (outcome instanceof Output && state.pendingCommits.has(nodeId)) {
      // A child START: its expansion committed its value, which its success names.
      requireExpansionOutput(state, nodeId, outcome.value);
      prepared = prepareNodeSucceeded(state, nodeId);
    } else if (outcome instanceof Output) {
      let commit: PreparedTransition;
      try {
        const from = isDerivedCommit(state, nodeId) ? reads : null;
        commit = preparePoolCommit(state, nodeId, outcome.value, { commitAs: outcome.commitAs, from });
      } catch (error) {
        if (!(error instanceof NodeDataError)) throw error;
        const failure = this.#prepareContractFailure(nodeId, "contract.output", error.original, error.origin);
        return commitPrepared(this.log, state, failure);
      }
      // The output, then the node's messages, then the success that accepts both, with nothing
      // between: a crash before the success leaves orphan commits, and the node reruns.
      commitPrepared(this.log, state, commit);
      if (state.flow.nodes.get(nodeId)!.producesMessages) commitPrepared(this.log, state, prepareMessagesCommit(state, nodeId));
      prepared = prepareNodeSucceeded(state, nodeId, { output: outcome.value });
    } else if (outcome instanceof Route) {
      try {
        prepared = prepareNodeRouted(state, nodeId, outcome.handle);
      } catch (error) {
        if (!(error instanceof NodeDataError)) throw error;
        prepared = this.#prepareContractFailure(nodeId, "contract.route", error.original, error.origin);
      }
    } else if (outcome instanceof Grow) {
      try {
        // The child STARTs' values are written before the expansion that admits them: a crash
        // between them leaves orphan commits, and the spawner reruns.
        const planned = planExpansion(state, nodeId, outcome, this.growBuilder);
        for (const [key] of planned.commits) {
          commitPrepared(this.log, state, prepareExpansionCommit(state, nodeId, key, reads));
        }
        prepared = prepareExpandedFromPlan(state, planned);
      } catch (error) {
        if (!(error instanceof GrowthRejected)) throw error;
        prepared = this.#prepareContractFailure(nodeId, "contract.growth", error.original, null);
      }
    } else {
      return this.#commitPending(nodeId, outcome, steerInbox);
    }
    return commitPrepared(this.log, state, prepared);
  }

  /**
   * Commit a Pending outcome. Host calls: `node_waiting` naming them. A take or sleep: decide now
   * and append the result in this synchronous section, so no delivery lands between the decision
   * and its records. A node that opened a steer inbox waits with it (`steerInbox`).
   */
  #commitPending(nodeId: string, outcome: Pending, steerInbox: string | null): readonly RunEvent[] {
    const { state, log } = this;
    if (outcome.wait === null) {
      return commitPrepared(log, state, prepareNodeWaiting(state, nodeId, { callIds: outcome.callIds, steerInbox }));
    }
    const callId = toolCallId(state.runId!, nodeId);
    const decision = decideWait(state.waits, callId, outcome.wait, this.#now());
    if (decision.kind === "take") {
      // The commit and its success go in together: nothing is appended between them.
      commitPrepared(log, state, prepareTakeCommit(state, nodeId, decision.commit));
      return commitPrepared(log, state, prepareNodeSucceeded(state, nodeId));
    }
    if (decision.kind === "slept") {
      let commit: PreparedTransition;
      try {
        commit = preparePoolCommit(state, nodeId, null);
      } catch (error) {
        if (!(error instanceof NodeDataError)) throw error;
        return commitPrepared(log, state, this.#prepareContractFailure(nodeId, "contract.output", error.original, error.origin));
      }
      commitPrepared(log, state, commit);
      return commitPrepared(log, state, prepareNodeSucceeded(state, nodeId));
    }
    // The decision left is to wait.
    return commitPrepared(log, state, prepareNodeWaiting(state, nodeId, { callIds: [callId], waits: [decision.entry], steerInbox }));
  }

  /**
   * Keep one scheduler sleep for the earliest open deadline of a waiting node, at which that node
   * is due; it nudges the dispatcher. `blocked` says a wakeable node waits for a worker slot.
   */
  #armTimer(blocked = false): void {
    const at = wakeAt(this.state.waits);
    if (at === (this.#timer?.at ?? null)) return;
    this.#disarmTimer();
    if (at === null) return;
    const scheduler = this.scheduler ?? realScheduler;
    const now = scheduler.now();
    const ms = recordTimeMs(at, "deadline") - now;
    if (ms <= 0) {
      // A deadline passed since the dispatcher last looked: wake its node now. A due node that
      // waits for a worker slot is looked at again once the next completion frees one, so nudging
      // now would only spin.
      if (!blocked && dueTimeouts(this.state.waits, now).size > 0) this.#queue?.nudge();
      return;
    }
    const timer = { at, controller: new AbortController() };
    this.#timer = timer;
    scheduler.sleep(ms, timer.controller.signal).then(
      () => {
        if (this.#timer === timer) this.#timer = null;
        this.#queue?.nudge();
      },
      // Disarmed.
      () => {},
    );
  }

  /** Drop the dispatcher's scheduler sleep, if any. */
  #disarmTimer(): void {
    this.#timer?.controller.abort();
    this.#timer = null;
  }

  /** The oldest waiting node that is wakeable or has a due call, with its due calls; else null. */
  #nextWake(): { readonly nodeId: string; readonly timedOut: readonly string[] } | null {
    const w = this.state.waits;
    const due = dueTimeouts(w, this.#now());
    for (const nodeId of w.waiting.keys()) {
      const timedOut = due.get(nodeId);
      if (timedOut !== undefined || isWakeable(w, nodeId)) return { nodeId, timedOut: timedOut ?? [] };
    }
    return null;
  }

  /** The engine's clock: the scheduler's, as the run log's record times are. */
  #now(): number {
    return (this.scheduler ?? realScheduler).now();
  }

  /**
   * Prepare one canonical failure of `nodeId`.
   *
   * @throws Error when the failure names another node.
   */
  #prepareFailure(nodeId: string, failure: Failure): PreparedTransition {
    if (failure.nodeId !== nodeId) {
      const found = failure.nodeId === null ? "null" : reprText(failure.nodeId);
      throw new Error(`raw evaluation failure node identity mismatch: expected ${reprText(nodeId)}, found ${found}`);
    }
    return prepareNodeFailed(this.state, failure);
  }

  /** Prepare the failure of one outcome the reducer's contract refused. */
  #prepareContractFailure(
    nodeId: string,
    code: FailureCode,
    original: unknown,
    origin: OriginRef | null,
  ): PreparedTransition {
    const node = this.state.flow.nodes.get(nodeId)!;
    return this.#prepareFailure(nodeId, evaluationFailure(node, code, original, origin).failure);
  }

  /**
   * Start one started node's evaluation in a fresh async resource, so it begins in the `run()`
   * caller's async context and a store the node sets (`AsyncLocalStorage.enterWith`, even in its
   * synchronous part) stays in its own chain, never reaching the dispatcher or another node.
   */
  #startEvaluation(nodeId: string): Promise<EvaluatedOutcome> {
    return new AsyncResource("yian.node").runInAsyncScope(() => this.#evaluate(nodeId));
  }

  /**
   * Evaluate one started node under its own signal, aborted with the run's reason when the run
   * signal aborts; the run state changes only through its recorded steps.
   */
  async #evaluate(nodeId: string): Promise<EvaluatedOutcome> {
    const { state, onEvent, onStream, onConsole, live } = this;
    const node = state.flow.nodes.get(nodeId)!;
    // A derived node's commit names each pool name its binding read, with the seq of the
    // pool_commit that wrote it, kept in memory until the commit.
    let sourceResolver: SourceResolver | null = null;
    if (node.derived) {
      const reads = new Map<string, number>();
      this.#reads.set(nodeId, reads);
      sourceResolver = (names) => {
        const { name } = referenceRoot(names, state.flow.startId, (held) => state.producers.has(held));
        const seq = name === null ? undefined : state.producers.get(name);
        if (seq !== undefined) reads.set(name!, seq);
        return resolveFromPool(names, state.pool, state.producers);
      };
    }
    // A per-node controller, linked while the node runs, so the listeners node code leaves on its
    // signal die with the node instead of piling up on the long-lived run signal.
    const runSignal = this.#controller.signal;
    const nodeController = new AbortController();
    // Both callers check the run signal right before, so it is not aborted yet.
    const relay = (): void => nodeController.abort(runSignal.reason);
    runSignal.addEventListener("abort", relay, { once: true });
    try {
      return await evalNode(node, state.flow, state.pool, {
        wiringOverride: state.wiringOverrides.get(nodeId) ?? null,
        signal: nodeController.signal,
        emitStream: onStream === null && live === null ? null : (chunk) => this.#stream(chunk),
        emitEvent: onEvent === null && live === null ? null : (event) => this.#event(event),
        console: onConsole === null ? null : (method, text) => this.#console(onConsole, { runId: state.runId!, node: nodeId, method, text }),
        sourceResolver,
        tools: this.tools,
        memories: this.memories,
        skills: this.skills,
        scheduler: this.scheduler,
        random: this.random,
        runDir: this.runDir,
        models: this.models,
        steps: this.#nodeSteps(nodeId, nodeController.signal),
        steerInbox: this.#steerInbox(nodeId, nodeController.signal),
        // A capability fault ends the run now, not when the node settles, so a node that catches
        // it can't let a later stop replace it. For a failing host callback this is a no-op: the
        // engine's callback wrapper records the `callback` stop before the fault reaches here.
        onFault: (original) => void this.#fail(original),
      });
    } finally {
      runSignal.removeEventListener("abort", relay);
      live?.settle(nodeId);
    }
  }

  /** Pass one stream chunk to the live channel (an AGENT's text) and to `onStream`. */
  #stream(chunk: StreamChunk): void {
    const { live, onStream } = this;
    const { nodeId, n, index, final, chunk: text } = chunk;
    // The channel never throws, so it needs no host-callback guard; a stopping run sends nothing.
    if (live !== null && n !== null && index !== null && !final && !this.#controller.signal.aborted) {
      live.chunk(nodeId, n, index, text);
    }
    if (onStream !== null) this.#emit(onStream, chunk);
  }

  /**
   * Pass one line a node's code printed to `onConsole`. A throw stops the run (`#callHost`) and
   * goes no further: the line comes from a worker's message, not from the node's own code.
   */
  #console(onConsole: (line: ConsoleLine) => unknown, line: ConsoleLine): void {
    try {
      this.#callHost(onConsole, line);
    } catch {
      // `#callHost` has stopped the run with the error.
    }
  }

  /** Pass one in-loop event to the live channel (a model call's start or end) and to `onEvent`. */
  #event(event: RunEvent): void {
    const { live, onEvent } = this;
    if (live !== null && !this.#controller.signal.aborted) {
      if (event instanceof ModelCallStarted) {
        const { node_id, n, model, purpose, started_at } = event.toJSON();
        live.start(node_id, n, model, purpose, started_at);
      } else if (event instanceof ModelCallFinished) {
        live.end(event.nodeId, event.n, event.error);
      }
    }
    if (onEvent !== null) this.#emit(onEvent, event);
  }

  /**
   * Pass one transient chunk or in-loop event to its callback unless the run is stopping; a
   * stopping run with no result still passes a settled call's `ToolCallFinished`.
   */
  #emit<T>(callback: (item: T) => unknown, item: T): void {
    if (this.#controller.signal.aborted && !(item instanceof ToolCallFinished && this.state.result === null)) return;
    this.#callHost(callback, item);
  }

  /** Add each post-durable observer to the run's events, then pass it to `onEvent`. */
  #publish(observers: readonly RunEvent[], events: RunEvent[]): void {
    for (const observer of observers) {
      // Added first, so a delivery the callback makes lands after it, in log order.
      events.push(observer);
      if (this.onEvent !== null) this.#callHost(this.onEvent, observer);
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
   * @internal Stop the run for a host callback the caller invoked itself, whose returned promise
   * rejected with `error`: the run boundary publishes `RunStarted` before the engine exists. As
   * for the engine's own callbacks, this does nothing once the run has ended or is stopping.
   */
  stopForCallback(error: unknown): void {
    this.#stop("callback", error);
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
      throw new YianRangeError("FlowEngine requires an initialized RunState");
    }
    if (state.runId !== log.runId) {
      throw new YianRangeError(`run-state/log run_id mismatch: state ${reprText(state.runId)}, log ${reprText(log.runId)}`);
    }
    if (state.nextSequence !== log.nextSequence) {
      throw new YianRangeError(`run-state/log sequence mismatch: state ${state.nextSequence}, log ${log.nextSequence}`);
    }
    if (state.numWorkers !== this.numWorkers) {
      throw new YianRangeError(`run-state/engine worker mismatch: state ${state.numWorkers}, engine ${this.numWorkers}`);
    }
  }
}
