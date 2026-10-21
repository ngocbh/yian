/**
 * Reverse computation and explicit, record-authorized backward attempts.
 *
 * The fold consumes recorded routes and node-owned backward hooks. It does not interpret kinds,
 * evaluate expressions, drive child flows, or write logs/memory. Small durable attribution
 * references live in `runtime/contracts.ts` below this module; materialized interpretation data and
 * ordering keys remain transient. The shared learning projection owns formatting without
 * performing this callback fold. The internal attempt driver surrounds that pure computation with
 * accepted records and injected memory resources. It never resumes proposals or invokes public run
 * lifecycle operations.
 *
 * Never imports: compile, compose, the engine, run or the CLI.
 */

import { reprText } from "../diagnostics.ts";
import { type FeedbackValue, sameSource, sourceKey } from "../expr/lineage.ts";
import {
  LearningFailure,
  MemoryOperationFailure,
  sameMemorySnapshotEnvelope,
  sameMemorySpec,
} from "../memory/contracts.ts";
import {
  type BackwardContext,
  type BackwardOptions,
  BackwardResult,
  copyBackwardValue,
  markerMessage,
  validateBackwardResult,
} from "../nodes/base.ts";
import { isExactly, sameItems, sameOptional } from "../nodes/contracts.ts";
import { jsTypeName } from "../typesys/values.ts";
import { AttributedBackwardResult, BackwardFailure, BackwardOutcome, type MemoryProposal } from "./contracts.ts";
import {
  addMessage,
  ContextCache,
  contextualTransfers,
  materializeContext,
  memoryCommitId,
  OwnerContextError,
  prefixLearning,
  projectOwnerContexts,
  projectRootBackward,
  proposalBatches,
  seedBackward,
  takeMessages,
} from "./learning.ts";
import {
  BackwardFinishedRecord,
  BackwardReadyRecord,
  BackwardStartedRecord,
  MemoryCommitFailedRecord,
  MemoryCommitRequestedRecord,
  MemoryCommitSucceededRecord,
  type RunLog,
  type RunRecord,
} from "./log.ts";
import { LearningProposalError, MemoryRuntime } from "./memory_runtime.ts";
import { commitPrepared, prepareRecord, RunState } from "./state.ts";
import type { RetainedTrace } from "./trace.ts";

/** @internal A callback or result-contract failure attributed to one occurrence; forward state is unchanged. */
export class BackwardComputeError extends Error {
  override name = "BackwardComputeError";
  /** The attributed failure; the error's message is its message. */
  readonly failure: BackwardFailure;

  /** Keep `failure`, and the original error as the `cause`. */
  constructor(failure: BackwardFailure, options: { cause?: unknown } = {}) {
    super(failure.message, options);
    this.failure = failure;
  }
}

/**
 * Return an `Error`'s stack followed by its string `notes`, one per line, else `null`; Python's
 * traceback text prints `__notes__` the same way. Local copy of eval_node.ts's tracebackText.
 */
function tracebackText(error: unknown): string | null {
  if (!(error instanceof Error)) return null;
  const notes: unknown = (error as { notes?: unknown }).notes;
  const lines = [error.stack, ...(Array.isArray(notes) ? notes : [])].filter((line) => typeof line === "string");
  return lines.length > 0 ? lines.join("\n") : null;
}

/** Return `error` as a `code` failure of the occurrence `context` describes. */
function computeFailure(context: BackwardContext, code: string, error: unknown): BackwardComputeError {
  return new BackwardComputeError(
    new BackwardFailure("backward", code, markerMessage(error), {
      scope: context.scope,
      runId: context.runId,
      nodeId: context.nodeId,
      nodeStartedSeq: context.nodeStartedSeq,
      causeType: error instanceof Error ? error.name : jsTypeName(error),
      traceback: tracebackText(error),
    }),
    { cause: error },
  );
}

/**
 * Project root feedback through one accepted-outcome reverse walk.
 *
 * @param trace A retained terminal execution with private backward callbacks.
 * @param rootFeedback One ordinary text, record, positional array, or `null` value selecting the
 *   actual terminal output. Failed terminals accept text or `null`.
 * @param options.cache The pass's route-context cache over `trace`, so the owner projection can
 *   reuse it; `null` or omitted starts a new one.
 * @param options.models Builds the model a callback calls (AGENT backward); passed to each
 *   callback, which defaults it to `defaultModelAccess()`.
 * @param options.onEvent Receives the transient events callbacks publish; passed to each callback.
 * @returns Root input feedback and results in accepted-outcome order, including unreached
 *   occurrences whose callbacks were not invoked.
 * @throws TypeError (as a rejection, as are all below) for malformed feedback or trace carriers.
 * @throws RangeError for an unfinished trace, invalid feedback shape, or broken retained
 *   route/source authority.
 * @throws Error when an occurrence that receives feedback has no retained handler.
 * @throws BackwardComputeError for an attributed callback or result-contract error; any error a
 *   callback throws, a host capability fault included, is attributed.
 */
export async function computeBackward(
  trace: RetainedTrace,
  rootFeedback: FeedbackValue,
  options: { cache?: ContextCache | null } & BackwardOptions = {},
): Promise<[BackwardResult, readonly AttributedBackwardResult[]]> {
  const { models, onEvent } = options;
  const cache = options.cache ?? new ContextCache(trace);
  const [pending, directRoot] = seedBackward(trace, rootFeedback);
  const occurrences = trace.occurrences;
  const results = new Map(
    occurrences.map((item) => [
      sourceKey(item.source),
      new AttributedBackwardResult(item.source, [], [], new BackwardResult(), false),
    ]),
  );
  let rootInputs: Readonly<Record<string, readonly FeedbackValue[]>> = {};
  for (const occurrence of [...occurrences].reverse()) {
    const source = occurrence.source;
    const messages = takeMessages(pending, source);
    if (messages.length === 0) continue;
    // Trace lookups stay outside the callback's `try`: their faults aren't the callback's.
    const context = trace.backwardContext(source, {
      downstreamContext: messages.map((message, i) => materializeContext(message, i, cache)),
    });
    const received = messages.map((item) => item.feedback);
    const feedback = Object.freeze(received.map((message) => copyBackwardValue(message, "feedback") as FeedbackValue));
    const handler = trace.backwardHandlers.get(sourceKey(source));
    if (handler === undefined) throw new Error("backward occurrence has no retained handler");
    let result: BackwardResult;
    try {
      result = await handler(context, feedback, { models, onEvent });
    } catch (error) {
      throw computeFailure(context, "backward.execute", error);
    }
    try {
      validateBackwardResult(result, context);
    } catch (error) {
      if (error instanceof TypeError || error instanceof RangeError) {
        throw computeFailure(context, "backward.result", error);
      }
      throw error;
    }
    results.set(
      sourceKey(source),
      new AttributedBackwardResult(source, received, messages.map((item) => item.attribution), result, true),
    );
    if (sameOptional(source, trace.rootStartSource, sameSource)) rootInputs = result.inputFeedback;
    for (const [target, transfer] of contextualTransfers(trace, occurrence, context, messages, result, cache)) {
      addMessage(pending, target, transfer);
    }
  }
  const root = projectRootBackward(trace, pending, rootInputs, directRoot);
  return [root, occurrences.map((item) => results.get(sourceKey(item.source))!)];
}

/** Whether `log` has the `RunLog` members the driver reads. Local copy of engine.ts's isRunLog. */
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
 * Append and reduce one exact observer-free backward transition; its prepare and commit run in one
 * synchronous step, so no other drive can advance the state in between.
 */
function acceptBackward(state: RunState, log: RunLog, record: RunRecord): void {
  if (commitPrepared(log, state, prepareRecord(state, record)).length > 0) {
    throw new Error("backward records cannot emit observers");
  }
}

/**
 * Finish attempt `attemptId` and return only the outcome of the accepted Finished record. A drive
 * finishes only its own attempt: the reducer refuses a Finished whose attempt is no longer the
 * unfinished current one.
 */
function finishBackward(
  state: RunState,
  log: RunLog,
  attemptId: number,
  outcome: BackwardOutcome,
  attributed: readonly AttributedBackwardResult[] = [],
): BackwardOutcome {
  acceptBackward(state, log, new BackwardFinishedRecord(attemptId, outcome, attributed));
  return state.backwardState.finished!.outcome;
}

/** Compute once, stage every memory, then request and observe each memory key in order. */
async function driveOpenedBackward(
  state: RunState,
  log: RunLog,
  memory: MemoryRuntime | null,
  options: BackwardOptions,
): Promise<BackwardOutcome> {
  const started = state.backwardState.started!;
  const attemptId = started.attemptId;
  const trace = state.trace!;
  const cache = new ContextCache(trace);
  let root: BackwardResult;
  let attributed: readonly AttributedBackwardResult[];
  try {
    [root, attributed] = await computeBackward(trace, started.feedback, { cache, ...options });
  } catch (error) {
    if (!(error instanceof BackwardComputeError)) throw error;
    return finishBackward(
      state,
      log,
      attemptId,
      new BackwardOutcome("failed", { failure: error.failure, learning: prefixLearning(state.backwardState) }),
    );
  }
  let proposals: readonly MemoryProposal[];
  try {
    const contexts = projectOwnerContexts(
      trace,
      state.memories ?? [],
      state.memoryStacks ?? [],
      started.feedback,
      root,
      attributed,
      { cache },
    );
    proposals = memory === null ? [] : await memory.proposeLearning(contexts);
  } catch (error) {
    let key: string;
    let context: BackwardContext | null = null;
    let cause: MemoryOperationFailure;
    if (error instanceof OwnerContextError) {
      key = error.memory;
      context = error.context;
      let message = "memory learn context could not be constructed";
      // A trace's backward context always names its occurrence's node.
      if (context !== null) message += ` for node ${reprText(context.nodeId!)}`;
      const text = `${message}: ${error.original.message}`;
      cause = new MemoryOperationFailure("learn", "memory.invalid_context", text, false);
    } else if (error instanceof LearningProposalError) {
      key = error.memory;
      cause = error.failure;
    } else {
      throw error;
    }
    const failure = new BackwardFailure("learn", cause.code, cause.message, {
      scope: context?.scope ?? null,
      runId: context?.runId ?? null,
      nodeId: context?.nodeId ?? null,
      nodeStartedSeq: context?.nodeStartedSeq ?? null,
      memory: key,
      cause,
    });
    const learning = prefixLearning(state.backwardState, { learnerFailure: new LearningFailure(key, "learn", cause) });
    return finishBackward(
      state,
      log,
      attemptId,
      new BackwardOutcome("failed", { feedback: root, failure, learning }),
      attributed,
    );
  }

  acceptBackward(state, log, new BackwardReadyRecord(attemptId, root, attributed, proposals));
  const backward = state.backwardState;
  const batches = proposalBatches(backward, backward.ready!);
  for (const [index, key] of backward.memoryOrder.entries()) {
    const batch = batches[index]!;
    const commitId = memoryCommitId(state.runId!, key, batch, backward.commitAttemptKey!);
    acceptBackward(state, log, new MemoryCommitRequestedRecord(key, commitId, batch, attemptId));
    // The backend receives the accepted request, re-read after its append.
    const cause = await memory!.commitLearning(state.backwardState.current!.requests.at(-1)!);
    if (cause !== null) {
      acceptBackward(state, log, new MemoryCommitFailedRecord(key, commitId, cause, attemptId));
      const failure = new BackwardFailure("commit", cause.code, cause.message, { memory: key, cause });
      return finishBackward(
        state,
        log,
        attemptId,
        new BackwardOutcome("failed", { feedback: root, failure, learning: prefixLearning(state.backwardState) }),
      );
    }
    acceptBackward(state, log, new MemoryCommitSucceededRecord(key, commitId, attemptId));
  }
  return finishBackward(
    state,
    log,
    attemptId,
    new BackwardOutcome("succeeded", {
      feedback: root,
      learning: prefixLearning(state.backwardState, { succeeded: true }),
    }),
  );
}

/**
 * Drive one new explicit whole backward attempt against an accepted terminal run.
 *
 * Every call starts a new attempt, including after a successful, failed, or committing earlier
 * attempt.
 *
 * @param state Verified retained forward and backward record authority.
 * @param log The open append authority for that state.
 * @param rootFeedback Ordinary feedback on the recorded terminal.
 * @param options.openMemory Fresh resource opener using the recorded memory table and snapshots;
 *   required only for enabled memories, and called only then. This driver closes the returned
 *   runtime on every exit.
 * @param options.models Builds the model a callback calls (AGENT backward); `computeBackward`'s.
 * @param options.onEvent Receives the transient events callbacks publish; `computeBackward`'s.
 * @returns The exact accepted Finished disposition. An unfinished earlier attempt, including one
 *   with unconfirmed commit requests, is first closed as interrupted; all callbacks and proposals
 *   are recomputed.
 * @throws TypeError (as a rejection, as are all below) when `state` isn't exactly a `RunState`,
 *   `log` isn't a `RunLog`, the feedback is malformed, or the opener returns something other than
 *   a `MemoryRuntime`.
 * @throws RangeError for invalid feedback or terminal state, a missing opener, an opened runtime
 *   that is closed or doesn't match the recorded memory table and snapshots, or a state another
 *   drive advanced meanwhile.
 * @throws Error when a backward record emits observers. Host, setup, unexpected backend, and
 *   append/reducer faults escape with their accepted prefix intact; cleanup never manufactures a
 *   completion record.
 */
export async function driveBackward(
  state: RunState,
  log: RunLog,
  rootFeedback: FeedbackValue,
  options: { openMemory?: (() => MemoryRuntime | Promise<MemoryRuntime>) | null } & BackwardOptions = {},
): Promise<BackwardOutcome> {
  if (!isExactly(state, RunState) || !isRunLog(log)) {
    throw new TypeError("backward driving requires a RunState and RunLog");
  }
  const started = new BackwardStartedRecord(state.backwardState.attempts.length + 1, rootFeedback);
  seedBackward(state.trace!, started.feedback);
  const backward = state.backwardState;
  const interrupted = backward.started !== null && backward.finished === null;
  const first = interrupted ? new BackwardFinishedRecord(backward.started!.attemptId, backward.outcome!) : started;
  // A gate only: `seedBackward` already refused the learn-disabled and terminal cases; this
  // refuses a record that passes that check but can't be prepared (e.g. feedback that isn't
  // canonical JSON) before the opener runs. The record is prepared again with its commit, since
  // the opener's await lets another drive advance the state.
  prepareRecord(state, first);
  let memory: MemoryRuntime | null = null;
  // Boxed, so a thrown `undefined` still counts as the primary error.
  let primary: { error: unknown } | null = null;
  try {
    if (backward.memoryOrder.length > 0) {
      const { openMemory = null } = options;
      if (typeof openMemory !== "function") {
        throw new RangeError("enabled backward memories require a fresh memory runtime");
      }
      const opened: unknown = await openMemory();
      if (!(opened instanceof MemoryRuntime)) throw new TypeError("backward memory opener must return a MemoryRuntime");
      memory = opened;
      if (
        memory.closed ||
        !sameItems(memory.memories, state.memories ?? [], sameMemorySpec) ||
        !sameItems(memory.snapshots, state.memorySnapshots ?? [], sameMemorySnapshotEnvelope)
      ) {
        throw new RangeError("opened memory runtime does not match the recorded memory table and snapshots");
      }
    }
    acceptBackward(state, log, first);
    if (interrupted) acceptBackward(state, log, started);
    return await driveOpenedBackward(state, log, memory, { models: options.models, onEvent: options.onEvent });
  } catch (error) {
    primary = { error };
    throw error;
  } finally {
    await memory?.closePreserving(primary);
  }
}
