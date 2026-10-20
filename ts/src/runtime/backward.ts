/**
 * Reverse computation and explicit, record-authorized backward attempts.
 *
 * The fold consumes recorded routes and node-owned backward hooks. It does not interpret kinds,
 * evaluate expressions, drive child flows, or write logs/memory. Small durable attribution
 * references live in `runtime/contracts.ts` below this module; materialized interpretation data and
 * ordering keys remain transient. The shared learning projection owns formatting without
 * performing this callback fold.
 *
 * Never imports: compile, compose, the engine, run or the CLI.
 */

import { type FeedbackValue, sameSource, sourceKey } from "../expr/lineage.ts";
import {
  type BackwardContext,
  BackwardResult,
  copyBackwardValue,
  markerMessage,
  validateBackwardResult,
} from "../nodes/base.ts";
import { sameOptional } from "../nodes/contracts.ts";
import { jsTypeName } from "../typesys/values.ts";
import { AttributedBackwardResult, BackwardFailure } from "./contracts.ts";
import {
  addMessage,
  ContextCache,
  contextualTransfers,
  materializeContext,
  projectRootBackward,
  seedBackward,
  takeMessages,
} from "./learning.ts";
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
  options: { cache?: ContextCache | null } = {},
): Promise<[BackwardResult, readonly AttributedBackwardResult[]]> {
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
      result = await handler(context, feedback);
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
