/**
 * An AGENT's model call, recorded as a `model_call` step, and its transient observers: streamed
 * reply text, model and tool call brackets, usage.
 *
 * Chunks and events go to evaluator-supplied sinks and nowhere else; the reply is the same with
 * or without sinks. The call itself is written to the node's steps (`AgentSteps`) once it returns,
 * before anything acts on it, and a resumed start takes a recorded reply back in place of the call,
 * publishing nothing. Every live call streams on the wire; a chat reply's text reaches the host
 * only when a stream sink exists.
 *
 * A model call retries under `YIAN_MODEL_RETRIES` and publishes one bracket per attempt: each
 * attempt's `ModelCallStarted`, its chunks, then its `ModelCallFinished`, all naming the step the
 * attempt becomes (`nodeId` + `n`). A finished event with an error and no final chunk tells the
 * host to discard that attempt's text; the next attempt streams under the next `n`.
 *
 * Sink errors arrive already wrapped as `CapabilityFault` by the evaluator and propagate
 * unchanged; provider and tool errors are reported by name only, because their text can contain
 * credentials, then rethrown.
 *
 * Knows about:   diagnostics, events, llm, llm_clients auth, nodes/base, nodes/binding, tools
 *                context, typesys, peer AGENT modules.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import type { FailureCode } from "../../diagnostics.ts";
import {
  ModelCallFinished,
  type ModelCallPurpose,
  ModelCallStarted,
  StreamChunk,
  TokenUsage,
  ToolCallFinished,
  ToolCallStarted,
} from "../../events.ts";
import { ERROR_KINDS, type ErrorKind, ProviderError } from "../../llm/errors.ts";
import { type ChatModel, ReplyAssembler, TextDelta, UsageReport } from "../../llm/events.ts";
import { callWithRetries } from "../../llm/executor.ts";
import { Message, type Request, Response, TextPart } from "../../llm/messages.ts";
import { AuthError, isAuthFailure } from "../../llm_clients/auth/common.ts";
import { realScheduler, type Scheduler } from "../../platform.ts";
import { isPending } from "../../tools/context.ts";
import type { JsonObject } from "../../typesys/json.ts";
import type { Value } from "../../typesys/values.ts";
import { CapabilityFault, type NodeCaps } from "../base.ts";
import { copyValue } from "../binding.ts";
import {
  type AgentRunContext,
  type AgentSteps,
  fileToInline,
  inlineToFile,
  type PriorAttempts,
  type RetrySettings,
} from "./context.ts";
import { ReplyInterrupted } from "./interrupts.ts";

/** The stream key of AGENT reply text. */
export const STREAM_KEY = "text";

/** The failure codes of a marked provider error, by kind; other kinds are `node.execute`. */
const PROVIDER_CODES: Partial<Record<ErrorKind, FailureCode>> = {
  rate_limit: "provider.rate_limit",
  quota: "provider.quota",
  overloaded: "provider.unavailable",
  server: "provider.unavailable",
  timeout: "provider.unavailable",
  connection: "provider.unavailable",
  incomplete: "provider.unavailable",
  context_length: "provider.context_length",
  refused: "provider.refused",
};

// A set rather than a property on the error, so the error stays exactly as raised.
const modelFailures = new WeakSet<object>();

/** Mark `error` as raised by the AGENT's own model build or call; a primitive stays unmarked. */
export function markModelFailure(error: unknown): void {
  if (typeof error === "object" && error !== null) modelFailures.add(error);
}

/**
 * Return the failure code of an error `markModelFailure` marked, or `null` when unmarked.
 *
 * An auth failure (`isAuthFailure`) is `provider.auth`; a `ProviderError` takes its kind's
 * `provider.*` code; anything else is `node.execute`.
 */
export function modelFailureCode(error: unknown): FailureCode | null {
  if (typeof error !== "object" || error === null || !modelFailures.has(error)) return null;
  if (isAuthFailure(error)) return "provider.auth";
  return (error instanceof ProviderError ? PROVIDER_CODES[error.kind] : undefined) ?? "node.execute";
}

/**
 * An error of a name yian rebuilds from a recorded failure without its own class: the name and
 * message are the recorded ones.
 */
class RecordedCallError extends Error {
  override name: string;

  constructor(name: string, message: string) {
    super(message);
    this.name = name;
  }
}

/**
 * Return the error a failed `model_call` step recorded as `{kind, message}`: a `ProviderError` of
 * that kind (not retryable, from `provider`), an `AuthError`, or an error of that name, each with
 * the message.
 */
function recordedFailure({ kind, message }: { kind: string; message: string }, provider: string): Error {
  if ((ERROR_KINDS as readonly string[]).includes(kind)) {
    return new ProviderError(kind as ErrorKind, message, { retryable: false, provider });
  }
  return kind === "AuthError" ? new AuthError(message) : new RecordedCallError(kind, message);
}

/** The transient sinks of one AGENT invocation plus its effective model identity. */
export interface LiveSinks {
  readonly nodeId: string;
  readonly provider: string | null;
  readonly model: string | null;
  readonly emitEvent: NonNullable<NodeCaps["emitEvent"]> | null;
  readonly emitStream: NonNullable<NodeCaps["emitStream"]> | null;
}

/** Select the sinks of one AGENT loop context; `null` when neither sink exists. */
export function liveSinks(ctx: Pick<AgentRunContext, "nodeId" | "llmConfig" | "emitEvent" | "emitStream">): LiveSinks | null {
  const { emitEvent, emitStream } = ctx;
  if (!emitEvent && !emitStream) return null;
  const provider = ctx.llmConfig?.provider;
  const model = ctx.llmConfig?.model;
  return {
    nodeId: ctx.nodeId,
    provider: typeof provider === "string" ? provider : null,
    model: typeof model === "string" ? model : null,
    emitEvent: emitEvent ?? null,
    emitStream: emitStream ?? null,
  };
}

/** Return the reply's provider-reported usage as a `TokenUsage`; `null` when it has none. */
export function usageOf(response: Response): TokenUsage | null {
  const usage = response.usage;
  return usage === null ? null : new TokenUsage(usage);
}

/** The name an error is reported by: its `name` for an `Error`, else its `typeof`. */
export function errorName(error: unknown): string {
  return error instanceof Error ? error.name : typeof error;
}

/** Where `callModel` writes a call: the node's steps, and the request's digest (`requestSha256`). */
interface CallRecord {
  readonly steps: AgentSteps;
  readonly digest: string;
}

/** How `callModel` makes a call; see its `options`. */
interface CallOptions {
  readonly purpose: ModelCallPurpose;
  readonly signal?: AbortSignal;
  readonly interrupt?: AbortSignal;
  readonly retry?: RetrySettings;
  readonly record?: CallRecord;
}

/** What `callModel` returns: the reply, its `model_call` step's `n` (`N`), and whether it was replayed. */
interface Called<N extends number | null> {
  readonly response: Response;
  readonly n: N;
  readonly replayed: boolean;
}

/**
 * Make one model call through `callWithRetries`, publishing each attempt to the sinks and, with
 * `record`, writing it as a `model_call` step.
 *
 * Each attempt is named by the step it becomes: its `n` is the next step's (`AgentSteps.nextN`),
 * or the attempt's number without `record`. Each attempt emits `ModelCallStarted` (with its start
 * time), then, for purpose `chat` with a stream sink, one chunk per non-empty text delta, each with
 * the attempt's `n` and an `index` counting from 0; a retried attempt ends with
 * `ModelCallFinished` naming its error and no final chunk. The successful attempt ends with an
 * empty `final` chunk (the next `index`) when streaming, then `ModelCallFinished` with its usage.
 * A failure is marked with `markModelFailure` and reported by name on the last attempt's
 * `ModelCallFinished`, with or without sinks; a `CapabilityFault`,
 * or any error once `signal` aborted, passes through untouched with nothing more emitted. Sinks
 * throw `CapabilityFault` (the evaluator wraps them); any other sink error counts as a model failure.
 *
 * With `record`, a reply an earlier start recorded for this call is returned first, as recorded,
 * with nothing emitted or written (`AgentSteps.takeRecorded`); a recorded reply a host interrupt
 * cut throws `ReplyInterrupted` with its text, and a recorded call that failed throws its error
 * again (rebuilt from its kind and message, marked). The first live call after the replay
 * continues the failed attempts recorded after the last reply: it waits the last one's
 * `retry_in` and numbers its attempts after them (`AgentSteps.priorAttempts`).
 *
 * With `record`, the request goes to the model with its file media inlined (`fileToInline`), and
 * the call is written before anything acts on it: each failed attempt, before its
 * `ModelCallFinished`, with its wait (`retry_in`, `null` when it isn't retried), what arrived and
 * its usage; the reply, with its media stored as run files (`inlineToFile`), before this returns
 * (as settled, since it arrived while the call was live); a reply a host interrupt cut, with
 * finish `interrupted`. A reply whose media can't be stored (data that isn't base64, a refused
 * `media` folder, a failed write) fails the call, unretried, as a last attempt's error does. A
 * call the stop cut writes nothing, nor does a retry wait an interrupt ended.
 *
 * With `interrupt`, a host interrupt cuts the call. During an attempt it is checked right before
 * each non-empty `TextDelta` that isn't the attempt's first event (a later chunk, as a host sees
 * it); a cut closes the stream, sends the final chunk when streaming and `ModelCallFinished` with
 * the usage of the last `UsageReport` before the cut (else `null`) and no error. A reply whose
 * text had all arrived is never cut. During a retry wait the interrupt ends the wait at once with
 * nothing more emitted. A stop wins over an interrupt.
 *
 * @param live The sinks, or `null` when there are none.
 * @param options `purpose` goes on every event; `signal` stops the call;
 *   `interrupt` is the host's interrupt signal; `retry`'s `scheduler` and `random` run the retry
 *   waits; `record` holds the steps to write to and the request's digest (`requestSha256`).
 * @returns The successful attempt's reply, its media in file form with `record`; with `record`
 *   the `n` of the reply's `model_call` step (typed `number` then; `null` without); and whether the
 *   reply was replayed.
 * @throws ReplyInterrupted carrying the text shown before the cut and its usage, or an empty
 *   reply and `null` usage for a cut wait; it is never marked.
 * @throws CapabilityFault when a recorded call's purpose or digest differs from this one's.
 */
export async function callModel(
  live: LiveSinks | null,
  model: ChatModel,
  request: Request,
  options: CallOptions & { record: CallRecord },
): Promise<Called<number>>;
export async function callModel(
  live: LiveSinks | null,
  model: ChatModel,
  request: Request,
  options: CallOptions,
): Promise<Called<number | null>>;
export async function callModel(
  live: LiveSinks | null,
  model: ChatModel,
  request: Request,
  options: CallOptions,
): Promise<Called<number | null>> {
  const { purpose, signal, interrupt, retry = {}, record } = options;
  let prior: PriorAttempts | undefined;
  if (record !== undefined) {
    const recorded = record.steps.takeRecorded(purpose, record.digest);
    if (recorded !== null && "failure" in recorded) {
      const error = recordedFailure(recorded.failure, model.provider);
      markModelFailure(error);
      throw error;
    }
    if (recorded !== null) {
      const { response, n } = recorded;
      // A reply a host interrupt cut ends the call as that interrupt did.
      if (response.finishReason === "interrupted") throw new ReplyInterrupted(response.message, usageOf(response));
      return { response, n, replayed: true };
    }
    prior = record.steps.priorAttempts(purpose, record.digest) ?? undefined;
  }
  const { scheduler = realScheduler, random } = retry;
  const emitEvent = live?.emitEvent ?? null;
  const emitStream = purpose === "chat" ? (live?.emitStream ?? null) : null;
  const identity = { nodeId: live?.nodeId ?? "", purpose, provider: live?.provider, model: live?.model };
  // The current attempt's step and its next chunk's number.
  let n = 0;
  let index = 0;
  const chunk = (text: string, final = false) =>
    new StreamChunk({ nodeId: identity.nodeId, key: STREAM_KEY, chunk: text, final, n, index: index++ });
  const runDir = record?.steps.runDir ?? null;
  const sent = runDir === null ? request : await fileToInline(request, runDir);
  // The current attempt so far: whether no event arrived yet, the text shown, the latest usage,
  // when it started and its first event arrived, and what arrived, for its step.
  let first = true;
  let shownText = "";
  let usage: TokenUsage | null = null;
  let startedAt = 0;
  let firstTokenAt: number | null = null;
  let arrived = new ReplyAssembler();
  let waiting = false;
  const call = () => ({
    purpose,
    model: model.model,
    requestSha256: record!.digest,
    startedAt,
    firstTokenAt,
  });
  let response: Response;
  try {
    response = await callWithRetries(model, sent, {
      signal,
      random,
      prior,
      scheduler: interrupt === undefined ? scheduler : interruptible(scheduler, interrupt, () => (waiting = true)),
      onAttempt: (k) => {
        n = record === undefined ? k : record.steps.nextN();
        index = 0;
        waiting = false;
        first = true;
        shownText = "";
        usage = null;
        firstTokenAt = null;
        arrived = new ReplyAssembler();
        startedAt = record === undefined ? scheduler.now() : record.steps.now();
        emitEvent?.(new ModelCallStarted({ ...identity, n, startedAt }));
      },
      onEvent: (event) => {
        const text = event instanceof TextDelta ? event.text : "";
        if (text !== "" && !first && interrupt?.aborted) {
          const parts = shownText ? [new TextPart(shownText)] : [];
          throw new ReplyInterrupted(new Message({ role: "assistant", parts }), usage);
        }
        if (first && record !== undefined) firstTokenAt = record.steps.now();
        first = false;
        if (record !== undefined) arrived.add(event);
        if (event instanceof UsageReport) usage = new TokenUsage(event.usage);
        if (text === "") return;
        shownText += text;
        emitStream?.(chunk(text));
      },
      onRetry: (info) => {
        record?.steps.failedModelCall({ ...call(), output: info.partial, usage, retryIn: info.retryIn }, info.error);
        emitEvent?.(new ModelCallFinished({ ...identity, n, error: errorName(info.error) }));
      },
    });
    if (record !== undefined) {
      // Stored inside the try: the call happened, so media that can't be stored fails it like a
      // provider error, with its step written and its bracket closed, and a resume doesn't pay again.
      const { message, usage: used, finishReason, responseId } = response;
      response = new Response({ message: inlineToFile(message, runDir), usage: used, finishReason, responseId });
    }
    emitStream?.(chunk("", true));
  } catch (error) {
    if (error instanceof CapabilityFault || signal?.aborted) throw error;
    if (error instanceof ReplyInterrupted) {
      // A cut wait already reported its attempt's end, with the error it was retrying.
      if (!waiting) {
        record?.steps.modelCall({
          ...call(),
          output: error.reply,
          responseId: null,
          finish: "interrupted",
          usage: error.usage,
        });
        emitStream?.(chunk("", true));
        emitEvent?.(new ModelCallFinished({ ...identity, n, usage: error.usage }));
      }
      throw error;
    }
    record?.steps.failedModelCall({ ...call(), output: arrived.partial(), usage, retryIn: null }, error);
    markModelFailure(error);
    emitEvent?.(new ModelCallFinished({ ...identity, n, error: errorName(error) }));
    throw error;
  }
  let step: number | null = null;
  if (record !== undefined) {
    const { message: output, usage: used, finishReason: finish, responseId } = response;
    step = record.steps.modelCall({ ...call(), output, responseId, finish, usage: used }, { settled: true });
  }
  emitEvent?.(new ModelCallFinished({ ...identity, n, usage: usageOf(response) }));
  return { response, n: step, replayed: false };
}

/**
 * Return `scheduler` with its `sleep` also ended by `interrupt`: the inner sleep waits on a child
 * signal that the stop and the interrupt both abort, so its timer is cancelled either way. An
 * interrupt ends the wait with `ReplyInterrupted` (an empty reply, `null` usage) unless the stop
 * aborted too. `onWait` runs as each sleep starts.
 */
function interruptible(scheduler: Scheduler, interrupt: AbortSignal, onWait: () => void): Scheduler {
  return {
    now: () => scheduler.now(),
    async sleep(ms, stop) {
      const child = new AbortController();
      const abort = () => child.abort();
      stop?.addEventListener("abort", abort, { once: true });
      interrupt.addEventListener("abort", abort, { once: true });
      if (stop?.aborted || interrupt.aborted) child.abort();
      onWait();
      try {
        await scheduler.sleep(ms, child.signal);
      } catch (error) {
        if (stop?.aborted) throw stop.reason;
        if (interrupt.aborted) throw new ReplyInterrupted(new Message({ role: "assistant", parts: [] }), null);
        throw error;
      } finally {
        stop?.removeEventListener("abort", abort);
        interrupt.removeEventListener("abort", abort);
      }
    },
  };
}

/** The `error` of the finished event of a tool call a host interrupt skipped. */
export const SKIPPED_TOOL_ERROR = "interrupted";

/** One requested tool call, as its brackets name it. */
export interface ToolCallIdentity {
  /** The `n` of the reply's `model_call` step. */
  readonly n: number;
  readonly callIndex: number;
  readonly toolCallId: string;
  readonly name: string;
  readonly arguments: JsonObject;
}

/** Return a copy of a call's arguments for its started event; `null` when they can't be copied. */
function shown(args: JsonObject): Value | null {
  try {
    return copyValue(args);
  } catch {
    return null;
  }
}

/** Return the text a finished event shows for `result`; `null` when `text` throws. */
function shownResult(result: unknown, text: (result: unknown) => string): string | null {
  try {
    return text(result);
  } catch {
    // The caller's own representation step owns this error.
    return null;
  }
}

/** Return the events identifying one tool call of `live`'s node. */
function toolEventIdentity(live: LiveSinks, identity: ToolCallIdentity) {
  const { n, callIndex, toolCallId, name } = identity;
  return { nodeId: live.nodeId, n, callIndex, toolCallId, name };
}

/**
 * Bracket one granted tool call with started/finished observers.
 *
 * Emits `ToolCallStarted` with a copy of the arguments, awaits `invoke()`, then emits
 * `ToolCallFinished` with `result: text(result)` (`null` when `text` throws), or with the
 * rejection's `name` (`typeof` for a non-`Error`) as `error`. No finished event is emitted once
 * `signal` has aborted, or for a call whose handler returned `pending()` (it hasn't finished), and
 * a `CapabilityFault` passes through with none. Without an event sink nothing is emitted.
 *
 * @returns What `invoke()` resolves to.
 * @throws Whatever `invoke()` rejects with, unchanged.
 */
export async function observeToolCall(
  live: LiveSinks | null,
  invoke: () => Promise<unknown>,
  identity: ToolCallIdentity,
  options: { signal?: AbortSignal; text: (result: unknown) => string },
): Promise<unknown> {
  const emitEvent = live?.emitEvent ?? null;
  if (live === null || emitEvent === null) return await invoke();
  const { signal, text } = options;
  const ids = toolEventIdentity(live, identity);
  emitEvent(new ToolCallStarted({ ...ids, arguments: shown(identity.arguments) }));
  let result: unknown;
  try {
    result = await invoke();
  } catch (error) {
    if (!(error instanceof CapabilityFault) && !signal?.aborted) {
      emitEvent(new ToolCallFinished({ ...ids, error: errorName(error) }));
    }
    throw error;
  }
  // A call that waits for a later answer hasn't finished; hosts learn of the wait from the run.
  if (isPending(result)) return result;
  if (!signal?.aborted) emitEvent(new ToolCallFinished({ ...ids, result: shownResult(result, text) }));
  return result;
}

/**
 * Report a tool call that a host interrupt skipped as started, then finished `interrupted`.
 *
 * The same pair a run call emits, so a host can mark that call's part as interrupted; the tool
 * itself never runs. Without an event sink nothing is emitted.
 */
export function observeSkippedToolCall(live: LiveSinks | null, identity: ToolCallIdentity): void {
  const emitEvent = live?.emitEvent ?? null;
  if (live === null || emitEvent === null) return;
  const ids = toolEventIdentity(live, identity);
  emitEvent(new ToolCallStarted({ ...ids, arguments: shown(identity.arguments) }));
  emitEvent(new ToolCallFinished({ ...ids, error: SKIPPED_TOOL_ERROR }));
}
