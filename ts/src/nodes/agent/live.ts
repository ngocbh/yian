/**
 * Transient AGENT observers: streamed reply text, model and tool call brackets, usage.
 *
 * Everything here is callback-only: chunks and events go to evaluator-supplied sinks and nowhere
 * else. Nothing is recorded, retained in memory, or replayed on resume, and the reply is the same
 * with or without sinks. Every call streams on the wire; a chat reply's text reaches the host only
 * when a stream sink exists.
 *
 * A model call retries under `YIAN_MODEL_RETRIES` and publishes one bracket per attempt: each
 * attempt's `ModelCallStarted`, its chunks, then its `ModelCallFinished`. A finished event with
 * an error and no final chunk tells the host to discard that attempt's text; the next attempt
 * follows in the same exchange.
 *
 * Sink errors arrive already wrapped as `CapabilityFault` by the evaluator and propagate
 * unchanged; provider and tool errors are reported by name only, because their text can contain
 * credentials, then rethrown.
 *
 * Knows about:   diagnostics, events, llm, llm_clients auth, nodes/base, nodes/binding, typesys,
 *                peer AGENT modules.
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
import { type ErrorKind, ProviderError } from "../../llm/errors.ts";
import { type ChatModel, TextDelta } from "../../llm/events.ts";
import { callWithRetries } from "../../llm/executor.ts";
import type { Request, Response } from "../../llm/messages.ts";
import { isAuthFailure } from "../../llm_clients/auth/common.ts";
import type { JsonObject } from "../../typesys/json.ts";
import type { Value } from "../../typesys/values.ts";
import { CapabilityFault, type NodeCaps } from "../base.ts";
import { copyValue } from "../binding.ts";
import type { AgentRunContext } from "./context.ts";

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

/** Return a fresh model exchange id: 32 hex characters. */
export function newExchangeId(): string {
  return crypto.randomUUID().replaceAll("-", "");
}

/** The name an error is reported by: its `name` for an `Error`, else its `typeof`. */
function errorName(error: unknown): string {
  return error instanceof Error ? error.name : typeof error;
}

/**
 * Make one model call through `callWithRetries`, publishing each attempt to the sinks.
 *
 * Each attempt emits `ModelCallStarted`, then, for purpose `chat` with a stream sink, one chunk
 * per non-empty text delta; a retried attempt ends with `ModelCallFinished` naming its error and
 * no final chunk. The successful attempt ends with an empty `final` chunk when streaming, then
 * `ModelCallFinished` with its usage. A failure is marked with `markModelFailure` and reported
 * by name on the last attempt's `ModelCallFinished`, with or without sinks; a `CapabilityFault`,
 * or any error once `signal` aborted, passes through untouched with nothing more emitted. Sinks
 * throw `CapabilityFault` (the evaluator wraps them); any other sink error counts as a model failure.
 *
 * @param live The sinks, or `null` when there are none.
 * @param options `purpose` and `exchangeId` go on every event; `signal` stops the call.
 * @returns The successful attempt's reply.
 */
export async function callModel(
  live: LiveSinks | null,
  model: ChatModel,
  request: Request,
  options: { purpose: ModelCallPurpose; exchangeId: string; signal?: AbortSignal },
): Promise<Response> {
  const { purpose, exchangeId, signal } = options;
  const emitEvent = live?.emitEvent ?? null;
  const emitStream = purpose === "chat" ? (live?.emitStream ?? null) : null;
  const identity = { nodeId: live?.nodeId ?? "", exchangeId, purpose, provider: live?.provider, model: live?.model };
  const chunk = (text: string, attempt: number, final = false) =>
    new StreamChunk({ nodeId: identity.nodeId, key: STREAM_KEY, chunk: text, final, exchangeId, attempt });
  let attempt = 0;
  let response: Response;
  try {
    response = await callWithRetries(model, request, {
      signal,
      onAttempt: (k) => {
        attempt = k;
        emitEvent?.(new ModelCallStarted({ ...identity, attempt: k }));
      },
      onEvent: (event, k) => {
        if (emitStream !== null && event instanceof TextDelta && event.text !== "") emitStream(chunk(event.text, k));
      },
      onRetry: (retry) => {
        emitEvent?.(new ModelCallFinished({ ...identity, error: errorName(retry.error), attempt: retry.attempt }));
      },
    });
    emitStream?.(chunk("", attempt, true));
  } catch (error) {
    if (error instanceof CapabilityFault || signal?.aborted) throw error;
    markModelFailure(error);
    emitEvent?.(new ModelCallFinished({ ...identity, error: errorName(error), attempt }));
    throw error;
  }
  emitEvent?.(new ModelCallFinished({ ...identity, usage: usageOf(response), attempt }));
  return response;
}

/** The `error` of the finished event of a tool call a host interrupt skipped. */
export const SKIPPED_TOOL_ERROR = "interrupted";

/** One requested tool call, as its brackets name it. */
export interface ToolCallIdentity {
  readonly exchangeId: string;
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
  const { exchangeId, callIndex, toolCallId, name } = identity;
  return { nodeId: live.nodeId, exchangeId, callIndex, toolCallId, name };
}

/**
 * Bracket one granted tool call with started/finished observers.
 *
 * Emits `ToolCallStarted` with a copy of the arguments, awaits `invoke()`, then emits
 * `ToolCallFinished` with `result: text(result)` (`null` when `text` throws), or with the
 * rejection's `name` (`typeof` for a non-`Error`) as `error`. No finished event is emitted once
 * `signal` has aborted, and a `CapabilityFault` passes through with none. Without an event sink
 * nothing is emitted.
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
