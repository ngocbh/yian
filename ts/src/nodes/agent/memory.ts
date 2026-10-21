/**
 * How an AGENT uses memory: model input from recalled notes and transcripts, and model and tool
 * effects recorded back into the memory turn.
 *
 * Here so far: framing recalled notes into the system prompt; the compatibility descriptor: the
 * provider and message format a stored transcript was written with (changing either requires an
 * explicit memory reset; model, prompt and tool-grant changes keep history); and the text the
 * model reads for a tool result, a reported tool error or a skipped call; the message codec, which
 * stores a `Message` as a `ConversationItem` and reads it back strictly; and the memory boundary,
 * which keeps codec and backend faults apart from errors a model retry may handle; and recording
 * model calls and tool effects in the turn: each request stored before the call, then its reply,
 * result or sanitized failure, a pending stop recorded `interrupted`, and a failed observation
 * never hiding the call's own error. Projecting stored history for the model is not ported yet.
 *
 * The descriptor's `message_format` is `MESSAGE_FORMAT` of `llm/messages.ts` (`yian.message.v1`),
 * not Python yian's `yian.langchain-message.v1`: TS stores its own message shape and never reads
 * a Python memory store.
 *
 * Knows about:   defaults (the `Env` type), diagnostics, llm (messages), llm_clients, memory
 *                contracts, nodes/base, nodes/contracts, tools (errors), typesys, and the peer
 *                AGENT modules `interrupts` and `live`.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import type { Env } from "../../defaults.ts";
import { reprText } from "../../diagnostics.ts";
import { Message, MESSAGE_FORMAT, MessageFormatError, type Response, ToolResultPart } from "../../llm/messages.ts";
import type { LLMConfig } from "../../llm_clients/config.ts";
import { type ConnectionSnapshot, resolveModelConfig } from "../../llm_clients/connection.ts";
import {
  type ConversationEffectKind,
  type ConversationEffectStatus,
  ConversationItem,
  type ConversationTurn,
  MemoryOperationFailure,
  type MemorySection,
} from "../../memory/contracts.ts";
import { isReportedToolError } from "../../tools/errors.ts";
import { canonicalJson, type JsonObject, logJson } from "../../typesys/json.ts";
import { fromHost, writeAsText } from "../../typesys/values.ts";
import { CapabilityFault } from "../base.ts";
import { isExactly } from "../contracts.ts";
import { ReplyInterrupted } from "./interrupts.ts";
import { errorName, newExchangeId } from "./live.ts";

/** The version of the history projection a backend summarizes. */
export const HISTORY_PROJECTION = "yian.api-history.v2";

/** The result the model reads for a tool call a host interrupt skipped. */
export const SKIPPED_TOOL_RESULT = "Not run: interrupted by the user.";

/**
 * Frame opaque memory text as labelled outer-to-local system sections.
 *
 * @returns `base` unchanged without sections; otherwise `base`, a header line, and canonical
 *   JSON `{"order": "outer-to-local", "sections": [{"index", "memory", "text"}, ...]}`, indices
 *   from 1.
 */
export function systemPromptWithMemory(base: string, sections: readonly MemorySection[]): string {
  if (sections.length === 0) return base;
  const payload = {
    order: "outer-to-local",
    sections: sections.map((section, index) => ({ index: index + 1, memory: section.memory, text: section.text })),
  };
  // The keys are ASCII, so canonical JSON's key order is Python's `sort_keys` order.
  return (
    base +
    "\n\nMemory context (canonical JSON; sections are ordered from outer " +
    "workflow to local agent):\n" +
    canonicalJson(payload)
  );
}

/**
 * Return the provider and versioned message format `config` selects, independent of the model,
 * current grants and credentials.
 *
 * An `LLMConfig` gives its `.provider`, a record its `provider`. A provider that is unset, `null`
 * or `""` resolves as the model factory resolves it (`resolveModelConfig` with `options`). The
 * provider is lowercased.
 *
 * Example: `compatibility({provider: "OpenAI", api_key: "k"})` gives
 * `{provider: "openai", message_format: "yian.message.v1"}`.
 *
 * Throws `RangeError` if the provider is absent and no default sets one. Other errors from
 * `resolveModelConfig` (such as a `ValidationError` for `provider: ""`) propagate.
 */
export function compatibility(
  config: Readonly<Record<string, unknown>> | LLMConfig | null,
  options: { snapshot?: ConnectionSnapshot; env?: Env } = {},
): JsonObject {
  const given = config?.["provider"];
  const provider =
    typeof given === "string" && given !== ""
      ? given
      : (resolveModelConfig(config ?? {}, options)["provider"] as string);
  return { provider: provider.toLowerCase(), message_format: MESSAGE_FORMAT };
}

/**
 * Run `fn` and return its result, keeping codec and backend faults apart from errors eligible for
 * model retries.
 *
 * A `CapabilityFault` passes through, and so does the stop: the reason of an aborted `signal`.
 * Anything else, also another error after an abort, rejects as `new CapabilityFault(error)`.
 */
export async function memoryBoundary<T>(fn: () => T | Promise<T>, signal?: AbortSignal): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (error instanceof CapabilityFault || isStop(error, signal)) throw error;
    throw new CapabilityFault(error);
  }
}

/** Whether `error` is the stop: the reason of the aborted `signal`. */
function isStop(error: unknown, signal: AbortSignal | undefined): boolean {
  return signal?.aborted === true && error === signal.reason;
}

/**
 * Store `message` as a `ConversationItem` under a fresh item id and `exchangeId`.
 *
 * @param exchangeId The nonempty adapter-generated request identity, not a provider's tool-call
 *   id.
 * @returns A frozen item of format `yian.message.v1` with payload `{exchange_id, message}`, the
 *   message as `toJSON` writes it; it is decoded once to check it reads back unchanged.
 * @throws RangeError as `decodeMessage` does.
 */
export function encodeMessage(message: Message, exchangeId: string): ConversationItem {
  const item = new ConversationItem(newExchangeId(), MESSAGE_FORMAT, {
    exchange_id: exchangeId,
    message: message.toJSON(),
  });
  decodeMessage(item);
  return item;
}

/**
 * Read a stored message back, strictly.
 *
 * @throws TypeError when `item` is not exactly a `ConversationItem`.
 * @throws RangeError for another format, payload keys other than `exchange_id` and `message`, an
 *   empty or non-string exchange id, or a message `Message.fromJSON` refuses or would write back
 *   changed ("invalid stored conversation message", the error as its `cause`).
 */
export function decodeMessage(item: ConversationItem): Message {
  if (!isExactly(item, ConversationItem)) throw new TypeError("stored messages must be exact ConversationItem values");
  if (item.format !== MESSAGE_FORMAT) {
    throw new RangeError("unsupported conversation message format; reset is required");
  }
  const payload = item.payload;
  const keys = Object.keys(payload);
  if (keys.length !== 2 || !Object.hasOwn(payload, "exchange_id") || !Object.hasOwn(payload, "message")) {
    throw new RangeError("conversation message fields do not match the supported format");
  }
  const exchangeId = payload["exchange_id"];
  if (typeof exchangeId !== "string" || exchangeId === "") {
    throw new RangeError("conversation exchange_id must be a nonempty string");
  }
  try {
    const message = Message.fromJSON(payload["message"]);
    // Every stored field must survive a read and a write unchanged. `logJson`, not `canonicalJson`:
    // provider payloads hold floats and large ints, which JCS refuses.
    if (logJson(message.toJSON()) !== logJson(payload["message"])) {
      throw new RangeError("stored message would be changed by decoding");
    }
    return message;
  } catch (error) {
    if (error instanceof MessageFormatError || error instanceof TypeError || error instanceof RangeError) {
      throw new RangeError("invalid stored conversation message", { cause: error });
    }
    throw error;
  }
}

/**
 * Return a tool result as the text the model reads: text as itself, any other value in the text
 * form (design 3.2.3), `null` as `null`.
 *
 * Example: `toolText("lookup", {a: 1})` gives `{"a":1}`.
 *
 * @throws TypeCheckError when the result is not a yian value, read as `tool '<name>' result`.
 * @throws TypeError as `writeAsText` does.
 */
export function toolText(name: string, result: unknown): string {
  return writeAsText(fromHost(result, `tool ${reprText(name)} result`));
}

/**
 * Return a reported tool error as the text the model reads: its message (`String(error)` for a
 * non-`Error`), or its `name` when that is empty (`typeof` for a non-`Error`).
 *
 * @throws TypeCheckError when the message is not yian text (`toolText`).
 */
export function reportedText(name: string, error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return toolText(name, message) || errorName(error);
}

// Local copy of memory/session.ts's addNote: nodes/agent must not import runtime, and session.ts
// keeps its own private.
/**
 * Append the note `text()` to `error`'s own `notes` array, creating it when absent (Python's
 * `add_note`). A failure, such as a frozen error, a non-array `notes`, a primitive or a throwing
 * `text`, loses the note, as Python swallows a failing `add_note`.
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
 * Persist the request and reply of exactly one external model call.
 *
 * `newMessages` is the actual new input, not the whole provider context. A surfaced reply is
 * stored before the stop check, unless `checkReply` rejects it: a rejected reply is not stored, and
 * the effect is recorded `failed` (or `interrupted`, throwing the stop, when a stop is pending).
 * Without a turn, `invoke` runs, the stop is checked, then `checkReply`.
 *
 * @param options.signal The node's stop signal; a stop is pending once it aborted.
 * @param options.exchangeId The caller's correlation id; a fresh one by default.
 * @param options.purpose The request purpose; `chat` by default.
 * @param options.checkReply Called with the provider's return before it is stored or returned;
 *   it throws to reject the reply.
 * @returns The provider's `Response`.
 * @throws CapabilityFault when encoding or publishing messages fails.
 * @throws The stop (`signal.reason`) when it is pending after the reply is stored, or when the call
 *   fails or its reply is rejected while it is pending; the effect is then recorded `interrupted`.
 * @throws ReplyInterrupted when a host interrupt cut the streamed reply, after the effect is
 *   recorded `interrupted` with the partial reply.
 * @throws The provider's error otherwise, unchanged, also when recording the failure fails, except
 *   that an ordinary error is then wrapped in `CapabilityFault`; `checkReply`'s error is recorded
 *   the same way.
 */
export async function invokeModel(
  turn: ConversationTurn | null | undefined,
  invoke: () => Promise<Response>,
  options: {
    newMessages: readonly Message[];
    signal?: AbortSignal;
    exchangeId?: string;
    purpose?: string;
    checkReply?: (response: Response) => void;
  },
): Promise<Response> {
  const { signal, checkReply } = options;
  if (!turn) {
    const response = await invoke();
    signal?.throwIfAborted();
    checkReply?.(response);
    return response;
  }
  const exchangeId = options.exchangeId || newExchangeId();
  const purpose = options.purpose ?? "chat";
  const ordinal = await memoryBoundary(() => {
    const items = options.newMessages.map((message) => encodeMessage(message, exchangeId));
    return turn.request("model", { exchange_id: exchangeId, purpose }, { items });
  }, signal);
  let response: Response;
  try {
    response = await invoke();
  } catch (error) {
    if (error instanceof ReplyInterrupted) {
      await memoryBoundary(
        () => turn.observe(ordinal, "interrupted", { items: [encodeMessage(error.reply, exchangeId)] }),
        signal,
      );
      throw error;
    }
    await observeFailure(turn, ordinal, "model", error, signal);
    throw error;
  }
  // Python also refuses a reply holding a non-finite number here; a TS `Message` can't hold one.
  try {
    checkReply?.(response);
  } catch (rejected) {
    // Without a turn the stop is checked before `checkReply`, so a pending stop wins over any
    // rejection here too, a host fault included.
    if (signal?.aborted) {
      await observeFailure(turn, ordinal, "model", signal.reason, signal);
      throw signal.reason;
    }
    await observeFailure(turn, ordinal, "model", rejected, signal);
    throw rejected;
  }
  await memoryBoundary(
    () => turn.observe(ordinal, "completed", { items: [encodeMessage(response.message, exchangeId)] }),
    signal,
  );
  signal?.throwIfAborted();
  return response;
}

/**
 * Observe `status`, keeping `primary`'s identity when the observation fails.
 *
 * A failed observation is noted on the original error. A `CapabilityFault` `primary` is then left
 * for the caller to rethrow, even when the observation threw the stop: a host fault outranks the
 * stop. Otherwise a stop the observation throws is thrown: a session turn throws a pending stop
 * only after publishing, so the observation did not fail. Failing that, an ordinary `primary` is
 * wrapped in `CapabilityFault`, and a stop `primary` is left for the caller to rethrow.
 */
async function observePreservingError(
  turn: ConversationTurn,
  ordinal: number,
  status: ConversationEffectStatus,
  primary: unknown,
  signal: AbortSignal | undefined,
  failure: MemoryOperationFailure | null = null,
): Promise<void> {
  try {
    await turn.observe(ordinal, status, { failure });
  } catch (secondary) {
    const stopped = isStop(secondary, signal);
    const original = primary instanceof CapabilityFault ? primary.original : primary;
    if (!stopped && secondary !== primary && secondary !== original) {
      addNote(original, () => {
        const message = secondary instanceof Error ? secondary.message : String(secondary);
        return `Conversation observation failed: ${errorName(secondary)}: ${message}`;
      });
    }
    // A host fault keeps priority over the stop, as it does without memory.
    if (primary instanceof CapabilityFault) return;
    if (stopped) throw secondary;
    if (!isStop(primary, signal)) throw new CapabilityFault(primary);
  }
}

/**
 * Record a sanitized external-effect failure; a pending stop wins over an ordinary error.
 *
 * With a stop pending, the effect is recorded `interrupted` and the stop is thrown. Otherwise, and
 * always for a `CapabilityFault`, the caller rethrows `primary`.
 */
async function observeFailure(
  turn: ConversationTurn,
  ordinal: number,
  operation: ConversationEffectKind,
  primary: unknown,
  signal: AbortSignal | undefined,
): Promise<void> {
  const interrupted = isStop(primary, signal);
  // A session turn honors a pending stop only after recording `failed`, so the stop is checked
  // first to record the effect `interrupted`.
  if (!interrupted && !(primary instanceof CapabilityFault) && signal?.aborted) {
    await recordFailure(turn, ordinal, operation, signal.reason, true, signal);
    throw signal.reason;
  }
  await recordFailure(turn, ordinal, operation, primary, interrupted, signal);
}

/** Observe an effect `failed` or `interrupted` with a sanitized failure. */
async function recordFailure(
  turn: ConversationTurn,
  ordinal: number,
  operation: ConversationEffectKind,
  primary: unknown,
  interrupted: boolean,
  signal: AbortSignal | undefined,
): Promise<void> {
  const status = interrupted ? "interrupted" : "failed";
  // External error text may hold credentials or request headers, so only the name is kept.
  const failure = new MemoryOperationFailure(
    operation,
    `api.${operation}.${status}`,
    `${operation[0]!.toUpperCase()}${operation.slice(1)} invocation ${status} (${errorName(primary)}).`,
    false,
  );
  await observePreservingError(turn, ordinal, status, primary, signal, failure);
}

/** The identity of one tool call the model requested. */
export interface ToolCallIdentity {
  /** The granted tool name. */
  name: string;
  /** The requested arguments. */
  arguments: JsonObject;
  /** The model exchange that requested the call. */
  modelExchangeId: string;
  /** The call's zero-based position in that model reply. */
  callIndex: number;
  /** The provider's correlation id, which may repeat across exchanges. */
  toolCallId: string;
}

/** The request payload a run or skipped tool effect stores. */
function toolPayload(call: ToolCallIdentity): JsonObject {
  return {
    name: call.name,
    arguments: call.arguments,
    model_exchange_id: call.modelExchangeId,
    call_index: call.callIndex,
    tool_call_id: call.toolCallId,
  };
}

/** A tool message answering `call` with `content`. */
function toolMessage(call: ToolCallIdentity, content: string, isError: boolean): Message {
  return new Message({
    role: "tool",
    parts: [new ToolResultPart({ callId: call.toolCallId, name: call.name, content, isError })],
  });
}

/**
 * Record a requested tool call a host interrupt skipped, paired with an error result, so every
 * recorded call still has exactly one result.
 *
 * @param options.signal The node's stop signal; a stop the turn throws passes through.
 * @returns The `SKIPPED_TOOL_RESULT` error result standing in for the call.
 * @throws CapabilityFault when encoding or memory access fails.
 */
export async function recordSkippedTool(
  turn: ConversationTurn | null | undefined,
  options: ToolCallIdentity & { signal?: AbortSignal },
): Promise<Message> {
  const message = toolMessage(options, SKIPPED_TOOL_RESULT, true);
  if (!turn) return message;
  await memoryBoundary(async () => {
    const ordinal = await turn.request("tool", toolPayload(options));
    await turn.observe(ordinal, "interrupted", { items: [encodeMessage(message, options.modelExchangeId)] });
  }, options.signal);
  return message;
}

/**
 * Ask a turn that supports it to close as interrupted when the body returns; `interrupt()` is
 * optional, and a turn without it closes as usual.
 *
 * @throws CapabilityFault when the turn's `interrupt` fails.
 */
export async function interruptTurn(turn: ConversationTurn | null | undefined): Promise<void> {
  if (!turn) return;
  await memoryBoundary(() => {
    if (typeof turn.interrupt === "function") turn.interrupt();
  });
}

/**
 * Persist one tool call separately from representing its result.
 *
 * Without a turn, `invoke` runs and the stop is checked.
 *
 * @param invoke The tool's checked invocation.
 * @param options.signal The node's stop signal; a stop is pending once it aborted.
 * @returns The represented result as a tool message, already stored. A `ReportedToolError` gives
 *   an error result of its text (`reportedText`), recorded `completed` like any other result.
 * @throws CapabilityFault when encoding or memory access fails.
 * @throws The stop (`signal.reason`) when it is pending after the result is stored, or when the
 *   tool fails or reports an error while it is pending (the effect is then recorded
 *   `interrupted`), or when representing the result fails while it is pending.
 * @throws The tool's other errors and a representation error (`toolText`'s) otherwise, unchanged;
 *   a representation error leaves the effect `completed` without items.
 */
export async function invokeToolEffect(
  turn: ConversationTurn | null | undefined,
  invoke: () => Promise<unknown>,
  options: ToolCallIdentity & { signal?: AbortSignal },
): Promise<Message> {
  const { name, signal } = options;
  if (!turn) {
    let result: unknown;
    try {
      result = await invoke();
    } catch (error) {
      if (!isReportedToolError(error)) throw error;
      signal?.throwIfAborted();
      return toolMessage(options, reportedText(name, error), true);
    }
    signal?.throwIfAborted();
    return toolMessage(options, toolText(name, result), false);
  }
  const ordinal = await memoryBoundary(() => turn.request("tool", toolPayload(options)), signal);
  let result: unknown;
  // A reported tool error is an object, so `null` means the tool returned.
  let reported: unknown = null;
  try {
    result = await invoke();
  } catch (error) {
    if (!isReportedToolError(error)) {
      await observeFailure(turn, ordinal, "tool", error, signal);
      throw error;
    }
    // A stop pending when the tool reports its error wins, as it does for a failure.
    if (signal?.aborted) {
      await recordFailure(turn, ordinal, "tool", signal.reason, true, signal);
      throw signal.reason;
    }
    reported = error;
  }
  let message: Message;
  let item: ConversationItem;
  try {
    const text = reported === null ? toolText(name, result) : reportedText(name, reported);
    ({ message, item } = await memoryBoundary(() => {
      const message = toolMessage(options, text, reported !== null);
      return { message, item: encodeMessage(message, options.modelExchangeId) };
    }, signal));
  } catch (primary) {
    await observePreservingError(turn, ordinal, "completed", primary, signal);
    throw primary;
  }
  await memoryBoundary(() => turn.observe(ordinal, "completed", { items: [item] }), signal);
  signal?.throwIfAborted();
  return message;
}
