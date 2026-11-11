/**
 * The Anthropic Messages wire format (`POST /v1/messages`).
 *
 * Messages are formatted as LangChain's Anthropic client formats them, so requests stay the
 * same: runs of user and tool messages merge into one `user` turn, leading system messages are
 * lifted to `system`, and a later system message stays in place only where the model accepts
 * it, else it is hoisted to `system` with a warning. An assistant message is sent as its blocks
 * in part order, so a replayed turn goes back as the API sent it; its thinking goes back with its
 * signature to the provider and account that wrote it, from any Claude model. Tool ids the API
 * refuses are hashed to ones it accepts, the same on the call and its result. Tools are sent with
 * their schema as `input_schema`, and a tool choice only with tools. Every call streams.
 *
 * Replies stream text, thinking and tool calls as deltas; each content block becomes a part,
 * with dense part indexes in the order the blocks open. Thinking parts carry replay records
 * holding their signature or redacted data, envelope-only when unsigned. Usage merges across
 * `message_start` and `message_delta`, and the finish comes at `message_stop` or the end of the
 * stream. A reply that failed, was refused, paused or ended early raises `ProviderError`; an
 * error message holds the provider's text uncapped, which `RouteModel` scrubs of the route's
 * credentials and then caps.
 */

import { createHash } from "node:crypto";

import type { JsonObject, JsonValue } from "../../typesys/json.ts";
import { encodeBody, mergeRequestBody, withAuthorWrapper } from "../body.ts";
import { modelProfile } from "../capabilities.ts";
import { type ErrorKind, kindForStatus, ProviderError } from "../errors.ts";
import {
  type Event,
  Finish,
  parseArguments,
  PartDone,
  ReasoningDelta,
  TextDelta,
  ToolCallDelta,
  UsageReport,
} from "../events.ts";
import type { SseProtocol } from "../executor.ts";
import type { SSEEvent } from "../framing.ts";
import { warn } from "../../platform.ts";
import {
  copyJson,
  isPlainObject,
  MediaPart,
  type Message,
  NamedTool,
  type Part,
  ReasoningPart,
  readReplayRecord,
  type ReplayDegradedReason,
  type ReplayKey,
  replayRecord,
  type Request,
  TextPart,
  type ToolChoice,
  ToolCallPart,
  ToolResultPart,
  type ToolSpec,
  Usage,
} from "../messages.ts";
import { eventData, notIntMessage, tokenCount, usageOrProtocolError, wrongPart } from "./common.ts";

/** The protocol id, in `Finish.protocol` and replay records. */
export const PROTOCOL = "anthropic_messages";
const NO_MEDIA = "the anthropic_messages route does not send media parts yet";
// The `tool_choice` type for each choice yian names by string.
const CHOICE_TYPES = { auto: "auto", required: "any", none: "none" } as const;
// The tool ids the API accepts on `tool_use` and `tool_result`.
const TOOL_ID = /^[a-zA-Z0-9_-]+$/;
// The fields of a reasoning part's replay record; it holds exactly one.
const REASONING_FIELDS = ["signature", "redacted"];

// The string field each delta type carries; other delta types (`citations_delta`) are ignored. Read
// before the block lookup, so a delta's field is checked and unknown deltas skipped in one place.
const DELTA_FIELDS: ReadonlyMap<string, string> = new Map([
  ["text_delta", "text"],
  ["thinking_delta", "thinking"],
  ["signature_delta", "signature"],
  ["input_json_delta", "partial_json"],
]);
// Stop reasons that mean the reply failed; any other reason is passed through.
const FAILED_STOP: ReadonlyMap<string, ErrorKind> = new Map<string, ErrorKind>([
  ["refusal", "refused"],
  ["pause_turn", "incomplete"],
  ["model_context_window_exceeded", "context_length"],
]);
// Error event types; `invalid_request_error` is read from its message, any other type is a server error.
const STREAM_ERROR_KINDS: ReadonlyMap<string, ErrorKind> = new Map<string, ErrorKind>([
  ["overloaded_error", "overloaded"],
  ["rate_limit_error", "rate_limit"],
  ["api_error", "server"],
  ["timeout_error", "timeout"],
  ["authentication_error", "auth"],
  ["permission_error", "permission"],
  ["billing_error", "quota"],
  ["not_found_error", "not_found"],
  ["request_too_large", "bad_request"],
]);

/**
 * A content block being streamed. `index` is its part index, `null` for a text block until its
 * first text; a call's `args` is its joined `partial_json`, `null` until the first piece.
 */
type Block =
  | { kind: "text"; index: number | null; text: string }
  | { kind: "thinking"; index: number | null; text: string; signature: string }
  | { kind: "redacted"; index: number | null; data: string }
  | { kind: "call"; index: number | null; id: string; name: string; args: string | null; input: JsonValue | undefined };

/** A turn after merging: consecutive system texts, a user turn's blocks, or an assistant message. */
type Turn =
  | { role: "system"; texts: string[] }
  | { role: "user"; blocks: JsonObject[] }
  | { role: "assistant"; message: Message };

/** What `AnthropicMessages` is built from. */
export interface AnthropicMessagesOptions {
  /** The model name sent as `model`. */
  model: string;
  /** The route's provider name, for error messages and replay records. */
  provider: string;
  /** A non-secret account id, for replay records; replay matches it besides the provider. */
  account?: string | null;
  /** The provider block, merged into every body after yian's fields; copied. */
  block?: JsonObject | null;
  /** Called once per assistant message holding a reasoning record that can't be used. */
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
}

/** The Anthropic Messages wire protocol for one model. */
export class AnthropicMessages implements SseProtocol {
  readonly name = PROTOCOL;
  readonly framing = "sse";
  /** The model name. */
  readonly model: string;
  /** The route's provider name. */
  readonly provider: string;
  /** The account id for replay records, or `null`. */
  readonly account: string | null;
  // Copied so later changes to the caller's block cannot change future bodies.
  readonly #block: JsonObject | null;
  readonly #key: ReplayKey;
  readonly #onReplayDegraded: ((reason: ReplayDegradedReason) => void) | null;
  // The hoist warning is shown once, not on every call.
  #hoistWarned = false;

  constructor(options: AnthropicMessagesOptions) {
    this.model = options.model;
    this.provider = options.provider;
    this.account = options.account ?? null;
    this.#block = options.block ? copyJson(options.block) : null;
    this.#key = { protocol: PROTOCOL, provider: this.provider, model: this.model, account: this.account };
    this.#onReplayDegraded = options.onReplayDegraded ?? null;
  }

  /**
   * The provider block the bodies merge, or `null`; a copy, for callers that choose request
   * fields by it (e.g. structured output under `thinking`).
   */
  get block(): JsonObject | null {
    return this.#block ? copyJson(this.#block) : null;
  }

  /**
   * Build the Messages request body: `model`, `max_tokens` (the model's profile's), `messages`,
   * `system` when there is one, `tools` and `tool_choice` when tools are sent,
   * `output_config: {format: {type: "json_schema", schema}}` when the request has an
   * `outputFormat` (its schema copied; its name and `strict` are not sent, the API having neither),
   * and `stream: true`, then the provider block, whose keys replace yian's.
   *
   * A tool is its name, its parameters as `input_schema`, its description (always) and `strict`
   * when set. The tool choice is sent only with tools; `parallelToolCalls`, when set, becomes
   * `disable_parallel_tool_use` on it (an `auto` choice when none is set), but never on `none`.
   *
   * Runs of user and tool messages merge into one `user` turn, in which whitespace-only text
   * blocks are dropped. A user message's author is sent as the author wrapper, one text block
   * per tag, closed before the turn's next message; the author tag is escaped in user text and
   * tool results. Leading system messages are lifted to `system`. A later one stays in
   * place when the model accepts that, it follows a user turn and an assistant turn or the end
   * comes next; otherwise it is hoisted to `system`, with LangChain's warning through
   * `warn` (once per instance). An assistant message is sent as its blocks in
   * part order: thinking, redacted thinking, text and `tool_use` (an invalid call with empty
   * `input`), one lone text as a string; refusals are not sent. Its reasoning goes back from its
   * replay records to the provider and account that wrote it, from any Claude model, only with a
   * text or call, and is all left out when any reasoning part is unsigned or has a record that
   * can't be used, `onReplayDegraded` being called once for the latter (a throw from the callback
   * propagates). The final assistant message is stripped at the end, and an empty assistant
   * message is skipped unless it is the final one.
   *
   * @throws {TypeError} If a message holds a media part or a part its role cannot carry, a
   *   system message must be hoisted when `system` is already set, or a tool's parameters have
   *   a top-level `oneOf` or `anyOf`.
   */
  buildBody(request: Request): JsonObject {
    const [system, messages] = this.#messages(request.messages.map((message) => withAuthorWrapper(message)));
    const body: JsonObject = {
      model: this.model,
      max_tokens: modelProfile(PROTOCOL, this.model).maxTokens,
      messages,
    };
    if (system !== null) body["system"] = system;
    if (request.tools.length > 0) {
      body["tools"] = request.tools.map(tool);
      const choice = toolChoice(request.toolChoice, request.parallelToolCalls);
      if (choice !== null) body["tool_choice"] = choice;
    }
    if (request.outputFormat !== null) {
      body["output_config"] = { format: { type: "json_schema", schema: copyJson(request.outputFormat.schema) } };
    }
    body["stream"] = true;
    if (this.#block) mergeRequestBody(body, this.#block);
    return body;
  }

  /**
   * Turn a streamed reply's events into stream events.
   *
   * The event type is the data's `type`, else the SSE `event` field; `ping`, unknown events,
   * empty data and non-object data are skipped. Each `text`, `thinking`, `redacted_thinking` and
   * `tool_use` block opens a part, with dense indexes in the order they open; a text block opens
   * at its first text, so an empty one opens none. Other block types open nothing, and deltas or
   * stops for them, or for an index never started, are ignored. Text and thinking stream as
   * deltas, a call as one `ToolCallDelta` with its id and name, then its `partial_json` pieces;
   * a `signature_delta` replaces the thinking's signature. A call's arguments are its joined
   * pieces read by `parseArguments` (not repaired), else its start `input`. The last string stop
   * reason wins, and usage merges key by key across `message_start` and `message_delta`, a
   * `null` value replacing nothing.
   *
   * @param frames The reply's server-sent events, in order.
   * @returns The deltas, a `PartDone` at each block's stop (a thinking part with a record of its
   *   non-empty signature, a redacted one of its non-empty data), then at `message_stop` or the
   *   end of the stream a `PartDone` for each part still open in index order, `UsageReport` when
   *   any usage came, and `Finish`: `length` for `max_tokens`, `tool_calls` for any other reason
   *   when a call came, `stop` for `end_turn`, `stop_sequence` and `tool_use`, else the reason.
   *   Nothing after `message_stop` is read.
   * @throws {ProviderError} For an error event (kind from its type), a `refusal` (`refused`),
   *   `pause_turn` (`incomplete`) or `model_context_window_exceeded` (`context_length`) stop
   *   reason, a stream that ends with no stop reason (`incomplete`, retryable), or an event that
   *   is not JSON or holds a field or usage that cannot be read (`protocol`). The message holds
   *   the provider's text uncapped.
   */
  async *parse(frames: AsyncIterable<SSEEvent>): AsyncGenerator<Event> {
    const provider = this.provider;
    // The started blocks by wire index; a block type that opens no part has none.
    const blocks = new Map<bigint, Block>();
    // The opened parts not yet done, by part index.
    const open = new Map<number, Block>();
    let count = 0;
    const openPart = (block: Block): number => {
      block.index = count++;
      open.set(block.index, block);
      return block.index;
    };
    let calls = false;
    let stopReason: string | null = null;
    let usage: JsonObject | null = null;
    let responseId: string | null = null;
    let responseModel: string | null = null;
    for await (const frame of frames) {
      // Some gateways send keep-alive frames with empty data.
      if (frame.data === "") continue;
      const data = eventData(frame.data, provider);
      if (!isPlainObject(data)) continue;
      const rawType = data["type"];
      const typed = typeof rawType === "string" && rawType !== "";
      const type = typed ? rawType : frame.event;
      // A proxy may send the error object without a type, under any SSE event name.
      if (type === "error" || (!typed && isPlainObject(data["error"]))) throw this.#streamFailure(data["error"]);
      if (type === "message_stop") break;
      const invalid = () => new ProviderError("protocol", `invalid ${type} in ${provider} stream event`);
      if (type === "message_start") {
        const message = data["message"];
        if (!isPlainObject(message)) throw invalid();
        if (typeof message["id"] === "string") responseId = message["id"];
        if (typeof message["model"] === "string") responseModel = message["model"];
        usage = mergeUsage(usage, message["usage"], provider);
      } else if (type === "message_delta") {
        const delta = data["delta"];
        if (!isPlainObject(delta)) throw invalid();
        const reason = delta["stop_reason"];
        if (typeof reason === "string") stopReason = reason;
        else if (reason !== undefined && reason !== null) throw invalid();
        usage = mergeUsage(usage, data["usage"], provider);
      } else if (type === "content_block_start" || type === "content_block_delta" || type === "content_block_stop") {
        const wireIndex = data["index"];
        if (typeof wireIndex !== "bigint") throw invalid();
        if (type === "content_block_start") {
          const content = data["content_block"];
          if (!isPlainObject(content)) throw invalid();
          const block = startBlock(content, invalid);
          if (block === null) continue;
          blocks.set(wireIndex, block);
          if (block.kind === "text") {
            if (block.text) yield new TextDelta(openPart(block), block.text);
          } else if (block.kind === "thinking") {
            const index = openPart(block);
            if (block.text) yield new ReasoningDelta(index, block.text);
          } else if (block.kind === "redacted") {
            openPart(block);
          } else {
            calls = true;
            yield new ToolCallDelta({ index: openPart(block), id: block.id, name: block.name, arguments: "" });
          }
        } else if (type === "content_block_delta") {
          const delta = data["delta"];
          if (!isPlainObject(delta)) throw invalid();
          const deltaType = delta["type"];
          const field = typeof deltaType === "string" ? DELTA_FIELDS.get(deltaType) : undefined;
          if (field === undefined) continue;
          const piece = delta[field];
          if (typeof piece !== "string") throw invalid();
          const block = blocks.get(wireIndex);
          // A delta after its block's stop is dropped: the part is done.
          if (block?.index != null && !open.has(block.index)) continue;
          if (deltaType === "text_delta" && block?.kind === "text") {
            if (!piece) continue;
            block.text += piece;
            yield new TextDelta(block.index ?? openPart(block), piece);
          } else if (deltaType === "thinking_delta" && block?.kind === "thinking") {
            block.text += piece;
            if (piece) yield new ReasoningDelta(block.index!, piece);
          } else if (deltaType === "signature_delta" && block?.kind === "thinking") {
            block.signature = piece;
          } else if (deltaType === "input_json_delta" && block?.kind === "call") {
            block.args = (block.args ?? "") + piece;
            yield new ToolCallDelta({ index: block.index!, id: null, name: null, arguments: piece });
          }
        } else {
          const block = blocks.get(wireIndex);
          if (block?.index == null || !open.delete(block.index)) continue;
          yield new PartDone(block.index, this.#part(block));
        }
      }
    }
    if (stopReason === null) {
      throw new ProviderError("incomplete", `${provider} stream ended before a complete response`, { retryable: true });
    }
    const failed = FAILED_STOP.get(stopReason);
    if (failed !== undefined) {
      throw new ProviderError(failed, `${provider} response did not complete (${stopReason})`);
    }
    // Counted first, so usage that can't be read fails before any part is done.
    const counts = usage === null ? null : usageOf(usage, provider);
    // Parts open in index order and a Map keeps insertion order, so this is index order.
    for (const [index, block] of open) yield new PartDone(index, this.#part(block));
    if (counts !== null) yield new UsageReport(counts);
    const fields: JsonObject = {};
    if (responseId !== null) fields["response_id"] = responseId;
    if (responseModel !== null) fields["response_model"] = responseModel;
    const providerPayload = replayRecord(this.#key, fields);
    let reason = stopReason;
    // `max_tokens` says the calls may be cut short, so `length` wins over them.
    if (stopReason === "max_tokens") reason = "length";
    else if (calls) reason = "tool_calls";
    else if (stopReason === "end_turn" || stopReason === "stop_sequence" || stopReason === "tool_use") reason = "stop";
    yield new Finish({ reason, responseId, protocol: PROTOCOL, providerPayload });
  }

  /**
   * Build a streamed block's part: a thinking part with a record of its signature, a redacted
   * one of its data, each an envelope-only record when that is empty, so every reasoning part
   * names the route that wrote it; text and calls carry no payload.
   */
  #part(block: Block): Part {
    if (block.kind === "text") return new TextPart(block.text);
    if (block.kind === "thinking") {
      const fields: JsonObject = block.signature ? { signature: block.signature } : {};
      return new ReasoningPart(block.text, replayRecord(this.#key, fields));
    }
    if (block.kind === "redacted") {
      return new ReasoningPart("", replayRecord(this.#key, block.data ? { redacted: block.data } : {}));
    }
    const [args, rawArguments] = block.args === null ? startInput(block.input) : parseArguments(block.args);
    return new ToolCallPart({ id: block.id, name: block.name, arguments: args, rawArguments });
  }

  /** Return the error for an error event's error object; its kind from its type. */
  #streamFailure(error: JsonValue | undefined): ProviderError {
    const body = isPlainObject(error) ? error : {};
    const message = typeof body["message"] === "string" ? body["message"] : "";
    const type = body["type"];
    const kind =
      type === "invalid_request_error"
        ? kindForStatus(400, message)
        : ((typeof type === "string" ? STREAM_ERROR_KINDS.get(type) : undefined) ?? "server");
    return new ProviderError(kind, `${this.provider} stream failed (${kind}): ${message || "no message"}`);
  }

  /** Return the `system` field (or `null`) and the rendered turns, as LangChain formats them. */
  #messages(messages: readonly Message[]): [JsonValue | null, JsonObject[]] {
    if (messages.some((message) => message.parts.some((part) => part instanceof MediaPart))) {
      throw new TypeError(NO_MEDIA);
    }
    const turns = merge(messages);
    const lastNonSystem = turns.findLastIndex((turn) => turn.role !== "system");
    const inPlace = modelProfile(PROTOCOL, this.model).inPlaceSystem;
    let system: JsonValue | null = null;
    const rendered: JsonObject[] = [];
    // Later system messages held until the next rendered turn shows whether they stay in place.
    let pending: JsonValue[] = [];
    for (const [i, turn] of turns.entries()) {
      if (turn.role === "system") {
        const content = systemContent(turn.texts);
        if (i === 0) system = content;
        else if (inPlace && rendered.at(-1)?.["role"] === "user") pending.push(content);
        else system = this.#hoist(system, content);
        continue;
      }
      let content: JsonValue;
      if (turn.role === "assistant") {
        content = assistantContent(turn.message, i === lastNonSystem, this.#reasoning(turn.message));
        // The API refuses empty content on every message but the final assistant one.
        if (content === "" && i !== lastNonSystem) continue;
        for (const each of pending) rendered.push({ role: "system", content: each });
      } else {
        content = userContent(turn.blocks);
        for (const each of pending) system = this.#hoist(system, each);
      }
      pending = [];
      rendered.push({ role: turn.role, content });
    }
    for (const each of pending) rendered.push({ role: "system", content: each });
    return [system, rendered];
  }

  /**
   * Read an assistant message's reasoning records, calling `onReplayDegraded` once with the first
   * reason when one can't be used.
   *
   * All of the message's reasoning is left out when any reasoning part is unsigned (no record, or
   * this route's envelope-only record) or has an unusable record: with preserved thinking a
   * signature binds every block before it, so sending part of a turn's reasoning risks a refusal.
   *
   * @returns Each part's `thinking` or `redacted_thinking` block, `null` for a part that isn't
   *   reasoning, all `null` when the reasoning is left out.
   * @throws {TypeError} If the message holds a tool result.
   */
  #reasoning(message: Message): (JsonObject | null)[] {
    const blocks: (JsonObject | null)[] = [];
    let reason: ReplayDegradedReason | null = null;
    let unsigned = false;
    for (const part of message.parts) {
      // Refused here so a bad message never fires onReplayDegraded first.
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
      if (!(part instanceof ReasoningPart)) {
        blocks.push(null);
        continue;
      }
      if (part.providerPayload === null) {
        unsigned = true;
        blocks.push(null);
        continue;
      }
      const read = reasoningBlock(part, this.#key);
      if (!read.ok) reason ??= read.reason;
      else if (read.block === null) unsigned = true;
      blocks.push(read.ok ? read.block : null);
    }
    if (reason !== null) this.#onReplayDegraded?.(reason);
    // An unsigned part is no record mismatch, so nothing is signalled for it.
    return reason !== null || unsigned ? message.parts.map(() => null) : blocks;
  }

  /** Return `content` as the new `system` field, warning the first time that it was moved there. */
  #hoist(system: JsonValue | null, content: JsonValue): JsonValue {
    if (system !== null) throw new TypeError("Received multiple non-consecutive system messages.");
    if (this.#hoistWarned) return content;
    this.#hoistWarned = true;
    // LangChain's text, with the model in single quotes (model ids hold no quote or escape).
    warn(
      "A non-leading `SystemMessage` was moved to the top-level `system` field and now applies to the " +
        "entire conversation. To send it in place, use a supported model and place it after a human or " +
        `tool message, either last or before an AI message (model: '${this.model}').`,
    );
    return content;
  }
}

/**
 * Merge the conversation into turns: consecutive system messages into one, runs of user and
 * tool messages into one user turn (a text block per user text part, a `tool_result` per tool
 * result, in order). A system, user or tool message without parts adds nothing.
 */
function merge(messages: readonly Message[]): Turn[] {
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role === "assistant") {
      turns.push({ role: "assistant", message });
      continue;
    }
    if (message.parts.length === 0) continue;
    const last = turns.at(-1);
    if (message.role === "system") {
      const texts = textsOf(message);
      if (last?.role === "system") last.texts.push(...texts);
      else turns.push({ role: "system", texts });
      continue;
    }
    const blocks = message.role === "user" ? textsOf(message).map(textBlock) : message.parts.map(toolResult);
    if (last?.role === "user") last.blocks.push(...blocks);
    else turns.push({ role: "user", blocks });
  }
  return turns;
}

/** Return a system or user message's texts. */
function textsOf(message: Message): string[] {
  return message.parts.map((part) => {
    if (!(part instanceof TextPart)) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
    return part.text;
  });
}

/** Render a tool message's part as a `tool_result` block; `is_error` is always sent. */
function toolResult(part: Part): JsonObject {
  if (!(part instanceof ToolResultPart)) throw new TypeError(wrongPart(PROTOCOL, part, "tool"));
  return { type: "tool_result", content: part.content, tool_use_id: toolId(part.callId), is_error: part.isError };
}

/** Return a `text` block. */
function textBlock(text: string): JsonObject {
  return { type: "text", text };
}

/** Return a system turn's content: one text as a string, several as text blocks. */
function systemContent(texts: string[]): JsonValue {
  return texts.length === 1 ? texts[0]! : texts.map(textBlock);
}

/** Return a user turn's content: one text block as a string, else the blocks without whitespace-only texts. */
function userContent(blocks: JsonObject[]): JsonValue {
  return loneText(blocks) ?? blocks.filter(isSendable);
}

/**
 * Return an assistant message's content: one lone text block as a string, else its blocks in
 * part order without whitespace-only texts; `""` when it has no text or call.
 *
 * Each text part is its own block and an empty one gives none; a call is a `tool_use` block, an
 * invalid one with empty `input`; a reasoning part is its block in `reasoning`, if any.
 * Refusals are not sent.
 *
 * @param final Whether this is the last non-system message, whose content is stripped at the end.
 * @param reasoning Each part's reasoning block, or `null`.
 */
function assistantContent(message: Message, final: boolean, reasoning: readonly (JsonObject | null)[]): JsonValue {
  const blocks: JsonObject[] = [];
  for (const [index, part] of message.parts.entries()) {
    const thinking = reasoning[index] ?? null;
    if (part instanceof TextPart) {
      if (part.text !== "") blocks.push(textBlock(part.text));
    } else if (part instanceof ToolCallPart) {
      // Copied so changing a body cannot change the message it came from.
      const input = part.arguments === null ? {} : copyJson(part.arguments);
      blocks.push({ type: "tool_use", id: toolId(part.id), name: part.name, input });
    } else if (thinking !== null) {
      blocks.push(thinking);
    }
  }
  const lone = loneText(blocks);
  if (lone !== null) return final ? lone.trimEnd() : lone;
  // Claude often streams a whitespace-only text block before a `tool_use`, which the API refuses back.
  const content = blocks.filter(isSendable);
  // Reasoning goes back only with the text or call it led to.
  if (!content.some((block) => block["type"] === "text" || block["type"] === "tool_use")) return "";
  const last = content.at(-1);
  if (final && last?.["type"] === "text") last["text"] = (last["text"] as string).trimEnd();
  return content;
}

/**
 * Read a reasoning part's replay record and return the block it replays as: `thinking` with its
 * `signature`, or `redacted_thinking` with its `redacted` data; `null` for an envelope-only
 * record, which this route writes for unsigned thinking.
 *
 * A record with fields but not exactly one of them as a non-empty string is `malformed`.
 */
function reasoningBlock(
  part: ReasoningPart,
  key: ReplayKey,
): { ok: true; block: JsonObject | null } | { ok: false; reason: ReplayDegradedReason } {
  // Anthropic accepts thinking another Claude model wrote, so the model need not match.
  const read = readReplayRecord(part.providerPayload, key, REASONING_FIELDS, { matchModel: false });
  if (!read.ok) return read;
  if (Object.keys(read.fields).length === 0) return { ok: true, block: null };
  const { signature, redacted } = read.fields;
  if (Object.keys(read.fields).length === 1) {
    if (typeof signature === "string" && signature !== "") {
      return { ok: true, block: { type: "thinking", thinking: part.text, signature } };
    }
    if (typeof redacted === "string" && redacted !== "") {
      return { ok: true, block: { type: "redacted_thinking", data: redacted } };
    }
  }
  return { ok: false, reason: "malformed" };
}

/**
 * Return the block a `content_block_start` opens, or `null` for a block type that opens no part.
 *
 * @throws {ProviderError} From `invalid`, when a field the block needs is missing or not a string.
 */
function startBlock(content: JsonObject, invalid: () => ProviderError): Block | null {
  // An optional field may be absent or `null`, which is empty.
  const string = (key: string, optional: boolean): string => {
    const value = content[key];
    if (typeof value === "string") return value;
    if (optional && (value === undefined || value === null)) return "";
    throw invalid();
  };
  const type = content["type"];
  if (type === "text") return { kind: "text", index: null, text: string("text", true) };
  if (type === "thinking") {
    return { kind: "thinking", index: null, text: string("thinking", true), signature: string("signature", true) };
  }
  if (type === "redacted_thinking") return { kind: "redacted", index: null, data: string("data", false) };
  if (type !== "tool_use") return null;
  const id = string("id", false);
  const name = string("name", false);
  if (!id || !name) throw invalid();
  return { kind: "call", index: null, id, name, args: null, input: content["input"] };
}

/** Return a call's arguments from its start `input`: missing is `{}`, an object as sent, anything else invalid. */
function startInput(input: JsonValue | undefined): [JsonObject | null, string | null] {
  if (input === undefined) return [{}, null];
  return isPlainObject(input) ? [input, null] : [null, encodeBody(input)];
}

/**
 * Merge an event's `usage` into the usage so far, key by key; a `null` value replaces nothing
 * and a nested object replaces whole. A `usage` that is not an object is ignored.
 *
 * @returns The merged usage, or `merged` when there is nothing to merge.
 * @throws {ProviderError} The event's details are not objects, or a count is not an int within
 *   `Number.MAX_SAFE_INTEGER` (kind `protocol`).
 */
function mergeUsage(merged: JsonObject | null, usage: JsonValue | undefined, provider: string): JsonObject | null {
  if (!isPlainObject(usage)) return merged;
  const cacheCreation = usage["cache_creation"] ?? {};
  const outputDetails = usage["output_tokens_details"] ?? {};
  if (!isPlainObject(cacheCreation) || !isPlainObject(outputDetails)) {
    throw new ProviderError("protocol", `${provider} response usage details are not objects`);
  }
  const notInt = notIntMessage(provider);
  for (const key of ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]) {
    tokenCount(usage[key], notInt);
  }
  for (const key of ephemeralKeys(cacheCreation)) tokenCount(cacheCreation[key], notInt);
  tokenCount(outputDetails["thinking_tokens"], notInt);
  const out: JsonObject = { ...merged };
  for (const [key, value] of Object.entries(usage)) if (value !== null) out[key] = value;
  return out;
}

/** Return the keys of `cache_creation` that count cache writes by time to live. */
function ephemeralKeys(cacheCreation: JsonObject): string[] {
  return Object.keys(cacheCreation).filter((key) => key.startsWith("ephemeral_") && key.endsWith("_input_tokens"));
}

/**
 * Return the token counts of the merged usage. Input counts cache reads and writes; cache writes
 * are the sum of the `cache_creation` breakdown when above zero, else `cache_creation_input_tokens`.
 *
 * @throws {ProviderError} A sum is past `Number.MAX_SAFE_INTEGER` (kind `protocol`).
 */
function usageOf(usage: JsonObject, provider: string): Usage {
  const notInt = notIntMessage(provider);
  const cacheCreation = (usage["cache_creation"] ?? {}) as JsonObject;
  const outputDetails = (usage["output_tokens_details"] ?? {}) as JsonObject;
  let breakdown = 0;
  for (const key of ephemeralKeys(cacheCreation)) breakdown += tokenCount(cacheCreation[key], notInt) ?? 0;
  const cacheRead = tokenCount(usage["cache_read_input_tokens"], notInt);
  const cacheWrite = breakdown > 0 ? breakdown : tokenCount(usage["cache_creation_input_tokens"], notInt);
  const inputTokens = (tokenCount(usage["input_tokens"], notInt) ?? 0) + (cacheRead ?? 0) + (cacheWrite ?? 0);
  const outputTokens = tokenCount(usage["output_tokens"], notInt) ?? 0;
  return usageOrProtocolError(
    {
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + outputTokens,
      cacheReadTokens: cacheRead,
      cacheCreationTokens: cacheWrite,
      reasoningTokens: tokenCount(outputDetails["thinking_tokens"], notInt),
    },
    notInt,
  );
}

/** Return the text of `blocks` when they are one text block, else `null`. */
function loneText(blocks: JsonObject[]): string | null {
  const [first] = blocks;
  return blocks.length === 1 && first!["type"] === "text" ? (first!["text"] as string) : null;
}

/** Return whether a block may go in a list: anything but a whitespace-only text block. */
function isSendable(block: JsonObject): boolean {
  return block["type"] !== "text" || (block["text"] as string).trim() !== "";
}

/**
 * Render a tool: its name, its parameters (copied) as `input_schema`, its description and
 * `strict` when set.
 *
 * @throws {TypeError} If its parameters have a top-level `oneOf` or `anyOf`.
 */
function tool(spec: ToolSpec): JsonObject {
  const combinators = ["oneOf", "anyOf"].filter((key) => Object.hasOwn(spec.parameters, key));
  if (combinators.length > 0) {
    // The tool name in single quotes (tool names the API accepts hold no quote or escape).
    throw new TypeError(
      `tool '${spec.name}' has a top-level ${combinators.join("/")} in its parameters, which the Anthropic API ` +
        "does not support",
    );
  }
  // Copied so changing a body cannot change the spec it came from.
  const rendered: JsonObject = {
    name: spec.name,
    input_schema: copyJson(spec.parameters),
    description: spec.description,
  };
  if (spec.strict !== null) rendered["strict"] = spec.strict;
  return rendered;
}

/**
 * Render the tool choice, or `null` for none. When `parallel` is set, `disable_parallel_tool_use`
 * is its negation, on an `auto` choice when none is set; never on `none`, which has no such field.
 */
function toolChoice(choice: ToolChoice, parallel: boolean | null): JsonObject | null {
  let rendered: JsonObject | null = null;
  if (choice instanceof NamedTool) rendered = { type: "tool", name: choice.name };
  else if (choice !== null) rendered = { type: CHOICE_TYPES[choice] };
  if (parallel !== null && choice !== "none") {
    rendered ??= { type: "auto" };
    rendered["disable_parallel_tool_use"] = !parallel;
  }
  return rendered;
}

/** Return `id`, or when the API refuses it, `toolu_` and the first 24 hex digits of its SHA-256. */
function toolId(id: string): string {
  if (id === "" || TOOL_ID.test(id)) return id;
  return `toolu_${createHash("sha256").update(id).digest("hex").slice(0, 24)}`;
}
