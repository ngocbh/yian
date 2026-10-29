/**
 * The OpenAI Responses API wire format (`POST /responses`).
 *
 * System and user messages become `message` items, an assistant message is replayed from the
 * output items it was parsed from, and tool results become `function_call_output` items. The
 * ChatGPT subscription speaks the same format with its own rules: no stored state, so no
 * message ids; system text moves to `instructions`; encrypted reasoning is included. Every call
 * streams.
 *
 * Replies parse into parts whose payloads are replay records, so the next turn can replay them.
 * A reply that failed or stopped early raises `ProviderError`, except a stop at
 * `max_output_tokens`, which finishes with `length`. An error message holds the provider's text
 * uncapped; `RouteModel` scrubs it of the route's credentials and then caps it.
 */

import type { JsonObject, JsonValue } from "../../typesys/json.ts";
import { encodeBody, mergeRequestBody, withAuthorWrapper } from "../body.ts";
import { ProviderError } from "../errors.ts";
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
import {
  checkJsonValue,
  copyJson,
  isPlainObject,
  MediaPart,
  type Message,
  MessageFormatError,
  NamedTool,
  type OutputFormat,
  type Part,
  put,
  ReasoningPart,
  readReplayRecord,
  RefusalPart,
  type ReplayDegradedReason,
  type ReplayKey,
  replayRecord,
  type Request,
  TextPart,
  ToolCallPart,
  type ToolSpec,
  ToolResultPart,
  Usage,
} from "../messages.ts";
import {
  eventData,
  notIntMessage,
  OPENAI_STREAM_ERROR_KINDS,
  tokenCount,
  usageOrProtocolError,
  wrongPart,
} from "./common.ts";

/** The protocol id, in `Finish.protocol` and replay records. */
export const PROTOCOL = "openai_responses";
/** The subscription's `instructions` when the conversation has no system text. */
export const DEFAULT_INSTRUCTIONS = "You are a helpful assistant.";
const ENCRYPTED_REASONING = "reasoning.encrypted_content";
const NO_MEDIA = "the openai_responses route does not send media parts yet";
const SUBSCRIPTION_TEXT_ONLY = "OpenAI subscription instructions must contain only text";
// The fields each part's replay record may hold besides its envelope.
const ITEM_FIELDS = ["item"];
const TEXT_FIELDS = ["id", "phase", "annotations"];
const REFUSAL_FIELDS = ["id", "phase"];
// Stream events about one output item, which must say which.
const ITEM_EVENTS: ReadonlySet<string> = new Set([
  "response.output_item.added",
  "response.output_item.done",
  "response.content_part.added",
  "response.output_text.delta",
  "response.refusal.delta",
  "response.refusal.done",
  "response.reasoning_summary_text.delta",
  "response.function_call_arguments.delta",
]);
// The subscription backend may end a reply with `response.done` instead of `response.completed`.
const TERMINAL_EVENTS: ReadonlySet<string> = new Set(["response.completed", "response.done", "response.incomplete"]);

/** An assistant part that can carry a replay record. */
type ReplayPart = TextPart | RefusalPart | ReasoningPart | ToolCallPart;

/** What `OpenAIResponses` is built from. */
export interface OpenAIResponsesOptions {
  /** The model name sent as `model` (an Azure deployment name on Azure). */
  model: string;
  /** The route's provider name, for replay records. */
  provider: string;
  /** A non-secret account id for replay records, or null/absent. */
  account?: string | null;
  /** The `openai` provider block, merged into every body after yian's fields; copied. */
  block?: JsonObject | null;
  /** Apply the ChatGPT subscription's rules. */
  subscription?: boolean;
  /** Called once per assistant message whose replay records can't be used. */
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
}

/** The Responses wire protocol for one model. */
export class OpenAIResponses implements SseProtocol {
  readonly name = PROTOCOL;
  readonly framing = "sse";
  /** The model name. */
  readonly model: string;
  /** The route's provider name. */
  readonly provider: string;
  /** The account id for replay records, or `null`. */
  readonly account: string | null;
  /**
   * Whether the subscription's rules apply: `store: false`, system text in `instructions`,
   * encrypted reasoning, no message ids.
   */
  readonly subscription: boolean;
  // Copied so later changes to the caller's block cannot change future bodies.
  readonly #block: JsonObject | null;
  readonly #key: ReplayKey;
  readonly #onReplayDegraded: ((reason: ReplayDegradedReason) => void) | null;

  constructor(options: OpenAIResponsesOptions) {
    this.model = options.model;
    this.provider = options.provider;
    this.account = options.account ?? null;
    this.#block = options.block ? copyJson(options.block) : null;
    this.subscription = options.subscription ?? false;
    this.#key = { protocol: PROTOCOL, provider: this.provider, model: this.model, account: this.account };
    this.#onReplayDegraded = options.onReplayDegraded ?? null;
  }

  /**
   * Build the Responses request body: `model`, `input` and `stream: true`, then the tools, the
   * tool choice and `parallel_tool_calls` when set, then `text.format` when the request sets
   * `outputFormat`, then the provider block, then (on the subscription) its settings.
   *
   * A user message's author is sent as the author wrapper, one `input_text` part per tag; the
   * author tag is escaped in user text and tool output.
   *
   * An assistant message whose replay records can't be used is sent from its neutral fields,
   * and `onReplayDegraded` is called once for it; a throw from the callback propagates.
   *
   * @throws {TypeError} If a message holds a media part, a part its role cannot carry, or (on
   *   the subscription) a system message holds anything but text.
   */
  buildBody(request: Request): JsonObject {
    const [items, instructions] = this.#input(request.messages);
    const body: JsonObject = { model: this.model, input: items, stream: true };
    if (request.tools.length > 0) body["tools"] = request.tools.map(tool);
    const choice = request.toolChoice;
    if (choice instanceof NamedTool) body["tool_choice"] = { type: "function", name: choice.name };
    else if (choice !== null) body["tool_choice"] = choice;
    if (request.parallelToolCalls !== null) body["parallel_tool_calls"] = request.parallelToolCalls;
    if (request.outputFormat !== null) body["text"] = { format: textFormat(request.outputFormat) };
    if (this.#block) mergeRequestBody(body, this.#block);
    if (this.subscription) {
      // Applied after the block so it cannot switch them off: the backend keeps no state, so
      // every turn must carry the full, replayable conversation.
      body["store"] = false;
      body["stream"] = true;
      const include = body["include"];
      const asked = Array.isArray(include) ? include : [];
      body["include"] = [...new Set([...asked, ENCRYPTED_REASONING])];
      body["instructions"] = instructions.join("\n\n") || DEFAULT_INSTRUCTIONS;
      delete body["previous_response_id"];
    }
    return body;
  }

  /**
   * Turn a streamed reply's events into stream events.
   *
   * Deltas stream as they arrive; each `output_item.done` gives one `PartDone` per part of the
   * done item. At the terminal event each streamed output index is matched to at most one item
   * of its `output`: by the item ids and call ids its events gave, wherever the item sits; else
   * the item at its own position, when the stream gave none of that item's ids; else, for an
   * unfinished index whose events gave no ids, the first item left whose ids the stream never
   * gave. An item that never got its done event takes its parts from its match; with none, its
   * deltas build its parts. Items of that `output` no index took, and whose ids the stream never
   * gave, give their parts after all the others. Part indexes are dense: streamed parts in the
   * order each (output index, content index) is first seen, then each such item's parts in
   * order. An added item's own text or arguments are ignored, since the deltas that follow carry
   * them. Unknown events are ignored, and a `[DONE]` sentinel ends the stream.
   *
   * @param frames The reply's server-sent events, in order.
   * @returns The deltas and parts, then `UsageReport` (when the reply has usage) and `Finish` on
   *   `response.completed` (or `response.done`) or a length-truncated `response.incomplete` (or
   *   `response.done`).
   * @throws {ProviderError} For `response.failed`, a terminal reply with status `failed` that
   *   carries an error, or an `error` event (kind from the error code, else its type), a stream
   *   that ends without a terminal event or a reply that did not complete (kind `incomplete`, or
   *   `refused` for a content filter), or an event that is not JSON, lacks the output index it
   *   needs, or holds an item or usage that cannot be read (kind `protocol`). The message holds
   *   the provider's text uncapped.
   */
  async *parse(frames: AsyncIterable<SSEEvent>): AsyncGenerator<Event> {
    // Part index by output index, then content index.
    const indexes = new Map<number, Map<number, number>>();
    let count = 0;
    const index = (outputIndex: number, contentIndex = 0): number => {
      let slots = indexes.get(outputIndex);
      if (slots === undefined) indexes.set(outputIndex, (slots = new Map()));
      let slot = slots.get(contentIndex);
      if (slot === undefined) slots.set(contentIndex, (slot = count++));
      return slot;
    };
    const opened = new Map<number, JsonObject>();
    const finished = new Set<number>();
    // The item ids and call ids the stream gave, by output index.
    const streamedIds = new Map<number, Set<string>>();
    const sawIds = (outputIndex: number, ids: string[]): void => {
      let known = streamedIds.get(outputIndex);
      if (known === undefined) streamedIds.set(outputIndex, (known = new Set()));
      for (const id of ids) known.add(id);
    };
    const lastSummary = new Map<number, JsonValue | undefined>();
    // Refusal text by output index, then content index.
    const refusals = new Map<number, Map<number, string>>();
    const refusalTexts = (outputIndex: number): Map<number, string> => {
      let texts = refusals.get(outputIndex);
      if (texts === undefined) refusals.set(outputIndex, (texts = new Map()));
      return texts;
    };
    let sawCalls = false;
    let responseId: string | null = null;
    for await (const frame of frames) {
      // The openai SDK stops at a `[DONE]` sentinel; the stream is then over.
      if (frame.data.startsWith("[DONE]")) break;
      // Some gateways send keep-alive frames with empty data.
      if (frame.data === "") continue;
      const data = eventData(frame.data, this.provider);
      if (!isPlainObject(data)) continue;
      const rawType = data["type"] || frame.event;
      const type = typeof rawType === "string" ? rawType : null;
      const nested = data["error"];
      if (type === "error" || isPlainObject(nested)) {
        throw this.#streamFailure(isPlainObject(nested) ? nested : data);
      }
      let outputIndex = -1;
      if (type !== null && ITEM_EVENTS.has(type)) {
        const raw = data["output_index"];
        if (typeof raw !== "bigint") {
          throw new ProviderError("protocol", `${this.provider} stream event ${type} has no output_index`);
        }
        outputIndex = Number(raw);
        if (typeof data["item_id"] === "string") sawIds(outputIndex, [data["item_id"]]);
      }
      const rawContentIndex = data["content_index"];
      const contentIndex = typeof rawContentIndex === "bigint" ? Number(rawContentIndex) : 0;
      const rawDelta = data["delta"];
      let delta = typeof rawDelta === "string" ? rawDelta : "";
      if (type === "response.created") {
        const response = data["response"];
        if (isPlainObject(response) && typeof response["id"] === "string") responseId = response["id"];
      } else if (type === "response.output_item.added") {
        const item = data["item"];
        if (!isPlainObject(item)) continue;
        opened.set(outputIndex, item);
        sawIds(outputIndex, itemIds(item));
        if (item["type"] === "function_call") {
          sawCalls = true;
          yield new ToolCallDelta({
            index: index(outputIndex),
            id: strOrNull(item["call_id"]),
            name: strOrNull(item["name"]),
            arguments: "",
          });
        } else if (item["type"] === "reasoning") {
          index(outputIndex);
        }
      } else if (type === "response.content_part.added") {
        // Taking the index here keeps a content part that never gets a delta (an empty text)
        // in its place.
        index(outputIndex, contentIndex);
      } else if (type === "response.output_text.delta") {
        yield new TextDelta(index(outputIndex, contentIndex), delta);
      } else if (type === "response.refusal.delta") {
        index(outputIndex, contentIndex);
        const texts = refusalTexts(outputIndex);
        texts.set(contentIndex, (texts.get(contentIndex) ?? "") + delta);
      } else if (type === "response.refusal.done") {
        // The part itself comes from `output_item.done`; the final text is kept in case that
        // event never arrives.
        index(outputIndex, contentIndex);
        const text = data["refusal"];
        if (typeof text === "string") refusalTexts(outputIndex).set(contentIndex, text);
      } else if (type === "response.reasoning_summary_text.delta") {
        const partIndex = index(outputIndex);
        const summaryIndex = data["summary_index"];
        // Summary parts are joined with newlines in the final part; the deltas add the same
        // separator so the live text matches it.
        if (lastSummary.has(partIndex) && lastSummary.get(partIndex) !== summaryIndex) delta = "\n" + delta;
        lastSummary.set(partIndex, summaryIndex);
        yield new ReasoningDelta(partIndex, delta);
      } else if (type === "response.function_call_arguments.delta") {
        yield new ToolCallDelta({ index: index(outputIndex), id: null, name: null, arguments: delta });
      } else if (type === "response.output_item.done") {
        const item = data["item"];
        if (!isPlainObject(item)) continue;
        finished.add(outputIndex);
        sawIds(outputIndex, itemIds(item));
        if (item["type"] === "function_call") sawCalls = true;
        for (const [itemContentIndex, part] of itemParts(item, this.#key)) {
          yield new PartDone(index(outputIndex, itemContentIndex), part);
        }
      } else if (type !== null && TERMINAL_EVENTS.has(type)) {
        const response = data["response"];
        if (!isPlainObject(response)) {
          throw new ProviderError("protocol", `${this.provider} ${type} event has no response`);
        }
        const rawOutput = response["output"];
        const output = Array.isArray(rawOutput) ? rawOutput : [];
        const started = new Set([...opened.keys(), ...indexes.keys()]);
        const unfinished = [...started].filter((each) => !finished.has(each)).sort((a, b) => a - b);
        const { matches, rest } = matchTerminal(output, started, finished, streamedIds);
        for (const each of unfinished) {
          const item = matches.get(each);
          if (item === undefined) {
            // Text, reasoning and calls are built from their deltas; a refusal streams none, so
            // its part is built here.
            const openedItem = opened.get(each) ?? {};
            for (const [refusalIndex, text] of refusals.get(each) ?? []) {
              yield new PartDone(index(each, refusalIndex), refusalPart(text, openedItem, this.#key));
            }
            continue;
          }
          for (const [itemContentIndex, part] of itemParts(item, this.#key)) {
            if (part instanceof ToolCallPart) sawCalls = true;
            yield new PartDone(index(each, itemContentIndex), part);
          }
        }
        // Output items no streamed index took, and whose ids the stream never gave, still belong
        // to the reply, after the others.
        for (const item of rest) {
          // A new slot each: its position may also be an output index the stream used.
          for (const [, part] of itemParts(item, this.#key)) {
            if (part instanceof ToolCallPart) sawCalls = true;
            yield new PartDone(count++, part);
          }
        }
        yield* this.#finish(response, sawCalls, responseId, type);
        return;
      } else if (type === "response.failed") {
        const response = data["response"];
        const error = isPlainObject(response) ? response["error"] : undefined;
        throw this.#streamFailure(isPlainObject(error) ? error : {});
      }
    }
    const message = this.subscription
      ? "OpenAI subscription stream ended before a complete response"
      : `${this.provider} stream ended before a complete response`;
    throw new ProviderError("incomplete", message, { retryable: true });
  }

  /**
   * Return a finished reply's `UsageReport` and `Finish`, or throw if it did not finish.
   *
   * `response.completed` must carry status `completed`, and `response.done` status `completed`
   * or `incomplete`. Among incomplete replies only a stop at `max_output_tokens` signalled by
   * `response.incomplete`, or by `response.done` with status `incomplete`, counts as finished;
   * it finishes with `length` even when the reply has calls, since a truncated call cannot be
   * trusted. A reply with status `failed` that carries an `error` object fails as
   * `response.failed` does.
   *
   * @param response The terminal event's response.
   * @param hasCalls Whether the reply holds a function call.
   * @param responseId The id to use when `response` has none.
   * @param type The terminal event type.
   * @throws {ProviderError} The reply failed with an error (kind from its code, else its type),
   *   did not complete (kind `incomplete`, or `refused` for a content filter), or its usage
   *   cannot be read (kind `protocol`).
   */
  #finish(response: JsonObject, hasCalls: boolean, responseId: string | null, type: string): Event[] {
    const status = type === "response.incomplete" ? "incomplete" : response["status"];
    const details = response["incomplete_details"];
    const reason = isPlainObject(details) ? details["reason"] : undefined;
    let finishReason: string;
    if (status === "completed") {
      finishReason = hasCalls ? "tool_calls" : "stop";
    } else if (status === "incomplete" && reason === "max_output_tokens" && type !== "response.completed") {
      finishReason = "length";
    } else if (status === "failed" && isPlainObject(response["error"])) {
      throw this.#streamFailure(response["error"]);
    } else {
      const prefix = this.subscription ? "OpenAI subscription response" : `${this.provider} response`;
      const shown = typeof reason === "string" && reason ? reason : "unknown";
      const kind = reason === "content_filter" ? "refused" : "incomplete";
      throw new ProviderError(kind, `${prefix} did not complete (${shown})`);
    }
    const events: Event[] = [];
    const usage = usageOf(response["usage"], this.provider);
    if (usage !== null) events.push(new UsageReport(usage));
    // `||`, not `??`: an empty id counts as none.
    const id = strOrNull(response["id"]) || responseId;
    const fields: JsonObject = {};
    if (id !== null) fields["response_id"] = id;
    // The model that answered, which may differ from the one asked for (an Azure deployment name).
    if (typeof response["model"] === "string") fields["response_model"] = response["model"];
    const providerPayload = replayRecord(this.#key, fields);
    events.push(new Finish({ reason: finishReason, responseId: id, protocol: PROTOCOL, providerPayload }));
    return events;
  }

  /**
   * Return the error for `response.failed`, a failed terminal reply or an `error` event; its
   * kind from the error's code, else its type.
   */
  #streamFailure(error: JsonObject): ProviderError {
    // An error without a string code may still name its kind in `type` (`invalid_request_error`).
    const code = typeof error["code"] === "string" ? error["code"] : error["type"];
    const kind = (typeof code === "string" ? OPENAI_STREAM_ERROR_KINDS.get(code) : undefined) ?? "server";
    const rawMessage = error["message"];
    const message = typeof rawMessage === "string" && rawMessage ? rawMessage : "no message";
    const text = this.subscription
      ? `OpenAI subscription stream ended before a complete response: ${message}`
      : `${this.provider} stream failed (${kind}): ${message}`;
    return new ProviderError(kind, text);
  }

  /**
   * Render the conversation as `input` items.
   *
   * @returns The items, and on the subscription the system messages' texts, which it sends as
   *   `instructions` instead.
   */
  #input(messages: readonly Message[]): [JsonObject[], string[]] {
    const items: JsonObject[] = [];
    const instructions: string[] = [];
    // The call ids of every function call sent so far, kept as items are appended.
    const callIds = new Set<string>();
    for (const message of messages.map((each) => withAuthorWrapper(each))) {
      if (message.role === "system" && this.subscription) {
        const text = instructionText(message);
        if (text) instructions.push(text);
      } else if (message.role === "system" || message.role === "user") {
        const item = inputMessage(message);
        if (item !== null) items.push(item);
      } else if (message.role === "assistant") {
        this.#assistant(message, items, callIds);
      } else {
        for (const part of message.parts) {
          if (part instanceof MediaPart) throw new TypeError(NO_MEDIA);
          if (!(part instanceof ToolResultPart)) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
          // Responses has no error flag; the error text is the output.
          items.push({ type: "function_call_output", call_id: part.callId, output: part.content });
        }
      }
    }
    return [items, instructions];
  }

  /**
   * Append an assistant message's items to `items`, in part order.
   *
   * Calls without a record go after the message's other items, and are skipped when their call
   * id was already sent.
   *
   * @param callIds The call ids of the function calls in `items`; updated as calls are appended.
   */
  #assistant(message: Message, items: JsonObject[], callIds: Set<string>): void {
    const records = this.#records(message);
    const fallbackCalls: ToolCallPart[] = [];
    for (const [index, part] of message.parts.entries()) {
      const record = records[index] ?? null;
      if (part instanceof ReasoningPart) {
        if (record !== null) appendPayloadItem(dropNulls(record["item"] as JsonObject), items, callIds);
      } else if (part instanceof ToolCallPart) {
        if (record === null) {
          fallbackCalls.push(part);
          continue;
        }
        const item = dropNulls(record["item"] as JsonObject);
        // The ChatGPT backend has never received a replayed call's status.
        if (this.subscription) delete item["status"];
        appendPayloadItem(item, items, callIds);
      } else if (part instanceof TextPart || part instanceof RefusalPart) {
        this.#messageContent(part, record, items);
      }
    }
    for (const call of fallbackCalls) {
      if (callIds.has(call.id)) continue;
      callIds.add(call.id);
      // The constructor guarantees raw text when the arguments did not parse.
      const args = call.arguments !== null ? encodeBody(call.arguments, "tool call arguments") : call.rawArguments!;
      items.push({ type: "function_call", name: call.name, arguments: args, call_id: call.id });
    }
  }

  /**
   * Read an assistant message's replay records, calling `onReplayDegraded` when one can't be used.
   *
   * The whole message falls back to its neutral fields because the API refuses a replayed call
   * or message item whose paired reasoning item is missing; so does a message holding reasoning
   * without a payload, or (on the subscription) without encrypted content.
   *
   * @returns Each part's record fields, `null` for a part without a payload; all `null` when the
   *   message renders neutrally.
   * @throws {TypeError} If the message holds a media part or a tool result.
   */
  #records(message: Message): (JsonObject | null)[] {
    const records: (JsonObject | null)[] = [];
    let reason: ReplayDegradedReason | null = null;
    // Set by reasoning that can't be replayed with no record at fault: no payload, or (on the
    // subscription) no encrypted content.
    let unpaired = false;
    for (const part of message.parts) {
      if (part instanceof MediaPart) throw new TypeError(NO_MEDIA);
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
      if (part.providerPayload === null) {
        if (part instanceof ReasoningPart) unpaired = true;
        records.push(null);
        continue;
      }
      const read = readRecord(part, this.#key);
      if (!read.ok) {
        reason ??= read.reason;
        records.push(null);
        continue;
      }
      records.push(read.fields);
      if (this.subscription && part instanceof ReasoningPart && !encrypted(read.fields["item"] as JsonObject)) {
        unpaired = true;
      }
    }
    const neutral = message.parts.map(() => null);
    if (reason !== null) {
      this.#onReplayDegraded?.(reason);
      return neutral;
    }
    // Reasoning without a payload can't be replayed, nor (without stored state) can unencrypted
    // reasoning; that is no record mismatch, so nothing is signalled.
    return unpaired ? neutral : records;
  }

  /**
   * Add a text or refusal part to its message item, opening one when needed.
   *
   * A part joins an earlier message item only when its id is sent and that item carries it.
   *
   * @param record The part's record fields, or `null` to render it without an id or phase.
   */
  #messageContent(part: TextPart | RefusalPart, record: JsonObject | null, items: JsonObject[]): void {
    const fields = record ?? {};
    const content: JsonObject =
      part instanceof TextPart
        ? { type: "output_text", text: part.text, annotations: dropNulls(fields["annotations"] ?? []) }
        : { type: "refusal", refusal: part.text };
    // The subscription keeps no state, so it has no message ids to refer to.
    const id = this.subscription ? null : (fields["id"] as string | undefined) || null;
    const phase = fields["phase"] ?? null;
    if (id !== null) {
      for (const item of items) {
        if (item["type"] === "message" && item["id"] === id) {
          (item["content"] as JsonValue[]).push(content);
          if (phase !== null) item["phase"] = phase;
          return;
        }
      }
    }
    const item: JsonObject = { type: "message", role: "assistant", content: [content] };
    if (id !== null) item["id"] = id;
    if (phase !== null) item["phase"] = phase;
    items.push(item);
  }
}

/**
 * Read a part's replay record and check its shape for the part.
 *
 * A reasoning or call record's `item` must be an object of the part's item type, a call's with
 * the part's call id; a text or refusal record's `id` and `phase` must be strings and its
 * `annotations` an array, when present.
 */
function readRecord(
  part: ReplayPart,
  key: ReplayKey,
): { ok: true; fields: JsonObject } | { ok: false; reason: ReplayDegradedReason } {
  const allowed = part instanceof TextPart ? TEXT_FIELDS : part instanceof RefusalPart ? REFUSAL_FIELDS : ITEM_FIELDS;
  const read = readReplayRecord(part.providerPayload, key, allowed);
  if (!read.ok) return read;
  const fields = read.fields;
  let fits: boolean;
  if (part instanceof ReasoningPart || part instanceof ToolCallPart) {
    const item = fields["item"];
    const type = part instanceof ReasoningPart ? "reasoning" : "function_call";
    fits =
      isPlainObject(item) && item["type"] === type && (part instanceof ReasoningPart || item["call_id"] === part.id);
  } else {
    fits =
      (!Object.hasOwn(fields, "id") || typeof fields["id"] === "string") &&
      (!Object.hasOwn(fields, "phase") || typeof fields["phase"] === "string") &&
      (!Object.hasOwn(fields, "annotations") || Array.isArray(fields["annotations"]));
  }
  return fits ? read : { ok: false, reason: "malformed" };
}

/**
 * Match a terminal event's `output` items to the streamed output indexes.
 *
 * Each streamed output index (in `started` or `finished`) takes at most one item, and each item
 * goes to at most one index. By ids first, wherever the item sits, since a gateway may renumber
 * output indexes; then the item at the index's own position, when the stream gave none of its
 * ids, since a gateway may also rewrite ids for the final response; then, for an unfinished
 * index whose events gave no ids, the first item left that has ids the stream never gave.
 *
 * @param output The terminal response's output items.
 * @param started The output indexes an item was added at or a part was indexed for.
 * @param finished The output indexes whose item got its done event.
 * @param streamedIds The item ids and call ids each output index's events gave.
 * @returns The item each matched index took, and the items no index took whose ids the stream
 *   never gave, in `output` order.
 */
export function matchTerminal(
  output: readonly JsonValue[],
  started: ReadonlySet<number>,
  finished: ReadonlySet<number>,
  streamedIds: ReadonlyMap<number, ReadonlySet<string>>,
): { matches: Map<number, JsonValue>; rest: JsonValue[] } {
  const seenIds = new Set([...streamedIds.values()].flatMap((ids) => [...ids]));
  const streamed = [...new Set([...started, ...finished])].sort((a, b) => a - b);
  const unfinished = [...started].filter((each) => !finished.has(each)).sort((a, b) => a - b);
  const matches = new Map<number, JsonValue>();
  const claimed = new Set<number>();
  const take = (each: number, position: number): void => {
    if (position < 0 || claimed.has(position) || matches.has(each)) return;
    claimed.add(position);
    matches.set(each, output[position]!);
  };
  const fresh = (item: JsonValue): boolean => !itemIds(item).some((id) => seenIds.has(id));
  for (const each of streamed) {
    const ids = streamedIds.get(each);
    if (ids !== undefined && ids.size > 0) {
      take(each, output.findIndex((item) => itemIds(item).some((id) => ids.has(id))));
    }
  }
  for (const each of streamed) {
    if (each < output.length && fresh(output[each]!)) take(each, each);
  }
  // An unfinished index the stream gave no ids takes the first item left whose ids it never
  // gave, so no item of `output` is shown twice.
  for (const each of unfinished) {
    if (matches.has(each) || (streamedIds.get(each)?.size ?? 0) > 0) continue;
    take(
      each,
      output.findIndex((item, position) => !claimed.has(position) && itemIds(item).length > 0 && fresh(item)),
    );
  }
  const rest = output.filter((item, position) => !claimed.has(position) && fresh(item));
  return { matches, rest };
}

/** Return an output item's string `id`, and its string `call_id` when it is a function call. */
function itemIds(item: JsonValue): string[] {
  if (!isPlainObject(item)) return [];
  const ids = [item["id"], item["type"] === "function_call" ? item["call_id"] : undefined];
  return ids.filter((id) => typeof id === "string");
}

/** Return `value` when it is a string, else `null`. */
function strOrNull(value: JsonValue | undefined): string | null {
  return typeof value === "string" ? value : null;
}

/**
 * Return the token counts of a Responses `usage` object; `null` when it is not an object.
 *
 * Missing or `null` input and output counts are 0, and a missing total is their sum. Cache and
 * reasoning counts are read whatever the service tier.
 *
 * @throws {ProviderError} A count is not an int within `Number.MAX_SAFE_INTEGER`, or the
 *   details are not objects (kind `protocol`).
 */
function usageOf(usage: JsonValue | undefined, provider: string): Usage | null {
  if (!isPlainObject(usage)) return null;
  const inputDetails = usage["input_tokens_details"] ?? {};
  const outputDetails = usage["output_tokens_details"] ?? {};
  if (!isPlainObject(inputDetails) || !isPlainObject(outputDetails)) {
    throw new ProviderError("protocol", `${provider} response usage details are not objects`);
  }
  const notInt = notIntMessage(provider);
  const inputTokens = tokenCount(usage["input_tokens"], notInt) ?? 0;
  const outputTokens = tokenCount(usage["output_tokens"], notInt) ?? 0;
  return usageOrProtocolError(
    {
      inputTokens,
      outputTokens,
      totalTokens: tokenCount(usage["total_tokens"], notInt) ?? inputTokens + outputTokens,
      cacheReadTokens: tokenCount(inputDetails["cached_tokens"], notInt),
      cacheCreationTokens: tokenCount(inputDetails["cache_write_tokens"], notInt),
      reasoningTokens: tokenCount(outputDetails["reasoning_tokens"], notInt),
    },
    notInt,
  );
}

/**
 * Return the parts of one output item, each with its content index.
 *
 * A reasoning item or function call gives one part at content index 0; a message gives one part
 * per text or refusal content part. Other items give none.
 *
 * @throws {ProviderError} The item cannot become a part (kind `protocol`).
 */
function itemParts(item: JsonValue, key: ReplayKey): [number, Part][] {
  if (!isPlainObject(item)) {
    throw new ProviderError("protocol", `${key.provider} response has an output item that is not an object`);
  }
  const type = item["type"];
  try {
    // Bounds the depth before the builders recurse into the item.
    checkJsonValue(item, "item");
    if (type === "reasoning") return [[0, reasoningPart(item, key)]];
    if (type === "message") return messageParts(item, key);
    if (type === "function_call") return [[0, callPart(item, key)]];
    return [];
  } catch (error) {
    if (error instanceof MessageFormatError) {
      throw new ProviderError("protocol", `${key.provider} response has an invalid ${type} item: ${error.message}`);
    }
    throw error;
  }
}

/** Build a reasoning part: the summary texts joined with newlines, the item in its record. */
function reasoningPart(item: JsonObject, key: ReplayKey): ReasoningPart {
  const payload = dropNulls(item);
  const summary = payload["summary"];
  const texts: string[] = [];
  for (const entry of Array.isArray(summary) ? summary : []) {
    if (isPlainObject(entry) && typeof entry["text"] === "string") texts.push(entry["text"]);
  }
  return new ReasoningPart(texts.join("\n"), replayRecord(key, { item: payload }));
}

/** Build a message item's text and refusal parts, keyed by content index. */
function messageParts(item: JsonObject, key: ReplayKey): [number, Part][] {
  const content = item["content"] ?? [];
  if (!Array.isArray(content)) {
    throw new ProviderError("protocol", `${key.provider} response message content is not a list`);
  }
  const fields = contentFields(item["id"], item["phase"]);
  const parts: [number, Part][] = [];
  for (const [contentIndex, entry] of content.entries()) {
    if (!isPlainObject(entry)) continue;
    if (entry["type"] === "output_text") {
      const text = entry["text"];
      if (typeof text !== "string") {
        throw new ProviderError("protocol", `${key.provider} response output_text has no text`);
      }
      const annotations = entry["annotations"];
      const kept = Array.isArray(annotations) ? dropNulls(annotations) : [];
      parts.push([contentIndex, new TextPart(text, replayRecord(key, { ...fields, annotations: kept }))]);
    } else if (entry["type"] === "refusal") {
      const refusal = entry["refusal"];
      if (typeof refusal !== "string") {
        throw new ProviderError("protocol", `${key.provider} response refusal has no text`);
      }
      parts.push([contentIndex, new RefusalPart(refusal, replayRecord(key, fields))]);
    }
  }
  return parts;
}

/**
 * Build a refusal part from streamed text, for an item the reply never finished.
 *
 * @param item The added message item, whose id and phase go in the record.
 */
function refusalPart(text: string, item: JsonObject, key: ReplayKey): RefusalPart {
  // The record holds only strings the JSON parser checked, so it always builds.
  return new RefusalPart(text, replayRecord(key, contentFields(item["id"], item["phase"])));
}

/** Return a text or refusal part's record fields: the message id and its phase, each when a string. */
function contentFields(id: JsonValue | undefined, phase: JsonValue | undefined): JsonObject {
  const fields: JsonObject = {};
  if (typeof id === "string") fields["id"] = id;
  if (typeof phase === "string") fields["phase"] = phase;
  return fields;
}

/** Build a tool call from a function call item, the item in its record; invalid arguments are kept raw. */
function callPart(item: JsonObject, key: ReplayKey): ToolCallPart {
  const payload = dropNulls(item);
  const { call_id: id, name, arguments: text } = payload;
  if (typeof id !== "string" || typeof name !== "string" || typeof text !== "string") {
    throw new ProviderError("protocol", `${key.provider} response function_call needs call_id, name and arguments`);
  }
  const [args, raw] = parseArguments(text);
  const providerPayload = replayRecord(key, { item: payload });
  return new ToolCallPart({ id, name, arguments: args, rawArguments: raw, providerPayload });
}

/** Return whether a reasoning item holds non-empty `encrypted_content`. */
function encrypted(item: JsonObject): boolean {
  const content = item["encrypted_content"];
  return typeof content === "string" && content !== "";
}

/** Append a replayed item, recording its call id when it is a function call. */
function appendPayloadItem(item: JsonObject, items: JsonObject[], callIds: Set<string>): void {
  items.push(item);
  const callId = item["call_id"];
  if (item["type"] === "function_call" && typeof callId === "string") callIds.add(callId);
}

/** Copy a JSON value without its `null` object members, at every depth. */
function dropNulls<T extends JsonValue>(value: T): T {
  if (typeof value !== "object" || value === null) return value;
  if (Array.isArray(value)) return value.map(dropNulls) as T;
  const copy: JsonObject = {};
  for (const key of Object.keys(value)) {
    const member = value[key]!;
    if (member !== null) put(copy, key, dropNulls(member));
  }
  return copy as T;
}

/**
 * Render a system or user message; `null` when it has no parts.
 *
 * One text part is sent as a string, several as `input_text` parts.
 */
function inputMessage(message: Message): JsonObject | null {
  const texts: string[] = [];
  for (const part of message.parts) {
    if (part instanceof MediaPart) throw new TypeError(NO_MEDIA);
    if (!(part instanceof TextPart)) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
    texts.push(part.text);
  }
  if (texts.length === 0) return null;
  const content: JsonValue = texts.length === 1 ? texts[0]! : texts.map((text) => ({ type: "input_text", text }));
  return { type: "message", role: message.role, content };
}

/**
 * Return a system message's text for the subscription's `instructions`.
 *
 * The parts are joined with no separator, as the subscription joins a system message's text
 * blocks today.
 */
function instructionText(message: Message): string {
  if (message.parts.some((part) => part instanceof MediaPart)) throw new TypeError(NO_MEDIA);
  const texts: string[] = [];
  for (const part of message.parts) {
    if (!(part instanceof TextPart)) throw new TypeError(SUBSCRIPTION_TEXT_ONLY);
    texts.push(part.text);
  }
  return texts.join("");
}

/** Render a tool as a Responses function tool; `strict` only when set. */
function tool(spec: ToolSpec): JsonObject {
  const rendered: JsonObject = {
    type: "function",
    name: spec.name,
    description: spec.description,
    // Copied so changing a body cannot change the spec it came from.
    parameters: copyJson(spec.parameters),
  };
  if (spec.strict !== null) rendered["strict"] = spec.strict;
  return rendered;
}

/** Render an output format as a `text.format` entry, `strict` only when set. */
function textFormat(format: OutputFormat): JsonObject {
  // Copied so changing a body cannot change the format it came from.
  const rendered: JsonObject = { type: "json_schema", name: format.name, schema: copyJson(format.schema) };
  if (format.strict !== null) rendered["strict"] = format.strict;
  return rendered;
}
