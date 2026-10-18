/**
 * The OpenAI Responses API wire format (`POST /responses`).
 *
 * System and user messages become `message` items, an assistant message is replayed from the
 * output items it was parsed from, and tool results become `function_call_output` items. The
 * ChatGPT subscription speaks the same format with its own rules: no stored state, so no
 * message ids; system text moves to `instructions`; encrypted reasoning is included. Every call
 * streams.
 */

import type { JsonObject, JsonValue } from "../../typesys/json.ts";
import { encodeBody, mergeRequestBody } from "../body.ts";
import type { Event } from "../events.ts";
import type { SseProtocol } from "../executor.ts";
import type { SSEEvent } from "../framing.ts";
import {
  copyJson,
  isPlainObject,
  MediaPart,
  type Message,
  NamedTool,
  type Part,
  put,
  ReasoningPart,
  readReplayRecord,
  RefusalPart,
  type ReplayDegradedReason,
  type ReplayKey,
  type Request,
  TextPart,
  ToolCallPart,
  type ToolSpec,
  ToolResultPart,
} from "../messages.ts";

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
   * tool choice and `parallel_tool_calls` when set, then the provider block, then (on the
   * subscription) its settings.
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

  /** Turn a streamed reply's events into stream events. */
  async *parse(_frames: AsyncIterable<SSEEvent>): AsyncGenerator<Event> {
    throw new Error("not implemented");
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
    for (const message of messages) {
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
          if (!(part instanceof ToolResultPart)) throw new TypeError(wrongPart(part, message.role));
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
   * or message item whose paired reasoning item is missing.
   *
   * @returns Each part's record fields, `null` for a part without a payload; all `null` when the
   *   message renders neutrally.
   * @throws {TypeError} If the message holds a media part or a tool result.
   */
  #records(message: Message): (JsonObject | null)[] {
    const records: (JsonObject | null)[] = [];
    let reason: ReplayDegradedReason | null = null;
    for (const part of message.parts) {
      if (part instanceof MediaPart) throw new TypeError(NO_MEDIA);
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(part, message.role));
      if (part.providerPayload === null) {
        records.push(null);
        continue;
      }
      const read = readRecord(part, this.#key);
      if (read.ok) records.push(read.fields);
      else reason ??= read.reason;
    }
    const neutral = message.parts.map(() => null);
    if (reason !== null) {
      this.#onReplayDegraded?.(reason);
      return neutral;
    }
    // Without stored state only encrypted reasoning can be replayed; that is no record mismatch,
    // so nothing is signalled.
    const unreplayable = (part: Part, index: number) =>
      part instanceof ReasoningPart && records[index] !== null && !encrypted(records[index]!["item"] as JsonObject);
    if (this.subscription && message.parts.some(unreplayable)) return neutral;
    return records;
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
 * the part's call id; a text or refusal record's `id` must be a string and its `annotations` an
 * array, when present.
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
      (!Object.hasOwn(fields, "annotations") || Array.isArray(fields["annotations"]));
  }
  return fits ? read : { ok: false, reason: "malformed" };
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
    if (!(part instanceof TextPart)) throw new TypeError(wrongPart(part, message.role));
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

/** Return the error text for a part a message of `role` cannot carry. */
function wrongPart(part: Part, role: string): string {
  const article = role === "assistant" ? "an" : "a";
  return `the openai_responses route cannot send a ${part.constructor.name} in ${article} ${role} message`;
}
