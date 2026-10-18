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
import { mergeRequestBody } from "../body.ts";
import type { Event } from "../events.ts";
import type { SseProtocol } from "../executor.ts";
import type { SSEEvent } from "../framing.ts";
import {
  copyJson,
  MediaPart,
  type Message,
  NamedTool,
  type Part,
  RefusalPart,
  type ReplayDegradedReason,
  type Request,
  TextPart,
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
  /** Whether the subscription's rules apply: `store: false`, system text in `instructions`, encrypted reasoning, no message ids. */
  readonly subscription: boolean;
  // Copied so later changes to the caller's block cannot change future bodies.
  readonly #block: JsonObject | null;

  constructor(options: OpenAIResponsesOptions) {
    this.model = options.model;
    this.provider = options.provider;
    this.account = options.account ?? null;
    this.#block = options.block ? copyJson(options.block) : null;
    this.subscription = options.subscription ?? false;
  }

  /**
   * Build the Responses request body: `model`, `input` and `stream: true`, then the tools, the
   * tool choice and `parallel_tool_calls` when set, then the provider block, then (on the
   * subscription) its settings.
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
    if (this.#block && Object.keys(this.#block).length > 0) mergeRequestBody(body, this.#block);
    if (this.subscription) {
      // Applied after the block so it cannot switch them off: the backend keeps no state, so
      // every turn must carry the full, replayable conversation.
      body["store"] = false;
      body["stream"] = true;
      const include = body["include"];
      const asked = Array.isArray(include) ? include.filter((each) => typeof each === "string") : [];
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
    for (const message of messages) {
      if (message.role === "system" && this.subscription) {
        const text = instructionText(message);
        if (text) instructions.push(text);
      } else if (message.role === "system" || message.role === "user") {
        const item = inputMessage(message);
        if (item !== null) items.push(item);
      } else if (message.role === "assistant") {
        this.#assistant(message, items);
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

  /** Append an assistant message's items to `items`, in part order. */
  #assistant(message: Message, items: JsonObject[]): void {
    for (const part of message.parts) {
      if (part instanceof TextPart || part instanceof RefusalPart) {
        this.#messageContent(part, items);
      } else if (part instanceof MediaPart) {
        throw new TypeError(NO_MEDIA);
      } else {
        throw new TypeError(wrongPart(part, message.role));
      }
    }
  }

  /** Add a text or refusal part to `items` as a new assistant message item. */
  #messageContent(part: TextPart | RefusalPart, items: JsonObject[]): void {
    const content: JsonObject =
      part instanceof TextPart
        ? { type: "output_text", text: part.text, annotations: [] }
        : { type: "refusal", refusal: part.text };
    items.push({ type: "message", role: "assistant", content: [content] });
  }
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
