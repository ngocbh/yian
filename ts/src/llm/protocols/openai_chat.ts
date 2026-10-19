/**
 * The OpenAI Chat Completions wire format (`POST /chat/completions`), spoken by Azure OpenAI and
 * the OpenAI-compatible vendors (xAI, DeepSeek, Qwen, GLM, MiniMax, OpenRouter, vLLM).
 *
 * Each message becomes a chat message: system and user text, an assistant message's text and
 * tool calls, and one `tool` message per tool result. Every call streams. The vendors differ in
 * a few request rules, which `ChatOptions` holds per route: DeepSeek gets earlier reasoning back
 * as `reasoning_content`, MiniMax's reasoning models get `reasoning_split`, some models refuse a
 * named tool choice, and most Azure deployments refuse `parallel_tool_calls`.
 *
 * Reasoning parts carry replay records naming the wire field they came from, so only reasoning
 * this route parsed from `reasoning_content` is echoed. An error message holds the provider's
 * text uncapped; `RouteModel` scrubs it of the route's credentials and then caps it.
 */

import type { JsonObject, JsonValue } from "../../typesys/json.ts";
import { encodeBody, mergeRequestBody } from "../body.ts";
import { getCapabilities } from "../capabilities.ts";
import type { Event } from "../events.ts";
import type { SseProtocol } from "../executor.ts";
import type { SSEEvent } from "../framing.ts";
import {
  copyJson,
  MediaPart,
  type Message,
  NamedTool,
  type Part,
  ReasoningPart,
  readReplayRecord,
  type ReplayDegradedReason,
  type ReplayKey,
  type Request,
  TextPart,
  ToolCallPart,
  type ToolSpec,
  ToolResultPart,
} from "../messages.ts";

/** The protocol id, in `Finish.protocol` and replay records. */
export const PROTOCOL = "openai_chat";
const NO_MEDIA = "the openai_chat route does not send media parts yet";
// The wire field a reasoning record names when its text may be echoed.
const ECHOED_FIELD = "reasoning_content";

/** The request rules that differ between Chat Completions routes. */
export interface ChatOptions {
  /** Send earlier assistant reasoning back as `reasoning_content` (DeepSeek). */
  readonly echoReasoningContent: boolean;
  /** Send `reasoning_split: true`, so reasoning arrives apart from the text (MiniMax M2). */
  readonly reasoningSplit: boolean;
  /** Send a `NamedTool` choice; when false it is left out. */
  readonly namedToolChoice: boolean;
  /** Send `parallel_tool_calls` when the request sets it. */
  readonly sendParallelToolCalls: boolean;
  /** Send `stream_options: {include_usage: true}`, so the stream reports usage. */
  readonly streamUsage: boolean;
}

/**
 * Return the request rules for a route, frozen.
 *
 * Only DeepSeek echoes reasoning, for any model. MiniMax's reasoning models get
 * `reasoning_split`. A named tool choice is sent unless the model's capabilities, read by model
 * name only, refuse it; Azure always sends it. Azure sends `parallel_tool_calls` only for a
 * deployment named `gpt-4o`.
 *
 * @param provider The route's provider name.
 * @param model The model name (the deployment name on Azure).
 */
export function chatOptions(provider: string, model: string): ChatOptions {
  const capabilities = getCapabilities(model);
  const azure = provider === "azure";
  return Object.freeze({
    echoReasoningContent: provider === "deepseek",
    reasoningSplit: (provider === "minimax" || provider === "minimax-cn") && capabilities.requiresReasoningSplit,
    namedToolChoice: azure || capabilities.supportsToolChoice,
    // LangChain's Azure client disables `parallel_tool_calls` for every deployment but `gpt-4o`;
    // kept so today's requests stay the same.
    sendParallelToolCalls: !azure || model === "gpt-4o",
    streamUsage: true,
  });
}

/** What `OpenAIChat` is built from. */
export interface OpenAIChatOptions {
  /** The model name sent as `model` (the deployment name on Azure). */
  model: string;
  /** The route's provider name, for messages and replay records. */
  provider: string;
  /** The route's request rules; defaults to `chatOptions(provider, model)`. */
  options?: ChatOptions;
  /** The provider block, merged into every body after yian's fields; copied. */
  block?: JsonObject | null;
  /** Called once per assistant message holding a reasoning record that can't be used. */
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
}

/** The Chat Completions wire protocol for one model. */
export class OpenAIChat implements SseProtocol {
  readonly name = PROTOCOL;
  readonly framing = "sse";
  /** The model name. */
  readonly model: string;
  /** The route's provider name. */
  readonly provider: string;
  /** The route's request rules. */
  readonly options: ChatOptions;
  // Copied so later changes to the caller's block cannot change future bodies.
  readonly #block: JsonObject | null;
  readonly #key: ReplayKey;
  readonly #onReplayDegraded: ((reason: ReplayDegradedReason) => void) | null;

  constructor(options: OpenAIChatOptions) {
    this.model = options.model;
    this.provider = options.provider;
    this.options = options.options ?? chatOptions(this.provider, this.model);
    this.#block = options.block ? copyJson(options.block) : null;
    this.#key = { protocol: PROTOCOL, provider: this.provider, model: this.model };
    this.#onReplayDegraded = options.onReplayDegraded ?? null;
  }

  /**
   * Build the Chat Completions request body: `model`, `messages`, `stream: true` and (when the
   * route reports usage) `stream_options`, then the tools, the tool choice and
   * `parallel_tool_calls` as the route allows, then `reasoning_split` when the route needs it,
   * then the provider block.
   *
   * On a route that echoes reasoning, an assistant message holding a reasoning record that
   * can't be used is sent without that reasoning, and `onReplayDegraded` is called once for it;
   * a throw from the callback propagates.
   *
   * @throws {TypeError} If a message holds a media part or a part its role cannot carry.
   */
  buildBody(request: Request): JsonObject {
    const body: JsonObject = { model: this.model, messages: this.#messages(request.messages), stream: true };
    if (this.options.streamUsage) body["stream_options"] = { include_usage: true };
    if (request.tools.length > 0) body["tools"] = request.tools.map(tool);
    const choice = request.toolChoice;
    if (choice instanceof NamedTool) {
      if (this.options.namedToolChoice) body["tool_choice"] = { type: "function", function: { name: choice.name } };
    } else if (choice !== null) {
      body["tool_choice"] = choice;
    }
    if (request.parallelToolCalls !== null && this.options.sendParallelToolCalls) {
      body["parallel_tool_calls"] = request.parallelToolCalls;
    }
    // Set before the merge, so a block's own value wins.
    if (this.options.reasoningSplit) body["reasoning_split"] = true;
    if (this.#block) mergeRequestBody(body, this.#block);
    return body;
  }

  /** Turn a streamed reply's events into stream events; not implemented yet. */
  async *parse(_frames: AsyncIterable<SSEEvent>): AsyncGenerator<Event> {
    throw new Error("not implemented");
  }

  /** Render the conversation as chat messages, in order. */
  #messages(messages: readonly Message[]): JsonObject[] {
    const rendered: JsonObject[] = [];
    for (const message of messages) {
      if (message.role === "system" || message.role === "user") {
        const item = inputMessage(message);
        if (item !== null) rendered.push(item);
      } else if (message.role === "assistant") {
        rendered.push(this.#assistant(message));
      } else {
        for (const part of message.parts) {
          if (part instanceof MediaPart) throw new TypeError(NO_MEDIA);
          if (!(part instanceof ToolResultPart)) throw new TypeError(wrongPart(part, message.role));
          // Chat Completions has no error flag; the error text is the content.
          rendered.push({ role: "tool", tool_call_id: part.callId, content: part.content });
        }
      }
    }
    return rendered;
  }

  /**
   * Render an assistant message: its text parts joined with newlines, its calls valid first
   * (LangChain's order), and on an echo route the reasoning parsed from `reasoning_content`.
   *
   * Refusals are not sent. `content` is `null` when the message has calls and no text.
   */
  #assistant(message: Message): JsonObject {
    const texts: string[] = [];
    const validCalls: JsonObject[] = [];
    const invalidCalls: JsonObject[] = [];
    const reasoning: string[] = [];
    let reason: ReplayDegradedReason | null = null;
    for (const part of message.parts) {
      if (part instanceof MediaPart) throw new TypeError(NO_MEDIA);
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(part, message.role));
      if (part instanceof TextPart) {
        texts.push(part.text);
      } else if (part instanceof ToolCallPart) {
        if (part.arguments !== null) {
          validCalls.push(toolCall(part, encodeBody(part.arguments, "tool call arguments")));
        } else {
          // The constructor guarantees raw text when the arguments did not parse.
          invalidCalls.push(toolCall(part, part.rawArguments!));
        }
      } else if (part instanceof ReasoningPart && this.options.echoReasoningContent && part.providerPayload !== null) {
        // Records are read part by part, not for the whole message: the Chat API pairs nothing
        // by id, so only the reasoning text depends on its record.
        const read = readField(part.providerPayload, this.#key);
        if (!read.ok) reason ??= read.reason;
        else if (read.field === ECHOED_FIELD) reasoning.push(part.text);
      }
    }
    if (reason !== null) this.#onReplayDegraded?.(reason);
    const calls = [...validCalls, ...invalidCalls];
    const content = texts.join("\n");
    const rendered: JsonObject = { role: "assistant", content: content === "" && calls.length > 0 ? null : content };
    if (calls.length > 0) rendered["tool_calls"] = calls;
    if (reasoning.length > 0) rendered["reasoning_content"] = reasoning.join("");
    return rendered;
  }
}

/**
 * Read a reasoning part's replay record and return the wire field it names.
 *
 * A record without a string `field` is `malformed`.
 */
function readField(
  payload: JsonObject,
  key: ReplayKey,
): { ok: true; field: string } | { ok: false; reason: ReplayDegradedReason } {
  const read = readReplayRecord(payload, key, ["field"]);
  if (!read.ok) return read;
  const field = read.fields["field"];
  return typeof field === "string" ? { ok: true, field } : { ok: false, reason: "malformed" };
}

/** Render a tool call with its arguments text. */
function toolCall(part: ToolCallPart, args: string): JsonObject {
  return { type: "function", id: part.id, function: { name: part.name, arguments: args } };
}

/**
 * Render a system or user message; `null` when it has no parts.
 *
 * One text part is sent as a string, several as `text` parts.
 */
function inputMessage(message: Message): JsonObject | null {
  const texts: string[] = [];
  for (const part of message.parts) {
    if (part instanceof MediaPart) throw new TypeError(NO_MEDIA);
    if (!(part instanceof TextPart)) throw new TypeError(wrongPart(part, message.role));
    texts.push(part.text);
  }
  if (texts.length === 0) return null;
  const content: JsonValue = texts.length === 1 ? texts[0]! : texts.map((text) => ({ type: "text", text }));
  return { role: message.role, content };
}

/** Render a tool as a Chat Completions function tool; `strict` only when set. */
function tool(spec: ToolSpec): JsonObject {
  const fn: JsonObject = {
    name: spec.name,
    description: spec.description,
    // Copied so changing a body cannot change the spec it came from.
    parameters: copyJson(spec.parameters),
  };
  if (spec.strict !== null) fn["strict"] = spec.strict;
  return { type: "function", function: fn };
}

/** Return the error text for a part a message of `role` cannot carry. */
function wrongPart(part: Part, role: string): string {
  const article = role === "assistant" ? "an" : "a";
  return `the openai_chat route cannot send a ${part.constructor.name} in ${article} ${role} message`;
}
