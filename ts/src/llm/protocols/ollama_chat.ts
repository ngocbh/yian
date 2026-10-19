/**
 * Ollama's native `/api/chat` wire format, every call streamed as newline-delimited JSON. yian
 * uses the native endpoint, not Ollama's OpenAI-compatible `/v1`, because only the native one
 * carries `think`, `options` and `keep_alive`.
 *
 * The body is `model`, `stream: true`, `options: {}`, `messages` and `tools`, in that order, then
 * the provider block merged over them; `think`, `keep_alive`, `logprobs`, `top_logprobs` and
 * `options.*` come only from the block. Every message is sent in place, neither merged nor lifted,
 * with empty fields left out: a system or user message as its text parts joined with `"\n"`, an
 * assistant message as its joined text, its thinking and its calls (with their ids) in part order,
 * its refusals left out, and each tool result as its own `tool` message carrying the tool's name
 * and the call id. Thinking goes back only from reasoning this route parsed for the same provider,
 * model and account; one record that doesn't match leaves all of that message's thinking out.
 * Tools go with their parameters as written; the tool choice, parallel tool calls and `strict` are
 * not sent, Ollama having none of them.
 *
 * Each line of the reply is one JSON object holding a piece of the message; the line with
 * `done: true` ends it with the finish reason and the token counts. A reply that failed or ended
 * early raises `ProviderError`; an error message holds the provider's text uncapped, which
 * `RouteModel` scrubs of the route's credentials and then caps.
 */

import { mergeRequestBody } from "../body.ts";
import type { JsonObject, JsonValue } from "../../typesys/json.ts";
import type { Event } from "../events.ts";
import type { NdjsonProtocol } from "../executor.ts";
import {
  copyJson,
  MediaPart,
  type Message,
  type Part,
  readReplayRecord,
  ReasoningPart,
  type ReplayDegradedReason,
  type ReplayKey,
  type Request,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  type ToolSpec,
} from "../messages.ts";

/** The protocol id, in `Finish.protocol` and replay records. */
export const PROTOCOL = "ollama_chat";
const NO_MEDIA = "the ollama_chat route does not send media parts yet";

/** What `OllamaChat` is built from. */
export interface OllamaChatOptions {
  /** The model the body names. */
  model: string;
  /** The route's provider name, for error messages and replay records. */
  provider: string;
  /** A non-secret account id, for replay records. */
  account?: string | null;
  /** The provider block, merged into every body after yian's fields; copied. */
  block?: JsonObject | null;
  /** Called once per assistant message holding a replay record that can't be used. */
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
}

/** The Ollama `/api/chat` wire protocol for one model. */
export class OllamaChat implements NdjsonProtocol {
  readonly name = PROTOCOL;
  readonly framing = "ndjson";
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

  constructor(options: OllamaChatOptions) {
    this.model = options.model;
    this.provider = options.provider;
    this.account = options.account ?? null;
    this.#block = options.block ? copyJson(options.block) : null;
    this.#key = { protocol: PROTOCOL, provider: this.provider, model: this.model, account: this.account };
    this.#onReplayDegraded = options.onReplayDegraded ?? null;
  }

  /**
   * Build the `/api/chat` request body: `model`, `stream: true`, `options: {}`, `messages` and
   * `tools` (`[]` with none), then the block merged over them.
   *
   * Every message is sent in place. A system or user message is `{role, content}`, its text parts
   * joined with `"\n"`. An assistant message is `{role, content, thinking, tool_calls}`: its text
   * parts joined with `"\n"`, its reasoning texts joined with `""`, and each call in part order as
   * `{id, function: {name, arguments}}`, the arguments copied (`{}` for an invalid call); refusal
   * parts are not sent. Reasoning goes back only when every reasoning record on the message reads
   * back for this protocol, provider, model and account; otherwise none of it is sent and
   * `onReplayDegraded` is called once for the message with the first reason in part order. A
   * reasoning part without a payload is never sent and signals nothing. A tool message is one
   * `{role: "tool", content, tool_name, tool_call_id}` per result; `isError` is not sent. An empty
   * `content`, `thinking` and call list are left out, so an empty assistant message is
   * `{role: "assistant"}`.
   *
   * Each tool is `{type: "function", function: {name, description, parameters}}`, the parameters
   * copied as written; `strict`, the tool choice and `parallelToolCalls` are not sent. The block
   * is merged last: its lists and scalars replace yian's, its maps merge into them, and a `null`
   * leaf is not sent.
   *
   * @throws {TypeError} If a message holds a media part (checked first) or a part its role
   *   cannot carry.
   */
  buildBody(request: Request): JsonObject {
    if (request.messages.some((message) => message.parts.some((part) => part instanceof MediaPart))) {
      throw new TypeError(NO_MEDIA);
    }
    const messages: JsonObject[] = [];
    for (const message of request.messages) {
      if (message.role === "tool") messages.push(...message.parts.map(toolMessage));
      else if (message.role === "assistant") messages.push(this.#assistant(message));
      else messages.push(withContent({ role: message.role }, joinedText(message)));
    }
    const body: JsonObject = {
      model: this.model,
      stream: true,
      options: {},
      messages,
      tools: request.tools.map(functionTool),
    };
    if (this.#block) mergeRequestBody(body, this.#block);
    return body;
  }

  /**
   * Render an assistant message: its text parts joined with `"\n"`, its thinking, and its calls in
   * part order; refusals are not sent.
   *
   * @throws {TypeError} If the message holds a tool result.
   */
  #assistant(message: Message): JsonObject {
    const texts: string[] = [];
    const calls: JsonObject[] = [];
    for (const part of message.parts) {
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(part, "assistant"));
      if (part instanceof TextPart) {
        texts.push(part.text);
      } else if (part instanceof ToolCallPart) {
        // Copied so changing a body cannot change the message it came from.
        const args = part.arguments === null ? {} : copyJson(part.arguments);
        calls.push({ id: part.id, function: { name: part.name, arguments: args } });
      }
    }
    // Read after the loop above, so a message it refuses never fires onReplayDegraded first.
    const thinking = this.#thinking(message);
    const rendered = withContent({ role: "assistant" }, texts.join("\n"));
    if (thinking !== "") rendered["thinking"] = thinking;
    if (calls.length > 0) rendered["tool_calls"] = calls;
    return rendered;
  }

  /**
   * Return an assistant message's reasoning texts joined with `""`, or `""` when any reasoning
   * record can't be used, calling `onReplayDegraded` once with the first reason.
   *
   * A reply's thinking comes from one model, so a mismatch on one part means all of them. A
   * reasoning part without a payload is skipped and signals nothing.
   */
  #thinking(message: Message): string {
    const texts: string[] = [];
    let reason: ReplayDegradedReason | null = null;
    for (const part of message.parts) {
      if (!(part instanceof ReasoningPart) || part.providerPayload === null) continue;
      const read = readReplayRecord(part.providerPayload, this.#key, []);
      if (!read.ok) reason ??= read.reason;
      else texts.push(part.text);
    }
    if (reason === null) return texts.join("");
    this.#onReplayDegraded?.(reason);
    return "";
  }

  /** Turn the reply's NDJSON lines into stream events; not built yet. */
  async *parse(_frames: AsyncIterable<JsonValue>): AsyncGenerator<Event> {
    throw new Error("the ollama_chat route cannot read replies yet");
  }
}

/**
 * Return a system or user message's text parts joined with `"\n"`.
 *
 * @throws {TypeError} If the message holds a part that is not text.
 */
function joinedText(message: Message): string {
  return message.parts
    .map((part) => {
      if (!(part instanceof TextPart)) throw new TypeError(wrongPart(part, message.role));
      return part.text;
    })
    .join("\n");
}

/**
 * Render a tool message's part as its own `tool` message with the tool's name and the call id.
 *
 * @throws {TypeError} If the part is not a tool result.
 */
function toolMessage(part: Part): JsonObject {
  if (!(part instanceof ToolResultPart)) throw new TypeError(wrongPart(part, "tool"));
  const rendered = withContent({ role: "tool" }, part.content);
  rendered["tool_name"] = part.name;
  rendered["tool_call_id"] = part.callId;
  return rendered;
}

/** Set `content` on `rendered` unless it is empty, and return `rendered`. */
function withContent(rendered: JsonObject, content: string): JsonObject {
  if (content !== "") rendered["content"] = content;
  return rendered;
}

/** Render a tool as a function entry; Ollama has no `strict`. */
function functionTool(spec: ToolSpec): JsonObject {
  // Copied so changing a body cannot change the spec it came from.
  return {
    type: "function",
    function: { name: spec.name, description: spec.description, parameters: copyJson(spec.parameters) },
  };
}

/** Return the error text for a part a message of `role` cannot carry. */
function wrongPart(part: Part, role: string): string {
  const article = role === "assistant" ? "an" : "a";
  return `the ollama_chat route cannot send a ${part.constructor.name} in ${article} ${role} message`;
}
