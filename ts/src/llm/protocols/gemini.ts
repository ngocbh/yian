/**
 * The Gemini `generateContent` wire format (Developer API), every call streamed through
 * `models/<model>:streamGenerateContent?alt=sse`; the model is in the URL, not the body.
 *
 * Messages are formatted as LangChain's Google client formats them, so requests stay the same,
 * with its gaps closed: every system message's text joins `systemInstruction`, each user message
 * is its own `user` content (an empty one is not sent, with a warning), each run of tool messages
 * becomes one `user` content of `functionResponse` parts carrying the call ids, and an assistant
 * message is a `model` content of its text and `functionCall` parts in part order, its reasoning
 * and refusals left out. `safetySettings: []` and `generationConfig.candidateCount: 1` are always
 * sent.
 *
 * Replies are not read yet.
 */

import { JSONDecodeError, type JsonObject, type JsonValue, parseJson } from "../../typesys/json.ts";
import type { Event } from "../events.ts";
import type { SseProtocol } from "../executor.ts";
import type { SSEEvent } from "../framing.ts";
import {
  copyJson,
  isPlainObject,
  MediaPart,
  type Message,
  type Part,
  type ReplayDegradedReason,
  type ReplayKey,
  type Request,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from "../messages.ts";

/** The protocol id, in `Finish.protocol` and replay records. */
export const PROTOCOL = "gemini";
const NO_MEDIA = "the gemini route does not send media parts yet";
const EMPTY_USER = "an empty user message was not sent; Gemini rejects empty content";

/** What `Gemini` is built from. */
export interface GeminiOptions {
  /** The model the route's path names; for replay records. */
  model: string;
  /** The route's provider name, for error messages and replay records. */
  provider: string;
  /** A non-secret account id, for replay records. */
  account?: string | null;
  /** The provider block, merged into every body after yian's fields; copied. */
  block?: JsonObject | null;
  /** Called once per assistant message holding a signature record that can't be used. */
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
}

/** The Gemini `generateContent` wire protocol for one model. */
export class Gemini implements SseProtocol {
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
  // Python's `warnings.warn` shows a warning once per call site, so warn once here.
  #emptyUserWarned = false;

  constructor(options: GeminiOptions) {
    this.model = options.model;
    this.provider = options.provider;
    this.account = options.account ?? null;
    this.#block = options.block ? copyJson(options.block) : null;
    this.#key = { protocol: PROTOCOL, provider: this.provider, model: this.model, account: this.account };
    this.#onReplayDegraded = options.onReplayDegraded ?? null;
  }

  /**
   * Build the `generateContent` request body: `contents`, `systemInstruction` when a system
   * message has text, `safetySettings: []` and `generationConfig: {candidateCount: 1}`.
   *
   * Every system message's non-empty text parts, wherever the message is, join
   * `systemInstruction.parts` in order. A user message is a `user` content of its non-empty text
   * parts; one with none is not sent, with a warning through `process.emitWarning` (once per
   * instance). Each run of consecutive tool messages is one `user` content of a
   * `functionResponse` per result, with the result's call id and name; its `response` is the
   * result's text read as JSON when that is a plain object, else `{output: <the value>}`, or
   * `{output: <the text>}` when the text is not plain JSON. `isError` is not sent. An assistant
   * message is a `model` content in part order: a non-empty text part as `{text}`, a call as
   * `functionCall` with its id, name and arguments (`{}` for an invalid call); reasoning and
   * refusal parts are not sent, and a message rendering no part is skipped. A trailing assistant
   * message goes as written.
   *
   * @throws {TypeError} If a message holds a media part (checked first) or a part its role
   *   cannot carry.
   */
  buildBody(request: Request): JsonObject {
    const [system, contents] = this.#contents(request.messages);
    const body: JsonObject = { contents };
    if (system.length > 0) body["systemInstruction"] = { parts: system };
    body["safetySettings"] = [];
    body["generationConfig"] = { candidateCount: 1n };
    return body;
  }

  /** Turn a streamed reply's events into stream events. Not implemented yet. */
  async *parse(_frames: AsyncIterable<SSEEvent>): AsyncGenerator<Event> {
    throw new Error("not implemented");
  }

  /** Return the `systemInstruction` parts and the `contents`. */
  #contents(messages: readonly Message[]): [JsonObject[], JsonObject[]] {
    if (messages.some((message) => message.parts.some((part) => part instanceof MediaPart))) {
      throw new TypeError(NO_MEDIA);
    }
    const system: JsonObject[] = [];
    const contents: JsonObject[] = [];
    // The parts of the content the current run of tool messages goes into, or `null` outside a run.
    let results: JsonValue[] | null = null;
    for (const message of messages) {
      if (message.role === "tool") {
        const parts = message.parts.map(functionResponse);
        if (results !== null) results.push(...parts);
        else if (parts.length > 0) contents.push({ role: "user", parts: (results = parts) });
        continue;
      }
      results = null;
      if (message.role === "system") {
        system.push(...textParts(message));
      } else if (message.role === "user") {
        const parts = textParts(message);
        if (parts.length > 0) contents.push({ role: "user", parts });
        else this.#warnEmptyUser();
      } else {
        const parts = this.#modelParts(message);
        if (parts.length > 0) contents.push({ role: "model", parts });
      }
    }
    return [system, contents];
  }

  /**
   * Return an assistant message's parts in part order: a non-empty text as `{text}`, a call as
   * `functionCall`; reasoning and refusals are not sent.
   *
   * @throws {TypeError} If the message holds a tool result.
   */
  #modelParts(message: Message): JsonObject[] {
    const parts: JsonObject[] = [];
    for (const part of message.parts) {
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(part, message.role));
      if (part instanceof TextPart) {
        if (part.text !== "") parts.push({ text: part.text });
      } else if (part instanceof ToolCallPart) {
        // Copied so changing a body cannot change the message it came from.
        const args = part.arguments === null ? {} : copyJson(part.arguments);
        parts.push({ functionCall: { id: part.id, name: part.name, args } });
      }
    }
    return parts;
  }

  /** Warn, the first time, that an empty user message was not sent. */
  #warnEmptyUser(): void {
    if (this.#emptyUserWarned) return;
    this.#emptyUserWarned = true;
    process.emitWarning(EMPTY_USER);
  }
}

/**
 * Return a system or user message's non-empty texts as `{text}` parts; Gemini refuses an empty
 * text part ("Unable to submit request because it has an empty text parameter").
 *
 * @throws {TypeError} If the message holds a part that is not text.
 */
function textParts(message: Message): JsonObject[] {
  const parts: JsonObject[] = [];
  for (const part of message.parts) {
    if (!(part instanceof TextPart)) throw new TypeError(wrongPart(part, message.role));
    if (part.text !== "") parts.push({ text: part.text });
  }
  return parts;
}

/**
 * Render a tool message's part as a `functionResponse` part with the result's call id, name and
 * response.
 *
 * @throws {TypeError} If the part is not a tool result.
 */
function functionResponse(part: Part): JsonObject {
  if (!(part instanceof ToolResultPart)) throw new TypeError(wrongPart(part, "tool"));
  return { functionResponse: { id: part.callId, name: part.name, response: toolResponse(part.content) } };
}

/**
 * Return a tool result's `response`: its text read as plain JSON when that is an object, else
 * `{output: <the value>}`, or `{output: <the text>}` when the text is not plain JSON (Gemini
 * takes only an object here).
 */
function toolResponse(content: string): JsonObject {
  let value: JsonValue;
  try {
    value = parseJson(content, "plain");
  } catch (error) {
    if (error instanceof JSONDecodeError) return { output: content };
    throw error;
  }
  return isPlainObject(value) ? value : { output: value };
}

/** Return the error text for a part a message of `role` cannot carry. */
function wrongPart(part: Part, role: string): string {
  const article = role === "assistant" ? "an" : "a";
  return `the gemini route cannot send a ${part.constructor.name} in ${article} ${role} message`;
}
