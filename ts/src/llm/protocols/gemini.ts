/**
 * The Gemini `generateContent` wire format (Developer API), every call streamed through
 * `models/<model>:streamGenerateContent?alt=sse`; the model is in the URL, not the body.
 *
 * Messages are formatted as LangChain's Google client formats them, so requests stay the same,
 * with its gaps closed: every system message's text joins `systemInstruction`, each user message
 * is its own `user` content (an empty one is not sent, with a warning), each run of tool messages
 * becomes one `user` content of `functionResponse` parts carrying the call ids, and an assistant
 * message is a `model` content of its text and `functionCall` parts in part order, its reasoning
 * and refusals left out. A text or call part's thought signature goes back from its replay record
 * only to the provider, model and account that wrote it, and a function call Gemini didn't write
 * for this model goes out with Google's placeholder signature. Every tool goes in one
 * `functionDeclarations` entry with its parameters as written, and a tool choice becomes
 * `toolConfig`, both only with tools. `safetySettings: []` and `generationConfig.candidateCount: 1`
 * are always sent, and the provider block is merged last.
 *
 * Replies are not read yet.
 */

import { mergeRequestBody } from "../body.ts";
import { JSONDecodeError, type JsonObject, type JsonValue, parseJson } from "../../typesys/json.ts";
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
  readReplayRecord,
  type ReplayDegradedReason,
  type ReplayKey,
  type Request,
  TextPart,
  ToolCallPart,
  type ToolChoice,
  ToolResultPart,
  type ToolSpec,
} from "../messages.ts";

/** The protocol id, in `Finish.protocol` and replay records. */
export const PROTOCOL = "gemini";
const NO_MEDIA = "the gemini route does not send media parts yet";
const EMPTY_USER = "an empty user message was not sent; Gemini rejects empty content";
const PART_FIELDS = ["thought_signature"];
const MESSAGE_FIELDS = ["response_id", "response_model"];
// Google's documented signature for a function call that Gemini didn't write.
const PLACEHOLDER = "skip_thought_signature_validator";

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
   * message has text, `tools` and `toolConfig`, `safetySettings: []` and
   * `generationConfig: {candidateCount: 1}`, then the block merged over them.
   *
   * Every system message's non-empty text parts, wherever the message is, join
   * `systemInstruction.parts` in order. A user message is a `user` content of its non-empty text
   * parts; one with none is not sent, with a warning through `process.emitWarning` (once per
   * instance). Each run of tool messages is one `user` content of a `functionResponse` per
   * result, with the result's call id and name; only a user or model content that is sent ends
   * the run, so a system message or a skipped message in between does not. Its `response` is the
   * result's text read as JSON when that is a plain object, else `{output: <the value>}`, or
   * `{output: <the text>}` when the text is not plain JSON. `isError` is not sent. An assistant
   * message is a `model` content in part order: a non-empty text part as `{text}`, a call as
   * `functionCall` with its id, name and arguments (`{}` for an invalid call); reasoning and
   * refusal parts are not sent, and a message rendering no part is skipped. A trailing assistant
   * message goes as written.
   *
   * A text or call part's replay record puts its `thoughtSignature` on its part (an empty text
   * part is sent only with one). When any of a message's records can't be used, none of its
   * signatures is sent and `onReplayDegraded` is called once with the first reason in part order;
   * an exception it throws propagates. A message's first call that goes out unsigned gets
   * `thoughtSignature: "skip_thought_signature_validator"` when the message is not this model's
   * own reply (its message record doesn't read back for this protocol, provider, model and
   * account) or its records couldn't be used; this model's own reply otherwise goes as received.
   *
   * Every tool goes in one `functionDeclarations` entry: its name, description and parameters
   * (copied, as written) as `parametersJsonSchema`; `strict` is not sent. The tool choice is sent
   * only with tools, as `toolConfig.functionCallingConfig`: `required` is `ANY`, `none` is `NONE`,
   * a named tool is `ANY` limited to that name, and `auto` or no choice sends nothing.
   * `parallelToolCalls` is not sent. The block is merged last: its lists and scalars replace
   * yian's, its maps merge into them, and a `null` leaf is not sent.
   *
   * @throws {TypeError} If a message holds a media part (checked first) or a part its role
   *   cannot carry.
   */
  buildBody(request: Request): JsonObject {
    const [system, contents] = this.#contents(request.messages);
    const body: JsonObject = { contents };
    if (system.length > 0) body["systemInstruction"] = { parts: system };
    if (request.tools.length > 0) {
      body["tools"] = [{ functionDeclarations: request.tools.map(functionDeclaration) }];
      const config = toolConfig(request.toolChoice);
      if (config !== null) body["toolConfig"] = config;
    }
    body["safetySettings"] = [];
    body["generationConfig"] = { candidateCount: 1n };
    if (this.#block) mergeRequestBody(body, this.#block);
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
    // The parts of the content the current run of tool messages goes into, or `null` outside a
    // run. Only a pushed user or model content ends a run: Gemini wants every response to one
    // model turn's calls in the content right after it, so a message that adds no content
    // (a system message, a skipped user or assistant message) must not split them.
    let results: JsonValue[] | null = null;
    for (const message of messages) {
      if (message.role === "tool") {
        const parts = message.parts.map(functionResponse);
        if (results !== null) {
          results.push(...parts);
        } else if (parts.length > 0) {
          results = parts;
          contents.push({ role: "user", parts: results });
        }
      } else if (message.role === "system") {
        system.push(...textParts(message));
      } else {
        const user = message.role === "user";
        const parts = user ? textParts(message) : this.#modelParts(message);
        if (parts.length > 0) {
          contents.push({ role: user ? "user" : "model", parts });
          results = null;
        } else if (user) {
          this.#warnEmptyUser();
        }
      }
    }
    return [system, contents];
  }

  /**
   * Return an assistant message's parts in part order: a text as `{text}` (an empty one only when
   * it carries a signature), a call as `functionCall`, each with its usable signature as
   * `thoughtSignature`; reasoning and refusals are not sent.
   *
   * The first call gets the placeholder signature when it goes out unsigned and the message is
   * not this model's own reply, or its signature records were left out.
   *
   * @throws {TypeError} If the message holds a tool result.
   */
  #modelParts(message: Message): JsonObject[] {
    const [signatures, degraded] = this.#signatures(message);
    const parts: JsonObject[] = [];
    let firstCall: JsonObject | null = null;
    for (const [index, part] of message.parts.entries()) {
      const signature = signatures[index] ?? null;
      let rendered: JsonObject;
      if (part instanceof TextPart) {
        if (part.text === "" && signature === null) continue;
        rendered = { text: part.text };
      } else if (part instanceof ToolCallPart) {
        // Copied so changing a body cannot change the message it came from.
        const args = part.arguments === null ? {} : copyJson(part.arguments);
        rendered = { functionCall: { id: part.id, name: part.name, args } };
        firstCall ??= rendered;
      } else {
        continue;
      }
      if (signature !== null) rendered["thoughtSignature"] = signature;
      parts.push(rendered);
    }
    // Gemini 3 refuses a step whose first call is unsigned. This model's own reply goes back as
    // received (Gemini 2.5 leaves calls unsigned); a call yian can't send a signature for, from
    // another model or built by hand, gets the literal Google documents for such calls.
    const own = readReplayRecord(message.providerPayload, this.#key, MESSAGE_FIELDS).ok;
    if (firstCall !== null && (!own || degraded) && !Object.hasOwn(firstCall, "thoughtSignature")) {
      firstCall["thoughtSignature"] = PLACEHOLDER;
    }
    return parts;
  }

  /**
   * Read an assistant message's text and call signature records, calling `onReplayDegraded` once
   * with the first reason when one can't be used.
   *
   * Every signature is left out when any record can't be used: a reply's signatures come from one
   * model, so a mismatch on one part means all of them. A part without a payload has no signature
   * and signals nothing.
   *
   * @returns Each part's signature, `null` for none, all `null` when they are left out; and
   *   whether they were left out.
   * @throws {TypeError} If the message holds a tool result.
   */
  #signatures(message: Message): [(string | null)[], boolean] {
    const signatures: (string | null)[] = [];
    let reason: ReplayDegradedReason | null = null;
    for (const part of message.parts) {
      // Refused here so a bad message never fires onReplayDegraded first.
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(part, message.role));
      if (!(part instanceof TextPart || part instanceof ToolCallPart) || part.providerPayload === null) {
        signatures.push(null);
        continue;
      }
      const read = thoughtSignature(part.providerPayload, this.#key);
      if (!read.ok) reason ??= read.reason;
      signatures.push(read.ok ? read.signature : null);
    }
    if (reason === null) return [signatures, false];
    this.#onReplayDegraded?.(reason);
    return [message.parts.map(() => null), true];
  }

  /** Warn, the first time, that an empty user message was not sent. */
  #warnEmptyUser(): void {
    if (this.#emptyUserWarned) return;
    this.#emptyUserWarned = true;
    process.emitWarning(EMPTY_USER);
  }
}

/**
 * Read a text or call part's replay record and return its `thought_signature`; a record without
 * it as a non-empty string is `malformed`.
 */
function thoughtSignature(
  payload: JsonObject,
  key: ReplayKey,
): { ok: true; signature: string } | { ok: false; reason: ReplayDegradedReason } {
  const read = readReplayRecord(payload, key, PART_FIELDS);
  if (!read.ok) return read;
  const signature = read.fields["thought_signature"];
  if (typeof signature === "string" && signature !== "") return { ok: true, signature };
  return { ok: false, reason: "malformed" };
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

/** Render a tool as a function declaration; Gemini has no per-tool `strict`. */
function functionDeclaration(spec: ToolSpec): JsonObject {
  // Copied so changing a body cannot change the spec it came from.
  return { name: spec.name, description: spec.description, parametersJsonSchema: copyJson(spec.parameters) };
}

/** Render the tool choice as a `toolConfig`, or `null` for none. */
function toolConfig(choice: ToolChoice): JsonObject | null {
  if (choice instanceof NamedTool) {
    return { functionCallingConfig: { mode: "ANY", allowedFunctionNames: [choice.name] } };
  }
  if (choice === "required") return { functionCallingConfig: { mode: "ANY" } };
  if (choice === "none") return { functionCallingConfig: { mode: "NONE" } };
  return null;
}

/** Return the error text for a part a message of `role` cannot carry. */
function wrongPart(part: Part, role: string): string {
  const article = role === "assistant" ? "an" : "a";
  return `the gemini route cannot send a ${part.constructor.name} in ${article} ${role} message`;
}
