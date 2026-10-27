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
 * only to the provider, model and account that wrote it. Only the first function call of a model
 * content can get Google's placeholder signature: when it goes out unsigned and the message is not
 * this model's own reply, or its records couldn't be used. Every tool goes in one
 * `functionDeclarations` entry with its parameters as written, and a tool choice becomes
 * `toolConfig`, both only with tools. `safetySettings: []` and `generationConfig.candidateCount: 1`
 * are always sent, and the provider block is merged last.
 *
 * Each streamed chunk is a whole `GenerateContentResponse`, of which the first candidate is read.
 * Text and thought pieces stream as deltas and join into one text or reasoning part until a call,
 * a switch between thought and text, or a second signature starts another, with dense part
 * indexes; a part whose wire parts carried a thought signature gets a replay record holding it,
 * and an unsigned thought an envelope-only one.
 * A call arrives whole, as one delta, with `yian_call_<n>` as its id when Gemini sends none or
 * repeats one. Usage counts merge field by field across chunks, and the finish comes at the end of
 * the stream. A reply that failed, was refused, blocked or ended early raises `ProviderError`; an
 * error message holds the provider's text uncapped, which `RouteModel` scrubs of the route's
 * credentials and then caps.
 */

import { encodeBody, escapeAuthorTagsInJson, mergeRequestBody, withAuthorWrapper } from "../body.ts";
import { JSONDecodeError, type JsonObject, type JsonValue, parseJson } from "../../typesys/json.ts";
import { kindForStatus, ProviderError } from "../errors.ts";
import { type Event, Finish, PartDone, ReasoningDelta, TextDelta, ToolCallDelta, UsageReport } from "../events.ts";
import type { SseProtocol } from "../executor.ts";
import type { SSEEvent } from "../framing.ts";
import { warn } from "../../platform.ts";
import {
  copyJson,
  isPlainObject,
  MediaPart,
  type Message,
  MessageFormatError,
  NamedTool,
  type Part,
  ReasoningPart,
  readReplayRecord,
  type ReplayDegradedReason,
  type ReplayKey,
  replayRecord,
  type Request,
  TextPart,
  ToolCallPart,
  type ToolChoice,
  ToolResultPart,
  type ToolSpec,
  Usage,
} from "../messages.ts";
import { callId, eventData, notIntMessage, tokenCount, wrongPart } from "./common.ts";

/** The protocol id, in `Finish.protocol` and replay records. */
export const PROTOCOL = "gemini";
const NO_MEDIA = "the gemini route does not send media parts yet";
const EMPTY_USER = "an empty user message was not sent; Gemini rejects empty content";
// The fields of a text, call or reasoning part's replay record.
const PART_FIELDS = ["thought_signature"];
// The fields of a reply's message record, which marks a message as this model's own reply.
const MESSAGE_FIELDS = ["response_id", "response_model"];
// Google's documented signature for a function call that Gemini didn't write.
const PLACEHOLDER = "skip_thought_signature_validator";
// The usage counts read; Vertex adds count-free fields (`trafficType`), which are ignored.
const COUNT_FIELDS = [
  "promptTokenCount",
  "toolUsePromptTokenCount",
  "candidatesTokenCount",
  "thoughtsTokenCount",
  "cachedContentTokenCount",
  "totalTokenCount",
];
// Finish reasons that mean Gemini refused the content; retrying the same request can't help.
const REFUSED_FINISH: ReadonlySet<string> = new Set([
  "SAFETY",
  "RECITATION",
  "LANGUAGE",
  "BLOCKLIST",
  "PROHIBITED_CONTENT",
  "SPII",
  "IMAGE_SAFETY",
  "IMAGE_PROHIBITED_CONTENT",
  "IMAGE_RECITATION",
  "MODEL_ARMOR",
  "PUP_LIMITED_DISABLED",
]);

/** A part being streamed, with the thought signature its wire parts carried, `null` for none. */
type Streamed =
  | { kind: "text" | "reasoning"; text: string; signature: string | null }
  | {
      kind: "call";
      id: string;
      name: string;
      args: JsonObject | null;
      rawArguments: string | null;
      signature: string | null;
    };

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
   * `generationConfig: {candidateCount: 1}`, then the block merged over them. A request with an
   * `outputFormat` adds `responseMimeType: "application/json"` and its schema (copied, as
   * written) as `responseJsonSchema` to `generationConfig`; its name and `strict` are not sent.
   *
   * Every system message's non-empty text parts, wherever the message is, join
   * `systemInstruction.parts` in order. A user message is a `user` content of its non-empty text
   * parts; one with none is not sent, with a warning through `warn` (once per instance). Each
   * run of tool messages is one `user` content of a `functionResponse` per result, with the
   * result's call id and name; only a user or model content that is sent ends
   * the run, so a system message or a skipped message in between does not. Its `response` is the
   * result's text read as JSON when that is a plain object, else `{output: <the value>}`, or
   * `{output: <the text>}` when the text is not plain JSON. `isError` is not sent. An assistant
   * message is a `model` content in part order: a text part as `{text}`, a call as
   * `functionCall` with its id, name and arguments (`{}` for an invalid call); reasoning and
   * refusal parts are not sent, and a message rendering no part is skipped. A trailing assistant
   * message goes as written.
   *
   * A user message's author is sent as the author wrapper, one text part per tag. The author tag
   * is escaped in user text and tool results, and again in a result's decoded JSON.
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
    const [system, contents] = this.#contents(request.messages.map((message) => withAuthorWrapper(message)));
    const body: JsonObject = { contents };
    if (system.length > 0) body["systemInstruction"] = { parts: system };
    if (request.tools.length > 0) {
      body["tools"] = [{ functionDeclarations: request.tools.map(functionDeclaration) }];
      const config = toolConfig(request.toolChoice);
      if (config !== null) body["toolConfig"] = config;
    }
    body["safetySettings"] = [];
    const generationConfig: JsonObject = { candidateCount: 1n };
    if (request.outputFormat !== null) {
      generationConfig["responseMimeType"] = "application/json";
      generationConfig["responseJsonSchema"] = copyJson(request.outputFormat.schema);
    }
    body["generationConfig"] = generationConfig;
    if (this.#block) mergeRequestBody(body, this.#block);
    return body;
  }

  /**
   * Turn a streamed reply's events into stream events.
   *
   * Each event's data is a whole `GenerateContentResponse`; empty and non-object data are
   * skipped, and an explicit `null` reads as absent. The last string `responseId` and
   * `modelVersion` win, and the last non-empty `finishReason` with its `finishMessage`; every
   * chunk is read (usage may come after the `finishReason`), and only the first candidate's parts
   * are read, in order. A `functionCall` part is a call; else a part with `text` is a text piece,
   * or a reasoning piece when `thought` is true; else a part with a `thoughtSignature` but no
   * `text` (of any other kind too, whose content is dropped) is an empty piece with that
   * signature; other unsigned part kinds are skipped. A piece joins the reply's last part when
   * that is a text or reasoning part of its kind and not both are signed, else it opens a new
   * part (an empty, unsigned piece opens nothing); its signature goes to the part it joins.
   * Non-empty pieces stream as deltas. A call's id is its own unless missing or already in the
   * reply, else `yian_call_<n>`, `n` its position among the calls, with `_<k>` appended while
   * taken; its `args` are as sent, `{}` when missing, and anything but an object makes an invalid
   * call with the raw args. Each of the six usage counts merges across chunks, a missing one
   * replacing nothing; `totalTokenCount` is checked but not used, the total being input + output.
   *
   * @param frames The reply's server-sent events, in order.
   * @returns The deltas (one `ToolCallDelta` per call, with its id, name and encoded args), then
   *   at the end of the stream a `PartDone` for each part in index order (a part with a
   *   signature carries a record of it), `UsageReport` when a count other than the total is
   *   above zero, and `Finish`: `length` for `MAX_TOKENS`, else `tool_calls` when a call came,
   *   else `stop`, with a record of the response id and model.
   * @throws {ProviderError} For an `error` object in a chunk (kind from its `code`), a blocked
   *   prompt or a safety-type finish reason (`refused`), `CONTINUATION` (`protocol`), any other
   *   finish reason but `STOP` and `MAX_TOKENS` (`incomplete`, retryable for
   *   `MALFORMED_RESPONSE`), a stream that ends with no finish reason (`incomplete`, retryable),
   *   or an event that is not JSON or holds a field or usage that cannot be read (`protocol`).
   *   The message holds the provider's text uncapped.
   */
  async *parse(frames: AsyncIterable<SSEEvent>): AsyncGenerator<Event> {
    const provider = this.provider;
    const invalid = (field: string) => new ProviderError("protocol", `invalid ${field} in ${provider} stream event`);
    // Part index = position, so indexes are dense in the order parts open.
    const parts: Streamed[] = [];
    // Every call id in the reply so far, the server's and those generated.
    const taken = new Set<string>();
    let calls = 0;
    const counts = new Map<string, number>();
    let responseId: string | null = null;
    let responseModel: string | null = null;
    let finishReason: string | null = null;
    let finishMessage: string | null = null;
    let blockReason: string | null = null;
    let blockMessage: string | null = null;
    for await (const frame of frames) {
      if (frame.data === "") continue;
      const data = eventData(frame.data, provider);
      if (!isPlainObject(data)) continue;
      if (isPlainObject(data["error"])) throw this.#streamFailure(data["error"]);
      if (typeof data["responseId"] === "string") responseId = data["responseId"];
      if (typeof data["modelVersion"] === "string") responseModel = data["modelVersion"];
      mergeUsage(counts, data["usageMetadata"], provider);
      const feedback = data["promptFeedback"] ?? null;
      if (feedback !== null) {
        if (!isPlainObject(feedback)) throw invalid("promptFeedback");
        const reason = feedback["blockReason"];
        if (typeof reason === "string" && reason !== "") {
          blockReason = reason;
          blockMessage = typeof feedback["blockReasonMessage"] === "string" ? feedback["blockReasonMessage"] : null;
        }
      }
      const candidate = firstCandidate(data, invalid);
      if (candidate === null) continue;
      for (const part of candidateParts(candidate, invalid)) {
        if (part === null) continue;
        if (!isPlainObject(part)) throw invalid("part");
        const text = part["text"] ?? null;
        const rawSignature = part["thoughtSignature"] ?? null;
        if (text !== null && typeof text !== "string") throw invalid("part");
        if (rawSignature !== null && typeof rawSignature !== "string") throw invalid("part");
        const signature = rawSignature || null;
        const call = part["functionCall"] ?? null;
        if (call !== null) {
          const { id: wireId, name, args, rawArguments } = readCall(call, invalid);
          const id = callId(wireId, calls++, taken);
          const index = parts.push({ kind: "call", id, name, args, rawArguments, signature }) - 1;
          yield new ToolCallDelta({ index, id, name, arguments: rawArguments ?? encodeBody(args) });
          continue;
        }
        if (text === null && signature === null) continue;
        const kind = part["thought"] === true ? "reasoning" : "text";
        const piece = text ?? "";
        const last = parts.at(-1);
        let index: number;
        // A call closes the open part; two signatures never share one.
        if (last !== undefined && last.kind !== "call" && last.kind === kind && !(last.signature && signature)) {
          last.text += piece;
          last.signature ??= signature;
          index = parts.length - 1;
        } else if (piece === "" && signature === null) {
          continue;
        } else {
          index = parts.push({ kind, text: piece, signature }) - 1;
        }
        if (piece !== "") yield kind === "text" ? new TextDelta(index, piece) : new ReasoningDelta(index, piece);
      }
      const reason = candidate["finishReason"];
      // An empty reason is no finish, as an empty id or signature is none.
      if (typeof reason === "string" && reason !== "") {
        finishReason = reason;
        finishMessage = typeof candidate["finishMessage"] === "string" ? candidate["finishMessage"] : null;
      }
    }
    if (blockReason !== null) {
      throw new ProviderError("refused", withDetail(`${provider} blocked the prompt (${blockReason})`, blockMessage));
    }
    if (finishReason === null) {
      throw new ProviderError("incomplete", `${provider} stream ended before a complete response`, { retryable: true });
    }
    const failure = finishFailure(finishReason, finishMessage, provider);
    if (failure !== null) throw failure;
    // Counted first, so usage that can't be read fails before any part is done.
    const usage = usageOf(counts, provider);
    for (const [index, part] of parts.entries()) yield new PartDone(index, this.#part(part));
    if (usage !== null) yield new UsageReport(usage);
    const fields: JsonObject = {};
    if (responseId !== null) fields["response_id"] = responseId;
    if (responseModel !== null) fields["response_model"] = responseModel;
    const providerPayload = replayRecord(this.#key, fields);
    // `MAX_TOKENS` says the calls may be cut short, so `length` wins over them.
    const reason = finishReason === "MAX_TOKENS" ? "length" : calls > 0 ? "tool_calls" : "stop";
    yield new Finish({ reason, responseId, protocol: PROTOCOL, providerPayload });
  }

  /**
   * Build a streamed part, with a record of its signature when it has one; an unsigned thought
   * gets an envelope-only record, so every reasoning part names the route that wrote it.
   */
  #part(part: Streamed): Part {
    const record = part.signature === null ? null : replayRecord(this.#key, { thought_signature: part.signature });
    if (part.kind === "reasoning") return new ReasoningPart(part.text, record ?? replayRecord(this.#key, {}));
    if (part.kind !== "call") return new TextPart(part.text, record);
    const { id, name, args, rawArguments } = part;
    return new ToolCallPart({ id, name, arguments: args, rawArguments, providerPayload: record });
  }

  /**
   * Return the error for a chunk's `error` object: its kind from its `code` when that is an HTTP
   * error status, else `server`; its message from `message`, else `status`.
   */
  #streamFailure(error: JsonObject): ProviderError {
    const code = error["code"];
    // A 1xx-3xx code would map to kinds that don't fit an error inside a 200 stream.
    const status = typeof code === "bigint" && code >= 400n && code <= 599n ? Number(code) : null;
    const kind = status === null ? "server" : kindForStatus(status, encodeBody(error));
    const text = (key: string): string | null => {
      const value = error[key];
      return typeof value === "string" && value !== "" ? value : null;
    };
    const message = text("message") ?? text("status") ?? "no message";
    return new ProviderError(kind, `${this.provider} stream failed (${kind}): ${message}`);
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
    if (firstCall !== null && !Object.hasOwn(firstCall, "thoughtSignature")) {
      if (degraded || !readReplayRecord(message.providerPayload, this.#key, MESSAGE_FIELDS).ok) {
        firstCall["thoughtSignature"] = PLACEHOLDER;
      }
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
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
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
    warn(EMPTY_USER);
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
    if (!(part instanceof TextPart)) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
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
  if (!(part instanceof ToolResultPart)) throw new TypeError(wrongPart(PROTOCOL, part, "tool"));
  return { functionResponse: { id: part.callId, name: part.name, response: toolResponse(part.content) } };
}

/**
 * Return a tool result's `response`: its text read as plain JSON when that is an object, else
 * `{output: <the value>}`, or `{output: <the text>}` when the text is not plain JSON (Gemini
 * takes only an object here). The author tag is escaped in the decoded strings and keys.
 */
function toolResponse(content: string): JsonObject {
  let value: JsonValue;
  try {
    value = parseJson(content, "plain");
  } catch (error) {
    if (error instanceof JSONDecodeError) return { output: content };
    throw error;
  }
  // The text was escaped as text, but a JSON escape such as `\u003c` decodes to a literal `<`.
  value = escapeAuthorTagsInJson(value);
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

/**
 * Return a chunk's first candidate, or `null` when it has none; yian sends `candidateCount: 1`.
 *
 * @throws {ProviderError} From `invalid`, when `candidates` is not a list or the candidate not an object.
 */
function firstCandidate(data: JsonObject, invalid: (field: string) => ProviderError): JsonObject | null {
  const candidates = data["candidates"] ?? null;
  if (candidates === null) return null;
  if (!Array.isArray(candidates)) throw invalid("candidates");
  const candidate = candidates[0] ?? null;
  if (candidate === null) return null;
  if (!isPlainObject(candidate)) throw invalid("candidate");
  return candidate;
}

/**
 * Return a candidate's `content.parts`, empty when either is absent.
 *
 * @throws {ProviderError} From `invalid`, when `content` is not an object or `parts` not a list.
 */
function candidateParts(candidate: JsonObject, invalid: (field: string) => ProviderError): JsonValue[] {
  const content = candidate["content"] ?? null;
  if (content === null) return [];
  if (!isPlainObject(content)) throw invalid("content");
  const parts = content["parts"] ?? null;
  if (parts === null) return [];
  if (!Array.isArray(parts)) throw invalid("parts");
  return parts;
}

/**
 * Read a `functionCall`: its id (`null` when missing or empty), name, and arguments: `{}` when
 * missing, an object as sent, else `null` with the raw args.
 *
 * @throws {ProviderError} From `invalid`, when the call is not an object, its id not a string, or
 *   its name missing, empty or not a string.
 */
function readCall(
  call: JsonValue,
  invalid: (field: string) => ProviderError,
): { id: string | null; name: string; args: JsonObject | null; rawArguments: string | null } {
  if (!isPlainObject(call)) throw invalid("functionCall");
  const rawId = call["id"] ?? null;
  const name = call["name"];
  if ((rawId !== null && typeof rawId !== "string") || typeof name !== "string" || name === "") {
    throw invalid("functionCall");
  }
  const id = rawId || null;
  const args = call["args"] ?? null;
  if (args === null) return { id, name, args: {}, rawArguments: null };
  if (isPlainObject(args)) return { id, name, args, rawArguments: null };
  return { id, name, args: null, rawArguments: encodeBody(args) };
}

/**
 * Merge a chunk's `usageMetadata` counts into `counts`, field by field; a missing or `null` count
 * replaces nothing, so a count-free object keeps earlier counts.
 *
 * @throws {ProviderError} `usageMetadata` is not an object, or a count is not an int within
 *   `Number.MAX_SAFE_INTEGER` (kind `protocol`).
 */
function mergeUsage(counts: Map<string, number>, usage: JsonValue | undefined, provider: string): void {
  if (usage === undefined || usage === null) return;
  if (!isPlainObject(usage)) throw new ProviderError("protocol", `${provider} response usage is not an object`);
  const notInt = notIntMessage(provider);
  for (const field of COUNT_FIELDS) {
    const count = tokenCount(usage[field], notInt);
    if (count !== null) counts.set(field, count);
  }
}

/**
 * Return the usage of the merged counts, or `null` when no count it uses is above zero. Input is
 * the prompt and tool-use prompt, output the candidates and thoughts, and the total their sum
 * (`totalTokenCount` is checked but not used).
 *
 * @throws {ProviderError} A sum is past `Number.MAX_SAFE_INTEGER` (kind `protocol`).
 */
function usageOf(counts: Map<string, number>, provider: string): Usage | null {
  const count = (field: string): number => counts.get(field) ?? 0;
  const thoughts = count("thoughtsTokenCount");
  const inputTokens = count("promptTokenCount") + count("toolUsePromptTokenCount");
  const outputTokens = count("candidatesTokenCount") + thoughts;
  if (inputTokens + outputTokens === 0 && count("cachedContentTokenCount") === 0) return null;
  try {
    return new Usage({
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + outputTokens,
      cacheReadTokens: counts.get("cachedContentTokenCount") ?? null,
      cacheCreationTokens: null,
      reasoningTokens: thoughts > 0 ? thoughts : null,
    });
  } catch (error) {
    // `Usage` refuses a sum past `Number.MAX_SAFE_INTEGER`, which `Number` arithmetic rounds.
    if (error instanceof MessageFormatError) throw new ProviderError("protocol", notIntMessage(provider));
    throw error;
  }
}

/**
 * Return the error for a finish reason that means the reply failed, or `null` for `STOP` and
 * `MAX_TOKENS`: a safety-type reason is `refused`, `CONTINUATION` `protocol`, any other
 * `incomplete`, retryable only for `MALFORMED_RESPONSE`.
 */
function finishFailure(reason: string, message: string | null, provider: string): ProviderError | null {
  if (reason === "STOP" || reason === "MAX_TOKENS") return null;
  if (reason === "CONTINUATION") {
    const text = `${provider} asked to continue the reply in another request, which yian does not send`;
    return new ProviderError("protocol", text);
  }
  const text = withDetail(`${provider} response did not complete (${reason})`, message);
  if (REFUSED_FINISH.has(reason)) return new ProviderError("refused", text);
  // Gemini sporadically ends a valid request with `MALFORMED_RESPONSE`, and a resend recovers.
  return new ProviderError("incomplete", text, { retryable: reason === "MALFORMED_RESPONSE" });
}

/** Return `text`, followed by `: <detail>` when `detail` is a non-empty string. */
function withDetail(text: string, detail: string | null): string {
  return detail ? `${text}: ${detail}` : text;
}
