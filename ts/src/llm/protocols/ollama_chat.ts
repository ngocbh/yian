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
 * `done: true` ends it with the finish reason and the token counts, and nothing after it is read.
 * Thinking and content stream as deltas into the reply's one reasoning part and one text part,
 * even when a call or the other kind comes between their pieces, because an Ollama message has
 * one `thinking` and one `content` field and both go back whole. Each call arrives whole, as one
 * delta, keeping the server's id unless it is missing, empty or repeated. The reasoning part
 * carries a replay record, and the reply's record holds the model the server named. A `done` line
 * with `done_reason: "load"` and missing or blank content is skipped; logprobs and other unknown
 * fields are not read. A reply that failed or ended early raises `ProviderError`; an error message
 * holds the provider's text uncapped, which `RouteModel` scrubs of the route's credentials and
 * then caps.
 */

import { encodeBody, mergeRequestBody } from "../body.ts";
import type { JsonObject, JsonValue } from "../../typesys/json.ts";
import { ProviderError } from "../errors.ts";
import {
  type Event,
  Finish,
  PartDone,
  parseArguments,
  ReasoningDelta,
  TextDelta,
  ToolCallDelta,
  UsageReport,
} from "../events.ts";
import type { NdjsonProtocol } from "../executor.ts";
import {
  copyJson,
  isPlainObject,
  MediaPart,
  type Message,
  MessageFormatError,
  type Part,
  readReplayRecord,
  ReasoningPart,
  type ReplayDegradedReason,
  type ReplayKey,
  replayRecord,
  type Request,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  type ToolSpec,
  Usage,
} from "../messages.ts";

/** The protocol id, in `Finish.protocol` and replay records. */
export const PROTOCOL = "ollama_chat";
const NO_MEDIA = "the ollama_chat route does not send media parts yet";

/** A text or reasoning part being streamed. */
type Prose = { kind: "text" | "reasoning"; text: string };

/** A part being streamed. */
type Streamed =
  | Prose
  | { kind: "call"; id: string; name: string; args: JsonObject | null; rawArguments: string | null };

/** A call read from a line's `tool_calls`. */
interface WireCall {
  id: string | null;
  name: string;
  args: JsonObject | null;
  rawArguments: string | null;
}

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

  /**
   * Turn the reply's NDJSON lines into stream events.
   *
   * Lines are read in order until the one with `done: true`. A missing or `null` `message`,
   * `content`, `thinking` or `tool_calls` reads as absent, and unknown fields (logprobs, durations,
   * `remote_model`, anything new) are ignored. A `done` line with `done_reason: "load"` and missing
   * or blank content is skipped whole, its `model` included, and does not end the reply. The last
   * string `model` on any line read is kept. On each line its thinking, then its content, then its
   * calls are read: thinking pieces join the reply's one reasoning part and content pieces its one
   * text part, each opened by its first non-empty piece, with dense part indexes in the order parts
   * open. Each call is a new part: its id is its own unless missing, empty or already in the reply,
   * else `yian_call_<n>`, `n` its position among the calls, with `_<k>` appended while taken; its
   * `arguments` are an object as sent, a string parsed with `parseArguments`, `{}` when missing,
   * and anything else makes an invalid call with the raw JSON. Usage comes from the `done` line:
   * `prompt_eval_count` is input and `eval_count` output (a missing one is 0, the total their sum)
   * and `prompt_eval_cached_count` the cache reads.
   *
   * @param frames The reply's parsed NDJSON lines, in order.
   * @returns The deltas (one `ToolCallDelta` per call, with its id, name and arguments text), then
   *   at the `done` line a `PartDone` for each part in index order (the reasoning part with a
   *   replay record), `UsageReport` unless every count is 0 or missing, and `Finish`: `length` for
   *   `done_reason: "length"`, else `tool_calls` when a call came, else `stop` for `stop`, a
   *   missing, `null` or empty reason, else the reason as given, with a record of the model.
   * @throws {ProviderError} For a line holding a non-`null` `error` (`server`, retryable; its
   *   message the error text as given, `no message` when that is blank, or the JSON of a value
   *   that is not a string); a line that is not a JSON object, a `message` that is not an object,
   *   `content` or `thinking` neither a string nor `null`, `tool_calls` not a list, a call or its
   *   `function` not an object, a call id neither a string nor `null`, a call with no name, a
   *   `done_reason` neither a string nor `null`, or a count or total that is not an int within
   *   `Number.MAX_SAFE_INTEGER` (`protocol`); or a reply that ends with no `done` line
   *   (`incomplete`, retryable).
   */
  async *parse(frames: AsyncIterable<JsonValue>): AsyncGenerator<Event> {
    const provider = this.provider;
    // Part index = position, so indexes are dense in the order parts open.
    const parts: Streamed[] = [];
    // The reply's one text part and one reasoning part, with their indexes, once opened.
    const opened = new Map<Prose["kind"], { index: number; part: Prose }>();
    // Every call id in the reply so far, the server's and those generated.
    const taken = new Set<string>();
    let calls = 0;
    let responseModel: string | null = null;
    for await (const line of frames) {
      if (!isPlainObject(line)) throw new ProviderError("protocol", `${provider} stream line is not a JSON object`);
      const error = line["error"] ?? null;
      if (error !== null) throw this.#streamFailure(error);
      const message = line["message"] ?? {};
      if (!isPlainObject(message)) throw invalidMessage(provider);
      const content = message["content"] ?? null;
      const thinking = message["thinking"] ?? null;
      const toolCalls = message["tool_calls"] ?? [];
      if (!isStringOrNull(content) || !isStringOrNull(thinking) || !Array.isArray(toolCalls)) {
        throw invalidMessage(provider);
      }
      const done = line["done"] === true;
      const doneReason = line["done_reason"] ?? null;
      if (!isStringOrNull(doneReason)) {
        throw new ProviderError("protocol", `${provider} stream line has an invalid done_reason`);
      }
      // Ollama sends this for a request with no messages, which only loads the model.
      if (done && doneReason === "load" && (content ?? "").trim() === "") continue;
      if (typeof line["model"] === "string") responseModel = line["model"];
      for (const [kind, piece] of [
        ["reasoning", thinking],
        ["text", content],
      ] as const) {
        if (!piece) continue;
        let open = opened.get(kind);
        if (open === undefined) {
          const part = { kind, text: "" };
          open = { index: parts.push(part) - 1, part };
          opened.set(kind, open);
        }
        open.part.text += piece;
        yield kind === "text" ? new TextDelta(open.index, piece) : new ReasoningDelta(open.index, piece);
      }
      for (const entry of toolCalls) {
        const { id: wireId, name, args, rawArguments } = readCall(entry, calls, provider);
        const id = callId(wireId, calls++, taken);
        const index = parts.push({ kind: "call", id, name, args, rawArguments }) - 1;
        yield new ToolCallDelta({ index, id, name, arguments: rawArguments ?? encodeBody(args) });
      }
      if (!done) continue;
      // Counted first, so usage that can't be read fails before any part is done.
      const usage = usageOf(line, provider);
      for (const [index, part] of parts.entries()) yield new PartDone(index, this.#part(part));
      if (usage !== null) yield new UsageReport(usage);
      const providerPayload = replayRecord(this.#key, responseModel === null ? {} : { response_model: responseModel });
      // `length` says the calls may be cut short, so it wins over them.
      const reason = doneReason === "length" ? "length" : calls > 0 ? "tool_calls" : doneReason || "stop";
      yield new Finish({ reason, responseId: null, protocol: PROTOCOL, providerPayload });
      return;
    }
    throw new ProviderError("incomplete", `${provider} stream ended before a complete response`, { retryable: true });
  }

  /**
   * Return the error for a line's `error`: kind `server`, its message the text as given, `no
   * message` when that is blank, or the JSON of a value that is not a string.
   */
  #streamFailure(error: JsonValue): ProviderError {
    let text: string;
    if (typeof error === "string") text = error.trim() === "" ? "no message" : error;
    else text = encodeBody(error);
    return new ProviderError("server", `${this.provider} stream failed (server): ${text}`);
  }

  /** Build a streamed part; only a reasoning part carries a record, of the key alone. */
  #part(part: Streamed): Part {
    if (part.kind !== "call") {
      return part.kind === "text" ? new TextPart(part.text) : new ReasoningPart(part.text, replayRecord(this.#key, {}));
    }
    const { id, name, args, rawArguments } = part;
    return new ToolCallPart({ id, name, arguments: args, rawArguments });
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

/** Return whether `value` is a string or `null`. */
function isStringOrNull(value: JsonValue): value is string | null {
  return value === null || typeof value === "string";
}

/** Return the error for a line whose `message` has a bad shape. */
function invalidMessage(provider: string): ProviderError {
  return new ProviderError("protocol", `${provider} stream line has an invalid message`);
}

/**
 * Read a `tool_calls` entry: its id (`null` when missing or empty), name, and arguments: an
 * object as sent, a string through `parseArguments`, `{}` when missing, else `null` with the raw
 * JSON.
 *
 * @param n The call's position among the reply's calls, for the error message.
 * @throws {ProviderError} When the entry or its `function` is not an object, its id is neither a
 *   string nor `null`, or its name is missing, empty or not a string (kind `protocol`).
 */
function readCall(entry: JsonValue, n: number, provider: string): WireCall {
  if (!isPlainObject(entry)) throw invalidMessage(provider);
  const fn = entry["function"];
  const rawId = entry["id"] ?? null;
  if (!isPlainObject(fn) || !isStringOrNull(rawId)) throw invalidMessage(provider);
  const id = rawId || null;
  const name = fn["name"];
  if (typeof name !== "string" || name === "") {
    throw new ProviderError("protocol", `${provider} tool call ${n} has no name`);
  }
  const value = fn["arguments"] ?? null;
  if (value === null) return { id, name, args: {}, rawArguments: null };
  if (isPlainObject(value)) return { id, name, args: value, rawArguments: null };
  if (typeof value === "string") {
    const [args, rawArguments] = parseArguments(value);
    return { id, name, args, rawArguments };
  }
  return { id, name, args: null, rawArguments: encodeBody(value) };
}

/**
 * Return a call's id: its own unless missing or in `taken`, else `yian_call_<n>` with `_<k>`
 * appended while that is in `taken`. The id it gets is added to `taken`.
 *
 * @param n The call's position among the reply's calls.
 * @param taken The ids in the reply so far: the server's and those generated.
 */
function callId(id: string | null, n: number, taken: Set<string>): string {
  let out = id;
  if (out === null || taken.has(out)) {
    out = `yian_call_${n}`;
    for (let k = 1; taken.has(out); k++) out = `yian_call_${n}_${k}`;
  }
  taken.add(out);
  return out;
}

/**
 * Return the `done` line's usage, or `null` when input, output and cache reads are all 0 or
 * missing. A missing input or output count is 0 and the total their sum; the cache reads are
 * `prompt_eval_cached_count`, `null` when missing.
 *
 * @throws {ProviderError} A count, or the total, is not an int within `Number.MAX_SAFE_INTEGER`
 *   (kind `protocol`).
 */
function usageOf(line: JsonObject, provider: string): Usage | null {
  const notInt = `${provider} response usage has a count that is not an int`;
  const inputTokens = tokenCount(line["prompt_eval_count"], notInt) ?? 0;
  const outputTokens = tokenCount(line["eval_count"], notInt) ?? 0;
  const cacheReadTokens = tokenCount(line["prompt_eval_cached_count"], notInt);
  if (inputTokens + outputTokens === 0 && (cacheReadTokens ?? 0) === 0) return null;
  try {
    return new Usage({
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + outputTokens,
      cacheReadTokens,
      cacheCreationTokens: null,
      reasoningTokens: null,
    });
  } catch (error) {
    // `Usage` refuses a sum past `Number.MAX_SAFE_INTEGER`, which `Number` arithmetic rounds.
    if (error instanceof MessageFormatError) throw new ProviderError("protocol", notInt);
    throw error;
  }
}

/** Return a usage count as a number, `null` when it is `null` or absent; throw `notInt` otherwise. */
function tokenCount(value: JsonValue | undefined, notInt: string): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "bigint" || !Number.isSafeInteger(Number(value))) throw new ProviderError("protocol", notInt);
  return Number(value);
}
