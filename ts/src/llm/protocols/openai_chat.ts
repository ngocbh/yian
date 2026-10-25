/**
 * The OpenAI Chat Completions wire format (`POST /chat/completions`), spoken by Azure OpenAI and
 * the OpenAI-compatible vendors (xAI, DeepSeek, Qwen, GLM, MiniMax, OpenRouter, vLLM).
 *
 * Each message becomes a chat message: system and user text, an assistant message's text and
 * tool calls, and one `tool` message per tool result. Every call streams. The vendors differ in
 * a few request rules, which `ChatOptions` holds per route: DeepSeek gets earlier reasoning back
 * as `reasoning_content`, MiniMax's reasoning models get `reasoning_split`, some models refuse a
 * named tool choice, and most Azure deployments refuse `parallel_tool_calls`. A user message's
 * author is folded into its first and last texts as the author wrapper (`name` is built, but no
 * route sends it).
 *
 * Replies stream text, reasoning and tool calls as deltas; a refusal is collected into one part.
 * A call without an id gets one (`yian_call_<n>`). The finish is held until the stream ends, so
 * the usage chunk that follows it is read. Reasoning parts carry replay records naming the wire
 * field they came from, so only reasoning this route parsed from `reasoning_content` is echoed.
 * DeepSeek requires every earlier turn's `reasoning_content` when the request sends tools, so on
 * a tool request a turn with no reasoning to echo sends `reasoning_content: ""`.
 * A reply that failed, was filtered, ran out of capacity or ended early raises `ProviderError`.
 * An error message holds the provider's text uncapped; `RouteModel` scrubs it of the route's
 * credentials and then caps it.
 */

import type { JsonObject, JsonValue } from "../../typesys/json.ts";
import { encodeBody, mergeRequestBody, withAuthorWrapper } from "../body.ts";
import { getCapabilities } from "../capabilities.ts";
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
import {
  copyJson,
  isPlainObject,
  MediaPart,
  type Message,
  MessageFormatError,
  NamedTool,
  type OutputFormat,
  type Part,
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
import { eventData, notIntMessage, tokenCount, wrongPart } from "./common.ts";
import { STREAM_ERROR_KINDS } from "./openai_responses.ts";

/** The protocol id, in `Finish.protocol` and replay records. */
export const PROTOCOL = "openai_chat";
const NO_MEDIA = "the openai_chat route does not send media parts yet";
// The wire field a reasoning record names when its text may be echoed.
const ECHOED_FIELD = "reasoning_content";
// Error chunk codes: the Responses table, plus OpenRouter's content filter codes. Any other
// code, or none, is a server error.
const CHUNK_ERROR_KINDS: ReadonlyMap<string, ErrorKind> = new Map<string, ErrorKind>([
  ...STREAM_ERROR_KINDS,
  ["content_filter", "refused"],
  ["content_policy_violation", "refused"],
]);
// Finish reasons that mean the reply failed; any other reason is passed through.
const FAILED_FINISH: ReadonlyMap<string, ErrorKind> = new Map<string, ErrorKind>([
  ["content_filter", "refused"],
  // DeepSeek: its inference system ran out of capacity.
  ["insufficient_system_resource", "overloaded"],
  // OpenRouter's normalized failure when it sent no error object, and DeepSeek's abort.
  ["error", "server"],
  ["aborted", "server"],
]);

/** A part being streamed; its index is its position among the reply's parts. */
type Assembly = TextAssembly | CallAssembly;

/** A reasoning, text or refusal part being streamed. */
interface TextAssembly {
  kind: "reasoning" | "text" | "refusal";
  text: string;
  /** For reasoning, the wire field of the piece that opened it. */
  field: string | null;
}

/** A tool call being streamed. */
interface CallAssembly {
  kind: "call";
  /** The first non-empty id sent, or `null`. */
  id: string | null;
  /** The first non-empty name sent, or `null`. */
  name: string | null;
  /** The arguments pieces so far, joined. */
  args: string;
  /** The wire `index` of the entry that opened it, or `null`. */
  wireIndex: bigint | null;
}

/** One `tool_calls` entry of a delta; an `id` or `name` that is not a non-empty string is `null`. */
interface CallPiece {
  wireIndex: bigint | null;
  id: string | null;
  name: string | null;
  args: string;
}

/** The request rules that differ between Chat Completions routes. */
export interface ChatOptions {
  /**
   * Send earlier assistant reasoning back as `reasoning_content` (DeepSeek), and `""` for a turn
   * with none when the request sends tools.
   */
  readonly echoReasoningContent: boolean;
  /** Send `reasoning_split: true`, so reasoning arrives apart from the text (MiniMax M2). */
  readonly reasoningSplit: boolean;
  /** Send a `NamedTool` choice; when false it is left out. */
  readonly namedToolChoice: boolean;
  /** Send `parallel_tool_calls` when the request sets it. */
  readonly sendParallelToolCalls: boolean;
  /** Send `stream_options: {include_usage: true}`, so the stream reports usage. */
  readonly streamUsage: boolean;
  /**
   * Send a user message's author as `name`, on routes known to read it; other Chat Completions
   * servers may drop the field silently, so they get the author wrapper instead.
   */
  readonly messageName: boolean;
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
    // Off on every route: the wrapper attributes the same way everywhere, and some servers drop
    // `name` silently.
    messageName: false,
  });
}

// The conservative `name` pattern Chat Completions accepts; other authors go in the wrapper.
const NAME = /^[A-Za-z0-9_-]{1,64}$/;

/** What `OpenAIChat` is built from. */
export interface OpenAIChatOptions {
  /** The model name sent as `model` (the deployment name on Azure). */
  model: string;
  /** The route's provider name, for error messages and replay records. */
  provider: string;
  /** The route's request rules; defaults to `chatOptions(provider, model)`. */
  options?: ChatOptions;
  /** The provider block, merged into every body after yian's fields; copied. */
  block?: JsonObject | null;
  /**
   * Called once per assistant message, on a route that echoes reasoning, holding a reasoning
   * record that can't be used. The message is still sent, with `reasoning_content: ""` when the
   * request sends tools and nothing else is echoed.
   */
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
   * `parallel_tool_calls` as the route allows, then `response_format` when the request sets
   * `outputFormat`, then `reasoning_split` when the route needs it, then the provider block.
   *
   * A user message's author goes as `name` when the route sends it and the author matches
   * `^[A-Za-z0-9_-]{1,64}$`, and otherwise as the author wrapper folded into its first and last
   * texts; the author tag is escaped in user and tool text.
   *
   * On a route that echoes reasoning, an assistant message holding a reasoning record that
   * can't be used is sent without that reasoning, and `onReplayDegraded` is called once for it;
   * a throw from the callback propagates. When such a request sends tools, an assistant message
   * with no reasoning to echo is sent with `reasoning_content: ""` (tools added only by the
   * provider block don't count).
   *
   * @throws {TypeError} If a message holds a media part or a part its role cannot carry.
   */
  buildBody(request: Request): JsonObject {
    const body: JsonObject = { model: this.model, messages: this.#messages(request), stream: true };
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
    if (request.outputFormat !== null) body["response_format"] = responseFormat(request.outputFormat);
    // Set before the merge, so a block's own value wins.
    if (this.options.reasoningSplit) body["reasoning_split"] = true;
    if (this.#block) mergeRequestBody(body, this.#block);
    return body;
  }

  /**
   * Turn a streamed reply's chunks into stream events.
   *
   * Reasoning, text and tool calls stream as deltas; refusal pieces are collected. Part indexes
   * are dense, in the order each part gets its first non-empty piece (a call, its first entry);
   * inside one chunk the order is reasoning, text, refusal, calls. A call entry joins the call
   * at its wire `index` with its id, else the last call at that index unless it brings a
   * different id; without an index it joins the call with its id, else the last call. A call's
   * id and name are the first non-empty ones, carried by the delta that brings them. A chunk's
   * reasoning is its first non-empty source of `reasoning_content`, `reasoning` and the
   * `reasoning.text` items of `reasoning_details`; on a route that echoes reasoning an empty
   * `reasoning_content` opens the part too. A list `content` counts its text items and strings,
   * joined with newlines. The finish reason is the last non-empty one sent, and is held until
   * `[DONE]` or the end of the stream, so a usage chunk after it is read. Chunks without choices
   * contribute only usage, unknown keys are ignored, and empty or non-object event data is skipped.
   *
   * @param frames The reply's server-sent events, in order.
   * @returns The deltas, then one `PartDone` per part in index order (reasoning with its replay
   *   record; a call without an id named `yian_call_<n>`, `n` its position among the calls),
   *   then `UsageReport` (when any usage came; the last one wins) and `Finish`, whose reason is
   *   `tool_calls` when any call came, unless it is `length`.
   * @throws {ProviderError} For an error chunk (kind from its HTTP code, else its code or
   *   type), a failing finish reason (`refused` for a content filter, `overloaded` for
   *   `insufficient_system_resource`, `server` for `error` or `aborted`), a stream that ends
   *   with no finish reason (`incomplete`, retryable), a call with no name, or an event that is
   *   not JSON or holds a choice, delta, tool call or usage that cannot be read (`protocol`).
   *   The message holds the provider's text uncapped.
   */
  async *parse(frames: AsyncIterable<SSEEvent>): AsyncGenerator<Event> {
    const provider = this.provider;
    // The reply's parts by index: dense, in the order each is first seen.
    const parts: Assembly[] = [];
    // A reply has one choice, so at most one reasoning, text and refusal part.
    const append = (kind: TextAssembly["kind"], text: string, field: string | null = null): number => {
      const part = parts.find((each): each is TextAssembly => each.kind === kind);
      if (part === undefined) return parts.push({ kind, text, field }) - 1;
      part.text += text;
      return parts.indexOf(part);
    };
    // Return the call a tool-call entry joins, opening one when none fits.
    const callFor = (piece: CallPiece): CallAssembly => {
      const calls = parts.filter(isCall);
      let call: CallAssembly | undefined;
      if (piece.wireIndex !== null) {
        // Some vendors send two calls at the same index, told apart by their ids: an entry joins
        // the call there with its id, else the last call there unless both ids are set and differ.
        const atIndex = calls.filter((each) => each.wireIndex === piece.wireIndex);
        call = (piece.id === null ? undefined : atIndex.findLast((each) => each.id === piece.id)) ?? atIndex.at(-1);
        if (call?.id && piece.id && call.id !== piece.id) call = undefined;
      } else {
        call = piece.id === null ? calls.at(-1) : calls.findLast((each) => each.id === piece.id);
      }
      if (call === undefined) {
        call = { kind: "call", id: null, name: null, args: "", wireIndex: piece.wireIndex };
        parts.push(call);
      }
      return call;
    };
    let finishReason: string | null = null;
    let usage: Usage | null = null;
    let responseId: string | null = null;
    let responseModel: string | null = null;
    for await (const frame of frames) {
      // The openai SDK stops at a `[DONE]` sentinel; the stream is then over.
      if (frame.data.startsWith("[DONE]")) break;
      // Some gateways send keep-alive frames with empty data.
      if (frame.data === "") continue;
      const data = eventData(frame.data, provider);
      if (!isPlainObject(data)) continue;
      const error = data["error"];
      if (isPlainObject(error)) throw this.#streamFailure(error);
      if (typeof error === "string" && error) {
        throw new ProviderError("server", `${provider} stream failed (server): ${error}`);
      }
      // Azure's prompt filter chunk comes first with an empty id and model, so empty ones count as none.
      const id = data["id"];
      if (responseId === null && typeof id === "string" && id) responseId = id;
      const model = data["model"];
      if (responseModel === null && typeof model === "string" && model) responseModel = model;
      usage = usageOf(data["usage"], provider) ?? usage;
      const choice = choiceOf(data["choices"], provider);
      if (choice === null) continue;
      const reason = choice["finish_reason"];
      // An empty reason is no finish, as in langchain_openai, which reads it by truthiness.
      if (typeof reason === "string" && reason !== "") finishReason = reason;
      const delta = isPlainObject(choice["delta"]) ? choice["delta"] : {};
      checkDelta(delta, provider);
      const pieces = callPieces(delta["tool_calls"], provider);
      const reasoning = reasoningPiece(delta, this.options.echoReasoningContent);
      if (reasoning !== null) {
        const [text, field] = reasoning;
        const index = append("reasoning", text, field);
        if (text) yield new ReasoningDelta(index, text);
      }
      const text = contentPiece(delta["content"]);
      if (text) yield new TextDelta(append("text", text), text);
      const refusal = delta["refusal"];
      if (typeof refusal === "string" && refusal) append("refusal", refusal);
      for (const piece of pieces) {
        const call = callFor(piece);
        // A delta carries an id or name only when it is the call's first, which wins.
        const id = call.id === null ? piece.id : null;
        const name = call.name === null ? piece.name : null;
        call.id ??= id;
        call.name ??= name;
        call.args += piece.args;
        yield new ToolCallDelta({ index: parts.indexOf(call), id, name, arguments: piece.args });
      }
    }
    if (finishReason === null) {
      throw new ProviderError("incomplete", `${provider} stream ended before a complete response`, { retryable: true });
    }
    const failed = FAILED_FINISH.get(finishReason);
    if (failed !== undefined) {
      throw new ProviderError(failed, `${provider} response did not complete (${finishReason})`);
    }
    // Every part is built before the first PartDone, so a call with no name fails first.
    const taken = new Set<string>();
    for (const part of parts) if (isCall(part) && part.id !== null) taken.add(part.id);
    const done: Part[] = [];
    let callCount = 0;
    for (const part of parts) {
      if (isCall(part)) {
        done.push(callPart(part, callCount, taken, provider));
        callCount += 1;
      } else {
        done.push(this.#part(part));
      }
    }
    for (const [index, part] of done.entries()) yield new PartDone(index, part);
    if (usage !== null) yield new UsageReport(usage);
    const fields: JsonObject = {};
    if (responseId !== null) fields["response_id"] = responseId;
    // The model that answered, which may differ from the one asked for (an Azure deployment name).
    if (responseModel !== null) fields["response_model"] = responseModel;
    const providerPayload = replayRecord(this.#key, fields);
    // `length` says the calls may be cut short, so it wins; any other reason with calls is a tool call finish.
    const reason = finishReason !== "length" && parts.some(isCall) ? "tool_calls" : finishReason;
    yield new Finish({ reason, responseId, protocol: PROTOCOL, providerPayload });
  }

  /** Build a streamed part; a reasoning part carries a record naming its wire field. */
  #part(assembly: TextAssembly): Part {
    if (assembly.kind === "reasoning") {
      return new ReasoningPart(assembly.text, replayRecord(this.#key, { field: assembly.field }));
    }
    if (assembly.kind === "refusal") return new RefusalPart(assembly.text);
    return new TextPart(assembly.text);
  }

  /**
   * Return the error for an error chunk's error object; its kind from its HTTP code with the whole
   * error object classified, else from its code or type.
   */
  #streamFailure(error: JsonObject): ProviderError {
    const rawMessage = error["message"];
    const message = typeof rawMessage === "string" ? rawMessage : "";
    const code = error["code"];
    let kind: ErrorKind;
    if (typeof code === "bigint" && code >= 400n && code <= 599n) {
      // Classified as the HTTP path classifies the same body, so a content-filter code in it counts.
      kind = kindForStatus(Number(code), encodeBody({ error }, "stream error"));
    } else {
      // An error without a string code may still name its kind in `type` (`invalid_request_error`).
      const name = typeof code === "string" ? code : error["type"];
      kind = (typeof name === "string" ? CHUNK_ERROR_KINDS.get(name) : undefined) ?? "server";
    }
    return new ProviderError(kind, `${this.provider} stream failed (${kind}): ${message || "no message"}`);
  }

  /** Render the conversation as chat messages, in order. */
  #messages(request: Request): JsonObject[] {
    // DeepSeek refuses a request with tools unless every earlier assistant turn carries
    // `reasoning_content`; without tools it ignores the field.
    const requireReasoning = this.options.echoReasoningContent && request.tools.length > 0;
    const rendered: JsonObject[] = [];
    for (const message of request.messages) {
      const native =
        this.options.messageName && message.role === "user" && message.author !== null && NAME.test(message.author);
      // Folded inline, so a one-text message still goes as a string.
      const shown = withAuthorWrapper(message, { wrap: !native, inline: true });
      if (shown.role === "system" || shown.role === "user") {
        const item = inputMessage(shown);
        if (item !== null) {
          if (native) item["name"] = message.author;
          rendered.push(item);
        }
      } else if (shown.role === "assistant") {
        rendered.push(this.#assistant(shown, requireReasoning));
      } else {
        for (const part of shown.parts) {
          if (part instanceof MediaPart) throw new TypeError(NO_MEDIA);
          if (!(part instanceof ToolResultPart)) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
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
   * Refusals are not sent. `content` is `null` when the message has calls and no text. With
   * `requireReasoning`, a message with no reasoning to echo sends `reasoning_content: ""`, never
   * reasoning another route wrote.
   */
  #assistant(message: Message, requireReasoning: boolean): JsonObject {
    const texts: string[] = [];
    const validCalls: JsonObject[] = [];
    const invalidCalls: JsonObject[] = [];
    const reasoning: string[] = [];
    let reason: ReplayDegradedReason | null = null;
    for (const part of message.parts) {
      if (part instanceof MediaPart) throw new TypeError(NO_MEDIA);
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
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
    if (reasoning.length > 0 || requireReasoning) rendered["reasoning_content"] = reasoning.join("");
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

/**
 * Return a chunk's choice, or `null` when `choices` is absent, `null` or empty.
 *
 * @throws {ProviderError} `choices` is not a list, its first item is not an object, or that
 *   item's `finish_reason` is neither a string nor `null` (kind `protocol`).
 */
function choiceOf(choices: JsonValue | undefined, provider: string): JsonObject | null {
  if (choices === undefined || choices === null) return null;
  if (Array.isArray(choices) && choices.length === 0) return null;
  const choice = Array.isArray(choices) ? choices[0] : null;
  if (isPlainObject(choice)) {
    const reason = choice["finish_reason"];
    if (reason === undefined || reason === null || typeof reason === "string") return choice;
  }
  throw new ProviderError("protocol", `invalid choice in ${provider} stream event`);
}

/**
 * Check the types of a delta's text fields.
 *
 * @throws {ProviderError} `content` is not a string or a list, `reasoning_content`, `reasoning`
 *   or `refusal` is not a string, or `reasoning_details` is not a list, when present and not
 *   `null` (kind `protocol`).
 */
function checkDelta(delta: JsonObject, provider: string): void {
  const absent = (value: JsonValue | undefined) => value === undefined || value === null;
  const content = delta["content"];
  const details = delta["reasoning_details"];
  const strings = [delta["reasoning_content"], delta["reasoning"], delta["refusal"]];
  if (
    !(absent(content) || typeof content === "string" || Array.isArray(content)) ||
    !strings.every((value) => absent(value) || typeof value === "string") ||
    !(absent(details) || Array.isArray(details))
  ) {
    throw new ProviderError("protocol", `invalid delta in ${provider} stream event`);
  }
}

/** Return whether a streamed part is a tool call. */
function isCall(part: Assembly): part is CallAssembly {
  return part.kind === "call";
}

/**
 * Read a delta's `tool_calls` entries; none when it is absent or `null`.
 *
 * A wire `index` that is not an int counts as absent, as does an `id` or `function.name` that is
 * not a non-empty string; a missing or `null` `arguments` piece is empty.
 *
 * @throws {ProviderError} `tool_calls` is not a list, an entry is not an object, its `function`
 *   is not an object or its `arguments` not a string, when present and not `null` (kind
 *   `protocol`).
 */
function callPieces(toolCalls: JsonValue | undefined, provider: string): CallPiece[] {
  if (toolCalls === undefined || toolCalls === null) return [];
  const invalid = () => new ProviderError("protocol", `invalid tool call in ${provider} stream event`);
  if (!Array.isArray(toolCalls)) throw invalid();
  return toolCalls.map((entry) => {
    if (!isPlainObject(entry)) throw invalid();
    const fn = entry["function"] ?? {};
    if (!isPlainObject(fn)) throw invalid();
    const args = fn["arguments"] ?? "";
    if (typeof args !== "string") throw invalid();
    const index = entry["index"];
    const id = entry["id"];
    const name = fn["name"];
    return {
      wireIndex: typeof index === "bigint" ? index : null,
      id: typeof id === "string" && id ? id : null,
      name: typeof name === "string" && name ? name : null,
      args,
    };
  });
}

/**
 * Build a finished tool call, its arguments read by `parseArguments` (not repaired).
 *
 * A call without an id gets `yian_call_<n>`, with `_<k>` appended while that id is in `taken`;
 * the id it gets is added to `taken`.
 *
 * @param n The call's position among the reply's calls.
 * @param taken The ids in the reply so far: every real id, and those generated.
 * @throws {ProviderError} The call has no name (kind `protocol`).
 */
function callPart(call: CallAssembly, n: number, taken: Set<string>, provider: string): ToolCallPart {
  if (call.name === null) throw new ProviderError("protocol", `${provider} tool call ${n} has no name`);
  let id = call.id;
  if (id === null) {
    id = `yian_call_${n}`;
    for (let k = 1; taken.has(id); k++) id = `yian_call_${n}_${k}`;
    taken.add(id);
  }
  const [args, rawArguments] = parseArguments(call.args);
  return new ToolCallPart({ id, name: call.name, arguments: args, rawArguments });
}

/**
 * Return a delta's reasoning piece and its wire field, or `null` when it has none.
 *
 * The piece is the first non-empty of `reasoning_content`, `reasoning` and the joined `text` of
 * `reasoning_details` items of type `reasoning.text`. vLLM and OpenRouter send two sources with
 * the same text, which this counts once.
 *
 * @param echo Whether the route echoes reasoning, so an empty `reasoning_content` is a piece.
 */
function reasoningPiece(delta: JsonObject, echo: boolean): [string, string] | null {
  const details = delta["reasoning_details"];
  const texts: string[] = [];
  for (const item of Array.isArray(details) ? details : []) {
    if (isPlainObject(item) && item["type"] === "reasoning.text" && typeof item["text"] === "string") {
      texts.push(item["text"]);
    }
  }
  const sources: [string, JsonValue | undefined][] = [
    ["reasoning_content", delta["reasoning_content"]],
    ["reasoning", delta["reasoning"]],
    ["reasoning_details", texts.join("")],
  ];
  for (const [field, value] of sources) {
    if (typeof value === "string" && value) return [value, field];
  }
  // Opened even when empty, so the next turn echoes `reasoning_content` back.
  if (echo && typeof delta["reasoning_content"] === "string") return ["", ECHOED_FIELD];
  return null;
}

/** Return a delta's text: a string `content`, or a list's strings and `text` items joined with newlines. */
function contentPiece(content: JsonValue | undefined): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const texts: string[] = [];
  for (const item of content) {
    if (typeof item === "string") texts.push(item);
    else if (isPlainObject(item) && item["type"] === "text" && typeof item["text"] === "string")
      texts.push(item["text"]);
  }
  return texts.join("\n");
}

/**
 * Return the token counts of a Chat Completions `usage` object; `null` when it is not an object.
 *
 * Missing or `null` prompt and completion counts are 0, and a missing total is their sum.
 * Cache reads come from `prompt_tokens_details.cached_tokens`, else DeepSeek's
 * `prompt_cache_hit_tokens`. `null` details are empty, and the service tier is ignored.
 *
 * @throws {ProviderError} A count is not an int within `Number.MAX_SAFE_INTEGER`, or the
 *   details are not objects (kind `protocol`).
 */
function usageOf(usage: JsonValue | undefined, provider: string): Usage | null {
  if (!isPlainObject(usage)) return null;
  const promptDetails = usage["prompt_tokens_details"] ?? {};
  const completionDetails = usage["completion_tokens_details"] ?? {};
  if (!isPlainObject(promptDetails) || !isPlainObject(completionDetails)) {
    throw new ProviderError("protocol", `${provider} response usage details are not objects`);
  }
  const notInt = notIntMessage(provider);
  const inputTokens = tokenCount(usage["prompt_tokens"], notInt) ?? 0;
  const outputTokens = tokenCount(usage["completion_tokens"], notInt) ?? 0;
  try {
    return new Usage({
      inputTokens,
      outputTokens,
      totalTokens: tokenCount(usage["total_tokens"], notInt) ?? inputTokens + outputTokens,
      cacheReadTokens:
        tokenCount(promptDetails["cached_tokens"], notInt) ?? tokenCount(usage["prompt_cache_hit_tokens"], notInt),
      cacheCreationTokens: tokenCount(promptDetails["cache_write_tokens"], notInt),
      reasoningTokens: tokenCount(completionDetails["reasoning_tokens"], notInt),
    });
  } catch (error) {
    // `Usage` refuses a count past `Number.MAX_SAFE_INTEGER`, which `Number()` rounds.
    if (error instanceof MessageFormatError) throw new ProviderError("protocol", notInt);
    throw error;
  }
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
    if (!(part instanceof TextPart)) throw new TypeError(wrongPart(PROTOCOL, part, message.role));
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

/** Render an output format as a `response_format` entry, `strict` only when set. */
function responseFormat(format: OutputFormat): JsonObject {
  // Copied so changing a body cannot change the format it came from.
  const jsonSchema: JsonObject = { name: format.name, schema: copyJson(format.schema) };
  if (format.strict !== null) jsonSchema["strict"] = format.strict;
  return { type: "json_schema", json_schema: jsonSchema };
}
