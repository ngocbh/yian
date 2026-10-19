/**
 * The Anthropic Messages wire format (`POST /v1/messages`).
 *
 * Messages are formatted as LangChain's Anthropic client formats them, so requests stay the
 * same: runs of user and tool messages merge into one `user` turn, leading system messages are
 * lifted to `system`, and a later system message stays in place only where the model accepts
 * it, else it is hoisted to `system` with a warning. An assistant message is sent as its blocks
 * in part order, so a replayed turn goes back as the API sent it. Tool ids the API refuses are
 * hashed to ones it accepts, the same on the call and its result. Every call streams.
 */

import { createHash } from "node:crypto";

import type { JsonObject, JsonValue } from "../../typesys/json.ts";
import { mergeRequestBody } from "../body.ts";
import type { Event } from "../events.ts";
import type { SseProtocol } from "../executor.ts";
import type { SSEEvent } from "../framing.ts";
import {
  copyJson,
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
export const PROTOCOL = "anthropic_messages";
const NO_MEDIA = "the anthropic_messages route does not send media parts yet";
// The default `max_tokens` by exact model id, from langchain-anthropic 1.7.5's model profiles;
// any other model gets 4096.
const MAX_TOKENS: ReadonlyMap<string, bigint> = new Map([
  ...[
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
  ].map((model) => [model, 64000n] as const),
  ...[
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
  ].map((model) => [model, 128000n] as const),
]);
const DEFAULT_MAX_TOKENS = 4096n;
// Models that accept a system message in the middle of the conversation (LangChain's list).
const IN_PLACE_SYSTEM_PREFIXES = [
  "claude-fable-5",
  "claude-mythos-5",
  "claude-opus-4-8",
  "claude-opus-5",
  "claude-sonnet-5-5",
];
// The tool ids the API accepts on `tool_use` and `tool_result`.
const TOOL_ID = /^[a-zA-Z0-9_-]+$/;

/** A turn after merging: consecutive system texts, a user turn's blocks, or an assistant message. */
type Turn =
  | { role: "system"; texts: string[] }
  | { role: "user"; blocks: JsonObject[] }
  | { role: "assistant"; message: Message };

/** What `AnthropicMessages` is built from. */
export interface AnthropicMessagesOptions {
  /** The model name sent as `model`. */
  model: string;
  /** The route's provider name, for error messages and replay records. */
  provider: string;
  /** A non-secret account id, for replay records. */
  account?: string | null;
  /** The provider block, merged into every body after yian's fields; copied. */
  block?: JsonObject | null;
  /** Called once per assistant message holding a reasoning record that can't be used. */
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
}

/** The Anthropic Messages wire protocol for one model. */
export class AnthropicMessages implements SseProtocol {
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

  constructor(options: AnthropicMessagesOptions) {
    this.model = options.model;
    this.provider = options.provider;
    this.account = options.account ?? null;
    this.#block = options.block ? copyJson(options.block) : null;
    this.#key = { protocol: PROTOCOL, provider: this.provider, model: this.model, account: this.account };
    this.#onReplayDegraded = options.onReplayDegraded ?? null;
  }

  /**
   * Build the Messages request body: `model`, `max_tokens` (the model's default), `messages`,
   * `system` when there is one and `stream: true`, then the provider block.
   *
   * Runs of user and tool messages merge into one `user` turn, in which whitespace-only text
   * blocks are dropped. Leading system messages are lifted to `system`. A later one stays in
   * place when the model accepts that, it follows a user turn and an assistant turn or the end
   * comes next; otherwise it is hoisted to `system`, with LangChain's warning through
   * `process.emitWarning`. An assistant message is sent as its text and `tool_use` blocks in
   * part order (an invalid call with empty `input`), one lone text as a string; refusals and
   * reasoning are not sent. The final assistant message is stripped at the end, and an empty
   * assistant message is skipped unless it is the final one.
   *
   * @throws {TypeError} If a message holds a media part or a part its role cannot carry, or a
   *   system message must be hoisted when `system` is already set.
   */
  buildBody(request: Request): JsonObject {
    const [system, messages] = this.#messages(request.messages);
    const body: JsonObject = {
      model: this.model,
      max_tokens: MAX_TOKENS.get(this.model) ?? DEFAULT_MAX_TOKENS,
      messages,
    };
    if (system !== null) body["system"] = system;
    body["stream"] = true;
    if (this.#block) mergeRequestBody(body, this.#block);
    return body;
  }

  /**
   * Turn a streamed reply's events into stream events. Not implemented yet.
   *
   * @throws {Error} Always.
   */
  async *parse(_frames: AsyncIterable<SSEEvent>): AsyncGenerator<Event> {
    throw new Error("not implemented");
  }

  /** Return the `system` field (or `null`) and the rendered turns, as LangChain formats them. */
  #messages(messages: readonly Message[]): [JsonValue | null, JsonObject[]] {
    if (messages.some((message) => message.parts.some((part) => part instanceof MediaPart))) {
      throw new TypeError(NO_MEDIA);
    }
    const turns = merge(messages);
    const lastNonSystem = turns.findLastIndex((turn) => turn.role !== "system");
    const inPlace = IN_PLACE_SYSTEM_PREFIXES.some((prefix) => this.model.startsWith(prefix));
    let system: JsonValue | null = null;
    const rendered: JsonObject[] = [];
    // Later system messages held until the next rendered turn shows whether they stay in place.
    let pending: JsonValue[] = [];
    for (const [i, turn] of turns.entries()) {
      if (turn.role === "system") {
        const content = systemContent(turn.texts);
        if (i === 0) system = content;
        else if (inPlace && (pending.length > 0 || rendered.at(-1)?.["role"] === "user")) pending.push(content);
        else system = this.#hoist(system, content);
        continue;
      }
      const content =
        turn.role === "user" ? userContent(turn.blocks) : assistantContent(turn.message, i === lastNonSystem);
      // The API refuses empty content on every message but the final assistant one.
      if (turn.role === "assistant" && content === "" && i !== lastNonSystem) continue;
      if (turn.role === "assistant") {
        for (const each of pending) rendered.push({ role: "system", content: each });
      } else {
        for (const each of pending) system = this.#hoist(system, each);
      }
      pending = [];
      rendered.push({ role: turn.role, content });
    }
    for (const each of pending) rendered.push({ role: "system", content: each });
    return [system, rendered];
  }

  /** Return `content` as the new `system` field, warning that it was moved there. */
  #hoist(system: JsonValue | null, content: JsonValue): JsonValue {
    if (system !== null) throw new TypeError("Received multiple non-consecutive system messages.");
    // LangChain's text, with the model in Python repr quotes (model ids hold no quote or escape).
    process.emitWarning(
      "A non-leading `SystemMessage` was moved to the top-level `system` field and now applies to the " +
        "entire conversation. To send it in place, use a supported model and place it after a human or " +
        `tool message, either last or before an AI message (model: '${this.model}').`,
    );
    return content;
  }
}

/**
 * Merge the conversation into turns: consecutive system messages into one, runs of user and
 * tool messages into one user turn (a text block per user text part, a `tool_result` per tool
 * result, in order). A system, user or tool message without parts adds nothing.
 */
function merge(messages: readonly Message[]): Turn[] {
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role === "assistant") {
      turns.push({ role: "assistant", message });
      continue;
    }
    if (message.parts.length === 0) continue;
    const last = turns.at(-1);
    if (message.role === "system") {
      const texts = textsOf(message);
      if (last?.role === "system") last.texts.push(...texts);
      else turns.push({ role: "system", texts });
      continue;
    }
    const blocks = message.role === "user" ? textsOf(message).map(textBlock) : message.parts.map(toolResult);
    if (last?.role === "user") last.blocks.push(...blocks);
    else turns.push({ role: "user", blocks });
  }
  return turns;
}

/** Return a system or user message's texts. */
function textsOf(message: Message): string[] {
  return message.parts.map((part) => {
    if (!(part instanceof TextPart)) throw new TypeError(wrongPart(part, message.role));
    return part.text;
  });
}

/** Render a tool message's part as a `tool_result` block; `is_error` is always sent. */
function toolResult(part: Part): JsonObject {
  if (!(part instanceof ToolResultPart)) throw new TypeError(wrongPart(part, "tool"));
  return { type: "tool_result", content: part.content, tool_use_id: toolId(part.callId), is_error: part.isError };
}

function textBlock(text: string): JsonObject {
  return { type: "text", text };
}

/** Return a system turn's content: one text as a string, several as text blocks. */
function systemContent(texts: string[]): JsonValue {
  return texts.length === 1 ? texts[0]! : texts.map(textBlock);
}

/** Return a user turn's content: one text block as a string, else the blocks without whitespace-only texts. */
function userContent(blocks: JsonObject[]): JsonValue {
  const [first] = blocks;
  if (blocks.length === 1 && first!["type"] === "text") return first!["text"]!;
  return blocks.filter(isSendable);
}

/**
 * Return an assistant message's content: one lone text block as a string, else its blocks in
 * part order without whitespace-only texts; `""` when it has none.
 *
 * Each text part is its own block and an empty one gives none; a call is a `tool_use` block, an
 * invalid one with empty `input`. Reasoning and refusals are not sent.
 *
 * @param final Whether this is the last non-system message, whose content is stripped at the end.
 */
function assistantContent(message: Message, final: boolean): JsonValue {
  const blocks: JsonObject[] = [];
  for (const part of message.parts) {
    if (part instanceof ToolResultPart) throw new TypeError(wrongPart(part, message.role));
    if (part instanceof TextPart) {
      if (part.text !== "") blocks.push(textBlock(part.text));
    } else if (part instanceof ToolCallPart) {
      // Copied so changing a body cannot change the message it came from.
      const input = part.arguments === null ? {} : copyJson(part.arguments);
      blocks.push({ type: "tool_use", id: toolId(part.id), name: part.name, input });
    }
  }
  const [first] = blocks;
  if (blocks.length === 1 && first!["type"] === "text") {
    const text = first!["text"] as string;
    return final ? text.trimEnd() : text;
  }
  // Claude often streams a whitespace-only text block before a `tool_use`, which the API refuses back.
  const content = blocks.filter(isSendable);
  const last = content.at(-1);
  if (final && last?.["type"] === "text") last["text"] = (last["text"] as string).trimEnd();
  return content.length === 0 ? "" : content;
}

/** Return whether a block may go in a list: anything but a whitespace-only text block. */
function isSendable(block: JsonObject): boolean {
  return block["type"] !== "text" || (block["text"] as string).trim() !== "";
}

/** Return `id`, or when the API refuses it, `toolu_` and the first 24 hex digits of its SHA-256. */
function toolId(id: string): string {
  if (id === "" || TOOL_ID.test(id)) return id;
  return `toolu_${createHash("sha256").update(id).digest("hex").slice(0, 24)}`;
}

/** Return the error text for a part a message of `role` cannot carry. */
function wrongPart(part: Part, role: string): string {
  const article = role === "assistant" ? "an" : "a";
  return `the anthropic_messages route cannot send a ${part.constructor.name} in ${article} ${role} message`;
}
