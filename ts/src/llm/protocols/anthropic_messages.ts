/**
 * The Anthropic Messages wire format (`POST /v1/messages`).
 *
 * Messages are formatted as LangChain's Anthropic client formats them, so requests stay the
 * same: runs of user and tool messages merge into one `user` turn, leading system messages are
 * lifted to `system`, and a later system message stays in place only where the model accepts
 * it, else it is hoisted to `system` with a warning. An assistant message is sent as its blocks
 * in part order, so a replayed turn goes back as the API sent it; its thinking goes back with its
 * signature only to the provider, model and account that wrote it. Tool ids the API refuses are
 * hashed to ones it accepts, the same on the call and its result. Tools are sent with their
 * schema as `input_schema`, and a tool choice only with tools. Every call streams.
 */

import { createHash } from "node:crypto";

import type { JsonObject, JsonValue } from "../../typesys/json.ts";
import { mergeRequestBody } from "../body.ts";
import { ProviderError } from "../errors.ts";
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
  ReasoningPart,
  readReplayRecord,
  type ReplayDegradedReason,
  type ReplayKey,
  type Request,
  TextPart,
  type ToolChoice,
  ToolCallPart,
  ToolResultPart,
  type ToolSpec,
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
// The `tool_choice` type for each choice yian names by string.
const CHOICE_TYPES = { auto: "auto", required: "any", none: "none" } as const;
// The tool ids the API accepts on `tool_use` and `tool_result`.
const TOOL_ID = /^[a-zA-Z0-9_-]+$/;
// The fields of a reasoning part's replay record; it holds exactly one.
const REASONING_FIELDS = ["signature", "redacted"];
// The required-signature error's text after its source.
const REQUIRED_SIGNATURE =
  "anthropic needs its signed thinking to continue a tool call with thinking on; finish the tool turn on the " +
  "original model or turn thinking off in the anthropic block";

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
   * `system` when there is one, `tools` and `tool_choice` when tools are sent, and
   * `stream: true`, then the provider block, whose keys replace yian's.
   *
   * A tool is its name, its parameters as `input_schema`, its description (always) and `strict`
   * when set. The tool choice is sent only with tools; `parallelToolCalls`, when set, becomes
   * `disable_parallel_tool_use` on it (an `auto` choice when none is set), but never on `none`.
   *
   * Runs of user and tool messages merge into one `user` turn, in which whitespace-only text
   * blocks are dropped. Leading system messages are lifted to `system`. A later one stays in
   * place when the model accepts that, it follows a user turn and an assistant turn or the end
   * comes next; otherwise it is hoisted to `system`, with LangChain's warning through
   * `process.emitWarning`. An assistant message is sent as its blocks in part order: thinking,
   * redacted thinking, text and `tool_use` (an invalid call with empty `input`), one lone text
   * as a string; refusals are not sent. Its reasoning goes back from its replay records, only
   * with a text or call, and is all left out when any reasoning part has no record or one that
   * can't be used, `onReplayDegraded` being called once for the latter (a throw from the callback
   * propagates). The final assistant message is stripped at the end, and an empty assistant
   * message is skipped unless it is the final one.
   *
   * @throws {ProviderError} A `bad_request` when thinking is `enabled`, the request continues a
   *   tool loop, and the loop's first assistant message had its reasoning left out for an
   *   unusable record: the API requires that turn's signed thinking.
   * @throws {TypeError} If a message holds a media part or a part its role cannot carry, a
   *   system message must be hoisted when `system` is already set, or a tool's parameters have
   *   a top-level `oneOf` or `anyOf`.
   */
  buildBody(request: Request): JsonObject {
    const [system, messages, degraded] = this.#messages(request.messages);
    const body: JsonObject = {
      model: this.model,
      max_tokens: MAX_TOKENS.get(this.model) ?? DEFAULT_MAX_TOKENS,
      messages,
    };
    if (system !== null) body["system"] = system;
    if (request.tools.length > 0) {
      body["tools"] = request.tools.map(tool);
      const choice = toolChoice(request.toolChoice, request.parallelToolCalls);
      if (choice !== null) body["tool_choice"] = choice;
    }
    body["stream"] = true;
    if (this.#block) mergeRequestBody(body, this.#block);
    // Under adaptive thinking a turn may start without thinking, so only `enabled` needs it.
    const thinking = body["thinking"];
    if (isPlainObject(thinking) && thinking["type"] === "enabled") {
      const source = degradedTurnStart(messages, degraded);
      if (source !== null) {
        throw new ProviderError("bad_request", `the last assistant turn came from ${source}; ${REQUIRED_SIGNATURE}`, {
          provider: this.provider,
        });
      }
    }
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

  /**
   * Return the `system` field (or `null`) and the rendered turns, as LangChain formats them, and
   * for each rendered assistant turn whose reasoning was left out for an unusable record, where
   * that turn came from.
   */
  #messages(messages: readonly Message[]): [JsonValue | null, JsonObject[], Map<JsonObject, string>] {
    if (messages.some((message) => message.parts.some((part) => part instanceof MediaPart))) {
      throw new TypeError(NO_MEDIA);
    }
    const turns = merge(messages);
    const lastNonSystem = turns.findLastIndex((turn) => turn.role !== "system");
    const inPlace = IN_PLACE_SYSTEM_PREFIXES.some((prefix) => this.model.startsWith(prefix));
    let system: JsonValue | null = null;
    const rendered: JsonObject[] = [];
    const degraded = new Map<JsonObject, string>();
    // Later system messages held until the next rendered turn shows whether they stay in place.
    let pending: JsonValue[] = [];
    for (const [i, turn] of turns.entries()) {
      if (turn.role === "system") {
        const content = systemContent(turn.texts);
        if (i === 0) system = content;
        else if (inPlace && rendered.at(-1)?.["role"] === "user") pending.push(content);
        else system = this.#hoist(system, content);
        continue;
      }
      let content: JsonValue;
      let source: string | null = null;
      if (turn.role === "assistant") {
        const [reasoning, unusable] = this.#reasoning(turn.message);
        content = assistantContent(turn.message, i === lastNonSystem, reasoning);
        // The API refuses empty content on every message but the final assistant one.
        if (content === "" && i !== lastNonSystem) continue;
        for (const each of pending) rendered.push({ role: "system", content: each });
        if (unusable !== null) source = sourceOf(turn.message, unusable);
      } else {
        content = userContent(turn.blocks);
        for (const each of pending) system = this.#hoist(system, each);
      }
      pending = [];
      const rendering: JsonObject = { role: turn.role, content };
      if (source !== null) degraded.set(rendering, source);
      rendered.push(rendering);
    }
    for (const each of pending) rendered.push({ role: "system", content: each });
    return [system, rendered, degraded];
  }

  /**
   * Read an assistant message's reasoning records, calling `onReplayDegraded` once with the first
   * reason when one can't be used.
   *
   * All of the message's reasoning is left out when any reasoning part has no record or an
   * unusable one: with preserved thinking a signature binds every block before it, so sending
   * part of a turn's reasoning risks a refusal.
   *
   * @returns Each part's `thinking` or `redacted_thinking` block, `null` for a part that isn't
   *   reasoning, all `null` when the reasoning is left out; and the first unusable record, or `null`.
   * @throws {TypeError} If the message holds a tool result.
   */
  #reasoning(message: Message): [(JsonObject | null)[], JsonObject | null] {
    const blocks: (JsonObject | null)[] = [];
    let reason: ReplayDegradedReason | null = null;
    let unusable: JsonObject | null = null;
    let unsigned = false;
    for (const part of message.parts) {
      // Refused here so a bad message never fires onReplayDegraded first.
      if (part instanceof ToolResultPart) throw new TypeError(wrongPart(part, message.role));
      if (!(part instanceof ReasoningPart)) {
        blocks.push(null);
        continue;
      }
      if (part.providerPayload === null) {
        unsigned = true;
        blocks.push(null);
        continue;
      }
      const read = reasoningBlock(part, this.#key);
      if (!read.ok) {
        reason ??= read.reason;
        unusable ??= part.providerPayload;
      }
      blocks.push(read.ok ? read.block : null);
    }
    if (reason !== null) this.#onReplayDegraded?.(reason);
    // A part without a record is no record mismatch, so nothing is signalled for it.
    return reason !== null || unsigned ? [message.parts.map(() => null), unusable] : [blocks, null];
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

/** Return a `text` block. */
function textBlock(text: string): JsonObject {
  return { type: "text", text };
}

/** Return a system turn's content: one text as a string, several as text blocks. */
function systemContent(texts: string[]): JsonValue {
  return texts.length === 1 ? texts[0]! : texts.map(textBlock);
}

/** Return a user turn's content: one text block as a string, else the blocks without whitespace-only texts. */
function userContent(blocks: JsonObject[]): JsonValue {
  return loneText(blocks) ?? blocks.filter(isSendable);
}

/**
 * Return an assistant message's content: one lone text block as a string, else its blocks in
 * part order without whitespace-only texts; `""` when it has no text or call.
 *
 * Each text part is its own block and an empty one gives none; a call is a `tool_use` block, an
 * invalid one with empty `input`; a reasoning part is its block in `reasoning`, if any.
 * Refusals are not sent.
 *
 * @param final Whether this is the last non-system message, whose content is stripped at the end.
 * @param reasoning Each part's reasoning block, or `null`.
 */
function assistantContent(message: Message, final: boolean, reasoning: readonly (JsonObject | null)[]): JsonValue {
  const blocks: JsonObject[] = [];
  for (const [index, part] of message.parts.entries()) {
    const thinking = reasoning[index] ?? null;
    if (part instanceof TextPart) {
      if (part.text !== "") blocks.push(textBlock(part.text));
    } else if (part instanceof ToolCallPart) {
      // Copied so changing a body cannot change the message it came from.
      const input = part.arguments === null ? {} : copyJson(part.arguments);
      blocks.push({ type: "tool_use", id: toolId(part.id), name: part.name, input });
    } else if (thinking !== null) {
      blocks.push(thinking);
    }
  }
  const lone = loneText(blocks);
  if (lone !== null) return final ? lone.trimEnd() : lone;
  // Claude often streams a whitespace-only text block before a `tool_use`, which the API refuses back.
  const content = blocks.filter(isSendable);
  // Reasoning goes back only with the text or call it led to.
  if (!content.some((block) => block["type"] === "text" || block["type"] === "tool_use")) return "";
  const last = content.at(-1);
  if (final && last?.["type"] === "text") last["text"] = (last["text"] as string).trimEnd();
  return content;
}

/**
 * Read a reasoning part's replay record and return the block it replays as: `thinking` with its
 * `signature`, or `redacted_thinking` with its `redacted` data.
 *
 * A record without exactly one of them as a non-empty string is `malformed`.
 */
function reasoningBlock(
  part: ReasoningPart,
  key: ReplayKey,
): { ok: true; block: JsonObject } | { ok: false; reason: ReplayDegradedReason } {
  const read = readReplayRecord(part.providerPayload, key, REASONING_FIELDS);
  if (!read.ok) return read;
  const { signature, redacted } = read.fields;
  if (Object.keys(read.fields).length === 1) {
    if (typeof signature === "string" && signature !== "") {
      return { ok: true, block: { type: "thinking", thinking: part.text, signature } };
    }
    if (typeof redacted === "string" && redacted !== "") {
      return { ok: true, block: { type: "redacted_thinking", data: redacted } };
    }
  }
  return { ok: false, reason: "malformed" };
}

/**
 * Return where an assistant message came from, `<provider>/<model>`: from its message record,
 * else from `record` (its first unusable reasoning record); `an unknown model` when neither
 * names both.
 */
function sourceOf(message: Message, record: JsonObject): string {
  for (const payload of [message.providerPayload, record]) {
    const provider = payload?.["provider"];
    const model = payload?.["model"];
    if (typeof provider === "string" && typeof model === "string") return `${provider}/${model}`;
  }
  return "an unknown model";
}

/**
 * Return where the current tool loop's first assistant turn came from when its reasoning was
 * left out for an unusable record, else `null`; `null` too when the request doesn't continue a
 * tool loop (its last user turn holds no `tool_result`).
 *
 * The API checks the signed thinking at the start of the assistant turn, which spans every step
 * of a tool loop: without interleaved thinking the later steps rightly have none. A later step
 * that sends its own thinking was taken by the API as a new turn start, so nothing is missing.
 *
 * @param degraded Where each such rendered assistant turn came from.
 */
function degradedTurnStart(rendered: JsonObject[], degraded: Map<JsonObject, string>): string | null {
  const isUser = (turn: JsonObject) => turn["role"] === "user";
  const last = rendered.findLast(isUser);
  if (last === undefined || !hasBlock(last, "tool_result")) return null;
  const start = rendered.findLastIndex((turn) => isUser(turn) && !hasBlock(turn, "tool_result"));
  const [first, ...later] = rendered.slice(start + 1).filter((turn) => turn["role"] === "assistant");
  const thinks = (turn: JsonObject) => hasBlock(turn, "thinking", "redacted_thinking");
  return first === undefined || later.some(thinks) ? null : (degraded.get(first) ?? null);
}

/** Return whether a rendered turn's content is a list holding a block of one of `types`. */
function hasBlock(turn: JsonObject, ...types: string[]): boolean {
  const content = turn["content"];
  return Array.isArray(content) && content.some((block) => types.includes((block as JsonObject)["type"] as string));
}

/** Return the text of `blocks` when they are one text block, else `null`. */
function loneText(blocks: JsonObject[]): string | null {
  const [first] = blocks;
  return blocks.length === 1 && first!["type"] === "text" ? (first!["text"] as string) : null;
}

/** Return whether a block may go in a list: anything but a whitespace-only text block. */
function isSendable(block: JsonObject): boolean {
  return block["type"] !== "text" || (block["text"] as string).trim() !== "";
}

/**
 * Render a tool: its name, its parameters (copied) as `input_schema`, its description and
 * `strict` when set.
 *
 * @throws {TypeError} If its parameters have a top-level `oneOf` or `anyOf`.
 */
function tool(spec: ToolSpec): JsonObject {
  const combinators = ["oneOf", "anyOf"].filter((key) => Object.hasOwn(spec.parameters, key));
  if (combinators.length > 0) {
    // Python's repr quotes (tool names the API accepts hold no quote or escape).
    throw new TypeError(
      `tool '${spec.name}' has a top-level ${combinators.join("/")} in its parameters, which the Anthropic API ` +
        "does not support",
    );
  }
  // Copied so changing a body cannot change the spec it came from.
  const rendered: JsonObject = {
    name: spec.name,
    input_schema: copyJson(spec.parameters),
    description: spec.description,
  };
  if (spec.strict !== null) rendered["strict"] = spec.strict;
  return rendered;
}

/**
 * Render the tool choice, or `null` for none. When `parallel` is set, `disable_parallel_tool_use`
 * is its negation, on an `auto` choice when none is set; never on `none`, which has no such field.
 */
function toolChoice(choice: ToolChoice, parallel: boolean | null): JsonObject | null {
  let rendered: JsonObject | null = null;
  if (choice instanceof NamedTool) rendered = { type: "tool", name: choice.name };
  else if (choice !== null) rendered = { type: CHOICE_TYPES[choice] };
  if (parallel !== null && choice !== "none") {
    rendered ??= { type: "auto" };
    rendered["disable_parallel_tool_use"] = !parallel;
  }
  return rendered;
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
