/**
 * The built-in `Message` type as a yian value; its conversion to and from the model message
 * (`llm/messages.ts`) an AGENT sends and receives; and the conversation check a request needs.
 *
 * A `Message` value holds what a `yian.message.v1` message holds, without `format`, with
 * snake_case keys, an omitted or `null` optional field, and media only as a run-folder file
 * (`FileRef`). Its role is `user`, `assistant` or `tool`: a system prompt is the agent's own.
 *
 * Knows about: llm (messages), diagnostics, typesys.
 * Never imports: compose, compile, runtime, the run boundary or the CLI.
 */

import { reprText } from "../../diagnostics.ts";
import { MediaPart, MESSAGE_FORMAT, Message as ModelMessage } from "../../llm/messages.ts";
import type { JsonObject } from "../../typesys/json.ts";
import { builtinType } from "../../typesys/types.ts";
import { buildValueAs, FileRef } from "../../typesys/values.ts";

/** One part of a `Message`, told apart by `type`; `provider_payload` is opaque provider data, kept untouched. */
export type MessagePart =
  | { type: "text" | "reasoning" | "refusal"; text: string; provider_payload?: JsonObject | null }
  | { type: "tool_call"; id: string; name: string; arguments?: JsonObject | null; raw_arguments?: string | null; provider_payload?: JsonObject | null }
  | { type: "tool_result"; call_id: string; name: string; content: string; is_error?: boolean | null; provider_payload?: JsonObject | null }
  | { type: "media"; mime: string; file: FileRef; name?: string | null; provider_payload?: JsonObject | null };

/** A message as a yian value: the built-in `Message` type. */
export type Message = {
  role: "user" | "assistant" | "tool";
  parts: MessagePart[];
  author?: string | null;
  protocol?: string | null;
  provider_payload?: JsonObject | null;
};

/**
 * Return `value` as a `Message`: checked against the built-in type, each file's JSON form read as
 * a `FileRef` (as a value decoded with `TypedValue.fromWire` holds it).
 *
 * @throws RangeError (a `TypeCheckError`) naming the path of the first part that doesn't fit.
 */
export function checkMessage(value: unknown): Message {
  return buildValueAs(builtinType("Message")!, value).value as Message;
}

/** Return a file's `yian.message.v1` form: `uri`, then `mime` and `name` unless `null`. */
function fileJSON(file: FileRef): Record<string, string> {
  const data: Record<string, string> = { uri: file.uri };
  if (file.mime !== null) data["mime"] = file.mime;
  if (file.name !== null) data["name"] = file.name;
  return data;
}

/** Return `fields` without its `null` and absent entries, each `FileRef` in its JSON form. */
function present(fields: Readonly<Record<string, unknown>>): Record<string, unknown> {
  const data: Record<string, unknown> = {};
  for (const [key, field] of Object.entries(fields)) {
    if (field === null || field === undefined) continue;
    data[key] = FileRef.isFileRef(field) ? fileJSON(field) : field;
  }
  return data;
}

/**
 * Return the model message a `Message` value sends, read by the model message's strict
 * `yian.message.v1` reader.
 *
 * @throws MessageFormatError naming the path (e.g. `parts[0].arguments`) of a value the model message refuses.
 */
export function toModelMessage(value: Message): ModelMessage {
  const { author: _author, parts, ...rest } = value;
  return ModelMessage.fromJSON({ format: MESSAGE_FORMAT, ...present(rest), parts: parts.map((part) => present(part)) });
}

/**
 * Return a model message as a `Message` value.
 *
 * @throws TypeError for a system message, or a media part given inline or by URL rather than as a run-folder file.
 */
export function fromModelMessage(message: ModelMessage): Message {
  if (message.role === "system") throw new TypeError("a Message can't be a system message");
  const { format: _format, ...data } = message.toJSON();
  const parts = message.parts.map((part, index) => {
    const json = part.toJSON();
    if (!(part instanceof MediaPart)) return json;
    if (part.file === null) throw new TypeError(`parts[${index}]: a Message holds media only as a run-folder file`);
    return { ...json, file: new FileRef(part.file) };
  });
  return checkMessage({ ...data, parts });
}

/** A conversation no model route can send: the message at `index` and why. */
export class ConversationError extends RangeError {
  override name = "ConversationError";
  /** The index of the offending message in the list checked. */
  readonly index: number;
  /** What is wrong with it. */
  readonly reason: string;

  constructor(index: number, reason: string) {
    super(`message ${index}: ${reason}`);
    this.index = index;
    this.reason = reason;
  }
}

/** What `checkConversation` reads of a message; a model message (`llm/messages.ts`) fits it. */
export interface ConversationMessage {
  readonly role: string;
  readonly parts: readonly {
    readonly type: string;
    /** A tool call's id. */
    readonly id?: string;
    /** A tool call's tool name (a media part's file name is `string | null`). */
    readonly name?: string | null;
    /** The call a tool result answers. */
    readonly callId?: string;
  }[];
}

// The parts each role's messages may hold, as every route's adapter renders them.
const ROLE_PARTS: ReadonlyMap<string, ReadonlySet<string>> = new Map([
  ["user", new Set(["text", "media"])],
  ["assistant", new Set(["text", "reasoning", "refusal", "tool_call"])],
  ["tool", new Set(["tool_result"])],
]);

/**
 * Check that `messages`, a conversation after the system message, is one a model route can send.
 *
 * Each message has a part, and each part is one its role can carry: a user message holds text and
 * media; an assistant message text, reasoning, refusals and tool calls; a tool message tool results.
 * Every tool call an assistant message makes is answered by the tool messages that directly follow
 * it (its exchange), before the next user or assistant message or the end of the list. A tool result
 * answers the open call of its id in the current exchange. Ids may repeat across exchanges, as routes
 * that mint `yian_call_<n>` per reply make them; only one id twice in one reply is refused, since
 * its results couldn't be told apart. Messages of one role may follow each other.
 *
 * @throws ConversationError naming the first offending message: for an unanswered call, the
 *   assistant message that made it.
 */
export function checkConversation(messages: readonly ConversationMessage[]): void {
  const open = new Map<string, { name: string; index: number }>();
  const unanswered = (before: string): ConversationError => {
    const [id, call] = open.entries().next().value!;
    return new ConversationError(call.index, `tool call ${reprText(id)} (${call.name}) has no result before ${before}`);
  };
  messages.forEach((message, index) => {
    const allowed = ROLE_PARTS.get(message.role);
    if (allowed === undefined) {
      throw new ConversationError(index, "a system message can't be in a prompt list; use system_prompt:");
    }
    if (message.parts.length === 0) throw new ConversationError(index, "the message has no parts");
    for (const part of message.parts) {
      if (!allowed.has(part.type)) throw new ConversationError(index, `${message.role} messages can't hold a ${part.type} part`);
    }
    if (message.role === "tool") {
      for (const part of message.parts) {
        const id = part.callId!;
        if (!open.delete(id)) throw new ConversationError(index, `tool result ${reprText(id)} answers no open tool call`);
      }
      return;
    }
    // A new exchange: the previous one must be fully answered, so `open` holds only this reply's calls.
    if (open.size > 0) throw unanswered(`the next ${message.role} message`);
    for (const part of message.parts) {
      if (part.type !== "tool_call") continue;
      const id = part.id!;
      if (open.has(id)) throw new ConversationError(index, `tool call id ${reprText(id)} appears twice in one reply`);
      open.set(id, { name: part.name ?? "", index });
    }
  });
  if (open.size > 0) throw unanswered("the end of the list");
}
