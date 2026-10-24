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

import type { JsonObject } from "../../typesys/json.ts";
import { builtinType } from "../../typesys/types.ts";
import { buildValueAs, type FileRef } from "../../typesys/values.ts";

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
