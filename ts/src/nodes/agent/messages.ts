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
