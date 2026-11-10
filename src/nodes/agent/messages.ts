/**
 * The built-in `Message` type as a yian value; its conversion to and from the model message
 * (`llm/messages.ts`) an AGENT sends and receives; and the conversation check a request needs.
 *
 * A `Message` value holds what a `yian.message.v1` message holds, without `format`, with
 * snake_case keys, an omitted or `null` optional field, and media only as a run-folder file
 * (`FileRef`). Its role is `user`, `assistant` or `tool`: a system prompt is the agent's own.
 *
 * An AGENT's new messages (`messageSteps`, `rebuildMessages`) are rebuilt from its recorded steps,
 * so the reducer can rebuild them on every append and replay. The steps fold into exchanges by one
 * set of rules (`foldSteps`), which the loop's replay and the request rebuild read too.
 *
 * Knows about: llm (messages), diagnostics, typesys.
 * Never imports: compose, compile, runtime, the run boundary or the CLI.
 */

import { reprText } from "../../diagnostics.ts";
import {
  MediaPart,
  MESSAGE_FORMAT,
  MessageFormatError,
  Message as ModelMessage,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from "../../llm/messages.ts";
import type { JsonObject } from "../../typesys/json.ts";
import { builtinType, newMessagesType } from "../../typesys/types.ts";
import {
  buildValueAs,
  FileRef,
  TypeCheckError,
  type TypedValue,
  type Value,
  type ValueRecord,
  writeAsJson,
  writeAsText,
} from "../../typesys/values.ts";

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
  const { parts, ...rest } = value;
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

/** Return `value` as a `Message` when it is one, else `null`. */
function asMessage(value: unknown): Message | null {
  try {
    return checkMessage(value);
  } catch (error) {
    if (error instanceof TypeCheckError) return null;
    throw error;
  }
}

/**
 * Return the text parts a steer `Message` keeps, each of its parts as a user message carries it:
 * text as is, a reasoning or refusal part's text, and any other part but media in its text form.
 * Media is dropped (no route sends it yet), each drop told to `onDrop`.
 */
function steerParts(message: Message, onDrop: (part: number) => void): TextPart[] {
  const parts: TextPart[] = [];
  message.parts.forEach((part, index) => {
    if (part.type === "media") onDrop(index);
    else if (part.type === "text" || part.type === "reasoning" || part.type === "refusal") parts.push(new TextPart(part.text));
    else {
      const { provider_payload: _payload, ...rest } = part;
      parts.push(new TextPart(writeAsText(rest as Value)));
    }
  });
  return parts;
}

/**
 * Return the events a `steer` step read, which the engine attaches as `values` to each copy of the
 * step it hands out; the run log itself holds only their seqs.
 *
 * @throws RangeError when the step has no `values`: it didn't come from the engine.
 */
export function steerValues(step: Readonly<Record<string, unknown>>): readonly unknown[] {
  const values = step["values"];
  if (!Array.isArray(values)) throw new RangeError("a steer step without its values, which only the engine's copy of the step carries");
  return values;
}

/** Renders one steer event to text: an AGENT's `render:` template over `${event}`. */
export type SteerRender = (value: unknown) => string;

/** How `steerMessages` renders a read, and who hears what it drops or falls back on. */
export interface SteerMessageOptions {
  /** The event renderer; `null` or omitted for the default rendering. */
  readonly render?: SteerRender | null;
  /** Hears each media part a `Message` event drops: the event's and the part's index. */
  readonly onDrop?: (event: number, part: number) => void;
  /** Hears each event whose `render` threw, which got its default rendering instead. */
  readonly onFallback?: (event: number, error: unknown) => void;
}

/** The text parts an event's default rendering gives inside one message. */
function defaultParts(value: unknown, onDrop: (part: number) => void): TextPart[] {
  const message = typeof value === "string" ? null : asMessage(value);
  if (message === null) return [new TextPart(writeAsText(value as Value))];
  const parts = steerParts(message, onDrop);
  return parts.length > 0 ? parts : [new TextPart(writeAsText(value as Value))];
}

/**
 * Return the model messages one steer read sends. Without a render: each `Message` event (a value
 * `checkMessage` accepts) as a user message keeping its author, parts a user message can't carry
 * turned into text and media dropped (`onDrop(event, part)` hears each drop); each run of other
 * events as one user message with a text part per event (a `str` as is, anything else in its text
 * form, `writeAsText`). A `Message` left with no part holds its event's text form. With a render,
 * the read is one user message with one text part per event, `render(event)`, `Message` events
 * included; an event whose render throws gets its default rendering's parts there instead
 * (`onFallback(event, error)` hears it). Never throws: the events are already consumed when it
 * runs.
 */
export function steerMessages(values: readonly unknown[], options: SteerMessageOptions = {}): ModelMessage[] {
  const { render = null, onDrop = () => {}, onFallback = () => {} } = options;
  if (render !== null) {
    const parts = values.flatMap((value, event) => {
      try {
        return [new TextPart(render(value))];
      } catch (error) {
        onFallback(event, error);
        return defaultParts(value, (part) => onDrop(event, part));
      }
    });
    return parts.length > 0 ? [new ModelMessage({ role: "user", parts })] : [];
  }
  const messages: ModelMessage[] = [];
  let run: TextPart[] = [];
  const flush = (): void => {
    if (run.length > 0) messages.push(new ModelMessage({ role: "user", parts: run }));
    run = [];
  };
  values.forEach((value, event) => {
    const message = typeof value === "string" ? null : asMessage(value);
    if (message === null) {
      run.push(new TextPart(writeAsText(value as Value)));
      return;
    }
    flush();
    // An empty author can't be sent; the message goes without it.
    const parts = defaultParts(value, (part) => onDrop(event, part));
    messages.push(new ModelMessage({ role: "user", parts, author: message.author || null }));
  });
  flush();
  return messages;
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

/** What `checkModelConversation` reads of a message; a model message (`llm/messages.ts`) fits it. */
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
 * Check that `messages`, `Message` values (as `checkMessage` returns them) in conversation order,
 * are a conversation a model route can send: each converts to a model message, and the list
 * passes `checkModelConversation`.
 *
 * @throws ConversationError naming the first offending message: one the model message refuses
 *   (its reason is the `MessageFormatError` text, e.g. an empty `author`), or one that breaks a
 *   conversation rule.
 */
export function checkConversation(messages: readonly Message[]): void {
  const models = messages.map((message, index) => {
    try {
      return toModelMessage(message);
    } catch (error) {
      if (error instanceof MessageFormatError) throw new ConversationError(index, error.message);
      throw error;
    }
  });
  checkModelConversation(models);
}

/**
 * Check that `messages`, a conversation after the system message in its model-message form, is one
 * a model route can send.
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
export function checkModelConversation(messages: readonly ConversationMessage[]): void {
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

// --- An AGENT's new messages ------------------------------------------------------------------ //

/**
 * The message that ends an interrupted AGENT's new messages: the user's act, so the next turn's
 * history shows why the reply stopped.
 */
export const INTERRUPTED_MARKER = "[interrupted by the user]";

/** A recorded AGENT step and its `n`, its place among the node's steps (numbered from 1). */
export interface NumberedStep<S = ValueRecord> {
  readonly n: number;
  readonly step: S;
}

/** One logical model call of an AGENT, with the steers read before it and the results after it (`foldSteps`). */
export interface StepExchange<S = ValueRecord> {
  /**
   * The `steer` steps read before the call, oldest first. A call can follow several: a resume
   * replays a steer recorded before an unmade call, then reads live and records another before it.
   */
  readonly steers: readonly NumberedStep<S>[];
  /** The call's `model_call` steps: its failed attempts that were retried, then its reply or its last failure. */
  readonly attempts: readonly NumberedStep<S>[];
  /** The `tool_result` steps written after the call, until the next call ends, by call id. */
  readonly results: ReadonlyMap<string, NumberedStep<S>>;
}

/** An AGENT's steps as `foldSteps` groups them. */
export interface FoldedSteps<S = ValueRecord> {
  /** The logical calls, in order. */
  readonly exchanges: readonly StepExchange<S>[];
  /**
   * What follows the last logical call: the steers read for the next call and its retried failed
   * attempts so far, the start of a call not over yet. Its results are always empty.
   */
  readonly open: StepExchange<S>;
  /** The `interrupted` step, the honor of a host interrupt, or `null`. */
  readonly honored: NumberedStep<S> | null;
}

/**
 * Fold an AGENT's steps, numbered from 1 in order, into exchanges: each successful `model_call`,
 * and each failed one that wasn't to be retried, ends one logical call, which the failed attempts
 * before it belong to; each `tool_result` is filed under the call before it (one before any call
 * answers nothing), each `steer` under the call after it. Steps of any other kind are skipped. The
 * AGENT loop replays its calls from this fold; its requests and new messages rebuild from it.
 */
export function foldSteps<S extends Readonly<Record<string, unknown>>>(steps: readonly S[]): FoldedSteps<S> {
  const exchanges: { steers: NumberedStep<S>[]; attempts: NumberedStep<S>[]; results: Map<string, NumberedStep<S>> }[] = [];
  let steers: NumberedStep<S>[] = [];
  let attempts: NumberedStep<S>[] = [];
  let honored: NumberedStep<S> | null = null;
  for (const [index, step] of steps.entries()) {
    const numbered = { n: index + 1, step };
    if (step["kind"] === "model_call") {
      attempts.push(numbered);
      // A reply, or a failure with no retry after it, ends the call.
      if (!Object.hasOwn(step, "error") || step["retry_in"] == null) {
        exchanges.push({ steers, attempts, results: new Map() });
        steers = [];
        attempts = [];
      }
    } else if (step["kind"] === "tool_result") {
      // Call ids repeat across replies (an adapter may number each reply's calls from 0), so a
      // result is filed under the call before it; a later start's closer still follows it.
      exchanges.at(-1)?.results.set(step["call_id"] as string, numbered);
    } else if (step["kind"] === "steer") {
      // A steer is read only before a logical call, never between its attempts, so it belongs
      // to the next call.
      steers.push(numbered);
    } else if (step["kind"] === "interrupted") {
      honored = numbered;
    }
  }
  return { exchanges, open: { steers, attempts, results: new Map() }, honored };
}

/** Whether a `model_call` step's `purpose` is a structured-output call's. */
function isStructured(purpose: unknown): boolean {
  return purpose === "structured_native" || purpose === "structured_fallback";
}

/**
 * Return the `n`s of the AGENT steps its new messages are made of: each chat `model_call` whose
 * reply arrived (a cut one included), every `tool_result`, every `steer` read, the honor of a
 * host interrupt (`interrupted`), and the accepted structured reply (the last `model_call`, when
 * it is a structured call with no error that wasn't cut), in order. Failed attempts, earlier
 * structured calls and steps of any other kind are left out.
 *
 * @param steps The agent's recorded steps across its starts; step `n` is `steps[n - 1]`.
 */
export function messageSteps(steps: readonly Readonly<JsonObject>[]): number[] {
  const last = steps.findLastIndex((step) => step["kind"] === "model_call");
  const picked: number[] = [];
  steps.forEach((step, index) => {
    const arrived = step["kind"] === "model_call" && !Object.hasOwn(step, "error");
    const chat = arrived && step["purpose"] === "chat";
    const accepted = arrived && index === last && isStructured(step["purpose"]) && step["finish"] !== "interrupted";
    const kept = step["kind"] === "tool_result" || step["kind"] === "steer" || step["kind"] === "interrupted";
    if (chat || accepted || kept) picked.push(index + 1);
  });
  return picked;
}

/**
 * Rebuild an AGENT's new messages from the steps `messageSteps` picked: each reply, then a `tool`
 * message per call it answered, in call order (results are recorded in finish order; `foldSteps`
 * files them under the call before them). A call with no result (its reply was cut, or it waits on
 * an answer) is left out, so every tool call has a result; so is media held inline, which only a
 * run without a folder keeps. A message left with no parts is left out. The accepted structured
 * call becomes one assistant message holding `output` as JSON text; its recorded reply (an
 * internal output-tool call, or the raw JSON) isn't shown. A `steer` step, which carries the
 * events it read as `values`, adds the user messages `steerMessages` makes of them where it was
 * read: after the previous reply's results, before the next reply. An `interrupted` step adds the
 * user message `INTERRUPTED_MARKER` after the cut reply (kept only when it holds text) and its
 * results.
 *
 * @param output The node's output value, read only for a structured call.
 * @param render The agent's steer render, which steer events render through as the live loop
 *   rendered them.
 * @returns The `list(Message)` value.
 * @throws RangeError when a structured call is given with no output value, or a steer step
 *   without its values (`steerValues`).
 */
export function rebuildMessages(
  steps: readonly Readonly<ValueRecord>[],
  output: TypedValue | null,
  render: SteerRender | null = null,
): TypedValue {
  const messages: Message[] = [];
  /** Add `reply` and a `tool` message per call `results` answers. */
  const flush = (reply: ModelMessage, results: StepExchange<Readonly<ValueRecord>>["results"]): void => {
    const kept = reply.parts.filter((part) =>
      part instanceof ToolCallPart ? results.has(part.id) : !(part instanceof MediaPart && part.file === null),
    );
    if (kept.length > 0) {
      const { author, protocol, providerPayload } = reply;
      messages.push(fromModelMessage(new ModelMessage({ role: reply.role, parts: kept, author, protocol, providerPayload })));
    }
    for (const part of kept) {
      if (!(part instanceof ToolCallPart)) continue;
      const { step } = results.get(part.id)!;
      const content = step["output"] as string;
      const answer = new ToolResultPart({ callId: part.id, name: part.name, content, isError: step["is_error"] === true });
      messages.push(fromModelMessage(new ModelMessage({ role: "tool", parts: [answer] })));
    }
  };
  const { exchanges, open, honored } = foldSteps(steps);
  for (const exchange of [...exchanges, open]) {
    for (const { step } of exchange.steers) {
      for (const message of steerMessages(steerValues(step), { render })) messages.push(fromModelMessage(message));
    }
    let reply: ModelMessage | null = null;
    for (const { step } of exchange.attempts) {
      // Only the call's last reply has results.
      if (reply !== null) flush(reply, new Map());
      reply = null;
      if (isStructured(step["purpose"])) {
        // The final value as the flow reads it, not the lowered reply that produced it.
        if (output === null) throw new RangeError("the accepted structured call of an AGENT needs its output value");
        const text = new TextPart(writeAsJson(output.toObject()));
        messages.push(fromModelMessage(new ModelMessage({ role: "assistant", parts: [text] })));
      } else {
        reply = ModelMessage.fromJSON(step["output"]);
      }
    }
    if (reply !== null) flush(reply, exchange.results);
  }
  // The honor is the node's last step: it follows the cut reply and its results.
  if (honored !== null) {
    messages.push(fromModelMessage(new ModelMessage({ role: "user", parts: [new TextPart(INTERRUPTED_MARKER)] })));
  }
  return buildValueAs(newMessagesType(), messages);
}
