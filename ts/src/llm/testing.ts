/**
 * Test helpers for model calls: `ScriptedModel` is a `ChatModel` that answers from a script,
 * for tests above the HTTP layer.
 */

import { logJson } from "../typesys/json.ts";
import {
  EventStream,
  Finish,
  PartDone,
  ReasoningDelta,
  TextDelta,
  ToolCallDelta,
  UsageReport,
  type ChatModel,
  type Event,
} from "./events.ts";
import { ReasoningPart, Response, TextPart, ToolCallPart, type Request } from "./messages.ts";

/** One scripted reply: a `Response`, the events to stream with errors at their positions, or an error. */
export type ScriptedReply = Response | readonly (Event | Error)[] | Error;

/**
 * Yield the events a provider would stream for `response`.
 *
 * A tool call's delta text is `logJson` text (sorted keys; ints written as digits), unlike
 * Python's `json.dumps`; `collect` is unaffected because the `PartDone` wins.
 */
function* responseEvents(response: Response): Generator<Event> {
  const parts = response.message.parts;
  for (let index = 0; index < parts.length; index++) {
    const part = parts[index]!;
    if (part instanceof TextPart) {
      yield new TextDelta(index, part.text);
    } else if (part instanceof ReasoningPart) {
      yield new ReasoningDelta(index, part.text);
    } else if (part instanceof ToolCallPart) {
      const args = part.arguments === null ? part.rawArguments! : logJson(part.arguments);
      yield new ToolCallDelta({ index, id: part.id, name: part.name, arguments: args });
    }
    yield new PartDone(index, part);
  }
  if (response.usage !== null) yield new UsageReport(response.usage);
  yield new Finish({
    reason: response.finishReason,
    responseId: response.responseId,
    protocol: response.message.protocol,
    providerPayload: response.message.providerPayload,
  });
}

/** Yield the reply's events, throwing a scripted error at its position or `signal.reason` once aborted. */
async function* replyEvents(reply: ScriptedReply, signal: AbortSignal | undefined): AsyncGenerator<Event> {
  const items = reply instanceof Response ? responseEvents(reply) : reply instanceof Error ? [reply] : [...reply];
  for (const item of items) {
    signal?.throwIfAborted();
    if (item instanceof Error) throw item;
    yield item;
  }
}

/**
 * A `ChatModel` that answers each request with the next scripted reply.
 *
 * A `Response` reply streams as events: per part a delta (text, reasoning, or the tool call's
 * arguments as `logJson` text, its raw text when the arguments didn't parse) then its
 * `PartDone`, or only the `PartDone` for parts with no delta kind (media, refusal), then a
 * `UsageReport` when there is usage, then a `Finish` carrying the finish reason, response id
 * and the message's protocol and provider payload, so `collect` of the events equals the
 * `Response`. A list reply streams as given, throwing any error in it at that position. An
 * error reply is thrown by the stream's first `next()`.
 */
export class ScriptedModel implements ChatModel {
  /** The provider name. */
  readonly provider: string;
  /** The model name. */
  readonly model: string;
  /** Every request received, in order. */
  readonly requests: Request[] = [];
  /** How many streams returned by `stream` have been closed. */
  closed = 0;
  #replies: ScriptedReply[];

  /**
   * @param replies The replies, one per request.
   * @param options `provider` and `model` name the model; both default to `scripted`.
   */
  constructor(replies: Iterable<ScriptedReply>, options: { provider?: string; model?: string } = {}) {
    this.#replies = Array.from(replies);
    this.provider = options.provider ?? "scripted";
    this.model = options.model ?? "scripted";
  }

  /**
   * Record `request` and answer it with the next reply's events.
   *
   * @param request The call's input; recorded.
   * @param options `signal` aborts the stream: the next `next()` throws its reason.
   * @returns The reply's events.
   * @throws {Error} No reply is left; the test's own error, so it is thrown at once.
   */
  stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    this.requests.push(request);
    const reply = this.#replies.shift();
    if (reply === undefined) throw new Error(`ScriptedModel has no reply left for request ${this.requests.length}`);
    return new EventStream(replyEvents(reply, options.signal), () => {
      this.closed++;
    });
  }
}
