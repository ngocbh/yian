/**
 * The signal a host interrupt raises out of a streamed AGENT model reply or its retry wait, and
 * the result of an AGENT invocation that honored one.
 *
 * A leaf module, so the stream loop and the effect recorder share it without importing each
 * other.
 *
 * Knows about:   events, llm messages.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import type { TokenUsage } from "../../events.ts";
import type { Message } from "../../llm/messages.ts";

/**
 * A streamed model reply cut off by a host interrupt at a chunk boundary or before it started, or
 * a model call whose retry wait a host interrupt ended.
 *
 * `reply` is an assistant message holding only the visible text streamed so far (none for a cut
 * wait or a reply cut before it started, reasoning included); a partial tool call is never kept
 * because it cannot run. `usage` is what the provider reported in the chunks already received, or
 * `null` (always for a cut wait).
 */
export class ReplyInterrupted extends Error {
  override name = "ReplyInterrupted";
  /** The visible text streamed before the interrupt, as an assistant message. */
  readonly reply: Message;
  /** The token usage reported so far, or `null`. */
  readonly usage: TokenUsage | null;

  constructor(reply: Message, usage: TokenUsage | null = null) {
    super("model reply interrupted by the host");
    this.reply = reply;
    this.usage = usage;
  }
}

/**
 * An AGENT invocation ended by a host interrupt it honored: `by` is the seq of the `interrupted`
 * record (its `interrupted {by}` step names it). The agent's output is null, which its declared
 * output decides.
 */
export class Interrupted {
  readonly by: number;

  constructor(by: number) {
    this.by = by;
  }
}
