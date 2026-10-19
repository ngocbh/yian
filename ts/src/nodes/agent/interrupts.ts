/**
 * The signal a host interrupt raises out of a streamed AGENT model reply.
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
 * A streamed model reply cut off at a chunk boundary by a host interrupt.
 *
 * `reply` is an assistant message holding only the visible text streamed so far; a partial tool
 * call is never kept because it cannot run. `usage` is what the provider reported in the chunks
 * already received, or `null`.
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
