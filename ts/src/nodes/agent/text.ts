/**
 * Visible-text extraction and the interactions learning reads, for the AGENT loop.
 *
 * Knows about:   llm messages, memory contracts.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import type { Message } from "../../llm/messages.ts";
import type { VisibleInteraction } from "../../memory/contracts.ts";

/**
 * Return only the model content represented as visible text.
 *
 * @param message A model reply, or text already extracted.
 * @returns A string as it is; for a message, its text parts joined with `"\n"` (reasoning,
 *   refusal and tool-call parts are left out), or `""` when it has none.
 */
export function textOf(message: Message | string): string {
  return typeof message === "string" ? message : message.text();
}

/**
 * Append `make()`'s interaction to `interactions`, unless its carrier refuses the fields (a
 * `TypeError` or `RangeError`): what learning reads is observational and never fails the pass.
 */
export function pushVisible(interactions: VisibleInteraction[], make: () => VisibleInteraction): void {
  try {
    interactions.push(make());
  } catch (error) {
    if (!(error instanceof TypeError || error instanceof RangeError)) throw error;
  }
}
