/**
 * Visible-text extraction for the AGENT loop.
 *
 * The observational trace helpers that record visible model text and tool results are not
 * ported yet and come with `EvaluationTrace`.
 *
 * Knows about:   llm messages.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import type { Message } from "../../llm/messages.ts";

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
