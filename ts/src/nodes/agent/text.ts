/**
 * Visible-text extraction and observational trace helpers for the AGENT loop.
 *
 * Knows about:   llm messages, memory contracts.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

import type { Message } from "../../llm/messages.ts";
import {
  type EvaluationTrace,
  ModelTextInteraction,
  ModelValueInteraction,
  ToolCallInteraction,
  ToolResultInteraction,
} from "../../memory/contracts.ts";

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

/** Whether `error` is a carrier's refusal of its fields, which the trace helpers ignore. */
function isCarrierError(error: unknown): boolean {
  return error instanceof TypeError || error instanceof RangeError;
}

/** Record encodable visible model text without affecting node behavior. */
export function recordModelText(trace: EvaluationTrace | null | undefined, text: string): void {
  if (trace == null) return;
  let interaction: ModelTextInteraction;
  try {
    interaction = new ModelTextInteraction(text);
  } catch (error) {
    if (isCarrierError(error)) return;
    throw error;
  }
  trace.recordInteraction(interaction);
}

/** Record an encodable structured model reply's value without affecting node behavior. */
export function recordModelValue(trace: EvaluationTrace | null | undefined, value: unknown): void {
  if (trace == null) return;
  let interaction: ModelValueInteraction;
  try {
    interaction = new ModelValueInteraction(value);
  } catch (error) {
    if (isCarrierError(error)) return;
    throw error;
  }
  trace.recordInteraction(interaction);
}

/** Record an encodable requested tool call without affecting the loop. */
export function recordToolCall(trace: EvaluationTrace | null | undefined, name: string, args: unknown): void {
  if (trace == null) return;
  let interaction: ToolCallInteraction;
  try {
    interaction = new ToolCallInteraction(name, args);
  } catch (error) {
    // Evidence stays observational for malformed arguments.
    if (isCarrierError(error)) return;
    throw error;
  }
  trace.recordInteraction(interaction);
}

/** Record an encodable visible tool result without affecting its invocation. */
export function recordToolResult(trace: EvaluationTrace | null | undefined, name: string, text: string): void {
  if (trace == null) return;
  let interaction: ToolResultInteraction;
  try {
    interaction = new ToolResultInteraction(name, text);
  } catch (error) {
    if (isCarrierError(error)) return;
    throw error;
  }
  trace.recordInteraction(interaction);
}
