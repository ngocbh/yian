/**
 * How an AGENT uses memory: recalled notes framed into the system prompt, and the text the model
 * reads for a tool result, a reported tool error, a skipped call, or a call a crash, a deadline or
 * an interrupt closed.
 *
 * Python's `nodes/agent/memory.py` also records model and tool effects into a session memory's
 * transcript and projects that history for the model; TS yian has no session memory, so an
 * AGENT's model reads only this start's messages.
 *
 * Knows about:   diagnostics, memory contracts, typesys, and the peer AGENT module `live`.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { reprText } from "../../diagnostics.ts";
import type { MemorySection } from "../../memory/contracts.ts";
import { canonicalJson } from "../../typesys/json.ts";
import { fromHost, writeAsText } from "../../typesys/values.ts";
import { errorName } from "./live.ts";

/**
 * Frame opaque memory text as labelled outer-to-local system sections.
 *
 * @returns `base` unchanged without sections; otherwise `base`, a header line, and canonical
 *   JSON `{"order": "outer-to-local", "sections": [{"index", "memory", "text"}, ...]}`, indices
 *   from 1.
 */
export function systemPromptWithMemory(base: string, sections: readonly MemorySection[]): string {
  if (sections.length === 0) return base;
  const payload = {
    order: "outer-to-local",
    sections: sections.map((section, index) => ({ index: index + 1, memory: section.memory, text: section.text })),
  };
  // The keys are ASCII, so canonical JSON's key order is Python's `sort_keys` order.
  return (
    base +
    "\n\nMemory context (canonical JSON; sections are ordered from outer " +
    "workflow to local agent):\n" +
    canonicalJson(payload)
  );
}

/** The result the model reads for a tool call a host interrupt skipped. */
export const SKIPPED_TOOL_RESULT = "Not run: interrupted by the user.";

/** The result the model reads for a tool call whose run a crash or a stop cut off before its result. */
export const TOOL_OUTCOME_UNKNOWN =
  "The tool call was interrupted; its outcome is unknown. Check its effect before retrying.";

/** The result the model reads for an `ask_user` call nobody answered before its deadline. */
export const ASK_TIMED_OUT_RESULT = "timed out: no answer";

/** The result the model reads for a pending call a host interrupt closed before its answer. */
export const TOOL_INTERRUPTED_OUTCOME =
  "The user interrupted this tool call; its outcome is unknown. Check its effect before retrying.";

/** Why a call was closed without its result, as an agent's closer records it. */
export type CloserReason = "crashed" | "timed_out" | "interrupted";

/** The text the model reads for a call closed for each reason. */
const CLOSER_TEXT: Readonly<Record<CloserReason, string>> = {
  crashed: TOOL_OUTCOME_UNKNOWN,
  timed_out: ASK_TIMED_OUT_RESULT,
  interrupted: TOOL_INTERRUPTED_OUTCOME,
};

/** Return the closer an agent writes for a call closed for `reason`: the text the model reads and the `closed` word. */
export function closerResult(reason: CloserReason): { readonly output: string; readonly closed: CloserReason } {
  return { output: CLOSER_TEXT[reason], closed: reason };
}

/**
 * Return a tool result as the text the model reads: text as itself, any other value in the text
 * form (design 3.2.3), `null` as `null`.
 *
 * Example: `toolText("lookup", {a: 1})` gives `{"a":1}`.
 *
 * @throws TypeCheckError when the result is not a yian value, read as `tool '<name>' result`.
 * @throws TypeError as `writeAsText` does.
 */
export function toolText(name: string, result: unknown): string {
  return writeAsText(fromHost(result, `tool ${reprText(name)} result`));
}

/**
 * Return a reported tool error as the text the model reads: its message (`String(error)` for a
 * non-`Error`), or its `name` when that is empty (`typeof` for a non-`Error`).
 *
 * @throws TypeCheckError when the message is not yian text (`toolText`).
 */
export function reportedText(name: string, error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return toolText(name, message) || errorName(error);
}
