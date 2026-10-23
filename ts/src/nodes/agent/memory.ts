/**
 * How an AGENT uses memory: recalled notes framed into the system prompt, and the text the model
 * reads for a tool result or a reported tool error.
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
