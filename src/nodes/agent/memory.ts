/**
 * How an AGENT uses memory: recalled notes framed into the system prompt.
 *
 * yian has no session memory, so an AGENT's model reads only this start's messages.
 *
 * Knows about:   memory contracts and typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import type { MemorySection } from "../../memory/contracts.ts";
import { canonicalJson } from "../../typesys/json.ts";

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
  // The keys are ASCII, so canonical JSON's key order is plain sorted order.
  return (
    base +
    "\n\nMemory context (canonical JSON; sections are ordered from outer " +
    "workflow to local agent):\n" +
    canonicalJson(payload)
  );
}
