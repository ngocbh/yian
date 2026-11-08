/**
 * What a memory value renders as in prompt text: its recorded recall, with the hint shown only
 * when the rendering agent holds the hint's tool. Rendering reads only its arguments, so it is
 * pure.
 *
 * It imports only `typesys`.
 */

import type { Recall } from "../typesys/memory.ts";

// Text earlier runs wrote must not close a `<memory>` frame around it, even with whitespace
// around the `/`. This protects only frames written with that tag; yian owns no frame.
const FRAME_CLOSE = /<(\s*)\/(\s*memory)/gi;

/** Return `text` with a `\` before the `/` of every closing `memory` tag, in any case and spacing. */
function escapeFrame(text: string): string {
  return text.replace(FRAME_CLOSE, "<$1\\/$2");
}

/**
 * Render a memory's recall: its text, then its cut note, then its hint, one per line, leaving out
 * an empty one. The whole rendering is escaped, the hint's tool name included, so no part of it
 * closes a `<memory>` frame.
 *
 * @param recall The recorded recall, or `null` for an Optional memory left out, which renders empty.
 * @param hintTool The model-visible name of the hint's tool when the agent holds it, or `null` to
 *   leave the hint out. Every `{tool}` in the hint becomes this name.
 */
export function renderRecall(recall: Recall | null, hintTool: string | null): string {
  if (recall === null) return "";
  const lines = [recall.text, recall.cut ?? ""];
  if (recall.hint !== null && hintTool !== null) {
    lines.push(recall.hint.text.replaceAll("{tool}", () => hintTool));
  }
  return escapeFrame(lines.filter((line) => line !== "").join("\n"));
}
