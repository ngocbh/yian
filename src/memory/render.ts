/**
 * What a memory value renders as in prompt text: its recorded recall, with the hint shown only
 * when the rendering agent holds the hint's tool. Rendering reads only its arguments, so it is
 * pure.
 *
 * It imports only `typesys`.
 */

import type { Recall } from "../typesys/memory.ts";

// Text earlier runs wrote must not close a `<memory>` frame around it. This protects only frames
// written with that tag; yian owns no frame.
const FRAME_CLOSE = /<\/(memory)/gi;

/** Return `text` with every `</memory`, in any case, written `<\/memory`. */
function escapeFrame(text: string): string {
  return text.replace(FRAME_CLOSE, "<\\/$1");
}

/**
 * Render a memory's recall: its text, then its cut note, then its hint, one per line.
 *
 * @param recall The recorded recall, or `null` for an Optional memory left out, which renders empty.
 * @param hintTool The model-visible name of the hint's tool when the agent holds it, or `null` to
 *   leave the hint out. Every `{tool}` in the hint becomes this name.
 */
export function renderRecall(recall: Recall | null, hintTool: string | null): string {
  if (recall === null) return "";
  const lines: string[] = [];
  if (recall.text !== "") lines.push(escapeFrame(recall.text));
  if (recall.cut !== null) lines.push(escapeFrame(recall.cut));
  if (recall.hint !== null && hintTool !== null) {
    lines.push(recall.hint.text.replaceAll("{tool}", () => hintTool));
  }
  return lines.join("\n");
}
