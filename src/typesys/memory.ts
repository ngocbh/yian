/**
 * The plain shapes a memory value records: its recall and its tool specs. They hold data only;
 * `memory/base.ts` re-exports them beside the contract that produces them.
 */

import type { JsonObject } from "./json.ts";

/**
 * What a memory shows in prompt text, read once when the run starts.
 *
 * `text` is the always-on part, `cut` says what was left out to keep it within the type's cap,
 * and `hint` says how to reach the rest: `{tool}` in its `text` becomes the model-visible name of
 * the tool `tool` when the rendering agent holds it.
 */
export type Recall = { text: string; cut: string | null; hint: { tool: string; text: string } | null };

/** One tool of a memory as its value records it: a write tool's call goes through `apply`. */
export type RecordedMemoryTool = { write: boolean; description: string; parameters: JsonObject };
