/**
 * How an AGENT uses memory: model input from recalled notes and transcripts, and model and tool
 * effects recorded back into the memory turn.
 *
 * Here so far: framing recalled notes into the system prompt; the compatibility descriptor: the
 * provider and message format a stored transcript was written with (changing either requires an
 * explicit memory reset; model, prompt and tool-grant changes keep history); and the text the
 * model reads for a tool result, a reported tool error or a skipped call. Projecting and recording
 * transcript messages are not ported yet.
 *
 * The descriptor's `message_format` is `MESSAGE_FORMAT` of `llm/messages.ts` (`yian.message.v1`),
 * not Python yian's `yian.langchain-message.v1`: TS stores its own message shape and never reads
 * a Python memory store.
 *
 * Knows about:   defaults (the `Env` type), diagnostics, llm (message constants), llm_clients,
 *                memory contracts, typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import type { Env } from "../../defaults.ts";
import { reprText } from "../../diagnostics.ts";
import { MESSAGE_FORMAT } from "../../llm/messages.ts";
import type { LLMConfig } from "../../llm_clients/config.ts";
import { type ConnectionSnapshot, resolveModelConfig } from "../../llm_clients/connection.ts";
import type { MemorySection } from "../../memory/contracts.ts";
import { canonicalJson, type JsonObject } from "../../typesys/json.ts";
import { fromHost, writeAsText } from "../../typesys/values.ts";

/** The result the model reads for a tool call a host interrupt skipped. */
export const SKIPPED_TOOL_RESULT = "Not run: interrupted by the user.";

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
 * Return the provider and versioned message format `config` selects, independent of the model,
 * current grants and credentials.
 *
 * An `LLMConfig` gives its `.provider`, a record its `provider`. A provider that is unset, `null`
 * or `""` resolves as the model factory resolves it (`resolveModelConfig` with `options`). The
 * provider is lowercased.
 *
 * Example: `compatibility({provider: "OpenAI", api_key: "k"})` gives
 * `{provider: "openai", message_format: "yian.message.v1"}`.
 *
 * Throws `RangeError` if the provider is absent and no default sets one. Other errors from
 * `resolveModelConfig` (such as a `ValidationError` for `provider: ""`) propagate.
 */
export function compatibility(
  config: Readonly<Record<string, unknown>> | LLMConfig | null,
  options: { snapshot?: ConnectionSnapshot; env?: Env } = {},
): JsonObject {
  const given = config?.["provider"];
  const provider =
    typeof given === "string" && given !== ""
      ? given
      : (resolveModelConfig(config ?? {}, options)["provider"] as string);
  return { provider: provider.toLowerCase(), message_format: MESSAGE_FORMAT };
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
  return toolText(name, message) || (error instanceof Error ? error.name : typeof error);
}
