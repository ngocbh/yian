/**
 * How an AGENT uses memory: model input from recalled notes and transcripts, and model and tool
 * effects recorded back into the memory turn.
 *
 * Only the compatibility descriptor is here so far: the provider and message format a stored
 * transcript was written with. Changing either requires an explicit memory reset; model, prompt
 * and tool-grant changes keep history. Framing recalled notes and projecting and recording
 * transcript messages are not ported yet.
 *
 * The descriptor's `message_format` is `MESSAGE_FORMAT` of `llm/messages.ts` (`yian.message.v1`),
 * not Python yian's `yian.langchain-message.v1`: TS stores its own message shape and never reads
 * a Python memory store.
 *
 * Knows about:   defaults (the `Env` type), llm (message constants), llm_clients, typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import type { Env } from "../../defaults.ts";
import { MESSAGE_FORMAT } from "../../llm/messages.ts";
import type { LLMConfig } from "../../llm_clients/config.ts";
import { type ConnectionSnapshot, resolveModelConfig } from "../../llm_clients/connection.ts";
import type { JsonObject } from "../../typesys/json.ts";

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
