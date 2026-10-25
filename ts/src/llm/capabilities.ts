/**
 * Declarative per-model capability table for OpenAI-compatible providers.
 *
 * This is the single place that knows which model IDs reject which API parameters or
 * require which structured-output method. Protocol adapters consult
 * `getCapabilities(model)` instead of hardcoding model-name `if` ladders, so adding a new
 * model (or a new provider quirk) means editing this table, not the adapter code.
 *
 * Pattern adapted from the per-model `compat:` flags DeepSeek themselves publish in their
 * integration guides (e.g. the Oh My Pi config schema documents `supportsToolChoice`,
 * `requiresReasoningContentForToolCalls` as declarative per-model fields).
 */

/**
 * How a model returns structured output.
 *
 * - `function_calling`: uses tools; respects `supportsToolChoice`.
 * - `json_mode`: uses `response_format: {type: "json_object"}`.
 * - `json_schema`: uses `response_format: {type: "json_schema", ...}`.
 * - `none`: no structured output available; the caller falls back to free text.
 */
export type StructuredMethod = "function_calling" | "json_mode" | "json_schema" | "none";

/** What an OpenAI-compatible model accepts at the API level. */
export interface ModelCapabilities {
  /** Whether the model accepts the `tool_choice` parameter. */
  readonly supportsToolChoice: boolean;
  /** The structured-output method to use for this model. */
  readonly preferredStructuredMethod: StructuredMethod;
  /**
   * Whether requests must set `reasoning_split: true`.
   *
   * MiniMax M2.x reasoning models need it so the `<think>` block lands in
   * `reasoning_details` instead of polluting `content`. Non-reasoning MiniMax models
   * (Coding Plan, MiniMax-Text-01, etc.) reject the flag, so it is set only where the
   * model consumes it. (#826)
   */
  readonly requiresReasoningSplit: boolean;
}

// DeepSeek's thinking models accept the `tools` array but reject the `tool_choice`
// parameter (official Oh My Pi integration guide and the 400 response in issue #678).
// Their official tool-calling examples (api-docs.deepseek.com/guides/tool_calls) pass
// `tools` without `tool_choice`; supportsToolChoice is false so the caller mirrors that
// and leaves `tool_choice` out. Their echo of `reasoning_content` on later turns is
// switched by `chatOptions` in openai_chat.ts, per provider.
const DEEPSEEK_THINKING: ModelCapabilities = Object.freeze({
  supportsToolChoice: false,
  preferredStructuredMethod: "function_calling",
  requiresReasoningSplit: false,
});

const DEEPSEEK_CHAT: ModelCapabilities = Object.freeze({
  supportsToolChoice: true,
  preferredStructuredMethod: "function_calling",
  requiresReasoningSplit: false,
});

// MiniMax M2.x reasoning models accept the tools array, but their tool_choice parameter is
// restricted to the enum {"none", "auto"} (platform.minimax.io/docs/api-reference/text-post).
// A function_calling request that names the schema's function in tool_choice sends an
// object, which MiniMax 400s: the same shape as the DeepSeek bug. supportsToolChoice is
// false so the caller leaves tool_choice out; the schema still ships as a tool. json_mode
// response_format is only for MiniMax-Text-01, not M2.x.
const MINIMAX_THINKING: ModelCapabilities = Object.freeze({
  supportsToolChoice: false,
  preferredStructuredMethod: "function_calling",
  requiresReasoningSplit: true,
});

const DEFAULT: ModelCapabilities = Object.freeze({
  supportsToolChoice: true,
  preferredStructuredMethod: "function_calling",
  requiresReasoningSplit: false,
});

// Catalog sentinel: a single entry whose structured method is `none`. It exists so the
// prompt-injection fallback is reachable through the real capability source on a concrete
// (provider, model) pair; no real provider is marked no-native today.
const NO_NATIVE_STRUCTURED: ModelCapabilities = Object.freeze({
  supportsToolChoice: false,
  preferredStructuredMethod: "none",
  requiresReasoningSplit: false,
});

// Exact-ID matches take precedence over pattern matches.
const BY_ID: ReadonlyMap<string, ModelCapabilities> = new Map([
  ["deepseek-chat", DEEPSEEK_CHAT],
  // Test/escape-hatch sentinel: the only entry with no native structured output.
  ["no-structured-sentinel", NO_NATIVE_STRUCTURED],
]);

// Thinking-mode families: `deepseek-reasoner`, `deepseek-v4-*` and MiniMax's M2.x lineup
// (platform.minimax.io/docs/api-reference/text-openai-api), and future `deepseek-v5-*` or
// `MiniMax-M3*` variants without a table edit.
const BY_PATTERN: readonly (readonly [RegExp, ModelCapabilities])[] = [
  [/^deepseek-v\d/, DEEPSEEK_THINKING],
  [/^deepseek-reasoner/, DEEPSEEK_THINKING],
  [/^MiniMax-M\d/, MINIMAX_THINKING],
];

/** Resolve `model`'s capabilities by exact ID, then pattern, then the default profile. */
export function getCapabilities(model: string): ModelCapabilities {
  const exact = BY_ID.get(model);
  if (exact !== undefined) return exact;
  for (const [pattern, caps] of BY_PATTERN) {
    if (pattern.test(model)) return caps;
  }
  return DEFAULT;
}

// Providers with their own native structured output that the OpenAI-compatible table above
// does not cover. For these, native structured output is always available whatever the
// model id.
const NATIVE_STRUCTURED_PROVIDERS: ReadonlySet<string> = new Set(["anthropic", "google", "azure", "ollama"]);

/**
 * Whether `(provider, model)` has native structured output.
 *
 * Native-by-default providers (`anthropic`/`google`/`azure`/`ollama`, any case) always do;
 * they route to their own protocols, not the OpenAI-compatible table. Every other provider
 * (OpenAI-compatible names, or an unknown future provider) consults the per-model table: a
 * model whose `preferredStructuredMethod` is `none` (today only the catalog sentinel) has
 * no native path, so the caller falls back to prompt injection. An unknown model resolves
 * to the default profile (`function_calling`) and thus stays native.
 */
export function supportsNativeStructured(provider: string, model: string): boolean {
  if (NATIVE_STRUCTURED_PROVIDERS.has(provider.toLowerCase())) return true;
  return getCapabilities(model).preferredStructuredMethod !== "none";
}
