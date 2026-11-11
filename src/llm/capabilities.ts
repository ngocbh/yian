/**
 * Per-model API quirks, in one table of model profiles keyed by wire protocol and model.
 *
 * Each row names a protocol, a model (an exact id, or a pattern for a family or a prefix) and the
 * profile fields it sets. A model's profile is `DEFAULT_PROFILE` with every row of its protocol
 * that matches it applied, in table order. Protocol adapters and the structured-output plan read
 * `modelProfile` rather than keeping model lists of their own, so a new model or quirk is a row
 * here.
 *
 * The per-model `compat:` flags DeepSeek publishes in its integration guides (e.g. the Oh My Pi
 * config schema's `supportsToolChoice`) are the pattern for the Chat Completions rows.
 */

/** What one model's API accepts and needs beyond its protocol's defaults. */
export interface ModelProfile {
  /** Anthropic Messages: the `max_tokens` sent when the request's block sets none. */
  readonly maxTokens: bigint;
  /** Anthropic Messages: whether a system message after the first may stay where it is. */
  readonly inPlaceSystem: boolean;
  /** Whether the API accepts a tool choice that names one tool (a forced tool). */
  readonly namedToolChoice: boolean;
  /** Chat Completions: whether requests must set `reasoning_split: true`. */
  readonly reasoningSplit: boolean;
  /**
   * Azure: whether a structured schema goes as a forced tool rather than a JSON-schema output
   * format (a legacy deployment).
   */
  readonly schemaAsTool: boolean;
}

/** A model's profile when no row matches it. */
const DEFAULT_PROFILE: ModelProfile = Object.freeze({
  maxTokens: 4096n,
  inPlaceSystem: false,
  namedToolChoice: true,
  reasoningSplit: false,
  schemaAsTool: false,
});

const ANTHROPIC = "anthropic_messages";
const CHAT = "openai_chat";
const RESPONSES = "openai_responses";

/** One table row: the protocol, the model (an exact id, or a pattern) and the fields it sets. */
type Row = readonly [protocol: string, model: string | RegExp, fields: Partial<ModelProfile>];

/** Return a pattern matching the model ids that start with `start`. */
function prefix(start: string): RegExp {
  return new RegExp(`^${start.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`);
}

/** Return one row per entry of `models` on `protocol`, each setting `fields`. */
function rows(protocol: string, models: readonly (string | RegExp)[], fields: Partial<ModelProfile>): Row[] {
  return models.map((model) => [protocol, model, fields] as const);
}

const ROWS: readonly Row[] = Object.freeze([
  // The default `max_tokens` by exact id, from langchain-anthropic 1.7.5's model profiles; those
  // carry no limits for the Mythos models (not on models.dev), so theirs are the 128K max output
  // from Anthropic's model docs.
  ...rows(
    ANTHROPIC,
    [
      "claude-haiku-4-5",
      "claude-haiku-4-5-20251001",
      "claude-opus-4-5",
      "claude-opus-4-5-20251101",
      "claude-sonnet-4-5",
      "claude-sonnet-4-5-20250929",
    ],
    { maxTokens: 64000n },
  ),
  ...rows(
    ANTHROPIC,
    [
      "claude-fable-5",
      "claude-fable-5-1",
      "claude-mythos-5",
      "claude-mythos-5-1",
      "claude-opus-4-6",
      "claude-opus-4-7",
      "claude-opus-4-8",
      "claude-opus-5",
      "claude-opus-5-5",
      "claude-sonnet-4-6",
      "claude-sonnet-5",
      "claude-sonnet-5-5",
    ],
    { maxTokens: 128000n },
  ),
  // Models that accept a system message in the middle of the conversation (LangChain's list).
  ...rows(
    ANTHROPIC,
    ["claude-fable-5", "claude-mythos-5", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5-5"].map(prefix),
    { inPlaceSystem: true },
  ),
  // Models the Messages API refuses a forced tool choice for.
  ...rows(
    ANTHROPIC,
    ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-mythos-5-1"].map(prefix),
    { namedToolChoice: false },
  ),
  // DeepSeek's thinking models (`deepseek-reasoner`, `deepseek-v4-*` and later) accept the `tools`
  // array but 400 on `tool_choice` (Oh My Pi's integration guide); their tool-calling examples
  // (api-docs.deepseek.com/guides/tool_calls) send `tools` alone. MiniMax's M2.x reasoning models
  // take only `"none"` or `"auto"` (platform.minimax.io/docs/api-reference/text-post), and need
  // `reasoning_split` so the `<think>` block lands in `reasoning_details`; their non-reasoning
  // models reject it. A future `deepseek-v5-*` or `MiniMax-M3*` matches without an edit.
  ...rows(CHAT, [/^deepseek-v\d/, /^deepseek-reasoner/], { namedToolChoice: false }),
  ...rows(CHAT, [/^MiniMax-M\d/], { namedToolChoice: false, reasoningSplit: true }),
  // Legacy Azure deployments get the schema as a forced tool (LangChain's check for `gpt-3*`,
  // `gpt-4-*` and `gpt-4`), on either API.
  ...[CHAT, RESPONSES].flatMap((protocol) =>
    rows(protocol, [prefix("gpt-3"), prefix("gpt-4-"), "gpt-4"], { schemaAsTool: true }),
  ),
]);

/**
 * Return `model`'s profile on `protocol` (a wire protocol name, such as `anthropic_messages`):
 * `DEFAULT_PROFILE` with every matching row applied in table order, frozen.
 */
export function modelProfile(protocol: string, model: string): ModelProfile {
  let profile = DEFAULT_PROFILE;
  for (const [rowProtocol, match, fields] of ROWS) {
    if (rowProtocol !== protocol || !(typeof match === "string" ? match === model : match.test(model))) continue;
    profile = { ...profile, ...fields };
  }
  return Object.freeze(profile);
}

// Providers with their own native structured output, whatever the model id.
const NATIVE_STRUCTURED_PROVIDERS: ReadonlySet<string> = new Set(["anthropic", "google", "azure", "ollama"]);

// Test and escape-hatch sentinel: the one model id with no native structured output, so the
// prompt fallback is reachable through a concrete (provider, model) pair.
const NO_NATIVE_STRUCTURED: ReadonlySet<string> = new Set(["no-structured-sentinel"]);

/**
 * Whether `(provider, model)` has native structured output.
 *
 * Native-by-default providers (`anthropic`/`google`/`azure`/`ollama`, any case) always do. Every
 * other provider does unless its model is the no-native sentinel, which falls back to prompt
 * injection; an unknown model stays native.
 */
export function supportsNativeStructured(provider: string, model: string): boolean {
  if (NATIVE_STRUCTURED_PROVIDERS.has(provider.toLowerCase())) return true;
  return !NO_NATIVE_STRUCTURED.has(model);
}
