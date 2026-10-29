/**
 * Shared model catalog for CLI selections and the known-model check.
 */

import { compareCodeUnits } from "../diagnostics.ts";

/** A model choice: the label shown in a picker and the model id it selects. */
export type ModelOption = readonly [label: string, value: string];

/** Selection mode (`quick`, `deep`) to the model options offered in that mode. */
export type ModeOptions = Readonly<Record<string, readonly ModelOption[]>>;

/** Freeze a table into a null-prototype record so only its own keys are entries. */
function table<T>(entries: Record<string, T>): Readonly<Record<string, T>> {
  return Object.freeze(Object.assign(Object.create(null) as Record<string, T>, entries));
}

/** Freeze one mode's option list, each option included. */
function options(...entries: ModelOption[]): readonly ModelOption[] {
  return Object.freeze(entries.map((entry) => Object.freeze(entry)));
}

// Shared model list for GLM via Z.AI (international) and BigModel (China).
// Source: docs.z.ai (GLM Coding Plan supported models + LLM guides).
// All GLM 4.7+ entries support thinking mode via thinking={"type":"enabled"}.
const GLM_MODELS: ModeOptions = table({
  quick: options(
    ["GLM-5-Turbo - Fast, switchable thinking modes", "glm-5-turbo"],
    ["GLM-4.7 - Previous-gen flagship", "glm-4.7"],
    ["GLM-4.5-Air - Lightweight, cost-efficient", "glm-4.5-air"],
    ["Custom model ID", "custom"],
  ),
  deep: options(
    ["GLM-5.1 - Latest flagship, 204K ctx", "glm-5.1"],
    ["GLM-5 - Flagship, 204K ctx", "glm-5"],
    ["GLM-4.7 - Previous-gen flagship", "glm-4.7"],
    ["Custom model ID", "custom"],
  ),
});

// Shared model list for Qwen's global (dashscope-intl) and CN (dashscope) endpoints.
// Source: modelstudio.console.alibabacloud.com (Featured Models — Flagship + Cost-optimized).
//
// Only versioned IDs are exposed in the dropdown. The version-less aliases
// (qwen-plus, qwen-flash) are documented by Alibaba as auto-upgrading
// pointers ("backbone, latest, and snapshot ... have been upgraded to the
// Qwen3 series"), which means their behavior shifts when Alibaba rotates
// the backing model. Users who want a specific generation pick it
// explicitly; users who really want auto-latest can enter the alias via
// "Custom model ID".
const QWEN_MODELS: ModeOptions = table({
  quick: options(
    ["Qwen 3.6 Flash - Latest fast, agentic coding + vision-language", "qwen3.6-flash"],
    ["Qwen 3.5 Flash - Previous-gen fast", "qwen3.5-flash"],
    ["Custom model ID", "custom"],
  ),
  deep: options(
    ["Qwen 3.7 Max - Latest flagship reasoning agent, 1M ctx", "qwen3.7-max"],
    ["Qwen 3.6 Plus - Vision-language, agentic coding", "qwen3.6-plus"],
    ["Qwen 3.5 Plus - Previous-gen flagship", "qwen3.5-plus"],
    ["Custom model ID", "custom"],
  ),
});

// Shared model list for MiniMax's global and CN endpoints (same IDs).
// Full official lineup per platform.minimax.io/docs/api-reference/text-openai-api.
// All M2.x models share a 204,800-token context window.
const MINIMAX_MODELS: ModeOptions = table({
  quick: options(
    ["MiniMax-M2.7-highspeed - Faster M2.7, 204K ctx, ~100 TPS", "MiniMax-M2.7-highspeed"],
    ["MiniMax-M2.5-highspeed - Previous-gen highspeed, 204K ctx", "MiniMax-M2.5-highspeed"],
    ["MiniMax-M2.1-highspeed - M2.1 highspeed, 204K ctx", "MiniMax-M2.1-highspeed"],
    ["Custom model ID", "custom"],
  ),
  deep: options(
    ["MiniMax-M2.7 - Flagship, SOTA on coding/agent benchmarks, 204K ctx", "MiniMax-M2.7"],
    ["MiniMax-M2.7-highspeed - Same quality as M2.7, ~100 TPS", "MiniMax-M2.7-highspeed"],
    ["MiniMax-M2.5 - Previous-gen flagship, 204K ctx", "MiniMax-M2.5"],
    ["MiniMax-M2.1 - Earlier M2 line, 204K ctx", "MiniMax-M2.1"],
    ["MiniMax-M2 - Base M2, 204K ctx", "MiniMax-M2"],
    ["Custom model ID", "custom"],
  ),
});

/** Provider to its model options per selection mode. */
export const MODEL_OPTIONS: Readonly<Record<string, ModeOptions>> = table({
  openai: table({
    quick: options(
      ["GPT-5.4 Mini - Fast, strong coding and tool use", "gpt-5.4-mini"],
      ["GPT-5.4 Nano - Cheapest, high-volume tasks", "gpt-5.4-nano"],
      ["GPT-5.5 - Latest frontier, 1M context", "gpt-5.5"],
      ["GPT-4.1 - Smartest non-reasoning model", "gpt-4.1"],
    ),
    deep: options(
      ["GPT-5.5 - Latest frontier, 1M context", "gpt-5.5"],
      ["GPT-5.4 - Previous-gen frontier, 1M context, cost-effective", "gpt-5.4"],
      ["GPT-5.2 - Strong reasoning, cost-effective", "gpt-5.2"],
      ["GPT-5.5 Pro - Most capable, expensive ($30/$180 per 1M tokens)", "gpt-5.5-pro"],
    ),
  }),
  anthropic: table({
    quick: options(
      ["Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"],
      ["Claude Haiku 4.5 - Fastest with near-frontier intelligence", "claude-haiku-4-5"],
      ["Claude Sonnet 4.5 - High-performance for agents and coding", "claude-sonnet-4-5"],
    ),
    deep: options(
      ["Claude Mythos 5.1", "claude-mythos-5-1"],
      ["Claude Opus 4.8 - Latest frontier, agentic coding and reasoning", "claude-opus-4-8"],
      ["Claude Opus 4.7 - Previous frontier, long-running agents", "claude-opus-4-7"],
      ["Claude Opus 4.6 - Frontier intelligence, agents and coding", "claude-opus-4-6"],
      ["Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"],
    ),
  }),
  google: table({
    quick: options(
      ["Gemini 3.5 Flash - Latest, frontier agentic + coding (GA)", "gemini-3.5-flash"],
      ["Gemini 3.1 Flash Lite - Most cost-efficient (GA)", "gemini-3.1-flash-lite"],
      ["Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"],
      ["Gemini 2.5 Flash Lite - Fast, low-cost", "gemini-2.5-flash-lite"],
    ),
    deep: options(
      ["Gemini 3.1 Pro - Reasoning-first, complex workflows (preview)", "gemini-3.1-pro-preview"],
      ["Gemini 3.5 Flash - Latest GA, strong agentic + coding", "gemini-3.5-flash"],
      ["Gemini 2.5 Pro - Stable pro model", "gemini-2.5-pro"],
      ["Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"],
    ),
  }),
  xai: table({
    quick: options(
      ["Grok 4.3 - Latest flagship, fast with built-in reasoning", "grok-4.3"],
      ["Grok Build 0.1 - Coding-specialized, 256K ctx", "grok-build-0.1"],
      ["Grok 4 Fast (Non-Reasoning) - Speed optimized", "grok-4-fast-non-reasoning"],
    ),
    deep: options(
      ["Grok 4.3 - Latest flagship, built-in reasoning, 1M ctx", "grok-4.3"],
      ["Grok 4.20 (Reasoning) - Previous-gen reasoning", "grok-4.20-0309-reasoning"],
      ["Grok 4 Fast (Reasoning) - High-performance", "grok-4-fast-reasoning"],
      ["Grok 4 - Flagship (dated build)", "grok-4-0709"],
    ),
  }),
  deepseek: table({
    quick: options(
      ["DeepSeek V4 Flash - Latest V4 fast model", "deepseek-v4-flash"],
      ["DeepSeek V3.2", "deepseek-chat"],
      ["Custom model ID", "custom"],
    ),
    deep: options(
      ["DeepSeek V4 Pro - Latest V4 flagship model", "deepseek-v4-pro"],
      ["DeepSeek V3.2 (thinking)", "deepseek-reasoner"],
      ["DeepSeek V3.2", "deepseek-chat"],
      ["Custom model ID", "custom"],
    ),
  }),
  // Each region pair serves the same model ids, so its two provider keys share one list.
  qwen: QWEN_MODELS,
  "qwen-cn": QWEN_MODELS,
  glm: GLM_MODELS,
  "glm-cn": GLM_MODELS,
  minimax: MINIMAX_MODELS,
  "minimax-cn": MINIMAX_MODELS,
  // OpenRouter: fetched dynamically. Azure: any deployed model name.
  // Ollama display labels intentionally omit a "local" marker — the
  // endpoint is configurable via OLLAMA_BASE_URL, so the same labels
  // apply whether the user runs ollama-serve on localhost or against a
  // remote host.
  // "Custom model ID" lets users pick any model they have pulled via
  // `ollama pull` beyond the three suggested defaults.
  ollama: table({
    quick: options(
      ["Qwen3:latest (8B)", "qwen3:latest"],
      ["GPT-OSS:latest (20B)", "gpt-oss:latest"],
      ["GLM-4.7-Flash:latest (30B)", "glm-4.7-flash:latest"],
      ["Custom model ID", "custom"],
    ),
    deep: options(
      ["GLM-4.7-Flash:latest (30B)", "glm-4.7-flash:latest"],
      ["GPT-OSS:latest (20B)", "gpt-oss:latest"],
      ["Qwen3:latest (8B)", "qwen3:latest"],
      ["Custom model ID", "custom"],
    ),
  }),
});

/**
 * Return the shared model options for `provider` (case-insensitive) in selection `mode`.
 *
 * @throws RangeError when the catalog has no such provider or mode.
 */
export function getModelOptions(provider: string, mode: string): readonly ModelOption[] {
  const key = provider.toLowerCase();
  const modes = Object.hasOwn(MODEL_OPTIONS, key) ? MODEL_OPTIONS[key]! : undefined;
  if (modes === undefined || !Object.hasOwn(modes, mode)) {
    throw new RangeError(`unknown model options: ${key}/${mode}`);
  }
  return modes[mode]!;
}

/**
 * Build each catalog provider's known model ids: its distinct option values, sorted, without the
 * picker's `"custom"` entry, which asks for a model id rather than naming one.
 */
export function getKnownModels(): Record<string, string[]> {
  const known = Object.create(null) as Record<string, string[]>;
  for (const [provider, modes] of Object.entries(MODEL_OPTIONS)) {
    const values = new Set(Object.values(modes).flatMap((entries) => entries.map(([, value]) => value)));
    values.delete("custom");
    known[provider] = [...values].sort(compareCodeUnits);
  }
  return known;
}

// Built once: the known-model check runs on every model build.
const KNOWN_MODELS: Readonly<Record<string, readonly string[]>> = table(getKnownModels());

/**
 * Whether `model` passes the known-model check for `provider` (case-insensitive provider, exact
 * model): it is one of the provider's known model ids. Any model passes for `ollama`, which serves
 * whatever was pulled, and for a provider outside the catalog (OpenRouter, vLLM, Azure, ...).
 */
export function isKnownModel(provider: string, model: string): boolean {
  const key = provider.toLowerCase();
  if (key === "ollama" || !Object.hasOwn(KNOWN_MODELS, key)) return true;
  return KNOWN_MODELS[key]!.includes(model);
}
