/**
 * Model name validators for each provider.
 */

import { getKnownModels } from "./model_catalog.ts";

// Self-hosted / aggregator providers serve arbitrary model names, so we don't
// constrain them to a catalog.
const ANY_MODEL_PROVIDERS: readonly string[] = ["ollama", "openrouter", "vllm"];

/** Provider to its known model ids, for every catalog provider that is not an any-model one. */
export const VALID_MODELS: Readonly<Record<string, readonly string[]>> = Object.freeze(
  Object.assign(
    Object.create(null) as Record<string, readonly string[]>,
    Object.fromEntries(
      Object.entries(getKnownModels())
        .filter(([provider]) => !ANY_MODEL_PROVIDERS.includes(provider))
        .map(([provider, models]) => [provider, Object.freeze(models)]),
    ),
  ),
);

/**
 * Check whether `model` is a valid name for `provider` (case-insensitive provider, exact model).
 *
 * Any model is accepted for ollama, openrouter, vllm, and for providers outside the catalog.
 */
export function validateModel(provider: string, model: string): boolean {
  const key = provider.toLowerCase();
  if (ANY_MODEL_PROVIDERS.includes(key)) return true;
  if (!Object.hasOwn(VALID_MODELS, key)) return true;
  return VALID_MODELS[key]!.includes(model);
}
