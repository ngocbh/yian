/**
 * Canonical provider to API-key environment variable mapping.
 *
 * The single source of truth for which environment variable holds the API key for each supported
 * LLM provider. It holds names only and reads no environment; `connection.ts` reads the
 * variables. A config `apiKey` takes precedence over these variables when the client is built.
 *
 * When adding a new provider, register its variable here so the CLI asks for it instead of the
 * first API call failing.
 */

/** Provider name to the variable that holds its API key; `null` for a provider that takes none. */
export const PROVIDER_API_KEY_ENV: Readonly<Record<string, string | null>> = Object.freeze(
  Object.assign(Object.create(null) as Record<string, string | null>, {
    openai: "OPENAI_API_KEY",
    anthropic: "ANTHROPIC_API_KEY",
    google: "GOOGLE_API_KEY",
    azure: "AZURE_OPENAI_API_KEY",
    xai: "XAI_API_KEY",
    deepseek: "DEEPSEEK_API_KEY",
    // Dual-region providers each carry their own account; keys are not interchangeable between
    // the international and China endpoints.
    qwen: "DASHSCOPE_API_KEY",
    "qwen-cn": "DASHSCOPE_CN_API_KEY",
    glm: "ZHIPU_API_KEY",
    "glm-cn": "ZHIPU_CN_API_KEY",
    minimax: "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    openrouter: "OPENROUTER_API_KEY",
    // Local runtimes do not authenticate.
    ollama: null,
    // vLLM is keyless by default but can be started with `--api-key`; the key is optional (see
    // OPTIONAL_API_KEY_PROVIDERS).
    vllm: "VLLM_API_KEY",
  }),
);

/**
 * Self-hosted runtimes whose key variable may be unset.
 *
 * vLLM runs keyless unless its server was started with `--api-key`, so its client sends a dummy
 * key and the CLI does not ask for one.
 */
export const OPTIONAL_API_KEY_PROVIDERS: ReadonlySet<string> = new Set(["vllm"]);

/**
 * Return the variable name for `provider`'s API key, or `null` if none applies.
 *
 * Unknown providers also give `null`: callers treat that as "no key check possible", not as "no
 * key required".
 */
export function getApiKeyEnv(provider: string): string | null {
  const key = provider.toLowerCase();
  return Object.hasOwn(PROVIDER_API_KEY_ENV, key) ? (PROVIDER_API_KEY_ENV[key] ?? null) : null;
}
