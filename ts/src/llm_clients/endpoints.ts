/**
 * Each provider's default endpoint and the protocol constants that go with it, shared by the
 * model routes, model listing, connection records and OpenAI login. It imports nothing, so any
 * module may import it.
 */

/** The OpenAI API endpoint used when nothing configures another. */
export const OPENAI_API_ENDPOINT = "https://api.openai.com/v1";

/** The endpoint an OpenAI subscription (ChatGPT login) calls. */
export const SUBSCRIPTION_ENDPOINT = "https://chatgpt.com/backend-api/codex";

/** The Anthropic API root used when nothing configures another. */
export const ANTHROPIC_ENDPOINT = "https://api.anthropic.com";

/** The `anthropic-version` header value every Anthropic request sends. */
export const ANTHROPIC_VERSION = "2023-06-01";

/** The Gemini Developer API root used when nothing configures another. */
export const GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com";

/** The local Ollama server used when `OLLAMA_BASE_URL` is unset. */
export const OLLAMA_ENDPOINT = "http://localhost:11434";

/**
 * Providers on an OpenAI-compatible API. Ollama is not one: it has its own native route, which
 * sends the `ollama` block to `/api/chat` as written.
 */
export const OPENAI_COMPATIBLE: readonly string[] = Object.freeze([
  "openai",
  "xai",
  "deepseek",
  "qwen",
  "qwen-cn",
  "glm",
  "glm-cn",
  "minimax",
  "minimax-cn",
  "openrouter",
  "vllm",
]);

/**
 * The default base URL of each OpenAI-compatible provider other than `openai`, whose endpoint the
 * connection records. Dual-region providers keep separate endpoints because international and
 * China accounts cannot share credentials.
 */
export const PROVIDER_BASE_URL: Readonly<Record<string, string>> = Object.freeze({
  xai: "https://api.x.ai/v1",
  deepseek: "https://api.deepseek.com",
  qwen: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
  "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  glm: "https://api.z.ai/api/paas/v4/",
  "glm-cn": "https://open.bigmodel.cn/api/paas/v4/",
  minimax: "https://api.minimax.io/v1",
  "minimax-cn": "https://api.minimaxi.com/v1",
  openrouter: "https://openrouter.ai/api/v1",
  vllm: "http://localhost:8000/v1",
});
