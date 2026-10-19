/**
 * Mirrors `tests/test_llm_clients.py`: only the `api_key_env` test; the client tests and
 * `validate_model` go with the factory.
 */

import { expect, test } from "bun:test";

import { getApiKeyEnv, OPTIONAL_API_KEY_PROVIDERS, PROVIDER_API_KEY_ENV } from "../src/llm_clients/api_key_env.ts";

test("api key env maps dual region keys", () => {
  expect(getApiKeyEnv("qwen-cn")).toBe("DASHSCOPE_CN_API_KEY");
  expect(getApiKeyEnv("glm-cn")).toBe("ZHIPU_CN_API_KEY");
  expect(getApiKeyEnv("ollama")).toBeNull(); // local runtime, no key
});

test("the table lists the providers in order", () => {
  expect(Object.keys(PROVIDER_API_KEY_ENV)).toEqual([
    "openai",
    "anthropic",
    "google",
    "azure",
    "xai",
    "deepseek",
    "qwen",
    "qwen-cn",
    "glm",
    "glm-cn",
    "minimax",
    "minimax-cn",
    "openrouter",
    "ollama",
    "vllm",
  ]);
});

test("lookups lower-case the provider and know only the table's providers", () => {
  expect(getApiKeyEnv("OPENAI")).toBe("OPENAI_API_KEY");
  expect(getApiKeyEnv("no-such-provider")).toBeNull();
  expect(getApiKeyEnv("constructor")).toBeNull();
});

test("vllm is the only provider whose key is optional", () => {
  expect(OPTIONAL_API_KEY_PROVIDERS).toEqual(new Set(["vllm"]));
});
