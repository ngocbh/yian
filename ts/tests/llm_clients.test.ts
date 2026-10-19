/**
 * Mirrors `tests/test_llm_clients.py`: the `api_key_env` test, the model catalog, and the
 * `validate_model` tests; the client tests (including the Anthropic unknown-model warning half of
 * the Mythos test) go with the factory.
 */

import { expect, test } from "bun:test";

import { getApiKeyEnv, OPTIONAL_API_KEY_PROVIDERS, PROVIDER_API_KEY_ENV } from "../src/llm_clients/api_key_env.ts";
import { getKnownModels, getModelOptions, MODEL_OPTIONS } from "../src/llm_clients/model_catalog.ts";
import { VALID_MODELS, validateModel } from "../src/llm_clients/validators.ts";

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

test("vllm accepts any model name", () => {
  expect(validateModel("vllm", "any/arbitrary-served-model")).toBe(true);
});

test("ollama and openrouter accept any model name", () => {
  expect(validateModel("ollama", "any/arbitrary-served-model")).toBe(true);
  expect(validateModel("openrouter", "any/arbitrary-served-model")).toBe(true);
});

test("claude mythos 5.1 is a known anthropic model", () => {
  expect(validateModel("anthropic", "claude-mythos-5-1")).toBe(true);
});

test("validateModel accepts any model for providers outside the catalog", () => {
  expect(validateModel("azure", "my-deployment")).toBe(true);
  expect(validateModel("no-such-provider", "anything")).toBe(true);
  expect(validateModel("toString", "anything")).toBe(true);
  expect(validateModel("__proto__", "anything")).toBe(true);
});

test("validateModel lowercases the provider but not the model", () => {
  expect(validateModel("OpenAI", "gpt-5.5")).toBe(true);
  expect(validateModel("openai", "GPT-5.5")).toBe(false);
  expect(validateModel("openai", "no-such-model")).toBe(false);
  expect(validateModel("VLLM", "anything")).toBe(true);
});

test("custom counts as known where the catalog lists it", () => {
  expect(validateModel("deepseek", "custom")).toBe(true);
  expect(validateModel("glm-cn", "custom")).toBe(true);
  expect(validateModel("openai", "custom")).toBe(false);
  expect(validateModel("anthropic", "custom")).toBe(false);
});

test("VALID_MODELS holds the known models of every catalog provider but the any-model ones", () => {
  const known = getKnownModels();
  expect(Object.keys(VALID_MODELS)).toEqual(Object.keys(known).filter((p) => p !== "ollama"));
  for (const [provider, models] of Object.entries(VALID_MODELS)) expect(models).toEqual(known[provider]!);
});

test("getKnownModels gives each provider's sorted distinct values", () => {
  const known = getKnownModels();
  expect(Object.keys(known)).toEqual(Object.keys(MODEL_OPTIONS));
  expect(known["anthropic"]!).toEqual([
    "claude-haiku-4-5",
    "claude-mythos-5-1",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
  ]);
  // MiniMax-M2.7-highspeed and custom are in both quick and deep but listed once.
  expect(known["minimax"]!).toEqual([
    "MiniMax-M2",
    "MiniMax-M2.1",
    "MiniMax-M2.1-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "custom",
  ]);
});

test("the catalog keeps Python's providers, labels and order", () => {
  expect(Object.keys(MODEL_OPTIONS)).toEqual([
    "openai",
    "anthropic",
    "google",
    "xai",
    "deepseek",
    "qwen",
    "qwen-cn",
    "glm",
    "glm-cn",
    "minimax",
    "minimax-cn",
    "ollama",
  ]);
  expect(getModelOptions("openai", "quick")[0]).toEqual(["GPT-5.4 Mini - Fast, strong coding and tool use", "gpt-5.4-mini"]);
  expect(getModelOptions("GLM-CN", "deep")[0]).toEqual(["GLM-5.1 - Latest flagship, 204K ctx", "glm-5.1"]);
  expect(getModelOptions("ollama", "quick").at(-1)).toEqual(["Custom model ID", "custom"]);
  expect(getModelOptions("minimax", "deep")).toBe(getModelOptions("minimax-cn", "deep"));
});

test("the catalog tables are frozen", () => {
  expect(Object.isFrozen(MODEL_OPTIONS)).toBe(true);
  expect(Object.isFrozen(MODEL_OPTIONS["openai"])).toBe(true);
  expect(Object.isFrozen(getModelOptions("openai", "quick"))).toBe(true);
  expect(Object.isFrozen(getModelOptions("openai", "quick")[0])).toBe(true);
  expect(Object.isFrozen(VALID_MODELS)).toBe(true);
  expect(Object.isFrozen(VALID_MODELS["openai"])).toBe(true);
});

test("getModelOptions rejects an unknown provider or mode", () => {
  expect(() => getModelOptions("nope", "quick")).toThrow(new RangeError("unknown model options: nope/quick"));
  expect(() => getModelOptions("openai", "nope")).toThrow(new RangeError("unknown model options: openai/nope"));
  expect(() => getModelOptions("toString", "quick")).toThrow(RangeError);
  expect(() => getModelOptions("openai", "constructor")).toThrow(RangeError);
});
