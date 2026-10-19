/**
 * Mirrors `tests/test_llm_clients.py`: the `api_key_env` test, the model catalog, the
 * `validate_model` tests, and the client tests, which build the model with `chatModelFromConfig`
 * and check its route and protocol in place of the LangChain client's fields. The Responses
 * reasoning round trip keeps only its `Bearer` half here; the round trip itself is the
 * `openai_responses` cassette tests'. Ollama's no-`think` default is checked on the built body.
 *
 * Elsewhere: the capability tests are `llm/capabilities.test.ts`'s; the Google no-temperature,
 * streaming-usage and normalization tests are covered by the protocol body and parse tests, and
 * the Anthropic early-close test by the transport tests. Left out: the Ollama `max_retries`
 * kwarg test and the normalize-reply tests (LangChain), and the DeepSeek non-streaming test (TS
 * streams DeepSeek).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test } from "bun:test";

import type { Env } from "../src/defaults.ts";
import { collect } from "../src/llm/events.ts";
import { Message, Request, TextPart } from "../src/llm/messages.ts";
import { AnthropicMessages } from "../src/llm/protocols/anthropic_messages.ts";
import { OllamaChat } from "../src/llm/protocols/ollama_chat.ts";
import { OpenAIChat } from "../src/llm/protocols/openai_chat.ts";
import { OpenAIResponses } from "../src/llm/protocols/openai_responses.ts";
import { getApiKeyEnv, OPTIONAL_API_KEY_PROVIDERS, PROVIDER_API_KEY_ENV } from "../src/llm_clients/api_key_env.ts";
import { AZURE_NEEDS_BASE_URL, LLMConfig, SUPPORTED_LLM_PROVIDERS } from "../src/llm_clients/config.ts";
import { chatModelFromConfig, DEDICATED_PROVIDERS, OPENAI_COMPATIBLE } from "../src/llm_clients/factory.ts";
import { getKnownModels, getModelOptions, MODEL_OPTIONS } from "../src/llm_clients/model_catalog.ts";
import { ValidationError } from "../src/llm_clients/provider_blocks.ts";
import { VALID_MODELS, validateModel } from "../src/llm_clients/validators.ts";
import { type Send, virtualScheduler } from "../src/platform.ts";

// Each test's environment: a private, empty store and only the variables the test sets.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-llm-clients-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Build `config` with this test's environment, collecting the warnings into `warnings`. */
function build(config: Record<string, unknown>, options: { send?: Send; warnings?: string[] } = {}) {
  const { send, warnings = [] } = options;
  return chatModelFromConfig(config, {
    env,
    send,
    scheduler: virtualScheduler(),
    onWarning: (message) => void warnings.push(message),
  });
}

const FACTORY_SUPPORTED_PROVIDERS = [...OPENAI_COMPATIBLE, ...DEDICATED_PROVIDERS];
const HI = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("hi")] })] });

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

test("claude mythos 5.1 is a known anthropic model", async () => {
  expect(validateModel("anthropic", "claude-mythos-5-1")).toBe(true);
  const warnings: string[] = [];
  const model = await build({ provider: "anthropic", model: "claude-mythos-5-1", api_key: "k" }, { warnings });
  expect(model.protocol).toBeInstanceOf(AnthropicMessages);
  expect(warnings).toEqual([]);
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

// --- azure --- //

test("azure endpoint base stripping", async () => {
  vars.set("AZURE_OPENAI_API_KEY", "x");
  vars.set("OPENAI_API_VERSION", "2024-02-01");
  // A portal's v1 surface is stripped back to the base resource URL.
  const model = await build({ provider: "azure", model: "gpt-4o", base_url: "https://res.openai.azure.com/openai/v1" });
  expect(model.route.baseUrl).toBe("https://res.openai.azure.com");
});

test("azure reads neither deployment nor endpoint from the environment", async () => {
  vars.set("AZURE_OPENAI_API_KEY", "x");
  vars.set("OPENAI_API_VERSION", "2024-02-01");
  vars.set("AZURE_OPENAI_ENDPOINT", "https://env-resource.openai.azure.com");
  vars.set("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deployment");
  for (const deployment of ["gpt-5.5", "gpt-4.1"]) {
    const model = await build({ provider: "azure", model: deployment, base_url: "https://res.openai.azure.com/" });
    expect(model.route.url()).toBe(
      `https://res.openai.azure.com/openai/deployments/${deployment}/chat/completions?api-version=2024-02-01`,
    );
  }
});

test("azure without base_url fails and says to set it", async () => {
  vars.set("AZURE_OPENAI_API_KEY", "x");
  vars.set("OPENAI_API_VERSION", "2024-02-01");
  vars.set("AZURE_OPENAI_ENDPOINT", "https://env-resource.openai.azure.com");
  const error = await build({ provider: "azure", model: "gpt-4o" }).catch((thrown: Error) => thrown);
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe(AZURE_NEEDS_BASE_URL);
  expect(AZURE_NEEDS_BASE_URL).toContain("llm_config.base_url");
});

// --- factory: dual-region providers --- //

test.each(FACTORY_SUPPORTED_PROVIDERS)("llm config accepts every factory supported provider (%s)", (provider) => {
  expect<string | null>(new LLMConfig({ provider }).provider).toBe(provider);
});

test("llm config and factory provider vocabularies match", () => {
  expect(new Set<string>(SUPPORTED_LLM_PROVIDERS)).toEqual(new Set(FACTORY_SUPPORTED_PROVIDERS));
});

test("llm config still rejects unknown provider", () => {
  expect(() => new LLMConfig({ provider: "not-a-provider" })).toThrow(ValidationError);
  expect(() => new LLMConfig({ provider: "not-a-provider" })).toThrow(/provider/);
});

const DUAL_REGION = ["qwen-cn", "glm-cn", "minimax", "minimax-cn"];

test.each(DUAL_REGION)("factory resolves new dual region providers (%s)", async (provider) => {
  vars.set(getApiKeyEnv(provider)!, "k");
  const model = await build({ provider, model: "some-model" });
  expect(model.protocol).toBeInstanceOf(OpenAIChat);
  expect(model.provider).toBe(provider);
});

// --- vLLM (OpenAI-compatible, optional key) --- //

test("factory resolves vllm as openai compatible", async () => {
  const model = await build({ provider: "vllm", model: "meta-llama/Llama-3.1-8B-Instruct" });
  expect(model.protocol).toBeInstanceOf(OpenAIChat);
  expect(model.provider).toBe("vllm");
});

test("vllm keyless sends dummy key and default base url", async () => {
  const model = await build({ provider: "vllm", model: "some-local-model" });
  expect(model.route.baseUrl).toBe("http://localhost:8000/v1");
  // Keyless gets the placeholder, no error.
  expect(await model.route.auth.headers()).toEqual({ Authorization: "Bearer EMPTY" });
});

test("vllm uses api key and base url when set", async () => {
  vars.set("VLLM_API_KEY", "secret-token");
  vars.set("VLLM_BASE_URL", "http://gpu-box:8001/v1");
  const model = await build({ provider: "vllm", model: "some-local-model" });
  expect(model.route.baseUrl).toBe("http://gpu-box:8001/v1");
  expect(await model.route.auth.headers()).toEqual({ Authorization: "Bearer secret-token" });
});

// --- Ollama (native route, no default think) --- //

test("factory routes ollama to native client", async () => {
  const model = await build({ provider: "ollama", model: "qwen3.5:35b" });
  expect(model.protocol).toBeInstanceOf(OllamaChat);
  expect(model.route.path).toBe("api/chat");
});

test("ollama sets no thinking default and strips v1", async () => {
  // An OpenAI-style base URL (with /v1) must be normalized to the native one.
  vars.set("OLLAMA_BASE_URL", "http://gpu-box:11434/v1");
  const model = await build({ provider: "ollama", model: "qwen3.5:35b" });
  // No default: `think` is sent only from the ollama block.
  expect(model.protocol.buildBody(HI)).not.toHaveProperty("think");
  expect(model.route.baseUrl).toBe("http://gpu-box:11434");
});

// --- OpenAI Responses --- //

test("openai responses send the key as a bearer token", async () => {
  const requests: Headers[] = [];
  const send: Send = async (_input, init) => {
    requests.push(new Headers(init.headers));
    return new Response(JSON.stringify({ error: { message: "refused" } }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  };
  const model = await build({ provider: "openai", model: "gpt-5.5", api_key: "synthetic-api-key" }, { send });
  await collect(model.stream(HI)).catch(() => {});
  expect(requests.map((headers) => headers.get("authorization"))).toEqual(["Bearer synthetic-api-key"]);
});

test.each([null, "https://gateway.example/v1"])(
  "native openai uses responses api with or without a base_url (%p)",
  async (baseUrl) => {
    const config: Record<string, unknown> = { provider: "openai", model: "gpt-5.5", api_key: "x" };
    if (baseUrl !== null) config["base_url"] = baseUrl;
    const model = await build(config);
    expect(model.protocol).toBeInstanceOf(OpenAIResponses);
    expect(model.route.url()).toBe(`${baseUrl ?? "https://api.openai.com/v1"}/responses`);
  },
);
