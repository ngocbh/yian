/**
 * The provider key variables, the model catalog, `isKnownModel`, and the clients, built with
 * `chatModelFromConfig` and checked by their route, protocol and body.
 *
 * Also the `@ngocbh/yian/llm_clients` entry, imported by package self-reference, which resolves
 * through `package.json`'s "exports" as an installed copy does; with `YIAN_TEST_NODE` set to a
 * Node binary, also on Node, built.
 */

import { spawnSync } from "node:child_process";
import { mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import type { Env } from "../../src/defaults.ts";
import { collect } from "../../src/llm/events.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { AnthropicMessages } from "../../src/llm/protocols/anthropic_messages.ts";
import { OllamaChat } from "../../src/llm/protocols/ollama_chat.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { OpenAIResponses } from "../../src/llm/protocols/openai_responses.ts";
import {
  apiKeyEnvNames,
  getApiKeyEnv,
  OPTIONAL_API_KEY_PROVIDERS,
  PROVIDER_API_KEY_ENV,
} from "../../src/llm_clients/api_key_env.ts";
import {
  ApiKeyCredentials,
  AuthError,
  HttpResponse,
  OAuthCredentials,
  ProviderSelection,
} from "../../src/llm_clients/auth/common.ts";
import { OpenAIAuth } from "../../src/llm_clients/auth/openai.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import {
  AZURE_NEEDS_BASE_URL,
  azureResourceEndpoint,
  LLM_CONFIG_KEYS,
  LLMConfig,
  SUPPORTED_LLM_PROVIDERS,
} from "../../src/llm_clients/config.ts";
import { OPENAI_COMPATIBLE } from "../../src/llm_clients/endpoints.ts";
import { chatModelFromConfig, chatModelTarget, DEDICATED_PROVIDERS } from "../../src/llm_clients/factory.ts";
import { getKnownModels, getModelOptions, isKnownModel, MODEL_OPTIONS } from "../../src/llm_clients/model_catalog.ts";
import { listModels, ModelListError } from "../../src/llm_clients/model_list.ts";
import { ValidationError } from "../../src/llm_clients/provider_blocks.ts";
import { type Send, virtualScheduler } from "../../src/platform.ts";
import { testNode } from "../helpers/test_node.ts";

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

test("apiKeyEnvNames lists a provider's key variables in the order its client reads them", () => {
  expect(apiKeyEnvNames("google")).toEqual(["GOOGLE_API_KEY", "GEMINI_API_KEY"]);
  expect(apiKeyEnvNames("Google")).toEqual(["GOOGLE_API_KEY", "GEMINI_API_KEY"]);
  expect(apiKeyEnvNames("openai")).toEqual(["OPENAI_API_KEY"]);
  expect(apiKeyEnvNames("ollama")).toEqual([]);
  expect(apiKeyEnvNames("no-such-provider")).toEqual([]);
  expect(apiKeyEnvNames("constructor")).toEqual([]);
});

test("vllm is the only provider whose key is optional", () => {
  expect(OPTIONAL_API_KEY_PROVIDERS).toEqual(new Set(["vllm"]));
});

test("vllm accepts any model name", () => {
  expect(isKnownModel("vllm", "any/arbitrary-served-model")).toBe(true);
});

test("ollama and openrouter accept any model name", () => {
  expect(isKnownModel("ollama", "any/arbitrary-served-model")).toBe(true);
  expect(isKnownModel("openrouter", "any/arbitrary-served-model")).toBe(true);
});

test("claude mythos 5.1 is a known anthropic model", async () => {
  expect(isKnownModel("anthropic", "claude-mythos-5-1")).toBe(true);
  const warnings: string[] = [];
  const model = await build({ provider: "anthropic", model: "claude-mythos-5-1", api_key: "k" }, { warnings });
  expect(model.protocol).toBeInstanceOf(AnthropicMessages);
  expect(warnings).toEqual([]);
});

test("isKnownModel accepts any model for providers outside the catalog", () => {
  expect(isKnownModel("azure", "my-deployment")).toBe(true);
  expect(isKnownModel("no-such-provider", "anything")).toBe(true);
  expect(isKnownModel("toString", "anything")).toBe(true);
  expect(isKnownModel("__proto__", "anything")).toBe(true);
});

test("isKnownModel lowercases the provider but not the model", () => {
  expect(isKnownModel("OpenAI", "gpt-5.5")).toBe(true);
  expect(isKnownModel("openai", "GPT-5.5")).toBe(false);
  expect(isKnownModel("openai", "no-such-model")).toBe(false);
  expect(isKnownModel("VLLM", "anything")).toBe(true);
});

test("the picker's custom sentinel is not a known model", () => {
  expect(isKnownModel("glm", "custom")).toBe(false);
  expect(isKnownModel("deepseek", "custom")).toBe(false);
  expect(isKnownModel("glm-cn", "custom")).toBe(false);
  expect(isKnownModel("openai", "custom")).toBe(false);
  expect(isKnownModel("anthropic", "custom")).toBe(false);
});

test("isKnownModel knows every catalog model of every catalog provider", () => {
  for (const [provider, models] of Object.entries(getKnownModels())) {
    for (const model of models) expect(isKnownModel(provider, model)).toBe(true);
    if (provider !== "ollama") expect(isKnownModel(provider, "no-such-model")).toBe(false);
  }
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
  // MiniMax-M2.7-highspeed is in both quick and deep but listed once; the custom sentinel isn't a model.
  expect(known["minimax"]!).toEqual([
    "MiniMax-M2",
    "MiniMax-M2.1",
    "MiniMax-M2.1-highspeed",
    "MiniMax-M2.5",
    "MiniMax-M2.5-highspeed",
    "MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
  ]);
});

test("the catalog's providers, labels and order", () => {
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
  expect(FACTORY_SUPPORTED_PROVIDERS).toHaveLength(SUPPORTED_LLM_PROVIDERS.length);
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

test("structured_output reaches the built model and its credential-free target, native when unset", async () => {
  for (const [config, expected] of [
    [{ provider: "vllm", model: "m", structured_output: "prompt" }, "prompt"],
    [{ provider: "anthropic", model: "claude-opus-4-8", api_key: "k", structured_output: "native" }, "native"],
    [{ provider: "vllm", model: "m" }, "native"],
  ] as const) {
    expect((await build(config)).structuredOutput).toBe(expected);
    expect(chatModelTarget(config, { env }).structuredOutput).toBe(expected);
  }
});

test("structured_output takes only native or prompt, and dumps when set", () => {
  expect(new LLMConfig({ structured_output: "prompt" }).dump()).toEqual({
    provider: null,
    model: null,
    structured_output: "prompt",
  });
  expect(new LLMConfig({}).structuredOutput).toBeNull();
  for (const value of ["tools", 1n, true]) {
    expect(() => new LLMConfig({ structured_output: value })).toThrow("Input should be 'native' or 'prompt'");
  }
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

// --- api: Responses or Chat Completions --- //

describe("llm_config.api", () => {
  const AZURE = "https://res.openai.azure.com/";

  /**
   * Build `config` with keys for vLLM, DeepSeek and Azure, returning its route's URL and protocol;
   * the structured target read off the config, unbuilt, names the same protocol.
   */
  async function routed(config: Record<string, unknown>): Promise<[string, string]> {
    vars.set("DEEPSEEK_API_KEY", "k");
    vars.set("AZURE_OPENAI_API_KEY", "k");
    vars.set("OPENAI_API_VERSION", "2025-03-01-preview");
    const model = await build(config);
    expect(chatModelTarget(config, { env }).protocol).toBe(model.protocol.name);
    return [model.route.url(), model.protocol.constructor.name];
  }

  test("unset, the model's name picks the API", async () => {
    expect(await routed({ provider: "vllm", model: "gpt-5-codex" })).toEqual([
      "http://localhost:8000/v1/responses",
      "OpenAIResponses",
    ]);
    expect(await routed({ provider: "azure", model: "pro-east", base_url: AZURE })).toEqual([
      "https://res.openai.azure.com/openai/deployments/pro-east/chat/completions?api-version=2025-03-01-preview",
      "OpenAIChat",
    ]);
  });

  test("set, it picks the API whatever the model's name", async () => {
    expect(await routed({ provider: "vllm", model: "gpt-5-codex", api: "chat" })).toEqual([
      "http://localhost:8000/v1/chat/completions",
      "OpenAIChat",
    ]);
    expect(await routed({ provider: "deepseek", model: "deepseek-chat", api: "responses" })).toEqual([
      "https://api.deepseek.com/responses",
      "OpenAIResponses",
    ]);
    expect(await routed({ provider: "azure", model: "pro-east", base_url: AZURE, api: "responses" })).toEqual([
      "https://res.openai.azure.com/openai/responses?api-version=2025-03-01-preview",
      "OpenAIResponses",
    ]);
    expect(await routed({ provider: "azure", model: "my-codex", base_url: AZURE, api: "chat" })).toEqual([
      "https://res.openai.azure.com/openai/deployments/my-codex/chat/completions?api-version=2025-03-01-preview",
      "OpenAIChat",
    ]);
  });

  test("a block reaches a model the name would send to Responses when api is chat", async () => {
    const config = { provider: "vllm", model: "gpt-5-codex", api: "chat", vllm: { top_k: 5n } };
    const model = await build(config);
    expect(model.protocol).toBeInstanceOf(OpenAIChat);
  });

  test("is responses or chat, kept in the dump when set", () => {
    expect(new LLMConfig({ provider: "vllm", api: "chat" }).api).toBe("chat");
    expect(new LLMConfig({ provider: "azure", api: "responses" }).dump()).toEqual({
      provider: "azure",
      model: null,
      api: "responses",
    });
    expect(new LLMConfig({ api: null }).api).toBeNull();
    expect(new LLMConfig({}).dump()).toEqual({ provider: null, model: null });
    for (const api of ["Chat", "completions", 1n]) {
      const error = (() => {
        try {
          new LLMConfig({ api });
        } catch (thrown) {
          return thrown as ValidationError;
        }
        throw new Error("accepted");
      })();
      expect(error.issues.map(({ loc, msg }) => ({ loc, msg }))).toEqual([
        { loc: ["api"], msg: 'Input should be "responses" or "chat"' },
      ]);
    }
  });

  test.each(["openai", "anthropic", "google", "ollama"])("is refused beside provider %s", (provider) => {
    let error: ValidationError | null = null;
    try {
      new LLMConfig({ provider, api: "responses" });
    } catch (thrown) {
      error = thrown as ValidationError;
    }
    expect(error!.issues.map(({ loc, error }) => ({ loc, error }))).toEqual([
      {
        loc: ["api"],
        error:
          "api applies only to provider azure and the Chat Completions providers " +
          "(xai, deepseek, qwen, qwen-cn, glm, glm-cn, minimax, minimax-cn, openrouter, vllm)",
      },
    ]);
  });
});

const ROOT = join(import.meta.dirname, "..", "..");
const BUILT = join(ROOT, "dist/src/llm_clients/index.js");
/**
 * The public entry. It resolves by self-reference from inside ROOT (this file, and children whose
 * cwd is ROOT). Typed string so tsc doesn't resolve it to dist/, missing before a build.
 */
const ENTRY: string = "@ngocbh/yian/llm_clients";

const NODE = testNode(BUILT);

describe("the @ngocbh/yian/llm_clients entry", () => {
  test.if(process.versions.bun !== undefined)("on Bun it is the source module yian itself uses", async () => {
    expect(realpathSync(fileURLToPath(import.meta.resolve(ENTRY)))).toBe(
      realpathSync(join(ROOT, "src/llm_clients/index.ts")),
    );
    const entry = (await import(ENTRY)) as typeof import("../../src/llm_clients/index.ts");
    expect(entry.LLMConfig).toBe(LLMConfig);
    expect(entry.LLM_CONFIG_KEYS).toBe(LLM_CONFIG_KEYS);
    expect(entry.SUPPORTED_LLM_PROVIDERS).toBe(SUPPORTED_LLM_PROVIDERS);
    expect(entry.ValidationError).toBe(ValidationError);
    expect(entry.azureResourceEndpoint).toBe(azureResourceEndpoint);
    expect(entry.listModels).toBe(listModels);
    expect(entry.ModelListError).toBe(ModelListError);
    expect(entry.AuthStore).toBe(AuthStore);
    expect(entry.OpenAIAuth).toBe(OpenAIAuth);
    expect(entry.AuthError).toBe(AuthError);
    expect(entry.HttpResponse).toBe(HttpResponse);
    expect(entry.OAuthCredentials).toBe(OAuthCredentials);
    expect(entry.ApiKeyCredentials).toBe(ApiKeyCredentials);
    expect(entry.ProviderSelection).toBe(ProviderSelection);
  });

  test.skipIf(!NODE)(
    "on Node it resolves to the build and runs listing, validation and statuses",
    () => {
      const script = `
const entry = await import(${JSON.stringify(ENTRY)});
const names = [
  "LLMConfig", "LLM_CONFIG_KEYS", "SUPPORTED_LLM_PROVIDERS", "ValidationError", "azureResourceEndpoint",
  "listModels", "ModelListError", "AuthStore", "OpenAIAuth", "AuthError", "HttpResponse",
  "OAuthCredentials", "ApiKeyCredentials", "ProviderSelection",
];
const types = Object.fromEntries(names.map((n) => [n, typeof entry[n]]));
let listing = "listed";
try {
  await entry.listModels("openai", { apiKey: "sk-entry-test", request: async () => new entry.HttpResponse(401, null) });
} catch (error) {
  listing = { modelListError: error instanceof entry.ModelListError, status: error.status };
}
const config = entry.LLMConfig.fromHost({ provider: "anthropic", api_key: "sk-entry-test", anthropic: { max_tokens: 1024 } });
let invalid = "accepted";
try {
  new entry.LLMConfig({ provider: "provider-entry-test" });
} catch (error) {
  invalid = { validationError: error instanceof entry.ValidationError, named: error.message.includes("provider-entry-test") };
}
console.log(JSON.stringify({
  resolved: import.meta.resolve(${JSON.stringify(ENTRY)}),
  types,
  listing,
  config: [config.provider, config.blockFor("anthropic").max_tokens === 1024n],
  invalid,
}));
`;
      const result = spawnSync(NODE!, ["--input-type=module", "-e", script], {
        cwd: ROOT,
        encoding: "utf8",
        timeout: 60_000,
      });
      expect(result.status, `signal: ${result.signal}\nstderr: ${result.stderr}`).toBe(0);
      expect(result.stderr).toBe("");
      const { resolved, types, listing, config, invalid } = JSON.parse(result.stdout);
      expect(realpathSync(fileURLToPath(resolved))).toBe(realpathSync(BUILT));
      expect(types).toEqual({
        LLMConfig: "function",
        LLM_CONFIG_KEYS: "object",
        SUPPORTED_LLM_PROVIDERS: "object",
        ValidationError: "function",
        azureResourceEndpoint: "function",
        listModels: "function",
        ModelListError: "function",
        AuthStore: "function",
        OpenAIAuth: "function",
        AuthError: "function",
        HttpResponse: "function",
        OAuthCredentials: "function",
        ApiKeyCredentials: "function",
        ProviderSelection: "function",
      });
      expect(listing).toEqual({ modelListError: true, status: 401 });
      expect(config).toEqual(["anthropic", true]);
      expect(invalid).toEqual({ validationError: true, named: false });
    },
    90_000,
  );
});
