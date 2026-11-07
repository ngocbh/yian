/**
 * The resolver case table for `src/llm_clients/factory.ts`'s `chatModelFromConfig`.
 *
 * Each case gives a config, an optional saved login and an environment, and replays one golden
 * cassette from `cassettes/<protocol>/`, deep-copied and edited only where the case differs (the
 * URL, a header, the body). A case that fails names the error's class and exact message, with no
 * request sent. Every case sends the request the matching `*_cassettes.test.ts` sends for its
 * golden (`REQUESTS`). Later providers append cases and requests.
 *
 * Every case that sends also checks the credential superset (s2c's credential rule): each value
 * the route's auth sent, less vLLM's keyless `EMPTY`, is one `callCredentials` lists for the same
 * config, so the scrub knows every credential on the wire.
 *
 * Below the table: reserved block keys refused at build; no Anthropic block field reaching the
 * beta endpoint; and, with every SDK header and credential variable in the environment, only the
 * configured credential reaching the wire.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";

import type { Env } from "../../src/defaults.ts";
import { collect } from "../../src/llm/events.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import type { RouteModel } from "../../src/llm/executor.ts";
import { type Cassette, CASSETTE_REDACTED, CassetteReplay, IGNORED_HEADERS, loadCassette } from "../../src/llm/testing.ts";
import { ApiKeyCredentials, AuthError, isAuthFailure, OAuthCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { LLMConfig } from "../../src/llm_clients/config.ts";
import {
  type AgentConfig,
  callCredentials,
  type Connection,
  keyIdentity,
  legacyConnectionConfig,
  resolveModelConfig,
} from "../../src/llm_clients/connection.ts";
import { OPENAI_API_ENDPOINT, prefersResponses } from "../../src/llm_clients/endpoints.ts";
import { accountHash, chatModelFromConfig, warnOnce } from "../../src/llm_clients/factory.ts";
import { AnthropicMessagesBlock } from "../../src/llm_clients/provider_blocks.ts";
import { virtualScheduler } from "../../src/platform.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

const CASSETTES = fileURLToPath(new URL("../llm/cassettes/", import.meta.url));
// Far enough ahead that the build-time read never refreshes: the default refresh would use
// `OpenAIAuth`'s own transport, not the case's replay.
const FAR_FUTURE = 4_102_444_800;
const HOST_KEY = "test-host-key";
const SAVED_KEY = "test-saved-key";
const ENV_KEY = "test-env-key";
const ACCESS = "test-access-token";
const ACCOUNT = "test-account-1";
const AD_TOKEN = "test-ad-token-1";
const AZURE_ENDPOINT = "https://res.openai.azure.com";
const API_VERSION = "2025-03-01-preview";

const system = (text: string) => new Message({ role: "system", parts: [new TextPart(text)] });
const user = (text: string) => new Message({ role: "user", parts: [new TextPart(text)] });
const PLAIN = new Request({ messages: [system("You are terse."), user("Say hello.")] });
const TERSE_HI = new Request({ messages: [system("You are terse."), user("Hi")] });
const HI = new Request({ messages: [user("hi")] });

/** The request each golden's own cassette test sends, by `<protocol>/<cassette>`. */
const REQUESTS: Record<string, Request> = {
  "openai_responses/plain": PLAIN,
  "openai_responses/connection_settings": TERSE_HI,
  "openai_responses/subscription_plain": new Request({ messages: [system("You are terse."), user("Double 3.")] }),
  "openai_responses/subscription_no_system": new Request({ messages: [user("Double 3.")] }),
  "openai_responses/openrouter_codex": TERSE_HI,
  "openai_responses/vllm_codex": TERSE_HI,
  "openai_responses/azure_codex": TERSE_HI,
  "openai_chat/xai_plain": PLAIN,
  "openai_chat/xai_block_merge": HI,
  "openai_chat/connection_settings": HI,
  "openai_chat/deepseek_reasoner": PLAIN,
  "openai_chat/qwen_plain": PLAIN,
  "openai_chat/qwen_cn_plain": PLAIN,
  "openai_chat/glm_plain": PLAIN,
  "openai_chat/glm_cn_plain": PLAIN,
  "openai_chat/minimax_text01_plain": PLAIN,
  "openai_chat/minimax_m2": PLAIN,
  "openai_chat/minimax_block_merge": HI,
  "openai_chat/minimax_cn_plain": PLAIN,
  "openai_chat/openrouter_plain": PLAIN,
  "openai_chat/vllm_plain": PLAIN,
  "openai_chat/vllm_keyless": HI,
  "openai_chat/azure_plain": PLAIN,
  "openai_chat/azure_ad_token": HI,
  "openai_chat/azure_connection_settings": HI,
  "openai_chat/azure_block_merge": HI,
  "anthropic_messages/plain": PLAIN,
  "anthropic_messages/custom_base_url": PLAIN,
  "anthropic_messages/connection_settings": PLAIN,
  "anthropic_messages/block_sampling": PLAIN,
  "anthropic_messages/error_401": PLAIN,
  "gemini/plain": PLAIN,
  "gemini/custom_base_url": PLAIN,
  "gemini/block_merge": PLAIN,
  "gemini/error_400_api_key_invalid": PLAIN,
  "ollama_chat/plain": PLAIN,
  "ollama_chat/custom_base_url": PLAIN,
  "ollama_chat/block_merge": PLAIN,
};

/** A login saved into the case's store before the config is resolved. */
interface Login {
  /** Defaults to `openai`. */
  provider?: string;
  /** `null` for a keyless login. */
  credentials: ApiKeyCredentials | OAuthCredentials | null;
  model: string;
  /** The Azure resource endpoint. */
  endpoint?: string;
  organization?: string;
  project?: string;
}

/** What a config function may use: the case's environment and store. */
interface Setup {
  env: Env;
  store: AuthStore;
}

/** How the golden's one recorded request differs for the case. */
interface Edit {
  url?: string;
  /** Header values to set; `null` removes the header. */
  headers?: Record<string, string | null>;
  /** Top-level body fields to set. */
  json?: JsonObject;
  /** Top-level body fields to remove. */
  omit?: string[];
}

/** A model that builds and answers. */
interface Built {
  /** The credential headers sent, by lower-case name. */
  credential: Record<string, string>;
  provider: string;
  model: string;
  account: string | null;
  /** The `onWarning` texts; defaults to none. */
  warnings?: string[];
}

/** A model that fails at build or on its first request, before anything is sent. */
interface Failed {
  error: new (...args: never[]) => Error;
  message: string;
  at: "build" | "request";
}

interface Case {
  name: string;
  /** The config, or a function building it once `saved` is in the store. */
  config: AgentConfig | LLMConfig | ((setup: Setup) => AgentConfig | Promise<AgentConfig>);
  saved?: Login;
  env: Record<string, string>;
  /** `<protocol>/<cassette>`, also the key of the case's request in `REQUESTS`. */
  golden: string;
  edit?: Edit;
  /** Environment changes made after the build, before the request (failing cases only). */
  afterBuild?: Record<string, string>;
  expect: Built | Failed;
}

const OPENAI = { provider: "openai", model: "gpt-5.5" };
const SUBSCRIPTION: Login = {
  credentials: new OAuthCredentials({
    accessToken: ACCESS,
    refreshToken: "test-refresh-token",
    expiresAt: FAR_FUTURE,
    accountId: ACCOUNT,
    residency: "us",
  }),
  model: "gpt-5.5",
};
const SUBSCRIPTION_NO_RESIDENCY: Login = {
  ...SUBSCRIPTION,
  credentials: new OAuthCredentials({
    accessToken: ACCESS,
    refreshToken: "test-refresh-token",
    expiresAt: FAR_FUTURE,
    accountId: ACCOUNT,
  }),
};
const SAVED: Login = { credentials: new ApiKeyCredentials(SAVED_KEY), model: "gpt-5.5" };
const SAVED_SETTINGS: Login = {
  credentials: new ApiKeyCredentials(SAVED_KEY, { "X-Gateway": "gateway-1" }),
  model: "gpt-5.5",
  organization: "org-test",
  project: "proj-test",
};
const SETTINGS = { organization: "org-test", project: "proj-test", headers: { "X-Gateway": "gateway-1" } };

/** The built expectation of an OpenAI API-key route sending `key`. */
function openaiKey(key: string): Built {
  return { credential: { authorization: `Bearer ${key}` }, ...OPENAI, account: accountHash(key) };
}

const SUBSCRIBED: Built = {
  credential: { authorization: `Bearer ${ACCESS}`, "chatgpt-account-id": ACCOUNT },
  ...OPENAI,
  account: accountHash(ACCOUNT),
};

/** The unknown-model warning for `model` on `provider`. */
function unknownModel(provider: string, model: string): string {
  return `Model '${model}' is not in the known model list for provider '${provider}'. Continuing anyway.`;
}

/**
 * The built expectation of a route sending `credential`: no account unless given, and the
 * unknown-model warning when `warns`.
 */
function sends(
  provider: string,
  model: string,
  credential: Record<string, string>,
  options: { account?: string | null; warns?: boolean } = {},
): Built {
  const { account = null, warns = false } = options;
  return { credential, provider, model, account, warnings: warns ? [unknownModel(provider, model)] : [] };
}

const XAI = { provider: "xai", model: "grok-4" };
const VLLM = { provider: "vllm", model: "served-model" };
const AZURE = { provider: "azure", model: "my-deploy", base_url: `${AZURE_ENDPOINT}/` };
const AZURE_ENV = { OPENAI_API_VERSION: API_VERSION };
const VENDOR_HEADERS = { "x-gateway": "gateway-1", "x-team": "blue" };
const bearer = (key: string) => ({ authorization: `Bearer ${key}` });
const apiKey = (key: string) => ({ "api-key": key });

/** The missing-key error for `provider`'s key `variables`. */
function missingKey(provider: string, ...variables: string[]): Failed {
  return {
    error: RangeError,
    message:
      `API key for provider '${provider}' is not set. Please set the ${variables.join(" or ")} environment variable ` +
      `(e.g. add ${variables[0]}=your_key to your .env file).`,
    at: "build",
  };
}

/** The error for a block beside `model`, which `provider` routes to the Responses API. */
function blockOnResponses(provider: string, model: string): Failed {
  return {
    error: RangeError,
    message:
      `The '${provider}' block mirrors the Chat Completions request, but model '${model}' is routed to ` +
      "the Responses API; remove the block or use a model served through Chat Completions.",
    at: "build",
  };
}

const AZURE_NO_CREDENTIAL: Failed = {
  error: RangeError,
  message:
    "provider azure has no credential; set llm_config.api_key with llm_config.base_url, run " +
    "'yian auth azure', or set AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN",
  at: "build",
};
const AZURE_NO_VERSION: Failed = {
  error: RangeError,
  message: "provider azure needs OPENAI_API_VERSION, the Azure OpenAI API version (for example 2025-03-01-preview)",
  at: "build",
};

const ANTHROPIC = { provider: "anthropic", model: "claude-sonnet-4-5" };
const GOOGLE = { provider: "google", model: "gemini-2.5-flash" };
const OLLAMA = { provider: "ollama", model: "qwen3" };
const xApiKey = (key: string) => ({ "x-api-key": key });
const googKey = (key: string) => ({ "x-goog-api-key": key });

/** The built expectation of the Anthropic route sending `key`, whose account is the key's. */
function anthropicKey(key: string): Built {
  return sends("anthropic", "claude-sonnet-4-5", xApiKey(key), { account: accountHash(key) });
}

/** Resolve the default config against the saved subscription, then give it a host key. */
function hostKeyOnSubscription({ env }: Setup): AgentConfig {
  const config = resolveModelConfig({}, { env });
  expect((config["_connection"] as Connection).method).toBe("subscription");
  return { ...config, api_key: HOST_KEY };
}

const CASES: Case[] = [
  {
    name: "openai host key",
    config: { ...OPENAI, api_key: HOST_KEY },
    env: {},
    golden: "openai_responses/plain",
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai host key in an LLMConfig",
    config: new LLMConfig({ ...OPENAI, api_key: HOST_KEY }),
    env: {},
    golden: "openai_responses/plain",
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai saved key",
    config: {},
    saved: SAVED,
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: openaiKey(SAVED_KEY),
  },
  {
    name: "openai OPENAI_API_KEY",
    config: OPENAI,
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: openaiKey(ENV_KEY),
  },
  {
    name: "openai legacy record",
    config: legacyConnectionConfig(OPENAI),
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: openaiKey(ENV_KEY),
  },
  {
    name: "openai host key with base_url",
    config: { ...OPENAI, api_key: HOST_KEY, base_url: "https://gateway.example/v1" },
    env: { OPENAI_BASE_URL: "https://ignored.example/v1" },
    golden: "openai_responses/plain",
    edit: { url: "https://gateway.example/v1/responses" },
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai OPENAI_BASE_URL",
    config: OPENAI,
    env: { OPENAI_API_KEY: ENV_KEY, OPENAI_BASE_URL: "https://proxy.example/v1" },
    golden: "openai_responses/plain",
    edit: { url: "https://proxy.example/v1/responses" },
    expect: openaiKey(ENV_KEY),
  },
  {
    name: "openai host key with connection settings",
    config: { ...OPENAI, api_key: HOST_KEY, ...SETTINGS },
    env: {},
    golden: "openai_responses/connection_settings",
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai host key with a header value holding a tab",
    config: { ...OPENAI, api_key: HOST_KEY, ...SETTINGS, headers: { "X-Gateway": "gateway\t1" } },
    env: {},
    golden: "openai_responses/connection_settings",
    edit: { headers: { "x-gateway": "gateway\t1" } },
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai saved login with saved settings",
    config: {},
    saved: SAVED_SETTINGS,
    env: {},
    golden: "openai_responses/connection_settings",
    expect: openaiKey(SAVED_KEY),
  },
  {
    name: "openai host key with a block",
    config: { ...OPENAI, api_key: HOST_KEY, openai: { reasoning: { effort: "high" }, max_output_tokens: 4096n } },
    env: {},
    golden: "openai_responses/plain",
    edit: { json: { reasoning: { effort: "high" }, max_output_tokens: 4096n } },
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "subscription with residency",
    config: {},
    saved: SUBSCRIPTION,
    env: { OPENAI_API_KEY: ENV_KEY, OPENAI_BASE_URL: "https://ignored.example/v1" },
    golden: "openai_responses/subscription_plain",
    expect: SUBSCRIBED,
  },
  {
    name: "subscription without residency",
    config: {},
    saved: SUBSCRIPTION_NO_RESIDENCY,
    env: {},
    golden: "openai_responses/subscription_plain",
    edit: { headers: { "x-openai-internal-codex-residency": null } },
    expect: SUBSCRIBED,
  },
  {
    name: "subscription with no system message",
    config: OPENAI,
    saved: SUBSCRIPTION_NO_RESIDENCY,
    env: {},
    golden: "openai_responses/subscription_no_system",
    expect: SUBSCRIBED,
  },
  {
    name: "host key on a subscription connection, OPENAI_BASE_URL set",
    config: hostKeyOnSubscription,
    saved: SUBSCRIPTION,
    env: { OPENAI_BASE_URL: "https://proxy.example/v1" },
    golden: "openai_responses/plain",
    edit: { url: "https://proxy.example/v1/responses" },
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "host key on a subscription connection, OPENAI_BASE_URL empty",
    config: hostKeyOnSubscription,
    saved: SUBSCRIPTION,
    env: { OPENAI_BASE_URL: "" },
    golden: "openai_responses/plain",
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "host key on a subscription connection, OPENAI_BASE_URL unset",
    config: hostKeyOnSubscription,
    saved: SUBSCRIPTION,
    env: {},
    golden: "openai_responses/plain",
    expect: openaiKey(HOST_KEY),
  },
  {
    name: "openai missing key",
    config: OPENAI,
    env: {},
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message: "No OpenAI connection was selected; run 'yian auth openai' or set OPENAI_API_KEY before starting a new run",
      at: "build",
    },
  },
  {
    name: "openai environment key changed before the build",
    config: {
      ...OPENAI,
      _connection: {
        method: "api_key",
        source: "environment",
        credential_id: keyIdentity("test-old-key"),
        endpoint: OPENAI_API_ENDPOINT,
        store_path: null,
      },
    },
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message: "The OpenAI environment API key changed since this run started; start a new run",
      at: "build",
    },
  },
  {
    name: "openai host connection without its key",
    config: {
      ...OPENAI,
      _connection: {
        method: "api_key",
        source: "host",
        credential_id: null,
        endpoint: OPENAI_API_ENDPOINT,
        store_path: null,
      },
    },
    env: { OPENAI_API_KEY: ENV_KEY },
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message:
        "This run's openai API key came from the host's llm_config.api_key and is not recorded; " +
        "pass apiKey to advance",
      at: "build",
    },
  },
  {
    name: "openai environment key changed after the build",
    config: OPENAI,
    env: { OPENAI_API_KEY: ENV_KEY },
    afterBuild: { OPENAI_API_KEY: "test-other-key" },
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message: "The OpenAI environment API key changed since this run started; start a new run",
      at: "request",
    },
  },
  {
    name: "openai saved header names differ from the recorded ones",
    config: ({ env }) => {
      const config = resolveModelConfig({}, { env });
      (config["_connection"] as Connection).header_names = ["X-Other"];
      return config;
    },
    saved: { ...SAVED_SETTINGS, credentials: new ApiKeyCredentials(SAVED_KEY, { "X-Gateway": "gateway-secret" }) },
    env: {},
    golden: "openai_responses/connection_settings",
    expect: {
      error: AuthError,
      message:
        "The saved openai login's connection headers differ from the ones this run recorded; " +
        "run 'yian auth openai' and start a new run",
      at: "build",
    },
  },
  {
    name: "openai saved login saved again after resolving",
    config: async ({ env, store }) => {
      const config = resolveModelConfig({}, { env });
      await store.save("openai", new ApiKeyCredentials("test-new-key"), "gpt-5.5");
      return config;
    },
    saved: SAVED,
    env: {},
    golden: "openai_responses/plain",
    expect: {
      error: AuthError,
      message: "The saved connection changed. Run 'yian auth openai' and start a new run",
      at: "build",
    },
  },
  {
    name: "subscription refuses a temperature",
    config: { openai: { temperature: 0.2 } },
    saved: SUBSCRIPTION,
    env: {},
    golden: "openai_responses/subscription_plain",
    expect: {
      error: RangeError,
      message:
        "`openai.temperature` is not accepted by the ChatGPT subscription endpoint (a subset of the Responses " +
        "API); allowed: access_programs, max_tool_calls, metadata, moderation, prompt_cache_key, " +
        "prompt_cache_options, prompt_cache_retention, reasoning, safety_identifier, service_tier, text, " +
        "top_logprobs, top_p, truncation, user",
      at: "build",
    },
  },
  {
    name: "xai host key",
    config: { ...XAI, api_key: HOST_KEY },
    env: {},
    golden: "openai_chat/xai_plain",
    expect: sends("xai", "grok-4", bearer(HOST_KEY), { warns: true }),
  },
  {
    name: "xai saved key",
    config: {},
    saved: { provider: "xai", credentials: new ApiKeyCredentials(SAVED_KEY), model: "grok-4" },
    env: { XAI_API_KEY: ENV_KEY },
    golden: "openai_chat/xai_plain",
    expect: sends("xai", "grok-4", bearer(SAVED_KEY), { warns: true }),
  },
  {
    name: "xai XAI_API_KEY",
    config: XAI,
    env: { XAI_API_KEY: ENV_KEY },
    golden: "openai_chat/xai_plain",
    expect: sends("xai", "grok-4", bearer(ENV_KEY), { warns: true }),
  },
  {
    name: "xai host key with a block",
    config: {
      ...XAI,
      api_key: HOST_KEY,
      xai: { reasoning_effort: "high", max_completion_tokens: 100n, stop: ["END"], temperature: 0.2 },
    },
    env: {},
    golden: "openai_chat/xai_block_merge",
    expect: sends("xai", "grok-4", bearer(HOST_KEY), { warns: true }),
  },
  {
    name: "xai host key with headers",
    config: { ...XAI, api_key: HOST_KEY, headers: VENDOR_HEADERS },
    env: {},
    golden: "openai_chat/connection_settings",
    expect: sends("xai", "grok-4", bearer(HOST_KEY), { warns: true }),
  },
  {
    name: "xai saved login with saved headers",
    config: {},
    saved: { provider: "xai", credentials: new ApiKeyCredentials(SAVED_KEY, VENDOR_HEADERS), model: "grok-4" },
    env: {},
    golden: "openai_chat/connection_settings",
    expect: sends("xai", "grok-4", bearer(SAVED_KEY), { warns: true }),
  },
  {
    name: "xai host key with base_url",
    config: { ...XAI, api_key: HOST_KEY, base_url: "https://gateway.example/v1" },
    env: {},
    golden: "openai_chat/xai_plain",
    edit: { url: "https://gateway.example/v1/chat/completions" },
    expect: sends("xai", "grok-4", bearer(HOST_KEY), { warns: true }),
  },
  {
    name: "deepseek DEEPSEEK_API_KEY",
    config: { provider: "deepseek", model: "deepseek-reasoner" },
    env: { DEEPSEEK_API_KEY: ENV_KEY },
    golden: "openai_chat/deepseek_reasoner",
    expect: sends("deepseek", "deepseek-reasoner", bearer(ENV_KEY)),
  },
  {
    name: "qwen DASHSCOPE_API_KEY",
    config: { provider: "qwen", model: "qwen3-max" },
    env: { DASHSCOPE_API_KEY: ENV_KEY },
    golden: "openai_chat/qwen_plain",
    expect: sends("qwen", "qwen3-max", bearer(ENV_KEY), { warns: true }),
  },
  {
    name: "qwen-cn DASHSCOPE_CN_API_KEY",
    config: { provider: "qwen-cn", model: "qwen3-max" },
    env: { DASHSCOPE_CN_API_KEY: ENV_KEY, DASHSCOPE_API_KEY: "test-intl-key" },
    golden: "openai_chat/qwen_cn_plain",
    expect: sends("qwen-cn", "qwen3-max", bearer(ENV_KEY), { warns: true }),
  },
  {
    name: "glm ZHIPU_API_KEY",
    config: { provider: "glm", model: "glm-4.6" },
    env: { ZHIPU_API_KEY: ENV_KEY },
    golden: "openai_chat/glm_plain",
    expect: sends("glm", "glm-4.6", bearer(ENV_KEY), { warns: true }),
  },
  {
    name: "glm-cn ZHIPU_CN_API_KEY",
    config: { provider: "glm-cn", model: "glm-4.6" },
    env: { ZHIPU_CN_API_KEY: ENV_KEY },
    golden: "openai_chat/glm_cn_plain",
    expect: sends("glm-cn", "glm-4.6", bearer(ENV_KEY), { warns: true }),
  },
  {
    name: "minimax MiniMax-Text-01",
    config: { provider: "minimax", model: "MiniMax-Text-01" },
    env: { MINIMAX_API_KEY: ENV_KEY },
    golden: "openai_chat/minimax_text01_plain",
    expect: sends("minimax", "MiniMax-Text-01", bearer(ENV_KEY), { warns: true }),
  },
  {
    name: "minimax MiniMax-M2.7",
    config: { provider: "minimax", model: "MiniMax-M2.7" },
    env: { MINIMAX_API_KEY: ENV_KEY },
    golden: "openai_chat/minimax_m2",
    expect: sends("minimax", "MiniMax-M2.7", bearer(ENV_KEY)),
  },
  {
    name: "minimax with a block",
    config: {
      provider: "minimax",
      model: "MiniMax-M2.7",
      minimax: { thinking: { type: "adaptive" }, reasoning_effort: "high", max_tokens: 64n },
    },
    env: { MINIMAX_API_KEY: ENV_KEY },
    golden: "openai_chat/minimax_block_merge",
    expect: sends("minimax", "MiniMax-M2.7", bearer(ENV_KEY)),
  },
  {
    name: "minimax-cn MINIMAX_CN_API_KEY",
    config: { provider: "minimax-cn", model: "MiniMax-M2.7" },
    env: { MINIMAX_CN_API_KEY: ENV_KEY },
    golden: "openai_chat/minimax_cn_plain",
    expect: sends("minimax-cn", "MiniMax-M2.7", bearer(ENV_KEY)),
  },
  {
    name: "openrouter OPENROUTER_API_KEY",
    config: { provider: "openrouter", model: "anthropic/claude-sonnet-4.5" },
    env: { OPENROUTER_API_KEY: ENV_KEY },
    golden: "openai_chat/openrouter_plain",
    expect: sends("openrouter", "anthropic/claude-sonnet-4.5", bearer(ENV_KEY)),
  },
  {
    name: "openrouter openai/gpt-5-pro stays on Chat Completions",
    config: { provider: "openrouter", model: "openai/gpt-5-pro" },
    env: { OPENROUTER_API_KEY: ENV_KEY },
    golden: "openai_chat/openrouter_plain",
    edit: { json: { model: "openai/gpt-5-pro" } },
    expect: sends("openrouter", "openai/gpt-5-pro", bearer(ENV_KEY)),
  },
  {
    name: "vllm VLLM_API_KEY",
    config: VLLM,
    env: { VLLM_API_KEY: ENV_KEY },
    golden: "openai_chat/vllm_plain",
    expect: sends("vllm", "served-model", bearer(ENV_KEY)),
  },
  {
    name: "vllm keyless",
    config: VLLM,
    env: { VLLM_API_KEY: "" },
    golden: "openai_chat/vllm_keyless",
    expect: sends("vllm", "served-model", bearer("EMPTY")),
  },
  {
    name: "vllm keyless saved login",
    config: {},
    saved: { provider: "vllm", credentials: null, model: "served-model" },
    env: {},
    golden: "openai_chat/vllm_keyless",
    expect: sends("vllm", "served-model", bearer("EMPTY")),
  },
  {
    name: "vllm VLLM_BASE_URL",
    config: VLLM,
    env: { VLLM_API_KEY: ENV_KEY, VLLM_BASE_URL: "http://gpu-box:8000/v1" },
    golden: "openai_chat/vllm_plain",
    edit: { url: "http://gpu-box:8000/v1/chat/completions" },
    expect: sends("vllm", "served-model", bearer(ENV_KEY)),
  },
  {
    name: "vllm VLLM_BASE_URL empty",
    config: VLLM,
    env: { VLLM_API_KEY: ENV_KEY, VLLM_BASE_URL: "" },
    golden: "openai_chat/vllm_plain",
    expect: sends("vllm", "served-model", bearer(ENV_KEY)),
  },
  {
    name: "vllm config base_url beside VLLM_BASE_URL",
    config: { ...VLLM, api_key: HOST_KEY, base_url: "https://gateway.example/v1" },
    env: { VLLM_BASE_URL: "http://gpu-box:8000/v1" },
    golden: "openai_chat/vllm_plain",
    edit: { url: "https://gateway.example/v1/chat/completions" },
    expect: sends("vllm", "served-model", bearer(HOST_KEY)),
  },
  {
    name: "azure host key with base_url, the AD token ignored",
    config: { ...AZURE, api_key: HOST_KEY },
    env: { ...AZURE_ENV, AZURE_OPENAI_AD_TOKEN: AD_TOKEN, AZURE_OPENAI_API_KEY: ENV_KEY },
    golden: "openai_chat/azure_plain",
    expect: sends("azure", "my-deploy", apiKey(HOST_KEY)),
  },
  {
    name: "azure saved login",
    config: {},
    saved: {
      provider: "azure",
      credentials: new ApiKeyCredentials(SAVED_KEY),
      model: "my-deploy",
      endpoint: `${AZURE_ENDPOINT}/`,
    },
    env: { ...AZURE_ENV, AZURE_OPENAI_API_KEY: ENV_KEY },
    golden: "openai_chat/azure_plain",
    expect: sends("azure", "my-deploy", apiKey(SAVED_KEY)),
  },
  {
    name: "azure AZURE_OPENAI_API_KEY",
    config: AZURE,
    env: { ...AZURE_ENV, AZURE_OPENAI_API_KEY: ENV_KEY },
    golden: "openai_chat/azure_plain",
    expect: sends("azure", "my-deploy", apiKey(ENV_KEY)),
  },
  {
    name: "azure base_url ending /openai/v1/",
    config: { ...AZURE, base_url: `${AZURE_ENDPOINT}/openai/v1/` },
    env: { ...AZURE_ENV, AZURE_OPENAI_API_KEY: ENV_KEY },
    golden: "openai_chat/azure_plain",
    expect: sends("azure", "my-deploy", apiKey(ENV_KEY)),
  },
  {
    name: "azure deployment name percent-encoded in the path",
    config: { ...AZURE, model: "my deploy/1" },
    env: { ...AZURE_ENV, AZURE_OPENAI_API_KEY: ENV_KEY },
    golden: "openai_chat/azure_plain",
    edit: {
      url: `${AZURE_ENDPOINT}/openai/deployments/my%20deploy%2F1/chat/completions?api-version=${API_VERSION}`,
      json: { model: "my deploy/1" },
    },
    expect: sends("azure", "my deploy/1", apiKey(ENV_KEY)),
  },
  {
    name: "azure AD token wins over AZURE_OPENAI_API_KEY",
    config: AZURE,
    env: { ...AZURE_ENV, AZURE_OPENAI_AD_TOKEN: AD_TOKEN, AZURE_OPENAI_API_KEY: ENV_KEY },
    golden: "openai_chat/azure_ad_token",
    expect: sends("azure", "my-deploy", bearer(AD_TOKEN)),
  },
  {
    name: "azure empty AD token falls to AZURE_OPENAI_API_KEY",
    config: AZURE,
    env: { ...AZURE_ENV, AZURE_OPENAI_AD_TOKEN: "", AZURE_OPENAI_API_KEY: ENV_KEY },
    golden: "openai_chat/azure_plain",
    expect: sends("azure", "my-deploy", apiKey(ENV_KEY)),
  },
  {
    name: "azure host key with headers",
    config: { ...AZURE, api_key: HOST_KEY, headers: { "x-gateway": "gateway-1" } },
    env: AZURE_ENV,
    golden: "openai_chat/azure_connection_settings",
    expect: sends("azure", "my-deploy", apiKey(HOST_KEY)),
  },
  {
    name: "azure host key with a block",
    config: {
      ...AZURE,
      api_key: HOST_KEY,
      azure: {
        max_tokens: 100n,
        reasoning_effort: "low",
        verbosity: "low",
        user_security_context: { application_name: "a" },
      },
    },
    env: AZURE_ENV,
    golden: "openai_chat/azure_block_merge",
    expect: sends("azure", "my-deploy", apiKey(HOST_KEY)),
  },
  {
    name: "openrouter name-routed openai/gpt-5-codex",
    config: { provider: "openrouter", model: "openai/gpt-5-codex" },
    env: { OPENROUTER_API_KEY: ENV_KEY },
    golden: "openai_responses/openrouter_codex",
    expect: sends("openrouter", "openai/gpt-5-codex", bearer(ENV_KEY), { account: accountHash(ENV_KEY) }),
  },
  {
    name: "vllm name-routed my-codex-finetune, keyless",
    config: { provider: "vllm", model: "my-codex-finetune" },
    env: {},
    golden: "openai_responses/vllm_codex",
    expect: sends("vllm", "my-codex-finetune", bearer("EMPTY")),
  },
  {
    name: "azure name-routed gpt-5-codex, host key",
    config: { ...AZURE, model: "gpt-5-codex", api_key: HOST_KEY },
    env: AZURE_ENV,
    golden: "openai_responses/azure_codex",
    expect: sends("azure", "gpt-5-codex", apiKey(HOST_KEY), { account: accountHash(HOST_KEY) }),
  },
  {
    name: "azure name-routed gpt-5-codex, AD token",
    config: { ...AZURE, model: "gpt-5-codex" },
    env: { ...AZURE_ENV, AZURE_OPENAI_AD_TOKEN: AD_TOKEN },
    golden: "openai_responses/azure_codex",
    edit: { headers: { "api-key": null, authorization: "<redacted>" } },
    expect: sends("azure", "gpt-5-codex", bearer(AD_TOKEN), { account: accountHash(AZURE_ENDPOINT) }),
  },
  {
    name: "xai key missing",
    config: XAI,
    env: {},
    golden: "openai_chat/xai_plain",
    expect: missingKey("xai", "XAI_API_KEY"),
  },
  {
    name: "xai key empty",
    config: XAI,
    env: { XAI_API_KEY: "" },
    golden: "openai_chat/xai_plain",
    expect: missingKey("xai", "XAI_API_KEY"),
  },
  {
    name: "qwen-cn with only DASHSCOPE_API_KEY",
    config: { provider: "qwen-cn", model: "qwen3-max" },
    env: { DASHSCOPE_API_KEY: ENV_KEY },
    golden: "openai_chat/qwen_cn_plain",
    expect: missingKey("qwen-cn", "DASHSCOPE_CN_API_KEY"),
  },
  {
    name: "azure with no credential",
    config: AZURE,
    env: { ...AZURE_ENV, OPENAI_API_KEY: ENV_KEY },
    golden: "openai_chat/azure_plain",
    expect: AZURE_NO_CREDENTIAL,
  },
  {
    name: "azure with only an empty AD token",
    config: AZURE,
    env: { ...AZURE_ENV, AZURE_OPENAI_AD_TOKEN: "" },
    golden: "openai_chat/azure_plain",
    expect: AZURE_NO_CREDENTIAL,
  },
  {
    name: "azure with an empty AZURE_OPENAI_API_KEY",
    config: AZURE,
    env: { ...AZURE_ENV, AZURE_OPENAI_API_KEY: "" },
    golden: "openai_chat/azure_plain",
    expect: AZURE_NO_CREDENTIAL,
  },
  {
    name: "azure with no credential and no OPENAI_API_VERSION",
    config: AZURE,
    env: {},
    golden: "openai_chat/azure_plain",
    expect: AZURE_NO_CREDENTIAL,
  },
  {
    name: "azure OPENAI_API_VERSION unset",
    config: AZURE,
    env: { AZURE_OPENAI_API_KEY: ENV_KEY },
    golden: "openai_chat/azure_plain",
    expect: AZURE_NO_VERSION,
  },
  {
    name: "azure OPENAI_API_VERSION empty",
    config: { ...AZURE, api_key: HOST_KEY },
    env: { OPENAI_API_VERSION: "" },
    golden: "openai_chat/azure_plain",
    expect: AZURE_NO_VERSION,
  },
  {
    name: "azure name-routed deployment with a block",
    config: { ...AZURE, model: "gpt-5-codex", api_key: HOST_KEY, azure: { max_tokens: 100n } },
    env: AZURE_ENV,
    golden: "openai_responses/azure_codex",
    expect: blockOnResponses("azure", "gpt-5-codex"),
  },
  {
    name: "openrouter name-routed model with a block",
    config: { provider: "openrouter", model: "openai/gpt-5-codex", openrouter: { max_tokens: 64n } },
    env: { OPENROUTER_API_KEY: ENV_KEY },
    golden: "openai_responses/openrouter_codex",
    expect: blockOnResponses("openrouter", "openai/gpt-5-codex"),
  },
  {
    // The block is checked with the config (`checkModelConfig`), before the key is looked up.
    name: "openrouter name-routed model with a block and no key",
    config: { provider: "openrouter", model: "openai/gpt-5-codex", openrouter: { max_tokens: 64n } },
    env: {},
    golden: "openai_responses/openrouter_codex",
    expect: blockOnResponses("openrouter", "openai/gpt-5-codex"),
  },
  {
    // The block is checked with the config, before the API version is read.
    name: "azure name-routed deployment with a block and no OPENAI_API_VERSION",
    config: { ...AZURE, model: "gpt-5-codex", api_key: HOST_KEY, azure: { max_tokens: 100n } },
    env: {},
    golden: "openai_responses/azure_codex",
    expect: blockOnResponses("azure", "gpt-5-codex"),
  },
  {
    name: "vllm name-routed model with a block",
    config: { provider: "vllm", model: "my-codex-finetune", vllm: { max_tokens: 64n } },
    env: {},
    golden: "openai_responses/vllm_codex",
    expect: blockOnResponses("vllm", "my-codex-finetune"),
  },
  {
    name: "anthropic host key",
    config: { ...ANTHROPIC, api_key: HOST_KEY },
    env: { ANTHROPIC_API_KEY: ENV_KEY },
    golden: "anthropic_messages/plain",
    expect: anthropicKey(HOST_KEY),
  },
  {
    name: "anthropic saved key",
    config: {},
    saved: { provider: "anthropic", credentials: new ApiKeyCredentials(SAVED_KEY), model: "claude-sonnet-4-5" },
    env: { ANTHROPIC_API_KEY: ENV_KEY },
    golden: "anthropic_messages/plain",
    expect: anthropicKey(SAVED_KEY),
  },
  {
    name: "anthropic ANTHROPIC_API_KEY",
    config: ANTHROPIC,
    env: { ANTHROPIC_API_KEY: ENV_KEY },
    golden: "anthropic_messages/plain",
    expect: anthropicKey(ENV_KEY),
  },
  {
    name: "anthropic ANTHROPIC_BASE_URL",
    config: ANTHROPIC,
    env: { ANTHROPIC_API_KEY: ENV_KEY, ANTHROPIC_BASE_URL: "https://proxy.example/anthropic" },
    golden: "anthropic_messages/plain",
    edit: { url: "https://proxy.example/anthropic/v1/messages" },
    expect: anthropicKey(ENV_KEY),
  },
  {
    name: "anthropic ANTHROPIC_BASE_URL empty",
    config: ANTHROPIC,
    env: { ANTHROPIC_API_KEY: ENV_KEY, ANTHROPIC_BASE_URL: "" },
    golden: "anthropic_messages/plain",
    expect: anthropicKey(ENV_KEY),
  },
  {
    name: "anthropic config base_url beside ANTHROPIC_BASE_URL",
    config: { ...ANTHROPIC, api_key: HOST_KEY, base_url: "https://gateway.example.com/anthropic" },
    env: { ANTHROPIC_BASE_URL: "https://proxy.example/anthropic" },
    golden: "anthropic_messages/custom_base_url",
    expect: anthropicKey(HOST_KEY),
  },
  {
    name: "anthropic host key with headers",
    config: { ...ANTHROPIC, api_key: HOST_KEY, headers: { "X-Gateway": "team-a", "x-request-source": "yian" } },
    env: {},
    golden: "anthropic_messages/connection_settings",
    expect: anthropicKey(HOST_KEY),
  },
  {
    name: "anthropic host key with a block",
    config: { ...ANTHROPIC, api_key: HOST_KEY, anthropic: { temperature: 0.5, top_k: 40n } },
    env: {},
    golden: "anthropic_messages/block_sampling",
    expect: anthropicKey(HOST_KEY),
  },
  {
    name: "anthropic key missing",
    config: ANTHROPIC,
    env: { ANTHROPIC_API_KEY: "" },
    golden: "anthropic_messages/plain",
    expect: missingKey("anthropic", "ANTHROPIC_API_KEY"),
  },
  {
    name: "google host key",
    config: { ...GOOGLE, api_key: HOST_KEY },
    env: { GOOGLE_API_KEY: ENV_KEY },
    golden: "gemini/plain",
    expect: sends("google", "gemini-2.5-flash", googKey(HOST_KEY)),
  },
  {
    name: "google saved key",
    config: {},
    saved: { provider: "google", credentials: new ApiKeyCredentials(SAVED_KEY), model: "gemini-2.5-flash" },
    env: { GOOGLE_API_KEY: ENV_KEY },
    golden: "gemini/plain",
    expect: sends("google", "gemini-2.5-flash", googKey(SAVED_KEY)),
  },
  {
    name: "google GOOGLE_API_KEY before GEMINI_API_KEY",
    config: GOOGLE,
    env: { GOOGLE_API_KEY: ENV_KEY, GEMINI_API_KEY: "test-gemini-key" },
    golden: "gemini/plain",
    expect: sends("google", "gemini-2.5-flash", googKey(ENV_KEY)),
  },
  {
    name: "google only GEMINI_API_KEY",
    config: GOOGLE,
    env: { GEMINI_API_KEY: ENV_KEY },
    golden: "gemini/plain",
    expect: sends("google", "gemini-2.5-flash", googKey(ENV_KEY)),
  },
  {
    name: "google empty GOOGLE_API_KEY falls to GEMINI_API_KEY",
    config: GOOGLE,
    env: { GOOGLE_API_KEY: "", GEMINI_API_KEY: ENV_KEY },
    golden: "gemini/plain",
    expect: sends("google", "gemini-2.5-flash", googKey(ENV_KEY)),
  },
  {
    name: "google models/ prefix dropped",
    config: { ...GOOGLE, model: "models/gemini-2.5-flash" },
    env: { GOOGLE_API_KEY: ENV_KEY },
    golden: "gemini/plain",
    // The config's model as written is not a catalog id, so it warns.
    expect: {
      ...sends("google", "gemini-2.5-flash", googKey(ENV_KEY)),
      warnings: [unknownModel("google", "models/gemini-2.5-flash")],
    },
  },
  {
    name: "google tunedModels/ keeps its collection",
    config: { ...GOOGLE, model: "tunedModels/x" },
    env: { GOOGLE_API_KEY: ENV_KEY },
    golden: "gemini/plain",
    edit: {
      url: "https://generativelanguage.googleapis.com/v1beta/tunedModels/x:streamGenerateContent?alt=sse",
    },
    expect: sends("google", "tunedModels/x", googKey(ENV_KEY), { warns: true }),
  },
  {
    name: "google tunedModels/ id percent-encoded in the path",
    config: { ...GOOGLE, model: "tunedModels/a b/c" },
    env: { GOOGLE_API_KEY: ENV_KEY },
    golden: "gemini/plain",
    edit: {
      url: "https://generativelanguage.googleapis.com/v1beta/tunedModels/a%20b%2Fc:streamGenerateContent?alt=sse",
    },
    expect: sends("google", "tunedModels/a b/c", googKey(ENV_KEY), { warns: true }),
  },
  {
    name: "google models/ id percent-encoded in the path",
    config: { ...GOOGLE, model: "models/gemini x" },
    env: { GOOGLE_API_KEY: ENV_KEY },
    golden: "gemini/plain",
    edit: {
      url: "https://generativelanguage.googleapis.com/v1beta/models/gemini%20x:streamGenerateContent?alt=sse",
    },
    expect: {
      ...sends("google", "gemini x", googKey(ENV_KEY)),
      warnings: [unknownModel("google", "models/gemini x")],
    },
  },
  {
    name: "google config base_url is the API root",
    config: { ...GOOGLE, api_key: HOST_KEY, base_url: "https://gateway.example.com/gemini" },
    env: {},
    golden: "gemini/custom_base_url",
    expect: sends("google", "gemini-2.5-flash", googKey(HOST_KEY)),
  },
  {
    name: "google host key with a block",
    config: {
      ...GOOGLE,
      api_key: HOST_KEY,
      google: {
        generationConfig: {
          thinkingConfig: { thinkingBudget: 1024n, includeThoughts: true },
          maxOutputTokens: 512n,
          topK: 40n,
          temperature: 0.5,
          stopSequences: ["STOP"],
          topP: null,
        },
        safetySettings: [{ category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" }],
        serviceTier: "flex",
      },
    },
    env: {},
    golden: "gemini/block_merge",
    // The golden's `seed` is a field `checkModelConfig` refuses in a google block.
    edit: { omit: ["seed"] },
    expect: sends("google", "gemini-2.5-flash", googKey(HOST_KEY)),
  },
  {
    name: "google key missing names both key variables",
    config: GOOGLE,
    env: { GOOGLE_API_KEY: "", GEMINI_API_KEY: "" },
    golden: "gemini/plain",
    expect: missingKey("google", "GOOGLE_API_KEY", "GEMINI_API_KEY"),
  },
  {
    name: "ollama OLLAMA_BASE_URL unset",
    config: OLLAMA,
    env: {},
    golden: "ollama_chat/plain",
    expect: sends("ollama", "qwen3", {}),
  },
  {
    name: "ollama OLLAMA_BASE_URL empty",
    config: OLLAMA,
    env: { OLLAMA_BASE_URL: "" },
    golden: "ollama_chat/plain",
    expect: sends("ollama", "qwen3", {}),
  },
  {
    name: "ollama a config api_key and OLLAMA_API_KEY are never sent",
    config: { ...OLLAMA, api_key: HOST_KEY },
    env: { OLLAMA_API_KEY: ENV_KEY },
    golden: "ollama_chat/plain",
    expect: sends("ollama", "qwen3", {}),
  },
  {
    name: "ollama OLLAMA_BASE_URL LAN http",
    config: OLLAMA,
    env: { OLLAMA_BASE_URL: "http://gpu-box:11434" },
    golden: "ollama_chat/custom_base_url",
    expect: sends("ollama", "qwen3", {}),
  },
  {
    name: "ollama OLLAMA_BASE_URL with /v1 stripped",
    config: OLLAMA,
    env: { OLLAMA_BASE_URL: "http://localhost:11500/v1" },
    golden: "ollama_chat/custom_base_url",
    edit: { url: "http://localhost:11500/api/chat" },
    expect: sends("ollama", "qwen3", {}),
  },
  {
    name: "ollama OLLAMA_BASE_URL scheme-less",
    config: OLLAMA,
    env: { OLLAMA_BASE_URL: "gpu-box:11434" },
    golden: "ollama_chat/custom_base_url",
    expect: sends("ollama", "qwen3", {}),
  },
  {
    name: "ollama OLLAMA_BASE_URL userinfo sent as Basic auth",
    config: OLLAMA,
    env: { OLLAMA_BASE_URL: "http://u:test-pass-1@gpu-box:11434" },
    golden: "ollama_chat/custom_base_url",
    edit: { headers: { authorization: "<redacted>" } },
    expect: sends("ollama", "qwen3", { authorization: "Basic dTp0ZXN0LXBhc3MtMQ==" }),
  },
  {
    name: "ollama with a block",
    config: {
      ...OLLAMA,
      ollama: {
        options: { num_ctx: 8192n, temperature: 0.6, stop: ["X"], seed: null },
        keep_alive: -1n,
        think: false,
        logprobs: true,
        top_logprobs: 2n,
      },
    },
    env: {},
    golden: "ollama_chat/block_merge",
    expect: sends("ollama", "qwen3", {}),
  },
  {
    name: "ollama malformed OLLAMA_BASE_URL",
    config: OLLAMA,
    env: { OLLAMA_BASE_URL: "gpu-box" },
    golden: "ollama_chat/plain",
    expect: {
      error: RangeError,
      message:
        "OLLAMA_BASE_URL must be an http:// or https:// URL of a host, an optional port, and a path, " +
        "without a query or fragment",
      at: "build",
    },
  },
];

let home: string;

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-resolver-"));
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/**
 * Return the case's environment (`vars` and an auth file in `home`, changeable through the
 * returned map) and store, with `saved` saved into it.
 */
async function caseSetup(vars: Record<string, string>, saved?: Login): Promise<Setup & { vars: Map<string, string> }> {
  const map = new Map(Object.entries({ ...vars, YIAN_AUTH_FILE: join(home, "auth.json") }));
  const env: Env = (name) => map.get(name);
  const store = new AuthStore(null, { env });
  if (saved) {
    const { provider = "openai", credentials, model, endpoint, organization, project } = saved;
    await store.save(provider, credentials, model, { endpoint, organization, project });
  }
  return { env, store, vars: map };
}

/** Return the golden cassette, deep-copied, with `edit` applied to its one request. */
function editedGolden(golden: string, edit: Edit = {}): Cassette {
  const cassette = structuredClone(loadCassette(join(CASSETTES, `${golden}.json`)));
  expect(cassette.interactions.length).toBe(1);
  const request = cassette.interactions[0]!.request;
  if (edit.url !== undefined) request.url = edit.url;
  const headers = (request.headers ??= {});
  for (const [name, value] of Object.entries(edit.headers ?? {})) {
    if (value === null) delete headers[name];
    else headers[name] = value;
  }
  if (edit.json !== undefined) Object.assign(request.json as JsonObject, structuredClone(edit.json));
  for (const name of edit.omit ?? []) delete (request.json as JsonObject)[name];
  return cassette;
}

/** Replay `golden` checking only its URL (or `url`): any headers and any body match. */
function urlOnlyReplay(golden: string, url?: string): CassetteReplay {
  const cassette = editedGolden(golden, { url });
  const request = cassette.interactions[0]!.request;
  request.headers = {};
  request.json = CASSETTE_REDACTED;
  return new CassetteReplay(cassette);
}

/**
 * Check that every credential `model`'s auth sent, less vLLM's keyless `EMPTY`, is one
 * `callCredentials` lists for `config`; a failure counts the others, never showing them.
 */
function expectTracked(model: RouteModel, config: AgentConfig | LLMConfig, env: Env): void {
  const tracked = new Set(callCredentials(config, { env }));
  const sent = model.route.auth.credentials();
  expect(sent.filter((value) => value !== "EMPTY" && !tracked.has(value)).length).toBe(0);
}

describe("chatModelFromConfig", () => {
  for (const each of CASES) {
    test(each.name, async () => {
      const { env, store, vars } = await caseSetup(each.env, each.saved);
      const config = typeof each.config === "function" ? await each.config({ env, store }) : each.config;
      const cassette = editedGolden(each.golden, each.edit);
      const replay = new CassetteReplay(cassette);
      const warnings: string[] = [];
      const build = () =>
        chatModelFromConfig(config, {
          env,
          send: replay.send,
          scheduler: virtualScheduler(),
          onWarning: (message) => void warnings.push(message),
        });
      const request = REQUESTS[each.golden]!;
      const expected = each.expect;

      if ("error" in expected) {
        let outcome: Promise<unknown>;
        if (expected.at === "build") {
          outcome = build();
        } else {
          // Built outside the catch, so a build failure fails the case.
          const model = await build();
          for (const [name, value] of Object.entries(each.afterBuild ?? {})) vars.set(name, value);
          outcome = collect(model.stream(request));
        }
        const error = await outcome.then(
          () => null,
          (thrown: unknown) => thrown,
        );
        expect(error).toBeInstanceOf(expected.error);
        expect((error as Error).message).toBe(expected.message);
        expect(replay.requests.length).toBe(0);
        return;
      }

      const model = await build();
      await collect(model.stream(request));
      replay.assertDone();
      expect(model.provider).toBe(expected.provider);
      expect(model.model).toBe(expected.model);
      expect((model.protocol as { account?: string | null }).account ?? null).toBe(expected.account);
      expect(warnings).toEqual(expected.warnings ?? []);
      const sent = replay.requests[0]!.headers;
      for (const [name, value] of Object.entries(expected.credential)) expect(sent.get(name)).toBe(value);
      // CassetteReplay checks only the headers the cassette records; this checks nothing else was sent.
      const recorded = Object.keys(cassette.interactions[0]!.request.headers ?? {}).map((name) => name.toLowerCase());
      const names = [...sent.keys()].filter((name) => !IGNORED_HEADERS.has(name));
      expect(names.sort()).toEqual(recorded.filter((name) => !IGNORED_HEADERS.has(name)).sort());
      expectTracked(model, config, env);
    });
  }
});

describe("reserved block keys", () => {
  const CHAT = ["xai", "deepseek", "qwen", "qwen-cn", "glm", "glm-cn", "minimax", "minimax-cn", "openrouter", "vllm", "azure"];
  const ROWS: [provider: string, path: string][] = [
    ...CHAT.flatMap((provider): [string, string][] => [
      [provider, "stream"],
      [provider, "stream_options"],
    ]),
    ["minimax", "reasoning_split"],
    ...["stream", "messages", "system", "tools", "tool_choice", "output_config.format"].map(
      (path): [string, string] => ["anthropic", path],
    ),
    ...["tools", "toolConfig", "cachedContent", "generationConfig.responseJsonSchema"].map(
      (path): [string, string] => ["google", path],
    ),
    ["ollama", "stream"],
    ["ollama", "format"],
    ["openai", "stream"],
    ["openai", "text.format"],
  ];
  for (const [provider, path] of ROWS) {
    test(`${provider}.${path} is refused at build`, async () => {
      const { env } = await caseSetup({ OPENAI_API_VERSION: API_VERSION });
      // A dotted path nests: `text.format` gives `{text: {format: true}}`.
      const block = path.split(".").reduceRight<JsonValue>((value, key) => ({ [key]: value }), true);
      const config = { provider, model: "m", api_key: HOST_KEY, [provider]: block, ...(provider === "azure" ? { base_url: AZURE.base_url } : {}) };
      let sends = 0;
      const error = await chatModelFromConfig(config, {
        env,
        send: async () => {
          sends++;
          throw new Error("never sent");
        },
        onWarning: () => {},
      }).then(
        () => null,
        (thrown: unknown) => thrown,
      );
      expect(error).toBeInstanceOf(RangeError);
      expect((error as Error).message).toContain(`\`${provider}.${path}\` is built by yian (`);
      expect(sends).toBe(0);
    });
  }
});

describe("never beta", () => {
  // One value for every field the anthropic block accepts: `anthropic_messages_body.test.ts`'s
  // no-beta block plus the rest.
  const SAMPLE: JsonObject = {
    thinking: { type: "enabled", budget_tokens: 1024n },
    metadata: { user_id: "u" },
    output_config: { effort: "high" },
    service_tier: "auto",
    cache_control: { type: "ephemeral" },
    inference_geo: "us",
    max_tokens: 1024n,
    stop_sequences: ["END"],
    temperature: 0.5,
    top_k: 40n,
    top_p: 0.9,
  };

  test("the sample holds every field of the anthropic block", () => {
    expect(Object.keys(SAMPLE).sort()).toEqual(Object.keys(AnthropicMessagesBlock.fields).sort());
  });

  const blocks: [name: string, block: JsonObject][] = [
    ...Object.entries(SAMPLE).map(([key, value]): [string, JsonObject] => [key, { [key]: value }]),
    ["the whole sample", SAMPLE],
  ];
  for (const [name, block] of blocks) {
    test(`${name} goes to /v1/messages with no beta`, async () => {
      const { env } = await caseSetup({});
      const replay = urlOnlyReplay("anthropic_messages/plain");
      const model = await chatModelFromConfig(
        { ...ANTHROPIC, api_key: HOST_KEY, anthropic: block },
        { env, send: replay.send, scheduler: virtualScheduler() },
      );
      await collect(model.stream(PLAIN));
      replay.assertDone();
      const [sent] = replay.requests;
      expect(sent!.url).toBe("https://api.anthropic.com/v1/messages");
      expect(sent!.headers.has("anthropic-beta")).toBe(false);
    });
  }
});

describe("environment values never reach the wire", () => {
  // Every SDK header and credential variable, each value holding `env-`.
  const POISONED: Record<string, string> = {
    OPENAI_CUSTOM_HEADERS:
      "X-Env-Injected: from-env\nAuthorization: Bearer env-custom-auth\napi-key: env-custom-api-key\n" +
      "ChatGPT-Account-Id: env-account",
    OPENAI_ORG_ID: "env-org",
    OPENAI_ORGANIZATION: "env-org-2",
    OPENAI_PROJECT_ID: "env-project",
    AZURE_OPENAI_AD_TOKEN: "env-ad-token",
    OPENAI_API_VERSION: API_VERSION,
    ANTHROPIC_CUSTOM_HEADERS:
      "X-Env-Injected: from-env\nX-Api-Key: env-custom-x-api-key\nAuthorization: Bearer env-custom-anthropic-auth",
    ANTHROPIC_AUTH_TOKEN: "env-anthropic-token",
    OPENAI_ADMIN_KEY: "env-admin",
    GOOGLE_GENAI_DEFAULT_HEADERS: '{"X-Env-Injected": "from-env"}',
    GOOGLE_GENAI_ACCESS_TOKEN: "env-google-access-token",
    GOOGLE_GENAI_API_KEY: "env-google-genai-key",
  };

  /**
   * Build `config` in the poisoned environment plus `vars`, with `saved` in the store, send the
   * golden's request to its URL (or `url`), check the credential superset, and return the headers
   * sent.
   */
  async function sent(
    config: AgentConfig,
    golden: string,
    options: { vars?: Record<string, string>; saved?: Login; url?: string } = {},
  ): Promise<Headers> {
    const { env } = await caseSetup({ ...POISONED, ...options.vars }, options.saved);
    const replay = urlOnlyReplay(golden, options.url);
    const model = await chatModelFromConfig(config, {
      env,
      send: replay.send,
      scheduler: virtualScheduler(),
      onWarning: () => {},
    });
    await collect(model.stream(REQUESTS[golden]!));
    replay.assertDone();
    expectTracked(model, config, env);
    return replay.requests[0]!.headers;
  }

  /** Check that `headers` hold no `env-` value and none of the SDK-only headers; never shows a value. */
  function expectClean(headers: Headers): void {
    const leaked = [...headers].filter(([, value]) => value.toLowerCase().includes("env-")).map(([name]) => name);
    expect(leaked).toEqual([]);
    for (const name of ["openai-organization", "openai-project", "x-env-injected"]) expect(headers.has(name)).toBe(false);
  }

  test("openai_responses_send_only_the_explicit_key", async () => {
    const headers = await sent({ ...OPENAI, api_key: "sk-explicit" }, "openai_responses/plain");
    expect(headers.get("authorization")).toBe("Bearer sk-explicit");
    expectClean(headers);
  });

  test("openai_callable_key_sends_only_its_value", async () => {
    // A saved key is read per request.
    const saved = { credentials: new ApiKeyCredentials("sk-explicit"), model: "gpt-5.5" };
    const headers = await sent({}, "openai_responses/plain", { saved });
    expect(headers.get("authorization")).toBe("Bearer sk-explicit");
    expectClean(headers);
  });

  test("openai_compatible_provider_sends_only_the_explicit_key", async () => {
    const config = { provider: "deepseek", model: "deepseek-chat", api_key: "ds-explicit" };
    const headers = await sent(config, "openai_chat/deepseek_reasoner", { vars: { DEEPSEEK_API_KEY: "env-deepseek-key" } });
    expect(headers.get("authorization")).toBe("Bearer ds-explicit");
    expectClean(headers);
  });

  test("subscription_sends_only_its_own_headers", async () => {
    const headers = await sent({}, "openai_responses/subscription_plain", { saved: SUBSCRIPTION });
    expect(headers.get("authorization")).toBe(`Bearer ${ACCESS}`);
    expect(headers.get("chatgpt-account-id")).toBe(ACCOUNT);
    expectClean(headers);
  });

  test("azure_sends_only_the_explicit_key", async () => {
    const headers = await sent({ ...AZURE, api_key: "az-explicit" }, "openai_chat/azure_plain");
    expect(headers.get("api-key")).toBe("az-explicit");
    expect(headers.has("authorization")).toBe(false);
    expectClean(headers);
  });

  test("azure_without_a_key_keeps_the_sdk_order", async () => {
    // With no key, AZURE_OPENAI_AD_TOKEN goes first, as the SDK orders it; OPENAI_API_KEY never.
    const headers = await sent(AZURE, "openai_chat/azure_plain", { vars: { OPENAI_API_KEY: "sk-openai" } });
    expect(headers.get("authorization")).toBe("Bearer env-ad-token");
    expect(headers.has("api-key")).toBe(false);
    expect([...headers].filter(([, value]) => value.includes("sk-openai")).length).toBe(0);
    for (const name of ["openai-organization", "openai-project", "x-env-injected"]) expect(headers.has(name)).toBe(false);
  });

  /** Check the Anthropic headers carry only the explicit key and the API version. */
  function expectAnthropicKeyOnly(headers: Headers): void {
    expect(headers.get("x-api-key")).toBe("sk-ant-explicit");
    expect(headers.has("authorization")).toBe(false);
    expect(headers.has("anthropic-version")).toBe(true);
    expectClean(headers);
  }

  test("anthropic_sends_only_the_explicit_key", async () => {
    expectAnthropicKeyOnly(await sent({ ...ANTHROPIC, api_key: "sk-ant-explicit" }, "anthropic_messages/plain"));
  });

  test("anthropic_with_a_base_url_sends_only_the_explicit_key", async () => {
    const config = { ...ANTHROPIC, api_key: "sk-ant-explicit", base_url: "https://gateway.test" };
    expectAnthropicKeyOnly(await sent(config, "anthropic_messages/plain", { url: "https://gateway.test/v1/messages" }));
  });

  test("google_sends_only_the_explicit_key", async () => {
    const vars = { GOOGLE_API_KEY: "ENV-LEAK-1", GEMINI_API_KEY: "env-gemini-key" };
    const headers = await sent({ ...GOOGLE, api_key: "g-explicit" }, "gemini/plain", { vars });
    expect(headers.get("x-goog-api-key")).toBe("g-explicit");
    expectClean(headers);
  });

  test("google_without_any_key_fails_on_the_developer_api", async () => {
    // The Vertex AI switch is never read, so a keyless setup fails on the Developer API.
    const { env } = await caseSetup({ ...POISONED, GOOGLE_GENAI_USE_VERTEXAI: "true", GOOGLE_CLOUD_PROJECT: "p" });
    const error = await chatModelFromConfig(GOOGLE, { env, onWarning: () => {} }).then(
      () => null,
      (thrown: unknown) => thrown,
    );
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe(missingKey("google", "GOOGLE_API_KEY", "GEMINI_API_KEY").message);
  });

  test("ollama_does_not_send_the_environment_key", async () => {
    const headers = await sent(OLLAMA, "ollama_chat/plain", { vars: { OLLAMA_API_KEY: "env-ollama-key" } });
    expect(headers.has("authorization")).toBe(false);
    expectClean(headers);
  });

  test("ollama_keeps_userinfo_auth_from_its_base_url", async () => {
    const vars = { OLLAMA_API_KEY: "env-ollama-key", OLLAMA_BASE_URL: "http://user:pw@localhost:11434" };
    const headers = await sent(OLLAMA, "ollama_chat/plain", { vars });
    expect(headers.get("authorization")).toBe("Basic dXNlcjpwdw==");
  });
});

describe("anthropic connection headers", () => {
  const config = { ...ANTHROPIC, api_key: HOST_KEY, headers: { "X-Gateway": "gateway-secret-1" } };

  test("ride on the auth, leaving only the API version in the route headers", async () => {
    const { env } = await caseSetup({});
    const model = await chatModelFromConfig(config, { env });
    expect({ ...model.route.headers }).toEqual({ "anthropic-version": "2023-06-01" });
  });

  test("are scrubbed from an error body", async () => {
    const { env } = await caseSetup({});
    const cassette = editedGolden("anthropic_messages/error_401", { headers: { "x-gateway": "gateway-secret-1" } });
    const body = cassette.interactions[0]!.response.json as { error: { message: string } };
    body.error.message = "unknown gateway gateway-secret-1";
    const replay = new CassetteReplay(cassette);
    const model = await chatModelFromConfig(config, { env, send: replay.send, scheduler: virtualScheduler() });
    const error = await collect(model.stream(PLAIN)).then(
      () => null,
      (thrown: unknown) => thrown,
    );
    expect((error as Error).message).toBe("anthropic returned HTTP 401 (auth): unknown gateway [REDACTED]");
    replay.assertDone();
  });
});

describe("google auth failure", () => {
  test("a 400 naming API_KEY_INVALID only in its details is an auth failure", async () => {
    const { env } = await caseSetup({});
    const replay = new CassetteReplay(editedGolden("gemini/error_400_api_key_invalid"));
    const model = await chatModelFromConfig(
      { ...GOOGLE, api_key: HOST_KEY },
      { env, send: replay.send, scheduler: virtualScheduler() },
    );
    const error = await collect(model.stream(PLAIN)).then(
      () => null,
      (thrown: unknown) => thrown,
    );
    expect(error).not.toBeNull();
    expect(isAuthFailure(error)).toBe(true);
    replay.assertDone();
  });
});

describe("unknown-model warning", () => {
  /** Build `config` (sending nothing) and return the `onWarning` texts. */
  async function warningsFor(config: AgentConfig, vars: Record<string, string>, saved?: Login): Promise<string[]> {
    const { env } = await caseSetup(vars, saved);
    const warnings: string[] = [];
    await chatModelFromConfig(config, { env, onWarning: (message) => void warnings.push(message) });
    return warnings;
  }

  test("an unknown xai model warns that it isn't in the known model list", async () => {
    const warnings = await warningsFor({ provider: "xai", model: "grok-resolver-unknown" }, { XAI_API_KEY: ENV_KEY });
    expect(warnings).toEqual([
      "Model 'grok-resolver-unknown' is not in the known model list for provider 'xai'. Continuing anyway.",
    ]);
  });

  test("a known model doesn't warn", async () => {
    expect(await warningsFor({ provider: "google", model: "gemini-2.5-pro" }, { GOOGLE_API_KEY: ENV_KEY })).toEqual([]);
  });

  const quiet: [string, AgentConfig, Record<string, string>][] = [
    ["azure", { ...AZURE, model: "resolver-unknown-deploy" }, { ...AZURE_ENV, AZURE_OPENAI_API_KEY: ENV_KEY }],
    ["openrouter", { provider: "openrouter", model: "resolver/unknown" }, { OPENROUTER_API_KEY: ENV_KEY }],
    ["vllm", { provider: "vllm", model: "resolver-unknown" }, {}],
    ["ollama", { provider: "ollama", model: "resolver-unknown" }, {}],
  ];
  for (const [provider, config, vars] of quiet) {
    test(`${provider} never warns`, async () => expect(await warningsFor(config, vars)).toEqual([]));
  }

  test("the subscription never warns", async () => {
    const config = { provider: "openai", model: "resolver-unknown-gpt" };
    expect(await warningsFor(config, {}, { ...SUBSCRIPTION, model: "resolver-unknown-gpt" })).toEqual([]);
  });

  test("the default onWarning emits each text once", async () => {
    const emit = spyOn(process, "emitWarning").mockImplementation(() => {});
    try {
      const { env } = await caseSetup({ XAI_API_KEY: ENV_KEY });
      // A fresh name per run: the default's seen-set lives for the whole process.
      const model = `grok-resolver-default-warning-${crypto.randomUUID()}`;
      const config = { provider: "xai", model };
      await chatModelFromConfig(config, { env });
      await chatModelFromConfig(config, { env });
      expect(emit.mock.calls.map((args) => [...args])).toEqual([[unknownModel("xai", model), "RuntimeWarning"]]);
    } finally {
      emit.mockRestore();
    }
  });
});

describe("warnOnce", () => {
  test("emits a text only when the seen-set lacks it, then adds it", () => {
    const emit = spyOn(process, "emitWarning").mockImplementation(() => {});
    try {
      const seen = new Set<string>(["already seen"]);
      warnOnce(seen, "first");
      warnOnce(seen, "first");
      warnOnce(seen, "already seen");
      warnOnce(seen, "second");
      expect(emit.mock.calls.map((args) => [...args])).toEqual([
        ["first", "RuntimeWarning"],
        ["second", "RuntimeWarning"],
      ]);
      expect([...seen]).toEqual(["already seen", "first", "second"]);
    } finally {
      emit.mockRestore();
    }
  });
});

describe("prefersResponses", () => {
  const routed = [
    "gpt-5-pro",
    "gpt-5-pro-2025-10-06",
    "gpt-5.2-pro",
    "gpt-5.4-pro",
    "gpt-5.5-pro",
    "gpt-5.6-sol",
    "codex-mini",
    "gpt-5-codex",
    "my-codex-finetune",
  ];
  for (const model of routed) test(`routes ${model} to Responses`, () => expect(prefersResponses(model)).toBe(true));
  for (const model of ["gpt-5", "gpt-4o", "openai/gpt-5-pro", "gpt-5.3-pro", "Codex"]) {
    test(`keeps ${model} on Chat Completions`, () => expect(prefersResponses(model)).toBe(false));
  }
});

describe("accountHash", () => {
  test("is the first 16 hex digits of the SHA-256", () => {
    expect(accountHash("abc")).toBe("ba7816bf8f01cfea");
  });
});
