/**
 * The resolver case table for `src/llm_clients/factory.ts`'s `chatModelFromConfig`: the planned
 * case table of #366's resolver stages (`tests/llm/resolver_cases.py` and `test_resolver*.py` in
 * Python, never built), so it is a new file rather than a port of one.
 *
 * Each case gives a config, an optional saved login and an environment, and replays one golden
 * cassette from `cassettes/<protocol>/`, deep-copied and edited only where the case differs (the
 * URL, a header, the body). A case that fails names the error's class and exact message, with no
 * request sent. Every case sends the request the matching `*_cassettes.test.ts` sends for its
 * golden (`REQUESTS`). Later providers append cases and requests.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import type { Env } from "../../src/defaults.ts";
import { collect } from "../../src/llm/events.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { type Cassette, CassetteReplay, IGNORED_HEADERS, loadCassette } from "../../src/llm/testing.ts";
import { ApiKeyCredentials, AuthError, OAuthCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { LLMConfig } from "../../src/llm_clients/config.ts";
import {
  type AgentConfig,
  type Connection,
  keyIdentity,
  legacyConnectionConfig,
  OPENAI_API_ENDPOINT,
  resolveModelConfig,
} from "../../src/llm_clients/connection.ts";
import { accountHash, chatModelFromConfig, prefersResponses } from "../../src/llm_clients/factory.ts";
import { virtualScheduler } from "../../src/platform.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

const CASSETTES = fileURLToPath(new URL("cassettes/", import.meta.url));
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

/** The missing-key error for `provider`'s `variable`. */
function missingKey(provider: string, variable: string): Failed {
  return {
    error: RangeError,
    message:
      `API key for provider '${provider}' is not set. Please set the ${variable} environment variable ` +
      `(e.g. add ${variable}=your_key to your .env file).`,
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
        "pass api_key to resume_flow or RunResult.backward",
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
    name: "vllm name-routed model with a block",
    config: { provider: "vllm", model: "my-codex-finetune", vllm: { max_tokens: 64n } },
    env: {},
    golden: "openai_responses/vllm_codex",
    expect: blockOnResponses("vllm", "my-codex-finetune"),
  },
];

let home: string;

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-resolver-"));
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

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
  return cassette;
}

describe("chatModelFromConfig", () => {
  for (const each of CASES) {
    test(each.name, async () => {
      const vars = new Map(Object.entries({ ...each.env, YIAN_AUTH_FILE: join(home, "auth.json") }));
      const env: Env = (name) => vars.get(name);
      const store = new AuthStore(null, { env });
      if (each.saved) {
        const { provider = "openai", credentials, model, endpoint, organization, project } = each.saved;
        await store.save(provider, credentials, model, { endpoint, organization, project });
      }
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
    });
  }
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
