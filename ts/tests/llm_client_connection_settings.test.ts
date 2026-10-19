/**
 * Mirrors `tests/test_llm_client_connection_settings.py`: connection settings (organization,
 * project, headers) reach the request and nothing else. The models are built with
 * `chatModelFromConfig` from a config's `organization`, `project` and `headers` (Python's
 * `connection_headers`) or a saved login, in an environment holding every SDK header and
 * credential variable as an `env-` value, and call a recording `send` (it answers 400, so a test
 * checks the request). Python's sync and async calls are one call here.
 *
 * Ported: the openai, compatible-provider, Azure, Anthropic and Google sends tests (without the
 * env-header warning halves: TS reads no header variables, so it has nothing to warn about),
 * `model_from_config` with and without settings and with Anthropic headers, the saved-login tests,
 * and the Ollama and subscription refusals (`LLMConfig` refuses headers for Ollama, and headers
 * without a config key, so a subscription connection never gets them; 0 requests). Adapted: the
 * stay-out-of-the-model-fields tests check the header values are not in `route.headers` or
 * `inspect(model)` and are in `route.auth.credentials()`, so the scrub knows them; an OpenAI
 * organization and project are not secret in TS, so they are route headers and that half is
 * dropped.
 *
 * Elsewhere: the `get_llm_checks_connection_headers` and
 * `anthropic_and_google_get_llm_check_connection_headers` texts are the `validateHeaders` and
 * `LLMConfig` tests' in `llm_config_connection_settings.test.ts`; the three no-redirect tests are
 * the transport tests' (TS refuses every redirect). Dropped: the env-warning tests, the Vertex
 * switch test (TS reads neither), and the shared-HTTP-client, SDK-client, proxy, aiohttp and host
 * HTTP client tests (TS has no SDK or host HTTP clients).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inspect } from "node:util";

import { afterEach, beforeEach, expect, test } from "bun:test";

import type { Env } from "../src/defaults.ts";
import { ExtraHeadersAuth } from "../src/llm/auth.ts";
import { collect } from "../src/llm/events.ts";
import { Message, Request, TextPart } from "../src/llm/messages.ts";
import { ApiKeyCredentials, OAuthCredentials } from "../src/llm_clients/auth/common.ts";
import { AuthStore } from "../src/llm_clients/auth/store.ts";
import { LLMConfig } from "../src/llm_clients/config.ts";
import { chatModelFromConfig } from "../src/llm_clients/factory.ts";
import { ValidationError } from "../src/llm_clients/provider_blocks.ts";
import type { Send } from "../src/platform.ts";

const GATEWAY = "gw-secret-value";
const AZURE_URL = "https://res.openai.azure.com";

// Every SDK header and credential variable, as Python's `poisoned_env` sets them.
const POISONED: Record<string, string> = {
  OPENAI_CUSTOM_HEADERS:
    "X-Env-Injected: from-env\nAuthorization: Bearer env-custom-auth\napi-key: env-custom-api-key\n" +
    "ChatGPT-Account-Id: env-account",
  OPENAI_ORG_ID: "env-org",
  OPENAI_ORGANIZATION: "env-org-2",
  OPENAI_PROJECT_ID: "env-project",
  AZURE_OPENAI_AD_TOKEN: "env-ad-token",
  OPENAI_API_VERSION: "2025-04-01-preview",
  ANTHROPIC_CUSTOM_HEADERS:
    "X-Env-Injected: from-env\nX-Api-Key: env-custom-x-api-key\nAuthorization: Bearer env-custom-anthropic-auth",
  ANTHROPIC_AUTH_TOKEN: "env-anthropic-token",
  OPENAI_ADMIN_KEY: "env-admin",
};

let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-connection-settings-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")], ...Object.entries(POISONED)]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** A `send` that records each request's headers and answers it 400. */
function recorder(): { requests: Headers[]; send: Send } {
  const requests: Headers[] = [];
  const send: Send = async (_input, init) => {
    requests.push(new Headers(init.headers));
    return new Response(JSON.stringify({ error: { message: "refused" } }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  };
  return { requests, send };
}

const HI = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("hi")] })] });

/** Build `config` without a transport. */
function build(config: Record<string, unknown> | LLMConfig) {
  return chatModelFromConfig(config, { env, onWarning: () => {} });
}

/** Build `config`, ask it "hi" and return the one request's headers; the 400 is swallowed. */
async function ask(config: Record<string, unknown> | LLMConfig): Promise<Headers> {
  const { requests, send } = recorder();
  const model = await chatModelFromConfig(config, { env, send, onWarning: () => {} });
  await collect(model.stream(HI)).catch(() => null);
  expect(requests).toHaveLength(1);
  return requests[0]!;
}

/** Assert `headers` carry nothing taken from the environment. */
function assertNoEnvValues(headers: Headers): void {
  expect([...headers].filter(([, value]) => value.toLowerCase().includes("env-"))).toEqual([]);
  expect(headers.has("x-env-injected")).toBe(false);
}

/** The value of each of `names` in `headers`; a missing header is `null`. */
function picked(headers: Headers, names: string[]): Record<string, string | null> {
  return Object.fromEntries(names.map((name) => [name, headers.get(name)]));
}

const OPENAI_SETTINGS = ["authorization", "openai-organization", "openai-project", "x-gateway"];

test("openai sends organization project and headers", async () => {
  const config = {
    provider: "openai",
    model: "gpt-5.5",
    api_key: "sk-explicit",
    organization: "org-config",
    project: "proj-config",
    headers: { "X-Gateway": GATEWAY },
  };
  const headers = await ask(config);
  expect(picked(headers, OPENAI_SETTINGS)).toEqual({
    authorization: "Bearer sk-explicit",
    "openai-organization": "org-config",
    "openai-project": "proj-config",
    "x-gateway": GATEWAY,
  });
  assertNoEnvValues(headers);
  // The ids are route headers; the header value rides on the auth.
  expect((await build(config)).route.headers).toEqual({
    "OpenAI-Organization": "org-config",
    "OpenAI-Project": "proj-config",
  });
});

test("openai compatible provider sends openai project as a header", async () => {
  const config = {
    provider: "deepseek",
    model: "deepseek-chat",
    api_key: "ds-explicit",
    headers: { "OpenAI-Project": "proj-gateway", "X-Gateway": GATEWAY },
  };
  const headers = await ask(config);
  expect(picked(headers, ["authorization", "openai-project", "x-gateway"])).toEqual({
    authorization: "Bearer ds-explicit",
    "openai-project": "proj-gateway",
    "x-gateway": GATEWAY,
  });
  assertNoEnvValues(headers);
  // Outside openai it is one more secret header, not a route header.
  const model = await build(config);
  expect(model.route.headers).toEqual({});
  expect(model.route.auth.credentials()).toContain("proj-gateway");
});

test("azure sends headers", async () => {
  const headers = await ask({
    provider: "azure",
    model: "dep",
    base_url: AZURE_URL,
    api_key: "az-explicit",
    headers: { "X-Gateway": GATEWAY },
  });
  expect(picked(headers, ["api-key", "authorization", "x-gateway"])).toEqual({
    "api-key": "az-explicit",
    authorization: null,
    "x-gateway": GATEWAY,
  });
  assertNoEnvValues(headers);
});

const HEADER_CONFIGS: Record<string, Record<string, unknown>> = {
  openai: { provider: "openai", model: "gpt-5.5", api_key: "sk-explicit", organization: "org-config", project: "proj-config" },
  deepseek: { provider: "deepseek", model: "deepseek-chat", api_key: "ds-explicit" },
  azure: { provider: "azure", model: "dep", base_url: AZURE_URL, api_key: "az-explicit" },
  anthropic: { provider: "anthropic", model: "claude-sonnet-4-5", api_key: "sk-ant-explicit" },
  google: { provider: "google", model: "gemini-2.5-flash", api_key: "g-explicit" },
};

/** Assert the header value is not on the route's plain fields and is known to the scrub. */
async function assertHeaderValueStaysOut(provider: string): Promise<void> {
  const model = await build({ ...HEADER_CONFIGS[provider], headers: { "X-Gateway": GATEWAY } });
  expect(JSON.stringify(model.route.headers)).not.toContain(GATEWAY);
  expect(inspect(model)).not.toContain(GATEWAY);
  expect(model.route.auth.credentials()).toContain(GATEWAY);
}

test.each(["openai", "deepseek", "azure"])("header values stay out of the model fields (%s)", assertHeaderValueStaysOut);

test.each(["anthropic", "google"])(
  "anthropic and google header values stay out of the model fields (%s)",
  assertHeaderValueStaysOut,
);

// Known gap: the auth objects keep the header values (and keys) in TS `private` fields, which
// `Bun.inspect`, `console.log` and a deep `util.inspect` print; `#private` fields would hide
// them, as `ApiKeyCredentials` does. Remove `.failing` once they do.
test.failing("header values stay out of a full inspect of the model", async () => {
  const model = await build({ ...HEADER_CONFIGS["deepseek"], headers: { "X-Gateway": GATEWAY } });
  expect(Bun.inspect(model)).not.toContain(GATEWAY);
});

test.each(["LLMConfig", "dict"])("model from config sends the settings (%s)", async (carrier) => {
  const fields = {
    provider: "openai",
    model: "gpt-5.5",
    api_key: "sk-host",
    organization: "org-config",
    project: "proj-config",
    headers: { "X-Gateway": GATEWAY },
  };
  const headers = await ask(carrier === "LLMConfig" ? new LLMConfig(fields) : fields);
  expect(picked(headers, OPENAI_SETTINGS)).toEqual({
    authorization: "Bearer sk-host",
    "openai-organization": "org-config",
    "openai-project": "proj-config",
    "x-gateway": GATEWAY,
  });
  assertNoEnvValues(headers);
});

test("model from config without settings passes no connection headers", async () => {
  const model = await build({ provider: "openai", model: "gpt-5.5", api_key: "sk-host" });
  expect(model.route.headers).toEqual({});
  expect(model.route.auth).not.toBeInstanceOf(ExtraHeadersAuth);
  expect(await model.route.auth.headers()).toEqual({ Authorization: "Bearer sk-host" });
});

test("model from config sends anthropic headers", async () => {
  const headers = await ask(
    new LLMConfig({
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      api_key: "sk-ant-host",
      headers: { "X-Gateway": GATEWAY },
    }),
  );
  expect(picked(headers, ["x-api-key", "authorization", "x-gateway"])).toEqual({
    "x-api-key": "sk-ant-host",
    authorization: null,
    "x-gateway": GATEWAY,
  });
  assertNoEnvValues(headers);
});

test("anthropic sends headers", async () => {
  const headers = await ask({ ...HEADER_CONFIGS["anthropic"], headers: { "X-Gateway": GATEWAY } });
  expect(picked(headers, ["x-api-key", "authorization", "x-gateway"])).toEqual({
    "x-api-key": "sk-ant-explicit",
    authorization: null,
    "x-gateway": GATEWAY,
  });
  expect(headers.has("anthropic-version")).toBe(true);
  assertNoEnvValues(headers);
});

test("google sends headers", async () => {
  const headers = await ask({ ...HEADER_CONFIGS["google"], headers: { "X-Gateway": GATEWAY } });
  expect(picked(headers, ["x-goog-api-key", "content-type", "x-gateway"])).toEqual({
    "x-goog-api-key": "g-explicit",
    "content-type": "application/json",
    "x-gateway": GATEWAY,
  });
  assertNoEnvValues(headers);
});

/** Build `config` with a recording `send` and return the build error and the requests sent. */
async function refused(config: Record<string, unknown>): Promise<[Error, Headers[]]> {
  const { requests, send } = recorder();
  const error = await chatModelFromConfig(config, { env, send, onWarning: () => {} }).catch((thrown: Error) => thrown);
  expect(error).toBeInstanceOf(ValidationError);
  expect((error as Error).message).not.toContain(GATEWAY);
  return [error as Error, requests];
}

test("ollama refuses connection headers", async () => {
  const [error, requests] = await refused({ provider: "ollama", model: "qwen3", headers: { "X-Gateway": GATEWAY } });
  expect(error.message).toContain("provider ollama does not take llm_config.headers");
  expect(requests).toEqual([]);
});

test("subscription model refuses connection headers", async () => {
  await new AuthStore(null, { env }).save(
    "openai",
    new OAuthCredentials({
      accessToken: "synthetic-access",
      refreshToken: "synthetic-refresh",
      expiresAt: Date.now() / 1000 + 3600,
      accountId: "synthetic-account",
    }),
    "gpt-5.5",
  );
  const [error, requests] = await refused({ provider: "openai", model: "gpt-5.5", headers: { "X-Gateway": GATEWAY } });
  expect(error.message).toContain("llm_config.headers needs llm_config.api_key");
  expect(requests).toEqual([]);
});

test("a saved openai login sends its settings", async () => {
  await new AuthStore(null, { env }).save("openai", new ApiKeyCredentials("sk-saved", { "X-Gateway": GATEWAY }), "gpt-5.5", {
    organization: "org-saved",
    project: "proj-saved",
  });
  const headers = await ask({ provider: "openai", model: "gpt-5.5" });
  expect(picked(headers, OPENAI_SETTINGS)).toEqual({
    authorization: "Bearer sk-saved",
    "openai-organization": "org-saved",
    "openai-project": "proj-saved",
    "x-gateway": GATEWAY,
  });
  assertNoEnvValues(headers);
});

test("a saved anthropic login sends its headers", async () => {
  await new AuthStore(null, { env }).save(
    "anthropic",
    new ApiKeyCredentials("sk-ant-saved", { "X-Gateway": GATEWAY }),
    "claude-sonnet-4-5",
  );
  const headers = await ask({ provider: "anthropic", model: "claude-sonnet-4-5" });
  expect(picked(headers, ["x-api-key", "x-gateway"])).toEqual({ "x-api-key": "sk-ant-saved", "x-gateway": GATEWAY });
  assertNoEnvValues(headers);
});
