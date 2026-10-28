/**
 * The credentials a call sends are scrubbed: each scrub test runs on text through `redactKeys`
 * and on errors through `scrubCallCredentials`. The scrub set follows what yian's own clients
 * send: Azure doesn't fall back to `OPENAI_API_KEY`, `ANTHROPIC_AUTH_TOKEN` is never listed, and
 * a subscription lists its account id.
 */

import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import type { Env } from "../src/defaults.ts";
import { ProviderError } from "../src/llm/errors.ts";
import { ApiKeyCredentials, OAuthCredentials } from "../src/llm_clients/auth/common.ts";
import { AuthStore } from "../src/llm_clients/auth/store.ts";
import { LLMConfig } from "../src/llm_clients/config.ts";
import {
  type AgentConfig,
  callCredentials,
  type Connection,
  legacyConnectionConfig,
  resolveModelConfig,
  scrubCallCredentials,
} from "../src/llm_clients/connection.ts";
import { redactKeys } from "../src/llm_clients/redact.ts";

const AZURE_URL = "https://r.openai.azure.com/";

// Each test's environment: a store path in a fresh directory and no key, token or default.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-call-credentials-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth", "auth.json")]]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Return the test's store. */
function authStore(): AuthStore {
  return new AuthStore(null, { env });
}

/** Return `callCredentials` of `config` in the test's environment. */
function listed(config: unknown): readonly string[] {
  return callCredentials(config, { env });
}

/** Return `{provider, model, ...fields}` resolved, with its connection. */
function resolved(provider: string, model: string, fields: AgentConfig = {}): AgentConfig {
  const config = resolveModelConfig({ provider, model, ...fields }, { env });
  expect(Object.hasOwn(config, "_connection")).toBe(true);
  return config;
}

/** Return the connection of a resolved config. */
function connectionOf(config: AgentConfig): Connection {
  return config["_connection"] as Connection;
}

test("host key is the only key read", () => {
  vars.set("VLLM_API_KEY", "sk-env-vllm");
  const host = "sk-host-sentinel-zqxw";
  expect(listed({ provider: "vllm", model: "m", api_key: host })).toEqual([host]);
  expect(listed(new LLMConfig({ provider: "vllm", model: "m", api_key: host }))).toEqual([host]);
});

test("provider key variable", () => {
  vars.set("ANTHROPIC_API_KEY", "sk-env-anthropic");
  const config = resolved("anthropic", "claude-x");
  expect(connectionOf(config).source).toBe("environment");
  expect(listed(config)).toEqual(["sk-env-anthropic"]);
});

test("openai key variable", () => {
  vars.set("OPENAI_API_KEY", "sk-env-openai");
  const config = resolved("openai", "gpt-x");
  expect(connectionOf(config).source).toBe("environment");
  expect(listed(config)).toEqual(["sk-env-openai"]);
  const legacy = legacyConnectionConfig({ provider: "openai", model: "gpt-x" });
  expect(connectionOf(legacy).source).toBe("legacy_environment");
  expect(listed(legacy)).toEqual(["sk-env-openai"]);
});

test("saved api key login wins the variable", async () => {
  await authStore().save("anthropic", new ApiKeyCredentials("sk-saved-anthropic"), "claude-x");
  vars.set("ANTHROPIC_API_KEY", "sk-env-anthropic");
  const config = resolved("anthropic", "claude-x");
  expect(connectionOf(config).source).toBe("store");
  expect(listed(config)).toEqual(["sk-saved-anthropic"]);
});

describe("saved subscription tokens", () => {
  test.each([
    ["fresh", 3600],
    ["expired", -3600],
  ])("%s", async (_id, expiresIn) => {
    await authStore().save(
      "openai",
      new OAuthCredentials({
        accessToken: "access-secret",
        refreshToken: "refresh-secret",
        expiresAt: Date.now() / 1000 + expiresIn,
        accountId: "acct",
      }),
      "gpt-x",
    );
    const config = resolved("openai", "gpt-x");
    expect(connectionOf(config).method).toBe("subscription");
    expect(new Set(listed(config))).toEqual(new Set(["access-secret", "refresh-secret", "acct"]));
  });
});

test("sdk fallback variables", () => {
  vars.set("GEMINI_API_KEY", "sk-env-gemini");
  expect(listed(resolved("google", "gemini-x"))).toEqual(["sk-env-gemini"]);
  // An empty value counts as unset.
  vars.set("GOOGLE_API_KEY", "");
  expect(listed(resolved("google", "gemini-x"))).toEqual(["sk-env-gemini"]);
  vars.set("GOOGLE_API_KEY", "sk-env-google");
  expect(listed(resolved("google", "gemini-x"))).toEqual(["sk-env-google"]);

  // Azure reads only its own key variable.
  vars.set("OPENAI_API_KEY", "sk-env-openai");
  const azure = resolved("azure", "dep", { base_url: AZURE_URL });
  expect(connectionOf(azure).source).toBe("environment");
  expect(listed(azure)).toEqual([]);
});

test("always sent variables", () => {
  vars.set("AZURE_OPENAI_API_KEY", "sk-env-azure");
  vars.set("AZURE_OPENAI_AD_TOKEN", "ad-token-secret");
  const azure = resolved("azure", "dep", { base_url: AZURE_URL });
  expect(new Set(listed(azure))).toEqual(new Set(["sk-env-azure", "ad-token-secret"]));
  const hosted = { provider: "azure", model: "dep", base_url: AZURE_URL, api_key: "sk-host" };
  expect(new Set(listed(hosted))).toEqual(new Set(["sk-host", "ad-token-secret"]));

  // yian's Anthropic client never sends ANTHROPIC_AUTH_TOKEN.
  vars.set("ANTHROPIC_API_KEY", "sk-env-anthropic");
  vars.set("ANTHROPIC_AUTH_TOKEN", "auth-token-secret");
  expect(listed(resolved("anthropic", "claude-x"))).toEqual(["sk-env-anthropic"]);
});

test("environment header values are not listed", () => {
  // The clients drop SDK header variables, so their values are never sent.
  vars.set("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer custom-secret");
  vars.set("ANTHROPIC_CUSTOM_HEADERS", "Authorization: Bearer custom-secret");
  for (const provider of ["openai", "anthropic"]) {
    expect(listed({ provider, model: "m", api_key: "sk-host" })).toEqual(["sk-host"]);
  }
});

test("unresolved config is resolved first", async () => {
  await authStore().save("anthropic", new ApiKeyCredentials("sk-saved-anthropic"), "claude-x");
  expect(listed({ provider: "anthropic", model: "c" })).toEqual(["sk-saved-anthropic"]);
});

test("a malformed store gives no store secrets", () => {
  vars.set("ANTHROPIC_API_KEY", "sk-env-anthropic");
  mkdirSync(join(home, "broken"), { mode: 0o700 });
  const path = join(home, "broken", "auth.json");
  writeFileSync(path, "{not json");
  chmodSync(path, 0o600);
  const config = {
    provider: "anthropic",
    model: "c",
    _connection: { method: "api_key", source: "store", credential_id: "abc", endpoint: null, store_path: path },
  };
  expect(listed(config)).toEqual([]);

  vars.set("YIAN_AUTH_FILE", path);
  expect(listed({ provider: "anthropic", model: "c" })).toEqual(["sk-env-anthropic"]);
});

test("a store connection without a usable store path gives no store secrets", async () => {
  await authStore().save("anthropic", new ApiKeyCredentials("sk-saved-anthropic"), "c");
  const pinned = { method: "api_key", source: "store", credential_id: "abc", endpoint: null };
  for (const connection of [pinned, { ...pinned, store_path: 5 }]) {
    expect(listed({ provider: "anthropic", model: "c", _connection: connection })).toEqual([]);
  }
  expect(listed({ provider: "anthropic", model: "c", _connection: { ...pinned, store_path: null } })).toEqual([
    "sk-saved-anthropic",
  ]);
});

test("an invalid provider still gives the host key", () => {
  expect(listed({ provider: "nope", api_key: "sk-x" })).toEqual(["sk-x"]);
});

describe("an empty config uses the saved default", () => {
  test.each([[null], [{}]])("%#", async (config) => {
    expect(listed(config)).toEqual([]);
    await authStore().save("anthropic", new ApiKeyCredentials("sk-saved-anthropic"), "claude-x");
    expect(listed(config)).toEqual(["sk-saved-anthropic"]);
  });
});

describe("garbage configs never throw", () => {
  test.each([["not a config"], [5n], [{ _connection: "junk" }], [{ provider: 3n }]])("%#", (config) => {
    expect(listed(config)).toEqual([]);
  });
});

test("a masked saved key is removed through redactKeys", async () => {
  const key = "sk-saved-sentinel-zqxw";
  await authStore().save("anthropic", new ApiKeyCredentials(key), "claude-x");
  const config = resolved("anthropic", "claude-x");
  const message = redactKeys(`invalid x-api-key: ${key.slice(0, 8)}****${key.slice(-4)}`, listed(config));
  expect(message).toBe("invalid x-api-key: [REDACTED]");
  expect(message).not.toContain(key.slice(0, 8));
  expect(message).not.toContain(key.slice(-4));
});

test("a masked saved key is removed from a ProviderError cause", async () => {
  const key = "sk-saved-sentinel-zqxw";
  await authStore().save("anthropic", new ApiKeyCredentials(key), "claude-x");
  const config = resolved("anthropic", "claude-x");
  const cause = new ProviderError("auth", `invalid x-api-key: ${key.slice(0, 8)}****${key.slice(-4)}`);
  const error = new Error("model call failed", { cause });
  // Bun builds a stack on its first read; reading it now, as a logger may have, makes the scrub rewrite it.
  expect(cause.stack).toContain(key.slice(0, 8));
  expect(scrubCallCredentials(error, config, { env })).toBe(error);
  expect(cause.message).toBe("invalid x-api-key: [REDACTED]");
  for (const text of [cause.message, cause.stack!]) {
    expect(text).not.toContain(key.slice(0, 8));
    expect(text).not.toContain(key.slice(-4));
  }
});

test("a keyless provider sends nothing", () => {
  vars.set("OLLAMA_API_KEY", "env-ollama-key");
  expect(listed({ provider: "ollama", model: "m" })).toEqual([]);
});

describe("ollama base url credentials", () => {
  test("userinfo lists the password and the basic token", () => {
    vars.set("OLLAMA_BASE_URL", "http://u:test-pass-1@gpu-box:11434");
    expect(listed({ provider: "ollama", model: "m" })).toEqual(["test-pass-1", "dTp0ZXN0LXBhc3MtMQ=="]);
  });

  test("an empty user sends no auth, so nothing is listed", () => {
    vars.set("OLLAMA_BASE_URL", "http://:pw@h");
    expect(listed({ provider: "ollama", model: "m" })).toEqual([]);
  });

  test("a malformed value lists nothing and doesn't throw", () => {
    vars.set("OLLAMA_BASE_URL", "http://u:test-pass-1@h:0");
    expect(listed({ provider: "ollama", model: "m" })).toEqual([]);
  });

  test("unset or empty lists nothing", () => {
    expect(listed({ provider: "ollama", model: "m" })).toEqual([]);
    vars.set("OLLAMA_BASE_URL", "");
    expect(listed({ provider: "ollama", model: "m" })).toEqual([]);
  });
});

test("a missing openai connection sends nothing", () => {
  const config = resolved("openai", "gpt-x");
  expect(connectionOf(config).source).toBe("missing");
  expect(listed(config)).toEqual([]);
});

test("host header values are listed", () => {
  const host = "sk-host-sentinel-zqxw";
  const headers = { "X-Gateway": "gw-token-plain", "X-Env": "prod" };
  const expected = new Set([host, "gw-token-plain", "prod"]);
  expect(new Set(listed({ provider: "deepseek", model: "m", api_key: host, headers }))).toEqual(expected);
  expect(new Set(listed(new LLMConfig({ provider: "deepseek", model: "m", api_key: host, headers })))).toEqual(
    expected,
  );
});

test("saved login header values are listed", async () => {
  await authStore().save(
    "anthropic",
    new ApiKeyCredentials("sk-saved-anthropic", { "X-Gateway": "gw-saved", "X-Env": "prod" }),
    "claude-x",
  );
  vars.set("ANTHROPIC_API_KEY", "sk-env-anthropic");
  const config = resolved("anthropic", "claude-x");
  expect(connectionOf(config).source).toBe("store");
  expect(new Set(listed(config))).toEqual(new Set(["sk-saved-anthropic", "gw-saved", "prod"]));
});

test("a host key never lists the saved header values", async () => {
  await authStore().save(
    "anthropic",
    new ApiKeyCredentials("sk-saved-anthropic", { "X-Gateway": "gw-saved" }),
    "claude-x",
  );
  expect(listed({ provider: "anthropic", model: "c", api_key: "sk-host" })).toEqual(["sk-host"]);
});

test("a short header value is removed through redactKeys", () => {
  const config = {
    provider: "deepseek",
    model: "m",
    api_key: "sk-host-sentinel-zqxw",
    headers: { "X-Env": "prod" },
  };
  expect(redactKeys("gateway rejected X-Env=prod for tenant prod-eu", listed(config))).toBe(
    "gateway rejected X-Env=[REDACTED] for tenant [REDACTED]-eu",
  );
});

test("a short header value is removed from a ProviderError", () => {
  const config = {
    provider: "deepseek",
    model: "m",
    api_key: "sk-host-sentinel-zqxw",
    headers: { "X-Env": "prod" },
  };
  const error = new ProviderError("auth", "gateway rejected X-Env=prod for tenant prod-eu");
  scrubCallCredentials(error, config, { env });
  expect(error.message).toBe("gateway rejected X-Env=[REDACTED] for tenant [REDACTED]-eu");
});

test("a sent credential is removed with the listed ones", () => {
  const config = { provider: "deepseek", model: "m", api_key: "sk-host-sentinel-zqxw" };
  const error = new Error("rejected sk-host-sentinel-zqxw and route-token-sentinel");
  scrubCallCredentials(error, config, { env, sent: ["route-token-sentinel"] });
  expect(error.message).toBe("rejected [REDACTED] and [REDACTED]");
});

test("a saved subscription's account id is removed", async () => {
  await authStore().save(
    "openai",
    new OAuthCredentials({
      accessToken: "access-secret",
      refreshToken: "refresh-secret",
      expiresAt: Date.now() / 1000 + 3600,
      accountId: "acct-sentinel-zqxw",
    }),
    "gpt-x",
  );
  const error = new ProviderError("auth", "account acct-sentinel-zqxw has no access-secret");
  scrubCallCredentials(error, resolved("openai", "gpt-x"), { env });
  expect(error.message).toBe("account [REDACTED] has no [REDACTED]");
});

describe("scrubbing with a garbage config never throws", () => {
  test.each([["not a config"], [5n], [{ _connection: "junk" }], [{ provider: 3n }]])("%#", (config) => {
    const error = new Error("rejected route-token-sentinel");
    expect(scrubCallCredentials(error, config, { env, sent: ["route-token-sentinel"] })).toBe(error);
    expect(error.message).toBe("rejected [REDACTED]");
    const broken: Iterable<string> = {
      [Symbol.iterator]() {
        throw new Error("no credentials");
      },
    };
    expect(scrubCallCredentials(error, config, { env, sent: broken })).toBe(error);
    // A throwing sent still leaves the config's own credentials scrubbed.
    const host = { provider: "vllm", model: "m", api_key: "sk-host-sentinel-zqxw" };
    const fresh = new Error("rejected sk-host-sentinel-zqxw");
    expect(scrubCallCredentials(fresh, host, { env, sent: broken })).toBe(fresh);
    expect(fresh.message).toBe("rejected [REDACTED]");
  });
});
