/**
 * Mirrors `tests/test_openai_connection.py`; model calls go with the factory,
 * `llm_clients/factory.ts`; runs with `runFlow`; the cascade with `compile/llm_cascade.ts`; the key
 * preflight with the CLI. `test_missing_identity_does_not_adopt_later_environment_key` is here as
 * its `connectionCredentials` twin.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";

import type { Env } from "../src/defaults.ts";
import { ApiKeyCredentials, AuthError, OAuthCredentials, type ProviderSelection } from "../src/llm_clients/auth/common.ts";
import { OpenAIAuth } from "../src/llm_clients/auth/openai.ts";
import { AuthStore } from "../src/llm_clients/auth/store.ts";
import { LLMConfig } from "../src/llm_clients/config.ts";
import {
  checkModelConfig,
  type Connection,
  connectionCredentials,
  connectionSnapshot,
  legacyConnectionConfig,
  resolveModelConfig,
  splitModelConfig,
} from "../src/llm_clients/connection.ts";

// Each test's environment: a private store in a fresh directory and no default provider, model,
// OpenAI key or endpoint, as Python's `isolated_auth` leaves it.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-openai-connection-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Return the test's store. */
function authStore(): AuthStore {
  return new AuthStore(null, { env });
}

/** Save an OpenAI subscription for `gpt-selected` and make it the default. */
function subscription(): Promise<ProviderSelection> {
  return authStore().save(
    "openai",
    new OAuthCredentials({
      accessToken: "access-secret",
      refreshToken: "refresh-secret",
      expiresAt: Date.now() / 1000 + 3600,
      accountId: "account",
    }),
    "gpt-selected",
  );
}

/** Return the error `run` throws, failing the test if it throws none. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

/** Return the error `promise` rejects with, failing the test if it resolves. */
async function rejected(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

test("no builtin provider or model", () => {
  let error = thrown(() => resolveModelConfig({}, { env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/^No model is set; .*'yian auth <provider>'/);
  error = thrown(() => resolveModelConfig({ provider: "anthropic" }, { env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/^No model is set for provider 'anthropic'; .*'yian auth anthropic'/);
});

test("azure deployment environment is not a model default", () => {
  // The Azure deployment is always llm_config.model; no environment default fills it.
  vars.set("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deployment");
  expect(Object.hasOwn(connectionSnapshot({ env }).providerModels, "azure")).toBe(false);
  expect(() => resolveModelConfig({ provider: "azure" }, { env })).toThrow("No model is set for provider 'azure'");
});

test("saved selection wins incidental key and is nonsecret", async () => {
  const selected = await subscription();
  vars.set("OPENAI_API_KEY", "incidental-secret");
  const config = resolveModelConfig({}, { env });
  expect(config["model"]).toBe("gpt-selected");
  const connection = config["_connection"] as Connection;
  expect(connection.method).toBe("subscription");
  expect(connection.credential_id).toBe(selected.credentialId);
  expect(JSON.stringify(config)).not.toContain("secret");
});

test("authored model wins saved default", async () => {
  await subscription();
  expect(resolveModelConfig({ model: "explicit" }, { env })["model"]).toBe("explicit");
});

test("changed environment key says start a new run", async () => {
  // A changed key is reported after the failure is recorded, so restoring it can't help.
  vars.set("OPENAI_API_KEY", "first-secret");
  const connection = connectionSnapshot({ env }).connections["openai"];
  vars.set("OPENAI_API_KEY", "second-secret");
  const error = await rejected(connectionCredentials(connection, "openai", { env }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("changed since this run started; start a new run");
  expect(error.message).not.toContain("restore");
});

test("reconnected store credential names one fix", async () => {
  // A pinned saved credential that was replaced names one fix, without 'reconnect'.
  await authStore().save("openai", new ApiKeyCredentials("old-secret"), "gpt-5.5");
  const connection = connectionSnapshot({ env }).connections["openai"];
  await authStore().save("openai", new ApiKeyCredentials("new-secret"), "gpt-5.5");
  const error = await rejected(connectionCredentials(connection, "openai", { env }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("changed");
  expect(error.message.split("start a new run").length - 1).toBe(1);
  expect(error.message).not.toContain("reconnect");
});

test("missing identity does not adopt later environment key", async () => {
  const config = resolveModelConfig({ provider: "openai", model: "gpt-5.5" }, { env });
  vars.set("OPENAI_API_KEY", "new-secret");
  const error = await rejected(connectionCredentials(config["_connection"], "openai", { env }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("yian auth openai");
});

test("legacy openai cannot select subscription", async () => {
  await subscription();
  vars.set("OPENAI_API_KEY", "legacy-secret");
  const old = { provider: "openai", model: "old" };
  const config = legacyConnectionConfig(old);
  expect(old).toEqual({ provider: "openai", model: "old" });
  expect((config["_connection"] as Connection).method).toBe("api_key");
  expect((config["_connection"] as Connection).source).toBe("legacy_environment");
  expect(legacyConnectionConfig(config)).toEqual(config);
});

test("internal metadata is not YAML configuration", () => {
  const config = resolveModelConfig({ provider: "openai", model: "gpt-5.5" }, { env });
  expect(() => new LLMConfig(config)).toThrow(/Extra inputs/);
  (config["_connection"] as Record<string, unknown>)["token"] = "bad";
  expect(() => splitModelConfig(config)).toThrow(/connection/);
});

/** Save an OpenAI subscription for `gpt-5.5` that expires in a second and make it the default. */
function expiringSubscription(): Promise<ProviderSelection> {
  return authStore().save(
    "openai",
    new OAuthCredentials({
      accessToken: "expired-token",
      refreshToken: "refresh-token",
      expiresAt: Date.now() / 1000 + 1,
      accountId: "account",
    }),
    "gpt-5.5",
  );
}

/** Return fresh subscription tokens for the same account. */
function freshTokens(): OAuthCredentials {
  return new OAuthCredentials({
    accessToken: "fresh-token",
    refreshToken: "new-refresh",
    expiresAt: Date.now() / 1000 + 3600,
    accountId: "account",
  });
}

test("refresh retains connection identity on model requests", async () => {
  // The connection half of Python's test: model requests resolve credentials this way.
  const selected = await expiringSubscription();
  const refreshed: OAuthCredentials[] = [];
  const spy = spyOn(OpenAIAuth.prototype, "refresh").mockImplementation(async (credentials) => {
    refreshed.push(credentials);
    return freshTokens();
  });
  try {
    const connection = resolveModelConfig({}, { env })["_connection"] as Connection;
    const tokens: string[] = [];
    for (let call = 0; call < 2; call++) {
      const credentials = await connectionCredentials(connection, "openai", { env });
      tokens.push((credentials as OAuthCredentials).accessToken);
    }
    expect(refreshed.length).toBe(1);
    expect(refreshed[0]!.accessToken).toBe("expired-token");
    expect(tokens).toEqual(["fresh-token", "fresh-token"]);
    expect(connection.credential_id).toBe(selected.credentialId);
    expect(authStore().selection("openai")?.credentialId).toBe(selected.credentialId);
  } finally {
    spy.mockRestore();
  }
});

test("an explicit refresh wins over OpenAI's", async () => {
  await expiringSubscription();
  const spy = spyOn(OpenAIAuth.prototype, "refresh").mockImplementation(async () => {
    throw new Error("OpenAI's refresh ran");
  });
  try {
    const connection = resolveModelConfig({}, { env })["_connection"];
    const credentials = await connectionCredentials(connection, "openai", { env, refresh: () => freshTokens() });
    expect((credentials as OAuthCredentials).accessToken).toBe("fresh-token");
    expect(spy).not.toHaveBeenCalled();
  } finally {
    spy.mockRestore();
  }
});

test("endpoint metadata cannot record embedded secrets", () => {
  vars.set("OPENAI_BASE_URL", "https://api.example/v1?key=hidden-secret");
  const error = thrown(() => connectionSnapshot({ env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("without credentials");
  expect(error.message).not.toContain("hidden-secret");
});

test("unsupported default provider environment is named", () => {
  vars.set("YIAN_DEFAULT_PROVIDER", "claude");
  const error = thrown(() => connectionSnapshot({ env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toMatch(/YIAN_DEFAULT_PROVIDER='claude'.*anthropic/);
});

describe("check model config checks the openai block against the subscription endpoint", () => {
  test.each([[{ temperature: 0.2 }], [{ max_output_tokens: 100n }]])("%#", async (block) => {
    await subscription();
    const config = resolveModelConfig({ openai: block }, { env });
    const error = thrown(() => checkModelConfig(config));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain(`\`openai.${Object.keys(block)[0]}\``);
    expect(error.message).toContain("ChatGPT subscription endpoint");
    // A host key sends the same agent to the Responses API, which accepts the block.
    checkModelConfig({ ...config, api_key: "host-key" });
  });
});

describe("check model config accepts the openai block on an api key connection", () => {
  test.each([[{ temperature: 0.2 }], [{ max_output_tokens: 100n }]])("%#", async (block) => {
    await authStore().save("openai", new ApiKeyCredentials("saved-key"), "gpt-5.5");
    const config = resolveModelConfig({ openai: block }, { env });
    expect((config["_connection"] as Connection).method).toBe("api_key");
    checkModelConfig(config);
  });
});
