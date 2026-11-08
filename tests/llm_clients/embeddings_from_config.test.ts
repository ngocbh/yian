/**
 * `embedderFromConfig` and `defaultEmbed`: an embedding model's provider resolves to the key,
 * endpoint and headers its chat calls would use, saved logins included. A recording `send`
 * answers every request, so a test checks what was sent.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test } from "bun:test";

import type { Env } from "../../src/defaults.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { ApiKeyCredentials, AuthError, OAuthCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { defaultEmbed, embedderFromConfig } from "../../src/llm_clients/factory.ts";
import type { Send } from "../../src/platform.ts";

/** One request the recording `send` received. */
interface Sent {
  url: string;
  headers: Headers;
  body: { model: string; input: string[] };
}

/** A `send` that records each request and answers one two-number vector per input. */
function recorder(): { requests: Sent[]; send: Send } {
  const requests: Sent[] = [];
  const send: Send = async (url, init) => {
    const body = JSON.parse(String(init.body));
    requests.push({ url: String(url), headers: new Headers(init.headers), body });
    return Response.json({ data: body.input.map((_: string, index: number) => ({ index, embedding: [index, 1] })) });
  };
  return { requests, send };
}

/** Return `[url, authorization]` for each request. */
function routes(requests: Sent[]): [string, string | null][] {
  return requests.map(({ url, headers }) => [url, headers.get("authorization")]);
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

// Each test's environment: a private store in a fresh directory and the chat defaults an
// embedding model must not pick up.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);
const signal = new AbortController().signal;

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-embeddings-"));
  vars = new Map([
    ["YIAN_AUTH_FILE", join(home, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
  ]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Return the test's store. */
function authStore(): AuthStore {
  return new AuthStore(null, { env });
}

test("openai uses the saved API-key login's key, endpoint and settings", async () => {
  vars.set("OPENAI_BASE_URL", "https://gateway.example/v1");
  vars.set("OPENAI_API_KEY", "sk-env-unused");
  await authStore().save("openai", new ApiKeyCredentials("sk-saved-4b1f", { "X-Gateway": "gw-7c2e91" }), "gpt-x", {
    organization: "org-saved",
  });
  const { requests, send } = recorder();
  const embed = await embedderFromConfig({ provider: "openai", model: "text-embedding-3-small" }, { env, send });
  const vectors = await embed(["a", "b"], { signal });
  expect(vectors.map((v) => [...v])).toEqual([[0, 1], [1, 1]]);
  expect(routes(requests)).toEqual([["https://gateway.example/v1/embeddings", "Bearer sk-saved-4b1f"]]);
  expect(requests[0]!.body).toEqual({ model: "text-embedding-3-small", input: ["a", "b"] });
  const { headers } = requests[0]!;
  expect([headers.get("openai-organization"), headers.get("x-gateway")]).toEqual(["org-saved", "gw-7c2e91"]);
});

test("openai with no saved login uses OPENAI_API_KEY at the default endpoint", async () => {
  vars.set("OPENAI_API_KEY", "sk-env-91d0");
  const { requests, send } = recorder();
  await (await embedderFromConfig({ provider: "openai", model: "text-embedding-3-small" }, { env, send }))(["a"], { signal });
  expect(routes(requests)).toEqual([["https://api.openai.com/v1/embeddings", "Bearer sk-env-91d0"]]);
});

test("a vendor uses its environment key and PROVIDER_BASE_URL", async () => {
  vars.set("OPENROUTER_API_KEY", "or-key-2a7f");
  const { requests, send } = recorder();
  const embed = await embedderFromConfig({ provider: "openrouter", model: "openai/text-embedding-3-small" }, { env, send });
  await embed(["a"], { signal });
  expect(routes(requests)).toEqual([["https://openrouter.ai/api/v1/embeddings", "Bearer or-key-2a7f"]]);
});

test("a vendor's saved login wins over its environment key", async () => {
  vars.set("OPENROUTER_API_KEY", "or-env-key");
  await authStore().save("openrouter", new ApiKeyCredentials("or-saved-key"), "m", { makeDefault: false });
  const { requests, send } = recorder();
  await (await embedderFromConfig({ provider: "openrouter", model: "e" }, { env, send }))(["a"], { signal });
  expect(routes(requests)).toEqual([["https://openrouter.ai/api/v1/embeddings", "Bearer or-saved-key"]]);
});

test("an error reply echoing a saved login's secret header value comes back scrubbed", async () => {
  const value = "gw-secret-5e1d90";
  await authStore().save("openrouter", new ApiKeyCredentials("or-saved-key", { "X-Gateway": value }), "m", {
    makeDefault: false,
  });
  const sent: Headers[] = [];
  const send: Send = async (_url, init) => {
    sent.push(new Headers(init.headers));
    return Response.json({ error: { message: `bad gateway token ${value}` } }, { status: 400 });
  };
  const embed = await embedderFromConfig({ provider: "openrouter", model: "e" }, { env, send });
  const error = (await rejected(embed(["a"], { signal }))) as ProviderError;
  expect(sent.map((headers) => headers.get("x-gateway"))).toEqual([value]);
  expect(error).toBeInstanceOf(ProviderError);
  expect(error.message).toBe("openrouter returned HTTP 400 (bad_request): bad gateway token [REDACTED]");
  expect(error.stack).not.toContain(value);
});

test("an OpenAI key changed after the build fails the next request with AuthError, sending nothing", async () => {
  await authStore().save("openai", new ApiKeyCredentials("sk-first-0a1b"), "gpt-x");
  const { requests, send } = recorder();
  const embed = await embedderFromConfig({ provider: "openai", model: "text-embedding-3-small" }, { env, send });
  await embed(["a"], { signal });
  await authStore().save("openai", new ApiKeyCredentials("sk-second-2c3d"), "gpt-x");
  const error = await rejected(embed(["b"], { signal }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("yian auth openai");
  expect(error.message).not.toContain("sk-second-2c3d");
  expect(routes(requests)).toEqual([["https://api.openai.com/v1/embeddings", "Bearer sk-first-0a1b"]]);
});

test("a vendor with no key is refused, naming its key variable", async () => {
  const error = await rejected(embedderFromConfig({ provider: "openrouter", model: "e" }, { env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("OPENROUTER_API_KEY");
});

test("vllm uses VLLM_BASE_URL, with no key", async () => {
  vars.set("VLLM_BASE_URL", "http://gpu-box:8000/v1");
  const { requests, send } = recorder();
  await (await embedderFromConfig({ provider: "vllm", model: "bge-m3" }, { env, send }))(["a"], { signal });
  expect(routes(requests)).toEqual([["http://gpu-box:8000/v1/embeddings", null]]);
});

test("a provider with no embeddings API is refused before anything is read", async () => {
  const others = "xai, deepseek, qwen, qwen-cn, glm, glm-cn, minimax, minimax-cn, openrouter, vllm";
  // A store that can't be read: a refusal after reading it would say so instead.
  writeFileSync(join(home, "auth.json"), "not json");
  const unread = await rejected(embedderFromConfig({ provider: "openai", model: "m" }, { env }));
  expect(unread.message).toContain("auth.json");
  for (const provider of ["anthropic", "google", "azure", "ollama"]) {
    const error = await rejected(embedderFromConfig({ provider, model: "m" }, { env }));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toBe(
      `provider \`${provider}\` has no embeddings API yian supports; use openai or an OpenAI-compatible provider (${others})`,
    );
  }
});

test("an empty model is refused, never filled from the chat defaults", async () => {
  vars.set("OPENAI_API_KEY", "sk-env-91d0");
  const error = await rejected(embedderFromConfig({ provider: "openai", model: "" }, { env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe("an embedding model needs a model name");
});

test("an OpenAI subscription login is refused", async () => {
  await authStore().save(
    "openai",
    new OAuthCredentials({ accessToken: "access", refreshToken: "refresh", expiresAt: 4102444800, accountId: "acct" }),
    "gpt-5.5",
  );
  const error = await rejected(embedderFromConfig({ provider: "openai", model: "text-embedding-3-small" }, { env }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toBe("a ChatGPT subscription can't embed; sign in with an API key");
});

test("defaultEmbed builds a client per call, from that call's provider, model and saved login", async () => {
  vars.set("VLLM_BASE_URL", "http://gpu-box:8000/v1");
  await authStore().save("openrouter", new ApiKeyCredentials("or-first"), "m", { makeDefault: false });
  const { requests, send } = recorder();
  const embed = defaultEmbed(env, { send });
  await embed(["a"], { provider: "openrouter", model: "e1", signal });
  await authStore().save("openrouter", new ApiKeyCredentials("or-second"), "m", { makeDefault: false });
  await embed(["b"], { provider: "openrouter", model: "e1", signal });
  const vectors = await embed(["c", "d"], { provider: "vllm", model: "e2", signal });
  expect(vectors.map((v) => [...v])).toEqual([[0, 1], [1, 1]]);
  expect(routes(requests)).toEqual([
    ["https://openrouter.ai/api/v1/embeddings", "Bearer or-first"],
    ["https://openrouter.ai/api/v1/embeddings", "Bearer or-second"],
    ["http://gpu-box:8000/v1/embeddings", null],
  ]);
  expect(requests.map(({ body }) => body)).toEqual([
    { model: "e1", input: ["a"] },
    { model: "e1", input: ["b"] },
    { model: "e2", input: ["c", "d"] },
  ]);
});

test("defaultEmbed passes its signal to the request", async () => {
  vars.set("VLLM_BASE_URL", "http://gpu-box:8000/v1");
  const stop = new AbortController();
  const send: Send = (_url, init) =>
    new Promise((_resolve, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason)));
  const pending = defaultEmbed(env, { send })(["a"], { provider: "vllm", model: "e", signal: stop.signal });
  // Let the build finish and the request start.
  await new Promise((resolve) => setTimeout(resolve, 10));
  stop.abort(new Error("stopped"));
  expect((await rejected(pending)).message).toBe("stopped");
});
