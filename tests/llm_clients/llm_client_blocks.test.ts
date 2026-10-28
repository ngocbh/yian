/**
 * Provider blocks reach the request body of the API each model calls. The models are built with
 * `chatModelFromConfig` and call a recording `send` (it answers 400, so a test checks the
 * request, not a reply).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test } from "bun:test";

import type { Env } from "../src/defaults.ts";
import { collect } from "../src/llm/events.ts";
import { Message, Request, TextPart } from "../src/llm/messages.ts";
import { SUPPORTED_LLM_PROVIDERS } from "../src/llm_clients/config.ts";
import { chatModelFromConfig } from "../src/llm_clients/factory.ts";
import type { Send } from "../src/platform.ts";

/** A model id and a block holding one vendor-specific and one standard field. */
interface Case {
  model: string;
  block: Record<string, unknown>;
}

const BLOCK_CASES: Record<string, Case> = {
  xai: { model: "grok-4", block: { reasoning_effort: "high", temperature: 0.3 } },
  deepseek: { model: "deepseek-chat", block: { thinking: { type: "disabled" }, max_tokens: 50 } },
  qwen: { model: "qwen-plus", block: { enable_search: true, top_p: 0.5 } },
  "qwen-cn": { model: "qwen-plus", block: { enable_thinking: false, max_tokens: 64 } },
  glm: { model: "glm-4.6", block: { do_sample: false, temperature: 0.5 } },
  "glm-cn": { model: "glm-4.6", block: { thinking: { type: "disabled" }, max_tokens: 64 } },
  minimax: { model: "MiniMax-M2.7", block: { thinking: { type: "adaptive" }, top_p: 0.9 } },
  "minimax-cn": { model: "MiniMax-M2.7", block: { reasoning_effort: "low", temperature: 0.7 } },
  openrouter: { model: "openai/gpt-5.5", block: { reasoning: { effort: "high" }, top_p: 0.9 } },
  vllm: { model: "served-model", block: { top_k: 20, max_tokens: 32 } },
};

const AZURE_ENDPOINT = "https://res.openai.azure.com/";

/** A client with its own API: the endpoint path it calls. */
const DIRECT_CASES: Record<string, Case & { baseUrl?: string; path: string }> = {
  openai: {
    model: "gpt-5.5",
    block: { reasoning: { effort: "high" }, text: { verbosity: "low" } },
    path: "/v1/responses",
  },
  azure: {
    model: "my-deployment",
    block: { max_tokens: 100, reasoning_effort: "low" },
    baseUrl: AZURE_ENDPOINT,
    path: "/openai/deployments/my-deployment/chat/completions",
  },
};

// The providers whose block tests are the protocol body tests'.
const OTHER_CASES: Record<string, Case> = {
  anthropic: { model: "claude-sonnet-4-5", block: { max_tokens: 64, top_k: 5 } },
  google: { model: "gemini-2.5-flash", block: { generationConfig: { topK: 5, maxOutputTokens: 64 } } },
  ollama: { model: "qwen3", block: { options: { num_ctx: 8192, temperature: 0.2 }, keep_alive: "5m" } },
};

// Each test's environment: a private, empty store and the Azure API version.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-client-blocks-"));
  vars = new Map([
    ["YIAN_AUTH_FILE", join(home, "auth.json")],
    ["OPENAI_API_VERSION", "2025-03-01-preview"],
  ]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** One request a `recorder` received. */
interface Sent {
  url: URL;
  body: Record<string, unknown>;
}

/** A `send` that records each request and answers it 400. */
function recorder(): { requests: Sent[]; send: Send } {
  const requests: Sent[] = [];
  const send: Send = async (input, init) => {
    requests.push({ url: new URL(String(input)), body: JSON.parse(String(init.body)) });
    return new Response(JSON.stringify({ error: { message: "refused" } }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  };
  return { requests, send };
}

const HI = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("hi")] })] });

/** Return the config for `provider` with a test key and, when given, its block. */
function config(provider: string, model: string, block?: Record<string, unknown>, fields: Record<string, unknown> = {}) {
  const built: Record<string, unknown> = { provider, model, api_key: "sk-test", ...fields };
  if (block !== undefined) built[provider] = block;
  return built;
}

/** Build `llmConfig`, ask it "hi" and return what `send` received; the 400 is swallowed. */
async function ask(llmConfig: Record<string, unknown>): Promise<Sent[]> {
  const { requests, send } = recorder();
  const model = await chatModelFromConfig(llmConfig, { env, send, onWarning: () => {} });
  await collect(model.stream(HI)).catch(() => null);
  return requests;
}

test.each(Object.keys(DIRECT_CASES).sort())("direct block reaches the body unchanged (%s)", async (provider) => {
  const { model, block, baseUrl, path } = DIRECT_CASES[provider]!;
  const [request] = await ask(config(provider, model, block, baseUrl ? { base_url: baseUrl } : {}));
  expect(request!.url.pathname).toBe(path);
  for (const [key, value] of Object.entries(block)) expect(request!.body[key]).toEqual(value);
});

test.each(Object.keys(BLOCK_CASES).sort())("openai compatible block reaches the body unchanged (%s)", async (provider) => {
  const { model, block } = BLOCK_CASES[provider]!;
  const [request] = await ask(config(provider, model, block));
  expect(request!.url.pathname).toEndWith("/chat/completions");
  for (const [key, value] of Object.entries(block)) expect(request!.body[key]).toEqual(value);
});

test("azure responses deployment rejects a chat block", async () => {
  const { requests, send } = recorder();
  const error = await chatModelFromConfig(
    config("azure", "gpt-5-codex", { max_tokens: 100 }, { base_url: AZURE_ENDPOINT }),
    { env, send, onWarning: () => {} },
  ).catch((thrown: Error) => thrown);
  expect(error).toBeInstanceOf(RangeError);
  for (const part of ["azure", "gpt-5-codex", "Chat Completions", "Responses API"]) {
    expect((error as Error).message).toContain(part);
  }
  expect(requests).toEqual([]);
});

test("azure responses deployment without a block still runs", async () => {
  const [request] = await ask(config("azure", "gpt-5-codex", undefined, { base_url: AZURE_ENDPOINT }));
  expect(request!.url.pathname).toBe("/openai/responses");
});

test("every supported provider has a request body case", async () => {
  const cases: Record<string, Case & { baseUrl?: string }> = { ...BLOCK_CASES, ...DIRECT_CASES, ...OTHER_CASES };
  expect(new Set(Object.keys(cases))).toEqual(new Set<string>(SUPPORTED_LLM_PROVIDERS));
  for (const [provider, { model, block, baseUrl }] of Object.entries(cases)) {
    const fields = baseUrl ? { base_url: baseUrl } : {};
    // Ollama sends no key, so its config takes none.
    const llmConfig = provider === "ollama" ? { provider, model, ollama: block } : config(provider, model, block, fields);
    expect([provider, (await ask(llmConfig)).length]).toEqual([provider, 1]);
  }
});
