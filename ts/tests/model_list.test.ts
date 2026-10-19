/**
 * Mirrors `tests/test_model_list.py`: model listing per provider host, through an injected
 * `request` (no network), plus two loopback tests through the default `request`.
 */

import http from "node:http";
import type { AddressInfo } from "node:net";
import { inspect } from "node:util";

import { describe, expect, test } from "bun:test";

import { current } from "../src/defaults.ts";
import { ProviderError } from "../src/llm/errors.ts";
import { HttpResponse } from "../src/llm_clients/auth/common.ts";
import type { request } from "../src/llm_clients/auth/http.ts";
import { OPENAI_COMPATIBLE, PROVIDER_BASE_URL } from "../src/llm_clients/factory.ts";
import { listModels, ModelListError } from "../src/llm_clients/model_list.ts";
import { virtualScheduler } from "../src/platform.ts";

const KEY = "sk-test-SENTINEL-0123456789";
const BODY_MARKER = "Incorrect API key provided";
const AUTH_HEADERS = ["authorization", "x-api-key", "x-goog-api-key"];
const GATEWAY = "gw-SENTINEL-token-9876";
const ORG = "org-SENTINEL-1234";
const PROJECT = "proj_SENTINEL-5678";

/** One call the fake `request` saw. */
interface Call {
  url: string;
  base: string;
  params: Record<string, string>;
  /** Header names lowercased to their values. */
  headers: Record<string, string>;
  /** Header names as sent. */
  names: string[];
  timeoutMs: number;
  bodyLimit: number | undefined;
}

type Fake = typeof request & { calls: Call[] };

/** Return a fake `request` that records each call and answers it with `handler`. */
function server(handler: (call: Call) => HttpResponse | Promise<HttpResponse>): Fake {
  const calls: Call[] = [];
  const fake = async (method: string, url: string, options: Parameters<typeof request>[2]) => {
    expect(method).toBe("GET");
    const call: Call = {
      url,
      base: url.split("?")[0]!,
      params: Object.fromEntries(new URL(url).searchParams),
      headers: Object.fromEntries(Object.entries(options.headers).map(([name, value]) => [name.toLowerCase(), value])),
      names: Object.keys(options.headers),
      timeoutMs: options.timeoutMs,
      bodyLimit: options.bodyLimit,
    };
    calls.push(call);
    return handler(call);
  };
  return Object.assign(fake, { calls });
}

function ok(data: unknown): Fake {
  return server(() => new HttpResponse(200, data));
}

/** A `request` that fails the test if called (its throw would surface as the wrong reason). */
async function neverCalled(): Promise<HttpResponse> {
  throw new Error("no request expected");
}

/** Return a `request` that throws `raised`. */
function raising(raised: unknown): typeof request {
  return async () => {
    throw raised;
  };
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

/** Every text an error shows. */
function texts(error: Error): string[] {
  return [String(error), inspect(error), error.stack ?? ""];
}

function assertClean(error: Error, message: string): void {
  expect(error).toBeInstanceOf(ModelListError);
  expect(error.name).toBe("ModelListError");
  expect(error.message).toBe(message);
  expect("cause" in error).toBe(false);
  for (const text of texts(error)) {
    expect(text).not.toContain(KEY);
    expect(text).not.toContain(BODY_MARKER);
  }
}

function assertNoSettings(error: Error): void {
  for (const text of texts(error)) {
    for (const value of [GATEWAY, ORG, PROJECT]) expect(text).not.toContain(value);
  }
}

test("anthropic pages with after_id and key headers", async () => {
  const fake = server((call) =>
    "after_id" in call.params
      ? new HttpResponse(200, {
          data: [{ id: "claude-c" }, { id: "claude-a" }],
          has_more: false,
          last_id: "claude-c",
        })
      : new HttpResponse(200, {
          data: [{ id: "claude-b" }, { id: "claude-a" }],
          has_more: true,
          last_id: "claude-a",
        }),
  );
  expect(await listModels("anthropic", { apiKey: KEY, request: fake })).toEqual(["claude-a", "claude-b", "claude-c"]);
  expect(fake.calls.map((call) => call.base)).toEqual(Array(2).fill("https://api.anthropic.com/v1/models"));
  expect(fake.calls[0]!.params).toEqual({ limit: "1000" });
  expect(fake.calls[1]!.params).toEqual({ limit: "1000", after_id: "claude-a" });
  for (const call of fake.calls) {
    expect(call.headers["x-api-key"]).toBe(KEY);
    expect(call.headers["anthropic-version"]).toBe("2023-06-01");
    expect(call.headers).not.toHaveProperty("authorization");
    expect(call.timeoutMs).toBeGreaterThan(0);
    expect(call.timeoutMs).toBeLessThanOrEqual(8000);
    expect(call.bodyLimit).toBe(current().maxModelListBytes);
  }
});

test("anthropic custom base url", async () => {
  const fake = ok({ data: [{ id: "m" }], has_more: false });
  expect(await listModels("anthropic", { apiKey: KEY, baseUrl: "https://gateway.example.com/", request: fake })).toEqual(
    ["m"],
  );
  expect(fake.calls[0]!.base).toBe("https://gateway.example.com/v1/models");
});

test("google uses key header, filters generateContent and strips the prefix", async () => {
  const fake = server((call) =>
    "pageToken" in call.params
      ? new HttpResponse(200, {
          models: [
            { name: "models/gemini-2.5-flash", supportedGenerationMethods: ["generateContent"] },
            { name: "models/aqa" },
          ],
        })
      : new HttpResponse(200, {
          models: [
            { name: "models/gemini-2.5-pro", supportedGenerationMethods: ["generateContent", "countTokens"] },
            { name: "models/text-embedding-004", supportedGenerationMethods: ["embedContent"] },
          ],
          nextPageToken: "tok2",
        }),
  );
  expect(await listModels("google", { apiKey: KEY, request: fake })).toEqual(["gemini-2.5-flash", "gemini-2.5-pro"]);
  expect(fake.calls).toHaveLength(2);
  expect(fake.calls[1]!.params["pageToken"]).toBe("tok2");
  for (const call of fake.calls) {
    expect(call.base).toBe("https://generativelanguage.googleapis.com/v1beta/models");
    expect(call.headers["x-goog-api-key"]).toBe(KEY);
    expect(call.params).not.toHaveProperty("key");
    expect(call.url).not.toContain("key=");
    expect(call.url).not.toContain(KEY);
  }
});

const DEFAULT_LISTING_URL: Record<string, string> = {
  openai: "https://api.openai.com/v1/models",
  xai: "https://api.x.ai/v1/models",
  deepseek: "https://api.deepseek.com/models",
  qwen: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
  "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
  glm: "https://api.z.ai/api/paas/v4/models",
  "glm-cn": "https://open.bigmodel.cn/api/paas/v4/models",
  minimax: "https://api.minimax.io/v1/models",
  "minimax-cn": "https://api.minimaxi.com/v1/models",
  openrouter: "https://openrouter.ai/api/v1/models",
  vllm: "http://localhost:8000/v1/models",
};

test("every openai-compatible route has a listing url", () => {
  expect(Object.keys(DEFAULT_LISTING_URL).sort()).toEqual([...OPENAI_COMPATIBLE].sort());
});

describe("openai-compatible default host", () => {
  for (const provider of OPENAI_COMPATIBLE) {
    test(provider, async () => {
      const fake = ok({ object: "list", data: [{ id: "z" }, { id: "a" }, { id: "z" }] });
      expect(await listModels(provider, { apiKey: KEY, request: fake })).toEqual(["a", "z"]);
      expect(fake.calls.map((call) => call.url)).toEqual([DEFAULT_LISTING_URL[provider]!]);
      expect(fake.calls[0]!.headers["authorization"]).toBe(`Bearer ${KEY}`);
    });
  }
});

describe("openai custom base url", () => {
  for (const baseUrl of ["https://gateway.example.com/v1", "https://gateway.example.com/v1/"]) {
    test(baseUrl, async () => {
      const fake = ok({ data: [{ id: "m" }] });
      expect(await listModels("openai", { apiKey: KEY, baseUrl, request: fake })).toEqual(["m"]);
      expect(fake.calls.map((call) => call.url)).toEqual(["https://gateway.example.com/v1/models"]);
    });
  }
});

test("routed provider without an endpoint never falls back to openai", async () => {
  // Python adds a routed provider without an endpoint by monkeypatching; the TS tables are frozen,
  // so pin that every routed provider but openai has its own endpoint, and that names the tables
  // only inherit are unsupported.
  for (const provider of OPENAI_COMPATIBLE.filter((name) => name !== "openai")) {
    expect(Object.hasOwn(PROVIDER_BASE_URL, provider)).toBe(true);
  }
  for (const provider of ["constructor", "toString", "__proto__"]) {
    assertClean(await rejected(listModels(provider, { apiKey: KEY, request: neverCalled })), "unsupported provider");
  }
});

describe("ollama lists tags", () => {
  const rows: [string | null, string][] = [
    [null, "http://localhost:11434/api/tags"],
    ["http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/api/tags"],
    ["http://127.0.0.1:11434/v1/", "http://127.0.0.1:11434/api/tags"],
  ];
  for (const [baseUrl, url] of rows) {
    test(String(baseUrl), async () => {
      const fake = ok({ models: [{ name: "qwen3:8b" }, { name: "llama3.2:latest" }] });
      expect(await listModels("ollama", { baseUrl, request: fake })).toEqual(["llama3.2:latest", "qwen3:8b"]);
      expect(fake.calls.map((call) => call.url)).toEqual([url]);
    });
  }
});

test("ollama sends no key even when given one", async () => {
  // No key is sent, so a LAN http host is not held to the base_url rules.
  const fake = ok({ models: [{ name: "m" }] });
  expect(await listModels("ollama", { apiKey: KEY, baseUrl: "http://192.168.1.5:11434", request: fake })).toEqual([
    "m",
  ]);
  const call = fake.calls[0]!;
  expect(call.url).toBe("http://192.168.1.5:11434/api/tags");
  expect(AUTH_HEADERS.filter((name) => name in call.headers)).toEqual([]);
  expect(JSON.stringify(call)).not.toContain(KEY);
});

test("azure has no listing", async () => {
  expect(await listModels("azure", { apiKey: KEY, request: neverCalled })).toBeNull();
});

describe("unknown provider raises", () => {
  for (const provider of ["nope", null]) {
    test(String(provider), async () => {
      assertClean(await rejected(listModels(provider, { apiKey: KEY, request: neverCalled })), "unsupported provider");
    });
  }
});

describe("no key sends no auth header even with env keys", () => {
  const rows: [string, unknown][] = [
    ["openai", { data: [] }],
    ["anthropic", { data: [], has_more: false }],
    ["google", { models: [] }],
    ["ollama", { models: [] }],
  ];
  for (const [provider, body] of rows) {
    test(provider, async () => {
      const names = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"];
      const saved = names.map((name) => process.env[name]);
      for (const name of names) process.env[name] = "env-key-must-not-be-sent";
      try {
        const fake = ok(body);
        expect(await listModels(provider, { request: fake })).toEqual([]);
        expect(AUTH_HEADERS.filter((name) => name in fake.calls[0]!.headers)).toEqual([]);
        expect(JSON.stringify(fake.calls)).not.toContain("env-key-must-not-be-sent");
      } finally {
        names.forEach((name, at) => {
          if (saved[at] === undefined) delete process.env[name];
          else process.env[name] = saved[at];
        });
      }
    });
  }
});

describe("invalid request sends nothing", () => {
  const rows: [string, string | null, unknown, unknown][] = [
    // A key is sent, so the base URL must pass validateBaseUrl.
    ["openai", "http://gateway.example.com/v1", KEY, 8],
    ["openai", "https://gateway.example.com/v1?x=1", KEY, 8],
    ["anthropic", "https://user:pw@gateway.example.com", KEY, 8],
    ["google", "https://gateway.example.com/\n", KEY, 8],
    // Not a URL at all, even without a key.
    ["openai", "gateway.example.com/v1", null, 8],
    ["vllm", "ftp://localhost/v1", null, 8],
    ["vllm", "http://[::1/v1", null, 8],
    // A key the HTTP client would refuse in a header.
    ["openai", null, `${KEY}\n`, 8],
    ["openai", null, "sk-é", 8],
    ["openai", null, new TextEncoder().encode("bytes-key"), 8],
    // The timeout.
    ["openai", null, KEY, 0],
    ["openai", null, KEY, -1],
    ["openai", null, KEY, Number.NaN],
    ["openai", null, KEY, Number.POSITIVE_INFINITY],
    ["openai", null, KEY, "8"],
  ];
  for (const [provider, baseUrl, apiKey, timeout] of rows) {
    test(`${provider} ${JSON.stringify(baseUrl)} ${inspect(apiKey)} ${String(timeout)}`, async () => {
      const options = { apiKey: apiKey as string, baseUrl, timeout: timeout as number, request: neverCalled };
      assertClean(await rejected(listModels(provider, options)), "invalid request");
    });
  }
});

test("keyless http base url is allowed", async () => {
  const fake = ok({ data: [{ id: "m" }] });
  expect(await listModels("vllm", { baseUrl: "http://10.0.0.5:8000/v1", request: fake })).toEqual(["m"]);
});

describe("http error carries only the status", () => {
  for (const provider of ["openai", "anthropic", "google", "openrouter"]) {
    test(provider, async () => {
      const fake = server(() => new HttpResponse(401, { error: { message: `${BODY_MARKER}: ${KEY}` } }));
      assertClean(await rejected(listModels(provider, { apiKey: KEY, request: fake })), "HTTP 401");
    });
  }
});

test("refused redirect is an http status", async () => {
  const fake = server(() => new HttpResponse(302, null));
  assertClean(await rejected(listModels("anthropic", { apiKey: KEY, request: fake })), "HTTP 302");
  expect(fake.calls).toHaveLength(1);
});

describe("transport failures map to a clean reason", () => {
  const rows: [Error, string][] = [
    [new ProviderError("timeout", `slow ${KEY}`), "timed out"],
    [new ProviderError("connection", `refused ${KEY}`), "can't reach host"],
    [new ProviderError("server", `reset ${KEY}`), "can't reach host"],
    [new TypeError(`fetch failed ${KEY}`), "can't reach host"],
    [new Error(`${BODY_MARKER} ${KEY}`), "can't reach host"],
    [new ProviderError("protocol", `too large ${KEY}`), "invalid response"],
  ];
  for (const [raised, message] of rows) {
    test(`${raised.name} ${raised instanceof ProviderError ? raised.kind : ""}`, async () => {
      assertClean(await rejected(listModels("openai", { apiKey: KEY, request: raising(raised) })), message);
    });
  }
});

test("a caller abort rejects with signal.reason", async () => {
  const controller = new AbortController();
  const reason = new Error("stop");
  const fake = async (): Promise<HttpResponse> => {
    controller.abort(reason);
    throw new ProviderError("connection", "aborted");
  };
  expect(await rejected(listModels("openai", { apiKey: KEY, signal: controller.signal, request: fake }))).toBe(reason);
  // The default request rejects before sending anything once the signal has aborted.
  const options = { apiKey: KEY, baseUrl: "http://127.0.0.1:9/v1", signal: controller.signal };
  expect(await rejected(listModels("openai", options))).toBe(reason);
});

describe("malformed response is invalid", () => {
  const rows: unknown[] = [
    new HttpResponse(200, null),
    new HttpResponse(200, [{ id: "m" }]),
    new HttpResponse(200, { data: "m" }),
    new HttpResponse(200, { data: [{ id: 5 }] }),
    new HttpResponse(200, { data: [{ id: 5n }] }),
    new HttpResponse(200, { data: [{ id: { x: 1 } }] }),
    new HttpResponse(200, { models: [] }),
    { data: [{ id: "m" }] },
  ];
  rows.forEach((response, at) => {
    test(`row ${at}`, async () => {
      const fake = async () => response as HttpResponse;
      assertClean(await rejected(listModels("openai", { apiKey: KEY, request: fake })), "invalid response");
    });
  });
});

test("timeout is shared by every page", async () => {
  // Each page takes 150 ms and always has more; a 400 ms budget allows three, each getting what's left.
  const scheduler = virtualScheduler();
  const fake = server(async () => {
    await scheduler.sleep(150);
    return new HttpResponse(200, { data: [{ id: "m" }], has_more: true, last_id: "m" });
  });
  const error = await rejected(listModels("anthropic", { apiKey: KEY, timeout: 0.4, scheduler, request: fake }));
  assertClean(error, "timed out");
  expect(fake.calls.map((call) => call.timeoutMs)).toEqual([400, 250, 100]);
});

/** Start `server` on a loopback port and return its URL root. */
async function listen(server: http.Server): Promise<string> {
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

/** Stop `server`, dropping any open connection. */
async function stop(server: http.Server): Promise<void> {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
}

test("slow drip body stops at the deadline", async () => {
  // Real transport over loopback: the server sends headers, then a byte every 50 ms.
  let closed = false;
  const drip = http.createServer((_req, res) => {
    res.writeHead(200, { "Content-Length": "100000" });
    res.write("{");
    let sent = 0;
    const timer = setInterval(() => (sent++ < 100 ? res.write(" ") : clearInterval(timer)), 50);
    res.on("close", () => {
      clearInterval(timer);
      closed = true;
    });
  });
  const root = await listen(drip);
  try {
    const started = performance.now();
    const error = await rejected(listModels("openai", { apiKey: KEY, baseUrl: `${root}/v1`, timeout: 0.5 }));
    const elapsed = performance.now() - started;
    assertClean(error, "timed out");
    expect(elapsed).toBeLessThan(1000);
    for (let wait = 0; !closed && wait < 20; wait++) await new Promise((resolve) => setTimeout(resolve, 50));
    expect(closed).toBe(true);
  } finally {
    await stop(drip);
  }
});

describe("listing sends connection headers beside the key", () => {
  const rows: [string, unknown, string][] = [
    ["openai", { data: [{ id: "m" }] }, "authorization"],
    ["openrouter", { data: [{ id: "m" }] }, "authorization"],
    ["anthropic", { data: [{ id: "m" }], has_more: false }, "x-api-key"],
    ["google", { models: [] }, "x-goog-api-key"],
  ];
  for (const [provider, body, keyHeader] of rows) {
    test(provider, async () => {
      const fake = ok(body);
      await listModels(provider, { apiKey: KEY, headers: { "X-Gateway": GATEWAY }, request: fake });
      const headers = fake.calls[0]!.headers;
      expect(headers["x-gateway"]).toBe(GATEWAY);
      expect(headers[keyHeader]).toContain(KEY);
    });
  }
});

test("listing takes header values as strings", async () => {
  // Python's other half passes a SecretStr; TS has no such wrapper.
  const fake = ok({ data: [{ id: "m" }] });
  await listModels("openai", { apiKey: KEY, headers: { "X-Gateway": GATEWAY }, request: fake });
  expect(fake.calls[0]!.headers["x-gateway"]).toBe(GATEWAY);
});

test("openai listing sends organization and project", async () => {
  const fake = ok({ data: [{ id: "m" }] });
  await listModels("openai", { apiKey: KEY, organization: ORG, project: PROJECT, request: fake });
  const headers = fake.calls[0]!.headers;
  expect(headers["openai-organization"]).toBe(ORG);
  expect(headers["openai-project"]).toBe(PROJECT);
  expect(headers["authorization"]).toBe(`Bearer ${KEY}`);
});

test("connection headers never replace the protocol headers", async () => {
  const fake = ok({ data: [{ id: "m" }], has_more: false });
  await listModels("anthropic", { apiKey: KEY, headers: { "Anthropic-Version": "1999-01-01" }, request: fake });
  const call = fake.calls[0]!;
  // One spelling only, since the HTTP client would join two into one value.
  expect(call.names.filter((name) => name.toLowerCase() === "anthropic-version")).toEqual(["anthropic-version"]);
  expect(call.headers["anthropic-version"]).toBe("2023-06-01");
});

describe("invalid connection settings send nothing", () => {
  const rows: [string, string | null, Record<string, unknown>][] = [
    // Settings travel with a key: none without one, and never to keyless ollama.
    ["openai", null, { headers: { "X-Gateway": GATEWAY } }],
    ["vllm", null, { headers: { "X-Gateway": GATEWAY } }],
    ["ollama", KEY, { headers: { "X-Gateway": GATEWAY } }],
    ["openai", null, { organization: ORG }],
    // Organization and project are OpenAI's own.
    ["anthropic", KEY, { organization: ORG }],
    ["openrouter", KEY, { project: PROJECT }],
    // Refused names and values.
    ["openai", KEY, { headers: { Authorization: GATEWAY } }],
    ["anthropic", KEY, { headers: { "x-api-key": GATEWAY } }],
    ["openai", KEY, { headers: { "OpenAI-Organization": ORG } }],
    ["openai", KEY, { headers: { "X-Gateway": `${GATEWAY}\n` } }],
    ["openai", KEY, { headers: { "X-Gateway": `${GATEWAY}é` } }],
    ["openai", KEY, { headers: { "X Gateway": GATEWAY } }],
    ["openai", KEY, { headers: { "X-Gateway": 5 } }],
    ["openai", KEY, { headers: { "x-a": GATEWAY, "X-A": GATEWAY } }],
    ["openai", KEY, { headers: [["X-Gateway", GATEWAY]] }],
    ["openai", KEY, { organization: `${ORG} x` }],
    ["openai", KEY, { project: "" }],
  ];
  rows.forEach(([provider, apiKey, settings], at) => {
    test(`row ${at}: ${provider}`, async () => {
      const error = await rejected(listModels(provider, { apiKey, request: neverCalled, ...settings }));
      assertClean(error, "invalid request");
      assertNoSettings(error);
    });
  });
});

test("gateway header name openai refuses goes to other providers", async () => {
  const fake = ok({ data: [{ id: "m" }] });
  await listModels("openrouter", { apiKey: KEY, headers: { "OpenAI-Organization": ORG }, request: fake });
  expect(fake.calls[0]!.headers["openai-organization"]).toBe(ORG);
});

test("http error never echoes connection settings", async () => {
  const echoed = `${BODY_MARKER}: ${GATEWAY} ${ORG} ${PROJECT}`;
  const fake = server(() => new HttpResponse(401, { error: { message: echoed } }));
  const options = { apiKey: KEY, headers: { "X-Gateway": GATEWAY }, organization: ORG, project: PROJECT };
  const error = await rejected(listModels("openai", { ...options, request: fake }));
  assertClean(error, "HTTP 401");
  assertNoSettings(error);
});

describe("transport failure never echoes connection settings", () => {
  const rows = [new Error(`${GATEWAY} ${ORG}`), new TypeError(GATEWAY), new ProviderError("protocol", PROJECT)];
  for (const raised of rows) {
    test(raised.name, async () => {
      const options = { apiKey: KEY, headers: { "X-Gateway": GATEWAY }, organization: ORG, project: PROJECT };
      const error = await rejected(listModels("openai", { ...options, request: raising(raised) }));
      expect(error).toBeInstanceOf(ModelListError);
      expect("cause" in error).toBe(false);
      assertNoSettings(error);
    });
  }
});

test("default transport does not follow a redirect with the headers", async () => {
  // Real transport over loopback: the first host redirects to a second origin, which must never be
  // contacted, so the gateway header goes only to the host asked.
  const seen: http.IncomingHttpHeaders[] = [];
  let contacted = 0;
  const second = http.createServer((_req, res) => res.end("{}"));
  second.on("connection", () => contacted++);
  const target = `${await listen(second)}/v1/models`;
  const first = http.createServer((req, res) => {
    seen.push(req.headers);
    res.writeHead(307, { Location: target, "Content-Length": "0", Connection: "close" });
    res.end();
  });
  const root = await listen(first);
  try {
    const options = { apiKey: KEY, baseUrl: `${root}/v1`, headers: { "X-Gateway": GATEWAY }, timeout: 1 };
    const error = await rejected(listModels("openai", options));
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(contacted).toBe(0);
    expect(seen).toHaveLength(1);
    expect(seen[0]!["x-gateway"]).toBe(GATEWAY);
    assertClean(error, "HTTP 307");
    assertNoSettings(error);
  } finally {
    await Promise.all([stop(first), stop(second)]);
  }
});

test("empty headers need no key", async () => {
  const fake = ok({ models: [] });
  expect(await listModels("ollama", { headers: {}, request: fake })).toEqual([]);
});
