/**
 * Base URLs and host fields in `LLMConfig`, pinned connections, `splitOllamaBaseUrl`, and the
 * requests each client sends to a fake `send` (it answers 400, so a test checks the request, not
 * a reply). `safeFetch` refuses every 3xx, so one redirect test runs through the factory for
 * Anthropic and Google, over two loopback servers with the default transport.
 */

import { mkdtempSync, rmSync } from "node:fs";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { inspect } from "node:util";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import type { Env } from "../../src/defaults.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { collect } from "../../src/llm/events.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { OAuthCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import {
  azureResourceEndpoint,
  HostConnection,
  hostOnlyFieldError,
  LLMConfig,
  popApiKey,
  popHostCredentials,
  splitOllamaBaseUrl,
} from "../../src/llm_clients/config.ts";
import {
  checkModelConfig,
  type Connection,
  resolveModelConfig,
  splitModelConfig,
} from "../../src/llm_clients/connection.ts";
import { chatModelFromConfig } from "../../src/llm_clients/factory.ts";
import { ValidationError } from "../../src/llm_clients/provider_blocks.ts";
import { type Send, virtualScheduler } from "../../src/platform.ts";

const SENTINEL = "sk-sentinel-7d1f3b20";
const AZURE_V1 = "https://res.openai.azure.com/openai/v1/";
const AZURE_RESOURCE = "https://res.openai.azure.com/";
const GATEWAY = "https://gateway.example.com/";

// Each test's environment: a private store in a fresh directory and no key, endpoint or default.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-base-url-"));
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Return each field of `host`, read through its getters. */
function fields(host: HostConnection): Record<string, unknown> {
  const { apiKey, baseUrl, organization, project, headers } = host;
  return { apiKey, baseUrl, organization, project, headers };
}

const NO_CONNECTION = { apiKey: null, baseUrl: null, organization: null, project: null, headers: null };

/** Return the error `run` throws. */
function caught(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Return the error that building an `LLMConfig` from `data` throws. */
function thrown(data: Record<string, unknown>): ValidationError {
  try {
    new LLMConfig(data);
  } catch (error) {
    expect(error).toBeInstanceOf(ValidationError);
    return error as ValidationError;
  }
  throw new Error("expected a ValidationError");
}

test("a set base_url dumps and an unset one adds nothing", () => {
  const config = new LLMConfig({ provider: "openai", model: "gpt-5.4", base_url: AZURE_V1, api_key: SENTINEL });
  expect(config.dump()).toEqual({ provider: "openai", model: "gpt-5.4", base_url: AZURE_V1 });
  expect(JSON.stringify(config)).not.toContain(SENTINEL);
  // Recorded snapshots compare full dumps, so an unset base_url must add nothing.
  expect(Object.hasOwn(new LLMConfig({ provider: "openai", model: "gpt-5.5" }).dump(), "base_url")).toBe(false);
});

test.each([
  AZURE_V1,
  "https://gateway.example.com",
  "http://localhost:8000/v1",
  "http://127.0.0.1:11434/v1",
  "http://[::1]:8000/v1",
  "http://[::1]/v1",
  "https://[::1]:8443",
  "https://gateway.example.com:443/v1",
  "https://gateway.example.com:65535/v1",
  "https://xn--gtewy-hra.example.com/v1",
  "https://gateway.example.com/openai/v1/deployments/a-b_c~d%20e",
  GATEWAY + "a".repeat(2048 - GATEWAY.length),
])("https and local http url %# is accepted", (url) => {
  expect(new LLMConfig({ provider: "vllm", base_url: url, api_key: SENTINEL }).baseUrl).toBe(url);
});

test.each([
  "http://gateway.example.com/v1",
  "http://localhost.example.com/v1",
  "ftp://localhost/v1",
  "res.openai.azure.com/openai/v1/",
  "https://",
  "https://user:pw@res.openai.azure.com/openai/v1/",
  "https://res.openai.azure.com/openai/v1/?api-version=1",
  "https://res.openai.azure.com/openai/v1/#frag",
  "https://[bad/v1",
  "https://res.openai.azure.com/openai/v1/?",
  "https://res.openai.azure.com/openai/v1/#",
  "https://user@res.openai.azure.com/openai/v1/",
  "//res.openai.azure.com/openai/v1/",
  // Whitespace and control characters.
  " https://res.openai.azure.com/openai/v1/",
  "https://res.openai.azure.com/openai/v1/ ",
  "https://res.openai.azure.com/openai/v1/\n",
  "https://res.openai.azure.com/openai/v1/\r\n",
  "https://res.openai.\tazure.com/openai/v1/",
  "http://local\thost/v1",
  "https://res.openai azure.com/openai/v1/",
  "https://res.openai.azure.com/openai/v1/\x00",
  "https://res.openai.azure.com/openai/v1/\x7f",
  "https://res.openai.azure.com/openai/v1/\u00a0",
  // Ports.
  "https://gateway.example.com:abc/v1",
  "https://gateway.example.com:99999/v1",
  "https://gateway.example.com:65536/v1",
  "https://gateway.example.com:0/v1",
  "https://gateway.example.com:/v1",
  "https://gateway.example.com:+443/v1",
  "https://gateway.example.com:4_43/v1",
  // Hosts: ASCII only, no IDNA normalisation, exact local names.
  "https://gateway\u3002example.com/v1",
  "https://g\u00e4teway.example.com/v1",
  "https://\uff47ateway.example.com/v1",
  "https://%67ateway.example.com/v1",
  "https://gateway.example.com\\@collector.example.com/v1",
  "http://localhost.evil.com/v1",
  "http://127.0.0.1.nip.io/v1",
  "http://127.0.0.1@gateway.example.com/v1",
  "http://[0:0:0:0:0:0:0:1]:8000/v1",
  "https://[::1%25lo0]/v1",
  "https://[gateway.example.com]/v1",
  "http://[::1].example.com/v1",
  // The scheme must be lowercase, as written.
  "HTTPS://res.openai.azure.com/openai/v1/",
  "Https://res.openai.azure.com/openai/v1/",
  "HTTP://localhost:8000/v1",
  // Length.
  "https://" + "a".repeat(100_000) + ".example.com/v1",
  GATEWAY + "a".repeat(2049 - GATEWAY.length),
])("other url %# is rejected without echoing it", (url) => {
  const message = thrown({ provider: "openai", base_url: url, api_key: SENTINEL }).message;
  expect(message).toContain("base_url");
  // The shape message contains "https://", so the bare "https://" value can't be checked for an echo.
  if (url !== "https://") expect(message).not.toContain(url);
  expect(message).not.toContain(SENTINEL);
});

test.each([
  "https://res.openai.azure.com/openai/v1/?api-version=1",
  "https://res.openai.azure.com/openai/v1/#frag",
  "https://res.openai.azure.com/openai/v1/?",
  "https://res.openai.azure.com/openai/v1/#",
  "https://res.openai.azure.com/openai/v1/a?b/c",
  "https://res.openai.azure.com?api-version=1",
  "https://res.openai.azure.com#frag",
  "http://127.0.0.1:1/v1?key=" + SENTINEL,
  "http://[::1]:8000/v1#f",
  "http://[::1]?x=1",
])("a url with a query or fragment %# is rejected as a query or fragment", (url) => {
  const message = thrown({ provider: "openai", base_url: url, api_key: SENTINEL }).message;
  expect(message).toContain("without credentials, query, or fragment");
  expect(message).not.toContain("ASCII URL characters");
  expect(message).not.toContain(url);
  expect(message).not.toContain(SENTINEL);
});

test("a url with whitespace and a query gets the character message", () => {
  const url = "https://res.openai azure.com/openai/v1/?x=1";
  const message = thrown({ provider: "openai", base_url: url, api_key: SENTINEL }).message;
  expect(message).toContain("ASCII URL characters");
  expect(message).not.toContain(url);
});

test.each([[{ provider: "openai" }], [{ api_key: SENTINEL }], [{}]])(
  "base_url needs a key and provider in the same config %#",
  (fields) => {
    const message = thrown({ base_url: AZURE_V1, ...fields }).message;
    expect(message).toContain("api_key and llm_config.provider");
    expect(message).not.toContain(SENTINEL);
  },
);

test("ollama rejects base_url", () => {
  expect(thrown({ provider: "ollama", base_url: AZURE_V1, api_key: SENTINEL }).message).toContain(
    "provider ollama does not take",
  );
});

test("azure takes its resource endpoint as base_url with or without a key", () => {
  const config = new LLMConfig({ provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE });
  expect(config.dump()).toEqual({ provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE });
  const keyed = new LLMConfig({ provider: "azure", base_url: AZURE_RESOURCE, api_key: SENTINEL });
  expect(keyed.baseUrl).toBe(AZURE_RESOURCE);
  expect(thrown({ provider: "azure", base_url: "http://res.openai.azure.com/" }).message).toContain("https");
});

test("an authored layer may set base_url only beside provider azure", () => {
  expect(hostOnlyFieldError({ provider: "azure", base_url: AZURE_RESOURCE })).toBeNull();
  for (const layer of [{ base_url: AZURE_RESOURCE }, { provider: "openai", base_url: AZURE_V1 }]) {
    expect(hostOnlyFieldError(layer)?.[0]).toBe("base_url");
    expect(hostOnlyFieldError(layer)?.[1]).toContain("provider: azure");
  }
  expect(hostOnlyFieldError({ provider: "azure", api_key: SENTINEL })).not.toBeNull();
});

test("popHostCredentials takes the url only with the layer's key", () => {
  let layer: Record<string, unknown> = { provider: "openai", model: "gpt-5.4", base_url: AZURE_V1, api_key: SENTINEL };
  let host = popHostCredentials(layer);
  expect([host.apiKey, host.baseUrl]).toEqual([SENTINEL, AZURE_V1]);
  expect(layer).toEqual({ provider: "openai", model: "gpt-5.4" });
  host = popHostCredentials({ provider: "openai", api_key: SENTINEL });
  expect([host.apiKey, host.baseUrl]).toEqual([SENTINEL, null]);
  expect(fields(popHostCredentials({ provider: "openai" }))).toEqual(NO_CONNECTION);

  let error = caught(() => popHostCredentials({ provider: "openai", base_url: AZURE_V1 }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("api_key");

  // Without a host key, an Azure endpoint is an ordinary layer setting.
  layer = { provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE };
  expect(fields(popHostCredentials(layer))).toEqual(NO_CONNECTION);
  expect(layer).toEqual({ provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE });
  // Empty headers are no headers, so the endpoint stays a layer setting beside them too.
  for (const headers of [{}, { "X-A": undefined }]) {
    layer = { provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE, headers };
    expect(fields(popHostCredentials(layer))).toEqual(NO_CONNECTION);
    expect(layer).toEqual({ provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE });
  }
  // A host Azure key never goes to an endpoint another layer names.
  error = caught(() => popHostCredentials({ provider: "azure", model: "gpt-4o", api_key: SENTINEL }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("base_url");
  expect(error.message).not.toContain(SENTINEL);
  host = popHostCredentials({ provider: "azure", base_url: AZURE_RESOURCE, api_key: SENTINEL });
  expect([host.apiKey, host.baseUrl]).toEqual([SENTINEL, AZURE_RESOURCE]);

  const unsafe = { provider: "openai", base_url: "http://gateway.example.com", api_key: SENTINEL };
  error = caught(() => popHostCredentials(unsafe));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("https");
  expect(error.message).not.toContain("gateway.example.com");
  expect(error.message).not.toContain(SENTINEL);
});

test("azure config without base_url fails the model check", () => {
  expect(() => checkModelConfig({ provider: "azure", model: "gpt-4o" })).toThrow(/set llm_config\.base_url/);
  expect(() => checkModelConfig({ provider: "azure", model: "gpt-4o", api_key: SENTINEL })).toThrow(
    /set llm_config\.base_url/,
  );
  const [validated] = checkModelConfig({ provider: "azure", model: "gpt-4o", base_url: AZURE_RESOURCE });
  expect(validated.baseUrl).toBe(AZURE_RESOURCE);
});

test("an openai base_url pins a host connection to that endpoint", () => {
  const resolved = resolveModelConfig(
    { provider: "openai", model: "gpt-5.4", base_url: AZURE_V1, api_key: SENTINEL },
    { env },
  );
  expect(resolved["base_url"]).toBe(AZURE_V1);
  expect((resolved["_connection"] as Connection).source).toBe("host");
  expect((resolved["_connection"] as Connection).endpoint).toBe(AZURE_V1);
  const { api_key: _key, ...rest } = resolved;
  expect(JSON.stringify(rest)).not.toContain(SENTINEL);
  const [validated, connection] = splitModelConfig(resolved);
  expect([validated.baseUrl, connection!.endpoint]).toEqual([AZURE_V1, AZURE_V1]);
});

test("a base_url must match its pinned connection", async () => {
  const resolved = resolveModelConfig(
    { provider: "openai", model: "gpt-5.4", base_url: AZURE_V1, api_key: SENTINEL },
    { env },
  );
  const moved = {
    ...resolved,
    _connection: { ...(resolved["_connection"] as Connection), endpoint: "https://api.openai.com/v1" },
  };
  const stored = {
    ...resolved,
    _connection: {
      method: "api_key",
      source: "environment",
      credential_id: "a".repeat(64),
      endpoint: AZURE_V1,
      store_path: null,
    },
  };
  for (const config of [moved, stored]) {
    const error = caught(() => splitModelConfig(config));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toContain("host connection endpoint");
    const built = await chatModelFromConfig(config, { env, onWarning: () => {} }).catch((thrown: Error) => thrown);
    expect(built).toBeInstanceOf(RangeError);
    expect((built as Error).message).toContain("host connection endpoint");
  }
});

// --- clients send the key to the base_url --- //

interface Sent {
  url: string;
  headers: Headers;
  body: string;
}

/** A `send` that records each request and answers it 400. */
function recorder(): { requests: Sent[]; send: Send } {
  const requests: Sent[] = [];
  const send: Send = async (input, init) => {
    requests.push({ url: String(input), headers: new Headers(init.headers), body: String(init.body) });
    return new Response(JSON.stringify({ error: { message: "refused" } }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  };
  return { requests, send };
}

const HI = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("hi")] })] });

/** Build `config` with `send` (else the default transport), ask "hi" and return the error, or `null`. */
async function ask(config: Record<string, unknown>, send?: Send): Promise<unknown> {
  // The default transport does real I/O, which a virtual clock would race to its deadline.
  const scheduler = send ? virtualScheduler() : undefined;
  const model = await chatModelFromConfig(config, { env, send, scheduler, onWarning: () => {} });
  return collect(model.stream(HI)).then(
    () => null,
    (thrown: unknown) => thrown,
  );
}

const COMPATIBLE = ["vllm", "deepseek", "openrouter"];

test.each(COMPATIBLE)("openai-compatible clients send the key to the base_url (%s)", async (provider) => {
  // The provider's own endpoint variable and key must not win over the config.
  vars.set("VLLM_BASE_URL", "http://127.0.0.1:9/v1");
  vars.set(`${provider.toUpperCase()}_API_KEY`, "sk-environment-key");
  const { requests, send } = recorder();
  await ask({ provider, model: "m", base_url: "https://gateway.example.com/v1", api_key: SENTINEL }, send);
  expect(requests.map(({ url }) => url)).toEqual(["https://gateway.example.com/v1/chat/completions"]);
  expect(requests[0]!.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
});

test("openai sends the key to an azure v1 base_url", async () => {
  vars.set("OPENAI_API_KEY", "sk-environment-key");
  vars.set("OPENAI_BASE_URL", "https://proxy.example.com/v1");
  // A saved subscription must not take over a keyed config either.
  await new AuthStore(null, { env }).save(
    "openai",
    new OAuthCredentials({
      accessToken: "access-secret",
      refreshToken: "refresh-secret",
      expiresAt: 4102444800,
      accountId: "account",
    }),
    "gpt-5.5",
  );
  const { requests, send } = recorder();
  await ask({ provider: "openai", model: "gpt-5.4", base_url: AZURE_V1, api_key: SENTINEL }, send);
  expect(requests.map(({ url }) => url)).toEqual(["https://res.openai.azure.com/openai/v1/responses"]);
  expect(requests[0]!.headers.get("authorization")).toBe(`Bearer ${SENTINEL}`);
  expect(JSON.parse(requests[0]!.body).model).toBe("gpt-5.4");
});

// Gemini's base_url is the API root, and the route streams with `alt=sse`.
const DEDICATED: [string, string, string, string][] = [
  ["anthropic", "claude-sonnet-4-5", "x-api-key", "v1/messages"],
  ["google", "gemini-2.5-flash", "x-goog-api-key", "v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"],
];

test.each(DEDICATED)("dedicated clients send the key to the base_url (%s)", async (provider, model, header, path) => {
  // Neither the provider's own key nor its endpoint variables may win over the config.
  for (const name of ["ANTHROPIC_BASE_URL", "ANTHROPIC_API_URL", "GOOGLE_GEMINI_BASE_URL"]) {
    vars.set(name, "https://decoy.example.com");
  }
  vars.set(`${provider.toUpperCase()}_API_KEY`, "sk-environment-key");
  const { requests, send } = recorder();
  const error = await ask({ provider, model, base_url: GATEWAY, api_key: SENTINEL }, send);
  expect(error).toBeInstanceOf(ProviderError);
  expect(requests.map(({ url, headers }) => [url, headers.get(header)])).toEqual([[`${GATEWAY}${path}`, SENTINEL]]);
});

const REDIRECT = "dedicated clients do not follow a redirect with the key (%s)";

/** Listen on a free loopback port and return the server's root URL. */
async function listen(server: http.Server): Promise<string> {
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

/** Stop `server`, dropping any open connection. */
async function stop(server: http.Server): Promise<void> {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
}

test.each(DEDICATED)(REDIRECT, async (provider, model, header, path) => {
  // The first server answers 307 to the second, which must never be contacted.
  const seen: http.IncomingMessage[] = [];
  let contacted = 0;
  const second = http.createServer((_req, res) => res.end("{}"));
  second.on("connection", () => contacted++);
  const target = `${await listen(second)}/`;
  const first = http.createServer((req, res) => {
    seen.push(req);
    res.writeHead(307, { Location: target, "Content-Length": "0", Connection: "close" });
    res.end();
  });
  const root = await listen(first);
  try {
    const error = await ask({ provider, model, base_url: `${root}/`, api_key: SENTINEL });
    expect(error).toBeInstanceOf(ProviderError);
    expect((error as ProviderError).kind).toBe("redirect");
    expect(seen.map((req) => [req.url, req.headers[header]])).toEqual([[`/${path}`, SENTINEL]]);
    expect(contacted).toBe(0);
  } finally {
    await Promise.all([stop(first), stop(second)]);
  }
});

const OLLAMA_SHAPE =
  "OLLAMA_BASE_URL must be an http:// or https:// URL of a host, an optional port, and a path, " +
  "without a query or fragment";
const OLLAMA_PORT = "OLLAMA_BASE_URL port must be a number from 1 to 65535";

test.each<[string, string, [string, string] | null]>([
  ["http://gpu-box:11434", "http://gpu-box:11434", null],
  ["https://gpu-box", "https://gpu-box", null],
  ["gpu-box:11434", "http://gpu-box:11434", null],
  ["localhost:11434", "http://localhost:11434", null],
  ["[::1]:11434", "http://[::1]:11434", null],
  ["gpu-box:11434/v1", "http://gpu-box:11434", null],
  ["u:p@gpu-box:11434", "http://gpu-box:11434", ["u", "p"]],
  ["http://ollama_1:11434", "http://ollama_1:11434", null],
  ["http://[FE80::1]:11434", "http://[fe80::1]:11434", null],
  ["http://GPU-Box:11434", "http://gpu-box:11434", null],
  ["HTTP://gpu-box", "http://gpu-box:11434", null],
  ["HTTPS://gpu-box", "https://gpu-box:11434", null],
  ["Http://gpu-box:8080", "http://gpu-box:8080", null],
  ["HTTP://:pw@gpu-box", "http://gpu-box:11434", null],
  ["HTTP://u:p@gpu-box", "http://gpu-box", ["u", "p"]],
  ["http://h/v1", "http://h", null],
  ["http://h/v1/", "http://h", null],
  ["http://h/v1//", "http://h", null],
  ["http://h/a//", "http://h/a", null],
  ["http://h//a/b/", "http://h/a/b", null],
  ["http://h:", "http://h", null],
  ["http://h:011434", "http://h:11434", null],
  ["http://u%40x:p%3Ab@h", "http://h", ["u@x", "p:b"]],
  ["http://u:p@x@h:1", "http://h:1", ["u", "p@x"]],
  ["http://u@h", "http://h", ["u", ""]],
  ["http://@h", "http://h", null],
  ["http://:pw@h", "http://h", null],
])("ollama base url %p is accepted", (value, url, credentials) => {
  expect(splitOllamaBaseUrl(value)).toEqual([url, credentials]);
});

test("ollama base url with a long run of inner slashes splits in linear time", () => {
  // `/\/+$/` backtracks quadratically here: about 6 s for this input.
  const started = performance.now();
  expect(splitOllamaBaseUrl(`http://h${"/".repeat(100_000)}a/`)).toEqual([`http://h/a`, null]);
  expect(performance.now() - started).toBeLessThan(1000);
});

test("ollama userinfo decodes like unquote", () => {
  const user = (userinfo: string) => splitOllamaBaseUrl(`http://${userinfo}@h`)[1];
  expect(user("u%zz:p")).toEqual(["u%zz", "p"]);
  expect(user("%C3%A9:%e2%82%ac")).toEqual(["\u00e9", "\u20ac"]);
  // Invalid UTF-8 becomes U+FFFD per maximal subpart; a BOM stays; a bad escape stays as written.
  expect(user("%FF:%F0%80%80a")).toEqual(["\ufffd", "\ufffd\ufffd\ufffda"]);
  expect(user("%EF%BB%BFx:%E2%82")).toEqual(["\ufeffx", "\ufffd"]);
  expect(user("%zz%41%:%4")).toEqual(["%zzA%", "%4"]);
  expect(user("\u00fc%41:p")).toEqual(["\u00fcA", "p"]);
});

test.each<[unknown, string]>([
  ["gpu-box", OLLAMA_SHAPE],
  ["ftp://h", OLLAMA_SHAPE],
  ["//gpu-box:11434", OLLAMA_SHAPE],
  ["http://:11434", OLLAMA_SHAPE],
  [":11434", OLLAMA_SHAPE],
  ["http:///p", OLLAMA_SHAPE],
  ["http://h:1/p?q=1", OLLAMA_SHAPE],
  ["http://h:1/p#f", OLLAMA_SHAPE],
  [" http://h:1", OLLAMA_SHAPE],
  ["http://h:1\n", OLLAMA_SHAPE],
  ["http://gpu box:1", OLLAMA_SHAPE],
  ["http://h:1/a b", OLLAMA_SHAPE],
  ["http://gp\u00fc-box:1", OLLAMA_SHAPE],
  ["http://h%41:1", OLLAMA_SHAPE],
  ["http://h:abc", OLLAMA_SHAPE],
  ["http://u:secret-pw-9@h:1/?q", OLLAMA_SHAPE],
  [11434, OLLAMA_SHAPE],
  [null, OLLAMA_SHAPE],
  ["http://h:0", OLLAMA_PORT],
  ["http://h:70000", OLLAMA_PORT],
  ["http://u:secret-pw-9@h:0", OLLAMA_PORT],
])("ollama base url %p is refused", (value, message) => {
  const error = caught(() => splitOllamaBaseUrl(value));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe(message);
  if (typeof value === "string") expect(error.message).not.toContain(value);
  expect(error.message).not.toContain("secret-pw-9");
});

describe("edge cases", () => {
  test("azureResourceEndpoint strips /openai/v1, /openai and trailing slashes", () => {
    expect(azureResourceEndpoint(AZURE_V1)).toBe("https://res.openai.azure.com");
    expect(azureResourceEndpoint("https://res.openai.azure.com/openai")).toBe("https://res.openai.azure.com");
    expect(azureResourceEndpoint("https://res.openai.azure.com//openai/v1//")).toBe("https://res.openai.azure.com");
    expect(azureResourceEndpoint(AZURE_RESOURCE)).toBe("https://res.openai.azure.com");
    // Only one suffix is stripped.
    expect(azureResourceEndpoint("https://res.example.com/openai/openai/v1")).toBe("https://res.example.com/openai");
    expect(azureResourceEndpoint("https://res.example.com/v1")).toBe("https://res.example.com/v1");
  });

  test("a HostConnection hides its key and headers and is frozen", () => {
    const host = popHostCredentials({
      provider: "openai",
      api_key: SENTINEL,
      organization: "org-1",
      headers: { "X-Team": "sentinel-header-value" },
    });
    expect(fields(host)).toEqual({
      apiKey: SENTINEL,
      baseUrl: null,
      organization: "org-1",
      project: null,
      headers: { "X-Team": "sentinel-header-value" },
    });
    for (const shown of [JSON.stringify(host), inspect(host, { showHidden: true, depth: null })]) {
      expect(shown).not.toContain(SENTINEL);
      expect(shown).not.toContain("sentinel-header-value");
    }
    expect(JSON.parse(JSON.stringify(host))).toEqual({ baseUrl: null, organization: "org-1", project: null });
    expect(Object.isFrozen(host)).toBe(true);
    expect(Object.isFrozen(host.headers)).toBe(true);
    expect(host.headers).not.toBe(host.headers);
    expect(fields(new HostConnection())).toEqual(NO_CONNECTION);
  });

  test("popHostCredentials removes the fields it takes, null ones too", () => {
    const layer: Record<string, unknown> = {
      provider: "openai",
      api_key: null,
      organization: null,
      project: null,
      headers: null,
      base_url: null,
      model: "m",
    };
    expect(fields(popHostCredentials(layer))).toEqual(NO_CONNECTION);
    expect(layer).toEqual({ provider: "openai", model: "m" });
    expect(Object.keys(layer)).toEqual(["provider", "model"]);
  });

  test("popHostCredentials refuses Azure settings without a key", () => {
    const layer: Record<string, unknown> = { provider: "azure", organization: "org-1", base_url: AZURE_RESOURCE };
    const error = caught(() => popHostCredentials(layer));
    expect(error).toBeInstanceOf(ValidationError);
    expect(error.message).toContain("apply only to provider openai");
    expect(caught(() => popHostCredentials({ provider: "azure", headers: { "X-Team": "t" } })).message).toContain(
      "llm_config.headers needs llm_config.api_key",
    );
    const badUrl = caught(() => popHostCredentials({ provider: "azure", base_url: "http://res.openai.azure.com/" }));
    expect(badUrl).toBeInstanceOf(RangeError);
    expect(badUrl).not.toBeInstanceOf(ValidationError);
    expect(badUrl.message).toContain("https");
  });

  test("hostOnlyFieldError counts a null value and follows the field order", () => {
    expect(hostOnlyFieldError({ api_key: null })).toEqual([
      "api_key",
      "llm_config.api_key is a host credential; pass it in the run's llm_config, not in a flow file",
    ]);
    expect(hostOnlyFieldError({ headers: {}, project: "p" })).toEqual([
      "project",
      "llm_config.project goes wherever the host's api_key is sent; pass it with that key in the run's " +
        "llm_config, or save it with the login by `yian auth`, not in a flow file",
    ]);
    expect(hostOnlyFieldError({ api_key: undefined, model: "m" })).toBeNull();
    expect(hostOnlyFieldError(["api_key"])).toBeNull();
    expect(hostOnlyFieldError(null)).toBeNull();
  });

  test("popApiKey needs a provider", () => {
    for (const provider of [undefined, null]) {
      const layer: Record<string, unknown> = { provider, api_key: SENTINEL };
      const error = caught(() => popApiKey(layer));
      expect(error).toBeInstanceOf(RangeError);
      expect(error).not.toBeInstanceOf(ValidationError);
      expect(error.message).toBe(
        "llm_config.api_key needs llm_config.provider, so the key goes only to that provider",
      );
      expect(Object.hasOwn(layer, "api_key")).toBe(false);
    }
  });

  test("popApiKey names a provider that is set but not a provider", () => {
    for (const provider of [false, "", 0n, 0, [], {}]) {
      const layer: Record<string, unknown> = { provider, api_key: SENTINEL };
      const error = caught(() => popApiKey(layer));
      expect(error).toBeInstanceOf(ValidationError);
      expect((error as ValidationError).issues.map(({ loc, type }) => [loc, type])).toEqual([
        [["provider"], "literal_error"],
      ]);
      expect(error.message).not.toContain(SENTINEL);
      expect(Object.hasOwn(layer, "api_key")).toBe(false);
    }
  });

  test("popApiKey returns a valid key and refuses a bad one", () => {
    const layer: Record<string, unknown> = { provider: "openai", api_key: SENTINEL };
    expect(popApiKey(layer)).toBe(SENTINEL);
    expect(layer).toEqual({ provider: "openai" });
    expect(popApiKey({ api_key: null })).toBeNull();
    expect(caught(() => popApiKey({ provider: "bogus", api_key: SENTINEL }))).toBeInstanceOf(ValidationError);
    expect(caught(() => popApiKey({ provider: "openai", api_key: "" })).message).toContain("must not be empty");
  });
});
