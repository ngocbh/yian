/**
 * The LLM transport's exit scenarios, written once for both runtimes: plain async functions
 * using `node:assert/strict`, so the same checks run as Bun tests (`transport.test.ts`) and,
 * built, on Node, where this module is the main one and prints one JSON line per scenario.
 */

import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { RedirectError, realScheduler, safeFetch, type Scheduler, type Send, USER_AGENT, virtualScheduler } from "../../src/platform.ts";
import { type Auth, BearerAuth, HeaderKeyAuth, REDACTED } from "../../src/llm/auth.ts";
import { type ErrorKind, errorFromResponse, ProviderError, scrubError } from "../../src/llm/errors.ts";
import { type ChatModel, collect, Finish, TextDelta } from "../../src/llm/events.ts";
import { callWithRetries, type RetryInfo, RouteModel, type SseProtocol } from "../../src/llm/executor.ts";
import { Message, Request, Response as ModelResponse, TextPart } from "../../src/llm/messages.ts";
import { Route } from "../../src/llm/route.ts";
import { type Cassette, CASSETTE_REDACTED, CassetteRecorder, CassetteReplay, ScriptedModel } from "../../src/llm/testing.ts";
import { openExchange, shortRequest } from "../../src/llm/transport.ts";
import { OAuthCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import { chatModelFromConfig } from "../../src/llm_clients/factory.ts";
import { redactKeys } from "../../src/llm_clients/redact.ts";

// The "/" gives the key a JSON-escaped form (`\/`) that differs from the plain one.
const KEY = "sk-test/key-0123456789abcdef";
const ESCAPED_KEY = KEY.replaceAll("/", "\\/");
// `\/` and `e` mixed: only the parse undoes this, so the executor rebuilds the error.
const MIXED_KEY = ESCAPED_KEY.replace("key", "k\\u0065y");
const ENDPOINT = "https://api.example.test/v1";
const REQUEST = new Request({ messages: [] });
const S = 1000;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** A local server on 127.0.0.1 and an ephemeral port. */
interface Server {
  url: string;
  /** Drop every connection and stop listening. */
  close(): Promise<void>;
}

/** Start a server answering with `handler`. */
async function listen(handler: http.RequestListener): Promise<Server> {
  const server = http.createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    close: () =>
      new Promise((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

/** Return the error `promise` rejects with, failing when it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new assert.AssertionError({ message: "expected a rejection" });
}

/** Check that `error` is a `ProviderError` of `kind` with `message`. */
function assertProviderError(error: unknown, kind: ErrorKind, message: string): void {
  assert.ok(error instanceof ProviderError, `not a ProviderError: ${String(error)}`);
  assert.equal(error.kind, kind);
  assert.equal(error.message, message);
}

/** A test SSE protocol: each data line is a `TextDelta`, ending at `[DONE]` or the end with a `Finish`. */
function testProtocol(): SseProtocol {
  return {
    name: "test",
    model: "test-model",
    framing: "sse",
    buildBody: () => ({ model: "test-model" }),
    async *parse(events) {
      for await (const event of events) {
        if (event.data === "[DONE]") break;
        yield new TextDelta(0, event.data);
      }
      yield new Finish({ reason: "stop", protocol: "test" });
    },
  };
}

/** A route model for provider `test` at `baseUrl`, sending with `send` on `scheduler`. */
function model(
  options: { baseUrl?: string; auth?: Auth; headers?: Record<string, string>; query?: Record<string, string> },
  send?: Send,
  scheduler?: Scheduler,
): RouteModel {
  const { baseUrl = ENDPOINT, auth = new BearerAuth(KEY), headers, query } = options;
  const route = new Route({ provider: "test", baseUrl, path: "chat", auth, headers, query });
  return new RouteModel({ route, protocol: testProtocol(), send, scheduler });
}

/** A body playing `steps` (virtual time in ms, text) on `clock`, one per read; a cancel stops it. */
function timedBody(clock: Scheduler, steps: [at: number, text: string][]): ReadableStream<Uint8Array> {
  const stop = new AbortController();
  let next = 0;
  return new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        const step = steps[next++];
        if (step === undefined) return controller.close();
        try {
          await clock.sleep(step[0] - clock.now(), stop.signal);
        } catch {
          return; // Cancelled.
        }
        controller.enqueue(encoder.encode(step[1]));
      },
      cancel() {
        stop.abort();
      },
    },
    { highWaterMark: 0 },
  );
}

/** Resolve as `work` does, or fail with `what` after `ms` real milliseconds. */
async function within<T>(work: Promise<T>, ms: number, what: string): Promise<T> {
  const stop = new AbortController();
  const timer = realScheduler.sleep(ms, stop.signal).then(() => {
    throw new assert.AssertionError({ message: what });
  });
  try {
    return await Promise.race([work, timer]);
  } finally {
    stop.abort();
  }
}

/** A 302 to another local server reaches no client as a redirect, and the target is never contacted. */
async function redirects(): Promise<void> {
  let targetHits = 0;
  const target = await listen((_req, res) => {
    targetHits++;
    res.end("stolen");
  });
  const origin = await listen((_req, res) => {
    res.writeHead(302, { location: `${target.url}/steal?token=${encodeURIComponent(KEY)}` });
    res.end();
  });
  try {
    const detail = `redirect to ${target.url}`;
    const error = await rejection(safeFetch(`${origin.url}/v1`));
    assert.ok(error instanceof RedirectError);
    assert.equal(error.message, `HTTP 302 ${detail}`);

    const short = await rejection(shortRequest(`${origin.url}/v1/models`, { method: "GET" }, { name: "models" }));
    assertProviderError(short, "redirect", `models returned HTTP 302 (redirect): ${detail}`);

    for (const auth of [
      new BearerAuth(KEY),
      new HeaderKeyAuth("x-api-key", KEY),
      new HeaderKeyAuth("api-key", KEY),
      new HeaderKeyAuth("x-goog-api-key", KEY),
    ]) {
      const failed = await rejection(collect(model({ baseUrl: `${origin.url}/v1`, auth }).stream(REQUEST)));
      assertProviderError(failed, "redirect", `test returned HTTP 302 (redirect): ${detail}`);
    }
    assert.equal(targetHits, 0);
  } finally {
    await origin.close();
    await target.close();
  }
}

/** The runtime adds only the documented headers to a model request, and none holds the key. */
async function headersOnTheWire(): Promise<void> {
  let captured: http.IncomingHttpHeaders = {};
  const server = await listen((req, res) => {
    captured = req.headers;
    req.resume();
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.end("data: hi\n\ndata: [DONE]\n\n");
  });
  try {
    const reply = await collect(model({ baseUrl: `${server.url}/v1`, headers: { "x-route": "r" } }).stream(REQUEST));
    assert.equal(reply.message.text(), "hi");
  } finally {
    await server.close();
  }
  assert.equal(captured["content-type"], "application/json");
  assert.equal(captured["accept"], "application/json");
  assert.equal(captured["user-agent"], USER_AGENT);
  assert.equal(captured["authorization"], `Bearer ${KEY}`);
  assert.equal(captured["x-route"], "r");
  const ours = new Set(["content-type", "accept", "user-agent", "authorization", "x-route"]);
  const runtimes = new Set(["host", "connection", "content-length", "accept-encoding", "accept-language", "sec-fetch-mode"]);
  for (const [name, value] of Object.entries(captured)) {
    if (!ours.has(name)) assert.ok(runtimes.has(name), `the runtime added header ${name}`);
    if (name !== "authorization") assert.ok(!String(value).includes(KEY), `header ${name} holds the key`);
  }
}

/** A chunk every 10 s for 20 minutes streams to the end through a route model. */
async function twentyMinuteStream(): Promise<void> {
  const clock = virtualScheduler(0);
  const steps = Array.from({ length: 121 }, (_, i): [number, string] => [i * 10 * S, `data: ${i}\n\n`]);
  const send: Send = async () => new Response(timedBody(clock, steps), { status: 200 });
  const reply = await collect(model({}, send, clock).stream(REQUEST));
  assert.equal(reply.message.text(), steps.map((_, i) => i).join(""));
  assert.equal(clock.now(), 1200 * S);
}

/** 241 s of silence fails before the first byte and between chunks; a stalled real server times out idle. */
async function silence(): Promise<void> {
  const open = (steps: [number, string][], clock: Scheduler) =>
    openExchange(
      ENDPOINT,
      { method: "POST", body: "{}" },
      { provider: "test", credentials: () => [], scheduler: clock, send: async () => new Response(timedBody(clock, steps)) },
    );

  const beforeFirst = virtualScheduler(0);
  const first = await rejection(open([[241 * S, "a"]], beforeFirst));
  assertProviderError(first, "timeout", "test request failed (timeout): no response within 240 s");
  assert.equal(beforeFirst.now(), 240 * S);

  const between = virtualScheduler(0);
  const exchange = await open([[0, "a"], [241 * S, "b"]], between);
  const chunks: string[] = [];
  const idle = await rejection(
    (async () => {
      for await (const chunk of exchange.body) chunks.push(decoder.decode(chunk));
    })(),
  );
  assert.deepEqual(chunks, ["a"]);
  assertProviderError(idle, "timeout", "test request failed (timeout): no bytes for 240 s");
  assert.equal(between.now(), 240 * S);

  let socketClosed: Promise<void> | undefined;
  const server = await listen((req, res) => {
    socketClosed = new Promise((resolve) => req.socket.once("close", () => resolve()));
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.write("data: a\n\n");
  });
  try {
    const stalled = await openExchange(
      `${server.url}/v1`,
      { method: "POST", body: "{}" },
      { provider: "test", credentials: () => [], scheduler: realScheduler, firstResponseMs: 5000, idleMs: 200 },
    );
    const seen: string[] = [];
    const error = await rejection(
      (async () => {
        for await (const chunk of stalled.body) seen.push(decoder.decode(chunk));
      })(),
    );
    assert.deepEqual(seen, ["data: a\n\n"]);
    assertProviderError(error, "timeout", "test request failed (timeout): no bytes for 0.2 s");
    assert.ok(socketClosed);
    await within(socketClosed, 3000, "the server never saw the socket close");
  } finally {
    await server.close();
  }
}

const OK = new ModelResponse({
  message: new Message({ role: "assistant", parts: [new TextPart("ok")] }),
  finishReason: "stop",
});
const PARTIAL = new TextDelta(0, "par");

/** A scripted model whose first reply throws `error`, after partial output when `after`, then answers `OK`. */
function once(error: Error, after = false): ScriptedModel {
  return new ScriptedModel([after ? [PARTIAL, error] : [error], OK], { provider: "test" });
}

/** The error `errorFromResponse` builds for a reply with `status` and `headers`. */
function statusError(status: number, headers: Record<string, string> = {}): ProviderError {
  return errorFromResponse(status, new Headers(headers), "", { provider: "test" });
}

/** A route model on `clock` whose first send does `first` with the request's signal, then replies `ok`. */
function routed(first: (signal: AbortSignal) => Promise<Response>): (clock: Scheduler) => ChatModel {
  return (clock) => {
    let calls = 0;
    const send: Send = async (_url, init) =>
      calls++ === 0 ? first(init.signal!) : new Response("data: ok\n\ndata: [DONE]\n\n");
    return model({}, send, clock);
  };
}

/** One retry-table row: its name, the model, the error kind, whether it's retried, and whether partial output came first. */
type Row = [name: string, make: (clock: Scheduler) => ChatModel, kind: ErrorKind, retried: boolean, after?: boolean];

const ROWS: Row[] = [
  ["429", () => once(statusError(429)), "rate_limit", true],
  ["500 after partial output", () => once(statusError(500), true), "server", true, true],
  ["401", () => once(statusError(401)), "auth", false],
  ["402 quota", () => once(statusError(402)), "quota", false],
  ["402 quota with x-should-retry: true", () => once(statusError(402, { "x-should-retry": "true" })), "quota", true],
  [
    "a first-response timeout",
    routed((signal) => new Promise((_, reject) => signal.addEventListener("abort", () => reject(signal.reason)))),
    "timeout",
    true,
  ],
  [
    "a connection failure",
    routed(async () => {
      throw new TypeError("fetch failed", { cause: Object.assign(new Error("connect refused"), { code: "ECONNREFUSED" }) });
    }),
    "connection",
    true,
  ],
  ["a stream with no Finish after partial output", () => new ScriptedModel([[PARTIAL], OK]), "incomplete", true, true],
  [
    "an adapter's incomplete",
    () => once(new ProviderError("incomplete", "stopped with pause_turn", { retryable: false })),
    "incomplete",
    false,
  ],
  ["refused", () => once(new ProviderError("refused", "the model refused")), "refused", false],
  [
    "a redirect",
    routed(async () => new Response(null, { status: 307, headers: { location: "https://other.example/" } })),
    "redirect",
    false,
  ],
];

/** Run `model` with a one-retry budget on `clock`, recording each retry. */
async function runOnce(model: ChatModel, clock: Scheduler): Promise<{ response?: ModelResponse; error?: unknown; retries: RetryInfo[] }> {
  const retries: RetryInfo[] = [];
  const options = { budget: 1, scheduler: clock, random: () => 0, onRetry: (retry: RetryInfo) => void retries.push(retry) };
  try {
    return { response: await callWithRetries(model, REQUEST, options), retries };
  } catch (error) {
    return { error, retries };
  }
}

/** Each kind of failure is retried or stops as the retry policy says; a defect is never retried. */
async function retryTable(): Promise<void> {
  for (const [name, make, kind, retried, after = false] of ROWS) {
    const clock = virtualScheduler(0);
    const { response, error, retries } = await runOnce(make(clock), clock);
    if (retried) {
      assert.equal(response?.message.text(), "ok", `${name}: ${String(error)}`);
      assert.equal(retries.length, 1, name);
      assert.equal(retries[0]!.error.kind, kind, name);
      assert.equal(retries[0]!.partial === null, !after, name);
    } else {
      assert.ok(error instanceof ProviderError, `${name}: ${String(error)}`);
      assert.equal(error.kind, kind, name);
      assert.equal(retries.length, 0, name);
    }
  }
  const defect = new TypeError("adapter bug");
  const buggy = new ScriptedModel([[PARTIAL, defect], OK]);
  const { error, retries } = await runOnce(buggy, virtualScheduler(0));
  assert.equal(error, defect);
  assert.equal(retries.length, 0);
  assert.equal(buggy.requests.length, 1);
}

/** Check that `error` holds no form of the key in its message, stack or string form. */
function assertNoKey(error: unknown, what: string): void {
  assert.ok(error instanceof ProviderError, `${what}: not a ProviderError: ${String(error)}`);
  assert.ok(error.message.includes(REDACTED), `${what}: ${error.message}`);
  for (const text of [error.message, error.stack ?? "", String(error)]) {
    for (const form of [KEY, ESCAPED_KEY, MIXED_KEY]) assert.ok(!text.includes(form), `${what} holds the key`);
  }
}

/** An echoed key never reaches a status or transport error, and a recorded cassette holds none. */
async function redaction(): Promise<void> {
  const echo = (key: string) => `{"error": {"message": "Incorrect API key provided: ${key}."}}`;
  for (const [form, body] of [["plain", echo(KEY)], ["JSON-escaped", echo(ESCAPED_KEY)], ["mixed-escape", echo(MIXED_KEY)]]) {
    const send: Send = async () => new Response(body, { status: 401 });
    const error = await rejection(collect(model({}, send).stream(REQUEST)));
    assertNoKey(error, `a status error echoing the key ${form}`);
    assert.equal((error as ProviderError).kind, "auth");
  }

  const runtimeError = Object.assign(new TypeError(`cannot reach ${ENDPOINT}/?key=${KEY} (${ESCAPED_KEY})`), {
    code: "ECONNRESET",
  });
  const unreachable = await rejection(
    collect(
      model({}, async () => {
        throw runtimeError;
      }).stream(REQUEST),
    ),
  );
  assertNoKey(unreachable, "a transport error");
  assert.equal((unreachable as ProviderError).kind, "connection");

  const auth = new BearerAuth(KEY);
  const reply = `data: {"echo": "${KEY}"}\n\ndata: [DONE]\n\n`;
  const recorder = new CassetteRecorder(async () => new Response(reply, { headers: { "content-type": "text/event-stream" } }), {
    secrets: () => auth.credentials(),
  });
  await collect(model({ auth, query: { key: KEY } }, recorder.send).stream(REQUEST));
  const cassette = recorder.cassette({ description: "redaction" });
  const saved = JSON.stringify(cassette);
  assert.ok(!saved.includes(KEY) && !saved.includes(encodeURIComponent(KEY)), "the cassette holds the key");
  const { request, response } = cassette.interactions[0]!;
  assert.equal(request.headers?.["authorization"], CASSETTE_REDACTED);
  assert.equal(new URL(request.url).searchParams.get("key"), CASSETTE_REDACTED);
  assert.deepEqual(response.chunks, [`data: {"echo": "${CASSETTE_REDACTED}"}\n\n`, "data: [DONE]\n\n"]);

  // A header name is saved as sent, so only the value scan catches a secret there.
  const named = new CassetteRecorder(async () => new Response("ok"), { secrets: ["hidden-name-0123"] });
  await (await safeFetch(`${ENDPOINT}/models`, { headers: { "x-hidden-name-0123": "v" } }, named.send)).text();
  assert.throws(() => named.cassette({ description: "names" }), {
    message: "the cassette still holds a credential (plain) in interactions[0].request.headers (a key)",
  });
}

/** `scrubError` leaves no form of the key in a message, stack, cause or `AggregateError` member. */
async function scrubbedErrors(): Promise<void> {
  const masked = `${KEY.slice(0, 8)}****${KEY.slice(-4)}`;
  const echo = `plain ${KEY}, escaped ${ESCAPED_KEY}, masked ${masked}`;
  const member = new Error(`member: ${echo}`);
  const group = new AggregateError([member, `string member: ${echo}`], `group: ${echo}`, { cause: `string cause: ${echo}` });
  const inner = new ProviderError("auth", `inner: ${echo}`);
  inner.cause = group;
  const outer = new Error(`outer: ${echo}`, { cause: inner });
  const graph = [outer, inner, group, member];
  // Reading each stack first builds it while the message still holds the key.
  for (const error of graph) assert.ok(error.stack?.includes(KEY), `${error.name}'s stack holds the key before scrubbing`);

  assert.equal(scrubError(outer, (text) => redactKeys(text, [KEY])), outer);
  const texts: string[] = [group.cause as string, group.errors[1] as string];
  for (const error of graph) {
    assert.ok(error.message.includes(REDACTED), error.message);
    texts.push(error.message, error.stack ?? "", String(error));
  }
  for (const text of texts) {
    for (const form of [KEY, ESCAPED_KEY, masked]) assert.ok(!text.includes(form), `holds ${form}: ${text}`);
  }
  assert.ok(inner instanceof ProviderError && inner.kind === "auth");
}

/** A Server-Sent Events body: one `data:` line per object, optionally named by `event`. */
function sse(events: [event: string | null, data: object][]): string[] {
  return events.map(([event, data]) => `${event === null ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`);
}

// A one-word reply ("ok") in each wire protocol.
const RESPONSES_OK = sse([
  ["response.output_item.added", { type: "response.output_item.added", output_index: 0, item: { id: "m", type: "message", role: "assistant", content: [] } }],
  ["response.output_text.delta", { type: "response.output_text.delta", output_index: 0, content_index: 0, item_id: "m", delta: "ok" }],
  [
    "response.completed",
    {
      type: "response.completed",
      response: {
        id: "r",
        status: "completed",
        output: [{ id: "m", type: "message", role: "assistant", content: [{ type: "output_text", text: "ok" }] }],
        usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
      },
    },
  ],
]);
const CHAT_OK = [
  ...sse([
    [null, { choices: [{ index: 0, delta: { role: "assistant", content: "ok" }, finish_reason: null }] }],
    [null, { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }],
  ]),
  "data: [DONE]\n\n",
];
const ANTHROPIC_OK = sse([
  ["message_start", { type: "message_start", message: { id: "m", type: "message", role: "assistant", content: [], usage: { input_tokens: 1, output_tokens: 1 } } }],
  ["content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }],
  ["content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } }],
  ["content_block_stop", { type: "content_block_stop", index: 0 }],
  ["message_delta", { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 1 } }],
  ["message_stop", { type: "message_stop" }],
]);
const GEMINI_OK = sse([[null, { candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP", index: 0 }] }]]);
const OLLAMA_OK = [`${JSON.stringify({ model: "qwen3", message: { role: "assistant", content: "ok" }, done: true, done_reason: "stop" })}\n`];

/** One family's route: the config and environment, the URL and credential headers it must send, and the reply. */
interface RouteCase {
  name: string;
  config: Record<string, unknown>;
  env: Record<string, string>;
  url: string;
  credential: Record<string, string>;
  reply: string[];
  /** The reply's content type; defaults to Server-Sent Events. */
  contentType?: string;
  /** A login saved for `openai` before the build. */
  login?: OAuthCredentials;
}

const ROUTE_CASES: RouteCase[] = [
  {
    name: "openai key",
    config: { provider: "openai", model: "gpt-5.5", api_key: "test-openai-key" },
    env: {},
    url: "https://api.openai.com/v1/responses",
    credential: { authorization: "Bearer test-openai-key" },
    reply: RESPONSES_OK,
  },
  {
    name: "subscription",
    config: {},
    env: {},
    login: new OAuthCredentials({
      accessToken: "test-access-token",
      refreshToken: "test-refresh-token",
      // Far ahead, so the build-time read never refreshes.
      expiresAt: 4_102_444_800,
      accountId: "test-account-1",
    }),
    url: "https://chatgpt.com/backend-api/codex/responses",
    credential: { authorization: "Bearer test-access-token", "chatgpt-account-id": "test-account-1" },
    reply: RESPONSES_OK,
  },
  {
    name: "xai",
    config: { provider: "xai", model: "grok-4" },
    env: { XAI_API_KEY: "test-xai-key" },
    url: "https://api.x.ai/v1/chat/completions",
    credential: { authorization: "Bearer test-xai-key" },
    reply: CHAT_OK,
  },
  {
    name: "azure",
    config: { provider: "azure", model: "my-deploy", base_url: "https://res.openai.azure.com/", api_key: "test-azure-key" },
    env: { OPENAI_API_VERSION: "2025-03-01-preview" },
    url: "https://res.openai.azure.com/openai/deployments/my-deploy/chat/completions?api-version=2025-03-01-preview",
    credential: { "api-key": "test-azure-key" },
    reply: CHAT_OK,
  },
  {
    name: "anthropic",
    config: { provider: "anthropic", model: "claude-sonnet-4-5" },
    env: { ANTHROPIC_API_KEY: "test-anthropic-key" },
    url: "https://api.anthropic.com/v1/messages",
    credential: { "x-api-key": "test-anthropic-key", "anthropic-version": "2023-06-01" },
    reply: ANTHROPIC_OK,
  },
  {
    name: "google",
    config: { provider: "google", model: "gemini-2.5-flash" },
    env: { GOOGLE_API_KEY: "test-google-key" },
    url: "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
    credential: { "x-goog-api-key": "test-google-key" },
    reply: GEMINI_OK,
  },
  {
    name: "ollama with userinfo",
    config: { provider: "ollama", model: "qwen3" },
    env: { OLLAMA_BASE_URL: "http://user:pw@gpu-box:11434" },
    url: "http://gpu-box:11434/api/chat",
    credential: { authorization: "Basic dXNlcjpwdw==" },
    reply: OLLAMA_OK,
    contentType: "application/x-ndjson",
  },
];

/**
 * `chatModelFromConfig` builds one model per route family, each sending to its URL with its
 * credential, against inline cassettes (the built tree holds no cassette files).
 */
async function resolverRoutes(): Promise<void> {
  const home = mkdtempSync(join(tmpdir(), "yian-routes-"));
  try {
    for (const each of ROUTE_CASES) {
      const vars = new Map(Object.entries({ ...each.env, YIAN_AUTH_FILE: join(home, `${each.name}.json`) }));
      const env = (name: string) => vars.get(name);
      if (each.login) await new AuthStore(null, { env }).save("openai", each.login, "gpt-5.5");
      const cassette: Cassette = {
        format: "yian.cassette.v1",
        source: "synthetic",
        description: each.name,
        interactions: [
          {
            request: { method: "POST", url: each.url, headers: each.credential, json: CASSETTE_REDACTED },
            response: { status: 200, headers: { "content-type": each.contentType ?? "text/event-stream" }, chunks: each.reply },
          },
        ],
      };
      const replay = new CassetteReplay(cassette);
      const model = await chatModelFromConfig(each.config, {
        env,
        send: replay.send,
        scheduler: virtualScheduler(0),
        onWarning: () => {},
      });
      const reply = await collect(model.stream(new Request({ messages: [new Message({ role: "user", parts: [new TextPart("Hi")] })] })));
      assert.equal(reply.message.text(), "ok", each.name);
      replay.assertDone();
      assert.equal(replay.requests[0]!.url, each.url, each.name);
    }
  } finally {
    rmSync(home, { recursive: true, force: true });
  }
}

/** The scenarios by name, in the order `main` runs them. */
export const SCENARIOS: Readonly<Record<string, () => Promise<void>>> = {
  "redirects, every client": redirects,
  "headers on the wire": headersOnTheWire,
  "20-minute stream succeeds": twentyMinuteStream,
  "241 s of silence fails": silence,
  "retry table": retryTable,
  redaction,
  scrubbedErrors,
  resolverRoutes,
};

/** Run every scenario, printing `{"name", "ok", "error"?}` per line; return how many failed. */
async function main(): Promise<number> {
  let failures = 0;
  for (const [name, run] of Object.entries(SCENARIOS)) {
    try {
      await run();
      console.log(JSON.stringify({ name, ok: true }));
    } catch (error) {
      failures++;
      const text = error instanceof Error ? (error.stack ?? error.message) : String(error);
      console.log(JSON.stringify({ name, ok: false, error: text }));
    }
  }
  return failures;
}

// `import.meta.main` doesn't exist on Node 22.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  // No process.exit: a leaked timer or socket then shows as a hang.
  process.exitCode = (await main()) ? 1 : 0;
}
