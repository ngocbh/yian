/**
 * The LLM transport's exit scenarios, written once for both runtimes: plain async functions
 * using `node:assert/strict`, so the same checks run as Bun tests (`transport.test.ts`) and,
 * built, on Node, where this module is the main one and prints one JSON line per scenario.
 */

import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { fileURLToPath } from "node:url";

import { RedirectError, realScheduler, safeFetch, type Scheduler, type Send, USER_AGENT, virtualScheduler } from "../../src/platform.ts";
import { type Auth, BearerAuth, HeaderKeyAuth } from "../../src/llm/auth.ts";
import { type ErrorKind, errorFromResponse, ProviderError } from "../../src/llm/errors.ts";
import { type ChatModel, collect, Finish, TextDelta } from "../../src/llm/events.ts";
import { callWithRetries, type RetryInfo, RouteModel, type SseProtocol } from "../../src/llm/executor.ts";
import { Message, Request, Response as ModelResponse, TextPart } from "../../src/llm/messages.ts";
import { Route } from "../../src/llm/route.ts";
import { CASSETTE_REDACTED, CassetteRecorder, ScriptedModel } from "../../src/llm/testing.ts";
import { openExchange, shortRequest } from "../../src/llm/transport.ts";

// The "/" gives the key a JSON-escaped form (`\/`) that differs from the plain one.
const KEY = "sk-test/key-0123456789abcdef";
const ESCAPED_KEY = KEY.replaceAll("/", "\\/");
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
  assert.ok(error.message.includes("[REDACTED]"), `${what}: ${error.message}`);
  for (const text of [error.message, error.stack ?? "", String(error)]) {
    assert.ok(!text.includes(KEY) && !text.includes(ESCAPED_KEY), `${what} holds the key`);
  }
}

/** An echoed key never reaches a status or transport error, and a recorded cassette holds none. */
async function redaction(): Promise<void> {
  const echo = (key: string) => `{"error": {"message": "Incorrect API key provided: ${key}."}}`;
  // `\/` and `e` mixed: only the parse undoes this, so the executor rebuilds the error.
  const mixed = ESCAPED_KEY.replace("key", "k\\u0065y");
  for (const [form, body] of [["plain", echo(KEY)], ["JSON-escaped", echo(ESCAPED_KEY)], ["mixed-escape", echo(mixed)]]) {
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

/** The scenarios by name, in the order `main` runs them. */
export const SCENARIOS: Readonly<Record<string, () => Promise<void>>> = {
  "redirects, every client": redirects,
  "headers on the wire": headersOnTheWire,
  "20-minute stream succeeds": twentyMinuteStream,
  "241 s of silence fails": silence,
  "retry table": retryTable,
  redaction,
};

/** Run every scenario, printing `{"name", "ok", "error"?}` per line; return how many failed. */
export async function main(): Promise<number> {
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
  const failures = await main();
  // Exit once stdout has flushed: a pipe's writes are asynchronous on macOS.
  process.stdout.write("", () => process.exit(failures ? 1 : 0));
}
