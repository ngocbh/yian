/** Tests for routes in `src/llm/route.ts`, and `RouteModel` and `callWithRetries` in `src/llm/executor.ts`. */

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";
import { type Scheduler, type Send, USER_AGENT, virtualScheduler } from "../../src/platform.ts";
import { type Auth, BearerAuth, HeaderKeyAuth, NoAuth } from "../../src/llm/auth.ts";
import { type ErrorKind, errorFromResponse, ProviderError } from "../../src/llm/errors.ts";
import { type ChatModel, type Event, EventStream, Finish, TextDelta, ToolCallDelta } from "../../src/llm/events.ts";
import {
  callWithRetries,
  type NdjsonProtocol,
  type RetryInfo,
  retryDelay,
  RouteModel,
  type SseProtocol,
  type WireProtocol,
} from "../../src/llm/executor.ts";
import { SSEEvent } from "../../src/llm/framing.ts";
import { Message, Request, Response as ModelResponse, TextPart } from "../../src/llm/messages.ts";
import { type Cassette, CassetteReplay, ScriptedModel } from "../../src/llm/testing.ts";
import { Route } from "../../src/llm/route.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

function route(overrides: Partial<ConstructorParameters<typeof Route>[0]> = {}): Route {
  return new Route({
    provider: "openai",
    baseUrl: "https://api.example/v1",
    path: "responses",
    auth: new BearerAuth("sk-test-key-0123456789"),
    ...overrides,
  });
}

/** Return the error `run` throws, failing when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

describe("Route.url()", () => {
  for (const baseUrl of ["https://api.example/v1", "https://api.example/v1/"]) {
    for (const path of ["chat/completions", "/chat/completions"]) {
      test(`joins ${baseUrl} and ${path} with one slash`, () => {
        expect(route({ baseUrl, path }).url()).toBe("https://api.example/v1/chat/completions");
      });
    }
  }

  test("the query is encoded", () => {
    const url = route({ query: { "api-version": "a b&c=d" } }).url();
    const parsed = new URL(url);
    expect(parsed.searchParams.get("api-version")).toBe("a b&c=d");
    expect(url).not.toContain("a b");
    expect(url).not.toContain("&c=d");
    expect(parsed.pathname).toBe("/v1/responses");
  });

  test("no query adds no question mark", () => {
    expect(route({ query: {} }).url()).toBe("https://api.example/v1/responses");
  });
});

describe("Route", () => {
  test("copies query and headers read-only", () => {
    const query = { "api-version": "1" };
    const headers = { "x-extra": "a" };
    const r = route({ query, headers });
    query["api-version"] = "2";
    headers["x-extra"] = "b";
    expect(r.query).toEqual({ "api-version": "1" });
    expect(r.headers).toEqual({ "x-extra": "a" });
    expect(Object.isFrozen(r.query)).toBe(true);
    expect(Object.isFrozen(r.headers)).toBe(true);
    expect(() => {
      (r.query as Record<string, string>)["api-version"] = "3";
    }).toThrow(TypeError);
  });

  test("defaults to no query and no headers", () => {
    const r = route();
    expect(r.query).toEqual({});
    expect(r.headers).toEqual({});
  });

  for (const name of ["x extra", "x:extra"]) {
    test(`header name ${JSON.stringify(name)} throws`, () => {
      const error = thrown(() => route({ headers: { [name]: "v" } }));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe(`route header name ${JSON.stringify(name)} is not a valid HTTP header name`);
    });
  }

  for (const value of ["secret\nvalue", "sécret"]) {
    test(`header value ${JSON.stringify(value)} throws without echoing it`, () => {
      const error = thrown(() => route({ headers: { "x-extra": value } }));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe("route header x-extra must be non-empty HTTP header text");
      expect(error.stack ?? "").not.toContain(value);
    });
  }
});

// RouteModel

const KEY = "sk-test-key-0123456789";
const BODY = { model: "m", input: "héllo ✓", n: 1.5 };
const REQUEST = new Request({ messages: [] });
const encoder = new TextEncoder();
/** A body item: a chunk, an error that ends the body, or a read that never settles. */
const HANG = Symbol("hang");
type Item = string | Uint8Array | Error | typeof HANG;

/** What the fake `send` replies. */
interface Reply {
  status?: number;
  headers?: Record<string, string>;
  /** The body's items, pulled one per read; no body when absent. */
  body?: Item[];
  /** Whether `send` waits until the request signal aborts, then rejects with its reason. */
  hang?: boolean;
  /** An error `send` throws instead of replying. */
  error?: Error;
}

/** One request the fake `send` received, and what happened to its body. */
interface Sent {
  url: string;
  method: string | undefined;
  headers: Headers;
  body: unknown;
  signal: AbortSignal;
  /** How many body items were pulled, and whether the body was cancelled. */
  stats: { pulled: number; cancelled: boolean };
}

/** Return a body that hands out one item per pull, so `stats.pulled` counts the reads. */
function playBody(items: Item[], stats: Sent["stats"]): ReadableStream<Uint8Array> {
  let next = 0;
  return new ReadableStream<Uint8Array>(
    {
      pull(controller) {
        if (next >= items.length) return controller.close();
        const item = items[next++]!;
        if (item === HANG) return new Promise<void>(() => {});
        if (item instanceof Error) return controller.error(item);
        stats.pulled++;
        controller.enqueue(typeof item === "string" ? encoder.encode(item) : item);
      },
      cancel() {
        stats.cancelled = true;
      },
    },
    { highWaterMark: 0 },
  );
}

/** A fake `send` answering with `replies` in order, recording each request in `sent`. */
function fake(...replies: Reply[]): { send: Send; sent: Sent[] } {
  const sent: Sent[] = [];
  const send: Send = async (url, init) => {
    const reply = replies.shift();
    if (reply === undefined) throw new Error("unexpected request");
    const stats = { pulled: 0, cancelled: false };
    const signal = init.signal!;
    sent.push({ url: String(url), method: init.method, headers: new Headers(init.headers), body: init.body, signal, stats });
    if (reply.hang) {
      await new Promise((_, reject) => signal.addEventListener("abort", () => reject(signal.reason)));
    }
    if (reply.error) throw reply.error;
    const body = reply.body === undefined ? null : playBody(reply.body, stats);
    return new Response(body, { status: reply.status ?? 200, headers: reply.headers });
  };
  return { send, sent };
}

/** An SSE reply carrying `texts` as data lines, then `[DONE]`. */
function sseReply(...texts: string[]): Reply {
  return { body: [...texts.map((text) => `data: ${text}\n\n`), "data: [DONE]\n\n"] };
}

/** A protocol that records the frames it is given. */
type Recording<P> = P & { frames: unknown[] };

/** A test SSE protocol: each data line is a `TextDelta`, ending at `[DONE]` or the end with a `Finish`. */
function sse(body: unknown = BODY): Recording<SseProtocol> {
  const frames: unknown[] = [];
  return {
    name: "test",
    model: "test-model",
    framing: "sse",
    frames,
    buildBody: () => body as JsonObject,
    async *parse(events) {
      for await (const event of events) {
        frames.push(event);
        if (event.data === "[DONE]") break;
        yield new TextDelta(0, event.data);
      }
      yield new Finish({ reason: "stop", protocol: "test" });
    },
  };
}

/** A test NDJSON protocol: each line's `text` is a `TextDelta`, then a `Finish`. */
function ndjson(): Recording<NdjsonProtocol> {
  const frames: unknown[] = [];
  return {
    name: "test",
    model: "test-model",
    framing: "ndjson",
    frames,
    buildBody: () => BODY,
    async *parse(values) {
      for await (const value of values) {
        frames.push(value);
        yield new TextDelta(0, (value as { text: string }).text);
      }
      yield new Finish({ reason: "stop", protocol: "test" });
    },
  };
}

/** A model over `route(overrides)` with `protocol`, sending with `send`. */
function model(send: Send, protocol: WireProtocol = sse(), overrides: Parameters<typeof route>[0] = {}): RouteModel {
  return new RouteModel({ route: route({ query: { "api-version": "2026-01-01" }, ...overrides }), protocol, send });
}

/** Read every event of `stream`. */
async function drain(stream: EventStream): Promise<Event[]> {
  const events: Event[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

/** Return the error `promise` rejects with, failing when it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

/** Return the error a call over `auth` ends with, its reply being `reply`. */
async function callError(reply: Reply, auth: Auth): Promise<ProviderError> {
  const { send } = fake(reply);
  const error = await rejection(drain(model(send, sse(), { auth }).stream(REQUEST)));
  expect(error).toBeInstanceOf(ProviderError);
  return error as ProviderError;
}

/** Check that `error` holds no `key` in its message, stack or string form. */
function expectScrubbed(error: ProviderError, key: string): void {
  expect(error.message).toContain("[REDACTED]");
  expect(error.message).not.toContain(key);
  expect(error.stack ?? "").not.toContain(key);
  expect(String(error)).not.toContain(key);
}

describe("RouteModel request", () => {
  test("request shape", async () => {
    const { send, sent } = fake(sseReply("1"));
    await drain(model(send, sse(), { headers: { "x-extra": "a" } }).stream(REQUEST));
    expect(sent).toHaveLength(1);
    const [request] = sent;
    expect(request!.method).toBe("POST");
    expect(request!.url).toBe("https://api.example/v1/responses?api-version=2026-01-01");
    expect(request!.body).toBe('{"model":"m","input":"héllo ✓","n":1.5}');
    expect(request!.headers.get("content-type")).toBe("application/json");
    expect(request!.headers.get("accept")).toBe("application/json");
    expect(request!.headers.get("authorization")).toBe(`Bearer ${KEY}`);
    expect(request!.headers.get("x-extra")).toBe("a");
    expect(request!.headers.get("user-agent")).toBe(USER_AGENT);
  });

  test("auth beats a route header of any case", async () => {
    const { send, sent } = fake(sseReply("1"));
    await drain(model(send, sse(), { headers: { AUTHORIZATION: "x" } }).stream(REQUEST));
    expect(sent[0]!.headers.get("authorization")).toBe(`Bearer ${KEY}`);
  });

  test("route headers override the defaults of any case", async () => {
    const { send, sent } = fake(sseReply("1"));
    const headers = { accept: "text/event-stream", "CONTENT-TYPE": "text/plain" };
    await drain(model(send, sse(), { headers, auth: new NoAuth() }).stream(REQUEST));
    expect(sent[0]!.headers.get("accept")).toBe("text/event-stream");
    expect(sent[0]!.headers.get("content-type")).toBe("text/plain");
    expect(sent[0]!.headers.has("authorization")).toBe(false);
  });

  const refused: [string, unknown][] = [
    ["NaN", { x: NaN }],
    ["Infinity", { x: Infinity }],
    ["a lone surrogate", { text: "a\ud800b" }],
    ["a Map", { x: new Map() }],
  ];
  for (const [name, body] of refused) {
    test(`a body with ${name} throws from the first next() and sends nothing`, async () => {
      const { send, sent } = fake();
      const stream = model(send, sse(body)).stream(REQUEST);
      expect(await rejection(stream.next())).toBeInstanceOf(TypeError);
      expect(sent).toHaveLength(0);
    });
  }

  test("nothing is sent before the first next()", async () => {
    const { send, sent } = fake(sseReply("1"));
    let resolved = 0;
    const auth = new BearerAuth(() => {
      resolved++;
      return KEY;
    });
    const stream = model(send, sse(), { auth }).stream(REQUEST);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(sent).toHaveLength(0);
    expect(resolved).toBe(0);
    expect(await stream.next()).toEqual({ done: false, value: new TextDelta(0, "1") });
    expect(sent).toHaveLength(1);
    expect(resolved).toBe(1);
    await stream.close();
  });

  test("a caller signal aborted before the first next() never calls the auth source", async () => {
    const { send, sent } = fake(sseReply("1"));
    let resolved = 0;
    const auth = new BearerAuth(() => {
      resolved++;
      return KEY;
    });
    const controller = new AbortController();
    const reason = new Error("stop");
    controller.abort(reason);
    const stream = model(send, sse(), { auth }).stream(REQUEST, { signal: controller.signal });
    expect(await rejection(stream.next())).toBe(reason);
    expect(resolved).toBe(0);
    expect(sent).toHaveLength(0);
  });

  test("closing before the first next() sends nothing", async () => {
    const { send, sent } = fake(sseReply("1"));
    const stream = model(send).stream(REQUEST);
    await stream.close();
    expect(await stream.next()).toEqual({ done: true, value: undefined });
    expect(sent).toHaveLength(0);
  });

  test("auth is resolved per call", async () => {
    const keys = ["key-one", "key-two"].values();
    const { send, sent } = fake(sseReply("1"), sseReply("2"));
    const m = model(send, sse(), { auth: new BearerAuth(() => keys.next().value!) });
    await drain(m.stream(REQUEST));
    await drain(m.stream(REQUEST));
    expect(sent.map((request) => request.headers.get("authorization"))).toEqual(["Bearer key-one", "Bearer key-two"]);
  });

  const failed = new Error("refresh failed");
  const sources: [string, () => string | Promise<string>][] = [
    ["a throwing source", () => {
      throw failed;
    }],
    ["a rejecting source", async () => {
      throw failed;
    }],
  ];
  for (const [name, source] of sources) {
    test(`an auth source's error propagates unchanged from ${name} and nothing is sent`, async () => {
      const { send, sent } = fake(sseReply("1"));
      const auth = new BearerAuth(source);
      expect(await rejection(drain(model(send, sse(), { auth }).stream(REQUEST)))).toBe(failed);
      expect(sent).toHaveLength(0);
    });
  }

  test("provider and model come from the route and the protocol", () => {
    const { send } = fake();
    const protocol = sse();
    const m = model(send, protocol, { provider: "example" });
    expect(m.provider).toBe("example");
    expect(m.model).toBe("test-model");
    expect(m.route.provider).toBe("example");
    expect(m.protocol).toBe(protocol);
  });

  test("send and scheduler reach the transport", async () => {
    const { send, sent } = fake({ hang: true });
    const m = new RouteModel({ route: route(), protocol: sse(), send, scheduler: virtualScheduler(0) });
    const error = await rejection(drain(m.stream(REQUEST)));
    expect(error).toBeInstanceOf(ProviderError);
    expect((error as ProviderError).kind).toBe("timeout");
    expect(sent).toHaveLength(1);
  });
});

describe("RouteModel status errors", () => {
  test("a 401 echoing the key is an auth error without the key", async () => {
    const key = "sk-live-secret-key-abcdef";
    const body = JSON.stringify({ error: { message: `Incorrect API key provided: ${key}.` } });
    const error = await callError({ status: 401, body: [body] }, new HeaderKeyAuth("x-api-key", key));
    expect(error.kind).toBe("auth");
    expect(error.status).toBe(401);
    expect(error.provider).toBe("openai");
    expectScrubbed(error, key);
  });

  test("a 401 echoing the key JSON-escaped (\\/) holds no key", async () => {
    const key = "sk-live/secret/key-abcdef";
    const body = JSON.stringify({ error: { message: `Incorrect API key provided: ${key}.` } }).replaceAll("/", "\\/");
    expect(body).not.toContain(key);
    const error = await callError({ status: 401, body: [body] }, new BearerAuth(key));
    expect(error.message).toContain("Incorrect API key provided: [REDACTED].");
    expectScrubbed(error, key);
  });

  test("a 401 echoing the key all \\u-escaped holds no key", async () => {
    const key = "sk-live-secret-key-abcdef";
    const escaped = Array.from(key, (char) => `\\u${char.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")}`);
    const body = `{"error": {"message": "Incorrect API key provided: ${escaped.join("")}."}}`;
    const error = await callError({ status: 401, body: [body] }, new BearerAuth(key));
    expect(error.message).toContain("Incorrect API key provided: [REDACTED].");
    expectScrubbed(error, key);
  });

  test("a 401 echoing the key in a mixed escape only the parse undoes is rebuilt without it", async () => {
    const key = "sk-live-secret-key-abcdef";
    const body = `{"error": {"message": "Incorrect API key provided: sk-live-secret-k\\u0065y-abcdef."}}`;
    const error = await callError(
      { status: 401, headers: { "retry-after": "3" }, body: [body] },
      new BearerAuth(key),
    );
    expect(error.message).toBe("openai returned HTTP 401 (auth): Incorrect API key provided: [REDACTED].");
    expectScrubbed(error, key);
    expect(error.kind).toBe("auth");
    expect(error.status).toBe(401);
    expect(error.retryAfter).toBe(3);
    expect(error.retryable).toBe(false);
    expect(error.provider).toBe("openai");
  });

  test("the longest credential is replaced first", async () => {
    const keys = ["abcdefgh", "abcdefgh-longer"].values();
    const { send } = fake(sseReply("1"), { status: 401, body: ["saw abcdefgh-longer"] });
    const m = model(send, sse(), { auth: new BearerAuth(() => keys.next().value!) });
    await drain(m.stream(REQUEST));
    const error = await rejection(drain(m.stream(REQUEST)));
    expect((error as ProviderError).message.endsWith("saw [REDACTED]")).toBe(true);
  });

  test("a non-retryable status names the body's message", async () => {
    const error = await callError({ status: 400, body: ['{"error": {"message": "bad field"}}'] }, new BearerAuth(KEY));
    expect(error.kind).toBe("bad_request");
    expect(error.message).toContain("bad field");
  });

  test("the error body read stops at 64 KiB and aborts the request", async () => {
    const chunk = new Uint8Array(16 * 1024).fill(0x78);
    const { send, sent } = fake({ status: 400, body: Array.from({ length: 640 }, () => chunk) });
    const error = await rejection(drain(model(send).stream(REQUEST)));
    expect((error as ProviderError).kind).toBe("bad_request");
    expect(sent[0]!.stats.pulled).toBe(4);
    expect(sent[0]!.signal.aborted).toBe(true);
  });

  test("a body read failure after the status keeps what arrived", async () => {
    const reset = Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
    const error = await callError({ status: 500, body: ["upstream ", "exploded", reset] }, new BearerAuth(KEY));
    expect(error.kind).toBe("server");
    expect(error.message).toBe("openai returned HTTP 500 (server): upstream exploded");
  });

  test("a caller abort during the error body read rethrows its reason", async () => {
    const { send, sent } = fake({ status: 500, body: ["partial", HANG] });
    const controller = new AbortController();
    const stream = model(send).stream(REQUEST, { signal: controller.signal });
    const pending = rejection(stream.next());
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(sent[0]!.stats.pulled).toBe(1);
    const reason = new Error("stop");
    controller.abort(reason);
    expect(await pending).toBe(reason);
  });

  test("a 3xx is a redirect error, not retryable, after one request", async () => {
    const { send, sent } = fake({ status: 307, headers: { location: "https://evil.example/steal?k=1" }, body: ["moved"] });
    const error = (await rejection(drain(model(send).stream(REQUEST)))) as ProviderError;
    expect(error).toBeInstanceOf(ProviderError);
    expect(error.kind).toBe("redirect");
    expect(error.retryable).toBe(false);
    expect(error.message).toContain("https://evil.example");
    expect(error.message).not.toContain("/steal");
    expect(sent).toHaveLength(1);
  });
});

describe("RouteModel reply", () => {
  test("SSE frames reach the protocol", async () => {
    const { send } = fake({ body: ["data: Hel\n\nda", "ta: lo\n\n", "data: [DONE]\n\n"] });
    const protocol = sse();
    const events = await drain(model(send, protocol).stream(REQUEST));
    expect(events).toEqual([new TextDelta(0, "Hel"), new TextDelta(0, "lo"), new Finish({ reason: "stop", protocol: "test" })]);
    expect(protocol.frames).toEqual([
      new SSEEvent({ data: "Hel" }),
      new SSEEvent({ data: "lo" }),
      new SSEEvent({ data: "[DONE]" }),
    ]);
  });

  test("NDJSON values reach the protocol", async () => {
    const { send } = fake({ body: ['{"text": "a"}\n', '\n{"te', 'xt": "b"}'] });
    const protocol = ndjson();
    const events = await drain(model(send, protocol).stream(REQUEST));
    expect(events).toEqual([new TextDelta(0, "a"), new TextDelta(0, "b"), new Finish({ reason: "stop", protocol: "test" })]);
    expect(protocol.frames).toEqual([{ text: "a" }, { text: "b" }]);
  });

  test("invalid NDJSON is a protocol error with the route's provider", async () => {
    const { send } = fake({ body: ['{"text": "a"\n'] });
    const error = (await rejection(drain(model(send, ndjson()).stream(REQUEST)))) as ProviderError;
    expect(error).toBeInstanceOf(ProviderError);
    expect(error.kind).toBe("protocol");
    expect(error.provider).toBe("openai");
  });

  test("comment-only SSE gives the protocol no frames", async () => {
    const { send } = fake({ body: [": keep-alive\n\n", ": again\n\n"] });
    const protocol = sse();
    const events = await drain(model(send, protocol).stream(REQUEST));
    expect(protocol.frames).toEqual([]);
    expect(events).toEqual([new Finish({ reason: "stop", protocol: "test" })]);
  });

  test("an SSE parse error with no provider gets the route's", async () => {
    const { send } = fake(sseReply("a", "b"));
    const protocol: SseProtocol = {
      ...sse(),
      async *parse(events) {
        for await (const event of events) {
          yield new TextDelta(0, event.data);
          throw new ProviderError("protocol", "bad frame");
        }
      },
    };
    const stream = model(send, protocol).stream(REQUEST);
    expect(await stream.next()).toEqual({ done: false, value: new TextDelta(0, "a") });
    const error = (await rejection(stream.next())) as ProviderError;
    expect(error.message).toBe("bad frame");
    expect(error.provider).toBe("openai");
  });

  test("an SSE parse error with a provider keeps it", async () => {
    const { send } = fake(sseReply("a"));
    const protocol: SseProtocol = {
      ...sse(),
      async *parse() {
        throw new ProviderError("protocol", "bad frame", { provider: "other" });
      },
    };
    const error = (await rejection(drain(model(send, protocol).stream(REQUEST)))) as ProviderError;
    expect(error.provider).toBe("other");
  });

  test("a parse error echoing the key is rebuilt without it, keeping its fields", async () => {
    const { send } = fake(sseReply("a"));
    const protocol: SseProtocol = {
      ...sse(),
      async *parse() {
        throw new ProviderError("server", `openai stream failed (server): bad key ${KEY}`, { retryable: false });
      },
    };
    const error = (await rejection(drain(model(send, protocol, { auth: new BearerAuth(KEY) }).stream(REQUEST)))) as ProviderError;
    expect(error).toBeInstanceOf(ProviderError);
    expectScrubbed(error, KEY);
    expect(error.message).toBe("openai stream failed (server): bad key [REDACTED]");
    expect(error.kind).toBe("server");
    expect(error.retryable).toBe(false);
    expect(error.provider).toBe("openai");
  });

  test("a parse error without a credential is rethrown as the same object", async () => {
    const { send } = fake(sseReply("a"));
    const failure = new ProviderError("server", "openai stream failed (server): overloaded");
    const protocol: SseProtocol = {
      ...sse(),
      async *parse() {
        throw failure;
      },
    };
    expect(await rejection(drain(model(send, protocol, { auth: new BearerAuth(KEY) }).stream(REQUEST)))).toBe(failure);
  });

  test("an adapter defect propagates unchanged", async () => {
    const { send, sent } = fake(sseReply("a"));
    const defect = new TypeError("adapter bug");
    const protocol: SseProtocol = {
      ...sse(),
      async *parse() {
        throw defect;
      },
    };
    expect(await rejection(drain(model(send, protocol).stream(REQUEST)))).toBe(defect);
    expect(sent[0]!.signal.aborted).toBe(true);
  });
});

describe("RouteModel stream lifetime", () => {
  test("closing the stream early aborts the request", async () => {
    const { send, sent } = fake({ body: ["data: a\n\n", "data: b\n\n", HANG] });
    const stream = model(send).stream(REQUEST);
    expect(await stream.next()).toEqual({ done: false, value: new TextDelta(0, "a") });
    expect(sent[0]!.signal.aborted).toBe(false);
    await stream.close();
    expect(sent[0]!.signal.aborted).toBe(true);
  });

  test("closing the stream while the request is opening aborts it", async () => {
    const { send, sent } = fake({ hang: true });
    const stream = model(send).stream(REQUEST);
    const pending = rejection(stream.next());
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(sent).toHaveLength(1);
    await stream.close();
    expect(sent[0]!.signal.aborted).toBe(true);
    expect(await pending).toBeInstanceOf(Error);
  });

  test("closing a stream whose protocol yielded before reading a frame releases the body", async () => {
    const { send, sent } = fake({ body: ["data: a\n\n", HANG] });
    const protocol: SseProtocol = {
      ...sse(),
      async *parse() {
        yield new TextDelta(0, "early");
      },
    };
    const stream = model(send, protocol).stream(REQUEST);
    expect(await stream.next()).toEqual({ done: false, value: new TextDelta(0, "early") });
    await stream.close();
    expect(sent[0]!.stats.cancelled).toBe(true);
  });

  test("a protocol that throws before reading a frame releases the body", async () => {
    const { send, sent } = fake({ body: ["data: a\n\n", HANG] });
    const protocol: SseProtocol = {
      ...sse(),
      async *parse() {
        throw new ProviderError("protocol", "bad start");
      },
    };
    await rejection(drain(model(send, protocol).stream(REQUEST)));
    expect(sent[0]!.stats.cancelled).toBe(true);
  });

  test("exhaustion closes the request", async () => {
    const { send, sent } = fake(sseReply("a"));
    await drain(model(send).stream(REQUEST));
    expect(sent[0]!.signal.aborted).toBe(true);
  });

  test("an error closes the request", async () => {
    const { send, sent } = fake({ body: ["data: a\n\n", "data: b\n\n", HANG] });
    const protocol: SseProtocol = {
      ...sse(),
      async *parse(events) {
        for await (const event of events) {
          yield new TextDelta(0, event.data);
          throw new ProviderError("protocol", "bad frame");
        }
      },
    };
    await rejection(drain(model(send, protocol).stream(REQUEST)));
    expect(sent[0]!.signal.aborted).toBe(true);
  });

  test("a caller abort ends the stream with the reason", async () => {
    const { send, sent } = fake({ body: ["data: a\n\n", HANG] });
    const controller = new AbortController();
    const stream = model(send).stream(REQUEST, { signal: controller.signal });
    expect(await stream.next()).toEqual({ done: false, value: new TextDelta(0, "a") });
    const pending = rejection(stream.next());
    const reason = new Error("stop");
    controller.abort(reason);
    expect(await pending).toBe(reason);
    expect(sent[0]!.signal.aborted).toBe(true);
    expect(await stream.next()).toEqual({ done: true, value: undefined });
  });
});

/** A cassette answering the test route's request with four SSE chunks, one per read. */
function replayedReply(): Cassette {
  const request = { method: "POST", url: "https://api.example/v1/responses?api-version=2026-01-01", json: BODY };
  const chunks = ["data: a\n\n", "data: b\n\n", "data: c\n\n", "data: [DONE]\n\n"];
  return { format: "yian.cassette.v1", source: "synthetic", description: "test", interactions: [{ request, response: { status: 200, chunks } }] };
}

describe("RouteModel over a cassette", () => {
  test("a stop after the first event ends the stream with the reason", async () => {
    const replay = new CassetteReplay(replayedReply());
    const controller = new AbortController();
    const stream = model(replay.send).stream(REQUEST, { signal: controller.signal });
    expect(await stream.next()).toEqual({ done: false, value: new TextDelta(0, "a") });
    const reason = new Error("stop");
    controller.abort(reason);
    expect(await rejection(drain(stream))).toBe(reason);
  });
});

// callWithRetries

const OK = new ModelResponse({
  message: new Message({ role: "assistant", parts: [new TextPart("ok")] }),
  finishReason: "stop",
});
const PARTIAL = new TextDelta(0, "par");
const RESET = Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });

/** The error `errorFromResponse` builds for an openai reply with `status`, `headers` and `body`. */
function statusError(status: number, headers: Record<string, string> = {}, body = ""): ProviderError {
  return errorFromResponse(status, new Headers(headers), body, { provider: "openai" });
}

/** A scripted model whose first reply throws `error`, after partial output when `after`, then answers `OK`. */
function failingOnce(error: Error, after: boolean): ScriptedModel {
  return new ScriptedModel([after ? [PARTIAL, error] : [error], OK], { provider: "openai" });
}

/** A runtime's connection-refused failure, as fetch reports it. */
function refused(): Error {
  return new TypeError("fetch failed", { cause: Object.assign(new Error("connect refused"), { code: "ECONNREFUSED" }) });
}

/** One retry-table row: its name, whether the failure comes after partial output, the model, and the error kind. */
type Row = [name: string, after: boolean, make: (clock: Scheduler) => ChatModel, kind: ErrorKind];

/** Return the rows for the error `build` makes, before the first event and after partial output. */
function rows(name: string, kind: ErrorKind, build: () => Error): Row[] {
  return [
    [`${name} before the first event`, false, () => failingOnce(build(), false), kind],
    [`${name} after partial output`, true, () => failingOnce(build(), true), kind],
  ];
}

/** A row for a route model on the test's clock whose first reply is `first`, then an `ok` reply. */
function routed(name: string, after: boolean, kind: ErrorKind, first: Reply, protocol: WireProtocol = sse()): Row {
  const second = protocol.framing === "sse" ? sseReply("ok") : { body: ['{"text": "ok"}\n'] };
  const make = (clock: Scheduler) =>
    new RouteModel({ route: route(), protocol, send: fake(first, second).send, scheduler: clock });
  return [name, after, make, kind];
}

const RETRIED: Row[] = [
  ...rows("429", "rate_limit", () => statusError(429)),
  ...rows("529 overloaded", "overloaded", () => statusError(529)),
  ...rows("500", "server", () => statusError(500)),
  ...rows("502", "server", () => statusError(502)),
  ...rows("503", "server", () => statusError(503)),
  ...rows("408", "timeout", () => statusError(408)),
  ...rows("409", "bad_request", () => statusError(409)),
  ...rows("x-should-retry: true on a 400", "bad_request", () => statusError(400, { "x-should-retry": "true" })),
  ...rows("x-should-retry: true on a 402 quota", "quota", () => statusError(402, { "x-should-retry": "true" })),
  routed("a first-response timeout before the first event", false, "timeout", { hang: true }),
  routed("an idle timeout before the first event", false, "timeout", { body: [": ping\n\n", HANG] }),
  routed("an idle timeout after partial output", true, "timeout", { body: ["data: a\n\n", HANG] }),
  routed("a connection failure before the first event", false, "connection", { error: refused() }),
  routed("a connection failure after partial output", true, "connection", { body: ["data: a\n\n", RESET] }),
  ["a stream with no Finish after partial output", true, () => new ScriptedModel([[PARTIAL], OK]), "incomplete"],
];

const STOPPED: Row[] = [
  ...rows("400", "bad_request", () => statusError(400)),
  ...rows("400 context length", "context_length", () =>
    statusError(400, {}, '{"error": {"code": "context_length_exceeded"}}'),
  ),
  ...rows("401", "auth", () => statusError(401)),
  ...rows("403", "permission", () => statusError(403)),
  ...rows("404", "not_found", () => statusError(404)),
  ...rows("402 quota", "quota", () => statusError(402)),
  ...rows("429 insufficient_quota", "quota", () => statusError(429, {}, "insufficient_quota")),
  ...rows("x-should-retry: false on a 500", "server", () => statusError(500, { "x-should-retry": "false" })),
  ...rows("retry-after: 121", "rate_limit", () => statusError(429, { "retry-after": "121" })),
  routed("a redirect before the first event", false, "redirect", {
    status: 307,
    headers: { location: "https://other.example/" },
  }),
  routed("an invalid NDJSON line before the first event", false, "protocol", { body: ["{\n"] }, ndjson()),
  routed("an invalid NDJSON line after partial output", true, "protocol", { body: ['{"text": "a"}\n', "{\n"] }, ndjson()),
  ...rows("incomplete for pause_turn", "incomplete", () =>
    new ProviderError("incomplete", "stopped with pause_turn", { retryable: false }),
  ),
  ...rows("incomplete for response.failed", "incomplete", () =>
    new ProviderError("incomplete", "response.failed", { retryable: false }),
  ),
  ...rows("refused", "refused", () => new ProviderError("refused", "the model refused")),
];

/** What `runOnce` saw: the reply or the error the call ended with, and each retry. */
interface Outcome {
  response?: ModelResponse;
  error?: unknown;
  retries: RetryInfo[];
}

/** Run `model` with a one-retry budget on `clock`, recording each retry. */
async function runOnce(model: ChatModel, clock: Scheduler): Promise<Outcome> {
  const retries: RetryInfo[] = [];
  const options = { budget: 1, scheduler: clock, random: () => 0, onRetry: (retry: RetryInfo) => void retries.push(retry) };
  try {
    return { response: await callWithRetries(model, REQUEST, options), retries };
  } catch (error) {
    return { error, retries };
  }
}

describe("callWithRetries retry table", () => {
  test.each(RETRIED)("retried: %s", async (_name, after, make, kind) => {
    const clock = virtualScheduler(0);
    const { response, retries } = await runOnce(make(clock), clock);
    expect(response?.message.text()).toBe("ok");
    expect(retries).toHaveLength(1);
    expect(retries[0]!.attempt).toBe(1);
    expect(retries[0]!.error.kind).toBe(kind);
    expect(retries[0]!.partial === null).toBe(!after);
  });

  test.each(STOPPED)("stopped: %s", async (_name, _after, make, kind) => {
    const clock = virtualScheduler(0);
    const { error, retries } = await runOnce(make(clock), clock);
    expect(error).toBeInstanceOf(ProviderError);
    expect(error).toMatchObject({ kind });
    expect(retries).toHaveLength(0);
  });

  test("a defect thrown mid-stream propagates at once, with no retry", async () => {
    const defect = new TypeError("adapter bug");
    const model = new ScriptedModel([[PARTIAL, defect], OK]);
    const { error, retries } = await runOnce(model, virtualScheduler(0));
    expect(error).toBe(defect);
    expect(retries).toHaveLength(0);
    expect(model.requests).toHaveLength(1);
  });

  test("an auth source's error propagates unchanged and unretried", async () => {
    const failed = new Error("refresh failed");
    const auth = new BearerAuth(() => {
      throw failed;
    });
    const { send, sent } = fake();
    const model = new RouteModel({ route: route({ auth }), protocol: sse(), send });
    const { error, retries } = await runOnce(model, virtualScheduler(0));
    expect(error).toBe(failed);
    expect(retries).toHaveLength(0);
    expect(sent).toHaveLength(0);
  });

  test("retry-after: inf is retried with backoff", async () => {
    const clock = virtualScheduler(0);
    const failure = statusError(429, { "retry-after": "inf" });
    expect(failure.retryAfter).toBe(Infinity);
    const { response, retries } = await runOnce(failingOnce(failure, false), clock);
    expect(response).toEqual(OK);
    expect(retries[0]!.retryIn).toBe(0.5);
    expect(clock.now()).toBe(500);
  });
});

describe("callWithRetries order and hooks", () => {
  test("onAttempt, events, onRetry, wait, then the next attempt", async () => {
    const clock = virtualScheduler(0);
    const call = new ToolCallDelta({ index: 1, id: "c1", name: "f", arguments: '{"x"' });
    const model = new ScriptedModel([[new TextDelta(0, "a"), call, statusError(500)], OK]);
    const log: string[] = [];
    let retry: RetryInfo | undefined;
    const response = await callWithRetries(model, REQUEST, {
      scheduler: clock,
      random: () => 0,
      onAttempt: (attempt) => void log.push(`attempt ${attempt} at ${clock.now()}`),
      onEvent: (event, attempt) => void log.push(`${event.type} ${attempt}`),
      onRetry: (info) => {
        retry = info;
        log.push(`retry ${info.attempt} at ${clock.now()}`);
      },
    });
    expect(response).toEqual(OK);
    expect(log).toEqual([
      "attempt 1 at 0",
      "TextDelta 1",
      "ToolCallDelta 1",
      "retry 1 at 0",
      "attempt 2 at 500",
      "TextDelta 2",
      "PartDone 2",
      "Finish 2",
    ]);
    expect(retry!.error.kind).toBe("server");
    expect(retry!.retryIn).toBe(0.5);
    expect(retry!.partial!.parts.map((part) => part.type)).toEqual(["text", "tool_call"]);
    expect(retry!.partial!.text()).toBe("a");
  });

  test("onRetry gets partial: null before the first event", async () => {
    const { retries } = await runOnce(failingOnce(statusError(500), false), virtualScheduler(0));
    expect(retries[0]!.partial).toBeNull();
  });

  test("an async onRetry is awaited before the wait", async () => {
    const clock = virtualScheduler(0);
    const log: string[] = [];
    await callWithRetries(failingOnce(statusError(500), false), REQUEST, {
      scheduler: clock,
      random: () => 0,
      onAttempt: (attempt) => void log.push(`attempt ${attempt}`),
      onRetry: async () => {
        await Promise.resolve();
        log.push(`retry recorded at ${clock.now()}`);
      },
    });
    expect(log).toEqual(["attempt 1", "retry recorded at 0", "attempt 2"]);
  });

  test("a throwing onRetry ends the call with that error", async () => {
    const model = failingOnce(statusError(500), false);
    const failed = new Error("record failed");
    const error = await rejection(
      callWithRetries(model, REQUEST, {
        scheduler: virtualScheduler(0),
        onRetry: () => {
          throw failed;
        },
      }),
    );
    expect(error).toBe(failed);
    expect(model.requests).toHaveLength(1);
    expect(model.closed).toBe(model.requests.length);
  });

  test("a throw from onEvent ends the call, closes the stream and is never retried", async () => {
    const { send, sent } = fake({ body: ["data: a\n\n", "data: b\n\n", HANG] }, sseReply("ok"));
    const model = new RouteModel({ route: route(), protocol: sse(), send });
    // Retryable, to show a hook's error is never retried.
    const failed = statusError(500);
    let retried = false;
    const error = await rejection(
      callWithRetries(model, REQUEST, {
        scheduler: virtualScheduler(0),
        onEvent: () => {
          throw failed;
        },
        onRetry: () => {
          retried = true;
        },
      }),
    );
    expect(error).toBe(failed);
    expect(retried).toBe(false);
    expect(sent).toHaveLength(1);
    expect(sent[0]!.signal.aborted).toBe(true);
  });

  test("an async onEvent's rejection ends the call and is never retried", async () => {
    const model = new ScriptedModel([[PARTIAL, statusError(500)], OK]);
    const failed = new Error("publish failed");
    let retried = false;
    const error = await rejection(
      callWithRetries(model, REQUEST, {
        scheduler: virtualScheduler(0),
        onEvent: async () => {
          await Promise.resolve();
          throw failed;
        },
        onRetry: () => {
          retried = true;
        },
      }),
    );
    expect(error).toBe(failed);
    expect(retried).toBe(false);
    expect(model.requests).toHaveLength(1);
  });

  test("a throw from onAttempt ends the call and is never retried", async () => {
    const model = failingOnce(statusError(500), false);
    const failed = new Error("publish failed");
    const error = await rejection(
      callWithRetries(model, REQUEST, {
        scheduler: virtualScheduler(0),
        onAttempt: (attempt) => {
          if (attempt === 2) throw failed;
        },
      }),
    );
    expect(error).toBe(failed);
    expect(model.requests).toHaveLength(1);
    expect(model.closed).toBe(model.requests.length);
  });

  test("the assembler's provider-less incomplete gets the model's provider", async () => {
    const model = new ScriptedModel([[PARTIAL], [PARTIAL]], { provider: "example" });
    const { error, retries } = await runOnce(model, virtualScheduler(0));
    expect(retries[0]!.error).toMatchObject({ kind: "incomplete", provider: "example" });
    expect(error).toBeInstanceOf(ProviderError);
    expect(error).toMatchObject({ kind: "incomplete", provider: "example" });
  });
});

describe("callWithRetries budget", () => {
  afterEach(() => install(BUILTIN));

  test("budget: 0 makes one attempt", async () => {
    const model = failingOnce(statusError(500), false);
    const error = await rejection(callWithRetries(model, REQUEST, { budget: 0, scheduler: virtualScheduler(0) }));
    expect((error as ProviderError).kind).toBe("server");
    expect(model.requests).toHaveLength(1);
  });

  test("the default budget reads MODEL_RETRIES and exhaustion throws the last error", async () => {
    install({ ...BUILTIN, modelRetries: 2 });
    const last = statusError(503, {}, "third");
    const model = new ScriptedModel([[statusError(500, {}, "first")], [statusError(502)], [last], OK]);
    const error = await rejection(callWithRetries(model, REQUEST, { scheduler: virtualScheduler(0) }));
    expect(error).toBe(last);
    expect(model.requests).toHaveLength(3);
    expect(model.closed).toBe(model.requests.length);
  });

  test("success after retries returns the successful attempt's reply", async () => {
    const model = new ScriptedModel([[statusError(500)], [PARTIAL, statusError(502)], OK]);
    expect(await callWithRetries(model, REQUEST, { scheduler: virtualScheduler(0) })).toEqual(OK);
    expect(model.requests).toHaveLength(3);
  });
});

describe("callWithRetries delays", () => {
  const delays: [string, ProviderError, number, number][] = [
    ["retry-after-ms: 1500 waits 1.5 s", statusError(429, { "retry-after-ms": "1500" }), 0, 1.5],
    ["retry-after: 2 waits 2 s", statusError(429, { "retry-after": "2" }), 3, 2],
    ["retry-after: 0 falls back to backoff", statusError(429, { "retry-after": "0" }), 1, 1],
    ["a past HTTP date falls back to backoff", statusError(429, { "retry-after": "Wed, 21 Oct 2015 07:28:00 GMT" }), 2, 2],
    [
      "retry-after: 121 with x-should-retry: true falls back to backoff",
      statusError(429, { "retry-after": "121", "x-should-retry": "true" }),
      0,
      0.5,
    ],
    ["a huge retry count stays finite", statusError(500), 5000, 8],
  ];
  test.each(delays)("%s", (_name, error, retries, expected) => {
    expect(retryDelay(error, retries, () => 0)).toBe(expected);
  });

  test("backoff doubles from 0.5 s to a cap of 8 s, with jitter down to 0.75x", () => {
    const error = statusError(500);
    const retries = [0, 1, 2, 3, 4, 5];
    expect(retries.map((n) => retryDelay(error, n, () => 0))).toEqual([0.5, 1, 2, 4, 8, 8]);
    const low = retries.map((n) => retryDelay(error, n, () => 1 - 2 ** -53));
    expect(low).toEqual([0.375, 0.75, 1.5, 3, 6, 6]);
  });

  test("the virtual clock advances exactly by the delays", async () => {
    const clock = virtualScheduler(0);
    const model = new ScriptedModel([[statusError(500)], [statusError(500)], [statusError(429, { "retry-after": "2" })], OK]);
    const starts: number[] = [];
    await callWithRetries(model, REQUEST, {
      scheduler: clock,
      random: () => 0,
      onAttempt: () => void starts.push(clock.now()),
    });
    expect(starts).toEqual([0, 500, 1500, 3500]);
    expect(clock.now()).toBe(3500);
  });
});

describe("callWithRetries stops", () => {
  test("a stop during the wait rejects with the signal's reason and makes no further attempt", async () => {
    const clock = virtualScheduler(0);
    const controller = new AbortController();
    const reason = new Error("stop");
    const stopping: Scheduler = {
      now: () => clock.now(),
      sleep(ms, signal) {
        const waiting = clock.sleep(ms, signal);
        controller.abort(reason);
        return waiting;
      },
    };
    const model = failingOnce(statusError(500), false);
    const error = await rejection(
      callWithRetries(model, REQUEST, { signal: controller.signal, scheduler: stopping, random: () => 0 }),
    );
    expect(error).toBe(reason);
    expect(model.requests).toHaveLength(1);
    expect(clock.now()).toBe(0);
  });

  test("an already aborted signal makes no attempt and rejects with its reason", async () => {
    const controller = new AbortController();
    const reason = new Error("stop");
    controller.abort(reason);
    const model = new ScriptedModel([OK]);
    const attempts: number[] = [];
    const error = await rejection(
      callWithRetries(model, REQUEST, { signal: controller.signal, onAttempt: (attempt) => void attempts.push(attempt) }),
    );
    expect(error).toBe(reason);
    expect(attempts).toEqual([]);
    expect(model.requests).toHaveLength(0);
  });

  test("a stop after the first event of a replayed reply rejects with the reason and makes no retry", async () => {
    const replay = new CassetteReplay(replayedReply());
    const model = new RouteModel({ route: route({ query: { "api-version": "2026-01-01" } }), protocol: sse(), send: replay.send });
    const controller = new AbortController();
    const reason = new Error("stop");
    const retries: RetryInfo[] = [];
    const error = await rejection(
      callWithRetries(model, REQUEST, {
        signal: controller.signal,
        scheduler: virtualScheduler(0),
        onEvent: () => controller.abort(reason),
        onRetry: (retry) => void retries.push(retry),
      }),
    );
    expect(error).toBe(reason);
    expect(retries).toEqual([]);
    expect(replay.requests).toHaveLength(1);
  });

  test("a stop landing as a retryable error is thrown rejects with the reason and never calls onRetry", async () => {
    const controller = new AbortController();
    const reason = new Error("stop");
    const model: ChatModel = {
      provider: "openai",
      model: "m",
      stream: () =>
        new EventStream(
          (async function* (): AsyncGenerator<Event> {
            controller.abort(reason);
            throw statusError(500);
          })(),
        ),
    };
    const retries: RetryInfo[] = [];
    const error = await rejection(
      callWithRetries(model, REQUEST, {
        signal: controller.signal,
        scheduler: virtualScheduler(0),
        onRetry: (retry) => void retries.push(retry),
      }),
    );
    expect(error).toBe(reason);
    expect(retries).toEqual([]);
  });

  test("the signal reaches model.stream", async () => {
    const signals: (AbortSignal | undefined)[] = [];
    class Recording extends ScriptedModel {
      override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
        signals.push(options.signal);
        return super.stream(request, options);
      }
    }
    const controller = new AbortController();
    await callWithRetries(new Recording([OK]), REQUEST, { signal: controller.signal });
    expect(signals).toEqual([controller.signal]);
  });
});
