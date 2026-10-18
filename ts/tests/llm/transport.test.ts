/** Tests for `openExchange` in `src/llm/transport.ts`: its timers, stops and error mapping. */

import { describe, expect, test } from "bun:test";

import { RedirectError, type Scheduler, type Send, virtualScheduler } from "../../src/platform.ts";
import { REDACTED } from "../../src/llm/auth.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { type Exchange, type ExchangeOptions, openExchange } from "../../src/llm/transport.ts";

const ENDPOINT = "https://api.example.test/v1/chat";
const S = 1000;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** One body event at virtual time `at` (ms): a chunk, or an error that ends the body. */
type Step = { at: number; chunk?: string | Uint8Array; error?: unknown };

/** What the fake `send` replies, after `after` ms: an error, or a response. */
interface Reply {
  after?: number;
  error?: unknown;
  status?: number;
  headers?: Record<string, string>;
  /** The body's events; `null` for no body. */
  steps?: Step[] | null;
  /** Whether the body closes after its last step (default true). */
  end?: boolean;
  /** A ready-made body, used instead of `steps`. */
  body?: unknown;
  /** Whether the body errors with the request signal's reason once it aborts, as the runtime's does. */
  abortBody?: boolean;
  /** Whether `send` waits out `after` ignoring the request signal. */
  ignoreSignal?: boolean;
}

/** A scheduler wrapper counting the sleeps made through it and those still pending. */
function counting(inner: Scheduler): { scheduler: Scheduler; stats: { armed: number; pending: number } } {
  const stats = { armed: 0, pending: 0 };
  return {
    stats,
    scheduler: {
      now: () => inner.now(),
      async sleep(ms, signal) {
        stats.armed++;
        stats.pending++;
        try {
          await inner.sleep(ms, signal);
        } finally {
          stats.pending--;
        }
      },
    },
  };
}

/**
 * Return a body stream playing `steps` on `clock`, stopped by its `cancel()` (counted in
 * `cancels`) or, when `signal` is given, errored with its reason once it aborts.
 */
function playBody(
  clock: Scheduler,
  steps: Step[],
  end: boolean,
  cancels: { count: number },
  signal?: AbortSignal,
): ReadableStream<Uint8Array> {
  const stop = new AbortController();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      signal?.addEventListener("abort", () => {
        stop.abort();
        controller.error(signal.reason);
      });
      void (async () => {
        try {
          for (const step of steps) {
            await clock.sleep(step.at - clock.now(), stop.signal);
            if (step.error !== undefined) return controller.error(step.error);
            const chunk = step.chunk ?? "";
            controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
          }
          if (end) controller.close();
        } catch {
          // Cancelled.
        }
      })();
    },
    cancel() {
      cancels.count++;
      stop.abort();
    },
  });
}

/** A harness: the virtual clock, the counted scheduler for the transport, and a fake `send`. */
function harness(reply: Reply) {
  const clock = virtualScheduler(0);
  const { scheduler, stats } = counting(clock);
  const calls: RequestInit[] = [];
  const cancels = { count: 0 };
  const send: Send = async (_url, init) => {
    calls.push(init);
    await clock.sleep(reply.after ?? 0, reply.ignoreSignal ? undefined : (init.signal ?? undefined));
    if (reply.error !== undefined) throw reply.error;
    const status = reply.status ?? 200;
    const headers = new Headers(reply.headers);
    if (reply.body !== undefined) return { status, headers, body: reply.body } as unknown as Response;
    const signal = reply.abortBody ? (init.signal ?? undefined) : undefined;
    const body = reply.steps === null ? null : playBody(clock, reply.steps ?? [], reply.end ?? true, cancels, signal);
    return new Response(body, { status, headers });
  };
  const open = (options: Partial<ExchangeOptions> = {}, url = ENDPOINT) =>
    openExchange(
      url,
      { method: "POST", body: "{}" },
      { provider: "openai", credentials: () => [], scheduler, send, ...options },
    );
  return { clock, stats, calls, cancels, send, open };
}

/** Read the whole body as text. */
async function readAll(exchange: Exchange): Promise<string[]> {
  const out: string[] = [];
  for await (const chunk of exchange.body) out.push(decoder.decode(chunk));
  return out;
}

/** Return the error `promise` rejects with, failing when it resolves. */
async function failure(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

/** Let any leftover sleeper on the virtual clock fire (each advance takes one macrotask). */
async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

/** Return a body whose reads never settle, even after a cancel, after yielding `chunks`. */
function hangingBody(...chunks: string[]) {
  const queue = chunks.map((c) => encoder.encode(c));
  const reader = {
    read: () =>
      queue.length > 0 ? Promise.resolve({ done: false, value: queue.shift()! }) : new Promise<never>(() => {}),
    cancel: () => Promise.resolve(),
    releaseLock() {},
  };
  return { getReader: () => reader, cancel: () => Promise.resolve() };
}

/** Check that `error` is a retryable `timeout` `ProviderError` from openai with `message`. */
function expectTimeout(error: unknown, message: string): void {
  expect(error).toBeInstanceOf(ProviderError);
  const e = error as ProviderError;
  expect(e.kind).toBe("timeout");
  expect(e.retryable).toBe(true);
  expect(e.provider).toBe("openai");
  expect(e.message).toBe(message);
}

const NO_RESPONSE = "openai request failed (timeout): no response within 240 s";
const NO_BYTES = "openai request failed (timeout): no bytes for 240 s";

describe("first response", () => {
  test("a reply after 241 s times out at 240 s", async () => {
    const h = harness({ after: 241 * S, steps: [{ at: 241 * S, chunk: "a" }] });
    expectTimeout(await failure(h.open()), NO_RESPONSE);
    expect(h.clock.now()).toBe(240 * S);
  });

  test("headers at once but the first byte after 241 s times out the same", async () => {
    const h = harness({ steps: [{ at: 241 * S, chunk: "a" }] });
    expectTimeout(await failure(h.open()), NO_RESPONSE);
    expect(h.clock.now()).toBe(240 * S);
  });

  test("a first byte at 239 s is fine and is the first chunk", async () => {
    const h = harness({ after: 100 * S, steps: [{ at: 239 * S, chunk: "a" }, { at: 300 * S, chunk: "b" }] });
    const exchange = await h.open();
    expect(h.clock.now()).toBe(239 * S);
    expect(exchange.status).toBe(200);
    expect(await readAll(exchange)).toEqual(["a", "b"]);
  });

  test("empty chunks don't stop the first-response timer", async () => {
    const h = harness({ steps: [{ at: 100 * S, chunk: "" }, { at: 200 * S, chunk: "" }, { at: 241 * S, chunk: "a" }] });
    expectTimeout(await failure(h.open()), NO_RESPONSE);
  });

  test("a body that ends with no bytes resolves and yields nothing", async () => {
    const h = harness({ steps: [{ at: 5 * S, chunk: "" }] });
    const exchange = await h.open();
    expect(await readAll(exchange)).toEqual([]);
  });

  test("a null body counts as ended", async () => {
    const h = harness({ status: 204, steps: null });
    const exchange = await h.open();
    expect(exchange.status).toBe(204);
    expect(await readAll(exchange)).toEqual([]);
  });

  test("a non-2xx reply's status, headers and body come through", async () => {
    const h = harness({ status: 429, headers: { "retry-after": "3" }, steps: [{ at: 0, chunk: "slow down" }] });
    const exchange = await h.open();
    expect(exchange.status).toBe(429);
    expect(exchange.headers.get("retry-after")).toBe("3");
    expect(await readAll(exchange)).toEqual(["slow down"]);
  });

  test("explicit timeouts override the defaults, in seconds in the message", async () => {
    const h = harness({ after: 300 });
    expectTimeout(
      await failure(h.open({ firstResponseMs: 200 })),
      "openai request failed (timeout): no response within 0.2 s",
    );
    expect(h.clock.now()).toBe(200);
  });
});

describe("idle", () => {
  test("a 241 s gap after the first chunk times out", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 241 * S, chunk: "b" }] });
    const exchange = await h.open();
    const seen: string[] = [];
    const error = await failure(
      (async () => {
        for await (const chunk of exchange.body) seen.push(decoder.decode(chunk));
      })(),
    );
    expect(seen).toEqual(["a"]);
    expectTimeout(error, NO_BYTES);
    expect(h.clock.now()).toBe(240 * S);
    expect(h.calls[0]!.signal!.aborted).toBe(true);
  });

  test("an idleMs override sets the idle timer", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 600, chunk: "b" }] });
    const exchange = await h.open({ idleMs: 500 });
    expectTimeout(await failure(readAll(exchange)), "openai request failed (timeout): no bytes for 0.5 s");
    expect(h.clock.now()).toBe(500);
  });

  test("a 239 s gap is fine", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 239 * S, chunk: "b" }, { at: 478 * S, chunk: "c" }] });
    expect(await readAll(await h.open())).toEqual(["a", "b", "c"]);
  });

  test("a 20-minute stream with a chunk every 10 s ends normally", async () => {
    const steps = Array.from({ length: 121 }, (_, i) => ({ at: i * 10 * S, chunk: `${i}` }));
    const h = harness({ steps });
    const chunks = await readAll(await h.open());
    expect(chunks.length).toBe(121);
    expect(h.clock.now()).toBe(1200 * S);
  });

  test("an SSE comment every 200 s keeps a 20-minute stream alive", async () => {
    const steps = Array.from({ length: 7 }, (_, i) => ({ at: i * 200 * S, chunk: ": ping\n\n" }));
    const h = harness({ steps });
    const chunks = await readAll(await h.open());
    expect(chunks.length).toBe(7);
    expect(h.clock.now()).toBe(1200 * S);
  });

  test("a one-byte chunk resets the idle timer", async () => {
    const h = harness({
      steps: [
        { at: 0, chunk: "a" },
        { at: 200 * S, chunk: new Uint8Array([0x20]) },
        { at: 400 * S, chunk: "b" },
      ],
    });
    expect(await readAll(await h.open())).toEqual(["a", " ", "b"]);
  });

  test("empty chunks are skipped and don't reset the idle timer", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 200 * S, chunk: "" }, { at: 441 * S, chunk: "b" }] });
    const exchange = await h.open();
    const iterator = exchange.body;
    expect(decoder.decode((await iterator.next()).value!)).toBe("a");
    expectTimeout(await failure(iterator.next()), NO_BYTES);
    expect(h.clock.now()).toBe(240 * S);
  });

  test("time the consumer spends between reads doesn't count", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 10 * S, chunk: "b" }] });
    const exchange = await h.open();
    const iterator = exchange.body;
    expect(decoder.decode((await iterator.next()).value!)).toBe("a");
    await h.clock.sleep(1_000_000 * S);
    expect(decoder.decode((await iterator.next()).value!)).toBe("b");
    expect((await iterator.next()).done).toBe(true);
  });
});

describe("no timer outlives its wait", () => {
  test("after a normal end", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 30 * S, chunk: "b" }] });
    await readAll(await h.open());
    await settle();
    expect(h.stats.armed).toBeGreaterThan(0);
    expect(h.stats.pending).toBe(0);
    expect(h.clock.now()).toBe(30 * S);
  });

  test("after a timeout", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 500 * S, chunk: "b" }] });
    const exchange = await h.open();
    await failure(readAll(exchange));
    await settle();
    expect(h.stats.pending).toBe(0);
    expect(h.clock.now()).toBe(240 * S);
  });

  test("after an early return()", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 10 * S, chunk: "b" }, { at: 500 * S, chunk: "c" }] });
    const exchange = await h.open();
    for await (const chunk of exchange.body) {
      if (decoder.decode(chunk) === "b") break;
    }
    await settle();
    expect(h.stats.pending).toBe(0);
    expect(h.clock.now()).toBe(10 * S);
    expect(h.calls[0]!.signal!.aborted).toBe(true);
  });

  test("after a caller abort", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 500 * S, chunk: "b" }] });
    const caller = new AbortController();
    const exchange = await h.open({ signal: caller.signal });
    void h.clock.sleep(50 * S).then(() => caller.abort(new Error("stop")));
    await failure(readAll(exchange));
    await settle();
    expect(h.stats.pending).toBe(0);
    expect(h.clock.now()).toBe(50 * S);
  });
});

describe("a read that never settles", () => {
  test("times out at 240 s on the first read", async () => {
    const h = harness({ body: hangingBody() });
    expectTimeout(await failure(h.open()), NO_RESPONSE);
    expect(h.clock.now()).toBe(240 * S);
  });

  test("times out at 240 s on a later read", async () => {
    const h = harness({ body: hangingBody("a") });
    const exchange = await h.open();
    expectTimeout(await failure(readAll(exchange)), NO_BYTES);
    expect(h.clock.now()).toBe(240 * S);
  });

  test("throws the caller's reason at once on a caller abort", async () => {
    const h = harness({ body: hangingBody("a") });
    const caller = new AbortController();
    const reason = new Error("user stop");
    const exchange = await h.open({ signal: caller.signal });
    void h.clock.sleep(5 * S).then(() => caller.abort(reason));
    expect(await failure(readAll(exchange))).toBe(reason);
    expect(h.clock.now()).toBe(5 * S);
  });

  test("is ended by abort() with an AbortError, never a timeout", async () => {
    const h = harness({ body: hangingBody("a") });
    const exchange = await h.open();
    const iterator = exchange.body;
    await iterator.next();
    const pending = iterator.next();
    void h.clock.sleep(5 * S).then(() => exchange.abort());
    const error = await failure(pending);
    expect(error).not.toBeInstanceOf(ProviderError);
    expect((error as Error).name).toBe("AbortError");
    expect(h.clock.now()).toBe(5 * S);
    await settle();
    expect(h.stats.pending).toBe(0);
  });
});

describe("the request's signal", () => {
  test("is aborted on a first-response timeout", async () => {
    const h = harness({ steps: [{ at: 300 * S, chunk: "a" }] });
    await failure(h.open());
    expect(h.calls[0]!.signal!.aborted).toBe(true);
  });

  test("is aborted on an early return()", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 10 * S, chunk: "b" }] });
    const exchange = await h.open();
    await exchange.body.next();
    expect(h.calls[0]!.signal!.aborted).toBe(false);
    await exchange.body.return(undefined);
    expect(h.calls[0]!.signal!.aborted).toBe(true);
  });

  test("is aborted by abort()", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 10 * S, chunk: "b" }] });
    const exchange = await h.open();
    exchange.abort();
    expect(h.calls[0]!.signal!.aborted).toBe(true);
    await settle();
    expect(h.clock.now()).toBe(0);
  });

  test("returning before the first read leaves it unaborted; abort() aborts it", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 10 * S, chunk: "b" }] });
    const exchange = await h.open();
    expect((await exchange.body.return(undefined)).done).toBe(true);
    expect(h.calls[0]!.signal!.aborted).toBe(false);
    exchange.abort();
    expect(h.calls[0]!.signal!.aborted).toBe(true);
  });

  test("abort() during a pending read rejects it instead of ending the body", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 100 * S, chunk: "b" }], abortBody: true });
    const exchange = await h.open();
    const iterator = exchange.body;
    await iterator.next();
    const pending = iterator.next();
    void h.clock.sleep(5 * S).then(() => exchange.abort());
    const error = await failure(pending);
    expect(error).not.toBeInstanceOf(ProviderError);
    expect((error as Error).name).toBe("AbortError");
  });

  test("a reply from a sender ignoring the signal after a timeout is released", async () => {
    const h = harness({ after: 300 * S, ignoreSignal: true, steps: [{ at: 300 * S, chunk: "a" }] });
    expectTimeout(await failure(h.open()), NO_RESPONSE);
    expect(h.clock.now()).toBe(240 * S);
    await settle();
    expect(h.clock.now()).toBe(300 * S);
    expect(h.cancels.count).toBe(1);
  });
});

describe("a caller abort", () => {
  test("before the request throws the caller's reason and sends nothing", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }] });
    const caller = new AbortController();
    const reason = new Error("stopped early");
    caller.abort(reason);
    expect(await failure(h.open({ signal: caller.signal }))).toBe(reason);
    expect(h.calls.length).toBe(0);
  });

  test("while waiting for headers throws the caller's reason", async () => {
    const h = harness({ after: 100 * S, steps: [{ at: 100 * S, chunk: "a" }] });
    const caller = new AbortController();
    const reason = new Error("stopped");
    void h.clock.sleep(50 * S).then(() => caller.abort(reason));
    expect(await failure(h.open({ signal: caller.signal }))).toBe(reason);
    expect(h.clock.now()).toBe(50 * S);
    expect(h.calls[0]!.signal!.aborted).toBe(true);
  });

  test("while waiting for the first byte throws the caller's reason", async () => {
    const h = harness({ steps: [{ at: 100 * S, chunk: "a" }] });
    const caller = new AbortController();
    const reason = "a string reason";
    void h.clock.sleep(50 * S).then(() => caller.abort(reason));
    expect(await failure(h.open({ signal: caller.signal }))).toBe(reason);
    expect(h.clock.now()).toBe(50 * S);
  });

  test("during a later read throws the caller's reason, never a ProviderError", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 100 * S, chunk: "b" }] });
    const caller = new AbortController();
    const exchange = await h.open({ signal: caller.signal });
    void h.clock.sleep(50 * S).then(() => caller.abort());
    const error = await failure(readAll(exchange));
    expect(error).toBe(caller.signal.reason);
    expect(error).not.toBeInstanceOf(ProviderError);
  });
});

/** Return `error` with `fields` set on it, the way runtimes attach `code`. */
function withCode<E extends Error>(error: E, code: string): E {
  return Object.assign(error, { code });
}

describe("error mapping", () => {
  test("Node's connection refused is a retryable connection error naming the code", async () => {
    const error = new TypeError("fetch failed", { cause: withCode(new Error("connect ECONNREFUSED"), "ECONNREFUSED") });
    const h = harness({ error });
    const mapped = (await failure(h.open())) as ProviderError;
    expect(mapped).toBeInstanceOf(ProviderError);
    expect(mapped.kind).toBe("connection");
    expect(mapped.retryable).toBe(true);
    expect(mapped.provider).toBe("openai");
    expect(mapped.message).toBe("openai request failed (connection): TypeError: fetch failed (ECONNREFUSED)");
    expect(mapped.causeName).toBe("TypeError");
    expect(mapped.cause).toBeUndefined();
    expect(h.calls.length).toBe(1);
  });

  test("Bun's connection refused (code on the TypeError) is a connection error", async () => {
    const error = withCode(new TypeError("Unable to connect."), "ConnectionRefused");
    const mapped = (await failure(harness({ error }).open())) as ProviderError;
    expect(mapped.kind).toBe("connection");
    expect(mapped.retryable).toBe(true);
    expect(mapped.message).toBe(
      "openai request failed (connection): TypeError: Unable to connect. (ConnectionRefused)",
    );
  });

  for (const code of ["UND_ERR_BODY_TIMEOUT", "UND_ERR_HEADERS_TIMEOUT", "UND_ERR_CONNECT_TIMEOUT", "ETIMEDOUT"]) {
    test(`a ${code} cause is a timeout`, async () => {
      const error = new TypeError("fetch failed", { cause: withCode(new Error("timed out"), code) });
      const mapped = (await failure(harness({ error }).open())) as ProviderError;
      expect(mapped.kind).toBe("timeout");
      expect(mapped.retryable).toBe(true);
      expect(mapped.message).toBe(`openai request failed (timeout): TypeError: fetch failed (${code})`);
    });
  }

  test("a cause cycle ends the walk and still maps", async () => {
    const error = new TypeError("fetch failed");
    const inner = withCode(new Error("reset"), "ECONNRESET");
    error.cause = inner;
    inner.cause = error;
    const mapped = (await failure(harness({ error }).open())) as ProviderError;
    expect(mapped.kind).toBe("connection");
    expect(mapped.message).toBe("openai request failed (connection): TypeError: fetch failed (ECONNRESET)");
  });

  test("a codeless error that is its own cause propagates unchanged", async () => {
    const error = new Error("loop");
    error.cause = error;
    expect(await failure(harness({ error }).open())).toBe(error);
  });

  test("a TimeoutError nested in the cause is a timeout", async () => {
    const cause = new DOMException("The operation timed out.", "TimeoutError");
    const error = new TypeError("fetch failed", { cause });
    const mapped = (await failure(harness({ error }).open())) as ProviderError;
    expect(mapped.kind).toBe("timeout");
    expect(mapped.message).toBe("openai request failed (timeout): TypeError: fetch failed");
    expect(mapped.causeName).toBe("TypeError");
  });

  test("a thrown non-Error propagates unchanged", async () => {
    const error = { code: "ECONNRESET", message: "not an Error" };
    expect(await failure(harness({ error }).open())).toBe(error);
  });

  test("a TimeoutError is a timeout", async () => {
    const error = new DOMException("The operation timed out.", "TimeoutError");
    const mapped = (await failure(harness({ error }).open())) as ProviderError;
    expect(mapped.kind).toBe("timeout");
    expect(mapped.message).toBe("openai request failed (timeout): TimeoutError: The operation timed out.");
    expect(mapped.causeName).toBe("TimeoutError");
  });

  test("Node's terminated mid-body is a connection error", async () => {
    const socket = Object.assign(new Error("other side closed"), { name: "SocketError", code: "UND_ERR_SOCKET" });
    const error = new TypeError("terminated", { cause: socket });
    const h = harness({ steps: [{ at: 0, chunk: "a" }, { at: 5 * S, error }] });
    const exchange = await h.open();
    const mapped = (await failure(readAll(exchange))) as ProviderError;
    expect(mapped).toBeInstanceOf(ProviderError);
    expect(mapped.kind).toBe("connection");
    expect(mapped.message).toBe("openai request failed (connection): TypeError: terminated (UND_ERR_SOCKET)");
  });

  test("Node's fetch failed with a codeless cause is a connection error", async () => {
    const error = new TypeError("fetch failed", { cause: new Error("something broke") });
    const mapped = (await failure(harness({ error }).open())) as ProviderError;
    expect(mapped.kind).toBe("connection");
    expect(mapped.message).toBe("openai request failed (connection): TypeError: fetch failed");
  });

  test("a credential in the runtime message is scrubbed", async () => {
    const error = withCode(new TypeError("cannot reach https://h.test/?key=sk-secret-123"), "ECONNRESET");
    const mapped = (await failure(harness({ error }).open({ credentials: () => ["sk-secret-123"] }))) as ProviderError;
    expect(mapped.message).not.toContain("sk-secret-123");
    expect(mapped.message).toContain(REDACTED);
  });

  test("a redirect is a redirect error, not retryable", async () => {
    const location = "https://evil.test/steal?k=1";
    const h = harness({ status: 307, headers: { location }, steps: [{ at: 0, chunk: "x" }] });
    const mapped = (await failure(h.open())) as ProviderError;
    expect(mapped.kind).toBe("redirect");
    expect(mapped.retryable).toBe(false);
    expect(mapped.status).toBe(307);
    expect(mapped.provider).toBe("openai");
    expect(mapped.message).toBe("openai returned HTTP 307 (redirect): redirect to https://evil.test");
  });

  test("a RedirectError thrown by send is mapped the same", async () => {
    const error = new RedirectError(302, "redirect to a relative Location");
    const mapped = (await failure(harness({ error }).open())) as ProviderError;
    expect(mapped.kind).toBe("redirect");
    expect(mapped.message).toBe("openai returned HTTP 302 (redirect): redirect to a relative Location");
  });

  const unchanged: [string, () => unknown][] = [
    ["a TypeError with no code (an invalid header value)", () => new TypeError("invalid header value")],
    [
      "Node's invalid URL shape",
      () => new TypeError("Failed to parse URL", { cause: withCode(new TypeError("Invalid URL"), "ERR_INVALID_URL") }),
    ],
    ["Bun's invalid argument shape", () => withCode(new TypeError("bad argument"), "ERR_INVALID_ARG_VALUE")],
    ["a plain Error from the fake", () => new Error("bug in the fake")],
  ];
  for (const [name, make] of unchanged) {
    test(`${name} propagates unchanged after one attempt`, async () => {
      const error = make();
      const h = harness({ error });
      expect(await failure(h.open())).toBe(error);
      expect(h.calls.length).toBe(1);
    });
  }

  test("the user-info refusal propagates unchanged and sends nothing", async () => {
    const h = harness({ steps: [{ at: 0, chunk: "a" }] });
    const error = await failure(h.open({}, "https://user:pw@api.example.test/v1"));
    expect(error).toBeInstanceOf(TypeError);
    expect(error).not.toBeInstanceOf(ProviderError);
    expect((error as Error).message).toBe("request URL must not contain user info");
    expect(h.calls.length).toBe(0);
  });
});
