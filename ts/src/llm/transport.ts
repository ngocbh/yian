/**
 * The HTTP layer under every model call: one exchange through `safeFetch`, timed only
 * while a read is pending, with runtime failures mapped to `ProviderError`s.
 *
 * Timers run only while the transport waits, so time the consumer spends between reads
 * never counts as silence. Each wait races its timer and the caller's signal against
 * the work, so a stop or a timeout ends even a read the runtime never settles.
 */

import { current } from "../defaults.ts";
import { RedirectError, realScheduler, safeFetch, type Scheduler, type Send } from "../platform.ts";
import { scrub } from "./auth.ts";
import { ProviderError } from "./errors.ts";

/** Options for `openExchange`. */
export interface ExchangeOptions {
  /** The provider name, for error messages. */
  provider: string;
  /** The secrets to scrub from error text, read when an error is built. */
  credentials: () => Iterable<string>;
  /** Stops the exchange; a stop rejects with `signal.reason` from wherever it is pending. */
  signal?: AbortSignal;
  /** The clock the timers run on; defaults to `realScheduler`. */
  scheduler?: Scheduler;
  /** The raw sender passed to `safeFetch`. */
  send?: Send;
  /** Longest wait from sending to the first body bytes; defaults to `current().modelFirstResponseTimeoutSeconds`. */
  firstResponseMs?: number;
  /** Longest wait between body bytes; defaults to `current().modelIdleTimeoutSeconds`. */
  idleMs?: number;
}

/** An open exchange: the reply's status and headers, and its body. */
export interface Exchange {
  readonly status: number;
  readonly headers: Headers;
  /**
   * The body's non-empty chunks, each read under the idle timer. Finishing, failing or
   * returning early aborts the request.
   */
  readonly body: AsyncGenerator<Uint8Array>;
  /** Abort the request and release the body. */
  abort(): void;
}

/** The part of a body reader the transport uses; Node's and Bun's reader types differ. */
type BodyReader = {
  read(): Promise<{ done: true; value?: undefined } | { done: false; value: Uint8Array }>;
  cancel(): Promise<void>;
};

const TIMEOUT_CODES = new Set([
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_BODY_TIMEOUT",
  "UND_ERR_CONNECT_TIMEOUT",
  "ETIMEDOUT",
]);

/**
 * Wait for `work` under a timer of `ms` on `scheduler` and the caller's `signal`.
 *
 * Rejects with `signal.reason` once the signal aborts, and with `onTimeout()`'s error once
 * the timer fires; otherwise settles as `work` does. The timer never outlives the wait.
 */
async function timed<T>(
  work: Promise<T>,
  ms: number,
  scheduler: Scheduler,
  signal: AbortSignal | undefined,
  onTimeout: () => Error,
): Promise<T> {
  const perWait = new AbortController();
  const stop = () => perWait.abort(signal?.reason);
  signal?.addEventListener("abort", stop);
  if (signal?.aborted) stop();
  try {
    const timer = scheduler.sleep(ms, perWait.signal).then(() => {
      throw onTimeout();
    });
    return await Promise.race([work, timer]);
  } finally {
    signal?.removeEventListener("abort", stop);
    // Ends the timer's sleep: a leftover would keep a real process alive or move a virtual clock.
    perWait.abort();
  }
}

/**
 * Return the `ProviderError` for a runtime transport failure, or `undefined` when `error`
 * isn't one (a configuration error or a defect), so it propagates unchanged.
 *
 * Reads the first string `code` along the `cause` chain: Node wraps the system error in
 * `cause`, Bun sets `code` on its `TypeError`.
 */
function transportError(error: unknown, provider: string, credentials: Iterable<string>): ProviderError | undefined {
  let code: string | undefined;
  let timeout = false;
  const seen = new Set<unknown>();
  for (let link = error; typeof link === "object" && link !== null && !seen.has(link); ) {
    seen.add(link);
    const { name, code: linkCode, cause } = link as { name?: unknown; code?: unknown; cause?: unknown };
    if (name === "TimeoutError") timeout = true;
    if (code === undefined && typeof linkCode === "string") code = linkCode;
    link = cause;
  }
  // A runtime's argument error (Bun's ERR_INVALID_ARG_VALUE, Node's ERR_INVALID_URL) is
  // a configuration error: retrying it only delays the report.
  const runtimeCode = code !== undefined && !code.startsWith("ERR_INVALID");
  const wrapped =
    error instanceof TypeError &&
    (error.message === "fetch failed" || error.message === "terminated") &&
    error.cause !== undefined;
  let kind: "timeout" | "connection";
  if (timeout || (code !== undefined && TIMEOUT_CODES.has(code))) kind = "timeout";
  else if (runtimeCode || wrapped) kind = "connection";
  else return undefined;
  const { name, message } = error as Error;
  const text = `${provider} request failed (${kind}): ${name}: ${message}` + (code === undefined ? "" : ` (${code})`);
  return new ProviderError(kind, scrub(text, credentials), { causeName: name, provider });
}

/** Return the next non-empty chunk, or `undefined` once the body ends. */
async function nextChunk(reader: BodyReader): Promise<Uint8Array | undefined> {
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return undefined;
    if (value.byteLength > 0) return value;
  }
}

/**
 * Send one request through `safeFetch` and return once its headers and first non-empty
 * chunk (or the end of its body) have arrived, all under the first-response timer.
 *
 * A caller abort rejects with `signal.reason`; a timeout with a retryable `timeout`
 * `ProviderError`; a 3xx with a non-retryable `redirect` one; a runtime network failure
 * with a retryable `timeout` or `connection` one. Any other error propagates unchanged.
 * A non-2xx reply resolves like any other: its body is the caller's to read.
 */
export async function openExchange(url: string, init: RequestInit, options: ExchangeOptions): Promise<Exchange> {
  const { provider, credentials, signal, scheduler = realScheduler, send } = options;
  const firstResponseMs = options.firstResponseMs ?? current().modelFirstResponseTimeoutSeconds * 1000;
  const idleMs = options.idleMs ?? current().modelIdleTimeoutSeconds * 1000;
  signal?.throwIfAborted();
  const internal = new AbortController();
  let timedOut: ProviderError | undefined;
  let reader: BodyReader | undefined;

  /** Return a callback that ends the exchange with the timeout error `detail`. */
  const onTimeout = (detail: string) => () => {
    timedOut = new ProviderError("timeout", `${provider} request failed (timeout): ${detail}`, { provider });
    internal.abort(timedOut);
    return timedOut;
  };
  /** Return the error to throw for a failed wait. */
  const failure = (error: unknown): unknown => {
    if (signal?.aborted) return signal.reason;
    if (timedOut) return timedOut;
    if (error instanceof RedirectError) {
      return new ProviderError("redirect", `${provider} returned HTTP ${error.status} (redirect): ${error.detail}`, {
        status: error.status,
        retryable: false,
        provider,
      });
    }
    return transportError(error, provider, credentials()) ?? error;
  };
  const release = () => {
    void reader?.cancel().catch(() => {});
    internal.abort();
  };

  let response: Response;
  let first: Uint8Array | undefined;
  try {
    const requestSignal = signal ? AbortSignal.any([signal, internal.signal]) : internal.signal;
    [response, first] = await timed(
      (async () => {
        const reply = await safeFetch(url, { ...init, signal: requestSignal }, send);
        reader = reply.body?.getReader();
        return [reply, reader && (await nextChunk(reader))] as const;
      })(),
      firstResponseMs,
      scheduler,
      signal,
      onTimeout(`no response within ${firstResponseMs / 1000} s`),
    );
  } catch (error) {
    const thrown = failure(error);
    release();
    throw thrown;
  }
  // Nothing more to read: free the request now.
  if (first === undefined) release();

  const noBytes = onTimeout(`no bytes for ${idleMs / 1000} s`);
  async function* body(): AsyncGenerator<Uint8Array> {
    try {
      if (first === undefined) return;
      yield first;
      for (;;) {
        const chunk = await timed(nextChunk(reader!), idleMs, scheduler, signal, noBytes);
        if (chunk === undefined) return;
        yield chunk;
      }
    } catch (error) {
      throw failure(error);
    } finally {
      release();
    }
  }

  return { status: response.status, headers: response.headers, body: body(), abort: release };
}
