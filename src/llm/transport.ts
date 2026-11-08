/**
 * The HTTP layer under yian's provider calls, with runtime failures mapped to
 * `ProviderError`s: `openExchange` for a model call's stream, timed only while a read is
 * pending, and `shortRequest` for a small reply read whole under one total timer.
 *
 * A model call's timers run only while the transport waits, so time the consumer spends
 * between reads never counts as silence. Each wait races its timer and the request's
 * signal (the caller's stop, our timeout, `abort()`) against the work, so any of them
 * ends even a read the runtime never settles.
 */

import { current } from "../defaults.ts";
import { RedirectError, realScheduler, safeFetch, type Scheduler, type Send } from "../platform.ts";
import { dropCredentialTail, scrub } from "./auth.ts";
import { ProviderError } from "./errors.ts";
import { JSONDecodeError, type JsonValue, parseJson } from "../typesys/json.ts";

/** Options for `openExchange`. */
export interface ExchangeOptions {
  /** The provider name, for error messages. */
  provider: string;
  /** The secrets to scrub from error text, read when an error is built. */
  credentials: () => Iterable<string>;
  /**
   * Stops the exchange; a stop rejects with `signal.reason` from wherever it is pending, even
   * when the sender's body ignores the signal.
   */
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
   * returning early aborts the request. Returning before the first read doesn't run the
   * body; call `abort()` to release an exchange whose body you won't read.
   */
  readonly body: AsyncGenerator<Uint8Array>;
  /**
   * Abort the request and release the body; a pending read rejects with an `AbortError`,
   * even when the sender's body ignores the signal.
   */
  abort(): void;
}

/** Options for `shortRequest`. */
export interface ShortRequestOptions {
  /** What is being asked, for error messages, such as a provider name. */
  name: string;
  /** The secrets to scrub from error text, read when an error is built; defaults to none. */
  credentials?: () => Iterable<string>;
  /** Longest wait for the whole reply; defaults to `current().requestTimeoutSeconds`. */
  timeoutMs?: number;
  /** The largest body accepted, in bytes; defaults to no cap. */
  maxBytes?: number;
  /**
   * Whether a non-2xx body is read only to its first `MAX_ERROR_BODY_BYTES`, as a model call's
   * is, with its text cut by `errorBodyText`, instead of whole under `maxBytes`; defaults to false.
   */
  cutErrorBody?: boolean;
  /** Stops the request; a stop rejects with `signal.reason`. */
  signal?: AbortSignal;
  /** The clock the timer runs on; defaults to `realScheduler`. */
  scheduler?: Scheduler;
  /** The raw sender passed to `safeFetch`. */
  send?: Send;
}

/** A short request's whole reply. */
export interface ShortReply {
  readonly status: number;
  readonly headers: Headers;
  /** The body decoded as UTF-8, invalid bytes replaced. */
  readonly text: string;
  /**
   * Decode `text` as plain JSON, so yian's `$float` and `$record` tags stay records.
   *
   * @throws ProviderError Of kind `protocol` when the text isn't JSON.
   */
  json(): JsonValue;
}

/** The most bytes of an error reply's body read: its error text never needs more. */
export const MAX_ERROR_BODY_BYTES = 64 * 1024;

/**
 * Return the first `MAX_ERROR_BODY_BYTES` of `chunks`, `size` bytes in all, decoded as UTF-8 with
 * replacement. Text cut at the cap loses what `dropCredentialTail` drops: the cut may split a
 * credential, and no scrub matches the part before it.
 */
export function errorBodyText(chunks: readonly Uint8Array[], size: number, credentials: readonly string[]): string {
  const bytes = new Uint8Array(Math.min(size, MAX_ERROR_BODY_BYTES));
  let at = 0;
  for (const chunk of chunks) {
    const part = chunk.subarray(0, bytes.length - at);
    bytes.set(part, at);
    at += part.length;
  }
  const text = new TextDecoder().decode(bytes);
  return size < MAX_ERROR_BODY_BYTES ? text : dropCredentialTail(text, credentials);
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
 * Wait for `work` under a timer of `ms` on `scheduler` and `signal`.
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
    const result = await Promise.race([work, timer]);
    // A read that settles at once (a replayed chunk, the end a cancel gives) can win the race
    // against a stop or `abort()`.
    if (signal?.aborted) throw signal.reason;
    return result;
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
  if (!(error instanceof Error)) return undefined;
  let code: string | undefined;
  let timeout = false;
  const seen = new Set<unknown>();
  for (let link: unknown = error; typeof link === "object" && link !== null && !seen.has(link); ) {
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
  const { name, message } = error;
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

/** Options for `requestControl`. */
interface RequestOptions {
  provider: string;
  credentials: () => Iterable<string>;
  signal: AbortSignal | undefined;
  send: Send | undefined;
}

/** Return the abort wiring and error mapping for one request; `send()` sends it. */
function requestControl(url: string, init: RequestInit, options: RequestOptions) {
  const { provider, credentials, signal } = options;
  const internal = new AbortController();
  const requestSignal = signal ? AbortSignal.any([signal, internal.signal]) : internal.signal;
  let timedOut: ProviderError | undefined;
  let reader: BodyReader | undefined;

  /** Abort the request and release its body. */
  function release(): void {
    // Abort first: a cancel would end a pending read as a normal end of the body.
    internal.abort();
    void reader?.cancel().catch(() => {});
  }
  /** Return a callback that ends the request with the timeout error `detail`. */
  function onTimeout(detail: string): () => ProviderError {
    return () => {
      timedOut = new ProviderError("timeout", `${provider} request failed (timeout): ${detail}`, { provider });
      internal.abort(timedOut);
      return timedOut;
    };
  }
  /** Return the error to throw for a failed wait. */
  function failure(error: unknown): unknown {
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
  }
  /** Send the request; return the reply and its body's reader, if it has a body. */
  async function send(): Promise<[Response, BodyReader | undefined]> {
    const reply = await safeFetch(url, { ...init, signal: requestSignal }, options.send);
    reader = reply.body?.getReader();
    // A sender that ignores the signal can reply after a timeout or a stop already released.
    if (internal.signal.aborted) release();
    return [reply, reader];
  }
  return { signal: requestSignal, release, onTimeout, failure, send };
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
  const request = requestControl(url, init, { provider, credentials, signal, send });
  let response: Response;
  let reader: BodyReader | undefined;
  let first: Uint8Array | undefined;
  try {
    [response, reader, first] = await timed(
      (async () => {
        const [reply, replyReader] = await request.send();
        return [reply, replyReader, replyReader && (await nextChunk(replyReader))] as const;
      })(),
      firstResponseMs,
      scheduler,
      request.signal,
      request.onTimeout(`no response within ${firstResponseMs / 1000} s`),
    );
  } catch (error) {
    const thrown = request.failure(error);
    request.release();
    throw thrown;
  }
  // Nothing more to read: free the request now.
  if (first === undefined) request.release();

  const noBytes = request.onTimeout(`no bytes for ${idleMs / 1000} s`);
  async function* body(): AsyncGenerator<Uint8Array> {
    try {
      if (first === undefined) return;
      yield first;
      for (;;) {
        const chunk = await timed(nextChunk(reader!), idleMs, scheduler, request.signal, noBytes);
        if (chunk === undefined) return;
        yield chunk;
      }
    } catch (error) {
      throw request.failure(error);
    } finally {
      request.release();
    }
  }

  return { status: response.status, headers: response.headers, body: body(), abort: request.release };
}

/**
 * Send one short request through `safeFetch` and read its whole body, all under one total
 * timer, for replies such as a model listing or a sign-in call.
 *
 * Errors map as in `openExchange`, named by `name`. A body larger than `maxBytes` aborts
 * the request and throws a `protocol` `ProviderError`, except a non-2xx body under
 * `cutErrorBody`, which is cut instead. A non-2xx reply is returned: the caller decides what it
 * means.
 */
export async function shortRequest(url: string, init: RequestInit, options: ShortRequestOptions): Promise<ShortReply> {
  const {
    name,
    credentials = () => [],
    maxBytes = Infinity,
    cutErrorBody = false,
    signal,
    scheduler = realScheduler,
    send,
  } = options;
  const timeoutMs = options.timeoutMs ?? current().requestTimeoutSeconds * 1000;
  signal?.throwIfAborted();
  const request = requestControl(url, init, { provider: name, credentials, signal, send });
  let response: Response;
  let chunks: Uint8Array[];
  let size: number;
  let cut: boolean;
  try {
    [response, chunks, size, cut] = await timed(
      (async () => {
        const [reply, reader] = await request.send();
        const cutting = cutErrorBody && (reply.status < 200 || reply.status >= 300);
        const read: Uint8Array[] = [];
        let total = 0;
        while (reader) {
          const chunk = await nextChunk(reader);
          if (chunk === undefined) break;
          total += chunk.byteLength;
          if (!cutting && total > maxBytes) {
            const message = `${name} response is larger than ${maxBytes} bytes`;
            throw new ProviderError("protocol", message, { status: reply.status, provider: name });
          }
          read.push(chunk);
          // The rest of an error body can't change what the error says.
          if (cutting && total >= MAX_ERROR_BODY_BYTES) break;
        }
        return [reply, read, total, cutting] as const;
      })(),
      timeoutMs,
      scheduler,
      request.signal,
      request.onTimeout(`no complete response within ${timeoutMs / 1000} s`),
    );
  } catch (error) {
    throw request.failure(error);
  } finally {
    request.release();
  }
  let text: string;
  if (cut) {
    text = errorBodyText(chunks, size, [...credentials()]);
  } else {
    const bytes = new Uint8Array(size);
    let at = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, at);
      at += chunk.byteLength;
    }
    text = new TextDecoder().decode(bytes);
  }
  return {
    status: response.status,
    headers: response.headers,
    text,
    json() {
      try {
        return parseJson(text, "plain");
      } catch (error) {
        if (!(error instanceof JSONDecodeError)) throw error;
        // No `cause`: the decode error's text holds the body.
        throw new ProviderError("protocol", `${name} returned an invalid JSON body`, { provider: name });
      }
    },
  };
}
