/**
 * Send a model request over a route and read its reply as events: `RouteModel` joins a
 * route (where the body goes and how it authenticates), a wire protocol (how a `Request`
 * becomes a body and the reply's frames become events) and the transport, making one
 * attempt per call. `callWithRetries` is the one retry loop above it.
 *
 * A status error or a `ProviderError` from parsing is scrubbed of the credentials the route
 * sent, in its message and its `stack`; a parse error's message is then capped at 2,000 code
 * points, so protocols leave provider text uncapped. The transport scrubs its own errors, and
 * the AGENT scrubs every error a call raises.
 */

import { current } from "../defaults.ts";
import { realScheduler, type Scheduler, type Send } from "../platform.ts";
import type { JsonObject, JsonValue } from "../typesys/json.ts";
import { dropCredentialTail, scrub } from "./auth.ts";
import { encodeBody } from "./body.ts";
import { capDetail, errorFromResponse, MAX_RETRY_AFTER_SECONDS, nonStreamError, ProviderError } from "./errors.ts";
import { type ChatModel, type Event, EventStream, ReplyAssembler } from "./events.ts";
import { iterNdjson, iterSse, type SSEEvent, type SseStats } from "./framing.ts";
import type { Message, Request, Response } from "./messages.ts";
import type { Route } from "./route.ts";
import { type Exchange, openExchange } from "./transport.ts";

// The openai and anthropic SDKs send this Accept on every request, streamed ones too.
const DEFAULT_HEADERS = { "content-type": "application/json", accept: "application/json" };
const MAX_ERROR_BODY = 64 * 1024;
// Fixed, not a setting: YIAN_MODEL_RETRIES bounds how long retrying takes.
/** openai's MAX_RETRY_DELAY: the longest backoff between model retries, in seconds. */
const MAX_RETRY_DELAY_SECONDS = 8;

/** What every wire protocol has. */
interface ProtocolBase {
  /** The protocol id, e.g. `openai_responses`; adapters put it in `Finish.protocol`. */
  readonly name: string;
  /** The model name the bodies ask for. */
  readonly model: string;
  /** Build the JSON request body for `request`. */
  buildBody(request: Request): JsonObject;
}

/** A wire protocol whose replies are server-sent events. */
export interface SseProtocol extends ProtocolBase {
  readonly framing: "sse";
  /** Turn the reply's events into stream events. */
  parse(frames: AsyncIterable<SSEEvent>): AsyncIterable<Event>;
}

/** A wire protocol whose replies are newline-delimited JSON. */
export interface NdjsonProtocol extends ProtocolBase {
  readonly framing: "ndjson";
  /** Turn the reply's JSON values into stream events. */
  parse(frames: AsyncIterable<JsonValue>): AsyncIterable<Event>;
}

/** A provider wire format: how a `Request` is sent and how its reply is read. */
export type WireProtocol = SseProtocol | NdjsonProtocol;

/** A `ChatModel` that sends a protocol's bodies over one route. */
export class RouteModel implements ChatModel {
  /** The route's provider name. */
  readonly provider: string;
  /** The protocol's model name. */
  readonly model: string;
  /** Where requests go and how they authenticate. */
  readonly route: Route;
  /** Builds the bodies and parses the replies. */
  readonly protocol: WireProtocol;
  readonly #send: Send | undefined;
  readonly #scheduler: Scheduler | undefined;

  /**
   * @param options `send` and `scheduler` go to the transport; they default to the
   *   runtime's fetch and the real clock.
   */
  constructor(options: { route: Route; protocol: WireProtocol; send?: Send; scheduler?: Scheduler }) {
    this.route = options.route;
    this.protocol = options.protocol;
    this.provider = options.route.provider;
    this.model = options.protocol.model;
    this.#send = options.send;
    this.#scheduler = options.scheduler;
  }

  /**
   * Start one call and return its events at once; nothing runs until the first `next()`.
   *
   * The first `next()` builds and encodes the body, resolves the auth headers and POSTs the
   * body with `content-type` and `accept` of `application/json`, overridden by the route's
   * headers, then by the auth headers, case-insensitively. A non-2xx reply throws a
   * `ProviderError` built from at most 64 KiB of its body; a 2xx reply's frames go to the
   * protocol. When an SSE parse ends `incomplete` but the body held text and no SSE line (a JSON
   * or HTML reply), a non-retryable `protocol` error built from the body's first 64 KiB takes
   * its place. A `ProviderError` from parsing that names no provider gets the route's;
   * the status error and the parse error are both scrubbed of the route's credentials, and the
   * parse error's message is then capped at 2,000 code points.
   * Closing the stream aborts the request.
   *
   * Errors: a `TypeError` for a body that isn't JSON or an invalid header value, before
   * sending; whatever the auth source throws, unchanged; the transport's errors; and the
   * protocol's.
   */
  stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    const stop = new AbortController();
    const signal = options.signal ? AbortSignal.any([options.signal, stop.signal]) : stop.signal;
    return new EventStream(this.#call(request, signal), () => stop.abort());
  }

  /** Make the call's one attempt, releasing its exchange however the parse ends. */
  async *#call(request: Request, signal: AbortSignal): AsyncGenerator<Event> {
    const { route, protocol } = this;
    signal.throwIfAborted();
    const body = encodeBody(protocol.buildBody(request));
    const headers = new Headers(DEFAULT_HEADERS);
    for (const [name, value] of Object.entries(route.headers)) headers.set(name, value);
    for (const [name, value] of Object.entries(await route.auth.headers())) headers.set(name, value);
    const exchange = await openExchange(
      route.url(),
      { method: "POST", headers, body },
      {
        provider: route.provider,
        credentials: () => route.auth.credentials(),
        signal,
        send: this.#send,
        scheduler: this.#scheduler,
      },
    );
    if (exchange.status < 200 || exchange.status >= 300) {
      const credentials = route.auth.credentials();
      const text = scrub(await readErrorBody(exchange, credentials), credentials);
      const error = errorFromResponse(exchange.status, exchange.headers, text, {
        provider: route.provider,
        credentials,
      });
      throw scrubbed(error, credentials);
    }
    const stats: SseStats = { sseLines: 0 };
    const head: Head = { chunks: [], size: 0 };
    try {
      if (protocol.framing === "sse") {
        yield* protocol.parse(iterSse(keepHead(exchange.body, head, stats), undefined, stats));
      } else {
        yield* protocol.parse(iterNdjson(exchange.body));
      }
    } catch (error) {
      if (!(error instanceof ProviderError)) throw error;
      const credentials = route.auth.credentials();
      let failure = error;
      // A body with no SSE line was never a stream, so sending it again gets the same reply.
      if (protocol.framing === "sse" && error.kind === "incomplete" && stats.sseLines === 0) {
        const text = scrub(headText(head.chunks, head.size, credentials), credentials);
        if (/\S/.test(text)) {
          failure = nonStreamError(exchange.status, exchange.headers, text, { provider: route.provider, credentials });
        }
      }
      // Protocol parsers don't know the route.
      if (failure.provider === null) failure.provider = route.provider;
      // Provider text, SSE error events included, may echo a credential the protocol never holds.
      // Capped only after the scrub, so a cut can't leave part of a credential unmatched.
      throw scrubbed(failure, credentials, true);
    } finally {
      // `body.return()` releases nothing before the body's first read.
      exchange.abort();
    }
  }
}

/**
 * Read at most 64 KiB of an error reply's body, decoded as UTF-8 with replacement. The body
 * aborts the request however the read ends: stopping at the cap returns it early, which
 * runs its own release. A `ProviderError` from a read keeps what arrived; any other error,
 * such as a caller abort, propagates.
 *
 * A body cut at the cap loses what `dropCredentialTail` drops: the cut may split a credential,
 * and no scrub matches the part before it.
 */
async function readErrorBody(exchange: Exchange, credentials: readonly string[]): Promise<string> {
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for await (const chunk of exchange.body) {
      chunks.push(chunk);
      size += chunk.byteLength;
      if (size >= MAX_ERROR_BODY) break;
    }
  } catch (error) {
    // The status already names the failure.
    if (!(error instanceof ProviderError)) throw error;
  }
  return headText(chunks, size, credentials);
}

/** The first bytes of a 2xx body, kept in case it turns out not to be a stream. */
interface Head {
  /** Copies of the body's first chunks, at most 64 KiB in all. */
  chunks: Uint8Array[];
  /** Their total size in bytes. */
  size: number;
}

/**
 * Yield `chunks` unchanged, copying into `head` what arrives while it holds under 64 KiB and
 * `stats` has counted no SSE line.
 */
async function* keepHead(chunks: AsyncIterable<Uint8Array>, head: Head, stats: SseStats): AsyncGenerator<Uint8Array> {
  for await (const chunk of chunks) {
    if (stats.sseLines === 0 && head.size < MAX_ERROR_BODY) {
      // Copied, since the source may reuse the chunk's buffer once it moves on.
      const part = chunk.slice(0, MAX_ERROR_BODY - head.size);
      head.chunks.push(part);
      head.size += part.byteLength;
    }
    yield chunk;
  }
}

/**
 * Return the first 64 KiB of `chunks`, `size` bytes in all, decoded as UTF-8 with replacement.
 * Text cut at the cap loses what `dropCredentialTail` drops: the cut may split a credential, and
 * no scrub matches the part before it.
 */
function headText(chunks: readonly Uint8Array[], size: number, credentials: readonly string[]): string {
  const bytes = new Uint8Array(Math.min(size, MAX_ERROR_BODY));
  let at = 0;
  for (const chunk of chunks) {
    const part = chunk.subarray(0, bytes.length - at);
    bytes.set(part, at);
    at += part.length;
  }
  const text = new TextDecoder().decode(bytes);
  return size < MAX_ERROR_BODY ? text : dropCredentialTail(text, credentials);
}

/**
 * Return `error`, or a copy with every field kept and the credentials replaced in its
 * message when it still holds one, or (with `cap`) when it is longer than 2,000 code points; a
 * capped message keeps its first 2,000 code points and "…", cut after the scrub.
 *
 * The body is scrubbed before parsing, but an echo with mixed escapes only matches a
 * credential once the parse has undone them. The error is rebuilt, not edited: a `stack` read
 * before the scrub would keep the old message.
 */
function scrubbed(error: ProviderError, credentials: readonly string[], cap = false): ProviderError {
  const clean = scrub(error.message, credentials);
  const message = cap ? capDetail(clean) : clean;
  if (message === error.message) return error;
  const { kind, status, retryAfter, retryable, provider, causeName } = error;
  return new ProviderError(kind, message, { status, retryAfter, retryable, provider, causeName });
}

/** A failed attempt about to be retried. */
export interface RetryInfo {
  /** The failed attempt's number, from 1. */
  readonly attempt: number;
  /** Why it failed. */
  readonly error: ProviderError;
  /** What arrived before the failure, or `null` before the first event. */
  readonly partial: Message | null;
  /** Seconds until the next attempt. */
  readonly retryIn: number;
}

/** Options for `callWithRetries`. */
export interface CallOptions {
  /** Stops the call: an attempt or a wait rejects with `signal.reason`. */
  signal?: AbortSignal;
  /** Retries allowed after the first attempt; defaults to `current().modelRetries`. */
  budget?: number;
  /** The clock waits run on; defaults to `realScheduler`. */
  scheduler?: Scheduler;
  /** A value in [0, 1) for backoff jitter; defaults to `Math.random`. */
  random?: () => number;
  /**
   * Attempts an earlier process already made and lost to a crash: the call first waits the last
   * one's `retryIn` seconds, then numbers its attempts after them, and they count toward `budget`.
   */
  prior?: { readonly attempts: number; readonly retryIn: number };
  /** Runs, and is awaited, right before each attempt's `model.stream`, after any wait. */
  onAttempt?(attempt: number): void | Promise<void>;
  /** Runs, and is awaited, for each event an attempt streams, with the attempt's number. */
  onEvent?(event: Event, attempt: number): void | Promise<void>;
  /** Runs, and is awaited, before each wait. */
  onRetry?(retry: RetryInfo): void | Promise<void>;
}

/**
 * Return the seconds to wait before the next attempt, as the openai SDK computes it.
 *
 * The error's `retryAfter` when finite and in (0, 120]; otherwise a backoff of
 * `min(0.5 * 2 ** retries, 8)` scaled by `1 - 0.25 * random()`.
 *
 * @param error The failure being retried.
 * @param retries The retries made so far.
 * @param random A value in [0, 1).
 */
export function retryDelay(error: ProviderError, retries: number, random: () => number): number {
  const after = error.retryAfter;
  // A non-finite or out-of-range retry-after (an adapter-built error, `inf`) gets the backoff instead.
  if (after !== null && Number.isFinite(after) && after > 0 && after <= MAX_RETRY_AFTER_SECONDS) return after;
  // The exponent is capped as openai does, so a huge retry count can't overflow.
  return Math.min(0.5 * 2 ** Math.min(retries, 1000), MAX_RETRY_DELAY_SECONDS) * (1 - 0.25 * random());
}

/**
 * Call `model` until one attempt's events assemble into a reply, retrying a retryable
 * `ProviderError` while the budget lasts, before or after partial output.
 *
 * Per failed attempt the order is `onAttempt(k)`, its events through `onEvent`, `onRetry`,
 * the wait of `retryDelay` seconds through the scheduler, then `onAttempt(k + 1)`. With `prior`,
 * the call starts with the prior wait and its first attempt is `prior.attempts + 1`. A
 * `ProviderError` naming no provider (the assembler's `incomplete`) gets `model.provider`
 * before `onRetry` sees it or it is thrown.
 *
 * Errors: a non-retryable `ProviderError`, or the last one when the budget runs out; any
 * other error from the model, such as a defect or an auth source's failure, unchanged; a
 * hook's error, never retried; and `signal.reason` on a stop.
 *
 * @returns The successful attempt's reply.
 */
export async function callWithRetries(model: ChatModel, request: Request, options: CallOptions = {}): Promise<Response> {
  const { signal, scheduler = realScheduler, random = Math.random, prior, onAttempt, onEvent, onRetry } = options;
  const budget = options.budget ?? current().modelRetries;
  if (prior !== undefined) {
    signal?.throwIfAborted();
    await scheduler.sleep(prior.retryIn * 1000, signal);
  }
  for (let attempt = (prior?.attempts ?? 0) + 1; ; attempt++) {
    // Before `onAttempt`, so a stopped call never announces an attempt that won't be sent.
    signal?.throwIfAborted();
    await onAttempt?.(attempt);
    const assembler = new ReplyAssembler();
    // Set while `onEvent` runs, so a hook's error is never taken for the model's.
    let inHook = false;
    try {
      for await (const event of model.stream(request, { signal })) {
        assembler.add(event);
        inHook = true;
        await onEvent?.(event, attempt);
        inHook = false;
      }
      return assembler.response();
    } catch (error) {
      // A stopped call ends with its reason, never an `onRetry` for an attempt it won't make.
      if (signal?.aborted) throw signal.reason;
      if (inHook || !(error instanceof ProviderError)) throw error;
      error.provider ??= model.provider;
      const retries = attempt - 1;
      if (!error.retryable || retries >= budget) throw error;
      const retry = { attempt, error, partial: assembler.partial(), retryIn: retryDelay(error, retries, random) };
      await onRetry?.(retry);
      await scheduler.sleep(retry.retryIn * 1000, signal);
    }
  }
}
