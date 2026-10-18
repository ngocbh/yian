/**
 * Send a model request over a route and read its reply as events: `RouteModel` joins a
 * route (where the body goes and how it authenticates), a wire protocol (how a `Request`
 * becomes a body and the reply's frames become events) and the transport, making one
 * attempt per call. Retries happen above the call.
 *
 * Every error leaving a call is scrubbed of the credentials the route sent, in its
 * message and its `stack`.
 */

import type { Scheduler, Send } from "../platform.ts";
import type { JsonObject, JsonValue } from "../typesys/json.ts";
import { scrub } from "./auth.ts";
import { encodeBody } from "./body.ts";
import { errorFromResponse, ProviderError } from "./errors.ts";
import { type ChatModel, type Event, EventStream } from "./events.ts";
import { iterNdjson, iterSse, type SSEEvent } from "./framing.ts";
import type { Request } from "./messages.ts";
import type { Route } from "./route.ts";
import { type Exchange, openExchange } from "./transport.ts";

// The openai and anthropic SDKs send this Accept on every request, streamed ones too.
const DEFAULT_HEADERS = { "content-type": "application/json", accept: "application/json" };
const MAX_ERROR_BODY = 64 * 1024;

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
   * protocol, and a `ProviderError` from parsing that names no provider gets the route's.
   * Closing the stream aborts the request.
   *
   * Errors: a `TypeError` for a body that isn't JSON or an invalid header value, before
   * sending; whatever the auth source throws, unchanged; the transport's errors; and the
   * protocol's.
   */
  stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    const stop = new AbortController();
    const signal = options.signal ? AbortSignal.any([options.signal, stop.signal]) : stop.signal;
    let exchange: Exchange | undefined;
    const events = this.#call(request, signal, (opened) => {
      exchange = opened;
    });
    // Aborting `stop` also ends an `openExchange` still pending; `body.return()` alone would
    // release nothing before the body's first read.
    return new EventStream(events, () => {
      stop.abort();
      exchange?.abort();
    });
  }

  /** Make the call's one attempt, handing the exchange to `opened` once it exists. */
  async *#call(request: Request, signal: AbortSignal, opened: (exchange: Exchange) => void): AsyncGenerator<Event> {
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
    opened(exchange);
    if (exchange.status < 200 || exchange.status >= 300) {
      const credentials = route.auth.credentials();
      const text = scrub(await readErrorBody(exchange), credentials);
      const error = errorFromResponse(exchange.status, exchange.headers, text, { provider: route.provider });
      throw scrubbed(error, credentials);
    }
    try {
      if (protocol.framing === "sse") yield* protocol.parse(iterSse(exchange.body));
      else yield* protocol.parse(iterNdjson(exchange.body));
    } catch (error) {
      // Protocol parsers don't know the route.
      if (error instanceof ProviderError && error.provider === null) error.provider = route.provider;
      throw error;
    }
  }
}

/**
 * Read at most 64 KiB of an error reply's body, decoded as UTF-8 with replacement, and
 * abort the request. A `ProviderError` from a read keeps what arrived; any other error,
 * such as a caller abort, propagates.
 */
async function readErrorBody(exchange: Exchange): Promise<string> {
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
  } finally {
    exchange.abort();
  }
  const bytes = new Uint8Array(Math.min(size, MAX_ERROR_BODY));
  let at = 0;
  for (const chunk of chunks) {
    const part = chunk.subarray(0, bytes.length - at);
    bytes.set(part, at);
    at += part.length;
  }
  return new TextDecoder().decode(bytes);
}

/**
 * Return `error`, or a copy with every field kept and the credentials replaced in its
 * message when it still holds one.
 *
 * The body is scrubbed before parsing, but an echo with mixed escapes only matches a
 * credential once the parse has undone them. The error is rebuilt, not edited, because its
 * `stack` keeps the message it was built with.
 */
function scrubbed(error: ProviderError, credentials: readonly string[]): ProviderError {
  const message = scrub(error.message, credentials);
  if (message === error.message) return error;
  const { kind, status, retryAfter, retryable, provider, causeName } = error;
  return new ProviderError(kind, message, { status, retryAfter, retryable, provider, causeName });
}
