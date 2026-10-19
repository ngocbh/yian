/**
 * Authentication HTTP requests with a capped, secret-safe result.
 *
 * One total timer covers the whole exchange, and redirects are never followed. Failures are
 * `shortRequest`'s, so callers can tell them apart: a `ProviderError` of kind `timeout` when the
 * time runs out, `protocol` for a body over the limit, `connection` or `timeout` for a network
 * failure; a caller abort rejects with its reason.
 */

import { current } from "../../defaults.ts";
import { ProviderError } from "../../llm/errors.ts";
import { type ShortReply, shortRequest } from "../../llm/transport.ts";
import type { Scheduler, Send } from "../../platform.ts";
import { HttpResponse } from "./common.ts";

/** Options for `request`. */
export interface HttpRequestOptions {
  /** The request headers. */
  headers: Record<string, string>;
  /** Longest wait for the whole reply. */
  timeoutMs: number;
  /** A body sent as JSON text; wins over `form`. */
  jsonBody?: Readonly<Record<string, unknown>>;
  /** A body sent as URL-encoded form text. */
  form?: Readonly<Record<string, string>>;
  /** Stops the request; a stop rejects with `signal.reason`. */
  signal?: AbortSignal;
  /** The clock the timer runs on; defaults to `realScheduler`. */
  scheduler?: Scheduler;
  /** The largest body read, in bytes; defaults to `current().maxAuthResponseBytes`. */
  bodyLimit?: number;
  /** The secrets to scrub from error text. */
  credentials?: () => Iterable<string>;
  /** The raw sender passed to `safeFetch`. */
  send?: Send;
}

/**
 * Send one request and return its status and decoded JSON body, without leaking credentials
 * through redirects or logs.
 *
 * `data` is `null` for a body that isn't JSON; integers in it are `bigint`s. A 3xx is never
 * followed: it returns its status with `null` data. Every other failure propagates as
 * `shortRequest` throws it.
 */
export async function request(method: string, url: string, options: HttpRequestOptions): Promise<HttpResponse> {
  const { headers, timeoutMs, jsonBody, form, signal, scheduler, credentials, send } = options;
  const bodyLimit = options.bodyLimit ?? current().maxAuthResponseBytes;
  let body: string | undefined;
  if (jsonBody !== undefined) body = JSON.stringify(jsonBody);
  else if (form !== undefined) body = new URLSearchParams(form).toString();
  let reply: ShortReply;
  try {
    reply = await shortRequest(
      url,
      { method, headers, body },
      { name: "HTTP", timeoutMs, maxBytes: bodyLimit, signal, scheduler, credentials, send },
    );
  } catch (error) {
    if (error instanceof ProviderError && error.kind === "redirect" && error.status !== null) {
      return new HttpResponse(error.status, null);
    }
    throw error;
  }
  let data: unknown;
  try {
    data = reply.json();
  } catch (error) {
    if (!(error instanceof ProviderError && error.kind === "protocol")) throw error;
    data = null;
  }
  return new HttpResponse(reply.status, data);
}
