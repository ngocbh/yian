/**
 * Embeddings over the OpenAI-compatible `/embeddings` protocol: texts in, one vector per text out.
 *
 * `openaiEmbeddings` returns an `Embedder` bound to one endpoint, auth and model. A call posts the
 * texts in batches of at most `current().embeddingBatchItems`, each a short request under
 * `current().embeddingTimeoutSeconds`, and checks that every reply holds one vector per text, all
 * of one length. Nothing is retried: the caller decides what a failed call means.
 */

import { current } from "../defaults.ts";
import type { Scheduler, Send } from "../platform.ts";
import type { JsonValue } from "../typesys/json.ts";
import { type Auth, scrub } from "./auth.ts";
import { encodeBody } from "./body.ts";
import { errorFromResponse, ProviderError } from "./errors.ts";
import { isPlainObject } from "./messages.ts";
import { Route } from "./route.ts";
import { shortRequest } from "./transport.ts";

/** Return one vector per text, in the texts' order, all of one length; `signal` stops the call. */
export type Embedder = (texts: readonly string[], o: { signal: AbortSignal }) => Promise<Float32Array[]>;

/** An embedding model: the provider whose `/embeddings` endpoint serves it, and its name there. */
export interface EmbeddingModel {
  readonly provider: string;
  readonly model: string;
}

/** Embed `texts` with the embedding model `o.provider`/`o.model`, as an `Embedder` does. */
export type Embed = (
  texts: readonly string[],
  o: EmbeddingModel & { signal: AbortSignal },
) => Promise<Float32Array[]>;

/** Options for `openaiEmbeddings`. */
export interface EmbeddingsOptions {
  /** The provider name, for error messages. */
  provider: string;
  /** The API base `embeddings` is joined to, for example `"https://api.openai.com/v1"`. */
  baseUrl: string;
  /** The credential headers; their values are scrubbed from error text. */
  auth: Auth;
  /** The embedding model each request names. */
  model: string;
  /** Non-secret extra headers, such as `OpenAI-Project`; the auth headers win a clash. */
  headers?: Record<string, string>;
  /** The raw sender passed to `safeFetch`. */
  send?: Send;
  /** The clock the request timer runs on; defaults to the real one. */
  scheduler?: Scheduler;
}

// What the openai SDK sends on every request.
const DEFAULT_HEADERS = { "content-type": "application/json", accept: "application/json" };

/**
 * Return an `Embedder` that POSTs `{model, input}` to `<baseUrl>/embeddings`.
 *
 * Each request sends `content-type` and `accept` of `application/json`, overridden by `headers`,
 * then by the auth headers, case-insensitively. A non-2xx reply throws `errorFromResponse`'s
 * `ProviderError`, scrubbed of the auth's credentials; a reply that isn't one vector of finite
 * numbers per text, all of one non-zero length across every batch, throws a `protocol` one.
 * Transport failures and a stop reject as `shortRequest` does. No texts send no request.
 *
 * @throws TypeError At construction, if a header name or value may not be sent.
 */
export function openaiEmbeddings(options: EmbeddingsOptions): Embedder {
  const { model, send, scheduler } = options;
  const route = new Route({
    provider: options.provider,
    baseUrl: options.baseUrl,
    path: "embeddings",
    auth: options.auth,
    headers: options.headers,
  });
  return async (texts, { signal }) => {
    // Read per call, so a run's scoped defaults apply to the calls it makes.
    const { embeddingBatchItems, embeddingTimeoutSeconds } = current();
    const vectors: Float32Array[] = [];
    for (let start = 0; start < texts.length; start += embeddingBatchItems) {
      const batch = texts.slice(start, start + embeddingBatchItems);
      const headers = new Headers(DEFAULT_HEADERS);
      for (const [name, value] of Object.entries(route.headers)) headers.set(name, value);
      for (const [name, value] of Object.entries(await route.auth.headers())) headers.set(name, value);
      const reply = await shortRequest(
        route.url(),
        { method: "POST", headers, body: encodeBody({ model, input: batch }) },
        {
          name: route.provider,
          credentials: () => route.auth.credentials(),
          timeoutMs: embeddingTimeoutSeconds * 1000,
          signal,
          scheduler,
          send,
        },
      );
      if (reply.status < 200 || reply.status >= 300) {
        const credentials = route.auth.credentials();
        throw errorFromResponse(reply.status, reply.headers, scrub(reply.text, credentials), {
          provider: route.provider,
          credentials,
        });
      }
      for (const vector of vectorsOf(reply.json(), batch.length, route.provider, vectors[0]?.length ?? null)) {
        vectors.push(vector);
      }
    }
    return vectors;
  };
}

/**
 * Return the vectors of an `/embeddings` reply to `count` texts, in input order: `data` must list
 * `count` items, each with a distinct int `index` below `count` and an `embedding` list of numbers
 * that are finite as float32, all of `length` (or, when `null`, of one non-zero length).
 *
 * @throws ProviderError Of kind `protocol`, naming what is wrong, otherwise.
 */
function vectorsOf(reply: JsonValue, count: number, provider: string, length: number | null): Float32Array[] {
  const invalid = (problem: string) =>
    new ProviderError("protocol", `${provider} returned an invalid embeddings reply: ${problem}`, { provider });
  const data = isPlainObject(reply) ? reply["data"] : undefined;
  if (!Array.isArray(data)) throw invalid("no data list");
  if (data.length !== count) throw invalid(`${data.length} vectors for ${count} texts`);
  const vectors: (Float32Array | undefined)[] = Array.from({ length: count });
  for (const item of data) {
    const index = isPlainObject(item) ? item["index"] : undefined;
    // The reply is read as yian JSON, so an int index is a bigint.
    if (typeof index !== "bigint" || index < 0n || index >= count || vectors[Number(index)] !== undefined) {
      throw invalid("a missing, repeated or out-of-range index");
    }
    const embedding = isPlainObject(item) ? item["embedding"] : undefined;
    const numbers = Array.isArray(embedding) && embedding.every((x) => typeof x === "number" || typeof x === "bigint");
    const vector = numbers ? Float32Array.from(embedding as (number | bigint)[], Number) : null;
    // A number past float32's range becomes an infinity, which no similarity can use.
    if (vector === null || !vector.every(Number.isFinite)) throw invalid("an embedding that isn't a list of numbers");
    vectors[Number(index)] = vector;
  }
  const expected = length ?? vectors[0]!.length;
  if (expected === 0 || vectors.some((vector) => vector!.length !== expected)) {
    throw invalid("vectors of different lengths, or an empty one");
  }
  return vectors as Float32Array[];
}
