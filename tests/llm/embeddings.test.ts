/** Tests for `openaiEmbeddings` in `src/llm/embeddings.ts`: the request, the reply's checks, errors and batching. */

import { expect, test } from "bun:test";

import { BUILTIN, withEnv } from "../../src/defaults.ts";
import { BearerAuth, REDACTED } from "../../src/llm/auth.ts";
import { openaiEmbeddings } from "../../src/llm/embeddings.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { type Send, virtualScheduler } from "../../src/platform.ts";

const KEY = "sk-embed-5c0e7a";

/** One request the fake `send` received. */
interface Sent {
  url: string;
  body: unknown;
  auth: string | null;
}

/** A `send` that records each request and answers it with `reply(body, index)`. */
function fake(reply: (body: { input: string[] }, index: number) => Response): { sent: Sent[]; send: Send } {
  const sent: Sent[] = [];
  const send: Send = async (url, init) => {
    const body = JSON.parse(String(init.body));
    sent.push({ url: String(url), body, auth: new Headers(init.headers).get("authorization") });
    return reply(body, sent.length - 1);
  };
  return { sent, send };
}

/** A reply embedding each input as `[its index in the request, its length]`. */
function echo(body: { input: string[] }): Response {
  return Response.json({ data: body.input.map((text, index) => ({ index, embedding: [index, text.length] })) });
}

/** Return an embedder over `send` with the test key. */
function embedder(send: Send, scheduler = virtualScheduler()) {
  return openaiEmbeddings({ provider: "openai", baseUrl: "https://api.example/v1", auth: new BearerAuth(KEY), model: "m", send, scheduler });
}

/** Return the error `promise` rejects with, failing the test if it resolves. */
async function rejected(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

const signal = new AbortController().signal;

test("texts are embedded in one POST to <base>/embeddings, vectors in input order", async () => {
  const sent: { url: string; body: unknown; auth: string | null }[] = [];
  const send = async (url: string | URL, init: RequestInit) => {
    sent.push({ url: String(url), body: JSON.parse(String(init.body)), auth: new Headers(init.headers).get("authorization") });
    return Response.json({ data: [{ index: 1, embedding: [0, 1] }, { index: 0, embedding: [1, 0] }] });
  };
  const embed = openaiEmbeddings({ provider: "openai", baseUrl: "https://api.example/v1", auth: new BearerAuth("k"), model: "text-embedding-3-small", send });
  const vectors = await embed(["a", "b"], { signal: new AbortController().signal });
  expect(sent).toEqual([{ url: "https://api.example/v1/embeddings", body: { model: "text-embedding-3-small", input: ["a", "b"] }, auth: "Bearer k" }]);
  expect(vectors.map((v) => [...v])).toEqual([[1, 0], [0, 1]]);
  expect(vectors.every((v) => v instanceof Float32Array)).toBe(true);
});

test("route headers and the JSON content type go with the request", async () => {
  let headers = new Headers();
  const send: Send = async (_url, init) => {
    headers = new Headers(init.headers);
    return Response.json({ data: [{ index: 0, embedding: [0.5] }] });
  };
  const embed = openaiEmbeddings({
    provider: "openai",
    baseUrl: "https://api.example/v1/",
    auth: new BearerAuth(KEY),
    model: "m",
    headers: { "OpenAI-Project": "proj-1" },
    send,
  });
  expect((await embed(["a"], { signal })).map((v) => [...v])).toEqual([[0.5]]);
  expect([headers.get("content-type"), headers.get("openai-project")]).toEqual(["application/json", "proj-1"]);
});

test("no texts send no request", async () => {
  const { sent, send } = fake(echo);
  expect(await embedder(send)([], { signal })).toEqual([]);
  expect(sent).toEqual([]);
});

test("a 4xx reply is a ProviderError with the key scrubbed", async () => {
  const { send } = fake(() =>
    Response.json({ error: { message: `Incorrect API key provided: ${KEY}.` } }, { status: 401 }),
  );
  const error = (await rejected(embedder(send)(["a"], { signal }))) as ProviderError;
  expect(error).toBeInstanceOf(ProviderError);
  expect([error.kind, error.status, error.provider, error.retryable]).toEqual(["auth", 401, "openai", false]);
  expect(error.message).toBe(`openai returned HTTP 401 (auth): Incorrect API key provided: ${REDACTED}.`);
  expect(error.stack).not.toContain(KEY);
});

test("a 429 reply is a retryable rate_limit error", async () => {
  const { send } = fake(() => Response.json({ error: { message: "slow down" } }, { status: 429 }));
  const error = (await rejected(embedder(send)(["a"], { signal }))) as ProviderError;
  expect([error.kind, error.retryable]).toEqual(["rate_limit", true]);
});

test("a reply whose vectors differ in length or count is a protocol error", async () => {
  const index = "a missing, repeated or out-of-range index";
  const numbers = "an embedding that isn't a list of numbers";
  const lengths = "vectors of different lengths, or an empty one";
  const replies: [string, string, string][] = [
    ["a body that isn't a reply", '{"error": null}', "no data list"],
    ["one vector too few", '{"data": [{"index": 0, "embedding": [1, 0]}]}', "1 vectors for 2 texts"],
    ["one vector too many", `{"data": [${'{"index": 0, "embedding": [1]},'.repeat(2)}{"index": 2, "embedding": [1]}]}`, "3 vectors for 2 texts"],
    ["vectors of different lengths", '{"data": [{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1]}]}', lengths],
    ["empty vectors", '{"data": [{"index": 0, "embedding": []}, {"index": 1, "embedding": []}]}', lengths],
    ["an index given twice", '{"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}]}', index],
    ["an index out of range", '{"data": [{"index": 0, "embedding": [1]}, {"index": 2, "embedding": [2]}]}', index],
    ["a float index", '{"data": [{"index": 0, "embedding": [1]}, {"index": 1.0, "embedding": [2]}]}', index],
    ["a vector item that isn't a number", '{"data": [{"index": 0, "embedding": [1]}, {"index": 1, "embedding": ["2"]}]}', numbers],
    ["a number past float32", '{"data": [{"index": 0, "embedding": [1]}, {"index": 1, "embedding": [1e300]}]}', numbers],
    ["a base64 vector", '{"data": [{"index": 0, "embedding": "AAAA"}, {"index": 1, "embedding": "AAAA"}]}', numbers],
  ];
  for (const [what, reply, problem] of replies) {
    const { send } = fake(() => new Response(reply, { status: 200 }));
    const error = (await rejected(embedder(send)(["a", "b"], { signal }))) as ProviderError;
    expect([what, error instanceof ProviderError, error.kind, error.provider, error.retryable, error.message]).toEqual([
      what,
      true,
      "protocol",
      "openai",
      false,
      `openai returned an invalid embeddings reply: ${problem}`,
    ]);
  }
  const { send } = fake(() => new Response("not json", { status: 200 }));
  const error = (await rejected(embedder(send)(["a"], { signal }))) as ProviderError;
  expect([error.kind, error.message]).toEqual(["protocol", "openai returned an invalid JSON body"]);
});

test("more than EMBEDDING_BATCH_ITEMS texts go in several requests", async () => {
  expect(BUILTIN.embeddingBatchItems).toBe(64);
  const texts = Array.from({ length: 65 }, (_, i) => "x".repeat(i + 1));
  const { sent, send } = fake(echo);
  const vectors = await embedder(send)(texts, { signal });
  expect(sent.map(({ body }) => (body as { input: string[] }).input.length)).toEqual([64, 1]);
  expect(vectors.map((v) => [...v])).toEqual(texts.map((text, i) => [i % 64, text.length]));

  const scoped = fake(echo);
  const env = (name: string) => (name === "YIAN_EMBEDDING_BATCH_ITEMS" ? "2" : undefined);
  const five = ["a", "b", "c", "d", "e"];
  await withEnv(env, () => embedder(scoped.send)(five, { signal }));
  expect(scoped.sent.map(({ body }) => (body as { input: string[] }).input)).toEqual([["a", "b"], ["c", "d"], ["e"]]);
});

test("batches whose vectors differ in length are a protocol error", async () => {
  const env = (name: string) => (name === "YIAN_EMBEDDING_BATCH_ITEMS" ? "1" : undefined);
  const { send } = fake((_body, index) => Response.json({ data: [{ index: 0, embedding: index === 0 ? [1, 0] : [1] }] }));
  const error = (await rejected(withEnv(env, () => embedder(send)(["a", "b"], { signal })))) as ProviderError;
  expect([error.kind, error.provider]).toEqual(["protocol", "openai"]);
});

test("a request gets EMBEDDING_TIMEOUT_SECONDS to answer", async () => {
  expect(BUILTIN.embeddingTimeoutSeconds).toBe(60);
  const clock = virtualScheduler();
  const send: Send = (_url, init) =>
    new Promise((_resolve, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason)));
  const error = (await rejected(embedder(send, clock)(["a"], { signal }))) as ProviderError;
  expect([error.kind, error.message]).toEqual(["timeout", "openai request failed (timeout): no complete response within 60 s"]);
  expect(clock.now()).toBe(60_000);
});

test("a caller stop rejects with its reason", async () => {
  const stop = new AbortController();
  const send: Send = (_url, init) =>
    new Promise((_resolve, reject) => init.signal!.addEventListener("abort", () => reject(init.signal!.reason)));
  const pending = embedder(send)(["a"], { signal: stop.signal });
  stop.abort(new Error("stopped"));
  expect((await rejected(pending)).message).toBe("stopped");
});
