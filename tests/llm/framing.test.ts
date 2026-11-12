/** Tests for SSE and NDJSON framing over byte chunks in `src/llm/framing.ts`. */

import { describe, expect, test } from "bun:test";

import { withEnv } from "../../src/defaults.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { iterNdjson, iterSse, SSEEvent } from "../../src/llm/framing.ts";

const encoder = new TextEncoder();

/** Return the UTF-8 bytes of `text`. */
function utf8(text: string): Uint8Array {
  return encoder.encode(text);
}

/** Return the bytes of `codes`, one byte per number. */
function bytes(...codes: number[]): Uint8Array {
  return Uint8Array.from(codes);
}

/** Return the concatenation of `parts`. */
function concat(...parts: Uint8Array[]): Uint8Array {
  return Uint8Array.from(parts.flatMap((part) => Array.from(part)));
}

/** Collect every value of an async generator. */
async function collect<T>(values: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const value of values) out.push(value);
  return out;
}

/** Return the events of a stream sent as `chunks`, each a string (UTF-8) or bytes. */
function sse(...chunks: (string | Uint8Array)[]): Promise<SSEEvent[]> {
  return collect(iterSse(chunks.map((chunk) => (typeof chunk === "string" ? utf8(chunk) : chunk))));
}

/** Return the values of an NDJSON stream sent as `chunks`, each a string (UTF-8) or bytes. */
function ndjson(...chunks: (string | Uint8Array)[]) {
  return collect(iterNdjson(chunks.map((chunk) => (typeof chunk === "string" ? utf8(chunk) : chunk))));
}

/** Return the error `promise` rejects with; fail when it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (err) {
    return err;
  }
  throw new Error("expected a rejection");
}

/** Expect `err` to be the non-retryable protocol error `message`. */
function expectOverCap(err: unknown, message: string): void {
  expect(err).toBeInstanceOf(ProviderError);
  const error = err as ProviderError;
  expect(error.kind).toBe("protocol");
  expect(error.retryable).toBe(false);
  expect(error.message).toBe(message);
}

/** Return a source that yields `first`, then `chunk` `count` times, and a record of how many `chunk`s it gave. */
function countedSource(first: Uint8Array, chunk: Uint8Array, count: number) {
  const state = { given: 0 };
  function* source(): Generator<Uint8Array> {
    yield first;
    for (; state.given < count; ) {
      state.given += 1;
      yield chunk;
    }
  }
  return { source: source(), state };
}

/**
 * Return an async source that yields `chunk` forever, and a record of whether its iterator's
 * `return()` ran.
 */
function endlessSource(chunk: Uint8Array) {
  const state = { returned: false };
  const source: AsyncIterable<Uint8Array> = {
    [Symbol.asyncIterator]() {
      return {
        next: async () => ({ done: false, value: chunk }),
        return: async () => {
          state.returned = true;
          return { done: true, value: undefined };
        },
      };
    },
  };
  return { source, state };
}

/** Return a `ReadableStream` that enqueues `chunk` on every pull, and a record of whether it was cancelled. */
function endlessStream(chunk: Uint8Array) {
  const state = { cancelled: false };
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      controller.enqueue(chunk);
    },
    cancel() {
      state.cancelled = true;
    },
  });
  return { stream, state };
}

describe("SSE", () => {
  test("an event split at every offset", async () => {
    const raw = utf8('event: delta\nid: 7\ndata: {"a": "é"}\n\n');
    const expected = [new SSEEvent({ data: '{"a": "é"}', event: "delta", id: "7" })];
    for (let cut = 0; cut <= raw.length; cut++) {
      expect(await sse(raw.subarray(0, cut), raw.subarray(cut))).toEqual(expected);
    }
  });

  test("one-byte chunks", async () => {
    const raw = utf8("data: one\r\ndata: two\r\n\r\ndata: ü\n\n");
    const chunks = Array.from(raw, (_, i) => raw.subarray(i, i + 1));
    expect(await sse(...chunks)).toEqual([new SSEEvent({ data: "one\ntwo" }), new SSEEvent({ data: "ü" })]);
  });

  test("multi-line data", async () => {
    expect(await sse("data: a\ndata:b\ndata:  c\n\n")).toEqual([new SSEEvent({ data: "a\nb\n c" })]);
  });

  test("comments are ignored", async () => {
    expect(await sse(": keep-alive\n\n: ping\ndata: x\n: mid\n\n")).toEqual([new SSEEvent({ data: "x" })]);
  });

  test("event and id fields", async () => {
    const events = await sse("event: start\nid: 1\ndata: a\n\ndata: b\n\n");
    expect(events[0]).toEqual(new SSEEvent({ data: "a", event: "start", id: "1" }));
    // The event type resets after a dispatch; the last event id carries over (WHATWG).
    expect(events[1]).toEqual(new SSEEvent({ data: "b", event: null, id: "1" }));
  });

  test("an empty event field reads as null and an id holding NUL is ignored", async () => {
    expect(await sse("id: 1\ndata: a\n\nevent:\nid: 2\0\ndata: b\n\n")).toEqual([
      new SSEEvent({ data: "a", id: "1" }),
      new SSEEvent({ data: "b", event: null, id: "1" }),
    ]);
  });

  test("retry and unknown fields are ignored", async () => {
    expect(await sse("retry: 100\nfoo: bar\ndata: x\nnocolon\n\n")).toEqual([new SSEEvent({ data: "x" })]);
  });

  test("a field without a value", async () => {
    expect(await sse("data\n\ndata:\n\n")).toEqual([new SSEEvent({ data: "" }), new SSEEvent({ data: "" })]);
  });

  test("a CRLF split across chunks", async () => {
    expect(await sse("data: a\r", "\ndata: b\r", "\n\r", "\n")).toEqual([new SSEEvent({ data: "a\nb" })]);
  });

  test("an LF alone after a split CRLF is a blank line", async () => {
    expect(await sse("data: a\r", "\n", "\n")).toEqual([new SSEEvent({ data: "a" })]);
  });

  test("a bare CR", async () => {
    expect(await sse("data: a\rdata: b\r\rdata: c\r\r")).toEqual([
      new SSEEvent({ data: "a\nb" }),
      new SSEEvent({ data: "c" }),
    ]);
  });

  test("a BOM split across chunks is stripped", async () => {
    expect(await sse(bytes(0xef, 0xbb), concat(bytes(0xbf), utf8("data: x\n\n")))).toEqual([
      new SSEEvent({ data: "x" }),
    ]);
  });

  test("only one BOM is stripped", async () => {
    expect(await sse(concat(bytes(0xef, 0xbb, 0xbf, 0xef, 0xbb, 0xbf), utf8("data: x\n\n")))).toEqual([]);
  });

  test("a UTF-8 character split across chunks", async () => {
    const raw = utf8("data: 日本\n\n");
    const cut = utf8("data: 日").length + 1;
    expect(await sse(raw.subarray(0, cut), raw.subarray(cut))).toEqual([new SSEEvent({ data: "日本" })]);
  });

  test("invalid UTF-8 is replaced", async () => {
    expect(await sse(concat(utf8("data: "), bytes(0xff), utf8("\n\n")))).toEqual([new SSEEvent({ data: "�" })]);
  });

  test("an event without data is not dispatched", async () => {
    expect(await sse("event: ping\nid: 3\n\ndata: x\n\n")).toEqual([new SSEEvent({ data: "x", id: "3" })]);
  });

  test("an unterminated event at the end is dropped", async () => {
    expect(await sse("data: a\n\ndata: b\n")).toEqual([new SSEEvent({ data: "a" })]);
    expect(await sse("data: a\n\ndata: b")).toEqual([new SSEEvent({ data: "a" })]);
  });

  test("a megabyte line in small chunks is linear", async () => {
    const raw = utf8("data: " + "x".repeat(1_000_000) + "\n\n");
    const chunks: Uint8Array[] = [];
    for (let i = 0; i < raw.length; i += 64) chunks.push(raw.subarray(i, i + 64));
    const start = performance.now();
    const events = await collect(iterSse(chunks));
    expect(performance.now() - start).toBeLessThan(2000);
    expect(events.length).toBe(1);
    expect(events[0]!.data.length).toBe(1_000_000);
  });

  test("a line past the default cap throws before the stream ends", async () => {
    const mib = utf8("x".repeat(1024 * 1024));
    const { source, state } = countedSource(utf8("data: "), mib, 64);
    const err = await rejection(collect(iterSse(source)));
    expectOverCap(err, "SSE event at line 1 is over the limit of 16777216 bytes (YIAN_MAX_MODEL_EVENT_BYTES)");
    expect(state.given).toBeLessThan(64);
  });

  test("data lines past the cap throw naming the line", async () => {
    const err = await rejection(collect(iterSse([utf8("data: aaaa\ndata: bbbb\ndata: cccc\n\n")], 14)));
    expectOverCap(err, "SSE event at line 3 is over the limit of 14 bytes (YIAN_MAX_MODEL_EVENT_BYTES)");
  });

  test("an event at the cap is read and the next event starts afresh", async () => {
    const events = await collect(iterSse([utf8("data: aaaa\n\ndata: bbbb\n\n")], 10));
    expect(events).toEqual([new SSEEvent({ data: "aaaa" }), new SSEEvent({ data: "bbbb" })]);
  });

  test("the cap counts UTF-8 bytes", async () => {
    const err = await rejection(collect(iterSse([utf8("data: \u00e9\u00e9\n\n")], 9)));
    expectOverCap(err, "SSE event at line 1 is over the limit of 9 bytes (YIAN_MAX_MODEL_EVENT_BYTES)");
    const events = await collect(iterSse([utf8("data: \u00e9\u00e9\n\n")], 10));
    expect(events).toEqual([new SSEEvent({ data: "\u00e9\u00e9" })]);
  });

  test("the cap defaults to YIAN_MAX_MODEL_EVENT_BYTES", async () => {
    const env = (name: string) => (name === "YIAN_MAX_MODEL_EVENT_BYTES" ? "4" : undefined);
    const err = await rejection(withEnv(env, () => collect(iterSse([utf8("data: abc\n\n")]))));
    expectOverCap(err, "SSE event at line 1 is over the limit of 4 bytes (YIAN_MAX_MODEL_EVENT_BYTES)");
  });

  test("is lazy", async () => {
    function* chunks() {
      yield utf8("data: a\n\n");
      throw new Error("read past the first event");
    }
    expect((await iterSse(chunks()).next()).value).toEqual(new SSEEvent({ data: "a" }));
  });

  test("a break after the first event closes the source", async () => {
    const { source, state } = endlessSource(utf8("data: a\n\n"));
    for await (const event of iterSse(source)) {
      expect(event).toEqual(new SSEEvent({ data: "a" }));
      break;
    }
    expect(state.returned).toBe(true);
  });

  test("a return() after the first event cancels a ReadableStream source", async () => {
    const { stream, state } = endlessStream(utf8("data: a\n\n"));
    const events = iterSse(stream);
    expect((await events.next()).value).toEqual(new SSEEvent({ data: "a" }));
    await events.return(undefined);
    expect(state.cancelled).toBe(true);
  });
});

describe("SSE line count", () => {
  /** Return how many SSE lines `stats` counted in `text`, sent in chunks of `size` bytes. */
  async function count(text: string, size = 3): Promise<number> {
    const raw = utf8(text);
    const chunks: Uint8Array[] = [];
    for (let i = 0; i < raw.length; i += size) chunks.push(raw.subarray(i, i + size));
    const stats = { sseLines: 0 };
    await collect(iterSse(chunks, undefined, stats));
    return stats.sseLines;
  }

  test("comments and data, event, id and retry fields count", async () => {
    expect(await count(": keep-alive\ndata: a\nevent: x\nid: 1\nretry: 5\ndata\n\n")).toBe(6);
  });

  test("unknown fields and blank lines don't count", async () => {
    expect(await count('{"error": {"message": "bad key"}}\n\n<html>\nfoo: bar\n\r\n')).toBe(0);
  });

  test("an unfinished last line counts when it is a comment or a known field", async () => {
    expect(await count('data: {"id": "chatcmpl-1", "choices": [')).toBe(1);
    expect(await count("event: message_st")).toBe(1);
    expect(await count(": OPENROUTER PROC")).toBe(1);
    expect(await count('{"error": {"message": "bad key"}}')).toBe(0);
    expect(await count("<html>gateway</html>")).toBe(0);
  });

  test("no stats leaves parsing unchanged", async () => {
    expect(await sse(": c\nfoo: bar\ndata: a\n\n")).toEqual([new SSEEvent({ data: "a" })]);
  });
});

describe("NDJSON", () => {
  test("lines split at every offset", async () => {
    const raw = utf8('{"a": 1}\n{"b": [2, 3]}\n');
    for (let cut = 0; cut <= raw.length; cut++) {
      expect(await ndjson(raw.subarray(0, cut), raw.subarray(cut))).toEqual([{ a: 1n }, { b: [2n, 3n] }]);
    }
  });

  test("blank lines are skipped", async () => {
    expect(await ndjson("\n1\n\n  \r\n\t\v\f\n2\r\n\n")).toEqual([1n, 2n]);
  });

  test("a line holding only a no-break space is not blank", async () => {
    const err = await rejection(ndjson("1\n\u00a0\n"));
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).message).toBe("invalid JSON line 2 in NDJSON stream");
  });

  test("a final line without a newline counts", async () => {
    expect(await ndjson('{"a": 1}\n{"done": tr', "ue}")).toEqual([{ a: 1n }, { done: true }]);
  });

  test("invalid JSON throws a protocol error naming the line number, not the line", async () => {
    const values = iterNdjson([utf8('{"a": 1}\n\n{"secret": sk-live-abc\n')]);
    expect((await values.next()).value).toEqual({ a: 1n });
    const err = await rejection(values.next());
    expect(err).toBeInstanceOf(ProviderError);
    const error = err as ProviderError;
    expect(error.kind).toBe("protocol");
    expect(error.message).toBe("invalid JSON line 3 in NDJSON stream");
    expect(String(error.stack)).not.toContain("sk-live-abc");
    expect("cause" in error).toBe(false);
  });

  test("an invalid final line counts its number", async () => {
    const err = await rejection(ndjson("1\nnope"));
    expect((err as ProviderError).message).toBe("invalid JSON line 2 in NDJSON stream");
  });

  test("a duplicate key is invalid JSON", async () => {
    const err = await rejection(ndjson('{"a": 1, "a": 2}\n'));
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).kind).toBe("protocol");
  });

  test("a line past the received-JSON bounds throws a protocol error naming the bound, not the line", async () => {
    const deep = "[".repeat(10_000) + "]".repeat(10_000);
    expectOverCap(await rejection(ndjson(`1\n${deep}\n`)), "NDJSON line 2 nests deeper than 512 levels");
    const few = (name: string) => (name === "YIAN_MAX_RECEIVED_JSON_VALUES" ? "3" : undefined);
    expect(await withEnv(few, () => ndjson("[1,2]\n"))).toEqual([[1n, 2n]]);
    expectOverCap(
      await withEnv(few, () => rejection(ndjson("[1,2,3]\n"))),
      "NDJSON line 1 has more than 3 JSON values (YIAN_MAX_RECEIVED_JSON_VALUES)",
    );
  });

  test("invalid UTF-8 throws a protocol error", async () => {
    const err = await rejection(ndjson(concat(utf8('"'), bytes(0xff), utf8('"\n'))));
    expect(err).toBeInstanceOf(ProviderError);
    expect((err as ProviderError).kind).toBe("protocol");
    expect("cause" in (err as ProviderError)).toBe(false);
  });

  test("a line past the cap throws before the stream ends, naming its number", async () => {
    const { source, state } = countedSource(utf8("1\n"), utf8("x".repeat(4)), 64);
    const err = await rejection(collect(iterNdjson(source, 10)));
    expectOverCap(err, "NDJSON line 2 is over the limit of 10 bytes (YIAN_MAX_MODEL_EVENT_BYTES)");
    expect(state.given).toBeLessThan(64);
  });

  test("a line at the cap is read", async () => {
    expect(await collect(iterNdjson([utf8('"12345678"\n"1234567"')], 10))).toEqual(["12345678", "1234567"]);
  });

  test("a final line past the cap throws", async () => {
    const err = await rejection(collect(iterNdjson([utf8("1\n12345678901")], 10)));
    expectOverCap(err, "NDJSON line 2 is over the limit of 10 bytes (YIAN_MAX_MODEL_EVENT_BYTES)");
  });

  test("a break after the first value closes the source", async () => {
    const { source, state } = endlessSource(utf8("1\n"));
    for await (const value of iterNdjson(source)) {
      expect(value).toBe(1n);
      break;
    }
    expect(state.returned).toBe(true);
  });

  test("a return() after the first value cancels a ReadableStream source", async () => {
    const { stream, state } = endlessStream(utf8("1\n"));
    const values = iterNdjson(stream);
    expect((await values.next()).value).toBe(1n);
    await values.return(undefined);
    expect(state.cancelled).toBe(true);
  });
});
