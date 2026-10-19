/**
 * Tests for Ollama reply parsing in `src/llm/protocols/ollama_chat.ts`: NDJSON lines into deltas,
 * thinking and content pieces joined into one reasoning and one text part, tool calls with their
 * ids and arguments, the `done` line's finish reason and usage, the message record, ignored
 * fields, the skipped `load` line, the end of the reply, truncation and cancellation.
 */

import { describe, expect, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import {
  collect,
  type Event,
  Finish,
  PartDone,
  ReasoningDelta,
  TextDelta,
  ToolCallDelta,
  UsageReport,
} from "../../src/llm/events.ts";
import {
  type Part,
  ReasoningPart,
  type ReplayKey,
  replayRecord,
  TextPart,
  ToolCallPart,
  Usage,
} from "../../src/llm/messages.ts";
import { OllamaChat, PROTOCOL } from "../../src/llm/protocols/ollama_chat.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

const MODEL = "qwen3";
const KEY: ReplayKey = { protocol: PROTOCOL, provider: "ollama", model: MODEL };
const USAGE = new Usage({ inputTokens: 17, outputTokens: 9, totalTokens: 26 });

function ollama(): OllamaChat {
  return new OllamaChat({ model: MODEL, provider: "ollama" });
}

interface ChunkOptions {
  thinking?: string;
  toolCalls?: JsonValue[];
  model?: string;
}

/** Return a streamed line holding a piece of the message, as the probe's `o_chunk` builds it. */
function oChunk(content = "", options: ChunkOptions = {}): JsonObject {
  const message: JsonObject = { role: "assistant", content };
  if (options.thinking !== undefined) message["thinking"] = options.thinking;
  if (options.toolCalls !== undefined) message["tool_calls"] = options.toolCalls;
  return { model: options.model ?? MODEL, created_at: "2026-10-16T00:00:00Z", message, done: false };
}

/** Return the `done` line, as the probe's `o_done` builds it, with `extra` over its fields. */
function oDone(extra: JsonObject = {}): JsonObject {
  return {
    model: MODEL,
    created_at: "2026-10-16T00:00:00Z",
    message: { role: "assistant", content: "" },
    done: true,
    done_reason: "stop",
    total_duration: 1000n,
    load_duration: 10n,
    prompt_eval_count: 17n,
    prompt_eval_duration: 100n,
    eval_count: 9n,
    eval_duration: 200n,
    ...extra,
  };
}

/** Return a `done` line without `fields`. */
function doneWithout(...fields: string[]): JsonObject {
  const line = oDone();
  for (const field of fields) delete line[field];
  return line;
}

/** Return a wire tool call named `double`, with `id` when given and `args` as its arguments. */
function wireCall(id?: string, args: JsonValue = { value: 3n }): JsonObject {
  const call: JsonObject = { function: { name: "double", arguments: args } };
  if (id !== undefined) call["id"] = id;
  return call;
}

async function* frames(lines: JsonValue[]): AsyncGenerator<JsonValue> {
  yield* lines;
}

async function parse(lines: JsonValue[], protocol = ollama()): Promise<Event[]> {
  const out: Event[] = [];
  for await (const event of protocol.parse(frames(lines))) out.push(event);
  return out;
}

/** Return the `ProviderError` parsing `lines` throws; fail the test if none is thrown. */
async function failure(lines: JsonValue[]): Promise<ProviderError> {
  try {
    await parse(lines);
  } catch (error) {
    expect(error).toBeInstanceOf(ProviderError);
    return error as ProviderError;
  }
  throw new Error("expected a ProviderError");
}

async function partsOf(lines: JsonValue[]): Promise<readonly Part[]> {
  return (await collect(await parse(lines))).message.parts;
}

/** Return the usage the reply ending in `done` reports, `null` for none. */
async function usageOf(done: JsonObject): Promise<Usage | null> {
  return (await collect(await parse([oChunk("Hi"), done]))).usage;
}

/** Return the finish reason of the reply `lines`. */
async function reasonOf(lines: JsonValue[]): Promise<string | null> {
  return (await collect(await parse(lines))).finishReason;
}

function finish(reason: string, fields: JsonObject = { response_model: MODEL }): Finish {
  return new Finish({ reason, responseId: null, protocol: PROTOCOL, providerPayload: replayRecord(KEY, fields) });
}

function thought(text: string): ReasoningPart {
  return new ReasoningPart(text, replayRecord(KEY, {}));
}

/** Return a call part as the parser builds one named `double`. */
function callPart(
  id: string,
  args: JsonObject | null = { value: 3n },
  rawArguments: string | null = null,
): ToolCallPart {
  return new ToolCallPart({ id, name: "double", arguments: args, rawArguments });
}

function callDelta(index: number, id: string, args = '{"value":3}'): ToolCallDelta {
  return new ToolCallDelta({ index, id, name: "double", arguments: args });
}

describe("deltas and parts", () => {
  test("plain text over chunks streams its deltas, then its PartDone, usage and finish", async () => {
    expect(await parse([oChunk("Hel"), oChunk("lo."), oDone()])).toEqual([
      new TextDelta(0, "Hel"),
      new TextDelta(0, "lo."),
      new PartDone(0, new TextPart("Hello.")),
      new UsageReport(USAGE),
      finish("stop"),
    ]);
  });

  test("thinking then content are a reasoning part with a record at 0 and a text part at 1", async () => {
    expect(await parse([oChunk("", { thinking: "Hmm." }), oChunk("Hi."), oDone()])).toEqual([
      new ReasoningDelta(0, "Hmm."),
      new TextDelta(1, "Hi."),
      new PartDone(0, thought("Hmm.")),
      new PartDone(1, new TextPart("Hi.")),
      new UsageReport(USAGE),
      finish("stop"),
    ]);
  });

  test("content then thinking are a text part at 0 and a reasoning part at 1", async () => {
    const out = await parse([oChunk("Hi."), oChunk("", { thinking: "Hmm." }), oDone()]);
    expect(out.slice(0, 4)).toEqual([
      new TextDelta(0, "Hi."),
      new ReasoningDelta(1, "Hmm."),
      new PartDone(0, new TextPart("Hi.")),
      new PartDone(1, thought("Hmm.")),
    ]);
  });

  test("content pieces join one text part when a call comes between them", async () => {
    const out = await parse([oChunk("A"), oChunk("", { toolCalls: [wireCall()] }), oChunk("B"), oDone()]);
    expect(out.slice(0, 5)).toEqual([
      new TextDelta(0, "A"),
      callDelta(1, "yian_call_0"),
      new TextDelta(0, "B"),
      new PartDone(0, new TextPart("AB")),
      new PartDone(1, callPart("yian_call_0")),
    ]);
  });

  test("thinking pieces join one reasoning part when content comes between them", async () => {
    const lines = [oChunk("", { thinking: "x" }), oChunk("Hi"), oChunk("", { thinking: "y" }), oDone()];
    const out = await parse(lines);
    expect(out.slice(0, 3)).toEqual([new ReasoningDelta(0, "x"), new TextDelta(1, "Hi"), new ReasoningDelta(0, "y")]);
    expect(await partsOf(lines)).toEqual([thought("xy"), new TextPart("Hi")]);
  });

  test("empty pieces open no part", async () => {
    expect(await parse([oChunk("", { thinking: "" }), oDone()])).toEqual([new UsageReport(USAGE), finish("stop")]);
    const lines = [oChunk("", { thinking: "" }), oChunk("Hi", { thinking: "" }), oChunk("", { thinking: "" }), oDone()];
    expect(await parse(lines)).toEqual([
      new TextDelta(0, "Hi"),
      new PartDone(0, new TextPart("Hi")),
      new UsageReport(USAGE),
      finish("stop"),
    ]);
  });

  test("a done line's thinking, content and call are emitted before its finish", async () => {
    const done = oDone({
      message: { role: "assistant", content: "Hi", thinking: "Hmm", tool_calls: [wireCall()] },
    });
    expect(await parse([done])).toEqual([
      new ReasoningDelta(0, "Hmm"),
      new TextDelta(1, "Hi"),
      callDelta(2, "yian_call_0"),
      new PartDone(0, thought("Hmm")),
      new PartDone(1, new TextPart("Hi")),
      new PartDone(2, callPart("yian_call_0")),
      new UsageReport(USAGE),
      finish("tool_calls"),
    ]);
  });

  test("a done line with no message, or a null one, ends the reply", async () => {
    for (const message of [undefined, null]) {
      const done = message === undefined ? doneWithout("message") : oDone({ message });
      expect(await parse([oChunk("Hi"), done])).toEqual([
        new TextDelta(0, "Hi"),
        new PartDone(0, new TextPart("Hi")),
        new UsageReport(USAGE),
        finish("stop"),
      ]);
    }
  });

  test("null content and thinking are no pieces", async () => {
    const message: JsonObject = { role: "assistant", content: null, thinking: null, tool_calls: null };
    const line: JsonObject = { ...oChunk(), message };
    expect(await parse([line, oDone()])).toEqual([new UsageReport(USAGE), finish("stop")]);
  });
});

describe("tool calls", () => {
  test("one call streams its delta at once and finishes as tool_calls", async () => {
    expect(await parse([oChunk("", { toolCalls: [wireCall()] }), oDone()])).toEqual([
      callDelta(0, "yian_call_0"),
      new PartDone(0, callPart("yian_call_0")),
      new UsageReport(USAGE),
      finish("tool_calls"),
    ]);
  });

  test("two calls in two lines or in one line are yian_call_0 and yian_call_1", async () => {
    const want = [callPart("yian_call_0"), callPart("yian_call_1")];
    const twoLines = [oChunk("", { toolCalls: [wireCall()] }), oChunk("", { toolCalls: [wireCall()] }), oDone()];
    expect(await partsOf(twoLines)).toEqual(want);
    expect(await partsOf([oChunk("", { toolCalls: [wireCall(), wireCall()] }), oDone()])).toEqual(want);
  });

  test("the server's call ids are kept", async () => {
    const lines = [oChunk("", { toolCalls: [wireCall("call_ab12cd34"), wireCall("call_ef56gh78")] }), oDone()];
    expect(await partsOf(lines)).toEqual([callPart("call_ab12cd34"), callPart("call_ef56gh78")]);
  });

  test("a repeated or empty wire id becomes yian_call_<n>, with _1 appended when that is taken", async () => {
    const repeated = [oChunk("", { toolCalls: [wireCall("call_a"), wireCall("call_a")] }), oDone()];
    expect(await partsOf(repeated)).toEqual([callPart("call_a"), callPart("yian_call_1")]);
    const empty = [oChunk("", { toolCalls: [wireCall(""), wireCall()] }), oDone()];
    expect(await partsOf(empty)).toEqual([callPart("yian_call_0"), callPart("yian_call_1")]);
    const taken = [oChunk("", { toolCalls: [wireCall("yian_call_1"), wireCall("yian_call_1")] }), oDone()];
    expect(await partsOf(taken)).toEqual([callPart("yian_call_1"), callPart("yian_call_1_1")]);
  });

  test("object arguments are kept as sent, string values and a functionName key untouched", async () => {
    const args: JsonObject = { value: "[1, 2]", nested: '{"a": 1}', functionName: "double" };
    const out = await parse([oChunk("", { toolCalls: [wireCall(undefined, args)] }), oDone()]);
    expect(out[0]).toEqual(callDelta(0, "yian_call_0", encodeBody(args)));
    expect((await collect(out)).message.parts).toEqual([callPart("yian_call_0", args)]);
  });

  test("string arguments are parsed", async () => {
    const out = await parse([oChunk("", { toolCalls: [wireCall(undefined, '{"value": 3}')] }), oDone()]);
    expect(out[0]).toEqual(callDelta(0, "yian_call_0"));
    expect((await collect(out)).message.parts).toEqual([callPart("yian_call_0")]);
  });

  test("an empty string, missing or null arguments are {}", async () => {
    for (const call of [wireCall(undefined, ""), { function: { name: "double" } }, wireCall(undefined, null)]) {
      const out = await parse([oChunk("", { toolCalls: [call] }), oDone()]);
      expect(out[0]).toEqual(callDelta(0, "yian_call_0", "{}"));
      expect((await collect(out)).message.parts).toEqual([callPart("yian_call_0", {})]);
    }
  });

  test("a string that is not a JSON object makes an invalid call with the raw text", async () => {
    for (const args of ["[1]", "{bad"]) {
      const out = await parse([oChunk("", { toolCalls: [wireCall(undefined, args)] }), oDone()]);
      expect(out[0]).toEqual(callDelta(0, "yian_call_0", args));
      expect((await collect(out)).message.parts).toEqual([callPart("yian_call_0", null, args)]);
    }
  });

  test("a number or list makes an invalid call with its JSON as the raw text", async () => {
    for (const [args, raw] of [
      [7n, "7"],
      [[1n, "a"], '[1,"a"]'],
    ] as [JsonValue, string][]) {
      const out = await parse([oChunk("", { toolCalls: [wireCall(undefined, args)] }), oDone()]);
      expect(out[0]).toEqual(callDelta(0, "yian_call_0", raw));
      expect((await collect(out)).message.parts).toEqual([callPart("yian_call_0", null, raw)]);
    }
  });
});

describe("finish reason", () => {
  test("stop, a missing, null or empty done_reason are stop", async () => {
    const dones = [oDone(), doneWithout("done_reason"), oDone({ done_reason: null }), oDone({ done_reason: "" })];
    for (const done of dones) {
      expect(await reasonOf([oChunk("Hi"), done])).toBe("stop");
    }
  });

  test("length wins over calls", async () => {
    const lines = [oChunk("", { toolCalls: [wireCall()] }), oDone({ done_reason: "length" })];
    expect(await reasonOf(lines)).toBe("length");
    expect(await reasonOf([oChunk("Hi"), oDone({ done_reason: "length" })])).toBe("length");
  });

  test("a load line with content finishes as load, its text emitted", async () => {
    const done = oDone({ done_reason: "load", message: { role: "assistant", content: "Hi" } });
    const out = await parse([done]);
    expect(out[0]).toEqual(new TextDelta(0, "Hi"));
    expect(out.at(-1)).toEqual(finish("load"));
  });

  test("other reasons are passed through", async () => {
    expect(await reasonOf([oChunk("Hi"), oDone({ done_reason: "unload" })])).toBe("unload");
  });
});

describe("usage", () => {
  test("both counts are input and output, with a total and no cache or reasoning counts", async () => {
    const usage = await usageOf(oDone());
    expect(usage).toEqual(USAGE);
    expect(usage?.cacheReadTokens).toBeNull();
    expect(usage?.cacheCreationTokens).toBeNull();
    expect(usage?.reasoningTokens).toBeNull();
  });

  test("a missing or null count is 0", async () => {
    const outputOnly = new Usage({ inputTokens: 0, outputTokens: 9, totalTokens: 9 });
    expect(await usageOf(doneWithout("prompt_eval_count"))).toEqual(outputOnly);
    expect(await usageOf(oDone({ prompt_eval_count: null }))).toEqual(outputOnly);
    expect(await usageOf(doneWithout("eval_count"))).toEqual(
      new Usage({ inputTokens: 17, outputTokens: 0, totalTokens: 17 }),
    );
  });

  test("no counts, null counts or zero counts report no usage", async () => {
    for (const done of [
      doneWithout("prompt_eval_count", "eval_count"),
      oDone({ prompt_eval_count: null, eval_count: null }),
      oDone({ prompt_eval_count: 0n, eval_count: 0n }),
    ]) {
      const out = await parse([oChunk("Hi"), done]);
      expect(out.some((event) => event instanceof UsageReport)).toBe(false);
    }
  });

  test("prompt_eval_cached_count is the cache reads, 0 included", async () => {
    const cached = (count: number) =>
      new Usage({ inputTokens: 17, outputTokens: 9, totalTokens: 26, cacheReadTokens: count });
    expect(await usageOf(oDone({ prompt_eval_cached_count: 12n }))).toEqual(cached(12));
    expect(await usageOf(oDone({ prompt_eval_cached_count: 0n }))).toEqual(cached(0));
  });

  test("a cached count alone reports usage", async () => {
    const done = { ...doneWithout("prompt_eval_count", "eval_count"), prompt_eval_cached_count: 5n };
    expect(await usageOf(done)).toEqual(
      new Usage({ inputTokens: 0, outputTokens: 0, totalTokens: 0, cacheReadTokens: 5 }),
    );
  });
});

describe("message record and ignored fields", () => {
  test("the last string model on any line is the record's response_model", async () => {
    const out = await parse([oChunk("Hi", { model: "qwen3:8b" }), { ...oDone(), model: "qwen3:14b" }]);
    expect(out.at(-1)).toEqual(finish("stop", { response_model: "qwen3:14b" }));
    const kept = await parse([oChunk("Hi", { model: "qwen3:8b" }), doneWithout("model")]);
    expect(kept.at(-1)).toEqual(finish("stop", { response_model: "qwen3:8b" }));
  });

  test("no model on any line gives an empty record", async () => {
    const chunk = oChunk("Hi");
    delete chunk["model"];
    const out = await parse([chunk, doneWithout("model")]);
    expect(out.at(-1)).toEqual(finish("stop", {}));
  });

  test("unknown fields on a line or its message change nothing", async () => {
    const plain = [oChunk("Hi", { thinking: "Hmm", toolCalls: [wireCall("call_1")] }), oDone()];
    const extra: JsonObject = {
      logprobs: [{ token: "Hi", logprob: -0.1 }],
      remote_model: "gpt-oss:120b",
      remote_host: "https://ollama.com:443",
      brand_new: { a: 1n },
    };
    const chunk = oChunk("Hi", { thinking: "Hmm", toolCalls: [wireCall("call_1")] });
    (chunk["message"] as JsonObject)["images"] = null;
    (chunk["message"] as JsonObject)["brand_new"] = "x";
    const noisy = [{ ...chunk, ...extra }, { ...oDone(), ...extra }];
    expect(await parse(noisy)).toEqual(await parse(plain));
  });
});

describe("the load line and the end of the reply", () => {
  test("a load line with missing, empty or blank content is skipped whole, then the reply is read", async () => {
    for (const content of [undefined, "", " \n\t "]) {
      const message: JsonObject = { role: "assistant", thinking: "Hmm", tool_calls: [wireCall()] };
      if (content !== undefined) message["content"] = content;
      const load = oDone({ done_reason: "load", message });
      expect(await parse([load, oChunk("Hi"), oDone()])).toEqual([
        new TextDelta(0, "Hi"),
        new PartDone(0, new TextPart("Hi")),
        new UsageReport(USAGE),
        finish("stop"),
      ]);
    }
  });

  test("lines after the done line are never read, and the source is closed", async () => {
    let closed = false;
    let pulled = 0;
    async function* source(): AsyncGenerator<JsonValue> {
      try {
        pulled++;
        yield oChunk("Hi");
        pulled++;
        yield oDone();
        throw new Error("read past the done line");
      } finally {
        closed = true;
      }
    }
    const out: Event[] = [];
    for await (const event of ollama().parse(source())) out.push(event);
    expect(out.at(-1)).toEqual(finish("stop"));
    expect(pulled).toBe(2);
    expect(closed).toBe(true);
  });

  test("a reply that ends before its done line is incomplete and retryable", async () => {
    const load = oDone({ done_reason: "load", message: { role: "assistant", content: "" } });
    for (const lines of [[oChunk("Hi")], [], [load]]) {
      const error = await failure(lines);
      expect(error.kind).toBe("incomplete");
      expect(error.retryable).toBe(true);
      expect(error.message).toBe("ollama stream ended before a complete response");
    }
  });

  test("a line that is not a plain object is a protocol error", async () => {
    for (const line of [[1n], "x", null, 3n]) {
      const error = await failure([oChunk("Hi"), line, oDone()]);
      expect(error.kind).toBe("protocol");
      expect(error.message).toBe("ollama stream line is not a JSON object");
    }
  });
});

describe("cancellation", () => {
  test("a consumer that stops after the first delta closes the source after one pull", async () => {
    let closed = false;
    let pulled = 0;
    async function* source(): AsyncGenerator<JsonValue> {
      try {
        for (const line of [oChunk("Hel"), oChunk("lo."), oDone()]) {
          pulled++;
          yield line;
        }
      } finally {
        closed = true;
      }
    }
    for await (const event of ollama().parse(source())) {
      expect(event).toEqual(new TextDelta(0, "Hel"));
      break;
    }
    expect(closed).toBe(true);
    expect(pulled).toBe(1);
  });
});
