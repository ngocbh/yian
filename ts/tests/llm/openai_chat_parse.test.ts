/**
 * Tests for OpenAI Chat Completions reply parsing in `src/llm/protocols/openai_chat.ts`: SSE
 * chunks into deltas, parts with replay records, dense indexes, tool-call assembly, usage, finish
 * reasons, error chunks and stream errors, and parsed replies sent back.
 */

import { describe, expect, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import { type ErrorKind, ProviderError } from "../../src/llm/errors.ts";
import {
  collect,
  type Event,
  Finish,
  parseArguments,
  PartDone,
  ReasoningDelta,
  ReplyAssembler,
  TextDelta,
  ToolCallDelta,
  UsageReport,
} from "../../src/llm/events.ts";
import { SSEEvent } from "../../src/llm/framing.ts";
import {
  Message,
  type Part,
  ReasoningPart,
  readReplayRecord,
  RefusalPart,
  type ReplayDegradedReason,
  type ReplayKey,
  replayRecord,
  Request,
  Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  Usage,
} from "../../src/llm/messages.ts";
import { OpenAIChat, PROTOCOL } from "../../src/llm/protocols/openai_chat.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

const USAGE: JsonObject = {
  prompt_tokens: 21,
  completion_tokens: 40,
  total_tokens: 61,
  prompt_tokens_details: { cached_tokens: 5 },
  completion_tokens_details: { reasoning_tokens: 30 },
};
// DeepSeek reports cache hits in its own fields, not prompt_tokens_details.
const USAGE_DEEPSEEK: JsonObject = {
  prompt_tokens: 21,
  completion_tokens: 40,
  total_tokens: 61,
  prompt_cache_hit_tokens: 5,
  prompt_cache_miss_tokens: 16,
  completion_tokens_details: { reasoning_tokens: 30 },
};
const USAGE_COUNTS = new Usage({
  inputTokens: 21,
  outputTokens: 40,
  totalTokens: 61,
  cacheReadTokens: 5,
  reasoningTokens: 30,
});

function route(provider: string, model: string): OpenAIChat {
  return new OpenAIChat({ model, provider });
}

function xai(): OpenAIChat {
  return route("xai", "grok-4");
}

function deepseek(): OpenAIChat {
  return route("deepseek", "deepseek-reasoner");
}

function keyOf(protocol: OpenAIChat): ReplayKey {
  return { protocol: PROTOCOL, provider: protocol.provider, model: protocol.model };
}

interface ChunkOptions {
  finishReason?: JsonValue;
  usage?: JsonValue;
  choices?: JsonValue;
  id?: JsonValue;
  model?: JsonValue;
}

/** Return a Chat Completions stream chunk carrying `delta` in its one choice. */
function chunk(delta: JsonValue, options: ChunkOptions = {}): JsonObject {
  const choices = Object.hasOwn(options, "choices")
    ? options.choices!
    : [{ index: 0, delta, finish_reason: options.finishReason ?? null, logprobs: null }];
  const data: JsonObject = {
    id: Object.hasOwn(options, "id") ? options.id! : "chatcmpl-1",
    object: "chat.completion.chunk",
    created: 1700000000,
    model: Object.hasOwn(options, "model") ? options.model! : "m",
    choices,
  };
  if (Object.hasOwn(options, "usage")) data["usage"] = options.usage!;
  return data;
}

/** Return `data` without `key`. */
function without(data: JsonObject, key: string): JsonObject {
  const { [key]: _, ...rest } = data;
  return rest;
}

interface StreamOptions {
  usage?: JsonObject | null;
  model?: string;
  finishReason?: string;
  usageChunk?: boolean;
  reasoningKey?: string;
}

/** Return the chunks a vendor streams for assistant message `msg`, as the probes build them. */
function streamChunks(msg: JsonObject, options: StreamOptions = {}): JsonObject[] {
  const model = options.model ?? "m";
  const usage = options.usage === undefined ? USAGE : options.usage;
  const reasoningKey = options.reasoningKey ?? "reasoning_content";
  const out = [chunk({ role: "assistant", content: "" }, { model })];
  const reasoning = msg[reasoningKey];
  if (typeof reasoning === "string" && reasoning) {
    const half = reasoning.length >> 1;
    for (const piece of [reasoning.slice(0, half), reasoning.slice(half)]) {
      out.push(chunk({ [reasoningKey]: piece }, { model }));
    }
  }
  if (msg["reasoning_details"]) out.push(chunk({ reasoning_details: msg["reasoning_details"] }, { model }));
  const text = msg["content"];
  if (typeof text === "string" && text) {
    for (const piece of [text.slice(0, 5), text.slice(5)]) out.push(chunk({ content: piece }, { model }));
  }
  out.push(chunk({}, { finishReason: options.finishReason ?? "stop", model }));
  if ((options.usageChunk ?? true) && usage !== null) out.push(chunk(null, { usage, model, choices: [] }));
  return out;
}

/** Return each chunk as the frame the API sends, then `[DONE]` unless `done` is false. */
function sse(chunks: (JsonValue | SSEEvent)[], { done = true }: { done?: boolean } = {}): SSEEvent[] {
  const frames = chunks.map((each) => (each instanceof SSEEvent ? each : new SSEEvent({ data: encodeBody(each) })));
  if (done) frames.push(new SSEEvent({ data: "[DONE]" }));
  return frames;
}

async function* frames(events: SSEEvent[]): AsyncGenerator<SSEEvent> {
  yield* events;
}

/** Parse `chunks` with `protocol`, pushing each event to `out` as it comes. */
async function drain(chunks: (JsonValue | SSEEvent)[], protocol: OpenAIChat, done: boolean, out: Event[]): Promise<void> {
  for await (const event of protocol.parse(frames(sse(chunks, { done })))) out.push(event);
}

async function parse(chunks: (JsonValue | SSEEvent)[], protocol = xai(), done = true): Promise<Event[]> {
  const out: Event[] = [];
  await drain(chunks, protocol, done, out);
  return out;
}

/** Return the events parsed before the failure and the `ProviderError`; fail the test if none is thrown. */
async function failing(
  chunks: (JsonValue | SSEEvent)[],
  protocol = xai(),
  done = true,
): Promise<[Event[], ProviderError]> {
  const out: Event[] = [];
  try {
    await drain(chunks, protocol, done, out);
  } catch (error) {
    expect(error).toBeInstanceOf(ProviderError);
    expect((error as ProviderError).provider).toBeNull();
    return [out, error as ProviderError];
  }
  throw new Error("expected a ProviderError");
}

async function failure(chunks: (JsonValue | SSEEvent)[], protocol = xai(), done = true): Promise<ProviderError> {
  return (await failing(chunks, protocol, done))[1];
}

/** Return the `Finish` the parser gives for a reply from `protocol`. */
function finish(
  reason: string,
  protocol = xai(),
  fields: JsonObject = { response_id: "chatcmpl-1", response_model: "m" },
) {
  const responseId = typeof fields["response_id"] === "string" ? fields["response_id"] : null;
  return new Finish({ reason, responseId, protocol: PROTOCOL, providerPayload: replayRecord(keyOf(protocol), fields) });
}

function reasoningPart(text: string, field: string, protocol = xai()): ReasoningPart {
  return new ReasoningPart(text, replayRecord(keyOf(protocol), { field }));
}

function partsDone(events: Event[]): PartDone[] {
  return events.filter((event) => event instanceof PartDone);
}

/** Return the parts a stream of `chunks` collects to. */
async function partsOf(chunks: (JsonValue | SSEEvent)[], protocol = xai()): Promise<readonly Part[]> {
  return (await collect(await parse(chunks, protocol))).message.parts;
}

const STOP = chunk({}, { finishReason: "stop" });

describe("deltas and parts", () => {
  test("the leading empty content opens no part; a text delta and its PartDone share an index", async () => {
    const events = await parse([chunk({ role: "assistant", content: "" }), chunk({ content: "Hi" }), STOP]);
    expect(events).toEqual([new TextDelta(0, "Hi"), new PartDone(0, new TextPart("Hi")), finish("stop")]);
  });

  test("reasoning then text have dense indexes; parts, usage and finish follow the deltas", async () => {
    const msg = { role: "assistant", content: "Hello there.", reasoning_content: "Thinking." };
    expect(await parse(streamChunks(msg))).toEqual([
      new ReasoningDelta(0, "Thin"),
      new ReasoningDelta(0, "king."),
      new TextDelta(1, "Hello"),
      new TextDelta(1, " there."),
      new PartDone(0, reasoningPart("Thinking.", "reasoning_content")),
      new PartDone(1, new TextPart("Hello there.")),
      new UsageReport(USAGE_COUNTS),
      finish("stop"),
    ]);
  });

  test.each([
    ["reasoning_content", { reasoning_content: "Thinking." }],
    ["reasoning", { reasoning: "Thinking." }],
    ["reasoning_details", { reasoning_details: [{ type: "reasoning.text", id: "r1", text: "Thinking." }] }],
  ])("reasoning from %s", async (field, delta) => {
    const events = await parse([chunk(delta), STOP]);
    expect(events).toEqual([
      new ReasoningDelta(0, "Thinking."),
      new PartDone(0, reasoningPart("Thinking.", field)),
      finish("stop"),
    ]);
  });

  test("two sources in one chunk count once: the first non-empty one", async () => {
    const cases: [JsonObject, string][] = [
      // vLLM sends both fields with the same text.
      [{ reasoning_content: "Thinking.", reasoning: "Thinking." }, "reasoning_content"],
      // OpenRouter sends `reasoning` and `reasoning_details`.
      [{ reasoning: "Thinking.", reasoning_details: [{ type: "reasoning.text", text: "Thinking." }] }, "reasoning"],
      [
        { reasoning_content: null, reasoning: "", reasoning_details: [{ type: "reasoning.text", text: "Thinking." }] },
        "reasoning_details",
      ],
    ];
    for (const [delta, field] of cases) {
      expect(await partsOf([chunk(delta), STOP])).toEqual([reasoningPart("Thinking.", field)]);
    }
  });

  test("field comes from the first piece", async () => {
    const parts = await partsOf([chunk({ reasoning: "Think" }), chunk({ reasoning_content: "ing." }), STOP]);
    expect(parts).toEqual([reasoningPart("Thinking.", "reasoning")]);
  });

  test("an empty reasoning_content beside a non-empty source gives that source's field", async () => {
    for (const protocol of [xai(), deepseek()]) {
      const parts = await partsOf([chunk({ reasoning_content: "", reasoning: "x" }), STOP], protocol);
      expect(parts).toEqual([reasoningPart("x", "reasoning", protocol)]);
    }
  });

  test("only reasoning.text items with string text count in reasoning_details", async () => {
    const details: JsonValue[] = [
      { type: "reasoning.summary", summary: "S" },
      { type: "reasoning.encrypted", data: "E" },
      { type: "reasoning.text", text: "One " },
      { type: "reasoning.text", text: 5 },
      { type: "reasoning.text" },
      { text: "untyped" },
      "bare",
      5,
      null,
      [],
      { type: "reasoning.text", text: "two." },
    ];
    expect(await partsOf([chunk({ reasoning_details: details }), STOP])).toEqual([
      reasoningPart("One two.", "reasoning_details"),
    ]);
  });

  test("on an echo route an empty reasoning_content opens the part; elsewhere it doesn't", async () => {
    const chunks = [chunk({ role: "assistant", content: "", reasoning_content: "" }), chunk({ content: "Hi" }), STOP];
    const echo = await parse(chunks, deepseek());
    expect(partsDone(echo)).toEqual([
      new PartDone(0, reasoningPart("", "reasoning_content", deepseek())),
      new PartDone(1, new TextPart("Hi")),
    ]);
    expect(echo.filter((event) => event instanceof ReasoningDelta)).toEqual([]);
    expect(partsDone(await parse(chunks, xai()))).toEqual([new PartDone(0, new TextPart("Hi"))]);
  });

  test("an echo-route stream whose only reasoning is empty gives an empty reasoning part", async () => {
    const parts = await partsOf([chunk({ reasoning_content: "" }), chunk({ reasoning_content: "" }), STOP], deepseek());
    expect(parts).toEqual([reasoningPart("", "reasoning_content", deepseek())]);
  });

  test("an empty reasoning_content that opened the part keeps its field", async () => {
    const parts = await partsOf([chunk({ reasoning_content: "" }), chunk({ reasoning: "x" }), STOP], deepseek());
    expect(parts).toEqual([reasoningPart("x", "reasoning_content", deepseek())]);
  });

  test("one delta's reasoning, content and refusal take indexes 0, 1, 2", async () => {
    const events = await parse([chunk({ refusal: "No.", content: "Hi", reasoning_content: "Hm." }), STOP]);
    expect(events).toEqual([
      new ReasoningDelta(0, "Hm."),
      new TextDelta(1, "Hi"),
      new PartDone(0, reasoningPart("Hm.", "reasoning_content")),
      new PartDone(1, new TextPart("Hi")),
      new PartDone(2, new RefusalPart("No.")),
      finish("stop"),
    ]);
  });

  test("content after the finish chunk is kept", async () => {
    const parts = await partsOf([chunk({ content: "Hello" }), STOP, chunk({ content: " there." })]);
    expect(parts).toEqual([new TextPart("Hello there.")]);
  });

  test("the finish reason is the last non-null one", async () => {
    const lengthThenStop = [chunk({}, { finishReason: "length" }), STOP];
    expect((await parse(lengthThenStop)).at(-1)).toEqual(finish("stop"));
    const lengthThenNull = [chunk({}, { finishReason: "length" }), chunk({ content: "x" })];
    expect((await parse(lengthThenNull)).at(-1)).toEqual(finish("length"));
  });

  test("chunks whose choices are absent, null or empty contribute only usage", async () => {
    for (const data of [
      without(chunk(null, { usage: USAGE }), "choices"),
      chunk(null, { usage: USAGE, choices: null }),
      chunk(null, { usage: USAGE, choices: [] }),
    ]) {
      const events = await parse([chunk({ content: "Hi" }), STOP, data]);
      expect(events).toEqual([
        new TextDelta(0, "Hi"),
        new PartDone(0, new TextPart("Hi")),
        new UsageReport(USAGE_COUNTS),
        finish("stop"),
      ]);
    }
  });

  test("a missing, null or non-object delta is empty", async () => {
    const choice = { index: 0, finish_reason: null };
    const chunks: JsonValue[] = [
      chunk(null, { choices: [choice] }),
      chunk(null, { choices: [{ ...choice, delta: null }] }),
      chunk(null, { choices: [{ ...choice, delta: "text" }] }),
      chunk(null, { choices: [{ ...choice, delta: 5 }] }),
      chunk({ content: "Hi" }),
      STOP,
    ];
    expect(await partsOf(chunks)).toEqual([new TextPart("Hi")]);
  });

  test("unknown keys are ignored", async () => {
    const data = {
      ...chunk({ content: "Hi", audio: { id: "a" }, role: "assistant" }),
      system_fingerprint: "fp",
      extra: [1],
    };
    const choice = { index: 0, delta: {}, finish_reason: "stop", native_finish_reason: "end_turn", logprobs: null };
    expect(await partsOf([data, chunk(null, { choices: [choice] })])).toEqual([new TextPart("Hi")]);
  });

  test("list content: text items and strings joined with newlines per chunk, concatenated across chunks", async () => {
    const first: JsonValue[] = [
      "Part A.",
      { type: "thinking", thinking: "hidden" },
      { type: "text", text: "Part B." },
      { type: "image_url", image_url: { url: "https://example.com/a.png" } },
      5,
      { type: "text", text: 5 },
      null,
    ];
    const events = await parse([
      chunk({ content: first }),
      chunk({ content: [{ type: "text", text: " More." }] }),
      STOP,
    ]);
    expect(events.filter((event) => event instanceof TextDelta)).toEqual([
      new TextDelta(0, "Part A.\nPart B."),
      new TextDelta(0, " More."),
    ]);
    expect(partsDone(events)).toEqual([new PartDone(0, new TextPart("Part A.\nPart B. More."))]);
  });

  test("list content that joins to nothing opens no part", async () => {
    expect(await parse([chunk({ content: [] }), chunk({ content: [{ type: "image_url" }, 5] }), STOP])).toEqual([
      finish("stop"),
    ]);
  });

  test("refusal pieces are collected into one refusal part, with no deltas", async () => {
    const events = await parse([
      chunk({ refusal: "I can't " }),
      chunk({ refusal: "" }),
      chunk({ refusal: "help." }),
      STOP,
    ]);
    expect(events).toEqual([new PartDone(0, new RefusalPart("I can't help.")), finish("stop")]);
    const response = await collect(events);
    expect(response.message.parts).toEqual([new RefusalPart("I can't help.")]);
    expect(response.message.text()).toBe("");
  });

  test("a frame with empty data is skipped", async () => {
    expect(await partsOf([new SSEEvent({ data: "" }), chunk({ content: "Hi" }), STOP])).toEqual([new TextPart("Hi")]);
  });

  test("event data that isn't an object is skipped", async () => {
    expect(await partsOf([[], "text", 5, null, true, chunk({ content: "Hi" }), STOP])).toEqual([new TextPart("Hi")]);
  });

  test("[DONE] ends the stream", async () => {
    const chunks = [chunk({ content: "Hi" }), STOP, new SSEEvent({ data: "[DONE]" }), chunk({ content: " more" })];
    expect(await partsOf(chunks)).toEqual([new TextPart("Hi")]);
  });
});

describe("collected replies", () => {
  /** Return the response a reply from `protocol` collects to. */
  function expected(protocol: OpenAIChat, parts: Part[], model: string, usage = USAGE_COUNTS): Response {
    const providerPayload = replayRecord(keyOf(protocol), { response_id: "chatcmpl-1", response_model: model });
    const message = new Message({ role: "assistant", parts, protocol: PROTOCOL, providerPayload });
    return new Response({ message, usage, finishReason: "stop", responseId: "chatcmpl-1" });
  }

  test("xai plain reply", async () => {
    const msg = { role: "assistant", content: "Hello there." };
    const response = await collect(await parse(streamChunks(msg, { model: "grok-4" })));
    expect(response).toEqual(expected(xai(), [new TextPart("Hello there.")], "grok-4"));
    expect(response.message.text()).toBe("Hello there.");
    expect(response.message.protocol).toBe("openai_chat");
  });

  test("deepseek reasoner reply reads the cache from prompt_cache_hit_tokens", async () => {
    const msg = { role: "assistant", content: "Hello there.", reasoning_content: "Thinking." };
    const chunks = streamChunks(msg, { model: "deepseek-reasoner", usage: USAGE_DEEPSEEK });
    const response = await collect(await parse(chunks, deepseek()));
    const parts = [reasoningPart("Thinking.", "reasoning_content", deepseek()), new TextPart("Hello there.")];
    expect(response).toEqual(expected(deepseek(), parts, "deepseek-reasoner"));
    expect(response.usage!.cacheReadTokens).toBe(5);
  });

  test("minimax reply with reasoning_details", async () => {
    const protocol = route("minimax", "MiniMax-M2.7");
    const details = [{ type: "reasoning.text", id: "r1", format: "MiniMax-response-v1", index: 0, text: "Thinking." }];
    const msg = { role: "assistant", content: "Hello there.", reasoning_details: details };
    const response = await collect(await parse(streamChunks(msg, { model: "MiniMax-M2.7" }), protocol));
    const parts = [reasoningPart("Thinking.", "reasoning_details", protocol), new TextPart("Hello there.")];
    expect(response).toEqual(expected(protocol, parts, "MiniMax-M2.7"));
  });
});

describe("usage and finish", () => {
  test("Finish comes only after the usage chunk that follows the finish chunk", async () => {
    const events = await parse([chunk({ content: "Hi" }), STOP, chunk(null, { usage: USAGE, choices: [] })]);
    expect(events.map((event) => event.type)).toEqual(["TextDelta", "PartDone", "UsageReport", "Finish"]);
  });

  test("no usage chunk gives no usage report", async () => {
    const events = await parse(streamChunks({ content: "Hello there." }, { usageChunk: false }));
    expect(events.some((event) => event instanceof UsageReport)).toBe(false);
    expect((await collect(events)).usage).toBeNull();
  });

  test("usage on the finish chunk", async () => {
    const events = await parse([chunk({ content: "Hi" }), chunk({}, { finishReason: "stop", usage: USAGE })]);
    expect(events.slice(-2)).toEqual([new UsageReport(USAGE_COUNTS), finish("stop")]);
  });

  test("usage on every chunk: the last wins, reported once", async () => {
    const early = { prompt_tokens: 21, completion_tokens: 1, total_tokens: 22 };
    const events = await parse([
      chunk({ content: "Hi" }, { usage: early }),
      chunk({}, { finishReason: "stop", usage: early }),
      chunk(null, { usage: USAGE, choices: [] }),
    ]);
    expect(events.filter((event) => event instanceof UsageReport)).toEqual([new UsageReport(USAGE_COUNTS)]);
  });

  test("a null or non-object usage is ignored, and does not replace an earlier one", async () => {
    for (const later of [null, 5, "usage", [1]]) {
      const events = await parse([
        STOP,
        chunk(null, { usage: USAGE, choices: [] }),
        chunk(null, { usage: later, choices: [] }),
      ]);
      expect(events).toEqual([new UsageReport(USAGE_COUNTS), finish("stop")]);
    }
    expect(await parse([chunk({}, { finishReason: "stop", usage: null })])).toEqual([finish("stop")]);
  });

  test("usage maps cache and reasoning counts; cached_tokens wins over prompt_cache_hit_tokens", async () => {
    const usage = {
      prompt_tokens: 100,
      completion_tokens: 50,
      total_tokens: 150,
      prompt_cache_hit_tokens: 9,
      prompt_tokens_details: { cached_tokens: 40, cache_write_tokens: 7 },
      completion_tokens_details: { reasoning_tokens: 20 },
    };
    const events = await parse([chunk({}, { finishReason: "stop", usage })]);
    const counts = { inputTokens: 100, outputTokens: 50, totalTokens: 150 };
    const details = { cacheReadTokens: 40, cacheCreationTokens: 7, reasoningTokens: 20 };
    expect(events[0]).toEqual(new UsageReport(new Usage({ ...counts, ...details })));
  });

  test("usage defaults: null or missing counts are 0, a missing total is their sum, null details are empty", async () => {
    const cases: [JsonObject, Usage][] = [
      [{ prompt_tokens: 3, completion_tokens: null }, new Usage({ inputTokens: 3, outputTokens: 0, totalTokens: 3 })],
      [{ completion_tokens: 4 }, new Usage({ inputTokens: 0, outputTokens: 4, totalTokens: 4 })],
      [
        { prompt_tokens: 3, completion_tokens: 4, total_tokens: null },
        new Usage({ inputTokens: 3, outputTokens: 4, totalTokens: 7 }),
      ],
      [
        { prompt_tokens: 3, completion_tokens: 4, prompt_tokens_details: null, completion_tokens_details: null },
        new Usage({ inputTokens: 3, outputTokens: 4, totalTokens: 7 }),
      ],
      [{}, new Usage({ inputTokens: 0, outputTokens: 0, totalTokens: 0 })],
    ];
    for (const [usage, counts] of cases) {
      const events = await parse([chunk({}, { finishReason: "stop", usage })]);
      expect(events[0]).toEqual(new UsageReport(counts));
    }
  });

  test("a reply with a service tier still maps its usage", async () => {
    for (const tier of ["priority", "flex", "default"]) {
      const data = { ...chunk(null, { usage: USAGE, choices: [] }), service_tier: tier };
      const events = await parse([{ ...STOP, service_tier: tier }, data]);
      expect(events[0]).toEqual(new UsageReport(USAGE_COUNTS));
    }
  });

  test("the response id is the first non-empty chunk id, the response model the first non-empty model", async () => {
    // Azure's prompt filter chunk comes first, with an empty id and model.
    const filter = { id: "", object: "", created: 0, model: "", choices: [], prompt_filter_results: [] };
    const events = await parse([
      filter,
      chunk({ content: "Hi" }, { id: 5, model: null }),
      chunk({}, { id: "chatcmpl-2", model: "gpt-5.5-2025" }),
      chunk({}, { id: "chatcmpl-3", model: "other", finishReason: "stop" }),
    ]);
    const fields = { response_id: "chatcmpl-2", response_model: "gpt-5.5-2025" };
    expect(events.at(-1)).toEqual(finish("stop", xai(), fields));
  });

  test("a reply without ids or models finishes with an empty record", async () => {
    const events = await parse([without(without(STOP, "id"), "model")]);
    expect(events).toEqual([finish("stop", xai(), {})]);
    expect((events[0] as Finish).responseId).toBeNull();
  });

  test("Finish names the protocol", async () => {
    const events = await parse([STOP]);
    expect((events[0] as Finish).protocol).toBe("openai_chat");
  });
});

describe("finish reasons", () => {
  test.each(["length", "stop", "something_new"])("%s is passed through", async (reason) => {
    expect((await parse([chunk({ content: "Hi" }), chunk({}, { finishReason: reason })])).at(-1)).toEqual(
      finish(reason),
    );
  });

  test.each([
    ["content_filter", "refused", false],
    ["insufficient_system_resource", "overloaded", true],
    ["error", "server", true],
    ["aborted", "server", true],
  ] as const)("%s raises %s", async (reason, kind, retryable) => {
    const [events, error] = await failing([
      chunk({ content: "Hi" }),
      chunk({}, { finishReason: reason }),
      chunk(null, { usage: USAGE, choices: [] }),
    ]);
    expect(error.kind).toBe(kind);
    expect(error.retryable).toBe(retryable);
    expect(error.message).toBe(`xai response did not complete (${reason})`);
    expect(events).toEqual([new TextDelta(0, "Hi")]);
  });
});

describe("incomplete streams", () => {
  test("a stream cut before the finish, [DONE] with no finish, or an empty body is incomplete", async () => {
    const cases: [JsonValue[], boolean][] = [
      [[chunk({ role: "assistant", content: "" }), chunk({ content: "Hi" })], false],
      [[chunk({ content: "Hi" }), chunk(null, { usage: USAGE, choices: [] })], true],
      [[], false],
      [[], true],
    ];
    for (const [chunks, done] of cases) {
      const [events, error] = await failing(chunks, xai(), done);
      expect(error.kind).toBe("incomplete");
      expect(error.retryable).toBe(true);
      expect(error.message).toBe("xai stream ended before a complete response");
      expect(events).toEqual(chunks.length > 0 ? [new TextDelta(0, "Hi")] : []);
    }
  });

  test("a finish without [DONE] is complete", async () => {
    const events = await parse(streamChunks({ content: "Hello there." }), xai(), false);
    expect(events.at(-1)).toEqual(finish("stop"));
  });
});

describe("protocol errors", () => {
  test("choices that aren't a list, a first choice that isn't an object, a finish_reason that isn't a string", async () => {
    const cases: JsonObject[] = [
      chunk(null, { choices: { index: 0 } }),
      chunk(null, { choices: "x" }),
      chunk(null, { choices: ["x"] }),
      chunk(null, { choices: [5] }),
      chunk(null, { choices: [null] }),
      chunk({}, { finishReason: 1 }),
      chunk({}, { finishReason: true }),
    ];
    for (const data of cases) {
      const error = await failure([data, STOP]);
      expect(error.kind).toBe("protocol");
      expect(error.retryable).toBe(false);
      expect(error.message).toBe("invalid choice in xai stream event");
    }
  });

  test("delta fields of the wrong type", async () => {
    const deltas: JsonObject[] = [
      { content: 5 },
      { content: {} },
      { content: true },
      { reasoning_content: 5 },
      { reasoning: {} },
      { reasoning: [] },
      { refusal: 1 },
      { reasoning_details: {} },
      { reasoning_details: "text" },
    ];
    for (const delta of deltas) {
      const error = await failure([chunk(delta), STOP]);
      expect(error.kind).toBe("protocol");
      expect(error.retryable).toBe(false);
      expect(error.message).toBe("invalid delta in xai stream event");
    }
  });

  test("messages name the route's provider", async () => {
    const error = await failure([chunk({ content: 5 })], route("deepseek", "deepseek-chat"));
    expect(error.message).toBe("invalid delta in deepseek stream event");
  });
});

describe("usage errors", () => {
  test.each(["prompt_tokens_details", "completion_tokens_details"])("usage %s that aren't objects", async (field) => {
    for (const details of ["text", [1], 3, true]) {
      const error = await failure([chunk({}, { finishReason: "stop", usage: { ...USAGE, [field]: details } })]);
      expect(error.kind).toBe("protocol");
      expect(error.retryable).toBe(false);
      expect(error.message).toBe("xai response usage details are not objects");
    }
  });

  const counts: [string, (count: JsonValue) => JsonObject][] = [
    ["prompt_tokens", (count) => ({ ...USAGE, prompt_tokens: count })],
    ["completion_tokens", (count) => ({ ...USAGE, completion_tokens: count })],
    ["total_tokens", (count) => ({ ...USAGE, total_tokens: count })],
    ["cached_tokens", (count) => ({ ...USAGE, prompt_tokens_details: { cached_tokens: count } })],
    ["prompt_cache_hit_tokens", (count) => ({ ...USAGE_DEEPSEEK, prompt_cache_hit_tokens: count })],
    ["cache_write_tokens", (count) => ({ ...USAGE, prompt_tokens_details: { cache_write_tokens: count } })],
    ["reasoning_tokens", (count) => ({ ...USAGE, completion_tokens_details: { reasoning_tokens: count } })],
  ];
  test.each(counts)("a usage %s that isn't a safe int", async (_, usage) => {
    // Written as raw JSON so a float token that is an integer (`1.0`, `1e3`) reaches the parser.
    for (const token of ["1.0", "1e3", "1.5", "9007199254740992", "18446744073709551616", '"3"', "true"]) {
      const data = encodeBody(chunk(null, { usage: usage("COUNT"), choices: [] })).replace('"COUNT"', token);
      const error = await failure([STOP, new SSEEvent({ data })]);
      expect(error.kind).toBe("protocol");
      expect(error.retryable).toBe(false);
      expect(error.message).toBe("xai response usage has a count that is not an int");
    }
  });

  test("an invalid usage fails in its own chunk, after the deltas before it", async () => {
    const [events, error] = await failing([
      chunk({ content: "Hi" }),
      chunk({ content: "!" }, { usage: { prompt_tokens: "3" } }),
      STOP,
    ]);
    expect(error.message).toBe("xai response usage has a count that is not an int");
    expect(events).toEqual([new TextDelta(0, "Hi")]);
  });

  test("a missing total whose sum is past 2^53 - 1 is a protocol error", async () => {
    const usage = { prompt_tokens: 2n ** 53n - 1n, completion_tokens: 1n };
    const error = await failure([chunk({}, { finishReason: "stop", usage })]);
    expect(error.kind).toBe("protocol");
    expect(error.message).toBe("xai response usage has a count that is not an int");
  });
});

describe("invalid event JSON", () => {
  const INVALID: [string, string][] = [
    ["not JSON", "{not json test-key-1"],
    ["a lone surrogate escape", '{"choices": [{"delta": {"content": "\\ud800 test-key-1"}}]}'],
    ["a duplicate key", '{"id": "chatcmpl-1", "id": "test-key-1"}'],
  ];

  test.each(INVALID)("%s is a protocol error that doesn't echo the event", async (_, data) => {
    const error = await failure([chunk({ content: "Hi" }), new SSEEvent({ data }), STOP]);
    expect(error.kind).toBe("protocol");
    expect(error.retryable).toBe(false);
    expect(error.message).toBe("invalid JSON in xai stream event");
    expect(error.cause).toBeUndefined();
    expect(String(error.stack)).not.toContain("test-key-1");
  });
});

describe("error chunks", () => {
  /** Return a chunk carrying `error` and nothing else that could fail. */
  function errorChunk(error: JsonValue): JsonObject {
    return { ...chunk({ content: "" }), error };
  }

  const STRING_CODES: [string, ErrorKind, boolean][] = [
    ["rate_limit_exceeded", "rate_limit", true],
    ["insufficient_quota", "quota", false],
    ["context_length_exceeded", "context_length", false],
    ["server_error", "server", true],
    ["invalid_request_error", "bad_request", false],
    ["image_content_policy_violation", "refused", false],
    ["content_filter", "refused", false],
    ["content_policy_violation", "refused", false],
    ["something_new", "server", true],
  ];

  test.each(STRING_CODES)("string code %p raises %s", async (code, kind, retryable) => {
    const [events, error] = await failing([chunk({ content: "Hi" }), errorChunk({ code, message: "It broke." }), STOP]);
    expect(error.kind).toBe(kind);
    expect(error.retryable).toBe(retryable);
    expect(error.message).toBe(`xai stream failed (${kind}): It broke.`);
    expect(events).toEqual([new TextDelta(0, "Hi")]);
  });

  test.each([
    [502, "Bad gateway.", "server", true],
    [529, "Busy.", "overloaded", true],
    [429, "Slow down.", "rate_limit", true],
    [429, "insufficient_quota", "quota", false],
    [400, "This model's maximum context length is 128000 tokens.", "context_length", false],
    [400, "Bad input.", "bad_request", false],
    [401, "Bad key.", "auth", false],
  ] as const)("int code %p with message %p raises %s", async (code, message, kind, retryable) => {
    const error = await failure([errorChunk({ code, message, type: "server_error" })]);
    expect(error.kind).toBe(kind);
    expect(error.retryable).toBe(retryable);
    expect(error.message).toBe(`xai stream failed (${kind}): ${message}`);
  });

  test("codes that aren't strings or ints in 400-599, and no code, are server errors", async () => {
    // Raw JSON, so `502.0` reaches the parser as a float.
    for (const code of ["0", "200", "399", "600", "true", "502.0", "null", "[502]"]) {
      const data = encodeBody(errorChunk({ code: "CODE", message: "It broke." })).replace('"CODE"', code);
      const error = await failure([new SSEEvent({ data })]);
      expect(error.kind).toBe("server");
      expect(error.message).toBe("xai stream failed (server): It broke.");
    }
    expect((await failure([errorChunk({ message: "It broke." })])).kind).toBe("server");
  });

  test("a string type decides when the code isn't a string or an int in 400-599", async () => {
    for (const code of [null, 200, true, 1.5]) {
      const error = await failure([errorChunk({ code, type: "invalid_request_error", message: "Bad input." })]);
      expect(error.kind).toBe("bad_request");
      expect(error.message).toBe("xai stream failed (bad_request): Bad input.");
    }
    expect((await failure([errorChunk({ type: "content_filter" })])).kind).toBe("refused");
    // A string code wins over the type, even an unknown one; an int code in 400-599 too.
    expect((await failure([errorChunk({ code: "something_new", type: "invalid_request_error" })])).kind).toBe("server");
    expect((await failure([errorChunk({ code: 429, type: "invalid_request_error" })])).kind).toBe("rate_limit");
  });

  test("a non-empty string error is a server error with its text", async () => {
    const error = await failure([errorChunk("Upstream went away.")]);
    expect(error.kind).toBe("server");
    expect(error.retryable).toBe(true);
    expect(error.message).toBe("xai stream failed (server): Upstream went away.");
  });

  test("an empty, missing or non-string message is no message", async () => {
    for (const message of ["", null, 3, ["It broke."]]) {
      const error = await failure([errorChunk({ code: "server_error", message })]);
      expect(error.message).toBe("xai stream failed (server): no message");
    }
    const error = await failure([errorChunk({})]);
    expect(error.kind).toBe("server");
    expect(error.message).toBe("xai stream failed (server): no message");
  });

  test("the message is the provider's text uncapped and unscrubbed", async () => {
    for (const message of ["x".repeat(2500), "Bad key sk-test-123 in Authorization: Bearer abc\n  next line"]) {
      const error = await failure([errorChunk({ message })]);
      expect(error.message).toBe(`xai stream failed (server): ${message}`);
    }
  });

  test("OpenRouter's error chunk with finish_reason error: the error object wins", async () => {
    const protocol = route("openrouter", "anthropic/claude-sonnet-4.5");
    const data = {
      id: "gen-1",
      object: "chat.completion.chunk",
      created: 1700000000,
      model: "anthropic/claude-sonnet-4.5",
      provider: "Anthropic",
      error: { code: 502, message: "Provider returned error", metadata: { provider_name: "Anthropic" } },
      choices: [{ index: 0, delta: { content: "" }, finish_reason: "error", native_finish_reason: "error" }],
    };
    const [events, error] = await failing([chunk({ content: "Hi" }), data], protocol);
    expect(error.kind).toBe("server");
    expect(error.message).toBe("openrouter stream failed (server): Provider returned error");
    expect(events).toEqual([new TextDelta(0, "Hi")]);
  });

  test("the error is checked before anything else in the chunk", async () => {
    const data = { ...chunk({ content: 5 }, { usage: "bad", finishReason: 1 }), error: { code: "insufficient_quota" } };
    expect((await failure([data])).kind).toBe("quota");
  });

  test("error values null, false, 0, empty string and lists are ignored", async () => {
    for (const value of [null, false, 0, "", [], ["x"]]) {
      expect(await partsOf([{ ...chunk({ content: "Hi" }), error: value }, STOP])).toEqual([new TextPart("Hi")]);
    }
  });
});

describe("replay records", () => {
  test("a reasoning part's record reads back with its field", async () => {
    const protocol = deepseek();
    const [part] = await partsOf([chunk({ reasoning_content: "Thinking." }), STOP], protocol);
    const payload = part!.providerPayload!;
    expect(payload).toMatchObject({ kind: PROTOCOL, v: 1n, provider: "deepseek", model: "deepseek-reasoner" });
    expect(Object.hasOwn(payload, "account")).toBe(false);
    expect(readReplayRecord(payload, keyOf(protocol), ["field"])).toEqual({
      ok: true,
      fields: { field: "reasoning_content" },
    });
  });

  test("text and refusal parts carry no payload", async () => {
    const parts = await partsOf([chunk({ content: "Hi", refusal: "No." }), STOP]);
    expect(parts.map((part) => part.providerPayload)).toEqual([null, null]);
  });

  test("the Finish record holds the response id and model", async () => {
    const protocol = route("qwen", "qwen3-max");
    const events = await parse(
      [chunk({}, { finishReason: "stop", id: "chatcmpl-9", model: "qwen3-max-2025" })],
      protocol,
    );
    const payload = (events.at(-1) as Finish).providerPayload;
    expect(readReplayRecord(payload, keyOf(protocol), ["response_id", "response_model"])).toEqual({
      ok: true,
      fields: { response_id: "chatcmpl-9", response_model: "qwen3-max-2025" },
    });
  });
});

/** Return a tool-call wire entry; fields given as `undefined` are left out. */
function wire(
  index: JsonValue | undefined,
  id: JsonValue | undefined,
  name: JsonValue | undefined,
  args: JsonValue | undefined,
): JsonObject {
  const fn: JsonObject = {};
  if (name !== undefined) fn["name"] = name;
  if (args !== undefined) fn["arguments"] = args;
  const entry: JsonObject = { type: "function", function: fn };
  if (index !== undefined) entry["index"] = index;
  if (id !== undefined) entry["id"] = id;
  return entry;
}

/** Return a chunk whose delta carries `entries` as its `tool_calls`. */
function calls(...entries: JsonValue[]): JsonObject {
  return chunk({ tool_calls: entries });
}

/** Return the call part the parser builds from `raw` arguments text. */
function callPart(id: string, raw: string, name = "double"): ToolCallPart {
  const [args, rawArguments] = parseArguments(raw);
  return new ToolCallPart({ id, name, arguments: args, rawArguments });
}

function delta(index: number, id: string | null, name: string | null, args: string): ToolCallDelta {
  return new ToolCallDelta({ index, id, name, arguments: args });
}

const HEAD = chunk({ role: "assistant", content: "" });
const CALLS_DONE = chunk({}, { finishReason: "tool_calls" });

describe("tool calls", () => {
  test("one call over several deltas", async () => {
    const events = await parse([
      HEAD,
      calls(wire(0, "call_1", "double", "")),
      calls(wire(0, undefined, undefined, '{"value"')),
      calls(wire(0, undefined, undefined, ":3}")),
      CALLS_DONE,
    ]);
    expect(events).toEqual([
      delta(0, "call_1", "double", ""),
      delta(0, null, null, '{"value"'),
      delta(0, null, null, ":3}"),
      new PartDone(0, callPart("call_1", '{"value":3}')),
      finish("tool_calls"),
    ]);
  });

  test("two parallel calls after text: the text first, then the calls in order", async () => {
    const parts = await partsOf([
      chunk({ content: "Let me." }),
      calls(wire(0, "call_a", "double", '{"value":'), wire(1, "call_b", "double", '{"value":')),
      calls(wire(0, undefined, undefined, "3}")),
      calls(wire(1, undefined, undefined, "4}")),
      CALLS_DONE,
    ]);
    expect(parts).toEqual([
      new TextPart("Let me."),
      callPart("call_a", '{"value":3}'),
      callPart("call_b", '{"value":4}'),
    ]);
  });

  test("an id and name repeated on every delta are kept once, and carried once", async () => {
    const events = await parse([
      calls(wire(0, "call_1", "double", '{"value"')),
      calls(wire(0, "call_1", "double", ":3}")),
      CALLS_DONE,
    ]);
    expect(events).toEqual([
      delta(0, "call_1", "double", '{"value"'),
      delta(0, null, null, ":3}"),
      new PartDone(0, callPart("call_1", '{"value":3}')),
      finish("tool_calls"),
    ]);
  });

  test("a delta without an index joins the call with its id, else the last call, else opens one", async () => {
    const byId = await partsOf([
      calls(wire(0, "call_a", "double", '{"value":'), wire(1, "call_b", "double", '{"value":')),
      calls(wire(undefined, "call_a", undefined, "3}")),
      calls(wire(undefined, undefined, undefined, "4}")),
      CALLS_DONE,
    ]);
    expect(byId).toEqual([callPart("call_a", '{"value":3}'), callPart("call_b", '{"value":4}')]);
    const opened = await parse([
      calls(wire(undefined, undefined, "double", '{"value":')),
      calls(wire(undefined, undefined, undefined, "3}")),
      CALLS_DONE,
    ]);
    expect(opened).toEqual([
      delta(0, null, "double", '{"value":'),
      delta(0, null, null, "3}"),
      new PartDone(0, callPart("yian_call_0", '{"value":3}')),
      finish("tool_calls"),
    ]);
  });

  test("a delta without an index whose id no call has opens a new call", async () => {
    const parts = await partsOf([
      calls(wire(0, "call_a", "double", '{"value":3}')),
      calls(wire(undefined, "call_b", "double", '{"value":4}')),
      CALLS_DONE,
    ]);
    expect(parts).toEqual([callPart("call_a", '{"value":3}'), callPart("call_b", '{"value":4}')]);
  });

  test("a reused index with a new id opens a second call; a later id-less delta there joins it", async () => {
    const events = await parse([
      calls(wire(0, "call_1", "double", '{"value":3}')),
      calls(wire(0, "call_2", "double", '{"value":')),
      calls(wire(0, undefined, undefined, "4}")),
      CALLS_DONE,
    ]);
    expect(events).toEqual([
      delta(0, "call_1", "double", '{"value":3}'),
      delta(1, "call_2", "double", '{"value":'),
      delta(1, null, null, "4}"),
      new PartDone(0, callPart("call_1", '{"value":3}')),
      new PartDone(1, callPart("call_2", '{"value":4}')),
      finish("tool_calls"),
    ]);
  });

  test("an open call without an id takes a later delta's id", async () => {
    const events = await parse([
      calls(wire(0, undefined, "double", '{"value"')),
      calls(wire(0, "call_1", "other", ":3}")),
      CALLS_DONE,
    ]);
    expect(events).toEqual([
      delta(0, null, "double", '{"value"'),
      delta(0, "call_1", null, ":3}"),
      new PartDone(0, callPart("call_1", '{"value":3}')),
      finish("tool_calls"),
    ]);
  });

  test("a tool-call index given as a float token or a string counts as absent", async () => {
    for (const token of ["0.0", '"0"']) {
      const data = encodeBody(calls(wire("INDEX", undefined, undefined, "4}"))).replace('"INDEX"', token);
      const parts = await partsOf([
        calls(wire(0, "call_a", "double", '{"value":'), wire(1, "call_b", "double", '{"value":')),
        // Absent, so it joins the last call opened rather than the call at index 0.
        new SSEEvent({ data }),
        calls(wire(0, undefined, undefined, "3}")),
        CALLS_DONE,
      ]);
      expect(parts).toEqual([callPart("call_a", '{"value":3}'), callPart("call_b", '{"value":4}')]);
    }
  });

  test("a non-string or empty id or name counts as absent", async () => {
    const events = await parse([
      calls(wire(0, 5, 7, "")),
      calls(wire(0, "", "", '{"value":3}')),
      calls(wire(0, "call_1", "double", "")),
      CALLS_DONE,
    ]);
    expect(events.filter((event) => event instanceof ToolCallDelta)).toEqual([
      delta(0, null, null, ""),
      delta(0, null, null, '{"value":3}'),
      delta(0, "call_1", "double", ""),
    ]);
    expect(partsDone(events)).toEqual([new PartDone(0, callPart("call_1", '{"value":3}'))]);
  });

  test("tool calls of the wrong shape are protocol errors", async () => {
    const deltas: JsonObject[] = [
      { tool_calls: { index: 0 } },
      { tool_calls: "call" },
      { tool_calls: ["call"] },
      { tool_calls: [null] },
      { tool_calls: [wire(0, "call_1", "double", "{}"), 5] },
      { tool_calls: [{ index: 0, id: "call_1", function: "double" }] },
      { tool_calls: [{ index: 0, id: "call_1", function: [] }] },
      { tool_calls: [wire(0, "call_1", "double", { value: 3 })] },
      { tool_calls: [wire(0, "call_1", "double", 3)] },
    ];
    for (const each of deltas) {
      const error = await failure([chunk(each), CALLS_DONE]);
      expect(error.kind).toBe("protocol");
      expect(error.retryable).toBe(false);
      expect(error.message).toBe("invalid tool call in xai stream event");
    }
  });

  test("null tool_calls, function and arguments add nothing", async () => {
    const parts = await partsOf([
      chunk({ tool_calls: null }),
      calls(wire(0, "call_1", "double", '{"value":3}')),
      calls(wire(0, undefined, undefined, null)),
      calls({ index: 0, function: null }),
      calls({ index: 0 }),
      CALLS_DONE,
    ]);
    expect(parts).toEqual([callPart("call_1", '{"value":3}')]);
  });

  test("arguments that aren't a JSON object are kept raw, unrepaired; empty arguments are {}", async () => {
    for (const raw of ['{"value":', "null", "3"]) {
      const [part] = await partsOf([calls(wire(0, "call_1", "double", raw)), CALLS_DONE]);
      expect(part).toEqual(new ToolCallPart({ id: "call_1", name: "double", arguments: null, rawArguments: raw }));
    }
    const [part] = await partsOf([calls(wire(0, "call_1", "double", "")), CALLS_DONE]);
    expect(part).toEqual(new ToolCallPart({ id: "call_1", name: "double", arguments: {} }));
  });

  test("an id-less call gets yian_call_<n> on its PartDone only; reasoning and text don't count", async () => {
    const events = await parse([
      chunk({ reasoning_content: "Hm.", content: "Let me." }),
      calls(wire(0, "call_a", "double", "{}"), wire(1, undefined, "double", "{}")),
      CALLS_DONE,
    ]);
    expect(events.filter((event) => event instanceof ToolCallDelta)).toEqual([
      delta(2, "call_a", "double", "{}"),
      delta(3, null, "double", "{}"),
    ]);
    expect(partsDone(events).slice(2)).toEqual([
      new PartDone(2, callPart("call_a", "{}")),
      new PartDone(3, callPart("yian_call_1", "{}")),
    ]);
  });

  test("a generated id differs from every real id in the reply", async () => {
    const once = await partsOf([
      calls(wire(0, undefined, "double", "{}"), wire(1, "yian_call_0", "double", "{}")),
      CALLS_DONE,
    ]);
    expect(once).toEqual([callPart("yian_call_0_1", "{}"), callPart("yian_call_0", "{}")]);
    const twice = await partsOf([
      calls(wire(0, undefined, "double", "{}"), wire(1, "yian_call_0", "double", "{}")),
      calls(wire(2, "yian_call_0_1", "double", "{}")),
      CALLS_DONE,
    ]);
    expect(twice).toEqual([
      callPart("yian_call_0_2", "{}"),
      callPart("yian_call_0", "{}"),
      callPart("yian_call_0_1", "{}"),
    ]);
  });

  test("two real calls sharing an id keep it", async () => {
    const parts = await partsOf([
      calls(wire(0, "call_1", "double", '{"value":3}'), wire(1, "call_1", "double", '{"value":4}')),
      CALLS_DONE,
    ]);
    expect(parts).toEqual([callPart("call_1", '{"value":3}'), callPart("call_1", '{"value":4}')]);
  });

  test("a call with no name fails before any PartDone, ahead of the id rule", async () => {
    const [events, error] = await failing([
      chunk({ content: "Hi" }),
      calls(wire(0, "call_a", "double", "{}"), wire(1, undefined, undefined, "{}"), wire(2, "call_c", "", "{}")),
      CALLS_DONE,
    ]);
    expect(error.kind).toBe("protocol");
    expect(error.retryable).toBe(false);
    expect(error.message).toBe("xai tool call 1 has no name");
    expect(partsDone(events)).toEqual([]);
  });

  test("calls make the finish tool_calls, from stop or function_call; length wins", async () => {
    const call = calls(wire(0, "call_1", "double", "{}"));
    for (const reason of ["stop", "function_call", "tool_calls"]) {
      expect((await parse([call, chunk({}, { finishReason: reason })])).at(-1)).toEqual(finish("tool_calls"));
    }
    expect((await parse([call, chunk({}, { finishReason: "length" })])).at(-1)).toEqual(finish("length"));
  });

  test("a stream cut after an id-less call fails incomplete; its partial reply leaves the call out", async () => {
    const [events, error] = await failing(
      [chunk({ content: "Hi" }), calls(wire(0, undefined, "double", '{"value":3}'))],
      xai(),
      false,
    );
    expect(error.kind).toBe("incomplete");
    expect(error.retryable).toBe(true);
    expect(events).toEqual([new TextDelta(0, "Hi"), delta(1, null, "double", '{"value":3}')]);
    const assembler = new ReplyAssembler();
    for (const event of events) assembler.add(event);
    expect(assembler.partial()!.parts).toEqual([new TextPart("Hi")]);
  });

  test("a content_filter finish with a nameless call is refused", async () => {
    const error = await failure([calls(wire(0, "call_1", undefined, "{}")), chunk({}, { finishReason: "content_filter" })]);
    expect(error.kind).toBe("refused");
  });

  test("a tool-only stream gives exactly a call part", async () => {
    const parts = await partsOf([HEAD, calls(wire(0, "call_1", "double", '{"value":3}')), CALLS_DONE]);
    expect(parts).toEqual([callPart("call_1", '{"value":3}')]);
  });
});

describe("round trips", () => {
  const ASK = new Message({ role: "user", parts: [new TextPart("Double 3.")] });

  /** Return the reply `protocol` parses from `chunks`. */
  async function reply(chunks: JsonValue[], protocol: OpenAIChat): Promise<Message> {
    return (await collect(await parse(chunks, protocol))).message;
  }

  /** Return the messages `protocol` sends for the question and `answer`. */
  function sentBack(answer: Message, protocol: OpenAIChat): JsonValue {
    return protocol.buildBody(new Request({ messages: [ASK, answer] }))["messages"]!;
  }

  const ASKED: JsonObject = { role: "user", content: "Double 3." };

  test("a DeepSeek reasoner reply echoes its reasoning and replays its call with compact arguments", async () => {
    const answer = await reply(
      [
        chunk({ reasoning_content: "Thinking." }),
        calls(wire(0, "call_1", "double", '{"value": 3}')),
        CALLS_DONE,
      ],
      deepseek(),
    );
    const call = { type: "function", id: "call_1", function: { name: "double", arguments: '{"value":3}' } };
    expect(sentBack(answer, deepseek())).toEqual([
      ASKED,
      { role: "assistant", content: null, tool_calls: [call], reasoning_content: "Thinking." },
    ]);
    // The tool result follows as its own message.
    const result = new Message({
      role: "tool",
      parts: [new ToolResultPart({ callId: "call_1", name: "double", content: "6", isError: false })],
    });
    const messages = deepseek().buildBody(new Request({ messages: [ASK, answer, result] }))["messages"];
    expect((messages as JsonValue[]).at(-1)).toEqual({ role: "tool", tool_call_id: "call_1", content: "6" });
  });

  test("a reply whose only reasoning is an empty reasoning_content echoes it empty", async () => {
    const answer = await reply([chunk({ reasoning_content: "" }), chunk({ content: "6" }), STOP], deepseek());
    expect(sentBack(answer, deepseek())).toEqual([ASKED, { role: "assistant", content: "6", reasoning_content: "" }]);
  });

  test("a reply sent to another DeepSeek model leaves the reasoning out and signals other_model once", async () => {
    const answer = await reply(
      [chunk({ reasoning_content: "Think" }), chunk({ reasoning_content: "ing.", content: "6" }), STOP],
      deepseek(),
    );
    const reasons: ReplayDegradedReason[] = [];
    const protocol = new OpenAIChat({
      model: "deepseek-v4-pro",
      provider: "deepseek",
      onReplayDegraded: (reason) => reasons.push(reason),
    });
    expect(sentBack(answer, protocol)).toEqual([ASKED, { role: "assistant", content: "6" }]);
    expect(reasons).toEqual(["other_model"]);
  });

  test("an xai reply's reasoning is never sent back", async () => {
    const answer = await reply([chunk({ reasoning_content: "Thinking.", content: "6" }), STOP], xai());
    expect(sentBack(answer, xai())).toEqual([ASKED, { role: "assistant", content: "6" }]);
  });

  test("an id-less call's generated id is sent back as written", async () => {
    const answer = await reply([calls(wire(0, undefined, "double", '{"value":3}')), CALLS_DONE], xai());
    const call = { type: "function", id: "yian_call_0", function: { name: "double", arguments: '{"value":3}' } };
    expect(sentBack(answer, xai())).toEqual([ASKED, { role: "assistant", content: null, tool_calls: [call] }]);
  });
});
