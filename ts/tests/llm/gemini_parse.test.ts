/**
 * Tests for Gemini reply parsing in `src/llm/protocols/gemini.ts`: SSE chunks into deltas, text
 * and reasoning pieces joined into parts, thought signatures on the part they land in, tool calls
 * and their ids, usage merged across chunks, the finish point and finish reasons, replay records,
 * cancellation, stream errors, failed finish reasons, prompt blocks, truncation, invalid JSON, shape
 * and usage errors, and a reply sent back through `buildBody`.
 */

import { describe, expect, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import { type ErrorKind, ProviderError } from "../../src/llm/errors.ts";
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
import { SSEEvent } from "../../src/llm/framing.ts";
import {
  Message,
  type Part,
  ReasoningPart,
  readReplayRecord,
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
import { Gemini, PROTOCOL } from "../../src/llm/protocols/gemini.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

const FLASH = "gemini-2.5-flash";
const KEY: ReplayKey = { protocol: PROTOCOL, provider: "google", model: FLASH };

// The probe's reply shapes.
const USAGE: JsonObject = {
  promptTokenCount: 12n,
  candidatesTokenCount: 40n,
  thoughtsTokenCount: 25n,
  cachedContentTokenCount: 8n,
  totalTokenCount: 77n,
};
const SIG = "c2lnLW9uZQ==";
const SIG2 = "c2lnLXR3bw==";
const THOUGHT: JsonObject = { text: "Let me think.", thought: true };
const THOUGHT_SIGNED: JsonObject = { text: "Let me think.", thought: true, thoughtSignature: SIG2 };
const TEXT: JsonObject = { text: "Hello there." };
const TEXT_B: JsonObject = { text: "Part B." };
const CALL: JsonObject = { functionCall: { name: "double", args: { value: 3n } }, thoughtSignature: SIG };
const USAGE_COUNTS = new Usage({
  inputTokens: 12,
  outputTokens: 65,
  totalTokens: 77,
  cacheReadTokens: 8,
  reasoningTokens: 25,
});

function gemini(account: string | null = null): Gemini {
  return new Gemini({ model: FLASH, provider: "google", account });
}

interface ReplyOptions {
  usage?: JsonValue;
  finish?: string | null;
  model?: string;
}

/** Return a whole `GenerateContentResponse` holding `parts`, as the probe's `reply_obj` builds it. */
function replyObj(parts: JsonObject[], options: ReplyOptions = {}): JsonObject {
  const candidate: JsonObject = { content: { role: "model", parts }, index: 0n };
  const finish = options.finish === undefined ? "STOP" : options.finish;
  if (finish !== null) candidate["finishReason"] = finish;
  return {
    candidates: [candidate],
    usageMetadata: options.usage ?? USAGE,
    modelVersion: options.model ?? FLASH,
    responseId: "resp-1",
  };
}

/**
 * Return the chunks Gemini streams for `parts`, as the probe's `sse_chunks` builds them: one
 * chunk per part, a text longer than 4 split after its fifth character (the signature stays on
 * the first piece), usage on every chunk, `finishReason` on the last.
 */
function sseChunks(parts: JsonObject[], options: ReplyOptions & { splitText?: boolean } = {}): JsonObject[] {
  const pieces: JsonObject[] = [];
  for (const part of parts) {
    const text = part["text"];
    if (options.splitText !== false && typeof text === "string" && text.length > 4) {
      const { thoughtSignature: _, ...unsigned } = part;
      pieces.push({ ...part, text: text.slice(0, 5) }, { ...unsigned, text: text.slice(5) });
    } else {
      pieces.push(part);
    }
  }
  const finish = options.finish === undefined ? "STOP" : options.finish;
  return pieces.map((piece, i) =>
    replyObj([piece], { ...options, finish: i === pieces.length - 1 ? finish : null }),
  );
}

/** Return a chunk with one candidate holding `parts`, and the chunk's other `fields`. */
function chunk(parts: JsonValue[], fields: JsonObject = {}): JsonObject {
  return { candidates: [{ content: { role: "model", parts }, index: 0n }], ...fields };
}

/** Return a chunk that ends the reply with `reason`. */
function finishChunk(reason = "STOP", fields: JsonObject = {}): JsonObject {
  return { candidates: [{ content: { role: "model", parts: [] }, finishReason: reason, index: 0n }], ...fields };
}

/** Return each chunk as the frame Gemini sends: its JSON as `data`, no event name. */
function sse(items: (JsonObject | SSEEvent)[]): SSEEvent[] {
  return items.map((item) => (item instanceof SSEEvent ? item : new SSEEvent({ data: encodeBody(item) })));
}

async function* frames(items: (JsonObject | SSEEvent)[]): AsyncGenerator<SSEEvent> {
  yield* sse(items);
}

async function parse(items: (JsonObject | SSEEvent)[], protocol = gemini()): Promise<Event[]> {
  const out: Event[] = [];
  for await (const event of protocol.parse(frames(items))) out.push(event);
  return out;
}

/**
 * Return the events parsing `items` emitted and the `ProviderError` it threw, whose provider must
 * be unset; fail the test if none is thrown.
 */
async function failing(items: (JsonObject | SSEEvent)[], protocol = gemini()): Promise<[Event[], ProviderError]> {
  const out: Event[] = [];
  try {
    for await (const event of protocol.parse(frames(items))) out.push(event);
  } catch (error) {
    expect(error).toBeInstanceOf(ProviderError);
    expect((error as ProviderError).provider).toBeNull();
    return [out, error as ProviderError];
  }
  throw new Error("expected a ProviderError");
}

/** Return the `ProviderError` parsing `items` throws; fail the test if none is thrown. */
async function failure(items: (JsonObject | SSEEvent)[], protocol = gemini()): Promise<ProviderError> {
  return (await failing(items, protocol))[1];
}

async function collected(items: (JsonObject | SSEEvent)[], protocol = gemini()): Promise<Response> {
  return collect(await parse(items, protocol));
}

async function partsOf(items: (JsonObject | SSEEvent)[]): Promise<readonly Part[]> {
  return (await collected(items)).message.parts;
}

/** Return the parts of a reply streamed as one chunk per part in `parts`, then a STOP chunk. */
async function partsOfChunks(...parts: JsonValue[]): Promise<readonly Part[]> {
  return partsOf([...parts.map((part) => chunk([part])), finishChunk()]);
}

/** Return the usage a reply of `chunks` reports, `null` for none. */
async function usageOf(chunks: JsonObject[]): Promise<Usage | null> {
  return (await collected(chunks)).usage;
}

function partsDone(items: Event[]): PartDone[] {
  return items.filter((event) => event instanceof PartDone);
}

function record(fields: JsonObject, key: ReplayKey = KEY): JsonObject {
  return replayRecord(key, fields);
}

const FINISH_FIELDS: JsonObject = { response_id: "resp-1", response_model: FLASH };

/** Return the `Finish` the parser gives for the probe's reply. */
function finish(reason: string): Finish {
  return new Finish({ reason, responseId: "resp-1", protocol: PROTOCOL, providerPayload: record(FINISH_FIELDS) });
}

/** Return the response a probe reply with `parts` collects to. */
function expected(parts: Part[], finishReason: string, usage: Usage | null = USAGE_COUNTS): Response {
  const message = new Message({ role: "assistant", parts, protocol: PROTOCOL, providerPayload: record(FINISH_FIELDS) });
  return new Response({ message, usage, finishReason, responseId: "resp-1" });
}

function signedText(text: string, signature: string): TextPart {
  return new TextPart(text, record({ thought_signature: signature }));
}

function signedThought(text: string, signature: string): ReasoningPart {
  return new ReasoningPart(text, record({ thought_signature: signature }));
}

const TEXT_PART = new TextPart("Hello there.");
const CALL_PART = new ToolCallPart({
  id: "yian_call_0",
  name: "double",
  arguments: { value: 3n },
  providerPayload: record({ thought_signature: SIG }),
});
const UNSIGNED_CALL: JsonObject = { functionCall: { name: "double", args: { value: 3n } } };

/** Return a call part as the parser builds an unsigned one. */
function callPart(
  id: string,
  args: JsonObject | null = { value: 3n },
  rawArguments: string | null = null,
): ToolCallPart {
  return new ToolCallPart({ id, name: "double", arguments: args, rawArguments });
}

/** Return a wire `functionCall` part named `double`, with `id` when given. */
function wireCall(id?: string, args: JsonValue = { value: 3n }): JsonObject {
  const call: JsonObject = { name: "double", args };
  if (id !== undefined) call["id"] = id;
  return { functionCall: call };
}

describe("deltas and parts", () => {
  test("a plain reply streams its text, then its PartDone, usage and finish", async () => {
    expect(await parse(sseChunks([TEXT]))).toEqual([
      new TextDelta(0, "Hello"),
      new TextDelta(0, " there."),
      new PartDone(0, TEXT_PART),
      new UsageReport(USAGE_COUNTS),
      finish("stop"),
    ]);
  });

  test("a plain reply collects to its text, usage, finish and response id", async () => {
    const response = await collected(sseChunks([TEXT]));
    expect(response).toEqual(expected([TEXT_PART], "stop"));
    expect(response.message.protocol).toBe("gemini");
  });

  test("a whole reply in one chunk collects the same", async () => {
    expect(await collected([replyObj([TEXT])])).toEqual(expected([TEXT_PART], "stop"));
  });

  test("a tool reply collects to its text and signed call", async () => {
    const response = await collected(sseChunks([TEXT, CALL]));
    expect(response).toEqual(expected([TEXT_PART, CALL_PART], "tool_calls"));
  });

  test("text split over chunks is one part", async () => {
    const chunks = [chunk([{ text: "Hel" }]), chunk([{ text: "lo " }]), chunk([{ text: "there." }]), finishChunk()];
    const out = await parse(chunks);
    expect(out.filter((event) => event instanceof TextDelta)).toEqual([
      new TextDelta(0, "Hel"),
      new TextDelta(0, "lo "),
      new TextDelta(0, "there."),
    ]);
    expect(partsDone(out)).toEqual([new PartDone(0, TEXT_PART)]);
  });

  test("two text parts in one chunk or over chunks are one part, as today's stream text", async () => {
    for (const chunks of [[replyObj([TEXT, TEXT_B])], sseChunks([TEXT, TEXT_B])]) {
      const response = await collected(chunks);
      expect(response.message.parts).toEqual([new TextPart("Hello there.Part B.")]);
      expect(response.message.text()).toBe("Hello there.Part B.");
    }
  });

  test("a thought then text are a reasoning part and a text part, with dense indexes", async () => {
    const out = await parse(sseChunks([THOUGHT, TEXT]));
    expect(out.slice(0, -2)).toEqual([
      new ReasoningDelta(0, "Let m"),
      new ReasoningDelta(0, "e think."),
      new TextDelta(1, "Hello"),
      new TextDelta(1, " there."),
      new PartDone(0, new ReasoningPart("Let me think.", record({}))),
      new PartDone(1, TEXT_PART),
    ]);
  });

  test("a signed thought carries its signature record", async () => {
    expect(await partsOf(sseChunks([THOUGHT_SIGNED, TEXT]))).toEqual([
      signedThought("Let me think.", SIG2),
      TEXT_PART,
    ]);
  });

  test("a signature on the first text piece keeps later pieces in that part (Gemini 2.5)", async () => {
    const wire = [{ text: "Hel", thoughtSignature: SIG }, { text: "lo" }, UNSIGNED_CALL];
    const want = [signedText("Hello", SIG), callPart("yian_call_0")];
    expect(await partsOf([chunk(wire), finishChunk()])).toEqual(want);
    expect(await partsOfChunks(...wire)).toEqual(want);
  });

  test("a part holding only a signature signs the open text part", async () => {
    expect(await partsOfChunks({ text: "Hello" }, { thoughtSignature: SIG })).toEqual([signedText("Hello", SIG)]);
  });

  test("a signature-only thought signs the open reasoning part, or opens an empty one", async () => {
    expect(await partsOfChunks(THOUGHT, { thought: true, thoughtSignature: SIG })).toEqual([
      signedThought("Let me think.", SIG),
    ]);
    expect(await partsOfChunks(TEXT, { thought: true, thoughtSignature: SIG })).toEqual([
      TEXT_PART,
      signedThought("", SIG),
    ]);
  });

  test("Gemini 3's trailing empty signed text signs the open text part, with no extra part", async () => {
    const tail = chunk([{ text: "", thoughtSignature: SIG }]);
    const out = await parse([...sseChunks([TEXT], { finish: null }), tail, finishChunk()]);
    expect(out.filter((event) => event instanceof TextDelta)).toHaveLength(2);
    const response = await collect(out);
    expect(response.message.parts).toEqual([signedText("Hello there.", SIG)]);
    expect(response.message.text()).toBe("Hello there.");
  });

  test("an empty signed piece with no open text part opens an empty text part with the record", async () => {
    expect(await partsOfChunks({ text: "", thoughtSignature: SIG })).toEqual([signedText("", SIG)]);
    expect(await partsOfChunks(UNSIGNED_CALL, { text: "", thoughtSignature: SIG })).toEqual([
      callPart("yian_call_0"),
      signedText("", SIG),
    ]);
  });

  test("a second signature while the open part is signed opens a new part", async () => {
    expect(
      await partsOfChunks({ text: "A.", thoughtSignature: SIG }, { text: "B.", thoughtSignature: SIG2 }),
    ).toEqual([signedText("A.", SIG), signedText("B.", SIG2)]);
    expect(
      await partsOfChunks({ text: "A.", thoughtSignature: SIG }, { text: "", thoughtSignature: SIG2 }),
    ).toEqual([signedText("A.", SIG), signedText("", SIG2)]);
  });

  test("an empty signed text after a thought opens its own empty text part", async () => {
    const tail = chunk([{ text: "", thoughtSignature: SIG }]);
    const out = await parse([chunk([TEXT]), chunk([THOUGHT]), tail, finishChunk()]);
    expect(partsDone(out)).toEqual([
      new PartDone(0, TEXT_PART),
      new PartDone(1, new ReasoningPart("Let me think.", record({}))),
      new PartDone(2, signedText("", SIG)),
    ]);
  });

  test("a call closes the open text, so text after it is a new part", async () => {
    const out = await parse(sseChunks([TEXT, UNSIGNED_CALL, TEXT_B]));
    expect(partsDone(out)).toEqual([
      new PartDone(0, TEXT_PART),
      new PartDone(1, callPart("yian_call_0")),
      new PartDone(2, new TextPart("Part B.")),
    ]);
    expect(out.filter((event) => event instanceof TextDelta).slice(-2)).toEqual([
      new TextDelta(2, "Part "),
      new TextDelta(2, "B."),
    ]);
  });

  test("a call closes the open text, so a thought after it is a new reasoning part", async () => {
    const out = await parse(sseChunks([TEXT, UNSIGNED_CALL, THOUGHT]));
    expect(partsDone(out)).toEqual([
      new PartDone(0, TEXT_PART),
      new PartDone(1, callPart("yian_call_0")),
      new PartDone(2, new ReasoningPart("Let me think.", record({}))),
    ]);
    expect(out.filter((event) => event instanceof ReasoningDelta)).toEqual([
      new ReasoningDelta(2, "Let m"),
      new ReasoningDelta(2, "e think."),
    ]);
  });

  test("an empty unsigned piece opens nothing and emits nothing", async () => {
    expect(await parse([chunk([{ text: "" }]), ...sseChunks([TEXT])])).toEqual(await parse(sseChunks([TEXT])));
    const out = await parse([chunk([{ text: "" }, { text: "", thought: true }]), finishChunk()]);
    expect(out).toEqual([new Finish({ reason: "stop", protocol: PROTOCOL, providerPayload: record({}) })]);
  });

  test("unknown part kinds are skipped", async () => {
    const unknown: JsonObject[] = [
      { inlineData: { mimeType: "image/png", data: "aGk=" } },
      { executableCode: { language: "PYTHON", code: "print(1)" } },
      { codeExecutionResult: { outcome: "OUTCOME_OK", output: "1" } },
    ];
    expect(await partsOfChunks({ text: "A" }, ...unknown, { text: "B" })).toEqual([new TextPart("AB")]);
  });

  test("a signed part of another kind signs the open text part, or opens an empty text part", async () => {
    const signedImage: JsonObject = { inlineData: { mimeType: "image/png", data: "aGk=" }, thoughtSignature: SIG };
    expect(await partsOfChunks({ text: "A" }, signedImage)).toEqual([signedText("A", SIG)]);
    expect(await partsOfChunks(signedImage)).toEqual([signedText("", SIG)]);
    expect(await partsOfChunks(THOUGHT, { ...signedImage, thought: true })).toEqual([
      signedThought("Let me think.", SIG),
    ]);
  });

  test("explicit nulls read as absent", async () => {
    const chunks: JsonObject[] = [
      { candidates: null, promptFeedback: null, usageMetadata: null },
      { candidates: [{ content: null, index: 0n }] },
      { candidates: [{ content: { role: "model", parts: null }, index: 0n }] },
      chunk([{ text: "Hello", thoughtSignature: null }]),
      chunk([{ ...UNSIGNED_CALL, text: null }]),
      finishChunk(),
    ];
    expect(await partsOf(chunks)).toEqual([new TextPart("Hello"), callPart("yian_call_0")]);
  });
});

describe("tool calls", () => {
  test("a call streams one ToolCallDelta with its id, name and encoded args, then its part", async () => {
    const args = { value: 12345678901234567890n, name: "x" };
    const out = await parse([chunk([wireCall("call-srv-1", args)]), finishChunk()]);
    expect(out.slice(0, 2)).toEqual([
      new ToolCallDelta({ index: 0, id: "call-srv-1", name: "double", arguments: encodeBody(args) }),
      new PartDone(0, callPart("call-srv-1", args)),
    ]);
    expect(((out[1] as PartDone).part as ToolCallPart).arguments!["value"]).toBe(12345678901234567890n);
  });

  test("a signed call carries its signature record", async () => {
    expect(await partsOfChunks(CALL)).toEqual([CALL_PART]);
  });

  test("missing args are {}", async () => {
    const out = await parse([chunk([{ functionCall: { name: "double" } }]), finishChunk()]);
    expect(out[0]).toEqual(new ToolCallDelta({ index: 0, id: "yian_call_0", name: "double", arguments: "{}" }));
    expect(partsDone(out)).toEqual([new PartDone(0, callPart("yian_call_0", {}))]);
  });

  test("args that are not an object make an invalid call with the raw args", async () => {
    for (const args of [[1n, 2n], "value", 3n]) {
      const out = await parse([chunk([wireCall(undefined, args)]), finishChunk()]);
      expect(out[0]).toEqual(
        new ToolCallDelta({ index: 0, id: "yian_call_0", name: "double", arguments: encodeBody(args) }),
      );
      expect(partsDone(out)).toEqual([new PartDone(0, callPart("yian_call_0", null, encodeBody(args)))]);
    }
  });

  test("call ids: server ids kept, missing and repeated ids generated by position", async () => {
    const ids = async (...calls: JsonObject[]): Promise<string[]> =>
      (await partsOfChunks(...calls)).map((part) => (part as ToolCallPart).id);
    expect(await ids(wireCall("call-srv-1"))).toEqual(["call-srv-1"]);
    expect(await ids(wireCall())).toEqual(["yian_call_0"]);
    expect(await ids(wireCall(), wireCall())).toEqual(["yian_call_0", "yian_call_1"]);
    expect(await ids(wireCall(), wireCall("yian_call_0"))).toEqual(["yian_call_0", "yian_call_1"]);
    expect(await ids(wireCall("call-srv-1"), wireCall("call-srv-1"))).toEqual(["call-srv-1", "yian_call_1"]);
    expect(await ids(wireCall("yian_call_1"), wireCall())).toEqual(["yian_call_1", "yian_call_1_1"]);
  });

  test("candidates beyond the first are ignored", async () => {
    const second = { content: { role: "model", parts: [TEXT_B, UNSIGNED_CALL] }, finishReason: "SAFETY", index: 1n };
    const chunks = [{ candidates: [{ content: { role: "model", parts: [TEXT] }, index: 0n }, second] }, finishChunk()];
    expect(await partsOf(chunks)).toEqual([TEXT_PART]);
  });
});

describe("usage", () => {
  test("the probe's usage: input, output, total, cache read and reasoning", async () => {
    expect(await usageOf(sseChunks([TEXT]))).toEqual(USAGE_COUNTS);
    expect((await usageOf(sseChunks([TEXT])))!.cacheCreationTokens).toBeNull();
  });

  test("tool-use prompt tokens count as input", async () => {
    const usage = await usageOf([replyObj([TEXT], { usage: { ...USAGE, toolUsePromptTokenCount: 5n } })]);
    expect(usage).toEqual(new Usage({ ...USAGE_COUNTS, inputTokens: 17, totalTokens: 82 }));
  });

  test("a missing totalTokenCount is input + output", async () => {
    const { totalTokenCount: _, ...usage } = USAGE;
    expect(await usageOf([replyObj([TEXT], { usage })])).toEqual(USAGE_COUNTS);
  });

  test("totalTokenCount is not used: the total is input + output", async () => {
    expect(await usageOf([replyObj([TEXT], { usage: { ...USAGE, totalTokenCount: 999n } })])).toEqual(USAGE_COUNTS);
  });

  test("a totalTokenCount alone reports no usage", async () => {
    const out = await parse([replyObj([TEXT], { usage: { totalTokenCount: 5n } })]);
    expect(out.some((event) => event instanceof UsageReport)).toBe(false);
  });

  test("a missing cache count is null and zero thoughts are no reasoning count", async () => {
    const usage = { promptTokenCount: 12n, candidatesTokenCount: 40n, thoughtsTokenCount: 0n, totalTokenCount: 52n };
    expect(await usageOf([replyObj([TEXT], { usage })])).toEqual(
      new Usage({ inputTokens: 12, outputTokens: 40, totalTokens: 52, cacheReadTokens: null, reasoningTokens: null }),
    );
  });

  test("growing counts: the last wins", async () => {
    const chunks = [
      chunk([{ text: "Hel" }], { usageMetadata: { promptTokenCount: 12n, candidatesTokenCount: 1n } }),
      chunk([{ text: "lo" }], { usageMetadata: { promptTokenCount: 12n, candidatesTokenCount: 2n } }),
      finishChunk("STOP", { usageMetadata: { promptTokenCount: 12n, candidatesTokenCount: 3n } }),
    ];
    expect(await usageOf(chunks)).toEqual(new Usage({ inputTokens: 12, outputTokens: 3, totalTokens: 15 }));
  });

  test("counts on the last chunk only are reported", async () => {
    const chunks = [chunk([TEXT]), finishChunk("STOP", { usageMetadata: USAGE })];
    expect(await usageOf(chunks)).toEqual(USAGE_COUNTS);
  });

  test("a trailing count-free usageMetadata doesn't wipe the counts", async () => {
    const chunks = [...sseChunks([TEXT]), { usageMetadata: { trafficType: "ON_DEMAND" } }];
    expect(await usageOf(chunks)).toEqual(USAGE_COUNTS);
  });

  test("a dropping promptTokenCount on the last chunk wins", async () => {
    const chunks = [
      chunk([TEXT], { usageMetadata: { promptTokenCount: 20n, candidatesTokenCount: 1n } }),
      finishChunk("STOP", { usageMetadata: { promptTokenCount: 12n, candidatesTokenCount: 3n } }),
    ];
    expect(await usageOf(chunks)).toEqual(new Usage({ inputTokens: 12, outputTokens: 3, totalTokens: 15 }));
  });

  test("no counts, or all-zero counts, report no usage", async () => {
    const usages: JsonObject[] = [{}, { trafficType: "ON_DEMAND" }, { promptTokenCount: 0n, candidatesTokenCount: 0n }];
    for (const usage of usages) {
      const out = await parse([replyObj([TEXT], { usage })]);
      expect(out.some((event) => event instanceof UsageReport)).toBe(false);
    }
    const out = await parse([chunk([TEXT]), finishChunk()]);
    expect(out.some((event) => event instanceof UsageReport)).toBe(false);
  });
});

describe("finish", () => {
  test("STOP is stop, STOP with a call is tool_calls, MAX_TOKENS is length even with calls", async () => {
    expect((await parse(sseChunks([TEXT]))).at(-1)).toEqual(finish("stop"));
    expect((await parse(sseChunks([TEXT, CALL]))).at(-1)).toEqual(finish("tool_calls"));
    expect((await parse(sseChunks([TEXT], { finish: "MAX_TOKENS" }))).at(-1)).toEqual(finish("length"));
    expect((await parse(sseChunks([TEXT, CALL], { finish: "MAX_TOKENS" }))).at(-1)).toEqual(finish("length"));
  });

  test("chunks after the finishReason chunk are still read", async () => {
    const chunks = [chunk([TEXT]), finishChunk(), { usageMetadata: USAGE }];
    expect(await usageOf(chunks)).toEqual(USAGE_COUNTS);
  });

  test("the last finishReason wins", async () => {
    expect((await parse([chunk([TEXT]), finishChunk("STOP"), finishChunk("MAX_TOKENS")])).at(-1)!).toEqual(
      new Finish({ reason: "length", protocol: PROTOCOL, providerPayload: record({}) }),
    );
    expect((await parse([chunk([TEXT]), finishChunk("MAX_TOKENS"), finishChunk("STOP")])).at(-1)!).toEqual(
      new Finish({ reason: "stop", protocol: PROTOCOL, providerPayload: record({}) }),
    );
  });

  test("a finishMessage goes with its finishReason", async () => {
    const safety = { candidates: [{ finishReason: "SAFETY", finishMessage: "Blocked.", index: 0n }] };
    const error = await failure([chunk([TEXT]), safety, finishChunk("OTHER")]);
    expect(error.message).toBe("google response did not complete (OTHER)");
  });

  test("an empty finishReason is no finish", async () => {
    const error = await failure([chunk([TEXT]), finishChunk("")]);
    expect(error.kind).toBe("incomplete");
    expect(error.retryable).toBe(true);
    expect(error.message).toBe("google stream ended before a complete response");
  });

  test("PartDone, UsageReport and Finish come only at the end of the stream", async () => {
    const log: string[] = [];
    async function* source(): AsyncGenerator<SSEEvent> {
      for (const frame of sse(sseChunks([TEXT]))) {
        log.push("frame");
        yield frame;
      }
      log.push("end");
    }
    for await (const event of gemini().parse(source())) log.push(event.type);
    expect(log).toEqual(["frame", "TextDelta", "frame", "TextDelta", "end", "PartDone", "UsageReport", "Finish"]);
  });
});

describe("replay records", () => {
  test("every signature record reads back for the key, with v 1 and the account when set", async () => {
    for (const account of [null, "acct-1"]) {
      const key: ReplayKey = { ...KEY, account };
      const wire = [THOUGHT_SIGNED, { text: "Hi", thoughtSignature: SIG2 }, CALL];
      const response = await collected(sseChunks(wire), gemini(account));
      const parts = response.message.parts;
      expect(parts).toHaveLength(3);
      for (const [part, signature] of [
        [parts[0]!, SIG2],
        [parts[1]!, SIG2],
        [parts[2]!, SIG],
      ] as const) {
        expect(readReplayRecord(part.providerPayload, key, ["thought_signature"])).toEqual({
          ok: true,
          fields: { thought_signature: signature },
        });
        expect(part.providerPayload!["v"]).toBe(1n);
        expect(part.providerPayload!["account"]).toEqual(account ?? undefined);
      }
    }
  });

  test("an unsigned thought carries its route's record; unsigned text and calls carry none", async () => {
    const parts = await partsOf(sseChunks([THOUGHT, TEXT, UNSIGNED_CALL]));
    expect(parts.map((part) => part.providerPayload)).toEqual([replayRecord(KEY, {}), null, null]);
  });

  test("the Finish record holds the response id and model, the last of each; its protocol is gemini", async () => {
    const chunks = [
      chunk([TEXT], { responseId: "resp-0", modelVersion: "gemini-2.5-flash-001" }),
      finishChunk("STOP", { responseId: "resp-2", modelVersion: "gemini-2.5-flash-002" }),
    ];
    const last = (await parse(chunks)).at(-1) as Finish;
    expect(last.protocol).toBe("gemini");
    expect(last.responseId).toBe("resp-2");
    expect(readReplayRecord(last.providerPayload, KEY, ["response_id", "response_model"])).toEqual({
      ok: true,
      fields: { response_id: "resp-2", response_model: "gemini-2.5-flash-002" },
    });
  });

  test("a reply with no responseId and no modelVersion still has a message record", async () => {
    const last = (await parse([chunk([TEXT]), finishChunk()])).at(-1) as Finish;
    expect(last).toEqual(new Finish({ reason: "stop", protocol: PROTOCOL, providerPayload: record({}) }));
  });
});

describe("cancellation", () => {
  test("a consumer that stops after the first delta closes the frame source", async () => {
    let closed = false;
    let pulled = 0;
    async function* source(): AsyncGenerator<SSEEvent> {
      try {
        for (const frame of sse(sseChunks([TEXT, TEXT_B]))) {
          pulled++;
          yield frame;
        }
      } finally {
        closed = true;
      }
    }
    for await (const event of gemini().parse(source())) {
      expect(event).toEqual(new TextDelta(0, "Hello"));
      break;
    }
    expect(closed).toBe(true);
    expect(pulled).toBe(1);
  });
});

/** The deltas the probe's text reply streams before its last chunk. */
const TEXT_DELTAS = [new TextDelta(0, "Hello"), new TextDelta(0, " there.")];

describe("failed finish reasons", () => {
  const REFUSED = [
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
    "MODEL_ARMOR",
    "PUP_LIMITED_DISABLED",
  ];
  const INCOMPLETE = ["OTHER", "MALFORMED_FUNCTION_CALL", "FINISH_REASON_UNSPECIFIED", "IMAGE_OTHER", "SOMETHING_NEW"];
  const FAILED: [string, ErrorKind, boolean][] = [
    ...REFUSED.map((reason): [string, ErrorKind, boolean] => [reason, "refused", false]),
    ["MALFORMED_RESPONSE", "incomplete", true],
    ...INCOMPLETE.map((reason): [string, ErrorKind, boolean] => [reason, "incomplete", false]),
  ];

  /** Return the probe's text reply ending with `reason`, its last candidate also holding `fields`. */
  function ending(reason: string, fields: JsonObject = {}): JsonObject[] {
    const last: JsonObject = { content: { role: "model", parts: [] }, finishReason: reason, index: 0n, ...fields };
    return [...sseChunks([TEXT], { finish: null }), { candidates: [last], usageMetadata: USAGE }];
  }

  test.each(FAILED)("%s raises %s (retryable %p) after its deltas", async (reason, kind, retryable) => {
    const message = `google response did not complete (${reason})`;
    for (const [finishMessage, text] of [
      [undefined, message],
      ["The reply was stopped.", `${message}: The reply was stopped.`],
      ["", message],
      [null, message],
    ] as [JsonValue | undefined, string][]) {
      const [out, error] = await failing(ending(reason, finishMessage === undefined ? {} : { finishMessage }));
      expect(error.kind).toBe(kind);
      expect(error.retryable).toBe(retryable);
      expect(error.message).toBe(text);
      expect(out).toEqual(TEXT_DELTAS);
    }
  });

  test("CONTINUATION is a protocol error that names the request yian does not send", async () => {
    const [out, error] = await failing(
      ending("CONTINUATION", { finishMessage: "Continue.", continuationToken: "dG9rZW4=" }),
    );
    expect(error.kind).toBe("protocol");
    expect(error.retryable).toBe(false);
    expect(error.message).toBe("google asked to continue the reply in another request, which yian does not send");
    expect(out).toEqual(TEXT_DELTAS);
  });
});

describe("prompt blocks", () => {
  test.each(["SAFETY", "OTHER"])("a %s block with no candidates is refused", async (reason) => {
    for (const [blockReasonMessage, text] of [
      [undefined, `google blocked the prompt (${reason})`],
      ["The prompt was blocked.", `google blocked the prompt (${reason}): The prompt was blocked.`],
      ["", `google blocked the prompt (${reason})`],
    ] as [string | undefined, string][]) {
      const feedback: JsonObject = { blockReason: reason };
      if (blockReasonMessage !== undefined) feedback["blockReasonMessage"] = blockReasonMessage;
      const [out, error] = await failing([{ promptFeedback: feedback, usageMetadata: { promptTokenCount: 7n } }]);
      expect(error.kind).toBe("refused");
      expect(error.retryable).toBe(false);
      expect(error.message).toBe(text);
      expect(out).toEqual([]);
    }
  });
});

describe("stream errors", () => {
  const UNAVAILABLE = "The service is currently unavailable.";
  const API_KEY_INVALID: JsonObject = {
    code: 400n,
    message: "API key not valid. Please pass a valid API key.",
    status: "INVALID_ARGUMENT",
    details: [
      { "@type": "type.googleapis.com/google.rpc.ErrorInfo", reason: "API_KEY_INVALID", domain: "googleapis.com" },
    ],
  };
  const TOO_LONG = "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).";
  const ERRORS: [string, JsonObject, ErrorKind, boolean, string][] = [
    ["503 unavailable", { code: 503n, message: UNAVAILABLE, status: "UNAVAILABLE" }, "server", true, UNAVAILABLE],
    [
      "503 overloaded",
      { code: 503n, message: "The model is overloaded. Please try again later.", status: "UNAVAILABLE" },
      "overloaded",
      true,
      "The model is overloaded. Please try again later.",
    ],
    [
      "429",
      { code: 429n, message: "Quota exceeded.", status: "RESOURCE_EXHAUSTED" },
      "rate_limit",
      true,
      "Quota exceeded.",
    ],
    ["400 API_KEY_INVALID", API_KEY_INVALID, "auth", false, "API key not valid. Please pass a valid API key."],
    ["400 context", { code: 400n, message: TOO_LONG, status: "INVALID_ARGUMENT" }, "context_length", false, TOO_LONG],
    ["no code", { message: "It broke." }, "server", true, "It broke."],
    ["a string code", { code: "503", message: "It broke." }, "server", true, "It broke."],
    ["code 0", { code: 0n, message: "It broke." }, "server", true, "It broke."],
    ["code 200", { code: 200n, message: "It broke." }, "server", true, "It broke."],
    ["code 302", { code: 302n, message: "It broke." }, "server", true, "It broke."],
    ["code 600", { code: 600n, message: "It broke." }, "server", true, "It broke."],
    ["a float code", { code: 503.5, message: "It broke." }, "server", true, "It broke."],
  ];

  test.each(ERRORS)("an error chunk (%s) after text raises its kind", async (_, body, kind, retryable, text) => {
    const [out, error] = await failing([chunk([TEXT]), { error: body }, finishChunk()]);
    expect(error.kind).toBe(kind);
    expect(error.retryable).toBe(retryable);
    expect(error.message).toBe(`google stream failed (${kind}): ${text}`);
    expect(out).toEqual([new TextDelta(0, "Hello there.")]);
  });

  test("the message falls back to the status, then to no message", async () => {
    const cases: [JsonObject, string][] = [
      [{ code: 503n, status: "UNAVAILABLE" }, "UNAVAILABLE"],
      [{ code: 503n, message: "", status: "UNAVAILABLE" }, "UNAVAILABLE"],
      [{ code: 503n, message: 5n, status: "UNAVAILABLE" }, "UNAVAILABLE"],
      [{ code: 503n }, "no message"],
      [{ code: 503n, message: null, status: "" }, "no message"],
      [{}, "no message"],
    ];
    for (const [body, text] of cases) {
      expect((await failure([{ error: body }])).message).toBe(`google stream failed (server): ${text}`);
    }
  });

  test("an error chunk as the first chunk raises the same way", async () => {
    for (const [, body, kind, retryable, text] of ERRORS) {
      const [out, error] = await failing([{ error: body }, ...sseChunks([TEXT])]);
      const message = `google stream failed (${kind}): ${text}`;
      expect([error.kind, error.retryable, error.message]).toEqual([kind, retryable, message]);
      expect(out).toEqual([]);
    }
  });

  test("an error chunk after the finishReason chunk raises, with no PartDone, UsageReport or Finish", async () => {
    const [out, error] = await failing([...sseChunks([TEXT]), { error: { code: 503n, message: UNAVAILABLE } }]);
    expect(error.kind).toBe("server");
    expect(error.message).toBe(`google stream failed (server): ${UNAVAILABLE}`);
    expect(out).toEqual(TEXT_DELTAS);
  });

  test("the message is the provider's text uncapped", async () => {
    const message = "x".repeat(2500);
    const error = await failure([{ error: { code: 500n, message } }]);
    expect(error.message).toBe(`google stream failed (server): ${message}`);
  });

  test("an error that isn't an object is not a stream error", async () => {
    expect(await parse([{ error: "It broke." }, ...sseChunks([TEXT])])).toEqual(await parse(sseChunks([TEXT])));
  });
});

describe("truncation", () => {
  const TRUNCATED = "google stream ended before a complete response";

  test("text chunks with no finishReason are incomplete and retryable", async () => {
    const [out, error] = await failing(sseChunks([TEXT], { finish: null }));
    expect([error.kind, error.retryable, error.message]).toEqual(["incomplete", true, TRUNCATED]);
    expect(out).toEqual(TEXT_DELTAS);
  });

  test("usage-only chunks with no finishReason are incomplete", async () => {
    const [out, error] = await failing([{ usageMetadata: USAGE }, { usageMetadata: USAGE, responseId: "resp-1" }]);
    expect([error.kind, error.retryable, error.message]).toEqual(["incomplete", true, TRUNCATED]);
    expect(out).toEqual([]);
  });

  test("an empty stream is incomplete", async () => {
    const [out, error] = await failing([]);
    expect([error.kind, error.retryable, error.message]).toEqual(["incomplete", true, TRUNCATED]);
    expect(out).toEqual([]);
  });
});

describe("invalid event JSON", () => {
  const INVALID: [string, string][] = [
    ["not JSON", "{not json test-key-1"],
    ["a lone surrogate escape", '{"candidates": [{"content": {"parts": [{"text": "\\ud800 test-key-1"}]}}]}'],
    ["a duplicate key", '{"responseId": "resp-1", "responseId": "test-key-1"}'],
  ];

  test.each(INVALID)("%s is a protocol error that doesn't echo the event", async (_, data) => {
    const [out, error] = await failing([chunk([TEXT]), new SSEEvent({ data }), finishChunk()]);
    expect(error.kind).toBe("protocol");
    expect(error.retryable).toBe(false);
    expect(error.message).toBe("invalid JSON in google stream event");
    expect(error.cause).toBeUndefined();
    expect(error.causeName).toBeNull();
    expect(String(error.stack)).not.toContain("test-key-1");
    expect(out).toEqual([new TextDelta(0, "Hello there.")]);
  });
});

describe("shape errors", () => {
  /** Assert the chunk `data` (after a text chunk) raises a protocol error naming `field`. */
  async function invalid(field: string, data: JsonObject): Promise<void> {
    const [out, error] = await failing([chunk([TEXT]), data, finishChunk()]);
    expect(error.kind).toBe("protocol");
    expect(error.retryable).toBe(false);
    expect(error.message).toBe(`invalid ${field} in google stream event`);
    expect(out).toEqual([new TextDelta(0, "Hello there.")]);
  }

  test("candidates that aren't a list", async () => {
    for (const candidates of [{}, "x", 1n, true] as JsonValue[]) await invalid("candidates", { candidates });
  });

  test("a candidate that isn't an object", async () => {
    for (const candidate of ["x", 1n, [], true] as JsonValue[]) await invalid("candidate", { candidates: [candidate] });
  });

  test("content that isn't an object, and parts that aren't a list", async () => {
    for (const content of ["x", 1n, [], true] as JsonValue[]) {
      await invalid("content", { candidates: [{ content, index: 0n }] });
    }
    for (const parts of [{}, "x", 1n, true] as JsonValue[]) {
      await invalid("parts", { candidates: [{ content: { role: "model", parts }, index: 0n }] });
    }
  });

  test("a part that isn't an object, or whose text or thoughtSignature isn't a string", async () => {
    for (const part of ["x", 1n, [], true] as JsonValue[]) await invalid("part", chunk([part]));
    for (const value of [5n, true, ["x"], {}] as JsonValue[]) {
      await invalid("part", chunk([{ text: value }]));
      await invalid("part", chunk([{ text: "x", thoughtSignature: value }]));
      await invalid("part", chunk([{ ...UNSIGNED_CALL, thoughtSignature: value }]));
    }
  });

  test("promptFeedback that isn't an object", async () => {
    for (const promptFeedback of ["SAFETY", 1n, [], true] as JsonValue[]) {
      await invalid("promptFeedback", { promptFeedback });
    }
  });

  test("a functionCall that isn't an object, or whose id or name can't be read", async () => {
    for (const call of ["double", 1n, [], true] as JsonValue[]) {
      await invalid("functionCall", chunk([{ functionCall: call }]));
    }
    const calls: JsonObject[] = [
      { id: 5n, name: "double", args: {} },
      { id: ["call-1"], name: "double" },
      { args: {} },
      { name: "", args: {} },
      { name: null, args: {} },
      { name: 5n, args: {} },
    ];
    for (const call of calls) await invalid("functionCall", chunk([{ functionCall: call }]));
  });
});

describe("usage errors", () => {
  const NOT_INT = "google response usage has a count that is not an int";

  test("usageMetadata that isn't an object", async () => {
    for (const usage of ["x", [1n], 3n, true] as JsonValue[]) {
      const [out, error] = await failing([chunk([TEXT]), { usageMetadata: usage }, finishChunk()]);
      expect(error.kind).toBe("protocol");
      expect(error.retryable).toBe(false);
      expect(error.message).toBe("google response usage is not an object");
      expect(out).toEqual([new TextDelta(0, "Hello there.")]);
    }
  });

  const FIELDS = [
    "promptTokenCount",
    "toolUsePromptTokenCount",
    "candidatesTokenCount",
    "thoughtsTokenCount",
    "cachedContentTokenCount",
    "totalTokenCount",
  ];
  test.each(FIELDS)("a %s that isn't a safe int", async (field) => {
    // Raw JSON, so a float count that is an integer (`1.0`, `1e3`) reaches the parser.
    for (const token of ["1.0", "1e3", "1.5", "9007199254740992", "18446744073709551616", '"3"', "true", "-1.5"]) {
      const data = encodeBody({ usageMetadata: { [field]: "COUNT" } }).replace('"COUNT"', token);
      const [out, error] = await failing([chunk([TEXT]), new SSEEvent({ data }), finishChunk()]);
      expect(error.kind).toBe("protocol");
      expect(error.retryable).toBe(false);
      expect(error.message).toBe(NOT_INT);
      expect(out).toEqual([new TextDelta(0, "Hello there.")]);
    }
  });

  test("a bad count fails on its own chunk, even when a later chunk replaces it", async () => {
    const data = encodeBody({ usageMetadata: { promptTokenCount: "COUNT" } }).replace('"COUNT"', "1.0");
    const error = await failure([new SSEEvent({ data }), replyObj([TEXT])]);
    expect([error.kind, error.message]).toEqual(["protocol", NOT_INT]);
  });

  test("a sum past 2^53 - 1 is a protocol error before any part is done", async () => {
    const usage = { promptTokenCount: 2n ** 53n - 1n, candidatesTokenCount: 1n };
    const [out, error] = await failing([chunk([TEXT]), finishChunk("STOP", { usageMetadata: usage })]);
    expect([error.kind, error.message]).toEqual(["protocol", NOT_INT]);
    expect(out).toEqual([new TextDelta(0, "Hello there.")]);
  });
});

describe("round trips", () => {
  const ASK = new Message({ role: "user", parts: [new TextPart("Double 3.")] });
  const RESULT = new Message({
    role: "tool",
    parts: [new ToolResultPart({ callId: "yian_call_0", name: "double", content: "6", isError: false })],
  });
  const SENT_ASK: JsonObject = { role: "user", parts: [{ text: "Double 3." }] };
  const SENT_RESULT: JsonObject = {
    role: "user",
    parts: [{ functionResponse: { id: "yian_call_0", name: "double", response: { output: 6n } } }],
  };
  const SENT_CALL: JsonObject = { id: "yian_call_0", name: "double", args: { value: 3n } };

  /** Return a protocol for `model`, pushing each degraded reason to `reasons`. */
  function protocolFor(model: string, reasons: ReplayDegradedReason[] = []): Gemini {
    return new Gemini({ model, provider: "google", onReplayDegraded: (reason) => reasons.push(reason) });
  }

  /** Return the contents `protocol` sends for `messages`. */
  function sentBy(protocol: Gemini, messages: Message[]): JsonValue {
    return protocol.buildBody(new Request({ messages }))["contents"]!;
  }

  test("a [THOUGHT_SIGNED, TEXT, CALL] reply goes back with its call signature and the result's id", async () => {
    const protocol = protocolFor(FLASH);
    const reply = (await collected(sseChunks([THOUGHT_SIGNED, TEXT, CALL]), protocol)).message;
    const reasons: ReplayDegradedReason[] = [];
    expect(sentBy(protocolFor(FLASH, reasons), [ASK, reply, RESULT])).toEqual([
      SENT_ASK,
      { role: "model", parts: [{ text: "Hello there." }, { functionCall: SENT_CALL, thoughtSignature: SIG }] },
      SENT_RESULT,
    ]);
    expect(reasons).toEqual([]);
  });

  test("a gemini-3-pro-preview reply sent to gemini-2.5-flash drops its signature for the placeholder", async () => {
    const reply = (await collected(sseChunks([THOUGHT_SIGNED, TEXT, CALL]), protocolFor("gemini-3-pro-preview")))
      .message;
    const reasons: ReplayDegradedReason[] = [];
    expect(sentBy(protocolFor(FLASH, reasons), [ASK, reply, RESULT])).toEqual([
      SENT_ASK,
      {
        role: "model",
        parts: [
          { text: "Hello there." },
          { functionCall: SENT_CALL, thoughtSignature: "skip_thought_signature_validator" },
        ],
      },
      SENT_RESULT,
    ]);
    expect(reasons).toEqual(["other_model"]);
  });

  test("a reply with no response id or model and an unsigned call goes back with no placeholder", async () => {
    const protocol = protocolFor(FLASH);
    const reply = (await collected([chunk([TEXT, UNSIGNED_CALL]), finishChunk()], protocol)).message;
    expect(sentBy(protocol, [ASK, reply, RESULT])).toEqual([
      SENT_ASK,
      { role: "model", parts: [{ text: "Hello there." }, { functionCall: SENT_CALL }] },
      SENT_RESULT,
    ]);
  });
});
