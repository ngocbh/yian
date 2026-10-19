/**
 * Tests for Anthropic Messages reply parsing in `src/llm/protocols/anthropic_messages.ts`: SSE
 * events into deltas, parts with replay records, dense indexes, tool-call arguments, usage, the
 * finish point and finish reasons, stream errors, failed stop reasons, truncation, shape errors,
 * and parsed replies sent back.
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
import { AnthropicMessages, PROTOCOL } from "../../src/llm/protocols/anthropic_messages.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

const SONNET = "claude-sonnet-4-5";
const KEY: ReplayKey = { protocol: PROTOCOL, provider: "anthropic", model: SONNET };

// Realistic Messages reply shapes.
const A_USAGE: JsonObject = {
  input_tokens: 12n,
  output_tokens: 40n,
  cache_read_input_tokens: 100n,
  cache_creation_input_tokens: 30n,
};
const THINKING: JsonObject = { type: "thinking", thinking: "Let me think.", signature: "SIG-1" };
const REDACTED: JsonObject = { type: "redacted_thinking", data: "REDACTED-DATA-1" };
const TEXT: JsonObject = { type: "text", text: "Hello there." };
const TOOL_USE: JsonObject = { type: "tool_use", id: "toolu_01", name: "double", input: { value: 3n } };
const USAGE_COUNTS = new Usage({
  inputTokens: 142,
  outputTokens: 40,
  totalTokens: 182,
  cacheReadTokens: 100,
  cacheCreationTokens: 30,
});

function route(account: string | null = null): AnthropicMessages {
  return new AnthropicMessages({ model: SONNET, provider: "anthropic", account });
}

interface MessageOptions {
  usage?: JsonObject;
  stopReason?: JsonValue;
  model?: string;
}

/** Return a Messages reply object with `content`, as the probe's `message_obj` builds it. */
function messageObj(content: JsonObject[], options: MessageOptions = {}): JsonObject {
  return {
    id: "msg_1",
    type: "message",
    role: "assistant",
    model: options.model ?? SONNET,
    content,
    stop_reason: options.stopReason === undefined ? "end_turn" : options.stopReason,
    stop_sequence: null,
    usage: options.usage ?? A_USAGE,
  };
}

interface EventOptions {
  dropStop?: boolean;
  deltaUsage?: JsonValue;
}

/** Return the SSE events the API streams for `message`, as the probe's `a_sse_events` builds them. */
function events(message: JsonObject, options: EventOptions = {}): JsonObject[] {
  const usage = message["usage"] as JsonObject;
  const startUsage: JsonObject = { input_tokens: usage["input_tokens"] ?? 0n, output_tokens: 1n };
  for (const [key, value] of Object.entries(usage)) if (key.startsWith("cache")) startUsage[key] = value;
  const out: JsonObject[] = [
    { type: "message_start", message: { ...message, content: [], stop_reason: null, usage: startUsage } },
  ];
  for (const [index, block] of (message["content"] as JsonObject[]).entries()) {
    const i = BigInt(index);
    const type = block["type"];
    if (type === "text") {
      out.push(blockStart(i, { type: "text", text: "" }));
      const text = block["text"] as string;
      for (const piece of [text.slice(0, 5), text.slice(5)]) {
        out.push(blockDelta(i, { type: "text_delta", text: piece }));
      }
    } else if (type === "thinking") {
      out.push(blockStart(i, { type: "thinking", thinking: "", signature: "" }));
      out.push(blockDelta(i, { type: "thinking_delta", thinking: block["thinking"]! }));
      out.push(blockDelta(i, { type: "signature_delta", signature: block["signature"]! }));
    } else if (type === "redacted_thinking") {
      out.push(blockStart(i, block));
    } else if (type === "tool_use") {
      out.push(blockStart(i, { ...block, input: {} }));
      const args = encodeBody(block["input"]);
      const half = args.length >> 1;
      for (const piece of [args.slice(0, half), args.slice(half)]) {
        out.push(blockDelta(i, { type: "input_json_delta", partial_json: piece }));
      }
    }
    out.push(blockStop(i));
  }
  const deltaUsage = options.deltaUsage === undefined ? usage : options.deltaUsage;
  out.push({
    type: "message_delta",
    delta: { stop_reason: message["stop_reason"]!, stop_sequence: null },
    usage: deltaUsage,
  });
  if (!options.dropStop) out.push({ type: "message_stop" });
  return out;
}

function blockStart(index: bigint, block: JsonObject): JsonObject {
  return { type: "content_block_start", index, content_block: block };
}

function blockDelta(index: bigint, delta: JsonObject): JsonObject {
  return { type: "content_block_delta", index, delta };
}

function blockStop(index: bigint): JsonObject {
  return { type: "content_block_stop", index };
}

/** Return a `message_start` event with the probe's message and `usage`. */
function start(usage: JsonValue = { input_tokens: 12n, output_tokens: 1n }): JsonObject {
  return { type: "message_start", message: { ...messageObj([]), stop_reason: null, usage } };
}

/** Return a `message_delta` event with `stopReason` and, when given, `usage`. */
function messageDelta(stopReason: JsonValue = "end_turn", usage?: JsonValue): JsonObject {
  const data: JsonObject = { type: "message_delta", delta: { stop_reason: stopReason, stop_sequence: null } };
  if (usage !== undefined) data["usage"] = usage;
  return data;
}

const STOP: JsonObject = { type: "message_stop" };

/** Wrap a reply's body events in `message_start`, `message_delta` and `message_stop`. */
function reply(body: JsonObject[], stopReason: JsonValue = "end_turn"): JsonObject[] {
  return [start(), ...body, messageDelta(stopReason), STOP];
}

/** Return each event as the frame the API sends: its type as `event`, its JSON as `data`. */
function sse(items: (JsonObject | SSEEvent)[]): SSEEvent[] {
  return items.map((item) => {
    if (item instanceof SSEEvent) return item;
    const type = item["type"];
    return new SSEEvent({ data: encodeBody(item), event: typeof type === "string" ? type : null });
  });
}

async function* frames(items: (JsonObject | SSEEvent)[]): AsyncGenerator<SSEEvent> {
  yield* sse(items);
}

async function parse(items: (JsonObject | SSEEvent)[], protocol = route()): Promise<Event[]> {
  const out: Event[] = [];
  for await (const event of protocol.parse(frames(items))) out.push(event);
  return out;
}

/** Return the events parsed before the failure and the `ProviderError`; fail the test if none is thrown. */
async function failing(items: (JsonObject | SSEEvent)[], protocol = route()): Promise<[Event[], ProviderError]> {
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

async function failure(items: (JsonObject | SSEEvent)[], protocol = route()): Promise<ProviderError> {
  return (await failing(items, protocol))[1];
}

async function collected(items: (JsonObject | SSEEvent)[], protocol = route()): Promise<Response> {
  return collect(await parse(items, protocol));
}

async function partsOf(items: (JsonObject | SSEEvent)[]): Promise<readonly Part[]> {
  return (await collected(items)).message.parts;
}

function partsDone(items: Event[]): PartDone[] {
  return items.filter((event) => event instanceof PartDone);
}

const FINISH_FIELDS: JsonObject = { response_id: "msg_1", response_model: SONNET };

/** Return the `Finish` the parser gives for the probe's reply. */
function finish(reason: string): Finish {
  const providerPayload = replayRecord(KEY, FINISH_FIELDS);
  return new Finish({ reason, responseId: "msg_1", protocol: PROTOCOL, providerPayload });
}

// What `reply` ends with: the usage of `start()` and the finish.
const END: Event[] = [
  new UsageReport(new Usage({ inputTokens: 12, outputTokens: 1, totalTokens: 13 })),
  finish("stop"),
];

/** Return the response a reply with `parts` collects to. */
function expected(parts: Part[], finishReason: string, usage: Usage | null = USAGE_COUNTS): Response {
  const message = new Message({
    role: "assistant",
    parts,
    protocol: PROTOCOL,
    providerPayload: replayRecord(KEY, FINISH_FIELDS),
  });
  return new Response({ message, usage, finishReason, responseId: "msg_1" });
}

function thought(text: string, signature: string): ReasoningPart {
  return new ReasoningPart(text, replayRecord(KEY, { signature }));
}

function redactedPart(data: string): ReasoningPart {
  return new ReasoningPart("", replayRecord(KEY, { redacted: data }));
}

const TEXT_PART = new TextPart("Hello there.");
const CALL_PART = new ToolCallPart({ id: "toolu_01", name: "double", arguments: { value: 3n } });

/** Return the one tool call a reply with a `tool_use` start of `input` and `deltas` gives. */
async function callOf(input: JsonValue | undefined, deltas: string[] = []): Promise<ToolCallPart> {
  const block: JsonObject = { type: "tool_use", id: "toolu_01", name: "double" };
  if (input !== undefined) block["input"] = input;
  const body = [blockStart(0n, block)];
  for (const piece of deltas) body.push(blockDelta(0n, { type: "input_json_delta", partial_json: piece }));
  body.push(blockStop(0n));
  const [part] = await partsOf(reply(body, "tool_use"));
  return part as ToolCallPart;
}

describe("deltas and parts", () => {
  test("a plain reply streams its text, then its PartDone, usage and finish", async () => {
    expect(await parse(events(messageObj([TEXT])))).toEqual([
      new TextDelta(0, "Hello"),
      new TextDelta(0, " there."),
      new PartDone(0, TEXT_PART),
      new UsageReport(USAGE_COUNTS),
      finish("stop"),
    ]);
  });

  test("a plain reply collects to its text, usage, finish and response id", async () => {
    const response = await collected(events(messageObj([TEXT])));
    expect(response).toEqual(expected([TEXT_PART], "stop"));
    expect(response.message.protocol).toBe("anthropic_messages");
  });

  test("several text blocks are separate parts, joined with newlines by text()", async () => {
    const response = await collected(events(messageObj([TEXT, { type: "text", text: "Second." }])));
    expect(response.message.parts).toEqual([TEXT_PART, new TextPart("Second.")]);
    expect(response.message.text()).toBe("Hello there.\nSecond.");
  });

  test("a tool reply collects to its text and call", async () => {
    const response = await collected(events(messageObj([TEXT, TOOL_USE], { stopReason: "tool_use" })));
    expect(response).toEqual(expected([TEXT_PART, CALL_PART], "tool_calls"));
  });

  test("thinking, redacted thinking, text and a call collect with their records", async () => {
    const response = await collected(
      events(messageObj([THINKING, REDACTED, TEXT, TOOL_USE], { stopReason: "tool_use" })),
    );
    expect(response).toEqual(
      expected(
        [thought("Let me think.", "SIG-1"), redactedPart("REDACTED-DATA-1"), TEXT_PART, CALL_PART],
        "tool_calls",
      ),
    );
  });

  test("thinking, redacted, text and tool_use at wire indexes 0-3 take part indexes 0-3", async () => {
    const out = await parse(events(messageObj([THINKING, REDACTED, TEXT, TOOL_USE], { stopReason: "tool_use" })));
    expect(out.slice(0, -2)).toEqual([
      new ReasoningDelta(0, "Let me think."),
      new PartDone(0, thought("Let me think.", "SIG-1")),
      new PartDone(1, redactedPart("REDACTED-DATA-1")),
      new TextDelta(2, "Hello"),
      new TextDelta(2, " there."),
      new PartDone(2, TEXT_PART),
      new ToolCallDelta({ index: 3, id: "toolu_01", name: "double", arguments: "" }),
      new ToolCallDelta({ index: 3, id: null, name: null, arguments: '{"val' }),
      new ToolCallDelta({ index: 3, id: null, name: null, arguments: 'ue":3}' }),
      new PartDone(3, CALL_PART),
    ]);
  });

  test("an empty text block between opens nothing and later parts stay dense", async () => {
    const content = [THINKING, { type: "text", text: "" }, TEXT, TOOL_USE];
    const out = await parse(events(messageObj(content, { stopReason: "tool_use" })));
    expect(partsDone(out)).toEqual([
      new PartDone(0, thought("Let me think.", "SIG-1")),
      new PartDone(1, TEXT_PART),
      new PartDone(2, CALL_PART),
    ]);
    expect(out.filter((event) => event instanceof TextDelta)).toEqual([
      new TextDelta(1, "Hello"),
      new TextDelta(1, " there."),
    ]);
  });

  test("a text start with text emits a TextDelta", async () => {
    const out = await parse(
      reply([
        blockStart(0n, { type: "text", text: "Hi" }),
        blockDelta(0n, { type: "text_delta", text: " there" }),
        blockStop(0n),
      ]),
    );
    expect(out.slice(0, 3)).toEqual([
      new TextDelta(0, "Hi"),
      new TextDelta(0, " there"),
      new PartDone(0, new TextPart("Hi there")),
    ]);
  });

  test("an empty text start opens nothing until its first non-empty delta", async () => {
    const out = await parse(
      reply([
        blockStart(0n, { type: "text", text: "" }),
        blockDelta(0n, { type: "text_delta", text: "" }),
        blockStart(1n, { type: "tool_use", id: "toolu_01", name: "double", input: {} }),
        blockDelta(0n, { type: "text_delta", text: "Hi" }),
        blockStop(0n),
        blockStop(1n),
      ]),
    );
    expect(out.slice(0, 4)).toEqual([
      new ToolCallDelta({ index: 0, id: "toolu_01", name: "double", arguments: "" }),
      new TextDelta(1, "Hi"),
      new PartDone(1, new TextPart("Hi")),
      new PartDone(0, new ToolCallPart({ id: "toolu_01", name: "double", arguments: {} })),
    ]);
  });

  test("a text start with null text opens nothing", async () => {
    const out = await parse(reply([blockStart(0n, { type: "text", text: null }), blockStop(0n)]));
    expect(out).toEqual(END);
  });

  test("a thinking start without thinking or signature, or with both null, is an empty unsigned part", async () => {
    const starts: JsonObject[] = [{ type: "thinking" }, { type: "thinking", thinking: null, signature: null }];
    for (const content of starts) {
      const out = await parse(reply([blockStart(0n, content), blockStop(0n)]));
      expect(out).toEqual([new PartDone(0, new ReasoningPart("", null)), ...END]);
    }
  });

  test("a text start without text opens nothing", async () => {
    expect(await parse(reply([blockStart(0n, { type: "text" }), blockStop(0n)]))).toEqual(END);
  });

  test("a thinking start with text and a signature", async () => {
    const out = await parse(
      reply([blockStart(0n, { type: "thinking", thinking: "Hm.", signature: "SIG-0" }), blockStop(0n)]),
    );
    expect(out.slice(0, 2)).toEqual([new ReasoningDelta(0, "Hm."), new PartDone(0, thought("Hm.", "SIG-0"))]);
  });

  test("a signature_delta replaces the start signature", async () => {
    const parts = await partsOf(
      reply([
        blockStart(0n, { type: "thinking", thinking: "Hm.", signature: "SIG-0" }),
        blockDelta(0n, { type: "signature_delta", signature: "SIG-1" }),
        blockStop(0n),
      ]),
    );
    expect(parts).toEqual([thought("Hm.", "SIG-1")]);
  });

  test("a thinking block whose signature stays empty has no payload", async () => {
    const parts = await partsOf(
      reply([
        blockStart(0n, { type: "thinking", thinking: "", signature: "" }),
        blockDelta(0n, { type: "thinking_delta", thinking: "Hm." }),
        blockStop(0n),
      ]),
    );
    expect(parts).toEqual([new ReasoningPart("Hm.")]);
  });

  test("an omitted-display block, a signature only, is an empty reasoning part with its record", async () => {
    const out = await parse(
      reply([
        blockStart(0n, { type: "thinking", thinking: "", signature: "" }),
        blockDelta(0n, { type: "signature_delta", signature: "SIG-1" }),
        blockStop(0n),
        blockStart(1n, { type: "text", text: "" }),
        blockDelta(1n, { type: "text_delta", text: "Hi" }),
        blockStop(1n),
      ]),
    );
    expect(out.filter((event) => event instanceof ReasoningDelta)).toEqual([]);
    expect(partsDone(out)).toEqual([new PartDone(0, thought("", "SIG-1")), new PartDone(1, new TextPart("Hi"))]);
  });

  test("redacted data goes in the record; empty data gives no payload", async () => {
    expect(await partsOf(reply([blockStart(0n, REDACTED), blockStop(0n)]))).toEqual([redactedPart("REDACTED-DATA-1")]);
    expect(await partsOf(reply([blockStart(0n, { type: "redacted_thinking", data: "" }), blockStop(0n)]))).toEqual([
      new ReasoningPart(""),
    ]);
  });

  test("citations_delta and an unknown delta type are ignored", async () => {
    const parts = await partsOf(
      reply([
        blockStart(0n, { type: "text", text: "" }),
        blockDelta(0n, { type: "text_delta", text: "Hi" }),
        blockDelta(0n, { type: "citations_delta", citation: { type: "char_location", cited_text: "x" } }),
        blockDelta(0n, { type: "future_delta", text: "x" }),
        blockStop(0n),
      ]),
    );
    expect(parts).toEqual([new TextPart("Hi")]);
  });

  test("an unknown block type opens nothing and its deltas and stop are ignored", async () => {
    const out = await parse(
      reply([
        blockStart(0n, { type: "server_tool_use", id: "srvtoolu_1", name: "web_search", input: {} }),
        blockDelta(0n, { type: "input_json_delta", partial_json: '{"query": "x"}' }),
        blockStop(0n),
        blockStart(1n, { type: "web_search_tool_result", tool_use_id: "srvtoolu_1", content: [] }),
        blockStop(1n),
        blockStart(2n, { type: "text", text: "Hi" }),
        blockStop(2n),
      ]),
    );
    expect(out).toEqual([new TextDelta(0, "Hi"), new PartDone(0, new TextPart("Hi")), ...END]);
  });

  test("a delta or stop for an index never started is ignored", async () => {
    const out = await parse(
      reply([
        blockDelta(5n, { type: "text_delta", text: "lost" }),
        blockStop(5n),
        blockStart(0n, { type: "text", text: "Hi" }),
        blockStop(0n),
      ]),
    );
    expect(out).toEqual([new TextDelta(0, "Hi"), new PartDone(0, new TextPart("Hi")), ...END]);
  });

  test("a delta after its block's stop is ignored", async () => {
    const out = await parse(
      reply([
        blockStart(0n, { type: "text", text: "a" }),
        blockStop(0n),
        blockDelta(0n, { type: "text_delta", text: "b" }),
        blockStart(1n, { type: "thinking", thinking: "t", signature: "SIG-1" }),
        blockStop(1n),
        blockDelta(1n, { type: "thinking_delta", thinking: "u" }),
        blockDelta(1n, { type: "signature_delta", signature: "SIG-2" }),
      ]),
    );
    expect(out).toEqual([
      new TextDelta(0, "a"),
      new PartDone(0, new TextPart("a")),
      new ReasoningDelta(1, "t"),
      new PartDone(1, thought("t", "SIG-1")),
      ...END,
    ]);
  });

  test("ping, unknown events, empty data and non-object data are ignored", async () => {
    const out = await parse([
      { type: "ping" },
      start(),
      { type: "ping" },
      { type: "future_event", index: "x" },
      new SSEEvent({ data: "", event: "ping" }),
      new SSEEvent({ data: "[1, 2]", event: "content_block_start" }),
      new SSEEvent({ data: '"text"' }),
      new SSEEvent({ data: "null" }),
      blockStart(0n, { type: "text", text: "Hi" }),
      blockStop(0n),
      messageDelta(),
      STOP,
    ]);
    expect(out).toEqual([new TextDelta(0, "Hi"), new PartDone(0, new TextPart("Hi")), ...END]);
  });

  test("the type comes from the SSE event field when data has none", async () => {
    const untyped = (item: JsonObject) => {
      const { type, ...rest } = item;
      return new SSEEvent({ data: encodeBody(rest), event: type as string });
    };
    const out = await parse(reply([blockStart(0n, TEXT), blockStop(0n)]).map(untyped));
    expect(out).toEqual([new TextDelta(0, "Hello there."), new PartDone(0, TEXT_PART), ...END]);
  });

  test("an untyped frame with an error object is an error event whatever its SSE event name", async () => {
    const data = encodeBody({ error: { type: "overloaded_error", message: "Busy." } });
    const frame = new SSEEvent({ data, event: "message" });
    const error = await failure([start(), frame, messageDelta(), STOP]);
    expect(error.kind).toBe("overloaded");
    expect(error.message).toBe("anthropic stream failed (overloaded): Busy.");
  });

  test("a typed frame with an error, and an untyped frame whose error isn't an object, are ignored", async () => {
    const ignored = [
      new SSEEvent({ data: encodeBody({ type: "ping", error: { type: "overloaded_error" } }), event: "ping" }),
      new SSEEvent({ data: encodeBody({ error: "Busy." }), event: "ping" }),
      new SSEEvent({ data: encodeBody({ error: null }), event: "ping" }),
    ];
    const out = await parse([start(), ...ignored, blockStart(0n, TEXT), blockStop(0n), messageDelta(), STOP]);
    expect(out).toEqual([new TextDelta(0, "Hello there."), new PartDone(0, TEXT_PART), ...END]);
  });
});

describe("tool calls", () => {
  test("ToolCallDelta carries the id and name once", async () => {
    const out = await parse(events(messageObj([TOOL_USE], { stopReason: "tool_use" })));
    expect(out.filter((event) => event instanceof ToolCallDelta)).toEqual([
      new ToolCallDelta({ index: 0, id: "toolu_01", name: "double", arguments: "" }),
      new ToolCallDelta({ index: 0, id: null, name: null, arguments: '{"val' }),
      new ToolCallDelta({ index: 0, id: null, name: null, arguments: 'ue":3}' }),
    ]);
  });

  test("the arguments are the joined partial_json", async () => {
    const part = await callOf({}, ['{"value"', ": 3", ', "x": [1.5]}']);
    expect(part.arguments).toEqual({ value: 3n, x: [1.5] });
    expect(part.rawArguments).toBeNull();
  });

  test("with no input_json_delta the start input is the arguments", async () => {
    expect((await callOf(undefined)).arguments).toEqual({});
    expect((await callOf({ value: 3n })).arguments).toEqual({ value: 3n });
    for (const input of ["x", [1n], 5n, null] as JsonValue[]) {
      const part = await callOf(input);
      expect(part.arguments).toBeNull();
      expect(part.rawArguments).toBe(encodeBody(input));
    }
  });

  test("deltas win over the start input", async () => {
    expect((await callOf({ value: 1n }, ['{"value": 2}'])).arguments).toEqual({ value: 2n });
  });

  test("truncated JSON is not repaired", async () => {
    const part = await callOf({}, ['{"value": ', "3"]);
    expect(part.arguments).toBeNull();
    expect(part.rawArguments).toBe('{"value": 3');
  });

  test("empty partial_json pieces are an empty object", async () => {
    expect((await callOf({ value: 1n }, ["", ""])).arguments).toEqual({});
  });
});

describe("usage", () => {
  test("A_USAGE: input with cache reads and creation, output and total", async () => {
    expect((await collected(events(messageObj([TEXT])))).usage).toEqual(USAGE_COUNTS);
  });

  /** Return the usage of a reply with `startUsage` on `message_start` and `deltaUsage` on `message_delta`. */
  async function usageOf(startUsage: JsonValue, deltaUsage?: JsonValue): Promise<Usage | null> {
    return (await collected([start(startUsage), messageDelta("end_turn", deltaUsage), STOP])).usage;
  }

  test("the ephemeral breakdown's sum wins when above zero", async () => {
    const usage = await usageOf({
      input_tokens: 12n,
      output_tokens: 1n,
      cache_creation_input_tokens: 30n,
      cache_creation: { ephemeral_5m_input_tokens: 10n, ephemeral_1h_input_tokens: 20n },
    });
    expect(usage?.cacheCreationTokens).toBe(30);
    const sum = await usageOf({
      input_tokens: 12n,
      output_tokens: 1n,
      cache_creation_input_tokens: 99n,
      cache_creation: { ephemeral_5m_input_tokens: 10n, ephemeral_1h_input_tokens: 5n },
    });
    expect(sum?.cacheCreationTokens).toBe(15);
    expect(sum?.inputTokens).toBe(27);
    const zero = await usageOf({
      input_tokens: 12n,
      output_tokens: 1n,
      cache_creation_input_tokens: 30n,
      cache_creation: { ephemeral_5m_input_tokens: 0n, ephemeral_1h_input_tokens: 0n },
    });
    expect(zero?.cacheCreationTokens).toBe(30);
  });

  test("a classic message_delta usage, output only, keeps the start's input", async () => {
    const usage = await usageOf({ input_tokens: 12n, output_tokens: 1n }, { output_tokens: 40n });
    expect(usage).toEqual(new Usage({ inputTokens: 12, outputTokens: 40, totalTokens: 52 }));
  });

  test("a full message_delta usage replaces the start's", async () => {
    const usage = await usageOf({ input_tokens: 1n, output_tokens: 1n }, A_USAGE);
    expect(usage).toEqual(USAGE_COUNTS);
  });

  test("a null field doesn't replace an earlier value", async () => {
    const usage = await usageOf(
      { input_tokens: 12n, output_tokens: 1n, cache_read_input_tokens: 100n },
      { input_tokens: null, output_tokens: 40n, cache_read_input_tokens: null },
    );
    expect(usage).toEqual(new Usage({ inputTokens: 112, outputTokens: 40, totalTokens: 152, cacheReadTokens: 100 }));
  });

  test("null cache_creation and output_tokens_details count as absent", async () => {
    const usage = await usageOf({
      input_tokens: 12n,
      output_tokens: 1n,
      cache_creation_input_tokens: 30n,
      cache_creation: null,
      output_tokens_details: null,
    });
    expect(usage).toEqual(new Usage({ inputTokens: 42, outputTokens: 1, totalTokens: 43, cacheCreationTokens: 30 }));
  });

  test("thinking_tokens are the reasoning tokens", async () => {
    const usage = await usageOf({
      input_tokens: 12n,
      output_tokens: 40n,
      output_tokens_details: { thinking_tokens: 30n },
    });
    expect(usage?.reasoningTokens).toBe(30);
  });

  test("no usage at all gives no UsageReport", async () => {
    const out = await parse([
      { type: "message_start", message: { id: "msg_1", model: SONNET } },
      messageDelta(),
      STOP,
    ]);
    expect(out).toEqual([finish("stop")]);
  });

  test("a usage that is not an object is ignored", async () => {
    expect(await usageOf("x", 5n)).toBeNull();
    expect(await usageOf([1n], { output_tokens: 40n })).toEqual(
      new Usage({ inputTokens: 0, outputTokens: 40, totalTokens: 40 }),
    );
  });
});

describe("finish", () => {
  test("a message_delta without a stop_reason has no reason", async () => {
    const error = await failure([start(), { type: "message_delta", delta: {} }, STOP]);
    expect(error.kind).toBe("incomplete");
    const out = await parse([start(), messageDelta("max_tokens"), { type: "message_delta", delta: {} }, STOP]);
    expect(out.at(-1)).toEqual(finish("length"));
  });

  test("a later null stop_reason doesn't clear an earlier one", async () => {
    const out = await parse([start(), messageDelta("max_tokens"), messageDelta(null), STOP]);
    expect(out.at(-1)).toEqual(finish("length"));
  });

  test("UsageReport and Finish come only at message_stop", async () => {
    const log: string[] = [];
    async function* source(): AsyncGenerator<SSEEvent> {
      for (const frame of sse(events(messageObj([TEXT])))) {
        log.push(`frame ${frame.event}`);
        yield frame;
      }
    }
    for await (const event of route().parse(source())) log.push(event.type);
    expect(log.slice(-5)).toEqual(["PartDone", "frame message_delta", "frame message_stop", "UsageReport", "Finish"]);
  });

  test("the parser stops reading at message_stop", async () => {
    let pulled = false;
    async function* source(): AsyncGenerator<SSEEvent> {
      yield* sse(events(messageObj([TEXT])));
      pulled = true;
      throw new Error("read past message_stop");
    }
    const out: Event[] = [];
    for await (const event of route().parse(source())) out.push(event);
    expect(out.at(-1)).toEqual(finish("stop"));
    expect(pulled).toBe(false);
  });

  test("a consumer that stops early closes the frame source", async () => {
    let closed = false;
    async function* source(): AsyncGenerator<SSEEvent> {
      try {
        yield* sse(events(messageObj([TEXT])));
      } finally {
        closed = true;
      }
    }
    for await (const event of route().parse(source())) {
      expect(event).toEqual(new TextDelta(0, "Hello"));
      break;
    }
    expect(closed).toBe(true);
  });

  test("a message_delta without message_stop completes at the end of input", async () => {
    const whole = await parse(events(messageObj([TEXT])));
    expect(await parse(events(messageObj([TEXT]), { dropStop: true }))).toEqual(whole);
  });

  test("parts still open at the finish point get their PartDone there, in index order", async () => {
    const items = events(messageObj([THINKING, REDACTED, TEXT, TOOL_USE], { stopReason: "tool_use" })).filter(
      (item) => item["type"] !== "content_block_stop",
    );
    for (const ending of [items, items.slice(0, -1)]) {
      const out = await parse(ending);
      expect(out.slice(-6)).toEqual([
        new PartDone(0, thought("Let me think.", "SIG-1")),
        new PartDone(1, redactedPart("REDACTED-DATA-1")),
        new PartDone(2, TEXT_PART),
        new PartDone(3, CALL_PART),
        new UsageReport(USAGE_COUNTS),
        finish("tool_calls"),
      ]);
    }
  });

  test("a part stopped early keeps its place; the rest close at the finish point", async () => {
    const out = await parse(
      reply([
        blockStart(0n, { type: "text", text: "A" }),
        blockStart(1n, { type: "text", text: "B" }),
        blockStart(2n, { type: "text", text: "C" }),
        blockStop(1n),
      ]),
    );
    expect(partsDone(out)).toEqual([
      new PartDone(1, new TextPart("B")),
      new PartDone(0, new TextPart("A")),
      new PartDone(2, new TextPart("C")),
    ]);
  });

  const reasons: [string, JsonObject[], string][] = [
    ["end_turn", [TEXT], "stop"],
    ["stop_sequence", [TEXT], "stop"],
    ["end_turn", [TEXT, TOOL_USE], "tool_calls"],
    ["tool_use", [TEXT, TOOL_USE], "tool_calls"],
    ["tool_use", [TEXT], "stop"],
    ["max_tokens", [TEXT], "length"],
    ["max_tokens", [TEXT, TOOL_USE], "length"],
    ["future_reason", [TEXT], "future_reason"],
    ["future_reason", [TOOL_USE], "tool_calls"],
  ];
  for (const [stopReason, content, reason] of reasons) {
    const types = content.map((block) => block["type"]).join(" + ");
    test(`stop reason ${stopReason} with ${types} finishes ${reason}`, async () => {
      const out = await parse(events(messageObj(content, { stopReason })));
      expect(out.at(-1)).toEqual(finish(reason));
    });
  }
});

describe("records", () => {
  test("every reasoning payload reads back with its key, v 1 and the account when set", async () => {
    for (const account of [null, "acct-1"]) {
      const key: ReplayKey = { ...KEY, account };
      const response = await collected(events(messageObj([THINKING, REDACTED, TEXT])), route(account));
      const [thinking, redacted] = response.message.parts as ReasoningPart[];
      expect(readReplayRecord(thinking!.providerPayload, key, ["signature", "redacted"])).toEqual({
        ok: true,
        fields: { signature: "SIG-1" },
      });
      expect(readReplayRecord(redacted!.providerPayload, key, ["signature", "redacted"])).toEqual({
        ok: true,
        fields: { redacted: "REDACTED-DATA-1" },
      });
      for (const part of [thinking!, redacted!]) {
        expect(part.providerPayload!["v"]).toBe(1n);
        expect(part.providerPayload!["account"]).toEqual(account ?? undefined);
      }
      expect(response.message.parts[2]!.providerPayload).toBeNull();
    }
  });

  test("the Finish record holds the response id and model; its protocol is anthropic_messages", async () => {
    const out = await parse(events(messageObj([TEXT], { model: "claude-sonnet-4-5-20250929" })));
    const last = out.at(-1) as Finish;
    expect(last.protocol).toBe("anthropic_messages");
    expect(last.responseId).toBe("msg_1");
    expect(readReplayRecord(last.providerPayload, KEY, ["response_id", "response_model"])).toEqual({
      ok: true,
      fields: { response_id: "msg_1", response_model: "claude-sonnet-4-5-20250929" },
    });
  });

  test("text and call parts carry no payload", async () => {
    const parts = await partsOf(events(messageObj([TEXT, TOOL_USE], { stopReason: "tool_use" })));
    expect(parts.map((part) => part.providerPayload)).toEqual([null, null]);
  });
});

describe("error events", () => {
  /** Return an `error` event carrying `error`. */
  function errorEvent(error: JsonValue): JsonObject {
    return { type: "error", error };
  }

  const TEXT_START = blockStart(0n, { type: "text", text: "Hi" });

  const TYPES: [string, ErrorKind, boolean][] = [
    ["overloaded_error", "overloaded", true],
    ["rate_limit_error", "rate_limit", true],
    ["api_error", "server", true],
    ["timeout_error", "timeout", true],
    ["authentication_error", "auth", false],
    ["permission_error", "permission", false],
    ["billing_error", "quota", false],
    ["not_found_error", "not_found", false],
    ["request_too_large", "bad_request", false],
    ["invalid_request_error", "bad_request", false],
    ["future_error", "server", true],
  ];

  test.each(TYPES)("an error event of type %p after text raises %s", async (type, kind, retryable) => {
    const [out, error] = await failing([start(), TEXT_START, errorEvent({ type, message: "It broke." }), STOP]);
    expect(error.kind).toBe(kind);
    expect(error.retryable).toBe(retryable);
    expect(error.message).toBe(`anthropic stream failed (${kind}): It broke.`);
    expect(out).toEqual([new TextDelta(0, "Hi")]);
  });

  test("an error with no type, or an error that isn't an object, is a server error", async () => {
    const error = await failure([start(), errorEvent({ message: "It broke." })]);
    expect(error.kind).toBe("server");
    expect(error.retryable).toBe(true);
    expect(error.message).toBe("anthropic stream failed (server): It broke.");
    for (const value of [{ type: 5n, message: "It broke." }, "It broke.", null]) {
      expect((await failure([start(), errorEvent(value)])).kind).toBe("server");
    }
    expect((await failure([start(), { type: "error" }])).message).toBe("anthropic stream failed (server): no message");
  });

  test("invalid_request_error with a context phrase is context_length", async () => {
    const message = "prompt is too long: 210000 tokens > 200000 maximum";
    const error = await failure([start(), errorEvent({ type: "invalid_request_error", message })]);
    expect(error.kind).toBe("context_length");
    expect(error.retryable).toBe(false);
    expect(error.message).toBe(`anthropic stream failed (context_length): ${message}`);
  });

  test("an empty, missing or non-string message is no message", async () => {
    for (const message of ["", null, 3n, ["It broke."], undefined] as (JsonValue | undefined)[]) {
      const body: JsonObject = { type: "overloaded_error" };
      if (message !== undefined) body["message"] = message;
      const error = await failure([start(), errorEvent(body)]);
      expect(error.message).toBe("anthropic stream failed (overloaded): no message");
    }
  });

  test("an untyped error frame raises the same way", async () => {
    for (const type of TYPES.map(([each]) => each)) {
      const body = { type, message: "It broke." };
      const typed = await failure([start(), TEXT_START, errorEvent(body)]);
      const frame = new SSEEvent({ data: encodeBody({ error: body }) });
      const [out, untyped] = await failing([start(), TEXT_START, frame, STOP]);
      expect([untyped.kind, untyped.retryable, untyped.message]).toEqual([typed.kind, typed.retryable, typed.message]);
      expect(out).toEqual([new TextDelta(0, "Hi")]);
    }
  });

  test("an error after a message_delta with end_turn raises with no UsageReport or Finish", async () => {
    const items = [start(), TEXT_START, blockStop(0n), messageDelta("end_turn", A_USAGE)];
    for (const ending of [[errorEvent({ type: "overloaded_error", message: "Busy." }), STOP], [errorEvent({})]]) {
      const [out] = await failing([...items, ...ending]);
      expect(out).toEqual([new TextDelta(0, "Hi"), new PartDone(0, new TextPart("Hi"))]);
    }
  });

  test("the message is the provider's text uncapped", async () => {
    const message = "x".repeat(2500);
    const error = await failure([start(), errorEvent({ type: "api_error", message })]);
    expect(error.message).toBe(`anthropic stream failed (server): ${message}`);
  });
});

describe("failed stop reasons", () => {
  const FAILED: [string, ErrorKind][] = [
    ["refusal", "refused"],
    ["pause_turn", "incomplete"],
    ["model_context_window_exceeded", "context_length"],
  ];

  test.each(FAILED)("%s raises %s at message_stop and at the end of input", async (stopReason, kind) => {
    const items = events(messageObj([TEXT], { stopReason }));
    for (const ending of [items, items.slice(0, -1)]) {
      const [out, error] = await failing(ending);
      expect(error.kind).toBe(kind);
      expect(error.retryable).toBe(false);
      expect(error.message).toBe(`anthropic response did not complete (${stopReason})`);
      expect(out).toEqual([new TextDelta(0, "Hello"), new TextDelta(0, " there."), new PartDone(0, TEXT_PART)]);
    }
  });
});

describe("truncation", () => {
  const TRUNCATED = "anthropic stream ended before a complete response";
  const TEXT_START: JsonObject = { type: "text", text: "" };

  test("the end of input after text deltas with no message_delta is incomplete and retryable", async () => {
    const items = [start(), blockStart(0n, TEXT_START), blockDelta(0n, { type: "text_delta", text: "Hi" })];
    const [out, error] = await failing(items);
    expect(error.kind).toBe("incomplete");
    expect(error.retryable).toBe(true);
    expect(error.message).toBe(TRUNCATED);
    expect(out).toEqual([new TextDelta(0, "Hi")]);
  });

  test("a message_stop with no stop reason is incomplete", async () => {
    const items = [start(), blockStart(0n, TEXT), blockStop(0n), messageDelta(null, A_USAGE), STOP];
    const [out, error] = await failing(items);
    expect([error.kind, error.retryable, error.message]).toEqual(["incomplete", true, TRUNCATED]);
    expect(out).toEqual([new TextDelta(0, "Hello there."), new PartDone(0, TEXT_PART)]);
    expect((await failure([start(), STOP])).message).toBe(TRUNCATED);
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
    ["a lone surrogate escape", '{"type": "content_block_delta", "index": 0, "delta": {"text": "\\ud800 test-key-1"}}'],
    ["a duplicate key", '{"type": "message_delta", "type": "test-key-1"}'],
  ];

  test.each(INVALID)("%s is a protocol error that doesn't echo the event", async (_, data) => {
    const [out, error] = await failing([start(), blockStart(0n, TEXT), new SSEEvent({ data }), STOP]);
    expect(error.kind).toBe("protocol");
    expect(error.retryable).toBe(false);
    expect(error.message).toBe("invalid JSON in anthropic stream event");
    expect(error.cause).toBeUndefined();
    expect(error.causeName).toBeNull();
    expect(String(error.stack)).not.toContain("test-key-1");
    expect(out).toEqual([new TextDelta(0, "Hello there.")]);
  });
});

describe("shape errors", () => {
  /** Assert `items` (after a `message_start`) raise a protocol error naming `type`. */
  async function invalid(type: string, ...items: (JsonObject | SSEEvent)[]): Promise<void> {
    const error = await failure([start(), ...items, messageDelta(), STOP]);
    expect(error.kind).toBe("protocol");
    expect(error.retryable).toBe(false);
    expect(error.message).toBe(`invalid ${type} in anthropic stream event`);
  }

  test("a block event whose index isn't an int", async () => {
    for (const index of ["0", null, true, [0n]] as JsonValue[]) {
      await invalid("content_block_start", { type: "content_block_start", index, content_block: TEXT });
      const delta = { type: "text_delta", text: "x" };
      await invalid("content_block_delta", { type: "content_block_delta", index, delta });
      await invalid("content_block_stop", { type: "content_block_stop", index });
    }
    await invalid("content_block_stop", { type: "content_block_stop" });
    // Raw JSON, so `0.0` reaches the parser as a float.
    const data = encodeBody({ type: "content_block_stop", index: "INDEX" }).replace('"INDEX"', "0.0");
    await invalid("content_block_stop", blockStart(0n, TEXT), new SSEEvent({ data }));
  });

  test("a start whose content_block isn't an object", async () => {
    for (const content of ["text", null, [TEXT], 1n] as JsonValue[]) {
      await invalid("content_block_start", { type: "content_block_start", index: 0n, content_block: content });
    }
    await invalid("content_block_start", { type: "content_block_start", index: 0n });
  });

  test("a delta whose delta isn't an object", async () => {
    for (const delta of ["x", null, [], 1n] as JsonValue[]) {
      await invalid("content_block_delta", blockStart(0n, TEXT), { type: "content_block_delta", index: 0n, delta });
    }
  });

  test("a required delta string missing or not a string", async () => {
    const starts: [string, string, JsonObject][] = [
      ["text_delta", "text", { type: "text", text: "" }],
      ["thinking_delta", "thinking", { type: "thinking", thinking: "", signature: "" }],
      ["signature_delta", "signature", { type: "thinking", thinking: "", signature: "" }],
      ["input_json_delta", "partial_json", { type: "tool_use", id: "toolu_01", name: "double", input: {} }],
    ];
    for (const [deltaType, field, content] of starts) {
      for (const value of [undefined, null, 5n, {}] as (JsonValue | undefined)[]) {
        const delta: JsonObject = { type: deltaType };
        if (value !== undefined) delta[field] = value;
        await invalid("content_block_delta", blockStart(0n, content), blockDelta(0n, delta));
      }
    }
  });

  test("a redacted_thinking start whose data is missing or not a string", async () => {
    for (const data of [undefined, null, 5n, ["x"]] as (JsonValue | undefined)[]) {
      const content: JsonObject = { type: "redacted_thinking" };
      if (data !== undefined) content["data"] = data;
      await invalid("content_block_start", blockStart(0n, content));
    }
  });

  test("an optional start string present, not null and not a string", async () => {
    const cases: JsonObject[] = [
      { type: "text", text: 5n },
      { type: "text", text: ["x"] },
      { type: "thinking", thinking: 5n, signature: "" },
      { type: "thinking", thinking: "", signature: {} },
      { type: "thinking", thinking: true },
    ];
    for (const content of cases) await invalid("content_block_start", blockStart(0n, content));
  });

  test("a tool_use start without a non-empty string id and name", async () => {
    const cases: JsonObject[] = [
      { type: "tool_use", name: "double", input: {} },
      { type: "tool_use", id: "", name: "double", input: {} },
      { type: "tool_use", id: 5n, name: "double", input: {} },
      { type: "tool_use", id: "toolu_01", input: {} },
      { type: "tool_use", id: "toolu_01", name: "", input: {} },
      { type: "tool_use", id: "toolu_01", name: null, input: {} },
    ];
    for (const content of cases) await invalid("content_block_start", blockStart(0n, content));
  });

  test("a message_start whose message isn't an object", async () => {
    for (const message of [null, "msg_1", [], undefined] as (JsonValue | undefined)[]) {
      const data: JsonObject = { type: "message_start" };
      if (message !== undefined) data["message"] = message;
      const error = await failure([data, messageDelta(), STOP]);
      expect(error.kind).toBe("protocol");
      expect(error.message).toBe("invalid message_start in anthropic stream event");
    }
  });

  test("a message_delta whose delta isn't an object or whose stop_reason isn't a string or null", async () => {
    for (const delta of [null, "end_turn", [], undefined] as (JsonValue | undefined)[]) {
      const data: JsonObject = { type: "message_delta" };
      if (delta !== undefined) data["delta"] = delta;
      await invalid("message_delta", data);
    }
    for (const stopReason of [5n, true, {}, ["end_turn"]] as JsonValue[]) {
      await invalid("message_delta", messageDelta(stopReason));
    }
  });

  test("message_start id and model that aren't strings are ignored", async () => {
    const out = await parse([{ type: "message_start", message: { id: 5n, model: null } }, messageDelta(), STOP]);
    const providerPayload = replayRecord(KEY, {});
    expect(out).toEqual([new Finish({ reason: "stop", responseId: null, protocol: PROTOCOL, providerPayload })]);
  });
});

describe("usage errors", () => {
  const DETAILS = "anthropic response usage details are not objects";
  const NOT_INT = "anthropic response usage has a count that is not an int";

  test.each(["cache_creation", "output_tokens_details"])("usage %s that aren't objects", async (field) => {
    for (const details of ["x", [1n], 3n, true] as JsonValue[]) {
      const usage = { input_tokens: 12n, output_tokens: 1n, [field]: details };
      for (const items of [[start(usage)], [start(), messageDelta("end_turn", usage)]]) {
        const error = await failure([...items, STOP]);
        expect(error.kind).toBe("protocol");
        expect(error.retryable).toBe(false);
        expect(error.message).toBe(DETAILS);
      }
    }
  });

  const counts: [string, (count: JsonValue) => JsonObject][] = [
    ["input_tokens", (count) => ({ input_tokens: count })],
    ["output_tokens", (count) => ({ output_tokens: count })],
    ["cache_read_input_tokens", (count) => ({ cache_read_input_tokens: count })],
    ["cache_creation_input_tokens", (count) => ({ cache_creation_input_tokens: count })],
    ["ephemeral_5m_input_tokens", (count) => ({ cache_creation: { ephemeral_5m_input_tokens: count } })],
    ["thinking_tokens", (count) => ({ output_tokens_details: { thinking_tokens: count } })],
  ];
  test.each(counts)("a usage %s that isn't a safe int", async (_, usage) => {
    // Raw JSON, so a float count that is an integer (`1.0`, `1e3`) reaches the parser.
    for (const token of ["1.0", "1e3", "1.5", "9007199254740992", "18446744073709551616", '"3"', "true", "-1.5"]) {
      const data = encodeBody(messageDelta("end_turn", usage("COUNT"))).replace('"COUNT"', token);
      const [out, error] = await failing([start(), blockStart(0n, TEXT), new SSEEvent({ data }), STOP]);
      expect(error.kind).toBe("protocol");
      expect(error.retryable).toBe(false);
      expect(error.message).toBe(NOT_INT);
      expect(out).toEqual([new TextDelta(0, "Hello there.")]);
    }
  });

  test("a bad count fails on its own event, even when a later message_delta replaces it", async () => {
    for (const token of ["1.0", "9007199254740992"]) {
      const data = encodeBody(start({ input_tokens: "COUNT", output_tokens: 1n })).replace('"COUNT"', token);
      const error = await failure([new SSEEvent({ data }), messageDelta("end_turn", { input_tokens: 5n }), STOP]);
      expect(error.kind).toBe("protocol");
      expect(error.message).toBe(NOT_INT);
    }
  });

  test("a sum past 2^53 - 1 is a protocol error before any part is done", async () => {
    const usage = { input_tokens: 2n ** 53n - 1n, output_tokens: 1n };
    const [out, error] = await failing([start(usage), blockStart(0n, TEXT), messageDelta(), STOP]);
    expect(error.kind).toBe("protocol");
    expect(error.message).toBe(NOT_INT);
    expect(out).toEqual([new TextDelta(0, "Hello there.")]);
  });
});

describe("round trips", () => {
  const ASK = new Message({ role: "user", parts: [new TextPart("Double 3.")] });
  const RESULT = new Message({
    role: "tool",
    parts: [new ToolResultPart({ callId: "toolu_01", name: "double", content: "6", isError: false })],
  });
  const ENABLED: JsonObject = { thinking: { type: "enabled", budget_tokens: 2048n } };
  const TOOL_RESULT: JsonObject = {
    role: "user",
    content: [{ type: "tool_result", content: "6", tool_use_id: "toolu_01", is_error: false }],
  };

  /** Return the reply parsed from the Messages SSE events streamed for `content`. */
  async function replyOf(content: JsonObject[]): Promise<Message> {
    return (await collected(events(messageObj(content, { stopReason: "tool_use" })))).message;
  }

  /** Return a protocol for `model` with `block` and `account`, pushing each degraded reason to `reasons`. */
  function protocolFor(
    model: string,
    block: JsonObject | null,
    reasons: ReplayDegradedReason[] = [],
    account: string | null = null,
  ) {
    const onReplayDegraded = (reason: ReplayDegradedReason) => reasons.push(reason);
    return new AnthropicMessages({ model, provider: "anthropic", account, block, onReplayDegraded });
  }

  /** Return the messages `protocol` sends for `messages`. */
  function sentBy(protocol: AnthropicMessages, messages: Message[]): JsonValue {
    return protocol.buildBody(new Request({ messages }))["messages"]!;
  }

  const FIRST_BLOCKS: JsonObject[] = [
    { type: "thinking", thinking: "Let me think.", signature: "SIG-1" },
    { type: "redacted_thinking", data: "REDACTED-DATA-1" },
    { type: "text", text: "Hello there." },
    { type: "tool_use", id: "toolu_01", name: "double", input: { value: 3n } },
  ];

  test("a thinking, redacted, text and tool_use reply goes back in order with its signature and data", async () => {
    const first = await replyOf([THINKING, REDACTED, TEXT, TOOL_USE]);
    const reasons: ReplayDegradedReason[] = [];
    expect(sentBy(protocolFor(SONNET, ENABLED, reasons), [ASK, first, RESULT])).toEqual([
      { role: "user", content: "Double 3." },
      { role: "assistant", content: FIRST_BLOCKS },
      TOOL_RESULT,
    ]);
    expect(reasons).toEqual([]);
  });

  test("a later tool_use-only reply with its tool result goes out without error", async () => {
    const first = await replyOf([THINKING, REDACTED, TEXT, TOOL_USE]);
    const second = await replyOf([{ ...TOOL_USE, id: "toolu_02" }]);
    const secondResult = new Message({
      role: "tool",
      parts: [new ToolResultPart({ callId: "toolu_02", name: "double", content: "12", isError: false })],
    });
    expect(sentBy(protocolFor(SONNET, ENABLED), [ASK, first, RESULT, second, secondResult])).toEqual([
      { role: "user", content: "Double 3." },
      { role: "assistant", content: FIRST_BLOCKS },
      TOOL_RESULT,
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_02", name: "double", input: { value: 3n } }] },
      { role: "user", content: [{ type: "tool_result", content: "12", tool_use_id: "toolu_02", is_error: false }] },
    ]);
  });

  test("a reply replays to claude-opus-4-5, thinking enabled or not; another account leaves its reasoning out", async () => {
    const first = await replyOf([THINKING, REDACTED, TEXT, TOOL_USE]);
    for (const block of [ENABLED, null]) {
      const reasons: ReplayDegradedReason[] = [];
      expect(sentBy(protocolFor("claude-opus-4-5", block, reasons), [ASK, first, RESULT])).toEqual([
        { role: "user", content: "Double 3." },
        { role: "assistant", content: FIRST_BLOCKS },
        TOOL_RESULT,
      ]);
      expect(reasons).toEqual([]);
    }

    const other: ReplayDegradedReason[] = [];
    expect(sentBy(protocolFor(SONNET, ENABLED, other, "acct"), [ASK, first, RESULT])).toEqual([
      { role: "user", content: "Double 3." },
      { role: "assistant", content: FIRST_BLOCKS.slice(2) },
      TOOL_RESULT,
    ]);
    expect(other).toEqual(["other_account"]);
  });
});
