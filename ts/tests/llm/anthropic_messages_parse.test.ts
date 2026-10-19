/**
 * Tests for Anthropic Messages reply parsing in `src/llm/protocols/anthropic_messages.ts`: SSE
 * events into deltas, parts with replay records, dense indexes, tool-call arguments, usage, the
 * finish point and finish reasons.
 */

import { describe, expect, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
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
  type ReplayKey,
  replayRecord,
  Response,
  TextPart,
  ToolCallPart,
  Usage,
} from "../../src/llm/messages.ts";
import { AnthropicMessages, PROTOCOL } from "../../src/llm/protocols/anthropic_messages.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

const SONNET = "claude-sonnet-4-5";
const KEY: ReplayKey = { protocol: PROTOCOL, provider: "anthropic", model: SONNET };

// The s3 probe's reply shapes.
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
async function callOf(input: JsonValue | undefined, deltas: string[] | null): Promise<ToolCallPart> {
  const block: JsonObject = { type: "tool_use", id: "toolu_01", name: "double" };
  if (input !== undefined) block["input"] = input;
  const body = [blockStart(0n, block)];
  for (const piece of deltas ?? []) body.push(blockDelta(0n, { type: "input_json_delta", partial_json: piece }));
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
    expect(out).toEqual([...END]);
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
    expect((await callOf(undefined, null)).arguments).toEqual({});
    expect((await callOf({ value: 3n }, null)).arguments).toEqual({ value: 3n });
    for (const input of ["x", [1n], 5n, null] as JsonValue[]) {
      const part = await callOf(input, null);
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
