/**
 * Tests for OpenAI Responses reply parsing in `src/llm/protocols/openai_responses.ts`: SSE
 * events into deltas, parts with replay records, dense indexes, terminal events, usage and
 * finish reasons.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";
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
import { SSEEvent } from "../../src/llm/framing.ts";
import {
  Message,
  type Part,
  ReasoningPart,
  RefusalPart,
  type ReplayKey,
  replayRecord,
  Response,
  TextPart,
  ToolCallPart,
  Usage,
} from "../../src/llm/messages.ts";
import { OpenAIResponses, PROTOCOL } from "../../src/llm/protocols/openai_responses.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

afterEach(() => install(BUILTIN));

const KEY: ReplayKey = { protocol: PROTOCOL, provider: "openai", model: "gpt-5.5", account: null };
const REASONING: JsonObject = {
  id: "rs_1",
  type: "reasoning",
  summary: [{ type: "summary_text", text: "Thinking about it." }],
  encrypted_content: "ENC-CONTINUATION-1",
};
const MESSAGE: JsonObject = {
  id: "msg_1",
  type: "message",
  role: "assistant",
  status: "completed",
  content: [{ type: "output_text", text: "Hello there.", annotations: [] }],
};
const CALL: JsonObject = {
  id: "fc_1",
  type: "function_call",
  name: "double",
  arguments: '{"value": 3}',
  call_id: "call_1",
  status: "completed",
};
const USAGE: JsonObject = {
  input_tokens: 21n,
  input_tokens_details: { cached_tokens: 5n },
  output_tokens: 40n,
  output_tokens_details: { reasoning_tokens: 30n },
  total_tokens: 61n,
};
const REASONING_PART = new ReasoningPart("Thinking about it.", replayRecord(KEY, { item: REASONING }));
const TEXT_PART = new TextPart("Hello there.", replayRecord(KEY, { id: "msg_1", annotations: [] }));
const CALL_PART = new ToolCallPart({
  id: "call_1",
  name: "double",
  arguments: { value: 3n },
  providerPayload: replayRecord(KEY, { item: CALL }),
});
const USAGE_COUNTS = new Usage({
  inputTokens: 21,
  outputTokens: 40,
  totalTokens: 61,
  cacheReadTokens: 5,
  reasoningTokens: 30,
});
const ITEM_EVENTS = [
  "response.output_item.added",
  "response.output_item.done",
  "response.content_part.added",
  "response.output_text.delta",
  "response.refusal.delta",
  "response.refusal.done",
  "response.reasoning_summary_text.delta",
  "response.function_call_arguments.delta",
];

function apiKey(): OpenAIResponses {
  return new OpenAIResponses({ model: "gpt-5.5", provider: "openai" });
}

function subscription(): OpenAIResponses {
  return new OpenAIResponses({ model: "gpt-5.5", provider: "openai", subscription: true });
}

/** Return the `Finish` the parser gives for a reply with `responseId`. */
function finish(reason: string, responseId = "resp_1"): Finish {
  return new Finish({
    reason,
    responseId,
    protocol: PROTOCOL,
    providerPayload: replayRecord(KEY, { response_id: responseId }),
  });
}

/** Return the response a reply with `parts` collects to. */
function expected(parts: Part[], finishReason: string): Response {
  const message = new Message({
    role: "assistant",
    parts,
    protocol: PROTOCOL,
    providerPayload: replayRecord(KEY, { response_id: "resp_1" }),
  });
  return new Response({ message, usage: USAGE_COUNTS, finishReason, responseId: "resp_1" });
}

function reply(output: JsonValue[] = [], fields: JsonObject = {}): JsonObject {
  return { id: "resp_1", object: "response", status: "completed", model: "gpt-5.5", output, usage: USAGE, ...fields };
}

/** Return each event as the frame the API sends: its type as `event`, its JSON as `data`. */
function sse(events: (JsonObject | SSEEvent)[]): SSEEvent[] {
  return events.map((event) => {
    if (event instanceof SSEEvent) return event;
    const type = event["type"];
    return new SSEEvent({ data: encodeBody(event), event: typeof type === "string" ? type : null });
  });
}

async function* frames(events: (JsonObject | SSEEvent)[]): AsyncGenerator<SSEEvent> {
  yield* sse(events);
}

async function parse(events: (JsonObject | SSEEvent)[], protocol = apiKey()): Promise<Event[]> {
  const out: Event[] = [];
  for await (const event of protocol.parse(frames(events))) out.push(event);
  return out;
}

/** Return the `ProviderError` that parsing `events` throws; fail the test if it doesn't. */
async function failure(events: (JsonObject | SSEEvent)[], protocol = apiKey()): Promise<ProviderError> {
  try {
    await parse(events, protocol);
  } catch (error) {
    expect(error).toBeInstanceOf(ProviderError);
    return error as ProviderError;
  }
  throw new Error("expected a ProviderError");
}

function created(responseId = "resp_1"): JsonObject {
  return { type: "response.created", response: { id: responseId, status: "in_progress" } };
}

function added(outputIndex: number, item: JsonObject): JsonObject {
  return { type: "response.output_item.added", output_index: outputIndex, item };
}

function done(outputIndex: number, item: JsonValue): JsonObject {
  return { type: "response.output_item.done", output_index: outputIndex, item };
}

function contentPartAdded(outputIndex: number, contentIndex: number): JsonObject {
  return { type: "response.content_part.added", output_index: outputIndex, content_index: contentIndex };
}

function textDelta(outputIndex: number, contentIndex: number, delta: string, itemId = "msg_1"): JsonObject {
  return {
    type: "response.output_text.delta",
    item_id: itemId,
    output_index: outputIndex,
    content_index: contentIndex,
    delta,
  };
}

function refusalDelta(outputIndex: number, contentIndex: number, delta: string): JsonObject {
  return {
    type: "response.refusal.delta",
    item_id: "msg_1",
    output_index: outputIndex,
    content_index: contentIndex,
    delta,
  };
}

function summaryDelta(outputIndex: number, delta: string, summaryIndex = 0): JsonObject {
  return {
    type: "response.reasoning_summary_text.delta",
    item_id: "rs_1",
    output_index: outputIndex,
    summary_index: summaryIndex,
    delta,
  };
}

function argumentsDelta(outputIndex: number, delta: string): JsonObject {
  return { type: "response.function_call_arguments.delta", item_id: "fc_1", output_index: outputIndex, delta };
}

function completed(output: JsonValue[] = [], fields: JsonObject = {}): JsonObject {
  return { type: "response.completed", response: reply(output, fields) };
}

function incomplete(reason: string, output: JsonValue[] = []): JsonObject {
  const response = reply(output, { status: "incomplete", incomplete_details: { reason } });
  return { type: "response.incomplete", response };
}

/** Return a message item as `output_item.added` carries it: in progress, with no content yet. */
function started(item: JsonObject): JsonObject {
  return item["type"] === "message" ? { ...item, status: "in_progress", content: [] } : item;
}

/** Return the events the API streams for `response`, ending with `response.<terminal>`. */
function sseEventsFor(response: JsonObject, terminal = "completed"): JsonObject[] {
  const events: JsonObject[] = [
    { type: "response.created", response: { ...response, status: "in_progress", output: [], usage: null } },
  ];
  const output = response["output"] as JsonObject[];
  output.forEach((item, i) => {
    const at = { output_index: i, item_id: item["id"]! };
    if (item["type"] === "reasoning") {
      events.push(added(i, { ...item, summary: [], encrypted_content: null }));
      (item["summary"] as JsonObject[]).forEach((part, si) => {
        const summary = { ...at, summary_index: si };
        const empty = { type: "summary_text", text: "" };
        events.push({ type: "response.reasoning_summary_part.added", ...summary, part: empty });
        events.push({ type: "response.reasoning_summary_text.delta", ...summary, delta: part["text"]! });
      });
    } else if (item["type"] === "function_call") {
      events.push(added(i, { ...item, arguments: "", status: "in_progress" }));
      const args = item["arguments"] as string;
      for (const piece of [args.slice(0, args.length >> 1), args.slice(args.length >> 1)]) {
        events.push({ type: "response.function_call_arguments.delta", ...at, delta: piece });
      }
      events.push({ type: "response.function_call_arguments.done", ...at, arguments: args });
    } else {
      events.push(added(i, { ...item, content: [], status: "in_progress" }));
      (item["content"] as JsonObject[]).forEach((part, ci) => {
        const where = { ...at, content_index: ci };
        events.push({ type: "response.content_part.added", ...where, part: { type: "output_text", text: "" } });
        if (part["type"] === "refusal") {
          events.push({ type: "response.refusal.delta", ...where, delta: part["refusal"]! });
          events.push({ type: "response.refusal.done", ...where, refusal: part["refusal"]! });
        } else {
          const text = part["text"] as string;
          for (const piece of [text.slice(0, 5), text.slice(5)]) {
            events.push({ type: "response.output_text.delta", ...where, delta: piece });
          }
          events.push({ type: "response.output_text.done", ...where, text });
        }
        events.push({ type: "response.content_part.done", ...where, part });
      });
    }
    events.push(done(i, item));
  });
  events.push({ type: `response.${terminal}`, response: { ...response, status: terminal } });
  return events;
}

/** Reasoning, then a message, streamed as the API sends them. */
function streamedReply(): JsonObject[] {
  return [
    created(),
    added(0, { ...REASONING, summary: [], encrypted_content: null }),
    { type: "response.reasoning_summary_part.added", output_index: 0, summary_index: 0 },
    summaryDelta(0, "Thinking "),
    summaryDelta(0, "about it."),
    done(0, REASONING),
    added(1, started(MESSAGE)),
    contentPartAdded(1, 0),
    textDelta(1, 0, "Hello"),
    textDelta(1, 0, " there."),
    { type: "response.output_text.done", output_index: 1, content_index: 0, text: "Hello there." },
    done(1, MESSAGE),
    completed([REASONING, MESSAGE]),
  ];
}

function partsDone(events: Event[]): PartDone[] {
  return events.filter((event) => event instanceof PartDone);
}

describe("deltas and parts", () => {
  test("reply becomes deltas, parts, usage and finish", async () => {
    expect(await parse(streamedReply())).toEqual([
      new ReasoningDelta(0, "Thinking "),
      new ReasoningDelta(0, "about it."),
      new PartDone(0, REASONING_PART),
      new TextDelta(1, "Hello"),
      new TextDelta(1, " there."),
      new PartDone(1, TEXT_PART),
      new UsageReport(USAGE_COUNTS),
      finish("stop"),
    ]);
  });

  test("plain reply collects to the expected response", async () => {
    const response = await collect(await parse(sseEventsFor(reply([REASONING, MESSAGE]))));
    expect(response).toEqual(expected([REASONING_PART, TEXT_PART], "stop"));
    expect(response.message.protocol).toBe("openai_responses");
    expect(response.message.text()).toBe("Hello there.");
  });

  test("tool reply collects to the expected response", async () => {
    const response = await collect(await parse(sseEventsFor(reply([REASONING, CALL]))));
    expect(response).toEqual(expected([REASONING_PART, CALL_PART], "tool_calls"));
    expect(response.message.toolCalls()).toEqual([CALL_PART]);
  });

  test("multipart reply collects to the expected response", async () => {
    const commentary: JsonObject = {
      id: "msg_c",
      type: "message",
      role: "assistant",
      phase: "commentary",
      content: [{ type: "output_text", text: "Let me check.", annotations: [] }],
    };
    const final: JsonObject = {
      id: "msg_f",
      type: "message",
      role: "assistant",
      phase: "final_answer",
      content: [
        { type: "output_text", text: "Part A.", annotations: [] },
        { type: "output_text", text: "Part B.", annotations: [] },
      ],
    };
    const response = await collect(await parse(sseEventsFor(reply([commentary, final]))));
    const record = (id: string, phase: string) => replayRecord(KEY, { id, phase, annotations: [] });
    expect(response).toEqual(
      expected(
        [
          new TextPart("Let me check.", record("msg_c", "commentary")),
          new TextPart("Part A.", record("msg_f", "final_answer")),
          new TextPart("Part B.", record("msg_f", "final_answer")),
        ],
        "stop",
      ),
    );
  });

  test("refusal reply collects to the expected response", async () => {
    const item = { ...MESSAGE, content: [{ type: "refusal", refusal: "I can't help." }] };
    const response = await collect(await parse(sseEventsFor(reply([item]))));
    expect(response).toEqual(expected([new RefusalPart("I can't help.", replayRecord(KEY, { id: "msg_1" }))], "stop"));
  });

  test("a function call only in the terminal output makes the finish tool_calls", async () => {
    const events = await parse([created(), completed([CALL])]);
    expect(events.at(-1)).toEqual(finish("tool_calls"));
  });

  test("reasoning summary deltas are separated by newlines", async () => {
    const item = {
      ...REASONING,
      summary: [
        { type: "summary_text", text: "First." },
        { type: "summary_text", text: "Second." },
      ],
    };
    const events = await parse([
      added(0, { ...item, summary: [] }),
      summaryDelta(0, "First.", 0),
      summaryDelta(0, "Second.", 1),
      completed([item]),
    ]);
    const deltas = events.filter((event) => event instanceof ReasoningDelta);
    expect(deltas.map((delta) => delta.text).join("")).toBe("First.\nSecond.");
  });

  test("a done reasoning item joins its summaries with newlines", async () => {
    const item: JsonObject = {
      id: "rs_1",
      type: "reasoning",
      summary: [
        { type: "summary_text", text: "First." },
        { type: "summary_text", text: "Second." },
      ],
    };
    const events = await parse([done(0, item), completed([item])]);
    expect(events[0]).toEqual(new PartDone(0, new ReasoningPart("First.\nSecond.", replayRecord(KEY, { item }))));
  });

  test("a done item's null members are dropped from its record", async () => {
    const events = await parse([done(0, { ...REASONING, status: null, content: null }), completed()]);
    expect(events[0]).toEqual(new PartDone(0, REASONING_PART));
  });

  test("a function call streams tool call deltas", async () => {
    const events = await parse([
      created(),
      added(0, { ...CALL, arguments: "", status: "in_progress" }),
      argumentsDelta(0, '{"value"'),
      argumentsDelta(0, ": 3}"),
      { type: "response.function_call_arguments.done", output_index: 0, arguments: '{"value": 3}' },
      done(0, CALL),
      completed([CALL]),
    ]);
    expect(events.slice(0, 4)).toEqual([
      new ToolCallDelta({ index: 0, id: "call_1", name: "double", arguments: "" }),
      new ToolCallDelta({ index: 0, id: null, name: null, arguments: '{"value"' }),
      new ToolCallDelta({ index: 0, id: null, name: null, arguments: ": 3}" }),
      new PartDone(0, CALL_PART),
    ]);
    expect(events.at(-1)).toEqual(finish("tool_calls"));
  });

  test("the done item is the record's item", async () => {
    // The done item has fields the added item lacked (status, encrypted content).
    const events = await parse([
      added(0, { ...REASONING, summary: [], encrypted_content: null }),
      done(0, REASONING),
      added(1, { ...CALL, arguments: "", status: "in_progress" }),
      done(1, CALL),
      completed([REASONING, CALL]),
    ]);
    const parts = (await collect(events)).message.parts;
    expect(parts[0]!.providerPayload).toEqual(replayRecord(KEY, { item: REASONING }));
    expect(parts[1]!.providerPayload).toEqual(replayRecord(KEY, { item: CALL }));
    expect((parts[1]!.providerPayload!["item"] as JsonObject)["status"]).toBe("completed");
  });

  test("a two-part message gives two parts with id, annotations and phase in their records", async () => {
    const annotation = { type: "url_citation", url: "https://example.com", start_index: 0n, title: null };
    const item: JsonObject = {
      id: "msg_2",
      type: "message",
      role: "assistant",
      phase: "final_answer",
      content: [
        { type: "output_text", text: "Part A.", annotations: [annotation] },
        { type: "output_text", text: "Part B." },
      ],
    };
    const events = await parse([
      added(0, started(item)),
      textDelta(0, 0, "Part A.", "msg_2"),
      textDelta(0, 1, "Part B.", "msg_2"),
      done(0, item),
      completed([item]),
    ]);
    const kept = { type: "url_citation", url: "https://example.com", start_index: 0n };
    const ids = { id: "msg_2", phase: "final_answer" };
    expect(events.slice(0, 4)).toEqual([
      new TextDelta(0, "Part A."),
      new TextDelta(1, "Part B."),
      new PartDone(0, new TextPart("Part A.", replayRecord(KEY, { ...ids, annotations: [kept] }))),
      new PartDone(1, new TextPart("Part B.", replayRecord(KEY, { ...ids, annotations: [] }))),
    ]);
  });

  test("a refusal is collected into a refusal part", async () => {
    const item = { ...MESSAGE, content: [{ type: "refusal", refusal: "I can't help with that." }] };
    const where = { output_index: 0, content_index: 0, item_id: "msg_1" };
    const events = await parse([
      added(0, started(item)),
      { type: "response.refusal.delta", ...where, delta: "I can't " },
      { type: "response.refusal.delta", ...where, delta: "help with that." },
      { type: "response.refusal.done", ...where, refusal: "I can't help with that." },
      done(0, item),
      completed([item]),
    ]);
    const part = new RefusalPart("I can't help with that.", replayRecord(KEY, { id: "msg_1" }));
    expect(partsDone(events)).toEqual([new PartDone(0, part)]);
    const response = await collect(events);
    expect(response.message.parts).toEqual([part]);
    expect(response.message.text()).toBe("");
  });

  test("part indexes are dense in first-seen order", async () => {
    const events = await parse([
      added(0, { ...REASONING, summary: [] }),
      summaryDelta(0, "Thinking about it."),
      added(1, started(MESSAGE)),
      textDelta(1, 0, "Hello there."),
      added(2, { ...CALL, arguments: "" }),
      // Reasoning and text resume after the call has started.
      summaryDelta(0, ""),
      argumentsDelta(2, '{"value": 3}'),
      textDelta(1, 0, ""),
      done(2, CALL),
      done(0, REASONING),
      done(1, MESSAGE),
      completed([REASONING, MESSAGE, CALL]),
    ]);
    const indexed = events.filter((event) => "index" in event).map((event) => [event.type, event.index]);
    expect(indexed).toEqual([
      ["ReasoningDelta", 0],
      ["TextDelta", 1],
      ["ToolCallDelta", 2],
      ["ReasoningDelta", 0],
      ["ToolCallDelta", 2],
      ["TextDelta", 1],
      ["PartDone", 2],
      ["PartDone", 0],
      ["PartDone", 1],
    ]);
    expect((await collect(events)).message.parts).toEqual([REASONING_PART, TEXT_PART, CALL_PART]);
  });

  test("invalid arguments keep the raw text", async () => {
    const item = { ...CALL, arguments: '{"value": 3' };
    const events = await parse([
      added(0, { ...item, arguments: "" }),
      argumentsDelta(0, '{"value": 3'),
      done(0, item),
      completed([item]),
    ]);
    const call = (await collect(events)).message.toolCalls()[0]!;
    expect(call.arguments).toBeNull();
    expect(call.rawArguments).toBe('{"value": 3');
    expect(call.providerPayload).toEqual(replayRecord(KEY, { item }));
  });

  test("empty arguments are an empty object", async () => {
    const item = { ...CALL, arguments: "" };
    const events = await parse([done(0, item), completed([item])]);
    const record = replayRecord(KEY, { item });
    const call = new ToolCallPart({ id: "call_1", name: "double", arguments: {}, providerPayload: record });
    expect(events[0]).toEqual(new PartDone(0, call));
  });

  test("arguments accept raw control characters in strings", async () => {
    const item = { ...CALL, arguments: '{"text": "a\tb"}' };
    const events = await parse([done(0, item), completed([item])]);
    expect(((events[0] as PartDone).part as ToolCallPart).arguments).toEqual({ text: "a\tb" });
  });

  test("unknown events are ignored", async () => {
    const events = await parse([
      created(),
      { type: "response.in_progress", response: { id: "resp_1" } },
      { type: "response.something_new", output_index: 0 },
      { type: "keepalive" },
      added(0, started(MESSAGE)),
      textDelta(0, 0, "Hello there."),
      done(0, MESSAGE),
      completed([MESSAGE]),
    ]);
    expect(events).toEqual([
      new TextDelta(0, "Hello there."),
      new PartDone(0, TEXT_PART),
      new UsageReport(USAGE_COUNTS),
      finish("stop"),
    ]);
  });

  test("unknown output items and content types are skipped in a done event", async () => {
    const search = { type: "web_search_call", id: "ws_1" };
    const item = { ...MESSAGE, content: [{ type: "mystery" }, ...(MESSAGE["content"] as JsonValue[])] };
    const events = await parse([added(0, search), done(0, search), done(1, item), completed([search, item])]);
    expect(partsDone(events)).toEqual([new PartDone(0, TEXT_PART)]);
  });

  test("unknown output items and content types are skipped in the terminal output", async () => {
    const search = { type: "web_search_call", id: "ws_1" };
    const item = { ...MESSAGE, content: [{ type: "mystery" }, ...(MESSAGE["content"] as JsonValue[])] };
    const events = await parse([added(0, search), added(1, started(item)), completed([search, item])]);
    expect(partsDone(events)).toEqual([new PartDone(0, TEXT_PART)]);
  });

  test("a non-string type is ignored", async () => {
    for (const type of [[1], { a: 1 }, 3]) {
      const malformed = new SSEEvent({ data: encodeBody({ type, output_index: 0 }) });
      expect(await parse([created(), malformed, completed()])).toEqual([new UsageReport(USAGE_COUNTS), finish("stop")]);
    }
  });

  test("the type falls back to the SSE event field", async () => {
    const { type: _, ...data } = completed();
    const frame = new SSEEvent({ data: encodeBody(data), event: "response.completed" });
    expect(await parse([frame])).toEqual([new UsageReport(USAGE_COUNTS), finish("stop")]);
  });

  test("[DONE] ends the stream", async () => {
    const error = await failure([created(), new SSEEvent({ data: "[DONE]" }), completed()]);
    expect(error.kind).toBe("incomplete");
  });
});

describe("terminal output", () => {
  test("an empty first content part keeps its slot", async () => {
    const item = {
      ...MESSAGE,
      content: [
        { type: "output_text", text: "", annotations: [] },
        { type: "output_text", text: "Second.", annotations: [] },
      ],
    };
    const events = await parse([
      added(0, started(item)),
      contentPartAdded(0, 0),
      contentPartAdded(0, 1),
      textDelta(0, 1, "Second."),
      done(0, item),
      completed([item]),
    ]);
    const record = replayRecord(KEY, { id: "msg_1", annotations: [] });
    expect((await collect(events)).message.parts).toEqual([new TextPart("", record), new TextPart("Second.", record)]);
  });

  test("an item without a done event takes its part from the terminal output", async () => {
    const events = await parse([
      created(),
      added(0, started(MESSAGE)),
      textDelta(0, 0, "Hello there."),
      completed([MESSAGE]),
    ]);
    expect(events.slice(0, 2)).toEqual([new TextDelta(0, "Hello there."), new PartDone(0, TEXT_PART)]);
    expect(await collect(events)).toEqual(expected([TEXT_PART], "stop"));
  });

  test("an item missing from the terminal output keeps its delta-built part", async () => {
    const events = await parse([
      added(0, started(MESSAGE)),
      textDelta(0, 0, "Hello "),
      textDelta(0, 0, "there."),
      completed(),
    ]);
    expect(partsDone(events)).toEqual([]);
    expect((await collect(events)).message.parts).toEqual([new TextPart("Hello there.")]);
  });

  test("a refusal missing from the terminal output is built from its deltas", async () => {
    const item = { ...MESSAGE, phase: "final_answer" };
    const events = await parse([
      added(0, started(item)),
      refusalDelta(0, 0, "I can't "),
      refusalDelta(0, 0, "help."),
      completed(),
    ]);
    const part = new RefusalPart("I can't help.", replayRecord(KEY, { id: "msg_1", phase: "final_answer" }));
    expect(partsDone(events)).toEqual([new PartDone(0, part)]);
  });

  test("a refusal missing from the terminal output takes its done text", async () => {
    const events = await parse([
      added(0, started(MESSAGE)),
      refusalDelta(0, 0, "I can't"),
      { type: "response.refusal.done", output_index: 0, content_index: 0, refusal: "I can't help." },
      completed(),
    ]);
    const part = new RefusalPart("I can't help.", replayRecord(KEY, { id: "msg_1" }));
    expect(partsDone(events)).toEqual([new PartDone(0, part)]);
  });

  test("arguments on the added item are not counted twice", async () => {
    // The added item's arguments are ignored; the deltas carry the same text.
    const events = await parse([
      added(0, { ...CALL, status: "in_progress" }),
      argumentsDelta(0, '{"value"'),
      argumentsDelta(0, ": 3}"),
      completed(),
    ]);
    expect(events[0]).toEqual(new ToolCallDelta({ index: 0, id: "call_1", name: "double", arguments: "" }));
    expect((await collect(events)).message.toolCalls()[0]!.arguments).toEqual({ value: 3n });
  });
});

describe("item errors", () => {
  test("an output item that isn't an object", async () => {
    const error = await failure([added(0, started(MESSAGE)), completed([3])]);
    expect(error.kind).toBe("protocol");
    expect(error.message).toBe("openai response has an output item that is not an object");
  });

  test("message content that isn't a list", async () => {
    const error = await failure([done(0, { ...MESSAGE, content: "Hello" }), completed()]);
    expect(error.kind).toBe("protocol");
    expect(error.message).toBe("openai response message content is not a list");
  });

  test("output_text without string text", async () => {
    const error = await failure([done(0, { ...MESSAGE, content: [{ type: "output_text", text: 3 }] }), completed()]);
    expect(error.kind).toBe("protocol");
    expect(error.message).toBe("openai response output_text has no text");
  });

  test("refusal without string refusal", async () => {
    const error = await failure([done(0, { ...MESSAGE, content: [{ type: "refusal" }] }), completed()]);
    expect(error.kind).toBe("protocol");
    expect(error.message).toBe("openai response refusal has no text");
  });

  test.each(["call_id", "name", "arguments"])("a function_call without string %s", async (field) => {
    const without = Object.fromEntries(Object.entries(CALL).filter(([key]) => key !== field));
    for (const item of [{ ...CALL, [field]: 3 }, without]) {
      const error = await failure([done(0, item), completed()]);
      expect(error.kind).toBe("protocol");
      expect(error.message).toBe("openai response function_call needs call_id, name and arguments");
    }
  });

  test("a part constructor's MessageFormatError", async () => {
    install({ ...BUILTIN, maxValueNesting: 3 });
    const error = await failure([done(0, REASONING), completed()]);
    expect(error.kind).toBe("protocol");
    expect(error.message).toMatch(/^openai response has an invalid reasoning item: .*nests deeper than 3 levels$/);
  });

  describe.each(ITEM_EVENTS)("%s", (type) => {
    test("without an output_index", async () => {
      const error = await failure([{ type, content_index: 0, delta: "Hi" }, completed()]);
      expect(error.kind).toBe("protocol");
      expect(error.message).toBe(`openai stream event ${type} has no output_index`);
    });

    test.each(["1.0", "true"])("with output_index %s", async (token) => {
      const data = encodeBody({ type, content_index: 0, delta: "Hi" }).replace(/^\{/, `{"output_index":${token},`);
      const error = await failure([new SSEEvent({ data, event: type }), completed()]);
      expect(error.kind).toBe("protocol");
      expect(error.message).toBe(`openai stream event ${type} has no output_index`);
    });
  });
});

describe("usage and finish", () => {
  test.each(["flex", "priority", "default", null])("usage under service tier %p", async (tier) => {
    const events = await parse([completed([MESSAGE], { service_tier: tier })]);
    expect(events.at(-2)).toEqual(new UsageReport(USAGE_COUNTS));
  });

  test("usage maps cache and reasoning counts", async () => {
    const usage = {
      input_tokens: 100,
      input_tokens_details: { cached_tokens: 40, cache_write_tokens: 7 },
      output_tokens: 50,
      output_tokens_details: { reasoning_tokens: 20 },
      total_tokens: 150,
    };
    const events = await parse([completed([], { usage })]);
    const counts = { inputTokens: 100, outputTokens: 50, totalTokens: 150 };
    const details = { cacheReadTokens: 40, cacheCreationTokens: 7, reasoningTokens: 20 };
    expect(events.at(-2)).toEqual(new UsageReport(new Usage({ ...counts, ...details })));
  });

  test("usage defaults: null or missing counts are 0, a missing total is their sum", async () => {
    const cases: [JsonObject, Usage][] = [
      [{ input_tokens: 3, output_tokens: null }, new Usage({ inputTokens: 3, outputTokens: 0, totalTokens: 3 })],
      [{ output_tokens: 4 }, new Usage({ inputTokens: 0, outputTokens: 4, totalTokens: 4 })],
      [
        { input_tokens: 3, output_tokens: 4, total_tokens: null },
        new Usage({ inputTokens: 3, outputTokens: 4, totalTokens: 7 }),
      ],
      [{}, new Usage({ inputTokens: 0, outputTokens: 0, totalTokens: 0 })],
    ];
    for (const [usage, counts] of cases) {
      const events = await parse([completed([], { usage })]);
      expect(events.at(-2)).toEqual(new UsageReport(counts));
    }
  });

  test("null or absent usage gives no usage report", async () => {
    const absent = completed();
    delete (absent["response"] as JsonObject)["usage"];
    for (const terminal of [completed([], { usage: null }), absent]) {
      const events = await parse([terminal]);
      expect(events).toEqual([finish("stop")]);
      expect((await collect(events)).usage).toBeNull();
    }
  });

  test("the response id comes from response.created when the terminal lacks one", async () => {
    const final = completed([MESSAGE]);
    delete (final["response"] as JsonObject)["id"];
    const events = await parse([created("resp_9"), final]);
    expect(events.at(-1)).toEqual(finish("stop", "resp_9"));
  });

  test("the terminal response's id wins over response.created", async () => {
    const events = await parse([created("resp_9"), completed()]);
    expect(events.at(-1)).toEqual(finish("stop"));
  });

  test("a reply without an id finishes with an empty message record", async () => {
    const final = completed();
    delete (final["response"] as JsonObject)["id"];
    const events = await parse([final]);
    const providerPayload = replayRecord(KEY, {});
    expect(events.at(-1)).toEqual(new Finish({ reason: "stop", protocol: PROTOCOL, providerPayload }));
  });

  test.each([
    ["api key", apiKey],
    ["subscription", subscription],
  ])("max_output_tokens finishes with length on the %s route, even with calls", async (_, protocol) => {
    const events = await parse(
      [
        created(),
        added(0, { ...CALL, arguments: "" }),
        argumentsDelta(0, '{"value": 3}'),
        incomplete("max_output_tokens", [CALL]),
      ],
      protocol(),
    );
    expect(events.at(-2)).toEqual(new UsageReport(USAGE_COUNTS));
    expect(events.at(-1)).toEqual(finish("length"));
  });

  test("an incomplete reply streamed as the API sends it finishes with length", async () => {
    const response = reply([MESSAGE], { incomplete_details: { reason: "max_output_tokens" } });
    const events = await parse(sseEventsFor(response, "incomplete"));
    expect(events.at(-1)).toEqual(finish("length"));
  });

  test("response.done with status completed finishes like response.completed", async () => {
    const events = await parse([added(0, started(MESSAGE)), { type: "response.done", response: reply([MESSAGE]) }]);
    expect(events).toEqual([new PartDone(0, TEXT_PART), new UsageReport(USAGE_COUNTS), finish("stop")]);
  });

  test("the parser returns at the terminal event", async () => {
    const events = await parse([completed(), { type: "error", code: "server_error", message: "late" }]);
    expect(events.at(-1)).toEqual(finish("stop"));
  });
});
