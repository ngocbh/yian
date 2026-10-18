/**
 * The scripted model, a port of the ScriptedModel part of `tests/llm/test_testing.py`, plus an
 * aborted signal ending a stream.
 */

import { describe, expect, test } from "bun:test";

import { collect, type Event, EventStream, Finish, PartDone, ReasoningDelta, TextDelta, ToolCallDelta, UsageReport } from "../../src/llm/events.ts";
import { Message, ReasoningPart, RefusalPart, Request, Response, TextPart, ToolCallPart, Usage } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";

/** Return a request of one user message with this text. */
function ask(text: string): Request {
  return new Request({ messages: [new Message({ role: "user", parts: [new TextPart(text)] })] });
}

const REQUEST = ask("hi");

/** Return a reply with reasoning, text, a parsed and an unparsed tool call, and a refusal. */
function fullResponse(): Response {
  const message = new Message({
    role: "assistant",
    parts: [
      new ReasoningPart("thinking", { signature: "s" }),
      new TextPart("hello"),
      new ToolCallPart({ id: "c1", name: "search", arguments: { q: "x", n: 1n } }),
      new ToolCallPart({ id: "c2", name: "broken", arguments: null, rawArguments: "{not json" }),
      new RefusalPart("I can't help.", { id: "msg_1" }),
    ],
    protocol: "openai_responses",
    providerPayload: { id: "msg_1" },
  });
  return new Response({
    message,
    usage: new Usage({ inputTokens: 3, outputTokens: 4, totalTokens: 7 }),
    finishReason: "tool_calls",
    responseId: "r1",
  });
}

/** Return every event `stream` still yields. */
async function drain(stream: EventStream): Promise<Event[]> {
  const out: Event[] = [];
  for await (const event of stream) out.push(event);
  return out;
}

/** Return what `promise` rejects with; fail the test if it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

describe("ScriptedModel", () => {
  test("a response streams as events that collect back to it", async () => {
    const reply = fullResponse();
    const model = new ScriptedModel([reply]);

    const stream = model.stream(REQUEST);
    const events = await drain(stream);

    expect(stream).toBeInstanceOf(EventStream);
    const parts = reply.message.parts;
    expect(events).toEqual([
      new ReasoningDelta(0, "thinking"),
      new PartDone(0, parts[0]!),
      new TextDelta(1, "hello"),
      new PartDone(1, parts[1]!),
      new ToolCallDelta({ index: 2, id: "c1", name: "search", arguments: '{"n":1,"q":"x"}' }),
      new PartDone(2, parts[2]!),
      new ToolCallDelta({ index: 3, id: "c2", name: "broken", arguments: "{not json" }),
      new PartDone(3, parts[3]!),
      new PartDone(4, parts[4]!),
      new UsageReport(new Usage({ inputTokens: 3, outputTokens: 4, totalTokens: 7 })),
      new Finish({ reason: "tool_calls", responseId: "r1", protocol: "openai_responses", providerPayload: { id: "msg_1" } }),
    ]);
    expect(await collect(events)).toEqual(reply);
  });

  test("collecting the stream gives the reply and records the request", async () => {
    const reply = fullResponse();
    const model = new ScriptedModel([reply]);

    expect(await collect(model.stream(REQUEST))).toEqual(reply);
    expect(model.requests).toEqual([REQUEST]);
  });

  test("a raw event list streams as given", async () => {
    const events = [new TextDelta(0, "a"), new TextDelta(0, "b"), new Finish({ reason: "stop" })];
    const model = new ScriptedModel([events]);

    expect(await drain(model.stream(REQUEST))).toEqual(events);
  });

  test("a bare error is thrown by the first next()", async () => {
    const down = new Error("down");
    const model = new ScriptedModel([down]);

    const stream = model.stream(REQUEST);

    expect(model.requests).toEqual([REQUEST]);
    expect(await rejection(stream.next())).toBe(down);
  });

  test("an error in a list is thrown at its position and closes the stream", async () => {
    const cut = new Error("cut");
    const model = new ScriptedModel([[new TextDelta(0, "par"), cut]]);
    const stream = model.stream(REQUEST);

    expect(await stream.next()).toEqual({ done: false, value: new TextDelta(0, "par") });
    expect(await rejection(stream.next())).toBe(cut);
    expect(model.closed).toBe(1);
  });

  test("records each request and counts closed streams", async () => {
    const second = ask("again");
    const model = new ScriptedModel([fullResponse(), fullResponse(), fullResponse()], { provider: "p", model: "m" });
    expect([model.provider, model.model]).toEqual(["p", "m"]);

    const stream = model.stream(REQUEST);
    expect(model.closed).toBe(0);
    await stream.close();
    await stream.close();
    await collect(model.stream(second));
    await drain(model.stream(REQUEST));

    expect(model.requests).toEqual([REQUEST, second, REQUEST]);
    expect(model.closed).toBe(3);
  });

  test("names default to scripted", () => {
    const model = new ScriptedModel([]);
    expect([model.provider, model.model]).toEqual(["scripted", "scripted"]);
  });

  test("running out of replies throws at once", () => {
    const model = new ScriptedModel([]);
    expect(() => model.stream(REQUEST)).toThrow("no reply");
  });

  test("an aborted signal throws its reason before the next event", async () => {
    const model = new ScriptedModel([fullResponse()]);
    const controller = new AbortController();
    const stream = model.stream(REQUEST, { signal: controller.signal });

    expect(await stream.next()).toEqual({ done: false, value: new ReasoningDelta(0, "thinking") });
    const reason = new Error("stop now");
    controller.abort(reason);

    expect(await rejection(stream.next())).toBe(reason);
    expect(model.closed).toBe(1);
  });
});
