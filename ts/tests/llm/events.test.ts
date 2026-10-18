/**
 * Stream events: argument parsing, collecting a reply and the owning stream, a port of
 * `tests/llm/test_events.py`,
 * plus JavaScript's own cases: ints of any size, raw control characters inside strings, async
 * sources, and the rule that a stream ends with exactly one `Finish`.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import {
  collect,
  type Event,
  EventStream,
  Finish,
  parseArguments,
  PartDone,
  ReasoningDelta,
  ReplyAssembler,
  TextDelta,
  ToolCallDelta,
  UsageReport,
} from "../../src/llm/events.ts";
import { Message, ReasoningPart, TextPart, ToolCallPart, Usage } from "../../src/llm/messages.ts";

afterEach(() => install(BUILTIN));

/** Return what `promise` rejects with; fail the test if it resolves. */
async function rejection(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

/** Return a tool-call delta. */
function call(index: number, id: string | null, name: string | null, args: string): ToolCallDelta {
  return new ToolCallDelta({ index, id, name, arguments: args });
}

/** Return a `Usage` with these counts. */
function usage(inputTokens: number, outputTokens: number, totalTokens: number): Usage {
  return new Usage({ inputTokens, outputTokens, totalTokens });
}

const STOP = new Finish({ reason: "stop" });

/** Return every event `stream` still yields. */
async function drain(stream: EventStream): Promise<Event[]> {
  const out: Event[] = [];
  for await (const event of stream) out.push(event);
  return out;
}

describe("parseArguments", () => {
  test("empty text is an empty object", () => {
    expect(parseArguments("")).toEqual([{}, null]);
  });

  test("object", () => {
    expect(parseArguments('{"a": [1, 2.5, null, true]}')).toEqual([{ a: [1n, 2.5, null, true] }, null]);
  });

  test.each(["[1, 2]", "{bad", '{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', '"s"', "1e400"])(
    "rejects %p",
    (text) => {
      expect(parseArguments(text)).toEqual([null, text]);
    },
  );

  test("a 400-digit int parses as a bigint", () => {
    const digits = "9".repeat(400);
    expect(parseArguments(`{"n": -${digits}}`)).toEqual([{ n: -BigInt(digits) }, null]);
  });

  test("nesting limit", () => {
    install({ ...BUILTIN, maxValueNesting: 3 });
    expect(parseArguments('{"a": {"b": [1]}}')).toEqual([{ a: { b: [1n] } }, null]);
    expect(parseArguments('{"a": {"b": [[1]]}}')).toEqual([null, '{"a": {"b": [[1]]}}']);
  });

  test.each(['{"a": "\\ud800"}', '{"\\udfff": 1}', '{"a": "\ud800"}'])("unpaired surrogate %p is a failure", (text) => {
    expect(parseArguments(text)).toEqual([null, text]);
  });

  test("escaped surrogate pair parses", () => {
    expect(parseArguments('{"a": "\\ud83d\\ude00"}')).toEqual([{ a: "\u{1f600}" }, null]);
  });

  test("deep brackets fail without a stack overflow", () => {
    let text = "[".repeat(100_000);
    expect(parseArguments(text)).toEqual([null, text]);
    text = '{"a":'.repeat(100_000);
    expect(parseArguments(text)).toEqual([null, text]);
  });

  test("raw tab and newline inside a string parse", () => {
    expect(parseArguments('{"a": "x\ty\nz"}')).toEqual([{ a: "x\ty\nz" }, null]);
  });

  test("raw newlines between tokens parse", () => {
    expect(parseArguments('{\n"a":\r\n1\n}\n')).toEqual([{ a: 1n }, null]);
  });

  test("a raw control character outside a string fails", () => {
    const text = '{"a": 1\u0001}';
    expect(parseArguments(text)).toEqual([null, text]);
  });

  test("an escaped backslash before a closing quote ends the string", () => {
    expect(parseArguments('{"a": "\\\\", "b": "\t"}')).toEqual([{ a: "\\", b: "\t" }, null]);
    // Read as still inside the string, the newline after the closing quote would be escaped and fail.
    expect(parseArguments('{"a": "\\\\"\n}')).toEqual([{ a: "\\" }, null]);
  });

  test("a duplicate key fails", () => {
    const text = '{"a": 1, "a": 2}';
    expect(parseArguments(text)).toEqual([null, text]);
  });
});

describe("collect", () => {
  test("interleaved text and tool call", async () => {
    const response = await collect([
      new TextDelta(0, "Hel"),
      call(1, "call_1", null, '{"q": '),
      new TextDelta(0, "lo"),
      call(1, null, "search", '"x"}'),
      call(1, "ignored", "ignored", ""),
      new Finish({ reason: "tool_calls" }),
    ]);
    expect(response.message).toEqual(
      new Message({
        role: "assistant",
        parts: [new TextPart("Hello"), new ToolCallPart({ id: "call_1", name: "search", arguments: { q: "x" } })],
      }),
    );
    expect(response.finishReason).toBe("tool_calls");
  });

  test("parts ordered by index", async () => {
    const response = await collect([
      new ReasoningDelta(10, "b"),
      new TextDelta(9, "a"),
      new ReasoningDelta(10, "c"),
      STOP,
    ]);
    // Numeric order: 9 before 10, which a text sort would reverse.
    expect(response.message.parts).toEqual([new TextPart("a"), new ReasoningPart("bc")]);
  });

  test("PartDone wins", async () => {
    const final = new ReasoningPart("full", { signature: "s" });
    const response = await collect([new ReasoningDelta(0, "fu"), new PartDone(0, final), new ReasoningDelta(0, "ll"), STOP]);
    expect(response.message.parts).toEqual([final]);
  });

  test("PartDone supplies the tool call id", async () => {
    const done = new ToolCallPart({ id: "assigned", name: "f", arguments: {} });
    const response = await collect([call(0, null, "f", "{}"), new PartDone(0, done), STOP]);
    expect(response.message.toolCalls()).toEqual([done]);
  });

  test("unparsable arguments keep the raw text", async () => {
    const response = await collect([call(0, "c", "f", "{bad"), STOP]);
    expect(response.message.parts).toEqual([new ToolCallPart({ id: "c", name: "f", arguments: null, rawArguments: "{bad" })]);
  });

  test("empty arguments are an empty object", async () => {
    const response = await collect([call(0, "c", "f", ""), STOP]);
    expect(response.message.parts).toEqual([new ToolCallPart({ id: "c", name: "f", arguments: {} })]);
  });

  test("the last usage wins", async () => {
    const response = await collect([
      new UsageReport(usage(1, 1, 2)),
      new TextDelta(0, "x"),
      new UsageReport(usage(3, 4, 7)),
      new Finish({ reason: "stop", responseId: "r1" }),
    ]);
    expect(response.usage).toEqual(usage(3, 4, 7));
    expect(response.finishReason).toBe("stop");
    expect(response.responseId).toBe("r1");
  });

  test("the finish sets protocol and payload", async () => {
    const response = await collect([
      new TextDelta(0, "x"),
      new Finish({ reason: "stop", protocol: "openai_chat", providerPayload: { system_fingerprint: "fp" } }),
    ]);
    expect(response.message.protocol).toBe("openai_chat");
    expect(response.message.providerPayload).toEqual({ system_fingerprint: "fp" });
  });

  test("mixed delta kinds name the index", async () => {
    let error = await rejection(collect([new TextDelta(0, "a"), new ReasoningDelta(0, "b"), STOP]));
    expect(error.message).toBe("index 0 mixes TextDelta and ReasoningDelta");
    error = await rejection(collect([call(2, "c", "f", ""), new TextDelta(2, "a"), STOP]));
    expect(error.message).toBe("index 2 mixes ToolCallDelta and TextDelta");
  });

  test("a tool call without an id fails", async () => {
    const error = await rejection(collect([call(3, null, "f", "{}"), STOP]));
    expect(error).toBeInstanceOf(ProviderError);
    expect((error as ProviderError).kind).toBe("protocol");
    expect(error.message).toBe("tool call 3 has no id");
  });

  test("a tool call without a name fails", async () => {
    const error = await rejection(collect([call(0, "c", null, "{}"), STOP]));
    expect(error).toBeInstanceOf(ProviderError);
    expect((error as ProviderError).kind).toBe("protocol");
    expect(error.message).toBe("tool call 0 has no name");
  });

  test("a non-event fails and closes the source", async () => {
    const state: string[] = [];
    function* events(): Generator<unknown> {
      try {
        yield new TextDelta(0, "a");
        yield "not an event";
        yield new TextDelta(0, "b");
      } finally {
        state.push("closed");
      }
    }
    const error = await rejection(collect(events() as Iterable<Event>));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe("not a stream event: string");
    expect(state).toEqual(["closed"]);
  });

  test("the empty stream is incomplete", async () => {
    const error = await rejection(collect([]));
    expect(error).toBeInstanceOf(ProviderError);
    expect((error as ProviderError).kind).toBe("incomplete");
    expect((error as ProviderError).retryable).toBe(true);
    expect(error.message).toBe("the stream ended before its finish event");
  });

  test("deltas without a finish are incomplete", async () => {
    const error = await rejection(collect([new TextDelta(0, "a"), new UsageReport(usage(1, 1, 2))]));
    expect(error).toBeInstanceOf(ProviderError);
    expect((error as ProviderError).kind).toBe("incomplete");
  });

  test.each([
    ["a second Finish", STOP],
    ["a delta", new TextDelta(0, "b")],
  ])("%s after the finish fails and closes the source", async (_, late) => {
    const state: string[] = [];
    function* events(): Generator<Event> {
      try {
        yield new TextDelta(0, "a");
        yield STOP;
        yield late;
        yield new TextDelta(0, "c");
      } finally {
        state.push("closed");
      }
    }
    const error = await rejection(collect(events()));
    expect(error.message).toBe("event after finish");
    expect(state).toEqual(["closed"]);
  });

  test("an async generator source", async () => {
    async function* events(): AsyncGenerator<Event> {
      yield new TextDelta(0, "a");
      await Promise.resolve();
      yield new TextDelta(0, "b");
      yield STOP;
    }
    expect((await collect(events())).message.text()).toBe("ab");
  });
});

describe("ReplyAssembler.partial", () => {
  test("null before any event", () => {
    expect(new ReplyAssembler().partial()).toBeNull();
  });

  test("holds the parts that arrived", () => {
    const assembler = new ReplyAssembler();
    assembler.add(new TextDelta(0, "hi"));
    assembler.add(new ReasoningDelta(1, "why"));
    assembler.add(call(2, "c1", "f", '{"a": 1}'));
    assembler.add(call(3, "c2", null, "{"));
    expect(assembler.partial()).toEqual(
      new Message({
        role: "assistant",
        parts: [
          new TextPart("hi"),
          new ReasoningPart("why"),
          new ToolCallPart({ id: "c1", name: "f", arguments: { a: 1n } }),
        ],
      }),
    );
  });

  test("leaves out a part it can't build, without throwing", () => {
    const assembler = new ReplyAssembler();
    assembler.add(new TextDelta(0, "ok"));
    assembler.add(new TextDelta(1, "\ud800"));
    expect(assembler.partial()).toEqual(new Message({ role: "assistant", parts: [new TextPart("ok")] }));
  });
});

describe("EventStream", () => {
  test("close is idempotent and later iteration is empty", async () => {
    const closes: number[] = [];
    const stream = new EventStream([new TextDelta(0, "a")].values(), () => void closes.push(1));
    const first = stream.close();
    expect(stream.close()).toBe(first);
    await first;
    await stream.close();
    expect(closes).toEqual([1]);
    expect(await drain(stream)).toEqual([]);
  });

  test("closes when exhausted, once", async () => {
    const closes: number[] = [];
    const stream = new EventStream([new TextDelta(0, "a"), new TextDelta(0, "b")].values(), () => void closes.push(1));
    expect(await drain(stream)).toEqual([new TextDelta(0, "a"), new TextDelta(0, "b")]);
    expect(closes).toEqual([1]);
    await stream.close();
    expect(closes).toEqual([1]);
  });

  test("closes on await using exit", async () => {
    const closes: number[] = [];
    {
      await using stream = new EventStream([new TextDelta(0, "a"), new TextDelta(0, "b")].values(), () => {
        closes.push(1);
      });
      expect((await stream.next()).value).toEqual(new TextDelta(0, "a"));
    }
    expect(closes).toEqual([1]);
  });

  test("closes the inner async generator", async () => {
    const state: string[] = [];
    async function* events(): AsyncGenerator<Event> {
      try {
        yield new TextDelta(0, "a");
        yield new TextDelta(0, "b");
      } finally {
        state.push("closed");
      }
    }
    const stream = new EventStream(events());
    await stream.next();
    await stream.close();
    expect(state).toEqual(["closed"]);
  });

  test("closes when the inner iterator throws", async () => {
    const closes: number[] = [];
    async function* events(): AsyncGenerator<Event> {
      yield new TextDelta(0, "a");
      throw new ProviderError("connection", "dropped");
    }
    const stream = new EventStream(events(), () => void closes.push(1));
    expect((await stream.next()).value).toEqual(new TextDelta(0, "a"));
    const error = await rejection(stream.next());
    expect(error).toBeInstanceOf(ProviderError);
    expect(error.message).toBe("dropped");
    expect(closes).toEqual([1]);
    expect(await drain(stream)).toEqual([]);
    expect(closes).toEqual([1]);
  });

  test("collect reads an EventStream", async () => {
    const stream = new EventStream([new TextDelta(0, "a"), STOP].values());
    expect((await collect(stream)).message.text()).toBe("a");
  });

  test("break from for await closes it", async () => {
    const closes: number[] = [];
    const state: string[] = [];
    async function* events(): AsyncGenerator<Event> {
      try {
        yield new TextDelta(0, "a");
        yield new TextDelta(0, "b");
      } finally {
        state.push("closed");
      }
    }
    const stream = new EventStream(events(), () => void closes.push(1));
    for await (const event of stream) {
      expect(event).toEqual(new TextDelta(0, "a"));
      break;
    }
    expect(closes).toEqual([1]);
    expect(state).toEqual(["closed"]);
  });

  test("the callback runs once when the inner return() throws", async () => {
    const closes: number[] = [];
    const events: Iterator<Event> = {
      next: () => ({ done: false, value: new TextDelta(0, "a") }),
      return: () => {
        throw new Error("boom");
      },
    };
    const stream = new EventStream(events, () => void closes.push(1));
    expect((await rejection(stream.close())).message).toBe("boom");
    expect((await rejection(stream.close())).message).toBe("boom");
    expect(closes).toEqual([1]);
  });

  test("close ends a read that waits until an abort", async () => {
    const closes: number[] = [];
    const state: string[] = [];
    const controller = new AbortController();
    const signal = controller.signal;
    async function* events(): AsyncGenerator<Event> {
      try {
        await new Promise((_, reject) => signal.addEventListener("abort", () => reject(signal.reason)));
        yield new TextDelta(0, "never");
      } finally {
        state.push("closed");
      }
    }
    const stream = new EventStream(events(), () => {
      closes.push(1);
      controller.abort();
    });
    const pending = rejection(stream.next());
    await stream.close();
    expect((await pending).name).toBe("AbortError");
    expect(closes).toEqual([1]);
    expect(state).toEqual(["closed"]);
  });
});
