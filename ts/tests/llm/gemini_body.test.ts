/**
 * Tests for Gemini body building in `src/llm/protocols/gemini.ts`: the protocol object, the
 * contents (system, user, tool and assistant messages), tool results and their ids, and the media
 * and wrong-part refusals.
 */

import { describe, expect, spyOn, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import {
  MediaPart,
  Message,
  type Part,
  ReasoningPart,
  RefusalPart,
  type ReplayKey,
  Request,
  replayRecord,
  type Role,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from "../../src/llm/messages.ts";
import { Gemini, PROTOCOL } from "../../src/llm/protocols/gemini.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

const FLASH = "gemini-2.5-flash";
const KEY: ReplayKey = { protocol: PROTOCOL, provider: "google", model: FLASH };
const EMPTY_USER = "an empty user message was not sent; Gemini rejects empty content";

function message(role: Role, ...parts: Part[]): Message {
  return new Message({ role, parts });
}

function system(...texts: string[]): Message {
  return message("system", ...texts.map((text) => new TextPart(text)));
}

function user(...texts: string[]): Message {
  return message("user", ...texts.map((text) => new TextPart(text)));
}

/** Return this protocol's own assistant message: it carries this model's message record. */
function own(...parts: Part[]): Message {
  return new Message({ role: "assistant", parts, protocol: PROTOCOL, providerPayload: replayRecord(KEY, {}) });
}

function tool(...results: ToolResultPart[]): Message {
  return message("tool", ...results);
}

function text(value: string): TextPart {
  return new TextPart(value);
}

function result(callId: string, content: string, isError = false): ToolResultPart {
  return new ToolResultPart({ callId, name: "double", content, isError });
}

function call(id: string, args: JsonObject): ToolCallPart {
  return new ToolCallPart({ id, name: "double", arguments: args });
}

function gemini(): Gemini {
  return new Gemini({ model: FLASH, provider: "google" });
}

function bodyOf(messages: Message[]): JsonObject {
  return gemini().buildBody(new Request({ messages }));
}

/** Return the `contents` of a body built from `messages`. */
function contents(messages: Message[]): JsonValue {
  return bodyOf(messages)["contents"]!;
}

/** Return the `response` sent for a tool result with `content`. */
function response(content: string, isError = false): JsonValue {
  const sent = contents([tool(result("call_1", content, isError))]) as JsonObject[];
  const parts = sent[0]!["parts"] as JsonObject[];
  return (parts[0]!["functionResponse"] as JsonObject)["response"]!;
}

/** Run `run` with `process.emitWarning` stubbed; return the warnings it emitted. */
function warnings(run: () => unknown): unknown[][] {
  const emit = spyOn(process, "emitWarning").mockImplementation(() => {});
  try {
    run();
    return emit.mock.calls.map((args) => [...args]);
  } finally {
    emit.mockRestore();
  }
}

/** Assert `run` throws a `TypeError` with exactly `text`. */
function refuses(run: () => unknown, text: string): void {
  let error: unknown = null;
  try {
    run();
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toBe(text);
}

describe("the protocol object", () => {
  test("protocol name and framing", () => {
    const protocol = gemini();
    expect(PROTOCOL).toBe("gemini");
    expect(protocol.name).toBe("gemini");
    expect(protocol.framing).toBe("sse");
    expect(protocol.model).toBe(FLASH);
    expect(protocol.provider).toBe("google");
    expect(protocol.account).toBeNull();
    expect(new Gemini({ model: FLASH, provider: "google", account: "acct-1" }).account).toBe("acct-1");
  });
});

describe("system", () => {
  test("system and user give exactly the fixed body, with no model key", () => {
    expect(bodyOf([system("You are terse."), user("Say hello.")])).toEqual({
      contents: [{ role: "user", parts: [{ text: "Say hello." }] }],
      systemInstruction: { parts: [{ text: "You are terse." }] },
      safetySettings: [],
      generationConfig: { candidateCount: 1n },
    });
  });

  test("a system message with two text parts gives two parts", () => {
    expect(bodyOf([system("A.", "B."), user("Hi.")])["systemInstruction"]).toEqual({
      parts: [{ text: "A." }, { text: "B." }],
    });
  });

  test("a late system message joins after the leading one", () => {
    const body = bodyOf([system("S1."), user("Hi."), own(text("Hello.")), system("S2."), user("More.")]);
    expect(body["systemInstruction"]).toEqual({ parts: [{ text: "S1." }, { text: "S2." }] });
    expect(body["contents"]).toEqual([
      { role: "user", parts: [{ text: "Hi." }] },
      { role: "model", parts: [{ text: "Hello." }] },
      { role: "user", parts: [{ text: "More." }] },
    ]);
  });

  test("a late system message with no leading one is sent", () => {
    const body = bodyOf([user("Hi."), own(text("Hello.")), system("Late.")]);
    expect(body["systemInstruction"]).toEqual({ parts: [{ text: "Late." }] });
  });

  test("no system message sends no systemInstruction", () => {
    expect(Object.hasOwn(bodyOf([user("Hi.")]), "systemInstruction")).toBe(false);
  });

  test("an empty system text part is left out", () => {
    expect(bodyOf([system("", "A."), user("Hi.")])["systemInstruction"]).toEqual({ parts: [{ text: "A." }] });
    expect(Object.hasOwn(bodyOf([system(""), user("Hi.")]), "systemInstruction")).toBe(false);
  });
});

describe("user", () => {
  test("consecutive user messages are not merged", () => {
    expect(contents([user("A."), user("B.")])).toEqual([
      { role: "user", parts: [{ text: "A." }] },
      { role: "user", parts: [{ text: "B." }] },
    ]);
  });

  test("a multi-part user message is one content with several parts", () => {
    expect(contents([user("A.", "B.")])).toEqual([{ role: "user", parts: [{ text: "A." }, { text: "B." }] }]);
  });

  test("an empty user message is skipped with a warning, once per instance", () => {
    const protocol = gemini();
    const request = new Request({ messages: [user(""), message("user"), user("Hi.")] });
    let bodies: JsonObject[] = [];
    const emitted = warnings(() => {
      bodies = [protocol.buildBody(request), protocol.buildBody(request)];
    });
    expect(emitted).toEqual([[EMPTY_USER]]);
    for (const body of bodies) expect(body["contents"]).toEqual([{ role: "user", parts: [{ text: "Hi." }] }]);
    // A new instance warns again.
    expect(warnings(() => gemini().buildBody(request))).toEqual([[EMPTY_USER]]);
  });

  test("an empty text part is left out of a user message", () => {
    expect(contents([user("", "a")])).toEqual([{ role: "user", parts: [{ text: "a" }] }]);
  });
});

describe("tool results", () => {
  test("a JSON scalar or list is wrapped in output, a plain object is the response", () => {
    expect(response("6")).toEqual({ output: 6n });
    expect(response("true")).toEqual({ output: true });
    expect(response("[1, 2]")).toEqual({ output: [1n, 2n] });
    expect(response('{"result": 2}')).toEqual({ result: 2n });
    expect(response("null")).toEqual({ output: null });
    expect(response('"s"')).toEqual({ output: "s" });
  });

  test("text that is not plain JSON goes as output text", () => {
    expect(response("plain")).toEqual({ output: "plain" });
    expect(response("1e400")).toEqual({ output: "1e400" });
    expect(response('{"a":1,"a":2}')).toEqual({ output: '{"a":1,"a":2}' });
  });

  test("an error result has the same shape and no error field", () => {
    expect(response("boom", true)).toEqual({ output: "boom" });
    expect(response('{"result": 2}', true)).toEqual({ result: 2n });
  });

  test("each functionResponse carries the call id and the name", () => {
    expect(contents([tool(result("call_7", "6"))])).toEqual([
      { role: "user", parts: [{ functionResponse: { id: "call_7", name: "double", response: { output: 6n } } }] },
    ]);
  });

  test("consecutive tool messages become one user content", () => {
    expect(contents([tool(result("call_1", "1")), tool(result("call_2", "2"))])).toEqual([
      {
        role: "user",
        parts: [
          { functionResponse: { id: "call_1", name: "double", response: { output: 1n } } },
          { functionResponse: { id: "call_2", name: "double", response: { output: 2n } } },
        ],
      },
    ]);
  });

  test("a tool message then a user message are two contents", () => {
    expect(contents([tool(result("call_1", "1")), user("Next.")])).toEqual([
      { role: "user", parts: [{ functionResponse: { id: "call_1", name: "double", response: { output: 1n } } }] },
      { role: "user", parts: [{ text: "Next." }] },
    ]);
  });

  test("an orphan tool result is still sent", () => {
    // Gemini would refuse it; yian leaves that answer to the API.
    expect(contents([user("Hi."), own(text("Hello.")), tool(result("call_9", "6"))])).toEqual([
      { role: "user", parts: [{ text: "Hi." }] },
      { role: "model", parts: [{ text: "Hello." }] },
      { role: "user", parts: [{ functionResponse: { id: "call_9", name: "double", response: { output: 6n } } }] },
    ]);
  });
});

describe("assistant", () => {
  test("a text part is a text part", () => {
    expect(contents([user("Hi."), own(text("Hello."))])).toEqual([
      { role: "user", parts: [{ text: "Hi." }] },
      { role: "model", parts: [{ text: "Hello." }] },
    ]);
  });

  test("text, call, text keep their order; a call is a functionCall with its id", () => {
    const sent = contents([own(text("A."), call("call_1", { value: 3n }), text("B."))]);
    expect(sent).toEqual([
      {
        role: "model",
        parts: [
          { text: "A." },
          { functionCall: { id: "call_1", name: "double", args: { value: 3n } } },
          { text: "B." },
        ],
      },
    ]);
  });

  test("an invalid call goes with empty args", () => {
    const invalid = new ToolCallPart({ id: "call_1", name: "double", arguments: null, rawArguments: "{bad" });
    expect(contents([own(invalid)])).toEqual([
      { role: "model", parts: [{ functionCall: { id: "call_1", name: "double", args: {} } }] },
    ]);
  });

  test("args holding a bigint encode as digits", () => {
    const body = bodyOf([own(call("call_1", { value: 12345678901234567890n }))]);
    expect(encodeBody(body)).toContain('"args":{"value":12345678901234567890}');
  });

  test("the args are copied", () => {
    const args: JsonObject = { value: 3n };
    const body = bodyOf([own(call("call_1", args))]);
    const sent = (((body["contents"] as JsonObject[])[0]!["parts"] as JsonObject[])[0]!["functionCall"] as JsonObject)[
      "args"
    ] as JsonObject;
    sent["value"] = 4n;
    expect(args).toEqual({ value: 3n });
  });

  test("an empty unsigned text part gives no part", () => {
    expect(contents([own(text(""), call("call_1", {}))])).toEqual([
      { role: "model", parts: [{ functionCall: { id: "call_1", name: "double", args: {} } }] },
    ]);
  });

  test("reasoning and refusal parts are not sent", () => {
    expect(contents([own(new ReasoningPart("Thinking."), text("A."), new RefusalPart("No."))])).toEqual([
      { role: "model", parts: [{ text: "A." }] },
    ]);
  });

  test("a reasoning-only message is skipped", () => {
    expect(contents([user("Hi."), own(new ReasoningPart("Thinking.")), user("More.")])).toEqual([
      { role: "user", parts: [{ text: "Hi." }] },
      { role: "user", parts: [{ text: "More." }] },
    ]);
  });

  test("a trailing assistant message goes as written", () => {
    expect(contents([user("Hi."), own(text("prior  "))])).toEqual([
      { role: "user", parts: [{ text: "Hi." }] },
      { role: "model", parts: [{ text: "prior  " }] },
    ]);
  });
});

describe("refusals", () => {
  const media = new MediaPart({ mime: "image/png", url: "https://example.com/a.png" });

  test("a media part is refused, checked before any other part", () => {
    refuses(() => bodyOf([user("Hi."), message("user", media)]), "the gemini route does not send media parts yet");
    refuses(
      () => bodyOf([message("user", call("call_1", {})), message("user", media)]),
      "the gemini route does not send media parts yet",
    );
  });

  test("a part a role cannot carry is refused", () => {
    refuses(
      () => bodyOf([message("user", call("call_1", {}))]),
      "the gemini route cannot send a ToolCallPart in a user message",
    );
    refuses(
      () => bodyOf([own(result("call_1", "6"))]),
      "the gemini route cannot send a ToolResultPart in an assistant message",
    );
    refuses(() => bodyOf([message("tool", text("6"))]), "the gemini route cannot send a TextPart in a tool message");
    refuses(
      () => bodyOf([message("system", new ReasoningPart("x"))]),
      "the gemini route cannot send a ReasoningPart in a system message",
    );
  });
});
