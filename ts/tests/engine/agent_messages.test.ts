/**
 * The built-in `Message` value: its check, its conversion to and from the model message, the
 * public value codec round trip, its agreement with `message.schema.json`, and the conversation
 * check a request needs.
 *
 * TS-only: Python yian has no public message type.
 */
import { describe, expect, test } from "bun:test";

import {
  MediaPart, Message as ModelMessage, ReasoningPart, RefusalPart, replayRecord, TextPart, ToolCallPart, ToolResultPart,
} from "../../src/llm/messages.ts";
import {
  checkConversation, checkMessage, ConversationError, fromModelMessage, type Message, toModelMessage,
} from "../../src/nodes/agent/messages.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { FileRef, TypeCheckError, TypedValue, ValueKind, type ValueRecord } from "../../src/typesys/values.ts";
import { loadSchema } from "../schema.ts";

const FILE = { uri: "media/abc.png", mime: "image/png", name: null };

test("checkMessage returns the value with each file's JSON form as a FileRef", () => {
  const checked = checkMessage({ role: "user", parts: [{ type: "media", mime: "image/png", file: FILE }] });
  expect(checked.parts[0]).toEqual({ type: "media", mime: "image/png", file: new FileRef(FILE) });
});

test("checkMessage refuses a value that isn't a Message", () => {
  expect(() => checkMessage({ role: "user", parts: [{ type: "text" }] })).toThrow(TypeCheckError);
});

const KEY = { protocol: "anthropic_messages", provider: "anthropic", model: "claude-x" };

/** An assistant reply with tagged reasoning, text, a refusal and a call; its tool result; a user message with media. */
function history(): ModelMessage[] {
  return [
    new ModelMessage({
      role: "assistant",
      protocol: "anthropic_messages",
      parts: [
        new ReasoningPart("think", replayRecord(KEY, { signature: "sig" })),
        new TextPart("Reading."),
        new RefusalPart(""),
        new ToolCallPart({ id: "c1", name: "read_file", arguments: { path: "a.txt" }, rawArguments: '{"path":"a.txt"}' }),
      ],
    }),
    new ModelMessage({ role: "tool", parts: [new ToolResultPart({ callId: "c1", name: "read_file", content: "hi", isError: true })] }),
    new ModelMessage({ role: "user", parts: [new MediaPart({ mime: "image/png", file: FILE, name: "x.png" })] }),
  ];
}

describe("conversion", () => {
  test("a model message round-trips through its value, every part kept", () => {
    for (const message of history()) {
      expect(toModelMessage(fromModelMessage(message)).toJSON()).toEqual(message.toJSON());
    }
  });

  test("the value is the yian.message.v1 shape without format, the file as a FileRef", () => {
    const [reply, result, media] = history().map(fromModelMessage);
    expect(reply!.parts[0]).toEqual({ type: "reasoning", text: "think", provider_payload: replayRecord(KEY, { signature: "sig" }) });
    expect(reply!.protocol).toBe("anthropic_messages");
    expect(result!.parts[0]).toEqual({ type: "tool_result", call_id: "c1", name: "read_file", content: "hi", is_error: true });
    expect(FileRef.isFileRef((media!.parts[0] as { file: unknown }).file)).toBe(true);
  });

  test("null optional fields are left out on the way to the model", () => {
    const value: Message = { role: "user", author: null, protocol: null, parts: [{ type: "text", text: "Hi", provider_payload: null }] };
    expect(toModelMessage(value).toJSON()).toEqual({ format: "yian.message.v1", role: "user", parts: [{ type: "text", text: "Hi" }] });
  });

  test("a value the model message refuses fails with its path", () => {
    const value = { role: "assistant", parts: [{ type: "tool_call", id: "c", name: "t" }] } as Message;
    expect(() => toModelMessage(value)).toThrow("parts[0].arguments: a tool call needs arguments or raw_arguments");
  });

  test.each([
    ["a system message", new ModelMessage({ role: "system", parts: [new TextPart("x")] }), "a Message can't be a system message"],
    ["inline media", new ModelMessage({ role: "user", parts: [new MediaPart({ mime: "image/png", data: "AA==" })] }), "parts[0]: a Message holds media only as a run-folder file"],
    ["url media", new ModelMessage({ role: "user", parts: [new MediaPart({ mime: "image/png", url: "https://x" })] }), "parts[0]: a Message holds media only as a run-folder file"],
  ])("%s has no Message form", (_, message, text) => {
    expect(() => fromModelMessage(message)).toThrow(new TypeError(text));
  });
});

test("a Message and a list of them survive the run log's codec; checkMessage restores the files", () => {
  const values = history().map(fromModelMessage);
  for (const value of values) {
    const wire = new TypedValue(ValueKind.OBJECT, value as unknown as ValueRecord).toWire();
    expect(checkMessage(TypedValue.fromWire(parseJson(logJson(wire), "codec")).value)).toEqual(value);
  }
  const listWire = new TypedValue(ValueKind.LIST_OBJECT, values as unknown as ValueRecord[]).toWire();
  const list = TypedValue.fromWire(parseJson(logJson(listWire), "codec"));
  const restored = (list.value as unknown[]).map(checkMessage);
  expect(restored).toEqual(values);
  // `toEqual` doesn't tell a FileRef from its mapping, so check the class itself.
  const fileOf = (message: Message): unknown => (message.parts[0] as { file: unknown }).file;
  expect(FileRef.isFileRef(fileOf(restored[2]!))).toBe(true);
  // Undecoded, the file is its mapping, which is why a reader calls checkMessage.
  const undecoded = fileOf((list.value as Message[])[2]!);
  expect(FileRef.isFileRef(undecoded)).toBe(false);
  expect(undecoded).toEqual(FILE);
});

describe("agreement with message.schema.json", () => {
  const SCHEMA = loadSchema("message.schema.json") as { properties: Record<string, unknown>; $defs: Record<string, { properties: Record<string, unknown> }> };
  const message = typeFor("Message", new Map());
  const parts = message.fields!.get("parts")!.element!.variants!;
  const partFields = (tag: string) =>
    [...parts.find((p) => p.fields!.get("type")!.tags!.has(tag))!.fields!.keys()].sort();

  test("every key a model message may hold has a field in Message", () => {
    const keys = Object.keys(SCHEMA.properties).filter((key) => key !== "format");
    expect(keys.filter((key) => !message.fields!.has(key))).toEqual([]);
  });

  test.each(["text", "reasoning", "refusal", "tool_call", "tool_result", "media"])("the %s part has the schema's keys", (tag) => {
    const keys = Object.keys(SCHEMA.$defs[tag]!.properties).filter((key) => !(tag === "media" && (key === "data" || key === "url")));
    expect(partFields(tag)).toEqual(keys.sort());
  });
});

const user = (text = "Hi") => new ModelMessage({ role: "user", parts: [new TextPart(text)] });
const said = (text: string) => new ModelMessage({ role: "assistant", parts: [new TextPart(text)] });
const calls = (...ids: string[]) =>
  new ModelMessage({ role: "assistant", parts: [new TextPart("On it."), ...ids.map((id) => new ToolCallPart({ id, name: "read_file", arguments: {} }))] });
const result = (id: string) => new ModelMessage({ role: "tool", parts: [new ToolResultPart({ callId: id, name: "read_file", content: "ok" })] });

describe("checkConversation", () => {
  test.each([
    ["text turns", [user(), said("Hello"), user()]],
    ["a call answered by its result", [user(), calls("c1"), result("c1"), said("Done"), user()]],
    ["two calls answered in any order, one tool message each", [user(), calls("c1", "c2"), result("c2"), result("c1"), user()]],
    ["two results in one tool message", [user(), calls("c1", "c2"), new ModelMessage({ role: "tool", parts: [
      new ToolResultPart({ callId: "c1", name: "read_file", content: "a" }), new ToolResultPart({ callId: "c2", name: "read_file", content: "b" })] }), user()]],
    ["reasoning and a refusal in an assistant message", [user(), new ModelMessage({ role: "assistant", parts: [new ReasoningPart("r"), new RefusalPart("no")] }), user()]],
    ["a call id reused once its result is in", [user(), calls("c1"), result("c1"), calls("c1"), result("c1"), user()]],
    // A Gemini or Ollama history: each reply mints yian_call_<n> again.
    ["yian_call_0 in two turns", [user(), calls("yian_call_0"), result("yian_call_0"), said("Next."),
      calls("yian_call_0", "yian_call_1"), result("yian_call_1"), result("yian_call_0"), said("Done"), user()]],
    ["yian_call_0 in back-to-back exchanges", [user(), calls("yian_call_0"), result("yian_call_0"),
      calls("yian_call_0"), result("yian_call_0"), user()]],
    ["a structural text message, as the render passes it", [{ role: "user", parts: [{ type: "text" }] }]],
    ["two user messages in a row", [user(), user()]],
    ["media in a user message", [new ModelMessage({ role: "user", parts: [new MediaPart({ mime: "image/png", file: FILE })] })]],
  ])("%s passes", (_, messages) => {
    expect(() => checkConversation(messages)).not.toThrow();
  });

  test.each([
    ["a tool call in a user message", [new ModelMessage({ role: "user", parts: [new ToolCallPart({ id: "c", name: "t", arguments: {} })] })], 0,
      "user messages can't hold a tool_call part"],
    ["text in a tool message", [user(), calls("c1"), new ModelMessage({ role: "tool", parts: [new TextPart("x")] })], 2,
      "tool messages can't hold a text part"],
    ["a tool result in an assistant message", [user(), new ModelMessage({ role: "assistant", parts: [new ToolResultPart({ callId: "c", name: "t", content: "x" })] })], 1,
      "assistant messages can't hold a tool_result part"],
    ["media in an assistant message", [user(), new ModelMessage({ role: "assistant", parts: [new MediaPart({ mime: "image/png", file: FILE })] })], 1,
      "assistant messages can't hold a media part"],
    ["reasoning in a user message", [new ModelMessage({ role: "user", parts: [new ReasoningPart("r")] })], 0,
      "user messages can't hold a reasoning part"],
    ["an empty message", [user(), new ModelMessage({ role: "assistant", parts: [] }), user()], 1, "the message has no parts"],
    ["a system message", [new ModelMessage({ role: "system", parts: [new TextPart("x")] })], 0,
      "a system message can't be in a prompt list; use system_prompt:"],
    ["a call followed by a user message", [user(), calls("c1"), user()], 1,
      "tool call 'c1' (read_file) has no result before the next user message"],
    ["a call followed by another assistant message", [user(), calls("c1"), said("x"), user()], 1,
      "tool call 'c1' (read_file) has no result before the next assistant message"],
    ["one of two calls answered", [user(), calls("c1", "c2"), result("c1"), user()], 1,
      "tool call 'c2' (read_file) has no result before the next user message"],
    ["a call at the end", [user(), calls("c1")], 1, "tool call 'c1' (read_file) has no result before the end of the list"],
    ["a result with no call", [user(), result("c9"), user()], 1, "tool result 'c9' answers no open tool call"],
    ["a result for a call already answered", [user(), calls("c1"), result("c1"), result("c1"), user()], 3,
      "tool result 'c1' answers no open tool call"],
    ["a call id twice in one reply", [user(), calls("c1", "c1")], 1, "tool call id 'c1' appears twice in one reply"],
    ["a result for an earlier exchange's call", [user(), calls("yian_call_0"), result("yian_call_0"), calls("yian_call_1"),
      result("yian_call_0"), user()], 4, "tool result 'yian_call_0' answers no open tool call"],
    ["a repeated id whose second turn goes unanswered", [user(), calls("yian_call_0"), result("yian_call_0"),
      calls("yian_call_0"), user()], 3, "tool call 'yian_call_0' (read_file) has no result before the next user message"],
  ])("%s is refused", (_, messages, index, reason) => {
    let error: unknown;
    try {
      checkConversation(messages);
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(ConversationError);
    expect([(error as ConversationError).index, (error as ConversationError).reason]).toEqual([index, reason]);
    expect((error as Error).message).toBe(`message ${index}: ${reason}`);
  });
});
