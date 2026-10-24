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
import { checkMessage, fromModelMessage, type Message, toModelMessage } from "../../src/nodes/agent/messages.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { FileRef, TypeCheckError, TypedValue, ValueKind, type ValueRecord } from "../../src/typesys/values.ts";

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
