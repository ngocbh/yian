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
  checkConversation, checkMessage, checkModelConversation, ConversationError, fromModelMessage, type Message, messageSteps,
  rebuildMessages, toModelMessage,
} from "../../src/nodes/agent/messages.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { type JsonObject, logJson, parseJson } from "../../src/typesys/json.ts";
import { messagesKey, NEW_MESSAGES, VariablePool } from "../../src/typesys/pool.ts";
import { newMessagesType, typeFor } from "../../src/typesys/types.ts";
import { buildValue, buildValueAs, FileRef, TypeCheckError, TypedValue, ValueKind, type ValueRecord } from "../../src/typesys/values.ts";
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

  test("author goes to the model message and back", () => {
    const value: Message = { role: "user", author: "ana", parts: [{ type: "text", text: "Hi" }] };
    const model = toModelMessage(value);
    expect(model.author).toBe("ana");
    expect(fromModelMessage(model)).toEqual(value);
  });

  test("an empty author is refused", () => {
    expect(() => toModelMessage({ role: "user", author: "", parts: [{ type: "text", text: "Hi" }] })).toThrow(
      "author: expected a non-empty string",
    );
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

  test("Message has exactly the schema's message keys", () => {
    expect([...message.fields!.keys()].sort()).toEqual(Object.keys(SCHEMA.properties).filter((key) => key !== "format").sort());
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

describe("checkModelConversation", () => {
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
    expect(() => checkModelConversation(messages)).not.toThrow();
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
      checkModelConversation(messages);
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(ConversationError);
    expect([(error as ConversationError).index, (error as ConversationError).reason]).toEqual([index, reason]);
    expect((error as Error).message).toBe(`message ${index}: ${reason}`);
  });
});

describe("checkConversation over public Message values", () => {
  const read = (): Message => checkMessage({ role: "user", parts: [{ type: "text", text: "Read" }] });
  const call = (): Message =>
    checkMessage({ role: "assistant", parts: [{ type: "tool_call", id: "c1", name: "read_file", arguments: {} }] });
  const answer = (): Message =>
    checkMessage({ role: "tool", parts: [{ type: "tool_result", call_id: "c1", name: "read_file", content: "hi" }] });
  const again = (): Message => checkMessage({ role: "user", parts: [{ type: "text", text: "again" }] });

  /** Return the error `checkConversation(messages)` throws. */
  function refusal(messages: readonly Message[]): unknown {
    try {
      checkConversation(messages);
    } catch (caught) {
      return caught;
    }
    throw new Error("checkConversation did not throw");
  }

  test("a call answered by its result, read by call_id, passes", () => {
    expect(() => checkConversation([read(), call(), answer(), again()])).not.toThrow();
  });

  test("an orphan result is a ConversationError", () => {
    const error = refusal([read(), answer(), again()]);
    expect(error).toBeInstanceOf(ConversationError);
    expect((error as Error).message).toBe("message 1: tool result 'c1' answers no open tool call");
  });

  test("an unanswered call is a ConversationError", () => {
    const error = refusal([read(), call(), again()]);
    expect(error).toBeInstanceOf(ConversationError);
    expect((error as Error).message).toBe("message 1: tool call 'c1' (read_file) has no result before the next user message");
  });

  test("a message the model message refuses is a ConversationError naming it", () => {
    const error = refusal([read(), checkMessage({ role: "user", author: "", parts: [{ type: "text", text: "Hi" }] })]);
    expect(error).toBeInstanceOf(ConversationError);
    expect((error as Error).message).toBe("message 1: author: expected a non-empty string");
  });

  test("a model message isn't a Message, at compile time", () => {
    const models = [new ModelMessage({ role: "user", parts: [new TextPart("x")] })];
    // `tsc` checks the directive: the public check takes Message values, not model messages.
    // @ts-expect-error
    const check = () => checkConversation(models);
    expect(typeof check).toBe("function");
  });
});

test("an AGENT's new messages live under <id>.new_messages, typed list(Message)", () => {
  expect(NEW_MESSAGES).toBe("new_messages");
  expect(messagesKey("turn#3/reply")).toBe("turn#3/reply.new_messages");
  expect(newMessagesType()).toEqual(typeFor("list(Message)", new Map()));
  expect(newMessagesType()).toBe(newMessagesType());
});

test("${x.new_messages[.path]} reads the pool name x.new_messages", () => {
  const pool = new VariablePool();
  const said = [{ role: "assistant", parts: [{ type: "text", text: "hi" }] }];
  pool.set("x", "answer");
  pool.set(messagesKey("x"), said, newMessagesType());
  expect(pool.resolve("x", ["output"])).toBe("answer");
  expect(pool.resolve("x", ["new_messages"])).toEqual(said);
  expect(pool.resolve("y", ["new_messages"])).toBeNull();
  expect(pool.resolve("x", ["new_messages", "role"])).toBeNull();
});

/** A chat `model_call` step whose reply arrived, as `AgentSteps` writes it. */
const replyStep = (message: ModelMessage, extra: Record<string, unknown> = {}): JsonObject =>
  ({ kind: "model_call", purpose: "chat", output: message.toJSON(), finish: "stop", ...extra }) as JsonObject;
/** A `tool_result` step. */
const resultStep = (callId: string, output: string, extra: Record<string, unknown> = {}): JsonObject =>
  ({ kind: "tool_result", call_id: callId, started_at: null, output, is_error: false, ...extra }) as JsonObject;
/** An assistant reply calling `look` once per id. */
const calling = (...ids: string[]) =>
  new ModelMessage({ role: "assistant", parts: ids.map((id) => new ToolCallPart({ id, name: "look", arguments: { q: id } })) });
const saying = (text: string) => new ModelMessage({ role: "assistant", parts: [new TextPart(text)] });
const callPart = (id: string) => ({ type: "tool_call", id, name: "look", arguments: { q: id } });
/** A tool result part as a `Message` holds it: `is_error` only when set. */
const resultPart = (id: string, content: string, isError = false) =>
  ({ type: "tool_result", call_id: id, name: "look", content, ...(isError ? { is_error: true } : {}) });

test("AGENT produces messages through the module's functions", () => {
  const node = new AgentNode("chat");
  expect(node.producesMessages).toBe(true);
  const steps = [replyStep(saying("hi"))];
  expect(node.messageSteps(steps)).toEqual([1]);
  expect(node.rebuildMessages(steps, null).toObject()).toEqual([{ role: "assistant", parts: [{ type: "text", text: "hi" }] }]);
});

describe("an AGENT's new messages", () => {
  const steps: JsonObject[] = [
    { kind: "llm_config", llm_config: {} }, // 1: not the loop's
    { kind: "model_call", purpose: "chat", output: null, error: { kind: "rate_limit", message: "x" }, retry_in: 1 }, // 2: a failed attempt
    replyStep(calling("a", "b")), // 3
    resultStep("b", "B"), // 4: results are recorded in finish order
    resultStep("a", "A"), // 5
    replyStep(saying("done")), // 6
    { kind: "model_call", purpose: "structured_fallback", output: saying("nope").toJSON() }, // 7: an invalid structured attempt
    { kind: "model_call", purpose: "structured_fallback", output: saying('{"a":1}').toJSON() }, // 8: the accepted one
  ] as JsonObject[];
  const output = buildValue({ a: 1n });
  const answer = { role: "assistant", parts: [{ type: "text", text: '{"a":1}' }] };

  test("messageSteps picks each chat reply that arrived, every tool result and the accepted structured call", () => {
    expect(messageSteps(steps)).toEqual([3, 4, 5, 6, 8]);
    expect(messageSteps(steps.slice(0, 6))).toEqual([3, 4, 5, 6]);
    expect(messageSteps([])).toEqual([]);
  });

  test("rebuildMessages gives each reply, its results in call order, then the structured value as JSON", () => {
    const picked = messageSteps(steps).map((n) => steps[n - 1]!);
    const value = rebuildMessages(picked, output);
    expect(value.kind).toBe(ValueKind.LIST_OBJECT);
    const messages = value.toObject() as Message[];
    expect(messages).toEqual([
      { role: "assistant", parts: [callPart("a"), callPart("b")] },
      { role: "tool", parts: [resultPart("a", "A")] },
      { role: "tool", parts: [resultPart("b", "B")] },
      { role: "assistant", parts: [{ type: "text", text: "done" }] },
      answer,
    ] as unknown as Message[]);
    checkConversation(messages);
  });

  test("a tool-less structured agent's messages are its value as JSON; a native reply's output-tool call never shows", () => {
    const native = { kind: "model_call", purpose: "structured_native", output: calling("out").toJSON() } as JsonObject;
    expect(messageSteps([native])).toEqual([1]);
    expect(rebuildMessages([native], output).toObject()).toEqual([answer]);
    const text = buildValueAs(typeFor("str", new Map()), "plain");
    expect(rebuildMessages([native], text).toObject()).toEqual([{ role: "assistant", parts: [{ type: "text", text: '"plain"' }] }]);
  });

  test("a structured call that isn't the last model_call, failed, or was cut is not picked", () => {
    const cut = { kind: "model_call", purpose: "structured_native", output: saying("").toJSON(), finish: "interrupted" } as JsonObject;
    const failed = { kind: "model_call", purpose: "structured_native", output: null, error: { kind: "x", message: "y" } } as JsonObject;
    expect(messageSteps([steps[7]!, cut])).toEqual([]);
    expect(messageSteps([steps[7]!, failed])).toEqual([]);
  });

  test("the accepted structured call needs the output value", () => {
    expect(() => rebuildMessages([steps[7]!], null)).toThrow(
      new RangeError("the accepted structured call of an AGENT needs its output value"),
    );
  });

  test("a call with no result is dropped, and a message left empty with it", () => {
    const unparsed = new ModelMessage({
      role: "assistant",
      parts: [new TextPart("trying"), new ToolCallPart({ id: "z", name: "look", arguments: null, rawArguments: "{oops" })],
    });
    const messages = rebuildMessages([replyStep(unparsed), replyStep(calling("q"), { finish: "interrupted" })], null).toObject();
    expect(messages).toEqual([{ role: "assistant", parts: [{ type: "text", text: "trying" }] }]);
  });

  test("closer results (skipped, crashed) count as results", () => {
    const messages = rebuildMessages([replyStep(calling("a")), resultStep("a", "x", { is_error: true, closed: "crashed" })], null)
      .toObject() as Message[];
    expect(messages[1]).toEqual({ role: "tool", parts: [resultPart("a", "x", true)] } as unknown as Message);
  });

  test("ids repeated across replies pair within their reply", () => {
    const messages = rebuildMessages(
      [replyStep(calling("c0")), resultStep("c0", "1"), replyStep(calling("c0")), resultStep("c0", "2"), replyStep(saying("ok"))],
      null,
    ).toObject() as Message[];
    expect(messages.filter((m) => m.role === "tool").map((m) => (m.parts[0] as { content: string }).content)).toEqual(["1", "2"]);
    checkConversation(messages);
  });

  test("inline media (a run with no folder) is dropped; file media is kept", () => {
    const media = new ModelMessage({
      role: "assistant",
      parts: [
        new TextPart("look"),
        new MediaPart({ mime: "image/png", data: "iVBORw0KGgo=" }),
        new MediaPart({ mime: "image/png", file: { uri: "media/abc.png", mime: "image/png" } }),
      ],
    });
    const inline = new ModelMessage({ role: "assistant", parts: [new MediaPart({ mime: "image/png", data: "iVBORw0KGgo=" })] });
    const messages = rebuildMessages([replyStep(media), replyStep(inline)], null).toObject() as Message[];
    expect(messages).toEqual([
      {
        role: "assistant",
        parts: [
          { type: "text", text: "look" },
          { type: "media", mime: "image/png", file: new FileRef({ uri: "media/abc.png", mime: "image/png", name: null }) },
        ],
      },
    ] as unknown as Message[]);
  });

  test("a reply keeps its author and protocol", () => {
    const authored = new ModelMessage({ role: "assistant", author: "ada", protocol: "anthropic_messages", parts: [new TextPart("hi")] });
    expect(rebuildMessages([replyStep(authored)], null).toObject()).toEqual([
      { role: "assistant", author: "ada", protocol: "anthropic_messages", parts: [{ type: "text", text: "hi" }] },
    ]);
  });

  test("no steps give the empty list", () => {
    expect(rebuildMessages([], null).toObject()).toEqual([]);
  });
});
