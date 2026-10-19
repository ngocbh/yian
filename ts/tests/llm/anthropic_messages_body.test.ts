/**
 * Tests for Anthropic Messages body building in `src/llm/protocols/anthropic_messages.ts`: the
 * top-level keys, turn merging, the system field, user, tool and assistant turns, tool ids.
 */

import { describe, expect, spyOn, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import {
  MediaPart,
  Message,
  type Part,
  ReasoningPart,
  RefusalPart,
  Request,
  type Role,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from "../../src/llm/messages.ts";
import { AnthropicMessages, PROTOCOL } from "../../src/llm/protocols/anthropic_messages.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

const SONNET = "claude-sonnet-4-5";

function message(role: Role, ...parts: Part[]): Message {
  return new Message({ role, parts });
}

function system(...texts: string[]): Message {
  return message("system", ...texts.map((text) => new TextPart(text)));
}

function user(...texts: string[]): Message {
  return message("user", ...texts.map((text) => new TextPart(text)));
}

function assistant(...parts: Part[]): Message {
  return message("assistant", ...parts);
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

function invalidCall(id: string, raw: string): ToolCallPart {
  return new ToolCallPart({ id, name: "double", arguments: null, rawArguments: raw });
}

function bodyOf(messages: Message[], model = SONNET): JsonObject {
  return new AnthropicMessages({ model, provider: "anthropic" }).buildBody(new Request({ messages }));
}

/** Return the rendered `messages` of a body built from `messages`. */
function sent(messages: Message[], model = SONNET): JsonValue {
  return bodyOf(messages, model)["messages"]!;
}

/** Build a body with `process.emitWarning` stubbed; return it and the warnings it emitted. */
function warned(messages: Message[], model = SONNET): [JsonObject, unknown[][]] {
  const emit = spyOn(process, "emitWarning").mockImplementation(() => {});
  try {
    const body = bodyOf(messages, model);
    return [body, emit.mock.calls.map((args) => [...args])];
  } finally {
    emit.mockRestore();
  }
}

/** Return the error `run` throws, failing when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Assert `run` throws a `TypeError` with exactly `text`. */
function refuses(run: () => unknown, text: string): void {
  const error = thrown(run);
  expect(error).toBeInstanceOf(TypeError);
  expect(error.message).toBe(text);
}

function hoistWarning(model: string): string {
  return (
    "A non-leading `SystemMessage` was moved to the top-level `system` field and now applies to the " +
    "entire conversation. To send it in place, use a supported model and place it after a human or " +
    `tool message, either last or before an AI message (model: '${model}').`
  );
}

describe("the protocol object", () => {
  test("protocol name and framing", () => {
    const protocol = new AnthropicMessages({ model: SONNET, provider: "anthropic" });
    expect(PROTOCOL).toBe("anthropic_messages");
    expect(protocol.name).toBe("anthropic_messages");
    expect(protocol.framing).toBe("sse");
    expect(protocol.model).toBe(SONNET);
    expect(protocol.provider).toBe("anthropic");
    expect(protocol.account).toBeNull();
    expect(new AnthropicMessages({ model: SONNET, provider: "anthropic", account: "org-1" }).account).toBe("org-1");
  });
});

describe("top-level keys", () => {
  test("model, max_tokens, messages, system and stream are sent and nothing else", () => {
    expect(bodyOf([system("You are terse."), user("Say hello.")])).toEqual({
      model: SONNET,
      max_tokens: 64000n,
      messages: [{ role: "user", content: "Say hello." }],
      system: "You are terse.",
      stream: true,
    });
  });

  test("no system message sends no system", () => {
    expect(Object.hasOwn(bodyOf([user("Hi.")]), "system")).toBe(false);
  });
});

describe("merging", () => {
  test("user and tool messages merge into one user turn", () => {
    expect(sent([user("Hi."), tool(result("call_1", "6"))])).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "Hi." },
          { type: "tool_result", content: "6", tool_use_id: "call_1", is_error: false },
        ],
      },
    ]);
  });

  test("user messages merge, their strings becoming text blocks", () => {
    expect(sent([user("A."), user("B.")])).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "A." },
          { type: "text", text: "B." },
        ],
      },
    ]);
  });

  test("whitespace-only text blocks are dropped from merged and multi-part lists", () => {
    expect(sent([user("A."), user("  ")])).toEqual([{ role: "user", content: [{ type: "text", text: "A." }] }]);
    expect(sent([user("A.", "\n")])).toEqual([{ role: "user", content: [{ type: "text", text: "A." }] }]);
  });

  test("a lone whitespace or empty user string is kept", () => {
    expect(sent([user("  ")])).toEqual([{ role: "user", content: "  " }]);
    expect(sent([user(""), assistant(text("Hi."))])).toEqual([
      { role: "user", content: "" },
      { role: "assistant", content: "Hi." },
    ]);
  });

  test("a tool result after a tool call comes before the next user text in one turn", () => {
    const messages = [
      user("Double 3."),
      assistant(call("call_1", { value: 3n })),
      tool(result("call_1", "6")),
      user("Thanks."),
    ];
    expect(sent(messages)).toEqual([
      { role: "user", content: "Double 3." },
      { role: "assistant", content: [{ type: "tool_use", id: "call_1", name: "double", input: { value: 3n } }] },
      {
        role: "user",
        content: [
          { type: "tool_result", content: "6", tool_use_id: "call_1", is_error: false },
          { type: "text", text: "Thanks." },
        ],
      },
    ]);
  });
});

describe("system messages", () => {
  test("a leading system with one text part is lifted as a string", () => {
    expect(bodyOf([system("S."), user("Hi.")])["system"]).toBe("S.");
  });

  test("a leading system with several text parts is lifted as a list, whitespace kept", () => {
    expect(bodyOf([system("A.", " "), user("Hi.")])["system"]).toEqual([
      { type: "text", text: "A." },
      { type: "text", text: " " },
    ]);
  });

  test("leading system messages merge into one list", () => {
    const body = bodyOf([system("A."), system("B."), user("Hi.")]);
    expect(body["system"]).toEqual([
      { type: "text", text: "A." },
      { type: "text", text: "B." },
    ]);
    expect(body["messages"]).toEqual([{ role: "user", content: "Hi." }]);
  });

  test("a later system message is hoisted with LangChain's warning", () => {
    const [body, warnings] = warned([user("Hi."), system("Late."), assistant(text("Ok."))]);
    expect(body["system"]).toBe("Late.");
    expect(body["messages"]).toEqual([
      { role: "user", content: "Hi." },
      { role: "assistant", content: "Ok." },
    ]);
    expect(warnings).toEqual([[hoistWarning(SONNET)]]);
  });

  const inPlaceModels = [
    "claude-fable-5-1",
    "claude-mythos-5",
    "claude-opus-4-8",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
  ];
  for (const model of inPlaceModels) {
    test(`a later system message stays in place on ${model}`, () => {
      const [body, warnings] = warned([system("S."), user("Hi."), system("Late."), assistant(text("Ok."))], model);
      expect(body["system"]).toBe("S.");
      expect(body["messages"]).toEqual([
        { role: "user", content: "Hi." },
        { role: "system", content: "Late." },
        { role: "assistant", content: "Ok." },
      ]);
      expect(warnings).toEqual([]);
    });
  }

  test("a later system message is not kept in place on claude-sonnet-4-5", () => {
    const [body, warnings] = warned([user("Hi."), system("Late."), assistant(text("Ok."))], SONNET);
    expect(body["messages"]).toEqual([
      { role: "user", content: "Hi." },
      { role: "assistant", content: "Ok." },
    ]);
    expect(warnings).toHaveLength(1);
  });

  test("a later system message followed by a user turn is hoisted on a listed model", () => {
    const model = "claude-opus-5-5";
    const [body, warnings] = warned([user("Hi."), system("Late."), user("More.")], model);
    expect(body["system"]).toBe("Late.");
    expect(body["messages"]).toEqual([
      { role: "user", content: "Hi." },
      { role: "user", content: "More." },
    ]);
    expect(warnings).toEqual([[hoistWarning(model)]]);
  });

  test("a trailing system message stays in place on claude-opus-5-5", () => {
    const [body, warnings] = warned([user("Hi."), system("Late.")], "claude-opus-5-5");
    expect(Object.hasOwn(body, "system")).toBe(false);
    expect(body["messages"]).toEqual([
      { role: "user", content: "Hi." },
      { role: "system", content: "Late." },
    ]);
    expect(warnings).toEqual([]);
  });

  test("a second non-consecutive system message is refused", () => {
    const text = "Received multiple non-consecutive system messages.";
    refuses(() => bodyOf([system("S."), user("Hi."), system("Late."), assistant(new TextPart("Ok."))]), text);
    refuses(() => bodyOf([system("S."), user("Hi."), system("Late."), user("More.")], "claude-opus-5-5"), text);
  });
});

describe("user and tool messages", () => {
  test("one user text part is sent as a string, several as text blocks", () => {
    expect(sent([user("One.")])).toEqual([{ role: "user", content: "One." }]);
    expect(sent([user("One.", "Two.")])).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "One." },
          { type: "text", text: "Two." },
        ],
      },
    ]);
  });

  test("each tool result carries is_error, true or false, and no name", () => {
    expect(sent([tool(result("call_1", "6"), result("call_2", "boom", true))])).toEqual([
      {
        role: "user",
        content: [
          { type: "tool_result", content: "6", tool_use_id: "call_1", is_error: false },
          { type: "tool_result", content: "boom", tool_use_id: "call_2", is_error: true },
        ],
      },
    ]);
  });
});

describe("tool ids", () => {
  test("an id Anthropic refuses is hashed the same on tool_use and tool_result", () => {
    const id = "functions.double:0";
    const hashed = "toolu_3c02df8844016dad56fde9d3";
    expect(sent([user("Go."), assistant(call(id, { value: 3n })), tool(result(id, "6"))])).toEqual([
      { role: "user", content: "Go." },
      { role: "assistant", content: [{ type: "tool_use", id: hashed, name: "double", input: { value: 3n } }] },
      { role: "user", content: [{ type: "tool_result", content: "6", tool_use_id: hashed, is_error: false }] },
    ]);
  });

  test("a valid id and an empty id are sent unchanged", () => {
    for (const id of ["toolu_01-AbC_9", ""]) {
      const [, turn, results] = sent([user("Go."), assistant(call(id, {})), tool(result(id, "6"))]) as JsonObject[];
      expect((turn!["content"] as JsonObject[])[0]!["id"]).toBe(id);
      expect((results!["content"] as JsonObject[])[0]!["tool_use_id"]).toBe(id);
    }
  });
});

describe("assistant messages", () => {
  /** Render `parts` as a non-final assistant turn and return its content. */
  function nonFinal(...parts: Part[]): JsonValue {
    const [, turn] = sent([user("Hi."), assistant(...parts), user("Next.")]) as JsonObject[];
    return turn!["content"]!;
  }

  /** Render `parts` as the final assistant turn and return its content. */
  function final(...parts: Part[]): JsonValue {
    const [, turn] = sent([user("Hi."), assistant(...parts)]) as JsonObject[];
    return turn!["content"]!;
  }

  test("one text part is sent as a string, a call makes a list", () => {
    expect(nonFinal(text("Hello."))).toBe("Hello.");
    expect(nonFinal(text("Calling."), call("call_1", { value: 3n }))).toEqual([
      { type: "text", text: "Calling." },
      { type: "tool_use", id: "call_1", name: "double", input: { value: 3n } },
    ]);
  });

  test("two adjacent text parts are two blocks", () => {
    expect(nonFinal(text("A."), text("B."))).toEqual([
      { type: "text", text: "A." },
      { type: "text", text: "B." },
    ]);
  });

  test("text, call, text is three blocks in part order", () => {
    expect(nonFinal(text("A."), call("call_1", { value: 3n }), text("B."))).toEqual([
      { type: "text", text: "A." },
      { type: "tool_use", id: "call_1", name: "double", input: { value: 3n } },
      { type: "text", text: "B." },
    ]);
  });

  test("a whitespace-only text block before a call is dropped", () => {
    expect(nonFinal(text("\n\n"), call("call_1", { value: 3n }))).toEqual([
      { type: "tool_use", id: "call_1", name: "double", input: { value: 3n } },
    ]);
  });

  test("an empty text part gives no block", () => {
    expect(nonFinal(text(""), text("A."))).toBe("A.");
  });

  test("a lone whitespace text is kept as a string", () => {
    expect(nonFinal(text("  "))).toBe("  ");
  });

  test("the final assistant string is stripped at the end", () => {
    expect(final(text("  Hello. \n"))).toBe("  Hello.");
  });

  test("the final assistant's last text block is stripped at the end", () => {
    expect(final(call("call_1", { value: 3n }), text("Done. \n"))).toEqual([
      { type: "tool_use", id: "call_1", name: "double", input: { value: 3n } },
      { type: "text", text: "Done." },
    ]);
  });

  test("a final text before a call is not stripped", () => {
    expect(final(text("Calling. "), call("call_1", { value: 3n }))).toEqual([
      { type: "text", text: "Calling. " },
      { type: "tool_use", id: "call_1", name: "double", input: { value: 3n } },
    ]);
  });

  test("a non-final assistant is not stripped", () => {
    expect(nonFinal(text("Hello. \n"))).toBe("Hello. \n");
  });

  test("an empty non-final assistant is skipped, an empty final one is sent with empty content", () => {
    expect(sent([user("Hi."), assistant(), user("Next.")])).toEqual([
      { role: "user", content: "Hi." },
      { role: "user", content: "Next." },
    ]);
    expect(sent([user("Hi."), assistant()])).toEqual([
      { role: "user", content: "Hi." },
      { role: "assistant", content: "" },
    ]);
    expect(final(text(" \n"))).toBe("");
  });

  test("refusals are not sent", () => {
    expect(nonFinal(new RefusalPart("No."), text("Hello."))).toBe("Hello.");
    expect(sent([user("Hi."), assistant(new RefusalPart("No.")), user("Next.")])).toHaveLength(2);
  });

  test("an invalid call is sent with empty input", () => {
    expect(nonFinal(invalidCall("call_bad", '{"value": 3'))).toEqual([
      { type: "tool_use", id: "call_bad", name: "double", input: {} },
    ]);
  });

  test("call input keeps its bigints, encoded as digits", () => {
    const body = bodyOf([user("Hi."), assistant(call("call_1", { value: 3n, nested: { list: [1n, 2.5] } }))]);
    expect(encodeBody(body)).toContain('"input":{"value":3,"nested":{"list":[1,2.5]}}');
  });

  test("call input is copied", () => {
    const args: JsonObject = { nested: { value: 3n } };
    const [, turn] = sent([user("Hi."), assistant(call("call_1", args))]) as JsonObject[];
    const input = (turn!["content"] as JsonObject[])[0]!["input"] as JsonObject;
    (input["nested"] as JsonObject)["value"] = 4n;
    expect(args).toEqual({ nested: { value: 3n } });
  });
});

describe("refused parts", () => {
  const media = new MediaPart({ mime: "image/png", url: "https://x.test/a.png" });
  const withMedia: [string, Message][] = [
    ["system", message("system", text("Look."), media)],
    ["user", message("user", text("Look."), media)],
    ["assistant", message("assistant", text("Here."), media)],
    ["tool", message("tool", result("call_1", "6"), media)],
  ];
  for (const [role, carrier] of withMedia) {
    test(`media parts are refused in a ${role} message`, () => {
      refuses(() => bodyOf([user("Hi."), carrier]), "the anthropic_messages route does not send media parts yet");
    });
  }

  test("a media part is refused before a wrong part", () => {
    refuses(
      () => bodyOf([message("user", call("call_1", {})), message("user", media)]),
      "the anthropic_messages route does not send media parts yet",
    );
  });

  const wrongParts: [string, Message, string][] = [
    ["user", message("user", call("call_1", { value: 3n })), "cannot send a ToolCallPart in a user message"],
    ["assistant", assistant(result("call_1", "6")), "cannot send a ToolResultPart in an assistant message"],
    ["tool", message("tool", text("6")), "cannot send a TextPart in a tool message"],
    ["system", message("system", new ReasoningPart("r")), "cannot send a ReasoningPart in a system message"],
  ];
  for (const [role, carrier, text] of wrongParts) {
    test(`a part the role cannot carry is refused (${role})`, () => {
      refuses(() => bodyOf([user("Hi."), carrier]), `the anthropic_messages route ${text}`);
    });
  }
});
