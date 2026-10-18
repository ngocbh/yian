/** Tests for OpenAI Responses body building in `src/llm/protocols/openai_responses.ts`: input items, tools, tool choice, block merge, subscription. */

import { describe, expect, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import {
  MediaPart,
  Message,
  NamedTool,
  type Part,
  Request,
  type Role,
  TextPart,
  type ToolChoice,
  ToolResultPart,
  ToolSpec,
} from "../../src/llm/messages.ts";
import { OpenAIResponses } from "../../src/llm/protocols/openai_responses.ts";
import { type JsonObject, type JsonValue, parseJson } from "../../src/typesys/json.ts";

const DOUBLE = new ToolSpec({
  name: "double",
  description: "Double an integer.",
  parameters: { properties: { value: { type: "integer" } }, required: ["value"], type: "object" },
});

function message(role: Role, ...parts: Part[]): Message {
  return new Message({ role, parts });
}

function system(text: string): Message {
  return message("system", new TextPart(text));
}

function user(text: string): Message {
  return message("user", new TextPart(text));
}

function result(callId: string, content: string, isError = false): ToolResultPart {
  return new ToolResultPart({ callId, name: "double", content, isError });
}

interface BodyOptions {
  model?: string;
  provider?: string;
  subscription?: boolean;
  block?: JsonObject | null;
  tools?: ToolSpec[];
  toolChoice?: ToolChoice;
  parallelToolCalls?: boolean | null;
}

function bodyOf(messages: Message[], options: BodyOptions = {}): JsonObject {
  const protocol = new OpenAIResponses({
    model: options.model ?? "gpt-5.5",
    provider: options.provider ?? "openai",
    block: options.block,
    subscription: options.subscription,
  });
  return protocol.buildBody(
    new Request({
      messages,
      tools: options.tools,
      toolChoice: options.toolChoice,
      parallelToolCalls: options.parallelToolCalls,
    }),
  );
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

describe("the protocol object", () => {
  test("protocol name and framing", () => {
    const protocol = new OpenAIResponses({ model: "gpt-5.5", provider: "openai" });
    expect(protocol.name).toBe("openai_responses");
    expect(protocol.framing).toBe("sse");
    expect(protocol.model).toBe("gpt-5.5");
    expect(protocol.provider).toBe("openai");
    expect(protocol.account).toBeNull();
    expect(protocol.subscription).toBe(false);
  });
});

describe("top-level keys", () => {
  test("model, input and stream are always sent and nothing else", () => {
    expect(bodyOf([system("You are terse."), user("Say hello.")])).toEqual({
      model: "gpt-5.5",
      input: [
        { type: "message", role: "system", content: "You are terse." },
        { type: "message", role: "user", content: "Say hello." },
      ],
      stream: true,
    });
  });
});

describe("system and user messages", () => {
  test("several system messages each become an item in place", () => {
    expect(bodyOf([system("A."), user("Hi."), system("B.")])["input"]).toEqual([
      { type: "message", role: "system", content: "A." },
      { type: "message", role: "user", content: "Hi." },
      { type: "message", role: "system", content: "B." },
    ]);
  });

  test("one user text part is sent as a string", () => {
    expect(bodyOf([user("Say hello.")])["input"]).toEqual([{ type: "message", role: "user", content: "Say hello." }]);
  });

  test("several user text parts become input_text parts", () => {
    const several = message("user", new TextPart("One."), new TextPart("Two."));
    expect(bodyOf([several])["input"]).toEqual([
      {
        type: "message",
        role: "user",
        content: [
          { type: "input_text", text: "One." },
          { type: "input_text", text: "Two." },
        ],
      },
    ]);
  });

  const media = new MediaPart({ mime: "image/png", url: "https://x.test/a.png" });
  const withMedia: [string, Message][] = [
    ["system", message("system", new TextPart("Look."), media)],
    ["user", message("user", new TextPart("Look."), media)],
    ["assistant", message("assistant", new TextPart("Here."), media)],
    ["tool", message("tool", result("call_1", "6"), media)],
  ];
  for (const subscription of [false, true]) {
    for (const [role, carrier] of withMedia) {
      test(`media parts are refused in a ${role} message (subscription ${subscription})`, () => {
        refuses(
          () => bodyOf([user("Hi."), carrier], { subscription }),
          "the openai_responses route does not send media parts yet",
        );
      });
    }
  }

  const wrongParts: [string, Message, string][] = [
    ["user", message("user", result("call_1", "6")), "cannot send a ToolResultPart in a user message"],
    ["tool", message("tool", new TextPart("6")), "cannot send a TextPart in a tool message"],
    ["assistant", message("assistant", result("call_1", "6")), "cannot send a ToolResultPart in an assistant message"],
  ];
  for (const [role, carrier, text] of wrongParts) {
    test(`a part the role cannot carry is refused (${role})`, () => {
      refuses(() => bodyOf([user("Hi."), carrier]), `the openai_responses route ${text}`);
    });
  }

  for (const role of ["system", "user"] as const) {
    test(`a ${role} message without parts is skipped`, () => {
      expect(bodyOf([message(role), user("Hi.")])["input"]).toEqual([{ type: "message", role: "user", content: "Hi." }]);
    });
  }

  test("message name is never sent", () => {
    const input = bodyOf([user("Hi."), message("tool", result("call_1", "6"))])["input"] as JsonObject[];
    for (const item of input) expect(Object.hasOwn(item, "name")).toBe(false);
  });
});

describe("tool messages", () => {
  test("each tool result becomes a function_call_output and is_error is not sent", () => {
    const results = message("tool", result("call_1", "6"), result("call_2", "boom", true));
    expect(bodyOf([results])["input"]).toEqual([
      { type: "function_call_output", call_id: "call_1", output: "6" },
      { type: "function_call_output", call_id: "call_2", output: "boom" },
    ]);
  });
});

describe("tools, tool choice, parallel tool calls", () => {
  test("strict is sent only when set", () => {
    const open = new ToolSpec({ name: "open", description: "Open schema.", parameters: { type: "object" }, strict: false });
    const sent = bodyOf([user("Hi.")], { tools: [DOUBLE, open] })["tools"] as JsonObject[];
    expect(Object.hasOwn(sent[0]!, "strict")).toBe(false);
    expect(sent[1]).toEqual({
      type: "function",
      name: "open",
      description: "Open schema.",
      parameters: { type: "object" },
      strict: false,
    });
    const strict = new ToolSpec({ name: "s", description: "", parameters: { type: "object" }, strict: true });
    const tools = bodyOf([user("Hi.")], { tools: [strict] })["tools"] as JsonObject[];
    expect(tools[0]!["strict"]).toBe(true);
  });

  test("tool parameters are copied", () => {
    const tools = bodyOf([user("Hi.")], { tools: [DOUBLE] })["tools"] as JsonObject[];
    ((tools[0]!["parameters"] as JsonObject)["required"] as JsonValue[]).push("x");
    expect(DOUBLE.parameters["required"]).toEqual(["value"]);
  });

  const choices: [ToolChoice, JsonValue][] = [
    [new NamedTool("Answer"), { type: "function", name: "Answer" }],
    ["auto", "auto"],
    ["required", "required"],
    ["none", "none"],
  ];
  for (const [choice, sent] of choices) {
    test(`tool choice ${JSON.stringify(sent)}`, () => {
      expect(bodyOf([user("Hi.")], { tools: [DOUBLE], toolChoice: choice })["tool_choice"]).toEqual(sent);
    });
  }

  test("no tools or tool choice omits them", () => {
    const body = bodyOf([user("Hi.")]);
    expect(Object.hasOwn(body, "tools")).toBe(false);
    expect(Object.hasOwn(body, "tool_choice")).toBe(false);
  });

  for (const value of [false, true]) {
    test(`parallel_tool_calls ${value} is sent when set`, () => {
      expect(bodyOf([user("Hi.")], { parallelToolCalls: value })["parallel_tool_calls"]).toBe(value);
      expect(Object.hasOwn(bodyOf([user("Hi.")]), "parallel_tool_calls")).toBe(false);
    });
  }
});

describe("block merge", () => {
  test("block is merged last", () => {
    const block: JsonObject = {
      reasoning: { effort: "high", summary: "auto" },
      text: { verbosity: "low" },
      max_output_tokens: 4096n,
      temperature: 0.2,
      service_tier: "flex",
      metadata: { a: "b" },
      prompt_cache_key: "k1",
      parallel_tool_calls: true,
      tools: [{ type: "web_search" }],
      store: null,
    };
    const body = bodyOf([user("Hi.")], { block, parallelToolCalls: false, tools: [DOUBLE] });
    expect(body).toEqual({
      model: "gpt-5.5",
      input: [{ type: "message", role: "user", content: "Hi." }],
      stream: true,
      tools: [{ type: "web_search" }],
      parallel_tool_calls: true,
      reasoning: { effort: "high", summary: "auto" },
      text: { verbosity: "low" },
      max_output_tokens: 4096n,
      temperature: 0.2,
      service_tier: "flex",
      metadata: { a: "b" },
      prompt_cache_key: "k1",
    });
  });

  test("block is not shared with the body", () => {
    const block: JsonObject = { metadata: { a: "b" } };
    const body = bodyOf([user("Hi.")], { block });
    (body["metadata"] as JsonObject)["a"] = "changed";
    expect(block).toEqual({ metadata: { a: "b" } });
  });

  test("later changes to the caller's block do not change bodies", () => {
    const block: JsonObject = { temperature: 0.2 };
    const protocol = new OpenAIResponses({ model: "gpt-5.5", provider: "openai", block });
    block["temperature"] = 0.9;
    block["max_output_tokens"] = 10n;
    const body = protocol.buildBody(new Request({ messages: [user("Hi.")] }));
    expect(body["temperature"]).toBe(0.2);
    expect(Object.hasOwn(body, "max_output_tokens")).toBe(false);
  });

  test("a null block member is unset and a __proto__ key stays an ordinary key", () => {
    // Built with parseJson: an object literal's "__proto__" would set the prototype instead.
    const block = parseJson('{"reasoning": {"summary": null}, "metadata": {"__proto__": "x"}}', "plain") as JsonObject;
    const body = bodyOf([user("Hi.")], { block });
    expect(Object.hasOwn(body["reasoning"] as JsonObject, "summary")).toBe(false);
    const metadata = body["metadata"] as JsonObject;
    expect(Object.hasOwn(metadata, "__proto__")).toBe(true);
    expect(encodeBody(body)).toContain('"metadata":{"__proto__":"x"}');
  });

  test("name-routed models get the API key body", () => {
    expect(bodyOf([user("Hi.")], { model: "prod-gpt-5-codex", provider: "azure" })).toEqual({
      model: "prod-gpt-5-codex",
      input: [{ type: "message", role: "user", content: "Hi." }],
      stream: true,
    });
  });
});

describe("subscription", () => {
  test("subscription body", () => {
    expect(bodyOf([system("You are terse."), user("Double 3.")], { subscription: true })).toEqual({
      model: "gpt-5.5",
      input: [{ type: "message", role: "user", content: "Double 3." }],
      stream: true,
      store: false,
      include: ["reasoning.encrypted_content"],
      instructions: "You are terse.",
    });
  });

  test("subscription joins system messages into instructions", () => {
    const joined = message("system", new TextPart("B1"), new TextPart("B2"));
    const body = bodyOf([system("A."), user("Hi."), joined], { subscription: true });
    expect(body["instructions"]).toBe("A.\n\nB1B2");
    expect(body["input"]).toEqual([{ type: "message", role: "user", content: "Hi." }]);
  });

  test("subscription default instructions", () => {
    expect(bodyOf([user("Hi.")], { subscription: true })["instructions"]).toBe("You are a helpful assistant.");
  });

  test("subscription skips empty system text in instructions", () => {
    const body = bodyOf([system("A."), system(""), message("system"), system("B.")], { subscription: true });
    expect(body["instructions"]).toBe("A.\n\nB.");
  });

  test("subscription with only empty system text uses the default instructions", () => {
    const body = bodyOf([system(""), message("system"), user("Hi.")], { subscription: true });
    expect(body["instructions"]).toBe("You are a helpful assistant.");
  });

  test("subscription refuses non-text system content", () => {
    refuses(
      () => bodyOf([message("system", result("call_1", "6")), user("Hi.")], { subscription: true }),
      "OpenAI subscription instructions must contain only text",
    );
  });

  test("subscription include keeps the block order and deduplicates", () => {
    let block: JsonObject = { include: ["file_search_call.results", "reasoning.encrypted_content"] };
    expect(bodyOf([user("Hi.")], { subscription: true, block })["include"]).toEqual([
      "file_search_call.results",
      "reasoning.encrypted_content",
    ]);
    block = { include: ["file_search_call.results", "file_search_call.results"] };
    expect(bodyOf([user("Hi.")], { subscription: true, block })["include"]).toEqual([
      "file_search_call.results",
      "reasoning.encrypted_content",
    ]);
  });

  test("subscription settings are applied after the block", () => {
    const block: JsonObject = {
      store: true,
      stream: false,
      include: ["x"],
      instructions: "block",
      previous_response_id: "resp_0",
      reasoning: { effort: "low" },
    };
    const body = bodyOf([system("S."), user("Hi.")], { subscription: true, block });
    expect(body["store"]).toBe(false);
    expect(body["stream"]).toBe(true);
    expect(body["include"]).toEqual(["x", "reasoning.encrypted_content"]);
    expect(body["instructions"]).toBe("S.");
    expect(Object.hasOwn(body, "previous_response_id")).toBe(false);
    expect(body["reasoning"]).toEqual({ effort: "low" });
  });

  test("subscription tools, tool choice and parallel calls serialize as on the API key route", () => {
    const request = { tools: [DOUBLE], toolChoice: new NamedTool("double"), parallelToolCalls: false };
    const sub = bodyOf([user("Hi.")], { subscription: true, ...request });
    const key = bodyOf([user("Hi.")], request);
    for (const name of ["tools", "tool_choice", "parallel_tool_calls"]) expect(sub[name]).toEqual(key[name]!);
  });
});
