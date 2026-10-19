/**
 * Tests for OpenAI Chat Completions body building in `src/llm/protocols/openai_chat.ts`: the
 * per-route options, messages, tools, tool choice, block merge, reasoning echo.
 */

import { describe, expect, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import {
  MediaPart,
  Message,
  NamedTool,
  type Part,
  ReasoningPart,
  RefusalPart,
  type ReplayDegradedReason,
  type ReplayKey,
  replayRecord,
  Request,
  type Role,
  TextPart,
  ToolCallPart,
  type ToolChoice,
  ToolResultPart,
  ToolSpec,
} from "../../src/llm/messages.ts";
import { type ChatOptions, chatOptions, OpenAIChat, PROTOCOL } from "../../src/llm/protocols/openai_chat.ts";
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

function assistant(...parts: Part[]): Message {
  return message("assistant", ...parts);
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

interface BodyOptions {
  model?: string;
  provider?: string;
  options?: ChatOptions;
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
  block?: JsonObject | null;
  tools?: ToolSpec[];
  toolChoice?: ToolChoice;
  parallelToolCalls?: boolean | null;
}

function bodyOf(messages: Message[], options: BodyOptions = {}): JsonObject {
  const protocol = new OpenAIChat({
    model: options.model ?? "grok-4",
    provider: options.provider ?? "xai",
    options: options.options,
    block: options.block,
    onReplayDegraded: options.onReplayDegraded,
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

/** Return the rendered `messages` of a body built from `messages`. */
function sent(messages: Message[], options: BodyOptions = {}): JsonValue {
  return bodyOf(messages, options)["messages"]!;
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

const DEFAULTS: ChatOptions = {
  echoReasoningContent: false,
  reasoningSplit: false,
  namedToolChoice: true,
  sendParallelToolCalls: true,
  streamUsage: true,
};

describe("chatOptions", () => {
  const rows: [string, string, Partial<ChatOptions>][] = [
    ["xai", "grok-4", {}],
    ["deepseek", "deepseek-reasoner", { echoReasoningContent: true, namedToolChoice: false }],
    ["deepseek", "deepseek-chat", { echoReasoningContent: true }],
    ["minimax", "MiniMax-M2.7", { reasoningSplit: true, namedToolChoice: false }],
    ["minimax", "MiniMax-Text-01", {}],
    ["minimax-cn", "MiniMax-M2", { reasoningSplit: true, namedToolChoice: false }],
    ["qwen", "MiniMax-M2.5", { namedToolChoice: false }],
    ["openrouter", "deepseek/deepseek-r1", {}],
    ["vllm", "deepseek-reasoner", { namedToolChoice: false }],
    ["azure", "gpt-4o", {}],
    ["azure", "my-deploy", { sendParallelToolCalls: false }],
  ];
  for (const [provider, model, changes] of rows) {
    test(`${provider} ${model}`, () => {
      const options = chatOptions(provider, model);
      expect(options).toEqual({ ...DEFAULTS, ...changes });
      expect(Object.isFrozen(options)).toBe(true);
    });
  }
});

describe("the protocol object", () => {
  test("protocol name and framing", () => {
    const protocol = new OpenAIChat({ model: "grok-4", provider: "xai" });
    expect(PROTOCOL).toBe("openai_chat");
    expect(protocol.name).toBe("openai_chat");
    expect(protocol.framing).toBe("sse");
    expect(protocol.model).toBe("grok-4");
    expect(protocol.provider).toBe("xai");
  });

  test("the options default to chatOptions(provider, model)", () => {
    for (const [provider, model] of [
      ["deepseek", "deepseek-reasoner"],
      ["minimax", "MiniMax-M2.7"],
      ["azure", "my-deploy"],
    ] as const) {
      expect(new OpenAIChat({ model, provider }).options).toEqual(chatOptions(provider, model));
    }
  });

  test("given options are used", () => {
    const options: ChatOptions = { ...DEFAULTS, echoReasoningContent: true };
    expect(new OpenAIChat({ model: "grok-4", provider: "xai", options }).options).toEqual(options);
  });
});

describe("top-level keys", () => {
  test("model, messages, stream and stream_options are sent and nothing else", () => {
    expect(bodyOf([system("You are terse."), user("Say hello.")])).toEqual({
      model: "grok-4",
      messages: [
        { role: "system", content: "You are terse." },
        { role: "user", content: "Say hello." },
      ],
      stream: true,
      stream_options: { include_usage: true },
    });
  });

  test("streamUsage false drops stream_options", () => {
    const body = bodyOf([user("Hi.")], { options: { ...DEFAULTS, streamUsage: false } });
    expect(body).toEqual({ model: "grok-4", messages: [{ role: "user", content: "Hi." }], stream: true });
  });

  test("an Azure deployment name is sent as model", () => {
    expect(bodyOf([user("Hi.")], { model: "my-deploy", provider: "azure" })).toEqual({
      model: "my-deploy",
      messages: [{ role: "user", content: "Hi." }],
      stream: true,
      stream_options: { include_usage: true },
    });
  });
});

describe("system and user messages", () => {
  for (const role of ["system", "user"] as const) {
    test(`one ${role} text part is sent as a string`, () => {
      expect(sent([message(role, new TextPart("One."))])).toEqual([{ role, content: "One." }]);
    });

    test(`several ${role} text parts become text parts`, () => {
      expect(sent([message(role, new TextPart("One."), new TextPart("Two."))])).toEqual([
        {
          role,
          content: [
            { type: "text", text: "One." },
            { type: "text", text: "Two." },
          ],
        },
      ]);
    });

    test(`a ${role} message without parts is skipped`, () => {
      expect(sent([message(role), user("Hi.")])).toEqual([{ role: "user", content: "Hi." }]);
    });
  }

  test("system messages stay in place", () => {
    expect(sent([system("A."), user("Hi."), system("B.")])).toEqual([
      { role: "system", content: "A." },
      { role: "user", content: "Hi." },
      { role: "system", content: "B." },
    ]);
  });

  const media = new MediaPart({ mime: "image/png", url: "https://x.test/a.png" });
  const withMedia: [string, Message][] = [
    ["system", message("system", new TextPart("Look."), media)],
    ["user", message("user", new TextPart("Look."), media)],
    ["assistant", message("assistant", new TextPart("Here."), media)],
    ["tool", message("tool", result("call_1", "6"), media)],
  ];
  for (const [role, carrier] of withMedia) {
    test(`media parts are refused in a ${role} message`, () => {
      refuses(() => bodyOf([user("Hi."), carrier]), "the openai_chat route does not send media parts yet");
    });
  }

  const wrongParts: [string, Message, string][] = [
    ["user", message("user", call("call_1", { value: 3n })), "cannot send a ToolCallPart in a user message"],
    ["tool", message("tool", new TextPart("6")), "cannot send a TextPart in a tool message"],
    ["assistant", assistant(result("call_1", "6")), "cannot send a ToolResultPart in an assistant message"],
    ["system", message("system", new ReasoningPart("r")), "cannot send a ReasoningPart in a system message"],
  ];
  for (const [role, carrier, text] of wrongParts) {
    test(`a part the role cannot carry is refused (${role})`, () => {
      refuses(() => bodyOf([user("Hi."), carrier]), `the openai_chat route ${text}`);
    });
  }
});

describe("assistant messages", () => {
  test("no text and no calls sends empty content", () => {
    expect(sent([assistant()])).toEqual([{ role: "assistant", content: "" }]);
  });

  test("calls without text send null content", () => {
    const [rendered] = sent([assistant(call("call_1", { value: 3n }))]) as JsonObject[];
    expect(rendered!["content"]).toBeNull();
  });

  test("calls with an empty text send null content", () => {
    const [rendered] = sent([assistant(new TextPart(""), call("call_1", { value: 3n }))]) as JsonObject[];
    expect(rendered!["content"]).toBeNull();
  });

  test("one text part is sent as is, several join with newlines", () => {
    expect(sent([assistant(new TextPart("Hello."))])).toEqual([{ role: "assistant", content: "Hello." }]);
    expect(sent([assistant(new TextPart("A."), new TextPart("B."))])).toEqual([
      { role: "assistant", content: "A.\nB." },
    ]);
  });

  test("refusals are not sent", () => {
    expect(sent([assistant(new RefusalPart("No."), new TextPart("Hello."))])).toEqual([
      { role: "assistant", content: "Hello." },
    ]);
  });

  test("reasoning is not sent off echo routes", () => {
    const key: ReplayKey = { protocol: PROTOCOL, provider: "xai", model: "grok-4" };
    const thought = new ReasoningPart("Thinking.", replayRecord(key, { field: "reasoning_content" }));
    expect(sent([assistant(thought, new TextPart("Hello."))])).toEqual([{ role: "assistant", content: "Hello." }]);
  });

  test("valid calls go first with compact arguments, then invalid calls with their raw text", () => {
    const parts = [
      new TextPart("Calling."),
      invalidCall("call_bad", '{"value": 3'),
      call("call_1", { name: "Bùi" }),
      call("yian_call_0", { value: 3n, nested: { list: [1n, 2.5] } }),
    ];
    expect(sent([assistant(...parts)])).toEqual([
      {
        role: "assistant",
        content: "Calling.",
        tool_calls: [
          { type: "function", id: "call_1", function: { name: "double", arguments: '{"name":"Bùi"}' } },
          {
            type: "function",
            id: "yian_call_0",
            function: { name: "double", arguments: '{"value":3,"nested":{"list":[1,2.5]}}' },
          },
          { type: "function", id: "call_bad", function: { name: "double", arguments: '{"value": 3' } },
        ],
      },
    ]);
  });
});

describe("tool messages", () => {
  test("each result becomes a tool message, is_error not sent", () => {
    expect(sent([message("tool", result("call_1", "6"), result("call_2", "boom", true))])).toEqual([
      { role: "tool", tool_call_id: "call_1", content: "6" },
      { role: "tool", tool_call_id: "call_2", content: "boom" },
    ]);
  });

  test("name is never sent", () => {
    const messages = [system("S."), user("U."), assistant(new TextPart("A.")), message("tool", result("c", "6"))];
    for (const rendered of sent(messages) as JsonObject[]) expect(Object.hasOwn(rendered, "name")).toBe(false);
  });
});

describe("tools, tool choice, parallel tool calls", () => {
  test("tools are function tools, strict only when set", () => {
    const strict = new ToolSpec({ name: "s", description: "Strict.", parameters: { type: "object" }, strict: false });
    expect(bodyOf([user("Hi.")], { tools: [DOUBLE, strict] })["tools"]).toEqual([
      {
        type: "function",
        function: {
          name: "double",
          description: "Double an integer.",
          parameters: { properties: { value: { type: "integer" } }, required: ["value"], type: "object" },
        },
      },
      {
        type: "function",
        function: { name: "s", description: "Strict.", parameters: { type: "object" }, strict: false },
      },
    ]);
  });

  test("tool parameters are copied", () => {
    const tools = bodyOf([user("Hi.")], { tools: [DOUBLE] })["tools"] as JsonObject[];
    const fn = tools[0]!["function"] as JsonObject;
    ((fn["parameters"] as JsonObject)["required"] as JsonValue[]).push("x");
    expect(DOUBLE.parameters["required"]).toEqual(["value"]);
  });

  test("no tools or tool choice omits them", () => {
    const body = bodyOf([user("Hi.")]);
    expect(Object.hasOwn(body, "tools")).toBe(false);
    expect(Object.hasOwn(body, "tool_choice")).toBe(false);
  });

  const choices: [ToolChoice, JsonValue][] = [
    [new NamedTool("Answer"), { type: "function", function: { name: "Answer" } }],
    ["auto", "auto"],
    ["required", "required"],
    ["none", "none"],
  ];
  for (const [choice, expected] of choices) {
    test(`tool choice ${JSON.stringify(expected)}`, () => {
      expect(bodyOf([user("Hi.")], { tools: [DOUBLE], toolChoice: choice })["tool_choice"]).toEqual(expected);
    });

    const named = choice instanceof NamedTool;
    const verb = named ? "is" : "is not";
    test(`tool choice ${JSON.stringify(expected)} ${verb} suppressed without namedToolChoice`, () => {
      const options = { ...DEFAULTS, namedToolChoice: false };
      const body = bodyOf([user("Hi.")], { tools: [DOUBLE], toolChoice: choice, options });
      if (named) expect(Object.hasOwn(body, "tool_choice")).toBe(false);
      else expect(body["tool_choice"]).toEqual(expected);
    });
  }

  test("deepseek-reasoner drops a named choice but keeps parallel_tool_calls", () => {
    const body = bodyOf([user("Hi.")], {
      provider: "deepseek",
      model: "deepseek-reasoner",
      tools: [DOUBLE],
      toolChoice: new NamedTool("double"),
      parallelToolCalls: false,
    });
    expect(Object.hasOwn(body, "tool_choice")).toBe(false);
    expect(body["parallel_tool_calls"]).toBe(false);
  });

  for (const value of [false, true]) {
    test(`parallel_tool_calls ${value} is sent when set`, () => {
      expect(bodyOf([user("Hi.")], { parallelToolCalls: value })["parallel_tool_calls"]).toBe(value);
    });

    test(`parallel_tool_calls ${value} is dropped without sendParallelToolCalls`, () => {
      const options = { ...DEFAULTS, sendParallelToolCalls: false };
      expect(Object.hasOwn(bodyOf([user("Hi.")], { parallelToolCalls: value, options }), "parallel_tool_calls")).toBe(
        false,
      );
    });
  }

  test("parallel_tool_calls unset is not sent", () => {
    expect(Object.hasOwn(bodyOf([user("Hi.")]), "parallel_tool_calls")).toBe(false);
  });

  test("an Azure deployment other than gpt-4o drops parallel_tool_calls; gpt-4o keeps it", () => {
    const options = { provider: "azure", parallelToolCalls: false, toolChoice: new NamedTool("double") };
    const other = bodyOf([user("Hi.")], { ...options, model: "my-deploy", tools: [DOUBLE] });
    expect(Object.hasOwn(other, "parallel_tool_calls")).toBe(false);
    expect(other["tool_choice"]).toEqual({ type: "function", function: { name: "double" } });
    expect(bodyOf([user("Hi.")], { ...options, model: "gpt-4o", tools: [DOUBLE] })["parallel_tool_calls"]).toBe(false);
  });
});

describe("reasoning_split and the block", () => {
  test("MiniMax M2 sends reasoning_split", () => {
    expect(bodyOf([user("Hi.")], { provider: "minimax", model: "MiniMax-M2.7" })).toEqual({
      model: "MiniMax-M2.7",
      messages: [{ role: "user", content: "Hi." }],
      stream: true,
      stream_options: { include_usage: true },
      reasoning_split: true,
    });
  });

  test("a block reasoning_split false wins", () => {
    const block = { reasoning_split: false };
    const body = bodyOf([user("Hi.")], { provider: "minimax", model: "MiniMax-M2.7", block });
    expect(body["reasoning_split"]).toBe(false);
  });

  test("MiniMax-Text-01 sends no reasoning_split", () => {
    const body = bodyOf([user("Hi.")], { provider: "minimax", model: "MiniMax-Text-01" });
    expect(Object.hasOwn(body, "reasoning_split")).toBe(false);
  });

  test("block is merged last", () => {
    const block: JsonObject = {
      reasoning_effort: "high",
      max_completion_tokens: 4096n,
      temperature: 0.2,
      stop: ["END"],
      tools: [{ type: "function", function: { name: "other", parameters: {} } }],
      parallel_tool_calls: true,
      stream_options: { include_usage: false },
      user: null,
    };
    const body = bodyOf([user("Hi.")], { block, parallelToolCalls: false, tools: [DOUBLE] });
    expect(body).toEqual({
      model: "grok-4",
      messages: [{ role: "user", content: "Hi." }],
      stream: true,
      stream_options: { include_usage: false },
      tools: [{ type: "function", function: { name: "other", parameters: {} } }],
      parallel_tool_calls: true,
      reasoning_effort: "high",
      max_completion_tokens: 4096n,
      temperature: 0.2,
      stop: ["END"],
    });
  });

  test("block is not shared with the body", () => {
    const block: JsonObject = { thinking: { type: "enabled" } };
    const body = bodyOf([user("Hi.")], { block });
    (body["thinking"] as JsonObject)["type"] = "changed";
    expect(block).toEqual({ thinking: { type: "enabled" } });
  });

  test("later changes to the caller's block do not change bodies", () => {
    const block: JsonObject = { temperature: 0.2 };
    const protocol = new OpenAIChat({ model: "grok-4", provider: "xai", block });
    block["temperature"] = 0.9;
    block["max_tokens"] = 10n;
    const body = protocol.buildBody(new Request({ messages: [user("Hi.")] }));
    expect(body["temperature"]).toBe(0.2);
    expect(Object.hasOwn(body, "max_tokens")).toBe(false);
  });

  test("a null block leaf is unset and a __proto__ key stays an ordinary key", () => {
    // Built with parseJson: an object literal's "__proto__" would set the prototype instead.
    const block = parseJson('{"thinking": {"type": null}, "metadata": {"__proto__": "x"}}', "plain") as JsonObject;
    const body = bodyOf([user("Hi.")], { block });
    expect(Object.hasOwn(body["thinking"] as JsonObject, "type")).toBe(false);
    expect(Object.hasOwn(body["metadata"] as JsonObject, "__proto__")).toBe(true);
    expect(encodeBody(body)).toContain('"metadata":{"__proto__":"x"}');
  });
});

const KEY: ReplayKey = { protocol: PROTOCOL, provider: "deepseek", model: "deepseek-reasoner" };
const DEEPSEEK = { provider: "deepseek", model: "deepseek-reasoner" };

/** Return a replay record for `fields`, keyed to the DeepSeek route unless `key` is given. */
function record(fields: JsonObject, key: ReplayKey = KEY): JsonObject {
  return replayRecord(key, fields);
}

function thought(text: string, payload: JsonObject | null = record({ field: "reasoning_content" })): ReasoningPart {
  return new ReasoningPart(text, payload);
}

/** Render `messages` on the DeepSeek route and return them and the reasons passed to `onReplayDegraded`. */
function echoed(messages: Message[], options: BodyOptions = {}): [JsonValue, ReplayDegradedReason[]] {
  const reasons: ReplayDegradedReason[] = [];
  const rendered = sent(messages, { ...DEEPSEEK, ...options, onReplayDegraded: (reason) => reasons.push(reason) });
  return [rendered, reasons];
}

describe("reasoning echo", () => {
  test("a matching record is echoed on a call turn", () => {
    const turn = assistant(thought("Need the tool."), call("call_1", { value: 3n }));
    expect(echoed([turn])).toEqual([
      [
        {
          role: "assistant",
          content: null,
          tool_calls: [{ type: "function", id: "call_1", function: { name: "double", arguments: '{"value":3}' } }],
          reasoning_content: "Need the tool.",
        },
      ],
      [],
    ]);
  });

  test("a matching record is echoed on a text turn", () => {
    expect(echoed([assistant(thought("Hmm."), new TextPart("Hello."))])).toEqual([
      [{ role: "assistant", content: "Hello.", reasoning_content: "Hmm." }],
      [],
    ]);
  });

  test("several parts join with no separator", () => {
    const [rendered] = echoed([assistant(thought("A."), new TextPart("Hi."), thought("B."))]);
    expect((rendered as JsonObject[])[0]!["reasoning_content"]).toBe("A.B.");
  });

  test("an empty text is sent as an empty string", () => {
    const [rendered] = echoed([assistant(thought(""), new TextPart("Hi."))]);
    expect((rendered as JsonObject[])[0]).toEqual({ role: "assistant", content: "Hi.", reasoning_content: "" });
  });

  test("a record from another wire field is left out silently", () => {
    const parts = [thought("Other.", record({ field: "reasoning" })), new TextPart("Hi.")];
    expect(echoed([assistant(...parts)])).toEqual([[{ role: "assistant", content: "Hi." }], []]);
  });

  test("a usable record beside one from another field still echoes", () => {
    const parts = [thought("Other.", record({ field: "reasoning" })), thought("Kept."), new TextPart("Hi.")];
    expect(echoed([assistant(...parts)])).toEqual([
      [{ role: "assistant", content: "Hi.", reasoning_content: "Kept." }],
      [],
    ]);
  });

  const field = { field: "reasoning_content" };
  const unusable: [string, JsonObject, ReplayDegradedReason][] = [
    ["another provider", record(field, { ...KEY, provider: "openrouter" }), "other_provider"],
    ["another model", record(field, { ...KEY, model: "deepseek-chat" }), "other_model"],
    ["another protocol", record(field, { ...KEY, protocol: "openai_responses" }), "other_protocol"],
    ["an unknown version", { ...record(field), v: 2n }, "unknown_version"],
    ["a malformed envelope", { ...record(field), provider: 5n }, "malformed"],
    ["an unknown field", record({ ...field, extra: "x" }), "unknown_field"],
    ["no field", record({}), "malformed"],
    ["a non-string field", record({ field: 1n }), "malformed"],
  ];
  for (const [name, payload, reason] of unusable) {
    test(`a record with ${name} is left out and signals ${reason} once`, () => {
      const turn = assistant(thought("Kept.", record(field)), thought("Dropped.", payload), new TextPart("Hi."));
      expect(echoed([turn])).toEqual([[{ role: "assistant", content: "Hi.", reasoning_content: "Kept." }], [reason]]);
    });
  }

  test("the signal names the first unusable part's reason", () => {
    const turn = assistant(
      thought("A.", record(field, { ...KEY, model: "deepseek-chat" })),
      thought("B.", { ...record(field), v: 2n }),
      new TextPart("Hi."),
    );
    expect(echoed([turn])).toEqual([[{ role: "assistant", content: "Hi." }], ["other_model"]]);
  });

  test("each message signals once", () => {
    const bad = assistant(thought("A.", record(field, { ...KEY, model: "deepseek-chat" })), new TextPart("Hi."));
    const [, reasons] = echoed([bad, assistant(thought("B."), new TextPart("Hi.")), bad]);
    expect(reasons).toEqual(["other_model", "other_model"]);
  });

  test("a message whose only records are unusable sends no reasoning_content", () => {
    const turn = assistant(thought("A.", record({})), call("call_1", { value: 3n }));
    const [rendered, reasons] = echoed([turn]);
    expect(Object.hasOwn((rendered as JsonObject[])[0]!, "reasoning_content")).toBe(false);
    expect(reasons).toEqual(["malformed"]);
  });

  test("a reasoning part without a payload is left out with no signal", () => {
    expect(echoed([assistant(thought("Bare.", null), new TextPart("Hi."))])).toEqual([
      [{ role: "assistant", content: "Hi." }],
      [],
    ]);
  });

  test("without a callback an unusable record is still left out", () => {
    const turn = assistant(thought("A.", record(field, { ...KEY, model: "deepseek-chat" })), new TextPart("Hi."));
    expect(sent([turn], DEEPSEEK)).toEqual([{ role: "assistant", content: "Hi." }]);
  });

  test("a throwing callback propagates from buildBody", () => {
    const turn = assistant(thought("A.", record(field, { ...KEY, model: "deepseek-chat" })), new TextPart("Hi."));
    const onReplayDegraded = () => {
      throw new Error("stop");
    };
    expect(() => bodyOf([user("Hi."), turn], { ...DEEPSEEK, onReplayDegraded })).toThrow("stop");
  });

  test("an echo-off route with a DeepSeek record sends nothing and signals nothing", () => {
    const reasons: ReplayDegradedReason[] = [];
    const turn = assistant(thought("A."), new TextPart("Hi."));
    const rendered = sent([turn], { onReplayDegraded: (reason) => reasons.push(reason) });
    expect([rendered, reasons]).toEqual([[{ role: "assistant", content: "Hi." }], []]);
  });
});

describe("reasoning echo on a request with tools", () => {
  const field = { field: "reasoning_content" };
  const tools = [DOUBLE];
  const callTurn = (...parts: Part[]) => assistant(...parts, call("call_1", { value: 3n }));
  const callSent = (extra: JsonObject = {}): JsonObject => ({
    role: "assistant",
    content: null,
    tool_calls: [{ type: "function", id: "call_1", function: { name: "double", arguments: '{"value":3}' } }],
    ...extra,
  });

  // Each turn has no reasoning DeepSeek can take back: it gets `""` with tools, nothing without.
  const empty: [string, Message, ReplayDegradedReason[]][] = [
    ["a turn with no reasoning part (another provider's, or deepseek-chat's)", callTurn(), []],
    ["a turn whose reasoning part has no payload", callTurn(thought("Bare.", null)), []],
    ["a turn whose record is from another wire field", callTurn(thought("Other.", record({ field: "reasoning" }))), []],
    [
      "a turn whose record is from another model",
      callTurn(thought("Other model.", record(field, { ...KEY, model: "deepseek-v4-pro" }))),
      ["other_model"],
    ],
    [
      "a turn whose record is from another provider",
      callTurn(thought("Other provider.", record(field, { protocol: PROTOCOL, provider: "xai", model: "grok-4" }))),
      ["other_provider"],
    ],
    ["a turn whose record is malformed", callTurn(thought("Bad.", record({}))), ["malformed"]],
  ];
  for (const [name, turn, reasons] of empty) {
    test(`${name} gets reasoning_content "" with tools`, () => {
      expect(echoed([turn], { tools })).toEqual([[callSent({ reasoning_content: "" })], reasons]);
    });
    test(`${name} gets no reasoning_content without tools`, () => {
      expect(echoed([turn])).toEqual([[callSent()], reasons]);
    });
  }

  test("a usable record is echoed unchanged with tools", () => {
    const turn = callTurn(thought("Need the tool."));
    expect(echoed([turn], { tools })).toEqual([[callSent({ reasoning_content: "Need the tool." })], []]);
  });

  test("a usable record beside an unusable one echoes only the usable text", () => {
    const other = thought("Other model.", record(field, { ...KEY, model: "deepseek-v4-pro" }));
    const turn = callTurn(other, thought("Kept."));
    expect(echoed([turn], { tools })).toEqual([[callSent({ reasoning_content: "Kept." })], ["other_model"]]);
  });

  test("every earlier assistant turn gets one, text turns too", () => {
    const messages = [user("Double 3."), callTurn(), message("tool", result("call_1", "6")), assistant(new TextPart("6."))];
    expect(echoed(messages, { tools })).toEqual([
      [
        { role: "user", content: "Double 3." },
        callSent({ reasoning_content: "" }),
        { role: "tool", tool_call_id: "call_1", content: "6" },
        { role: "assistant", content: "6.", reasoning_content: "" },
      ],
      [],
    ]);
  });

  test("an echo-off route with tools sends no reasoning_content", () => {
    expect(sent([callTurn(thought("A."))], { tools })).toEqual([callSent()]);
  });
});
