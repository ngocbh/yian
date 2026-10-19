/**
 * Tests for Ollama body building in `src/llm/protocols/ollama_chat.ts`: the protocol object, the
 * fixed fields and their order, system, user, assistant and tool messages sent in place, call ids
 * and tool results, thinking sent back from matching replay records and left out when one can't be
 * used, tools with their schemas as written, the tool choice left out, the output format, the
 * block, the copies, and the media and wrong-part refusals.
 */

import { describe, expect, test } from "bun:test";

import { encodeBody } from "../../src/llm/body.ts";
import {
  MediaPart,
  Message,
  NamedTool,
  OutputFormat,
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
import { OllamaChat, PROTOCOL } from "../../src/llm/protocols/ollama_chat.ts";
import { type JsonObject, type JsonValue, parseJson } from "../../src/typesys/json.ts";

const QWEN = "qwen3";
const NO_MEDIA = "the ollama_chat route does not send media parts yet";

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

const DOUBLE = new ToolSpec({
  name: "double",
  description: "Double an integer.",
  parameters: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
});

function ollama(block: JsonObject | null = null): OllamaChat {
  return new OllamaChat({ model: QWEN, provider: "ollama", block });
}

interface BodyOptions {
  account?: string | null;
  block?: JsonObject | null;
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
  tools?: ToolSpec[];
  toolChoice?: ToolChoice;
  parallelToolCalls?: boolean | null;
  outputFormat?: OutputFormat | null;
}

function bodyOf(messages: Message[], options: BodyOptions = {}): JsonObject {
  const protocol = new OllamaChat({
    model: QWEN,
    provider: "ollama",
    account: options.account,
    block: options.block,
    onReplayDegraded: options.onReplayDegraded,
  });
  return protocol.buildBody(
    new Request({
      messages,
      tools: options.tools,
      toolChoice: options.toolChoice,
      parallelToolCalls: options.parallelToolCalls,
      outputFormat: options.outputFormat,
    }),
  );
}

/** Return the `messages` of a body built from `messages`. */
function sent(messages: Message[]): JsonValue {
  return bodyOf(messages)["messages"]!;
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
  test("protocol name, framing, model, provider and account", () => {
    const protocol = ollama();
    expect(PROTOCOL).toBe("ollama_chat");
    expect(protocol.name).toBe("ollama_chat");
    expect(protocol.framing).toBe("ndjson");
    expect(protocol.model).toBe(QWEN);
    expect(protocol.provider).toBe("ollama");
    expect(protocol.account).toBeNull();
    expect(new OllamaChat({ model: QWEN, provider: "ollama", account: "acct-1" }).account).toBe("acct-1");
  });
});

describe("the fixed fields", () => {
  test("a plain request gives exactly the fixed body, in order", () => {
    const body = bodyOf([system("You are terse."), user("Say hello.")]);
    expect(body).toEqual({
      model: QWEN,
      stream: true,
      options: {},
      messages: [
        { role: "system", content: "You are terse." },
        { role: "user", content: "Say hello." },
      ],
      tools: [],
    });
    expect(Object.keys(body)).toEqual(["model", "stream", "options", "messages", "tools"]);
    expect(encodeBody(body)).toBe(
      '{"model":"qwen3","stream":true,"options":{},"messages":[{"role":"system","content":"You are terse."},' +
        '{"role":"user","content":"Say hello."}],"tools":[]}',
    );
  });

  test("no tools and no block send tools: [] and options: {}", () => {
    const body = bodyOf([user("Hi.")]);
    expect(body["tools"]).toEqual([]);
    expect(body["options"]).toEqual({});
  });
});

describe("system and user", () => {
  test("text parts join with a newline and no leading newline", () => {
    expect(sent([system("A", "B"), user("A", "B")])).toEqual([
      { role: "system", content: "A\nB" },
      { role: "user", content: "A\nB" },
    ]);
  });

  test("empty content is left out", () => {
    expect(sent([system(""), user(""), message("user")])).toEqual([
      { role: "system" },
      { role: "user" },
      { role: "user" },
    ]);
  });

  test("messages stay in place: a late system message, two users not merged", () => {
    expect(sent([user("A."), system("S."), user("B."), user("C.")])).toEqual([
      { role: "user", content: "A." },
      { role: "system", content: "S." },
      { role: "user", content: "B." },
      { role: "user", content: "C." },
    ]);
  });
});

describe("assistant", () => {
  test("text and calls: text joined, calls in order with their ids", () => {
    const messages = sent([assistant(text("A"), call("call_1", { value: 3n }), text("B"), call("call_2", {}))]);
    expect(messages).toEqual([
      {
        role: "assistant",
        content: "A\nB",
        tool_calls: [
          { id: "call_1", function: { name: "double", arguments: { value: 3n } } },
          { id: "call_2", function: { name: "double", arguments: {} } },
        ],
      },
    ]);
    expect(Object.keys((messages as JsonObject[])[0]!)).toEqual(["role", "content", "tool_calls"]);
  });

  test("an invalid call goes with arguments: {}", () => {
    const invalid = new ToolCallPart({ id: "call_1", name: "double", arguments: null, rawArguments: "{bad" });
    expect(sent([assistant(invalid)])).toEqual([
      { role: "assistant", tool_calls: [{ id: "call_1", function: { name: "double", arguments: {} } }] },
    ]);
  });

  test("refusal parts and reasoning without a payload are left out", () => {
    expect(sent([assistant(new ReasoningPart("Thinking."), text("A."), new RefusalPart("No."))])).toEqual([
      { role: "assistant", content: "A." },
    ]);
  });

  test("an empty assistant message is sent as {role: assistant}", () => {
    expect(sent([user("Hi."), assistant(), assistant(text("")), assistant(new RefusalPart("No."))])).toEqual([
      { role: "user", content: "Hi." },
      { role: "assistant" },
      { role: "assistant" },
      { role: "assistant" },
    ]);
  });
});

const KEY: ReplayKey = { protocol: PROTOCOL, provider: "ollama", model: QWEN };

/** Return a reasoning part carrying an envelope-only replay record for `key`. */
function thought(value: string, key: ReplayKey = KEY): ReasoningPart {
  return new ReasoningPart(value, replayRecord(key, {}));
}

/** Build a body from `messages`; return its messages and the reasons passed to `onReplayDegraded`. */
function sendWithReasons(messages: Message[], options: BodyOptions = {}): [JsonValue, ReplayDegradedReason[]] {
  const reasons: ReplayDegradedReason[] = [];
  const body = bodyOf(messages, { ...options, onReplayDegraded: (reason) => reasons.push(reason) });
  return [body["messages"]!, reasons];
}

describe("thinking sent back", () => {
  test("an own reasoning record sends its text as thinking, before the calls", () => {
    const [messages, reasons] = sendWithReasons([assistant(thought("Hmm."), text("A."), call("call_1", {}))]);
    expect(messages).toEqual([
      {
        role: "assistant",
        content: "A.",
        thinking: "Hmm.",
        tool_calls: [{ id: "call_1", function: { name: "double", arguments: {} } }],
      },
    ]);
    expect(Object.keys((messages as JsonObject[])[0]!)).toEqual(["role", "content", "thinking", "tool_calls"]);
    expect(reasons).toEqual([]);
  });

  test("two own reasoning parts join with no separator", () => {
    expect(sendWithReasons([assistant(thought("A"), text("T."), thought("B"))])).toEqual([
      [{ role: "assistant", content: "T.", thinking: "AB" }],
      [],
    ]);
  });

  test("empty thinking is left out", () => {
    expect(sendWithReasons([assistant(thought(""), text("T."))])).toEqual([[{ role: "assistant", content: "T." }], []]);
  });

  const own = replayRecord(KEY, {});
  const unusable: [string, JsonObject, ReplayDegradedReason][] = [
    ["another model", replayRecord({ ...KEY, model: "llama3" }, {}), "other_model"],
    ["another provider", replayRecord({ ...KEY, provider: "ollama-cloud" }, {}), "other_provider"],
    ["another account", replayRecord({ ...KEY, account: "acct-2" }, {}), "other_account"],
    ["another protocol", replayRecord({ ...KEY, protocol: "openai_chat" }, {}), "other_protocol"],
    ["an unknown version", { ...own, v: 2n }, "unknown_version"],
    ["an unknown field", replayRecord(KEY, { extra: "x" }), "unknown_field"],
    ["a malformed record", { ...own, model: 1n }, "malformed"],
  ];
  for (const [name, payload, reason] of unusable) {
    test(`${name}: no thinking, text and calls kept, signalled once`, () => {
      const parts = [new ReasoningPart("Hmm.", payload), text("A."), call("call_1", {})];
      expect(sendWithReasons([assistant(...parts)])).toEqual([
        [
          {
            role: "assistant",
            content: "A.",
            tool_calls: [{ id: "call_1", function: { name: "double", arguments: {} } }],
          },
        ],
        [reason],
      ]);
    });
  }

  test("two unusable parts signal once, with the first reason in part order", () => {
    const parts = [thought("A", { ...KEY, model: "llama3" }), thought("B", { ...KEY, provider: "other" })];
    expect(sendWithReasons([assistant(...parts)])).toEqual([[{ role: "assistant" }], ["other_model"]]);
  });

  test("a usable record beside an unusable one: no thinking is sent", () => {
    const parts = [thought("A"), text("T."), thought("B", { ...KEY, model: "llama3" })];
    expect(sendWithReasons([assistant(...parts)])).toEqual([[{ role: "assistant", content: "T." }], ["other_model"]]);
  });

  test("reasoning without a payload is not sent and signals nothing", () => {
    expect(sendWithReasons([assistant(new ReasoningPart("Hmm."), text("T."))])).toEqual([
      [{ role: "assistant", content: "T." }],
      [],
    ]);
    // Skipped, so it neither blocks nor joins an own record's thinking.
    expect(sendWithReasons([assistant(new ReasoningPart("X"), thought("A"))])).toEqual([
      [{ role: "assistant", thinking: "A" }],
      [],
    ]);
  });

  test("an account key matches only records with that account", () => {
    const account = "acct-1";
    expect(sendWithReasons([assistant(thought("A", { ...KEY, account }))], { account })).toEqual([
      [{ role: "assistant", thinking: "A" }],
      [],
    ]);
    expect(sendWithReasons([assistant(thought("A"))], { account })).toEqual([
      [{ role: "assistant" }],
      ["other_account"],
    ]);
  });

  test("without a callback thinking is still left out, and nothing throws", () => {
    expect(sent([assistant(thought("A", { ...KEY, model: "llama3" }), text("T."))])).toEqual([
      { role: "assistant", content: "T." },
    ]);
  });

  test("a throwing callback propagates", () => {
    const onReplayDegraded = () => {
      throw new Error("stop");
    };
    const messages = [assistant(thought("A", { ...KEY, model: "llama3" }))];
    expect(() => bodyOf(messages, { onReplayDegraded })).toThrow("stop");
  });

  test("a tool result is refused before an unusable record is signalled", () => {
    const reasons: ReplayDegradedReason[] = [];
    refuses(
      () =>
        bodyOf([assistant(thought("A", { ...KEY, model: "llama3" }), result("call_1", "6"))], {
          onReplayDegraded: (reason) => reasons.push(reason),
        }),
      "the ollama_chat route cannot send a ToolResultPart in an assistant message",
    );
    expect(reasons).toEqual([]);
  });

  test("each message with an unusable record signals once per body", () => {
    const messages = [
      assistant(thought("A", { ...KEY, model: "llama3" })),
      user("Go on."),
      assistant(thought("B", { ...KEY, provider: "other" })),
    ];
    expect(sendWithReasons(messages)[1]).toEqual(["other_model", "other_provider"]);
  });
});

describe("tool results", () => {
  test("a result carries tool_name and tool_call_id", () => {
    const messages = sent([tool(result("call_7", "6"))]);
    expect(messages).toEqual([{ role: "tool", content: "6", tool_name: "double", tool_call_id: "call_7" }]);
    expect(Object.keys((messages as JsonObject[])[0]!)).toEqual(["role", "content", "tool_name", "tool_call_id"]);
  });

  test("an error result has the same shape and no error field", () => {
    const body = bodyOf([tool(result("call_1", "boom", true))]);
    expect(body["messages"]).toEqual([{ role: "tool", content: "boom", tool_name: "double", tool_call_id: "call_1" }]);
    expect(encodeBody(body)).not.toMatch(/error/i);
  });

  test("empty content is left out", () => {
    expect(sent([tool(result("call_1", ""))])).toEqual([
      { role: "tool", tool_name: "double", tool_call_id: "call_1" },
    ]);
  });

  test("two results in one tool message are two messages", () => {
    expect(sent([tool(result("call_1", "1"), result("call_2", "2"))])).toEqual([
      { role: "tool", content: "1", tool_name: "double", tool_call_id: "call_1" },
      { role: "tool", content: "2", tool_name: "double", tool_call_id: "call_2" },
    ]);
  });

  test("an empty tool message sends nothing", () => {
    expect(sent([user("Hi."), message("tool")])).toEqual([{ role: "user", content: "Hi." }]);
  });
});

describe("tools", () => {
  test("each tool is a function entry with its description and parameters", () => {
    const blank = new ToolSpec({ name: "blank", description: "", parameters: { type: "object" } });
    expect(bodyOf([user("Hi.")], { tools: [DOUBLE, blank] })["tools"]).toEqual([
      {
        type: "function",
        function: {
          name: "double",
          description: "Double an integer.",
          parameters: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
        },
      },
      { type: "function", function: { name: "blank", description: "", parameters: { type: "object" } } },
    ]);
  });

  test("a rich schema goes as written", () => {
    const parameters: JsonObject = {
      $defs: { Unit: { type: "string", enum: ["c", "f"] } },
      type: "object",
      properties: {
        unit: { $ref: "#/$defs/Unit" },
        value: { anyOf: [{ type: "integer" }, { type: "string" }] },
        place: {
          type: "object",
          properties: { city: { type: "string", format: "hostname" }, zip: { type: "string" } },
          required: ["city"],
        },
        count: { type: "integer", minimum: 0n, default: 1n },
      },
      required: ["unit", "value"],
      additionalProperties: false,
    };
    const rich = new ToolSpec({ name: "rich", description: "Rich.", parameters });
    const [entry] = bodyOf([user("Hi.")], { tools: [rich] })["tools"] as JsonObject[];
    const schema = (entry!["function"] as JsonObject)["parameters"]!;
    expect(schema).toEqual(parameters);
    // toEqual ignores key order; the bytes pin it.
    expect(encodeBody(schema)).toBe(encodeBody(parameters));
  });

  test("strict is not sent", () => {
    const strict = new ToolSpec({ name: "s", description: "S.", parameters: { type: "object" }, strict: true });
    const body = bodyOf([user("Hi.")], { tools: [strict] });
    expect(body["tools"]).toEqual([
      { type: "function", function: { name: "s", description: "S.", parameters: { type: "object" } } },
    ]);
    expect(encodeBody(body)).not.toContain("strict");
  });

  test("the tool choice and parallelToolCalls are never sent", () => {
    for (const toolChoice of ["required", new NamedTool("double")] as ToolChoice[]) {
      const body = bodyOf([user("Hi.")], { tools: [DOUBLE], toolChoice, parallelToolCalls: false });
      expect(Object.keys(body)).toEqual(["model", "stream", "options", "messages", "tools"]);
      expect(encodeBody(body)).not.toMatch(/choice|parallel/i);
    }
  });
});

describe("the output format", () => {
  const RECORD = new OutputFormat({
    name: "Record",
    schema: {
      title: "Record",
      type: "object",
      properties: { answer: { type: "string" }, score: { type: "integer" } },
      required: ["answer", "score"],
    },
  });

  test("outputFormat is sent as format, the schema as given, and tools: [] stays", () => {
    const body = bodyOf([user("Hi.")], { outputFormat: RECORD });
    expect(body).toEqual({
      model: QWEN,
      stream: true,
      options: {},
      messages: [{ role: "user", content: "Hi." }],
      format: RECORD.schema,
      tools: [],
    });
    expect(Object.keys(body)).toEqual(["model", "stream", "options", "messages", "format", "tools"]);
  });

  test("a block's think lands beside format", () => {
    const body = bodyOf([user("Hi.")], { block: { think: true }, outputFormat: RECORD });
    expect(body["format"]).toEqual(RECORD.schema);
    expect(body["think"]).toBe(true);
  });

  test("a body without outputFormat has no format", () => {
    expect(Object.hasOwn(bodyOf([user("Hi.")]), "format")).toBe(false);
  });

  test("the rendered schema is a copy", () => {
    const body = bodyOf([user("Hi.")], { outputFormat: RECORD });
    ((body["format"] as JsonObject)["required"] as JsonValue[]).push("x");
    expect(RECORD.schema["required"]).toEqual(["answer", "score"]);
  });
});

describe("the block", () => {
  test("options merge into {} and the top-level fields land as written, after yian's", () => {
    const block: JsonObject = {
      options: { num_ctx: 8192n, temperature: 0.2, stop: ["\n\n"] },
      think: "high",
      keep_alive: -1n,
      logprobs: true,
      top_logprobs: 3n,
    };
    const body = bodyOf([user("Hi.")], { block });
    expect(body).toEqual({
      model: QWEN,
      stream: true,
      options: { num_ctx: 8192n, temperature: 0.2, stop: ["\n\n"] },
      messages: [{ role: "user", content: "Hi." }],
      tools: [],
      think: "high",
      keep_alive: -1n,
      logprobs: true,
      top_logprobs: 3n,
    });
    expect(Object.keys(body)).toEqual([
      "model",
      "stream",
      "options",
      "messages",
      "tools",
      "think",
      "keep_alive",
      "logprobs",
      "top_logprobs",
    ]);
  });

  test("think true, false or a level, keep_alive a number or a duration", () => {
    for (const think of [true, false, "high"] as JsonValue[]) {
      expect(bodyOf([user("Hi.")], { block: { think } })["think"]).toBe(think);
    }
    expect(bodyOf([user("Hi.")], { block: { keep_alive: -1n } })["keep_alive"]).toBe(-1n);
    expect(bodyOf([user("Hi.")], { block: { keep_alive: "10m" } })["keep_alive"]).toBe("10m");
  });

  test("a null leaf is not sent", () => {
    const block = parseJson('{"think": null, "options": {"temperature": null, "num_ctx": 4096}}', "plain") as JsonObject;
    const body = bodyOf([user("Hi.")], { block });
    expect(Object.hasOwn(body, "think")).toBe(false);
    expect(body["options"]).toEqual({ num_ctx: 4096n });
  });

  test("later changes to the caller's block do not change bodies", () => {
    const block: JsonObject = { options: { temperature: 0.2 } };
    const protocol = ollama(block);
    (block["options"] as JsonObject)["temperature"] = 0.9;
    block["think"] = true;
    const body = protocol.buildBody(new Request({ messages: [user("Hi.")] }));
    expect(body["options"]).toEqual({ temperature: 0.2 });
    expect(Object.hasOwn(body, "think")).toBe(false);
  });

  test("the block is not shared with the body", () => {
    const protocol = ollama({ options: { stop: ["x"] } });
    const request = new Request({ messages: [user("Hi.")] });
    ((protocol.buildBody(request)["options"] as JsonObject)["stop"] as JsonValue[]).push("y");
    expect(protocol.buildBody(request)["options"]).toEqual({ stop: ["x"] });
  });
});

describe("copies", () => {
  test("the call arguments are copied", () => {
    const args: JsonObject = { value: 3n };
    const [sentMessage] = sent([assistant(call("call_1", args))]) as JsonObject[];
    const [sentCall] = sentMessage!["tool_calls"] as JsonObject[];
    ((sentCall!["function"] as JsonObject)["arguments"] as JsonObject)["value"] = 4n;
    expect(args).toEqual({ value: 3n });
  });

  test("the tool parameters are copied", () => {
    const [entry] = bodyOf([user("Hi.")], { tools: [DOUBLE] })["tools"] as JsonObject[];
    (((entry!["function"] as JsonObject)["parameters"] as JsonObject)["required"] as JsonValue[]).push("x");
    expect(DOUBLE.parameters["required"]).toEqual(["value"]);
  });
});

describe("refusals", () => {
  const media = new MediaPart({ mime: "image/png", url: "https://example.com/a.png" });

  test("a media part is refused, checked before any other part", () => {
    refuses(() => bodyOf([user("Hi."), message("user", media)]), NO_MEDIA);
    refuses(() => bodyOf([assistant(text("A."), media)]), NO_MEDIA);
    refuses(() => bodyOf([message("user", call("call_1", {})), assistant(media)]), NO_MEDIA);
  });

  test("a part a role cannot carry is refused", () => {
    refuses(
      () => bodyOf([message("user", call("call_1", {}))]),
      "the ollama_chat route cannot send a ToolCallPart in a user message",
    );
    refuses(
      () => bodyOf([message("system", new ReasoningPart("x"))]),
      "the ollama_chat route cannot send a ReasoningPart in a system message",
    );
    refuses(
      () => bodyOf([assistant(text("A."), result("call_1", "6"))]),
      "the ollama_chat route cannot send a ToolResultPart in an assistant message",
    );
    refuses(
      () => bodyOf([message("tool", text("6"))]),
      "the ollama_chat route cannot send a TextPart in a tool message",
    );
  });
});
