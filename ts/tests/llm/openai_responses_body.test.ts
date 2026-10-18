/**
 * Tests for OpenAI Responses body building in `src/llm/protocols/openai_responses.ts`: input
 * items, assistant replay, tools, tool choice, block merge, subscription.
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
import { OpenAIResponses, PROTOCOL } from "../../src/llm/protocols/openai_responses.ts";
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
  account?: string | null;
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
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
    account: options.account,
    block: options.block,
    subscription: options.subscription,
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
      const input = bodyOf([message(role), user("Hi.")])["input"];
      expect(input).toEqual([{ type: "message", role: "user", content: "Hi." }]);
    });
  }
});

const KEY: ReplayKey = { protocol: PROTOCOL, provider: "openai", model: "gpt-5.5" };
const REASONING_ITEM: JsonObject = {
  id: "rs_1",
  type: "reasoning",
  summary: [{ type: "summary_text", text: "Thinking about it." }],
  encrypted_content: "ENC-CONTINUATION-1",
};
const CALL_ITEM: JsonObject = {
  id: "fc_1",
  type: "function_call",
  name: "double",
  arguments: '{"value": 3}',
  call_id: "call_1",
  status: "completed",
};

/** Return a replay record for `fields`, keyed to the default protocol object unless `key` is given. */
function record(fields: JsonObject, key: ReplayKey = KEY): JsonObject {
  return replayRecord(key, fields);
}

function assistant(...parts: Part[]): Message {
  return message("assistant", ...parts);
}

function reasoning(item: JsonObject = REASONING_ITEM): ReasoningPart {
  return new ReasoningPart("Thinking about it.", record({ item }));
}

function call(id: string, args: JsonObject, item: JsonObject | null = null): ToolCallPart {
  return new ToolCallPart({ id, name: "double", arguments: args, providerPayload: item ? record({ item }) : null });
}

/** A text part with a record of `fields`, or no payload when `fields` is absent. */
function text(value: string, fields?: JsonObject): TextPart {
  return new TextPart(value, fields ? record({ annotations: [], ...fields }) : null);
}

function outputText(value: string, annotations: JsonValue[] = []): JsonObject {
  return { type: "output_text", text: value, annotations };
}

function refusal(value: string): JsonObject {
  return { type: "refusal", refusal: value };
}

function assistantItem(content: JsonObject[], id?: string, phase?: string): JsonObject {
  const item: JsonObject = { type: "message", role: "assistant", content };
  if (id !== undefined) item["id"] = id;
  if (phase !== undefined) item["phase"] = phase;
  return item;
}

function fallback(callId: string, args: string, name = "double"): JsonObject {
  return { type: "function_call", name, arguments: args, call_id: callId };
}

function withoutStatus(item: JsonObject): JsonObject {
  const { status: _, ...rest } = item;
  return rest;
}

/** Return the input items after the leading user message. */
function replayed(messages: Message[], options: BodyOptions = {}): JsonValue[] {
  return (bodyOf([user("Hi."), ...messages], options)["input"] as JsonValue[]).slice(1);
}

describe("assistant messages: reasoning", () => {
  test("a reasoning item is replayed without null members at any depth", () => {
    const item: JsonObject = {
      ...REASONING_ITEM,
      status: null,
      summary: [{ type: "summary_text", text: "Thinking about it.", extra: null }],
    };
    expect(replayed([assistant(reasoning(item))])).toEqual([REASONING_ITEM]);
  });

  test("reasoning without payload is dropped", () => {
    expect(replayed([assistant(new ReasoningPart("Thinking."))])).toEqual([]);
  });
});

describe("assistant messages: tool calls", () => {
  test("a call item is replayed", () => {
    expect(replayed([assistant(call("call_1", { value: 3n }, CALL_ITEM))])).toEqual([CALL_ITEM]);
  });

  test("a call without payload uses the fallback shape with compact non-ASCII arguments", () => {
    const greet = new ToolCallPart({ id: "call_1", name: "greet", arguments: { name: "Bùi" } });
    expect(replayed([assistant(greet)])).toEqual([fallback("call_1", '{"name":"Bùi"}', "greet")]);
  });

  test("a fallback call with unparsed arguments sends the raw text", () => {
    const raw = new ToolCallPart({ id: "call_1", name: "double", arguments: null, rawArguments: '{"value": 3' });
    expect(replayed([assistant(raw)])).toEqual([fallback("call_1", '{"value": 3')]);
  });

  test("fallback calls go after the message's other items", () => {
    const message = assistant(call("call_2", { value: 4n }), reasoning(), text("Let me check.", { id: "msg_1" }));
    expect(replayed([message])).toEqual([
      REASONING_ITEM,
      assistantItem([outputText("Let me check.")], "msg_1"),
      fallback("call_2", '{"value":4}'),
    ]);
  });

  test("a fallback call whose call id is already in input is skipped", () => {
    const sent = call("call_1", { value: 3n }, CALL_ITEM);
    const duplicate = call("call_1", { value: 3n });
    expect(replayed([assistant(sent, duplicate)])).toEqual([CALL_ITEM]);
    // An earlier message's replayed call counts too.
    expect(replayed([assistant(sent), assistant(duplicate)])).toEqual([CALL_ITEM]);
  });

  test("only the first of two fallback calls sharing a call id is sent", () => {
    const first = call("call_1", { value: 3n });
    const second = call("call_1", { value: 4n });
    expect(replayed([assistant(first, second)])).toEqual([fallback("call_1", '{"value":3}')]);
    expect(replayed([assistant(first), assistant(second)])).toEqual([fallback("call_1", '{"value":3}')]);
  });
});

describe("assistant messages: message items", () => {
  test("a text part becomes a message item with its id and annotations", () => {
    const annotation: JsonObject = { type: "url_citation", url: "https://x.test", start_index: 0n };
    const part = text("Hello there.", { id: "msg_1", annotations: [annotation] });
    expect(replayed([assistant(part)])).toEqual([assistantItem([outputText("Hello there.", [annotation])], "msg_1")]);
  });

  test("annotations are sent without null members", () => {
    const annotation: JsonObject = { type: "url_citation", title: null, url: "https://x.test" };
    const part = text("Hello.", { id: "msg_1", annotations: [annotation] });
    expect(replayed([assistant(part)])).toEqual([
      assistantItem([outputText("Hello.", [{ type: "url_citation", url: "https://x.test" }])], "msg_1"),
    ]);
  });

  test("a text part without payload has no id", () => {
    expect(replayed([assistant(text("Hello."))])).toEqual([assistantItem([outputText("Hello.")])]);
  });

  test("an empty text part is rendered like any other", () => {
    expect(replayed([assistant(text("", { id: "msg_1" }))])).toEqual([assistantItem([outputText("")], "msg_1")]);
  });

  test("parts sharing a message id merge and phase takes the last non-null value", () => {
    const items = replayed([
      assistant(
        text("Let me check.", { id: "msg_1", phase: "commentary" }),
        text("Part A.", { id: "msg_2", phase: "final_answer" }),
        text("Part B.", { id: "msg_2" }),
      ),
    ]);
    expect(items).toEqual([
      assistantItem([outputText("Let me check.")], "msg_1", "commentary"),
      assistantItem([outputText("Part A."), outputText("Part B.")], "msg_2", "final_answer"),
    ]);
  });

  test("a merge scans all earlier items and updates phase", () => {
    const later = text("B.", { id: "msg_1", phase: "final_answer" });
    const message = assistant(text("A.", { id: "msg_1" }), reasoning(), later);
    expect(replayed([message])).toEqual([
      assistantItem([outputText("A."), outputText("B.")], "msg_1", "final_answer"),
      REASONING_ITEM,
    ]);
  });

  test("a merge reaches an earlier assistant message", () => {
    const [first, second] = [assistant(text("A.", { id: "msg_1" })), assistant(text("B.", { id: "msg_1" }))];
    const items = replayed([first, user("More."), second]);
    expect(items).toEqual([
      assistantItem([outputText("A."), outputText("B.")], "msg_1"),
      { type: "message", role: "user", content: "More." },
    ]);
  });

  test("a merge only targets message items", () => {
    // A reasoning item with the same id is not a message to merge into.
    const items = replayed([assistant(reasoning(), text("A.", { id: "rs_1" }))]);
    expect(items).toEqual([REASONING_ITEM, assistantItem([outputText("A.")], "rs_1")]);
  });

  test("a refusal round-trips inside its message item", () => {
    const part = new RefusalPart("I can't help with that.", record({ id: "msg_1", phase: "final_answer" }));
    expect(replayed([assistant(part)])).toEqual([
      assistantItem([refusal("I can't help with that.")], "msg_1", "final_answer"),
    ]);
  });

  test("a refusal merges with text of the same message", () => {
    const message = assistant(text("Partly.", { id: "msg_1" }), new RefusalPart("No more.", record({ id: "msg_1" })));
    expect(replayed([message])).toEqual([assistantItem([outputText("Partly."), refusal("No more.")], "msg_1")]);
  });

  test("reasoning and a call round-trip into input", () => {
    const body = bodyOf(
      [
        system("You are terse."),
        user("Double 3."),
        assistant(reasoning(), call("call_1", { value: 3n }, CALL_ITEM)),
        message("tool", result("call_1", "6")),
      ],
      { tools: [DOUBLE] },
    );
    expect(body).toEqual({
      model: "gpt-5.5",
      input: [
        { type: "message", role: "system", content: "You are terse." },
        { type: "message", role: "user", content: "Double 3." },
        REASONING_ITEM,
        CALL_ITEM,
        { type: "function_call_output", call_id: "call_1", output: "6" },
      ],
      stream: true,
      tools: [{ type: "function", name: "double", description: "Double an integer.", parameters: DOUBLE.parameters }],
    });
  });
});

/** Payload overrides for `mixed`'s parts. */
interface MixedPayloads {
  reasoning?: JsonObject;
  text?: JsonObject;
  refusal?: JsonObject;
  call?: JsonObject;
}

const TEXT_FIELDS: JsonObject = { id: "msg_1", phase: "final_answer", annotations: [{ type: "x" }] };

/** An assistant message with one part of each kind, each with a usable record for `key` unless overridden. */
function mixed(payloads: MixedPayloads = {}, key: ReplayKey = KEY): Message {
  return assistant(
    new ReasoningPart("r", payloads.reasoning ?? record({ item: REASONING_ITEM }, key)),
    new TextPart("Hello.", payloads.text ?? record(TEXT_FIELDS, key)),
    new RefusalPart("No.", payloads.refusal ?? record({ id: "msg_1" }, key)),
    new ToolCallPart({
      id: "call_1",
      name: "double",
      arguments: { value: 3n },
      providerPayload: payloads.call ?? record({ item: CALL_ITEM }, key),
    }),
  );
}

const OWN: JsonValue[] = [
  REASONING_ITEM,
  assistantItem([outputText("Hello.", [{ type: "x" }]), refusal("No.")], "msg_1", "final_answer"),
  CALL_ITEM,
];
const NEUTRAL: JsonValue[] = [
  assistantItem([outputText("Hello.")]),
  assistantItem([refusal("No.")]),
  fallback("call_1", '{"value":3}'),
];

/** Render `messages` and return the items and the reasons passed to `onReplayDegraded`. */
function degraded(messages: Message[], options: BodyOptions = {}): [JsonValue[], ReplayDegradedReason[]] {
  const reasons: ReplayDegradedReason[] = [];
  const items = replayed(messages, { ...options, onReplayDegraded: (reason) => reasons.push(reason) });
  return [items, reasons];
}

describe("assistant messages: replay records", () => {
  test("usable records render the message from them, signalling nothing", () => {
    expect(degraded([mixed()])).toEqual([OWN, []]);
  });

  const other = (change: Partial<ReplayKey>) => record(TEXT_FIELDS, { ...KEY, ...change });
  const unusable: [string, MixedPayloads, ReplayDegradedReason][] = [
    ["another protocol", { text: other({ protocol: "anthropic_messages" }) }, "other_protocol"],
    ["another provider", { text: other({ provider: "azure" }) }, "other_provider"],
    ["another model", { text: other({ model: "gpt-5" }) }, "other_model"],
    ["an account the key lacks", { text: other({ account: "acct_1" }) }, "other_account"],
    ["an unknown version", { text: { ...record(TEXT_FIELDS), v: 2n } }, "unknown_version"],
    ["a malformed envelope", { text: { ...record(TEXT_FIELDS), provider: 5n } }, "malformed"],
    ["an unknown field", { refusal: record({ id: "msg_1", annotations: [] }) }, "unknown_field"],
    [
      "a reasoning item of the wrong type",
      { reasoning: record({ item: { ...REASONING_ITEM, type: "message" } }) },
      "malformed",
    ],
    ["a reasoning record without an item", { reasoning: record({}) }, "malformed"],
    ["a call item of the wrong type", { call: record({ item: { ...CALL_ITEM, type: "reasoning" } }) }, "malformed"],
    ["a call item for another call id", { call: record({ item: { ...CALL_ITEM, call_id: "call_2" } }) }, "malformed"],
    ["a non-string id", { text: record({ ...TEXT_FIELDS, id: 5n }) }, "malformed"],
    ["non-array annotations", { text: record({ ...TEXT_FIELDS, annotations: "x" }) }, "malformed"],
  ];
  for (const [name, payloads, reason] of unusable) {
    test(`a record with ${name} renders the whole message neutrally and signals ${reason} once`, () => {
      expect(degraded([mixed(payloads)])).toEqual([NEUTRAL, [reason]]);
    });
  }

  test("a key with an account refuses records without one", () => {
    const keyed = { ...KEY, account: "acct_1" };
    const message = mixed({ text: record(TEXT_FIELDS) }, keyed);
    expect(degraded([message], { account: "acct_1" })).toEqual([NEUTRAL, ["other_account"]]);
  });

  test("a matching record with an account replays", () => {
    const message = mixed({}, { ...KEY, account: "acct_1" });
    expect(degraded([message], { account: "acct_1" })).toEqual([OWN, []]);
  });

  test("the signal names the first unusable part's reason", () => {
    const message = mixed({
      text: record(TEXT_FIELDS, { ...KEY, model: "gpt-5" }),
      call: { ...record({ item: CALL_ITEM }), v: 2n },
    });
    expect(degraded([message])).toEqual([NEUTRAL, ["other_model"]]);
  });

  test("each degraded message signals once", () => {
    const bad = mixed({ text: record(TEXT_FIELDS, { ...KEY, model: "gpt-5" }) });
    const [, reasons] = degraded([bad, mixed(), bad]);
    expect(reasons).toEqual(["other_model", "other_model"]);
  });

  test("parts without payloads render neutrally and signal nothing", () => {
    const calls = call("call_1", { value: 3n });
    const bare = assistant(new ReasoningPart("r"), text("Hello."), new RefusalPart("No."), calls);
    expect(degraded([bare])).toEqual([NEUTRAL, []]);
  });

  for (const subscription of [false, true]) {
    test(`reasoning without a payload beside usable records renders the message neutrally without a signal (subscription: ${subscription})`, () => {
      const message = assistant(
        new ReasoningPart("r"),
        new TextPart("Hello.", record(TEXT_FIELDS)),
        new RefusalPart("No.", record({ id: "msg_1" })),
        new ToolCallPart({ id: "call_1", name: "double", arguments: { value: 3n }, providerPayload: record({ item: CALL_ITEM }) }),
      );
      expect(degraded([message], { subscription })).toEqual([NEUTRAL, []]);
    });
  }

  test("without a callback an unusable record still renders neutrally", () => {
    expect(replayed([mixed({ text: record(TEXT_FIELDS, { ...KEY, model: "gpt-5" }) })])).toEqual(NEUTRAL);
  });

  test("a throwing callback propagates from buildBody", () => {
    const message = mixed({ text: record(TEXT_FIELDS, { ...KEY, model: "gpt-5" }) });
    const onReplayDegraded = () => {
      throw new Error("stop");
    };
    expect(() => bodyOf([user("Hi."), message], { onReplayDegraded })).toThrow("stop");
  });

  test("a bigint and a number version are both accepted", () => {
    const withNumber = { ...record({ item: CALL_ITEM }), v: 1 };
    expect(degraded([mixed({ call: withNumber })])).toEqual([OWN, []]);
    expect(record({})["v"]).toBe(1n);
  });
});

describe("subscription assistant messages", () => {
  test("message ids are omitted, so parts never merge", () => {
    const message = assistant(
      text("Part A.", { id: "msg_2", phase: "final_answer" }),
      text("Part B.", { id: "msg_2", phase: "final_answer" }),
    );
    expect(replayed([message], { subscription: true })).toEqual([
      assistantItem([outputText("Part A.")], undefined, "final_answer"),
      assistantItem([outputText("Part B.")], undefined, "final_answer"),
    ]);
  });

  test("status is dropped from replayed calls", () => {
    const items = replayed([assistant(call("call_1", { value: 3n }, CALL_ITEM))], { subscription: true });
    expect(items).toEqual([withoutStatus(CALL_ITEM)]);
  });

  test("encrypted reasoning is replayed", () => {
    expect(degraded([mixed()], { subscription: true })).toEqual([
      [
        REASONING_ITEM,
        assistantItem([outputText("Hello.", [{ type: "x" }])], undefined, "final_answer"),
        assistantItem([refusal("No.")]),
        withoutStatus(CALL_ITEM),
      ],
      [],
    ]);
  });

  const { encrypted_content: _, ...plain } = REASONING_ITEM;
  const unencrypted = [plain, { ...plain, encrypted_content: "" }, { ...plain, encrypted_content: null }];
  for (const item of unencrypted) {
    const shown = JSON.stringify(item["encrypted_content"]);
    test(`reasoning without encrypted content renders the message neutrally without a signal (${shown})`, () => {
      expect(degraded([mixed({ reasoning: record({ item }) })], { subscription: true })).toEqual([NEUTRAL, []]);
    });
  }

  test("reasoning without encrypted content and an unusable later record signals the record's reason", () => {
    const message = mixed({ reasoning: record({ item: plain }), call: { ...record({ item: CALL_ITEM }), v: 2n } });
    expect(degraded([message], { subscription: true })).toEqual([NEUTRAL, ["unknown_version"]]);
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
    const open = new ToolSpec({
      name: "open",
      description: "Open schema.",
      parameters: { type: "object" },
      strict: false,
    });
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

  test("an Azure deployment name is sent as model", () => {
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
