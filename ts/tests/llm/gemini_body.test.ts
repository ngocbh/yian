/**
 * Tests for Gemini body building in `src/llm/protocols/gemini.ts`: the protocol object, the
 * contents (system, user, tool and assistant messages), tool results and their ids, the media and
 * wrong-part refusals, tools, the tool choice, the block, determinism, thought signatures sent back
 * from replay records and the placeholder signature.
 */

import { describe, expect, spyOn, test } from "bun:test";

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
  Request,
  replayRecord,
  type Role,
  TextPart,
  ToolCallPart,
  type ToolChoice,
  ToolResultPart,
  ToolSpec,
} from "../../src/llm/messages.ts";
import { Gemini, PROTOCOL } from "../../src/llm/protocols/gemini.ts";
import { type JsonObject, type JsonValue, parseJson } from "../../src/typesys/json.ts";

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
  return recorded(KEY, ...parts);
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

function gemini(block: JsonObject | null = null): Gemini {
  return new Gemini({ model: FLASH, provider: "google", block });
}

interface BodyOptions {
  model?: string;
  account?: string | null;
  block?: JsonObject | null;
  onReplayDegraded?: (reason: ReplayDegradedReason) => void;
  tools?: ToolSpec[];
  toolChoice?: ToolChoice;
  parallelToolCalls?: boolean | null;
}

function bodyOf(messages: Message[], options: BodyOptions = {}): JsonObject {
  const protocol = new Gemini({
    model: options.model ?? FLASH,
    provider: "google",
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
    }),
  );
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

  test("a message that adds no content does not split a run of tool messages", () => {
    const both = [
      {
        role: "user",
        parts: [
          { functionResponse: { id: "call_1", name: "double", response: { output: 1n } } },
          { functionResponse: { id: "call_2", name: "double", response: { output: 2n } } },
        ],
      },
    ];
    const body = bodyOf([tool(result("call_1", "1")), system("S."), tool(result("call_2", "2"))]);
    expect(body["contents"]).toEqual(both);
    expect(body["systemInstruction"]).toEqual({ parts: [{ text: "S." }] });
    let sent = null as JsonValue;
    const emitted = warnings(() => {
      sent = contents([tool(result("call_1", "1")), user(""), tool(result("call_2", "2"))]);
    });
    expect(sent).toEqual(both);
    expect(emitted).toEqual([[EMPTY_USER]]);
    const thinking = own(new ReasoningPart("Thinking."));
    expect(contents([tool(result("call_1", "1")), thinking, tool(result("call_2", "2"))])).toEqual(both);
    expect(contents([tool(result("call_1", "1")), message("tool"), tool(result("call_2", "2"))])).toEqual(both);
  });

  test("an empty tool message adds nothing", () => {
    expect(contents([user("Hi."), message("tool")])).toEqual([{ role: "user", parts: [{ text: "Hi." }] }]);
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

describe("tools", () => {
  test("every tool goes in one functionDeclarations entry, in order, description always", () => {
    const blank = new ToolSpec({ name: "blank", description: "", parameters: { type: "object" } });
    expect(bodyOf([user("Hi.")], { tools: [DOUBLE, blank] })["tools"]).toEqual([
      {
        functionDeclarations: [
          {
            name: "double",
            description: "Double an integer.",
            parametersJsonSchema: {
              type: "object",
              properties: { value: { type: "integer" } },
              required: ["value"],
            },
          },
          { name: "blank", description: "", parametersJsonSchema: { type: "object" } },
        ],
      },
    ]);
  });

  test("parameters go as written", () => {
    const parameters: JsonObject = {
      title: "Pick",
      type: "object",
      properties: {
        kind: { const: "a" },
        count: { type: "integer", minimum: 0n, default: 1n },
        value: { anyOf: [{ type: "integer" }, { type: "string" }] },
      },
      additionalProperties: false,
    };
    const pick = new ToolSpec({ name: "pick", description: "Pick.", parameters });
    const [entry] = bodyOf([user("Hi.")], { tools: [pick] })["tools"] as JsonObject[];
    const [declaration] = entry!["functionDeclarations"] as JsonObject[];
    expect(declaration!["parametersJsonSchema"]).toEqual(parameters);
    // toEqual ignores key order; the bytes pin it.
    expect(encodeBody(declaration!["parametersJsonSchema"]!)).toBe(encodeBody(parameters));
  });

  test("tool parameters are copied", () => {
    const [entry] = bodyOf([user("Hi.")], { tools: [DOUBLE] })["tools"] as JsonObject[];
    const [declaration] = entry!["functionDeclarations"] as JsonObject[];
    ((declaration!["parametersJsonSchema"] as JsonObject)["required"] as JsonValue[]).push("x");
    expect(DOUBLE.parameters["required"]).toEqual(["value"]);
  });

  test("strict is not sent", () => {
    const strict = new ToolSpec({ name: "s", description: "S.", parameters: { type: "object" }, strict: true });
    const body = bodyOf([user("Hi.")], { tools: [strict] });
    expect(body["tools"]).toEqual([
      { functionDeclarations: [{ name: "s", description: "S.", parametersJsonSchema: { type: "object" } }] },
    ]);
    expect(encodeBody(body)).not.toContain("strict");
  });

  test("no tools sends no tools", () => {
    expect(Object.hasOwn(bodyOf([user("Hi.")]), "tools")).toBe(false);
  });
});

describe("tool choice", () => {
  const choices: [string, ToolChoice, JsonObject][] = [
    ["required", "required", { functionCallingConfig: { mode: "ANY" } }],
    ["none", "none", { functionCallingConfig: { mode: "NONE" } }],
    [
      'NamedTool("double")',
      new NamedTool("double"),
      { functionCallingConfig: { mode: "ANY", allowedFunctionNames: ["double"] } },
    ],
  ];
  for (const [label, choice, expected] of choices) {
    test(`tool choice ${label} -> ${JSON.stringify(expected)}`, () => {
      expect(bodyOf([user("Hi.")], { tools: [DOUBLE], toolChoice: choice })["toolConfig"]).toEqual(expected);
    });
  }

  test("a null or auto choice sends no toolConfig", () => {
    for (const toolChoice of [null, "auto"] as ToolChoice[]) {
      expect(Object.hasOwn(bodyOf([user("Hi.")], { tools: [DOUBLE], toolChoice }), "toolConfig")).toBe(false);
    }
  });

  test("no tools sends no toolConfig, even with a choice set", () => {
    for (const [, toolChoice] of choices) {
      const body = bodyOf([user("Hi.")], { toolChoice });
      expect(Object.hasOwn(body, "toolConfig")).toBe(false);
      expect(Object.hasOwn(body, "tools")).toBe(false);
    }
  });

  test("parallelToolCalls is never sent", () => {
    for (const parallelToolCalls of [false, true]) {
      for (const toolChoice of [null, "auto", "required", "none", new NamedTool("double")] as ToolChoice[]) {
        const body = bodyOf([user("Hi.")], { tools: [DOUBLE], toolChoice, parallelToolCalls });
        expect(encodeBody(body)).not.toMatch(/parallel/i);
      }
    }
  });
});

describe("the block", () => {
  test("block is merged last: a generationConfig beside candidateCount, safetySettings replaced", () => {
    const block: JsonObject = {
      generationConfig: { thinkingConfig: { thinkingBudget: 1024n, includeThoughts: true }, maxOutputTokens: 512n },
      safetySettings: [{ category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" }],
    };
    const body = bodyOf([system("S."), user("Hi.")], { block, tools: [DOUBLE], toolChoice: "required" });
    expect(body).toEqual({
      contents: [{ role: "user", parts: [{ text: "Hi." }] }],
      systemInstruction: { parts: [{ text: "S." }] },
      tools: [
        {
          functionDeclarations: [
            {
              name: "double",
              description: "Double an integer.",
              parametersJsonSchema: {
                type: "object",
                properties: { value: { type: "integer" } },
                required: ["value"],
              },
            },
          ],
        },
      ],
      toolConfig: { functionCallingConfig: { mode: "ANY" } },
      safetySettings: [{ category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" }],
      generationConfig: {
        candidateCount: 1n,
        thinkingConfig: { thinkingBudget: 1024n, includeThoughts: true },
        maxOutputTokens: 512n,
      },
    });
    expect(Object.keys(body)).toEqual([
      "contents",
      "systemInstruction",
      "tools",
      "toolConfig",
      "safetySettings",
      "generationConfig",
    ]);
  });

  test("a block candidateCount replaces yian's", () => {
    // Refusing a count other than 1 is the route's job, not the protocol's.
    const body = bodyOf([user("Hi.")], { block: { generationConfig: { candidateCount: 3n } } });
    expect(body["generationConfig"]).toEqual({ candidateCount: 3n });
  });

  test("a null block leaf is not sent and a __proto__ key stays an ordinary key", () => {
    // Built with parseJson: an object literal's "__proto__" would set the prototype instead.
    const block = parseJson(
      '{"cachedContent": null, "generationConfig": {"temperature": null, "__proto__": "x"}}',
      "plain",
    ) as JsonObject;
    const body = bodyOf([user("Hi.")], { block });
    expect(Object.hasOwn(body, "cachedContent")).toBe(false);
    const config = body["generationConfig"] as JsonObject;
    expect(Object.hasOwn(config, "temperature")).toBe(false);
    expect(Object.hasOwn(config, "__proto__")).toBe(true);
    expect(encodeBody(body)).toContain('"generationConfig":{"candidateCount":1,"__proto__":"x"}');
  });

  test("block is not shared with the body", () => {
    const block: JsonObject = { safetySettings: [{ category: "C", threshold: "T" }] };
    const protocol = gemini(block);
    const request = new Request({ messages: [user("Hi.")] });
    const body = protocol.buildBody(request);
    ((body["safetySettings"] as JsonObject[])[0]!)["threshold"] = "changed";
    expect(block).toEqual({ safetySettings: [{ category: "C", threshold: "T" }] });
    // This one checks sharing: a body holding the protocol's copy of the block would change the next body.
    expect(protocol.buildBody(request)["safetySettings"]).toEqual([{ category: "C", threshold: "T" }]);
  });

  test("later changes to the caller's block do not change bodies", () => {
    const block: JsonObject = { generationConfig: { temperature: 0.2 } };
    const protocol = gemini(block);
    (block["generationConfig"] as JsonObject)["temperature"] = 0.9;
    block["safetySettings"] = [{ category: "C", threshold: "T" }];
    const body = protocol.buildBody(new Request({ messages: [user("Hi.")] }));
    expect(body["generationConfig"]).toEqual({ candidateCount: 1n, temperature: 0.2 });
    expect(body["safetySettings"]).toEqual([]);
  });
});

describe("determinism", () => {
  const history = [
    system("S."),
    user("Double 3."),
    own(text("Calling."), call("call_1", { value: 3n })),
    tool(result("call_1", "6")),
    own(text("It is 6.")),
  ];

  test("the same request built twice encodes to the same bytes", () => {
    const request = new Request({ messages: history, tools: [DOUBLE], toolChoice: "required" });
    const protocol = gemini({ generationConfig: { temperature: 0.2 } });
    expect(encodeBody(protocol.buildBody(request))).toBe(encodeBody(protocol.buildBody(request)));
  });

  test("appending a user turn after a tool loop leaves every earlier content equal", () => {
    const before = bodyOf(history);
    const after = bodyOf([...history, user("Thanks.")]);
    const earlier = before["contents"] as JsonObject[];
    const later = after["contents"] as JsonObject[];
    expect(later).toHaveLength(earlier.length + 1);
    expect(later.slice(0, earlier.length).map((each) => encodeBody(each))).toEqual(
      earlier.map((each) => encodeBody(each)),
    );
    expect(encodeBody(after["systemInstruction"]!)).toBe(encodeBody(before["systemInstruction"]!));
  });
});

const PRO = "gemini-3-pro-preview";
const SIG = "c2lnLTE=";
const SIG2 = "c2lnLTI=";
const PLACEHOLDER = "skip_thought_signature_validator";
const ARGS: JsonObject = { value: 3n };
const FUNCTION_CALL = { functionCall: { id: "call_1", name: "double", args: { value: 3n } } };
const SECOND_CALL = { functionCall: { id: "call_2", name: "double", args: { value: 4n } } };

/** Return a replay record for `key` (this protocol's key by default) holding `fields`. */
function record(fields: JsonObject, key: ReplayKey = KEY): JsonObject {
  return replayRecord(key, fields);
}

/** Return a text part carrying a signature record for `key`. */
function signedText(value: string, signature: string, key: ReplayKey = KEY): TextPart {
  return new TextPart(value, record({ thought_signature: signature }, key));
}

/** Return a call part carrying `payload`. */
function callWith(id: string, args: JsonObject, payload: JsonObject | null): ToolCallPart {
  return new ToolCallPart({ id, name: "double", arguments: args, providerPayload: payload });
}

/** Return a call part carrying a signature record for `key`. */
function signedCall(id: string, args: JsonObject, signature: string, key: ReplayKey = KEY): ToolCallPart {
  return callWith(id, args, record({ thought_signature: signature }, key));
}

/** Return an assistant message carrying a message record for `key`. */
function recorded(key: ReplayKey, ...parts: Part[]): Message {
  return new Message({ role: "assistant", parts, protocol: key.protocol, providerPayload: record({}, key) });
}

/** Build a body from `messages`; return its contents and the reasons passed to `onReplayDegraded`. */
function sendWithReasons(messages: Message[], options: BodyOptions = {}): [JsonObject[], ReplayDegradedReason[]] {
  const reasons: ReplayDegradedReason[] = [];
  const body = bodyOf(messages, { ...options, onReplayDegraded: (reason) => reasons.push(reason) });
  return [body["contents"] as JsonObject[], reasons];
}

/** Send `assistant` after a user turn; return its model parts (`null` when skipped) and the reasons. */
function modelParts(assistant: Message, options: BodyOptions = {}): [JsonValue, ReplayDegradedReason[]] {
  const [sent, reasons] = sendWithReasons([user("Hi."), assistant], options);
  return [sent.length > 1 ? sent[1]!["parts"]! : null, reasons];
}

describe("thought signatures sent back", () => {
  test("a signed text part and a signed call go back with thoughtSignature in place", () => {
    const parts = [signedText("Calling.", SIG), signedCall("call_1", ARGS, SIG2), text("Done.")];
    expect(modelParts(own(...parts))).toEqual([
      [{ text: "Calling.", thoughtSignature: SIG }, { ...FUNCTION_CALL, thoughtSignature: SIG2 }, { text: "Done." }],
      [],
    ]);
  });

  test("an empty text part with a usable signature is sent as empty text with it", () => {
    expect(modelParts(own(text("Hello."), signedText("", SIG)))).toEqual([
      [{ text: "Hello." }, { text: "", thoughtSignature: SIG }],
      [],
    ]);
  });

  test("an empty text part with an unusable record is not sent", () => {
    const parts = [text("Hello."), signedText("", SIG, { ...KEY, model: PRO })];
    expect(modelParts(own(...parts))).toEqual([[{ text: "Hello." }], ["other_model"]]);
  });

  test("a signed reasoning part is not sent", () => {
    const parts = [new ReasoningPart("Thinking.", record({ thought_signature: SIG })), text("Hello.")];
    expect(modelParts(own(...parts))).toEqual([[{ text: "Hello." }], []]);
  });

  test("a tool result is refused before an unusable record is signalled", () => {
    const reasons: ReplayDegradedReason[] = [];
    refuses(
      () =>
        bodyOf([own(signedText("A.", SIG, { ...KEY, model: PRO }), result("call_1", "6"))], {
          onReplayDegraded: (reason) => reasons.push(reason),
        }),
      "the gemini route cannot send a ToolResultPart in an assistant message",
    );
    expect(reasons).toEqual([]);
  });

  const signed = record({ thought_signature: SIG });
  const signedFor = (change: Partial<ReplayKey>) => record({ thought_signature: SIG }, { ...KEY, ...change });
  const anthropic = { protocol: "anthropic_messages", provider: "anthropic", model: "claude-sonnet-4-5" };
  const unusable: [string, JsonObject, ReplayDegradedReason, string | null][] = [
    ["an anthropic_messages record", record({ signature: SIG }, anthropic), "other_protocol", null],
    ["another provider", signedFor({ provider: "vertex" }), "other_provider", null],
    ["another model", signedFor({ model: PRO }), "other_model", null],
    ["a record with an account, a key without", signedFor({ account: "acct" }), "other_account", null],
    ["a record without an account, a key with one", signed, "other_account", "acct"],
    ["an unknown version", { ...signed, v: 2n }, "unknown_version", null],
    ["a malformed envelope", { ...signed, provider: 1n }, "malformed", null],
    ["an unknown field", record({ thought_signature: SIG, extra: "x" }), "unknown_field", null],
    ["no thought_signature", record({}), "malformed", null],
    ["an empty thought_signature", record({ thought_signature: "" }), "malformed", null],
    ["a non-string thought_signature", record({ thought_signature: 1n }), "malformed", null],
  ];
  for (const [name, payload, reason, account] of unusable) {
    test(`${name}: no signature from the message, text and calls kept, signalled once`, () => {
      // The call's own record is usable; the whole message degrades, so the call goes out with
      // the placeholder instead.
      const usable = account === null ? KEY : { ...KEY, account };
      const parts = [new TextPart("Calling.", payload), signedCall("call_1", ARGS, SIG2, usable), text("Done.")];
      expect(modelParts(own(...parts), { account })).toEqual([
        [{ text: "Calling." }, { ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER }, { text: "Done." }],
        [reason],
      ]);
    });
  }

  test("two unusable parts signal once, with the first reason in part order", () => {
    const parts = [
      signedText("A.", SIG, { ...KEY, model: PRO }),
      signedCall("call_1", ARGS, SIG2, { ...KEY, provider: "vertex" }),
    ];
    expect(modelParts(own(...parts))[1]).toEqual(["other_model"]);
  });

  test("a usable record beside an unusable one: neither is sent", () => {
    const parts = [signedText("A.", SIG), signedText("B.", SIG2, { ...KEY, model: PRO })];
    expect(modelParts(own(...parts))).toEqual([[{ text: "A." }, { text: "B." }], ["other_model"]]);
  });

  test("without a callback the signatures are still left out", () => {
    const parts = [signedText("A.", SIG), signedText("B.", SIG2, { ...KEY, model: PRO })];
    expect(contents([own(...parts)])).toEqual([{ role: "model", parts: [{ text: "A." }, { text: "B." }] }]);
  });

  test("a throwing callback propagates", () => {
    const onReplayDegraded = () => {
      throw new Error("stop");
    };
    const messages = [user("Hi."), own(signedText("A.", SIG, { ...KEY, model: PRO }))];
    expect(() => bodyOf(messages, { onReplayDegraded })).toThrow("stop");
  });

  test("a matching record with an account replays", () => {
    const key = { ...KEY, account: "acct" };
    const parts = [signedText("Calling.", SIG, key), signedCall("call_1", ARGS, SIG2, key)];
    expect(modelParts(recorded(key, ...parts), { account: "acct" })).toEqual([
      [{ text: "Calling.", thoughtSignature: SIG }, { ...FUNCTION_CALL, thoughtSignature: SIG2 }],
      [],
    ]);
  });

  test("a bigint and a number version are both accepted", () => {
    const parts = [
      new TextPart("A.", { ...signed, v: 1n }),
      new TextPart("B.", { ...record({ thought_signature: SIG2 }), v: 1 }),
    ];
    expect(modelParts(own(...parts))).toEqual([
      [
        { text: "A.", thoughtSignature: SIG },
        { text: "B.", thoughtSignature: SIG2 },
      ],
      [],
    ]);
  });

  test("a part without a payload has no signature and signals nothing", () => {
    const parts = [signedText("A.", SIG), text("B."), call("call_1", ARGS)];
    expect(modelParts(own(...parts))).toEqual([
      [{ text: "A.", thoughtSignature: SIG }, { text: "B." }, FUNCTION_CALL],
      [],
    ]);
  });

  test("a reasoning part with a foreign record signals nothing", () => {
    const thinking = new ReasoningPart("Thinking.", record({ thought_signature: SIG }, { ...KEY, model: PRO }));
    expect(modelParts(own(thinking, signedText("A.", SIG2)))).toEqual([[{ text: "A.", thoughtSignature: SIG2 }], []]);
  });
});

describe("the placeholder signature", () => {
  /** An `openai_chat` reply calling `double`: its own message record, no part records. */
  const openaiChat = new Message({
    role: "assistant",
    parts: [text("Calling."), call("call_1", ARGS)],
    protocol: "openai_chat",
    providerPayload: record(
      { response_id: "chatcmpl-1" },
      { protocol: "openai_chat", provider: "openai", model: "gpt-4o" },
    ),
  });

  for (const model of [FLASH, PRO]) {
    test(`a call from openai_chat gets the placeholder on ${model}, signalling nothing`, () => {
      expect(modelParts(openaiChat, { model })).toEqual([
        [{ text: "Calling." }, { ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER }],
        [],
      ]);
    });
  }

  test("a hand-built message with no payload gets the placeholder, signalling nothing", () => {
    expect(modelParts(message("assistant", call("call_1", ARGS)))).toEqual([
      [{ ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER }],
      [],
    ]);
  });

  test("this model's own reply with an unsigned call goes as received", () => {
    // Gemini 2.5 signs the first part, here the text, and leaves the call unsigned.
    expect(modelParts(own(signedText("Hello", SIG), call("call_1", ARGS)))).toEqual([
      [{ text: "Hello", thoughtSignature: SIG }, FUNCTION_CALL],
      [],
    ]);
    // With thinking off it signs nothing.
    expect(modelParts(own(text("Hello"), call("call_1", ARGS)))).toEqual([[{ text: "Hello" }, FUNCTION_CALL], []]);
  });

  test("a reply from another model: its call record degrades, and the call gets the placeholder", () => {
    const pro = { ...KEY, model: PRO };
    expect(modelParts(recorded(pro, text("Calling."), signedCall("call_1", ARGS, SIG, pro)))).toEqual([
      [{ text: "Calling." }, { ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER }],
      ["other_model"],
    ]);
  });

  test("a reply for another account gets the placeholder", () => {
    const other = { ...KEY, account: "acct-2" };
    expect(modelParts(recorded(other, call("call_1", ARGS)), { account: "acct-1" })).toEqual([
      [{ ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER }],
      [],
    ]);
    expect(modelParts(recorded(other, call("call_1", ARGS)))).toEqual([
      [{ ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER }],
      [],
    ]);
  });

  test("this model's own reply whose call record is malformed gets the placeholder", () => {
    const bad = callWith("call_1", ARGS, record({ thought_signature: 1n }));
    expect(modelParts(own(text("Calling."), bad))).toEqual([
      [{ text: "Calling." }, { ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER }],
      ["malformed"],
    ]);
  });

  test("two unsigned parallel calls in a foreign message: only the first gets it", () => {
    expect(modelParts(message("assistant", call("call_1", ARGS), call("call_2", { value: 4n })))).toEqual([
      [{ ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER }, SECOND_CALL],
      [],
    ]);
  });

  test("a foreign message whose first call is unsigned and second signed: only the first gets the placeholder", () => {
    const pro = { ...KEY, model: PRO };
    const parts = [call("call_1", ARGS), signedCall("call_2", { value: 4n }, SIG)];
    expect(modelParts(recorded(pro, ...parts))).toEqual([
      [
        { ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER },
        { ...SECOND_CALL, thoughtSignature: SIG },
      ],
      [],
    ]);
  });

  test("a foreign message whose first call is signed gets none", () => {
    const pro = { ...KEY, model: PRO };
    const parts = [signedCall("call_1", ARGS, SIG), call("call_2", { value: 4n })];
    expect(modelParts(recorded(pro, ...parts))).toEqual([
      [{ ...FUNCTION_CALL, thoughtSignature: SIG }, SECOND_CALL],
      [],
    ]);
  });

  test("each foreign model content in a two-step loop gets one", () => {
    const [sent, reasons] = sendWithReasons([
      user("Double 3, then double 6."),
      message("assistant", call("call_1", ARGS)),
      tool(result("call_1", "6")),
      message("assistant", call("call_2", { value: 4n }), call("call_3", { value: 5n })),
      tool(result("call_2", "8"), result("call_3", "10")),
    ]);
    expect(sent[1]!["parts"]).toEqual([{ ...FUNCTION_CALL, thoughtSignature: PLACEHOLDER }]);
    expect(sent[3]!["parts"]).toEqual([
      { ...SECOND_CALL, thoughtSignature: PLACEHOLDER },
      { functionCall: { id: "call_3", name: "double", args: { value: 5n } } },
    ]);
    expect(reasons).toEqual([]);
  });

  test("a foreign model content with text only gets none", () => {
    expect(modelParts(message("assistant", text("Hello.")))).toEqual([[{ text: "Hello." }], []]);
    const chat = new Message({ role: "assistant", parts: [text("Hi.")], providerPayload: openaiChat.providerPayload });
    expect(modelParts(chat)).toEqual([[{ text: "Hi." }], []]);
  });
});
