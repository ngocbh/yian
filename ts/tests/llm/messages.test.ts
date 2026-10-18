/**
 * Messages: the stable JSON shape, strict decoding and construction checks, a port of
 * `tests/llm/test_messages.py`, plus JavaScript's own values: ints of any size, `-0`, a
 * `"__proto__"` key, non-plain objects, `undefined` and array holes.
 */

import { afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";
import {
  checkJsonValue,
  isPart,
  MESSAGE_FORMAT,
  MediaPart,
  Message,
  MessageFormatError,
  ReasoningPart,
  RefusalPart,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from "../../src/llm/messages.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";

/** JSON data a test edits freely. */
type Data = { [key: string]: any };

/** Return `value` typed as whatever a constructor wants, to pass it values its type forbids. */
const bad = (value: unknown): never => value as never;

function fullMessage(): Message {
  return new Message({
    role: "assistant",
    parts: [
      new ReasoningPart("thinking", { signature: "abc", n: [1, 2.5, null] }),
      new TextPart("hello ", { id: "msg_1" }),
      new TextPart("world"),
      new MediaPart({ mime: "image/png", data: "aGVsbG8=", name: "pic.png", providerPayload: { detail: "low" } }),
      new MediaPart({ mime: "image/jpeg", url: "https://example.com/a.jpg" }),
      new ToolCallPart({ id: "call_1", name: "search", arguments: { q: "x", deep: { k: [true, false] } } }),
      new ToolCallPart({ id: "call_2", name: "broken", arguments: null, rawArguments: "{oops" }),
      new ToolCallPart({ id: "call_3", name: "both", arguments: { a: 1n }, rawArguments: '{"a": 1}' }),
      new ToolResultPart({ callId: "call_1", name: "search", content: "result", isError: true, providerPayload: { x: 1n } }),
      new ToolResultPart({ callId: "call_2", name: "broken", content: "ok" }),
      new RefusalPart("I can't help with that", { id: "msg_2", phase: "final" }),
    ],
    protocol: "anthropic_messages",
    providerPayload: { stop_sequence: null, meta: { k: "v" } },
  });
}

function canonical(): Data {
  return {
    format: MESSAGE_FORMAT,
    role: "assistant",
    parts: [
      { type: "reasoning", text: "t", provider_payload: { signature: "s" } },
      { type: "text", text: "hi" },
      { type: "media", mime: "image/png", url: "https://e.com/x.png" },
      { type: "media", mime: "image/png", data: "AAAA", name: "x" },
      { type: "tool_call", id: "c1", name: "f", arguments: { a: [1n, 2n] } },
      { type: "tool_call", id: "c2", name: "g", raw_arguments: "{bad" },
      { type: "tool_result", call_id: "c1", name: "f", content: "r", is_error: true },
      { type: "tool_result", call_id: "c2", name: "g", content: "r" },
      { type: "refusal", text: "no", provider_payload: { id: "msg_1" } },
    ],
    protocol: "openai_responses",
    provider_payload: { id: "resp_1" },
  };
}

/** Return the `MessageFormatError` that `run` throws, or fail. */
function formatError(run: () => unknown): MessageFormatError {
  try {
    run();
  } catch (error) {
    if (error instanceof MessageFormatError) return error;
    throw error;
  }
  throw new Error("no MessageFormatError was thrown");
}

/** Return the path of the error decoding `data` throws. */
function errorPath(data: unknown): string {
  return formatError(() => Message.fromJSON(data)).path;
}

/** Return the message read back from `message`'s run-log JSON. */
function throughCodec(message: Message): Message {
  return Message.fromJSON(parseJson(logJson(message.toJSON()), "codec"));
}

afterEach(() => install(BUILTIN));

describe("round trip", () => {
  test("every part kind round-trips through JSON", () => {
    const message = fullMessage();
    const data = message.toJSON();
    expect(data["format"]).toBe("yian.message.v1");
    expect(Message.fromJSON(data)).toEqual(message);
    expect(throughCodec(message)).toEqual(message);
  });

  test("the canonical form round-trips", () => {
    const data = canonical();
    expect(Message.fromJSON(data).toJSON()).toEqual(data);
  });

  test("defaults are omitted", () => {
    const message = new Message({
      role: "user",
      parts: [new TextPart("hi"), new ToolResultPart({ callId: "c", name: "f", content: "r" })],
    });
    expect(message.toJSON()).toEqual({
      format: MESSAGE_FORMAT,
      role: "user",
      parts: [
        { type: "text", text: "hi" },
        { type: "tool_result", call_id: "c", name: "f", content: "r" },
      ],
    });
  });

  test("protocol round-trips", () => {
    const message = new Message({ role: "assistant", parts: [new TextPart("x")], protocol: "catalog:minimax" });
    expect(message.toJSON()["protocol"]).toBe("catalog:minimax");
    expect(Message.fromJSON(message.toJSON()).protocol).toBe("catalog:minimax");
  });

  test("a refusal part round-trips", () => {
    const message = new Message({ role: "assistant", parts: [new RefusalPart("no", { id: "m" })] });
    expect(message.toJSON()["parts"]).toEqual([{ type: "refusal", text: "no", provider_payload: { id: "m" } }]);
    expect(Message.fromJSON(message.toJSON())).toEqual(message);
    expect(Message.fromJSON(message.toJSON()).parts[0]).toBeInstanceOf(RefusalPart);
  });

  test("payload records shaped like the codec's tags round-trip through the codec", () => {
    const payload = { inf: { $float: "inf" }, rec: { $record: 1n } };
    const message = new Message({ role: "assistant", parts: [new TextPart("x", payload)] });
    const line = logJson(message.toJSON());
    expect(line).toContain('{"$record":{"$float":"inf"}}');
    expect(line).toContain('{"$record":{"$record":1}}');
    const back = throughCodec(message);
    expect(back).toEqual(message);
    expect((back.parts[0] as TextPart).providerPayload).toEqual(payload);
  });
});

describe("strict decode", () => {
  test("an unknown top-level key", () => {
    const data = canonical();
    data["extra"] = 1;
    expect(errorPath(data)).toBe("extra");
  });

  test("an unknown part key", () => {
    const data = canonical();
    data["parts"][1]["bold"] = true;
    expect(errorPath(data)).toBe("parts[1].bold");
  });

  test("an unknown part type", () => {
    const data = canonical();
    data["parts"][2]["type"] = "video";
    expect(errorPath(data)).toBe("parts[2].type");
    expect(formatError(() => Message.fromJSON(data)).reason).toBe("unknown part type");
  });

  test("a missing format", () => {
    const data = canonical();
    delete data["format"];
    expect(formatError(() => Message.fromJSON(data)).message).toBe("format: missing");
  });

  test("an old format", () => {
    const data = canonical();
    data["format"] = "yian.message.v0";
    expect(formatError(() => Message.fromJSON(data)).message).toBe("format: expected 'yian.message.v1'");
  });

  test("media with data and url", () => {
    const data = canonical();
    data["parts"][2]["data"] = "AAAA";
    expect(errorPath(data)).toBe("parts[2]");
  });

  test("media with neither", () => {
    const data = canonical();
    delete data["parts"][2]["url"];
    expect(formatError(() => Message.fromJSON(data)).message).toBe(
      "parts[2]: media needs exactly one of data or url",
    );
  });

  test("a tool call without arguments", () => {
    const data = canonical();
    delete data["parts"][5]["raw_arguments"];
    expect(formatError(() => Message.fromJSON(data)).message).toBe(
      "parts[5].arguments: a tool call needs arguments or raw_arguments",
    );
  });

  test("a payload holding a Map", () => {
    const data = canonical();
    data["parts"][0]["provider_payload"] = { a: [1n, new Map()] };
    expect(errorPath(data)).toBe("parts[0].provider_payload.a[1]");
  });

  test("a payload holding NaN", () => {
    const data = canonical();
    data["provider_payload"] = { x: NaN };
    expect(formatError(() => Message.fromJSON(data)).message).toBe("provider_payload.x: float is not finite");
  });

  test("a payload with a symbol key", () => {
    const data = canonical();
    data["parts"][1]["provider_payload"] = { ok: { [Symbol("k")]: "v" } };
    expect(formatError(() => Message.fromJSON(data)).message).toBe(
      "parts[1].provider_payload.ok: object key is symbol, not string",
    );
  });

  test("arguments holding Infinity", () => {
    const data = canonical();
    data["parts"][4]["arguments"] = { a: [1n, Infinity] };
    expect(errorPath(data)).toBe("parts[4].arguments.a[1]");
  });

  test("an unknown role", () => {
    const data = canonical();
    data["role"] = "bot";
    expect(formatError(() => Message.fromJSON(data)).message).toBe("role: expected system, user, assistant or tool");
  });

  test("a wrong type", () => {
    const data = canonical();
    data["parts"][1]["text"] = 3n;
    expect(formatError(() => Message.fromJSON(data)).message).toBe("parts[1].text: expected string, got bigint");
  });

  test("is_error must be a boolean", () => {
    const data = canonical();
    data["parts"][6]["is_error"] = "yes";
    expect(formatError(() => Message.fromJSON(data)).message).toBe("parts[6].is_error: expected boolean, got string");
    data["parts"][6]["is_error"] = null;
    expect(formatError(() => Message.fromJSON(data)).message).toBe("parts[6].is_error: expected boolean, got null");
  });

  test("a missing part field", () => {
    const data = canonical();
    delete data["parts"][4]["id"];
    expect(formatError(() => Message.fromJSON(data)).message).toBe("parts[4].id: missing");
  });

  test("parts not an array", () => {
    const data = canonical();
    data["parts"] = { type: "text" };
    expect(formatError(() => Message.fromJSON(data)).message).toBe("parts: expected an array, got object");
  });

  test("not an object", () => {
    expect(errorPath([])).toBe("");
    expect(formatError(() => Message.fromJSON([])).message).toBe("expected an object, got array");
  });

  test("the message names the path", () => {
    const data = canonical();
    data["parts"][4]["arguments"] = { a: new Map() };
    expect(() => Message.fromJSON(data)).toThrow(/^parts\[4\]\.arguments\.a: Map is not a JSON value$/);
  });

  test("MessageFormatError is an Error with its name", () => {
    const error = new MessageFormatError("p", "r");
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("MessageFormatError");
  });

  test("an explicit null for a key toJSON omits", () => {
    const cases: [(data: Data) => void, string, string][] = [
      [(data) => (data["protocol"] = null), "protocol", "expected string, got null"],
      [(data) => (data["provider_payload"] = null), "provider_payload", "expected an object, got null"],
      [(data) => (data["parts"][0]["provider_payload"] = null), "parts[0].provider_payload", "expected an object, got null"],
      [(data) => (data["parts"][2]["data"] = null), "parts[2].data", "expected string, got null"],
      [(data) => (data["parts"][3]["url"] = null), "parts[3].url", "expected string, got null"],
      [(data) => (data["parts"][2]["name"] = null), "parts[2].name", "expected string, got null"],
      [(data) => (data["parts"][4]["raw_arguments"] = null), "parts[4].raw_arguments", "expected string, got null"],
      [(data) => (data["parts"][5]["arguments"] = null), "parts[5].arguments", "expected an object, got null"],
    ];
    for (const [edit, path, reason] of cases) {
      const data = canonical();
      edit(data);
      const error = formatError(() => Message.fromJSON(data));
      expect([error.path, error.reason]).toEqual([path, reason]);
    }
  });
});

describe("strict decode of odd objects", () => {
  test("a symbol key on the message", () => {
    const data = canonical();
    data[Symbol("s") as never] = 1;
    const error = formatError(() => Message.fromJSON(data));
    expect([error.path, error.reason]).toEqual(["", "object key is symbol, not string"]);
  });

  test("a symbol key on a part", () => {
    const data = canonical();
    data["parts"][1][Symbol("s")] = 1;
    const error = formatError(() => Message.fromJSON(data));
    expect([error.path, error.reason]).toEqual(["parts[1]", "object key is symbol, not string"]);
  });

  test("each key is read once", () => {
    const data = canonical();
    const reads = { payload: 0, protocol: 0 };
    // A second read would answer null, which the explicit-null check refuses.
    Object.defineProperty(data["parts"][1], "provider_payload", {
      get: () => (++reads.payload === 1 ? { id: "p" } : null),
      enumerable: true,
    });
    Object.defineProperty(data, "protocol", {
      get: () => (++reads.protocol === 1 ? "p" : null),
      enumerable: true,
    });
    const message = Message.fromJSON(data);
    expect(reads).toEqual({ payload: 1, protocol: 1 });
    expect((message.parts[1] as TextPart).providerPayload).toEqual({ id: "p" });
    expect(message.protocol).toBe("p");
  });

  test("a message or part that isn't a plain object", () => {
    expect(formatError(() => Message.fromJSON(Object.assign(new (class {})(), canonical()))).message).toBe(
      "expected an object, got object",
    );
    const data = canonical();
    data["parts"][1] = Object.assign(Object.create(Object.create(null)) as Data, data["parts"][1]);
    expect(errorPath(data)).toBe("parts[1]");
  });
});

describe("MessageFormatError", () => {
  test("the message is the path and reason, or the reason alone", () => {
    expect(new MessageFormatError("parts[1].text", "missing").message).toBe("parts[1].text: missing");
    expect(new MessageFormatError("", "media needs exactly one of data or url").message).toBe(
      "media needs exactly one of data or url",
    );
  });

  test("within places the path under a prefix", () => {
    expect(new MessageFormatError("text", "r").within("parts[1]").path).toBe("parts[1].text");
    expect(new MessageFormatError("[2]", "r").within("parts[1]").path).toBe("parts[1][2]");
    expect(new MessageFormatError("", "r").within("parts[1]").path).toBe("parts[1]");
    expect(new MessageFormatError("text", "r").within("").path).toBe("text");
    expect(new MessageFormatError("text", "r").within("parts[1]").reason).toBe("r");
  });
});

describe("construction", () => {
  test("invalid parts raise", () => {
    const makers = [
      () => new MediaPart({ mime: "image/png", data: "a", url: "https://e.com" }),
      () => new ToolCallPart({ id: "c", name: "f", arguments: null }),
      () => new TextPart(bad(3)),
      () => new TextPart("x", bad({ a: new Map() })),
      () => new ToolCallPart({ id: "c", name: "f", arguments: bad([1, 2]) }),
      () => new Message({ role: bad("bot"), parts: [] }),
      () => new Message({ role: "user", parts: bad(["text"]) }),
    ];
    for (const make of makers) expect(make).toThrow(MessageFormatError);
    expect(formatError(() => new ToolCallPart({ id: "c", name: "f", arguments: bad([1, 2]) })).message).toBe(
      "arguments: expected an object, got array",
    );
    expect(formatError(() => new Message({ role: "user", parts: bad(["text"]) })).message).toBe(
      "parts[0]: expected a part, got string",
    );
    expect(formatError(() => new Message({ role: "user", parts: bad("text") })).message).toBe(
      "parts: expected an array, got string",
    );
  });

  test("the payload is copied on construction", () => {
    const payload = { a: [1n, { b: 2n }] } as Data;
    const part = new TextPart("x", payload);
    payload["a"][1]["b"] = 3n;
    payload["c"] = 4n;
    expect(part.providerPayload).toEqual({ a: [1n, { b: 2n }] });
  });

  test("arguments are copied on construction", () => {
    const args = { q: ["x"] };
    const call = new ToolCallPart({ id: "c", name: "f", arguments: args });
    args.q.push("y");
    expect(call.arguments).toEqual({ q: ["x"] });
  });

  test("toJSON output is a copy", () => {
    const message = fullMessage();
    const data = message.toJSON() as Data;
    data["provider_payload"]["meta"]["k"] = "changed";
    data["parts"][0]["provider_payload"]["n"].push(9n);
    data["parts"][5]["arguments"]["deep"]["k"].length = 0;
    expect(message).toEqual(fullMessage());
  });

  test("the parts array is copied and frozen", () => {
    const parts = [new TextPart("a")];
    const message = new Message({ role: "user", parts });
    parts.push(new TextPart("b"));
    expect(message.parts).toEqual([new TextPart("a")]);
    expect(Object.isFrozen(message.parts)).toBe(true);
  });

  test("parts compare by value", () => {
    expect(fullMessage()).toEqual(fullMessage());
    expect(new TextPart("a")).not.toEqual(new TextPart("b"));
  });

  test("a deep payload raises MessageFormatError, not RangeError", () => {
    const deep: Data = {};
    let node = deep;
    for (let i = 0; i < 10_000; i++) {
      const child: Data = {};
      node["k"] = child;
      node = child;
    }
    expect(formatError(() => new TextPart("x", deep)).reason).toBe("value nests deeper than 100 levels");
  });

  test("the nesting limit", () => {
    install({ ...BUILTIN, maxValueNesting: 3 });
    new TextPart("x", { a: { b: { c: 1n } } });
    const error = formatError(() => new TextPart("x", { a: { b: { c: [] } } }));
    expect(error.message).toBe("provider_payload.a.b.c: value nests deeper than 3 levels");
    new ToolCallPart({ id: "c", name: "f", arguments: { a: [[1n]] } });
    expect(formatError(() => new ToolCallPart({ id: "c", name: "f", arguments: { a: [[[1n]]] } })).path).toBe(
      "arguments.a[0][0]",
    );
  });

  test("ints of any size are accepted", () => {
    const big = 10n ** 5000n;
    expect(new TextPart("x", { n: big, m: -big }).providerPayload).toEqual({ n: big, m: -big });
  });

  test("-0 is stored as 0", () => {
    const part = new TextPart("x", { z: -0, a: [-0] });
    expect(Object.is(part.providerPayload!["z"], 0)).toBe(true);
    expect(Object.is((part.providerPayload!["a"] as number[])[0], 0)).toBe(true);
    const call = new ToolCallPart({ id: "c", name: "f", arguments: { z: -0 } });
    expect(Object.is(call.arguments!["z"], 0)).toBe(true);
  });

  test("a __proto__ key stays an own key", () => {
    const payload = JSON.parse('{"__proto__": {"x": 1}, "y": 2}') as Data;
    const part = new TextPart("x", payload);
    const own = (value: unknown) => {
      expect(Object.getPrototypeOf(value)).toBe(Object.prototype);
      expect(Object.hasOwn(value as object, "__proto__")).toBe(true);
      expect(Object.keys(value as object)).toEqual(["__proto__", "y"]);
    };
    own(part.providerPayload);
    const message = new Message({ role: "user", parts: [part] });
    own((message.toJSON() as Data)["parts"][0]["provider_payload"]);
    own((throughCodec(message).parts[0] as TextPart).providerPayload);
    own((Message.fromJSON(message.toJSON()).parts[0] as TextPart).providerPayload);
  });

  test("values that aren't JSON are refused", () => {
    expect(formatError(() => new TextPart("x", bad({ d: new Date(0) }))).message).toBe(
      "provider_payload.d: Date is not a JSON value",
    );
    expect(formatError(() => new TextPart("x", bad({ u: undefined }))).message).toBe(
      "provider_payload.u: undefined is not a JSON value",
    );
    expect(formatError(() => new TextPart("x", bad({ a: [1, , 3] }))).path).toBe("provider_payload.a[1]");
    expect(formatError(() => new TextPart("x", bad({ f: () => 1 }))).message).toBe(
      "provider_payload.f: function is not a JSON value",
    );
    expect(formatError(() => new TextPart("x", bad(new Map()))).message).toBe(
      "provider_payload: expected an object, got Map",
    );
  });

  test("an object with a null prototype is a JSON object", () => {
    const payload = Object.assign(Object.create(null) as Data, { a: 1n });
    expect(new TextPart("x", payload).providerPayload).toEqual({ a: 1n });
  });

  for (const value of ["\ud800", "a\udfffb", "\ude00\ud83d"]) {
    test(`a payload string with an unpaired surrogate (${JSON.stringify(value)})`, () => {
      let error = formatError(() => new TextPart("x", { a: ["ok", value] }));
      expect([error.path, error.reason]).toEqual(["provider_payload.a[1]", "string has an unpaired surrogate"]);
      error = formatError(() => new TextPart("x", { [value]: 1n }));
      expect([error.path, error.reason]).toEqual(["provider_payload", "object key has an unpaired surrogate"]);
    });
  }

  const STRING_FIELDS: [string, (value: string) => unknown, string][] = [
    ["TextPart", (value) => new TextPart(value), "text"],
    ["MediaPart", (value) => new MediaPart({ mime: "image/png", url: value }), "url"],
    ["ToolCallPart", (value) => new ToolCallPart({ id: "c", name: value, arguments: {} }), "name"],
    ["ToolResultPart", (value) => new ToolResultPart({ callId: "c", name: "f", content: value }), "content"],
    ["ReasoningPart", (value) => new ReasoningPart(value), "text"],
    ["RefusalPart", (value) => new RefusalPart(value), "text"],
    ["Message", (value) => new Message({ role: "user", parts: [], protocol: value }), "protocol"],
  ];
  for (const [owner, make, path] of STRING_FIELDS) {
    test(`${owner}.${path} with an unpaired surrogate`, () => {
      const error = formatError(() => make("x\ud800"));
      expect([error.path, error.reason]).toEqual([path, "string has an unpaired surrogate"]);
    });
  }

  test("a paired surrogate is one character", () => {
    expect(new TextPart("\u{1f600}").text).toBe("\u{1f600}");
  });

  test("the path in a large value", () => {
    const rows: Data[] = Array.from({ length: 2000 }, () => ({ a: Array.from({ length: 50 }, (_, i) => i) }));
    const wide = { k: rows };
    new TextPart("x", wide);
    rows[1999]!["a"][49] = new Map();
    expect(formatError(() => new TextPart("x", wide)).path).toBe("provider_payload.k[1999].a[49]");
  });

  test("the first offending value in document order is reported", () => {
    const error = formatError(() => new TextPart("x", bad({ a: [1, new Map()], b: NaN })));
    expect(error.path).toBe("provider_payload.a[1]");
  });

  test("key paths quote odd keys", () => {
    expect(formatError(() => new TextPart("x", bad({ "a b": new Map() }))).path).toBe('provider_payload["a b"]');
  });

  test("checkJsonValue returns a deep copy", () => {
    const value = { a: [{ b: 1n }] };
    const copy = checkJsonValue(value, "v") as Data;
    expect(copy).toEqual(value);
    expect(copy["a"]).not.toBe(value.a);
    expect(copy["a"][0]).not.toBe(value.a[0]);
    expect(formatError(() => checkJsonValue({ a: Symbol("s") }, "v")).message).toBe(
      "v.a: symbol is not a JSON value",
    );
  });

  test("an instance of an anonymous class isn't a plain object", () => {
    const instance = Object.assign(new (class {})(), { a: 1n });
    expect(formatError(() => new TextPart("x", bad(instance))).message).toBe(
      "provider_payload: expected an object, got object",
    );
    expect(formatError(() => new TextPart("x", bad({ v: instance }))).message).toBe(
      "provider_payload.v: object is not a JSON value",
    );
    // A class may even be named `object`; TypeScript forbids declaring one, so set the name.
    const Named = Object.defineProperty(class {}, "name", { value: "object" });
    expect(Named.name).toBe("object");
    expect(formatError(() => new TextPart("x", bad({ v: new Named() }))).path).toBe("provider_payload.v");
  });

  test("a prototype chain with no constructor isn't a plain object", () => {
    const chained = Object.assign(Object.create(Object.create(null)) as Data, { a: 1n });
    expect(formatError(() => new TextPart("x", bad(chained))).path).toBe("provider_payload");
    expect(formatError(() => new TextPart("x", bad({ v: [chained] }))).message).toBe(
      "provider_payload.v[0]: object is not a JSON value",
    );
  });

  test("a hole in the parts array is refused", () => {
    const parts = [new TextPart("a"), , new TextPart("b")];
    expect(formatError(() => new Message({ role: "user", parts: bad(parts) })).message).toBe(
      "parts[1]: expected a part, got undefined",
    );
  });

  test("an overridden iterator on the parts array is not used", () => {
    const hiding = Object.defineProperty(["text"], Symbol.iterator, {
      value: function* () {
        yield new TextPart("ok");
      },
    });
    expect(formatError(() => new Message({ role: "user", parts: bad(hiding) })).path).toBe("parts[0]");
    const parts = Object.defineProperty([new TextPart("a")], Symbol.iterator, {
      value: function* () {
        yield new TextPart("b");
      },
    });
    expect(new Message({ role: "user", parts }).parts).toEqual([new TextPart("a")]);
  });

  test("isPart", () => {
    for (const part of fullMessage().parts) expect(isPart(part)).toBe(true);
    expect(isPart({ type: "text", text: "x" })).toBe(false);
    expect(isPart(null)).toBe(false);
  });
});

describe("helpers", () => {
  test("text() joins the text parts in order with newlines", () => {
    expect(fullMessage().text()).toBe("hello \nworld");
    expect(new Message({ role: "user", parts: [] }).text()).toBe("");
  });

  test("text() leaves out refusal and reasoning", () => {
    const message = new Message({
      role: "assistant",
      parts: [new ReasoningPart("r"), new TextPart("a"), new RefusalPart("no"), new TextPart("b")],
    });
    expect(message.text()).toBe("a\nb");
    expect(new Message({ role: "assistant", parts: [new RefusalPart("no")] }).text()).toBe("");
  });

  test("toolCalls()", () => {
    expect(fullMessage().toolCalls().map((call) => call.id)).toEqual(["call_1", "call_2", "call_3"]);
  });
});
