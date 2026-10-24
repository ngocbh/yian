/**
 * Messages, requests and replies: the stable JSON shape, strict decoding and construction checks, a port of
 * `tests/llm/test_messages.py`; replay records; plus JavaScript's own values: ints of any size, `-0`, a
 * `"__proto__"` key, non-plain objects, `undefined` and array holes; and
 * `conformance/schema/message.schema.json`, which describes the JSON shape.
 */

import { afterEach, describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { BUILTIN, install } from "../../src/defaults.ts";
import {
  checkJsonValue,
  isPart,
  MESSAGE_FORMAT,
  MediaPart,
  Message,
  MessageFormatError,
  NamedTool,
  OutputFormat,
  readReplayRecord,
  ReasoningPart,
  RefusalPart,
  REPLAY_VERSION,
  type ReplayKey,
  replayRecord,
  Request,
  Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  ToolSpec,
  Usage,
} from "../../src/llm/messages.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { FileRef } from "../../src/typesys/values.ts";
import { type Case, decode, readCase, type Value } from "../conformance.ts";
import { lintSchema, loadSchema, type Schema, validate } from "../schema.ts";

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
      new ToolResultPart({
        callId: "call_1",
        name: "search",
        content: "result",
        isError: true,
        providerPayload: { x: 1n },
      }),
      new ToolResultPart({ callId: "call_2", name: "broken", content: "ok" }),
      new RefusalPart("I can't help with that", { id: "msg_2", phase: "final" }),
      new MediaPart({ mime: "image/png", file: new FileRef({ uri: "media/ab.png", mime: "image/png", name: "ab.png" }) }),
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
      { type: "media", mime: "image/png", file: { uri: "media/ab.png", mime: "image/png", name: "ab.png" } },
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

  test("a media part's file form takes a FileRef's fields and round-trips", () => {
    const file = new FileRef({ uri: "media/ab.png", mime: "image/png", name: "ab.png" });
    const part = new MediaPart({ mime: "image/png", file, name: "chart.png" });
    const fields = { uri: "media/ab.png", mime: "image/png", name: "ab.png" };
    expect([part.file, part.data, part.url]).toEqual([fields, null, null]);
    expect(Object.isFrozen(part.file)).toBe(true);
    const message = new Message({ role: "assistant", parts: [part] });
    expect(message.toJSON()["parts"]).toEqual([
      { type: "media", mime: "image/png", file: { uri: "media/ab.png", mime: "image/png", name: "ab.png" }, name: "chart.png" },
    ]);
    for (const read of [Message.fromJSON(message.toJSON()), throughCodec(message)]) {
      expect((read.parts[0] as MediaPart).file).toEqual(fields);
      expect(read.toJSON()).toEqual(message.toJSON());
    }
    // A file's unset fields are left out, as the message's own are.
    const bare = new MediaPart({ mime: "image/png", file: new FileRef({ uri: "media/a.png" }) });
    expect(bare.toJSON()).toEqual({ type: "media", mime: "image/png", file: { uri: "media/a.png" } });
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

describe("author", () => {
  const authored = () => new Message({ role: "user", parts: [new TextPart("hi")], author: "alice" });

  test("round-trips, written after parts and before protocol", () => {
    const data = authored().toJSON();
    expect(Object.keys(data)).toEqual(["format", "role", "parts", "author"]);
    expect(Message.fromJSON(data)).toEqual(authored());
    expect(throughCodec(authored())).toEqual(authored());
    const both = new Message({ role: "user", parts: [], author: "alice", protocol: "p" });
    expect(Object.keys(both.toJSON())).toEqual(["format", "role", "parts", "author", "protocol"]);
  });

  test("is null and left out when not given", () => {
    const message = new Message({ role: "user", parts: [new TextPart("hi")] });
    expect(message.author).toBeNull();
    expect("author" in message.toJSON()).toBe(false);
  });

  test("any role may carry it", () => {
    for (const role of ["system", "user", "assistant", "tool"] as const) {
      const parts = role === "tool" ? [new ToolResultPart({ callId: "c", name: "f", content: "r" })] : [new TextPart("x")];
      expect(new Message({ role, parts, author: "a" }).author).toBe("a");
    }
  });

  test.each([
    ["an empty author", "", "author: expected a non-empty string"],
    ["a non-string author", 3, "author: expected string, got number"],
    ["an unpaired surrogate", "\ud800", "author: string has an unpaired surrogate"],
  ])("refuses %s", (_, value, message) => {
    expect(formatError(() => new Message({ role: "user", parts: [], author: bad(value) })).message).toBe(message);
  });

  test("strict decode refuses an explicit null or an empty author", () => {
    const data = authored().toJSON() as Data;
    expect(errorPath({ ...data, author: null })).toBe("author");
    expect(errorPath({ ...data, author: "" })).toBe("author");
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
      "parts[2]: media needs exactly one of data, url or file",
    );
  });

  test("media with url and file", () => {
    const data = canonical();
    data["parts"][9]["url"] = "https://e.com/x.png";
    expect(formatError(() => Message.fromJSON(data)).message).toBe(
      "parts[9]: media needs exactly one of data, url or file",
    );
  });

  test.each([
    ["a file that isn't an object", (file: Data) => file, "media/x.png", "parts[9].file: expected an object, got string"],
    ["a file without uri", (file: Data) => (delete file["uri"], file), null, "parts[9].file.uri: missing"],
    ["an unknown file key", (file: Data) => ({ ...file, size: 3n }), null, "parts[9].file.size: unknown key"],
    ["a non-string uri", (file: Data) => ({ ...file, uri: 3n }), null, "parts[9].file.uri: expected string, got bigint"],
    ["an explicit null file mime", (file: Data) => ({ ...file, mime: null }), null, "parts[9].file.mime: expected string, got null"],
    ["an explicit null file name", (file: Data) => ({ ...file, name: null }), null, "parts[9].file.name: expected string, got null"],
  ])("media with %s", (_, edit, replacement, message) => {
    const data = canonical();
    data["parts"][9]["file"] = replacement ?? edit(data["parts"][9]["file"]);
    expect(formatError(() => Message.fromJSON(data)).message).toBe(message);
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
      [(data) => (data["author"] = null), "author", "expected string, got null"],
      [
        (data) => (data["parts"][0]["provider_payload"] = null),
        "parts[0].provider_payload",
        "expected an object, got null",
      ],
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
      "expected an object, got non-plain object",
    );
    const data = canonical();
    data["parts"][1] = Object.assign(Object.create(Object.create(null)) as Data, data["parts"][1]);
    expect(errorPath(data)).toBe("parts[1]");
  });
});

describe("MessageFormatError", () => {
  test("the message is the path and reason, or the reason alone", () => {
    expect(new MessageFormatError("parts[1].text", "missing").message).toBe("parts[1].text: missing");
    expect(new MessageFormatError("", "media needs exactly one of data, url or file").message).toBe(
      "media needs exactly one of data, url or file",
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
      () => new MediaPart({ mime: "image/png", data: "a", file: new FileRef({ uri: "media/a.png" }) }),
      () => new MediaPart({ mime: "image/png", file: bad("media/a.png") }),
      () => new MediaPart({ mime: "image/png", file: bad({ uri: 3 }) }),
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
      "provider_payload: expected an object, got non-plain object",
    );
    expect(formatError(() => new TextPart("x", bad({ v: instance }))).message).toBe(
      "provider_payload.v: non-plain object is not a JSON value",
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
      "provider_payload.v[0]: non-plain object is not a JSON value",
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

describe("tool specs", () => {
  test("NaN in parameters is refused", () => {
    expect(formatError(() => new ToolSpec({ name: "f", description: "d", parameters: { a: NaN } })).message).toBe(
      "parameters.a: float is not finite",
    );
  });

  test("parameters must be an object", () => {
    expect(formatError(() => new ToolSpec({ name: "f", description: "d", parameters: bad(null) })).message).toBe(
      "parameters: expected an object, got null",
    );
    expect(formatError(() => new ToolSpec({ name: "f", description: "d", parameters: bad([]) })).message).toBe(
      "parameters: expected an object, got array",
    );
  });

  test("parameters are copied", () => {
    const parameters = { type: "object", properties: {} as Data };
    const spec = new ToolSpec({ name: "f", description: "d", parameters });
    parameters.properties["x"] = {};
    expect(spec.parameters).toEqual({ type: "object", properties: {} });
  });

  test("the nesting limit applies to parameters", () => {
    install({ ...BUILTIN, maxValueNesting: 3 });
    new ToolSpec({ name: "f", description: "d", parameters: { a: [{}] } });
    expect(formatError(() => new ToolSpec({ name: "f", description: "d", parameters: { a: [[{}]] } })).path).toBe(
      "parameters.a[0][0]",
    );
  });

  test("a description with an unpaired surrogate", () => {
    const error = formatError(() => new ToolSpec({ name: "f", description: "x\ud800", parameters: {} }));
    expect([error.path, error.reason]).toEqual(["description", "string has an unpaired surrogate"]);
  });

  test("strict is a boolean, null when absent", () => {
    expect(new ToolSpec({ name: "f", description: "d", parameters: {} }).strict).toBeNull();
    expect(new ToolSpec({ name: "f", description: "d", parameters: {}, strict: true }).strict).toBe(true);
    const error = formatError(() => new ToolSpec({ name: "f", description: "d", parameters: {}, strict: bad(1) }));
    expect(error.message).toBe("strict: expected boolean, got number");
  });
});

describe("OutputFormat", () => {
  test("fields are kept, strict null when absent", () => {
    const format = new OutputFormat({ name: "Rec", schema: { type: "object" }, strict: true });
    expect([format.name, format.schema, format.strict]).toEqual(["Rec", { type: "object" }, true]);
    expect(new OutputFormat({ name: "Rec", schema: {} }).strict).toBeNull();
  });

  test("the schema is copied", () => {
    const schema = { type: "object", properties: {} as Data };
    const format = new OutputFormat({ name: "Rec", schema });
    schema.properties["x"] = {};
    expect(format.schema).toEqual({ type: "object", properties: {} });
  });

  test("a non-string name, non-object schema or non-boolean strict is refused", () => {
    expect(formatError(() => new OutputFormat({ name: bad(1), schema: {} })).message).toBe(
      "name: expected string, got number",
    );
    expect(formatError(() => new OutputFormat({ name: "Rec", schema: bad([]) })).message).toBe(
      "schema: expected an object, got array",
    );
    expect(formatError(() => new OutputFormat({ name: "Rec", schema: bad(null) })).message).toBe(
      "schema: expected an object, got null",
    );
    expect(formatError(() => new OutputFormat({ name: "Rec", schema: {}, strict: bad("yes") })).message).toBe(
      "strict: expected boolean, got string",
    );
  });
});

describe("requests and replies", () => {
  const hello = () => new Message({ role: "user", parts: [new TextPart("hi")] });

  test("request defaults", () => {
    const message = hello();
    const request = new Request({ messages: [message] });
    expect(request.messages).toEqual([message]);
    expect(request.tools).toEqual([]);
    expect(request.toolChoice).toBeNull();
    expect(request.parallelToolCalls).toBeNull();
    expect(request.outputFormat).toBeNull();
  });

  test("a request with tools and a named tool", () => {
    const spec = new ToolSpec({ name: "f", description: "does f", parameters: { type: "object" }, strict: true });
    const request = new Request({ messages: [], tools: [spec], toolChoice: new NamedTool("f") });
    expect(request.tools).toEqual([spec]);
    expect(request.toolChoice).toEqual(new NamedTool("f"));
  });

  test("a request's arrays are copied and frozen", () => {
    const messages = [hello()];
    const request = new Request({ messages });
    messages.push(hello());
    expect(request.messages).toHaveLength(1);
    expect(Object.isFrozen(request.messages)).toBe(true);
    expect(Object.isFrozen(request.tools)).toBe(true);
  });

  test("a request refuses a non-message", () => {
    expect(formatError(() => new Request({ messages: bad([hello(), { role: "user" }]) })).message).toBe(
      "messages[1]: expected a Message, got object",
    );
    expect(formatError(() => new Request({ messages: bad(hello()) })).message).toBe(
      "messages: expected an array, got Message",
    );
    expect(formatError(() => new Request({ messages: bad([hello(), , hello()]) })).path).toBe("messages[1]");
  });

  test("a request refuses a non-tool-spec", () => {
    expect(formatError(() => new Request({ messages: [], tools: bad([{ name: "f" }]) })).message).toBe(
      "tools[0]: expected a ToolSpec, got object",
    );
    expect(formatError(() => new Request({ messages: [], tools: bad("f") })).message).toBe(
      "tools: expected an array, got string",
    );
  });

  for (const choice of ["auto", "required", "none", new NamedTool("f"), null] as const) {
    test(`a request accepts the tool choice ${JSON.stringify(choice)}`, () => {
      expect(new Request({ messages: [], toolChoice: choice }).toolChoice).toEqual(choice);
    });
  }

  for (const choice of ["any", "Auto", { name: "f" }, 1, true]) {
    test(`a request refuses the tool choice ${JSON.stringify(choice)}`, () => {
      expect(formatError(() => new Request({ messages: [], toolChoice: bad(choice) })).message).toBe(
        "tool_choice: expected auto, required, none, a NamedTool or null",
      );
    });
  }

  test("parallelToolCalls", () => {
    expect(new Request({ messages: [], parallelToolCalls: true }).parallelToolCalls).toBe(true);
    expect(new Request({ messages: [], parallelToolCalls: false }).parallelToolCalls).toBe(false);
    expect(new Request({ messages: [] }).parallelToolCalls).toBeNull();
    expect(formatError(() => new Request({ messages: [], parallelToolCalls: bad("yes") })).message).toBe(
      "parallel_tool_calls: expected boolean, got string",
    );
  });

  test("outputFormat", () => {
    const format = new OutputFormat({ name: "Rec", schema: { type: "object" } });
    expect(new Request({ messages: [] }).outputFormat).toBeNull();
    expect(new Request({ messages: [], outputFormat: null }).outputFormat).toBeNull();
    expect(new Request({ messages: [], outputFormat: format }).outputFormat).toBe(format);
    expect(
      formatError(() => new Request({ messages: [], outputFormat: bad({ name: "Rec", schema: {} }) })).message,
    ).toBe("output_format: expected an OutputFormat or null");
  });

  test("a response's message must be an assistant message", () => {
    expect(formatError(() => new Response({ message: hello() })).message).toBe("message.role: expected assistant");
    expect(formatError(() => new Response({ message: bad({ role: "assistant" }) })).message).toBe(
      "message: expected a Message, got object",
    );
  });

  test("a named tool's name is a string", () => {
    expect(formatError(() => new NamedTool(bad(1))).message).toBe("name: expected string, got number");
  });

  test("a response checks its usage, finish reason and response id", () => {
    const message = new Message({ role: "assistant", parts: [] });
    expect(formatError(() => new Response({ message, usage: bad({}) })).message).toBe(
      "usage: expected a Usage, got object",
    );
    expect(formatError(() => new Response({ message, finishReason: bad(1) })).message).toBe(
      "finish_reason: expected string, got number",
    );
    expect(formatError(() => new Response({ message, responseId: "x\ud800" })).message).toBe(
      "response_id: string has an unpaired surrogate",
    );
  });

  test("response and usage", () => {
    const usage = new Usage({ inputTokens: 1, outputTokens: 2, totalTokens: 3, cacheReadTokens: 1 });
    const response = new Response({
      message: new Message({ role: "assistant", parts: [] }),
      usage,
      finishReason: "stop",
    });
    expect(response.usage).toBe(usage);
    expect(response.usage!.reasoningTokens).toBeNull();
    expect(response.usage!.cacheCreationTokens).toBeNull();
    expect(response.finishReason).toBe("stop");
    expect(response.responseId).toBeNull();
    expect(new Response({ message: new Message({ role: "assistant", parts: [] }) }).usage).toBeNull();
  });

  const USAGE_CASES: [Data, string, string][] = [
    [{ inputTokens: 1.5 }, "input_tokens", "number"],
    [{ outputTokens: true }, "output_tokens", "boolean"],
    [{ totalTokens: null }, "total_tokens", "null"],
    [{ totalTokens: "3" }, "total_tokens", "string"],
    [{ cacheReadTokens: 1n }, "cache_read_tokens", "bigint"],
    [{ cacheCreationTokens: false }, "cache_creation_tokens", "boolean"],
    [{ cacheCreationTokens: 1.5 }, "cache_creation_tokens", "number"],
    [{ reasoningTokens: "0" }, "reasoning_tokens", "string"],
    [{ inputTokens: 2 ** 53 }, "input_tokens", "number"],
  ];
  for (const [fields, path, type] of USAGE_CASES) {
    test(`Usage refuses a non-integer ${path} (${type})`, () => {
      const error = formatError(() => new Usage({ inputTokens: 1, outputTokens: 2, totalTokens: 3, ...fields }));
      expect([error.path, error.reason]).toEqual([path, `expected integer, got ${type}`]);
    });
  }

  test("Usage allows absent or null optional counts", () => {
    const usage = new Usage({
      inputTokens: 1.0,
      outputTokens: 0,
      totalTokens: 0,
      cacheCreationTokens: 4,
      reasoningTokens: null,
    });
    expect(usage.inputTokens).toBe(1);
    expect(usage.cacheCreationTokens).toBe(4);
    expect(usage.reasoningTokens).toBeNull();
    expect(usage.cacheReadTokens).toBeNull();
  });
});

describe("replay records", () => {
  const KEY: ReplayKey = { protocol: "openai_responses", provider: "openai", model: "gpt-5" };
  const FIELDS = ["encrypted_content", "item_id"];

  test("the built record without an account", () => {
    expect(REPLAY_VERSION).toBe(1);
    const record = replayRecord(KEY, { encrypted_content: "e", item_id: "i" });
    expect(record).toEqual({
      kind: "openai_responses",
      v: 1n,
      provider: "openai",
      model: "gpt-5",
      encrypted_content: "e",
      item_id: "i",
    });
    expect(Object.keys(record)).toEqual(["kind", "v", "provider", "model", "encrypted_content", "item_id"]);
    expect(Object.keys(replayRecord({ ...KEY, account: null }, {}))).toEqual(["kind", "v", "provider", "model"]);
  });

  test("the built record with an account", () => {
    const record = replayRecord({ ...KEY, account: "acct_1" }, { item_id: "i" });
    expect(Object.keys(record)).toEqual(["kind", "v", "provider", "model", "account", "item_id"]);
    expect(record["account"]).toBe("acct_1");
  });

  test("the fields are checked and copied", () => {
    const fields = { item_id: "i", list: [1n] };
    const record = replayRecord(KEY, fields);
    fields.list.push(2n);
    expect(record["list"]).toEqual([1n]);
    expect(formatError(() => replayRecord(KEY, { a: [NaN] })).message).toBe("a[0]: float is not finite");
  });

  for (const name of ["kind", "v", "provider", "model", "account"]) {
    test(`a field named ${name} is refused`, () => {
      const error = formatError(() => replayRecord(KEY, { item_id: "i", [name]: "x" }));
      expect([error.path, error.reason]).toEqual([name, "reserved replay record key"]);
    });
  }

  /** Return a valid record for `KEY`, edited by `edit`. */
  function edited(edit: (record: Data) => void): Data {
    const record = replayRecord(KEY, { item_id: "i" }) as Data;
    edit(record);
    return record;
  }

  test("missing", () => {
    expect(readReplayRecord(null, KEY, FIELDS)).toEqual({ ok: false, reason: "missing" });
  });

  test("malformed", () => {
    const cases: ((record: Data) => void)[] = [
      (record) => (record["kind"] = 1n),
      (record) => delete record["kind"],
      (record) => (record["provider"] = null),
      (record) => (record["model"] = ["gpt-5"]),
      (record) => delete record["v"],
      (record) => (record["account"] = 1n),
      (record) => (record["account"] = null),
    ];
    for (const edit of cases) {
      expect(readReplayRecord(edited(edit), KEY, FIELDS)).toEqual({ ok: false, reason: "malformed" });
    }
  });

  test("unknown_version", () => {
    for (const v of [2n, 0, "1", null, 1.5]) {
      // The version is checked before the protocol.
      const record = edited((record) => {
        record["v"] = v;
        record["kind"] = "anthropic_messages";
      });
      expect(readReplayRecord(record, KEY, FIELDS)).toEqual({ ok: false, reason: "unknown_version" });
    }
  });

  test("other_protocol", () => {
    // A record with several mismatches reports the first.
    const record = edited((record) => {
      record["kind"] = "anthropic_messages";
      record["model"] = "other";
    });
    expect(readReplayRecord(record, KEY, FIELDS)).toEqual({ ok: false, reason: "other_protocol" });
  });

  test("other_provider", () => {
    const record = edited((record) => {
      record["provider"] = "azure";
      record["model"] = "other";
    });
    expect(readReplayRecord(record, KEY, FIELDS)).toEqual({ ok: false, reason: "other_provider" });
  });

  test("other_model", () => {
    const record = edited((record) => {
      record["model"] = "gpt-4";
      record["account"] = "acct_1";
    });
    expect(readReplayRecord(record, KEY, FIELDS)).toEqual({ ok: false, reason: "other_model" });
  });

  test("matchModel false reads another model's record, still checking the account", () => {
    const other = replayRecord({ ...KEY, model: "gpt-4" }, { item_id: "i" });
    expect(readReplayRecord(other, KEY, FIELDS, { matchModel: false })).toEqual({ ok: true, fields: { item_id: "i" } });
    expect(readReplayRecord(other, KEY, FIELDS)).toEqual({ ok: false, reason: "other_model" });
    const otherAccount = replayRecord({ ...KEY, model: "gpt-4", account: "acct_1" }, { item_id: "i" });
    expect(readReplayRecord(otherAccount, KEY, FIELDS, { matchModel: false })).toEqual({
      ok: false,
      reason: "other_account",
    });
  });

  test("other_account", () => {
    const withAccount = replayRecord({ ...KEY, account: "acct_1" }, { item_id: "i" });
    expect(readReplayRecord(withAccount, KEY, FIELDS)).toEqual({ ok: false, reason: "other_account" });
    // The account is checked before the fields.
    const withExtra = replayRecord({ ...KEY, account: "acct_1" }, { item_id: "i", extra: 1n });
    expect(readReplayRecord(withExtra, KEY, FIELDS)).toEqual({ ok: false, reason: "other_account" });
    expect(readReplayRecord(withAccount, { ...KEY, account: "acct_2" }, FIELDS)).toEqual({
      ok: false,
      reason: "other_account",
    });
    const without = replayRecord(KEY, { item_id: "i" });
    expect(readReplayRecord(without, { ...KEY, account: "acct_1" }, FIELDS)).toEqual({
      ok: false,
      reason: "other_account",
    });
    expect(readReplayRecord(without, { ...KEY, account: null }, FIELDS).ok).toBe(true);
    expect(readReplayRecord(withAccount, { ...KEY, account: "acct_1" }, FIELDS).ok).toBe(true);
  });

  test("unknown_field", () => {
    const record = edited((record) => (record["extra"] = 1n));
    expect(readReplayRecord(record, KEY, FIELDS)).toEqual({ ok: false, reason: "unknown_field" });
  });

  test("a valid record's fields come back without the envelope, as a copy", () => {
    const record = replayRecord({ ...KEY, account: "acct_1" }, { item_id: "i", encrypted_content: "e" }) as Data;
    const read = readReplayRecord(record, { ...KEY, account: "acct_1" }, FIELDS);
    expect(read).toEqual({ ok: true, fields: { item_id: "i", encrypted_content: "e" } });
    const nested = replayRecord(KEY, { item_id: { deep: ["x"] } });
    const back = readReplayRecord(nested, KEY, FIELDS);
    if (!back.ok) throw new Error("expected a valid record");
    (back.fields["item_id"] as Data)["deep"].push("y");
    expect(nested["item_id"]).toEqual({ deep: ["x"] });
  });

  test("a number v of 1 is accepted", () => {
    const record = edited((record) => (record["v"] = 1));
    expect(readReplayRecord(record, KEY, FIELDS)).toEqual({ ok: true, fields: { item_id: "i" } });
  });

  test("a malformed record in a payload doesn't stop a message", () => {
    const payload = { kind: 1n, v: "two" };
    const message = new Message({ role: "assistant", parts: [new ReasoningPart("r", payload)] });
    const back = Message.fromJSON(parseJson(logJson(message.toJSON()), "codec"));
    expect(back).toEqual(message);
    const stored = (back.parts[0] as ReasoningPart).providerPayload;
    expect(readReplayRecord(stored, KEY, FIELDS)).toEqual({ ok: false, reason: "malformed" });
  });
});

describe("message schema", () => {
  const SCHEMA = loadSchema("message.schema.json");
  const RECORD = (SCHEMA.$defs as Record<string, Schema>).replay_record!;
  const KEY: ReplayKey = { protocol: "openai_responses", provider: "openai", model: "gpt-5" };

  test("uses only the keywords the validator knows, and every $ref resolves", () => {
    expect(lintSchema(SCHEMA, SCHEMA)).toEqual([]);
  });

  test("declares draft 2020-12 and yian.message.v1", () => {
    expect(SCHEMA.$schema).toBe("https://json-schema.org/draft/2020-12/schema");
    expect(SCHEMA.title).toBe(MESSAGE_FORMAT);
  });

  test("the messages toJSON writes validate", () => {
    expect(validate(SCHEMA, SCHEMA, fullMessage().toJSON())).toEqual([]);
    expect(validate(SCHEMA, SCHEMA, canonical())).toEqual([]);
    for (const role of ["system", "user", "assistant", "tool"] as const) {
      const message = new Message({ role, parts: [new TextPart("x")] });
      expect(validate(SCHEMA, SCHEMA, message.toJSON())).toEqual([]);
    }
  });

  test.each([
    ["an unknown top-level key", (data: Data) => (data["extra"] = 1)],
    ["an unknown part key", (data: Data) => (data["parts"][1]["bold"] = true)],
    ["an unknown part type", (data: Data) => (data["parts"][2]["type"] = "video")],
    ["a missing format", (data: Data) => delete data["format"]],
    ["another format", (data: Data) => (data["format"] = "yian.message.v0")],
    ["media with data and url", (data: Data) => (data["parts"][2]["data"] = "AAAA")],
    ["media with neither", (data: Data) => delete data["parts"][2]["url"]],
    ["media with data and file", (data: Data) => (data["parts"][9]["data"] = "AAAA")],
    ["a file without uri", (data: Data) => delete data["parts"][9]["file"]["uri"]],
    ["an unknown file key", (data: Data) => (data["parts"][9]["file"]["size"] = 3n)],
    ["an explicit null file mime", (data: Data) => (data["parts"][9]["file"]["mime"] = null)],
    ["a tool call with neither", (data: Data) => delete data["parts"][5]["raw_arguments"]],
    ["a wrong field type", (data: Data) => (data["parts"][1]["text"] = 3n)],
    ["is_error not a boolean", (data: Data) => (data["parts"][6]["is_error"] = "yes")],
    ["a missing part field", (data: Data) => delete data["parts"][4]["id"]],
    ["an explicit null protocol", (data: Data) => (data["protocol"] = null)],
    ["an explicit null provider_payload", (data: Data) => (data["provider_payload"] = null)],
    ["an explicit null part payload", (data: Data) => (data["parts"][0]["provider_payload"] = null)],
    ["an explicit null data", (data: Data) => (data["parts"][2]["data"] = null)],
    ["an explicit null url", (data: Data) => (data["parts"][3]["url"] = null)],
    ["an explicit null name", (data: Data) => (data["parts"][2]["name"] = null)],
    ["an explicit null raw_arguments", (data: Data) => (data["parts"][4]["raw_arguments"] = null)],
    ["an explicit null arguments", (data: Data) => (data["parts"][5]["arguments"] = null)],
    ["an explicit null is_error", (data: Data) => (data["parts"][6]["is_error"] = null)],
  ])("rejects %s, as decoding does", (_, edit) => {
    const data = canonical();
    edit(data);
    expect(() => Message.fromJSON(data)).toThrow(MessageFormatError);
    expect(validate(SCHEMA, SCHEMA, data)).not.toEqual([]);
  });

  test("additionalProperties, not the validator, refuses an unlisted key", () => {
    const part = { type: "text", text: "x", extra: 1 };
    const text = { $ref: "#/$defs/text" };
    expect(validate(SCHEMA, text, part)).toEqual(["$: undeclared key extra"]);
    const open = structuredClone(SCHEMA);
    delete (open.$defs as Record<string, Schema>).text!.additionalProperties;
    expect(validate(open, text, part)).toEqual([]);
  });

  test("a replay record validates as $defs/replay_record", () => {
    expect(validate(SCHEMA, RECORD, replayRecord(KEY, { item_id: "i" }))).toEqual([]);
    expect(validate(SCHEMA, RECORD, replayRecord({ ...KEY, account: "acct_1" }, {}))).toEqual([]);
    expect(validate(SCHEMA, RECORD, { ...replayRecord(KEY, {}), v: "1" })).toEqual([
      "$.v: string is not integer",
    ]);
    expect(validate(SCHEMA, RECORD, { ...replayRecord(KEY, {}), v: 0n })).toEqual(["$.v: 0 is below 1"]);
  });

  test("every scripted reply and request message in the conformance cases validates", () => {
    const root = fileURLToPath(new URL("../../../conformance/cases", import.meta.url));
    const files = { reply: new Set<string>(), messages: new Set<string>() };
    const counts = { reply: 0, messages: 0 };
    const check = (kind: "reply" | "messages", file: string, value: Value) => {
      const data = Message.fromJSON(value).toJSON();
      expect([file, validate(SCHEMA, SCHEMA, data)]).toEqual([file, []]);
      files[kind].add(file);
      counts[kind] += 1;
    };
    for (const file of readdirSync(root, { recursive: true, encoding: "utf8" }).sort()) {
      if (!file.endsWith(".yaml")) continue;
      const host = readCase(readFileSync(join(root, file), "utf8"))["host"] as Case | undefined;
      if (host?.["models"] === undefined) continue;
      const models = decode(host["models"]) as Record<string, Record<string, Case[]>>;
      for (const consumers of Object.values(models)) {
        for (const entries of Object.values(consumers)) {
          for (const entry of entries) {
            if (entry["reply"] !== undefined) check("reply", file, entry["reply"]);
            const request = entry["request"] as Case | undefined;
            for (const message of (request?.["messages"] as Value[] | undefined) ?? []) {
              check("messages", file, message);
            }
          }
        }
      }
    }
    // Minimums, so new cases don't break this, while proving the cases were read.
    expect(files.reply.size).toBeGreaterThanOrEqual(35);
    expect(counts.reply).toBeGreaterThanOrEqual(70);
    expect(files.messages.size).toBeGreaterThanOrEqual(4);
    expect(counts.messages).toBeGreaterThanOrEqual(10);
  });
});
