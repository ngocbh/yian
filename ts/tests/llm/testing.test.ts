/**
 * Test helpers, a port of `tests/llm/test_testing.py`: cassette replay through `RouteModel` and
 * `safeFetch`, `findSecrets`, and the scripted model, plus an aborted signal ending a stream.
 */

import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { safeFetch } from "../../src/platform.ts";
import { BearerAuth } from "../../src/llm/auth.ts";
import { collect, type Event, EventStream, Finish, PartDone, ReasoningDelta, TextDelta, ToolCallDelta, UsageReport } from "../../src/llm/events.ts";
import { RouteModel, type SseProtocol } from "../../src/llm/executor.ts";
import { Message, ReasoningPart, RefusalPart, Request, Response, TextPart, ToolCallPart, Usage } from "../../src/llm/messages.ts";
import { Route } from "../../src/llm/route.ts";
import {
  type Cassette,
  CASSETTE_REDACTED,
  CassetteReplay,
  type CassetteRequest,
  type CassetteResponse,
  findSecrets,
  loadCassette,
  ScriptedModel,
} from "../../src/llm/testing.ts";

const CASSETTE = new URL("./cassettes/example_sse.json", import.meta.url).pathname;
// A fake key shaped like a real one, so findSecrets sees it.
const KEY = "sk-test-0123456789abcdefghijKLMN";
const BASE = "https://api.example.test/v1";

/** Return a cassette holding one interaction, answering `{"ok": true}` by default. */
function oneInteraction(request: CassetteRequest, response: CassetteResponse = { status: 200, json: { ok: true } }): Cassette {
  return { format: "yian.cassette.v1", source: "synthetic", description: "test", interactions: [{ request, response }] };
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

/** A protocol whose body asks the first message's text and whose events are each frame's data. */
const EXAMPLE_PROTOCOL: SseProtocol = {
  name: "test",
  model: "m",
  framing: "sse",
  buildBody: (request) => ({ model: "m", input: [{ role: "user", content: (request.messages[0]!.parts[0] as TextPart).text }] }),
  async *parse(frames) {
    for await (const frame of frames) yield new TextDelta(0, frame.data);
    yield new Finish({ reason: "stop" });
  },
};

/** A model over the example route, sending through `replay`. */
function exampleModel(replay: CassetteReplay): RouteModel {
  const route = new Route({ provider: "example", baseUrl: BASE, path: "responses", auth: new BearerAuth(KEY), query: { "api-version": "2026-01-01" } });
  return new RouteModel({ route, protocol: EXAMPLE_PROTOCOL, send: replay.send });
}

/** Return a request of one user message with this text. */
function ask(text: string): Request {
  return new Request({ messages: [new Message({ role: "user", parts: [new TextPart(text)] })] });
}

/** Send one request through `replay` under `safeFetch`, with `json` as the body when given. */
function send(
  replay: CassetteReplay,
  url: string,
  options: { method?: string; headers?: Record<string, string>; json?: unknown } = {},
): Promise<globalThis.Response> {
  const body = options.json === undefined ? undefined : JSON.stringify(options.json);
  return safeFetch(url, { method: options.method ?? "GET", headers: options.headers, body }, replay.send);
}

/** Return what `promise` rejects with; fail the test if it resolves. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

/** Return the message of the error a request through a replay of `recorded` throws. */
async function mismatch(recorded: CassetteRequest, url: string, options: Parameters<typeof send>[2] = {}): Promise<string> {
  const error = await rejection(send(new CassetteReplay(oneInteraction(recorded)), url, options));
  expect(error).toBeInstanceOf(Error);
  return (error as Error).message;
}

/** Return a temporary file path holding `text`. */
function tempFile(text: string): string {
  const path = join(mkdtempSync(join(tmpdir(), "yian-cassette-")), "cassette.json");
  writeFileSync(path, text);
  return path;
}

// Replay

describe("CassetteReplay", () => {
  test("replays interactions in order through RouteModel", async () => {
    const replay = new CassetteReplay(CASSETTE);
    const model = exampleModel(replay);
    const replies: string[][] = [];
    for (const text of ["first", "second"]) {
      const events = await drain(model.stream(ask(text)));
      replies.push(events.filter((event) => event instanceof TextDelta).map((event) => event.text));
    }

    expect(replies).toEqual([['{"text": "one"}', "[DONE]"], ['{"text": "two"}', "[DONE]"]]);
    const sent = replay.requests.map((request) => JSON.parse(request.body!).input[0].content);
    expect(sent).toEqual(["first", "second"]);
    expect(replay.requests[0]!.method).toBe("POST");
    expect(replay.requests[0]!.url).toBe(`${BASE}/responses?api-version=2026-01-01`);
    expect(replay.requests[0]!.headers.get("authorization")).toBe(`Bearer ${KEY}`);
    replay.assertDone();
  });

  test("assertDone fails naming the unused count", async () => {
    const replay = new CassetteReplay(CASSETTE);
    const stream = exampleModel(replay).stream(ask("first"));
    await stream.next();
    await stream.close();

    expect(() => replay.assertDone()).toThrow("1 of 2 interaction(s) were not used");
  });

  test("an extra request fails", async () => {
    const replay = new CassetteReplay(oneInteraction({ method: "GET", url: "https://h.test/a" }));
    await send(replay, "https://h.test/a");

    const error = await rejection(send(replay, "https://h.test/a"));
    expect((error as Error).message).toBe("unexpected request 2: the cassette has 1 interaction(s)");
    expect(replay.requests).toHaveLength(2);
  });

  test("a path loads the file, and an object is checked", () => {
    expect(loadCassette(CASSETTE).source).toBe("synthetic");
    expect(new CassetteReplay(CASSETTE).requests).toEqual([]);
    expect(() => new CassetteReplay({ format: "other" } as unknown as Cassette)).toThrow("format");
  });

  const request = { method: "GET", url: "https://h.test/" };
  const bad: [string, unknown, string][] = [
    ["a bad format", { format: "other" }, "cassette: format must be 'yian.cassette.v1'"],
    ["a missing key", { format: "yian.cassette.v1", source: "synthetic", description: "d" }, "cassette: missing 'interactions'"],
    [
      "unknown keys",
      { ...oneInteraction(request), zeta: 1, alpha: 2 },
      "cassette: unknown keys ['alpha', 'zeta']",
    ],
    ["a bad source", { ...oneInteraction(request), source: "made-up" }, "cassette: source must be 'synthetic' or 'recorded'"],
    ["no body", oneInteraction(request, { status: 200 }), "interactions[0].response: needs exactly one of chunks, json, text"],
    [
      "two bodies",
      oneInteraction(request, { status: 200, json: {}, text: "x" }),
      "interactions[0].response: needs exactly one of chunks, json, text",
    ],
    ["a string status", oneInteraction(request, { status: "200" as never, text: "" }), "interactions[0].response: status must be an int"],
    ["a float status", oneInteraction(request, { status: 200.5, text: "" }), "interactions[0].response: status must be an int"],
    [
      "chunks not strings",
      oneInteraction(request, { status: 200, chunks: ["a", 1 as never] }),
      "interactions[0].response: chunks must be a list of strings",
    ],
    ["a non-string text", oneInteraction(request, { status: 200, text: 1 as never }), "interactions[0].response: text must be a string"],
    [
      "headers not strings",
      oneInteraction({ ...request, headers: { a: 1 as never } }),
      "interactions[0].request.headers: expected an object of strings",
    ],
    ["a non-string url", oneInteraction({ method: "GET", url: 1 as never }), "interactions[0].request: method and url must be strings"],
    ["an interaction not an object", { ...oneInteraction(request), interactions: [[]] }, "interactions[0]: expected an object, got array"],
  ];
  for (const [name, data, message] of bad) {
    test(`a load error names the part: ${name}`, () => {
      const error = thrown(() => loadCassette(tempFile(JSON.stringify(data))));
      expect(error).toBeInstanceOf(TypeError);
      expect(error.message).toBe(message);
    });
  }

  test("a file that isn't JSON is named", () => {
    const path = tempFile("{not json");
    const error = thrown(() => loadCassette(path));
    expect(error.message).toStartWith(`cassette ${path} is not JSON: Expecting property name enclosed in double quotes`);
  });

  test("a file's int status reads as a bigint and replays", async () => {
    const path = tempFile(JSON.stringify(oneInteraction({ method: "GET", url: "https://h.test/" }, { status: 201, text: "made" })));
    expect(loadCassette(path).interactions[0]!.response.status).toBe(201n);
    const reply = await send(new CassetteReplay(path), "https://h.test/");
    expect(reply.status).toBe(201);
    expect(await reply.text()).toBe("made");
  });

  const responses: [CassetteResponse, string][] = [
    [{ status: 201, json: { a: 1n, b: [1.5, "x"] } }, '{"a":1,"b":[1.5,"x"]}'],
    [{ status: 500, headers: { "x-y": "z" }, text: "oops" }, "oops"],
  ];
  for (const [response, expected] of responses) {
    test(`replay serves a ${"json" in response ? "JSON" : "text"} response`, async () => {
      const replay = new CassetteReplay(oneInteraction({ method: "GET", url: "https://h.test/" }, response));
      const reply = await send(replay, "https://h.test/");
      expect(reply.status).toBe(Number(response.status));
      expect(await reply.text()).toBe(expected);
      if (response.headers) expect(reply.headers.get("x-y")).toBe("z");
    });
  }

  test("replay serves chunks one per read", async () => {
    const chunks = ["data: one\n\n", "data: two\n\n"];
    const replay = new CassetteReplay(oneInteraction({ method: "GET", url: "https://h.test/" }, { status: 200, chunks }));
    const reader = (await send(replay, "https://h.test/")).body!.getReader();
    const decoder = new TextDecoder();
    const reads: string[] = [];
    for (let read = await reader.read(); !read.done; read = await reader.read()) reads.push(decoder.decode(read.value));
    expect(reads).toEqual(chunks);
  });
});

// Mismatches

describe("CassetteReplay mismatches", () => {
  const exampleBody = (text: string) => ({ model: "m", input: [{ role: "user", content: text }] });

  test("a body mismatch names the JSON path", async () => {
    const message = await mismatch({ method: "POST", url: "https://h.test/p", json: exampleBody("first") }, "https://h.test/p", {
      method: "POST",
      json: exampleBody("changed"),
    });
    expect(message).toBe("request 1 does not match the cassette:\n  body input[0].content: differs");
    expect(message).not.toContain("changed");
  });

  test("body keys missing or unexpected, a length and a non-JSON body are named", async () => {
    const recorded = { method: "POST", url: "https://h.test/p", json: { a: 1n, list: [1n, 2n], nested: { b: true } } };
    const message = await mismatch(recorded, "https://h.test/p", { method: "POST", json: { list: [1], nested: { b: true, c: 1 }, d: 1 } });
    expect(message.split("\n").slice(1)).toEqual([
      "  body a: missing",
      "  body list: length 1, recorded 2",
      "  body nested.c: unexpected",
      "  body d: unexpected",
    ]);
    const raw = await rejection(
      safeFetch("https://h.test/p", { method: "POST", body: "{nope" }, new CassetteReplay(oneInteraction(recorded)).send),
    );
    expect((raw as Error).message).toEndWith("body: not JSON");
    const extra = await mismatch({ method: "POST", url: "https://h.test/p" }, "https://h.test/p", { method: "POST", json: {} });
    expect(extra).toEndWith("body: present, recorded none");
  });

  test("an int and a float differ", async () => {
    const path = tempFile(JSON.stringify(oneInteraction({ method: "POST", url: "https://h.test/p", json: { n: 1 } })));
    const replay = new CassetteReplay(path);
    const error = await rejection(safeFetch("https://h.test/p", { method: "POST", body: '{"n": 1.0}' }, replay.send));
    expect((error as Error).message).toEndWith("body n: differs");
    await safeFetch("https://h.test/p", { method: "POST", body: '{"n": 1}' }, new CassetteReplay(path).send);
  });

  test("<redacted> matches any present value but not a missing one", async () => {
    const recorded = {
      method: "POST",
      url: "https://h.test/p?key=<redacted>&v=1",
      headers: { "x-api-key": CASSETTE_REDACTED },
      json: { token: CASSETTE_REDACTED, n: 1n },
    };
    const call = (url: string, headers: Record<string, string>, json: unknown) =>
      send(new CassetteReplay(oneInteraction(recorded)), url, { method: "POST", headers, json });

    await call("https://h.test/p?key=anything&v=1", { "x-api-key": "whatever" }, { token: "abc", n: 1 });
    const header = await rejection(call("https://h.test/p?key=anything&v=1", {}, { token: "abc", n: 1 }));
    expect((header as Error).message).toEndWith("header 'x-api-key': missing");
    const query = await rejection(call("https://h.test/p?v=1", { "x-api-key": "whatever" }, { token: "abc", n: 1 }));
    expect((query as Error).message).toEndWith("query parameter 'key': 0 value(s), recorded 1");
    const body = await rejection(call("https://h.test/p?key=anything&v=1", { "x-api-key": "whatever" }, { n: 1 }));
    expect((body as Error).message).toEndWith("body token: missing");
  });

  test("query parameters match as a multimap", async () => {
    const recorded = { method: "GET", url: "https://h.test/p?a=1&a=2&b=3" };
    await send(new CassetteReplay(oneInteraction(recorded)), "https://h.test/p?b=3&a=2&a=1");
    expect(await mismatch(recorded, "https://h.test/p?a=1&b=3")).toEndWith("query parameter 'a': 1 value(s), recorded 2");
    expect(await mismatch(recorded, "https://h.test/p?a=1&a=1&b=3")).toEndWith("query parameter 'a': value differs");
  });

  test("ignored headers may differ", async () => {
    const recorded = {
      method: "GET",
      url: "https://h.test/p",
      headers: { "user-agent": "someone-else/1.0", "content-length": "999", accept: "*/*" },
    };
    const replay = new CassetteReplay(oneInteraction(recorded));
    await send(replay, "https://h.test/p", { headers: { Accept: "*/*" } });
    replay.assertDone();
  });

  test("a mismatch never shows header or query values", async () => {
    const recorded = {
      method: "POST",
      url: "https://h.test/p?key=recorded-query-value",
      headers: { "X-Api-Key": "recorded-header-value" },
      json: {},
    };
    const message = await mismatch(recorded, `https://h.test/p?key=${KEY}`, { method: "POST", headers: { "x-api-key": KEY }, json: {} });
    expect(message).toBe(
      "request 1 does not match the cassette:\n  query parameter 'key': value differs\n  header 'x-api-key': value differs",
    );
    for (const value of [KEY, "recorded-query-value", "recorded-header-value"]) expect(message).not.toContain(value);
  });

  test("a method or URL mismatch fails", async () => {
    const recorded = { method: "post", url: "https://h.test/p" };
    await send(new CassetteReplay(oneInteraction(recorded)), "https://h.test/p", { method: "POST" });
    const cases: [string, string, string][] = [
      ["GET", "https://h.test/p", "method GET, recorded POST"],
      ["POST", "https://other.test/p", "url host differs"],
      ["POST", "https://h.test/q", "url path differs"],
      ["POST", "http://h.test/p", "url scheme differs"],
      ["POST", "https://h.test:8443/p", "url port differs"],
    ];
    for (const [method, url, diff] of cases) {
      expect(await mismatch(recorded, url, { method })).toBe(`request 1 does not match the cassette:\n  ${diff}`);
    }
  });

  test("at most 10 differences are shown", async () => {
    const recorded = Object.fromEntries(Array.from({ length: 12 }, (_, i) => [`k${i}`, i]));
    const actual = Object.fromEntries(Array.from({ length: 12 }, (_, i) => [`k${i}`, -1]));
    const message = await mismatch({ method: "POST", url: "https://h.test/p", json: recorded }, "https://h.test/p", {
      method: "POST",
      json: actual,
    });
    const lines = message.split("\n");
    expect(lines).toHaveLength(12);
    expect(lines[10]).toBe("  body k9: differs");
    expect(lines[11]).toBe("  ... and 2 more");
  });
});

// findSecrets

/** Escape every character of `text` as `\uXXXX`. */
function unicodeEscaped(text: string, upper = false): string {
  return Array.from(text, (c) => {
    const hex = c.charCodeAt(0).toString(16).padStart(4, "0");
    return `\\u${upper ? hex.toUpperCase() : hex}`;
  }).join("");
}

describe("findSecrets", () => {
  const patterns: [string, string][] = [
    ["key sk-proj-abcdefghijklmnop1234", "openai_key"],
    ["key sk-ant-api03-abcdefghijklmnop", "anthropic_key"],
    ["AIzaSyA1234567890abcdefghijklmnopqrstu", "google_api_key"],
    ["eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N", "jwt"],
    ["Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "bearer_token"],
  ];
  for (const [text, kind] of patterns) {
    test(`flags ${kind}`, () => {
      expect(findSecrets(text)).toContain(kind);
    });
  }

  const escaped = [
    "key \\u0073\\u006b\\u002dproj-abcdefghijklmnop1234",
    "key \\u0073\\u006B\\u002Dproj-abcdefghijklmnop1234",
    `key ${unicodeEscaped(KEY)}`,
    '{"key": "\\u0041IzaSyA1234567890abcdefghijklmnopqrstu"}',
  ];
  for (const text of escaped) {
    test(`flags an escaped key: ${text.slice(0, 30)}`, () => {
      expect(findSecrets(text).length).toBeGreaterThan(0);
    });
  }

  test("names kinds, not values", () => {
    const found = findSecrets("Bearer abcdefghijklmnopqrstuvwxyz and AIzaSyA1234567890abcdefghijklmnopqrstu");
    expect(found).toEqual(["google_api_key", "bearer_token"]);
  });

  test("one kind per match", () => {
    expect(findSecrets("sk-proj-abcdefghijklmnop1234 sk-proj-abcdefghijklmnop5678")).toEqual(["openai_key", "openai_key"]);
  });

  for (const text of ["Authorization: Bearer <redacted>", "task-0123456789abcdefgh", "risk-0123456789abcdefghijk", "no secrets here"]) {
    test(`ignores ${JSON.stringify(text)}`, () => {
      expect(findSecrets(text)).toEqual([]);
    });
  }
});

const REQUEST = ask("hi");

/** Return a reply with reasoning, text, a parsed and an unparsed tool call, and a refusal. */
function fullResponse(): Response {
  const message = new Message({
    role: "assistant",
    parts: [
      new ReasoningPart("thinking", { signature: "s" }),
      new TextPart("hello"),
      new ToolCallPart({ id: "c1", name: "search", arguments: { q: "x", n: 1n } }),
      new ToolCallPart({ id: "c2", name: "broken", arguments: null, rawArguments: "{not json" }),
      new RefusalPart("I can't help.", { id: "msg_1" }),
    ],
    protocol: "openai_responses",
    providerPayload: { id: "msg_1" },
  });
  return new Response({
    message,
    usage: new Usage({ inputTokens: 3, outputTokens: 4, totalTokens: 7 }),
    finishReason: "tool_calls",
    responseId: "r1",
  });
}

/** Return every event `stream` still yields. */
async function drain(stream: EventStream): Promise<Event[]> {
  const out: Event[] = [];
  for await (const event of stream) out.push(event);
  return out;
}

describe("ScriptedModel", () => {
  test("a response streams as events that collect back to it", async () => {
    const reply = fullResponse();
    const model = new ScriptedModel([reply]);

    const stream = model.stream(REQUEST);
    const events = await drain(stream);

    expect(stream).toBeInstanceOf(EventStream);
    const parts = reply.message.parts;
    expect(events).toEqual([
      new ReasoningDelta(0, "thinking"),
      new PartDone(0, parts[0]!),
      new TextDelta(1, "hello"),
      new PartDone(1, parts[1]!),
      new ToolCallDelta({ index: 2, id: "c1", name: "search", arguments: '{"n":1,"q":"x"}' }),
      new PartDone(2, parts[2]!),
      new ToolCallDelta({ index: 3, id: "c2", name: "broken", arguments: "{not json" }),
      new PartDone(3, parts[3]!),
      new PartDone(4, parts[4]!),
      new UsageReport(new Usage({ inputTokens: 3, outputTokens: 4, totalTokens: 7 })),
      new Finish({ reason: "tool_calls", responseId: "r1", protocol: "openai_responses", providerPayload: { id: "msg_1" } }),
    ]);
    expect(await collect(events)).toEqual(reply);
  });

  test("collecting the stream gives the reply and records the request", async () => {
    const reply = fullResponse();
    const model = new ScriptedModel([reply]);

    expect(await collect(model.stream(REQUEST))).toEqual(reply);
    expect(model.requests).toEqual([REQUEST]);
  });

  test("a raw event list streams as given", async () => {
    const events = [new TextDelta(0, "a"), new TextDelta(0, "b"), new Finish({ reason: "stop" })];
    const model = new ScriptedModel([events]);

    expect(await drain(model.stream(REQUEST))).toEqual(events);
  });

  test("a bare error is thrown by the first next()", async () => {
    const down = new Error("down");
    const model = new ScriptedModel([down]);

    const stream = model.stream(REQUEST);

    expect(model.requests).toEqual([REQUEST]);
    expect(await rejection(stream.next())).toBe(down);
  });

  test("an error in a list is thrown at its position and closes the stream", async () => {
    const cut = new Error("cut");
    const model = new ScriptedModel([[new TextDelta(0, "par"), cut]]);
    const stream = model.stream(REQUEST);

    expect(await stream.next()).toEqual({ done: false, value: new TextDelta(0, "par") });
    expect(await rejection(stream.next())).toBe(cut);
    expect(model.closed).toBe(1);
  });

  test("records each request and counts closed streams", async () => {
    const second = ask("again");
    const model = new ScriptedModel([fullResponse(), fullResponse(), fullResponse()], { provider: "p", model: "m" });
    expect([model.provider, model.model]).toEqual(["p", "m"]);

    const stream = model.stream(REQUEST);
    expect(model.closed).toBe(0);
    await stream.close();
    await stream.close();
    await collect(model.stream(second));
    await drain(model.stream(REQUEST));

    expect(model.requests).toEqual([REQUEST, second, REQUEST]);
    expect(model.closed).toBe(3);
  });

  test("names default to scripted", () => {
    const model = new ScriptedModel([]);
    expect([model.provider, model.model]).toEqual(["scripted", "scripted"]);
  });

  test("running out of replies throws at once", () => {
    const model = new ScriptedModel([]);
    expect(() => model.stream(REQUEST)).toThrow("no reply");
  });

  test("an aborted signal throws its reason before the next event", async () => {
    const model = new ScriptedModel([fullResponse()]);
    const controller = new AbortController();
    const stream = model.stream(REQUEST, { signal: controller.signal });

    expect(await stream.next()).toEqual({ done: false, value: new ReasoningDelta(0, "thinking") });
    const reason = new Error("stop now");
    controller.abort(reason);

    expect(await rejection(stream.next())).toBe(reason);
    expect(model.closed).toBe(1);
  });
});
