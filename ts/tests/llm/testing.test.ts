/**
 * Test helpers, a port of `tests/llm/test_testing.py`: cassette replay and recording through
 * `RouteModel` and `safeFetch`, `recordMode`, `findSecrets`, and the scripted model, plus an
 * aborted signal ending a stream.
 */

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { Buffer } from "node:buffer";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { safeFetch, type Send } from "../../src/platform.ts";
import { type Auth, BearerAuth, HeaderKeyAuth } from "../../src/llm/auth.ts";
import { collect, type Event, EventStream, Finish, PartDone, ReasoningDelta, TextDelta, ToolCallDelta, UsageReport } from "../../src/llm/events.ts";
import { RouteModel, type SseProtocol } from "../../src/llm/executor.ts";
import { Message, ReasoningPart, RefusalPart, Request, Response, TextPart, ToolCallPart, Usage } from "../../src/llm/messages.ts";
import { Route } from "../../src/llm/route.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";
import {
  type Cassette,
  CASSETTE_REDACTED,
  CassetteRecorder,
  CassetteReplay,
  type CassetteRequest,
  type CassetteResponse,
  findSecrets,
  loadCassette,
  recordMode,
  ScriptedModel,
} from "../../src/llm/testing.ts";

const CASSETTE = fileURLToPath(new URL("./cassettes/example_sse.json", import.meta.url));
// A fake key shaped like a real one, so findSecrets sees it.
const KEY = "sk-test-0123456789abcdefghijKLMN";
const OTHER_KEY = "sk-other-9876543210zyxwvutsrqPONM";
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

/** A protocol whose body is `buildBody`'s and whose events are each frame's data. */
function textProtocol(buildBody: (request: Request) => JsonObject): SseProtocol {
  return {
    name: "test",
    model: "m",
    framing: "sse",
    buildBody,
    async *parse(frames) {
      for await (const frame of frames) yield new TextDelta(0, frame.data);
      yield new Finish({ reason: "stop" });
    },
  };
}

/** A protocol whose body asks the first message's text. */
const EXAMPLE_PROTOCOL = textProtocol((request) => ({
  model: "m",
  input: [{ role: "user", content: (request.messages[0]!.parts[0] as TextPart).text }],
}));

/** A model over the example route, sending through `send`. */
function exampleModel(
  send: Send,
  options: { auth?: Auth; query?: Record<string, string>; protocol?: SseProtocol } = {},
): RouteModel {
  const { auth = new BearerAuth(KEY), query = { "api-version": "2026-01-01" }, protocol = EXAMPLE_PROTOCOL } = options;
  const route = new Route({ provider: "example", baseUrl: BASE, path: "responses", auth, query });
  return new RouteModel({ route, protocol, send });
}

/** Return the text of each `TextDelta` in `events`. */
function texts(events: Event[]): string[] {
  return events.flatMap((event) => (event instanceof TextDelta ? [event.text] : []));
}

/** Return a request of one user message with this text. */
function ask(text: string): Request {
  return new Request({ messages: [new Message({ role: "user", parts: [new TextPart(text)] })] });
}

/** Send one request through `replay` under `safeFetch`, with `json` as the body when given. */
function send(
  replay: { send: Send },
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

let tempDir = "";
let tempFiles = 0;
beforeAll(() => {
  tempDir = mkdtempSync(join(tmpdir(), "yian-cassette-"));
});
afterAll(() => {
  rmSync(tempDir, { recursive: true, force: true });
});

/** Return the path of a new temporary file holding `text`. */
function tempFile(text: string): string {
  const path = join(tempDir, `cassette-${++tempFiles}.json`);
  writeFileSync(path, text);
  return path;
}

// Replay

describe("CassetteReplay", () => {
  test("replays interactions in order through RouteModel", async () => {
    const replay = new CassetteReplay(CASSETTE);
    const model = exampleModel(replay.send);
    const replies: string[][] = [];
    for (const text of ["first", "second"]) {
      replies.push(texts(await drain(model.stream(ask(text)))));
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
    const stream = exampleModel(replay.send).stream(ask("first"));
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

  test("replay serves a 204 with no body", async () => {
    const replay = new CassetteReplay(oneInteraction({ method: "GET", url: "https://h.test/" }, { status: 204, text: "" }));
    const reply = await send(replay, "https://h.test/");
    expect(reply.status).toBe(204);
    expect(reply.body).toBeNull();
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
    const recorded = { method: "POST", url: "https://h.test/p", json: { a: 1, list: [1, 2], nested: { b: true } } };
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

  test("a bigint matches a number of its value, and no other", async () => {
    const post = (recorded: JsonValue, body: string) => {
      const replay = new CassetteReplay(oneInteraction({ method: "POST", url: "https://h.test/p", json: { n: recorded } }));
      return safeFetch("https://h.test/p", { method: "POST", body }, replay.send);
    };
    await post(1n, '{"n": 1.0}');
    await post(1, '{"n": 1}');
    const error = await rejection(post(1n, '{"n": 1.5}'));
    expect((error as Error).message).toEndWith("body n: differs");
  });

  test("<redacted> matches any present value but not a missing one", async () => {
    const recorded = {
      method: "POST",
      url: "https://h.test/p?key=<redacted>&v=1",
      headers: { "x-api-key": CASSETTE_REDACTED },
      json: { token: CASSETTE_REDACTED, n: 1 },
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

// Recording

/** A `Send` answering every request with `chunks` as the body, one per read, noting reads and a cancel. */
function inner(chunks: string[], headers: Record<string, string> = {}): { send: Send; stats: { pulled: number; cancelled: boolean } } {
  const stats = { pulled: 0, cancelled: false };
  const send: Send = async () => {
    let next = 0;
    const body = new ReadableStream<Uint8Array>(
      {
        pull(controller) {
          if (next >= chunks.length) return controller.close();
          stats.pulled++;
          controller.enqueue(new TextEncoder().encode(chunks[next++]!));
        },
        cancel() {
          stats.cancelled = true;
        },
      },
      { highWaterMark: 0 },
    );
    return new globalThis.Response(body, { status: 200, headers });
  };
  return { send, stats };
}

/** Record one SSE exchange whose key is in a header, the query, the body and the split reply. */
async function recordExchange(): Promise<Cassette> {
  const reply = ['data: {"echo": "' + KEY.slice(0, 12), KEY.slice(12) + '"}\n\ndata: [DONE]\n\n'];
  const headers = {
    "content-type": "text/event-stream",
    "retry-after": "1",
    "set-cookie": "session=abc",
    "openai-organization": "org-x",
  };
  const recorder = new CassetteRecorder(inner(reply, headers).send, { secrets: [KEY] });
  const model = exampleModel(recorder.send, {
    auth: new HeaderKeyAuth("x-api-key", KEY),
    query: { key: KEY },
    protocol: textProtocol(() => ({ model: "m", note: KEY })),
  });
  expect(texts(await drain(model.stream(ask("hi"))))).toEqual([`{"echo": "${KEY}"}`, "[DONE]"]);
  return recorder.cassette({ description: "a recorded exchange" });
}

/** Return the `key` query value of `url`. */
function queryKey(url: string): string | null {
  return new URL(url).searchParams.get("key");
}

/** Return `value` as JSON text, writing bigints as digits. */
function dump(value: unknown): string {
  return JSON.stringify(value, (_, item: unknown) => (typeof item === "bigint" ? Number(item) : item));
}

/** Return `text` as a JSON string holds it with everything outside printable ASCII `\\u`-escaped, as Python writes it. */
function asciiJson(text: string): string {
  return JSON.stringify(text).replace(/[^\x20-\x7e]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
}

describe("CassetteRecorder", () => {
  test("redacts the key everywhere, keeps allowlisted response headers and drops ignored request headers", async () => {
    const cassette = await recordExchange();
    expect(cassette.format).toBe("yian.cassette.v1");
    expect(cassette.source).toBe("recorded");
    expect(cassette.description).toBe("a recorded exchange");
    const [interaction] = cassette.interactions;
    const { request, response } = interaction!;

    expect(request.method).toBe("POST");
    expect(request.headers).toEqual({ "content-type": "application/json", accept: "application/json", "x-api-key": CASSETTE_REDACTED });
    expect(queryKey(request.url)).toBe(CASSETTE_REDACTED);
    expect(request.json).toEqual({ model: "m", note: CASSETTE_REDACTED });
    expect(response.status).toBe(200);
    expect(response.chunks).toEqual(['data: {"echo": "<redacted>"}\n\n', "data: [DONE]\n\n"]);
    expect(response.headers).toEqual({ "content-type": "text/event-stream", "retry-after": "1" });
    expect(dump(cassette)).not.toContain(KEY);
    expect(findSecrets(dump(cassette))).toEqual([]);
    new CassetteReplay(JSON.parse(dump(cassette)) as Cassette);
  });

  test("a recorded cassette replays with a different key", async () => {
    const replay = new CassetteReplay(await recordExchange());
    const model = exampleModel(replay.send, {
      auth: new HeaderKeyAuth("x-api-key", OTHER_KEY),
      query: { key: OTHER_KEY },
      protocol: textProtocol(() => ({ model: "m", note: OTHER_KEY })),
    });

    expect(texts(await drain(model.stream(ask("hi"))))).toEqual(['{"echo": "<redacted>"}', "[DONE]"]);
    replay.assertDone();
  });

  test("redacts percent, plus and JSON-escaped secrets, and other header values", async () => {
    const secret = "abc+def/ghi= " + "x".repeat(16);
    // A JSON encoder may escape "/" as "\/"; JSON.stringify does not, so do it by hand.
    const echo = JSON.stringify({ echo: secret }).replaceAll("/", "\\/");
    const recorder = new CassetteRecorder(inner([echo], { "content-type": "application/json" }).send, { secrets: [secret] });
    const url = `${BASE}/responses?${new URLSearchParams({ key: secret })}`;
    const init = { method: "POST", headers: { "x-note": `note ${secret}` }, body: JSON.stringify({ note: secret }) };
    expect(await (await safeFetch(url, init, recorder.send)).json()).toEqual({ echo: secret });
    const [interaction] = recorder.cassette({ description: "encoded secrets" }).interactions;
    const { request, response } = interaction!;

    expect(queryKey(request.url)).toBe(CASSETTE_REDACTED);
    expect(request.headers!["x-note"]).toBe("note <redacted>");
    expect(request.json).toEqual({ note: CASSETTE_REDACTED });
    expect(response.chunks).toEqual(['{"echo":"<redacted>"}']);
    const dumped = dump(interaction);
    const percent = encodeURIComponent(secret);
    for (const form of [secret, percent, percent.replaceAll("%20", "+"), secret.replaceAll("/", "\\/")]) {
      expect(dumped).not.toContain(form);
      expect(request.url + response.chunks!.join("")).not.toContain(form);
    }
  });

  test("redacts \\u-escaped secrets", async () => {
    const quoted = 'pa"ss\\wörd-' + "y".repeat(16);
    const escapedKey = unicodeEscaped(KEY);
    const reply = `{"a": "${escapedKey}", "b": "${unicodeEscaped(KEY, true)}", "c": ${asciiJson(quoted)}, "d": ${JSON.stringify(quoted)}}`;
    const recorder = new CassetteRecorder(inner([reply]).send, { secrets: [KEY, quoted] });
    const args = `{"key": "${escapedKey}", "other": ${asciiJson(quoted)}}`;
    const init = { method: "POST", body: JSON.stringify({ arguments: args }) };
    expect(await (await safeFetch(`${BASE}/responses`, init, recorder.send)).json()).toEqual({ a: KEY, b: KEY, c: quoted, d: quoted });
    const cassette = recorder.cassette({ description: "escaped secrets" });
    const { request, response } = cassette.interactions[0]!;

    expect(response.chunks).toEqual(['{"a": "<redacted>", "b": "<redacted>", "c": "<redacted>", "d": "<redacted>"}']);
    expect(request.json).toEqual({ arguments: '{"key": "<redacted>", "other": "<redacted>"}' });
    const saved = response.chunks!.join("") + (request.json as { arguments: string }).arguments;
    for (const secret of [KEY, quoted]) {
      const forms = [secret, JSON.stringify(secret).slice(1, -1), asciiJson(secret).slice(1, -1), unicodeEscaped(secret), unicodeEscaped(secret, true)];
      for (const form of forms) expect(saved).not.toContain(form);
    }
    expect(findSecrets(dump(cassette))).toEqual([]);
  });

  test("closing a stream early records what was read", async () => {
    const body = inner(["data: one\n\n", "data: two\n\n"]);
    const recorder = new CassetteRecorder(body.send, { secrets: [] });
    const stream = exampleModel(recorder.send).stream(ask("first"));

    expect(await stream.next()).toEqual({ done: false, value: new TextDelta(0, "one") });
    await stream.close();

    expect(body.stats).toEqual({ pulled: 1, cancelled: true });
    const [interaction] = recorder.cassette({ description: "half" }).interactions;
    expect(interaction!.response.chunks).toEqual(["data: one\n\n"]);
    expect(interaction!.request.headers!["authorization"]).toBe(CASSETTE_REDACTED);
  });

  test("the response text is saved one chunk per SSE event", async () => {
    const recorder = new CassetteRecorder(inner(["data: a\n\ndata: b\r\n\r\nda", "ta: c"]).send, { secrets: [] });
    await (await safeFetch(`${BASE}/responses`, {}, recorder.send)).text();
    const [interaction] = recorder.cassette({ description: "events" }).interactions;
    expect(interaction!.response.chunks).toEqual(["data: a\n\n", "data: b\r\n\r\n", "data: c"]);
    expect(interaction!.request).toEqual({ method: "GET", url: `${BASE}/responses`, headers: {} });
  });

  test("a function's secrets are read when the cassette is built", async () => {
    let secrets: string[] = [];
    const recorder = new CassetteRecorder(inner([`{"echo": "${KEY}"}`]).send, { secrets: () => secrets });
    await (await safeFetch(`${BASE}/responses`, {}, recorder.send)).text();
    secrets = [KEY];
    expect(recorder.cassette({ description: "lazy" }).interactions[0]!.response.chunks).toEqual(['{"echo": "<redacted>"}']);
  });

  // Every 3-byte run of "~" encodes as "fn5+", so each alignment's fragment holds a "+".
  const BASE64_SECRET = "secret-" + "~".repeat(12) + "-value";
  for (const alignment of [0, 1, 2]) {
    for (const urlSafe of [false, true]) {
      const form = urlSafe ? "URL-safe base64" : "base64";
      test(`the value scan throws on a ${form} form at alignment ${alignment}`, async () => {
        let blob = Buffer.from("ab".slice(0, alignment) + BASE64_SECRET).toString("base64");
        if (urlSafe) blob = blob.replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
        const recorder = new CassetteRecorder(inner([`{"blob": "${blob}"}`]).send, { secrets: [BASE64_SECRET] });
        await (await safeFetch(`${BASE}/responses`, {}, recorder.send)).text();

        const error = thrown(() => recorder.cassette({ description: "base64" }));
        expect(error.message).toBe(`the cassette still holds a credential (${form}) in interactions[0].response.chunks[0]`);
        expect(error.message).not.toContain(blob.slice(4, 12));
      });
    }
  }

  test("the value scan names the place of a form the scrub can't reach", async () => {
    // A header name is saved as sent, so the scrub never sees it.
    const recorder = new CassetteRecorder(inner(["ok"]).send, { secrets: ["hidden-name-0123"] });
    await (await safeFetch(`${BASE}/responses`, { headers: { "x-hidden-name-0123": "v" } }, recorder.send)).text();
    const error = thrown(() => recorder.cassette({ description: "names" }));
    expect(error.message).toBe("the cassette still holds a credential (plain) in interactions[0].request.headers (a key)");
  });

  test("a leftover findSecrets shape throws naming the kind", async () => {
    const jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N";
    const recorder = new CassetteRecorder(inner([`{"token": "${jwt}"}`]).send, { secrets: [] });
    await (await safeFetch(`${BASE}/responses`, {}, recorder.send)).text();

    const error = thrown(() => recorder.cassette({ description: "jwt" }));
    expect(error.message).toBe("the cassette still holds a credential (jwt) in interactions[0].response.chunks[0]");
    expect(error.message).not.toContain("eyJ");
  });
});

describe("recordMode", () => {
  const env = (values: Record<string, string>) => (name: string) => values[name];

  test("RECORD=1 records", () => {
    expect(recordMode(env({ RECORD: "1" }))).toBe(true);
    expect(recordMode(env({ RECORD: "1", CI: "" }))).toBe(true);
  });

  test("no RECORD=1 replays, with or without CI", () => {
    expect(recordMode(env({}))).toBe(false);
    expect(recordMode(env({ RECORD: "true" }))).toBe(false);
    expect(recordMode(env({ CI: "true" }))).toBe(false);
  });

  test("RECORD=1 with CI set throws", () => {
    expect(() => recordMode(env({ RECORD: "1", CI: "true" }))).toThrow("RECORD=1 is refused when CI is set");
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
