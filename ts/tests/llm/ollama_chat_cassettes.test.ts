/**
 * Golden scenarios for `src/llm/protocols/ollama_chat.ts`, replayed from streamed cassettes in
 * `cassettes/ollama_chat/` through `RouteModel` (and `callWithRetries` for the retry scenarios).
 * Each cassette records the request the adapter must send to Ollama's native `/api/chat` (#366),
 * with every reply streamed as NDJSON.
 */

import { describe, expect, spyOn, test } from "bun:test";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { type Auth, BearerAuth, HeaderKeyAuth, NoAuth } from "../../src/llm/auth.ts";
import { type ErrorKind, ProviderError } from "../../src/llm/errors.ts";
import { collect } from "../../src/llm/events.ts";
import { callWithRetries, type RetryInfo, RouteModel } from "../../src/llm/executor.ts";
import { iterNdjson } from "../../src/llm/framing.ts";
import {
  Message,
  type Part,
  ReasoningPart,
  type ReplayDegradedReason,
  type ReplayKey,
  replayRecord,
  Request,
  type Response,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  ToolSpec,
  Usage,
} from "../../src/llm/messages.ts";
import { OllamaChat, PROTOCOL } from "../../src/llm/protocols/ollama_chat.ts";
import { Route } from "../../src/llm/route.ts";
import { CassetteReplay, IGNORED_HEADERS, loadCassette } from "../../src/llm/testing.ts";
import { virtualScheduler } from "../../src/platform.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

const DIR = fileURLToPath(new URL("cassettes/ollama_chat/", import.meta.url));
const BASE_URL = "http://localhost:11434";
const PROVIDER = "ollama";
const QWEN = "qwen3";
// The probe's server-assigned call ids.
const ID_1 = "call_k3x9q2ab";
const ID_2 = "call_m7p4z8cd";

const DOUBLE = new ToolSpec({
  name: "double",
  description: "Double an integer.",
  parameters: { properties: { value: { type: "integer" } }, required: ["value"], type: "object" },
});
const PICK = new ToolSpec({
  name: "pick",
  description: "Pick one.",
  parameters: {
    $defs: { Size: { enum: ["s", "m"], type: "string" } },
    additionalProperties: false,
    properties: {
      size: { $ref: "#/$defs/Size" },
      note: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
      count: { type: "integer", minimum: 1n, default: 1n },
      when: { type: "string", format: "date" },
      box: { type: "object", properties: { w: { type: "integer" } }, required: ["w"] },
    },
    required: ["size", "box"],
    title: "Pick",
    type: "object",
  },
});
// The probe's `done` line: prompt 17, eval 9.
const USAGE = new Usage({ inputTokens: 17, outputTokens: 9, totalTokens: 26 });

/** Build a scenario's route: Ollama (or `baseUrl`) with `api/chat`. */
function route(baseUrl: string, auth: Auth, headers: Record<string, string>): Route {
  return new Route({ provider: PROVIDER, baseUrl, path: "api/chat", auth, headers });
}

const system = (text: string) => new Message({ role: "system", parts: [new TextPart(text)] });
const user = (text: string) => new Message({ role: "user", parts: [new TextPart(text)] });
const toolResult = (callId: string, content: string) =>
  new Message({ role: "tool", parts: [new ToolResultPart({ callId, name: "double", content, isError: false })] });

function request(messages: Message[], options: { tools?: ToolSpec[] } = {}): Request {
  return new Request({ messages, ...options });
}

const PLAIN = [system("You are terse."), user("Say hello.")];
const TOOLS = [system("You are terse."), user("Double 3.")];

/**
 * Parse one recorded reply of a cassette straight from its chunks, for a later turn's prior
 * (replaying it through `CassetteReplay` would need its request to match).
 */
async function priorFrom(name: string, protocol: OllamaChat, interaction = 0): Promise<Message> {
  const chunks = loadCassette(join(DIR, `${name}.json`)).interactions[interaction]!.response.chunks!;
  const response = await collect(protocol.parse(iterNdjson(chunks.map((chunk) => new TextEncoder().encode(chunk)))));
  return response.message;
}

interface Call {
  id: string;
  name: string;
  arguments: JsonObject | null;
  rawArguments: string | null;
}

/**
 * What one turn's reply must parse to; `calls` defaults to none, `finish` to `stop`, `usage` to
 * the probe's, `records` (each part's replay record fields, `null` for no payload) to no payload
 * on any part. `reasoning` (the reasoning parts' joined text) is checked when given.
 */
interface Reply {
  text: string;
  parts: string[];
  calls?: Call[];
  finish?: string;
  reasoning?: string;
  records?: (JsonObject | null)[];
  usage?: Usage;
}

/** A turn's request, built from the earlier turns' replies and the scenario's protocol. */
type Turn = (previous: Response[], protocol: OllamaChat) => Request | Promise<Request>;

interface Scenario {
  /** Defaults to `http://localhost:11434`. */
  baseUrl?: string;
  /** Defaults to `NoAuth`. */
  auth?: Auth;
  /** Connection headers. */
  headers?: Record<string, string>;
  /** Header values every request must send (the cassette compares credentials redacted). */
  sent?: Record<string, string>;
  block?: JsonObject;
  turns: Turn[];
  /** One per turn, unless the last turn fails. */
  replies: Reply[];
  /** The last turn's error. */
  error?: { kind: ErrorKind; retryable: boolean; message: string };
  /** Run each turn through `callWithRetries`. */
  retries?: boolean;
  /**
   * The `onRetry` calls: the error kind and message, the partial message's text (`null` for no
   * partial), the wait.
   */
  retried?: { kind: ErrorKind; message: string; partial: string | null; retryIn: number }[];
  /** The `onReplayDegraded` calls. */
  degraded?: ReplayDegradedReason[];
}

const call = (id: string, value: bigint): Call => ({
  id,
  name: "double",
  arguments: { value },
  rawArguments: null,
});
const text = (value: string, extra: Partial<Reply> = {}): Reply => ({ text: value, parts: ["text"], ...extra });
const hello = text("Hello there.");
const answer6 = text("The answer is 6.");
const thoughtHello: Reply = { ...hello, parts: ["reasoning", "text"], reasoning: "Let me think.", records: [{}, null] };
const doubleCall: Reply = { text: "", parts: ["tool_call"], calls: [call("yian_call_0", 3n)], finish: "tool_calls" };
const thoughtCall: Reply = {
  text: "",
  parts: ["reasoning", "tool_call"],
  calls: [call(ID_1, 3n)],
  finish: "tool_calls",
  reasoning: "Let me think.",
  records: [{}, null],
};

const PLAIN_TURN: Turn = () => request(PLAIN);
const TOOLS_TURN: Turn = () => request(TOOLS, { tools: [DOUBLE] });

/** The second turn after the first turn's tool call reply, with a result for each call id. */
function replyTurn2(results: [string, string][]): Turn {
  return ([first]) =>
    request([...TOOLS, first!.message, ...results.map(([callId, content]) => toolResult(callId, content))], {
      tools: [DOUBLE],
    });
}

const KEY_AUTH = new BearerAuth("test-key-1");
const SENT_KEY = { authorization: "Bearer test-key-1" };

const SCENARIOS: Record<string, Scenario> = {
  plain: { turns: [PLAIN_TURN], replies: [hello] },
  user_multipart: {
    turns: [
      () => {
        const parts = [new TextPart("A"), new TextPart("B")];
        return request([system("You are terse."), new Message({ role: "user", parts })]);
      },
    ],
    replies: [hello],
  },
  think_true: { block: { think: true }, turns: [PLAIN_TURN], replies: [thoughtHello] },
  think_high: { block: { think: "high" }, turns: [PLAIN_TURN], replies: [thoughtHello] },

  tools: { turns: [TOOLS_TURN], replies: [doubleCall] },
  turn2: {
    turns: [
      async (_, protocol) =>
        request([...TOOLS, await priorFrom("tools", protocol), toolResult("yian_call_0", "6")], { tools: [DOUBLE] }),
    ],
    replies: [answer6],
  },
  two_call_chunks: {
    turns: [
      TOOLS_TURN,
      replyTurn2([
        [ID_1, "6"],
        [ID_2, "8"],
      ]),
    ],
    replies: [
      { text: "", parts: ["tool_call", "tool_call"], calls: [call(ID_1, 3n), call(ID_2, 4n)], finish: "tool_calls" },
      answer6,
    ],
  },
  thinking_tools: {
    block: { think: true },
    turns: [TOOLS_TURN, replyTurn2([[ID_1, "6"]])],
    replies: [thoughtCall, answer6],
  },
  replay_degraded: {
    block: { think: true },
    turns: [
      async () => {
        // Parsed for `qwen3:32b`, so its reasoning record doesn't match a `qwen3` protocol.
        const prior = await priorFrom("thinking_tools", new OllamaChat({ model: "qwen3:32b", provider: PROVIDER }));
        return request([...TOOLS, prior, toolResult(ID_1, "6")], { tools: [DOUBLE] });
      },
    ],
    replies: [answer6],
    degraded: ["other_model"],
  },
  foreign_history: {
    turns: [
      () => {
        const prior = new Message({
          role: "assistant",
          parts: [
            new ReasoningPart(
              "Let me think.",
              replayRecord(
                { protocol: "openai_chat", provider: "deepseek", model: "deepseek-reasoner" },
                { field: "reasoning_content" },
              ),
            ),
            new ToolCallPart({ id: "functions.double:0", name: "double", arguments: { value: 3n } }),
          ],
          protocol: "openai_chat",
        });
        return request([...TOOLS, prior, toolResult("functions.double:0", "6")], { tools: [DOUBLE] });
      },
    ],
    replies: [answer6],
    degraded: ["other_protocol"],
  },
  rich_schema: {
    turns: [() => request([system("You are terse."), user("Pick.")], { tools: [PICK] })],
    replies: [hello],
  },

  block_merge: {
    block: {
      options: { num_ctx: 8192n, temperature: 0.6, stop: ["X"], seed: null },
      keep_alive: -1n,
      think: false,
      logprobs: true,
      top_logprobs: 2n,
    },
    turns: [PLAIN_TURN],
    replies: [hello],
  },
  connection_settings: { headers: { "X-Gateway": "gw-1" }, turns: [PLAIN_TURN], replies: [hello] },
  custom_base_url: { baseUrl: "http://gpu-box:11434", turns: [PLAIN_TURN], replies: [hello] },
  api_key: { auth: KEY_AUTH, sent: SENT_KEY, turns: [PLAIN_TURN], replies: [hello] },
  header_key_echo: {
    auth: new HeaderKeyAuth("x-gateway-token", "test-key-2"),
    sent: { "x-gateway-token": "test-key-2" },
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "auth", retryable: false, message: "ollama returned HTTP 401 (auth): token [REDACTED] rejected" },
  },
  error_401_key_echo: {
    auth: KEY_AUTH,
    sent: SENT_KEY,
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "auth", retryable: false, message: "ollama returned HTTP 401 (auth): invalid key [REDACTED]" },
  },
  stream_error_key_echo: {
    auth: KEY_AUTH,
    sent: SENT_KEY,
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [
      {
        kind: "server",
        message: "ollama stream failed (server): upstream rejected [REDACTED]",
        partial: "Hello",
        retryIn: 0.5,
      },
    ],
  },

  usage_cached: {
    turns: [PLAIN_TURN],
    replies: [
      text("Hello there.", {
        usage: new Usage({ inputTokens: 17, outputTokens: 9, totalTokens: 26, cacheReadTokens: 12 }),
      }),
    ],
  },
  cached_prompt: {
    turns: [PLAIN_TURN],
    replies: [text("Hello there.", { usage: new Usage({ inputTokens: 0, outputTokens: 9, totalTokens: 9 }) })],
  },
  length: { turns: [TOOLS_TURN], replies: [{ ...doubleCall, finish: "length" }] },

  error_404: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "not_found",
      retryable: false,
      message: "ollama returned HTTP 404 (not_found): model 'qwen3' not found",
    },
  },
  error_400_context: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "context_length",
      retryable: false,
      message:
        "ollama returned HTTP 400 (context_length): the prompt is longer than the context length currently " +
        "available to the model; shorten the prompt, adjust the context length in settings, or use a model with " +
        "a longer context length",
    },
  },
  error_400_context_size: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "context_length",
      retryable: false,
      message:
        "ollama returned HTTP 400 (context_length): request (302 tokens) exceeds the available context size " +
        "(256 tokens), try increasing it",
    },
  },
  error_429_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [
      {
        kind: "rate_limit",
        message: "ollama returned HTTP 429 (rate_limit): too many requests",
        partial: null,
        retryIn: 0.5,
      },
    ],
  },
  error_500_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [
      {
        kind: "server",
        message: "ollama returned HTTP 500 (server): llama runner process has terminated: exit status 2",
        partial: null,
        retryIn: 0.5,
      },
    ],
  },
  redirect_307: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "redirect",
      retryable: false,
      message: "ollama returned HTTP 307 (redirect): redirect to https://evil.example",
    },
  },
  stream_error_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [
      {
        kind: "server",
        message: "ollama stream failed (server): an error was encountered while running the model: unexpected EOF",
        partial: "Hello",
        retryIn: 0.5,
      },
    ],
  },
  truncated_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [
      {
        kind: "incomplete",
        message: "ollama stream ended before a complete response",
        partial: "Hello there.",
        retryIn: 0.5,
      },
    ],
  },
  invalid_line: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "protocol", retryable: false, message: "invalid JSON line 2 in NDJSON stream" },
  },
  split_lines: { turns: [PLAIN_TURN], replies: [hello] },
};

function expectReply(response: Response, reply: Reply, key: ReplayKey): void {
  const { message } = response;
  expect(message.text()).toBe(reply.text);
  expect(message.parts.map((part) => part.type)).toEqual(reply.parts as never);
  const calls = message.toolCalls().map(({ id, name, arguments: args, rawArguments }) => ({
    id,
    name,
    arguments: args,
    rawArguments,
  }));
  expect(calls).toEqual(reply.calls ?? []);
  if (reply.reasoning !== undefined) {
    const reasoning = message.parts.filter((part: Part): part is ReasoningPart => part instanceof ReasoningPart);
    expect(reasoning.map((part) => part.text).join("")).toBe(reply.reasoning);
  }
  const records = reply.records ?? reply.parts.map(() => null);
  const payloads = records.map((fields) => (fields === null ? null : replayRecord(key, fields)));
  expect(message.parts.map((part) => part.providerPayload)).toEqual(payloads);
  expect(message.protocol).toBe(PROTOCOL);
  expect(message.providerPayload).toEqual(replayRecord(key, { response_model: QWEN }));
  expect(response.usage).toEqual(reply.usage ?? USAGE);
  expect(response.finishReason).toBe(reply.finish ?? "stop");
  expect(response.responseId).toBeNull();
}

async function runScenario(name: string, scenario: Scenario): Promise<void> {
  const cassette = loadCassette(join(DIR, `${name}.json`));
  const replay = new CassetteReplay(cassette);
  const degraded: ReplayDegradedReason[] = [];
  const protocol = new OllamaChat({
    model: QWEN,
    provider: PROVIDER,
    block: scenario.block,
    onReplayDegraded: (reason) => degraded.push(reason),
  });
  const model = new RouteModel({
    route: route(scenario.baseUrl ?? BASE_URL, scenario.auth ?? new NoAuth(), scenario.headers ?? {}),
    protocol,
    send: replay.send,
  });
  expect(model.provider).toBe(protocol.provider);
  const key: ReplayKey = { protocol: PROTOCOL, provider: PROVIDER, model: QWEN, account: null };

  const retried: RetryInfo[] = [];
  const send = (req: Request): Promise<Response> =>
    scenario.retries
      ? callWithRetries(model, req, {
          scheduler: virtualScheduler(),
          random: () => 0,
          onRetry: (retry) => void retried.push(retry),
        })
      : collect(model.stream(req));
  const responses: Response[] = [];
  // Stubbed so an unexpected empty-user warning fails the scenario rather than being printed.
  const emit = spyOn(process, "emitWarning").mockImplementation(() => {});
  try {
    for (const [index, turn] of scenario.turns.entries()) {
      const req = await turn(responses, protocol);
      if (index === scenario.turns.length - 1 && scenario.error) {
        const failed = send(req);
        await expect(failed).rejects.toBeInstanceOf(ProviderError);
        await expect(failed).rejects.toMatchObject(scenario.error);
        break;
      }
      responses.push(await send(req));
    }
    expect(emit.mock.calls).toEqual([]);
  } finally {
    emit.mockRestore();
  }
  replay.assertDone();

  expect(responses.length).toBe(scenario.replies.length);
  scenario.replies.forEach((reply, index) => expectReply(responses[index]!, reply, key));
  expect(degraded).toEqual(scenario.degraded ?? []);
  const seen = retried.map((retry) => ({
    kind: retry.error.kind,
    message: retry.error.message,
    partial: retry.partial === null ? null : retry.partial.text(),
    retryIn: retry.retryIn,
  }));
  expect(seen).toEqual(scenario.retried ?? []);
  for (const received of replay.requests) {
    for (const [header, value] of Object.entries(scenario.sent ?? {})) expect(received.headers.get(header)).toBe(value);
  }

  // CassetteReplay checks that every recorded header was sent; this checks the reverse.
  replay.requests.forEach((received, index) => {
    const recorded = Object.keys(cassette.interactions[index]!.request.headers ?? {}).map((each) =>
      each.toLowerCase(),
    );
    const sent = [...received.headers.keys()].filter((each) => !IGNORED_HEADERS.has(each));
    expect(sent.filter((each) => !recorded.includes(each))).toEqual([]);
  });
}

describe("Ollama chat cassettes", () => {
  for (const [name, scenario] of Object.entries(SCENARIOS)) {
    test(name, () => runScenario(name, scenario));
  }

  test("every cassette file belongs to a scenario and loads", () => {
    const names = readdirSync(DIR)
      .filter((file) => file.endsWith(".json"))
      .map((file) => file.slice(0, -".json".length))
      .sort();
    expect(names).toEqual(Object.keys(SCENARIOS).sort());
    for (const name of names) expect(loadCassette(join(DIR, `${name}.json`)).source).toBe("synthetic");
  });
});
