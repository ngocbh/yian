/**
 * Golden scenarios for `src/llm/protocols/gemini.ts`, replayed from streamed cassettes in
 * `cassettes/gemini/` through `RouteModel` (and `callWithRetries` for the retry scenarios). Each
 * cassette records the request the adapter must send to the Gemini Developer API's
 * `streamGenerateContent` (#366), with every call streamed.
 */

import { describe, expect, spyOn, test } from "bun:test";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { HeaderKeyAuth } from "../../src/llm/auth.ts";
import { type ErrorKind, ProviderError } from "../../src/llm/errors.ts";
import { collect } from "../../src/llm/events.ts";
import { callWithRetries, type RetryInfo, RouteModel } from "../../src/llm/executor.ts";
import { iterSse } from "../../src/llm/framing.ts";
import {
  Message,
  NamedTool,
  type Part,
  ReasoningPart,
  type ReplayDegradedReason,
  type ReplayKey,
  replayRecord,
  Request,
  type Response,
  TextPart,
  ToolCallPart,
  type ToolChoice,
  ToolResultPart,
  ToolSpec,
  Usage,
} from "../../src/llm/messages.ts";
import { Gemini, PROTOCOL } from "../../src/llm/protocols/gemini.ts";
import { Route } from "../../src/llm/route.ts";
import { CassetteReplay, IGNORED_HEADERS, loadCassette } from "../../src/llm/testing.ts";
import { virtualScheduler } from "../../src/platform.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

const DIR = fileURLToPath(new URL("cassettes/gemini/", import.meta.url));
const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";
const PROVIDER = "google";
const FLASH = "gemini-2.5-flash";
const PRO_3 = "gemini-3-pro-preview";
// The probe's signatures.
const SIG = "c2lnLW9uZQ==";
const SIG2 = "c2lnLXR3bw==";

const DOUBLE = new ToolSpec({
  name: "double",
  description: "Double an integer.",
  parameters: { properties: { value: { type: "integer" } }, required: ["value"], type: "object" },
});
const ANSWER = new ToolSpec({
  name: "Answer",
  description: "An answer.",
  parameters: {
    properties: { answer: { type: "string" }, score: { type: "integer" } },
    required: ["answer", "score"],
    type: "object",
  },
});
const PICK = new ToolSpec({
  name: "pick",
  description: "Pick one.",
  parameters: {
    title: "Pick",
    type: "object",
    additionalProperties: false,
    properties: {
      kind: { const: "z", title: "Kind" },
      count: { type: "integer", minimum: 1n, default: 1n },
      note: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
    },
    required: ["kind"],
  },
});
// Every chunk repeats the probe's cumulative `usageMetadata`: prompt 12, candidates 40,
// thoughts 25, cached 8, total 77.
const USAGE = new Usage({
  inputTokens: 12,
  outputTokens: 65,
  totalTokens: 77,
  cacheReadTokens: 8,
  reasoningTokens: 25,
});

/** Build a scenario's route: the Gemini API (or `baseUrl`) with the model's streaming path and `alt=sse`. */
function route(model: string, baseUrl: string, headers: Record<string, string>): Route {
  return new Route({
    provider: PROVIDER,
    baseUrl,
    path: `models/${model}:streamGenerateContent`,
    query: { alt: "sse" },
    auth: new HeaderKeyAuth("x-goog-api-key", "test-key-1"),
    headers,
  });
}

const system = (text: string) => new Message({ role: "system", parts: [new TextPart(text)] });
const user = (text: string) => new Message({ role: "user", parts: [new TextPart(text)] });
const toolResult = (callId: string, content: string, isError = false) =>
  new Message({ role: "tool", parts: [new ToolResultPart({ callId, name: "double", content, isError })] });

function request(messages: Message[], options: { tools?: ToolSpec[]; toolChoice?: ToolChoice } = {}): Request {
  return new Request({ messages, ...options });
}

const PLAIN = [system("You are terse."), user("Say hello.")];
const TOOLS = [system("You are terse."), user("Double 3.")];

/**
 * Parse one recorded reply of a cassette straight from its chunks, for a later turn's prior
 * (replaying it through `CassetteReplay` would need its request to match).
 */
async function priorFrom(name: string, protocol: Gemini, interaction = 0): Promise<Message> {
  const chunks = loadCassette(join(DIR, `${name}.json`)).interactions[interaction]!.response.chunks!;
  const response = await collect(protocol.parse(iterSse(chunks.map((chunk) => new TextEncoder().encode(chunk)))));
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
type Turn = (previous: Response[], protocol: Gemini) => Request | Promise<Request>;

interface Scenario {
  /** Defaults to `gemini-2.5-flash`. */
  model?: string;
  /** Defaults to `https://generativelanguage.googleapis.com/v1beta`. */
  baseUrl?: string;
  /** Connection headers. */
  headers?: Record<string, string>;
  block?: JsonObject;
  turns: Turn[];
  /** One per turn, unless the last turn fails. */
  replies: Reply[];
  /** The last turn's error. */
  error?: { kind: ErrorKind; retryable: boolean; message: string };
  /** Run each turn through `callWithRetries`. */
  retries?: boolean;
  /** The `onRetry` calls: the error kind, the partial message's text (`null` for no partial), the wait. */
  retried?: { kind: ErrorKind; partial: string | null; retryIn: number }[];
  /** The `onReplayDegraded` calls. */
  degraded?: ReplayDegradedReason[];
}

const call = (id: string, name: string, args: JsonObject | null, rawArguments: string | null = null): Call => ({
  id,
  name,
  arguments: args,
  rawArguments,
});
const text = (value: string, extra: Partial<Reply> = {}): Reply => ({ text: value, parts: ["text"], ...extra });
const hello = text("Hello there.");
const answer6 = text("The answer is 6.");
const doubleCall = call("yian_call_0", "double", { value: 3n });
const helloCall: Reply = {
  text: "Hello there.",
  parts: ["text", "tool_call"],
  calls: [doubleCall],
  finish: "tool_calls",
  records: [null, { thought_signature: SIG }],
};

const PLAIN_TURN: Turn = () => request(PLAIN);
const TOOLS_TURN: Turn = () => request(TOOLS, { tools: [DOUBLE] });

/** The second turn after the `tools` cassette, its prior parsed by `parser` (the scenario's protocol by default). */
function toolsTurn2(parser?: Gemini): Turn {
  return async (_, protocol) =>
    request([...TOOLS, await priorFrom("tools", parser ?? protocol), toolResult("yian_call_0", "6")], {
      tools: [DOUBLE],
    });
}

/** The second turn after the first turn's tool call reply, with a result for each call id. */
function replyTurn2(results: [string, string][]): Turn {
  return ([first]) =>
    request([...TOOLS, first!.message, ...results.map(([callId, content]) => toolResult(callId, content))], {
      tools: [DOUBLE],
    });
}

/** A protocol for `gemini-3-pro-preview`, whose records a `gemini-2.5-flash` protocol can't replay. */
const pro3 = () => new Gemini({ model: PRO_3, provider: PROVIDER });

const SCENARIOS: Record<string, Scenario> = {
  plain: { turns: [PLAIN_TURN], replies: [hello] },
  multitext: { turns: [PLAIN_TURN], replies: [text("Hello there.Part B.")] },
  thought: {
    block: { generationConfig: { thinkingConfig: { includeThoughts: true } } },
    turns: [PLAIN_TURN],
    replies: [
      {
        ...hello,
        parts: ["reasoning", "text"],
        reasoning: "Let me think.",
        records: [{ thought_signature: SIG2 }, null],
      },
    ],
  },

  tools: { turns: [TOOLS_TURN], replies: [helloCall] },
  turn2: { turns: [toolsTurn2()], replies: [answer6] },
  turn2_server_id: {
    turns: [TOOLS_TURN, replyTurn2([["call-srv-1", "6"]])],
    replies: [
      {
        text: "",
        parts: ["tool_call"],
        calls: [call("call-srv-1", "double", { value: 3n })],
        finish: "tool_calls",
        records: [{ thought_signature: SIG }],
      },
      answer6,
    ],
  },
  turn2_text_signature: {
    model: PRO_3,
    turns: [PLAIN_TURN, ([first]) => request([...PLAIN, first!.message, user("And again.")])],
    replies: [
      text("Hello there.", { records: [{ thought_signature: SIG2 }] }),
      text("Part B.", { records: [{ thought_signature: SIG2 }] }),
    ],
  },
  placeholder: {
    model: PRO_3,
    turns: [
      () => {
        const prior = new Message({
          role: "assistant",
          parts: [new ToolCallPart({ id: "call_1", name: "double", arguments: { value: 3n } })],
          protocol: "openai_chat",
        });
        return request([...TOOLS, prior, toolResult("call_1", "6")], { tools: [DOUBLE] });
      },
    ],
    replies: [answer6],
  },
  turn2_unsigned_call: {
    turns: [TOOLS_TURN, replyTurn2([["yian_call_0", "6"]])],
    replies: [
      {
        text: "Hello",
        parts: ["text", "tool_call"],
        calls: [doubleCall],
        finish: "tool_calls",
        records: [{ thought_signature: SIG }, null],
      },
      answer6,
    ],
  },
  parallel_calls: {
    turns: [
      TOOLS_TURN,
      replyTurn2([
        ["yian_call_0", "6"],
        ["yian_call_1", "8"],
      ]),
    ],
    replies: [
      {
        text: "",
        parts: ["tool_call", "tool_call"],
        calls: [doubleCall, call("yian_call_1", "double", { value: 4n })],
        finish: "tool_calls",
        records: [{ thought_signature: SIG }, null],
      },
      answer6,
    ],
  },
  tool_results: {
    turns: [
      () => {
        const ids = ["c0", "c1", "c2", "c3", "c4", "c5"];
        const prior = new Message({
          role: "assistant",
          parts: ids.map((id) => new ToolCallPart({ id, name: "double", arguments: { value: 3n } })),
        });
        const results = [
          toolResult("c0", "6"),
          toolResult("c1", "true"),
          toolResult("c2", "[1, 2]"),
          toolResult("c3", '{"result": 2}'),
          toolResult("c4", "plain"),
          toolResult("c5", "error: boom", true),
        ];
        return request([...TOOLS, prior, ...results], { tools: [DOUBLE] });
      },
    ],
    replies: [answer6],
  },
  rich_schema: {
    turns: [() => request([system("You are terse."), user("Pick.")], { tools: [PICK] })],
    replies: [hello],
  },
  two_tools: { turns: [() => request(TOOLS, { tools: [DOUBLE, ANSWER] })], replies: [hello] },
  tool_choice_named: {
    turns: [() => request([user("hi")], { tools: [ANSWER], toolChoice: new NamedTool("Answer") })],
    replies: [
      {
        text: "",
        parts: ["tool_call"],
        calls: [call("yian_call_0", "Answer", { answer: "hi", score: 3n })],
        finish: "tool_calls",
        records: [{ thought_signature: SIG }],
      },
    ],
  },
  system_multi: {
    turns: [
      () =>
        request([
          system("S1"),
          system("S2"),
          user("Hi."),
          new Message({ role: "assistant", parts: [new TextPart("Hello.")] }),
          system("S3"),
          user("Bye."),
        ]),
    ],
    replies: [hello],
  },
  system_late_only: { turns: [() => request([user("Hi."), system("Answer in French.")])], replies: [hello] },

  block_merge: {
    block: {
      generationConfig: {
        thinkingConfig: { thinkingBudget: 1024n, includeThoughts: true },
        maxOutputTokens: 512n,
        topK: 40n,
        temperature: 0.5,
        stopSequences: ["STOP"],
        topP: null,
      },
      safetySettings: [{ category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" }],
      serviceTier: "flex",
      seed: 7n,
    },
    turns: [PLAIN_TURN],
    replies: [hello],
  },
  connection_settings: { headers: { "X-Gateway": "gw-1" }, turns: [PLAIN_TURN], replies: [hello] },
  custom_base_url: { baseUrl: "https://gateway.example.com/gemini/v1beta", turns: [PLAIN_TURN], replies: [hello] },

  error_429_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [{ kind: "rate_limit", partial: null, retryIn: 0.5 }],
  },
  error_503_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [{ kind: "overloaded", partial: null, retryIn: 0.5 }],
  },
  error_400_context: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "context_length",
      retryable: false,
      message:
        "google returned HTTP 400 (context_length): The input token count (1048577) exceeds the maximum number " +
        "of tokens allowed (1048576).",
    },
  },
  error_400_api_key_invalid: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "auth",
      retryable: false,
      message: "google returned HTTP 400 (auth): API key not valid. Please pass a valid API key.",
    },
  },
  error_401: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "auth",
      retryable: false,
      message: "google returned HTTP 401 (auth): Request had invalid authentication credentials.",
    },
  },
  redirect_307: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "redirect",
      retryable: false,
      message: "google returned HTTP 307 (redirect): redirect to https://evil.example",
    },
  },
  stream_error_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [{ kind: "overloaded", partial: "Hello", retryIn: 0.5 }],
  },
  truncated_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [{ kind: "incomplete", partial: "Hello there.", retryIn: 0.5 }],
  },
  malformed_response_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [hello],
    retried: [{ kind: "incomplete", partial: "Hello there.", retryIn: 0.5 }],
  },
  safety: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "refused", retryable: false, message: "google response did not complete (SAFETY)" },
  },
  prompt_blocked: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "refused", retryable: false, message: "google blocked the prompt (SAFETY)" },
  },
  max_tokens_stop: { turns: [TOOLS_TURN], replies: [{ ...helloCall, finish: "length" }] },
  malformed_function_call: {
    retries: true,
    turns: [TOOLS_TURN],
    replies: [],
    error: {
      kind: "incomplete",
      retryable: false,
      message: "google response did not complete (MALFORMED_FUNCTION_CALL)",
    },
  },
  continuation: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "protocol",
      retryable: false,
      message: "google asked to continue the reply in another request, which yian does not send",
    },
  },
  invalid_event_json: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "protocol", retryable: false, message: "invalid JSON in google stream event" },
  },
  usage_partial: {
    turns: [PLAIN_TURN],
    replies: [text("Hello there.", { usage: new Usage({ inputTokens: 12, outputTokens: 40, totalTokens: 52 }) })],
  },
  replay_degraded: { turns: [toolsTurn2(pro3())], replies: [answer6], degraded: ["other_model"] },
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
  expect(message.providerPayload).toEqual(replayRecord(key, { response_id: "resp-1", response_model: key.model }));
  expect(response.usage).toEqual(reply.usage ?? USAGE);
  expect(response.finishReason).toBe(reply.finish ?? "stop");
  expect(response.responseId).toBe("resp-1");
}

async function runScenario(name: string, scenario: Scenario): Promise<void> {
  const cassette = loadCassette(join(DIR, `${name}.json`));
  const replay = new CassetteReplay(cassette);
  const modelName = scenario.model ?? FLASH;
  const degraded: ReplayDegradedReason[] = [];
  const protocol = new Gemini({
    model: modelName,
    provider: PROVIDER,
    block: scenario.block,
    onReplayDegraded: (reason) => degraded.push(reason),
  });
  const model = new RouteModel({
    route: route(modelName, scenario.baseUrl ?? BASE_URL, scenario.headers ?? {}),
    protocol,
    send: replay.send,
  });
  expect(model.provider).toBe(protocol.provider);
  const key: ReplayKey = { protocol: PROTOCOL, provider: PROVIDER, model: modelName, account: null };

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
    partial: retry.partial === null ? null : retry.partial.text(),
    retryIn: retry.retryIn,
  }));
  expect(seen).toEqual(scenario.retried ?? []);

  // CassetteReplay checks that every recorded header was sent; this checks the reverse.
  replay.requests.forEach((received, index) => {
    const recorded = Object.keys(cassette.interactions[index]!.request.headers ?? {}).map((each) =>
      each.toLowerCase(),
    );
    const sent = [...received.headers.keys()].filter((each) => !IGNORED_HEADERS.has(each));
    expect(sent.filter((each) => !recorded.includes(each))).toEqual([]);
  });
}

describe("Gemini cassettes", () => {
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
