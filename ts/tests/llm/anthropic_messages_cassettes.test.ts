/**
 * Golden scenarios for `src/llm/protocols/anthropic_messages.ts`, replayed from streamed
 * cassettes in `cassettes/anthropic_messages/` through `RouteModel` (and `callWithRetries` for
 * the retry scenarios). Each cassette records the request the adapter must send to the Anthropic
 * Messages API (#366), with every call streamed.
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
import { AnthropicMessages, PROTOCOL } from "../../src/llm/protocols/anthropic_messages.ts";
import { Route } from "../../src/llm/route.ts";
import { CassetteReplay, IGNORED_HEADERS, loadCassette } from "../../src/llm/testing.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { type JsonObject, parseJson } from "../../src/typesys/json.ts";

const DIR = fileURLToPath(new URL("cassettes/anthropic_messages/", import.meta.url));
const PROVIDER = "anthropic";
const SONNET = "claude-sonnet-4-5";
const OPUS_45 = "claude-opus-4-5";
const OPUS_55 = "claude-opus-5-5";

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
// `message_start` carries the input and cache counts, `message_delta` the full usage (or only
// `output_tokens` in `classic_usage`); either way input is 12 + 100 read + 30 created.
const USAGE = new Usage({
  inputTokens: 142,
  outputTokens: 40,
  totalTokens: 182,
  cacheReadTokens: 100,
  cacheCreationTokens: 30,
});
const THINKING_ON: JsonObject = { thinking: { type: "enabled", budget_tokens: 2048n } };
const REQUIRED_SIGNATURE =
  "the last assistant turn came from anthropic/claude-opus-4-5; anthropic needs its signed thinking to continue " +
  "a tool call with thinking on; finish the tool turn on the original model or turn thinking off in the " +
  "anthropic block";

/** Build a scenario's route: the Anthropic API (or `baseUrl`) with `v1/messages` and the API version header. */
function route(baseUrl = "https://api.anthropic.com", headers: Record<string, string> = {}): Route {
  return new Route({
    provider: PROVIDER,
    baseUrl,
    path: "v1/messages",
    auth: new HeaderKeyAuth("x-api-key", "test-key-1"),
    headers: { "anthropic-version": "2023-06-01", ...headers },
  });
}

const system = (text: string) => new Message({ role: "system", parts: [new TextPart(text)] });
const user = (text: string) => new Message({ role: "user", parts: [new TextPart(text)] });
const toolResult = (callId: string, content: string, isError = false) =>
  new Message({ role: "tool", parts: [new ToolResultPart({ callId, name: "double", content, isError })] });

function request(
  messages: Message[],
  options: { tools?: ToolSpec[]; toolChoice?: ToolChoice; parallelToolCalls?: boolean } = {},
): Request {
  return new Request({ messages, ...options });
}

const PLAIN = [system("You are terse."), user("Say hello.")];
const TOOLS = [system("You are terse."), user("Double 3.")];

/**
 * Parse one recorded reply of a cassette straight from its chunks, for a later turn's prior
 * (replaying it through `CassetteReplay` would need its request to match).
 */
async function priorFrom(name: string, protocol: AnthropicMessages, interaction = 0): Promise<Message> {
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
 * What one turn's reply must parse to; `calls` defaults to none, `finish` to `stop`,
 * `responseId` to `msg_1`. `reasoning` (the reasoning parts' joined text) and `records` (each
 * part's replay record fields, `null` for no payload) are checked when given.
 */
interface Reply {
  text: string;
  parts: string[];
  responseId?: string;
  calls?: Call[];
  finish?: string;
  reasoning?: string;
  records?: (JsonObject | null)[];
}

/** A turn's request, built from the earlier turns' replies and the scenario's protocol. */
type Turn = (previous: Response[], protocol: AnthropicMessages) => Request | Promise<Request>;

interface Scenario {
  /** Defaults to `claude-sonnet-4-5`. */
  model?: string;
  /** Defaults to `https://api.anthropic.com`. */
  baseUrl?: string;
  /** Connection headers beside `anthropic-version`. */
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
  /** The `process.emitWarning` texts. */
  warnings?: string[];
}

const call = (id: string, name: string, args: JsonObject | null, rawArguments: string | null = null): Call => ({
  id,
  name,
  arguments: args,
  rawArguments,
});
const text = (value: string, extra: Partial<Reply> = {}): Reply => ({ text: value, parts: ["text"], ...extra });
const hello = text("Hello there.");
const answer6 = text("The answer is 6.", { responseId: "msg_2" });
const doubleCall = call("toolu_01", "double", { value: 3n });
const helloCall: Reply = {
  text: "Hello there.",
  parts: ["text", "tool_call"],
  calls: [doubleCall],
  finish: "tool_calls",
};

const PLAIN_TURN: Turn = () => request(PLAIN);
const TOOLS_TURN: Turn = () => request(TOOLS, { tools: [DOUBLE] });

/** The second turn after a tools cassette, its prior parsed by `parser` (the scenario's protocol by default). */
function toolsTurn2(name: string, parser?: AnthropicMessages, result = toolResult("toolu_01", "6")): Turn {
  return async (_, protocol) =>
    request([...TOOLS, await priorFrom(name, parser ?? protocol), result], { tools: [DOUBLE] });
}

/** The second turn after the first turn's tool call reply, with the tool's result `6`. */
const REPLY_TURN2: Turn = ([first]) =>
  request([...TOOLS, first!.message, toolResult("toolu_01", "6")], { tools: [DOUBLE] });

/** A protocol for `claude-opus-4-5`, whose records a `claude-sonnet-4-5` protocol can't replay. */
const opus45 = () => new AnthropicMessages({ model: OPUS_45, provider: PROVIDER });

// LangChain's warning for a hoisted system message.
const HOIST_WARNING =
  "A non-leading `SystemMessage` was moved to the top-level `system` field and now applies to the entire " +
  "conversation. To send it in place, use a supported model and place it after a human or tool message, either " +
  "last or before an AI message (model: 'claude-sonnet-4-5').";

const SCENARIOS: Record<string, Scenario> = {
  plain: { turns: [PLAIN_TURN], replies: [hello] },
  multitext: { turns: [PLAIN_TURN], replies: [text("Hello there.\nPart B.", { parts: ["text", "text"] })] },
  max_tokens_128000: { model: OPUS_55, turns: [PLAIN_TURN], replies: [hello] },
  max_tokens_fallback: { model: "claude-opus-4-1", turns: [PLAIN_TURN], replies: [hello] },

  tools: { turns: [TOOLS_TURN], replies: [helloCall] },
  turn2: { turns: [toolsTurn2("tools")], replies: [answer6] },
  turn2_tool_error: {
    turns: [toolsTurn2("tools", undefined, toolResult("toolu_01", "error: boom", true))],
    replies: [answer6],
  },
  turn2_text_tool_text: {
    turns: [TOOLS_TURN, REPLY_TURN2],
    replies: [
      { text: "A\nB", parts: ["text", "tool_call", "text"], calls: [doubleCall], finish: "tool_calls" },
      answer6,
    ],
  },

  thinking_tools: {
    block: THINKING_ON,
    turns: [TOOLS_TURN],
    replies: [
      {
        ...helloCall,
        parts: ["reasoning", "reasoning", "text", "tool_call"],
        reasoning: "Let me think.",
        records: [{ signature: "SIG-1" }, { redacted: "REDACTED-DATA-1" }, null, null],
      },
    ],
  },
  thinking_turn2: { block: THINKING_ON, turns: [toolsTurn2("thinking_tools")], replies: [answer6] },
  thinking_two_steps: {
    block: THINKING_ON,
    turns: [
      TOOLS_TURN,
      REPLY_TURN2,
      ([first, second]) =>
        request(
          [...TOOLS, first!.message, toolResult("toolu_01", "6"), second!.message, toolResult("toolu_02", "12")],
          { tools: [DOUBLE] },
        ),
    ],
    replies: [
      {
        text: "",
        parts: ["reasoning", "tool_call"],
        calls: [doubleCall],
        finish: "tool_calls",
        reasoning: "Let me think.",
        records: [{ signature: "SIG-1" }, null],
      },
      {
        text: "",
        parts: ["tool_call"],
        calls: [call("toolu_02", "double", { value: 6n })],
        finish: "tool_calls",
        responseId: "msg_2",
        records: [null],
      },
      text("Done: 12.", { responseId: "msg_3" }),
    ],
  },
  thinking_omitted: {
    model: OPUS_55,
    block: { thinking: { type: "adaptive", display: "omitted" } },
    turns: [TOOLS_TURN, REPLY_TURN2],
    replies: [
      {
        text: "",
        parts: ["reasoning", "tool_call"],
        calls: [doubleCall],
        finish: "tool_calls",
        reasoning: "",
        records: [{ signature: "SIG-2" }, null],
      },
      answer6,
    ],
  },
  replay_degraded: { turns: [toolsTurn2("thinking_tools", opus45())], replies: [answer6], degraded: ["other_model"] },
  signature_required: {
    block: THINKING_ON,
    turns: [toolsTurn2("thinking_tools", opus45())],
    replies: [],
    error: { kind: "bad_request", retryable: false, message: REQUIRED_SIGNATURE },
    degraded: ["other_model"],
  },

  user_merge: {
    turns: [() => request([system("You are terse."), user("Answer."), user("Your reply was not valid; try again.")])],
    replies: [hello],
  },
  foreign_tool_id: {
    turns: [
      () => {
        const prior = new Message({
          role: "assistant",
          parts: [new ToolCallPart({ id: "functions.double:0", name: "double", arguments: { value: 3n } })],
          protocol: "openai_chat",
        });
        return request([...TOOLS, prior, toolResult("functions.double:0", "6")], { tools: [DOUBLE] });
      },
    ],
    replies: [answer6],
  },
  system_hoisted: {
    turns: [() => request([user("Hi."), system("Be brief.")])],
    replies: [hello],
    warnings: [HOIST_WARNING],
  },
  system_in_place: {
    model: OPUS_55,
    turns: [() => request([system("You are terse."), user("Hi."), system("Answer in French.")])],
    replies: [hello],
  },

  block_merge: {
    model: OPUS_45,
    block: {
      ...THINKING_ON,
      max_tokens: 8000n,
      output_config: { effort: "high" },
      cache_control: { type: "ephemeral", ttl: "1h" },
      // Parsed, so `__proto__` is an own key rather than the prototype.
      metadata: parseJson('{"user_id": "user-1", "__proto__": "x"}', "plain") as JsonObject,
      service_tier: "auto",
      stop_sequences: ["STOP"],
      inference_geo: "us",
      top_k: null,
    },
    turns: [PLAIN_TURN],
    replies: [
      { ...hello, parts: ["reasoning", "text"], reasoning: "Let me think.", records: [{ signature: "SIG-1" }, null] },
    ],
  },
  block_sampling: { block: { temperature: 0.5, top_k: 40n }, turns: [PLAIN_TURN], replies: [hello] },
  tool_choice_named: {
    turns: [() => request([user("hi")], { tools: [ANSWER], toolChoice: new NamedTool("Answer") })],
    replies: [
      {
        text: "",
        parts: ["tool_call"],
        calls: [call("toolu_02", "Answer", { answer: "hi", score: 3n })],
        finish: "tool_calls",
      },
    ],
  },
  parallel_off: {
    turns: [() => request(TOOLS, { tools: [DOUBLE], parallelToolCalls: false })],
    replies: [helloCall],
  },
  connection_settings: {
    headers: { "X-Gateway": "team-a", "x-request-source": "yian" },
    turns: [PLAIN_TURN],
    replies: [hello],
  },
  custom_base_url: { baseUrl: "https://gateway.example.com/anthropic", turns: [PLAIN_TURN], replies: [hello] },

  error_529_then_200: {
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
      message: "anthropic returned HTTP 400 (context_length): prompt is too long: 250000 tokens > 200000 maximum",
    },
  },
  error_401: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "auth", retryable: false, message: "anthropic returned HTTP 401 (auth): invalid x-api-key" },
  },
  stream_error_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [text("Hello there.", { responseId: "msg_2" })],
    retried: [{ kind: "overloaded", partial: "Hello there.", retryIn: 0.5 }],
  },
  truncated_then_200: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [text("Hello there.", { responseId: "msg_2" })],
    retried: [{ kind: "incomplete", partial: "Hello there.", retryIn: 0.5 }],
  },
  refusal: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "refused", retryable: false, message: "anthropic response did not complete (refusal)" },
  },
  max_tokens_stop: { turns: [TOOLS_TURN], replies: [{ ...helloCall, finish: "length" }] },
  pause_turn: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "incomplete", retryable: false, message: "anthropic response did not complete (pause_turn)" },
  },
  context_window_exceeded: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: {
      kind: "context_length",
      retryable: false,
      message: "anthropic response did not complete (model_context_window_exceeded)",
    },
  },
  invalid_event_json: {
    retries: true,
    turns: [PLAIN_TURN],
    replies: [],
    error: { kind: "protocol", retryable: false, message: "invalid JSON in anthropic stream event" },
  },
  truncated_tool_json: {
    turns: [TOOLS_TURN],
    replies: [
      {
        text: "",
        parts: ["tool_call"],
        calls: [call("toolu_01", "double", null, '{"value": 3')],
        finish: "tool_calls",
      },
    ],
  },
  classic_usage: { turns: [PLAIN_TURN], replies: [hello] },
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
  if (reply.records !== undefined) {
    const payloads = reply.records.map((fields) => (fields === null ? null : replayRecord(key, fields)));
    expect(message.parts.map((part) => part.providerPayload)).toEqual(payloads);
  }
  const responseId = reply.responseId ?? "msg_1";
  expect(message.protocol).toBe(PROTOCOL);
  expect(message.providerPayload).toEqual(replayRecord(key, { response_id: responseId, response_model: key.model }));
  expect(response.usage).toEqual(USAGE);
  expect(response.finishReason).toBe(reply.finish ?? "stop");
  expect(response.responseId).toBe(responseId);
}

async function runScenario(name: string, scenario: Scenario): Promise<void> {
  const cassette = loadCassette(join(DIR, `${name}.json`));
  const replay = new CassetteReplay(cassette);
  const modelName = scenario.model ?? SONNET;
  const degraded: ReplayDegradedReason[] = [];
  const protocol = new AnthropicMessages({
    model: modelName,
    provider: PROVIDER,
    block: scenario.block,
    onReplayDegraded: (reason) => degraded.push(reason),
  });
  const model = new RouteModel({ route: route(scenario.baseUrl, scenario.headers), protocol, send: replay.send });
  expect(model.provider).toBe(protocol.provider);
  const key: ReplayKey = { protocol: PROTOCOL, provider: PROVIDER, model: modelName };

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
  // Stubbed so a hoisted system message's warning is asserted rather than printed.
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
    expect(emit.mock.calls.map((args) => [...args])).toEqual((scenario.warnings ?? []).map((each) => [each]));
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

describe("Anthropic Messages cassettes", () => {
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
