/**
 * Golden scenarios for `src/llm/protocols/openai_chat.ts`, replayed from streamed cassettes in
 * `cassettes/openai_chat/` through `RouteModel` (and `callWithRetries` for the retry scenarios).
 * Each cassette records the request the adapter must send on Azure OpenAI or one of the
 * OpenAI-compatible vendors (#366), with every call streamed.
 */

import { describe, expect, test } from "bun:test";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { type Auth, BearerAuth, HeaderKeyAuth } from "../../src/llm/auth.ts";
import { type ErrorKind, ProviderError } from "../../src/llm/errors.ts";
import { collect } from "../../src/llm/events.ts";
import { callWithRetries, type RetryInfo, RouteModel } from "../../src/llm/executor.ts";
import { iterSse } from "../../src/llm/framing.ts";
import {
  Message,
  NamedTool,
  type Part,
  ReasoningPart,
  RefusalPart,
  type ReplayDegradedReason,
  Request,
  type Response,
  TextPart,
  ToolCallPart,
  type ToolChoice,
  ToolResultPart,
  ToolSpec,
  Usage,
} from "../../src/llm/messages.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { Route } from "../../src/llm/route.ts";
import { CassetteReplay, IGNORED_HEADERS, loadCassette } from "../../src/llm/testing.ts";
import { virtualScheduler } from "../../src/platform.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

const DIR = fileURLToPath(new URL("cassettes/openai_chat/", import.meta.url));

const BASE_URLS: Record<string, string> = {
  xai: "https://api.x.ai/v1",
  deepseek: "https://api.deepseek.com",
  qwen: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
  "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  glm: "https://api.z.ai/api/paas/v4/",
  "glm-cn": "https://open.bigmodel.cn/api/paas/v4/",
  minimax: "https://api.minimax.io/v1",
  "minimax-cn": "https://api.minimaxi.com/v1",
  openrouter: "https://openrouter.ai/api/v1",
  vllm: "http://localhost:8000/v1",
};
const DEFAULT_MODELS: Record<string, string> = {
  xai: "grok-4",
  deepseek: "deepseek-reasoner",
  qwen: "qwen3-max",
  "qwen-cn": "qwen3-max",
  glm: "glm-4.6",
  "glm-cn": "glm-4.6",
  minimax: "MiniMax-M2.7",
  "minimax-cn": "MiniMax-M2.7",
  openrouter: "anthropic/claude-sonnet-4.5",
  vllm: "served-model",
  azure: "my-deploy",
};

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
const OPEN = new ToolSpec({
  name: "Open",
  description: "",
  parameters: {
    properties: { answer: { type: "string" }, extra: { additionalProperties: true, type: "object" } },
    required: ["answer", "extra"],
    type: "object",
  },
});
const USAGE = new Usage({
  inputTokens: 21,
  outputTokens: 40,
  totalTokens: 61,
  cacheReadTokens: 5,
  reasoningTokens: 30,
});

/**
 * Build a scenario's route: a vendor's base URL with `chat/completions`, or an Azure deployment
 * URL with its api-version, the model naming the deployment.
 */
function route(provider: string, model: string, auth?: Auth, headers?: Record<string, string>): Route {
  if (provider === "azure") {
    return new Route({
      provider,
      baseUrl: "https://res.openai.azure.com",
      path: `openai/deployments/${model}/chat/completions`,
      query: { "api-version": "2025-03-01-preview" },
      auth: auth ?? new HeaderKeyAuth("api-key", "test-key-1"),
      headers,
    });
  }
  return new Route({
    provider,
    baseUrl: BASE_URLS[provider]!,
    path: "chat/completions",
    auth: auth ?? new BearerAuth("test-key-1"),
    headers,
  });
}

const system = (text: string) => new Message({ role: "system", parts: [new TextPart(text)] });
const user = (text: string) => new Message({ role: "user", parts: [new TextPart(text)] });
const toolResult = (callId: string, content: string) =>
  new Message({ role: "tool", parts: [new ToolResultPart({ callId, name: "double", content, isError: false })] });

function request(
  messages: Message[],
  options: { tools?: ToolSpec[]; toolChoice?: ToolChoice; parallelToolCalls?: boolean } = {},
): Request {
  return new Request({ messages, ...options });
}

const PLAIN = [system("You are terse."), user("Say hello.")];
const TOOLS = [system("You are terse."), user("Double 3.")];
const HI = [user("hi")];

/**
 * Parse one recorded reply of a cassette straight from its chunks, for a later turn's prior
 * (replaying it through `CassetteReplay` would need its request to match).
 */
async function priorFrom(name: string, protocol: OpenAIChat, interaction = 0): Promise<Message> {
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
 * the shared `USAGE`, `responseId` to `chatcmpl-1`. `reasoning` and `refusal` are checked when
 * given.
 */
interface Reply {
  text: string;
  parts: string[];
  responseId?: string;
  calls?: Call[];
  finish?: string;
  usage?: Usage | null;
  reasoning?: string;
  refusal?: string;
}

/** A turn's request, built from the earlier turns' replies and the scenario's protocol. */
type Turn = (previous: Response[], protocol: OpenAIChat) => Request | Promise<Request>;

interface Scenario {
  /** Defaults to `xai`. */
  provider?: string;
  /** Defaults to the provider's model in `DEFAULT_MODELS`. */
  model?: string;
  /** Defaults to the provider's key auth. */
  auth?: Auth;
  headers?: Record<string, string>;
  block?: JsonObject;
  turns: Turn[];
  /** One per turn, unless the last turn fails. */
  replies: Reply[];
  /** The `authorization` header every request must send (the cassette compares it redacted). */
  sentAuthorization?: string;
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
const hello = (parts = ["text"], extra: Partial<Reply> = {}): Reply => ({ text: "Hello there.", parts, ...extra });
const thinkingHello = hello(["reasoning", "text"], { reasoning: "Thinking." });
const ok = text("ok");
const itIs6 = text("It is 6.");
const toolCall = (calls: Call[], parts = ["tool_call"], extra: Partial<Reply> = {}): Reply => ({
  text: "",
  parts,
  calls,
  finish: "tool_calls",
  ...extra,
});
const doubleCall = call("call_1", "double", { value: 3n });
const answerCall = call("call_1", "Answer", { answer: "x", score: 1n });
const openCall = call("call_1", "Open", { answer: "x", extra: { k: 1n } });

const PLAIN_TURN: Turn = () => request(PLAIN);
const TOOLS_TURN: Turn = () => request(TOOLS, { tools: [DOUBLE] });
const HI_TURN: Turn = () => request(HI);
const ANSWER_TURN: Turn = () =>
  request([user("Answer.")], { tools: [ANSWER], toolChoice: new NamedTool("Answer"), parallelToolCalls: false });
const OPEN_TURN: Turn = () =>
  request([user("Answer.")], { tools: [OPEN], toolChoice: new NamedTool("Open"), parallelToolCalls: false });

/** The second turn after a tools cassette, its prior parsed by `parser` (the scenario's protocol by default). */
function toolsTurn2(name: string, parser?: OpenAIChat): Turn {
  return async (_, protocol) =>
    request([...TOOLS, await priorFrom(name, parser ?? protocol), toolResult("call_1", "6")], { tools: [DOUBLE] });
}

/** A second turn sending the first turn's reply back, then `followUp`. */
function followUp(messages: Message[], followUpText: string): Turn {
  return ([first]) => request([...messages, first!.message, user(followUpText)]);
}

const SCENARIOS: Record<string, Scenario> = {
  xai_plain: { turns: [PLAIN_TURN], replies: [thinkingHello] },
  xai_user_multipart: {
    turns: [
      () =>
        request([
          new Message({ role: "system", parts: [new TextPart("You are terse."), new TextPart("Answer in one line.")] }),
          new Message({ role: "user", parts: [new TextPart("Say hello."), new TextPart("Be brief.")] }),
        ]),
    ],
    replies: [hello()],
  },
  xai_tools: { turns: [TOOLS_TURN], replies: [toolCall([doubleCall], ["reasoning", "tool_call"])] },
  xai_turn2: { turns: [toolsTurn2("xai_tools")], replies: [itIs6] },
  xai_turn2_mixed_calls: {
    turns: [
      () => {
        const prior = new Message({
          role: "assistant",
          parts: [
            new TextPart("Let me check."),
            new ToolCallPart({ id: "call_2", name: "double", arguments: null, rawArguments: '{"value": 3' }),
            new ToolCallPart({ id: "call_1", name: "double", arguments: { value: 3n, label: "Bùi" } }),
          ],
        });
        const results = new Message({
          role: "tool",
          parts: [
            new ToolResultPart({ callId: "call_2", name: "double", content: "invalid arguments", isError: false }),
            new ToolResultPart({ callId: "call_1", name: "double", content: "6", isError: false }),
          ],
        });
        return request([...TOOLS, prior, results], { tools: [DOUBLE] });
      },
    ],
    replies: [itIs6],
  },
  xai_parallel: {
    turns: [TOOLS_TURN],
    replies: [
      toolCall([doubleCall, call("call_2", "double", { value: 4n })], ["text", "tool_call", "tool_call"], {
        text: "Calling.",
      }),
    ],
  },
  xai_tool_choice_named: { turns: [ANSWER_TURN], replies: [toolCall([answerCall])] },
  xai_block_merge: {
    block: { reasoning_effort: "high", max_completion_tokens: 100n, stop: ["END"], temperature: 0.2 },
    turns: [HI_TURN],
    replies: [ok],
  },
  xai_tool_call_shapes: {
    turns: [TOOLS_TURN],
    replies: [
      toolCall(
        [doubleCall, call("call_2", "double", { value: 4n }), call("call_3", "double", null, '{"value": 5')],
        ["tool_call", "tool_call", "tool_call"],
      ),
    ],
  },
  xai_idless_call: {
    turns: [
      TOOLS_TURN,
      ([first]) => request([...TOOLS, first!.message, toolResult("yian_call_0", "6")], { tools: [DOUBLE] }),
    ],
    replies: [toolCall([call("yian_call_0", "double", { value: 3n })]), itIs6],
  },
  xai_nameless_call: {
    turns: [TOOLS_TURN],
    replies: [],
    error: { kind: "protocol", retryable: false, message: "xai tool call 0 has no name" },
  },
  xai_invalid_event_json: {
    turns: [HI_TURN],
    replies: [],
    error: { kind: "protocol", retryable: false, message: "invalid JSON in xai stream event" },
  },
  connection_settings: { headers: { "x-gateway": "gateway-1", "x-team": "blue" }, turns: [HI_TURN], replies: [ok] },
  error_429_then_200: {
    retries: true,
    turns: [HI_TURN],
    replies: [ok],
    retried: [{ kind: "rate_limit", partial: null, retryIn: 0.5 }],
  },
  error_400: {
    turns: [HI_TURN],
    replies: [],
    error: { kind: "bad_request", retryable: false, message: "xai returned HTTP 400 (bad_request): Invalid value" },
  },
  truncated_then_200: {
    retries: true,
    turns: [HI_TURN],
    replies: [ok],
    retried: [{ kind: "incomplete", partial: "Hello there.", retryIn: 0.5 }],
  },

  deepseek_reasoner: { provider: "deepseek", turns: [PLAIN_TURN], replies: [thinkingHello] },
  deepseek_tools: {
    provider: "deepseek",
    turns: [TOOLS_TURN],
    replies: [toolCall([doubleCall], ["reasoning", "tool_call"], { reasoning: "Need the tool." })],
  },
  deepseek_turn2: { provider: "deepseek", turns: [toolsTurn2("deepseek_tools")], replies: [itIs6] },
  deepseek_turn2_text: {
    provider: "deepseek",
    turns: [
      async (_, protocol) => request([...PLAIN, await priorFrom("deepseek_reasoner", protocol), user("And then?")]),
    ],
    replies: [ok],
  },
  deepseek_chat_plain: {
    provider: "deepseek",
    model: "deepseek-chat",
    turns: [PLAIN_TURN, followUp(PLAIN, "And then?")],
    replies: [hello(), ok],
  },
  deepseek_replay_degraded: {
    provider: "deepseek",
    model: "deepseek-v4-pro",
    turns: [toolsTurn2("deepseek_tools", new OpenAIChat({ model: "deepseek-reasoner", provider: "deepseek" }))],
    replies: [itIs6],
    degraded: ["other_model"],
  },
  deepseek_tool_choice_suppressed: {
    provider: "deepseek",
    turns: [ANSWER_TURN],
    replies: [toolCall([answerCall], ["reasoning", "tool_call"])],
  },
  deepseek_block_merge: {
    provider: "deepseek",
    block: { thinking: { type: "enabled" }, reasoning_effort: "max", max_tokens: 64n, user_id: "u_1" },
    turns: [HI_TURN],
    replies: [ok],
  },
  deepseek_context_overflow: {
    provider: "deepseek",
    turns: [HI_TURN],
    replies: [],
    error: {
      kind: "context_length",
      retryable: false,
      message:
        "deepseek returned HTTP 400 (context_length): This model's maximum context length is 65536 tokens. " +
        "However, you requested 70000 tokens.",
    },
  },

  qwen_plain: { provider: "qwen", turns: [PLAIN_TURN], replies: [thinkingHello] },
  qwen_cn_plain: { provider: "qwen-cn", turns: [PLAIN_TURN], replies: [thinkingHello] },
  qwen_block_merge: {
    provider: "qwen",
    block: {
      enable_thinking: true,
      thinking_budget: 512n,
      search_options: { forced_search: true },
      top_k: 20n,
      max_tokens: 64n,
      temperature: 0.5,
    },
    turns: [HI_TURN],
    replies: [ok],
  },
  qwen_serving_minimax_name: {
    provider: "qwen",
    model: "MiniMax-M2.5",
    turns: [ANSWER_TURN],
    replies: [toolCall([answerCall])],
  },

  glm_plain: { provider: "glm", turns: [PLAIN_TURN], replies: [thinkingHello] },
  glm_cn_plain: { provider: "glm-cn", turns: [PLAIN_TURN], replies: [thinkingHello] },
  glm_block_merge: {
    provider: "glm",
    block: { thinking: { type: "enabled", clear_thinking: false }, do_sample: false, max_tokens: 64n },
    turns: [HI_TURN],
    replies: [ok],
  },

  minimax_m2: { provider: "minimax", turns: [PLAIN_TURN], replies: [thinkingHello] },
  minimax_cn_plain: { provider: "minimax-cn", turns: [PLAIN_TURN], replies: [thinkingHello] },
  minimax_m2_tool_choice_suppressed: { provider: "minimax", turns: [ANSWER_TURN], replies: [toolCall([answerCall])] },
  minimax_text01_plain: { provider: "minimax", model: "MiniMax-Text-01", turns: [PLAIN_TURN], replies: [hello()] },
  minimax_block_merge: {
    provider: "minimax",
    block: { thinking: { type: "adaptive" }, reasoning_effort: "high", max_tokens: 64n },
    turns: [HI_TURN],
    replies: [ok],
  },

  openrouter_plain: { provider: "openrouter", turns: [PLAIN_TURN], replies: [thinkingHello] },
  openrouter_block_merge: {
    provider: "openrouter",
    block: {
      reasoning: { effort: "high" },
      provider: { order: ["anthropic"] },
      models: ["a/b"],
      cache_control: { type: "ephemeral" },
      verbosity: "low",
      top_k: 5n,
      max_tokens: 64n,
    },
    turns: [HI_TURN],
    replies: [ok],
  },
  openrouter_prefixed_tool_choice: {
    provider: "openrouter",
    model: "deepseek/deepseek-r1",
    turns: [ANSWER_TURN],
    replies: [toolCall([answerCall])],
  },
  openrouter_error_chunk: {
    provider: "openrouter",
    retries: true,
    turns: [HI_TURN],
    replies: [ok],
    retried: [{ kind: "server", partial: "ok", retryIn: 0.5 }],
  },

  vllm_plain: { provider: "vllm", turns: [PLAIN_TURN], replies: [thinkingHello] },
  vllm_keyless: {
    provider: "vllm",
    auth: new BearerAuth("EMPTY"),
    sentAuthorization: "Bearer EMPTY",
    turns: [HI_TURN],
    replies: [ok],
  },
  vllm_deepseek_name: {
    provider: "vllm",
    model: "deepseek-reasoner",
    turns: [ANSWER_TURN],
    replies: [toolCall([answerCall])],
  },
  vllm_inline_think: { provider: "vllm", turns: [HI_TURN], replies: [text("<think>\nhidden\n</think>\n\nVisible.")] },
  vllm_list_content: { provider: "vllm", turns: [HI_TURN], replies: [text("Part A.\nPart B.")] },
  vllm_no_usage_chunk: { provider: "vllm", turns: [HI_TURN], replies: [text("ok", { usage: null })] },

  azure_plain: { provider: "azure", turns: [PLAIN_TURN], replies: [hello(["text"], { responseId: "chatcmpl-az1" })] },
  azure_tools: {
    provider: "azure",
    turns: [TOOLS_TURN],
    replies: [toolCall([doubleCall], ["tool_call"], { responseId: "chatcmpl-az1" })],
  },
  azure_turn2: {
    provider: "azure",
    turns: [toolsTurn2("azure_tools")],
    replies: [text("It is 6.", { responseId: "chatcmpl-az2" })],
  },
  azure_ad_token: {
    provider: "azure",
    auth: new BearerAuth("test-ad-token-1"),
    sentAuthorization: "Bearer test-ad-token-1",
    turns: [HI_TURN],
    replies: [ok],
  },
  azure_block_merge: {
    provider: "azure",
    block: {
      max_tokens: 100n,
      reasoning_effort: "low",
      verbosity: "low",
      user_security_context: { application_name: "a" },
    },
    turns: [HI_TURN],
    replies: [ok],
  },
  azure_tool_choice_named: { provider: "azure", turns: [OPEN_TURN], replies: [toolCall([openCall])] },
  azure_tool_choice_gpt4o: { provider: "azure", model: "gpt-4o", turns: [OPEN_TURN], replies: [toolCall([openCall])] },
  azure_refusal: {
    provider: "azure",
    turns: [HI_TURN, followUp(HI, "Why not?")],
    replies: [{ text: "", parts: ["refusal"], refusal: "I can't help." }, ok],
  },
  azure_content_filter: {
    provider: "azure",
    retries: true,
    turns: [HI_TURN],
    replies: [],
    error: { kind: "refused", retryable: false, message: "azure response did not complete (content_filter)" },
  },
  azure_connection_settings: {
    provider: "azure",
    headers: { "x-gateway": "gateway-1" },
    turns: [HI_TURN],
    replies: [ok],
  },
  azure_429_retry_after_ms: {
    provider: "azure",
    retries: true,
    turns: [HI_TURN],
    replies: [ok],
    retried: [{ kind: "rate_limit", partial: null, retryIn: 1.5 }],
  },
};

function partText(response: Response, kind: typeof ReasoningPart | typeof RefusalPart): string {
  return response.message.parts
    .filter((part: Part): part is ReasoningPart | RefusalPart => part instanceof kind)
    .map((part) => part.text)
    .join("");
}

function expectReply(response: Response, reply: Reply): void {
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
  if (reply.reasoning !== undefined) expect(partText(response, ReasoningPart)).toBe(reply.reasoning);
  if (reply.refusal !== undefined) expect(partText(response, RefusalPart)).toBe(reply.refusal);
  expect(response.usage).toEqual(reply.usage === undefined ? USAGE : reply.usage);
  expect(response.finishReason).toBe(reply.finish ?? "stop");
  expect(response.responseId).toBe(reply.responseId ?? "chatcmpl-1");
}

async function runScenario(name: string, scenario: Scenario): Promise<void> {
  const cassette = loadCassette(join(DIR, `${name}.json`));
  const replay = new CassetteReplay(cassette);
  const provider = scenario.provider ?? "xai";
  const modelName = scenario.model ?? DEFAULT_MODELS[provider]!;
  const degraded: ReplayDegradedReason[] = [];
  const protocol = new OpenAIChat({
    model: modelName,
    provider,
    block: scenario.block,
    onReplayDegraded: (reason) => degraded.push(reason),
  });
  const model = new RouteModel({
    route: route(provider, modelName, scenario.auth, scenario.headers),
    protocol,
    send: replay.send,
  });
  expect(model.provider).toBe(protocol.provider);

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
  replay.assertDone();

  expect(responses.length).toBe(scenario.replies.length);
  scenario.replies.forEach((reply, index) => expectReply(responses[index]!, reply));
  expect(degraded).toEqual(scenario.degraded ?? []);
  const seen = retried.map((retry) => ({
    kind: retry.error.kind,
    partial: retry.partial === null ? null : retry.partial.text(),
    retryIn: retry.retryIn,
  }));
  expect(seen).toEqual(scenario.retried ?? []);
  if (scenario.sentAuthorization !== undefined) {
    for (const received of replay.requests) {
      expect(received.headers.get("authorization")).toBe(scenario.sentAuthorization);
    }
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

describe("OpenAI Chat Completions cassettes", () => {
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
