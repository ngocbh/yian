/**
 * Golden scenarios for `src/llm/protocols/openai_responses.ts`, replayed from streamed cassettes
 * in `cassettes/openai_responses/` through `RouteModel` (and `callWithRetries` for the retry
 * scenarios). Each cassette records the request the adapter must send (#366), with every call
 * streamed.
 */

import { describe, expect, test } from "bun:test";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { BearerAuth, HeaderKeyAuth, SubscriptionAuth } from "../../src/llm/auth.ts";
import { type ErrorKind, ProviderError } from "../../src/llm/errors.ts";
import { collect } from "../../src/llm/events.ts";
import { callWithRetries, type RetryInfo, RouteModel } from "../../src/llm/executor.ts";
import { iterSse } from "../../src/llm/framing.ts";
import {
  Message,
  NamedTool,
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
import { OpenAIResponses } from "../../src/llm/protocols/openai_responses.ts";
import { Route } from "../../src/llm/route.ts";
import { CassetteReplay, IGNORED_HEADERS, loadCassette } from "../../src/llm/testing.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { type JsonObject, parseJson } from "../../src/typesys/json.ts";

const DIR = fileURLToPath(new URL("cassettes/openai_responses/", import.meta.url));

const DOUBLE = new ToolSpec({
  name: "double",
  description: "Double an integer.",
  parameters: { properties: { value: { type: "integer" } }, required: ["value"], type: "object" },
});
const ANSWER = new ToolSpec({
  name: "Answer",
  description: "",
  parameters: {
    properties: { answer: { type: "string" }, score: { type: "integer" } },
    required: ["answer", "score"],
    type: "object",
  },
});
const OPEN_ANSWER = new ToolSpec({
  name: "OpenAnswer",
  description: "",
  parameters: {
    properties: { answer: { type: "string" }, extra: { additionalProperties: true, type: "object" } },
    required: ["answer", "extra"],
    type: "object",
  },
  strict: false,
});
const USAGE = new Usage({
  inputTokens: 21,
  outputTokens: 40,
  totalTokens: 61,
  cacheReadTokens: 5,
  reasoningTokens: 30,
});

function subscription(residency: string | null): Route {
  return new Route({
    provider: "openai",
    baseUrl: "https://chatgpt.com/backend-api/codex",
    path: "responses",
    auth: new SubscriptionAuth(() => ({ accessToken: "test-token-1", accountId: "test-account-1", residency })),
  });
}

function bearer(provider: string, baseUrl: string): Route {
  return new Route({ provider, baseUrl, path: "responses", auth: new BearerAuth("test-key-1") });
}

const system = (text: string) => new Message({ role: "system", parts: [new TextPart(text)] });
const user = (text: string) => new Message({ role: "user", parts: [new TextPart(text)] });
const toolResult = (callId: string, name: string, content: string) =>
  new Message({ role: "tool", parts: [new ToolResultPart({ callId, name, content, isError: false })] });

function request(
  messages: Message[],
  options: { tools?: ToolSpec[]; toolChoice?: ToolChoice; parallelToolCalls?: boolean } = {},
): Request {
  return new Request({ messages, ...options });
}

/**
 * Parse one recorded reply of a cassette straight from its chunks, for a later turn's prior
 * (replaying it through `CassetteReplay` would need its request to match).
 */
async function priorFrom(name: string, protocol: OpenAIResponses, interaction = 0): Promise<Message> {
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

/** What one turn's reply must parse to; `finish` defaults to `stop`, `usage` to the shared `USAGE`. */
interface Reply {
  text: string;
  parts: string[];
  responseId: string;
  calls?: Call[];
  finish?: string;
  usage?: Usage;
}

/** A turn's request, built from the earlier turns' replies and the scenario's protocol. */
type Turn = (previous: Response[], protocol: OpenAIResponses) => Request | Promise<Request>;

interface Scenario {
  /** Defaults to the API key route. */
  route?: Route;
  /** Defaults to `gpt-5.5`. */
  model?: string;
  block?: JsonObject;
  turns: Turn[];
  /** One per turn, unless the last turn fails. */
  replies: Reply[];
  /** The last turn's error. */
  error?: { kind: ErrorKind; retryable: boolean; message: string };
  /** Run each turn through `callWithRetries`. */
  retries?: boolean;
  /** The `onRetry` calls: the error kind and the partial message's text (`null` for no partial). */
  retried?: { kind: ErrorKind; partial: string | null }[];
  /** The `onReplayDegraded` calls. */
  degraded?: ReplayDegradedReason[];
}

const hello = (responseId: string, parts = ["text"]): Reply => ({ text: "Hello there.", parts, responseId });
const doubleCall: Call = { id: "call_1", name: "double", arguments: { value: 3n }, rawArguments: null };
const toolReply: Reply = {
  text: "",
  parts: ["reasoning", "tool_call"],
  responseId: "resp_1",
  calls: [doubleCall],
  finish: "tool_calls",
};
const HI_TURN: Turn = () => request([system("You are terse."), user("Hi")]);
const DOUBLE_TURN: Turn = () => request([system("You are terse."), user("Say hello.")], { tools: [DOUBLE] });

/** The second turn after `tools`, its prior parsed by `parser` (the scenario's protocol by default). */
function turn2(parser?: OpenAIResponses): Turn {
  return async (_, protocol) => {
    // Only `tools`' reply is reused; its turn-1 user text ("Say hello.") differs from this turn's
    // ("Double 3.") because the two requests come from separate captures.
    const prior = await priorFrom("tools", parser ?? protocol);
    const messages = [system("You are terse."), user("Double 3."), prior, toolResult("call_1", "double", "6")];
    return request(messages, { tools: [DOUBLE] });
  };
}

/** The second turn after `multipart`, its prior parsed by the scenario's protocol. */
const MULTIPART_TURN2: Turn = async (_, protocol) =>
  request([user("hi"), await priorFrom("multipart", protocol), user("And then?")]);

const SCENARIOS: Record<string, Scenario> = {
  plain: {
    turns: [() => request([system("You are terse."), user("Say hello.")])],
    replies: [hello("resp_1", ["reasoning", "text"])],
  },
  multipart: {
    turns: [() => request([user("hi")])],
    replies: [
      { text: "Let me check.\nPart A.\nPart B.", parts: ["reasoning", "text", "text", "text"], responseId: "resp_1" },
    ],
  },
  tools: { turns: [DOUBLE_TURN], replies: [toolReply] },
  turn2: { turns: [turn2()], replies: [hello("resp_2")] },
  turn2_multipart_ids: { turns: [MULTIPART_TURN2], replies: [hello("resp_2")] },
  refusal: {
    turns: [
      () => request([user("hi")]),
      ([first]) => request([user("hi"), first!.message, user("Why not?")]),
    ],
    replies: [{ text: "", parts: ["reasoning", "refusal"], responseId: "resp_1" }, hello("resp_2")],
  },
  block_merge: {
    // Parsed, so `"__proto__"` is an own key and the `null` leaf is kept as written.
    block: parseJson(
      `{"reasoning": {"effort": "high", "summary": "auto"}, "text": {"verbosity": "low", "format": null},
        "max_output_tokens": 4096, "temperature": 0.2, "service_tier": "flex", "prompt_cache_key": "k1",
        "metadata": {"a": "b", "__proto__": "x"}}`,
      "plain",
    ) as JsonObject,
    turns: [() => request([system("You are terse."), user("Double 3.")])],
    replies: [
      {
        ...hello("resp_2"),
        // A flex reply keeps its cache and reasoning counts.
        usage: new Usage({
          inputTokens: 21,
          outputTokens: 40,
          totalTokens: 61,
          cacheReadTokens: 5,
          cacheCreationTokens: 2,
          reasoningTokens: 30,
        }),
      },
    ],
  },
  tool_choice_named: {
    turns: [
      () =>
        request([system("You are terse."), user("Double 3.")], {
          tools: [ANSWER],
          toolChoice: new NamedTool("Answer"),
          parallelToolCalls: false,
        }),
    ],
    replies: [
      {
        ...toolReply,
        calls: [{ id: "call_1", name: "Answer", arguments: { answer: "x", score: 2n }, rawArguments: null }],
      },
    ],
  },
  open_schema_strict_false: {
    turns: [
      () =>
        request([system("You are terse."), user("Double 3.")], {
          tools: [OPEN_ANSWER],
          toolChoice: new NamedTool("OpenAnswer"),
          parallelToolCalls: false,
        }),
    ],
    replies: [
      {
        ...toolReply,
        parts: ["tool_call"],
        calls: [{ id: "call_1", name: "OpenAnswer", arguments: { answer: "x", extra: { k: 1n } }, rawArguments: null }],
      },
    ],
  },
  fallback_call_render: {
    turns: [
      () => {
        // No payloads; the call comes first so the test shows it is sent after the message items.
        const prior = new Message({
          role: "assistant",
          parts: [
            new ToolCallPart({ id: "call_x", name: "greet", arguments: { name: "Bùi" } }),
            new TextPart("Let me greet you."),
          ],
        });
        const result = toolResult("call_x", "greet", "Hello, Bùi!");
        return request([system("You are terse."), user("Greet Bùi."), prior, result]);
      },
    ],
    replies: [hello("resp_2")],
  },
  replay_degraded: {
    turns: [turn2(new OpenAIResponses({ model: "gpt-5.4", provider: "openai" }))],
    replies: [hello("resp_2")],
    degraded: ["other_model"],
  },
  subscription_plain: {
    route: subscription("us"),
    turns: [() => request([system("You are terse."), user("Double 3.")])],
    replies: [hello("resp_2", ["reasoning", "text"])],
  },
  subscription_tools: {
    route: subscription(null),
    turns: [() => request([user("Double 3.")], { tools: [DOUBLE] })],
    replies: [toolReply],
  },
  subscription_turn2: {
    route: subscription(null),
    turns: [
      async (_, protocol) => {
        const prior = await priorFrom("subscription_tools", protocol);
        return request([user("Double 3."), prior, toolResult("call_1", "double", "6")], { tools: [DOUBLE] });
      },
    ],
    replies: [hello("resp_2", ["reasoning", "text"])],
  },
  subscription_turn2_multipart: {
    route: subscription(null),
    turns: [MULTIPART_TURN2],
    replies: [hello("resp_2", ["reasoning", "text"])],
  },
  subscription_no_system: {
    route: subscription(null),
    turns: [() => request([user("Double 3.")])],
    replies: [hello("resp_2", ["reasoning", "text"])],
  },
  azure_codex: {
    route: new Route({
      provider: "azure",
      baseUrl: "https://res.openai.azure.com/openai",
      path: "responses",
      query: { "api-version": "2025-03-01-preview" },
      auth: new HeaderKeyAuth("api-key", "test-key-1"),
    }),
    model: "gpt-5-codex",
    turns: [HI_TURN],
    replies: [hello("resp_1")],
  },
  openrouter_codex: {
    route: bearer("openrouter", "https://openrouter.ai/api/v1"),
    model: "openai/gpt-5-codex",
    turns: [HI_TURN],
    replies: [hello("resp_1")],
  },
  vllm_codex: {
    route: bearer("vllm", "http://localhost:8000/v1"),
    model: "my-codex-finetune",
    turns: [HI_TURN],
    replies: [hello("resp_1")],
  },
  connection_settings: {
    route: new Route({
      provider: "openai",
      baseUrl: "https://api.openai.com/v1",
      path: "responses",
      auth: new BearerAuth("test-key-1"),
      headers: { "OpenAI-Organization": "org-test", "OpenAI-Project": "proj-test", "X-Gateway": "gateway-1" },
    }),
    turns: [HI_TURN],
    replies: [hello("resp_1")],
  },
  error_429_then_200: {
    retries: true,
    turns: [HI_TURN],
    replies: [hello("resp_1")],
    retried: [{ kind: "rate_limit", partial: null }],
  },
  error_400: {
    retries: true,
    turns: [HI_TURN],
    replies: [],
    error: {
      kind: "bad_request",
      retryable: false,
      message: "openai returned HTTP 400 (bad_request): Invalid value: 'foo'.",
    },
  },
  retry_after_partial: {
    retries: true,
    turns: [HI_TURN],
    // The result is the second attempt's.
    replies: [hello("resp_2")],
    retried: [{ kind: "server", partial: "Hello there." }],
  },
};

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
  expect(response.usage).toEqual(reply.usage ?? USAGE);
  expect(response.finishReason).toBe(reply.finish ?? "stop");
  expect(response.responseId).toBe(reply.responseId);
}

describe("OpenAI Responses cassettes", () => {
  for (const [name, scenario] of Object.entries(SCENARIOS)) {
    test(name, async () => {
      const cassette = loadCassette(join(DIR, `${name}.json`));
      const replay = new CassetteReplay(cassette);
      const route = scenario.route ?? bearer("openai", "https://api.openai.com/v1");
      const degraded: ReplayDegradedReason[] = [];
      const protocol = new OpenAIResponses({
        model: scenario.model ?? "gpt-5.5",
        provider: route.provider,
        block: scenario.block,
        subscription: route.auth instanceof SubscriptionAuth,
        onReplayDegraded: (reason) => degraded.push(reason),
      });
      const model = new RouteModel({ route, protocol, send: replay.send });
      expect(model.provider).toBe(protocol.provider);

      const retried: RetryInfo[] = [];
      const call = (req: Request): Promise<Response> =>
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
          const failed = call(req);
          await expect(failed).rejects.toBeInstanceOf(ProviderError);
          await expect(failed).rejects.toMatchObject(scenario.error);
          break;
        }
        responses.push(await call(req));
      }
      replay.assertDone();

      expect(responses.length).toBe(scenario.replies.length);
      scenario.replies.forEach((reply, index) => expectReply(responses[index]!, reply));
      expect(degraded).toEqual(scenario.degraded ?? []);
      const seen = retried.map((retry) => ({
        kind: retry.error.kind,
        partial: retry.partial === null ? null : retry.partial.text(),
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
    });
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
