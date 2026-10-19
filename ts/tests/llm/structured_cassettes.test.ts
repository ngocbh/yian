/**
 * Golden structured-output scenarios, replayed from streamed cassettes in `cassettes/structured/`
 * through `RouteModel` and `callWithRetries`: `planStructured` picks the strategy, the adapter
 * sends `plan.apply(request)` as each cassette records it, and `plan.parse` reads the reply. The
 * cassettes record the requests today's Python yian sends for `with_structured_output`.
 */

import { describe, expect, test } from "bun:test";
import { readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { BearerAuth, SubscriptionAuth } from "../../src/llm/auth.ts";
import type { ErrorKind } from "../../src/llm/errors.ts";
import { callWithRetries, type RetryInfo, RouteModel } from "../../src/llm/executor.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { OpenAIResponses } from "../../src/llm/protocols/openai_responses.ts";
import { Route } from "../../src/llm/route.ts";
import {
  MissingToolCallError,
  planStructured,
  type Strategy,
  StructuredOutputError,
  structuredTarget,
} from "../../src/llm/structured.ts";
import { CassetteReplay, IGNORED_HEADERS, loadCassette } from "../../src/llm/testing.ts";
import { type Send, virtualScheduler } from "../../src/platform.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

const DIR = fileURLToPath(new URL("cassettes/structured/", import.meta.url));

/** Python's `type_to_schema(...).model_json_schema()` for the `corpus:*` fixtures: ints are `bigint`s. */
const SCHEMAS: Record<string, JsonObject> = {
  "corpus:Rec": {
    properties: { answer: { title: "Answer", type: "string" }, score: { title: "Score", type: "integer" } },
    required: ["answer", "score"],
    title: "Record",
    type: "object",
  },
  "corpus:Open": {
    properties: {
      answer: { title: "Answer", type: "string" },
      data: { additionalProperties: true, title: "Data", type: "object" },
    },
    required: ["answer", "data"],
    title: "Record",
    type: "object",
  },
  "corpus:Union": {
    $defs: {
      buy: {
        additionalProperties: false,
        properties: { kind: { const: "buy", title: "Kind", type: "string" }, x: { title: "X", type: "integer" } },
        required: ["kind", "x"],
        title: "buy",
        type: "object",
      },
      sell: {
        additionalProperties: false,
        properties: { kind: { const: "sell", title: "Kind", type: "string" }, n: { title: "N", type: "integer" } },
        required: ["kind", "n"],
        title: "sell",
        type: "object",
      },
    },
    properties: { value: { anyOf: [{ $ref: "#/$defs/buy" }, { $ref: "#/$defs/sell" }], title: "Value" } },
    required: ["value"],
    title: "ScalarWrapper",
    type: "object",
  },
  "corpus:_ReflectionResponse": {
    $defs: {
      _ReflectionAction: {
        additionalProperties: false,
        description: "One closed model-proposed lesson action.",
        properties: {
          action: { enum: ["add", "revise", "remove", "noop"], title: "Action", type: "string" },
          lesson_id: { anyOf: [{ type: "string" }, { type: "null" }], default: null, title: "Lesson Id" },
          text: { anyOf: [{ type: "string" }, { type: "null" }], default: null, title: "Text" },
        },
        required: ["action"],
        title: "_ReflectionAction",
        type: "object",
      },
    },
    additionalProperties: false,
    description: "Strict structured result from one reflective model call.",
    properties: {
      actions: { items: { $ref: "#/$defs/_ReflectionAction" }, minItems: 1n, title: "Actions", type: "array" },
    },
    required: ["actions"],
    title: "_ReflectionResponse",
    type: "object",
  },
};

const BASE_URLS: Record<string, string> = {
  xai: "https://api.x.ai/v1",
  deepseek: "https://api.deepseek.com",
  qwen: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
  minimax: "https://api.minimax.io/v1",
  openrouter: "https://openrouter.ai/api/v1",
  vllm: "http://localhost:8000/v1",
};

const REC = { answer: "x", score: 1n };
const OPEN = { answer: "x", data: { k: 1n } };
const UNION = { value: { kind: "buy", x: 1n } };
const MISSING = {
  error: MissingToolCallError,
  message: "the reply did not call the Record tool; call it with the answer",
};

/** What `plan.parse` gives: a value, or an error's class and message. */
type Expected = { value: JsonObject } | { error: typeof StructuredOutputError; message: string };

interface Scenario {
  /** The route: the OpenAI Responses API with a key, the ChatGPT subscription, or Chat Completions. */
  family: "responses" | "subscription" | "chat";
  provider: string;
  model: string;
  schema: string;
  strategy: Strategy;
  expected: Expected;
  block?: JsonObject;
  /** The retry budget `callWithRetries` gets; 0 when absent. */
  budget?: number;
  /** The kinds of the errors `callWithRetries` retried, in order; none when absent. */
  retried?: ErrorKind[];
}

/** A scenario on `family`'s route to `provider`, `extra` adding its block, budget or retries. */
const scenario = (
  family: Scenario["family"],
  provider: string,
  model: string,
  schema: string,
  strategy: Strategy,
  expected: Expected,
  extra: Partial<Scenario> = {},
): Scenario => ({ family, provider, model, schema, strategy, expected, ...extra });

const SCENARIOS: Record<string, Scenario> = {
  r_forced_closed: scenario("responses", "openai", "gpt-5.5", "corpus:Rec", "forced_tool", { value: REC }),
  // The open schema's tool is sent with `strict: false`.
  r_forced_open: scenario("responses", "openai", "gpt-5.5", "corpus:Open", "forced_tool", { value: OPEN }),
  r_union: scenario("responses", "openai", "gpt-5.5", "corpus:Union", "forced_tool", { value: UNION }),
  // The block's `text.verbosity` is sent and no `text.format` joins it.
  r_block_text: scenario("responses", "openai", "gpt-5.5", "corpus:Rec", "forced_tool", { value: REC }, {
    block: { text: { verbosity: "low" } },
  }),
  r_no_call: scenario("responses", "openai", "gpt-5.5", "corpus:Rec", "forced_tool", MISSING),
  r_two_calls: scenario("responses", "openai", "gpt-5.5", "corpus:Rec", "forced_tool", {
    value: { answer: "first", score: 1n },
  }),
  r_other_tool_first: scenario("responses", "openai", "gpt-5.5", "corpus:Rec", "forced_tool", { value: REC }),
  r_invalid_then_valid: scenario("responses", "openai", "gpt-5.5", "corpus:Rec", "forced_tool", { value: REC }),
  // The arguments come back as sent: the caller validates them.
  r_bad_args: scenario("responses", "openai", "gpt-5.5", "corpus:Rec", "forced_tool", {
    value: { answer: "x", score: "notint" },
  }),
  // The schema's description is the tool's.
  r_reflection: scenario("responses", "openai", "gpt-5.5", "corpus:_ReflectionResponse", "forced_tool", {
    value: { actions: [{ action: "add", lesson_id: null, text: "be terse" }] },
  }),
  sub_forced_closed: scenario("subscription", "openai", "gpt-5.5", "corpus:Rec", "forced_tool", { value: REC }),
  sub_forced_open: scenario("subscription", "openai", "gpt-5.5", "corpus:Open", "forced_tool", { value: OPEN }),
  x_forced_closed: scenario("chat", "xai", "grok-4", "corpus:Rec", "forced_tool", { value: REC }),
  // Chat Completions sends the open schema's tool with no `strict`.
  x_forced_open: scenario("chat", "xai", "grok-4", "corpus:Open", "forced_tool", { value: OPEN }),
  x_union: scenario("chat", "xai", "grok-4", "corpus:Union", "forced_tool", { value: UNION }),
  x_no_call: scenario("chat", "xai", "grok-4", "corpus:Rec", "forced_tool", MISSING),
  // No `tool_choice`, `parallel_tool_calls: false`.
  ds_reasoner_unforced: scenario("chat", "deepseek", "deepseek-reasoner", "corpus:Rec", "unforced_tool", {
    value: REC,
  }),
  ds_chat_forced: scenario("chat", "deepseek", "deepseek-chat", "corpus:Rec", "forced_tool", { value: REC }),
  // MiniMax's route sends `reasoning_split: true`.
  mm_m27_unforced: scenario("chat", "minimax", "MiniMax-M2.7", "corpus:Rec", "unforced_tool", { value: REC }),
  // The model name decides, whatever the route.
  vllm_ds_name: scenario("chat", "vllm", "deepseek-reasoner", "corpus:Rec", "unforced_tool", { value: REC }),
  or_ds_prefixed: scenario("chat", "openrouter", "deepseek/deepseek-r1", "corpus:Rec", "forced_tool", {
    value: REC,
  }),
  // A MiniMax model off MiniMax's route sends no `reasoning_split`.
  qwen_minimax_name: scenario("chat", "qwen", "MiniMax-M2.5", "corpus:Rec", "unforced_tool", { value: REC }),
  // Truncated arguments leave the call unparsed.
  chat_invalid_args: scenario("chat", "xai", "grok-4", "corpus:Rec", "forced_tool", MISSING),
  chat_other_tool_first: scenario("chat", "xai", "grok-4", "corpus:Rec", "forced_tool", { value: REC }),
  // `[1, 2]` is no JSON object.
  chat_nonobject_args: scenario("chat", "xai", "grok-4", "corpus:Rec", "forced_tool", MISSING),
  // The first attempt's stream is cut short; the retry's reply is parsed.
  x_stream_retry: scenario("chat", "xai", "grok-4", "corpus:Rec", "forced_tool", { value: REC }, {
    budget: 1,
    retried: ["incomplete"],
  }),
};

/** Build a scenario's model, as the Responses and Chat cassette tests build theirs. */
function buildModel(scenario: Scenario, send: Send): RouteModel {
  const { family, provider, model, block } = scenario;
  if (family === "chat") {
    const route = new Route({
      provider,
      baseUrl: BASE_URLS[provider]!,
      path: "chat/completions",
      auth: new BearerAuth("test-key-1"),
    });
    return new RouteModel({ route, protocol: new OpenAIChat({ model, provider, block }), send });
  }
  const subscription = family === "subscription";
  const account = { accessToken: "test-token-1", accountId: "test-account-1", residency: null };
  const route = subscription
    ? new Route({
        provider,
        baseUrl: "https://chatgpt.com/backend-api/codex",
        path: "responses",
        auth: new SubscriptionAuth(() => account),
      })
    : new Route({
        provider,
        baseUrl: "https://api.openai.com/v1",
        path: "responses",
        auth: new BearerAuth("test-key-1"),
      });
  return new RouteModel({ route, protocol: new OpenAIResponses({ model, provider, block, subscription }), send });
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

const REQUEST = new Request({
  messages: [
    new Message({ role: "system", parts: [new TextPart("You are terse.")] }),
    new Message({ role: "user", parts: [new TextPart("Answer.")] }),
  ],
});

/** Replay a scenario's cassette: check the plan's strategy, the retries, and what the reply parses to. */
async function runScenario(name: string, scenario: Scenario): Promise<void> {
  const cassette = loadCassette(join(DIR, `${name}.json`));
  const replay = new CassetteReplay(cassette);
  const model = buildModel(scenario, replay.send);

  const plan = planStructured(structuredTarget(model), SCHEMAS[scenario.schema]!);
  expect(plan.strategy).toBe(scenario.strategy);
  const retried: RetryInfo[] = [];
  const response = await callWithRetries(model, plan.apply(REQUEST), {
    budget: scenario.budget ?? 0,
    scheduler: virtualScheduler(),
    random: () => 0,
    onRetry: (retry) => void retried.push(retry),
  });
  replay.assertDone();
  expect(retried.map((retry) => retry.error.kind)).toEqual(scenario.retried ?? []);

  const { expected } = scenario;
  if ("value" in expected) {
    expect(plan.parse(response)).toEqual(expected.value);
  } else {
    const error = thrown(() => plan.parse(response));
    expect(error).toBeInstanceOf(expected.error);
    expect(error.name).toBe(expected.error.name);
    expect(error.message).toBe(expected.message);
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

describe("structured output cassettes", () => {
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
