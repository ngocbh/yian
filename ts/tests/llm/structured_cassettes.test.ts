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
import { callWithRetries, RouteModel } from "../../src/llm/executor.ts";
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
}

const responses = (schema: string, expected: Expected, block?: JsonObject): Scenario => ({
  family: "responses",
  provider: "openai",
  model: "gpt-5.5",
  schema,
  strategy: "forced_tool",
  expected,
  block,
});
const chat = (provider: string, model: string, strategy: Strategy, expected: Expected = { value: REC }): Scenario => ({
  family: "chat",
  provider,
  model,
  schema: "corpus:Rec",
  strategy,
  expected,
});

const SCENARIOS: Record<string, Scenario> = {
  r_forced_closed: responses("corpus:Rec", { value: REC }),
  // The open schema's tool is sent with `strict: false`.
  r_forced_open: responses("corpus:Open", { value: OPEN }),
  r_union: responses("corpus:Union", { value: UNION }),
  // The block's `text.verbosity` is sent and no `text.format` joins it.
  r_block_text: responses("corpus:Rec", { value: REC }, { text: { verbosity: "low" } }),
  r_no_call: responses("corpus:Rec", MISSING),
  r_two_calls: responses("corpus:Rec", { value: { answer: "first", score: 1n } }),
  r_other_tool_first: responses("corpus:Rec", { value: REC }),
  r_invalid_then_valid: responses("corpus:Rec", { value: REC }),
  // The arguments come back as sent: the caller validates them.
  r_bad_args: responses("corpus:Rec", { value: { answer: "x", score: "notint" } }),
  // The schema's description is the tool's.
  r_reflection: responses("corpus:_ReflectionResponse", {
    value: { actions: [{ action: "add", lesson_id: null, text: "be terse" }] },
  }),
  sub_forced_closed: { ...responses("corpus:Rec", { value: REC }), family: "subscription" },
  sub_forced_open: { ...responses("corpus:Open", { value: OPEN }), family: "subscription" },
  x_forced_closed: chat("xai", "grok-4", "forced_tool"),
  // Chat Completions sends the open schema's tool with no `strict`.
  x_forced_open: { ...chat("xai", "grok-4", "forced_tool", { value: OPEN }), schema: "corpus:Open" },
  x_union: { ...chat("xai", "grok-4", "forced_tool", { value: UNION }), schema: "corpus:Union" },
  x_no_call: chat("xai", "grok-4", "forced_tool", MISSING),
  // No `tool_choice`, `parallel_tool_calls: false`.
  ds_reasoner_unforced: chat("deepseek", "deepseek-reasoner", "unforced_tool"),
  ds_chat_forced: chat("deepseek", "deepseek-chat", "forced_tool"),
  // MiniMax's route sends `reasoning_split: true`.
  mm_m27_unforced: chat("minimax", "MiniMax-M2.7", "unforced_tool"),
  // The model name decides, whatever the route.
  vllm_ds_name: chat("vllm", "deepseek-reasoner", "unforced_tool"),
  or_ds_prefixed: chat("openrouter", "deepseek/deepseek-r1", "forced_tool"),
  // A MiniMax model off MiniMax's route sends no `reasoning_split`.
  qwen_minimax_name: chat("qwen", "MiniMax-M2.5", "unforced_tool"),
  // Truncated arguments leave the call unparsed.
  chat_invalid_args: chat("xai", "grok-4", "forced_tool", MISSING),
  chat_other_tool_first: chat("xai", "grok-4", "forced_tool"),
  // `[1, 2]` is no JSON object.
  chat_nonobject_args: chat("xai", "grok-4", "forced_tool", MISSING),
  // The first attempt's stream is cut short; the retry's reply is parsed.
  x_stream_retry: { ...chat("xai", "grok-4", "forced_tool"), budget: 1 },
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

/** Return what `fn` throws, or `undefined`. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  return undefined;
}

const REQUEST = new Request({
  messages: [
    new Message({ role: "system", parts: [new TextPart("You are terse.")] }),
    new Message({ role: "user", parts: [new TextPart("Answer.")] }),
  ],
});

describe("structured output cassettes", () => {
  test.each(Object.entries(SCENARIOS))("%s", async (name, scenario) => {
    const cassette = loadCassette(join(DIR, `${name}.json`));
    const replay = new CassetteReplay(cassette);
    const model = buildModel(scenario, replay.send);

    const plan = planStructured(structuredTarget(model), SCHEMAS[scenario.schema]!);
    expect(plan.strategy).toBe(scenario.strategy);
    const response = await callWithRetries(model, plan.apply(REQUEST), {
      budget: scenario.budget ?? 0,
      scheduler: virtualScheduler(),
      random: () => 0,
    });
    replay.assertDone();

    const { expected } = scenario;
    if ("value" in expected) {
      expect(plan.parse(response)).toEqual(expected.value);
    } else {
      const error = thrown(() => plan.parse(response)) as Error;
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
  });

  test("every cassette file belongs to a scenario and loads", () => {
    const names = readdirSync(DIR)
      .filter((file) => file.endsWith(".json"))
      .map((file) => file.slice(0, -".json".length))
      .sort();
    expect(names).toEqual(Object.keys(SCENARIOS).sort());
    for (const name of names) expect(loadCassette(join(DIR, `${name}.json`)).source).toBe("synthetic");
  });
});
