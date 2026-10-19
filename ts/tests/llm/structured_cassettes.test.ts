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

import { BearerAuth, HeaderKeyAuth, NoAuth, SubscriptionAuth } from "../../src/llm/auth.ts";
import { type ErrorKind, ProviderError } from "../../src/llm/errors.ts";
import { callWithRetries, type RetryInfo, RouteModel } from "../../src/llm/executor.ts";
import { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { AnthropicMessages } from "../../src/llm/protocols/anthropic_messages.ts";
import { Gemini } from "../../src/llm/protocols/gemini.ts";
import { OllamaChat } from "../../src/llm/protocols/ollama_chat.ts";
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
  "corpus:Bounded": {
    properties: {
      answer: { title: "Answer", type: "string" },
      score: { maximum: 10n, minimum: 0n, title: "Score", type: "integer" },
    },
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
const THINKING: JsonObject = { thinking: { type: "adaptive" }, output_config: { effort: "high" } };
const REFUSED = { error: StructuredOutputError, message: "the model refused: I can't help with that." };
const INVALID = { error: StructuredOutputError, message: "Invalid json output: not json" };
const TRUNCATED = { error: StructuredOutputError, message: 'Invalid json output: {"answer": "x", "score": 1' };

/**
 * What `plan.parse` gives, a value or an error's class and message; or the kind of the
 * `ProviderError` the call itself rejects with, leaving nothing to parse.
 */
type Expected =
  | { value: JsonObject }
  | { error: typeof StructuredOutputError; message: string }
  | { providerError: ErrorKind };

interface Scenario {
  /**
   * The wire: the Responses API with a key (OpenAI's or Azure's), the ChatGPT subscription, or
   * Chat Completions (a vendor's or an Azure deployment's), Anthropic's Messages API, Ollama's chat
   * API, or the Gemini API.
   */
  family: "responses" | "subscription" | "chat" | "anthropic" | "ollama" | "gemini";
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
  az_strict_closed: scenario("chat", "azure", "my-deploy", "corpus:Rec", "json_schema", { value: REC }),
  // The bounds move into the description.
  az_strict_moved: scenario("chat", "azure", "my-deploy", "corpus:Bounded", "json_schema", { value: REC }),
  // `minItems` moves into the description; the schema's title names the format.
  az_reflection_min_items: scenario("chat", "azure", "my-deploy", "corpus:_ReflectionResponse", "json_schema", {
    value: { actions: [{ action: "add", lesson_id: null, text: "be terse" }] },
  }),
  az_union: scenario("chat", "azure", "my-deploy", "corpus:Union", "json_schema", { value: UNION }),
  // No `parallel_tool_calls`.
  az_open_forced: scenario("chat", "azure", "my-deploy", "corpus:Open", "forced_tool", { value: OPEN }),
  // `parallel_tool_calls: false`.
  az_gpt4o_open: scenario("chat", "azure", "gpt-4o", "corpus:Open", "forced_tool", { value: OPEN }),
  // A model too old for `json_schema` gets the schema as a tool.
  az_legacy_gpt4_turbo: scenario("chat", "azure", "gpt-4-turbo", "corpus:Bounded", "forced_tool", { value: REC }),
  // The block's `temperature` is sent beside the forced tool.
  az_open_block: scenario("chat", "azure", "my-deploy", "corpus:Open", "forced_tool", { value: OPEN }, {
    block: { temperature: 0.2 },
  }),
  az_strict_moved_block: scenario("chat", "azure", "my-deploy", "corpus:Bounded", "json_schema", { value: REC }, {
    block: { temperature: 0.2, max_completion_tokens: 100n },
  }),
  az_refusal: scenario("chat", "azure", "my-deploy", "corpus:Rec", "json_schema", REFUSED),
  // JSON beside a refusal is parsed.
  az_refusal_and_json: scenario("chat", "azure", "my-deploy", "corpus:Rec", "json_schema", { value: REC }),
  // Out of the moved bounds: the caller validates.
  az_out_of_range: scenario("chat", "azure", "my-deploy", "corpus:Bounded", "json_schema", {
    value: { answer: "x", score: 11n },
  }),
  az_unmoved_invalid: scenario("chat", "azure", "my-deploy", "corpus:Rec", "json_schema", {
    value: { answer: "x", score: "notint" },
  }),
  // Finish reason `length`.
  az_truncated: scenario("chat", "azure", "my-deploy", "corpus:Rec", "json_schema", TRUNCATED),
  az_content_filter: scenario("chat", "azure", "my-deploy", "corpus:Rec", "json_schema", {
    providerError: "refused",
  }),
  az_resp_closed: scenario("responses", "azure", "gpt-5-codex", "corpus:Rec", "json_schema", { value: REC }),
  // The tool is sent with `strict: false` and no `parallel_tool_calls`.
  az_resp_open: scenario("responses", "azure", "gpt-5-codex", "corpus:Open", "forced_tool", { value: OPEN }),
  az_resp_moved: scenario("responses", "azure", "gpt-5-codex", "corpus:Bounded", "json_schema", { value: REC }),
  az_resp_refusal: scenario("responses", "azure", "gpt-5-codex", "corpus:Rec", "json_schema", REFUSED),
  // `response.incomplete` with `max_output_tokens`.
  az_resp_incomplete: scenario("responses", "azure", "gpt-5-codex", "corpus:Rec", "json_schema", TRUNCATED),
  // No `disable_parallel_tool_use` beside the forced tool.
  a_forced_closed: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "forced_tool", { value: REC }),
  a_union: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Union", "forced_tool", { value: UNION }),
  a_forced_open: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Open", "forced_tool", {
    value: OPEN,
  }),
  // Another tool's call comes first; the Record call is parsed by name.
  a_forced_other_first: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "forced_tool", {
    value: REC,
  }),
  // `output_config.format` is sent beside the block's `effort`, with no tools.
  a_thinking_closed: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", {
    value: REC,
  }, { block: THINKING }),
  // The block's `max_tokens` replaces the model's.
  a_thinking_enabled_closed: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", {
    value: REC,
  }, { block: { thinking: { type: "enabled", budget_tokens: 2048n }, max_tokens: 4096n } }),
  a_between_tools_closed: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", {
    value: REC,
  }, { block: { thinking: { type: "between_tools" }, output_config: { effort: "high" } } }),
  // Thinking rules out a forced tool, and the open schema rules out a format: no `tool_choice`.
  a_thinking_open: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Open", "unforced_tool", {
    value: OPEN,
  }, { block: THINKING }),
  a_thinking_open_no_call: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Open", "unforced_tool",
    MISSING, { block: THINKING }),
  // Thinking turned off keeps the forced tool; the block's `thinking` is sent as written.
  a_thinking_disabled: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "forced_tool", {
    value: REC,
  }, { block: { thinking: { type: "disabled" } } }),
  // Models that refuse a forced tool get it unforced.
  a_refuse_opus_5_5: scenario("anthropic", "anthropic", "claude-opus-5-5", "corpus:Rec", "unforced_tool", {
    value: REC,
  }),
  // Tools only, no `output_config.format`.
  a_refuse_mythos_5_1: scenario("anthropic", "anthropic", "claude-mythos-5-1", "corpus:Rec", "unforced_tool", {
    value: REC,
  }),
  a_refuse_thinking_closed: scenario("anthropic", "anthropic", "claude-opus-5-5", "corpus:Rec", "json_schema", {
    value: REC,
  }, { block: THINKING }),
  a_no_call_unforced: scenario("anthropic", "anthropic", "claude-opus-5-5", "corpus:Rec", "unforced_tool", MISSING),
  a_other_tool_first: scenario("anthropic", "anthropic", "claude-opus-5-5", "corpus:Rec", "unforced_tool", {
    value: REC,
  }),
  a_json_fenced: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", {
    value: REC,
  }, { block: THINKING }),
  // The fence body runs to the last fence.
  a_json_fence_in_string: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", {
    value: { answer: "```py\nx\n```", score: 1n },
  }, { block: THINKING }),
  a_json_invalid: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", INVALID, {
    block: THINKING,
  }),
  a_json_invalid_value: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", {
    value: { answer: "x", score: "notint" },
  }, { block: THINKING }),
  // Cut short by max_tokens: the call does not fail, the text goes to the parser and is not repaired.
  a_json_truncated: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", TRUNCATED, {
    block: THINKING,
  }),
  a_json_prose_then_fence: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", {
    value: REC,
  }, { block: THINKING }),
  a_json_two_objects: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", {
    error: StructuredOutputError,
    message: 'Invalid json output: {"answer": "a", "score": 1}\n{"answer": "b", "score": 2}',
  }, { block: THINKING }),
  // One object split inside the answer string over two text blocks: the newline join lands in the value.
  a_json_two_text_blocks: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:Rec", "json_schema", {
    value: { answer: "x\n", score: 1n },
  }, { block: { thinking: { type: "adaptive" } } }),
  // The output schema keeps the nullable fields' `{default: null}`.
  a_thinking_reflection: scenario("anthropic", "anthropic", "claude-sonnet-4-5", "corpus:_ReflectionResponse",
    "json_schema", { value: { actions: [{ action: "add", lesson_id: null, text: "be terse" }] } }, {
    block: THINKING,
  }),
  o_format_closed: scenario("ollama", "ollama", "qwen3", "corpus:Rec", "json_schema", { value: REC }),
  o_format_open: scenario("ollama", "ollama", "qwen3", "corpus:Open", "json_schema", { value: OPEN }),
  // `format` is sent beside the block's `think`; the thinking is not parsed.
  o_think_format: scenario("ollama", "ollama", "qwen3", "corpus:Rec", "json_schema", { value: REC }, {
    block: { think: true },
  }),
  o_invalid_json: scenario("ollama", "ollama", "qwen3", "corpus:Rec", "json_schema", INVALID),
  // The schema is sent raw, `$defs`, `$ref` and `const` included.
  o_union: scenario("ollama", "ollama", "qwen3", "corpus:Union", "json_schema", { value: UNION }),
  // `const` is sent as a one-member `enum`.
  g_union_const_enum: scenario("gemini", "google", "gemini-2.5-flash", "corpus:Union", "json_schema", {
    value: UNION,
  }),
  g_closed: scenario("gemini", "google", "gemini-2.5-flash", "corpus:Rec", "json_schema", { value: REC }),
  g_open: scenario("gemini", "google", "gemini-2.5-flash", "corpus:Open", "json_schema", { value: OPEN }),
  // A fenced reply after a thought part; the thought is not parsed.
  g_fenced: scenario("gemini", "google", "gemini-2.5-flash", "corpus:Rec", "json_schema", { value: REC }),
  g_two_parts: scenario("gemini", "google", "gemini-2.5-flash", "corpus:Rec", "json_schema", {
    error: StructuredOutputError,
    message: 'Invalid json output: {"answer": "a", "score": 1}{"answer": "b", "score": 2}',
  }),
  // The adapter merges the unsigned text parts into one object.
  g_split_parts: scenario("gemini", "google", "gemini-2.5-flash", "corpus:Rec", "json_schema", {
    value: { answer: "a", score: 1n },
  }),
  g_invalid: scenario("gemini", "google", "gemini-2.5-flash", "corpus:Rec", "json_schema", INVALID),
  // The block's `generationConfig.maxOutputTokens` is sent beside the JSON schema.
  g_block: scenario("gemini", "google", "gemini-2.5-flash", "corpus:Rec", "json_schema", { value: REC }, {
    block: { generationConfig: { maxOutputTokens: 64n } },
  }),
  // A model with no structured output gets the instruction; its fenced reply is parsed.
  p_sentinel: scenario("chat", "xai", "no-structured-sentinel", "corpus:Rec", "prompt", { value: REC }),
};

/**
 * Build a scenario's route as the Responses, Chat, Anthropic, Ollama and Gemini cassette tests
 * build theirs: Anthropic's Messages API, the ChatGPT subscription's, Azure's on either wire, a
 * Chat vendor's base URL, OpenAI's Responses API, a local Ollama, or the Gemini API.
 */
function route({ family, provider, model }: Scenario): Route {
  const azure = { query: { "api-version": "2025-03-01-preview" }, auth: new HeaderKeyAuth("api-key", "test-key-1") };
  switch (family) {
    case "anthropic":
      return new Route({
        provider,
        baseUrl: "https://api.anthropic.com",
        path: "v1/messages",
        auth: new HeaderKeyAuth("x-api-key", "test-key-1"),
        headers: { "anthropic-version": "2023-06-01" },
      });
    case "subscription": {
      const account = { accessToken: "test-token-1", accountId: "test-account-1", residency: null };
      return new Route({
        provider,
        baseUrl: "https://chatgpt.com/backend-api/codex",
        path: "responses",
        auth: new SubscriptionAuth(() => account),
      });
    }
    case "chat":
      // Azure's Chat Completions names the deployment in the path.
      if (provider === "azure") {
        return new Route({
          provider,
          baseUrl: "https://res.openai.azure.com",
          path: `openai/deployments/${model}/chat/completions`,
          ...azure,
        });
      }
      return new Route({
        provider,
        baseUrl: BASE_URLS[provider]!,
        path: "chat/completions",
        auth: new BearerAuth("test-key-1"),
      });
    case "responses":
      // Azure's Responses API takes the deployment as the model.
      if (provider === "azure") {
        return new Route({ provider, baseUrl: "https://res.openai.azure.com/openai", path: "responses", ...azure });
      }
      return new Route({
        provider,
        baseUrl: "https://api.openai.com/v1",
        path: "responses",
        auth: new BearerAuth("test-key-1"),
      });
    case "ollama":
      return new Route({ provider, baseUrl: "http://localhost:11434", path: "api/chat", auth: new NoAuth() });
    case "gemini":
      return new Route({
        provider,
        baseUrl: "https://generativelanguage.googleapis.com/v1beta",
        path: `models/${model}:streamGenerateContent`,
        query: { alt: "sse" },
        auth: new HeaderKeyAuth("x-goog-api-key", "test-key-1"),
      });
  }
}

/** Build a scenario's model on its route, speaking its family's protocol. */
function buildModel(scenario: Scenario, send: Send): RouteModel {
  const { family, provider, model, block } = scenario;
  const routed = { route: route(scenario), send };
  switch (family) {
    case "anthropic":
      return new RouteModel({ ...routed, protocol: new AnthropicMessages({ model, provider, block }) });
    case "chat":
      return new RouteModel({ ...routed, protocol: new OpenAIChat({ model, provider, block }) });
    case "responses":
    case "subscription": {
      const subscription = family === "subscription";
      return new RouteModel({ ...routed, protocol: new OpenAIResponses({ model, provider, block, subscription }) });
    }
    case "ollama":
      return new RouteModel({ ...routed, protocol: new OllamaChat({ model, provider, block }) });
    case "gemini":
      return new RouteModel({ ...routed, protocol: new Gemini({ model, provider, block }) });
  }
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

/**
 * Replay a scenario's cassette: check the plan's strategy, what the reply parses to (or how the
 * call fails), and the retries.
 */
async function runScenario(name: string, scenario: Scenario): Promise<void> {
  const cassette = loadCassette(join(DIR, `${name}.json`));
  const replay = new CassetteReplay(cassette);
  const model = buildModel(scenario, replay.send);

  const plan = planStructured(structuredTarget(model), SCHEMAS[scenario.schema]!);
  expect(plan.strategy).toBe(scenario.strategy);
  let request = REQUEST;
  if (plan.strategy === "prompt") {
    // The prompt strategy sends the request as it is; the caller appends the instruction.
    const instruction = new Message({ role: "user", parts: [new TextPart(plan.instruction())] });
    request = new Request({ messages: [...REQUEST.messages, instruction] });
    expect(plan.apply(request)).toBe(request);
  }
  const retried: RetryInfo[] = [];
  const call = callWithRetries(model, plan.apply(request), {
    budget: scenario.budget ?? 0,
    scheduler: virtualScheduler(),
    random: () => 0,
    onRetry: (retry) => void retried.push(retry),
  });
  const { expected } = scenario;
  if ("providerError" in expected) {
    await expect(call).rejects.toBeInstanceOf(ProviderError);
    // Not retryable: the call fails at once rather than being corrected under `retries:`.
    await expect(call).rejects.toMatchObject({ kind: expected.providerError, retryable: false });
  } else if ("value" in expected) {
    expect(plan.parse(await call)).toEqual(expected.value);
  } else {
    const response = await call;
    const error = thrown(() => plan.parse(response));
    expect(error).toBeInstanceOf(expected.error);
    expect(error.name).toBe(expected.error.name);
    expect(error.message).toBe(expected.message);
  }
  replay.assertDone();
  expect(retried.map((retry) => retry.error.kind)).toEqual(scenario.retried ?? []);

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
