/**
 * AGENT in-loop observers, model call brackets with token usage: mirrors
 * `tests/engine/test_agent_live_events.py`.
 *
 * Ported at the `callModel` level and through the node (`evalNode` with a `ScriptedModel` from
 * `models`): `test_event_sink_alone_reports_usage_without_streaming` (a started/finished pair
 * with usage and `attempt: 1`), `test_provider_without_usage_reports_none`,
 * `test_model_error_is_reported_by_type_then_fails_the_node` (reported by its `name`, rethrown and
 * marked), `test_refused_credentials_with_an_event_sink_fail_provider_auth` (the error is reported
 * by its `name` and coded `provider.auth`) and `test_event_sink_failure_escapes_as_a_host_fault`
 * (at the `callModel` level the sink throws a `CapabilityFault`, as the evaluator's sink does;
 * through the node the host's own error escapes).
 *
 * Ported through the node: `test_agent_without_sinks_invokes_exactly_as_before` and
 * `test_structured_fallback_reports_usage_and_streams_nothing`.
 *
 * Changed: a `ScriptedModel` stands in for Python's scripted LangChain chat; every call streams on
 * the wire, so there is no invoke/stream mode to check.
 * `test_agent_run_context_receives_only_supplied_sinks` can't replace `runAgent` (an ESM export),
 * so it runs the node with each sink combination and checks that only the supplied sinks receive
 * anything. `test_structured_native_retries_the_same_with_or_without_an_event_sink` uses the
 * native `json_schema` plan, so its replies are text that doesn't parse, a value that fails the
 * check, then a valid value; the check runs after the call, so no finished event names an error
 * (Python's second reads `ValidationError`).
 *
 * Later (T5-11b), the tool loop: `test_agent_with_tools_orders_model_and_tool_observers`,
 * `test_tool_error_is_reported_by_type_then_fails_the_node` and
 * `test_two_tool_calls_in_one_reply_get_their_own_indexed_brackets`.
 *
 * Later (T6-6), session memory: `test_history_summary_call_is_observed_through_the_wrapper`,
 * `test_history_summary_calls_publish_their_own_observers` and
 * `test_structured_native_turn_reports_usage_and_retains_the_raw_reply`.
 *
 * TS-only: `liveSinks` (`null` without sinks; provider and model only when strings); `usageOf`
 * with every count; `newExchangeId`; a call without sinks still marks its error; a retryable
 * `ProviderError` then a reply gives one started/finished pair per attempt in one exchange; a stop
 * emits nothing more and rethrows the reason unmarked; `modelFailureCode` for every kind of the
 * mapping, an `AuthError`, a 401 status, a marked non-provider error and an unmarked error;
 * `observeToolCall` emits `ToolCallStarted` (its `arguments` a copy) then `ToolCallFinished` with
 * the result's text and the call's identity, reports a rejecting call by its `name` and rethrows,
 * lets a `CapabilityFault` through with no finished event, emits no finished event once the given
 * signal has aborted (success or failure), and without an event sink emits nothing and returns the
 * result; `observeSkippedToolCall` emits started, then finished with `error: "interrupted"`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import type { FailureCode } from "../../src/diagnostics.ts";
import {
  ModelCallFinished,
  ModelCallStarted,
  type StreamChunk,
  TokenUsage,
  ToolCallFinished,
  ToolCallStarted,
} from "../../src/events.ts";
import { AuthError } from "../../src/llm_clients/auth/common.ts";
import { ERROR_KINDS, type ErrorKind, ProviderError } from "../../src/llm/errors.ts";
import { Finish, TextDelta } from "../../src/llm/events.ts";
import { Message, Request, Response, TextPart, Usage } from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import {
  callModel,
  type LiveSinks,
  liveSinks,
  markModelFailure,
  modelFailureCode,
  newExchangeId,
  observeSkippedToolCall,
  observeToolCall,
  SKIPPED_TOOL_ERROR,
  usageOf,
} from "../../src/nodes/agent/live.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CapabilityFault, type Output } from "../../src/nodes/base.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, rejected } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };
const REQUEST = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("hi")] })] });

function reply(text: string, usage: Usage | null = null): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), usage, finishReason: "stop" });
}

/** Sinks that collect what they receive; `events: false` leaves the event sink out. */
function sinks(options: { events?: boolean; stream?: boolean } = {}) {
  const events: unknown[] = [];
  const chunks: StreamChunk[] = [];
  const live = liveSinks({
    nodeId: "writer",
    llmConfig: CONFIG,
    emitEvent: options.events === false ? undefined : (event) => void events.push(event),
    emitStream: options.stream === false ? undefined : (chunk) => void chunks.push(chunk),
  });
  return { live, events, chunks };
}

function model(...replies: ScriptedReply[]): ScriptedModel {
  return new ScriptedModel(replies, CONFIG);
}

const identity = (exchangeId: string) => ({ nodeId: "writer", exchangeId, purpose: "chat" as const, ...CONFIG });

test("liveSinks is null without sinks and reads provider and model only when strings", () => {
  expect(liveSinks({ nodeId: "n", llmConfig: CONFIG })).toBeNull();
  const emitEvent = () => {};
  expect(liveSinks({ nodeId: "n", llmConfig: CONFIG, emitEvent })).toEqual({
    nodeId: "n",
    provider: "openai",
    model: "gpt-5.5",
    emitEvent,
    emitStream: null,
  });
  const live = liveSinks({ nodeId: "n", llmConfig: { provider: 3, model: null }, emitStream: () => {} });
  expect([live?.provider, live?.model, live?.emitEvent]).toEqual([null, null, null]);
  expect(liveSinks({ nodeId: "n", llmConfig: null, emitEvent })?.provider).toBeNull();
});

test("usageOf projects every count and is null without usage", () => {
  const usage = new Usage({ inputTokens: 7, outputTokens: 2, totalTokens: 9, cacheReadTokens: 4, cacheCreationTokens: 1, reasoningTokens: 1 });
  expect(usageOf(reply("ok", usage))).toEqual(
    new TokenUsage({ inputTokens: 7, outputTokens: 2, totalTokens: 9, cacheReadTokens: 4, cacheCreationTokens: 1, reasoningTokens: 1 }),
  );
  expect(usageOf(reply("ok"))).toBeNull();
});

test("newExchangeId is 32 hex characters, fresh per call", () => {
  const id = newExchangeId();
  expect(id).toMatch(/^[0-9a-f]{32}$/);
  expect(newExchangeId()).not.toBe(id);
});

test("event sink alone reports usage without streaming", async () => {
  const { live, events, chunks } = sinks({ stream: false });
  const usage = new Usage({ inputTokens: 7, outputTokens: 2, totalTokens: 9, cacheReadTokens: 4, reasoningTokens: 1 });
  const response = await callModel(live, model(reply("ok", usage)), REQUEST, { purpose: "chat", exchangeId: "x1" });
  expect(response.message.text()).toBe("ok");
  expect(chunks).toEqual([]);
  expect(events).toEqual([
    new ModelCallStarted({ ...identity("x1"), attempt: 1 }),
    new ModelCallFinished({
      ...identity("x1"),
      usage: new TokenUsage({ inputTokens: 7, outputTokens: 2, totalTokens: 9, cacheReadTokens: 4, reasoningTokens: 1 }),
      attempt: 1,
    }),
  ]);
});

test("provider without usage reports null", async () => {
  const { live, events } = sinks();
  await callModel(live, model(reply("ok")), REQUEST, { purpose: "chat", exchangeId: "x1" });
  const finished = events.at(-1) as ModelCallFinished;
  expect([finished.usage, finished.error]).toEqual([null, null]);
});

test("model error is reported by its name, rethrown and marked", async () => {
  const { live, events, chunks } = sinks();
  const error = new RangeError("key=sk-hidden");
  const call = callModel(live, model(error), REQUEST, { purpose: "chat", exchangeId: "x1" });
  expect(await call.catch((caught: unknown) => caught)).toBe(error);
  expect(chunks).toEqual([]);
  expect(events).toEqual([
    new ModelCallStarted({ ...identity("x1"), attempt: 1 }),
    new ModelCallFinished({ ...identity("x1"), error: "RangeError", attempt: 1 }),
  ]);
  expect(JSON.stringify(events)).not.toContain("sk-hidden");
  expect(modelFailureCode(error)).toBe("node.execute");
});

test("refused credentials are reported by name and coded provider.auth", async () => {
  const { live, events } = sinks({ stream: false });
  const error = new ProviderError("auth", "unauthorized", { status: 401 });
  await callModel(live, model(error), REQUEST, { purpose: "chat", exchangeId: "x1" }).catch(() => {});
  expect((events.at(-1) as ModelCallFinished).error).toBe("ProviderError");
  expect(modelFailureCode(error)).toBe("provider.auth");
});

test("a call without sinks still marks its error", async () => {
  const error = new ProviderError("quota", "out of credit");
  await callModel(null, model(error), REQUEST, { purpose: "chat", exchangeId: "x1" }).catch(() => {});
  expect(modelFailureCode(error)).toBe("provider.quota");
});

test("a retryable provider error then a reply publishes one pair per attempt in one exchange", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const { live, events } = sinks();
  const failure = new ProviderError("server", "boom", { retryAfter: 0.001 });
  const response = await callModel(live, model(failure, reply("ok")), REQUEST, { purpose: "chat", exchangeId: "x1" });
  expect(response.message.text()).toBe("ok");
  expect(events).toEqual([
    new ModelCallStarted({ ...identity("x1"), attempt: 1 }),
    new ModelCallFinished({ ...identity("x1"), error: "ProviderError", attempt: 1 }),
    new ModelCallStarted({ ...identity("x1"), attempt: 2 }),
    new ModelCallFinished({ ...identity("x1"), attempt: 2 }),
  ]);
  expect(modelFailureCode(failure)).toBeNull();
});

test("event sink failure escapes unchanged and unmarked, with no finished event", async () => {
  const events: unknown[] = [];
  const fault = new CapabilityFault(new Error("host"));
  const live: LiveSinks = {
    nodeId: "writer",
    provider: "openai",
    model: "gpt-5.5",
    emitEvent: (event) => {
      events.push(event);
      if (event instanceof ModelCallStarted) throw fault;
    },
    emitStream: null,
  };
  const call = callModel(live, model(reply("text")), REQUEST, { purpose: "chat", exchangeId: "x1" });
  expect(await call.catch((caught: unknown) => caught)).toBe(fault);
  expect(events.map((event) => (event as { type: string }).type)).toEqual(["ModelCallStarted"]);
  expect(modelFailureCode(fault)).toBeNull();
});

test("a stop emits nothing more and rethrows the reason unmarked", async () => {
  const controller = new AbortController();
  const reason = new Error("stopped");
  const events: unknown[] = [];
  const chunks: StreamChunk[] = [];
  const live = liveSinks({
    nodeId: "writer",
    llmConfig: CONFIG,
    emitEvent: (event) => void events.push(event),
    emitStream: (chunk) => {
      chunks.push(chunk);
      controller.abort(reason);
    },
  });
  const replyEvents = [new TextDelta(0, "a"), new TextDelta(0, "b"), new Finish({ reason: "stop" })];
  const call = callModel(live, model(replyEvents), REQUEST, { purpose: "chat", exchangeId: "x1", signal: controller.signal });
  expect(await call.catch((caught: unknown) => caught)).toBe(reason);
  expect(chunks.map((chunk) => chunk.chunk)).toEqual(["a"]);
  expect(events.map((event) => (event as { type: string }).type)).toEqual(["ModelCallStarted"]);
  expect(modelFailureCode(reason)).toBeNull();
});

test("modelFailureCode maps every provider error kind of a marked error", () => {
  const expected: Partial<Record<ErrorKind, FailureCode>> = {
    auth: "provider.auth",
    rate_limit: "provider.rate_limit",
    quota: "provider.quota",
    overloaded: "provider.unavailable",
    server: "provider.unavailable",
    timeout: "provider.unavailable",
    connection: "provider.unavailable",
    incomplete: "provider.unavailable",
    context_length: "provider.context_length",
    refused: "provider.refused",
  };
  for (const kind of ERROR_KINDS) {
    const error = new ProviderError(kind, "failed");
    expect(modelFailureCode(error)).toBeNull();
    markModelFailure(error);
    expect([kind, modelFailureCode(error)]).toEqual([kind, expected[kind] ?? "node.execute"]);
  }
});

test("modelFailureCode gives provider.auth for an auth failure first", () => {
  const missing = new AuthError("no key");
  const status = Object.assign(new Error("denied"), { status: 401 });
  const permission = new ProviderError("permission", "forbidden", { status: 403 });
  for (const error of [missing, status, permission]) {
    markModelFailure(error);
    expect(modelFailureCode(error)).toBe("provider.auth");
  }
  // A content-policy refusal is not an auth failure, whatever its status.
  const refused = new ProviderError("refused", "content policy", { status: 403 });
  markModelFailure(refused);
  expect(modelFailureCode(refused)).toBe("provider.refused");
});

test("modelFailureCode is null for an unmarked error and a thrown primitive", () => {
  expect(modelFailureCode(new AuthError("no key"))).toBeNull();
  markModelFailure("text");
  expect(modelFailureCode("text")).toBeNull();
});

// ---------- tool call brackets ----------

const CALL = { exchangeId: "x1", callIndex: 2, toolCallId: "c1", name: "lookup", arguments: { q: { deep: ["x"] } } };
const toolIdentity = { nodeId: "writer", exchangeId: "x1", callIndex: 2, toolCallId: "c1", name: "lookup" };
const shown = (result: unknown) => `text:${String(result)}`;

test("observeToolCall brackets a call with started, then finished with the result's text", async () => {
  const { live, events } = sinks();
  const result = await observeToolCall(live, async () => 42, CALL, { text: shown });
  expect(result).toBe(42);
  expect(events).toEqual([
    new ToolCallStarted({ ...toolIdentity, arguments: { q: { deep: ["x"] } } }),
    new ToolCallFinished({ ...toolIdentity, result: "text:42" }),
  ]);
  // The started event shows a copy, so the arguments can't change under the host.
  const started = events[0] as ToolCallStarted;
  expect(started.arguments).not.toBe(CALL.arguments);
  expect((started.arguments as typeof CALL.arguments).q).not.toBe(CALL.arguments.q);
});

test("observeToolCall reports a result whose text can't be written as null", async () => {
  const { live, events } = sinks();
  const text = () => {
    throw new TypeError("no text");
  };
  expect(await observeToolCall(live, async () => "r", CALL, { text })).toBe("r");
  expect((events[1] as ToolCallFinished).result).toBeNull();
});

test("observeToolCall reports a rejecting call by its name and rethrows", async () => {
  const { live, events } = sinks();
  const error = new RangeError("secret=sk-hidden");
  const call = observeToolCall(live, () => Promise.reject(error), CALL, { text: shown });
  expect(await call.catch((caught: unknown) => caught)).toBe(error);
  expect(events).toEqual([
    new ToolCallStarted({ ...toolIdentity, arguments: { q: { deep: ["x"] } } }),
    new ToolCallFinished({ ...toolIdentity, error: "RangeError" }),
  ]);
  expect(JSON.stringify(events)).not.toContain("sk-hidden");
  const thrown = observeToolCall(live, () => Promise.reject("text"), CALL, { text: shown });
  expect(await thrown.catch((caught: unknown) => caught)).toBe("text");
  expect((events.at(-1) as ToolCallFinished).error).toBe("string");
});

test("observeToolCall lets a CapabilityFault through with no finished event", async () => {
  const { live, events } = sinks();
  const fault = new CapabilityFault(new Error("host"));
  const call = observeToolCall(live, () => Promise.reject(fault), CALL, { text: shown });
  expect(await call.catch((caught: unknown) => caught)).toBe(fault);
  expect(events.map((event) => (event as { type: string }).type)).toEqual(["ToolCallStarted"]);
});

test("observeToolCall emits no finished event once the signal has aborted", async () => {
  const { live, events } = sinks();
  const stop = new AbortController();
  const done = await observeToolCall(
    live,
    async () => {
      stop.abort(new Error("stopped"));
      return "late";
    },
    CALL,
    { signal: stop.signal, text: shown },
  );
  expect(done).toBe("late");
  const error = new Error("boom");
  const failed = observeToolCall(live, () => Promise.reject(error), CALL, { signal: stop.signal, text: shown });
  expect(await failed.catch((caught: unknown) => caught)).toBe(error);
  expect(events.map((event) => (event as { type: string }).type)).toEqual(["ToolCallStarted", "ToolCallStarted"]);
});

test("observeToolCall without an event sink emits nothing and returns the result", async () => {
  const { live, chunks } = sinks({ events: false });
  expect(await observeToolCall(live, async () => "r", CALL, { text: shown })).toBe("r");
  expect(await observeToolCall(null, async () => "s", CALL, { text: shown })).toBe("s");
  expect(chunks).toEqual([]);
});

test("observeSkippedToolCall emits started, then finished interrupted", () => {
  const { live, events } = sinks();
  observeSkippedToolCall(live, CALL);
  observeSkippedToolCall(null, CALL);
  expect(SKIPPED_TOOL_ERROR).toBe("interrupted");
  expect(events).toEqual([
    new ToolCallStarted({ ...toolIdentity, arguments: { q: { deep: ["x"] } } }),
    new ToolCallFinished({ ...toolIdentity, error: "interrupted" }),
  ]);
});

// ---------- through the node ----------

const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
const INT = Type.scalar(ValueKind.INTEGER);

/**
 * Evaluate an AGENT `writer` over `chat` with the sinks asked for (both by default); return its
 * outcome and what the sinks got.
 */
async function evaluate(
  chat: ScriptedModel,
  options: { stream?: boolean; events?: boolean; output?: Type; retries?: number } = {},
) {
  const { stream = true, events = true, output = null, retries } = options;
  const node = new AgentNode("writer", { prompt: "hi", llmConfig: { provider: chat.provider, model: chat.model }, retries });
  if (output !== null) node.outputContract = new ValueContract(output);
  const chunks: StreamChunk[] = [];
  const observed: (ModelCallStarted | ModelCallFinished)[] = [];
  const outcome = await drive(node, {
    models: { env: () => undefined, createModel: () => chat },
    emitStream: stream ? (item) => void chunks.push(item) : null,
    emitEvent: events ? (event) => void observed.push(event as ModelCallStarted | ModelCallFinished) : null,
  });
  return { outcome, chunks, observed };
}

/** The last observed event, which must be a `ModelCallFinished`. */
function lastFinished(observed: readonly (ModelCallStarted | ModelCallFinished)[]): ModelCallFinished {
  const event = observed.at(-1);
  expect(event).toBeInstanceOf(ModelCallFinished);
  return event as ModelCallFinished;
}

test("an AGENT's sinks reach it only when supplied", async () => {
  const seen: [boolean, boolean][] = [];
  for (const [stream, events] of [
    [true, true],
    [false, true],
    [false, false],
  ] as const) {
    const { chunks, observed } = await evaluate(model(reply("ok")), { stream, events });
    seen.push([chunks.length > 0, observed.length > 0]);
  }
  expect(seen).toEqual([
    [true, true],
    [false, true],
    [false, false],
  ]);
});

test("an AGENT without sinks answers as with them", async () => {
  const chat = model(reply("same answer"));
  const { outcome, chunks, observed } = await evaluate(chat, { stream: false, events: false });
  expect((outcome as Output).value).toBe("same answer");
  expect(chat.requests).toHaveLength(1);
  expect([chunks, observed]).toEqual([[], []]);
});

test("an AGENT's event sink alone reports usage without streaming", async () => {
  const usage = new Usage({ inputTokens: 7, outputTokens: 2, totalTokens: 9, cacheReadTokens: 4, reasoningTokens: 1 });
  const { outcome, chunks, observed } = await evaluate(model(reply("ok", usage)), { stream: false });
  expect((outcome as Output).value).toBe("ok");
  expect(chunks).toEqual([]);
  expect(lastFinished(observed).usage).toEqual(
    new TokenUsage({ inputTokens: 7, outputTokens: 2, totalTokens: 9, cacheReadTokens: 4, reasoningTokens: 1 }),
  );
});

test("an AGENT whose provider reports no usage reports null", async () => {
  const { observed } = await evaluate(model(reply("ok")));
  expect([lastFinished(observed).usage, lastFinished(observed).error]).toEqual([null, null]);
});

test("an AGENT's model error is reported by its name, then fails the node", async () => {
  const { outcome, chunks, observed } = await evaluate(model(new RangeError("key=sk-hidden")));
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  expect(chunks).toEqual([]);
  expect([lastFinished(observed).error, lastFinished(observed).usage]).toEqual(["RangeError", null]);
  expect(JSON.stringify(observed)).not.toContain("sk-hidden");
});

test("an AGENT's refused credentials with an event sink fail provider.auth", async () => {
  const chat = model(new ProviderError("auth", "unauthorized", { status: 401 }));
  const { outcome, chunks, observed } = await evaluate(chat, { stream: false });
  expect((outcome as EvaluationFailure).failure.code).toBe("provider.auth");
  expect(chunks).toEqual([]);
  expect(lastFinished(observed).error).toBe("ProviderError");
});

test("an AGENT's event sink failure escapes the node as the host's error", async () => {
  class HostBroken extends Error {}
  const broken = new HostBroken("host");
  const run = drive(new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG }), {
    models: { env: () => undefined, createModel: () => model(reply("text")) },
    emitEvent: () => {
      throw broken;
    },
  });
  expect(await rejected(run)).toBe(broken);
});

test("the structured fallback reports usage and streams nothing", async () => {
  const usage = new Usage({ inputTokens: 9, outputTokens: 1, totalTokens: 10 });
  const chat = new ScriptedModel([reply('{"value": 3}', usage)], PROMPTED);
  const { outcome, chunks, observed } = await evaluate(chat, { output: INT });
  expect((outcome as Output).value).toBe(3n);
  expect(chunks).toEqual([]);
  expect(observed.map((event) => event.constructor)).toEqual([ModelCallStarted, ModelCallFinished]);
  expect(new Set(observed.map((event) => event.purpose))).toEqual(new Set(["structured_fallback"]));
  expect(lastFinished(observed).usage).toEqual(new TokenUsage({ inputTokens: 9, outputTokens: 1, totalTokens: 10 }));
});

test.each([false, true])("structured native retries the same with or without an event sink (events: %p)", async (events) => {
  const usage = (input: number, output: number) =>
    new Usage({ inputTokens: input, outputTokens: output, totalTokens: input + output });
  const chat = model(reply("no json", usage(5, 1)), reply('{"value": "four"}', usage(6, 1)), reply('{"value": 4}', usage(7, 2)));
  const { outcome, chunks, observed } = await evaluate(chat, { stream: false, events, output: INT, retries: 2 });
  expect((outcome as Output).value).toBe(4n);
  expect(chat.requests).toHaveLength(3);
  expect(chunks).toEqual([]);
  const finished = observed.filter((event) => event instanceof ModelCallFinished);
  const tokens = (input: number, output: number) =>
    new TokenUsage({ inputTokens: input, outputTokens: output, totalTokens: input + output });
  expect(finished.map((event) => [event.purpose, event.usage, event.error])).toEqual(
    events
      ? [
          ["structured_native", tokens(5, 1), null],
          ["structured_native", tokens(6, 1), null],
          ["structured_native", tokens(7, 2), null],
        ]
      : [],
  );
});
