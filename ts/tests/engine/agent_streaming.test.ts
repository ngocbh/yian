/**
 * AGENT reply streaming, visible chat text reaching `emitStream` as it arrives: mirrors
 * `tests/engine/test_agent_streaming.py`.
 *
 * Ported: `test_stream_chunk_exchange_id_defaults_to_none` (with `attempt`).
 *
 * Ported at the `callModel` level and through the node (`evalNode` with a `ScriptedModel` from
 * `models`): `test_plain_agent_streams_text_chunks_then_one_final_chunk` (each chunk also carries
 * `attempt`), `test_stream_sink_alone_streams_without_observers`,
 * `test_stream_sink_failure_escapes_as_a_host_fault` (at the `callModel` level the sink throws a
 * `CapabilityFault`, as the evaluator's sink does; through the node the host's own error escapes)
 * and `test_mid_stream_failure_sends_no_final_chunk_and_fails_the_node`.
 *
 * Changed: in `test_mid_stream_failure_sends_no_final_chunk_and_fails_the_node` the dropped
 * connection is a `ProviderError` of kind `connection`, so `ModelCallFinished.error` reads
 * `ProviderError` where Python's reads `ConnectionError`.
 * `test_reply_that_fails_normalization_sends_no_final_chunk` (at the `callModel` level and through
 * the node) is a reply the assembler refuses (a stream without its `Finish`, a `ProviderError` of
 * kind `incomplete`); a `ScriptedModel` stands in for Python's scripted LangChain chat.
 * `test_refused_credentials_with_both_sinks_fail_provider_auth`: its node half, through `evalNode`
 * rather than `run_flow` with session memory (which only rides along), for the before-any-text and
 * mid-stream refusals (a `ProviderError` of kind `auth`, reported as `ProviderError`); its
 * client-without-stream row is not ported (every `ChatModel` streams); the `run_flow` half Later
 * (T4-8/T3-22).
 *
 * Later (T5-11b), the tool loop: `test_tool_calling_agent_streams_each_exchange_between_its_observers`
 * and `test_two_streamed_tool_calls_aggregate_by_index`.
 *
 * Later (T6-6), session memory: `test_streamed_tool_exchange_is_retained_and_restored_like_an_invoked_one`
 * and `test_mid_stream_failure_in_a_session_turn_retains_a_failed_model_effect`.
 *
 * Not ported: `test_duck_typed_client_without_stream_emits_one_chunk` (every `ChatModel` streams;
 * there is no duck-typed client or LangChain normalization).
 *
 * TS-only: structured purposes stream nothing; a retried attempt's chunks carry `attempt: 2`;
 * empty text deltas send no chunk.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallFinished, type ModelCallPurpose, ModelCallStarted, StreamChunk, TokenUsage } from "../../src/events.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { Finish, type Event, TextDelta, UsageReport } from "../../src/llm/events.ts";
import { Message, Request, TextPart, Usage } from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import { callModel, liveSinks } from "../../src/nodes/agent/live.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CapabilityFault, type Output } from "../../src/nodes/base.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { drive, rejected } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };
const REQUEST = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("hi")] })] });
const FINISH = new Finish({ reason: "stop" });

/** The events of a text reply streamed as `deltas`. */
function streamed(...deltas: string[]): Event[] {
  return [...deltas.map((text) => new TextDelta(0, text)), FINISH];
}

function sinks(options: { events?: boolean } = {}) {
  const events: unknown[] = [];
  const chunks: StreamChunk[] = [];
  const live = liveSinks({
    nodeId: "writer",
    llmConfig: CONFIG,
    emitEvent: options.events === false ? undefined : (event) => void events.push(event),
    emitStream: (chunk) => void chunks.push(chunk),
  });
  return { live, events, chunks };
}

function model(...replies: ScriptedReply[]): ScriptedModel {
  return new ScriptedModel(replies, CONFIG);
}

const chunk = (text: string, attempt = 1, final = false) =>
  new StreamChunk({ nodeId: "writer", key: "text", chunk: text, final, exchangeId: "x1", attempt });

test("stream chunk exchangeId and attempt default to null", () => {
  expect(new StreamChunk({ nodeId: "n", key: "text", chunk: "hi" }).exchangeId).toBeNull();
  expect(new StreamChunk({ nodeId: "n", key: "text", chunk: "hi" }).attempt).toBeNull();
  expect(new StreamChunk({ nodeId: "n", key: "text", chunk: "hi", exchangeId: "x" }).exchangeId).toBe("x");
});

test("plain chat streams text chunks then one final chunk", async () => {
  const { live, events, chunks } = sinks();
  const usage = new Usage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 });
  const replyEvents = [...streamed("hello", " big", " world").slice(0, -1), new UsageReport(usage), FINISH];
  const response = await callModel(live, model(replyEvents), REQUEST, { purpose: "chat", exchangeId: "x1" });
  expect(response.message.text()).toBe("hello big world");
  expect(chunks).toEqual([chunk("hello"), chunk(" big"), chunk(" world"), chunk("", 1, true)]);
  const identity = { nodeId: "writer", exchangeId: "x1", purpose: "chat" as const, ...CONFIG, attempt: 1 };
  expect(events).toEqual([
    new ModelCallStarted(identity),
    new ModelCallFinished({ ...identity, usage: new TokenUsage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 }) }),
  ]);
});

test("stream sink alone streams without observers", async () => {
  const { live, events, chunks } = sinks({ events: false });
  await callModel(live, model(streamed("just", " text")), REQUEST, { purpose: "chat", exchangeId: "x1" });
  expect(events).toEqual([]);
  expect(chunks.map((item) => [item.chunk, item.final])).toEqual([
    ["just", false],
    [" text", false],
    ["", true],
  ]);
});

test("structured purposes stream nothing", async () => {
  for (const purpose of ["structured_native", "structured_fallback"] satisfies ModelCallPurpose[]) {
    const { live, events, chunks } = sinks();
    await callModel(live, model(streamed('{"value": ', "3}")), REQUEST, { purpose, exchangeId: "x1" });
    expect(chunks).toEqual([]);
    expect(events.map((event) => (event as ModelCallStarted).purpose)).toEqual([purpose, purpose]);
  }
});

test("mid-stream failure sends no final chunk", async () => {
  const { live, events, chunks } = sinks();
  const cut = [new TextDelta(0, "partial"), new TextDelta(0, " answer"), new ProviderError("connection", "reset by sk-hidden")];
  await callModel(live, model(cut), REQUEST, { purpose: "chat", exchangeId: "x1" }).catch(() => {});
  expect(chunks).toEqual([chunk("partial"), chunk(" answer")]);
  expect(events.at(-1)).toEqual(
    new ModelCallFinished({ nodeId: "writer", exchangeId: "x1", purpose: "chat", ...CONFIG, error: "ProviderError", attempt: 1 }),
  );
  expect(JSON.stringify(events)).not.toContain("sk-hidden");
});

test("a reply the assembler refuses sends no final chunk", async () => {
  const { live, events, chunks } = sinks();
  const unfinished = [new TextDelta(0, "hello"), new TextDelta(0, " world")];
  const error = await callModel(live, model(unfinished), REQUEST, { purpose: "chat", exchangeId: "x1" }).catch((caught: unknown) => caught);
  expect((error as ProviderError).kind).toBe("incomplete");
  expect(chunks.map((item) => [item.chunk, item.final])).toEqual([
    ["hello", false],
    [" world", false],
  ]);
  expect((events.at(-1) as ModelCallFinished).error).toBe("ProviderError");
});

test("stream sink failure escapes unchanged, with no finished event", async () => {
  const events: unknown[] = [];
  const fault = new CapabilityFault(new Error("host"));
  const live = liveSinks({
    nodeId: "writer",
    llmConfig: CONFIG,
    emitEvent: (event) => void events.push(event),
    emitStream: () => {
      throw fault;
    },
  });
  const call = callModel(live, model(streamed("text")), REQUEST, { purpose: "chat", exchangeId: "x1" });
  expect(await call.catch((caught: unknown) => caught)).toBe(fault);
  expect(events.map((event) => (event as { type: string }).type)).toEqual(["ModelCallStarted"]);
});

test("a retried attempt's chunks carry attempt 2", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const { live, events, chunks } = sinks();
  const cut = [new TextDelta(0, "par"), new ProviderError("server", "boom", { retryAfter: 0.001 })];
  await callModel(live, model(cut, streamed("whole")), REQUEST, { purpose: "chat", exchangeId: "x1" });
  expect(chunks).toEqual([chunk("par"), chunk("whole", 2), chunk("", 2, true)]);
  expect(events.map((event) => [(event as { type: string }).type, (event as ModelCallStarted).attempt])).toEqual([
    ["ModelCallStarted", 1],
    ["ModelCallFinished", 1],
    ["ModelCallStarted", 2],
    ["ModelCallFinished", 2],
  ]);
});

test("empty text deltas send no chunk", async () => {
  const { live, chunks } = sinks({ events: false });
  await callModel(live, model(streamed("", "hi", "")), REQUEST, { purpose: "chat", exchangeId: "x1" });
  expect(chunks).toEqual([chunk("hi"), chunk("", 1, true)]);
});

// ---------- through the node ----------

/** Evaluate an AGENT `writer` over `model` with the given sinks; return its outcome and what they got. */
async function evaluate(model: ScriptedModel, options: { events?: boolean } = {}) {
  const chunks: StreamChunk[] = [];
  const observed: (ModelCallStarted | ModelCallFinished)[] = [];
  const outcome = await drive(new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG }), {
    models: { env: () => undefined, createModel: () => model },
    emitStream: (item) => void chunks.push(item),
    emitEvent: options.events === false ? null : (event) => void observed.push(event as ModelCallStarted),
  });
  return { outcome, chunks, observed };
}

test("a plain agent streams text chunks then one final chunk", async () => {
  const usage = new Usage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 });
  const replyEvents = [...streamed("hello", " big", " world").slice(0, -1), new UsageReport(usage), FINISH];
  const { outcome, chunks, observed } = await evaluate(model(replyEvents));
  expect((outcome as Output).value).toBe("hello big world");
  const exchangeId = observed[0]!.exchangeId;
  const at = (text: string, final = false) =>
    new StreamChunk({ nodeId: "writer", key: "text", chunk: text, final, exchangeId, attempt: 1 });
  expect(chunks).toEqual([at("hello"), at(" big"), at(" world"), at("", true)]);
  const identity = { nodeId: "writer", exchangeId, purpose: "chat" as const, ...CONFIG, attempt: 1 };
  expect(observed).toEqual([
    new ModelCallStarted(identity),
    new ModelCallFinished({ ...identity, usage: new TokenUsage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 }) }),
  ]);
});

test("a stream sink alone streams through the node without observers", async () => {
  const { outcome, chunks, observed } = await evaluate(model(streamed("just", " text")), { events: false });
  expect((outcome as Output).value).toBe("just text");
  expect(observed).toEqual([]);
  expect(chunks.map((item) => [item.chunk, item.final])).toEqual([
    ["just", false],
    [" text", false],
    ["", true],
  ]);
});

test("a stream sink failure escapes the node as the host's error", async () => {
  class HostBroken extends Error {}
  const broken = new HostBroken("host");
  const node = new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG });
  const run = drive(node, {
    models: { env: () => undefined, createModel: () => model(streamed("text")) },
    emitStream: () => {
      throw broken;
    },
  });
  expect(await rejected(run)).toBe(broken);
});

test("a mid-stream failure sends no final chunk and fails the node", async () => {
  const cut = [new TextDelta(0, "partial"), new TextDelta(0, " answer"), new ProviderError("connection", "reset by sk-hidden")];
  const { outcome, chunks, observed } = await evaluate(model(cut));
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const exchangeId = observed[0]!.exchangeId;
  expect(chunks.map((item) => [item.exchangeId, item.chunk, item.final])).toEqual([
    [exchangeId, "partial", false],
    [exchangeId, " answer", false],
  ]);
  expect(observed.at(-1)).toEqual(
    new ModelCallFinished({ nodeId: "writer", exchangeId, purpose: "chat", ...CONFIG, error: "ProviderError", attempt: 1 }),
  );
  expect(JSON.stringify(observed)).not.toContain("sk-hidden");
});

test("a reply the assembler refuses sends no final chunk and fails the node", async () => {
  const { outcome, chunks, observed } = await evaluate(model([new TextDelta(0, "hello"), new TextDelta(0, " world")]));
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  expect(chunks.map((item) => [item.chunk, item.final])).toEqual([
    ["hello", false],
    [" world", false],
  ]);
  expect((observed.at(-1) as ModelCallFinished).error).toBe("ProviderError");
});

test.each([
  ["before any text", () => new ProviderError("auth", "unauthorized", { status: 401 })],
  ["mid-stream", () => [new TextDelta(0, "partial"), new ProviderError("auth", "unauthorized", { status: 401 })]],
])("refused credentials with both sinks fail provider.auth (%s)", async (_name, script) => {
  const { outcome, chunks, observed } = await evaluate(model(script()));
  expect((outcome as EvaluationFailure).failure.code).toBe("provider.auth");
  expect(chunks.some((item) => item.final)).toBe(false);
  expect(observed.filter((event) => event instanceof ModelCallFinished).map((event) => event.error)).toEqual([
    "ProviderError",
  ]);
});
