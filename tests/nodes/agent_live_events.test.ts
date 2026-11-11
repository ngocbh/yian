/**
 * AGENT in-loop observers: model call brackets with token usage, tool call brackets, `model_call`
 * steps and `ReplayDegraded`, at the `callModel` level and through the node.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import type { FailureCode } from "../../src/diagnostics.ts";
import {
  ModelCallFinished,
  ModelCallStarted,
  ReplayDegraded,
  RunSucceeded,
  type StreamChunk,
  TokenUsage,
  ToolCallFinished,
  ToolCallStarted,
} from "../../src/events.ts";
import { AuthError } from "../../src/llm_clients/auth/common.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { ERROR_KINDS, type ErrorKind, ProviderError } from "../../src/llm/errors.ts";
import { Finish, TextDelta, UsageReport } from "../../src/llm/events.ts";
import {
  Message,
  type ReplayDegradedReason,
  Request,
  Response,
  TextPart,
  ToolCallPart,
  Usage,
} from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import {
  callModel,
  type LiveSinks,
  liveSinks,
  markModelFailure,
  modelFailureCode,
  observeSkippedToolCall,
  observeToolCall,
  SKIPPED_TOOL_ERROR,
  usageOf,
} from "../../src/nodes/agent/live.ts";
import { ReplyInterrupted } from "../../src/nodes/agent/interrupts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CapabilityFault, type NodeCaps, type Output } from "../../src/nodes/base.ts";
import { NodeSteps, ValueContract } from "../../src/nodes/contracts.ts";
import { type Scheduler, virtualScheduler } from "../../src/platform.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind, type ValueRecord } from "../../src/typesys/values.ts";
import { agentFlow, drive, NO_ENV, rejected, runAgentFlow, serving } from "../helpers/run_state_fixtures.ts";

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

const identity = (n = 1) => ({ nodeId: "writer", n, purpose: "chat" as const, ...CONFIG });

/** A started event's time, which `callModel` reads from the run's clock. */
const ANY_TIME = expect.any(Number) as unknown as number;

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

test("event sink alone reports usage without streaming", async () => {
  const { live, events, chunks } = sinks({ stream: false });
  const usage = new Usage({ inputTokens: 7, outputTokens: 2, totalTokens: 9, cacheReadTokens: 4, reasoningTokens: 1 });
  const { response } = await callModel(live, model(reply("ok", usage)), REQUEST, { purpose: "chat" });
  expect(response.message.text()).toBe("ok");
  expect(chunks).toEqual([]);
  expect(events).toEqual([
    new ModelCallStarted({ ...identity(1), startedAt: ANY_TIME }),
    new ModelCallFinished({ ...identity(1),
      usage: new TokenUsage({ inputTokens: 7, outputTokens: 2, totalTokens: 9, cacheReadTokens: 4, reasoningTokens: 1 }) }),
  ]);
});

test("provider without usage reports null", async () => {
  const { live, events } = sinks();
  await callModel(live, model(reply("ok")), REQUEST, { purpose: "chat" });
  const finished = events.at(-1) as ModelCallFinished;
  expect([finished.usage, finished.error]).toEqual([null, null]);
});

test("model error is reported by its name, rethrown and marked", async () => {
  const { live, events, chunks } = sinks();
  const error = new RangeError("key=sk-hidden");
  const call = callModel(live, model(error), REQUEST, { purpose: "chat" });
  expect(await call.catch((caught: unknown) => caught)).toBe(error);
  expect(chunks).toEqual([]);
  expect(events).toEqual([
    new ModelCallStarted({ ...identity(1), startedAt: ANY_TIME }),
    new ModelCallFinished({ ...identity(1), error: "RangeError" }),
  ]);
  expect(JSON.stringify(events)).not.toContain("sk-hidden");
  expect(modelFailureCode(error)).toBe("node.execute");
});

test("refused credentials are reported by name and coded provider.auth", async () => {
  const { live, events } = sinks({ stream: false });
  const error = new ProviderError("auth", "unauthorized", { status: 401 });
  await callModel(live, model(error), REQUEST, { purpose: "chat" }).catch(() => {});
  expect((events.at(-1) as ModelCallFinished).error).toBe("ProviderError");
  expect(modelFailureCode(error)).toBe("provider.auth");
});

test("a call without sinks still marks its error", async () => {
  const error = new ProviderError("quota", "out of credit");
  await callModel(null, model(error), REQUEST, { purpose: "chat" }).catch(() => {});
  expect(modelFailureCode(error)).toBe("provider.quota");
});

test("a retryable provider error then a reply publishes one pair per attempt, each its own step", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const { live, events } = sinks();
  const failure = new ProviderError("server", "boom", { retryAfter: 0.001 });
  const { response } = await callModel(live, model(failure, reply("ok")), REQUEST, { purpose: "chat" });
  expect(response.message.text()).toBe("ok");
  expect(events).toEqual([
    new ModelCallStarted({ ...identity(1), startedAt: ANY_TIME }),
    new ModelCallFinished({ ...identity(1), error: "ProviderError" }),
    new ModelCallStarted({ ...identity(2), startedAt: ANY_TIME }),
    new ModelCallFinished({ ...identity(2) }),
  ]);
  expect(modelFailureCode(failure)).toBeNull();
});

test("an interrupted call reports the usage received so far and no error", async () => {
  const interrupt = new AbortController();
  const events: unknown[] = [];
  const live = liveSinks({
    nodeId: "writer",
    llmConfig: CONFIG,
    emitEvent: (event) => void events.push(event),
    emitStream: (chunk) => void (chunk.chunk === "a" && interrupt.abort(new Error("interrupted"))),
  });
  const usage = new Usage({ inputTokens: 9, outputTokens: 1, totalTokens: 10 });
  const replyEvents = [new UsageReport(usage), new TextDelta(0, "a"), new TextDelta(0, "b"), new Finish({ reason: "stop" })];
  const options = { purpose: "chat" as const, interrupt: interrupt.signal };
  const cut = await rejected(callModel(live, model(replyEvents), REQUEST, options));
  expect(cut).toBeInstanceOf(ReplyInterrupted);
  const counted = new TokenUsage({ inputTokens: 9, outputTokens: 1, totalTokens: 10 });
  expect((cut as ReplyInterrupted).usage).toEqual(counted);
  expect(events).toEqual([
    new ModelCallStarted({ ...identity(1), startedAt: ANY_TIME }),
    new ModelCallFinished({ ...identity(1), usage: counted }),
  ]);
  expect(modelFailureCode(cut)).toBeNull();
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
  const call = callModel(live, model(reply("text")), REQUEST, { purpose: "chat" });
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
  const call = callModel(live, model(replyEvents), REQUEST, { purpose: "chat", signal: controller.signal });
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

const CALL = { n: 1, callIndex: 2, toolCallId: "c1", name: "lookup", arguments: { q: { deep: ["x"] } } };
const toolIdentity = { nodeId: "writer", n: 1, callIndex: 2, toolCallId: "c1", name: "lookup" };
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

const PROMPTED = { provider: "vllm", model: "m", structuredOutput: "prompt" as const };
const INT = Type.scalar(ValueKind.INTEGER);

/** The events an AGENT's event sink receives. */
type LiveEvent = Parameters<NonNullable<NodeCaps["emitEvent"]>>[0];

/**
 * Evaluate an AGENT `writer` over `chat` with the sinks asked for (both by default), granted
 * `lookup` when `tools` is given; return its outcome and what the sinks got.
 */
async function evaluate(
  chat: ScriptedModel,
  options: { stream?: boolean; events?: boolean; output?: Type; retries?: number; tools?: ToolSet } = {},
) {
  const { stream = true, events = true, output = null, retries, tools } = options;
  const llmConfig = { provider: chat.provider, model: chat.model };
  const node = new AgentNode("writer", { prompt: "hi", llmConfig, retries, tools: tools ? ["lookup"] : undefined });
  if (output !== null) node.outputContract = new ValueContract(output);
  const chunks: StreamChunk[] = [];
  const observed: LiveEvent[] = [];
  const outcome = await drive(node, {
    models: { env: NO_ENV, createModel: () => chat },
    tools,
    emitStream: stream ? (item) => void chunks.push(item) : null,
    emitEvent: events ? (event) => void observed.push(event) : null,
  });
  return { outcome, chunks, observed };
}

/** The last observed event, which must be a `ModelCallFinished`. */
function lastFinished(observed: readonly LiveEvent[]): ModelCallFinished {
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
    models: { env: NO_ENV, createModel: () => model(reply("text")) },
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
  const modelEvents = observed.filter((event) => event instanceof ModelCallStarted || event instanceof ModelCallFinished);
  expect(new Set(modelEvents.map((event) => event.purpose))).toEqual(new Set(["structured_fallback"]));
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

/** A tool set holding `lookup`, which records each argument record in `seen` and runs `action` on it. */
function lookupTools(
  seen: unknown[] = [],
  action = (args: Record<string, unknown>): unknown => `found:${args.topic}`,
): ToolSet {
  const lookup = new FunctionTool({
    name: "lookup",
    description: "Look it up.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => {
      seen.push(args);
      return action(args);
    },
  });
  return new ToolSet({ lookup });
}

/** A reply of `text` (when not empty) and `lookup` calls on each `[topic, id]`, with `usage`. */
function callReply(text: string, calls: [string, string][], usage: Usage | null = null): Response {
  const parts = calls.map(([topic, id]) => new ToolCallPart({ id, name: "lookup", arguments: { topic } }));
  const message = new Message({ role: "assistant", parts: text === "" ? parts : [new TextPart(text), ...parts] });
  return new Response({ message, usage, finishReason: "tool_calls" });
}

/** The identity of call `index` of `lookup` as `toolCallId` in the reply of step `n`. */
const callIdentity = (n: number, callIndex: number, toolCallId: string) =>
  ({ nodeId: "writer", n, callIndex, toolCallId, name: "lookup" });

test("an AGENT with tools orders its model and tool observers", async () => {
  const seen: unknown[] = [];
  const usage = (input: number, output: number) =>
    new Usage({ inputTokens: input, outputTokens: output, totalTokens: input + output });
  const chat = model(callReply("let me check", [["acme", "call-1"]], usage(10, 4)), reply("acme is fine", usage(20, 3)));

  const { outcome, chunks, observed } = await evaluate(chat, { stream: false, tools: lookupTools(seen) });

  expect((outcome as Output).value).toBe("acme is fine");
  expect(chunks).toEqual([]);
  expect(seen).toEqual([{ topic: "acme" }]);
  const first = (observed[0] as ModelCallStarted).n;
  const second = (observed[4] as ModelCallStarted).n;
  expect(first).not.toBe(second);
  const tokens = (input: number, output: number) =>
    new TokenUsage({ inputTokens: input, outputTokens: output, totalTokens: input + output });
  expect(observed).toEqual([
    new ModelCallStarted({ ...identity(first), startedAt: ANY_TIME }),
    new ModelCallFinished({ ...identity(first), usage: tokens(10, 4) }),
    new ToolCallStarted({ ...callIdentity(first, 0, "call-1"), arguments: { topic: "acme" } }),
    new ToolCallFinished({ ...callIdentity(first, 0, "call-1"), result: "found:acme" }),
    new ModelCallStarted({ ...identity(second), startedAt: ANY_TIME }),
    new ModelCallFinished({ ...identity(second), usage: tokens(20, 3) }),
  ]);
});

test("a tool error is reported by its name, then fails the node", async () => {
  const tools = lookupTools([], () => {
    throw new RangeError("secret detail");
  });

  const { outcome, observed } = await evaluate(model(callReply("", [["x", "call-1"]])), { tools });

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const n = (observed[0] as ModelCallStarted).n;
  expect(observed.at(-1)).toEqual(new ToolCallFinished({ ...callIdentity(n, 0, "call-1"), error: "RangeError" }));
  expect(JSON.stringify(observed)).not.toContain("secret detail");
});

test("two tool calls in one reply get their own indexed brackets", async () => {
  const seen: unknown[] = [];
  const chat = model(callReply("", [["a", "call-a"], ["b", "call-b"]]), reply("both fine"));

  const { outcome, observed } = await evaluate(chat, { stream: false, tools: lookupTools(seen) });

  expect((outcome as Output).value).toBe("both fine");
  expect(seen).toEqual([{ topic: "a" }, { topic: "b" }]);
  const brackets = observed.filter((event) => event instanceof ToolCallStarted || event instanceof ToolCallFinished);
  expect(brackets.map((event) => [event.constructor, event.callIndex, event.toolCallId])).toEqual([
    [ToolCallStarted, 0, "call-a"],
    [ToolCallFinished, 0, "call-a"],
    [ToolCallStarted, 1, "call-b"],
    [ToolCallFinished, 1, "call-b"],
  ]);
  expect(new Set(brackets.map((event) => event.n))).toEqual(new Set([(observed[0] as ModelCallStarted).n]));
});

// ---------- model_call steps ----------

/** A virtual scheduler recording each sleep it is asked for, in milliseconds. */
function recordingScheduler(): { scheduler: Scheduler; sleeps: number[] } {
  const inner = virtualScheduler();
  const sleeps: number[] = [];
  const scheduler: Scheduler = {
    now: () => inner.now(),
    sleep(ms, signal) {
      sleeps.push(ms);
      return inner.sleep(ms, signal);
    },
  };
  return { scheduler, sleeps };
}

/**
 * Evaluate AGENT `writer` over `chat` with one model retry, a recording scheduler and a draw of 1;
 * its steps (unless `steps: false`) go through `record`, else into `log` (a fresh one unless
 * given), which also gets every event, in order.
 */
async function withSteps(
  chat: ScriptedModel,
  options: {
    steps?: false;
    record?: (step: ValueRecord) => number;
    log?: unknown[];
    tools?: ToolSet;
    llmConfig?: Record<string, unknown>;
    output?: Type;
  } = {},
) {
  install({ ...BUILTIN, modelRetries: 1 });
  const { tools, output } = options;
  const llmConfig = options.llmConfig ?? { provider: chat.provider, model: chat.model };
  const node = new AgentNode("writer", { prompt: "hi", llmConfig, tools: tools ? ["lookup"] : undefined });
  if (output !== undefined) node.outputContract = new ValueContract(output);
  const log = options.log ?? [];
  // The engine returns a step's n; here, its place among the log's items will do.
  const record = options.record ?? ((step: ValueRecord) => log.push(step));
  const steps = options.steps === false ? null : new NodeSteps(1, [], record);
  const { scheduler, sleeps } = recordingScheduler();
  const outcome = await drive(node, {
    models: { env: NO_ENV, createModel: () => chat },
    tools,
    steps,
    scheduler,
    random: () => 1,
    emitEvent: (event) => void log.push(event),
  });
  return { outcome, log, sleeps };
}

/** What `log` holds, an event by its type and error, a step by its kind and, when it failed, its error's kind. */
function shape(log: readonly unknown[]): unknown[] {
  return log.map((item) => {
    if (item instanceof ModelCallStarted || item instanceof ModelCallFinished) {
      return [item.type, item instanceof ModelCallFinished ? item.error : null];
    }
    const step = item as ValueRecord;
    const error = step["error"] as ValueRecord | undefined;
    return error === undefined ? String(step["kind"]) : `${String(step["kind"])} failed ${String(error["kind"])}`;
  });
}

/** The steps among `log` (its plain records; events are class instances), in order. */
function stepsOf(log: readonly unknown[]): ValueRecord[] {
  return log.filter((item): item is ValueRecord => Object.getPrototypeOf(item) === Object.prototype);
}

test("a retried model call writes a failed step before its failed attempt's finished event", async () => {
  const failure = new ProviderError("server", "boom", { status: 500 });
  const { outcome, log, sleeps } = await withSteps(model(failure, reply("ok")));
  expect((outcome as Output).value).toBe("ok");
  // A draw of 1 scales the first backoff, 0.5 s, by 0.75.
  expect(sleeps).toEqual([375]);
  expect(shape(log)).toEqual([
    ["ModelCallStarted", null],
    "model_call failed server",
    ["ModelCallFinished", "ProviderError"],
    ["ModelCallStarted", null],
    "model_call",
    ["ModelCallFinished", null],
  ]);
  expect(stepsOf(log)[0]).toMatchObject({
    error: { kind: "server", message: "boom" },
    output: null,
    usage: null,
    retry_in: 0.375,
  });
});

test("a failed model_call step holds what arrived before the failure", async () => {
  const cut = [new TextDelta(0, "half"), new ProviderError("incomplete", "stream ended", { retryable: true })];
  const { log } = await withSteps(model(cut, reply("ok")));
  const step = stepsOf(log)[0]!;
  expect(step["output"]).toEqual(new Message({ role: "assistant", parts: [new TextPart("half")] }).toJSON());
  expect(step["error"]).toEqual({ kind: "incomplete", message: "stream ended" });
});

test("a tool loop writes its model calls and tool results in order", async () => {
  const failure = new ProviderError("rate_limit", "slow down", { status: 429 });
  const chat = model(callReply("", [["acme", "call-1"]]), failure, reply("acme is fine"));
  const { outcome, log } = await withSteps(chat, { tools: lookupTools() });
  expect((outcome as Output).value).toBe("acme is fine");
  expect(shape(stepsOf(log))).toEqual(["model_call", "tool_result", "model_call failed rate_limit", "model_call"]);
  expect(stepsOf(log)[1]).toMatchObject({ call_id: "call-1", is_error: false });
});

test("the final structured pass after a chat reply writes its calls after the chat call", async () => {
  const failure = new ProviderError("server", "boom", { status: 500 });
  const chat = model(reply("about four"), failure, reply('{"value": 4}'));
  const { outcome, log } = await withSteps(chat, { tools: lookupTools(), output: INT });
  expect((outcome as Output).value).toBe(4n);
  const steps = stepsOf(log);
  expect(shape(steps)).toEqual(["model_call", "model_call failed server", "model_call"]);
  expect(steps[0]!["purpose"]).toBe("chat");
  expect(steps[1]!["purpose"]).toBe(steps[2]!["purpose"]);
  expect(steps[1]!["purpose"]).not.toBe("chat");
});

test("a key echoed in a retried error is scrubbed in its failed model_call step", async () => {
  const key = "sk-sentinel-500-zqxw";
  const llmConfig = { provider: "vllm", model: "m", api_key: key };
  const chat = new ScriptedModel([new ProviderError("server", `upstream saw ${key}`, { status: 500 }), reply("ok")], llmConfig);
  const { log } = await withSteps(chat, { llmConfig });
  expect(stepsOf(log)[0]!["error"]).toEqual({ kind: "server", message: "upstream saw [REDACTED]" });
});

test("without a steps capability nothing is recorded and the call still retries", async () => {
  const chat = model(new ProviderError("server", "boom", { status: 500 }), reply("ok"));
  const { outcome, log, sleeps } = await withSteps(chat, { steps: false });
  expect((outcome as Output).value).toBe("ok");
  expect(sleeps).toEqual([375]);
  expect(shape(log)).toEqual([
    ["ModelCallStarted", null],
    ["ModelCallFinished", "ProviderError"],
    ["ModelCallStarted", null],
    ["ModelCallFinished", null],
  ]);
});

test("a steps.record fault ends the call as a host fault, unretried, with no finished event after it", async () => {
  const broken = new Error("log down");
  const chat = model(new ProviderError("server", "boom", { status: 500 }), reply("ok"));
  const record = () => {
    throw new CapabilityFault(broken);
  };
  const log: unknown[] = [];
  expect(await rejected(withSteps(chat, { record, log }))).toBe(broken);
  expect(chat.requests).toHaveLength(1);
  expect(shape(log)).toEqual([["ModelCallStarted", null]]);
});

// ---------- replay degraded ----------

test("a model's replay-degraded reasons reach the host once per reason per invocation", async () => {
  const chat = model(callReply("", [["acme", "call-1"]]), reply("acme is fine"));
  // Each request signals what its adapter would for the turns it replays.
  const reasons: ReplayDegradedReason[][] = [["other_account"], ["other_account", "other_provider"]];
  const models: ModelAccess = {
    env: NO_ENV,
    createModel: (_config, { onReplayDegraded }) => ({
      provider: chat.provider,
      model: chat.model,
      stream(request, options) {
        for (const reason of reasons.shift() ?? []) onReplayDegraded?.(reason);
        return chat.stream(request, options);
      },
    }),
  };
  const node = new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG, tools: ["lookup"] });
  const observed: unknown[] = [];
  const outcome = await drive(node, { models, tools: lookupTools(), emitEvent: (event) => void observed.push(event) });
  expect((outcome as Output).value).toBe("acme is fine");
  expect(observed.filter((event) => event instanceof ReplayDegraded)).toEqual([
    new ReplayDegraded({ nodeId: "writer", reason: "other_account" }),
    new ReplayDegraded({ nodeId: "writer", reason: "other_provider" }),
  ]);
});

const tokenUsage = (input: number, output: number) =>
  new TokenUsage({ inputTokens: input, outputTokens: output, totalTokens: input + output });

test("a structured native turn reports usage", async () => {
  const chat = model(reply('{"value": 4}', new Usage({ inputTokens: 7, outputTokens: 2, totalTokens: 9 })));
  const events: unknown[] = [];
  const result = await runAgentFlow(agentFlow({ llmConfig: CONFIG, output: INT }), {
    inputs: { question: "q" },
    models: serving(chat),
    onEvent: (event) => void events.push(event),
  });
  expect(result).toEqual(new RunSucceeded({ output: 4n }));
  const finished = events.filter((event) => event instanceof ModelCallFinished);
  expect(finished.map((event) => [event.purpose, event.usage])).toEqual([["structured_native", tokenUsage(7, 2)]]);
});
