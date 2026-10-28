/**
 * AGENT reply streaming: visible chat text reaches `emitStream` as it arrives, each chunk names its
 * step (`n`) and `index`, and a failed or refused reply sends no final chunk.
 */

import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import {
  ModelCallFinished,
  type ModelCallPurpose,
  ModelCallStarted,
  RunFailed,
  RunSucceeded,
  StreamChunk,
  TokenUsage,
  ToolCallStarted,
} from "../../src/events.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import { Finish, type Event, TextDelta, ToolCallDelta, UsageReport } from "../../src/llm/events.ts";
import { Message, Request, TextPart, ToolCallPart, ToolResultPart, Usage } from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import { ReplyInterrupted } from "../../src/nodes/agent/interrupts.ts";
import { AgentSteps } from "../../src/nodes/agent/context.ts";
import { callModel, liveSinks } from "../../src/nodes/agent/live.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CapabilityFault, type NodeCaps, type Output } from "../../src/nodes/base.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { decodeRecord, NodeStepRecord } from "../../src/runtime/log.ts";
import { runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";
import {
  agentFlow,
  drive,
  NO_ENV,
  PLAIN_CONFIG,
  PLAIN_FLOW,
  rejected,
  runAgentFlow,
  runWith,
  serving,
} from "../helpers/run_state_fixtures.ts";

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

/** Chunk `index` of step `n` of `writer`. */
const chunk = (text: string, index: number, n = 1, final = false) =>
  new StreamChunk({ nodeId: "writer", key: "text", chunk: text, final, n, index });

/** A started event's time, which `callModel` reads from the real clock without `record`. */
const ANY_TIME = expect.any(Number) as unknown as number;

test("stream chunk n and index default to null", () => {
  expect(new StreamChunk({ nodeId: "n", key: "text", chunk: "hi" }).n).toBeNull();
  expect(new StreamChunk({ nodeId: "n", key: "text", chunk: "hi" }).index).toBeNull();
  expect(new StreamChunk({ nodeId: "n", key: "text", chunk: "hi", n: 2, index: 0 }).n).toBe(2);
});

test("plain chat streams text chunks then one final chunk", async () => {
  const { live, events, chunks } = sinks();
  const usage = new Usage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 });
  const replyEvents = [...streamed("hello", " big", " world").slice(0, -1), new UsageReport(usage), FINISH];
  const { response } = await callModel(live, model(replyEvents), REQUEST, { purpose: "chat" });
  expect(response.message.text()).toBe("hello big world");
  expect(chunks).toEqual([chunk("hello", 0), chunk(" big", 1), chunk(" world", 2), chunk("", 3, 1, true)]);
  const identity = { nodeId: "writer", n: 1, purpose: "chat" as const, ...CONFIG };
  expect(events).toEqual([
    new ModelCallStarted({ ...identity, startedAt: ANY_TIME }),
    new ModelCallFinished({ ...identity, usage: new TokenUsage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 }) }),
  ]);
});

test("stream sink alone streams without observers", async () => {
  const { live, events, chunks } = sinks({ events: false });
  await callModel(live, model(streamed("just", " text")), REQUEST, { purpose: "chat" });
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
    await callModel(live, model(streamed('{"value": ', "3}")), REQUEST, { purpose });
    expect(chunks).toEqual([]);
    expect(events.map((event) => (event as ModelCallStarted).purpose)).toEqual([purpose, purpose]);
  }
});

test("mid-stream failure sends no final chunk", async () => {
  const { live, events, chunks } = sinks();
  const cut = [new TextDelta(0, "partial"), new TextDelta(0, " answer"), new ProviderError("connection", "reset by sk-hidden")];
  await callModel(live, model(cut), REQUEST, { purpose: "chat" }).catch(() => {});
  expect(chunks).toEqual([chunk("partial", 0), chunk(" answer", 1)]);
  expect(events.at(-1)).toEqual(
    new ModelCallFinished({ nodeId: "writer", n: 1, purpose: "chat", ...CONFIG, error: "ProviderError" }),
  );
  expect(JSON.stringify(events)).not.toContain("sk-hidden");
});

test("a reply the assembler refuses sends no final chunk", async () => {
  const { live, events, chunks } = sinks();
  const unfinished = [new TextDelta(0, "hello"), new TextDelta(0, " world")];
  const error = await callModel(live, model(unfinished), REQUEST, { purpose: "chat" }).catch((caught: unknown) => caught);
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
  const call = callModel(live, model(streamed("text")), REQUEST, { purpose: "chat" });
  expect(await call.catch((caught: unknown) => caught)).toBe(fault);
  expect(events.map((event) => (event as { type: string }).type)).toEqual(["ModelCallStarted"]);
});

test("a retried attempt's chunks carry n 2 without steps, their index from 0", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const { live, events, chunks } = sinks();
  const cut = [new TextDelta(0, "par"), new ProviderError("server", "boom", { retryAfter: 0.001 })];
  await callModel(live, model(cut, streamed("whole")), REQUEST, { purpose: "chat" });
  expect(chunks).toEqual([chunk("par", 0), chunk("whole", 0, 2), chunk("", 1, 2, true)]);
  expect(events.map((event) => [(event as { type: string }).type, (event as ModelCallStarted).n])).toEqual([
    ["ModelCallStarted", 1],
    ["ModelCallFinished", 1],
    ["ModelCallStarted", 2],
    ["ModelCallFinished", 2],
  ]);
});

test("empty text deltas send no chunk", async () => {
  const { live, chunks } = sinks({ events: false });
  await callModel(live, model(streamed("", "hi", "")), REQUEST, { purpose: "chat" });
  expect(chunks).toEqual([chunk("hi", 0), chunk("", 1, 1, true)]);
});

test("an interrupted call ends its stream with the final chunk", async () => {
  const interrupt = new AbortController();
  const events: unknown[] = [];
  const chunks: StreamChunk[] = [];
  const live = liveSinks({
    nodeId: "writer",
    llmConfig: CONFIG,
    emitEvent: (event) => void events.push(event),
    emitStream: (item) => {
      chunks.push(item);
      if (item.chunk === " big") interrupt.abort(new Error("interrupted"));
    },
  });
  const options = { purpose: "chat" as const, interrupt: interrupt.signal };
  const cut = await rejected(callModel(live, model(streamed("hello", " big", " world")), REQUEST, options));
  expect(cut).toBeInstanceOf(ReplyInterrupted);
  expect((cut as ReplyInterrupted).reply).toEqual(new Message({ role: "assistant", parts: [new TextPart("hello big")] }));
  expect(chunks).toEqual([chunk("hello", 0), chunk(" big", 1), chunk("", 2, 1, true)]);
  expect(events.map((event) => (event as { type: string }).type)).toEqual(["ModelCallStarted", "ModelCallFinished"]);
});

// ---------- through the node ----------

/**
 * Evaluate an AGENT `writer` over `model` with the given sinks, granted `lookup` when `tools` is
 * given; return its outcome and what they got.
 */
async function evaluate(model: ScriptedModel, options: { events?: boolean; tools?: ToolSet } = {}) {
  const { tools } = options;
  const chunks: StreamChunk[] = [];
  const observed: Parameters<NonNullable<NodeCaps["emitEvent"]>>[0][] = [];
  const node = new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG, tools: tools ? ["lookup"] : undefined });
  const outcome = await drive(node, {
    models: { env: NO_ENV, createModel: () => model },
    tools,
    emitStream: (item) => void chunks.push(item),
    emitEvent: options.events === false ? null : (event) => void observed.push(event),
  });
  return { outcome, chunks, observed };
}

test("a plain agent streams text chunks then one final chunk", async () => {
  const usage = new Usage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 });
  const replyEvents = [...streamed("hello", " big", " world").slice(0, -1), new UsageReport(usage), FINISH];
  const { outcome, chunks, observed } = await evaluate(model(replyEvents));
  expect((outcome as Output).value).toBe("hello big world");
  expect(chunks).toEqual([chunk("hello", 0), chunk(" big", 1), chunk(" world", 2), chunk("", 3, 1, true)]);
  const identity = { nodeId: "writer", n: 1, purpose: "chat" as const, ...CONFIG };
  expect(observed).toEqual([
    new ModelCallStarted({ ...identity, startedAt: (observed[0] as ModelCallStarted).startedAt }),
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
    models: { env: NO_ENV, createModel: () => model(streamed("text")) },
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
  expect(chunks.map((item) => [item.n, item.index, item.chunk, item.final])).toEqual([
    [1, 0, "partial", false],
    [1, 1, " answer", false],
  ]);
  expect(observed.at(-1)).toEqual(
    new ModelCallFinished({ nodeId: "writer", n: 1, purpose: "chat", ...CONFIG, error: "ProviderError" }),
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
  const chunks: StreamChunk[] = [];
  const events: unknown[] = [];
  const result = await runWith(
    await loadFlow(PLAIN_FLOW),
    { question: "q1" },
    null,
    new ScriptedModel([script()], PLAIN_CONFIG),
    { onStream: (item) => void chunks.push(item), onEvent: (event) => void events.push(event) },
  );
  expect(result.status).toBe("failed");
  expect(result.failure?.code).toBe("provider.auth");
  expect(chunks.some((item) => item.final)).toBe(false);
  expect(events.filter((event) => event instanceof ModelCallFinished).map((event) => event.error)).toEqual([
    "ProviderError",
  ]);
});

/** A tool set holding `lookup`, which records each argument record in `seen` and answers `found:<topic>`. */
function lookupTools(seen: unknown[] = []): ToolSet {
  return new ToolSet({ lookup: lookupTool(seen) });
}

/** The `lookup` tool: records each argument record in `seen` and answers `found:<topic>`. */
function lookupTool(seen: unknown[] = []): FunctionTool {
  return new FunctionTool({
    name: "lookup",
    description: "Look it up.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => {
      seen.push(args);
      return `found:${args.topic}`;
    },
  });
}

/**
 * The events of a reply streaming `text` word by word, then each `lookup` call on `[topic, id]`
 * as two deltas sharing its index.
 */
function streamedCalls(text: string, calls: [string, string][]): Event[] {
  const words = text === "" ? [] : text.split(" ").map((word, i) => (i === 0 ? word : ` ${word}`));
  const deltas: Event[] = words.map((word) => new TextDelta(0, word));
  const first = deltas.length > 0 ? 1 : 0;
  for (const [i, [topic, id]] of calls.entries()) {
    const index = first + i;
    deltas.push(new ToolCallDelta({ index, id, name: "lookup", arguments: '{"topic": ' }));
    deltas.push(new ToolCallDelta({ index, id: null, name: null, arguments: JSON.stringify(topic) + "}" }));
  }
  return [...deltas, new Finish({ reason: "tool_calls" })];
}

test("a tool-calling agent streams each exchange between its observers", async () => {
  const chat = model(streamedCalls("let me check", [["acme", "call-1"]]), streamed("acme", " is", " fine"));

  const { outcome, chunks, observed } = await evaluate(chat, { tools: lookupTools() });

  expect((outcome as Output).value).toBe("acme is fine");
  // The reply is step 1, its tool result step 2, so the second reply is step 3.
  const [first, second] = [(observed[0] as ModelCallStarted).n, (observed[4] as ModelCallStarted).n];
  expect([first, second]).toEqual([1, 3]);
  const call = { nodeId: "writer", n: first, callIndex: 0, toolCallId: "call-1", name: "lookup" };
  expect(observed[2]).toEqual(new ToolCallStarted({ ...call, arguments: { topic: "acme" } }));
  expect(chunks.map((item) => [item.n, item.index, item.chunk, item.final])).toEqual([
    [first, 0, "let", false],
    [first, 1, " me", false],
    [first, 2, " check", false],
    [first, 3, "", true],
    [second, 0, "acme", false],
    [second, 1, " is", false],
    [second, 2, " fine", false],
    [second, 3, "", true],
  ]);
});

test("two streamed tool calls aggregate by index", async () => {
  const seen: unknown[] = [];
  const chat = model(streamedCalls("", [["a", "call-a"], ["b", "call-b"]]), streamed("both", " fine"));

  const { outcome, chunks, observed } = await evaluate(chat, { tools: lookupTools(seen) });

  expect((outcome as Output).value).toBe("both fine");
  expect(seen).toEqual([{ topic: "a" }, { topic: "b" }]);
  const started = observed.filter((event) => event instanceof ToolCallStarted);
  expect(started.map((event) => [event.callIndex, event.toolCallId, event.arguments])).toEqual([
    [0, "call-a", { topic: "a" }],
    [1, "call-b", { topic: "b" }],
  ]);
  expect(chunks.map((item) => [item.chunk, item.final])).toEqual([
    ["", true],
    ["both", false],
    [" fine", false],
    ["", true],
  ]);
});

// One AGENT run through `FlowEngine`.

test("a streamed tool exchange streams each reply's text", async () => {
  const chat = model(streamedCalls("", [["a", "call-1"]]), streamed("one"));
  const chunks: StreamChunk[] = [];
  const result = await runAgentFlow(agentFlow({ tools: ["lookup"] }), {
    tools: lookupTools(),
    models: serving(chat),
    inputs: { question: "q1" },
    onStream: (item) => void chunks.push(item),
  });
  expect(result).toEqual(new RunSucceeded({ output: "one" }));
  expect(chunks.filter((item) => item.chunk !== "").map((item) => item.chunk)).toEqual(["one"]);
  expect(chat.requests).toHaveLength(2);
  // The second request carries the assembled tool call and its result.
  const [, , assistant, tool] = chat.requests[1]!.messages;
  const call = assistant!.parts.find((part) => part instanceof ToolCallPart) as ToolCallPart;
  expect([call.id, call.name, call.arguments]).toEqual(["call-1", "lookup", { topic: "a" }]);
  expect(tool!.parts).toEqual([new ToolResultPart({ callId: "call-1", name: "lookup", content: "found:a", isError: false })]);
});

test("a mid-stream failure sends no final chunk and reports the error", async () => {
  const chat = model([new TextDelta(0, "partial"), new ProviderError("connection", "reset")]);
  const chunks: StreamChunk[] = [];
  const events: unknown[] = [];
  const result = await runAgentFlow(agentFlow(), {
    inputs: { question: "q1" },
    models: serving(chat),
    onStream: (item) => void chunks.push(item),
    onEvent: (event) => void events.push(event),
  });
  expect(result).toBeInstanceOf(RunFailed);
  expect(chunks.map((item) => [item.chunk, item.final])).toEqual([["partial", false]]);
  expect(events.filter((event) => event instanceof ModelCallFinished).map((event) => event.error)).toEqual([
    "ProviderError",
  ]);
});

// ---------- step ids: chunks and call events name the step their record becomes ----------

/** A fresh run log path in its own folder. */
function stepLog(): string {
  return join(mkdtempSync(join(tmpdir(), "yian-agent-streaming-")), "log.jsonl");
}

/** The `[n, kind]` of each step the log at `path` holds for `nodeId`. */
function loggedSteps(path: string, nodeId = "agent"): [number, string][] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .flatMap((line) => {
      const { record } = decodeRecord(line);
      return record instanceof NodeStepRecord && record.nodeId === nodeId ? [[record.n, String(record.step["kind"])]] : [];
    }) as [number, string][];
}

/** Run `PLAIN_FLOW`-style `flow` over `chat` with a log, collecting its chunks and events. */
async function streamedRun(
  chat: ScriptedModel,
  options: { tools?: Record<string, FunctionTool>; modelRetries?: number; flow?: string } = {},
) {
  const { tools, modelRetries = 0, flow = PLAIN_FLOW } = options;
  const log = stepLog();
  const chunks: StreamChunk[] = [];
  const events: unknown[] = [];
  const env = (name: string) => (name === "YIAN_MODEL_RETRIES" ? String(modelRetries) : NO_ENV(name));
  const result = await runFlow(await loadFlow(flow), { question: "q1" }, {
    log,
    env,
    random: () => 0,
    models: serving(chat, env),
    tools,
    onStream: (item) => void chunks.push(item),
    onEvent: (event) => void events.push(event),
  });
  return { log, result, chunks, events };
}

test("chunks carry their model_call step's n and a dense index from 0", async () => {
  const flow = PLAIN_FLOW.replace("    output: str", "    tools: [lookup]\n    output: str");
  const chat = model(streamedCalls("let me check", [["acme", "call-1"]]), streamed("acme", " is", " fine"));
  const { log, result, chunks, events } = await streamedRun(chat, { tools: { lookup: lookupTool() }, flow });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(loggedSteps(log)).toEqual([
    [1, "model_call"],
    [2, "tool_result"],
    [3, "model_call"],
  ]);
  expect(chunks.map((item) => [item.n, item.index, item.chunk, item.final])).toEqual([
    [1, 0, "let", false],
    [1, 1, " me", false],
    [1, 2, " check", false],
    [1, 3, "", true],
    [3, 0, "acme", false],
    [3, 1, " is", false],
    [3, 2, " fine", false],
    [3, 3, "", true],
  ]);
  // The call events name the same step, and a tool call names its reply's step.
  const named = events.flatMap((event) =>
    event instanceof ModelCallStarted || event instanceof ModelCallFinished || event instanceof ToolCallStarted
      ? [[event.type, event.nodeId, event.n]]
      : [],
  );
  expect(named).toEqual([
    ["ModelCallStarted", "agent", 1],
    ["ModelCallFinished", "agent", 1],
    ["ToolCallStarted", "agent", 1],
    ["ModelCallStarted", "agent", 3],
    ["ModelCallFinished", "agent", 3],
  ]);
});

test("a retried attempt streams under the next n, its index restarting at 0", async () => {
  const cut = [new TextDelta(0, "par"), new TextDelta(0, "tial"), new ProviderError("server", "boom", { retryAfter: 0.001 })];
  const { log, result, chunks, events } = await streamedRun(model(cut, streamed("whole", " reply")), { modelRetries: 1 });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(loggedSteps(log)).toEqual([
    [1, "model_call"],
    [2, "model_call"],
  ]);
  expect(chunks.map((item) => [item.n, item.index, item.chunk, item.final])).toEqual([
    [1, 0, "par", false],
    [1, 1, "tial", false],
    [2, 0, "whole", false],
    [2, 1, " reply", false],
    [2, 2, "", true],
  ]);
  const calls = events.filter((event) => event instanceof ModelCallStarted || event instanceof ModelCallFinished);
  expect(calls.map((event) => [event.type, event.n, event instanceof ModelCallFinished ? event.error : null])).toEqual([
    ["ModelCallStarted", 1, null],
    ["ModelCallFinished", 1, "ProviderError"],
    ["ModelCallStarted", 2, null],
    ["ModelCallFinished", 2, null],
  ]);
});

test("a call's n follows the steps its start recorded before it", async () => {
  const prior: ValueRecord[] = [{ kind: "llm_config" }];
  const recorded: ValueRecord[] = [];
  const nodeSteps = new NodeSteps(2, prior, (step) => {
    recorded.push(step);
    return prior.length + recorded.length;
  });
  expect(nodeSteps.next).toBe(2);
  // A call-time agent records its config before the loop builds its steps.
  nodeSteps.record({ kind: "llm_config" });
  expect(nodeSteps.next).toBe(3);
  const steps = new AgentSteps(nodeSteps, () => 0, null, (error) => error, "writer");
  expect(steps.nextN()).toBe(3);
  const { live, events, chunks } = sinks();
  const record = { steps, digest: "d".repeat(64) };
  const { n } = await callModel(live, model(streamed("hi", "!")), REQUEST, { purpose: "chat", record });
  expect(n).toBe(3);
  expect(chunks.map((item) => [item.n, item.index])).toEqual([
    [3, 0],
    [3, 1],
    [3, 2],
  ]);
  expect(events.map((event) => (event as ModelCallStarted).n)).toEqual([3, 3]);
  expect(steps.nextN()).toBe(4);
});
