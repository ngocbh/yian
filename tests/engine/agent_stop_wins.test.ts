/**
 * A stop pending when a model, tool or structured call fails stops the node cleanly: the stop wins
 * over the failure, and a host callback fault escapes.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallFinished, ModelCallStarted, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { FunctionTool, type StandardSchemaV1 } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, NO_ENV, rejected, textReply } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };

/** A model whose call runs `during`, then throws `error`. */
function failing(error: unknown, during: () => void, config = CONFIG): ChatModel {
  return {
    ...config,
    stream: () =>
      new EventStream(
        (async function* (): AsyncGenerator<Event> {
          during();
          throw error;
        })(),
      ),
  };
}

/** Evaluate `node` (by default an AGENT `writer`) over `model` with the other `drive` options. */
function evaluate(model: ChatModel, options: Parameters<typeof drive>[1] & { node?: AgentNode } = {}) {
  const { node = new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG }), ...rest } = options;
  return drive(node, { models: { env: NO_ENV, createModel: () => model }, ...rest });
}

test("a stop during a failing model call stops the node", async () => {
  const stop = new AbortController();
  const model = failing(new Error("provider boom"), () => stop.abort(new Error("stopped")));
  expect(await rejected(evaluate(model, { signal: stop.signal }))).toBe(stop.signal.reason);
});

test("a stop during a failing structured call stops the node", async () => {
  const stop = new AbortController();
  const node = new AgentNode("writer", { prompt: "hi", llmConfig: PROMPTED, retries: 2 });
  node.outputContract = new ValueContract(Type.scalar(ValueKind.INTEGER));
  const model = failing(new Error("provider boom"), () => stop.abort(new Error("stopped")), PROMPTED);
  expect(await rejected(evaluate(model, { node, signal: stop.signal }))).toBe(stop.signal.reason);
});

test("a stop raised while observing a failed model call wins", async () => {
  const stop = new AbortController();
  const observed: string[] = [];
  const emitEvent = (event: unknown) => {
    if (event instanceof ModelCallFinished) {
      observed.push(event.error ?? "");
      stop.abort(new Error("stopped"));
    }
  };
  const model = failing(new Error("boom"), () => {});
  expect(await rejected(evaluate(model, { signal: stop.signal, emitEvent }))).toBe(stop.signal.reason);
  expect(observed).toEqual(["Error"]);
});

test("a stop failing a model call is the node's rejection, with no finished event", async () => {
  const stop = new AbortController();
  const events: string[] = [];
  const emitEvent = (event: unknown) => void events.push((event as { type: string }).type);
  // The call fails with the stop's reason, as a call the signal aborted does.
  const model = failing(new Error("unreached"), () => {
    stop.abort(new Error("stopped"));
    stop.signal.throwIfAborted();
  });
  expect(await rejected(evaluate(model, { signal: stop.signal, emitEvent }))).toBe(stop.signal.reason);
  expect(events).toEqual(["ModelCallStarted"]);
});

test("a host callback fault escapes a pending stop", async () => {
  const stop = new AbortController();
  const fault = new RangeError("sink boom");
  const emitEvent = (event: unknown) => {
    if (event instanceof ModelCallStarted) {
      stop.abort(new Error("stopped"));
      throw fault;
    }
  };
  const reply = new Response({ message: new Message({ role: "assistant", parts: [new TextPart("fine")] }), finishReason: "stop" });
  const model = new ScriptedModel([reply], CONFIG);
  expect(await rejected(evaluate(model, { signal: stop.signal, emitEvent }))).toBe(fault);
});

/** A reply calling `lookup` with no arguments as `call-1`. */
function lookupCall(): Response {
  const call = new ToolCallPart({ id: "call-1", name: "lookup", arguments: {} });
  return new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
}

/** A tool set holding `lookup`, which takes no arguments and runs `execute`. */
function lookupTools(execute: FunctionTool["execute"], parametersValidator: StandardSchemaV1 | null = null): ToolSet {
  const parametersJsonSchema = { type: "object", properties: {} };
  return new ToolSet({
    lookup: new FunctionTool({ name: "lookup", description: "Look it up.", parametersJsonSchema, execute, parametersValidator }),
  });
}

test("a stop cutting a tool's argument check short is the stop", async () => {
  const stop = new AbortController();
  const validator: StandardSchemaV1 = {
    "~standard": {
      version: 1,
      vendor: "test",
      validate: () => {
        stop.abort(new Error("stopped"));
        throw new Error("checker boom");
      },
    },
  };
  const ran: unknown[] = [];
  const events: unknown[] = [];
  const node = new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG, tools: ["lookup"] });
  const run = node.forward(
    {},
    {
      models: { env: NO_ENV, createModel: () => new ScriptedModel([lookupCall()], CONFIG) },
      tools: lookupTools((args) => ran.push(args), validator),
      signal: stop.signal,
      emitEvent: (event) => void events.push(event),
    },
  );
  expect(await rejected(run)).toBe(stop.signal.reason);
  expect(ran).toEqual([]);
  expect(events.some((event) => event instanceof ToolCallStarted)).toBe(false);
});

/** An AGENT `writer` granted `lookup`. */
const toolAgent = () => new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG, tools: ["lookup"] });

/** A model calling `lookup`, then answering `done`. */
const lookupModel = () => new ScriptedModel([lookupCall(), textReply("done")], CONFIG);

test("a stop during a failing tool call stops the node", async () => {
  const stop = new AbortController();
  const tools = lookupTools(() => {
    stop.abort(new Error("stopped"));
    throw new Error("tool boom");
  });
  const outcome = evaluate(lookupModel(), { node: toolAgent(), tools, signal: stop.signal });
  expect(await rejected(outcome)).toBe(stop.signal.reason);
});

// Each row: the tool's body, then the finished event's `error`; its `result` is `null` in both.
test.each([
  [
    "a failed tool call",
    () => {
      throw new Error("boom");
    },
    "Error",
  ],
  ["a tool result that can't be written", () => new Date(0), null],
] as const)("a stop raised while observing %s wins", async (_label, execute, error) => {
  const stop = new AbortController();
  const observed: [string | null, string | null][] = [];
  const emitEvent = (event: unknown) => {
    if (event instanceof ToolCallFinished) {
      observed.push([event.error, event.result]);
      stop.abort(new Error("stopped"));
    }
  };
  const tools = lookupTools(execute);
  const outcome = evaluate(lookupModel(), { node: toolAgent(), tools, signal: stop.signal, emitEvent });
  expect(await rejected(outcome)).toBe(stop.signal.reason);
  expect(observed).toEqual([[error, null]]);
});

test("a stop failing a tool call is the node's rejection, with no finished event", async () => {
  const stop = new AbortController();
  const events: string[] = [];
  const emitEvent = (event: unknown) => void events.push((event as { type: string }).type);
  // The call fails with the stop's reason, as a call the signal aborted does.
  const tools = lookupTools(() => {
    stop.abort(new Error("stopped"));
    stop.signal.throwIfAborted();
  });
  const outcome = evaluate(lookupModel(), { node: toolAgent(), tools, signal: stop.signal, emitEvent });
  expect(await rejected(outcome)).toBe(stop.signal.reason);
  expect(events).toEqual(["ModelCallStarted", "ModelCallFinished", "ToolCallStarted"]);
});
