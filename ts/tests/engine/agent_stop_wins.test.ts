/**
 * A stop pending when a model call fails stops the node cleanly: mirrors
 * `tests/engine/test_agent_stop_wins.py`.
 *
 * Changed: the rows run an AGENT through `evalNode` (`drive`) with a model from `models`, the tool
 * in the run's `ToolSet`, and a stop signal, where Python runs `_run_flow_controlled` with a stop
 * event; the stop is the signal, so a stopped node rejects with the signal's reason. Ported this
 * way, the rows without session memory: `test_a_stop_during_a_failing_model_call_stops_the_run`,
 * `test_a_stop_during_a_failing_tool_call_stops_the_run`,
 * `test_a_stop_during_a_failing_structured_call_stops_the_run` and
 * `test_a_host_callback_fault_escapes_a_pending_stop`.
 * `test_a_stop_raised_while_observing_a_failure_wins` (3 rows): TS has no memory turn observing
 * the call, so the observer is the event sink, which aborts the stop when it sees the call's
 * finished event: the failed model call's `ModelCallFinished`, the failed tool call's
 * `ToolCallFinished`, or (the representation row) the `ToolCallFinished` of a call whose result
 * can't be written as text (a `Date`, reported with `result: null`); the stop wins over the
 * failure. `test_a_stop_failing_a_call_yields_to_an_observed_stop_only_if_derived` (the model and
 * tool rows): TS has one stop, the signal's reason, with no initiating or derived kind; a call the
 * stop fails emits no finished event and the node rejects with the reason.
 *
 * Ported the same way through `FlowEngine` with a `SessionMemory` (`runSessionFlow`), the session
 * rows of `test_a_stop_during_a_failing_model_call_stops_the_run`,
 * `test_a_stop_during_a_failing_tool_call_stops_the_run` and
 * `test_a_host_callback_fault_escapes_a_pending_stop`, and the interrupted-effect records of
 * `test_a_stop_during_a_failing_structured_call_stops_the_run`: the stop's reason carries no notes
 * (Python's `_contexts` chain has no TS form). The host-callback row runs through `evalNode` with
 * the runtime's opener instead, since the engine stops a run with the first of the host's stop and
 * its sink's error, and the sink aborts the stop first.
 *
 * TS-only: a stop that cuts a tool's argument check short (its validator aborts the stop, then
 * throws) is the node's rejection, with no tool brackets and no recorded tool call.
 *
 * Not ported: `test_a_stop_wins_over_a_malformed_native_carrier` (TS has no LangChain structured
 * carrier).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallFinished, ModelCallStarted, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import {
  ConversationInvocation,
  type ConversationView,
  type EvaluationTrace,
  MemoryScope,
  type VisibleInteraction,
} from "../../src/memory/contracts.ts";
import { SessionMemory } from "../../src/memory/session.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { FunctionTool, type StandardSchemaV1 } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, rejected, serving, textReply } from "./run_state_fixtures.ts";
import { runSessionFlow, sessionFlow, view, withSessionOpener } from "./session_fakes.ts";

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
  return drive(node, { models: { env: () => undefined, createModel: () => model }, ...rest });
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
  const interactions: VisibleInteraction[] = [];
  const evaluationTrace: EvaluationTrace = {
    recordPrompt: () => {},
    recordInteraction: (interaction) => void interactions.push(interaction),
  };
  const node = new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG, tools: ["lookup"] });
  const run = node.forward(
    {},
    {
      models: { env: () => undefined, createModel: () => new ScriptedModel([lookupCall()], CONFIG) },
      tools: lookupTools((args) => ran.push(args), validator),
      signal: stop.signal,
      emitEvent: (event) => void events.push(event),
      evaluationTrace,
    },
  );
  expect(await rejected(run)).toBe(stop.signal.reason);
  expect(ran).toEqual([]);
  expect(events.some((event) => event instanceof ToolCallStarted)).toBe(false);
  expect(interactions).toEqual([]);
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

// The session rows: the AGENT of `sessionFlow` through `FlowEngine` with a `SessionMemory`.

/**
 * Run `sessionFlow(flowOptions)` over `model` with a fresh `SessionMemory` and the other engine
 * options; return what the run rejects with and the memory's conversation.
 */
async function stoppedSession(
  model: ChatModel,
  flowOptions: Parameters<typeof sessionFlow>[0],
  options: Parameters<typeof runSessionFlow>[2],
) {
  const memory = new SessionMemory();
  const error = await rejected(
    runSessionFlow(sessionFlow(flowOptions), memory, { inputs: { question: "q" }, models: serving(model), ...options }),
  );
  return { error, view: view(memory) };
}

/** The status of each effect of `stored` and the failure code of its last. */
const outcome = (stored: ConversationView) => ({
  statuses: stored.effects.map((effect) => effect.status),
  code: stored.effects.at(-1)!.failure?.code,
  closed: stored.closedInvocations.at(-1)!.status,
});

test("a stop during a failing model call stops the run and records the call interrupted", async () => {
  const stop = new AbortController();
  const model = failing(new Error("provider boom"), () => stop.abort(new Error("stopped")));
  const { error, view: stored } = await stoppedSession(model, {}, { signal: stop.signal });
  expect(error).toBe(stop.signal.reason);
  expect(error).not.toHaveProperty("notes");
  expect(outcome(stored)).toEqual({ statuses: ["interrupted"], code: "api.model.interrupted", closed: "interrupted" });
});

test("a stop during a failing tool call stops the run and records the call interrupted", async () => {
  const stop = new AbortController();
  const tools = lookupTools(() => {
    stop.abort(new Error("stopped"));
    throw new Error("tool boom");
  });
  const options = { tools, signal: stop.signal };
  const { error, view: stored } = await stoppedSession(lookupModel(), { tools: ["lookup"] }, options);
  expect(error).toBe(stop.signal.reason);
  expect(error).not.toHaveProperty("notes");
  expect(outcome(stored)).toEqual({
    statuses: ["completed", "interrupted"],
    code: "api.tool.interrupted",
    closed: "interrupted",
  });
});

test("a stop during a failing structured call stops the run and records the call interrupted", async () => {
  const stop = new AbortController();
  const model = failing(new Error("provider boom"), () => stop.abort(new Error("stopped")), PROMPTED);
  const flowOptions = { llmConfig: PROMPTED, retries: 2, output: Type.scalar(ValueKind.INTEGER) };
  const { error, view: stored } = await stoppedSession(model, flowOptions, { signal: stop.signal });
  expect(error).toBe(stop.signal.reason);
  expect(error).not.toHaveProperty("notes");
  expect(outcome(stored)).toEqual({ statuses: ["interrupted"], code: "api.model.interrupted", closed: "interrupted" });
});

test("a host callback fault escapes a pending stop in a session, the call recorded failed", async () => {
  const stop = new AbortController();
  const fault = new RangeError("sink boom");
  const emitEvent = (event: unknown) => {
    if (event instanceof ModelCallStarted) {
      stop.abort(new Error("stopped"));
      throw fault;
    }
  };
  const memory = new SessionMemory();
  const node = new AgentNode("writer", { prompt: "hi", llmConfig: CONFIG });
  node.memoryScope = new MemoryScope(["writer"], { memories: ["session"] });
  const model = new ScriptedModel([textReply("fine")], CONFIG);
  // Through the evaluator, not the engine: the engine would stop the run with the host's stop,
  // which comes first.
  const running = withSessionOpener(memory, "writer", (memoryContextOpener) =>
    evaluate(model, {
      node,
      signal: stop.signal,
      emitEvent,
      memoryContextOpener,
      memoryInvocation: new ConversationInvocation("run", 1),
    }),
  );
  expect(await rejected(running)).toBe(fault);
  expect(outcome(view(memory))).toEqual({ statuses: ["failed"], code: "api.model.failed", closed: "interrupted" });
});
