/**
 * Host steering of a running AGENT: mirrors `tests/engine/test_agent_steering.py`.
 *
 * Python's inbox answers `interrupted(node_id)` when asked; TS steering is push-based (design
 * 6.8): the test inbox hands out one `AbortSignal` per node, aborted to interrupt, and `drain`
 * returns (or promises) the queued steers. Flows are hand-built (`START -> chat [-> after] -> END`)
 * and run by `FlowEngine` with a `ScriptedModel` from `models` and tools in the run's `ToolSet`
 * (Python: `load_flow` and `run_flow`); a reply Python's recorder makes inside the call sets the
 * interrupt from `onEvent` here. Python's `ValueError` is a `RangeError`.
 *
 * Ported: `test_non_steerable_agent_never_calls_the_inbox`,
 * `test_steers_are_sent_as_user_messages_at_the_top_of_each_iteration`,
 * `test_inbox_failure_escapes_as_a_host_fault` (a sync throw and a rejected promise),
 * `test_interrupt_before_the_first_model_call_returns_empty_text`,
 * `test_interrupt_before_a_tool_call_skips_only_the_remaining_calls` (without the memory half),
 * `test_skipped_tool_calls_reach_the_host_as_interrupted_tool_events`,
 * `test_interrupt_at_the_top_of_a_later_iteration_returns_the_latest_reply`,
 * `test_interrupted_node_succeeds_and_the_flow_continues` and
 * `test_interrupt_for_a_finished_node_is_ignored`.
 *
 * Ported through `evalNode` (`drive`): `test_malformed_inbox_answers_escape_as_host_type_errors`
 * and `test_malformed_interrupted_answer_escapes_as_a_host_type_error` (an `interruptSignal`
 * answer that isn't an `AbortSignal`).
 *
 * Changed: `test_run_context_receives_steering_only_when_steerable` can't replace `runAgent` (an
 * ESM export), so the inbox records its calls: a steerable agent's reach it, a non-steerable
 * agent's don't. `test_interrupt_after_a_reply_keeps_it_and_records_skipped_calls`: the event half
 * (the interrupt set once the reply finished; both calls skipped with brackets); the transcript
 * half Later (T6-6). `test_python_built_steerable_agent_with_structured_output_fails_before_the_model`:
 * the `run({})` half; its `eval_node` half Later (T3-22).
 *
 * Later: `test_steers_are_recorded_with_the_next_model_request` and every `memory=True` half
 * (T6-6, session memory); `test_interrupted_agent_in_a_loop_lets_the_loop_continue` (T4-10, LOOP);
 * `test_run_boundary_rejects_an_object_without_the_inbox_methods`,
 * `test_resume_reuses_a_finished_steered_node_without_calling_the_inbox` and
 * `test_retry_after_a_mid_node_kill_sees_the_recorded_steer` (T4-8, the run boundary and resume).
 *
 * Not ported: `test_inbox_satisfies_the_public_protocol` (it checks the test's inbox against a
 * `typing.Protocol`; TS checks `Steering` statically).
 *
 * TS-only: a steer an async `drain` promises is awaited before the model call; a drain that
 * resolves after the run stopped rejects the node with the stop's reason and sends no steer; a
 * steer queued between two calls of a reply joins after both results; a too-deep tool call fails
 * the node even with the interrupt set, since the argument check runs first. Interrupts around
 * concurrent tool groups are in `agent_tool_groups.test.ts`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { type CompiledFlow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallFinished, ModelCallStarted, RunFailed, RunSucceeded, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Steering } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, engine, graph, rejected, serving } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };

/** A host inbox: steers queued per node id, one interrupt signal per node; `calls` records each call. */
class Inbox implements Steering {
  readonly queued: Map<string, string[]>;
  readonly calls: [string, string][] = [];
  readonly #controllers = new Map<string, AbortController>();

  constructor(steers: Readonly<Record<string, readonly string[]>> = {}, interrupt: readonly string[] = []) {
    this.queued = new Map(Object.entries(steers).map(([nodeId, messages]) => [nodeId, [...messages]]));
    for (const nodeId of interrupt) this.interrupt(nodeId);
  }

  /** Interrupt node `nodeId`. */
  interrupt(nodeId: string): void {
    this.#controller(nodeId).abort(new Error("interrupted"));
  }

  interruptSignal(nodeId: string): AbortSignal {
    this.calls.push(["interruptSignal", nodeId]);
    return this.#controller(nodeId).signal;
  }

  drain(nodeId: string): readonly string[] | Promise<readonly string[]> {
    this.calls.push(["drain", nodeId]);
    const messages = this.queued.get(nodeId) ?? [];
    this.queued.delete(nodeId);
    return messages;
  }

  #controller(nodeId: string): AbortController {
    let controller = this.#controllers.get(nodeId);
    if (controller === undefined) this.#controllers.set(nodeId, (controller = new AbortController()));
    return controller;
  }
}

/** A reply with `text` (none when empty) then one tool call per `[name, arguments, id]`. */
function reply(text: string, ...calls: (readonly [string, JsonObject, string])[]): Response {
  const parts = [
    ...(text ? [new TextPart(text)] : []),
    ...calls.map(([name, args, id]) => new ToolCallPart({ id, name, arguments: args })),
  ];
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: calls.length > 0 ? "tool_calls" : "stop" });
}

/** Python's `_two_calls`: text and two `lookup` calls. */
function twoCalls(): Response {
  return reply("Checking both.", ["lookup", { topic: "a" }, "call-1"], ["lookup", { topic: "b" }, "call-2"]);
}

/** A `lookup` tool running `body` with the topic. */
function lookupTool(body: (topic: string) => string): FunctionTool {
  return new FunctionTool({
    name: "lookup",
    description: "Look up a topic.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => body(args.topic as string),
  });
}

/** One `chat` agent granted `tools`, output `${chat.output}`; with `after`, a summarizing agent reads it. */
function chatFlow(options: { steerable?: boolean; tools?: readonly string[]; after?: boolean } = {}): CompiledFlow {
  const { steerable = true, tools = [], after = false } = options;
  const chat = new AgentNode("chat", { prompt: "Help the user.", llmConfig: CONFIG, steerable, tools: [...tools] });
  if (!after) return graph([chat], [[START_ID, "chat"]], { outputs: [new FlowOutput("output", "${chat.output}")] });
  const next = new AgentNode("after", { prompt: "summarize: ${draft}", llmConfig: CONFIG });
  next.params = [new Parameter("draft")];
  return graph(
    [chat, next],
    [
      [START_ID, "chat"],
      ["chat", "after"],
    ],
    { wiring: { chat: {}, after: { draft: "${chat.output}" } }, outputs: [new FlowOutput("output", "${after.output}")] },
  );
}

/** Run `flow` with `model` and `options`; return the run's output, failing unless it succeeded. */
async function run(
  flow: CompiledFlow,
  model: ScriptedModel,
  options: { steering?: Steering; tools?: ToolSet; onEvent?: (event: unknown) => void; signal?: AbortSignal } = {},
): Promise<unknown> {
  const events = await engine(flow, { models: serving(model), ...options }).run();
  const last = events.at(-1);
  expect(last).toBeInstanceOf(RunSucceeded);
  return (last as RunSucceeded).output;
}

/** The `[role, text]` of each message. */
function shown(messages: readonly Message[]): [string, string][] {
  return messages.map((message) => [message.role, message.text()]);
}

/** The tool call brackets among `events`, as `[kind, callIndex, toolCallId, result or error]`. */
function toolEvents(events: readonly unknown[]): unknown[][] {
  return events.flatMap((event) => {
    if (event instanceof ToolCallStarted) return [["started", event.callIndex, event.toolCallId]];
    if (event instanceof ToolCallFinished) return [["finished", event.callIndex, event.toolCallId, event.error ?? event.result]];
    return [];
  });
}

// --- the capability reaches only steerable nodes ---------------------------- //

test("a non-steerable agent never calls the inbox", async () => {
  const model = new ScriptedModel([reply("done")], CONFIG);
  const steering = new Inbox({ chat: ["ignored"] }, ["chat"]);
  expect(await run(chatFlow({ steerable: false }), model, { steering })).toBe("done");
  expect(steering.calls).toEqual([]);
});

test("test_run_context_receives_steering_only_when_steerable (Changed: the inbox records its calls)", async () => {
  for (const steerable of [true, false]) {
    const steering = new Inbox();
    const model = new ScriptedModel([reply("ok"), reply("ok")], CONFIG);
    const node = new AgentNode("chat", { prompt: "hi", llmConfig: CONFIG, steerable });
    await drive(node, { steering, models: serving(model) });
    await drive(node, { models: serving(model) });
    expect(steering.calls).toEqual(
      steerable
        ? [
            ["interruptSignal", "chat"],
            ["drain", "chat"],
          ]
        : [],
    );
  }
});

test("python built steerable agent with structured output fails before the model (run half)", async () => {
  const contract = new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["answer", Type.scalar(ValueKind.INTEGER)]]),
      required: new Set(["answer"]),
    }),
  );
  const llmConfig = { provider: "anthropic", model: "claude-test" };
  const node = new AgentNode("chat", { prompt: "hi", llmConfig, steerable: true });
  node.outputContract = contract;
  const result = node.run({});
  await expect(result).rejects.toBeInstanceOf(RangeError);
  await expect(result).rejects.toThrow("AGENT node 'chat': steerable needs `str` output");
});

// --- steer messages ----------------------------------------------------------- //

test("steers are sent as user messages at the top of each iteration", async () => {
  const steering = new Inbox({ chat: ["use dataset B"] });
  const lookup = lookupTool((topic) => {
    steering.queued.set("chat", ["only the 2024 rows", "and cite them"]);
    return `found ${topic}`;
  });
  const model = new ScriptedModel([reply("", ["lookup", { topic: "a" }, "call-1"]), reply("done")], CONFIG);
  const tools = new ToolSet({ lookup });
  expect(await run(chatFlow({ tools: ["lookup"] }), model, { steering, tools })).toBe("done");
  expect(shown(model.requests[0]!.messages).slice(-2)).toEqual([
    ["user", "Help the user."],
    ["user", "use dataset B"],
  ]);
  expect(shown(model.requests[1]!.messages).slice(-3)).toEqual([
    ["tool", ""],
    ["user", "only the 2024 rows"],
    ["user", "and cite them"],
  ]);
  expect(model.requests[1]!.messages.at(-3)!.parts[0]).toMatchObject({ content: "found a" });
  expect(steering.calls.filter(([method]) => method === "drain")).toEqual([
    ["drain", "chat"],
    ["drain", "chat"],
  ]);
});

test("a steer an async drain promises is awaited before the model call", async () => {
  const steering = new Inbox();
  steering.drain = (nodeId) => {
    steering.calls.push(["drain", nodeId]);
    return new Promise((resolve) => setTimeout(() => resolve(["late steer"]), 5));
  };
  const model = new ScriptedModel([reply("done")], CONFIG);
  expect(await run(chatFlow(), model, { steering })).toBe("done");
  expect(shown(model.requests[0]!.messages).slice(-1)).toEqual([["user", "late steer"]]);
});

test("a drain that resolves after the run stopped rejects with the stop and sends no steer", async () => {
  const stop = new AbortController();
  const reason = new Error("stopped");
  const drained = Promise.withResolvers<readonly string[]>();
  const draining = Promise.withResolvers<void>();
  const steering: Steering = {
    interruptSignal: () => new AbortController().signal,
    drain: () => {
      draining.resolve();
      return drained.promise;
    },
  };
  const model = new ScriptedModel([reply("done")], CONFIG);
  const ctx: AgentRunContext = {
    nodeId: "chat",
    prompt: "go",
    tools: [],
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    signal: stop.signal,
    memoryContext: new MemoryContext(),
    toolset: null,
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    steering,
    models: serving(model),
    sent: () => [],
  };
  const running = runAgent(ctx);
  await draining.promise;
  stop.abort(reason);
  drained.resolve(["too late"]);
  expect(await rejected(running)).toBe(reason);
  expect(model.requests).toEqual([]);
});

test("a steer queued between two calls of a reply joins after both results", async () => {
  const steering = new Inbox();
  let calls = 0;
  const lookup = lookupTool((topic) => {
    if (++calls === 1) steering.queued.set("chat", ["mid-turn steer"]);
    return `found ${topic}`;
  });
  const model = new ScriptedModel([twoCalls(), reply("done")], CONFIG);
  expect(await run(chatFlow({ tools: ["lookup"] }), model, { steering, tools: new ToolSet({ lookup }) })).toBe("done");
  expect(model.requests[1]!.messages.slice(-4).map((message) => message.role)).toEqual(["assistant", "tool", "tool", "user"]);
  expect(model.requests[1]!.messages.at(-1)!.text()).toBe("mid-turn steer");
});

class HostBroken extends Error {
  override name = "HostBroken";
}

for (const [label, drain] of [
  [
    "a sync throw",
    () => {
      throw new HostBroken("inbox down");
    },
  ],
  ["a rejected promise", () => Promise.reject(new HostBroken("inbox down"))],
] as const) {
  test(`an inbox failure escapes as a host fault (${label})`, async () => {
    const steering = new Inbox();
    steering.drain = drain;
    const model = new ScriptedModel([reply("done")], CONFIG);
    const error = await rejected(engine(chatFlow(), { steering, models: serving(model) }).run());
    expect(error).toBeInstanceOf(HostBroken);
    expect(model.requests).toEqual([]);
  });
}

for (const [label, value] of [
  ["a string", "one message"],
  ["a non-string item", [1]],
  ["a promised string", Promise.resolve("one message")],
] as const) {
  test(`malformed inbox answers escape as host type errors (${label})`, async () => {
    const steering = new Inbox();
    steering.drain = () => value as never;
    const node = new AgentNode("chat", { prompt: "hi", llmConfig: CONFIG, steerable: true });
    const model = new ScriptedModel([reply("done")], CONFIG);
    const error = await rejected(drive(node, { steering, models: serving(model) }));
    expect(error).toEqual(new TypeError("steering.drain must return an array of strings"));
    expect(model.requests).toEqual([]);
  });
}

// --- interrupts at loop safe points ------------------------------------------ //

test("an interrupt before the first model call returns empty text", async () => {
  const model = new ScriptedModel([], CONFIG);
  const steering = new Inbox({ chat: ["never read"] }, ["chat"]);
  expect(await run(chatFlow(), model, { steering })).toBe("");
  expect(model.requests).toEqual([]);
  expect(steering.queued).toEqual(new Map([["chat", ["never read"]]]));
});

test("an interrupt after a reply keeps it and skips its calls (event half)", async () => {
  const steering = new Inbox();
  const ran: string[] = [];
  const events: unknown[] = [];
  const onEvent = (event: unknown) => {
    events.push(event);
    if (event instanceof ModelCallFinished) steering.interrupt("chat");
  };
  const lookup = lookupTool((topic) => {
    ran.push(topic);
    return "found";
  });
  const model = new ScriptedModel([twoCalls(), reply("next run")], CONFIG);
  const output = await run(chatFlow({ tools: ["lookup"] }), model, { steering, tools: new ToolSet({ lookup }), onEvent });
  expect(output).toBe("Checking both.");
  expect(ran).toEqual([]);
  expect(model.requests).toHaveLength(1);
  expect(toolEvents(events)).toEqual([
    ["started", 0, "call-1"],
    ["finished", 0, "call-1", "interrupted"],
    ["started", 1, "call-2"],
    ["finished", 1, "call-2", "interrupted"],
  ]);
});

test("an interrupt before a tool call skips only the remaining calls", async () => {
  const steering = new Inbox();
  const ran: string[] = [];
  const lookup = lookupTool((topic) => {
    ran.push(topic);
    steering.interrupt("chat");
    return "found";
  });
  const model = new ScriptedModel([twoCalls()], CONFIG);
  expect(await run(chatFlow({ tools: ["lookup"] }), model, { steering, tools: new ToolSet({ lookup }) })).toBe(
    "Checking both.",
  );
  expect(ran).toEqual(["a"]);
  expect(model.requests).toHaveLength(1);
});

test("skipped tool calls reach the host as interrupted tool events", async () => {
  const steering = new Inbox();
  const events: unknown[] = [];
  const lookup = lookupTool((topic) => {
    steering.interrupt("chat");
    return `found ${topic}`;
  });
  const model = new ScriptedModel([twoCalls()], CONFIG);
  await run(chatFlow({ tools: ["lookup"] }), model, {
    steering,
    tools: new ToolSet({ lookup }),
    onEvent: (event) => void events.push(event),
  });
  const brackets = events.filter((event) => event instanceof ToolCallStarted || event instanceof ToolCallFinished);
  const exchangeId = (brackets[0] as ToolCallStarted).exchangeId;
  const ids = (callIndex: number) => ({ nodeId: "chat", exchangeId, callIndex, toolCallId: `call-${callIndex + 1}`, name: "lookup" });
  expect(brackets).toEqual([
    new ToolCallStarted({ ...ids(0), arguments: { topic: "a" } }),
    new ToolCallFinished({ ...ids(0), result: "found a" }),
    new ToolCallStarted({ ...ids(1), arguments: { topic: "b" } }),
    new ToolCallFinished({ ...ids(1), error: "interrupted" }),
  ]);
});

test("an interrupt at the top of a later iteration returns the latest reply", async () => {
  const steering = new Inbox();
  const lookup = lookupTool(() => {
    steering.queued.set("chat", ["too late"]);
    steering.interrupt("chat");
    return "found";
  });
  // The interrupt lands after the only call, so the next top-of-loop check ends it.
  const model = new ScriptedModel([reply("Looking.", ["lookup", { topic: "a" }, "call-1"])], CONFIG);
  expect(await run(chatFlow({ tools: ["lookup"] }), model, { steering, tools: new ToolSet({ lookup }) })).toBe("Looking.");
  expect(model.requests).toHaveLength(1);
  expect(steering.queued).toEqual(new Map([["chat", ["too late"]]]));
});

test("an interrupted node succeeds and the flow continues", async () => {
  const steering = new Inbox({}, ["chat"]);
  const model = new ScriptedModel([reply("summary")], CONFIG);
  expect(await run(chatFlow({ after: true }), model, { steering })).toBe("summary");
  expect(shown(model.requests[0]!.messages).at(-1)).toEqual(["user", "summarize: "]);
  // The downstream agent is not steerable, so the inbox only ever saw `chat`.
  expect(new Set(steering.calls.map(([, nodeId]) => nodeId))).toEqual(new Set(["chat"]));
});

test("a malformed interruptSignal answer escapes as a host type error", async () => {
  const steering = new Inbox();
  steering.interruptSignal = () => "yes" as never;
  const node = new AgentNode("chat", { prompt: "hi", llmConfig: CONFIG, steerable: true });
  const model = new ScriptedModel([reply("done")], CONFIG);
  const error = await rejected(drive(node, { steering, models: serving(model) }));
  expect(error).toEqual(new TypeError("steering.interruptSignal must return an AbortSignal"));
  expect(model.requests).toEqual([]);
});

test("an interrupt for a finished node is ignored", async () => {
  const steering = new Inbox();
  let started = 0;
  const onEvent = (event: unknown) => {
    if (event instanceof ModelCallStarted && ++started === 2) steering.interrupt("chat");
  };
  const model = new ScriptedModel([reply("chat done"), reply("summary")], CONFIG);
  expect(await run(chatFlow({ after: true }), model, { steering, onEvent })).toBe("summary");
  expect(shown(model.requests[1]!.messages).at(-1)).toEqual(["user", "summarize: chat done"]);
});

test("a too-deep tool call fails the node even with the interrupt set", async () => {
  const steering = new Inbox();
  const ran: string[] = [];
  const lookup = lookupTool((topic) => {
    ran.push(topic);
    return "found";
  });
  const deep = '{"a": '.repeat(100) + '"leaf"' + "}".repeat(100);
  const call = new ToolCallPart({ id: "call-1", name: "lookup", arguments: null, rawArguments: `{"x": ${deep}}` });
  const model = new ScriptedModel(
    [new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" })],
    CONFIG,
  );
  const onEvent = (event: unknown) => {
    if (event instanceof ModelCallStarted) steering.interrupt("chat");
  };
  const events = await engine(chatFlow({ tools: ["lookup"] }), {
    steering,
    models: serving(model),
    tools: new ToolSet({ lookup }),
    onEvent,
  }).run();
  const last = events.at(-1);
  expect(last).toBeInstanceOf(RunFailed);
  expect((last as RunFailed).failure.code).toBe("node.execute");
  expect((last as RunFailed).failure.message).toContain("tool call 'lookup' arguments at x");
  expect(ran).toEqual([]);
});
