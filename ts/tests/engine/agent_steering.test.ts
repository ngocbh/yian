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
 * `test_interrupt_before_a_tool_call_skips_only_the_remaining_calls`,
 * `test_skipped_tool_calls_reach_the_host_as_interrupted_tool_events`,
 * `test_interrupt_at_the_top_of_a_later_iteration_returns_the_latest_reply`,
 * `test_interrupted_node_succeeds_and_the_flow_continues`,
 * `test_interrupt_for_a_finished_node_is_ignored`,
 * `test_interrupt_mid_stream_keeps_the_partial_text_and_ends_the_stream`,
 * `test_interrupted_stream_reports_the_usage_received_so_far`,
 * `test_interrupt_on_the_last_chunk_keeps_the_complete_reply` (one usage, the call skipped with
 * brackets) and `test_interrupt_during_a_single_chunk_reply_keeps_the_complete_reply`
 * (a `Response` reply, the interrupt set on `ModelCallStarted`: its text is the attempt's first
 * event, so there is no later text to cut). A reply's events are listed where Python's chat
 * streams chunks; the interrupt is checked before each non-empty text delta after an attempt's
 * first event.
 *
 * Ported through `evalNode` (`drive`): `test_malformed_inbox_answers_escape_as_host_type_errors`
 * and `test_malformed_interrupted_answer_escapes_as_a_host_type_error` (an `interruptSignal`
 * answer that isn't an `AbortSignal`).
 *
 * Changed: `test_run_context_receives_steering_only_when_steerable` can't replace `runAgent` (an
 * ESM export), so the inbox records its calls: a steerable agent's reach it, a non-steerable
 * agent's don't. `test_interrupt_after_a_reply_keeps_it_and_records_skipped_calls`: the interrupt
 * is set once the reply finished, and both calls are also skipped with brackets.
 * `test_python_built_steerable_agent_with_structured_output_fails_before_the_model`:
 * the output contract is built directly where Python reads it from a loaded flow; the evaluator
 * half goes through `drive`.
 * `test_interrupt_mid_stream_in_a_pooled_run_keeps_partial_text` runs with `numWorkers: 2`; the
 * model pauses after its second word until the host saw it, as Python's thread does.
 * `test_interrupted_agent_in_a_loop_lets_the_loop_continue` loads its flow and runs `runFlow`, with
 * the model and its provider passed in (`models`, `llmConfig`) where Python patches
 * `model_from_config`; the fold is inline code, as the fixtures have no `reply_fold`.
 *
 * Ported through `loadFlow`, `runFlow` and `resumeFlow` (`runWith`, `resumeWith`), Python's
 * `_flow` loaded (`steerFlow`): `test_run_boundary_rejects_an_object_without_the_inbox_methods`,
 * `test_resume_reuses_a_finished_steered_node_without_calling_the_inbox`. Changed: TS steering is push-based
 * (design 6.8: `Steering` in `src/nodes/base.ts`, checked by `requireSteering` in
 * `src/runtime/eval_node.ts`), so the boundary wants `interruptSignal` and `drain` and its message
 * names both (Python: "steering must provide drain"); a crash is a frozen log (`CrashAt`,
 * `crashes`) where Python's model raises `_SimulatedCrash`, so no record of the crashed
 * request's outcome reaches the log either way.
 *
 * Not ported: `test_inbox_satisfies_the_public_protocol` (it checks the test's inbox against a
 * `typing.Protocol`; TS checks `Steering` statically); the session rows,
 * `test_steers_are_recorded_with_the_next_model_request`,
 * `test_retry_after_a_mid_node_kill_sees_the_recorded_steer` and the `memory=True` rows above (TS
 * yian has no session memory, so a steer read before a crash is lost and the retry doesn't see it).
 *
 * TS-only: a steer an async `drain` promises is awaited before the model call; a drain that
 * resolves after the run stopped rejects the node with the stop's reason and sends no steer; an
 * interrupt that lands during an async drain returns `""` before any model call; a
 * steer queued between two calls of a reply joins after both results; a too-deep tool call fails
 * the node even with the interrupt set, since the argument check runs first; a cut with no stream
 * sink keeps the text received, and closes the stream; a cut reply's tool call never runs; a
 * non-text event (an empty text delta, usage, a tool-call delta, reasoning, the finish) after the
 * interrupt cuts nothing; a stop that lands after a reply was cut, before the loop sees the cut,
 * wins over its text; at the `callModel` level, an interrupt during a model-retry wait, or already
 * set when it begins, ends the wait at once with an empty `ReplyInterrupted` and no second
 * attempt, leaving no sleep pending, and a stop during the wait still rejects with the stop;
 * through the engine, an interrupt during a later model call's retry wait (on the run's scheduler)
 * ends the loop with `""` and no retry attempt, leaving no sleep pending.
 * Interrupts around concurrent tool groups are in `agent_tool_groups.test.ts`.
 */

import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import { type CompiledFlow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import {
  ModelCallFinished,
  ModelCallStarted,
  RunFailed,
  RunSucceeded,
  StreamChunk,
  TokenUsage,
  ToolCallFinished,
  ToolCallStarted,
} from "../../src/events.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import {
  type ChatModel,
  type Event,
  EventStream,
  Finish,
  ReasoningDelta,
  TextDelta,
  ToolCallDelta,
  UsageReport,
} from "../../src/llm/events.ts";
import { Message, Request, Response, TextPart, ToolCallPart, Usage } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { ReplyInterrupted } from "../../src/nodes/agent/interrupts.ts";
import { callModel, liveSinks } from "../../src/nodes/agent/live.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import type { Steering } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import type { Scheduler } from "../../src/platform.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { CrashAt, crashes, drive, engine, graph, NO_ENV, rejected, resumeWith, runWith, serving } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

// One directory for the run logs of the boundary and resume tests, each under its own name.
const dir = mkdtempSync(join(tmpdir(), "agent-steering-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

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
  const finishReason = calls.length > 0 ? "tool_calls" : "stop";
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason });
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

/**
 * Python's `_flow`, loaded: one `chat` agent granted `tools`, output `${chat.output}`; with
 * `after`, a summarizing agent reads it.
 */
function steerFlow(options: { tools?: readonly string[]; after?: boolean } = {}): string {
  const { tools = [], after = false } = options;
  return (
    "id: steer\nname: steer\nllm_config: {provider: anthropic, model: claude-test}\nnodes:\n" +
    "  chat:\n    kind: agent\n    steerable: true\n" +
    `    tools: [${tools.join(", ")}]\n    prompt: Help the user.\n    output: str\n` +
    (after
      ? '  after:\n    kind: agent\n    input: {draft: "${chat.output}"}\n    prompt: "summarize: ${draft}"\n' +
        "    output: str\noutput: ${after.output}\n"
      : "output: ${chat.output}\n")
  );
}

/** Run `flow` with `model` and `options`; return the run's output, failing unless it succeeded. */
async function run(
  flow: CompiledFlow,
  model: ChatModel,
  options: {
    steering?: Steering;
    tools?: ToolSet;
    onEvent?: (event: unknown) => void;
    onStream?: (chunk: StreamChunk) => void;
    numWorkers?: number;
    scheduler?: Scheduler;
  } = {},
): Promise<unknown> {
  const last = (await engine(flow, { models: serving(model), ...options }).run()).at(-1);
  expect(last).toBeInstanceOf(RunSucceeded);
  return (last as RunSucceeded).output;
}

/** A direct loop context: agent `chat` prompted "go" with no tools, calling `model`; `extra` overrides. */
function agentContext(model: ChatModel, extra: Partial<AgentRunContext>): AgentRunContext {
  return {
    nodeId: "chat",
    messages: [userMessage("go")],
    promptText: "go",
    tools: [],
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    memoryContext: new MemoryContext(),
    toolset: null,
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: serving(model),
    sent: () => [],
    ...extra,
  };
}

/** The `[role, text]` of each message. */
function shown(messages: readonly Message[]): [string, string][] {
  return messages.map((message) => [message.role, message.text()]);
}

/** The tool call brackets among `events`, as `[kind, callIndex, toolCallId, result or error]`. */
function toolEvents(events: readonly unknown[]): unknown[][] {
  return events.flatMap((event) => {
    if (event instanceof ToolCallStarted) return [["started", event.callIndex, event.toolCallId]];
    if (event instanceof ToolCallFinished) {
      return [["finished", event.callIndex, event.toolCallId, event.error ?? event.result]];
    }
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

test("python built steerable agent with structured output fails before the model", async () => {
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
  // Malformed node setup escapes the evaluator, as for any other kind.
  const model = new ScriptedModel([], CONFIG);
  const evaluated = drive(node, { steering: new Inbox({}, ["chat"]), models: serving(model) });
  await expect(evaluated).rejects.toThrow("AGENT node 'chat': steerable needs `str` output");
  expect(model.requests).toEqual([]);
});

test.each([
  ["an object", {}],
  ["drain only", { drain: () => [] }],
])("the run boundary rejects an object without the inbox methods: %s", async (_name, bad) => {
  const log = join(dir, "boundary.jsonl");
  const error = await rejected(runFlow(await loadFlow(steerFlow()), {}, { steering: bad as never, log }));
  expect(error).toEqual(new TypeError("steering must provide interruptSignal(nodeId) and drain(nodeId), or be null"));
  expect(existsSync(log)).toBe(false);
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
    // Settles a few microtasks later, so the loop must await it.
    return Promise.resolve().then(() => Promise.resolve(["late steer"]));
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
  const running = runAgent(agentContext(model, { signal: stop.signal, steering }));
  await draining.promise;
  stop.abort(reason);
  drained.resolve(["too late"]);
  expect(await rejected(running)).toBe(reason);
  expect(model.requests).toEqual([]);
});

test("an interrupt that lands during an async drain returns before any model call", async () => {
  const interrupt = new AbortController();
  const drained = Promise.withResolvers<readonly string[]>();
  const draining = Promise.withResolvers<void>();
  const steering: Steering = {
    interruptSignal: () => interrupt.signal,
    drain: () => {
      draining.resolve();
      return drained.promise;
    },
  };
  const model = new ScriptedModel([reply("done")], CONFIG);
  const running = runAgent(agentContext(model, { steering }));
  await draining.promise;
  interrupt.abort();
  drained.resolve(["too late"]);
  expect((await running).value).toBe("");
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
  const roles = model.requests[1]!.messages.slice(-4).map((message) => message.role);
  expect(roles).toEqual(["assistant", "tool", "tool", "user"]);
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

test("an interrupt after a reply keeps it and records skipped calls", async () => {
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
  const model = new ScriptedModel([twoCalls()], CONFIG);
  const flow = chatFlow({ tools: ["lookup"] });
  const output = await run(flow, model, { steering, tools: new ToolSet({ lookup }), onEvent });
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
  const flow = chatFlow({ tools: ["lookup"] });
  expect(await run(flow, model, { steering, tools: new ToolSet({ lookup }) })).toBe("Checking both.");
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
  const ids = (callIndex: number) => ({
    nodeId: "chat",
    exchangeId,
    callIndex,
    toolCallId: `call-${callIndex + 1}`,
    name: "lookup",
  });
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
  const output = await run(chatFlow({ tools: ["lookup"] }), model, { steering, tools: new ToolSet({ lookup }) });
  expect(output).toBe("Looking.");
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

// Python's `test_agent_in_loop.FLOW` with a steerable agent: two LOOP turns, each an AGENT reply
// folded into the transcript.
const LOOP_FLOW = `
id: t
name: t
input:
  seed: !str
defs:
  body:
    input:
      transcript: !str
      exited: !bool
    nodes:
      reply:
        kind: agent
        steerable: true
        tools: [ping]
        input:
          transcript: \${input.transcript}
        output: str
        prompt: "history \${transcript}"
      fold:
        kind: code
        code: |
          return { transcript: inputs.transcript + "|" + inputs.reply, exited: inputs.exited }
        input:
          transcript: \${input.transcript}
          reply:      \${reply.output}
          exited:     \${input.exited}
        output:
          transcript: str
          exited: bool
    output: \${fold.output}
nodes:
  loop:
    kind: loop
    call: body
    input:
      transcript: \${input.seed}
      exited: false
    times: 2
output: \${loop.output.transcript}
`;

test("an interrupted agent in a loop lets the loop continue", async () => {
  const steering = new Inbox();
  let started = 0;
  // The first occurrence is interrupted during its model call, so its ping call is skipped.
  const onEvent = (event: unknown) => {
    if (event instanceof ModelCallStarted && ++started === 1) steering.interrupt(event.nodeId);
  };
  const ping = new FunctionTool({
    name: "ping",
    description: "Ping.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "pong",
  });
  const model = new ScriptedModel([reply("partial", ["ping", {}, "p-1"]), reply("turn2")], CONFIG);
  const result = await runFlow(await loadFlow(LOOP_FLOW), { seed: "s" }, {
    steering,
    tools: { ping },
    onEvent,
    llmConfig: CONFIG,
    models: serving(model),
    env: NO_ENV,
  });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("s|partial|turn2");
  // Each LOOP occurrence has its own node id, so the interrupt stayed with the first.
  expect(new Set(steering.calls.map(([, nodeId]) => nodeId)).size).toBe(2);
});

// --- resume ----------------------------------------------------------------- //

test("resume reuses a finished steered node without calling the inbox", async () => {
  const log = join(dir, "resume.jsonl");
  const flow = await loadFlow(steerFlow({ after: true }));
  const first = new CrashAt([reply("chat done"), reply("never recorded")], 2, CONFIG);
  await crashes(first, () => runWith(flow, {}, null, first, { steering: new Inbox({ chat: ["use dataset B"] }), log }));
  expect(shown(first.requests[0]!.messages).at(-1)).toEqual(["user", "use dataset B"]);

  const model = new ScriptedModel([reply("summary")], CONFIG);
  const steering = new Inbox({ chat: ["new steer"] }, ["chat"]);
  const resumed = await resumeWith(flow, log, null, model, { steering });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("summary");
  expect(shown(model.requests[0]!.messages).at(-1)).toEqual(["user", "summarize: chat done"]);
  expect(steering.calls).toEqual([]);
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

// --- the stream is a safe point ------------------------------------------------ //

/** A reply's events: one text delta per word, then `rest`. */
function words(...items: (string | Event)[]): Event[] {
  return items.map((item) => (typeof item === "string" ? new TextDelta(0, item) : item));
}

/** The `[chunk, final]` of each stream chunk. */
function chunked(chunks: readonly StreamChunk[]): [string, boolean][] {
  return chunks.map((chunk) => [chunk.chunk, chunk.final]);
}

/** A model streaming `events` once, calling `after(index)` once the consumer took event `index`. */
class Paced implements ChatModel {
  readonly provider = "anthropic";
  readonly model = "claude-test";
  /** The events handed out so far. */
  readonly sent: Event[] = [];
  closed = 0;
  readonly #events: readonly Event[];
  readonly #after: (index: number) => void | Promise<void>;

  constructor(events: readonly Event[], after: (index: number) => void | Promise<void>) {
    this.#events = events;
    this.#after = after;
  }

  stream(): EventStream {
    const events = this.#events;
    const after = this.#after;
    const sent = this.sent;
    async function* paced(): AsyncGenerator<Event> {
      for (const [index, event] of events.entries()) {
        sent.push(event);
        yield event;
        await after(index);
      }
    }
    return new EventStream(paced(), () => void this.closed++);
  }
}

const LOOKUP_CALL = new ToolCallDelta({ index: 1, id: "call-1", name: "lookup", arguments: '{"topic": "a"}' });

test("an interrupt mid-stream keeps the partial text and ends the stream", async () => {
  const model = new ScriptedModel(
    [words("hello", " big", " world", LOOKUP_CALL, new Finish({ reason: "tool_calls" }))],
    CONFIG,
  );
  const steering = new Inbox();
  const chunks: StreamChunk[] = [];
  const events: unknown[] = [];
  const ran: string[] = [];
  const onStream = (chunk: StreamChunk) => {
    chunks.push(chunk);
    if (chunk.chunk === " big") steering.interrupt("chat");
  };
  const lookup = lookupTool((topic) => {
    ran.push(topic);
    return "found";
  });
  const output = await run(chatFlow({ tools: ["lookup"] }), model, {
    steering,
    tools: new ToolSet({ lookup }),
    onStream,
    onEvent: (event) => void events.push(event),
  });
  expect(output).toBe("hello big");
  expect(ran).toEqual([]);
  expect(chunked(chunks)).toEqual([
    ["hello", false],
    [" big", false],
    ["", true],
  ]);
  const finished = events.filter((event) => event instanceof ModelCallFinished);
  expect(finished).toHaveLength(1);
  expect([finished[0]!.error, finished[0]!.usage, finished[0]!.attempt]).toEqual([null, null, 1]);
  expect(toolEvents(events)).toEqual([]);
  // The cut closes the provider stream.
  expect(model.closed).toBe(1);
});

test("an interrupted stream reports the usage received so far", async () => {
  const usage = new Usage({ inputTokens: 1000, outputTokens: 1, totalTokens: 1001 });
  const streamed = [new UsageReport(usage), ...words("hello", " big", " world", new Finish({ reason: "stop" }))];
  const model = new ScriptedModel([streamed], CONFIG);
  const steering = new Inbox();
  const events: unknown[] = [];
  const onStream = (chunk: StreamChunk) => {
    if (chunk.chunk === " big") steering.interrupt("chat");
  };
  const output = await run(chatFlow(), model, { steering, onStream, onEvent: (event) => void events.push(event) });
  expect(output).toBe("hello big");
  const finished = events.filter((event) => event instanceof ModelCallFinished);
  expect(finished.map((event) => [event.usage, event.error])).toEqual([
    [new TokenUsage({ inputTokens: 1000, outputTokens: 1, totalTokens: 1001 }), null],
  ]);
});

test("an interrupt mid-stream in a pooled run keeps partial text", async () => {
  const all = ["hello", " big", " wide", " world"];
  const shown = Promise.withResolvers<void>();
  // Pause after the second word until the host has seen it.
  const model = new Paced(words(...all, new Finish({ reason: "stop" })), (index) => {
    return index === 1 ? shown.promise : undefined;
  });
  const steering = new Inbox();
  const onStream = (chunk: StreamChunk) => {
    if (chunk.chunk === " big") {
      steering.interrupt("chat");
      shown.resolve();
    }
  };
  const output = (await run(chatFlow(), model, { steering, onStream, numWorkers: 2 })) as string;
  const full = all.join("");
  expect(output.length).toBeGreaterThan(0);
  expect(full.startsWith(output) && output !== full).toBe(true);
  expect(model.sent.length).toBeLessThan(all.length);
});

test("an interrupt on the last chunk keeps the complete reply", async () => {
  const usage = new Usage({ inputTokens: 500, outputTokens: 20, totalTokens: 520 });
  const model = new ScriptedModel(
    [
      [
        new ToolCallDelta({ index: 0, id: "call-1", name: "lookup", arguments: '{"topic": "a"}' }),
        new UsageReport(usage),
        new TextDelta(1, "ok"),
        new TextDelta(1, " go"),
        new Finish({ reason: "tool_calls" }),
      ],
    ],
    CONFIG,
  );
  const steering = new Inbox();
  const ran: string[] = [];
  const events: unknown[] = [];
  const onStream = (chunk: StreamChunk) => {
    // The reply's last text chunk.
    if (chunk.chunk === " go") steering.interrupt("chat");
  };
  const lookup = lookupTool((topic) => {
    ran.push(topic);
    return "found";
  });
  const output = await run(chatFlow({ tools: ["lookup"] }), model, {
    steering,
    tools: new ToolSet({ lookup }),
    onStream,
    onEvent: (event) => void events.push(event),
  });
  expect(output).toBe("ok go");
  expect(ran).toEqual([]);
  expect(events.filter((event) => event instanceof ModelCallFinished).map((event) => event.usage)).toEqual([
    new TokenUsage({ inputTokens: 500, outputTokens: 20, totalTokens: 520 }),
  ]);
  expect(toolEvents(events)).toEqual([
    ["started", 0, "call-1"],
    ["finished", 0, "call-1", "interrupted"],
  ]);
});

test("an interrupt during a single-chunk reply keeps the complete reply", async () => {
  const steering = new Inbox();
  const ran: string[] = [];
  const events: unknown[] = [];
  const onEvent = (event: unknown) => {
    events.push(event);
    if (event instanceof ModelCallStarted) steering.interrupt("chat");
  };
  const lookup = lookupTool((topic) => {
    ran.push(topic);
    return "found";
  });
  const model = new ScriptedModel([twoCalls()], CONFIG);
  const output = await run(chatFlow({ tools: ["lookup"] }), model, {
    steering,
    tools: new ToolSet({ lookup }),
    onEvent,
    onStream: () => {},
  });
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

test("a cut with no stream sink still keeps the text received", async () => {
  const steering = new Inbox();
  const model = new Paced(words("hello", " big", " world", new Finish({ reason: "stop" })), (index) => {
    if (index === 1) steering.interrupt("chat");
  });
  expect(await run(chatFlow(), model, { steering })).toBe("hello big");
  expect(model.closed).toBe(1);
});

test("a stop that lands after a cut, before the loop sees it, wins over the cut text", async () => {
  const stop = new AbortController();
  const reason = new Error("stopped");
  const steering = new Inbox();
  // The cut's `ModelCallFinished` is emitted after the reply was cut and before the loop sees it.
  const emitEvent = (event: unknown) => {
    if (event instanceof ModelCallStarted) steering.interrupt("chat");
    if (event instanceof ModelCallFinished) stop.abort(reason);
  };
  const model = new ScriptedModel([words("hello", " world", new Finish({ reason: "stop" }))], CONFIG);
  expect(await rejected(runAgent(agentContext(model, { signal: stop.signal, steering, emitEvent })))).toBe(reason);
});

test("a partial reply's tool call never runs", async () => {
  const steering = new Inbox();
  const ran: string[] = [];
  const events: unknown[] = [];
  const model = new ScriptedModel(
    [[new TextDelta(0, "hello"), LOOKUP_CALL, new TextDelta(2, " big"), new Finish({ reason: "tool_calls" })]],
    CONFIG,
  );
  const lookup = lookupTool((topic) => {
    ran.push(topic);
    return "found";
  });
  const output = await run(chatFlow({ tools: ["lookup"] }), model, {
    steering,
    tools: new ToolSet({ lookup }),
    onStream: (chunk) => void (chunk.chunk === "hello" && steering.interrupt("chat")),
    onEvent: (event) => void events.push(event),
  });
  expect(output).toBe("hello");
  expect(ran).toEqual([]);
  expect(toolEvents(events)).toEqual([]);
});

test("a non-text event after the interrupt doesn't cut the reply", async () => {
  const steering = new Inbox();
  const ran: string[] = [];
  const events: unknown[] = [];
  const usage = new Usage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 });
  const model = new ScriptedModel(
    [
      [
        new TextDelta(0, "hello"),
        new TextDelta(0, ""),
        new UsageReport(usage),
        LOOKUP_CALL,
        new ReasoningDelta(2, "thinking"),
        new Finish({ reason: "tool_calls" }),
      ],
    ],
    CONFIG,
  );
  const lookup = lookupTool((topic) => {
    ran.push(topic);
    return "found";
  });
  const output = await run(chatFlow({ tools: ["lookup"] }), model, {
    steering,
    tools: new ToolSet({ lookup }),
    onStream: (chunk) => void (chunk.chunk === "hello" && steering.interrupt("chat")),
    onEvent: (event) => void events.push(event),
  });
  expect(output).toBe("hello");
  expect(ran).toEqual([]);
  expect(events.filter((event) => event instanceof ModelCallFinished).map((event) => event.usage)).toEqual([
    new TokenUsage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 }),
  ]);
  expect(toolEvents(events)).toEqual([
    ["started", 0, "call-1"],
    ["finished", 0, "call-1", "interrupted"],
  ]);
});

/**
 * A scheduler whose sleeps wait only on their signal (one already aborted rejects at once);
 * `pending` counts the sleeps not settled.
 */
function waitingScheduler(): Scheduler & { pending: number; slept: Promise<void> } {
  const slept = Promise.withResolvers<void>();
  const scheduler = {
    pending: 0,
    slept: slept.promise,
    now: () => 0,
    sleep: (_ms: number, signal?: AbortSignal) =>
      new Promise<void>((_resolve, reject) => {
        if (signal?.aborted) return reject(signal.reason);
        scheduler.pending++;
        signal?.addEventListener(
          "abort",
          () => {
            scheduler.pending--;
            reject(signal.reason);
          },
          { once: true },
        );
        slept.resolve();
      }),
  };
  return scheduler;
}

/** The `[type, error]` of each model call event. */
function typesAndErrors(events: readonly unknown[]): [string, string | null][] {
  return events.map((event) => [(event as { type: string }).type, (event as ModelCallFinished).error ?? null]);
}

const REQUEST = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("hi")] })] });

test("an interrupt during a model-retry wait ends the wait at once with an empty reply", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const events: unknown[] = [];
  const live = liveSinks({ nodeId: "chat", llmConfig: CONFIG, emitEvent: (event) => void events.push(event) });
  const model = new ScriptedModel([new ProviderError("server", "boom", { retryAfter: 60 }), reply("never")], CONFIG);
  const scheduler = waitingScheduler();
  const interrupt = new AbortController();
  const options = { purpose: "chat", exchangeId: "x1", interrupt: interrupt.signal, retry: { scheduler } } as const;
  const call = callModel(live, model, REQUEST, options);
  await scheduler.slept;
  interrupt.abort(new Error("interrupted"));
  const cut = await rejected(call);
  expect(cut).toBeInstanceOf(ReplyInterrupted);
  expect((cut as ReplyInterrupted).reply.text()).toBe("");
  expect((cut as ReplyInterrupted).reply.parts).toEqual([]);
  expect((cut as ReplyInterrupted).usage).toBeNull();
  expect(typesAndErrors(events)).toEqual([
    ["ModelCallStarted", null],
    ["ModelCallFinished", "ProviderError"],
  ]);
  expect(model.requests).toHaveLength(1);
  expect(scheduler.pending).toBe(0);
});

test("an interrupt already set when a model-retry wait begins ends it with an empty reply", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const events: unknown[] = [];
  const interrupt = new AbortController();
  // Set during the failing attempt, so it is already aborted when the wait starts.
  const emitEvent = (event: unknown) => {
    events.push(event);
    if (event instanceof ModelCallStarted) interrupt.abort(new Error("interrupted"));
  };
  const live = liveSinks({ nodeId: "chat", llmConfig: CONFIG, emitEvent });
  const model = new ScriptedModel([new ProviderError("server", "boom", { retryAfter: 60 }), reply("never")], CONFIG);
  const scheduler = waitingScheduler();
  const options = { purpose: "chat", exchangeId: "x1", interrupt: interrupt.signal, retry: { scheduler } } as const;
  const cut = await rejected(callModel(live, model, REQUEST, options));
  expect(cut).toBeInstanceOf(ReplyInterrupted);
  expect((cut as ReplyInterrupted).reply.parts).toEqual([]);
  expect((cut as ReplyInterrupted).usage).toBeNull();
  expect(typesAndErrors(events)).toEqual([
    ["ModelCallStarted", null],
    ["ModelCallFinished", "ProviderError"],
  ]);
  expect(model.requests).toHaveLength(1);
  expect(scheduler.pending).toBe(0);
});

test("a stop during a model-retry wait still rejects with the stop", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const model = new ScriptedModel([new ProviderError("server", "boom", { retryAfter: 60 }), reply("never")], CONFIG);
  const scheduler = waitingScheduler();
  const stop = new AbortController();
  const reason = new Error("stopped");
  const interrupt = new AbortController();
  const call = callModel(null, model, REQUEST, {
    purpose: "chat",
    exchangeId: "x1",
    signal: stop.signal,
    interrupt: interrupt.signal,
    retry: { scheduler },
  });
  await scheduler.slept;
  stop.abort(reason);
  interrupt.abort(new Error("interrupted"));
  expect(await rejected(call)).toBe(reason);
  expect(model.requests).toHaveLength(1);
  expect(scheduler.pending).toBe(0);
});

test("an interrupt during a later model call's retry wait ends the loop with an empty reply", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const events: unknown[] = [];
  const model = new ScriptedModel(
    [
      reply("Checking.", ["lookup", { topic: "a" }, "call-1"]),
      new ProviderError("server", "boom", { retryAfter: 60 }),
      reply("never"),
    ],
    CONFIG,
  );
  const steering = new Inbox();
  const scheduler = waitingScheduler();
  const tools = new ToolSet({ lookup: lookupTool((topic) => `found ${topic}`) });
  const onEvent = (event: unknown) => void events.push(event);
  const output = run(chatFlow({ tools: ["lookup"] }), model, { steering, tools, scheduler, onEvent });
  await scheduler.slept;
  steering.interrupt("chat");
  expect(await output).toBe("");
  // The retry the wait was for never starts.
  expect(events.filter((event) => event instanceof ModelCallStarted)).toHaveLength(2);
  expect(model.requests).toHaveLength(2);
  expect(scheduler.pending).toBe(0);
});
