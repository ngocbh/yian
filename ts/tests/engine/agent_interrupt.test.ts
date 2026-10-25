/**
 * A host's `interrupt(run, inbox)` of a running or waiting AGENT reading `inbox` as its steer
 * inbox: from `tests/engine/test_agent_steering.py`'s interrupt rows, whose meaning TS yian changes.
 *
 * Python's node returns its latest reply's text after an interrupt. TS yian records the cut (a
 * `model_call` with finish `interrupted` and the text so far, `skipped` results for calls not
 * started, `interrupted` closers for the reply's pending calls), writes the honor step
 * `interrupted {by: <record seq>}`, and the agent's output is null: `Optional(...)` succeeds with
 * it, anything else (no `output:` included) fails `agent.interrupted`. Flows are hand-built
 * (`START(notes) -> chat [-> after] -> END`) and run by `FlowEngine` with a `ScriptedModel` from
 * `models`; the host interrupts through the live engine's `deliver({kind: "interrupt"})`, from a
 * tool, an event, a stream chunk, or the agent's model build (after it opened its inbox). Steers
 * the interrupted agent never read stay unread.
 *
 * Ported (changed): `test_interrupt_before_the_first_model_call_returns_empty_text`,
 * `test_interrupt_after_a_reply_keeps_it_and_records_skipped_calls`,
 * `test_interrupt_before_a_tool_call_skips_only_the_remaining_calls`,
 * `test_skipped_tool_calls_reach_the_host_as_interrupted_tool_events`,
 * `test_interrupt_at_the_top_of_a_later_iteration_returns_the_latest_reply`,
 * `test_interrupted_node_succeeds_and_the_flow_continues`,
 * `test_interrupt_for_a_finished_node_is_ignored`,
 * `test_interrupted_agent_in_a_loop_lets_the_loop_continue`,
 * `test_interrupt_mid_stream_keeps_the_partial_text_and_ends_the_stream`,
 * `test_interrupted_stream_reports_the_usage_received_so_far`,
 * `test_interrupt_mid_stream_in_a_pooled_run_keeps_partial_text`,
 * `test_interrupt_on_the_last_chunk_keeps_the_complete_reply` and
 * `test_interrupt_during_a_single_chunk_reply_keeps_the_complete_reply`: each output is null and
 * the text stays in the cut step.
 *
 * Not ported: `test_malformed_interrupted_answer_escapes_as_a_host_type_error` (the interrupt is
 * the engine's, not a host callback).
 *
 * TS-only: the honor step and its `by`; a pending interrupt at start; a final answer before the
 * interrupt stands; a stop wins; the structured calls are cut; pending calls close `interrupted`;
 * the output contract; one interrupt cuts every reader of its inbox; the model-retry wait rows at
 * the `callModel` level.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { type CompiledFlow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import {
  CallClosed,
  ModelCallFinished,
  ModelCallStarted,
  type RunEvent,
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
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { Interrupted, ReplyInterrupted } from "../../src/nodes/agent/interrupts.ts";
import { callModel, liveSinks } from "../../src/nodes/agent/live.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { TOOL_INTERRUPTED_OUTCOME } from "../../src/nodes/agent/memory.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { generateStructured, userMessage } from "../../src/nodes/agent/structured.ts";
import { CapabilityFault } from "../../src/nodes/base.ts";
import { NodeSteps, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import type { Scheduler } from "../../src/platform.ts";
import { interrupt, runFlow } from "../../src/run.ts";
import type { FlowEngine } from "../../src/runtime/engine.ts";
import { InterruptedRecord, NodeStepRecord, NodeWaitingRecord, readRunLog } from "../../src/runtime/log.ts";
import { commitPrepared, prepareSend } from "../../src/runtime/state.ts";
import { unreadEvents } from "../../src/runtime/waits.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { typeFor } from "../../src/typesys/types.ts";
import { inboxType, Type, type ValueRecord, ValueKind } from "../../src/typesys/values.ts";
import { inboxEngine } from "./inbox_fixtures.ts";
import { graph, interruptible, NO_ENV, rejected, serving } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };

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

/** A tool `name` taking `{topic}` and running `body` with it. */
function topicTool(name: string, body: (topic: string) => unknown): FunctionTool {
  return new FunctionTool({
    name,
    description: `Tool ${name}.`,
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => body(args.topic as string),
  });
}

/** A `lookup` tool running `body` with the topic. */
const lookupTool = (body: (topic: string) => string) => topicTool("lookup", body);

/** The START's inbox `notes: Inbox(str)`. */
const NOTES = new Parameter("notes", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.STRING))) });

/** The contract of the declared output `text` (`Optional(str)` by default; `null` for none). */
function contract(text: string | null): ValueContract | null {
  if (text === null) return null;
  return new ValueContract(text === "Optional({a: int})" ? recordA(true) : typeFor(text, new Map()));
}

/**
 * One `chat` agent granted `tools` with output `output`, reading steer inbox `notes` unless
 * `steerInbox` is false; with `after`, a summarizing agent reads its output, `(interrupted)` when
 * null. `readers` more agents read `notes` beside it, each answering on its own.
 */
function chatFlow(
  options: { steerInbox?: boolean; tools?: readonly string[]; after?: boolean; output?: string | null; readers?: readonly string[] } = {},
): CompiledFlow {
  const { steerInbox = true, tools = [], after = false, output = "Optional(str)", readers = [] } = options;
  const start = new StartNode(START_ID, { params: [NOTES] });
  const chat = new AgentNode("chat", { prompt: "Help the user.", llmConfig: CONFIG, steerInbox, tools: [...tools] });
  chat.outputContract = contract(output);
  const steer = { [STEER_INBOX_KEY]: "${input.notes}" };
  const others = readers.map((id) => {
    const node = new AgentNode(id, { prompt: "Help the user.", llmConfig: CONFIG, steerInbox: true, tools: [...tools] });
    node.outputContract = contract("Optional(str)");
    return node;
  });
  const wiring: Record<string, Record<string, unknown>> = { chat: steerInbox ? steer : {} };
  for (const node of others) wiring[node.id] = steer;
  if (!after) {
    return graph([start, chat, ...others], [[START_ID, "chat"], ...others.map((node) => [START_ID, node.id] as const)], {
      wiring,
      outputs: [new FlowOutput("output", "${chat.output}"), ...others.map((node) => new FlowOutput(node.id, `\${${node.id}.output}`))],
    });
  }
  const next = new AgentNode("after", { prompt: 'summarize: ${draft :- "(interrupted)"}', llmConfig: CONFIG });
  next.params = [new Parameter("draft")];
  return graph(
    [start, chat, next],
    [
      [START_ID, "chat"],
      ["chat", "after"],
    ],
    { wiring: { ...wiring, after: { draft: "${chat.output}" } }, outputs: [new FlowOutput("output", "${after.output}")] },
  );
}

/** The engine `run` drives, so a host can deliver to it. */
let running: FlowEngine | null = null;

/** Interrupt the running engine's readers of `notes`, as a host does while the run is live; return the record's seq. */
function interruptNow(): number | null {
  return running!.deliver({ kind: "interrupt", inbox: "notes" }).seq;
}

/** Send `event` to the running engine's `notes` inbox. */
function steer(event: string): void {
  running!.deliver({ kind: "send", inbox: "notes", event });
}

/** `model` served so that building it (after the agent opened its steer inbox) interrupts. */
function interruptingAtBuild(model: ChatModel): ModelAccess {
  const base = serving(model);
  const createModel: ModelAccess["createModel"] = (config, options) => (interruptNow(), base.createModel(config, options));
  return { ...base, createModel };
}

/**
 * Run `flow` with `model` (or `models`) and `options`, after sending `before` to its `notes` inbox;
 * return its last event. The engine is `running` while it runs, and after.
 */
async function runEngine(
  flow: CompiledFlow,
  model: ChatModel,
  options: {
    tools?: ToolSet;
    onEvent?: (event: unknown) => void;
    onStream?: (chunk: StreamChunk) => void;
    numWorkers?: number;
    scheduler?: Scheduler;
    before?: readonly string[];
    models?: ModelAccess;
  } = {},
): Promise<RunEvent | undefined> {
  const { before = [], models = serving(model), ...rest } = options;
  running = inboxEngine(flow, { models, ...rest });
  for (const event of before) commitPrepared(running.log, running.state, prepareSend(running.state, "notes", event));
  return (await running.run()).at(-1);
}

/** As `runEngine`, failing unless the run succeeded; return its output. */
async function run(flow: CompiledFlow, model: ChatModel, options: Parameters<typeof runEngine>[2] = {}): Promise<unknown> {
  const last = await runEngine(flow, model, options);
  expect(last, (last as RunFailed | undefined)?.failure?.message).toBeInstanceOf(RunSucceeded);
  return (last as RunSucceeded).output;
}

/** The unread events of the last run's `notes` inbox, as values. */
function unread(): unknown[] {
  return (unreadEvents(running!.state.waits).get("notes") ?? []).map((seq) => running!.state.inboxEvents.get(seq)!.toObject());
}

/** The steps node `node` recorded in the last run. */
function steps(node = "chat"): ValueRecord[] {
  return running!.log.records().flatMap(({ record }) => (record instanceof NodeStepRecord && record.nodeId === node ? [record.step] : []));
}

/** The last run's step kinds of `node`, a closer or skipped result as its `closed` word, a cut call as `cut`. */
function shape(node = "chat"): string[] {
  return steps(node).map((step) => {
    if (step["kind"] === "model_call") return step["finish"] === "interrupted" ? "cut" : Object.hasOwn(step, "error") ? "failed" : "model_call";
    return typeof step["closed"] === "string" ? (step["closed"] as string) : String(step["kind"]);
  });
}

/** The seq of the last run's `interrupted` record. */
function recordSeq(): number {
  return running!.log.records().find(({ record }) => record instanceof InterruptedRecord)!.seq;
}

/** The `by` of the last run's honor steps of `node`. */
function honors(node = "chat"): number[] {
  return steps(node).flatMap((step) => (step["kind"] === "interrupted" ? [Number(step["by"])] : []));
}

/** The text of `node`'s cut `model_call` step in the last run. */
function cutText(node = "chat"): string {
  const step = steps(node).find((each) => each["finish"] === "interrupted")!;
  return Message.fromJSON(step["output"]).text();
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

/** The record type `{a: int}`, `Optional(...)` when `nullable`. */
function recordA(nullable = false): Type {
  return new Type({ kind: ValueKind.OBJECT, fields: new Map([["a", Type.scalar(ValueKind.INTEGER)]]), required: new Set(["a"]), nullable });
}

/** Steps recorded outside a run, with the list they go to. */
function recorded(): { steps: NodeSteps; written: ValueRecord[] } {
  const written: ValueRecord[] = [];
  return { steps: new NodeSteps(1, [], (step) => written.push(step)), written };
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

// --- the loop's honor points (loop level) ------------------------------------------------ //

test("an interrupt mid-stream: the cut model_call, then {kind: interrupted, by}; the result is Interrupted", async () => {
  const host = interruptible();
  const { steps: nodeSteps, written } = recorded();
  const model = new Paced(words("hello", " big", " world", new Finish({ reason: "stop" })), (index) => {
    if (index === 1) host.interrupt(7);
  });
  const result = await runAgent(agentContext(model, { steerInbox: host.steerInbox, steps: nodeSteps }));
  expect(result).toEqual(new Interrupted(7));
  expect(written.map((step) => [step["kind"], step["finish"] ?? step["by"] ?? null])).toEqual([
    ["model_call", "interrupted"],
    ["interrupted", 7n],
  ]);
});

test("a pending interrupt at start: no model call, only the honor step", async () => {
  const host = interruptible();
  host.interrupt(3);
  const { steps: nodeSteps, written } = recorded();
  const model = new ScriptedModel([], CONFIG);
  expect(await runAgent(agentContext(model, { steerInbox: host.steerInbox, steps: nodeSteps }))).toEqual(new Interrupted(3));
  expect(model.requests).toEqual([]);
  expect(written).toEqual([{ kind: "interrupted", by: 3n }]);
});

test("an interrupt after the final reply changes nothing: its text, no honor step", async () => {
  const host = interruptible();
  const { steps: nodeSteps, written } = recorded();
  const emitEvent = (event: unknown) => void (event instanceof ModelCallFinished && host.interrupt());
  const model = new ScriptedModel([reply("done")], CONFIG);
  const result = await runAgent(agentContext(model, { steerInbox: host.steerInbox, steps: nodeSteps, emitEvent }));
  expect(result).toMatchObject({ value: "done" });
  expect(written.map((step) => step["kind"])).toEqual(["model_call"]);
});

test("a stop that lands after a cut, before the loop sees it, wins over the interrupt", async () => {
  const stop = new AbortController();
  const reason = new Error("stopped");
  const host = interruptible();
  // The cut's `ModelCallFinished` is emitted after the reply was cut and before the loop sees it.
  const emitEvent = (event: unknown) => {
    if (event instanceof ModelCallStarted) host.interrupt();
    if (event instanceof ModelCallFinished) stop.abort(reason);
  };
  const model = new ScriptedModel([words("hello", " world", new Finish({ reason: "stop" }))], CONFIG);
  const ctx = agentContext(model, { signal: stop.signal, steerInbox: host.steerInbox, emitEvent });
  expect(await rejected(runAgent(ctx))).toBe(reason);
});

test("a cut with no pending interrupt is a host fault", async () => {
  const controller = new AbortController();
  controller.abort();
  const steerInbox = { ...interruptible().steerInbox, interrupt: controller.signal };
  const fault = await rejected(runAgent(agentContext(new ScriptedModel([], CONFIG), { steerInbox })));
  expect(fault).toBeInstanceOf(CapabilityFault);
  expect(((fault as CapabilityFault).original as Error).message).toBe("node 'chat' was cut with no pending interrupt");
});

// --- the structured calls are cut too ---------------------------------------------------- //

test("a no-tools structured agent is cut in its structured call", async () => {
  const host = interruptible();
  const { steps: nodeSteps, written } = recorded();
  const model = new Paced(words('{"a"', ": 1", "}", new Finish({ reason: "stop" })), (index) => {
    if (index === 0) host.interrupt();
  });
  const outputType = recordA(true);
  const result = await runAgent(agentContext(model, { steerInbox: host.steerInbox, steps: nodeSteps, outputType }));
  expect(result).toEqual(new Interrupted(7));
  expect(written.map((step) => [step["kind"], step["finish"] ?? null])).toEqual([
    ["model_call", "interrupted"],
    ["interrupted", null],
  ]);
});

test("the trailing structured call after tools is cut, or never starts once an interrupt is pending", async () => {
  const host = interruptible();
  const { steps: nodeSteps, written } = recorded();
  const outputType = recordA(true);
  const toolset = new ToolSet({ lookup: lookupTool((topic) => `found ${topic}`) });
  // The final chat answer arrives whole, then the interrupt lands before the structured call.
  const emitEvent = (event: unknown) => void (event instanceof ModelCallFinished && event.n === 3 && host.interrupt());
  const model = new ScriptedModel([reply("", ["lookup", { topic: "a" }, "call-1"]), reply("answer"), reply('{"a": 1}')], CONFIG);
  const ctx = agentContext(model, { steerInbox: host.steerInbox, steps: nodeSteps, outputType, toolset, tools: ["lookup"], emitEvent });
  expect(await runAgent(ctx)).toEqual(new Interrupted(7));
  expect(model.requests).toHaveLength(2);
  expect(written.map((step) => step["kind"])).toEqual(["model_call", "tool_result", "model_call", "interrupted"]);
});

test("a resumed start past the honor before the trailing structured call honors it again: no live call", async () => {
  const outputType = recordA(true);
  const toolset = () => new ToolSet({ lookup: lookupTool((topic) => `found ${topic}`) });
  const host = interruptible();
  const { steps: first, written } = recorded();
  const emitEvent = (event: unknown) => void (event instanceof ModelCallFinished && event.n === 3 && host.interrupt());
  const live = new ScriptedModel([reply("", ["lookup", { topic: "a" }, "call-1"]), reply("answer"), reply('{"a": 1}')], CONFIG);
  const tools = ["lookup"];
  await runAgent(agentContext(live, { steerInbox: host.steerInbox, steps: first, outputType, toolset: toolset(), tools, emitEvent }));
  expect(written.map((step) => step["kind"])).toEqual(["model_call", "tool_result", "model_call", "interrupted"]);

  // A crash before the commit: the honor step deleted the pending interrupt, so the signal is unset.
  const resumed: ValueRecord[] = [];
  const steps = new NodeSteps(2, written, (step) => resumed.push(step));
  const model = new ScriptedModel([reply('{"a": 1}')], CONFIG);
  const ctx = agentContext(model, { steerInbox: interruptible().steerInbox, steps, outputType, toolset: toolset(), tools });
  expect(await runAgent(ctx)).toEqual(new Interrupted(7));
  expect(model.requests).toEqual([]);
  expect(resumed).toEqual([]);
});

/**
 * Run `chat` live over `replies`, interrupted once model call `cutAt` finished, then resume it
 * from the steps it wrote with nothing pending; return what each start wrote and the resume's
 * result and requests.
 */
async function resumedAfterHonor(replies: Response[], cutAt: number, extra: Partial<AgentRunContext>) {
  const host = interruptible();
  const { steps: first, written } = recorded();
  const emitEvent = (event: unknown) => void (event instanceof ModelCallFinished && event.n === cutAt && host.interrupt());
  const live = await runAgent(agentContext(new ScriptedModel(replies, CONFIG), { ...extra, steerInbox: host.steerInbox, steps: first, emitEvent }));
  // A crash before the commit: the honor step deleted the pending interrupt, so the signal is unset.
  const resumed: ValueRecord[] = [];
  const steps = new NodeSteps(2, written, (step) => resumed.push(step));
  const model = new ScriptedModel(replies.slice(cutAt), CONFIG);
  const result = await runAgent(agentContext(model, { ...extra, steerInbox: interruptible().steerInbox, steps }));
  return { live, written: written.map((step) => step["kind"]), result, requests: model.requests.length, resumed };
}

test("a resumed start past the honor after an invalid structured reply makes no correction call", async () => {
  const run = await resumedAfterHonor([reply("not json"), reply("still not"), reply("nope")], 1, { outputType: recordA(true), retries: 2 });
  expect([run.live, run.written]).toEqual([new Interrupted(7), ["model_call", "interrupted"]]);
  expect([run.result, run.requests, run.resumed]).toEqual([new Interrupted(7), 0, []]);
});

test("a resumed start past the honor after a trailing structured attempt makes no correction call", async () => {
  const toolset = new ToolSet({ lookup: lookupTool((topic) => `found ${topic}`) });
  const replies = [reply("", ["lookup", { topic: "a" }, "call-1"]), reply("answer"), reply("not json"), reply("nope"), reply("nope")];
  const run = await resumedAfterHonor(replies, 4, { outputType: recordA(true), retries: 2, toolset, tools: ["lookup"] });
  expect([run.live, run.written]).toEqual([new Interrupted(7), ["model_call", "tool_result", "model_call", "model_call", "interrupted"]]);
  expect([run.result, run.requests, run.resumed]).toEqual([new Interrupted(7), 0, []]);
});

test("generateStructured doesn't retry a cut call", async () => {
  const interruptSignal = new AbortController();
  const model = new Paced(words('{"a"', ": x", "}", new Finish({ reason: "stop" })), (index) => {
    if (index === 0) interruptSignal.abort();
  });
  const cut = await rejected(
    generateStructured(model, [userMessage("go")], recordA(), { retries: 3, interrupt: interruptSignal.signal }),
  );
  expect(cut).toBeInstanceOf(ReplyInterrupted);
  expect(model.sent).toHaveLength(2);
});

// --- interrupts at loop safe points (engine) ---------------------------------------------- //

test("an interrupt before the first model call: no call, null, the steer stays unread", async () => {
  const model = new ScriptedModel([], CONFIG);
  expect(await run(chatFlow(), model, { models: interruptingAtBuild(model), before: ["never read"] })).toBeNull();
  expect(model.requests).toEqual([]);
  expect(unread()).toEqual(["never read"]);
  expect(shape()).toEqual(["interrupted"]);
  expect(honors()).toEqual([recordSeq()]);
});

test("an agent without a steer inbox isn't interrupted: an empty Stop is recorded; the event stays unread", async () => {
  const model = new ScriptedModel([reply("done")], CONFIG);
  expect(await run(chatFlow({ steerInbox: false }), model, { models: interruptingAtBuild(model), before: ["ignored"] })).toBe("done");
  const stops = running!.log.records().filter(({ record }) => record instanceof InterruptedRecord);
  expect(stops.map(({ record }) => [...(record as InterruptedRecord).nodes])).toEqual([[]]);
  expect(unread()).toEqual(["ignored"]);
  expect(shown(model.requests[0]!.messages).at(-1)).toEqual(["user", "Help the user."]);
});

test("an interrupt after a reply skips its calls and ends null", async () => {
  const ran: string[] = [];
  const events: unknown[] = [];
  const onEvent = (event: unknown) => {
    events.push(event);
    if (event instanceof ModelCallFinished) interruptNow();
  };
  const lookup = lookupTool((topic) => (ran.push(topic), "found"));
  const model = new ScriptedModel([twoCalls()], CONFIG);
  expect(await run(chatFlow({ tools: ["lookup"] }), model, { tools: new ToolSet({ lookup }), onEvent })).toBeNull();
  expect(ran).toEqual([]);
  expect(model.requests).toHaveLength(1);
  expect(toolEvents(events)).toEqual([
    ["started", 0, "call-1"],
    ["finished", 0, "call-1", "interrupted"],
    ["started", 1, "call-2"],
    ["finished", 1, "call-2", "interrupted"],
  ]);
  expect(shape()).toEqual(["model_call", "skipped", "skipped", "interrupted"]);
  // The record closed no host call, so no observer announced it.
  expect(events.filter((event) => event instanceof CallClosed)).toEqual([]);
});

test("an interrupt before a tool call skips only the remaining calls", async () => {
  const ran: string[] = [];
  const lookup = lookupTool((topic) => (ran.push(topic), interruptNow(), "found"));
  const model = new ScriptedModel([twoCalls()], CONFIG);
  expect(await run(chatFlow({ tools: ["lookup"] }), model, { tools: new ToolSet({ lookup }) })).toBeNull();
  expect(ran).toEqual(["a"]);
  expect(model.requests).toHaveLength(1);
  expect(shape()).toEqual(["model_call", "tool_result", "skipped", "interrupted"]);
});

test("skipped tool calls reach the host as interrupted tool events", async () => {
  const events: unknown[] = [];
  const lookup = lookupTool((topic) => (interruptNow(), `found ${topic}`));
  const model = new ScriptedModel([twoCalls()], CONFIG);
  await run(chatFlow({ tools: ["lookup"] }), model, { tools: new ToolSet({ lookup }), onEvent: (event) => void events.push(event) });
  const brackets = events.filter((event) => event instanceof ToolCallStarted || event instanceof ToolCallFinished);
  const ids = (callIndex: number) => ({ nodeId: "chat", n: 1, callIndex, toolCallId: `call-${callIndex + 1}`, name: "lookup" });
  expect(brackets).toEqual([
    new ToolCallStarted({ ...ids(0), arguments: { topic: "a" } }),
    new ToolCallFinished({ ...ids(0), result: "found a" }),
    new ToolCallStarted({ ...ids(1), arguments: { topic: "b" } }),
    new ToolCallFinished({ ...ids(1), error: "interrupted" }),
  ]);
});

test("an interrupt at the top of a later iteration ends null; a steer sent with it stays queued", async () => {
  const lookup = lookupTool(() => (steer("too late"), interruptNow(), "found"));
  const model = new ScriptedModel([reply("Looking.", ["lookup", { topic: "a" }, "call-1"])], CONFIG);
  expect(await run(chatFlow({ tools: ["lookup"] }), model, { tools: new ToolSet({ lookup }) })).toBeNull();
  expect(model.requests).toHaveLength(1);
  expect(unread()).toEqual(["too late"]);
  expect(shape()).toEqual(["model_call", "tool_result", "interrupted"]);
});

test("an interrupted Optional(str) agent succeeds null and the flow continues", async () => {
  const model = new ScriptedModel([reply("summary")], CONFIG);
  expect(await run(chatFlow({ after: true }), model, { models: interruptingAtBuild(model) })).toBe("summary");
  expect(shown(model.requests[0]!.messages).at(-1)).toEqual(["user", "summarize: (interrupted)"]);
});

test("an interrupt after the reader finished records an empty Stop and stops nothing", async () => {
  let started = 0;
  const seqs: (number | null)[] = [];
  const onEvent = (event: unknown) => void (event instanceof ModelCallStarted && ++started === 2 && seqs.push(interruptNow()));
  const model = new ScriptedModel([reply("chat done"), reply("summary")], CONFIG);
  expect(await run(chatFlow({ after: true }), model, { onEvent })).toBe("summary");
  expect(seqs).toHaveLength(1);
  const record = running!.log.records().find(({ seq }) => seq === seqs[0])!.record as InterruptedRecord;
  expect([...record.nodes]).toEqual([]);
  expect(shown(model.requests[1]!.messages).at(-1)).toEqual(["user", "summarize: chat done"]);
});

test("a too-deep tool call fails the node even with the interrupt pending", async () => {
  const ran: string[] = [];
  const lookup = lookupTool((topic) => (ran.push(topic), "found"));
  const deep = '{"a": '.repeat(100) + '"leaf"' + "}".repeat(100);
  const call = new ToolCallPart({ id: "call-1", name: "lookup", arguments: null, rawArguments: `{"x": ${deep}}` });
  const model = new ScriptedModel(
    [new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" })],
    CONFIG,
  );
  const onEvent = (event: unknown) => void (event instanceof ModelCallStarted && interruptNow());
  const last = await runEngine(chatFlow({ tools: ["lookup"] }), model, { tools: new ToolSet({ lookup }), onEvent });
  expect(last).toBeInstanceOf(RunFailed);
  expect((last as RunFailed).failure.code).toBe("node.execute");
  expect((last as RunFailed).failure.message).toContain("tool call 'lookup' arguments at x");
  expect(ran).toEqual([]);
});

// Python's `test_agent_in_loop.FLOW` with an agent reading a steer inbox: two LOOP turns, each an
// AGENT reply folded into the transcript; an interrupted reply folds as "-".
const LOOP_FLOW = `
id: t
name: t
input:
  seed: !str
  notes: !Inbox(str)
defs:
  body:
    input:
      transcript: !str
      exited: !bool
      notes: !Inbox(str)
    nodes:
      reply:
        kind: agent
        steer_inbox: \${input.notes}
        tools: [ping]
        input:
          transcript: \${input.transcript}
        output: Optional(str)
        prompt: "history \${transcript}"
      fold:
        kind: code
        code: |
          return { transcript: inputs.transcript + "|" + (inputs.reply ?? "-"), exited: inputs.exited }
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
      notes: \${input.notes}
    times: 2
output: \${loop.output.transcript}
`;

test("an interrupt names only the running loop occurrence; the next one opens while the Stop stands and stops too", async () => {
  const log = `${process.env["TMPDIR"] ?? "/tmp"}/yian-agent-interrupt-loop-${process.pid}-${Date.now()}.jsonl`;
  let started = 0;
  // The first occurrence is interrupted during its model call, so its ping call is skipped.
  const onEvent = (event: unknown) => void (event instanceof ModelCallStarted && ++started === 1 && interrupt(log, "notes"));
  const ping = new FunctionTool({ name: "ping", description: "Ping.", parametersJsonSchema: { type: "object", properties: {} }, execute: () => "pong" });
  const model = new ScriptedModel([reply("partial", ["ping", {}, "p-1"]), reply("turn2")], CONFIG);
  const result = await runFlow(await loadFlow(LOOP_FLOW), { seed: "s" }, {
    tools: { ping },
    onEvent,
    llmConfig: CONFIG,
    models: serving(model),
    env: NO_ENV,
    log,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  // No take or wait ends the Stop between the occurrences, so the second is stopped before its
  // model call by its own record.
  expect(result.output).toBe("s|-|-");
  expect(model.requests).toHaveLength(1);
  const stops = readRunLog(log).flatMap(({ record }) => (record instanceof InterruptedRecord ? [[...record.nodes]] : []));
  expect(stops).toEqual([["loop#0/reply"], ["loop#1/reply"]]);
});

// --- the stream is a safe point ----------------------------------------------------------- //

test("an interrupt mid-stream records the partial text and ends the stream", async () => {
  const model = new ScriptedModel([words("hello", " big", " world", LOOKUP_CALL, new Finish({ reason: "tool_calls" }))], CONFIG);
  const chunks: StreamChunk[] = [];
  const events: unknown[] = [];
  const ran: string[] = [];
  const onStream = (chunk: StreamChunk) => {
    chunks.push(chunk);
    if (chunk.chunk === " big") interruptNow();
  };
  const lookup = lookupTool((topic) => (ran.push(topic), "found"));
  const output = await run(chatFlow({ tools: ["lookup"] }), model, {
    tools: new ToolSet({ lookup }),
    onStream,
    onEvent: (event) => void events.push(event),
  });
  expect(output).toBeNull();
  expect(cutText()).toBe("hello big");
  expect(ran).toEqual([]);
  expect(chunked(chunks)).toEqual([
    ["hello", false],
    [" big", false],
    ["", true],
  ]);
  const finished = events.filter((event) => event instanceof ModelCallFinished);
  expect(finished).toHaveLength(1);
  expect([finished[0]!.error, finished[0]!.usage, finished[0]!.n]).toEqual([null, null, 1]);
  expect(toolEvents(events)).toEqual([]);
  expect(shape()).toEqual(["cut", "interrupted"]);
});

test("an interrupted stream reports the usage received so far", async () => {
  const usage = new Usage({ inputTokens: 1000, outputTokens: 1, totalTokens: 1001 });
  const model = new ScriptedModel([[new UsageReport(usage), ...words("hello", " big", " world", new Finish({ reason: "stop" }))]], CONFIG);
  const events: unknown[] = [];
  const onStream = (chunk: StreamChunk) => void (chunk.chunk === " big" && interruptNow());
  expect(await run(chatFlow(), model, { onStream, onEvent: (event) => void events.push(event) })).toBeNull();
  const finished = events.filter((event) => event instanceof ModelCallFinished);
  expect(finished.map((event) => [event.usage, event.error])).toEqual([
    [new TokenUsage({ inputTokens: 1000, outputTokens: 1, totalTokens: 1001 }), null],
  ]);
});

test("an interrupt mid-stream in a pooled run records partial text", async () => {
  const all = ["hello", " big", " wide", " world"];
  const seen = Promise.withResolvers<void>();
  // Pause after the second word until the host has seen it.
  const model = new Paced(words(...all, new Finish({ reason: "stop" })), (index) => (index === 1 ? seen.promise : undefined));
  const onStream = (chunk: StreamChunk) => {
    if (chunk.chunk === " big") {
      interruptNow();
      seen.resolve();
    }
  };
  expect(await run(chatFlow(), model, { onStream, numWorkers: 2 })).toBeNull();
  const full = all.join("");
  const text = cutText();
  expect(text.length).toBeGreaterThan(0);
  expect(full.startsWith(text) && text !== full).toBe(true);
  expect(model.sent.length).toBeLessThan(all.length);
});

test("an interrupt on the last chunk records the whole reply and skips its call", async () => {
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
  const ran: string[] = [];
  const events: unknown[] = [];
  // The reply's last text chunk.
  const onStream = (chunk: StreamChunk) => void (chunk.chunk === " go" && interruptNow());
  const lookup = lookupTool((topic) => (ran.push(topic), "found"));
  const output = await run(chatFlow({ tools: ["lookup"] }), model, {
    tools: new ToolSet({ lookup }),
    onStream,
    onEvent: (event) => void events.push(event),
  });
  expect(output).toBeNull();
  expect(ran).toEqual([]);
  expect(events.filter((event) => event instanceof ModelCallFinished).map((event) => event.usage)).toEqual([
    new TokenUsage({ inputTokens: 500, outputTokens: 20, totalTokens: 520 }),
  ]);
  expect(toolEvents(events)).toEqual([
    ["started", 0, "call-1"],
    ["finished", 0, "call-1", "interrupted"],
  ]);
  expect(shape()).toEqual(["model_call", "skipped", "interrupted"]);
});

test("an interrupt during a single-chunk reply records the whole reply and skips its calls", async () => {
  const ran: string[] = [];
  const events: unknown[] = [];
  const onEvent = (event: unknown) => {
    events.push(event);
    if (event instanceof ModelCallStarted) interruptNow();
  };
  const lookup = lookupTool((topic) => (ran.push(topic), "found"));
  const model = new ScriptedModel([twoCalls()], CONFIG);
  const output = await run(chatFlow({ tools: ["lookup"] }), model, { tools: new ToolSet({ lookup }), onEvent, onStream: () => {} });
  expect(output).toBeNull();
  expect(ran).toEqual([]);
  expect(model.requests).toHaveLength(1);
  expect(shape()).toEqual(["model_call", "skipped", "skipped", "interrupted"]);
});

test("a cut with no stream sink still records the text received", async () => {
  const model = new Paced(words("hello", " big", " world", new Finish({ reason: "stop" })), (index) => void (index === 1 && interruptNow()));
  expect(await run(chatFlow(), model)).toBeNull();
  expect(cutText()).toBe("hello big");
  expect(model.closed).toBe(1);
});

test("a partial reply's tool call never runs", async () => {
  const ran: string[] = [];
  const events: unknown[] = [];
  const model = new ScriptedModel(
    [[new TextDelta(0, "hello"), LOOKUP_CALL, new TextDelta(2, " big"), new Finish({ reason: "tool_calls" })]],
    CONFIG,
  );
  const lookup = lookupTool((topic) => (ran.push(topic), "found"));
  const output = await run(chatFlow({ tools: ["lookup"] }), model, {
    tools: new ToolSet({ lookup }),
    onStream: (chunk) => void (chunk.chunk === "hello" && interruptNow()),
    onEvent: (event) => void events.push(event),
  });
  expect(output).toBeNull();
  expect(cutText()).toBe("hello");
  expect(ran).toEqual([]);
  expect(toolEvents(events)).toEqual([]);
});

test("a non-text event after the interrupt doesn't cut the reply", async () => {
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
  const lookup = lookupTool((topic) => (ran.push(topic), "found"));
  const output = await run(chatFlow({ tools: ["lookup"] }), model, {
    tools: new ToolSet({ lookup }),
    onStream: (chunk) => void (chunk.chunk === "hello" && interruptNow()),
    onEvent: (event) => void events.push(event),
  });
  expect(output).toBeNull();
  expect(ran).toEqual([]);
  expect(events.filter((event) => event instanceof ModelCallFinished).map((event) => event.usage)).toEqual([
    new TokenUsage({ inputTokens: 5, outputTokens: 3, totalTokens: 8 }),
  ]);
  expect(toolEvents(events)).toEqual([
    ["started", 0, "call-1"],
    ["finished", 0, "call-1", "interrupted"],
  ]);
});

// --- the output contract ------------------------------------------------------------------- //

test.each([
  ["str", "str"],
  ["no output:", null],
])("an interrupted agent with output %s fails agent.interrupted, with no messages commit", async (_name, output) => {
  const model = new ScriptedModel([], CONFIG);
  const last = await runEngine(chatFlow({ output }), model, { models: interruptingAtBuild(model) });
  expect(last).toBeInstanceOf(RunFailed);
  expect((last as RunFailed).failure.code).toBe("agent.interrupted");
  expect((last as RunFailed).failure.nodeId).toBe("chat");
  expect((last as RunFailed).failure.message).toBe(
    "agent node 'chat' was interrupted and its output doesn't allow null; declare output: Optional(...) to route on an interrupt",
  );
  expect(running!.state.pool.store.has("chat.new_messages")).toBe(false);
});

test("an interrupted Optional({a: int}) agent succeeds null", async () => {
  const model = new ScriptedModel([], CONFIG);
  expect(await run(chatFlow({ output: "Optional({a: int})" }), model, { models: interruptingAtBuild(model) })).toBeNull();
});

// --- pending calls close interrupted ------------------------------------------------------ //

/** `hold` returns pending(); `save` runs `body`. Both unsafe, so `hold` runs alone first. */
function holdAndSave(body: () => unknown): ToolSet {
  return new ToolSet({
    hold: topicTool("hold", () => pending()),
    save: topicTool("save", body),
  });
}

/** A reply calling `hold` then `save`. */
const holdThenSave = () => reply("", ["hold", { topic: "h" }, "c-hold"], ["save", { topic: "s" }, "c-save"]);

test("an interrupt while a sibling runs closes the pending call: its closer, then the honor step; no wait", async () => {
  const events: unknown[] = [];
  const tools = holdAndSave(() => (interruptNow(), "saved"));
  const model = new ScriptedModel([holdThenSave()], CONFIG);
  const flow = chatFlow({ tools: ["hold", "save"] });
  expect(await run(flow, model, { tools, onEvent: (event) => void events.push(event) })).toBeNull();
  expect(shape()).toEqual(["model_call", "pending", "tool_result", "interrupted", "interrupted"]);
  expect(steps()[3]).toMatchObject({ call_id: "c-hold", closed: "interrupted", output: TOOL_INTERRUPTED_OUTCOME, is_error: true });
  expect(running!.log.records().some(({ record }) => record instanceof NodeWaitingRecord)).toBe(false);
  expect(events.filter((event) => event instanceof CallClosed)).toEqual([
    new CallClosed({ callId: `${running!.state.runId}/chat/1/0`, reason: "interrupted" }),
  ]);
});

test("a pending call the host answered before the interrupt keeps its answer: no closer", async () => {
  const callId = () => `${running!.state.runId}/chat/1/0`;
  const tools = holdAndSave(() => {
    running!.deliver({ kind: "complete", callId: callId(), result: { output: "held" } });
    interruptNow();
    return "saved";
  });
  const model = new ScriptedModel([holdThenSave()], CONFIG);
  expect(await run(chatFlow({ tools: ["hold", "save"] }), model, { tools })).toBeNull();
  expect(shape()).toEqual(["model_call", "pending", "tool_result", "tool_result", "interrupted"]);
  expect(steps()[2]).toMatchObject({ call_id: "c-hold", output: "held" });
});

test("a call that pends after the record is registered closed; its closer, then the honor step", async () => {
  const events: unknown[] = [];
  // One safe group: `save` interrupts while `hold`, already started, pends a tick later.
  const saveThenHold = reply("", ["save", {}, "c-save"], ["hold", {}, "c-hold"]);
  const model = new ScriptedModel([saveThenHold], CONFIG);
  const safe = new ToolSet({
    hold: new FunctionTool({ name: "hold", description: "Hold.", parametersJsonSchema: { type: "object", properties: {} }, isConcurrencySafe: () => true, execute: async () => (await Promise.resolve(), pending()) }),
    save: new FunctionTool({ name: "save", description: "Save.", parametersJsonSchema: { type: "object", properties: {} }, isConcurrencySafe: () => true, execute: () => (interruptNow(), "saved") }),
  });
  expect(await run(chatFlow({ tools: ["hold", "save"] }), model, { tools: safe, onEvent: (event) => void events.push(event) })).toBeNull();
  expect(shape()).toEqual(["model_call", "tool_result", "pending", "interrupted", "interrupted"]);
  expect(events.filter((event) => event instanceof CallClosed)).toEqual([
    new CallClosed({ callId: `${running!.state.runId}/chat/1/1`, reason: "interrupted" }),
  ]);
});

// --- one interrupt, every reader ----------------------------------------------------------- //

test("one interrupt cuts every running reader of the inbox, in flight order", async () => {
  let started = 0;
  const both = Promise.withResolvers<void>();
  const seqs: (number | null)[] = [];
  // Each reader streams one word; the second to do so interrupts, then both go on.
  const model: ChatModel = {
    provider: "anthropic",
    model: "claude-test",
    stream: () =>
      new EventStream(
        (async function* () {
          yield new TextDelta(0, "hi");
          if (++started === 2) {
            seqs.push(interruptNow());
            both.resolve();
          }
          await both.promise;
          yield new TextDelta(0, " there");
          yield new Finish({ reason: "stop" });
        })(),
      ),
  };
  const last = await runEngine(chatFlow({ readers: ["b"] }), model, { numWorkers: 2 });
  expect(last, (last as RunFailed).failure?.message).toBeInstanceOf(RunSucceeded);
  expect((last as RunSucceeded).output).toEqual({ output: null, b: null });
  const record = running!.log.records().find(({ record }) => record instanceof InterruptedRecord)!.record as InterruptedRecord;
  expect([...record.nodes]).toEqual(["chat", "b"]);
  expect(seqs.filter((seq) => seq !== null)).toHaveLength(1);
  expect([honors("chat"), honors("b")]).toEqual([[recordSeq()], [recordSeq()]]);
});

// --- the model-retry wait ------------------------------------------------------------------ //

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
  const interruptSignal = new AbortController();
  const call = callModel(live, model, REQUEST, { purpose: "chat", interrupt: interruptSignal.signal, retry: { scheduler } });
  await scheduler.slept;
  interruptSignal.abort(new Error("interrupted"));
  const cut = await rejected(call);
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

test("an interrupt already set when a model-retry wait begins ends it with an empty reply", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const events: unknown[] = [];
  const interruptSignal = new AbortController();
  // Set during the failing attempt, so it is already aborted when the wait starts.
  const emitEvent = (event: unknown) => {
    events.push(event);
    if (event instanceof ModelCallStarted) interruptSignal.abort(new Error("interrupted"));
  };
  const live = liveSinks({ nodeId: "chat", llmConfig: CONFIG, emitEvent });
  const model = new ScriptedModel([new ProviderError("server", "boom", { retryAfter: 60 }), reply("never")], CONFIG);
  const scheduler = waitingScheduler();
  const cut = await rejected(callModel(live, model, REQUEST, { purpose: "chat", interrupt: interruptSignal.signal, retry: { scheduler } }));
  expect(cut).toBeInstanceOf(ReplyInterrupted);
  expect((cut as ReplyInterrupted).reply.parts).toEqual([]);
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
  const interruptSignal = new AbortController();
  const call = callModel(null, model, REQUEST, {
    purpose: "chat",
    signal: stop.signal,
    interrupt: interruptSignal.signal,
    retry: { scheduler },
  });
  await scheduler.slept;
  stop.abort(reason);
  interruptSignal.abort(new Error("interrupted"));
  expect(await rejected(call)).toBe(reason);
  expect(model.requests).toHaveLength(1);
  expect(scheduler.pending).toBe(0);
});

test("an interrupt during a later model call's retry wait ends the loop null: no model_call for the cut wait", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const events: unknown[] = [];
  const model = new ScriptedModel(
    [reply("Checking.", ["lookup", { topic: "a" }, "call-1"]), new ProviderError("server", "boom", { retryAfter: 60 }), reply("never")],
    CONFIG,
  );
  const scheduler = waitingScheduler();
  const tools = new ToolSet({ lookup: lookupTool((topic) => `found ${topic}`) });
  const output = run(chatFlow({ tools: ["lookup"] }), model, { tools, scheduler, onEvent: (event) => void events.push(event) });
  await scheduler.slept;
  interruptNow();
  expect(await output).toBeNull();
  // The retry the wait was for never starts.
  expect(events.filter((event) => event instanceof ModelCallStarted)).toHaveLength(2);
  expect(model.requests).toHaveLength(2);
  expect(scheduler.pending).toBe(0);
  expect(shape()).toEqual(["model_call", "tool_result", "failed", "interrupted"]);
});

