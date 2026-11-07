/**
 * Steers to a running AGENT. A steer is a send to the agent's steer inbox (`steer_inbox:`, a
 * top-level `Inbox(str)`), which the agent reads at the top of each iteration as a recorded `steer`
 * step. Flows are hand-built (`START(notes) -> chat [-> after] -> END`) and run by `FlowEngine`
 * with a `ScriptedModel` from `models`; a tool sends a steer through the live engine's `deliver`.
 * Interrupts are in `agent_interrupt.test.ts`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import { type CompiledFlow, copyGraph, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { RunStarted, RunSucceeded, StreamChunk, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { type ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import { Output, type SteerInbox } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { send } from "../../src/run.ts";
import type { FlowEngine } from "../../src/runtime/engine.ts";
import { type LiveFrame, LiveChannel, LiveView } from "../../src/runtime/live.ts";
import { NodeStepRecord, readRunLog } from "../../src/runtime/log.ts";
import { commitPrepared, prepareSend, replayRecords, RunState } from "../../src/runtime/state.ts";
import { unreadEvents } from "../../src/runtime/waits.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { type Scheduler, withWarnings } from "../../src/platform.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { type JsonObject, logJson } from "../../src/typesys/json.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { InboxRef, inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { inboxEngine } from "../helpers/inbox_fixtures.ts";
import {
  CrashAt,
  crashes,
  drive,
  graph,
  NO_ENV,
  noGrowth,
  resumeWith,
  runWith,
  serving,
} from "../helpers/run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

// One directory for the run logs of the boundary and resume tests, each under its own name.
const dir = mkdtempSync(join(tmpdir(), "agent-steering-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

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

/** Text and two `lookup` calls. */
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

/**
 * One `chat` agent granted `tools`, output `${chat.output}`; with `after`, a summarizing agent reads
 * it. START declares the inbox `notes: Inbox(str)`, which `chat` reads as its steer inbox unless
 * `steerInbox` is false.
 */
function chatFlow(
  options: { steerInbox?: boolean; tools?: readonly string[]; after?: boolean; steerRender?: string } = {},
): CompiledFlow {
  const { steerInbox = true, tools = [], after = false, steerRender = null } = options;
  const start = new StartNode(START_ID, { params: [NOTES] });
  const chat = new AgentNode("chat", { prompt: "Help the user.", llmConfig: CONFIG, steerInbox, steerRender, tools: [...tools] });
  const chatWiring = steerInbox ? { [STEER_INBOX_KEY]: "${input.notes}" } : {};
  if (!after) {
    return graph([start, chat], [[START_ID, "chat"]], {
      wiring: { chat: chatWiring },
      outputs: [new FlowOutput("output", "${chat.output}")],
    });
  }
  const next = new AgentNode("after", { prompt: "summarize: ${draft}", llmConfig: CONFIG });
  next.params = [new Parameter("draft")];
  return graph(
    [start, chat, next],
    [
      [START_ID, "chat"],
      ["chat", "after"],
    ],
    {
      wiring: { chat: chatWiring, after: { draft: "${chat.output}" } },
      outputs: [new FlowOutput("output", "${after.output}")],
    },
  );
}

/** The START's inbox `notes: Inbox(str)`. */
const NOTES = new Parameter("notes", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.STRING))) });

/** The engine `run` drives, so a tool can send to its inbox. */
let running: FlowEngine | null = null;

/** Send `event` to the running engine's `notes` inbox, as a host does while the run is live. */
function steer(event: string): void {
  running!.deliver({ kind: "send", inbox: "notes", event });
}

/** A steer inbox opener that never has anything queued, for a node driven without an engine. */
const NO_STEERS = (inbox: string): SteerInbox => ({ inbox, queued: () => false, read: () => null, interrupt: new AbortController().signal, interruptedBy: () => null });

/**
 * One loaded `chat` agent granted `tools`, output `${chat.output}`; with
 * `after`, a summarizing agent reads it.
 */
function steerFlow(options: { tools?: readonly string[]; after?: boolean } = {}): string {
  const { tools = [], after = false } = options;
  return (
    "id: steer\nname: steer\nllm_config: {provider: anthropic, model: claude-test}\n" +
    "input:\n  notes: !Inbox(str)\nnodes:\n" +
    "  chat:\n    kind: agent\n    steer_inbox: ${input.notes}\n" +
    `    tools: [${tools.join(", ")}]\n    prompt: Help the user.\n    output: str\n` +
    (after
      ? '  after:\n    kind: agent\n    input: {draft: "${chat.output}"}\n    prompt: "summarize: ${draft}"\n' +
        "    output: str\noutput: ${after.output}\n"
      : "output: ${chat.output}\n")
  );
}

/**
 * Run `flow` with `model` and `options`, after sending `before` to its `notes` inbox; return the
 * run's output, failing unless it succeeded. The engine is `running` while it runs, and after.
 */
async function run(
  flow: CompiledFlow,
  model: ChatModel,
  options: {
    tools?: ToolSet;
    onEvent?: (event: unknown) => void;
    onStream?: (chunk: StreamChunk) => void;
    numWorkers?: number;
    scheduler?: Scheduler;
    before?: readonly string[];
  } = {},
): Promise<unknown> {
  const { before = [], ...rest } = options;
  running = inboxEngine(flow, { models: serving(model), ...rest });
  for (const event of before) commitPrepared(running.log, running.state, prepareSend(running.state, "notes", event));
  const last = (await running.run()).at(-1);
  expect(last).toBeInstanceOf(RunSucceeded);
  return (last as RunSucceeded).output;
}

/** The unread events of the last `run`'s `notes` inbox, as values. */
function unread(): unknown[] {
  return (unreadEvents(running!.state.waits).get("notes") ?? []).map((seq) => running!.state.inboxEvents.get(seq)!.toObject());
}

/** A direct loop context: agent `chat` prompted "go" with no tools, calling `model`; `extra` overrides. */
function agentContext(model: ChatModel, extra: Partial<AgentRunContext>): AgentRunContext {
  return {
    nodeId: "chat",
    messages: [userMessage("go")],
    tools: [],
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    toolset: null,
    systemPrompt: DEFAULT_SYSTEM,
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

// --- the capability reaches only agents with a steer inbox --------------------- //

test("an agent without a steer inbox never reads it; the event stays unread", async () => {
  const model = new ScriptedModel([reply("done")], CONFIG);
  expect(await run(chatFlow({ steerInbox: false }), model, { before: ["ignored"] })).toBe("done");
  expect(unread()).toEqual(["ignored"]);
  expect(shown(model.requests[0]!.messages).at(-1)).toEqual(["user", "Help the user."]);
});

test("only an agent with a steer inbox opens it", async () => {
  for (const steerInbox of [true, false]) {
    const opened: string[] = [];
    const open = (inbox: string) => (opened.push(inbox), NO_STEERS(inbox));
    const model = new ScriptedModel([reply("ok")], CONFIG);
    const node = new AgentNode("chat", { prompt: "hi", llmConfig: CONFIG, steerInbox });
    const pool = new VariablePool();
    pool.set(START_ID, { notes: new InboxRef("notes") });
    const flow = steerInbox ? chatFlow() : chatFlow({ steerInbox: false });
    await drive(node, { steerInbox: open, models: serving(model), pool, flow });
    expect(opened).toEqual(steerInbox ? ["notes"] : []);
  }
});

test.each([
  ["a structured output", new Type({ kind: ValueKind.OBJECT, fields: new Map([["answer", Type.scalar(ValueKind.INTEGER)]]), required: new Set(["answer"]) })],
  ["an Optional(str) output", new Type({ kind: ValueKind.STRING, nullable: true })],
])("an agent built in code with a steer inbox and %s runs", async (_name, type) => {
  const node = new AgentNode("chat", { prompt: "hi", llmConfig: CONFIG, steerInbox: true });
  node.outputContract = new ValueContract(type);
  const answer = type.kind === ValueKind.STRING ? "ok" : '{"answer": 1}';
  const model = new ScriptedModel([reply(answer)], CONFIG);
  const pool = new VariablePool();
  pool.set(START_ID, { notes: new InboxRef("notes") });
  const outcome = await drive(node, { steerInbox: NO_STEERS, models: serving(model), pool, flow: chatFlow() });
  expect(outcome).toEqual(new Output(type.kind === ValueKind.STRING ? "ok" : { answer: 1n }));
});

// --- steer messages ----------------------------------------------------------- //

test("steers are sent as one user message per read at the top of each iteration", async () => {
  const lookup = lookupTool((topic) => {
    steer("only the 2024 rows");
    steer("and cite them");
    return `found ${topic}`;
  });
  const model = new ScriptedModel([reply("", ["lookup", { topic: "a" }, "call-1"]), reply("done")], CONFIG);
  const tools = new ToolSet({ lookup });
  const flow = chatFlow({ tools: ["lookup"] });
  expect(await run(flow, model, { tools, before: ["use dataset B"] })).toBe("done");
  expect(shown(model.requests[0]!.messages).slice(-2)).toEqual([
    ["user", "Help the user."],
    ["user", "use dataset B"],
  ]);
  const last = model.requests[1]!.messages.at(-1)!;
  expect([last.role, last.parts.map((part) => (part as TextPart).text)]).toEqual([
    "user",
    ["only the 2024 rows", "and cite them"],
  ]);
  expect(model.requests[1]!.messages.at(-2)!.parts[0]).toMatchObject({ content: "found a" });
  const steps = running!.log.records().flatMap(({ record }) => (record instanceof NodeStepRecord ? [record.step["kind"]] : []));
  expect(steps).toEqual(["steer", "model_call", "tool_result", "steer", "model_call"]);
  expect(unread()).toEqual([]);
});

test("the agent's new messages hold each steer read where it joined, and replay rebuilds them", async () => {
  const lookup = lookupTool((topic) => {
    steer("only the 2024 rows");
    steer("and cite them");
    return `found ${topic}`;
  });
  const model = new ScriptedModel([reply("", ["lookup", { topic: "a" }, "call-1"]), reply("done")], CONFIG);
  const flow = chatFlow({ tools: ["lookup"] });
  expect(await run(flow, model, { tools: new ToolSet({ lookup }), before: ["use dataset B"] })).toBe("done");
  const value = running!.state.pool.store.get("chat.new_messages")!;
  const shape = (value.toObject() as { role: string; parts: { type: string; text?: string }[] }[]).map((message) => [
    message.role,
    message.parts.map((part) => part.text ?? part.type),
  ]);
  expect(shape).toEqual([
    ["user", ["use dataset B"]],
    ["assistant", ["tool_call"]],
    ["tool", ["tool_result"]],
    ["user", ["only the 2024 rows", "and cite them"]],
    ["assistant", ["done"]],
  ]);
  const replayed = replayRecords(new RunState(copyGraph(flow)), running!.log.records(), { growBuilder: noGrowth });
  expect(logJson(replayed.pool.store.get("chat.new_messages")!.toWire())).toBe(logJson(value.toWire()));
});

/** The `[role, texts]` of the last run's `chat.new_messages`. */
function newMessages(state: RunState): [string, string[]][] {
  const value = state.pool.store.get("chat.new_messages")!.toObject() as { role: string; parts: { text?: string; type: string }[] }[];
  return value.map((message) => [message.role, message.parts.map((part) => part.text ?? part.type)]);
}

test("a rendered steer appears in new_messages as its rendered text; replay rebuilds it", async () => {
  const model = new ScriptedModel([reply("done")], CONFIG);
  const flow = chatFlow({ steerRender: "Owner: ${event}" });
  expect(await run(flow, model, { before: ["use B", "cite"] })).toBe("done");
  expect(newMessages(running!.state)).toEqual([
    ["user", ["Owner: use B", "Owner: cite"]],
    ["assistant", ["done"]],
  ]);
  expect(model.requests[0]!.messages.at(-1)!.parts.map((part) => (part as TextPart).text)).toEqual(["Owner: use B", "Owner: cite"]);
  const replayed = replayRecords(new RunState(copyGraph(flow)), running!.log.records(), { growBuilder: noGrowth });
  expect(newMessages(replayed)).toEqual(newMessages(running!.state));
});

test("a render that falls back live falls back the same way in new_messages", async () => {
  const model = new ScriptedModel([reply("done")], CONFIG);
  // A str event has no field, so the render fails for it and the event goes as is.
  const flow = chatFlow({ steerRender: "Owner: ${event.text}" });
  expect(await withWarnings(() => {}, () => run(flow, model, { before: ["use B"] }))).toBe("done");
  expect(newMessages(running!.state)).toEqual([
    ["user", ["use B"]],
    ["assistant", ["done"]],
  ]);
});

test("a steer queued between two calls of a reply joins after both results", async () => {
  let calls = 0;
  const lookup = lookupTool((topic) => {
    if (++calls === 1) steer("mid-turn steer");
    return `found ${topic}`;
  });
  const model = new ScriptedModel([twoCalls(), reply("done")], CONFIG);
  expect(await run(chatFlow({ tools: ["lookup"] }), model, { tools: new ToolSet({ lookup }) })).toBe("done");
  const roles = model.requests[1]!.messages.slice(-4).map((message) => message.role);
  expect(roles).toEqual(["assistant", "tool", "tool", "user"]);
  expect(model.requests[1]!.messages.at(-1)!.text()).toBe("mid-turn steer");
});

// --- live frames ------------------------------------------------------------ //

test("after a steer read, the next call's live frames carry its model_call step's n, so a view drops it at that step", async () => {
  const log = join(dir, "live-after-steer.jsonl");
  const flow = await loadFlow(steerFlow());
  const live = new LiveChannel();
  const view = LiveView.from([], { last_seq: -1, calls: [] });
  const frames: LiveFrame[] = [];
  let gaps = 0;
  live.subscribe((frame) => {
    if (frame.type !== "record") frames.push(frame);
    if (!view.apply(frame)) gaps++;
  });
  const onEvent = (event: unknown) => {
    if (event instanceof RunStarted) void send(log, "notes", "use dataset B");
  };
  const result = await runWith(flow, {}, new ScriptedModel([reply("done")], CONFIG), { log, onEvent, live });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const steps = readRunLog(log).flatMap(({ record }) => (record instanceof NodeStepRecord ? [[record.step["kind"], record.n]] : []));
  expect(steps).toEqual([
    ["steer", 1],
    ["model_call", 2],
  ]);
  expect(frames.map((frame) => [frame.type, (frame as { n: number }).n])).toEqual([
    ["start", 2],
    ["chunk", 2],
    ["end", 2],
  ]);
  expect(gaps).toBe(0);
  expect(view.calls()).toEqual([]);
});

// --- resume ----------------------------------------------------------------- //

test("resume reuses a finished node with a steer inbox without reading it again", async () => {
  const log = join(dir, "resume.jsonl");
  const flow = await loadFlow(steerFlow({ after: true }));
  const first = new CrashAt([reply("chat done"), reply("never recorded")], 2, CONFIG);
  const onEvent = (event: unknown) => {
    if (event instanceof RunStarted) void send(log, "notes", "use dataset B");
  };
  await crashes(first, () => runWith(flow, {}, first, { log, onEvent }));
  expect(shown(first.requests[0]!.messages).at(-1)).toEqual(["user", "use dataset B"]);

  await send(log, "notes", "new steer", { env: NO_ENV });
  const model = new ScriptedModel([reply("summary")], CONFIG);
  const resumed = await resumeWith(flow, log, model);
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("summary");
  expect(shown(model.requests[0]!.messages).at(-1)).toEqual(["user", "summarize: chat done"]);
  // The steer sent while down stays unread: the finished agent never reads again.
  expect([...resumed.unread.keys()]).toEqual(["notes"]);
});
