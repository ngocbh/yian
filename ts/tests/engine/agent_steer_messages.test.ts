/**
 * An AGENT's steer inbox: the reserved `__steer_inbox` wiring, how a steer read becomes model
 * messages, the loop's reads before each model call, and replay of recorded steers. TS-only:
 * Python yian has no inboxes.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunEvent, RunStarted } from "../../src/events.ts";

import type { ChatModel } from "../../src/llm/events.ts";
import { Message as ModelMessage, Request, type Response, TextPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { type AgentRunContext, DEFAULT_SYSTEM, requestSha256 } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { steerMessages } from "../../src/nodes/agent/messages.ts";
import { AgentNode, STEER_INBOX_KEY } from "../../src/nodes/agent/node.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import type { Output, SteerInbox } from "../../src/nodes/base.ts";
import { Binding, NodeSteps } from "../../src/nodes/contracts.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { replayRunLog, type RunFlowOptions, send } from "../../src/run.ts";
import { JsonlRunLog, NodeStepRecord, PoolCommitRecord, readRunLog } from "../../src/runtime/log.ts";
import type { RunState } from "../../src/runtime/state.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { FileRef, InboxRef, type ValueRecord } from "../../src/typesys/values.ts";
import { callReply, crashAfter, NO_ENV, resumeWith, runWith, serving, textReply } from "./run_state_fixtures.ts";

/** A pool whose START holds the handle of inbox `name`, as START binds an Inbox input. */
function poolWithInbox(name: string): VariablePool {
  const pool = new VariablePool("__start__");
  pool.set("__start__", { [name]: new InboxRef(name) });
  return pool;
}

test("an agent with a steer inbox reserves its key, hides it from the model and reads its inbox", () => {
  const node = new AgentNode("chat", { prompt: "hi", steerInbox: true });
  expect(node.readsSteerInbox).toBe(true);
  expect([...node.reservedWiringKeys()]).toEqual([STEER_INBOX_KEY]);
  expect(node.reservedKeyLabel(STEER_INBOX_KEY)).toBe("steer_inbox");
  expect(node.modelHiddenInputs().has(STEER_INBOX_KEY)).toBe(true);
  expect(node.backwardInputNames()).not.toContain(STEER_INBOX_KEY);
  expect(new AgentNode("plain", { prompt: "hi" }).readsSteerInbox).toBe(false);
});

test("an agent keeps its steer render; a render without a steer inbox, or an empty one, is refused", () => {
  expect(new AgentNode("c", { steerInbox: true, steerRender: "Note: ${event}" }).steerRender).toBe("Note: ${event}");
  expect(new AgentNode("c", { steerInbox: true }).steerRender).toBeNull();
  expect(() => new AgentNode("c", { steerRender: "x" })).toThrow(new RangeError("AGENT node 'c': a steer render needs a steer inbox"));
  expect(() => new AgentNode("c", { steerInbox: true, steerRender: "" })).toThrow(
    new TypeError("AGENT node 'c': steer render must be non-empty text or null"),
  );
});

test("bindReserved resolves the steer inbox binding to the inbox name", () => {
  const node = new AgentNode("chat", { prompt: "hi", steerInbox: true });
  const wiring = new Map([[STEER_INBOX_KEY, new Binding("${input.m}")]]);
  expect(node.bindReserved(wiring, poolWithInbox("m"))).toEqual({ [STEER_INBOX_KEY]: "m" });
});

test("bindReserved refuses a steer inbox binding that isn't an inbox", () => {
  const node = new AgentNode("chat", { prompt: "hi", steerInbox: true });
  const pool = new VariablePool("__start__");
  pool.set("__start__", { m: "text" });
  const wiring = new Map([[STEER_INBOX_KEY, new Binding("${input.m}")]]);
  expect(() => node.bindReserved(wiring, pool)).toThrow("AGENT node 'chat': steer_inbox must be an inbox");
});

/** Each message of `messages` as `[role, author, texts]`. */
const shape = (messages: readonly ModelMessage[]) =>
  messages.map((message) => [message.role, message.author, message.parts.map((part) => (part as TextPart).text)]);

test("one read's str and record events share one user message, a text part each", () => {
  expect(shape(steerMessages(["use dataset B", { year: 2024n }]))).toEqual([
    ["user", null, ["use dataset B", '{"year":2024}']],
  ]);
});

test("a Message event is its own user message, author and parts kept", () => {
  const message = { role: "user", author: "ana", parts: [{ type: "text", text: "hi" }, { type: "text", text: "there" }] };
  expect(shape(steerMessages([message]))).toEqual([["user", "ana", ["hi", "there"]]]);
});

test("Message events and other events keep their order", () => {
  const message = { role: "user", parts: [{ type: "text", text: "m" }] };
  expect(shape(steerMessages(["a", message, "b", "c"]))).toEqual([
    ["user", null, ["a"]],
    ["user", null, ["m"]],
    ["user", null, ["b", "c"]],
  ]);
});

test("two Message events in one read are two user messages, in send order, authors kept", () => {
  const from = (author: string, text: string) => ({ role: "user", author, parts: [{ type: "text", text }] });
  expect(shape(steerMessages([from("ana", "one"), from("bo", "two")]))).toEqual([
    ["user", "ana", ["one"]],
    ["user", "bo", ["two"]],
  ]);
});

test("a Message steer with another role becomes a user message, author kept", () => {
  const message = { role: "assistant", author: "ana", parts: [{ type: "text", text: "x" }] };
  expect(shape(steerMessages([message]))).toEqual([["user", "ana", ["x"]]]);
});

test("a tool call part in a steer Message becomes text; nothing throws", () => {
  const message = {
    role: "assistant",
    parts: [
      { type: "reasoning", text: "thinking" },
      { type: "tool_call", id: "c1", name: "lookup", arguments: { q: "x" } },
    ],
  };
  expect(shape(steerMessages([message]))).toEqual([
    ["user", null, ["thinking", '{"type":"tool_call","id":"c1","name":"lookup","arguments":{"q":"x"}}']],
  ]);
});

test("a Message steer's media is dropped and reported, never a failure", () => {
  const media = { role: "user", parts: [{ type: "media", mime: "image/png", file: new FileRef({ uri: "files/a.png", mime: "image/png", name: null }) }] };
  const drops: [number, number][] = [];
  const messages = steerMessages(["first", media], (event, part) => drops.push([event, part]));
  expect(drops).toEqual([[1, 0]]);
  expect(messages).toHaveLength(2);
  expect(messages[1]!.parts).toHaveLength(1);
  expect((messages[1]!.parts[0] as TextPart).text).toContain("image/png");
});

test("an empty author is left out", () => {
  const message = { role: "user", author: "", parts: [{ type: "text", text: "x" }] };
  expect(shape(steerMessages([message]))).toEqual([["user", null, ["x"]]]);
});

// --- the loop's reads (runAgent with a steer inbox double) --- //

/** A steer inbox double: `queue` is what a read takes; `reads` logs each read's values, `null` for none. */
function inboxDouble(queue: string[] = []) {
  const reads: (readonly unknown[] | null)[] = [];
  let seq = 100;
  const inbox: SteerInbox = {
    inbox: "notes",
    queued: () => queue.length > 0,
    read: () => {
      if (queue.length === 0) {
        reads.push(null);
        return null;
      }
      const values = queue.splice(0);
      reads.push(values);
      return { n: 0, from: values.map(() => seq++), values };
    },
  };
  return { inbox, queue, reads };
}

const CONFIG = { provider: "anthropic", model: "claude-test" };
const SYSTEM = new ModelMessage({ role: "system", parts: [new TextPart(DEFAULT_SYSTEM)] });

/** A lookup tool returning `found <topic>`, running `side` first. */
function lookup(side: () => void = () => {}): FunctionTool {
  return new FunctionTool({
    name: "lookup",
    description: "Look up a topic.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => {
      side();
      return `found ${String(args["topic"])}`;
    },
  });
}

/** A direct loop context: agent `chat` prompted "go", granted `lookup` when `tool` is given. */
function loopContext(model: ChatModel, extra: Partial<AgentRunContext> & { tool?: FunctionTool } = {}): AgentRunContext {
  const { tool, ...rest } = extra;
  return {
    nodeId: "chat",
    messages: [userMessage("go")],
    promptText: "go",
    tools: tool === undefined ? [] : ["lookup"],
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    memoryContext: new MemoryContext(),
    toolset: tool === undefined ? null : new ToolSet({ lookup: tool }),
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: serving(model),
    sent: () => [],
    ...rest,
  };
}

/** Engine-like steps over `prior`, recording each new step into `recorded`. */
function recording(prior: ValueRecord[] = [], recorded: ValueRecord[] = []): NodeSteps {
  return new NodeSteps(1, prior, (step) => prior.length + recorded.push(step));
}

/** A recorded steer step reading `values`, as the engine's copy carries them. */
const steerStep = (values: string[]): ValueRecord => ({ kind: "steer", inbox: "notes", from: values.map((_, i) => i + 1), values });

/** Each message of `request` as `[role, texts]`. */
const sent = (request: Request) => request.messages.map((message) => [message.role, message.parts.map((part) => (part as TextPart).text ?? "")]);

test("queued steers join the first request as one user message, after the prompt", async () => {
  const { inbox, reads } = inboxDouble(["a", "b"]);
  const model = new ScriptedModel([textReply("done")], CONFIG);
  expect(((await runAgent(loopContext(model, { steerInbox: inbox }))) as Output).value).toBe("done");
  expect(sent(model.requests[0]!)).toEqual([
    ["system", [DEFAULT_SYSTEM]],
    ["user", ["go"]],
    ["user", ["a", "b"]],
  ]);
  expect(reads).toEqual([["a", "b"]]);
});

test("steers queued while a tool runs join the next request after the tool result", async () => {
  const double = inboxDouble();
  const tool = lookup(() => double.queue.push("x", "y"));
  const model = new ScriptedModel([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], CONFIG);
  await runAgent(loopContext(model, { steerInbox: double.inbox, tool }));
  expect(model.requests[1]!.messages.slice(-3).map((message) => message.role)).toEqual(["assistant", "tool", "user"]);
  expect(sent(model.requests[1]!).at(-1)).toEqual(["user", ["x", "y"]]);
});

test("nothing queued: a read per iteration, no extra message", async () => {
  const { inbox, reads } = inboxDouble();
  const model = new ScriptedModel([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], CONFIG);
  await runAgent(loopContext(model, { steerInbox: inbox, tool: lookup() }));
  expect(reads).toEqual([null, null]);
  expect(model.requests[0]!.messages).toHaveLength(2);
});

test("a replaying loop takes recorded steers and reads live only after replay", async () => {
  // The first start reads "a", calls lookup and records the call and its result.
  const first = inboxDouble(["a"]);
  const recorded: ValueRecord[] = [];
  const crashing = new ScriptedModel([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], CONFIG);
  await runAgent(loopContext(crashing, { steerInbox: first.inbox, tool: lookup(), steps: recording([], recorded) }));
  // A start that resumes after the tool result: the engine hands the steer back with its values.
  const prior = [steerStep(["a"]), recorded[0]!, recorded[1]!];
  const double = inboxDouble(["b"]);
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const output = await runAgent(loopContext(model, { steerInbox: double.inbox, tool: lookup(), steps: recording(prior) }));
  expect((output as Output).value).toBe("done");
  expect(model.requests).toHaveLength(1);
  expect(sent(model.requests[0]!).slice(-4).map(([role, texts]) => [role, (texts as string[]).join("|")])).toEqual([
    ["user", "a"],
    ["assistant", ""],
    ["tool", ""],
    ["user", "b"],
  ]);
  expect(double.reads).toEqual([["b"]]);
});

test("a trailing recorded steer and a live read both join the first live call, in that order", async () => {
  const recorded: ValueRecord[] = [];
  const crashing = new ScriptedModel([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], CONFIG);
  await runAgent(loopContext(crashing, { steerInbox: inboxDouble().inbox, tool: lookup(), steps: recording([], recorded) }));
  const prior = [recorded[0]!, recorded[1]!, steerStep(["a"])];
  const double = inboxDouble(["b"]);
  const model = new ScriptedModel([textReply("done")], CONFIG);
  await runAgent(loopContext(model, { steerInbox: double.inbox, tool: lookup(), steps: recording(prior) }));
  expect(sent(model.requests[0]!).slice(-2)).toEqual([
    ["user", ["a"]],
    ["user", ["b"]],
  ]);
});

test("no live read while recorded attempts continue the call; the queued steer joins late, after its answer", async () => {
  const request = new Request({ messages: [SYSTEM, userMessage("go"), ...steerMessages(["a"])], tools: [] });
  const failed: ValueRecord = {
    kind: "model_call",
    purpose: "chat",
    model: "claude-test",
    request_sha256: requestSha256(request),
    started_at: "2026-10-22T10:00:00.000Z",
    first_token_at: null,
    output: null,
    usage: null,
    error: { kind: "server", message: "busy" },
    retry_in: 0,
  };
  const double = inboxDouble(["b"]);
  const model = new ScriptedModel([textReply("done"), textReply("done with b")], CONFIG);
  const steps = recording([steerStep(["a"]), failed]);
  expect(((await runAgent(loopContext(model, { steerInbox: double.inbox, steps }))) as Output).value).toBe("done with b");
  expect(sent(model.requests[0]!).slice(-1)).toEqual([["user", ["a"]]]);
  expect(sent(model.requests[1]!).slice(-2)).toEqual([
    ["assistant", ["done"]],
    ["user", ["b"]],
  ]);
  expect(double.reads).toEqual([["b"]]);
});

// --- through runFlow and advance --- //

const root = mkdtempSync(join(tmpdir(), "yian-steer-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

/** A fresh run folder's log path. */
const freshLog = () => join(mkdtempSync(join(root, "run-")), "log.jsonl");

/** One agent `chat` reading steer inbox `notes`, granted `lookup`; with `take`, a take after it also reads `notes`. */
function steerFlow(options: { steerInbox?: boolean; take?: boolean } = {}): string {
  const { steerInbox = true, take = false } = options;
  return `
id: steer
name: steer
llm_config: {provider: anthropic, model: claude-test}
input:
  notes: !Inbox(str)
nodes:
  chat:
    kind: agent
${steerInbox ? "    steer_inbox: ${input.notes}\n" : ""}    tools: [lookup]
    system_prompt: Help.
    prompt: Help the user.
    output: str
${take ? "  heard:\n    kind: tool\n    tool_id: take\n    input:\n      from: ${input.notes}\n    depends_on: [chat]\n" : ""}output: \${chat.output}
`;
}

/** The node step bodies of the log at `log`, each with its seq. */
function stepsOf(log: string): { seq: number; kind: string; step: Record<string, unknown> }[] {
  return readRunLog(log).flatMap(({ seq, record }) =>
    record instanceof NodeStepRecord ? [{ seq, kind: String(record.step["kind"]), step: record.step }] : [],
  );
}

/** The seqs of the sends to `notes` in the log at `log`. */
const sendSeqs = (log: string) =>
  readRunLog(log).flatMap(({ seq, record }) => (record instanceof PoolCommitRecord && record.key === "__start__.notes" ? [seq] : []));

/** The state the steer flow's log at `log` replays to. */
async function replayed(log: string): Promise<RunState> {
  const authority = JsonlRunLog.resume(log);
  try {
    return replayRunLog(await loadFlow(steerFlow()), authority);
  } finally {
    authority.close();
  }
}

/** Run `flow` at `log` with `model`, the tool `lookup` and `options`. */
async function runSteer(flow: string, log: string, model: ChatModel, tool: FunctionTool, options: RunFlowOptions = {}) {
  return await runWith(await loadFlow(flow), {}, null, model, { log, tools: { lookup: tool }, ...options });
}

test("a send while the agent runs joins the next request and writes steer {inbox, from}", async () => {
  const log = freshLog();
  const tool = lookup(() => void send(log, "notes", "only 2024"));
  const model = new ScriptedModel([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], CONFIG);
  const result = await runSteer(steerFlow(), log, model, tool);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(sent(model.requests[1]!).at(-1)).toEqual(["user", ["only 2024"]]);
  const [seq] = sendSeqs(log);
  expect(stepsOf(log).map(({ kind, step }) => (kind === "steer" ? step : kind))).toEqual([
    "model_call",
    "tool_result",
    { kind: "steer", inbox: "notes", from: [BigInt(seq!)] },
    "model_call",
  ]);
  expect(result.unread).toEqual(new Map());
});

test("a send before the run joins the first request", async () => {
  const log = freshLog();
  const onEvent = (event: RunEvent) => {
    if (event instanceof RunStarted) void send(log, "notes", "be brief");
  };
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const result = await runSteer(steerFlow(), log, model, lookup(), { onEvent });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(sent(model.requests[0]!).slice(-2)).toEqual([
    ["user", ["Help the user."]],
    ["user", ["be brief"]],
  ]);
});

test("an agent without steer_inbox never reads; the take after it gets the event", async () => {
  const log = freshLog();
  const tool = lookup(() => void send(log, "notes", "for the take"));
  const model = new ScriptedModel([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], CONFIG);
  const result = await runSteer(steerFlow({ steerInbox: false, take: true }), log, model, tool);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(model.requests[1]!.messages.at(-1)!.role).toBe("tool");
  expect(stepsOf(log).map(({ kind }) => kind)).not.toContain("steer");
  expect(result.unread).toEqual(new Map());
});

test("steer values are dropped when the agent settles", async () => {
  const log = freshLog();
  const tool = lookup(() => void send(log, "notes", "only 2024"));
  const model = new ScriptedModel([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], CONFIG);
  await runSteer(steerFlow(), log, model, tool);
  const state = await replayed(log);
  expect(state.inboxEvents.has(sendSeqs(log)[0]!)).toBe(false);
});

test("a send that lands while the model call is in flight joins the next read, not that request", async () => {
  const log = freshLog();
  const inner = new ScriptedModel([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], CONFIG);
  let calls = 0;
  const model: ChatModel = {
    provider: inner.provider,
    model: inner.model,
    stream: (request, options) => {
      if (++calls === 1) void send(log, "notes", "mid-call");
      return inner.stream(request, options);
    },
  };
  const result = await runSteer(steerFlow(), log, model, lookup());
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(inner.requests[0]!.messages.at(-1)!.role).toBe("user");
  expect(sent(inner.requests[0]!).at(-1)).toEqual(["user", ["Help the user."]]);
  expect(sent(inner.requests[1]!).at(-1)).toEqual(["user", ["mid-call"]]);
  expect(stepsOf(log).map(({ kind }) => kind)).toEqual(["model_call", "tool_result", "steer", "model_call"]);
});

test("several sends while the run is live are each read once, none lost", async () => {
  const log = freshLog();
  let call = 0;
  const tool = lookup(() => {
    call += 1;
    void send(log, "notes", `s${call}a`);
    void send(log, "notes", `s${call}b`);
  });
  const model = new ScriptedModel(
    [callReply(["lookup", { topic: "a" }, "c1"]), callReply(["lookup", { topic: "b" }, "c2"]), textReply("done")],
    CONFIG,
  );
  const result = await runSteer(steerFlow(), log, model, tool);
  expect(result.status, result.failure?.message).toBe("succeeded");
  const read = stepsOf(log).flatMap(({ kind, step }) => (kind === "steer" ? (step["from"] as number[]).map(Number) : []));
  expect(read).toEqual(sendSeqs(log));
  expect(result.unread).toEqual(new Map());
});

// --- crash and replay --- //

test("crash after a steer step: advance replays the steer, the digest matches, no host fault", async () => {
  const replies = () => [callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")];
  const first = freshLog();
  const sending = (log: string) => lookup(() => void send(log, "notes", "a"));
  await runSteer(steerFlow(), first, new ScriptedModel(replies(), CONFIG), sending(first));
  const steerSeq = stepsOf(first).find(({ kind }) => kind === "steer")!.seq;
  const log = freshLog();
  await crashAfter(steerSeq + 1, () => runSteer(steerFlow(), log, new ScriptedModel(replies(), CONFIG), sending(log)));
  await send(log, "notes", "b", { env: NO_ENV });
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resumeWith(await loadFlow(steerFlow()), log, null, model, { tools: { lookup: lookup() } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(model.requests).toHaveLength(1);
  expect(sent(model.requests[0]!).slice(-2)).toEqual([
    ["user", ["a"]],
    ["user", ["b"]],
  ]);
  const steers = stepsOf(log).filter(({ kind }) => kind === "steer");
  const [sa, sb] = sendSeqs(log);
  expect(steers.map(({ step }) => (step["from"] as number[]).map(Number))).toEqual([[sa!], [sb!]]);
});

test("crash after the model_call that read the steer: replay takes the recorded reply", async () => {
  const replies = () => [callReply(["lookup", { topic: "t" }, "c1"]), callReply(["lookup", { topic: "u" }, "c2"]), textReply("done")];
  const sending = (log: string) => {
    let calls = 0;
    return lookup(() => {
      if (++calls === 1) void send(log, "notes", "a");
    });
  };
  const first = freshLog();
  await runSteer(steerFlow(), first, new ScriptedModel(replies(), CONFIG), sending(first));
  const steered = stepsOf(first).findIndex(({ kind }) => kind === "steer");
  const call = stepsOf(first)[steered + 1]!;
  expect(call.kind).toBe("model_call");
  const log = freshLog();
  await crashAfter(call.seq + 1, () => runSteer(steerFlow(), log, new ScriptedModel(replies(), CONFIG), sending(log)));
  // The reply that read the steer is replayed; only the last call is made.
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resumeWith(await loadFlow(steerFlow()), log, null, model, { tools: { lookup: lookup() } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(model.requests).toHaveLength(1);
  expect(stepsOf(log).filter(({ kind }) => kind === "steer")).toHaveLength(1);
});

test("a steer read, more sends, then a crash and advance: the read values survive", async () => {
  const replies = () => [callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")];
  const first = freshLog();
  const sending = (log: string) => lookup(() => void send(log, "notes", "a"));
  await runSteer(steerFlow(), first, new ScriptedModel(replies(), CONFIG), sending(first));
  const steerSeq = stepsOf(first).find(({ kind }) => kind === "steer")!.seq;
  const log = freshLog();
  await crashAfter(steerSeq + 1, () => runSteer(steerFlow(), log, new ScriptedModel(replies(), CONFIG), sending(log)));
  await send(log, "notes", "b", { env: NO_ENV });
  await send(log, "notes", "c", { env: NO_ENV });
  const state = await replayed(log);
  const [sa] = sendSeqs(log);
  expect(state.inboxEvents.get(sa!)!.toObject()).toBe("a");
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await resumeWith(await loadFlow(steerFlow()), log, null, model, { tools: { lookup: lookup() } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(sent(model.requests[0]!).slice(-2)).toEqual([
    ["user", ["a"]],
    ["user", ["b", "c"]],
  ]);
});

/** A request's messages as `[role, text]` pairs, a tool result as its content. */
const requestShape = (request: Request) =>
  request.messages.map((message) => [
    message.role,
    message.parts.map((part) => (part instanceof ToolResultPart ? part.content : ((part as TextPart).text ?? ""))).join("|"),
  ]);

/**
 * Cut a steered run after every record from the agent's start on, then advance it: the same
 * output, each send read by exactly one steer step, and, when no call was closed and no send was
 * lost with the crash, the resumed model sees the requests the uncut run sent from that call on.
 * `replies` is the model's script; the `i`th lookup call sends `sends[i]`, when there is one.
 */
async function crashEverywhere(replies: Response[], sends: readonly string[]): Promise<void> {
  const sending = (log: string) => {
    let calls = 0;
    return lookup(() => {
      const event = sends[calls++];
      if (event !== undefined) void send(log, "notes", event).catch(() => {});
    });
  };
  const uncut = freshLog();
  const uncutModel = new ScriptedModel([...replies], CONFIG);
  await runSteer(steerFlow(), uncut, uncutModel, sending(uncut));
  expect(sendSeqs(uncut)).toHaveLength(sends.length);
  const total = readRunLog(uncut).length;
  const chatStarted = readRunLog(uncut).findIndex(
    ({ kind, record }) => kind === "node_started" && (record as { nodeId?: string }).nodeId === "chat",
  );
  expect(chatStarted).toBeGreaterThan(0);
  let compared = 0;
  for (let k = chatStarted + 1; k < total; k++) {
    const log = freshLog();
    await crashAfter(k, () => runSteer(steerFlow(), log, new ScriptedModel([...replies], CONFIG), sending(log)));
    // A send whose record the crash refused is lost; the host sends it again while down.
    const lost = sends.slice(sendSeqs(log).length);
    for (const event of lost) await send(log, "notes", event, { env: NO_ENV });
    const made = stepsOf(log).filter(({ kind, step }) => kind === "model_call" && !Object.hasOwn(step, "error")).length;
    const model = new ScriptedModel(replies.slice(made), CONFIG);
    const resumed = await resumeWith(await loadFlow(steerFlow()), log, null, model, { tools: { lookup: lookup() } });
    expect([k, resumed.status, resumed.output]).toEqual([k, "succeeded", "done"]);
    const read = stepsOf(log).flatMap(({ kind, step }) => (kind === "steer" ? (step["from"] as bigint[]).map(Number) : []));
    expect([k, read]).toEqual([k, sendSeqs(log)]);
    expect([k, resumed.unread]).toEqual([k, new Map()]);
    const closed = stepsOf(log).some(({ kind, step }) => kind === "tool_result" && Object.hasOwn(step, "closed"));
    if (!closed && lost.length === 0) {
      expect([k, model.requests.map(requestShape)]).toEqual([k, uncutModel.requests.slice(made).map(requestShape)]);
      compared += 1;
    }
  }
  expect(compared).toBeGreaterThan(0);
}

test("a crash after every record, then advance: the same output, and each send read by one steer step", async () => {
  await crashEverywhere([callReply(["lookup", { topic: "t" }, "c1"]), textReply("done")], ["a"]);
});

test("two sends across two replies, cut after every record: each read once, the live requests unchanged", async () => {
  const replies = [callReply(["lookup", { topic: "t" }, "c1"]), callReply(["lookup", { topic: "u" }, "c2"]), textReply("done")];
  await crashEverywhere(replies, ["a", "b"]);
});
