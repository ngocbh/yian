/**
 * TS-only: an AGENT's success writes its output commit, its messages commit, then `node_succeeded`
 * naming both; the reducer checks the steps and rebuilds `<node>.new_messages`.
 *
 * The messages commit holds no value but `steps`: the seqs of the agent's steps its new messages
 * are made of (each chat reply that arrived, every tool result, and a structured agent's accepted
 * call). The reducer derives the list itself and refuses any other; it rebuilds the value, a
 * `list(Message)`, when the commit is prepared, puts it in the pool and the producer index on the
 * success (and, in a learnable run, gives the trace a second current writer), and drops it with
 * its agent on a prune. A crash before the success leaves orphans, and the agent reruns.
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { SourceRef } from "../../src/expr/lineage.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { checkConversation, fromModelMessage, type Message as MessageValue } from "../../src/nodes/agent/messages.ts";
import { virtualScheduler } from "../../src/platform.ts";
import {
  decodeRecord,
  MemoryRunLog,
  NodeStepRecord,
  NodeSucceededRecord,
  PoolCommitRecord,
  readRunLog,
  type RecordEnvelope,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  poolResolver,
  prepareMessagesCommit,
  prepareNodeSucceeded,
  prepareRecord,
  type RunState,
} from "../../src/runtime/state.ts";
import { advance, foldRunLog, replayRunLog, runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { logJson } from "../../src/typesys/json.ts";
import { crashAfter, NO_ENV, serving, textReply } from "./run_state_fixtures.ts";

const root = mkdtempSync(join(tmpdir(), "yian-agent-messages-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };
const START = Date.UTC(2026, 9, 24, 10, 0, 0, 0);

/** A flow whose agent `writer` may call `tools`, with output type `output`, then a CODE node `after`. */
function flowText(tools: readonly string[] = ["lookup"], output = "str"): string {
  return `
id: messages
name: messages
nodes:
  writer:
    kind: agent
    tools: [${tools.join(", ")}]
    prompt: hi
    output: ${output}
  after:
    kind: code
    input:
      x: \${writer.output}
    code: return inputs['x']
output: \${after.output}
`;
}

/** A tool `name` returning `<name>:<topic>`. */
function topicTool(name: string): FunctionTool {
  return new FunctionTool({
    name,
    description: `Tool ${name}.`,
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => `${name}:${String(args["topic"])}`,
  });
}

/** A reply calling `parts` (`[id, topic]`) of `lookup` in order. */
function calls(...parts: [string, string][]): Response {
  return new Response({
    message: new Message({
      role: "assistant",
      parts: parts.map(([id, topic]) => new ToolCallPart({ id, name: "lookup", arguments: { topic } })),
    }),
    finishReason: "tool_calls",
  });
}

/** A fresh run folder's log path. */
function runLog(): string {
  return join(mkdtempSync(join(root, "run-")), "log.jsonl");
}

/** The decoded records of the log at `path`. */
function records(path: string): RecordEnvelope[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => decodeRecord(line));
}

/** The seqs of the log's step records of `nodeId` whose step kind is `kind`. */
function stepSeqs(path: string, kind: string, nodeId = "writer"): number[] {
  return records(path).flatMap(({ seq, record }) =>
    record instanceof NodeStepRecord && record.nodeId === nodeId && record.step["kind"] === kind ? [seq] : [],
  );
}

/** The messages commits of the log at `path`, with their seqs. */
function messagesCommits(path: string): { seq: number; record: PoolCommitRecord }[] {
  return records(path).flatMap(({ seq, record }) =>
    record instanceof PoolCommitRecord && record.steps !== null ? [{ seq, record }] : [],
  );
}

/** An environment with no host setting but `YIAN_MODEL_RETRIES`. */
function retriesEnv(modelRetries: number): (name: string) => string | undefined {
  return (name) => (name === "YIAN_MODEL_RETRIES" ? String(modelRetries) : NO_ENV(name));
}

const TOOLS = { lookup: topicTool("lookup") };

/** Run `flow` over `model` with a log, a virtual clock at `START` and `modelRetries`. */
async function run(
  model: ChatModel,
  options: { log?: string; flow?: string; modelRetries?: number; learnable?: boolean } = {},
): Promise<{ log: string; flow: ComposedFlow; result: Awaited<ReturnType<typeof runFlow>> }> {
  const { log = runLog(), flow: text = flowText(), modelRetries = 0, learnable = false } = options;
  const flow = await loadFlow(text);
  const result = await runFlow(flow, {}, {
    llmConfig: CONFIG,
    env: retriesEnv(modelRetries),
    scheduler: virtualScheduler(START),
    random: () => 1,
    models: serving(model),
    runId: "r",
    tools: TOOLS,
    learnable,
    log,
  });
  return { log, flow, result };
}

/** Resume the run at `log` over `model`. */
function resume(log: string, model: ChatModel) {
  return advance({
    log,
    tools: TOOLS,
    env: retriesEnv(0),
    scheduler: virtualScheduler(START + 60_000),
    random: () => 1,
    models: serving(model),
  });
}

/** A one-tool turn: a call, its result, then the text. */
const toolTurn = () => new ScriptedModel([calls(["c1", "a"]), textReply("done")], CONFIG);

/** The state a replay of the first `k` records of the log at `log` leaves. */
function replayed(flow: ComposedFlow, log: string, k: number): RunState {
  return replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(log).slice(0, k)));
}

/** The value the log at `log` folds to at seq `seq`, as plain messages. */
async function foldedAt(log: string, seq: number): Promise<MessageValue[] | null> {
  const folded = await foldRunLog(log, { env: retriesEnv(0) });
  return (folded.records[seq]!.value?.toObject() ?? null) as MessageValue[] | null;
}

// --- the engine's records --- //

test("an AGENT's success is its output commit, its messages commit, then node_succeeded naming both", async () => {
  const { log, result } = await run(toolTurn());
  expect(result.status, result.failure?.message).toBe("succeeded");
  const tail = records(log).filter(({ record }) => (record as { nodeId?: string }).nodeId === "writer").slice(-3);
  const [output, messages, success] = tail.map(({ record }) => record) as [PoolCommitRecord, PoolCommitRecord, NodeSucceededRecord];
  expect([output.key, messages.key]).toEqual(["writer", "writer.new_messages"]);
  expect(messages.value).toBeNull();
  expect([success.commit, success.messages]).toEqual([tail[0]!.seq, tail[1]!.seq]);
  expect(tail[1]!.seq).toBe(tail[0]!.seq + 1);
  expect(tail[2]!.seq).toBe(tail[1]!.seq + 1);
  // The reply, its result, the final reply.
  expect(messages.steps).toEqual([...stepSeqs(log, "model_call"), ...stepSeqs(log, "tool_result")].sort((a, b) => a - b));
  expect(messages.steps).toHaveLength(3);
  // A CODE node's success names no messages commit.
  const after = records(log).find(({ record }) => record instanceof NodeSucceededRecord && record.nodeId === "after")!;
  expect((after.record as NodeSucceededRecord).messages).toBeNull();
  expect(messagesCommits(log)).toHaveLength(1);
});

test("steps leaves out a failed attempt that was retried", async () => {
  const model = new ScriptedModel([new ProviderError("rate_limit", "slow down", { status: 429 }), textReply("ok")], CONFIG);
  const { log, result } = await run(model, { modelRetries: 1 });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const [failed, reply] = stepSeqs(log, "model_call");
  const { record } = messagesCommits(log)[0]!;
  expect(record.steps).toEqual([reply!]);
  expect(record.steps).not.toContain(failed);
});

test("the value is the conversation the agent added, as the model saw it", async () => {
  const model = new ScriptedModel([calls(["c1", "a"], ["c2", "b"]), textReply("done")], CONFIG);
  const { log, result } = await run(model);
  expect(result.status, result.failure?.message).toBe("succeeded");
  const last = model.requests.at(-1)!.messages;
  const added = last.slice(last.findIndex((message) => message.role === "assistant"));
  const expected = [...added, new Message({ role: "assistant", parts: [new TextPart("done")] })].map(fromModelMessage);
  const { seq } = messagesCommits(log)[0]!;
  const value = await foldedAt(log, seq);
  expect(value).toEqual(expected);
  checkConversation(value!);
});

test("a tool-less text agent's messages are its reply", async () => {
  const { log } = await run(new ScriptedModel([textReply("hello")], CONFIG), { flow: flowText([]) });
  const { seq, record } = messagesCommits(log)[0]!;
  expect(record.steps).toEqual(stepSeqs(log, "model_call"));
  expect(await foldedAt(log, seq)).toEqual([{ role: "assistant", parts: [{ type: "text", text: "hello" }] }]);
});

/** A reply whose text is `text`. */
const saying = (text: string) => new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" });

test("a structured agent without tools writes steps: [<its structured call>] and the value [assistant(<JSON>)]", async () => {
  const { log, result } = await run(new ScriptedModel([saying('{"answer": "yes"}')], CONFIG), {
    flow: flowText([], "{answer: str}"),
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const { seq, record } = messagesCommits(log)[0]!;
  expect(record.steps).toEqual(stepSeqs(log, "model_call"));
  expect(await foldedAt(log, seq)).toEqual([{ role: "assistant", parts: [{ type: "text", text: '{"answer":"yes"}' }] }]);
});

test("a structured agent with tools ends with its last chat reply, then the value as JSON", async () => {
  const model = new ScriptedModel([calls(["c1", "a"]), textReply("found it"), saying('{"answer": "yes"}')], CONFIG);
  const { log, result } = await run(model, { flow: flowText(["lookup"], "{answer: str}") });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const { seq } = messagesCommits(log)[0]!;
  const value = (await foldedAt(log, seq))!;
  expect(value.map((message) => message.role)).toEqual(["assistant", "tool", "assistant", "assistant"]);
  expect(value.slice(-2)).toEqual([
    { role: "assistant", parts: [{ type: "text", text: "found it" }] },
    { role: "assistant", parts: [{ type: "text", text: '{"answer":"yes"}' }] },
  ]);
  checkConversation(value);
});

test("a structured agent whose first structured reply was invalid shows only the accepted one", async () => {
  const model = new ScriptedModel([saying("not json"), saying('{"answer": "yes"}')], CONFIG);
  const { log, result } = await run(model, { flow: flowText([], "{answer: str}") });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const [first, accepted] = stepSeqs(log, "model_call");
  const { seq, record } = messagesCommits(log)[0]!;
  expect(record.steps).toEqual([accepted!]);
  expect(record.steps).not.toContain(first);
  expect(await foldedAt(log, seq)).toEqual([{ role: "assistant", parts: [{ type: "text", text: '{"answer":"yes"}' }] }]);
});

test("a failed agent writes no messages commit", async () => {
  const model = new ScriptedModel([new ProviderError("server", "boom", { status: 500 })], CONFIG);
  const { log, result } = await run(model);
  expect(result.status).toBe("failed");
  expect(messagesCommits(log)).toEqual([]);
});

test("an agent whose output fails its contract writes no messages commit", async () => {
  const model = new ScriptedModel([saying("no"), saying("still no"), saying("never")], CONFIG);
  const { log, result } = await run(model, { flow: flowText([], "{answer: int}") });
  expect(result.status).toBe("failed");
  expect(messagesCommits(log)).toEqual([]);
});

// --- the reducer --- //

test("the state keeps each unsettled node's step seqs beside its steps", async () => {
  const { log, flow } = await run(toolTurn());
  const { seq } = messagesCommits(log)[0]!;
  const state = replayed(flow, log, seq);
  const seqs = [...stepSeqs(log, "model_call"), ...stepSeqs(log, "tool_result")].sort((a, b) => a - b);
  expect(state.nodeStepSeqs.get("writer")).toEqual(seqs);
  expect(state.nodeSteps.get("writer")).toHaveLength(seqs.length);
  // The success forgets both.
  const done = replayed(flow, log, readRunLog(log).length);
  expect(done.nodeStepSeqs.has("writer")).toBe(false);
});

test("prepareMessagesCommit names the node's picked steps by seq", async () => {
  const { log, flow } = await run(toolTurn());
  const { seq, record } = messagesCommits(log)[0]!;
  const prepared = prepareMessagesCommit(replayed(flow, log, seq), "writer");
  expect((prepared.record as PoolCommitRecord).steps).toEqual(record.steps);
  expect(prepared.expectedSequence).toBe(seq);
});

test.each<[string, (steps: readonly number[]) => PoolCommitRecord, string]>([
  [
    "other steps",
    (steps) => new PoolCommitRecord("writer", "writer.new_messages", null, { steps: steps.slice(1) }),
    "the messages commit of 'writer' must name steps [STEPS], found [REST]",
  ],
  [
    "another key",
    (steps) => new PoolCommitRecord("writer", "other.new_messages", null, { steps }),
    "the messages commit of 'writer' is under 'other.new_messages', not 'writer.new_messages'",
  ],
  [
    "a node that produces none",
    (steps) => new PoolCommitRecord("after", "after.new_messages", null, { steps }),
    "a messages commit by node 'after', which produces no messages",
  ],
])("a messages commit with %s is refused", async (_name, build, message) => {
  const { log, flow } = await run(toolTurn());
  const { seq, record } = messagesCommits(log)[0]!;
  const steps = record.steps!;
  const state = replayed(flow, log, seq);
  const expected = message.replace("STEPS", steps.join(", ")).replace("REST", steps.slice(1).join(", "));
  expect(() => prepareRecord(state, build(steps))).toThrow(new RangeError(expected));
});

test("a messages commit before the output commit, or a second one, is refused", async () => {
  const { log, flow } = await run(toolTurn());
  const { seq, record } = messagesCommits(log)[0]!;
  expect(() => prepareRecord(replayed(flow, log, seq - 1), record)).toThrow(
    new RangeError("the messages commit of 'writer' must follow its output commit"),
  );
  expect(() => prepareRecord(replayed(flow, log, seq + 1), record)).toThrow(
    new RangeError(`node 'writer' already has a pending messages commit at seq ${seq}`),
  );
});

test("a success of an AGENT without its messages commit is refused, live and on replay", async () => {
  const { log, flow } = await run(toolTurn());
  const { seq } = messagesCommits(log)[0]!;
  const state = replayed(flow, log, seq);
  expect(() => prepareNodeSucceeded(state, "writer", { output: "done" })).toThrow(
    new RangeError("node_succeeded of 'writer' has no messages commit to accept"),
  );
  const success = readRunLog(log)[seq + 1]!.record as NodeSucceededRecord;
  expect(() => prepareRecord(state, new NodeSucceededRecord("writer", success.commit, { messages: seq }))).toThrow(
    new RangeError(`node_succeeded of 'writer' names messages ${seq}, not its pending messages commit (none)`),
  );
  const withMessages = replayed(flow, log, seq + 1);
  expect(() => prepareRecord(withMessages, new NodeSucceededRecord("writer", success.commit))).toThrow(
    new RangeError(`node_succeeded of 'writer' names messages (none), not its pending messages commit ${seq}`),
  );
});

test("the success puts the value under writer.new_messages and points the producer index at the messages commit", async () => {
  const { log, flow } = await run(toolTurn());
  const { seq } = messagesCommits(log)[0]!;
  const state = replayed(flow, log, seq + 1);
  expect(state.pendingMessages.get("writer")?.seq).toBe(seq);
  expect(state.pool.store.has("writer.new_messages")).toBe(false);
  commitPrepared(MemoryRunLog.fromRecords(readRunLog(log).slice(0, seq + 1)), state, prepareNodeSucceeded(state, "writer", { output: "done" }));
  expect(state.pendingMessages.size).toBe(0);
  expect(state.producers.get("writer.new_messages")).toBe(seq);
  const value = state.pool.store.get("writer.new_messages")!.toObject() as MessageValue[];
  expect(value).toEqual((await foldedAt(log, seq))!);
  expect(state.pool.resolve("writer", ["new_messages"])).toEqual(value);
});

// --- crashes --- //

test("a crash between the messages commit and the success reruns the agent; the new commits supersede the orphans", async () => {
  const uncut = await run(toolTurn());
  const { seq } = messagesCommits(uncut.log)[0]!;
  const log = runLog();
  await crashAfter(seq + 1, () => run(toolTurn(), { log }));
  const resumed = await resume(log, new ScriptedModel([], CONFIG));
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  const [orphan, kept] = messagesCommits(log);
  expect(orphan!.seq).toBe(seq);
  const success = records(log).find(({ record }) => record instanceof NodeSucceededRecord && record.nodeId === "writer")!;
  expect((success.record as NodeSucceededRecord).messages).toBe(kept!.seq);
  expect(await foldedAt(log, orphan!.seq)).toBeNull();
  expect(await foldedAt(log, kept!.seq)).toEqual(await foldedAt(uncut.log, seq));
});

test("a crash between the output commit and the messages commit reruns the agent", async () => {
  const uncut = await run(toolTurn());
  const { seq } = messagesCommits(uncut.log)[0]!;
  const log = runLog();
  await crashAfter(seq, () => run(toolTurn(), { log }));
  const resumed = await resume(log, new ScriptedModel([], CONFIG));
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  const [only] = messagesCommits(log);
  expect(messagesCommits(log)).toHaveLength(1);
  expect(await foldedAt(log, only!.seq)).toEqual(await foldedAt(uncut.log, seq));
});

test("a resumed agent (crash after a tool result) rebuilds the messages of both starts in order", async () => {
  const uncut = await run(toolTurn());
  const cut = stepSeqs(uncut.log, "tool_result")[0]! + 1;
  const log = runLog();
  await crashAfter(cut, () => run(toolTurn(), { log }));
  const resumed = await resume(log, new ScriptedModel([textReply("done")], CONFIG));
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  const { seq, record } = messagesCommits(log)[0]!;
  expect(record.steps).toEqual([...stepSeqs(log, "model_call"), ...stepSeqs(log, "tool_result")].sort((a, b) => a - b));
  const value = (await foldedAt(log, seq))!;
  expect(value).toEqual((await foldedAt(uncut.log, messagesCommits(uncut.log)[0]!.seq))!);
  checkConversation(value);
});

test("replay rebuilds the same value, learnable or not", async () => {
  for (const learnable of [false, true]) {
    const { log, flow } = await run(toolTurn(), { learnable });
    const { seq } = messagesCommits(log)[0]!;
    const state = replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(log)));
    // The run ended, so END's success forgot nothing of the pool.
    const value = state.pool.store.get("writer.new_messages")!;
    expect(logJson(value.toWire())).toBe(logJson((await foldRunLog(log, { env: retriesEnv(0) })).records[seq]!.value!.toWire()));
  }
});

// --- growth and the trace --- //

const CHAT_LOOP = `
id: talk
name: talk
defs:
  turn:
    input:
      n: !int
    nodes:
      reply:
        kind: agent
        input:
          n: \${input.n}
        prompt: turn \${n}
        output: str
    output:
      n: \${input.n}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      n: 0
    times: 12
output: \${chat.output.n}
`;

test("a LOOP of agents keeps a bounded pool and producer index", async () => {
  for (const learnable of [false, true]) {
    const model = new ScriptedModel(Array.from({ length: 12 }, (_, index) => textReply(`turn ${index}`)), CONFIG);
    const { log, flow, result } = await run(model, { flow: CHAT_LOOP, learnable });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(messagesCommits(log)).toHaveLength(12);
    const held: number[] = [];
    const writers: number[] = [];
    const state = replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(log)), {
      onPrepared: (_prepared, at) => {
        held.push([...at.pool.store.keys()].filter((key) => key.endsWith(".new_messages")).length);
        if (at.trace !== null) writers.push([...at.trace.currentWriters.keys()].filter((key) => key.endsWith(".new_messages")).length);
      },
    });
    expect(Math.max(...held)).toBeLessThanOrEqual(2);
    if (learnable) expect(Math.max(...writers)).toBeLessThanOrEqual(2);
    expect(state.producers.size).toBe(state.pool.store.size + state.pool.system.size);
  }
});

test("in a learnable run, the success adds the writer x.new_messages, sourced at the messages commit", async () => {
  const { log, flow } = await run(toolTurn(), { learnable: true });
  const { seq } = messagesCommits(log)[0]!;
  const successSeq = seq + 1;
  const state = replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(log)));
  const trace = state.trace!;
  const entry = trace.currentWriters.get("writer.new_messages")!;
  expect(entry.source).toEqual(new SourceRef("r", seq));
  expect(entry.outcomeSeq).toBe(successSeq);
  expect(entry.valueJson).toBe(logJson(state.pool.store.get("writer.new_messages")!.toWire()));
  const occurrence = trace.occurrences.find((each) => each.nodeId === "writer")!;
  expect(occurrence.messagesSource).toEqual(entry.source);
  expect(trace.occurrences.find((each) => each.nodeId === "after")!.messagesSource).toBeNull();
  // A derived binding reading the whole pool resolves the messages through the trace.
  expect(poolResolver(state)("writer.new_messages", seq).source).toEqual(entry.source);
});
