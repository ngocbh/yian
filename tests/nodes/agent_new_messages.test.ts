/**
 * Reading `${x.new_messages}` at run time. A reader gets the AGENT's new messages from its
 * messages commit; a derived reader (a child END, a LOOP's carried value) names that commit in its
 * `from`, learnable or not, and replay rebuilds the same value. A LOOP builds a history from each
 * turn's new messages with a bounded pool, and a skipped agent's read takes its default.
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import type { ChatModel } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { checkConversation, type Message as MessageValue } from "../../src/nodes/agent/messages.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { decodeRecord, MemoryRunLog, PoolCommitRecord, readRunLog, type RecordEnvelope } from "../../src/runtime/log.ts";
import { BackwardAttribution } from "../../src/runtime/contracts.ts";
import {
  addMessage,
  BackwardMessage,
  type PendingMessages,
  takeOccurrenceMessages,
  workFeedback,
} from "../../src/runtime/learning.ts";
import { growthIndex } from "../../src/runtime/state.ts";
import { advance, foldRunLog, replayRunLog, runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { logJson } from "../../src/typesys/json.ts";
import { crashAfter, NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";

const root = mkdtempSync(join(tmpdir(), "yian-agent-new-messages-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };
const START = Date.UTC(2026, 9, 24, 10, 0, 0, 0);

const TOOLS = {
  lookup: new FunctionTool({
    name: "lookup",
    description: "Tool lookup.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => `lookup:${String(args["topic"])}`,
  }),
};

/** A reply calling `lookup` once with id `id`. */
function lookup(id: string, topic: string): Response {
  return new Response({
    message: new Message({ role: "assistant", parts: [new ToolCallPart({ id, name: "lookup", arguments: { topic } })] }),
    finishReason: "tool_calls",
  });
}

/** A one-tool turn: a call, its result, then the text. */
const toolTurn = () => new ScriptedModel([lookup("c1", "a"), textReply("done")], CONFIG);

/** The decoded records of the log at `path`. */
function records(path: string): RecordEnvelope[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => decodeRecord(line));
}

/** The pool commits of the log at `path`, with their seqs. */
function commits(path: string): { seq: number; record: PoolCommitRecord }[] {
  return records(path).flatMap(({ seq, record }) => (record instanceof PoolCommitRecord ? [{ seq, record }] : []));
}

/** The seqs of the messages commits of the log at `path`. */
function messagesSeqs(path: string): number[] {
  return commits(path).flatMap(({ seq, record }) => (record.steps !== null ? [seq] : []));
}

/** Run `text` over `model` with a fresh log. */
async function run(
  model: ChatModel,
  text: string,
  options: { learnable?: boolean; inputs?: Record<string, unknown>; log?: string } = {},
): Promise<{ log: string; flow: ComposedFlow; result: Awaited<ReturnType<typeof runFlow>> }> {
  const log = options.log ?? join(mkdtempSync(join(root, "run-")), "log.jsonl");
  const flow = await loadFlow(text);
  const result = await runFlow(flow, options.inputs ?? {}, {
    llmConfig: CONFIG,
    env: (name) => (name === "YIAN_MODEL_RETRIES" ? "0" : NO_ENV(name)),
    scheduler: virtualScheduler(START),
    random: () => 1,
    models: serving(model),
    runId: "r",
    tools: TOOLS,
    learnable: options.learnable ?? false,
    log,
  });
  expect(result.status, result.failure?.message).toBe("succeeded");
  return { log, flow, result };
}

// --- a derived reader points at the messages commit --- //

const TOP = `
id: f
name: f
nodes:
  reply:
    kind: agent
    tools: [lookup]
    prompt: hi
    output: str
output:
  history: \${reply.new_messages}
  answer: \${reply.output}
`;

test("a flow output reading reply.new_messages gets the conversation the agent added", async () => {
  const { result } = await run(toolTurn(), TOP);
  const history = (result.output as { history: MessageValue[] }).history;
  expect(history.map((message) => message.role)).toEqual(["assistant", "tool", "assistant"]);
  checkConversation(history);
  expect((result.output as { answer: string }).answer).toBe("done");
});

const CHILD = `
id: f
name: f
defs:
  ask:
    nodes:
      reply:
        kind: agent
        tools: [lookup]
        prompt: hi
        output: str
    output: \${reply.new_messages}
nodes:
  c:
    kind: call
    call: ask
output: \${c.output}
`;

test("a CALL child's END reading it writes from: {'<call>/reply.new_messages': <messages seq>}", async () => {
  for (const learnable of [false, true]) {
    const { log, result } = await run(toolTurn(), CHILD, { learnable });
    const [seq] = messagesSeqs(log);
    const end = commits(log).find(({ record }) => record.nodeId === "c/__end__")!.record;
    expect(end.from).toEqual(new Map([["c/reply.new_messages", seq!]]));
    expect((result.output as MessageValue[]).map((message) => message.role)).toEqual(["assistant", "tool", "assistant"]);
  }
});

test("replay rebuilds the reader's value from the messages commit", async () => {
  for (const learnable of [false, true]) {
    const { log, flow, result } = await run(toolTurn(), CHILD, { learnable });
    const end = commits(log).find(({ record }) => record.nodeId === "c/__end__")!;
    const folded = await foldRunLog(log, { env: (name) => (name === "YIAN_MODEL_RETRIES" ? "0" : NO_ENV(name)) });
    expect(folded.records[end.seq]!.value!.toObject()).toEqual(result.output as MessageValue[]);
    const state = replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(log)));
    expect(logJson(state.pool.store.get("c")!.toWire())).toBe(logJson(folded.records[end.seq]!.value!.toWire()));
  }
});

test("a crash between a tool call and its result: the resumed reader's history holds the unknown result", async () => {
  const uncut = await run(toolTurn(), TOP);
  const callSeq = records(uncut.log).find(({ record }) => (record as { step?: { kind?: string } }).step?.kind === "model_call")!.seq;
  const log = join(mkdtempSync(join(root, "run-")), "log.jsonl");
  await crashAfter(callSeq + 1, () => run(toolTurn(), TOP, { log }));
  const resumed = await advance({
    log,
    tools: TOOLS,
    env: (name) => (name === "YIAN_MODEL_RETRIES" ? "0" : NO_ENV(name)),
    scheduler: virtualScheduler(START + 60_000),
    models: serving(new ScriptedModel([textReply("done")], CONFIG)),
  });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  const history = (resumed.output as { history: MessageValue[] }).history;
  expect(history.map((message) => message.role)).toEqual(["assistant", "tool", "assistant"]);
  const result = history[1]!.parts[0] as { type: string; call_id: string; is_error?: boolean; content: string };
  expect([result.type, result.call_id, result.is_error]).toEqual(["tool_result", "c1", true]);
  expect(result.content).toContain("outcome is unknown");
  checkConversation(history);
  expect(messagesSeqs(log)).toHaveLength(1);
});

// --- a LOOP history --- //

const HISTORY = `
id: talk
name: talk
defs:
  turn:
    input:
      history: !list(Message)
      n: !int
    nodes:
      reply:
        kind: agent
        input:
          history: \${input.history}
          n: \${input.n}
        prompt:
          - \${history}
          - user: "turn \${n}"
        output: str
    output:
      history: \${input.history + reply.new_messages}
      n: \${input.n + 1}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      history: []
      n: 0
    times: TIMES
output: \${chat.output.history}
`;

test("a LOOP accumulates each turn's new_messages; each request carries the history so far", async () => {
  const model = new ScriptedModel([textReply("one"), textReply("two"), textReply("three")], CONFIG);
  const { result } = await run(model, HISTORY.replace("TIMES", "3"));
  const history = result.output as MessageValue[];
  // The prompt's user item is the agent's input, not a message it added.
  expect(history.map((message) => message.role)).toEqual(["assistant", "assistant", "assistant"]);
  expect(history.map((message) => (message.parts[0] as { text: string }).text)).toEqual(["one", "two", "three"]);
  const third = model.requests[2]!.messages.filter((message) => message.role !== "system").map((message) => [
    message.role,
    message.parts.map((part) => (part instanceof TextPart ? part.text : part.type)).join(""),
  ]);
  expect(third).toEqual([
    ["assistant", "one"],
    ["assistant", "two"],
    ["user", "turn 2"],
  ]);
});

test("a LOOP history keeps a bounded number of new_messages in the pool", async () => {
  for (const learnable of [false, true]) {
    const model = new ScriptedModel(Array.from({ length: 8 }, (_, index) => textReply(`turn ${index}`)), CONFIG);
    const { log, flow, result } = await run(model, HISTORY.replace("TIMES", "8"), { learnable });
    expect(result.output as MessageValue[]).toHaveLength(8);
    const held: number[] = [];
    replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(log)), {
      onPrepared: (_prepared, at) => {
        held.push([...at.pool.store.keys()].filter((key) => key.endsWith(".new_messages")).length);
      },
    });
    expect(Math.max(...held)).toBeLessThanOrEqual(2);
  }
});

test("a reader of x.new_messages is a referrer of x", async () => {
  const text = `
id: f
name: f
nodes:
  reply:
    kind: agent
    prompt: hi
    output: str
  reader:
    kind: code
    input:
      history: \${reply.new_messages}
    code: return inputs['history'].length
    output: int
output: \${reader.output}
`;
  const { log, flow } = await run(new ScriptedModel([textReply("hi")], CONFIG), text);
  const state = replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(log)));
  expect(growthIndex(state).referrers.get("reply")).toContain("reader");
});

// --- skips --- //

test("a skipped agent's read takes its default", async () => {
  const { result } = await run(
    new ScriptedModel([], CONFIG),
    `
id: f
name: f
input:
  go: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: "\${input.go}"
        then: reply
    else: other
  reply:
    kind: agent
    prompt: hi
    output: str
  other:
    kind: code
    code: return 1
    output: int
output: \${reply.new_messages :- []}
`,
    { inputs: { go: false } },
  );
  expect(result.output).toEqual([]);
});

test("a structured agent's new_messages read by END ends with its value as JSON", async () => {
  const reply = new Response({
    message: new Message({ role: "assistant", parts: [new TextPart('{"answer": "yes"}')] }),
    finishReason: "stop",
  });
  const { result } = await run(
    new ScriptedModel([reply], CONFIG),
    `
id: f
name: f
nodes:
  reply:
    kind: agent
    prompt: hi
    output: {answer: str}
output: \${reply.new_messages}
`,
  );
  const history = result.output as MessageValue[];
  expect(history.at(-1)!.role).toBe("assistant");
  expect(JSON.parse((history.at(-1)!.parts[0] as { text: string }).text)).toEqual({ answer: "yes" });
});

// --- flow asserts --- //

/** A flow whose tool-less agent `reply` is checked by the flow assert `check`. */
const ASSERTED = (check: string) => `
id: f
name: f
nodes:
  reply:
    kind: agent
    prompt: hi
    output: str
output: \${reply.output}
asserts:
  - ${check}
`;

test("a flow assert reads reply.new_messages: it passes, or fails the run at END", async () => {
  await run(new ScriptedModel([textReply("hi")], CONFIG), ASSERTED("reply.new_messages != []"));
  const flow = await loadFlow(ASSERTED("reply.new_messages == []"));
  const result = await runFlow(flow, {}, {
    llmConfig: CONFIG,
    env: (name) => (name === "YIAN_MODEL_RETRIES" ? "0" : NO_ENV(name)),
    scheduler: virtualScheduler(START),
    models: serving(new ScriptedModel([textReply("hi")], CONFIG)),
    tools: TOOLS,
  });
  expect([result.status, result.failure?.code, result.failure?.nodeId]).toEqual(["failed", "assertion.failed", "__end__"]);
});

// --- learning --- //

test("workFeedback keeps text and flattens a selection of new messages to labelled lines", () => {
  expect(workFeedback("too long")).toBe("too long");
  expect(workFeedback(null)).toBeNull();
  expect(workFeedback([null, { parts: [{ text: "wrong tool" }] }])).toBe("new_messages[1].parts[0].text: wrong tool");
  expect(workFeedback([{ role: "a", parts: "b" }])).toBe("new_messages[0].parts: b\nnew_messages[0].role: a");
  // A selection with no text leaves carries no feedback.
  expect(workFeedback([null, null])).toBeNull();
  expect(workFeedback([{ parts: [null] }])).toBeNull();
});

test("takeOccurrenceMessages adds the messages sent to an AGENT's new messages, as work feedback, in order", async () => {
  const { log, flow } = await run(new ScriptedModel([textReply("hi")], CONFIG), TOP.replace("tools: [lookup]", "tools: []"), {
    learnable: true,
  });
  const state = replayRunLog(flow, MemoryRunLog.fromRecords(readRunLog(log)));
  const occurrence = state.trace!.occurrences.find((each) => each.nodeId === "reply")!;
  const end = state.trace!.occurrences.find((each) => each.nodeId === "__end__")!;
  const attribution = (index: number) => new BackwardAttribution(end.source, "history", index, 0, [0]);
  const pending: PendingMessages = new Map();
  addMessage(pending, occurrence.source, new BackwardMessage("short", attribution(0), [0, 1]));
  addMessage(pending, occurrence.messagesSource!, new BackwardMessage([{ parts: [{ text: "curt" }] }], attribution(1), [0, 0]));
  const taken = takeOccurrenceMessages(pending, occurrence);
  expect(taken.map((message) => message.feedback)).toEqual(["new_messages[0].parts[0].text: curt", "short"]);
  expect(taken.map((message) => message.attribution.messageIndex)).toEqual([1, 0]);
  expect(pending.size).toBe(0);
  // A node without new messages takes only its own.
  expect(end.messagesSource).toBeNull();
  addMessage(pending, end.source, new BackwardMessage("x", attribution(2), [1]));
  expect(takeOccurrenceMessages(pending, end).map((message) => message.feedback)).toEqual(["x"]);
});

/** The reply of an AGENT backward call: no feedback for inputs or variables, `self` for itself. */
const backwardReply = (self: string[] = []) =>
  textReply(JSON.stringify({ input_feedback: {}, self_feedback: self, variable_feedback: {} }));

/** Run `text` learnably with a tool-less agent replying "hi", then its backward over `feedback`. */
async function learn(text: string, feedback: unknown) {
  const model = new ScriptedModel([textReply("hi"), backwardReply(["ok"])], CONFIG);
  const { result, log } = await run(model, text.replace("tools: [lookup]", "tools: []"), { learnable: true });
  const outcome = await result.backward(feedback as never);
  expect(outcome.status, JSON.stringify(outcome.failure)).toBe("succeeded");
  expect(model.requests).toHaveLength(2);
  const prompt = model.requests[1]!.messages.map((message) => message.text()).join("\n");
  const payload = JSON.parse(prompt.slice(prompt.indexOf('{"forward"'))) as { forward: { node_id: string }; feedback: unknown[] };
  expect(payload.forward.node_id).toBe("reply");
  return { received: payload.feedback, outcome, log };
}

const MESSAGES_ONLY = `
id: f
name: f
nodes:
  reply:
    kind: agent
    tools: [lookup]
    prompt: hi
    output: str
output: \${reply.new_messages}
`;

test("feedback on a flow output bound to reply.new_messages reaches the agent's backward as text", async () => {
  const { received } = await learn(MESSAGES_ONLY, [{ parts: [{ text: "too curt" }] }]);
  expect(received).toEqual(["new_messages[0].parts[0].text: too curt"]);
});

test("text feedback on it reaches the backward unchanged", async () => {
  const { received } = await learn(MESSAGES_ONLY, "too curt");
  expect(received).toEqual(["too curt"]);
});

test("feedback on both .output and .new_messages reaches one backward call with both messages in order", async () => {
  // END sends its inputs' feedback in input order: `history`, then `answer`.
  const { received } = await learn(TOP, { answer: "wrong", history: "rambling" });
  expect(received).toEqual(["rambling", "wrong"]);
});

test("a reopened run with feedback on new_messages replays its backward, and learns again identically", async () => {
  const feedback = [{ parts: [{ text: "too curt" }] }];
  const { log, outcome } = await learn(MESSAGES_ONLY, feedback);
  const model = new ScriptedModel([backwardReply(["ok"])], CONFIG);
  const reopened = await advance({
    log,
    tools: TOOLS,
    env: (name) => (name === "YIAN_MODEL_RETRIES" ? "0" : NO_ENV(name)),
    scheduler: virtualScheduler(START + 60_000),
    models: serving(model),
  });
  // Replay validates the recorded attempt's transfers, the routed work feedback included.
  expect(reopened.backwardOutcome).toEqual(outcome);
  const again = await reopened.backward(feedback as never);
  expect(again.status, JSON.stringify(again.failure)).toBe("succeeded");
  const prompt = model.requests[0]!.messages.map((message) => message.text()).join("\n");
  expect(JSON.parse(prompt.slice(prompt.indexOf('{"forward"'))).feedback).toEqual(["new_messages[0].parts[0].text: too curt"]);
});
