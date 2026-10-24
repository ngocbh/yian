/**
 * TS-only: `examples/chat.yaml` driven through the host calls across every Phase 2 feature: send,
 * reply, a steer mid-reply, an interrupt, a pending call completed, a crash and `advance`, `close`.
 * The output equals the expected history.
 */

import { afterAll, expect, test } from "bun:test";
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { NodeSucceeded, type RunEvent } from "../../src/events.ts";
import { type ChatModel, EventStream, Finish, TextDelta } from "../../src/llm/events.ts";
import type { Request } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/memory.ts";
import { INTERRUPTED_MARKER } from "../../src/nodes/agent/messages.ts";
import { advance, close, complete, foldRunLog, interrupt, pendingCalls, runFlow, send } from "../../src/run.ts";
import {
  InboxClosedRecord,
  InterruptedRecord,
  NodeStepRecord,
  PoolCommitRecord,
  readRunLog,
  RunResumedRecord,
} from "../../src/runtime/log.ts";
import { DeliveryRefused } from "../../src/runtime/state.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { callReply, crashAfter, NO_ENV, serving, textReply, userText as user } from "./run_state_fixtures.ts";

const EXAMPLES_DIR = fileURLToPath(new URL("../../../examples", import.meta.url));
const CONFIG = { provider: "anthropic", model: "claude-test" };
const tmp = mkdtempSync(join(tmpdir(), "yian-chat-e2e-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

/** The example with host tools granted to `reply` (the shipped example grants none). */
async function chatFlow() {
  const steer = "        steer_inbox: ${input.owner_messages}\n";
  const text = readFileSync(join(EXAMPLES_DIR, "chat.yaml"), "utf8");
  expect(text).toContain(steer);
  return loadFlow(text.replace(steer, `${steer}        tools: [lookup, ask_owner]\n`), { searchPaths: [EXAMPLES_DIR] });
}

const tools = {
  lookup: new FunctionTool({
    name: "lookup",
    description: "Look a topic up.",
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => `about ${String(args["topic"])}`,
  }),
  ask_owner: new FunctionTool({
    name: "ask_owner",
    description: "Ask the owner; they answer later.",
    parametersJsonSchema: { type: "object", properties: { question: { type: "string" } }, required: ["question"] },
    execute: () => pending(),
  }),
};

/**
 * `model`, running `during(n, k)` in its nth request (from 1) before its stream starts (`k` 0) and
 * after each of its events (`k` from 1), awaiting what it returns before the stream goes on.
 */
function hooked(model: ScriptedModel, during: (request: number, events: number) => unknown): ChatModel {
  return {
    provider: model.provider,
    model: model.model,
    stream(request: Request, options?: { signal?: AbortSignal }): EventStream {
      const inner = model.stream(request, options);
      const n = model.requests.length;
      return new EventStream(
        (async function* () {
          let k = 0;
          await during(n, k);
          for await (const event of inner) {
            yield event;
            await during(n, ++k);
          }
        })(),
        () => inner.close(),
      );
    },
  };
}

const assistant = (text: string) => ({ role: "assistant", parts: [{ type: "text", text }] });
const calls = (...made: [string, string, object][]) => ({
  role: "assistant",
  parts: made.map(([id, name, args]) => ({ type: "tool_call", id, name, arguments: args })),
});
const tool = (call_id: string, name: string, content: string, is_error?: true) => ({
  role: "tool",
  parts: [{ type: "tool_result", call_id, name, content, ...(is_error ? { is_error } : {}) }],
});

/** The flow output's history. */
const history = (output: unknown) => (output as { history: unknown[] }).history;

/** The records of the log at `log`. */
const records = (log: string) => readRunLog(log);

/** A short label of each record of the log at `log` that the script's order is read from. */
function milestones(log: string): string[] {
  return records(log).flatMap(({ record }) => {
    if (record instanceof NodeStepRecord && ["steer", "pending"].includes(String(record.step["kind"]))) {
      return [`${String(record.step["kind"])} ${record.nodeId}`];
    }
    if (record instanceof InterruptedRecord) return [`interrupted ${record.nodes.join(", ")}`];
    if (record instanceof RunResumedRecord) return ["run_resumed"];
    if (record instanceof InboxClosedRecord) return ["inbox_closed"];
    return [];
  });
}

test("the Phase 2 chat end to end: steer, interrupt, pending call, crash and advance, close", async () => {
  const flow = await chatFlow();
  const log = join(tmp, "chat.jsonl");
  const env = { env: NO_ENV };
  const script = new ScriptedModel(
    [
      textReply("hello"),
      callReply(["lookup", { topic: "x" }, "c1"]),
      textReply("x, briefly"),
      [new TextDelta(0, "Once upon"), new TextDelta(0, " a time"), new TextDelta(0, " there was"), new Finish({ reason: "stop" })],
      callReply(["ask_owner", { question: "ok?" }, "c2"]),
      textReply("the owner said yes"),
    ],
    CONFIG,
  );
  // The owner steers during the lookup call's reply (request 2): the steer lands live and the
  // agent reads it once the lookup result is in.
  // The owner steers during the lookup call's reply (request 2), and interrupts the story (request
  // 4) once its second chunk is shown, then sends the next message at once.
  const model = hooked(script, async (n, k) => {
    if (n === 2 && k === 0) await send(log, "owner_messages", user("briefly"), env);
    if (n === 4 && k === 2) {
      await interrupt(log, "owner_messages", env);
      await send(log, "owner_messages", user("ask the owner"), env);
    }
  });
  const cut: string[] = [];
  const onEvent = (event: RunEvent) => void (event instanceof NodeSucceeded && event.interrupted && cut.push(event.nodeId));
  const options = { ...env, numWorkers: 0, tools, models: serving(model), onEvent };

  // 1. Start: the first take waits.
  expect((await runFlow(flow, {}, { ...options, log, llmConfig: CONFIG })).status).toBe("waiting");
  // 2. Turn 1: a message and its reply.
  await send(log, "owner_messages", user("hi"), env);
  expect((await advance({ ...options, log })).status).toBe("waiting");
  // 3. Turn 2: a steer mid-reply joins the request after the lookup result.
  await send(log, "owner_messages", user("tell me about x"), env);
  expect((await advance({ ...options, log })).status).toBe("waiting");
  // 4. Turn 3: the host interrupts the streaming reply, which keeps both chunks shown; its output
  //    is null and the turn ends. The message sent after the interrupt isn't read as a steer: it
  //    stays queued, so turn 4 takes it at once.
  // 5. Turn 4: the reply's call to the owner pends; the host completes it.
  await send(log, "owner_messages", user("a story please"), env);
  expect((await advance({ ...options, log })).status).toBe("waiting");
  expect(cut).toEqual(["chat#2/answered/reply"]);
  const open = pendingCalls(log);
  expect(open.map((call) => [call.node, call.tool, call.arguments])).toEqual([
    ["chat#3/answered/reply", "ask_owner", { question: "ok?" }],
  ]);
  await complete(log, open[0]!.callId, { output: "yes" }, env);
  // 6. The process is lost as the woken reply's final model call returns, before its record.
  //    A sibling run of the same log, with no crash, gives that record's seq.
  const sibling = join(tmp, "sibling.jsonl");
  copyFileSync(log, sibling);
  const copied = records(log).at(-1)!.seq;
  const siblingModel = serving(new ScriptedModel([textReply("the owner said yes")], CONFIG));
  expect((await advance({ ...options, log: sibling, models: siblingModel })).status).toBe("waiting");
  const k = records(sibling).find(
    ({ seq, record }) =>
      seq > copied &&
      record instanceof NodeStepRecord &&
      record.nodeId === "chat#3/answered/reply" &&
      record.step["kind"] === "model_call",
  )!.seq;
  await crashAfter(k, () => advance({ ...options, log }));
  // The crashed attempt used up the last reply; the advance after the crash asks for it again, once.
  const again = new ScriptedModel([textReply("the owner said yes")], CONFIG);
  expect((await advance({ ...options, log, models: serving(again) })).status).toBe("waiting");
  expect(again.requests).toHaveLength(1);
  // 7. Close: the last take gets nothing and the loop ends.
  await close(log, "owner_messages", env);
  const done = await advance({ ...options, log });
  expect(done.status, done.failure?.message).toBe("succeeded");

  expect(history(done.output)).toEqual([
    user("hi"),
    assistant("hello"),
    user("tell me about x"),
    calls(["c1", "lookup", { topic: "x" }]),
    tool("c1", "lookup", "about x"),
    user("briefly"),
    assistant("x, briefly"),
    user("a story please"),
    assistant("Once upon a time"),
    user(INTERRUPTED_MARKER),
    user("ask the owner"),
    calls(["c2", "ask_owner", { question: "ok?" }]),
    tool("c2", "ask_owner", "yes"),
    assistant("the owner said yes"),
  ]);
  expect(milestones(log)).toEqual([
    "steer chat#1/answered/reply",
    "interrupted chat#2/answered/reply",
    "pending chat#3/answered/reply",
    "run_resumed",
    "inbox_closed",
  ]);
  // The call was answered before the crash, so the resume closed nothing.
  const resumed = records(log).find(({ record }) => record instanceof RunResumedRecord)!.record as RunResumedRecord;
  expect(resumed.closedCalls).toEqual([]);
  // Each turn's END commit points at its parts (the carried history, the take, the messages
  // commit) instead of copying the history.
  const ends = records(log).flatMap(({ record }) =>
    record instanceof PoolCommitRecord && /^chat#\d+\/__end__$/.test(record.nodeId) ? [record] : [],
  );
  expect(ends).toHaveLength(5);
  for (const end of ends) expect([end.value, end.from === null]).toEqual([null, false]);
  // The replies' messages commits hold their new messages: the steer, and the cut text then the
  // interrupt marker.
  const folded = await foldRunLog(log, env);
  const newMessages = (node: string) =>
    folded.records
      .find(
        ({ entry }) =>
          entry.envelope.record instanceof PoolCommitRecord && entry.envelope.record.key === `${node}.new_messages`,
      )!
      .value!.toObject() as unknown;
  expect(newMessages("chat#1/answered/reply")).toEqual([
    calls(["c1", "lookup", { topic: "x" }]),
    tool("c1", "lookup", "about x"),
    user("briefly"),
    assistant("x, briefly"),
  ]);
  expect(newMessages("chat#2/answered/reply")).toEqual([assistant("Once upon a time"), user(INTERRUPTED_MARKER)]);
  expect(pendingCalls(log)).toEqual([]);
}, 60_000);

test("a crash with the owner's call open closes it: the reply reads its outcome is unknown", async () => {
  const flow = await chatFlow();
  const log = join(tmp, "chat-open.jsonl");
  const env = { env: NO_ENV };
  const script = new ScriptedModel([textReply("hello"), callReply(["ask_owner", { question: "ok?" }, "c2"])], CONFIG);
  const options = { ...env, numWorkers: 0, tools, models: serving(script) };
  expect((await runFlow(flow, {}, { ...options, log, llmConfig: CONFIG })).status).toBe("waiting");
  await send(log, "owner_messages", user("hi"), env);
  expect((await advance({ ...options, log })).status).toBe("waiting");
  await send(log, "owner_messages", user("ask the owner"), env);
  // The process is lost right after the call's pending step, before the node waits.
  const sibling = join(tmp, "sibling-open.jsonl");
  copyFileSync(log, sibling);
  const siblingModel = serving(new ScriptedModel([callReply(["ask_owner", { question: "ok?" }, "c2"])], CONFIG));
  expect((await advance({ ...options, log: sibling, models: siblingModel })).status).toBe("waiting");
  const pendingSeq = records(sibling).find(
    ({ record }) => record instanceof NodeStepRecord && record.step["kind"] === "pending",
  )!.seq;
  await crashAfter(pendingSeq + 1, () => advance({ ...options, log }));
  const after = new ScriptedModel([textReply("I couldn't reach them")], CONFIG);
  expect((await advance({ ...options, log, models: serving(after) })).status).toBe("waiting");
  const resumed = records(log).find(({ record }) => record instanceof RunResumedRecord)!.record as RunResumedRecord;
  expect(resumed.closedCalls).toHaveLength(1);
  expect(resumed.closedCalls[0]).toEndWith("/chat#1/answered/reply/1/0");
  const refused = await complete(log, resumed.closedCalls[0]!, { output: "yes" }, env).catch((error: unknown) => error);
  expect(refused).toBeInstanceOf(DeliveryRefused);
  // The closed call's reply has settled since, so the call is gone.
  expect((refused as DeliveryRefused).reason).toBe("unknown_call");
  await close(log, "owner_messages", env);
  const done = await advance({ ...options, log });
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(history(done.output)).toEqual([
    user("hi"),
    assistant("hello"),
    user("ask the owner"),
    calls(["c2", "ask_owner", { question: "ok?" }]),
    tool("c2", "ask_owner", TOOL_OUTCOME_UNKNOWN, true),
    assistant("I couldn't reach them"),
  ]);
}, 60_000);
