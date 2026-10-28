/**
 * The host call `interrupt(log, inbox)` on a run log, live through this process's engine or
 * offline under the log's lock, and the agents it cuts.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { ModelCallFinished, NodeSucceeded, NodeWaiting, type RunEvent, type StreamChunk } from "../../src/events.ts";
import { Finish, TextDelta } from "../../src/llm/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { FileLockError, lockFile, virtualScheduler } from "../../src/platform.ts";
import {
  advance,
  close,
  complete,
  DeliveryRefused,
  interrupt,
  pendingCalls,
  replayRunLog,
  RunBusyError,
  runFlow,
  send,
} from "../../src/run.ts";
import { decodeRecord, InterruptedRecord, JsonlRunLog, NodeStepRecord, readRunLog } from "../../src/runtime/log.ts";
import { applyWaitsRecord, WaitsState } from "../../src/runtime/waits.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { callReply, crashAfter, rejected, serving, textReply, userText } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-interrupt-host-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

const CONFIG = { provider: "openai", model: "test-model" };
const env = hostEnv(dir);

/** One AGENT `reply` reading steer inbox `m`, granted `hold` and `lookup`, with output `output` (none when `null`). */
const replyFlow = (output: string | null = "Optional(str)", steerInbox = true) => `
id: interrupt_host
name: interrupt_host
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
nodes:
  reply:
    kind: agent
    prompt: go
    tools: [hold, lookup]
${steerInbox ? "    steer_inbox: \${input.m}\n" : ""}${output === null ? "" : `    output: ${output}\n`}${steerInbox ? "" : HEARD}output: \${reply.output}
`;

/** A take from `m`, so a flow whose agent reads no steer inbox still reads its inbox. */
const HEARD = `  heard:
    kind: tool
    tool_id: take
    input:
      from: \${input.m}
      take: all
`;

/** `hold` pends; `lookup` answers "found". */
const tools = () => ({
  hold: tool({ name: "hold", description: "Hold.", parameters: { type: "object", properties: {} }, execute: () => pending() }),
  lookup: tool({ name: "lookup", description: "Look up.", parameters: { type: "object", properties: {} }, execute: () => "found" }),
});

/** The kinds of the log's records, a step as its own `kind`. */
const kinds = (log: string) =>
  readFileSync(log, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => decodeRecord(line))
    .map(({ kind, record }) => (record instanceof NodeStepRecord ? String(record.step["kind"]) : kind));

/** The steps of node `node` in the log, in order. */
const stepsOf = (log: string, node: string) =>
  readRunLog(log).flatMap(({ record }) => (record instanceof NodeStepRecord && record.nodeId === node ? [record.step] : []));

/** A waiting run whose agent pends `hold`. */
async function waitingReply(model: ScriptedModel, output: string | null = "Optional(str)") {
  const log = freshLog();
  const flow = await loadFlow(replyFlow(output));
  const options = { env, log, tools: tools(), models: serving(model), runId: "r" };
  const result = await runFlow(flow, {}, options);
  expect(result.status, result.failure?.message).toBe("waiting");
  return { log, flow, options };
}

test("an idle interrupt of a waiting agent closes its call, returns it, and refuses a later complete", async () => {
  const model = new ScriptedModel([callReply(["hold", {}, "c1"])], CONFIG);
  const { log } = await waitingReply(model);
  const [call] = pendingCalls(log);
  const { seq, nodes, calls } = await interrupt(log, "m", { env });
  expect(seq).toBe(readRunLog(log).at(-1)!.seq);
  expect(nodes).toEqual(["reply"]);
  expect(calls).toEqual([call!]);
  expect(pendingCalls(log)).toEqual([]);
  const record = readRunLog(log).at(-1)!.record as InterruptedRecord;
  expect([record.inbox, [...record.nodes]]).toEqual(["m", ["reply"]]);
  const refused = (await rejected(complete(log, call!.callId, { output: "late" }, { env }))) as DeliveryRefused;
  expect(refused).toBeInstanceOf(DeliveryRefused);
  expect([refused.reason, refused.callState]).toEqual(["call_settled", "closed"]);
});

test("interrupt while idle at the take appends nothing", async () => {
  const log = freshLog();
  const flow = await loadFlow(replyFlow("Optional(str)", false));
  const model = new ScriptedModel([callReply(["hold", {}, "c1"])], CONFIG);
  await runFlow(flow, {}, { env, log, tools: tools(), models: serving(model), runId: "r" });
  const before = readFileSync(log);
  expect(await interrupt(log, "m", { env })).toEqual({ seq: null, nodes: [], calls: [] });
  expect(readFileSync(log).equals(before)).toBe(true);
});

test("interrupt refusals: unknown inbox; RunBusyError under another writer", async () => {
  const model = new ScriptedModel([callReply(["hold", {}, "c1"])], CONFIG);
  const { log } = await waitingReply(model);
  const unknown = (await rejected(interrupt(log, "nope", { env }))) as DeliveryRefused;
  expect([unknown.reason, unknown.message]).toEqual(["unknown_inbox", "the run has no inbox 'nope'"]);
  expect(await rejected(interrupt(log, 1 as never, { env }))).toBeInstanceOf(TypeError);
  const holder = lockFile(`${realpathSync(log)}.lock`);
  try {
    const busy = (await rejected(interrupt(log, "m", { env }))) as RunBusyError;
    expect([busy instanceof RunBusyError, busy.cause instanceof FileLockError]).toEqual([true, true]);
  } finally {
    holder.release();
  }
});

test("an interrupt of a closed inbox still cuts a waiting reader", async () => {
  const model = new ScriptedModel([callReply(["hold", {}, "c1"])], CONFIG);
  const { log } = await waitingReply(model);
  await close(log, "m", { env });
  expect((await interrupt(log, "m", { env })).calls.length).toBe(1);
  expect(kinds(log).at(-1)).toBe("interrupted");
});

test("a crashed run: an offline interrupt names only waiting readers", async () => {
  // The agent was mid-call when the run crashed: it is in flight, not waiting, so offline the
  // record names no node; it only records the Stop.
  const flow = await loadFlow(replyFlow());
  const script = () => new ScriptedModel([callReply(["lookup", {}, "c1"]), textReply("done")], CONFIG);
  const whole = freshLog();
  await runFlow(flow, {}, { env, log: whole, tools: tools(), models: serving(script()), runId: "r" });
  const cut = kinds(whole).indexOf("model_call") + 1;
  const log = freshLog();
  await crashAfter(cut, () => runFlow(flow, {}, { env, log, tools: tools(), models: serving(script()), runId: "r" }));
  const before = readRunLog(log).length;
  expect(await interrupt(log, "m", { env })).toEqual({ seq: before, nodes: [], calls: [] });
  const appended = readRunLog(log).slice(before);
  expect(appended.map(({ record }) => record)).toEqual([new InterruptedRecord("m", [])]);
});

test("advance after an idle interrupt: the wake, the interrupted closer, the honor step; output null", async () => {
  const model = new ScriptedModel([callReply(["hold", {}, "c1"], ["lookup", {}, "c2"])], CONFIG);
  const { log, options } = await waitingReply(model);
  await interrupt(log, "m", { env });
  const done = await advance(options);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toBeNull();
  expect(kinds(log).slice(-9, -5)).toEqual(["node_woken", "tool_result", "interrupted", "pool_commit"]);
  const steps = stepsOf(log, "reply");
  expect(steps.at(-2)).toMatchObject({ call_id: "c1", closed: "interrupted", is_error: true });
  const record = readRunLog(log).find(({ record }) => record instanceof InterruptedRecord)!;
  expect(steps.at(-1)).toEqual({ kind: "interrupted", by: BigInt(record.seq) });
  // The model was asked once: the woken agent honored the interrupt instead of calling it again.
  expect(model.requests).toHaveLength(1);
});

test("an idle interrupt of a str agent fails it agent.interrupted on advance", async () => {
  const model = new ScriptedModel([callReply(["hold", {}, "c1"])], CONFIG);
  const { log, options } = await waitingReply(model, "str");
  await interrupt(log, "m", { env });
  const done = await advance(options);
  expect(done.status).toBe("failed");
  expect(done.failure?.code).toBe("agent.interrupted");
});

test("a live interrupt from onStream cuts the reply at the next chunk", async () => {
  const log = freshLog();
  const flow = await loadFlow(replyFlow());
  const model = new ScriptedModel([[new TextDelta(0, "hello"), new TextDelta(0, " big"), new TextDelta(0, " world"), new Finish({ reason: "stop" })]], CONFIG);
  const onStream = (chunk: StreamChunk) => void (chunk.chunk === " big" && interrupt(log, "m", { env }));
  const result = await runFlow(flow, {}, { env, log, tools: tools(), models: serving(model), runId: "r", onStream, numWorkers: 0 });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBeNull();
  expect(stepsOf(log, "reply").map((step) => [step["kind"], step["finish"] ?? null])).toEqual([
    ["model_call", "interrupted"],
    ["interrupted", null],
  ]);
});

/** The `interrupted` flag of each `NodeSucceeded` the events hold, by node. */
const succeeded = (events: readonly RunEvent[]) =>
  Object.fromEntries(events.flatMap((event) => (event instanceof NodeSucceeded ? [[event.nodeId, event.interrupted]] : [])));

test("NodeSucceeded says interrupted: true for an interrupted Optional agent, live and on a resume past its honor step", async () => {
  const flow = await loadFlow(replyFlow());
  const script = () =>
    new ScriptedModel([[new TextDelta(0, "hello"), new TextDelta(0, " big"), new TextDelta(0, " world"), new Finish({ reason: "stop" })]], CONFIG);
  const whole = freshLog();
  const events: RunEvent[] = [];
  const cutAt = (log: string) => (chunk: StreamChunk) => void (chunk.chunk === " big" && interrupt(log, "m", { env }));
  const result = await runFlow(flow, {}, {
    env, log: whole, tools: tools(), models: serving(script()), runId: "r", numWorkers: 0, onStream: cutAt(whole), onEvent: (event) => void events.push(event),
  });
  expect(result.output).toBeNull();
  expect(succeeded(events)).toEqual({ __start__: false, reply: true, __end__: false });
  expect(events.find((event) => event instanceof NodeSucceeded && event.nodeId === "reply")!.toJSON()).toMatchObject({ interrupted: true });

  // Cut right after the honor step: the resumed agent replays to it and says the same.
  const log = freshLog();
  const honor = kinds(whole).indexOf("interrupted", kinds(whole).indexOf("model_call"));
  await crashAfter(honor, () =>
    runFlow(flow, {}, { env, log, tools: tools(), models: serving(script()), runId: "r", numWorkers: 0, onStream: cutAt(log) }),
  );
  const resumed: RunEvent[] = [];
  const done = await advance({ env, log, tools: tools(), models: serving(script()), onEvent: (event) => void resumed.push(event) });
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(succeeded(resumed)).toEqual({ reply: true, __end__: false });
});

test("an interrupt that lands after the final reply: the agent answers and NodeSucceeded says interrupted: false", async () => {
  const log = freshLog();
  const flow = await loadFlow(replyFlow());
  const model = new ScriptedModel([[new TextDelta(0, "hello"), new TextDelta(0, " world"), new Finish({ reason: "stop" })]], CONFIG);
  const events: RunEvent[] = [];
  const onEvent = (event: RunEvent) => {
    events.push(event);
    if (event instanceof ModelCallFinished) void interrupt(log, "m", { env });
  };
  const result = await runFlow(flow, {}, { env, log, tools: tools(), models: serving(model), runId: "r", numWorkers: 0, onEvent });
  expect(result.output).toBe("hello world");
  // The interrupt landed and named the agent, but the reply was whole: nothing was cut.
  expect(readRunLog(log).some(({ record }) => record instanceof InterruptedRecord)).toBe(true);
  expect(succeeded(events)["reply"]).toBe(false);
});

// --- The remembered Stop through the host calls ----------------------------------------------- //

/** The `nodes` of each `interrupted` record at `log`, in log order. */
const stopsAt = (log: string) => readRunLog(log).flatMap(({ record }) => (record instanceof InterruptedRecord ? [[...record.nodes]] : []));

/** The waits the records at `log` fold to. */
function foldedWaits(log: string): WaitsState {
  const w = new WaitsState();
  for (const envelope of readRunLog(log)) applyWaitsRecord(w, envelope);
  return w;
}

/** The step kinds node `node` recorded at `log`. */
const stepKinds = (log: string, node: string) => stepsOf(log, node).map((step) => String(step["kind"]));

/** `heard` takes one message from `m`; `gate`, a node beside it, holds the reply back; then `reply` reads steer inbox `m`. */
const gateFlow = (gate: string) => `
id: stop_gate
name: stop_gate
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
nodes:
  heard:
    kind: tool
    tool_id: take
    input:
      from: \${input.m}
      take: one
  gate:
${gate}
  reply:
    kind: agent
    prompt: "answer \${got}"
    steer_inbox: \${input.m}
    input:
      got: \${heard.output.events}
      gated: \${gate.output}
    output: Optional(str)
output: \${reply.output}
`;

const ASK_GATE = `    kind: tool
    tool_id: ask
    input:
      prompt: approve?`;

const SLEEP_GATE = `    kind: tool
    tool_id: sleep
    input:
      for: 60s`;

/** `heard` takes one message from `m`, then `reply` reads steer inbox `m` and answers it. */
const TAKE_REPLY = gateFlow(`    kind: code
    code: return 1`);

/** A run of `text` waiting at its take, then `send` "hi" and an advance to the next wait. */
async function heardHi(text: string, model: ScriptedModel, extra: Record<string, unknown> = {}) {
  const log = freshLog();
  const flow = await loadFlow(text);
  const options = { env, log, models: serving(model), runId: "r", tools: { ask: () => pending() }, ...extra };
  expect((await runFlow(flow, {}, options)).status).toBe("waiting");
  await send(log, "m", "hi", { env });
  return { log, flow, options };
}

/** The reply was stopped by its own applied record: honored, no model call, null. */
function expectStoppedByApplied(log: string, output: unknown, model: ScriptedModel) {
  expect(output).toBeNull();
  expect(model.requests).toHaveLength(0);
  expect(stopsAt(log).at(-1)).toEqual(["reply"]);
  expect(stepKinds(log, "reply")).toEqual(["interrupted"]);
}

test("idle at the take: interrupt records nothing", async () => {
  const log = freshLog();
  const model = new ScriptedModel([], CONFIG);
  expect((await runFlow(await loadFlow(TAKE_REPLY), {}, { env, log, models: serving(model), runId: "r" })).status).toBe("waiting");
  const before = readFileSync(log);
  expect(await interrupt(log, "m", { env })).toEqual({ seq: null, nodes: [], calls: [] });
  expect(readFileSync(log).equals(before)).toBe(true);
});

test("a Stop during an approval ask stops the reply after it", async () => {
  const model = new ScriptedModel([], CONFIG);
  const { log, options } = await heardHi(gateFlow(ASK_GATE), model);
  expect((await advance(options)).status).toBe("waiting");
  const [ask] = pendingCalls(log);
  const stop = await interrupt(log, "m", { env });
  expect(stop).toEqual({ seq: readRunLog(log).at(-1)!.seq, nodes: [], calls: [] });
  await complete(log, ask!.callId, { output: "yes" }, { env });
  const done = await advance(options);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expectStoppedByApplied(log, done.output, model);
  expect(stopsAt(log)).toEqual([[], ["reply"]]);
});

test("a Stop during a sleep stops the reply when the sleep ends", async () => {
  const start = Date.parse("2026-01-01T00:00:00Z");
  const model = new ScriptedModel([], CONFIG);
  const { log, options } = await heardHi(gateFlow(SLEEP_GATE), model, { scheduler: virtualScheduler(start) });
  expect((await advance(options)).status).not.toBe("succeeded");
  expect((await interrupt(log, "m", { env })).seq).not.toBeNull();
  const done = await advance({ ...options, scheduler: virtualScheduler(start + 61_000) });
  expect(done.status, done.failure?.message).toBe("succeeded");
  expectStoppedByApplied(log, done.output, model);
});

test("a Stop survives a host stop and a re-wait before it stopped anyone", async () => {
  const model = new ScriptedModel([], CONFIG);
  const { log, options } = await heardHi(gateFlow(ASK_GATE), model);
  expect((await advance(options)).status).toBe("waiting");
  const [ask] = pendingCalls(log);
  await interrupt(log, "m", { env });
  const stopped = new AbortController();
  stopped.abort(new Error("host stop"));
  await rejected(advance({ ...options, signal: stopped.signal }));
  expect((await advance(options)).status).toBe("waiting");
  expect(foldedWaits(log).stops.has("m")).toBe(true);
  await complete(log, ask!.callId, { output: "yes" }, { env });
  const done = await advance(options);
  expectStoppedByApplied(log, done.output, model);
});

test("a Stop after the send, before the take wakes, stops the reply", async () => {
  const model = new ScriptedModel([], CONFIG);
  const { log, options } = await heardHi(TAKE_REPLY, model);
  expect(await interrupt(log, "m", { env })).toEqual({ seq: readRunLog(log).at(-1)!.seq, nodes: [], calls: [] });
  const done = await advance(options);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expectStoppedByApplied(log, done.output, model);
});

const EXAMPLES_DIR = fileURLToPath(new URL("../../examples", import.meta.url));
/** `examples/chat.yaml`. */
const chatExample = () => loadFlow(readFileSync(join(EXAMPLES_DIR, "chat.yaml"), "utf8"), { searchPaths: [EXAMPLES_DIR] });
const CHUNKS = () => [new TextDelta(0, "one"), new TextDelta(0, " two"), new TextDelta(0, " three"), new Finish({ reason: "stop" })];

test("a message sent after a cut reply is taken at the next turn and answered in full", async () => {
  const log = freshLog();
  const flow = await chatExample();
  const model = new ScriptedModel([CHUNKS(), textReply("the second answer")], CONFIG);
  const options = { env, log, models: serving(model), runId: "r", numWorkers: 0, llmConfig: CONFIG };
  expect((await runFlow(flow, {}, options)).status).toBe("waiting");
  await send(log, "owner_messages", userText("first"), { env });
  const onStream = (chunk: StreamChunk) => void (chunk.chunk === " two" && interrupt(log, "owner_messages", { env }));
  const onEvent = (event: RunEvent) =>
    void (event instanceof NodeSucceeded && event.nodeId === "chat#0/answered/reply" && send(log, "owner_messages", userText("second"), { env }));
  expect((await advance({ ...options, onStream, onEvent })).status).toBe("waiting");
  expect(stopsAt(log)).toEqual([["chat#0/answered/reply"]]);
  expect(model.requests).toHaveLength(2);
  expect(stepKinds(log, "chat#1/answered/reply")).toEqual(["model_call"]);
  expect(foldedWaits(log).stops.size).toBe(0);
});

test("Esc after the reply finished records an empty Stop; the next take ends it", async () => {
  const log = freshLog();
  const flow = await chatExample();
  const model = new ScriptedModel([textReply("answer")], CONFIG);
  const options = { env, log, models: serving(model), runId: "r", numWorkers: 0, llmConfig: CONFIG };
  expect((await runFlow(flow, {}, options)).status).toBe("waiting");
  await send(log, "owner_messages", userText("first"), { env });
  let stop: Promise<{ seq: number | null }> | null = null;
  const onEvent = (event: RunEvent) => {
    if (event instanceof NodeSucceeded && event.nodeId === "chat#0/answered/reply") stop = interrupt(log, "owner_messages", { env });
  };
  expect((await advance({ ...options, onEvent })).status).toBe("waiting");
  expect((await stop!).seq).not.toBeNull();
  expect(stopsAt(log)).toEqual([[]]);
  expect(stepKinds(log, "chat#0/answered/reply")).toEqual(["model_call"]);
  expect(foldedWaits(log).stops.size).toBe(0);
});

/** `heard`, then `reply1` (a reader), then `approve` (an ask), then `reply2` (a reader). */
const TWO_REPLIES = `
id: two_replies
name: two_replies
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
nodes:
  heard:
    kind: tool
    tool_id: take
    input:
      from: \${input.m}
      take: one
  reply1:
    kind: agent
    prompt: "answer \${got}"
    steer_inbox: \${input.m}
    input:
      got: \${heard.output.events}
    output: Optional(str)
  approve:
    kind: tool
    tool_id: ask
    input:
      prompt: 'approve? \${reply1.output :- "-"}'
  reply2:
    kind: agent
    prompt: "go on \${ok}"
    steer_inbox: \${input.m}
    input:
      ok: \${approve.output}
    output: Optional(str)
output: \${reply2.output}
`;

test("a Stop that stopped an agent ends at the next wait: the reply after the approval answers in full", async () => {
  const model = new ScriptedModel([CHUNKS(), textReply("second reply")], CONFIG);
  const { log, options } = await heardHi(TWO_REPLIES, model, { numWorkers: 0 });
  const onStream = (chunk: StreamChunk) => void (chunk.nodeId === "reply1" && chunk.chunk === "one" && interrupt(log, "m", { env }));
  expect((await advance({ ...options, onStream })).status).toBe("waiting");
  expect(stopsAt(log)).toEqual([["reply1"]]);
  expect(foldedWaits(log).stops.size).toBe(0);
  await complete(log, pendingCalls(log)[0]!.callId, { output: "yes" }, { env });
  const done = await advance(options);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toBe("second reply");
  expect(stopsAt(log)).toEqual([["reply1"]]);
  expect(stepKinds(log, "reply2")).toEqual(["model_call"]);
});

test("a repeated Esc during the reply records nothing and never carries over", async () => {
  const model = new ScriptedModel([CHUNKS(), textReply("second reply")], CONFIG);
  const { log, options } = await heardHi(TWO_REPLIES, model, { numWorkers: 0 });
  let second: unknown = null;
  const onStream = (chunk: StreamChunk) => {
    if (chunk.nodeId !== "reply1" || chunk.chunk !== "one") return;
    void (async () => {
      await interrupt(log, "m", { env });
      second = await interrupt(log, "m", { env });
    })();
  };
  expect((await advance({ ...options, onStream })).status).toBe("waiting");
  expect(second).toEqual({ seq: null, nodes: [], calls: [] });
  expect(stopsAt(log)).toEqual([["reply1"]]);
  await complete(log, pendingCalls(log)[0]!.callId, { output: "yes" }, { env });
  const done = await advance(options);
  expect(done.output).toBe("second reply");
  expect(stopsAt(log)).toEqual([["reply1"]]);
});

test("a repeated Esc during an approval ask records nothing; the reply after it gets one applied record", async () => {
  const model = new ScriptedModel([], CONFIG);
  const { log, options } = await heardHi(gateFlow(ASK_GATE), model);
  expect((await advance(options)).status).toBe("waiting");
  expect((await interrupt(log, "m", { env })).seq).not.toBeNull();
  const before = readFileSync(log);
  expect(await interrupt(log, "m", { env })).toEqual({ seq: null, nodes: [], calls: [] });
  expect(readFileSync(log).equals(before)).toBe(true);
  await complete(log, pendingCalls(log)[0]!.callId, { output: "yes" }, { env });
  const done = await advance(options);
  expectStoppedByApplied(log, done.output, model);
  expect(stopsAt(log)).toEqual([[], ["reply"]]);
});

test("a crashed reader is stopped after an offline interrupt: it replays its recorded call, then honors", async () => {
  const flow = await loadFlow(replyFlow());
  const script = () => new ScriptedModel([callReply(["lookup", {}, "c1"]), textReply("done")], CONFIG);
  const whole = freshLog();
  await runFlow(flow, {}, { env, log: whole, tools: tools(), models: serving(script()), runId: "r" });
  const log = freshLog();
  await crashAfter(kinds(whole).indexOf("model_call") + 1, () =>
    runFlow(flow, {}, { env, log, tools: tools(), models: serving(script()), runId: "r" }),
  );
  expect((await interrupt(log, "m", { env })).nodes).toEqual([]);
  const model = new ScriptedModel([], CONFIG);
  const done = await advance({ env, log, tools: tools(), models: serving(model) });
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toBeNull();
  expect(model.requests).toHaveLength(0);
  expect(stopsAt(log)).toEqual([[], ["reply"]]);
  expect(stepKinds(log, "reply").filter((kind) => kind === "model_call")).toHaveLength(1);
  expect(stepKinds(log, "reply").at(-1)).toBe("interrupted");
});

test("a crashed reader whose final answer was recorded keeps it", async () => {
  const flow = await loadFlow(replyFlow());
  const script = () => new ScriptedModel([textReply("done")], CONFIG);
  const whole = freshLog();
  await runFlow(flow, {}, { env, log: whole, tools: tools(), models: serving(script()), runId: "r" });
  const log = freshLog();
  await crashAfter(kinds(whole).indexOf("model_call") + 1, () =>
    runFlow(flow, {}, { env, log, tools: tools(), models: serving(script()), runId: "r" }),
  );
  await interrupt(log, "m", { env });
  const events: RunEvent[] = [];
  const model = new ScriptedModel([], CONFIG);
  const done = await advance({ env, log, tools: tools(), models: serving(model), onEvent: (event) => void events.push(event) });
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toBe("done");
  expect(model.requests).toHaveLength(0);
  const succeededReply = events.find((event) => event instanceof NodeSucceeded && event.nodeId === "reply") as NodeSucceeded;
  expect(succeededReply.interrupted).toBe(false);
});

test("a Stop recorded offline before the take wakes stops the reply across a crash at every boundary", async () => {
  const flow = await loadFlow(TAKE_REPLY);
  const start = async (log: string) => {
    expect((await runFlow(flow, {}, { env, log, models: serving(new ScriptedModel([], CONFIG)), runId: "r" })).status).toBe("waiting");
    await send(log, "m", "hi", { env });
    return (await interrupt(log, "m", { env })).seq!;
  };
  const whole = freshLog();
  const r = await start(whole);
  expect((await advance({ env, log: whole, models: serving(new ScriptedModel([], CONFIG)) })).status).toBe("succeeded");
  const n = readRunLog(whole).length;
  for (let k = r + 1; k < n; k++) {
    const log = freshLog();
    await start(log);
    await crashAfter(k, () => advance({ env, log, models: serving(new ScriptedModel([], CONFIG)) }));
    const model = new ScriptedModel([], CONFIG);
    const done = await advance({ env, log, models: serving(model) });
    expect(done.status, `cut at ${k}: ${done.failure?.message}`).toBe("succeeded");
    expect(done.output).toBeNull();
    expect(model.requests).toHaveLength(0);
    expect(stopsAt(log), `cut at ${k}`).toEqual([[], ["reply"]]);
    expect(stepKinds(log, "reply").filter((kind) => kind === "interrupted")).toHaveLength(1);
    expect(stepKinds(log, "reply")).not.toContain("model_call");
    const authority = JsonlRunLog.resume(log);
    try {
      expect(foldedWaits(log).toJSON()).toEqual(replayRunLog(flow, authority).waits.toJSON());
    } finally {
      authority.close();
    }
  }
}, 60_000);

test("a host call re-pended after a replay stays open; an interrupt closes it", async () => {
  const model = new ScriptedModel([callReply(["hold", {}, "c1"])], CONFIG);
  const { log, options } = await waitingReply(model);
  const stopped = new AbortController();
  stopped.abort(new Error("host stop"));
  await rejected(advance({ ...options, signal: stopped.signal }));
  expect((await advance(options)).status).toBe("waiting");
  const [call] = pendingCalls(log);
  expect(call).toBeDefined();
  const { nodes, calls } = await interrupt(log, "m", { env });
  expect([nodes, calls]).toEqual([["reply"], [call!]]);
  const refused = (await rejected(complete(log, call!.callId, { output: "late" }, { env }))) as DeliveryRefused;
  expect(refused.reason).toBe("call_settled");
  const done = await advance(options);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toBeNull();
  expect(stepKinds(log, "reply").slice(-2)).toEqual(["tool_result", "interrupted"]);
});

test("offline Esc on a crashed run with a take waiting and a node in flight is recorded", async () => {
  // `gate`, a host tool beside the take, was mid-call when the run crashed.
  const text = gateFlow(`    kind: tool
    tool_id: lookup`);
  const flow = await loadFlow(text);
  const lookup = tool({ name: "lookup", description: "Look up.", parameters: { type: "object", properties: {} }, execute: () => "found" });
  const whole = freshLog();
  await runFlow(flow, {}, { env, log: whole, tools: { lookup }, models: serving(new ScriptedModel([], CONFIG)), runId: "r" });
  // Cut once `heard` waits, with `gate` started and not yet committed.
  const heardWaits = readRunLog(whole).findIndex(({ kind, record }) => kind === "node_waiting" && (record as { nodeId: string }).nodeId === "heard");
  const log = freshLog();
  await crashAfter(heardWaits + 1, () =>
    runFlow(flow, {}, { env, log, tools: { lookup }, models: serving(new ScriptedModel([], CONFIG)), runId: "r" }),
  );
  expect([...foldedWaits(log).waiting.keys()]).toEqual(["heard"]);
  expect(kinds(log).at(-1)).toBe("node_waiting");
  expect(readRunLog(log).some(({ kind, record }) => kind === "node_started" && (record as { nodeId: string }).nodeId === "gate")).toBe(true);
  const before = readRunLog(log).length;
  expect(await interrupt(log, "m", { env })).toEqual({ seq: before, nodes: [], calls: [] });
});

/** `reply` reads steer inbox `m` and calls `slow`; beside it `heard` takes from `m` once `gate` has run. */
const PARALLEL_TAKE = `
id: parallel_take
name: parallel_take
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
nodes:
  reply:
    kind: agent
    prompt: go
    tools: [slow]
    steer_inbox: \${input.m}
    output: Optional(str)
  gate:
    kind: tool
    tool_id: open
  heard:
    kind: tool
    tool_id: take
    depends_on: [gate]
    input:
      from: \${input.m}
      take: one
output: \${reply.output}
`;

/**
 * Run PARALLEL_TAKE at `log`: while `reply` runs `slow`, a live interrupt names it, then `gate`
 * opens and `heard`'s take waits on `m`, which ends the Stop with `reply` still stopping.
 */
async function stopThenTakeWaits(log: string) {
  let openGate!: () => void;
  const gateOpen = new Promise<void>((resolve) => (openGate = resolve));
  let heardWaits!: () => void;
  const heardWaiting = new Promise<void>((resolve) => (heardWaits = resolve));
  const parameters = { type: "object", properties: {} };
  const open = tool({ name: "open", description: "Open.", parameters, execute: async () => (await gateOpen, "open") });
  const slow = tool({
    name: "slow",
    description: "Slow.",
    parameters,
    execute: async () => {
      await interrupt(log, "m", { env });
      openGate();
      await heardWaiting;
      return "slow";
    },
  });
  const onEvent = (event: RunEvent) => void (event instanceof NodeWaiting && event.nodeId === "heard" && heardWaits());
  const model = new ScriptedModel([callReply(["slow", {}, "c1"]), textReply("done")], CONFIG);
  return runFlow(await loadFlow(PARALLEL_TAKE), {}, { env, log, tools: { open, slow }, models: serving(model), runId: "r", onEvent });
}

test("offline, a crashed reader still holding its interrupt is being stopped: a repeated Esc records nothing", async () => {
  const whole = freshLog();
  await stopThenTakeWaits(whole);
  const heardWaits = readRunLog(whole).findIndex(({ kind, record }) => kind === "node_waiting" && (record as { nodeId: string }).nodeId === "heard");
  const log = freshLog();
  await crashAfter(heardWaits + 1, () => stopThenTakeWaits(log));
  const w = foldedWaits(log);
  expect(stopsAt(log)).toEqual([["reply"]]);
  expect(w.stops.size).toBe(0);
  expect([...w.interrupts.keys()]).toEqual(["reply"]);
  const before = readFileSync(log);
  expect(await interrupt(log, "m", { env })).toEqual({ seq: null, nodes: [], calls: [] });
  expect(readFileSync(log).equals(before)).toBe(true);
});

/** `noter` reads steer inbox `b` and calls `slow`; after it, `reply` reads steer inbox `a`. */
const TWO_INBOXES = `
id: two_inboxes
name: two_inboxes
llm_config: {provider: openai, model: test-model}
input:
  a: !Inbox(str)
  b: !Inbox(str)
nodes:
  noter:
    kind: agent
    prompt: note
    tools: [slow]
    steer_inbox: \${input.b}
    output: Optional(str)
  reply:
    kind: agent
    prompt: answer
    steer_inbox: \${input.a}
    depends_on: [noter]
    output: Optional(str)
output: \${reply.output}
`;

/** Run TWO_INBOXES at `log`: while `noter` runs `slow`, a live interrupt of `b` names it. */
async function stopNoter(log: string) {
  const parameters = { type: "object", properties: {} };
  const slow = tool({ name: "slow", description: "Slow.", parameters, execute: async () => (await interrupt(log, "b", { env }), "slow") });
  const model = new ScriptedModel([callReply(["slow", {}, "c1"]), textReply("noted"), textReply("answer")], CONFIG);
  return runFlow(await loadFlow(TWO_INBOXES), {}, { env, log, tools: { slow }, models: serving(model), runId: "r" });
}

test("offline, a crashed reader stopping by another inbox doesn't make an Esc a no-op", async () => {
  const whole = freshLog();
  await stopNoter(whole);
  const stopped = readRunLog(whole).findIndex(({ record }) => record instanceof InterruptedRecord);
  const log = freshLog();
  await crashAfter(stopped + 1, () => stopNoter(log));
  expect([...foldedWaits(log).interrupts.keys()]).toEqual(["noter"]);
  const before = readRunLog(log).length;
  expect(await interrupt(log, "a", { env })).toEqual({ seq: before, nodes: [], calls: [] });
});
