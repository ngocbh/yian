/**
 * TS-only: the host call `interrupt(log, inbox)` on a run log, live through this process's engine
 * or offline under the log's lock, and the agents it cuts.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { ModelCallFinished, NodeSucceeded, type RunEvent, type StreamChunk } from "../../src/events.ts";
import { Finish, TextDelta } from "../../src/llm/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { FileLockError, lockFile } from "../../src/platform.ts";
import {
  advance,
  close,
  complete,
  DeliveryRefused,
  interrupt,
  pendingCalls,
  RunBusyError,
  runFlow,
} from "../../src/run.ts";
import { decodeRecord, InterruptedRecord, NodeStepRecord, readRunLog } from "../../src/runtime/log.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { callReply, crashAfter, rejected, serving, textReply } from "./run_state_fixtures.ts";

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
