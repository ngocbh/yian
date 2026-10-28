/**
 * Inbox runs end to end through the host calls: a run waits on a take of its `Inbox` input, the
 * host sends and closes, and `advance` moves it on; across a call, around a loop, and an `ask`
 * that times out.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { advance, cancel, close, complete, foldRunLog, pendingCalls, runFlow, send } from "../../src/run.ts";
import { PoolCommitRecord, readRunLog, RunCancelledRecord, RunStartedRecord } from "../../src/runtime/log.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { handClock } from "../helpers/inbox_fixtures.ts";
import { rejected } from "../helpers/run_state_fixtures.ts";
import { processChatFlow, processEnv as env } from "../helpers/yian-host-api-child-hold.ts";

const ROOT = mkdtempSync(join(tmpdir(), "host-api-e2e-"));
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));
let logs = 0;

/** A new log path in the test folder. */
function freshLog(): string {
  logs += 1;
  return join(ROOT, `run-${logs}.jsonl`);
}

/** A `gate` tool that opens at once. */
const tools = {
  gate: new FunctionTool({
    name: "gate",
    description: "opens at once",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "opened",
  }),
};

test("an inbox input: START holds its handle, the run-start commit omits it; send, close, advance", async () => {
  const log = freshLog();
  const flow = processChatFlow();
  expect((await runFlow(flow, {}, { log, env, tools })).status).toBe("waiting");
  const records = readRunLog(log).map((envelope) => envelope.record);
  expect((records[0] as RunStartedRecord).startWiringOverride).toEqual({ m: { inbox: "m" } });
  const starts = records.filter(
    (record): record is PoolCommitRecord => record instanceof PoolCommitRecord && record.key === START_ID,
  );
  // The run start's input commit, then the START node's output commit.
  expect(starts.map((record) => record.value)).toEqual([
    { kind: "object", value: {} },
    { kind: "object", value: { m: { inbox: "m" } } },
  ]);
  await send(log, "m", "hi", { env, flow });
  await close(log, "m", { env, flow });
  const advanced = await advance({ log, flow, env, tools });
  expect([advanced.status, advanced.output]).toEqual(["succeeded", { events: ["hi"], closed: true }]);
});

test("a run input for an inbox is refused", async () => {
  const error = await rejected(runFlow(processChatFlow(), { m: ["hi"] }, { log: freshLog(), env, tools }));
  expect(error).toEqual(new RangeError("inbox `m` receives events through send(); it takes no run input"));
});

const CHAT = `id: chat
name: chat
input:
  m: !Inbox(str)
nodes:
  heard:
    kind: tool
    tool_id: take
    input:
      from: \${input.m}
      take: all
output: \${heard.output}
`;

test("YAML: an inbox input waits on its take; send, close and advance end it", async () => {
  const log = freshLog();
  const flow = await loadFlow(CHAT);
  expect((await runFlow(flow, {}, { log, env })).status).toBe("waiting");
  const records = readRunLog(log).map((envelope) => envelope.record);
  expect((records[0] as RunStartedRecord).startWiringOverride).toEqual({ m: { inbox: "m" } });
  const starts = records.filter(
    (record): record is PoolCommitRecord => record instanceof PoolCommitRecord && record.key === START_ID,
  );
  expect(starts.map((record) => record.value)).toEqual([
    { kind: "object", value: {} },
    { kind: "object", value: { m: { inbox: "m" } } },
  ]);
  await send(log, "m", "hi", { env });
  await close(log, "m", { env });
  const advanced = await advance({ log, env });
  expect([advanced.status, advanced.output]).toEqual(["succeeded", { events: ["hi"], closed: true }]);
  const error = await rejected(runFlow(flow, { m: ["hi"] }, { log: freshLog(), env }));
  expect(error).toEqual(new RangeError("inbox `m` receives events through send(); it takes no run input"));
});

const CALLED = `id: called
name: called
input:
  m: !Inbox(str)
defs:
  listen:
    input:
      m: !Inbox(str)
    nodes:
      heard:
        kind: tool
        tool_id: take
        input:
          from: \${input.m}
    output: \${heard.output.events}
nodes:
  child:
    kind: call
    call: listen
    input:
      m: \${input.m}
output: \${child.output}
`;

test("YAML: the inbox handle crosses a call into the child's take", async () => {
  const log = freshLog();
  expect((await runFlow(await loadFlow(CALLED), {}, { log, env })).status).toBe("waiting");
  // The expansion's derived commit of the child's START rebuilds to the handle.
  const folded = await foldRunLog(log, { env });
  const childStart = folded.records.find(
    (folded) => folded.entry.envelope.record instanceof PoolCommitRecord && folded.entry.envelope.record.key === `child/${START_ID}`,
  );
  expect(childStart!.value!.toWire()).toEqual({ kind: "object", value: { m: { inbox: "m" } } });
  await send(log, "m", "hello", { env });
  const advanced = await advance({ log, env });
  expect([advanced.status, advanced.output]).toEqual(["succeeded", ["hello"]]);
});

const LOOP_CHAT = `id: chat
name: chat
input:
  m: !Inbox(str)
defs:
  turn:
    input:
      replies: !list(str)
      closed: !bool
      m: !Inbox(str)
    nodes:
      heard:
        kind: tool
        tool_id: take
        input:
          from: \${input.m}
          take: all
      reply:
        kind: code
        input:
          events: \${heard.output.events}
        output: list(str)
        code: |
          return inputs.events.map((e: string) => "re: " + e)
    output:
      replies: \${input.replies + reply.output}
      closed: \${heard.output.closed}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      replies: []
      closed: false
      m: \${input.m}
    while: "not closed"
output: \${chat.output}
`;

test("YAML: a loop chat turn per send; close ends the run with every reply", async () => {
  const log = freshLog();
  expect((await runFlow(await loadFlow(LOOP_CHAT), {}, { log, env })).status).toBe("waiting");
  for (const message of ["a", "b", "c"]) {
    await send(log, "m", message, { env });
    expect((await advance({ log, env })).status).toBe("waiting");
  }
  await close(log, "m", { env });
  const done = await advance({ log, env });
  expect(done.status).toBe("succeeded");
  // The loop's output carries its fields, never the inbox.
  expect(done.output).toEqual({ replies: ["re: a", "re: b", "re: c"], closed: true });
});

/** A flow asking with a one-second timeout, its output `output`. */
const askTimeout = (output: string) => `id: approve
name: approve
typedefs:
  Approval:
    decision: Literal(approve, reject)
nodes:
  lead:
    kind: tool
    tool_id: ask
    input:
      prompt: Approve?
      timeout: 1s
    output: ${output}
output: \${lead.output}
`;

const T0 = Date.UTC(2026, 9, 24, 9, 0, 0);

test("YAML: an unanswered ask with an Optional answer type times out to null", async () => {
  const log = freshLog();
  const tools = { ask: () => pending() };
  const flow = await loadFlow(askTimeout("Optional(Approval)"));
  expect((await runFlow(flow, {}, { log, env, tools, scheduler: handClock(T0) })).status).toBe("waiting");
  const advanced = await advance({ log, env, tools, scheduler: handClock(T0 + 2000) });
  expect([advanced.status, advanced.output]).toEqual(["succeeded", null]);
});

test("YAML: an unanswered ask with a required answer type fails with tool.timeout", async () => {
  const log = freshLog();
  const tools = { ask: () => pending() };
  const flow = await loadFlow(askTimeout("Approval"));
  expect((await runFlow(flow, {}, { log, env, tools, scheduler: handClock(T0) })).status).toBe("waiting");
  const advanced = await advance({ log, env, tools, scheduler: handClock(T0 + 2000) });
  expect([advanced.status, advanced.failure?.code]).toEqual(["failed", "tool.timeout"]);
});

const DRAIN = `id: drain
name: drain
input:
  m: !Inbox(str)
defs:
  turn:
    input:
      seen: !list(str)
      closed: !bool
      m: !Inbox(str)
    nodes:
      heard:
        kind: tool
        tool_id: take
        input:
          from: \${input.m}
          take: one
    output:
      seen: \${input.seen + heard.output.events}
      closed: \${heard.output.closed}
nodes:
  drain:
    kind: loop
    call: turn
    input:
      seen: []
      closed: false
      m: \${input.m}
    while: "not closed"
output: \${drain.output.seen}
`;

test("YAML: after close, takes drain what was sent, then read closed", async () => {
  const log = freshLog();
  expect((await runFlow(await loadFlow(DRAIN), {}, { log, env })).status).toBe("waiting");
  await send(log, "m", "a", { env });
  await send(log, "m", "b", { env });
  await close(log, "m", { env });
  const done = await advance({ log, env });
  expect([done.status, done.output]).toEqual(["succeeded", ["a", "b"]]);
});

const ASK = `id: approve
name: approve
nodes:
  lead:
    kind: tool
    tool_id: ask
    input:
      prompt: Approve this plan?
      timeout: 1h
output: \${lead.output}
`;

test("YAML: a host rebuilds its open ask from pendingCalls and completes it", async () => {
  const log = freshLog();
  const tools = { ask: () => pending() };
  const started = await runFlow(await loadFlow(ASK), {}, { log, env, tools, scheduler: handClock(T0) });
  expect(started.status).toBe("waiting");
  const [call, ...rest] = pendingCalls(log);
  expect(rest).toEqual([]);
  expect(call).toEqual({
    callId: `${started.runId}/lead`,
    node: "lead",
    tool: "ask",
    arguments: expect.objectContaining({ prompt: "Approve this plan?" }),
    pendingSeq: expect.any(Number),
    deadline: "2026-10-24T10:00:00.000Z",
  });
  expect(readRunLog(log)[call!.pendingSeq]!.kind).toBe("node_step");
  expect(typeof (await complete(log, call!.callId, { output: "yes" }, { env }))).toBe("number");
  expect(pendingCalls(log)).toEqual([]);
  const done = await advance({ log, env, tools, scheduler: handClock(T0) });
  expect([done.status, done.output]).toEqual(["succeeded", "yes"]);
});

test("YAML: cancel closes the open ask and ends the run", async () => {
  const log = freshLog();
  const tools = { ask: () => pending() };
  expect((await runFlow(await loadFlow(ASK), {}, { log, env, tools, scheduler: handClock(T0) })).status).toBe("waiting");
  const open = pendingCalls(log);
  expect(await cancel(log, { env })).toEqual(open);
  expect(readRunLog(log).at(-1)!.record).toBeInstanceOf(RunCancelledRecord);
  expect(pendingCalls(log)).toEqual([]);
  expect((await advance({ log, env, tools })).status).toBe("cancelled");
});
