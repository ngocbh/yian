/**
 * Who made each host call: `by` on `send`, `complete`, `release`, `close`, `withdraw`, `interrupt`
 * and `cancel` lands on the call's record and never in a node's step or a model request; a Stop's
 * `by` rides the record the run applies for it; a keyed repeat by someone else is `key_reused`; a
 * withdraw by someone takes back only their sends; a live send publishes `InboxSent` with its `by`.
 */
import { afterAll, expect, test } from "bun:test";
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../src/compose/loader.ts";
import { InboxSent, type RunEvent, type StreamChunk } from "../src/events.ts";
import { Finish, TextDelta } from "../src/llm/events.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import type { Author } from "../src/memory/base.ts";
import {
  advance,
  cancel,
  close,
  complete,
  DeliveryRefused,
  interrupt,
  pendingCalls,
  release,
  runFlow,
  send,
  withdraw,
} from "../src/run.ts";
import {
  InboxClosedRecord,
  InboxWithdrawnRecord,
  InterruptedRecord,
  NodeStepRecord,
  PoolCommitRecord,
  readRunLog,
  RunCancelledRecord,
  type RunRecord,
} from "../src/runtime/log.ts";
import { pending } from "../src/tools/context.ts";
import { tool } from "../src/tools/function_tools.ts";
import { YianRangeError, YianTypeError } from "../src/typesys/errors.ts";
import { gated } from "./helpers/gate.ts";
import { hostEnv } from "./helpers/mcp_helpers.ts";
import { callReply, rejected, serving, textReply } from "./helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-host-by-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

const env = hostEnv(dir);
const CONFIG = { provider: "openai", model: "test-model" };

const ADA: Author = { kind: "user", name: "Ada" };
const OWNER: Author = { kind: "agent", name: "yian-chat" };
const SOP: Author = { kind: "run", name: "review r1", run: "chat-r1" };
const HOST: Author = { kind: "host", name: "daemon" };

/** The last record of the log at `log`. */
const last = (log: string): RunRecord => readRunLog(log).at(-1)!.record;

/** The nodes and `by` of each `interrupted` record at `log`, in log order. */
const stopsAt = (log: string) =>
  readRunLog(log).flatMap(({ record }) => (record instanceof InterruptedRecord ? [[[...record.nodes], record.by]] : []));

/** A take of every message on `m`, waiting. */
const NOTES = `
id: notes
name: notes
input:
  m: !Inbox(str)
nodes:
  heard:
    kind: call
    call: take
    input:
      from: \${input.m}
      take: all
output: \${heard.output.events}
`;

/** One AGENT `reply` reading steer inbox `m`, granted `hold`, which pends, and `lookup`. */
const REPLY = `
id: reply
name: reply
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
nodes:
  reply:
    kind: agent
    prompt: go
    tools: [hold, lookup]
    steer_inbox: \${input.m}
    output: !Optional(str)
output: \${reply.output}
`;

/** An ask holds the reply back; then `reply` reads steer inbox `m`. */
const ASK_THEN_REPLY = `
id: ask_then_reply
name: ask_then_reply
llm_config: {provider: openai, model: test-model}
input:
  m: !Inbox(str)
nodes:
  gate:
    kind: call
    call: ask
    input:
      prompt: approve?
  reply:
    kind: agent
    prompt: go
    steer_inbox: \${input.m}
    input:
      gated: \${gate.output}
    output: !Optional(str)
output: \${reply.output}
`;

const tools = () => ({
  hold: tool({ name: "hold", description: "Hold.", parameters: { type: "object", properties: {} }, execute: () => pending() }),
  lookup: tool({ name: "lookup", description: "Look up.", parameters: { type: "object", properties: {} }, execute: () => "found" }),
});

test("send, withdraw and close record their by; a waiting run's cancel and a cancel from the records alone do too", async () => {
  const log = freshLog();
  expect((await runFlow(await loadFlow(NOTES), {}, { env, log })).status).toBe("waiting");
  await send(log, "m", "hi", { env, by: ADA });
  expect((last(log) as PoolCommitRecord).by).toEqual(ADA);
  await withdraw(log, "m", { env, by: ADA });
  expect((last(log) as InboxWithdrawnRecord).by).toEqual(ADA);
  await close(log, "m", { env, by: SOP });
  expect((last(log) as InboxClosedRecord).by).toEqual(SOP);
  // A copy has no state in this process, so its cancel reads only the records.
  const copy = freshLog();
  copyFileSync(log, copy);
  await cancel(log, { env, by: HOST });
  await cancel(copy, { env, by: ADA });
  expect([(last(log) as RunCancelledRecord).by, (last(copy) as RunCancelledRecord).by]).toEqual([HOST, ADA]);
  // A call that passes none records none.
  const plain = freshLog();
  await runFlow(await loadFlow(NOTES), {}, { env, log: plain });
  await send(plain, "m", "hi", { env });
  expect((last(plain) as PoolCommitRecord).by).toBeNull();
  expect(readFileSync(plain, "utf8")).not.toContain('"by"');
});

test("a withdraw by someone takes back only their unread sends, each with its by; the run reads the rest", async () => {
  const log = freshLog();
  const options = { env, log };
  expect((await runFlow(await loadFlow(NOTES), {}, options)).status).toBe("waiting");
  const a = await send(log, "m", "a", { env, by: ADA });
  const b = await send(log, "m", "b", { env, by: OWNER });
  const c = await send(log, "m", "c", { env });
  const d = await send(log, "m", "d", { env, by: ADA });
  expect(await withdraw(log, "m", { env, by: ADA })).toEqual([
    { seq: a, value: "a", by: ADA },
    { seq: d, value: "d", by: ADA },
  ]);
  const before = readFileSync(log);
  for (const by of [ADA, HOST]) expect(await withdraw(log, "m", { env, by })).toEqual([]);
  expect(readFileSync(log).equals(before)).toBe(true);
  // A copy replays the withdraw by Ada; a withdraw by no one takes back every sender's.
  const copy = freshLog();
  copyFileSync(log, copy);
  expect(await withdraw(copy, "m", { env })).toEqual([
    { seq: b, value: "b", by: OWNER },
    { seq: c, value: "c", by: null },
  ]);
  await close(log, "m", { env });
  expect((await advance(options)).output).toEqual(["b", "c"]);
});

test("on a closed inbox, the take that held a withdrawn send gets the others' send, and the next take the close", async () => {
  const log = freshLog();
  const flow = await loadFlow(
    "id: two\nname: two\ninput:\n  m: !Inbox(str)\nnodes:\n  first: {kind: call, call: take, input: {from: \"${input.m}\"}}\n" +
      "  second: {kind: call, call: take, runs_after: [first], input: {from: \"${input.m}\"}}\n" +
      "output: {first: \"${first.output}\", second: \"${second.output}\"}\n",
  );
  const options = { env, log, numWorkers: 0 };
  expect((await runFlow(flow, {}, options)).status).toBe("waiting");
  const x = await send(log, "m", "x", { env, by: ADA });
  await send(log, "m", "y", { env, by: OWNER });
  await close(log, "m", { env });
  expect(await withdraw(log, "m", { env, by: ADA })).toEqual([{ seq: x, value: "x", by: ADA }]);
  expect((await advance(options)).output).toEqual({
    first: { events: ["y"], closed: true },
    second: { events: [], closed: true },
  });
});

test("a send to a run moving in this process publishes InboxSent; a live withdraw by someone keeps the others' sends", async () => {
  const log = freshLog();
  const model = new ScriptedModel(
    [[new TextDelta(0, "hello"), new TextDelta(0, " big"), new Finish({ reason: "stop" })], textReply("done")],
    CONFIG,
  );
  const events: RunEvent[] = [];
  let taken: unknown = null;
  const onStream = (chunk: StreamChunk) => {
    if (chunk.chunk !== " big") return;
    void (async () => {
      await send(log, "m", "from Ada", { env, by: ADA });
      await send(log, "m", "from the owner", { env, by: OWNER });
      taken = await withdraw(log, "m", { env, by: ADA });
    })();
  };
  const options = { env, log, tools: tools(), models: serving(model), runId: "r", onStream, numWorkers: 0 };
  const done = await runFlow(await loadFlow(REPLY), {}, { ...options, onEvent: (event: RunEvent) => void events.push(event) });
  expect(done.output).toBe("done");
  const sends = readRunLog(log).filter(({ record }) => record instanceof PoolCommitRecord && record.by !== null);
  expect(events.filter((event) => event instanceof InboxSent)).toEqual([
    new InboxSent({ inbox: "m", seq: sends[0]!.seq, by: ADA }),
    new InboxSent({ inbox: "m", seq: sends[1]!.seq, by: OWNER }),
  ]);
  expect(taken).toEqual([{ seq: sends[0]!.seq, value: "from Ada", by: ADA }]);
  // The late steer reads only the owner's line.
  const steered = JSON.stringify(model.requests[1]);
  expect([steered.includes("from the owner"), steered.includes("from Ada")]).toEqual([true, false]);
});

test("a send to a waiting run publishes nothing, nor does the advance that replays it", async () => {
  const log = freshLog();
  const events: RunEvent[] = [];
  const options = { env, log, onEvent: (event: RunEvent) => void events.push(event) };
  expect((await runFlow(await loadFlow(NOTES), {}, options)).status).toBe("waiting");
  await send(log, "m", "hi", { env, by: ADA });
  await close(log, "m", { env });
  const copy = freshLog();
  copyFileSync(log, copy);
  expect((await advance(options)).output).toEqual(["hi"]);
  expect((await advance({ ...options, log: copy })).output).toEqual(["hi"]);
  expect(events.filter((event) => event instanceof InboxSent)).toEqual([]);
});

test("complete records its by beside the step, which holds none, and the model never reads it", async () => {
  const log = freshLog();
  const model = new ScriptedModel([callReply(["hold", {}, "c1"]), textReply("done")], CONFIG);
  const options = { env, log, tools: tools(), models: serving(model), runId: "r" };
  expect((await runFlow(await loadFlow(REPLY), {}, options)).status).toBe("waiting");
  const [call] = pendingCalls(log);
  const person: Author = { kind: "user", name: "Zelda Quartermain" };
  await complete(log, call!.callId, { output: "ok" }, { env, by: person });
  const record = last(log) as NodeStepRecord;
  expect([record.step["kind"], record.by]).toEqual(["tool_result", person]);
  expect(Object.keys(record.step)).not.toContain("by");
  expect((await advance(options)).status).toBe("succeeded");
  expect(JSON.stringify(model.requests)).not.toContain("Zelda Quartermain");
});

/** A run of `REPLY` waiting on its agent's one `lookup` call, which the hook held. */
async function heldRun() {
  const log = freshLog();
  const lookup = tool({ name: "lookup", description: "Look up.", parameters: { type: "object", properties: {} }, execute: () => "found" });
  const models = serving(new ScriptedModel([callReply(["lookup", {}, "c1"])], CONFIG));
  const flow = await loadFlow(REPLY.replace("[hold, lookup]", "[lookup]"));
  const result = await runFlow(flow, {}, gated({ env, log, tools: { lookup }, models }, () => pending()));
  expect(result.status, result.failure?.message).toBe("waiting");
  return { log, callId: pendingCalls(log)[0]!.callId };
}

test("release records its by beside the released step", async () => {
  const { log, callId } = await heldRun();
  await release(log, callId, { env, by: OWNER, note: "fine" });
  const record = last(log) as NodeStepRecord;
  expect(record.step).toEqual({ kind: "released", call_id: callId, note: "fine" });
  expect(record.by).toEqual(OWNER);
});

test("a repeated complete or release by the same by is null, by anyone else call_settled, also from a replay", async () => {
  const answered = freshLog();
  const model = new ScriptedModel([callReply(["hold", {}, "c1"])], CONFIG);
  await runFlow(await loadFlow(REPLY), {}, { env, log: answered, tools: tools(), models: serving(model), runId: "r" });
  const [call] = pendingCalls(answered);
  await complete(answered, call!.callId, { output: "ok" }, { env, by: ADA });
  const held = await heldRun();
  await release(held.log, held.callId, { env, by: OWNER, note: "fine" });
  type Again = (at: string, by: Author | null) => Promise<number | null>;
  const cases: [string, Again, Author, "answered" | "released"][] = [
    [answered, (at, by) => complete(at, call!.callId, { output: "ok" }, { env, by }), ADA, "answered"],
    // A release's note may differ: the first one stands.
    [held.log, (at, by) => release(at, held.callId, { env, by, note: "again" }), OWNER, "released"],
  ];
  for (const [log, again, first, state] of cases) {
    const copy = freshLog();
    copyFileSync(log, copy);
    for (const at of [log, copy]) {
      const before = readFileSync(at);
      expect(await again(at, first)).toBeNull();
      for (const by of [HOST, null]) {
        const error = (await rejected(again(at, by))) as DeliveryRefused;
        expect([error.reason, error.callState]).toEqual(["call_settled", state]);
      }
      expect(readFileSync(at).equals(before)).toBe(true);
    }
  }
});

test("a Stop's by rides the record the run applies for it when the reply opens the inbox", async () => {
  const log = freshLog();
  const model = new ScriptedModel([], CONFIG);
  const options = { env, log, models: serving(model), runId: "r", tools: { ask: () => pending() } };
  expect((await runFlow(await loadFlow(ASK_THEN_REPLY), {}, options)).status).toBe("waiting");
  const [ask] = pendingCalls(log);
  const stop = await interrupt(log, "m", { env, by: ADA });
  expect(stop.nodes).toEqual([]);
  await complete(log, ask!.callId, { output: "yes" }, { env, by: OWNER });
  expect((await advance(options)).output).toBeNull();
  expect(stopsAt(log)).toEqual([
    [[], ADA],
    [["reply"], ADA],
  ]);
});

test("a resume from a copy of the log between a Stop and the record the run applies for it keeps the Stop's by", async () => {
  const log = freshLog();
  const options = { env, log, models: serving(new ScriptedModel([], CONFIG)), runId: "r", tools: { ask: () => pending() } };
  expect((await runFlow(await loadFlow(ASK_THEN_REPLY), {}, options)).status).toBe("waiting");
  const [ask] = pendingCalls(log);
  await interrupt(log, "m", { env, by: ADA });
  // A copy has no state in this process: its complete and advance replay the records.
  const copy = freshLog();
  copyFileSync(log, copy);
  await complete(copy, ask!.callId, { output: "yes" }, { env, by: OWNER });
  expect((await advance({ ...options, log: copy })).output).toBeNull();
  expect(stopsAt(copy)).toEqual([
    [[], ADA],
    [["reply"], ADA],
  ]);
});

test("a second interrupt while a Stop stands records nothing, whoever makes it", async () => {
  const log = freshLog();
  const options = { env, log, models: serving(new ScriptedModel([], CONFIG)), runId: "r", tools: { ask: () => pending() } };
  expect((await runFlow(await loadFlow(ASK_THEN_REPLY), {}, options)).status).toBe("waiting");
  await interrupt(log, "m", { env, by: ADA });
  const before = readFileSync(log);
  expect(await interrupt(log, "m", { env, by: OWNER })).toEqual({ seq: null, nodes: [], calls: [], released: [] });
  expect(readFileSync(log).equals(before)).toBe(true);
});

test("a live interrupt records its by", async () => {
  const log = freshLog();
  const model = new ScriptedModel([[new TextDelta(0, "hello"), new TextDelta(0, " big"), new Finish({ reason: "stop" })]], CONFIG);
  const onStream = (chunk: StreamChunk) => void (chunk.chunk === " big" && interrupt(log, "m", { env, by: ADA }));
  const options = { env, log, tools: tools(), models: serving(model), runId: "r", onStream, numWorkers: 0 };
  expect((await runFlow(await loadFlow(REPLY), {}, options)).output).toBeNull();
  expect(stopsAt(log)).toEqual([[["reply"], ADA]]);
});

test("a keyed send repeated by someone else is key_reused, also from a replay", async () => {
  const log = freshLog();
  await runFlow(await loadFlow(NOTES), {}, { env, log });
  const seq = await send(log, "m", "hi", { env, key: "k1", by: ADA });
  expect(await send(log, "m", "hi", { env, key: "k1", by: { name: "Ada", kind: "user" } })).toBe(seq);
  const copy = freshLog();
  copyFileSync(log, copy);
  for (const at of [log, copy]) {
    const before = readFileSync(at);
    for (const by of [OWNER, null]) {
      const error = (await rejected(send(at, "m", "hi", { env, key: "k1", by }))) as DeliveryRefused;
      expect([error.reason, error.message]).toEqual(["key_reused", `send key 'k1' was used by send ${seq} with another value, inbox or by`]);
    }
    expect(readFileSync(at).equals(before)).toBe(true);
  }
});

test("a by that isn't an author is refused before the log is opened", async () => {
  const missing = join(dir, "missing.jsonl");
  const cases: [unknown, Error][] = [
    ["Ada", new YianTypeError("a host call's by must be {kind, name, run?, node?} or null")],
    [{ kind: "robot", name: "x" }, new YianTypeError("a host call's by kind must be one of agent, tool, user, host, run")],
    [{ kind: "user", name: "" }, new YianRangeError("a host call's by name must be a non-empty string")],
  ];
  for (const [by, error] of cases) {
    expect(await rejected(send(missing, "m", "x", { by: by as Author }))).toEqual(error);
    expect(await rejected(cancel(missing, { by: by as Author }))).toEqual(error);
  }
});
