/**
 * Who made each host call: `by` on `send`, `complete`, `release`, `close`, `withdraw`, `interrupt`
 * and `cancel` lands on the call's record and never in a node's step or a model request; a Stop's
 * `by` rides the record the run applies for it; a keyed repeat by someone else is `key_reused`.
 */
import { afterAll, expect, test } from "bun:test";
import { copyFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../src/compose/loader.ts";
import type { StreamChunk } from "../src/events.ts";
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
  await withdraw(log, "m", { env, by: OWNER });
  expect((last(log) as InboxWithdrawnRecord).by).toEqual(OWNER);
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

test("release records its by beside the released step", async () => {
  const log = freshLog();
  const lookup = tool({ name: "lookup", description: "Look up.", parameters: { type: "object", properties: {} }, execute: () => "found" });
  const models = serving(new ScriptedModel([callReply(["lookup", {}, "c1"])], CONFIG));
  const flow = await loadFlow(REPLY.replace("[hold, lookup]", "[lookup]"));
  const result = await runFlow(flow, {}, gated({ env, log, tools: { lookup }, models }, () => pending()));
  expect(result.status, result.failure?.message).toBe("waiting");
  const [held] = pendingCalls(log);
  await release(log, held!.callId, { env, by: OWNER, note: "fine" });
  const record = last(log) as NodeStepRecord;
  expect(record.step).toEqual({ kind: "released", call_id: held!.callId, note: "fine" });
  expect(record.by).toEqual(OWNER);
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
