/**
 * TS-only: the host call `withdraw(log, inbox)`, live and offline.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../src/compose/loader.ts";
import type { StreamChunk } from "../src/events.ts";
import { Finish, TextDelta } from "../src/llm/events.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { FileLockError, lockFile } from "../src/platform.ts";
import { advance, cancel, close, DeliveryRefused, RunBusyError, runFlow, send, withdraw } from "../src/run.ts";
import { decodeRecord, InboxWithdrawnRecord, NodeStepRecord, PoolCommitRecord, readRunLog } from "../src/runtime/log.ts";
import { hostEnv } from "./helpers/mcp_helpers.ts";
import { rejected, serving, textReply } from "./helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-withdraw-host-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

const CONFIG = { provider: "openai", model: "test-model" };
const env = hostEnv(dir);

/** `heard` takes one message from `m`; then `reply`, an AGENT reading steer inbox `m`, answers it. */
const FLOW = `
id: withdraw_host
name: withdraw_host
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
  reply:
    kind: agent
    prompt: "answer \${got}"
    steer_inbox: \${input.m}
    input:
      got: \${heard.output.events}
    output: Optional(str)
output: \${reply.output}
`;

/** The kinds of the log's records, a step as its own `kind`. */
const kinds = (log: string) =>
  readFileSync(log, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => decodeRecord(line))
    .map(({ kind, record }) => (record instanceof NodeStepRecord ? String(record.step["kind"]) : kind));

/** A run waiting at its take, with `model` serving the reply. */
async function waitingAtTake(model = new ScriptedModel([], CONFIG)) {
  const log = freshLog();
  const flow = await loadFlow(FLOW);
  const options = { env, log, models: serving(model), runId: "r" };
  const result = await runFlow(flow, {}, options);
  expect(result.status, result.failure?.message).toBe("waiting");
  return { log, flow, options, model };
}

test("offline: a send assigned to the waiting take comes back with its seq and value; the take waits again", async () => {
  const { log, options } = await waitingAtTake();
  const seq = await send(log, "m", "hi", { env });
  expect(await withdraw(log, "m", { env })).toEqual([{ seq, value: "hi" }]);
  const record = readRunLog(log).at(-1)!.record as InboxWithdrawnRecord;
  expect([record.inbox, [...record.events]]).toEqual(["m", [seq]]);
  // Nothing to wake: no model call.
  expect((await advance(options)).status).toBe("waiting");
  expect(kinds(log).filter((kind) => kind === "model_call")).toEqual([]);
});

test("nothing unread: [] and nothing appended", async () => {
  const { log } = await waitingAtTake();
  const before = readFileSync(log);
  expect(await withdraw(log, "m", { env })).toEqual([]);
  expect(readFileSync(log).equals(before)).toBe(true);
});

test("two withdraws: the second returns []", async () => {
  const { log } = await waitingAtTake();
  const first = await send(log, "m", "a", { env });
  const second = await send(log, "m", "b", { env });
  expect(await withdraw(log, "m", { env })).toEqual([
    { seq: first, value: "a" },
    { seq: second, value: "b" },
  ]);
  expect(await withdraw(log, "m", { env })).toEqual([]);
});

test("a closed inbox: allowed; the take gets {events: [], closed: true}", async () => {
  const model = new ScriptedModel([textReply("nothing to answer")], CONFIG);
  const { log, options } = await waitingAtTake(model);
  const seq = await send(log, "m", "hi", { env });
  await close(log, "m", { env });
  expect(await withdraw(log, "m", { env })).toEqual([{ seq, value: "hi" }]);
  const done = await advance(options);
  expect(done.status, done.failure?.message).toBe("succeeded");
  const take = readRunLog(log).find(({ record }) => record instanceof PoolCommitRecord && record.take !== null)!.record as PoolCommitRecord;
  expect(take.take).toEqual({ inbox: "m", events: [], closed: true });
});

test("refusals: ended, unknown_inbox; TypeError for a non-string inbox; RunBusyError under another writer", async () => {
  const { log } = await waitingAtTake();
  await send(log, "m", "hi", { env });
  const unknown = (await rejected(withdraw(log, "nope", { env }))) as DeliveryRefused;
  expect([unknown.reason, unknown.message]).toEqual(["unknown_inbox", "the run has no inbox 'nope'"]);
  expect(await rejected(withdraw(log, 1 as never, { env }))).toBeInstanceOf(TypeError);
  const holder = lockFile(`${realpathSync(log)}.lock`);
  try {
    const busy = (await rejected(withdraw(log, "m", { env }))) as RunBusyError;
    expect([busy instanceof RunBusyError, busy.cause instanceof FileLockError]).toEqual([true, true]);
  } finally {
    holder.release();
  }
  await cancel(log, { env });
  const ended = (await rejected(withdraw(log, "m", { env }))) as DeliveryRefused;
  expect(ended.reason).toBe("ended");
});

test("live: a steer queued during the reply's final answer comes back, and no late steer reads it", async () => {
  // The reply's only model call is a tool-free final answer, so a queued send would be read by the
  // late-steer path and answered by a second model call.
  const chunks = ["a", " b", " c", " d", " e", " f"];
  const model = new ScriptedModel([[...chunks.map((text) => new TextDelta(0, text)), new Finish({ reason: "stop" })]], CONFIG);
  const { log, options } = await waitingAtTake(model);
  await send(log, "m", "hi", { env });
  let sent: number | null = null;
  let taken: unknown = null;
  const onStream = (chunk: StreamChunk) => {
    if (chunk.chunk !== "a") return;
    void (async () => {
      sent = await send(log, "m", "more", { env });
      taken = await withdraw(log, "m", { env });
    })();
  };
  const done = await advance({ ...options, onStream, numWorkers: 0 });
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toBe(chunks.join(""));
  expect(taken).toEqual([{ seq: sent, value: "more" }]);
  const steps = readRunLog(log).flatMap(({ record }) => (record instanceof NodeStepRecord && record.nodeId === "reply" ? [record.step["kind"]] : []));
  expect(steps).toEqual(["model_call"]);
});

test("two hosts racing: two live withdraws in one tick give the sends once and []", async () => {
  const chunks = ["a", " b", " c", " d", " e", " f"];
  const model = new ScriptedModel([[...chunks.map((text) => new TextDelta(0, text)), new Finish({ reason: "stop" })]], CONFIG);
  const { log, options } = await waitingAtTake(model);
  await send(log, "m", "hi", { env });
  let sent: number | null = null;
  let both: unknown = null;
  const onStream = (chunk: StreamChunk) => {
    if (chunk.chunk !== "a") return;
    void (async () => {
      sent = await send(log, "m", "more", { env });
      both = await Promise.all([withdraw(log, "m", { env }), withdraw(log, "m", { env })]);
    })();
  };
  const done = await advance({ ...options, onStream, numWorkers: 0 });
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(both).toEqual([[{ seq: sent, value: "more" }], []]);
});
