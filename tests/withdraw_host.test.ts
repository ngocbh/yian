/**
 * TS-only: the host call `withdraw(log, inbox)`, of every unread send or of chosen ones, live and
 * offline.
 */
import { afterAll, expect, test } from "bun:test";
import { copyFileSync, mkdtempSync, readFileSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../src/compose/loader.ts";
import type { StreamChunk } from "../src/events.ts";
import { Finish, TextDelta } from "../src/llm/events.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import type { Author } from "../src/memory/base.ts";
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
    kind: call
    call: take
    input:
      from: \${input.m}
      take: one
  reply:
    kind: agent
    prompt: "answer \${got}"
    steer_inbox: \${input.m}
    input:
      got: \${heard.output.events}
    output: !Optional(str)
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

test("offline: a send assigned to the waiting take comes back with its seq, value and by; the take waits again", async () => {
  const { log, options } = await waitingAtTake();
  const seq = await send(log, "m", "hi", { env });
  expect(await withdraw(log, "m", { env })).toEqual([{ seq, value: "hi", by: null }]);
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
    { seq: first, value: "a", by: null },
    { seq: second, value: "b", by: null },
  ]);
  expect(await withdraw(log, "m", { env })).toEqual([]);
});

test("a closed inbox: allowed; the take gets {events: [], closed: true}", async () => {
  const model = new ScriptedModel([textReply("nothing to answer")], CONFIG);
  const { log, options } = await waitingAtTake(model);
  const seq = await send(log, "m", "hi", { env });
  await close(log, "m", { env });
  expect(await withdraw(log, "m", { env })).toEqual([{ seq, value: "hi", by: null }]);
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
  expect(taken).toEqual([{ seq: sent, value: "more", by: null }]);
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
  expect(both).toEqual([[{ seq: sent, value: "more", by: null }], []]);
});

test("a send nested past the value limit is refused, appending nothing, and withdraw then takes back nothing", async () => {
  const log = freshLog();
  const flow = await loadFlow(
    "id: notes\nname: notes\ninput:\n  notes: !Inbox(Any)\nnodes:\n  heard:\n    kind: call\n    call: take\n" +
      "    input: {from: \"${input.notes}\", take: all}\noutput: ${heard.output.events}\n",
  );
  expect((await runFlow(flow, {}, { env, log, runId: "r" })).status).toBe("waiting");
  const nest = (levels: number) => {
    let value: unknown = 1;
    for (let i = 0; i < levels; i++) value = [value];
    return value;
  };
  const before = readFileSync(log);
  for (const levels of [99, 200_000]) {
    const error = (await rejected(send(log, "notes", nest(levels), { env }))) as DeliveryRefused;
    expect([error instanceof DeliveryRefused, error.reason]).toEqual([true, "invalid_value"]);
  }
  expect(readFileSync(log).equals(before)).toBe(true);
  expect(await withdraw(log, "notes", { env })).toEqual([]);
  // At the limit less the take's two levels, the send lands and the run reads it.
  await send(log, "notes", nest(98), { env });
  await close(log, "notes", { env });
  expect((await advance({ log, env })).status).toBe("succeeded");
});

// --- Chosen sends ------------------------------------------------------------------------------ //

const ADA: Author = { kind: "user", name: "Ada" };
const OWNER: Author = { kind: "agent", name: "yian-chat" };

/** `first` takes one message from `m`, then `second` takes all of them; `other` takes all of `o`. */
const TWO_TAKES = `
id: two_takes
name: two_takes
input:
  m: !Inbox(str)
  o: !Inbox(str)
nodes:
  first: {kind: call, call: take, input: {from: "\${input.m}"}}
  second: {kind: call, call: take, runs_after: [first], input: {from: "\${input.m}", take: all}}
  other: {kind: call, call: take, input: {from: "\${input.o}", take: all}}
output: {first: "\${first.output.events}", second: "\${second.output.events}", other: "\${other.output.events}"}
`;

/** A run of `TWO_TAKES` waiting at `first`, and `finish`, which closes both inboxes and returns its output. */
async function twoTakes() {
  const log = freshLog();
  const options = { env, log, numWorkers: 0 };
  expect((await runFlow(await loadFlow(TWO_TAKES), {}, options)).status).toBe("waiting");
  const finish = async () => {
    await close(log, "m", { env });
    await close(log, "o", { env });
    const done = await advance(options);
    expect(done.status, done.failure?.message).toBe("succeeded");
    return done.output;
  };
  return { log, options, finish };
}

test("chosen sends: only those still unread come back; the rest stay queued, and a replay reads the record back", async () => {
  const { log, finish } = await twoTakes();
  const a = await send(log, "m", "a", { env });
  const b = await send(log, "m", "b", { env, by: ADA });
  const c = await send(log, "m", "c", { env });
  expect(await withdraw(log, "m", { env, events: [c, a] })).toEqual({
    withdrawn: [
      { seq: a, value: "a", by: null },
      { seq: c, value: "c", by: null },
    ],
    alreadyRead: [],
    alreadyWithdrawn: [],
  });
  const record = readRunLog(log).at(-1)!.record as InboxWithdrawnRecord;
  expect([record.inbox, [...record.events], record.by, record.chosen]).toEqual(["m", [a, c], null, true]);
  // A copy replays the record: only b is left to take back.
  const copy = freshLog();
  copyFileSync(log, copy);
  expect(await withdraw(copy, "m", { env })).toEqual([{ seq: b, value: "b", by: ADA }]);
  expect(await finish()).toEqual({ first: ["b"], second: [], other: [] });
});

test("a named send already read is reported apart from one taken back before; none unread appends nothing", async () => {
  const { log, options, finish } = await twoTakes();
  const x = await send(log, "m", "x", { env });
  // `first` reads x; `second` then waits.
  expect((await advance(options)).status).toBe("waiting");
  const y = await send(log, "m", "y", { env });
  await send(log, "m", "z", { env });
  expect(await withdraw(log, "m", { env, events: [y, x] })).toEqual({
    withdrawn: [{ seq: y, value: "y", by: null }],
    alreadyRead: [x],
    alreadyWithdrawn: [],
  });
  // A second client deleting y is told it was taken back, not that the run read it; a replay agrees.
  const before = readFileSync(log);
  const again = { withdrawn: [], alreadyRead: [x], alreadyWithdrawn: [y] };
  expect(await withdraw(log, "m", { env, events: [x, y] })).toEqual(again);
  expect(readFileSync(log).equals(before)).toBe(true);
  const copy = freshLog();
  copyFileSync(log, copy);
  expect(await withdraw(copy, "m", { env, events: [x, y] })).toEqual(again);
  expect(await finish()).toEqual({ first: ["x"], second: ["z"], other: [] });
});

test("with by, chosen sends must be by's own: another sender's is refused unknown_send, read or not", async () => {
  const { log, options, finish } = await twoTakes();
  const x = await send(log, "m", "x", { env, by: OWNER });
  expect((await advance(options)).status).toBe("waiting");
  const a = await send(log, "m", "a", { env, by: ADA });
  const b = await send(log, "m", "b", { env, by: OWNER });
  const before = readFileSync(log);
  for (const events of [[a, b], [x]]) {
    const refused = (await rejected(withdraw(log, "m", { env, by: ADA, events }))) as DeliveryRefused;
    const seq = events.at(-1);
    expect([refused.reason, refused.message]).toEqual(["unknown_send", `send ${seq} to inbox 'm' was made by another sender`]);
  }
  expect(readFileSync(log).equals(before)).toBe(true);
  expect(await withdraw(log, "m", { env, by: ADA, events: [a] })).toEqual({
    withdrawn: [{ seq: a, value: "a", by: ADA }],
    alreadyRead: [],
    alreadyWithdrawn: [],
  });
  expect((readRunLog(log).at(-1)!.record as InboxWithdrawnRecord).by).toEqual(ADA);
  expect(await finish()).toEqual({ first: ["x"], second: ["b"], other: [] });
});

test("chosen-send refusals: unknown_send for a seq no send to the inbox has; a bad list is refused before the log opens", async () => {
  const { log } = await twoTakes();
  const m = await send(log, "m", "m", { env });
  const o = await send(log, "o", "o", { env });
  const before = readFileSync(log);
  for (const seq of [o, o + 1, 1]) {
    const refused = (await rejected(withdraw(log, "m", { env, events: [m, seq] }))) as DeliveryRefused;
    expect([refused.reason, refused.message]).toEqual(["unknown_send", `the run has no send ${seq} to inbox 'm'`]);
  }
  expect(readFileSync(log).equals(before)).toBe(true);
  // Checked before the log is read: a missing log isn't reached.
  const missing = join(dir, "missing.jsonl");
  for (const events of [[1.5], ["1"], 1, [m, , o]]) {
    expect(await rejected(withdraw(missing, "m", { env, events: events as never }))).toBeInstanceOf(TypeError);
  }
  for (const [events, message] of [
    [[], "a withdraw's events must name at least one send"],
    [[0, m], "a withdraw's events must be positive seqs, got 0"],
    [[m, o, m], `a withdraw's events name send ${m} twice`],
  ] as const) {
    const error = (await rejected(withdraw(missing, "m", { env, events }))) as Error;
    expect([error instanceof RangeError, error.message]).toEqual([true, message]);
  }
});

test("live: a chosen withdraw through the engine takes back that send only, and a late steer reads the other", async () => {
  const chunks = ["a", " b", " c", " d", " e", " f", " g", " h", " i"];
  const model = new ScriptedModel(
    [[...chunks.map((text) => new TextDelta(0, text)), new Finish({ reason: "stop" })], textReply("read it")],
    CONFIG,
  );
  const { log, options } = await waitingAtTake(model);
  await send(log, "m", "hi", { env });
  let sent: number[] = [];
  let taken: unknown = null;
  const onStream = (chunk: StreamChunk) => {
    if (chunk.chunk !== "a") return;
    void (async () => {
      sent = [await send(log, "m", "take-me-back", { env }), await send(log, "m", "keep-me", { env })];
      taken = await withdraw(log, "m", { env, events: [sent[0]!] });
    })();
  };
  const done = await advance({ ...options, onStream, numWorkers: 0 });
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(done.output).toBe("read it");
  expect(taken).toEqual({ withdrawn: [{ seq: sent[0], value: "take-me-back", by: null }], alreadyRead: [], alreadyWithdrawn: [] });
  const late = JSON.stringify(model.requests.at(-1)!.messages);
  expect([late.includes("keep-me"), late.includes("take-me-back")]).toEqual([true, false]);
});
