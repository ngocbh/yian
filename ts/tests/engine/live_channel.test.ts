/**
 * TS-only: the live channel of a run. `LiveChannel` forwards each appended record and each AGENT
 * call's start, chunks and end as plain-JSON frames named by the step they become (`node` + `n`),
 * and its `snapshot()` lists the calls in flight with their chunks so far, one buffer per call, so a
 * late viewer rebuilds every call's text from the log, the snapshot and the frames after it.
 *
 * First on a bare channel, then through runs with a model that streams one chunk per gate the test
 * opens, so a test can look at the channel between any two chunks.
 */

import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import type { RunEvent } from "../../src/events.ts";
import { type ChatModel, type Event, EventStream, Finish, TextDelta } from "../../src/llm/events.ts";
import { Message, type Request } from "../../src/llm/messages.ts";
import { LiveChannel, type LiveFrame, type LiveSnapshot } from "../../src/runtime/live.ts";
import {
  decodeRecord,
  encodeRecord,
  iterRunRecords,
  MemoryRunLog,
  NodeStartedRecord,
  NodeStepRecord,
} from "../../src/runtime/log.ts";
import { withWarnings } from "../../src/platform.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
import { logJson } from "../../src/typesys/json.ts";
import { crashAfter, NO_ENV, rejected, serving } from "./run_state_fixtures.ts";

/** A `model_call` step record of `node`'s step `n`. */
function step(node: string, n: number): NodeStepRecord {
  return new NodeStepRecord(node, n, { kind: "model_call" });
}

const T0 = "2026-10-23T10:00:00.000Z";

/**
 * A channel, the frames its subscriber got, and `append`, which appends a record to a process-local
 * log the channel wraps and returns the appended line.
 */
function subscribed(options?: ConstructorParameters<typeof LiveChannel>[0]) {
  const channel = new LiveChannel(options);
  const frames: LiveFrame[] = [];
  const stop = channel.subscribe((frame) => void frames.push(frame));
  const log = channel.wrap(new MemoryRunLog("run-1", { clock: () => Date.parse(T0) }));
  const append = (record: NodeStepRecord | NodeStartedRecord) => encodeRecord(log.append(record));
  return { channel, frames, stop, log, append };
}

test("frames reach subscribers in call order and survive a JSON round trip", () => {
  const { channel, frames, append } = subscribed();
  const started = append(new NodeStartedRecord("a"));
  channel.start("a", 1, "gpt-test", "chat", T0);
  channel.chunk("a", 1, 0, "hel");
  channel.chunk("a", 1, 1, "lo");
  channel.end("a", 1, null);
  expect(frames).toEqual([
    { type: "record", seq: 0, line: started },
    { type: "start", node: "a", n: 1, model: "gpt-test", purpose: "chat", started_at: T0 },
    { type: "chunk", node: "a", n: 1, index: 0, text: "hel" },
    { type: "chunk", node: "a", n: 1, index: 1, text: "lo" },
    { type: "end", node: "a", n: 1, error: null },
  ]);
  for (const frame of frames) expect(JSON.parse(logJson(frame))).toEqual(frame);
});

test("the snapshot lists interleaved calls with their own chunks", () => {
  const { channel, append } = subscribed();
  expect(channel.snapshot()).toEqual({ last_seq: -1, calls: [] });
  append(new NodeStartedRecord("a"));
  channel.start("a", 1, null, "chat", T0);
  channel.start("b", 1, null, "chat", T0);
  channel.chunk("a", 1, 0, "a0");
  channel.chunk("b", 1, 0, "b0");
  channel.chunk("a", 1, 1, "a1");
  const snapshot = channel.snapshot();
  expect(snapshot).toEqual({
    last_seq: 0,
    calls: [
      { node: "a", n: 1, chunks: ["a0", "a1"], next_index: 2 },
      { node: "b", n: 1, chunks: ["b0"], next_index: 1 },
    ],
  });
  expect(JSON.parse(logJson(snapshot))).toEqual(snapshot);
  // A snapshot is a copy: later chunks don't change it.
  channel.chunk("b", 1, 1, "b1");
  expect(snapshot.calls[1]!.chunks).toEqual(["b0"]);
});

test("a call's step record drops only that call; an end with an error drops it too", () => {
  const { channel, append } = subscribed();
  channel.start("a", 1, null, "chat", T0);
  channel.start("b", 1, null, "chat", T0);
  channel.chunk("a", 1, 0, "a0");
  channel.chunk("b", 1, 0, "b0");
  // Another node's step with the same n, and a record that isn't a step, drop nothing.
  append(step("c", 1));
  append(new NodeStartedRecord("a"));
  expect(channel.snapshot().calls.map((call) => call.node)).toEqual(["a", "b"]);
  append(step("a", 1));
  expect(channel.snapshot()).toEqual({ last_seq: 2, calls: [{ node: "b", n: 1, chunks: ["b0"], next_index: 1 }] });
  channel.start("b", 2, null, "chat", T0);
  channel.end("b", 1, "ProviderError");
  expect(channel.snapshot().calls).toEqual([{ node: "b", n: 2, chunks: [], next_index: 0 }]);
  // An end without an error leaves the call to its record.
  channel.end("b", 2, null);
  expect(channel.snapshot().calls).toEqual([{ node: "b", n: 2, chunks: [], next_index: 0 }]);
});

test("settling a node drops its calls", () => {
  const { channel } = subscribed();
  channel.start("a", 1, null, "chat", T0);
  channel.chunk("a", 1, 0, "cut");
  channel.start("b", 1, null, "chat", T0);
  channel.settle("a");
  expect(channel.snapshot().calls.map((call) => call.node)).toEqual(["b"]);
});

test("an unsubscribed function gets nothing more", () => {
  const { channel, frames, stop } = subscribed();
  channel.start("a", 1, null, "chat", T0);
  stop();
  channel.chunk("a", 1, 0, "late");
  expect(frames.map((frame) => frame.type)).toEqual(["start"]);
  // The channel still keeps the call for a later viewer.
  expect(channel.snapshot().calls).toEqual([{ node: "a", n: 1, chunks: ["late"], next_index: 1 }]);
});

test("a subscriber that throws is dropped and reported; the others and the log go on", () => {
  const reported: [unknown, LiveFrame][] = [];
  const { channel, frames, log, append } = subscribed({
    onSubscriberError: (error, frame) => void reported.push([error, frame]),
  });
  const boom = new Error("viewer gone");
  let calls = 0;
  channel.subscribe(() => {
    calls++;
    throw boom;
  });
  const after: LiveFrame[] = [];
  channel.subscribe((frame) => void after.push(frame));
  append(new NodeStartedRecord("a"));
  channel.start("a", 1, null, "chat", T0);
  expect(calls).toBe(1);
  expect(reported).toEqual([[boom, frames[0]!]]);
  expect(log.nextSequence).toBe(1);
  expect(frames).toEqual(after);
  expect(frames.map((frame) => frame.type)).toEqual(["record", "start"]);
});

test("without onSubscriberError, a subscriber's error is a yian warning, never thrown", () => {
  const { channel } = subscribed();
  channel.subscribe(() => {
    throw new TypeError("bad viewer");
  });
  const warnings: string[] = [];
  withWarnings((message) => void warnings.push(message), () => channel.start("a", 1, null, "chat", T0));
  expect(warnings).toEqual(["live channel subscriber failed and was dropped: TypeError: bad viewer"]);
});

// ---------- through a run ----------

const CONFIG = { provider: "openai", model: "gpt-test" };

/**
 * A model that answers a request whose last message's text is `key` with the deltas `replies[key]`,
 * streaming each one, then its finish, only after `release(key)` opens its gate.
 */
class GatedModel implements ChatModel {
  readonly provider = CONFIG.provider;
  readonly model = CONFIG.model;
  readonly #replies: Record<string, string[]>;
  readonly #permits = new Map<string, number>();
  readonly #waiters = new Map<string, (() => void)[]>();

  constructor(replies: Record<string, string[]>) {
    this.#replies = replies;
  }

  /** Let the call for `key` stream its next delta. */
  release(key: string): void {
    const waiter = this.#waiters.get(key)?.shift();
    if (waiter !== undefined) waiter();
    else this.#permits.set(key, (this.#permits.get(key) ?? 0) + 1);
  }

  stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    const key = request.messages.at(-1)!.text();
    const deltas = this.#replies[key]!;
    const gate = (): Promise<void> => {
      const permits = this.#permits.get(key) ?? 0;
      if (permits > 0) {
        this.#permits.set(key, permits - 1);
        return Promise.resolve();
      }
      return new Promise((resolve, reject) => {
        let waiters = this.#waiters.get(key);
        if (waiters === undefined) this.#waiters.set(key, (waiters = []));
        waiters.push(resolve);
        options.signal?.addEventListener("abort", () => reject(options.signal!.reason), { once: true });
      });
    };
    async function* events(): AsyncGenerator<Event> {
      for (const text of deltas) {
        await gate();
        yield new TextDelta(0, text);
      }
      await gate();
      yield new Finish({ reason: "stop" });
    }
    return new EventStream(events());
  }
}

/** One AGENT `write`, prompted with the run's input `topic`. */
const ONE_AGENT = `
id: live
name: live
llm_config: {provider: openai, model: gpt-test}
input:
  topic: !str
nodes:
  write:
    kind: agent
    prompt: \${topic}
    input:
      topic: \${input.topic}
    output: str
output: \${write.output}
`;

/** A MAP of an AGENT over the run's input `topics`, two at a time. */
const MAP_AGENTS = `
id: live-map
name: live_map
llm_config: {provider: openai, model: gpt-test}
input:
  topics: !list(str)
defs:
  item_flow:
    input:
      topic: !str
    nodes:
      write:
        kind: agent
        prompt: \${topic}
        input:
          topic: \${input.topic}
        output: str
    output: \${write.output}
nodes:
  each:
    kind: map
    call: item_flow
    over: \${input.topics}
    input:
      topic: \${item}
output: \${each.output}
`;

/** A flow of one CODE node, which streams nothing. */
const CODE_ONLY = `
id: plain
name: plain
nodes:
  one: {kind: code, output: int, code: return 1}
output: \${one.output}
`;

/** A fresh run log path in its own folder. */
function runLog(): string {
  return join(mkdtempSync(join(tmpdir(), "yian-live-channel-")), "log.jsonl");
}

/** Wait, a macrotask at a time, until `ready()` holds; fail after a bounded number of turns. */
async function until(ready: () => boolean): Promise<void> {
  for (let turn = 0; turn < 2000; turn++) {
    if (ready()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("timed out waiting for the run");
}

/** The step id of `node`'s step `n`. */
const key = (node: string, n: number) => `${node} ${n}`;

/** The text of each `model_call` step in the log at `path`, by step id. */
function callTexts(path: string): Map<string, string> {
  const texts = new Map<string, string>();
  for (const { envelope } of iterRunRecords(path)) {
    const { record } = envelope;
    if (record instanceof NodeStepRecord && record.step["kind"] === "model_call") {
      texts.set(key(record.nodeId, record.n), Message.fromJSON(record.step["output"]).text());
    }
  }
  return texts;
}

/** Start `flow` with `inputs` over `model` and a log, with `live` and the other options given. */
function start(
  flow: string,
  inputs: Record<string, unknown>,
  model: ChatModel,
  options: Parameters<typeof runFlow>[2] = {},
): { log: string; run: Promise<Awaited<ReturnType<typeof runFlow>>> } {
  const log = runLog();
  const run = loadFlow(flow).then((loaded) =>
    runFlow(loaded, inputs, { log, env: NO_ENV, models: serving(model), numWorkers: 2, ...options }),
  );
  return { log, run };
}

test("every appended record reaches subscribers as a record frame, in seq order, once it is on disk", async () => {
  const live = new LiveChannel();
  const model = new GatedModel({ t: ["x", "y"] });
  const seen: { seq: number; onDisk: boolean }[] = [];
  const log = runLog();
  live.subscribe((frame) => {
    if (frame.type !== "record") return;
    // The callback runs right after the append returned, so the line is in the file.
    seen.push({ seq: frame.seq, onDisk: readFileSync(log, "utf8").split("\n")[frame.seq] === frame.line });
  });
  const run = runFlow(await loadFlow(ONE_AGENT), { topic: "t" }, { log, env: NO_ENV, models: serving(model), live });
  await until(() => live.snapshot().calls.length === 1);
  for (let gate = 0; gate < 3; gate++) model.release("t");
  expect((await run).status).toBe("succeeded");
  const lines = readFileSync(log, "utf8").split("\n").filter((line) => line !== "");
  expect(seen).toEqual(lines.map((_, seq) => ({ seq, onDisk: true })));
  // `run_started`, the run inputs, the step and the outcomes are all forwarded.
  expect(lines.map((line) => decodeRecord(line).kind)).toContain("run_started");
  expect(lines.map((line) => decodeRecord(line).kind)).toContain("node_step");
  expect(live.snapshot()).toEqual({ last_seq: lines.length - 1, calls: [] });
});

test("subscribers throwing on run_started, a node_step and a chunk leave the run and its log whole", async () => {
  const reported: string[] = [];
  const live = new LiveChannel({
    onSubscriberError: (error, frame) => void reported.push(`${(error as Error).message} ${frame.type}`),
  });
  const kindOf = (frame: LiveFrame) => (frame.type === "record" ? decodeRecord(frame.line).kind : frame.type);
  const thrower = (kind: string) => (frame: LiveFrame) => {
    if (kindOf(frame) === kind) throw new Error(`on ${kind}`);
  };
  for (const kind of ["run_started", "node_step", "chunk"]) live.subscribe(thrower(kind));
  const frames: LiveFrame[] = [];
  live.subscribe((frame) => void frames.push(frame));
  const model = new GatedModel({ t: ["x", "y"] });
  const log = runLog();
  const run = runFlow(await loadFlow(ONE_AGENT), { topic: "t" }, { log, env: NO_ENV, models: serving(model), live });
  await until(() => live.snapshot().calls.length === 1);
  for (let gate = 0; gate < 3; gate++) model.release("t");
  const result = await run;
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("xy");
  expect(reported).toEqual(["on run_started record", "on chunk chunk", "on node_step record"]);
  // The log is complete, and the healthy subscriber got every record and chunk.
  const lines = readFileSync(log, "utf8").split("\n").filter((line) => line !== "");
  expect(lines.map((line) => decodeRecord(line).kind).at(-1)).toBe("node_succeeded");
  expect(frames.filter((frame) => frame.type === "record").map((frame) => frame.line)).toEqual(lines);
  expect(frames.filter((frame) => frame.type === "chunk").map((frame) => frame.text)).toEqual(["x", "y"]);
});

test("onEvent gets the same events with or without a live channel", async () => {
  const runEvents = async (live: LiveChannel | null) => {
    const events: RunEvent[] = [];
    const onEvent = (event: RunEvent) => void events.push(event);
    const result = await runFlow(await loadFlow(CODE_ONLY), {}, { log: runLog(), live, onEvent });
    expect(result.status).toBe("succeeded");
    return events.map((event) => logJson([event.type, event.toJSON()]));
  };
  const live = new LiveChannel();
  const frames: LiveFrame[] = [];
  live.subscribe((frame) => void frames.push(frame));
  expect(await runEvents(live)).toEqual(await runEvents(null));
  expect(new Set(frames.map((frame) => frame.type))).toEqual(new Set(["record"]));
});

test("the snapshot holds a streaming call's chunks so far until its model_call record lands", async () => {
  const live = new LiveChannel();
  const model = new GatedModel({ t: ["hel", "lo", "!"] });
  const frames: LiveFrame[] = [];
  live.subscribe((frame) => void frames.push(frame));
  const { log, run } = start(ONE_AGENT, { topic: "t" }, model, { live });
  await until(() => live.snapshot().calls.length === 1);
  expect(live.snapshot().calls).toEqual([{ node: "write", n: 1, chunks: [], next_index: 0 }]);
  model.release("t");
  await until(() => live.snapshot().calls[0]!.chunks.length === 1);
  model.release("t");
  await until(() => live.snapshot().calls[0]!.chunks.length === 2);
  expect(live.snapshot().calls).toEqual([{ node: "write", n: 1, chunks: ["hel", "lo"], next_index: 2 }]);
  // The record that drops the call comes before its end frame, and nothing streams after it.
  const stepAt = () =>
    frames.findIndex((frame) => frame.type === "record" && decodeRecord(frame.line).record instanceof NodeStepRecord);
  model.release("t");
  await until(() => live.snapshot().calls[0]!.chunks.length === 3);
  // After its last chunk, the call stays until its record lands.
  expect(live.snapshot().calls).toEqual([{ node: "write", n: 1, chunks: ["hel", "lo", "!"], next_index: 3 }]);
  model.release("t");
  expect((await run).status).toBe("succeeded");
  expect(live.snapshot().calls).toEqual([]);
  const after = frames.slice(stepAt()).filter((frame) => frame.type !== "record");
  expect(after).toEqual([{ type: "end", node: "write", n: 1, error: null }]);
  const started = frames.find((frame) => frame.type === "start");
  expect(started).toMatchObject({ node: "write", n: 1, model: "gpt-test", purpose: "chat" });
  expect(callTexts(log)).toEqual(new Map([[key("write", 1), "hello!"]]));
});

test("a call a host stop cuts leaves the snapshot when its node settles", async () => {
  const live = new LiveChannel();
  const model = new GatedModel({ t: ["cut", "never"] });
  const stop = new AbortController();
  const { log, run } = start(ONE_AGENT, { topic: "t" }, model, { live, signal: stop.signal });
  model.release("t");
  await until(() => live.snapshot().calls[0]?.chunks.length === 1);
  stop.abort(new Error("stopped"));
  expect(((await rejected(run)) as Error).message).toBe("stopped");
  expect(live.snapshot().calls).toEqual([]);
  // A stop writes no model_call.
  expect(callTexts(log).size).toBe(0);
});

test("two parallel streaming AGENTs show at once; each call leaves only when its own record lands", async () => {
  const live = new LiveChannel();
  const model = new GatedModel({ a: ["a0", "a1"], b: ["b0", "b1", "b2"] });
  const { log, run } = start(MAP_AGENTS, { topics: ["a", "b"] }, model, { live });
  await until(() => live.snapshot().calls.length === 2);
  const byText = (snapshot: LiveSnapshot) =>
    Object.fromEntries(snapshot.calls.map((call) => [call.chunks.join(""), call.n]));
  model.release("a");
  model.release("b");
  await until(() => live.snapshot().calls.every((call) => call.chunks.length === 1));
  expect(byText(live.snapshot())).toEqual({ a0: 1, b0: 1 });
  const nodes = live.snapshot().calls.map((call) => call.node);
  expect(new Set(nodes).size).toBe(2);
  // `a`'s record ends its call; `b`'s stays with its chunks.
  model.release("a");
  model.release("a");
  await until(() => live.snapshot().calls.length === 1);
  expect(live.snapshot().calls.map((call) => call.chunks)).toEqual([["b0"]]);
  for (let gate = 0; gate < 3; gate++) model.release("b");
  expect((await run).status).toBe("succeeded");
  expect(live.snapshot().calls).toEqual([]);
  expect([...callTexts(log).values()].sort()).toEqual(["a0a1", "b0b1b2"]);
});

test("a resumed run's channel starts after the records the log holds and forwards only new ones", async () => {
  const log = runLog();
  const flow = await loadFlow(ONE_AGENT);
  // Cut the run after its seventh record, before the AGENT calls its model.
  await crashAfter(7, () => runFlow(flow, { topic: "t" }, { log, env: NO_ENV, models: serving(new GatedModel({})) }));
  const held = readFileSync(log, "utf8").split("\n").filter((line) => line !== "").length;
  expect(held).toBe(7);
  const live = new LiveChannel();
  const seqs: number[] = [];
  live.subscribe((frame) => void (frame.type === "record" && seqs.push(frame.seq)));
  const model = new GatedModel({ t: ["again"] });
  const resumed = resumeFlow({ log, env: NO_ENV, models: serving(model), live });
  await until(() => live.snapshot().calls.length === 1);
  expect(seqs[0]).toBe(held);
  model.release("t");
  model.release("t");
  expect((await resumed).status).toBe("succeeded");
  const lines = readFileSync(log, "utf8").split("\n").filter((line) => line !== "");
  expect(seqs).toEqual(lines.slice(held).map((_, index) => held + index));
  expect(live.snapshot().last_seq).toBe(lines.length - 1);
});

// ---------- a late viewer ----------

/**
 * A viewer that joined a run: the log up to the snapshot's seq, the snapshot, then frames. It
 * rebuilds each call's text from the snapshot and later chunks, and checks it against the call's
 * `model_call` record when that arrives.
 */
class Viewer {
  lastSeq: number;
  readonly live = new Map<string, { text: string; next: number }>();
  /** Each call followed live, by step id: its rebuilt text and its record's text. */
  readonly compared = new Map<string, [string, string]>();

  constructor(snapshot: LiveSnapshot) {
    this.lastSeq = snapshot.last_seq;
    for (const call of snapshot.calls) {
      this.live.set(key(call.node, call.n), { text: call.chunks.join(""), next: call.next_index });
    }
  }

  apply(frame: LiveFrame): void {
    if (frame.type === "record") {
      if (frame.seq <= this.lastSeq) return;
      expect(frame.seq).toBe(this.lastSeq + 1);
      this.lastSeq = frame.seq;
      const { record } = decodeRecord(frame.line);
      if (!(record instanceof NodeStepRecord) || record.step["kind"] !== "model_call") return;
      const id = key(record.nodeId, record.n);
      const call = this.live.get(id);
      if (call === undefined) return;
      this.compared.set(id, [call.text, Message.fromJSON(record.step["output"]).text()]);
      this.live.delete(id);
    } else if (frame.type === "start") {
      this.live.set(key(frame.node, frame.n), { text: "", next: 0 });
    } else if (frame.type === "chunk") {
      const call = this.live.get(key(frame.node, frame.n))!;
      if (frame.index < call.next) return;
      expect(frame.index).toBe(call.next);
      call.text += frame.text;
      call.next = frame.index + 1;
    } else if (frame.error !== null) {
      this.live.delete(key(frame.node, frame.n));
    }
  }
}

/**
 * Run `flow` over `replies`, opening gates in `order` (the reply keys; each call's chunks, then its
 * finish), and have a viewer join after `joinAt` of them; return the viewer once the run ends, with
 * the log's call texts.
 */
async function lateViewer(
  flow: string,
  inputs: Record<string, unknown>,
  replies: Record<string, string[]>,
  order: string[],
  joinAt: number,
): Promise<{ viewer: Viewer; texts: Map<string, string>; lastSeq: number }> {
  const live = new LiveChannel();
  const model = new GatedModel(replies);
  const calls = Object.keys(replies).length;
  let frames = 0;
  live.subscribe(() => void frames++);
  const { log, run } = start(flow, inputs, model, { live });
  await until(() => live.snapshot().calls.length === calls);
  let viewer: Viewer | null = null;
  for (let released = 0; released <= order.length; released++) {
    if (released === joinAt) {
      // 1. subscribe; 2. read the log to its end; 3. take the snapshot; 4. follow.
      const pending: LiveFrame[] = [];
      live.subscribe((frame) => (viewer === null ? pending.push(frame) : viewer.apply(frame)));
      let read = -1;
      for (const { envelope } of iterRunRecords(log)) read = envelope.seq;
      const joined = new Viewer(live.snapshot());
      expect(joined.lastSeq).toBe(read);
      for (const frame of pending) joined.apply(frame);
      viewer = joined;
    }
    if (released === order.length) break;
    const before = frames;
    model.release(order[released]!);
    // A chunk, or the call's record, arrives.
    await until(() => frames > before);
  }
  expect((await run).status).toBe("succeeded");
  const texts = callTexts(log);
  const lastSeq = readFileSync(log, "utf8").split("\n").filter((line) => line !== "").length - 1;
  return { viewer: viewer!, texts, lastSeq };
}

test("a late viewer rebuilds one call's text exactly, joining at every chunk boundary", async () => {
  const replies = { t: ["he", "ll", "o"] };
  const order = ["t", "t", "t", "t"];
  for (let joinAt = 0; joinAt < order.length; joinAt++) {
    const { viewer, texts, lastSeq } = await lateViewer(ONE_AGENT, { topic: "t" }, replies, order, joinAt);
    expect([...viewer.compared.values()]).toEqual([["hello", "hello"]]);
    expect([...texts.values()]).toEqual(["hello"]);
    expect(viewer.lastSeq).toBe(lastSeq);
    expect(viewer.live.size).toBe(0);
  }
});

test("a late viewer rebuilds two parallel calls' texts exactly, joining at every chunk boundary", async () => {
  const replies = { a: ["a0", "a1", "a2"], b: ["b0", "b1"] };
  // Each call's chunks, then its finish; `b` finishes while `a` still streams.
  const order = ["a", "b", "a", "b", "b", "a", "a"];
  for (let joinAt = 0; joinAt < order.length; joinAt++) {
    const { viewer, texts, lastSeq } = await lateViewer(MAP_AGENTS, { topics: ["a", "b"] }, replies, order, joinAt);
    // A call that finished before the viewer joined is in the log it read; every other one is
    // followed from the snapshot, text for text.
    const finishedBefore = joinAt > 4 ? 1 : 0;
    expect(viewer.compared.size).toBe(2 - finishedBefore);
    for (const [id, [rebuilt, recorded]] of viewer.compared) {
      expect(rebuilt).toBe(recorded);
      expect(texts.get(id)).toBe(recorded);
    }
    expect(viewer.lastSeq).toBe(lastSeq);
    expect(viewer.live.size).toBe(0);
  }
});
