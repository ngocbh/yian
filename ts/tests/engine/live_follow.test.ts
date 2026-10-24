/**
 * TS-only: following a live run from another process. `LiveChannel.follow` hands a viewer the
 * snapshot, then every later frame, in one call; `LiveView` folds the log's records, that snapshot
 * and the frames into each call's text. A two-process test shows they compose over a byte pipe.
 */

import { readFileSync } from "node:fs";

import { expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { Failure } from "../../src/diagnostics.ts";
import { LiveChannel, type LiveFollowFrame, type LiveFrame, type LiveSnapshot, LiveView } from "../../src/runtime/live.ts";
import {
  decodeRecord,
  encodeRecord,
  iterRunRecords,
  MemoryRunLog,
  NodeFailedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  type RunRecord,
  RunStoppedRecord,
} from "../../src/runtime/log.ts";
import { runFlow } from "../../src/run.ts";
import { logJson } from "../../src/typesys/json.ts";
import { callTexts, GatedModel, key, MAP_AGENTS, ONE_AGENT, runLog, until } from "./live_fixtures.ts";
import { NO_ENV, serving } from "./run_state_fixtures.ts";

const T0 = "2026-10-23T10:00:00.000Z";

/** A channel over a process-local log, and `append`, which appends a record and returns its line. */
function channelWithLog(options?: ConstructorParameters<typeof LiveChannel>[0]) {
  const channel = new LiveChannel(options);
  const log = channel.wrap(new MemoryRunLog("run-1", { clock: () => Date.parse(T0) }));
  const append = (record: NodeStepRecord | NodeStartedRecord) => encodeRecord(log.append(record));
  return { channel, append };
}

// ---------- follow ----------

test("follow sends the snapshot first, then each later frame once", () => {
  const { channel, append } = channelWithLog();
  append(new NodeStartedRecord("a"));
  channel.start("a", 1, null, "chat", T0);
  channel.chunk("a", 1, 0, "before");
  const frames: LiveFollowFrame[] = [];
  const stop = channel.follow((frame) => void frames.push(frame));
  // What happened before is only in the snapshot.
  expect(frames).toEqual([
    { type: "snapshot", snapshot: { last_seq: 0, calls: [{ node: "a", n: 1, chunks: ["before"], next_index: 1 }] } },
  ]);
  channel.chunk("a", 1, 1, "after");
  const line = append(new NodeStepRecord("a", 1, { kind: "model_call" }));
  stop();
  channel.end("a", 1, null);
  expect(frames.slice(1)).toEqual([
    { type: "chunk", node: "a", n: 1, index: 1, text: "after" },
    { type: "record", seq: 1, line },
  ]);
  for (const frame of frames) expect(JSON.parse(logJson(frame))).toEqual(frame);
});

test("for two interleaved calls, the snapshot and the later frames rebuild both texts exactly", () => {
  const { channel } = channelWithLog();
  channel.start("a", 1, null, "chat", T0);
  channel.start("b", 1, null, "chat", T0);
  channel.chunk("a", 1, 0, "a0");
  channel.chunk("b", 1, 0, "b0");
  const texts = new Map<string, string>();
  channel.follow((frame) => {
    if (frame.type === "snapshot") {
      for (const call of frame.snapshot.calls) texts.set(call.node, call.chunks.join(""));
    } else if (frame.type === "chunk") {
      texts.set(frame.node, texts.get(frame.node)! + frame.text);
    }
  });
  channel.chunk("b", 1, 1, "b1");
  channel.chunk("a", 1, 1, "a1");
  channel.chunk("a", 1, 2, "a2");
  expect(Object.fromEntries(texts)).toEqual({ a: "a0a1a2", b: "b0b1" });
});

test("a follower that throws on the snapshot is dropped and reported, and gets nothing more", () => {
  const reported: [unknown, LiveFollowFrame][] = [];
  const { channel } = channelWithLog({ onSubscriberError: (error, frame) => void reported.push([error, frame]) });
  const boom = new Error("viewer gone");
  let calls = 0;
  channel.follow(() => {
    calls++;
    throw boom;
  });
  channel.start("a", 1, null, "chat", T0);
  expect(calls).toBe(1);
  expect(reported).toEqual([[boom, { type: "snapshot", snapshot: { last_seq: -1, calls: [] } }]]);
});

// ---------- LiveView on frames ----------

/** A log of `run-1` and `frame`, which appends a record and returns its record frame. */
function recording() {
  const log = new MemoryRunLog("run-1", { clock: () => Date.parse(T0) });
  const frame = (record: RunRecord): LiveFrame => {
    const envelope = log.append(record);
    return { type: "record", seq: envelope.seq, line: encodeRecord(envelope) };
  };
  return { log, frame };
}

/** The frame of chunk `index` of `node`'s step `n`. */
const chunk = (node: string, n: number, index: number, text: string): LiveFrame => ({
  type: "chunk",
  node,
  n,
  index,
  text,
});

/** The start frame of `node`'s step `n`. */
const start = (node: string, n: number): LiveFrame => ({
  type: "start",
  node,
  n,
  model: null,
  purpose: "chat",
  started_at: T0,
});

test("a view starts from the log's records up to the snapshot's seq and the snapshot's calls", () => {
  const { log, frame } = recording();
  frame(new NodeStartedRecord("a"));
  frame(new NodeStartedRecord("b"));
  const snapshot: LiveSnapshot = { last_seq: 1, calls: [{ node: "a", n: 1, chunks: ["he", "l"], next_index: 2 }] };
  const view = LiveView.from(log.records(), snapshot);
  expect(view.lastSeq).toBe(1);
  expect(view.calls()).toEqual([{ node: "a", n: 1, text: "hel", nextIndex: 2 }]);
  // Records past the snapshot's seq are left to the frames that bring them again.
  frame(new NodeStepRecord("a", 1, { kind: "model_call" }));
  expect(LiveView.from(log.records(), snapshot).lastSeq).toBe(1);
  expect(LiveView.from([], { last_seq: -1, calls: [] }).lastSeq).toBe(-1);
});

test("a view refuses records that don't run from 0 to the snapshot's seq", () => {
  const { log, frame } = recording();
  frame(new NodeStartedRecord("a"));
  frame(new NodeStartedRecord("b"));
  const records = [...log.records()];
  expect(() => LiveView.from(records.slice(0, 1), { last_seq: 1, calls: [] })).toThrow(RangeError);
  expect(() => LiveView.from(records.slice(1), { last_seq: 1, calls: [] })).toThrow(RangeError);
});

test("a view folds frames into each call's text, one buffer per call", () => {
  const { frame } = recording();
  const view = LiveView.from([], { last_seq: -1, calls: [] });
  for (const next of [start("a", 1), start("b", 1), chunk("a", 1, 0, "a0"), chunk("b", 1, 0, "b0")]) {
    expect(view.apply(next)).toBe(true);
  }
  expect(view.apply(chunk("a", 1, 1, "a1"))).toBe(true);
  expect(view.calls()).toEqual([
    { node: "a", n: 1, text: "a0a1", nextIndex: 2 },
    { node: "b", n: 1, text: "b0", nextIndex: 1 },
  ]);
  // A call's step record drops only that call; another node's step with the same n drops nothing.
  expect(view.apply(frame(new NodeStepRecord("c", 1, { kind: "model_call" })))).toBe(true);
  expect(view.calls().length).toBe(2);
  expect(view.apply(frame(new NodeStepRecord("a", 1, { kind: "model_call" })))).toBe(true);
  expect(view.apply({ type: "end", node: "a", n: 1, error: null })).toBe(true);
  expect(view.calls()).toEqual([{ node: "b", n: 1, text: "b0", nextIndex: 1 }]);
  expect(view.lastSeq).toBe(1);
  // An end with an error drops its call.
  expect(view.apply({ type: "end", node: "b", n: 1, error: "ProviderError" })).toBe(true);
  expect(view.calls()).toEqual([]);
});

test("a duplicated chunk, start or record changes nothing", () => {
  const { frame } = recording();
  const view = LiveView.from([], { last_seq: -1, calls: [] });
  const started = frame(new NodeStartedRecord("a"));
  for (const next of [started, start("a", 1), chunk("a", 1, 0, "x"), chunk("a", 1, 1, "y")]) {
    expect(view.apply(next)).toBe(true);
    expect(view.apply(next)).toBe(true);
  }
  expect(view.apply(chunk("a", 1, 0, "x"))).toBe(true);
  expect(view.apply(started)).toBe(true);
  expect(view.lastSeq).toBe(0);
  expect(view.calls()).toEqual([{ node: "a", n: 1, text: "xy", nextIndex: 2 }]);
});

test("a gap in records or chunks returns false and changes nothing", () => {
  const { frame } = recording();
  const view = LiveView.from([], { last_seq: -1, calls: [{ node: "a", n: 1, chunks: ["x"], next_index: 1 }] });
  frame(new NodeStartedRecord("a"));
  // Seq 0 never arrived.
  expect(view.apply(frame(new NodeStepRecord("a", 1, { kind: "model_call" })))).toBe(false);
  expect(view.lastSeq).toBe(-1);
  // Chunk 1 never arrived.
  expect(view.apply(chunk("a", 1, 2, "z"))).toBe(false);
  // The start of `b` never arrived.
  expect(view.apply(chunk("b", 1, 0, "b0"))).toBe(false);
  expect(view.calls()).toEqual([{ node: "a", n: 1, text: "x", nextIndex: 1 }]);
});

test("a node's success drops its calls; a node's failure or a run stop drops every call", () => {
  const failed = new NodeFailedRecord(new Failure({ code: "node.execute", message: "boom", nodeId: "a" }));
  for (const [ending, left] of [
    [new NodeSucceededRecord("a", 1), ["b"]],
    [failed, []],
    [new RunStoppedRecord("signal", "stopped", ["a", "b"]), []],
  ] as const) {
    const { frame } = recording();
    const view = LiveView.from([], { last_seq: -1, calls: [] });
    for (const next of [start("a", 1), start("a", 2), start("b", 1)]) view.apply(next);
    expect(view.apply(frame(ending))).toBe(true);
    expect(view.calls().map((call) => call.node)).toEqual([...left]);
  }
});

// ---------- LiveView following a run ----------

/** What a viewer that joined a run through `follow` saw. */
interface Followed {
  view: LiveView;
  /** Each call the view saw settle, by step id: its text in the view just before its record. */
  settled: Map<string, string>;
  /** Each call's `model_call` output text in the log, by step id. */
  texts: Map<string, string>;
  /** The log's last seq once the run ended. */
  lastSeq: number;
}

/**
 * Run `flow` over `replies`, opening gates in `order` (the reply keys; each call's chunks, then its
 * finish). After `joinAt` gates a viewer calls `follow`; `gap` more gates open while it reads the
 * whole log (the frames meanwhile buffered); then it builds a `LiveView`, applies the buffered
 * frames, and follows to the end, applying each frame twice. `drop` skips that frame after the
 * view is built and stops at the first `false`.
 */
async function followLate(
  flow: string,
  inputs: Record<string, unknown>,
  replies: Record<string, string[]>,
  order: string[],
  joinAt: number,
  options: { gap?: number; drop?: (frame: LiveFrame) => boolean } = {},
): Promise<Followed & { gapAt: LiveFrame | null }> {
  const { gap = 2, drop = () => false } = options;
  const live = new LiveChannel();
  const model = new GatedModel(replies);
  let sent = 0;
  live.subscribe(() => void sent++);
  const log = runLog();
  const run = runFlow(await loadFlow(flow), inputs, { log, env: NO_ENV, models: serving(model), numWorkers: 2, live });
  await until(() => live.snapshot().calls.length === Object.keys(replies).length);
  const open = async (gate: number) => {
    const before = sent;
    model.release(order[gate]!);
    // A chunk, or the call's record, arrives.
    await until(() => sent > before);
  };
  let released = 0;
  for (; released < joinAt; released++) await open(released);
  const settled = new Map<string, string>();
  let gapAt: LiveFrame | null = null;
  let view: LiveView | null = null;
  const apply = (frame: LiveFrame) => {
    if (gapAt !== null || drop(frame)) return;
    if (frame.type === "record") {
      const { record } = decodeRecord(frame.line);
      const call = view!.calls().find((c) => record instanceof NodeStepRecord && c.node === record.nodeId && c.n === record.n);
      if (call !== undefined && frame.seq > view!.lastSeq) settled.set(key(call.node, call.n), call.text);
    }
    if (!view!.apply(frame)) {
      gapAt = frame;
      return;
    }
    // A duplicate changes nothing.
    const before = logJson([view!.lastSeq, view!.calls()]);
    expect(view!.apply(frame)).toBe(true);
    expect(logJson([view!.lastSeq, view!.calls()])).toBe(before);
  };
  let snapshot: LiveSnapshot | null = null;
  const buffered: LiveFrame[] = [];
  live.follow((frame) => {
    if (frame.type === "snapshot") snapshot = frame.snapshot;
    else if (view === null) buffered.push(frame);
    else apply(frame);
  });
  for (const end = Math.min(order.length, released + gap); released < end; released++) await open(released);
  view = LiveView.from(iterRunRecords(log), snapshot!);
  for (const frame of buffered) apply(frame);
  for (; released < order.length; released++) await open(released);
  expect((await run).status).toBe("succeeded");
  const lastSeq = readFileSync(log, "utf8").split("\n").filter((line) => line !== "").length - 1;
  return { view, settled, texts: callTexts(log), lastSeq, gapAt };
}

/** Check that the view rebuilt the `followed` calls' recorded texts and reached the log's end. */
function expectRebuilt(result: Followed & { gapAt: LiveFrame | null }, followed: number): void {
  const { view, settled, texts, lastSeq, gapAt } = result;
  expect(gapAt).toBeNull();
  expect(settled.size).toBe(followed);
  for (const [id, text] of settled) expect(text).toBe(texts.get(id)!);
  expect(view.lastSeq).toBe(lastSeq);
  expect(view.calls()).toEqual([]);
}

test("a view built through follow rebuilds one call's text, joining at every chunk boundary", async () => {
  const replies = { t: ["he", "ll", "o"] };
  const order = ["t", "t", "t", "t"];
  for (let joinAt = 0; joinAt < order.length; joinAt++) {
    expectRebuilt(await followLate(ONE_AGENT, { topic: "t" }, replies, order, joinAt), 1);
  }
});

test("a view built through follow rebuilds two parallel calls' texts, joining at every chunk boundary", async () => {
  const replies = { a: ["a0", "a1", "a2"], b: ["b0", "b1"] };
  // Each call's chunks, then its finish; `b` finishes while `a` still streams.
  const order = ["a", "b", "a", "b", "b", "a", "a"];
  for (let joinAt = 0; joinAt < order.length; joinAt++) {
    for (const gap of [0, 2]) {
      // A call whose record came before the snapshot is in the log the viewer read.
      const result = await followLate(MAP_AGENTS, { topics: ["a", "b"] }, replies, order, joinAt, { gap });
      expectRebuilt(result, joinAt > 4 ? 1 : 2);
    }
  }
});

test("a dropped chunk makes the view report a gap", async () => {
  const replies = { t: ["he", "ll", "o"] };
  const result = await followLate(ONE_AGENT, { topic: "t" }, replies, ["t", "t", "t", "t"], 1, {
    gap: 0,
    drop: (frame) => frame.type === "chunk" && frame.index === 1,
  });
  expect(result.gapAt).toMatchObject({ type: "chunk", index: 2, text: "o" });
});
