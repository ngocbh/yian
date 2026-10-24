/**
 * TS-only: the live channel of a run. `LiveChannel` forwards each appended record and each AGENT
 * call's start, chunks and end as plain-JSON frames named by the step they become (`node` + `n`),
 * and its `snapshot()` lists the calls in flight with their chunks so far, one buffer per call, so a
 * late viewer rebuilds every call's text from the log, the snapshot and the frames after it.
 *
 * First on a bare channel, then through runs with a scripted model that streams one chunk per gate.
 */

import { expect, test } from "bun:test";

import { LiveChannel, type LiveFrame } from "../../src/runtime/live.ts";
import { encodeRecord, NodeStartedRecord, NodeStepRecord, RecordEnvelope, SCHEMA_VERSION } from "../../src/runtime/log.ts";
import { logJson } from "../../src/typesys/json.ts";

/** The text of record `seq` holding `record`, as a log appends it. */
function line(seq: number, record: NodeStepRecord | NodeStartedRecord): string {
  return encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "run-1", seq, record, "2026-10-23T10:00:00.000Z"));
}

/** A `model_call` step record of `node`'s step `n`. */
function step(node: string, n: number): NodeStepRecord {
  return new NodeStepRecord(node, n, { kind: "model_call" });
}

/** A channel and the frames its subscriber got. */
function subscribed(): { channel: LiveChannel; frames: LiveFrame[]; stop: () => void } {
  const channel = new LiveChannel();
  const frames: LiveFrame[] = [];
  const stop = channel.subscribe((frame) => void frames.push(frame));
  return { channel, frames, stop };
}

const T0 = "2026-10-23T10:00:00.000Z";

test("frames reach subscribers in call order and survive a JSON round trip", () => {
  const { channel, frames } = subscribed();
  const started = line(0, new NodeStartedRecord("a"));
  channel.record(0, started);
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
  const { channel } = subscribed();
  expect(channel.snapshot()).toEqual({ last_seq: -1, calls: [] });
  channel.record(0, line(0, new NodeStartedRecord("a")));
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
  const { channel } = subscribed();
  channel.start("a", 1, null, "chat", T0);
  channel.start("b", 1, null, "chat", T0);
  channel.chunk("a", 1, 0, "a0");
  channel.chunk("b", 1, 0, "b0");
  // Another node's step with the same n, and a record that isn't a step, drop nothing.
  channel.record(3, line(3, step("c", 1)));
  channel.record(4, line(4, new NodeStartedRecord("a")));
  expect(channel.snapshot().calls.map((call) => call.node)).toEqual(["a", "b"]);
  channel.record(5, line(5, step("a", 1)));
  expect(channel.snapshot()).toEqual({ last_seq: 5, calls: [{ node: "b", n: 1, chunks: ["b0"], next_index: 1 }] });
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
