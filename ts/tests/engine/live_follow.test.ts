/**
 * TS-only: following a live run from another process. `LiveChannel.follow` hands a viewer the
 * snapshot, then every later frame, in one call; `LiveView` folds the log's records, that snapshot
 * and the frames into each call's text. A two-process test shows they compose over a byte pipe.
 */

import { expect, test } from "bun:test";

import { LiveChannel, type LiveFollowFrame } from "../../src/runtime/live.ts";
import { encodeRecord, MemoryRunLog, NodeStartedRecord, NodeStepRecord } from "../../src/runtime/log.ts";
import { logJson } from "../../src/typesys/json.ts";

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
