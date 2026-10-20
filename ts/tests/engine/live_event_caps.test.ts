/**
 * The evaluator supplies in-loop observer sinks only to opted-in nodes: a port of
 * `tests/engine/test_live_event_caps.py`.
 *
 * `needsLiveEvents` gates `caps.emitStream` and `caps.emitEvent`; each is present only when the
 * host supplied that sink, and a failing host sink escapes as the original host error rather than
 * a node failure.
 *
 * Ported: `test_node_without_trait_receives_no_sinks`,
 * `test_opted_in_node_receives_only_supplied_sinks` (4 rows),
 * `test_failing_host_sink_propagates_original_exception` (2 rows) and
 * `test_emit_event_must_be_callable`.
 *
 * Changed: every node also receives `signal`, so the capability names the probe sees include it.
 *
 * TS-only: a sink that throws a `CapabilityFault` rejects `evalNode` with that fault's original,
 * not with a fault wrapping the fault.
 */

import { expect, test } from "bun:test";

import { StreamChunk } from "../../src/events.ts";
import { CapabilityFault, Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { drive, rejected } from "./run_state_fixtures.ts";

/** Record the received capability names and publish through whichever sinks exist. */
class Probe extends Node {
  readonly kind = NodeKind.CODE;
  seen: string[] = [];

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    this.seen = Object.keys(caps).sort();
    caps.emitStream?.(new StreamChunk({ nodeId: this.id, key: "text", chunk: "hi" }));
    caps.emitEvent?.("in-loop" as never);
    return new Output("done");
  }
}

/** The same probe with the live-events trait. */
class LiveProbe extends Probe {
  override readonly needsLiveEvents = true;
}

test("node without trait receives no sinks", async () => {
  const node = new Probe("probe");
  const chunks: unknown[] = [];
  const events: unknown[] = [];

  expect(
    await drive(node, { emitStream: (chunk) => chunks.push(chunk), emitEvent: (event) => events.push(event) }),
  ).toEqual(new Output("done"));
  expect([node.seen, chunks, events]).toEqual([["signal"], [], []]);
});

for (const [stream, event, expected] of [
  [true, true, ["emitEvent", "emitStream", "signal"]],
  [true, false, ["emitStream", "signal"]],
  [false, true, ["emitEvent", "signal"]],
  [false, false, ["signal"]],
] as const) {
  test(`opted in node receives only supplied sinks (stream ${stream}, event ${event})`, async () => {
    const node = new LiveProbe("probe");
    const chunks: unknown[] = [];
    const events: unknown[] = [];

    await drive(node, {
      emitStream: stream ? (chunk) => chunks.push(chunk) : null,
      emitEvent: event ? (item) => events.push(item) : null,
    });

    expect(node.seen).toEqual([...expected]);
    expect(chunks).toEqual(stream ? [new StreamChunk({ nodeId: "probe", key: "text", chunk: "hi" })] : []);
    expect(events).toEqual(event ? ["in-loop"] : []);
  });
}

for (const sink of ["emitStream", "emitEvent"] as const) {
  test(`failing host sink propagates original exception (${sink})`, async () => {
    class HostBroken extends Error {}
    const expected = new HostBroken("host");

    const error = await rejected(
      drive(new LiveProbe("probe"), {
        [sink]: () => {
          throw expected;
        },
      }),
    );

    expect(error).toBe(expected);
  });

  test(`a sink throwing a capability fault rejects with its original (${sink})`, async () => {
    const original = new Error("inner");

    const error = await rejected(
      drive(new LiveProbe("probe"), {
        [sink]: () => {
          throw new CapabilityFault(original);
        },
      }),
    );

    expect(error).toBe(original);
  });
}

test("emit event must be callable", async () => {
  expect(await rejected(drive(new LiveProbe("probe"), { emitEvent: "not callable" as never }))).toEqual(
    new TypeError("emit_event must be callable or null"),
  );
});
