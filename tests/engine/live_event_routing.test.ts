/**
 * The engine passes in-loop observers to `onEvent` between a node's start and its outcome: a port
 * of `tests/engine/test_live_event_routing.py`. Serial and pooled runs deliver them in node order,
 * leave them out of the returned observers, and supply no sink when the host set no callback.
 *
 * Ported: `test_in_loop_observers_reach_on_event_between_start_and_outcome`,
 * `test_no_callbacks_means_no_sinks` and `test_event_callback_alone_supplies_only_the_event_sink`,
 * each serial and with two workers. The in-loop observers are stand-in arrays passed by cast, as
 * Python passes tuples.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeStarted, NodeSucceeded, StreamChunk } from "../../src/events.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import type { FlowEngine, FlowEngineOptions } from "../../src/runtime/engine.ts";
import { engine, graph } from "./run_state_fixtures.ts";

/** Publishes one chunk and two in-loop observers through whichever sinks exist. */
class Live extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsLiveEvents = true;
  /** The sink names the node received. */
  seen: string[] = [];

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    this.seen = Object.keys(caps)
      .filter((key) => key.startsWith("emit"))
      .sort();
    caps.emitStream?.(new StreamChunk({ nodeId: this.id, key: "text", chunk: "hi" }));
    caps.emitEvent?.(["in-loop", 1] as never);
    caps.emitEvent?.(["in-loop", 2] as never);
    return new Output("done");
  }
}

/** Return an engine for `START -> node -> END`; the other options go to the fixture. */
function single(node: Node, numWorkers: number, options: Partial<FlowEngineOptions> = {}): FlowEngine {
  return engine(graph([node], [
    [START_ID, node.id],
    [node.id, END_ID],
  ]), { numWorkers, ...options });
}

test.each([0, 2])("in loop observers reach onEvent between start and outcome (numWorkers %p)", async (numWorkers) => {
  const timeline: unknown[] = [];
  const built = single(new Live("work"), numWorkers, {
    onEvent: (event) => timeline.push(event),
    onStream: (chunk) => timeline.push(chunk),
  });

  const returned = await built.run();

  const work = timeline.filter(
    (item) => Array.isArray(item) || (item as { nodeId?: string }).nodeId === "work",
  );
  expect(work[0]).toBeInstanceOf(NodeStarted);
  expect(work.at(-1)).toBeInstanceOf(NodeSucceeded);
  expect(work.slice(1, -1)).toEqual([
    new StreamChunk({ nodeId: "work", key: "text", chunk: "hi" }),
    ["in-loop", 1],
    ["in-loop", 2],
  ]);
  expect(returned.some((event) => Array.isArray(event))).toBe(false);
});

test.each([0, 2])("no callbacks means no sinks (numWorkers %p)", async (numWorkers) => {
  const node = new Live("work");
  await single(node, numWorkers).run();
  expect(node.seen).toEqual([]);
});

test.each([0, 2])("event callback alone supplies only the event sink (numWorkers %p)", async (numWorkers) => {
  const node = new Live("work");
  await single(node, numWorkers, { onEvent: () => {} }).run();
  expect(node.seen).toEqual(["emitEvent"]);
});
