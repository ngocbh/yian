/** The durable steps a node records. */

import { describe, expect, test } from "bun:test";

import { copyGraph, END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeStarted, NodeSucceeded, type RunEvent, RunFailed, RunSucceeded, StreamChunk } from "../../src/events.ts";
import { CapabilityFault, Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { evalNode } from "../../src/runtime/eval_node.ts";
import {
  NodeFailedRecord,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  PoolCommitRecord,
  type RunRecord,
} from "../../src/runtime/log.ts";
import { commitPrepared, prepareRunResumed, replayRecords, RunState } from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import type { ValueRecord } from "../../src/typesys/values.ts";
import { aborted, engine, graph, noGrowth, rejected } from "./run_state_fixtures.ts";

/** A node that opts into steps and runs a scripted body over its capabilities; it may stream. */
class StepNode extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsSteps: boolean = true;
  override readonly needsLiveEvents = true;
  readonly body: (caps: NodeCaps) => NodeResult | Promise<NodeResult>;

  constructor(nodeId: string, body: (caps: NodeCaps) => NodeResult | Promise<NodeResult>) {
    super(nodeId);
    this.body = body;
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult | Promise<NodeResult> {
    return this.body(caps);
  }
}

/** The same scripted node without the steps trait. */
class PlainNode extends StepNode {
  override readonly needsSteps = false;
}

test("NodeSteps checks its fields", () => {
  const record = (): number => 1;

  expect(() => new NodeSteps(true as never, [], record)).toThrow(TypeError);
  expect(() => new NodeSteps(0, [], record)).toThrow(RangeError);
  expect(() => new NodeSteps(1, ["x"] as never, record)).toThrow(TypeError);
  expect(() => new NodeSteps(1, ["x"] as never, record)).toThrow(/^prior must be an array of records$/);
  expect(() => new NodeSteps(1, [], null as never)).toThrow(TypeError);
});

test("NodeSteps.record returns the n its recorder returns", () => {
  const steps = new NodeSteps(1, [], () => 7);
  expect(steps.record({ kind: "test" })).toBe(7);
});

test("eval node passes steps only to opted in nodes and checks the type", async () => {
  const steps = new NodeSteps(1, [], () => 1);
  const seen: unknown[] = [];
  const body = (caps: NodeCaps): NodeResult => {
    seen.push(caps.steps);
    return new Output("done");
  };
  const pool = new VariablePool();

  await evalNode(new StepNode("a", body), null, pool, { steps });
  await evalNode(new PlainNode("b", body), null, pool, { steps });
  await evalNode(new StepNode("c", body), null, pool);

  const given = seen[0] as NodeSteps;
  expect(given).toBeInstanceOf(NodeSteps);
  expect([given.starts, given.prior]).toEqual([1, []]);
  expect(seen.slice(1)).toEqual([undefined, undefined]);
  const recorded: ValueRecord[] = [];
  const recording = new NodeSteps(2, [{ kind: "test", attempt: 1n }], (step) => recorded.push(step));
  const recordingBody = (caps: NodeCaps): NodeResult => {
    seen.push(caps.steps);
    caps.steps!.record({ kind: "test", attempt: 2n });
    return new Output("done");
  };
  await evalNode(new StepNode("e", recordingBody), null, pool, { steps: recording });
  const passed = seen.at(-1) as NodeSteps;
  expect([passed.starts, passed.prior]).toEqual([2, [{ kind: "test", attempt: 1n }]]);
  expect(recorded).toEqual([{ kind: "test", attempt: 2n }]);
  await expect(evalNode(new StepNode("d", body), null, pool, { steps: { starts: 1 } as never })).rejects.toThrow(
    new TypeError("steps must be a NodeSteps or null"),
  );
});

test("a node that catches the recorder's fault still rejects evalNode with its original", async () => {
  const fault = new Error("disk full");
  const steps = new NodeSteps(1, [], () => {
    throw new CapabilityFault(fault);
  });
  const node = new StepNode("a", (caps) => {
    try {
      caps.steps!.record({ kind: "test", event: "started" });
    } catch {}
    return new Output("done");
  });
  expect(await rejected(evalNode(node, null, new VariablePool(), { steps }))).toBe(fault);
});

test("a kept steps capability is closed once evalNode resolves", async () => {
  const recorded: ValueRecord[] = [];
  const steps = new NodeSteps(1, [], (step) => recorded.push(step));
  const kept: NodeSteps[] = [];
  const node = new StepNode("a", (caps) => {
    kept.push(caps.steps!);
    return new Output("done");
  });
  await evalNode(node, null, new VariablePool(), { steps });

  let late: unknown;
  try {
    kept[0]!.record({ event: "late" });
  } catch (error) {
    late = error;
  }
  expect(late).toBeInstanceOf(CapabilityFault);
  expect(((late as CapabilityFault).original as Error).message).toBe("node steps capability is closed");
  expect(recorded).toEqual([]);
});

/** Return an engine for `START -> node -> END`; the other options go to the fixture. */
function single(node: Node, options: Parameters<typeof engine>[1] = {}): FlowEngine {
  return engine(graph([node], [
    [START_ID, node.id],
    [node.id, END_ID],
  ]), options);
}

/** Return an engine running `first` and `second` side by side after START. */
function pair(first: Node, second: Node, options: Parameters<typeof engine>[1] = {}): FlowEngine {
  return engine(graph([first, second], [
    [START_ID, first.id],
    [START_ID, second.id],
    [first.id, END_ID],
    [second.id, END_ID],
  ]), options);
}

/** Each worker count: serial and two workers. */
const WORKERS = [0, 2];

/** Return the records about `nodeId`, in log order. */
function records(built: FlowEngine, nodeId: string): RunRecord[] {
  return built.log
    .records()
    .map((envelope) => envelope.record)
    .filter((record) => (record as { nodeId?: string }).nodeId === nodeId);
}

/** Whether the log holds a record of class `kind`. */
function logged(built: FlowEngine, kind: abstract new (...args: never[]) => unknown): boolean {
  return built.log.records().some((envelope) => envelope.record instanceof kind);
}

test.each(WORKERS)("steps are durable before record returns (numWorkers %p)", async (numWorkers) => {
  const seen: unknown[] = [];
  let built!: FlowEngine;
  const node = new StepNode("work", (caps) => {
    const steps = caps.steps!;
    seen.push([steps.constructor, steps.starts, steps.prior]);
    for (const attempt of [1n, 2n]) {
      steps.record({ kind: "test", event: "started", attempt });
      const stepRecords = built.log.records().filter((envelope) => envelope.record instanceof NodeStepRecord);
      seen.push((stepRecords.at(-1)!.record as NodeStepRecord).step);
    }
    return new Output("done");
  });
  built = single(node, { numWorkers });
  const events = await built.run();

  expect(seen).toEqual([
    [NodeSteps, 1, []],
    { attempt: 1n, event: "started", kind: "test" },
    { attempt: 2n, event: "started", kind: "test" },
  ]);
  const work = records(built, "work");
  expect(work.slice(0, 3)).toEqual([
    new NodeStartedRecord("work"),
    new NodeStepRecord("work", 1, { kind: "test", event: "started", attempt: 1n }),
    new NodeStepRecord("work", 2, { kind: "test", event: "started", attempt: 2n }),
  ]);
  // The output's commit, then the success that accepts it.
  expect(work[3]).toBeInstanceOf(PoolCommitRecord);
  expect(work[4]).toBeInstanceOf(NodeSucceededRecord);
  expect(work.length).toBe(5);
  expect(built.state.result).toBeInstanceOf(RunSucceeded);
  expect(built.state.nodeSteps.has("work")).toBe(false);
  // A step publishes no observer: the node's observers are its start and its outcome.
  const workEvents = events.filter((event) => (event as { nodeId?: string }).nodeId === "work");
  expect(workEvents.map((event) => event.constructor)).toEqual([NodeStarted, NodeSucceeded]);
});

test.each(WORKERS)("a node without the trait receives no steps (numWorkers %p)", async (numWorkers) => {
  const seen: boolean[] = [];
  const built = single(
    new PlainNode("work", (caps) => {
      seen.push(caps.steps !== undefined);
      return new Output("done");
    }),
    { numWorkers },
  );
  await built.run();
  expect(seen).toEqual([false]);
  expect(built.state.result).toBeInstanceOf(RunSucceeded);
});

test("prior is a snapshot taken at the start", async () => {
  const seen: unknown[] = [];
  const built = single(
    new StepNode("work", (caps) => {
      const steps = caps.steps!;
      steps.record({ kind: "test", event: "started" });
      seen.push(steps.prior);
      return new Output("done");
    }),
  );
  await built.run();
  expect(seen).toEqual([[]]);
  expect(logged(built, NodeStepRecord)).toBe(true);
});

/**
 * Return a node whose first start records a failed step and crashes through a throwing
 * `onStream`, and whose later starts pass `steps` to `later`.
 */
function crashingNode(later: (steps: NodeSteps) => NodeResult): StepNode {
  return new StepNode("work", (caps) => {
    const steps = caps.steps!;
    if (steps.starts === 1) {
      steps.record({ kind: "test", event: "failed", attempt: 1n, nested: { retry_in: 1n } });
      caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "crash" }));
    }
    return later(steps);
  });
}

/** The callback error that stops the first start. */
const CRASH = new Error("crash");

/** Return an engine for `START -> node -> END` whose `onStream` throws `CRASH`. */
function crashing(node: StepNode, numWorkers = 0): FlowEngine {
  return single(node, {
    numWorkers,
    onStream: () => {
      throw CRASH;
    },
  });
}

/**
 * Run `first`, which crashes, then resume it with `numWorkers` and return the resumed engine: on
 * its live state, or with `replay` on a fresh state replayed from its log.
 */
async function crashAndResume(first: FlowEngine, numWorkers = 0, replay = false): Promise<FlowEngine> {
  expect(await rejected(first.run())).toBe(CRASH);
  const { log } = first;
  const state = replay
    ? replayRecords(new RunState(copyGraph(first.state.flow)), log.records(), { growBuilder: noGrowth })
    : first.state;
  expect([...state.inFlight.keys()]).toEqual(["work"]);
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers }));
  const resumed = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers });
  await resumed.run();
  return resumed;
}

describe.each(WORKERS)("resume sees the second start and the prior steps (numWorkers %p)", (numWorkers) => {
  for (const replay of [false, true]) {
    test(replay ? "replayed state" : "live state", async () => {
      const seen: unknown[] = [];
      const node = crashingNode((steps) => {
        seen.push([steps.starts, steps.prior]);
        return new Output("done");
      });
      const resumed = await crashAndResume(crashing(node, numWorkers), numWorkers, replay);
      const failed = { attempt: 1n, event: "failed", kind: "test", nested: { retry_in: 1n } };
      expect(seen).toEqual([[2, [failed]]]);
      expect(resumed.state.result).toBeInstanceOf(RunSucceeded);
      const steps = records(resumed, "work").filter((record) => record instanceof NodeStepRecord);
      expect(steps.map((record) => (record as NodeStepRecord).step)).toEqual([failed]);
    });
  }
});

test.each(WORKERS)("an engine step's n is the node's step count, across starts (numWorkers %p)", async (numWorkers) => {
  const seen: number[] = [];
  const node = crashingNode((steps) => {
    seen.push(steps.record({ kind: "test", event: "started" }));
    return new Output("done");
  });
  const resumed = await crashAndResume(crashing(node, numWorkers), numWorkers);
  // `record` returns the committed record's n, which counts the first start's step.
  expect(seen).toEqual([2]);
  const steps = records(resumed, "work").filter((record) => record instanceof NodeStepRecord);
  expect(steps.map((record) => (record as NodeStepRecord).n)).toEqual([1, 2]);
});

test("resume hands detached prior steps", async () => {
  const kept: unknown[] = [];
  let first!: FlowEngine;
  const node = crashingNode((steps) => {
    // The node gets copies: changing them must not reach the engine's state.
    (steps.prior[0]!["nested"] as Record<string, unknown>)["retry_in"] = 99n;
    kept.push(structuredClone(first.state.nodeSteps.get("work")));
    return new Output("done");
  });
  first = crashing(node);
  await crashAndResume(first);
  expect(kept).toEqual([[{ attempt: 1n, event: "failed", kind: "test", nested: { retry_in: 1n } }]]);
});

test.each(WORKERS)("a recorder log fault escapes as a host error (numWorkers %p)", async (numWorkers) => {
  const after: string[] = [];
  const built = single(
    new StepNode("work", (caps) => {
      caps.steps!.record({ kind: "test", event: "started", attempt: 2n });
      after.push("continued");
      return new Output("done");
    }),
    { numWorkers },
  );
  const fault = new Error("disk full");
  const append = built.log.append.bind(built.log);
  built.log.append = (record) => {
    if (record instanceof NodeStepRecord) throw fault;
    return append(record);
  };
  expect(await rejected(built.run())).toBe(fault);
  expect(after).toEqual([]);
  expect(logged(built, NodeFailedRecord)).toBe(false);
  expect(built.state.result).toBeNull();
});

test("a node that catches the recorder's log fault still fails the run with it", async () => {
  const built = single(
    new StepNode("work", (caps) => {
      try {
        caps.steps!.record({ kind: "test", event: "started" });
      } catch {}
      return new Output("done");
    }),
  );
  const fault = new Error("disk full");
  const append = built.log.append.bind(built.log);
  built.log.append = (record) => {
    if (record instanceof NodeStepRecord) throw fault;
    return append(record);
  };
  expect(await rejected(built.run())).toBe(fault);
  expect(logged(built, NodeSucceededRecord)).toBe(true);
  expect(records(built, "work").map((record) => record.constructor)).toEqual([NodeStartedRecord]);
  expect(built.state.result).toBeNull();
});

const invalidSteps: [string, unknown, ErrorConstructor][] = [
  ["object", { when: new Date(0) }, RangeError],
  ["nan", { retry_in: NaN }, RangeError],
  ["list", ["x"], TypeError],
  ["no kind", { event: "started" }, TypeError],
];

test.each(WORKERS.flatMap((numWorkers) => invalidSteps.map((row) => [numWorkers, ...row] as const)))(
  "an invalid step is a host error (numWorkers %p): %s",
  async (numWorkers, _, step, kind) => {
    const built = single(
      new StepNode("work", (caps) => {
        caps.steps!.record(step as ValueRecord);
        return new Output("done");
      }),
      { numWorkers },
    );
    expect(await rejected(built.run())).toBeInstanceOf(kind);
    expect(logged(built, NodeFailedRecord)).toBe(false);
    expect(logged(built, NodeStepRecord)).toBe(false);
    expect(built.state.result).toBeNull();
  },
);

test("serial skips a step once stop is set", async () => {
  const stop = new Error("host listener failed");
  const raised: unknown[] = [];
  const node = new StepNode("work", async (caps) => {
    await aborted(caps.signal!).catch(() => {});
    try {
      caps.steps!.record({ kind: "test", event: "started", attempt: 2n });
    } catch (error) {
      raised.push(error);
      throw error;
    }
    return new Output("done");
  });
  // The rejected promise is the stop: the engine aborts the run signal with its error.
  const onEvent = (event: RunEvent): Promise<void> | undefined =>
    event instanceof NodeStarted && event.nodeId === "work" ? Promise.reject(stop) : undefined;
  const built = single(node, { onEvent });

  expect(await rejected(built.run())).toBe(stop);
  expect(raised).toEqual([stop]);
  expect(logged(built, NodeStepRecord)).toBe(false);
});

test("a sibling failure stops a pooled node's steps", async () => {
  const raised: unknown[] = [];
  const reasons: unknown[] = [];
  const work = new StepNode("work", async (caps) => {
    // The sibling's failure aborts the run signal.
    await aborted(caps.signal!).catch(() => {});
    reasons.push(caps.signal!.reason);
    try {
      caps.steps!.record({ kind: "test", event: "started", attempt: 2n });
    } catch (error) {
      raised.push(error);
      throw error;
    }
    return new Output("done");
  });
  const boom = new PlainNode("boom", () => {
    throw new Error("boom");
  });
  const built = pair(work, boom, { numWorkers: 2 });

  await built.run();

  expect(built.state.result).toBeInstanceOf(RunFailed);
  expect((built.state.result as RunFailed).failure.nodeId).toBe("boom");
  expect(raised.length).toBe(1);
  expect(raised[0]).toBe(reasons[0]);
  expect(raised[0]).toBe(built.signal.reason);
  expect(records(built, "work").some((record) => record instanceof NodeStepRecord)).toBe(false);
});

test.each(WORKERS)("a kept recorder is closed after its evaluation (numWorkers %p)", async (numWorkers) => {
  const kept: NodeSteps[] = [];
  const late: unknown[] = [];
  const keeper = new StepNode("keeper", (caps) => {
    kept.push(caps.steps!);
    return new Output("kept");
  });
  const later = new PlainNode("later", () => {
    // A later node's evaluation tries the first node's recorder.
    try {
      kept[0]!.record({ event: "late" });
    } catch (error) {
      late.push(error);
    }
    return new Output("done");
  });
  const built = engine(graph([keeper, later], [
    [START_ID, "keeper"],
    ["keeper", "later"],
    ["later", END_ID],
  ]), { numWorkers });
  await built.run();

  expect(built.state.result).toBeInstanceOf(RunSucceeded);
  expect(late.length).toBe(1);
  expect(late[0]).toBeInstanceOf(CapabilityFault);
  expect(((late[0] as CapabilityFault).original as Error).message).toBe("node steps capability is closed");

  // After the run, a call fails at once, without touching the log.
  const size = built.log.nextSequence;
  expect(() => kept[0]!.record({ event: "late" })).toThrow(CapabilityFault);
  expect(built.log.nextSequence).toBe(size);
  expect(logged(built, NodeStepRecord)).toBe(false);
});

test("pooled nodes interleave their own steps", async () => {
  // The nodes take turns: a records, then b, then a, then b.
  const turns = new Map([
    ["a", Promise.withResolvers<void>()],
    ["b", Promise.withResolvers<void>()],
  ]);
  const bodyFor = (nodeId: string, other: string) => async (caps: NodeCaps): Promise<NodeResult> => {
    for (const attempt of [1n, 2n]) {
      if (nodeId === "b" || attempt === 2n) {
        // A fault elsewhere aborts the run; ending the wait then lets the drain finish.
        await Promise.race([turns.get(nodeId)!.promise, aborted(caps.signal!)]);
        turns.set(nodeId, Promise.withResolvers<void>());
      }
      caps.steps!.record({ kind: "test", node: nodeId, attempt });
      turns.get(other)!.resolve();
    }
    return new Output(nodeId);
  };
  const built = pair(new StepNode("a", bodyFor("a", "b")), new StepNode("b", bodyFor("b", "a")), { numWorkers: 2 });

  await built.run();

  expect(built.state.result).toBeInstanceOf(RunSucceeded);
  const steps = built.log
    .records()
    .map((envelope) => envelope.record)
    .filter((record) => record instanceof NodeStepRecord)
    .map((record) => [(record as NodeStepRecord).nodeId, (record as NodeStepRecord).n, (record as NodeStepRecord).step]);
  expect(steps).toEqual([
    ["a", 1, { attempt: 1n, kind: "test", node: "a" }],
    ["b", 1, { attempt: 1n, kind: "test", node: "b" }],
    ["a", 2, { attempt: 2n, kind: "test", node: "a" }],
    ["b", 2, { attempt: 2n, kind: "test", node: "b" }],
  ]);
});
