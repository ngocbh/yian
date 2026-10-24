/**
 * The stop at the raw node boundary: a port of `tests/engine/test_stop_event.py`.
 *
 * TS has no stop event and no `_HostStop`: a stop is the node's aborted signal, which every node
 * receives as `caps.signal`.
 *
 * Changed: `test_stop_cap_is_exact_event_and_only_reaches_opted_in_node` is "every node gets the
 * given signal as `caps.signal`" (there is no opt-in trait);
 * `test_direct_opted_in_eval_gets_private_unset_event_when_host_omits_one` is "`evalNode` without
 * a signal gives an un-aborted one"; `test_preset_stop_prevents_node_execution_and_propagates_host_control`
 * is "an aborted signal rejects with its reason and `forward` never runs" (the `primary` flag has
 * no counterpart); `test_node_host_stop_bypasses_evaluation_failure_with_identity` is "a node
 * rejecting after its signal aborted rejects `evalNode` with the reason, not a failure".
 *
 * Not ported: `test_public_run_flow_signature_has_no_stop_event`, the file's only run row: TS
 * reads no signatures, and `runFlow` takes `{ signal }`. The run boundary's stop rows (an
 * already-aborted signal, a stop recorded as `RunStopped` and resumed) are in `run.test.ts`.
 *
 * TS-only, the engine's stop (serial and pooled): a host abort records one `RunStopped` (`signal`)
 * listing the in-flight nodes in dispatch order before the drain; a synchronous `onEvent` throw, a
 * rejected `onEvent` promise, a throwing `onStream` the node catches, a rejected `onStream` promise,
 * a throwing steering method and a rejected steering `drain` promise record a `callback` stop; the
 * message is an `Error`'s message or a non-`Error` reason's text; an already-aborted signal rejects
 * before the first start with the log unchanged; nothing is recorded after a terminal result, after
 * an earlier stop in the segment (also one a previous engine recorded over the same state), for a
 * host fault (an invalid step, a malformed contract, an opener fault), or for a late rejection after
 * `run()` settled; an opener's `MemoryRecallError` is the node's failure, a `RunFailed`. A refused
 * step or an opener fault the node catches ends the run at once, so a host abort after it records
 * nothing and `run()` rejects with the fault. Pooled
 * only: a host abort or a late `onEvent` rejection during a host fault's drain changes nothing; a
 * host abort during a `RunFailed`'s drain is ignored and `run()` returns normally; an outcome
 * settling after a callback stop is discarded; a callback throw during a node's synchronous start
 * starts no further node. No test leaves an unhandled rejection or an uncaught exception.
 *
 * TS-only, steps and events after a stop (serial and pooled): after a host stop, a settled step
 * (`{ settled: true }`) is still appended after the `RunStopped` and a `ToolCallFinished` still
 * published; another step throws the node signal's reason and appends nothing, and another event or
 * chunk is dropped; a sink a node keeps drops what it gets once its evaluation ended, and a steering
 * it keeps throws a `CapabilityFault` without calling the inbox, during a later node and after
 * `run()` resolved, so the run neither stops nor aborts. Pooled only:
 * after a sibling's host fault, the settled step and the `ToolCallFinished` are still kept; after a
 * terminal `RunFailed`, they are refused.
 *
 * TS-only, a poisoned log (serial and pooled, a `JsonlRunLog` whose sync fails): a failing append
 * during an outcome, or during a step the node catches before a host abort, ends the run as a host
 * error with the log's error and no `RunStopped`; a log refusing the `RunStopped` of a host abort
 * leaves a note on the host's reason, which `run()` still rejects with; a failing rollback rejects
 * with the `AggregateError`; a step refused before reaching the log leaves the log accepting
 * appends.
 *
 * TS-only, nested runs (serial and pooled): a node running a child `FlowEngine` with
 * `signal: caps.signal` stops the child when the parent stops, and the child records its own
 * `RunStopped` (`signal`); a child's `RunFailed` doesn't abort the parent.
 *
 * TS-only, the per-node signals (serial and pooled): over 50 nodes, each node's link to the run
 * signal is removed when it settles, even for nodes that leave listeners on their own signal; the
 * host's signal keeps no listener once `run()` settles, whatever the ending (success, `RunFailed`,
 * a host stop, a host fault, an already-aborted signal).
 */

import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import {
  NodeStarted,
  NodeSucceeded,
  RunFailed,
  type RunEvent,
  RunSucceeded,
  StreamChunk,
  ToolCallFinished,
  ToolCallStarted,
} from "../../src/events.ts";
import { MemoryRecallError, MemoryScope } from "../../src/memory/contracts.ts";
import { CapabilityFault, Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import type { OutputContract } from "../../src/nodes/contracts.ts";
import { FlowEngine, type FlowEngineOptions } from "../../src/runtime/engine.ts";
import { evalNode, type MemoryContextOpener } from "../../src/runtime/eval_node.ts";
import {
  JsonlRunLog,
  NodeStartedRecord,
  NodeStepRecord,
  NodeSucceededRecord,
  RunStoppedRecord,
} from "../../src/runtime/log.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { aborted, engine, FailNode, FuncNode, graph, noGrowth, rejected } from "./run_state_fixtures.ts";

/** Record the capabilities supplied to one direct evaluation. */
class CapNode extends Node {
  readonly kind = NodeKind.CODE;
  caps: NodeCaps = {};

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    this.caps = caps;
    return new Output("done");
  }
}

/** Expose whether an aborted signal allowed the node body to start. */
class NeverRunsNode extends Node {
  readonly kind = NodeKind.CODE;
  ran = false;

  override forward(): NodeResult {
    this.ran = true;
    return new Output("unexpected");
  }
}

test("every node gets the given signal as caps.signal", async () => {
  const signal = new AbortController().signal;
  const first = new CapNode("first");
  const second = new CapNode("second");

  expect((await evalNode(first, null, new VariablePool(), { signal })).outcome).toEqual(new Output("done"));
  expect((await evalNode(second, null, new VariablePool(), { signal })).outcome).toEqual(new Output("done"));

  expect(first.caps).toEqual({ signal });
  expect(first.caps.signal).toBe(signal);
  expect(second.caps.signal).toBe(signal);
});

test("evalNode without a signal gives an un-aborted one", async () => {
  const node = new CapNode("node");

  expect((await evalNode(node, null, new VariablePool())).outcome).toEqual(new Output("done"));

  expect(Object.keys(node.caps)).toEqual(["signal"]);
  expect(node.caps.signal).toBeInstanceOf(AbortSignal);
  expect(node.caps.signal!.aborted).toBe(false);
});

test("an aborted signal rejects with its reason and forward never runs", async () => {
  const controller = new AbortController();
  const reason = new Error("stopped");
  controller.abort(reason);
  const node = new NeverRunsNode("never");

  expect(await rejected(evalNode(node, null, new VariablePool(), { signal: controller.signal }))).toBe(reason);
  expect(node.ran).toBe(false);
});

test("a node rejecting after its signal aborted rejects evalNode with the reason, not a failure", async () => {
  const controller = new AbortController();
  const reason = new Error("terminal EOF");

  class AbortsThenThrows extends Node {
    readonly kind = NodeKind.CODE;

    override forward(): NodeResult {
      controller.abort(reason);
      throw new Error("interrupted");
    }
  }

  expect(
    await rejected(evalNode(new AbortsThenThrows("work"), null, new VariablePool(), { signal: controller.signal })),
  ).toBe(reason);
});

/** Each worker mode: serial and two pooled workers. */
const MODES = [
  ["serial", 0],
  ["pooled", 2],
] as const;

const unhandled: unknown[] = [];
const collect = (reason: unknown): void => {
  unhandled.push(reason);
};
beforeEach(() => {
  unhandled.length = 0;
  process.on("unhandledRejection", collect);
  process.on("uncaughtException", collect);
});
afterEach(async () => {
  // A late rejection surfaces only after the microtasks of the test drain.
  await new Promise((resolve) => setTimeout(resolve, 0));
  process.off("unhandledRejection", collect);
  process.off("uncaughtException", collect);
  expect(unhandled).toEqual([]);
});

/** Runs a scripted body over its capabilities and outputs its value; it may record steps, stream and steer. */
class ScriptNode extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsSteps = true;
  override readonly needsLiveEvents = true;
  override readsSteerInbox = true;
  readonly body: (caps: NodeCaps) => unknown;

  constructor(nodeId: string, body: (caps: NodeCaps) => unknown) {
    super(nodeId);
    this.body = body;
  }

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    return new Output(await this.body(caps));
  }
}

/** A `ScriptNode` that opens memory context. */
class MemoryScriptNode extends ScriptNode {
  override readonly needsMemoryContext = true;
}

/**
 * Return a node that blocks until its signal aborts, then outputs what `onAbort` returns given its
 * signal and capabilities (by default it rejects with the signal's reason), and a promise that
 * resolves once it blocks.
 */
function waiter(
  nodeId: string,
  onAbort: (signal: AbortSignal, caps: NodeCaps) => unknown = (signal) => {
    throw signal.reason;
  },
): { node: ScriptNode; entered: Promise<void> } {
  const { promise, resolve } = Promise.withResolvers<void>();
  const node = new ScriptNode(nodeId, async (caps) => {
    resolve();
    await aborted(caps.signal!).catch(() => {});
    return await onAbort(caps.signal!, caps);
  });
  return { node, entered: promise };
}

/** Return START fanning out to `nodes`, each joined into an output-less END. */
function fan(...nodes: readonly Node[]) {
  return graph(nodes, nodes.flatMap((node) => [[START_ID, node.id], [node.id, END_ID]] as [string, string][]));
}

/** Return an engine over `fan(...nodes)`. */
function fanEngine(nodes: readonly Node[], options: Partial<FlowEngineOptions> = {}): FlowEngine {
  return engine(fan(...nodes), options);
}

/** The run's `RunStopped` records, in log order. */
function stops(built: FlowEngine): RunStoppedRecord[] {
  return built.log
    .records()
    .map((envelope) => envelope.record)
    .filter((record): record is RunStoppedRecord => record instanceof RunStoppedRecord);
}

/** Whether the run logged a record of class `kind` about `nodeId`. */
function logged(built: FlowEngine, kind: abstract new (...args: never[]) => unknown, nodeId: string): boolean {
  return built.log
    .records()
    .some((envelope) => envelope.record instanceof kind && (envelope.record as { nodeId?: string }).nodeId === nodeId);
}

/** The record the run appended last. */
function lastRecord(built: FlowEngine): unknown {
  return built.log.records().at(-1)!.record;
}

describe.each(MODES)("the host signal (%s)", (_, numWorkers) => {
  test("a host abort records one RunStopped listing the in-flight nodes before the drain", async () => {
    const host = new AbortController();
    const reason = new Error("stop now");
    const seen: RunStoppedRecord[][] = [];
    let built!: FlowEngine;
    const first = waiter("first", (signal) => {
      seen.push(stops(built));
      throw signal.reason;
    });
    const second = waiter("second", (signal) => {
      seen.push(stops(built));
      throw signal.reason;
    });
    built = fanEngine([first.node, second.node], { numWorkers, signal: host.signal });

    const running = built.run();
    await first.entered;
    if (numWorkers > 0) await second.entered;
    host.abort(reason);

    expect(await rejected(running)).toBe(reason);
    const nodeIds = numWorkers === 0 ? ["first"] : ["first", "second"];
    expect(stops(built)).toEqual([new RunStoppedRecord("signal", "stop now", nodeIds)]);
    // Each blocked node saw the stop in the log when its signal aborted.
    expect(seen).toEqual(nodeIds.map(() => stops(built)));
    expect(lastRecord(built)).toBeInstanceOf(RunStoppedRecord);
    expect(logged(built, NodeStartedRecord, "second")).toBe(numWorkers > 0);
    expect(built.state.result).toBeNull();
    expect([...built.state.inFlight.keys()]).toEqual(nodeIds);
    expect(built.signal.reason).toBe(reason);
  });

  test("an already-aborted signal rejects before the first start with the log unchanged", async () => {
    const host = new AbortController();
    const reason = new Error("stopped before");
    host.abort(reason);
    const ran: string[] = [];
    const built = fanEngine([new FuncNode("work", () => ran.push("work"))], { numWorkers, signal: host.signal });
    const before = built.log.records().length;

    expect(await rejected(built.run())).toBe(reason);
    expect(built.log.records().length).toBe(before);
    expect(ran).toEqual([]);
  });

  test("the message is an Error's message or a non-Error reason's text", async () => {
    const messages: string[] = [];
    const reasons: unknown[] = [];
    for (const abort of [(c: AbortController) => c.abort(), (c: AbortController) => c.abort("plain text"), (c: AbortController) => c.abort(42)]) {
      const host = new AbortController();
      const { node, entered } = waiter("work");
      const built = fanEngine([node], { numWorkers, signal: host.signal });
      const running = built.run();
      await entered;
      abort(host);
      expect(await rejected(running)).toBe(host.signal.reason);
      reasons.push(host.signal.reason);
      messages.push(...stops(built).map((record) => record.message));
    }
    // The default reason is a `DOMException` `AbortError`, whose text differs between runtimes.
    expect(reasons[0]).toBeInstanceOf(Error);
    expect(messages).toEqual([(reasons[0] as Error).message, "plain text", "42"]);
  });

  test("a host abort while RunSucceeded is published is ignored", async () => {
    const host = new AbortController();
    const onEvent = (event: RunEvent): void => {
      if (event instanceof RunSucceeded) host.abort(new Error("too late"));
    };
    const built = fanEngine([new FuncNode("work", () => "done")], { numWorkers, signal: host.signal, onEvent });

    expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
    expect(stops(built)).toEqual([]);
    expect(built.signal.aborted).toBe(false);
  });

  test.each([
    ["RunSucceeded", () => new FuncNode("work", () => "done"), RunSucceeded],
    ["RunFailed", () => new FailNode("work"), RunFailed],
  ] as const)("an onEvent that throws while %s is published records nothing", async (_name, build, terminal) => {
    const error = new Error("observer failed");
    const onEvent = (event: RunEvent): void => {
      if (event instanceof terminal) throw error;
    };
    const built = fanEngine([build()], { numWorkers, onEvent });

    expect(await rejected(built.run())).toBe(error);
    expect(built.state.result).toBeInstanceOf(terminal);
    expect(stops(built)).toEqual([]);
  });

  test("a late rejection after run() settled is ignored", async () => {
    const late = Promise.withResolvers<void>();
    const onEvent = (event: RunEvent): Promise<void> | undefined =>
      event instanceof NodeStarted && event.nodeId === "work" ? late.promise : undefined;
    const built = fanEngine([new FuncNode("work", () => "done")], { numWorkers, onEvent });

    expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
    const after = built.log.records().length;
    late.reject(new Error("late"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(built.log.records().length).toBe(after);
    expect(built.signal.aborted).toBe(false);
  });
});

describe.each(MODES)("callback stops (%s)", (_, numWorkers) => {
  test("a synchronous onEvent throw records a callback stop", async () => {
    const error = new Error("observer failed");
    const ran: string[] = [];
    const onEvent = (event: RunEvent): void => {
      if (event instanceof NodeStarted && event.nodeId === "work") throw error;
    };
    const built = fanEngine([new FuncNode("work", () => ran.push("work"))], { numWorkers, onEvent });

    expect(await rejected(built.run())).toBe(error);
    expect(stops(built)).toEqual([new RunStoppedRecord("callback", "observer failed", ["work"])]);
    expect(lastRecord(built)).toBeInstanceOf(RunStoppedRecord);
    expect(ran).toEqual([]);
  });

  test("a rejected onEvent promise records a callback stop and discards the outcome settling after it", async () => {
    const error = new Error("observer rejected");
    const seen: RunStoppedRecord[][] = [];
    let built!: FlowEngine;
    const { node } = waiter("work", () => {
      seen.push(stops(built));
      return "late";
    });
    const onEvent = (event: RunEvent): Promise<void> | undefined =>
      event instanceof NodeStarted && event.nodeId === "work" ? Promise.reject(error) : undefined;
    built = fanEngine([node], { numWorkers, onEvent });

    expect(await rejected(built.run())).toBe(error);
    const stopped = [new RunStoppedRecord("callback", "observer rejected", ["work"])];
    expect(stops(built)).toEqual(stopped);
    expect(seen).toEqual([stopped]);
    expect(logged(built, NodeSucceededRecord, "work")).toBe(false);
    expect(built.signal.reason).toBe(error);
  });

  test("an onStream throw the node catches still records a callback stop", async () => {
    const error = new Error("stream callback failed");
    const node = new ScriptNode("work", (caps) => {
      try {
        caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "a" }));
      } catch {}
      return "done";
    });
    const onStream = (): void => {
      throw error;
    };
    const built = fanEngine([node], { numWorkers, onStream });

    expect(await rejected(built.run())).toBe(error);
    expect(stops(built)).toEqual([new RunStoppedRecord("callback", "stream callback failed", ["work"])]);
    expect(logged(built, NodeSucceededRecord, "work")).toBe(false);
  });

  test("a throwing steering method records a callback stop", async () => {
    const error = new Error("inbox failed");
    const node = new ScriptNode("work", (caps) => {
      try {
        caps.steering!.interruptSignal("work");
      } catch {}
      return "done";
    });
    const steering = {
      interruptSignal: (): AbortSignal => {
        throw error;
      },
    };
    const built = fanEngine([node], { numWorkers, steering });

    expect(await rejected(built.run())).toBe(error);
    expect(stops(built)).toEqual([new RunStoppedRecord("callback", "inbox failed", ["work"])]);
  });

  test("a rejected onStream promise records a callback stop", async () => {
    const error = new Error("stream callback rejected");
    const node = new ScriptNode("work", (caps) => {
      caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "a" }));
      return "done";
    });
    const built = fanEngine([node], { numWorkers, onStream: () => Promise.reject(error) });

    expect(await rejected(built.run())).toBe(error);
    expect(stops(built)).toEqual([new RunStoppedRecord("callback", "stream callback rejected", ["work"])]);
    expect(logged(built, NodeSucceededRecord, "work")).toBe(false);
  });

  test("a segment that already has a RunStopped gets no second one", async () => {
    const error = new Error("observer failed");
    const first = new FuncNode("first", () => "done");
    const { node: second, entered } = waiter("second");
    const flow = graph([first, second], [
      [START_ID, "first"],
      ["first", "second"],
      ["second", END_ID],
    ]);
    const onEvent = (event: RunEvent): void => {
      if (event instanceof NodeSucceeded && event.nodeId === "first") throw error;
    };
    const stopped = engine(flow, { numWorkers, onEvent });
    expect(await rejected(stopped.run())).toBe(error);
    expect(stops(stopped)).toEqual([new RunStoppedRecord("callback", "observer failed", [])]);

    // A second engine over the same state and log is still in the stopped segment.
    const host = new AbortController();
    const reason = new Error("stop now");
    const again = new FlowEngine(stopped.state, stopped.log, { growBuilder: noGrowth, numWorkers, signal: host.signal });
    const running = again.run();
    await entered;
    host.abort(reason);

    expect(await rejected(running)).toBe(reason);
    expect(stops(again)).toEqual(stops(stopped));
    expect(logged(again, NodeStartedRecord, "second")).toBe(true);
  });

  test("a host abort, then a callback failure, records one stop", async () => {
    const host = new AbortController();
    const reason = new Error("stop now");
    const late = Promise.withResolvers<void>();
    const { node, entered } = waiter("work", async (signal) => {
      late.reject(new Error("observer rejected late"));
      await late.promise.catch(() => {});
      throw signal.reason;
    });
    const onEvent = (event: RunEvent): Promise<void> | undefined =>
      event instanceof NodeStarted && event.nodeId === "work" ? late.promise : undefined;
    const built = fanEngine([node], { numWorkers, signal: host.signal, onEvent });

    const running = built.run();
    await entered;
    host.abort(reason);

    expect(await rejected(running)).toBe(reason);
    expect(stops(built)).toEqual([new RunStoppedRecord("signal", "stop now", ["work"])]);
  });
});

describe.each(MODES)("host faults record no stop (%s)", (_, numWorkers) => {
  test("an invalid step", async () => {
    let refusal: unknown;
    const node = new ScriptNode("work", (caps) => {
      try {
        caps.steps!.record({ when: new Date() } as never);
      } catch (error) {
        refusal = (error as { original?: unknown }).original;
        throw error;
      }
      return "done";
    });
    const built = fanEngine([node], { numWorkers });

    expect(await rejected(built.run())).toBe(refusal);
    expect(refusal).toBeInstanceOf(Error);
    expect(stops(built)).toEqual([]);
  });

  test("a refused step the node catches ends the run at once: a later host abort records nothing", async () => {
    const host = new AbortController();
    let refusal: unknown;
    const node = new ScriptNode("work", (caps) => {
      try {
        caps.steps!.record({ when: new Date() } as never);
      } catch (error) {
        refusal = (error as { original?: unknown }).original;
      }
      host.abort(new Error("stop now"));
      return "done";
    });
    const built = fanEngine([node], { numWorkers, signal: host.signal });

    expect(await rejected(built.run())).toBe(refusal);
    expect(refusal).toBeInstanceOf(Error);
    expect(stops(built)).toEqual([]);
    expect(built.signal.reason).toBe(refusal);
  });

  test("a malformed contract", async () => {
    const built = fanEngine([new FuncNode("work", () => "done")], { numWorkers });
    built.state.flow.nodes.get("work")!.outputContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;

    expect(await rejected(built.run())).toBeInstanceOf(TypeError);
    expect(stops(built)).toEqual([]);
  });

  test("an opener fault", async () => {
    const error = new Error("opening failed");
    const node = new MemoryScriptNode("work", async (caps) => {
      await caps.openMemoryContext!("system", "prompt");
      return "done";
    });
    node.memoryScope = new MemoryScope(["work"], { memories: ["session"] });
    const opener: MemoryContextOpener = async () => {
      throw error;
    };
    const built = fanEngine([node], { numWorkers, memoryContextOpener: opener });

    expect(await rejected(built.run())).toBe(error);
    expect(stops(built)).toEqual([]);
  });

  test("an opener fault the node catches ends the run at once: a later host abort records nothing", async () => {
    const host = new AbortController();
    const error = new Error("opening failed");
    const node = new MemoryScriptNode("work", async (caps) => {
      await caps.openMemoryContext!("system", "prompt").catch(() => {});
      host.abort(new Error("stop now"));
      return "done";
    });
    node.memoryScope = new MemoryScope(["work"], { memories: ["session"] });
    const opener: MemoryContextOpener = async () => {
      throw error;
    };
    const built = fanEngine([node], { numWorkers, memoryContextOpener: opener, signal: host.signal });

    expect(await rejected(built.run())).toBe(error);
    expect(stops(built)).toEqual([]);
    expect(built.signal.reason).toBe(error);
  });

  test("an opener's MemoryRecallError is the node's failure, not a host fault", async () => {
    const error = new MemoryRecallError("recall failed");
    const node = new MemoryScriptNode("work", async (caps) => {
      await caps.openMemoryContext!("system", "prompt");
      return "done";
    });
    node.memoryScope = new MemoryScope(["work"], { memories: ["session"] });
    const opener: MemoryContextOpener = async () => {
      throw error;
    };
    const built = fanEngine([node], { numWorkers, memoryContextOpener: opener });

    const terminal = (await built.run()).at(-1);
    expect(terminal).toBeInstanceOf(RunFailed);
    expect((terminal as RunFailed).failure.causeType).toBe("MemoryRecallError");
    expect(stops(built)).toEqual([]);
    // The run signal aborted with its default reason after the `RunFailed`, not with the error.
    expect(built.signal.reason).not.toBe(error);
    expect((built.signal.reason as Error).name).toBe("AbortError");
  });
});

describe("the pooled drain", () => {
  /** Return a node that waits until `blocked` blocks, then records an invalid step, and its refusal. */
  function badStep(blocked: Promise<void>): { node: ScriptNode; refusal: () => unknown } {
    let refusal: unknown;
    const node = new ScriptNode("step", async (caps) => {
      await blocked;
      try {
        caps.steps!.record({ when: new Date() } as never);
      } catch (error) {
        refusal = (error as { original?: unknown }).original;
        throw error;
      }
      return "done";
    });
    return { node, refusal: () => refusal };
  }

  test("a host abort during a host fault's drain records nothing and keeps the fault", async () => {
    const host = new AbortController();
    const blocked = waiter("blocked", () => {
      host.abort(new Error("late stop"));
      return "late";
    });
    const step = badStep(blocked.entered);
    const built = fanEngine([blocked.node, step.node], { numWorkers: 2, signal: host.signal });

    expect(await rejected(built.run())).toBe(step.refusal());
    expect(host.signal.aborted).toBe(true);
    expect(stops(built)).toEqual([]);
    expect(built.signal.reason).toBe(step.refusal());
  });

  test("a late onEvent rejection during a host fault's drain records nothing and keeps the fault", async () => {
    const late = Promise.withResolvers<void>();
    const blocked = waiter("blocked", async () => {
      late.reject(new Error("observer rejected late"));
      await late.promise.catch(() => {});
      return "late";
    });
    const step = badStep(blocked.entered);
    const onEvent = (event: RunEvent): Promise<void> | undefined =>
      event instanceof NodeStarted && event.nodeId === "blocked" ? late.promise : undefined;
    const built = fanEngine([blocked.node, step.node], { numWorkers: 2, onEvent });

    expect(await rejected(built.run())).toBe(step.refusal());
    expect(stops(built)).toEqual([]);
  });

  test("a host abort during a RunFailed's drain is ignored", async () => {
    const host = new AbortController();
    const blocked = waiter("blocked", () => {
      host.abort(new Error("late stop"));
      return "late";
    });
    const fail = new ScriptNode("fail", async () => {
      await blocked.entered;
      throw new Error("boom");
    });
    const built = fanEngine([blocked.node, fail], { numWorkers: 2, signal: host.signal });

    expect((await built.run()).at(-1)).toBeInstanceOf(RunFailed);
    expect(host.signal.aborted).toBe(true);
    expect(stops(built)).toEqual([]);
    expect(logged(built, NodeSucceededRecord, "blocked")).toBe(false);
  });

  test("a callback throw during a node's synchronous start starts no further node", async () => {
    const error = new Error("stream callback failed");
    const first = new ScriptNode("first", (caps) => {
      caps.emitStream!(new StreamChunk({ nodeId: "first", key: "output", chunk: "a" }));
      return "done";
    });
    const second = new FuncNode("second", () => "done");
    const onStream = (): void => {
      throw error;
    };
    const built = fanEngine([first, second], { numWorkers: 2, onStream });

    expect(await rejected(built.run())).toBe(error);
    expect(logged(built, NodeStartedRecord, "second")).toBe(false);
    expect(stops(built)).toEqual([new RunStoppedRecord("callback", "stream callback failed", ["first"])]);
  });
});

/** A tool call's events, as a TOOL node would emit them. */
const callFields = { nodeId: "work", n: 1, callIndex: 0, toolCallId: "call-1", name: "lookup" };

describe.each(MODES)("steps and events after a stop (%s)", (_, numWorkers) => {
  test("a settled step and a ToolCallFinished after a host stop are recorded and published after the RunStopped", async () => {
    const host = new AbortController();
    const reason = new Error("stop now");
    const seen: RunEvent[] = [];
    const { node, entered } = waiter("work", (signal, caps) => {
      // The call settled while the signal was live; the node reports it after the stop.
      caps.steps!.record({ kind: "test", call: "call-1" }, { settled: true });
      caps.emitEvent!(new ToolCallFinished({ ...callFields, result: "ok" }));
      throw signal.reason;
    });
    const built = fanEngine([node], { numWorkers, signal: host.signal, onEvent: (event) => void seen.push(event) });

    const running = built.run();
    await entered;
    host.abort(reason);

    expect(await rejected(running)).toBe(reason);
    const records = built.log.records().map((envelope) => envelope.record);
    expect(records.slice(-2)).toEqual([
      new RunStoppedRecord("signal", "stop now", ["work"]),
      new NodeStepRecord("work", 1, { kind: "test", call: "call-1" }),
    ]);
    expect(seen.at(-1)).toEqual(new ToolCallFinished({ ...callFields, result: "ok" }));
  });

  test("another step and event after a host stop are refused", async () => {
    const host = new AbortController();
    const reason = new Error("stop now");
    const seen: unknown[] = [];
    const raised: unknown[] = [];
    const nodeSignals: AbortSignal[] = [];
    const { node, entered } = waiter("work", (signal, caps) => {
      nodeSignals.push(signal);
      try {
        caps.steps!.record({ kind: "test", call: "call-1" });
      } catch (error) {
        raised.push(error);
      }
      caps.emitEvent!(new ToolCallStarted(callFields));
      caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "a" }));
      throw signal.reason;
    });
    const built = fanEngine([node], {
      numWorkers,
      signal: host.signal,
      onEvent: (event) => void seen.push(event),
      onStream: (chunk) => void seen.push(chunk),
    });

    const running = built.run();
    await entered;
    const before = seen.length;
    host.abort(reason);

    expect(await rejected(running)).toBe(reason);
    expect(raised).toEqual([reason]);
    expect(raised[0]).toBe(nodeSignals[0]!.reason);
    expect(logged(built, NodeStepRecord, "work")).toBe(false);
    expect(seen.length).toBe(before);
    expect(lastRecord(built)).toBeInstanceOf(RunStoppedRecord);
  });

  test("a kept sink drops what it gets once its evaluation ended", async () => {
    const seen: unknown[] = [];
    let keptCaps!: NodeCaps;
    const keeper = new ScriptNode("keeper", (caps) => {
      keptCaps = caps;
      return "kept";
    });
    const later = new FuncNode("later", () => {
      keptCaps.emitEvent!(new ToolCallFinished({ ...callFields, nodeId: "keeper", result: "late" }));
      keptCaps.emitStream!(new StreamChunk({ nodeId: "keeper", key: "output", chunk: "late" }));
      return "done";
    });
    const flow = graph([keeper, later], [
      [START_ID, "keeper"],
      ["keeper", "later"],
      ["later", END_ID],
    ]);
    const built = engine(flow, {
      numWorkers,
      onEvent: (event) => void seen.push(event),
      onStream: (chunk) => void seen.push(chunk),
    });

    expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
    expect(seen.filter((item) => item instanceof ToolCallFinished || item instanceof StreamChunk)).toEqual([]);
  });

  test("a kept steering refuses calls once its evaluation ended, without reaching the inbox", async () => {
    const calls: string[] = [];
    const steering = {
      interruptSignal: (): AbortSignal => {
        calls.push("interruptSignal");
        throw new Error("inbox failed");
      },
    };
    let keptCaps!: NodeCaps;
    const keeper = new ScriptNode("keeper", (caps) => {
      keptCaps = caps;
      return "kept";
    });
    const refusals: unknown[] = [];
    const refuse = (): void => {
      for (const call of [() => keptCaps.steering!.interruptSignal("keeper")]) {
        try {
          call();
        } catch (error) {
          refusals.push(error);
        }
      }
    };
    const later = new FuncNode("later", () => {
      refuse();
      return "done";
    });
    const flow = graph([keeper, later], [
      [START_ID, "keeper"],
      ["keeper", "later"],
      ["later", END_ID],
    ]);
    const built = engine(flow, { numWorkers, steering });

    expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
    refuse();
    expect(refusals.length).toBe(2);
    for (const refusal of refusals) {
      expect(refusal).toBeInstanceOf(CapabilityFault);
      expect((refusal as CapabilityFault).original).toEqual(new Error("steering capability is closed"));
    }
    expect(calls).toEqual([]);
    expect(stops(built)).toEqual([]);
    expect(built.signal.aborted).toBe(false);
  });
});

test("a settled step and a ToolCallFinished are refused after a terminal result", async () => {
  const seen: RunEvent[] = [];
  const raised: unknown[] = [];
  const blocked = waiter("work", (signal, caps) => {
    try {
      caps.steps!.record({ kind: "test", call: "call-1" }, { settled: true });
    } catch (error) {
      raised.push(error);
    }
    caps.emitEvent!(new ToolCallFinished({ ...callFields, result: "ok" }));
    throw signal.reason;
  });
  const fail = new ScriptNode("fail", async () => {
    await blocked.entered;
    throw new Error("boom");
  });
  const built = fanEngine([blocked.node, fail], { numWorkers: 2, onEvent: (event) => void seen.push(event) });

  expect((await built.run()).at(-1)).toBeInstanceOf(RunFailed);
  expect(raised).toEqual([built.signal.reason]);
  expect(logged(built, NodeStepRecord, "work")).toBe(false);
  expect(seen.some((event) => event instanceof ToolCallFinished)).toBe(false);
});

test("a settled step and a ToolCallFinished are still recorded after a sibling's host fault", async () => {
  const seen: RunEvent[] = [];
  const blocked = waiter("work", (signal, caps) => {
    caps.steps!.record({ kind: "test", call: "call-1" }, { settled: true });
    caps.emitEvent!(new ToolCallFinished({ ...callFields, result: "ok" }));
    throw signal.reason;
  });
  let refusal: unknown;
  const step = new ScriptNode("step", async (caps) => {
    await blocked.entered;
    try {
      caps.steps!.record({ when: new Date() } as never);
    } catch (error) {
      refusal = (error as { original?: unknown }).original;
      throw error;
    }
    return "done";
  });
  const built = fanEngine([blocked.node, step], { numWorkers: 2, onEvent: (event) => void seen.push(event) });

  expect(await rejected(built.run())).toBe(refusal);
  expect(refusal).toBeInstanceOf(Error);
  expect(stops(built)).toEqual([]);
  expect(lastRecord(built)).toEqual(new NodeStepRecord("work", 1, { kind: "test", call: "call-1" }));
  expect(seen.at(-1)).toEqual(new ToolCallFinished({ ...callFields, result: "ok" }));
});

describe.each(MODES)("nested runs (%s)", (_, numWorkers) => {
  test("a child run given the node's signal stops when the parent stops and records its own RunStopped", async () => {
    const host = new AbortController();
    const reason = new Error("stop now");
    const child = waiter("child");
    let inner!: FlowEngine;
    const parent = new ScriptNode("parent", async (caps) => {
      inner = fanEngine([child.node], { numWorkers, signal: caps.signal });
      await inner.run();
      return "done";
    });
    const built = fanEngine([parent], { numWorkers, signal: host.signal });

    const running = built.run();
    await child.entered;
    host.abort(reason);

    expect(await rejected(running)).toBe(reason);
    expect(stops(built)).toEqual([new RunStoppedRecord("signal", "stop now", ["parent"])]);
    expect(stops(inner)).toEqual([new RunStoppedRecord("signal", "stop now", ["child"])]);
    expect(inner.signal.reason).toBe(reason);
  });

  test("a child failure doesn't abort the parent", async () => {
    let inner!: FlowEngine;
    let parentSignal!: AbortSignal;
    const parent = new ScriptNode("parent", async (caps) => {
      parentSignal = caps.signal!;
      inner = fanEngine([new FailNode("child")], { numWorkers, signal: caps.signal });
      return ((await inner.run()).at(-1) as RunFailed).failure.message;
    });
    const built = fanEngine([parent], { numWorkers });

    expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
    expect(inner.state.result).toBeInstanceOf(RunFailed);
    // The child's run signal aborted with its `RunFailed`; the parent's node signal didn't.
    expect(inner.signal.aborted).toBe(true);
    expect(parentSignal.aborted).toBe(false);
    expect(built.signal.aborted).toBe(false);
  });
});

/**
 * Track the abort listeners added to `signal` and not removed since; `added` counts every add. A
 * listener that fired with `{ once: true }` stays counted as live.
 */
function spyListeners(signal: AbortSignal): { live: Set<unknown>; added: number } {
  const spy = { live: new Set<unknown>(), added: 0 };
  const add = signal.addEventListener.bind(signal);
  const remove = signal.removeEventListener.bind(signal);
  signal.addEventListener = (...args: Parameters<AbortSignal["addEventListener"]>): void => {
    if (args[0] === "abort") {
      spy.live.add(args[1]);
      spy.added += 1;
    }
    add(...args);
  };
  signal.removeEventListener = (...args: Parameters<AbortSignal["removeEventListener"]>): void => {
    if (args[0] === "abort") spy.live.delete(args[1]);
    remove(...args);
  };
  return spy;
}

describe.each(MODES)("per-node signals (%s)", (_, numWorkers) => {
  test("each node's link to the run signal is removed, even when the node leaves listeners on its own", async () => {
    const nodes = Array.from(
      { length: 50 },
      (_, index) =>
        new ScriptNode(`node-${index}`, (caps) => {
          caps.signal!.addEventListener("abort", () => {});
          return "done";
        }),
    );
    const built = fanEngine(nodes, { numWorkers });
    const spy = spyListeners(built.signal);

    expect((await built.run()).at(-1)).toBeInstanceOf(RunSucceeded);
    const starts = built.log.records().filter((envelope) => envelope.record instanceof NodeStartedRecord).length;
    expect(starts).toBe(52);
    expect(spy.added).toBe(starts);
    expect(spy.live.size).toBe(0);
  });

  test.each(["success", "RunFailed", "a host stop", "a host fault", "an already-aborted signal"] as const)(
    "the host's signal keeps no listener after %s",
    async (ending) => {
      const host = new AbortController();
      const spy = spyListeners(host.signal);
      if (ending === "an already-aborted signal") host.abort(new Error("stopped before"));
      const blocked = waiter("work");
      const node =
        ending === "RunFailed"
          ? new FailNode("work")
          : ending === "a host stop"
            ? blocked.node
            : new FuncNode("work", () => "done");
      const built = fanEngine([node], { numWorkers, signal: host.signal });
      if (ending === "a host fault") {
        built.state.flow.nodes.get("work")!.outputContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;
      }

      const running = built.run().catch(() => {});
      if (ending === "a host stop") {
        await blocked.entered;
        host.abort(new Error("stop now"));
      }
      await running;

      expect(spy.added).toBe(ending === "an already-aborted signal" ? 0 : 1);
      expect(spy.live.size).toBe(0);
    },
  );
});

describe.each(MODES)("a poisoned log (%s)", (_, numWorkers) => {
  let dir: string;
  let log: JsonlRunLog | null = null;
  /** Whether the log's next sync fails; `"always"` fails the rollback's sync too. */
  let failSync: "once" | "always" | null = null;
  const failure = new Error("fsync failed");
  let fsyncSpy: ReturnType<typeof spyOn>;

  beforeEach(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "yian-stop-"));
    failSync = null;
    const real = fs.fsyncSync;
    fsyncSpy = spyOn(fs, "fsyncSync").mockImplementation((fd) => {
      if (failSync === null) return real(fd);
      if (failSync === "once") failSync = null;
      throw failure;
    });
  });

  afterEach(() => {
    fsyncSpy.mockRestore();
    log?.close();
    log = null;
    fs.rmSync(dir, { recursive: true, force: true });
  });

  /** Return an engine over `fan(...nodes)` whose log is a `JsonlRunLog` holding the fixture's start. */
  function jsonlEngine(nodes: readonly Node[], options: Partial<FlowEngineOptions> = {}): FlowEngine {
    const built = fanEngine(nodes, { numWorkers });
    log = JsonlRunLog.create(path.join(dir, "run.jsonl"), built.log.runId);
    for (const envelope of built.log.records()) log.append(envelope.record);
    return new FlowEngine(built.state, log, { growBuilder: noGrowth, numWorkers, ...options });
  }

  test("a failing append during an outcome ends the run as a host error with no RunStopped", async () => {
    const node = new FuncNode("work", () => {
      failSync = "once";
      return "done";
    });
    const built = jsonlEngine([node]);

    expect(await rejected(built.run())).toBe(failure);
    expect(stops(built)).toEqual([]);
    expect(logged(built, NodeSucceededRecord, "work")).toBe(false);
    expect(built.state.result).toBeNull();
    expect(() => built.log.append(new NodeStartedRecord("late"))).toThrow(/refuses appends after a failed append/);
  });

  test("a failing append during a step the node catches, then a host abort: no RunStopped", async () => {
    const host = new AbortController();
    const node = new ScriptNode("work", (caps) => {
      failSync = "once";
      try {
        caps.steps!.record({ kind: "test", call: "call-1" });
      } catch {}
      host.abort(new Error("stop now"));
      return "done";
    });
    const built = jsonlEngine([node], { signal: host.signal });

    expect(await rejected(built.run())).toBe(failure);
    expect(stops(built)).toEqual([]);
    expect(logged(built, NodeStepRecord, "work")).toBe(false);
    expect(built.signal.reason).toBe(failure);
  });

  test("a refused RunStopped is a note on the host's reason, and the run rejects with the reason", async () => {
    const host = new AbortController();
    const reason: Error & { notes?: string[] } = new Error("stop now");
    const { node, entered } = waiter("work");
    const built = jsonlEngine([node], { signal: host.signal });

    const running = built.run();
    await entered;
    failSync = "once";
    host.abort(reason);

    expect(await rejected(running)).toBe(reason);
    expect(reason.notes).toEqual(["RunStopped was not recorded: fsync failed"]);
    expect(stops(built)).toEqual([]);
    expect(built.signal.reason).toBe(reason);
    expect(() => built.log.append(new NodeStartedRecord("late"))).toThrow(/refuses appends after a failed append/);
  });

  test("an append whose rollback also fails rejects with the AggregateError", async () => {
    const node = new FuncNode("work", () => {
      failSync = "always";
      return "done";
    });
    const built = jsonlEngine([node]);

    const error = (await rejected(built.run())) as AggregateError;
    failSync = null;
    expect(error).toBeInstanceOf(AggregateError);
    expect(error.errors).toEqual([failure, failure]);
    expect(stops(built)).toEqual([]);
  });

  test("a step refused before reaching the log leaves the log accepting appends", async () => {
    // The `NodeStepRecord` constructor refuses the step, so `JsonlRunLog.append` is never called.
    const node = new ScriptNode("work", (caps) => {
      caps.steps!.record({ when: new Date() } as never);
      return "done";
    });
    const built = jsonlEngine([node]);

    expect(await rejected(built.run())).toBeInstanceOf(RangeError);
    expect(stops(built)).toEqual([]);
    const sequence = built.log.nextSequence;
    expect(built.log.append(new NodeStartedRecord("late")).seq).toBe(sequence);
  });
});
