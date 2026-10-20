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
 * Not ported: `test_public_run_flow_signature_has_no_stop_event`: TS reads no signatures, and
 * `runFlow` takes `{ signal }`.
 *
 * TS-only, the engine's stop (serial and pooled): a host abort records one `RunStopped` (`signal`)
 * listing the in-flight nodes in dispatch order before the drain; a synchronous `onEvent` throw, a
 * rejected `onEvent` promise, a throwing `onStream` the node catches and a throwing steering method
 * record a `callback` stop; the message is an `Error`'s message or a non-`Error` reason's text; an
 * already-aborted signal rejects before the first start with the log unchanged; nothing is recorded
 * after a terminal result, after an earlier stop in the segment, for a host fault (an invalid step,
 * a malformed contract, an opener fault), or for a late rejection after `run()` settled. Pooled
 * only: a host abort or a late `onEvent` rejection during a host fault's drain changes nothing; a
 * host abort during a `RunFailed`'s drain is ignored and `run()` returns normally; an outcome
 * settling after a callback stop is discarded; a callback throw during a node's synchronous start
 * starts no further node. No test leaves an unhandled rejection or an uncaught exception.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { NodeStarted, RunFailed, type RunEvent, RunSucceeded, StreamChunk } from "../../src/events.ts";
import { MemoryScope } from "../../src/memory/contracts.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import type { OutputContract } from "../../src/nodes/contracts.ts";
import type { FlowEngine, FlowEngineOptions } from "../../src/runtime/engine.ts";
import { evalNode, type MemoryContextOpener } from "../../src/runtime/eval_node.ts";
import { NodeStartedRecord, NodeSucceededRecord, RunStoppedRecord } from "../../src/runtime/log.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { aborted, engine, FailNode, FuncNode, graph, rejected } from "./run_state_fixtures.ts";

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
  override needsSteering = true;
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
 * Return a node that blocks until its signal aborts, then outputs what `onAbort` returns (by
 * default it rejects with the signal's reason), and a promise that resolves once it blocks.
 */
function waiter(
  nodeId: string,
  onAbort: (signal: AbortSignal) => unknown = (signal) => {
    throw signal.reason;
  },
): { node: ScriptNode; entered: Promise<void> } {
  const { promise, resolve } = Promise.withResolvers<void>();
  const node = new ScriptNode(nodeId, async (caps) => {
    resolve();
    await aborted(caps.signal!).catch(() => {});
    return await onAbort(caps.signal!);
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
        caps.steering!.drain("work");
      } catch {}
      return "done";
    });
    const steering = {
      interruptSignal: () => new AbortController().signal,
      drain: (): string[] => {
        throw error;
      },
    };
    const built = fanEngine([node], { numWorkers, steering });

    expect(await rejected(built.run())).toBe(error);
    expect(stops(built)).toEqual([new RunStoppedRecord("callback", "inbox failed", ["work"])]);
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

  test("a malformed contract", async () => {
    const built = fanEngine([new FuncNode("work", () => "done")], { numWorkers });
    built.state.flow.nodes.get("work")!.outputContract = Type.scalar(ValueKind.STRING) as unknown as OutputContract;

    expect(await rejected(built.run())).toBeInstanceOf(TypeError);
    expect(stops(built)).toEqual([]);
  });

  test("an opener fault", async () => {
    const error = new Error("opening failed");
    const node = new MemoryScriptNode("work", async (caps) => {
      await caps.openMemoryContext!("system", "prompt", {});
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
