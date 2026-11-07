/**
 * The engine's half of `interrupt(run, inbox)`: steer inbox readers, their interrupt signal, the
 * delivery naming them, the live call view, and a waiting reader's wake.
 */
import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { CallClosed } from "../../src/events.ts";
import { type CompleteResult, Node, NodeKind, type NodeCaps, type NodeResult, Output, type SteerInbox } from "../../src/nodes/base.ts";
import { acceptsNull, type CallView, HostToolCall, NodeSteps } from "../../src/nodes/contracts.ts";
import { toolCallId } from "../../src/tools/context.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { noGrowth, RUN_ID } from "../helpers/run_state_fixtures.ts";
import { InterruptedRecord, NodeWokenRecord, RunStoppedRecord } from "../../src/runtime/log.ts";
import { typeFor } from "../../src/typesys/types.ts";
import {
  commitPrepared,
  prepareAppliedStop,
  prepareInterrupt,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeWaiting,
  prepareRunResumed,
} from "../../src/runtime/state.ts";
import { FlowEngine } from "../../src/runtime/engine.ts";
import { InboxRef } from "../../src/typesys/values.ts";
import { FakeHost, GateNode, inboxEngine, inboxFlow } from "../helpers/inbox_fixtures.ts";
import { succeed } from "../helpers/waits_fixtures.ts";

/** A node reading steer inbox `inbox` (when `open`): it opens it, waits for `release()`, then returns whether its interrupt signal aborted. */
class InterruptProbe extends Node {
  readonly kind = NodeKind.CODE;
  override readsSteerInbox = true;
  override readonly needsSteps = true;
  opened: SteerInbox | null = null;
  readonly #started = Promise.withResolvers<void>();
  readonly #gate = Promise.withResolvers<void>();

  readonly inbox: string;
  readonly open: boolean;

  constructor(id: string, inbox = "m", open = true) {
    super(id);
    this.inbox = inbox;
    this.open = open;
  }

  get started(): Promise<void> {
    return this.#started.promise;
  }

  release(): void {
    this.#gate.resolve();
  }

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    if (this.open) this.opened = caps.openSteerInbox!(this.inbox);
    this.#started.resolve();
    await this.#gate.promise;
    return new Output(this.opened?.interrupt.aborted ?? false);
  }
}

/** An `InterruptProbe` that starts, then waits for `letOpen()` before it opens its inbox. */
class GatedProbe extends InterruptProbe {
  readonly #beforeOpen = Promise.withResolvers<void>();
  readonly #open = Promise.withResolvers<void>();

  /** Resolves once the node has started and waits to open its inbox. */
  get startedBeforeOpen(): Promise<void> {
    return this.#beforeOpen.promise;
  }

  letOpen(): void {
    this.#open.resolve();
  }

  override async forward(inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    this.#beforeOpen.resolve();
    await this.#open.promise;
    return super.forward(inputs, caps);
  }
}

/** The run's `interrupted` records as `[inbox, nodes]`, in log order. */
const stopsOf = (engine: FlowEngine) =>
  engine.log
    .records()
    .flatMap(({ record }) => (record instanceof InterruptedRecord ? [[record.inbox, [...record.nodes]] as const] : []));

/** A host call node reading steer inbox `m`: it opens it and waits on its call. */
class SteerHost extends FakeHost {
  override readsSteerInbox = true;

  override forward(inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    caps.openSteerInbox!("m");
    return super.forward(inputs, caps);
  }
}

/** A serial run of START -> `nodes` in a chain -> END, START declaring inboxes `m` and `o`. */
function chain(...nodes: Node[]) {
  return chainWith({}, ...nodes);
}

/** As `chain`, with engine `options`. */
function chainWith(options: Parameters<typeof inboxEngine>[1], ...nodes: Node[]) {
  const ids = [START_ID, ...nodes.map((node) => node.id), END_ID];
  return inboxEngine(inboxFlow(["m", "o"], nodes, ids.slice(1).map((id, index) => [ids[index]!, id] as const)), options);
}

test("deliver(interrupt) names the opened running reader and aborts its signal in the same tick", async () => {
  const probe = new InterruptProbe("p");
  const engine = chain(probe);
  const running = engine.run();
  await probe.started;
  expect(probe.opened!.interrupt.aborted).toBe(false);
  expect(probe.opened!.interruptedBy()).toBeNull();
  const { seq } = engine.deliver({ kind: "interrupt", inbox: "m" });
  expect(seq).not.toBeNull();
  // Before any await.
  expect(probe.opened!.interrupt.aborted).toBe(true);
  expect(probe.opened!.interruptedBy()).toBe(seq);
  const record = engine.log.records().at(-1)!.record as InterruptedRecord;
  expect([record.inbox, [...record.nodes]]).toEqual(["m", ["p"]]);
  probe.release();
  await running;
  expect(engine.state.pool.store.get("p")!.toObject()).toBe(true);
});

test("a reader that hasn't opened its inbox yet isn't named: an empty Stop is recorded", async () => {
  const probe = new InterruptProbe("p", "m", false);
  const engine = chain(probe);
  const running = engine.run();
  await probe.started;
  const before = engine.log.records().length;
  const { seq, events, record } = engine.deliver({ kind: "interrupt", inbox: "m" });
  expect(seq).toBe(before);
  expect(events).toEqual([]);
  expect(record).toEqual(engine.log.records().at(-1)!.record);
  expect([...(record as InterruptedRecord).nodes]).toEqual([]);
  probe.release();
  await running;
});

test("a reader of another inbox isn't named: an empty Stop is recorded", async () => {
  const probe = new InterruptProbe("p", "o");
  const engine = chain(probe);
  const running = engine.run();
  await probe.started;
  const { seq, record } = engine.deliver({ kind: "interrupt", inbox: "m" });
  expect(seq).not.toBeNull();
  expect([...(record as InterruptedRecord).nodes]).toEqual([]);
  expect(probe.opened!.interrupt.aborted).toBe(false);
  probe.release();
  await running;
});

test("a node that opens with a pending interrupt gets an aborted signal and interruptedBy = the record seq", async () => {
  const probe = new InterruptProbe("p");
  const engine = chain(probe);
  const { log, state } = engine;
  // The run crashed with `p` in flight after an interrupt named it; the resume retries it.
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  succeed(log, state, START_ID, { m: new InboxRef("m"), o: new InboxRef("o") });
  commitPrepared(log, state, prepareNodeStarted(state, "p"));
  const interrupt = prepareInterrupt(state, "m", ["p"])!;
  commitPrepared(log, state, interrupt);
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  const running = engine.run();
  await probe.started;
  expect(probe.opened!.interrupt.aborted).toBe(true);
  expect(probe.opened!.interruptedBy()).toBe(interrupt.expectedSequence);
  probe.release();
  await running;
  // It holds the interrupt already, so the standing Stop appends no applied record.
  expect(stopsOf(engine)).toEqual([["m", ["p"]]]);
});

test("the reader leaves the engine's readers when its outcome commits", async () => {
  const probe = new InterruptProbe("p");
  const gate = new GateNode("g");
  const engine = chain(probe, gate);
  const running = engine.run();
  await probe.started;
  probe.release();
  await gate.started;
  // No take waits, so the Stop is recorded; no reader is left to name.
  const { seq, record } = engine.deliver({ kind: "interrupt", inbox: "m" });
  expect(seq).not.toBeNull();
  expect([...(record as InterruptedRecord).nodes]).toEqual([]);
  gate.release();
  await running;
});

test("an interrupt wakes a waiting reader, whose call it closed", async () => {
  const host = new SteerHost("h");
  const gate = new GateNode("g");
  const engine = inboxEngine(
    inboxFlow(["m"], [host, gate], [
      [START_ID, "h"],
      [START_ID, "g"],
      ["h", END_ID],
      ["g", END_ID],
    ]),
  );
  const running = engine.run();
  await gate.started;
  expect(engine.state.waits.waiting.get("h")!.steerInbox).toBe("m");
  const { events } = engine.deliver({ kind: "interrupt", inbox: "m" });
  expect(events).toEqual([new CallClosed({ callId: host.callId, reason: "interrupted" })]);
  gate.release();
  await running;
  const woken = engine.log.records().find((entry) => entry.record instanceof NodeWokenRecord)!.record as NodeWokenRecord;
  expect([woken.nodeId, [...woken.timedOut]]).toEqual(["h", []]);
  // The host node reads its call closed and ends with null; its interrupt is dropped with it.
  expect(engine.state.pool.store.get("h")!.toObject()).toBeNull();
  expect(engine.state.waits.interrupts.size).toBe(0);
});

/** A host call node that pends its call, waits for `release()`, then returns its call as `currentCalls()` and `calls` see it. */
class CallReader extends Node {
  readonly kind = NodeKind.TOOL;
  override readonly needsSteps = true;
  readonly tool = "lookup";
  readonly #started = Promise.withResolvers<void>();
  readonly #gate = Promise.withResolvers<void>();
  seen: { current: readonly CallView[]; snapshot: readonly CallView[] } | null = null;

  get started(): Promise<void> {
    return this.#started.promise;
  }

  release(): void {
    this.#gate.resolve();
  }

  get callId(): string {
    return toolCallId(RUN_ID, this.id);
  }

  override hostToolCall(): HostToolCall {
    return new HostToolCall(this.tool, null, [], new Map(), null);
  }

  override waitResultStep(call: CallView, result: CompleteResult): JsonObject {
    return { kind: "tool_result", call_id: call.callId, output: { kind: "str", value: String((result as { output: unknown }).output) } };
  }

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    const steps = caps.steps!;
    steps.record({ kind: "pending", call_id: this.callId, tool: this.tool, arguments: {} });
    this.#started.resolve();
    await this.#gate.promise;
    this.seen = { current: steps.currentCalls(), snapshot: steps.calls };
    return new Output("done");
  }
}

test("NodeSteps.currentCalls is calls outside a run", () => {
  const call: CallView = { callId: "c", tool: "ask", resultKey: "c", state: "open", closedReason: null, result: null, deadline: null };
  const steps = new NodeSteps(1, [], () => 1, [call]);
  expect(steps.currentCalls()).toBe(steps.calls);
});

test("in a run, currentCalls sees an answer that landed since the evaluation began; calls doesn't", async () => {
  const reader = new CallReader("h");
  const engine = chain(reader);
  const running = engine.run();
  await reader.started;
  engine.deliver({ kind: "complete", callId: reader.callId, result: { output: "yes" } });
  reader.release();
  await running;
  expect(reader.seen!.snapshot).toEqual([]);
  expect(reader.seen!.current.map((call) => [call.callId, call.state])).toEqual([[reader.callId, "answered"]]);
  expect(reader.seen!.current[0]!.result).toMatchObject({ kind: "tool_result", call_id: reader.callId });
  expect(Object.isFrozen(reader.seen!.current[0])).toBe(true);
});

test("acceptsNull: Optional(...) accepts null, str doesn't", () => {
  expect(acceptsNull(typeFor("Optional(str)", new Map()))).toBe(true);
  expect(acceptsNull(typeFor("str", new Map()))).toBe(false);
});

// --- The remembered Stop, applied when a reader opens ----------------------------------------- //

test("a reader that opens while a Stop stands gets an applied interrupted record, an aborted signal, and interruptedBy", async () => {
  const probe = new GatedProbe("p");
  const engine = chain(probe);
  const running = engine.run();
  await probe.startedBeforeOpen;
  const host = engine.deliver({ kind: "interrupt", inbox: "m" });
  expect([...(host.record as InterruptedRecord).nodes]).toEqual([]);
  probe.letOpen();
  await probe.started;
  const applied = engine.log.records().at(-1)!;
  expect(applied.seq).toBe(host.seq! + 1);
  expect([(applied.record as InterruptedRecord).inbox, [...(applied.record as InterruptedRecord).nodes]]).toEqual(["m", ["p"]]);
  expect(probe.opened!.interrupt.aborted).toBe(true);
  expect(probe.opened!.interruptedBy()).toBe(host.seq! + 1);
  expect(engine.state.waits.stops.get("m")).toEqual({ seq: host.seq! + 1, stopped: true });
  probe.release();
  await running;
  expect(engine.state.pool.store.get("p")!.toObject()).toBe(true);
  // END's success ends the window.
  expect(engine.state.waits.stops.size).toBe(0);
});

test("the applied record also names a waiting reader without one, and its CallClosed is published", async () => {
  // `h` started and waited on its call while the Stop stood, with no engine to apply it (the
  // records were written by hand); the next reader to open names it.
  const host = new SteerHost("h");
  const probe = new InterruptProbe("p");
  const events: unknown[] = [];
  const engine = inboxEngine(
    inboxFlow(["m"], [host, probe], [
      [START_ID, "h"],
      [START_ID, "p"],
      ["h", END_ID],
      ["p", END_ID],
    ]),
    { onEvent: (event) => void events.push(event) },
  );
  const { log, state } = engine;
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  succeed(log, state, START_ID, { m: new InboxRef("m") });
  commitPrepared(log, state, prepareInterrupt(state, "m", [])!);
  commitPrepared(log, state, prepareNodeStarted(state, "h"));
  commitPrepared(log, state, prepareNodeStep(state, "h", { kind: "pending", call_id: host.callId, tool: "lookup", arguments: {} }));
  commitPrepared(log, state, prepareNodeWaiting(state, "h", { callIds: [host.callId], steerInbox: "m" }));
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  const running = engine.run();
  await probe.started;
  expect(stopsOf(engine)).toEqual([
    ["m", []],
    ["m", ["p", "h"]],
  ]);
  expect(events).toContainEqual(new CallClosed({ callId: host.callId, reason: "interrupted" }));
  probe.release();
  await running;
  expect(engine.state.pool.store.get("h")!.toObject()).toBeNull();
});

test("no applied record once the run has a result", async () => {
  const probe = new GatedProbe("p");
  const engine = chain(probe);
  const running = engine.run();
  await probe.startedBeforeOpen;
  engine.deliver({ kind: "interrupt", inbox: "m" });
  engine.deliver({ kind: "cancel" });
  probe.letOpen();
  await probe.started;
  probe.release();
  await running;
  expect(stopsOf(engine)).toEqual([["m", []]]);
  expect(probe.opened!.interrupt.aborted).toBe(false);
});

test("no append during a stop drain; the rerun gets the Stop", async () => {
  const probe = new GatedProbe("p");
  const stop = new AbortController();
  const first = chainWith({ signal: stop.signal }, probe);
  const running = first.run();
  await probe.startedBeforeOpen;
  first.deliver({ kind: "interrupt", inbox: "m" });
  stop.abort(new Error("host stop"));
  probe.letOpen();
  await probe.started;
  expect(stopsOf(first)).toEqual([["m", []]]);
  expect(probe.opened!.interrupt.aborted).toBe(false);
  probe.release();
  await expect(running).rejects.toThrow("host stop");
  const { state, log } = first;
  expect(log.records().some(({ record }) => record instanceof RunStoppedRecord)).toBe(true);
  // The window outlives the stop: the rerun's reader (the same probe, its gates open) gets it.
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  const rerun = new FlowEngine(state, log, { growBuilder: noGrowth, numWorkers: 0 });
  await rerun.run();
  expect(stopsOf(rerun)).toEqual([
    ["m", []],
    ["m", ["p"]],
  ]);
  expect(probe.opened!.interrupt.aborted).toBe(true);
  expect(state.pool.store.get("p")!.toObject()).toBe(true);
});

test("two readers opening in one tick each get exactly one interrupt (pooled)", async () => {
  const p = new GatedProbe("p");
  const q = new GatedProbe("q");
  const engine = inboxEngine(
    inboxFlow(["m"], [p, q], [
      [START_ID, "p"],
      [START_ID, "q"],
      ["p", END_ID],
      ["q", END_ID],
    ]),
    { numWorkers: 2 },
  );
  const running = engine.run();
  await Promise.all([p.startedBeforeOpen, q.startedBeforeOpen]);
  engine.deliver({ kind: "interrupt", inbox: "m" });
  p.letOpen();
  q.letOpen();
  await Promise.all([p.started, q.started]);
  expect(stopsOf(engine)).toEqual([
    ["m", []],
    ["m", ["p"]],
    ["m", ["q"]],
  ]);
  expect([p.opened!.interrupt.aborted, q.opened!.interrupt.aborted]).toEqual([true, true]);
  p.release();
  q.release();
  await running;
});

test("a host interrupt while the Stop stands appends nothing; a reader opening after it still gets the applied record", async () => {
  const probe = new GatedProbe("p");
  const engine = chain(probe);
  const running = engine.run();
  await probe.startedBeforeOpen;
  expect(engine.deliver({ kind: "interrupt", inbox: "m" }).seq).not.toBeNull();
  expect(engine.deliver({ kind: "interrupt", inbox: "m" })).toEqual({ seq: null, events: [], record: null });
  probe.letOpen();
  await probe.started;
  expect(stopsOf(engine)).toEqual([
    ["m", []],
    ["m", ["p"]],
  ]);
  // The reader is being stopped: a third interrupt names no one.
  expect(engine.deliver({ kind: "interrupt", inbox: "m" }).seq).toBeNull();
  probe.release();
  await running;
});

/** A chain run of `probe` whose log, written by hand, recorded an empty Stop and then `probe`'s start, then crashed. */
function crashedAfterStart(probe: Node) {
  const engine = chain(probe);
  const { log, state } = engine;
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  succeed(log, state, START_ID, { m: new InboxRef("m"), o: new InboxRef("o") });
  commitPrepared(log, state, prepareInterrupt(state, "m", [])!);
  commitPrepared(log, state, prepareNodeStarted(state, probe.id));
  return engine;
}

test("a crash before the applied record: the resumed reader opens and gets it", async () => {
  const probe = new InterruptProbe("p");
  const engine = crashedAfterStart(probe);
  commitPrepared(engine.log, engine.state, prepareRunResumed(engine.state, { numWorkers: 0 }));
  const running = engine.run();
  await probe.started;
  expect(stopsOf(engine)).toEqual([
    ["m", []],
    ["m", ["p"]],
  ]);
  expect(probe.opened!.interrupt.aborted).toBe(true);
  probe.release();
  await running;
});

test("a crash after the applied record: the resumed reader holds the interrupt; no second record", async () => {
  const probe = new InterruptProbe("p");
  const engine = crashedAfterStart(probe);
  const { log, state } = engine;
  const applied = prepareAppliedStop(state, "m", "p");
  commitPrepared(log, state, applied);
  commitPrepared(log, state, prepareRunResumed(state, { numWorkers: 0 }));
  const running = engine.run();
  await probe.started;
  expect(stopsOf(engine)).toEqual([
    ["m", []],
    ["m", ["p"]],
  ]);
  expect(probe.opened!.interrupt.aborted).toBe(true);
  expect(probe.opened!.interruptedBy()).toBe(applied.expectedSequence);
  probe.release();
  await running;
});
