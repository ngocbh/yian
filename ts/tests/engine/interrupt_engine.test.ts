/**
 * TS-only: the engine's half of `interrupt(run, inbox)`: steer inbox readers, their interrupt
 * signal, the delivery naming them, the live call view, and a waiting reader's wake.
 */
import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { CallClosed } from "../../src/events.ts";
import { Node, NodeKind, type NodeCaps, type NodeResult, Output, type SteerInbox } from "../../src/nodes/base.ts";
import { InterruptedRecord, NodeWokenRecord } from "../../src/runtime/log.ts";
import { commitPrepared, prepareInterrupt, prepareNodeStarted, prepareRunResumed } from "../../src/runtime/state.ts";
import { InboxRef } from "../../src/typesys/values.ts";
import { FakeHost, GateNode, inboxEngine, inboxFlow } from "./inbox_fixtures.ts";
import { succeed } from "./waits_fixtures.ts";

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
  const ids = [START_ID, ...nodes.map((node) => node.id), END_ID];
  return inboxEngine(inboxFlow(["m", "o"], nodes, ids.slice(1).map((id, index) => [ids[index]!, id] as const)));
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

test("a reader that hasn't opened its inbox yet isn't named: the interrupt is a no-op", async () => {
  const probe = new InterruptProbe("p", "m", false);
  const engine = chain(probe);
  const running = engine.run();
  await probe.started;
  const before = engine.log.records().length;
  expect(engine.deliver({ kind: "interrupt", inbox: "m" })).toEqual({ seq: null, events: [] });
  expect(engine.log.records().length).toBe(before);
  probe.release();
  await running;
});

test("a reader of another inbox isn't named", async () => {
  const probe = new InterruptProbe("p", "o");
  const engine = chain(probe);
  const running = engine.run();
  await probe.started;
  expect(engine.deliver({ kind: "interrupt", inbox: "m" }).seq).toBeNull();
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
});

test("the reader leaves the engine's readers when its outcome commits", async () => {
  const probe = new InterruptProbe("p");
  const gate = new GateNode("g");
  const engine = chain(probe, gate);
  const running = engine.run();
  await probe.started;
  probe.release();
  await gate.started;
  expect(engine.deliver({ kind: "interrupt", inbox: "m" }).seq).toBeNull();
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
