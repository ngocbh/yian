import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { CallClosed, NodeWaiting, type RunEvent, RunCancelled, RunSucceeded, RunWaiting } from "../../src/events.ts";
import { Node, NodeKind, type NodeCaps, type NodeResult, Output, Pending } from "../../src/nodes/base.ts";
import { NodeStepRecord, NodeWokenRecord, PoolCommitRecord } from "../../src/runtime/log.ts";
import { withWarnings } from "../../src/platform.ts";
import { DeliveryRefused } from "../../src/runtime/state.ts";
import { FakeHost, FakeTake, GateNode, inboxEngine, inboxFlow, recordNames } from "../helpers/inbox_fixtures.ts";
import { RUN_ID } from "../helpers/run_state_fixtures.ts";

/** A run whose host call `ask` waits, with inbox `m` declared, already resolved waiting. */
async function waitingRun() {
  const host = new FakeHost("ask");
  const run = inboxEngine(inboxFlow(["m"], [host], [[START_ID, "ask"], ["ask", END_ID]]));
  await run.run();
  return { run, host };
}

/** The `DeliveryRefused` `deliver` throws for `op`. */
function refusal(run: Awaited<ReturnType<typeof waitingRun>>["run"], op: Parameters<typeof run.deliver>[0]): DeliveryRefused {
  try {
    run.deliver(op);
  } catch (error) {
    if (error instanceof DeliveryRefused) return error;
    throw error;
  }
  throw new Error("expected a DeliveryRefused");
}

test("deliver before run() is an Error, and nothing is appended", () => {
  const run = inboxEngine(inboxFlow(["m"], [new FakeHost("ask")], [[START_ID, "ask"], ["ask", END_ID]]));
  const before = run.log.nextSequence;
  expect(() => run.deliver({ kind: "send", inbox: "m", event: "hi" })).toThrow("FlowEngine.deliver needs run() first");
  expect(run.log.nextSequence).toBe(before);
});

test("a refused delivery appends nothing", async () => {
  const { run } = await waitingRun();
  const before = run.log.nextSequence;
  expect(refusal(run, { kind: "send", inbox: "nope", event: "hi" }).reason).toBe("unknown_inbox");
  expect(run.log.nextSequence).toBe(before);
});

test("a send to a declared inbox after run() resolved waiting appends it", async () => {
  const { run } = await waitingRun();
  const out = run.deliver({ kind: "send", inbox: "m", event: "hi" });
  expect(out.seq).toBe(run.log.nextSequence - 1);
  const record = run.log.records()[out.seq!]!;
  expect(record.seq).toBe(out.seq!);
  expect(record.record).toBeInstanceOf(PoolCommitRecord);
});

test("a keyed send lands once: a repeat gives its seq and appends nothing; another value is key_reused", async () => {
  const { run } = await waitingRun();
  const first = run.deliver({ kind: "send", inbox: "m", event: "hi", key: "k1" });
  const after = run.log.nextSequence;
  expect(run.deliver({ kind: "send", inbox: "m", event: "hi", key: "k1" })).toEqual({ seq: first.seq, events: [], record: null });
  expect(refusal(run, { kind: "send", inbox: "m", event: "bye", key: "k1" }).reason).toBe("key_reused");
  expect(run.log.nextSequence).toBe(after);
  expect((run.log.records()[first.seq!]!.record as PoolCommitRecord).sendKey).toBe("k1");
});

test("complete answers once; an identical repeat is a no-op and a different one is call_settled", async () => {
  const { run, host } = await waitingRun();
  const first = run.deliver({ kind: "complete", callId: host.callId, result: { output: "yes" } });
  expect(first.seq).toBe(run.log.nextSequence - 1);
  expect(run.deliver({ kind: "complete", callId: host.callId, result: { output: "yes" } })).toEqual({ seq: null, events: [], record: null });
  const settled = refusal(run, { kind: "complete", callId: host.callId, result: { output: "no" } });
  expect([settled.reason, settled.callState]).toEqual(["call_settled", "answered"]);
});

test("release refuses a host call its tool pended, which wasn't held, and appends nothing", async () => {
  const { run, host } = await waitingRun();
  const before = run.log.nextSequence;
  expect(refusal(run, { kind: "release", callId: host.callId, note: "ok" }).reason).toBe("not_held");
  expect(refusal(run, { kind: "release", callId: "nobody" }).reason).toBe("unknown_call");
  expect(run.log.nextSequence).toBe(before);
});

/**
 * An agent's records without its loop: its first evaluation records a reply calling `lookup` once
 * and the host's hold of that call, and waits; a later one notes the call's state in `seen` and
 * records the call's result.
 */
class Holder extends Node {
  readonly kind = NodeKind.AGENT;
  override readonly needsSteps = true;
  readonly seen: string[] = [];

  get callId(): string {
    return `${RUN_ID}/${this.id}/1/0`;
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    const steps = caps.steps!;
    const view = steps.calls.find((call) => call.callId === this.callId);
    if (view === undefined) {
      steps.record({ kind: "model_call", output: { parts: [{ type: "tool_call", id: "c0", name: "lookup", arguments: {} }] } });
      const source = { kind: "host", group: null, tool: "lookup" };
      steps.record({ kind: "pending", call_id: this.callId, tool: "lookup", tool_call_id: "c0", reply: 1n, index: 0n, held: true, source, group_end: 0n });
      return new Pending([this.callId]);
    }
    this.seen.push(view.state);
    steps.record({ kind: "tool_result", call_id: "c0", started_at: null, output: "ran", is_error: false });
    return new Output("ran");
  }
}

test("release appends the hold's released step, publishing nothing, and its waiting node wakes to it", async () => {
  const holder = new Holder("a");
  const delivered: { seq: number | null; events: readonly RunEvent[] }[] = [];
  const run = inboxEngine(inboxFlow([], [holder], [[START_ID, "a"], ["a", END_ID]]), {
    onEvent: (event: RunEvent) => {
      if (event instanceof RunWaiting) delivered.push(run.deliver({ kind: "release", callId: holder.callId, note: "ok" }));
    },
  });
  await run.run();
  expect(run.state.result).toBeInstanceOf(RunSucceeded);
  const [{ seq, events }] = delivered as [(typeof delivered)[number]];
  const released = run.log.records()[seq!]!.record as NodeStepRecord;
  expect([released.nodeId, released.step, events]).toEqual(["a", { kind: "released", call_id: holder.callId, note: "ok" }, []]);
  expect(run.log.records()[seq! + 1]!.record).toBeInstanceOf(NodeWokenRecord);
  expect(holder.seen).toEqual(["released"]);
});

test("closing a closed inbox is a no-op", async () => {
  const { run } = await waitingRun();
  expect(run.deliver({ kind: "close", inbox: "m" }).seq).toBe(run.log.nextSequence - 1);
  expect(run.deliver({ kind: "close", inbox: "m" })).toEqual({ seq: null, events: [], record: null });
});

test.each([0, 2])("a cancel during a live run ends it cancelled and aborts running nodes (%p workers)", async (numWorkers) => {
  const host = new FakeHost("ask");
  const gate = new GateNode("gate");
  const waited = Promise.withResolvers<void>();
  const run = inboxEngine(
    inboxFlow([], [host, gate], [[START_ID, "ask"], [START_ID, "gate"], ["ask", END_ID], ["gate", END_ID]]),
    {
      numWorkers,
      onEvent: (event: RunEvent) => {
        if (event instanceof NodeWaiting && event.nodeId === "ask") waited.resolve();
      },
    },
  );
  const running = run.run();
  await Promise.all([gate.started, waited.promise]);
  const out = run.deliver({ kind: "cancel" });
  const events = await running;
  expect(out.events).toEqual([new RunCancelled(), new CallClosed({ callId: host.callId, reason: "cancelled" })]);
  expect(gate.aborted).toBe(true);
  expect(run.state.result).toBeInstanceOf(RunCancelled);
  expect(events.at(-2)).toBeInstanceOf(RunCancelled);
  const names = recordNames(run.log);
  expect(names.at(-1)).toBe("RunCancelledRecord");
  expect(names).not.toContain("RunStoppedRecord");
  expect(refusal(run, { kind: "complete", callId: host.callId, result: { output: "late" } }).reason).toBe("ended");
});

test("after the loop has ended, a throwing onEvent rethrows, the record stays, and no run_stopped follows", async () => {
  const failure = new Error("observer failed");
  let armed = false;
  const run = inboxEngine(inboxFlow(["m"], [new FakeHost("ask")], [[START_ID, "ask"], ["ask", END_ID]]), {
    onEvent: () => {
      if (armed) throw failure;
    },
  });
  const events = await run.run();
  expect(events.at(-1)).toBeInstanceOf(RunWaiting);
  armed = true;
  const before = run.log.nextSequence;
  // A send has no observers; a cancel's are RunCancelled and its closed calls.
  expect(() => run.deliver({ kind: "cancel" })).toThrow(failure);
  expect(run.log.nextSequence).toBe(before + 1);
  expect(recordNames(run.log).at(-1)).toBe("RunCancelledRecord");
});

test("after the loop has ended, a rejection from onEvent's promise is ignored", async () => {
  let armed = false;
  const run = inboxEngine(inboxFlow(["m"], [new FakeHost("ask")], [[START_ID, "ask"], ["ask", END_ID]]), {
    onEvent: () => (armed ? Promise.reject(new Error("late observer failed")) : undefined),
  });
  await run.run();
  armed = true;
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown): void => void unhandled.push(reason);
  process.on("unhandledRejection", onUnhandled);
  try {
    expect(run.deliver({ kind: "cancel" }).events[0]).toBeInstanceOf(RunCancelled);
    await new Promise((resolve) => setImmediate(resolve));
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }
  expect(unhandled).toEqual([]);
});

test("an observer's delivery keeps run()'s events in log order, ending with the terminal", async () => {
  const run: ReturnType<typeof inboxEngine> = inboxEngine(
    inboxFlow(["m"], [new FakeTake("heard", "m")], [[START_ID, "heard"], ["heard", END_ID]]),
    {
      onEvent: (event: RunEvent) => {
        if (event instanceof RunWaiting) run.deliver({ kind: "cancel" });
      },
    },
  );
  const events = await run.run();
  expect(events.slice(-3).map((event) => event.constructor.name)).toEqual(["NodeWaiting", "RunWaiting", "RunCancelled"]);
  expect(recordNames(run.log).slice(-3)).toEqual(["NodeWaitingRecord", "RunWaitingRecord", "RunCancelledRecord"]);
});

/** A node that opens steer inbox `m`, adds `listeners` to its sends, calls `opened`, and returns once `until` resolves. */
class SendWatcher extends Node {
  readonly kind = NodeKind.CODE;
  override readsSteerInbox = true;
  override readonly needsSteps = true;
  readonly listeners: readonly (() => void)[];
  readonly opened: () => void;
  readonly until: Promise<void>;

  constructor(id: string, listeners: readonly (() => void)[], opened: () => void, until: Promise<void>) {
    super(id);
    this.listeners = listeners;
    this.opened = opened;
    this.until = until;
  }

  override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
    const steer = caps.openSteerInbox!("m");
    for (const listener of this.listeners) steer.onSend(listener);
    this.opened();
    await this.until;
    return new Output("done");
  }
}

test("a send to a running node's steer inbox calls its send listeners; one that throws is warned about, and the send stands", async () => {
  const heard: string[] = [];
  const opened = Promise.withResolvers<void>();
  const until = Promise.withResolvers<void>();
  const broken = () => {
    throw new Error("broke");
  };
  const watcher = new SendWatcher("chat", [broken, () => heard.push("sent")], opened.resolve, until.promise);
  const run = inboxEngine(inboxFlow(["m", "other"], [watcher], [[START_ID, "chat"], ["chat", END_ID]]));
  const running = run.run();
  await opened.promise;
  const warnings: string[] = [];
  const [other, sent] = withWarnings(
    (message) => warnings.push(message),
    () => [run.deliver({ kind: "send", inbox: "other", event: "x" }), run.deliver({ kind: "send", inbox: "m", event: "a" })],
  );
  until.resolve();
  await running;

  expect(heard).toEqual(["sent"]);
  expect(warnings).toEqual(["node 'chat': a listener to its steer inbox's sends threw: broke"]);
  expect(sent.seq).toBe(other.seq! + 1);
  expect(run.log.records()[sent.seq!]!.record).toBeInstanceOf(PoolCommitRecord);
  expect(run.state.result).toBeInstanceOf(RunSucceeded);
});

