/**
 * The chat's flusher, without a terminal or a run: its sends go to a fake inbox, its lines to a
 * list, and the chat's state is set through its hooks.
 */
import { describe, expect, test } from "bun:test";

import { EventQueue, InFlight } from "../../src/cli/chat_events.ts";
import { ChatFlusher } from "../../src/cli/chat_flush.ts";
import type { ChildStatus } from "../../src/cli/chat_runs.ts";
import { DeliveryRefused } from "../../src/run.ts";

/** A cancelled status of run `run` numbered `seq`. */
function cancelled(run: string, seq: number): ChildStatus {
  return { run, status: "cancelled", log: `/logs/${run}`, seq };
}

/** Let the pending promise callbacks and timers of this turn run. */
function tick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * A flusher over a fresh queue and in-flight count, whose inbox (`sent`) gives each send the next
 * seq from 100 once `gate` settles; `refuse` and `fail` make the send of a value with that text
 * refuse or throw.
 */
function flusher({ message = true, refuse = "", fail = "", gate = Promise.resolve() } = {}) {
  const wake = new EventTarget();
  const queue = new EventQueue({ wake });
  const inFlight = new InFlight(wake);
  let interrupts = 0;
  const sent: unknown[] = [];
  const lines: string[] = [];
  const faults: unknown[] = [];
  const shown: string[] = [];
  const textOf = (value: unknown) =>
    typeof value === "string" ? value : (value as { parts: { text: string }[] }).parts[0]!.text;
  const flush = new ChatFlusher(
    { queue, inFlight, wake, onInterrupt: () => void (interrupts += 1) },
    {
      message,
      send: async (value) => {
        await gate;
        const text = textOf(value);
        if (refuse !== "" && text.includes(refuse)) throw new DeliveryRefused("ended", "the run has ended");
        if (fail !== "" && text.includes(fail)) throw new Error("the log broke");
        sent.push(value);
        return 99 + sent.length;
      },
      write: async (text) => void lines.push(text),
      fault: (error) => void faults.push(error),
      sent: (event) => void shown.push(event.text),
    },
  );
  const texts = () => sent.map(textOf);
  return { flush, queue, inFlight, sent, texts, lines, faults, shown, interrupts: () => interrupts };
}

describe("ChatFlusher", () => {
  test("each event sent is shown once its send is appended; one refused isn't", async () => {
    const f = flusher({ refuse: "r2" });
    f.flush.started();
    f.queue.push(cancelled("r1", 1));
    f.queue.push(cancelled("r2", 1));
    await f.flush.sendQueued();
    expect(f.shown).toEqual(["run r1 was cancelled"]);
    expect(f.lines).toEqual(["(not sent: run r2 was cancelled)\n"]);
  });

  test("nothing goes before the run has started; starting sends nothing outside a turn or prompt", async () => {
    const f = flusher();
    f.queue.push(cancelled("r1", 1));
    f.flush.beginTurn();
    await tick();
    expect(f.sent).toEqual([]);
    f.flush.started();
    await tick();
    // Started mid-turn: what was queued goes at once.
    expect(f.texts()).toEqual(["run r1 was cancelled"]);
    expect(await f.flush.endTurn()).toEqual([100]);
  });

  test("mid-turn, an event goes at once as a yian message, unless a tool call is in flight", async () => {
    const f = flusher();
    f.flush.started();
    f.flush.beginTurn();
    f.inFlight.enter();
    f.queue.push(cancelled("r1", 1));
    await tick();
    expect(f.sent).toEqual([]);
    f.inFlight.leave();
    await tick();
    expect(f.sent).toEqual([{ role: "user", author: "yian", parts: [{ type: "text", text: "run r1 was cancelled" }] }]);
    expect(await f.flush.endTurn()).toEqual([100]);
  });

  test("at the prompt, an event takes it down once, and the chat sends it in a turn of its own", async () => {
    const f = flusher();
    f.flush.started();
    let downs = 0;
    f.flush.promptUp(() => void (downs += 1));
    f.queue.push(cancelled("r1", 1));
    f.queue.push(cancelled("r2", 2));
    expect(downs).toBe(1);
    expect(f.flush.promptDown()).toBe(true);
    expect(f.flush.ready()).toBe(true);
    expect(await f.flush.sendQueued()).toEqual([100, 101]);
    expect(f.flush.ready()).toBe(false);
    // A prompt the flusher didn't take down says so.
    f.flush.promptUp(() => {});
    expect(f.flush.promptDown()).toBe(false);
  });

  test("held: nothing is sent, each new event is announced once, and the next line sends them first", async () => {
    const f = flusher();
    f.flush.started();
    f.flush.beginTurn();
    f.flush.interrupted(true);
    expect(f.interrupts()).toBe(1);
    f.queue.push(cancelled("r1", 1));
    await tick();
    expect(f.sent).toEqual([]);
    expect(await f.flush.endTurn()).toEqual([]);
    expect(f.flush.ready()).toBe(false);
    await f.flush.announce();
    let downs = 0;
    f.flush.promptUp(() => void (downs += 1));
    f.queue.push(cancelled("r2", 2));
    expect(downs).toBe(1);
    f.flush.promptDown();
    await f.flush.announce();
    await f.flush.announce();
    expect(f.lines).toEqual([
      "(child event waiting; sent with your next message: run r1 was cancelled)\n",
      "(child event waiting; sent with your next message: run r2 was cancelled)\n",
    ]);
    // At the prompt, Esc holds without an interrupt.
    f.flush.interrupted(false);
    expect(f.interrupts()).toBe(1);
    await f.flush.lineSent();
    expect(f.flush.held).toBe(false);
    expect(f.texts()).toEqual(["run r1 was cancelled", "run r2 was cancelled"]);
  });

  test("a batch queued behind another sends nothing once the events are held", async () => {
    const { promise: gate, resolve: open } = Promise.withResolvers<void>();
    const f = flusher({ gate });
    f.flush.started();
    // A first batch, held up by its send.
    f.queue.push(cancelled("r1", 1));
    const first = f.flush.sendQueued();
    await tick();
    f.flush.beginTurn();
    f.queue.push(cancelled("r2", 2));
    f.flush.interrupted(true);
    open();
    expect(await first).toEqual([100]);
    expect(await f.flush.endTurn()).toEqual([]);
    expect(f.texts()).toEqual(["run r1 was cancelled"]);
    expect(f.queue.size).toBe(1);
  });

  test("a refused send is printed; a failed one requeues the rest for the end and faults", async () => {
    const refusing = flusher({ refuse: "r1" });
    refusing.flush.started();
    refusing.queue.push(cancelled("r1", 1));
    refusing.queue.push(cancelled("r2", 2));
    expect(await refusing.flush.sendQueued()).toEqual([100]);
    expect(refusing.lines).toEqual(["(not sent: run r1 was cancelled)\n"]);

    const failing = flusher({ fail: "r2" });
    failing.flush.started();
    failing.flush.beginTurn();
    failing.queue.push(cancelled("r1", 1));
    failing.queue.push(cancelled("r2", 2));
    failing.queue.push(cancelled("r3", 3));
    expect(await failing.flush.endTurn()).toEqual([]);
    expect(failing.faults.map((error) => (error as Error).message)).toEqual(["the log broke"]);
    expect(failing.texts()).toEqual(["run r1 was cancelled"]);
    await failing.flush.printLeft();
    expect(failing.lines).toEqual(["(not sent: run r2 was cancelled)\n", "(not sent: run r3 was cancelled)\n"]);
  });

  test("reclaim takes back messages authored yian, as they were, and leaves the user's", async () => {
    const f = flusher();
    f.flush.started();
    f.queue.push(cancelled("r1", 4));
    await f.flush.sendQueued();
    const yian = { role: "user", author: "yian", parts: [{ type: "text", text: "run r1 was cancelled" }] };
    const other = { role: "user", author: "yian", parts: [{ type: "text", text: "run r9 was cancelled" }] };
    const typed = { role: "user", parts: [{ type: "text", text: "hi" }] };
    const back = [
      { seq: 100, value: yian },
      { seq: 101, value: typed },
      { seq: 102, value: other },
    ];
    expect(await f.flush.reclaim(back, false)).toEqual([{ seq: 101, value: typed }]);
    expect(f.lines).toEqual(["(withdrew 2 child events; sent again before your next message)\n"]);
    expect(f.queue.peek()).toEqual([
      {
        run: "r1",
        seq: 4,
        text: "run r1 was cancelled",
        card: { status: "cancelled", verb: "was cancelled", body: "" },
      },
      { run: null, seq: 0, text: "run r9 was cancelled" },
    ]);
  });

  test("to an Inbox(str), reclaim knows its own sends by seq; after the close, it prints them", async () => {
    const f = flusher({ message: false });
    f.flush.started();
    f.queue.push(cancelled("r1", 4));
    await f.flush.sendQueued();
    const back = [
      { seq: 100, value: "run r1 was cancelled" },
      { seq: 101, value: "hi" },
    ];
    expect(await f.flush.reclaim(back, true)).toEqual([{ seq: 101, value: "hi" }]);
    expect(f.lines).toEqual(["(not sent: run r1 was cancelled)\n"]);
    expect(f.queue.size).toBe(0);
  });

  test("afterTurn forgets the sends the run read and asks for a move for one it didn't, unless held", async () => {
    const f = flusher({ message: false });
    f.flush.started();
    expect(f.flush.tracking([])).toBe(false);
    f.queue.push(cancelled("r1", 1));
    f.queue.push(cancelled("r2", 2));
    const seqs = await f.flush.sendQueued();
    expect(f.flush.tracking([])).toBe(true);
    expect(f.flush.afterTurn(seqs, 100)).toBe(true);
    // 100 was read, so Esc no longer knows it as the chat's; 101 still is.
    const back = [
      { seq: 100, value: "run r1 was cancelled" },
      { seq: 101, value: "run r2 was cancelled" },
    ];
    expect(await f.flush.reclaim(back, false)).toEqual([back[0]!]);
    expect(f.flush.tracking([])).toBe(false);
    f.flush.interrupted(false);
    expect(f.flush.afterTurn([105], 100)).toBe(false);
  });

  test("after dispose, a push wakes nothing; printLeft prints what is queued", async () => {
    const f = flusher();
    f.flush.started();
    f.flush.dispose();
    let downs = 0;
    f.flush.promptUp(() => void (downs += 1));
    f.queue.push(cancelled("r1", 1));
    expect(downs).toBe(0);
    await f.flush.printLeft();
    expect(f.lines).toEqual(["(not sent: run r1 was cancelled)\n"]);
  });
});
