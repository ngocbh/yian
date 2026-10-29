/**
 * The chat's flusher: it sends the child events queued for `yian chat`'s assistant into the chat
 * flow's inbox, each as a message authored `yian`, at the moment the chat's state allows. The
 * chat (`inboxChat`) tells it when a turn begins and ends, when the prompt is up, when the user
 * interrupts and when a line is sent; the flusher sends, holds, announces and takes events back.
 *
 * Knows about: the chat's child events, the run boundary's delivery refusal and withdrawn events,
 * and the sanitizing of text printed on the terminal.
 */

import { DeliveryRefused, type WithdrawnEvent } from "../run.ts";
import { CHAT_WAKE, type ChildEvent, type EventQueue, toMessage } from "./chat_events.ts";
import { sanitize } from "./sanitize.ts";

/** The child events a chat sends into its inbox, and what tells it when (`inboxChat`). */
export interface ChatEvents {
  /** The events not yet sent. */
  queue: EventQueue;
  /** Whether one of the chat agent's tool calls is in flight: no event is sent meanwhile. */
  inFlight: { readonly busy: boolean };
  /** Gets a `CHAT_WAKE` event when an event is queued and when the last tool call in flight ends. */
  wake: EventTarget;
  /** Called when Ctrl-C or Esc interrupts a turn, to end the tool calls that wait meanwhile. */
  onInterrupt?: () => void;
}

/** What the flusher does its sending and printing through. */
export interface FlushIo {
  /** Whether the inbox takes messages (`Inbox(Message)`) rather than text (`Inbox(str)`). */
  message: boolean;
  /** Send `value` to the chat's inbox; resolves with the send's seq. */
  send: (value: unknown) => Promise<number>;
  /** Write `text` (one or more whole lines) to the chat's stderr. */
  write: (text: string) => Promise<void>;
  /** A send made at a wake failed other than by a refusal: the chat stops with `error`. */
  fault: (error: unknown) => void;
}

/** The text of a withdrawn message: its text parts joined. */
function messageText(value: unknown): string {
  const parts = (value as { parts?: readonly { type?: unknown; text?: unknown }[] }).parts ?? [];
  return parts.map((part) => (part.type === "text" && typeof part.text === "string" ? part.text : "")).join("");
}

/**
 * Sends a chat's queued child events into its inbox:
 *
 * - while a turn runs, at once at each wake, unless a tool call is in flight; the last one ending
 *   wakes it again;
 * - while the prompt is up, it takes the prompt down (`promptDown` says so), and the chat sends
 *   them (`sendQueued`) in a turn of their own;
 * - once held (`interrupted`), until the user's next line (`lineSent`), which they go before;
 *   meanwhile each event queued is announced above the prompt (`announce`), the prompt taken down
 *   for it when it is up.
 *
 * Esc's withdrawn events come back through `reclaim`; the events left at the end are printed
 * (`printLeft`).
 */
export class ChatFlusher {
  readonly #events: ChatEvents;
  readonly #io: FlushIo;
  readonly #onWake = () => this.wake();
  /** Set once the log holds the run's start: before it, a send has no log to go to. */
  #started = false;
  /** Set once the inbox is closed: nothing more is sent. */
  #closed = false;
  #turning = false;
  /** Set by an interrupt, cleared by the user's next line: events wait for that line. */
  #held = false;
  /** Takes the prompt down while it is up; `null` otherwise. */
  #takeDown: (() => void) | null = null;
  /** Set once a send failed other than by a refusal: the chat is stopping, and nothing more is sent. */
  #failed = false;
  /** Set when the flusher took the prompt down, to send or announce. */
  #tookDown = false;
  /** The sends made at wakes while the current turn runs, each resolving with its seqs. */
  #turnSends: Promise<number[]>[] = [];
  /** The last batch of sends: batches go one after another, so events keep their order. */
  #sending: Promise<unknown> = Promise.resolve();
  /** The events sent and not known read, by their send's seq: Esc puts them back as they were. */
  readonly #sent = new Map<number, ChildEvent>();
  /** The held events already announced. */
  readonly #announced = new WeakSet<ChildEvent>();

  constructor(events: ChatEvents, io: FlushIo) {
    this.#events = events;
    this.#io = io;
    events.wake.addEventListener(CHAT_WAKE, this.#onWake);
  }

  /** Whether events are held for the user's next line. */
  get held(): boolean {
    return this.#held;
  }

  /** The run's log holds its start: what is queued may go now. */
  started(): void {
    this.#started = true;
    this.wake();
  }

  /** A turn begins: what is queued goes now. */
  beginTurn(): void {
    this.#turning = true;
    this.#turnSends = [];
    this.wake();
  }

  /** The turn ends; resolves with the seqs of the sends made at wakes during it. */
  async endTurn(): Promise<number[]> {
    this.#turning = false;
    const sends = this.#turnSends;
    this.#turnSends = [];
    return (await Promise.all(sends)).flat();
  }

  /** The prompt is up; `takeDown` takes it down. */
  promptUp(takeDown: () => void): void {
    this.#takeDown = takeDown;
    this.#tookDown = false;
  }

  /** The prompt is down; returns whether the flusher took it down. */
  promptDown(): boolean {
    const took = this.#tookDown;
    this.#takeDown = null;
    this.#tookDown = false;
    return took;
  }

  /** The inbox is closed: nothing more is sent. */
  close(): void {
    this.#closed = true;
  }

  /** Ctrl-C or Esc: hold the events for the user's next line; during a turn, tell `onInterrupt`. */
  interrupted(turning: boolean): void {
    this.#held = true;
    if (turning) this.#events.onInterrupt?.();
  }

  /** Whether the chat should send what is queued in a turn of its own, rather than prompt. */
  ready(): boolean {
    return this.#started && !this.#closed && !this.#held && this.#events.queue.size > 0;
  }

  /** The user's line is about to be sent: the hold ends, and what is queued goes before it. */
  async lineSent(): Promise<void> {
    this.#held = false;
    if (this.#events.queue.size > 0) await this.sendQueued();
  }

  /**
   * An event was queued, or the last tool call in flight ended: send what is queued if a turn
   * runs with no tool call in flight; take the prompt down if it is up, to send it, or, when held,
   * to announce an event not yet announced.
   */
  wake(): void {
    if (!this.#started || this.#closed || this.#failed || this.#events.queue.size === 0) return;
    if (this.#held) {
      if (!this.#turning && this.#unannounced()) this.#promptDown();
      return;
    }
    if (!this.#turning) {
      this.#promptDown();
      return;
    }
    if (this.#events.inFlight.busy) return;
    this.#turnSends.push(
      this.sendQueued({ unlessHeld: true }).catch((error: unknown) => {
        this.#io.fault(error);
        return [];
      }),
    );
  }

  /**
   * Send every queued event, after the batches before; resolves with the seqs appended. A refused
   * send prints the event (`(not sent: ...)`). Another failure puts the events not sent back at
   * the front of the queue, for `printLeft`, and rejects; later batches then send nothing. With
   * `unlessHeld`, a batch that starts once the events are held sends nothing.
   */
  sendQueued(options: { unlessHeld?: boolean } = {}): Promise<number[]> {
    const batch = this.#sending.then(async () => {
      if (this.#failed || (options.unlessHeld === true && this.#held)) return [];
      const events = this.#events.queue.takeAll();
      const seqs: number[] = [];
      for (const [at, event] of events.entries()) {
        try {
          const seq = await this.#io.send(this.#io.message ? toMessage(event) : event.text);
          this.#sent.set(seq, event);
          seqs.push(seq);
        } catch (error) {
          if (!(error instanceof DeliveryRefused)) {
            this.#failed = true;
            this.#events.queue.requeueFront(events.slice(at));
            throw error;
          }
          await this.#printNotSent([event]);
        }
      }
      return seqs;
    });
    this.#sending = batch.catch(() => {});
    return batch;
  }

  /**
   * Announce each queued event not announced yet, while held, until none is left: the line `(child
   * event waiting; sent with your next message: <event>)`.
   */
  async announce(): Promise<void> {
    while (this.#held && this.#unannounced()) {
      const fresh = this.#events.queue.peek().filter((event) => !this.#announced.has(event));
      for (const event of fresh) {
        this.#announced.add(event);
        await this.#io.write(`(child event waiting; sent with your next message: ${sanitize(event.text)})\n`);
      }
    }
  }

  /**
   * Take the events among `withdrawn` (Esc's) back, and return the rest, the user's lines: to an
   * `Inbox(Message)` an event is a message authored `yian`, to an `Inbox(str)` a send this flusher
   * made. They go back to the front of the queue, sent again before the next line, or, once the
   * inbox is `closed`, are printed.
   */
  async reclaim(withdrawn: readonly WithdrawnEvent[], closed: boolean): Promise<WithdrawnEvent[]> {
    const isEvent = (event: WithdrawnEvent) =>
      this.#io.message ? (event.value as { author?: unknown }).author === "yian" : this.#sent.has(event.seq);
    const relayed = withdrawn.filter(isEvent);
    const typed = withdrawn.filter((event) => !relayed.includes(event));
    // One the chat didn't send goes back under no run.
    const events = relayed.map(
      (event) => this.#sent.get(event.seq) ?? { run: null, seq: 0, text: messageText(event.value) },
    );
    for (const event of relayed) this.#sent.delete(event.seq);
    if (events.length > 0) {
      if (closed) await this.#printNotSent(events);
      else {
        this.#events.queue.requeueFront(events);
        const count = `${events.length} child event${events.length === 1 ? "" : "s"}`;
        await this.#io.write(`(withdrew ${count}; sent again before your next message)\n`);
      }
    }
    return typed;
  }

  /** Whether `afterTurn` has anything to judge for a turn that sent `seqs`: it needs the log read. */
  tracking(seqs: readonly number[]): boolean {
    return seqs.length > 0 || this.#sent.size > 0;
  }

  /**
   * After a turn that left the run waiting with its last `run_waiting` at `lastWaiting`: forget
   * the sends at or before it, which the run has read; return whether any of `seqs` (the turn's
   * sends) came after it, sent once the engine's loop had ended, so the run should move again to
   * read it. Never while held.
   */
  afterTurn(seqs: readonly number[], lastWaiting: number): boolean {
    for (const seq of this.#sent.keys()) if (seq <= lastWaiting) this.#sent.delete(seq);
    return !this.#held && seqs.some((seq) => seq > lastWaiting);
  }

  /** Once the sends in progress end, print the events still queued (`(not sent: ...)`). */
  async printLeft(): Promise<void> {
    await this.#sending;
    await this.#printNotSent(this.#events.queue.takeAll());
  }

  /** Stop listening for wakes. */
  dispose(): void {
    this.#events.wake.removeEventListener(CHAT_WAKE, this.#onWake);
  }

  /** Whether a queued event isn't announced yet. */
  #unannounced(): boolean {
    return this.#events.queue.peek().some((event) => !this.#announced.has(event));
  }

  /** Take the prompt down, if it is up and not already taken down. */
  #promptDown(): void {
    if (this.#takeDown === null || this.#tookDown) return;
    this.#tookDown = true;
    this.#takeDown();
  }

  /** Print each of `events` as not sent. */
  async #printNotSent(events: readonly ChildEvent[]): Promise<void> {
    for (const event of events) await this.#io.write(`(not sent: ${sanitize(event.text)})\n`);
  }
}
