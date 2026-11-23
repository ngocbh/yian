/**
 * The child events `yian chat`'s assistant hears about: a child run's question, reply or end,
 * queued as one line each until the chat sends it into its own flow's inbox as a message
 * authored `yian`; and the count of the assistant's tool calls in flight, which tells the chat
 * when such a send would land in the middle of one.
 *
 * Knows about: the chat's child statuses and their events (`chat_runs.ts`), and the host tool API.
 */

import { type FunctionTool, withExecute } from "../tools/function_tools.ts";
import { type ChildEvent, type ChildStatus, childEvent } from "./chat_runs.ts";

/** The type of the event a queue push or the last tool call ending dispatches on the wake target. */
export const CHAT_WAKE = "wake";

/** The message `event` is sent to the chat's inbox as: a user message authored `yian`. */
export function toMessage(event: ChildEvent): {
  role: "user";
  author: "yian";
  parts: [{ type: "text"; text: string }];
} {
  return { role: "user", author: "yian", parts: [{ type: "text", text: event.text }] };
}

/** A queued event, and whether it was taken before (sent, then withdrawn). */
interface Queued {
  event: ChildEvent;
  readonly taken: boolean;
}

/**
 * The child events not yet sent to the assistant, oldest first. A child's statuses are
 * cumulative (its open asks, its unread replies), so a newer one replaces that child's event
 * still queued; an event already taken once is never replaced.
 */
export class EventQueue {
  readonly #wake: EventTarget | null;
  readonly #markRead: ((run: string, seq: number) => void) | null;
  #queued: Queued[] = [];
  /** By run, the last seq marked seen or taken. */
  readonly #seen = new Map<string, number>();

  /**
   * @param options.wake Gets a `CHAT_WAKE` event each time a push queues something.
   * @param options.markRead Told each event `takeAll` takes, by run and seq: the child's replies
   *   up to it are read (`ChildRuns.markRead`).
   */
  constructor(
    options: { wake?: EventTarget | null; markRead?: ((run: string, seq: number) => void) | null } = {},
  ) {
    this.#wake = options.wake ?? null;
    this.#markRead = options.markRead ?? null;
  }

  /** How many events are queued. */
  get size(): number {
    return this.#queued.length;
  }

  /** Queue the event `status` is (`childEvent`), if any, and wake. */
  push(status: ChildStatus): void {
    const event = childEvent(status);
    if (event === null) return;
    const fresh = this.#queued.find((queued) => !queued.taken && queued.event.run === event.run);
    if (fresh !== undefined) fresh.event = event;
    else this.#queued.push({ event, taken: false });
    this.#wake?.dispatchEvent(new Event(CHAT_WAKE));
  }

  /** The last seq of `run` the assistant has seen: marked seen, or taken to be sent; 0 for none. */
  seen(run: string): number {
    return this.#seen.get(run) ?? 0;
  }

  /** Drop `run`'s events up to status `seq`: the assistant has seen that status. */
  markSeen(run: string, seq: number): void {
    this.#see(run, seq);
    this.#queued = this.#queued.filter((queued) => queued.event.run !== run || queued.event.seq > seq);
  }

  /** Take every queued event, oldest first, marking each one of a run read (`markRead`). */
  takeAll(): ChildEvent[] {
    const taken = this.#queued.map((queued) => queued.event);
    this.#queued = [];
    for (const event of taken) {
      if (event.run === null) continue;
      this.#see(event.run, event.seq);
      this.#markRead?.(event.run, event.seq);
    }
    return taken;
  }

  /** The queued events, oldest first, left queued. */
  peek(): readonly ChildEvent[] {
    return this.#queued.map((queued) => queued.event);
  }

  /** Put `events`, taken before and not delivered after all, back at the front, in their order. */
  requeueFront(events: readonly ChildEvent[]): void {
    this.#queued = [...events.map((event) => ({ event, taken: true })), ...this.#queued];
  }

  /** Advance `run`'s last seen seq to `seq`. */
  #see(run: string, seq: number): void {
    this.#seen.set(run, Math.max(seq, this.seen(run)));
  }
}

/** The count of tool calls in flight. */
export class InFlight {
  readonly #wake: EventTarget | null;
  #count = 0;

  /** @param wake Gets a `CHAT_WAKE` event each time the last call in flight ends. */
  constructor(wake: EventTarget | null = null) {
    this.#wake = wake;
  }

  /** Whether any call is in flight. */
  get busy(): boolean {
    return this.#count > 0;
  }

  /** A call starts. */
  enter(): void {
    this.#count += 1;
  }

  /** A call ends: the last one wakes. */
  leave(): void {
    this.#count -= 1;
    if (this.#count === 0) this.#wake?.dispatchEvent(new Event(CHAT_WAKE));
  }
}

/** `tools`, each counted in `counter` from its call until it returns or throws. */
export function trackInFlight(tools: readonly FunctionTool[], counter: InFlight): FunctionTool[] {
  return tools.map((each) =>
    withExecute(each, async (args, ctx) => {
      counter.enter();
      try {
        return await each.execute(args, ctx);
      } finally {
        counter.leave();
      }
    }),
  );
}
