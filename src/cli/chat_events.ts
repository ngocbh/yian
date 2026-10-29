/**
 * The child events `yian chat`'s assistant hears about: a child run's question, reply or end,
 * queued as one line each until the chat sends it into its own flow's inbox as a message
 * authored `yian`; and the count of the assistant's tool calls in flight, which tells the chat
 * when such a send would land in the middle of one.
 *
 * Knows about: the chat's child statuses and the host tool API.
 */

import { TAKE } from "../tools/context.ts";
import { FunctionTool } from "../tools/function_tools.ts";
import type { ChildStatus } from "./chat_runs.ts";

/** The most characters of a child event's line. */
const EVENT_BODY_CHARS = 2000;

/** The type of the event a queue push or the last tool call ending dispatches on the wake target. */
export const CHAT_WAKE = "wake";

/** One child event: the line the assistant reads, and the status it came from. */
export interface ChildEvent {
  readonly run: string;
  readonly seq: number;
  readonly text: string;
}

/** What a waiting status waits on, as far as `childEvent` reads it. */
interface Wait {
  readonly tool?: unknown;
  readonly inbox?: unknown;
  readonly settled?: unknown;
}

/**
 * The event `status` is for the assistant, or `null` when it says nothing to act on (a run
 * `running`, or waiting only for a timer with nothing said). The line names the run, then what
 * it now does (asks, with each question's call id; waits for a message on an inbox; succeeded,
 * failed or was cancelled), then what it said; child text is JSON-quoted, so the line stays one
 * line, and the line is cut at `EVENT_BODY_CHARS` with a note.
 */
export function childEvent(status: ChildStatus): ChildEvent | null {
  if (status.status === "running") return null;
  const clauses: string[] = [];
  const quote = (value: unknown) => JSON.stringify(value) ?? "null";
  if (status.status === "succeeded") clauses.push(`succeeded with output ${quote(status.output ?? null)}`);
  else if (status.status === "cancelled") clauses.push("was cancelled");
  else if (status.status === "failed") {
    const failure = status.failure;
    clauses.push(
      failure === undefined
        ? `failed: ${quote(status.note ?? "")}`
        : `failed (${failure.code}${failure.node === null ? "" : ` at ${failure.node}`}): ${quote(failure.message)}`,
    );
  } else if ((status.asks ?? []).length > 0) {
    clauses.push(`asks ${status.asks!.map((ask) => `(call ${ask.call}): ${quote(ask.question)}`).join(", ")}`);
  } else {
    const waits = (status.waits ?? []) as readonly Wait[];
    const inbox = waits.find((wait) => wait.tool === TAKE && wait.settled !== true && typeof wait.inbox === "string");
    if (inbox !== undefined) clauses.push(`waits for a message on inbox ${quote(inbox.inbox)}`);
  }
  if ((status.replies ?? []).length > 0) clauses.push(`said ${status.replies!.map(quote).join(", ")}`);
  if (clauses.length === 0) return null;
  let text = `run ${status.run} ${clauses.join("; ")}`;
  if (text.length > EVENT_BODY_CHARS) text = `${text.slice(0, EVENT_BODY_CHARS)}... (cut; flow_status shows all)`;
  return { run: status.run, seq: status.seq, text };
}

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

  /** Drop `run`'s events up to status `seq`: the assistant has seen that status. */
  markSeen(run: string, seq: number): void {
    this.#queued = this.#queued.filter((queued) => queued.event.run !== run || queued.event.seq > seq);
  }

  /** Take every queued event, oldest first, marking each read (`markRead`). */
  takeAll(): ChildEvent[] {
    const taken = this.#queued.map((queued) => queued.event);
    this.#queued = [];
    for (const event of taken) this.#markRead?.(event.run, event.seq);
    return taken;
  }

  /** Put `events`, taken before and not delivered after all, back at the front, in their order. */
  requeueFront(events: readonly ChildEvent[]): void {
    this.#queued = [...events.map((event) => ({ event, taken: true })), ...this.#queued];
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
  return tools.map(
    (each) =>
      new FunctionTool({
        name: each.name,
        description: each.description,
        parametersJsonSchema: each.parametersJsonSchema,
        isConcurrencySafe: each.isConcurrencySafe,
        parametersValidator: each.parametersValidator,
        outputJsonSchema: each.outputJsonSchema,
        outputValidator: each.outputValidator,
        execute: async (args, ctx) => {
          counter.enter();
          try {
            return await each.execute(args, ctx);
          } finally {
            counter.leave();
          }
        },
      }),
  );
}
