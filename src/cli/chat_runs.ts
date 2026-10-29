/**
 * The runs `yian chat`'s assistant starts: each child flow runs in the background under one
 * driver, which is the only caller of `runFlow` and `advance` on its log, and publishes the
 * child's status each time a pass of it settles differently.
 *
 * A child moves only when its driver runs a pass (`runFlow`, then `advance`): `advance` refuses a
 * run whose engine still loops, so a second caller would race the first. A delivery reaches a
 * live engine directly or is appended to the log offline; either way the deliverer then nudges
 * the driver, which runs one more pass once the current one ends. Nothing else wakes a waiting
 * run at its `wakeAt`, so between passes the driver sleeps until then.
 *
 * A child's `ask` returns `pending()`, so the child waits durably for the user's answer, which the
 * chat relays with `answer`. The children's stop is their own, tied to the chat's exit by
 * `dispose`, never to the call that started them.
 *
 * Knows about: the run boundary (`runFlow`, `advance`, the deliveries, `pendingCalls`,
 * `runLogPath`), the waits' `wakeAt` and the record-time reader, compose, the run events, the tool
 * registry and the `pending` marker, skills, the CLI's key preflight (`startup.ts`).
 */

import { dirname } from "node:path";
import { isDeepStrictEqual } from "node:util";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import type { Env } from "../defaults.ts";
import type { ModelAccess } from "../llm_clients/factory.ts";
import type { Scheduler } from "../platform.ts";
import { NodeSucceeded, type RunEvent, RunStarted } from "../events.ts";
import {
  advance,
  cancel,
  close,
  complete,
  type DeliveryOptions,
  DeliveryRefused,
  pendingCalls,
  type RunResult,
  runFlow,
  runLogPath,
  send,
} from "../run.ts";
import { recordTimeMs } from "../runtime/log.ts";
import { wakeAt } from "../runtime/waits.ts";
import type { Skill } from "../skills/catalog.ts";
import { pending, TAKE } from "../tools/context.ts";
import { registeredTools, type ToolEntry } from "../tools/registry.ts";
import { builtinType } from "../typesys/types.ts";
import { isInboxType, typeEquals, type Value, writeAsText } from "../typesys/values.ts";
import { removeEmptyDir } from "./run.ts";
import { sleepUntil } from "./utils.ts";

/** Where a child run stands, as the chat's assistant is told. */
export interface ChildStatus {
  /** The child's id in the chat (`r1`, `r2`, ...). */
  run: string;
  status: "running" | "waiting" | "succeeded" | "failed" | "cancelled";
  /** The flow's output, once it succeeded. */
  output?: unknown;
  /** Why it failed, once it failed. */
  failure?: { code: string; message: string; node: string | null };
  /** What a waiting child waits on (`WaitInfo`s). */
  waits?: unknown[];
  /** When a waiting child next wakes by itself, as a record time, or `null` when only a delivery wakes it. */
  wake_at?: string | null;
  /** The questions a waiting child asks, each answered by its call id. */
  asks?: { call: string; question: string }[];
  /** What the child's nodes named `reply` said during the pass this status ends. */
  replies?: string[];
  /**
   * Something the chat couldn't do with the run, in words: a delivery the run refused (only on the
   * status that call returns), or the error that stopped the driver of a failed child.
   */
  note?: string;
  /** The child's run log. */
  log: string;
  /**
   * The status's number: every status any child publishes gets the next one, so a reader marks
   * the statuses it has seen by the last number it read.
   */
  seq: number;
}

/** What a `ChildRuns` needs from the chat. */
export interface ChildRunsOptions {
  /** The folder the chat's flows are confined to: their CODE files must resolve inside it. */
  root: string;
  /** The folder each child's run folder goes in. */
  logDir: string;
  env: Env;
  /** The `--provider`/`--model` layer, possibly empty. */
  llmConfig: Readonly<Record<string, string>>;
  /** For tests: the model builder AGENT nodes use. */
  models?: ModelAccess | null;
  /** The clock the children's records and waits run on. */
  scheduler: Scheduler;
  /** The chat's own run id: each child's is `<chatRunId>-r<n>`, and its parent is the chat. */
  chatRunId: string;
  /** The skills a child's flow grants, as `yian run` reads them, or `null` for none. */
  skillsFor: (flow: ComposedFlow) => ReadonlyMap<string, Skill> | null;
  /** @internal For tests: the `advance` the drivers call. */
  advance?: typeof advance;
}

/** A child left waiting when the chat ends, and what it waits on. */
export interface WaitingChild {
  run: string;
  /** `question` when it asks something, else `inbox` when it waits for a message, else `timer`. */
  kind: "question" | "inbox" | "timer";
  log: string;
}

/** One child and its driver's state. */
interface Child {
  readonly id: string;
  readonly flow: ComposedFlow;
  readonly log: string;
  /** The last published status. */
  status: ChildStatus;
  /** The `next` calls waiting for a status after the given number. */
  readonly waiters: { after: number; resolve: (status: ChildStatus) => void }[];
  /** Set by a nudge: the driver runs one more pass before it rests. */
  dirty: boolean;
  /** The driver's rest between passes, which a nudge ends; `null` while it doesn't rest. */
  rest: AbortController | null;
  /** Whether a pass runs. */
  passing: boolean;
  /** What the child's `reply` nodes said during the current pass. */
  readonly replies: string[];
  /** The child's own stop: it stops the driver and leaves the log as it is. */
  readonly halt: AbortController;
  /** Settles once the driver has stopped; `null` until it starts. */
  done: Promise<void> | null;
}

/** The terminal statuses: no pass follows them. */
const ENDED: ReadonlySet<ChildStatus["status"]> = new Set(["succeeded", "failed", "cancelled"]);

/** Whether `event` ends a node named `reply`, at any depth: what an inbox child says each turn. */
function isReply(event: RunEvent): event is NodeSucceeded {
  return event instanceof NodeSucceeded && event.nodeId.slice(event.nodeId.lastIndexOf("/") + 1) === "reply";
}

/** The question text of an ask's recorded arguments: a TOOL `ask`'s prompt, an agent's question. */
function questionText(args: Readonly<Record<string, unknown>> | null): string {
  const text = args?.["prompt"] ?? args?.["question"];
  if (typeof text === "string" && text !== "") return text;
  // A form asks its questions as a list.
  return args?.["questions"] == null ? "" : JSON.stringify(args["questions"]);
}

/**
 * The chat's child runs: start each in the background, read its status as it changes, and relay
 * the user's answers and messages to it.
 */
export class ChildRuns {
  readonly #options: ChildRunsOptions;
  readonly #children = new Map<string, Child>();
  #seq = 0;
  #started = 0;
  #disposed = false;

  constructor(options: ChildRunsOptions) {
    this.#options = options;
  }

  /**
   * Start `flow` with `inputs` as the next child (`r1`, `r2`, ...): return its id and log once the
   * log holds the run's start, while the run goes on in the background. `options.signal` is the
   * caller's: aborting it before the start rejects, and after it, does nothing to the child.
   *
   * @throws {Error} When the chat was disposed; the signal's reason once it aborted; what `runFlow`
   *   throws before the run starts (an input it doesn't declare, say), with no log left behind.
   */
  async start(
    flow: ComposedFlow,
    inputs: Readonly<Record<string, unknown>>,
    options: { signal?: AbortSignal | null } = {},
  ): Promise<{ id: string; log: string }> {
    if (this.#disposed) throw new Error("the chat's runs were disposed");
    options.signal?.throwIfAborted();
    const id = `r${++this.#started}`;
    const runId = `${this.#options.chatRunId}-${id}`;
    const log = runLogPath(this.#options.logDir, runId);
    const child: Child = {
      id,
      flow,
      log,
      status: { run: id, status: "running", log, seq: ++this.#seq },
      waiters: [],
      dirty: false,
      rest: null,
      passing: false,
      replies: [],
      halt: new AbortController(),
      done: null,
    };
    this.#children.set(id, child);
    const started = Promise.withResolvers<void>();
    child.done = this.#drive(child, runId, inputs, started);
    try {
      await started.promise;
    } catch (error) {
      this.#children.delete(id);
      throw error;
    }
    return { id, log };
  }

  /**
   * Return child `id`'s last published status.
   *
   * @throws {RangeError} When no child has the id.
   */
  status(id: string): ChildStatus {
    return this.#child(id).status;
  }

  /**
   * Resolve with child `id`'s first status numbered after `after` (default: its current one's), or
   * with its current status once its driver has stopped and none will follow, or `cancelWaits`
   * is called.
   *
   * @throws {RangeError} When no child has the id.
   */
  next(id: string, after?: number): Promise<ChildStatus> {
    const child = this.#child(id);
    const since = after ?? child.status.seq;
    if (child.status.seq > since || !this.#driving(child)) return Promise.resolve(child.status);
    const { promise, resolve } = Promise.withResolvers<ChildStatus>();
    child.waiters.push({ after: since, resolve });
    return promise;
  }

  /** Resolve every waiting `next` call with its child's current status. */
  cancelWaits(): void {
    for (const child of this.#children.values()) this.#release(child);
  }

  /**
   * Have child `id`'s driver move it again: at once when it rests, else once its current pass
   * ends. Each delivery here nudges; call it after a delivery made elsewhere.
   *
   * @throws {RangeError} When no child has the id.
   */
  nudge(id: string): void {
    const child = this.#child(id);
    child.dirty = true;
    child.rest?.abort();
  }

  /**
   * Answer child `id`'s open question `call` with `value` (any JSON but `null`). Returns its
   * current status, with a `note` when the run refused the answer.
   *
   * @throws {RangeError} When no child has the id.
   * @throws {TypeError} When `value` is `undefined`.
   */
  answer(id: string, call: string, value: unknown): Promise<ChildStatus> {
    return this.#deliver(this.#child(id), (log, options) => complete(log, call, { output: value }, options));
  }

  /**
   * Send `text` to child `id`'s inbox `inbox`: as is to an `Inbox(str)`, as a user message holding
   * it to an `Inbox(Message)`. Returns its current status, with a `note` when the run refused it.
   *
   * @throws {RangeError} When no child has the id.
   */
  message(id: string, inbox: string, text: string): Promise<ChildStatus> {
    const child = this.#child(id);
    const param = child.flow.input.find((input) => input.name === inbox);
    const type = param?.contract?.type ?? null;
    const message = isInboxType(type) && typeEquals(type!.element!, builtinType("Message")!);
    const event = message ? { role: "user", parts: [{ type: "text", text }] } : text;
    return this.#deliver(child, (log, options) => send(log, inbox, event, options));
  }

  /**
   * Close child `id`'s inbox `inbox`, so its turns end. Returns its current status, with a `note`
   * when the run refused the close.
   *
   * @throws {RangeError} When no child has the id.
   */
  closeInbox(id: string, inbox: string): Promise<ChildStatus> {
    return this.#deliver(this.#child(id), (log, options) => close(log, inbox, options));
  }

  /**
   * Cancel child `id`. Returns its current status, with a `note` when it had already ended.
   *
   * @throws {RangeError} When no child has the id.
   */
  stop(id: string): Promise<ChildStatus> {
    return this.#deliver(this.#child(id), (log, options) => cancel(log, options));
  }

  /**
   * End the chat's hold on its children: cancel each one a pass is moving, stop the driver of each
   * waiting one, leaving its log to resume, and resolve every waiting `next`. Returns the waiting
   * children, by id. Later starts throw.
   */
  async dispose(): Promise<WaitingChild[]> {
    this.#disposed = true;
    const left: WaitingChild[] = [];
    const stopping = [...this.#children.values()].map(async (child) => {
      if (!this.#driving(child)) return;
      if (child.passing) {
        try {
          await cancel(child.log, this.#deliveryOptions(child));
          this.nudge(child.id);
        } catch (error) {
          // One that ended meanwhile needs nothing; any other failure leaves the stop to the halt.
          if (!(error instanceof DeliveryRefused)) child.halt.abort();
        }
      } else {
        left.push({ run: child.id, kind: this.#waitKind(child), log: child.log });
        child.halt.abort();
      }
      await child.done;
    });
    await Promise.all(stopping);
    this.cancelWaits();
    // Listed before any await, so in id order.
    return left;
  }

  /** @throws {RangeError} When no child has the id. */
  #child(id: string): Child {
    const child = this.#children.get(id);
    if (child === undefined) throw new RangeError(`no run '${id}' in this chat`);
    return child;
  }

  /** Whether `child`'s driver may still publish a status. */
  #driving(child: Child): boolean {
    return !ENDED.has(child.status.status) && !child.halt.signal.aborted;
  }

  /** What waiting `child` waits on, by its last status. */
  #waitKind(child: Child): WaitingChild["kind"] {
    if ((child.status.asks ?? []).length > 0) return "question";
    const waits = (child.status.waits ?? []) as { tool: string; settled: boolean }[];
    return waits.some((wait) => wait.tool === TAKE && !wait.settled) ? "inbox" : "timer";
  }

  /** The options a delivery to `child`'s log passes. */
  #deliveryOptions(child: Child): DeliveryOptions {
    return {
      env: this.#options.env,
      codeRoot: this.#options.root,
      // A flow built in code is the run's; one loaded from files is compiled again from the log.
      flow: child.flow.source === null ? child.flow : null,
    };
  }

  /** Make `deliver`'s delivery to `child`'s log and nudge its driver; a refusal is a note. */
  async #deliver(
    child: Child,
    deliver: (log: string, options: DeliveryOptions) => Promise<unknown>,
  ): Promise<ChildStatus> {
    try {
      await deliver(child.log, this.#deliveryOptions(child));
    } catch (error) {
      if (!(error instanceof DeliveryRefused)) throw error;
      return { ...child.status, note: error.message };
    }
    this.nudge(child.id);
    return child.status;
  }

  /**
   * Drive `child` from its start to an end: a pass, then, while it waits, one more pass after
   * each nudge and at each `wakeAt`. `started` settles once the log holds the run's start, or
   * rejects with what stopped it before.
   */
  async #drive(
    child: Child,
    runId: string,
    inputs: Readonly<Record<string, unknown>>,
    started: PromiseWithResolvers<void>,
  ): Promise<void> {
    const { env, llmConfig, models = null, scheduler, chatRunId, advance: move = advance } = this.#options;
    const onEvent = (event: RunEvent) => {
      if (event instanceof RunStarted) started.resolve();
      // An interrupted reply succeeds with null when its output is `Optional(...)`.
      if (isReply(event) && !event.interrupted && event.output !== null) {
        child.replies.push(writeAsText(event.output as Value));
      }
    };
    const common = { ...this.#shared(child), onEvent, signal: child.halt.signal, env, scheduler, models };
    try {
      let result = await this.#pass(child, () =>
        runFlow(child.flow, inputs, {
          log: child.log,
          runId,
          parent: { runId: chatRunId },
          llmConfig: Object.keys(llmConfig).length > 0 ? llmConfig : null,
          ...common,
        }),
      );
      // The deadline the run was last moved at. A node is due once any of its calls passes its
      // deadline, so that pass times it out and `wakeAt` comes later; were it to stay, resting
      // until it would only spin.
      let spent: string | null = null;
      for (;;) {
        this.#publish(child, this.#settledStatus(child, result));
        if (result.status !== "waiting") return;
        const next = wakeAt(result.waits);
        const wake: string | null = next === spent ? null : next;
        if (await this.#rest(child, wake)) spent = wake;
        if (child.halt.signal.aborted) return;
        result = await this.#pass(child, () =>
          move({ log: child.log, ...this.#deliveryOptions(child), ...common }),
        );
      }
    } catch (error) {
      // Before the start, `start` throws it; after, it is the child's failure.
      started.reject(error);
      // A halted driver leaves the run as it is.
      if (child.halt.signal.aborted) return;
      this.#publish(child, {
        run: child.id,
        status: "failed",
        note: `the chat couldn't move the run: ${(error as Error | null)?.message ?? String(error)}`,
        log: child.log,
      });
    } finally {
      // A run that never started leaves no folder behind.
      removeEmptyDir(dirname(child.log));
      this.#release(child);
    }
  }

  /** Run `move` as a pass of `child`. */
  async #pass(child: Child, move: () => Promise<RunResult>): Promise<RunResult> {
    child.passing = true;
    try {
      return await move();
    } finally {
      child.passing = false;
    }
  }

  /**
   * Rest until a nudge, the child's halt or `wake` (a record time; `null` waits for a nudge only),
   * at once when a nudge came during the pass. Returns whether `wake` ended it.
   */
  async #rest(child: Child, wake: string | null): Promise<boolean> {
    if (child.dirty) {
      child.dirty = false;
      return false;
    }
    const rest = new AbortController();
    child.rest = rest;
    const halt = () => rest.abort();
    child.halt.signal.addEventListener("abort", halt, { once: true });
    if (child.halt.signal.aborted) rest.abort();
    try {
      if (wake === null) {
        if (!rest.signal.aborted) {
          await new Promise<void>((resolve) => rest.signal.addEventListener("abort", () => resolve(), { once: true }));
        }
        return false;
      }
      return await sleepUntil(this.#options.scheduler, recordTimeMs(wake, "wakeAt"), rest.signal).then(
        () => true,
        // A nudge or the halt.
        () => false,
      );
    } finally {
      child.halt.signal.removeEventListener("abort", halt);
      child.rest = null;
      child.dirty = false;
    }
  }

  /** The options every pass of `child` passes. */
  #shared(child: Child): { tools: Record<string, ToolEntry>; skills: ReadonlyMap<string, Skill> | null } {
    // One handler serves `ask` and `ask_user` (passing both is refused).
    return { tools: { ...registeredTools(), ask: () => pending() }, skills: this.#options.skillsFor(child.flow) };
  }

  /** The status `result` leaves `child` in, with the replies of the pass that ended. */
  #settledStatus(child: Child, result: RunResult): Omit<ChildStatus, "seq"> {
    const replies = child.replies.splice(0);
    const base = { run: child.id, log: child.log, ...(replies.length > 0 ? { replies } : {}) };
    if (result.status === "succeeded") return { ...base, status: "succeeded", output: result.output };
    if (result.status === "failed") {
      const failure = result.failure!;
      return { ...base, status: "failed", failure: { code: failure.code, message: failure.message, node: failure.nodeId } };
    }
    if (result.status === "cancelled") return { ...base, status: "cancelled" };
    const asks = pendingCalls(child.log).map((call) => ({ call: call.callId, question: questionText(call.arguments) }));
    return {
      ...base,
      status: "waiting",
      waits: result.waits.map((wait) => ({ ...wait })),
      wake_at: wakeAt(result.waits),
      ...(asks.length > 0 ? { asks } : {}),
    };
  }

  /**
   * Make `status` `child`'s, numbered, unless it says what the last one did and has no replies;
   * wake its readers.
   */
  #publish(child: Child, status: Omit<ChildStatus, "seq">): void {
    const { seq: _, ...last } = child.status;
    if (status.replies === undefined && isDeepStrictEqual(last, status)) return;
    child.status = { ...status, seq: ++this.#seq };
    const waiting = child.waiters.splice(0);
    for (const waiter of waiting) {
      if (child.status.seq > waiter.after) waiter.resolve(child.status);
      else child.waiters.push(waiter);
    }
  }

  /** Resolve every waiting `next` call on `child` with its current status. */
  #release(child: Child): void {
    for (const waiter of child.waiters.splice(0)) waiter.resolve(child.status);
  }
}
