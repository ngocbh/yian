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
 * Every status a child publishes carries the replies no reader has marked read (`markRead`), so a
 * reader that falls behind loses none; asks are always the run's open ones.
 *
 * A child's `ask` returns `pending()`, so the child waits durably for the user's answer, which the
 * chat relays with `answer`. The children's stop is their own, tied to the chat's exit by
 * `dispose`, never to the call that started them.
 *
 * Knows about: the run boundary (`runFlow`, `advance`, the deliveries, `pendingCalls`,
 * `runLogPath`), the waits' `wakeAt` and the record-time reader, compose, the run events, the tool
 * registry and the `pending` marker, skills, the CLI's key preflight (`startup.ts`).
 */

import { appendFileSync } from "node:fs";
import { dirname } from "node:path";
import { format, isDeepStrictEqual } from "node:util";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import type { Env } from "../defaults.ts";
import { NodeSucceeded, type RunEvent, RunStarted } from "../events.ts";
import type { ModelAccess } from "../llm_clients/factory.ts";
import type { Scheduler } from "../platform.ts";
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
import { ensureProviderKeys } from "./startup.ts";
import { CommandEnv, sleepUntil } from "./utils.ts";

/** Where a child run stands, as the chat's assistant is told. */
export interface ChildStatus {
  /** The child's id in the chat (`r1`, `r2`, ...). */
  readonly run: string;
  /**
   * `running` from the start and from each delivery the run took until the pass it causes
   * settles; `waiting`, `succeeded`, `failed` or `cancelled` as a pass leaves it. A pass at a
   * `wakeAt` publishes only where it settles.
   */
  readonly status: "running" | "waiting" | "succeeded" | "failed" | "cancelled";
  /** The flow's output, once it succeeded. */
  readonly output?: unknown;
  /**
   * Why it failed, once the run failed. A child whose driver a host fault stopped (the log
   * unreadable, another writer holding it) is `failed` with only a `note`: such a fault has no
   * failure code, the codes being the run's closed list.
   */
  readonly failure?: { readonly code: string; readonly message: string; readonly node: string | null };
  /** What a waiting child waits on (`WaitInfo`s). */
  readonly waits?: readonly unknown[];
  /** When a waiting child next wakes by itself, as a record time, or `null` when only a delivery wakes it. */
  readonly wake_at?: string | null;
  /** The questions a waiting child asks, each answered by its call id. */
  readonly asks?: readonly { readonly call: string; readonly question: string }[];
  /**
   * What the child's nodes named `reply` said and no reader has marked read (`markRead`), oldest
   * first.
   */
  readonly replies?: readonly string[];
  /**
   * Something the chat couldn't do with the run, in words: a delivery the run refused or one made
   * after the chat ended (only on the status that call returns), or the error that stopped the
   * driver of a failed child.
   */
  readonly note?: string;
  /** The child's run log. */
  readonly log: string;
  /**
   * The status's number: every status any child publishes gets the next one, so a reader marks
   * the statuses it has seen by the last number it read.
   */
  readonly seq: number;
}

/** A status before it is numbered. */
type Unnumbered = Omit<ChildStatus, "seq">;

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
  /**
   * The file the console output of the children's CODE nodes is appended to while any pass runs,
   * so it doesn't land on the chat's terminal; `null` (the default) leaves it on the console.
   */
  consoleFile?: string | null;
  /**
   * Sees each status any child publishes, once it is the child's (not a child's first `running`,
   * which `start` returns). What it throws is dropped.
   */
  onStatus?: ((status: ChildStatus) => void) | null;
  /** @internal For tests: the `advance` the drivers call. */
  advance?: typeof advance;
  /** @internal For tests: the `cancel` `stop` and `dispose` call. */
  cancel?: typeof cancel;
}

/** A child left waiting when the chat ends, and what it waits on. */
export interface WaitingChild {
  readonly run: string;
  /**
   * `question` when it asks something, else `inbox` when it waits for a message, else `timer`;
   * `halted` for a running one `haltAll` stopped before its cancel was done.
   */
  readonly kind: "question" | "inbox" | "timer" | "halted";
  readonly log: string;
}

/** A child still running or waiting, as `live` lists it. */
export interface LiveChild {
  readonly id: string;
  /** What it runs, as its start named it: the flow's file, or else the flow's name. */
  readonly name: string;
  /** When it started, on the children's clock. */
  readonly since: number;
  readonly status: ChildStatus;
}

/** A `next` call waiting for a status numbered after `after`. */
interface Waiter {
  readonly after: number;
  readonly resolve: (status: ChildStatus) => void;
}

/** One child and its driver's state. */
interface Child {
  readonly id: string;
  readonly flow: ComposedFlow;
  /** What its start named it (`LiveChild.name`). */
  readonly name: string;
  /** When it started, on the children's clock. */
  readonly since: number;
  readonly log: string;
  /** The tools and skills every pass of the child is given. */
  readonly tools: Record<string, ToolEntry>;
  readonly skills: ReadonlyMap<string, Skill> | null;
  /** The last published status. */
  status: ChildStatus;
  readonly waiters: Set<Waiter>;
  /** Set by a nudge: the driver runs one more pass before it rests. */
  dirty: boolean;
  /** The driver's rest between passes, which a nudge ends; `null` while it doesn't rest. */
  rest: AbortController | null;
  /** What the child's `reply` nodes said during the current pass. */
  readonly fresh: string[];
  /** The replies published and not marked read, each with the seq of the status that first held it. */
  unread: { readonly seq: number; readonly text: string }[];
  /** Whether the run's log holds its start. */
  started: boolean;
  /** The child's own stop: it stops the driver and leaves the log as it is. */
  readonly halt: AbortController;
  /** Settles once the driver has stopped. */
  done: Promise<void>;
}

/** The console methods a CODE worker's output is replayed through. */
const CONSOLE_METHODS = ["log", "info", "warn", "error", "debug"] as const;

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
 * Append the console methods' output to `file` instead, and return what puts them back as they
 * were at this call (undoing any change made to them meanwhile).
 *
 * A CODE worker's console lines are replayed through this process's `console` with no run named,
 * so routing the whole console while a child runs is the only way to keep them off the chat's
 * terminal; a sink per run would replace this. A CODE error reported after its body returned can
 * still arrive once the last pass has ended, and then reaches the terminal.
 */
function routeConsole(file: string): () => void {
  const saved = new Map(CONSOLE_METHODS.map((method) => [method, console[method]] as const));
  for (const method of CONSOLE_METHODS) {
    console[method] = (...args: unknown[]) => {
      try {
        appendFileSync(file, `${format(...args)}\n`);
      } catch {
        // A line that can't be kept mustn't fail the node that printed it.
      }
    };
  }
  return () => {
    for (const [method, original] of saved) console[method] = original;
  };
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
  /** The start in progress: starts run one at a time, in call order. */
  #starting: Promise<unknown> = Promise.resolve();
  #disposed = false;
  /** Aborted by `haltAll`: a dispose waits no longer. */
  readonly #halting = new AbortController();
  /** The passes running, and what puts the console back once none does. */
  #passes = 0;
  #restoreConsole: (() => void) | null = null;

  constructor(options: ChildRunsOptions) {
    this.#options = options;
  }

  /**
   * Start `flow` with `inputs` as the next child (`r1`, `r2`, ...): return its id, its log and the
   * `seq` of its first status (`running`) once the log holds the run's start, while the run goes
   * on in the background; `next(id, seq)` then sees every later status.
   *
   * Starts run one at a time in call order, so ids follow it, and an id is taken only by a start
   * that succeeds. Each first checks the provider keys the flow's agents need, without a prompt.
   * `options.signal` is the caller's: aborting it before the start begins rejects, and after it
   * does nothing to the child. `options.name` is what `live` lists it by (the flow's file, say);
   * the flow's name when absent. `options.tools` are host tools this child alone gets, beside the
   * registered ones.
   *
   * @throws {MissingKeyError} When an agent's provider needs a key that isn't set, before anything
   *   is written; `ensureProviderKeys`' other errors likewise.
   * @throws {Error} When the chat was disposed; the signal's reason once it aborted; what `runFlow`
   *   throws before the run starts (a tool the flow names that the run lacks, say), with no log
   *   left behind.
   */
  start(
    flow: ComposedFlow,
    inputs: Readonly<Record<string, unknown>>,
    options: {
      signal?: AbortSignal | null;
      name?: string;
      tools?: Readonly<Record<string, ToolEntry>>;
    } = {},
  ): Promise<{ id: string; log: string; seq: number }> {
    const name = options.name ?? flow.name ?? flow.compiled.flowId ?? "flow";
    const tools = options.tools ?? {};
    const signal = options.signal ?? null;
    const turn = this.#starting.then(() => this.#begin(flow, inputs, name, tools, signal));
    this.#starting = turn.catch(() => {});
    return turn;
  }

  /** Start `flow` as the next child, once the starts called before it are done. */
  async #begin(
    flow: ComposedFlow,
    inputs: Readonly<Record<string, unknown>>,
    name: string,
    tools: Readonly<Record<string, ToolEntry>>,
    signal: AbortSignal | null,
  ): Promise<{ id: string; log: string; seq: number }> {
    const check = () => {
      if (this.#disposed) throw new Error("the chat's runs were disposed");
      signal?.throwIfAborted();
    };
    check();
    const { env, llmConfig, logDir, chatRunId, skillsFor } = this.#options;
    // No terminal: a missing key is reported to the assistant rather than prompted for mid-chat.
    await ensureProviderKeys(flow, llmConfig, inputs, { env: new CommandEnv(env), terminal: null });
    check();
    const id = `r${this.#started + 1}`;
    const runId = `${chatRunId}-${id}`;
    const log = runLogPath(logDir, runId);
    const seq = ++this.#seq;
    const child: Child = {
      id,
      flow,
      name,
      since: this.#options.scheduler.now(),
      log,
      // One handler serves `ask` and `ask_user` (passing both is refused).
      tools: { ...registeredTools(), ...tools, ask: () => pending() },
      skills: skillsFor(flow),
      status: { run: id, status: "running", log, seq },
      waiters: new Set(),
      dirty: false,
      rest: null,
      fresh: [],
      unread: [],
      started: false,
      halt: new AbortController(),
      done: Promise.resolve(),
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
    this.#started += 1;
    return { id, log, seq };
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
   * Resolve with child `id`'s latest status once its seq is past `after` (without it, past the
   * current status's, so a status published before the call isn't seen; pass the `seq` read
   * before). Resolve with its current status instead once its driver has stopped and none will
   * follow, `cancelWaits` is called, or `options.signal` aborts (at once when it already has).
   *
   * @throws {RangeError} When no child has the id.
   */
  next(id: string, after?: number, options: { signal?: AbortSignal | null } = {}): Promise<ChildStatus> {
    const child = this.#child(id);
    const since = after ?? child.status.seq;
    const signal = options.signal ?? null;
    if (child.status.seq > since || !this.#driving(child) || signal?.aborted) return Promise.resolve(child.status);
    const { promise, resolve } = Promise.withResolvers<ChildStatus>();
    const onAbort = () => {
      child.waiters.delete(waiter);
      resolve(child.status);
    };
    const waiter: Waiter = {
      after: since,
      resolve: (status) => {
        signal?.removeEventListener("abort", onAbort);
        resolve(status);
      },
    };
    child.waiters.add(waiter);
    signal?.addEventListener("abort", onAbort, { once: true });
    return promise;
  }

  /** The children still running or waiting (and driven), by id. */
  live(): LiveChild[] {
    return [...this.#children.values()]
      .filter((child) => this.#driving(child))
      .map(({ id, name, since, status }) => ({ id, name, since, status }));
  }

  /** @internal For tests: how many `next` calls wait, over all children. */
  waitingNexts(): number {
    let count = 0;
    for (const child of this.#children.values()) count += child.waiters.size;
    return count;
  }

  /**
   * Stop every child's driver at once, leaving each log as it is, and end a `dispose` that waits
   * on a cancel: it returns with each running child it hadn't cancelled yet as `halted`. For a
   * second Ctrl-C while the chat ends.
   */
  haltAll(): void {
    this.#halting.abort();
    for (const child of this.#children.values()) child.halt.abort();
  }

  /** Resolve every waiting `next` call with its child's current status. */
  cancelWaits(): void {
    for (const child of this.#children.values()) this.#release(child);
  }

  /**
   * Mark child `id`'s replies published at or before status `seq` read: later statuses leave them
   * out. Call it when a status is handed to the model.
   *
   * @throws {RangeError} When no child has the id.
   */
  markRead(id: string, seq: number): void {
    const child = this.#child(id);
    child.unread = child.unread.filter((reply) => reply.seq > seq);
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
   * status: `running` once the run took the answer, else the current one with a `note`.
   *
   * @throws {RangeError} When no child has the id.
   * @throws {TypeError} When `value` is `undefined`.
   */
  answer(id: string, call: string, value: unknown): Promise<ChildStatus> {
    return this.#deliver(this.#child(id), (log, options) => complete(log, call, { output: value }, options));
  }

  /**
   * Send `text` to child `id`'s inbox `inbox`: as is to an `Inbox(str)`, as a user message holding
   * it to an `Inbox(Message)`. Returns its status as `answer` does.
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
   * Close child `id`'s inbox `inbox`, so its turns end. Returns its status as `answer` does.
   *
   * @throws {RangeError} When no child has the id.
   */
  closeInbox(id: string, inbox: string): Promise<ChildStatus> {
    return this.#deliver(this.#child(id), (log, options) => close(log, inbox, options));
  }

  /**
   * Cancel child `id`. Returns its status as `answer` does; the note says when it had ended.
   *
   * @throws {RangeError} When no child has the id.
   */
  stop(id: string): Promise<ChildStatus> {
    const move = this.#options.cancel ?? cancel;
    return this.#deliver(this.#child(id), (log, options) => move(log, options));
  }

  /**
   * End the chat's hold on its children, by each one's published status: cancel each `running`
   * one; stop the driver of each `waiting` one, a pass at its `wakeAt` included, leaving its log to
   * resume; and resolve every waiting `next`. Returns the waiting children, by id. Later starts
   * throw, and later deliveries change nothing.
   */
  async dispose(): Promise<WaitingChild[]> {
    this.#disposed = true;
    const left: WaitingChild[] = [];
    const halted = new Promise<"halted">((resolve) => {
      if (this.#halting.signal.aborted) resolve("halted");
      this.#halting.signal.addEventListener("abort", () => resolve("halted"), { once: true });
    });
    const move = this.#options.cancel ?? cancel;
    const stopping = [...this.#children.values()].map(async (child) => {
      if (!this.#driving(child)) return;
      if (child.status.status === "running") {
        try {
          const cancelled = await Promise.race([move(child.log, this.#deliveryOptions(child)), halted]);
          if (cancelled === "halted") left.push({ run: child.id, kind: "halted", log: child.log });
          else this.nudge(child.id);
        } catch (error) {
          // One that ended meanwhile needs nothing; any other failure leaves the stop to the halt.
          if (!(error instanceof DeliveryRefused)) child.halt.abort();
        }
      } else {
        left.push({ run: child.id, kind: this.#waitKind(child), log: child.log });
        child.halt.abort();
      }
      // A halt ends the driver's pass; a pass slow to stop isn't waited for once halted.
      await Promise.race([child.done, halted]);
    });
    await Promise.all(stopping);
    this.cancelWaits();
    return left.sort((a, b) => Number(a.run.slice(1)) - Number(b.run.slice(1)));
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
    const waits = (child.status.waits ?? []) as readonly { tool: string; settled: boolean }[];
    return waits.some((wait) => wait.tool === TAKE && !wait.settled) ? "inbox" : "timer";
  }

  /** The options a delivery to `child`'s log passes. */
  #deliveryOptions(child: Child): DeliveryOptions {
    return {
      env: this.#options.env,
      // A flow confined to the chat's folder is held to it; a bundled one was loaded with no root.
      codeRoot: child.flow.source?.codeRoot == null ? null : this.#options.root,
      // A flow built in code is the run's; one loaded from files is compiled again from the log.
      flow: child.flow.source === null ? child.flow : null,
    };
  }

  /**
   * Make `deliver`'s delivery to `child`'s log, publish the child `running` and nudge its driver;
   * return its status. A refusal, or the chat having ended, is a note on the status returned, and
   * changes nothing.
   */
  async #deliver(
    child: Child,
    deliver: (log: string, options: DeliveryOptions) => Promise<unknown>,
  ): Promise<ChildStatus> {
    // Its runs are left as they were, to resume later.
    if (this.#disposed) return { ...child.status, note: "the chat has ended" };
    try {
      await deliver(child.log, this.#deliveryOptions(child));
    } catch (error) {
      if (!(error instanceof DeliveryRefused)) throw error;
      return { ...child.status, note: error.message };
    }
    // Until its pass settles, the child moves on what it was given.
    if (this.#driving(child)) this.#publish(child, { run: child.id, status: "running", log: child.log });
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
      if (event instanceof RunStarted) {
        child.started = true;
        started.resolve();
      }
      // An interrupted reply succeeds with null when its output is `Optional(...)`.
      if (isReply(event) && !event.interrupted && event.output !== null) {
        child.fresh.push(writeAsText(event.output as Value));
      }
    };
    const common = {
      tools: child.tools,
      skills: child.skills,
      onEvent,
      signal: child.halt.signal,
      env,
      scheduler,
      models,
    };
    try {
      let result = await this.#pass(() =>
        runFlow(child.flow, inputs, {
          log: child.log,
          runId,
          parent: { runId: chatRunId },
          llmConfig: Object.keys(llmConfig).length > 0 ? llmConfig : null,
          ...common,
        }),
      );
      // Already settled by `RunStarted`; kept so that `start` can never hang.
      started.resolve();
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
        result = await this.#pass(() => move({ log: child.log, ...this.#deliveryOptions(child), ...common }));
      }
    } catch (error) {
      // Before the start, `start` throws it; after, it is the child's failure.
      started.reject(error);
      // A halted driver leaves the run as it is, and a child that never started has no status.
      if (child.halt.signal.aborted || !child.started) return;
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

  /** Run `move` as a pass, the console routed to the console file meanwhile. */
  async #pass(move: () => Promise<RunResult>): Promise<RunResult> {
    const file = this.#options.consoleFile ?? null;
    if (this.#passes++ === 0 && file !== null) this.#restoreConsole = routeConsole(file);
    try {
      return await move();
    } finally {
      if (--this.#passes === 0) {
        this.#restoreConsole?.();
        this.#restoreConsole = null;
      }
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

  /** The status `result` leaves `child` in. */
  #settledStatus(child: Child, result: RunResult): Unnumbered {
    const base = { run: child.id, log: child.log };
    if (result.status === "succeeded") return { ...base, status: "succeeded", output: result.output };
    if (result.status === "failed") {
      const failure = result.failure!;
      return { ...base, status: "failed", failure: { code: failure.code, message: failure.message, node: failure.nodeId } };
    }
    if (result.status === "cancelled") return { ...base, status: "cancelled" };
    // A pass over the whole log, once per settled pass: the asks are what the log says is open.
    const asks = pendingCalls(child.log).map((call) => ({
      call: call.callId,
      question: questionText(call.arguments),
    }));
    return {
      ...base,
      status: "waiting",
      waits: result.waits.map((wait) => ({ ...wait })),
      wake_at: wakeAt(result.waits),
      ...(asks.length > 0 ? { asks } : {}),
    };
  }

  /**
   * Make `status` `child`'s, numbered and with its unread replies, unless it says what the last
   * one did and the pass said nothing new; tell `onStatus`, and wake its readers.
   */
  #publish(child: Child, status: Unnumbered): void {
    const fresh = child.fresh.splice(0);
    const seq = this.#seq + 1;
    const unread = [...child.unread, ...fresh.map((text) => ({ seq, text }))];
    const replies = unread.map((reply) => reply.text);
    const next: Unnumbered = { ...status, ...(replies.length > 0 ? { replies } : {}) };
    const { seq: _, ...last } = child.status;
    if (fresh.length === 0 && isDeepStrictEqual(last, next)) return;
    this.#seq = seq;
    child.unread = unread;
    child.status = { ...next, seq };
    try {
      this.#options.onStatus?.(child.status);
    } catch {
      // A reader's fault mustn't stop the child's driver.
    }
    for (const waiter of [...child.waiters]) {
      if (seq > waiter.after) {
        child.waiters.delete(waiter);
        waiter.resolve(child.status);
      }
    }
  }

  /** Resolve every waiting `next` call on `child` with its current status. */
  #release(child: Child): void {
    const waiting = [...child.waiters];
    child.waiters.clear();
    for (const waiter of waiting) waiter.resolve(child.status);
  }
}
