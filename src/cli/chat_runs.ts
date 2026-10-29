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
 * Knows about: the run boundary (`runFlow`, `advance`, `runLogPath`), the waits' `wakeAt` and the
 * record-time reader, compose, the tool registry and the
 * `pending` marker, skills, the CLI's key preflight (`startup.ts`).
 */

import { isDeepStrictEqual } from "node:util";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import type { Env } from "../defaults.ts";
import type { ModelAccess } from "../llm_clients/factory.ts";
import type { Scheduler } from "../platform.ts";
import { advance, type RunResult, runFlow, runLogPath } from "../run.ts";
import { recordTimeMs } from "../runtime/log.ts";
import { wakeAt } from "../runtime/waits.ts";
import type { Skill } from "../skills/catalog.ts";
import { pending } from "../tools/context.ts";
import { registeredTools, type ToolEntry } from "../tools/registry.ts";
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
  /** Something the chat couldn't do with the run, in words. */
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
  /** The driver's rest between passes, which a nudge ends; `null` while a pass runs. */
  rest: AbortController | null;
}

/** The terminal statuses: no pass follows them. */
const ENDED: ReadonlySet<ChildStatus["status"]> = new Set(["succeeded", "failed", "cancelled"]);

/**
 * The chat's child runs: start each in the background, and read its status as it changes.
 */
export class ChildRuns {
  readonly #options: ChildRunsOptions;
  readonly #children = new Map<string, Child>();
  #seq = 0;

  constructor(options: ChildRunsOptions) {
    this.#options = options;
  }

  /**
   * Start `flow` with `inputs` as the next child (`r1`, `r2`, ...) and return its id and log at
   * once; the run goes on in the background.
   */
  async start(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>): Promise<{ id: string; log: string }> {
    const id = `r${this.#children.size + 1}`;
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
    };
    this.#children.set(id, child);
    void this.#drive(child, runId, inputs);
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
   * with its current status once it has ended and none will follow.
   *
   * @throws {RangeError} When no child has the id.
   */
  next(id: string, after?: number): Promise<ChildStatus> {
    const child = this.#child(id);
    const since = after ?? child.status.seq;
    if (child.status.seq > since || ENDED.has(child.status.status)) return Promise.resolve(child.status);
    const { promise, resolve } = Promise.withResolvers<ChildStatus>();
    child.waiters.push({ after: since, resolve });
    return promise;
  }

  /**
   * Have child `id`'s driver move it again: at once when it rests, else once its current pass
   * ends. Call it after each delivery to the child's log.
   *
   * @throws {RangeError} When no child has the id.
   */
  nudge(id: string): void {
    const child = this.#child(id);
    child.dirty = true;
    child.rest?.abort();
  }

  /** Wait for every child's driver to stop. */
  async dispose(): Promise<void> {}

  /** @throws {RangeError} When no child has the id. */
  #child(id: string): Child {
    const child = this.#children.get(id);
    if (child === undefined) throw new RangeError(`no run '${id}' in this chat`);
    return child;
  }

  /**
   * Drive `child` from its start to an end: a pass, then, while it waits, one more pass after
   * each nudge and at each `wakeAt`.
   */
  async #drive(child: Child, runId: string, inputs: Readonly<Record<string, unknown>>): Promise<void> {
    const { env, llmConfig, models = null, scheduler, chatRunId, advance: move = advance } = this.#options;
    const common = { ...this.#shared(child), env, scheduler, models };
    let result = await runFlow(child.flow, inputs, {
      log: child.log,
      runId,
      parent: { runId: chatRunId },
      llmConfig: Object.keys(llmConfig).length > 0 ? llmConfig : null,
      ...common,
    });
    // The deadline the run was last moved at. A node is due once any of its calls passes its
    // deadline, so that pass times it out and `wakeAt` comes later; were it to stay, resting until
    // it would only spin.
    let spent: string | null = null;
    for (;;) {
      this.#publish(child, this.#settledStatus(child, result));
      if (result.status !== "waiting") return;
      const next = wakeAt(result.waits);
      const wake: string | null = next === spent ? null : next;
      if (await this.#rest(child, wake)) spent = wake;
      result = await move({
        log: child.log,
        // A flow built in code is the run's; one loaded from files is compiled again from the log.
        flow: child.flow.source === null ? child.flow : null,
        codeRoot: this.#options.root,
        ...common,
      });
    }
  }

  /**
   * Rest until a nudge or `wake` (a record time; `null` waits for a nudge only), at once when a
   * nudge came during the pass. Returns whether `wake` ended it.
   */
  async #rest(child: Child, wake: string | null): Promise<boolean> {
    if (child.dirty) {
      child.dirty = false;
      return false;
    }
    const rest = new AbortController();
    child.rest = rest;
    try {
      if (wake === null) {
        await new Promise<void>((resolve) => rest.signal.addEventListener("abort", () => resolve(), { once: true }));
        return false;
      }
      return await sleepUntil(this.#options.scheduler, recordTimeMs(wake, "wakeAt"), rest.signal).then(
        () => true,
        // A nudge.
        () => false,
      );
    } finally {
      child.rest = null;
      child.dirty = false;
    }
  }

  /** The options every pass of `child` passes. */
  #shared(child: Child): { tools: Record<string, ToolEntry>; skills: ReadonlyMap<string, Skill> | null } {
    // A child's question waits durably for the user's answer, which the chat relays with `complete`.
    return { tools: { ...registeredTools(), ask: () => pending() }, skills: this.#options.skillsFor(child.flow) };
  }

  /** The status `result` leaves `child` in. */
  #settledStatus(child: Child, result: RunResult): Omit<ChildStatus, "seq"> {
    const base = { run: child.id, log: child.log };
    if (result.status === "succeeded") return { ...base, status: "succeeded", output: result.output };
    if (result.status === "failed") {
      const failure = result.failure!;
      return { ...base, status: "failed", failure: { code: failure.code, message: failure.message, node: failure.nodeId } };
    }
    if (result.status === "cancelled") return { ...base, status: "cancelled" };
    return { ...base, status: "waiting", waits: result.waits.map((wait) => ({ ...wait })), wake_at: wakeAt(result.waits) };
  }

  /** Make `status` `child`'s, numbered, unless it says what the last one did; wake its readers. */
  #publish(child: Child, status: Omit<ChildStatus, "seq">): void {
    const { seq: _, ...last } = child.status;
    if (isDeepStrictEqual(last, status)) return;
    child.status = { ...status, seq: ++this.#seq };
    const waiting = child.waiters.splice(0);
    for (const waiter of waiting) {
      if (child.status.seq > waiter.after) waiter.resolve(child.status);
      else child.waiters.push(waiter);
    }
  }
}
