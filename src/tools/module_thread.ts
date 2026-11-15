/**
 * A tool module's thread, from the host's side: one worker thread per module, which imports it
 * once, lists its tools and runs their calls (`tools/module_worker`). It runs the CODE worker
 * entry, so no build lists another entry, and like a CODE thread it shares this process's live
 * environment and isn't a sandbox. Each request carries the caller's `current()`, so a run under
 * `withEnv` keeps its own values. The thread lives until it exits or is terminated, but its handle
 * here is referenced only while a request is open, so an idle thread never keeps this process
 * alive.
 *
 * A tool's checks run there as its stand-in's validators: a check of a call's arguments runs the
 * module's validator and `isConcurrencySafe` and returns a handle to the validator's output, which
 * stays on the thread for every attempt of the call; it is released once the call settles, unless
 * a TOOL node may retry its error, or once the handle is collected. Each check's issues and
 * failures come back as the registry's own, so they act as a host tool's do.
 *
 * A call's error is rebuilt here with the name, message, stack and tool error kinds it had on the
 * thread, so it steers an AGENT and TOOL retries as a host tool's error does, and a pending marker
 * comes back as `pending()`.
 *
 * What the module prints while a request is open, from its promises and timers too, goes to the
 * console that request was given; anything else, the import's lines among them, to this process's
 * `console`.
 *
 * A thread that exits (through `process.exit`, or an uncaught error such as one thrown after a
 * call returned) fails each request in flight with a `ToolModuleError`, `tool.module_exited`.
 * Once the module has listed, the next request starts a fresh thread, which imports the module
 * again and must list it as the first one did, in every field; a call checked on a thread that
 * exited is checked again there. When a fresh thread can't take over, that request and every later
 * one fail with `tool.module_restart`.
 *
 * Knows about: tools/context, tools/errors, tools/function_tools' types, tools/module_worker,
 * tools/registry (a validator's message), events' types, defaults, diagnostics, platform, typesys
 * and `node:*`.
 */

import { basename } from "node:path";
import { SHARE_ENV, Worker } from "node:worker_threads";

import { current } from "../defaults.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import type { ConsoleMethod } from "../events.ts";
import { realScheduler } from "../platform.ts";
import { packFiles, unpackFiles } from "../typesys/crossing.ts";
import { YianRangeError, YianTypeError } from "../typesys/errors.ts";
import { canonicalJson } from "../typesys/json.ts";
import { TypeCheckError, toHost, type Value } from "../typesys/values.ts";
import { pending, type ToolContext } from "./context.ts";
import { isPermanentToolError, isToolCheckError, TOOL_ERROR, ToolCheckError, ToolModuleError } from "./errors.ts";
import type { StandardResult } from "./function_tools.ts";
import type {
  ToolModuleConsole,
  ToolModuleData,
  ToolModuleListing,
  ToolModuleReply,
  ToolModuleRequest,
} from "./module_worker.ts";
import { validatorFailure } from "./registry.ts";

// As `nodes/code/node.ts` finds the entry: `bun build --compile` bundles it at
// `nodes/code/worker.js` beside the bundle's entry, whose URL this module then has.
const WORKER_URL = import.meta.url.endsWith(".ts")
  ? new URL("../nodes/code/worker.ts", import.meta.url)
  : import.meta.url.endsWith("/tools/module_thread.js")
    ? new URL("../nodes/code/worker.js", import.meta.url)
    : new URL("./nodes/code/worker.js", import.meta.url);

/** The request for the module's listing. */
const LIST = (id: number): ToolModuleRequest => ({ kind: "list", id, defaults: current() });

/** Takes one line a module's code printed: the `console` method it called and its text. */
export type ConsoleSink = (method: ConsoleMethod, text: string) => void;

/** An open request: how it settles, and what to undo once it has. */
interface OpenRequest {
  resolve(reply: ToolModuleReply): void;
  reject(error: Error): void;
  release(): void;
  /** Whether it is a call, whose outcome an exit leaves unknown. */
  readonly call: boolean;
  /** Takes what the module prints while it is open; `null` leaves that to this process's `console`. */
  readonly console: ConsoleSink | null;
}

/**
 * A module tool's arguments, checked on its thread: what its stand-in's validator returns, which
 * the registry hands its `isConcurrencySafe` and `execute`. Frozen.
 */
export class CheckedArgs {
  /** The check's id, by which the thread keeps the validator's output; `null` when it kept none. */
  readonly check: number | null;
  /** Whether the module's `isConcurrencySafe` says the call may run beside other calls. */
  readonly safe: boolean;
  /** The arguments as they were checked, packed; each call carries them. */
  readonly args: unknown;

  constructor(check: number | null, safe: boolean, args: unknown) {
    this.check = check;
    this.safe = safe;
    this.args = args;
    Object.freeze(this);
  }
}

/** A tool module's thread, started when this is made, and started again after it exits. */
export class ToolModuleThread {
  // A handle dropped before its call settled (a call that never ran, or a TOOL node out of
  // retries) releases its output once it is collected.
  static readonly #dropped = new FinalizationRegistry<{ thread: ToolModuleThread; check: number }>(
    ({ thread, check }) => thread.#post({ kind: "release", id: check }),
  );

  /** The module file's real path. */
  readonly path: string;
  /** How messages name the file. */
  readonly label: string;
  /** The thread serving requests: the first one, or the fresh one that took over after an exit. */
  #worker: Worker;
  /** Each open request, by id. */
  readonly #open = new Map<number, OpenRequest>();
  // Ids run on across threads, so a fresh thread never takes a check id an earlier one kept.
  #lastId = 0;
  /** Why the thread's requests fail, once it failed or exited. */
  #ended: ToolModuleError | null = null;
  #exitCode: number | null = null;
  /** The module's first listing, which a fresh thread must match; `null` until it listed. */
  #listing: ToolModuleListing | null = null;
  /** A fresh thread's start, until it has listed. */
  #restarting: Promise<void> | null = null;
  /** Why every later request fails, once a fresh thread couldn't take over. */
  #broken: ToolModuleError | null = null;
  #terminated = false;

  /**
   * Start the thread of the module at `path`.
   *
   * @param label How messages name the file; its base name by default.
   */
  constructor(path: string, label: string = basename(path)) {
    this.path = path;
    this.label = label;
    this.#worker = this.#spawn();
  }

  /** The code the thread exited with, or `null` while it, or a fresh one in its place, runs. */
  get exitCode(): number | null {
    return this.#exitCode;
  }

  /**
   * List the module's tools, importing it first if no request has.
   *
   * @throws Error naming the file when the import throws, or the thread fails or exits first.
   */
  async list(): Promise<ToolModuleListing> {
    const listing = listingOf(this.label, await this.#request(LIST));
    this.#listing ??= listing;
    return listing;
  }

  /**
   * Check `args` for the module's tool exported as `tool` on the thread, as the registry checks a
   * host tool's arguments with its validator: run the module's validator, if any, then its
   * `isConcurrencySafe` on what `execute` will get. The thread keeps the validator's output for
   * the call.
   *
   * @param name The tool's name in the registry, which messages name it by.
   * @param sink Takes what the module prints meanwhile.
   * @returns The validator's issues, each path as text and numbers, or the checked arguments.
   * @throws TypeError, as the registry's, when the validator returns no Standard Schema result.
   * @throws What the validator threw, rebuilt as `call` rebuilds a tool's throw.
   * @throws ToolModuleError when the thread fails or exits first, or can't be started again.
   */
  async check(
    tool: string,
    name: string,
    args: unknown,
    sink: ConsoleSink | null = null,
  ): Promise<StandardResult<CheckedArgs>> {
    const packed = packFiles(args);
    const where = `tool ${reprText(name)} arguments`;
    const reply = await this.#request(
      (id) => ({ kind: "check", id, defaults: current(), tool, value: packed, where }),
      { sink },
    );
    if (reply.as === "issues") return { issues: reply.issues };
    if (reply.as === "nonstandard") throw new YianTypeError(reply.message);
    if (reply.as !== "checked") throw replyError(this.label, reply);
    const checked = new CheckedArgs(reply.kept ? reply.id : null, reply.safe, packed);
    if (reply.kept) ToolModuleThread.#dropped.register(checked, { thread: this, check: reply.id }, checked);
    return { value: checked };
  }

  /**
   * Check the module's tool exported as `tool`'s result `value` with its output validator on the
   * thread, as the registry checks a host tool's result.
   *
   * @param name The tool's name in the registry, which messages name it by.
   * @param sink Takes what the module prints meanwhile.
   * @returns The validator's issues, each path as text and numbers, or its output as a host value.
   * @throws ToolCheckError, as the registry's, when the validator returns no Standard Schema
   *   result, or an output that isn't a yian value.
   * @throws What the validator threw, rebuilt as `call` rebuilds a tool's throw.
   * @throws ToolModuleError when the thread fails or exits first, or can't be started again.
   */
  async checkOutput(
    tool: string,
    name: string,
    value: unknown,
    sink: ConsoleSink | null = null,
  ): Promise<StandardResult<unknown>> {
    const packed = packFiles(value);
    const where = `tool ${reprText(name)} result`;
    const reply = await this.#request(
      (id) => ({ kind: "output", id, defaults: current(), tool, value: packed, where }),
      { sink },
    );
    if (reply.as === "issues") return { issues: reply.issues };
    if (reply.as === "nonstandard") throw new ToolCheckError(reply.message);
    if (reply.as === "result") return { value: toHost(unpackFiles(reply.value) as Value) };
    throw replyError(this.label, reply);
  }

  /**
   * Call the module's tool exported as `tool` on the thread with the arguments `check` returned,
   * and return its result as a host value, or `pending()` when it returned a pending marker.
   * `execute` gets the validator's output the thread kept, else a fresh copy of the arguments; on a
   * fresh thread the arguments are checked again first. Its context there holds `ctx`'s call id,
   * attempt, run id and node, the export name as its tool name, and no host object; a stop of
   * `ctx.signal` aborts its signal there, and the call settles as the tool does. Once it settles,
   * the thread releases the kept output, unless a TOOL node may try the call again: its error is
   * neither permanent nor a failed check, and no stop ended it.
   *
   * @param sink Takes what the module prints meanwhile.
   * @throws TypeError when `checked` isn't what `check` returned.
   * @throws What the tool threw, rebuilt as an `Error` with its name, message and stack, its tool
   *   error kinds and, when transient, its `retryAfter`; a thrown non-`Error` is an `Error` with
   *   the value's text, and the value as its `cause` when it is a yian value.
   * @throws ToolCheckError when the result isn't a yian value; or, reported too, when a fresh
   *   thread's check of the arguments finds issues.
   * @throws ToolModuleError when the thread fails or exits first, or can't be started again.
   */
  async call(tool: string, checked: unknown, ctx: ToolContext, sink: ConsoleSink | null = null): Promise<unknown> {
    if (!(checked instanceof CheckedArgs)) {
      throw new YianTypeError(`${this.label} export ${reprText(tool)} takes the arguments its check returned`);
    }
    const { callId, attempt, runId, node, signal } = ctx;
    const { check, args } = checked;
    const where = `tool ${reprText(ctx.toolName)} arguments`;
    let settled = true;
    try {
      const reply = await this.#request(
        (id) => ({ kind: "call", id, defaults: current(), tool, args, check, where, callId, attempt, runId, node }),
        { signal, sink },
      );
      if (reply.as === "result") return toHost(unpackFiles(reply.value) as Value);
      if (reply.as === "pending") return pending();
      // A fresh thread's check failed: as at the call's own check, the model reads its issues.
      if (reply.as === "issues") throw reportedCheck(validatorFailure(`${where} failed its validator`, reply.issues));
      if (reply.as === "nonstandard") throw new ToolCheckError(reply.message);
      throw replyError(this.label, reply);
    } catch (error) {
      settled = signal.aborted || isPermanentToolError(error) || isToolCheckError(error);
      throw error;
    } finally {
      if (settled) this.#release(checked);
    }
  }

  /** Stop the thread for good; open requests reject as for an exit, and no fresh thread starts. */
  async terminate(): Promise<void> {
    this.#terminated = true;
    await this.#worker.terminate();
  }

  /** Start a thread on the module, whose ends end this one's requests while it serves them. */
  #spawn(): Worker {
    const data: ToolModuleData = { path: this.path, label: this.label };
    // `SHARE_ENV`: the module reads and writes the host's live environment, as a CODE body does.
    const worker = new Worker(WORKER_URL, { env: SHARE_ENV, workerData: { yianToolModule: data } });
    worker.unref();
    // Ids are never reused, so a reply or a line from a thread that ended finds no request.
    worker.on("message", (reply: ToolModuleReply | ToolModuleConsole) => {
      if ("console" in reply) {
        const { console: method, text, id } = reply;
        const sink = id === null ? null : (this.#open.get(id)?.console ?? null);
        if (sink === null) console[method](text);
        else sink(method, text);
        return;
      }
      const request = this.#open.get(reply.id);
      if (request === undefined) return;
      this.#settled(reply.id);
      request.resolve(reply);
    });
    const failed = (error: unknown): void => {
      if (worker !== this.#worker) return;
      const text = error instanceof Error ? error.message : String(error);
      this.#end(new ToolModuleError("tool.module_exited", `${this.label}'s thread failed: ${text}`, { cause: error }));
      // A thread that couldn't read a message runs on; it must not run beside a fresh one.
      void worker.terminate();
    };
    worker.on("error", failed);
    worker.on("messageerror", failed);
    worker.on("exit", (code: number) => {
      if (worker !== this.#worker) return;
      this.#exitCode = code;
      this.#end(new ToolModuleError("tool.module_exited", `${this.label}'s thread exited (code ${code})`));
    });
    return worker;
  }

  /**
   * Return the reply to the request `make` builds with a fresh id, what the module prints while it is
   * open going to `sink`. After the thread exited, a fresh thread takes over first, unless the module
   * never listed or the thread was terminated; a stop of `signal` while it starts rejects with the
   * signal's reason.
   *
   * @throws ToolModuleError when the thread fails or exits first, or a fresh one can't take over.
   * @throws Error when the request can't be posted.
   */
  async #request(make: (id: number) => ToolModuleRequest, options: RequestOptions = {}): Promise<ToolModuleReply> {
    if (this.#broken !== null) throw this.#broken;
    if (this.#ended !== null && this.#listing !== null && !this.#terminated) this.#restarting ??= this.#restart();
    if (this.#restarting !== null) await untilStopped(this.#restarting, options.signal);
    return await this.#send(make, options);
  }

  /**
   * Post the request `make` builds with a fresh id to the thread and return its reply. While it is
   * open, a stop of `signal` is posted as its abort, and what the module prints goes to `sink`.
   *
   * @throws ToolModuleError when the thread failed or exits first.
   * @throws Error when the request can't be posted.
   */
  #send(make: (id: number) => ToolModuleRequest, options: RequestOptions = {}): Promise<ToolModuleReply> {
    const { signal, sink = null } = options;
    if (this.#ended !== null) return Promise.reject(this.#ended);
    const id = ++this.#lastId;
    const worker = this.#worker;
    const abort = (): void => worker.postMessage({ kind: "abort", id } satisfies ToolModuleRequest);
    return new Promise((resolve, reject) => {
      const request = make(id);
      if (this.#open.size === 0) worker.ref();
      const release = () => signal?.removeEventListener("abort", abort);
      this.#open.set(id, { resolve, reject, release, call: request.kind === "call", console: sink });
      try {
        worker.postMessage(request);
      } catch (error) {
        this.#settled(id);
        reject(error);
        return;
      }
      if (signal?.aborted) abort();
      else signal?.addEventListener("abort", abort, { once: true });
    });
  }

  /**
   * Start a fresh thread in place of the one that ended, and have it list the module within
   * `YIAN_TOOL_MODULE_START_TIMEOUT_SECONDS`, as the first one listed it.
   *
   * @throws ToolModuleError, `tool.module_restart`, which every later request rejects with too,
   *   when its import throws or outlasts the timeout, it ends first, or its listing differs.
   */
  async #restart(): Promise<void> {
    const ended = this.#ended!;
    this.#worker = this.#spawn();
    this.#ended = null;
    this.#exitCode = null;
    try {
      const listed = this.#send(LIST).then((reply) => listingOf(this.label, reply));
      const change = listingChange(this.#listing!, await listedInTime(this.label, listed));
      if (change !== null) throw new Error(`${change} since the module was first imported`);
    } catch (error) {
      const why = `a fresh thread can't take over: ${error instanceof Error ? error.message : String(error)}`;
      this.#broken = new ToolModuleError("tool.module_restart", `${ended.message}; ${why}`, { cause: error });
      void this.#worker.terminate();
      throw this.#broken;
    } finally {
      this.#restarting = null;
    }
  }

  /** Have the thread drop the output `checked` kept, unless it kept none or was released. */
  #release(checked: CheckedArgs): void {
    if (checked.check === null || !ToolModuleThread.#dropped.unregister(checked)) return;
    this.#post({ kind: "release", id: checked.check });
  }

  /** Post `message`, which has no reply, unless the thread ended: an ended thread holds nothing it names. */
  #post(message: ToolModuleRequest): void {
    if (this.#ended !== null) return;
    try {
      this.#worker.postMessage(message);
    } catch {}
  }

  /** Forget the request `id`; with none left open, the handle no longer keeps this process alive. */
  #settled(id: number): void {
    this.#open.get(id)?.release();
    this.#open.delete(id);
    if (this.#open.size === 0) this.#worker.unref();
  }

  /**
   * Fail every open request with `ended`, unless an earlier end did, and every later one until a
   * fresh thread takes over; a call's error says its outcome is unknown.
   */
  #end(ended: ToolModuleError): void {
    this.#ended ??= ended;
    const { message, cause } = this.#ended;
    for (const [id, request] of this.#open) {
      this.#settled(id);
      const unknown = `${message}; the call's outcome is unknown`;
      request.reject(request.call ? new ToolModuleError("tool.module_exited", unknown, { cause }) : this.#ended);
    }
  }
}

/** How a request is sent: the stop that aborts it, and what takes the lines the module prints while it is open. */
interface RequestOptions {
  readonly signal?: AbortSignal;
  readonly sink?: ConsoleSink | null;
}

/** Return the listing `reply` holds, else throw why the listing failed. `label` names the file. */
function listingOf(label: string, reply: ToolModuleReply): ToolModuleListing {
  if (reply.as === "listing") return reply.listing;
  throw replyError(label, reply);
}

/**
 * Return how `fresh` lists the module's exports otherwise than `first` did: a tool, in name order,
 * that is gone, is new or holds another value in a field, or other exports that changed; `null`
 * when they agree in every field.
 */
function listingChange(first: ToolModuleListing, fresh: ToolModuleListing): string | null {
  const was = new Map(first.tools.map((tool) => [tool.name, tool]));
  const now = new Map(fresh.tools.map((tool) => [tool.name, tool]));
  for (const name of [...new Set([...was.keys(), ...now.keys()])].sort(compareCodeUnits)) {
    const before = was.get(name);
    const after = now.get(name);
    const head = `export ${reprText(name)}`;
    if (after === undefined) return `${head} is no longer a tool`;
    if (before === undefined) return `${head} is a new tool`;
    const field = (Object.keys(before) as (keyof typeof before)[]).find(
      (key) => canonicalJson(before[key]) !== canonicalJson(after[key]),
    );
    if (field !== undefined) return `${head} ${field} changed`;
  }
  const others = (listing: ToolModuleListing) => canonicalJson({ refused: listing.refused, others: listing.others });
  return others(first) === others(fresh) ? null : "the exports that aren't tools changed";
}

/**
 * Return `listing` once it lists within `YIAN_TOOL_MODULE_START_TIMEOUT_SECONDS`.
 *
 * @param label How messages name the file.
 * @param signal Stops the wait.
 * @throws RangeError naming the file when the listing fails or outlasts the timeout.
 * @throws The signal's reason when it aborts first.
 */
async function listedInTime(
  label: string,
  listing: Promise<ToolModuleListing>,
  signal?: AbortSignal,
): Promise<ToolModuleListing> {
  const seconds = current().toolModuleStartTimeoutSeconds;
  // Ends the timer once the listing settles.
  const settled = new AbortController();
  const stops = signal === undefined ? [settled.signal] : [signal, settled.signal];
  try {
    return await Promise.race([
      listing.catch((error: Error) => {
        throw new YianRangeError(error.message, { cause: error });
      }),
      realScheduler.sleep(seconds * 1000, AbortSignal.any(stops)).then(() => {
        throw new YianRangeError(`${label} didn't import within ${seconds} s (YIAN_TOOL_MODULE_START_TIMEOUT_SECONDS)`);
      }),
    ]);
  } finally {
    settled.abort();
  }
}

/** Return what `promise` settles with, or reject with `signal`'s reason once it aborts first. */
function untilStopped<T>(promise: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
  if (signal === undefined) return promise;
  return new Promise((resolve, reject) => {
    const stop = (): void => reject(signal.reason);
    if (signal.aborted) stop();
    signal.addEventListener("abort", stop, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", stop));
  });
}

/** Return a failed check, as `ToolCheckError` with `message`, that an AGENT's model reads. */
function reportedCheck(message: string): ToolCheckError {
  const error = new ToolCheckError(message);
  Object.defineProperty(error, TOOL_ERROR, { value: Object.freeze(["check", "reported"]) });
  return error;
}

/**
 * Return the error a reply rejects its request with: what the call's tool threw, a refused result
 * as a failed check, or why the request failed. `label` names the file.
 */
function replyError(label: string, reply: ToolModuleReply): Error {
  switch (reply.as) {
    case "thrown":
      return rebuiltError(reply);
    case "refused":
      return new ToolCheckError(reply.message, { cause: new TypeCheckError(reply.message, { path: reply.path }) });
    case "failed":
      return new Error(reply.message);
    default:
      return new Error(`${label}'s thread sent a ${reply.as} reply out of turn`);
  }
}

/**
 * Return a call's error as the thread read it: an `Error` with the thrown one's name, message and
 * stack, else with the thrown value's text (and the value as its `cause` when it is a yian value,
 * as a host tool's thrown value is wrapped); branded with its tool error kinds, and holding its
 * `retryAfter` when transient.
 */
function rebuiltError(reply: Extract<ToolModuleReply, { as: "thrown" }>): Error {
  const { thrown, kinds, retryAfter } = reply;
  let error: Error;
  if (thrown.kind === "error") {
    error = new Error(thrown.message);
    error.name = thrown.name;
    if (thrown.stack !== undefined) error.stack = thrown.stack;
  } else if (thrown.kind === "value") {
    const value = unpackFiles(thrown.value);
    error = new Error(String(value), { cause: value });
  } else {
    error = new Error(thrown.text);
  }
  if (kinds.length > 0) Object.defineProperty(error, TOOL_ERROR, { value: Object.freeze([...kinds]) });
  if (kinds.includes("transient")) Object.defineProperty(error, "retryAfter", { value: retryAfter });
  return error;
}

/**
 * Start the thread of the module at `path` and list its tools, within
 * `YIAN_TOOL_MODULE_START_TIMEOUT_SECONDS`. A thread whose listing fails, takes too long or is
 * stopped is terminated.
 *
 * @param label How messages name the file.
 * @param signal Stops the start.
 * @returns The thread, still running, and its listing.
 * @throws RangeError naming the file when the import throws or outlasts the timeout, or the thread
 *   fails or exits first.
 * @throws The signal's reason when it aborts first.
 */
export async function startToolModule(
  path: string,
  label: string,
  signal: AbortSignal,
): Promise<{ readonly thread: ToolModuleThread; readonly listing: ToolModuleListing }> {
  signal.throwIfAborted();
  const thread = new ToolModuleThread(path, label);
  try {
    return { thread, listing: await listedInTime(label, thread.list(), signal) };
  } catch (error) {
    await thread.terminate();
    throw error;
  }
}
