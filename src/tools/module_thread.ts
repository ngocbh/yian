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
 * thread, so it steers an AGENT and TOOL retries as a host tool's error does.
 *
 * Knows about: tools/context, tools/errors, tools/function_tools' types, tools/module_worker,
 * defaults, diagnostics, platform, typesys and `node:*`.
 */

import { basename } from "node:path";
import { SHARE_ENV, Worker } from "node:worker_threads";

import { current } from "../defaults.ts";
import { reprText } from "../diagnostics.ts";
import { realScheduler } from "../platform.ts";
import { packFiles, unpackFiles } from "../typesys/crossing.ts";
import { YianRangeError, YianTypeError } from "../typesys/errors.ts";
import { TypeCheckError, toHost, type Value } from "../typesys/values.ts";
import type { ToolContext } from "./context.ts";
import { isPermanentToolError, isToolCheckError, TOOL_ERROR, ToolCheckError } from "./errors.ts";
import type { StandardResult } from "./function_tools.ts";
import type { ToolModuleData, ToolModuleListing, ToolModuleReply, ToolModuleRequest } from "./module_worker.ts";

// As `nodes/code/node.ts` finds the entry: `bun build --compile` bundles it at
// `nodes/code/worker.js` beside the bundle's entry, whose URL this module then has.
const WORKER_URL = import.meta.url.endsWith(".ts")
  ? new URL("../nodes/code/worker.ts", import.meta.url)
  : import.meta.url.endsWith("/tools/module_thread.js")
    ? new URL("../nodes/code/worker.js", import.meta.url)
    : new URL("./nodes/code/worker.js", import.meta.url);

/** An open request: how it settles, and what to undo once it has. */
interface OpenRequest {
  resolve(reply: ToolModuleReply): void;
  reject(error: Error): void;
  release(): void;
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

/** A tool module's thread, started when this is made. */
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
  readonly #worker: Worker;
  /** Each open request, by id. */
  readonly #open = new Map<number, OpenRequest>();
  #lastId = 0;
  /** Why every later request fails, once the thread failed or exited. */
  #ended: Error | null = null;
  #exitCode: number | null = null;

  /**
   * Start the thread of the module at `path`.
   *
   * @param label How messages name the file; its base name by default.
   */
  constructor(path: string, label: string = basename(path)) {
    this.path = path;
    this.label = label;
    const data: ToolModuleData = { path, label };
    // `SHARE_ENV`: the module reads and writes the host's live environment, as a CODE body does.
    this.#worker = new Worker(WORKER_URL, { env: SHARE_ENV, workerData: { yianToolModule: data } });
    this.#worker.unref();
    this.#worker.on("message", (reply: ToolModuleReply) => {
      const request = this.#open.get(reply.id);
      if (request === undefined) return;
      this.#settled(reply.id);
      request.resolve(reply);
    });
    const failed = (error: unknown): void => {
      const text = error instanceof Error ? error.message : String(error);
      this.#end(new Error(`${label}'s thread failed: ${text}`, { cause: error }));
    };
    this.#worker.on("error", failed);
    this.#worker.on("messageerror", failed);
    this.#worker.on("exit", (code: number) => {
      this.#exitCode = code;
      this.#end(new Error(`${label}'s thread exited (code ${code})`));
    });
  }

  /** The code the thread exited with, or `null` while it runs. */
  get exitCode(): number | null {
    return this.#exitCode;
  }

  /**
   * List the module's tools, importing it first if no request has.
   *
   * @throws Error naming the file when the import throws, or the thread fails or exits first.
   */
  async list(): Promise<ToolModuleListing> {
    const reply = await this.#request((id) => ({ kind: "list", id, defaults: current() }));
    if (reply.as === "listing") return reply.listing;
    throw replyError(this.label, reply);
  }

  /**
   * Check `args` for the module's tool exported as `tool` on the thread, as the registry checks a
   * host tool's arguments with its validator: run the module's validator, if any, then its
   * `isConcurrencySafe` on what `execute` will get. The thread keeps the validator's output for
   * the call.
   *
   * @param name The tool's name in the registry, which messages name it by.
   * @returns The validator's issues, each path as text and numbers, or the checked arguments.
   * @throws TypeError, as the registry's, when the validator returns no Standard Schema result.
   * @throws What the validator threw, rebuilt as `call` rebuilds a tool's throw.
   * @throws Error naming the file when the thread fails or exits first.
   */
  async check(tool: string, name: string, args: unknown): Promise<StandardResult<CheckedArgs>> {
    const packed = packFiles(args);
    const where = `tool ${reprText(name)} arguments`;
    const reply = await this.#request((id) => ({ kind: "check", id, defaults: current(), tool, value: packed, where }));
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
   * @returns The validator's issues, each path as text and numbers, or its output as a host value.
   * @throws ToolCheckError, as the registry's, when the validator returns no Standard Schema
   *   result, or an output that isn't a yian value.
   * @throws What the validator threw, rebuilt as `call` rebuilds a tool's throw.
   * @throws Error naming the file when the thread fails or exits first.
   */
  async checkOutput(tool: string, name: string, value: unknown): Promise<StandardResult<unknown>> {
    const packed = packFiles(value);
    const where = `tool ${reprText(name)} result`;
    const reply = await this.#request((id) => ({ kind: "output", id, defaults: current(), tool, value: packed, where }));
    if (reply.as === "issues") return { issues: reply.issues };
    if (reply.as === "nonstandard") throw new ToolCheckError(reply.message);
    if (reply.as === "result") return { value: toHost(unpackFiles(reply.value) as Value) };
    throw replyError(this.label, reply);
  }

  /**
   * Call the module's tool exported as `tool` on the thread with the arguments `check` returned,
   * and return its result as a host value. `execute` gets the validator's output the thread kept,
   * else a fresh copy of the arguments. Its context there holds `ctx`'s call id, attempt, run id
   * and node, the export name as its tool name, and no host object; a stop of `ctx.signal` aborts
   * its signal there, and the call settles as the tool does. Once it settles, the thread releases
   * the kept output, unless a TOOL node may try the call again: its error is neither permanent nor
   * a failed check, and no stop ended it.
   *
   * @throws TypeError when `checked` isn't what `check` returned.
   * @throws What the tool threw, rebuilt as an `Error` with its name, message and stack, its tool
   *   error kinds and, when transient, its `retryAfter`; a thrown non-`Error` is an `Error` with
   *   the value's text, and the value as its `cause` when it is a yian value.
   * @throws ToolCheckError when the result isn't a yian value, or is `pending()`.
   * @throws Error naming the file when the thread fails or exits first.
   */
  async call(tool: string, checked: unknown, ctx: ToolContext): Promise<unknown> {
    if (!(checked instanceof CheckedArgs)) {
      throw new YianTypeError(`${this.label} export ${reprText(tool)} takes the arguments its check returned`);
    }
    const { callId, attempt, runId, node, signal } = ctx;
    const { check, args } = checked;
    let settled = true;
    try {
      const reply = await this.#request(
        (id) => ({ kind: "call", id, defaults: current(), tool, args, check, callId, attempt, runId, node }),
        signal,
      );
      if (reply.as === "result") return toHost(unpackFiles(reply.value) as Value);
      throw replyError(this.label, reply);
    } catch (error) {
      settled = signal.aborted || isPermanentToolError(error) || isToolCheckError(error);
      throw error;
    } finally {
      if (settled) this.#release(checked);
    }
  }

  /** Stop the thread; open requests reject as for an exit. */
  async terminate(): Promise<void> {
    await this.#worker.terminate();
  }

  /**
   * Post the request `make` builds with a fresh id and return its reply. While it is open, a stop
   * of `signal` is posted as its abort.
   *
   * @throws Error naming the file when the thread fails or exits first, or the request can't be
   *   posted.
   */
  #request(make: (id: number) => ToolModuleRequest, signal?: AbortSignal): Promise<ToolModuleReply> {
    if (this.#ended !== null) return Promise.reject(this.#ended);
    const id = ++this.#lastId;
    const abort = (): void => this.#worker.postMessage({ kind: "abort", id } satisfies ToolModuleRequest);
    return new Promise((resolve, reject) => {
      if (this.#open.size === 0) this.#worker.ref();
      this.#open.set(id, { resolve, reject, release: () => signal?.removeEventListener("abort", abort) });
      try {
        this.#worker.postMessage(make(id));
      } catch (error) {
        this.#settled(id);
        reject(error);
        return;
      }
      if (signal?.aborted) abort();
      else signal?.addEventListener("abort", abort, { once: true });
    });
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

  /** Fail every open request and every later one with `error`, unless an earlier end did. */
  #end(error: Error): void {
    this.#ended ??= error;
    for (const [id, request] of this.#open) {
      this.#settled(id);
      request.reject(this.#ended);
    }
  }
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
  const seconds = current().toolModuleStartTimeoutSeconds;
  const thread = new ToolModuleThread(path, label);
  // Ends the timer once the listing settles.
  const settled = new AbortController();
  try {
    const listing = await Promise.race([
      thread.list().catch((error: Error) => {
        throw new YianRangeError(error.message, { cause: error });
      }),
      realScheduler.sleep(seconds * 1000, AbortSignal.any([signal, settled.signal])).then(() => {
        throw new YianRangeError(`${label} didn't import within ${seconds} s (YIAN_TOOL_MODULE_START_TIMEOUT_SECONDS)`);
      }),
    ]);
    return { thread, listing };
  } catch (error) {
    await thread.terminate();
    throw error;
  } finally {
    settled.abort();
  }
}
