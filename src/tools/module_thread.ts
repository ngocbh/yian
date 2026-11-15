/**
 * A tool module's thread, from the host's side: one worker thread per module, which imports it
 * once, lists its tools and runs their calls (`tools/module_worker`). It runs the CODE worker
 * entry, so no build lists another entry, and like a CODE thread it shares this process's live
 * environment and isn't a sandbox. Each request carries the caller's `current()`, so a run under
 * `withEnv` keeps its own values. The thread lives until it exits or is terminated, but its handle
 * here is referenced only while a request is open, so an idle thread never keeps this process
 * alive.
 *
 * A call's error is rebuilt here with the name, message, stack and tool error kinds it had on the
 * thread, so it steers an AGENT and TOOL retries as a host tool's error does.
 *
 * Knows about: tools/context, tools/errors, tools/module_worker, defaults, platform, typesys and
 * `node:*`.
 */

import { basename } from "node:path";
import { SHARE_ENV, Worker } from "node:worker_threads";

import { current } from "../defaults.ts";
import { realScheduler } from "../platform.ts";
import { packFiles, unpackFiles } from "../typesys/crossing.ts";
import { YianRangeError } from "../typesys/errors.ts";
import { TypeCheckError, toHost, type Value } from "../typesys/values.ts";
import type { ToolContext } from "./context.ts";
import { TOOL_ERROR, ToolCheckError } from "./errors.ts";
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

/** A tool module's thread, started when this is made. */
export class ToolModuleThread {
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
   * Call the module's tool exported as `tool` with `args` on the thread, and return its result as
   * a host value. Its context there holds `ctx`'s call id, attempt, run id and node, the export
   * name as its tool name, and no host object; a stop of `ctx.signal` aborts its signal there, and
   * the call settles as the tool does.
   *
   * @throws What the tool threw, rebuilt as an `Error` with its name, message and stack, its tool
   *   error kinds and, when transient, its `retryAfter`; a thrown non-`Error` is an `Error` with
   *   the value's text, and the value as its `cause` when it is a yian value.
   * @throws ToolCheckError when the result isn't a yian value, or is `pending()`.
   * @throws Error naming the file when the thread fails or exits first.
   */
  async call(tool: string, args: Record<string, unknown>, ctx: ToolContext): Promise<unknown> {
    const { callId, attempt, runId, node, signal } = ctx;
    const packed = packFiles(args);
    const reply = await this.#request(
      (id) => ({ kind: "call", id, defaults: current(), tool, args: packed, callId, attempt, runId, node }),
      signal,
    );
    if (reply.as === "result") return toHost(unpackFiles(reply.value) as Value);
    throw replyError(this.label, reply);
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
