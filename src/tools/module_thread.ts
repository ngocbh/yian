/**
 * A tool module's thread, from the host's side: one worker thread per module, which imports it
 * once and lists its tools (`tools/module_worker`). It runs the CODE worker entry, so no build
 * lists another entry, and like a CODE thread it shares this process's live environment and
 * isn't a sandbox. Each request carries the caller's `current()`, so a run under `withEnv` keeps
 * its own values. The thread lives until it exits or is terminated, but its handle here is
 * referenced only while a request is open, so an idle thread never keeps this process alive.
 *
 * Knows about: tools/module_worker, defaults and `node:*`.
 */

import { basename } from "node:path";
import { SHARE_ENV, Worker } from "node:worker_threads";

import { current } from "../defaults.ts";
import type { ToolModuleData, ToolModuleListing, ToolModuleReply, ToolModuleRequest } from "./module_worker.ts";

// As `nodes/code/node.ts` finds the entry: `bun build --compile` bundles it at
// `nodes/code/worker.js` beside the bundle's entry, whose URL this module then has.
const WORKER_URL = import.meta.url.endsWith(".ts")
  ? new URL("../nodes/code/worker.ts", import.meta.url)
  : import.meta.url.endsWith("/tools/module_thread.js")
    ? new URL("../nodes/code/worker.js", import.meta.url)
    : new URL("./nodes/code/worker.js", import.meta.url);

/** A tool module's thread, started when this is made. */
export class ToolModuleThread {
  /** The module file's real path. */
  readonly path: string;
  /** How messages name the file. */
  readonly label: string;
  readonly #worker: Worker;
  /** How each open request settles, by id. */
  readonly #open = new Map<number, { resolve(listing: ToolModuleListing): void; reject(error: Error): void }>();
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
      if (reply.ok) request.resolve(reply.listing);
      else request.reject(new Error(reply.message));
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
  list(): Promise<ToolModuleListing> {
    if (this.#ended !== null) return Promise.reject(this.#ended);
    return new Promise((resolve, reject) => {
      const id = ++this.#lastId;
      if (this.#open.size === 0) this.#worker.ref();
      this.#open.set(id, { resolve, reject });
      this.#worker.postMessage({ id, defaults: current() } satisfies ToolModuleRequest);
    });
  }

  /** Stop the thread; open requests reject as for an exit. */
  async terminate(): Promise<void> {
    await this.#worker.terminate();
  }

  /** Forget the request `id`; with none left open, the handle no longer keeps this process alive. */
  #settled(id: number): void {
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
