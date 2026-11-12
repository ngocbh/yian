/**
 * The CODE node's worker thread: runs inline bodies and `./file.ts:fn` references one job at a
 * time and posts each outcome to the main thread, plus the helpers both threads use.
 *
 * The main thread (`nodes/code/node.ts`) starts a worker on this module with
 * `workerData.yianCodeWorker` set and posts it jobs (`{id, ...job}`); the module takes jobs only
 * when that key is set, so importing it for its helpers runs nothing. For each job the worker
 * installs the caller's defaults, builds the body (which the main thread already stripped of its
 * types, so no worker loads the stripper) or imports the reference (a module is imported once per
 * thread and cached there), calls it with the frozen `inputs` proxy, takes the result in
 * by host intake, and posts one `CodeReply` carrying the job's id (a result too deep for the copy
 * is refused as an output, by the commit's nesting check). It forwards each
 * `console.log/info/warn/error/debug` call as one formatted text on the same port, so the main
 * thread logs them in order. A late timer throw or unhandled rejection while a job runs replies
 * as a throw of that job's body (it may come from an earlier body on this thread); one while the
 * worker is idle is forwarded as a console error line. Either is marked `late`, and the main
 * thread retires the worker. While a job runs the port doesn't keep the thread alive, so a body
 * awaiting a promise nothing can resolve ends the thread instead of hanging it.
 *
 * Hosts that bundle yian import this module as `@ngocbh/yian/code-worker` for a bundle entry, so
 * every effect must stay behind the `yianCodeWorker` check.
 *
 * Values cross by structured clone, which drops a `FileRef`'s class, so a `FileRef` crosses as a
 * `Map` of its fields (no yian value holds a `Map`).
 *
 * Knows about: typesys/values, diagnostics, defaults and `node:*`.
 */

import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { format } from "node:util";
import { parentPort, workerData } from "node:worker_threads";

import { current, type Defaults, install } from "../../defaults.ts";
import { reprText } from "../../diagnostics.ts";
import { YianTypeError } from "../../typesys/errors.ts";
import {
  FileRef,
  fromHost,
  isList,
  isRecord,
  jsTypeName,
  rejectDeepNesting,
  setField,
  toHost,
  TypeCheckError,
  type Value,
} from "../../typesys/values.ts";

/** An inline body or a referenced function: a function of the `inputs` record. */
export type CodeBody = (inputs: Readonly<Record<string, unknown>>) => unknown;

// The constructor isn't a global; every async function's `constructor` is it.
export const AsyncFunction = (async () => {}).constructor as new (...args: string[]) => CodeBody;

/** What the main thread asks a worker to run. */
export interface CodeJob {
  readonly nodeId: string;
  readonly mode: "reference" | "inline";
  /** The `./file.ts:fn` reference, or the inline body already stripped of its types. */
  readonly code: string;
  readonly baseDir: string | null;
  /** The bound input record as a host value, through `packFiles`. */
  readonly inputs: unknown;
  /** The caller's `current()` values. */
  readonly defaults: Defaults;
}

/** A thrown `Error`'s fields. */
export interface ThrownError {
  kind: "error";
  name: string;
  message: string;
  stack?: string;
}

/** A thrown value as it crosses: an `Error`'s fields, a yian value (packed), or its text. */
export type Thrown = ThrownError | { kind: "value"; value: unknown } | { kind: "text"; text: string };

/**
 * A job's outcome: the result as a packed yian value; a failure the main thread wraps in a
 * `CodeError` (`"code"`) or rethrows as it is (`"plain"`); or a result intake refused.
 */
export type CodeReply =
  | { ok: true; value: unknown }
  | { ok: false; as: "code" | "plain"; thrown: Thrown }
  | { ok: false; as: "output"; message: string; path: (string | number)[] };

/** The console methods a worker forwards. */
export type ConsoleMethod = "log" | "info" | "warn" | "error" | "debug";

/**
 * A message a worker posts: one console call's text, or a job's reply with its id. `late` marks
 * an error that arrived after its body returned or from a body still running in the background.
 */
export type CodeMessage = ({ console: ConsoleMethod; text: string } | (CodeReply & { id: number })) & {
  late?: true;
};

/** The prologue an inline body is built with, on its first line, so line numbers hold. */
export const STRICT_PROLOGUE = '"use strict"; ';

/**
 * Build an inline body, already stripped of its types (`stripTypes`), as a strict async function
 * of `inputs`.
 *
 * The `sourceURL` comment names the body's stack frames `yian-code:<nodeId>:<line>:<column>`, and
 * the strict prologue shares the first line, so frame lines stay body lines plus the header (and a
 * first-line column is shifted by the prologue's length).
 *
 * @throws SyntaxError when the constructor refuses the stripped body.
 */
export function buildBody(nodeId: string, js: string): CodeBody {
  return new AsyncFunction("inputs", STRICT_PROLOGUE + js + "\n//# sourceURL=yian-code:" + nodeId);
}

/** Split a `./file.ts:fn` reference into its path and its export name. */
export function splitReference(reference: string): { path: string; name: string } {
  const colon = reference.lastIndexOf(":");
  return { path: reference.slice(0, colon), name: reference.slice(colon + 1) };
}

/** Return the error a CODE body throws when it reads an input the node doesn't declare. */
export function undeclaredInput(name: string): ReferenceError {
  return new ReferenceError(`CODE input ${reprText(name)} is not declared`);
}

/**
 * Return `record` behind a proxy that throws on a string key the record doesn't hold.
 *
 * The keys the runtime itself probes pass: inherited `Object.prototype` members, `then` (promise
 * resolution), `toJSON` (`JSON.stringify`) and symbols.
 */
export function inputsProxy(record: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  return new Proxy(record, {
    get(target, key, receiver) {
      if (typeof key === "symbol" || key in target || key === "then" || key === "toJSON") {
        return Reflect.get(target, key, receiver);
      }
      throw undeclaredInput(key);
    },
  });
}

/** Return a copy of a host value with each `FileRef` as a `Map` of its fields. */
export function packFiles(value: unknown): unknown {
  return swapLeaves(value, (part) =>
    FileRef.isFileRef(part)
      ? new Map([
          ["uri", part.uri],
          ["mime", part.mime],
          ["name", part.name],
        ])
      : part,
  );
}

/** Return a copy of a packed value with each `Map` back as a `FileRef`. */
export function unpackFiles(value: unknown): unknown {
  return swapLeaves(value, (part) =>
    part instanceof Map ? new FileRef({ uri: part.get("uri"), mime: part.get("mime"), name: part.get("name") }) : part,
  );
}

/** An open array or record in `swapLeaves`, with the fresh copy being filled. */
interface SwapFrame {
  readonly source: Record<string | number, unknown>;
  /** The record's keys, or `null` for an array. */
  readonly keys: readonly string[] | null;
  readonly copy: unknown[] | Record<string, unknown>;
  /** The number of items or keys to copy. */
  readonly size: number;
  index: number;
}

/**
 * Return a copy of the arrays and records of `value` with each other object part replaced by
 * `swap(part)`. A part shared by two parents is copied once, and the walk uses an explicit
 * stack, as `toHost` does, so any depth works.
 */
function swapLeaves(value: unknown, swap: (part: object) => unknown): unknown {
  const stack: SwapFrame[] = [];
  const copies = new Map<object, unknown[] | Record<string, unknown>>();
  const enter = (item: unknown): unknown => {
    if (typeof item !== "object" || item === null) return item;
    const list = isList(item);
    if (!list && !isRecord(item)) return swap(item);
    const shared = copies.get(item);
    if (shared !== undefined) return shared;
    const copy = list ? [] : {};
    copies.set(item, copy);
    const keys = list ? null : Object.keys(item);
    const size = keys === null ? (item as unknown[]).length : keys.length;
    stack.push({ source: item as Record<string, unknown>, keys, copy, size, index: 0 });
    return copy;
  };
  const root = enter(value);
  while (stack.length > 0) {
    const frame = stack[stack.length - 1]!;
    if (frame.index === frame.size) {
      stack.pop();
      continue;
    }
    const index = frame.index++;
    if (frame.keys === null) {
      (frame.copy as unknown[]).push(enter(frame.source[index]));
    } else {
      const key = frame.keys[index]!;
      setField(frame.copy as Record<string, Value>, key, enter(frame.source[key]) as Value);
    }
  }
  return root;
}

/**
 * Return how a value the code threw crosses to the main thread: an `Error`'s fields, a yian value,
 * else its text. A yian value nested too deep to copy back crosses as its type name.
 */
function thrownOf(value: unknown): Thrown {
  if (value instanceof Error) {
    const fields: ThrownError = { kind: "error", name: String(value.name), message: String(value.message) };
    if (typeof value.stack === "string") fields.stack = value.stack;
    return fields;
  }
  try {
    const thrown = fromHost(value, "error");
    // Too deep for the copy back, which would overflow in `postMessage`: name its type instead.
    if (tooDeepToCopy(thrown)) return { kind: "text", text: jsTypeName(value) };
    return { kind: "value", value: packFiles(toHost(thrown)) };
  } catch {
    // Not a yian value: its text, else its type name when it has none, as a stage marker reads.
    try {
      return { kind: "text", text: String(value) };
    } catch {
      return { kind: "text", text: jsTypeName(value) };
    }
  }
}

/** Import a reference's module, resolved against `baseDir` (the current directory when `null`). */
async function importReference(reference: string, baseDir: string | null): Promise<CodeBody | CodeReply> {
  const { path, name } = splitReference(reference);
  let module: Record<string, unknown>;
  try {
    module = await import(pathToFileURL(resolve(baseDir ?? process.cwd(), path)).href);
  } catch (error) {
    return { ok: false, as: "code", thrown: thrownOf(error) };
  }
  const fn = module[name];
  if (typeof fn !== "function") throw new YianTypeError(`${reprText(path)} exports no function ${reprText(name)}`);
  return fn as CodeBody;
}

// Copying a result back recurses on both threads: measured, Node's main thread reads about
// 2,100 levels and Bun about 40,000. A deeper result is refused here, by the commit's own check;
// a shallower one over the limit is copied and still meets the node's post-asserts first.
// Not a YIAN_* setting: YIAN_MAX_VALUE_NESTING is at most 100, so this bound never refuses an
// accepted value; it only picks which check reports a too-deep result. Raised past what the
// copy carries (about 2,100 levels on Node), it would let the copy crash instead.
const COPY_DEPTH = 1000;

/** Whether `value` nests deeper than the copy back to the main thread can carry. */
function tooDeepToCopy(value: Value): boolean {
  try {
    rejectDeepNesting(value, "output", { enclosingLevels: current().maxValueNesting - COPY_DEPTH });
    return false;
  } catch (error) {
    if (error instanceof TypeCheckError) return true;
    throw error;
  }
}

/**
 * Run `job` and return its reply.
 *
 * @throws Error (posted as `"plain"`) when the body doesn't build, the reference has no such
 *   export, or the code returns `undefined`.
 */
async function runJob(job: CodeJob): Promise<CodeReply> {
  const body = job.mode === "reference" ? await importReference(job.code, job.baseDir) : buildBody(job.nodeId, job.code);
  if (typeof body !== "function") return body;
  const record = Object.freeze(unpackFiles(job.inputs) as Record<string, unknown>);
  const proxy = inputsProxy(record);
  let result: unknown;
  try {
    result = await body(proxy);
  } catch (error) {
    return { ok: false, as: "code", thrown: thrownOf(error) };
  }
  if (result === undefined) throw new Error("CODE body returned no value; return null for no value");
  let value: Value;
  try {
    // Intake runs here, not after the copy: the clone would turn a class instance or an accessor
    // into a plain record and refuse a function with its own error.
    value = fromHost(result === proxy ? record : result, "output");
    if (tooDeepToCopy(value)) rejectDeepNesting(value, "output");
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    return { ok: false, as: "output", message: error.message, path: [...error.path] };
  }
  return { ok: true, value: packFiles(value) };
}

const port = parentPort;
if ((workerData as { yianCodeWorker?: boolean } | null)?.yianCodeWorker === true && port !== null) {
  /** The id of the job running now, else `null`. */
  let running: number | null = null;
  // Building or posting a reply can throw (a getter on a thrown error, a value the clone
  // refuses); the main thread must still hear back.
  const reply = (id: number, make: () => CodeReply, late?: true): void => {
    try {
      port.postMessage({ ...make(), id, late } satisfies CodeMessage);
    } catch (error) {
      port.postMessage({ ok: false, as: "plain", thrown: thrownOf(error), id, late } satisfies CodeMessage);
    }
  };
  const lateError = (error: unknown): void => {
    if (running !== null) {
      reply(running, () => ({ ok: false, as: "code", thrown: thrownOf(error) }), true);
      return;
    }
    const text = error instanceof Error ? `${error.name}: ${error.message}` : format(error);
    port.postMessage({ console: "error", text: `CODE error after its body returned: ${text}`, late: true });
  };
  process.on("uncaughtException", lateError);
  process.on("unhandledRejection", lateError);
  // Under `bun test` a worker's uncaught throw or unhandled rejection skips the `process.on`
  // handlers but still dispatches the global `error` event (never `unhandledrejection`), and the
  // worker then dies; `preventDefault` doesn't save it.
  globalThis.addEventListener?.("error", (event) => {
    event.preventDefault();
    lateError((event as ErrorEvent).error);
  });
  for (const method of ["log", "info", "warn", "error", "debug"] as const) {
    console[method] = (...args: unknown[]) => {
      port.postMessage({ console: method, text: format(...args) } satisfies CodeMessage);
    };
  }
  port.on("message", (job: CodeJob & { id: number }) => {
    running = job.id;
    // A listened port keeps the thread alive; while a job runs only the body's own work should.
    port.unref();
    install(job.defaults);
    const done = (make: () => CodeReply): void => {
      reply(job.id, make);
      if (running === job.id) {
        running = null;
        port.ref();
      }
    };
    runJob(job).then(
      (outcome) => done(() => outcome),
      (error: unknown) => done(() => ({ ok: false, as: "plain", thrown: thrownOf(error) })),
    );
  });
}
