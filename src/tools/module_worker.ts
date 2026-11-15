/**
 * The job a tool module's thread runs, and which exports of a module are tools.
 *
 * The CODE worker entry (`nodes/code/worker.ts`) imports this module statically, so its bundle
 * holds it, and calls `serveToolModule` when `workerData.yianToolModule` is set. The thread
 * imports its module once and answers each request with the caller's `current()` installed. Its
 * port stays referenced, so a module awaiting a promise keeps the thread alive.
 *
 * Named exports are read in code-unit order; `default` is never a tool. A tool is a `tool()` value
 * from any yian copy, whose fields are copied, or a plain object (prototype `Object.prototype` or
 * `null`) with an `execute` function, which this copy's `tool()` builds, its functions called with
 * the object as `this`; its `name` must be its export's. A plain object missing a field, holding a
 * mistyped one or a key `tool()` doesn't take, and an export that can't be read (in its TDZ,
 * behind a throwing getter) are refused, naming the export. Anything else is ignored. Only ES
 * modules: Node and Bun list a CommonJS file's exports differently.
 *
 * A call runs its tool's `execute` with a context of the call's fields, whose `signal` the host
 * aborts by message, and calls interleave. The reply is the result, taken in by host intake (so a
 * class instance is refused here, before the copy flattens it), or the throw with its tool error
 * kinds, which the host rebuilds. Values cross as `typesys/crossing` packs them.
 *
 * Knows about: tools/context, tools/errors, tools/function_tools, defaults, diagnostics, typesys
 * and `node:*`.
 */

import { pathToFileURL } from "node:url";
import type { MessagePort } from "node:worker_threads";

import { type Defaults, install } from "../defaults.ts";
import { compareCodeUnits, getCloseMatches, reprText } from "../diagnostics.ts";
import { packFiles, type Thrown, thrownOf, tooDeepToCopy, unpackFiles } from "../typesys/crossing.ts";
import type { JsonObject } from "../typesys/json.ts";
import { fromHost, jsTypeName, rejectDeepNesting, TypeCheckError } from "../typesys/values.ts";
import { isPending, ToolContext } from "./context.ts";
import { type ToolErrorKind, toolErrorKinds } from "./errors.ts";
import { FunctionTool, isFunctionTool, tool, type ToolOptions } from "./function_tools.ts";

/** What a tool module's thread starts with, as `workerData.yianToolModule`. */
export interface ToolModuleData {
  /** The module file's real path. */
  readonly path: string;
  /** How messages name the file. */
  readonly label: string;
}

/** One call of a module's tool, as the host posts it. */
export interface ToolModuleCall {
  /** The tool's export name. */
  readonly tool: string;
  /** The checked arguments, a host value packed by `packFiles`. */
  readonly args: unknown;
  readonly callId: string;
  readonly attempt: number;
  readonly runId: string;
  readonly node: string;
}

/**
 * A message the host posts: a request for the module's listing or a call of one of its tools,
 * each with its id and the caller's `current()`, or the stop of the open call `id`.
 */
export type ToolModuleRequest =
  | { readonly kind: "list"; readonly id: number; readonly defaults: Defaults }
  | ({ readonly kind: "call"; readonly id: number; readonly defaults: Defaults } & ToolModuleCall)
  | { readonly kind: "abort"; readonly id: number };

/**
 * A thread's reply to a request: the listing; a call's result, a packed yian value; what its tool
 * threw, with its tool error kinds and, when transient, its `retryAfter`; why its result was
 * refused; or why the request failed (the import threw).
 */
export type ToolModuleReply = { readonly id: number } & ReplyBody;

/** A reply without its id. */
type ReplyBody =
  | { readonly as: "listing"; readonly listing: ToolModuleListing }
  | { readonly as: "result"; readonly value: unknown }
  | {
      readonly as: "thrown";
      readonly thrown: Thrown;
      readonly kinds: readonly ToolErrorKind[];
      readonly retryAfter: number | null;
    }
  | { readonly as: "refused"; readonly message: string; readonly path: readonly (string | number)[] }
  | { readonly as: "failed"; readonly message: string };

/** One tool a module exports, as its thread lists it; its name is its export's. */
export interface ListedTool {
  readonly name: string;
  readonly description: string;
  readonly parametersJsonSchema: JsonObject;
  readonly outputJsonSchema: JsonObject | null;
  readonly resumable: boolean;
  readonly hasParametersValidator: boolean;
  readonly hasOutputValidator: boolean;
  readonly hasIsConcurrencySafe: boolean;
}

/** A module's named exports as its thread lists them, each list in export-name order. */
export interface ToolModuleListing {
  readonly tools: readonly ListedTool[];
  /** The exports refused as tools or that can't be read, each with why. */
  readonly refused: readonly { readonly name: string; readonly message: string }[];
  /** The exports that aren't tools, `default` among them. */
  readonly others: readonly string[];
}

/** The keys a plain-object tool may hold: the fields of `tool()`'s options. */
const OPTION_KEYS: readonly string[] = [
  "name", "description", "parameters", "parametersJsonSchema", "output", "outputJsonSchema",
  "execute", "isConcurrencySafe", "resumable",
] satisfies (keyof ToolOptions)[];

/** The fields `tool()` needs besides `execute`. */
const REQUIRED_KEYS = ["name", "description", "parameters"] as const;

/** Why reading an export failed; its message is what the read threw, as text. */
class UnreadableExport extends Error {}

/** Return what `get` returns, or throw `UnreadableExport` with what it threw. */
function guarded<T>(get: () => T): T {
  try {
    return get();
  } catch (error) {
    throw new UnreadableExport(thrownText(error));
  }
}

/** Return `holder[key]`, as `guarded` reads it. */
function read(holder: object, key: PropertyKey): unknown {
  return guarded(() => (holder as Record<PropertyKey, unknown>)[key]);
}

/**
 * Return a thrown value's text: an `Error`'s name and message (its message alone when not
 * `named`), else the value as text, else its type name. Never throws.
 */
function thrownText(error: unknown, named = true): string {
  try {
    if (error instanceof Error) return named ? `${String(error.name)}: ${String(error.message)}` : String(error.message);
    return String(error);
  } catch {
    return jsTypeName(error);
  }
}

/** Return `fn` called with `owner` as `this`, or `fn` itself when it isn't a function. */
function boundTo(owner: object, fn: unknown): unknown {
  return typeof fn === "function" ? (...args: unknown[]) => Reflect.apply(fn, owner, args) : fn;
}

/**
 * Return what export `name` holding `value` is: a tool this copy built, why it is refused, or
 * `null` when it isn't a tool. Each of a plain object's fields is read once.
 *
 * @param head The export, as messages name it.
 * @throws UnreadableExport when reading the value or one of its fields throws.
 */
function exportOutcome(head: string, name: string, value: unknown): FunctionTool | { refused: string } | null {
  let shown: unknown;
  let build: () => FunctionTool;
  if (isFunctionTool(value)) {
    shown = read(value, "name");
    // The constructor copies each field; one an older copy lacks is `undefined`, read as absent.
    build = () => new FunctionTool(value);
  } else {
    if (typeof value !== "object" || value === null) return null;
    const proto = guarded(() => Object.getPrototypeOf(value));
    if (proto !== Object.prototype && proto !== null) return null;
    const execute = read(value, "execute");
    if (typeof execute !== "function") return null;
    const unknown = guarded(() => Object.keys(value)).find((key) => !OPTION_KEYS.includes(key));
    if (unknown !== undefined) {
      const [close] = getCloseMatches(unknown, OPTION_KEYS, 1);
      const hint = close === undefined ? "" : `; did you mean ${reprText(close)}?`;
      return { refused: `${head} has an unknown key ${reprText(unknown)}${hint}` };
    }
    const options = Object.fromEntries(OPTION_KEYS.map((key) => [key, key === "execute" ? execute : read(value, key)]));
    const missing = REQUIRED_KEYS.find((key) => options[key] === undefined);
    if (missing !== undefined) return { refused: `${head} has no ${reprText(missing)}` };
    shown = options.name;
    // Called as the object's methods, so one that reads `this` reads the object.
    options.execute = boundTo(value, execute);
    options.isConcurrencySafe = boundTo(value, options.isConcurrencySafe);
    build = () => tool(options as unknown as ToolOptions);
  }
  if (typeof shown !== "string") return { refused: `${head} name must be a string, got ${jsTypeName(shown)}` };
  if (shown !== name) return { refused: `${head} holds a tool named ${reprText(shown)}; rename one` };
  try {
    return build();
  } catch (error) {
    // `tool()` and `FunctionTool` name the tool first, by the export's name here: name the export.
    const message = thrownText(error, false);
    for (const prefix of [`tool ${reprText(name)} `, `FunctionTool ${reprText(name)} `]) {
      if (message.startsWith(prefix)) return { refused: `${head} ${message.slice(prefix.length)}` };
    }
    return { refused: `${head} is refused: ${message}` };
  }
}

/**
 * Return the listing of a module's `namespace`, and each listed tool, built by this copy of yian,
 * by export name.
 *
 * @param label How messages name the module's file: `crm.ts export 'find' holds a tool named
 *   'lookup'; rename one`.
 */
export function listToolModule(
  namespace: object,
  label: string,
): { readonly listing: ToolModuleListing; readonly tools: ReadonlyMap<string, FunctionTool> } {
  const tools = new Map<string, FunctionTool>();
  const refused: { name: string; message: string }[] = [];
  const others: string[] = [];
  // `Reflect.ownKeys` reads no binding: `Object.keys` throws on an export in its TDZ.
  const names = Reflect.ownKeys(namespace).filter((key) => typeof key === "string");
  for (const name of names.sort(compareCodeUnits)) {
    const head = `${label} export ${reprText(name)}`;
    let outcome: ReturnType<typeof exportOutcome>;
    try {
      outcome = name === "default" ? null : exportOutcome(head, name, read(namespace, name));
    } catch (error) {
      if (!(error instanceof UnreadableExport)) throw error;
      outcome = { refused: `${head} can't be read: ${error.message}` };
    }
    if (outcome === null) others.push(name);
    else if (outcome instanceof FunctionTool) tools.set(name, outcome);
    else refused.push({ name, message: outcome.refused });
  }
  const listed = [...tools.values()].map((each) => ({
    name: each.name,
    description: each.description,
    parametersJsonSchema: each.parametersJsonSchema,
    outputJsonSchema: each.outputJsonSchema,
    resumable: each.resumable,
    hasParametersValidator: each.parametersValidator !== null,
    hasOutputValidator: each.outputValidator !== null,
    hasIsConcurrencySafe: each.isConcurrencySafe !== null,
  }));
  return { listing: { tools: listed, refused, others }, tools };
}

/**
 * Return the reply to a call of `tool` (`undefined` when the module has no such tool): its result
 * taken in, as a yian value, or why it was refused, or what `execute` threw.
 *
 * @param head The export, as messages name it.
 */
async function callReply(
  head: string,
  tool: FunctionTool | undefined,
  call: ToolModuleCall & { readonly defaults: Defaults },
  signal: AbortSignal,
): Promise<ReplyBody> {
  if (tool === undefined) return { as: "failed", message: `${head} isn't a tool` };
  const { callId, attempt, runId, node } = call;
  // The host's app object can't cross a thread.
  const ctx = new ToolContext({ callId, attempt, toolName: call.tool, runId, node, context: undefined, signal });
  let result: unknown;
  try {
    result = await tool.execute(unpackFiles(call.args) as Record<string, unknown>, ctx);
  } catch (error) {
    return thrownReply(error);
  }
  // Another call may have installed its values while this one ran.
  install(call.defaults);
  if (isPending(result)) {
    return { as: "refused", message: `${head} returned pending(), which a tool module can't do yet`, path: [] };
  }
  const where = `${head} result`;
  try {
    // Intake runs here, not after the copy: the clone would flatten a class instance or an accessor.
    const value = fromHost(result === undefined ? null : result, where);
    if (tooDeepToCopy(value)) rejectDeepNesting(value, where);
    return { as: "result", value: packFiles(value) };
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    return { as: "refused", message: error.message, path: [...error.path] };
  }
}

/**
 * Return how a tool's throw crosses: as `thrownOf` reads it, with its tool error kinds and, when
 * transient, its `retryAfter`.
 */
function thrownReply(error: unknown): ReplyBody {
  const kinds = toolErrorKinds(error);
  let retryAfter: unknown = null;
  if (kinds.includes("transient")) {
    try {
      retryAfter = (error as { retryAfter?: unknown }).retryAfter;
    } catch {}
  }
  const after = typeof retryAfter === "number" ? retryAfter : null;
  return { as: "thrown", thrown: thrownOf(error), kinds, retryAfter: after };
}

/** Serve a tool module's thread on `port`: import the module once and answer each request. */
export function serveToolModule(port: MessagePort, data: ToolModuleData): void {
  let imported: Promise<object> | null = null;
  /** The tools of the first listing, by export name, which calls run. */
  let tools: ReadonlyMap<string, FunctionTool> | null = null;
  /** Each open call's stop, by request id. */
  const calls = new Map<number, AbortController>();
  // Building or posting a reply can throw (a getter on a thrown error, a value the clone
  // refuses); the host must still hear back.
  const reply = (id: number, make: () => ReplyBody): void => {
    try {
      port.postMessage({ id, ...make() } satisfies ToolModuleReply);
    } catch (error) {
      port.postMessage({ id, ...thrownReply(error) } satisfies ToolModuleReply);
    }
  };
  // The listener keeps the port referenced for the thread's life, unlike a CODE thread's.
  port.on("message", (request: ToolModuleRequest) => {
    if (request.kind === "abort") {
      calls.get(request.id)?.abort();
      return;
    }
    const { id, defaults } = request;
    // Made now, so a stop that arrives before the import ends still reaches the call.
    const stop = request.kind === "call" ? new AbortController() : null;
    if (stop !== null) calls.set(id, stop);
    install(defaults);
    imported ??= import(pathToFileURL(data.path).href) as Promise<object>;
    imported
      .then(
        async (namespace): Promise<ReplyBody> => {
          // Another request may have installed its values while this one waited.
          install(defaults);
          if (request.kind === "list") {
            const listed = listToolModule(namespace, data.label);
            tools ??= listed.tools;
            return { as: "listing", listing: listed.listing };
          }
          tools ??= listToolModule(namespace, data.label).tools;
          const head = `${data.label} export ${reprText(request.tool)}`;
          return await callReply(head, tools.get(request.tool), request, stop!.signal);
        },
        (error: unknown): ReplyBody => ({
          as: "failed",
          message: `${data.label} failed to import: ${thrownText(error)}`,
        }),
      )
      .then(
        (body) => reply(id, () => body),
        (error: unknown) => reply(id, () => thrownReply(error)),
      )
      .finally(() => calls.delete(id));
  });
}
