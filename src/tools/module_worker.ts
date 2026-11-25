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
 * A check runs a tool's validator, if any, then its `isConcurrencySafe` on what `execute` will get,
 * and keeps the validator's output here, by the check's id, until the host releases it: an output
 * may be any value, which the copy would flatten. A call runs its tool's `execute` on that output,
 * else on the call's own copy of the arguments, with a context of the call's fields, whose `signal`,
 * `interrupted` and `steered` the host aborts by message, and calls interleave. A call checked on an
 * earlier thread of the module, which exited, is checked again here, and its output kept as its check's. The output
 * validator runs on a result the host checked. Each validator's return is read as the registry reads it; values the host takes are
 * taken in by host intake here (so a class instance is refused before the copy flattens it), and a
 * throw crosses with its tool error kinds, which the host rebuilds. A result that is a pending
 * marker crosses as one. Values cross as `typesys/crossing` packs them.
 *
 * Each `console.log/info/warn/error/debug` call is posted as one formatted text, tagged with the
 * request the code that made it runs for (an `AsyncLocalStorage` store, which promises and timers
 * keep), or with none during the import.
 *
 * Knows about: tools/context, tools/errors, tools/function_tools, events' types, defaults,
 * diagnostics, typesys and `node:*`.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { pathToFileURL } from "node:url";
import { format } from "node:util";
import type { MessagePort } from "node:worker_threads";

import { type Defaults, install } from "../defaults.ts";
import { compareCodeUnits, getCloseMatches, reprText } from "../diagnostics.ts";
import type { ConsoleMethod } from "../events.ts";
import { packFiles, type Thrown, thrownOf, tooDeepToCopy, unpackFiles } from "../typesys/crossing.ts";
import type { JsonObject } from "../typesys/json.ts";
import { fromHost, jsTypeName, rejectDeepNesting, TypeCheckError } from "../typesys/values.ts";
import { isPending, ToolContext } from "./context.ts";
import { type ToolErrorKind, toolErrorKinds } from "./errors.ts";
import { FunctionTool, isFunctionTool, type StandardIssue, tool, type ToolOptions } from "./function_tools.ts";

/** What a tool module's thread starts with, as `workerData.yianToolModule`. */
export interface ToolModuleData {
  /** The module file's real path. */
  readonly path: string;
  /** How messages name the file. */
  readonly label: string;
}

/**
 * One check of a module tool's arguments by its validator, or of its result by its output
 * validator, as the host posts it.
 */
export interface ToolModuleCheck {
  /** The tool's export name. */
  readonly tool: string;
  /** The arguments or the result, a host value packed by `packFiles`. */
  readonly value: unknown;
  /** How the registry names what is checked, for messages: `tool 'lookup' arguments`. */
  readonly where: string;
}

/** One call of a module's tool, as the host posts it. */
export interface ToolModuleCall {
  /** The tool's export name. */
  readonly tool: string;
  /** The checked arguments, a host value packed by `packFiles`. */
  readonly args: unknown;
  /** The id of the check whose kept output `execute` gets, or `null` when it kept none. */
  readonly check: number | null;
  /** How the registry names the arguments, for messages of a check made again here. */
  readonly where: string;
  readonly callId: string;
  readonly attempt: number;
  readonly runId: string;
  readonly node: string;
}

/**
 * A message the host posts: a request for the module's listing, a check of a tool's arguments or
 * its result, or a call of one of its tools, each with its id and the caller's `current()`; the
 * stop, the interrupt or the steer of the open call `id`; or the release of the output check `id`
 * kept.
 */
export type ToolModuleRequest =
  | { readonly kind: "list"; readonly id: number; readonly defaults: Defaults }
  | ({ readonly kind: "check"; readonly id: number; readonly defaults: Defaults } & ToolModuleCheck)
  | ({ readonly kind: "output"; readonly id: number; readonly defaults: Defaults } & ToolModuleCheck)
  | ({ readonly kind: "call"; readonly id: number; readonly defaults: Defaults } & ToolModuleCall)
  | { readonly kind: "abort"; readonly id: number }
  | { readonly kind: "interrupt"; readonly id: number }
  | { readonly kind: "steer"; readonly id: number }
  | { readonly kind: "release"; readonly id: number };

/**
 * A thread's reply to a request: the listing; a passed check of the arguments, saying whether the
 * call is safe to run beside others and whether the validator's output was kept; a validator's
 * issues, or why its return isn't a Standard Schema result; a call's result or the output
 * validator's, a packed yian value; a call's pending marker; what a tool or a validator threw, with
 * its tool error kinds and, when transient, its `retryAfter`; why a result was refused; or why the
 * request failed (the import threw). A call whose check ran on an earlier thread is checked again
 * first, so its reply may be a check's.
 */
export type ToolModuleReply = { readonly id: number } & ReplyBody;

/**
 * One `console` call of the module's code, as the thread posts it: its method, its arguments as
 * one formatted text, and the id of the request the code ran for, or `null` for none (the import).
 */
export interface ToolModuleConsole {
  readonly console: ConsoleMethod;
  readonly text: string;
  readonly id: number | null;
}

/** One validator issue as it crosses: its message, and its path as text and numbers. */
export interface CrossedIssue {
  readonly message: string;
  readonly path: readonly (string | number)[];
}

/** A reply without its id. */
type ReplyBody =
  | { readonly as: "listing"; readonly listing: ToolModuleListing }
  | { readonly as: "checked"; readonly safe: boolean; readonly kept: boolean }
  | { readonly as: "issues"; readonly issues: readonly CrossedIssue[] }
  | { readonly as: "nonstandard"; readonly message: string }
  | { readonly as: "result"; readonly value: unknown }
  | { readonly as: "pending" }
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
  "execute", "isConcurrencySafe", "resumable", "gateTarget", "refuseUngated",
] satisfies (keyof ToolOptions)[];

/** The options only a host tool may set: a run's stand-in for a module's tool carries neither. */
const HOST_ONLY_KEYS = ["gateTarget", "refuseUngated"] as const;

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
    const built = build();
    const hostOnly = HOST_ONLY_KEYS.find((key) => built[key] !== null);
    if (hostOnly !== undefined) {
      return { refused: `${head} sets ${hostOnly}, which only a host tool passed in the run's tools may` };
    }
    return built;
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
 * A validator's return as the registry reads it: its output, or the reply for its issues or for a
 * return that isn't a Standard Schema result.
 */
type StandardRead =
  | { readonly as: "value"; readonly value: unknown }
  | Extract<ReplyBody, { as: "issues" | "nonstandard" }>;

/**
 * Read a validator's return as the registry does: an object whose `issues` is an array holds
 * issues, which cross with each path segment as a number or as text; one whose `issues` is absent
 * and which holds `value` holds the output; anything else isn't a Standard Schema result.
 *
 * @param head The validator, as messages name it: `tool 'lookup' arguments: the validator`.
 */
function standardRead(result: unknown, head: string): StandardRead {
  if (typeof result === "object" && result !== null) {
    const issues: unknown = (result as { issues?: unknown }).issues;
    if (Array.isArray(issues)) {
      const crossed = Array.from(issues, (issue: StandardIssue) => ({
        message: `${issue.message}`,
        path: Array.from(issue.path ?? [], (segment) => {
          const key = typeof segment === "object" && segment !== null ? segment.key : segment;
          return typeof key === "number" ? key : String(key);
        }),
      }));
      return { as: "issues", issues: crossed };
    }
    if (issues === undefined && "value" in result) return { as: "value", value: (result as { value: unknown }).value };
  }
  return { as: "nonstandard", message: `${head} returned ${jsTypeName(result)}, not a Standard Schema result` };
}

/**
 * Return whether `tool`'s `isConcurrencySafe` says a call with `args` may run beside others, as
 * the registry reads it: only exactly `true` is safe, and no function, a Promise or a throw isn't.
 */
function concurrencySafe(tool: FunctionTool, args: unknown): boolean {
  try {
    const check = tool.isConcurrencySafe;
    if (check === null) return false;
    const safe: unknown = check(args as Record<string, unknown>);
    if (typeof (safe as { then?: unknown } | null)?.then === "function") {
      // A Promise isn't `true`; handle its rejection so it can't crash the thread.
      Promise.resolve(safe).catch(() => {});
      return false;
    }
    return safe === true;
  } catch {
    return false;
  }
}

/**
 * Return the reply to a check of `tool`'s arguments: what its validator threw, its issues, or why
 * its return isn't a Standard Schema result; else whether `isConcurrencySafe` says the call is
 * safe, on what `execute` will get, after `keep` took the validator's output.
 */
async function checkReply(
  tool: FunctionTool,
  check: ToolModuleCheck & { readonly defaults: Defaults },
  keep: (output: unknown) => void,
): Promise<ReplyBody> {
  const read = await validated(tool, check, keep);
  if (read.as !== "value") return read;
  return { as: "checked", safe: concurrencySafe(tool, read.value), kept: tool.parametersValidator !== null };
}

/**
 * Return what `tool`'s validator makes of `check`'s arguments, once `keep` took its output, or the
 * arguments when it has none; else the reply for what it threw, its issues, or a return that
 * isn't a Standard Schema result.
 */
async function validated(
  tool: FunctionTool,
  check: ToolModuleCheck & { readonly defaults: Defaults },
  keep: (output: unknown) => void,
): Promise<StandardRead | ReplyBody> {
  const args = unpackFiles(check.value);
  const validator = tool.parametersValidator;
  if (validator === null) return { as: "value", value: args };
  let read: StandardRead;
  try {
    const returned: unknown = await validator["~standard"].validate(args);
    read = standardRead(returned, `${check.where}: the validator`);
  } catch (error) {
    return thrownReply(error);
  }
  // Another request may have installed its values while the validator ran.
  install(check.defaults);
  if (read.as === "value") keep(read.value);
  return read;
}

/**
 * Return the reply to a check of `tool`'s result by its output validator: what the validator
 * threw, its issues, or why its return isn't a Standard Schema result; else its output taken in.
 */
async function outputReply(
  tool: FunctionTool,
  check: ToolModuleCheck & { readonly defaults: Defaults },
): Promise<ReplyBody> {
  const validator = tool.outputValidator;
  if (validator === null) return { as: "failed", message: `${check.where}: the tool has no output validator` };
  let read: StandardRead;
  try {
    const returned: unknown = await validator["~standard"].validate(unpackFiles(check.value));
    read = standardRead(returned, `${check.where}: the output validator`);
  } catch (error) {
    return thrownReply(error);
  }
  if (read.as !== "value") return read;
  install(check.defaults);
  return takenIn(read.value, check.where);
}

/**
 * Return the reply to a call of `tool` with `args`: its result taken in, as a yian value, or why
 * it was refused; that it returned a pending marker; or what `execute` threw.
 *
 * @param head The export, as messages name it.
 */
async function callReply(
  head: string,
  tool: FunctionTool,
  args: unknown,
  call: ToolModuleCall & { readonly defaults: Defaults },
  signal: AbortSignal,
  interrupted: AbortSignal,
  steered: AbortSignal,
): Promise<ReplyBody> {
  const { callId, attempt, runId, node } = call;
  // The host's app object can't cross a thread.
  const fields = { callId, attempt, toolName: call.tool, runId, node, context: undefined };
  const ctx = new ToolContext({ ...fields, signal, interrupted, steered });
  let result: unknown;
  try {
    result = await tool.execute(args as Record<string, unknown>, ctx);
  } catch (error) {
    return thrownReply(error);
  }
  // Another call may have installed its values while this one ran.
  install(call.defaults);
  if (isPending(result)) return { as: "pending" };
  return takenIn(result === undefined ? null : result, `${head} result`);
}

/** Return the reply holding `value` taken in as a yian value, or why intake refused it, naming it as `where`. */
function takenIn(value: unknown, where: string): ReplyBody {
  try {
    // Intake runs here, not after the copy: the clone would flatten a class instance or an accessor.
    const taken = fromHost(value, where);
    if (tooDeepToCopy(taken)) rejectDeepNesting(taken, where);
    return { as: "result", value: packFiles(taken) };
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

/** The console methods a thread forwards. */
const CONSOLE_METHODS: readonly ConsoleMethod[] = ["log", "info", "warn", "error", "debug"];

/**
 * Serve a tool module's thread on `port`: import the module once, answer each request, and post
 * each line the module prints.
 */
export function serveToolModule(port: MessagePort, data: ToolModuleData): void {
  let imported: Promise<object> | null = null;
  /** The id of the request the running code serves, kept across its promises and timers. */
  const serving = new AsyncLocalStorage<number>();
  for (const method of CONSOLE_METHODS) {
    console[method] = (...args: unknown[]) => {
      const id = serving.getStore() ?? null;
      port.postMessage({ console: method, text: format(...args), id } satisfies ToolModuleConsole);
    };
  }
  /**
   * The id of the first request this thread got: the host numbers requests across the module's
   * threads, so a check below it ran on an earlier thread.
   */
  let first: number | null = null;
  /** The tools of the first listing, by export name, which calls run. */
  let tools: ReadonlyMap<string, FunctionTool> | null = null;
  /** Each open call's stop, interrupt and steer, which the host aborts by message, by request id. */
  const calls = new Map<number, Readonly<Record<"stop" | "interrupted" | "steered", AbortController>>>();
  /** Each validator's output a check kept for its calls, by the check's request id, until the host releases it. */
  const kept = new Map<number, unknown>();
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
      calls.get(request.id)?.stop.abort();
      return;
    }
    if (request.kind === "interrupt") {
      calls.get(request.id)?.interrupted.abort();
      return;
    }
    if (request.kind === "steer") {
      calls.get(request.id)?.steered.abort();
      return;
    }
    if (request.kind === "release") {
      kept.delete(request.id);
      return;
    }
    const { id, defaults } = request;
    first ??= id;
    // Made now, so a stop, an interrupt or a steer that arrives before the import ends still reaches the call.
    const open =
      request.kind === "call"
        ? { stop: new AbortController(), interrupted: new AbortController(), steered: new AbortController() }
        : null;
    if (open !== null) calls.set(id, open);
    install(defaults);
    // Started outside every request's store, so no request takes what the import prints.
    imported ??= import(pathToFileURL(data.path).href) as Promise<object>;
    const loading = imported;
    // What the module prints while it serves this request, from its promises and timers too.
    serving.run(id, () =>
      loading
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
            const tool = tools.get(request.tool);
            if (tool === undefined) return { as: "failed", message: `${head} isn't a tool` };
            if (request.kind === "check") return await checkReply(tool, request, (output) => kept.set(id, output));
            if (request.kind === "output") return await outputReply(tool, request);
            const { check } = request;
            // A call gets the output its check kept, else its own copy of the arguments, as on the host.
            let args = check === null ? unpackFiles(request.args) : kept.get(check);
            if (check !== null && !kept.has(check)) {
              if (check >= first!) {
                return { as: "failed", message: `${head}: its checked arguments aren't kept on this thread` };
              }
              // Checked on a thread that exited: check again, keeping the output for later attempts.
              const keep = (output: unknown) => kept.set(check, output);
              const read = await validated(tool, { ...request, value: request.args }, keep);
              if (read.as !== "value") return read;
              args = read.value;
            }
            const { stop, interrupted, steered } = open!;
            return await callReply(head, tool, args, request, stop.signal, interrupted.signal, steered.signal);
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
        .finally(() => calls.delete(id)),
    );
  });
}
