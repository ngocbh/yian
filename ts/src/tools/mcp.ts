/**
 * MCP servers whose tools a flow calls as host tools.
 *
 * `mcpServer` describes a Model Context Protocol server: a command started as a subprocess that
 * speaks over stdio, or a streamable HTTP endpoint. `McpServer.open` starts the server's client,
 * lists the server's tools once and exposes each as a `FunctionTool`; `close` stops the client and
 * the subprocess. A failed call throws an `McpError` whose tool error kinds tell an AGENT's tool
 * loop whether its model reads the error and a TOOL node whether `retries:` apply.
 *
 * Knows about: the MCP SDK client (`@modelcontextprotocol/client`, imported when a server opens),
 * `defaults`, `platform` (the version), `tools/context`, `tools/errors`, `tools/function_tools`,
 * `llm/body` (exact JSON writes), typesys and diagnostics.
 */

import { type ChildProcess, spawn } from "node:child_process";
import type { Writable } from "node:stream";

import type { Client } from "@modelcontextprotocol/client";

import { current } from "../defaults.ts";
import { reprText } from "../diagnostics.ts";
import { encodeBody } from "../llm/body.ts";
import { VERSION } from "../platform.ts";
import { type JsonObject, parseJson } from "../typesys/json.ts";
import { isRecord, jsTypeName, writeAsText } from "../typesys/values.ts";
import type { ToolContext } from "./context.ts";
import { TOOL_ERROR, type ToolErrorKind } from "./errors.ts";
import { FunctionTool } from "./function_tools.ts";

/**
 * The brands of `McpError` and `McpServer`. yian checks them rather than `instanceof`, so an error
 * or server from another copy of yian (an npm install beside a workspace one) is still known.
 */
const MCP_ERROR = Symbol.for("yian.mcp_error");
const MCP_SERVER = Symbol.for("yian.mcp_server");

const REPORTED: readonly ToolErrorKind[] = Object.freeze(["reported"]);
const REPORTED_PERMANENT: readonly ToolErrorKind[] = Object.freeze(["reported", "permanent"]);
const REPORTED_TRANSIENT: readonly ToolErrorKind[] = Object.freeze(["reported", "transient"]);
const PERMANENT: readonly ToolErrorKind[] = Object.freeze(["permanent"]);

/** Base class of the errors yian throws for an MCP server. */
export class McpError extends Error {
  override name = "McpError";
}

Object.defineProperty(McpError.prototype, MCP_ERROR, { value: true });

/**
 * An MCP server didn't start, didn't list its tools in time, or listed bad tools.
 *
 * The message names the server's label. For a stdio server it ends with the last lines the server
 * wrote to stderr; for an HTTP server it names only the error type and HTTP status, never the URL,
 * which may hold a token.
 */
export class McpConnectError extends McpError {
  override name = "McpConnectError";
}

/**
 * An MCP server returned an error result for a tool call, or the call failed locally.
 *
 * For an error result the message is the result's content rendered as text, as a successful
 * call's text result is: text blocks, then one-line markers for media. For any other failure
 * while the server is up (for example structured content that fails the tool's output schema) it
 * names only the server, the tool and the error type. An AGENT's model reads it as the call's
 * result.
 */
export class McpToolError extends McpError {
  override name = "McpToolError";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = REPORTED;
}

/**
 * An MCP server answered a tool call with a JSON-RPC error.
 *
 * The message names the server and the tool and holds the server's error message and code. An
 * AGENT's model reads it as the call's result.
 */
export class McpRequestError extends McpError {
  override name = "McpRequestError";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = REPORTED;
  /** The JSON-RPC error code. */
  readonly code: number;

  /**
   * @param message The error message.
   * @param code The JSON-RPC error code.
   */
  constructor(message: string, code: number) {
    super(message);
    this.code = code;
  }
}

/**
 * An MCP server rejected a tool call as malformed or unknown; retrying can't help.
 *
 * Thrown for the JSON-RPC codes -32700, -32600, -32601 and -32602.
 */
export class McpInvalidRequestError extends McpRequestError {
  override name = "McpInvalidRequestError";
  override readonly [TOOL_ERROR]: readonly ToolErrorKind[] = REPORTED_PERMANENT;
}

/** A tool call took longer than the server's call timeout. */
export class McpTimeoutError extends McpError {
  override name = "McpTimeoutError";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = REPORTED_TRANSIENT;
  /** Always `null`: the next attempt waits the computed backoff. */
  readonly retryAfter: number | null = null;
}

/**
 * A tool call failed because the HTTP connection to the server failed.
 *
 * The message names only the server and the error type, never the URL or the error text, which
 * may hold a token.
 */
export class McpTransportError extends McpError {
  override name = "McpTransportError";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = REPORTED_TRANSIENT;
  /** Always `null`: the next attempt waits the computed backoff. */
  readonly retryAfter: number | null = null;
}

/**
 * The MCP server is gone: yian closed it, or its process, connection or session ended.
 *
 * Later calls to the same server throw it at once; the host still closes it.
 */
export class McpServerClosed extends McpError {
  override name = "McpServerClosed";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = PERMANENT;
}

/** The calling run stopped while the tool call waited, so the call was cancelled. */
export class McpCallCancelled extends McpError {
  override name = "McpCallCancelled";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = PERMANENT;
}

/** Thrown by `open()` and `openForRun()` when the server is already open or still opening. */
export class AlreadyOpenError extends Error {
  override name = "AlreadyOpenError";
}

/** Return whether `value` carries `brand`; never throws, so a hostile object has no brand. */
function hasBrand(value: unknown, brand: symbol): boolean {
  if (value === null || (typeof value !== "object" && typeof value !== "function")) return false;
  try {
    return (value as Record<symbol, unknown>)[brand] === true;
  } catch {
    return false;
  }
}

/** Return whether `value` is an `McpError` from any copy of yian. */
export function isMcpError(value: unknown): value is McpError {
  return hasBrand(value, MCP_ERROR);
}

/** The options of `mcpServer`. */
export interface McpServerOptions {
  /** The executable that starts a stdio server. Pass exactly one of `command` and `url`. */
  command?: string;
  /** The command's arguments. Only with `command`. */
  args?: readonly string[];
  /**
   * Variables merged over the MCP SDK's safe default environment (`PATH`, `HOME` and a few
   * others), so the host's secrets reach the server only when passed here. Only with `command`.
   */
  env?: Readonly<Record<string, string>>;
  /** The command's working directory; the host's when absent. Only with `command`. */
  cwd?: string;
  /** The `http` or `https` endpoint of a streamable HTTP server. */
  url?: string;
  /** HTTP headers sent with every request, for example `Authorization`. Only with `url`. */
  headers?: Readonly<Record<string, string>>;
  /**
   * Longest wait in seconds for one tool call. When absent, `YIAN_MCP_CALL_TIMEOUT_SECONDS` (300)
   * is read when the server opens.
   */
  timeout?: number;
  /**
   * Longest wait in seconds for the server to start and list its tools. When absent,
   * `YIAN_MCP_CONNECT_TIMEOUT_SECONDS` (30) is read when the server opens.
   */
  connectTimeout?: number;
  /** The name used in errors; the command's basename, or the URL's host, when absent. */
  label?: string;
  /** Whether the server's tools may run beside other calls: all of them, or those named. */
  concurrencySafe?: boolean | readonly string[];
}

/** The checked options of an MCP server, as `mcpServer` builds them. */
export interface McpServerSpec {
  readonly command: string | undefined;
  readonly args: readonly string[];
  readonly env: Readonly<Record<string, string>> | undefined;
  readonly cwd: string | undefined;
  readonly url: string | undefined;
  readonly headers: Readonly<Record<string, string>> | undefined;
  readonly timeout: number | undefined;
  readonly connectTimeout: number | undefined;
  readonly label: string | undefined;
  readonly concurrencySafe: boolean | readonly string[];
}

/**
 * Describe an MCP server, started as a command over stdio or reached by URL.
 *
 * Nothing starts until the server is opened with `open()`. An option given as `null` counts as
 * absent, as `undefined` does.
 *
 * @returns The server, not yet open.
 * @throws {TypeError} An option has the wrong type: `args` a string or holding a non-string, `env`
 *   or `headers` not a record of strings to strings, a timeout not a number, `concurrencySafe`
 *   neither a boolean nor an array of strings.
 * @throws {RangeError} Neither or both of `command` and `url` are given, `url` is not an `http` or
 *   `https` URL with a host, an option is given for the other kind of server, a timeout is not
 *   positive and finite, or `command` or `label` is empty.
 *
 * @example
 * const files = mcpServer({ command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem", "."] });
 */
export function mcpServer(options: McpServerOptions): McpServer {
  // As Python's None, a `null` option is absent.
  const absent = <T>(value: T | null): T | undefined => (value === null ? undefined : value);
  const command = absent(options.command);
  const args = absent(options.args);
  const env = absent(options.env);
  const cwd = absent(options.cwd);
  const url = absent(options.url);
  const headers = absent(options.headers);
  const timeout = absent(options.timeout);
  const connectTimeout = absent(options.connectTimeout);
  const label = absent(options.label);
  const concurrencySafe = absent(options.concurrencySafe);
  if ((command === undefined) === (url === undefined)) {
    throw new RangeError("mcpServer needs exactly one of command or url");
  }
  if (command !== undefined) {
    if (typeof command !== "string") {
      throw new TypeError(`mcpServer command must be a string, got ${jsTypeName(command)}`);
    }
    if (command === "") throw new RangeError("mcpServer command must not be empty");
    if (headers !== undefined) {
      throw new RangeError("mcpServer headers needs url; a command server takes env instead");
    }
  } else {
    if (typeof url !== "string") throw new TypeError(`mcpServer url must be a string, got ${jsTypeName(url)}`);
    if (urlHost(url) === "") throw new RangeError("mcpServer url must be an http or https URL with a host");
    // As Python's `bool(args)`: an empty args array or string counts as not given.
    const argsGiven = args !== undefined && (args as unknown) !== "" && !(Array.isArray(args) && args.length === 0);
    for (const [name, given] of [
      ["args", argsGiven],
      ["env", env !== undefined],
      ["cwd", cwd !== undefined],
    ] as const) {
      if (given) throw new RangeError(`mcpServer ${name} needs command; a url server takes headers instead`);
    }
  }
  const checkedArgs = checkArgs(args);
  const checkedEnv = checkStringRecord("env", env);
  const checkedHeaders = checkStringRecord("headers", headers);
  if (cwd !== undefined && typeof cwd !== "string") {
    throw new TypeError(`mcpServer cwd must be a string, got ${jsTypeName(cwd)}`);
  }
  checkSeconds("timeout", timeout);
  checkSeconds("connectTimeout", connectTimeout);
  if (label !== undefined) {
    if (typeof label !== "string") throw new TypeError(`mcpServer label must be a string, got ${jsTypeName(label)}`);
    if (label === "") throw new RangeError("mcpServer label must not be empty");
  }
  return new McpServer({
    command,
    args: checkedArgs,
    env: checkedEnv,
    cwd,
    url,
    headers: checkedHeaders,
    timeout,
    connectTimeout,
    label,
    concurrencySafe: checkConcurrencySafe(concurrencySafe),
  });
}

/**
 * Return the hostname of an `http` or `https` URL, or `""` when `url` is not one. The scheme and
 * `//` must be spelled out: `new URL` alone reads `http:example.com` as having a host.
 */
function urlHost(url: string): string {
  if (!/^https?:\/\//i.test(url)) return "";
  try {
    return new URL(url).hostname;
  } catch {
    return "";
  }
}

/** Return a frozen copy of `args`, or throw `TypeError` unless it is an array of strings. */
function checkArgs(args: unknown): readonly string[] {
  if (args === undefined) return Object.freeze([]);
  if (!Array.isArray(args)) {
    throw new TypeError("mcpServer args must be an array of strings, not a single string");
  }
  const copy: string[] = [];
  // Index every slot, so a hole reads as `undefined` and fails.
  for (let i = 0; i < args.length; i++) {
    const arg: unknown = args[i];
    if (typeof arg !== "string") throw new TypeError("mcpServer args must hold only strings");
    copy.push(arg);
  }
  return Object.freeze(copy);
}

/** Return a frozen copy of `value`, or throw `TypeError` unless it is a record of strings to strings. */
function checkStringRecord(name: string, value: unknown): Readonly<Record<string, string>> | undefined {
  if (value === undefined) return undefined;
  if (!isRecord(value)) throw new TypeError(`mcpServer ${name} must be a record of strings to strings`);
  const entries = Object.entries(value);
  for (const [key, item] of entries) {
    if (typeof item !== "string") {
      throw new TypeError(`mcpServer ${name} must map strings to strings; ${reprText(key)} doesn't`);
    }
  }
  return Object.freeze(Object.fromEntries(entries) as Record<string, string>);
}

/** Throw unless `value` is `undefined` or a positive, finite number of seconds. */
function checkSeconds(name: string, value: unknown): void {
  if (value === undefined) return;
  if (typeof value !== "number") {
    throw new TypeError(`mcpServer ${name} must be a number of seconds, got ${jsTypeName(value)}`);
  }
  if (!(value > 0 && value < Infinity)) {
    throw new RangeError(`mcpServer ${name} must be positive and finite, got ${value}`);
  }
}

/** Return `value` (a frozen copy of an array), `false` when absent, or throw `TypeError`. */
function checkConcurrencySafe(value: unknown): boolean | readonly string[] {
  const message = "mcpServer concurrencySafe must be a boolean or an array of tool names";
  if (value === undefined) return false;
  if (typeof value === "boolean") return value;
  if (!Array.isArray(value)) throw new TypeError(message);
  const names: string[] = [];
  for (let i = 0; i < value.length; i++) {
    const name: unknown = value[i];
    if (typeof name !== "string") throw new TypeError(message);
    names.push(name);
  }
  return Object.freeze(names);
}

/** A server that keeps returning cursors would otherwise keep `open()` listing forever. */
const MAX_PAGES = 100;

/** The owner of a server the host opened. */
const HOST = "host";

/** One connection of a server: what one `open()` started, and what `close()` ends. */
interface Generation {
  /** Aborted when the open fails, times out or is closed; ends a pending connect or listing. */
  readonly stop: AbortController;
  client: Client | undefined;
  transport: StdioTransport | undefined;
  /** `"host"`, or the token of the run that opened the server, once its tools are listed. */
  owner: string | undefined;
  tools: readonly FunctionTool[] | undefined;
  /** Each listed tool's `outputSchema`, by name, for the check of its results. */
  outputSchemas: ReadonlyMap<string, JsonObject>;
  /** The longest wait for one call in seconds, read when the server opened. */
  callTimeout: number;
  /** Whether `close()` ran on this connection. */
  closed: boolean;
  /** Whether the server is known gone: its transport closed. */
  gone: boolean;
  /** The calls in flight, which `close()` aborts. */
  readonly calls: Set<AbortController>;
  /** The shutdown, once started. */
  closing: Promise<void> | undefined;
}

/**
 * An MCP server whose tools a flow calls as host tools.
 *
 * Build one with `mcpServer`. While open, `tools` holds the tools the server listed when it
 * opened; a later change to the server's tool list is ignored. A server opened with `open()`
 * belongs to the host, which closes it; `await using` does both.
 *
 * @example
 * await using files = await mcpServer({ command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem", "."] }).open();
 * const names = files.tools.map((tool) => tool.name);
 */
export class McpServer {
  /** The name used in errors: the `label` option, else the command's basename, else the URL's host. */
  readonly label: string;
  /** The checked options, read when the server opens. */
  private readonly spec: McpServerSpec;
  /** The connection from `open()` until `close()`; a later open replaces it. */
  private generation: Generation | undefined;

  /** Keep a spec `mcpServer` checked; build a server with `mcpServer`, not this constructor. */
  constructor(spec: McpServerSpec) {
    this.spec = spec;
    if (spec.label !== undefined) this.label = spec.label;
    // As Python's POSIX `os.path.basename`: a command ending in `/` keeps the whole command.
    else if (spec.command !== undefined) this.label = spec.command.split("/").pop() || spec.command;
    else this.label = urlHost(spec.url ?? "") || "mcp";
  }

  /** Whether the server is open and its tools are listed. */
  get isOpen(): boolean {
    return this.generation?.owner !== undefined;
  }

  /** Whether an open server has gone away, so its calls fail at once until it is reopened. */
  get isClosed(): boolean {
    const generation = this.generation;
    return generation?.owner !== undefined && generation.gone;
  }

  /** Whether a run, not the host, opened the server and has not closed it. */
  get openedByRun(): boolean {
    const owner = this.generation?.owner;
    return owner !== undefined && owner !== HOST;
  }

  /**
   * The server's tools, in the order it listed them.
   *
   * @throws {Error} The server is not open.
   */
  get tools(): readonly FunctionTool[] {
    const generation = this.generation;
    if (generation?.owner === undefined || generation.tools === undefined) {
      throw new Error(`MCP server '${this.label}' is not open`);
    }
    return generation.tools;
  }

  /**
   * Start the server, wait until it lists its tools, and keep it running; the host must then
   * `close()` it.
   *
   * @param options.signal Aborting it while the server starts closes the server and rejects with
   *   the signal's reason.
   * @returns This server.
   * @throws {AlreadyOpenError} The server is already open, or still opening.
   * @throws {Error} The MCP SDK (`@modelcontextprotocol/client`) is not installed.
   * @throws {McpConnectError} The server doesn't start, doesn't list its tools within the connect
   *   timeout, lists a tool twice or a tool yian can't use, or `concurrencySafe` names a tool it
   *   doesn't list; or `close()` ran while it started.
   */
  open(options: { signal?: AbortSignal } = {}): Promise<this> {
    return this.start(HOST, options.signal);
  }

  /**
   * Open the server on behalf of the run `token`, which closes it when it ends.
   *
   * @param token The opening run's ownership token.
   * @param options.signal As `open()`'s.
   * @returns This server.
   * @throws As `open()` throws.
   */
  openForRun(token: string, options: { signal?: AbortSignal } = {}): Promise<this> {
    return this.start(token, options.signal);
  }

  /**
   * Stop the server and its client: end the calls in flight, then the client and the server's
   * process group. Never rejects; closing twice, or before `open()`, does nothing more. A close
   * that is still running when the server is opened again leaves the new connection alone.
   */
  close(): Promise<void> {
    const generation = this.generation;
    if (generation === undefined) return Promise.resolve();
    generation.closed = true;
    // Detached at once, so an `open()` right after starts a connection this close leaves alone.
    this.generation = undefined;
    return this.shutdown(generation);
  }

  /** Close the server, for `await using`. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.close();
  }

  /** Open a new connection and record `owner` once its tools are listed. */
  private async start(owner: string, signal: AbortSignal | undefined): Promise<this> {
    // Checked and set before the first `await`, so two racing opens can't both start.
    if (this.generation !== undefined) throw new AlreadyOpenError(`MCP server '${this.label}' is already open`);
    const generation: Generation = {
      stop: new AbortController(),
      client: undefined,
      transport: undefined,
      owner: undefined,
      tools: undefined,
      outputSchemas: new Map(),
      callTimeout: 0,
      closed: false,
      gone: false,
      calls: new Set(),
      closing: undefined,
    };
    this.generation = generation;
    try {
      await this.connect(generation, owner, signal);
    } catch (error) {
      await this.shutdown(generation);
      if (this.generation === generation) this.generation = undefined;
      throw error;
    }
    return this;
  }

  /**
   * Load the SDK, start the server, connect and list its tools within one connect deadline, and
   * record `owner`; throw the error `open()` reports.
   */
  private async connect(generation: Generation, owner: string, signal: AbortSignal | undefined): Promise<void> {
    const label = this.label;
    const settings = current();
    const connectTimeout = this.spec.connectTimeout ?? settings.mcpConnectTimeoutSeconds;
    generation.callTimeout = this.spec.timeout ?? settings.mcpCallTimeoutSeconds;
    const deadline = performance.now() + connectTimeout * 1000;
    const stop = generation.stop;
    let timedOut = false;
    let started = false;
    // The transport is closed directly, since `client.close()` can't reach one the SDK hasn't
    // attached yet.
    const halt = () => {
      stop.abort();
      void generation.transport?.close();
    };
    const timer = setTimeout(() => {
      timedOut = true;
      halt();
    }, connectTimeout * 1000);
    signal?.addEventListener("abort", halt, { once: true });
    // Run after every `await`: a timeout, `close()` or abort that landed while nothing could be
    // closed still wins.
    const check = () => {
      if (stop.signal.aborted || generation.closed || signal?.aborted) throw new Error("stopped while starting");
    };
    const request = () => ({ signal: stop.signal, timeout: Math.max(1, deadline - performance.now()) });
    try {
      check();
      const sdk = await loadSdk(this.spec.command !== undefined);
      check();
      const transport = (generation.transport = this.createTransport(sdk.defaultEnv, settings.maxMcpMessageBytes));
      const client = new sdk.Client({ name: "yian", version: VERSION }, { capabilities: {}, listMaxPages: MAX_PAGES });
      generation.client = client;
      client.onclose = () => {
        generation.gone = true;
      };
      started = true;
      await client.connect(transport, request());
      check();
      const listed = await listAll(
        {
          listTools: (params, options) => client.listTools(params, options),
          getServerCapabilities: () => client.getServerCapabilities(),
        },
        {
          label,
          concurrencySafe: this.spec.concurrencySafe,
          call: (name, args, ctx) => this.callTool(name, args, ctx),
          ...request(),
        },
      );
      check();
      generation.tools = listed.tools;
      generation.outputSchemas = listed.outputSchemas;
      generation.owner = owner;
    } catch (error) {
      const tail = generation.transport?.stderrTail() ?? "";
      const late = `MCP server '${label}' didn't start within ${connectTimeout} s` + tail;
      if (timedOut) throw new McpConnectError(late);
      if (generation.closed) throw new McpConnectError(`MCP server '${label}' was closed while starting`);
      if (signal?.aborted) throw signal.reason;
      // The SDK's timeout for a request is what was left of the deadline, so it may fire just
      // before yian's timer.
      if ((error as { code?: unknown } | null)?.code === "REQUEST_TIMEOUT") throw new McpConnectError(late);
      if (!started || error instanceof McpConnectError) throw error;
      throw new McpConnectError(`MCP server '${label}' didn't start: ${describeError(error)}` + tail, { cause: error });
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", halt);
    }
  }

  /** Return the transport for this server. */
  private createTransport(defaultEnv: Readonly<Record<string, string>>, maxMessageBytes: number): StdioTransport {
    const { command, args, env, cwd } = this.spec;
    if (command === undefined) throw new Error(`MCP server '${this.label}': HTTP servers are not ported yet`);
    return new StdioTransport({ command, args, env, cwd, maxMessageBytes, defaultEnv });
  }

  /** End `generation`: its calls, its client and its transport. Never rejects; runs once. */
  private shutdown(generation: Generation): Promise<void> {
    return (generation.closing ??= (async () => {
      generation.stop.abort();
      for (const call of generation.calls) call.abort();
      generation.calls.clear();
      const transportClosed = generation.transport?.close();
      try {
        await generation.client?.close();
      } catch {
        // The transport's own close below still ends the server.
      }
      await transportClosed;
    })());
  }

  /** Call the server's tool `name`. */
  private callTool(_name: string, _args: Record<string, unknown>, _ctx: ToolContext<unknown>): Promise<unknown> {
    return Promise.reject(new Error("MCP calls are not ported yet"));
  }
}

Object.defineProperty(McpServer.prototype, MCP_SERVER, { value: true });

/** Return whether `value` is an `McpServer` from any copy of yian. */
export function isMcpServer(value: unknown): value is McpServer {
  return hasBrand(value, MCP_SERVER);
}

/**
 * Import the MCP SDK client: its `Client`, and for a stdio server its default environment.
 *
 * @throws {Error} The SDK is not installed; the import error is the `cause`.
 */
async function loadSdk(stdio: boolean): Promise<{ Client: typeof Client; defaultEnv: Record<string, string> }> {
  try {
    const sdk = await import("@modelcontextprotocol/client");
    const defaultEnv = stdio ? (await import("@modelcontextprotocol/client/stdio")).getDefaultEnvironment() : {};
    return { Client: sdk.Client, defaultEnv };
  } catch (cause) {
    throw new Error("MCP servers need the MCP SDK; install @modelcontextprotocol/client 2.3.1", { cause });
  }
}

/** `<Name>: <message>` for an error, as Python's `type(error).__name__: error`. */
function describeError(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : `${jsTypeName(error)}: ${String(error)}`;
}

/** What `listAll` reads from a connected MCP client. */
export interface ToolLister {
  /** List every page of the server's tools. */
  listTools(
    params: undefined,
    options: { signal?: AbortSignal; timeout?: number },
  ): Promise<{ readonly tools: readonly unknown[] }>;
  /** The capabilities the server declared, or `undefined`. */
  getServerCapabilities(): { readonly tools?: unknown } | undefined;
}

/** The options of `listAll`. */
export interface ListOptions {
  /** The server's label, for errors. */
  label: string;
  /** The server's `concurrencySafe` option. */
  concurrencySafe: boolean | readonly string[];
  /** Runs one call of the server's tool `name`; each tool's `execute` delegates to it. */
  call: (name: string, args: Record<string, unknown>, ctx: ToolContext<unknown>) => Promise<unknown>;
  /** Aborts the listing. */
  signal?: AbortSignal;
  /** The listing's timeout in milliseconds. */
  timeout?: number;
}

/**
 * List the server's tools and wrap each in a `FunctionTool`. A server that declares no `tools`
 * capability lists none, and gets no `tools/list` request.
 *
 * @returns The tools, frozen and in listing order, and each listed `outputSchema` by tool name.
 * @throws {McpConnectError} The server lists more than 100 pages, a tool twice, a tool whose input
 *   schema's type is not `"object"` or a tool `FunctionTool` refuses; or `concurrencySafe` names
 *   a tool it doesn't list.
 */
export async function listAll(
  lister: ToolLister,
  options: ListOptions,
): Promise<{ tools: readonly FunctionTool[]; outputSchemas: Map<string, JsonObject> }> {
  const { label, concurrencySafe, call, signal, timeout } = options;
  const tools: FunctionTool[] = [];
  const outputSchemas = new Map<string, JsonObject>();
  const seen = new Set<string>();
  // As OpenCode does: the SDK would answer for such a server itself, with a `console.debug` line.
  if (lister.getServerCapabilities()?.tools) {
    let listed: readonly unknown[];
    try {
      ({ tools: listed } = await lister.listTools(undefined, { signal, timeout }));
    } catch (error) {
      if ((error as { code?: unknown } | null)?.code === "LIST_PAGINATION_EXCEEDED") {
        throw new McpConnectError(`MCP server '${label}' lists more than ${MAX_PAGES} pages of tools`);
      }
      throw error;
    }
    for (const tool of listed) {
      const { name, title, description, inputSchema, outputSchema } = tool as Record<string, unknown>;
      const shown = reprText(String(name));
      if (seen.has(name as string)) throw new McpConnectError(`MCP server '${label}' lists the tool ${shown} twice`);
      if (!isRecord(inputSchema) || inputSchema.type !== "object") {
        throw new McpConnectError(
          `MCP server '${label}' tool ${shown} has an input schema whose type is not 'object'`,
        );
      }
      const marked = concurrencySafe === true || (concurrencySafe !== false && concurrencySafe.includes(name as string));
      try {
        tools.push(
          new FunctionTool({
            name: name as string,
            description: (description || title || name) as string,
            parametersJsonSchema: inputSchema as JsonObject,
            isConcurrencySafe: marked ? () => true : null,
            execute: (args, ctx) => call(name as string, args, ctx),
          }),
        );
      } catch (error) {
        throw new McpConnectError(`MCP server '${label}' tool ${shown} is unusable: ${(error as Error).message}`, {
          cause: error,
        });
      }
      seen.add(name as string);
      if (isRecord(outputSchema)) outputSchemas.set(name as string, outputSchema as JsonObject);
    }
  }
  if (typeof concurrencySafe !== "boolean") {
    const unknown = concurrencySafe.filter((name) => !seen.has(name));
    if (unknown.length > 0) {
      throw new McpConnectError(
        `MCP server '${label}' marks tools it doesn't list as concurrency-safe: ${writeAsText(unknown)}`,
      );
    }
  }
  return { tools: Object.freeze(tools), outputSchemas };
}

/** How much of a stdio server's stderr yian keeps, and how many of its last lines an error shows. */
const STDERR_KEEP_CHARS = 16 * 1024;
const STDERR_LINES = 20;

/** The options of `StdioTransport`. */
export interface StdioTransportOptions {
  /** The executable to start. */
  command: string;
  /** Its arguments. */
  args: readonly string[];
  /** Variables merged over `defaultEnv`. */
  env?: Readonly<Record<string, string>>;
  /** The working directory; the host's when absent. */
  cwd?: string;
  /** The longest line read from the server, in bytes. */
  maxMessageBytes: number;
  /** The MCP SDK's `getDefaultEnvironment()`, the base of the server's environment. */
  defaultEnv: Readonly<Record<string, string>>;
}

/**
 * The stdio transport yian gives the MCP SDK client: it starts the server in its own process
 * group, writes and reads newline-delimited JSON-RPC with exact big ints, keeps the tail of the
 * server's stderr, and on close ends the whole group, so a wrapper (`npx`, `uvx`) leaves no
 * grandchildren behind. It has the SDK `Transport` shape without naming the SDK's types.
 */
export class StdioTransport {
  /** Called once when the server is gone: it exited, its stdout ended, or `close()` ran. */
  onclose?: () => void;
  /** Called with a write, read or decode error, or one thrown by `onmessage` or `onclose`. */
  onerror?: (error: Error) => void;
  /** Called with each JSON-RPC message the server sends. */
  onmessage?: (message: unknown) => void;
  private readonly options: StdioTransportOptions;
  private child: ChildProcess | undefined;
  /** stdout bytes after the last newline, and their total length. */
  private pending: Buffer[] = [];
  private pendingBytes = 0;
  private stderrText = "";
  private stderrEnded = false;
  private exited = false;
  private closed = false;
  private ended = false;
  private closeFired = false;
  private closing: Promise<void> | undefined;
  /** Settles on stdin's next `drain` or `close`; every send waiting on a full pipe shares it. */
  private drained: Promise<void> | undefined;
  private readonly settled: Promise<void>;
  private settle!: () => void;

  /** Keep `options`; nothing starts until `start()`. */
  constructor(options: StdioTransportOptions) {
    this.options = options;
    this.settled = new Promise((resolve) => (this.settle = resolve));
  }

  /** The server's process id, or `null` before it started. */
  get pid(): number | null {
    return this.child?.pid ?? null;
  }

  /** The last lines the server wrote to stderr, as an error message suffix, or `""`. */
  stderrTail(): string {
    if (this.stderrText === "") return "";
    const lines = this.stderrText.split(/\r\n|\r|\n/);
    if (lines[lines.length - 1] === "") lines.pop();
    return "\nserver stderr:\n" + lines.slice(-STDERR_LINES).join("\n");
  }

  /** Start the server; reject with the spawn error when it can't start. */
  start(): Promise<void> {
    if (this.closed) return Promise.reject(new Error("StdioTransport is closed"));
    if (this.child !== undefined) return Promise.reject(new Error("StdioTransport already started"));
    const { command, args, env, cwd, defaultEnv } = this.options;
    return new Promise((resolve, reject) => {
      let child: ChildProcess;
      try {
        child = spawn(command, [...args], {
          cwd,
          env: { ...defaultEnv, ...env },
          stdio: ["pipe", "pipe", "pipe"],
          // Its own process group, so close can signal the server and everything it started.
          detached: true,
          windowsHide: true,
        });
      } catch (error) {
        // Bun may throw here where Node emits `error`.
        this.closed = true;
        reject(error);
        return;
      }
      this.child = child;
      let spawned = false;
      child.once("spawn", () => {
        spawned = true;
        resolve();
      });
      child.on("error", (error) => {
        if (spawned) this.report(error);
        else {
          this.closed = true;
          reject(error);
        }
      });
      child.once("exit", () => {
        this.exited = true;
        this.end();
      });
      child.stdin!.on("error", (error) => this.report(error));
      child.stdout!.on("error", (error) => this.report(error));
      child.stdout!.on("data", (chunk: Buffer) => this.read(chunk));
      child.stdout!.once("end", () => this.end());
      child.stderr!.on("error", () => {});
      child.stderr!.setEncoding("utf8");
      child.stderr!.on("data", (text: string) => {
        this.stderrText = (this.stderrText + text).slice(-STDERR_KEEP_CHARS);
      });
      child.stderr!.once("close", () => (this.stderrEnded = true));
    });
  }

  /** Write `message` as one line; reject when closed or the write fails. */
  send(message: unknown): Promise<void> {
    const stdin = this.child?.stdin;
    if (this.closed || this.ended || stdin == null) return Promise.reject(new Error("Not connected"));
    let line: string;
    try {
      line = encodeBody(message, "MCP message") + "\n";
    } catch (error) {
      return Promise.reject(error);
    }
    return new Promise((resolve, reject) => {
      const flushed = stdin.write(line, (error) => {
        if (error) reject(error);
      });
      if (flushed) resolve();
      else this.drain(stdin).then(resolve, reject);
    });
  }

  /** Wait for `stdin` to drain; reject when it closes first. */
  private drain(stdin: Writable): Promise<void> {
    return (this.drained ??= new Promise<void>((resolve, reject) => {
      const done = (error?: Error) => {
        stdin.off("drain", onDrain);
        stdin.off("close", onClose);
        this.drained = undefined;
        if (error) reject(error);
        else resolve();
      };
      const onDrain = () => done();
      const onClose = () => done(new Error("Not connected"));
      stdin.once("drain", onDrain);
      stdin.once("close", onClose);
    }));
  }

  /**
   * End the server: close its stdin, give it 2 s to exit, then SIGTERM its process group, give
   * the group 2 s, then SIGKILL it. Never rejects; a second call returns the first one's promise.
   */
  close(): Promise<void> {
    return (this.closing ??= this.shutdown());
  }

  /** The body of `close()`, run once. */
  private async shutdown(): Promise<void> {
    this.closed = true;
    const child = this.child;
    const pid = child?.pid;
    if (child === undefined || pid === undefined) {
      this.fireClose();
      return;
    }
    const exit = new Promise<void>((resolve) => {
      if (this.exited) resolve();
      else child.once("exit", () => resolve());
    });
    try {
      child.stdin?.end();
    } catch {
      // Already gone.
    }
    await within(exit, 2000);
    // Even when the leader exited, what it started may still run in its group.
    if (signalGroup(pid, "SIGTERM")) {
      const deadline = Date.now() + 2000;
      while (signalGroup(pid, 0) && Date.now() < deadline) await sleep(50);
      signalGroup(pid, "SIGKILL");
    }
    await within(exit, 1000);
    // Let `onclose` fire with the stderr tail settled.
    await within(this.settled, 500);
    this.fireClose();
    child.stdout?.destroy();
    child.stderr?.destroy();
    child.stdin?.destroy();
  }

  /** Split `chunk` into lines on newline bytes and handle each; close past the size cap. */
  private read(chunk: Buffer): void {
    if (this.closed) return;
    const max = this.options.maxMessageBytes;
    let start = 0;
    for (let newline = chunk.indexOf(0x0a); newline >= 0; newline = chunk.indexOf(0x0a, start)) {
      const piece = chunk.subarray(start, newline);
      start = newline + 1;
      const size = this.pendingBytes + piece.length;
      const line = this.pending.length === 0 ? piece : Buffer.concat([...this.pending, piece], size);
      this.pending = [];
      this.pendingBytes = 0;
      if (size > max) return this.overflow(max);
      this.handleLine(line);
      if (this.closed) return;
    }
    if (start < chunk.length) {
      this.pending.push(chunk.subarray(start));
      this.pendingBytes += chunk.length - start;
      if (this.pendingBytes > max) this.overflow(max);
    }
  }

  /** Report a line longer than `max` bytes and close. */
  private overflow(max: number): void {
    this.pending = [];
    this.pendingBytes = 0;
    this.report(new Error(`MCP server sent a message over ${max} bytes`));
    void this.close();
  }

  /** Decode one line and hand it to `onmessage`. */
  private handleLine(bytes: Buffer): void {
    let text = bytes.toString("utf8");
    if (text.endsWith("\r")) text = text.slice(0, -1);
    if (text === "") return;
    let message: unknown;
    try {
      message = parseJson(text, "plain");
    } catch (error) {
      // The strict reader refuses duplicate keys and lone surrogates, which the SDK would read.
      try {
        message = JSON.parse(text);
      } catch {
        this.report(error);
        return;
      }
    }
    try {
      this.onmessage?.(hostInts(message));
    } catch (error) {
      // The SDK `JSON.stringify`s a late response for an unknown id, which throws on a `bigint`.
      this.report(error);
    }
  }

  /** Hand `error` to `onerror`; a throwing handler must not crash the host. */
  private report(error: unknown): void {
    try {
      this.onerror?.(error instanceof Error ? error : new Error(String(error)));
    } catch {
      // Nothing left to tell.
    }
  }

  /** Fire `onclose` once the process exited or stdout ended, after stderr drained or 200 ms. */
  private end(): void {
    if (this.ended) return;
    this.ended = true;
    if (this.stderrEnded) return this.fireClose();
    const timer = setTimeout(() => this.fireClose(), 200);
    this.child?.stderr?.once("close", () => {
      clearTimeout(timer);
      this.fireClose();
    });
  }

  /** Call `onclose`, once. */
  private fireClose(): void {
    if (this.closeFired) return;
    this.closeFired = true;
    this.settle();
    try {
      this.onclose?.();
    } catch (error) {
      this.report(error);
    }
  }
}

// The timers below keep the process alive on purpose: a host whose last statement is
// `await transport.close()` must not exit before the SIGKILL.

/** Resolve after `ms`. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Resolve when `promise` does or after `ms`, whichever comes first; no timer outlives it. */
async function within(promise: Promise<void>, ms: number): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([promise, new Promise<void>((resolve) => (timer = setTimeout(resolve, ms)))]);
  } finally {
    clearTimeout(timer);
  }
}

/** Send `signal` to process group `pid`; return whether the group still existed. Never throws. */
function signalGroup(pid: number, signal: NodeJS.Signals | 0): boolean {
  try {
    process.kill(-pid, signal);
    return true;
  } catch {
    // ESRCH: the group is gone; EPERM: the id now names another user's group. Any other error
    // counts as gone too, so `close()` never rejects.
    return false;
  }
}

/**
 * Turn the `bigint`s of a decoded message into what a host reads: a safe integer becomes a
 * `number`; a bigger one stays a `bigint` only inside a response's `result.structuredContent`,
 * where it is a tool's value, and elsewhere (ids, schema bounds) becomes a `number`. Mutates
 * `message` in place and returns it.
 */
function hostInts(message: unknown): unknown {
  const result = isRecord(message) && "id" in message && isRecord(message.result) ? message.result : undefined;
  const fix = (value: unknown, keep: boolean): unknown => {
    if (typeof value !== "bigint") return value;
    const n = Number(value);
    return keep && !Number.isSafeInteger(n) ? value : n;
  };
  if (typeof message !== "object" || message === null) return fix(message, false);
  const stack: [Record<string, unknown> | unknown[], boolean][] = [[message as Record<string, unknown>, false]];
  while (stack.length > 0) {
    const [container, keep] = stack.pop()!;
    for (const key of Object.keys(container)) {
      const value = (container as Record<string, unknown>)[key];
      if (typeof value === "bigint") (container as Record<string, unknown>)[key] = fix(value, keep);
      else if (typeof value === "object" && value !== null) {
        stack.push([value as Record<string, unknown>, keep || (container === result && key === "structuredContent")]);
      }
    }
  }
  return message;
}
