/**
 * MCP servers whose tools a flow calls as host tools.
 *
 * `mcpServer` describes a Model Context Protocol server: a command started as a subprocess that
 * speaks over stdio, or a streamable HTTP endpoint. `McpServer.open` starts the server's client,
 * lists the server's tools once and exposes each as a `FunctionTool`; `close` stops the client and
 * the subprocess. A failed call throws an `McpError` whose tool error kinds tell an AGENT's tool
 * loop whether its model reads the error and a TOOL node whether `retries:` apply.
 *
 * Knows about: `tools/errors`, `tools/function_tools`, typesys and diagnostics.
 */

import { reprText } from "../diagnostics.ts";
import { isRecord, jsTypeName } from "../typesys/values.ts";
import { TOOL_ERROR, type ToolErrorKind } from "./errors.ts";
import type { FunctionTool } from "./function_tools.ts";

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

/**
 * An MCP server whose tools a flow calls as host tools.
 *
 * Build one with `mcpServer`. While open, `tools` holds the tools the server listed when it
 * opened; a later change to the server's tool list is ignored. A server the host opened belongs to
 * the host, which closes it.
 */
export class McpServer {
  /** The name used in errors: the `label` option, else the command's basename, else the URL's host. */
  readonly label: string;
  /** The checked options, read when the server opens. */
  private readonly spec: McpServerSpec;

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
    return false;
  }

  /** Whether an open server has gone away, so its calls fail at once until it is reopened. */
  get isClosed(): boolean {
    return false;
  }

  /**
   * The server's tools, in the order it listed them.
   *
   * @throws {Error} The server is not open.
   */
  get tools(): readonly FunctionTool[] {
    throw new Error(`MCP server '${this.label}' is not open`);
  }

  /**
   * Start the server, wait until it lists its tools, and keep it running; the host must then
   * `close()` it. Not implemented yet: it lands with the MCP SDK client and its transports.
   *
   * @returns This server.
   */
  open(): Promise<this> {
    return Promise.reject(new Error("opening an MCP server is not ported yet (T6-2)"));
  }

  /** Stop the server and its client. Closing a server that is not open does nothing. */
  close(): Promise<void> {
    return Promise.resolve();
  }
}

Object.defineProperty(McpServer.prototype, MCP_SERVER, { value: true });

/** Return whether `value` is an `McpServer` from any copy of yian. */
export function isMcpServer(value: unknown): value is McpServer {
  return hasBrand(value, MCP_SERVER);
}
