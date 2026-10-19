/**
 * MCP servers whose tools a flow calls as host tools.
 *
 * `mcpServer` describes a Model Context Protocol server: a command started as a subprocess that
 * speaks over stdio, or a streamable HTTP endpoint. `McpServer.open` starts the server's client,
 * lists the server's tools once and exposes each as a `FunctionTool`; `close` stops the client and
 * the subprocess. A failed call throws an `McpError` whose tool error kinds tell an AGENT's tool
 * loop whether its model reads the error and a TOOL node whether `retries:` apply.
 *
 * Knows about: `tools/errors`, `tools/function_tools`, `llm/body` (exact JSON writes), `defaults`,
 * typesys and diagnostics.
 */

import { type ChildProcess, spawn } from "node:child_process";

import { reprText } from "../diagnostics.ts";
import { encodeBody } from "../llm/body.ts";
import { parseJson } from "../typesys/json.ts";
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
  onclose?: () => void;
  onerror?: (error: Error) => void;
  onmessage?: (message: unknown) => void;
  /** Always `null`: stderr is read into `stderrTail()`. The SDK reads it to see a stdio transport. */
  readonly stderr = null;
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
    if (this.child !== undefined || this.closed) return Promise.reject(new Error("StdioTransport already started"));
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
      if (flushed) {
        resolve();
        return;
      }
      const onDrain = () => {
        stdin.off("close", onClose);
        resolve();
      };
      const onClose = () => {
        stdin.off("drain", onDrain);
        reject(new Error("Not connected"));
      };
      stdin.once("drain", onDrain);
      stdin.once("close", onClose);
    });
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
    await Promise.race([exit, sleep(2000)]);
    // Even when the leader exited, what it started may still run in its group.
    if (signalGroup(pid, "SIGTERM")) {
      const deadline = Date.now() + 2000;
      while (signalGroup(pid, 0) && Date.now() < deadline) await sleep(50);
      signalGroup(pid, "SIGKILL");
    }
    await Promise.race([exit, sleep(1000)]);
    // Let `onclose` fire with the stderr tail settled.
    await Promise.race([this.settled, sleep(500)]);
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

/** Resolve after `ms`, without keeping the process alive. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms).unref());
}

/** Send `signal` to process group `pid`; return whether the group still existed. */
function signalGroup(pid: number, signal: NodeJS.Signals | 0): boolean {
  try {
    process.kill(-pid, signal);
    return true;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ESRCH" || code === "EPERM") return false;
    throw error;
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
