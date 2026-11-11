/**
 * MCP servers whose tools a flow calls as host tools.
 *
 * `mcpServer` describes a Model Context Protocol server: a command started as a subprocess that
 * speaks over stdio, or a streamable HTTP endpoint. `McpServer.open` starts the server's client,
 * lists the server's tools once and exposes each as a `FunctionTool`; `close` stops the client and
 * the subprocess. A call returns the server's structured content, or its content rendered as text.
 * A failed call throws an `McpError` whose tool error kinds tell an AGENT's tool loop whether its
 * model reads the error and a TOOL node whether `retries:` apply; a stop of the calling run or
 * `close()` ends a waiting call.
 *
 * Knows about: the MCP SDK client (`@modelcontextprotocol/client`, imported when a server opens),
 * `defaults`, `platform` (the version, and `safeFetch` for every HTTP request), `tools/context`,
 * `tools/errors`, `tools/function_tools`, `tools/types` (the output schema check), `llm/body`
 * (exact JSON writes), `llm/errors` and `llm/auth` (scrubbing the server's secrets from errors),
 * typesys and diagnostics.
 */

import { type ChildProcess, spawn } from "node:child_process";
import type { Writable } from "node:stream";

import type * as Sdk from "@modelcontextprotocol/client";

import { current } from "../defaults.ts";
import { reprText } from "../diagnostics.ts";
import { scrub } from "../llm/auth.ts";
import { encodeBody } from "../llm/body.ts";
import { scrubError } from "../llm/errors.ts";
import { RedirectError, safeFetch, VERSION } from "../platform.ts";
import { type JsonObject, parseJson } from "../typesys/json.ts";
import {
  buildValueAs,
  fromHost,
  isRecord,
  jsTypeName,
  renderValuePath,
  type Type,
  TypeCheckError,
  writeAsText,
} from "../typesys/values.ts";
import type { ToolContext } from "./context.ts";
import { TOOL_ERROR, ToolCheckError, type ToolErrorKind } from "./errors.ts";
import { FunctionTool } from "./function_tools.ts";
import { inputTypeFromJsonSchema } from "./types.ts";

/**
 * The brands of `McpError`, `AlreadyOpenError` and `McpServer`. yian checks them rather than
 * `instanceof`, so an error or server from another copy of yian (an npm install beside a workspace
 * one) is still known.
 */
const MCP_ERROR = Symbol.for("yian.mcp_error");
const ALREADY_OPEN = Symbol.for("yian.mcp_already_open");
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

/** The MCP SDK, which opening an MCP server needs, isn't installed; the import error is the `cause`. */
export class McpSdkMissingError extends McpError {
  override name = "McpSdkMissingError";
}

/**
 * An MCP server returned an error result for a tool call, or the call failed locally.
 *
 * For an error result the message is the result's content rendered as text, as a successful
 * call's text result is: text blocks, then one-line markers for media. For any other failure
 * while the server is up (for example structured content that isn't a yian value or fails the
 * tool's output schema, thrown as the subclass `McpOutputError`) it names only the server, the
 * tool and the error type. An AGENT's model reads it as the call's result.
 */
export class McpToolError extends McpError {
  override name = "McpToolError";
  readonly [TOOL_ERROR]: readonly ToolErrorKind[] = REPORTED;
}

/**
 * An MCP server's tool result failed yian's check: its structured content isn't a yian value, or
 * the tool lists an `outputSchema` and the result had no structured content or content that
 * doesn't fit the schema.
 *
 * It is permanent as well as reported, since yian ran the check and another attempt would call the
 * tool again.
 */
export class McpOutputError extends McpToolError {
  override name = "McpOutputError";
  override readonly [TOOL_ERROR]: readonly ToolErrorKind[] = REPORTED_PERMANENT;
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
 * The MCP server is gone: yian closed it, its process, connection or session ended, or it sent a
 * message over `YIAN_MAX_MCP_MESSAGE_BYTES`, which the message then names.
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

Object.defineProperty(AlreadyOpenError.prototype, ALREADY_OPEN, { value: true });

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

/** Return whether `value` is an `AlreadyOpenError` from any copy of yian. */
export function isAlreadyOpenError(value: unknown): value is AlreadyOpenError {
  return hasBrand(value, ALREADY_OPEN);
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
  // A `null` option is absent.
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
    // An empty args array or string counts as not given.
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

/** The longest delay `setTimeout` takes; a longer one fires at once. */
const MAX_TIMER_MS = 2 ** 31 - 1;

/** JSON-RPC codes for a request the server can't parse, route or accept. */
const INVALID_REQUEST_CODES: ReadonlySet<number> = new Set([-32700, -32600, -32601, -32602]);

/** The JSON-RPC code of a request the server refuses as invalid; a refused redirect gets it too. */
const INVALID_REQUEST = -32600;

/** How long closing an HTTP server waits for the server to end its session. */
const TERMINATE_SESSION_MS = 5000;

/** The largest integer JSON over HTTP carries exactly: the SDK writes it with `JSON.stringify`. */
const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);

/**
 * The errors `safeFetch` rejected an MCP HTTP request with because the request failed (no
 * connection, a reset), so a call that fails with one is a transport error.
 */
const transportFailures = new WeakSet<object>();

/**
 * Remember `error`, what `safeFetch` rejected the request `init` with, as a transport failure
 * unless it is a refused redirect or the request was aborted; then rethrow it.
 */
function remember(error: unknown, init: RequestInit | undefined): never {
  if (!(error instanceof RedirectError) && !init?.signal?.aborted && typeof error === "object" && error !== null) {
    transportFailures.add(error);
  }
  throw error;
}

const CR = 0x0d;
const LF = 0x0a;

/**
 * Return `response` with a body that fails once one message passes `max` bytes: one event of a
 * `text/event-stream` body (its bytes other than line ends, up to the blank line that ends it),
 * else the whole body. On the failure `onOverflow` runs and the body errors with a message that
 * names only the cap, never the bytes read.
 */
export function capResponse(response: Response, max: number, onOverflow: () => void): Response {
  // A 204 or 205 answer can't be rebuilt with a body.
  if (response.body === null || response.status === 204 || response.status === 205) return response;
  const type = response.headers.get("content-type")?.split(";")[0]!.trim().toLowerCase();
  const sse = type === "text/event-stream";
  let size = 0;
  // Whether the current line has no bytes yet, so a line end there is a blank line.
  let blank = true;
  // Whether the last byte was a CR, so an LF right after it ends no second line.
  let afterCr = false;
  const fail = (controller: TransformStreamDefaultController<Uint8Array>) => {
    onOverflow();
    controller.error(new Error(`MCP server sent a message over ${max} bytes`));
  };
  const capped = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      if (!sse) {
        size += chunk.length;
        if (size > max) return fail(controller);
      } else {
        for (let i = 0; i < chunk.length; i++) {
          const byte = chunk[i]!;
          if (byte === LF && afterCr) {
            afterCr = false;
            continue;
          }
          afterCr = byte === CR;
          if (byte === CR || byte === LF) {
            if (blank) size = 0;
            blank = true;
          } else {
            blank = false;
            if (++size > max) return fail(controller);
          }
        }
      }
      controller.enqueue(chunk);
    },
  });
  const { status, statusText, headers } = response;
  return new Response(response.body.pipeThrough(capped), { status, statusText, headers });
}

/** One connection of a server: what one `open()` started, and what `close()` ends. */
interface Generation {
  /** Aborted when the open fails, times out or is closed; ends a pending connect or listing. */
  readonly stop: AbortController;
  /** The SDK module, once loaded; its error classes classify a failed call. */
  sdk: typeof Sdk | undefined;
  client: Sdk.Client | undefined;
  transport: StdioTransport | Sdk.StreamableHTTPClientTransport | undefined;
  /** `"host"`, or the token of the run that opened the server, once its tools are listed. */
  owner: string | undefined;
  tools: readonly FunctionTool[] | undefined;
  /**
   * The type of each tool that listed an `outputSchema`, by name, for the check of its results;
   * `null` when the schema is too deep or unsupported, which skips the check.
   */
  outputTypes: ReadonlyMap<string, Type | null>;
  /** The longest wait for one call in seconds, read when the server opened. */
  callTimeout: number;
  /** Whether `close()` ran on this connection. */
  closed: boolean;
  /** Whether the server is known gone: its transport closed. */
  gone: boolean;
  /** The cap in bytes a message from the server passed, which closed the connection. */
  overflow: number | undefined;
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
  /**
   * Whether `close()` ran, or the last `open()` failed, since the last `open()` began, so a call
   * fails as closed, not as never opened.
   */
  private closeRequested = false;
  /** The texts scrubbed from every error `open()` and a call throw. */
  private readonly secrets: readonly string[];
  /** How an HTTP server negotiates the protocol era; tests set `"legacy"`. Stdio is always legacy. */
  private mode: "auto" | "legacy" = "auto";

  /** Keep a spec `mcpServer` checked; build a server with `mcpServer`, not this constructor. */
  constructor(spec: McpServerSpec) {
    this.spec = spec;
    this.secrets = secretsOf(spec);
    if (spec.label !== undefined) this.label = spec.label;
    // The text after the last `/`; a command ending in `/` keeps the whole command.
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
   * @throws {McpSdkMissingError} The MCP SDK (`@modelcontextprotocol/client`) is not installed.
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
    this.closeRequested = true;
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
    this.closeRequested = false;
    const generation: Generation = {
      stop: new AbortController(),
      sdk: undefined,
      client: undefined,
      transport: undefined,
      owner: undefined,
      tools: undefined,
      outputTypes: new Map(),
      callTimeout: 0,
      closed: false,
      gone: false,
      overflow: undefined,
      calls: new Set(),
      closing: undefined,
    };
    this.generation = generation;
    try {
      await this.connect(generation, owner, signal);
    } catch (error) {
      await this.shutdown(generation);
      // As a close: a tool kept from an earlier connection then fails as closed.
      if (this.generation === generation) {
        this.generation = undefined;
        this.closeRequested = true;
      }
      throw this.scrubbed(error);
    }
    return this;
  }

  /** Scrub this server's secrets from `error`'s message, stack and causes, in place. */
  private scrubbed<T>(error: T): T {
    if (this.secrets.length === 0) return error;
    return scrubError(error, (text) => scrub(text, this.secrets));
  }

  /**
   * Load the SDK, start the server, connect and list its tools within one connect deadline, and
   * record `owner`; throw the error `open()` reports.
   */
  private async connect(generation: Generation, owner: string, signal: AbortSignal | undefined): Promise<void> {
    const label = this.label;
    const http = this.spec.url !== undefined;
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
    const timer = setTimeout(
      () => {
        timedOut = true;
        halt();
      },
      Math.min(connectTimeout * 1000, MAX_TIMER_MS),
    );
    signal?.addEventListener("abort", halt, { once: true });
    // Run after every `await`: a timeout, `close()` or abort that landed while nothing could be
    // closed still wins.
    const check = () => {
      if (stop.signal.aborted || generation.closed || signal?.aborted) throw new Error("stopped while starting");
    };
    const request = () => ({
      signal: stop.signal,
      timeout: Math.min(Math.max(1, deadline - performance.now()), MAX_TIMER_MS),
    });
    try {
      check();
      const { sdk, defaultEnv } = await loadSdk(!http);
      check();
      generation.sdk = sdk;
      // As the stdio transport does on a line past the cap: the call in flight then fails as
      // closed, naming the cap, rather than waiting out its timeout.
      const overflow = (max: number) => {
        if (generation.overflow !== undefined) return;
        generation.overflow = max;
        generation.gone = true;
        void generation.transport?.close();
      };
      const transport = (generation.transport = this.createTransport(
        sdk,
        defaultEnv,
        settings.maxMcpMessageBytes,
        overflow,
      ));
      const client = new sdk.Client(
        { name: "yian", version: VERSION },
        { capabilities: {}, versionNegotiation: http ? { mode: this.mode } : undefined, listMaxPages: MAX_PAGES },
      );
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
          call: (tool, args, ctx) => this.callTool(tool, args, ctx),
          ...request(),
        },
      );
      check();
      generation.tools = listed.tools;
      generation.outputTypes = listed.outputTypes;
      generation.owner = owner;
    } catch (error) {
      const transport = generation.transport;
      const tail = transport instanceof StdioTransport ? transport.stderrTail() : "";
      const late = `MCP server '${label}' didn't start within ${secondsText(connectTimeout)} s` + tail;
      if (timedOut) throw new McpConnectError(late);
      if (generation.closed) throw new McpConnectError(`MCP server '${label}' was closed while starting`);
      if (generation.overflow !== undefined) {
        throw new McpConnectError(`MCP server '${label}' didn't start: it ${overflowText(generation.overflow)}`);
      }
      if (signal?.aborted) throw signal.reason;
      if (!started || error instanceof McpConnectError) throw error;
      const root = http ? leaf(error, generation.sdk!) : error;
      // The SDK's timeout for a request is what was left of the deadline, so it may fire just
      // before yian's timer.
      if ((root as { code?: unknown } | null)?.code === "REQUEST_TIMEOUT") throw new McpConnectError(late);
      if (http) {
        // Only the type and status: the URL and the error text may hold a token.
        const status = (root as { status?: unknown } | null)?.status;
        const shown = typeof status === "number" ? ` (HTTP ${status})` : "";
        throw new McpConnectError(`MCP server '${label}' didn't start: ${errorName(root)}${shown}`, { cause: root });
      }
      throw new McpConnectError(`MCP server '${label}' didn't start: ${describeError(error)}` + tail, { cause: error });
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", halt);
    }
  }

  /**
   * Return the transport for this server: yian's stdio transport, or the SDK's streamable HTTP
   * transport sending every request through `safeFetch`, which follows no redirect, and reading
   * every reply through `capResponse`. Either calls `onOverflow` with `maxMessageBytes` when a
   * message passes it.
   */
  private createTransport(
    sdk: typeof Sdk,
    defaultEnv: Readonly<Record<string, string>>,
    maxMessageBytes: number,
    onOverflow: (max: number) => void,
  ): StdioTransport | Sdk.StreamableHTTPClientTransport {
    const { command, args, env, cwd, url, headers } = this.spec;
    if (command !== undefined) {
      return new StdioTransport({ command, args, env, cwd, maxMessageBytes, defaultEnv, onOverflow });
    }
    return new sdk.StreamableHTTPClientTransport(new URL(url!), {
      fetch: async (input, init) => {
        const response = await safeFetch(input, init).catch((error: unknown) => remember(error, init));
        return capResponse(response, maxMessageBytes, () => onOverflow(maxMessageBytes));
      },
      requestInit: { headers },
    });
  }

  /** End `generation`: its calls, its client and its transport. Never rejects; runs once. */
  private shutdown(generation: Generation): Promise<void> {
    return (generation.closing ??= (async () => {
      generation.stop.abort();
      for (const call of generation.calls) call.abort();
      generation.calls.clear();
      const transport = generation.transport;
      if (transport !== undefined && !(transport instanceof StdioTransport) && transport.sessionId !== undefined) {
        // Before the transport closes, which would abort the DELETE.
        await within(transport.terminateSession().catch(() => {}), TERMINATE_SESSION_MS);
      }
      const transportClosed = transport?.close();
      try {
        await generation.client?.close();
      } catch {
        // The transport's own close below still ends the server.
      }
      await transportClosed;
    })());
  }

  /**
   * Call the server's tool `tool.name` with the checked `args` and return its result as a yian
   * value: the structured content when the server sent it, else the content rendered as text.
   * Every error it throws has this server's secrets scrubbed.
   *
   * @throws {ToolCheckError} An argument is `Infinity`, which JSON can't carry, or, for an HTTP
   *   server, an integer beyond ±(2^53-1); nothing is sent.
   * @throws {Error} The server was never opened.
   * @throws {McpToolError} The server returned an error result, or the call failed in a way no
   *   other error here covers; `McpOutputError` for a result that breaks the tool's `outputSchema`.
   * @throws {McpRequestError} The server answered with a JSON-RPC error; `McpInvalidRequestError`
   *   for a malformed or unknown request, or an HTTP server's redirect.
   * @throws {McpTimeoutError} The call outlasted the call timeout.
   * @throws {McpTransportError} The HTTP request failed, or the server answered it with an HTTP
   *   error status.
   * @throws {McpServerClosed} The server is closed or gone, or an HTTP server forgot the session.
   * @throws {McpCallCancelled} `ctx.signal` aborted before or while the call waited.
   */
  private async callTool(tool: ListedTool, args: Record<string, unknown>, ctx: ToolContext<unknown>): Promise<unknown> {
    try {
      return await this.sendCall(tool, args, ctx);
    } catch (error) {
      throw this.scrubbed(error);
    }
  }

  /** The body of `callTool`, before scrubbing. */
  private async sendCall(tool: ListedTool, args: Record<string, unknown>, ctx: ToolContext<unknown>): Promise<unknown> {
    const { name, inputSchema } = tool;
    const http = this.spec.url !== undefined;
    const refused = unsendable(args, http);
    if (refused !== null) {
      throw new ToolCheckError(`tool call '${name}' arguments at ${renderValuePath(refused.path)}: ${refused.reason}`);
    }
    const sent = http ? (numbersForBigints(args) as Record<string, unknown>) : args;
    // The connection this call goes to; a later open replaces it.
    const generation = this.generation;
    if (this.closeRequested || generation?.gone) throw this.closedError();
    if (generation?.owner === undefined || generation.client === undefined) {
      throw new Error(`MCP server '${this.label}' is not open`);
    }
    if (ctx.signal.aborted) throw cancelledError(name);
    const call = new AbortController();
    // A listener removed when the call settles, not `AbortSignal.any`, which would keep a
    // long-lived run signal holding every call it ever saw.
    const onAbort = () => call.abort();
    ctx.signal.addEventListener("abort", onAbort, { once: true });
    generation.calls.add(call);
    // The SDK starts its timer before sending, and `maxTotalTimeout` bounds input-required rounds.
    const ms = Math.min(generation.callTimeout * 1000, MAX_TIMER_MS);
    let result: unknown;
    try {
      result = await generation.client.callTool(
        { name, arguments: sent, _meta: { "yian/call_id": ctx.callId, "yian/attempt": ctx.attempt } },
        // No `outputSchema`, so the SDK never validates the output itself; yian checks it below.
        {
          toolDefinition: { name, inputSchema: inputSchema as Sdk.Tool["inputSchema"] },
          signal: call.signal,
          timeout: ms,
          maxTotalTimeout: ms,
        },
      );
    } catch (error) {
      throw this.callError(generation, name, error, ctx.signal);
    } finally {
      ctx.signal.removeEventListener("abort", onAbort);
      generation.calls.delete(call);
    }
    // `undefined`: the tool listed no `outputSchema`; `null`: one yian can't read as a type.
    this.checkOutput(name, generation.outputTypes.get(name), result as CallResult);
    return resultValue(result as CallResult);
  }

  /**
   * Return the error a failed call to `name` on `generation` throws. yian's own state comes first,
   * since the SDK reports any abort as a request timeout.
   */
  private callError(generation: Generation, name: string, error: unknown, signal: AbortSignal): Error {
    const label = this.label;
    if (generation.closed || generation.gone) return this.closedError(generation);
    if (signal.aborted) return cancelledError(name);
    const sdk = generation.sdk!;
    const code = error instanceof sdk.SdkError ? error.code : undefined;
    if (code === sdk.SdkErrorCode.RequestTimeout) {
      return new McpTimeoutError(
        `MCP server '${label}' tool '${name}' timed out after ${secondsText(generation.callTimeout)} s`,
      );
    }
    if (code === sdk.SdkErrorCode.ConnectionClosed || code === sdk.SdkErrorCode.NotConnected) {
      return this.closedError(generation);
    }
    if (code === undefined && error instanceof Error && error.message === "Not connected") {
      // The SDK's own error for a request sent after the transport closed.
      return this.closedError(generation);
    }
    const failure = httpFailure(error, sdk);
    if (failure instanceof RedirectError) {
      return new McpInvalidRequestError(
        `MCP server '${label}' redirected the request; yian doesn't follow redirects`,
        INVALID_REQUEST,
      );
    }
    if (failure instanceof sdk.SdkHttpError) {
      const transport = generation.transport;
      const session = !(transport instanceof StdioTransport) && transport?.sessionId !== undefined;
      // The server forgot the session (a restart), so this client can't reach it again.
      if (
        (failure.status === 404 && session) ||
        (failure.status === 400 && failure.message.includes("Server not initialized"))
      ) {
        return this.closedError(generation);
      }
      return new McpTransportError(`MCP server '${label}' transport error (HTTP ${failure.status})`);
    }
    if (failure !== undefined) {
      return new McpTransportError(`MCP server '${label}' transport error (connection failed)`);
    }
    if (error instanceof sdk.ProtocolError) {
      const text = `MCP server '${label}' tool '${name}': ${error.message} (code ${error.code})`;
      return INVALID_REQUEST_CODES.has(error.code)
        ? new McpInvalidRequestError(text, error.code)
        : new McpRequestError(text, error.code);
    }
    return new McpToolError(`MCP server '${label}' tool '${name}' failed (${errorName(error)})`, { cause: error });
  }

  /**
   * Mark `generation` gone, when given, and return the `McpServerClosed` a call throws; it names
   * the message cap when a message past it closed the connection.
   */
  private closedError(generation?: Generation): McpServerClosed {
    if (generation !== undefined) generation.gone = true;
    const overflow = (generation ?? this.generation)?.overflow;
    const reason = overflow === undefined ? "" : `: it ${overflowText(overflow)}`;
    return new McpServerClosed(`MCP server '${this.label}' is closed${reason}`);
  }

  /**
   * Check a non-error result of the tool `name`, as the registry checks a declared output: its
   * structured content, when there is any, must be a yian value. A tool that listed an
   * `outputSchema`, read open as `type`, must also send structured content that fits the type; a
   * `null` type checks only that it is a yian value, and `undefined` means the tool listed none.
   *
   * @throws {McpOutputError} The structured content isn't a yian value, or for a tool with an
   *   `outputSchema` is missing or doesn't fit the type (the `TypeCheckError` is the `cause`).
   */
  private checkOutput(name: string, type: Type | null | undefined, result: CallResult): void {
    if (result.isError) return;
    const failed = `MCP server '${this.label}' tool '${name}' failed`;
    if (result.structuredContent == null) {
      if (type === undefined) return;
      throw new McpOutputError(`${failed} (no structured content)`);
    }
    try {
      const value = fromHost(result.structuredContent, `MCP server '${this.label}' tool '${name}' structured content`);
      if (type !== null && type !== undefined) buildValueAs(type, value);
    } catch (error) {
      if (error instanceof TypeCheckError) throw new McpOutputError(`${failed} (TypeCheckError)`, { cause: error });
      throw error;
    }
  }
}

/** The `McpCallCancelled` for a call to `name`. */
function cancelledError(name: string): McpCallCancelled {
  return new McpCallCancelled(`MCP call to '${name}' cancelled by a run stop`);
}

/** Why a server's connection closed when one of its messages passed the `max` bytes cap. */
function overflowText(max: number): string {
  return `sent a message over ${max} bytes (YIAN_MAX_MCP_MESSAGE_BYTES)`;
}

/** `seconds` in `%g` form: six significant digits, `0.5`, `300`, `1e-05`. */
function secondsText(seconds: number): string {
  const [mantissa, exponentText] = seconds.toExponential(5).split("e") as [string, string];
  const exponent = Number(exponentText);
  const trim = (digits: string) => (digits.includes(".") ? digits.replace(/\.?0+$/, "") : digits);
  if (exponent < -4 || exponent >= 6) {
    return `${trim(mantissa)}e${exponent < 0 ? "-" : "+"}${String(Math.abs(exponent)).padStart(2, "0")}`;
  }
  return trim(seconds.toFixed(5 - exponent));
}

/**
 * The texts scrubbed from a server's errors, each of 8 or more characters: the `headers` values
 * and the token after a value's scheme word (`Bearer abcdefgh` gives `abcdefgh` too, since a
 * server may echo the bare token), the `env` values, and the URL's query-parameter values.
 * Shorter ones, like `DEBUG=1` or `x-test: v`, would garble messages and are rarely secrets; the
 * default environment's values (`HOME`, `PATH`) never count.
 */
function secretsOf(spec: McpServerSpec): string[] {
  const secrets: string[] = [];
  for (const value of Object.values(spec.headers ?? {})) {
    secrets.push(value);
    const token = /^\S+\s+(\S.*)$/.exec(value)?.[1];
    if (token !== undefined) secrets.push(token);
  }
  secrets.push(...Object.values(spec.env ?? {}));
  if (spec.url !== undefined && URL.canParse(spec.url)) {
    secrets.push(...new URL(spec.url).searchParams.values());
  }
  return secrets.filter((value) => value.length >= 8);
}

/** An error's class name. */
function errorName(error: unknown): string {
  return error instanceof Error ? error.name : jsTypeName(error);
}

/**
 * Return the first value in `args` that JSON can't carry to the server, depth first and in order:
 * `Infinity` or `-Infinity`, and for an HTTP server an integer beyond ±(2^53-1); its path and why;
 * or `null` when there is none.
 */
function unsendable(args: unknown, http: boolean): { path: (string | number)[]; reason: string } | null {
  const stack: [unknown, (string | number)[]][] = [[args, []]];
  // A host may call `execute` directly with a cyclic record; each container is walked once.
  const seen = new Set<object>();
  while (stack.length > 0) {
    const [value, path] = stack.pop()!;
    if (value === Infinity || value === -Infinity) return { path, reason: "Infinity can't be sent to an MCP server" };
    if (http && typeof value === "bigint" && (value > MAX_SAFE || value < -MAX_SAFE)) {
      return { path, reason: "an integer beyond ±(2^53-1) can't be sent to an MCP HTTP server" };
    }
    if (typeof value === "object" && value !== null) {
      if (seen.has(value)) continue;
      seen.add(value);
    }
    // Pushed in reverse, so entries pop in order.
    if (Array.isArray(value)) {
      for (let i = value.length - 1; i >= 0; i--) stack.push([value[i], [...path, i]]);
    } else if (isRecord(value)) {
      const keys = Object.keys(value);
      for (let i = keys.length - 1; i >= 0; i--) stack.push([value[keys[i]!], [...path, keys[i]!]]);
    }
  }
  return null;
}

/**
 * Return `value` with each `bigint` as a `number`, for an HTTP server, whose SDK writes requests
 * with `JSON.stringify`; `unsendable` has refused the unsafe ones. Copies only the arrays and
 * records on the way to a `bigint`, so the caller's arguments stay as they were.
 */
function numbersForBigints(value: unknown, done = new Map<object, unknown>()): unknown {
  if (typeof value === "bigint") return Number(value);
  if (typeof value !== "object" || value === null) return value;
  // A container met again gives its result, so one held twice is converted both times. Until its
  // walk ends it maps to itself, so a cycle is left as is; `JSON.stringify` refuses it.
  if (done.has(value)) return done.get(value);
  done.set(value, value);
  let result: unknown = value;
  if (Array.isArray(value)) {
    let copy: unknown[] | undefined;
    value.forEach((item, i) => {
      const next = numbersForBigints(item, done);
      if (next !== item) (copy ??= [...value])[i] = next;
    });
    result = copy ?? value;
  } else if (isRecord(value)) {
    let copy: Record<string, unknown> | undefined;
    for (const [key, item] of Object.entries(value)) {
      const next = numbersForBigints(item, done);
      if (next !== item) (copy ??= { ...value })[key] = next;
    }
    result = copy ?? value;
  }
  done.set(value, result);
  return result;
}

Object.defineProperty(McpServer.prototype, MCP_SERVER, { value: true });

/** Return whether `value` is an `McpServer` from any copy of yian. */
export function isMcpServer(value: unknown): value is McpServer {
  return hasBrand(value, MCP_SERVER);
}

/**
 * Import the MCP SDK client module, and for a stdio server its default environment.
 *
 * @throws {McpSdkMissingError} The SDK is not installed.
 */
async function loadSdk(stdio: boolean): Promise<{ sdk: typeof Sdk; defaultEnv: Record<string, string> }> {
  try {
    const sdk = await import("@modelcontextprotocol/client");
    const defaultEnv = stdio ? (await import("@modelcontextprotocol/client/stdio")).getDefaultEnvironment() : {};
    return { sdk, defaultEnv };
  } catch (cause) {
    throw new McpSdkMissingError("MCP servers need the MCP SDK; install @modelcontextprotocol/client 2.3.1", { cause });
  }
}

/**
 * Return the error an HTTP connect failed with: under the `EraNegotiationFailed` wrappers that
 * the `auto` probe puts around it, so both modes report the same error.
 */
function leaf(error: unknown, sdk: typeof Sdk): unknown {
  let current = error;
  while (
    current instanceof sdk.SdkError &&
    current.code === sdk.SdkErrorCode.EraNegotiationFailed &&
    current.cause !== undefined
  ) {
    current = current.cause;
  }
  return current;
}

/**
 * Return `error` when it is what failed an HTTP call: a refused redirect, a failed request or an
 * HTTP error status; else `undefined`.
 */
function httpFailure(error: unknown, sdk: typeof Sdk): unknown {
  // The client rejects a call with the transport's error as is; only the connect probe wraps one.
  if (error instanceof RedirectError || error instanceof sdk.SdkHttpError) return error;
  if (typeof error === "object" && error !== null && transportFailures.has(error)) return error;
  return undefined;
}

/** `<Name>: <message>` for an error. */
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

/** A listed tool as a call needs it: its name and its input schema. */
export interface ListedTool {
  readonly name: string;
  readonly inputSchema: JsonObject;
}

/** The options of `listAll`. */
export interface ListOptions {
  /** The server's label, for errors. */
  label: string;
  /** The server's `concurrencySafe` option. */
  concurrencySafe: boolean | readonly string[];
  /** Runs one call of the listed `tool`; each tool's `execute` delegates to it. */
  call: (tool: ListedTool, args: Record<string, unknown>, ctx: ToolContext<unknown>) => Promise<unknown>;
  /** Aborts the listing. */
  signal?: AbortSignal;
  /** The listing's timeout in milliseconds. */
  timeout?: number;
}

/**
 * List the server's tools and wrap each in a `FunctionTool`. A server that declares no `tools`
 * capability lists none, and gets no `tools/list` request.
 *
 * @returns The tools, frozen and in listing order, and the type of each listed `outputSchema`
 *   read open (`null` when too deep or unsupported), by tool name.
 * @throws {McpConnectError} The server lists more than 100 pages, a tool twice, a tool whose input
 *   schema's type is not `"object"` or a tool `FunctionTool` refuses; or `concurrencySafe` names
 *   a tool it doesn't list.
 */
export async function listAll(
  lister: ToolLister,
  options: ListOptions,
): Promise<{ tools: readonly FunctionTool[]; outputTypes: Map<string, Type | null> }> {
  const { label, concurrencySafe, call, signal, timeout } = options;
  const tools: FunctionTool[] = [];
  const outputTypes = new Map<string, Type | null>();
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
      const listed: ListedTool = { name: name as string, inputSchema: inputSchema as JsonObject };
      try {
        tools.push(
          new FunctionTool({
            name: name as string,
            description: (description || title || name) as string,
            parametersJsonSchema: inputSchema as JsonObject,
            isConcurrencySafe: marked ? () => true : null,
            execute: (args, ctx) => call(listed, args, ctx),
          }),
        );
        if (isRecord(outputSchema)) outputTypes.set(name as string, inputTypeFromJsonSchema(outputSchema));
      } catch (error) {
        throw new McpConnectError(`MCP server '${label}' tool ${shown} is unusable: ${(error as Error).message}`, {
          cause: error,
        });
      }
      seen.add(name as string);
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
  return { tools: Object.freeze(tools), outputTypes };
}

/** What `resultValue` reads from a `tools/call` result. */
export interface CallResult {
  readonly content?: readonly unknown[];
  readonly structuredContent?: unknown;
  readonly isError?: boolean;
}

/**
 * Return a tool call's result as a yian value: the structured content when the server sent it,
 * else the content rendered as one string (see `renderContent`).
 *
 * @throws {McpToolError} The result is an error result; its message is the rendered content.
 */
export function resultValue(result: CallResult): unknown {
  if (result.isError) throw new McpToolError(renderContent(result.content ?? []));
  if (result.structuredContent != null) return result.structuredContent;
  return renderContent(result.content ?? []);
}

/**
 * Render MCP content blocks as text, joined by blank lines.
 *
 * Text blocks and embedded text resources give their text. Images, audio, embedded blob resources
 * and resource links give a one-line marker with the mime type, when known, and the decoded size;
 * their bytes are dropped. Empty content renders as `""`. A lone surrogate, as a server that cuts
 * text inside an emoji sends, becomes U+FFFD.
 */
function renderContent(content: readonly unknown[]): string {
  const parts: string[] = [];
  const head = (uri: unknown, mimeType: unknown) => [uri, mimeType].filter(Boolean).map(String).join(" ");
  for (const item of content) {
    const block = (isRecord(item) ? item : {}) as Record<string, unknown>;
    const kind = block.type;
    if (kind === "text") {
      parts.push(String(block.text));
    } else if (kind === "image" || kind === "audio") {
      parts.push(`[${kind} ${String(block.mimeType)}, ${size(String(block.data))}]`);
    } else if (kind === "resource") {
      const resource = (isRecord(block.resource) ? block.resource : {}) as Record<string, unknown>;
      if (typeof resource.text === "string") parts.push(resource.text);
      else parts.push(`[resource ${head(resource.uri, resource.mimeType)}, ${size(String(resource.blob))}]`);
    } else if (kind === "resource_link") {
      parts.push(`[resource ${head(block.uri, block.mimeType)}]`);
    } else {
      parts.push(`[${String(kind)}]`);
    }
  }
  // The rendering is already a lossy view for the model, so half an emoji is repaired, not refused.
  return parts.join("\n\n").toWellFormed();
}

/** The base64 alphabet. */
const BASE64_CHAR = /[A-Za-z0-9+/]/;

/** Return the size of base64 `data` once decoded: `<n> B`, `<x.y> KB` or `<x.y> MB`. */
function size(data: string): string {
  // Estimated from the encoded length when the server sent malformed base64.
  const bytes = decodedSize(data) ?? Math.floor((data.length * 3) / 4);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${oneDecimal(bytes, 1024)} KB`;
  return `${oneDecimal(bytes, 1024 * 1024)} MB`;
}

/**
 * Return how many bytes base64-decoding `data` gives, or `null` where decoding fails:
 * characters outside the alphabet are skipped, decoding stops where padding completes a group of
 * four, and a final partial group without padding is an error.
 */
function decodedSize(data: string): number | null {
  let group = 0;
  let pads = 0;
  let bytes = 0;
  for (const char of data) {
    if (char === "=") {
      if (group >= 2 && group + ++pads >= 4) return bytes;
      continue;
    }
    if (!BASE64_CHAR.test(char)) continue;
    pads = 0;
    if (group > 0) bytes += 1;
    group = (group + 1) % 4;
  }
  return group === 0 ? bytes : null;
}

/** `bytes / unit` with one decimal, rounded half to even. */
function oneDecimal(bytes: number, unit: number): string {
  // Exact: `bytes * 10` is a safe integer and `unit` a power of two.
  const scaled = (bytes * 10) / unit;
  let tenths = Math.floor(scaled);
  const rest = scaled - tenths;
  if (rest > 0.5 || (rest === 0.5 && tenths % 2 === 1)) tenths += 1;
  return `${Math.floor(tenths / 10)}.${tenths % 10}`;
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
  /** Called with `maxMessageBytes` when a line passes it, before the transport closes. */
  onOverflow?: (max: number) => void;
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
  private stdoutEnded = false;
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
      child.stdout!.once("end", () => {
        this.stdoutEnded = true;
        this.end();
      });
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
    this.options.onOverflow?.(max);
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

  /**
   * Fire `onclose` once the process exited or stdout ended, after both stdout and stderr ended or
   * 200 ms, so a grandchild holding a pipe open can't delay it longer.
   */
  private end(): void {
    if (this.ended) return;
    this.ended = true;
    // `exit` may come before the pipes end: the server's last answer must reach `onmessage` first.
    const timer = setTimeout(() => this.fireClose(), 200);
    const check = () => {
      if (!this.stdoutEnded || !this.stderrEnded) return;
      clearTimeout(timer);
      this.fireClose();
    };
    this.child?.stdout?.once("end", check);
    this.child?.stderr?.once("close", check);
    check();
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
