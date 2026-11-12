/**
 * A fake MCP server with one tool per behaviour the MCP client tests need.
 *
 * Import `buildServer` to serve it in-process, `startHttpServer` to serve it over streamable HTTP
 * in-process, or run this file as a script to serve it over stdio (`bun --no-install
 * tests/fixtures/mcp/fake_server.ts [flags]`, or built, `node dist/tests/fixtures/mcp/fake_server.js
 * [flags]`).
 */

import { spawn, spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo, Socket } from "node:net";
import { PassThrough, type Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import {
  type CallToolRequest,
  type CallToolResult,
  createMcpHandler,
  isLegacyRequest,
  McpServer,
  ProtocolError,
  Server,
  type RegisteredTool,
  type ServerContext,
  type StandardSchemaWithJSON,
  type Tool,
  type ToolCallback,
  WebStandardStreamableHTTPServerTransport,
} from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";

/** A 1x1 transparent PNG. */
export const PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

// Shared by every server in this process: `gate` and `peak` measure concurrency, `calls` counts
// served calls.
let running = 0;
let peak = 0;
let calls = 0;

// The ids put `Point` and `Item` under `$defs`, so `nested`'s input schema carries `$ref`s.
// Module-level, since zod's registry refuses a second schema with the same id.
const Point = z.object({ x: z.int(), y: z.int() }).meta({ id: "Point" });
const Item = z.object({ name: z.string(), point: Point }).meta({ id: "Item" });

const RESULT = z.object({ result: z.int() });

/**
 * The tools that answer with a JSON-RPC error, by name: `[code, message]`. `McpServer` turns
 * anything a tool throws into an error result, so `buildServer` throws these from around its
 * `tools/call` handler, so the client sees a JSON-RPC error.
 */
const RPC_ERRORS: Record<string, [number, string]> = {
  bad_request: [-32602, "bad params"],
  busy: [-32000, "busy"],
  crash: [-32603, "internal"],
};

/** Options for {@link buildServer}. */
export interface FakeServerOptions {
  /** The server name it reports. Defaults to `"fake"`. */
  name?: string;
  /** Tool names to keep. Unset keeps every tool. */
  only?: Iterable<string>;
  /** The file `slow_big` appends `answered` to when it answers. */
  marker?: string;
  /** When set, `echo` answers with a text of this many bytes. */
  flood?: number;
}

/** A text-only tool result. */
function text(value: string): CallToolResult {
  return { content: [{ type: "text", text: value }] };
}

/** A tool result carrying `value` as structured content, and as JSON text. */
function structured(value: Record<string, unknown>): CallToolResult {
  return { content: [{ type: "text", text: JSON.stringify(value) }], structuredContent: value };
}

/** Resolve after `seconds`. */
function sleep(seconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}

/** Append `line` to `file`; a file holding a single line holds it bare. */
function appendLine(file: string, line: string): void {
  appendFileSync(file, (existsSync(file) && statSync(file).size > 0 ? "\n" : "") + line);
}

/** The `big` result: an integer past 2^53, which `JSON.stringify` writes as `18446744073709552000`. */
function big(): CallToolResult {
  return structured({ n: 2 ** 64 });
}

/** Build the fake server, not yet serving. */
export function buildServer({ name = "fake", only, marker, flood }: FakeServerOptions = {}): McpServer {
  const srv = new McpServer({ name, version: "0" });
  const keep = only === undefined ? undefined : new Set(only);
  const tool = <I extends StandardSchemaWithJSON | undefined = undefined>(
    toolName: string,
    config: { description: string; inputSchema?: I; outputSchema?: StandardSchemaWithJSON },
    cb: ToolCallback<I>,
  ): void => {
    if (keep === undefined || keep.has(toolName)) srv.registerTool(toolName, config, cb);
  };
  let late: RegisteredTool | undefined;

  tool("echo", { description: "Return `text`.", inputSchema: z.object({ text: z.string() }) }, ({ text: value }) =>
    text(flood === undefined ? value : "x".repeat(flood)),
  );
  tool(
    "add",
    { description: "Return `a + b`.", inputSchema: z.object({ a: z.int(), b: z.int() }), outputSchema: RESULT },
    ({ a, b }) => structured({ result: a + b }),
  );
  tool(
    "fail",
    {
      description: "Fail with `message`, which the server reports as an error result.",
      inputSchema: z.object({ message: z.string() }),
    },
    ({ message }) => {
      throw new Error(message);
    },
  );
  for (const errorTool of Object.keys(RPC_ERRORS)) {
    // Never reached: the wrapper below answers these.
    tool(errorTool, { description: "Fail with a JSON-RPC error." }, () => text(""));
  }
  tool("picture", { description: "Return an image block, then a text caption." }, () => ({
    content: [
      { type: "image", data: PNG_BASE64, mimeType: "image/png" },
      { type: "text", text: "caption" },
    ],
  }));
  tool("resource", { description: "Return an embedded text resource and a resource link." }, () => ({
    content: [
      { type: "resource", resource: { uri: "text://note", text: "note body", mimeType: "text/plain" } },
      { type: "resource_link", name: "a.png", uri: "file:///a.png", mimeType: "image/png" },
    ],
  }));
  tool("whoami", { description: "Return the call id and attempt yian put in the request `_meta`." }, (ctx) => {
    const meta = ctx.mcpReq._meta ?? {};
    return text(`${meta["yian/call_id"]} ${meta["yian/attempt"]}`);
  });
  tool(
    "slow",
    { description: 'Sleep `seconds`, then return "done".', inputSchema: z.object({ seconds: z.number() }) },
    async ({ seconds }) => {
      await sleep(seconds);
      return text("done");
    },
  );
  tool("pid", { description: "Return the server's process id." }, () => text(String(process.pid)));
  tool("exit_now", { description: "Exit the server process mid-call." }, () => process.exit(3));
  tool(
    "exit_loudly",
    { description: "Write a line naming `SECRET_TOKEN`'s value to stderr, then exit with code 3 mid-call." },
    () => {
      const secret = process.env.SECRET_TOKEN;
      const line = `FATAL: database file /data/x.db is locked${secret === undefined ? "" : ` (token ${secret})`}\n`;
      // Exits once the line is written: stderr is asynchronous on a macOS pipe.
      process.stderr.write(line, () => process.exit(3));
      return new Promise<never>(() => {});
    },
  );
  tool("kill_self", { description: "Kill the server process with SIGKILL mid-call." }, () => {
    process.kill(process.pid, "SIGKILL");
    return text("");
  });
  tool(
    "answer_then_exit",
    {
      description:
        'Exit at once, leaving a `sh` that writes the answer "bye" 50 ms later and holds stdout `hold` seconds more.',
      inputSchema: z.object({ hold: z.number().optional() }),
    },
    ({ hold = 0 }, ctx) => {
      // The `sh` holds stdout but not stderr, so stderr ends at the exit and the answer comes later.
      const answer = JSON.stringify({ jsonrpc: "2.0", id: ctx.mcpReq.id, result: text("bye") });
      spawn("sh", ["-c", `sleep 0.05; printf "%s\\n" "$ANSWER"; sleep ${hold}`], {
        env: { ...process.env, ANSWER: answer },
        stdio: ["ignore", "inherit", "ignore"],
      });
      process.exit(0);
    },
  );
  tool(
    "gate",
    {
      description: "Wait `seconds`, tracking how many calls overlap.",
      inputSchema: z.object({ seconds: z.number() }),
    },
    async ({ seconds }) => {
      running += 1;
      peak = Math.max(peak, running);
      try {
        await sleep(seconds);
      } finally {
        running -= 1;
      }
      return text("done");
    },
  );
  tool("peak", { description: "Return the most `gate` calls that ran at once.", outputSchema: RESULT }, () =>
    structured({ result: peak }),
  );
  tool(
    "calls",
    { description: "Return how many tool calls, other than `calls`, the server has served.", outputSchema: RESULT },
    () => structured({ result: calls }),
  );
  tool("nested", { description: "Return `item` as JSON.", inputSchema: z.object({ item: Item }) }, ({ item }) =>
    text(JSON.stringify(item)),
  );
  tool("add_tool", { description: "Register the tool `late`, which tells the client the tool list changed." }, () => {
    // `registerTool` sends `notifications/tools/list_changed` itself on a connected server.
    late ??= srv.registerTool("late", { description: "A tool added while the server runs." }, () => text("late"));
    return text("added");
  });
  tool(
    "env",
    { description: 'Return the variable `name` the server sees, or "".', inputSchema: z.object({ name: z.string() }) },
    ({ name: variable }) => text(process.env[variable] ?? ""),
  );
  tool("big", { description: "Return an integer past 2^53 as structured content." }, big);
  tool(
    "slow_big",
    {
      description: "Sleep `seconds`, record the answer in the marker file, then return like `big`.",
      inputSchema: z.object({ seconds: z.number() }),
    },
    async ({ seconds }) => {
      await sleep(seconds);
      if (marker !== undefined) appendLine(marker, "answered");
      return big();
    },
  );
  tool(
    "echo_secret",
    { description: "Fail naming `SECRET_TOKEN`'s value and, over HTTP, the `x-secret` header and its last word." },
    (ctx) => {
      const header = ctx.http?.req?.headers.get("x-secret");
      const echoed = header == null ? "" : `\nx-secret: ${header}\ntoken: ${header.split(" ").pop()}`;
      return { content: [{ type: "text", text: `secret: ${process.env.SECRET_TOKEN ?? ""}${echoed}` }], isError: true };
    },
  );
  tool("headers", { description: 'Return the request\'s `x-test` HTTP header, or "" when there is none.' }, (ctx) =>
    text(ctx.http?.req?.headers.get("x-test") ?? ""),
  );
  tool(
    "region",
    {
      description: "Return `region`, which a modern HTTP client also sends as the `Mcp-Param-Region` header.",
      inputSchema: z.object({ region: z.string().meta({ "x-mcp-header": "Region" }) }),
    },
    ({ region }) => text(region),
  );

  // Count every call but `calls`, and answer the error tools.
  const inner = toolsCallHandler(srv);
  if (inner !== undefined) {
    srv.server.setRequestHandler("tools/call", (request, ctx) => {
      const toolName = request.params.name;
      if (toolName !== "calls") calls += 1;
      const rpcError = RPC_ERRORS[toolName];
      if (rpcError !== undefined && (keep === undefined || keep.has(toolName))) throw new ProtocolError(...rpcError);
      return inner(request, ctx);
    });
  }
  return srv;
}

/** The `tools/call` handler `McpServer` installed, or `undefined` before it registered a tool. */
function toolsCallHandler(
  srv: McpServer,
): ((request: CallToolRequest, ctx: ServerContext) => Promise<CallToolResult>) | undefined {
  // `_getRequestHandler` is protected; the SDK has no public way to wrap a registered handler.
  return (srv.server as unknown as { _getRequestHandler(method: string): never })._getRequestHandler("tools/call");
}

/** Options for {@link startHttpServer}. */
export interface HttpServerOptions {
  /** Tool names to keep. Unset keeps every tool. */
  only?: Iterable<string>;
  /** The port to listen on; 0, the default, picks a free one. */
  port?: number;
  /**
   * Serve 2025-era traffic from one session-keeping transport, so a restart forgets the session;
   * otherwise each legacy request is served statelessly.
   */
  stateful?: boolean;
  /** Answer every request after the first `tools/list` with a 307 to another local URL. */
  redirectAfterInit?: boolean;
  /**
   * Answer every `tools/call` with the headers and the start of a body, then drop the connection:
   * a JSON body (`"json"`) or an SSE stream (`"sse"`).
   */
  cutCalls?: "json" | "sse";
}

/** The fake server over streamable HTTP, from {@link startHttpServer}. */
export interface HttpServer {
  /** The MCP endpoint. */
  readonly url: string;
  readonly port: number;
  /** The headers of every request served so far, names in lower case. */
  readonly recordedHeaders: Record<string, string>[];
  /** The method of every request served so far. */
  readonly recordedMethods: string[];
  /** Stop serving and drop open connections; a second call does nothing more. */
  stop(): Promise<void>;
}

/** Where `redirectAfterInit` sends requests; its path and query must never reach an error. */
export const REDIRECT_LOCATION = "http://127.0.0.1:1/elsewhere?token=SECRET-LOCATION";

/**
 * Serve `handle` on 127.0.0.1 through `node:http`, each request turned into a web `Request` and
 * the `Response` written back as it streams. `close` stops listening and drops open connections.
 */
function listen(
  port: number,
  handle: (request: Request, req: IncomingMessage) => Promise<Response>,
): Promise<{ port: number; close: () => Promise<void> }> {
  let bound = port;
  const sockets = new Set<Socket>();
  const listener = createServer((req, res) => {
    void (async () => {
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      const headers = new Headers();
      for (let i = 0; i < req.rawHeaders.length; i += 2) headers.append(req.rawHeaders[i]!, req.rawHeaders[i + 1]!);
      const method = req.method ?? "GET";
      const body = method === "POST" || method === "DELETE" ? Buffer.concat(chunks) : undefined;
      const response = await handle(new Request(`http://127.0.0.1:${bound}${req.url}`, { method, headers, body }), req);
      res.writeHead(response.status, Object.fromEntries(response.headers));
      // An SSE stream's headers must reach the client before its first event.
      res.flushHeaders();
      if (response.body !== null) {
        const reader = response.body.getReader();
        res.once("close", () => void reader.cancel().catch(() => {}));
        for (let chunk = await reader.read(); !chunk.done; chunk = await reader.read()) res.write(chunk.value);
      }
      res.end();
    })().catch(() => res.destroy());
  });
  // Tracked by hand: Bun's `closeAllConnections` leaves a streaming response's connection open.
  listener.on("connection", (socket: Socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
  });
  const close = async () => {
    const closed = new Promise<void>((resolve) => listener.close(() => resolve()));
    for (const socket of sockets) socket.destroy();
    await closed;
    // A turn of the event loop lets an in-process client see its connections close: Node's
    // `fetch` would otherwise send the next request on one, which then fails with ECONNRESET.
    await new Promise((resolve) => setImmediate(resolve));
  };
  return new Promise((resolve, reject) => {
    listener.once("error", reject);
    listener.listen(port, "127.0.0.1", () => {
      bound = (listener.address() as AddressInfo).port;
      resolve({ port: bound, close });
    });
  });
}

/** The method of the JSON-RPC request `body`, or `undefined`. */
function rpcMethod(body: string): unknown {
  try {
    return JSON.parse(body).method;
  } catch {
    return undefined;
  }
}

/**
 * A 200 answer that sends its headers and the start of a `kind` body, then fails, so `listen`
 * drops the connection.
 */
function cutResponse(kind: "json" | "sse"): Response {
  const start = kind === "json" ? '{"jsonrpc":"2.0","id":' : 'event: message\ndata: {"jsonrpc":"2.0","id":';
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(start));
    },
    async pull(controller) {
      // Late enough that the headers and the first bytes reach the client before the drop.
      await sleep(0.1);
      controller.error(new Error("cut"));
    },
  });
  const type = kind === "json" ? "application/json" : "text/event-stream";
  return new Response(body, { status: 200, headers: { "content-type": type } });
}

/**
 * Serve the fake server over streamable HTTP on 127.0.0.1: the SDK's modern handler with its
 * stateless legacy fallback, or with `stateful` a session-keeping transport for 2025-era traffic.
 * Each start builds fresh state, so a server stopped and started on the same port forgets its
 * sessions.
 */
export async function startHttpServer({
  only,
  port = 0,
  stateful = false,
  redirectAfterInit = false,
  cutCalls,
}: HttpServerOptions = {}): Promise<HttpServer> {
  const keep = only === undefined ? undefined : [...only];
  const modern = createMcpHandler(() => buildServer({ only: keep }));
  let session: WebStandardStreamableHTTPServerTransport | undefined;
  if (stateful) {
    session = new WebStandardStreamableHTTPServerTransport({ sessionIdGenerator: () => randomUUID() });
    await buildServer({ only: keep }).connect(session);
  }
  const recordedHeaders: Record<string, string>[] = [];
  const recordedMethods: string[] = [];
  let redirecting = false;
  const { port: bound, close } = await listen(port, async (request, req) => {
    recordedHeaders.push(
      Object.fromEntries(Object.entries(req.headers).map(([name, value]) => [name, [value].flat().join(", ")])),
    );
    recordedMethods.push(request.method);
    if (redirecting) return new Response(null, { status: 307, headers: { location: REDIRECT_LOCATION } });
    const method = redirectAfterInit || cutCalls !== undefined ? rpcMethod(await request.clone().text()) : undefined;
    if (redirectAfterInit && method === "tools/list") redirecting = true;
    if (cutCalls !== undefined && method === "tools/call") return cutResponse(cutCalls);
    if (session !== undefined && (await isLegacyRequest(request))) return session.handleRequest(request);
    return modern.fetch(request);
  });
  let stopped: Promise<void> | undefined;
  return {
    url: `http://localhost:${bound}/mcp`,
    port: bound,
    recordedHeaders,
    recordedMethods,
    stop: () =>
      (stopped ??= (async () => {
        await close();
        await modern.close();
        await session?.close();
      })()),
  };
}

/** A server that answers every request with a 307 to `target`, from {@link startRedirectServer}. */
export interface RedirectServer {
  readonly url: string;
  /** How many requests it answered. */
  readonly hits: number;
  /** Stop serving and drop open connections; a second call does nothing more. */
  stop(): Promise<void>;
}

/** Serve a 307 to `target` for every request on 127.0.0.1. */
export async function startRedirectServer(target: string): Promise<RedirectServer> {
  let hits = 0;
  let stopped: Promise<void> | undefined;
  const { port, close } = await listen(0, async () => {
    hits += 1;
    return new Response(null, { status: 307, headers: { location: target } });
  });
  return {
    url: `http://localhost:${port}/mcp`,
    get hits() {
      return hits;
    },
    stop: () => (stopped ??= close()),
  };
}

/** A low-level server listing one tool per page, following a cursor; tool `pageK` answers "K". */
function pagedServer(pages: number): Server {
  // `--pages 0` declares no `tools` capability, so the server has no tool handlers at all.
  const server = new Server({ name: "paged", version: "0" }, { capabilities: pages > 0 ? { tools: {} } : {} });
  if (pages === 0) return server;
  server.setRequestHandler("tools/list", (request) => {
    const cursor = request.params?.cursor;
    const index = cursor ? Number(cursor) : 1;
    const tool: Tool = { name: `page${index}`, inputSchema: { type: "object", properties: {} } };
    return { tools: [tool], ...(index < pages ? { nextCursor: String(index + 1) } : {}) };
  });
  server.setRequestHandler("tools/call", (request) => text(request.params.name.replace(/^page/, "")));
  return server;
}

/** The `outputSchema` both of `badOutputServer`'s tools declare. */
const OUTPUT_SCHEMA = {
  type: "object",
  properties: { n: { type: "integer" } },
  required: ["n"],
  additionalProperties: false,
} as const;

/**
 * A low-level server whose tools break their `outputSchema`: `bad_output` returns structured
 * content the schema rejects, `no_structured` returns only text. `add_tool` only tells the client
 * the tool list changed.
 */
function badOutputServer(): Server {
  const server = new Server(
    { name: "bad-output", version: "0" },
    { capabilities: { tools: { listChanged: true } } },
  );
  const inputSchema = { type: "object", properties: {} } as const;
  server.setRequestHandler("tools/list", () => ({
    tools: [
      {
        name: "bad_output",
        description: "Return structured content that violates the advertised schema.",
        inputSchema,
        outputSchema: OUTPUT_SCHEMA,
      },
      {
        name: "no_structured",
        description: "Return only text despite the advertised schema.",
        inputSchema,
        outputSchema: OUTPUT_SCHEMA,
      },
      { name: "add_tool", description: "Tell the client the tool list changed.", inputSchema },
    ],
  }));
  server.setRequestHandler("tools/call", async (request) => {
    if (request.params.name === "add_tool") {
      await server.sendToolListChanged();
      return text("added");
    }
    return request.params.name === "bad_output"
      ? { content: [{ type: "text", text: '{"n": "x"}' }], structuredContent: { n: "x" } }
      : text('{"n": 1}');
  });
  return server;
}

/** Text cut inside an emoji, as `"Hi 🙂".slice(0, 4)` cuts it: it ends in a lone surrogate. */
const CUT = "Hi \ud83d";

/**
 * A low-level server whose tools, none with an `outputSchema`, answer with `CUT`: in a text block
 * (`cut_text`), an embedded text resource (`cut_resource`), structured content (`cut_structured`)
 * and an error result (`cut_error`).
 */
function loneSurrogateServer(): Server {
  const server = new Server({ name: "cut", version: "0" }, { capabilities: { tools: {} } });
  const inputSchema = { type: "object", properties: {} } as const;
  const results: Record<string, CallToolResult> = {
    cut_text: text(CUT),
    cut_resource: { content: [{ type: "resource", resource: { uri: "text://cut", text: CUT } }] },
    cut_structured: { content: [{ type: "text", text: "{}" }], structuredContent: { k: CUT } },
    cut_error: { content: [{ type: "text", text: CUT }], isError: true },
  };
  server.setRequestHandler("tools/list", () => ({
    tools: Object.keys(results).map((name) => ({ name, description: `Answer with cut text: ${name}.`, inputSchema })),
  }));
  server.setRequestHandler("tools/call", (request) => results[request.params.name]!);
  return server;
}

/** A low-level server listing the tool `twin` twice. */
function twinsServer(): Server {
  const server = new Server({ name: "twins", version: "0" }, { capabilities: { tools: {} } });
  const twin: Tool = { name: "twin", inputSchema: { type: "object", properties: {} } };
  server.setRequestHandler("tools/list", () => ({ tools: [twin, twin] }));
  server.setRequestHandler("tools/call", () => text("twin"));
  return server;
}

/** Whether `line` is a JSON-RPC `initialize` request. */
function isInitialize(line: Buffer): boolean {
  try {
    return JSON.parse(line.toString("utf8")).method === "initialize";
  } catch {
    return false;
  }
}

/**
 * Read stdin's first line and exit at once unless it is an `initialize` request; otherwise
 * resolve with a stream that replays it, then the rest of stdin.
 */
function stdinAfterInitialize(): Promise<Readable> {
  return new Promise((resolve) => {
    let head = Buffer.alloc(0);
    const onData = (chunk: Buffer) => {
      head = Buffer.concat([head, chunk]);
      const end = head.indexOf(0x0a);
      if (end < 0) return;
      process.stdin.off("data", onData);
      if (!isInitialize(head.subarray(0, end))) process.exit(1);
      const replay = new PassThrough();
      replay.write(head);
      process.stdin.pipe(replay);
      resolve(replay);
    };
    process.stdin.on("data", onData);
  });
}

/** Run the fake server as a script, serving stdio. */
async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      name: { type: "string", default: "fake" },
      only: { type: "string" },
      marker: { type: "string" },
      hang: { type: "boolean" },
      stderr: { type: "string" },
      "exit-before-init": { type: "boolean" },
      "ignore-cancel": { type: "boolean" },
      pages: { type: "string" },
      "bad-output": { type: "boolean" },
      "lone-surrogate": { type: "boolean" },
      twins: { type: "boolean" },
      grandchild: { type: "string" },
      "grandchild-ignores-term": { type: "string" },
      stubborn: { type: "boolean" },
      flood: { type: "string" },
      restart: { type: "string" },
    },
  });
  // A process with nothing else to do keeps running on this (setInterval's longest delay).
  const keepAlive = () => setInterval(() => {}, 2 ** 31 - 1);

  // For `--restart fail` or `hang`: a start after the first, whose marker lists an earlier pid.
  const again =
    values.marker !== undefined && existsSync(values.marker) && /^\d+$/m.test(readFileSync(values.marker, "utf8"));
  if (values.marker !== undefined) appendLine(values.marker, String(process.pid));
  if (again && values.restart === "fail") {
    process.stderr.write("FATAL: can't start again\n");
    process.exitCode = 1;
    return;
  }
  if (again && values.restart === "hang") {
    keepAlive();
    return;
  }
  if (values.stderr !== undefined) {
    process.stderr.write(values.stderr + "\n");
    // Not `process.exit(1)`, which could drop the write: stderr is asynchronous on a macOS pipe.
    process.exitCode = 1;
    return;
  }
  if (values.hang) {
    keepAlive();
    return;
  }
  if (values.grandchild !== undefined) {
    // Not detached, so it stays in the server's process group.
    const child = spawn("sleep", ["300"], { stdio: "ignore" });
    appendLine(values.grandchild, String(child.pid));
  }
  if (values["grandchild-ignores-term"] !== undefined) {
    // An ignored signal stays ignored across `exec`, so `sleep` ignores SIGTERM too.
    const child = spawn("sh", ["-c", "trap '' TERM; exec sleep 300"], { stdio: "ignore" });
    // The pid is written once `sh` has exec'd `sleep`, so a reader knows SIGTERM is ignored.
    for (let command = commandOf(child.pid!); !command.startsWith("sleep"); command = commandOf(child.pid!)) {
      // `sh` died before it started `sleep`: exit rather than loop forever, detached and unlisted.
      if (command === "") process.exit(1);
      await sleep(0.01);
    }
    appendLine(values["grandchild-ignores-term"], String(child.pid));
    process.exit(0);
  }
  if (values.stubborn) {
    process.on("SIGTERM", () => {});
    keepAlive();
  }

  const server =
    values.pages !== undefined
      ? pagedServer(Number(values.pages))
      : values["bad-output"]
        ? badOutputServer()
        : values["lone-surrogate"]
          ? loneSurrogateServer()
          : values.twins
            ? twinsServer()
            : buildServer({
                name: values.name,
                only: values.only?.split(","),
                marker: values.marker,
                flood: values.flood === undefined ? undefined : Number(values.flood),
              });
  const transport = new StdioServerTransport(values["exit-before-init"] ? await stdinAfterInitialize() : undefined);
  await server.connect(transport);
  if (values["ignore-cancel"]) {
    // Dropped before the SDK sees it, which would otherwise drop the late response itself.
    const onmessage = transport.onmessage;
    transport.onmessage = (message) => {
      if (!("method" in message && message.method === "notifications/cancelled")) onmessage?.(message);
    };
  }
}

/** The command line of process `pid`, or `""` when it is gone. */
function commandOf(pid: number): string {
  return spawnSync("ps", ["-o", "command=", "-p", String(pid)], { encoding: "utf8" }).stdout.trim();
}

/** Return whether this file is the process's entry script. */
function isEntry(): boolean {
  // Node 22 has no `import.meta.main`; Node realpaths the main module's URL but not `argv[1]`.
  const entry = process.argv[1];
  if (entry === undefined) return false;
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    // `argv[1]` names no file.
    return false;
  }
}

if (isEntry()) {
  await main();
}
