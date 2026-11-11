/** MCP call failures map to yian's error kinds, and stops and `close()` end waiting calls. */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeAll, describe, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, install } from "../../src/defaults.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { runFlow } from "../../src/run.ts";
import { isPermanentToolError, isReportedToolError, isTransientToolError } from "../../src/tools/errors.ts";
import {
  McpCallCancelled,
  McpConnectError,
  McpInvalidRequestError,
  McpRequestError,
  type McpServer,
  McpServerClosed,
  mcpServer,
  type McpServerOptions,
  McpTimeoutError,
  McpToolError,
  McpTransportError,
} from "../../src/tools/mcp.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import {
  type HttpServer,
  type HttpServerOptions,
  REDIRECT_LOCATION,
  startHttpServer,
} from "../fixtures/mcp/fake_server.ts";
import {
  childAlive,
  hostEnv,
  killMarkers,
  pids,
  stdioOptions,
  strayProcesses,
  trackServers,
  waitFor,
} from "../helpers/mcp_helpers.ts";

// Process and HTTP tests get 15 s; the `< N s` assertions are the real check.
const SLOW = 15_000;
const dir = mkdtempSync(join(tmpdir(), "yian-mcp-errors-"));
const markers: string[] = [];
const tracker = trackServers();

/** An unopened fake server with `flags` and a fresh marker file, closed in `afterEach`. */
function fake(flags: string[] = [], options: Omit<McpServerOptions, "command" | "args"> = {}): McpServer {
  const file = join(dir, `marker-${markers.length + 1}`);
  markers.push(file);
  return tracker.track(mcpServer({ ...stdioOptions(...flags, "--marker", file), ...options }));
}

/** The marker file of the last server `fake` built, which lists the pid of each of its starts. */
function lastMarker(): string {
  return markers.at(-1)!;
}

/** A `ToolSet` of `server`'s tools. */
function tools(server: McpServer): ToolSet {
  return new ToolSet({ fake: [...server.tools] });
}

/** The error `promise` rejects with; fails the test when it resolves. */
async function rejection(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

/** How `promise` settled and the seconds it took. */
async function timed(promise: Promise<unknown>): Promise<{ result?: unknown; error?: Error; ended: number }> {
  const start = performance.now();
  try {
    const result = await promise;
    return { result, ended: (performance.now() - start) / 1000 };
  } catch (error) {
    return { error: error as Error, ended: (performance.now() - start) / 1000 };
  }
}

/** Resolve after `seconds`. */
function sleep(seconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}

afterEach(async () => {
  await tracker.closeAll();
  install(BUILTIN);
});

afterAll(() => {
  killMarkers(markers);
  rmSync(dir, { recursive: true, force: true });
  expect(strayProcesses()).toEqual([]);
});

describe("errors from a running server", () => {
  let server: McpServer;

  beforeAll(async () => {
    // Not tracked: it serves every test here and closes in `afterAll`.
    const file = join(dir, "marker-shared");
    markers.push(file);
    server = await mcpServer(stdioOptions("--marker", file)).open();
  }, SLOW);

  afterAll(async () => {
    await server?.close();
  }, SLOW);

  test("an error result is a reported but not permanent tool error", async () => {
    const error = await rejection(tools(server).invoke("fail", { message: "no luck" }));
    expect(error).toBeInstanceOf(McpToolError);
    expect(error.message).toContain("no luck");
    expect(isReportedToolError(error)).toBe(true);
    expect(isPermanentToolError(error)).toBe(false);
  });

  test("an invalid params answer is reported and permanent", async () => {
    const error = await rejection(tools(server).invoke("bad_request", {}));
    expect(error).toBeInstanceOf(McpInvalidRequestError);
    expect(error).toBeInstanceOf(McpRequestError);
    expect((error as McpInvalidRequestError).code).toBe(-32602);
    expect(isReportedToolError(error)).toBe(true);
    expect(isPermanentToolError(error)).toBe(true);
    expect(error.message).toBe(`MCP server '${server.label}' tool 'bad_request': bad params (code -32602)`);
  });

  test("a busy answer is a request error and the server still works", async () => {
    const error = await rejection(tools(server).invoke("busy", {}));
    expect(error).toBeInstanceOf(McpRequestError);
    expect(error.message).toContain("busy");
    expect((error as McpRequestError).code).toBe(-32000);
    expect(error).not.toBeInstanceOf(McpServerClosed);
    expect(isPermanentToolError(error)).toBe(false);
    expect(server.isClosed).toBe(false);
    expect(await tools(server).invoke("echo", { text: "still here" })).toBe("still here");
  });

  test("an internal error answer is a request error", async () => {
    const error = await rejection(tools(server).invoke("crash", {}));
    expect(error).toBeInstanceOf(McpRequestError);
    expect(error.message).toContain("internal");
    expect((error as McpRequestError).code).toBe(-32603);
    expect(error).not.toBeInstanceOf(McpTransportError);
    expect(isPermanentToolError(error)).toBe(false);
  });

  test(
    "a run stop cancels a waiting call and the server still works",
    async () => {
      const stop = new AbortController();
      const bound = tools(server).forRun("r", { signal: stop.signal });
      const outcome = timed(bound.invoke("slow", { seconds: 30 }));
      await sleep(0.3);
      stop.abort();
      const { error, ended } = await outcome;
      expect(error).toBeInstanceOf(McpCallCancelled);
      expect(ended).toBeLessThan(0.3 + 2);
      expect(isPermanentToolError(error)).toBe(true);
      expect(error!.message).toBe("MCP call to 'slow' cancelled by a run stop");
      expect(await tools(server).invoke("echo", { text: "after stop" })).toBe("after stop");
    },
    SLOW,
  );

  test(
    "a stop cancels only the call of its own run",
    async () => {
      const firstStop = new AbortController();
      const secondStop = new AbortController();
      const first = tools(server).forRun("r1", { signal: firstStop.signal });
      const second = tools(server).forRun("r2", { signal: secondStop.signal });
      const firstOutcome = timed(first.invoke("slow", { seconds: 30 }));
      const secondOutcome = timed(second.invoke("slow", { seconds: 1.5 }));
      await sleep(0.3);
      firstStop.abort();
      expect((await firstOutcome).error).toBeInstanceOf(McpCallCancelled);
      const { result, error } = await secondOutcome;
      expect(error).toBeUndefined();
      expect(result).toBe("done");
    },
    SLOW,
  );

  test("a call with the stop already set is cancelled before it is sent", async () => {
    const before = ((await tools(server).invoke("calls", {})) as { result: number }).result;
    const stop = new AbortController();
    stop.abort();
    const error = await rejection(tools(server).forRun("r", { signal: stop.signal }).invoke("echo", { text: "x" }));
    expect(error).toBeInstanceOf(McpCallCancelled);
    expect(((await tools(server).invoke("calls", {})) as { result: number }).result).toBe(before);
  });
});

describe("errors that end a call", () => {
  test(
    "a call past its timeout is a transient timeout error",
    async () => {
      const server = await fake([], { timeout: 0.5, label: "slowpoke" }).open();
      const set = tools(server);
      const { error, ended } = await timed(set.invoke("slow", { seconds: 5 }));
      expect(ended).toBeLessThan(3);
      expect(error).toBeInstanceOf(McpTimeoutError);
      expect(isReportedToolError(error)).toBe(true);
      expect(isTransientToolError(error)).toBe(true);
      expect(error!.message).toBe("MCP server 'slowpoke' tool 'slow' timed out after 0.5 s");
      expect(await set.invoke("echo", { text: "after" })).toBe("after");
    },
    SLOW,
  );

  test(
    "a server that exits fails the call in flight, and the next call starts it again",
    async () => {
      const server = await fake([], { label: "quitter" }).open();
      const set = tools(server);
      const error = await rejection(set.invoke("exit_now", {}));
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(isPermanentToolError(error)).toBe(true);
      expect(isReportedToolError(error)).toBe(false);
      expect(error.message).toBe("MCP server 'quitter' exited (code 3) during tool 'exit_now'");
      expect(server.isClosed).toBe(true);
      expect(server.isOpen).toBe(true);
      expect(await set.invoke("echo", { text: "x" })).toBe("x");
      expect(server.isClosed).toBe(false);
      expect(server.isOpen).toBe(true);
      // Two starts: the call in flight wasn't sent again, or it would have ended the second.
      const [first, second] = pids(lastMarker());
      expect(pids(lastMarker())).toHaveLength(2);
      expect(childAlive(first!)).toBe(false);
      expect(String(await set.invoke("pid", {}))).toBe(String(second));
    },
    SLOW,
  );

  test(
    "a server that exits during a call names its exit code and its last stderr lines",
    async () => {
      const server = await fake([], { label: "loud" }).open();
      const error = await rejection(tools(server).invoke("exit_loudly", {}));
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(error.message).toBe(
        "MCP server 'loud' exited (code 3) during tool 'exit_loudly'\n" +
          "server stderr:\nFATAL: database file /data/x.db is locked",
      );
    },
    SLOW,
  );

  test(
    "calls that find the server gone share one restart",
    async () => {
      const server = await fake([], { label: "shared" }).open();
      const set = tools(server);
      await rejection(set.invoke("exit_now", {}));
      const texts = ["a", "b", "c", "d"];
      expect(await Promise.all(texts.map((text) => set.invoke("echo", { text })))).toEqual(texts);
      expect(pids(lastMarker())).toHaveLength(2);
    },
    SLOW,
  );

  test(
    "a failed restart fails the calls that waited on it, and later calls at once with the exit code only",
    async () => {
      const server = await fake(["--restart", "fail"], { label: "once" }).open();
      const set = tools(server);
      await rejection(set.invoke("exit_now", {}));
      const waited = await rejection(set.invoke("echo", { text: "x" }));
      expect(waited).toBeInstanceOf(McpServerClosed);
      expect(waited.message).toStartWith("MCP server 'once' exited (code 3); reconnecting failed: it didn't start: ");
      expect(waited.message).toEndWith("\nserver stderr:\nFATAL: can't start again");
      // No second attempt, and the restart's stderr isn't copied into every later error.
      const { error: later, ended } = await timed(set.invoke("echo", { text: "x" }));
      expect(later).toBeInstanceOf(McpServerClosed);
      expect(later!.message).toBe("MCP server 'once' exited (code 3); reconnecting failed");
      expect(ended).toBeLessThan(0.5);
      expect(pids(lastMarker())).toHaveLength(2);
      expect(server.isClosed).toBe(true);
    },
    SLOW,
  );

  test(
    "a close during a restart ends the new server and fails the waiting call as closed",
    async () => {
      const server = await fake(["--restart", "hang"], { label: "hung" }).open();
      const set = tools(server);
      await rejection(set.invoke("exit_now", {}));
      const waiting = rejection(set.invoke("echo", { text: "x" }));
      expect(await waitFor(() => pids(lastMarker()).length === 2, 5000)).toBe(true);
      await server.close();
      const error = await waiting;
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(error.message).toBe("MCP server 'hung' is closed");
      expect(server.isOpen).toBe(false);
      expect(childAlive(pids(lastMarker())[1]!)).toBe(false);
    },
    SLOW,
  );

  test(
    "a stop ends a call waiting on a restart",
    async () => {
      const server = await fake(["--restart", "hang"], { label: "hung" }).open();
      await rejection(tools(server).invoke("exit_now", {}));
      const stop = new AbortController();
      const waiting = timed(tools(server).forRun("r", { signal: stop.signal }).invoke("echo", { text: "x" }));
      expect(await waitFor(() => pids(lastMarker()).length === 2, 5000)).toBe(true);
      stop.abort();
      const { error, ended } = await waiting;
      expect(error).toBeInstanceOf(McpCallCancelled);
      expect(ended).toBeLessThan(5);
    },
    SLOW,
  );

  test(
    "a call sent after the server exited but before yian saw it close starts it again",
    async () => {
      // The answer comes 50 ms after the exit, and a `sh` holds stdout 2 s more, so the next call
      // goes out before the transport closes, which it does 200 ms after the exit.
      const server = await fake([], { label: "parting" }).open();
      const set = tools(server);
      expect(await set.invoke("answer_then_exit", { hold: 2 })).toBe("bye");
      expect(await set.invoke("echo", { text: "x" })).toBe("x");
      expect(pids(lastMarker())).toHaveLength(2);
    },
    SLOW,
  );

  test(
    "a server killed during a call names the signal",
    async () => {
      const server = await fake([], { label: "killed" }).open();
      const error = await rejection(tools(server).invoke("kill_self", {}));
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(error.message).toBe("MCP server 'killed' was killed by SIGKILL during tool 'kill_self'");
    },
    SLOW,
  );

  test(
    "every call in flight when the server exits names the tool it was running",
    async () => {
      const server = await fake([], { label: "busy" }).open();
      const set = tools(server);
      const waiting = rejection(set.invoke("slow", { seconds: 30 }));
      await sleep(0.3);
      const exited = await rejection(set.invoke("exit_now", {}));
      expect(exited.message).toBe("MCP server 'busy' exited (code 3) during tool 'exit_now'");
      expect((await waiting).message).toBe("MCP server 'busy' exited (code 3) during tool 'slow'");
    },
    SLOW,
  );

  test(
    "a server whose answer reaches stdout after it exits still answers the call",
    async () => {
      // The exit comes before the answer reaches stdout, as it may when the runtime reports an
      // exit before reading the pipe's last bytes.
      const server = await fake([], { label: "parting" }).open();
      expect(await tools(server).invoke("answer_then_exit", {})).toBe("bye");
    },
    SLOW,
  );

  test(
    "a close ends a waiting call",
    async () => {
      const server = await fake([], { label: "closing" }).open();
      const set = tools(server);
      const outcome = timed(set.invoke("slow", { seconds: 30 }));
      await sleep(0.3);
      const closing = server.close();
      const { error, ended } = await outcome;
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(error!.message).toBe("MCP server 'closing' is closed");
      expect(ended).toBeLessThan(0.3 + 3);
      await closing;
      const later = await rejection(set.invoke("echo", { text: "x" }));
      expect(later).toBeInstanceOf(McpServerClosed);
      expect(later.message).toBe("MCP server 'closing' is closed");
    },
    SLOW,
  );

  test(
    "a late failure from an earlier connection leaves the reopened server alive",
    async () => {
      const server = await fake(["--only", "slow,echo"]).open();
      const stale = tools(server).invoke("slow", { seconds: 30 });
      await sleep(0.3);
      const closing = server.close();
      // The stale call fails while the new connection opens, after `close()` detached the old one.
      const reopening = server.open();
      // Still awaited below; this keeps a failing `stale` check from leaving it unhandled.
      reopening.catch(() => {});
      expect(await rejection(stale)).toBeInstanceOf(McpServerClosed);
      await reopening;
      await closing;
      expect(server.isOpen).toBe(true);
      expect(server.isClosed).toBe(false);
      expect(await tools(server).invoke("echo", { text: "alive" })).toBe("alive");
    },
    SLOW,
  );

  test(
    "a message over the cap closes the server",
    async () => {
      install({ ...BUILTIN, maxMcpMessageBytes: 1024 });
      const server = await fake(["--only", "echo", "--flood", "2048"]).open();
      const error = await rejection(tools(server).invoke("echo", { text: "x" }));
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(error.message).toMatch(/ is closed: it sent a message over 1024 bytes \(YIAN_MAX_MCP_MESSAGE_BYTES\)$/);
      expect(server.isClosed).toBe(true);
      // yian closed it, so the next call doesn't start it again.
      const later = await rejection(tools(server).invoke("echo", { text: "x" }));
      expect(later.message).toBe(error.message);
      expect(pids(lastMarker())).toHaveLength(1);
    },
    SLOW,
  );

  test(
    "a tool kept from an earlier connection fails as closed after a failed open",
    async () => {
      const server = fake(["--only", "echo"], { label: "again" });
      await server.open();
      const kept = tools(server);
      await server.close();
      const reason = new Error("stop");
      expect(await rejection(server.open({ signal: AbortSignal.abort(reason) }))).toBe(reason);
      expect(server.isOpen).toBe(false);
      const error = await rejection(kept.invoke("echo", { text: "x" }));
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(error.message).toBe("MCP server 'again' is closed");
    },
    SLOW,
  );
});

describe("scrubbing the server's secrets", () => {
  const secret = "s3cret-value-123";

  test(
    "an env value of 8 or more characters is scrubbed from a call's error",
    async () => {
      const server = await fake(["--only", "echo_secret"], { env: { SECRET_TOKEN: secret } }).open();
      const error = await rejection(tools(server).invoke("echo_secret", {}));
      expect(error).toBeInstanceOf(McpToolError);
      expect(error.message).toBe("secret: [REDACTED]");
      expect(error.cause).toBeUndefined();
      expect(error.stack ?? "").not.toContain(secret);
    },
    SLOW,
  );

  test(
    "an env value is scrubbed from the stderr of a server that exits during a call",
    async () => {
      const server = await fake([], { env: { SECRET_TOKEN: secret }, label: "leaky" }).open();
      const error = await rejection(tools(server).invoke("exit_loudly", {}));
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(error.message).toEndWith("\nserver stderr:\nFATAL: database file /data/x.db is locked (token [REDACTED])");
      expect(error.stack ?? "").not.toContain(secret);
    },
    SLOW,
  );

  test(
    "a shorter env value is left in the text",
    async () => {
      // As `DEBUG=1`: short values garble messages when scrubbed and are rarely secrets.
      const server = await fake(["--only", "echo_secret"], { env: { SECRET_TOKEN: "abc" } }).open();
      expect((await rejection(tools(server).invoke("echo_secret", {}))).message).toBe("secret: abc");
    },
    SLOW,
  );

  test(
    "an env value is scrubbed from an open error",
    async () => {
      const server = fake(["--stderr", `leaked ${secret}`], { env: { SECRET_TOKEN: secret }, label: "leaky" });
      const error = await rejection(server.open());
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toStartWith("MCP server 'leaky' didn't start: ");
      expect(error.message).toContain("leaked [REDACTED]");
      expect(error.message).not.toContain(secret);
      expect(error.stack ?? "").not.toContain(secret);
    },
    SLOW,
  );

  test(
    "a header's value and its bare token are scrubbed from a call's error",
    async () => {
      const http = await serveHttp();
      const server = await httpServer({ url: http.url, headers: { "x-secret": "Bearer tok-1234567" } }).open();
      const error = await rejection(tools(server).invoke("echo_secret", {}));
      expect(error).toBeInstanceOf(McpToolError);
      expect(error.message).toBe("secret: \nx-secret: [REDACTED]\ntoken: [REDACTED]");
      expect(error.stack ?? "").not.toContain("tok-1234567");
    },
    SLOW,
  );

  test(
    "a shorter header value is left in the text",
    async () => {
      const http = await serveHttp();
      const headers = { "x-test": "v", "x-secret": "Bearer tok-1234567" };
      const server = await httpServer({ url: http.url, headers }).open();
      const set = tools(server);
      const failed = await rejection(set.invoke("fail", { message: "a very vivid failure" }));
      expect(failed.message).toContain("a very vivid failure");
      expect(failed.message).not.toContain("[REDACTED]");
      expect((await rejection(set.invoke("echo_secret", {}))).message).toBe(
        "secret: \nx-secret: [REDACTED]\ntoken: [REDACTED]",
      );
    },
    SLOW,
  );
});

/** The fake server over HTTP, stopped in `afterEach`. */
async function serveHttp(options: HttpServerOptions = {}): Promise<HttpServer> {
  const http = await startHttpServer(options);
  tracker.track({ close: () => http.stop() });
  return http;
}

/** An unopened server built from `options` that connects in `mode`, closed in `afterEach`. */
function httpServer(options: McpServerOptions, mode: "auto" | "legacy" = "auto"): McpServer {
  const server = tracker.track(mcpServer(options));
  (server as unknown as { mode: string }).mode = mode;
  return server;
}

describe("errors over HTTP", () => {
  for (const mode of ["auto", "legacy"] as const) {
    test(`a redirect mid-session is an invalid request that hides the location (${mode})`, async () => {
      const http = await serveHttp({ redirectAfterInit: true });
      const server = await httpServer({ url: http.url }, mode).open();
      const error = await rejection(tools(server).invoke("echo", { text: "x" }));
      expect(error).toBeInstanceOf(McpInvalidRequestError);
      expect((error as McpInvalidRequestError).code).toBe(-32600);
      expect(error.message).toBe("MCP server 'localhost' redirected the request; yian doesn't follow redirects");
      expect(isPermanentToolError(error)).toBe(true);
      expect(error.message).not.toContain(new URL(REDIRECT_LOCATION).host);
      // Nothing carries the Location: the `RedirectError` isn't kept as the cause.
      expect(error.cause).toBeUndefined();
    }, SLOW);
  }

  for (const mode of ["auto", "legacy"] as const) {
    test(`a JSON answer cut mid-body fails only that call, as a tool error (${mode})`, async () => {
      const http = await serveHttp({ cutCalls: "json" });
      const server = await httpServer({ url: http.url }, mode).open();
      const error = await rejection(tools(server).invoke("echo", { text: "x" }));
      expect(error).toBeInstanceOf(McpToolError);
      expect(error.message).toMatch(/^MCP server 'localhost' tool 'echo' failed \(\w+\)$/);
      expect(isReportedToolError(error)).toBe(true);
      expect(isPermanentToolError(error)).toBe(false);
      expect(server.isClosed).toBe(false);
    }, SLOW);

    test(`an SSE answer cut mid-stream waits out the call timeout (${mode})`, async () => {
      const http = await serveHttp({ cutCalls: "sse" });
      const server = await httpServer({ url: http.url, timeout: 1 }, mode).open();
      const { error, ended } = await timed(tools(server).invoke("echo", { text: "x" }));
      expect(error).toBeInstanceOf(McpTimeoutError);
      expect(error!.message).toBe("MCP server 'localhost' tool 'echo' timed out after 1 s");
      expect(ended).toBeGreaterThan(0.9);
      expect(ended).toBeLessThan(3);
      expect(server.isClosed).toBe(false);
    }, SLOW);
  }

  // The fake answers a modern client's calls with JSON bodies and a legacy one's with SSE streams.
  for (const mode of ["auto", "legacy"] as const) {
    test(`a reply over the message cap closes the server and names the cap (${mode})`, async () => {
      install({ ...BUILTIN, maxMcpMessageBytes: 4096 });
      const http = await serveHttp({ only: ["echo"] });
      const server = await httpServer({ url: http.url }, mode).open();
      const set = tools(server);
      expect(await set.invoke("echo", { text: "small" })).toBe("small");
      const error = await rejection(set.invoke("echo", { text: "x".repeat(100_000) }));
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(error.message).toBe(
        "MCP server 'localhost' is closed: it sent a message over 4096 bytes (YIAN_MAX_MCP_MESSAGE_BYTES)",
      );
      expect(server.isClosed).toBe(true);
    }, SLOW);
  }

  test("a tool listing over the message cap fails the open and names the cap", async () => {
    install({ ...BUILTIN, maxMcpMessageBytes: 64 });
    const http = await serveHttp({ only: ["echo"] });
    const error = await rejection(httpServer({ url: http.url }).open());
    expect(error).toBeInstanceOf(McpConnectError);
    expect(error.message).toBe(
      "MCP server 'localhost' didn't start: it sent a message over 64 bytes (YIAN_MAX_MCP_MESSAGE_BYTES)",
    );
  }, SLOW);

  test("a stopped http server is a transport error and a restart recovers", async () => {
    const http = await serveHttp();
    const server = await httpServer({ url: `${http.url}?token=SECRET`, label: "remote" }).open();
    const set = tools(server);
    expect(await set.invoke("echo", { text: "up" })).toBe("up");
    const port = http.port;
    await http.stop();
    const error = await rejection(set.invoke("echo", { text: "down" }));
    expect(error).toBeInstanceOf(McpTransportError);
    expect(isReportedToolError(error)).toBe(true);
    expect(isTransientToolError(error)).toBe(true);
    expect(error.message).toBe("MCP server 'remote' transport error (connection failed)");
    expect(error.message).not.toContain("SECRET");
    expect(server.isOpen).toBe(true);
    expect(server.isClosed).toBe(false);
    await serveHttp({ port });
    expect(await set.invoke("echo", { text: "back" })).toBe("back");
  }, SLOW);

  test(
    "a call a restarted session http server refuses tries one new session",
    async () => {
      const http = await serveHttp({ stateful: true });
      const server = await httpServer({ url: http.url, label: "restarted" }, "legacy").open();
      const set = tools(server);
      expect(await set.invoke("echo", { text: "up" })).toBe("up");
      await http.stop();
      await serveHttp({ port: http.port, stateful: true });
      // Another client initializes the restarted server, which then answers the old session 404.
      // The fake keeps one session, that client's, so it refuses the new one too.
      await httpServer({ url: http.url }, "legacy").open();
      const error = await rejection(set.invoke("echo", { text: "lost session" }));
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(error.message).toBe(
        "MCP server 'restarted' is closed; reconnecting failed: it didn't start: SdkHttpError (HTTP 400)",
      );
      expect(server.isClosed).toBe(true);
    },
    SLOW,
  );

  test(
    "a restarted session server no client initialized yet gets a new session too",
    async () => {
      const http = await serveHttp({ stateful: true });
      const server = await httpServer({ url: http.url, label: "restarted" }, "legacy").open();
      const set = tools(server);
      expect(await set.invoke("echo", { text: "up" })).toBe("up");
      await http.stop();
      await serveHttp({ port: http.port, stateful: true });
      // The new server answers 400 "Server not initialized" until a client initializes it.
      expect(await set.invoke("echo", { text: "lost session" })).toBe("lost session");
      expect(server.isClosed).toBe(false);
    },
    SLOW,
  );

  test("closing a session server sends a DELETE", async () => {
    const http = await serveHttp({ stateful: true });
    const server = await httpServer({ url: http.url }, "legacy").open();
    expect(await tools(server).invoke("echo", { text: "x" })).toBe("x");
    expect(http.recordedMethods).not.toContain("DELETE");
    await server.close();
    expect(http.recordedMethods).toContain("DELETE");
  }, SLOW);
});

describe("runs", () => {
  const RETRY_FLOW = `
id: retry
name: retry
nodes:
  call:
    kind: tool
    tool_id: TOOL
    retries: 2
    input: INPUT
output: \${call.output}
`;

  for (const [tool, args, attempts] of [
    ["fail", "{message: nope}", 3],
    ["bad_request", "{}", 1],
  ] as const) {
    test(
      `tool node retries follow the error class (${tool})`,
      async () => {
        const server = await fake().open();
        const calls = async () => ((await tools(server).invoke("calls", {})) as { result: number }).result;
        const before = await calls();
        const flow = await loadFlow(RETRY_FLOW.replace("TOOL", tool).replace("INPUT", args));
        const result = await runFlow(flow, {}, {
          tools: { fake: [...server.tools] },
          env: hostEnv(dir, { YIAN_TOOL_RETRY_DELAY_SECONDS: "0" }),
        });
        expect(result.status).toBe("failed");
        expect((await calls()) - before).toBe(attempts);
      },
      SLOW,
    );
  }

  test(
    "an agent model reads an mcp error result and the node succeeds",
    async () => {
      const call = new ToolCallPart({ id: "c0", name: "fake_fail", arguments: { message: "no luck" } });
      const model = new ScriptedModel([
        new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" }),
        new Response({
          message: new Message({ role: "assistant", parts: [new TextPart("gave up")] }),
          finishReason: "stop",
        }),
      ]);
      const flow = await loadFlow(`
id: mcp_agent
name: mcp_agent
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [fake]
output: \${worker.output}
`);
      const server = await fake().open();
      const result = await runFlow(flow, {}, {
        tools: { fake: [...server.tools] },
        env: hostEnv(dir),
        models: { env: hostEnv(dir), createModel: () => model },
      });
      expect(result.failure).toBeNull();
      expect(result.output).toBe("gave up");
      const [message] = model.requests[1]!.messages.flatMap((each) =>
        each.role === "tool" ? each.parts.filter((part) => part instanceof ToolResultPart) : [],
      );
      expect(message!.isError).toBe(true);
      expect(message!.content).toContain("no luck");
    },
    SLOW,
  );
});
