/**
 * MCP call failures map to yian's error kinds, and stops and `close()` end waiting calls, a port
 * of `tests/engine/test_mcp_errors.py` over stdio:
 * `test_an_error_result_is_a_reported_but_not_permanent_tool_error`,
 * `test_an_invalid_params_answer_is_reported_and_permanent`,
 * `test_a_busy_answer_is_a_request_error_and_the_server_still_works`,
 * `test_an_internal_error_answer_is_a_request_error`,
 * `test_a_call_past_its_timeout_is_a_transient_timeout_error`,
 * `test_a_server_that_exits_is_closed_and_later_calls_fail_at_once`,
 * `test_a_run_stop_cancels_a_waiting_call_and_the_server_still_works`,
 * `test_a_stop_cancels_only_the_call_of_its_own_run`, `test_close_from_another_thread_ends_a_waiting_call`
 * (a `close()` not awaited), `test_a_late_failure_from_an_earlier_connection_leaves_the_reopened_server_alive`
 * (a real close and re-open) and `test_a_call_with_the_stop_already_set_is_cancelled_before_it_is_sent`;
 * plus the message cap through a server, a failed open, and the scrubbing of the server's secrets.
 *
 * Over HTTP: `test_a_stopped_http_server_is_a_transport_error_and_a_restart_recovers`,
 * `test_a_restarted_session_http_server_is_closed`, a redirect mid-session (Python's
 * `test_an_sdk_redirect_answer_hides_the_location`, through a real server), the DELETE that
 * closing a session server sends, and the scrubbing of a header's value and bare token.
 *
 * Elsewhere: `test_structured_content_failing_the_output_schema_is_a_tool_error` is in
 * `mcp_calls.test.ts`. Not ported: `test_a_raw_timeout_is_a_transport_error` (a failed request is
 * recognized by the error `safeFetch` rejects with, which the stopped-server row covers). The TOOL and AGENT rows
 * (`test_tool_node_retries_follow_the_error_class`,
 * `test_an_agent_model_reads_an_mcp_error_result_and_the_node_succeeds`) are run-level and land
 * with the runs that open servers. Not ported: `test_a_request_error_survives_a_pickle_round_trip`
 * (Python-only); `test_a_call_that_never_resolves_times_out_on_its_own_deadline` (no grace
 * deadline: the SDK's own timer bounds a call); `test_a_closed_stream_after_a_transport_failure_is_a_transport_error`
 * (no ping probe: the v2 client reports a closed connection as its own error, distinct from a
 * server's -32000); `test_all_proxy_is_the_fallback_proxy` (proxies are the runtime's); and the
 * `auto` halves of the parametrized rows, since stdio servers always connect with the legacy
 * handshake.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeAll, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";
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
import { killMarkers, stdioOptions, strayProcesses, trackServers } from "./mcp_helpers.ts";

// Process tests get 15 s; the `< N s` assertions are the real check.
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

  test("test_an_error_result_is_a_reported_but_not_permanent_tool_error", async () => {
    const error = await rejection(tools(server).invoke("fail", { message: "no luck" }));
    expect(error).toBeInstanceOf(McpToolError);
    expect(error.message).toContain("no luck");
    expect(isReportedToolError(error)).toBe(true);
    expect(isPermanentToolError(error)).toBe(false);
  });

  test("test_an_invalid_params_answer_is_reported_and_permanent", async () => {
    const error = await rejection(tools(server).invoke("bad_request", {}));
    expect(error).toBeInstanceOf(McpInvalidRequestError);
    expect(error).toBeInstanceOf(McpRequestError);
    expect((error as McpInvalidRequestError).code).toBe(-32602);
    expect(isReportedToolError(error)).toBe(true);
    expect(isPermanentToolError(error)).toBe(true);
    expect(error.message).toBe(`MCP server '${server.label}' tool 'bad_request': bad params (code -32602)`);
  });

  test("test_a_busy_answer_is_a_request_error_and_the_server_still_works", async () => {
    const error = await rejection(tools(server).invoke("busy", {}));
    expect(error).toBeInstanceOf(McpRequestError);
    expect(error.message).toContain("busy");
    expect((error as McpRequestError).code).toBe(-32000);
    expect(error).not.toBeInstanceOf(McpServerClosed);
    expect(isPermanentToolError(error)).toBe(false);
    expect(server.isClosed).toBe(false);
    expect(await tools(server).invoke("echo", { text: "still here" })).toBe("still here");
  });

  test("test_an_internal_error_answer_is_a_request_error", async () => {
    const error = await rejection(tools(server).invoke("crash", {}));
    expect(error).toBeInstanceOf(McpRequestError);
    expect(error.message).toContain("internal");
    expect((error as McpRequestError).code).toBe(-32603);
    expect(error).not.toBeInstanceOf(McpTransportError);
    expect(isPermanentToolError(error)).toBe(false);
  });

  test(
    "test_a_run_stop_cancels_a_waiting_call_and_the_server_still_works",
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
    "test_a_stop_cancels_only_the_call_of_its_own_run",
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

  test("test_a_call_with_the_stop_already_set_is_cancelled_before_it_is_sent", async () => {
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
    "test_a_call_past_its_timeout_is_a_transient_timeout_error",
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
    "test_a_server_that_exits_is_closed_and_later_calls_fail_at_once",
    async () => {
      const server = await fake([], { label: "quitter" }).open();
      const set = tools(server);
      const error = await rejection(set.invoke("exit_now", {}));
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(isPermanentToolError(error)).toBe(true);
      expect(isReportedToolError(error)).toBe(false);
      expect(error.message).toBe("MCP server 'quitter' is closed");
      expect(server.isClosed).toBe(true);
      expect(server.isOpen).toBe(true);
      const { error: later, ended } = await timed(set.invoke("echo", { text: "x" }));
      expect(later).toBeInstanceOf(McpServerClosed);
      expect(ended).toBeLessThan(0.5);
    },
    SLOW,
  );

  test(
    "test_close_from_another_thread_ends_a_waiting_call",
    async () => {
      const server = await fake([], { label: "closing" }).open();
      const set = tools(server);
      const outcome = timed(set.invoke("slow", { seconds: 30 }));
      await sleep(0.3);
      const closing = server.close();
      const { error, ended } = await outcome;
      expect(error).toBeInstanceOf(McpServerClosed);
      expect(ended).toBeLessThan(0.3 + 3);
      await closing;
      expect(await rejection(set.invoke("echo", { text: "x" }))).toBeInstanceOf(McpServerClosed);
    },
    SLOW,
  );

  test(
    "test_a_late_failure_from_an_earlier_connection_leaves_the_reopened_server_alive",
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
      expect(await rejection(tools(server).invoke("echo", { text: "x" }))).toBeInstanceOf(McpServerClosed);
      expect(server.isClosed).toBe(true);
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

  test("a header's value and its bare token are scrubbed from a call's error", async () => {
    const http = await serveHttp();
    const server = await httpServer({ url: http.url, headers: { "x-secret": "Bearer tok-1234567" } }).open();
    const error = await rejection(tools(server).invoke("echo_secret", {}));
    expect(error).toBeInstanceOf(McpToolError);
    expect(error.message).toBe("secret: \nx-secret: [REDACTED]\ntoken: [REDACTED]");
    expect(error.stack ?? "").not.toContain("tok-1234567");
  });
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
    });
  }

  test("test_a_stopped_http_server_is_a_transport_error_and_a_restart_recovers", async () => {
    let http = await serveHttp();
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
    http = await serveHttp({ port });
    expect(await set.invoke("echo", { text: "back" })).toBe("back");
  });

  test("test_a_restarted_session_http_server_is_closed", async () => {
    const http = await serveHttp({ stateful: true });
    const server = await httpServer({ url: http.url, label: "restarted" }, "legacy").open();
    const set = tools(server);
    expect(await set.invoke("echo", { text: "up" })).toBe("up");
    await http.stop();
    await serveHttp({ port: http.port, stateful: true });
    const error = await rejection(set.invoke("echo", { text: "lost session" }));
    expect(error).toBeInstanceOf(McpServerClosed);
    expect(error.message).toBe("MCP server 'restarted' is closed");
    expect(server.isClosed).toBe(true);
  });

  test("closing a session server sends a DELETE", async () => {
    const http = await serveHttp({ stateful: true });
    const server = await httpServer({ url: http.url }, "legacy").open();
    expect(await tools(server).invoke("echo", { text: "x" })).toBe("x");
    expect(http.recordedMethods).not.toContain("DELETE");
    await server.close();
    expect(http.recordedMethods).toContain("DELETE");
  });
});
