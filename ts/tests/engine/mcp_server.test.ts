/**
 * `mcpServer`'s argument checks, an unopened `McpServer` and the MCP error classes, a port of
 * the parts of `tests/engine/test_mcp_server.py` that need no running server:
 * `test_bad_arguments_raise_naming_the_field_and_start_nothing`, the label half of
 * `test_label_option_names_the_thread_and_url_label_is_the_host` and the closed half of
 * `test_open_twice_and_tools_when_closed_raise_and_close_is_idempotent`. Opening a server, and the
 * rest of `test_mcp_server.py` and `test_mcp_errors.py`, land with the MCP client.
 */

import { describe, expect, test } from "bun:test";

import {
  isPermanentToolError,
  isReportedToolError,
  isTransientToolError,
  TOOL_ERROR,
  toolErrorKinds,
  type ToolErrorKind,
} from "../../src/tools/errors.ts";
import {
  isMcpError,
  isMcpServer,
  McpCallCancelled,
  McpConnectError,
  McpError,
  McpInvalidRequestError,
  McpRequestError,
  McpServer,
  McpServerClosed,
  mcpServer,
  type McpServerOptions,
  McpTimeoutError,
  McpToolError,
  McpTransportError,
} from "../../src/tools/mcp.ts";

/** Call `mcpServer` with `options`, which may be ill-typed on purpose. */
function build(options: Record<string, unknown>): McpServer {
  return mcpServer(options as McpServerOptions);
}

describe("mcpServer arguments", () => {
  const rows: [Record<string, unknown>, typeof RangeError | typeof TypeError, string][] = [
    [{}, RangeError, "command"],
    [{ command: "x", url: "http://h/mcp" }, RangeError, "url"],
    [{ url: "ftp://x" }, RangeError, "url"],
    [{ command: "x", args: "x" }, TypeError, "args"],
    [{ command: "x", args: ["a", 1] }, TypeError, "args"],
    [{ command: "x", env: { A: 1 } }, TypeError, "env"],
    [{ url: "http://h/mcp", headers: { "x-test": 1 } }, TypeError, "headers"],
    [{ command: "x", timeout: 0 }, RangeError, "timeout"],
    [{ command: "x", connectTimeout: -1 }, RangeError, "connectTimeout"],
    [{ command: "x", timeout: "5" }, TypeError, "timeout"],
    [{ command: "x", cwd: 3 }, TypeError, "cwd"],
    [{ command: 3 }, TypeError, "command"],
    [{ command: "x", concurrencySafe: "x" }, TypeError, "concurrencySafe"],
    [{ command: "x", env: [] }, TypeError, "env"],
    [{ url: "not a url" }, RangeError, "url"],
    [{ url: "http:example.com" }, RangeError, "url"],
    [{ url: "http://h/mcp", args: "" }, TypeError, "args"],
  ];

  for (const [options, error, field] of rows) {
    test(`${JSON.stringify(options)} raises ${error.name} naming ${field}`, () => {
      let thrown: unknown;
      try {
        build(options);
      } catch (caught) {
        thrown = caught;
      }
      expect(thrown).toBeInstanceOf(error);
      expect((thrown as Error).message).toContain(field);
    });
  }

  test("the messages follow Python's wording", () => {
    expect(() => build({})).toThrow("mcpServer needs exactly one of command or url");
    expect(() => build({ command: 3 })).toThrow("mcpServer command must be a string, got number");
    expect(() => build({ command: "" })).toThrow("mcpServer command must not be empty");
    expect(() => build({ command: "x", headers: {} })).toThrow(
      "mcpServer headers needs url; a command server takes env instead",
    );
    expect(() => build({ url: 3 })).toThrow("mcpServer url must be a string, got number");
    expect(() => build({ url: "HTTPS://" })).toThrow("mcpServer url must be an http or https URL with a host");
    expect(() => build({ url: "http://h", args: ["a"] })).toThrow(
      "mcpServer args needs command; a url server takes headers instead",
    );
    expect(() => build({ url: "http://h", env: {} })).toThrow("mcpServer env needs command");
    expect(() => build({ url: "http://h", cwd: "/" })).toThrow("mcpServer cwd needs command");
    expect(() => build({ command: "x", args: "x" })).toThrow(
      "mcpServer args must be an array of strings, not a single string",
    );
    expect(() => build({ command: "x", args: ["a", 1] })).toThrow("mcpServer args must hold only strings");
    expect(() => build({ command: "x", env: [] })).toThrow("mcpServer env must be a record of strings to strings");
    expect(() => build({ command: "x", env: { A: 1 } })).toThrow(
      "mcpServer env must map strings to strings; 'A' doesn't",
    );
    expect(() => build({ command: "x", cwd: 3 })).toThrow("mcpServer cwd must be a string, got number");
    expect(() => build({ command: "x", timeout: "5" })).toThrow(
      "mcpServer timeout must be a number of seconds, got string",
    );
    expect(() => build({ command: "x", connectTimeout: -1 })).toThrow(
      "mcpServer connectTimeout must be positive and finite, got -1",
    );
    expect(() => build({ command: "x", timeout: Infinity })).toThrow(
      "mcpServer timeout must be positive and finite, got Infinity",
    );
    expect(() => build({ command: "x", timeout: NaN })).toThrow(RangeError);
    expect(() => build({ command: "x", label: 3 })).toThrow("mcpServer label must be a string, got number");
    expect(() => build({ command: "x", label: "" })).toThrow("mcpServer label must not be empty");
    expect(() => build({ command: "x", concurrencySafe: ["a", 1] })).toThrow(
      "mcpServer concurrencySafe must be a boolean or an array of tool names",
    );
  });

  test("a null option counts as absent, as Python's None", () => {
    const server = build({
      command: "x",
      url: null,
      args: null,
      env: null,
      cwd: null,
      headers: null,
      timeout: null,
      connectTimeout: null,
      label: null,
      concurrencySafe: null,
    });
    expect(server.label).toBe("x");
    expect((server as unknown as { spec: Record<string, unknown> }).spec).toEqual({
      command: "x",
      args: [],
      env: undefined,
      cwd: undefined,
      url: undefined,
      headers: undefined,
      timeout: undefined,
      connectTimeout: undefined,
      label: undefined,
      concurrencySafe: false,
    });
    expect(build({ url: "http://h/mcp", command: null, args: null, env: null, cwd: null }).label).toBe("h");
  });

  test("good arguments build an unopened server", () => {
    expect(isMcpServer(build({ url: "http://h/mcp", args: [] }))).toBe(true);
    expect(isMcpServer(build({ url: "HTTPS://h/mcp", headers: { Authorization: "Bearer t" } }))).toBe(true);
    const server = mcpServer({
      command: "npx",
      args: ["-y", "server"],
      env: { LOG_LEVEL: "warn" },
      cwd: "/srv",
      timeout: 5,
      connectTimeout: 0.5,
      concurrencySafe: ["read"],
    });
    expect(server).toBeInstanceOf(McpServer);
    expect(server.isOpen).toBe(false);
    expect(mcpServer({ command: "x", concurrencySafe: true }).isOpen).toBe(false);
  });
});

describe("McpServer", () => {
  test("label is the option, else the command's basename, else the URL's host", () => {
    expect(mcpServer({ command: "/usr/bin/python3", label: "tools" }).label).toBe("tools");
    expect(mcpServer({ command: "/usr/bin/python3" }).label).toBe("python3");
    expect(mcpServer({ command: "npx" }).label).toBe("npx");
    expect(mcpServer({ command: "bin/" }).label).toBe("bin/");
    expect(mcpServer({ url: "https://example.com:8443/mcp" }).label).toBe("example.com");
  });

  test("an unopened server has no tools and closes twice", async () => {
    const server = mcpServer({ command: "x", label: "files" });
    expect(server.isOpen).toBe(false);
    expect(server.isClosed).toBe(false);
    expect(() => server.tools).toThrow("MCP server 'files' is not open");
    await server.close();
    await server.close();
    expect(() => server.tools).toThrow("is not open");
  });

  test("open is not ported yet", async () => {
    await expect(mcpServer({ command: "x" }).open()).rejects.toThrow("not ported yet (T6-2)");
  });

  test("isMcpServer reads the brand", () => {
    expect(isMcpServer(mcpServer({ command: "x" }))).toBe(true);
    expect(isMcpServer({})).toBe(false);
    expect(isMcpServer(null)).toBe(false);
    expect(isMcpServer({ [Symbol.for("yian.mcp_server")]: true })).toBe(true);
  });
});

describe("MCP errors", () => {
  test("each class has its tool error kinds", () => {
    const cases: [Error, readonly ToolErrorKind[]][] = [
      [new McpError("m"), []],
      [new McpConnectError("m"), []],
      [new McpToolError("m"), ["reported"]],
      [new McpRequestError("m", -32000), ["reported"]],
      [new McpInvalidRequestError("m", -32602), ["reported", "permanent"]],
      [new McpTimeoutError("m"), ["reported", "transient"]],
      [new McpTransportError("m"), ["reported", "transient"]],
      [new McpServerClosed("m"), ["permanent"]],
      [new McpCallCancelled("m"), ["permanent"]],
    ];
    for (const [error, kinds] of cases) {
      expect(isMcpError(error)).toBe(true);
      expect(error).toBeInstanceOf(McpError);
      expect(error.name).toBe(error.constructor.name);
      expect(error.message).toBe("m");
      expect([...toolErrorKinds(error)].sort()).toEqual([...kinds].sort());
    }
    expect(isReportedToolError(new McpToolError("m"))).toBe(true);
    expect(isPermanentToolError(new McpInvalidRequestError("m", -32600))).toBe(true);
    expect(isTransientToolError(new McpTimeoutError("m"))).toBe(true);
    expect(isPermanentToolError(new McpTimeoutError("m"))).toBe(false);
    expect(TOOL_ERROR in new McpConnectError("m")).toBe(false);
  });

  test("request errors keep the code and transient ones have no retryAfter", () => {
    const error = new McpInvalidRequestError("bad params", -32602);
    expect(error).toBeInstanceOf(McpRequestError);
    expect(error.code).toBe(-32602);
    expect(new McpRequestError("m", -32000).code).toBe(-32000);
    expect(new McpTimeoutError("m").retryAfter).toBeNull();
    expect(new McpTransportError("m").retryAfter).toBeNull();
  });

  test("isMcpError reads the brand", () => {
    expect(isMcpError(new Error("m"))).toBe(false);
    expect(isMcpError("m")).toBe(false);
    expect(isMcpError({ [Symbol.for("yian.mcp_error")]: true })).toBe(true);
    const hostile = new Proxy(
      {},
      {
        get() {
          throw new Error("trap");
        },
      },
    );
    expect(isMcpError(hostile)).toBe(false);
    expect(isMcpServer(hostile)).toBe(false);
  });
});
