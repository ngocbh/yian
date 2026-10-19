/**
 * `mcpServer`'s argument checks, an unopened `McpServer` and the MCP error classes, a port of
 * the parts of `tests/engine/test_mcp_server.py` that need no running server:
 * `test_bad_arguments_raise_naming_the_field_and_start_nothing`, the label half of
 * `test_label_option_names_the_thread_and_url_label_is_the_host` and the closed half of
 * `test_open_twice_and_tools_when_closed_raise_and_close_is_idempotent`; and yian's stdio
 * transport driven by a bare SDK client, as `tests/engine/test_mcp_fake_server.py` drives the fake
 * server. Opening a server, and the rest of `test_mcp_server.py` and `test_mcp_errors.py`, land
 * with the MCP client.
 */

import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { Client } from "@modelcontextprotocol/client";
import { getDefaultEnvironment } from "@modelcontextprotocol/client/stdio";
import { afterAll, afterEach, describe, expect, test } from "bun:test";

import { BUILTIN, current, install } from "../../src/defaults.ts";

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
  StdioTransport,
} from "../../src/tools/mcp.ts";
import { childAlive, killMarkers, pids, stdioOptions, strayProcesses, trackServers, waitFor } from "./mcp_helpers.ts";

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

describe("StdioTransport", () => {
  // Process tests get 15 s; the `< N s` assertions are the real check.
  const SLOW = 15_000;
  const dir = mkdtempSync(join(tmpdir(), "yian-mcp-stdio-"));
  const markers: string[] = [];
  const tracker = trackServers();

  /** A path in the test directory, remembered so `afterAll` kills the pids it lists. */
  const marker = (name: string): string => {
    const file = join(dir, name);
    markers.push(file);
    return file;
  };

  /** A transport running the fake server with `flags`, closed in `afterEach`. */
  const transport = (flags: string[] = [], env?: Record<string, string>): StdioTransport =>
    tracker.track(
      new StdioTransport({
        ...stdioOptions(...flags),
        env,
        maxMessageBytes: current().maxMcpMessageBytes,
        defaultEnv: getDefaultEnvironment(),
      }),
    );

  /** A bare SDK client connected over `t`, closed in `afterEach`. */
  const connect = async (t: StdioTransport): Promise<Client> => {
    const client = tracker.track(new Client({ name: "test", version: "0" }));
    await client.connect(t);
    return client;
  };

  /** The first text block of a tool result. */
  const textOf = (result: unknown): string =>
    ((result as { content: { text: string }[] }).content[0] as { text: string }).text;

  afterEach(async () => {
    await tracker.closeAll();
    install(BUILTIN);
  });

  afterAll(() => {
    killMarkers(markers);
    rmSync(dir, { recursive: true, force: true });
    expect(strayProcesses()).toEqual([]);
  });

  test(
    "the server sees the default environment and env, not the host's other variables",
    async () => {
      process.env.YIAN_T62_PROBE = "leaked";
      try {
        const client = await connect(transport([], { PASSED: "yes" }));
        const env = async (name: string) => textOf(await client.callTool({ name: "env", arguments: { name } }));
        expect(await env("PATH")).toBe(process.env.PATH!);
        expect(await env("HOME")).toBe(process.env.HOME!);
        expect(await env("PASSED")).toBe("yes");
        expect(await env("YIAN_T62_PROBE")).toBe("");
      } finally {
        delete process.env.YIAN_T62_PROBE;
      }
    },
    SLOW,
  );

  test(
    "close ends the server's whole process group",
    async () => {
      const grandchild = marker("grandchild");
      const t = transport(["--grandchild", grandchild]);
      const client = await connect(t);
      const server = Number(textOf(await client.callTool({ name: "pid", arguments: {} })));
      expect(server).toBe(t.pid!);
      expect(await waitFor(() => pids(grandchild).length === 1, 5000)).toBe(true);
      const [sleeper] = pids(grandchild);
      expect(childAlive(sleeper!)).toBe(true);
      await t.close();
      expect(childAlive(server)).toBe(false);
      expect(childAlive(sleeper!)).toBe(false);
    },
    SLOW,
  );

  test(
    "close in a host with nothing else to do still SIGKILLs a grandchild that ignores SIGTERM",
    () => {
      const grandchild = marker("grandchild-ignores-term");
      const src = (path: string) => JSON.stringify(fileURLToPath(new URL(path, import.meta.url)));
      // `close()` is the host's last statement: only close's own timers keep the host alive.
      const script = `
        import { StdioTransport } from ${src("../../src/tools/mcp.ts")};
        import { pids, stdioOptions, waitFor } from ${src("./mcp_helpers.ts")};
        const file = ${JSON.stringify(grandchild)};
        const t = new StdioTransport({
          ...stdioOptions("--grandchild-ignores-term", file),
          maxMessageBytes: 1024,
          defaultEnv: { ...process.env },
        });
        await t.start();
        if (!(await waitFor(() => pids(file).length === 1, 5000))) throw new Error("no grandchild");
        await t.close();
      `;
      const runtimeFlags = process.versions.bun === undefined ? [] : ["--no-install"];
      const host = spawnSync(process.execPath, [...runtimeFlags, "-e", script], { encoding: "utf8", timeout: 12_000 });
      expect(host.stderr).toBe("");
      expect(host.status).toBe(0);
      const [sleeper] = pids(grandchild);
      expect(sleeper).toBeNumber();
      expect(childAlive(sleeper!)).toBe(false);
    },
    SLOW,
  );

  test(
    "a server that ignores SIGTERM ends by SIGKILL within about 5 s",
    async () => {
      const t = transport(["--stubborn"]);
      await connect(t);
      const pid = t.pid!;
      const start = performance.now();
      await t.close();
      expect(performance.now() - start).toBeLessThan(6000);
      expect(childAlive(pid)).toBe(false);
    },
    SLOW,
  );

  test(
    "a message over the cap is an error and closes the transport",
    async () => {
      install({ ...BUILTIN, maxMcpMessageBytes: 1024 });
      const t = transport(["--flood", "2048"]);
      const client = await connect(t);
      const errors: Error[] = [];
      let closed = 0;
      client.onerror = (error) => errors.push(error);
      client.onclose = () => (closed += 1);
      await expect(client.callTool({ name: "echo", arguments: { text: "hi" } })).rejects.toThrow();
      expect(errors.map((error) => error.message)).toContain("MCP server sent a message over 1024 bytes");
      expect(await waitFor(() => closed === 1, 3000)).toBe(true);
    },
    SLOW,
  );

  test(
    "a message under the cap passes",
    async () => {
      install({ ...BUILTIN, maxMcpMessageBytes: 1024 });
      // The response wraps the text in about 80 bytes of JSON-RPC.
      const client = await connect(transport(["--flood", "900"]));
      expect(textOf(await client.callTool({ name: "echo", arguments: { text: "hi" } }))).toBe("x".repeat(900));
    },
    SLOW,
  );

  test(
    "multibyte text split across pipe chunks echoes intact",
    async () => {
      const client = await connect(transport());
      const text = "é".repeat(50_000);
      expect(textOf(await client.callTool({ name: "echo", arguments: { text } }))).toBe(text);
    },
    SLOW,
  );

  test(
    "the server's stderr goes to the tail, not the host's stderr",
    async () => {
      const writes: string[] = [];
      const write = process.stderr.write.bind(process.stderr);
      process.stderr.write = ((chunk: string | Uint8Array, ...rest: never[]) => {
        writes.push(String(chunk));
        return write(chunk, ...rest);
      }) as typeof process.stderr.write;
      try {
        const t = transport(["--stderr", "boom"]);
        await expect(connect(t)).rejects.toThrow();
        expect(t.stderrTail()).toBe("\nserver stderr:\nboom");
      } finally {
        process.stderr.write = write;
      }
      expect(writes.join("")).not.toContain("boom");
      expect(transport().stderrTail()).toBe("");
    },
    SLOW,
  );

  test(
    "a response that arrives after its request timed out leaves the server running",
    async () => {
      const file = marker("late");
      const t = transport(["--ignore-cancel", "--marker", file]);
      const client = await connect(t);
      const errors: Error[] = [];
      client.onerror = (error) => errors.push(error);
      await expect(
        client.callTool({ name: "slow_big", arguments: { seconds: 1 } }, { timeout: 300 }),
      ).rejects.toThrow();
      expect(await waitFor(() => readFileSync(file, "utf8").includes("answered"), 5000)).toBe(true);
      // The SDK `JSON.stringify`s the late response for its unknown id, which throws on the
      // `bigint`; the transport hands that to `onerror` instead of crashing.
      expect(await waitFor(() => errors.length > 0, 2000)).toBe(true);
      expect(errors[0]).toBeInstanceOf(TypeError);
      expect(childAlive(t.pid!)).toBe(true);
      expect(textOf(await client.callTool({ name: "echo", arguments: { text: "still" } }))).toBe("still");
    },
    SLOW,
  );

  test(
    "big ints survive under structuredContent and safe ones are numbers",
    async () => {
      const client = await connect(transport());
      const big = await client.callTool({ name: "big", arguments: {} });
      expect(big.structuredContent).toEqual({ n: 18446744073709552000n });
      const added = await client.callTool({ name: "add", arguments: { a: 1, b: 2 } });
      expect(added.structuredContent).toEqual({ result: 3 });
      expect(typeof (added.structuredContent as { result: unknown }).result).toBe("number");
    },
    SLOW,
  );

  test(
    "send after close rejects and close twice is fine",
    async () => {
      const t = transport();
      await connect(t);
      await t.close();
      await t.close();
      await expect(t.send({ jsonrpc: "2.0", method: "notifications/initialized" })).rejects.toThrow();
      await expect(t.start()).rejects.toThrow("StdioTransport is closed");
    },
    SLOW,
  );

  test(
    "sends waiting on a full pipe share one drain and one close listener",
    async () => {
      // `--hang` never reads stdin, so the pipe fills and every send waits.
      const t = transport(["--hang"]);
      await t.start();
      const stdin = (t as unknown as { child: { stdin: NodeJS.WritableStream } }).child.stdin;
      const drains = stdin.listenerCount("drain");
      const closes = stdin.listenerCount("close");
      const message = { jsonrpc: "2.0", method: "notifications/x", params: { text: "x".repeat(100_000) } };
      // The first few writes fit in the pipe and resolve; the rest wait, and fail once close ends
      // the server.
      const sends = Promise.allSettled(Array.from({ length: 20 }, () => t.send(message)));
      expect(stdin.listenerCount("drain")).toBe(drains + 1);
      expect(stdin.listenerCount("close")).toBe(closes + 1);
      await t.close();
      expect((await sends).map((result) => result.status)).toContain("rejected");
    },
    SLOW,
  );

  test("start rejects with the spawn error", async () => {
    const t = tracker.track(
      new StdioTransport({
        command: join(dir, "missing"),
        args: [],
        maxMessageBytes: 1024,
        defaultEnv: getDefaultEnvironment(),
      }),
    );
    await expect(t.start()).rejects.toThrow(/ENOENT/);
  });
});
