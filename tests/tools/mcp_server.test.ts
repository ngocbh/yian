/** `McpServer` opens a server, lists its tools once, and closes. */

import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { Client } from "@modelcontextprotocol/client";
import { getDefaultEnvironment, StdioClientTransport } from "@modelcontextprotocol/client/stdio";
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
import { FunctionTool } from "../../src/tools/function_tools.ts";
import {
  AlreadyOpenError,
  capResponse,
  isAlreadyOpenError,
  isMcpError,
  isMcpServer,
  McpCallCancelled,
  McpConnectError,
  McpError,
  McpInvalidRequestError,
  McpRequestError,
  McpSdkMissingError,
  McpServer,
  McpServerClosed,
  mcpServer,
  type McpServerOptions,
  McpTimeoutError,
  McpToolError,
  McpTransportError,
  listAll,
  StdioTransport,
} from "../../src/tools/mcp.ts";
import { RedirectError } from "../../src/platform.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import {
  type HttpServer,
  type HttpServerOptions,
  startHttpServer,
  startRedirectServer,
} from "../fixtures/mcp/fake_server.ts";
import {
  childAlive,
  freePort,
  killMarkers,
  NODE,
  pids,
  ROOT,
  stdioOptions,
  strayProcesses,
  trackServers,
  waitFor,
} from "../helpers/mcp_helpers.ts";

/** How an HTTP server negotiates the protocol era: `auto`, the default, or the private `legacy`. */
type Mode = "auto" | "legacy";
const MODES: readonly Mode[] = ["auto", "legacy"];

/** Node rows get 90 s: Node starts the built code and the fixture from cold. */
const ON_NODE = 90_000;

/** The `file:` URL of `file` under the package root, as a JavaScript string literal for an inline script. */
const builtUrl = (file: string): string => JSON.stringify(pathToFileURL(join(ROOT, file)).href);

/** Run the module `script` on the Node from `YIAN_TEST_NODE`, with `flags` before it. */
const runNode = (script: string, flags: string[] = []) =>
  spawnSync(NODE!, [...flags, "--input-type=module", "-e", script], { cwd: ROOT, encoding: "utf8", timeout: 60_000 });

/** Call `mcpServer` with `options`, which may be ill-typed on purpose. */
function build(options: Record<string, unknown>): McpServer {
  return mcpServer(options as McpServerOptions);
}

describe("mcpServer arguments", () => {
  const rows: [Record<string, unknown>, typeof RangeError | typeof TypeError, string][] = [
    [{}, RangeError, "command"],
    [{ command: "x", url: "https://h/mcp" }, RangeError, "url"],
    [{ url: "ftp://x" }, RangeError, "url"],
    [{ command: "x", args: "x" }, TypeError, "args"],
    [{ command: "x", args: ["a", 1] }, TypeError, "args"],
    [{ command: "x", env: { A: 1 } }, TypeError, "env"],
    [{ url: "https://h/mcp", headers: { "x-test": 1 } }, TypeError, "headers"],
    [{ command: "x", timeout: 0 }, RangeError, "timeout"],
    [{ command: "x", connectTimeout: -1 }, RangeError, "connectTimeout"],
    [{ command: "x", timeout: "5" }, TypeError, "timeout"],
    [{ command: "x", cwd: 3 }, TypeError, "cwd"],
    [{ command: 3 }, TypeError, "command"],
    [{ command: "x", concurrencySafe: "x" }, TypeError, "concurrencySafe"],
    [{ command: "x", env: [] }, TypeError, "env"],
    [{ url: "not a url" }, RangeError, "url"],
    [{ url: "http:example.com" }, RangeError, "url"],
    [{ url: "https://h/mcp", args: "" }, TypeError, "args"],
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

  test("the messages name the option and the value's type", () => {
    expect(() => build({})).toThrow("mcpServer needs exactly one of command or url");
    expect(() => build({ command: 3 })).toThrow("mcpServer command must be a string, got number");
    expect(() => build({ command: "" })).toThrow("mcpServer command must not be empty");
    expect(() => build({ command: "x", headers: {} })).toThrow(
      "mcpServer headers needs url; a command server takes env instead",
    );
    expect(() => build({ url: 3 })).toThrow("mcpServer url must be a string, got number");
    expect(() => build({ url: "HTTPS://" })).toThrow("mcpServer url must be an http or https URL with a host");
    expect(() => build({ url: "https://h", args: ["a"] })).toThrow(
      "mcpServer args needs command; a url server takes headers instead",
    );
    expect(() => build({ url: "https://h", env: {} })).toThrow("mcpServer env needs command");
    expect(() => build({ url: "https://h", cwd: "/" })).toThrow("mcpServer cwd needs command");
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

  test("a null option counts as absent", () => {
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
    expect(build({ url: "https://h/mcp", command: null, args: null, env: null, cwd: null }).label).toBe("h");
  });

  test("an http url is refused unless its host is localhost, 127.0.0.1 or ::1, as base_url's is", () => {
    const refused = "mcpServer url must use https; http is allowed only for localhost, 127.0.0.1, and ::1";
    for (const url of [
      "http://mcp.example.com/mcp?token=abc",
      "http://192.168.1.5:8080/mcp",
      "http://127.0.0.2/mcp",
      "http://localhost.example.com/mcp",
      "http://[::2]/mcp",
      "HTTP://mcp.example.com/mcp",
    ]) {
      expect(() => build({ url, headers: { Authorization: "Bearer sk-secret-123" } })).toThrow(refused);
    }
    for (const url of [
      "http://localhost:3000/mcp",
      "http://LOCALHOST/mcp",
      "http://127.0.0.1/mcp",
      "http://[::1]:8080/mcp",
      "https://mcp.example.com/mcp",
    ]) {
      expect(isMcpServer(build({ url }))).toBe(true);
    }
  });

  test("good arguments build an unopened server", () => {
    expect(isMcpServer(build({ url: "https://h/mcp", args: [] }))).toBe(true);
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
      [new McpSdkMissingError("m"), []],
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
    expect(isAlreadyOpenError(hostile)).toBe(false);
  });

  test("isAlreadyOpenError reads the brand", () => {
    expect(isAlreadyOpenError(new AlreadyOpenError("m"))).toBe(true);
    expect(isAlreadyOpenError(new Error("m"))).toBe(false);
    expect(isAlreadyOpenError(new McpError("m"))).toBe(false);
    expect(isAlreadyOpenError({ [Symbol.for("yian.mcp_already_open")]: true })).toBe(true);
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
        import { pids, stdioOptions, waitFor } from ${src("../helpers/mcp_helpers.ts")};
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

  test.skipIf(!NODE)(
    "on Node, built: close in a host with nothing else to do still SIGKILLs a grandchild that ignores SIGTERM",
    () => {
      // Under `bun test` the host above is Bun, which finishes the close anyway; Node exits once
      // nothing keeps it alive, so this row is the one that guards close's timers.
      const grandchild = marker("grandchild-ignores-term-node");
      const script = `
        import { StdioTransport } from ${builtUrl("dist/src/tools/mcp.js")};
        import { pids, stdioOptions, waitFor } from ${builtUrl("dist/tests/helpers/mcp_helpers.js")};
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
      const host = runNode(script);
      expect(host.stderr).toBe("");
      expect(host.status).toBe(0);
      const [sleeper] = pids(grandchild);
      expect(sleeper).toBeNumber();
      expect(childAlive(sleeper!)).toBe(false);
    },
    ON_NODE,
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
      const message = { jsonrpc: "2.0", method: "notifications/x", params: { text: "x".repeat(1_000_000) } };
      // How many writes the runtime accepts at once varies (Bun on Linux takes 16 MB before a
      // write waits); every send after those waits for a drain that never comes, and fails once
      // close ends the server.
      const sends = Promise.allSettled(Array.from({ length: 40 }, () => t.send(message)));
      expect(stdin.listenerCount("drain")).toBe(drains + 1);
      expect(stdin.listenerCount("close")).toBe(closes + 1);
      await t.close();
      const statuses = (await sends).map((result) => result.status);
      expect(statuses).toContain("rejected");
      // No send resolves after one waited: "fulfilled" sorts before "rejected".
      expect(statuses).toEqual([...statuses].sort());
    },
    SLOW,
  );

  /** A transport running `script` under `sh`, closed in `afterEach`. */
  const shell = (script: string): StdioTransport =>
    tracker.track(
      new StdioTransport({
        command: "sh",
        args: ["-c", script],
        maxMessageBytes: 1024,
        defaultEnv: getDefaultEnvironment(),
      }),
    );

  test(
    "onclose comes once the exit code is known, which exitText gives with the stderr tail",
    async () => {
      const t = shell("echo oops >&2; exit 4");
      let seen: string | undefined;
      t.onclose = () => (seen = t.exitText());
      await t.start();
      expect(await waitFor(() => seen !== undefined, 3000)).toBe(true);
      expect(seen).toBe("exited (code 4)");
      expect(t.stderrTail()).toBe("\nserver stderr:\noops");
    },
    SLOW,
  );

  test(
    "a server that closes its stdout and keeps running closed its stdout, even once close kills it",
    async () => {
      const t = shell("exec >&-; sleep 30");
      let closed = false;
      t.onclose = () => (closed = true);
      await t.start();
      expect(await waitFor(() => closed, 3000)).toBe(true);
      expect(t.exitText()).toBe("closed its stdout");
      await t.close();
      expect(t.exitText()).toBe("closed its stdout");
    },
    SLOW,
  );

  test(
    "a server that close ends tells nothing of how it ended",
    async () => {
      const t = shell("sleep 30");
      await t.start();
      expect(t.exitText()).toBe("");
      await t.close();
      expect(t.exitText()).toBe("");
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

describe("capResponse", () => {
  /** A 200 response of `type` whose body arrives as `chunks`. */
  function reply(type: string, ...chunks: string[]): Response {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    });
    return new Response(body, { status: 200, headers: { "content-type": type } });
  }

  /** The body of `response` capped at `max` bytes, and how often the cap fired. */
  async function read(response: Response, max: number): Promise<{ text?: string; error?: Error; overflows: number }> {
    let overflows = 0;
    const capped = capResponse(response, max, () => (overflows += 1));
    try {
      return { text: await capped.text(), overflows };
    } catch (error) {
      return { error: error as Error, overflows };
    }
  }

  test("a JSON body is capped as a whole, across chunks", async () => {
    expect(await read(reply("application/json", "x".repeat(10), "x".repeat(6)), 16)).toEqual({
      text: "x".repeat(16),
      overflows: 0,
    });
    const over = await read(reply("application/json", "x".repeat(10), "secret".repeat(2)), 16);
    expect(over.overflows).toBe(1);
    expect(over.error?.message).toBe("MCP server sent a message over 16 bytes");
  });

  test("an SSE body is capped per event, so many small events pass", async () => {
    const event = `data: ${"x".repeat(10)}\n\n`;
    const body = event.repeat(5);
    expect(await read(reply("text/event-stream; charset=utf-8", body), 16)).toEqual({ text: body, overflows: 0 });
  });

  test("an SSE event's lines add up until its blank line, whatever the line ends", async () => {
    for (const end of ["\n", "\r", "\r\n"]) {
      const lines = `data: ${"x".repeat(5)}${end}`.repeat(3);
      const over = await read(reply("text/event-stream", lines, end), 32);
      expect(over.overflows).toBe(1);
      const split = `data: ${"x".repeat(10)}${end}${end}`;
      // A CRLF split across chunks still ends one line, not two.
      const passing = end === "\r\n" ? [split.slice(0, -1), "\n", split] : [split, split];
      expect(await read(reply("text/event-stream", ...passing), 16)).toEqual({
        text: passing.join(""),
        overflows: 0,
      });
    }
  });

  test("a body-less answer passes as is", () => {
    const empty = new Response(null, { status: 202 });
    expect(capResponse(empty, 1, () => {})).toBe(empty);
    const noContent = new Response(null, { status: 204 });
    expect(capResponse(noContent, 1, () => {})).toBe(noContent);
  });
});

describe("McpServer open and close", () => {
  // Process and HTTP tests get 15 s; the `< N s` assertions are the real check.
  const SLOW = 15_000;
  const dir = mkdtempSync(join(tmpdir(), "yian-mcp-open-"));
  const markers: string[] = [];
  const tracker = trackServers();
  let markerCount = 0;

  /** A fresh marker path in the test directory, remembered so `afterAll` kills the pids it lists. */
  const marker = (): string => {
    const file = join(dir, `marker-${(markerCount += 1)}`);
    markers.push(file);
    return file;
  };

  /** An unopened server running the fake server with `flags`, closed in `afterEach`. */
  const server = (flags: string[] = [], options: Omit<McpServerOptions, "command" | "args"> = {}): McpServer =>
    tracker.track(mcpServer({ ...stdioOptions(...flags), ...options }));

  /** The tools a bare SDK client lists from the fake server started with `flags`. */
  const bareListing = async (...flags: string[]) => {
    const client = tracker.track(new Client({ name: "test", version: "0" }));
    await client.connect(new StdioClientTransport(stdioOptions(...flags)));
    return (await client.listTools()).tools;
  };

  /** The error `promise` rejects with; fails the test when it resolves. */
  const rejection = async (promise: Promise<unknown>): Promise<Error> => {
    try {
      await promise;
    } catch (error) {
      return error as Error;
    }
    throw new Error("expected a rejection");
  };

  /** A stand-in lister serving `tools` as one listing. */
  const standIn = (tools: unknown[], capabilities: Record<string, unknown> = { tools: {} }) => {
    const calls: unknown[] = [];
    return {
      calls,
      listTools: async (params: unknown) => {
        calls.push(params);
        return { tools };
      },
      getServerCapabilities: () => capabilities,
    };
  };

  /** `listAll`'s options for a server labelled `label`. */
  const listing = (label: string, concurrencySafe: boolean | readonly string[] = false) => ({
    label,
    concurrencySafe,
    call: () => Promise.reject(new Error("no calls here")),
  });

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
    "open lists the tools and close stops the server",
    async () => {
      const listed = await bareListing();
      const file = marker();
      let opened: McpServer;
      let pid: number;
      {
        const s = await mcpServer(stdioOptions("--marker", file)).open();
        try {
          opened = s;
          expect(s.isOpen).toBe(true);
          expect(s.isClosed).toBe(false);
          expect(s.openedByRun).toBe(false);
          expect(s.label).toBe(basename(process.execPath));
          expect(s.tools.map((tool) => tool.name)).toEqual(listed.map((tool) => tool.name));
          expect(Object.isFrozen(s.tools)).toBe(true);
          s.tools.forEach((tool, i) => {
            expect(tool).toBeInstanceOf(FunctionTool);
            expect(tool.parametersJsonSchema as unknown).toEqual(listed[i]!.inputSchema);
            expect(tool.description).toBe(listed[i]!.description!);
            expect(tool.outputJsonSchema).toBeNull();
            expect(tool.isConcurrencySafe).toBeNull();
            expect(tool.resumable).toBe(false);
          });
          [pid] = pids(file) as [number];
          expect(childAlive(pid)).toBe(true);
        } finally {
          await s[Symbol.asyncDispose]();
        }
      }
      expect(opened.isOpen).toBe(false);
      expect(opened.openedByRun).toBe(false);
      expect(() => opened.tools).toThrow("is not open");
      expect(childAlive(pid)).toBe(false);
    },
    SLOW,
  );

  test(
    "the label option names the server and a url's label is its host",
    async () => {
      const s = server(["--only", "echo"], { label: "tools" });
      await s.open();
      expect(s.label).toBe("tools");
      expect(s.tools.map((tool) => tool.name)).toEqual(["echo"]);
    },
    SLOW,
  );

  test(
    "open for run records the run token",
    async () => {
      const s = server(["--only", "echo"]);
      try {
        expect(await s.openForRun("run-token")).toBe(s);
        expect(s.openedByRun).toBe(true);
        expect(s.tools.map((tool) => tool.name)).toEqual(["echo"]);
      } finally {
        await s.close();
      }
      expect(s.openedByRun).toBe(false);
    },
    SLOW,
  );

  test(
    "paging lists every page",
    async () => {
      const two = await server(["--pages", "2"]).open();
      expect(two.tools.map((tool) => tool.name)).toEqual(["page1", "page2"]);
      const three = await server(["--pages", "3"]).open();
      expect(three.tools.map((tool) => tool.name)).toEqual(["page1", "page2", "page3"]);
    },
    SLOW,
  );

  test(
    "more than 100 pages fails to open",
    async () => {
      const s = server(["--pages", "101"], { label: "paged" });
      const error = await rejection(s.open());
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toBe("MCP server 'paged' lists more than 100 pages of tools");
      expect(s.isOpen).toBe(false);
      const hundred = await server(["--pages", "100"]).open();
      expect(hundred.tools).toHaveLength(100);
    },
    SLOW,
  );

  test(
    "tool list is a snapshot",
    async () => {
      const s = await server().open();
      const names = s.tools.map((tool) => tool.name);
      expect(await new ToolSet({ fake: [...s.tools] }).invoke("add_tool", {})).toBe("added");
      await new Promise((resolve) => setTimeout(resolve, 300));
      expect(s.tools.map((tool) => tool.name)).toEqual(names);
      expect(names).not.toContain("late");
    },
    SLOW,
  );

  test(
    "a tool listed twice fails to open naming it",
    async () => {
      const s = server(["--twins"], { label: "listing" });
      const error = await rejection(s.open());
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toBe("MCP server 'listing' lists the tool 'twin' twice");
      expect(s.isOpen).toBe(false);
    },
    SLOW,
  );

  test("a tool whose schema is not an object is rejected", async () => {
    // The SDK refuses such a listing on both ends, so a stand-in lister serves it.
    const lister = standIn([{ name: "loose", inputSchema: { properties: {} } }]);
    const error = await rejection(listAll(lister, listing("listing")));
    expect(error).toBeInstanceOf(McpConnectError);
    expect(error.message).toBe("MCP server 'listing' tool 'loose' has an input schema whose type is not 'object'");
  });

  test("a schema FunctionTool refuses makes the tool unusable", async () => {
    let deep: Record<string, unknown> = { type: "string" };
    for (let i = 0; i < 120; i++) deep = { type: "object", properties: { a: deep } };
    const error = await rejection(listAll(standIn([{ name: "deep", inputSchema: deep }]), listing("listing")));
    expect(error).toBeInstanceOf(McpConnectError);
    expect(error.message).toStartWith("MCP server 'listing' tool 'deep' is unusable: ");
    expect(error.message).toContain("deeper than");
    expect(error.cause).toBeInstanceOf(TypeError);
  });

  test("a stand-in listing becomes FunctionTools, marked and with their descriptions", async () => {
    const schema = { type: "object", properties: {} };
    const lister = standIn([
      { name: "a", description: "The a tool.", title: "A", inputSchema: schema },
      { name: "b", title: "B", inputSchema: schema, outputSchema: schema },
      { name: "c", inputSchema: schema },
    ]);
    const { tools } = await listAll(lister, listing("listing", ["b"]));
    expect(tools.map((tool) => tool.description)).toEqual(["The a tool.", "B", "c"]);
    expect(tools.map((tool) => tool.isConcurrencySafe?.({}) ?? null)).toEqual([null, true, null]);
    expect(tools.map((tool) => tool.outputJsonSchema)).toEqual([null, null, null]);
    // An MCP call has no idempotency key yian can rely on, so no MCP tool is resumable.
    expect(tools.map((tool) => tool.resumable)).toEqual([false, false, false]);
    expect(lister.calls).toEqual([undefined]);
  });

  test("a server without the tools capability lists nothing and sends no tools/list", async () => {
    const lister = standIn([{ name: "a", inputSchema: { type: "object" } }], {});
    expect((await listAll(lister, listing("listing"))).tools).toEqual([]);
    expect(lister.calls).toEqual([]);
  });

  test(
    "--pages 0 opens with no tools and sends no tools/list",
    async () => {
      const debug = console.debug;
      const lines: unknown[] = [];
      console.debug = (...args: unknown[]) => lines.push(args);
      try {
        const s = await server(["--pages", "0"]).open();
        expect(s.tools).toEqual([]);
        expect(s.isOpen).toBe(true);
      } finally {
        console.debug = debug;
      }
      // The SDK prints a debug line when `listTools` runs against a server without `tools`.
      expect(lines).toEqual([]);
    },
    SLOW,
  );

  test(
    "missing command fails to open with the os error",
    async () => {
      const s = tracker.track(mcpServer({ command: "/nonexistent/yian-no-such-server" }));
      const error = await rejection(s.open());
      expect(error).toBeInstanceOf(McpConnectError);
      expect(isMcpError(error)).toBe(true);
      expect(error.message).toStartWith("MCP server 'yian-no-such-server' didn't start: ");
      expect(error.message).toContain("yian-no-such-server");
      expect((error.cause as NodeJS.ErrnoException).code).toBe("ENOENT");
      expect(s.isOpen).toBe(false);
    },
    SLOW,
  );

  test(
    "server stderr is in the connect error",
    async () => {
      const error = await rejection(server(["--stderr", "boom"], { label: "noisy" }).open());
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toStartWith("MCP server 'noisy' didn't start: ");
      // The tail also proves stderr is piped: an inherited stderr would leave it empty.
      expect(error.message).toEndWith("\nserver stderr:\nboom");
      expect(error.cause).toBeInstanceOf(Error);
    },
    SLOW,
  );

  test(
    "hung server times out and is killed",
    async () => {
      const file = marker();
      const s = server(["--hang", "--marker", file], { connectTimeout: 1, label: "hung" });
      const start = performance.now();
      const error = await rejection(s.open());
      expect(performance.now() - start).toBeLessThan(5000);
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toBe("MCP server 'hung' didn't start within 1 s");
      expect(s.isOpen).toBe(false);
      expect(childAlive(pids(file)[0]!)).toBe(false);
    },
    SLOW,
  );

  test(
    "a connect timeout too small for fixed notation prints in exponent form",
    async () => {
      const s = server(["--hang", "--marker", marker()], { connectTimeout: 0.00001, label: "hung" });
      const error = await rejection(s.open());
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toBe("MCP server 'hung' didn't start within 1e-05 s");
    },
    SLOW,
  );

  test(
    "a server that exits on a request before initialize still opens",
    async () => {
      const s = await server(["--exit-before-init", "--only", "echo"]).open();
      expect(s.tools.map((tool) => tool.name)).toEqual(["echo"]);
    },
    SLOW,
  );

  test(
    "open twice and tools when closed raise and close is idempotent",
    async () => {
      const s = server(["--only", "echo"], { label: "twice" });
      await s.close();
      expect(() => s.tools).toThrow("MCP server 'twice' is not open");
      await s.open();
      try {
        const error = await rejection(s.open());
        expect(error).toBeInstanceOf(AlreadyOpenError);
        expect(error.message).toBe("MCP server 'twice' is already open");
        expect(s.isOpen).toBe(true);
      } finally {
        await s.close();
      }
      await s.close();
      expect(() => s.tools).toThrow("not open");
      // Two opens that race: the second is refused at once, the first opens.
      const results = await Promise.allSettled([s.open(), s.openForRun("run")]);
      expect(results[0]!.status).toBe("fulfilled");
      expect((results[1] as PromiseRejectedResult).reason).toBeInstanceOf(AlreadyOpenError);
      expect(s.openedByRun).toBe(false);
    },
    SLOW,
  );

  test(
    "a close during open fails the open",
    async () => {
      const file = marker();
      const s = server(["--hang", "--marker", file], { connectTimeout: 20, label: "starting" });
      const opening = rejection(s.open());
      expect(await waitFor(() => pids(file).length === 1, 5000)).toBe(true);
      const start = performance.now();
      await s.close();
      const error = await opening;
      expect(performance.now() - start).toBeLessThan(3000);
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toBe("MCP server 'starting' was closed while starting");
      expect(s.isOpen).toBe(false);
      expect(childAlive(pids(file)[0]!)).toBe(false);
    },
    SLOW,
  );

  test(
    "a close during the SDK import, before the transport exists, still fails the open",
    async () => {
      const file = marker();
      const s = server(["--marker", file], { label: "early" });
      const opening = rejection(s.open());
      await s.close();
      const error = await opening;
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toBe("MCP server 'early' was closed while starting");
      expect(s.isOpen).toBe(false);
      expect(pids(file)).toEqual([]);
    },
    SLOW,
  );

  test(
    "an abort during open closes the server",
    async () => {
      const file = marker();
      const s = server(["--hang", "--marker", file], { connectTimeout: 20 });
      const controller = new AbortController();
      const opening = rejection(s.open({ signal: controller.signal }));
      expect(await waitFor(() => pids(file).length === 1, 5000)).toBe(true);
      const reason = new Error("stopped by the host");
      controller.abort(reason);
      expect(await opening).toBe(reason);
      expect(s.isOpen).toBe(false);
      expect(childAlive(pids(file)[0]!)).toBe(false);
    },
    SLOW,
  );

  test(
    "a close leaves alone a connection opened while it waited",
    async () => {
      const file = marker();
      const s = server(["--only", "echo", "--marker", file]);
      await s.open();
      const late = s.close();
      await s.open();
      await late;
      const [older, newer] = pids(file) as [number, number];
      expect(s.isOpen).toBe(true);
      expect(s.isClosed).toBe(false);
      expect(childAlive(older)).toBe(false);
      expect(childAlive(newer)).toBe(true);
      expect(s.tools.map((tool) => tool.name)).toEqual(["echo"]);
    },
    SLOW,
  );

  test(
    "concurrencySafe marks every tool, or those named",
    async () => {
      const safe = async (s: McpServer, name: string, args: Record<string, unknown>) => {
        const tools = new ToolSet({ fake: s });
        return tools.isConcurrencySafe(await tools.validate(name, args));
      };
      const all = await server(["--only", "echo,add"], { concurrencySafe: true }).open();
      expect(await safe(all, "echo", { text: "x" })).toBe(true);
      expect(await safe(all, "add", { a: 1, b: 2 })).toBe(true);
      const some = await server(["--only", "echo,add"], { concurrencySafe: ["echo"] }).open();
      expect(await safe(some, "echo", { text: "x" })).toBe(true);
      expect(await safe(some, "add", { a: 1, b: 2 })).toBe(false);
      expect(some.tools.find((tool) => tool.name === "add")!.isConcurrencySafe).toBeNull();
    },
    SLOW,
  );

  test(
    "concurrencySafe naming a tool the server doesn't list fails the open",
    async () => {
      const s = server(["--only", "echo"], { concurrencySafe: ["echo", "nope", "gone"], label: "marks" });
      const error = await rejection(s.open());
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toBe(`MCP server 'marks' marks tools it doesn't list as concurrency-safe: ["nope","gone"]`);
      expect(s.isOpen).toBe(false);
    },
    SLOW,
  );

  /** The fake server over HTTP, stopped in `afterEach`. */
  const serveHttp = async (options: HttpServerOptions = {}): Promise<HttpServer> => {
    const http = await startHttpServer(options);
    tracker.track({ close: () => http.stop() });
    return http;
  };

  /** An unopened server built from `options` that connects in `mode`, closed in `afterEach`. */
  const httpServer = (options: McpServerOptions, mode: Mode): McpServer => {
    const s = tracker.track(mcpServer(options));
    (s as unknown as { mode: Mode }).mode = mode;
    return s;
  };

  for (const mode of MODES) {
    test(`http server lists tools and sends headers (${mode})`, async () => {
      const http = await serveHttp();
      const s = httpServer({ url: http.url, headers: { "x-test": "v" } }, mode);
      await s.open();
      const names = s.tools.map((tool) => tool.name);
      expect(await new ToolSet({ fake: [...s.tools] }).invoke("headers", {})).toBe("v");
      await s.close();
      expect(http.recordedHeaders.some((headers) => headers["x-test"] === "v")).toBe(true);
      expect(names).toEqual((await bareListing()).map((tool) => tool.name));
      expect(s.label).toBe("localhost");
    }, SLOW);

    test(`redirect refusal at open (${mode})`, async () => {
      const real = await serveHttp();
      const redirect = await startRedirectServer(real.url);
      tracker.track({ close: () => redirect.stop() });
      const s = httpServer({ url: redirect.url, label: "moved" }, mode);
      const error = await rejection(s.open());
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).toBe("MCP server 'moved' didn't start: RedirectError (HTTP 307)");
      expect(error.cause).toBeInstanceOf(RedirectError);
      expect(s.isOpen).toBe(false);
      expect(redirect.hits).toBeGreaterThan(0);
      // Not even a request with the real server's host header reached it.
      expect(real.recordedHeaders).toEqual([]);
    }, SLOW);
  }

  test("http server that is down fails without leaking the url", async () => {
    const port = await freePort();
    const messages: string[] = [];
    for (const mode of MODES) {
      const s = httpServer({ url: `http://localhost:${port}/mcp?token=SECRET`, connectTimeout: 5 }, mode);
      const error = await rejection(s.open());
      expect(error).toBeInstanceOf(McpConnectError);
      expect(error.message).not.toContain("SECRET");
      expect(error.message).not.toContain(String(port));
      expect(error.cause).toBeInstanceOf(Error);
      expect(s.isOpen).toBe(false);
      messages.push(error.message);
    }
    // The `auto` probe's `EraNegotiationFailed` wrapper is unwrapped, so both modes say the same.
    expect(messages).toEqual(["MCP server 'localhost' didn't start: TypeError", messages[0]!]);
  }, SLOW);
});

describe("McpServer on Node, built", () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-mcp-node-"));
  const markers: string[] = [];

  /** A path in the test directory, remembered so `afterAll` kills the pids it lists. */
  const marker = (name: string): string => {
    const file = join(dir, name);
    markers.push(file);
    return file;
  };

  /**
   * Node flags that make every `@modelcontextprotocol/*` import fail as a missing package, through
   * a resolve hook that a `data:` URL registers.
   */
  const withoutSdk = (): string[] => {
    const hook =
      'export async function resolve(specifier, context, next) { if (specifier.startsWith("@modelcontextprotocol/"))' +
      ' throw Object.assign(new Error("Cannot find package " + specifier), { code: "ERR_MODULE_NOT_FOUND" });' +
      " return next(specifier, context); }";
    const register = `import { register } from "node:module"; register(${JSON.stringify(
      `data:text/javascript,${encodeURIComponent(hook)}`,
    )});`;
    return ["--import", `data:text/javascript,${encodeURIComponent(register)}`];
  };

  afterAll(() => {
    killMarkers(markers);
    rmSync(dir, { recursive: true, force: true });
    expect(strayProcesses()).toEqual([]);
  });

  test.skipIf(!NODE)(
    "the built fixture over stdio: echo, a lossless big integer, and the grandchild ended at close",
    () => {
      const grandchild = marker("grandchild");
      const script = `
        import assert from "node:assert/strict";
        import { mcpServer, ToolSet } from ${builtUrl("dist/src/tools/index.js")};
        import { childAlive, pids, stdioOptions, waitFor } from ${builtUrl("dist/tests/helpers/mcp_helpers.js")};
        const file = ${JSON.stringify(grandchild)};
        const server = mcpServer({ ...stdioOptions("--grandchild", file), label: "fake" });
        try {
          await server.open();
          const tools = new ToolSet({ fake: [...server.tools] });
          assert.equal(await tools.invoke("echo", { text: "hi" }), "hi");
          assert.deepEqual(await tools.invoke("big", {}), { n: 18446744073709552000n });
          assert.ok(await waitFor(() => pids(file).length === 1, 5000), "no grandchild");
          const [sleeper] = pids(file);
          assert.ok(childAlive(sleeper));
          await server.close();
          assert.ok(!childAlive(sleeper), "the grandchild outlived close");
        } finally {
          await server.close();
        }
        console.log("ok");
      `;
      const node = runNode(script);
      expect(node.stderr).toBe("");
      expect(node.stdout).toBe("ok\n");
      expect(node.status).toBe(0);
    },
    ON_NODE,
  );

  test.skipIf(!NODE)(
    "the built HTTP fixture behind a redirect fails to open with RedirectError, the target untouched",
    () => {
      const script = `
        import assert from "node:assert/strict";
        import { McpConnectError, mcpServer } from ${builtUrl("dist/src/tools/index.js")};
        import { RedirectError } from ${builtUrl("dist/src/platform.js")};
        import { startHttpServer, startRedirectServer } from ${builtUrl("dist/tests/fixtures/mcp/fake_server.js")};
        const real = await startHttpServer();
        const redirect = await startRedirectServer(real.url);
        const summary = {};
        try {
          for (const mode of ["auto", "legacy"]) {
            const server = mcpServer({ url: redirect.url, label: "moved" });
            server.mode = mode;
            try {
              await assert.rejects(server.open(), (error) => {
                assert.ok(error instanceof McpConnectError);
                assert.equal(error.message, "MCP server 'moved' didn't start: RedirectError (HTTP 307)");
                assert.ok(error.cause instanceof RedirectError);
                return true;
              });
              assert.equal(server.isOpen, false);
            } finally {
              await server.close();
            }
          }
          summary.hits = redirect.hits;
          summary.targetRequests = real.recordedHeaders.length;
        } finally {
          await redirect.stop();
          await real.stop();
        }
        console.log(JSON.stringify(summary));
      `;
      const node = runNode(script);
      expect(node.stderr).toBe("");
      expect(node.status).toBe(0);
      const summary = JSON.parse(node.stdout) as { hits: number; targetRequests: number };
      expect(summary.hits).toBeGreaterThan(0);
      expect(summary.targetRequests).toBe(0);
    },
    ON_NODE,
  );

  test.skipIf(!NODE)(
    "yian/tools imports without the sdk",
    () => {
      const script = `
        import assert from "node:assert/strict";
        const tools = await import(${builtUrl("dist/src/tools/index.js")});
        assert.equal(tools.McpServer.name, "McpServer");
        await assert.rejects(tools.mcpServer({ command: "x" }).open(), {
          message: "MCP servers need the MCP SDK; install @modelcontextprotocol/client 2.3.1",
        });
        console.log("ok");
      `;
      const node = runNode(script, withoutSdk());
      expect(node.stderr).toBe("");
      expect(node.stdout).toBe("ok\n");
      expect(node.status).toBe(0);
    },
    ON_NODE,
  );

  test.skipIf(!NODE)(
    "open without the sdk names the package to install",
    () => {
      const script = `
        import assert from "node:assert/strict";
        const { mcpServer } = await import(${builtUrl("dist/src/tools/mcp.js")});
        await assert.rejects(mcpServer({ command: "x" }).open(), (error) => {
          assert.equal(error.message, "MCP servers need the MCP SDK; install @modelcontextprotocol/client 2.3.1");
          assert.equal(error.name, "McpSdkMissingError");
          assert.equal(error.cause.code, "ERR_MODULE_NOT_FOUND");
          assert.match(error.cause.message, /@modelcontextprotocol\\/client/);
          return true;
        });
        console.log("ok");
      `;
      const node = runNode(script, withoutSdk());
      expect(node.stderr).toBe("");
      expect(node.stdout).toBe("ok\n");
      expect(node.status).toBe(0);
    },
    ON_NODE,
  );
});
