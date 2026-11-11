/**
 * `ToolSet` takes an open MCP server as a tool group, against a stand-in open server and the real
 * fixture (`tests/fixtures/mcp/fake_server.ts` over stdio).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, describe, expect, test } from "bun:test";

import { FunctionTool } from "../../src/tools/function_tools.ts";
import { McpServer, McpServerClosed, type McpServerSpec, mcpServer } from "../../src/tools/mcp.ts";
import { asToolGrant, expandGrants } from "../../src/tools/grants.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { killMarkers, stdioOptions, strayProcesses } from "../helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-mcp-toolset-"));
const marker = join(dir, "marker");

afterAll(() => {
  killMarkers([marker]);
  rmSync(dir, { recursive: true, force: true });
  expect(strayProcesses()).toEqual([]);
});

/** Build a `FunctionTool` named `name`, as a server would list it. */
function serverTool(name: string): FunctionTool {
  return new FunctionTool({
    name,
    description: `The ${name} tool.`,
    parametersJsonSchema: { type: "object", properties: { text: { type: "string" } } },
    execute: (args) => args.text,
  });
}

/** An MCP server that reports itself open with fixed tools, or open but gone away. */
class FakeServer extends McpServer {
  private readonly listed: readonly FunctionTool[];
  private readonly dead: boolean;

  constructor(label: string, listed: readonly FunctionTool[], options: { dead?: boolean } = {}) {
    const spec: McpServerSpec = {
      command: "fake",
      args: [],
      env: undefined,
      cwd: undefined,
      url: undefined,
      headers: undefined,
      timeout: undefined,
      connectTimeout: undefined,
      label,
      concurrencySafe: false,
    };
    super(spec);
    this.listed = listed;
    this.dead = options.dead ?? false;
  }

  override get isOpen(): boolean {
    return true;
  }

  override get isClosed(): boolean {
    return this.dead;
  }

  override get tools(): readonly FunctionTool[] {
    return this.listed;
  }
}

/** Return the message of the error `fn` throws, after checking its class. */
function thrown(fn: () => unknown, kind: typeof TypeError | typeof RangeError): string {
  try {
    fn();
  } catch (error) {
    expect((error as Error).constructor).toBe(kind);
    return (error as Error).message;
  }
  throw new Error("expected an error");
}

const server = new FakeServer("fake", [serverTool("echo"), serverTool("add")]);

describe("an MCP server in a ToolSet", () => {
  test("an open server is a group of its tools", async () => {
    const tools = new ToolSet({ fake: server });
    expect(tools.groups).toEqual(["fake"]);
    expect(tools.scope().group("fake")).toEqual(server.tools.map((tool) => tool.name));
    expect(tools.resolve(["echo"])).toEqual([server.tools[0]!]);
    expect(await tools.invoke("echo", { text: "hi" })).toBe("hi");
  });

  test("an unopened server is rejected", () => {
    const unopened = mcpServer({ command: "fake" });
    expect(thrown(() => new ToolSet({ fake: unopened }), TypeError)).toBe(
      `tools["fake"] is an MCP server that isn't open; open it with "await server.open()" or pass it to runFlow/advance, which open it`,
    );
  });

  test("a server inside a list group is rejected", () => {
    expect(thrown(() => new ToolSet({ g: [server as unknown as FunctionTool] }), TypeError)).toBe(
      'tools["g"][0] is an MCP server; pass it as its own tools entry, which becomes its group',
    );
  });

  test("two servers with one tool name are each reached through their group", async () => {
    const first = new FakeServer("a", [serverTool("echo")]);
    const second = new FakeServer("b", [serverTool("echo")]);
    const tools = new ToolSet({ a: first, b: second });
    expect(tools.resolve([{ group: "b", name: "echo" }])).toEqual([second.tools[0]!]);
    expect(await tools.invoke({ group: "a", name: "echo" }, { text: "hi" })).toBe("hi");
  });

  test("a server tool named like a top level key clashes", () => {
    expect(thrown(() => new ToolSet({ echo: serverTool("echo"), fake: server }), RangeError)).toBe(
      `tool 'echo' in group 'fake' clashes with tools["echo"]`,
    );
  });

  test("a server named like its own tool: the name reaches the tool, and a grant of it the server", async () => {
    const fetch = new FakeServer("fetch", [serverTool("fetch")]);
    const tools = new ToolSet({ fetch });
    expect(tools.resolve(["fetch"])).toEqual([fetch.tools[0]!]);
    expect(await tools.invoke("fetch", { text: "hi" })).toBe("hi");
    expect(expandGrants([asToolGrant("fetch")], tools.scopes()).map((each) => each.visible)).toEqual(["fetch_fetch"]);
  });

  test("one server under two keys binds each tool once, under the first key", () => {
    const tools = new ToolSet({ a: server, b: server });
    expect(tools.groups).toEqual(["a", "b"]);
    const granted = expandGrants(["a", "b"].map(asToolGrant), tools.scopes());
    expect(granted.map((each) => each.visible)).toEqual(server.tools.map((tool) => `a_${tool.name}`));
  });

  test("a renamed tool in a list group calls the servers tool", async () => {
    const echo = server.tools.find((tool) => tool.name === "echo")!;
    const renamed = new FunctionTool({
      name: "other",
      description: echo.description,
      parametersJsonSchema: echo.parametersJsonSchema,
      execute: echo.execute,
    });
    const tools = new ToolSet({ renamed: [renamed] });
    expect(tools.scope().group("renamed")).toEqual(["other"]);
    expect(tools.resolve(["other"])[0]!.execute).toBe(echo.execute);
    expect(await tools.invoke("other", { text: "hi" })).toBe("hi");
  });

  test("a server listing a non-FunctionTool is rejected", () => {
    const odd = new FakeServer("odd", [serverTool("echo"), { name: "add" } as unknown as FunctionTool]);
    expect(thrown(() => new ToolSet({ odd }), TypeError)).toBe('tools["odd"][1] must be a FunctionTool, got Object');
  });

  test("a dead server is rejected", () => {
    const quitter = new FakeServer("quitter", [serverTool("exit_now")], { dead: true });
    expect(thrown(() => new ToolSet({ fake: quitter }), RangeError)).toBe(
      `tools["fake"]: MCP server 'quitter' is closed`,
    );
  });

  test(
    "a dead server is rejected with a real server",
    async () => {
      const quitter = await mcpServer({ ...stdioOptions("--marker", marker), label: "quitter" }).open();
      try {
        let error: unknown;
        try {
          await new ToolSet({ fake: quitter }).invoke("exit_now", {});
        } catch (caught) {
          error = caught;
        }
        expect(error).toBeInstanceOf(McpServerClosed);
        expect(thrown(() => new ToolSet({ fake: quitter }), RangeError)).toBe(
          `tools["fake"]: MCP server 'quitter' is closed`,
        );
      } finally {
        await quitter[Symbol.asyncDispose]();
      }
    },
    // Opening spawns a process; the default 5 s is too tight on a loaded machine.
    15_000,
  );
});
