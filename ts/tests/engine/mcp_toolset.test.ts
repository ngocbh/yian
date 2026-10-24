/**
 * `ToolSet` takes an open MCP server as a tool group, a port of `tests/engine/test_mcp_toolset.py`.
 * The construction, `expand` and invoke rows run against a stand-in open server;
 * `test_a_dead_server_is_rejected` runs against the stand-in and against the real fixture
 * (`tests/fixtures/mcp/fake_server.ts` over stdio). The import-without-the-SDK row is among the
 * Node rows of `mcp_server.test.ts`; the exports row is the `tools/index.ts` exports row in
 * `tool_errors.test.ts`.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, describe, expect, test } from "bun:test";

import { FunctionTool } from "../../src/tools/function_tools.ts";
import { McpServer, McpServerClosed, type McpServerSpec, mcpServer } from "../../src/tools/mcp.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { killMarkers, stdioOptions, strayProcesses } from "./mcp_helpers.ts";

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
  test("test_an_open_server_is_a_group_of_its_tools", async () => {
    const tools = new ToolSet({ fake: server });
    expect(tools.groups).toEqual(["fake"]);
    expect(tools.expand(["fake"])).toEqual(server.tools.map((tool) => tool.name));
    expect(tools.resolve(["echo"])).toEqual([server.tools[0]!]);
    expect(await tools.invoke("echo", { text: "hi" })).toBe("hi");
  });

  test("test_an_unopened_server_is_rejected", () => {
    const unopened = mcpServer({ command: "fake" });
    expect(thrown(() => new ToolSet({ fake: unopened }), TypeError)).toBe(
      `tools["fake"] is an MCP server that isn't open; open it with "await server.open()" or pass it to runFlow/advance, which open it`,
    );
  });

  test("test_a_server_inside_a_list_group_is_rejected", () => {
    expect(thrown(() => new ToolSet({ g: [server as unknown as FunctionTool] }), TypeError)).toBe(
      'tools["g"][0] is an MCP server; pass it as its own tools entry, which becomes its group',
    );
  });

  test("test_two_servers_with_one_tool_name_clash", () => {
    const first = new FakeServer("a", [serverTool("echo")]);
    const second = new FakeServer("b", [serverTool("echo")]);
    expect(thrown(() => new ToolSet({ a: first, b: second }), RangeError)).toBe(
      "tool 'echo' is in both groups 'a' and 'b' as different tools",
    );
  });

  test("test_a_server_tool_named_like_a_top_level_key_clashes", () => {
    expect(thrown(() => new ToolSet({ echo: serverTool("echo"), fake: server }), RangeError)).toBe(
      `tool 'echo' in group 'fake' clashes with tools["echo"]`,
    );
  });

  test("test_one_server_under_two_keys_lists_each_tool_once", () => {
    const tools = new ToolSet({ a: server, b: server });
    expect(tools.groups).toEqual(["a", "b"]);
    expect(tools.expand(["a", "b"])).toEqual(server.tools.map((tool) => tool.name));
  });

  test("test_a_renamed_tool_in_a_list_group_calls_the_servers_tool", async () => {
    const echo = server.tools.find((tool) => tool.name === "echo")!;
    const renamed = new FunctionTool({
      name: "other",
      description: echo.description,
      parametersJsonSchema: echo.parametersJsonSchema,
      execute: echo.execute,
    });
    const tools = new ToolSet({ renamed: [renamed] });
    expect(tools.expand(["renamed"])).toEqual(["other"]);
    expect(tools.resolve(["other"])[0]!.execute).toBe(echo.execute);
    expect(await tools.invoke("other", { text: "hi" })).toBe("hi");
  });

  test("a server listing a non-FunctionTool is rejected", () => {
    const odd = new FakeServer("odd", [serverTool("echo"), { name: "add" } as unknown as FunctionTool]);
    expect(thrown(() => new ToolSet({ odd }), TypeError)).toBe('tools["odd"][1] must be a FunctionTool, got Object');
  });

  test("test_a_dead_server_is_rejected", () => {
    const quitter = new FakeServer("quitter", [serverTool("exit_now")], { dead: true });
    expect(thrown(() => new ToolSet({ fake: quitter }), RangeError)).toBe(
      `tools["fake"]: MCP server 'quitter' is closed`,
    );
  });

  test(
    "test_a_dead_server_is_rejected with a real server",
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
