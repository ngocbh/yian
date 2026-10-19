/**
 * `McpServer` tools call the server and map its results to yian values, a port of
 * `tests/engine/test_mcp_calls.py` over stdio: `test_echo_returns_the_text`,
 * `test_structured_content_is_the_value`, `test_the_call_meta_carries_the_call_id_and_attempt`,
 * `test_media_blocks_become_one_line_markers`, `test_bad_arguments_fail_before_the_server_is_called`,
 * `test_nested_arguments_reach_the_server`, `test_an_error_result_raises_a_reported_mcp_tool_error`,
 * `test_calls_from_many_threads_run_at_once` (as concurrent calls),
 * `test_result_value_renders_each_kind_of_content` and
 * `test_result_value_prefers_structured_content_and_renders_empty_content_as_empty`; and
 * `test_structured_content_failing_the_output_schema_is_a_tool_error` from `test_mcp_errors.py`.
 * Python runs each row in the `auto`, `legacy` and `http` modes; stdio connects only in `legacy`
 * here, and the `http` rows land with the HTTP transport. `test_a_tool_node_calls_a_host_opened_server`
 * is run-level and lands with the runs that open servers.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import { ToolContext } from "../../src/tools/context.ts";
import { isPermanentToolError, isReportedToolError, ToolCheckError } from "../../src/tools/errors.ts";
import { isMcpError, type McpServer, mcpServer, McpToolError, resultValue } from "../../src/tools/mcp.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { TypeCheckError } from "../../src/typesys/values.ts";
import { PNG_BASE64 } from "../fixtures/mcp/fake_server.ts";
import { killMarkers, stdioOptions, strayProcesses } from "./mcp_helpers.ts";

// Opening a server spawns a process; the default 5 s is too tight on a loaded machine.
const SLOW = 15_000;
const dir = mkdtempSync(join(tmpdir(), "yian-mcp-calls-"));
const markers: string[] = [];

/** Open the fake server with `flags` and a fresh marker file, which `afterAll` kills by pid. */
async function openFake(flags: string[], label?: string): Promise<McpServer> {
  const file = join(dir, `marker-${markers.length + 1}`);
  markers.push(file);
  return await mcpServer({ ...stdioOptions(...flags, "--marker", file), label }).open();
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

/** A context for calling a tool's `execute` directly. */
function context(toolName: string): ToolContext {
  return new ToolContext({
    callId: "c-1",
    attempt: 1,
    toolName,
    runId: "",
    node: "",
    context: undefined,
    signal: new AbortController().signal,
  });
}

afterAll(() => {
  killMarkers(markers);
  rmSync(dir, { recursive: true, force: true });
  expect(strayProcesses()).toEqual([]);
});

describe("calls to the fake server", () => {
  let server: McpServer;
  const tools = () => new ToolSet({ fake: [...server.tools] });

  beforeAll(async () => {
    server = await openFake([]);
  }, SLOW);

  afterAll(async () => {
    await server?.close();
  }, SLOW);

  test("test_echo_returns_the_text", async () => {
    expect(await tools().invoke("echo", { text: "hi" })).toBe("hi");
  });

  test("test_structured_content_is_the_value", async () => {
    const value = await tools().invoke("add", { a: 1, b: 2 });
    expect(value).toEqual({ result: 3 });
    expect(typeof (value as { result: unknown }).result).toBe("number");
  });

  test("test_the_call_meta_carries_the_call_id_and_attempt", async () => {
    const bound = tools().forRun("run-1");
    expect(await bound.invoke("whoami", {}, { node: "n", attempt: 2 })).toBe("run-1/n 2");
  });

  test("test_media_blocks_become_one_line_markers", async () => {
    const size = Buffer.from(PNG_BASE64, "base64").length;
    expect(await tools().invoke("picture", {})).toBe(`[image image/png, ${size} B]\n\ncaption`);
    expect(await tools().invoke("resource", {})).toBe("note body\n\n[resource file:///a.png image/png]");
  });

  test("test_bad_arguments_fail_before_the_server_is_called", async () => {
    const before = await tools().invoke("calls", {});
    expect(await rejection(tools().invoke("add", { a: "x", b: 1 }))).toBeInstanceOf(TypeCheckError);
    expect(await tools().invoke("calls", {})).toEqual(before as Record<string, unknown>);
  });

  test("test_nested_arguments_reach_the_server", async () => {
    const item = { name: "p", point: { x: 1, y: 2 } };
    expect(await tools().invoke("nested", { item })).toBe('{"name":"p","point":{"x":1,"y":2}}');
  });

  test("test_an_error_result_raises_a_reported_mcp_tool_error", async () => {
    const error = await rejection(tools().invoke("fail", { message: "no luck" }));
    expect(error).toBeInstanceOf(McpToolError);
    expect(error.message).toContain("no luck");
    expect(isMcpError(error)).toBe(true);
    expect(isReportedToolError(error)).toBe(true);
    expect(isPermanentToolError(error)).toBe(false);
  });

  test(
    "test_calls_from_many_threads_run_at_once",
    async () => {
      const set = tools();
      const results = await Promise.all(Array.from({ length: 8 }, () => set.invoke("gate", { seconds: 0.3 })));
      expect(results).toEqual(Array(8).fill("done"));
      expect(((await set.invoke("peak", {})) as { result: number }).result).toBeGreaterThan(1);
    },
    SLOW,
  );

  test("an integer past 2^53 in structured content stays exact", async () => {
    const big = server.tools.find((tool) => tool.name === "big")!;
    expect(await big.execute({}, context("big"))).toEqual({ n: 18446744073709552000n });
    const value = (await tools().invoke("big", {})) as { n: unknown };
    expect(typeof value.n).toBe("bigint");
    expect(value.n).toBe(18446744073709552000n);
  });

  test("Infinity in the arguments is refused before the server is called", async () => {
    const before = await tools().invoke("calls", {});
    const error = await rejection(tools().invoke("slow", { seconds: Infinity }));
    expect(error).toBeInstanceOf(ToolCheckError);
    expect(error.message).toBe("tool call 'slow' arguments at seconds: Infinity can't be sent to an MCP server");
    const negative = await rejection(tools().invoke("gate", { seconds: -Infinity }));
    expect(negative.message).toBe("tool call 'gate' arguments at seconds: Infinity can't be sent to an MCP server");
    expect(await tools().invoke("calls", {})).toEqual(before as Record<string, unknown>);
  });
});

describe("tools that break their output schema", () => {
  let server: McpServer;
  const tools = () => new ToolSet({ fake: [...server.tools] });

  beforeAll(async () => {
    server = await openFake(["--bad-output"], "bad");
  }, SLOW);

  afterAll(async () => {
    await server?.close();
  }, SLOW);

  test("test_structured_content_failing_the_output_schema_is_a_tool_error", async () => {
    const error = await rejection(tools().invoke("bad_output", {}));
    expect(error).toBeInstanceOf(McpToolError);
    expect(error.message).toBe("MCP server 'bad' tool 'bad_output' failed (TypeCheckError)");
    expect(error.cause).toBeInstanceOf(TypeCheckError);
    expect(isReportedToolError(error)).toBe(true);
    expect(isPermanentToolError(error)).toBe(false);
  });

  test("a result without structured content fails, also after the tool list changed", async () => {
    const message = "MCP server 'bad' tool 'no_structured' failed (no structured content)";
    const error = await rejection(tools().invoke("no_structured", {}));
    expect(error).toBeInstanceOf(McpToolError);
    expect(error.message).toBe(message);
    expect(await tools().invoke("add_tool", {})).toBe("added");
    // The notification arrived before the response; this gives the SDK's async cache eviction time
    // to finish, so the next call runs after it.
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect((await rejection(tools().invoke("no_structured", {}))).message).toBe(message);
  });
});

describe("resultValue", () => {
  /** `size` zero bytes in base64. */
  const b64 = (size: number) => Buffer.alloc(size).toString("base64");

  test("test_result_value_renders_each_kind_of_content", () => {
    const result = {
      content: [
        { type: "text", text: "one" },
        { type: "audio", data: b64(1023), mimeType: "audio/wav" },
        { type: "image", data: b64(2048), mimeType: "image/png" },
        {
          type: "resource",
          resource: { uri: "file:///b.bin", blob: b64((3 * 1024 * 1024) / 2), mimeType: "application/octet-stream" },
        },
        { type: "resource", resource: { uri: "file:///c.bin", blob: b64(10) } },
        { type: "resource", resource: { uri: "text://t", text: "two" } },
        { type: "resource_link", name: "d", uri: "file:///d" },
      ],
    };
    expect(resultValue(result)).toBe(
      [
        "one",
        "[audio audio/wav, 1023 B]",
        "[image image/png, 2.0 KB]",
        "[resource file:///b.bin application/octet-stream, 1.5 MB]",
        "[resource file:///c.bin, 10 B]",
        "two",
        "[resource file:///d]",
      ].join("\n\n"),
    );
  });

  test("test_result_value_prefers_structured_content_and_renders_empty_content_as_empty", () => {
    expect(resultValue({ content: [], structuredContent: [1, 2] })).toEqual([1, 2]);
    expect(resultValue({ content: [{ type: "text", text: "x" }], structuredContent: 0 })).toBe(0);
    expect(resultValue({ content: [] })).toBe("");
  });

  test("an error result throws its rendered content", () => {
    const error = (() => {
      try {
        resultValue({ content: [{ type: "text", text: "bad" }], isError: true, structuredContent: { a: 1 } });
      } catch (caught) {
        return caught as Error;
      }
      throw new Error("expected a throw");
    })();
    expect(error).toBeInstanceOf(McpToolError);
    expect(error.message).toBe("bad");
  });

  test("sizes decode base64 as Python does, and round to one decimal half to even", () => {
    const sizeOf = (data: string) => resultValue({ content: [{ type: "image", data, mimeType: "m" }] });
    // Characters outside the alphabet are skipped, and decoding stops where padding ends a quad.
    expect(sizeOf("aGk=\n")).toBe("[image m, 2 B]");
    expect(sizeOf("aGk=aGk=")).toBe("[image m, 2 B]");
    // Malformed input is estimated from its length.
    expect(sizeOf("aGk")).toBe("[image m, 2 B]");
    expect(sizeOf("aGkhaGkha")).toBe("[image m, 6 B]");
    // 1280 bytes are 1.25 KB, which Python's `.1f` rounds to even.
    expect(sizeOf(b64(1280))).toBe("[image m, 1.2 KB]");
    expect(sizeOf(b64(1331))).toBe("[image m, 1.3 KB]");
  });
});
