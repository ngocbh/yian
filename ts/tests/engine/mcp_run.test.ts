/**
 * `runFlow` opens the MCP servers passed to it unopened and closes them when the run ends: a port
 * of `tests/engine/test_mcp_run.py`, over the fake server in `tests/fixtures/mcp/`.
 *
 * Ported: `test_a_tool_node_calls_an_unopened_server_which_the_run_closes`,
 * `test_an_agent_gets_the_server_tools_and_calls_one`, `test_an_agent_calls_an_unopened_http_server`,
 * `test_an_agent_reads_a_tool_error_and_carries_on`, `test_a_failed_run_start_check_closes_the_server`,
 * `test_a_host_opened_server_stays_open_across_runs`, `test_one_server_under_two_keys_starts_once`,
 * `test_a_server_another_run_opened_is_rejected`,
 * `test_two_runs_racing_for_one_unopened_server_start_it_once`,
 * `test_a_server_under_a_bad_key_fails_before_it_starts` (both cases),
 * `test_a_check_that_needs_no_tools_fails_before_any_server_starts`,
 * `test_a_server_that_fails_to_open_closes_the_ones_already_open`,
 * `test_a_bad_entry_fails_before_any_server_starts`, `test_a_failing_node_still_closes_the_server`,
 * `test_a_stop_before_the_servers_open_starts_none` and
 * `test_a_stop_while_the_servers_open_closes_the_opened_one_and_opens_no_more`.
 *
 * Changed: an AGENT's model is a `ScriptedModel` from `models`, and every run sees only the test's
 * credential store in its environment. The bound tools are the first request's `tools`, and the
 * model reads a call's result as a `ToolResultPart`; `add`'s structured result reads as JSON text
 * (Python: its `repr`). The other-run error is a `RangeError` whose message quotes the key as JSON
 * and names `await server.open()` (Python: a `ValueError`, `with server:`). Concurrent runs are
 * concurrent promises (Python: threads). The non-string key is the empty key, the TS key
 * `ToolSet` refuses; the bad group entry is a plain function. A stop is the run's `signal`: an
 * already-aborted one rejects with its reason before anything starts, and the stop during opening
 * aborts it from a patched `McpServer.prototype.openForRun` (Python: `_HostStop` from
 * `_run_flow_controlled`).
 */

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { runFlow, type RunFlowOptions } from "../../src/run.ts";
import type { Skill } from "../../src/skills/catalog.ts";
import { McpConnectError, McpServer, mcpServer } from "../../src/tools/mcp.ts";
import type { ToolEntry } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { startHttpServer } from "../fixtures/mcp/fake_server.ts";
import { expectClosed, hostEnv, killMarkers, pids, stdioOptions, strayProcesses, waitFor } from "./mcp_helpers.ts";

// Each row starts at least one server process; the default 5 s is too tight on a loaded machine.
const SLOW = 30_000;
const markers: string[] = [];
let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-mcp-run-"));
});

afterEach(() => {
  killMarkers(markers.splice(0));
  rmSync(dir, { recursive: true, force: true });
});

afterAll(() => {
  expect(strayProcesses()).toEqual([]);
});

/** A fresh marker file the fake server appends its pid to; `afterEach` kills what it lists. */
function marker(name = "pid"): string {
  const file = join(dir, name);
  markers.push(file);
  return file;
}

/** An unopened fake server over stdio with `flags`. */
function fake(...flags: string[]): McpServer {
  return mcpServer(stdioOptions(...flags));
}

/** `runFlow` in `hostEnv(dir)`, so no host variable or login decides a row. */
function run(flow: ComposedFlow, tools: Readonly<Record<string, ToolEntry>>, options: RunFlowOptions = {}) {
  return runFlow(flow, {}, { env: hostEnv(dir), tools, ...options });
}

/** A flow whose one TOOL node `call` calls `toolId` with the YAML mapping `inputs`. */
function toolFlow(toolId: string, inputs = ""): Promise<ComposedFlow> {
  const line = inputs ? `    input: ${inputs}\n` : "";
  return loadFlow(
    `id: mcp_tool\nname: mcp_tool\nnodes:\n  call:\n    kind: tool\n    tool_id: ${toolId}\n` +
      `${line}output: \${call.output}\n`,
  );
}

/** A flow whose AGENT `worker` is granted `grant`. */
function agentFlow(grant: string): Promise<ComposedFlow> {
  return loadFlow(`
id: mcp_agent
name: mcp_agent
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [${grant}]
output: \${worker.output}
`);
}

/** One tool call, then a final answer. */
function callThenAnswer(name: string, args: JsonObject): ScriptedModel {
  const call = new ToolCallPart({ id: "c0", name, arguments: args });
  return new ScriptedModel([
    new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" }),
    new Response({ message: new Message({ role: "assistant", parts: [new TextPart("done")] }), finishReason: "stop" }),
  ]);
}

/** A model access that serves `model` for every agent. */
function serving(model: ScriptedModel): ModelAccess {
  return { env: hostEnv(dir), createModel: () => model };
}

/** The tool results the model read in request `index`. */
function toolResults(model: ScriptedModel, index: number): ToolResultPart[] {
  return model.requests[index]!.messages.flatMap((message) =>
    message.role === "tool" ? message.parts.filter((part) => part instanceof ToolResultPart) : [],
  );
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

/** The other-run error's message for `key`. */
function otherRun(key: string): string {
  return (
    `tools[${JSON.stringify(key)}]: this MCP server was opened by another run; ` +
    'open it with "await server.open()" to share it between runs'
  );
}

test(
  "test_a_tool_node_calls_an_unopened_server_which_the_run_closes",
  async () => {
    const file = marker();
    const server = fake("--marker", file);
    const result = await run(await toolFlow("echo", '{text: "hi"}'), { fake: server });
    expect(result.failure).toBeNull();
    expect(result.status).toBe("succeeded");
    expect(result.output).toBe("hi");
    await expectClosed(server, file);
  },
  SLOW,
);

test(
  "test_an_agent_gets_the_server_tools_and_calls_one",
  async () => {
    const model = callThenAnswer("add", { a: 2, b: 3 });
    const server = fake();
    const result = await run(await agentFlow("fake"), { fake: server }, { models: serving(model) });
    expect(result.failure).toBeNull();
    expect(result.status).toBe("succeeded");
    const bound = new Map(model.requests[0]!.tools.map((spec) => [spec.name, spec]));
    for (const name of ["echo", "add", "fail"]) expect(bound.has(name)).toBe(true);
    expect(Object.keys(bound.get("add")!.parameters["properties"] as JsonObject).sort()).toEqual(["a", "b"]);
    // `add` returns structured content, which the call's result is.
    expect(toolResults(model, 1).map((part) => part.content)).toEqual(['{"result":5}']);
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_an_agent_calls_an_unopened_http_server",
  async () => {
    const model = callThenAnswer("echo", { text: "over http" });
    const http = await startHttpServer();
    let server: McpServer;
    try {
      server = mcpServer({ url: http.url });
      const result = await run(await agentFlow("web"), { web: server }, { models: serving(model) });
      expect(result.failure).toBeNull();
      expect(result.status).toBe("succeeded");
    } finally {
      await http.stop();
    }
    expect(toolResults(model, 1).map((part) => part.content)).toEqual(["over http"]);
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_an_agent_reads_a_tool_error_and_carries_on",
  async () => {
    const model = callThenAnswer("fail", { message: "boom" });
    const server = fake();
    const result = await run(await agentFlow("fake"), { fake: server }, { models: serving(model) });
    expect(result.failure).toBeNull();
    expect(result.output).toBe("done");
    const [reply] = toolResults(model, 1);
    expect(reply!.content).toContain("boom");
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_a_failed_run_start_check_closes_the_server",
  async () => {
    const file = marker();
    const log = join(dir, "run.jsonl");
    const server = fake("--marker", file);
    const error = await rejection(run(await toolFlow("add", '{a: "x", b: 1}'), { fake: server }, { log }));
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toStartWith("TOOL node 'call' input 'a': tool 'add' takes int");
    expect(existsSync(log)).toBe(false);
    await expectClosed(server, file);
  },
  SLOW,
);

test(
  "test_a_host_opened_server_stays_open_across_runs",
  async () => {
    const flow = await toolFlow("echo", '{text: "hi"}');
    const server = fake();
    {
      const opened = await server.open();
      try {
        for (let i = 0; i < 2; i++) {
          const result = await run(flow, { fake: opened });
          expect(result.status).toBe("succeeded");
          expect(opened.isOpen).toBe(true);
        }
      } finally {
        await opened[Symbol.asyncDispose]();
      }
    }
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_one_server_under_two_keys_starts_once",
  async () => {
    const file = marker();
    const server = fake("--marker", file);
    const result = await run(await toolFlow("echo", '{text: "hi"}'), { a: server, b: server });
    expect(result.failure).toBeNull();
    expect(result.status).toBe("succeeded");
    expect(pids(file)).toHaveLength(1);
    await expectClosed(server, file);
  },
  SLOW,
);

test(
  "test_a_server_another_run_opened_is_rejected",
  async () => {
    const server = fake();
    const flow = await toolFlow("slow", "{seconds: 2}");
    const first = run(flow, { fake: server });
    let error: Error;
    try {
      expect(await waitFor(() => server.isOpen, 20_000)).toBe(true);
      error = await rejection(run(flow, { shared: server }));
    } finally {
      await first.catch(() => undefined);
    }
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toBe(otherRun("shared"));
    expect((await first).status).toBe("succeeded");
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_two_runs_racing_for_one_unopened_server_start_it_once",
  async () => {
    const file = marker();
    const server = fake("--marker", file);
    // The second run finds the server already being opened (`openForRun` claims it before its first
    // await); the slow tool keeps the winner's run alive meanwhile.
    const flow = await toolFlow("slow", "{seconds: 5}");
    const outcomes = await Promise.allSettled([run(flow, { a: server }), run(flow, { b: server })]);
    const results = outcomes.filter((outcome) => outcome.status === "fulfilled");
    const errors = outcomes.flatMap((outcome, i) => (outcome.status === "rejected" ? [[i, outcome.reason]] : []));
    expect(results).toHaveLength(1);
    expect(errors).toHaveLength(1);
    expect(results[0]!.value.status).toBe("succeeded");
    const [loser, loss] = errors[0]! as [number, Error];
    expect(loss).toBeInstanceOf(RangeError);
    expect(loss.message).toBe(otherRun(loser === 0 ? "a" : "b"));
    expect(pids(file)).toHaveLength(1);
    await expectClosed(server, file);
  },
  SLOW,
);

for (const [key, kind, message] of [
  ["ask_user", TypeError, 'tools["ask_user"] must be a handler function'],
  ["ask", TypeError, 'tools["ask"] must be a handler function'],
  ["take", RangeError, 'tools["take"]: the name is reserved by yian'],
  ["sleep", RangeError, 'tools["sleep"]: the name is reserved by yian'],
  ["load_skill", RangeError, 'tools["load_skill"]: the name is reserved by yian'],
  ["", TypeError, "tool ids in tools must be non-empty strings"],
] as const) {
  test(
    `test_a_server_under_a_bad_key_fails_before_it_starts (${JSON.stringify(key)})`,
    async () => {
      const file = marker();
      const server = fake("--marker", file);
      const error = await rejection(run(await toolFlow("echo", '{text: "hi"}'), { [key]: server }));
      expect(error).toBeInstanceOf(kind);
      expect(error.message).toContain(message);
      expect(existsSync(file)).toBe(false);
      await expectClosed(server);
    },
    SLOW,
  );
}

test(
  "test_a_check_that_needs_no_tools_fails_before_any_server_starts",
  async () => {
    const file = marker();
    const server = fake("--marker", file);
    const skills = new Map([["x", {} as Skill]]);
    const error = await rejection(run(await toolFlow("echo", '{text: "hi"}'), { fake: server }, { skills }));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toContain("skills");
    expect(existsSync(file)).toBe(false);
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_a_server_that_fails_to_open_closes_the_ones_already_open",
  async () => {
    const file = marker();
    const log = join(dir, "run.jsonl");
    const ok = fake("--marker", file);
    const bad = mcpServer({ command: "/nonexistent/x" });
    const error = await rejection(run(await toolFlow("echo", '{text: "hi"}'), { ok, bad }, { log }));
    expect(error).toBeInstanceOf(McpConnectError);
    expect(existsSync(log)).toBe(false);
    expect(pids(file)).toHaveLength(1);
    await expectClosed(ok, file);
    expect(bad.isOpen).toBe(false);
  },
  SLOW,
);

test(
  "test_a_bad_entry_fails_before_any_server_starts",
  async () => {
    const file = marker();
    const server = fake("--marker", file);
    const tools = { fake: server, g: [(text: string) => text] } as unknown as Record<string, ToolEntry>;
    const error = await rejection(run(await toolFlow("echo", '{text: "hi"}'), tools));
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toContain('tools["g"][0]');
    expect(existsSync(file)).toBe(false);
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_a_failing_node_still_closes_the_server",
  async () => {
    const file = marker();
    const server = fake("--marker", file);
    const result = await run(await toolFlow("bad_request"), { fake: server });
    expect(result.status).toBe("failed");
    await expectClosed(server, file);
  },
  SLOW,
);

test(
  "test_a_stop_before_the_servers_open_starts_none",
  async () => {
    const files = [marker("a.pid"), marker("b.pid")];
    const servers = files.map((file) => fake("--marker", file));
    const stop = new AbortController();
    stop.abort(new Error("stopped"));
    const error = await rejection(
      run(await toolFlow("echo", '{text: "hi"}'), { a: servers[0]!, b: servers[1]! }, { signal: stop.signal }),
    );
    expect(error).toBe(stop.signal.reason);
    expect(files.some((file) => existsSync(file))).toBe(false);
    for (const server of servers) await expectClosed(server);
  },
  SLOW,
);

test(
  "test_a_stop_while_the_servers_open_closes_the_opened_one_and_opens_no_more",
  async () => {
    const files = [marker("a.pid"), marker("b.pid")];
    const [first, second] = files.map((file) => fake("--marker", file)) as [McpServer, McpServer];
    const stop = new AbortController();
    const openForRun = McpServer.prototype.openForRun;
    // Open the server, then abort as if Ctrl-C came during the open.
    McpServer.prototype.openForRun = async function (this: McpServer, token, options) {
      const opened = await openForRun.call(this, token, options);
      stop.abort(new Error("stopped"));
      return opened;
    };
    let error: Error;
    try {
      error = await rejection(
        run(await toolFlow("echo", '{text: "hi"}'), { a: first, b: second }, { signal: stop.signal }),
      );
    } finally {
      McpServer.prototype.openForRun = openForRun;
    }
    expect(error).toBe(stop.signal.reason);
    expect(pids(files[0]!)).toHaveLength(1);
    expect(existsSync(files[1]!)).toBe(false);
    await expectClosed(first, files[0]);
    await expectClosed(second);
  },
  SLOW,
);
