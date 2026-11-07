/**
 * `runFlow` opens the MCP servers passed to it unopened and closes them when the run ends, over
 * the fake server in `tests/fixtures/mcp/`.
 */

import { afterAll, afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { NodeStarted, type RunEvent } from "../../src/events.ts";
import { Message, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { runFlow, type RunFlowOptions } from "../../src/run.ts";
import { isRunLogLocked } from "../../src/runtime/log.ts";
import type { Skill } from "../../src/skills/catalog.ts";
import { McpConnectError, McpServer, mcpServer } from "../../src/tools/mcp.ts";
import type { ToolEntry } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { startHttpServer } from "../fixtures/mcp/fake_server.ts";
import { expectClosed, hostEnv, killMarkers, pids, stdioOptions, strayProcesses, waitFor } from "../helpers/mcp_helpers.ts";

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
  "a tool node calls an unopened server which the run closes",
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
  "an agent gets the server tools and calls one",
  async () => {
    const model = callThenAnswer("fake_add", { a: 2, b: 3 });
    const server = fake();
    const result = await run(await agentFlow("fake"), { fake: server }, { models: serving(model) });
    expect(result.failure).toBeNull();
    expect(result.status).toBe("succeeded");
    const bound = new Map(model.requests[0]!.tools.map((spec) => [spec.name, spec]));
    for (const name of ["fake_echo", "fake_add", "fake_fail"]) expect(bound.has(name)).toBe(true);
    expect(Object.keys(bound.get("fake_add")!.parameters["properties"] as JsonObject).sort()).toEqual(["a", "b"]);
    // `add` returns structured content, which the call's result is.
    expect(toolResults(model, 1).map((part) => part.content)).toEqual(['{"result":5}']);
    await expectClosed(server);
  },
  SLOW,
);

test(
  "an agent calls an unopened http server",
  async () => {
    const model = callThenAnswer("web_echo", { text: "over http" });
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
  "an agent reads a tool error and carries on",
  async () => {
    const model = callThenAnswer("fake_fail", { message: "boom" });
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
  "a failed run start check closes the server",
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

const stopped = new Error("stopped");
const thrown = new Error("onEvent threw");

for (const [name, toolId, inputs, outcome] of [
  ["succeeds", "echo", '{text: "hi"}', "succeeded"],
  ["fails", "bad_request", "", "failed"],
  ["is stopped", "slow", "{seconds: 5}", stopped],
  ["rejects for a throwing onEvent", "echo", '{text: "hi"}', thrown],
] as const) {
  test(
    `the run's servers close before its log is released when the run ${name}`,
    async () => {
      const file = marker();
      const log = join(dir, "run.jsonl");
      const server = fake("--marker", file);
      const locked: boolean[] = [];
      const close = McpServer.prototype.close;
      // A delivery or an `advance` waits until the log is released, so the server is closed by then.
      const spy = spyOn(McpServer.prototype, "close").mockImplementation(function (this: McpServer) {
        locked.push(isRunLogLocked(log));
        return close.call(this);
      });
      const stop = new AbortController();
      const onEvent = (event: RunEvent) => {
        if (outcome === thrown) throw thrown;
        if (outcome === stopped && event instanceof NodeStarted && event.nodeId === "call") stop.abort(stopped);
      };
      try {
        const running = run(await toolFlow(toolId, inputs), { fake: server }, { log, onEvent, signal: stop.signal });
        if (typeof outcome === "string") expect((await running).status).toBe(outcome);
        else expect(await rejection(running)).toBe(outcome);
      } finally {
        spy.mockRestore();
      }
      expect(locked).toEqual([true]);
      await expectClosed(server, file);
    },
    SLOW,
  );
}

test(
  "a host opened server stays open across runs",
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
  "one server under two keys starts once",
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
  "a server another run opened is rejected",
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
  "two runs racing for one unopened server start it once",
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

test(
  "a server from another copy of yian that is still opening is rejected as opened by another run",
  async () => {
    // A second instance of the module stands in for another installed copy of yian.
    const specifier: string = "../../src/tools/mcp.ts?copy=2";
    const copy = (await import(specifier)) as typeof import("../../src/tools/mcp.ts");
    expect(copy.McpServer).not.toBe(McpServer);
    const server = copy.mcpServer(stdioOptions());
    const opening = server.open();
    try {
      const error = await rejection(run(await toolFlow("echo", '{text: "hi"}'), { shared: server }));
      expect(error).toBeInstanceOf(RangeError);
      expect(error.message).toBe(otherRun("shared"));
      expect((error.cause as Error).name).toBe("AlreadyOpenError");
    } finally {
      await opening.catch(() => undefined);
      await server.close();
    }
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
    `a server under a bad key fails before it starts (${JSON.stringify(key)})`,
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
  "a check that needs no tools fails before any server starts",
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
  "a server that fails to open closes the ones already open",
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
  "a bad entry fails before any server starts",
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
  "a failing node still closes the server",
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
  "a stop before the servers open starts none",
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
  "a stop while the servers open closes the opened one and opens no more",
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
