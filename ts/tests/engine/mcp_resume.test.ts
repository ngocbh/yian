/**
 * `resumeFlow` opens the MCP servers passed to it unopened only for an unfinished run: a port of
 * `tests/engine/test_mcp_resume.py`, over the fake server in `tests/fixtures/mcp/`.
 *
 * Ported: `test_a_resumed_tool_node_calls_an_unopened_server_which_the_resume_closes`,
 * `test_resuming_a_finished_run_starts_no_server`, `test_a_bad_entry_fails_before_the_log_is_touched`,
 * `test_a_server_that_fails_to_open_on_resume_closes_the_log`,
 * `test_a_check_that_needs_no_tools_fails_before_the_server_starts` and
 * `test_resuming_without_the_runs_memory_fails_before_any_server_starts`.
 *
 * Changed:
 * - A crash is a frozen log: the local `whoami` tool freezes every durable log, whose next append
 *   throws a `Crash` the run rejects with (Python: the tool raises `_SimulatedCrash`).
 * - The bad group entry is an arrow function, refused as `tools["g"][0] must be a FunctionTool`
 *   (Python: "is a lambda"); the bad skills map is a `Map` holding a plain object.
 * - Memory is a `Map` of the shared `FakeMemory`.
 * - `test_a_stop_before_the_servers_open_on_resume_starts_none`: the stop is the `signal`, and an
 *   aborted one rejects with its reason before the log is locked (Python raises `_HostStop` after
 *   replay, from the server opening).
 *
 * Every run sees only the test's credential store in its environment.
 *
 * Covered by a case: none.
 */

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { RunStarted } from "../../src/events.ts";
import { MemorySetupError } from "../../src/memory/contracts.ts";
import { type ResumeFlowOptions, type RunFlowOptions, resumeFlow, runFlow } from "../../src/run.ts";
import { JsonlRunLog, type RunRecord } from "../../src/runtime/log.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { McpConnectError, type McpServer, mcpServer } from "../../src/tools/mcp.ts";
import { expectClosed, hostEnv, killMarkers, stdioOptions, strayProcesses } from "./mcp_helpers.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { Crash, rejected } from "./run_state_fixtures.ts";

const SLOW = 30_000;
const markers: string[] = [];
let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-mcp-resume-"));
});

afterEach(() => {
  killMarkers(markers.splice(0));
  rmSync(dir, { recursive: true, force: true });
});

afterAll(() => {
  expect(strayProcesses()).toEqual([]);
});

const WHOAMI_FLOW = `
id: mcp_resume
name: mcp_resume
nodes:
  call:
    kind: tool
    tool_id: whoami
output: \${call.output}
`;

const MEMORY_FLOW = `
id: mcp_memory
name: mcp_memory
long_term_memory: test
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: \${work.output}
`;

/** `runFlow` in `hostEnv(dir)`. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  return runFlow(flow, inputs, { env: hostEnv(dir), ...options });
}

/** `resumeFlow` in `hostEnv(dir)`. */
function resume(flow: ComposedFlow, options: ResumeFlowOptions) {
  return resumeFlow({ env: hostEnv(dir), ...options });
}

/** An unopened fake server over stdio, its pid appended to a fresh marker file. */
function fake(): [McpServer, string] {
  const file = join(dir, `marker-${markers.length + 1}`);
  markers.push(file);
  return [mcpServer(stdioOptions("--marker", file)), file];
}

/** Crash a run of `WHOAMI_FLOW` inside its TOOL node and return the call's context. */
async function crashedRun(log: string): Promise<ToolContext> {
  const seen: ToolContext[] = [];
  const crash = new Crash("simulated crash");
  let frozen = false;
  const append = JsonlRunLog.prototype.append;
  JsonlRunLog.prototype.append = function (this: JsonlRunLog, record: RunRecord) {
    if (frozen) throw crash;
    return append.call(this, record);
  };
  const whoami = tool({
    name: "whoami",
    description: "Crash the process on the call.",
    parameters: { type: "object", properties: {} },
    execute: (_args, ctx) => {
      seen.push(ctx);
      frozen = true;
      return "never recorded";
    },
  });
  try {
    const error = await rejected(run(await loadFlow(WHOAMI_FLOW), {}, { log, tools: { whoami }, runId: "run-5c2e" }));
    expect(error).toBe(crash);
  } finally {
    JsonlRunLog.prototype.append = append;
  }
  expect(seen).toHaveLength(1);
  return seen[0]!;
}

test(
  "test_a_resumed_tool_node_calls_an_unopened_server_which_the_resume_closes",
  async () => {
    const log = join(dir, "run.jsonl");
    const first = await crashedRun(log);
    const [server, marker] = fake();

    const resumed = await resume(await loadFlow(WHOAMI_FLOW), { log, tools: { fake: server } });

    expect(resumed.failure).toBeNull();
    expect(resumed.status).toBe("succeeded");
    expect(first.callId).toBe("run-5c2e/call");
    expect(resumed.output).toBe(`run-5c2e/call ${first.attempt + 1}`);
    await expectClosed(server, marker);
  },
  SLOW,
);

test(
  "test_resuming_a_finished_run_starts_no_server",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    const whoami = tool({
      name: "whoami",
      description: "Answer without a server.",
      parameters: { type: "object", properties: {} },
      execute: () => "local",
    });
    const finished = await run(flow, {}, { log, tools: { whoami } });
    expect(finished.status).toBe("succeeded");
    const [server, marker] = fake();

    const resumed = await resume(flow, { log, tools: { fake: server } });

    expect(resumed.status).toBe("succeeded");
    expect(resumed.output).toBe("local");
    expect(existsSync(marker)).toBe(false);
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_a_bad_entry_fails_before_the_log_is_touched",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    await crashedRun(log);
    const before = [readFileSync(log), statSync(log, { bigint: true }).mtimeNs];
    const [server, marker] = fake();

    const error = await rejected(
      resume(flow, { log, tools: { fake: server, g: [(text: string) => text] as never } }),
    );

    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toStartWith('tools["g"][0] must be a FunctionTool');
    expect([readFileSync(log), statSync(log, { bigint: true }).mtimeNs]).toEqual(before);
    expect(existsSync(marker)).toBe(false);
    await expectClosed(server);
    const resumed = await resume(flow, { log, tools: { fake: server } });
    expect(resumed.status).toBe("succeeded");
  },
  SLOW,
);

test(
  "test_a_server_that_fails_to_open_on_resume_closes_the_log",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    await crashedRun(log);
    const bad = mcpServer({ command: "/nonexistent/x" });

    expect(await rejected(resume(flow, { log, tools: { fake: bad } }))).toBeInstanceOf(McpConnectError);

    expect(bad.isOpen).toBe(false);
    const [server] = fake();
    const resumed = await resume(flow, { log, tools: { fake: server } });
    expect(resumed.status).toBe("succeeded");
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_a_check_that_needs_no_tools_fails_before_the_server_starts",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    await crashedRun(log);
    const [server, marker] = fake();

    const error = await rejected(resume(flow, { log, tools: { fake: server }, skills: new Map([["x", {}]]) as never }));

    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toContain("skills");
    expect(existsSync(marker)).toBe(false);
    await expectClosed(server);
    const resumed = await resume(flow, { log, tools: { fake: server } });
    expect(resumed.status).toBe("succeeded");
  },
  SLOW,
);

test(
  "test_resuming_without_the_runs_memory_fails_before_any_server_starts",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(MEMORY_FLOW);
    const stop = new Error("host stop");
    const stopAfterStart = (event: unknown) => {
      if (event instanceof RunStarted) throw stop;
    };
    expect(
      await rejected(run(flow, {}, { log, memory: new Map([["test", new FakeMemory()]]), onEvent: stopAfterStart })),
    ).toBe(stop);
    const [server, marker] = fake();

    const error = await rejected(resume(flow, { log, tools: { fake: server } }));

    expect(error).toBeInstanceOf(MemorySetupError);
    expect((error as Error).message).toBe("resuming this run requires its enabled memory mapping");
    expect(existsSync(marker)).toBe(false);
    await expectClosed(server);
  },
  SLOW,
);

test(
  "test_a_stop_before_the_servers_open_on_resume_starts_none",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    await crashedRun(log);
    const [a, markerA] = fake();
    const [b, markerB] = fake();
    const stop = new AbortController();
    stop.abort(new Error("stopped"));

    expect(await rejected(resume(flow, { log, tools: { a, b }, signal: stop.signal }))).toBe(stop.signal.reason);

    expect(existsSync(markerA) || existsSync(markerB)).toBe(false);
    await expectClosed(a);
    await expectClosed(b);
    const [server] = fake();
    const resumed = await resume(flow, { log, tools: { fake: server } });
    expect(resumed.status).toBe("succeeded");
    await expectClosed(server);
  },
  SLOW,
);
