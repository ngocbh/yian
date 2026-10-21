/**
 * Stopping a run ends its waiting MCP calls and leaves nothing running: a port of
 * `tests/engine/test_mcp_stop.py`, over the fake server in `tests/fixtures/mcp/`.
 *
 * Ported, each for 0 and 8 workers: `test_a_stop_ends_a_run_owned_server_call_and_the_log_resumes`
 * (the stop half), `test_ctrl_c_in_the_host_ends_a_run_owned_server_call`,
 * `test_a_stop_cancels_only_its_own_run_on_a_shared_host_server` and
 * `test_a_stop_ends_an_agent_tool_call`.
 *
 * Changed: a stop is the run's `signal`, and the run rejects with its reason (Python:
 * `_HostStop` from `_run_flow_controlled`). Ctrl-C is the host's to map onto the signal, so that
 * row aborts the signal with the reason a host's SIGINT handler would give, and needs no log
 * (Python: a real SIGINT and `KeyboardInterrupt`). The started node stays unfinished in the log,
 * which ends with the stop's `run_stopped` record.
 * A call in flight is read from the server's private connection (Python: `server._calls`).
 * Concurrent runs are concurrent promises (Python: threads). An AGENT's model is a
 * `ScriptedModel` from `models`, and every run sees only the test's credential store in its
 * environment.
 *
 * Later (resumeFlow): the resume half of `test_a_stop_ends_a_run_owned_server_call_and_the_log_resumes`.
 */

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type RunFlowOptions, type RunResult, runFlow } from "../../src/run.ts";
import { readRunLog } from "../../src/runtime/log.ts";
import { type McpServer, mcpServer } from "../../src/tools/mcp.ts";
import { expectClosed, hostEnv, killMarkers, stdioOptions, strayProcesses, waitFor } from "./mcp_helpers.ts";

// Bounds that keep a hang from stalling the suite; a clean stop takes well under a second.
const SLOW = 30_000;
const STOP_WITHIN = 3;
const markers: string[] = [];
let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-mcp-stop-"));
});

afterEach(() => {
  killMarkers(markers.splice(0));
  rmSync(dir, { recursive: true, force: true });
});

afterAll(() => {
  expect(strayProcesses()).toEqual([]);
});

const SLOW_FLOW = `
id: mcp_stop
name: mcp_stop
input:
  seconds: !float
nodes:
  call:
    kind: tool
    tool_id: slow
    input: {seconds: "\${input.seconds}"}
output: \${call.output}
`;

const AGENT_FLOW = `
id: mcp_stop_agent
name: mcp_stop_agent
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [fake]
output: \${worker.output}
`;

/** `runFlow` in `hostEnv(dir)`, so no host variable or login decides a row. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  return runFlow(flow, inputs, { env: hostEnv(dir), ...options });
}

/** An unopened fake server over stdio, its pid appended to a fresh marker file. */
function fake(): [McpServer, string] {
  const file = join(dir, `marker-${markers.length + 1}`);
  markers.push(file);
  return [mcpServer(stdioOptions("--marker", file)), file];
}

/** How many tool calls `server` has in flight, read from its private connection. */
function callsInFlight(server: McpServer): number {
  return (server as unknown as { generation?: { calls: Set<unknown> } }).generation?.calls.size ?? 0;
}

/** How `promise` settled, and when (in `performance.now()` ms). */
async function settled(promise: Promise<RunResult>): Promise<{ result?: RunResult; error?: unknown; ended: number }> {
  try {
    const result = await promise;
    return { result, ended: performance.now() };
  } catch (error) {
    return { error, ended: performance.now() };
  }
}

/** Abort `stop` once `server`'s call has waited 0.5 s; return when it was aborted. */
async function stopAfterCall(server: McpServer, stop: AbortController): Promise<number> {
  expect(await waitFor(() => callsInFlight(server) > 0, 20_000)).toBe(true);
  await new Promise((resolve) => setTimeout(resolve, 500));
  stop.abort(new Error("stopped"));
  return performance.now();
}

for (const numWorkers of [0, 8]) {
  test(
    `test_a_stop_ends_a_run_owned_server_call_and_the_log_resumes (${numWorkers} workers)`,
    async () => {
      const log = join(dir, "run.jsonl");
      const [server, file] = fake();
      const stop = new AbortController();
      const outcome = settled(
        run(
          await loadFlow(SLOW_FLOW),
          { seconds: 30 },
          { log, numWorkers, tools: { fake: server }, signal: stop.signal },
        ),
      );
      const stoppedAt = await stopAfterCall(server, stop);
      const { error, ended } = await outcome;

      // A host stop is control flow, not a run result: the run rejects with it and its started
      // node stays unfinished in the log.
      expect(error).toBe(stop.signal.reason);
      expect((ended - stoppedAt) / 1000).toBeLessThan(STOP_WITHIN);
      // START succeeded; `call` started and never finished.
      expect(readRunLog(log).map((envelope) => envelope.kind)).toEqual([
        "run_started",
        "node_started",
        "node_succeeded",
        "node_started",
        "run_stopped",
      ]);
      await expectClosed(server, file);
    },
    SLOW,
  );

  test(
    `test_ctrl_c_in_the_host_ends_a_run_owned_server_call (${numWorkers} workers)`,
    async () => {
      const [server, file] = fake();
      const stop = new AbortController();
      const outcome = settled(
        run(await loadFlow(SLOW_FLOW), { seconds: 30 }, { numWorkers, tools: { fake: server }, signal: stop.signal }),
      );
      expect(await waitFor(() => callsInFlight(server) > 0, 20_000)).toBe(true);
      await new Promise((resolve) => setTimeout(resolve, 500));
      // What a host's SIGINT handler does: abort the run's signal.
      const interrupt = Object.assign(new Error("interrupted"), { name: "KeyboardInterrupt" });
      stop.abort(interrupt);
      const sentAt = performance.now();
      const { error, ended } = await outcome;

      expect(error).toBe(interrupt);
      expect((ended - sentAt) / 1000).toBeLessThan(STOP_WITHIN);
      await expectClosed(server, file);
    },
    SLOW,
  );

  test(
    `test_a_stop_cancels_only_its_own_run_on_a_shared_host_server (${numWorkers} workers)`,
    async () => {
      const flow = await loadFlow(SLOW_FLOW);
      const stopA = new AbortController();
      const stopB = new AbortController();
      const [unopened] = fake();
      {
        await using server = await unopened.open();
        const outcomeA = settled(
          run(flow, { seconds: 30 }, { numWorkers, tools: { fake: server }, signal: stopA.signal }),
        );
        expect(await waitFor(() => callsInFlight(server) > 0, 20_000)).toBe(true);
        const outcomeB = settled(
          run(flow, { seconds: 1.5 }, { numWorkers, tools: { fake: server }, signal: stopB.signal }),
        );
        expect(await waitFor(() => callsInFlight(server) >= 2, 20_000)).toBe(true);
        const stoppedAt = await stopAfterCall(server, stopA);
        const a = await outcomeA;
        const b = await outcomeB;

        expect(a.error).toBe(stopA.signal.reason);
        expect((a.ended - stoppedAt) / 1000).toBeLessThan(STOP_WITHIN);
        expect(b.error).toBeUndefined();
        expect(b.result!.failure).toBeNull();
        expect(b.result!.output).toBe("done");
        expect(stopB.signal.aborted).toBe(false);
        expect(server.isOpen).toBe(true);
        const third = await run(flow, { seconds: 0 }, { numWorkers, tools: { fake: server } });
        expect(third.status).toBe("succeeded");
      }
      await expectClosed(unopened);
    },
    SLOW,
  );

  test(
    `test_a_stop_ends_an_agent_tool_call (${numWorkers} workers)`,
    async () => {
      const call = new ToolCallPart({ id: "c0", name: "slow", arguments: { seconds: 30 } });
      const model = new ScriptedModel([
        new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" }),
        new Response({
          message: new Message({ role: "assistant", parts: [new TextPart("done")] }),
          finishReason: "stop",
        }),
      ]);
      const [server, file] = fake();
      const stop = new AbortController();
      const outcome = settled(
        run(
          await loadFlow(AGENT_FLOW),
          {},
          {
            numWorkers,
            tools: { fake: server },
            signal: stop.signal,
            models: { env: hostEnv(dir), createModel: () => model },
          },
        ),
      );
      const stoppedAt = await stopAfterCall(server, stop);
      const { error, ended } = await outcome;

      expect(error).toBe(stop.signal.reason);
      expect((ended - stoppedAt) / 1000).toBeLessThan(STOP_WITHIN);
      await expectClosed(server, file);
    },
    SLOW,
  );
}
