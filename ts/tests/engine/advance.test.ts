/**
 * `advance` and the run boundary's waiting runs: the waiting result, when `advance` appends, the
 * lock released while a run waits, and the in-process registry of waiting runs.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { runFlow } from "../../src/run.ts";
import { isRunLogLocked } from "../../src/runtime/log.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { graph, NO_ENV, rejected } from "./run_state_fixtures.ts";

const ROOT = mkdtempSync(join(tmpdir(), "advance-"));
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));
let logs = 0;

/** A new log path in the test folder. */
function freshLog(): string {
  logs += 1;
  return join(ROOT, `run-${logs}.jsonl`);
}

/** A host tool `lookup` whose handler returns `pending()`. */
function pendingTool(): FunctionTool {
  return new FunctionTool({
    name: "lookup",
    description: "looks something up later",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => pending(),
  });
}

/** START -> a TOOL node `ask` calling `lookup` -> END. */
function hostFlow(): ComposedFlow {
  return new ComposedFlow(graph([new ToolNode("ask", { toolId: "lookup" })], [["__start__", "ask"], ["ask", "__end__"]]));
}

test("runFlow ends waiting: status, waits, no output, the lock released", async () => {
  const log = freshLog();
  const result = await runFlow(hostFlow(), {}, { log, env: NO_ENV, tools: { lookup: pendingTool() }, numWorkers: 0 });
  expect(result.status).toBe("waiting");
  expect(result.waits).toEqual([{ node: "ask", callId: `${result.runId}/ask`, tool: "lookup", inbox: null, deadline: null, settled: false }]);
  expect([result.output, result.unread.size, isRunLogLocked(log)]).toEqual([null, 0, false]);
});

test("backward needs a run that succeeded or failed", async () => {
  const result = await runFlow(hostFlow(), {}, { log: freshLog(), env: NO_ENV, tools: { lookup: pendingTool() }, learnable: true });
  const error = await rejected(result.backward("more"));
  expect(error).toEqual(new RangeError("backward needs a run that succeeded or failed; this one is waiting"));
});

import { readFileSync } from "node:fs";

import { Parameter } from "../../src/nodes/contracts.ts";
import { advance, replayRunLog } from "../../src/run.ts";
import { JsonlRunLog, NodeWokenRecord, readRunLog, RunResumedRecord, RunStoppedRecord } from "../../src/runtime/log.ts";
import { commitPrepared, prepareRunCancelled } from "../../src/runtime/state.ts";
import { completeByHand, gateTool, handClock } from "./inbox_fixtures.ts";
import { crashAfter } from "./run_state_fixtures.ts";

const T0 = Date.UTC(2026, 9, 24, 9, 0, 0);

/** The record kinds of the log at `log`, in order. */
const kinds = (log: string) => readRunLog(log).map((envelope) => envelope.kind);

/** A waiting run of `hostFlow()` at a fresh log, with what advancing it needs. */
async function waitingRun(options: { numWorkers?: number } = {}) {
  const log = freshLog();
  const flow = hostFlow();
  const tools = { lookup: pendingTool() };
  const result = await runFlow(flow, {}, { log, env: NO_ENV, tools, numWorkers: options.numWorkers ?? 0 });
  expect(result.status).toBe("waiting");
  return { log, flow, tools, callId: `${result.runId}/ask` };
}

test("advance on a waiting run that is neither ready nor due appends nothing", async () => {
  const { log, flow, tools } = await waitingRun();
  const before = readFileSync(log);
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect([result.status, result.events]).toEqual(["waiting", []]);
  expect(readFileSync(log).equals(before)).toBe(true);
  expect(isRunLogLocked(log)).toBe(false);
});

test("advance on a ready waiting run wakes the node, with no run_resumed", async () => {
  const { log, flow, tools, callId } = await waitingRun();
  completeByHand(log, flow, callId, { output: "yes" });
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect(result.status).toBe("succeeded");
  const answer = readRunLog(log).find((envelope) => envelope.kind === "pool_commit" && envelope.payload["node_id"] === "ask");
  expect(answer?.payload["value"]).toBe("yes");
  expect(kinds(log)).not.toContain("run_resumed");
  expect(kinds(log)).toContain("node_woken");
});

test("advance wakes a due sleep with timed_out", async () => {
  const log = freshLog();
  const nap = new ToolNode("nap", { toolId: "sleep" });
  nap.params = [new Parameter("for")];
  const flow = new ComposedFlow(graph([nap], [["__start__", "nap"], ["nap", "__end__"]], { wiring: { nap: { for: 60n } } }));
  const first = await runFlow(flow, {}, { log, env: NO_ENV, numWorkers: 0, scheduler: handClock(T0) });
  expect(first.status).toBe("waiting");
  const result = await advance({ log, flow, env: NO_ENV, scheduler: handClock(T0 + 61_000) });
  expect(result.status).toBe("succeeded");
  const woken = readRunLog(log).find((envelope) => envelope.record instanceof NodeWokenRecord)!.record as NodeWokenRecord;
  expect(woken.timedOut).toEqual([`${first.runId}/nap`]);
  expect(kinds(log)).not.toContain("run_resumed");
});

test("advance after a crash between node_waiting and run_waiting resumes, then waits", async () => {
  const probe = await waitingRun();
  const k = kinds(probe.log).indexOf("run_waiting");
  const log = freshLog();
  const flow = hostFlow();
  const tools = { lookup: pendingTool() };
  await crashAfter(k, () => runFlow(flow, {}, { log, env: NO_ENV, tools, numWorkers: 0 }));
  expect(kinds(log).at(-1)).toBe("node_waiting");
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect(result.status).toBe("waiting");
  expect(kinds(log).slice(-2)).toEqual(["run_resumed", "run_waiting"]);
  const resumed = readRunLog(log).at(-2)!.record as RunResumedRecord;
  expect(resumed.retryNodeIds).toEqual([]);
});

test("advance with another worker count writes run_resumed", async () => {
  const { log, flow, tools } = await waitingRun();
  const result = await advance({ log, flow, tools, env: NO_ENV, numWorkers: 2 });
  expect(result.status).toBe("waiting");
  expect(kinds(log).slice(-2)).toEqual(["run_resumed", "run_waiting"]);
});

test("advance after a stop while a node waits retries only the running node", async () => {
  const log = freshLog();
  const flow = new ComposedFlow(
    graph(
      [new ToolNode("ask", { toolId: "lookup" }), new ToolNode("gate", { toolId: "gate" })],
      [["__start__", "ask"], ["__start__", "gate"], ["ask", "__end__"], ["gate", "__end__"]],
    ),
  );
  const gate = gateTool();
  const tools = { lookup: pendingTool(), gate: gate.tool };
  const host = new AbortController();
  const stop = new Error("host stop");
  void gate.entered.then(() => host.abort(stop));
  expect(await rejected(runFlow(flow, {}, { log, env: NO_ENV, tools, numWorkers: 0, signal: host.signal }))).toBe(stop);
  const stopped = readRunLog(log).find((envelope) => envelope.record instanceof RunStoppedRecord)!.record as RunStoppedRecord;
  expect(stopped.nodeIds).toEqual(["gate"]);
  gate.release();
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect(result.status).toBe("waiting");
  const resumed = readRunLog(log).find((envelope) => envelope.record instanceof RunResumedRecord)!.record as RunResumedRecord;
  expect(resumed.retryNodeIds).toEqual(["gate"]);
});

test("advance on a cancelled log returns cancelled", async () => {
  const { log, flow, tools } = await waitingRun();
  const authority = JsonlRunLog.resume(log);
  try {
    const state = replayRunLog(flow, authority);
    commitPrepared(authority, state, prepareRunCancelled(state));
  } finally {
    authority.close();
  }
  const result = await advance({ log, flow, tools, env: NO_ENV });
  expect(result.status).toBe("cancelled");
});

test("advance on a missing log rejects with ENOENT", async () => {
  const error = await rejected(advance({ log: join(ROOT, "missing.jsonl"), env: NO_ENV }));
  expect((error as NodeJS.ErrnoException).code).toBe("ENOENT");
});
