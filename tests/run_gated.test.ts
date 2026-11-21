/**
 * A run a host gates: `runFlow` and `advance` with a `beforeToolCall` hook record it, every node's
 * tool set carries the hook, and an unfinished gated run is refused to a resume without one.
 */

import { afterAll, expect, spyOn, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ComposedFlow } from "../src/compose/build/pipeline.ts";
import type { NodeCaps } from "../src/nodes/base.ts";
import { ToolNode } from "../src/nodes/tool/node.ts";
import { advance, inspectRun, pendingCalls, runFlow } from "../src/run.ts";
import { readRunLog, RunResumedRecord, RunStartedRecord } from "../src/runtime/log.ts";
import { pending } from "../src/tools/context.ts";
import { FunctionTool } from "../src/tools/function_tools.ts";
import type { BeforeToolCall } from "../src/tools/gate.ts";
import { YianRangeError } from "../src/typesys/errors.ts";
import { gated } from "./helpers/gate.ts";
import { completeByHand } from "./helpers/inbox_fixtures.ts";
import { crashAfter, graph, NO_ENV, rejected } from "./helpers/run_state_fixtures.ts";

const ROOT = mkdtempSync(join(tmpdir(), "run-gated-"));
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));
let logs = 0;

/** A new log path in the test folder. */
function freshLog(): string {
  logs += 1;
  return join(ROOT, `run-${logs}.jsonl`);
}

/** A hook that runs every call; this file's flows make no agent call, so it is never asked. */
const HOOK: BeforeToolCall = () => undefined;

/** The host tool `lookup`, whose handler returns `pending()`. */
const LOOKUP = new FunctionTool({
  name: "lookup",
  description: "looks something up later",
  parametersJsonSchema: { type: "object", properties: {} },
  execute: () => pending(),
});

/** START -> TOOL node `first` -> TOOL node `second` -> END, each calling `lookup`. */
function twoCalls(): ComposedFlow {
  const nodes = [new ToolNode("first", { toolId: "lookup" }), new ToolNode("second", { toolId: "lookup" })];
  return new ComposedFlow(graph(nodes, [["__start__", "first"], ["first", "second"], ["second", "__end__"]]));
}

/** A run of `twoCalls()` waiting on `first`, started with `hook` when given. */
async function waitingRun(hook: BeforeToolCall | null = null) {
  const log = freshLog();
  const flow = twoCalls();
  const tools = { lookup: LOOKUP };
  const result = await runFlow(flow, {}, gated({ log, env: NO_ENV, tools, numWorkers: 0 }, hook));
  expect(result.status).toBe("waiting");
  return { log, flow, tools, runId: result.runId };
}

/** The `run_resumed` records of the log at `log`. */
const resumes = (log: string) =>
  readRunLog(log).flatMap((envelope) => (envelope.record instanceof RunResumedRecord ? [envelope.record] : []));

/** The message refusing a resume of the gated run at `log` without a hook. */
const refusal = (log: string) =>
  `run log ${log} is gated: a host checked its tool calls with beforeToolCall; ` +
  "resume it with one (advance({ log, beforeToolCall }))";

test("a run started with a hook records gated: true; one without leaves it out", async () => {
  const withHook = await waitingRun(HOOK);
  const without = await waitingRun();
  const start = (log: string) => readRunLog(log)[0]!;
  expect([(start(withHook.log).record as RunStartedRecord).gated, start(withHook.log).payload["gated"]]).toEqual([true, true]);
  expect(Object.hasOwn(start(without.log).payload, "gated")).toBe(false);
});

test("an unfinished gated run is refused to an advance without a hook, with the log unchanged", async () => {
  const { log, flow, tools, runId } = await waitingRun(HOOK);
  completeByHand(log, flow, `${runId}/first`, { output: "one" });
  const before = readFileSync(log);
  const error = await rejected(advance({ log, flow, tools, env: NO_ENV }));
  expect(error).toEqual(new YianRangeError(refusal(log)));
  expect(readFileSync(log).equals(before)).toBe(true);
  // A crashed gated run is refused the same way.
  const crashed = freshLog();
  await crashAfter(8, () => runFlow(flow, {}, gated({ log: crashed, env: NO_ENV, tools, numWorkers: 0 }, HOOK)));
  expect(await rejected(advance({ log: crashed, flow, tools, env: NO_ENV }))).toEqual(new YianRangeError(refusal(crashed)));
});

test("a gated run's every resume records gated, and its log reads without a hook", async () => {
  const { log, flow, tools, runId } = await waitingRun(HOOK);
  expect(pendingCalls(log).map((call) => call.callId)).toEqual([`${runId}/first`]);
  completeByHand(log, flow, `${runId}/first`, { output: "one" });
  // Another worker count forces a resume.
  expect((await advance(gated({ log, flow, tools, env: NO_ENV, numWorkers: 1 }, HOOK))).status).toBe("waiting");
  expect(resumes(log).map((record) => record.gated)).toEqual([true]);
  completeByHand(log, flow, `${runId}/second`, { output: "two" });
  expect((await advance(gated({ log, flow, tools, env: NO_ENV }, HOOK))).status).toBe("succeeded");
  // A finished run returns its result to any advance.
  expect((await advance({ log, flow, tools, env: NO_ENV })).status).toBe("succeeded");
  expect(inspectRun(log, { flow }).status).toBe("succeeded");
});

test("a hooked advance of an ungated waiting run records gated for good, even from a clean wait", async () => {
  const { log, flow, tools, runId } = await waitingRun();
  completeByHand(log, flow, `${runId}/first`, { output: "one" });
  expect(readRunLog(log).at(-2)!.kind).toBe("run_waiting");
  expect((await advance(gated({ log, flow, tools, env: NO_ENV }, HOOK))).status).toBe("waiting");
  expect(resumes(log).map((record) => [record.retryNodeIds, record.gated])).toEqual([[[], true]]);
  completeByHand(log, flow, `${runId}/second`, { output: "two" });
  expect(await rejected(advance({ log, flow, tools, env: NO_ENV }))).toEqual(new YianRangeError(refusal(log)));
});

test("a hooked advance that can't move appends nothing, and the run stays ungated", async () => {
  const { log, flow, tools, runId } = await waitingRun();
  const before = readFileSync(log);
  expect((await advance(gated({ log, flow, tools, env: NO_ENV }, HOOK))).status).toBe("waiting");
  expect(readFileSync(log).equals(before)).toBe(true);
  completeByHand(log, flow, `${runId}/first`, { output: "one" });
  expect((await advance({ log, flow, tools, env: NO_ENV })).status).toBe("waiting");
  expect(resumes(log)).toEqual([]);
});

test("runFlow and advance give their nodes a tool set holding the hook", async () => {
  const seen: (BeforeToolCall | null | undefined)[] = [];
  const forward = ToolNode.prototype.forward;
  const spy = spyOn(ToolNode.prototype, "forward").mockImplementation(function (this: ToolNode, inputs, caps: NodeCaps) {
    seen.push(caps.tools?.gate);
    return forward.call(this, inputs, caps);
  });
  try {
    const { log, flow, tools, runId } = await waitingRun(HOOK);
    completeByHand(log, flow, `${runId}/first`, { output: "one" });
    const other: BeforeToolCall = () => undefined;
    expect((await advance(gated({ log, flow, tools, env: NO_ENV }, other))).status).toBe("waiting");
    // `first` runs, then wakes to read its answer; `second` runs.
    expect(seen).toEqual([HOOK, other, other]);
  } finally {
    spy.mockRestore();
  }
});
