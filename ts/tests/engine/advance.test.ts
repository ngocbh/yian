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
