/**
 * TS-only: an AGENT's tool call may wait. A handler that returns `pending()` registers the call;
 * the agent runs the reply's other calls and waits; `complete` answers it; the woken agent
 * replays and calls the model with every result.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { advance, complete, pendingCalls, runFlow } from "../../src/run.ts";
import { decodeRecord, NodeStepRecord } from "../../src/runtime/log.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import type { ToolEntry } from "../../src/tools/registry.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { callReply, serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-agent-pending-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

const CONFIG = { provider: "openai", model: "test-model" };

/** A flow whose one AGENT node `agent` grants `granted`. */
const agentFlow = (granted: string) => `
id: agent_pending
name: agent_pending
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [${granted}]
output: \${agent.output}
`;

/** A host tool `lookup` answering `answer()`. */
const lookup = (answer: () => unknown) =>
  tool({ name: "lookup", description: "Look up.", parameters: { type: "object", properties: {} }, execute: answer });

/** The kinds of the log's records, a step as its own `kind`. */
const kinds = (log: string) =>
  readFileSync(log, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => decodeRecord(line))
    .map(({ kind, record }) => (record instanceof NodeStepRecord ? String(record.step["kind"]) : kind));

/** The tool results the model read in its request `index`: `[call id, content, is error]`. */
const sentResults = (model: ScriptedModel, index: number) =>
  model.requests[index]!.messages.flatMap((message) =>
    message.role === "tool" ? [message.parts[0] as ToolResultPart] : [],
  ).map((part) => [part.callId, part.content, part.isError]);

/** Run the agent flow granting `granted` with `tools`, its model `model`, under run id `r`. */
async function start(model: ScriptedModel, tools: Record<string, ToolEntry>, granted = "ask_user, lookup") {
  const log = freshLog();
  const options = { env: hostEnv(dir), log, tools, models: serving(model), runId: "r" };
  return { log, options, result: await runFlow(await loadFlow(agentFlow(granted)), {}, options) };
}

test("one call pends, its sibling's result is written, the agent waits; complete then advance answers", async () => {
  let asked = 0;
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"], ["lookup", {}, "c2"]), textReply("done")], CONFIG);
  const tools = { ask_user: () => (asked++, pending()), lookup: lookup(() => "found") };
  const { log, options, result } = await start(model, tools);

  expect(result.status, result.failure?.message).toBe("waiting");
  expect(result.waits.map((w) => [w.node, w.callId, w.tool, w.settled])).toEqual([["agent", "r/agent/1/0", "ask_user", false]]);
  expect(kinds(log).slice(-5)).toEqual(["model_call", "pending", "tool_result", "node_waiting", "run_waiting"]);
  expect(pendingCalls(log).map((c) => [c.callId, c.tool, c.arguments])).toEqual([["r/agent/1/0", "ask_user", { question: "Ship?" }]]);

  expect(await complete(log, "r/agent/1/0", { output: "yes" })).toBeNumber();
  const resumed = await advance(options);

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("done");
  expect(asked).toBe(1); // the handler is never called again
  expect(sentResults(model, 1)).toEqual([
    ["c1", "yes", false],
    ["c2", "found", false],
  ]);
  expect(pendingCalls(log)).toEqual([]);
});
