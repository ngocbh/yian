/**
 * An AGENT's call of a flow it grants as a tool: the call records a `pending` step holding its
 * arguments and its scope, the run grows the flow's scope under the agent and answers the call
 * from its END, cut to `YIAN_MAX_FLOW_RESULT_CHARS`; hosts never see the call.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { flowResultText } from "../../src/nodes/agent/flow_tools.ts";
import { advance, complete, pendingCalls, runFlow } from "../../src/run.ts";
import { decodeRecord, NodeStepRecord } from "../../src/runtime/log.ts";
import { DeliveryRefused } from "../../src/runtime/state.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { callReply, serving, textReply } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-agent-flow-tools-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

/** A flow whose agent `helper` grants the def `shout`, whose node `loud` returns `body`. */
const shoutFlow = (body = "return inputs.text.toUpperCase();") => `
id: helping
name: helping
llm_config: {provider: openai, model: test-model}
defs:
  shout:
    description: Shout the text back.
    input:
      text: !str
    nodes:
      loud:
        kind: code
        input:
          text: \${input.text}
        output: str
        code: ${body}
    output: \${loud.output}
nodes:
  helper:
    kind: agent
    prompt: go
    tools: [shout]
output: \${helper.output}
`;

/** The tool results the model read in its request `index`: `[call id, content, is error]`. */
const sentResults = (model: ScriptedModel, index: number) =>
  model.requests[index]!.messages
    .flatMap((message) => (message.role === "tool" ? [message.parts[0] as ToolResultPart] : []))
    .map((part) => [part.callId, part.content, part.isError]);

/** The agent `helper`'s steps in the log at `log`, in order. */
const helperSteps = (log: string) =>
  readFileSync(log, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => decodeRecord(line).record)
    .flatMap((record) => (record instanceof NodeStepRecord && record.nodeId === "helper" ? [record.step] : []));

test("a flow call runs without a log, and the model reads the flow's output", async () => {
  const model = new ScriptedModel([callReply(["shout", { text: "hi" }, "c1"]), textReply("done")]);
  const result = await runFlow(await loadFlow(shoutFlow()), {}, { models: serving(model), env: hostEnv(dir) });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("done");
  expect(sentResults(model, 1)).toEqual([["c1", "HI", false]]);
});

test("a flow call's pending step holds its arguments and scope; the agent records the answer", async () => {
  const log = freshLog();
  const model = new ScriptedModel([callReply(["shout", { text: "hi" }, "c1"]), textReply("done")]);
  const result = await runFlow(await loadFlow(shoutFlow()), {}, { log, runId: "r1", models: serving(model), env: hostEnv(dir) });
  expect(result.status).toBe("succeeded");
  const [, pendingStep, answer] = helperSteps(log);
  expect(pendingStep).toEqual({
    arguments: { text: "hi" },
    call_id: "r1/helper/1/0",
    flow: true,
    index: 0n,
    kind: "pending",
    reply: 1n,
    scope: "helper/shout#0",
    tool: "shout",
    tool_call_id: "c1",
  });
  expect(answer).toEqual({ call_id: "c1", is_error: false, kind: "tool_result", output: "HI", started_at: null });
});

test("a long output is cut to YIAN_MAX_FLOW_RESULT_CHARS, and the run log keeps it whole", async () => {
  const log = freshLog();
  const model = new ScriptedModel([callReply(["shout", { text: "abcdefghij" }, "c1"]), textReply("done")]);
  const env = hostEnv(dir, { YIAN_MAX_FLOW_RESULT_CHARS: "4" });
  const result = await runFlow(await loadFlow(shoutFlow()), {}, { log, models: serving(model), env });
  expect(result.status).toBe("succeeded");
  expect(sentResults(model, 1)).toEqual([["c1", "ABCD… (6 more characters); the run log holds the whole output", false]]);
  // The scope's END commit names the commit it read, which holds the whole output.
  expect(readFileSync(log, "utf8")).toContain('"key":"helper/shout#0/loud","kind":"str","node_id":"helper/shout#0/loud","value":"ABCDEFGHIJ"');
});

test("flowResultText gives text as itself and other values as JSON, cut without splitting a character", () => {
  expect(flowResultText("f", "plain text", 100)).toBe("plain text");
  expect(flowResultText("f", { n: 3n, ok: true }, 100)).toBe('{"n":3,"ok":true}');
  expect(flowResultText("f", null, 100)).toBe("null");
  expect(flowResultText("f", "a😀bc", 2)).toBe("a😀… (2 more characters); the run log holds the whole output");
});

test("a host sees no flow call: pendingCalls lists the flow's own host call, and complete refuses the flow call", async () => {
  const log = freshLog();
  const flow = await loadFlow(`
id: helping
name: helping
llm_config: {provider: openai, model: test-model}
defs:
  approve:
    description: Ask for approval.
    nodes:
      sign:
        kind: tool
        tool_id: sign
        output: str
    output: \${sign.output}
nodes:
  helper:
    kind: agent
    prompt: go
    tools: [approve]
output: \${helper.output}
`);
  const sign = tool({ name: "sign", description: "Sign.", parameters: { type: "object", properties: {} }, execute: () => pending() });
  const model = new ScriptedModel([callReply(["approve", {}, "c1"]), textReply("done")]);
  const options = { log, runId: "r1", models: serving(model), env: hostEnv(dir), tools: { sign } };
  const first = await runFlow(flow, {}, options);
  expect(first.status).toBe("waiting");
  expect(first.waits.map((wait) => [wait.node, wait.callId])).toEqual([["helper/approve#0/sign", "r1/helper/approve#0/sign"]]);
  expect(pendingCalls(log).map((call) => call.callId)).toEqual(["r1/helper/approve#0/sign"]);
  const refused = await complete(log, "r1/helper/1/0", { output: "x" }).catch((error) => error);
  expect(refused).toBeInstanceOf(DeliveryRefused);
  expect((refused as DeliveryRefused).reason).toBe("not_host_call");

  await complete(log, "r1/helper/approve#0/sign", { output: "signed" });
  const done = await advance({ log, models: serving(model), env: hostEnv(dir), tools: { sign } });
  expect(done.status).toBe("succeeded");
  expect(sentResults(model, 1)).toEqual([["c1", "signed", false]]);
});

test("arguments too large to record are the call's error result, and the flow never starts", async () => {
  const log = freshLog();
  const model = new ScriptedModel([callReply(["shout", { text: "x".repeat(5000) }, "c1"]), textReply("too big")]);
  const env = hostEnv(dir, { YIAN_MAX_EVENT_BYTES: "2000" });
  const result = await runFlow(await loadFlow(shoutFlow()), {}, { log, runId: "r1", models: serving(model), env });
  expect(result.status).toBe("succeeded");
  const [[callId, content, isError]] = sentResults(model, 1) as [[string, string, boolean]];
  expect([callId, isError]).toEqual(["c1", true]);
  expect(content).toMatch(/^Arguments for shout make a call of \d+ bytes, over the run's limit of 2000 \(YIAN_MAX_EVENT_BYTES\); send less\.$/);
  expect(readFileSync(log, "utf8")).not.toContain("helper/shout#0");
});

test("a failure inside a flow stops its other running nodes, whose late outcomes are dropped", async () => {
  const log = freshLog();
  const flow = await loadFlow(`
id: helping
name: helping
llm_config: {provider: openai, model: test-model}
defs:
  race:
    description: Race two nodes.
    nodes:
      slow:
        kind: code
        output: str
        code: |
          await new Promise((resolve) => setTimeout(resolve, 20000));
          return "late";
      boom:
        kind: code
        output: str
        code: |
          await new Promise((resolve) => setTimeout(resolve, 100));
          throw new Error("boom");
    output:
      slow: \${slow.output}
      boom: \${boom.output}
nodes:
  helper:
    kind: agent
    prompt: go
    tools: [race]
output: \${helper.output}
`);
  const model = new ScriptedModel([callReply(["race", {}, "c1"]), textReply("lost")]);
  const started = Date.now();
  const result = await runFlow(flow, {}, { log, runId: "r1", models: serving(model), env: hostEnv(dir), numWorkers: 4 });
  expect(result.status).toBe("succeeded");
  // Well under the slow node's 20 s: the failure stopped it.
  expect(Date.now() - started).toBeLessThan(15000);
  expect(sentResults(model, 1)).toEqual([["c1", "Flow race failed at node boom (node.execute): Error: boom", true]]);
  const records = readFileSync(log, "utf8").split("\n").filter(Boolean).map((line) => decodeRecord(line));
  const failedAt = records.findIndex(({ kind }) => kind === "node_failed");
  expect(records.slice(failedAt + 1).some(({ record }) => (record as { nodeId?: string }).nodeId === "helper/race#0/slow")).toBe(false);
});

test("a nested flow call's failure is its own call's error result; the outer call still answers", async () => {
  const flow = await loadFlow(`
id: helping
name: helping
llm_config: {provider: openai, model: test-model}
defs:
  inner:
    description: Fail.
    nodes:
      boom:
        kind: code
        output: str
        code: throw new Error("inner boom")
    output: \${boom.output}
  outer:
    description: Ask a sub-agent.
    nodes:
      sub:
        kind: agent
        prompt: try inner
        tools: [inner]
    output: \${sub.output}
nodes:
  helper:
    kind: agent
    prompt: go
    tools: [outer]
output: \${helper.output}
`);
  const model = new ScriptedModel([
    callReply(["outer", {}, "c1"]),
    callReply(["inner", {}, "c2"]),
    textReply("inner failed"),
    textReply("outer said inner failed"),
  ]);
  const result = await runFlow(flow, {}, { models: serving(model), env: hostEnv(dir), numWorkers: 4 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("outer said inner failed");
  expect(sentResults(model, 2)).toEqual([["c2", "Flow inner failed at node boom (node.execute): Error: inner boom", true]]);
  expect(sentResults(model, 3)).toEqual([["c1", "inner failed", false]]);
});

test("in one reply, a failed flow call and a good one each get their own result", async () => {
  const flow = await loadFlow(shoutFlow('if (inputs.text === "bad") throw new Error("bad text"); return inputs.text.toUpperCase();'));
  const model = new ScriptedModel([callReply(["shout", { text: "bad" }, "c1"], ["shout", { text: "ok" }, "c2"]), textReply("done")]);
  const result = await runFlow(flow, {}, { models: serving(model), env: hostEnv(dir), numWorkers: 4 });
  expect(result.status).toBe("succeeded");
  expect(sentResults(model, 1)).toEqual([
    ["c1", "Flow shout failed at node loud (node.execute): Error: bad text", true],
    ["c2", "OK", false],
  ]);
});
