/**
 * An AGENT's call of a flow it grants as a tool: the call records a `pending` step holding its
 * arguments and its scope, the run grows the flow's scope under the agent and answers the call
 * from its END, cut to `YIAN_MAX_FLOW_RESULT_CHARS`; hosts never see the call, but its live
 * events name it and its scope as its records do.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { Interrupted, NodeExpanded, NodeFailed, NodeWaiting, type RunEvent } from "../../src/events.ts";
import { ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { flowResultText } from "../../src/nodes/agent/flow_tools.ts";
import { advance, complete, interrupt, pendingCalls, runFlow } from "../../src/run.ts";
import {
  decodeRecord,
  InterruptedRecord,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeStepRecord,
  readRunLog,
} from "../../src/runtime/log.ts";
import { DeliveryRefused } from "../../src/runtime/state.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { callReply, crashAfter, serving, textReply } from "../helpers/run_state_fixtures.ts";

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
        output: !str
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

test("an int input takes the model's whole number, and the pending step records it as an int", async () => {
  const log = freshLog();
  const flow = await loadFlow(`
id: doubling
name: doubling
llm_config: {provider: openai, model: test-model}
defs:
  double:
    description: Double a number.
    input:
      qty: !int
    nodes:
      d: !int \${input.qty * 2}
    output: \${d.output}
nodes:
  helper:
    kind: agent
    prompt: Double 2.
    tools: [double]
output: \${helper.output}
`);
  const model = new ScriptedModel([callReply(["double", { qty: 2n }, "c1"]), textReply("done")]);
  const result = await runFlow(flow, {}, { log, models: serving(model), env: hostEnv(dir) });
  expect(result.status).toBe("succeeded");
  expect(sentResults(model, 1)).toEqual([["c1", "4", false]]);
  expect(helperSteps(log)[1]).toMatchObject({ kind: "pending", arguments: { qty: 2n } });
});

test("each input type takes the model's JSON as a flow's own inputs do: floats, Optional, records, lists, big ints", async () => {
  const log = freshLog();
  const flow = await loadFlow(`
id: echoing
name: echoing
llm_config: {provider: openai, model: test-model}
typedefs:
  Pair: {a: int, b: float}
defs:
  echo:
    description: Echo the inputs.
    input:
      f: !float
      whole: !float
      some: !Optional(int)
      none: !Optional(int)
      pair: !Pair
      nums: !list(int)
      big: !int
    nodes:
      all: !object
        f: \${input.f}
        whole: \${input.whole}
        some: \${input.some}
        none: \${input.none}
        pair: \${input.pair}
        nums: \${input.nums}
        big: \${input.big}
    output: \${all.output}
nodes:
  helper:
    kind: agent
    prompt: Echo.
    tools: [echo]
output: \${helper.output}
`);
  const args = { f: 2.5, whole: 2n, some: 3n, none: null, pair: { a: 1n, b: 4n }, nums: [1n, 2n], big: 2n ** 70n };
  const model = new ScriptedModel([callReply(["echo", args, "c1"]), textReply("done")]);
  const result = await runFlow(flow, {}, { log, models: serving(model), env: hostEnv(dir) });
  expect(result.status).toBe("succeeded");
  expect(sentResults(model, 1)).toEqual([
    ["c1", '{"big":1180591620717411303424,"f":2.5,"none":null,"nums":[1,2],"pair":{"a":1,"b":4},"some":3,"whole":2}', false],
  ]);
  expect(helperSteps(log)[1]).toMatchObject({ kind: "pending", arguments: args });
});

test("a library's def granted as a tool is a flow call shown as `<alias>_<def>`, its scope under that name", async () => {
  const library = "id: billing\nname: billing\ndefs:\n  quote:\n    description: Price an order.\n    input: {qty: !int}\n" +
    "    kind: code\n    code: return inputs.qty * 3\n    output: !int\n";
  writeFileSync(join(dir, "billing.yaml"), library);
  const text = "id: pricing\nname: pricing\nllm_config: {provider: openai, model: test-model}\nuses:\n  billing: ./billing\n" +
    "nodes:\n  helper: {kind: agent, prompt: go, tools: [{billing: [quote]}]}\noutput: ${helper.output}\n";
  const log = freshLog();
  const model = new ScriptedModel([callReply(["billing_quote", { qty: 2n }, "c1"]), textReply("done")]);
  const result = await runFlow(await loadFlow(text, { searchPaths: [dir] }), {}, { log, models: serving(model), env: hostEnv(dir) });
  expect(result.status).toBe("succeeded");
  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual(["billing_quote"]);
  expect(sentResults(model, 1)).toEqual([["c1", "6", false]]);
  expect(helperSteps(log)[1]).toMatchObject({ kind: "pending", tool: "billing_quote", scope: "helper/billing_quote#0" });
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
        kind: call
        call: sign
        output: !str
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
        output: !str
        code: |
          await new Promise((resolve) => setTimeout(resolve, 20000));
          return "late";
      boom:
        kind: code
        output: !str
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
        output: !str
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

// --- Live events ------------------------------------------------------------------------------ //

/** The records of class `cls` in the log at `log`, in order. */
const recordsOf = <T>(log: string, cls: abstract new (...args: never[]) => T): T[] =>
  readRunLog(log).flatMap(({ record }) => (record instanceof cls ? [record as T] : []));

/** The events of class `cls` in `events`, in order. */
const eventsOf = <T>(events: readonly RunEvent[], cls: abstract new (...args: never[]) => T): T[] =>
  events.flatMap((event) => (event instanceof cls ? [event as T] : []));

/** The flow call and scope of each `NodeFailed` in `events`, with the failing node. */
const failedEvents = (events: readonly RunEvent[]) =>
  eventsOf(events, NodeFailed).map(({ failure, callId, scope }) => [failure.nodeId, callId, scope]);

/** The flow call and scope of each `node_failed` record at `log`, with the failing node. */
const failedRecords = (log: string) =>
  recordsOf(log, NodeFailedRecord).map(({ failure, callId, scope }) => [failure.nodeId, callId, scope]);

test("a flow call's NodeExpanded names the call its node_expanded record names, its seed the call's scope", async () => {
  const log = freshLog();
  const events: RunEvent[] = [];
  const model = new ScriptedModel([callReply(["shout", { text: "hi" }, "c1"]), textReply("done")]);
  const onEvent = (event: RunEvent) => void events.push(event);
  const result = await runFlow(await loadFlow(shoutFlow()), {}, { log, runId: "r1", models: serving(model), env: hostEnv(dir), onEvent });
  expect(result.status).toBe("succeeded");
  const expanded = eventsOf(events, NodeExpanded).map(({ nodeId, callId, grow }) => [nodeId, callId, (grow.seed as { scope: string }).scope]);
  const [, pendingStep] = helperSteps(log);
  expect(expanded).toEqual([["helper", "r1/helper/1/0", "helper/shout#0"]]);
  expect(recordsOf(log, NodeExpandedRecord).map(({ spawnerId, callId }) => [spawnerId, callId])).toEqual([["helper", "r1/helper/1/0"]]);
  expect([pendingStep!["call_id"], pendingStep!["scope"]]).toEqual(["r1/helper/1/0", "helper/shout#0"]);
});

test("a failure inside a flow call: NodeFailed names the call and scope its node_failed record names, live and after a resume", async () => {
  const flow = await loadFlow(shoutFlow('throw new Error("bad text")'));
  const failing = [["helper/shout#0/loud", "r1/helper/1/0", "helper/shout#0"]];
  const log = freshLog();
  const events: RunEvent[] = [];
  const onEvent = (event: RunEvent) => void events.push(event);
  const model = () => new ScriptedModel([callReply(["shout", { text: "x" }, "c1"]), textReply("lost")]);
  const result = await runFlow(flow, {}, { log, runId: "r1", models: serving(model()), env: hostEnv(dir), onEvent });
  expect(result.status).toBe("succeeded");
  expect(failedEvents(events)).toEqual(failing);
  expect(failedRecords(log)).toEqual(failing);
  expect(events.some((event) => event.type === "RunFailed")).toBe(false);

  // Cut after the scope grew, the resume replays the growth silently and fails the same node.
  const cut = freshLog();
  const grownAt = readRunLog(log).findIndex(({ record }) => record instanceof NodeExpandedRecord);
  await crashAfter(grownAt + 1, () => runFlow(flow, {}, { log: cut, runId: "r1", models: serving(model()), env: hostEnv(dir) }));
  const resumedEvents: RunEvent[] = [];
  const resumed = await advance({
    log: cut,
    models: serving(new ScriptedModel([textReply("lost")])),
    env: hostEnv(dir),
    onEvent: (event) => void resumedEvents.push(event),
  });
  expect(resumed.status).toBe("succeeded");
  expect(eventsOf(resumedEvents, NodeExpanded)).toEqual([]);
  expect(failedEvents(resumedEvents)).toEqual(failing);
  expect(failedRecords(cut)).toEqual(failing);
});

test("an interrupt of an agent waiting on its flow call: Interrupted names its record's inbox and nodes, and the call's scope", async () => {
  const flow = await loadFlow(`
id: helping
name: helping
llm_config: {provider: openai, model: test-model}
input:
  notes: !Inbox(str)
defs:
  approve:
    description: Ask for approval.
    nodes:
      sign:
        kind: call
        call: sign
        output: !str
    output: \${sign.output}
nodes:
  helper:
    kind: agent
    prompt: go
    steer_inbox: \${input.notes}
    tools: [approve]
    output: !Optional(str)
output: \${helper.output}
`);
  const sign = tool({ name: "sign", description: "Sign.", parameters: { type: "object", properties: {} }, execute: () => pending() });
  const log = freshLog();
  const events: RunEvent[] = [];
  const env = hostEnv(dir);
  const onEvent = (event: RunEvent) => {
    events.push(event);
    // While the run is live, so the interrupt goes through its engine and publishes its events.
    if (event instanceof NodeWaiting && event.nodeId === "helper/approve#0/sign") void interrupt(log, "notes", { env });
  };
  const model = new ScriptedModel([callReply(["approve", {}, "c1"])]);
  const result = await runFlow(flow, {}, { log, runId: "r1", tools: { sign }, models: serving(model), env, onEvent });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBeNull();
  const interrupted = eventsOf(events, Interrupted).map(({ inbox, nodes, scopes }) => [inbox, nodes, scopes]);
  expect(interrupted).toEqual([["notes", ["helper"], ["helper/approve#0"]]]);
  expect(recordsOf(log, InterruptedRecord).map(({ inbox, nodes }) => [inbox, nodes])).toEqual([["notes", ["helper"]]]);
  const [, pendingStep] = helperSteps(log);
  expect(pendingStep!["scope"]).toBe("helper/approve#0");
  // The scope's host call closes after the event that ended its scope.
  const at = events.findIndex((event) => event instanceof Interrupted);
  expect(events[at + 1]?.toJSON()).toEqual({ call_id: "r1/helper/approve#0/sign", reason: "interrupted" });
});
