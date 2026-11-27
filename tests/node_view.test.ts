/**
 * A host reads one node attempt of a run log (`nodeView`): its kind, status, start and end times,
 * each declared input bound as the node bound it at that start, its output and an AGENT's new
 * messages as its success accepted them (derived ones rebuilt), a failure and a route's case. A
 * memory input reads as the value the run recorded, never from its store. When the recorded flow
 * no longer loads, what the log holds still reads, and the rest is `null` with the reason.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { ComposedFlow } from "../src/compose/build/pipeline.ts";
import { loadFlow } from "../src/compose/loader.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import { FileMemory } from "../src/memory/file.ts";
import {
  advance,
  cancel,
  complete,
  type NodeView,
  nodeView,
  pendingCalls,
  runFlow,
  type RunFlowOptions,
  send,
} from "../src/run.ts";
import { iterRunRecords } from "../src/runtime/log.ts";
import { pending } from "../src/tools/context.ts";
import { FunctionTool } from "../src/tools/function_tools.ts";
import { MemoryRef } from "../src/typesys/values.ts";
import { hostEnv } from "./helpers/mcp_helpers.ts";
import { callReply, crashAfter, editRecordedFlow, serving, textReply } from "./helpers/run_state_fixtures.ts";

const root = mkdtempSync(join(tmpdir(), "yian-node-view-"));
afterAll(() => rmSync(root, { recursive: true, force: true }));

const env = hostEnv(root);

/** A fresh run folder's log path. */
function runLog(): string {
  return join(mkdtempSync(join(root, "run-")), "log.jsonl");
}

/** Run `text` with `inputs` over a fresh log; the other options go to `runFlow`. */
async function run(text: string, inputs: Record<string, unknown> = {}, options: RunFlowOptions = {}) {
  const log = runLog();
  const result = await runFlow(await loadFlow(text), inputs, { env, ...options, log });
  return { log, result };
}

/** The time the log at `log` appended its `n`th (from 1) record of `kind` about `nodeId`. */
function at(log: string, kind: string, nodeId: string | null, n = 1): string {
  const found = [...iterRunRecords(log)].filter((entry) => entry.kind === kind && entry.nodeId === nodeId);
  return found[n - 1]!.at;
}

/** `view`'s inputs or outputs as `[name, type, value]`, each rebuilt. */
function values(view: NodeView, side: "inputs" | "outputs") {
  const list = view[side];
  expect(list).not.toBeNull();
  for (const each of list!) expect(each.missing).toBeNull();
  return list!.map(({ name, type, value }) => [name, type, value]);
}

const TOTALS = `
id: totals
name: totals
input:
  price: !int
  note: !str
nodes:
  total:
    kind: code
    input:
      price: \${input.price}
      tax: !int 2
    output: !int
    code: return inputs.price + inputs.tax
  summary:
    kind: code
    input:
      total: \${total.output}
      note: \${input.note}
    output:
      line: !str
      total: !int
    code: |
      return { line: inputs.note + ": " + inputs.total, total: inputs.total };
output:
  line: \${summary.output.line}
  total: \${total.output}
`;

test("a CODE node: its kind, status and times, its inputs as bound and its output, ints as bigints", async () => {
  const { log, result } = await run(TOTALS, { price: 10, note: "tea" });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const view = await nodeView(log, "summary", 1, { env });
  expect(view).toMatchObject({
    nodeId: "summary",
    attempt: 1,
    kind: "code",
    status: "succeeded",
    startedAt: at(log, "node_started", "summary"),
    endedAt: at(log, "node_succeeded", "summary"),
    failure: null,
    route: null,
    missing: null,
  });
  expect(values(view, "inputs")).toEqual([
    ["total", "int", 12n],
    ["note", "str", "tea"],
  ]);
  expect(values(view, "outputs")).toEqual([["output", '{"line":str,"total":int}', { line: "tea: 12", total: 12n }]]);
  expect(values(await nodeView(log, "total", 1, { env }), "inputs")).toEqual([
    ["price", "int", 10n],
    ["tax", "int", 2n],
  ]);
});

test("the run's START gives what the run received, and its END what it gathers and returns", async () => {
  const { log } = await run(TOTALS, { price: 10, note: "tea" });
  const start = await nodeView(log, "__start__", 1, { env });
  expect([start.kind, start.status]).toEqual(["start", "succeeded"]);
  expect(values(start, "inputs")).toEqual([
    ["price", "int", 10n],
    ["note", "str", "tea"],
  ]);
  expect(values(start, "outputs")).toEqual([["output", '{"note":str,"price":int}', { note: "tea", price: 10n }]]);
  const end = await nodeView(log, "__end__", 1, { env });
  expect([end.kind, end.status, end.endedAt]).toEqual(["end", "succeeded", at(log, "node_succeeded", "__end__")]);
  expect(values(end, "inputs")).toEqual([
    ["line", "str", "tea: 12"],
    ["total", "int", 12n],
  ]);
  expect(values(end, "outputs")).toEqual([["output", '{"line":str,"total":int}', { line: "tea: 12", total: 12n }]]);
});

test("a route: the case it took, its route's time, and no output", async () => {
  const { log } = await run(
    `
id: triage
name: triage
input:
  note: !str
nodes:
  gate:
    kind: case
    on: \${input.note}
    cases:
      - when: tea
        then: hot
    else: cold
  hot: !str hot
  cold: !str cold
output: \${gate.output}
`,
    { note: "tea" },
  );
  const view = await nodeView(log, "gate", 1, { env });
  expect(view).toMatchObject({ kind: "case", status: "routed", route: "hot", failure: null, outputs: [] });
  expect(view.endedAt).toBe(at(log, "node_routed", "gate"));
  const hot = await nodeView(log, "hot", 1, { env });
  expect(values(hot, "outputs")).toEqual([["output", "str", "hot"]]);
});

const COUNTDOWN = `
id: countdown
name: countdown
input:
  start: !int
nodes:
  count:
    kind: loop
    input:
      n: \${input.start}
    while: n > 0
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: !int
        code: return inputs.n - 1
    output:
      n: \${step.output}
output: \${count.output.n}
`;

test("a loop: the spawner expands, and each turn's START, body and END read by their grown ids", async () => {
  const { log, result } = await run(COUNTDOWN, { start: 2 });
  expect(result.output).toBe(0n);
  const loop = await nodeView(log, "count", 1, { env });
  expect(loop).toMatchObject({ kind: "loop", status: "expanded", outputs: [] });
  expect(loop.endedAt).toBe(at(log, "node_expanded", "count"));
  expect(values(loop, "inputs")).toEqual([["n", "int", 2n]]);
  // The turn's START and END commit derived values, rebuilt from the commits they read.
  const start = await nodeView(log, "count#1/__start__", 1, { env });
  expect([start.kind, start.status]).toEqual(["start", "succeeded"]);
  expect(values(start, "outputs")).toEqual([["output", '{"n":int}', { n: 1n }]]);
  const step = await nodeView(log, "count#1/step", 1, { env });
  expect(values(step, "inputs")).toEqual([["n", "int", 1n]]);
  expect(values(step, "outputs")).toEqual([["output", "int", 0n]]);
  const end = await nodeView(log, "count#1/__end__", 1, { env });
  expect(values(end, "inputs")).toEqual([["n", "int", 0n]]);
  expect(values(end, "outputs")).toEqual([["output", '{"n":int}', { n: 0n }]]);
  // The last turn's driver writes the loop's output.
  const driver = await nodeView(log, "count~2", 1, { env });
  expect([driver.kind, driver.status]).toEqual(["loop", "succeeded"]);
  expect(values(driver, "outputs")).toEqual([["output", '{"n":int}', { n: 0n }]]);
});

const GREETINGS = `
id: greetings
name: greetings
input:
  lead: !str
  team: !list(str)
defs:
  greet:
    input:
      name: !str
    nodes:
      line:
        kind: code
        input:
          name: \${input.name}
        output: !str
        code: return "hi " + inputs.name
    output: \${line.output}
nodes:
  first:
    kind: call
    call: greet
    input:
      name: \${input.lead}
  all:
    kind: map
    over: \${input.team}
    call: greet
    input:
      name: \${item}
output:
  first: \${first.output}
  all: \${all.output}
`;

test("a call and a map expand; the called flow's START and END, and each item's, read as nodes", async () => {
  const { log, result } = await run(GREETINGS, { lead: "ada", team: ["bo", "cy"] });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const call = await nodeView(log, "first", 1, { env });
  expect([call.kind, call.status, call.outputs]).toEqual(["call", "expanded", []]);
  expect(values(call, "inputs")).toEqual([["name", "str", "ada"]]);
  expect(values(await nodeView(log, "first/__start__", 1, { env }), "outputs")).toEqual([
    ["output", '{"name":str}', { name: "ada" }],
  ]);
  const end = await nodeView(log, "first/__end__", 1, { env });
  expect(values(end, "inputs")).toEqual([["result", "str", "hi ada"]]);
  expect(values(end, "outputs")).toEqual([["output", "str", "hi ada"]]);
  const map = await nodeView(log, "all", 1, { env });
  expect([map.kind, map.status]).toEqual(["map", "expanded"]);
  expect(values(map, "inputs")).toEqual([["over", "list(str)", ["bo", "cy"]]]);
  expect(values(await nodeView(log, "all#1/__start__", 1, { env }), "outputs")).toEqual([
    ["output", '{"name":str}', { name: "cy" }],
  ]);
  expect(values(await nodeView(log, "all#1/line", 1, { env }), "outputs")).toEqual([["output", "str", "hi cy"]]);
});

const ANSWER = `
id: answer
name: answer
llm_config: {provider: anthropic, model: claude-test}
input:
  question: !str
  lessons: !FileMemory
nodes:
  answer:
    kind: agent
    input:
      lessons: \${input.lessons}
      question: \${input.question}
    prompt:
      - user: \${lessons}
      - user: \${question}
    output: !str
output: \${answer.output}
`;

test("an agent: a memory input is the value the run recorded, its store unread; its output and messages", async () => {
  const memoryRoot = mkdtempSync(join(root, "memory-"));
  writeFileSync(join(memoryRoot, "MEMORY.md"), "Prefer short answers.");
  const model = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const { log, result } = await run(
    ANSWER,
    { question: "why?", lessons: new FileMemory({ root: memoryRoot, id: "t/lessons" }) },
    { models: serving(model, env) },
  );
  expect(result.status, result.failure?.message).toBe("succeeded");
  // The view never opens the memory: its folder can be gone.
  rmSync(memoryRoot, { recursive: true, force: true });
  const view = await nodeView(log, "answer", 1, { env });
  expect([view.kind, view.status]).toEqual(["agent", "succeeded"]);
  const [lessons, question] = view.inputs!;
  expect([lessons!.name, lessons!.type, lessons!.missing]).toEqual(["lessons", "FileMemory", null]);
  expect(lessons!.value).toBeInstanceOf(MemoryRef);
  const memory = lessons!.value as MemoryRef;
  expect([memory.identity, memory.type, memory.settings["index"], memory.recall.text]).toEqual([
    "t/lessons",
    "FileMemory",
    "MEMORY.md",
    "Prefer short answers.",
  ]);
  expect([question!.name, question!.type, question!.value]).toEqual(["question", "str", "why?"]);
  expect(values(view, "outputs")).toEqual([
    ["output", "str", "done"],
    ["new_messages", "list(Message)", [{ role: "assistant", parts: [{ type: "text", text: "done" }] }]],
  ]);
});

test("an agent's flow call: its scope's START, body and END read as nodes", async () => {
  const model = new ScriptedModel([callReply(["shout", { text: "hi" }, "c1"]), textReply("done")]);
  const { log, result } = await run(
    `
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
        code: return inputs.text.toUpperCase();
    output: \${loud.output}
nodes:
  helper:
    kind: agent
    prompt: go
    tools: [shout]
output: \${helper.output}
`,
    {},
    { models: serving(model, env) },
  );
  expect(result.status, result.failure?.message).toBe("succeeded");
  const start = await nodeView(log, "helper/shout#0/__start__", 1, { env });
  expect([start.kind, start.status]).toEqual(["start", "succeeded"]);
  expect(values(start, "outputs")).toEqual([["output", '{"text":str}', { text: "hi" }]]);
  expect(values(await nodeView(log, "helper/shout#0/loud", 1, { env }), "inputs")).toEqual([["text", "str", "hi"]]);
  const end = await nodeView(log, "helper/shout#0/__end__", 1, { env });
  expect(values(end, "inputs")).toEqual([["result", "str", "HI"]]);
  expect(values(end, "outputs")).toEqual([["output", "str", "HI"]]);
});

const PLACE = `
id: order
name: order
input:
  qty: !int
nodes:
  place:
    kind: call
    call: place_order
    input:
      qty: \${input.qty}
    retries: 1
    output: !str
output: \${place.output}
`;

test("a tool call's retries are one attempt; without the run's YIAN_* values, what the log holds reads", async () => {
  let calls = 0;
  const place = new FunctionTool({
    name: "place_order",
    description: "Place an order.",
    parametersJsonSchema: { type: "object", properties: { qty: { type: "integer" } }, required: ["qty"] },
    execute: (args) => {
      if (++calls === 1) throw new Error("busy");
      return `placed ${String(args["qty"])}`;
    },
  });
  const retrying = hostEnv(root, { YIAN_TOOL_RETRY_DELAY_SECONDS: "0" });
  const { log, result } = await run(PLACE, { qty: 2 }, { env: retrying, tools: { place_order: place } });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(calls).toBe(2);
  const view = await nodeView(log, "place", 1, { env: retrying });
  expect([view.kind, view.status]).toEqual(["tool", "succeeded"]);
  expect(values(view, "inputs")).toEqual([["qty", "int", 2n]]);
  expect(values(view, "outputs")).toEqual([["output", "str", "placed 2"]]);
  await expect(nodeView(log, "place", 2, { env: retrying })).rejects.toThrow("has no attempt 2 of node 'place'");

  // Without the run's value the flow isn't compiled: the recorded output still reads, untyped.
  const bare = await nodeView(log, "place", 1, { env });
  expect(bare.missing).toContain("YIAN_TOOL_RETRY_DELAY_SECONDS");
  expect([bare.kind, bare.inputs, bare.status, bare.endedAt]).toEqual([
    null,
    null,
    "succeeded",
    at(log, "node_succeeded", "place"),
  ]);
  expect(bare.outputs).toEqual([{ name: "output", type: null, value: "placed 2", missing: null }]);
});

const LISTEN = `
id: listen
name: listen
input:
  m: !Inbox(str)
nodes:
  heard:
    kind: call
    call: take
    input:
      from: \${input.m}
      take: all
output: \${heard.output.events}
`;

test("a take waits until a send wakes it, and its events are rebuilt from the sends it took", async () => {
  const { log, result } = await run(LISTEN);
  expect(result.status).toBe("waiting");
  const waiting = await nodeView(log, "heard", 1, { env });
  expect([waiting.kind, waiting.status, waiting.endedAt, waiting.outputs]).toEqual(["tool", "waiting", null, []]);
  expect(waiting.inputs!.map(({ name }) => name)).toEqual(["from", "take"]);
  await send(log, "m", "hi", { env });
  expect((await advance({ log, env })).status).toBe("succeeded");
  const view = await nodeView(log, "heard", 1, { env });
  expect([view.status, view.endedAt]).toEqual(["succeeded", at(log, "node_succeeded", "heard")]);
  expect(view.outputs.map(({ name, value }) => [name, value])).toEqual([["output", { events: ["hi"], closed: false }]]);
});

test("a node the run's cancel ends has ended, at the cancel's time", async () => {
  const { log } = await run(LISTEN);
  await cancel(log, { env });
  const view = await nodeView(log, "heard", 1, { env });
  const cancelled = at(log, "run_cancelled", null);
  expect([view.status, view.endedAt, view.outputs, view.failure]).toEqual(["ended", cancelled, [], null]);
});

test("a sleep and an ask: each ends with its output", async () => {
  const tools = { ask: () => pending() };
  const { log, result } = await run(`
id: nap
name: nap
nodes:
  nap:
    kind: call
    call: sleep
    input:
      until: "2026-01-01T00:00:00Z"
  gate:
    kind: call
    call: ask
    runs_after: [nap]
    input:
      prompt: approve?
output: \${gate.output}
`, {}, { tools });
  expect(result.status, result.failure?.message).toBe("waiting");
  const nap = await nodeView(log, "nap", 1, { env });
  expect([nap.kind, nap.status, nap.outputs.map(({ value }) => value)]).toEqual(["tool", "succeeded", [null]]);
  expect((await nodeView(log, "gate", 1, { env })).status).toBe("waiting");
  const [call] = pendingCalls(log);
  await complete(log, call!.callId, { output: "yes" }, { env });
  expect((await advance({ log, env, tools })).status).toBe("succeeded");
  const gate = await nodeView(log, "gate", 1, { env });
  expect(gate.status).toBe("succeeded");
  expect(gate.outputs.map(({ name, value }) => [name, value])).toEqual([["output", "yes"]]);
  expect(values(gate, "inputs")).toContainEqual(["prompt", "str", "approve?"]);
});

test("a failed node: its failure and its time, with no output", async () => {
  const { log, result } = await run(`
id: broken
name: broken
input:
  n: !int
nodes:
  boom:
    kind: code
    input:
      n: \${input.n}
    output: !int
    code: throw new Error("boom " + inputs.n)
output: \${boom.output}
`, { n: 3 });
  expect(result.status).toBe("failed");
  const view = await nodeView(log, "boom", 1, { env });
  const failed = at(log, "node_failed", "boom");
  expect([view.status, view.endedAt, view.outputs, view.route]).toEqual(["failed", failed, [], null]);
  expect([view.failure?.code, view.failure?.nodeId]).toEqual(["node.execute", "boom"]);
  expect(view.failure?.message).toContain("boom 3");
  expect(values(view, "inputs")).toEqual([["n", "int", 3n]]);
});

test("a crash in flight: a resume restarts the node, and each attempt reads on its own", async () => {
  const text = `
id: twice
name: twice
input:
  n: !int
nodes:
  first:
    kind: code
    input:
      n: \${input.n}
    output: !int
    code: return inputs.n + 1
  second:
    kind: code
    input:
      n: \${first.output}
    output: !int
    code: return inputs.n * 2
output: \${second.output}
`;
  const flow = await loadFlow(text);
  // Crash right after `second` starts, before it commits.
  const whole = runLog();
  await runFlow(flow, { n: 1 }, { env, log: whole });
  const starts = [...iterRunRecords(whole)].filter(({ kind }) => kind === "node_started");
  const started = starts.find(({ nodeId }) => nodeId === "second")!;
  const log = runLog();
  await crashAfter(started.seq + 1, () => runFlow(flow, { n: 1 }, { env, log }));
  expect((await advance({ log, env })).output).toBe(4n);

  const crashed = await nodeView(log, "second", 1, { env });
  expect([crashed.status, crashed.startedAt, crashed.endedAt, crashed.outputs]).toEqual([
    "restarted",
    at(log, "node_started", "second", 1),
    null,
    [],
  ]);
  expect(values(crashed, "inputs")).toEqual([["n", "int", 2n]]);
  const retried = await nodeView(log, "second", 2, { env });
  const restarted = at(log, "node_started", "second", 2);
  expect([retried.attempt, retried.status, retried.startedAt]).toEqual([2, "succeeded", restarted]);
  expect(values(retried, "inputs")).toEqual([["n", "int", 2n]]);
  expect(values(retried, "outputs")).toEqual([["output", "int", 4n]]);
});

test("with the code root gone, a recorded value reads; inputs and derived values are null with why", async () => {
  const codeRoot = mkdtempSync(join(root, "code-"));
  writeFileSync(join(codeRoot, "double.ts"), "export const double = (inputs: { n: number }) => inputs.n * 2;\n");
  const text = `
id: doubled
name: doubled
input:
  n: !int
defs:
  echo:
    input:
      n: !int
    nodes:
      same:
        kind: code
        input:
          n: \${input.n}
        output: !int
        code: return inputs.n
    output: \${same.output}
nodes:
  twice:
    kind: code
    code: ./double.ts:double
    input:
      n: \${input.n}
    output: !int
  wrap:
    kind: call
    call: echo
    input:
      n: \${twice.output}
output: \${wrap.output}
`;
  const flow = await loadFlow(text, { searchPaths: [codeRoot], sourceLabel: "flow.yaml", codeRoot });
  const log = runLog();
  const result = await runFlow(flow, { n: 3 }, { env, log });
  expect([result.status, result.output]).toEqual(["succeeded", 6n]);
  rmSync(codeRoot, { recursive: true, force: true });

  const view = await nodeView(log, "twice", 1, { env });
  expect(view.missing).toContain("double.ts");
  expect([view.kind, view.inputs, view.status]).toEqual([null, null, "succeeded"]);
  const times = [at(log, "node_started", "twice"), at(log, "node_succeeded", "twice")];
  expect([view.startedAt, view.endedAt]).toEqual(times);
  expect(view.outputs).toEqual([{ name: "output", type: null, value: 6n, missing: null }]);
  // A child START's commit holds no value, only the commits its binding read.
  const start = await nodeView(log, "wrap/__start__", 1, { env });
  expect(start.outputs).toEqual([{ name: "output", type: null, value: null, missing: view.missing }]);
});

test("an attempt the log doesn't hold is refused", async () => {
  const { log } = await run(TOTALS, { price: 1, note: "x" });
  await expect(nodeView(log, "nobody", 1, { env })).rejects.toThrow("has no attempt 1 of node 'nobody'");
  await expect(nodeView(log, "total", 0, { env })).rejects.toThrow("attempt must be a positive integer");
});

test("a node whose binding failed: its failure says why, and its inputs are null with the reason", async () => {
  const { log, result } = await run(
    `
id: typed
name: typed
input:
  data: !Any
nodes:
  read:
    kind: code
    input:
      n: !int \${input.data}
    output: !int
    code: return inputs.n
output: \${read.output}
`,
    { data: "owls" },
  );
  expect(result.failure?.code).toBe("binding.type");
  const view = await nodeView(log, "read", 1, { env });
  expect([view.kind, view.status, view.inputs, view.failure?.code]).toEqual(["code", "failed", null, "binding.type"]);
  expect(view.missing).toContain(result.failure!.message);
});

test("an agent of a run that used the host's key reads without the key, a call-time one too", async () => {
  for (const config of ["", "\n    llm_config: {model: \"${pick.output}\"}"]) {
    const model = new ScriptedModel([textReply("ok")], { provider: "openai", model: "gpt-x" });
    const { log, result } = await run(
      `
id: keyed
name: keyed
llm_config: {provider: openai, model: gpt-x}
input:
  question: !str
nodes:
  pick:
    kind: code
    output: !str
    code: return "gpt-picked"
  chat:
    kind: agent${config}
    input:
      question: \${input.question}
    prompt: \${question}
output: \${chat.output}
`,
      { question: "why?" },
      { models: serving(model, env), llmConfig: { provider: "openai", api_key: "sk-host-key" } },
    );
    expect(result.status, result.failure?.message).toBe("succeeded");
    const view = await nodeView(log, "chat", 1, { env });
    expect(values(view, "inputs")).toEqual([["question", "str", "why?"]]);
    expect(view.outputs.map(({ name, value }) => [name, value])[0]).toEqual(["output", "ok"]);
  }
});

test("an agent whose memory grant failed to bind still shows its data inputs", async () => {
  const memoryRoot = mkdtempSync(join(root, "memory-"));
  const lessons = new FileMemory({ root: memoryRoot, id: "t/twice" });
  const model = new ScriptedModel([textReply("done")], { provider: "anthropic", model: "claude-test" });
  const { log, result } = await run(
    `
id: twice_granted
name: twice_granted
llm_config: {provider: anthropic, model: claude-test}
input:
  question: !str
  a: !FileMemory
  b: !FileMemory
nodes:
  answer:
    kind: agent
    input:
      question: \${input.question}
    tools: [{a: [read]}, {b: [read]}]
    prompt: \${question}
    output: !str
output: \${answer.output}
`,
    { question: "why?", a: lessons, b: lessons },
    { models: serving(model, env) },
  );
  expect(result.failure?.code).toBe("contract.input");
  const view = await nodeView(log, "answer", 1, { env });
  expect([view.status, view.missing]).toEqual(["failed", null]);
  expect(values(view, "inputs")).toEqual([["question", "str", "why?"]]);
});

test("an optional input the run wasn't given is marked so, unlike one given as null", async () => {
  const text = `
id: maybe
name: maybe
input:
  a: !Optional(str)
nodes:
  echo:
    kind: code
    input:
      a: \${input.a}
    output: !Optional(str)
    code: return inputs.a ?? null
output: \${echo.output}
`;
  const left = (await nodeView((await run(text)).log, "__start__", 1, { env })).inputs;
  expect(left).toEqual([{ name: "a", type: "Optional(str)", value: null, missing: "not given" }]);
  const given = (await nodeView((await run(text, { a: null })).log, "__start__", 1, { env })).inputs;
  expect(given).toEqual([{ name: "a", type: "Optional(str)", value: null, missing: null }]);
});

test("a recorded flow that compiles to another fingerprint reads from the records alone, saying so", async () => {
  const { log } = await run(TOTALS, { price: 10, note: "tea" });
  editRecordedFlow(log, (text) => text.replace("inputs.price + inputs.tax", "inputs.price + inputs.tax + 1"));
  const view = await nodeView(log, "total", 1, { env });
  expect(view.missing).toContain("fingerprint");
  expect([view.kind, view.inputs, view.status]).toEqual([null, null, "succeeded"]);
  expect(view.outputs).toEqual([{ name: "output", type: null, value: 12n, missing: null }]);
});

test("a run whose flow was built in code reads with it as flow, and without it says to pass it", async () => {
  const built = new ComposedFlow((await loadFlow(TOTALS)).compiled);
  const log = runLog();
  await runFlow(built, { price: 10, note: "tea" }, { env, log });
  expect(values(await nodeView(log, "total", 1, { env, flow: built }), "outputs")).toEqual([["output", "int", 12n]]);
  await expect(nodeView(log, "total", 1, { env })).rejects.toThrow("pass it to nodeView as flow");
});
