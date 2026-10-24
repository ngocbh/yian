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
import { type RunEvent, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import { ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { AgentLoopError, type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/memory.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import { advance, cancel, complete, pendingCalls, runFlow, send } from "../../src/run.ts";
import { decodeRecord, NodeStepRecord, NodeWaitingRecord } from "../../src/runtime/log.ts";
import { pending } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { type ToolEntry, ToolSet } from "../../src/tools/registry.ts";
import { gateTool, handClock } from "./inbox_fixtures.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { callReply, crashAfter, NO_ENV, serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-agent-pending-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

const CONFIG = { provider: "openai", model: "test-model" };

/** A flow whose one AGENT node `agent` grants `granted`; `steerInbox` gives it the steer inbox `notes`. */
const agentFlow = (granted: string, steerInbox = false) => `
id: agent_pending
name: agent_pending
llm_config: {provider: openai, model: test-model}
${steerInbox ? "input:\n  notes: !Inbox(str)\n" : ""}nodes:
  agent:
    kind: agent
    prompt: go
    tools: [${granted}]
${steerInbox ? "    steer_inbox: \${input.notes}\n" : ""}output: \${agent.output}
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
async function start(
  model: ScriptedModel,
  tools: Record<string, ToolEntry>,
  granted = "ask_user, lookup",
  extra: { onEvent?: (event: RunEvent) => void; steerInbox?: boolean } = {},
) {
  const log = freshLog();
  const { steerInbox = false, ...hooks } = extra;
  const options = { env: hostEnv(dir), log, tools, models: serving(model), runId: "r", ...hooks };
  const flow = await loadFlow(agentFlow(granted, steerInbox));
  return { log, options, result: await runFlow(flow, {}, options) };
}

/** Start the agent flow granting `ask_user, gate` without awaiting it; `ask_user` pends. */
async function startGated(model: ScriptedModel) {
  const gate = gateTool();
  const log = freshLog();
  const options = { env: hostEnv(dir), log, tools: { ask_user: () => pending(), gate: gate.tool }, models: serving(model), runId: "r" };
  const running = runFlow(await loadFlow(agentFlow("ask_user, gate")), {}, options);
  return { gate, log, running };
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

test("an agent with a steer inbox waits with it; a steer sent while it waits joins the call after its results", async () => {
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"]), textReply("done")], CONFIG);
  const log = freshLog();
  const options = { env: hostEnv(dir), log, tools: { ask_user: () => pending() }, models: serving(model), runId: "r" };
  const flow = await loadFlow(agentFlow("ask_user", true));
  const result = await runFlow(flow, {}, options);

  expect(result.status, result.failure?.message).toBe("waiting");
  const records = readFileSync(log, "utf8").split("\n").filter(Boolean).map((line) => decodeRecord(line).record);
  const waiting = records.find((record) => record instanceof NodeWaitingRecord) as NodeWaitingRecord;
  expect(waiting.steerInbox).toBe("notes");

  await send(log, "notes", "only the 2024 rows");
  await complete(log, "r/agent/1/0", { output: "yes" });
  const resumed = await advance(options);

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("done");
  const last = model.requests[1]!.messages.slice(-2).map((message) => [message.role, message.text()]);
  expect(last).toEqual([
    ["tool", ""],
    ["user", "only the 2024 rows"],
  ]);
  expect(sentResults(model, 1)).toEqual([["c1", "yes", false]]);
  expect(kinds(log).filter((kind) => kind === "steer")).toHaveLength(1);
});

test("a complete landing while a sibling still runs is accepted, and the node wakes at once", async () => {
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"], ["gate", {}, "c2"]), textReply("done")], CONFIG);
  const { gate, log, running } = await startGated(model);
  await gate.entered;
  expect(await complete(log, "r/agent/1/0", { output: "yes" })).toBeNumber();
  gate.release("opened");
  const result = await running;

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("done");
  // The answer still goes through node_waiting, so there is one path; the node is woken at once.
  const all = kinds(log);
  expect(all.slice(all.indexOf("pending"), all.indexOf("node_woken") + 1)).toEqual([
    "pending",
    "tool_result",
    "tool_result",
    "node_waiting",
    "node_woken",
  ]);
  expect(all).not.toContain("run_waiting");
  expect(sentResults(model, 1)).toEqual([
    ["c1", "yes", false],
    ["c2", "opened", false],
  ]);
});

test("two calls pend: node_waiting names both, in call order; answers go back in call order", async () => {
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"], ["lookup", {}, "c2"]), textReply("done")], CONFIG);
  const { log, options, result } = await start(model, { ask_user: () => pending(), lookup: lookup(() => pending()) });

  expect(result.status, result.failure?.message).toBe("waiting");
  expect(result.waits.map((w) => w.callId)).toEqual(["r/agent/1/0", "r/agent/1/1"]);
  expect(pendingCalls(log).map((c) => [c.callId, c.tool, c.arguments])).toEqual([
    ["r/agent/1/0", "ask_user", { question: "Ship?" }],
    ["r/agent/1/1", "lookup", {}],
  ]);
  await complete(log, "r/agent/1/1", { output: "found" });
  // One answer of two: the agent still waits.
  expect((await advance(options)).status).toBe("waiting");
  await complete(log, "r/agent/1/0", { output: "yes" });
  const resumed = await advance(options);

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(sentResults(model, 1)).toEqual([
    ["c1", "yes", false],
    ["c2", "found", false],
  ]);
});

test("a completed error reads as an error result", async () => {
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"]), textReply("done")], CONFIG);
  const { log, options } = await start(model, { ask_user: () => pending() }, "ask_user");
  await complete(log, "r/agent/1/0", { error: "denied" });
  const resumed = await advance(options);

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(sentResults(model, 1)).toEqual([["c1", "denied", true]]);
});

test("cancel names a waiting agent's call with its reply's arguments", async () => {
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"])], CONFIG);
  const { log } = await start(model, { ask_user: () => pending() }, "ask_user");
  expect((await cancel(log)).map((c) => [c.callId, c.tool, c.arguments])).toEqual([["r/agent/1/0", "ask_user", { question: "Ship?" }]]);
  expect(pendingCalls(log)).toEqual([]);
});

test("an ask_user answered inline works as before", async () => {
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"]), textReply("got yes")], CONFIG);
  const { log, result } = await start(model, { ask_user: () => "yes" }, "ask_user");

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("got yes");
  expect(kinds(log)).not.toContain("pending");
  expect(sentResults(model, 1)).toEqual([["c1", "yes", false]]);
});

test("an agent with a steer inbox whose tool returns pending() waits", async () => {
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"])], CONFIG);
  const { log, result } = await start(model, { ask_user: () => pending() }, "ask_user", { steerInbox: true });

  expect(result.status, result.failure?.message).toBe("waiting");
  expect(pendingCalls(log).map((call) => call.callId)).toEqual(["r/agent/1/0"]);
});

test("pending() outside a durable run fails the agent", async () => {
  const model = new ScriptedModel([callReply(["lookup", {}, "c1"])], CONFIG);
  const ctx: AgentRunContext = {
    nodeId: "n",
    messages: [userMessage("go")],
    promptText: "go",
    tools: ["lookup"],
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    memoryContext: new MemoryContext(),
    toolset: new ToolSet({ lookup: lookup(() => pending()) }),
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: { env: NO_ENV, createModel: () => model },
    sent: () => [],
  };
  const error = await runAgent(ctx).then(
    () => null,
    (thrown: unknown) => thrown,
  );
  expect(error).toBeInstanceOf(AgentLoopError);
  expect((error as Error).message).toBe("agent node 'n': tool 'lookup' returned pending() outside a durable run; give the run a log");
});

test("a pended call is reported started and never finished", async () => {
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"], ["lookup", {}, "c2"])], CONFIG);
  const events: RunEvent[] = [];
  await start(model, { ask_user: () => pending(), lookup: lookup(() => "found") }, "ask_user, lookup", {
    onEvent: (event) => void events.push(event),
  });
  const brackets = events.flatMap((event) =>
    event instanceof ToolCallStarted || event instanceof ToolCallFinished ? [[event.type, event.toolCallId]] : [],
  );
  expect(brackets).toEqual([
    ["ToolCallStarted", "c1"],
    ["ToolCallStarted", "c2"],
    ["ToolCallFinished", "c2"],
  ]);
});

const MEMORY_FLOW = `
id: agent_pending_memory
name: agent_pending_memory
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    long_term_memory: notes
    prompt: go
    tools: [ask_user]
output: \${agent.output}
`;

/** The system message of `model`'s request `index`. */
const systemText = (model: ScriptedModel, index: number) => model.requests[index]!.messages[0]!.text();

test("memory that learned while the agent waited can't fail its wake; the new call recalls fresh", async () => {
  const notes = new FakeMemory({ recallText: "Note A." });
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"]), textReply("done")], CONFIG);
  const log = freshLog();
  const options = {
    env: hostEnv(dir),
    log,
    tools: { ask_user: () => pending() },
    models: serving(model),
    memory: new Map([["notes", notes]]),
    runId: "r",
  };
  expect((await runFlow(await loadFlow(MEMORY_FLOW), {}, options)).status).toBe("waiting");
  notes.recallText = "Note B.";
  await complete(log, "r/agent/1/0", { output: "yes" });
  const resumed = await advance(options);

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(systemText(model, 0)).toContain("Note A.");
  expect(systemText(model, 1)).toContain("Note B.");
  expect(systemText(model, 1)).not.toContain("Note A.");
});

test("a crash between a pending step and node_waiting resumes: the call is closed and the agent goes on", async () => {
  const notes = new FakeMemory({ recallText: "Note A." });
  const replies = () => [callReply(["ask_user", { question: "Ship?" }, "c1"]), textReply("done")];
  const options = (model: ScriptedModel, log: string) => ({
    env: hostEnv(dir),
    log,
    tools: { ask_user: () => pending() },
    models: serving(model),
    memory: new Map([["notes", notes]]),
    runId: "r",
  });
  const uncut = freshLog();
  await runFlow(await loadFlow(MEMORY_FLOW), {}, options(new ScriptedModel(replies(), CONFIG), uncut));
  const log = freshLog();
  // Keep every record up to and including the pending step.
  await crashAfter(kinds(uncut).indexOf("pending") + 1, async () =>
    runFlow(await loadFlow(MEMORY_FLOW), {}, options(new ScriptedModel(replies(), CONFIG), log)),
  );
  expect(kinds(log).at(-1)).toBe("pending");
  notes.recallText = "Note B.";
  const model = new ScriptedModel([textReply("done")], CONFIG);
  const resumed = await advance(options(model, log));

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(kinds(log)).toContain("run_resumed");
  // The recorded reply replayed; only the live call reached the model, with the fresh recall.
  expect(model.requests).toHaveLength(1);
  expect(systemText(model, 0)).toContain("Note B.");
  expect(sentResults(model, 0)).toEqual([["c1", TOOL_OUTCOME_UNKNOWN, true]]);
});

/** The log's records as `[kind, node]` (a step as its own kind; `null` for a record of no node). */
const nodeRecords = (log: string) =>
  readFileSync(log, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => decodeRecord(line))
    .map(({ kind, record }) => {
      const nodeId = (record as { nodeId?: unknown }).nodeId;
      return [record instanceof NodeStepRecord ? String(record.step["kind"]) : kind, typeof nodeId === "string" ? nodeId : null];
    });

/** The index of the first `[kind, node]` record of `log`, or -1. */
const indexOf = (records: (string | null)[][], kind: string, node: string) =>
  records.findIndex(([k, n]) => k === kind && n === node);

const POOLED = `
id: pooled
name: pooled
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [ask_user]
  slow:
    kind: tool
    tool_id: gate
output: \${agent.output}
`;

test("with one worker, a woken agent waits for the running node's slot", async () => {
  const gate = gateTool();
  const model = new ScriptedModel([callReply(["ask_user", { question: "q" }, "c1"]), textReply("done")], CONFIG);
  const log = freshLog();
  const running = runFlow(await loadFlow(POOLED), {}, {
    env: hostEnv(dir),
    log,
    numWorkers: 1,
    tools: { ask_user: () => pending(), gate: gate.tool },
    models: serving(model),
    runId: "r",
  });
  await gate.entered; // the agent pended and waits; slow holds the one slot
  await complete(log, "r/agent/1/0", { output: "yes" });
  await new Promise((resolve) => setTimeout(resolve, 5));
  expect(kinds(log)).not.toContain("node_woken");
  gate.release();
  const result = await running;

  expect(result.status, result.failure?.message).toBe("succeeded");
  const records = nodeRecords(log);
  expect(indexOf(records, "node_waiting", "agent")).toBeLessThan(indexOf(records, "node_started", "slow"));
  expect(indexOf(records, "node_succeeded", "slow")).toBeLessThan(indexOf(records, "node_woken", "agent"));
});

test("a due TOOL ask behind an agent waiting for the one slot doesn't spin the dispatcher", async () => {
  const START = Date.UTC(2026, 9, 23);
  const clock = handClock(START);
  const gate = gateTool();
  const flow = `
id: pooled_ask
name: pooled_ask
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [ask_user]
  lead:
    kind: tool
    tool_id: ask
    input:
      prompt: "Ship?"
      timeout: "1m"
    output: Optional(str)
  slow:
    kind: tool
    tool_id: gate
output: \${agent.output}
`;
  const model = new ScriptedModel([callReply(["ask_user", { question: "q" }, "c1"]), textReply("done")], CONFIG);
  const log = freshLog();
  const running = runFlow(await loadFlow(flow), {}, {
    env: hostEnv(dir),
    log,
    numWorkers: 1,
    scheduler: clock,
    tools: { ask: () => pending(), gate: gate.tool },
    models: serving(model),
    runId: "r",
  });
  await gate.entered; // agent and lead wait; slow holds the one slot
  await complete(log, "r/agent/1/0", { output: "yes" });
  clock.set(START + 120_000);
  // A macrotask: a dispatcher spinning on microtasks would starve it, and the run would never end.
  setTimeout(() => gate.release(), 20);
  const result = await running;

  expect(result.status, result.failure?.message).toBe("succeeded");
  const records = nodeRecords(log);
  expect(indexOf(records, "node_succeeded", "slow")).toBeLessThan(indexOf(records, "node_woken", "agent"));
  expect(indexOf(records, "node_woken", "lead")).toBeGreaterThan(indexOf(records, "node_woken", "agent"));
}, 5000);

test("a run with no log refuses an agent's pending(): nothing could answer it", async () => {
  const model = new ScriptedModel([callReply(["ask_user", { question: "Ship?" }, "c1"])], CONFIG);
  const result = await runFlow(await loadFlow(agentFlow("ask_user")), {}, {
    env: hostEnv(dir),
    log: null,
    tools: { ask_user: () => pending() },
    models: serving(model),
  });

  expect(result.status).toBe("failed");
  expect(result.failure?.nodeId).toBe("agent");
  expect(result.failure?.message).toBe(
    "agent node 'agent': tool 'ask_user' returned pending() outside a durable run; give the run a log",
  );
});
