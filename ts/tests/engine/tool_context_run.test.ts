/**
 * `runFlow`/`resumeFlow` pass the host `context` and per-call ids to tools in a run: a port of
 * `tests/engine/test_tool_context_run.py`.
 *
 * An AGENT's model is a `ScriptedModel` given as `models`, where Python patches
 * `model_from_config`; a crash is `crashAfter`'s log refusing the TOOL's outcome (the tool itself
 * returns). Every run sees only the test's credential store in its environment.
 *
 * Ported: `test_tool_node_sees_the_host_context_and_its_call_id`,
 * `test_a_resumed_tool_node_sees_the_same_call_id_and_the_new_context`,
 * `test_resume_without_context_gives_none` (`undefined`, JS's absent value),
 * `test_the_log_never_holds_the_context`, `test_a_registered_tool_sees_the_context`,
 * `test_agent_tool_calls_see_the_context_and_per_call_ids`,
 * `test_agent_tool_from_a_group_sees_the_context`, `test_agent_registered_tool_sees_the_context`,
 * `test_a_function_tool_in_a_tool_node_sees_the_context_and_its_call_id`,
 * `test_a_function_tool_granted_to_an_agent_sees_the_context_and_per_call_ids`,
 * `test_a_direct_agent_outside_a_run_gives_each_call_a_fresh_id` (through `node.run` with a
 * `ToolSet` holding the context) and
 * `test_ask_user_from_an_agent_and_a_human_input_node_sees_the_context` (the request is
 * camelCase, `nodeId`). TS tools are all `FunctionTool`s, so the "function tool" rows build one
 * with `new FunctionTool` where the others use `tool()`.
 *
 * Covered by `run-leaves/tool_context_run__each_loop_iteration_gets_its_own_call_id`:
 * `test_each_loop_iteration_gets_its_own_call_id`; by
 * `run-leaves/tool_context_run__each_map_item_gets_its_own_call_id`:
 * `test_each_map_item_gets_its_own_call_id`.
 */

import { afterAll, afterEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { Message, Response, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { type AskUserRequest, registerTool, TOOL_REGISTRY, ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { crashAfter, serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-tool-context-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// Keep process-global registrations isolated between tests.
const registered = new Set(TOOL_REGISTRY.keys());
afterEach(() => {
  for (const id of TOOL_REGISTRY.keys()) if (!registered.has(id)) TOOL_REGISTRY.delete(id);
});

/** A host app object that tools reach through `ctx.context`. */
class Db {}

/** A host app object whose text forms are one unique marker. */
class Marked {
  static readonly MARKER = "ctx-marker-5b1f0e9c";
  toString(): string {
    return Marked.MARKER;
  }
  toJSON(): string {
    return Marked.MARKER;
  }
}

const TOOL_FLOW = `
id: tool_ctx
name: tool_ctx
nodes:
  place:
    kind: tool
    tool_id: place
output: \${place.output}
`;

const AGENT_FLOW = `
id: agent_ctx
name: agent_ctx
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [GRANT]
output: \${worker.output}
`;

const AGENT_CALL_ID = /^run-7f3a\/worker\/([0-9a-f]{32})\/([01])$/;

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

/** A TOOL-node tool named `name` that records its context and returns its call id. */
const recorder = (seen: ToolContext[], name = "place") =>
  tool({
    name,
    description: "Place one order.",
    parameters: { type: "object", properties: {} },
    execute: (_args, ctx) => {
      seen.push(ctx);
      return ctx.callId;
    },
  });

/** An AGENT tool named `name` that records its context. */
const probe = (seen: ToolContext[], name = "probe") =>
  tool({
    name,
    description: "Look one thing up.",
    parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    execute: (args, ctx) => {
      seen.push(ctx);
      return `seen ${args.text}`;
    },
  });

/** One reply calling `calls` (`[name, args, id]`). */
function callReply(...calls: [string, JsonObject, string][]): Response {
  const parts = calls.map(([name, args, id]) => new ToolCallPart({ id, name, arguments: args }));
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/** Two calls in one reply, one more in a second reply, then a final answer. */
const twoCallsThenAnswer = (name: string) => [
  callReply([name, { text: "a" }, "c0"], [name, { text: "b" }, "c1"]),
  callReply([name, { text: "c" }, "c2"]),
  textReply("done"),
];

/** The run options of an AGENT row: the test's environment and a model answering `replies`. */
const agentHost = (replies: Response[]) => ({
  env: hostEnv(dir),
  models: serving(new ScriptedModel(replies, { provider: "openai", model: "test-model" })),
});

/** The `run_started` payload of the log at `path`. */
function runStartedPayload(path: string): Record<string, unknown> {
  const first = JSON.parse(readFileSync(path, "utf8").split("\n")[0]!);
  expect(first.kind).toBe("run_started");
  return first.payload;
}

test("tool node sees the host context and its call id", async () => {
  const seen: ToolContext[] = [];
  const db = new Db();

  const result = await runFlow(await loadFlow(TOOL_FLOW), {}, {
    tools: { place: recorder(seen) },
    context: db,
    runId: "run-7f3a",
    env: hostEnv(dir),
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(seen).toHaveLength(1);
  const [ctx] = seen;
  expect(ctx!.context).toBe(db);
  expect(ctx!.runId).toBe("run-7f3a");
  expect(ctx!.node).toBe("place");
  expect(ctx!.callId).toBe("run-7f3a/place");
  expect(ctx!.attempt).toBe(1);
  expect(ctx!.toolName).toBe("place");
});

test("a resumed tool node sees the same call id and the new context", async () => {
  const path = freshLog();
  const flow = await loadFlow(TOOL_FLOW);
  const seen: ToolContext[] = [];
  const db = new Db();
  const db2 = new Db();

  // Record 4 is the TOOL's outcome.
  await crashAfter(4, () =>
    runFlow(flow, {}, { log: path, tools: { place: recorder(seen) }, context: db, runId: "run-7f3a", env: hostEnv(dir) }),
  );
  const resumed = await resumeFlow(flow, { log: path, tools: { place: recorder(seen) }, context: db2, env: hostEnv(dir) });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  const [first, second] = seen;
  expect(first!.context).toBe(db);
  expect(second!.context).toBe(db2);
  expect(first!.callId).toBe("run-7f3a/place");
  expect(second!.callId).toBe(first!.callId);
  expect(resumed.output).toBe("run-7f3a/place");
});

test("resume without context gives undefined", async () => {
  const path = freshLog();
  const flow = await loadFlow(TOOL_FLOW);
  const seen: ToolContext[] = [];

  await crashAfter(4, () =>
    runFlow(flow, {}, { log: path, tools: { place: recorder(seen) }, context: new Db(), env: hostEnv(dir) }),
  );
  const resumed = await resumeFlow(flow, { log: path, tools: { place: recorder(seen) }, env: hostEnv(dir) });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(seen[1]!.context).toBeUndefined();
});

test("the log never holds the context", async () => {
  const flow = await loadFlow(TOOL_FLOW);
  const markedLog = freshLog();
  const plainLog = freshLog();
  const seen: ToolContext[] = [];

  const marked = await runFlow(flow, {}, {
    log: markedLog,
    tools: { place: recorder(seen) },
    context: new Marked(),
    env: hostEnv(dir),
  });
  const plain = await runFlow(flow, {}, { log: plainLog, tools: { place: recorder(seen) }, env: hostEnv(dir) });

  expect([marked.status, plain.status]).toEqual(["succeeded", "succeeded"]);
  expect(seen[0]!.context).toBeInstanceOf(Marked);
  expect(readFileSync(markedLog, "utf8")).not.toContain(Marked.MARKER);
  expect(Object.keys(runStartedPayload(markedLog))).toEqual(Object.keys(runStartedPayload(plainLog)));
});

test("a registered tool sees the context", async () => {
  const seen: ToolContext[] = [];
  registerTool("tool_ctx_run_registered_place", recorder(seen, "tool_ctx_run_registered_place"));
  const db = new Db();
  const flow = await loadFlow(TOOL_FLOW.replace("tool_id: place", "tool_id: tool_ctx_run_registered_place"));

  const result = await runFlow(flow, {}, { context: db, runId: "run-7f3a", env: hostEnv(dir) });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(seen).toHaveLength(1);
  expect(seen[0]!.context).toBe(db);
  expect(seen[0]!.callId).toBe("run-7f3a/place");
});

/** Three calls: two in the first reply and one in the second. */
function expectAgentCalls(seen: ToolContext[], db: Db): void {
  expect(seen).toHaveLength(3);
  for (const ctx of seen) {
    expect(ctx.context).toBe(db);
    expect(ctx.node).toBe("worker");
    expect(ctx.attempt).toBe(1);
    expect(ctx.runId).toBe("run-7f3a");
  }
  const matches = seen.map((ctx) => AGENT_CALL_ID.exec(ctx.callId));
  expect(matches.every((match) => match !== null), seen.map((ctx) => ctx.callId).join(", ")).toBe(true);
  const [[, firstExchange, firstIndex], [, secondExchange, secondIndex], [, thirdExchange]] = matches as [
    RegExpExecArray,
    RegExpExecArray,
    RegExpExecArray,
  ];
  expect(firstExchange).toBe(secondExchange!);
  expect([firstIndex, secondIndex]).toEqual(["0", "1"]);
  expect(thirdExchange).not.toBe(firstExchange!);
  expect(new Set(seen.map((ctx) => ctx.callId)).size).toBe(3);
}

test("agent tool calls see the context and per-call ids", async () => {
  const seen: ToolContext[] = [];
  const db = new Db();

  const result = await runFlow(await loadFlow(AGENT_FLOW.replace("GRANT", "probe")), {}, {
    tools: { probe: probe(seen) },
    context: db,
    runId: "run-7f3a",
    ...agentHost(twoCallsThenAnswer("probe")),
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expectAgentCalls(seen, db);
});

test("agent tool from a group sees the context", async () => {
  const seen: ToolContext[] = [];
  const db = new Db();

  const result = await runFlow(await loadFlow(AGENT_FLOW.replace("GRANT", "toolbox")), {}, {
    tools: { toolbox: [probe(seen)] },
    context: db,
    runId: "run-7f3a",
    ...agentHost(twoCallsThenAnswer("probe")),
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expectAgentCalls(seen, db);
});

test("agent registered tool sees the context", async () => {
  const seen: ToolContext[] = [];
  const db = new Db();
  registerTool("tool_ctx_run_registered_probe", probe(seen, "tool_ctx_run_registered_probe"));

  const result = await runFlow(await loadFlow(AGENT_FLOW.replace("GRANT", "tool_ctx_run_registered_probe")), {}, {
    context: db,
    runId: "run-7f3a",
    ...agentHost(twoCallsThenAnswer("tool_ctx_run_registered_probe")),
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expectAgentCalls(seen, db);
});

test("a function tool in a tool node sees the context and its call id", async () => {
  const seen: [ToolContext, unknown][] = [];
  const db = new Db();
  const place = new FunctionTool({
    name: "place",
    description: "Place one order.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: (args, ctx) => {
      seen.push([ctx, args]);
      return ctx.callId;
    },
  });

  const result = await runFlow(await loadFlow(TOOL_FLOW), {}, {
    tools: { place },
    context: db,
    runId: "run-7f3a",
    env: hostEnv(dir),
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(seen).toHaveLength(1);
  const [[ctx, args]] = seen as [[ToolContext, unknown]];
  expect(args).toEqual({});
  expect(ctx.context).toBe(db);
  expect(ctx.node).toBe("place");
  expect(ctx.toolName).toBe("place");
  expect(ctx.callId).toBe("run-7f3a/place");
  expect(result.output).toBe("run-7f3a/place");
});

test("a function tool granted to an agent sees the context and per-call ids", async () => {
  const seen: ToolContext[] = [];
  const texts: unknown[] = [];
  const db = new Db();
  const probeTool = new FunctionTool({
    name: "probe",
    description: "Look one thing up.",
    parametersJsonSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    execute: (args, ctx) => {
      seen.push(ctx);
      texts.push(args.text);
      return `seen ${args.text}`;
    },
  });

  const result = await runFlow(await loadFlow(AGENT_FLOW.replace("GRANT", "probe")), {}, {
    tools: { probe: probeTool },
    context: db,
    runId: "run-7f3a",
    ...agentHost(twoCallsThenAnswer("probe")),
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(texts).toEqual(["a", "b", "c"]);
  expect(seen.every((ctx) => ctx.toolName === "probe")).toBe(true);
  expectAgentCalls(seen, db);
});

test("a direct agent outside a run gives each call a fresh id", async () => {
  const seen: ToolContext[] = [];
  const db = new Db();
  const model = new ScriptedModel(twoCallsThenAnswer("probe"), { provider: "openai", model: "test-model" });
  const node = new AgentNode("worker", { prompt: "go", tools: ["probe"], llmConfig: { provider: "openai", model: "test-model" } });

  const output = await node.run({}, { tools: new ToolSet({ probe: probe(seen) }, { context: db }), models: serving(model) });

  expect(output).toEqual(new Output("done"));
  expect(seen.map((ctx) => ctx.context)).toEqual([db, db, db]);
  expect(seen.every((ctx) => /^[0-9a-f]{32}$/.test(ctx.callId)), seen.map((ctx) => ctx.callId).join(", ")).toBe(true);
  expect(new Set(seen.map((ctx) => ctx.callId)).size).toBe(3);
  expect(new Set(seen.map((ctx) => ctx.runId))).toEqual(new Set([""]));
});

const ASK_FLOW = `
id: ask_ctx
name: ask_ctx
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [ask_user]
  confirm:
    kind: human_input
    input: {draft: "\${worker.output}"}
    prompt: "Ship \${draft}?"
output: \${confirm.output}
`;

test("ask_user from an agent and a human input node sees the context", async () => {
  const seen: [ToolContext, AskUserRequest][] = [];
  const db = new Db();
  const ask = (request: AskUserRequest, ctx: ToolContext) => {
    seen.push([ctx, request]);
    return `yes to ${request.question}`;
  };

  const result = await runFlow(await loadFlow(ASK_FLOW), {}, {
    tools: { ask_user: ask },
    context: db,
    runId: "run-7f3a",
    ...agentHost([callReply(["ask_user", { question: "Which?" }, "c0"]), textReply("draft")]),
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("yes to Ship draft?");
  const [[agentCtx, agentRequest], [humanCtx, humanRequest]] = seen as [[ToolContext, AskUserRequest], [ToolContext, AskUserRequest]];
  expect(agentRequest.question).toBe("Which?");
  expect(agentCtx.context).toBe(db);
  expect(agentCtx.node).toBe("worker");
  expect(agentCtx.toolName).toBe("ask_user");
  expect(agentCtx.callId).toMatch(AGENT_CALL_ID);
  expect(humanRequest.nodeId).toBe("confirm");
  expect(humanCtx.context).toBe(db);
  expect(humanCtx.node).toBe("confirm");
  expect(humanCtx.toolName).toBe("ask_user");
  expect(humanCtx.callId).toBe("run-7f3a/confirm");
});
